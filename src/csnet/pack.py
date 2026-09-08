"""Packed source store: 100k tiny WAVs -> one flat int16 file per split.

Why this exists
---------------
Libri2Mix ships the *isolated* sources (``s1/``, ``s2/``), already 8 kHz and loudness
normalised, with the LibriSpeech utterance id in the filename. Pooling them lets us mix
any number of speakers on the fly instead of rendering an 8 GB corpus per N.

But reading 27,800 small WAVs per epoch is hopeless on Kaggle's input filesystem. So we
pack every utterance once into a single flat ``audio.i16`` plus an ``index.csv``, and read
it back through a memmap. One sequential file, no per-item open().

Layout::

    <store_root>/manifest.json
    <store_root>/<split>/audio.i16     flat int16, C order, no header
    <store_root>/<split>/index.csv     utt_id,speaker_id,offset,length,role
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from .audio import audio_frames, best_energy_window, float_to_int16, int16_to_float, read_wav
from .constants import CAP_LEN, MIN_UTT_SECONDS, SR
from .utils import get_logger, json_dump_atomic, json_load, sizeof_fmt

LOG = get_logger(__name__)

INDEX_FIELDS = ("utt_id", "speaker_id", "offset", "length", "role")
AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".m4a")
_SRC_DIR_RE = re.compile(r"^s\d+$")


# --------------------------------------------------------------------------- scanning

def find_source_dirs(split_dir: str) -> list[str]:
    """Return the ``s1``, ``s2``, ... subdirectory names present, in numeric order."""
    if not os.path.isdir(split_dir):
        return []
    names = [d for d in os.listdir(split_dir)
             if _SRC_DIR_RE.match(d) and os.path.isdir(os.path.join(split_dir, d))]
    return sorted(names, key=lambda d: int(d[1:]))


def scan_libri2mix(libri2mix_min_dir: str, split: str) -> dict[str, tuple[str, int]]:
    """Map ``utt_id -> (path, frames)`` for every isolated source in a split.

    Filenames are ``<utt1>_<utt2>[..._<uttK>].wav`` and the file living in ``sK`` holds
    utterance ``K``. The same utterance can appear in several pair files truncated to
    different lengths (``min`` mode), so we keep the longest copy of each.
    """
    split_dir = os.path.join(libri2mix_min_dir, split)
    source_dirs = find_source_dirs(split_dir)
    if not source_dirs:
        raise FileNotFoundError(
            f"no s1/s2/... source directories under {split_dir!r}. "
            f"Point --libri2mix_dir at the folder that contains <split>/s1.")

    best: dict[str, tuple[str, int]] = {}
    for k, sub in enumerate(source_dirs):
        directory = os.path.join(split_dir, sub)
        for name in sorted(os.listdir(directory)):
            if not name.lower().endswith(AUDIO_EXT):
                continue
            parts = os.path.splitext(name)[0].split("_")
            if k >= len(parts):
                continue
            utt_id = parts[k]
            path = os.path.join(directory, name)
            try:
                frames = audio_frames(path)
            except Exception:  # unreadable file -- skip rather than kill a 40-minute job
                LOG.warning("unreadable, skipping: %s", path)
                continue
            current = best.get(utt_id)
            if current is None or frames > current[1]:
                best[utt_id] = (path, frames)
    return best


def speaker_of(utt_id: str) -> str:
    """LibriSpeech utterance ids are ``<speaker>-<chapter>-<index>``."""
    return utt_id.split("-")[0]


def assign_roles(speakers: Iterable[str], babble_frac: float, seed: int) -> dict[str, str]:
    """Reserve a deterministic fraction of speakers for babble noise only.

    Babble built from speakers we also ask the model to separate would be an invisible
    leak, so those speakers never appear as targets.
    """
    ordered = sorted(set(speakers))
    rng = np.random.default_rng(seed)
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    n_babble = int(round(babble_frac * len(shuffled)))
    babble = set(shuffled[:n_babble])
    return {spk: ("babble" if spk in babble else "target") for spk in ordered}


# --------------------------------------------------------------------------- packing

def _write_index(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(INDEX_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def pack_split(libri2mix_min_dir: str, split: str, out_root: str, *,
               cap_len: int = CAP_LEN, babble_frac: float = 0.2, seed: int = 72,
               limit: int | None = None, progress: bool = True,
               min_seconds: float = MIN_UTT_SECONDS) -> dict:
    """Pack one split's isolated sources into ``<out_root>/<split>/``."""
    out_dir = os.path.join(out_root, split)
    os.makedirs(out_dir, exist_ok=True)
    audio_path = os.path.join(out_dir, "audio.i16")
    index_path = os.path.join(out_dir, "index.csv")

    found = scan_libri2mix(libri2mix_min_dir, split)
    utt_ids = sorted(found)
    if limit is not None:
        utt_ids = utt_ids[:limit]
    roles = assign_roles((speaker_of(u) for u in utt_ids), babble_frac, seed)

    min_len = int(min_seconds * SR)
    rows: list[dict] = []
    offset = 0
    iterator: Iterable[str] = utt_ids
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(utt_ids, desc=f"pack {split}", unit="utt")
        except ImportError:
            pass

    with open(audio_path, "wb") as sink:
        for utt_id in iterator:
            path, _ = found[utt_id]
            try:
                wave, _ = read_wav(path, sr=SR)
            except Exception:
                LOG.warning("unreadable, skipping: %s", path)
                continue
            if wave.size < min_len:
                continue
            if wave.size > cap_len:
                start = best_energy_window(wave, cap_len)
                wave = wave[start:start + cap_len]
            packed = float_to_int16(wave)
            sink.write(packed.tobytes())
            rows.append({"utt_id": utt_id, "speaker_id": speaker_of(utt_id),
                         "offset": offset, "length": int(packed.size),
                         "role": roles[speaker_of(utt_id)]})
            offset += int(packed.size)

    _write_index(index_path, rows)
    speakers = {r["speaker_id"] for r in rows}
    babble_speakers = {r["speaker_id"] for r in rows if r["role"] == "babble"}
    info = {
        "n_utt": len(rows),
        "n_speakers": len(speakers),
        "n_babble_speakers": len(babble_speakers),
        "n_target_utt": sum(1 for r in rows if r["role"] == "target"),
        "n_babble_utt": sum(1 for r in rows if r["role"] == "babble"),
        "total_samples": offset,
        "audio_bytes": offset * 2,
        "hours": offset / SR / 3600.0,
    }
    LOG.info("packed %s: %d utterances / %d speakers / %s (%.2f h)",
             split, info["n_utt"], info["n_speakers"], sizeof_fmt(info["audio_bytes"]),
             info["hours"])
    return info


def pack_noise_dir(noise_dir: str, out_root: str, *, cap_len: int = CAP_LEN,
                   seed: int = 72, limit: int | None = None,
                   progress: bool = True) -> dict:
    """Pack an arbitrary folder tree of noise recordings into ``<out_root>/noise/``.

    The immediate parent directory name becomes the ``speaker_id`` field, so ESC-50 /
    UrbanSound8K style category folders survive into the index.
    """
    out_dir = os.path.join(out_root, "noise")
    os.makedirs(out_dir, exist_ok=True)
    audio_path = os.path.join(out_dir, "audio.i16")
    index_path = os.path.join(out_dir, "index.csv")

    files: list[str] = []
    for root, _dirs, names in os.walk(noise_dir):
        for name in sorted(names):
            if name.lower().endswith(AUDIO_EXT):
                files.append(os.path.join(root, name))
    files.sort()
    if limit is not None:
        rng = np.random.default_rng(seed)
        if len(files) > limit:
            files = [files[i] for i in sorted(rng.choice(len(files), limit, replace=False))]

    rows: list[dict] = []
    offset = 0
    iterator: Iterable[str] = files
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(files, desc="pack noise", unit="file")
        except ImportError:
            pass

    with open(audio_path, "wb") as sink:
        for i, path in enumerate(iterator):
            try:
                wave, _ = read_wav(path, sr=SR)
            except Exception:
                LOG.warning("unreadable, skipping: %s", path)
                continue
            if wave.size < SR // 2:
                continue
            if wave.size > cap_len:
                start = best_energy_window(wave, cap_len)
                wave = wave[start:start + cap_len]
            packed = float_to_int16(wave)
            sink.write(packed.tobytes())
            rows.append({"utt_id": f"noise{i:06d}",
                         "speaker_id": os.path.basename(os.path.dirname(path)) or "noise",
                         "offset": offset, "length": int(packed.size), "role": "noise"})
            offset += int(packed.size)

    _write_index(index_path, rows)
    info = {"n_utt": len(rows), "total_samples": offset, "audio_bytes": offset * 2,
            "hours": offset / SR / 3600.0, "source": os.path.abspath(noise_dir)}
    LOG.info("packed noise: %d clips / %s", info["n_utt"], sizeof_fmt(info["audio_bytes"]))
    return info


# --------------------------------------------------------------------------- manifest

def manifest_path(store_root: str) -> str:
    """Path of the store manifest."""
    return os.path.join(store_root, "manifest.json")


def read_manifest(store_root: str) -> dict:
    """Load the store manifest, or an empty skeleton when it does not exist yet."""
    path = manifest_path(store_root)
    if os.path.exists(path):
        return json_load(path)
    return {"sr": SR, "cap_len": CAP_LEN, "splits": {}}


def write_manifest(store_root: str, manifest: dict) -> None:
    """Persist the store manifest atomically."""
    manifest.setdefault("created_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    json_dump_atomic(manifest, manifest_path(store_root))


def split_is_packed(store_root: str, split: str) -> bool:
    """True when both artefacts of a split exist and are non-empty."""
    directory = os.path.join(store_root, split)
    audio = os.path.join(directory, "audio.i16")
    index = os.path.join(directory, "index.csv")
    return (os.path.exists(audio) and os.path.exists(index)
            and os.path.getsize(audio) > 0 and os.path.getsize(index) > 0)


# --------------------------------------------------------------------------- reading

class SourceStore:
    """Read-only view of one packed split.

    Pickle-safe: the memmap handle is opened lazily so DataLoader workers (fork *and*
    spawn) each get their own, and the parent never ships an open file descriptor.
    """

    def __init__(self, root: str, split: str, mmap: bool = True) -> None:
        self.root = str(root)
        self.split = str(split)
        self.mmap = bool(mmap)
        self.dir = os.path.join(self.root, self.split)
        self.audio_path = os.path.join(self.dir, "audio.i16")
        self.index_path = os.path.join(self.dir, "index.csv")
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"no packed index at {self.index_path!r}; run 00_pack_sources.py")

        utt_ids: list[str] = []
        speakers: list[str] = []
        offsets: list[int] = []
        lengths: list[int] = []
        roles: list[str] = []
        with open(self.index_path, "r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                utt_ids.append(row["utt_id"])
                speakers.append(row["speaker_id"])
                offsets.append(int(row["offset"]))
                lengths.append(int(row["length"]))
                roles.append(row.get("role", "target"))

        self.utt_ids = utt_ids
        self.speaker_ids = np.array(speakers, dtype="<U24")
        self.offsets = np.array(offsets, dtype=np.int64)
        self.lengths = np.array(lengths, dtype=np.int64)
        self.roles = np.array(roles, dtype="<U8")
        self.target_idx = np.flatnonzero(self.roles == "target").astype(np.int64)
        self.babble_idx = np.flatnonzero(self.roles == "babble").astype(np.int64)
        self.noise_idx = np.flatnonzero(self.roles == "noise").astype(np.int64)

        by_speaker: dict[str, list[int]] = {}
        for i in self.target_idx:
            by_speaker.setdefault(str(self.speaker_ids[i]), []).append(int(i))
        self.by_speaker = {k: np.array(v, dtype=np.int64) for k, v in sorted(by_speaker.items())}
        self.speakers = list(self.by_speaker)

        babble_by_speaker: dict[str, list[int]] = {}
        for i in self.babble_idx:
            babble_by_speaker.setdefault(str(self.speaker_ids[i]), []).append(int(i))
        self.babble_speakers = sorted(babble_by_speaker)

        self._data: np.ndarray | None = None

    # -- lazy backing array ------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        """The int16 sample array, memory-mapped or fully resident."""
        if self._data is None:
            if self.mmap:
                self._data = np.memmap(self.audio_path, dtype=np.int16, mode="r")
            else:
                self._data = np.fromfile(self.audio_path, dtype=np.int16)
        return self._data

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_data"] = None  # never pickle an open memmap into a worker
        return state

    def __len__(self) -> int:
        return len(self.utt_ids)

    # -- access ------------------------------------------------------------
    def get(self, i: int, start: int = 0, n: int | None = None) -> np.ndarray:
        """Return utterance ``i`` as float32, from ``start``, at most ``n`` samples."""
        i = int(i)
        length = int(self.lengths[i])
        start = max(0, min(int(start), length))
        take = length - start if n is None else max(0, min(int(n), length - start))
        base = int(self.offsets[i]) + start
        return int16_to_float(np.asarray(self.data[base:base + take]))

    def duration(self, i: int) -> float:
        """Length of utterance ``i`` in seconds."""
        return float(self.lengths[i]) / SR

    def subset_speakers(self, speakers: Iterable[str]) -> "SourceStore":
        """A view restricted to the given target speakers; babble stays available.

        Used by the K-fold search to build speaker-disjoint validation folds without
        re-reading anything from disk.
        """
        import copy

        keep = {str(s) for s in speakers}
        missing = keep - set(self.by_speaker)
        if missing:
            raise KeyError(f"not target speakers of split {self.split!r}: {sorted(missing)[:5]}")
        view = copy.copy(self)
        view.by_speaker = {k: v for k, v in self.by_speaker.items() if k in keep}
        view.speakers = list(view.by_speaker)
        view.target_idx = (np.concatenate(list(view.by_speaker.values()))
                           if view.by_speaker else np.zeros(0, dtype=np.int64))
        return view

    def summary(self) -> dict:
        """Counts used by the EDA and audit scripts."""
        return {
            "split": self.split,
            "n_utt": len(self),
            "n_target_utt": int(self.target_idx.size),
            "n_babble_utt": int(self.babble_idx.size),
            "n_speakers": len(self.speakers),
            "n_babble_speakers": len(self.babble_speakers),
            "hours": float(self.lengths.sum()) / SR / 3600.0,
            "median_utt_seconds": float(np.median(self.lengths)) / SR,
            "median_utts_per_speaker": float(
                np.median([v.size for v in self.by_speaker.values()])) if self.by_speaker else 0.0,
        }


def open_store(root: str, split: str, mmap: bool = True) -> SourceStore | None:
    """Open a split if it exists, else return None (used for the optional noise store)."""
    try:
        return SourceStore(root, split, mmap=mmap)
    except FileNotFoundError:
        return None
