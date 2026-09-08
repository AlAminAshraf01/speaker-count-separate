"""Mixture recipes: the frozen-test-set artefact and the dynamic-mixing engine.

A *recipe* is one CSV row that names which packed utterances to crop, at what gain, with
which noise at which SNR. :func:`render_recipe` turns it back into audio bit-for-bit, so
the evaluation set is a few hundred kB of CSV rather than gigabytes of WAV -- and it can be
committed to the repo, which is what "freeze the test set" actually requires.

The construction order below is fixed. Do not reorder it: reproducibility depends on it.

1. crop each source, remove its mean, RMS-normalise, apply its dB gain
2. ``speech = sum(sources)``
3. scale the noise to the requested speech-to-noise ratio (or use silence)
4. ``mix = speech + noise``
5. divide *everything* by ``rms(mix)`` -- this is the anti-leak normalisation, and the
   factor is stored so a re-render reproduces it exactly

Invariant, asserted in ``tests/test_mixing.py``::

    max|mix - (sources.sum(0) + noise)| < 1e-5
"""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

from .audio import pad_or_trim, rms
from .constants import EPS, SEG_LEN

if TYPE_CHECKING:  # pragma: no cover
    from .noise import NoiseBank
    from .pack import SourceStore

RECIPE_FIELDS: tuple[str, ...] = (
    "mix_id", "n_src", "utt_idx", "crop_start", "gain_db",
    "noise_kind", "noise_id", "snr_db", "scale",
)
_SEP = "|"


# --------------------------------------------------------------------------- helpers

def _f32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _join_int(values: Iterable[int]) -> str:
    return _SEP.join(str(int(v)) for v in values)


def _join_float(values: Iterable[float]) -> str:
    # repr() round-trips a python float exactly, so a written recipe re-renders bit-for-bit.
    return _SEP.join(repr(float(v)) for v in values)


def _split_int(text: str) -> list[int]:
    text = str(text).strip()
    return [int(v) for v in text.split(_SEP)] if text else []


def _split_float(text: str) -> list[float]:
    text = str(text).strip()
    return [float(v) for v in text.split(_SEP)] if text else []


# --------------------------------------------------------------------------- sampling

def sample_recipe(store: "SourceStore", bank: "NoiseBank", n_src: int,
                  rng: np.random.Generator, *, seg_len: int = SEG_LEN,
                  gain_db_range: tuple[float, float] = (-5.0, 5.0),
                  snr_db_range: tuple[float, float] = (0.0, 20.0),
                  p_clean: float = 0.2, mix_id: str | None = None,
                  min_crop_rms_ratio: float = 0.3, max_tries: int = 8) -> dict:
    """Draw one mixture recipe: distinct speakers, live crops, gains, and a noise choice."""
    n_src = int(n_src)
    speakers = store.speakers
    if len(speakers) < n_src:
        raise ValueError(f"split {store.split!r} has {len(speakers)} target speakers, "
                         f"need {n_src} distinct ones")

    # Distinct SPEAKER ids first, then one utterance each -- a train-100 speaker owns ~111
    # utterances, so sampling utterances directly would repeat a speaker inside a mixture.
    chosen_speakers = rng.choice(len(speakers), size=n_src, replace=False)
    utt_idx: list[int] = []
    crop_start: list[int] = []
    for si in chosen_speakers:
        pool = store.by_speaker[speakers[int(si)]]
        index = int(pool[int(rng.integers(0, pool.size))])
        utt_idx.append(index)
        crop_start.append(_pick_crop(store, index, seg_len, rng,
                                     min_crop_rms_ratio, max_tries))

    gain_db = [float(rng.uniform(*gain_db_range)) for _ in range(n_src)]

    if float(rng.random()) < p_clean:
        noise_kind, noise_id, snr_db = "none", 0, 0.0
    else:
        noise_kind = str(rng.choice(bank.kinds, p=bank.weights))
        noise_id = int(rng.integers(0, 2 ** 31 - 1))
        snr_db = float(rng.uniform(*snr_db_range))

    if mix_id is None:
        mix_id = f"n{n_src}_" + "-".join(store.utt_ids[i] for i in utt_idx)

    recipe = {"mix_id": mix_id, "n_src": n_src, "utt_idx": utt_idx,
              "crop_start": crop_start, "gain_db": gain_db,
              "noise_kind": noise_kind, "noise_id": noise_id,
              "snr_db": snr_db, "scale": 1.0}
    # Fill in the true normalisation factor so a stored recipe re-renders identically.
    recipe["scale"] = float(_assemble(recipe, store, bank, seg_len, use_stored_scale=False)[3])
    return recipe


def _pick_crop(store: "SourceStore", index: int, seg_len: int, rng: np.random.Generator,
               min_crop_rms_ratio: float, max_tries: int) -> int:
    """Choose a crop offset that is not silence; fall back to the loudest attempt."""
    wave = store.get(index)
    max_start = max(0, wave.size - seg_len)
    if max_start == 0:
        return 0
    full = float(rms(wave)) + EPS
    best_start, best_ratio = 0, -1.0
    for _ in range(max(1, int(max_tries))):
        start = int(rng.integers(0, max_start + 1))
        ratio = float(rms(pad_or_trim(wave, seg_len, start))) / full
        if ratio >= min_crop_rms_ratio:
            return start
        if ratio > best_ratio:
            best_ratio, best_start = ratio, start
    return best_start


# --------------------------------------------------------------------------- rendering

def _assemble(recipe: dict, store: "SourceStore", bank: "NoiseBank", seg_len: int,
              use_stored_scale: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Shared body of render_recipe; returns ``(mix, sources, noise, scale)`` unscaled-aware."""
    utt_idx = recipe["utt_idx"]
    crop_start = recipe["crop_start"]
    gain_db = recipe["gain_db"]

    sources = np.zeros((len(utt_idx), seg_len), dtype=np.float32)
    for k, (index, start, gain) in enumerate(zip(utt_idx, crop_start, gain_db)):
        seg = pad_or_trim(store.get(int(index), int(start), seg_len), seg_len)
        seg = (seg - np.float32(seg.mean())).astype(np.float32)
        seg = (seg / np.float32(float(rms(seg)) + EPS)).astype(np.float32)
        sources[k] = (seg * np.float32(10.0 ** (float(gain) / 20.0))).astype(np.float32)

    speech = _f32(sources.sum(axis=0))

    kind = str(recipe["noise_kind"])
    if kind == "none":
        noise = np.zeros(seg_len, dtype=np.float32)
    else:
        noise = bank.render(kind, int(recipe["noise_id"]), seg_len)
        target = (float(rms(speech)) / (float(rms(noise)) + EPS)
                  * 10.0 ** (-float(recipe["snr_db"]) / 20.0))
        noise = (noise * np.float32(target)).astype(np.float32)

    mix = _f32(speech + noise)
    if use_stored_scale:
        scale = float(recipe["scale"])
    else:
        scale = 1.0 / (float(rms(mix)) + EPS)
    factor = np.float32(scale)
    return ((mix * factor).astype(np.float32),
            (sources * factor).astype(np.float32),
            (noise * factor).astype(np.float32),
            scale)


def render_recipe(recipe: dict, store: "SourceStore", bank: "NoiseBank", *,
                  seg_len: int = SEG_LEN) -> dict:
    """Turn a recipe back into audio. ``mix == sources.sum(0) + noise`` to 1e-5."""
    mix, sources, noise, _ = _assemble(recipe, store, bank, int(seg_len), use_stored_scale=True)
    return {"mix": mix, "sources": sources, "noise": noise,
            "n_src": int(recipe["n_src"]), "mix_id": str(recipe["mix_id"]),
            "is_noisy": int(str(recipe["noise_kind"]) != "none"),
            "snr_db": float(recipe["snr_db"]), "noise_kind": str(recipe["noise_kind"])}


# --------------------------------------------------------------------------- csv io

def write_recipes(rows: Sequence[dict], path: str) -> None:
    """Write recipes to CSV atomically."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(RECIPE_FIELDS)
        for r in rows:
            writer.writerow([
                r["mix_id"], int(r["n_src"]),
                _join_int(r["utt_idx"]), _join_int(r["crop_start"]),
                _join_float(r["gain_db"]), r["noise_kind"], int(r["noise_id"]),
                repr(float(r["snr_db"])), repr(float(r["scale"])),
            ])
    os.replace(tmp, path)


def read_recipes(path: str) -> list[dict]:
    """Read recipes back with every field parsed to its proper type."""
    rows: list[dict] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "mix_id": row["mix_id"],
                "n_src": int(row["n_src"]),
                "utt_idx": _split_int(row["utt_idx"]),
                "crop_start": _split_int(row["crop_start"]),
                "gain_db": _split_float(row["gain_db"]),
                "noise_kind": row["noise_kind"],
                "noise_id": int(row["noise_id"]),
                "snr_db": float(row["snr_db"]),
                "scale": float(row["scale"]),
            })
    return rows


def recipe_speakers(recipe: dict, store: "SourceStore") -> list[str]:
    """Speaker ids taking part in a recipe -- used by the leakage audit."""
    return [str(store.speaker_ids[int(i)]) for i in recipe["utt_idx"]]
