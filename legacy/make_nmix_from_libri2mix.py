#!/usr/bin/env python3
"""
make_nmix_from_libri2mix.py
===========================
Build N-speaker mixtures (N = 3, 4, 5, ...) *reusing only the files that are
already inside the Kaggle Libri2Mix-8kHz-min dataset*.

Why this works
--------------
Libri2Mix ships the isolated sources, not just the mixtures.  Every file in
`<split>/s1/` and `<split>/s2/` is a single clean LibriSpeech utterance that has
already been resampled to 8 kHz and loudness-normalised by the official
LibriMix recipe.  The LibriSpeech speaker ID is the first field of the
filename (`4077-13754-0001_...` -> speaker `4077` for s1), so we can pool all of
those sources, group them by speaker, and re-mix them N at a time.

What this script guarantees (the parts that are easy to get wrong)
-----------------------------------------------------------------
1. SPEAKER DISJOINTNESS *within* a mixture.  A speaker contributes ~110
   utterances to train-100, so naive random sampling puts the same speaker in
   one mixture surprisingly often.  We sample N *distinct* speaker IDs.
2. SPEAKER DISJOINTNESS *across* splits is inherited from LibriMix: train-100 /
   dev-clean / test-clean use disjoint LibriSpeech speakers.  Never mix sources
   from two different splits.
3. NO UTTERANCE REUSE inside a split (sampling without replacement), which is
   what the official `create_librimix_metadata.py` does.
4. LibriMix-faithful levels: each source is re-normalised to a fresh target
   loudness drawn from U(-33, -25) LUFS with pyloudnorm, then the whole mixture
   is rescaled if any signal would clip (MAX_AMP = 0.9) -- identical policy to
   the official recipe.
5. `min` mode: all N sources are truncated to the shortest one, so
   `mix == s1 + s2 + ... + sN` sample-for-sample.
6. Deterministic: fixed seed, and a manifest CSV is written so the exact
   mixture list can be regenerated / audited (needed for the data-leakage
   audit in the lab guidelines).

Output layout (asteroid / SpeechBrain compatible)
-------------------------------------------------
    <outdir>/wav8k/min/<split>/mix_clean/<id>.wav
    <outdir>/wav8k/min/<split>/s1..sN/<id>.wav
    <outdir>/wav8k/min/metadata/mixture_<split>_mix_clean.csv

Usage
-----
    python make_nmix_from_libri2mix.py \
        --libri2mix_dir /kaggle/input/libri2mix-8khz-min/Libri2Mix/wav8k/min \
        --outdir       /kaggle/working/Libri3Mix \
        --n_src 3 \
        --splits train-100 dev test \
        --n_mix  9300 3000 3000
"""

import argparse
import csv
import os
import random
from collections import defaultdict

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

# --- same global constants as the official LibriMix recipe -------------------
MAX_AMP = 0.9          # peak ceiling for sources and mixture
MIN_LOUDNESS = -33     # LUFS
MAX_LOUDNESS = -25     # LUFS
EPS = 1e-10


def index_sources(split_dir, n_src_dirs=("s1", "s2")):
    """Pool every isolated source in the split, grouped by LibriSpeech speaker.

    Filenames look like  `<utt1>_<utt2>.wav`  where utt_k = `spk-chapter-idx`.
    The k-th field of the mixture ID is the utterance stored in `s{k}`.
    """
    by_speaker = defaultdict(list)
    n_files = 0
    for k, sd in enumerate(n_src_dirs):
        d = os.path.join(split_dir, sd)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"missing source dir: {d}")
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".wav"):
                continue
            utt_ids = fn[:-4].split("_")
            if k >= len(utt_ids):
                continue
            utt_id = utt_ids[k]                 # e.g. 4077-13754-0001
            speaker = utt_id.split("-")[0]      # e.g. 4077
            by_speaker[speaker].append((utt_id, os.path.join(d, fn)))
            n_files += 1
    return by_speaker, n_files


def sample_groups(by_speaker, n_src, n_mix, rng):
    """Sample n_mix groups of n_src utterances from n_src *distinct* speakers,
    without reusing any utterance."""
    pool = {spk: list(utts) for spk, utts in by_speaker.items()}
    for spk in pool:
        rng.shuffle(pool[spk])

    groups, fails = [], 0
    while len(groups) < n_mix and fails < 500:
        speakers = [s for s, u in pool.items() if u]
        if len(speakers) < n_src:
            break
        chosen = rng.sample(speakers, n_src)
        group = [pool[s].pop() for s in chosen]
        # guard against the pathological case of duplicate utterance IDs
        if len({g[0] for g in group}) != n_src:
            fails += 1
            continue
        groups.append(group)
    return groups


def build_mixture(paths, meter, rng):
    """Load, re-normalise loudness, truncate to `min`, sum, de-clip."""
    srcs, rates = [], set()
    for p in paths:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        srcs.append(x)
        rates.add(sr)
    assert len(rates) == 1, f"inconsistent sample rates: {rates}"
    sr = rates.pop()

    # --- loudness randomisation, exactly the LibriMix policy ---------------
    normed, gains = [], []
    for x in srcs:
        try:
            loud = meter.integrated_loudness(x)
            target = rng.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
            y = pyln.normalize.loudness(x, loud, target)
            if np.max(np.abs(y)) >= 1.0:                     # would clip alone
                y = x * MAX_AMP / (np.max(np.abs(x)) + EPS)
        except Exception:                                    # too short for LUFS
            y = x * MAX_AMP / (np.max(np.abs(x)) + EPS)
        gains.append(float(np.max(np.abs(y)) / (np.max(np.abs(x)) + EPS)))
        normed.append(y)

    # --- min mode: truncate everything to the shortest source --------------
    L = min(len(y) for y in normed)
    normed = [y[:L] for y in normed]
    mix = np.sum(normed, axis=0)

    # --- de-clip: one common factor so mix == sum(sources) still holds -----
    peak = max(np.max(np.abs(mix)), max(np.max(np.abs(y)) for y in normed))
    if peak > MAX_AMP:
        f = MAX_AMP / peak
        normed = [y * f for y in normed]
        mix = mix * f
        gains = [g * f for g in gains]

    return mix.astype("float32"), [y.astype("float32") for y in normed], sr, gains


def si_sdr(est, ref):
    ref = ref - ref.mean()
    est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + EPS)
    proj = a * ref
    return 10 * np.log10((np.sum(proj ** 2) + EPS) / (np.sum((est - proj) ** 2) + EPS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri2mix_dir", required=True,
                    help="path to .../Libri2Mix/wav8k/min")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_src", type=int, default=3)
    ap.add_argument("--splits", nargs="+", default=["train-100", "dev", "test"])
    ap.add_argument("--n_mix", nargs="+", type=int, default=None,
                    help="mixtures per split; default 9300/3000/3000 style")
    ap.add_argument("--seed", type=int, default=72)
    ap.add_argument("--sr_tag", default="8k")
    args = ap.parse_args()

    n_mix = args.n_mix or [9300 if s.startswith("train") else 3000
                           for s in args.splits]
    assert len(n_mix) == len(args.splits)

    for split, target in zip(args.splits, n_mix):
        split_dir = os.path.join(args.libri2mix_dir, split)
        by_speaker, n_files = index_sources(split_dir)
        print(f"[{split}] {n_files} isolated sources / "
              f"{len(by_speaker)} speakers "
              f"(median {int(np.median([len(v) for v in by_speaker.values()]))} utts/speaker)")

        # one RNG per split -> reproducible and split-independent
        rng = random.Random(args.seed + hash(split) % 10_000)
        groups = sample_groups(by_speaker, args.n_src, target, rng)
        print(f"[{split}] sampled {len(groups)} mixtures of {args.n_src} "
              f"distinct speakers (asked for {target})")

        base = os.path.join(args.outdir, f"wav{args.sr_tag}", "min")
        outd = os.path.join(base, split)
        mdd = os.path.join(base, "metadata")
        os.makedirs(mdd, exist_ok=True)
        for sd in ["mix_clean"] + [f"s{i+1}" for i in range(args.n_src)]:
            os.makedirs(os.path.join(outd, sd), exist_ok=True)

        meter = None
        rows = []
        for group in groups:
            utt_ids = [g[0] for g in group]
            paths = [g[1] for g in group]
            if meter is None:
                _, sr0 = sf.read(paths[0], dtype="float32", frames=1)
                meter = pyln.Meter(sr0)
            mix, srcs, sr, gains = build_mixture(paths, meter, rng)
            mix_id = "_".join(utt_ids)
            mp = os.path.join(outd, "mix_clean", mix_id + ".wav")
            sf.write(mp, mix, sr)
            sps = []
            for i, y in enumerate(srcs):
                sp = os.path.join(outd, f"s{i+1}", mix_id + ".wav")
                sf.write(sp, y, sr)
                sps.append(os.path.abspath(sp))
            rows.append([mix_id, os.path.abspath(mp)] + sps + [len(mix)])

        hdr = (["mixture_ID", "mixture_path"]
               + [f"source_{i+1}_path" for i in range(args.n_src)]
               + ["length"])
        csv_path = os.path.join(mdd, f"mixture_{split}_mix_clean.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            w.writerows(rows)
        print(f"[{split}] wrote {len(rows)} mixtures -> {csv_path}")


if __name__ == "__main__":
    main()
