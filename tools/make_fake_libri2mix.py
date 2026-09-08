#!/usr/bin/env python3
"""Generate a tiny synthetic corpus shaped exactly like Libri2Mix, for smoke tests.

The real dataset is 10 GB. Every script in this repo must be runnable end to end in a
couple of minutes before anyone spends GPU quota on it, so this builds a ~30 MB stand-in
with the same directory layout and the same filename grammar::

    <out>/wav8k/min/{train-100,dev,test}/{mix_clean,s1,s2}/<uttA>_<uttB>.wav

"Speech" is a source-filter cartoon: a speaker-specific F0 glides through a harmonic
stack, three speaker-specific formants shape it, and voiced segments are separated by
silences. It is not speech, but it has the properties the pipeline cares about --
harmonic structure, formants that differ per speaker, sparsity in time, and a plausible
crest factor -- so counting and separation are non-trivial on it.

Usage::

    python tools/make_fake_libri2mix.py --out /tmp/fake --speakers 24 --utts 8
"""

from __future__ import annotations

import argparse
import os

import numpy as np

SR = 8000


def synth_utterance(rng: np.random.Generator, f0: float, formants: np.ndarray,
                    seconds: float) -> np.ndarray:
    """One cartoon utterance: voiced harmonic bursts shaped by fixed formants."""
    n = int(seconds * SR)
    t = np.arange(n, dtype=np.float64) / SR

    # slow pitch contour around the speaker's F0
    contour = f0 * (1.0 + 0.06 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6.28))
                    + 0.03 * np.sin(2 * np.pi * 1.9 * t + rng.uniform(0, 6.28)))
    phase = 2 * np.pi * np.cumsum(contour) / SR

    signal = np.zeros(n, dtype=np.float64)
    for harmonic in range(1, 26):
        freq = contour * harmonic
        gain = 1.0 / harmonic
        # crude formant emphasis
        for centre, width in zip(formants, (120.0, 160.0, 220.0)):
            gain = gain * (1.0 + 2.5 * np.exp(-0.5 * ((freq - centre) / width) ** 2))
        gain = np.where(freq < 0.45 * SR, gain, 0.0)
        signal += gain * np.sin(harmonic * phase)

    # a little breath noise, then syllable-rate amplitude modulation with real silences
    signal += 0.05 * rng.standard_normal(n)
    envelope = np.zeros(n, dtype=np.float64)
    pos = 0
    while pos < n:
        on = int(rng.uniform(0.12, 0.45) * SR)
        off = int(rng.uniform(0.04, 0.30) * SR)
        end = min(n, pos + on)
        ramp = np.hanning(max(2, end - pos))
        envelope[pos:end] = ramp[: end - pos] * rng.uniform(0.5, 1.0)
        pos = end + off
    signal *= envelope

    peak = float(np.max(np.abs(signal))) + 1e-9
    return (0.7 * signal / peak).astype(np.float32)


def build(out_root: str, n_speakers: int, n_utts: int, seed: int) -> None:
    """Write the full fake tree."""
    import soundfile as sf

    rng = np.random.default_rng(seed)
    splits = {"train-100": (0, n_speakers), "dev": (n_speakers, n_speakers + 12),
              "test": (n_speakers + 12, n_speakers + 24)}

    base = os.path.join(out_root, "wav8k", "min")
    for split, (lo, hi) in splits.items():
        for sub in ("mix_clean", "s1", "s2"):
            os.makedirs(os.path.join(base, split, sub), exist_ok=True)

        # speaker ids are globally disjoint across splits, exactly like LibriSpeech
        speakers = [f"{1000 + i}" for i in range(lo, hi)]
        utterances: list[tuple[str, np.ndarray]] = []
        for spk in speakers:
            spk_rng = np.random.default_rng(seed + int(spk))
            f0 = float(spk_rng.uniform(85, 240))
            formants = np.sort(spk_rng.uniform([350, 900, 2100], [850, 1900, 3200]))
            for u in range(n_utts):
                seconds = float(spk_rng.uniform(3.5, 9.0))
                utt_id = f"{spk}-{100 + u}-{u:04d}"
                utterances.append((utt_id, synth_utterance(spk_rng, f0, formants, seconds)))

        order = rng.permutation(len(utterances))
        n_pairs = len(order) // 2
        for p in range(n_pairs):
            (id_a, wav_a) = utterances[int(order[2 * p])]
            (id_b, wav_b) = utterances[int(order[2 * p + 1])]
            length = min(wav_a.size, wav_b.size)          # `min` mode truncation
            a, b = wav_a[:length], wav_b[:length]
            mix = a + b
            peak = float(np.max(np.abs(mix))) + 1e-9
            if peak > 0.9:
                factor = 0.9 / peak
                a, b, mix = a * factor, b * factor, mix * factor
            name = f"{id_a}_{id_b}.wav"
            sf.write(os.path.join(base, split, "s1", name), a, SR, subtype="PCM_16")
            sf.write(os.path.join(base, split, "s2", name), b, SR, subtype="PCM_16")
            sf.write(os.path.join(base, split, "mix_clean", name), mix, SR, subtype="PCM_16")

        print(f"{split:>10}: {len(speakers)} speakers, {len(utterances)} utterances, "
              f"{n_pairs} pairs")

    print(f"\nfake Libri2Mix at {os.path.join(base)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="root; <out>/wav8k/min/... is created")
    ap.add_argument("--speakers", type=int, default=24, help="train-100 speakers")
    ap.add_argument("--utts", type=int, default=8, help="utterances per speaker")
    ap.add_argument("--seed", type=int, default=72)
    args = ap.parse_args()
    build(args.out, args.speakers, args.utts, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
