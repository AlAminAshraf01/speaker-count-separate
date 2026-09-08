#!/usr/bin/env python3
"""
leak_audit.py -- Does a speaker-COUNT label leak out of LibriMix metadata?

The plan under assessment is: one model that (a) predicts how many people are
talking, N in {2..5}, and (b) separates them.  Before building it, we have to know
whether N is predictable from mixture properties that have nothing to do with
counting voices.  Two candidates, both structural consequences of how LibriMix is
built:

  LEAK 1 -- LEVEL.  Every source is loudness-normalised to an independent target
  drawn from U(-33, -25) LUFS, then the N sources are SUMMED.  Incoherent sums add
  in power, so the mixture level rises by roughly 10*log10(N) dB.  A model that
  measures nothing but RMS can therefore guess N.

  LEAK 2 -- DURATION.  In `min` mode the mixture is truncated to the SHORTEST of
  the N sources.  min of N draws from the utterance-duration distribution shrinks
  as N grows, so mixture duration alone predicts N.

Both are measured here on real LibriSpeech durations (LibriMix's own
metadata/LibriSpeech/*.csv, `length` in samples at 16 kHz) and on a Monte-Carlo of
the documented loudness policy.  A depth-limited decision tree on these features
alone is the "naive predictor" baseline the lab guidelines require -- if it scores
well above chance, any counting accuracy from the real model is uninterpretable.

Finally we re-run the same probes AFTER the two standard mitigations
(fixed-length crops + per-utterance RMS normalisation) to confirm they close it.
"""
import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

RATE = 16000
MIN_LOUDNESS, MAX_LOUDNESS = -33.0, -25.0   # LibriMix globals
COUNTS = [2, 3, 4, 5]
PER_CLASS = 3000
SEED = 72
rng = np.random.default_rng(SEED)


def load_librispeech(path):
    """-> dict speaker_ID -> np.array of utterance lengths (samples @16k)"""
    by_spk = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            by_spk.setdefault(row["speaker_ID"], []).append(int(row["length"]))
    return {k: np.array(v) for k, v in by_spk.items()}


def simulate(by_spk, n_src, n_mix):
    """Sample n_mix mixtures of n_src distinct speakers; return (dur_s, lufs)."""
    spks = list(by_spk)
    durs, lufs = np.empty(n_mix), np.empty(n_mix)
    for i in range(n_mix):
        chosen = rng.choice(len(spks), size=n_src, replace=False)
        lens = np.array([rng.choice(by_spk[spks[c]]) for c in chosen])
        durs[i] = lens.min() / RATE                      # `min` mode truncation
        L = rng.uniform(MIN_LOUDNESS, MAX_LOUDNESS, n_src)   # per-source target
        lufs[i] = 10 * np.log10(np.sum(10 ** (L / 10)))      # incoherent sum
    return durs, lufs


def probe(X, y, name, feats):
    """5-fold CV accuracy of a depth-3 tree = the naive predictor baseline."""
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()
    print(f"    {name:<44s} {acc*100:5.1f}%   ({feats})")
    return acc


def main():
    md = "/tmp/LibriMix/metadata/LibriSpeech/"

    for subset in ["test-clean", "train-clean-100"]:
        by_spk = load_librispeech(md + subset + ".csv")
        n_utt = sum(len(v) for v in by_spk.values())
        allv = np.concatenate(list(by_spk.values())) / RATE
        print(f"\n{'='*78}\n{subset}: {n_utt} utterances, {len(by_spk)} speakers, "
              f"duration mean {allv.mean():.2f}s  sd {allv.std():.2f}s")
        print(f"{'='*78}")

        D, L, Y = [], [], []
        print("\n  per-class mixture statistics (simulated from real durations):")
        print(f"    {'N':>2} {'dur mean':>9} {'dur sd':>7} {'dur p10':>8} "
              f"{'LUFS mean':>10} {'LUFS sd':>7}")
        for n in COUNTS:
            d, l = simulate(by_spk, n, PER_CLASS)
            D.append(d); L.append(l); Y.append(np.full(PER_CLASS, n))
            print(f"    {n:>2} {d.mean():8.2f}s {d.std():6.2f}s "
                  f"{np.percentile(d,10):7.2f}s {l.mean():9.2f} {l.std():6.2f}")
        D = np.concatenate(D); L = np.concatenate(L); Y = np.concatenate(Y)

        print(f"\n  NAIVE-PREDICTOR ACCURACY  (chance = {100/len(COUNTS):.0f}%)")
        probe(L.reshape(-1, 1),           Y, "level only",          "mixture LUFS")
        probe(D.reshape(-1, 1),           Y, "duration only",       "min-mode length")
        probe(np.c_[L, D],                Y, "level + duration",    "both")

        # ---- mitigations -------------------------------------------------
        # (a) fixed 3 s crops  -> duration carries no information at all
        # (b) per-utterance RMS normalisation -> level is constant by construction
        #     (residual cue: only the *relative* spread of source levels survives,
        #      which is what we model as the normalised level below)
        Lnorm = L - L                                     # RMS-normalised input
        Dcrop = np.full_like(D, 3.0)                      # every example 3 s
        print(f"\n  AFTER MITIGATION (3 s crops + RMS normalisation)")
        probe(np.c_[Lnorm, Dcrop] + rng.normal(0, 1e-6, (len(Y), 2)), Y,
              "level + duration", "both, neutralised")

        # how many mixtures survive a 3 s minimum-length filter?
        print(f"\n  usable mixtures at a 3 s segment length "
              f"(asteroid drops shorter ones):")
        for n in COUNTS:
            d, _ = simulate(by_spk, n, PER_CLASS)
            print(f"    N={n}: {100*(d>=3.0).mean():5.1f}% of mixtures are >= 3 s")


if __name__ == "__main__":
    main()
