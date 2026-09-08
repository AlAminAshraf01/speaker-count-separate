#!/usr/bin/env python3
"""
count_leak_probe.py -- is the speaker-count label leaking out of your mixtures?

Run this BEFORE you train a speaker-counting head, and put its output in the
report.  It does double duty:

  * it is the "naive predictor" benchmark the lab guidelines require -- a
    depth-3 decision tree on hand-made scalar features, no neural network; and
  * it is the data-leakage audit for the counting task.

Why it matters
--------------
LibriMix normalises every source to an independent loudness target drawn from
U(-33, -25) LUFS and then SUMS them, so the mixture level rises by about
10*log10(N) dB.  In `min` mode the mixture is also truncated to the shortest of
the N sources, so duration falls as N rises.  Either scalar alone predicts N far
above chance.  If your counting head scores 95% but this script scores 60%, most
of your accuracy is bookkeeping, not acoustics.

Feature groups
--------------
  ARTEFACT   duration, rms_db          -- consequences of the mixing recipe
  ACOUSTIC   crest, kurtosis, flatness, zcr, flux
                                       -- real density cues: summing N sparse
                                          speech signals makes the mixture less
                                          sparse and more Gaussian
The probe is run three ways: artefact only, acoustic only, and after the two
standard mitigations (fixed-length crop + per-utterance RMS normalisation), which
delete the artefact features by construction.

Usage
-----
    python count_leak_probe.py \
        --mix  2:/path/Libri2Mix/wav8k/min/test/mix_clean \
               3:/path/Libri3Mix/wav8k/min/test/mix_clean \
               4:/path/Libri4Mix/wav8k/min/test/mix_clean \
               5:/path/Libri5Mix/wav8k/min/test/mix_clean \
        --per_class 600 --crop 3.0
"""
import argparse
import os
import numpy as np
import soundfile as sf
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix

EPS = 1e-10
ARTEFACT = ["duration", "rms_db"]
ACOUSTIC = ["crest", "kurtosis", "flatness", "zcr", "flux"]


def features(x, sr, crop=None, rms_norm=False):
    """Scalar features from one mixture. crop/rms_norm apply the mitigations."""
    dur = len(x) / sr
    if crop is not None:
        n = int(crop * sr)
        if len(x) < n:
            return None
        s = (len(x) - n) // 2
        x = x[s:s + n]
        dur = crop
    rms = float(np.sqrt(np.mean(x ** 2)) + EPS)
    if rms_norm:
        x = x / rms
        rms = 1.0
    peak = float(np.max(np.abs(x)) + EPS)
    xm = x - x.mean()
    var = float(xm.var() + EPS)
    # spectral features from a single magnitude spectrogram
    n_fft, hop = 512, 256
    if len(x) >= n_fft:
        nfr = 1 + (len(x) - n_fft) // hop
        idx = np.arange(n_fft)[None, :] + hop * np.arange(nfr)[:, None]
        win = np.hanning(n_fft)
        S = np.abs(np.fft.rfft(x[idx] * win, axis=-1)) + EPS
        flatness = float(np.mean(np.exp(np.mean(np.log(S), 1)) / np.mean(S, 1)))
        flux = float(np.mean(np.abs(np.diff(S, axis=0)))) if nfr > 1 else 0.0
    else:
        flatness = flux = 0.0
    return {
        "duration": dur,
        "rms_db": 20 * np.log10(rms),
        "crest": 20 * np.log10(peak / rms),
        "kurtosis": float(np.mean(xm ** 4) / var ** 2),
        "flatness": flatness,
        "zcr": float(np.mean(np.abs(np.diff(np.sign(x))) > 0)),
        "flux": flux,
    }


def collect(dirs, per_class, crop, rms_norm, seed=0):
    rng = np.random.default_rng(seed)
    rows, ys = [], []
    for n, d in sorted(dirs.items()):
        files = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
        if len(files) > per_class:
            files = list(rng.choice(files, per_class, replace=False))
        kept = 0
        for fn in files:
            x, sr = sf.read(os.path.join(d, fn), dtype="float32")
            if x.ndim > 1:
                x = x[:, 0]
            f = features(x, sr, crop=crop, rms_norm=rms_norm)
            if f is None:
                continue
            rows.append(f); ys.append(n); kept += 1
        print(f"  N={n}: {kept} mixtures from {d}")
    keys = ARTEFACT + ACOUSTIC
    X = np.array([[r[k] for k in keys] for r in rows])
    return X, np.array(ys), keys


def probe(X, y, keys, use, label):
    cols = [keys.index(k) for k in use if k in keys]
    if not cols:
        return
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    acc = cross_val_score(clf, X[:, cols], y, cv=cv).mean()
    chance = 1.0 / len(np.unique(y))
    flag = "  <-- LEAK" if acc > chance * 1.5 else ""
    print(f"    {label:<38s} {acc*100:5.1f}%   (chance {chance*100:.0f}%){flag}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", nargs="+", required=True,
                    help="N:/path/to/mix_clean, one per speaker count")
    ap.add_argument("--per_class", type=int, default=600)
    ap.add_argument("--crop", type=float, default=3.0,
                    help="segment length used by the mitigated probe")
    args = ap.parse_args()
    dirs = {}
    for spec in args.mix:
        n, path = spec.split(":", 1)
        dirs[int(n)] = path

    print("\n[1] RAW MIXTURES  (as generated)")
    X, y, keys = collect(dirs, args.per_class, crop=None, rms_norm=False)
    print(f"\n  naive-predictor accuracy, depth-3 tree, 5-fold CV:")
    probe(X, y, keys, ARTEFACT, "ARTEFACT only (duration + level)")
    probe(X, y, keys, ACOUSTIC, "ACOUSTIC only (sparsity + spectrum)")
    probe(X, y, keys, ARTEFACT + ACOUSTIC, "everything")

    print("\n  per-class artefact statistics (this is what the tree is reading):")
    print(f"    {'N':>2} {'duration':>10} {'rms_db':>9} {'crest':>8} {'kurtosis':>9}")
    for n in sorted(set(y)):
        m = X[y == n].mean(0)
        print(f"    {n:>2} {m[keys.index('duration')]:9.2f}s "
              f"{m[keys.index('rms_db')]:8.2f} {m[keys.index('crest')]:7.2f} "
              f"{m[keys.index('kurtosis')]:8.2f}")

    print(f"\n[2] AFTER MITIGATION  ({args.crop} s centre crop + RMS normalisation)")
    Xm, ym, keys = collect(dirs, args.per_class, crop=args.crop, rms_norm=True)
    print(f"\n  naive-predictor accuracy:")
    probe(Xm, ym, keys, ARTEFACT, "ARTEFACT only (now constant by design)")
    probe(Xm, ym, keys, ACOUSTIC, "ACOUSTIC only")
    a = probe(Xm, ym, keys, ARTEFACT + ACOUSTIC, "everything")

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    from sklearn.model_selection import cross_val_predict
    pred = cross_val_predict(DecisionTreeClassifier(max_depth=3, random_state=0),
                             Xm, ym, cv=cv)
    labs = sorted(set(ym))
    print(f"\n  confusion matrix after mitigation (rows = true N):")
    print("       " + "".join(f"{p:>7}" for p in labs))
    for i, r in enumerate(confusion_matrix(ym, pred, labels=labs)):
        print(f"    {labs[i]:>2} " + "".join(f"{v:>7}" for v in r))

    print(f"\n  INTERPRETATION")
    print(f"    Report the mitigated 'everything' number ({a*100:.1f}%) as the naive")
    print(f"    baseline your counting head must beat.  If the ARTEFACT-only row in")
    print(f"    section [1] is well above chance, you MUST train and evaluate the")
    print(f"    counter on fixed-length, RMS-normalised input, or the accuracy you")
    print(f"    report is measuring the mixing script, not the speech.")


if __name__ == "__main__":
    main()
