"""The baselines the report has to beat -- or explain losing to.

Two families:

* **Oracle masks** (IRM / IBM). Computed *from the true sources*, so they are an upper
  bound on any masking approach, not a competitor. Worth keeping because on LibriMix
  Conv-TasNet beats them at 2 speakers and *loses* to them at 3 -- that crossover is a
  finding, and it is why the lab guidelines mandate ideal-mask baselines.

* **The naive count predictor**: a depth-limited decision tree on seven hand-made scalars.
  It does double duty as the mandated naive-predictor benchmark *and* as the data-leakage
  audit for the counting task. If the counting head scores 95 % and this scores 60 %, most
  of that accuracy is bookkeeping, not acoustics.

Feature groups
--------------
``ARTEFACT``  duration, rms_db -- consequences of the *mixing recipe*, not of speech
``ACOUSTIC``  crest, kurtosis, flatness, zcr, flux -- real density cues: summing N sparse
              speech signals makes the mixture less sparse and more Gaussian
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .audio import istft, magnitude_spectrogram, stft
from .constants import EPS, SR
from .metrics import si_sdr

ARTEFACT: tuple[str, ...] = ("duration", "rms_db")
ACOUSTIC: tuple[str, ...] = ("crest", "kurtosis", "flatness", "zcr", "flux")
ALL_FEATURES: tuple[str, ...] = ARTEFACT + ACOUSTIC


# --------------------------------------------------------------------------- oracle masks

def ideal_mask_estimates(mix: np.ndarray, sources: np.ndarray, *, sr: int = SR,
                         n_fft: int = 256, hop: int = 64, mode: str = "irm"
                         ) -> np.ndarray:
    """Oracle-masked estimates of every source, same shape as ``sources``."""
    mix = np.asarray(mix, dtype=np.float32)
    sources = np.asarray(sources, dtype=np.float32)
    mix_spec = stft(mix, n_fft=n_fft, hop=hop, sr=sr)
    source_specs = np.stack([stft(s, n_fft=n_fft, hop=hop, sr=sr) for s in sources])
    magnitudes = np.abs(source_specs)

    if mode == "irm":
        masks = magnitudes / (magnitudes.sum(axis=0, keepdims=True) + EPS)
    elif mode == "ibm":
        winner = np.argmax(magnitudes, axis=0)
        masks = np.stack([(winner == k).astype(np.float32) for k in range(len(sources))])
    else:
        raise ValueError(f"mode must be 'irm' or 'ibm', got {mode!r}")

    length = sources.shape[-1]
    return np.stack([istft(masks[k] * mix_spec, n_fft=n_fft, hop=hop, sr=sr, length=length)
                     for k in range(len(sources))])


def ideal_ratio_mask_sisdri(mix: np.ndarray, sources: np.ndarray, *, sr: int = SR,
                            n_fft: int = 256, hop: int = 64, mode: str = "irm"
                            ) -> np.ndarray:
    """Per-source SI-SDR improvement of the oracle-masked mixture (the upper bound)."""
    estimates = ideal_mask_estimates(mix, sources, sr=sr, n_fft=n_fft, hop=hop, mode=mode)
    mix = np.asarray(mix, dtype=np.float32)
    return np.array([float(si_sdr(estimates[k], sources[k]) - si_sdr(mix, sources[k]))
                     for k in range(len(sources))], dtype=np.float64)


# --------------------------------------------------------------------------- naive counter

def naive_count_features(x: np.ndarray, sr: int = SR) -> dict[str, float]:
    """The seven scalars the naive predictor sees. No neural network, no training."""
    x = np.asarray(x, dtype=np.float32)
    duration = float(x.size) / float(sr)
    rms = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))) + EPS)
    peak = float(np.max(np.abs(x)) + EPS) if x.size else EPS
    centred = x - float(x.mean())
    variance = float(centred.var() + EPS)

    spec = magnitude_spectrogram(x, n_fft=512, hop=256)
    if spec.shape[0] > 0:
        flatness = float(np.mean(np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1)))
        flux = float(np.mean(np.abs(np.diff(spec, axis=0)))) if spec.shape[0] > 1 else 0.0
    else:
        flatness = flux = 0.0

    return {
        "duration": duration,
        "rms_db": 20.0 * float(np.log10(rms)),
        "crest": 20.0 * float(np.log10(peak / rms)),
        "kurtosis": float(np.mean(centred.astype(np.float64) ** 4) / variance ** 2),
        "flatness": flatness,
        "zcr": float(np.mean(np.abs(np.diff(np.sign(x))) > 0)) if x.size > 1 else 0.0,
        "flux": flux,
    }


def features_to_matrix(rows: Sequence[dict], keys: Sequence[str] = ALL_FEATURES
                       ) -> tuple[np.ndarray, list[str]]:
    """Stack feature dicts into a design matrix."""
    keys = list(keys)
    matrix = np.array([[float(r[k]) for k in keys] for r in rows], dtype=np.float64)
    return matrix, keys


def naive_count_baseline(X: np.ndarray, y: np.ndarray, *, max_depth: int = 3,
                         n_folds: int = 5, feature_names: Sequence[str] | None = None,
                         seed: int = 0) -> dict:
    """Depth-limited decision tree with stratified CV -- the mandated naive predictor."""
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    labels = sorted(set(int(v) for v in y))
    n_splits = max(2, min(int(n_folds), int(np.min(np.bincount(y)[np.array(labels)]))))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = DecisionTreeClassifier(max_depth=int(max_depth), random_state=seed)
    accuracy = float(cross_val_score(clf, X, y, cv=cv).mean())
    predictions = cross_val_predict(clf, X, y, cv=cv)

    fitted = DecisionTreeClassifier(max_depth=int(max_depth), random_state=seed).fit(X, y)
    importance = fitted.feature_importances_
    names = list(feature_names) if feature_names is not None else \
        [f"f{i}" for i in range(X.shape[1])]

    return {
        "accuracy": accuracy,
        "chance": 1.0 / len(labels),
        "n_folds": n_splits,
        "labels": labels,
        "confusion": confusion_matrix(y, predictions, labels=labels),
        "feature_importance": {n: float(v) for n, v in zip(names, importance)},
        "n_samples": int(X.shape[0]),
    }


def probe_feature_groups(X: np.ndarray, y: np.ndarray, keys: Sequence[str], *,
                         max_depth: int = 3, n_folds: int = 5) -> dict[str, dict]:
    """Run the naive predictor three ways: artefact only, acoustic only, everything."""
    keys = list(keys)
    groups = {
        "ARTEFACT only (duration + level)": [k for k in ARTEFACT if k in keys],
        "ACOUSTIC only (sparsity + spectrum)": [k for k in ACOUSTIC if k in keys],
        "everything": keys,
    }
    results: dict[str, dict] = {}
    for label, group in groups.items():
        if not group:
            continue
        columns = [keys.index(k) for k in group]
        results[label] = naive_count_baseline(X[:, columns], y, max_depth=max_depth,
                                              n_folds=n_folds, feature_names=group)
    return results


def mixture_as_estimate_sisdri(mix: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """The floor: hand back the mixture unchanged. Zero by definition."""
    mix = np.asarray(mix, dtype=np.float32)
    return np.zeros(len(sources), dtype=np.float64)


def input_si_sdr(mix: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """SI-SDR of the unprocessed mixture against each source (falls as N grows)."""
    return np.array([float(si_sdr(mix, s)) for s in sources], dtype=np.float64)


PUBLISHED_REFERENCE: dict[str, dict[int, float]] = {
    "Conv-TasNet - LibriMix 8k min (asteroid)": {2: 14.76, 3: 11.98},
    "Conv-TasNet - WSJ0-mix (paper)": {2: 15.3, 3: 12.7},
    "SepFormer + dynamic mixing - LibriMix 8k": {2: 20.4, 3: 19.0},
    "OR-PIT recursive - WSJ0-mix (SDRi)": {2: 15.0, 3: 12.9, 4: 10.6},
    "SepEDA transformer - WSJ0-mix, unknown N": {2: 21.1, 3: 18.4, 4: 14.4, 5: 11.6},
}
"""Literature anchors, for the benchmark table. Ours are not directly comparable past N=2."""
