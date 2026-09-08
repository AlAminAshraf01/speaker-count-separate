"""Opening the box: the learned filterbank, mask geometry, and what carries the count.

This is the project's actual contribution, and it needs **no extra training** -- everything
here is forward passes on a checkpoint you already have. Protect this time when the
schedule slips.

The argument: as N grows, the network must partition the *same* ``n_filters`` encoder basis
among more sources. If mask overlap rises and sparsity falls with N, then the degradation
curve has a mechanistic explanation computed from the network's own internals, rather than
being asserted.

The count head makes that measurable rather than merely visible. It reads the same shared
features whose geometry we measure, so three questions become answerable:

1. does count-head confidence track measured pairwise mask overlap?
2. do miscounts have a mask-geometry signature?
3. which encoder basis functions carry the count? (ablate and watch accuracy fall)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from .constants import EPS, SR
from .metrics import best_permutation


# --------------------------------------------------------------------------- filterbank

def extract_filterbank(model: torch.nn.Module) -> np.ndarray:
    """Encoder basis functions as ``(n_filters, kernel)`` in the time domain."""
    net = getattr(model, "module", model)
    weight = net.encoder.weight.detach().cpu().numpy()
    return np.asarray(weight[:, 0, :], dtype=np.float64)


def filter_spectra(filters: np.ndarray, n_fft: int | None = None) -> np.ndarray:
    """Magnitude response of every filter, ``(n_filters, n_fft // 2 + 1)``."""
    filters = np.asarray(filters, dtype=np.float64)
    if n_fft is None:
        n_fft = max(256, 8 * filters.shape[1])
    return np.abs(np.fft.rfft(filters, n=n_fft, axis=-1))


def filter_centre_frequencies(filters: np.ndarray, sr: int = SR,
                              n_fft: int | None = None) -> np.ndarray:
    """Spectral centroid of each filter, in Hz -- how the basis tiles the spectrum."""
    spectra = filter_spectra(filters, n_fft)
    freqs = np.linspace(0.0, sr / 2.0, spectra.shape[1])
    weights = spectra / (spectra.sum(axis=1, keepdims=True) + EPS)
    return (weights * freqs[None, :]).sum(axis=1)


def peak_frequencies(filters: np.ndarray, sr: int = SR,
                     n_fft: int | None = None) -> np.ndarray:
    """Frequency of each filter's magnitude peak, in Hz."""
    spectra = filter_spectra(filters, n_fft)
    freqs = np.linspace(0.0, sr / 2.0, spectra.shape[1])
    return freqs[np.argmax(spectra, axis=1)]


def mel_reference(n: int, sr: int = SR, fmin: float = 0.0) -> np.ndarray:
    """``n`` mel-spaced centre frequencies, for comparison against the learned basis."""
    def to_mel(f: np.ndarray | float) -> Any:
        return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    def from_mel(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    return from_mel(np.linspace(to_mel(fmin), to_mel(sr / 2.0), int(n)))


# --------------------------------------------------------------------------- mask geometry

def hoyer_sparsity(x: np.ndarray) -> float:
    """Hoyer sparsity in [0, 1]; 1 means all the mass sits on one coefficient."""
    x = np.abs(np.asarray(x, dtype=np.float64)).ravel()
    d = x.size
    if d <= 1:
        return float("nan")
    l1, l2 = x.sum(), np.sqrt((x ** 2).sum())
    if l2 < EPS:
        return float("nan")
    return float((np.sqrt(d) - l1 / l2) / (np.sqrt(d) - 1.0))


def gini(x: np.ndarray) -> float:
    """Gini coefficient of the mask values -- a second, differently-biased sparsity read."""
    x = np.sort(np.abs(np.asarray(x, dtype=np.float64)).ravel())
    n = x.size
    total = x.sum()
    if n == 0 or total < EPS:
        return float("nan")
    index = np.arange(1, n + 1)
    return float(((2.0 * index - n - 1) * x).sum() / (n * total))


def pairwise_overlap(masks: np.ndarray) -> dict[str, float]:
    """Mean cosine and mean intersection-over-union between every pair of masks.

    ``masks`` is ``(k, n_filters, frames)``. Rising overlap with N is the mechanism behind
    the degradation curve.
    """
    masks = np.asarray(masks, dtype=np.float64)
    k = masks.shape[0]
    if k < 2:
        return {"cosine": float("nan"), "iou": float("nan")}
    flat = np.abs(masks.reshape(k, -1))
    cosines, ious = [], []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = flat[i], flat[j]
            denom = np.linalg.norm(a) * np.linalg.norm(b) + EPS
            cosines.append(float(a.dot(b) / denom))
            ious.append(float(np.minimum(a, b).sum() / (np.maximum(a, b).sum() + EPS)))
    return {"cosine": float(np.mean(cosines)), "iou": float(np.mean(ious))}


def mask_entropy(masks: np.ndarray) -> float:
    """Mean entropy (nats) of the per-cell distribution across slots.

    Low entropy means each time-frequency cell is claimed by one source; high entropy means
    the sources are sharing the basis.
    """
    masks = np.abs(np.asarray(masks, dtype=np.float64))
    k = masks.shape[0]
    if k < 2:
        return float("nan")
    total = masks.sum(axis=0, keepdims=True) + EPS
    probs = masks / total
    entropy = -(probs * np.log(probs + EPS)).sum(axis=0)
    return float(entropy.mean())


def mask_stats(masks: np.ndarray) -> dict[str, float]:
    """Every mask statistic for one utterance's selected slots, ``(k, n_filters, frames)``."""
    masks = np.asarray(masks, dtype=np.float64)
    overlap = pairwise_overlap(masks)
    return {
        "sparsity_hoyer": float(np.mean([hoyer_sparsity(m) for m in masks])),
        "sparsity_gini": float(np.mean([gini(m) for m in masks])),
        "overlap_cosine": overlap["cosine"],
        "overlap_iou": overlap["iou"],
        "entropy": mask_entropy(masks),
        "mean_mask_value": float(masks.mean()),
        "active_fraction": float((masks > 0.1 * (masks.max() + EPS)).mean()),
    }


def select_matched_slots(est: np.ndarray, refs: np.ndarray, n_src: int,
                         max_n_src: int) -> list[int]:
    """Which output slots the true sources actually landed in (PIT match at eval time)."""
    _, cols = best_permutation(est[:max_n_src], refs, n_src)
    return [int(c) for c in cols]


# --------------------------------------------------------------------------- ablation

@contextmanager
def ablate_encoder_filters(model: torch.nn.Module,
                           indices: Sequence[int]) -> Iterator[torch.nn.Module]:
    """Temporarily zero a set of encoder output channels.

    Ablate, re-measure counting accuracy and SI-SDRi, and you have an attribution curve
    mapping the count decision onto specific learned basis functions.
    """
    net = getattr(model, "module", model)
    index = torch.as_tensor(list(indices), dtype=torch.long)

    def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
        output = output.clone()
        output[:, index.to(output.device), :] = 0.0
        return output

    handle = net.encoder.register_forward_hook(hook)
    try:
        yield model
    finally:
        handle.remove()


def filter_importance_by_energy(model: torch.nn.Module, loader: Any, device: Any, *,
                                max_batches: int = 8) -> np.ndarray:
    """Mean encoder activation per filter -- the ranking used to choose what to ablate."""
    net = getattr(model, "module", model)
    net.eval()
    totals: torch.Tensor | None = None
    seen = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= int(max_batches):
                break
            mix = batch["mix"].to(device)
            out = net(mix, return_internals=True)
            energy = out["enc"].float().mean(dim=(0, 2))
            totals = energy if totals is None else totals + energy
            seen += 1
    if totals is None:
        return np.zeros(net.cfg.n_filters, dtype=np.float64)
    return (totals / max(1, seen)).cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------- collection

@torch.no_grad()
def collect_mask_records(model: torch.nn.Module, loader: Any, device: Any, *,
                         max_batches: int | None = None,
                         max_n_src: int = 5) -> list[dict]:
    """Per-utterance mask geometry, count prediction and confidence.

    This is the raw material for every figure in ``scripts/07_interpret.py``.
    """
    from .constants import class_to_n

    net = getattr(model, "module", model)
    net.eval()
    records: list[dict] = []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= int(max_batches):
            break
        mix = batch["mix"].to(device)
        out = net(mix, return_internals=True)
        masks = out["masks"].float().cpu().numpy()
        est = out["est"].float().cpu().numpy()
        logits = out["count_logits"].float().cpu().numpy()
        refs = batch["refs"].numpy()
        n_src = batch["n_src"].numpy()

        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

        for b in range(masks.shape[0]):
            n_true = int(n_src[b])
            n_pred = int(class_to_n(int(probs[b].argmax())))
            slots = select_matched_slots(est[b], refs[b, :n_true], n_true, max_n_src)
            stats = mask_stats(masks[b][slots])
            stats.update({
                "n_true": n_true,
                "n_pred": n_pred,
                "correct": int(n_pred == n_true),
                "confidence": float(probs[b].max()),
                "entropy_count": float(-(probs[b] * np.log(probs[b] + EPS)).sum()),
            })
            records.append(stats)
    return records


def group_by_n(records: Sequence[dict], keys: Sequence[str]) -> dict[int, dict[str, float]]:
    """Average the named statistics within each true speaker count."""
    out: dict[int, dict[str, float]] = {}
    for n in sorted({int(r["n_true"]) for r in records}):
        group = [r for r in records if int(r["n_true"]) == n]
        stats: dict[str, float] = {}
        for key in keys:
            # overlap/entropy are undefined for a single source, so a whole group can be
            # NaN. That is a real "not applicable", not a bug -- report it as NaN quietly.
            values = [float(r[key]) for r in group if key in r and np.isfinite(r[key])]
            stats[key] = float(np.mean(values)) if values else float("nan")
        stats["n"] = len(group)
        out[n] = stats
    return out


def correlate(records: Sequence[dict], x_key: str, y_key: str) -> dict[str, float]:
    """Pearson and Spearman correlation between two recorded statistics."""
    from scipy.stats import pearsonr, spearmanr

    x = np.array([r[x_key] for r in records], dtype=np.float64)
    y = np.array([r[y_key] for r in records], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_r": float("nan"), "spearman_p": float("nan"), "n": int(mask.sum())}
    pr, pp = pearsonr(x[mask], y[mask])
    sr, sp = spearmanr(x[mask], y[mask])
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp), "n": int(mask.sum())}
