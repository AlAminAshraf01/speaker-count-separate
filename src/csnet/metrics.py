"""Evaluation metrics: SI-SDR(i), P-SI-SNR, and counting reports.

Report four numbers, never one:

1. counting accuracy **plus the full confusion matrix** -- the classes are ordinal, so
   3 -> 4 is not the same failure as 3 -> 5 and an average hides that;
2. **P-SI-SNR** over the whole test set -- the honest end-to-end number, defined even when
   the predicted count is wrong;
3. **SI-SDRi per N on the count-correct subset only** -- isolates separation from counting
   and is what compares to the fixed-N literature;
4. the **naive-predictor** row from :mod:`csnet.baselines`, so accuracy has a floor.

Always report SI-SDR *improvement*: raw input SI-SDR itself falls with N, so raw numbers
across N are not comparable.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .constants import EPS, N_LIST, SILENCE_DB


def _np(x: Any) -> np.ndarray:
    """Accept torch tensors or numpy arrays; always return float64 numpy."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def si_sdr(est: Any, ref: Any, eps: float = EPS) -> np.ndarray:
    """Scale-invariant SDR in dB, reduced over the last axis."""
    est, ref = _np(est), _np(ref)
    est = est - est.mean(axis=-1, keepdims=True)
    ref = ref - ref.mean(axis=-1, keepdims=True)
    alpha = ((est * ref).sum(-1, keepdims=True)
             / ((ref ** 2).sum(-1, keepdims=True) + eps))
    proj = alpha * ref
    return (10.0 * np.log10((proj ** 2).sum(-1) + eps)
            - 10.0 * np.log10(((est - proj) ** 2).sum(-1) + eps))


def si_sdr_improvement(est: Any, ref: Any, mix: Any, eps: float = EPS) -> np.ndarray:
    """SI-SDR of the estimate minus SI-SDR of the unprocessed mixture."""
    est, ref, mix = _np(est), _np(ref), _np(mix)
    if mix.ndim < est.ndim:
        mix = np.broadcast_to(mix[..., None, :], ref.shape)
    return si_sdr(est, ref, eps) - si_sdr(mix, ref, eps)


def pairwise_si_sdr_np(est: np.ndarray, refs: np.ndarray, eps: float = EPS) -> np.ndarray:
    """(S, T) estimates against (R, T) references -> (R, S) SI-SDR matrix."""
    est, refs = _np(est), _np(refs)
    return si_sdr(est[None, :, :], refs[:, None, :], eps)


def best_permutation(est: np.ndarray, refs: np.ndarray, n_src: int | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Optimal assignment of references to output slots (rectangular Hungarian)."""
    from scipy.optimize import linear_sum_assignment

    est, refs = _np(est), _np(refs)
    if n_src is not None:
        refs = refs[: int(n_src)]
    cost = -pairwise_si_sdr_np(est, refs)
    rows, cols = linear_sum_assignment(cost)
    return np.asarray(rows), np.asarray(cols)


def matched_si_sdri(est: np.ndarray, refs: np.ndarray, mix: np.ndarray,
                    n_src: int) -> np.ndarray:
    """PIT-matched per-source SI-SDR improvement. Matching happens at eval time too."""
    est, refs, mix = _np(est), _np(refs), _np(mix)
    rows, cols = best_permutation(est, refs, n_src)
    out = np.empty(len(rows), dtype=np.float64)
    for k, (r, c) in enumerate(zip(rows, cols)):
        out[k] = si_sdr(est[c], refs[r]) - si_sdr(mix, refs[r])
    return out


def p_si_snr(est_slots: np.ndarray, refs: np.ndarray, n_true: int, n_pred: int,
             p_ref: float = SILENCE_DB, max_n_src: int | None = None) -> float:
    """Multi-Decoder-DPRNN P-SI-SNR: defined even when the predicted count is wrong.

    ``P-SI-SNR = (L_match + L_pad) / max(n_true, n_pred)`` where ``L_match`` sums SI-SNR
    over the matched channels and ``L_pad = p_ref * |n_true - n_pred|`` charges one
    silence-floor penalty per missing or spurious source.

    Which ``n_pred`` slots count as "predicted" is a choice, and this is ours: the
    ``n_pred`` highest-energy speaker slots.
    """
    est_slots, refs = _np(est_slots), _np(refs)
    if max_n_src is None:
        max_n_src = est_slots.shape[0]
    speaker_slots = est_slots[: int(max_n_src)]

    n_true, n_pred = int(n_true), int(n_pred)
    n_pred = max(0, min(n_pred, speaker_slots.shape[0]))
    if n_pred == 0:
        return float(p_ref)

    energies = (speaker_slots ** 2).mean(axis=-1)
    chosen = np.argsort(-energies)[:n_pred]
    est_sel = speaker_slots[chosen]
    refs_sel = refs[:n_true]

    n_match = min(n_true, n_pred)
    if n_match == 0:
        return float(p_ref)

    from scipy.optimize import linear_sum_assignment

    cost = -pairwise_si_sdr_np(est_sel, refs_sel)
    rows, cols = linear_sum_assignment(cost)
    matched = np.array([-cost[r, c] for r, c in zip(rows, cols)], dtype=np.float64)

    l_match = float(matched.sum())
    l_pad = float(p_ref) * abs(n_true - n_pred)
    return (l_match + l_pad) / max(n_true, n_pred)


def count_report(y_true: Sequence[int], y_pred: Sequence[int],
                 n_list: Sequence[int] = N_LIST) -> dict:
    """Accuracy, MAE, per-class accuracy and the full confusion matrix."""
    y_true = np.asarray(list(y_true), dtype=np.int64)
    y_pred = np.asarray(list(y_pred), dtype=np.int64)
    labels = [int(n) for n in n_list]
    index = {n: i for i, n in enumerate(labels)}

    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if int(t) in index and int(p) in index:
            confusion[index[int(t)], index[int(p)]] += 1

    per_class: dict[int, float] = {}
    for n in labels:
        mask = y_true == n
        per_class[n] = float((y_pred[mask] == n).mean()) if mask.any() else float("nan")

    return {
        "accuracy": float((y_true == y_pred).mean()) if y_true.size else float("nan"),
        "mae": float(np.abs(y_true - y_pred).mean()) if y_true.size else float("nan"),
        "per_class_acc": per_class,
        "support": {n: int((y_true == n).sum()) for n in labels},
        "confusion": confusion,
        "labels": labels,
    }


def format_confusion(confusion: np.ndarray, n_list: Sequence[int] = N_LIST,
                     normalise: bool = False) -> str:
    """Render a confusion matrix as text, rows = true N."""
    confusion = np.asarray(confusion)
    labels = [int(n) for n in n_list]
    head = "true\\pred " + "".join(f"{n:>8}" for n in labels) + "     total"
    lines = [head, "-" * len(head)]
    for i, n in enumerate(labels):
        row = confusion[i]
        total = int(row.sum())
        if normalise and total:
            cells = "".join(f"{100.0 * v / total:>7.1f}%" for v in row)
        else:
            cells = "".join(f"{int(v):>8}" for v in row)
        lines.append(f"{n:>9} {cells} {total:>9}")
    return "\n".join(lines)


def summarise_per_n(records: Sequence[dict], n_list: Sequence[int] = N_LIST) -> dict:
    """Group per-utterance records by true N and average the usual metrics.

    Each record needs ``n_true``, ``n_pred``, ``p_si_snr`` and (when the count is right)
    ``si_sdri``.
    """
    out: dict[int, dict] = {}
    for n in [int(x) for x in n_list]:
        group = [r for r in records if int(r["n_true"]) == n]
        if not group:
            continue
        correct = [r for r in group if int(r["n_pred"]) == n and r.get("si_sdri") is not None]
        out[n] = {
            "n": len(group),
            "count_acc": float(np.mean([int(r["n_pred"]) == n for r in group])),
            "p_si_snr": float(np.mean([r["p_si_snr"] for r in group])),
            "n_count_correct": len(correct),
            "si_sdri_count_correct": (float(np.mean([np.mean(r["si_sdri"]) for r in correct]))
                                      if correct else float("nan")),
            "input_si_sdr": (float(np.mean([r["input_si_sdr"] for r in group
                                            if r.get("input_si_sdr") is not None]))
                             if any(r.get("input_si_sdr") is not None for r in group)
                             else float("nan")),
        }
    return out
