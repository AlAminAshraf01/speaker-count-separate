"""Run the model over audio of any length, the way it was trained.

The model has only ever seen a **fixed 3-second crop divided by its RMS**. That crop is
half of the count-leak mitigation, not an implementation detail: without it the mixture's
level alone predicts the speaker count, because a LibriMix mixture of N sources is about
``10*log10(N)`` dB louder than one source. Feeding the network a ten-second utterance
therefore asks it a question it has never been asked, and the counting head -- which pools
mean and standard deviation over the whole time axis -- answers accordingly.

The gate-2 evaluation used to hand the official Libri2Mix test set to the model in single
ten-second blocks. Windowing it properly did not rescue that number -- 0 correct counts out
of 300 either way, against 52 % on our own three-second N=2 mixtures, so the transfer
failure is per-source normalisation rather than length -- but "the model was asked a
question it has never been asked" is not something to leave in place because fixing it
happened not to help.

So there is exactly one way to run this model on real audio, and it lives here rather than
in a script: overlapping 3 s windows, each RMS-normalised, permutation-aligned to the
window before it, Hann-overlap-added back together, with the per-window counts pooled.

The alignment step is the one that is easy to forget. A separator has no idea that slot 2
in window 7 is the same person as slot 4 in window 8 -- nothing in the training objective
ties slots to identities across time -- so concatenating windows without matching them
produces speaker tracks that swap people at every hop.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .audio import rms_normalize
from .constants import SEG_SECONDS, SR

WIN_SECONDS: float = SEG_SECONDS          # what the model was trained on
HOP_SECONDS: float = SEG_SECONDS / 2.0    # 50 % overlap


def window_starts(n_samples: int, win: int, hop: int) -> list[int]:
    """Window offsets covering the signal, with the last one flush to the end."""
    if n_samples <= win:
        return [0]
    starts = list(range(0, n_samples - win + 1, hop))
    if starts[-1] + win < n_samples:
        starts.append(n_samples - win)
    return starts


def align_permutation(previous: np.ndarray, current: np.ndarray) -> list[int]:
    """Match the current window's slots to the previous window's, over the overlap.

    ``previous`` and ``current`` are ``(slots, overlap)``. Returns, for each previous
    slot, the index of the current slot that continues it.
    """
    from scipy.optimize import linear_sum_assignment

    a = previous - previous.mean(axis=1, keepdims=True)
    b = current - current.mean(axis=1, keepdims=True)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    rows, cols = linear_sum_assignment(-np.abs(a @ b.T))
    mapping = list(range(current.shape[0]))
    for r, c in zip(rows, cols):
        mapping[int(r)] = int(c)
    return mapping


def pool_counts(probs: np.ndarray, agg: str = "mean_prob") -> int:
    """Turn per-window class probabilities into one class index.

    ``mean_prob`` averages the distributions and then decides, which is steadier than
    voting because a window that is genuinely ambiguous contributes its uncertainty
    instead of a hard wrong answer.
    """
    if agg == "mean_prob":
        return int(probs.mean(axis=0).argmax())
    per_window = probs.argmax(axis=1)
    if agg == "median":
        return int(np.median(per_window))
    if agg == "mode":
        values, counts = np.unique(per_window, return_counts=True)
        return int(values[counts.argmax()])
    raise ValueError(f"unknown aggregation: {agg!r}")


def separate_long(model: Any, audio: np.ndarray, device: Any, *,
                  win_seconds: float = WIN_SECONDS, hop_seconds: float = HOP_SECONDS,
                  batch_size: int = 8, agg: str = "mean_prob", sr: int = SR) -> dict:
    """Separate and count audio of any length. Returns est, per-window probabilities, class.

    ``est`` comes back at ``len(audio)`` samples, one row per model slot, in the same
    units as the input windows (each window is RMS-normalised, so the output scale is
    arbitrary -- every metric downstream is scale-invariant).
    """
    import torch

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    win = max(1, int(round(win_seconds * sr)))
    hop = max(1, int(round(hop_seconds * sr)))
    overlap = max(0, win - hop)

    padded = audio if len(audio) >= win else np.pad(audio, (0, win - len(audio)))
    starts = window_starts(len(padded), win, hop)

    n_slots = int(model.n_slots)
    accumulator = np.zeros((n_slots, len(padded)), dtype=np.float64)
    weights = np.zeros(len(padded), dtype=np.float64)
    # A rectangular window would leave a step at every hop; Hann sums to a constant at
    # 50 % overlap, so the division by `weights` below is exact in the interior.
    fade = np.hanning(win).astype(np.float64) if len(starts) > 1 else np.ones(win)

    logits_all: list[np.ndarray] = []
    previous_tail: np.ndarray | None = None

    with torch.no_grad():
        for begin in range(0, len(starts), batch_size):
            chunk = starts[begin:begin + batch_size]
            block = np.stack([rms_normalize(padded[s:s + win]) for s in chunk])
            out = model(torch.from_numpy(block.astype(np.float32)).to(device))
            estimates = out["est"].float().cpu().numpy()
            logits_all.append(out["count_logits"].float().cpu().numpy())

            for k, start in enumerate(chunk):
                est = estimates[k]
                if previous_tail is not None and overlap > 0:
                    est = est[align_permutation(previous_tail, est[:, :overlap])]
                accumulator[:, start:start + win] += est * fade[None, :]
                weights[start:start + win] += fade
                previous_tail = est[:, -overlap:] if overlap > 0 else None

    separated = (accumulator / np.maximum(weights, 1e-8)[None, :])[:, :len(audio)]

    logits = np.concatenate(logits_all, axis=0)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

    return {"est": separated, "probs": probs, "cls": pool_counts(probs, agg),
            "n_windows": len(starts), "win": win, "hop": hop}
