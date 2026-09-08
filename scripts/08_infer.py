#!/usr/bin/env python3
"""The product: any audio file in, "there are K people" + K clean speech tracks out.

    python scripts/08_infer.py --ckpt ckpt/best.pt --input meeting.mp3 --out separated/

Handles any sample rate and any length. Long files are processed in overlapping windows;
the tricky part is that a separator has no idea slot 2 in window 7 is the same person as
slot 4 in window 8, so windows are **permutation-aligned** against the previous window's
overlap region before being cross-faded together. Without that, speakers swap tracks every
few seconds.

The per-window counts are aggregated (median by default). Note the honest caveat: the
count head was trained and validated on 3-second fully-overlapped crops, so this
long-file aggregation is a demonstration, not a validated result.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from _common import banner, resolve


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
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    rows, cols = linear_sum_assignment(-np.abs(a @ b.T))
    mapping = list(range(current.shape[0]))
    for r, c in zip(rows, cols):
        mapping[int(r)] = int(c)
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="any audio file, any sample rate")
    ap.add_argument("--out", default="separated")
    ap.add_argument("--win", type=float, default=3.0, help="window length, seconds")
    ap.add_argument("--hop", type=float, default=1.5, help="window hop, seconds")
    ap.add_argument("--agg", default="mean_logit", choices=["mean_logit", "median", "majority"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--keep_noise", action="store_true", default=True)
    ap.add_argument("--no_noise", dest="keep_noise", action="store_false")
    args = ap.parse_args()

    from csnet.audio import read_wav, rms_normalize, write_wav
    from csnet.checkpoint import load_checkpoint
    from csnet.config import dict_to_cfg
    from csnet.constants import SR, class_to_n
    from csnet.model import build_model
    from csnet.utils import format_table, json_dump_atomic, pick_device

    device = pick_device(None if args.device == "auto" else args.device)
    out_dir = resolve(args.out) or args.out
    os.makedirs(out_dir, exist_ok=True)

    banner("08 - inference")
    ckpt_path = resolve(args.ckpt) or args.ckpt
    state = load_checkpoint(ckpt_path, map_location=str(device), restore_rng=False)
    cfg = dict_to_cfg(state.get("cfg", {}))
    model = build_model(cfg.model)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    audio, sr = read_wav(resolve(args.input) or args.input, sr=SR)
    duration = len(audio) / SR
    print(f"input : {args.input}  ({duration:.2f} s at {SR} Hz after resampling)")
    print(f"model : {model.describe()}")

    win = int(args.win * SR)
    hop = max(1, int(args.hop * SR))
    overlap = max(0, win - hop)
    padded = audio if len(audio) >= win else np.pad(audio, (0, win - len(audio)))
    starts = window_starts(len(padded), win, hop)

    n_slots = model.n_slots
    n_speaker_slots = cfg.model.max_n_src
    accumulator = np.zeros((n_slots, len(padded)), dtype=np.float64)
    weights = np.zeros(len(padded), dtype=np.float64)
    fade = np.hanning(win).astype(np.float64) if len(starts) > 1 else np.ones(win)

    logits_all: list[np.ndarray] = []
    previous_tail: np.ndarray | None = None

    with torch.no_grad():
        for batch_start in range(0, len(starts), args.batch_size):
            chunk = starts[batch_start:batch_start + args.batch_size]
            block = np.stack([rms_normalize(padded[s:s + win]) for s in chunk])
            out = model(torch.from_numpy(block).to(device))
            estimates = out["est"].float().cpu().numpy()
            logits_all.append(out["count_logits"].float().cpu().numpy())

            for k, start in enumerate(chunk):
                est = estimates[k]
                if previous_tail is not None and overlap > 0:
                    mapping = align_permutation(previous_tail, est[:, :overlap])
                    est = est[mapping]
                accumulator[:, start:start + win] += est * fade[None, :]
                weights[start:start + win] += fade
                previous_tail = est[:, -overlap:] if overlap > 0 else None

    weights = np.maximum(weights, 1e-8)
    separated = (accumulator / weights[None, :])[:, :len(audio)]

    logits = np.concatenate(logits_all, axis=0)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    per_window = np.array([class_to_n(int(c)) for c in probs.argmax(axis=1)])

    if args.agg == "mean_logit":
        n_speakers = int(class_to_n(int(probs.mean(axis=0).argmax())))
    elif args.agg == "median":
        n_speakers = int(np.median(per_window))
    else:
        values, counts = np.unique(per_window, return_counts=True)
        n_speakers = int(values[counts.argmax()])
    confidence = float(probs.mean(axis=0).max())

    banner(f"DETECTED {n_speakers} SPEAKER{'S' if n_speakers != 1 else ''}"
           f"   (confidence {confidence * 100:.1f} %)")
    dist = probs.mean(axis=0)
    print(format_table([[class_to_n(i), round(float(dist[i]) * 100, 1),
                         int((per_window == class_to_n(i)).sum())]
                        for i in range(len(dist))],
                       ["N", "mean prob %", "windows voting"]))
    print(f"\n{len(starts)} windows of {args.win:g} s, hop {args.hop:g} s, "
          f"aggregation '{args.agg}'")

    # Rank speaker slots by energy and keep the n_speakers loudest.
    energies = (separated[:n_speaker_slots] ** 2).mean(axis=1)
    order = np.argsort(-energies)[:n_speakers]
    peak = float(np.abs(audio).max()) + 1e-9

    written = []
    for rank, slot in enumerate(order, start=1):
        track = separated[int(slot)]
        track = track / (np.abs(track).max() + 1e-9) * min(0.95, peak * 2.0)
        path = os.path.join(out_dir, f"speaker{rank}.wav")
        write_wav(path, track.astype(np.float32), SR)
        written.append({"file": os.path.basename(path), "slot": int(slot),
                        "energy_db": float(10 * np.log10(energies[int(slot)] + 1e-12))})
    if args.keep_noise and model.noise_slot is not None:
        track = separated[model.noise_slot]
        track = track / (np.abs(track).max() + 1e-9) * 0.9
        path = os.path.join(out_dir, "noise.wav")
        write_wav(path, track.astype(np.float32), SR)
        written.append({"file": "noise.wav", "slot": int(model.noise_slot),
                        "energy_db": float(10 * np.log10(
                            (separated[model.noise_slot] ** 2).mean() + 1e-12))})

    summary = {
        "input": os.path.abspath(resolve(args.input) or args.input),
        "checkpoint": ckpt_path,
        "duration_seconds": duration,
        "n_speakers": n_speakers,
        "confidence": confidence,
        "count_distribution": {int(class_to_n(i)): float(dist[i]) for i in range(len(dist))},
        "per_window_counts": per_window.tolist(),
        "window_seconds": args.win, "hop_seconds": args.hop, "aggregation": args.agg,
        "outputs": written,
    }
    json_dump_atomic(summary, os.path.join(out_dir, "summary.json"))

    print(f"\nwrote {len(written)} files to {out_dir}:")
    for item in written:
        print(f"  {item['file']:<14s} slot {item['slot']}  "
              f"{item['energy_db']:+6.1f} dB")
    print(f"  summary.json")
    print("\nCaveat for the report: the count head was trained on 3 s fully-overlapped")
    print("crops. Aggregating window votes over a long, sparsely-overlapped recording is")
    print("a demonstration, not a validated result -- real conversation is a diarisation")
    print("problem, not a spectral-density judgement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
