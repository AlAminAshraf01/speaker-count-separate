#!/usr/bin/env python3
"""Phase 5b -- open the box. Forward passes only, no training. Protect this time.

    python scripts/07_interpret.py --ckpt /kaggle/working/ckpt/best.pt \
        --store /kaggle/input/csnet-store --recipes data/recipes_test.csv --out interpret

The argument, in one sentence: as N grows the network must partition the **same**
``n_filters`` encoder basis among more sources, so if mask overlap rises and sparsity falls
with N, the degradation curve has a mechanistic explanation computed from the network's own
internals instead of being asserted.

Four deliverables:

1. the learned filterbank -- our "custom spectral transformation", replacing the STFT;
2. mask sparsity / pairwise overlap / entropy as functions of N;
3. does count-head confidence track mask overlap? do miscounts have a geometry signature?
4. filter ablation: which basis functions carry the count?
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from _common import add_common_args, banner, build_store_and_bank, require_store, resolve, use_agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--recipes", default="data/recipes_test.csv")
    ap.add_argument("--out", default="interpret")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=40)
    ap.add_argument("--ablate_steps", nargs="+", type=int, default=[0, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--ablate_batches", type=int, default=12)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from csnet.checkpoint import load_checkpoint
    from csnet.config import dict_to_cfg, seg_len
    from csnet.constants import SR
    from csnet.datasets import FrozenMixDataset, build_loader
    from csnet.engine import evaluate
    from csnet.interpret import (ablate_encoder_filters, collect_mask_records, correlate,
                                 extract_filterbank, filter_centre_frequencies,
                                 filter_importance_by_energy, filter_spectra, mel_reference,
                                 group_by_n)
    from csnet.losses import RectangularPITLoss
    from csnet.model import build_model
    from csnet.utils import format_table, json_dump_atomic, pick_device

    plt = use_agg()
    store_root = require_store(args)
    out_dir = resolve(args.out) or args.out
    os.makedirs(out_dir, exist_ok=True)
    device = pick_device(None if args.device == "auto" else args.device)

    banner("07 - interpretability")
    ckpt_path = resolve(args.ckpt) or args.ckpt
    state = load_checkpoint(ckpt_path, map_location=str(device), restore_rng=False)
    cfg = dict_to_cfg(state.get("cfg", {}))
    model = build_model(cfg.model)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    print(f"checkpoint: {ckpt_path}\n  {model.describe()}")

    store, bank = build_store_and_bank(store_root, args.split, mmap=cfg.data.mmap,
                                       noise_store=cfg.data.noise_store,
                                       noise_kinds=cfg.data.noise_kinds,
                                       noise_weights=cfg.data.noise_weights)
    dataset = FrozenMixDataset(store, bank, resolve(args.recipes) or args.recipes,
                               seg_len=seg_len(cfg), max_n_src=cfg.model.max_n_src)
    loader = build_loader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    report: dict = {"checkpoint": ckpt_path}

    def save(fig, name: str) -> None:
        path = os.path.join(out_dir, name)
        fig.savefig(path)
        plt.close(fig)
        print(f"  wrote {path}")

    # ------------------------------------------------------------ 1. filterbank
    banner("1 - the learned filterbank (phase 2 deliverable)")
    filters = extract_filterbank(model)
    centres = filter_centre_frequencies(filters, SR)
    order = np.argsort(centres)
    spectra = filter_spectra(filters)
    print(f"{filters.shape[0]} filters of {filters.shape[1]} samples "
          f"({1000 * filters.shape[1] / SR:.1f} ms)")
    print(f"centre frequencies: min {centres.min():.0f} Hz, median "
          f"{np.median(centres):.0f} Hz, max {centres.max():.0f} Hz")
    report["filterbank"] = {"n_filters": int(filters.shape[0]),
                            "kernel": int(filters.shape[1]),
                            "centre_hz": centres.tolist()}

    n_show = min(24, filters.shape[0])
    picks = order[np.linspace(0, len(order) - 1, n_show).astype(int)]
    fig, axes = plt.subplots(4, 6, figsize=(11, 5.2))
    for ax, idx in zip(axes.ravel(), picks):
        ax.plot(filters[idx], lw=0.8)
        ax.set_title(f"{centres[idx]:.0f} Hz", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Learned encoder basis functions, sorted by centre frequency "
                 "(this replaces the STFT)")
    save(fig, "filterbank_time.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    freqs = np.linspace(0, SR / 2, spectra.shape[1])
    axes[0].imshow(spectra[order] / (spectra[order].max(axis=1, keepdims=True) + 1e-12),
                   aspect="auto", origin="lower", cmap="magma",
                   extent=[0, SR / 2, 0, filters.shape[0]])
    axes[0].set_xlabel("frequency (Hz)"); axes[0].set_ylabel("filter (sorted)")
    axes[0].set_title("Magnitude responses"); axes[0].grid(False)
    axes[1].plot(np.sort(centres), np.arange(len(centres)), label="learned")
    axes[1].plot(mel_reference(len(centres), SR), np.arange(len(centres)), "--", label="mel")
    axes[1].plot(np.linspace(0, SR / 2, len(centres)), np.arange(len(centres)), ":",
                 label="linear")
    axes[1].set_xlabel("centre frequency (Hz)"); axes[1].set_ylabel("filter rank")
    axes[1].set_title("How the basis tiles the spectrum"); axes[1].legend()
    save(fig, "filterbank_fft.png")

    # ------------------------------------------------------------ 2. mask geometry
    banner("2 - mask geometry as a function of N")
    records = collect_mask_records(model, loader, device, max_batches=args.max_batches,
                                  max_n_src=cfg.model.max_n_src)
    keys = ["sparsity_hoyer", "sparsity_gini", "overlap_cosine", "overlap_iou",
            "entropy", "active_fraction", "confidence"]
    grouped = group_by_n(records, keys)
    rows = [[n] + [round(grouped[n][k], 4) for k in keys] + [grouped[n]["n"]]
            for n in sorted(grouped)]
    print(format_table(rows, ["N"] + keys + ["mixes"]))
    report["mask_stats_by_n"] = grouped

    counts = sorted(grouped)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, key, label in zip(
            axes, ["sparsity_hoyer", "overlap_cosine", "entropy"],
            ["mask sparsity (Hoyer)", "mean pairwise mask overlap (cosine)",
             "mask entropy across slots (nats)"]):
        values = [grouped[n][key] for n in counts]
        ax.plot(counts, values, marker="o")
        ax.set_xlabel("number of speakers N"); ax.set_ylabel(label); ax.set_xticks(counts)
    fig.suptitle("The same encoder basis, partitioned among more sources")
    save(fig, "mask_stats_vs_n.png")

    # ------------------------------------------------------------ 3. the count probe
    banner("3 - does the count head read the mask geometry?")
    corr_overlap = correlate(records, "confidence", "overlap_cosine")
    corr_sparsity = correlate(records, "confidence", "sparsity_hoyer")
    print(f"corr(count confidence, mask overlap ) r = {corr_overlap['pearson_r']:+.3f} "
          f"(p = {corr_overlap['pearson_p']:.3g}, n = {corr_overlap['n']})")
    print(f"corr(count confidence, mask sparsity) r = {corr_sparsity['pearson_r']:+.3f} "
          f"(p = {corr_sparsity['pearson_p']:.3g}, n = {corr_sparsity['n']})")
    report["confidence_correlation"] = {"overlap": corr_overlap, "sparsity": corr_sparsity}

    right = [r for r in records if r["correct"]]
    wrong = [r for r in records if not r["correct"]]
    mis_rows = []
    for key in ("overlap_cosine", "sparsity_hoyer", "entropy", "confidence"):
        get = lambda group, k=key: [g[k] for g in group if np.isfinite(g[k])]
        mis_rows.append([key,
                         round(float(np.mean(get(right))), 4) if get(right) else float("nan"),
                         round(float(np.mean(get(wrong))), 4) if get(wrong) else float("nan")])
    print(f"\nmiscount signature ({len(right)} correct vs {len(wrong)} miscounted):")
    print(format_table(mis_rows, ["statistic", "count correct", "count wrong"]))
    report["miscount_signature"] = mis_rows

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    finite = [r for r in records if np.isfinite(r["overlap_cosine"])]
    if finite:
        axes[0].scatter([r["overlap_cosine"] for r in finite],
                        [r["confidence"] for r in finite],
                        c=[r["n_true"] for r in finite], cmap="viridis", s=12, alpha=0.7)
        axes[0].set_xlabel("mean pairwise mask overlap"); axes[0].set_ylabel("count confidence")
        axes[0].set_title(f"r = {corr_overlap['pearson_r']:+.2f}")
    for label, group in (("count correct", right), ("count wrong", wrong)):
        vals = [g["overlap_cosine"] for g in group if np.isfinite(g["overlap_cosine"])]
        if vals:
            axes[1].hist(vals, bins=25, alpha=0.55, density=True, label=label)
    axes[1].set_xlabel("mean pairwise mask overlap"); axes[1].set_ylabel("density")
    axes[1].set_title("Do miscounts have a mask-geometry signature?"); axes[1].legend()
    save(fig, "conf_vs_overlap.png")

    # ------------------------------------------------------------ 4. ablation
    banner("4 - which basis functions carry the count?")
    importance = filter_importance_by_energy(model, loader, device,
                                             max_batches=args.ablate_batches)
    ranked = np.argsort(-importance)
    loss_fn = RectangularPITLoss(
        max_n_src=cfg.model.max_n_src, predict_noise=cfg.model.predict_noise,
        n_classes=cfg.model.n_classes, w_sep=cfg.loss.w_sep, w_sil=cfg.loss.w_sil,
        w_count=cfg.loss.w_count, w_noise=cfg.loss.w_noise,
        silence_db=cfg.loss.silence_db, label_smoothing=cfg.loss.label_smoothing,
        clamp_si_sdr=cfg.loss.clamp_si_sdr).to(device)

    ablation_rows = []
    for k in args.ablate_steps:
        k = min(int(k), filters.shape[0] - 1)
        with ablate_encoder_filters(model, ranked[:k].tolist()):
            res = evaluate(model, loader, loss_fn, device, amp=False,
                           max_batches=args.ablate_batches, max_n_src=cfg.model.max_n_src,
                           n_list=cfg.data.n_list)
        ablation_rows.append([k, round(100.0 * k / filters.shape[0], 1),
                              round(res["count_acc"] * 100, 2),
                              round(res["p_si_snr"], 2),
                              round(res["sisdri_count_correct"], 2)])
        print(f"  zeroed top {k:4d} filters ({100.0 * k / filters.shape[0]:5.1f} %): "
              f"count {res['count_acc'] * 100:5.2f} %   P-SI-SNR {res['p_si_snr']:7.2f} dB")
    print()
    print(format_table(ablation_rows, ["filters zeroed", "% of basis", "count acc %",
                                       "P-SI-SNR", "SI-SDRi(cc)"]))
    report["ablation"] = ablation_rows

    fig, ax = plt.subplots(figsize=(6, 3.6))
    xs = [r[1] for r in ablation_rows]
    ax.plot(xs, [r[2] for r in ablation_rows], marker="o", label="counting accuracy (%)")
    ax2 = ax.twinx()
    ax2.plot(xs, [r[3] for r in ablation_rows], marker="s", color="tab:red",
             label="P-SI-SNR (dB)")
    ax2.grid(False)
    ax.set_xlabel("% of encoder basis zeroed (most active first)")
    ax.set_ylabel("counting accuracy (%)"); ax2.set_ylabel("P-SI-SNR (dB)")
    ax.set_title("Attribution: ablate the basis, watch the count decision degrade")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)
    save(fig, "filter_ablation.png")

    json_dump_atomic(report, os.path.join(out_dir, "interpret_report.json"))
    banner("what to write in the report")
    print("* The encoder is not an STFT: it learns a non-uniform basis (compare the")
    print("  learned curve against mel and linear in filterbank_fft.png).")
    print("* Mask overlap and sparsity as functions of N give the degradation curve a")
    print("  mechanism instead of an assertion.")
    print("* The count head is an interpretability probe, not a second project: it reads")
    print("  the same shared features whose geometry we just measured.")
    print(f"\nreport -> {os.path.join(out_dir, 'interpret_report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
