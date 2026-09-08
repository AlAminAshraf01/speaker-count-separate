#!/usr/bin/env python3
"""Phase 1 -- exploratory data analysis. CPU only, no GPU quota.

    python scripts/02_eda.py --store /kaggle/working/store --recipes data/recipes_test.csv --out eda

Produces the figures the report's EDA section needs, and one figure that is the whole
argument of the project's data section: mixture level and duration **before and after** the
3 s crop + RMS normalisation. Before, they encode N. After, they cannot.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

from _common import add_common_args, banner, build_store_and_bank, require_store, resolve, use_agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--split", default="test")
    ap.add_argument("--recipes", default="data/recipes_test.csv")
    ap.add_argument("--out", default="eda")
    ap.add_argument("--per_class", type=int, default=200,
                    help="mixtures analysed per speaker count")
    ap.add_argument("--seg_seconds", type=float, default=3.0)
    args = ap.parse_args()

    from csnet.audio import estimate_f0, rms
    from csnet.baselines import naive_count_features
    from csnet.constants import SR
    from csnet.metrics import si_sdr
    from csnet.mixing import read_recipes, render_recipe
    from csnet.pack import SourceStore
    from csnet.utils import format_table, json_dump_atomic

    plt = use_agg()
    store_root = require_store(args)
    out_dir = resolve(args.out) or args.out
    os.makedirs(out_dir, exist_ok=True)
    recipes_path = resolve(args.recipes) or args.recipes
    if not os.path.exists(recipes_path):
        raise SystemExit(f"no recipes at {recipes_path}; run scripts/01_make_frozen_sets.py")

    store, bank = build_store_and_bank(store_root, args.split,
                                       noise_store=resolve(args.noise_store))
    seg_len = int(round(args.seg_seconds * SR))
    report: dict = {"store": store_root, "split": args.split, "recipes": recipes_path}

    banner("02 - EDA")

    # ------------------------------------------------------------ splits table
    rows = []
    for split in ("train-100", "dev", "test"):
        try:
            info = SourceStore(store_root, split).summary()
        except FileNotFoundError:
            continue
        rows.append([split, info["n_utt"], info["n_speakers"], info["n_babble_speakers"],
                     round(info["hours"], 2), round(info["median_utt_seconds"], 2),
                     round(info["median_utts_per_speaker"], 1)])
        report.setdefault("splits", {})[split] = info
    print(format_table(rows, ["split", "utts", "speakers", "babble spk", "hours",
                              "median utt s", "utts/spk"]))

    # ------------------------------------------------------------ render a sample
    all_recipes = read_recipes(recipes_path)
    by_n: dict[int, list] = defaultdict(list)
    for recipe in all_recipes:
        by_n[int(recipe["n_src"])].append(recipe)
    counts = sorted(by_n)
    print(f"\nrendering up to {args.per_class} mixtures per N from {len(all_recipes)} recipes")

    data: dict[int, dict] = {}
    for n in counts:
        subset = by_n[n][: int(args.per_class)]
        corr_means, corr_max, input_sisdr, feats_after = [], [], [], []
        raw_level, raw_dur, f0_spread, same_pitch = [], [], [], []
        for recipe in subset:
            out = render_recipe(recipe, store, bank, seg_len=seg_len)
            sources, mix = out["sources"], out["mix"]

            # cross-source correlation, the N x N matrix the guidelines ask for
            if n > 1:
                normed = sources - sources.mean(axis=1, keepdims=True)
                denom = np.linalg.norm(normed, axis=1, keepdims=True) + 1e-12
                corr = np.abs((normed / denom) @ (normed / denom).T)
                off = corr[~np.eye(n, dtype=bool)]
                corr_means.append(float(off.mean()))
                corr_max.append(float(off.max()))
            input_sisdr.append(float(np.mean([si_sdr(mix, s) for s in sources])))
            feats_after.append(naive_count_features(mix))

            # the pre-mitigation view: sum the sources at their stored gains with no
            # RMS normalisation and no crop, exactly as a naive pipeline would
            full = [store.get(int(i)) for i in recipe["utt_idx"]]
            length = min(len(f) for f in full)
            unnormed = np.sum([f[:length] * (10.0 ** (g / 20.0))
                               for f, g in zip(full, recipe["gain_db"])], axis=0)
            raw_level.append(20.0 * float(np.log10(float(rms(unnormed)) + 1e-12)))
            raw_dur.append(length / SR)

            pitches = [estimate_f0(f) for f in full]
            pitches = [p for p in pitches if p == p]
            if len(pitches) > 1:
                f0_spread.append(float(np.std(pitches)))
                same_pitch.append(int(np.ptp(pitches) < 30.0))

        data[n] = {
            "corr_mean": np.array(corr_means), "corr_max": np.array(corr_max),
            "input_sisdr": np.array(input_sisdr),
            "raw_level_db": np.array(raw_level), "raw_duration_s": np.array(raw_dur),
            "after": feats_after, "f0_spread": np.array(f0_spread),
            "same_pitch_frac": float(np.mean(same_pitch)) if same_pitch else float("nan"),
            "n_mixtures": len(subset),
        }

    # ------------------------------------------------------------ per-N table
    table = []
    for n in counts:
        d = data[n]
        after = d["after"]
        table.append([
            n, d["n_mixtures"],
            round(float(np.mean(d["corr_mean"])), 4) if d["corr_mean"].size else float("nan"),
            round(float(np.mean(d["input_sisdr"])), 2),
            round(float(np.mean(d["raw_level_db"])), 2),
            round(float(np.mean(d["raw_duration_s"])), 2),
            round(float(np.mean([f["crest"] for f in after])), 2),
            round(float(np.mean([f["kurtosis"] for f in after])), 2),
            round(float(np.mean([f["flatness"] for f in after])), 4),
        ])
    header = ["N", "mixes", "mean|corr|", "in SI-SDR", "raw level dB", "raw dur s",
              "crest dB", "kurtosis", "flatness"]
    print("\n" + format_table(table, header))
    report["per_n"] = {int(r[0]): dict(zip(header[1:], r[1:])) for r in table}

    # ------------------------------------------------------------ figures
    def save(fig, name: str) -> None:
        path = os.path.join(out_dir, name)
        fig.savefig(path)
        plt.close(fig)
        print(f"  wrote {path}")

    print("\nfigures:")

    # 1. correlation matrices, one panel per N
    multi = [n for n in counts if n > 1]
    if multi:
        fig, axes = plt.subplots(1, len(multi), figsize=(3.1 * len(multi), 3.0))
        axes = np.atleast_1d(axes)
        for ax, n in zip(axes, multi):
            subset = by_n[n][: min(60, len(by_n[n]))]
            acc = np.zeros((n, n))
            for recipe in subset:
                s = render_recipe(recipe, store, bank, seg_len=seg_len)["sources"]
                z = s - s.mean(axis=1, keepdims=True)
                z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
                acc += np.abs(z @ z.T)
            acc /= len(subset)
            im = ax.imshow(acc, vmin=0, vmax=1, cmap="magma")
            ax.set_title(f"N = {n}")
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xlabel("source"); ax.grid(False)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{acc[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="w" if acc[i, j] < 0.6 else "k")
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="mean |correlation|")
        fig.suptitle("Cross-source correlation within a mixture")
        save(fig, "corr_matrix_per_n.png")

        fig, ax = plt.subplots(figsize=(6, 3.4))
        for n in multi:
            ax.hist(data[n]["corr_max"], bins=40, alpha=0.5, label=f"N = {n}", density=True)
        ax.set_xlabel("max pairwise |correlation| in a mixture")
        ax.set_ylabel("density")
        ax.set_title("Hard mixtures live in the right tail (the outlier class)")
        ax.legend()
        save(fig, "corr_hist.png")

    # 2. input SI-SDR per N
    fig, ax = plt.subplots(figsize=(6, 3.4))
    # set_xticklabels rather than boxplot(labels=...): that kwarg was renamed in
    # matplotlib 3.9 and removed in 3.11, and Kaggle's version is not ours.
    ax.boxplot([data[n]["input_sisdr"] for n in counts], showfliers=False)
    ax.set_xticks(range(1, len(counts) + 1))
    ax.set_xticklabels([str(n) for n in counts])
    ax.set_xlabel("number of speakers N")
    ax.set_ylabel("input SI-SDR (dB)")
    ax.set_title("Input SI-SDR falls with N -- why we report improvement, not raw SI-SDR")
    save(fig, "input_sisdr_per_n.png")

    # 3. THE leak figure: before vs after mitigation
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6))
    for n in counts:
        axes[0, 0].hist(data[n]["raw_level_db"], bins=30, alpha=0.5, label=f"N={n}", density=True)
        axes[0, 1].hist(data[n]["raw_duration_s"], bins=30, alpha=0.5, label=f"N={n}", density=True)
        axes[1, 0].hist([f["rms_db"] for f in data[n]["after"]], bins=30, alpha=0.5,
                        label=f"N={n}", density=True)
        axes[1, 1].hist([f["duration"] for f in data[n]["after"]], bins=30, alpha=0.5,
                        label=f"N={n}", density=True)
    axes[0, 0].set_title("BEFORE: mixture level leaks N"); axes[0, 0].set_xlabel("level (dB)")
    axes[0, 1].set_title("BEFORE: `min`-mode duration leaks N"); axes[0, 1].set_xlabel("seconds")
    axes[1, 0].set_title("AFTER: RMS normalised"); axes[1, 0].set_xlabel("level (dB)")
    axes[1, 1].set_title(f"AFTER: fixed {args.seg_seconds:g} s crop")
    axes[1, 1].set_xlabel("seconds")
    for ax in axes.ravel():
        ax.set_ylabel("density"); ax.legend(fontsize=7)
    fig.suptitle("The count label leaks out of the mixing recipe -- and the mitigation closes it")
    save(fig, "leak_before_after.png")

    # 4. the legitimate density cues
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, key, label in zip(axes, ["crest", "kurtosis", "flatness"],
                              ["crest factor (dB)", "kurtosis", "spectral flatness"]):
        means = [float(np.mean([f[key] for f in data[n]["after"]])) for n in counts]
        stds = [float(np.std([f[key] for f in data[n]["after"]])) for n in counts]
        ax.errorbar(counts, means, yerr=stds, marker="o", capsize=3)
        ax.set_xlabel("N"); ax.set_ylabel(label); ax.set_xticks(counts)
    fig.suptitle("What survives the mitigation: summing N sparse signals makes the mixture "
                 "less sparse")
    save(fig, "density_cues_vs_n.png")

    # 5. outliers: similar-pitch groups
    have_pitch = [n for n in counts if data[n]["f0_spread"].size]
    if have_pitch:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        for n in have_pitch:
            axes[0].hist(data[n]["f0_spread"], bins=30, alpha=0.5, label=f"N={n}", density=True)
        axes[0].set_xlabel("std of estimated F0 across sources (Hz)")
        axes[0].set_ylabel("density"); axes[0].legend(fontsize=7)
        axes[0].set_title("Pitch spread within a mixture")
        fracs = [data[n]["same_pitch_frac"] for n in have_pitch]
        axes[1].bar([str(n) for n in have_pitch], fracs)
        axes[1].set_xlabel("N"); axes[1].set_ylabel("fraction with F0 range < 30 Hz")
        axes[1].set_title("Similar-pitch mixtures: the outlier class")
        save(fig, "outlier_pairs.png")
        report["same_pitch_fraction"] = {int(n): data[n]["same_pitch_frac"] for n in have_pitch}

    json_dump_atomic(report, os.path.join(out_dir, "eda_report.json"))
    banner("interpretation")
    print("* Input SI-SDR falls with N, so every separation number MUST be reported as an")
    print("  improvement over the mixture, per N, never as a single average.")
    print("* Mean cross-source correlation rises with N and the right tail of the max-|corr|")
    print("  histogram is the outlier class -- same-sex / similar-F0 groups.")
    print("* leak_before_after.png is the figure that justifies the whole data design:")
    print("  raw level and raw duration separate the classes; after a fixed crop and RMS")
    print("  normalisation they do not. 03_count_leak_probe.py puts a number on that.")
    print(f"\nreport -> {os.path.join(out_dir, 'eda_report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
