#!/usr/bin/env python3
"""Phase 5a -- the benchmark. Frozen test set, four numbers, every baseline.

    python scripts/06_evaluate.py --ckpt /kaggle/working/ckpt/best.pt \
        --store /kaggle/input/csnet-store --recipes data/recipes_test.csv --out eval

Writes ``eval_report.md`` (paste it straight into the report) and ``eval_report.json``.

Reports, per the protocol fixed before training:

1. counting accuracy **and the full confusion matrix** -- the classes are ordinal;
2. P-SI-SNR over the whole test set -- honest end to end, defined when the count is wrong;
3. SI-SDRi per N on the **count-correct subset only** -- comparable to fixed-N literature;
4. the naive-predictor floor, the IRM/IBM oracle ceiling, and the 0 dB mixture floor.

``--libri2mix_dir`` additionally evaluates the untouched official Libri2Mix test set at
N=2. That is the only literature-comparable number in this project (target: within ~1 dB
of 14.76 dB SI-SDRi); everything at N>2 uses our own mixtures.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from _common import add_common_args, banner, build_store_and_bank, require_store, resolve, use_agg


def evaluate_official_libri2mix(model, libri_dir, split, device, limit, max_seconds,
                                max_n_src):
    """Gate 2: the official, unmodified Libri2Mix test set at a known N = 2."""
    from csnet.audio import read_wav, rms_normalize
    from csnet.constants import SR, n_to_class
    from csnet.metrics import matched_si_sdri

    mix_dir = os.path.join(libri_dir, split, "mix_clean")
    if not os.path.isdir(mix_dir):
        return None
    names = sorted(f for f in os.listdir(mix_dir) if f.lower().endswith(".wav"))[:limit]
    max_len = int(max_seconds * SR) if max_seconds else None

    sisdri, correct = [], []
    for name in names:
        mix, _ = read_wav(os.path.join(mix_dir, name), sr=SR)
        sources = []
        for k in (1, 2):
            path = os.path.join(libri_dir, split, f"s{k}", name)
            if not os.path.exists(path):
                sources = []
                break
            wave, _ = read_wav(path, sr=SR)
            sources.append(wave)
        if not sources:
            continue
        length = min([len(mix)] + [len(s) for s in sources])
        if max_len:
            length = min(length, max_len)
        mix, sources = mix[:length], np.stack([s[:length] for s in sources])

        tensor = torch.from_numpy(rms_normalize(mix)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
        est = out["est"][0].float().cpu().numpy()
        pred = int(out["count_logits"][0].argmax().item())
        correct.append(int(pred == n_to_class(2)))
        sisdri.append(float(np.mean(matched_si_sdri(est[:max_n_src], sources, mix, 2))))

    if not sisdri:
        return None
    return {"n_files": len(sisdri), "si_sdri": float(np.mean(sisdri)),
            "si_sdri_std": float(np.std(sisdri)), "count_acc": float(np.mean(correct)),
            "reference_asteroid": 14.76}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--recipes", default="data/recipes_test.csv")
    ap.add_argument("--out", default="eval")
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="cap the number of test mixtures")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--oracle_limit", type=int, default=200,
                    help="mixtures used for the (slow) IRM/IBM oracle rows")
    ap.add_argument("--libri2mix_dir", default=None,
                    help="official .../Libri2Mix/wav8k/min for the literature-comparable N=2 row")
    ap.add_argument("--libri2mix_limit", type=int, default=300)
    ap.add_argument("--libri2mix_max_seconds", type=float, default=10.0)
    args = ap.parse_args()

    from csnet.baselines import (PUBLISHED_REFERENCE, ideal_ratio_mask_sisdri,
                                 features_to_matrix, naive_count_features,
                                 naive_count_baseline, ALL_FEATURES)
    from csnet.checkpoint import load_checkpoint
    from csnet.config import dict_to_cfg, seg_len
    from csnet.constants import SR
    from csnet.datasets import FrozenMixDataset, build_loader
    from csnet.engine import evaluate
    from csnet.losses import RectangularPITLoss
    from csnet.metrics import count_report, format_confusion
    from csnet.mixing import read_recipes, render_recipe
    from csnet.model import build_model
    from csnet.utils import format_table, json_dump_atomic, pick_device

    plt = use_agg()
    store_root = require_store(args)
    out_dir = resolve(args.out) or args.out
    os.makedirs(out_dir, exist_ok=True)
    device = pick_device(None if args.device == "auto" else args.device)

    banner("06 - evaluate")
    ckpt_path = resolve(args.ckpt) or args.ckpt
    state = load_checkpoint(ckpt_path, map_location=str(device), restore_rng=False)
    cfg = dict_to_cfg(state.get("cfg", {}))
    length = seg_len(cfg)
    print(f"checkpoint: {ckpt_path}")
    print(f"  epoch {state.get('epoch')}, step {state.get('global_step')}, "
          f"best {state.get('best_metric'):.3f}, {state.get('wall_h', 0):.2f} h trained")

    model = build_model(cfg.model)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    print(f"  {model.describe()}")

    recipes_path = resolve(args.recipes) or args.recipes
    store, bank = build_store_and_bank(store_root, args.split, mmap=cfg.data.mmap,
                                       noise_store=cfg.data.noise_store,
                                       noise_kinds=cfg.data.noise_kinds,
                                       noise_weights=cfg.data.noise_weights)
    dataset = FrozenMixDataset(store, bank, recipes_path, seg_len=length,
                               max_n_src=cfg.model.max_n_src, limit=args.limit)
    loader = build_loader(dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers)
    print(f"test set  : {len(dataset)} frozen mixtures {dataset.counts_per_n()}")

    loss_fn = RectangularPITLoss(
        max_n_src=cfg.model.max_n_src, predict_noise=cfg.model.predict_noise,
        n_classes=cfg.model.n_classes, w_sep=cfg.loss.w_sep, w_sil=cfg.loss.w_sil,
        w_count=cfg.loss.w_count, w_noise=cfg.loss.w_noise,
        silence_db=cfg.loss.silence_db, label_smoothing=cfg.loss.label_smoothing,
        clamp_si_sdr=cfg.loss.clamp_si_sdr).to(device)

    results = evaluate(model, loader, loss_fn, device, amp=(device.type == "cuda"),
                       collect=True, max_n_src=cfg.model.max_n_src,
                       n_list=cfg.data.n_list, progress=True)
    records = results["records"]
    report: dict = {"checkpoint": ckpt_path, "n_test": len(records),
                    "config": state.get("cfg", {})}

    # ---------------------------------------------------------- 1. counting
    banner("1 - counting")
    counting = count_report(results["y_true"], results["y_pred"], cfg.data.n_list)
    print(f"accuracy {counting['accuracy'] * 100:.2f} %   MAE {counting['mae']:.3f}\n")
    print(format_confusion(counting["confusion"], counting["labels"]))
    print("\nrow-normalised (%):")
    print(format_confusion(counting["confusion"], counting["labels"], normalise=True))
    report["counting"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in counting.items()}

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    cm = counting["confusion"].astype(float)
    norm = cm / np.maximum(1, cm.sum(axis=1, keepdims=True))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(counting["labels"])), counting["labels"])
    ax.set_yticks(range(len(counting["labels"])), counting["labels"])
    ax.set_xlabel("predicted N"); ax.set_ylabel("true N"); ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center",
                    fontsize=8, color="w" if norm[i, j] > 0.5 else "k")
    ax.set_title(f"Speaker counting: {counting['accuracy'] * 100:.1f} % accurate")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(os.path.join(out_dir, "confusion.png")); plt.close(fig)

    # ---------------------------------------------------------- 2 & 3. separation
    banner("2 & 3 - separation")
    print(f"P-SI-SNR over the whole test set : {results['p_si_snr']:8.2f} dB")
    print(f"SI-SDRi on count-correct subset  : {results['sisdri_count_correct']:8.2f} dB")
    print(f"input SI-SDR (unprocessed)       : {results['input_si_sdr']:8.2f} dB\n")

    per_n_rows = []
    for n, stats in results["per_n"].items():
        per_n_rows.append([n, stats["n"], round(stats["count_acc"] * 100, 1),
                           round(stats["p_si_snr"], 2), stats["n_count_correct"],
                           round(stats["si_sdri_count_correct"], 2),
                           round(stats["input_si_sdr"], 2)])
    per_n_header = ["N", "mixes", "count %", "P-SI-SNR", "n correct", "SI-SDRi(cc)", "input SI-SDR"]
    print(format_table(per_n_rows, per_n_header))
    report["per_n"] = results["per_n"]
    report["overall"] = {k: results[k] for k in
                         ("p_si_snr", "sisdri_count_correct", "count_acc", "input_si_sdr", "loss")}

    # ---------------------------------------------------------- clean vs noisy
    banner("noise robustness")
    noise_rows = []
    for label, subset in (("clean", [r for r in records if not r["is_noisy"]]),
                          ("noisy", [r for r in records if r["is_noisy"]])):
        if not subset:
            continue
        correct = [r for r in subset if r["si_sdri"] is not None]
        noise_rows.append([label, len(subset),
                           round(100.0 * np.mean([r["n_pred"] == r["n_true"] for r in subset]), 1),
                           round(float(np.mean([r["p_si_snr"] for r in subset])), 2),
                           round(float(np.mean([np.mean(r["si_sdri"]) for r in correct])), 2)
                           if correct else float("nan")])
    noisy = [r for r in records if r["is_noisy"]]
    if noisy:
        edges = [0, 5, 10, 15, 21]
        for lo, hi in zip(edges[:-1], edges[1:]):
            subset = [r for r in noisy if lo <= r["snr_db"] < hi]
            if not subset:
                continue
            correct = [r for r in subset if r["si_sdri"] is not None]
            noise_rows.append([f"SNR {lo}-{hi} dB", len(subset),
                               round(100.0 * np.mean([r["n_pred"] == r["n_true"] for r in subset]), 1),
                               round(float(np.mean([r["p_si_snr"] for r in subset])), 2),
                               round(float(np.mean([np.mean(r["si_sdri"]) for r in correct])), 2)
                               if correct else float("nan")])
    print(format_table(noise_rows, ["subset", "mixes", "count %", "P-SI-SNR", "SI-SDRi(cc)"]))
    report["noise_breakdown"] = noise_rows

    # ---------------------------------------------------------- 4. baselines
    banner("4 - baselines")
    recipes = read_recipes(recipes_path)
    rng = np.random.default_rng(0)
    picks = rng.choice(len(recipes), size=min(args.oracle_limit, len(recipes)), replace=False)
    oracle: dict[str, dict[int, list]] = {"irm": {}, "ibm": {}}
    feature_rows, feature_labels = [], []
    for i in picks:
        recipe = recipes[int(i)]
        rendered = render_recipe(recipe, store, bank, seg_len=length)
        n = rendered["n_src"]
        feature_rows.append(naive_count_features(rendered["mix"]))
        feature_labels.append(n)
        if n > 1:
            for mode in ("irm", "ibm"):
                value = float(np.mean(ideal_ratio_mask_sisdri(
                    rendered["mix"], rendered["sources"], mode=mode)))
                oracle[mode].setdefault(n, []).append(value)

    naive_X, naive_keys = features_to_matrix(feature_rows, ALL_FEATURES)
    naive = naive_count_baseline(naive_X, np.array(feature_labels), max_depth=3,
                                 n_folds=5, feature_names=naive_keys)

    counts = sorted(results["per_n"])
    baseline_rows = [
        ["mixture as estimate (floor)"] + ["0.00" for _ in counts],
        ["IBM oracle (upper bound)"] + [
            f"{np.mean(oracle['ibm'][n]):.2f}" if n in oracle["ibm"] else "-" for n in counts],
        ["IRM oracle (upper bound)"] + [
            f"{np.mean(oracle['irm'][n]):.2f}" if n in oracle["irm"] else "-" for n in counts],
        ["ours, count-correct subset"] + [
            f"{results['per_n'][n]['si_sdri_count_correct']:.2f}" for n in counts],
    ]
    for name, values in PUBLISHED_REFERENCE.items():
        baseline_rows.append([name] + [f"{values[n]:.2f}" if n in values else "-" for n in counts])
    print("SI-SDRi (dB) by number of speakers:\n")
    print(format_table(baseline_rows, ["system"] + [f"N={n}" for n in counts]))
    print(f"\nnaive count predictor (depth-3 tree, {naive['n_folds']}-fold CV): "
          f"{naive['accuracy'] * 100:.1f} %  (chance {naive['chance'] * 100:.0f} %)")
    print(f"our count head                                     : "
          f"{counting['accuracy'] * 100:.1f} %")
    report["baselines"] = {
        "oracle_irm": {int(n): float(np.mean(v)) for n, v in oracle["irm"].items()},
        "oracle_ibm": {int(n): float(np.mean(v)) for n, v in oracle["ibm"].items()},
        "naive_count_accuracy": float(naive["accuracy"]),
        "naive_chance": float(naive["chance"]),
        "published": PUBLISHED_REFERENCE,
    }

    # ---------------------------------------------------------- gate 2
    official = None
    if args.libri2mix_dir:
        banner("gate 2 - official Libri2Mix test set (the literature-comparable number)")
        official = evaluate_official_libri2mix(
            model, resolve(args.libri2mix_dir), "test", device, args.libri2mix_limit,
            args.libri2mix_max_seconds, cfg.model.max_n_src)
        if official is None:
            print("could not read the official test set -- check --libri2mix_dir")
        else:
            gap = official["si_sdri"] - official["reference_asteroid"]
            print(f"files          : {official['n_files']}")
            print(f"SI-SDRi        : {official['si_sdri']:.2f} dB "
                  f"(sd {official['si_sdri_std']:.2f})")
            print(f"published      : {official['reference_asteroid']:.2f} dB (asteroid)")
            print(f"gap            : {gap:+.2f} dB   "
                  f"{'PASS (within 1 dB)' if abs(gap) <= 1.0 else 'below target'}")
            print(f"count accuracy : {official['count_acc'] * 100:.1f} % (should predict N=2)")
            report["official_libri2mix"] = official

    # ---------------------------------------------------------- markdown
    md = ["# Evaluation report", "",
          f"Checkpoint `{os.path.basename(ckpt_path)}` - epoch {state.get('epoch')}, "
          f"step {state.get('global_step')}, {state.get('wall_h', 0):.2f} h of training.",
          f"Test set: {len(records)} frozen mixtures from `{os.path.basename(recipes_path)}`.",
          "", "## 1. Counting", "",
          f"Accuracy **{counting['accuracy'] * 100:.2f} %**, MAE {counting['mae']:.3f}, "
          f"naive-predictor floor {naive['accuracy'] * 100:.1f} %, chance "
          f"{naive['chance'] * 100:.0f} %.", "",
          "Confusion matrix (rows = true N, counts):", "", "```",
          format_confusion(counting["confusion"], counting["labels"]), "```", "",
          "## 2 & 3. Separation", "",
          f"P-SI-SNR over the whole test set: **{results['p_si_snr']:.2f} dB**.", "",
          _md_table(per_n_header, per_n_rows), "",
          "## 4. Baselines", "",
          _md_table(["system"] + [f"N={n}" for n in counts], baseline_rows), "",
          "## Noise robustness", "",
          _md_table(["subset", "mixes", "count %", "P-SI-SNR", "SI-SDRi(cc)"], noise_rows), ""]
    if official:
        md += ["## Gate 2 - official Libri2Mix (N=2), literature-comparable", "",
               f"SI-SDRi **{official['si_sdri']:.2f} dB** over {official['n_files']} files "
               f"against a published {official['reference_asteroid']:.2f} dB "
               f"({official['si_sdri'] - official['reference_asteroid']:+.2f} dB).", ""]
    md += ["## Caveats", "",
           "- Only the N=2 row on the official test set is comparable to the literature; "
           "our N>2 mixtures are our own.",
           "- `min`-mode mixtures are fully overlapped, so counting here is one global "
           "judgement about spectral density, not a diarisation result.",
           "- SI-SDRi is reported as an improvement because input SI-SDR itself falls with N.",
           ""]
    md_path = os.path.join(out_dir, "eval_report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    json_dump_atomic(report, os.path.join(out_dir, "eval_report.json"))

    banner("done")
    print(f"markdown -> {md_path}")
    print(f"json     -> {os.path.join(out_dir, 'eval_report.json')}")
    print(f"figure   -> {os.path.join(out_dir, 'confusion.png')}")
    return 0


def _md_table(header, rows) -> str:
    """Render a markdown table."""
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
