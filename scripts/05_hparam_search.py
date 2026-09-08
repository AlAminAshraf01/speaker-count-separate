#!/usr/bin/env python3
"""Phase 3 -- structured hyperparameter search with speaker-disjoint K-fold.

    python scripts/05_hparam_search.py --config configs/search.yaml \
        --store /kaggle/input/csnet-store --out search

Resumable by design: ``search_results.json`` is rewritten after **every** (config, fold)
run, and a restart skips whatever is already in it. Kill this notebook whenever you like.

On the folds. Train and dev use disjoint LibriSpeech speakers already, so a validation
speaker is never "seen" in the first place. What K-fold buys here is a **variance
estimate**: three disjoint speaker groups give mean +/- std, which is the difference
between "config A beats config B" and "config A got a lucky fold". Report both.

On the budget. Every extra grid point costs (folds x steps) of GPU time. The defaults in
``configs/search.yaml`` are sized for the free tier on purpose; the script refuses a grid
bigger than ``search.max_points`` without ``--allow_big``. A stated budget is defensible,
a silently truncated search is not.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import torch

from _common import add_common_args, banner, build_store_and_bank, require_store, resolve


def build_points(space: dict, strategy: str, n_random: int, seed: int) -> list[dict]:
    """Cartesian product, or a random sample of it."""
    keys = sorted(space)
    values = [list(space[k]) for k in keys]
    points = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    if strategy == "random" and len(points) > n_random:
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(points), size=int(n_random), replace=False)
        points = [points[int(i)] for i in sorted(picks)]
    return points


def point_id(point: dict) -> str:
    """Stable short name for a grid point."""
    return ",".join(f"{k.split('.')[-1]}={v}" for k, v in sorted(point.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--config", default="configs/search.yaml")
    ap.add_argument("--set", nargs="*", default=None, metavar="KEY=VALUE")
    ap.add_argument("--out", default="search")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--val_per_class", type=int, default=60,
                    help="validation mixtures per N, per fold")
    ap.add_argument("--time_budget_h", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--allow_big", action="store_true")
    args = ap.parse_args()

    from csnet.checkpoint import TimeBudget, kaggle_resume_instructions, print_resume_banner
    from csnet.config import apply_overrides, dict_to_cfg, load_yaml, save_cfg, seg_len
    from csnet.datasets import DynamicMixDataset, FrozenMixDataset, build_loader
    from csnet.engine import build_optimizer, evaluate, train_one_epoch
    from csnet.losses import RectangularPITLoss
    from csnet.mixing import sample_recipe, write_recipes
    from csnet.model import build_model
    from csnet.utils import (format_table, human_time, json_dump_atomic, json_load,
                             pick_device, seed_everything)

    config_path = resolve(args.config) or args.config
    base_data = load_yaml(config_path)
    search = dict(base_data.pop("search", {}) or {})
    base_data = apply_overrides(base_data, args.set)

    folds = int(args.folds or search.get("folds", 3))
    steps = int(args.steps or search.get("steps", 1500))
    eval_batches = int(search.get("eval_batches", 40))
    seed = int(search.get("seed", 72))
    strategy = str(search.get("strategy", "grid"))
    max_points = int(search.get("max_points", 8))

    points = build_points(search.get("grid", {}), strategy,
                          int(search.get("n_random", 8)), seed)
    if not points:
        raise SystemExit("configs/search.yaml has an empty `search.grid`")
    if len(points) > max_points and not args.allow_big:
        raise SystemExit(
            f"the grid has {len(points)} points but search.max_points is {max_points}.\n"
            f"  {len(points)} configs x {folds} folds x {steps} steps will not finish on a\n"
            "  free-tier weekly quota. Shrink the grid, raise max_points, or pass --allow_big.")

    store_root = require_store(args)
    out_dir = resolve(args.out) or args.out
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "search_results.json")
    device = pick_device(None if args.device == "auto" else args.device)

    banner("05 - hyperparameter search")
    print(f"config  : {config_path}")
    print(f"grid    : {len(points)} points x {folds} speaker-disjoint folds x {steps} steps")
    print(f"device  : {device}")
    for i, point in enumerate(points):
        print(f"  [{i}] {point_id(point)}")

    base_cfg = dict_to_cfg(base_data)
    budget = TimeBudget(args.time_budget_h if args.time_budget_h is not None
                        else base_cfg.train.time_budget_h)

    # ------------------------------------------------------------ folds
    train_store, train_bank = build_store_and_bank(
        store_root, base_cfg.data.train_split, mmap=base_cfg.data.mmap,
        noise_store=base_cfg.data.noise_store, noise_kinds=base_cfg.data.noise_kinds)
    dev_store, dev_bank = build_store_and_bank(
        store_root, base_cfg.data.dev_split, mmap=base_cfg.data.mmap,
        noise_store=base_cfg.data.noise_store, noise_kinds=base_cfg.data.noise_kinds)

    speakers = list(dev_store.speakers)
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)
    fold_speakers = [speakers[k::folds] for k in range(folds)]
    length = seg_len(base_cfg)
    n_list = [n for n in base_cfg.data.n_list]

    print(f"\ndev speaker pool: {len(speakers)} speakers -> "
          f"{[len(f) for f in fold_speakers]} per fold")

    fold_paths = []
    for k, group in enumerate(fold_speakers):
        path = os.path.join(out_dir, f"recipes_fold{k}.csv")
        usable = [n for n in n_list if n <= len(group)]
        if usable != n_list:
            print(f"  fold {k}: only {len(group)} speakers, restricting N to {usable}")
        if not os.path.exists(path):
            view = dev_store.subset_speakers(group)
            fold_rng = np.random.default_rng([seed, k])
            rows = [sample_recipe(view, dev_bank, n, fold_rng, seg_len=length,
                                  gain_db_range=tuple(base_cfg.data.gain_db_range),
                                  snr_db_range=tuple(base_cfg.data.snr_db_range),
                                  p_clean=base_cfg.data.p_clean,
                                  mix_id=f"fold{k}_n{n}_{i:04d}")
                    for n in usable for i in range(int(args.val_per_class))]
            write_recipes(rows, path)
        fold_paths.append(path)

    # ------------------------------------------------------------ resume
    results: list[dict] = json_load(results_path) if os.path.exists(results_path) else []
    done = {(r["point"], r["fold"]) for r in results}
    if done:
        print(f"\nresuming: {len(done)} of {len(points) * folds} runs already finished")

    todo = [(i, k) for i in range(len(points)) for k in range(folds)
            if (point_id(points[i]), k) not in done]
    print(f"remaining runs: {len(todo)}")

    # ------------------------------------------------------------ the search
    stopped = False
    for i, k in todo:
        if budget.expired(margin_min=15.0):
            print("\ntime budget reached -- stopping between runs (nothing is lost)")
            stopped = True
            break
        point = points[i]
        name = point_id(point)
        banner(f"[{len(results) + 1}/{len(points) * folds}] {name}  fold {k}")

        cfg = dict_to_cfg(apply_overrides(dict(base_data),
                                          [f"{key}={value}" for key, value in point.items()]))
        cfg.data.store_root = store_root
        seed_everything(seed + 1000 * i + k)

        train_set = DynamicMixDataset(
            train_store, train_bank, n_list=cfg.data.n_list,
            steps=steps * cfg.train.batch_size, seg_len=length, seed=seed + k,
            max_n_src=cfg.model.max_n_src,
            gain_db_range=tuple(cfg.data.gain_db_range),
            snr_db_range=tuple(cfg.data.snr_db_range), p_clean=cfg.data.p_clean)
        val_set = FrozenMixDataset(dev_store, dev_bank, fold_paths[k], seg_len=length,
                                   max_n_src=cfg.model.max_n_src)
        train_loader = build_loader(train_set, batch_size=cfg.train.batch_size, shuffle=False,
                                    num_workers=cfg.train.num_workers, drop_last=True)
        val_loader = build_loader(val_set, batch_size=cfg.train.batch_size, shuffle=False,
                                  num_workers=0)

        model = build_model(cfg.model).to(device)
        loss_fn = RectangularPITLoss(
            max_n_src=cfg.model.max_n_src, predict_noise=cfg.model.predict_noise,
            n_classes=cfg.model.n_classes, w_sep=cfg.loss.w_sep, w_sil=cfg.loss.w_sil,
            w_count=cfg.loss.w_count, w_noise=cfg.loss.w_noise,
            silence_db=cfg.loss.silence_db, label_smoothing=cfg.loss.label_smoothing,
            clamp_si_sdr=cfg.loss.clamp_si_sdr).to(device)
        optimizer, scheduler = build_optimizer(model, cfg)
        use_amp = bool(cfg.train.amp) and device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

        print(f"{model.describe()}")
        start = time.time()
        train_logs = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device,
                                     scheduler=scheduler, grad_clip=cfg.train.grad_clip,
                                     log_every=0, budget=budget, amp=use_amp,
                                     max_steps=steps, progress=True)
        val_logs = evaluate(model, val_loader, loss_fn, device, amp=use_amp,
                            max_batches=eval_batches, max_n_src=cfg.model.max_n_src,
                            n_list=cfg.data.n_list)
        elapsed = time.time() - start

        record = {
            "point": name, "params": point, "fold": k,
            "steps": int(train_logs["steps"]), "seconds": round(elapsed, 1),
            "n_parameters": int(model.count_params()),
            "train_loss": train_logs.get("loss"), "train_sisdr": train_logs.get("sisdr"),
            "val_loss": val_logs.get("loss"), "val_sisdr": val_logs.get("sisdr"),
            "val_count_acc": val_logs.get("count_acc"),
            "val_p_si_snr": val_logs.get("p_si_snr"),
            "val_sisdri_count_correct": val_logs.get("sisdri_count_correct"),
            "per_n": val_logs.get("per_n", {}),
        }
        results.append(record)
        json_dump_atomic(results, results_path)   # written after EVERY run
        print(f"  P-SI-SNR {record['val_p_si_snr']:7.2f} dB | count "
              f"{record['val_count_acc'] * 100:5.1f} % | sisdr "
              f"{record['val_sisdr']:6.2f} dB | {human_time(elapsed)}")
        if train_logs["stopped_early"]:
            stopped = True
            break

    # ------------------------------------------------------------ ranking
    banner("ranking")
    summary = []
    for point in points:
        name = point_id(point)
        runs = [r for r in results if r["point"] == name]
        if not runs:
            continue
        scores = [r["val_p_si_snr"] for r in runs]
        counts = [r["val_count_acc"] for r in runs]
        summary.append({
            "point": name, "params": point, "folds": len(runs),
            "p_si_snr_mean": float(np.mean(scores)), "p_si_snr_std": float(np.std(scores)),
            "count_acc_mean": float(np.mean(counts)),
            "n_parameters": runs[0]["n_parameters"],
            "seconds_per_run": float(np.mean([r["seconds"] for r in runs])),
        })
    summary.sort(key=lambda s: -s["p_si_snr_mean"])
    print(format_table(
        [[s["point"], s["folds"], round(s["p_si_snr_mean"], 2), round(s["p_si_snr_std"], 2),
          round(s["count_acc_mean"] * 100, 1), f"{s['n_parameters'] / 1e6:.2f} M",
          human_time(s["seconds_per_run"])] for s in summary],
        ["config", "folds", "P-SI-SNR", "+/- sd", "count %", "params", "per run"]))

    if summary:
        best = summary[0]
        runner_up = summary[1] if len(summary) > 1 else None
        print(f"\nbest: {best['point']}  ({best['p_si_snr_mean']:.2f} "
              f"+/- {best['p_si_snr_std']:.2f} dB)")
        if runner_up:
            gap = best["p_si_snr_mean"] - runner_up["p_si_snr_mean"]
            pooled = max(best["p_si_snr_std"], runner_up["p_si_snr_std"], 1e-6)
            verdict = ("a real difference" if gap > 2 * pooled
                       else "WITHIN FOLD NOISE -- do not claim a winner")
            print(f"gap to runner-up: {gap:.2f} dB against a fold sd of {pooled:.2f} dB "
                  f"-> {verdict}")

        best_cfg = dict_to_cfg(apply_overrides(
            dict(base_data), [f"{k2}={v2}" for k2, v2 in best["params"].items()]))
        best_cfg.name = "csnet-search-best"
        save_cfg(best_cfg, os.path.join(out_dir, "best.yaml"))
        json_dump_atomic({"summary": summary, "runs": results, "folds": folds,
                          "steps": steps, "grid": search.get("grid", {})},
                         os.path.join(out_dir, "search_summary.json"))
        print(f"\nbest config -> {os.path.join(out_dir, 'best.yaml')}")
        print(f"all results -> {results_path}")

    print("\nFor the report: state the budget explicitly -- "
          f"{len(points)} configs x {folds} folds x {steps} steps was chosen to fit the")
    print("free-tier quota, and the fold standard deviation is the yardstick for whether")
    print("any ranking difference is real.")

    if stopped or len(results) < len(points) * folds:
        print_resume_banner(kaggle_resume_instructions(out_dir) + [
            "", "The search itself resumes from search_results.json -- just re-run the",
            "same command and it will skip every (config, fold) pair already finished."])
    return 0


if __name__ == "__main__":
    sys.exit(main())
