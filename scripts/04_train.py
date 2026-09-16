#!/usr/bin/env python3
"""Stage 3 -- train the joint counter + separator. Resumable, budget-aware.

    python scripts/04_train.py --config configs/paper.yaml \
        --store /kaggle/input/csnet-store --ckpt_dir /kaggle/working/ckpt

What makes this survive Kaggle:

* it times 20 real steps before committing and tells you how many epochs the remaining
  budget actually buys -- measure, do not trust a FLOP table;
* ``last.pt`` is written every ``ckpt_every_steps`` steps and at every epoch boundary,
  atomically, so a hard kill costs minutes;
* when the time budget expires it stops **cleanly and exits 0**, which is what lets Kaggle's
  Save Version capture the checkpoint;
* ``--resume auto`` finds the checkpoint by itself, including in a re-attached dataset from
  the previous session.

Gate 6 of the plan (count head alone, separator frozen) is::

    --set train.freeze_separator=True loss.w_sep=0.0 loss.w_sil=0.0 loss.w_noise=0.0
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

from _common import (banner, build_store_and_bank, code_version, require_store,
                     resolve)


def _open_log(path: str, fields: list[str]):
    """Open train_log.csv in append mode, writing a header only when it is new."""
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    handle = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    if not exists:
        writer.writeheader()
    return handle, writer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=None, metavar="KEY=VALUE")
    ap.add_argument("--store", default=None)
    ap.add_argument("--noise_store", default=None)
    ap.add_argument("--recipes_dev", default=None)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--out", default=None, help="where logs go (default: ckpt_dir)")
    ap.add_argument("--resume", default="auto", help="auto | none | /path/to/last.pt")
    ap.add_argument("--time_budget_h", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--extra_epochs", type=int, default=None,
                    help="train this many MORE epochs from wherever the checkpoint left "
                         "off. train.epochs is an absolute target, so resuming at epoch 38 "
                         "with train.epochs=12 asks for a range that is already finished.")
    ap.add_argument("--reset_count_head", action="store_true",
                    help="re-initialise the counting head after resuming, "
                         "keeping the separator. Use when a head has "
                         "collapsed: its weights are a local optimum with no "
                         "gradient out of it, so training longer cannot move "
                         "them.")
    ap.add_argument("--mem_guard_frac", type=float, default=0.85,
                    help="stop cleanly at this share of the container memory limit "
                         "(0 disables). A SIGKILL saves nothing; this saves a checkpoint.")
    ap.add_argument("--val_workers", type=int, default=0,
                    help="dataloader workers for validation (0 = in-process, least RAM)")
    ap.add_argument("--best_metric", default="p_si_snr",
                    choices=["p_si_snr", "sisdri_count_correct", "count_acc", "neg_loss"])
    ap.add_argument("--dry_run", action="store_true", help="5 steps + 1 eval batch, then exit")
    ap.add_argument("--dev_n", type=int, nargs="*", default=None,
                    help="validate only on these speaker counts (default: all of "
                         "them). A run that trains on one N wants a validation "
                         "number for that N, not an average over four counts it "
                         "never sees.")
    ap.add_argument("--no_throughput", action="store_true")
    args = ap.parse_args()

    from csnet.checkpoint import (TimeBudget, find_resume, kaggle_resume_instructions,
                                  keep_last_k, load_checkpoint, print_resume_banner,
                                  save_checkpoint, unwrap)
    from csnet.config import cfg_to_dict, load_cfg, save_cfg, seg_len
    from csnet.datasets import DynamicMixDataset, FrozenMixDataset, build_loader
    from csnet.engine import build_optimizer, evaluate, measure_throughput, train_one_epoch
    from csnet.losses import RectangularPITLoss
    from csnet.model import build_model
    from csnet.memory import (LeakWatch, MemoryBudgetExceeded, MemoryGuard,
                              log_memory, release)
    from csnet.utils import human_time, json_dump_atomic, pick_device, seed_everything

    cfg = load_cfg(resolve(args.config), args.set)
    store_root = require_store(args)
    cfg.data.store_root = store_root
    if args.noise_store:
        cfg.data.noise_store = resolve(args.noise_store)
    if args.ckpt_dir:
        cfg.train.ckpt_dir = resolve(args.ckpt_dir) or args.ckpt_dir
    if args.time_budget_h is not None:
        cfg.train.time_budget_h = float(args.time_budget_h)
    if args.dry_run:
        cfg.train.epochs, cfg.train.steps_per_epoch = 1, 5
        cfg.train.num_workers, cfg.train.val_batches = 0, 1

    ckpt_dir = cfg.train.ckpt_dir
    out_dir = resolve(args.out) or args.out or ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    seed_everything(cfg.train.seed)
    device = pick_device(None if args.device == "auto" else args.device)
    length = seg_len(cfg)

    banner("04 - train")
    print(cfg.summary())
    print(f"\ndevice: {device} | cuda devices: {torch.cuda.device_count()}")
    print(f"code  : {code_version()}")

    # ---------------------------------------------------------------- data
    train_store, train_bank = build_store_and_bank(
        store_root, cfg.data.train_split, mmap=cfg.data.mmap,
        noise_store=cfg.data.noise_store, noise_kinds=cfg.data.noise_kinds,
        noise_weights=cfg.data.noise_weights)
    dev_store, dev_bank = build_store_and_bank(
        store_root, cfg.data.dev_split, mmap=cfg.data.mmap,
        noise_store=cfg.data.noise_store, noise_kinds=cfg.data.noise_kinds,
        noise_weights=cfg.data.noise_weights)

    recipes_dev = resolve(args.recipes_dev) or os.path.join(
        resolve(cfg.data.recipes_dir) or cfg.data.recipes_dir, cfg.data.recipes_dev)
    if not os.path.exists(recipes_dev):
        raise SystemExit(f"no frozen dev set at {recipes_dev}\n"
                         "  Run scripts/01_make_frozen_sets.py first, or pass --recipes_dev.")

    train_set = DynamicMixDataset(
        train_store, train_bank, n_list=cfg.data.n_list,
        steps=cfg.train.steps_per_epoch * cfg.train.batch_size, seg_len=length,
        seed=cfg.train.seed, max_n_src=cfg.model.max_n_src,
        gain_db_range=tuple(cfg.data.gain_db_range), snr_db_range=tuple(cfg.data.snr_db_range),
        p_clean=cfg.data.p_clean, n_weights=cfg.data.n_weights,
        min_crop_rms_ratio=cfg.data.min_crop_rms_ratio)
    dev_set = FrozenMixDataset(dev_store, dev_bank, recipes_dev, seg_len=length,
                               max_n_src=cfg.model.max_n_src, n_list=args.dev_n)
    if args.dev_n:
        print(f"dev restricted to N in {sorted(args.dev_n)}: {len(dev_set)} mixtures")

    train_loader = build_loader(train_set, batch_size=cfg.train.batch_size, shuffle=False,
                                num_workers=cfg.train.num_workers, drop_last=True,
                                pin_memory=cfg.train.pin_memory)
    # The validation loader is entered once per epoch, so persistent workers would just be
    # processes holding a memory-mapped corpus open between uses -- host RAM a Kaggle GPU
    # session does not have to spare. Single-process costs a few seconds per epoch.
    dev_loader = build_loader(dev_set, batch_size=cfg.train.batch_size, shuffle=False,
                              num_workers=int(args.val_workers), persistent=False,
                              pin_memory=cfg.train.pin_memory)
    print(f"\ntrain: {train_store.split} dynamic mixing, {cfg.train.steps_per_epoch} steps/epoch"
          f" x batch {cfg.train.batch_size} | {train_bank.describe()}")
    print(f"dev  : {len(dev_set)} frozen mixtures {dev_set.counts_per_n()} from {recipes_dev}")
    if cfg.train.val_batches is not None:
        # Frozen recipes are written grouped by speaker count and the dev loader does
        # not shuffle, so "first K batches" means "only the smallest N". Truncating
        # would quietly validate on N=1 and N=2 alone and report it as the dev set.
        print(f"  WARNING: val_batches={cfg.train.val_batches} truncates a set"
              f" ordered by N; validation would cover only the lowest counts.")

    # ---------------------------------------------------------------- model
    model = build_model(cfg.model)
    print(f"\n{model.describe()}")
    log_memory("after model build")
    if cfg.train.freeze_separator:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("count_head")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"separator FROZEN -- training the count head alone ({trainable:,} parameters)")
    model.to(device)

    loss_fn = RectangularPITLoss(
        max_n_src=cfg.model.max_n_src, predict_noise=cfg.model.predict_noise,
        n_classes=cfg.model.n_classes, w_sep=cfg.loss.w_sep, w_sil=cfg.loss.w_sil,
        w_count=cfg.loss.w_count, w_noise=cfg.loss.w_noise, silence_db=cfg.loss.silence_db,
        label_smoothing=cfg.loss.label_smoothing, clamp_si_sdr=cfg.loss.clamp_si_sdr).to(device)

    optimizer, scheduler = build_optimizer(model, cfg)
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # ---------------------------------------------------------------- resume
    start_epoch, global_step, best_metric, history, consumed_h = 0, 0, float("-inf"), [], 0.0
    resume_path = find_resume(None if args.resume == "auto" else args.resume,
                              work_dir=ckpt_dir, search_inputs=(args.resume == "auto"))
    if args.resume.lower() == "none":
        resume_path = None
    if resume_path:
        print(f"\nresuming from {resume_path}")
        try:
            state = load_checkpoint(resume_path, model=model, optimizer=optimizer,
                                    scheduler=scheduler, scaler=scaler,
                                    map_location=str(device))
        except (RuntimeError, ValueError) as exc:
            # The architecture moved on. Throwing away a trained separator over a changed
            # head would be absurd, so load what still matches -- but say exactly what did
            # not, because a quiet partial load is how a "resumed" run starts from scratch
            # without anyone noticing.
            print(f"  checkpoint predates the current model: {str(exc).splitlines()[0]}")
            state = load_checkpoint(resume_path, model=model, map_location=str(device),
                                    strict=False)
            for label, keys in (("missing from the checkpoint",
                                 state.get("_missing_keys", [])),
                                ("no longer in the model",
                                 state.get("_unexpected_keys", []))):
                if keys:
                    shown = ", ".join(keys[:6]) + (" ..." if len(keys) > 6 else "")
                    print(f"  {len(keys)} parameter(s) {label}: {shown}")
            print("  Every weight that still exists was restored. Optimizer, scheduler and")
            print("  AMP-scaler state were NOT: they are indexed by the parameter list, and")
            print("  that list changed. Adam's moments restart; the weights are intact.")
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        best_metric = float(state.get("best_metric", float("-inf")))
        history = list(state.get("history", []))
        consumed_h = float(state.get("wall_h", 0.0))
        was = (state.get("extra") or {}).get("best_metric_name")
        if was is None and best_metric == best_metric and best_metric != float("-inf"):
            print(f"  checkpoint does not record which metric its best ({best_metric:.3f}) "
                  f"belongs to, so it starts over")
            best_metric = float("-inf")
        elif was and was != args.best_metric:
            # "best" is a number on one metric's scale. Carrying a P-SI-SNR of -21.8 into a
            # run scored on accuracy would make the first epoch look like a record.
            print(f"  best_metric changed ({was} -> {args.best_metric}); "
                  f"the stored best is on the old scale, so it starts over")
            best_metric = float("-inf")
        if args.reset_count_head:
            unwrap(model).reset_count_head()
            print("  counting head RE-INITIALISED; separator kept")
        print(f"  epoch {start_epoch}, step {global_step}, best {args.best_metric} "
              f"{best_metric:.3f}, {consumed_h:.2f} h already spent")
        if resume_path.startswith("/kaggle/input"):
            print("  (loaded from an attached dataset -- new checkpoints go to "
                  f"{ckpt_dir})")
    else:
        print("\nno checkpoint found -- starting from scratch")

    if torch.cuda.device_count() > 1 and cfg.train.dataparallel and device.type == "cuda":
        model = torch.nn.DataParallel(model)
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")

    budget = TimeBudget(cfg.train.time_budget_h, consumed_h=consumed_h)
    save_cfg(cfg, os.path.join(out_dir, "config.yaml"))

    # ---------------------------------------------------------------- throughput
    if args.extra_epochs is not None:
        cfg.train.epochs = start_epoch + max(1, int(args.extra_epochs))
        print(f"target: {args.extra_epochs} more epochs from epoch {start_epoch} "
              f"-> train.epochs = {cfg.train.epochs}")
    elif start_epoch >= int(cfg.train.epochs):
        # range(38, 12) is empty. Without this the run trains nothing, exits 0 and
        # reports "epochs completed: 39", which looks like success.
        raise SystemExit('\n'.join([
            f"nothing to train: resuming at epoch {start_epoch} with "
            f"train.epochs={cfg.train.epochs}.",
            "  train.epochs is an absolute target, not a count of additional epochs,",
            f"  so range({start_epoch}, {cfg.train.epochs}) is empty: this run would",
            "  do nothing and exit 0. Say what you meant instead:",
            "    --extra_epochs 12          12 more epochs from here",
            f"    --set train.epochs={start_epoch + 12}      the same thing, absolute",
            "    --resume none              start over",
        ]))

    planned_steps = max(1, (int(cfg.train.epochs) - start_epoch)
                        * int(cfg.train.steps_per_epoch))
    horizon_steps = planned_steps
    if not args.no_throughput and not args.dry_run:
        stats = measure_throughput(model, train_loader, loss_fn, optimizer, scaler, device,
                                   steps=20, amp=use_amp)
        steps_left = stats["steps_per_hour"] * budget.remaining_h()
        epochs_left = steps_left / max(1, cfg.train.steps_per_epoch)
        banner("measured throughput (this replaces the FLOP table)")
        print(f"  {stats['seconds_per_step'] * 1000:8.1f} ms per step "
              f"(batch {stats['batch_size']}, {stats['audio_seconds_per_step']:.0f} s of audio)")
        print(f"  {stats['steps_per_hour']:8.0f} steps per hour   ~{stats['tflops']:.2f} TFLOP/s")
        print(f"  budget remaining {budget.remaining_h():.2f} h -> about "
              f"{epochs_left:.1f} epochs of {cfg.train.steps_per_epoch} steps")
        # What this session will actually attempt: whichever of the plan and the budget
        # runs out first. That is the bar a memory leak has to clear to be survivable.
        horizon_steps = max(1, min(planned_steps, int(steps_left)))
        if start_epoch + epochs_left < cfg.train.epochs:
            print(f"  NOTE: cfg.train.epochs={cfg.train.epochs} will NOT finish this session.")
            print("        That is fine -- it resumes. But decide the epoch count now and")
            print("        state it in the report as a budget decision (docs/DESIGN.md sec 9).")

    log_fields = ["epoch", "global_step", "lr", "train_loss", "train_sisdr", "train_acc",
                  "val_loss", "val_sisdr", "val_count_acc", "val_p_si_snr",
                  "val_sisdri_count_correct", "wall_h", "seconds"]
    log_handle, log_writer = _open_log(os.path.join(out_dir, "train_log.csv"), log_fields)

    def checkpoint(path: str, epoch: int, step: int) -> None:
        save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
                        scaler=scaler, epoch=epoch, global_step=step,
                        best_metric=best_metric, history=history,
                        cfg_dict=cfg_to_dict(cfg), wall_h=budget.total_h(),
                        extra={"best_metric_name": args.best_metric})

    mem_guard = MemoryGuard(frac=float(args.mem_guard_frac)) if args.mem_guard_frac > 0 else None
    leak_watch = LeakWatch(mem_guard, horizon_steps) if mem_guard else None
    if mem_guard is not None:
        print(mem_guard.describe())
        print(leak_watch.describe())

    # ---------------------------------------------------------------- loop
    banner("training")
    stopped_early = False
    ran_any = False
    epoch = start_epoch
    try:
        for epoch in range(start_epoch, int(cfg.train.epochs)):
            ran_any = True
            train_set.set_epoch(epoch)
            t0 = time.time()

            last_saved = {"step": global_step}

            def on_step(step: int, _logs: dict) -> None:
                if (cfg.train.ckpt_every_steps > 0
                        and step - last_saved["step"] >= cfg.train.ckpt_every_steps):
                    checkpoint(os.path.join(ckpt_dir, "last.pt"), epoch, step)
                    last_saved["step"] = step

            train_logs = train_one_epoch(
                model, train_loader, loss_fn, optimizer, scaler, device,
                scheduler=scheduler, grad_clip=cfg.train.grad_clip,
                log_every=cfg.train.log_every, budget=budget, global_step=global_step,
                amp=use_amp, accum=cfg.train.accum, on_step=on_step,
                max_steps=cfg.train.steps_per_epoch, mem_guard=mem_guard,
                leak_watch=leak_watch)
            global_step = int(train_logs["global_step"])
            stopped_early = bool(train_logs["stopped_early"])

            do_val = ((epoch + 1) % max(1, cfg.train.val_every) == 0) or stopped_early
            val_logs: dict = {}
            if do_val:
                # Validation was where a 3-epoch run died with SIGKILL and no traceback,
                # so bracket it: if it happens again the log says which side it was on.
                log_memory(f"epoch {epoch + 1} before val")
                val_logs = evaluate(model, dev_loader, loss_fn, device, amp=use_amp,
                                    max_batches=cfg.train.val_batches,
                                    max_n_src=cfg.model.max_n_src,
                                    n_list=cfg.data.n_list, mem_guard=mem_guard)
                release()
                log_memory(f"epoch {epoch + 1} after val ")

            score = {
                "p_si_snr": val_logs.get("p_si_snr", float("-inf")),
                "sisdri_count_correct": val_logs.get("sisdri_count_correct", float("-inf")),
                "count_acc": val_logs.get("count_acc", float("-inf")),
                "neg_loss": -val_logs.get("loss", float("inf")),
            }[args.best_metric]
            if score != score:  # NaN early in training
                score = float("-inf")

            if scheduler is not None and val_logs and \
                    isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)

            elapsed = time.time() - t0
            row = {
                "epoch": epoch + 1, "global_step": global_step, "lr": train_logs["lr"],
                "train_loss": train_logs.get("loss"), "train_sisdr": train_logs.get("sisdr"),
                "train_acc": train_logs.get("acc"), "val_loss": val_logs.get("loss"),
                "val_sisdr": val_logs.get("sisdr"), "val_count_acc": val_logs.get("count_acc"),
                "val_p_si_snr": val_logs.get("p_si_snr"),
                "val_sisdri_count_correct": val_logs.get("sisdri_count_correct"),
                "wall_h": round(budget.total_h(), 4), "seconds": round(elapsed, 1),
            }
            history.append(row)
            log_writer.writerow(row)
            log_handle.flush()
            json_dump_atomic(history, os.path.join(out_dir, "history.json"))

            # A counting head that has collapsed reports exactly chance forever,
            # and separation keeps improving alongside it, so the run looks
            # healthy for as long as you let it. It cost thirteen hours once.
            chance = 1.0 / max(1, int(cfg.model.n_classes))
            acc_now = val_logs.get("count_acc", float("nan"))
            recent = [h.get("val_count_acc") for h in history[-4:]]
            recent = [a for a in recent if a is not None and a == a]
            if (len(recent) >= 4 and acc_now == acc_now
                    and max(recent) <= chance * 1.05):
                print(f"  WARNING: counting accuracy has been at chance ({100 * chance:.0f} %)")
                print("           for four epochs. A head that has collapsed reports exactly")
                print("           this and never recovers, because a dead unit gets no")
                print("           gradient. Check it before spending more hours:")
                print("             python scripts/11_inspect_count_head.py --ckpt "
                      f"{os.path.join(ckpt_dir, 'last.pt')} --store {store_root}")

            print(f"epoch {epoch + 1:3d}/{cfg.train.epochs} | "
                  f"train loss {train_logs.get('loss', float('nan')):7.3f} "
                  f"sisdr {train_logs.get('sisdr', float('nan')):6.2f} "
                  f"acc {train_logs.get('acc', float('nan')):.3f} | "
                  f"val sisdr {val_logs.get('sisdr', float('nan')):6.2f} "
                  f"count {val_logs.get('count_acc', float('nan')):.3f} "
                  f"P-SI-SNR {val_logs.get('p_si_snr', float('nan')):6.2f} | "
                  f"{human_time(elapsed)} | total {budget.total_h():.2f} h", flush=True)

            # Update best_metric BEFORE writing last.pt, or a resume reads a stale best
            # and re-saves best.pt on an epoch that did not actually improve.
            improved = score > best_metric
            if improved:
                best_metric = score
            checkpoint(os.path.join(ckpt_dir, "last.pt"), epoch + 1, global_step)
            if improved:
                checkpoint(os.path.join(ckpt_dir, "best.pt"), epoch + 1, global_step)
                print(f"          new best {args.best_metric} = {best_metric:.3f} -> best.pt")
            if cfg.train.keep_last_k:
                keep_last_k(ckpt_dir, cfg.train.keep_last_k)

            if args.dry_run:
                print("\ndry run complete -- the pipeline is wired correctly.")
                break
            if stopped_early:
                break
            if cfg.train.early_stop_patience:
                recent = [h.get("val_p_si_snr") or float("-inf")
                          for h in history[-cfg.train.early_stop_patience:]]
                if (len(recent) == cfg.train.early_stop_patience
                        and max(recent) <= best_metric - 1e-6):
                    print(f"early stop: no improvement in {cfg.train.early_stop_patience} epochs")
                    break
    except KeyboardInterrupt:
        print("\ninterrupted -- saving before exit")
        checkpoint(os.path.join(ckpt_dir, "last.pt"), epoch, global_step)
        stopped_early = True
    except MemoryBudgetExceeded as exc:
        # Losing the session is unavoidable at this point; losing the checkpoint and the
        # diagnosis is not. Exit 0 so the notebook still commits and the output survives.
        banner("STOPPED ON MEMORY")
        print(exc)
        log_memory("at the stop")
        print(mem_guard.describe())
        print(leak_watch.describe())
        print("")
        print("This is the container memory limit, not GPU memory. To try, in order:")
        print("  1. --set train.num_workers=1   (each worker holds the corpus open)")
        print("  2. --set train.num_workers=0   (loads in-process; slower but flat)")
        print("Send the ram lines from this log: the growth rate between them says")
        print("whether it climbs per step or jumps at validation -- different bugs.")
        checkpoint(os.path.join(ckpt_dir, "last.pt"), epoch, global_step)
        print(f"checkpoint saved -> {os.path.join(ckpt_dir, 'last.pt')}")
        stopped_early = True
    finally:
        log_handle.close()

    banner("done")
    print(f"epochs completed : {epoch + (0 if stopped_early else 1) if ran_any else 0}"
          f"{'' if ran_any else '  (no epoch ran)'}")
    print(f"global step      : {global_step}")
    print(f"best {args.best_metric:<12}: {best_metric:.3f}")
    print(f"wall clock       : {human_time(budget.total_h() * 3600)} across all sessions")
    print(f"checkpoints      : {ckpt_dir}")

    if stopped_early or (epoch + 1) < int(cfg.train.epochs):
        print_resume_banner(kaggle_resume_instructions(ckpt_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
