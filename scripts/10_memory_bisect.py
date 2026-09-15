#!/usr/bin/env python3
"""Find which training component leaks host memory, in one short GPU session.

A leak that only appears on Kaggle cannot be reasoned about from a laptop, and finding it
by changing one thing and running eleven hours costs a week of quota. This runs the real
training loop once per configuration, each with one component switched off, and reports
the memory slope of each. Whichever configuration comes out flat names the culprit.

    python scripts/10_memory_bisect.py --store /kaggle/input/.../store

About 20 minutes on a T4 x2 at the defaults. Nothing is trained and nothing is saved.

**Each configuration runs in its own subprocess.** Building several CUDA setups in one
process does not work: tearing down a DataParallel model while its kernels are still in
flight corrupts the context, and the next configuration dies with ``misaligned address``
somewhere unrelated. A fresh process per configuration also gives a fresh memory baseline,
which is what a slope measurement wants, and means one crash costs one row rather than the
whole run.

The slope that matters is the **cgroup** one: that is the number Kaggle kills a session
over, and it counts the dataloader workers and page cache as well as this process. Process
RSS sits next to it because the gap between them says whether the growth is inside the
trainer process or outside it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from _common import (add_common_args, banner, build_store_and_bank, code_version,
                     require_store, resolve)

# Ordered by suspicion, so the likely answer arrives early: if the run is cut short, the
# rows that did finish are still the ones worth having.
CONFIGS: list[tuple[str, dict]] = [
    ("baseline", {}),
    ("workers 0", {"train.num_workers": 0}),
    ("no dataparallel", {"train.dataparallel": False}),
    ("no amp", {"train.amp": False}),
    ("workers 0 + no dataparallel", {"train.num_workers": 0,
                                     "train.dataparallel": False}),
]
_MARKER = "BISECT_RESULT"


# --------------------------------------------------------------------------- the child

def run_one(name: str, args: argparse.Namespace, store_root: str) -> int:
    """Measure one configuration in this process, then print a machine-readable line."""
    import torch

    from csnet.config import load_cfg, seg_len
    from csnet.datasets import DynamicMixDataset, build_loader
    from csnet.engine import train_one_epoch
    from csnet.losses import RectangularPITLoss
    from csnet.memory import cgroup_usage_gb, rss_gb
    from csnet.model import build_model
    from csnet.utils import pick_device, seed_everything

    overrides = dict(CONFIGS)[name]
    # The shared --set goes first so a configuration's own switch always wins.
    override_list = list(args.overrides or []) + [f"{k}={v}" for k, v in overrides.items()]
    cfg = load_cfg(resolve(args.config) or args.config, override_list)
    seed_everything(cfg.train.seed)

    device = pick_device(None)
    seg = seg_len(cfg)
    use_amp = bool(cfg.train.amp) and device.type == "cuda"

    print(f"    amp={cfg.train.amp} dataparallel={cfg.train.dataparallel} "
          f"workers={cfg.train.num_workers} pin_memory={cfg.train.pin_memory}", flush=True)

    store, bank = build_store_and_bank(store_root, cfg.data.train_split, mmap=cfg.data.mmap)
    dataset = DynamicMixDataset(
        store, bank, n_list=cfg.data.n_list,
        steps=(args.warmup + args.steps + 8) * cfg.train.batch_size, seg_len=seg,
        seed=cfg.train.seed, max_n_src=cfg.model.max_n_src,
        gain_db_range=tuple(cfg.data.gain_db_range),
        snr_db_range=tuple(cfg.data.snr_db_range), p_clean=cfg.data.p_clean)
    loader = build_loader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                          num_workers=cfg.train.num_workers, drop_last=True,
                          pin_memory=cfg.train.pin_memory)

    model = build_model(cfg.model).to(device)
    if cfg.train.dataparallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    loss_fn = RectangularPITLoss(
        max_n_src=cfg.model.max_n_src, n_classes=cfg.model.n_classes,
        predict_noise=cfg.model.predict_noise, w_sep=cfg.loss.w_sep, w_sil=cfg.loss.w_sil,
        w_count=cfg.loss.w_count, w_noise=cfg.loss.w_noise,
        silence_db=cfg.loss.silence_db, label_smoothing=cfg.loss.label_smoothing,
        clamp_si_sdr=cfg.loss.clamp_si_sdr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    common = dict(scheduler=None, grad_clip=cfg.train.grad_clip, log_every=0,
                  amp=use_amp, accum=cfg.train.accum, progress=False)

    train_one_epoch(model, loader, loss_fn, optimizer, scaler, device,
                    max_steps=args.warmup, **common)
    if device.type == "cuda":
        # Measure once the queue has drained, or the warmup's allocations land inside the
        # measured window and inflate the slope.
        torch.cuda.synchronize()
    before_cg, _ = cgroup_usage_gb()
    before_rss = rss_gb()
    start = time.time()

    train_one_epoch(model, loader, loss_fn, optimizer, scaler, device,
                    max_steps=args.steps, **common)
    if device.type == "cuda":
        torch.cuda.synchronize()
    after_cg, _ = cgroup_usage_gb()
    after_rss = rss_gb()
    seconds = time.time() - start

    per = float(max(1, args.steps))
    print(f"{_MARKER}\t{name}\t{(after_cg - before_cg) * 1024.0 / per:.3f}"
          f"\t{(after_rss - before_rss) * 1024.0 / per:.3f}"
          f"\t{after_cg:.3f}\t{1000.0 * seconds / per:.0f}", flush=True)
    return 0


# -------------------------------------------------------------------------- the parent

def spawn(name: str, args: argparse.Namespace) -> tuple[list[str], str]:
    """Run one configuration in a fresh interpreter. Returns ``(row, error_text)``."""
    cmd = [sys.executable, os.path.abspath(__file__), "--child", name,
           "--config", args.config, "--warmup", str(args.warmup),
           "--steps", str(args.steps)]
    if args.store:
        cmd += ["--store", args.store]
    if args.overrides:
        cmd += ["--set", *args.overrides]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if not line.startswith(_MARKER):
            print(line, flush=True)
            continue
        _, got, cgroup, rss, total, ms = line.split("\t")
        return [got, cgroup, rss, total, ms], ""

    tail = (proc.stderr or "").strip().splitlines()
    reason = tail[-1] if tail else f"exit {proc.returncode} with no output"
    return [name, "FAILED", "FAILED", "-", "-"], reason


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--config", default="configs/paper.yaml")
    ap.add_argument("--warmup", type=int, default=30,
                    help="steps to run before measuring, so startup is not counted")
    ap.add_argument("--steps", type=int, default=100,
                    help="steps to measure the slope over")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these configurations by name")
    ap.add_argument("--set", dest="overrides", nargs="*", default=None,
                    help="config overrides applied to every configuration")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    store_root = require_store(args)
    if args.child:
        return run_one(args.child, args, store_root)

    from csnet.memory import cgroup_usage_gb, format_snapshot
    from csnet.utils import format_table

    names = [name for name, _ in CONFIGS]
    if args.only:
        wanted = {n.lower() for n in args.only}
        names = [n for n in names if n.lower() in wanted]
        if not names:
            raise SystemExit(f"--only matched nothing. Names are: {[n for n, _ in CONFIGS]}")

    banner("10 - memory bisect")
    print(f"store  : {store_root}")
    print(f"code   : {code_version()}")
    print(f"budget : {args.warmup} warmup + {args.steps} measured steps per configuration")
    print(f"configs: {', '.join(names)}")
    print(format_snapshot("at start"))
    _, limit_gb = cgroup_usage_gb()
    if limit_gb != limit_gb:
        print("\nNo cgroup limit is readable here, so no slope can be measured.")
        print("That is expected anywhere but inside a container.")

    rows: list[list[str]] = []
    errors: dict[str, str] = {}
    for name in names:
        print(f"\n--- {name} " + "-" * max(4, 66 - len(name)), flush=True)
        row, error = spawn(name, args)
        rows.append(row)
        if error:
            errors[name] = error
            print(f"    FAILED: {error}", flush=True)
        else:
            print(f"    cgroup {float(row[1]):+.2f} MiB/step | "
                  f"rss {float(row[2]):+.2f} MiB/step | {row[4]} ms/step", flush=True)

    banner("results")
    print(format_table(rows, ["configuration", "cgroup MiB/step", "rss MiB/step",
                              "cgroup GiB", "ms/step"]))
    if errors:
        print("\nfailed configurations:")
        for name, error in errors.items():
            print(f"  {name}: {error}")

    # nan is "not measured" and FAILED is "did not run"; neither means "not leaking".
    measured: list[tuple[str, float]] = []
    for row in rows:
        try:
            value = float(row[1])
        except ValueError:
            continue
        if value == value:
            measured.append((row[0], value))

    print()
    if not measured:
        # "Ran fine but there is no cgroup to read" and "every child died" are different
        # situations and must not share a message.
        if errors and len(errors) == len(rows):
            print("Every configuration crashed; the failures above say why.")
        else:
            print("Every configuration ran, but no cgroup readings were available, so no")
            print("slope could be measured. Inside a Kaggle container there would be.")
        return 0
    flat = [name for name, slope in measured if slope < 1.0]
    worst = max(measured, key=lambda kv: kv[1])
    if worst[1] < 1.0:
        print("Every configuration is flat -- no leak reproduced in this window.")
        print("Re-run with --steps 300 before concluding anything.")
    elif flat:
        print(f"FLAT : {', '.join(flat)}")
        print(f"WORST: {worst[0]} at {worst[1]:.1f} MiB/step")
        print("The switch that is off in the first flat row is the culprit. Put it in")
        print("configs/base.yaml and train with it off.")
    else:
        print(f"Nothing came out flat; worst is {worst[0]} at {worst[1]:.1f} MiB/step.")
        print("The leak survives every switch, so it is in the shared path: the loop, the")
        print("store, or the CUDA context. Send this table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
