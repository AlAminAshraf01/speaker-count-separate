#!/usr/bin/env python3
"""Find which training component leaks host memory, in one short GPU session.

A leak that only appears on Kaggle cannot be reasoned about from a laptop, and finding it
by changing one thing and running eleven hours costs a week of quota. This runs the real
training loop several times over, each with one component switched off, and reports the
memory slope of each. Whichever configuration comes out flat names the culprit.

    python scripts/10_memory_bisect.py --store /kaggle/input/.../store \\
        --recipes_dev /kaggle/input/.../recipes_dev.csv

About 15 minutes on a T4 x2 at the defaults. Nothing is trained and nothing is saved --
the model is thrown away between configurations.

The slope that matters is the **cgroup** one: that is the number Kaggle kills a session
over, and it counts the dataloader workers and page cache as well as this process. The
process RSS is reported next to it because the gap between them says whether the growth is
inside the trainer or outside it.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time

from _common import add_common_args, banner, build_store_and_bank, require_store, resolve


def _fresh(cfg, store_root: str, device, seg: int):
    """Build a complete, independent training setup. Nothing is shared between configs."""
    import torch

    from csnet.datasets import DynamicMixDataset, build_loader
    from csnet.losses import RectangularPITLoss
    from csnet.model import build_model

    train_store, train_bank = build_store_and_bank(store_root, cfg.data.train_split,
                                                   mmap=cfg.data.mmap)
    train_set = DynamicMixDataset(
        train_store, train_bank, n_list=cfg.data.n_list,
        steps=cfg.train.steps_per_epoch * cfg.train.batch_size, seg_len=seg,
        seed=cfg.train.seed, max_n_src=cfg.model.max_n_src,
        gain_db_range=tuple(cfg.data.gain_db_range),
        snr_db_range=tuple(cfg.data.snr_db_range), p_clean=cfg.data.p_clean)
    loader = build_loader(train_set, batch_size=cfg.train.batch_size, shuffle=False,
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
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.amp)
                                  and device.type == "cuda")
    return loader, model, loss_fn, optimizer, scaler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--config", default="configs/paper.yaml")
    ap.add_argument("--warmup", type=int, default=40,
                    help="steps to run before the first measurement (startup allocations)")
    ap.add_argument("--steps", type=int, default=120,
                    help="steps to measure the slope over, after the warmup")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these configurations by name")
    ap.add_argument("--set", dest="overrides", nargs="*", default=None,
                    help="config overrides applied to every configuration, e.g. "
                         "train.batch_size=8")
    args = ap.parse_args()

    import torch

    from csnet.config import load_cfg, seg_len
    from csnet.engine import train_one_epoch
    from csnet.memory import cgroup_usage_gb, format_snapshot, release, rss_gb
    from csnet.utils import format_table, pick_device, seed_everything

    store_root = require_store(args)
    device = pick_device(args.device if hasattr(args, "device") else None)

    banner("10 - memory bisect")
    print(f"store  : {store_root}")
    print(f"device : {device} | cuda devices: {torch.cuda.device_count()}")
    print(f"budget : {args.warmup} warmup + {args.steps} measured steps per configuration")
    print(format_snapshot("at start"))

    _, limit_gb = cgroup_usage_gb()
    if limit_gb != limit_gb:
        print("\nNo cgroup limit is readable here, so only process RSS will be meaningful.\n"
              "That is expected off Kaggle; on Kaggle it means something changed.")

    # One knob each, so a flat row names exactly one component.
    configs: list[tuple[str, dict]] = [
        ("baseline", {}),
        ("no amp", {"train.amp": False}),
        ("no dataparallel", {"train.dataparallel": False}),
        ("workers 0", {"train.num_workers": 0}),
        ("no amp + no dataparallel", {"train.amp": False, "train.dataparallel": False}),
    ]
    if args.only:
        wanted = {name.lower() for name in args.only}
        configs = [c for c in configs if c[0].lower() in wanted]
        if not configs:
            raise SystemExit(f"--only matched nothing; names are: "
                             f"{[c[0] for c in configs]}")

    rows = []
    for name, overrides in configs:
        # The shared --set goes first so a configuration's own knob always wins.
        overrides_list = list(args.overrides or []) + [f"{k}={v}" for k, v in
                                                       overrides.items()]
        cfg = load_cfg(resolve(args.config) or args.config, overrides_list)
        cfg.data.store_root = store_root
        seed_everything(cfg.train.seed)
        seg = seg_len(cfg)

        print(f"\n--- {name} " + "-" * (66 - len(name)))
        print(f"    amp={cfg.train.amp} dataparallel={cfg.train.dataparallel} "
              f"workers={cfg.train.num_workers} pin_memory={cfg.train.pin_memory}")

        loader, model, loss_fn, optimizer, scaler = _fresh(cfg, store_root,
                                                           device, seg)
        common = dict(scheduler=None, grad_clip=cfg.train.grad_clip, log_every=0,
                      amp=bool(cfg.train.amp) and device.type == "cuda",
                      accum=cfg.train.accum)

        train_one_epoch(model, loader, loss_fn, optimizer, scaler, device,
                        max_steps=args.warmup, **common)
        before_cg, _ = cgroup_usage_gb()
        before_rss = rss_gb()
        t0 = time.time()

        train_one_epoch(model, loader, loss_fn, optimizer, scaler, device,
                        max_steps=args.steps, **common)
        after_cg, _ = cgroup_usage_gb()
        after_rss = rss_gb()
        seconds = time.time() - t0

        cg_per_step = (after_cg - before_cg) * 1024.0 / max(1, args.steps)
        rss_per_step = (after_rss - before_rss) * 1024.0 / max(1, args.steps)
        rows.append([name, f"{cg_per_step:.2f}", f"{rss_per_step:.2f}",
                     f"{after_cg:.1f}", f"{1000.0 * seconds / max(1, args.steps):.0f}"])
        print(f"    cgroup {cg_per_step:+.2f} MiB/step | rss {rss_per_step:+.2f} MiB/step "
              f"| {1000.0 * seconds / max(1, args.steps):.0f} ms/step")

        # Tear the whole setup down, or the next configuration measures this one's mess.
        del loader, model, loss_fn, optimizer, scaler
        gc.collect()
        release()

    banner("results")
    print(format_table(rows, ["configuration", "cgroup MiB/step", "rss MiB/step",
                              "cgroup GiB", "ms/step"]))

    # nan is "not measured", not "not leaking" -- keep the two apart or the verdict lies.
    numeric = [(row[0], float(row[1])) for row in rows
               if float(row[1]) == float(row[1])]
    flat = [name for name, slope in numeric if slope < 1.0]
    worst = max(numeric, key=lambda kv: kv[1]) if numeric else ("", 0.0)
    print()
    if not rows:
        print("nothing ran")
    elif not numeric:
        print("No cgroup readings were available, so no slope could be measured. This tool")
        print("only says anything inside a container that exposes a memory cgroup, which")
        print("on Kaggle it does -- off Kaggle there is nothing here to see.")
    elif worst[1] < 1.0:
        print("Every configuration is flat. The leak is not in amp, DataParallel or the")
        print("dataloader workers -- re-run with more steps, or it is elsewhere entirely.")
    elif flat:
        print(f"Flat: {', '.join(flat)}")
        print(f"Leaking worst: {worst[0]} at {worst[1]:.1f} MiB/step")
        print("The component switched off in the first flat row is the culprit. Set it in")
        print("configs/base.yaml and train with it off.")
    else:
        print(f"Nothing came out flat; worst is {worst[0]} at {worst[1]:.1f} MiB/step.")
        print("The leak survives every switch here, so it is in the shared path: the loop,")
        print("the store, or the CUDA context. Send this table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
