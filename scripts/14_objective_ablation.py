#!/usr/bin/env python3
"""Is the separator broken, or is the task hard? Overfit one batch and find out.

A model that reaches +1.2 dB SI-SDRi after forty epochs could be broken -- a dead layer, a
mis-wired permutation, a gradient that never arrives -- or it could be a capable model on
a task that is simply harder than the budget allows. Those two diagnoses lead to opposite
work, and guessing between them costs GPU-hours either way.

A single fixed batch settles it. Memorising four mixtures is a trivial problem for a 2 M
parameter network: any model that cannot drive SI-SDR high on *one batch it sees every
step* has something wrong with it, and any model that can does not.

Running it twice, once with the separation term alone and once with the full production
loss, measures something else worth knowing: what the auxiliary objectives cost. The
silence penalty is not scale-invariant and SI-SDR is, so shrinking every slot drives the
penalty to zero at no cost to separation -- a free descent direction that teaches the
model nothing about which slot ought to be quiet.

    python scripts/14_objective_ablation.py --store /kaggle/input/.../store

Measured on CPU with ``configs/small.yaml``, batch 4, 250 steps:

    separation only : 28.36 dB
    full loss       : 19.66 dB      (silence term already 0.000 by step 100)

Both learn, so nothing is broken; the full objective is about 8.7 dB behind at matched
step count, so the auxiliary terms are a real tax and not the whole story. Minutes on a
GPU, about 25 minutes on CPU. Nothing is saved.
"""

from __future__ import annotations

import argparse
import sys
import time

from _common import (add_common_args, banner, build_store_and_bank, code_version,
                     require_store, resolve)

ARMS: list[tuple[str, dict]] = [
    ("separation only", {"w_sep": 1.0, "w_sil": 0.0, "w_count": 0.0, "w_noise": 0.0}),
    ("full loss", {"w_sep": 1.0, "w_sil": 1.0, "w_count": 0.5, "w_noise": 0.2}),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--n_src", type=int, default=2, help="speakers in every mixture")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch

    from csnet.config import load_cfg, seg_len
    from csnet.datasets import DynamicMixDataset, build_loader
    from csnet.losses import RectangularPITLoss
    from csnet.model import build_model
    from csnet.utils import format_table, pick_device, seed_everything

    store_root = require_store(args)
    cfg = load_cfg(resolve(args.config) or args.config, [])
    device = pick_device(None if args.device == "auto" else args.device)

    banner("14 - what does the objective cost?")
    print(f"code  : {code_version()}")
    print(f"config: {args.config}   device: {device}")
    print(f"budget: {args.steps} steps on one fixed batch of {args.batch_size} "
          f"{args.n_src}-speaker mixtures")

    store, bank = build_store_and_bank(store_root, cfg.data.train_split, mmap=cfg.data.mmap)
    seed_everything(cfg.train.seed)
    dataset = DynamicMixDataset(
        store, bank, n_list=[args.n_src], steps=args.batch_size, seg_len=seg_len(cfg),
        seed=cfg.train.seed, max_n_src=cfg.model.max_n_src,
        gain_db_range=tuple(cfg.data.gain_db_range),
        snr_db_range=tuple(cfg.data.snr_db_range), p_clean=1.0)
    batch = next(iter(build_loader(dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=0)))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    rows, best_by_arm = [], {}
    for name, weights in ARMS:
        seed_everything(cfg.train.seed)          # same init for both arms, or it proves nothing
        model = build_model(cfg.model).to(device)
        loss_fn = RectangularPITLoss(
            max_n_src=cfg.model.max_n_src, n_classes=cfg.model.n_classes,
            predict_noise=cfg.model.predict_noise, silence_db=cfg.loss.silence_db,
            label_smoothing=0.0, clamp_si_sdr=cfg.loss.clamp_si_sdr, **weights).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        print(f"\n--- {name} " + "-" * max(4, 60 - len(name)), flush=True)
        start, best = time.time(), float("-inf")
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            total, logs = loss_fn(model(batch["mix"]), batch)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            best = max(best, logs["sisdr"])
            if step == 1 or step % max(1, args.steps // 5) == 0:
                print(f"  step {step:4d}  loss {logs['loss']:8.3f}  "
                      f"sisdr {logs['sisdr']:7.2f} dB  sil {logs['sil']:7.3f}", flush=True)
        best_by_arm[name] = best
        rows.append([name, f"{best:.2f}", f"{logs['sil']:.3f}",
                     f"{time.time() - start:.0f}"])

    banner("results")
    print(format_table(rows, ["objective", "best SI-SDR dB", "final silence term", "seconds"]))

    a, b = best_by_arm[ARMS[0][0]], best_by_arm[ARMS[1][0]]
    print()
    if args.steps < 100:
        # Below about a hundred steps neither arm has gone anywhere, and "it did not learn
        # yet" reads exactly like "it cannot learn". Refuse to call it.
        print(f"{args.steps} steps is too few to conclude anything -- both arms are still")
        print("climbing. Re-run with --steps 250 before reading the verdict.")
        return 0
    if b < 3.0 and a < 3.0:
        print("NEITHER arm learns to separate one fixed batch. That is a bug, not a task")
        print("difficulty: check the model wiring and run `python -m csnet.losses`.")
    elif b < 3.0:
        print("Separation alone works and the full objective does not. The auxiliary terms")
        print("are not a tax here, they are the blocker -- look at the silence term first,")
        print("which is not scale-invariant while SI-SDR is.")
    else:
        print("Both arms learn, so the model, loss and optimiser are all capable: a model")
        print("that underperforms on the real task is facing task difficulty, not a bug.")
        print(f"The auxiliary objectives cost {b - a:+.2f} dB at matched step count.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
