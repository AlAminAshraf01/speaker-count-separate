#!/usr/bin/env python3
"""Ask a trained checkpoint why its counting head is not learning.

A counting head whose cross-entropy sits at exactly ``ln(n_classes)`` with accuracy at
exactly chance is not "finding the task hard" -- a hard task still moves the loss. It is
emitting the same logits for every input, and there are only a few ways that happens. This
runs real dev batches through the model and measures which one:

* **dead hidden layer** -- ``F.relu(fc1(pooled))`` is zero for every sample, so ``fc2`` can
  only output its bias. The bias then converges to the class marginal, which for a balanced
  set is uniform, which is exactly ``ln(K)``. No gradient reaches ``fc1`` or ``proj`` ever
  again, so it cannot recover on its own.
* **constant features** -- the pooled statistics do not vary across samples, so there is
  nothing for any head to separate.
* **saturated logits** -- huge equal logits, vanishing softmax gradient.
* **none of the above** -- the head varies its output and is simply wrong, which is a
  training problem rather than a broken one.

    python scripts/11_inspect_count_head.py --ckpt /kaggle/working/ckpt/last.pt \\
        --store /kaggle/input/.../store --recipes_dev /kaggle/input/.../recipes_dev.csv

CPU is fine and takes under a minute. Nothing is written.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from _common import (add_common_args, banner, build_store_and_bank, code_version,
                     require_store, resolve)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--ckpt", required=True, help="checkpoint to inspect (last.pt/best.pt)")
    ap.add_argument("--recipes_dev", default=None)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=12)
    args = ap.parse_args()

    import torch

    from csnet.checkpoint import load_checkpoint
    from csnet.config import dict_to_cfg, seg_len
    from csnet.datasets import FrozenMixDataset, build_loader
    from csnet.model import build_model
    from csnet.utils import format_table

    store_root = require_store(args)
    ckpt_path = resolve(args.ckpt) or args.ckpt

    banner("11 - why is the counting head silent")
    print(f"code  : {code_version()}")
    print(f"ckpt  : {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict_to_cfg(state["cfg"]) if "cfg" in state else None
    if cfg is None:
        raise SystemExit("checkpoint has no config; cannot rebuild the model")
    print(f"epoch : {state.get('epoch')}   step: {state.get('global_step')}")

    model = build_model(cfg.model)
    load_checkpoint(ckpt_path, model=model, map_location="cpu", restore_rng=False)
    model.eval()

    recipes = resolve(args.recipes_dev) or os.path.join(
        resolve(cfg.data.recipes_dir) or cfg.data.recipes_dir, cfg.data.recipes_dev)
    if not os.path.exists(recipes):
        raise SystemExit(f"no dev recipes at {recipes}; pass --recipes_dev")

    store, bank = build_store_and_bank(store_root, cfg.data.dev_split, mmap=cfg.data.mmap)
    dev = FrozenMixDataset(store, bank, recipes, seg_len=seg_len(cfg),
                           max_n_src=cfg.model.max_n_src)
    # The frozen set is ordered by speaker count, so take a shuffled sample or every batch
    # would hold one value of N and say nothing about whether the head discriminates.
    loader = build_loader(dev, batch_size=args.batch_size, shuffle=True, num_workers=0,
                          pin_memory=False)

    head = model.count_head
    captured: dict = {}

    def grab(name):
        def hook(_module, inputs, output):
            captured.setdefault(name, []).append(
                (inputs[0].detach().clone(), output.detach().clone()))
        return hook

    handles = [head.fc1.register_forward_hook(grab("fc1")),
               head.fc2.register_forward_hook(grab("fc2"))]

    labels, preds = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            out = model(batch["mix"])
            preds.append(out["count_logits"].argmax(dim=-1).cpu().numpy())
            labels.append(batch["cls"].cpu().numpy())
    for handle in handles:
        handle.remove()

    pooled = torch.cat([a for a, _ in captured["fc1"]]).numpy()      # (N, 2*hidden)
    pre_act = torch.cat([b for _, b in captured["fc1"]]).numpy()     # (N, hidden)
    hidden_in = torch.cat([a for a, _ in captured["fc2"]]).numpy()   # (N, hidden) post-relu
    logits = torch.cat([b for _, b in captured["fc2"]]).numpy()      # (N, n_classes)
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)

    # ---------------------------------------------------------------- measurements
    logit_spread_per_sample = float(np.mean(logits.std(axis=1)))
    logit_var_across_samples = float(np.mean(logits.std(axis=0)))
    alive_per_unit = (pre_act > 0).mean(axis=0)
    dead_units = int((alive_per_unit == 0.0).sum())
    pooled_var = float(np.mean(pooled.std(axis=0)))
    hidden_all_zero = float((hidden_in == 0.0).mean())
    accuracy = float((preds == labels).mean())

    rows = [
        ["samples measured", f"{len(labels)}"],
        ["accuracy", f"{100 * accuracy:.1f} %  (chance {100.0 / cfg.model.n_classes:.0f} %)"],
        ["distinct predictions", f"{len(np.unique(preds))} of {cfg.model.n_classes}"],
        ["logit spread within a sample", f"{logit_spread_per_sample:.4f}"],
        ["logit variation across samples", f"{logit_var_across_samples:.4f}"],
        ["pooled feature variation", f"{pooled_var:.4f}"],
        ["fc1 units never active", f"{dead_units} of {pre_act.shape[1]}"],
        ["post-ReLU values that are zero", f"{100 * hidden_all_zero:.1f} %"],
        ["fc2 bias", np.array2string(head.fc2.bias.detach().numpy(), precision=3)],
    ]
    print()
    print(format_table(rows, ["measurement", "value"]))

    print("\nprediction histogram (what the head actually says):")
    for cls in range(cfg.model.n_classes):
        n_true = int((labels == cls).sum())
        n_pred = int((preds == cls).sum())
        print(f"  N={cfg.data.n_list[cls]}   true {n_true:4d}   predicted {n_pred:4d}")

    # ---------------------------------------------------------------- verdict
    banner("verdict")
    if pooled_var < 1e-6:
        print("CONSTANT FEATURES: the pooled statistics are identical for every input, so")
        print("no head could tell the classes apart. The problem is upstream of the head.")
    elif dead_units == pre_act.shape[1]:
        print("DEAD HIDDEN LAYER: every fc1 unit is negative for every sample, so ReLU")
        print("zeroes the whole layer and fc2 can only emit its bias. That bias settles on")
        print("the class marginal -- uniform for a balanced set -- which is exactly the")
        print("ln(K) cross-entropy observed. No gradient reaches fc1 or proj, so it cannot")
        print("recover: the head has to be changed, not trained longer.")
    elif logit_var_across_samples < 1e-4:
        print("CONSTANT OUTPUT: the features vary and the hidden layer is alive, but the")
        print("logits do not move between samples. Look at fc2.")
    elif logit_spread_per_sample > 20.0:
        print("SATURATED LOGITS: the spread within a sample is enormous, so softmax is")
        print("saturated and its gradient has vanished. Normalise before the head.")
    else:
        print("The head varies its output and is simply inaccurate. That is a training")
        print("problem -- objective weighting, learning rate, capacity -- not a broken")
        print(f"head: {dead_units} of {pre_act.shape[1]} units are dead and the logits do")
        print("move between samples.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
