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
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compare_precision", action="store_true",
                    help="run the same batches in fp32 and fp16 autocast and "
                         "report where they diverge. Needs a GPU.")
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

    from csnet.utils import pick_device

    device = pick_device(None if args.device == "auto" else args.device)
    model = build_model(cfg.model)
    load_checkpoint(ckpt_path, model=model, map_location="cpu", restore_rng=False)
    # On the GPU because this runs inside a GPU session: the fp32 pass took six and a
    # half minutes on the CPU while two T4s sat idle.
    model.to(device).eval()
    print(f"device: {device}")

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

    labels, preds, batches = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            batches.append(batch)
            out = model(batch["mix"].to(device))
            preds.append(out["count_logits"].argmax(dim=-1).cpu().numpy())
            labels.append(batch["cls"].cpu().numpy())
    for handle in handles:
        handle.remove()

    pooled = torch.cat([a for a, _ in captured["fc1"]]).float().cpu().numpy()      # (N, 2*hidden)
    pre_act = torch.cat([b for _, b in captured["fc1"]]).float().cpu().numpy()     # (N, hidden)
    hidden_in = torch.cat([a for a, _ in captured["fc2"]]).float().cpu().numpy()   # (N, hidden) post-relu
    logits = torch.cat([b for _, b in captured["fc2"]]).float().cpu().numpy()      # (N, n_classes)
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
        # The absolute number means nothing without a scale. 0.0058 sounds small and is
        # small -- but only because the features it is measured against are of order 1.
        ["  as a fraction of their magnitude",
         f"{100 * pooled_var / (float(np.abs(pooled).mean()) + 1e-12):.1f} %"],
        ["fc1 units never active", f"{dead_units} of {pre_act.shape[1]}"],
        ["post-ReLU values that are zero", f"{100 * hidden_all_zero:.1f} %"],
        ["fc2 bias", np.array2string(head.fc2.bias.detach().float().cpu().numpy(),
                                     precision=3)],
    ]
    print()
    print(format_table(rows, ["measurement", "value"]))

    print("\nprediction histogram (what the head actually says):")
    for cls in range(cfg.model.n_classes):
        n_true = int((labels == cls).sum())
        n_pred = int((preds == cls).sum())
        print(f"  N={cfg.data.n_list[cls]}   true {n_true:4d}   predicted {n_pred:4d}")

    # ------------------------------------------------------- precision divergence
    if args.compare_precision:
        if device.type != "cuda":
            print()
            print("--compare_precision needs a GPU; autocast does nothing on CPU.")
        else:
            captured.clear()
            handles = [head.fc1.register_forward_hook(grab("fc1")),
                       head.fc2.register_forward_hook(grab("fc2"))]
            amp_preds = []
            with torch.no_grad():
                for batch in batches:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        out = model(batch["mix"].to(device))
                    amp_preds.append(out["count_logits"].argmax(dim=-1).cpu().numpy())
            for handle in handles:
                handle.remove()
            a_pooled = torch.cat([a for a, _ in captured["fc1"]]).float().cpu().numpy()
            a_logits = torch.cat([b for _, b in captured["fc2"]]).float().cpu().numpy()
            amp_preds = np.concatenate(amp_preds)

            banner("precision divergence")
            spread = max(1e-9, logit_spread_per_sample)
            rows = [
                ["pooled features", f"max |delta| {np.abs(a_pooled - pooled).max():.3e}",
                 f"{100 * np.abs(a_pooled - pooled).max() / max(1e-9, np.abs(pooled).max()):.1f} % of max"],
                ["logits", f"max |delta| {np.abs(a_logits - logits).max():.3e}",
                 f"{100 * np.abs(a_logits - logits).max() / spread:.0f} % of within-sample spread"],
                ["predictions agreeing", f"{100 * float((amp_preds == preds).mean()):.1f} %", ""],
                ["accuracy fp32", f"{100 * float((preds == labels).mean()):.1f} %", ""],
                ["accuracy fp16", f"{100 * float((amp_preds == labels).mean()):.1f} %", ""],
            ]
            print(format_table(rows, ["quantity", "fp16 vs fp32", "relative"]))
            print()
            print("fp32 predicts:", {int(cfg.data.n_list[c]): int((preds == c).sum())
                                     for c in sorted(set(preds.tolist()))})
            print("fp16 predicts:", {int(cfg.data.n_list[c]): int((amp_preds == c).sum())
                                     for c in sorted(set(amp_preds.tolist()))})

    # ---------------------------------------------------------------- verdict
    banner("verdict")
    # Relative, not absolute. An earlier run measured pooled variation of 0.0058 against
    # a pooled magnitude of order 1 -- features that barely move between a one-speaker
    # mixture and a five-speaker one -- and the 1e-6 threshold sailed past it, reporting
    # "simply inaccurate" for a head whose inputs carry almost nothing.
    pooled_scale = float(np.abs(pooled).mean()) + 1e-12
    pooled_ratio = pooled_var / pooled_scale
    if pooled_ratio < 0.05:
        print(f"CONSTANT FEATURES: the pooled statistics vary by {100 * pooled_ratio:.1f} % of")
        print("their own magnitude across inputs, so there is almost nothing for any head to")
        print("separate. The problem is upstream of the head: whatever the separator encodes")
        print("about how many people are talking is not reaching the pooled statistics.")
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
