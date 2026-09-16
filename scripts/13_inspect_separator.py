#!/usr/bin/env python3
"""Ask a trained checkpoint why it is not separating.

A separator stuck near +1 dB SI-SDRi after forty epochs is not "converging slowly". On a
single fixed batch this architecture reaches 13 dB in fifty steps and 28 dB in two
hundred and fifty, with nothing changed but the loss weights -- so the machinery works and
something is stopping it. There are only a few candidates, and they are all visible in one
forward pass:

* **dead mask channels** -- ``mask_act: relu`` zeroes any channel whose pre-activation is
  negative, and a zeroed ReLU receives no gradient ever again. The counting head in this
  project died exactly this way. If the silence term pushed the mask head down hard enough
  early on, a fraction of the mask is gone permanently and capacity went with it.
* **collapsed output scale** -- SI-SDR is scale-invariant but the silence penalty is not,
  so shrinking *every* slot drives the penalty to zero at no cost to the separation term.
  A model that took that route has quiet, undifferentiated slots.
* **undifferentiated slots** -- every slot emitting roughly the mixture, which is what an
  untrained mask head does and what a collapsed one falls back to.
* **none of the above** -- the slots differ, the masks are alive, and the model is simply
  not accurate enough yet, which is a budget question rather than a broken one.

    python scripts/13_inspect_separator.py --ckpt /kaggle/working/ckpt_count/best.pt \\
        --store /kaggle/input/.../store --recipes_dev /kaggle/input/.../recipes_dev.csv

CPU is fine and takes about a minute. Nothing is written.
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
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    import torch

    from csnet.checkpoint import load_checkpoint
    from csnet.config import dict_to_cfg, seg_len
    from csnet.datasets import FrozenMixDataset, build_loader
    from csnet.losses import pairwise_si_sdr
    from csnet.model import build_model
    from csnet.utils import format_table

    store_root = require_store(args)
    ckpt_path = resolve(args.ckpt) or args.ckpt

    banner("13 - why is the separator not separating")
    print(f"code  : {code_version()}")
    print(f"ckpt  : {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict_to_cfg(state["cfg"]) if "cfg" in state else None
    if cfg is None:
        raise SystemExit("checkpoint has no config; cannot rebuild the model")
    print(f"epoch : {state.get('epoch')}   step: {state.get('global_step')}")
    print(f"mask  : {cfg.model.mask_act}   slots: {cfg.model.max_n_src} + "
          f"{int(cfg.model.predict_noise)} noise")

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
    loader = build_loader(dev, batch_size=args.batch_size, shuffle=True, num_workers=0,
                          pin_memory=False)

    max_n = cfg.model.max_n_src
    alive_channels = None            # (slots, n_filters) bool: ever non-zero
    zero_fraction, slot_db, slot_sisdr, enc_alive = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            out = model(batch["mix"], return_internals=True)
            masks = out["masks"]                                  # (B, S, N, F)
            est, mix, refs = out["est"], batch["mix"], batch["refs"]

            ever = (masks.abs() > 0).any(dim=0).any(dim=-1)       # (S, N)
            alive_channels = ever if alive_channels is None else (alive_channels | ever)
            zero_fraction.append(float((masks == 0).float().mean()))
            enc_alive.append(float((out["enc"] > 0).float().mean()))

            mix_power = mix.pow(2).mean(dim=-1, keepdim=True)
            slot_db.append((10.0 * torch.log10(
                est.pow(2).mean(dim=-1) / (mix_power + 1e-8) + 1e-8)).numpy())

            # Best SI-SDR each slot achieves against any true source in its own mixture:
            # a slot that has learned a speaker scores well on one; a slot emitting the
            # mixture scores mediocre on all of them.
            pw = pairwise_si_sdr(est[:, :max_n], refs)            # (B, R, S)
            n_src = batch["n_src"]
            for b in range(est.shape[0]):
                slot_sisdr.append(pw[b, :int(n_src[b])].max(dim=0).values.numpy())

    slot_db_arr = np.concatenate(slot_db, axis=0)                 # (samples, slots)
    slot_sisdr_arr = np.stack(slot_sisdr)                         # (samples, max_n)
    dead = int((~alive_channels).sum())
    total = int(alive_channels.numel())

    rows = [
        ["mixtures measured", f"{slot_db_arr.shape[0]}"],
        ["mask channels never active", f"{dead} of {total}  ({100 * dead / total:.1f} %)"],
        ["mask values that are zero", f"{100 * float(np.mean(zero_fraction)):.1f} %"],
        ["encoder units active", f"{100 * float(np.mean(enc_alive)):.1f} %"],
        ["slot level vs mixture (mean)", f"{slot_db_arr.mean():+.1f} dB"],
        ["slot level spread across slots", f"{slot_db_arr.mean(axis=0).std():.2f} dB"],
        ["best SI-SDR any slot reaches", f"{slot_sisdr_arr.max(axis=1).mean():+.2f} dB"],
    ]
    print()
    print(format_table(rows, ["measurement", "value"]))

    print("\nper speaker slot:")
    per_slot = [[s, f"{slot_db_arr[:, s].mean():+.1f} dB",
                 f"{slot_sisdr_arr[:, s].mean():+.2f} dB",
                 f"{int((~alive_channels[s]).sum())} of {alive_channels.shape[1]}"]
                for s in range(max_n)]
    if cfg.model.predict_noise:
        per_slot.append(["noise", f"{slot_db_arr[:, max_n].mean():+.1f} dB", "-",
                         f"{int((~alive_channels[max_n]).sum())} of {alive_channels.shape[1]}"])
    print(format_table(per_slot, ["slot", "level vs mix", "best SI-SDR", "dead channels"]))

    # ---------------------------------------------------------------- verdict
    banner("verdict")
    dead_frac = dead / max(1, total)
    mean_db = float(slot_db_arr.mean())
    spread = float(slot_db_arr.mean(axis=0).std())
    reach = float(slot_sisdr_arr.max(axis=1).mean())

    if dead_frac > 0.5:
        print(f"DEAD MASK: {100 * dead_frac:.0f} % of mask channels never activate for any")
        print("input. ReLU zeroes them and a zeroed ReLU gets no gradient, so that capacity")
        print("is gone permanently -- the same way the counting head died. Retraining the")
        print("separator from these weights cannot recover it; the mask activation or the")
        print("pressure that pushed it down has to change first.")
    elif mean_db < -25.0:
        print(f"COLLAPSED SCALE: every slot sits {mean_db:.0f} dB below the mixture. SI-SDR is")
        print("scale-invariant, so shrinking all slots costs the separation term nothing and")
        print("drives the silence penalty to zero -- a free descent direction that teaches")
        print("the model nothing about which slot should be quiet.")
    elif spread < 0.5 and reach < 3.0:
        print("UNDIFFERENTIATED SLOTS: every slot emits about the same thing at about the")
        print("same level, and none of them matches a source better than the mixture does.")
        print("The mask head has not specialised at all.")
    else:
        print(f"The slots differ ({spread:.2f} dB apart) and the best one reaches "
              f"{reach:+.2f} dB")
        print("against a true source, so the separator is alive and simply not accurate")
        print("enough. That is a budget and objective-weighting question, not a broken")
        print("model: compare against a short fixed-N=2 run before changing the architecture.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
