#!/usr/bin/env python3
"""Stage 2 -- freeze the dev and test sets as recipe CSVs.

The LibriMix authors' rule is that a test set "shouldn't be changed under any
circumstance". Ours is a CSV of recipes plus a seed, so it is small enough to commit to
git and it re-renders bit-for-bit. Generate it once, commit it, never regenerate it.

    python scripts/01_make_frozen_sets.py --store /kaggle/working/store --out data

Add ``--render_wav <dir>`` to also dump listenable WAVs -- useful for the report, for
the leak probe's ``--mix`` mode, and for checking with your own ears that a 5-speaker
mixture really does contain five people.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from _common import add_common_args, banner, build_store_and_bank, require_store, resolve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--out", default="data", help="directory for recipes_<split>.csv")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--n_list", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--n_per_class", type=int, default=300,
                    help="mixtures per speaker count per split")
    ap.add_argument("--seg_seconds", type=float, default=3.0)
    ap.add_argument("--p_clean", type=float, default=0.25,
                    help="fraction with no noise at all")
    ap.add_argument("--snr_db", nargs=2, type=float, default=[0.0, 20.0])
    ap.add_argument("--gain_db", nargs=2, type=float, default=[-5.0, 5.0])
    ap.add_argument("--render_wav", default=None, help="also write WAVs to this directory")
    ap.add_argument("--render_limit", type=int, default=40,
                    help="how many mixtures per split to render as WAV")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from csnet.audio import write_wav
    from csnet.constants import SR
    from csnet.mixing import (read_recipes, render_recipe, sample_recipe, write_recipes)
    from csnet.utils import format_table

    store_root = require_store(args)
    out_dir = resolve(args.out) or args.out
    os.makedirs(out_dir, exist_ok=True)
    seg_len = int(round(float(args.seg_seconds) * SR))

    banner("01 - freeze the evaluation sets")
    print(f"store   : {store_root}")
    print(f"out     : {out_dir}")
    print(f"segment : {args.seg_seconds:.1f} s ({seg_len} samples), RMS-normalised")
    print(f"seed    : {args.seed}   n_list: {args.n_list}   per class: {args.n_per_class}")

    summary_rows = []
    for split in args.splits:
        path = os.path.join(out_dir, f"recipes_{split}.csv")
        if os.path.exists(path) and not args.force:
            rows = read_recipes(path)
            print(f"\n[{split}] {len(rows)} recipes already frozen at {path} -- keeping them.")
            print("         Delete the file or pass --force if you really mean to regenerate.")
        else:
            store, bank = build_store_and_bank(store_root, split, noise_store=resolve(args.noise_store))
            print(f"\n[{split}] {len(store)} utterances / {len(store.speakers)} target "
                  f"speakers / {bank.describe()}")
            rng = np.random.default_rng([int(args.seed), abs(hash(split)) % (2 ** 31)])
            rows = []
            for n_src in args.n_list:
                if n_src > len(store.speakers):
                    print(f"         n={n_src}: SKIPPED, only {len(store.speakers)} speakers")
                    continue
                for k in range(int(args.n_per_class)):
                    rows.append(sample_recipe(
                        store, bank, n_src, rng, seg_len=seg_len,
                        gain_db_range=tuple(args.gain_db), snr_db_range=tuple(args.snr_db),
                        p_clean=float(args.p_clean),
                        mix_id=f"{split}_n{n_src}_{k:05d}"))
            write_recipes(rows, path)
            print(f"         wrote {len(rows)} recipes -> {path}")

        store, bank = build_store_and_bank(store_root, split, noise_store=resolve(args.noise_store))
        per_n: dict[int, int] = {}
        noisy = 0
        for r in rows:
            per_n[r["n_src"]] = per_n.get(r["n_src"], 0) + 1
            noisy += int(r["noise_kind"] != "none")
        summary_rows.append([split, len(rows), dict(sorted(per_n.items())),
                             f"{100.0 * noisy / max(1, len(rows)):.0f} %"])

        # verify the invariant on a random sample instead of trusting it
        rng = np.random.default_rng(0)
        picks = rng.choice(len(rows), size=min(20, len(rows)), replace=False)
        worst = 0.0
        speaker_clashes = 0
        for i in picks:
            recipe = rows[int(i)]
            out = render_recipe(recipe, store, bank, seg_len=seg_len)
            worst = max(worst, float(np.abs(out["mix"] - (out["sources"].sum(0)
                                                          + out["noise"])).max()))
            speakers = [str(store.speaker_ids[j]) for j in recipe["utt_idx"]]
            speaker_clashes += int(len(set(speakers)) != len(speakers))
        status = "OK" if worst < 1e-5 and speaker_clashes == 0 else "FAILED"
        print(f"         check: max|mix - (sum(s)+noise)| = {worst:.2e}, "
              f"same-speaker-twice = {speaker_clashes}   {status}")
        if status == "FAILED":
            raise SystemExit("frozen set failed its own invariant -- do not train on this")

        if args.render_wav:
            wav_dir = os.path.join(resolve(args.render_wav) or args.render_wav, split)
            # Stride through the list rather than taking the first K: recipes are grouped
            # by N, so the first K would all be 1-speaker mixtures.
            stride = max(1, len(rows) // max(1, int(args.render_limit)))
            count = 0
            for recipe in rows[::stride]:
                if count >= int(args.render_limit):
                    break
                out = render_recipe(recipe, store, bank, seg_len=seg_len)
                base = os.path.join(wav_dir, f"n{out['n_src']}", out["mix_id"])
                write_wav(base + "_mix.wav", out["mix"] / (np.abs(out["mix"]).max() + 1e-9) * 0.9, SR)
                for k, source in enumerate(out["sources"]):
                    write_wav(f"{base}_s{k + 1}.wav",
                              source / (np.abs(out["mix"]).max() + 1e-9) * 0.9, SR)
                count += 1
            print(f"         rendered {count} mixtures as WAV -> {wav_dir}")

    banner("summary")
    print(format_table(summary_rows, ["split", "mixtures", "per N", "noisy"]))
    print("\nCOMMIT recipes_dev.csv and recipes_test.csv to git. They are the frozen "
          "evaluation protocol;\nregenerating them later invalidates every number you "
          "have already reported.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
