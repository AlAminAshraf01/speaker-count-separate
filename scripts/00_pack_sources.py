#!/usr/bin/env python3
"""Stage 1 -- pack Libri2Mix's isolated sources into a flat int16 store.

Run this ONCE, on a **CPU** notebook (it spends no GPU quota), then Save Version and attach
the output as a dataset to every later notebook.

    python scripts/00_pack_sources.py \
        --libri2mix_dir /kaggle/input/libri2mix-8khz-min/Libri2Mix/wav8k/min \
        --out /kaggle/working/store

Roughly 8-12 minutes for the full corpus; ~3.5 GB out, versus ~32 GB if you rendered
Libri3/4/5Mix to disk instead. Restartable: an already-packed split is skipped unless
``--force`` is given.

Optionally pack a real-noise corpus too (ESC-50, UrbanSound8K, WHAM, anything)::

    ... --noise_dir /kaggle/input/environmental-sound-classification-50/audio
"""

from __future__ import annotations

import argparse
import os
import sys

from _common import add_common_args, autodetect_libri2mix, banner, resolve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--libri2mix_dir", default=None,
                    help="path to .../Libri2Mix/wav8k/min (auto-detected on Kaggle)")
    ap.add_argument("--out", default="/kaggle/working/store", help="store root to create")
    ap.add_argument("--splits", nargs="+", default=["train-100", "dev", "test"])
    ap.add_argument("--babble_frac", type=float, default=0.2,
                    help="fraction of speakers reserved for babble noise only")
    ap.add_argument("--cap_seconds", type=float, default=8.0,
                    help="max stored excerpt per utterance (highest-energy window)")
    ap.add_argument("--limit", type=int, default=None, help="utterances per split (smoke tests)")
    ap.add_argument("--noise_dir", default=None, help="optional real-noise corpus to pack")
    ap.add_argument("--noise_limit", type=int, default=4000)
    ap.add_argument("--force", action="store_true", help="repack splits that already exist")
    ap.add_argument("--seed", type=int, default=72)
    args = ap.parse_args()

    from csnet.constants import SR
    from csnet.pack import (SourceStore, pack_noise_dir, pack_split, read_manifest,
                            split_is_packed, write_manifest)
    from csnet.utils import format_table, human_time, sizeof_fmt

    import time

    libri = autodetect_libri2mix(resolve(args.libri2mix_dir))
    if libri is None:
        raise SystemExit(
            "could not find Libri2Mix.\n"
            "  Expected a folder containing train-100/, dev/, test/ with s1/ and s2/ inside.\n"
            "  On Kaggle: + Add Input -> Datasets -> 'libri2mix-8khz-min' (by unconscious).\n"
            "  The mount point differs between Kaggle layouts, so the search looks for the\n"
            "  contents rather than a fixed path. If it still fails, locate it yourself:\n"
            "    find /kaggle/input -maxdepth 7 -type d -name s1\n"
            "  and pass the directory two levels above s1 as --libri2mix_dir.")

    out = resolve(args.out) or args.out
    os.makedirs(out, exist_ok=True)
    cap_len = int(round(float(args.cap_seconds) * SR))

    banner("00 - pack sources")
    print(f"source : {libri}")
    print(f"store  : {out}")
    print(f"cap    : {args.cap_seconds:.1f} s per utterance   babble speakers: "
          f"{args.babble_frac * 100:.0f} %")

    manifest = read_manifest(out)
    manifest.update({"sr": SR, "cap_len": cap_len, "seed": int(args.seed),
                     "source_dataset": libri, "babble_frac": float(args.babble_frac),
                     "noise_dir": resolve(args.noise_dir)})
    manifest.setdefault("splits", {})

    start = time.time()
    for split in args.splits:
        if split_is_packed(out, split) and not args.force:
            print(f"\n[{split}] already packed -- skipping (use --force to redo)")
            if split not in manifest["splits"]:
                manifest["splits"][split] = SourceStore(out, split).summary()
            continue
        print(f"\n[{split}] packing ...")
        manifest["splits"][split] = pack_split(
            libri, split, out, cap_len=cap_len, babble_frac=float(args.babble_frac),
            seed=int(args.seed), limit=args.limit)
        write_manifest(out, manifest)

    if args.noise_dir:
        noise_dir = resolve(args.noise_dir)
        if split_is_packed(out, "noise") and not args.force:
            print("\n[noise] already packed -- skipping")
        else:
            print(f"\n[noise] packing from {noise_dir}")
            manifest["noise"] = pack_noise_dir(noise_dir, out, cap_len=cap_len,
                                               seed=int(args.seed), limit=args.noise_limit)
    write_manifest(out, manifest)

    banner("summary")
    rows = []
    total_bytes = 0
    for split in args.splits:
        store = SourceStore(out, split)
        info = store.summary()
        size = int(store.lengths.sum()) * 2
        total_bytes += size
        rows.append([split, info["n_utt"], info["n_speakers"], info["n_babble_speakers"],
                     round(info["hours"], 2), round(info["median_utt_seconds"], 2),
                     sizeof_fmt(size)])
    print(format_table(rows, ["split", "utts", "speakers", "babble spk", "hours",
                              "median s", "size"]))

    # The audit that matters most: speakers must not cross a split boundary.
    sets = {s: set(SourceStore(out, s).speaker_ids.tolist()) for s in args.splits}
    print("\nspeaker disjointness across splits (must all be 0):")
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = len(sets[names[i]] & sets[names[j]])
            flag = "OK" if overlap == 0 else "LEAK"
            print(f"  {names[i]:>10} & {names[j]:<10} shared speakers: {overlap:>4}  {flag}")

    print(f"\ntotal packed audio: {sizeof_fmt(total_bytes)}   "
          f"elapsed {human_time(time.time() - start)}")
    print(f"manifest: {os.path.join(out, 'manifest.json')}")
    print("\nNEXT: scripts/01_make_frozen_sets.py, then 'Save Version' to publish this "
          "store as a dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
