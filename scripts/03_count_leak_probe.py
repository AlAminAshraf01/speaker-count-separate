#!/usr/bin/env python3
"""Phase 4 (leakage audit) + Phase 5 (naive predictor). CPU only, no GPU quota.

Run this BEFORE training a counting head, and put the output straight into the report --
it is two mandated sections for the price of one script.

    python scripts/03_count_leak_probe.py --store /kaggle/working/store \
        --recipes data/recipes_test.csv --out leak

Or point it at folders of rendered WAVs, the way the original probe worked::

    python scripts/03_count_leak_probe.py \
        --mix 2:/path/Libri2Mix/wav8k/min/test/mix_clean \
              3:/path/Libri3Mix/wav8k/min/test/mix_clean

Why it matters. LibriMix normalises every source to an independent loudness target drawn
from U(-33, -25) LUFS and then SUMS them, so mixture level rises by about 10*log10(N) dB.
In `min` mode the mixture is also truncated to the shortest of the N sources, so duration
falls as N rises. Either scalar alone predicts N far above chance. If your counting head
scores 95 % and this script scores 60 %, most of that accuracy is bookkeeping, not acoustics.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

from _common import add_common_args, banner, build_store_and_bank, autodetect_store, resolve


def collect_from_recipes(store, bank, recipes, seg_len, per_class, mitigated):
    """Feature rows from frozen recipes, with or without the mitigation applied."""
    from csnet.audio import rms_normalize
    from csnet.baselines import naive_count_features
    from csnet.mixing import render_recipe

    by_n = defaultdict(list)
    for recipe in recipes:
        by_n[int(recipe["n_src"])].append(recipe)

    rows, labels, speakers_per_mix = [], [], []
    for n in sorted(by_n):
        for recipe in by_n[n][:per_class]:
            if mitigated:
                out = render_recipe(recipe, store, bank, seg_len=seg_len)
                mix = rms_normalize(out["mix"])
            else:
                # the un-mitigated view: full length, no RMS normalisation
                full = [store.get(int(i)) for i in recipe["utt_idx"]]
                length = min(len(f) for f in full)
                mix = np.sum([f[:length] * (10.0 ** (g / 20.0))
                              for f, g in zip(full, recipe["gain_db"])], axis=0).astype("float32")
            rows.append(naive_count_features(mix))
            labels.append(n)
            speakers_per_mix.append([str(store.speaker_ids[int(i)]) for i in recipe["utt_idx"]])
    return rows, np.array(labels), speakers_per_mix


def collect_from_dirs(dirs, per_class, seg_seconds, mitigated):
    """Feature rows from folders of rendered mixtures, one folder per speaker count."""
    from csnet.audio import pad_or_trim, read_wav, rms_normalize
    from csnet.baselines import naive_count_features
    from csnet.constants import SR

    rng = np.random.default_rng(0)
    rows, labels = [], []
    seg_len = int(seg_seconds * SR)
    for n, directory in sorted(dirs.items()):
        names = sorted(f for f in os.listdir(directory) if f.lower().endswith((".wav", ".flac")))
        if len(names) > per_class:
            names = [names[i] for i in sorted(rng.choice(len(names), per_class, replace=False))]
        kept = 0
        for name in names:
            wave, _ = read_wav(os.path.join(directory, name), sr=SR)
            if mitigated:
                if wave.size < seg_len:
                    continue
                start = (wave.size - seg_len) // 2
                wave = rms_normalize(pad_or_trim(wave, seg_len, start))
            rows.append(naive_count_features(wave))
            labels.append(n)
            kept += 1
        print(f"  N={n}: {kept} mixtures from {directory}")
    return rows, np.array(labels), []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--split", default="test")
    ap.add_argument("--recipes", default="data/recipes_test.csv")
    ap.add_argument("--mix", nargs="+", default=None,
                    help="N:/path/to/dir_of_wavs pairs, instead of --store/--recipes")
    ap.add_argument("--per_class", type=int, default=400)
    ap.add_argument("--crop", type=float, default=3.0)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="leak")
    args = ap.parse_args()

    from csnet.baselines import ALL_FEATURES, features_to_matrix, probe_feature_groups
    from csnet.constants import SR
    from csnet.metrics import format_confusion
    from csnet.mixing import read_recipes
    from csnet.pack import SourceStore
    from csnet.utils import format_table, json_dump_atomic

    out_dir = resolve(args.out) or args.out
    os.makedirs(out_dir, exist_ok=True)
    report: dict = {}
    seg_len = int(args.crop * SR)

    banner("03 - count-label leakage audit + naive predictor")

    store = bank = None
    speakers_per_mix: list = []
    if args.mix:
        dirs = {}
        for spec in args.mix:
            n, path = spec.split(":", 1)
            dirs[int(n)] = path
        print("[1] RAW MIXTURES (as generated)")
        raw_rows, raw_y, _ = collect_from_dirs(dirs, args.per_class, args.crop, mitigated=False)
        print(f"\n[2] AFTER MITIGATION ({args.crop} s centre crop + RMS normalisation)")
        mit_rows, mit_y, _ = collect_from_dirs(dirs, args.per_class, args.crop, mitigated=True)
        report["source"] = {str(k): v for k, v in dirs.items()}
    else:
        store_root = autodetect_store(resolve(args.store))
        if store_root is None:
            raise SystemExit("pass --store, or use --mix N:/path/to/wavs")
        recipes_path = resolve(args.recipes) or args.recipes
        if not os.path.exists(recipes_path):
            raise SystemExit(f"no recipes at {recipes_path}; run 01_make_frozen_sets.py")
        store, bank = build_store_and_bank(store_root, args.split,
                                           noise_store=resolve(args.noise_store))
        recipes = read_recipes(recipes_path)
        print(f"store   : {store_root}\nrecipes : {recipes_path} ({len(recipes)} mixtures)")
        raw_rows, raw_y, speakers_per_mix = collect_from_recipes(
            store, bank, recipes, seg_len, args.per_class, mitigated=False)
        mit_rows, mit_y, _ = collect_from_recipes(
            store, bank, recipes, seg_len, args.per_class, mitigated=True)
        report["source"] = {"store": store_root, "recipes": recipes_path}

    keys = list(ALL_FEATURES)
    raw_X, _ = features_to_matrix(raw_rows, keys)
    mit_X, _ = features_to_matrix(mit_rows, keys)
    chance = 100.0 / len(set(raw_y.tolist()))

    # ------------------------------------------------------------ probes
    def run(label, X, y):
        print(f"\n{label}   (depth-{args.max_depth} tree, {args.folds}-fold CV, "
              f"chance {chance:.0f} %)")
        results = probe_feature_groups(X, y, keys, max_depth=args.max_depth,
                                       n_folds=args.folds)
        for name, res in results.items():
            flag = "   <-- LEAK" if res["accuracy"] > res["chance"] * 1.5 else ""
            print(f"    {name:<40s} {res['accuracy'] * 100:5.1f} %{flag}")
        return {k: {"accuracy": v["accuracy"], "chance": v["chance"],
                    "feature_importance": v["feature_importance"]}
                for k, v in results.items()}, results

    report["raw"], _ = run("[1] RAW MIXTURES (as generated)", raw_X, raw_y)
    print("\n  per-class artefact statistics -- this is what the tree is reading:")
    stats_rows = []
    for n in sorted(set(raw_y.tolist())):
        mask = raw_y == n
        stats_rows.append([n, int(mask.sum()),
                           round(float(raw_X[mask, keys.index("duration")].mean()), 2),
                           round(float(raw_X[mask, keys.index("rms_db")].mean()), 2),
                           round(float(raw_X[mask, keys.index("rms_db")].std()), 2),
                           round(float(raw_X[mask, keys.index("crest")].mean()), 2),
                           round(float(raw_X[mask, keys.index("kurtosis")].mean()), 2)])
    print(format_table(stats_rows, ["N", "n", "duration s", "level dB", "level sd",
                                    "crest dB", "kurtosis"]))
    report["raw_class_stats"] = stats_rows

    report["mitigated"], mit_results = run(
        f"[2] AFTER MITIGATION ({args.crop} s crop + RMS normalisation)", mit_X, mit_y)

    everything = mit_results["everything"]
    print(f"\n  confusion matrix after mitigation (rows = true N):")
    print(format_confusion(everything["confusion"], everything["labels"]))
    report["mitigated_confusion"] = everything["confusion"].tolist()
    report["naive_baseline_accuracy"] = float(everything["accuracy"])
    report["chance"] = float(everything["chance"])

    # ------------------------------------------------------------ speaker audits
    banner("speaker-disjointness audit")
    audit_rows = []
    if speakers_per_mix:
        clashes = sum(1 for s in speakers_per_mix if len(set(s)) != len(s))
        audit_rows.append(["same speaker twice inside a mixture", clashes,
                           "OK" if clashes == 0 else "LEAK"])
    try:
        sets = {}
        for split in ("train-100", "dev", "test"):
            try:
                sets[split] = set(SourceStore(
                    autodetect_store(resolve(args.store)), split).speaker_ids.tolist())
            except (FileNotFoundError, TypeError):
                pass
        names = list(sets)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                overlap = len(sets[names[i]] & sets[names[j]])
                audit_rows.append([f"speaker in both {names[i]} and {names[j]}", overlap,
                                   "OK" if overlap == 0 else "LEAK"])
    except Exception:
        pass
    if store is not None:
        target = {str(s) for s in store.speaker_ids[store.target_idx].tolist()}
        babble = {str(s) for s in store.speaker_ids[store.babble_idx].tolist()}
        overlap = len(target & babble)
        audit_rows.append(["babble noise reuses a target speaker", overlap,
                           "OK" if overlap == 0 else "LEAK"])
    if audit_rows:
        print(format_table(audit_rows, ["hazard", "count", "verdict"]))
        report["speaker_audit"] = audit_rows

    # ------------------------------------------------------------ verdict
    banner("what to write in the report")
    raw_artefact = report["raw"].get("ARTEFACT only (duration + level)", {}).get("accuracy", 0.0)
    mitigated_all = report["naive_baseline_accuracy"]
    print(f"* Before mitigation, two scalars with no acoustics in them "
          f"({raw_artefact * 100:.0f} % vs {chance:.0f} % chance) recover the count.")
    print(f"* After a fixed {args.crop:g} s crop and RMS normalisation the same tree gets "
          f"{mitigated_all * 100:.1f} %.")
    print(f"* Report {mitigated_all * 100:.1f} % as the NAIVE BASELINE the counting head "
          "must beat.")
    if raw_artefact > (chance / 100.0) * 1.5:
        print("* The artefact row is well above chance, so the counter MUST be trained and")
        print("  evaluated on fixed-length, RMS-normalised input -- which is what")
        print("  csnet.mixing.render_recipe does by construction.")
    print("* Hazard avoided: LibriCount is built from LibriSpeech test-clean, the same")
    print("  speakers as our test split, so it was not used for pre-training or validation.")

    json_dump_atomic(report, os.path.join(out_dir, "leak_report.json"))
    print(f"\nreport -> {os.path.join(out_dir, 'leak_report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
