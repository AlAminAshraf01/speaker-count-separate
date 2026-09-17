#!/usr/bin/env python3
"""Answer "will this session survive?" in thirty seconds, before it costs eleven hours.

Every expensive failure in this project so far was visible in the first minute and nobody
looked. A notebook ran cells imported two pushes ago. A run resumed from the five-step
dry-run checkpoint. A training loop was handed ``range(38, 12)`` and trained nothing. A
session filled 30 GiB of host RAM four hours in. None of those needed a GPU to detect.

So: one cell, at the top of every long notebook, that checks the things that have actually
gone wrong and refuses to continue when one of them has.

    python scripts/12_preflight.py --for recover --config configs/paper.yaml \\
        --store /kaggle/input/.../store --ckpt_dir /kaggle/working/ckpt_count \\
        --extra_epochs 12 --cells_src kaggle_02_train.py --cells_sha a1b2c3d4e5f6a7b8

Exit code is 1 when something is a STOP, 0 otherwise, so ``run(...)`` in the notebook
halts the session before the expensive cell rather than after it. Nothing is written and
no GPU work is done, so it costs no quota worth counting.

Every check is wrapped: a check that raises is reported as a warning, never as a crash.
A preflight that can take down the notebook is a worse bug than the ones it finds.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

from _common import (add_common_args, autodetect_store, banner, code_version,
                     rank_checkpoints, resolve)

OK, WARN, STOP = "OK", "WARN", "STOP"

# What each notebook needs, so one script covers all of them.
#   gpu: "want" = a GPU is the point | "none" = a GPU here burns quota | "any" = no opinion
#   ckpt: a trained checkpoint must already exist
#   recipes: which frozen recipe file has to be visible
PROFILES: dict[str, dict] = {
    "data":      {"gpu": "none", "ckpt": False, "recipes": None},
    "eda":       {"gpu": "none", "ckpt": False, "recipes": "dev"},
    "train":     {"gpu": "want", "ckpt": False, "recipes": "dev"},
    "recover":   {"gpu": "want", "ckpt": True, "recipes": "dev"},
    "search":    {"gpu": "want", "ckpt": False, "recipes": "dev"},
    "evaluate":  {"gpu": "want", "ckpt": True, "recipes": "test"},
    "interpret": {"gpu": "want", "ckpt": True, "recipes": "test"},
    "demo":      {"gpu": "any", "ckpt": True, "recipes": None},
}

Row = tuple[str, str, str, str]  # name, level, detail, fix


# ------------------------------------------------------------------------------- checks

def check_code(_args, _state) -> Row:
    return "code", OK, code_version(), ""


def check_cells(args, _state) -> Row:
    """Are these notebook cells as new as the repo they just cloned?"""
    if not args.cells_src or not args.cells_sha:
        return ("cells", WARN, "not stamped -- staleness cannot be checked",
                "re-import the notebook from notebooks/ to enable this")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(here, "notebooks", "src")
    sys.path.insert(0, os.path.join(here, "tools"))
    from build_notebooks import cells_fingerprint

    current = cells_fingerprint(src_dir, args.cells_src)
    if current == args.cells_sha:
        return "cells", OK, f"current ({current})", ""
    return ("cells", STOP, f"cells are {args.cells_sha}, repo is {current}",
            f"File -> Import Notebook -> notebooks/{args.cells_src[:-3]}.ipynb, "
            "then re-attach the inputs")


def check_gpu(args, state) -> Row:
    import torch

    want = PROFILES[args.profile]["gpu"]
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    state["n_gpu"] = n
    if n:
        props = torch.cuda.get_device_properties(0)
        detail = f"{n} x {props.name}, {props.total_memory / 1e9:.1f} GB each"
    else:
        detail = "none (CPU only)"
    if want == "want" and n == 0:
        return ("gpu", STOP, detail,
                "Settings -> Accelerator -> GPU T4 x2, then re-run")
    if want == "none" and n > 0:
        return ("gpu", WARN, detail,
                "this notebook needs no GPU; Accelerator -> None saves weekly quota")
    return "gpu", OK, detail, ""


def check_host_ram(_args, state) -> Row:
    from csnet.memory import cgroup_usage_gb, format_snapshot, unreclaimable_gb

    used, limit = cgroup_usage_gb()
    state["ram_limit"] = limit
    detail = format_snapshot().replace("ram ", "", 1).strip()
    if limit != limit:  # nan: no cgroup limit, so nothing to run out of
        return "host ram", OK, detail, ""
    if unreclaimable_gb() > 0.7 * limit:
        return ("host ram", STOP, detail,
                "most of the container's RAM is already taken before training starts; "
                "restart the session (Run -> Restart & Clear Cell Outputs)")
    return "host ram", OK, detail, ""


def check_disk(_args, _state) -> Row:
    """``/kaggle/working`` is 20 GB and a Save Version of a full one fails at the end."""
    work = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
    free_gb = shutil.disk_usage(work).free / 1e9
    detail = f"{free_gb:.1f} GB free in {work}"
    if free_gb < 2.0:
        return ("disk", STOP, detail,
                "delete old checkpoint directories in /kaggle/working before running")
    if free_gb < 5.0:
        return ("disk", WARN, detail, "a Save Version needs room for the whole output")
    return "disk", OK, detail, ""


def check_store(args, state) -> Row:
    import json

    store = autodetect_store(resolve(args.store))
    state["store"] = store
    if store is None:
        if args.profile == "data":
            return "store", OK, "none yet -- this notebook is the one that builds it", ""
        # The demo notebook can run on a wav the user uploads, so a missing store there
        # is a smaller problem than a stopped session.
        level = WARN if args.profile == "demo" else STOP
        return ("store", level, "not found under /kaggle/input",
                "+ Add Input -> Notebook Output -> 00_build_dataset")
    try:
        with open(os.path.join(store, "manifest.json"), "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        splits = manifest.get("splits", manifest)
        parts = [f"{k} {v.get('n_utts', v.get('utts', '?'))}"
                 for k, v in splits.items() if isinstance(v, dict)]
        return "store", OK, f"{store} ({', '.join(parts) or 'no split table'})", ""
    except Exception as exc:
        return "store", WARN, f"{store} (manifest unreadable: {exc})", ""


def check_recipes(args, state) -> Row:
    which = PROFILES[args.profile]["recipes"]
    if which is None:
        return "recipes", OK, "not needed for this notebook", ""
    from _common import describe_recipes, find_recipes

    path = resolve(args.recipes_dev if which == "dev" else args.recipes_test)
    if not path or not os.path.exists(path):
        path = find_recipes(f"recipes_{which}.csv", state.get("store"))
    if not path:
        return ("recipes", STOP, f"recipes_{which}.csv not found",
                "+ Add Input -> Notebook Output -> 00_build_dataset")
    facts = describe_recipes(path)
    state[f"recipes_{which}"] = path
    if facts["rows"] <= 0:
        return ("recipes", STOP, f"{path} has no rows",
                "re-run notebook 00; the frozen set is empty")
    # The sha is what lets two runs of two notebooks be compared. Reading a different
    # frozen set produces a different answer for the same checkpoint and says nothing.
    return ("recipes", OK,
            f"{which}: {facts['rows']} mixtures {facts['per_n']} sha {facts['sha']}", "")


def check_checkpoint(args, state) -> Row:
    """Which checkpoint would be used -- and is it the real run or the dry run?"""
    needed = PROFILES[args.profile]["ckpt"]
    ranked = rank_checkpoints("best.pt") + rank_checkpoints("last.pt")
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    state["ckpt_rank"] = ranked
    if not ranked:
        if needed:
            return ("checkpoint", STOP, "none visible",
                    "+ Add Input -> Notebook Output -> the 02_train notebook")
        return "checkpoint", OK, "none (this run starts from scratch)", ""

    path, step = ranked[0]
    state["ckpt"] = path
    state["ckpt_step"] = step
    detail = f"{path} at step {step}"
    if step < 100:
        # A leftover dry run is only a problem for a notebook that needs a trained model.
        # A fresh training run is entitled to ignore it.
        return ("checkpoint", STOP if needed else WARN,
                detail + " -- that is a dry run, not a trained model",
                "attach the notebook output that holds the real checkpoint directory"
                if needed else "this run starts from scratch, which is fine")
    if len(ranked) > 1:
        detail += f" (chosen from {len(ranked)} by step, not by name)"
    return "checkpoint", OK, detail, ""


def check_config(args, state) -> Row:
    if not args.config:
        return "config", OK, "no config for this notebook", ""
    from csnet.config import load_cfg

    try:
        cfg = load_cfg(resolve(args.config) or args.config, list(args.overrides or []))
    except Exception as exc:
        # load_cfg validates as it goes (n_classes against n_list, and so on). A config
        # that will not load is a stopper with a readable reason, not a traceback eleven
        # cells later.
        return ("config", STOP, f"{type(exc).__name__}: {exc}",
                "fix the config or the --set override above before running")
    state["cfg"] = cfg
    notes = [f"batch {cfg.train.batch_size}", f"workers {cfg.train.num_workers}",
             f"amp {bool(cfg.train.amp)}"]
    if cfg.train.dataparallel:
        return ("config", STOP, "dataparallel is ON -- " + ", ".join(notes),
                "it leaked 17 MiB/step and ran 35% slower; set train.dataparallel=false")
    if cfg.train.pin_memory:
        return ("config", WARN, "pin_memory is ON -- " + ", ".join(notes),
                "pinned host memory is never returned to the OS")
    return "config", OK, "dataparallel off, " + ", ".join(notes), ""


def check_epochs(args, state) -> Row:
    """The arithmetic that produced ``range(38, 12)`` and a 70-second run reported as done."""
    cfg = state.get("cfg")
    if cfg is None or args.profile not in {"train", "recover"}:
        return "epochs", OK, "not a training run", ""
    from csnet.checkpoint import find_resume

    work = resolve(args.ckpt_dir) or cfg.train.ckpt_dir
    resume = find_resume("auto", work_dir=work, contains=args.resume_contains)
    start = 0
    if resume:
        step = dict(state.get("ckpt_rank") or []).get(resume, -1)
        start = _epoch_of(resume)
        state["resume"] = resume
        state["resume_step"] = step
    if args.extra_epochs is not None:
        target = start + max(1, int(args.extra_epochs))
        state["to_run"] = target - start
        return ("epochs", OK, f"{start} -> {target} ({target - start} to run, "
                "counted from the checkpoint)", "")
    target = int(args.epochs if args.epochs is not None else cfg.train.epochs)
    state["target_epochs"] = target
    state["to_run"] = max(0, target - start)
    if start >= target:
        return ("epochs", STOP,
                f"resuming at epoch {start} with a target of {target}: range({start}, "
                f"{target}) is empty, so nothing would train",
                f"use --extra_epochs N for N more epochs, or raise the target above {start}")
    return "epochs", OK, f"{start} -> {target} ({target - start} to run)", ""


def _epoch_of(path: str) -> int:
    """Epochs already finished, read from ``history.json`` if it is there."""
    import json

    history = os.path.join(os.path.dirname(path), "history.json")
    try:
        with open(history, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
        return int(rows[-1].get("epoch", 0)) if rows else 0
    except Exception:
        pass
    try:
        import torch

        return int(torch.load(path, map_location="cpu", weights_only=False).get("epoch", 0))
    except Exception:
        return 0


def check_budget(args, state) -> Row:
    """Hours this plan needs against the hours this session and this week have."""
    cfg = state.get("cfg")
    if cfg is None or args.profile not in {"train", "recover", "search"}:
        return "budget", OK, "not a training run", ""
    ms = args.ms_per_step or _measured_ms(state, cfg)
    if not ms:
        return ("budget", WARN, "no measured ms/step yet, so no estimate",
                "the trainer times 20 real steps before it commits; read that line")
    n_epochs = state.get("to_run") or 0
    hours = n_epochs * cfg.train.steps_per_epoch * ms / 3.6e6
    budget = float(args.time_budget_h or cfg.train.time_budget_h)
    detail = (f"{n_epochs} epochs x {cfg.train.steps_per_epoch} steps x {ms:.0f} ms "
              f"= {hours:.1f} h of a {budget:.1f} h budget")
    if hours > budget:
        return ("budget", WARN, detail,
                f"it will stop cleanly on the budget after about "
                f"{int(budget * 3.6e6 / (cfg.train.steps_per_epoch * ms))} epochs and "
                "print a RESUME block; that is normal, not a failure")
    return "budget", OK, detail, ""


def _measured_ms(state: dict, cfg) -> float | None:
    """Milliseconds per step from this run's own history, not from a guess."""
    import json

    resume = state.get("resume") or state.get("ckpt")
    if not resume:
        return None
    try:
        with open(os.path.join(os.path.dirname(resume), "history.json"),
                  "r", encoding="utf-8") as fh:
            rows = json.load(fh)
        recent = [r for r in rows[-3:] if r.get("seconds")]
        if not recent:
            return None
        seconds = sum(float(r["seconds"]) for r in recent) / len(recent)
        return 1000.0 * seconds / max(1, int(cfg.train.steps_per_epoch))
    except Exception:
        return None


CHECKS = (check_code, check_cells, check_gpu, check_host_ram, check_disk, check_store,
          check_recipes, check_checkpoint, check_config, check_epochs, check_budget)


# --------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--for", dest="profile", default="train", choices=sorted(PROFILES),
                    help="which notebook is about to run")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", nargs="*", default=None, metavar="KEY=VALUE")
    ap.add_argument("--recipes_dev", default=None)
    ap.add_argument("--recipes_test", default=None)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--resume_contains", default=None, metavar="SUBSTRING",
                    help="the same filter 04_train.py gets: which attached run this "
                         "one may resume from. Without it the epoch arithmetic below "
                         "is done against whichever attached checkpoint has the most "
                         "steps, which is not necessarily this experiment")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--extra_epochs", type=int, default=None)
    ap.add_argument("--time_budget_h", type=float, default=None)
    ap.add_argument("--ms_per_step", type=float, default=None)
    ap.add_argument("--cells_src", default=None, help="set by the notebook's stamp cell")
    ap.add_argument("--cells_sha", default=None, help="set by the notebook's stamp cell")
    ap.add_argument("--strict", action="store_true", help="treat warnings as stoppers too")
    args = ap.parse_args()

    banner(f"12 - preflight for: {args.profile}")
    state: dict = {}
    rows: list[Row] = []
    for check in CHECKS:
        try:
            rows.append(check(args, state))
        except Exception as exc:  # a broken check must not become a broken session
            rows.append((check.__name__.replace("check_", ""), WARN,
                         f"check itself failed: {type(exc).__name__}: {exc}", ""))

    width = max(len(r[0]) for r in rows)
    print()
    for name, level, detail, fix in rows:
        print(f"{name:<{width}}  {level:<4}  {detail}")
        if fix and level != OK:
            print(f"{'':<{width}}        -> {fix}")

    stoppers = [r for r in rows if r[1] == STOP]
    warnings = [r for r in rows if r[1] == WARN]
    print()
    if stoppers:
        print(f"PREFLIGHT: STOP -- {len(stoppers)} blocking problem(s) above.")
        print("Nothing has been spent. Fix them and run this cell again.")
        return 1
    if warnings and args.strict:
        print(f"PREFLIGHT: STOP -- {len(warnings)} warning(s) and --strict was passed.")
        return 1
    if warnings:
        print(f"PREFLIGHT: GO, with {len(warnings)} warning(s). Read them, then continue.")
    else:
        print("PREFLIGHT: GO.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
