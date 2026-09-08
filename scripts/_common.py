"""Shared script plumbing: import path, store/bank construction, argument helpers.

Every script in this folder must run three ways without edits:

* ``python scripts/04_train.py ...`` from the repo root,
* ``!python /kaggle/working/speaker-count-separate/scripts/04_train.py ...`` from a notebook,
* ``from scripts._common import ...`` after the repo's ``src/`` is on ``sys.path``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


def resolve(path: str | None, base: str = REPO_ROOT) -> str | None:
    """Make a relative path absolute against the repo root."""
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


SPLIT_NAMES = ("train-100", "train-360", "dev", "test")
# Directories that are always leaves in a LibriMix tree. Descending into them means
# listing ~14k files for nothing, so the search prunes them.
_LEAF_DIRS = {"mix_clean", "mix_both", "mix_single", "noise", "metadata",
              *(f"s{i}" for i in range(1, 10))}


def _looks_like_libri2mix(path: str) -> bool:
    """True for a directory holding ``<split>/s1`` -- i.e. a ``wav8k/min`` root."""
    return any(os.path.isdir(os.path.join(path, split, "s1")) for split in SPLIT_NAMES)


def _looks_like_store(path: str) -> bool:
    """True for a packed store: a manifest plus at least one packed split."""
    if not os.path.exists(os.path.join(path, "manifest.json")):
        return False
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    return any(os.path.exists(os.path.join(path, name, "index.csv")) for name in entries)


def _search(roots: Sequence[str], predicate: Any, max_depth: int = 8) -> str | None:
    """Bounded top-down search for the first directory satisfying ``predicate``.

    Kaggle has changed its mount layout before -- datasets used to appear at
    ``/kaggle/input/<slug>`` and now arrive at ``/kaggle/input/datasets/<owner>/<slug>``.
    Rather than encode either shape, look for the *contents* we need. The walk is
    top-down and returns on the first hit, so it never descends into the 14k-file
    leaf directories underneath.
    """
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        base_depth = os.path.abspath(root).rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, _files in os.walk(root):
            if predicate(dirpath):
                return os.path.normpath(dirpath)
            depth = os.path.abspath(dirpath).count(os.sep) - base_depth
            if depth >= max_depth:
                dirnames[:] = []
            else:  # prune known leaves and hidden dirs, in place
                dirnames[:] = [d for d in sorted(dirnames)
                               if d not in _LEAF_DIRS and not d.startswith(".")]
    return None


def autodetect_libri2mix(hint: str | None = None) -> str | None:
    """Find ``.../Libri2Mix/wav8k/min``, wherever Kaggle decided to mount it."""
    for candidate in ([hint,
                       os.path.join(hint, "Libri2Mix", "wav8k", "min"),
                       os.path.join(hint, "wav8k", "min")] if hint else []):
        if candidate and os.path.isdir(candidate) and _looks_like_libri2mix(candidate):
            return os.path.normpath(candidate)
    return _search(["/kaggle/input", "/kaggle/working"], _looks_like_libri2mix)


def autodetect_store(hint: str | None = None) -> str | None:
    """Find a packed store (a directory containing ``manifest.json``)."""
    for candidate in ([hint, os.path.join(hint, "store")] if hint else []):
        if candidate and _looks_like_store(candidate):
            return os.path.normpath(candidate)
    for candidate in (os.path.join(REPO_ROOT, "store"), "/kaggle/working/store"):
        if _looks_like_store(candidate):
            return os.path.normpath(candidate)
    return _search(["/kaggle/input"], _looks_like_store)


def build_store_and_bank(store_root: str, split: str, *, mmap: bool = True,
                         noise_store: str | None = None,
                         noise_kinds: Sequence[str] | None = None,
                         noise_weights: Sequence[float] | None = None) -> tuple[Any, Any]:
    """Open a packed split and the noise bank that goes with it.

    Babble comes from the *same split*, so noise never borrows a speaker from another
    split -- and never a speaker the model is asked to separate.
    """
    from csnet.noise import ALL_KINDS, NoiseBank
    from csnet.pack import SourceStore, open_store

    store = SourceStore(store_root, split, mmap=mmap)
    real = None
    if noise_store:
        real = open_store(noise_store, "noise", mmap=mmap)
        if real is None:
            real = open_store(os.path.dirname(noise_store), "noise", mmap=mmap)
    else:
        real = open_store(store_root, "noise", mmap=mmap)
    bank = NoiseBank(store, real=real,
                     kinds=tuple(noise_kinds) if noise_kinds else ALL_KINDS,
                     weights=tuple(noise_weights) if noise_weights else None)
    return store, bank


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """``--store``, ``--seed`` and friends, shared by most scripts."""
    parser.add_argument("--store", default=None,
                        help="packed store root (auto-detected on Kaggle if omitted)")
    parser.add_argument("--noise_store", default=None,
                        help="optional separate store holding a real-noise corpus")
    parser.add_argument("--seed", type=int, default=72)
    return parser


def require_store(args: argparse.Namespace) -> str:
    """Resolve ``--store`` or fail with an actionable message."""
    store = autodetect_store(resolve(args.store))
    if store is None:
        raise SystemExit(
            "could not find a packed store.\n"
            "  Pass --store /path/to/store, or run scripts/00_pack_sources.py first.\n"
            "  On Kaggle, attach the dataset you saved from the data-building notebook.")
    return store


def banner(title: str, width: int = 74) -> None:
    """Print a section header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width, flush=True)


def use_agg() -> Any:
    """Configure matplotlib for headless figure writing and return pyplot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.3})
    return plt
