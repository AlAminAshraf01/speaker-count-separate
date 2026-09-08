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


def autodetect_libri2mix(hint: str | None = None) -> str | None:
    """Find ``.../Libri2Mix/wav8k/min`` under a hint or the usual Kaggle input locations."""
    candidates: list[str] = []
    if hint:
        candidates += [hint,
                       os.path.join(hint, "Libri2Mix", "wav8k", "min"),
                       os.path.join(hint, "wav8k", "min")]
    for root in ("/kaggle/input",):
        if os.path.isdir(root):
            for name in sorted(os.listdir(root)):
                base = os.path.join(root, name)
                candidates += [os.path.join(base, "Libri2Mix", "wav8k", "min"),
                               os.path.join(base, "wav8k", "min")]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate) and any(
                os.path.isdir(os.path.join(candidate, s)) for s in ("train-100", "dev", "test")):
            return os.path.normpath(candidate)
    return None


def autodetect_store(hint: str | None = None) -> str | None:
    """Find a packed store (a directory containing ``manifest.json``)."""
    candidates = [hint] if hint else []
    if os.path.isdir("/kaggle/input"):
        for name in sorted(os.listdir("/kaggle/input")):
            base = os.path.join("/kaggle/input", name)
            candidates += [base, os.path.join(base, "store")]
    candidates += [os.path.join(REPO_ROOT, "store"), "/kaggle/working/store"]
    for candidate in candidates:
        if candidate and os.path.exists(os.path.join(candidate, "manifest.json")):
            return os.path.normpath(candidate)
    return None


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
