"""Shared test fixtures. Also importable without pytest, so each test file can run alone.

Every test builds against a tiny synthetic corpus created on the fly, so the suite runs in
under a minute with no dataset attached.
"""

from __future__ import annotations

import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
TOOLS_ROOT = os.path.join(REPO_ROOT, "tools")
for path in (SRC_ROOT, TOOLS_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

_CACHE: dict[str, object] = {}


def tiny_corpus(speakers: int = 12, utts: int = 5, seed: int = 7) -> str:
    """Build (once per process) a synthetic Libri2Mix tree and return its ``wav8k/min``."""
    key = f"corpus-{speakers}-{utts}-{seed}"
    if key in _CACHE:
        return str(_CACHE[key])
    from make_fake_libri2mix import build

    root = tempfile.mkdtemp(prefix="csnet-corpus-")
    build(root, speakers, utts, seed)
    path = os.path.join(root, "wav8k", "min")
    _CACHE[key] = path
    return path


def tiny_store(split: str = "train-100") -> str:
    """Pack the synthetic corpus into a store (once per process) and return its root."""
    key = "store"
    if key not in _CACHE:
        from csnet.pack import pack_split, write_manifest

        root = tempfile.mkdtemp(prefix="csnet-store-")
        corpus = tiny_corpus()
        manifest = {"splits": {}}
        for name in ("train-100", "dev", "test"):
            manifest["splits"][name] = pack_split(corpus, name, root, babble_frac=0.25,
                                                  seed=7, progress=False)
        write_manifest(root, manifest)
        _CACHE[key] = root
    return str(_CACHE[key])


def store_and_bank(split: str = "train-100"):
    """Open a packed split plus its noise bank."""
    from csnet.noise import NoiseBank
    from csnet.pack import SourceStore

    store = SourceStore(tiny_store(), split)
    return store, NoiseBank(store)


def run_checks(checks: dict) -> int:
    """Run ``{name: callable}`` and report; returns a POSIX exit code."""
    failures = 0
    for name, fn in checks.items():
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - a test runner wants every failure
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(checks) - failures}/{len(checks)} passed")
    return 1 if failures else 0
