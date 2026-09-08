"""Small shared helpers: seeding, logging, timing, atomic JSON, device selection."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Iterator

import numpy as np


def stable_hash(text: str) -> int:
    """A process-stable 32-bit hash of a string.

    Python randomises ``hash()`` on strings per process via PYTHONHASHSEED, so using it to
    derive a seed makes "deterministic given a seed" quietly false -- two runs of the same
    command produce different data. CRC32 is stable across processes, machines and versions.
    """
    import zlib

    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


def seed_everything(seed: int = 72, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (cpu + cuda)."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:  # torch is optional for the CPU-only analysis scripts
        pass


def get_logger(name: str = "csnet", level: int = logging.INFO) -> logging.Logger:
    """Return a logger that prints once, even when a notebook re-imports the module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                                               datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


class Timer:
    """Context manager measuring wall time. ``with Timer() as t: ...; t.seconds``."""

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.seconds = 0.0
        self._t0 = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._t0


def human_time(seconds: float) -> str:
    """Format a duration as ``1h 02m 03s``."""
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def sizeof_fmt(num_bytes: float) -> str:
    """Format a byte count as a human-readable string."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def count_parameters(module: Any, trainable_only: bool = True) -> int:
    """Count parameters in a torch module."""
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(int(p.numel()) for p in params)


def pick_device(prefer: str | None = None) -> Any:
    """Return a torch device, preferring CUDA when available."""
    import torch

    if prefer and prefer != "auto":
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> str:
    """Create ``path`` (a directory) if needed and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def json_dump_atomic(obj: Any, path: str, indent: int = 2) -> None:
    """Write JSON via a temporary file + os.replace, so a kill never truncates it."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
    os.replace(tmp, path)


def json_load(path: str) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def set_worker_seed(worker_id: int) -> None:
    """``worker_init_fn`` for torch DataLoader: give every worker a distinct stream."""
    import torch

    base = int(torch.initial_seed()) % (2 ** 31)
    np.random.seed((base + worker_id) % (2 ** 32))
    random.seed(base + worker_id)


def chunked(seq: list, size: int) -> Iterator[list]:
    """Yield ``seq`` in chunks of at most ``size``."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def format_table(rows: list[list[Any]], header: list[str], floatfmt: str = "{:g}") -> str:
    """Render a plain-text table (no third-party tabulate dependency)."""
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return floatfmt.format(value)
        return str(value)

    body = [[cell(v) for v in row] for row in rows]
    widths = [len(h) for h in header]
    for row in body:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    rule = "  ".join("-" * w for w in widths)
    out = [line, rule]
    out += ["  ".join(v.ljust(widths[i]) for i, v in enumerate(row)) for row in body]
    return "\n".join(out)
