"""Resumable checkpointing and the Kaggle session clock.

Kaggle gives you a 12-hour session and then takes the machine away. Worse,
``/kaggle/working`` is wiped between sessions unless you **Save Version**, which turns the
output into a dataset you then attach to the next notebook. So "resume" here means three
different things and all of them are handled:

1. **Clean stop.** ``TimeBudget`` expires with a margin, training saves and exits 0, and
   the notebook's Save Version captures ``last.pt``.
2. **Hard kill** (browser closed, OOM, quota). Mid-epoch checkpoints every few hundred
   steps mean you lose minutes, not hours.
3. **Next session.** ``find_resume`` looks in ``/kaggle/working`` first, then searches every
   attached input dataset for the newest ``last.pt`` -- so re-attaching the previous
   notebook's output is the only manual step.

Every save is atomic (``.tmp`` then ``os.replace``): a kill during the write cannot leave a
truncated checkpoint behind.
"""

from __future__ import annotations

import glob
import os
import random
import time
from typing import Any

import numpy as np
import torch

from .utils import get_logger

LOG = get_logger(__name__)

CHECKPOINT_KEYS = ("model", "optimizer", "scheduler", "scaler", "epoch", "global_step",
                   "best_metric", "history", "cfg", "rng", "csnet_version", "wall_h")


class TimeBudget:
    """Wall-clock guard for a bounded session."""

    def __init__(self, hours: float, consumed_h: float = 0.0) -> None:
        self.hours = float(hours)
        self.consumed_h = float(consumed_h)
        self.start = time.time()

    def elapsed_h(self) -> float:
        """Hours spent in *this* process."""
        return (time.time() - self.start) / 3600.0

    def total_h(self) -> float:
        """Hours spent across every session of this run."""
        return self.consumed_h + self.elapsed_h()

    def remaining_h(self) -> float:
        """Hours left in this session's budget."""
        return max(0.0, self.hours - self.elapsed_h())

    def expired(self, margin_min: float = 20.0) -> bool:
        """True once less than ``margin_min`` minutes of budget remain."""
        return self.remaining_h() * 60.0 <= float(margin_min)

    def __repr__(self) -> str:
        return (f"TimeBudget(elapsed={self.elapsed_h():.2f}h, "
                f"remaining={self.remaining_h():.2f}h, total={self.total_h():.2f}h)")


def unwrap(model: Any) -> Any:
    """Strip DataParallel / DDP / torch.compile wrappers before saving state."""
    model = getattr(model, "_orig_mod", model)      # torch.compile
    model = getattr(model, "module", model)         # DataParallel / DDP
    return model


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict | None) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state["cuda"]])
    except Exception as exc:  # a changed torch version must not block a resume
        LOG.warning("could not restore RNG state (%s); continuing with a fresh one", exc)


def _as_byte_tensor(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    return tensor.cpu().to(torch.uint8)


def save_checkpoint(path: str, *, model: Any, optimizer: Any = None, scheduler: Any = None,
                    scaler: Any = None, epoch: int = 0, global_step: int = 0,
                    best_metric: float = float("-inf"), history: list | None = None,
                    cfg_dict: dict | None = None, extra: dict | None = None,
                    wall_h: float = 0.0) -> str:
    """Write a checkpoint atomically and return its path."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    from . import __version__

    payload = {
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "history": list(history or []),
        "cfg": dict(cfg_dict or {}),
        "rng": _rng_state(),
        "csnet_version": __version__,
        "wall_h": float(wall_h),
    }
    if extra:
        payload["extra"] = extra

    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str, *, model: Any = None, optimizer: Any = None,
                    scheduler: Any = None, scaler: Any = None,
                    map_location: str = "cpu", strict: bool = True,
                    restore_rng: bool = True) -> dict:
    """Load a checkpoint, optionally restoring model/optimizer/scheduler/scaler in place."""
    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only
        state = torch.load(path, map_location=map_location)

    if model is not None and state.get("model") is not None:
        unwrap(model).load_state_dict(state["model"], strict=strict)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(state["scheduler"])
        except Exception as exc:
            LOG.warning("scheduler state ignored (%s)", exc)
    if scaler is not None and state.get("scaler") is not None:
        try:
            scaler.load_state_dict(state["scaler"])
        except Exception as exc:
            LOG.warning("scaler state ignored (%s)", exc)
    if restore_rng:
        _restore_rng(state.get("rng"))
    return state


def find_resume(explicit: str | None = None, work_dir: str = "/kaggle/working/ckpt",
                search_inputs: bool = True,
                input_root: str = "/kaggle/input") -> str | None:
    """Locate a checkpoint to resume from.

    Order: an explicit path, then ``<work_dir>/last.pt``, then the newest ``last.pt``
    anywhere under the attached input datasets (that is the previous session's output).
    """
    if explicit and str(explicit).lower() not in {"none", "auto", ""}:
        return explicit if os.path.exists(explicit) else None

    local = os.path.join(work_dir, "last.pt")
    if os.path.exists(local):
        return local

    if search_inputs and os.path.isdir(input_root):
        candidates = glob.glob(os.path.join(input_root, "*", "**", "last.pt"), recursive=True)
        candidates = [c for c in candidates if os.path.isfile(c)]
        if candidates:
            return max(candidates, key=os.path.getmtime)
    return None


def keep_last_k(directory: str, k: int = 2, pattern: str = "epoch_*.pt") -> list[str]:
    """Delete all but the ``k`` newest per-epoch checkpoints. Returns what was removed."""
    files = sorted(glob.glob(os.path.join(directory, pattern)), key=os.path.getmtime)
    removed: list[str] = []
    for path in files[:-k] if k > 0 else files:
        try:
            os.remove(path)
            removed.append(path)
        except OSError:
            pass
    return removed


def print_resume_banner(lines: str | list[str]) -> None:
    """Print the block a user needs to restart this run in the next Kaggle session."""
    if isinstance(lines, str):
        lines = lines.splitlines()
    width = 74
    print("\n" + "=" * width)
    print("RESUME".center(width))
    print("=" * width)
    for line in lines:
        print(line)
    print("=" * width + "\n", flush=True)


def kaggle_resume_instructions(ckpt_dir: str, notebook_hint: str = "<this-notebook>") -> list[str]:
    """The exact steps to continue this run in a fresh Kaggle session."""
    return [
        "This session stopped on its time budget, not on an error. Progress is saved.",
        "",
        f"  checkpoint : {os.path.join(ckpt_dir, 'last.pt')}",
        f"  best so far: {os.path.join(ckpt_dir, 'best.pt')}",
        "",
        "To continue:",
        "  1. Click 'Save Version' -> 'Save & Run All (Commit)'. Wait for it to finish;",
        f"     the contents of /kaggle/working become the dataset output of {notebook_hint}.",
        "  2. Open the notebook again and, in the right-hand panel, click",
        "     '+ Add Input' -> 'Notebook Output' -> pick this notebook's latest version.",
        "  3. Re-run. find_resume() picks up /kaggle/input/**/last.pt automatically;",
        "     you do not have to change any path by hand.",
        "",
        "Nothing else needs editing. The epoch, optimizer state, scheduler, AMP scaler",
        "and RNG streams all resume exactly where they stopped.",
    ]
