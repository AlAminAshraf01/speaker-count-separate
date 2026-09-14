"""Host-RAM telemetry.

A Kaggle session that exceeds its memory limit is killed with SIGKILL. There is no
traceback, no Python exception, and no last words -- the log simply stops and the run
reports ``exit -9``. Guessing afterwards is expensive, so the training loop prints where
memory stands at every log line and at every epoch and validation boundary. When a run
does die, the last printed line says how close it was and which phase it was in.

Everything here degrades to ``nan`` off Linux and never raises: telemetry must not be able
to break a training run.
"""

from __future__ import annotations

import os

_PAGE = 4096.0
_GB = 1024.0 ** 3


def _read_first_int(path: str) -> float:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return float(fh.read().split()[0])
    except Exception:
        return float("nan")


def rss_gb(statm_path: str = "/proc/self/statm") -> float:
    """Resident set size of this process, in GiB.

    ``statm`` field 1 is resident pages (field 0 is total program size, which counts
    address space this process will never touch and so overstates memory badly).
    """
    try:
        with open(statm_path, "r", encoding="utf-8") as fh:
            return float(fh.read().split()[1]) * _PAGE / _GB
    except Exception:
        pass
    try:
        import resource

        # ru_maxrss is KiB on Linux, bytes on macOS; a peak, not a current value, so it is
        # only the fallback for platforms without /proc.
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024.0 / _GB
    except Exception:
        return float("nan")


def cgroup_usage_gb(pairs: tuple[tuple[str, str], ...] | None = None) -> tuple[float, float]:
    """``(used, limit)`` for this container's memory cgroup, in GiB.

    This is the number that actually gets a Kaggle session killed: the cgroup charges page
    cache as well as anonymous memory, so a process whose own RSS looks modest can still
    push the container over. Tries cgroup v2 then v1; returns nan where unavailable.
    """
    pairs = pairs or (("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
                      ("/sys/fs/cgroup/memory/memory.usage_in_bytes",
                       "/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    for used_path, limit_path in pairs:
        used = _read_first_int(used_path)
        if used != used:
            continue
        limit = _read_first_int(limit_path)
        # cgroup v1 reports "no limit" as a huge sentinel; v2 writes the string "max",
        # which _read_first_int turns into nan.
        if limit == limit and limit > 0 and limit < 2 ** 53:
            return used / _GB, limit / _GB
        return used / _GB, float("nan")
    return float("nan"), float("nan")


def available_gb(meminfo_path: str = "/proc/meminfo") -> float:
    """MemAvailable from /proc/meminfo, in GiB -- what the host thinks is spare."""
    try:
        with open(meminfo_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024.0 / _GB
    except Exception:
        pass
    return float("nan")


def snapshot() -> dict:
    """All four numbers at once."""
    used, limit = cgroup_usage_gb()
    return {"rss_gb": rss_gb(), "cgroup_gb": used, "cgroup_limit_gb": limit,
            "available_gb": available_gb()}


def format_snapshot(tag: str = "") -> str:
    """One compact line, e.g. ``ram rss 4.1G | cgroup 9.8/13.0G (75%) | avail 3.1G``."""
    snap = snapshot()
    parts = []
    if snap["rss_gb"] == snap["rss_gb"]:
        parts.append(f"rss {snap['rss_gb']:.1f}G")
    if snap["cgroup_gb"] == snap["cgroup_gb"]:
        if snap["cgroup_limit_gb"] == snap["cgroup_limit_gb"]:
            pct = 100.0 * snap["cgroup_gb"] / max(1e-9, snap["cgroup_limit_gb"])
            parts.append(f"cgroup {snap['cgroup_gb']:.1f}/{snap['cgroup_limit_gb']:.1f}G "
                         f"({pct:.0f}%)")
        else:
            parts.append(f"cgroup {snap['cgroup_gb']:.1f}G")
    if snap["available_gb"] == snap["available_gb"]:
        parts.append(f"avail {snap['available_gb']:.1f}G")
    if not parts:
        return f"ram {tag} unavailable".strip()
    return f"ram {tag + ' ' if tag else ''}" + " | ".join(parts)


def log_memory(tag: str = "", printer=print) -> None:
    """Print one telemetry line. Never raises."""
    try:
        printer(format_snapshot(tag))
    except Exception:
        pass


def release() -> None:
    """Drop what can be dropped between phases: Python garbage and the CUDA cache."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class MemoryBudgetExceeded(RuntimeError):
    """Raised when container memory crosses the guard threshold.

    The point is to lose the session on our terms instead of the kernel's. SIGKILL takes
    the process with no traceback, no final checkpoint and no clue; this leaves all three.
    """


class MemoryGuard:
    """Stop before the OOM killer does.

    ``frac`` is the share of the cgroup limit at which to give up. The default leaves
    enough headroom for one more batch, one more checkpoint write and the interpreter's
    own churn -- stopping at 99 % would just lose the race.

    Where no cgroup limit is readable the guard is inert rather than guessing at one; a
    guard that fires on a bad reading would be worse than no guard.
    """

    def __init__(self, frac: float = 0.85, check_every: int = 50) -> None:
        self.frac = float(frac)
        self.check_every = max(1, int(check_every))
        _, limit = cgroup_usage_gb()
        self.limit_gb = limit
        self.enabled = limit == limit and limit > 0
        self.peak_gb = 0.0

    def usage(self) -> float:
        used, _ = cgroup_usage_gb()
        if used == used:
            self.peak_gb = max(self.peak_gb, used)
        return used

    def exceeded(self) -> bool:
        if not self.enabled:
            return False
        used = self.usage()
        return used == used and used >= self.frac * self.limit_gb

    def check(self, where: str = "") -> None:
        """Raise :class:`MemoryBudgetExceeded` if over threshold."""
        if self.exceeded():
            raise MemoryBudgetExceeded(
                f"container memory {self.usage():.1f} GiB of {self.limit_gb:.1f} GiB "
                f"({self.frac * 100:.0f}% threshold) {('at ' + where) if where else ''}")

    def describe(self) -> str:
        if not self.enabled:
            return "MemoryGuard(inert -- no cgroup limit readable)"
        return (f"MemoryGuard(stop at {self.frac * 100:.0f}% of "
                f"{self.limit_gb:.1f} GiB, peak seen {self.peak_gb:.1f} GiB)")


if __name__ == "__main__":  # pragma: no cover - manual check
    print(format_snapshot("self-test"))
    print(snapshot())
    print(f"PID {os.getpid()}")
