"""Memory telemetry must parse real Linux formats and never raise anywhere else.

These run on Windows, where /proc does not exist, so every reader takes its path as an
argument and the tests feed it synthetic files in the exact kernel formats. Telemetry that
silently returns nan on Kaggle would be worse than none -- it would look like it worked.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

import csnet.memory as M
from csnet.memory import (LeakWatch, MemoryBudgetExceeded, MemoryGuard, available_gb,
                          cgroup_usage_gb, format_snapshot, log_memory, release,
                          rss_gb, snapshot)

_GB = 1024.0 ** 3
TMP = tempfile.mkdtemp()


def _write(name: str, text: str) -> str:
    path = os.path.join(TMP, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def test_rss_reads_statm_field_one() -> None:
    """statm is 'size resident shared text lib data dt' in pages; field 1 is resident."""
    # 2,000,000 resident pages x 4 KiB = 8000 MiB = 7.8125 GiB
    path = _write("statm", "9999999 2000000 12345 1 0 500000 0\n")
    got = rss_gb(path)
    expected = 2_000_000 * 4096 / _GB
    assert abs(got - expected) < 1e-9, f"{got} != {expected}"
    assert abs(got - 7.62939453125) < 1e-6, got


def test_rss_missing_file_does_not_raise() -> None:
    value = rss_gb(os.path.join(TMP, "definitely-not-here"))
    assert isinstance(value, float)          # nan or the resource fallback, never an error


def test_cgroup_v2_used_and_limit() -> None:
    used = _write("memory.current", "9663676416\n")       # 9 GiB
    limit = _write("memory.max", "13958643712\n")         # 13 GiB
    got_used, got_limit = cgroup_usage_gb(((used, limit),))
    assert abs(got_used - 9.0) < 1e-6, got_used
    assert abs(got_limit - 13.0) < 1e-6, got_limit


def test_cgroup_v2_unlimited_reads_max_as_nan() -> None:
    """cgroup v2 writes the literal string 'max' when there is no limit."""
    used = _write("c2_used", "1073741824\n")
    limit = _write("c2_max", "max\n")
    got_used, got_limit = cgroup_usage_gb(((used, limit),))
    assert abs(got_used - 1.0) < 1e-6
    assert got_limit != got_limit, "unlimited must be nan, not a number"


def test_cgroup_v1_sentinel_limit_is_rejected() -> None:
    """v1 reports 'no limit' as a huge sentinel; reporting it as a real limit is wrong."""
    used = _write("v1_used", "2147483648\n")
    limit = _write("v1_limit", "9223372036854771712\n")
    got_used, got_limit = cgroup_usage_gb(((used, limit),))
    assert abs(got_used - 2.0) < 1e-6
    assert got_limit != got_limit, "sentinel must not be reported as a limit"


def test_cgroup_falls_through_to_second_pair() -> None:
    """v2 paths absent -> try v1. That is the whole point of the pair list."""
    missing = os.path.join(TMP, "nope")
    used = _write("fall_used", "3221225472\n")
    limit = _write("fall_limit", "4294967296\n")
    got_used, got_limit = cgroup_usage_gb(((missing, missing), (used, limit)))
    assert abs(got_used - 3.0) < 1e-6
    assert abs(got_limit - 4.0) < 1e-6


def test_available_parses_meminfo() -> None:
    path = _write("meminfo", "MemTotal:       32659948 kB\n"
                             "MemFree:         1234567 kB\n"
                             "MemAvailable:    8388608 kB\n"
                             "Buffers:          123456 kB\n")
    assert abs(available_gb(path) - 8.0) < 1e-6, available_gb(path)


def test_available_missing_key_is_nan() -> None:
    path = _write("meminfo_nokey", "MemTotal: 123 kB\nMemFree: 12 kB\n")
    value = available_gb(path)
    assert value != value


def test_snapshot_and_format_never_raise() -> None:
    snap = snapshot()
    assert set(snap) == {"rss_gb", "cgroup_gb", "cgroup_limit_gb", "available_gb"}
    text = format_snapshot("phase")
    assert isinstance(text, str) and text.startswith("ram")
    lines: list[str] = []
    log_memory("tagged", printer=lines.append)
    assert len(lines) == 1


def test_log_memory_survives_a_broken_printer() -> None:
    """Telemetry must never be able to kill a training run."""
    def explode(_text: str) -> None:
        raise RuntimeError("printer is broken")

    log_memory("boom", printer=explode)      # must not propagate


def test_release_is_safe_to_call() -> None:
    release()
    release()


def _guard_with(used_bytes: int, limit_bytes: int, frac: float = 0.85):
    """A guard reading a synthetic cgroup, so the thresholds can be tested off Linux."""
    used = _write("g_used", str(used_bytes))
    limit = _write("g_limit", str(limit_bytes))
    real = M.cgroup_usage_gb
    M.cgroup_usage_gb = lambda pairs=None: real(((used, limit),))
    return MemoryGuard(frac=frac), real


def test_guard_fires_above_the_threshold_and_not_below() -> None:
    real = M.cgroup_usage_gb
    try:
        guard, _ = _guard_with(27_917_287_424, 32_212_254_720)     # 26 of 30 GiB = 87%
        assert guard.enabled and abs(guard.limit_gb - 30.0) < 1e-6
        assert guard.exceeded(), "87% must trip an 85% guard"
        raised = False
        try:
            guard.check("validation batch 40")
        except MemoryBudgetExceeded as exc:
            raised = True
            assert "26.0" in str(exc) and "30.0" in str(exc), str(exc)
            assert "validation batch 40" in str(exc), "the message must name the phase"
        assert raised, "check() must raise once over the threshold"
    finally:
        M.cgroup_usage_gb = real

    try:
        guard, _ = _guard_with(10_737_418_240, 32_212_254_720)     # 10 of 30 GiB = 33%
        assert not guard.exceeded(), "33% must not trip an 85% guard"
        guard.check("training step 100")                            # must not raise
    finally:
        M.cgroup_usage_gb = real


def test_guard_is_inert_without_a_readable_limit() -> None:
    """A guard that fires on an unreadable cgroup would be worse than no guard at all."""
    real = M.cgroup_usage_gb
    M.cgroup_usage_gb = lambda pairs=None: (float("nan"), float("nan"))
    try:
        guard = MemoryGuard(frac=0.01)       # absurdly low: still must not fire
        assert not guard.enabled
        assert not guard.exceeded()
        guard.check("anywhere")
        assert "inert" in guard.describe()
    finally:
        M.cgroup_usage_gb = real


def test_guard_tracks_its_peak() -> None:
    real = M.cgroup_usage_gb
    try:
        guard, _ = _guard_with(5_368_709_120, 32_212_254_720)       # 5 of 30 GiB
        guard.usage()
        assert abs(guard.peak_gb - 5.0) < 1e-6, guard.peak_gb
        assert "peak seen 5.0" in guard.describe(), guard.describe()
    finally:
        M.cgroup_usage_gb = real


class _FakeGuard:
    """A guard whose usage follows a chosen growth rate, so slopes can be tested exactly."""

    def __init__(self, start_gb: float, gb_per_step: float, limit_gb: float = 30.0,
                 frac: float = 0.85) -> None:
        self.start, self.rate = float(start_gb), float(gb_per_step)
        self.limit_gb, self.frac, self.enabled = float(limit_gb), float(frac), True
        self.step = 0

    def usage(self) -> float:
        return self.start + self.rate * self.step


def test_leakwatch_stops_a_run_that_cannot_finish_an_epoch() -> None:
    """The measured failure: 8 MiB/step from 2.1 GiB against a 30 GiB limit.

    The session planned 21 epochs of 1000 steps. 8 MiB/step eats the headroom in about
    2700 steps, so it would spend eleven hours of quota to bank under three epochs. That
    has to be caught in the first few minutes, not the fourth hour.
    """
    guard = _FakeGuard(start_gb=2.1, gb_per_step=8.0 / 1024.0)
    watch = LeakWatch(guard, horizon=21000, warmup=100, window=200)
    raised = None
    for step in range(0, 1000, 50):
        guard.step = step
        try:
            watch.observe(step)
        except MemoryBudgetExceeded as exc:
            raised = (step, str(exc))
            break
    assert raised, "8 MiB/step must be caught"
    step, message = raised
    assert step <= 350, f"caught at step {step}: too late to be useful"
    assert "8.0 MiB per step" in message, message
    assert "stopping now" in message, message
    assert abs(watch.slope_gb_per_step * 1024 - 8.0) < 0.01, watch.slope_gb_per_step


def test_leakwatch_tolerates_growth_the_session_can_absorb() -> None:
    """Slow growth is not a reason to throw away a session that can still do the work."""
    guard = _FakeGuard(start_gb=2.1, gb_per_step=0.05 / 1024.0)     # 0.05 MiB/step
    watch = LeakWatch(guard, horizon=21000, warmup=100, window=200)
    for step in range(0, 2000, 50):
        guard.step = step
        watch.observe(step)                                          # must not raise
    assert "survivable" in watch.describe(), watch.describe()


def test_leakwatch_reports_a_flat_run_as_healthy() -> None:
    guard = _FakeGuard(start_gb=4.0, gb_per_step=0.0)
    watch = LeakWatch(guard, horizon=21000, warmup=100, window=200)
    for step in range(0, 1000, 50):
        guard.step = step
        watch.observe(step)
    assert watch.describe() == "LeakWatch(no growth measured -- healthy)", watch.describe()


def test_leakwatch_ignores_startup_allocations() -> None:
    """A big one-off jump before warmup must not be read as a trend."""
    guard = _FakeGuard(start_gb=2.0, gb_per_step=0.0)
    watch = LeakWatch(guard, horizon=21000, warmup=100, window=200)
    for step in range(0, 100, 25):        # startup: 2 -> 6 GiB, all before warmup
        guard.start = 2.0 + step * 0.04
        guard.step = step
        watch.observe(step)
    guard.start = 6.0                      # then completely flat
    for step in range(100, 800, 50):
        guard.step = step
        watch.observe(step)                # must not raise
    assert "healthy" in watch.describe(), watch.describe()


def test_leakwatch_is_inert_without_a_guard() -> None:
    watch = LeakWatch(None, horizon=21000)
    for step in range(0, 1000, 50):
        watch.observe(step)
    assert not watch.enabled and "inert" in watch.describe()


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
                print(f"  PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  FAIL  {name}: {exc}")
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
