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

from csnet.memory import (available_gb, cgroup_usage_gb, format_snapshot, log_memory,
                          release, rss_gb, snapshot)

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
