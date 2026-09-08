#!/usr/bin/env python3
"""Run every test file, with or without pytest installed.

    python tools/run_all_tests.py
    python tools/run_all_tests.py --file test_losses.py

Exit code 0 means everything passed. Kaggle has pytest, but this works without it too, so
the suite is runnable from a notebook cell with a plain ``!python``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_ROOT, "tests")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=None, help="run only this test file")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    names = ([args.file] if args.file else
             sorted(f for f in os.listdir(TESTS_DIR)
                    if f.startswith("test_") and f.endswith(".py")))
    if not names:
        print("no test files found", file=sys.stderr)
        return 1

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(REPO_ROOT, "src"), os.path.join(REPO_ROOT, "tools"),
         env.get("PYTHONPATH", "")])

    results, failed = [], 0
    total_start = time.time()
    for name in names:
        path = os.path.join(TESTS_DIR, name)
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        start = time.time()
        proc = subprocess.run([sys.executable, path], env=env, cwd=REPO_ROOT,
                              capture_output=args.quiet, text=True)
        elapsed = time.time() - start
        if args.quiet and proc.returncode != 0:
            print(proc.stdout or "")
            print(proc.stderr or "", file=sys.stderr)
        results.append((name, proc.returncode, elapsed))
        failed += int(proc.returncode != 0)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    width = max(len(n) for n, _, _ in results)
    for name, code, elapsed in results:
        print(f"  {'PASS' if code == 0 else 'FAIL'}  {name.ljust(width)}  {elapsed:6.1f}s")
    print(f"\n{len(results) - failed}/{len(results)} files passed "
          f"in {time.time() - total_start:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
