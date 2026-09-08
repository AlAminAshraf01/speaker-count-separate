#!/usr/bin/env python3
"""Convert the percent-format sources in ``notebooks/src/`` into real ``.ipynb`` files.

Kaggle imports ``.ipynb`` directly (New Notebook -> File -> Import Notebook), but a
notebook is unreadable JSON in a git diff.  So the *source of truth* for every notebook in
this repo is a plain ``.py`` file using the Jupyter "percent" cell format, and this script
renders them.  No third-party dependency (no jupytext, no nbformat) -- stdlib only.

Cell markers understood::

    # %%                      -> code cell
    # %% [markdown]           -> markdown cell (leading "# " is stripped from each line)
    # %% [raw]                -> raw cell

Usage::

    python tools/build_notebooks.py                 # notebooks/src/*.py -> notebooks/*.ipynb
    python tools/build_notebooks.py --check         # fail if any .ipynb is stale
    python tools/build_notebooks.py --src A --out B
"""

from __future__ import annotations

import argparse
import json
import os
import sys

CELL_MARKER = "# %%"
INCLUDE_MARKER = "# %include "


def expand_includes(text: str, base_dir: str, depth: int = 0) -> str:
    """Replace ``# %include other.py`` lines with that file's contents.

    Included files carry their own cell markers, so the bootstrap block lives in one
    place and every notebook still ships self-contained -- which matters on Kaggle,
    where a notebook is uploaded alone.
    """
    if depth > 4:
        raise RecursionError("include nesting is too deep")
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(INCLUDE_MARKER):
            name = stripped[len(INCLUDE_MARKER):].strip()
            path = os.path.join(base_dir, name)
            with open(path, "r", encoding="utf-8") as fh:
                out.append(expand_includes(fh.read(), base_dir, depth + 1))
        else:
            out.append(line)
    return "\n".join(out)


def split_cells(text: str) -> list[tuple[str, list[str]]]:
    """Split percent-format source into ``(cell_type, lines)`` pairs."""
    lines = text.splitlines()
    cells: list[tuple[str, list[str]]] = []
    cur_type = "code"
    cur: list[str] = []

    def flush() -> None:
        if cur and any(ln.strip() for ln in cur):
            cells.append((cur_type, list(cur)))
        cur.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(CELL_MARKER):
            flush()
            tag = stripped[len(CELL_MARKER):].strip()
            if tag.startswith("[markdown]"):
                cur_type = "markdown"
            elif tag.startswith("[raw]"):
                cur_type = "raw"
            else:
                cur_type = "code"
            continue
        cur.append(line)
    flush()
    return cells


def _strip_comment(lines: list[str]) -> list[str]:
    """Turn ``# text`` comment lines back into plain markdown text."""
    out = []
    for ln in lines:
        if ln.startswith("# "):
            out.append(ln[2:])
        elif ln.rstrip() == "#":
            out.append("")
        else:
            out.append(ln)
    return out


def _trim(lines: list[str]) -> list[str]:
    """Drop leading/trailing blank lines."""
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _as_source(lines: list[str]) -> list[str]:
    """nbformat stores source as a list of lines, each keeping its trailing newline."""
    if not lines:
        return []
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def to_notebook(text: str, kernel: str = "python3") -> dict:
    """Render percent-format source text as an nbformat-4 notebook dict."""
    cells = []
    for cell_type, raw in split_cells(text):
        body = _trim(_strip_comment(raw) if cell_type != "code" else raw)
        if not body:
            continue
        cell: dict = {
            "cell_type": cell_type,
            "metadata": {},
            "source": _as_source(body),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": kernel,
            },
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--src", default=os.path.join(here, "notebooks", "src"))
    ap.add_argument("--out", default=os.path.join(here, "notebooks"))
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if any notebook is missing or stale")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"no such source dir: {args.src}", file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)

    stale, built = [], []
    for name in sorted(os.listdir(args.src)):
        # a leading underscore marks an include-only fragment, not a notebook
        if not name.endswith(".py") or name.startswith("_"):
            continue
        src_path = os.path.join(args.src, name)
        out_path = os.path.join(args.out, name[:-3] + ".ipynb")
        with open(src_path, "r", encoding="utf-8") as fh:
            nb = to_notebook(expand_includes(fh.read(), args.src))
        payload = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"

        if args.check:
            existing = None
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as fh:
                    existing = fh.read()
            if existing != payload:
                stale.append(out_path)
            continue

        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(payload)
        built.append((out_path, len(nb["cells"])))

    if args.check:
        for path in stale:
            print(f"STALE {path}")
        print(f"{len(stale)} stale notebook(s)")
        return 1 if stale else 0

    for path, n_cells in built:
        print(f"wrote {os.path.relpath(path, here)}  ({n_cells} cells)")
    print(f"{len(built)} notebook(s) built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
