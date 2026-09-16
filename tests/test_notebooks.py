"""Notebook cells are the one thing a git pull cannot fix on Kaggle.

``scripts/`` and ``src/`` update themselves: the bootstrap fast-forwards the clone every
session. Cells do not -- Kaggle owns them, and the only way to refresh them is to import
the ``.ipynb`` again. Two sessions have already been spent running a fix that was in the
repo and not in the cells.

The stamp closes that gap: the builder writes a fingerprint of the source into the
notebook, and the bootstrap recomputes it from the clone. These tests make sure the two
halves of that comparison cannot drift apart, because a staleness check that quietly
always passes is worse than none at all.
"""

from __future__ import annotations

import ast
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "notebooks", "src")
OUT_DIR = os.path.join(REPO_ROOT, "notebooks")


def _bootstrap_fingerprint():
    """The ``cells_fingerprint`` defined inside ``notebooks/src/_bootstrap.py``.

    That module cannot simply be imported: at import time it clones a git repository and
    chdirs into it. So pull the one function out of the syntax tree and compile that.
    """
    with io.open(os.path.join(SRC_DIR, "_bootstrap.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "cells_fingerprint":
            namespace: dict = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "_bootstrap.py", "exec"),
                 {"hashlib": __import__("hashlib"), "os": os}, namespace)
            return namespace["cells_fingerprint"]
    raise AssertionError("cells_fingerprint is missing from notebooks/src/_bootstrap.py")


def test_the_two_fingerprint_copies_agree() -> None:
    """The builder's copy and the bootstrap's copy must compute the same value.

    They are deliberately duplicated -- one runs on Kaggle inside a notebook cell, the
    other runs here -- so this is the only thing stopping them from drifting.
    """
    from build_notebooks import cells_fingerprint as builders

    theirs = _bootstrap_fingerprint()
    for name in sorted(os.listdir(SRC_DIR)):
        if name.endswith(".py") and not name.startswith("_"):
            assert builders(SRC_DIR, name) == theirs(SRC_DIR, name), name


def test_fingerprint_ignores_line_endings() -> None:
    """CRLF in a Windows checkout, LF in a Linux clone: the same source, the same hash."""
    from build_notebooks import cells_fingerprint

    directory = tempfile.mkdtemp()
    body = b"# %%\nprint('hello')\n"
    for name in ("nb.py", "_bootstrap.py"):
        with open(os.path.join(directory, name), "wb") as fh:
            fh.write(body)
    lf = cells_fingerprint(directory, "nb.py")
    for name in ("nb.py", "_bootstrap.py"):
        with open(os.path.join(directory, name), "wb") as fh:
            fh.write(body.replace(b"\n", b"\r\n"))
    assert cells_fingerprint(directory, "nb.py") == lf


def test_fingerprint_moves_when_either_file_does() -> None:
    """Editing the notebook source *or* the shared bootstrap must invalidate the stamp."""
    from build_notebooks import cells_fingerprint

    directory = tempfile.mkdtemp()
    for name in ("nb.py", "_bootstrap.py"):
        with open(os.path.join(directory, name), "wb") as fh:
            fh.write(b"original\n")
    before = cells_fingerprint(directory, "nb.py")
    for changed in ("nb.py", "_bootstrap.py"):
        with open(os.path.join(directory, changed), "wb") as fh:
            fh.write(b"edited\n")
        assert cells_fingerprint(directory, "nb.py") != before, changed
        with open(os.path.join(directory, changed), "wb") as fh:
            fh.write(b"original\n")


def test_every_notebook_carries_a_current_stamp() -> None:
    """The first cell of each built notebook must match the source it came from.

    A failure here means ``python tools/build_notebooks.py`` has not been run since the
    last edit -- which is exactly the situation the stamp exists to catch, so it cannot
    be allowed to ship.
    """
    from build_notebooks import cells_fingerprint

    for name in sorted(os.listdir(SRC_DIR)):
        if not name.endswith(".py") or name.startswith("_"):
            continue
        path = os.path.join(OUT_DIR, name[:-3] + ".ipynb")
        assert os.path.exists(path), f"{path} has never been built"
        with io.open(path, encoding="utf-8") as fh:
            first = "".join(json.load(fh)["cells"][0]["source"])
        assert f'CELLS_SRC = "{name}"' in first, f"{path} has no stamp cell first"
        assert f'CELLS_SHA = "{cells_fingerprint(SRC_DIR, name)}"' in first, (
            f"{path} is stale: run python tools/build_notebooks.py")


def test_notebooks_are_not_stale() -> None:
    """``build_notebooks.py --check`` is the same question, asked the builder's way."""
    from build_notebooks import expand_includes, stamp_cell, to_notebook
    from build_notebooks import cells_fingerprint

    for name in sorted(os.listdir(SRC_DIR)):
        if not name.endswith(".py") or name.startswith("_"):
            continue
        with io.open(os.path.join(SRC_DIR, name), encoding="utf-8") as fh:
            nb = to_notebook(expand_includes(fh.read(), SRC_DIR),
                             stamp=stamp_cell(name, cells_fingerprint(SRC_DIR, name)))
        expected = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
        with io.open(os.path.join(OUT_DIR, name[:-3] + ".ipynb"), encoding="utf-8") as fh:
            assert fh.read() == expected, f"{name}: run python tools/build_notebooks.py"


def test_every_notebook_source_compiles() -> None:
    """A syntax error in a cell should fail here, not forty minutes into a session.

    The percent format is valid Python by construction -- cell markers and markdown are
    comments -- so the whole expanded file can simply be compiled. This catches the
    unterminated f-strings and stray brackets that an edit to a `run(...)` command
    produces, which Kaggle would otherwise report only when it reached that cell.
    """
    from build_notebooks import expand_includes

    for name in sorted(os.listdir(SRC_DIR)):
        if not name.endswith(".py"):
            continue
        with io.open(os.path.join(SRC_DIR, name), encoding="utf-8") as fh:
            source = expand_includes(fh.read(), SRC_DIR)
        compile(source, name, "exec")  # raises SyntaxError with the line number


def test_no_notebook_picks_a_checkpoint_by_name() -> None:
    """``sorted(glob(...))[0]`` returns ``_dryrun/best.pt``. It must not come back.

    ``_`` sorts before every letter, so the alphabetically first checkpoint in a training
    notebook's output is the five-step plumbing check. Evaluating that produces a full
    report with plausible numbers and nothing anywhere saying it used the wrong model.
    """
    for name in sorted(os.listdir(SRC_DIR)):
        if not name.endswith(".py"):
            continue
        with io.open(os.path.join(SRC_DIR, name), encoding="utf-8") as fh:
            text = fh.read()
        assert "ckpts[0]" not in text, f"{name} still picks a checkpoint by path order"


CHECKS = {name: fn for name, fn in sorted(globals().items()) if name.startswith("test_")}

if __name__ == "__main__":
    print(__doc__)
    sys.exit(run_checks(CHECKS))
