"""The thirty-second check that stands between a typo and eleven hours of quota.

Each test here corresponds to a failure that has already happened once:

* a run resumed from ``_dryrun/last.pt`` because ``_`` sorts before ``c``;
* a recovery session handed ``range(38, 12)``, which trained nothing and exited 0;
* four hours of training killed by an OOM that DataParallel had been causing all along.

The last test is the one that matters most: a preflight that crashes is worse than no
preflight, because it takes down notebooks that were otherwise fine.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def _preflight():
    """Import ``scripts/12_preflight.py``, whose name is not a legal module name."""
    spec = importlib.util.spec_from_file_location(
        "preflight", os.path.join(SCRIPTS, "12_preflight.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_run(root: str, name: str, step: int, epoch: int, seconds: float = 400.0) -> str:
    """A checkpoint directory as the trainer leaves one: ``last.pt`` beside a history."""
    directory = os.path.join(root, name)
    os.makedirs(directory, exist_ok=True)
    for leaf in ("last.pt", "best.pt"):
        with open(os.path.join(directory, leaf), "wb") as fh:
            fh.write(b"not a real archive")
    with io.open(os.path.join(directory, "history.json"), "w", encoding="utf-8") as fh:
        json.dump([{"epoch": epoch, "global_step": step, "seconds": seconds}], fh)
    return directory


# ------------------------------------------------------------------ checkpoint ranking

def test_the_dry_run_does_not_win_on_alphabetical_order() -> None:
    """``_dryrun`` sorts first and must still lose: it is five steps, not fifty thousand."""
    from _common import rank_checkpoints

    root = tempfile.mkdtemp(prefix="csnet-rank-")
    _fake_run(root, "_dryrun", step=5, epoch=1)
    _fake_run(root, "ckpt_count", step=50800, epoch=50)
    ranked = rank_checkpoints("best.pt", roots=[root])
    assert len(ranked) == 2, ranked
    assert "ckpt_count" in ranked[0][0], ranked
    assert ranked[0][1] == 50800, ranked
    # And the thing the old code did, to show the two really do disagree:
    import glob

    assert "_dryrun" in sorted(glob.glob(os.path.join(root, "**", "best.pt"),
                                         recursive=True))[0]


def test_autodetect_prefers_the_furthest_along() -> None:
    from _common import autodetect_ckpt

    root = tempfile.mkdtemp(prefix="csnet-auto-")
    _fake_run(root, "_dryrun", step=5, epoch=1)
    wanted = _fake_run(root, "ckpt", step=38800, epoch=38)
    picked = autodetect_ckpt(roots=[root], verbose=False)
    assert os.path.dirname(picked) == wanted, picked


def test_an_explicit_path_is_honoured() -> None:
    from _common import autodetect_ckpt

    root = tempfile.mkdtemp(prefix="csnet-hint-")
    _fake_run(root, "_dryrun", step=5, epoch=1)
    _fake_run(root, "ckpt", step=38800, epoch=38)
    hint = os.path.join(root, "_dryrun", "best.pt")
    assert autodetect_ckpt(hint, roots=[root], verbose=False) == os.path.normpath(hint)


def test_a_named_run_beats_the_furthest_along_one() -> None:
    """A short control run must be selectable, or it can never be evaluated.

    `ckpt_gate2` stops at step 8000 by design and will always lose on step count to the
    50,800-step pooled model it exists to be compared against.
    """
    from _common import autodetect_ckpt

    root = tempfile.mkdtemp(prefix="csnet-only-")
    _fake_run(root, "ckpt_count", step=50800, epoch=50)
    wanted = _fake_run(root, "ckpt_gate2", step=8000, epoch=8)

    assert "ckpt_count" in autodetect_ckpt(roots=[root], verbose=False)
    picked = autodetect_ckpt(roots=[root], verbose=False, contains="ckpt_gate2")
    assert os.path.dirname(picked) == wanted, picked


def test_an_unmatched_filter_says_what_is_there() -> None:
    """"no checkpoint found" is useless; naming the directories that exist is not."""
    from _common import autodetect_ckpt

    root = tempfile.mkdtemp(prefix="csnet-miss-")
    _fake_run(root, "ckpt_count", step=50800, epoch=50)
    try:
        autodetect_ckpt(roots=[root], verbose=False, contains="ckpt_nope")
    except SystemExit as exc:
        assert "ckpt_count" in str(exc), exc
    else:
        raise AssertionError("an unmatched filter must not silently fall back")


def test_search_does_not_descend_into_a_corpus() -> None:
    """The store search prunes ``s1``/``mix_both``; so must this one, or it walks 14k files."""
    from _common import _find_files

    root = tempfile.mkdtemp(prefix="csnet-prune-")
    for leaf in ("s1", ".git", "ckpt"):
        os.makedirs(os.path.join(root, "test", leaf), exist_ok=True)
        with open(os.path.join(root, "test", leaf, "best.pt"), "wb") as fh:
            fh.write(b"x")
    found = _find_files([root], "best.pt")
    assert len(found) == 1, found
    assert os.path.basename(os.path.dirname(found[0])) == "ckpt", found


def test_recipes_are_not_taken_from_a_nested_clone() -> None:
    """A training notebook's output carries a whole git clone of this repo.

    ``glob("/kaggle/input/**/recipes_test.csv")[0]`` can therefore return the copy
    committed to the repo rather than the one the data notebook built, decided by
    filesystem walk order. Two notebooks then score one checkpoint against two different
    frozen sets and neither log says so.
    """
    from _common import find_recipes

    root = tempfile.mkdtemp(prefix="csnet-recipes-")
    nested = os.path.join(root, "kaggle-02-train", "speaker-count-separate", "data")
    beside = os.path.join(root, "kaggle-00-build-dataset")
    body = "\n".join(["mix_id,n_src", "x,1", ""])
    for directory in (nested, beside):
        os.makedirs(directory, exist_ok=True)
        with io.open(os.path.join(directory, "recipes_test.csv"), "w", encoding="utf-8") as fh:
            fh.write(body)

    picked = find_recipes("recipes_test.csv", beside)
    assert os.path.dirname(picked) == beside, picked


def test_the_recipe_fingerprint_notices_a_different_file() -> None:
    """Same row count, different content: only the hash tells them apart."""
    from _common import describe_recipes

    root = tempfile.mkdtemp(prefix="csnet-sha-")
    a = os.path.join(root, "a.csv")
    b = os.path.join(root, "b.csv")
    lines = ["mix_id,n_src", "x,1", "y,2", ""]
    with io.open(a, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    with io.open(b, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(["mix_id,n_src", "x,1", "y,3", ""]))

    fa, fb = describe_recipes(a), describe_recipes(b)
    assert fa["rows"] == fb["rows"] == 2
    assert fa["sha"] != fb["sha"], (fa, fb)
    assert fa["per_n"] == {1: 1, 2: 1}, fa
    # CRLF is not a different frozen set -- this repo has been bitten by that before.
    crlf = os.path.join(root, "c.csv")
    with open(crlf, "wb") as fh:
        fh.write("\r\n".join(lines).encode("utf-8"))
    assert describe_recipes(crlf)["sha"] == fa["sha"]


# ------------------------------------------------------------------ individual checks

def _args(**kwargs) -> argparse.Namespace:
    base = dict(profile="train", config=None, overrides=None, store=None,
                recipes_dev=None, recipes_test=None, ckpt_dir=None, epochs=None,
                extra_epochs=None, time_budget_h=None, ms_per_step=None,
                cells_src=None, cells_sha=None, strict=False, resume_contains=None)
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_an_empty_epoch_range_is_a_stopper() -> None:
    """``train.epochs`` is a target. Resuming past it means training nothing, silently."""
    pf = _preflight()
    from csnet.config import load_cfg

    root = tempfile.mkdtemp(prefix="csnet-epochs-")
    work = _fake_run(root, "ckpt", step=38800, epoch=38)
    cfg = load_cfg(os.path.join(REPO_ROOT, "configs", "tiny.yaml"), ["train.epochs=12"])
    name, level, detail, fix = pf.check_epochs(
        _args(profile="train", ckpt_dir=work), {"cfg": cfg})
    assert level == pf.STOP, (level, detail)
    assert "range(38, 12)" in detail, detail
    assert "--extra_epochs" in fix, fix


def test_extra_epochs_counts_from_the_checkpoint() -> None:
    pf = _preflight()
    from csnet.config import load_cfg

    root = tempfile.mkdtemp(prefix="csnet-extra-")
    work = _fake_run(root, "ckpt", step=38800, epoch=38)
    cfg = load_cfg(os.path.join(REPO_ROOT, "configs", "tiny.yaml"), ["train.epochs=12"])
    state: dict = {"cfg": cfg}
    _, level, detail, _ = pf.check_epochs(
        _args(profile="recover", ckpt_dir=work, extra_epochs=12), state)
    assert level == pf.OK, (level, detail)
    assert "38 -> 50" in detail, detail
    assert state["to_run"] == 12, state


def test_dataparallel_is_a_stopper() -> None:
    """It leaked 17 MiB/step, ran 35 % slower, and crashed fp32. Never again by accident."""
    pf = _preflight()
    _, level, detail, fix = pf.check_config(
        _args(config=os.path.join(REPO_ROOT, "configs", "tiny.yaml"),
              overrides=["train.dataparallel=True"]), {})
    assert level == pf.STOP, (level, detail)
    assert "dataparallel" in fix.lower(), fix


def test_a_sane_config_passes() -> None:
    pf = _preflight()
    _, level, detail, _ = pf.check_config(
        _args(config=os.path.join(REPO_ROOT, "configs", "paper.yaml")), {})
    assert level == pf.OK, (level, detail)


def test_a_config_that_will_not_load_is_a_readable_stopper() -> None:
    """``load_cfg`` validates n_classes against n_list. Report that, do not re-raise it."""
    pf = _preflight()
    _, level, detail, _ = pf.check_config(
        _args(config=os.path.join(REPO_ROOT, "configs", "tiny.yaml"),
              overrides=["model.n_classes=4"]), {})
    assert level == pf.STOP, (level, detail)
    assert "n_classes" in detail, detail


def test_stale_cells_stop_but_unstamped_cells_only_warn() -> None:
    pf = _preflight()
    _, level, _, _ = pf.check_cells(_args(cells_src="kaggle_02_train.py",
                                          cells_sha="0" * 16), {})
    assert level == pf.STOP
    _, level, _, _ = pf.check_cells(_args(), {})
    assert level == pf.WARN


def test_a_check_that_raises_becomes_a_warning_not_a_crash() -> None:
    """A preflight that can take down the notebook is worse than the bugs it finds."""
    pf = _preflight()

    def exploding(_args_, _state):
        raise RuntimeError("boom")

    original = pf.CHECKS
    argv = sys.argv
    try:
        pf.CHECKS = (exploding,)
        sys.argv = ["12_preflight.py", "--for", "demo"]
        assert pf.main() == 0
    finally:
        pf.CHECKS = original
        sys.argv = argv


def test_strict_turns_warnings_into_a_stop() -> None:
    pf = _preflight()

    def warner(_args_, _state):
        return "made up", pf.WARN, "something to look at", "look at it"

    original = pf.CHECKS
    argv = sys.argv
    try:
        pf.CHECKS = (warner,)
        sys.argv = ["12_preflight.py", "--for", "demo"]
        assert pf.main() == 0
        sys.argv = ["12_preflight.py", "--for", "demo", "--strict"]
        assert pf.main() == 1
    finally:
        pf.CHECKS = original
        sys.argv = argv


def test_every_profile_is_runnable() -> None:
    """A typo in PROFILES must not wait for the one notebook that uses that name."""
    pf = _preflight()
    for profile in pf.PROFILES:
        for check in (pf.check_recipes, pf.check_checkpoint):
            name, level, detail, _ = check(_args(profile=profile), {})
            assert level in (pf.OK, pf.WARN, pf.STOP), (profile, name, level)


CHECKS = {name: fn for name, fn in sorted(globals().items()) if name.startswith("test_")}

if __name__ == "__main__":
    print(__doc__)
    sys.exit(run_checks(CHECKS))
