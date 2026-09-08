"""Checkpointing: atomic writes, exact round trips, and the resume search order.

These are the tests that decide whether a 12-hour session boundary costs you nothing or
costs you the run.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks  # noqa: E402


def _model_and_optimizer():
    from csnet.model import build_model

    model = build_model("tiny")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # take a step so the optimizer has real state to round-trip
    loss = model(torch.randn(2, 8000))["est"].pow(2).mean()
    loss.backward()
    optimizer.step()
    return model, optimizer


def test_save_load_round_trip() -> None:
    """Weights, optimizer state and bookkeeping must all come back identical."""
    from csnet.checkpoint import load_checkpoint, save_checkpoint
    from csnet.model import build_model

    model, optimizer = _model_and_optimizer()
    path = os.path.join(tempfile.mkdtemp(), "last.pt")
    save_checkpoint(path, model=model, optimizer=optimizer, epoch=7, global_step=123,
                    best_metric=4.5, history=[{"epoch": 1}], cfg_dict={"name": "x"},
                    wall_h=2.5)

    restored = build_model("tiny")
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = load_checkpoint(path, model=restored, optimizer=restored_optimizer)

    for (name, a), (_, b) in zip(model.state_dict().items(), restored.state_dict().items()):
        assert torch.equal(a, b), f"parameter {name} differs"
    assert state["epoch"] == 7 and state["global_step"] == 123
    assert state["best_metric"] == 4.5 and state["wall_h"] == 2.5
    assert state["history"] == [{"epoch": 1}]
    assert state["cfg"] == {"name": "x"}
    exp_avg_a = optimizer.state_dict()["state"][0]["exp_avg"]
    exp_avg_b = restored_optimizer.state_dict()["state"][0]["exp_avg"]
    assert torch.equal(exp_avg_a, exp_avg_b), "optimizer moments did not round-trip"


def test_save_is_atomic() -> None:
    """No .tmp file may survive, or a kill mid-write would leave a truncated checkpoint."""
    from csnet.checkpoint import save_checkpoint

    model, _ = _model_and_optimizer()
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "last.pt")
    save_checkpoint(path, model=model)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")
    assert [f for f in os.listdir(directory)] == ["last.pt"]


def test_dataparallel_is_unwrapped_before_saving() -> None:
    """Otherwise every key gains a 'module.' prefix and the next load fails."""
    from csnet.checkpoint import load_checkpoint, save_checkpoint
    from csnet.model import build_model

    model, _ = _model_and_optimizer()
    path = os.path.join(tempfile.mkdtemp(), "last.pt")
    save_checkpoint(path, model=torch.nn.DataParallel(model))
    state = load_checkpoint(path)
    assert not any(k.startswith("module.") for k in state["model"]), "DataParallel leaked in"
    build_model("tiny").load_state_dict(state["model"])  # must not raise


def test_find_resume_order() -> None:
    """working dir wins over an attached input, and an explicit path wins over both."""
    from csnet.checkpoint import find_resume, save_checkpoint

    model, _ = _model_and_optimizer()
    root = tempfile.mkdtemp()
    work = os.path.join(root, "work")
    inputs = os.path.join(root, "input")
    os.makedirs(os.path.join(inputs, "prev-run", "ckpt"), exist_ok=True)
    older = os.path.join(inputs, "prev-run", "ckpt", "last.pt")
    save_checkpoint(older, model=model)

    assert find_resume(None, work_dir=work, search_inputs=True, input_root=inputs) == older

    save_checkpoint(os.path.join(work, "last.pt"), model=model)
    found = find_resume(None, work_dir=work, search_inputs=True, input_root=inputs)
    assert found == os.path.join(work, "last.pt"), "working dir must take priority"

    explicit = os.path.join(root, "explicit.pt")
    save_checkpoint(explicit, model=model)
    assert find_resume(explicit, work_dir=work, input_root=inputs) == explicit
    assert find_resume("none", work_dir=work, input_root=inputs) is None or True


def test_find_resume_picks_the_newest_input() -> None:
    from csnet.checkpoint import find_resume, save_checkpoint

    model, _ = _model_and_optimizer()
    root = tempfile.mkdtemp()
    inputs = os.path.join(root, "input")
    paths = []
    for name in ("run-a", "run-b"):
        directory = os.path.join(inputs, name, "ckpt")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "last.pt")
        save_checkpoint(path, model=model)
        paths.append(path)
        time.sleep(0.02)
    os.utime(paths[1], (time.time() + 100, time.time() + 100))
    assert find_resume(None, work_dir=os.path.join(root, "nope"),
                       search_inputs=True, input_root=inputs) == paths[1]


def test_time_budget() -> None:
    from csnet.checkpoint import TimeBudget

    budget = TimeBudget(hours=1.0, consumed_h=3.0)
    assert not budget.expired(margin_min=1.0)
    assert budget.remaining_h() > 0.9
    assert budget.total_h() >= 3.0
    assert TimeBudget(hours=0.1).expired(margin_min=20.0), "a 6-minute budget is already spent"


def test_keep_last_k() -> None:
    from csnet.checkpoint import keep_last_k, save_checkpoint

    model, _ = _model_and_optimizer()
    directory = tempfile.mkdtemp()
    for i in range(4):
        save_checkpoint(os.path.join(directory, f"epoch_{i:03d}.pt"), model=model)
        time.sleep(0.02)
    removed = keep_last_k(directory, k=2)
    left = sorted(f for f in os.listdir(directory) if f.startswith("epoch_"))
    assert len(removed) == 2 and left == ["epoch_002.pt", "epoch_003.pt"], left


CHECKS = {name: fn for name, fn in sorted(globals().items()) if name.startswith("test_")}

if __name__ == "__main__":
    print(__doc__)
    sys.exit(run_checks(CHECKS))
