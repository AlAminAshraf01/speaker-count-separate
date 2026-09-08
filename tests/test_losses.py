"""The four loss invariants from docs/CONTRACT.md section 7.

Invariant 2 is the one that catches real bugs: a loss that is permutation invariant but
not *shift* invariant silently punishes the model for using a high-numbered slot, which
looks like slow convergence rather than a bug.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks  # noqa: E402

MAX_N, SLOTS, LENGTH, BATCH = 5, 6, 4000, 4


def _setup(label_smoothing: float = 0.0):
    from csnet.losses import RectangularPITLoss

    torch.manual_seed(0)
    loss_fn = RectangularPITLoss(max_n_src=MAX_N, predict_noise=True, clamp_si_sdr=30.0,
                                 label_smoothing=label_smoothing)
    n_src = torch.tensor([1, 2, 3, 5])
    refs = torch.zeros(BATCH, MAX_N, LENGTH)
    for b in range(BATCH):
        refs[b, : int(n_src[b])] = torch.randn(int(n_src[b]), LENGTH)
    batch = {"refs": refs, "mix": refs.sum(1), "noise": torch.zeros(BATCH, LENGTH),
             "n_src": n_src, "cls": n_src - 1,
             "is_noisy": torch.zeros(BATCH, dtype=torch.long)}

    def build(order):
        est = torch.zeros(BATCH, SLOTS, LENGTH)
        for b in range(BATCH):
            for i, slot in enumerate(order[b]):
                est[b, slot] = refs[b, i]
        logits = torch.zeros(BATCH, 5)
        logits[torch.arange(BATCH), n_src - 1] = 20.0
        return {"est": est, "count_logits": logits}

    identity = [list(range(int(n))) for n in n_src]
    return loss_fn, batch, build, identity, n_src


def test_invariant_1_permutation() -> None:
    """Permuting correct estimates among slots must not change the loss."""
    loss_fn, batch, build, identity, _ = _setup()
    base, _ = loss_fn(build(identity), batch)
    permuted, _ = loss_fn(build([list(reversed(o)) for o in identity]), batch)
    delta = abs(float(permuted - base))
    assert delta < 1e-3, f"loss moved by {delta:.2e} under permutation"


def test_invariant_2_slot_shift() -> None:
    """Shifting them into a different SUBSET of slots must not change it either."""
    loss_fn, batch, build, identity, n_src = _setup()
    base, _ = loss_fn(build(identity), batch)
    shifted = [[s + (MAX_N - int(n)) for s in o] for o, n in zip(identity, n_src)]
    moved, _ = loss_fn(build(shifted), batch)
    delta = abs(float(moved - base))
    assert delta < 1e-3, f"loss moved by {delta:.2e} under a slot shift"


def test_invariant_3_surplus_slots_must_stay_quiet() -> None:
    """Leaking the mixture into leftover slots must be heavily punished."""
    loss_fn, batch, build, identity, n_src = _setup()
    base, _ = loss_fn(build(identity), batch)
    leaky = build(identity)
    for b in range(BATCH):
        for slot in range(int(n_src[b]), MAX_N):
            leaky["est"][b, slot] = batch["mix"][b]
    worse, _ = loss_fn(leaky, batch)
    rise = float(worse - base)
    assert rise > 20.0, f"leaking the mixture only cost {rise:.2f}"


def test_invariant_4_perfect_estimates() -> None:
    """Perfect estimates sit at the clamp ceiling and the count term goes to zero."""
    loss_fn, batch, build, identity, _ = _setup(label_smoothing=0.0)
    _, logs = loss_fn(build(identity), batch)
    assert abs(logs["sep"] + 30.0) < 0.2, f"sep {logs['sep']:.3f}, expected about -30"
    assert logs["count"] < 0.05, f"count ce {logs['count']:.4f}"
    assert logs["acc"] == 1.0


def test_assignment_matches_scipy_hungarian() -> None:
    """Brute force over P(5,n) must equal the optimal rectangular assignment."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    from csnet.losses import RectangularPITLoss, pairwise_si_sdr

    torch.manual_seed(1)
    loss_fn = RectangularPITLoss(max_n_src=MAX_N, predict_noise=False, clamp_si_sdr=None)
    for n in range(1, 6):
        est = torch.randn(3, MAX_N, 800)
        refs = torch.zeros(3, MAX_N, 800)
        refs[:, :n] = torch.randn(3, n, 800)
        pw = pairwise_si_sdr(est, refs)
        best, _, _ = loss_fn._assign(pw, pw, n)
        for b in range(3):
            cost = -pw[b, :n, :].detach().numpy()
            rows, cols = linear_sum_assignment(cost)
            reference = float(np.mean([-cost[r, c] for r, c in zip(rows, cols)]))
            assert abs(float(best[b]) - reference) < 1e-3, \
                f"n={n} b={b}: brute force {float(best[b]):.4f} vs Hungarian {reference:.4f}"


def test_pairwise_si_sdr_shapes_and_values() -> None:
    """(B,S,T) x (B,R,T) -> (B,R,S), and a signal against itself is large."""
    from csnet.losses import pairwise_si_sdr

    est = torch.randn(2, 6, 1000)
    refs = torch.zeros(2, 5, 1000)
    refs[:, :3] = est[:, :3]
    out = pairwise_si_sdr(est, refs)
    assert out.shape == (2, 5, 6)
    for k in range(3):
        assert float(out[0, k, k]) > 50.0, f"self-similarity too low: {float(out[0, k, k])}"


def test_loss_is_finite_under_autocast_and_zero_inputs() -> None:
    """Degenerate batches must not produce NaN."""
    from csnet.losses import RectangularPITLoss

    loss_fn = RectangularPITLoss(max_n_src=MAX_N, predict_noise=True)
    batch = {"refs": torch.zeros(2, MAX_N, 500), "mix": torch.zeros(2, 500),
             "noise": torch.zeros(2, 500), "n_src": torch.tensor([1, 2]),
             "cls": torch.tensor([0, 1]), "is_noisy": torch.zeros(2, dtype=torch.long)}
    out = {"est": torch.zeros(2, SLOTS, 500), "count_logits": torch.zeros(2, 5)}
    total, logs = loss_fn(out, batch)
    assert torch.isfinite(total), f"loss is {total}"
    assert all(v == v for v in logs.values()), f"NaN in logs: {logs}"


CHECKS = {name: fn for name, fn in sorted(globals().items()) if name.startswith("test_")}

if __name__ == "__main__":
    print(__doc__)
    sys.exit(run_checks(CHECKS))
