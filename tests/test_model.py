"""Model shape contracts. Output length must equal input length -- exactly, always."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks  # noqa: E402


def test_output_length_equals_input_length() -> None:
    """Including lengths that are not a multiple of the encoder stride."""
    from csnet.model import build_model

    model = build_model("tiny").eval()
    for length in (8000, 24000, 24001, 4321, 1000):
        with torch.no_grad():
            out = model(torch.randn(2, length))
        assert out["est"].shape == (2, model.n_slots, length), \
            f"T={length}: got {tuple(out['est'].shape)}"


def test_accepts_both_input_ranks() -> None:
    """(B, T) and (B, 1, T) must behave identically."""
    from csnet.model import build_model

    model = build_model("tiny").eval()
    x = torch.randn(2, 8000)
    with torch.no_grad():
        a = model(x)["est"]
        b = model(x.unsqueeze(1))["est"]
    assert torch.allclose(a, b, atol=1e-5)


def test_count_head_and_slot_counts() -> None:
    """5 speaker slots plus one noise slot; 5 count classes."""
    from csnet.model import ModelConfig, build_model

    model = build_model("tiny").eval()
    assert model.n_slots == 6
    assert model.noise_slot == 5
    with torch.no_grad():
        out = model(torch.randn(3, 8000))
    assert out["count_logits"].shape == (3, 5)

    without = build_model(ModelConfig(n_filters=64, kernel=32, bottleneck=32, hidden=64,
                                      skip=32, n_blocks=2, n_repeats=1,
                                      predict_noise=False)).eval()
    assert without.n_slots == 5 and without.noise_slot is None


def test_return_internals_shapes() -> None:
    """masks (B, slots, N, F), enc (B, N, F), feat (B, Sc, F)."""
    from csnet.model import build_model

    model = build_model("tiny").eval()
    with torch.no_grad():
        out = model(torch.randn(2, 24000), return_internals=True)
    frames = out["enc"].shape[-1]
    assert out["masks"].shape == (2, model.n_slots, model.cfg.n_filters, frames)
    assert out["enc"].shape == (2, model.cfg.n_filters, frames)
    assert out["feat"].shape == (2, model.cfg.skip, frames)


def test_preset_parameter_counts() -> None:
    """The paper preset must land near the 5.2 M the project record budgeted for."""
    from csnet.model import build_model

    paper = build_model("paper").count_params()
    assert 4.8e6 < paper < 5.8e6, f"paper preset has {paper / 1e6:.2f} M parameters"
    assert build_model("small").count_params() < paper
    assert build_model("tiny").count_params() < build_model("small").count_params()


def test_extra_speaker_slot_is_cheap() -> None:
    """The only N-dependent part is the mask head: about 66 k parameters per speaker."""
    from csnet.model import ModelConfig, build_model

    base = dict(n_filters=512, kernel=16, bottleneck=128, hidden=512, skip=128,
                n_blocks=8, n_repeats=3, predict_noise=False)
    four = build_model(ModelConfig(max_n_src=4, n_classes=4, **base)).count_params()
    five = build_model(ModelConfig(max_n_src=5, n_classes=5, **base)).count_params()
    delta = five - four
    assert 60_000 < delta < 72_000, f"an extra slot cost {delta} parameters"


def test_dataparallel_wrapping() -> None:
    """DataParallel gathers a dict of tensors along the batch dimension."""
    from csnet.model import build_model

    model = torch.nn.DataParallel(build_model("tiny").eval())
    with torch.no_grad():
        out = model(torch.randn(4, 8000))
    assert out["est"].shape[0] == 4 and out["count_logits"].shape == (4, 5)


def test_gradients_reach_every_parameter() -> None:
    """A dead branch (typically the count head) shows up here and nowhere else."""
    from csnet.losses import RectangularPITLoss
    from csnet.model import build_model

    model = build_model("tiny").train()
    loss_fn = RectangularPITLoss(max_n_src=5, predict_noise=True)
    refs = torch.zeros(2, 5, 8000)
    refs[:, :2] = torch.randn(2, 2, 8000) * 0.3
    batch = {"refs": refs, "mix": refs.sum(1), "noise": torch.zeros(2, 8000),
             "n_src": torch.tensor([2, 2]), "cls": torch.tensor([1, 1]),
             "is_noisy": torch.zeros(2, dtype=torch.long)}
    loss, _ = loss_fn(model(batch["mix"]), batch)
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
    assert not missing, f"no gradient reached: {missing[:6]}"


def test_causal_mode_builds_and_runs() -> None:
    """`causal: true` switches to cLN and left-only padding."""
    from csnet.model import ModelConfig, build_model

    model = build_model(ModelConfig(n_filters=64, kernel=32, bottleneck=32, hidden=64,
                                    skip=32, n_blocks=3, n_repeats=1, causal=True)).eval()
    assert model.cfg.norm == "cLN"
    with torch.no_grad():
        out = model(torch.randn(1, 8000))
    assert out["est"].shape == (1, model.n_slots, 8000)


CHECKS = {name: fn for name, fn in sorted(globals().items()) if name.startswith("test_")}

if __name__ == "__main__":
    print(__doc__)
    sys.exit(run_checks(CHECKS))
