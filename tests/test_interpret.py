"""The interpretability collectors, and the contract they have to share.

``07_interpret.py`` hands the same loader to three different collectors. Two of them
treated ``max_batches=None`` as "the whole loader" and the third called ``int(None)`` on
it, so a run got all the way to the last section and died there -- after two and a half
minutes of real work, with the figures already written.

That is the expensive shape of bug: everything before it succeeded, so the output looks
like progress right up to the traceback.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks, store_and_bank, tiny_store  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))


def _model_and_loader(batches: int = 3, batch_size: int = 2):
    """A tiny model and a loader over synthetic mixtures, enough for a forward pass."""
    from csnet.datasets import DynamicMixDataset, build_loader
    from csnet.model import build_model

    store, bank = store_and_bank("train-100")
    dataset = DynamicMixDataset(store, bank, n_list=[1, 2, 3], seed=0,
                                steps=batches * batch_size, seg_len=8000, max_n_src=5)
    loader = build_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return build_model("tiny").eval(), loader


def test_every_collector_accepts_max_batches_none() -> None:
    """All three must agree that None means the whole loader.

    ``07_interpret.py`` sizes its subsets itself and then passes None so nothing slices
    them again. One collector disagreeing crashes the run at its last section.
    """
    from csnet.engine import evaluate
    from csnet.interpret import collect_mask_records, filter_importance_by_energy
    from csnet.losses import RectangularPITLoss

    model, loader = _model_and_loader(batches=3, batch_size=2)
    device = torch.device("cpu")

    importance = filter_importance_by_energy(model, loader, device, max_batches=None)
    assert importance.shape == (model.cfg.n_filters,), importance.shape
    assert np.isfinite(importance).all()

    records = collect_mask_records(model, loader, device, max_batches=None, max_n_src=5)
    assert len(records) == 6, len(records)

    loss_fn = RectangularPITLoss(max_n_src=5, n_classes=5)
    result = evaluate(model, loader, loss_fn, device, amp=False, max_batches=None,
                      max_n_src=5)
    assert result["batches"] == 3, result["batches"]


def test_none_would_have_crashed_the_old_way() -> None:
    """int(None) is the exact failure; keep a test that names it.

    Without this, someone tightening the signature back to ``max_batches: int`` sees
    three green tests above and no reason to look.
    """
    from csnet.interpret import filter_importance_by_energy
    import inspect

    signature = inspect.signature(filter_importance_by_energy)
    annotation = signature.parameters["max_batches"].annotation
    assert "None" in str(annotation), annotation


def test_max_batches_still_truncates_when_it_is_a_number() -> None:
    """None must not have quietly become "ignore the budget" for everyone."""
    from csnet.interpret import collect_mask_records

    model, loader = _model_and_loader(batches=3, batch_size=2)
    records = collect_mask_records(model, loader, torch.device("cpu"), max_batches=1,
                                   max_n_src=5)
    assert len(records) == 2, len(records)


def test_mask_records_carry_the_fields_the_figures_need() -> None:
    """A missing key here surfaces as a KeyError three sections into a GPU session."""
    from csnet.interpret import collect_mask_records, group_by_n

    model, loader = _model_and_loader(batches=2, batch_size=2)
    records = collect_mask_records(model, loader, torch.device("cpu"), max_batches=None,
                                   max_n_src=5)
    needed = {"sparsity_hoyer", "sparsity_gini", "overlap_cosine", "overlap_iou",
              "entropy", "active_fraction", "confidence", "n_true", "n_pred", "correct"}
    assert needed <= set(records[0]), sorted(needed - set(records[0]))

    grouped = group_by_n(records, ["sparsity_hoyer", "overlap_cosine"])
    assert grouped, "grouping produced nothing"
    for n, stats in grouped.items():
        assert stats["n"] >= 1
        if n == 1:
            # Overlap needs a pair; one source has none. NaN is the correct answer and
            # the figures rely on it staying NaN rather than becoming 0.
            assert not np.isfinite(stats["overlap_cosine"]), stats


CHECKS = {name: fn for name, fn in sorted(globals().items()) if name.startswith("test_")}

if __name__ == "__main__":
    print(__doc__)
    sys.exit(run_checks(CHECKS))
