"""Dataset contracts: batch shapes, determinism, and no speaker twice in a mixture."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks, store_and_bank  # noqa: E402

EXPECTED_KEYS = {"mix", "refs", "noise", "n_src", "cls", "is_noisy", "snr_db", "mix_id"}


def test_item_keys_shapes_and_dtypes() -> None:
    from csnet.datasets import DynamicMixDataset

    store, bank = store_and_bank()
    dataset = DynamicMixDataset(store, bank, steps=16, seed=5)
    item = dataset[0]
    assert set(item) == EXPECTED_KEYS, set(item) ^ EXPECTED_KEYS
    assert item["mix"].shape == (24000,) and item["mix"].dtype == torch.float32
    assert item["refs"].shape == (5, 24000) and item["refs"].dtype == torch.float32
    assert item["noise"].shape == (24000,)
    assert item["n_src"].dtype == torch.int64 and item["n_src"].ndim == 0
    assert item["cls"].dtype == torch.int64 and item["cls"].ndim == 0
    assert isinstance(item["mix_id"], str)


def test_refs_are_zero_padded_past_n_src() -> None:
    """The loss relies on this: slots beyond n_src must be exactly zero."""
    from csnet.datasets import DynamicMixDataset

    store, bank = store_and_bank()
    dataset = DynamicMixDataset(store, bank, steps=64, seed=6)
    seen = set()
    for i in range(48):
        item = dataset[i]
        n = int(item["n_src"])
        seen.add(n)
        padding = item["refs"][n:]
        if padding.numel():  # empty when n == max_n_src, which is fine
            assert float(padding.abs().max()) == 0.0, f"padding is non-zero at N={n}"
        assert float(item["refs"][:n].abs().max()) > 0.0, "a real source is silent"
    assert len(seen) >= 3, f"only saw counts {sorted(seen)}"


def test_class_label_matches_n_src() -> None:
    from csnet.constants import N_LIST
    from csnet.datasets import DynamicMixDataset

    store, bank = store_and_bank()
    dataset = DynamicMixDataset(store, bank, steps=40, seed=7)
    for i in range(40):
        item = dataset[i]
        assert N_LIST[int(item["cls"])] == int(item["n_src"])


def test_determinism_given_a_seed() -> None:
    """Same seed and epoch -> identical mixtures; a new epoch -> different ones."""
    from csnet.datasets import DynamicMixDataset

    store, bank = store_and_bank()
    a = DynamicMixDataset(store, bank, steps=8, seed=11)
    b = DynamicMixDataset(store, bank, steps=8, seed=11)
    assert torch.equal(a[3]["mix"], b[3]["mix"]), "not reproducible at a fixed seed"

    b.set_epoch(1)
    assert not torch.equal(a[3]["mix"], b[3]["mix"]), "set_epoch did not change the draw"

    c = DynamicMixDataset(store, bank, steps=8, seed=12)
    assert not torch.equal(a[3]["mix"], c[3]["mix"]), "different seeds gave the same mixture"


def test_no_speaker_twice_over_many_samples() -> None:
    from csnet.datasets import DynamicMixDataset
    from csnet.mixing import sample_recipe

    store, bank = store_and_bank()
    rng = np.random.default_rng(21)
    clashes = 0
    for _ in range(500):
        n = int(rng.integers(2, 6))
        recipe = sample_recipe(store, bank, n, rng)
        speakers = [str(store.speaker_ids[i]) for i in recipe["utt_idx"]]
        clashes += int(len(set(speakers)) != n)
    assert clashes == 0, f"{clashes} of 500 mixtures repeated a speaker"


def test_collate_and_loader_with_workers() -> None:
    """mix_id stays a list of str; workers must not crash on the memmap."""
    from csnet.datasets import DynamicMixDataset, build_loader

    store, bank = store_and_bank()
    dataset = DynamicMixDataset(store, bank, steps=16, seed=9)
    for workers in (0, 2):
        loader = build_loader(dataset, batch_size=4, shuffle=False, num_workers=workers,
                              pin_memory=False)
        batch = next(iter(loader))
        assert batch["mix"].shape == (4, 24000)
        assert batch["refs"].shape == (4, 5, 24000)
        assert batch["n_src"].shape == (4,)
        assert isinstance(batch["mix_id"], list) and len(batch["mix_id"]) == 4
        del loader


def test_frozen_dataset_is_stable_and_reports_counts() -> None:
    from csnet.datasets import FrozenMixDataset
    from csnet.mixing import sample_recipe, write_recipes

    store, bank = store_and_bank()
    rng = np.random.default_rng(31)
    rows = [sample_recipe(store, bank, n, rng, mix_id=f"m{n}_{i}")
            for n in (1, 2, 3) for i in range(4)]
    path = os.path.join(tempfile.mkdtemp(), "recipes.csv")
    write_recipes(rows, path)

    a = FrozenMixDataset(store, bank, path)
    b = FrozenMixDataset(store, bank, path)
    assert len(a) == 12
    assert a.counts_per_n() == {1: 4, 2: 4, 3: 4}
    assert torch.equal(a[5]["mix"], b[5]["mix"]), "the frozen set is not reproducible"

    filtered = FrozenMixDataset(store, bank, path, n_list=[2])
    assert len(filtered) == 4 and filtered.counts_per_n() == {2: 4}


def test_n_weights_bias_the_draw() -> None:
    from csnet.datasets import DynamicMixDataset

    store, bank = store_and_bank()
    dataset = DynamicMixDataset(store, bank, n_list=(1, 2), n_weights=(0.0, 1.0),
                                steps=32, seed=13)
    counts = {int(dataset[i]["n_src"]) for i in range(24)}
    assert counts == {2}, f"n_weights ignored: saw {counts}"


CHECKS = {name: fn for name, fn in sorted(globals().items()) if name.startswith("test_")}

if __name__ == "__main__":
    print(__doc__)
    sys.exit(run_checks(CHECKS))
