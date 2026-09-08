"""The mixing invariants. If these fail, every downstream number is meaningless."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks, store_and_bank  # noqa: E402


def test_mix_equals_sum_of_sources_plus_noise() -> None:
    """mix == sources.sum(0) + noise, for every N, clean and noisy."""
    from csnet.mixing import render_recipe, sample_recipe

    store, bank = store_and_bank()
    rng = np.random.default_rng(0)
    for n in range(1, 6):
        for p_clean in (0.0, 1.0):
            recipe = sample_recipe(store, bank, n, rng, p_clean=p_clean)
            out = render_recipe(recipe, store, bank)
            error = float(np.abs(out["mix"] - (out["sources"].sum(0) + out["noise"])).max())
            assert error < 1e-5, f"N={n} p_clean={p_clean}: max error {error:.2e}"
            assert out["sources"].shape == (n, 24000)
            assert out["mix"].dtype == np.float32


def test_mix_is_rms_normalised() -> None:
    """The anti-leak mitigation: every rendered mixture has unit RMS by construction."""
    from csnet.mixing import render_recipe, sample_recipe

    store, bank = store_and_bank()
    rng = np.random.default_rng(1)
    for n in range(1, 6):
        out = render_recipe(sample_recipe(store, bank, n, rng), store, bank)
        level = float(np.sqrt(np.mean(out["mix"] ** 2)))
        assert abs(level - 1.0) < 1e-3, f"N={n}: RMS {level:.4f}"


def test_recipe_csv_round_trip_is_bit_exact() -> None:
    """A written-then-read recipe must re-render identically, or the test set is not frozen."""
    from csnet.mixing import read_recipes, render_recipe, sample_recipe, write_recipes

    store, bank = store_and_bank()
    rng = np.random.default_rng(2)
    rows = [sample_recipe(store, bank, n, rng) for n in range(1, 6)]
    path = os.path.join(tempfile.mkdtemp(), "r.csv")
    write_recipes(rows, path)
    for original, restored in zip(rows, read_recipes(path)):
        a = render_recipe(original, store, bank)
        b = render_recipe(restored, store, bank)
        assert np.array_equal(a["mix"], b["mix"]), "mixture differs after a CSV round trip"
        assert np.array_equal(a["sources"], b["sources"]), "sources differ"
        assert np.array_equal(a["noise"], b["noise"]), "noise differs"


def test_no_speaker_appears_twice_in_a_mixture() -> None:
    """A train-100 speaker owns ~111 utterances, so this is the easy mistake to make."""
    from csnet.mixing import sample_recipe

    store, bank = store_and_bank()
    rng = np.random.default_rng(3)
    for n in (2, 3, 4, 5):
        for _ in range(120):
            recipe = sample_recipe(store, bank, n, rng)
            speakers = [str(store.speaker_ids[i]) for i in recipe["utt_idx"]]
            assert len(set(speakers)) == n, f"repeated speaker in an N={n} mixture: {speakers}"


def test_noise_is_deterministic_unit_rms_and_zero_mean() -> None:
    """`render` must reproduce `sample` exactly, or frozen recipes do not reproduce."""
    store, bank = store_and_bank()
    for kind in bank.kinds:
        a = bank.render(kind, 4242, 24000)
        b = bank.render(kind, 4242, 24000)
        assert np.array_equal(a, b), f"{kind} is not deterministic"
        assert abs(float(np.sqrt(np.mean(a ** 2))) - 1.0) < 1e-3, f"{kind} is not unit RMS"
        assert abs(float(a.mean())) < 1e-3, f"{kind} is not zero mean"
        assert a.dtype == np.float32


def test_babble_never_uses_a_target_speaker() -> None:
    """Babble drawn from a separable speaker would be an invisible leak."""
    store, _ = store_and_bank()
    targets = {str(s) for s in store.speaker_ids[store.target_idx].tolist()}
    babble = {str(s) for s in store.speaker_ids[store.babble_idx].tolist()}
    assert not (targets & babble), f"shared speakers: {sorted(targets & babble)[:5]}"
    assert babble, "no babble speakers were reserved at all"


def test_splits_are_speaker_disjoint() -> None:
    """Inherited from LibriMix, but re-asserted because it is the leak that matters."""
    from csnet.pack import SourceStore

    from conftest import tiny_store

    root = tiny_store()
    sets = {s: set(SourceStore(root, s).speaker_ids.tolist())
            for s in ("train-100", "dev", "test")}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = sets[names[i]] & sets[names[j]]
            assert not shared, f"{names[i]} and {names[j]} share {sorted(shared)[:5]}"


def test_store_round_trips_audio() -> None:
    """int16 packing must lose no more than one quantisation step."""
    from csnet.audio import read_wav
    from csnet.pack import scan_libri2mix

    from conftest import tiny_corpus

    store, _ = store_and_bank()
    found = scan_libri2mix(tiny_corpus(), "train-100")
    checked = 0
    for i, utt_id in enumerate(store.utt_ids[:8]):
        original, _ = read_wav(found[utt_id][0])
        packed = store.get(i)
        # the store keeps the highest-energy window, so compare the best alignment
        if original.size > packed.size:
            best = min(range(0, original.size - packed.size + 1, 800),
                       key=lambda s: float(np.abs(original[s:s + packed.size] - packed).max()))
            original = original[best:best + packed.size]
        error = float(np.abs(original[:packed.size] - packed[:original.size]).max())
        assert error <= 2.0 / 32767.0, f"{utt_id}: max error {error:.2e}"
        checked += 1
    assert checked > 0


CHECKS = {name: fn for name, fn in sorted(globals().items()) if name.startswith("test_")}

if __name__ == "__main__":
    print(__doc__)
    sys.exit(run_checks(CHECKS))
