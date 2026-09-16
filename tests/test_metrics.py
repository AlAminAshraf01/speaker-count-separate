"""Metric behaviour, especially P-SI-SNR under a wrong speaker count."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks  # noqa: E402


def _sources(n: int = 2, length: int = 8000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length) / 8000.0
    out = []
    for k in range(n):
        f0 = 120.0 + 90.0 * k
        wave = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 8))
        wave = wave * (0.5 + 0.5 * np.sin(2 * np.pi * (2.0 + k) * t))
        out.append((wave / np.abs(wave).max()).astype(np.float32))
    return np.stack(out) + 1e-4 * rng.standard_normal((n, length)).astype(np.float32)


def test_si_sdr_extremes() -> None:
    """Perfect estimate is huge; unrelated noise is around or below 0 dB."""
    from csnet.metrics import si_sdr

    x = _sources(1)[0]
    assert float(si_sdr(x, x)) > 100.0
    rng = np.random.default_rng(1)
    assert float(si_sdr(rng.standard_normal(x.size).astype(np.float32), x)) < 5.0


def test_si_sdr_is_scale_invariant() -> None:
    """The 'SI' in SI-SDR. Also why RMS-normalising the input is safe."""
    from csnet.metrics import si_sdr

    sources = _sources(2)
    mix = sources.sum(0)
    a = float(si_sdr(mix, sources[0]))
    b = float(si_sdr(mix * 7.3, sources[0]))
    assert abs(a - b) < 1e-3, f"{a:.4f} vs {b:.4f}"


def test_si_sdr_improvement_matches_definition() -> None:
    from csnet.metrics import si_sdr, si_sdr_improvement

    sources = _sources(2)
    mix = sources.sum(0)
    direct = float(si_sdr(sources[0], sources[0]) - si_sdr(mix, sources[0]))
    via = float(si_sdr_improvement(sources[0], sources[0], mix))
    assert abs(direct - via) < 1e-6


def test_matched_si_sdri_is_permutation_and_shift_invariant() -> None:
    """Evaluation must PIT-match too, not just training."""
    from csnet.metrics import matched_si_sdri

    sources = _sources(2)
    mix = sources.sum(0)
    aligned = np.zeros((5, sources.shape[1]), dtype=np.float32)
    aligned[0], aligned[1] = sources[0], sources[1]
    shifted = np.zeros_like(aligned)
    shifted[3], shifted[4] = sources[1], sources[0]
    a = np.mean(matched_si_sdri(aligned, sources, mix, 2))
    b = np.mean(matched_si_sdri(shifted, sources, mix, 2))
    assert abs(float(a) - float(b)) < 1e-6


def test_p_si_snr_penalises_a_wrong_count() -> None:
    """The whole point: it is defined, and worse, when n_pred != n_true."""
    from csnet.metrics import p_si_snr

    sources = _sources(3)
    est = np.zeros((6, sources.shape[1]), dtype=np.float32)
    est[:3] = sources
    correct = p_si_snr(est, sources, 3, 3, max_n_src=5)
    under = p_si_snr(est, sources, 3, 2, max_n_src=5)
    over = p_si_snr(est, sources, 3, 4, max_n_src=5)
    assert correct > under, f"under-counting was not penalised ({correct:.2f} vs {under:.2f})"
    assert correct > over, f"over-counting was not penalised ({correct:.2f} vs {over:.2f})"
    assert np.isfinite(under) and np.isfinite(over)


def test_p_si_snr_uses_the_loudest_slots() -> None:
    """Slot selection is by energy, so silent surplus slots are never picked."""
    from csnet.metrics import p_si_snr

    sources = _sources(2)
    est = np.zeros((6, sources.shape[1]), dtype=np.float32)
    est[2], est[4] = sources[0], sources[1]     # the true sources, in odd slots
    est[0] = 1e-6 * np.random.default_rng(0).standard_normal(sources.shape[1])
    assert p_si_snr(est, sources, 2, 2, max_n_src=5) > 40.0


def test_count_report_and_confusion() -> None:
    from csnet.metrics import count_report, format_confusion

    report = count_report([1, 2, 2, 3, 3, 3], [1, 2, 3, 3, 3, 2], n_list=(1, 2, 3))
    assert abs(report["accuracy"] - 4 / 6) < 1e-9
    assert abs(report["mae"] - 2 / 6) < 1e-9
    assert report["confusion"].sum() == 6
    assert report["support"] == {1: 1, 2: 2, 3: 3}
    text = format_confusion(report["confusion"], (1, 2, 3))
    assert "true\\pred" in text


def test_oracle_masks_beat_the_mixture() -> None:
    """IRM/IBM are an upper bound, so they must be clearly positive."""
    from csnet.baselines import ideal_ratio_mask_sisdri

    sources = _sources(2)
    mix = sources.sum(0)
    for mode in ("irm", "ibm"):
        values = ideal_ratio_mask_sisdri(mix, sources, mode=mode)
        assert float(np.mean(values)) > 3.0, f"{mode} oracle only got {np.mean(values):.2f} dB"


def test_summarise_per_n_ignores_miscounted_for_sisdri() -> None:
    from csnet.metrics import summarise_per_n

    records = [
        {"n_true": 2, "n_pred": 2, "p_si_snr": 10.0, "si_sdri": [8.0, 6.0], "input_si_sdr": 0.0},
        {"n_true": 2, "n_pred": 3, "p_si_snr": -5.0, "si_sdri": None, "input_si_sdr": 0.0},
    ]
    out = summarise_per_n(records, (2,))
    assert out[2]["n"] == 2 and out[2]["n_count_correct"] == 1
    assert abs(out[2]["count_acc"] - 0.5) < 1e-9
    assert abs(out[2]["si_sdri_count_correct"] - 7.0) < 1e-9


def test_a_clean_single_speaker_mixture_has_no_improvement_to_measure() -> None:
    """mix == s1, so SI-SDR improvement is a measurement of EPS, not of the model.

    This one line of arithmetic turned a real +1.2 dB into a reported -8.68 dB. With
    ``mix`` identical to the reference, ``si_sdr(mix, ref)`` is bounded only by EPS and
    lands near 124 dB for a 3 s unit-RMS signal; subtracting that from any honest
    estimate gives about -114 dB, and a quarter of the N=1 test set was exactly this.
    """
    from csnet.metrics import matched_si_sdri, si_sdr, usable_si_sdri

    rng = np.random.default_rng(0)
    source = rng.standard_normal(24000)
    source /= np.sqrt((source ** 2).mean())
    refs = source[None, :]
    mix = source.copy()                                   # a clean single speaker
    est = np.stack([0.9 * source + 0.1 * rng.standard_normal(24000)] + [
        rng.standard_normal(24000) for _ in range(4)])

    assert si_sdr(mix, source) > 100.0, si_sdr(mix, source)
    naive = matched_si_sdri(est, refs, mix, 1)
    assert naive[0] < -80.0, naive                        # what used to be averaged in

    usable, dropped = usable_si_sdri(est, refs, mix, 1)
    assert dropped == 1, dropped
    assert usable.size == 0, usable


def test_a_noisy_single_speaker_mixture_is_still_measured() -> None:
    """Denoising one speaker is a real task with a real improvement. Keep it."""
    from csnet.metrics import si_sdr, usable_si_sdri

    rng = np.random.default_rng(1)
    source = rng.standard_normal(24000)
    noise = rng.standard_normal(24000)
    source /= np.sqrt((source ** 2).mean())
    noise /= np.sqrt((noise ** 2).mean())
    mix = source + 0.178 * noise                          # about 15 dB SNR
    refs = source[None, :]
    est = np.stack([source + 0.05 * noise] + [rng.standard_normal(24000) for _ in range(4)])

    assert 10.0 < si_sdr(mix, source) < 25.0, si_sdr(mix, source)
    usable, dropped = usable_si_sdri(est, refs, mix, 1)
    assert dropped == 0, dropped
    assert usable.size == 1 and usable[0] > 5.0, usable


def test_ordinary_multi_speaker_mixtures_are_untouched() -> None:
    """The guard must not quietly change any number that was already correct."""
    from csnet.metrics import matched_si_sdri, usable_si_sdri

    rng = np.random.default_rng(2)
    refs = rng.standard_normal((3, 24000))
    mix = refs.sum(axis=0)
    est = np.stack([refs[i] + 0.2 * rng.standard_normal(24000) for i in range(3)]
                   + [rng.standard_normal(24000) for _ in range(2)])
    before = matched_si_sdri(est, refs, mix, 3)
    after, dropped = usable_si_sdri(est, refs, mix, 3)
    assert dropped == 0
    assert np.allclose(np.sort(before), np.sort(after)), (before, after)


def test_the_guard_is_wired_into_the_record_builder() -> None:
    """A correct metric that the evaluation does not call is not a fix.

    ``batch_records`` is what produced the -8.68 dB headline, so assert there: a clean
    single-speaker item must contribute no SI-SDRi and must say how many sources it
    dropped, while a two-speaker item beside it is unaffected.
    """
    import torch

    from csnet.engine import batch_records

    rng = np.random.default_rng(3)
    T = 8000
    solo = rng.standard_normal(T).astype(np.float32)
    pair = rng.standard_normal((2, T)).astype(np.float32)

    refs = np.zeros((2, 5, T), dtype=np.float32)
    refs[0, 0] = solo
    refs[1, :2] = pair
    mix = np.stack([solo, pair.sum(axis=0)])              # item 0: mix IS the reference
    est = rng.standard_normal((2, 6, T)).astype(np.float32)
    est[0, 0] = solo + 0.05 * rng.standard_normal(T)
    est[1, :2] = pair + 0.05 * rng.standard_normal((2, T))

    logits = np.full((2, 5), -10.0, dtype=np.float32)
    logits[0, 0] = 10.0                                    # predicts N=1, correctly
    logits[1, 1] = 10.0                                    # predicts N=2, correctly

    out = {"est": torch.from_numpy(est), "count_logits": torch.from_numpy(logits)}
    batch = {"refs": torch.from_numpy(refs), "mix": torch.from_numpy(mix),
             "n_src": torch.tensor([1, 2]), "is_noisy": torch.tensor([0, 0]),
             "snr_db": torch.tensor([0.0, 0.0])}
    records = batch_records(out, batch, max_n_src=5)

    assert records[0]["si_sdri"] is None, records[0]["si_sdri"]
    assert records[0]["n_degenerate"] == 1, records[0]
    assert records[1]["si_sdri"] is not None and len(records[1]["si_sdri"]) == 2
    assert records[1]["n_degenerate"] == 0
    assert min(records[1]["si_sdri"]) > 5.0, records[1]["si_sdri"]


def test_counted_and_scored_are_reported_separately() -> None:
    """`count % x mixes` must reconcile with a column, or the whole table looks wrong.

    A clean single-speaker mixture can be counted perfectly and still have nothing to
    score, so one number cannot mean both. Reporting only the scored count made N=1 read
    "64 % of 300 correct" beside "122", which invites the reader to distrust the table.
    """
    from csnet.metrics import summarise_per_n

    records = [
        {"n_true": 1, "n_pred": 1, "p_si_snr": 8.0, "si_sdri": None,       # counted, not scored
         "n_degenerate": 1, "input_si_sdr": 124.0},
        {"n_true": 1, "n_pred": 1, "p_si_snr": 8.0, "si_sdri": [2.0],      # counted and scored
         "n_degenerate": 0, "input_si_sdr": 12.0},
        {"n_true": 1, "n_pred": 3, "p_si_snr": -9.0, "si_sdri": None,      # miscounted
         "n_degenerate": 0, "input_si_sdr": 11.0},
    ]
    stats = summarise_per_n(records, [1])[1]
    assert stats["n"] == 3
    assert abs(stats["count_acc"] - 2 / 3) < 1e-9, stats["count_acc"]
    assert stats["n_count_correct"] == 2, stats           # matches count_acc x n
    assert stats["n_scored"] == 1, stats                  # only one had anything to score
    assert abs(stats["si_sdri_count_correct"] - 2.0) < 1e-9


CHECKS = {name: fn for name, fn in sorted(globals().items()) if name.startswith("test_")}

if __name__ == "__main__":
    print(__doc__)
    sys.exit(run_checks(CHECKS))
