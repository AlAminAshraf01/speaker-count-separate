"""Running the model on audio that is not exactly three seconds long.

The model has only ever seen a fixed 3 s crop divided by its RMS. Gate 2 used to hand it
whole ten-second utterances in one block and scored **0 correct counts out of 300**, while
the same checkpoint got 52 % on our own 3 s two-speaker mixtures. The audio was fine; the
input shape was not.

``csnet.inference.separate_long`` is the one procedure that feeds the model what it was
trained on, and both ``08_infer.py`` and gate 2 now call it. These tests hold the pieces
of it in place -- especially the cross-window alignment, whose absence produces output
that sounds plausible and swaps speakers at every hop.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks  # noqa: E402


def test_windows_cover_the_whole_signal() -> None:
    """Every sample must be inside some window, including the ragged tail."""
    from csnet.inference import window_starts

    for n in (24000, 30000, 80001, 24001):
        starts = window_starts(n, 24000, 12000)
        assert starts[0] == 0
        assert starts[-1] + 24000 >= n, (n, starts[-1])
        covered = np.zeros(n, dtype=bool)
        for s in starts:
            covered[s:s + 24000] = True
        assert covered.all(), n


def test_audio_shorter_than_a_window_still_runs() -> None:
    from csnet.inference import window_starts

    assert window_starts(1000, 24000, 12000) == [0]


def test_alignment_undoes_a_slot_swap() -> None:
    """Two slots swapped between windows must be matched back, not concatenated as-is."""
    from csnet.inference import align_permutation

    rng = np.random.default_rng(0)
    previous = rng.standard_normal((3, 500))
    current = previous[[1, 2, 0]]      # current[0] is previous[1], and so on
    # The mapping answers "which current slot continues previous slot i", so rotating the
    # slots one way inverts to the other -- and `est[mapping]` then puts them back.
    mapping = align_permutation(previous, current)
    assert mapping == [2, 0, 1], mapping
    assert np.allclose(current[mapping], previous)
    # A sign flip is the same speaker: SI-SDR does not care about polarity, so nor should
    # the matcher, or a phase-inverted window would start a new track.
    assert align_permutation(previous, -current) == [2, 0, 1]


def test_pooling_prefers_the_confident_windows() -> None:
    """mean_prob lets an unsure window abstain; a vote makes it shout."""
    from csnet.inference import pool_counts

    probs = np.array([[0.34, 0.33, 0.33],     # barely picks class 0
                      [0.34, 0.33, 0.33],     # barely picks class 0
                      [0.02, 0.96, 0.02]])    # certain it is class 1
    assert pool_counts(probs, "mode") == 0
    assert pool_counts(probs, "mean_prob") == 1


def test_separate_long_returns_the_input_length_and_one_prob_per_window() -> None:
    from csnet.constants import SR
    from csnet.inference import separate_long
    from csnet.model import build_model

    model = build_model("tiny").eval()
    audio = np.random.default_rng(1).standard_normal(int(7.5 * SR)).astype(np.float32)
    result = separate_long(model, audio, torch.device("cpu"), batch_size=4)

    assert result["est"].shape == (model.n_slots, len(audio)), result["est"].shape
    assert result["probs"].shape[0] == result["n_windows"]
    assert np.allclose(result["probs"].sum(axis=1), 1.0, atol=1e-5)
    assert result["n_windows"] > 1, "7.5 s should not be one window"
    assert np.isfinite(result["est"]).all()


def test_separate_long_handles_audio_shorter_than_one_window() -> None:
    from csnet.constants import SR
    from csnet.inference import separate_long
    from csnet.model import build_model

    model = build_model("tiny").eval()
    audio = np.random.default_rng(2).standard_normal(SR).astype(np.float32)   # 1 s
    result = separate_long(model, audio, torch.device("cpu"))
    assert result["est"].shape == (model.n_slots, SR)
    assert result["n_windows"] == 1


def test_overlap_add_weights_never_vanish() -> None:
    """No sample may be covered by near-zero weight, or dividing by it explodes.

    ``np.hanning`` is the symmetric window, which sums to a constant only to O(1/N) at
    50 % overlap -- and that does not matter, because the code divides by the accumulated
    weights, which cancels the ripple exactly. What would matter is a sample whose total
    weight is ~0: the endpoints of a Hann window are zero, so the first and last window
    must overlap something.
    """
    from csnet.inference import window_starts

    win, hop = 24000, 12000

    def profile(n: int) -> np.ndarray:
        weights = np.zeros(n)
        fade = np.hanning(win)
        for s in window_starts(n, win, hop):
            weights[s:s + win] += fade
        return weights

    for n in (24000 * 3, 80001, 40000, 24001):
        # Nowhere in the interior does the total weight approach zero, so the division
        # is always a weighted average of real contributions rather than an
        # amplification of one attenuated one.
        interior = profile(n)[hop:-hop]
        assert interior.min() > 0.99, (n, interior.min())

    # On an exact multiple of the hop the profile is flat to 2e-5. On a ragged length the
    # last window is pulled flush to the end and double-covers a stretch, so the profile
    # ripples by ~15 %. That does NOT ripple the audio: the code divides by these weights,
    # which cancels it exactly. Asserting flatness here would be asserting something the
    # algorithm does not need and does not have.
    flat = profile(24000 * 3)[hop:-hop]
    assert flat.std() / flat.mean() < 1e-3, flat.std() / flat.mean()

    # The first and last sample sit at a Hann zero and come out silent -- one sample at
    # each end of a multi-window signal, which nothing downstream can measure. Written
    # down so it reads as a property rather than as a surprise.
    assert profile(24000 * 3)[0] == 0.0


CHECKS = {name: fn for name, fn in sorted(globals().items()) if name.startswith("test_")}

if __name__ == "__main__":
    print(__doc__)
    sys.exit(run_checks(CHECKS))
