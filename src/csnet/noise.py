"""Noise bank: synthetic colours, babble from held-out speakers, and optional real noise.

Two properties matter and are tested:

* every noise is **unit RMS and zero mean**, so the SNR maths in :mod:`csnet.mixing` is exact;
* :meth:`NoiseBank.render` reproduces byte-for-byte what :meth:`NoiseBank.sample` produced,
  given only ``(kind, noise_id, n)``. That is what lets the frozen test set be a CSV of
  recipes instead of gigabytes of WAV.

Babble is drawn only from speakers whose store role is ``babble``. Building babble from
speakers the model is also asked to separate would be an invisible leak.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from .audio import pad_or_trim
from .constants import EPS

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime dependency
    from .pack import SourceStore

ALL_KINDS: tuple[str, ...] = ("white", "pink", "brown", "babble", "real")
_MAX_ID = 2 ** 31 - 1


def _unit_rms(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-RMS, float32."""
    x = np.asarray(x, dtype=np.float32)
    x = (x - np.float32(x.mean())).astype(np.float32)
    level = np.float32(np.sqrt(np.mean(np.square(x.astype(np.float64)))) + EPS)
    return (x / level).astype(np.float32)


def _colored(n: int, exponent: float, seed: int) -> np.ndarray:
    """Gaussian noise shaped as ``1/f**exponent`` in the magnitude spectrum."""
    rng = np.random.default_rng(int(seed))
    white = rng.standard_normal(int(n)).astype(np.float32)
    if exponent == 0.0 or n < 4:
        return _unit_rms(white)
    spectrum = np.fft.rfft(white)
    freqs = np.arange(spectrum.size, dtype=np.float64)
    freqs[0] = 1.0  # leave DC alone instead of dividing by zero
    spectrum = spectrum / np.power(freqs, exponent)
    spectrum[0] = 0.0
    return _unit_rms(np.fft.irfft(spectrum, n=int(n)).astype(np.float32))


class NoiseBank:
    """Draws and deterministically re-renders background noise."""

    def __init__(self, store: "SourceStore | None" = None,
                 real: "SourceStore | None" = None,
                 kinds: Sequence[str] = ALL_KINDS,
                 weights: Sequence[float] | None = None) -> None:
        self.store = store
        self.real = real
        requested = [k for k in kinds if k in ALL_KINDS]
        if weights is not None and len(weights) == len(kinds):
            weight_map = {k: float(w) for k, w in zip(kinds, weights)}
        else:
            weight_map = {k: 1.0 for k in requested}

        available: list[str] = []
        for kind in requested:
            if kind == "babble" and (store is None or store.babble_idx.size == 0):
                continue
            if kind == "real" and (real is None or len(real) == 0):
                continue
            available.append(kind)
        if not available:
            available = ["white"]
            weight_map.setdefault("white", 1.0)

        self.kinds: tuple[str, ...] = tuple(available)
        raw = np.array([max(0.0, weight_map.get(k, 1.0)) for k in self.kinds], dtype=np.float64)
        if raw.sum() <= 0:
            raw = np.ones_like(raw)
        self.weights: np.ndarray = raw / raw.sum()

    # -- sampling ----------------------------------------------------------
    def sample(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, str, int]:
        """Draw a noise segment; returns ``(waveform, kind, noise_id)``."""
        kind = str(rng.choice(self.kinds, p=self.weights))
        noise_id = int(rng.integers(0, _MAX_ID))
        return self.render(kind, noise_id, n), kind, noise_id

    def render(self, kind: str, noise_id: int, n: int) -> np.ndarray:
        """Re-create exactly the noise ``sample`` produced for ``(kind, noise_id)``."""
        n = int(n)
        noise_id = int(noise_id)
        if kind == "none" or n <= 0:
            return np.zeros(max(0, n), dtype=np.float32)
        if kind == "white":
            return _colored(n, 0.0, noise_id)
        if kind == "pink":
            return _colored(n, 0.5, noise_id)
        if kind == "brown":
            return _colored(n, 1.0, noise_id)
        if kind == "babble":
            return self._babble(noise_id, n)
        if kind == "real":
            return self._real(noise_id, n)
        raise ValueError(f"unknown noise kind: {kind!r}")

    # -- kinds that need the store ----------------------------------------
    def _babble(self, noise_id: int, n: int) -> np.ndarray:
        """Sum of 4-8 held-out speech excerpts -- spectrally realistic background chatter."""
        if self.store is None or self.store.babble_idx.size == 0:
            return _colored(n, 0.5, noise_id)
        rng = np.random.default_rng(noise_id)
        pool = self.store.babble_idx
        n_voices = 4 + int(noise_id % 5)
        picks = rng.integers(0, pool.size, size=n_voices)
        total = np.zeros(n, dtype=np.float32)
        for pick in picks:
            index = int(pool[int(pick)])
            length = int(self.store.lengths[index])
            start = int(rng.integers(0, max(1, length - n + 1)))
            excerpt = pad_or_trim(self.store.get(index, start, n), n)
            if float(np.max(np.abs(excerpt))) < 1e-6:
                continue
            total = (total + _unit_rms(excerpt)).astype(np.float32)
        if float(np.max(np.abs(total))) < 1e-6:
            return _colored(n, 0.5, noise_id)
        return _unit_rms(total)

    def _real(self, noise_id: int, n: int) -> np.ndarray:
        """One excerpt from an attached real-noise corpus."""
        if self.real is None or len(self.real) == 0:
            return _colored(n, 0.0, noise_id)
        rng = np.random.default_rng(noise_id)
        index = int(rng.integers(0, len(self.real)))
        length = int(self.real.lengths[index])
        start = int(rng.integers(0, max(1, length - n + 1)))
        excerpt = pad_or_trim(self.real.get(index, start, n), n)
        if float(np.max(np.abs(excerpt))) < 1e-6:
            return _colored(n, 0.0, noise_id)
        return _unit_rms(excerpt)

    def describe(self) -> str:
        """Human-readable summary for logs."""
        parts = [f"{k}:{w:.2f}" for k, w in zip(self.kinds, self.weights)]
        return "NoiseBank(" + ", ".join(parts) + ")"
