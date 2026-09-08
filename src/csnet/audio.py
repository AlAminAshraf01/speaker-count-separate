"""Audio primitives: level maths, int16 packing, resampling, STFT, energy windows.

Every waveform in this project is float32, mono, in [-1, 1], shape ``(..., T)``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .constants import EPS, INT16_SCALE, SR


# --------------------------------------------------------------------------- levels

def rms(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> Any:
    """Root-mean-square level."""
    return np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)), axis=axis,
                           keepdims=keepdims)).astype(np.float32)


def rms_normalize(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Divide by RMS. This is the anti-leak mitigation -- apply to every model input."""
    x = np.asarray(x, dtype=np.float32)
    return (x / (float(rms(x)) + eps)).astype(np.float32)


def peak_normalize(x: np.ndarray, peak: float = 0.9, eps: float = EPS) -> np.ndarray:
    """Scale so the maximum absolute sample equals ``peak``."""
    x = np.asarray(x, dtype=np.float32)
    return (x * (peak / (float(np.max(np.abs(x))) + eps))).astype(np.float32)


def db_to_lin(db: float | np.ndarray) -> Any:
    """Convert a dB gain to a linear gain."""
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 20.0)


def lin_to_db(gain: float | np.ndarray, eps: float = EPS) -> Any:
    """Convert a linear gain to dB."""
    return 20.0 * np.log10(np.abs(np.asarray(gain, dtype=np.float64)) + eps)


def power_db(x: np.ndarray, eps: float = EPS) -> float:
    """Mean power of a signal, in dB."""
    return float(10.0 * np.log10(float(np.mean(np.square(np.asarray(x, dtype=np.float64)))) + eps))


# --------------------------------------------------------------------------- int16

def float_to_int16(x: np.ndarray) -> np.ndarray:
    """Quantise float [-1, 1] to int16 with clipping."""
    y = np.asarray(x, dtype=np.float32) * INT16_SCALE
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def int16_to_float(x: np.ndarray) -> np.ndarray:
    """Dequantise int16 back to float32 in [-1, 1]."""
    return (np.asarray(x, dtype=np.float32) / INT16_SCALE).astype(np.float32)


# --------------------------------------------------------------------------- io

def read_wav(path: str, sr: int | None = SR, mono: bool = True) -> tuple[np.ndarray, int]:
    """Read an audio file as float32; resample to ``sr`` when given."""
    import soundfile as sf

    x, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1 and mono:
        x = x.mean(axis=1)
    x = np.ascontiguousarray(x, dtype=np.float32)
    if sr is not None and file_sr != sr:
        x = resample_to(x, file_sr, sr)
        file_sr = sr
    return x, int(file_sr)


def write_wav(path: str, x: np.ndarray, sr: int = SR, subtype: str = "PCM_16") -> None:
    """Write a mono float32 waveform."""
    import soundfile as sf

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    sf.write(path, np.asarray(x, dtype=np.float32), int(sr), subtype=subtype)


def audio_frames(path: str) -> int:
    """Number of frames in an audio file, read from the header only (cheap)."""
    import soundfile as sf

    return int(sf.info(path).frames)


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resample. Exact for the rational ratios we use (16k -> 8k)."""
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(np.asarray(x, dtype=np.float32), sr_out // g, sr_in // g).astype(np.float32)


# --------------------------------------------------------------------------- shaping

def pad_or_trim(x: np.ndarray, n: int, start: int = 0) -> np.ndarray:
    """Return exactly ``n`` samples starting at ``start``, zero-padded on the right."""
    x = np.asarray(x, dtype=np.float32)
    seg = x[start:start + n]
    if seg.shape[-1] < n:
        seg = np.pad(seg, (0, n - seg.shape[-1]))
    return np.ascontiguousarray(seg, dtype=np.float32)


def moving_rms(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """RMS of every ``win``-sample frame, computed with a cumulative sum (O(n))."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < win:
        return np.array([float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0])
    power = np.concatenate([[0.0], np.cumsum(x ** 2)])
    starts = np.arange(0, x.size - win + 1, hop)
    sums = power[starts + win] - power[starts]
    return np.sqrt(sums / win).astype(np.float32)


def best_energy_window(x: np.ndarray, win: int, hop: int = 800) -> int:
    """Start index of the highest-energy ``win``-sample window.

    A random crop of a LibriSpeech utterance often lands in silence, which makes a
    degenerate separation target. Keeping the loudest window avoids that.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size <= win:
        return 0
    levels = moving_rms(x, win, hop)
    return int(np.argmax(levels) * hop)


def crop_rms_ratio(x: np.ndarray, start: int, n: int, eps: float = EPS) -> float:
    """RMS of a crop divided by the RMS of the whole signal -- a silence detector."""
    full = float(rms(x)) + eps
    return float(rms(pad_or_trim(x, n, start))) / full


# --------------------------------------------------------------------------- spectra

def stft(x: np.ndarray, n_fft: int = 256, hop: int = 64, sr: int = SR) -> np.ndarray:
    """Complex STFT, shape ``(freq, frames)``. Thin wrapper over scipy for consistency."""
    from scipy.signal import stft as _stft

    _, _, spec = _stft(np.asarray(x, dtype=np.float32), fs=sr, nperseg=n_fft,
                       noverlap=n_fft - hop, boundary="zeros", padded=True)
    return spec


def istft(spec: np.ndarray, n_fft: int = 256, hop: int = 64, sr: int = SR,
          length: int | None = None) -> np.ndarray:
    """Inverse of :func:`stft`, optionally cropped/padded to ``length``."""
    from scipy.signal import istft as _istft

    _, x = _istft(spec, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=True)
    x = np.asarray(x, dtype=np.float32)
    if length is not None:
        x = pad_or_trim(x, length)
    return x


def magnitude_spectrogram(x: np.ndarray, n_fft: int = 512, hop: int = 256) -> np.ndarray:
    """Magnitude spectrogram via a strided rFFT -- faster than scipy for scalar features."""
    x = np.asarray(x, dtype=np.float32)
    if x.size < n_fft:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float32)
    n_frames = 1 + (x.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    window = np.hanning(n_fft).astype(np.float32)
    return (np.abs(np.fft.rfft(x[idx] * window, axis=-1)) + EPS).astype(np.float32)


def estimate_f0(x: np.ndarray, sr: int = SR, fmin: float = 60.0, fmax: float = 400.0) -> float:
    """Crude autocorrelation pitch estimate, used only as an EDA grouping proxy."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if x.size < int(sr / fmin) * 2 or float(np.std(x)) < 1e-6:
        return float("nan")
    spec = np.fft.rfft(x, n=2 * x.size)
    acf = np.fft.irfft(spec * np.conj(spec))[: x.size]
    lo, hi = int(sr / fmax), min(int(sr / fmin), x.size - 1)
    if hi <= lo:
        return float("nan")
    lag = int(np.argmax(acf[lo:hi]) + lo)
    return float(sr / lag) if lag > 0 else float("nan")
