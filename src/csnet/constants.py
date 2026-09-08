"""Global constants. Everything in this project is 8 kHz, mono, 3-second segments."""

from __future__ import annotations

SR: int = 8000
"""Sample rate, Hz. Fixed everywhere -- never resample internally."""

SEG_SECONDS: float = 3.0
"""Training / evaluation segment length in seconds."""

SEG_LEN: int = 24000
"""SEG_SECONDS * SR."""

CAP_SECONDS: float = 8.0
"""Maximum stored excerpt per source utterance (the highest-energy window)."""

CAP_LEN: int = 64000
"""CAP_SECONDS * SR."""

MAX_N_SRC: int = 5
"""Number of speaker output slots."""

N_LIST: tuple[int, ...] = (1, 2, 3, 4, 5)
"""Speaker counts the model is trained on. Class index is the position in this tuple."""

N_CLASSES: int = len(N_LIST)

SILENCE_DB: float = -30.0
"""Silence floor for surplus slots and for the P-SI-SNR pad, in dB relative to the mixture."""

EPS: float = 1e-8

INT16_SCALE: float = 32767.0

MIN_UTT_SECONDS: float = 1.0
"""Utterances shorter than this are dropped by the packer -- they make degenerate targets."""


def n_to_class(n: int) -> int:
    """Map a speaker count to its class index."""
    return N_LIST.index(int(n))


def class_to_n(c: int) -> int:
    """Map a class index back to its speaker count."""
    return N_LIST[int(c)]
