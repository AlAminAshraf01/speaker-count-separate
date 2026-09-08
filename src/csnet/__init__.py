"""csnet -- count the talkers, then separate them.

A single Conv-TasNet-style network with a max-N mask head, a dedicated noise slot and a
speaker-count head, plus everything needed to build its data, train it on a free Kaggle
GPU session, evaluate it honestly and open it up.

Nothing heavy is imported here on purpose: the CPU-only analysis scripts must be able to
``import csnet`` without paying for torch.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .constants import (  # noqa: F401
    CAP_LEN,
    CAP_SECONDS,
    EPS,
    MAX_N_SRC,
    N_CLASSES,
    N_LIST,
    SEG_LEN,
    SEG_SECONDS,
    SILENCE_DB,
    SR,
    class_to_n,
    n_to_class,
)

__all__ = [
    "__version__",
    "SR",
    "SEG_LEN",
    "SEG_SECONDS",
    "CAP_LEN",
    "CAP_SECONDS",
    "MAX_N_SRC",
    "N_LIST",
    "N_CLASSES",
    "SILENCE_DB",
    "EPS",
    "n_to_class",
    "class_to_n",
]
