"""Torch datasets: dynamic mixing for training, frozen recipes for dev/test.

``DynamicMixDataset`` invents a fresh mixture on every ``__getitem__`` -- a different
speaker set, crop, gain, noise and SNR each time. That is "dynamic mixing" (Route C in the
project record), worth +0.3-0.6 dB in the literature, and here it is free because the
sources are already packed as int16 in one memmap.

``FrozenMixDataset`` reads a committed recipe CSV and is byte-identical every run. The
evaluation sets never change; that is the point of them.

Both yield the same dict, so the same collate and the same training step handle either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import MAX_N_SRC, N_LIST, SEG_LEN, n_to_class
from .mixing import read_recipes, render_recipe, sample_recipe

if TYPE_CHECKING:  # pragma: no cover
    from .noise import NoiseBank
    from .pack import SourceStore


def _to_batch(rendered: dict, max_n_src: int, seg_len: int) -> dict:
    """Pack a rendered mixture into the fixed-shape dict the model and loss expect."""
    n_src = int(rendered["n_src"])
    refs = np.zeros((max_n_src, seg_len), dtype=np.float32)
    refs[:n_src] = rendered["sources"][:max_n_src]
    return {
        "mix": torch.from_numpy(np.ascontiguousarray(rendered["mix"])),
        "refs": torch.from_numpy(refs),
        "noise": torch.from_numpy(np.ascontiguousarray(rendered["noise"])),
        "n_src": torch.tensor(n_src, dtype=torch.int64),
        "cls": torch.tensor(n_to_class(n_src), dtype=torch.int64),
        "is_noisy": torch.tensor(int(rendered["is_noisy"]), dtype=torch.int64),
        "snr_db": torch.tensor(float(rendered["snr_db"]), dtype=torch.float32),
        "mix_id": str(rendered["mix_id"]),
    }


class DynamicMixDataset(Dataset):
    """Endless training set: one freshly sampled mixture per index."""

    def __init__(self, store: "SourceStore", bank: "NoiseBank", *,
                 n_list: Sequence[int] = N_LIST, steps: int = 200_000,
                 seg_len: int = SEG_LEN, seed: int = 72, max_n_src: int = MAX_N_SRC,
                 gain_db_range: tuple[float, float] = (-5.0, 5.0),
                 snr_db_range: tuple[float, float] = (0.0, 20.0),
                 p_clean: float = 0.2, n_weights: Sequence[float] | None = None,
                 min_crop_rms_ratio: float = 0.3) -> None:
        self.store = store
        self.bank = bank
        self.n_list = tuple(int(n) for n in n_list)
        self.steps = int(steps)
        self.seg_len = int(seg_len)
        self.seed = int(seed)
        self.max_n_src = int(max_n_src)
        self.gain_db_range = tuple(gain_db_range)
        self.snr_db_range = tuple(snr_db_range)
        self.p_clean = float(p_clean)
        self.min_crop_rms_ratio = float(min_crop_rms_ratio)
        self.epoch_salt = 0

        if n_weights is None:
            weights = np.ones(len(self.n_list), dtype=np.float64)
        else:
            weights = np.asarray(n_weights, dtype=np.float64)
            if weights.size != len(self.n_list):
                raise ValueError("n_weights must have the same length as n_list")
        self.n_weights = weights / weights.sum()

        if max(self.n_list) > len(store.speakers):
            raise ValueError(f"split {store.split!r} has only {len(store.speakers)} target "
                             f"speakers but n_list needs {max(self.n_list)} distinct ones")

    def set_epoch(self, epoch: int) -> None:
        """Change the per-item seed salt so each epoch draws different mixtures."""
        self.epoch_salt = int(epoch)

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng((self.seed, self.epoch_salt, int(idx)))
        n_src = int(rng.choice(self.n_list, p=self.n_weights))
        recipe = sample_recipe(
            self.store, self.bank, n_src, rng, seg_len=self.seg_len,
            gain_db_range=self.gain_db_range, snr_db_range=self.snr_db_range,
            p_clean=self.p_clean, min_crop_rms_ratio=self.min_crop_rms_ratio,
            mix_id=f"dyn{self.epoch_salt}_{idx}_n{n_src}")
        return _to_batch(render_recipe(recipe, self.store, self.bank, seg_len=self.seg_len),
                         self.max_n_src, self.seg_len)


class FrozenMixDataset(Dataset):
    """Deterministic evaluation set rendered from a committed recipe CSV."""

    def __init__(self, store: "SourceStore", bank: "NoiseBank", recipes_path: str, *,
                 seg_len: int = SEG_LEN, max_n_src: int = MAX_N_SRC,
                 n_list: Sequence[int] | None = None, limit: int | None = None) -> None:
        self.store = store
        self.bank = bank
        self.seg_len = int(seg_len)
        self.max_n_src = int(max_n_src)
        self.recipes_path = str(recipes_path)
        rows = read_recipes(recipes_path)
        if n_list is not None:
            keep = {int(n) for n in n_list}
            rows = [r for r in rows if int(r["n_src"]) in keep]
        if limit is not None:
            rows = rows[: int(limit)]
        self.recipes = rows

    def __len__(self) -> int:
        return len(self.recipes)

    def __getitem__(self, idx: int) -> dict:
        rendered = render_recipe(self.recipes[int(idx)], self.store, self.bank,
                                 seg_len=self.seg_len)
        return _to_batch(rendered, self.max_n_src, self.seg_len)

    def counts_per_n(self) -> dict[int, int]:
        """How many mixtures of each speaker count the frozen set holds."""
        out: dict[int, int] = {}
        for r in self.recipes:
            out[int(r["n_src"])] = out.get(int(r["n_src"]), 0) + 1
        return dict(sorted(out.items()))


def collate_mix(items: Sequence[dict]) -> dict:
    """Stack tensors, keep ``mix_id`` as a list of strings."""
    out: dict = {}
    for key in items[0]:
        if key == "mix_id":
            out[key] = [str(it[key]) for it in items]
        else:
            out[key] = torch.stack([it[key] for it in items], dim=0)
    return out


def build_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, num_workers: int,
                 seed: int = 72, drop_last: bool = False,
                 pin_memory: bool | None = None) -> torch.utils.data.DataLoader:
    """DataLoader with the settings that keep a Kaggle T4 fed (2-3 workers is plenty)."""
    from .utils import set_worker_seed

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    kwargs: dict = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "collate_fn": collate_mix,
        "pin_memory": bool(pin_memory),
        "drop_last": bool(drop_last),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
        kwargs["worker_init_fn"] = set_worker_seed
    return torch.utils.data.DataLoader(dataset, **kwargs)
