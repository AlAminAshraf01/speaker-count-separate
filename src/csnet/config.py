"""Configuration: dataclasses, YAML files with single-level inheritance, dotted overrides.

Every script takes ``--config configs/paper.yaml --set train.lr=1e-3 model.kernel=32``,
so a run is reproducible from one line and a checkpoint carries the config that made it.
"""

from __future__ import annotations

import hashlib
import json
import os
from ast import literal_eval
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Sequence

from .constants import MAX_N_SRC, N_LIST, SEG_SECONDS
from .model import ModelConfig


@dataclass
class DataCfg:
    """Where the packed store lives and how mixtures are drawn from it."""

    store_root: str = "/kaggle/input/csnet-store"
    train_split: str = "train-100"
    dev_split: str = "dev"
    test_split: str = "test"
    recipes_dir: str = "data"
    recipes_dev: str = "recipes_dev.csv"
    recipes_test: str = "recipes_test.csv"
    seg_seconds: float = SEG_SECONDS
    n_list: tuple[int, ...] = N_LIST
    n_weights: tuple[float, ...] | None = None
    gain_db_range: tuple[float, float] = (-5.0, 5.0)
    snr_db_range: tuple[float, float] = (0.0, 20.0)
    p_clean: float = 0.2
    min_crop_rms_ratio: float = 0.3
    noise_kinds: tuple[str, ...] = ("white", "pink", "brown", "babble", "real")
    noise_weights: tuple[float, ...] | None = None
    noise_store: str | None = None
    mmap: bool = True


@dataclass
class TrainCfg:
    """Optimisation, the session clock, and everything about resuming."""

    batch_size: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 60
    steps_per_epoch: int = 1000
    num_workers: int = 2
    amp: bool = True
    grad_clip: float = 5.0
    accum: int = 1
    warmup_steps: int = 500
    sched: str = "cosine"
    time_budget_h: float = 11.0
    seed: int = 72
    dataparallel: bool = True
    val_every: int = 1
    val_batches: int | None = None
    ckpt_dir: str = "/kaggle/working/ckpt"
    ckpt_every_steps: int = 400
    keep_last_k: int = 1
    freeze_separator: bool = False
    early_stop_patience: int = 0
    log_every: int = 100


@dataclass
class LossCfg:
    """Term weights. ``w_count = 0`` trains a pure separator; see docs/DESIGN.md section 10."""

    w_sep: float = 1.0
    w_sil: float = 1.0
    w_count: float = 0.5
    w_noise: float = 0.2
    silence_db: float = -30.0
    label_smoothing: float = 0.05
    clamp_si_sdr: float | None = 30.0


@dataclass
class Cfg:
    """The whole run."""

    name: str = "csnet"
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainCfg = field(default_factory=TrainCfg)
    loss: LossCfg = field(default_factory=LossCfg)

    def hash(self) -> str:
        """Short stable hash of the whole configuration."""
        blob = json.dumps(cfg_to_dict(self), sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

    def summary(self) -> str:
        """Multi-line human-readable dump for the top of a training log."""
        lines = [f"config {self.name} [{self.hash()}]"]
        for section in ("data", "model", "train", "loss"):
            lines.append(f"  [{section}]")
            for key, value in sorted(asdict(getattr(self, section)).items()):
                lines.append(f"    {key:22s} {value}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- conversion

def cfg_to_dict(cfg: Cfg) -> dict:
    """Plain nested dict, JSON/YAML safe."""
    return _plain(asdict(cfg))


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def dict_to_cfg(data: dict) -> Cfg:
    """Rebuild a Cfg from a nested dict, ignoring unknown keys with a clear error."""
    data = dict(data or {})
    sections = {"data": DataCfg, "model": ModelConfig, "train": TrainCfg, "loss": LossCfg}
    kwargs: dict[str, Any] = {"name": data.get("name", "csnet")}
    for key, klass in sections.items():
        kwargs[key] = _build(klass, data.get(key, {}) or {})
    return Cfg(**kwargs)


def _build(klass: Any, values: dict) -> Any:
    valid = {f.name for f in fields(klass)}
    unknown = set(values) - valid
    if unknown:
        raise KeyError(f"unknown {klass.__name__} keys: {sorted(unknown)}; "
                       f"valid keys are {sorted(valid)}")
    coerced = {}
    for f in fields(klass):
        if f.name not in values:
            continue
        coerced[f.name] = _coerce(values[f.name], f.type)
    return klass(**coerced)


def _coerce(value: Any, annotation: Any) -> Any:
    """Turn YAML lists into tuples where the dataclass asks for one."""
    text = str(annotation)
    if isinstance(value, list) and "tuple" in text:
        return tuple(value)
    return value


# --------------------------------------------------------------------------- loading

def load_yaml(path: str) -> dict:
    """Read a YAML file, resolving a single ``_base_`` inheritance level."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    base = data.pop("_base_", None)
    if base:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        data = deep_merge(load_yaml(base_path), data)
    return data


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; ``override`` wins."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def apply_overrides(data: dict, overrides: Sequence[str] | None) -> dict:
    """Apply ``a.b.c=value`` strings; values go through ``literal_eval`` when possible."""
    out = dict(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value: Any = literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise TypeError(f"cannot descend into {key!r}: {part!r} is not a section")
        node[parts[-1]] = value
    return out


def load_cfg(path: str | None = None, overrides: Sequence[str] | None = None) -> Cfg:
    """Load a config from YAML (or defaults) and apply dotted CLI overrides."""
    data = load_yaml(path) if path else {}
    data = apply_overrides(data, overrides)
    cfg = dict_to_cfg(data)
    _validate(cfg)
    return cfg


def _validate(cfg: Cfg) -> None:
    if cfg.model.max_n_src < max(cfg.data.n_list):
        raise ValueError(f"model.max_n_src={cfg.model.max_n_src} is smaller than the largest "
                         f"entry of data.n_list={tuple(cfg.data.n_list)}")
    if cfg.model.n_classes != len(cfg.data.n_list):
        raise ValueError(f"model.n_classes={cfg.model.n_classes} must equal "
                         f"len(data.n_list)={len(cfg.data.n_list)}")
    if cfg.model.max_n_src > MAX_N_SRC:
        raise ValueError(f"model.max_n_src={cfg.model.max_n_src} exceeds MAX_N_SRC={MAX_N_SRC}; "
                         "raise the constant deliberately if you really want that")
    if cfg.data.n_weights is not None and len(cfg.data.n_weights) != len(cfg.data.n_list):
        raise ValueError("data.n_weights must have the same length as data.n_list")


def save_cfg(cfg: Cfg, path: str) -> None:
    """Write a config back out as YAML."""
    import yaml

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg_to_dict(cfg), fh, sort_keys=False, default_flow_style=False)


def seg_len(cfg: Cfg) -> int:
    """Segment length in samples for this config."""
    from .constants import SR

    return int(round(float(cfg.data.seg_seconds) * SR))


def add_config_args(parser: Any) -> Any:
    """Attach the standard ``--config`` / ``--set`` pair to an argparse parser."""
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--set", nargs="*", default=None, metavar="KEY=VALUE",
                        help="dotted overrides, e.g. --set train.lr=1e-3 model.kernel=32")
    return parser


def is_dataclass_instance(obj: Any) -> bool:
    """True for dataclass instances (not classes)."""
    return is_dataclass(obj) and not isinstance(obj, type)
