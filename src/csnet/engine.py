"""Training and evaluation loops: AMP, accumulation, budget-aware early stop.

``train_one_epoch`` never raises on a time-out -- it breaks cleanly and reports
``stopped_early``, so the caller can save a checkpoint and exit 0. That distinction
matters on Kaggle, where a non-zero exit loses the whole Save Version.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .constants import MAX_N_SRC, N_LIST, class_to_n
from .metrics import matched_si_sdri, p_si_snr, si_sdr, summarise_per_n
from .utils import get_logger

LOG = get_logger(__name__)


# --------------------------------------------------------------------------- optimiser

def build_optimizer(model: torch.nn.Module, cfg: Any) -> tuple[Any, Any]:
    """AdamW plus the scheduler named by ``cfg.train.sched``."""
    train = cfg.train
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 or name.endswith(".bias") else decay).append(param)

    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(train.weight_decay)},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(train.lr), betas=(0.9, 0.999), eps=1e-8)

    kind = str(getattr(train, "sched", "cosine")).lower()
    if kind == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    elif kind == "cosine":
        total = max(1, int(train.epochs) * int(train.steps_per_epoch))
        warmup = max(0, int(getattr(train, "warmup_steps", 0)))

        def lr_lambda(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif kind in {"none", "const", "constant"}:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _s: 1.0)
    else:
        raise ValueError(f"unknown scheduler: {kind!r}")
    return optimizer, scheduler


def is_plateau(scheduler: Any) -> bool:
    """True when the scheduler steps once per epoch on a validation metric."""
    return isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def move_batch(batch: dict, device: torch.device) -> dict:
    """Move every tensor in a collated batch to ``device`` (strings pass through)."""
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


# --------------------------------------------------------------------------- training

def train_one_epoch(model: torch.nn.Module, loader: Any, loss_fn: torch.nn.Module,
                    optimizer: Any, scaler: Any, device: torch.device, *,
                    scheduler: Any = None, grad_clip: float = 5.0, log_every: int = 50,
                    budget: Any = None, global_step: int = 0, amp: bool = True,
                    accum: int = 1, on_step: Callable[[int, dict], None] | None = None,
                    budget_check_every: int = 20, max_steps: int | None = None,
                    progress: bool = True) -> dict:
    """One pass over ``loader``. Returns aggregated logs plus ``stopped_early``."""
    model.train()
    use_amp = bool(amp) and device.type == "cuda"
    accum = max(1, int(accum))

    totals: dict[str, float] = {}
    n_batches = 0
    stopped_early = False
    step = int(global_step)

    iterator: Any = loader
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(loader, desc="train", leave=False, unit="batch")
        except ImportError:
            pass

    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(iterator):
        batch = move_batch(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp,
                                dtype=torch.float16):
            out = model(batch["mix"])
        loss, logs = loss_fn(out, batch)

        scaled = loss / accum
        if scaler is not None and scaler.is_enabled():
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        if (i + 1) % accum == 0:
            if grad_clip and grad_clip > 0:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and not is_plateau(scheduler):
                scheduler.step()
            step += 1

        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        n_batches += 1

        if on_step is not None:
            on_step(step, logs)
        if log_every and n_batches % log_every == 0:
            mean = {k: v / n_batches for k, v in totals.items()}
            LOG.info("step %d | loss %.3f | sisdr %.2f dB | count %.3f | acc %.3f | lr %.2e",
                     step, mean["loss"], mean["sisdr"], mean["count"], mean["acc"],
                     optimizer.param_groups[0]["lr"])
        if budget is not None and n_batches % max(1, budget_check_every) == 0 and budget.expired():
            LOG.warning("time budget reached after %d steps -- stopping cleanly", n_batches)
            stopped_early = True
            break
        if max_steps is not None and n_batches >= int(max_steps):
            break

    result = {k: v / max(1, n_batches) for k, v in totals.items()}
    result.update({"steps": n_batches, "global_step": step, "stopped_early": stopped_early,
                   "lr": float(optimizer.param_groups[0]["lr"])})
    return result


# --------------------------------------------------------------------------- evaluation

def batch_records(out: dict, batch: dict, max_n_src: int = MAX_N_SRC) -> list[dict]:
    """Per-utterance metric records. Shared by ``evaluate`` and ``06_evaluate.py``.

    Both call sites must produce identical numbers, so the extraction lives in one place.
    """
    est = out["est"].detach().float().cpu().numpy()
    logits = out["count_logits"].detach().float().cpu().numpy()
    refs = batch["refs"].detach().float().cpu().numpy()
    mix = batch["mix"].detach().float().cpu().numpy()
    n_src = batch["n_src"].detach().cpu().numpy()
    is_noisy = batch["is_noisy"].detach().cpu().numpy()
    snr_db = batch["snr_db"].detach().cpu().numpy() if "snr_db" in batch else np.zeros(len(n_src))
    mix_ids = batch.get("mix_id", [""] * len(n_src))

    probs = _softmax(logits)
    preds = probs.argmax(axis=1)

    records: list[dict] = []
    for b in range(est.shape[0]):
        n_true = int(n_src[b])
        n_pred = int(class_to_n(int(preds[b])))
        sources = refs[b, :n_true]
        record: dict[str, Any] = {
            "mix_id": mix_ids[b] if isinstance(mix_ids, list) else "",
            "n_true": n_true,
            "n_pred": n_pred,
            "confidence": float(probs[b].max()),
            "is_noisy": int(is_noisy[b]),
            "snr_db": float(snr_db[b]),
            "input_si_sdr": float(np.mean([si_sdr(mix[b], s) for s in sources])),
            "p_si_snr": float(p_si_snr(est[b], sources, n_true, n_pred, max_n_src=max_n_src)),
            "si_sdri": None,
        }
        if n_pred == n_true:
            record["si_sdri"] = matched_si_sdri(est[b, :max_n_src], sources,
                                                mix[b], n_true).tolist()
        records.append(record)
    return records


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: Any, loss_fn: torch.nn.Module,
             device: torch.device, *, amp: bool = True, max_batches: int | None = None,
             collect: bool = False, max_n_src: int = MAX_N_SRC,
             n_list: Sequence[int] = N_LIST, full_metrics: bool = True,
             progress: bool = False) -> dict:
    """Validation pass. Returns loss terms, count accuracy, SI-SDRi and P-SI-SNR."""
    model.eval()
    use_amp = bool(amp) and device.type == "cuda"

    totals: dict[str, float] = {}
    n_batches = 0
    records: list[dict] = []

    iterator: Any = loader
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(loader, desc="eval", leave=False, unit="batch")
        except ImportError:
            pass

    for i, batch in enumerate(iterator):
        if max_batches is not None and i >= int(max_batches):
            break
        batch = move_batch(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp,
                                dtype=torch.float16):
            out = model(batch["mix"])
        _, logs = loss_fn(out, batch)
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        n_batches += 1
        if full_metrics:
            records.extend(batch_records(out, batch, max_n_src=max_n_src))

    result = {k: v / max(1, n_batches) for k, v in totals.items()}
    result["batches"] = n_batches

    if records:
        correct = [r for r in records if r["si_sdri"] is not None]
        result["count_acc"] = float(np.mean([r["n_pred"] == r["n_true"] for r in records]))
        result["p_si_snr"] = float(np.mean([r["p_si_snr"] for r in records]))
        result["sisdri_count_correct"] = (
            float(np.mean([np.mean(r["si_sdri"]) for r in correct])) if correct else float("nan"))
        result["input_si_sdr"] = float(np.mean([r["input_si_sdr"] for r in records]))
        result["per_n"] = summarise_per_n(records, n_list)
        result["n_utterances"] = len(records)
        if collect:
            result["y_true"] = [r["n_true"] for r in records]
            result["y_pred"] = [r["n_pred"] for r in records]
            result["records"] = records
    else:
        result.setdefault("count_acc", result.get("acc", float("nan")))
        result.setdefault("p_si_snr", float("nan"))
        result.setdefault("sisdri_count_correct", float("nan"))
        result.setdefault("per_n", {})
    return result


# --------------------------------------------------------------------------- throughput

def measure_throughput(model: torch.nn.Module, loader: Any, loss_fn: torch.nn.Module,
                       optimizer: Any, scaler: Any, device: torch.device, *,
                       steps: int = 20, amp: bool = True, warmup: int = 3) -> dict:
    """Time a handful of real training steps so the epoch budget is measured, not guessed.

    The project record's FLOP table is a planning aid; this replaces it with a number.
    """
    import time

    model.train()
    use_amp = bool(amp) and device.type == "cuda"
    iterator = iter(loader)
    batch_size = 0
    seg_len = 0

    for i in range(int(warmup) + int(steps)):
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        batch_size, seg_len = batch["mix"].shape[0], batch["mix"].shape[-1]
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            out = model(batch["mix"])
        loss, _ = loss_fn(out, batch)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    per_step = elapsed / max(1, int(steps))
    from .constants import SR

    audio_seconds = batch_size * seg_len / SR
    net = getattr(model, "module", model)
    flops = getattr(net, "flops_per_second_of_audio", lambda: 0.0)() * audio_seconds * 3.0
    return {
        "seconds_per_step": per_step,
        "steps_per_hour": 3600.0 / per_step if per_step > 0 else float("inf"),
        "batch_size": batch_size,
        "audio_seconds_per_step": audio_seconds,
        "tflops": (flops / per_step) / 1e12 if per_step > 0 else 0.0,
    }
