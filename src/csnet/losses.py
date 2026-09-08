"""Rectangular permutation-invariant loss for a variable speaker count.

The output has a fixed 5 speaker slots but the truth has ``n_src`` of them, so each
reference must be matched to a *distinct* slot with ``5 - n_src`` slots left over. The
reference definition is a rectangular linear assignment::

    C = -pairwise_si_sdr(est, refs)          # (n_src, 5)
    rows, cols = linear_sum_assignment(C)
    loss = -matched.mean() + relu(power_db(leftover) - SILENCE_DB).mean() \
         + w_count * cross_entropy(count_logits, n_src)

We do **not** run scipy in the training loop. The number of injective maps from ``n``
references into 5 slots is ``P(5, n) <= 120``, so enumerating all of them is *exactly* the
optimal assignment and it vectorises onto the GPU with no host synchronisation:

===  ======
 n   P(5,n)
===  ======
 1        5
 2       20
 3       60
 4      120
 5      120
===  ======

Permutation search was never the bottleneck anyway -- measured at batch 24 / 3 s / 8 kHz it
costs tens of microseconds at every N, because it runs on the precomputed pairwise matrix
and never touches waveforms. What grows is the matrix, as N squared.

Four invariants are asserted in ``tests/test_losses.py`` and by ``python -m csnet.losses``:

1. permuting correct estimates among slots does not change the loss;
2. *shifting* them to a different subset of slots does not change it either;
3. filling the leftover slots with the mixture raises it by more than 20;
4. perfect estimates sit at the clamp ceiling and the count term goes to zero.
"""

from __future__ import annotations

from itertools import permutations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import EPS, MAX_N_SRC, N_CLASSES, SILENCE_DB


def pairwise_si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """SI-SDR in dB of every estimate against every reference.

    ``est`` (B, S, T), ``ref`` (B, R, T) -> (B, R, S). Computed in closed form so no
    (B, R, S, T) tensor is ever materialised.
    """
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)

    dot = torch.einsum("brt,bst->brs", ref, est)              # <est_s, ref_r>
    ref_energy = ref.pow(2).sum(dim=-1).unsqueeze(-1)         # (B, R, 1)
    est_energy = est.pow(2).sum(dim=-1).unsqueeze(1)          # (B, 1, S)

    signal = dot.pow(2) / (ref_energy + eps)                  # ||proj||^2
    noise = (est_energy - signal).clamp_min(eps)              # ||est - proj||^2
    return 10.0 * torch.log10(signal + eps) - 10.0 * torch.log10(noise)


def si_sdr_pair(est: torch.Tensor, ref: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """SI-SDR of matched (B, T) pairs."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    proj = alpha * ref
    return 10.0 * torch.log10(proj.pow(2).sum(-1) + eps) \
        - 10.0 * torch.log10((est - proj).pow(2).sum(-1) + eps)


def soft_clamp(sisdr: torch.Tensor, tau: float) -> torch.Tensor:
    """Saturate SI-SDR at ``tau`` dB so one easy example cannot dominate the batch."""
    return -10.0 * torch.log10(torch.pow(10.0, -sisdr / 10.0) + 10.0 ** (-tau / 10.0))


class RectangularPITLoss(nn.Module):
    """Separation + silence + noise-slot + counting loss for a variable speaker count."""

    def __init__(self, max_n_src: int = MAX_N_SRC, predict_noise: bool = True,
                 n_classes: int = N_CLASSES, w_sep: float = 1.0, w_sil: float = 1.0,
                 w_count: float = 0.5, w_noise: float = 0.2,
                 silence_db: float = SILENCE_DB, label_smoothing: float = 0.05,
                 clamp_si_sdr: float | None = 30.0, eps: float = EPS) -> None:
        super().__init__()
        self.max_n_src = int(max_n_src)
        self.predict_noise = bool(predict_noise)
        self.n_classes = int(n_classes)
        self.w_sep = float(w_sep)
        self.w_sil = float(w_sil)
        self.w_count = float(w_count)
        self.w_noise = float(w_noise)
        self.silence_db = float(silence_db)
        self.label_smoothing = float(label_smoothing)
        self.clamp_si_sdr = clamp_si_sdr
        self.eps = float(eps)

        for n in range(1, self.max_n_src + 1):
            arrangements = list(permutations(range(self.max_n_src), n))
            arr = torch.tensor(arrangements, dtype=torch.long)             # (P, n)
            comp = torch.ones(len(arrangements), self.max_n_src, dtype=torch.bool)
            comp.scatter_(1, arr, False)                                   # leftover slots
            self.register_buffer(f"arr_{n}", arr, persistent=False)
            self.register_buffer(f"comp_{n}", comp, persistent=False)

    # -- helpers -----------------------------------------------------------
    def _assign(self, pw_loss: torch.Tensor, pw_raw: torch.Tensor, n: int
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Best arrangement of ``n`` references into the speaker slots.

        Returns ``(best_score, best_raw, leftover_mask)`` for every item in the batch,
        computed as if every item had exactly ``n`` sources; the caller masks it.
        """
        batch = pw_loss.shape[0]
        arr: torch.Tensor = self.get_buffer(f"arr_{n}")
        comp: torch.Tensor = self.get_buffer(f"comp_{n}")
        n_arr = arr.shape[0]

        index = arr.view(1, n_arr, n, 1).expand(batch, n_arr, n, 1)
        sub_loss = pw_loss[:, :n, :].unsqueeze(1).expand(batch, n_arr, n, self.max_n_src)
        scores = sub_loss.gather(3, index).squeeze(-1).mean(dim=2)          # (B, P)
        best_score, best_idx = scores.max(dim=1)

        sub_raw = pw_raw[:, :n, :].unsqueeze(1).expand(batch, n_arr, n, self.max_n_src)
        raw = sub_raw.gather(3, index).squeeze(-1).mean(dim=2)              # (B, P)
        best_raw = raw.gather(1, best_idx.unsqueeze(1)).squeeze(1)

        return best_score, best_raw, comp[best_idx]

    # -- forward -----------------------------------------------------------
    def forward(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        """Return ``(loss, logs)``; ``logs`` values are plain floats, for logging only."""
        device_type = out["est"].device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            return self._forward(out, batch)

    def _forward(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        est = out["est"].float()
        logits = out["count_logits"].float()
        refs = batch["refs"].float()
        mix = batch["mix"].float()
        n_src = batch["n_src"].to(est.device)
        cls = batch["cls"].to(est.device)

        batch_size = est.shape[0]
        spk_est = est[:, : self.max_n_src]

        pw_raw = pairwise_si_sdr(spk_est, refs, self.eps)
        pw_loss = pw_raw if self.clamp_si_sdr is None else soft_clamp(pw_raw, self.clamp_si_sdr)

        # Power of every speaker slot relative to the mixture, in dB.
        mix_power = mix.pow(2).mean(dim=-1, keepdim=True)
        slot_db = 10.0 * torch.log10(
            spk_est.pow(2).mean(dim=-1) / (mix_power + self.eps) + self.eps)
        over_floor = F.relu(slot_db - self.silence_db)               # (B, max_n_src)

        # Evaluate every possible n and select with a mask -- no host sync, no indexing.
        sep_sum = torch.zeros((), device=est.device)
        raw_sum = torch.zeros((), device=est.device)
        sil_sum = torch.zeros((), device=est.device)
        sil_count = torch.zeros((), device=est.device)
        for n in range(1, self.max_n_src + 1):
            mask = (n_src == n).float()
            if float(mask.sum().item()) == 0.0:
                continue
            best, best_raw, leftover = self._assign(pw_loss, pw_raw, n)
            sep_sum = sep_sum + (best * mask).sum()
            raw_sum = raw_sum + (best_raw * mask).sum()
            leftover_f = leftover.float() * mask.unsqueeze(1)
            sil_sum = sil_sum + (over_floor * leftover_f).sum()
            sil_count = sil_count + leftover_f.sum()

        denom = torch.tensor(float(batch_size), device=est.device)
        sep = -(sep_sum / denom)
        matched_sisdr = raw_sum / denom
        sil = sil_sum / sil_count.clamp_min(1.0)

        noise_term = torch.zeros((), device=est.device)
        if self.predict_noise and est.shape[1] > self.max_n_src:
            noise_est = est[:, self.max_n_src]
            noise_ref = batch["noise"].float()
            is_noisy = batch["is_noisy"].to(est.device).float()
            noisy_sisdr = si_sdr_pair(noise_est, noise_ref, self.eps)
            if self.clamp_si_sdr is not None:
                noisy_sisdr = soft_clamp(noisy_sisdr, self.clamp_si_sdr)
            noise_db = 10.0 * torch.log10(
                noise_est.pow(2).mean(dim=-1) / (mix_power.squeeze(-1) + self.eps) + self.eps)
            quiet_penalty = F.relu(noise_db - self.silence_db)
            noise_term = (is_noisy * (-noisy_sisdr) + (1.0 - is_noisy) * quiet_penalty).mean()

        count = F.cross_entropy(logits, cls, label_smoothing=self.label_smoothing)
        acc = (logits.argmax(dim=-1) == cls).float().mean()

        total = (self.w_sep * sep + self.w_sil * sil
                 + self.w_count * count + self.w_noise * noise_term)

        logs = {
            "loss": float(total.detach()),
            "sep": float(sep.detach()),
            "sil": float(sil.detach()),
            "count": float(count.detach()),
            "noise": float(noise_term.detach()),
            "sisdr": float(matched_sisdr.detach()),
            "acc": float(acc.detach()),
        }
        return total, logs


# --------------------------------------------------------------------------- self-test

def _self_test() -> int:
    """Run the four invariants the contract requires. Returns a POSIX exit code."""
    torch.manual_seed(0)
    max_n, slots, T, B = 5, 6, 4000, 4
    # label_smoothing is off here: it puts a floor of ~0.8 on the cross-entropy for 5
    # classes, so a *perfect* prediction would look like a failure in invariant 4.
    loss_fn = RectangularPITLoss(max_n_src=max_n, predict_noise=True, clamp_si_sdr=30.0,
                                 w_count=0.5, w_noise=0.2, label_smoothing=0.0)

    n_src = torch.tensor([1, 2, 3, 5])
    refs = torch.zeros(B, max_n, T)
    for b in range(B):
        refs[b, : int(n_src[b])] = torch.randn(int(n_src[b]), T)
    mix = refs.sum(1)
    noise = torch.zeros(B, T)
    batch = {"refs": refs, "mix": mix, "noise": noise, "n_src": n_src,
             "cls": n_src - 1, "is_noisy": torch.zeros(B, dtype=torch.long)}

    def build(order: list[list[int]]) -> dict:
        est = torch.zeros(B, slots, T)
        for b in range(B):
            for i, slot in enumerate(order[b]):
                est[b, slot] = refs[b, i]
        logits = torch.zeros(B, 5)
        logits[torch.arange(B), n_src - 1] = 20.0
        return {"est": est, "count_logits": logits}

    identity = [list(range(int(n))) for n in n_src]
    base, logs = loss_fn(build(identity), batch)
    ok = True

    permuted = [list(reversed(o)) for o in identity]
    l_perm, _ = loss_fn(build(permuted), batch)
    delta_perm = abs(float(l_perm - base))
    ok &= delta_perm < 1e-3
    print(f"1. permutation invariance      delta = {delta_perm:.2e}   "
          f"{'PASS' if delta_perm < 1e-3 else 'FAIL'}")

    shifted = [[(s + (max_n - int(n))) for s in o] for o, n in zip(identity, n_src)]
    l_shift, _ = loss_fn(build(shifted), batch)
    delta_shift = abs(float(l_shift - base))
    ok &= delta_shift < 1e-3
    print(f"2. slot-shift invariance       delta = {delta_shift:.2e}   "
          f"{'PASS' if delta_shift < 1e-3 else 'FAIL'}")

    leaky = build(identity)
    for b in range(B):
        for slot in range(int(n_src[b]), max_n):
            leaky["est"][b, slot] = mix[b]
    l_leak, _ = loss_fn(leaky, batch)
    rise = float(l_leak - base)
    ok &= rise > 20.0
    print(f"3. leaking mixture into spares rise  = {rise:.2f} dB {'PASS' if rise > 20 else 'FAIL'}")

    near_ceiling = abs(logs["sep"] + 30.0) < 0.2 and logs["count"] < 0.05
    ok &= near_ceiling
    print(f"4. perfect estimates           sep = {logs['sep']:.3f} (ceiling -30), "
          f"count_ce = {logs['count']:.4f}, acc = {logs['acc']:.2f}   "
          f"{'PASS' if near_ceiling else 'FAIL'}")

    print("\nall invariants passed" if ok else "\nINVARIANTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
