"""CountSepNet: Conv-TasNet with a max-N mask head, a noise slot, and a count head.

Implemented from scratch (no asteroid) so the repo pins nothing and runs on whatever torch
Kaggle ships this week. The architecture is Luo & Mesgarani (2019) with three changes:

* the mask head emits ``max_n_src`` speaker slots regardless of how many talkers are
  actually present -- surplus slots are pushed to silence by the loss;
* one extra **noise slot**, so background noise has somewhere to go instead of smearing
  across the speech slots (+66 k parameters, +1.3 %);
* a **count head** hanging off the TCN skip-sum, which is both the speaker counter and,
  later, an interpretability probe: it reads the same shared features whose mask geometry
  we measure in ``csnet.interpret``.

Only the mask head depends on N. The TCN is 92-96 % of the FLOPs and is N-independent, so
five outputs cost about 4 % more than two.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import MAX_N_SRC, N_CLASSES, SR


@dataclass
class ModelConfig:
    """Conv-TasNet hyperparameters. Defaults are the paper's Table I configuration."""

    n_filters: int = 512      # N  encoder basis size
    kernel: int = 16          # L  encoder window in samples (2 ms at 8 kHz)
    bottleneck: int = 128     # B  TCN channel width
    hidden: int = 512         # H  depthwise-conv channel width
    skip: int = 128           # Sc skip-connection width
    conv_kernel: int = 3      # P  depthwise kernel
    n_blocks: int = 8         # X  blocks per repeat
    n_repeats: int = 3        # R  repeats
    max_n_src: int = MAX_N_SRC
    predict_noise: bool = True
    n_classes: int = N_CLASSES
    mask_act: str = "relu"    # relu | sigmoid | softmax
    norm: str = "gLN"         # gLN | cLN
    count_hidden: int = 128
    count_dropout: float = 0.1
    causal: bool = False

    def __post_init__(self) -> None:
        if self.kernel % 2 != 0:
            raise ValueError("kernel must be even (stride is kernel // 2)")
        if self.causal:
            self.norm = "cLN"


PRESETS: dict[str, ModelConfig] = {
    "paper": ModelConfig(),
    "small": ModelConfig(n_filters=256, kernel=32, hidden=256, n_blocks=8, n_repeats=2),
    "tiny": ModelConfig(n_filters=128, kernel=32, bottleneck=64, hidden=128, skip=64,
                        n_blocks=4, n_repeats=2, count_hidden=64),
}


# --------------------------------------------------------------------------- norms

class GlobalLayerNorm(nn.Module):
    """gLN: normalise over channels *and* time, per sample. Non-causal."""

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)."""
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class ChannelwiseLayerNorm(nn.Module):
    """cLN: normalise over channels only, per frame. Causal-safe."""

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)."""
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


def make_norm(kind: str, channels: int) -> nn.Module:
    """Build the normalisation layer named by ``kind``."""
    if kind == "gLN":
        return GlobalLayerNorm(channels)
    if kind == "cLN":
        return ChannelwiseLayerNorm(channels)
    if kind == "BN":
        return nn.BatchNorm1d(channels)
    raise ValueError(f"unknown norm: {kind!r}")


# --------------------------------------------------------------------------- tcn

class TemporalBlock(nn.Module):
    """One dilated depthwise-separable residual block."""

    def __init__(self, bottleneck: int, hidden: int, skip: int, kernel: int,
                 dilation: int, norm: str, causal: bool, use_residual: bool = True) -> None:
        super().__init__()
        self.causal = causal
        self.padding = (kernel - 1) * dilation

        self.conv_in = nn.Conv1d(bottleneck, hidden, 1)
        self.prelu_in = nn.PReLU()
        self.norm_in = make_norm(norm, hidden)
        self.depthwise = nn.Conv1d(hidden, hidden, kernel, dilation=dilation,
                                   groups=hidden, padding=0)
        self.prelu_out = nn.PReLU()
        self.norm_out = make_norm(norm, hidden)
        # The last block in the stack has nothing downstream to feed, so its residual
        # branch would be dead weight (~66 k parameters that never receive a gradient).
        self.res_out = nn.Conv1d(hidden, bottleneck, 1) if use_residual else None
        self.skip_out = nn.Conv1d(hidden, skip, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, B_ch, T) -> (residual, skip)."""
        y = self.norm_in(self.prelu_in(self.conv_in(x)))
        if self.causal:
            y = F.pad(y, (self.padding, 0))
        else:
            left = self.padding // 2
            y = F.pad(y, (left, self.padding - left))
        y = self.norm_out(self.prelu_out(self.depthwise(y)))
        residual = x if self.res_out is None else x + self.res_out(y)
        return residual, self.skip_out(y)


class TCN(nn.Module):
    """``n_repeats`` stacks of ``n_blocks`` blocks with exponentially growing dilation."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        n_total = cfg.n_repeats * cfg.n_blocks
        specs = [(r, b) for r in range(cfg.n_repeats) for b in range(cfg.n_blocks)]
        self.blocks = nn.ModuleList([
            TemporalBlock(cfg.bottleneck, cfg.hidden, cfg.skip, cfg.conv_kernel,
                          2 ** b, cfg.norm, cfg.causal,
                          use_residual=(i < n_total - 1))
            for i, (_r, b) in enumerate(specs)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sum of every block's skip output, (B, Sc, T)."""
        total: torch.Tensor | None = None
        for block in self.blocks:
            x, skip = block(x)
            total = skip if total is None else total + skip
        assert total is not None
        return total


# --------------------------------------------------------------------------- heads

class CountHead(nn.Module):
    """Statistics-pooled classifier over the shared TCN features."""

    def __init__(self, skip: int, hidden: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Conv1d(skip, hidden, 1)
        self.act = nn.PReLU()
        self.fc1 = nn.Linear(2 * hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, Sc, T) -> logits (B, n_classes)."""
        h = self.act(self.proj(feat))
        pooled = torch.cat([h.mean(dim=2), h.std(dim=2, unbiased=False)], dim=1)
        return self.fc2(self.drop(F.relu(self.fc1(pooled))))


# --------------------------------------------------------------------------- network

class CountSepNet(nn.Module):
    """Encoder -> TCN -> {mask head, count head} -> decoder."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stride = cfg.kernel // 2

        self.encoder = nn.Conv1d(1, cfg.n_filters, cfg.kernel, stride=self.stride, bias=False)
        self.pre_norm = make_norm(cfg.norm, cfg.n_filters)
        self.bottleneck = nn.Conv1d(cfg.n_filters, cfg.bottleneck, 1)
        self.tcn = TCN(cfg)
        self.mask_prelu = nn.PReLU()
        self.mask_conv = nn.Conv1d(cfg.skip, self.n_slots * cfg.n_filters, 1)
        self.decoder = nn.ConvTranspose1d(cfg.n_filters, 1, cfg.kernel,
                                          stride=self.stride, bias=False)
        self.count_head = CountHead(cfg.skip, cfg.count_hidden, cfg.n_classes,
                                    cfg.count_dropout)

    # -- shape helpers -----------------------------------------------------
    @property
    def n_slots(self) -> int:
        """Speaker slots plus the optional noise slot."""
        return self.cfg.max_n_src + int(self.cfg.predict_noise)

    @property
    def noise_slot(self) -> int | None:
        """Index of the noise slot, or None when it is disabled."""
        return self.cfg.max_n_src if self.cfg.predict_noise else None

    def _pad_len(self, n_samples: int) -> int:
        """Right-padding that makes the encoder/decoder lengths line up exactly."""
        kernel, stride = self.cfg.kernel, self.stride
        if n_samples < kernel:
            return kernel - n_samples
        rem = (n_samples - kernel) % stride
        return 0 if rem == 0 else stride - rem

    def _apply_mask_act(self, masks: torch.Tensor) -> torch.Tensor:
        act = self.cfg.mask_act
        if act == "relu":
            return F.relu(masks)
        if act == "sigmoid":
            return torch.sigmoid(masks)
        if act == "softmax":
            return torch.softmax(masks, dim=1)
        raise ValueError(f"unknown mask_act: {act!r}")

    # -- forward -----------------------------------------------------------
    def forward(self, mix: torch.Tensor, return_internals: bool = False) -> dict:
        """mix: (B, T) or (B, 1, T) -> dict with 'est' (B, n_slots, T) and 'count_logits'."""
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)
        batch, _, n_samples = mix.shape

        pad = self._pad_len(n_samples)
        x = F.pad(mix, (0, pad)) if pad else mix

        enc = F.relu(self.encoder(x))                       # (B, N, F)
        feat = self.tcn(self.bottleneck(self.pre_norm(enc)))  # (B, Sc, F)

        masks = self.mask_conv(self.mask_prelu(feat))
        masks = masks.view(batch, self.n_slots, self.cfg.n_filters, -1)
        masks = self._apply_mask_act(masks)

        masked = enc.unsqueeze(1) * masks                    # (B, S, N, F)
        flat = masked.reshape(batch * self.n_slots, self.cfg.n_filters, -1)
        est = self.decoder(flat).reshape(batch, self.n_slots, -1)
        est = est[..., :n_samples]

        out = {"est": est, "count_logits": self.count_head(feat)}
        if return_internals:
            out["masks"] = masks
            out["enc"] = enc
            out["feat"] = feat
        return out

    # -- introspection -----------------------------------------------------
    def count_params(self, trainable_only: bool = True) -> int:
        """Number of parameters."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(int(p.numel()) for p in params)

    def flops_per_second_of_audio(self, sr: int = SR) -> float:
        """Analytic forward FLOPs for one second of audio (1 MAC = 2 FLOP)."""
        cfg = self.cfg
        frames = sr / self.stride
        macs = cfg.n_filters * cfg.kernel * frames                 # encoder
        macs += cfg.n_filters * cfg.bottleneck * frames            # bottleneck 1x1
        per_block = (cfg.bottleneck * cfg.hidden
                     + cfg.hidden * cfg.conv_kernel                # depthwise
                     + cfg.hidden * cfg.bottleneck                 # residual 1x1
                     + cfg.hidden * cfg.skip)                      # skip 1x1
        macs += per_block * cfg.n_blocks * cfg.n_repeats * frames
        macs += cfg.skip * self.n_slots * cfg.n_filters * frames    # mask head
        macs += self.n_slots * cfg.n_filters * cfg.kernel * frames  # decoder
        macs += cfg.skip * cfg.count_hidden * frames               # count head projection
        return float(macs * 2.0)

    def describe(self) -> str:
        """One-line summary for logs and the report."""
        gflops = self.flops_per_second_of_audio() / 1e9
        return (f"CountSepNet(N={self.cfg.n_filters}, L={self.cfg.kernel}, "
                f"B={self.cfg.bottleneck}, H={self.cfg.hidden}, X={self.cfg.n_blocks}, "
                f"R={self.cfg.n_repeats}, slots={self.n_slots}) "
                f"{self.count_params() / 1e6:.2f} M params, "
                f"{gflops:.1f} GFLOP per second of audio")


def build_model(cfg: ModelConfig | str | dict) -> CountSepNet:
    """Build a model from a ModelConfig, a preset name, or a plain dict."""
    if isinstance(cfg, str):
        if cfg not in PRESETS:
            raise KeyError(f"unknown preset {cfg!r}; have {sorted(PRESETS)}")
        cfg = PRESETS[cfg]
    elif isinstance(cfg, dict):
        cfg = ModelConfig(**cfg)
    return CountSepNet(cfg)
