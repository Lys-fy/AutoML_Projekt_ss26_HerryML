"""Model and training utilities: op vocabulary, CNN/TCN builders,
augmentation policy, EMA, LR schedule, and the aug-safety probe."""

from __future__ import annotations

import copy as _copy_module
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Downsample planner. Spread max_ds stride-2 ops evenly across n_blocks,
# never letting min spatial dim drop below min_spatial.
# ---------------------------------------------------------------------------


def plan_downsamples(h: int, w: int, n_blocks: int, min_spatial: int = 4) -> List[bool]:
    smaller = max(min(h, w), 1)
    max_ds = int(math.floor(math.log2(smaller / min_spatial))) if smaller >= min_spatial else 0
    if max_ds <= 0 or n_blocks <= 0:
        return [False] * n_blocks
    plan = [False] * n_blocks
    step = n_blocks / max_ds
    for i in range(max_ds):
        idx = min(n_blocks - 1, int(round((i + 1) * step)) - 1)
        plan[idx] = True
    return plan


# ---------------------------------------------------------------------------
# Per-channel normalisation stats.
# ---------------------------------------------------------------------------


def per_channel_stats(train_x: np.ndarray):
    arr = train_x.astype(np.float32)
    mean = arr.mean(axis=(0, 2, 3))
    std = arr.std(axis=(0, 2, 3))
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


# ---------------------------------------------------------------------------
# DropPath (stochastic depth, per-sample) — used inside ResConvBlock.
# Drops the conv branch with probability `drop_prob`, rescaling so the
# expected output magnitude is unchanged in train mode. In eval mode, no-op.
# ---------------------------------------------------------------------------


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, device=x.device, dtype=x.dtype)
    mask.floor_()
    return x.div(keep_prob) * mask


# ---------------------------------------------------------------------------
# Op variants for the search space. Each is a small nn.Module mapping
# (in_ch, out_ch, stride) → tensor with stride applied. Used by both the
# fixed-arch builder and the random op-type search.
# ---------------------------------------------------------------------------


def _conv_bn_act(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _sepconv_bn_act(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=kernel, stride=stride, padding=pad,
                  groups=in_ch, bias=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _pool_op(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Module:
    """Max pool followed by a 1x1 projection to out_ch (handles channel doubling)."""
    pad = kernel // 2
    pool = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=pad)
    if in_ch == out_ch:
        return pool
    return nn.Sequential(pool, nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))


def _identity_op(in_ch: int, out_ch: int, stride: int) -> nn.Module:
    """Identity (when shapes match), 1x1 projection, or stride-2 avgpool.

    The AvgPool uses ceil_mode=True so odd input dims produce the same
    ceil(H/2) spatial output as the strided-conv path.
    """
    if stride == 1 and in_ch == out_ch:
        return nn.Identity()
    layers = []
    if stride != 1:
        layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True))
    if in_ch != out_ch:
        layers.append(nn.Conv2d(in_ch, out_ch, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
    return nn.Sequential(*layers) if len(layers) > 1 else layers[0]


# ---------------------------------------------------------------------------
# Grouped-conv primitives:
#   - GroupConv3 — single 3x3 grouped conv (up to 8 groups).
#   - ResNeXt    — bottleneck-grouped-expand block: 1x1 reduce →
#       3x3 grouped → 1x1 expand.
# ---------------------------------------------------------------------------


def _safe_groups(in_ch: int, out_ch: int, max_groups: int = 8) -> int:
    """Pick the largest `groups` value ≤ max_groups that divides both
    in_ch and out_ch, falling back to 1 (= ungrouped)."""
    g = min(max_groups, in_ch, out_ch)
    while g > 1 and (in_ch % g != 0 or out_ch % g != 0):
        g -= 1
    return max(1, g)


# ---------------------------------------------------------------------------
# Additional ops — Conv7, DilatedConv3, SE-Conv3, SpatialSelfAttn2D
# ---------------------------------------------------------------------------


def _dilated_conv_bn_act(in_ch: int, out_ch: int, stride: int,
                         kernel: int = 3, dilation: int = 2) -> nn.Sequential:
    """Dilated 3×3 conv → BN → ReLU (enlarged effective receptive field)."""
    pad = ((kernel - 1) * dilation) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                  padding=pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _SEConv3(nn.Module):
    """Conv3 + Squeeze-and-Excitation channel attention: a per-channel gain
    learned from a global-average-pooled summary."""

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 reduction: int = 16):
        super().__init__()
        self.conv = _conv_bn_act(in_ch, out_ch, 3, stride)
        squeeze = max(out_ch // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, squeeze, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze, out_ch, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y * self.se(y)


class _SpatialSelfAttn2D(nn.Module):
    """Multi-head self-attention over the flattened 2D spatial grid.

    Inputs with (H·W) > MAX_ATTN_TOKENS are adaptive-pooled to 16×16,
    attention runs on the small grid, and the output is bilinearly
    upsampled back and added as a residual to the projected input.
    Stride > 1 handled by a stride-2 depthwise downsample first.
    """

    MAX_ATTN_TOKENS = 256  # = 16×16 — caps memory for big inputs

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 num_heads: int = 4):
        super().__init__()
        self.down = (
            nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2,
                      groups=in_ch, bias=False)
            if stride > 1 else nn.Identity()
        )
        self.proj_in = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        # num_heads must divide out_ch — find the largest valid head count.
        nh = num_heads
        while out_ch % nh != 0 and nh > 1:
            nh -= 1
        self.attn = nn.MultiheadAttention(out_ch, nh, batch_first=True)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(self.down(x))                     # [B, C, H, W]
        B, C, H, W = x.shape
        if H * W > self.MAX_ATTN_TOKENS:
            # Memory-bounded path: pool → attention → upsample → residual.
            target = max(1, int(self.MAX_ATTN_TOKENS ** 0.5))
            x_small = F.adaptive_avg_pool2d(x, (target, target))
            B_, C_, h, w = x_small.shape
            tokens = x_small.flatten(2).transpose(1, 2)    # [B, target², C]
            attn_out, _ = self.attn(tokens, tokens, tokens)
            out_small = self.norm(tokens + attn_out)
            out_small = out_small.transpose(1, 2).reshape(B_, C_, h, w)
            out_up = F.interpolate(out_small, size=(H, W),
                                   mode="bilinear", align_corners=False)
            return out_up + x                              # residual to original
        # Direct path: small enough for native attention.
        tokens = x.flatten(2).transpose(1, 2)              # [B, H·W, C]
        attn_out, _ = self.attn(tokens, tokens, tokens)
        out = self.norm(tokens + attn_out)
        return out.transpose(1, 2).reshape(B, C, H, W)


def _grpconv_bn_act(in_ch: int, out_ch: int, kernel: int, stride: int,
                    max_groups: int = 8) -> nn.Sequential:
    """3x3 grouped conv → BN → ReLU (intra-group channel mixing only)."""
    pad = kernel // 2
    g = _safe_groups(in_ch, out_ch, max_groups=max_groups)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad,
                  groups=g, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _ResNeXtOp(nn.Module):
    """ResNeXt-style bottleneck-grouped-expand block:
    1x1 reduce → 3x3 grouped (cardinality up to 8) → 1x1 expand.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 max_cardinality: int = 8):
        super().__init__()
        D = out_ch  # bottleneck width
        c = _safe_groups(D, D, max_groups=max_cardinality)
        self.body = nn.Sequential(
            # 1x1 reduce (in_ch → D) — channel mixing happens here
            nn.Conv2d(in_ch, D, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True),
            # 3x3 grouped — the cardinality axis, applies stride
            nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1,
                      groups=c, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True),
            # 1x1 expand (D → out_ch) — re-mix across groups before residual
            nn.Conv2d(D, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _RowBiLSTM(nn.Module):
    """Bidirectional LSTM applied row-wise: treat the width axis as a
    sequence of length W, run a BiLSTM over each of the H rows, and map
    the recurrent output back to a feature map.

    For wide inputs the width is adaptive-pooled to MAX_LSTM_TIME, the
    LSTM runs on the short sequence, and the output is bilinearly
    interpolated back and added as a residual. stride > 1 handled by a
    depthwise stride-2 downsample first. Bidirectional hidden states are
    projected back to out_ch and added as a residual under a LayerNorm.
    """

    MAX_LSTM_TIME = 16  # cap on the recurrent sequence length (W after pool)

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 hidden: int = None):
        super().__init__()
        self.down = (
            nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2,
                      groups=in_ch, bias=False)
            if stride > 1 else nn.Identity()
        )
        self.proj_in = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        h = hidden if hidden is not None else max(8, out_ch // 2)
        self.lstm = nn.LSTM(out_ch, h, batch_first=True, bidirectional=True)
        self.proj_out = nn.Linear(2 * h, out_ch)
        self.norm = nn.LayerNorm(out_ch)

    def _run_rows(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] — run a BiLSTM along W for each of the H rows.
        B, C, H, W = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(B * H, W, C)    # [B·H, W, C]
        out, _ = self.lstm(seq)                             # [B·H, W, 2h]
        out = self.proj_out(out)                            # [B·H, W, C]
        out = self.norm(seq + out)                          # residual + LN
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(self.down(x))                      # [B, C, H, W]
        B, C, H, W = x.shape
        if W > self.MAX_LSTM_TIME:
            x_small = F.adaptive_avg_pool2d(x, (H, self.MAX_LSTM_TIME))
            out_small = self._run_rows(x_small)
            out_up = F.interpolate(out_small, size=(H, W),
                                   mode="bilinear", align_corners=False)
            return out_up + x                               # residual to original
        return self._run_rows(x)


class _CrossAxisBiLSTM(nn.Module):
    """Row-BiLSTM followed by column-BiLSTM: the first pass sweeps along W,
    the second along H (by transposing the spatial axes and reusing the row
    machinery). Stride is applied once, in the row pass."""

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.row = _RowBiLSTM(in_ch, out_ch, stride)
        self.col = _RowBiLSTM(out_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.row(x)                       # [B, C, H, W]
        xt = x.transpose(2, 3)                # [B, C, W, H] — H becomes the row axis
        xt = self.col(xt)
        return xt.transpose(2, 3)             # [B, C, H, W]


class _LinearAttn2D(nn.Module):
    """Bidirectional linear-attention spatial mixer, linear in H·W:
    attention computes Q·(Kᵀ·V), so one (rC × C) global-context matrix is
    formed once and reused for every position.

    Stride > 1 via a depthwise stride-2 downsample, then a 1×1 proj_in to
    out_ch, q/k/v 1×1 convs (q,k lifted to r·out_ch and ReLU'd), the
    linear-attention context, an output projection, and a residual under BN."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, r: int = 2):
        super().__init__()
        self.down = (
            nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2,
                      groups=in_ch, bias=False)
            if stride > 1 else nn.Identity()
        )
        self.proj_in = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        inner = r * out_ch
        self.to_q = nn.Conv2d(out_ch, inner, 1, bias=False)
        self.to_k = nn.Conv2d(out_ch, inner, 1, bias=False)
        self.to_v = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.proj = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(self.down(x))                        # [B, C, H, W]
        B, C, H, W = x.shape
        q = F.relu(self.to_q(x)).flatten(2).transpose(1, 2)   # [B, HW, rC]
        k = F.relu(self.to_k(x)).flatten(2).transpose(1, 2)   # [B, HW, rC]
        v = self.to_v(x).flatten(2).transpose(1, 2)           # [B, HW, C]
        S = k.transpose(1, 2) @ v                              # [B, rC, C] once
        num = q @ S                                           # [B, HW, C]
        denom = q @ k.sum(1, keepdim=True).transpose(1, 2) + 1e-6   # [B, HW, 1]
        ctx = (num / denom).transpose(1, 2).reshape(B, C, H, W)
        return F.relu(self.bn(x + self.proj(ctx)))


class _SparseGated(nn.Module):
    """Sparse-positive gated inverted bottleneck: a ReLU·ReLU gate with a
    learned bias on the up1 branch acting as a negative threshold.
    Inverted-bottleneck ratio r=3, pointwise throughout. stride > 1 via a
    depthwise stride-2 downsample first, then a residual under BN."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, r: int = 3):
        super().__init__()
        self.down = (
            nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2,
                      groups=in_ch, bias=False)
            if stride > 1 else nn.Identity()
        )
        self.proj_in = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        hid = r * out_ch
        self.up1 = nn.Conv2d(out_ch, hid, 1, bias=True)   # bias = learned neg threshold
        self.up2 = nn.Conv2d(out_ch, hid, 1, bias=False)
        self.down_proj = nn.Conv2d(hid, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(self.down(x))
        y = self.down_proj(F.relu(self.up1(x)) * F.relu(self.up2(x)))
        return F.relu(self.bn(x + y))


class _SharedRepeat(nn.Module):
    """Weight-shared depth: apply one residual block k times reusing the
    same weights. Only wrapped where the block is shape-preserving
    (in_ch == out_ch, stride 1) so repeated application is safe."""

    def __init__(self, block: nn.Module, k: int):
        super().__init__()
        self.block = block
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.k):
            x = self.block(x)
        return x


class _RepConv3(nn.Module):
    """RepVGG-style structural-reparameterization block. Trains as a
    multi-branch block -- 3x3 conv+BN, 1x1 conv+BN, and (when
    shape-preserving) an identity BN -- summed then ReLU'd.
    reparameterize() folds the branches exactly into a single 3x3
    conv+bias for inference."""

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn_id = (nn.BatchNorm2d(in_ch)
                      if (in_ch == out_ch and stride == 1) else None)
        self.act = nn.ReLU(inplace=True)
        self.fused: Optional[nn.Conv2d] = None   # set by reparameterize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused is not None:
            return self.act(self.fused(x))
        y = self.bn3(self.conv3(x)) + self.bn1(self.conv1(x))
        if self.bn_id is not None:
            y = y + self.bn_id(x)
        return self.act(y)

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
        std = (bn.running_var + bn.eps).sqrt()
        t = bn.weight / std
        return conv.weight * t.reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * t

    @torch.no_grad()
    def reparameterize(self) -> "_RepConv3":
        """Fold the 3 branches into one 3x3 conv+bias. Idempotent."""
        w, b = self._fuse_conv_bn(self.conv3, self.bn3)
        w1, b1 = self._fuse_conv_bn(self.conv1, self.bn1)
        w = w + F.pad(w1, [1, 1, 1, 1]); b = b + b1          # 1x1 -> centre of 3x3
        if self.bn_id is not None:
            idw = torch.zeros(self.out_ch, self.in_ch, 3, 3, device=w.device)
            for c in range(self.in_ch):
                idw[c, c, 1, 1] = 1.0                          # identity as a 3x3 conv
            std = (self.bn_id.running_var + self.bn_id.eps).sqrt()
            t = self.bn_id.weight / std
            w = w + idw * t.reshape(-1, 1, 1, 1)
            b = b + self.bn_id.bias - self.bn_id.running_mean * t
        fused = nn.Conv2d(self.in_ch, self.out_ch, 3, stride=self.stride,
                          padding=1, bias=True).to(w.device)
        fused.weight.data = w; fused.bias.data = b
        self.fused = fused
        return self


# The vocabulary used by the random op-type search. Each entry is a callable
# (in_ch, out_ch, stride) -> nn.Module.
OP_VOCAB = {
    "Conv3":          lambda i, o, s: _conv_bn_act(i, o, 3, s),
    "Conv5":          lambda i, o, s: _conv_bn_act(i, o, 5, s),
    "Conv7":          lambda i, o, s: _conv_bn_act(i, o, 7, s),
    "SepConv3":       lambda i, o, s: _sepconv_bn_act(i, o, 3, s),
    "SepConv5":       lambda i, o, s: _sepconv_bn_act(i, o, 5, s),
    "GroupConv3":     lambda i, o, s: _grpconv_bn_act(i, o, 3, s),
    "ResNeXt":        lambda i, o, s: _ResNeXtOp(i, o, s),
    "DilatedConv3":   lambda i, o, s: _dilated_conv_bn_act(i, o, s),
    "SEConv3":        lambda i, o, s: _SEConv3(i, o, s),
    "SpatialSelfAttn2D": lambda i, o, s: _SpatialSelfAttn2D(i, o, s),
    "LinearAttn2D":   lambda i, o, s: _LinearAttn2D(i, o, s),
    "SparseGated":    lambda i, o, s: _SparseGated(i, o, s),
    "RepConv3":       lambda i, o, s: _RepConv3(i, o, s),
    "RowBiLSTM":      lambda i, o, s: _RowBiLSTM(i, o, s),
    "CrossAxisBiLSTM": lambda i, o, s: _CrossAxisBiLSTM(i, o, s),
    "MaxPool":        lambda i, o, s: _pool_op(i, o, 3, s),
    "Identity":       lambda i, o, s: _identity_op(i, o, s),
}
OP_NAMES = tuple(OP_VOCAB.keys())


# ---------------------------------------------------------------------------
# ResConvBlock — residual wrapper around any of the OP_VOCAB ops. The
# residual path mirrors the op's spatial/channel transform with a cheap 1x1
# (or AvgPool + 1x1 when stride > 1). DropPath drops the op branch
# per-sample during training.
# ---------------------------------------------------------------------------


class ResConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, op_name: str = "Conv3",
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.op_name = op_name
        self.op = OP_VOCAB[op_name](in_ch, out_ch, stride)
        # Shortcut: identity when shapes match, else a 1x1 projection.
        # When stride > 1 the 1x1 conv is strided — same ceil(H/2) spatial
        # output as the op branch.
        if stride == 1 and in_ch == out_ch:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        branch = self.op(x)
        branch = drop_path(branch, self.drop_path_rate, self.training)
        return branch + residual


# ---------------------------------------------------------------------------
# build_cnn: stem → N res-blocks → GAP → linear. Each block's op_name comes
# from `ops` (length n_blocks). Downsample positions and channel widths are
# determined by plan_downsamples + width-doubling on each downsample.
# ---------------------------------------------------------------------------


def build_cnn(in_channels: int, num_classes: int, input_h: int, input_w: int,
              ops: Sequence[str], base_width: int = 48, drop_path_rate: float = 0.1,
              stem: str = "conv3", shared_repeat: Optional[int] = None) -> nn.Sequential:
    n_blocks = len(ops)
    plan = plan_downsamples(input_h, input_w, n_blocks)
    # Weight-shared depth: apply each shape-preserving block ×k with the
    # same weights (env V11_SHARED_REPEAT overrides; default 1 = off).
    if shared_repeat is None:
        shared_repeat = int(os.environ.get("V11_SHARED_REPEAT", "1"))

    # Stem variants:
    #   "conv3"        — plain 3x3 conv (default).
    #   "channel_embed"— 1x1 conv first (per-pixel channel mixing), then a
    #                    3x3 spatial conv.
    if stem == "channel_embed":
        stem_layers: List[nn.Module] = [
            _conv_bn_act(in_channels, base_width, 1, 1),   # 1x1 channel embed
            _conv_bn_act(base_width, base_width, 3, 1),    # then spatial
        ]
    else:
        stem_layers = [_conv_bn_act(in_channels, base_width, 3, 1)]
    layers: List[nn.Module] = list(stem_layers)
    ch = base_width
    for i, downsample in enumerate(plan):
        op_name = ops[i]
        next_ch = ch * 2 if downsample else ch
        stride = 2 if downsample else 1
        block: nn.Module = ResConvBlock(ch, next_ch, stride=stride, op_name=op_name,
                                        drop_path_rate=drop_path_rate)
        if shared_repeat > 1 and stride == 1 and ch == next_ch:
            block = _SharedRepeat(block, shared_repeat)
        layers.append(block)
        ch = next_ch

    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(ch, num_classes))
    return nn.Sequential(*layers)


def build_default_cnn(in_channels: int, num_classes: int, input_h: int, input_w: int,
                      base_width: int = 48, n_blocks: int = 6,
                      drop_path_rate: float = 0.1) -> nn.Sequential:
    """All-Conv3 default architecture — used as the search bootstrap and as
    the final retrain fallback if search fails."""
    return build_cnn(in_channels, num_classes, input_h, input_w,
                     ops=["Conv3"] * n_blocks,
                     base_width=base_width, drop_path_rate=drop_path_rate)


# ---------------------------------------------------------------------------
# Temporal Convolutional Networks — stacks of dilated convs + residuals
# whose receptive field grows exponentially with depth. Two variants:
#
#   tcn1d — reshape the input to a 1-D signal (C channels, length H*W) and
#           run 1-D dilated convs.
#   tcn2d — 2-D dilated-conv net that keeps spatial structure.
#
# Both: stem → N residual dilated-conv blocks (two dilated convs + residual,
# non-causal 'same' padding) → global-avg-pool → linear. Dilation is capped
# to the relevant extent so it never degenerates into all-padding on small
# inputs.
# ---------------------------------------------------------------------------


class _TCNBlock1D(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int):
        super().__init__()
        pad = (kernel - 1) * dilation // 2  # 'same' for odd kernel
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch), nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.net(x) + x)


class TCN1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, seq_len: int,
                 base_width: int = 48, n_blocks: int = 8, kernel: int = 7):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, base_width, 1, bias=False),
                                  nn.BatchNorm1d(base_width), nn.ReLU(inplace=True))
        max_dil = max(1, seq_len // 4)  # don't let RF exceed the sequence
        self.blocks = nn.Sequential(*[
            _TCNBlock1D(base_width, kernel, dilation=min(2 ** i, max_dil))
            for i in range(n_blocks)])
        self.head = nn.Linear(base_width, num_classes)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.contiguous().reshape(b, c, h * w)   # C×H×W → (C, L=H*W)
        x = self.blocks(self.stem(x))
        return self.head(x.mean(dim=2))           # global avg pool over L


class _TCNBlock2D(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.net(x) + x)


def _wn_conv2d(ch: int, kernel: int, pad: int, dilation: int) -> nn.Module:
    """Weight-normalized conv, canonical-TCN style (Bai et al. 2018)."""
    conv = nn.Conv2d(ch, ch, kernel, padding=pad, dilation=dilation)
    try:
        from torch.nn.utils.parametrizations import weight_norm
        return weight_norm(conv)
    except Exception:
        return nn.utils.weight_norm(conv)


class _TCNBlock2DCanonical(nn.Module):
    """Canonical TCN residual block adapted to 2-D: weight-normalized
    convs + ReLU + dropout inside the block, no BatchNorm."""

    def __init__(self, ch: int, kernel: int, dilation: int, p_drop: float = 0.1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            _wn_conv2d(ch, kernel, pad, dilation),
            nn.ReLU(inplace=True), nn.Dropout2d(p_drop),
            _wn_conv2d(ch, kernel, pad, dilation),
            nn.ReLU(inplace=True), nn.Dropout2d(p_drop),
        )

    def forward(self, x):
        return self.net(x) + x


class _TCNBlock2DGELU(nn.Module):
    """Same as _TCNBlock2D but with GELU instead of ReLU."""

    def __init__(self, ch: int, kernel: int, dilation: int):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.net(x) + x)


_TCN_BLOCKS = {"bn": _TCNBlock2D, "canonical": _TCNBlock2DCanonical,
               "gelu": _TCNBlock2DGELU}


class TCN2D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, input_h: int, input_w: int,
                 base_width: int = 48, n_blocks: int = 6, kernel: int = 3,
                 stem: str = "conv3", block: str = "bn"):
        super().__init__()
        # Optional channel-embed stem: 1x1 channel mixing, then 3x3 spatial.
        if stem == "channel_embed":
            self.stem = nn.Sequential(
                _conv_bn_act(in_channels, base_width, 1, 1),
                _conv_bn_act(base_width, base_width, 3, 1),
            )
        else:
            self.stem = _conv_bn_act(in_channels, base_width, 3, 1)
        max_dil = max(1, min(input_h, input_w) // 2)  # cap to spatial extent
        blk = _TCN_BLOCKS[block]
        self.blocks = nn.Sequential(*[
            blk(base_width, kernel, dilation=min(2 ** i, max_dil))
            for i in range(n_blocks)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(base_width, num_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class TokenSeqNet(nn.Module):
    """Token-sequence model: treats an embedding-shaped board [1, h, w] as
    h tokens of dim w and runs a transformer encoder with a mean-pool head.

    Carries a recipe tag (class attribute below) that switches the trainer
    to AdamW. Token dropout is built into forward()."""

    _v12_recipe = "adamw"

    def __init__(self, in_channels: int, num_classes: int, input_h: int,
                 input_w: int, dim: int = 256, depth: int = 4, heads: int = 8,
                 token_drop: float = 0.1):
        super().__init__()
        assert in_channels == 1, "token family is gated to in_ch == 1 boards"
        self.n_tok = input_h
        self.token_drop = token_drop
        self.inorm = nn.LayerNorm(input_w)   # per-token scale, data-independent
        self.proj = nn.Linear(input_w, dim)
        self.pos = nn.Parameter(torch.zeros(1, input_h, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(dim),
                                  nn.Linear(dim, num_classes))

    def forward(self, x):
        if x.dim() == 4:                     # [B, 1, h, w] -> [B, h, w]
            x = x.squeeze(1)
        x = self.inorm(x)
        if self.training and self.token_drop > 0:
            drop = (torch.rand(x.size(0), self.n_tok, 1, device=x.device)
                    < self.token_drop / self.n_tok)
            x = x.masked_fill(drop, 0.0)
        h = self.enc(self.proj(x) + self.pos)
        return self.head(h.mean(dim=1))


# ---------------------------------------------------------------------------
# Augmentation policy — decided from metadata alone (no codename lookups).
# Enables only cutout and small random crop, and only on image-like inputs
# (1 or 3 channels); flip and mixup are never enabled.
# ---------------------------------------------------------------------------


@dataclass
class AugPolicy:
    flip: bool = False
    crop_pad: int = 0           # 0 disables; otherwise pads by N then random-crops back
    cutout_size: int = 0        # 0 disables; otherwise length of cutout square
    mixup_alpha: float = 0.0    # 0 disables Mixup; otherwise Beta(alpha, alpha) λ

    def any_input_aug(self) -> bool:
        return self.flip or self.crop_pad > 0 or self.cutout_size > 0

    def describe(self) -> str:
        bits = []
        if self.flip:
            bits.append("flip")
        if self.crop_pad > 0:
            bits.append(f"crop_pad={self.crop_pad}")
        if self.cutout_size > 0:
            bits.append(f"cutout={self.cutout_size}")
        if self.mixup_alpha > 0:
            bits.append(f"mixup_a={self.mixup_alpha}")
        return ", ".join(bits) if bits else "none"


def aug_policy_for(metadata: dict) -> AugPolicy:
    """Pick the augmentation policy from metadata alone (no codename lookups).

    Setting TIER1_AUG_OFF=1 in the environment forces the empty policy
    regardless of metadata.
    """
    if os.environ.get("TIER1_AUG_OFF") == "1":
        return AugPolicy()

    shape = metadata.get("input_shape", [])
    if len(shape) == 4:
        C = int(shape[1])
        H = int(shape[-2])
        W = int(shape[-1])
    elif len(shape) == 3:
        C = int(shape[0])
        H = int(shape[-2])
        W = int(shape[-1])
    else:
        return AugPolicy()

    # Aug only when the input is image-like (1-ch grayscale or 3-ch RGB).
    if C not in (1, 3):
        return AugPolicy()

    p = AugPolicy()

    # Cutout — zeros a small patch. Length scales mildly with input size,
    # capped at 16 for big inputs.
    if H >= 16 and W >= 16:
        p.cutout_size = min(16, max(4, H // 8))

    # Small random crop — pad by 2 with reflection, then random-crop back
    # (at most a 2-pixel shift). Only on square inputs ≥ 24 px.
    if H >= 24 and W >= 24 and H == W:
        p.crop_pad = 2

    # Flip and Mixup remain disabled.
    return p


# ---------------------------------------------------------------------------
# AugmentedTensorDataset — applies the configured per-sample transforms in
# __getitem__. Inputs must already be normalised (Float32 tensors).
# ---------------------------------------------------------------------------


class AugmentedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, policy: AugPolicy):
        self.x = x
        self.y = y
        self.policy = policy

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        p = self.policy

        if p.flip and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])

        if p.crop_pad > 0:
            x = self._random_crop(x, pad=p.crop_pad)

        if p.cutout_size > 0:
            x = self._cutout(x, size=p.cutout_size)

        return x, self.y[idx]

    @staticmethod
    def _random_crop(x: torch.Tensor, pad: int) -> torch.Tensor:
        C, H, W = x.shape
        x_pad = F.pad(x, [pad, pad, pad, pad], mode="reflect")
        top = int(torch.randint(0, 2 * pad + 1, (1,)).item())
        left = int(torch.randint(0, 2 * pad + 1, (1,)).item())
        return x_pad[..., top : top + H, left : left + W]

    @staticmethod
    def _cutout(x: torch.Tensor, size: int) -> torch.Tensor:
        C, H, W = x.shape
        cy = int(torch.randint(0, H, (1,)).item())
        cx = int(torch.randint(0, W, (1,)).item())
        y1 = max(0, cy - size // 2)
        y2 = min(H, cy + size // 2)
        x1 = max(0, cx - size // 2)
        x2 = min(W, cx + size // 2)
        x = x.clone()
        x[..., y1:y2, x1:x2] = 0.0
        return x


# ---------------------------------------------------------------------------
# EMA — exponential moving average of model weights, used for prediction.
# ---------------------------------------------------------------------------


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy as _copy
        self.decay = decay
        self.ema = _copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        # Decay parameters; copy buffers (BN running stats) directly so EMA
        # tracks the live model's normalisation statistics.
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for ema_b, b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return self.ema.state_dict()


# ---------------------------------------------------------------------------
# LR schedule helper — manual cosine + linear warmup. Used by trainer.
# ---------------------------------------------------------------------------


def lr_for_epoch(epoch: int, total_epochs: int, warmup_epochs: int = 5,
                 base_lr: float = 0.025, min_lr: float = 0.0) -> float:
    if total_epochs <= 1:
        return base_lr
    if epoch < warmup_epochs:
        # Linear warmup from min_lr to base_lr.
        return min_lr + (base_lr - min_lr) * (epoch + 1) / warmup_epochs
    # Cosine anneal from base_lr to min_lr over (total - warmup) epochs.
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    progress = min(1.0, progress)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
