# =============================================================================
# STEP-1_1 v0.6 -- models.conditioner
# Purpose: shared conditioner c_eta(y) + FiLM head used by every expert.
# CONVENTION: NaN/shape errors -> logger.error + raise. No silent fallback.
# Changelog (v0.5 -> v0.6, SEQREF-I2 paired channel-contract rebuild):
#   * in_channels is now a REQUIRED kwarg (no default): the MRI conditioning
#     stack is the locked 3-channel [|x0|, Re(A^H r), Im(A^H r)] (EXEC 3.8),
#     so the silent single-channel Conv2d(1, ...) assumption is removed.
#     Input validation checks (B, in_channels, H, W).
#   * y-residual bypass unchanged and still default-OFF (EXEC 3.9 DISABLED);
#     when enabled, y_input_size = in_channels * H * W of the flattened input.
# Changelog (v0.4 -> v0.5): optional y-residual bypass (Glow rescue),
#   default OFF, byte-identical to v0.4 when disabled.
# Changelog (v0.3 -> v0.4): FiLMHead last-Linear init depends on output_form.
# Changelog (v0.2 -> v0.3): FiLMHead output_form kwarg.
# Changelog (v0.1 -> v0.2): Conditioner use_v2 kwarg.
# Update summary:
#   v0.6 lands the conditioner half of the paired I2 channel-contract
#   rebuild: input channel count is explicit, required, and validated; the
#   3-channel MRI value comes from the cell, never from a default.
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.6"
__abbr__ = "STEP-1_1"

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conditioner(nn.Module):
    # y (B, in_channels, h, w) -> h (B, h_dim). Same architecture for any
    # spatial size via adaptive pool. in_channels REQUIRED (v0.6).
    # use_v2 (default False) preserves v0.1 head:  Linear(w*16, h_dim).
    # use_v2=True activates v2 head: Linear(w*16, h_dim) -> GELU -> Linear.
    # v0.5 bypass (default OFF, EXEC 3.9 DISABLED): when
    # y_residual_alpha_init > 0: h = cnn_head(y) + alpha * Linear(y.flatten(1))
    def __init__(self, *, in_channels: int, width: int = 64, h_dim: int = 128,
                 use_v2: bool = False,
                 y_residual_alpha_init: float = 0.0,
                 y_input_size: int | None = None):
        super().__init__()
        if not isinstance(in_channels, int) or in_channels < 1:
            logger.error("[Conditioner] in_channels must be a positive int "
                         "(REQUIRED, no default), got %r", in_channels)
            raise ValueError(
                f"in_channels must be a positive int, got {in_channels!r}")
        if width not in (64, 128):
            logger.error("[Conditioner] width must be 64 or 128, got %s", width)
            raise ValueError(f"width must be 64 or 128, got {width}")
        if h_dim < 1:
            logger.error("[Conditioner] h_dim must be positive, got %s", h_dim)
            raise ValueError(f"h_dim must be positive, got {h_dim}")
        if y_residual_alpha_init < 0.0:
            logger.error("[Conditioner] y_residual_alpha_init must be >=0, "
                         "got %s", y_residual_alpha_init)
            raise ValueError(
                f"y_residual_alpha_init must be >=0, got {y_residual_alpha_init}")
        if y_residual_alpha_init > 0.0 and (y_input_size is None
                                            or y_input_size < 1):
            logger.error("[Conditioner] y_residual_alpha_init>0 requires "
                         "y_input_size (positive int), got %s", y_input_size)
            raise ValueError(
                "y_residual_alpha_init>0 requires positive y_input_size")
        self.in_channels = in_channels
        self.width = width
        self.h_dim = h_dim
        self.use_v2 = bool(use_v2)
        self.y_residual_enabled = (y_residual_alpha_init > 0.0)
        w = width
        if not self.use_v2:
            # v0.1 head -- byte-identical to legacy apart from in_channels
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, w // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(w // 2, w, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(w, w, 3, padding=1),      nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4),
                nn.Conv2d(w, w, 3, padding=1),      nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(w * 4 * 4, h_dim),
            )
        else:
            # v0.2 head -- Linear -> GELU -> Linear
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, w // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(w // 2, w, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(w, w, 3, padding=1),      nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4),
                nn.Conv2d(w, w, 3, padding=1),      nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(w * 4 * 4, h_dim),
                nn.GELU(),
                nn.Linear(h_dim, h_dim),
            )
        # v0.5: y-residual bypass (default OFF).
        if self.y_residual_enabled:
            self.y_residual_proj = nn.Linear(y_input_size, h_dim)
            self.y_residual_alpha = nn.Parameter(
                torch.tensor(float(y_residual_alpha_init)))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 4 or y.size(1) != self.in_channels:
            logger.error("[Conditioner] expected (B,%d,H,W), got %s",
                         self.in_channels, tuple(y.shape))
            raise ValueError(
                f"expected (B,{self.in_channels},H,W), got {tuple(y.shape)}")
        h = self.net(y)
        if self.y_residual_enabled:
            y_flat = y.flatten(1)            # (B, in_channels*H*W)
            h = h + self.y_residual_alpha * self.y_residual_proj(y_flat)
        if not torch.isfinite(h).all():
            logger.error("[Conditioner] non-finite h (any NaN=%s, Inf=%s)",
                         bool(torch.isnan(h).any()), bool(torch.isinf(h).any()))
            raise ValueError("non-finite h in Conditioner")
        return h


class FiLMHead(nn.Module):
    # h (B, h_dim) -> (gamma, beta) each (B, feat_width).
    # output_form='affine' (default): gamma = 1 + tanh(raw), identity at init.
    # output_form='residual' (Glow): raw gamma; caller does
    #   z' = z * (1 + gain * gamma_raw) + gain * beta.
    def __init__(self, h_dim: int, feat_width: int, hidden: int = 64,
                 *, depth: int = 1, use_gelu: bool = False,
                 output_form: str = "affine"):
        super().__init__()
        if depth < 1:
            logger.error("[FiLMHead] depth must be >=1, got %s", depth)
            raise ValueError(f"depth must be >=1, got {depth}")
        if output_form not in ("affine", "residual"):
            logger.error("[FiLMHead] output_form must be 'affine' or "
                         "'residual', got %r", output_form)
            raise ValueError(
                f"output_form must be 'affine' or 'residual', got {output_form!r}")
        self.output_form = output_form
        layers: list[nn.Module] = []
        in_dim = h_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            if use_gelu:
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU(inplace=True))
            in_dim = hidden
        last = nn.Linear(in_dim, 2 * feat_width)
        # v0.4: init depends on output_form ('affine' zero-init identity;
        # 'residual' small-normal so the chain has first-order gradient).
        if output_form == "affine":
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        else:  # "residual"
            nn.init.normal_(last.weight, mean=0.0, std=0.01)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.mlp = nn.Sequential(*layers)
        self.feat_width = feat_width

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(h)
        gamma_raw, beta = out.chunk(2, dim=-1)
        if self.output_form == "affine":
            gamma = 1.0 + torch.tanh(gamma_raw)      # range (0,2), centred 1
            return gamma, beta
        return gamma_raw, beta


class ConcatInjector(nn.Module):
    # Alternative to FiLM for the ablation: project h and residual-add.
    def __init__(self, h_dim: int, feat_width: int):
        super().__init__()
        self.proj = nn.Linear(h_dim, feat_width)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if feat.shape[0] != h.shape[0]:
            logger.error("[ConcatInjector] batch mismatch feat=%s h=%s",
                         feat.shape, h.shape)
            raise ValueError("batch dim mismatch")
        return feat + self.proj(h)
