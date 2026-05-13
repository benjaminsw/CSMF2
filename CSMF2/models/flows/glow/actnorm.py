# =============================================================================
# STEP-1_1 v0.1 -- models.flows.glow.actnorm
# Purpose: Actnorm (Glow paper section 3.1). Per-channel affine y = s*x + b
#          with data-dependent init on the first batch.
# CONVENTION: any non-finite -> logger.error + raise. No fallback.
# Changelog (NEW in v0.1):
#   * Introduced as part of glow_improved -> shared merge.
#   * `initialised` is a buffer so it survives state_dict() save/load.
#   * Layer RAISES if forward is called before init_from_batch.
# Update summary:
#   The data-dependent init is the standard idiom from the reference Glow
#   implementation. It is NOT a fallback; misuse raises loudly.
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_1"

import torch
import torch.nn as nn


class Actnorm(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        if num_channels < 1:
            logger.error("[Actnorm] num_channels must be >=1, got %d",
                         num_channels)
            raise ValueError(f"num_channels {num_channels} < 1")
        self.num_channels = num_channels
        self.eps = eps
        self.log_s = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.b     = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("initialised",
                             torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def init_from_batch(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            logger.error("[Actnorm.init_from_batch] expected (B,C,H,W), got %s",
                         tuple(x.shape))
            raise ValueError("expected (B,C,H,W)")
        if x.size(1) != self.num_channels:
            logger.error("[Actnorm.init_from_batch] C mismatch: layer=%d batch=%d",
                         self.num_channels, x.size(1))
            raise ValueError("C mismatch")
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        std  = x.std(dim=(0, 2, 3), keepdim=True)
        s = 1.0 / (std + self.eps)
        b = -mean * s
        self.log_s.data.copy_(torch.log(s.clamp_min(self.eps)))
        self.b.data.copy_(b)
        self.initialised.fill_(True)
        logger.info("[Actnorm] data-init: C=%d  s_mean=%.3f  b_mean=%.3f",
                    self.num_channels, float(s.mean()), float(b.mean()))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(self.initialised):
            logger.error("[Actnorm.forward] called before init_from_batch")
            raise RuntimeError("Actnorm called before data-dependent init")
        s = torch.exp(self.log_s)
        y = s * x + self.b
        H, W = x.shape[2], x.shape[3]
        ldj = (self.log_s.sum() * (H * W)).expand(x.shape[0])
        if not torch.isfinite(y).all() or not torch.isfinite(ldj).all():
            logger.error("[Actnorm.forward] non-finite output/ldj")
            raise ValueError("non-finite Actnorm output/ldj")
        return y, ldj

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if not bool(self.initialised):
            logger.error("[Actnorm.inverse] called before init_from_batch")
            raise RuntimeError("Actnorm.inverse before init")
        s = torch.exp(self.log_s)
        return (y - self.b) / s
