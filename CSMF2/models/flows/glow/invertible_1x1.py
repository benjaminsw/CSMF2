# =============================================================================
# STEP-1_1 v0.1 -- models.flows.glow.invertible_1x1
# Purpose: Invertible 1x1 convolution (Glow paper section 3.2) with LU
#          parameterisation W = P * L * (U + diag(sign_s * exp(log_s))).
#          logdet = H * W * sum(log|s|).
# CONVENTION: any non-finite -> logger.error + raise. No fallback.
# Changelog (NEW in v0.1):
#   * Introduced as part of glow_improved -> shared merge.
# Update summary:
#   For MNIST-after-squeeze C=4; both LU and full-W paths are cheap. LU used
#   here for clean logdet accounting and parity with the paper.
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_1"

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels: int, *, seed: int | None = None):
        super().__init__()
        if num_channels < 1:
            logger.error("[InvertibleConv1x1] num_channels must be >=1, got %d",
                         num_channels)
            raise ValueError(f"num_channels {num_channels} < 1")
        self.num_channels = num_channels
        c = num_channels
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(int(seed))
        W_init = torch.linalg.qr(torch.randn(c, c, generator=g))[0]
        P_, L_, U_ = torch.linalg.lu(W_init)
        s_ = torch.diagonal(U_).clone()
        sign_s_ = torch.sign(s_)
        log_s_ = torch.log(torch.abs(s_).clamp_min(1e-8))
        U_ = U_ - torch.diag(s_)
        self.register_buffer("P", P_)
        self.register_buffer("sign_s", sign_s_)
        self.register_buffer("L_mask",
                             torch.tril(torch.ones(c, c), diagonal=-1))
        self.register_buffer("U_mask",
                             torch.triu(torch.ones(c, c), diagonal=1))
        self.register_buffer("eye", torch.eye(c))
        self.L     = nn.Parameter(L_)
        self.U     = nn.Parameter(U_)
        self.log_s = nn.Parameter(log_s_)

    def _W(self) -> torch.Tensor:
        L = self.L * self.L_mask + self.eye
        U = self.U * self.U_mask
        S = torch.diag(self.sign_s * torch.exp(self.log_s))
        return self.P @ L @ (U + S)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 4 or x.size(1) != self.num_channels:
            logger.error("[InvertibleConv1x1.forward] expected (B,%d,H,W), got %s",
                         self.num_channels, tuple(x.shape))
            raise ValueError("shape mismatch")
        W = self._W()
        y = F.conv2d(x, W.view(self.num_channels, self.num_channels, 1, 1))
        H, W_spatial = x.shape[2], x.shape[3]
        ldj = (self.log_s.sum() * H * W_spatial).expand(x.shape[0])
        if not torch.isfinite(y).all() or not torch.isfinite(ldj).all():
            logger.error("[InvertibleConv1x1.forward] non-finite y/ldj")
            raise ValueError("non-finite InvertibleConv1x1 output/ldj")
        return y, ldj

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 4 or y.size(1) != self.num_channels:
            logger.error("[InvertibleConv1x1.inverse] expected (B,%d,H,W), got %s",
                         self.num_channels, tuple(y.shape))
            raise ValueError("shape mismatch")
        W = self._W()
        W_inv = torch.linalg.inv(W)
        return F.conv2d(y, W_inv.view(self.num_channels, self.num_channels, 1, 1))

    @torch.no_grad()
    def singular_values(self) -> torch.Tensor:
        # Used by sanity (w1x1_spectrum).
        return torch.linalg.svdvals(self._W().detach())
