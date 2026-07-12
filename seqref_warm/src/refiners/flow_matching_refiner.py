# SEQREF-FMR v0.1 -- refiners.flow_matching_refiner
# LIFETIME: KEEP
# Purpose: Level-3 flow-matching refiner (SEQREF-FMREFINE v0.1 §4-7).
#          vθ(x_t, t, cond) = small residual CNN; sinusoidal t-embed -> per-block
#          FiLM. Velocity head ZERO-INIT -> v≈0 -> Δx_FM≈0 at init (identity-
#          safe). Euler rollout from the MEAN x1 (Option-A: one shared field).
#          Gate: GatedUpdate on pooled trunk features from the first rollout
#          step (conditioning-dependent, deterministic).
# CONVENTION: CNN, NOT couplings -- coupling vθ would re-import the proven
#             Level-1 ceiling. No fallback/mock/pass; failures log+raise.
# Changelog (v0.1):
#   * TimeEmbed (sinusoidal+MLP), FMBlock (conv-GELU-conv residual + t-FiLM),
#     FMRefiner (vθ + features + rollout + gated apply).
from __future__ import annotations
import logging
import math

import torch
import torch.nn as nn

from .gated_update import GatedUpdate

logger = logging.getLogger("seqref_warm.refiners.flow_matching_refiner")
__version__ = "0.1"
__abbr__ = "SEQREF-FMR"


class TimeEmbed(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        if dim % 2 != 0:
            logger.error("[TimeEmbed] dim must be even, got %d", dim)
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                 nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0,1] -> (B, dim)
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half,
                                         device=t.device))
        ang = t[:, None] * freqs[None, :] * 2 * math.pi
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return self.mlp(emb)


class FMBlock(nn.Module):
    def __init__(self, hidden: int, t_dim: int):
        super().__init__()
        self.c1 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.c2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.film = nn.Linear(t_dim, 2 * hidden)
        self.act = nn.GELU()

    def forward(self, z: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        gb = self.film(temb)[:, :, None, None]
        gamma, beta = gb.chunk(2, dim=1)
        h = self.act(self.c1(z))
        h = gamma * h + beta
        h = self.c2(self.act(h))
        return z + h


class FMRefiner(nn.Module):
    def __init__(self, *, cond_channels: int, hidden: int = 64, depth: int = 4,
                 t_embed_dim: int = 64, g_max: float = 0.5,
                 g_init: float = 0.05):
        super().__init__()
        if cond_channels not in (3, 4):
            logger.error("[FMRefiner] cond_channels must be 3 (Arm A) or 4 "
                         "(Arm B), got %d", cond_channels)
            raise ValueError(f"cond_channels must be 3 or 4, got {cond_channels}")
        self.cond_channels = cond_channels
        self.hidden = hidden
        self.temb = TimeEmbed(t_embed_dim)
        self.conv_in = nn.Conv2d(1 + cond_channels, hidden, 3, padding=1)
        self.blocks = nn.ModuleList(FMBlock(hidden, t_embed_dim)
                                    for _ in range(depth))
        self.head = nn.Conv2d(hidden, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)    # v ~= 0 at init -> identity-safe
        nn.init.zeros_(self.head.bias)
        self.gate = GatedUpdate(hidden, g_max=g_max, g_init=g_init)
        self.g_max = g_max

    def _trunk(self, x_t: torch.Tensor, t: torch.Tensor,
               cond: torch.Tensor) -> torch.Tensor:
        if x_t.shape[1] != 1 or cond.shape[1] != self.cond_channels:
            logger.error("[FMRefiner] bad channels x_t=%s cond=%s",
                         tuple(x_t.shape), tuple(cond.shape))
            raise ValueError("FMRefiner channel mismatch")
        temb = self.temb(t)
        z = self.conv_in(torch.cat([x_t, cond], dim=1))
        for b in self.blocks:
            z = b(z, temb)
        return z

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor,
                 cond: torch.Tensor) -> torch.Tensor:
        v = self.head(self._trunk(x_t, t, cond))
        if not torch.isfinite(v).all():
            logger.error("[FMRefiner] non-finite velocity")
            raise ValueError("non-finite velocity")
        return v

    def features(self, x_t: torch.Tensor, t: torch.Tensor,
                 cond: torch.Tensor) -> torch.Tensor:
        # pooled trunk features -> (B, hidden); used for the gate head.
        return self._trunk(x_t, t, cond).mean(dim=(2, 3))

    def rollout(self, x1: torch.Tensor, cond: torch.Tensor,
                k: int) -> torch.Tensor:
        # Euler from the MEAN (Option-A). Returns xFM.
        if k < 1:
            logger.error("[FMRefiner] k must be >= 1, got %d", k)
            raise ValueError(f"k must be >= 1, got {k}")
        x = x1
        for i in range(k):
            t = torch.full((x.size(0),), (i + 1) / k, device=x.device,
                           dtype=x.dtype)
            x = x + (1.0 / k) * self.velocity(x, t, cond)
        return x

    def forward(self, x1: torch.Tensor, cond: torch.Tensor, k: int
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns (x2, dx_fm, g). One shared Δx per observation.
        xfm = self.rollout(x1, cond, k)
        dx = xfm - x1
        t0 = torch.full((x1.size(0),), 1.0 / k, device=x1.device,
                        dtype=x1.dtype)
        h = self.features(x1, t0, cond)
        x2, g = self.gate(x1, dx, h)
        return x2, dx, g
