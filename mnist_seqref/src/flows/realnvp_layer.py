# =============================================================================
# STEP-1_1 v0.3 -- models.flows.realnvp_layer
# LIFETIME: KEEP
# Purpose: RealNVP affine coupling (Dinh+ 2017), s,t produced by FiLM-conditioned
#          MLP on x1. y2 = x2 * exp(s(x1, h)) + t(x1, h).
# CONVENTION: s bounded by tanh to keep exp(s) finite; ldj finite-check on fwd.
#             Any non-finite -> logger.error + raise.
# Changelog (v0.2 -> v0.3, SEQREF-SMAX):
#   * s_max validated (>0) and unchanged default 2.0; now intended to be
#     config-driven via base_experts/train_base.
#   * NEW post_init_std kwarg (default 0.0 = exact v0.2 zeros-init behaviour).
#     >0 -> post.weight ~ N(0, std); post.bias stays zero.
#   * NEW s_clamp_frac diagnostic: after each _st call, records (no_grad)
#     self.last_s_clamp_frac = frac(|s| > 0.99 * s_max). Read by the
#     train_base per-epoch probe. No effect on forward/inverse math.
# Changelog (v0.1 -> v0.2):
#   * Threads film_hidden / film_depth / film_use_gelu kwargs to FiLMHead.
# Changelog (NEW in v0.1):
#   * Standard affine coupling, half-and-half alternating mask, FiLM in the
#     middle layer; s_max tanh-clipped at 2.0 by default.
# Update summary:
#   v0.3 wires the two SEQREF-SMAX levers (s_max, post_init_std) and the
#   s_clamp_frac ceiling diagnostic. Defaults preserve v0.2 behaviour exactly:
#   s_max=2.0, post_init_std=0.0 (zeros init), diagnostic is passive.
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.3"
__abbr__ = "STEP-1_1"

import torch
import torch.nn as nn

from ..conditioner import FiLMHead, ConcatInjector


class RealNVPCoupling(nn.Module):
    def __init__(self, dim: int, hidden: int, h_dim: int, *,
                 flip: bool, use_film: bool = True, s_max: float = 2.0,
                 post_init_std: float = 0.0,
                 film_hidden: int = 64, film_depth: int = 1,
                 film_use_gelu: bool = False):
        super().__init__()
        if dim % 2 != 0:
            logger.error("[RealNVPCoupling] dim must be even, got %d", dim)
            raise ValueError(f"dim must be even, got {dim}")
        if s_max <= 0.0:
            logger.error("[RealNVPCoupling] s_max must be > 0, got %s", s_max)
            raise ValueError(f"s_max must be > 0, got {s_max}")
        if post_init_std < 0.0:
            logger.error("[RealNVPCoupling] post_init_std must be >= 0, got %s",
                         post_init_std)
            raise ValueError(f"post_init_std must be >= 0, got {post_init_std}")
        self.dim = dim
        self.flip = flip
        self.d_in = dim // 2
        self.d_out = dim - self.d_in
        self.use_film = use_film
        self.s_max = float(s_max)
        self.last_s_clamp_frac = float("nan")   # diagnostic, set by _st
        self.pre  = nn.Linear(self.d_in, hidden)
        self.mid  = nn.Linear(hidden, hidden)
        self.post = nn.Linear(hidden, 2 * self.d_out)    # -> (s, t)
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)
        if post_init_std > 0.0:
            nn.init.normal_(self.post.weight, mean=0.0, std=post_init_std)
        if use_film:
            self.film = FiLMHead(h_dim, hidden,
                                 hidden=film_hidden,
                                 depth=film_depth,
                                 use_gelu=film_use_gelu)
        else:
            self.concat = ConcatInjector(h_dim, hidden)

    def _split(self, x):
        if self.flip:
            return x[..., self.d_in:], x[..., :self.d_in]
        return x[..., :self.d_in], x[..., self.d_in:]

    def _merge(self, a, b):
        return torch.cat([b, a], dim=-1) if self.flip else torch.cat([a, b], dim=-1)

    def _st(self, x1: torch.Tensor, h: torch.Tensor):
        z = torch.relu(self.pre(x1))
        if self.use_film:
            gamma, beta = self.film(h)
            z = gamma * z + beta
        else:
            z = self.concat(z, h)
        z = torch.relu(self.mid(z))
        out = self.post(z)
        s, t = out.chunk(2, dim=-1)
        s = self.s_max * torch.tanh(s)
        with torch.no_grad():
            # frac of s-values pinned near the tanh ceiling (|s| > 0.99*s_max)
            self.last_s_clamp_frac = float(
                (s.abs() > 0.99 * self.s_max).float().mean())
        return s, t

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        x1, x2 = self._split(x)
        s, t = self._st(x1, h)
        y2 = x2 * torch.exp(s) + t
        y = self._merge(x1, y2)
        ldj = s.sum(dim=-1)
        if not torch.isfinite(ldj).all():
            logger.error("[RealNVPCoupling.fwd] non-finite ldj")
            raise ValueError("non-finite ldj in RealNVP coupling forward")
        return y, ldj

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y1, y2 = self._split(y)
        s, t = self._st(y1, h)
        x2 = (y2 - t) * torch.exp(-s)
        return self._merge(y1, x2)
