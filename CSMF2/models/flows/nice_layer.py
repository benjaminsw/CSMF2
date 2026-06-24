# =============================================================================
# STEP-1_1 v0.3 -- models.flows.nice_layer
# Purpose: NICE additive coupling (Dinh+ 2014) with FiLM-conditioned m(x1 | h).
#          Unit Jacobian determinant by construction; invertibility trivial.
# CONVENTION: No silent fallback. Any NaN / shape error -> logger.error + raise.
# Changelog (v0.2 -> v0.3):
#   * NEW FixedPermute(dim, seed): fixed dim permutation, log-det 0, exactly
#     invertible (inv=argsort(perm)), registered buffers (not trained). For the
#     NICE-MIX (NCP-N8) expressiveness ablation -- inserted between additive
#     couplings so dims mix across splits without any exp(s)/learned-1x1. NICE
#     identity (additive, unit-Jacobian) preserved. NICECoupling/DiagScale
#     unchanged.
# Changelog (v0.1 -> v0.2):
#   * NEW kwargs film_hidden (default 64), film_depth (default 1),
#     film_use_gelu (default False) on NICECoupling.__init__. Defaults match
#     v0.1 byte-for-byte; the kwargs simply pass through to FiLMHead.
#   * No change to forward/inverse/_m logic. ConcatInjector path retained.
# Changelog (NEW in v0.1):
#   * Introduced. Additive coupling y2 = x2 + m(x1, h); y1 = x1.
#   * Supports FiLM (gamma * hidden + beta) or concat injection.
#   * Alternating masks handled by `flip` flag passed from the expert module.
# Update summary:
#   v0.2 lets the caller (CondNICE) thread richer FiLM configurations without
#   touching the coupling math. When film_depth=1 + film_use_gelu=False +
#   film_hidden=64 the layer is byte-identical to v0.1.
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.3"
__abbr__ = "STEP-1_1"

import torch
import torch.nn as nn

from ..conditioner import FiLMHead, ConcatInjector


class NICECoupling(nn.Module):
    def __init__(self, dim: int, hidden: int, h_dim: int, *,
                 flip: bool, use_film: bool = True,
                 film_hidden: int = 64, film_depth: int = 1,
                 film_use_gelu: bool = False):
        super().__init__()
        if dim % 2 != 0:
            logger.error("[NICECoupling] dim must be even, got %d", dim)
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.flip = flip
        self.d_in = dim // 2
        self.d_out = dim - self.d_in
        self.use_film = use_film
        self.pre  = nn.Linear(self.d_in, hidden)
        self.mid  = nn.Linear(hidden, hidden)
        self.post = nn.Linear(hidden, self.d_out)
        if use_film:
            self.film = FiLMHead(h_dim, hidden, hidden=film_hidden,
                                 depth=film_depth, use_gelu=film_use_gelu)
        else:
            self.concat = ConcatInjector(h_dim, hidden)

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.flip:
            return x[..., self.d_in:], x[..., :self.d_in]
        return x[..., :self.d_in], x[..., self.d_in:]

    def _merge(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([b, a], dim=-1) if self.flip else torch.cat([a, b], dim=-1)

    def _m(self, x1: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.pre(x1))
        if self.use_film:
            gamma, beta = self.film(h)
            z = gamma * z + beta
        else:
            z = self.concat(z, h)
        z = torch.relu(self.mid(z))
        return self.post(z)

    def forward(self, x: torch.Tensor, h: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = self._split(x)
        y2 = x2 + self._m(x1, h)
        y = self._merge(x1, y2)
        ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return y, ldj

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y1, y2 = self._split(y)
        x2 = y2 - self._m(y1, h)
        return self._merge(y1, x2)


class DiagScale(nn.Module):
    # NICE's top-level diagonal rescaling. Exposes s = exp(log_s); ldj = sum(log_s).
    def __init__(self, dim: int):
        super().__init__()
        self.log_s = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = x * torch.exp(self.log_s)
        ldj = self.log_s.sum().expand(x.shape[0])
        return y, ldj

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y * torch.exp(-self.log_s)


class FixedPermute(nn.Module):
    """Fixed (non-learned) dimension permutation for NICE-MIX (NCP-N8).
    log-det 0 and exactly invertible by construction (inv = argsort(perm)), so
    it preserves NICE's unit-Jacobian / trivial-invertibility contract and the
    f64 logdet sanity check is unaffected. Matches the (x, h) -> (y, ldj) /
    inverse(y, h) signature of NICECoupling so it drops straight into the
    _BaseExpert.encode/decode layer stack. The permutation is a registered
    buffer (NOT a Parameter): fixed at construction, never trained -- this keeps
    NICE-MIX additive-only and distinct from a learned 1x1 (Glow) mixing."""
    def __init__(self, dim: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(dim, generator=g)
        self.register_buffer("perm", perm)
        self.register_buffer("inv", torch.argsort(perm))

    def forward(self, x: torch.Tensor, h: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return x[..., self.perm], ldj

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return y[..., self.inv]
