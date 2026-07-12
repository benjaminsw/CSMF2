# =============================================================================
# STEP-1_1 v0.3 -- models.flows.nice_layer
# LIFETIME: KEEP
# Purpose: NICE additive coupling (Dinh+ 2014) with FiLM-conditioned m(x1 | h).
#          Unit Jacobian determinant by construction; invertibility trivial.
# CONVENTION: No silent fallback. Any NaN / shape error -> logger.error + raise.
# Changelog (v0.2 -> v0.3, SEQREF-NICER3):
#   * NEW post_init_std kwarg on NICECoupling (default None = v0.2 default
#     PyTorch init, byte-identical). If set (>=0): post.weight ~ N(0, std)
#     (0.0 -> exact zeros), post.bias zeroed -> near-identity coupling at init.
#   * NEW FixedPermute module (R3): fixed seeded permutation of feature dims,
#     exact inverse via precomputed argsort, ldj = 0, (x, h) signature to slot
#     into _BaseExpert.encode/decode. Permutation depends only on (dim, seed),
#     NOT on run RNG state -- identical across arms/seeds by construction, and
#     the module consumes no global RNG (private torch.Generator).
# Changelog (v0.1 -> v0.2):
#   * film_hidden / film_depth / film_use_gelu kwargs threaded to FiLMHead.
# Update summary:
#   v0.3 adds the two SEQREF-NICER3 ingredients: the post_init_std carry-over
#   (applied to BOTH A/B arms) and FixedPermute (the single A-vs-B difference).
#   Defaults (post_init_std=None, FixedPermute unused) reproduce v0.2 exactly.
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
                 post_init_std: float | None = None,
                 film_hidden: int = 64, film_depth: int = 1,
                 film_use_gelu: bool = False):
        super().__init__()
        if dim % 2 != 0:
            logger.error("[NICECoupling] dim must be even, got %d", dim)
            raise ValueError(f"dim must be even, got {dim}")
        if post_init_std is not None and post_init_std < 0.0:
            logger.error("[NICECoupling] post_init_std must be >= 0 or None, "
                         "got %s", post_init_std)
            raise ValueError(f"post_init_std must be >= 0 or None, got {post_init_std}")
        self.dim = dim
        self.flip = flip
        self.d_in = dim // 2
        self.d_out = dim - self.d_in
        self.use_film = use_film
        self.pre  = nn.Linear(self.d_in, hidden)
        self.mid  = nn.Linear(hidden, hidden)
        self.post = nn.Linear(hidden, self.d_out)
        if post_init_std is not None:
            nn.init.zeros_(self.post.weight)
            nn.init.zeros_(self.post.bias)
            if post_init_std > 0.0:
                nn.init.normal_(self.post.weight, mean=0.0, std=post_init_std)
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


class FixedPermute(nn.Module):
    # R3 (SEQREF-NICER3): fixed feature permutation between additive couplings.
    # Non-learnable, volume-preserving (ldj = 0), exact inverse. Permutation is
    # a pure function of (dim, seed) via a private Generator: identical across
    # runs/arms/seeds, and never touches global RNG state.
    def __init__(self, dim: int, seed: int):
        super().__init__()
        if dim < 2:
            logger.error("[FixedPermute] dim must be >= 2, got %d", dim)
            raise ValueError(f"dim must be >= 2, got {dim}")
        g = torch.Generator()
        g.manual_seed(int(seed))
        perm = torch.randperm(dim, generator=g)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))

    def forward(self, x: torch.Tensor, h: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.size(-1) != self.perm.numel():
            logger.error("[FixedPermute.fwd] dim mismatch: x %s vs perm %d",
                         tuple(x.shape), self.perm.numel())
            raise ValueError("FixedPermute dim mismatch")
        ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return x[..., self.perm], ldj

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if y.size(-1) != self.inv_perm.numel():
            logger.error("[FixedPermute.inv] dim mismatch: y %s vs perm %d",
                         tuple(y.shape), self.inv_perm.numel())
            raise ValueError("FixedPermute dim mismatch")
        return y[..., self.inv_perm]


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
