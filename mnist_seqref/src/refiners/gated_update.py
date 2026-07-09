# SEQREF-GATEUPD v0.1 -- refiners.gated_update
# LIFETIME: KEEP
# Purpose: identity-safe gated residual update x = x_prev + g * dx with
#          per-sample bounded gate g = g_max * sigmoid(Linear(h)),
#          0 <= g <= g_max. Gate bias initialised so g ~= g_init (default
#          0.05, NOT 0 -- an exact-zero gate starves dL/d(dx) when the loss
#          is on the final image). Weight zero-init -> uniform g_init at
#          step 0, sample-dependence learned.
# CONVENTION: no fallback/mock/pass; failures logger.error + raise.
# Changelog (v0.1):
#   * GatedUpdate module + g stats (mean/std/min/max, g_max_frac).
from __future__ import annotations
import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger("mnist_seqref.refiners.gated_update")
__version__ = "0.1"
__abbr__ = "SEQREF-GATEUPD"


class GatedUpdate(nn.Module):
    def __init__(self, h_dim: int, *, g_max: float = 0.5, g_init: float = 0.05):
        super().__init__()
        if not (0.0 < g_init < g_max):
            logger.error("[GatedUpdate] need 0 < g_init < g_max, got %s / %s",
                         g_init, g_max)
            raise ValueError(f"need 0 < g_init < g_max, got {g_init}/{g_max}")
        self.g_max = float(g_max)
        self.gate = nn.Linear(h_dim, 1)
        nn.init.zeros_(self.gate.weight)
        # sigmoid(bias) = g_init / g_max  =>  bias = logit(g_init/g_max)
        frac = g_init / g_max
        nn.init.constant_(self.gate.bias, math.log(frac / (1.0 - frac)))

    def g(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, h_dim) -> g: (B,) in (0, g_max)
        return self.g_max * torch.sigmoid(self.gate(h)).squeeze(-1)

    def forward(self, x_prev: torch.Tensor, dx: torch.Tensor,
                h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_prev, dx: (B,1,28,28); h: (B, h_dim). Returns (x_new, g (B,)).
        if x_prev.shape != dx.shape:
            logger.error("[GatedUpdate] shape mismatch x_prev=%s dx=%s",
                         tuple(x_prev.shape), tuple(dx.shape))
            raise ValueError("GatedUpdate shape mismatch")
        g = self.g(h)
        x_new = x_prev + g.view(-1, 1, 1, 1) * dx
        if not torch.isfinite(x_new).all():
            logger.error("[GatedUpdate] non-finite x_new")
            raise ValueError("non-finite gated update")
        return x_new, g

    @staticmethod
    def g_stats(g: torch.Tensor, g_max: float) -> dict:
        return {"g_mean": float(g.mean()), "g_std": float(g.std()),
                "g_min": float(g.min()), "g_max_val": float(g.max()),
                "g_max_frac": float((g > 0.95 * g_max).float().mean())}
