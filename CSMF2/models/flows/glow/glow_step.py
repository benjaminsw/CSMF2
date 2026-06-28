# =============================================================================
# STEP-1_1 v0.2 -- models.flows.glow.glow_step
# Purpose: one Glow step (paper Fig. 2a) = Actnorm -> InvertibleConv1x1 ->
#          coupling. ldj accumulates across the three sub-layers.
# CONVENTION: any non-finite -> logger.error + raise.
# Changelog (v0.1 -> v0.2) [FLOWPP v0.1]:
#   * NEW: coupling_type in {"affine","mix_logistic"} (default "affine") and
#     n_mixtures (default 4). "affine" -> AffineCoupling2D (Glow path, BYTE-
#     IDENTICAL to v0.1 since defaults reproduce the old call). "mix_logistic"
#     -> MixLogisticCoupling2D (Flow++ candidate). actnorm + inv1x1 are shared
#     unchanged: the coupling primitive is the only variable that moves.
# Changelog (NEW in v0.1):
#   * Introduced as part of glow_improved -> shared merge. Thin composer.
# Update summary:
#   v0.2 lets the same composer host either the affine (Glow) or logistic-
#   mixture (Flow++) coupling, so CondFlowpp reuses the Glow backbone verbatim.
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.2"
__abbr__ = "STEP-1_1"

import torch
import torch.nn as nn

from .actnorm import Actnorm
from .invertible_1x1 import InvertibleConv1x1
from .affine_coupling_2d import AffineCoupling2D
from .mix_logistic_coupling_2d import MixLogisticCoupling2D


class GlowStep(nn.Module):
    def __init__(self, num_channels: int, coupling_hidden: int, h_dim: int,
                 *, flip: bool, s_max: float,
                 film_hidden: int, film_depth: int, film_use_gelu: bool,
                 inv1x1_seed: int | None,
                 film_gain_init: float = 0.3,
                 coupling_type: str = "affine", n_mixtures: int = 4):
        super().__init__()
        self.actnorm = Actnorm(num_channels)
        self.inv1x1  = InvertibleConv1x1(num_channels, seed=inv1x1_seed)
        if coupling_type == "affine":
            self.coupling = AffineCoupling2D(
                num_channels=num_channels, hidden=coupling_hidden, h_dim=h_dim,
                flip=flip, s_max=s_max,
                film_hidden=film_hidden, film_depth=film_depth,
                film_use_gelu=film_use_gelu,
                film_gain_init=film_gain_init)
        elif coupling_type == "mix_logistic":
            self.coupling = MixLogisticCoupling2D(
                num_channels=num_channels, hidden=coupling_hidden, h_dim=h_dim,
                flip=flip, s_max=s_max, n_mixtures=n_mixtures,
                film_hidden=film_hidden, film_depth=film_depth,
                film_use_gelu=film_use_gelu,
                film_gain_init=film_gain_init)
        else:
            logger.error("[GlowStep] coupling_type must be 'affine' or "
                         "'mix_logistic', got %r", coupling_type)
            raise ValueError(
                f"coupling_type must be 'affine'/'mix_logistic', got "
                f"{coupling_type!r}")

    def forward(self, x: torch.Tensor, h: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        z, d1 = self.actnorm(x)
        z, d2 = self.inv1x1(z)
        z, d3 = self.coupling(z, h)
        return z, d1 + d2 + d3

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = self.coupling.inverse(y, h)
        x = self.inv1x1.inverse(x)
        x = self.actnorm.inverse(x)
        return x
