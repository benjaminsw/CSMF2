# =============================================================================
# STEP-1_1 v0.3 -- models.flows.glow.affine_coupling_2d
# Purpose: Glow-style affine coupling (channel-wise split). NN producing (s, t)
#          is a 3-conv block (3x3 -> 1x1 -> 3x3) with RESIDUAL FiLM injection
#          after each of the two hidden layers, gated by a learnable film_gain.
# CONVENTION: any non-finite -> logger.error + raise. No fallback.
# Changelog (v0.2 -> v0.3):
#   * RESIDUAL FiLM (replaces affine): each hidden layer applies
#         z' = z * (1 + film_gain * gamma_raw) + film_gain * beta
#     where (gamma_raw, beta) come from FiLMHead(output_form='residual') and
#     film_gain is a learnable scalar nn.Parameter (init film_gain_init=0.3
#     by default). This makes the conditioning contribution FIRST-ORDER in
#     the FiLM weights even when downstream conv3 has small weights.
#   * Why: v0.2 audit showed h.std stayed at 0.022 after 20 epochs of training
#     because the gradient path through FiLM was second-order (gradient ~
#     FiLM_weight * conv3_weight, both small). With residual FiLM and a non-
#     zero gain init, any movement of FiLM weights immediately changes the
#     output -> gradient ~ film_gain (initial 0.3) * FiLM_weight, ~100x larger.
#   * film_gain itself is trainable; if model abandons conditioning it will
#     decay; logged per-epoch in report.json for visibility.
# Changelog (v0.1 -> v0.2):
#   * BUGFIX: conv3 zero-init removed (small-normal std=0.01). First attempt
#     at fixing Glow conditioner death; insufficient alone -- see v0.3.
# Changelog (NEW in v0.1):
#   * Introduced as part of glow_improved -> shared merge.
# Update summary:
#   v0.3 is the structural fix for the Glow conditioner-death bug. The
#   conv3 small-normal init from v0.2 is KEPT (no harm); the residual FiLM
#   + learnable gain is what actually restores conditioning. NICE / RealNVP
#   are untouched (their FiLM still uses output_form='affine').
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.3"
__abbr__ = "STEP-1_1"

import torch
import torch.nn as nn

from ...conditioner import FiLMHead


class AffineCoupling2D(nn.Module):
    def __init__(self, num_channels: int, hidden: int, h_dim: int,
                 *, flip: bool, s_max: float = 2.0,
                 film_hidden: int = 128, film_depth: int = 2,
                 film_use_gelu: bool = True,
                 film_gain_init: float = 0.3):
        super().__init__()
        if num_channels < 2 or num_channels % 2:
            logger.error("[AffineCoupling2D] num_channels must be even >=2, got %d",
                         num_channels)
            raise ValueError(f"num_channels must be even >=2, got {num_channels}")
        if film_gain_init < 0.0:
            logger.error("[AffineCoupling2D] film_gain_init must be >=0, got %s",
                         film_gain_init)
            raise ValueError(f"film_gain_init must be >=0, got {film_gain_init}")
        self.num_channels = num_channels
        self.c_in  = num_channels // 2
        self.c_out = num_channels - self.c_in
        self.flip  = flip
        self.s_max = s_max
        self.conv1 = nn.Conv2d(self.c_in,  hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden,     hidden, 1, padding=0)
        self.conv3 = nn.Conv2d(hidden,     2 * self.c_out, 3, padding=1)
        # v0.2: small-normal init on conv3 (KEPT in v0.3 -- no harm).
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.conv3.bias)
        # v0.3: residual FiLM. FiLM heads return (gamma_raw, beta) and the
        # learnable film_gain controls the effective conditioning strength.
        self.film1 = FiLMHead(h_dim, hidden,
                              hidden=film_hidden,
                              depth=film_depth,
                              use_gelu=film_use_gelu,
                              output_form="residual")
        self.film2 = FiLMHead(h_dim, hidden,
                              hidden=film_hidden,
                              depth=film_depth,
                              use_gelu=film_use_gelu,
                              output_form="residual")
        self.film_gain = nn.Parameter(torch.tensor(float(film_gain_init)))

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.flip:
            return x[:, self.c_in:], x[:, :self.c_in]
        return x[:, :self.c_in], x[:, self.c_in:]

    def _merge(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([b, a], dim=1) if self.flip else torch.cat([a, b], dim=1)

    def _st(self, x1: torch.Tensor, h: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
        # v0.3: residual FiLM
        #   z' = z * (1 + film_gain * gamma_raw) + film_gain * beta
        # Equivalent to legacy when film_gain=1 AND FiLMHead returns
        # gamma_raw = gamma - 1 -- but we use gamma_raw centred at 0, so the
        # contribution is FIRST-ORDER in FiLM weights from step 0.
        g = self.film_gain
        z = torch.relu(self.conv1(x1))
        gamma_raw, beta = self.film1(h)
        z = z * (1.0 + g * gamma_raw[:, :, None, None]) \
            + g * beta[:, :, None, None]
        z = torch.relu(self.conv2(z))
        gamma_raw, beta = self.film2(h)
        z = z * (1.0 + g * gamma_raw[:, :, None, None]) \
            + g * beta[:, :, None, None]
        out = self.conv3(z)
        s, t = out.chunk(2, dim=1)
        s = self.s_max * torch.tanh(s)
        return s, t

    def forward(self, x: torch.Tensor, h: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            logger.error("[AffineCoupling2D.forward] expected (B,C,H,W), got %s",
                         tuple(x.shape))
            raise ValueError("expected (B,C,H,W)")
        x1, x2 = self._split(x)
        s, t = self._st(x1, h)
        y2 = x2 * torch.exp(s) + t
        y = self._merge(x1, y2)
        ldj = s.flatten(1).sum(dim=1)
        if not torch.isfinite(ldj).all():
            logger.error("[AffineCoupling2D.forward] non-finite ldj")
            raise ValueError("non-finite ldj in AffineCoupling2D.forward")
        return y, ldj

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y1, y2 = self._split(y)
        s, t = self._st(y1, h)
        x2 = (y2 - t) * torch.exp(-s)
        return self._merge(y1, x2)
