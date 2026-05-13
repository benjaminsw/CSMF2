# =============================================================================
# STEP-1_1 v0.1 -- models.flows.nsf_layer
# Purpose: Neural Spline Flow (Durkan+ 2019) coupling layer. Monotonic
#          rational-quadratic (RQ) spline on [-B, B] with linear tails; K bins.
#          Conditioner (FiLM) emits hidden features that the spline param net
#          turns into (widths, heights, derivatives).
# CONVENTION: Gregory-Delbourgo parameterisation. Numerical guards explicit; any
#             failure -> logger.error + raise.
# Changelog (NEW in v0.1):
#   * Introduced. Forward via bin lookup + closed-form quotient; inverse via
#     closed-form quadratic root (numerically stable branch).
#   * B = 3.0, K = 8 by default (paper defaults for tabular).
#   * Linear tails (identity outside [-B, B]) so the transform handles
#     unbounded logit-space MNIST inputs.
# Update summary:
#   Single-file RQ-spline implementation sufficient for step_1_1. This is not a
#   full multi-scale NSF; we compose ~6 such couplings over the flat (B, D)
#   MNIST vector. Matches the WP0 goal "same conditioner, same degradation
#   pipeline across expert families" -- the spline differs only in the elementwise
#   transform at the tail of the coupling block.
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_1"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..conditioner import FiLMHead, ConcatInjector

DEFAULT_MIN_BIN_WIDTH  = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def _rq_spline(inputs: torch.Tensor,
               unnorm_widths: torch.Tensor,
               unnorm_heights: torch.Tensor,
               unnorm_derivs: torch.Tensor,
               *, inverse: bool, B: float
               ) -> tuple[torch.Tensor, torch.Tensor]:
    # inputs: (..., D) ; params: (..., D, K) for widths/heights, (..., D, K-1) for derivs.
    inside = (inputs >= -B) & (inputs <= B)
    outside = ~inside

    out = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)
    out[outside] = inputs[outside]               # linear tails -> identity, ldj=0

    if inside.any():
        ins = inputs[inside]
        uw = unnorm_widths[inside]
        uh = unnorm_heights[inside]
        ud = unnorm_derivs[inside]

        K = uw.shape[-1]
        widths  = F.softmax(uw, dim=-1)
        widths  = DEFAULT_MIN_BIN_WIDTH + (1 - DEFAULT_MIN_BIN_WIDTH * K) * widths
        heights = F.softmax(uh, dim=-1)
        heights = DEFAULT_MIN_BIN_HEIGHT + (1 - DEFAULT_MIN_BIN_HEIGHT * K) * heights
        derivs_inner = DEFAULT_MIN_DERIVATIVE + F.softplus(ud)
        # pad with boundary derivatives = 1 so tails match slope of linear parts
        ones = torch.ones_like(derivs_inner[..., :1])
        derivs = torch.cat([ones, derivs_inner, ones], dim=-1)

        cum_w = torch.cumsum(widths, dim=-1)
        cum_w = F.pad(cum_w, [1, 0])
        cum_w = (2 * B) * cum_w - B
        cum_w[..., 0]  = -B
        cum_w[..., -1] =  B

        cum_h = torch.cumsum(heights, dim=-1)
        cum_h = F.pad(cum_h, [1, 0])
        cum_h = (2 * B) * cum_h - B
        cum_h[..., 0]  = -B
        cum_h[..., -1] =  B

        if inverse:
            bin_idx = torch.sum(cum_h[..., :-1] <= ins.unsqueeze(-1), dim=-1) - 1
        else:
            bin_idx = torch.sum(cum_w[..., :-1] <= ins.unsqueeze(-1), dim=-1) - 1
        bin_idx = bin_idx.clamp(0, K - 1).unsqueeze(-1)

        input_cum_w   = cum_w.gather(-1, bin_idx)[..., 0]
        input_bin_w   = widths.gather(-1, bin_idx)[..., 0] * (2 * B)
        input_cum_h   = cum_h.gather(-1, bin_idx)[..., 0]
        input_heights = heights.gather(-1, bin_idx)[..., 0] * (2 * B)
        input_deriv     = derivs.gather(-1, bin_idx)[..., 0]
        input_deriv_p1  = derivs.gather(-1, bin_idx + 1)[..., 0]
        input_delta     = input_heights / input_bin_w    # slope s^(k)

        if inverse:
            # solve quadratic: a xi^2 + b xi + c = 0
            a = ((ins - input_cum_h) *
                 (input_deriv_p1 + input_deriv - 2 * input_delta)
                 + input_heights * (input_delta - input_deriv))
            b = (input_heights * input_deriv
                 - (ins - input_cum_h) *
                 (input_deriv_p1 + input_deriv - 2 * input_delta))
            c = -input_delta * (ins - input_cum_h)
            disc = b ** 2 - 4 * a * c
            if (disc < 0).any():
                logger.error("[rq_spline.inverse] negative discriminant (min=%.3e)",
                             float(disc.min().item()))
                raise ValueError("rq_spline inverse: negative discriminant")
            root = 2 * c / (-b - torch.sqrt(disc))
            outputs_inside = root * input_bin_w + input_cum_w
            theta_1m = root * (1 - root)
            denom = input_delta + (
                input_deriv_p1 + input_deriv - 2 * input_delta) * theta_1m
            deriv_numer = (input_delta ** 2) * (
                input_deriv_p1 * root ** 2
                + 2 * input_delta * theta_1m
                + input_deriv * (1 - root) ** 2)
            logabs_inside = torch.log(deriv_numer) - 2 * torch.log(denom)
            out[inside] = outputs_inside
            logabsdet[inside] = -logabs_inside
        else:
            theta = (ins - input_cum_w) / input_bin_w
            theta_1m = theta * (1 - theta)
            numer = input_heights * (
                input_delta * theta ** 2 + input_deriv * theta_1m)
            denom = input_delta + (
                input_deriv_p1 + input_deriv - 2 * input_delta) * theta_1m
            outputs_inside = input_cum_h + numer / denom
            deriv_numer = (input_delta ** 2) * (
                input_deriv_p1 * theta ** 2
                + 2 * input_delta * theta_1m
                + input_deriv * (1 - theta) ** 2)
            logabs_inside = torch.log(deriv_numer) - 2 * torch.log(denom)
            out[inside] = outputs_inside
            logabsdet[inside] = logabs_inside

    if not torch.isfinite(out).all() or not torch.isfinite(logabsdet).all():
        logger.error("[_rq_spline] non-finite output (out=%s, ldj=%s)",
                     bool(torch.isfinite(out).all()), bool(torch.isfinite(logabsdet).all()))
        raise ValueError("non-finite output in RQ spline")
    return out, logabsdet


class NSFCoupling(nn.Module):
    def __init__(self, dim: int, hidden: int, h_dim: int, *,
                 flip: bool, K: int = 8, B: float = 3.0, use_film: bool = True):
        super().__init__()
        if dim % 2 != 0:
            logger.error("[NSFCoupling] dim must be even, got %d", dim)
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.flip = flip
        self.d_in = dim // 2
        self.d_out = dim - self.d_in
        self.K = K
        self.B = B
        self.use_film = use_film
        self.pre  = nn.Linear(self.d_in, hidden)
        self.mid  = nn.Linear(hidden, hidden)
        # 3K - 1 params per output dim: K widths, K heights, K-1 inner derivs
        self.post = nn.Linear(hidden, self.d_out * (3 * K - 1))
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)
        if use_film:
            self.film = FiLMHead(h_dim, hidden)
        else:
            self.concat = ConcatInjector(h_dim, hidden)

    def _split(self, x):
        if self.flip:
            return x[..., self.d_in:], x[..., :self.d_in]
        return x[..., :self.d_in], x[..., self.d_in:]

    def _merge(self, a, b):
        return torch.cat([b, a], dim=-1) if self.flip else torch.cat([a, b], dim=-1)

    def _params(self, x1: torch.Tensor, h: torch.Tensor):
        z = torch.relu(self.pre(x1))
        if self.use_film:
            gamma, beta = self.film(h)
            z = gamma * z + beta
        else:
            z = self.concat(z, h)
        z = torch.relu(self.mid(z))
        raw = self.post(z)                             # (B, d_out * (3K-1))
        raw = raw.view(x1.shape[0], self.d_out, 3 * self.K - 1)
        uw = raw[..., :self.K]
        uh = raw[..., self.K:2 * self.K]
        ud = raw[..., 2 * self.K:]
        return uw, uh, ud

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        x1, x2 = self._split(x)
        uw, uh, ud = self._params(x1, h)
        y2, ldj_dim = _rq_spline(x2, uw, uh, ud, inverse=False, B=self.B)
        y = self._merge(x1, y2)
        return y, ldj_dim.sum(dim=-1)

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y1, y2 = self._split(y)
        uw, uh, ud = self._params(y1, h)
        x2, _ = _rq_spline(y2, uw, uh, ud, inverse=True, B=self.B)
        return self._merge(y1, x2)
