# =============================================================================
# SEQREF-MRI-V04L v0.1 -- scripts.v04_objective
# LIFETIME: KEEP
# Purpose: V04 hybrid objective (V04PLAN v1.0 article 178 SS4, FROZEN):
#            L = L_density + lam * (s * L_rec_base),   s = 1e7 (locked, A6).
#          Thin adapter: CONSUMES the registered V03 objective terms verbatim
#          (src.flows.v03_objective v0.3 + train_free_flow.nll_objective),
#          never reimplements them. lam == 0.0 returns the density-only NLL
#          tensor UNTOUCHED (bit-exact path; fixture f02 gates this) and the
#          R0 decode branch is never entered.
# CONVENTION: logger.error + typed raise (V04LError). No fallback, no mock,
#             no placeholder, no silent pass.
# =============================================================================
from __future__ import annotations

import logging
import math

import torch

from seqref_mri.src.flows import v03_objective as v03l
from seqref_mri.scripts import train_free_flow as tff

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "SEQREF-MRI-V04L"

L_REC_SCALE = v03l.L_REC_SCALE   # locked 1e7 (SPEC Amendment A6, art. 171)


class V04LError(RuntimeError):
    """Single typed identity for the V04 objective module."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V04LError(code, message)


def _check_model_interface(model) -> None:
    """V04L consumes the free-flow model interface; verify, never assume."""
    for name in ("log_prob_free", "decode_scalars", "encode_free",
                 "condition"):
        if not callable(getattr(model, name, None)):
            _fail("V04L_INTERFACE_MISMATCH",
                  f"model lacks callable `{name}`; V04L requires the "
                  "free-flow model interface")


def _check_lam(lam: float) -> float:
    lam = float(lam)
    if not math.isfinite(lam) or lam < 0.0:
        _fail("LAMBDA_INVALID",
              f"lam={lam!r} must be finite and >= 0")
    return lam


def density_only(model, u_scaled: torch.Tensor, cond_in: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
    """The density-only path: the registered V02 production NLL, consumed."""
    _check_model_interface(model)
    return tff.nll_objective(model, u_scaled, cond_in, mask)


def rec_term(model, cond_in: torch.Tensor, mask: torch.Tensor,
             y_raw: torch.Tensor, amax: torch.Tensor,
             tgt_norm: torch.Tensor, maps: list) -> torch.Tensor:
    """L_rec = s * MSE(|R0|, tgt_norm) with the pinned scale s = 1e7 (A6)."""
    _check_model_interface(model)
    return v03l.l_rec(model, cond_in, mask, y_raw, amax, tgt_norm, maps)


def combined_loss(model, u_scaled: torch.Tensor, cond_in: torch.Tensor,
                  mask: torch.Tensor, *, lam: float,
                  y_raw: torch.Tensor, amax: torch.Tensor,
                  tgt_norm: torch.Tensor, maps: list) -> torch.Tensor:
    """L = L_density + lam * L_rec (L_rec carries the pinned 1e7 scale).
    lam == 0.0 returns the density-only NLL tensor UNTOUCHED (fixture f02
    gates bit-exactness; the R0 branch is never entered at lam=0)."""
    _check_model_interface(model)
    return v03l.combined_loss(model, u_scaled, cond_in, mask,
                              lam=_check_lam(lam), y_raw=y_raw, amax=amax,
                              tgt_norm=tgt_norm, maps=maps)
