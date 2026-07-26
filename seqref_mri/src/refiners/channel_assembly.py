# =============================================================================
# SEQREF-I2 v0.2 -- src.refiners.channel_assembly
# LIFETIME: KEEP
# Purpose: the locked 3.8 conditioning-channel contract, implemented as TWO
#   separated steps (review amendment):
#     raw      = assemble_raw_channels(x0, y, op)      -- ALWAYS valid
#     channels = normalize_channels(raw, scales)       -- locked contract
#   plus the model-facing gate:
#     model_channels(x0, y, op, scales)                -- provenance-gated
# Channel identities (LOCKED, EXEC 3.8): [|x0|, Re(A^H r), Im(A^H r)],
#   r = y - A(x0). Never a silent magnitude collapse.
# Normalization contract (LOCKED): normalized_c = channel_c / (scale_c + 1e-8);
#   every scale must be declared, finite, and strictly > 1e-8 or this module
#   RAISES. No clipping (out-of-range recorded by callers).
# Provenance gate: ChannelScales carries provenance in
#   {"provisional-zero-filled", "locked-I7"}. model_channels() accepts ONLY
#   "locked-I7" -- provisional zero-filled statistics are diagnostics and can
#   NEVER enter trainer/model construction. Rationale: for exact zero-filled
#   x0 = A^H y the residual channels are mathematically near zero (S3
#   measured ~1e-18), so zero-filled data cannot produce valid positive
#   scales for all three channels; the gate makes that impossibility a
#   structural error instead of a silent contract violation.
# CONVENTION: logger.error + raise on every failure path. No fallback.
# Changelog (v0.1 -> v0.2, bugfix): _as_complex_state checks dim() >= 3
#   before reading shape[-3] (malformed low-dim tensors raised an
#   uncontrolled IndexError instead of the convention error).
# Changelog (NEW in v0.1):
#   * Introduced ChannelScales, assemble_raw_channels, normalize_channels,
#     model_channels.
# Update summary: the 3.8 contract is now code with the provisional/final
#   separation enforced by construction, matching the locked timing (final
#   scales from the frozen I7 winner only).
# =============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from ..forward_operator import MaskedFourierOperator

logger = logging.getLogger("seqref_mri.refiners.channel_assembly")

__version__ = "0.2"
__abbr__ = "SEQREF-I2"

_NORM_EPS = 1e-8              # EXEC 3.8, locked
_SCALE_MIN = 1e-8             # EXEC 3.8: scale must be finite and > this
_PROVENANCES = ("provisional-zero-filled", "locked-I7")
_MODEL_PROVENANCE = "locked-I7"


@dataclass(frozen=True)
class ChannelScales:
    # One scale per locked channel, in contract order.
    s_mag: float      # scale for |x0|
    s_re: float       # scale for Re(A^H r)
    s_im: float       # scale for Im(A^H r)
    provenance: str   # "provisional-zero-filled" | "locked-I7"

    def __post_init__(self):
        if self.provenance not in _PROVENANCES:
            logger.error("[scales] provenance must be one of %s, got %r",
                         _PROVENANCES, self.provenance)
            raise ValueError(f"invalid provenance {self.provenance!r}")
        for name, v in (("s_mag", self.s_mag), ("s_re", self.s_re),
                        ("s_im", self.s_im)):
            fv = float(v)
            if not (fv == fv and abs(fv) != float("inf")):
                logger.error("[scales] %s not finite: %r", name, v)
                raise ValueError(f"scale {name} not finite: {v!r}")
            if fv <= _SCALE_MIN:
                logger.error("[scales] %s = %r <= %g violates the locked "
                             "contract (finite and > 1e-8 required)",
                             name, v, _SCALE_MIN)
                raise ValueError(f"scale {name} = {v!r} <= {_SCALE_MIN}")


def _as_complex_state(x0: torch.Tensor) -> torch.Tensor:
    # Accepts complex (..., H, W) or two-channel real (..., 2, H, W).
    if torch.is_complex(x0):
        return x0
    if x0.dim() >= 3 and x0.shape[-3] == 2:
        return torch.complex(x0[..., 0, :, :], x0[..., 1, :, :])
    logger.error("[assemble] x0 must be complex (...,H,W) or two-channel "
                 "(...,2,H,W), got %s dtype %s", tuple(x0.shape), x0.dtype)
    raise ValueError(f"bad x0 shape {tuple(x0.shape)} dtype {x0.dtype}")


def assemble_raw_channels(x0: torch.Tensor, y: torch.Tensor,
                          op: MaskedFourierOperator) -> torch.Tensor:
    # ALWAYS-valid raw assembly: (..., 3, H, W) float32 in the locked order
    # [|x0|, Re(A^H r), Im(A^H r)], r = y - A(x0). No normalization here.
    x0_c = _as_complex_state(x0)
    if not torch.is_complex(y):
        logger.error("[assemble] y must be complex, got %s", y.dtype)
        raise TypeError(f"y must be complex, got {y.dtype}")
    r = y - op.A(x0_c)
    ahr = op.A_adjoint(r)
    raw = torch.stack([x0_c.abs(), ahr.real, ahr.imag], dim=-3).float()
    if not torch.isfinite(raw).all():
        logger.error("[assemble] non-finite raw channels")
        raise ValueError("non-finite raw channels")
    return raw


def normalize_channels(raw: torch.Tensor, scales: ChannelScales
                       ) -> torch.Tensor:
    # Locked contract: channel_c / (scale_c + 1e-8); requires all three
    # declared scales (ChannelScales construction already enforces
    # finite/>1e-8). Undeclared scales are impossible by construction.
    if raw.shape[-3] != 3:
        logger.error("[normalize] expected (...,3,H,W), got %s",
                     tuple(raw.shape))
        raise ValueError(f"expected (...,3,H,W), got {tuple(raw.shape)}")
    if not isinstance(scales, ChannelScales):
        logger.error("[normalize] scales must be ChannelScales, got %r -- "
                     "undeclared scales forbidden (locked 3.8)", type(scales))
        raise TypeError("scales must be a ChannelScales instance")
    s = torch.tensor([scales.s_mag, scales.s_re, scales.s_im],
                     dtype=raw.dtype, device=raw.device)
    out = raw / (s.view(3, 1, 1) + _NORM_EPS)
    if not torch.isfinite(out).all():
        logger.error("[normalize] non-finite normalized channels")
        raise ValueError("non-finite normalized channels")
    return out


def model_channels(x0: torch.Tensor, y: torch.Tensor,
                   op: MaskedFourierOperator, scales: ChannelScales
                   ) -> torch.Tensor:
    # THE model-facing entry point. Provenance-gated: only locked-I7 scales
    # may feed trainer/model construction.
    if not isinstance(scales, ChannelScales):
        logger.error("[model_channels] scales must be ChannelScales, got %r",
                     type(scales))
        raise TypeError("scales must be a ChannelScales instance")
    if scales.provenance != _MODEL_PROVENANCE:
        logger.error("[model_channels] provenance %r REJECTED -- model-facing "
                     "normalization accepts only %r (provisional zero-filled "
                     "stats are diagnostics, never model inputs)",
                     scales.provenance, _MODEL_PROVENANCE)
        raise ValueError(
            f"model_channels requires provenance {_MODEL_PROVENANCE!r}, "
            f"got {scales.provenance!r}")
    return normalize_channels(assemble_raw_channels(x0, y, op), scales)
