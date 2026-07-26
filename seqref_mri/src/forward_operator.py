# =============================================================================
# SEQREF-I1 v0.2 -- src.forward_operator
# LIFETIME: KEEP
# Purpose: masked-Fourier forward operator for the seqref_mri campaign.
#   A = M o F,  Aᴴ = Fᴴ o M  on the complex 96x96 reconstruction state
#   (EXEC 3.14), centred orthonormal FFT pair (EXEC 3.2 convention), mask on
#   the LAST axis columns (EXEC 3.7). Includes the LOCKED 3.11 normalized
#   k-space consistency metric.
# CONVENTION: logger.error + raise on every failure path. No fallback.
# REPLACES the MNIST blur/downsample operator (S1 ledger: REBUILD). The old
#   file's degrade imports are gone; nothing here touches degrade.py.
# Changelog (v0.1 -> v0.2, pre-deployment review fix):
#   * Mask contract RESTRICTED to shape (W,) for I1 -- batched (B,W) masks
#     would broadcast incorrectly against (B,H,W) without a row dim;
#     normalisation to (...,1,W) is deferred until a stage needs batched
#     masks. dim!=1 now raises.
# Changelog (NEW in v0.1):
#   * Introduced MaskedFourierOperator (A, A_adjoint, consistency) and
#     complex_state helpers (two-channel <-> complex).
# Update summary: pure implementation of locked conventions; operator is an
#   object (S1 base_io note) so downstream code carries no blur/scale params.
# =============================================================================
from __future__ import annotations

import logging

import torch

from .fastmri_data import fft2c, ifft2c, CELL_HW

logger = logging.getLogger("seqref_mri.forward_operator")

__version__ = "0.2"
__abbr__ = "SEQREF-I1"

_CONSISTENCY_EPS = 1e-12          # EXEC 3.11, locked


def two_channel_to_complex(x: torch.Tensor) -> torch.Tensor:
    # (..., 2, H, W) float -> (..., H, W) complex
    if x.shape[-3] != 2:
        logger.error("[state] expected channel dim 2, got shape %s",
                     tuple(x.shape))
        raise ValueError(f"expected (...,2,H,W), got {tuple(x.shape)}")
    return torch.complex(x[..., 0, :, :], x[..., 1, :, :])


def complex_to_two_channel(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(x):
        logger.error("[state] expected complex tensor, got %s", x.dtype)
        raise TypeError(f"expected complex tensor, got {x.dtype}")
    return torch.stack([x.real, x.imag], dim=-3)


class MaskedFourierOperator:
    # mask: bool (W,) over columns -- 1-D ONLY for I1 (v0.2 contract).
    def __init__(self, mask: torch.Tensor, *, image_hw: int = CELL_HW):
        if mask.dtype != torch.bool:
            logger.error("[op] mask dtype must be bool, got %s", mask.dtype)
            raise TypeError(f"mask dtype must be bool, got {mask.dtype}")
        if mask.dim() != 1:
            logger.error("[op] mask must be 1-D (W,), got shape %s",
                         tuple(mask.shape))
            raise ValueError(f"mask must be 1-D (W,), got {tuple(mask.shape)}")
        if mask.shape[-1] != image_hw:
            logger.error("[op] mask width %d != image_hw %d",
                         mask.shape[-1], image_hw)
            raise ValueError(
                f"mask width {mask.shape[-1]} != image_hw {image_hw}")
        self.image_hw = image_hw
        self.mask = mask

    def _check_state(self, x: torch.Tensor, name: str) -> None:
        if not torch.is_complex(x):
            logger.error("[op] %s must be complex, got %s", name, x.dtype)
            raise TypeError(f"{name} must be complex, got {x.dtype}")
        if x.shape[-1] != self.image_hw or x.shape[-2] != self.image_hw:
            logger.error("[op] %s spatial shape %s != %dx%d", name,
                         tuple(x.shape), self.image_hw, self.image_hw)
            raise ValueError(f"{name} shape {tuple(x.shape)} != cell")

    def _m(self, k: torch.Tensor) -> torch.Tensor:
        return k * self.mask.to(device=k.device, dtype=k.dtype)

    def A(self, x: torch.Tensor) -> torch.Tensor:
        # complex image state -> masked k-space
        self._check_state(x, "x")
        return self._m(fft2c(x))

    def A_adjoint(self, r: torch.Tensor) -> torch.Tensor:
        # masked k-space residual -> complex image
        self._check_state(r, "r")
        return ifft2c(self._m(r))

    def consistency(self, x_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # EXEC 3.11 (LOCKED): ||M(F x_hat) - y||_2 / max(||y||_2, eps),
        # complex L2, per batch element if batched.
        self._check_state(x_hat, "x_hat")
        self._check_state(y, "y")
        num = torch.linalg.vector_norm(self.A(x_hat) - y, dim=(-2, -1))
        den = torch.linalg.vector_norm(y, dim=(-2, -1))
        out = num / torch.clamp(den, min=_CONSISTENCY_EPS)
        if not torch.isfinite(out).all():
            logger.error("[op] non-finite consistency value")
            raise ValueError("non-finite consistency")
        return out
