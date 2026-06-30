# =============================================================================
# SEQREF-FWDOP v0.1 -- mnist_seqref.src.forward_operator
# Purpose: the forward operator A = Downsample_s o Gauss_blur(sigma) and its
#          EXACT adjoint A^T, for the independent mnist_seqref tree.
# INDEPENDENCE: imports ONLY the local copied degrade (mnist_seqref.src.degrade),
#               never the live/archived CSMF2 package. Copy degrade.py into
#               src/ before using this module.
# CONVENTION: no fallback / mock / pass. Bad input -> logger.error + raise.
# Adjoint note (PORTED from archive — keep, do not re-derive):
#   A is linear, so A^T is its vector-Jacobian product. Computing it via
#   autograd yields the EXACT adjoint of degrade's A (reflect-pad blur +
#   avg-pool), including boundary terms. Verified historically to pass the
#   dot-product test <Ax,r>==<x,A^Tr> to ~1e-15 (float64).
# Changelog (NEW in v0.1):
#   * Introduced for the start-over tree. A_forward / A_adjoint over the local
#     degrade operator; image_hw-aware adjoint.
# Update summary:
#   v0.1 gives the single source of A / A^T for base training, refiner inputs
#   (A^T r), and the L1 adjoint pre-flight (src.adjoint_check).
# =============================================================================
from __future__ import annotations
import logging

import torch

from .degrade import blur, downsample  # local, copied from archive

logger = logging.getLogger("mnist_seqref.forward_operator")
__version__ = "0.1"
__abbr__ = "SEQREF-FWDOP"


def A_forward(x: torch.Tensor, blur_sigma: float, scale: int) -> torch.Tensor:
    # x: (B,1,H,W) pixel space -> y: (B,1,H/scale,W/scale) measurement space.
    if x.dim() != 4 or x.size(1) != 1:
        logger.error("[fwdop] A_forward expects (B,1,H,W), got %s", tuple(x.shape))
        raise ValueError(f"A_forward expects (B,1,H,W), got {tuple(x.shape)}")
    return downsample(blur(x, blur_sigma), scale)


def A_adjoint(r: torch.Tensor, blur_sigma: float, scale: int,
              image_hw: tuple[int, int]) -> torch.Tensor:
    # Exact adjoint via autograd VJP. r: (B,1,M,M) -> (B,1,H,W).
    if r.dim() != 4 or r.size(1) != 1:
        logger.error("[fwdop] A_adjoint expects (B,1,M,M), got %s", tuple(r.shape))
        raise ValueError(f"A_adjoint expects (B,1,M,M), got {tuple(r.shape)}")
    H, W = image_hw
    with torch.enable_grad():
        x = torch.zeros(r.size(0), 1, H, W, dtype=r.dtype, device=r.device,
                        requires_grad=True)
        y = A_forward(x, blur_sigma, scale)
        if y.shape != r.shape:
            logger.error("[fwdop] A(x) shape %s != r shape %s (check scale/hw)",
                         tuple(y.shape), tuple(r.shape))
            raise ValueError("A(x) / r shape mismatch in adjoint")
        (atr,) = torch.autograd.grad((y * r).sum(), x)
    return atr.detach()
