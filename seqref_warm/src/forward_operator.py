# SEQREF-FWDOP v0.1 -- forward_operator
# LIFETIME: KEEP
# A = Downsample_s o Gauss_blur(sigma); Aᵀ = exact autograd VJP of A.
# Imports ONLY local src.degrade (no CSMF2/archive). No fallback/mock/pass.
# Changelog (v0.1):
#   * A_forward / A_adjoint for the independent tree; adjoint via autograd VJP
#     (exact transpose of degrade's reflect-pad blur + avg-pool).
from __future__ import annotations
import logging
import torch
from .degrade import blur, downsample

logger = logging.getLogger("seqref_warm.forward_operator")
__version__ = "0.1"


def A_forward(x: torch.Tensor, blur_sigma: float, scale: int) -> torch.Tensor:
    if x.dim() != 4 or x.size(1) != 1:
        logger.error("[fwdop] A_forward expects (B,1,H,W), got %s", tuple(x.shape))
        raise ValueError(f"A_forward expects (B,1,H,W), got {tuple(x.shape)}")
    return downsample(blur(x, blur_sigma), scale)


def A_adjoint(r: torch.Tensor, blur_sigma: float, scale: int,
              image_hw: tuple[int, int]) -> torch.Tensor:
    if r.dim() != 4 or r.size(1) != 1:
        logger.error("[fwdop] A_adjoint expects (B,1,M,M), got %s", tuple(r.shape))
        raise ValueError(f"A_adjoint expects (B,1,M,M), got {tuple(r.shape)}")
    H, W = image_hw
    with torch.enable_grad():
        x = torch.zeros(r.size(0), 1, H, W, dtype=r.dtype, device=r.device,
                        requires_grad=True)
        y = A_forward(x, blur_sigma, scale)
        if y.shape != r.shape:
            logger.error("[fwdop] A(x) %s != r %s (check scale/hw)",
                         tuple(y.shape), tuple(r.shape))
            raise ValueError("A(x) / r shape mismatch in adjoint")
        (atr,) = torch.autograd.grad((y * r).sum(), x)
    return atr.detach()
