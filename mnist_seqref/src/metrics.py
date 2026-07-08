# SEQREF-METRIC v0.1 -- metrics
# LIFETIME: KEEP
# Pixel-space reconstruction metrics: mse, psnr, fwd_rel (||A x_hat - y|| / ||y||).
# No fallback/mock/pass. Failures logger.error + raise.
# Changelog (v0.1):
#   * mse/psnr (pixel [0,1]) + fwd_rel via forward_operator.A_forward.
from __future__ import annotations
import logging
import torch
from .forward_operator import A_forward

logger = logging.getLogger("mnist_seqref.metrics")
__version__ = "0.1"
_EPS = 1e-12


def mse(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    if x_hat.shape != x_true.shape:
        logger.error("[metrics.mse] shape %s != %s", tuple(x_hat.shape), tuple(x_true.shape))
        raise ValueError("mse shape mismatch")
    return float(((x_hat - x_true) ** 2).mean())


def psnr(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    # x in [0,1] => MAX_I = 1.
    m = mse(x_hat, x_true)
    if m <= 0.0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(1.0 / m)))


def fwd_rel(x_hat: torch.Tensor, y: torch.Tensor, blur_sigma: float,
            scale: int) -> float:
    # ||A x_hat - y|| / ||y||. x_hat: (B,1,H,W) pixel space.
    if x_hat.dim() != 4 or x_hat.size(1) != 1:
        logger.error("[metrics.fwd_rel] expected x_hat (B,1,H,W), got %s",
                     tuple(x_hat.shape))
        raise ValueError("fwd_rel expects (B,1,H,W)")
    ax = A_forward(x_hat, blur_sigma, scale)
    if ax.shape != y.shape:
        logger.error("[metrics.fwd_rel] A(x_hat) %s != y %s", tuple(ax.shape), tuple(y.shape))
        raise ValueError("fwd_rel A(x_hat)/y shape mismatch")
    num = torch.linalg.vector_norm(ax - y)
    den = torch.linalg.vector_norm(y) + _EPS
    return float(num / den)
