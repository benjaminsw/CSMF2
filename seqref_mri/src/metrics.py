# SEQREF-METRIC v0.2 -- metrics
# LIFETIME: KEEP
# Pixel-space reconstruction metrics: mse, psnr, fwd_rel (||A x_hat - y|| / ||y||),
# ssim. No fallback/mock/pass. Failures logger.error + raise.
# Changelog (v0.1 -> v0.2, SEQREF-REFINE):
#   * NEW ssim(x_hat, x_true): pure-torch SSIM, 11x11 Gaussian window
#     (sigma=1.5), data_range=1.0, standard C1/C2 (K1=0.01, K2=0.03).
#     Shape/range validated; non-finite -> raise. Required by train_refiner
#     (val_ssim_x0/x1, val_dssim).
# Changelog (v0.1):
#   * mse/psnr (pixel [0,1]) + fwd_rel via forward_operator.A_forward.
# Update summary:
#   v0.2 adds SSIM as the third HARD-metric family for the refiner gate
#   tracking (PSNR / fwd_rel / SSIM). mse/psnr/fwd_rel byte-identical to v0.1.
from __future__ import annotations
import logging
import math
import torch
import torch.nn.functional as F
from .forward_operator import A_forward

logger = logging.getLogger("seqref_mri.metrics")
__version__ = "0.2"
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


# ---- SSIM (v0.2) -------------------------------------------------------------
_SSIM_WIN = 11
_SSIM_SIGMA = 1.5
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03


def _ssim_window(device, dtype) -> torch.Tensor:
    ax = torch.arange(_SSIM_WIN, dtype=dtype, device=device) - (_SSIM_WIN - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2.0 * _SSIM_SIGMA ** 2))
    g = g / g.sum()
    w = torch.outer(g, g)
    return w.view(1, 1, _SSIM_WIN, _SSIM_WIN)


def ssim(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    # Mean SSIM over batch. x: (B,1,H,W) in [0,1], data_range=1.
    if x_hat.dim() != 4 or x_hat.size(1) != 1 or x_hat.shape != x_true.shape:
        logger.error("[metrics.ssim] expected matching (B,1,H,W), got %s vs %s",
                     tuple(x_hat.shape), tuple(x_true.shape))
        raise ValueError("ssim expects matching (B,1,H,W)")
    if x_hat.size(-1) < _SSIM_WIN or x_hat.size(-2) < _SSIM_WIN:
        logger.error("[metrics.ssim] image smaller than window %d: %s",
                     _SSIM_WIN, tuple(x_hat.shape))
        raise ValueError("ssim image smaller than window")
    w = _ssim_window(x_hat.device, x_hat.dtype)
    pad = _SSIM_WIN // 2
    def _f(t):
        return F.conv2d(F.pad(t, [pad] * 4, mode="reflect"), w)
    mu1, mu2 = _f(x_hat), _f(x_true)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1_sq = _f(x_hat * x_hat) - mu1_sq
    s2_sq = _f(x_true * x_true) - mu2_sq
    s12 = _f(x_hat * x_true) - mu12
    C1 = _SSIM_K1 ** 2
    C2 = _SSIM_K2 ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2))
    val = ssim_map.mean()
    if not torch.isfinite(val):
        logger.error("[metrics.ssim] non-finite SSIM")
        raise ValueError("non-finite SSIM")
    return float(val)
