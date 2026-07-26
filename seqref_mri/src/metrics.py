# SEQREF-METRIC v0.3 -- metrics
# LIFETIME: KEEP
# Magnitude-space reconstruction metrics for the MRI cell: mse, psnr, ssim,
# each with REQUIRED explicit data_range (no silent [0,1]/L=1 assumption)
# and per-sample variants. No fallback/mock/pass; failures logger.error +
# raise.
# Changelog (v0.2 -> v0.3, SEQREF-I2):
#   * psnr/ssim take REQUIRED keyword data_range (finite, > 0, validated);
#     the implicit MAX_I = 1 / L = 1 pixel assumption is removed. The MRI
#     data-range CONVENTION is provisional at I2 (HDF5 file-attr `max`,
#     label provisional-I2-file-attr-max) and is verified+locked at the
#     section-6 metric-sanity check.
#   * Semantics defined: mse/psnr/ssim return the BATCH-REDUCED float
#     (mean over samples for psnr/ssim); *_per_sample return (B,) tensors.
#   * Inputs validated as magnitudes: finite and non-negative required.
#   * fwd_rel REMOVED (S1 REBUILD verdict): superseded by the locked 3.11
#     k-space consistency in forward_operator.MaskedFourierOperator
#     .consistency(); the old blur/scale A_forward import no longer exists.
# Changelog (v0.1 -> v0.2, SEQREF-REFINE): added pure-torch SSIM (11x11
#   Gaussian window, sigma 1.5). Mechanics INHERIT (S1) -- unchanged here.
# Changelog (v0.1): mse/psnr + fwd_rel (now removed).
# Update summary:
#   v0.3 makes the data range an explicit, validated parameter with a
#   recorded provisional convention, defines reduction semantics, and
#   removes the MNIST-operator dependency.
from __future__ import annotations
import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger("seqref_mri.metrics")
__version__ = "0.3"
__abbr__ = "SEQREF-METRIC"
_EPS = 1e-12


def _validate_pair(x_hat, x_true, data_range, fn):
    if x_hat.dim() != 4 or x_hat.size(1) != 1 or x_hat.shape != x_true.shape:
        logger.error("[metrics.%s] expected matching (B,1,H,W), got %s vs %s",
                     fn, tuple(x_hat.shape), tuple(x_true.shape))
        raise ValueError(f"{fn} expects matching (B,1,H,W)")
    for name, t in (("x_hat", x_hat), ("x_true", x_true)):
        if not torch.isfinite(t).all():
            logger.error("[metrics.%s] non-finite %s", fn, name)
            raise ValueError(f"{fn}: non-finite {name}")
        if t.min().item() < 0.0:
            logger.error("[metrics.%s] %s has negative values -- magnitude "
                         "inputs required", fn, name)
            raise ValueError(f"{fn}: {name} not a magnitude (negative values)")
    dr = float(data_range)
    if not (dr == dr and dr not in (float("inf"), float("-inf"))) or dr <= 0.0:
        logger.error("[metrics.%s] data_range must be finite and > 0, got %r",
                     fn, data_range)
        raise ValueError(f"{fn}: data_range must be finite and > 0")
    return dr


def mse_per_sample(x_hat: torch.Tensor, x_true: torch.Tensor
                   ) -> torch.Tensor:
    # (B,) per-sample MSE. Shape/finite validation only (mse is range-free).
    if x_hat.shape != x_true.shape:
        logger.error("[metrics.mse] shape %s != %s", tuple(x_hat.shape),
                     tuple(x_true.shape))
        raise ValueError("mse shape mismatch")
    for name, t in (("x_hat", x_hat), ("x_true", x_true)):
        if not torch.isfinite(t).all():
            logger.error("[metrics.mse] non-finite %s", name)
            raise ValueError(f"mse: non-finite {name}")
    return ((x_hat - x_true) ** 2).flatten(1).mean(dim=1)


def mse(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    # Batch-reduced (mean over samples).
    return float(mse_per_sample(x_hat, x_true).mean())


def psnr_per_sample(x_hat: torch.Tensor, x_true: torch.Tensor, *,
                    data_range: float) -> torch.Tensor:
    # (B,) per-sample PSNR with REQUIRED explicit data_range.
    dr = _validate_pair(x_hat, x_true, data_range, "psnr")
    m = mse_per_sample(x_hat, x_true).clamp_min(_EPS)
    return 10.0 * torch.log10(torch.tensor(dr, dtype=m.dtype,
                                           device=m.device) ** 2 / m)


def psnr(x_hat: torch.Tensor, x_true: torch.Tensor, *,
         data_range: float) -> float:
    # Batch-reduced (mean of per-sample PSNR).
    return float(psnr_per_sample(x_hat, x_true, data_range=data_range).mean())


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


def ssim_per_sample(x_hat: torch.Tensor, x_true: torch.Tensor, *,
                    data_range: float) -> torch.Tensor:
    # (B,) per-sample mean-SSIM with REQUIRED explicit data_range:
    # C1=(K1*L)^2, C2=(K2*L)^2, L=data_range. Mechanics unchanged (INHERIT).
    L = _validate_pair(x_hat, x_true, data_range, "ssim")
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
    C1 = (_SSIM_K1 * L) ** 2
    C2 = (_SSIM_K2 * L) ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2))
    vals = ssim_map.flatten(1).mean(dim=1)
    if not torch.isfinite(vals).all():
        logger.error("[metrics.ssim] non-finite SSIM")
        raise ValueError("non-finite SSIM")
    return vals


def ssim(x_hat: torch.Tensor, x_true: torch.Tensor, *,
         data_range: float) -> float:
    # Batch-reduced (mean of per-sample SSIM).
    return float(ssim_per_sample(x_hat, x_true,
                                 data_range=data_range).mean())
