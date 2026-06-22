# =============================================================================
# STEP-1_4B v0.1 -- experiments.step_1_4b.consistency_loss  (RNVP-CONSIST v0.1)
# Purpose: weak reconstruction-consistency term for the 3f fine-tune:
#            consist = mean_b || A(x_hat_b) - y_b ||^2   (mean MSE over batch+pixels)
#          x_hat is the DETERMINISTIC conditional-mean reconstruction
#            x_hat_logit = CBExpert.decode(eps=0, h)   # eps=0 -> w = mu(h)
#          and A is the SAME forward operator RECGATE scores with:
#            A(x_logit) = downsample(blur(inverse_logit(x_logit), blur_sigma), scale)
#          so the fine-tune optimizes exactly the quantity the verdict measures.
# CONVENTION: non-finite consistency -> logger.error + raise. No fallback/mock.
#   This is the REAL degradation operator (reused from data.degrade), not a
#   surrogate. mean (not sum) MSE so beta has a stable scale across the grid.
# Changelog (v0.1 -> v0.2, RNVP-CONSIST v0.4):
#   * Noise/scale-normalized: mean-over-pixels (was sum) and divide by
#     sigma_eff^2 = max(noise_sigma, sigma_floor)^2 -- noise-relative units, so
#     the term is comparable across scale/noise and is the proper Gaussian
#     measurement-likelihood scaling. New required kwarg noise_sigma; sigma_floor
#     defaults 0.05 (guards /0 at noise_sigma=0).
# Changelog (NEW in v0.1):
#   * Introduced. consistency_term(model, h, y, blur_sigma, scale) + the
#     mu-mean x_hat helper. Mirrors step_1_3.scores._A exactly.
# Update summary:
#   v0.1 is the only new loss math for 3f; everything else reuses the CB trainer.
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.2"
__abbr__ = "STEP-1_4B"

import torch

from ...data.degrade import inverse_logit, blur, downsample

_IMAGE_HW = (28, 28)


def _A(x_logit: torch.Tensor, blur_sigma: float, scale: int,
       n: int) -> torch.Tensor:
    """Forward operator, IDENTICAL to step_1_3.scores._A (must match RECGATE)."""
    x_pix = inverse_logit(x_logit).view(n, 1, *_IMAGE_HW)
    return downsample(blur(x_pix, blur_sigma), scale)


def mu_mean_xhat_logit(model, h: torch.Tensor) -> torch.Tensor:
    """Deterministic conditional-mean reconstruction (logit space):
    eps=0 -> CBExpert.decode maps w = mu(h) + sigma(h)*0 = mu(h) -> x_logit.
    Kept differentiable (no no_grad) so consistency backprops into the flow,
    base, and conditioner."""
    eps = torch.zeros(h.size(0), model.dim, device=h.device, dtype=h.dtype)
    return model.decode(eps, h)


def consistency_term(model, h: torch.Tensor, y: torch.Tensor, *,
                     blur_sigma: float, scale: int,
                     noise_sigma: float, sigma_floor: float = 0.05
                     ) -> torch.Tensor:
    """Noise/scale-normalized consistency (RNVP-CONSIST v0.4, Option 1):
        consist = mean_pixels( (A(x_hat) - y)^2 ) / sigma_eff^2,
        sigma_eff = max(noise_sigma, sigma_floor)   # avoid /0 at noise_sigma=0
    Mean-over-pixels (not sum) + division by the noise variance expresses the
    residual in NOISE-RELATIVE units -- the Gaussian measurement-likelihood
    scaling -- so the term means the same thing across scale/noise settings and
    the diagnostics/plots are comparable. Differentiable; x_hat = decode(mu(h),h).
    """
    n = y.size(0)
    sigma_eff = max(float(noise_sigma), float(sigma_floor))
    x_hat_logit = mu_mean_xhat_logit(model, h)
    Ax = _A(x_hat_logit, blur_sigma, scale, n)
    if Ax.shape != y.shape:
        logger.error("[consistency] A(x_hat) shape %s != y shape %s",
                     tuple(Ax.shape), tuple(y.shape))
        raise ValueError("A(x_hat)/y shape mismatch")
    # mean over pixels per sample, then mean over batch, then /sigma_eff^2
    per_sample = (Ax - y).flatten(1).pow(2).mean(dim=1)   # (B,) mean over pixels
    consist = per_sample.mean() / (sigma_eff ** 2)        # noise-relative units
    if not torch.isfinite(consist).all():
        logger.error("[consistency] non-finite consistency term")
        raise RuntimeError("non-finite consistency term")
    return consist
