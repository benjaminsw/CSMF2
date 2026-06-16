# =============================================================================
# STEP-1_3 v0.1 -- experiments.step_1_3.scores
# Purpose: pure scoring primitives for the reconstruction-aware gate. No I/O.
#          Per-expert NLL (reuses step_1_2.per_expert_logp), per-expert
#          deterministic-proxy reconstruction residual over a FIXED SHARED
#          z-bank, frozen-calibration standardization, and the hybrid score.
# CONVENTION: non-finite tensors / sigma below floor -> logger.error + raise.
#             No fallback / mock / dummy. Experts frozen by caller.
# rec is a DETERMINISTIC PROXY (z from a fixed shared bank), NOT the true
#          posterior mean -- never reported as true expert reconstruction.
# Score (lower = better expert):
#   NLL_norm = (NLL_k - mu_nll_k) / sigma_nll_k
#   rec_norm = (rec_k - mu_rec_k) / sigma_rec_k
#   score_k  = alpha*NLL_norm + beta*rec_norm
# Changelog (v0.1 -> v0.2):
#   * calibration_stats(rec_norm=): NLL always per-expert standardized; rec is
#     'global' (pooled shared z-score, default) or 'per_expert' (debug). Fixes
#     the v0.2-run signal inversion where per-expert rec routed away from the
#     best absolute reconstructor.
# Changelog (NEW in v0.1):
#   * Introduced. make_z_bank, per_expert_nll, per_expert_rec,
#     per_expert_recon_pixels, calibration_stats, standardize, hybrid_score.
# Update summary:
#   v0.2 makes reconstruction normalization shared-scale so lower absolute
#   residual = better expert. The same seeded z-bank is fed to every expert so
#   the rec comparison isolates the expert, not the latent.
# =============================================================================
from __future__ import annotations
import logging

import torch

from ...data.degrade import inverse_logit, blur, downsample
from ..step_1_2.mixture import per_expert_logp

logger = logging.getLogger(__name__)
__version__ = "0.2"
__abbr__ = "STEP-1_3"

_IMAGE_HW = (28, 28)


def make_z_bank(dim: int, size: int, mode: str, seed: int, device, dtype):
    """Fixed shared z-bank (S, dim). mode='zero' -> a single zero latent."""
    if mode == "zero":
        return torch.zeros(1, dim, device=device, dtype=dtype)
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(size, dim, generator=g, device=device, dtype=dtype)


@torch.no_grad()
def per_expert_nll(experts, x_flat, y, ldj_deq) -> torch.Tensor:
    """NLL_{k} = -(log p_k(x|y) + ldj_deq). Returns (B, K)."""
    lp_ke = per_expert_logp(experts, x_flat, y)            # (B,K), raises if bad
    nll = -(lp_ke + ldj_deq.unsqueeze(1))
    if not torch.isfinite(nll).all():
        logger.error("[scores] non-finite per-expert NLL")
        raise RuntimeError("non-finite per-expert NLL")
    return nll


@torch.no_grad()
def _A(x_logit, blur_sigma, scale, n):
    x_pix = inverse_logit(x_logit).view(n, 1, *_IMAGE_HW)
    return downsample(blur(x_pix, blur_sigma), scale)


@torch.no_grad()
def per_expert_rec(experts, y, z_bank, *, blur_sigma, scale) -> torch.Tensor:
    """Deterministic-proxy rec residual, mean over the shared z-bank.
    rec_{k} = mean_s || A(decode(z_s, h_k(y))) - y ||^2. Returns (B, K)."""
    B = y.size(0)
    S = z_bank.size(0)
    K = len(experts)
    out = y.new_zeros(B, K)
    for k, ex in enumerate(experts):
        h = ex.cond(y)                                     # (B, h_dim)
        acc = y.new_zeros(B)
        for s in range(S):
            z = z_bank[s:s + 1].expand(B, -1)              # (B, dim) shared z_s
            x_logit = ex.decode(z, h)
            if not torch.isfinite(x_logit).all():
                logger.error("[scores] non-finite decode expert %d z %d", k, s)
                raise RuntimeError("non-finite decode in per_expert_rec")
            Ax = _A(x_logit, blur_sigma, scale, B)
            acc = acc + (Ax - y).flatten(1).pow(2).sum(dim=1)
        out[:, k] = acc / S
    return out


@torch.no_grad()
def per_expert_recon_pixels(experts, y, z_bank) -> torch.Tensor:
    """Mean pixel reconstruction per expert over the shared z-bank, for eval.
    Returns (B, K, 1, 28, 28) in [0,1]."""
    B = y.size(0); S = z_bank.size(0); K = len(experts)
    out = y.new_zeros(B, K, 1, *_IMAGE_HW)
    for k, ex in enumerate(experts):
        h = ex.cond(y)
        acc = y.new_zeros(B, 1, *_IMAGE_HW)
        for s in range(S):
            z = z_bank[s:s + 1].expand(B, -1)
            x_pix = inverse_logit(ex.decode(z, h)).view(B, 1, *_IMAGE_HW)
            acc = acc + x_pix
        out[:, k] = acc / S
    return out


def calibration_stats(nll_ke, rec_ke, *, min_sigma: float,
                      rec_norm: str = "global") -> dict:
    """Frozen calibration stats. NLL is ALWAYS per-expert standardized (raw
    NLL scales differ across flows). rec is 'global' (one shared mu/sigma
    pooled over all experts+samples -- residual is in shared y-space units)
    or 'per_expert' (debug only). sigma < floor -> raise."""
    def _check(sigma, name):
        bad = (sigma < min_sigma)
        if bool(bad.any()):
            idx = torch.nonzero(bad).flatten().tolist()
            logger.error("[scores] %s sigma below floor %.1e at %s (values %s)",
                         name, min_sigma, idx, sigma[bad].tolist())
            raise RuntimeError(f"{name} sigma < min_sigma at {idx}")

    K = nll_ke.size(1)
    mu_nll = nll_ke.mean(dim=0)
    sig_nll = nll_ke.std(dim=0, unbiased=False)
    _check(sig_nll, "NLL")

    if rec_norm == "global":
        pooled = rec_ke.reshape(-1)
        mu_g = pooled.mean()
        sig_g = pooled.std(unbiased=False)
        mu_rec = mu_g.expand(K).clone()           # shared across experts
        sig_rec = sig_g.expand(K).clone()
    elif rec_norm == "per_expert":
        mu_rec = rec_ke.mean(dim=0)
        sig_rec = rec_ke.std(dim=0, unbiased=False)
    else:
        logger.error("[scores] unknown rec_norm %r", rec_norm)
        raise ValueError(f"unknown rec_norm {rec_norm!r}")
    _check(sig_rec, "rec")

    return {"mu_nll": mu_nll, "sigma_nll": sig_nll,
            "mu_rec": mu_rec, "sigma_rec": sig_rec,
            "rec_norm": rec_norm}


def standardize(t, mu, sigma):
    return (t - mu) / sigma


def hybrid_score(nll_ke, rec_ke, stats, *, alpha, beta):
    """score_k = alpha*NLL_norm + beta*rec_norm (lower=better). Returns
    (score (B,K), nll_norm (B,K), rec_norm (B,K))."""
    nll_norm = standardize(nll_ke, stats["mu_nll"], stats["sigma_nll"])
    rec_norm = standardize(rec_ke, stats["mu_rec"], stats["sigma_rec"])
    score = alpha * nll_norm + beta * rec_norm
    if not torch.isfinite(score).all():
        logger.error("[scores] non-finite hybrid score")
        raise RuntimeError("non-finite hybrid score")
    return score, nll_norm, rec_norm
