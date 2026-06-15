# =============================================================================
# STEP-1_1_1_1 v0.1 -- experiments.step_1_1_1_1.map_core
# Purpose: pure latent-refinement primitives, no I/O. Importance-sampling
#          candidate generation + selection, the MAP objective, and the MAP
#          optimiser loop. Kept side-effect-free so they are unit-testable.
# CONVENTION: non-finite tensors / bad shapes -> logger.error + raise.
#             No fallback / mock / dummy. The flow is frozen by the caller;
#             only z carries gradients.
# Objective (per sample): ||A(decode(z,h)) - y||^2 + lambda_prior * ||z||^2,
#          with A = downsample o blur applied in PIXEL space (inverse_logit).
# Changelog (NEW in v0.1):
#   * Introduced. generate_candidates, select_best (argmin residual),
#     map_objective, run_map (Adam on z, objective curve, converged flag).
# Update summary:
#   v0.1 implements the three-arm core: random_map = run_map from N(0,I);
#   is_only = select_best with 0 steps; is_map = run_map from select_best.
#   Residual is reported both absolute (data term) and relative (||.||/||y||).
# =============================================================================
from __future__ import annotations
import logging

import torch

from ...data.degrade import inverse_logit, blur, downsample

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_1_1_1"

_IMAGE_HW = (28, 28)


def _forward_A(x_logit: torch.Tensor, blur_sigma: float, scale: int,
               n: int) -> torch.Tensor:
    """A = downsample o blur, applied in pixel space. x_logit: (n, dim)."""
    x_pix = inverse_logit(x_logit).view(n, 1, *_IMAGE_HW)
    return downsample(blur(x_pix, blur_sigma), scale)


def map_objective(z, h, expert, y, *, blur_sigma, scale, lambda_prior):
    """Return (total, data, prior, Ax) with per-sample data/prior/total.
    z:(n,dim) h:(n,h_dim) y:(n,1,hd,wd). decode is differentiable wrt z."""
    n = z.size(0)
    x_logit = expert.decode(z, h)
    if not torch.isfinite(x_logit).all():
        logger.error("[map_core] non-finite decode in map_objective")
        raise RuntimeError("non-finite decode(z,h)")
    Ax = _forward_A(x_logit, blur_sigma, scale, n)
    data = (Ax - y).flatten(1).pow(2).sum(dim=1)        # (n,)
    prior = z.flatten(1).pow(2).sum(dim=1)              # (n,)
    total = data + lambda_prior * prior
    return total, data, prior, Ax


@torch.no_grad()
def generate_candidates(expert, cond, y, *, K, blur_sigma, scale, generator):
    """Per image, draw K latents ~ N(0,I), decode, score by data residual.
    Returns (z_cand (n,K,dim), resid (n,K), h (n,h_dim))."""
    n = y.size(0)
    h = cond(y)                                          # (n, h_dim)
    dim = int(expert.dim)
    device, dtype = y.device, h.dtype
    z_cand = torch.randn(n, K, dim, generator=generator,
                         device=device, dtype=dtype)
    # flatten (n*K, dim); repeat h per candidate
    z_flat = z_cand.reshape(n * K, dim)
    h_rep = h.unsqueeze(1).expand(n, K, h.size(1)).reshape(n * K, h.size(1))
    x_logit = expert.decode(z_flat, h_rep)
    if not torch.isfinite(x_logit).all():
        logger.error("[map_core] non-finite decode in generate_candidates")
        raise RuntimeError("non-finite candidate decode")
    Ax = _forward_A(x_logit, blur_sigma, scale, n * K)
    y_rep = y.unsqueeze(1).expand(n, K, *y.shape[1:]).reshape(n * K, *y.shape[1:])
    resid = (Ax - y_rep).flatten(1).pow(2).sum(dim=1).reshape(n, K)
    return z_cand, resid, h


def select_best(z_cand, resid):
    """argmin residual per image. Returns (z0 (n,dim), best_resid (n,),
    mean_resid (n,), rank_gap (n,))."""
    idx = resid.argmin(dim=1)                            # (n,)
    n = z_cand.size(0)
    z0 = z_cand[torch.arange(n, device=z_cand.device), idx]
    best = resid.gather(1, idx.unsqueeze(1)).squeeze(1)
    mean = resid.mean(dim=1)
    # gap between best and 2nd-best (how rare the winner is)
    if resid.size(1) >= 2:
        srt, _ = resid.sort(dim=1)
        gap = srt[:, 1] - srt[:, 0]
    else:
        gap = torch.zeros_like(best)
    return z0, best, mean, gap


def run_map(z0, h, expert, y, *, blur_sigma, scale, lambda_prior,
            steps, lr_z, conv_tol, log_every=10):
    """Adam on z starting from z0 (cloned). Flow frozen. Returns dict with
    final z, per-step mean objective curve, grad norms, converged flag."""
    z = z0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr_z)
    curve, grad_norms = [], []
    prev = None
    converged = False
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        total, data, prior, _ = map_objective(
            z, h, expert, y, blur_sigma=blur_sigma, scale=scale,
            lambda_prior=lambda_prior)
        loss = total.sum()                               # per-row grads are independent
        loss.backward()
        gnorm = float(z.grad.flatten(1).norm(dim=1).mean())
        grad_norms.append(gnorm)
        opt.step()
        cur = float(total.mean())
        curve.append(cur)
        if prev is not None and abs(prev - cur) <= conv_tol * max(abs(prev), 1e-12):
            converged = True
            if (t % log_every) == 0:
                logger.info("[map_core] converged at step %d (obj=%.4f)", t, cur)
            break
        prev = cur
        if (t % log_every) == 0:
            logger.info("[map_core] step %d obj=%.4f grad=%.3e", t, cur, gnorm)
    return {"z": z.detach(), "objective_curve": curve,
            "grad_norms": grad_norms, "n_steps": len(curve),
            "converged": bool(converged)}
