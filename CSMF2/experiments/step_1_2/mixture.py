# =============================================================================
# STEP-1_2 v0.1 -- experiments.step_1_2.mixture
# Purpose: pure mixture math, gate modules, and gate-health metrics. No I/O.
#          Per-expert log-prob is expert.log_prob(x_flat, y) in logit space;
#          the dequantization Jacobian ldj_deq is SHARED across experts and is
#          added OUTSIDE the logsumexp as a common per-sample constant.
# CONVENTION: non-finite logp / weight-sum error -> logger.error + raise.
#             No fallback / mock / dummy. Experts are frozen by the caller.
# Math:
#   lp_k(x|y) = expert_k.log_prob(x_flat, y)                        # (B,)
#   log p(x|y)= logsumexp_k[ log w_k(y) + lp_k ]  +  ldj_deq        # (B,)
#   NLL       = -mean( log p(x|y) )
# Changelog (NEW in v0.1):
#   * Introduced. per_expert_logp, mixture_logp, UniformGate,
#     LearnedGlobalGate, gate_metrics (Neff/entropy/usage/argmax/safety).
# Update summary:
#   v0.1 implements MIX-SKEL v0.2 core math. Gate weights are GLOBAL (one
#   K-vector per image). Numerical safety (finite logp, weights sum to 1) are
#   hard asserts here, never soft-logged.
# =============================================================================
from __future__ import annotations
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_2"


def per_expert_logp(experts, x_flat, y) -> torch.Tensor:
    """Stack lp_k = expert_k.log_prob(x_flat, y) -> (B, K) in logit space."""
    cols = []
    for k, ex in enumerate(experts):
        lp = ex.log_prob(x_flat, y)
        if lp.dim() != 1 or lp.size(0) != x_flat.size(0):
            logger.error("[mixture] expert %d log_prob shape %s != (B,)",
                         k, tuple(lp.shape))
            raise ValueError(f"expert {k} log_prob bad shape {tuple(lp.shape)}")
        cols.append(lp)
    lp_ke = torch.stack(cols, dim=1)                       # (B, K)
    if not torch.isfinite(lp_ke).all():
        logger.error("[mixture] non-finite per-expert log_prob")
        raise RuntimeError("non-finite per-expert log_prob")
    return lp_ke


def mixture_logp(lp_ke, log_w, ldj_deq) -> torch.Tensor:
    """log p(x|y) = logsumexp_k(log_w + lp_ke) + ldj_deq. log_w broadcasts
    from (K,) [uniform] or (B,K) [learned]. Returns (B,)."""
    combined = log_w + lp_ke                               # (B,K)
    lp = torch.logsumexp(combined, dim=1) + ldj_deq        # (B,)
    if not torch.isfinite(lp).all():
        logger.error("[mixture] non-finite mixture log-prob")
        raise RuntimeError("non-finite mixture log-prob")
    return lp


class UniformGate(nn.Module):
    """Fixed w_k = 1/K. Trains nothing."""
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def log_weights(self, y) -> torch.Tensor:
        b = y.size(0)
        lw = torch.full((b, self.k), -torch.log(torch.tensor(float(self.k))),
                        device=y.device, dtype=y.dtype)
        return lw


class LearnedGlobalGate(nn.Module):
    """Global gate: one K-vector per image from y only (independent of any
    expert's conditioner). log_weights = log_softmax(MLP(y_flat)/tau)."""
    def __init__(self, y_in: int, k: int, hidden: int, tau: float):
        super().__init__()
        if tau <= 0.0:
            logger.error("[LearnedGlobalGate] tau must be > 0, got %s", tau)
            raise ValueError("tau must be > 0")
        self.tau = float(tau)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(y_in, hidden), nn.GELU(),
            nn.Linear(hidden, k))

    def log_weights(self, y) -> torch.Tensor:
        logits = self.net(y) / self.tau
        return F.log_softmax(logits, dim=1)                # (B, K)


@torch.no_grad()
def gate_metrics(log_w, lp_ke, *, weight_sum_tol: float) -> dict:
    """Gate-health + numerical-safety. Weight-sum violation -> raise."""
    w = log_w.exp()                                        # (B,K)
    sums = w.sum(dim=1)
    max_abs_err = float((sums - 1.0).abs().max())
    if max_abs_err > weight_sum_tol:
        logger.error("[mixture] weights do not sum to 1: max|sum-1|=%.3e "
                     "> tol %.1e", max_abs_err, weight_sum_tol)
        raise RuntimeError(f"gate weights sum error {max_abs_err:.3e}")
    neff = 1.0 / w.pow(2).sum(dim=1).clamp_min(1e-12)      # (B,)
    ent = -(w * w.clamp_min(1e-12).log()).sum(dim=1)       # (B,)
    argmax_counts = torch.bincount(w.argmax(dim=1),
                                   minlength=w.size(1)).tolist()
    best_expert = torch.bincount(lp_ke.argmax(dim=1),
                                 minlength=lp_ke.size(1)).tolist()
    return {
        "Neff_mean": float(neff.mean()), "Neff_min": float(neff.min()),
        "Neff_max": float(neff.max()),
        "gate_entropy": float(ent.mean()),
        "mean_weight_per_expert": w.mean(dim=0).tolist(),
        "weight_std_per_expert": w.std(dim=0).tolist(),
        "expert_usage_argmax_counts": argmax_counts,
        "best_expert_frequency": best_expert,
        "max_abs_weight_sum_minus_1": max_abs_err,
    }


@torch.no_grad()
def per_expert_nll(lp_ke, ldj_deq) -> dict:
    """Per-expert NLL = -(lp_k + ldj_deq). Returns means/std + logp gap."""
    nll_ke = -(lp_ke + ldj_deq.unsqueeze(1))               # (B,K)
    means = nll_ke.mean(dim=0)                             # (K,)
    stds = nll_ke.std(dim=0)
    srt, _ = means.sort()                                  # ascending NLL
    gap = float(srt[1] - srt[0]) if means.numel() >= 2 else 0.0
    return {"per_expert_nll_mean": means.tolist(),
            "per_expert_nll_std": stds.tolist(),
            "per_expert_logp_gap": gap,
            "single_nll": float(means.min())}              # best single expert
