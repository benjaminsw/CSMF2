# =============================================================================
# STEP-1_4A v0.1 -- experiments.step_1_4a.cond_base
# Purpose: posterior-safe conditional base N(mu(h), sigma(h)) for a flow
#          expert. Replaces the fixed N(0,I) base over the flow latent w with
#          a conditional Gaussian whose params are read from h = cond(y).
#          NLL stays EXACT: base log-prob is a proper normalized Gaussian.
# CONVENTION: non-finite / sigma<=0 -> logger.error + raise. No fallback/mock.
# Identity-safe init (base_init='zero_mu_unit_sigma'): zero the final layers so
#          mu(h)=0, logsigma(h)=0 -> N(0,I) -> CB expert == baseline at start.
# Diagnostics (base-collapse surface, mirrors FiLM gamma->0):
#   mu_std_across_y, log_sigma_std_across_y, base_shuffle_gap, base_alive
#   base_effect_magnitude = mean KL( N(mu(h),sigma(h)) || N(0,I) )  [tracked]
# Changelog (NEW in v0.1):
#   * Introduced. ConditionalBase (mu/logsigma nets, logsigma clamp, zero
#     init), base_logp, base_sample (eps->w), base_diagnostics, base_kl.
# Update summary:
#   v0.1 makes the base conditional and posterior-safe. base reads the SAME h
#   as the couplings (no interference). KL-to-N(0,I) tracked to catch "mu/logs
#   vary slightly but the base barely moves".
# =============================================================================
from __future__ import annotations
import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_4A"

_LOG2PI = math.log(2.0 * math.pi)


def _mlp(in_dim, hidden, out_dim, zero_final):
    net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                        nn.Linear(hidden, out_dim))
    if zero_final:
        nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
    return net


class ConditionalBase(nn.Module):
    """z-base N(mu(h), sigma(h)) over the flow latent w (dim D). h: (B, h_dim).
    base_init='zero_mu_unit_sigma' -> starts identical to N(0,I)."""
    def __init__(self, dim: int, h_dim: int, *, mu_hidden: int,
                 logsigma_hidden: int, logsigma_min: float, logsigma_max: float,
                 base_init: str = "zero_mu_unit_sigma", base_gain: float = 1.0):
        super().__init__()
        if logsigma_min >= logsigma_max:
            logger.error("[CBase] need logsigma_min < logsigma_max, got %s/%s",
                         logsigma_min, logsigma_max)
            raise ValueError("logsigma_min must be < logsigma_max")
        self.dim = int(dim)
        self.logsigma_min = float(logsigma_min)
        self.logsigma_max = float(logsigma_max)
        self.base_gain = float(base_gain)
        zero = (base_init == "zero_mu_unit_sigma")
        if base_init not in ("zero_mu_unit_sigma", "random"):
            logger.error("[CBase] unknown base_init %r", base_init)
            raise ValueError(f"unknown base_init {base_init!r}")
        self.mu_net = _mlp(h_dim, mu_hidden, dim, zero_final=zero)
        self.logsigma_net = _mlp(h_dim, logsigma_hidden, dim, zero_final=zero)

    def params(self, h):
        """Return (mu, logsigma, sigma), each (B, dim). logsigma clamped."""
        mu = self.base_gain * self.mu_net(h)
        logsigma = self.logsigma_net(h).clamp(self.logsigma_min, self.logsigma_max)
        sigma = logsigma.exp()
        if not (torch.isfinite(mu).all() and torch.isfinite(sigma).all()):
            logger.error("[CBase] non-finite base params")
            raise RuntimeError("non-finite conditional-base params")
        return mu, logsigma, sigma

    def log_prob(self, w, h):
        """log N(w; mu(h), sigma(h)) summed over dims. w,(B,D); h,(B,h_dim)."""
        mu, logsigma, sigma = self.params(h)
        lp = -0.5 * (((w - mu) / sigma) ** 2 + _LOG2PI) - logsigma
        return lp.sum(dim=-1)

    def to_w(self, eps, h):
        """Map standard-normal eps (B,D) to base sample w = mu + sigma*eps."""
        mu, _, sigma = self.params(h)
        return mu + sigma * eps


def base_kl(mu, logsigma, sigma):
    """KL( N(mu,sigma^2) || N(0,1) ) per sample, summed over dims. (B,)."""
    return (0.5 * (sigma ** 2 + mu ** 2 - 1.0) - logsigma).sum(dim=-1)


@torch.no_grad()
def base_diagnostics(base: ConditionalBase, cond, y, *, tau_b: float,
                     n_y: int = 64) -> dict:
    """Is the base actually conditional? Computes spread of mu/logsigma across
    y, a shuffle gap, KL-to-N(0,I), and the base_alive flag."""
    y_sel = y[:n_y]
    h = cond(y_sel)
    mu, logsigma, sigma = base.params(h)
    mu_std = float(mu.std(dim=0, unbiased=False).mean())
    ls_std = float(logsigma.std(dim=0, unbiased=False).mean())
    kl = base_kl(mu, logsigma, sigma)
    # shuffle gap: base logp of w drawn at h(y) vs h(y_shuffled), fixed w=mu
    perm = torch.randperm(h.size(0), device=h.device)
    lp_real = base.log_prob(mu, h)
    lp_shuf = base.log_prob(mu, h[perm])
    gap = float((lp_real - lp_shuf).mean())
    alive = bool(mu_std > tau_b or ls_std > tau_b)
    out = {
        "mu_std_across_y": mu_std,
        "log_sigma_std_across_y": ls_std,
        "mu_mean": float(mu.mean()),
        "log_sigma_mean": float(logsigma.mean()),
        "sigma_min": float(sigma.min()),
        "sigma_max": float(sigma.max()),
        "base_shuffle_gap": gap,
        "base_effect_magnitude": float(kl.mean()),   # mean KL to N(0,I) (tracked)
        "base_alive": alive,
        "tau_b": tau_b,
    }
    logger.info("[CBase] mu_std=%.3e ls_std=%.3e KL=%.3e gap=%.3e alive=%s",
                mu_std, ls_std, out["base_effect_magnitude"], gap, alive)
    return out
