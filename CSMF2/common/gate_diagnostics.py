# =============================================================================
# COND-GATE v0.3 -- common.gate_diagnostics
# Purpose: check #9 -- gate collapse probe. Separate module so Stages 1.2, 3.2,
#          and the WP3 ablation matrix re-use the same code.
# CONVENTION: NLL = LOSS (lower = better). Probe raises ValueError on fail and
#             logs via logger.error -- never silent pass / mock / dummy.
# Changelog (new in v0.3):
#   * Introduced. Reports neff_mean, entropy_mean, max_w_mean, per-input
#     variance of w, and argmax-expert histogram.
#   * Raises on any of: Neff<1.5, max_w>0.95, w constant across inputs,
#     weights that do not sum to 1.
# =============================================================================
from __future__ import annotations
import logging
import traceback
logger = logging.getLogger(__name__)
__version__ = "0.3"
__abbr__ = "COND-GATE"

import torch

NEFF_MIN     = 1.5
MAX_W_MAX    = 0.95
VAR_EPS      = 1e-6
SUM_TOL      = 1e-4


def gate_collapse_probe(gate_fn, y_batch,
                        neff_min=NEFF_MIN, max_w_max=MAX_W_MAX,
                        var_eps=VAR_EPS, raise_on_fail=True):
    # gate_fn(y) -> weights (B, K) summing to 1. Detects dead / collapsed gate.
    try:
        w = gate_fn(y_batch)
    except Exception:
        logger.error("[gate_collapse_probe] gate_fn crashed\n%s", traceback.format_exc())
        raise
    w = w.detach()
    if w.dim() != 2:
        logger.error("[gate_collapse_probe] expected (B,K) got shape %s", tuple(w.shape))
        raise ValueError(f"gate_collapse_probe: expected (B,K) got {tuple(w.shape)}")
    B, K = w.shape

    row_sums = w.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=SUM_TOL):
        lo, hi = float(row_sums.min().item()), float(row_sums.max().item())
        logger.error("[gate_collapse_probe] weights do not sum to 1: rows in [%.4f, %.4f]", lo, hi)
        if raise_on_fail:
            raise ValueError(f"gate_collapse_probe: row sums in [{lo:.4f},{hi:.4f}]")

    entropy = -(w * w.clamp_min(1e-12).log()).sum(-1)          # (B,)
    neff    = entropy.exp()                                     # (B,)
    max_w   = w.max(-1).values                                  # (B,)
    w_var_across_inputs = float(w.var(dim=0).mean().item())     # scalar

    neff_mean    = float(neff.mean().item())
    entropy_mean = float(entropy.mean().item())
    max_w_mean   = float(max_w.mean().item())

    if neff_mean < neff_min:
        logger.error("[gate_collapse_probe] Neff=%.3f < %.2f (gate collapsed)",
                     neff_mean, neff_min)
        if raise_on_fail:
            raise ValueError(f"gate_collapse_probe: Neff={neff_mean:.3f} < {neff_min}")
    if max_w_mean > max_w_max:
        logger.error("[gate_collapse_probe] mean max_w=%.3f > %.2f (one expert dominates)",
                     max_w_mean, max_w_max)
        if raise_on_fail:
            raise ValueError(f"gate_collapse_probe: max_w={max_w_mean:.3f} > {max_w_max}")
    if w_var_across_inputs < var_eps:
        logger.error("[gate_collapse_probe] w constant across inputs (var=%.2e)",
                     w_var_across_inputs)
        if raise_on_fail:
            raise ValueError(f"gate_collapse_probe: w constant across inputs "
                             f"(var={w_var_across_inputs:.2e})")

    argmax_hist = torch.bincount(w.argmax(-1), minlength=K).cpu().numpy()

    return {"neff_mean": neff_mean,
            "entropy_mean": entropy_mean,
            "max_w_mean": max_w_mean,
            "w_var_across_inputs": w_var_across_inputs,
            "neff_per_sample": neff.cpu().numpy(),
            "argmax_hist": argmax_hist,
            "K": K, "B": B}
