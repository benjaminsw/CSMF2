# =============================================================================
# STEP-2_3 v0.1 -- experiments.step_2_3.losses  (S2.3 objective terms + verdict)
# Purpose: the Stage 2.3 loss terms that are NEW relative to 1.3/1.4, plus the
#          explicit GO/STOP verdict. Per S2.3-PLAN v0.4:
#            all_expert_consistency  = mean_k rec_k   (NOT raw sum, NOT winner)
#            gate_entropy_loss        = -mean_y H(p(y))   (anti-collapse)
#            gate_load_balance_loss   = || mean_y p(y) - uniform ||^2
#          Per-expert rec_k REUSES step_1_4b.consistency_loss.consistency_term
#          (the SAME forward operator A as RECGATE/breakdown). NOTE: this training
#          term is the mu-mean (eps=0) reconstruction; the VERDICT's V2 metric uses
#          the z-bank deterministic-proxy residual (step_1_3.scores.per_expert_rec)
#          to match the frozen Stage-1.3 baseline. Same operator A, correlated,
#          but NOT the same reconstruction -- see all_expert_consistency docstring.
# CONVENTION: no fallback/mock. Non-finite -> logger.error + raise. Anti-collapse
#          terms are regularizers with their OWN small fixed coefficients, kept
#          OUTSIDE the L0/GradNorm-normalized objective (S2.3-PLAN NOTE 1); the
#          trainer still LOGS their grad norms (NOTE 2).
# Changelog (v0.1 -> v0.2, fix D after 2.3-A diverged on NSF spline inverse):
#   * all_expert_consistency gains expert_names + exclude: experts in `exclude`
#     are SKIPPED ENTIRELY (their consistency_term -> decode -> spline-inverse is
#     never run), so no rec gradient touches them. Motivation: NSF's RQ-spline
#     inverse drifts non-invertible under rec pressure (disc -> -3e-2, WORSE at
#     lower LR -> drift not float32 precision; crash pinned epoch 6 across
#     lr/clip), and NSF already wins rec ~4990/5000 so it needs no rec rescue.
#     per_expert_list entry is None for excluded experts. >=1 must remain.
# Changelog (NEW in v0.1):
#   * all_expert_consistency(experts, conds, y, ...) = mean_k consistency_term.
#   * gate_entropy_loss / gate_load_balance_loss from log_weights (B,K).
#   * stage_2_3a_verdict(...) -- executable GO/STOP gates (V1 tier, V2 mixture
#     vs NSF-only, G2 Neff/max_weight, NLL-regression + numeric guards).
# Update summary:
#   v0.1 is the only new loss math + the verdict logic for 2.3-A. argmax is used
#   ONLY for the rec_argmin REPORT (diagnostic), never in the training loss.
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.2"
__abbr__ = "STEP-2_3"

import torch

# REUSE the per-expert consistency (same operator A; mu-mean reconstruction).
# The verdict's V2 uses a z-bank proxy instead -- correlated, not identical.
from ..step_1_4b.consistency_loss import consistency_term


# --------------------------------------------------------------------------- #
# All-expert reconstruction consistency -- MEAN over experts (Option 1)
# --------------------------------------------------------------------------- #
def all_expert_consistency(experts, conds, y, *, blur_sigma: float, scale: int,
                           noise_sigma: float, expert_names=None, exclude=()):
    """L_rec = mean_{k not excluded} rec_k, rec_k = || A(x_hat_k) - y ||^2 (noise-rel).

    experts : list of K CBExpert models (UNFROZEN in 2.3).
    conds   : list of K conditioners (h_k = cond_k(y)). Each expert uses its OWN h.
    expert_names / exclude : if given, experts whose name is in `exclude` are
              SKIPPED ENTIRELY -- their consistency_term (-> decode -> spline
              inverse) is never computed, so a rec gradient never touches them.
              This is fix D: NSF's RQ-spline inverse goes non-invertible under rec
              pressure (disc -> -3e-2, worsening as LR drops -> drift, not float32
              precision), and NSF already wins rec ~4990/5000, so it needs no rec
              rescue. NSF stays full in NLL + gate + mixture; only the TRAINING rec
              gradient skips it. Excluded experts get per_expert_list entry None.
    Returns (L_rec_mean, per_expert_list aligned to `experts`; None where excluded).
    MEAN over the INCLUDED experts (stable scale); >=1 expert must remain."""
    if len(experts) != len(conds):
        logger.error("[all_expert_consistency] experts(%d) != conds(%d)",
                     len(experts), len(conds))
        raise ValueError("experts/conds length mismatch")
    if len(experts) == 0:
        logger.error("[all_expert_consistency] no experts")
        raise ValueError("no experts")
    exclude = set(exclude or ())
    if exclude and expert_names is None:
        logger.error("[all_expert_consistency] exclude=%s given but expert_names "
                     "is None -- cannot resolve which experts to skip", exclude)
        raise ValueError("exclude requires expert_names")
    names = list(expert_names) if expert_names is not None else [None] * len(experts)

    per = []                      # aligned to experts; None for excluded
    included = []                 # the rec tensors actually in the mean
    for k, (m, cond) in enumerate(zip(experts, conds)):
        if names[k] in exclude:
            per.append(None)      # SKIP: no decode, no spline inverse, no rec grad
            continue
        h = cond(y)
        rec_k = consistency_term(m, h, y, blur_sigma=blur_sigma, scale=scale,
                                 noise_sigma=noise_sigma)
        per.append(rec_k)
        included.append(rec_k)
    if not included:
        logger.error("[all_expert_consistency] exclude=%s removed ALL experts; "
                     ">=1 must remain in the rec term", exclude)
        raise ValueError("rec_exclude removed all experts")
    L_rec = torch.stack(included).mean()
    if not torch.isfinite(L_rec).all():
        logger.error("[all_expert_consistency] non-finite mean rec")
        raise RuntimeError("non-finite all-expert consistency")
    return L_rec, [float(r.detach()) if r is not None else None for r in per]


# --------------------------------------------------------------------------- #
# Anti-collapse regularizers (OUTSIDE the normalized objective; small fixed coef)
# --------------------------------------------------------------------------- #
def gate_entropy_loss(log_w: torch.Tensor) -> torch.Tensor:
    """-mean_y H(p(y)). MINIMIZING this term's NEGATIVE-entropy form would
    sharpen the gate; we want the OPPOSITE (keep it spread), so the trainer adds
    -entropy_lambda * H, i.e. this returns -H (to be ADDED with a +lambda) ...
    To avoid sign confusion: returns mean negative entropy = -mean_y H(p).
    Trainer adds (+entropy_lambda * gate_entropy_loss) -> penalizes LOW entropy.
    Wait: we want HIGH entropy -> penalize low entropy -> add -lambda*H.
    So this returns mean_y H(p) and the trainer SUBTRACTS it (loss -= lambda*H).
    Returns mean entropy H (>=0)."""
    w = log_w.exp()
    H = -(w * log_w).sum(dim=1)                      # (B,) entropy, nats
    H_mean = H.mean()
    if not torch.isfinite(H_mean).all():
        logger.error("[gate_entropy_loss] non-finite entropy")
        raise RuntimeError("non-finite gate entropy")
    return H_mean                                    # trainer: loss -= lambda*H_mean


def gate_load_balance_loss(log_w: torch.Tensor) -> torch.Tensor:
    """|| mean_y p(y) - uniform ||^2. Penalizes a gate that, on AVERAGE, ignores
    some experts (batch-level load imbalance). Added with +lambda."""
    w = log_w.exp()
    mean_p = w.mean(dim=0)                            # (K,)
    k = w.size(1)
    uniform = torch.full_like(mean_p, 1.0 / k)
    lb = (mean_p - uniform).pow(2).sum()
    if not torch.isfinite(lb).all():
        logger.error("[gate_load_balance_loss] non-finite load-balance")
        raise RuntimeError("non-finite load balance")
    return lb


# --------------------------------------------------------------------------- #
# Explicit GO/STOP verdict for 2.3-A (executable gates, S2.3-PLAN v0.4)
# --------------------------------------------------------------------------- #
def stage_2_3a_verdict(*, recargmin_tier: str, beats_nsf_only: bool,
                       neff_mean: float, max_weight: float,
                       nll_now: float, nll_baseline: float,
                       numeric_ok: bool,
                       neff_min: float = 1.5, max_weight_max: float = 0.70,
                       nll_regression_tol: float = 0.05) -> dict:
    """Encodes the approved GO/STOP gates as code, not prose.

    GO  if  V1 tier non-FLAT (PROVISIONAL+ preferred)
        AND V2 mixture beats NSF-only (soft_fwd_rel improves)
        AND G2 Neff > neff_min AND max_weight < max_weight_max
        AND no NLL critical regression AND numeric checks pass
    STOP if  tier FLAT, OR mixture does not beat NSF-only, OR gate still collapses.

    nll_regression_tol: fractional worsening of NLL allowed vs baseline (joint
    training may trade a little density for reconstruction; a LARGE regression is
    a fail). nll is negative here (lower=better), so 'critical regression' means
    nll_now > nll_baseline * (1 - tol) when baseline<0 -> compare on magnitude."""
    tier = str(recargmin_tier).upper()
    TIER_RANK = {"FLAT": 0, "WEAK": 1, "PROVISIONAL": 2,
                 "PROVISIONAL_CLUSTER": 3, "STRONG": 4}
    rank = TIER_RANK.get(tier, -1)
    if rank < 0:
        logger.error("[verdict] unknown tier %r", recargmin_tier)
        raise ValueError(f"unknown RECARGMIN tier {recargmin_tier!r}")

    v1_non_flat = rank >= TIER_RANK["WEAK"]
    v1_preferred = rank >= TIER_RANK["PROVISIONAL"]
    v2_beats = bool(beats_nsf_only)
    g2_neff = neff_mean > neff_min
    g2_weight = max_weight < max_weight_max
    g2_ok = g2_neff and g2_weight
    # NLL regression: both negative; allow small worsening. fail if magnitude
    # of improvement dropped by > tol relative to baseline magnitude.
    nll_ok = nll_now <= abs(nll_baseline) * nll_regression_tol + nll_baseline \
        if nll_baseline < 0 else nll_now <= nll_baseline * (1 + nll_regression_tol)

    gates = {"V1_non_flat": v1_non_flat, "V1_preferred_provisional+": v1_preferred,
             "V2_beats_nsf_only": v2_beats,
             "G2_neff_ok": g2_neff, "G2_max_weight_ok": g2_weight,
             "nll_no_critical_regression": bool(nll_ok),
             "numeric_ok": bool(numeric_ok)}
    go = (v1_non_flat and v2_beats and g2_ok and nll_ok and numeric_ok)
    verdict = "GO" if go else "STOP"
    # reason for STOP (first failing primary gate)
    if not go:
        if not v1_non_flat:
            reason = "RECARGMIN tier FLAT"
        elif not v2_beats:
            reason = "mixture does not beat NSF-only"
        elif not g2_ok:
            reason = f"gate collapse (Neff_mean={neff_mean:.2f}, max_w={max_weight:.2f})"
        elif not nll_ok:
            reason = "NLL critical regression"
        else:
            reason = "numeric check failed"
    else:
        reason = "all primary gates pass"
    return {"verdict": verdict, "reason": reason, "tier": tier, "gates": gates}
