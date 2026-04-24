# =============================================================================
# COND-GATE v0.3 -- common.cond_diagnostics
# Purpose: 8 conditioning sanity checks (h / FiLM / s,t / grads / cache /
#          determinism / per-layer FiLM). Check #9 (gate collapse) lives in
#          gate_diagnostics.py. Bundler run_global_gate() calls both.
# CONVENTION: NLL = LOSS (lower = better). All checks raise ValueError on fail
#             and log via logger.error -- never silent pass / mock / dummy.
# Changelog (v0.2 -> v0.3):
#   * Added film_stats_per_layer (check #8) to catch one broken FiLM head
#     that aggregate stats hide.
#   * Bundler run_global_gate() now optionally calls gate_collapse_probe
#     from gate_diagnostics when a gate module is provided.
#   * No threshold changes on checks 1-7.
# =============================================================================
from __future__ import annotations
import logging
import traceback
logger = logging.getLogger(__name__)
__version__ = "0.3"
__abbr__ = "COND-GATE"

import torch

# --- Default thresholds (override by passing tol=... where supported) --------
STD_EPS            = 1e-8
CACHE_TOL          = 1e-5
DETERMINISM_TOL    = 1e-6
GRAD_EPS           = 1e-10
SHUFFLE_DELTA_EPS  = 1e-4
DIVERSITY_EPS      = 1e-6


def _t(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


# ----- Check 1: h_stats ------------------------------------------------------
def h_stats(h, raise_on_fail=True):
    h = _t(h).detach()
    nan = int(torch.isnan(h).sum().item())
    inf = int(torch.isinf(h).sum().item())
    if nan or inf:
        logger.error("[h_stats] NaN=%d Inf=%d", nan, inf)
        if raise_on_fail:
            raise ValueError(f"h_stats: NaN={nan} Inf={inf}")
    fin = h[torch.isfinite(h)]
    if fin.numel() == 0:
        logger.error("[h_stats] no finite values")
        raise ValueError("h_stats: no finite values")
    mean = float(fin.mean().item())
    std  = float(fin.std().item())
    nrm  = float(torch.linalg.vector_norm(h.flatten()).item())
    nuq  = int(torch.unique(fin).numel())
    if std < STD_EPS:
        logger.error("[h_stats] h appears constant (std=%.2e)", std)
        if raise_on_fail:
            raise ValueError(f"h_stats: constant h (std={std:.2e})")
    return {"mean": mean, "std": std, "norm": nrm,
            "nan": nan, "inf": inf, "n_unique": nuq}


# ----- Check 2: h_diversity --------------------------------------------------
def h_diversity(h_batch, raise_on_fail=True):
    h = _t(h_batch).detach().flatten(start_dim=1)
    B = h.shape[0]
    if B < 2:
        logger.error("[h_diversity] need B>=2 got B=%d", B)
        raise ValueError(f"h_diversity: need B>=2 got {B}")
    d = torch.cdist(h, h)
    mask = ~torch.eye(B, dtype=torch.bool, device=h.device)
    off = d[mask]
    mean_d = float(off.mean().item())
    if mean_d < DIVERSITY_EPS:
        logger.error("[h_diversity] batch-collapsed (mean pairwise=%.2e)", mean_d)
        if raise_on_fail:
            raise ValueError(f"h_diversity: collapsed (mean={mean_d:.2e})")
    return {"mean_pairwise": mean_d,
            "min_pairwise":  float(off.min().item()),
            "max_pairwise":  float(off.max().item()),
            "pairwise_matrix": d.cpu().numpy()}


# ----- Check 3: film_stats (aggregate) ---------------------------------------
def film_stats(gamma, beta, raise_on_fail=True):
    out = {}
    for name, t in [("gamma", _t(gamma).detach()), ("beta", _t(beta).detach())]:
        nan = int(torch.isnan(t).sum().item())
        inf = int(torch.isinf(t).sum().item())
        fin = t[torch.isfinite(t)]
        mean = float(fin.mean().item()) if fin.numel() else float("nan")
        std  = float(fin.std().item())  if fin.numel() else float("nan")
        out[name] = {"mean": mean, "std": std, "nan": nan, "inf": inf}
        if nan or inf:
            logger.error("[film_stats] %s NaN=%d Inf=%d", name, nan, inf)
            if raise_on_fail:
                raise ValueError(f"film_stats: {name} NaN={nan} Inf={inf}")
        if std < STD_EPS:
            logger.error("[film_stats] %s constant (std=%.2e)", name, std)
            if raise_on_fail:
                raise ValueError(f"film_stats: {name} constant (std={std:.2e})")
    return out


# ----- Check 4: st_sensitivity ----------------------------------------------
def st_sensitivity(flow_fn, x, h, h_shuffled, raise_on_fail=True):
    # flow_fn(x, h) -> (s, t, logp). Shuffle should perturb all three.
    try:
        s1, t1, lp1 = flow_fn(_t(x), _t(h))
        s2, t2, lp2 = flow_fn(_t(x), _t(h_shuffled))
    except Exception:
        logger.error("[st_sensitivity] flow_fn crashed\n%s", traceback.format_exc())
        raise
    ds  = float((s1 - s2).abs().max().item())
    dt  = float((t1 - t2).abs().max().item())
    dlp = float((lp1 - lp2).abs().max().item())
    if ds < SHUFFLE_DELTA_EPS or dt < SHUFFLE_DELTA_EPS:
        logger.error("[st_sensitivity] (s,t) insensitive: ds=%.2e dt=%.2e", ds, dt)
        if raise_on_fail:
            raise ValueError(f"st_sensitivity: (s,t) ignores h (ds={ds:.2e} dt={dt:.2e})")
    if dlp < SHUFFLE_DELTA_EPS:
        logger.error("[st_sensitivity] logp insensitive: dlogp=%.2e", dlp)
        if raise_on_fail:
            raise ValueError(f"st_sensitivity: logp ignores h (dlogp={dlp:.2e})")
    return {"ds_max": ds, "dt_max": dt, "dlogp_max": dlp}


# ----- Check 5: grad_norms ---------------------------------------------------
def grad_norms(cond_net, film_heads, raise_on_fail=True):
    # Call AFTER loss.backward(). Reads .grad on each param.
    def _norm(params):
        tot, n = 0.0, 0
        for p in params:
            if p.grad is None:
                continue
            tot += float(p.grad.detach().pow(2).sum().item())
            n += 1
        return tot ** 0.5, n

    cn, cn_n = _norm(cond_net.parameters())
    if cn_n == 0 or cn < GRAD_EPS:
        logger.error("[grad_norms] conditioner grad=%.2e n_params=%d", cn, cn_n)
        if raise_on_fail:
            raise ValueError(f"grad_norms: conditioner no gradient ({cn:.2e})")
    heads = []
    for i, h in enumerate(film_heads):
        hn, hn_n = _norm(h.parameters())
        heads.append(hn)
        if hn_n == 0 or hn < GRAD_EPS:
            logger.error("[grad_norms] FiLM head %d grad=%.2e", i, hn)
            if raise_on_fail:
                raise ValueError(f"grad_norms: FiLM head {i} no gradient ({hn:.2e})")
    return {"conditioner": cn, "film_heads": heads}


# ----- Check 6: cache_check --------------------------------------------------
def cache_check(h_cached, h_fresh, tol=CACHE_TOL, raise_on_fail=True):
    hc = _t(h_cached).detach()
    hf = _t(h_fresh).detach()
    if hc.shape != hf.shape:
        logger.error("[cache_check] shape mismatch %s vs %s", hc.shape, hf.shape)
        raise ValueError("cache_check: shape mismatch")
    err = float((hc - hf).abs().max().item())
    if err > tol:
        logger.error("[cache_check] max|diff|=%.2e > tol=%.2e", err, tol)
        if raise_on_fail:
            raise ValueError(f"cache_check: max|diff|={err:.2e} > {tol:.2e}")
    return {"max_abs_diff": err}


# ----- Check 7: determinism_check -------------------------------------------
def determinism_check(cond_net, y, seed, set_seed_fn,
                      tol=DETERMINISM_TOL, raise_on_fail=True):
    # Run cond_net(y) twice under set_seed_fn(seed). Assert identical output.
    try:
        was_training = cond_net.training
        cond_net.eval()
        set_seed_fn(seed)
        with torch.no_grad():
            h1 = cond_net(_t(y)).detach().clone()
        set_seed_fn(seed)
        with torch.no_grad():
            h2 = cond_net(_t(y)).detach().clone()
        if was_training:
            cond_net.train()
    except Exception:
        logger.error("[determinism_check] cond_net crashed\n%s", traceback.format_exc())
        raise
    err = float((h1 - h2).abs().max().item())
    if err > tol:
        logger.error("[determinism_check] max|Δh|=%.2e > tol=%.2e (hidden RNG?)", err, tol)
        if raise_on_fail:
            raise ValueError(f"determinism_check: max|Δh|={err:.2e} > {tol:.2e}")
    return {"max_abs_delta_h": err}


# ----- Check 8: film_stats_per_layer ----------------------------------------
def film_stats_per_layer(film_head_outputs, raise_on_fail=True):
    # film_head_outputs: list[(gamma_i, beta_i)] -- one tuple per FiLM layer.
    per = []
    for i, (g, b) in enumerate(film_head_outputs):
        try:
            s = film_stats(g, b, raise_on_fail=raise_on_fail)
        except ValueError as e:
            logger.error("[film_stats_per_layer] layer %d failed: %s", i, e)
            raise
        s["layer_index"] = i
        per.append(s)
    return per


# ----- Move-forward gate -----------------------------------------------------
def check_move_forward(log):
    # log: dict aggregated across a completed run (expected keys below).
    # Returns (ok: bool, reasons: list[str]).
    reasons = []
    if not log.get("main_metric_improved", False):
        reasons.append("main metric did not improve")
    if log.get("nll_regression", False):
        reasons.append("NLL regression detected")
    if log.get("numerical_failure", False):
        reasons.append("numerical failure occurred")
    if log.get("n_seeds_ok", 0) < 3:
        reasons.append(f"only {log.get('n_seeds_ok',0)} seeds passed (need >=3)")
    if not log.get("h_finite_nonconstant", False):
        reasons.append("h not finite / not varying / not deterministic")
    if not log.get("film_valid_all_layers", False):
        reasons.append("one or more FiLM layers invalid")
    if not log.get("shuffle_changes_st_and_logp", False):
        reasons.append("shuffling h did not change s,t or logp")
    if not log.get("grads_flow_everywhere", False):
        reasons.append("conditioner or FiLM head has no gradient")
    if not log.get("cache_matches_fresh", True):
        reasons.append("cached h disagrees with fresh h")
    if not log.get("determinism_ok", False):
        reasons.append("(y,seed) does not reproduce identical h")
    if log.get("gate_used", False) and not log.get("gate_healthy", False):
        reasons.append("gate collapsed (Neff/max_w/variance check failed)")
    return (len(reasons) == 0, reasons)


# ----- Bundler ---------------------------------------------------------------
def run_global_gate(*, h=None, h_batch=None,
                    gamma_aggregate=None, beta_aggregate=None,
                    film_per_layer_outputs=None,
                    flow_fn=None, x=None, h_shuffled=None,
                    cond_net=None, film_heads=None,
                    h_cached=None, h_fresh=None,
                    y=None, seed=None, set_seed_fn=None,
                    gate_fn=None, y_batch=None,
                    raise_on_fail=False):
    # Runs all checks whose required args are provided. Others are skipped +
    # logged. Returns {passed, reasons, metrics}. Never silent.
    metrics, reasons = {}, []

    def _try(name, fn, *args, **kw):
        try:
            metrics[name] = fn(*args, **kw, raise_on_fail=True)
            return True
        except ValueError as e:
            reasons.append(f"{name}: {e}")
            return False
        except Exception as e:
            logger.error("[run_global_gate] %s crashed: %s\n%s",
                         name, e, traceback.format_exc())
            reasons.append(f"{name}: crash -- {e}")
            return False

    ok = True
    if h is not None:
        ok &= _try("h_stats", h_stats, h)
    if h_batch is not None:
        ok &= _try("h_diversity", h_diversity, h_batch)
    if gamma_aggregate is not None and beta_aggregate is not None:
        ok &= _try("film_stats", film_stats, gamma_aggregate, beta_aggregate)
    if film_per_layer_outputs is not None:
        ok &= _try("film_stats_per_layer", film_stats_per_layer, film_per_layer_outputs)
    if flow_fn is not None and x is not None and h is not None and h_shuffled is not None:
        ok &= _try("st_sensitivity", st_sensitivity, flow_fn, x, h, h_shuffled)
    if cond_net is not None and film_heads is not None:
        ok &= _try("grad_norms", grad_norms, cond_net, film_heads)
    if h_cached is not None and h_fresh is not None:
        ok &= _try("cache_check", cache_check, h_cached, h_fresh)
    if cond_net is not None and y is not None and seed is not None and set_seed_fn is not None:
        ok &= _try("determinism_check", determinism_check,
                   cond_net, y, seed, set_seed_fn)
    if gate_fn is not None and y_batch is not None:
        try:
            from .gate_diagnostics import gate_collapse_probe
        except ImportError:
            from gate_diagnostics import gate_collapse_probe   # flat-layout fallback
        ok &= _try("gate_collapse_probe", gate_collapse_probe, gate_fn, y_batch)

    if not ok and raise_on_fail:
        raise ValueError("COND-GATE failed: " + "; ".join(reasons))
    return {"passed": ok, "reasons": reasons, "metrics": metrics}
