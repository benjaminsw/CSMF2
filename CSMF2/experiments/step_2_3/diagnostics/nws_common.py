# =============================================================================
# NWS v0.4 -- CSMF2.experiments.step_2_3.diagnostics.nws_common
# Purpose: Shared helpers for Step 2.3-NWS (Near-Winner Sweep). Diagnostic ONLY:
#          re-scores frozen 2.3-A experts, trains nothing, does NOT change the
#          sealed 2.3-A STOP. Re-uses the PROJECT'S OWN scorer (step_1_3.scores.
#          per_expert_rec -- z-bank, sum-over-pixels) so NWS numbers are
#          definitionally consistent with the 5000/0/0 rec_argmin it re-credits.
# CONVENTION: NLL = LOSS. No silent fallback / mock / pass. Every failure path
#          -> logger.error + raise.
# IMPORTANT METRIC NOTE: rec_k here = per_expert_rec (z-bank, SUM over pixels,
#   NOT noise-normalized). This is the rec_argmin signal, a DIFFERENT
#   reconstruction from the V2 soft_fwd_rel (= 0.156, mu-mean). So the oracle /
#   verdict reference is NSF's OWN per_expert_rec mean, NOT 0.156. 0.156 is kept
#   for context only and never used as a pass line here.
# Changelog (v0.3 -> v0.4, final wire-up against the real tree):
#   * Scorer switched to step_1_3.scores.per_expert_rec (the gate's actual rec).
#     Removed the consistency_term replica (that was the V2/mu-mean metric).
#   * Experts loaded via step_2_3.model_io.load_experts_trainable(train=False);
#     val data via data.degrade.MNISTDegraded; z-bank via make_z_bank using the
#     run cfg (rec_z_bank_size=4, rec_z_bank_seed=1234).
#   * Verdict reference = nsf_mean_rec (not 0.156). pixel_A kept for soft-gate
#     blends + residual differences. inverse_logit kept for pixel-space x_hat.
# Update summary:
#   Converges NWS onto the project's own reconstruction scorer so the diagnostic
#   re-credits exactly the metric that produced the 2.3-A argmin, with no
#   parallel reimplementation that could drift.
# =============================================================================
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
__version__ = "0.4"
__abbr__ = "NWS"

# -----------------------------------------------------------------------------
# Constants (cell s2/n0.05; refs from sealed 2.3-A run 28eb52bbfe20)
# -----------------------------------------------------------------------------
EXPERTS: tuple[str, ...] = ("nsf", "realnvp", "nice_mix")   # cfg.expert_set order
NON_NSF: tuple[str, ...] = ("realnvp", "nice_mix")
TAUS: tuple[float, ...] = (0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50)

V2_SOFT_FWD_REL_CONTEXT: float = 0.156   # 2.3-A V2 metric -- CONTEXT ONLY (diff metric)
SIGNAL_TAU_MAX: float = 0.10
Q4_UNIFORM: float = 0.25

SEED_RNG: dict[int, int] = {0: 2026, 1: 2027, 2: 2028}
QUARTILE_LABELS: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")   # Q1 best rec .. Q4 worst

VERDICT_SIGNAL = "NWS-SIGNAL"
VERDICT_NO_SIGNAL = "NWS-NO-SIGNAL"


# -----------------------------------------------------------------------------
# WIREUP -- lazy, loud imports of the REAL project surface. No fallbacks.
#   load_experts_trainable(ckpt_dirs, device, *, train=False) -> (experts, ref)
#   per_expert_rec(experts, y, z_bank, *, blur_sigma, scale) -> (B,K)  [no_grad]
#   make_z_bank(dim, size, mode, seed, device, dtype) -> (S, dim)
#   MNISTDegraded(root, split=, sigma=, scale=, noise_sigma=)[i] -> (x(1,28,28), y(1,14,14))
#   export_trained_experts(s23_ckpt, s23_report, source_dirs, out_root, device) -> [dirs]
#   inverse_logit / blur / downsample from data.degrade
# -----------------------------------------------------------------------------
def _wire_load_experts_trainable():
    try:
        from CSMF2.experiments.step_2_3.model_io import load_experts_trainable
        return load_experts_trainable
    except Exception as e:
        logger.error("[WIREUP] load_experts_trainable unavailable: %s", e)
        raise


def _wire_rec_and_zbank():
    try:
        from CSMF2.experiments.step_1_3.scores import per_expert_rec, make_z_bank
        return per_expert_rec, make_z_bank
    except Exception as e:
        logger.error("[WIREUP] per_expert_rec/make_z_bank unavailable: %s", e)
        raise


def _wire_mnist_degraded():
    try:
        from CSMF2.data.degrade import MNISTDegraded
        return MNISTDegraded
    except Exception as e:
        logger.error("[WIREUP] MNISTDegraded unavailable from data.degrade: %s", e)
        raise


def _wire_export_trained_experts():
    try:
        from CSMF2.experiments.step_2_3.export_experts import export_trained_experts
        return export_trained_experts
    except Exception as e:
        logger.error("[WIREUP] export_trained_experts unavailable: %s", e)
        raise


def _wire_degrade_ops():
    try:
        from CSMF2.data.degrade import inverse_logit, blur, downsample
        return inverse_logit, blur, downsample
    except Exception as e:
        logger.error("[WIREUP] inverse_logit/blur/downsample unavailable: %s", e)
        raise


# -----------------------------------------------------------------------------
# Numerics (NSF spline inverse accumulates f32 root error -> score in f64)
# -----------------------------------------------------------------------------
def to_f64(t):
    import torch
    if not torch.is_tensor(t):
        logger.error("[to_f64] expected Tensor, got %s", type(t))
        raise TypeError(f"to_f64 expected Tensor, got {type(t)}")
    return t.double()


def pixel_A(x_pix_flat, blur_sigma, scale):
    """Forward operator on PIXEL-space x (B,784) -> y-space (B,Dy) float64.
    Same blur+downsample _A applies after inverse_logit, so a blended /
    differenced reconstruction is degraded with the SAME operator the per-expert
    rec used. For soft-gate blends and residual differences only."""
    inverse_logit, blur, downsample = _wire_degrade_ops()
    n = x_pix_flat.size(0)
    x2d = to_f64(x_pix_flat).view(n, 1, 28, 28)
    return downsample(blur(x2d, float(blur_sigma)), int(scale)).flatten(1)


# -----------------------------------------------------------------------------
# Cfg / paths / IO
# -----------------------------------------------------------------------------
def read_run_cfg(mixture_ckpt_dir: str) -> dict:
    """Read report.json['cfg'] (dict) of the 2.3-A mixture run for the scalar
    params NWS must reuse: blur_sigma, scale, noise_sigma, rec_z_bank_size,
    rec_z_bank_seed, rec_z_bank_mode, data_root, expert_set."""
    p = os.path.join(mixture_ckpt_dir, "report.json")
    if not os.path.exists(p):
        logger.error("[cfg] no report.json in %s", mixture_ckpt_dir)
        raise FileNotFoundError(p)
    try:
        rep = json.loads(open(p).read())
    except (OSError, json.JSONDecodeError) as e:
        logger.error("[cfg] cannot read %s: %s", p, e)
        raise
    if "cfg" not in rep:
        logger.error("[cfg] report.json missing 'cfg' in %s", mixture_ckpt_dir)
        raise KeyError("report.json missing 'cfg'")
    return rep["cfg"]


def results_dir(seed_index: int) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(here, "results", f"seed_{seed_index}")
    os.makedirs(d, exist_ok=True)
    return d


def plots_dir(seed_index: int) -> str:
    d = os.path.join(results_dir(seed_index), "plots")
    os.makedirs(d, exist_ok=True)
    return d


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_report(path: str, payload: dict) -> str:
    if not path.endswith("_report.json"):
        logger.error("[save_report] report path must end with _report.json: %s", path)
        raise ValueError(f"report path must end with _report.json: {path}")
    payload = {"abbr": __abbr__, "version": __version__,
               "written_utc": _now_iso(), **payload}
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=float)
        os.replace(tmp, path)
    except OSError as e:
        logger.error("[save_report] write failed %s: %s", path, e)
        raise
    logger.info("[save_report] wrote %s", path)
    return path


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s :: %(message)s"))
        root.addHandler(h)
    root.setLevel(level)


# -----------------------------------------------------------------------------
# Verdict (diagnostic; reference = NSF's own per_expert_rec mean, NOT 0.156)
# -----------------------------------------------------------------------------
def classify_verdict(*, first_nonnsf_tau, q4_overlap, class_concentration,
                     oracle_rec, nsf_mean_rec, soft_gate_rec=None) -> dict:
    reasons: list[str] = []

    tau_ok = first_nonnsf_tau is not None and first_nonnsf_tau <= SIGNAL_TAU_MAX
    reasons.append(f"first non-NSF near-win tau={first_nonnsf_tau} "
                   f"({'<=' if tau_ok else '>'} {SIGNAL_TAU_MAX})")

    q4_ok = q4_overlap is not None and q4_overlap > Q4_UNIFORM
    reasons.append(f"NSF-worst-quartile overlap={q4_overlap} "
                   f"({'>' if q4_ok else '<='} uniform {Q4_UNIFORM})")

    clustered = class_concentration is not None and class_concentration > Q4_UNIFORM
    reasons.append(f"class concentration={class_concentration} "
                   f"({'clustered' if clustered else 'scattered'})")

    oracle_ok = (oracle_rec is not None and nsf_mean_rec is not None
                 and oracle_rec < nsf_mean_rec)
    reasons.append(f"oracle_rec={oracle_rec} vs nsf_mean_rec={nsf_mean_rec} "
                   f"(signal-supporting: {oracle_ok})")

    soft_ok = (soft_gate_rec is not None and nsf_mean_rec is not None
               and soft_gate_rec < nsf_mean_rec)
    if soft_gate_rec is not None:
        reasons.append(f"soft-gate dry-run={soft_gate_rec} (beats NSF mean: {soft_ok})")

    structural = tau_ok and q4_ok and clustered
    supporting = oracle_ok or soft_ok
    label = VERDICT_SIGNAL if (structural and supporting) else VERDICT_NO_SIGNAL
    return {
        "label": label, "reasons": reasons,
        "structural_pass": structural, "supporting_pass": supporting,
        "confirmatory": ("oracle/soft-gate are signal-supporting only -- NOT a "
                         "standalone GO. Real GO needs a TRAINED x_final beating "
                         "NSF-only. Reference is NSF's own per_expert_rec mean; "
                         "0.156 is the V2 soft_fwd_rel (different metric)."),
    }
