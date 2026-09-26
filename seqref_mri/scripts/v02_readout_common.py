# SEQREF-V02C v0.1 -- scripts.v02_readout_common
# LIFETIME: KEEP
"""Neutral shared helpers for the V02 readout-diagnostic chain.

This module exists so the authoritative scientific driver (SEQREF-V02G,
scripts/v02_readout_diag.py) does NOT depend on the V02H selftest
script as a utility library: a production driver must not import
private helpers from a test harness. The helpers here are pure,
side-effect-free, and pinned to the locked V02AMEND v1.0 semantics:

  * the deterministic-backend pin/refusal (CUBLAS explicit contract);
  * the reserved PCG64(4) paired volume bootstrap -- ONE volume-level
    index matrix, identical resampled indices for R0 and Rk, sha-pinned
    against the registered golden value;
  * the observed-statistic / bootstrap-CI separation: the D1/D2
    practical-effect conjuncts test the OBSERVED statistics (mean
    paired difference; ratio of observed arithmetic means); the
    bootstrap supplies ONLY the CI95 bounds. The mean of bootstrap
    replicates is retained as a clearly-labelled diagnostic and is
    NEVER the acceptance input (for the nonlinear ratio statistic the
    replicate mean differs from the observed ratio);
  * the registered two-conjunct D1/D2 acceptance rule.

Locked constants (V02AMEND v1.0, LudWiki id 147; owner-accepted
2026-09-02, never revised after results): D1 mean paired dPSNR >=
+0.50 dB AND CI95 lower > 0; D2 mean NMSE_u ratio <= 0.95 AND CI95
upper < 1.0. Mean thresholds inclusive; CI null exclusions strict.

No fallback, no mock, no placeholder, no silent pass: every failure
path logger.error(...) + typed raise.
"""

import hashlib
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

__abbr__ = "SEQREF-V02C"
__version__ = "0.1"


class V02CError(Exception):
    """Single typed error identity: (error_code, reason)."""

    def __init__(self, error_code: str, reason: str):
        super().__init__(f"{error_code}: {reason}")
        self.error_code = error_code
        self.reason = reason


def _fail(error_code: str, reason: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, error_code, reason)
    raise V02CError(error_code, reason)


# ---------------------------------------------------------------------------
# Locked bootstrap / acceptance constants (V02AMEND v1.0 SS10-SS11).
# ---------------------------------------------------------------------------

BOOTSTRAP_SEED = 4                    # PCG64(4), registry-audited PASS
BOOTSTRAP_B = 10000
BOOTSTRAP_N_VOLUMES = 199
GOLDEN_BOOT_IDX_SHA256 = ("6124ac52087fcbacceb907a32a226ec1d4210dd4ad"
                          "1abcc58da021c1a9a5233c")
D1_PAIRED_PSNR_MIN_DB = 0.50
D1_CI_NULL = 0.0
D2_RATIO_MAX = 0.95
D2_CI_NULL = 1.0

CUBLAS_ACCEPTED = (":4096:8", ":16:8")


# ---------------------------------------------------------------------------
# Deterministic backend pinning (explicit CUBLAS pin/refusal FIRST).
# ---------------------------------------------------------------------------

def pin_deterministic_backend(torch) -> dict:
    """Pin the deterministic execution contract and RECORD the state.

    CUBLAS_WORKSPACE_CONFIG is handled FIRST, before any torch
    mutation: an unset variable is assigned ':4096:8'; a pre-existing
    torch-accepted value (':4096:8' or ':16:8') is kept; anything else
    is a BACKEND_CONFIG_CONFLICT refusal, never a coercion."""
    pre = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if pre is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        cublas, source = ":4096:8", "assigned"
    elif pre in CUBLAS_ACCEPTED:
        cublas, source = pre, "preexisting_accepted"
    else:
        _fail("BACKEND_CONFIG_CONFLICT",
              f"CUBLAS_WORKSPACE_CONFIG={pre!r} is not a torch-accepted "
              f"deterministic value {CUBLAS_ACCEPTED!r}; refusing to "
              f"override a pre-existing environment")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    state = {"num_threads": torch.get_num_threads(),
             "deterministic_algorithms":
                 torch.are_deterministic_algorithms_enabled(),
             "tf32_matmul_allowed":
                 torch.backends.cuda.matmul.allow_tf32,
             "tf32_cudnn_allowed": torch.backends.cudnn.allow_tf32,
             "cudnn_benchmark": torch.backends.cudnn.benchmark,
             "cudnn_deterministic": torch.backends.cudnn.deterministic,
             "cublas_workspace_config": cublas,
             "cublas_source": source,
             "torch_version": torch.__version__,
             "numpy_version": np.__version__}
    logger.info("[%s] deterministic backend pinned: %s", __abbr__, state)
    return state


# ---------------------------------------------------------------------------
# Reserved PCG64(4) paired bootstrap stream (V02AMEND v1.0 SS11).
# ---------------------------------------------------------------------------

def boot_index_matrix() -> np.ndarray:
    """The single and only consumption of the reserved PCG64(4) stream:
    the VOLUME-level paired bootstrap index matrix -- one matrix,
    identical resampled indices applied to R0 and R1 (the locked
    paired-bootstrap semantics). A fresh generator per call keeps the
    stream reproducible; callers draw this matrix exactly once per
    run."""
    return np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, BOOTSTRAP_N_VOLUMES,
        size=(BOOTSTRAP_B, BOOTSTRAP_N_VOLUMES))


def verify_boot_index_pin(idx: np.ndarray) -> None:
    """The drawn index matrix MUST reproduce the registered golden
    sha256; a generator or consumption-order change is an ERROR, never
    a re-pin."""
    if idx.shape != (BOOTSTRAP_B, BOOTSTRAP_N_VOLUMES):
        _fail("BOOTSTRAP_INPUT_INVALID",
              f"index matrix must be ({BOOTSTRAP_B}, "
              f"{BOOTSTRAP_N_VOLUMES}), got {idx.shape}")
    b_sha = hashlib.sha256(idx.tobytes(order="C")).hexdigest()
    if b_sha != GOLDEN_BOOT_IDX_SHA256:
        _fail("STREAM_PIN_MISMATCH",
              f"paired bootstrap index matrix sha256 {b_sha[:12]}... "
              f"!= the embedded golden pin "
              f"{GOLDEN_BOOT_IDX_SHA256[:12]}...; a generator or "
              f"consumption-order change fails the pin, never re-pins")


# ---------------------------------------------------------------------------
# Observed statistics versus bootstrap intervals -- the separation the
# locked D1/D2 rule requires (V02AMEND v1.0 SS10):
#   practical-effect conjunct  <- OBSERVED statistic
#   uncertainty conjunct       <- bootstrap CI95 bound ONLY
# ---------------------------------------------------------------------------

def observed_d1d2(r0: np.ndarray, rk: np.ndarray) -> dict:
    """The OBSERVED D1/D2 statistics on the measured paired arrays:
      D1 observed = mean(rk - r0)            (paired difference)
      D2 observed = mean(rk) / mean(r0)      (ratio of observed means)
    A zero observed r0 mean makes D2 undefined: ERROR, never coerced
    (the V02E denominator guard, mirrored)."""
    r0 = np.asarray(r0, dtype=np.float64)
    rk = np.asarray(rk, dtype=np.float64)
    if r0.shape != (BOOTSTRAP_N_VOLUMES,) or rk.shape != r0.shape:
        _fail("BOOTSTRAP_INPUT_INVALID",
              f"paired arrays must both be ({BOOTSTRAP_N_VOLUMES},), "
              f"got {r0.shape} and {rk.shape}")
    if not np.isfinite(r0).all() or not np.isfinite(rk).all():
        _fail("BOOTSTRAP_INPUT_INVALID",
              "paired arrays contain non-finite values")
    r0_mean = float(r0.mean())
    if r0_mean == 0.0:
        _fail("BOOTSTRAP_DENOMINATOR_ZERO",
              "the observed R0 mean is exactly 0.0; the D2 ratio is "
              "undefined and no coercion is permitted")
    return {"d1_observed_mean_paired_difference": float((rk - r0).mean()),
            "d2_observed_ratio_of_means": float(rk.mean()) / r0_mean}


def paired_bootstrap_ci(r0: np.ndarray, rk: np.ndarray,
                        idx: np.ndarray) -> dict:
    """The bootstrap CI95 bounds for D1/D2 -- intervals ONLY.

    The per-resample statistics are the locked estimators (per-resample
    mean of the paired difference; per-resample ratio of resampled
    arithmetic means) under ONE index matrix applied IDENTICALLY to
    both sides. The replicate means are returned as clearly-labelled
    diagnostics: they are NOT the observed statistics and MUST NOT
    feed the acceptance rule (for the nonlinear ratio statistic the
    replicate mean differs from the observed ratio of means)."""
    r0 = np.asarray(r0, dtype=np.float64)
    rk = np.asarray(rk, dtype=np.float64)
    if r0.shape != (BOOTSTRAP_N_VOLUMES,) or rk.shape != r0.shape:
        _fail("BOOTSTRAP_INPUT_INVALID",
              f"paired arrays must both be ({BOOTSTRAP_N_VOLUMES},), "
              f"got {r0.shape} and {rk.shape}")
    if not np.isfinite(r0).all() or not np.isfinite(rk).all():
        _fail("BOOTSTRAP_INPUT_INVALID",
              "paired arrays contain non-finite values")
    if idx.shape != (BOOTSTRAP_B, BOOTSTRAP_N_VOLUMES):
        _fail("BOOTSTRAP_INPUT_INVALID",
              f"index matrix must be ({BOOTSTRAP_B}, "
              f"{BOOTSTRAP_N_VOLUMES}), got {idx.shape}")
    diff_b = (rk - r0)[idx].mean(axis=1)             # D1, paired
    r0_b = r0[idx].mean(axis=1)
    if np.any(r0_b == 0.0):
        _fail("BOOTSTRAP_DENOMINATOR_ZERO",
              "a resampled R0 mean is exactly 0.0; the D2 ratio is "
              "undefined for that resample and no coercion is "
              "permitted")
    ratio_b = rk[idx].mean(axis=1) / r0_b            # D2, paired idx
    return {"generator": "PCG64", "seed": BOOTSTRAP_SEED,
            "B": int(BOOTSTRAP_B), "unit": "volume", "paired": True,
            "d1_mean_paired_difference": {
                "ci95": [float(v) for v in
                         np.percentile(diff_b, [2.5, 97.5])],
                "replicate_mean_diagnostic_only": float(diff_b.mean())},
            "d2_ratio_of_resampled_means": {
                "ci95": [float(v) for v in
                         np.percentile(ratio_b, [2.5, 97.5])],
                "replicate_mean_diagnostic_only": float(ratio_b.mean())}}


def d1d2_acceptance(d1: dict, d2: dict) -> dict:
    """The registered V02AMEND acceptance rule (owner-accepted
    2026-09-02), TWO conjuncts per statistic -- the practical-effect
    threshold on the OBSERVED statistic and the uncertainty threshold
    on the bootstrap CI bound, never one substituted for the other:
      D1 pass <=> observed mean paired dPSNR >= +0.50 dB
                  AND CI95 lower > 0
      D2 pass <=> observed mean ratio <= 0.95
                  AND CI95 upper < 1.0
    (mean thresholds inclusive; CI null exclusions strict). The inputs
    must carry the OBSERVED statistic under "mean" and the bootstrap
    interval under "ci95"."""
    d1_mean, d1_lo = float(d1["mean"]), float(d1["ci95"][0])
    d2_mean, d2_hi = float(d2["mean"]), float(d2["ci95"][1])
    d1_practical = d1_mean >= D1_PAIRED_PSNR_MIN_DB
    d1_uncertain = d1_lo > D1_CI_NULL
    d2_practical = d2_mean <= D2_RATIO_MAX
    d2_uncertain = d2_hi < D2_CI_NULL
    return {"d1": {"observed": d1_mean, "ci95_lower": d1_lo,
                   "practical_effect_ok": d1_practical,
                   "uncertainty_ok": d1_uncertain,
                   "pass": d1_practical and d1_uncertain},
            "d2": {"observed": d2_mean, "ci95_upper": d2_hi,
                   "practical_effect_ok": d2_practical,
                   "uncertainty_ok": d2_uncertain,
                   "pass": d2_practical and d2_uncertain}}


# ---------------------------------------------------------------------------
# Changelog
#
# v0.1 (2026-09-11): initial neutral shared-helpers module for the V02
#   readout chain -- factored so the authoritative V02G driver never
#   imports from the V02H selftest harness. Deterministic-backend
#   pin/refusal; reserved PCG64(4) one-matrix paired bootstrap with the
#   registered golden sha pin; observed-statistic / bootstrap-CI
#   separation (acceptance consumes OBSERVED statistics, intervals
#   only; replicate means diagnostic-only); registered two-conjunct
#   D1/D2 acceptance. Constants locked by V02AMEND v1.0 (id 147).
