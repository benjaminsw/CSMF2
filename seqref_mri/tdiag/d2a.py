# SEQREF-TDIAG v0.1 -- tdiag.d2a
# LIFETIME: KEEP
# =============================================================================
# Purpose: D2a -- latent geometry of the TRUE target (EXEC SS10.6 D2a,
#          locked 2026-08-15). Per TINY slice, at the replayed step-0 and
#          step-500 states: z_true = f(u_true|c) through the PRODUCTION
#          encode direction (flow.encode on the registered standardised
#          free-coordinate target st["target"], cast float32 exactly like
#          the production training batch); recorded per step: ||z_true||,
#          ||z_true||^2, log p_Z(z_true), a frozen per-coordinate
#          statistic set, the percentile of log p_Z(z_true) within the
#          Z_DIAG bank log-density distribution and a sha256 over the
#          float32 z_true bytes. Step 0 -> 500 deltas (norm, density,
#          cosine, displacement) and the top-K |Delta z| coordinates are
#          recorded per slice.
#          D2a is DESCRIPTIVE-MECHANISTIC: NO band, NO routing -- the
#          locked decision matrix does not consume D2a, and the named
#          patterns (plausible-but-not-at-origin / improbable /
#          toward-origin-without-reconstruction-gain) are interpretation
#          labels for the report; NO automatic pattern booleans are
#          emitted.
# State-identity invariant (review 2026-08-19): the step-0 measurements
#   run under the CAPTURED R0 step-0 state_dict, swapped into the SAME
#   model object (no second 256M-parameter model). state_hash equality
#   against the R0-registered step0/step500 hashes is verified before
#   the swap, after the load, after the restore and after the last
#   measurement; any drift is a typed D2A_STATE_MISMATCH. The step-0
#   state's lifetime is DRIVER-OWNED (D2b/D2c need it too; the driver
#   clears it after the last D2-family consumer).
# Bank reference: log p_Z over Z_DIAG is IDENTICAL for every slice and
#   both steps (same standard-Gaussian base, same bank as D1) -- computed
#   ONCE. The regenerated bank manifest must equal the D1-recorded one.
# Gaussian identity: log p_Z(z) = -0.5*||z||^2 - d/2*log(2pi) in float64
#   is checked against the production _gaussian_logprob on every bank
#   row and every z_true (GAUSS_LOGPROB_CHECK_TOL).
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * D2a slice (2026-08-19, under the same SS10.6 lock; NO contract
#     change): module introduced with the state-swap identity invariant,
#     the once-computed Z_DIAG density reference, the analytic Gaussian
#     identity check, the frozen coordinate-statistic set, step deltas
#     and the top-K drift record.
#   * D2b slice (2026-08-19, under the same SS10.6 lock; NO contract
#     change): the step-0 state_dict discard moved OUT of run_d2a --
#     the driver owns its lifetime because D2b (and later D2c) swap the
#     same verified state into the same model.
# Update summary:
#   v0.1 D2a lands the true-latent geometry measurements on the replay-
#   validated step-0/step-500 states with full state-identity, bank-
#   identity and base-density invariants; descriptive only, no routing.
# =============================================================================
from __future__ import annotations

import hashlib
import logging
import math
import time

import numpy as np
import torch

from seqref_mri.tdiag import _bootstrap  # noqa: F401

from preflight_parents import StageError
from seqref_mri.scripts import tiny_gate as tg
from seqref_mri.tdiag import replay
from seqref_mri.tdiag.estimators import z_diag_bank
from seqref_mri.tdiag.invariants import (D2A_PERCENTILE_TIE_RULE,
                                         D2A_TOP_K,
                                         GAUSS_LOGPROB_CHECK_TOL)

logger = logging.getLogger("SEQREF-TDIAG")


def _fail(code: str, message: str, **kwargs) -> StageError:
    logger.error("[SEQREF-TDIAG] %s: %s", code, message)
    return StageError(code, message, **kwargs)


# ---------------------------------------------------------------------------
# Base density: production helper (float64 evaluation) vs the analytic
# identity. The two differ only by float64 summation order (~1e-12); a
# real discrepancy (wrong constant, dropped term) is O(1).
# ---------------------------------------------------------------------------

def _gaussian_logprob_analytic(z64: np.ndarray) -> float:
    """log p_Z(z) = -0.5*||z||^2 - d/2*log(2*pi), float64."""
    d = int(z64.shape[-1])
    return float(-0.5 * float(np.sum(z64 * z64))
                 - 0.5 * d * math.log(2.0 * math.pi))


def _gaussian_logprob_prod(z64: np.ndarray) -> float:
    """The PRODUCTION _gaussian_logprob, evaluated on a float64 copy."""
    return float(tg.ffr._gaussian_logprob(
        torch.from_numpy(np.ascontiguousarray(z64, dtype=np.float64))))


def _gaussian_identity_check(vecs_f32: list) -> float:
    """Max |production - analytic| over the given float32 vectors; the
    check FAILS CLOSED beyond GAUSS_LOGPROB_CHECK_TOL."""
    worst = 0.0
    for v in vecs_f32:
        v64 = np.asarray(v, dtype=np.float64)
        diff = abs(_gaussian_logprob_prod(v64)
                   - _gaussian_logprob_analytic(v64))
        worst = max(worst, diff)
    if not math.isfinite(worst) or worst > GAUSS_LOGPROB_CHECK_TOL:
        raise _fail("D2A_GAUSSIAN_IDENTITY_MISMATCH",
                    f"production _gaussian_logprob deviates from the "
                    f"analytic identity by {worst!r} (tolerance "
                    f"{GAUSS_LOGPROB_CHECK_TOL}); the base-density "
                    f"reference is not trustworthy")
    return float(worst)


def _z_sha(z_f32: np.ndarray) -> str:
    """sha256 over the float32 C-order bytes of z_true (13824,)."""
    return hashlib.sha256(np.ascontiguousarray(
        z_f32, dtype=np.float32).tobytes(order="C")).hexdigest()


# ---------------------------------------------------------------------------
# Z_DIAG density reference: computed ONCE for the whole D2a block.
# ---------------------------------------------------------------------------

def _bank_reference(bank: dict) -> tuple:
    """Returns (public_record, logp_vector float64 (128,)). The public
    record carries the full 128-value log-density vector (auditable;
    feeds the ECDF figure) plus norm/logp summaries."""
    b = bank["bank"]                                   # (128, 13824) f32
    b64 = b.astype(np.float64)
    logp = np.asarray(tg.ffr._gaussian_logprob(
        torch.from_numpy(b64)).numpy(), dtype=np.float64)
    norms = np.sqrt(np.sum(b64 * b64, axis=1))
    rec = {"rule": "log p_Z over the D1 Z_DIAG bank, production "
                   "_gaussian_logprob in float64; IDENTICAL across "
                   "slices and steps (same standard-Gaussian base), "
                   "computed once",
           "manifest_sha256": bank["manifest_sha256"],
           "bank_sha256": bank["bank_sha256"],
           "n": int(b.shape[0]),
           "logp_values": [float(v) for v in logp],
           "logp_summary": {"min": float(np.min(logp)),
                            "median": float(np.median(logp)),
                            "max": float(np.max(logp)),
                            "mean": float(np.mean(logp)),
                            "std": float(np.std(logp))},
           "norm_summary": {"median": float(np.median(norms)),
                            "q05": float(np.quantile(norms, 0.05)),
                            "q95": float(np.quantile(norms, 0.95))}}
    return rec, logp


def bank_percentile(logp_obs: float, bank_logp: np.ndarray) -> dict:
    """Fully auditable percentile record: rank_le_count = #{bank <=
    observed}; fraction = rank / n; the tie rule is frozen."""
    n = int(bank_logp.shape[0])
    rank = int(np.sum(bank_logp <= logp_obs))
    frac = rank / n
    return {"bank_n": n,
            "rank_le_count": rank,
            "percentile_fraction": float(frac),
            "percentile_percent": float(100.0 * frac),
            "tie_rule": D2A_PERCENTILE_TIE_RULE}


# ---------------------------------------------------------------------------
# z_true through the production encode direction.
# ---------------------------------------------------------------------------

def z_true_slice(model, st: dict) -> np.ndarray:
    """z_true = f(u_true|c) for one slice: flow.encode on the registered
    standardised target st["target"] (encode_target, B3 contract), cast
    float32 EXACTLY like the production training batch; the condition is
    built through model.condition -- the same encode path the NLL
    objective uses. Returns float32 (13824,)."""
    if "target" not in st:
        raise _fail("D2A_TARGET_MISSING",
                    f"slice state {st.get('identity')!r} carries no "
                    f"'target' (the registered standardised free-"
                    f"coordinate target from encode_target); D2a cannot "
                    f"encode u_true without it")
    target32 = np.ascontiguousarray(st["target"], dtype=np.float32)
    if target32.shape != (1, tg.ffr.FLOW_DIM_REAL):
        raise _fail("D2A_TARGET_SHAPE",
                    f"slice {st['identity']!r}: target shape "
                    f"{tuple(target32.shape)} != (1, "
                    f"{tg.ffr.FLOW_DIM_REAL})")
    u_s = torch.from_numpy(target32)
    with torch.no_grad():
        h = model.condition(st["cond"], st["mask"])
        z_t, _ldj = model.flow.encode(u_s, h)
    z = np.asarray(z_t.detach().cpu().numpy())
    if z.shape != (1, tg.ffr.FLOW_DIM_REAL):
        raise _fail("D2A_Z_TRUE_SHAPE",
                    f"slice {st['identity']!r}: encode returned shape "
                    f"{tuple(z.shape)} != (1, {tg.ffr.FLOW_DIM_REAL})")
    z1 = np.ascontiguousarray(z[0], dtype=np.float32)
    if not bool(np.all(np.isfinite(z1))):
        raise _fail("D2A_Z_TRUE_NON_FINITE",
                    f"slice {st['identity']!r}: z_true contains "
                    f"non-finite coordinates; the true target does not "
                    f"encode through the frozen flow")
    return z1


def coordinate_stats(z64: np.ndarray) -> dict:
    """The frozen D2a per-coordinate statistic set (float64 over the
    float32 z_true values; population std, ddof=0)."""
    return {"mean": float(np.mean(z64)),
            "std": float(np.std(z64)),
            "rms": float(np.sqrt(np.mean(z64 * z64))),
            "mean_abs": float(np.mean(np.abs(z64))),
            "median": float(np.median(z64)),
            "q05": float(np.quantile(z64, 0.05)),
            "q25": float(np.quantile(z64, 0.25)),
            "q75": float(np.quantile(z64, 0.75)),
            "q95": float(np.quantile(z64, 0.95)),
            "min": float(np.min(z64)),
            "max": float(np.max(z64)),
            "max_abs": float(np.max(np.abs(z64)))}


def _step_record(st: dict, z_f32: np.ndarray,
                 bank_logp: np.ndarray) -> dict:
    """One step's D2a record for one slice."""
    z64 = z_f32.astype(np.float64)
    nsq = float(np.sum(z64 * z64))
    logp = _gaussian_logprob_prod(z64)
    return {"identity": st["identity"],
            "z_true_sha256": _z_sha(z_f32),
            "norm_z": float(math.sqrt(nsq)),
            "norm_z_squared": float(nsq),
            "log_pz": float(logp),
            "coordinate_stats": coordinate_stats(z64),
            "percentile": bank_percentile(logp, bank_logp)}


def _delta_record(z0_f32: np.ndarray, z500_f32: np.ndarray,
                  logp0: float, logp500: float) -> tuple:
    """Step 0 -> 500 movement record + the top-K |Delta z| coordinates.
    Returns (record, signed_delta_vector float64). A degenerate zero-norm
    z_true is a typed ERROR (the ratio/cosine would be undefined)."""
    z0 = z0_f32.astype(np.float64)
    z500 = z500_f32.astype(np.float64)
    d = z500 - z0
    n0 = float(math.sqrt(float(np.sum(z0 * z0))))
    n500 = float(math.sqrt(float(np.sum(z500 * z500))))
    if n0 == 0.0 or n500 == 0.0:
        raise _fail("D2A_Z_TRUE_DEGENERATE",
                    f"z_true sits exactly at the origin at step 0 "
                    f"(norm {n0!r}) or step 500 (norm {n500!r}); the "
                    f"norm ratio and cosine are undefined")
    dot = float(np.sum(z0 * z500))
    abs_d = np.abs(d)
    order = np.argsort(-abs_d, kind="stable")[:D2A_TOP_K]
    top_k = [{"coordinate_index": int(j),
              "z_step0": float(z0[j]),
              "z_step500": float(z500[j]),
              "delta": float(d[j]),
              "abs_delta": float(abs_d[j])} for j in order]
    rec = {"delta_norm_z": float(n500 - n0),
           "delta_norm_z_squared": float(float(np.sum(z500 * z500))
                                         - float(np.sum(z0 * z0))),
           "delta_log_pz": float(logp500 - logp0),
           "norm_ratio_500_over_0": float(n500 / n0),
           "cosine_similarity_z0_z500": float(dot / (n0 * n500)),
           "delta_z_l2": float(math.sqrt(float(np.sum(d * d)))),
           "delta_z_rms": float(math.sqrt(float(np.mean(d * d)))),
           "top_k_drift": top_k,
           "top_k_rule": f"top {D2A_TOP_K} coordinates by |z500 - z0|; "
                         "ties to the lower index (stable sort)"}
    return rec, d


# ---------------------------------------------------------------------------
# Orchestration: state-swap invariant -> step-0 measurements -> restore ->
# step-500 measurements -> deltas. The model is NEVER rebuilt and NEVER
# retrained; only its state_dict is swapped under hash verification.
# ---------------------------------------------------------------------------

def _verify_state(model, expected: str, label: str) -> str:
    h = replay.state_hash(replay.capture_state(model))
    if h != expected:
        raise _fail("D2A_STATE_MISMATCH",
                    f"{label}: live state hash {h[:12]}... != the "
                    f"R0-registered hash {expected[:12]}...; D2a refuses "
                    f"to diagnose an unverified model state")
    return h


def run_d2a(ctx, r0: dict, d1: dict) -> dict:
    """Execute the locked D2a measurements on the ReplayContext handed
    over from R0 (after D1). ctx.state0 must carry the captured R0
    step-0 state_dict; its lifetime is DRIVER-OWNED -- run_d2a NEVER
    discards it (D2b/D2c need the same verified state). Returns the
    JSON-serialisable D2a block (no z_true vectors, only statistics,
    hashes and the top-K drift)."""
    t_start = time.perf_counter()
    model = ctx.model
    model.eval()
    # Gradient hygiene: D1 already froze the handoff model; re-apply
    # idempotently so D2a is safe even if ever run standalone. D2a
    # computes NO gradients at all (every measurement is no_grad).
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None

    # Slice-order invariant: D2a measures EXACTLY the D1 slice order.
    d1_ids = [r["identity"] for r in d1["estimators"]["E0"]["per_slice"]]
    ids = [st["identity"] for st in ctx.states]
    if ids != d1_ids:
        raise _fail("D2A_SLICE_ORDER_MISMATCH",
                    f"the ReplayContext slice order {ids!r} != the D1 "
                    f"slice order {d1_ids!r}; D2a aggregates per slice "
                    f"over the identical set only")

    # State-identity invariant, boundary 1: the live model IS the
    # registered step-500 state before any swap.
    h_pre = _verify_state(model, r0["step500_state_hash"],
                          "pre-swap step-500")
    if ctx.state0 is None:
        raise _fail("D2A_STATE0_MISSING",
                    "the ReplayContext carries no captured step-0 "
                    "state_dict; D2a requires the R0-captured state0")
    if replay.state_hash(ctx.state0) != r0["step0_state_hash"]:
        raise _fail("D2A_STATE_MISMATCH",
                    "the captured step-0 state_dict hash != the "
                    "R0-registered step0_state_hash; refusing to load "
                    "an unverified state")

    # Bank reference, computed ONCE; the regenerated bank must be the
    # D1 bank.
    bank = z_diag_bank()
    if bank["manifest_sha256"] != d1["z_diag"]["manifest_sha256"]:
        raise _fail("D2A_BANK_MISMATCH",
                    f"regenerated Z_DIAG manifest "
                    f"{bank['manifest_sha256'][:12]}... != the "
                    f"D1-recorded manifest "
                    f"{d1['z_diag']['manifest_sha256'][:12]}...; the "
                    f"percentile reference must be the D1 bank")
    bank_ref, bank_logp = _bank_reference(bank)
    gauss_worst = _gaussian_identity_check(list(bank["bank"]))

    # Boundary 2: swap in the verified step-0 state.
    state500 = replay.capture_state(model)
    try:
        model.load_state_dict(ctx.state0)
    except Exception as exc:  # noqa: BLE001 -- typed boundary
        raise _fail("D2A_STATE_LOAD_FAILED",
                    f"loading the captured step-0 state_dict failed: "
                    f"{type(exc).__name__}: {exc}")
    h_0 = _verify_state(model, r0["step0_state_hash"], "step-0 loaded")
    z0s, rec0 = [], []
    for st in ctx.states:
        z = z_true_slice(model, st)
        z0s.append(z)
        rec0.append(_step_record(st, z, bank_logp))
    gauss_worst = max(gauss_worst, _gaussian_identity_check(z0s))

    # Boundary 3: restore the registered step-500 state.
    try:
        model.load_state_dict(state500)
    except Exception as exc:  # noqa: BLE001 -- typed boundary
        raise _fail("D2A_STATE_LOAD_FAILED",
                    f"restoring the step-500 state_dict failed: "
                    f"{type(exc).__name__}: {exc}")
    h_500 = _verify_state(model, r0["step500_state_hash"],
                          "step-500 restored")
    z500s, rec500 = [], []
    for st in ctx.states:
        z = z_true_slice(model, st)
        z500s.append(z)
        rec500.append(_step_record(st, z, bank_logp))
    gauss_worst = max(gauss_worst, _gaussian_identity_check(z500s))

    # Boundary 4: the measurements did not mutate the model.
    h_post = _verify_state(model, r0["step500_state_hash"],
                           "post-measurement step-500")

    slices, deltas = [], []
    for i, st in enumerate(ctx.states):
        delta_rec, d_vec = _delta_record(z0s[i], z500s[i],
                                         rec0[i]["log_pz"],
                                         rec500[i]["log_pz"])
        deltas.append(d_vec)
        slices.append({"identity": st["identity"],
                       "step0": rec0[i], "step500": rec500[i],
                       "delta": delta_rec})

    # Global top-K: coordinates with the largest max-over-slices |Delta
    # z|; the matrix carries the SIGNED per-slice deltas (figure 3).
    absmat = np.stack([np.abs(d) for d in deltas], axis=0)
    score = np.max(absmat, axis=0)
    g_idx = np.argsort(-score, kind="stable")[:D2A_TOP_K]
    global_topk = {
        "rule": f"top {D2A_TOP_K} coordinates by max-over-slices "
                f"|z500 - z0|; signed deltas per slice, slice order as "
                f"recorded",
        "coordinate_indices": [int(j) for j in g_idx],
        "delta_matrix": [[float(deltas[i][j]) for j in g_idx]
                         for i in range(len(deltas))]}

    d2a = {
        "spec": "EXEC SS10.6 D2a (SEQREF-TDIAG v0.1, locked 2026-08-15); "
                "evidence only -- no verdict, no routing",
        "purpose": "latent geometry of the TRUE target at the replayed "
                   "step-0 and step-500 states",
        "routing": "descriptive_mechanistic_only -- the locked decision "
                   "matrix does NOT consume D2a; the named patterns are "
                   "interpretation labels for the report and NO "
                   "automatic pattern booleans are emitted",
        "z_true_rule": ("z_true = flow.encode(float32(st['target']), "
                        "condition(cond, mask)) -- the production encode "
                        "direction on the registered standardised target "
                        "(encode_target, B3 contract), float32 cast "
                        "exactly like the production training batch"),
        "z_true_sha256_rule": "sha256 over the float32 C-order bytes of "
                              "z_true (13824,)",
        "state_identity": {
            "rule": "state_hash equality against the R0-registered "
                    "step0/step500 hashes at every swap boundary",
            "pre_swap_step500": {"hash": h_pre, "equal": True},
            "step0_loaded": {"hash": h_0, "equal": True},
            "step500_restored": {"hash": h_500, "equal": True},
            "post_measurement_step500": {"hash": h_post, "equal": True}},
        "bank_reference": bank_ref,
        "gaussian_identity": {
            "rule": "log p_Z(z) = -0.5*||z||^2 - d/2*log(2pi) in "
                    "float64 vs the production _gaussian_logprob",
            "tolerance": GAUSS_LOGPROB_CHECK_TOL,
            "max_abs_diff": float(gauss_worst),
            "vectors_checked": int(bank["bank"].shape[0]
                                   + 2 * len(ctx.states))},
        "slices": slices,
        "global_topk_drift": global_topk,
        "runtime": {"note": "descriptive provenance, never scientific "
                            "routing evidence",
                    "seconds": float(time.perf_counter() - t_start),
                    "n_slices": int(len(ctx.states)),
                    "encode_calls": int(2 * len(ctx.states))}}
    logger.info("[SEQREF-TDIAG] D2a complete: %d slices x 2 steps, "
                "Gaussian identity max|diff| %.3g, state-identity 4/4, "
                "%.1f s", len(ctx.states), gauss_worst,
                time.perf_counter() - t_start)
    return d2a
