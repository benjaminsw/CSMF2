# SEQREF-TDIAG v0.1 -- tdiag.d2b
# LIFETIME: KEEP
# =============================================================================
# Purpose: D2b -- signed NLL decomposition at the replayed step-0 and
#          step-500 states (EXEC SS10.6 D2b, locked 2026-08-15). For
#          z_true = f(u_true|c):
#            log p(u|c) = log p_Z(z_true) + log|det J|,
#            NLL        = -log p_Z(z_true) - log|det J|,
#            L_base     = -mean(log p_Z),  L_logdet = -mean(log|det J|),
#            NLL        = L_base + L_logdet,
#            delta X    = X@0 - X@500  (POSITIVE delta = likelihood
#            improvement).
#          A Jacobian-dominated gain (delta_L_logdet >> delta_L_base) is
#          a registered INTERPRETIVE pattern -- evidence, not causal
#          proof. D2b is descriptive-mechanistic: NO band, NO routing,
#          no automatic pattern booleans. The approximate prediction
#          from the D2a geometry (delta_L_base ~ -3.5k, delta_L_logdet
#          ~ +57.7k, delta_NLL ~ +54.2k) is a HUMAN sanity expectation
#          only: it enters NO gate and NO constant of this module (review
#          2026-08-19).
# Gates and invariants:
#   * state-swap identity: the same 4-boundary state_hash verification
#     as D2a against the R0-registered step0/step500 hashes (the step-0
#     state_dict is swapped into the SAME model object; its lifetime is
#     DRIVER-OWNED -- run_d2b never discards it);
#   * D2B_Z_TRUE_DRIFT: per slice x step the re-encoded z_true sha256
#     must equal the D2a-recorded sha256 EXACTLY -- D2a geometry and
#     the D2b decomposition refer to the SAME latent vector;
#   * D2B_SLICE_ORDER_MISMATCH: exact slice identity/order equality
#     between the ReplayContext and the D2a block;
#   * registered endpoint gate: the production NLL (tg._nll, the R0
#     metric engine verbatim) recomputed at both steps must equal the
#     R0-registered endpoint batch-mean NLL EXACTLY (18883.5859375 /
#     -35316.66015625), else D2B_NLL_ENDPOINT_MISMATCH and NO
#     interpretation;
#   * two-level reconciliation (review 2026-08-19): the production NLL
#     and the batch production-terms reduction are both recorded, plus
#     reconciliation_error = NLL_report - NLL_production per step --
#     RECORDED, not gated (no tolerance is frozen before the real code
#     path's numerical bound is characterised);
#   * identity_error = delta_NLL - (delta_L_base + delta_L_logdet) is
#     recorded per slice and in the aggregate.
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * D2b slice (2026-08-19, under the same SS10.6 lock; NO contract
#     change): module introduced with the signed decomposition, the D2a
#     z_true sha cross-tie, the registered-endpoint exact gate, the
#     two-level reconciliation record and the per-slice/aggregate delta
#     blocks (shares + sign counts, descriptive only).
# Update summary:
#   v0.1 D2b lands the likelihood-decomposition measurements on the
#   replay-validated step-0/step-500 states with state-identity, D2a
#   cross-tie and endpoint-exactness gates; descriptive only, no
#   routing.
# =============================================================================
from __future__ import annotations

import logging
import math
import time

import numpy as np
import torch

from seqref_mri.tdiag import _bootstrap  # noqa: F401

from preflight_parents import StageError
from seqref_mri.scripts import tiny_gate as tg
from seqref_mri.tdiag import replay
from seqref_mri.tdiag.d2a import _gaussian_logprob_prod, _z_sha

logger = logging.getLogger("SEQREF-TDIAG")


def _fail(code: str, message: str, **kwargs) -> StageError:
    logger.error("[SEQREF-TDIAG] %s: %s", code, message)
    return StageError(code, message, **kwargs)


def _verify_state(model, expected: str, label: str) -> str:
    h = replay.state_hash(replay.capture_state(model))
    if h != expected:
        raise _fail("D2B_STATE_MISMATCH",
                    f"{label}: live state hash {h[:12]}... != the "
                    f"R0-registered hash {expected[:12]}...; D2b refuses "
                    f"to decompose an unverified model state")
    return h


def _batch_tensors(states: list) -> tuple:
    """The production batch construction, mirrored from the R0 replay:
    targets = concatenated registered targets cast float32, cond/mask
    concatenated along the batch dim."""
    targets = torch.from_numpy(np.concatenate(
        [st["target"] for st in states], axis=0).astype(np.float32))
    cond = torch.cat([st["cond"] for st in states], dim=0)
    mask = torch.cat([st["mask"] for st in states], dim=0)
    return targets, cond, mask


def _encode_slice(model, st: dict) -> tuple:
    """One production encode for slice st -> (z_true f32 (13824,),
    ldj f32 scalar, log_pz f64). Same path as D2a (target cast float32
    exactly like the production training batch; condition through
    model.condition)."""
    target32 = np.ascontiguousarray(st["target"], dtype=np.float32)
    if target32.shape != (1, tg.ffr.FLOW_DIM_REAL):
        raise _fail("D2B_TARGET_SHAPE",
                    f"slice {st['identity']!r}: target shape "
                    f"{tuple(target32.shape)} != (1, "
                    f"{tg.ffr.FLOW_DIM_REAL})")
    with torch.no_grad():
        h = model.condition(st["cond"], st["mask"])
        z_t, ldj_t = model.flow.encode(torch.from_numpy(target32), h)
    z = np.asarray(z_t.detach().cpu().numpy())
    if z.shape != (1, tg.ffr.FLOW_DIM_REAL):
        raise _fail("D2B_Z_TRUE_SHAPE",
                    f"slice {st['identity']!r}: encode returned shape "
                    f"{tuple(z.shape)} != (1, {tg.ffr.FLOW_DIM_REAL})")
    z1 = np.ascontiguousarray(z[0], dtype=np.float32)
    ldj = float(ldj_t.detach().cpu().reshape(-1)[0])
    log_pz = _gaussian_logprob_prod(z1.astype(np.float64))
    if not (bool(np.all(np.isfinite(z1))) and math.isfinite(ldj)
            and math.isfinite(log_pz)):
        raise _fail("D2B_TERM_NON_FINITE",
                    f"slice {st['identity']!r}: non-finite z_true, ldj "
                    f"or log_pz; no decomposition term may be "
                    f"non-finite")
    return z1, ldj, log_pz


def _production_nll(model, states: list) -> tuple:
    """(NLL via the R0 metric engine verbatim, NLL from the batch
    production terms with the production reduction). The two are
    recorded side by side; their difference is the batch-vs-terms
    numerical provenance, never hidden."""
    targets, cond, mask = _batch_tensors(states)
    nll_prod = tg._nll(model, targets, cond, mask)
    with torch.no_grad():
        h = model.condition(cond, mask)
        z_b, ldj_b = model.flow.encode(targets, h)
        terms = tg.ffr._gaussian_logprob(z_b) + ldj_b      # (B,) f32
        nll_terms = float((-terms).mean())
    if not (math.isfinite(nll_prod) and math.isfinite(nll_terms)):
        raise _fail("D2B_TERM_NON_FINITE",
                    f"production NLL non-finite ({nll_prod!r} / "
                    f"{nll_terms!r})")
    return nll_prod, nll_terms


def _measure_step(model, states: list, step_label: str) -> dict:
    """All D2b measurements at one model state: per-slice reporting
    terms (float64 arithmetic over production float32 outputs) plus the
    production NLL pair."""
    per_slice, zs = [], []
    for st in states:
        z1, ldj, log_pz = _encode_slice(model, st)
        zs.append(z1)
        per_slice.append({
            "identity": st["identity"],
            "z_true_sha256": _z_sha(z1),
            "log_pz": float(log_pz),
            "ldj": float(ldj),
            "L_base_contribution": float(-log_pz),
            "L_logdet_contribution": float(-ldj),
            "nll_contribution": float(-log_pz - ldj)})
    nll_prod, nll_terms = _production_nll(model, states)
    return {"step": step_label, "per_slice": per_slice, "z": zs,
            "nll_production": float(nll_prod),
            "nll_from_production_terms": float(nll_terms)}


def _delta_block(v0: float, v500: float) -> float:
    """delta = value@0 - value@500; POSITIVE = improvement."""
    return float(v0 - v500)


def run_d2b(ctx, r0: dict, d2a: dict) -> dict:
    """Execute the locked D2b decomposition on the ReplayContext (after
    D1 and D2a). d2a is the D2a facts block -- the z_true sha256
    cross-tie binds the two diagnoses to the same latent vectors. The
    step-0 state_dict lifetime is driver-owned; run_d2b NEVER discards
    ctx.state0. Returns the JSON-serialisable D2b block."""
    t_start = time.perf_counter()
    model = ctx.model
    model.eval()
    for p in model.parameters():     # idempotent freeze (as run_d2a)
        p.requires_grad_(False)
        p.grad = None

    # Exact slice identity/order equality with the D2a block.
    d2a_ids = [s["identity"] for s in d2a["slices"]]
    ids = [st["identity"] for st in ctx.states]
    if ids != d2a_ids:
        raise _fail("D2B_SLICE_ORDER_MISMATCH",
                    f"the ReplayContext slice order {ids!r} != the D2a "
                    f"block order {d2a_ids!r}; D2b decomposes the "
                    f"identical slices in the identical order only")

    # State-identity invariant (same 4 boundaries as D2a).
    h_pre = _verify_state(model, r0["step500_state_hash"],
                          "pre-swap step-500")
    if ctx.state0 is None:
        raise _fail("D2B_STATE0_MISSING",
                    "the ReplayContext carries no captured step-0 "
                    "state_dict; D2b requires the R0-captured state0 "
                    "(its lifetime is driver-owned; D2a must not have "
                    "discarded it)")
    if replay.state_hash(ctx.state0) != r0["step0_state_hash"]:
        raise _fail("D2B_STATE_MISMATCH",
                    "the captured step-0 state_dict hash != the "
                    "R0-registered step0_state_hash; refusing to load "
                    "an unverified state")

    state500 = replay.capture_state(model)
    try:
        model.load_state_dict(ctx.state0)
    except Exception as exc:  # noqa: BLE001 -- typed boundary
        raise _fail("D2B_STATE_LOAD_FAILED",
                    f"loading the captured step-0 state_dict failed: "
                    f"{type(exc).__name__}: {exc}")
    h_0 = _verify_state(model, r0["step0_state_hash"], "step-0 loaded")
    m0 = _measure_step(model, ctx.states, "step0")

    try:
        model.load_state_dict(state500)
    except Exception as exc:  # noqa: BLE001 -- typed boundary
        raise _fail("D2B_STATE_LOAD_FAILED",
                    f"restoring the step-500 state_dict failed: "
                    f"{type(exc).__name__}: {exc}")
    h_500 = _verify_state(model, r0["step500_state_hash"],
                          "step-500 restored")
    m500 = _measure_step(model, ctx.states, "step500")
    h_post = _verify_state(model, r0["step500_state_hash"],
                           "post-measurement step-500")

    # Registered endpoint gate: the production NLL at both steps must
    # equal the R0-registered (replay-validated) endpoints EXACTLY.
    reg0 = r0["endpoints"]["initial"]["nll_batch_mean"]
    reg500 = r0["endpoints"]["final"]["nll_batch_mean"]
    if m0["nll_production"] != reg0 or m500["nll_production"] != reg500:
        raise _fail(
            "D2B_NLL_ENDPOINT_MISMATCH",
            f"production NLL recomputation ({m0['nll_production']!r} / "
            f"{m500['nll_production']!r}) != the R0-registered endpoints "
            f"({reg0!r} / {reg500!r}); D2b stops before interpretation")

    # D2a cross-tie: every re-encoded z_true must hash-identical to the
    # D2a record (same latent vector, not merely the same slice).
    checked = 0
    for i, st in enumerate(ctx.states):
        for rec, key in ((m0["per_slice"][i], "step0"),
                         (m500["per_slice"][i], "step500")):
            checked += 1
            if rec["z_true_sha256"] != d2a["slices"][i][key][
                    "z_true_sha256"]:
                raise _fail(
                    "D2B_Z_TRUE_DRIFT",
                    f"slice {st['identity']!r} {key}: D2b z_true sha256 "
                    f"{rec['z_true_sha256'][:12]}... != the D2a record "
                    f"{d2a['slices'][i][key]['z_true_sha256'][:12]}...; "
                    f"the decomposition must refer to the SAME latent "
                    f"vector D2a measured")

    # Per-slice deltas + identity errors.
    per_slice = []
    for i in range(len(ctx.states)):
        a, b = m0["per_slice"][i], m500["per_slice"][i]
        d_base = _delta_block(a["L_base_contribution"],
                              b["L_base_contribution"])
        d_ldj = _delta_block(a["L_logdet_contribution"],
                             b["L_logdet_contribution"])
        d_nll = _delta_block(a["nll_contribution"], b["nll_contribution"])
        per_slice.append({
            "identity": a["identity"],
            "step0": {k: a[k] for k in ("z_true_sha256", "log_pz", "ldj",
                                        "L_base_contribution",
                                        "L_logdet_contribution",
                                        "nll_contribution")},
            "step500": {k: b[k] for k in ("z_true_sha256", "log_pz",
                                          "ldj", "L_base_contribution",
                                          "L_logdet_contribution",
                                          "nll_contribution")},
            "delta": {"delta_L_base": d_base,
                      "delta_L_logdet": d_ldj,
                      "delta_NLL": d_nll,
                      "identity_error": float(d_nll
                                              - (d_base + d_ldj))}})

    def _step_agg(m):
        lb = float(np.mean([r["L_base_contribution"]
                            for r in m["per_slice"]]))
        ll = float(np.mean([r["L_logdet_contribution"]
                            for r in m["per_slice"]]))
        return {"L_base": lb, "L_logdet": ll,
                "NLL": float(lb + ll)}

    agg0, agg500 = _step_agg(m0), _step_agg(m500)
    d_base = _delta_block(agg0["L_base"], agg500["L_base"])
    d_ldj = _delta_block(agg0["L_logdet"], agg500["L_logdet"])
    d_nll = _delta_block(agg0["NLL"], agg500["NLL"])
    if d_nll != 0.0:
        shares = {"base_share_of_delta": float(d_base / d_nll),
                  "logdet_share_of_delta": float(d_ldj / d_nll)}
        shares_note = None
    else:
        shares = {"base_share_of_delta": None,
                  "logdet_share_of_delta": None}
        shares_note = ("delta_NLL == 0.0 exactly; the shares are "
                       "undefined and recorded as null (defined edge, "
                       "never a silent skip)")
    sign_counts = {
        "n_slices_delta_base_positive": int(sum(
            s["delta"]["delta_L_base"] > 0 for s in per_slice)),
        "n_slices_delta_base_negative": int(sum(
            s["delta"]["delta_L_base"] < 0 for s in per_slice)),
        "n_slices_delta_logdet_positive": int(sum(
            s["delta"]["delta_L_logdet"] > 0 for s in per_slice)),
        "n_slices_delta_logdet_negative": int(sum(
            s["delta"]["delta_L_logdet"] < 0 for s in per_slice)),
        "n_slices_delta_nll_positive": int(sum(
            s["delta"]["delta_NLL"] > 0 for s in per_slice)),
        "n_slices_delta_nll_negative": int(sum(
            s["delta"]["delta_NLL"] < 0 for s in per_slice))}

    d2b = {
        "spec": "EXEC SS10.6 D2b (SEQREF-TDIAG v0.1, locked 2026-08-15); "
                "evidence only -- no verdict, no routing",
        "purpose": "signed decomposition of the replayed training NLL "
                   "gain into base-density and Jacobian/volume terms",
        "routing": "descriptive_mechanistic_only -- a Jacobian-dominated "
                   "gain is a registered INTERPRETIVE pattern (evidence, "
                   "not causal proof); the locked decision matrix does "
                   "NOT consume D2b and no pattern booleans are emitted",
        "sign_convention": ("log p(u|c) = log p_Z(z) + log|det J|; NLL "
                            "= -log p_Z - log|det J| = L_base + "
                            "L_logdet with L_base = -mean(log p_Z), "
                            "L_logdet = -mean(log|det J|); delta X = "
                            "X@0 - X@500, POSITIVE = likelihood "
                            "improvement"),
        "z_true_rule": d2a["z_true_rule"],
        "z_true_sha256_rule": d2a["z_true_sha256_rule"],
        "state_identity": {
            "rule": "state_hash equality against the R0-registered "
                    "step0/step500 hashes at every swap boundary",
            "pre_swap_step500": {"hash": h_pre, "equal": True},
            "step0_loaded": {"hash": h_0, "equal": True},
            "step500_restored": {"hash": h_500, "equal": True},
            "post_measurement_step500": {"hash": h_post, "equal": True}},
        "d2a_cross_tie": {
            "rule": "per slice x step the re-encoded z_true sha256 must "
                    "equal the D2a record EXACTLY",
            "checked": int(checked), "equal": True},
        "endpoints": {
            "rule": "production NLL (tg._nll, the R0 metric engine "
                    "verbatim) must equal the R0-registered endpoints "
                    "EXACTLY",
            "registered": {"step0": reg0, "step500": reg500},
            "production": {"step0": m0["nll_production"],
                           "step500": m500["nll_production"]},
            "equal": True},
        "production_terms": {
            "rule": "batch production-terms reduction (same tensors, "
                    "precision, batching, reduction order as the "
                    "production NLL path); recorded alongside tg._nll, "
                    "never hidden",
            "nll_from_production_terms": {
                "step0": m0["nll_from_production_terms"],
                "step500": m500["nll_from_production_terms"]},
            "diff_vs_production": {
                "step0": float(m0["nll_from_production_terms"]
                               - m0["nll_production"]),
                "step500": float(m500["nll_from_production_terms"]
                                 - m500["nll_production"])}},
        "per_slice": per_slice,
        "aggregate": {
            "step0": agg0, "step500": agg500,
            "delta": {"delta_L_base": d_base,
                      "delta_L_logdet": d_ldj,
                      "delta_NLL": d_nll,
                      "identity_error": float(d_nll
                                              - (d_base + d_ldj)),
                      **shares},
            "shares_note": shares_note,
            "reconciliation_error": {
                "rule": "NLL_report - NLL_production per step; RECORDED "
                        "(review 2026-08-19: no tolerance frozen before "
                        "the real code path's numerical bound is "
                        "characterised)",
                "step0": float(agg0["NLL"] - m0["nll_production"]),
                "step500": float(agg500["NLL"]
                                 - m500["nll_production"])},
            "sign_counts": sign_counts},
        "runtime": {"note": "descriptive provenance, never scientific "
                            "routing evidence",
                    "seconds": float(time.perf_counter() - t_start),
                    "n_slices": int(len(ctx.states)),
                    "encode_calls": int(2 * len(ctx.states) + 2)}}
    logger.info("[SEQREF-TDIAG] D2b complete: dL_base=%.4f "
                "dL_logdet=%.4f dNLL=%.4f (identity_error %.3g), "
                "endpoint gate exact, cross-tie %d/%d, %.1f s",
                d_base, d_ldj, d_nll,
                d2b["aggregate"]["delta"]["identity_error"], checked,
                2 * len(ctx.states), time.perf_counter() - t_start)
    return d2b
