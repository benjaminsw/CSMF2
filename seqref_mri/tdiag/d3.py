# SEQREF-TDIAG v0.1 -- tdiag.d3
# LIFETIME: KEEP
# =============================================================================
# Purpose: D3 -- conditioner-perturbation sensitivity (EXEC SS10.6 D3,
#          locked 2026-08-15). The trained (step-500) model and the
#          PHYSICAL INVERSE PROBLEM per slice are FROZEN: the slice's own
#          measurement y, coordinate map, target, DC insertion and the
#          decoder are never touched. ONLY the conditioner inputs are
#          perturbed under the locked derangement p(i) = (i+1) mod n
#          (n = 8 = TINY slice count):
#            C0  own image + own mask       (reference; must reproduce
#                                           production EXACTLY)
#            C1  donor image + donor mask   (PRIMARY: the classification
#                                           routes on C1 ONLY)
#            C2  own image + donor mask     (attribution: mask channel)
#            C3  donor image + own mask     (attribution: image channel)
#            C4  neutral conditioner -- OMITTED + RECORDED: the
#                production conditioner interface
#                (seqref_mri/src/conditioner.py, reviewed 2026-08-20)
#                defines NO registered neutral input, so C4 is omitted
#                per the locked spec ("only if the production interface
#                defines one").
#          Sensitivity scores against the REGISTERED reference gains:
#            S_NLL(k)  = |NLL_batch(k) - NLL_batch(C0)| / NLL_GAIN_REF,
#            S_PSNR(k) = |mean delta_PSNR_z0(k vs C0)| / PSNR_GAIN_REF,
#          with NLL_GAIN_REF = 18883.5859375 - (-35316.66015625) and
#          PSNR_GAIN_REF = 32.1537681211221 - 31.533202821914884
#          (D1 E0 minus the registered PSNR anchor) -- single
#          authoritative literals, pinned in tdiag.invariants. Locked
#          bands: S >= 0.25 strong; S <= 0.01 negligible; between
#          weak/mixed. Classification (C1 only): strong iff S_NLL(1) >=
#          0.25 OR S_PSNR(1) >= 0.25; conditioner under-use-consistent
#          iff BOTH <= 0.01 (insensitivity is under-use EVIDENCE, never
#          a verdict); else mixed. Under-use routes per the decision
#          matrix to conditioner/interface redesign => candidate v0.2.
#          C2/C3 are ATTRIBUTION ONLY (descriptive band labels + a pure
#          ordering dominance note; no thresholds, no booleans, no
#          routing fields).
# Gates and invariants:
#   * C0 exactness (source-specific cross-ties, review 2026-08-20): the
#     C0 instrumentation run must EXACTLY reproduce (a) the R0-
#     registered step-500 batch NLL endpoint, (b) the D1 E0 per-slice
#     z=0 PSNR/NMSE_u AND aggregate means, (c) the D1 E1 per-slice
#     posterior-mean PSNR/NMSE_u AND aggregate means -- bitwise, else
#     D3_C0_MISMATCH and NO interpretation;
#   * Z_DIAG bank: reconstructed via the registered constructor
#     (estimators.z_diag_bank) and pinned against the D1 block's
#     bank/manifest sha256, else D3_BANK_MISMATCH;
#   * physical-problem immutability: a canonical fingerprint over every
#     state's target/y/amax/cond/mask/x_true_mag/u_true/cmap/vecs bytes
#     is taken BEFORE and AFTER the measurement; any mutation is
#     D3_STATE_TAMPER;
#   * model immutability: state_hash before/after must equal the
#     R0-registered step500 hash, else D3_STATE_MISMATCH (D3 is
#     step-500-only; it never touches the driver-owned step-0 state);
#   * every measured term must be finite (D3_TERM_NON_FINITE); per-slice
#     encode failures wrap as D3_ENCODE_FAILURE.
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * D3 slice (2026-08-20, under the same SS10.6 lock; NO contract
#     change): module introduced with the locked derangement, the C0-C3
#     conditioned measurement (instrumentation wrapper; ZERO production-
#     code changes), the C0 source-specific exact cross-ties, the bank
#     and immutability gates, the locked-band C1 classification and the
#     C2/C3 attribution-only blocks; C4 omitted + recorded.
# Update summary:
#   v0.1 D3 lands the conditioner-sensitivity diagnostic on the frozen
#   step-500 model: donor-conditioned NLL/z=0/posterior-mean
#   measurements through an instrumentation wrapper over the production
#   decode path, exact C0 ties to the R0/D1 anchors, locked-band
#   sensitivity scores with C1-only routing and descriptive C2/C3
#   attribution.
# =============================================================================
from __future__ import annotations

import hashlib
import logging
import math
import time

import numpy as np
import torch

from seqref_mri.tdiag import _bootstrap  # noqa: F401

from preflight_io import canonical_hash
from preflight_parents import StageError
from seqref_mri.scripts import tiny_gate as tg
from seqref_mri.tdiag import estimators, replay
from seqref_mri.tdiag.d2a import _gaussian_logprob_prod
from seqref_mri.tdiag import invariants as tinv

logger = logging.getLogger("SEQREF-TDIAG")

C4_OMISSION_REASON = ("the production conditioner interface "
                      "(seqref_mri/src/conditioner.py, reviewed "
                      "2026-08-20) defines NO registered neutral input; "
                      "C4 is omitted per EXEC SS10.6 D3 ('only if the "
                      "production interface defines one')")

_CONDITION_DESCRIPTIONS = {
    "C0": "own image + own mask (reference; must reproduce production "
          "EXACTLY)",
    "C1": "donor image + donor mask (PRIMARY -- the classification "
          "routes on C1 only)",
    "C2": "own image + donor mask (attribution: mask channel)",
    "C3": "donor image + own mask (attribution: image channel)"}


def _fail(code: str, message: str, **kwargs) -> StageError:
    logger.error("[SEQREF-TDIAG] %s: %s", code, message)
    return StageError(code, message, **kwargs)


def _verify_state(model, expected: str, label: str) -> str:
    h = replay.state_hash(replay.capture_state(model))
    if h != expected:
        raise _fail("D3_STATE_MISMATCH",
                    f"state-hash mismatch at boundary '{label}': model "
                    f"state {h} != registered {expected}")
    return h


# ---------------------------------------------------------------------------
# Locked derangement + per-condition conditioner-input sources.
# ---------------------------------------------------------------------------

def derangement(n: int) -> list:
    """Frozen derangement p(i) = (i+1) mod n (EXEC SS10.6 D3; production
    n = 8). Frozen in code and pinned by selftest so the donor mapping
    can never silently change (review 2026-08-20)."""
    if n < 2:
        raise _fail("D3_DERANGEMENT_UNDEFINED",
                    f"the locked derangement p(i) = (i+1) mod n needs "
                    f"n >= 2 distinct slices, got n={n}")
    return [(i + 1) % n for i in range(n)]


def _condition_sources(k: int, i: int, p: list) -> tuple:
    """(cond source index, mask source index) for recipient slice i
    under condition k -- the locked input sets: C0 own/own, C1
    donor/donor, C2 own/donor, C3 donor/own."""
    if k == 0:
        return i, i
    if k == 1:
        return p[i], p[i]
    if k == 2:
        return i, p[i]
    if k == 3:
        return p[i], i
    raise _fail("D3_CONDITION_UNKNOWN",
                f"condition index {k} is not one of the locked C0-C3")


# ---------------------------------------------------------------------------
# Instrumentation wrapper: the production decode/encode path with
# explicit (cond, mask) -- donor substitution alters ONLY the
# conditioner inputs. ZERO production-code changes (review 2026-08-20).
# ---------------------------------------------------------------------------

def _batch_tensors_conditioned(states: list, p: list, k: int) -> tuple:
    """The production batch construction (d2b._batch_tensors verbatim
    for the targets) with per-condition cond/mask concatenation."""
    targets = torch.from_numpy(np.concatenate(
        [st["target"] for st in states], axis=0).astype(np.float32))
    cond = torch.cat([states[_condition_sources(k, i, p)[0]]["cond"]
                      for i in range(len(states))], dim=0)
    mask = torch.cat([states[_condition_sources(k, i, p)[1]]["mask"]
                      for i in range(len(states))], dim=0)
    return targets, cond, mask


def _encode_slice_conditioned(model, st: dict, cond, mask) -> tuple:
    """d2b._encode_slice mirrored with explicit conditioner inputs:
    (z_true f32 (13824,), ldj f32 scalar, log_pz f64)."""
    target32 = np.ascontiguousarray(st["target"], dtype=np.float32)
    if target32.shape != (1, tg.ffr.FLOW_DIM_REAL):
        raise _fail("D3_TARGET_SHAPE",
                    f"slice {st['identity']!r}: target shape "
                    f"{tuple(target32.shape)} != (1, "
                    f"{tg.ffr.FLOW_DIM_REAL})")
    with torch.no_grad():
        h = model.condition(cond, mask)
        z_t, ldj_t = model.flow.encode(torch.from_numpy(target32), h)
    z = np.asarray(z_t.detach().cpu().numpy())
    if z.shape != (1, tg.ffr.FLOW_DIM_REAL):
        raise _fail("D3_Z_SHAPE",
                    f"slice {st['identity']!r}: encode returned shape "
                    f"{tuple(z.shape)} != (1, {tg.ffr.FLOW_DIM_REAL})")
    z1 = np.ascontiguousarray(z[0], dtype=np.float32)
    ldj = float(ldj_t.detach().cpu().reshape(-1)[0])
    log_pz = _gaussian_logprob_prod(z1.astype(np.float64))
    if not (bool(np.all(np.isfinite(z1))) and math.isfinite(ldj)
            and math.isfinite(log_pz)):
        raise _fail("D3_TERM_NON_FINITE",
                    f"slice {st['identity']!r}: non-finite z, ldj or "
                    "log_pz under the conditioned encode")
    return z1, ldj, log_pz


def _decode_z_conditioned(model, z, st: dict, cond, mask) -> tuple:
    """tg._decode_z mirrored with explicit conditioner inputs ->
    (COMPLEX image (96,96), unstandardised free vector c128). The
    recipient's own y/amax/cmap/vecs are NEVER substituted -- the DC
    insertion always uses the slice's own measurement."""
    x_hat = tg.ffr.decode_to_image(model, z, cond, mask, st["y"],
                                   st["amax"], st["cmap"], st["vecs"])
    us = model.decode_scalars(z, cond, mask)
    us_np = np.asarray(us.detach().to(torch.float64).cpu().numpy())
    re_s, im_s = tg.ffr.unpack_scalars(us_np)
    u_hat = tg.ffr.unstandardise_free(re_s, im_s, st["cmap"],
                                      st["vecs"])[0]
    return x_hat[0], u_hat


def _decode_bank_conditioned(model, st: dict, bank: np.ndarray,
                             cond, mask) -> np.ndarray:
    """estimators.decode_bank mirrored with explicit conditioner inputs:
    decode all Z_DIAG latents for one slice in ONE batched call ->
    (128, n_free) complex128 physical free vectors."""
    z = torch.from_numpy(bank)
    us = model.decode_scalars(z, cond, mask)
    us_np = np.asarray(us.detach().to(torch.float64).cpu().numpy())
    re_s, im_s = tg.ffr.unpack_scalars(us_np)
    return tg.ffr.unstandardise_free(re_s, im_s, st["cmap"], st["vecs"])


def _pm_metrics(model, st: dict, bank: np.ndarray, cond,
                mask) -> tuple:
    """D1-E1 posterior-mean metrics (the locked u-space convention):
    complex mean of the physical free vectors BEFORE image formation,
    reconstruct once through the production decode_normalised."""
    decodes = _decode_bank_conditioned(model, st, bank, cond, mask)
    if decodes.shape[0] != tinv.Z_DIAG_N:
        raise _fail("D3_BANK_LAYOUT_UNEXPECTED",
                    f"the posterior mean expects {tinv.Z_DIAG_N} shared "
                    f"decodes, got {decodes.shape[0]}")
    u_mean = decodes.mean(axis=0)
    mag = estimators.image_from_u(st, u_mean).abs()
    return (tg._psnr(mag, st["x_true_mag"]),
            tg._nmse(u_mean, st["u_true"]))


# ---------------------------------------------------------------------------
# Physical-problem immutability fingerprint (review 2026-08-20): donor
# substitution may alter ONLY the conditioner inputs of the CALL, never
# the state objects.
# ---------------------------------------------------------------------------

def _tensor_sha(t) -> str:
    arr = (t.detach().cpu().numpy() if isinstance(t, torch.Tensor)
           else np.asarray(t))
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _states_fingerprint(states: list) -> str:
    """Canonical hash over every physical-problem AND conditioner
    tensor's raw bytes per slice, in slice order. Captured before and
    after the full D3 measurement; any mutation is D3_STATE_TAMPER."""
    per = []
    for st in states:
        cmap = st["cmap"]
        per.append({
            "identity": st["identity"],
            "target": _tensor_sha(st["target"]),
            "y": _tensor_sha(st["y"]),
            "amax": _tensor_sha(st["amax"]),
            "cond": _tensor_sha(st["cond"]),
            "mask": _tensor_sha(st["mask"]),
            "x_true_mag": _tensor_sha(st["x_true_mag"]),
            "u_true": _tensor_sha(st["u_true"]),
            "cmap_free_rows": _tensor_sha(cmap.free_rows),
            "cmap_free_cols": _tensor_sha(cmap.free_cols),
            "vecs": canonical_hash({k: _tensor_sha(v)
                                    for k, v in st["vecs"].items()}),
            "excluded": bool(st["excluded"])})
    return canonical_hash(per)


# ---------------------------------------------------------------------------
# Measurement: one condition = batch production NLL + per-slice NLL +
# z=0 and posterior-mean PSNR/NMSE_u, all through the instrumentation
# wrapper.
# ---------------------------------------------------------------------------

def _measure_condition(model, states: list, p: list, k: int,
                       bank: np.ndarray) -> dict:
    label = f"C{k}"
    model.eval()
    targets, cond, mask = _batch_tensors_conditioned(states, p, k)
    nll_batch = tg._nll(model, targets, cond, mask)
    if not math.isfinite(nll_batch):
        raise _fail("D3_TERM_NON_FINITE",
                    f"{label}: the batch production NLL is non-finite "
                    f"({nll_batch!r})")
    dim = tg.ffr.FLOW_DIM_REAL
    z0 = torch.zeros(1, dim)
    per_slice = []
    with torch.no_grad():
        for i, st in enumerate(states):
            cs, ms = _condition_sources(k, i, p)
            cond_i = states[cs]["cond"]
            mask_i = states[ms]["mask"]
            try:
                _z, ldj, log_pz = _encode_slice_conditioned(
                    model, st, cond_i, mask_i)
            except StageError as exc:
                raise _fail("D3_ENCODE_FAILURE",
                            f"{label}, slice {st['identity']!r}: "
                            f"conditioned encode failed "
                            f"({exc.error_code}: {exc.reason})")
            nll = -(log_pz + ldj)
            if not math.isfinite(nll):
                raise _fail("D3_TERM_NON_FINITE",
                            f"{label}, slice {st['identity']!r}: "
                            "per-slice NLL is non-finite")
            x0_c, u0 = _decode_z_conditioned(model, z0, st, cond_i,
                                             mask_i)
            psnr0 = tg._psnr(x0_c.abs(), st["x_true_mag"])
            nmse0 = tg._nmse(u0, st["u_true"])
            psnr_pm, nmse_pm = _pm_metrics(model, st, bank, cond_i,
                                           mask_i)
            per_slice.append({
                "identity": st["identity"],
                "donor_identity": states[p[i]]["identity"],
                "cond_source": "own" if cs == i else "donor",
                "mask_source": "own" if ms == i else "donor",
                "nll": nll,
                "nll_per_dim": nll / dim,
                "z0_psnr": psnr0,
                "z0_nmse_u": nmse0,
                "pm_psnr": psnr_pm,
                "pm_nmse_u": nmse_pm})
    return {"condition": label, "nll_batch": nll_batch,
            "per_slice": per_slice}


# ---------------------------------------------------------------------------
# C0 exactness gates (source-specific cross-ties, review 2026-08-20):
# the C0 instrumentation run must reproduce production EXACTLY.
# ---------------------------------------------------------------------------

def _check_c0_ties(c0: dict, r0: dict, d1: dict) -> dict:
    mismatches = []
    try:
        reg_nll = float(r0["endpoints"]["final"]["nll_batch_mean"])
        e0 = d1["estimators"]["E0"]["per_slice"]
        e1 = d1["estimators"]["E1"]["per_slice"]
        agg0 = d1["aggregate"]["E0"]
        agg1 = d1["aggregate"]["E1"]
    except (KeyError, TypeError, ValueError) as exc:
        raise _fail("D3_ANCHORS_MISSING",
                    f"the R0/D1 blocks lack the registered C0 anchors: "
                    f"{exc}")
    if c0["nll_batch"] != reg_nll:
        mismatches.append(f"batch NLL {c0['nll_batch']!r} != the "
                          f"R0-registered step-500 endpoint {reg_nll!r}")
    if len(c0["per_slice"]) != len(e0) or len(e0) != len(e1):
        raise _fail("D3_C0_MISMATCH",
                    f"C0 slice count {len(c0['per_slice'])} != the D1 "
                    f"E0/E1 slice counts {len(e0)}/{len(e1)}; the slice "
                    "sets must be identical")
    for i, rec in enumerate(c0["per_slice"]):
        if rec["identity"] != e0[i]["identity"]:
            mismatches.append(f"slice {i}: identity drift "
                              f"({rec['identity']} != "
                              f"{e0[i]['identity']})")
            continue
        if rec["z0_psnr"] != e0[i]["psnr"]:
            mismatches.append(f"slice {i}: z0 psnr {rec['z0_psnr']!r} "
                              f"!= D1 E0 {e0[i]['psnr']!r}")
        if rec["z0_nmse_u"] != e0[i]["nmse_u"]:
            mismatches.append(f"slice {i}: z0 nmse_u "
                              f"{rec['z0_nmse_u']!r} != D1 E0 "
                              f"{e0[i]['nmse_u']!r}")
        if rec["pm_psnr"] != e1[i]["psnr"]:
            mismatches.append(f"slice {i}: pm psnr {rec['pm_psnr']!r} "
                              f"!= D1 E1 {e1[i]['psnr']!r}")
        if rec["pm_nmse_u"] != e1[i]["nmse_u"]:
            mismatches.append(f"slice {i}: pm nmse_u "
                              f"{rec['pm_nmse_u']!r} != D1 E1 "
                              f"{e1[i]['nmse_u']!r}")
    mean_z0_psnr = float(np.mean([r["z0_psnr"]
                                  for r in c0["per_slice"]]))
    mean_z0_nmse = float(np.mean([r["z0_nmse_u"]
                                  for r in c0["per_slice"]]))
    mean_pm_psnr = float(np.mean([r["pm_psnr"]
                                  for r in c0["per_slice"]]))
    mean_pm_nmse = float(np.mean([r["pm_nmse_u"]
                                  for r in c0["per_slice"]]))
    if mean_z0_psnr != agg0["mean_psnr"]:
        mismatches.append(f"aggregate z0 mean_psnr {mean_z0_psnr!r} != "
                          f"D1 E0 {agg0['mean_psnr']!r}")
    if mean_z0_nmse != agg0["mean_nmse_u"]:
        mismatches.append(f"aggregate z0 mean_nmse_u {mean_z0_nmse!r} "
                          f"!= D1 E0 {agg0['mean_nmse_u']!r}")
    if mean_pm_psnr != agg1["mean_psnr"]:
        mismatches.append(f"aggregate pm mean_psnr {mean_pm_psnr!r} != "
                          f"D1 E1 {agg1['mean_psnr']!r}")
    if mean_pm_nmse != agg1["mean_nmse_u"]:
        mismatches.append(f"aggregate pm mean_nmse_u {mean_pm_nmse!r} "
                          f"!= D1 E1 {agg1['mean_nmse_u']!r}")
    if mismatches:
        raise _fail("D3_C0_MISMATCH",
                    "the C0 instrumentation run does not EXACTLY "
                    "reproduce the registered production anchors; D3 is "
                    "not measuring the registered path -- NO condition "
                    "is interpreted: " + "; ".join(mismatches),
                    detail={"mismatches": mismatches})
    logger.info("[SEQREF-TDIAG] D3 C0 cross-ties exact: NLL vs R0 "
                "step-500 endpoint, z=0 vs D1 E0, posterior mean vs D1 "
                "E1 (%d slices)", len(c0["per_slice"]))
    return {"rule": "C0 through the instrumentation must reproduce the "
                    "registered production anchors EXACTLY (bitwise): "
                    "batch NLL vs the R0 step-500 endpoint, z=0 vs D1 "
                    "E0 (per-slice AND aggregate), posterior mean vs D1 "
                    "E1 (per-slice AND aggregate)",
            "nll": {"instrumentation": c0["nll_batch"],
                    "registered_r0_step500": reg_nll, "equal": True},
            "z0_vs_d1_e0": {"checked_slices": len(c0["per_slice"]),
                            "aggregate_mean_psnr": mean_z0_psnr,
                            "aggregate_mean_nmse_u": mean_z0_nmse,
                            "equal": True},
            "pm_vs_d1_e1": {"checked_slices": len(c0["per_slice"]),
                            "aggregate_mean_psnr": mean_pm_psnr,
                            "aggregate_mean_nmse_u": mean_pm_nmse,
                            "equal": True}}


# ---------------------------------------------------------------------------
# Aggregation: signed deltas vs C0, locked scores/bands, classification.
# ---------------------------------------------------------------------------

def _band_label(s: float) -> str:
    """Locked bands (EXEC SS10.6 D3): >= 0.25 strong, <= 0.01
    negligible, between weak. Pure label arithmetic."""
    if s >= tinv.D3_BAND_STRONG:
        return "strong"
    if s <= tinv.D3_BAND_NEGLIGIBLE:
        return "negligible"
    return "weak"


def _classify_c1(s_nll: float, s_psnr: float) -> dict:
    """Locked classification on C1 ONLY (EXEC SS10.6 D3): strong iff
    S_NLL(1) >= 0.25 OR S_PSNR(1) >= 0.25; conditioner
    under-use-consistent iff BOTH <= 0.01 (insensitivity is under-use
    EVIDENCE, never a verdict); else mixed. Under-use routes per the
    decision matrix to conditioner/interface redesign => candidate
    v0.2."""
    if (s_nll >= tinv.D3_BAND_STRONG
            or s_psnr >= tinv.D3_BAND_STRONG):
        return {"label": "strong_conditioner_use",
                "note": "C1 sensitivity is strong on at least one "
                        "channel (S >= 0.25): the conditioner inputs "
                        "materially drive the decoded density/"
                        "reconstruction -- evidence the conditioner is "
                        "USED"}
    if (s_nll <= tinv.D3_BAND_NEGLIGIBLE
            and s_psnr <= tinv.D3_BAND_NEGLIGIBLE):
        return {"label": "conditioner_under_use_consistent",
                "note": "C1 sensitivity is negligible on BOTH channels "
                        "(S <= 0.01): strong evidence of conditioner "
                        "UNDER-USE (NOT proof of a specific mechanism); "
                        "per the decision matrix this routes to "
                        "conditioner/interface redesign => candidate "
                        "v0.2"}
    return {"label": "mixed",
            "note": "between the locked 0.01/0.25 bands on C1: mixed "
                    "sensitivity evidence"}


def _sign_counts(vals: np.ndarray) -> dict:
    return {"n_positive": int(np.count_nonzero(vals > 0.0)),
            "n_zero": int(np.count_nonzero(vals == 0.0)),
            "n_negative": int(np.count_nonzero(vals < 0.0))}


def _aggregate_condition(meas: dict, c0: dict) -> dict:
    """Locked aggregation (EXEC SS10.6 D3): batch-mean NLL delta; for
    PSNR/NMSE per-slice signed deltas FIRST, then the arithmetic mean.
    Sign convention: delta = Ck - C0 (the signed EFFECT of the
    perturbation; the absolute value enters the scores)."""
    n = len(meas["per_slice"])
    dnll_batch = float(meas["nll_batch"] - c0["nll_batch"])
    dnll = np.array([meas["per_slice"][i]["nll"]
                     - c0["per_slice"][i]["nll"] for i in range(n)],
                    dtype=np.float64)
    dz0 = np.array([meas["per_slice"][i]["z0_psnr"]
                    - c0["per_slice"][i]["z0_psnr"] for i in range(n)],
                   dtype=np.float64)
    dpm = np.array([meas["per_slice"][i]["pm_psnr"]
                    - c0["per_slice"][i]["pm_psnr"] for i in range(n)],
                   dtype=np.float64)
    s_nll = float(abs(dnll_batch) / tinv.NLL_GAIN_REF)
    s_psnr = float(abs(dz0.mean()) / tinv.PSNR_GAIN_REF)
    return {"condition": meas["condition"],
            "nll_batch": meas["nll_batch"],
            "delta_nll_batch_vs_c0": dnll_batch,
            "mean_delta_nll_per_slice_vs_c0": float(dnll.mean()),
            "mean_delta_z0_psnr_vs_c0": float(dz0.mean()),
            "mean_delta_pm_psnr_vs_c0": float(dpm.mean()),
            "sign_counts": {"delta_nll": _sign_counts(dnll),
                            "delta_z0_psnr": _sign_counts(dz0),
                            "delta_pm_psnr": _sign_counts(dpm)},
            "S_NLL": s_nll,
            "S_PSNR": s_psnr,
            "band_nll": _band_label(s_nll),
            "band_psnr": _band_label(s_psnr)}


def _dominance_note(agg2: dict, agg3: dict) -> str:
    """Pure ordering statements over C2 (mask channel) vs C3 (image
    channel): attribution only, no thresholds, no booleans, never
    routed (review 2026-08-20)."""
    def _ord(metric):
        a, b = agg2[metric], agg3[metric]
        if a == b:
            return f"{metric}: C2 (mask) == C3 (image) ({a!r})"
        bigger, smaller = ("C2 (mask)", "C3 (image)") if a > b else (
            "C3 (image)", "C2 (mask)")
        return (f"{metric}: {bigger} S exceeds {smaller} "
                f"({max(a, b)!r} vs {min(a, b)!r})")
    return ("attribution-only ordering over the locked scores; "
            "descriptive, never routed -- " + "; ".join(
                _ord(m) for m in ("S_NLL", "S_PSNR")))


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def run_d3(ctx, r0: dict, d1: dict) -> dict:
    """Execute the locked D3 conditioner-perturbation diagnostic on the
    ReplayContext (after D2c; the driver has already cleared the step-0
    state_dict -- D3 is step-500-only). d1 is the D1 facts block: the
    C0 cross-ties bind the instrumentation to the registered production
    path. Returns the JSON-serialisable D3 block."""
    t_start = time.perf_counter()
    model = ctx.model
    states = ctx.states
    n = len(states)
    p = derangement(n)

    state_identity = {"pre_measurement_step500": {
        "equal": True,
        "hash": _verify_state(model, r0["step500_state_hash"],
                              "pre-measurement step500")}}
    fp_pre = _states_fingerprint(states)

    bank = estimators.z_diag_bank()
    zd = d1.get("z_diag")
    if not isinstance(zd, dict):
        raise _fail("D3_Z_DIAG_MISSING",
                    "the D1 block carries no z_diag manifest; the "
                    "shared-bank identity cannot be verified")
    if (bank["manifest_sha256"] != zd.get("manifest_sha256")
            or bank["bank_sha256"] != zd.get("bank_sha256")):
        raise _fail("D3_BANK_MISMATCH",
                    "the reconstructed Z_DIAG bank does not match the "
                    "D1 block's registered bank/manifest sha256; D3 "
                    "must use the SAME bank D1 measured")

    meas = [_measure_condition(model, states, p, k, bank["bank"])
            for k in range(4)]
    c0_ties = _check_c0_ties(meas[0], r0, d1)

    state_identity["post_measurement_step500"] = {
        "equal": True,
        "hash": _verify_state(model, r0["step500_state_hash"],
                              "post-measurement step500")}
    fp_post = _states_fingerprint(states)
    if fp_post != fp_pre:
        raise _fail("D3_STATE_TAMPER",
                    "the physical-problem/conditioner state fingerprint "
                    "changed across the D3 measurement; donor "
                    "substitution may alter ONLY the conditioner inputs "
                    "of the call, never the state objects")

    # Per-slice records: recipient + donor identity, all metrics per
    # condition, signed deltas vs C0 (review 2026-08-20).
    per_slice = []
    for i in range(n):
        rec = {"recipient_identity": states[i]["identity"],
               "donor_identity": states[p[i]]["identity"],
               "donor_offset": 1}
        c0r = meas[0]["per_slice"][i]
        rec["C0"] = {kk: c0r[kk] for kk in
                     ("nll", "nll_per_dim", "z0_psnr", "z0_nmse_u",
                      "pm_psnr", "pm_nmse_u")}
        for k in (1, 2, 3):
            r = meas[k]["per_slice"][i]
            rec[f"C{k}"] = {
                "cond_source": r["cond_source"],
                "mask_source": r["mask_source"],
                "nll": r["nll"], "nll_per_dim": r["nll_per_dim"],
                "z0_psnr": r["z0_psnr"], "z0_nmse_u": r["z0_nmse_u"],
                "pm_psnr": r["pm_psnr"], "pm_nmse_u": r["pm_nmse_u"],
                "delta_vs_c0": {
                    "delta_nll": float(r["nll"] - c0r["nll"]),
                    "delta_z0_psnr": float(r["z0_psnr"]
                                           - c0r["z0_psnr"]),
                    "delta_z0_nmse_u": float(r["z0_nmse_u"]
                                             - c0r["z0_nmse_u"]),
                    "delta_pm_psnr": float(r["pm_psnr"]
                                           - c0r["pm_psnr"]),
                    "delta_pm_nmse_u": float(r["pm_nmse_u"]
                                             - c0r["pm_nmse_u"])}}
        per_slice.append(rec)

    aggregates = {f"C{k}": _aggregate_condition(meas[k], meas[0])
                  for k in (1, 2, 3)}
    classification = _classify_c1(aggregates["C1"]["S_NLL"],
                                  aggregates["C1"]["S_PSNR"])

    runtime = {"seconds": time.perf_counter() - t_start,
               "n_slices": n,
               "n_conditions_measured": 4,
               "measure_calls": "4 conditions x (1 batch NLL + n "
                                "encodes + n z=0 decodes + n bank "
                                "decodes of 128)",
               "note": "descriptive provenance, never scientific "
                       "routing evidence"}
    logger.info("[%s] D3 complete: C1 S_NLL=%.4f S_PSNR=%.4f (%s); C2 "
                "%s/%s C3 %s/%s; C0 ties exact; %.1f s",
                "SEQREF-TDIAG", aggregates["C1"]["S_NLL"],
                aggregates["C1"]["S_PSNR"], classification["label"],
                aggregates["C2"]["band_nll"], aggregates["C2"]["band_psnr"],
                aggregates["C3"]["band_nll"], aggregates["C3"]["band_psnr"],
                runtime["seconds"])
    return {
        "spec": "EXEC SS10.6 D3 (SEQREF-TDIAG v0.1, locked 2026-08-15): "
                "conditioner-perturbation sensitivity on the frozen "
                "step-500 model with the physical inverse problem fixed "
                "per slice; ONLY conditioner inputs perturbed",
        "purpose": "does the decoded density/reconstruction actually "
                   "respond to the conditioner inputs? -- conditioner-"
                   "use vs under-use diagnostic of the TRAINED model",
        "routing": "classification on C1 ONLY: strong iff S_NLL(1) >= "
                   "0.25 OR S_PSNR(1) >= 0.25; conditioner "
                   "under-use-consistent iff BOTH <= 0.01 (=> decision "
                   "matrix: conditioner/interface redesign => candidate "
                   "v0.2); else mixed. C2/C3 are attribution-only and "
                   "NEVER route; insensitivity is under-use EVIDENCE, "
                   "never a verdict",
        "sign_convention": {
            "delta": "Ck - C0 per metric (the signed EFFECT of the "
                     "perturbation; the absolute value enters the "
                     "scores)",
            "aggregation": "batch-mean NLL; per-slice FIRST then the "
                           "arithmetic mean for PSNR/NMSE_u",
            "dim": tg.ffr.FLOW_DIM_REAL},
        "derangement": {"rule": "p(i) = (i+1) mod n; n = 8 (the TINY "
                                "slice count); donor of slice i is "
                                "slice p(i)",
                        "map": p},
        "conditions": {**_CONDITION_DESCRIPTIONS,
                       "C4": {"included": False,
                              "reason": C4_OMISSION_REASON}},
        "gain_references": {
            "NLL_GAIN_REF": tinv.NLL_GAIN_REF,
            "PSNR_GAIN_REF": tinv.PSNR_GAIN_REF,
            "source": "registered literals pinned in tdiag.invariants: "
                      "18883.5859375 - (-35316.66015625) (the R0/TINY "
                      "endpoint gain) and 32.1537681211221 - "
                      "31.533202821914884 (D1 E0 minus the registered "
                      "PSNR anchor)"},
        "state_identity": {**state_identity,
                           "rule": "state_hash equality against the "
                                   "R0-registered step500 hash before "
                                   "and after the measurement; D3 is "
                                   "step-500-only and never touches the "
                                   "driver-owned step-0 state"},
        "bank": {"rule": "Z_DIAG reconstructed via the registered "
                         "constructor (estimators.z_diag_bank) and "
                         "pinned against the D1 block's sha256 records",
                 "bank_sha256": bank["bank_sha256"],
                 "manifest_sha256": bank["manifest_sha256"],
                 "equal_to_d1": True},
        "c0_cross_ties": c0_ties,
        "immutability": {"rule": "canonical fingerprint over every "
                                 "state's target/y/amax/cond/mask/"
                                 "x_true_mag/u_true/cmap/vecs bytes "
                                 "before and after the measurement; "
                                 "donor substitution alters ONLY the "
                                 "conditioner inputs of the call",
                         "pre_sha256": fp_pre,
                         "post_sha256": fp_post,
                         "equal": True},
        "c0_reference": {"nll_batch": meas[0]["nll_batch"],
                         "mean_z0_psnr": float(np.mean(
                             [r["z0_psnr"]
                              for r in meas[0]["per_slice"]])),
                         "mean_pm_psnr": float(np.mean(
                             [r["pm_psnr"]
                              for r in meas[0]["per_slice"]]))},
        "conditions_measured": aggregates,
        "classification": {**classification,
                           "routing_condition": "C1 (donor image + "
                                                "donor mask)",
                           "S_NLL": aggregates["C1"]["S_NLL"],
                           "S_PSNR": aggregates["C1"]["S_PSNR"],
                           "band_rule": "S >= 0.25 strong; S <= 0.01 "
                                        "negligible; between weak/"
                                        "mixed (locked, EXEC SS10.6 "
                                        "D3)"},
        "c2_c3_attribution": {
            "rule": "ATTRIBUTION ONLY: C2 isolates the mask channel, "
                    "C3 the image channel; descriptive band labels plus "
                    "a pure ordering dominance note -- no thresholds, "
                    "no booleans, no routing fields (review 2026-08-20)",
            "C2": {"S_NLL": aggregates["C2"]["S_NLL"],
                   "S_PSNR": aggregates["C2"]["S_PSNR"],
                   "band_nll": aggregates["C2"]["band_nll"],
                   "band_psnr": aggregates["C2"]["band_psnr"]},
            "C3": {"S_NLL": aggregates["C3"]["S_NLL"],
                   "S_PSNR": aggregates["C3"]["S_PSNR"],
                   "band_nll": aggregates["C3"]["band_nll"],
                   "band_psnr": aggregates["C3"]["band_psnr"]},
            "dominance_note": _dominance_note(aggregates["C2"],
                                              aggregates["C3"])},
        "per_slice": per_slice,
        "runtime": runtime}
