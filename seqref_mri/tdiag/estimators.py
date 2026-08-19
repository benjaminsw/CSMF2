# SEQREF-TDIAG v0.1 -- tdiag.estimators
# LIFETIME: KEEP
# =============================================================================
# Purpose: D1 -- the estimator-slate diagnosis (EXEC SS10.6, locked
#          2026-08-15 pre-implementation). Operates on the FROZEN step-500
#          runtime handed over from R0 (replay.ReplayContext); the model
#          is never rebuilt and never retrained for D1.
#          Locked slate:
#            * Z_DIAG : PCG64(0), (128, 13824), generated float64, cast
#                       float32; ONE bank shared by all slices.
#            * E0     : z=0 decode; MUST exactly reproduce the R0 step-500
#                       per-slice record BEFORE E1-E4 are interpreted
#                       (any drift => D1_E0_R0_MISMATCH, typed ERROR).
#            * E1     : posterior mean -- complex mean of the 128 decoded
#                       physical free vectors BEFORE image formation;
#                       reconstruct once.
#            * E2     : coordinate-wise re/im median over the SAME 128
#                       decodes; reconstruct once.
#            * E3     : MAP-like multi-start (z=0 + Z_DIAG[0:7]; exactly 8
#                       starts; Adam 200 steps lr 1e-3; maximize
#                       log p_Z(z) + log|det J_f(u(z)|c)|; winner =
#                       highest final density, ties to the lowest start
#                       index; no ground truth enters the optimization).
#            * E4     : oracle multi-start (SAME 8 starts; minimize the
#                       physical-u squared error; diagnostic only, NEVER
#                       a reconstruction or routing estimator).
#            * JVP    : PCG64(2) integers(0,2) -> v = 2b-1, 16 x 13824
#                       float32 Rademacher probes;
#                       q_j = ||J(z=0) v_j||^2 for the physical-u decode
#                       map; J_F_hat = sqrt(mean(q_j)).
#          Aggregation: per-slice PSNR/NMSE_u, then the arithmetic mean
#          over the SAME 8 TINY slices for every estimator -- D1
#          introduces NO new exclusions; a genuine metric-invalidity
#          condition is a typed ERROR, never a silent change of N.
#          Materiality (frozen bands): mean_PSNR >= E0 + 2.0 dB OR
#          mean_NMSE_u <= 0.5 * E0. Decision fields record the locked
#          classification inputs; E4-only improvement creates NO new
#          routing rule. D1 is evidence, never a verdict.
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass.
# Changelog (NEW in v0.1, D1 slice 2026-08-18):
#   * Introduced with the D1 slice under the 2026-08-15 EXEC SS10.6 lock.
#   * Review-repair round (2026-08-18, pre-execution; NO contract
#     change): gradient hygiene -- the handoff model is FROZEN in place
#     at the start of run_d1 (parameters requires_grad_(False), grads
#     cleared); E3/E4 backward then computes z-gradients only, never
#     computes or retains ~256M parameter gradients.
#   * Review-hardened plan (2026-08-18, pre-execution; NO contract
#     change): E0/R0 exact-equivalence guard BEFORE E1-E4; E1/E2 share
#     ONE bank decode pass; E3/E4 record per-start density/error
#     decompositions, z-norms and checkpoint trajectories
#     {0,25,...,200}; deterministic tie-break to the lowest start index;
#     JVP reports all 16 q-values plus min/median/max; runtime and
#     nonfinite instrumentation recorded as descriptive provenance.
# Update summary:
#   v0.1 D1 lands the locked estimator slate on the frozen R0 runtime:
#   the shared Z_DIAG/JVP banks with manifests, E0-E4 with the E0/R0
#   equivalence gate, JVP forward-mode probes, per-slice-then-mean
#   aggregation over the identical slice set, frozen-band materiality
#   and the locked decision fields (usable vs oracle, mismatch /
#   oracle-negative) -- evidence only, no verdict, no routing.
# =============================================================================
from __future__ import annotations

import hashlib
import logging
import time

import numpy as np
import torch
from torch.func import jvp as _torch_jvp

from seqref_mri.tdiag import _bootstrap  # noqa: F401

from preflight_io import canonical_hash
from preflight_parents import StageError
from seqref_mri.scripts import tiny_gate as tg

from seqref_mri.tdiag.invariants import (JVP_N_PROBES, JVP_SEED,
                                         MAP_LR, MAP_N_STARTS, MAP_STEPS,
                                         MAP_TRAJ_CHECKPOINTS,
                                         MATERIAL_NMSE_RATIO,
                                         MATERIAL_PSNR_DELTA_DB,
                                         Z_DIAG_GENERATOR, Z_DIAG_N,
                                         Z_DIAG_SEED)

logger = logging.getLogger("SEQREF-TDIAG")

ESTIMATOR_NAMES = ("E0", "E1", "E2", "E3", "E4")
USABLE_ESTIMATORS = ("E1", "E2", "E3")     # E4 is oracle/diagnostic-only


def _fail(code: str, message: str, **kwargs) -> StageError:
    logger.error("[SEQREF-TDIAG] %s: %s", code, message)
    return StageError(code, message, **kwargs)


# ---------------------------------------------------------------------------
# Locked banks. The recipes below ARE the lock (EXEC SS10.6 D1); manifests
# pin the rule + the raw-byte hash of the realised bank.
# ---------------------------------------------------------------------------

def z_diag_bank() -> dict:
    """Z_DIAG = np.random.Generator(np.random.PCG64(0)).standard_normal(
    size=(128, 13824), dtype=np.float64).astype(np.float32) -- one bank,
    shared by every slice and by E3/E4 start selection (rows 0..6)."""
    bank = np.random.Generator(
        np.random.PCG64(Z_DIAG_SEED)).standard_normal(
            size=(Z_DIAG_N, tg.ffr.FLOW_DIM_REAL),
            dtype=np.float64).astype(np.float32)
    bank_sha = hashlib.sha256(bank.tobytes(order="C")).hexdigest()
    manifest = canonical_hash({
        "rule": "np.random.Generator(np.random.PCG64(0)).standard_normal("
                "size=(128, 13824), dtype=np.float64).astype(np.float32)",
        "generator": Z_DIAG_GENERATOR, "seed": Z_DIAG_SEED,
        "shape": [Z_DIAG_N, tg.ffr.FLOW_DIM_REAL],
        "construction_dtype": "float64", "dtype": "float32",
        "bank_sha256": bank_sha})
    logger.info("[SEQREF-TDIAG] Z_DIAG bank: (%d, %d) f32 manifest=%s",
                Z_DIAG_N, tg.ffr.FLOW_DIM_REAL, manifest[:12])
    return {"bank": bank, "bank_sha256": bank_sha,
            "manifest_sha256": manifest}


def jvp_probes() -> dict:
    """JVP probes = PCG64(2).integers(0, 2, size=(16, 13824)) -> v = 2b-1,
    float32 Rademacher."""
    bits = np.random.Generator(np.random.PCG64(JVP_SEED)).integers(
        0, 2, size=(JVP_N_PROBES, tg.ffr.FLOW_DIM_REAL))
    probes = (2 * bits - 1).astype(np.float32)
    probe_sha = hashlib.sha256(probes.tobytes(order="C")).hexdigest()
    manifest = canonical_hash({
        "rule": "np.random.Generator(np.random.PCG64(2)).integers(0, 2, "
                "size=(16, 13824)); v = 2b - 1 (Rademacher), float32",
        "generator": "PCG64", "seed": JVP_SEED,
        "shape": [JVP_N_PROBES, tg.ffr.FLOW_DIM_REAL], "dtype": "float32",
        "probes_sha256": probe_sha})
    logger.info("[SEQREF-TDIAG] JVP probes: (%d, %d) f32 manifest=%s",
                JVP_N_PROBES, tg.ffr.FLOW_DIM_REAL, manifest[:12])
    return {"probes": torch.from_numpy(probes),
            "probes_sha256": probe_sha, "manifest_sha256": manifest}


# ---------------------------------------------------------------------------
# Decode helpers. decode_u_physical mirrors the u-path of the production
# tg._decode_z EXACTLY (decode_scalars -> float64 unpack -> unstandardise);
# image_from_u is the production scatter/ifft path (decode_normalised)
# applied to an arbitrary physical free vector. E0 itself calls
# tg._decode_z verbatim -- the R0 metric path, not a copy.
# ---------------------------------------------------------------------------

def _decode(model, z, st, counter: dict) -> torch.Tensor:
    counter["n"] = counter.get("n", 0) + int(z.shape[0])
    return model.decode_scalars(z, st["cond"], st["mask"])


def decode_u_physical(model, z, st, counter: dict) -> np.ndarray:
    """One latent -> physical free vector (n_free,) complex128."""
    us = _decode(model, z, st, counter)
    us_np = np.asarray(us.detach().to(torch.float64).cpu().numpy())
    re_s, im_s = tg.ffr.unpack_scalars(us_np)
    return tg.ffr.unstandardise_free(re_s, im_s, st["cmap"], st["vecs"])[0]


def image_from_u(st: dict, u: np.ndarray) -> torch.Tensor:
    """Physical free vector (n_free,) -> COMPLEX image (96,96) through the
    production decode_normalised (measured k retained exactly)."""
    u_t = torch.from_numpy(np.ascontiguousarray(u[None]))
    return tg.dec.decode_normalised(st["y"], st["amax"], u_t,
                                    st["cmap"])[0]


def _phys_affine_torch(st: dict) -> tuple:
    """float32 torch copies of the P4 affine vectors and u_true for the
    differentiable E4/JVP physical-u maps. u_phys = u_s * scale + mean
    per component -- the same affine unstandardise_free applies (f64)."""
    v = st["vecs"]
    s_re = torch.from_numpy(np.ascontiguousarray(v["scale_re"],
                                                 dtype=np.float32))
    m_re = torch.from_numpy(np.ascontiguousarray(v["mean_re"],
                                                 dtype=np.float32))
    s_im = torch.from_numpy(np.ascontiguousarray(v["scale_im"],
                                                 dtype=np.float32))
    m_im = torch.from_numpy(np.ascontiguousarray(v["mean_im"],
                                                 dtype=np.float32))
    ut_re = torch.from_numpy(np.ascontiguousarray(st["u_true"].real,
                                                  dtype=np.float32))
    ut_im = torch.from_numpy(np.ascontiguousarray(st["u_true"].imag,
                                                  dtype=np.float32))
    return s_re, m_re, s_im, m_im, ut_re, ut_im


def _u_phys_torch(us: torch.Tensor, aff: tuple) -> torch.Tensor:
    """Standardised scalars (B, 13824) -> physical-u real layout
    (B, 13824), interleaved exactly like pack_scalars."""
    s_re, m_re, s_im, m_im, _, _ = aff
    out = torch.empty_like(us)
    out[:, 0::2] = us[:, 0::2] * s_re + m_re
    out[:, 1::2] = us[:, 1::2] * s_im + m_im
    return out


# ---------------------------------------------------------------------------
# E0 + the R0 equivalence gate (review-hardened 2026-08-18): recomputed E0
# MUST equal the R0 step-500 replay record per slice EXACTLY -- same
# model object, same production decode path (tg._decode_z), so equality
# is bitwise. Any drift means D1 is not operating on the replay-validated
# state and E1-E4 must not be interpreted: typed ERROR, no continuation.
# ---------------------------------------------------------------------------

def e0_slice(model, st: dict) -> dict:
    z0 = torch.zeros(1, tg.ffr.FLOW_DIM_REAL)
    with torch.no_grad():
        x0_c, u0 = tg._decode_z(model, z0, st)
    return {"identity": st["identity"],
            "psnr": tg._psnr(x0_c.abs(), st["x_true_mag"]),
            "nmse_u": tg._nmse(u0, st["u_true"])}


def check_e0_r0_equivalence(e0_records: list,
                            r0_final_per_slice: list) -> None:
    if len(e0_records) != len(r0_final_per_slice):
        raise _fail("D1_E0_R0_MISMATCH",
                    f"E0 slice count {len(e0_records)} != the R0 step-500 "
                    f"record's {len(r0_final_per_slice)}; D1 is not "
                    f"operating on the replay-validated slice set")
    mismatches = []
    for i, (e, r) in enumerate(zip(e0_records, r0_final_per_slice)):
        if e["identity"] != r["identity"]:
            mismatches.append(f"slice {i}: identity drift "
                              f"({e['identity']} != {r['identity']})")
            continue
        if r.get("nmse_u_z0") is None:
            raise _fail("D1_METRIC_INVALID",
                        f"slice {i}: the registered set contains an "
                        f"R_FREE_MIN-excluded slice (NMSE_u undefined); "
                        f"D1 cannot form the locked per-slice comparison "
                        f"-- typed ERROR, never a silent change of N")
        if e["psnr"] != r["psnr_z0"]:
            mismatches.append(f"slice {i}: psnr {e['psnr']!r} != R0 "
                              f"{r['psnr_z0']!r}")
        if e["nmse_u"] != r["nmse_u_z0"]:
            mismatches.append(f"slice {i}: nmse_u {e['nmse_u']!r} != R0 "
                              f"{r['nmse_u_z0']!r}")
    if mismatches:
        raise _fail("D1_E0_R0_MISMATCH",
                    "recomputed E0 does not EXACTLY reproduce the R0 "
                    "step-500 per-slice record; D1 is operating on a "
                    "different model/state/path -- E1-E4 are NOT "
                    "interpreted: " + "; ".join(mismatches),
                    detail={"mismatches": mismatches})
    logger.info("[SEQREF-TDIAG] E0/R0 equivalence: %d slices exactly "
                "reproduce the R0 step-500 record", len(e0_records))


# ---------------------------------------------------------------------------
# E1 / E2 over ONE shared bank decode pass.
# ---------------------------------------------------------------------------

def decode_bank(model, st: dict, bank: np.ndarray,
                counter: dict) -> np.ndarray:
    """Decode all Z_DIAG latents for one slice in ONE batched call ->
    (128, n_free) complex128 physical free vectors."""
    z = torch.from_numpy(bank)
    us = _decode(model, z, st, counter)
    us_np = np.asarray(us.detach().to(torch.float64).cpu().numpy())
    re_s, im_s = tg.ffr.unpack_scalars(us_np)
    return tg.ffr.unstandardise_free(re_s, im_s, st["cmap"], st["vecs"])


def e1_e2_from_decodes(st: dict, decodes: np.ndarray) -> tuple:
    """E1: complex mean of the physical free vectors BEFORE image
    formation (the locked u-space convention); reconstruct once.
    E2: coordinate-wise re/im median over the SAME decodes; reconstruct
    once."""
    if decodes.shape[0] != Z_DIAG_N:
        raise _fail("D1_BANK_LAYOUT_UNEXPECTED",
                    f"E1/E2 expect {Z_DIAG_N} shared decodes, got "
                    f"{decodes.shape[0]}")
    u_mean = decodes.mean(axis=0)
    mag1 = image_from_u(st, u_mean).abs()
    e1 = {"identity": st["identity"],
          "psnr": tg._psnr(mag1, st["x_true_mag"]),
          "nmse_u": tg._nmse(u_mean, st["u_true"])}
    u_med = (np.median(decodes.real, axis=0)
             + 1j * np.median(decodes.imag, axis=0)).astype(np.complex128)
    mag2 = image_from_u(st, u_med).abs()
    e2 = {"identity": st["identity"],
          "psnr": tg._psnr(mag2, st["x_true_mag"]),
          "nmse_u": tg._nmse(u_med, st["u_true"])}
    return e1, e2


# ---------------------------------------------------------------------------
# E3 / E4 multi-start latent optimization (locked: exactly 8 starts =
# z=0 + Z_DIAG[0:7]; Adam, 200 steps, lr 1e-3; trajectories at
# {0,25,...,200}; winner tie-break to the LOWEST start index).
# ---------------------------------------------------------------------------

def _map_starts(bank: np.ndarray) -> list:
    starts = [("z0", torch.zeros(1, tg.ffr.FLOW_DIM_REAL))]
    starts += [(f"Z_DIAG[{k}]",
                torch.from_numpy(np.ascontiguousarray(bank[k:k + 1])))
               for k in range(MAP_N_STARTS - 1)]
    if len(starts) != MAP_N_STARTS:
        raise _fail("D1_START_SET_INVALID",
                    f"the locked start set is exactly {MAP_N_STARTS} "
                    f"(z=0 + Z_DIAG[0:7]); built {len(starts)}")
    return starts


def _e3_evaluate(model, st: dict, counter: dict):
    """Objective parts for E3: total = log p_Z(z) + log|det J_f(u(z)|c)|.
    log p_Z comes from the production gaussian log-prob; the forward
    log-abs-det comes from the production flow.encode at the DECODED
    u(z) -- the registered full-density objective, not a round-trip
    identity assumption."""
    def evaluate(z: torch.Tensor):
        h = model.condition(st["cond"], st["mask"])
        u_s = _decode(model, z, st, counter)
        _, ldj = model.flow.encode(u_s, h)
        log_pz = tg.ffr._gaussian_logprob(z.to(torch.float32))
        total = log_pz + ldj
        f_pz, f_ldj = float(log_pz[0]), float(ldj[0])
        return total.sum(), {"total": f_pz + f_ldj,
                             "part_a": f_pz, "part_b": f_ldj}
    return evaluate


def _e4_evaluate(model, st: dict, aff: tuple, counter: dict):
    """Objective for E4: squared error in PHYSICAL u against u_true --
    ground truth enters here ONLY (oracle; diagnostic, never routing)."""
    _, _, _, _, ut_re, ut_im = aff

    def evaluate(z: torch.Tensor):
        u_s = _decode(model, z, st, counter)
        u_ph = _u_phys_torch(u_s, aff)
        err = ((u_ph[:, 0::2] - ut_re).pow(2)
               + (u_ph[:, 1::2] - ut_im).pow(2)).sum()
        f_err = float(err)
        return err, {"total": f_err, "part_a": None, "part_b": None}
    return evaluate


def _run_start(evaluate, z_start: torch.Tensor, *,
               maximize: bool) -> dict:
    """One locked start: Adam, MAP_STEPS steps, lr MAP_LR, trajectories
    at MAP_TRAJ_CHECKPOINTS. A non-finite objective marks the start
    non-finite (recorded) and stops it -- recorded, never silent."""
    z = z_start.clone().detach().to(torch.float32).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=MAP_LR)
    with torch.enable_grad():
        _, parts = evaluate(z)
        rec = {"initial_total": parts["total"],
               "initial_part_a": parts["part_a"],
               "initial_part_b": parts["part_b"],
               "initial_z_norm": float(z.detach().norm())}
        traj = {"0": parts["total"]}
        finite = bool(np.isfinite(parts["total"]))
        if finite:
            for step in range(1, MAP_STEPS + 1):
                opt.zero_grad()
                total_t, _ = evaluate(z)
                loss = (-total_t if maximize else total_t).sum()
                if not bool(torch.isfinite(loss)):
                    finite = False
                    break
                loss.backward()
                opt.step()
                if step in MAP_TRAJ_CHECKPOINTS:
                    with torch.no_grad():
                        _, parts = evaluate(z)
                    traj[str(step)] = parts["total"]
                    if not bool(np.isfinite(parts["total"])):
                        finite = False
                        break
    if finite:
        with torch.no_grad():
            _, parts = evaluate(z)
        rec.update({"final_total": parts["total"],
                    "final_part_a": parts["part_a"],
                    "final_part_b": parts["part_b"]})
    else:
        logger.error("[SEQREF-TDIAG] non-finite objective on a %s start; "
                     "the start is recorded non-finite and excluded from "
                     "winner selection", "E3" if maximize else "E4")
        rec.update({"final_total": None, "final_part_a": None,
                    "final_part_b": None})
    rec.update({"finite": bool(finite),
                "final_z_norm": float(z.detach().norm()),
                "trajectory": traj})
    return {"record": rec, "z_final": z.detach()}


def _select_winner(starts: list, maximize: bool) -> int:
    """Highest final density (E3) / lowest final error (E4) among FINITE
    starts; ties resolve to the LOWEST start index (strict comparison
    keeps the incumbent). All-non-finite is a typed ERROR."""
    best = None
    key = "final_total"
    for i, s in enumerate(starts):
        v = s["record"][key]
        if not s["record"]["finite"] or v is None or not np.isfinite(v):
            continue
        if best is None:
            best = i
            continue
        incumbent = starts[best]["record"][key]
        if (maximize and v > incumbent) or (not maximize
                                            and v < incumbent):
            best = i
    if best is None:
        raise _fail("D1_ALL_STARTS_NON_FINITE",
                    "every optimization start went non-finite; no winner "
                    "can be selected and no estimator record can be "
                    "formed -- typed ERROR, never a silent skip")
    return best


def _map_slice(model, st: dict, bank: np.ndarray, *, oracle: bool,
               counter: dict) -> dict:
    """Shared E3/E4 engine. oracle=False -> E3 full-density maximisation;
    oracle=True -> E4 physical-u squared-error minimisation."""
    aff = _phys_affine_torch(st)
    evaluate = (_e4_evaluate(model, st, aff, counter) if oracle
                else _e3_evaluate(model, st, counter))
    starts = []
    for idx, (source, z0) in enumerate(_map_starts(bank)):
        out = _run_start(evaluate, z0, maximize=not oracle)
        out["record"]["start_index"] = idx
        out["record"]["start_source"] = source
        starts.append(out)
    winner = _select_winner(starts, maximize=not oracle)
    records = []
    for i, out in enumerate(starts):
        r = out["record"]
        rec = {"start_index": r["start_index"],
               "start_source": r["start_source"],
               "initial_z_norm": r["initial_z_norm"],
               "final_z_norm": r["final_z_norm"],
               "winner": bool(i == winner),
               "finite": r["finite"],
               "trajectory": r["trajectory"]}
        if oracle:
            rec.update({
                "initial_squared_u_error": r["initial_total"],
                "final_squared_u_error": r["final_total"]})
        else:
            rec.update({
                "initial_total_log_density": r["initial_total"],
                "final_total_log_density": r["final_total"],
                "initial_log_pz": r["initial_part_a"],
                "final_log_pz": r["final_part_a"],
                "initial_logabsdet": r["initial_part_b"],
                "final_logabsdet": r["final_part_b"]})
        records.append(rec)
    with torch.no_grad():
        u_w = decode_u_physical(model, starts[winner]["z_final"], st,
                                counter)
    mag_w = image_from_u(st, u_w).abs()
    return {"identity": st["identity"],
            "starts": records,
            "winner_start_index": int(winner),
            "psnr": tg._psnr(mag_w, st["x_true_mag"]),
            "nmse_u": tg._nmse(u_w, st["u_true"]),
            "nonfinite_count": int(sum(1 for r in records
                                       if not r["finite"]))}


# ---------------------------------------------------------------------------
# JVP sensitivity at z=0: q_j = ||J(0) v_j||^2 for the physical-u decode
# map (float32 forward-mode JVP; no finite-difference tolerance anywhere).
# ---------------------------------------------------------------------------

def jvp_slice(model, st: dict, probes: torch.Tensor,
              counter: dict) -> dict:
    aff = _phys_affine_torch(st)
    z0 = torch.zeros(1, tg.ffr.FLOW_DIM_REAL)

    def u_phys(z):
        return _u_phys_torch(_decode(model, z, st, counter), aff)

    qs = []
    for j in range(JVP_N_PROBES):
        _, tangent = _torch_jvp(u_phys, (z0,), (probes[j:j + 1],))
        q = float(tangent.pow(2).sum())
        if not np.isfinite(q):
            raise _fail("D1_METRIC_NON_FINITE",
                        f"slice JVP probe {j}: q_j is non-finite "
                        f"({q!r}); no fallback is permitted")
        qs.append(q)
    arr = np.asarray(qs, dtype=np.float64)
    return {"identity": st["identity"],
            "q": qs,
            "sqrt_mean_q": float(np.sqrt(arr.mean())),
            "min_q": float(arr.min()),
            "median_q": float(np.median(arr)),
            "max_q": float(arr.max())}


# ---------------------------------------------------------------------------
# Aggregation + decision fields (frozen bands; identical slice set).
# ---------------------------------------------------------------------------

def aggregate_estimators(per_est: dict) -> tuple:
    """Per-slice then arithmetic mean over the IDENTICAL slice set for
    every estimator; materiality against the frozen bands. Returns
    (aggregate, thresholds)."""
    ids0 = [r["identity"] for r in per_est["E0"]]
    for name in ESTIMATOR_NAMES:
        ids = [r["identity"] for r in per_est[name]]
        if ids != ids0:
            raise _fail("D1_SLICE_SET_MISMATCH",
                        f"estimator {name} aggregates a different slice "
                        f"set than E0; D1 compares all estimators over "
                        f"the SAME slices -- a silent subset change is "
                        f"an ERROR, not an exclusion")
    e0_psnr = float(np.mean([r["psnr"] for r in per_est["E0"]]))
    e0_nmse = float(np.mean([r["nmse_u"] for r in per_est["E0"]]))
    aggregate = {}
    for name in ESTIMATOR_NAMES:
        mp = float(np.mean([r["psnr"] for r in per_est[name]]))
        mn = float(np.mean([r["nmse_u"] for r in per_est[name]]))
        by_psnr = bool(mp - e0_psnr >= MATERIAL_PSNR_DELTA_DB)
        by_nmse = bool(mn <= MATERIAL_NMSE_RATIO * e0_nmse)
        aggregate[name] = {
            "mean_psnr": mp,
            "mean_nmse_u": mn,
            "psnr_gain_vs_E0": mp - e0_psnr,
            "nmse_ratio_vs_E0": (float(mn / e0_nmse)
                                 if e0_nmse > 0.0 else None),
            "material_by_psnr": by_psnr,
            "material_by_nmse": by_nmse,
            "material_overall": bool(by_psnr or by_nmse)}
    thresholds = {"E0_mean_psnr": e0_psnr,
                  "E0_plus_2db": e0_psnr + MATERIAL_PSNR_DELTA_DB,
                  "E0_mean_nmse_u": e0_nmse,
                  "E0_half_nmse_u": MATERIAL_NMSE_RATIO * e0_nmse,
                  "psnr_delta_threshold_db": MATERIAL_PSNR_DELTA_DB,
                  "nmse_ratio_threshold": MATERIAL_NMSE_RATIO}
    return aggregate, thresholds


def decision_fields(aggregate: dict) -> dict:
    """Locked D1 classification inputs. E4 is oracle/diagnostic-only and
    never routes; the intermediate case (no usable improvement, oracle
    improvement) is recorded as-is and creates NO new routing rule."""
    usable = any(aggregate[k]["material_overall"]
                 for k in USABLE_ESTIMATORS)
    oracle = bool(aggregate["E4"]["material_overall"])
    return {"usable_estimator_material_improvement": bool(usable),
            "oracle_material_improvement": oracle,
            "estimator_mismatch": bool(usable),
            "oracle_negative": bool(not oracle),
            "note": ("E4 is oracle/diagnostic-only and never routes; "
                     "usable=E1|E2|E3, oracle=E4; the intermediate case "
                     "(usable=False, oracle=True) is informative "
                     "evidence but creates no new routing rule "
                     "(EXEC SS10.6 D1)")}


# ---------------------------------------------------------------------------
# D1 orchestration on the frozen R0 runtime.
# ---------------------------------------------------------------------------

def run_d1(ctx, r0: dict) -> dict:
    """Execute the locked D1 slate on the ReplayContext handed over from
    R0. The handoff model is FROZEN in place (parameters
    requires_grad_(False), grads cleared) -- D1 needs z-gradients only,
    and the model is never trained after this point. E0 runs FIRST and
    must exactly reproduce the R0 step-500 record before E1-E4 are
    computed. Returns the JSON-serialisable D1 block."""
    model = ctx.model
    model.eval()
    # Gradient hygiene (2026-08-18 repair): freeze the handoff model in
    # place BEFORE any E3/E4/JVP work. D1 optimizes/evaluates only z;
    # backward must never compute or retain gradients for the ~256M
    # frozen parameters. z-gradients still propagate through the frozen
    # network. The model is not used for training after this handoff.
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    bank = z_diag_bank()
    probes = jvp_probes()
    counter: dict = {}
    seconds: dict = {}

    t0 = time.perf_counter()
    e0 = [e0_slice(model, st) for st in ctx.states]
    check_e0_r0_equivalence(e0, r0["endpoints"]["final"]["per_slice"])
    seconds["E0"] = time.perf_counter() - t0
    e0_decodes = len(ctx.states)   # decoded through tg._decode_z (R0 path)

    t0 = time.perf_counter()
    e1, e2 = [], []
    for st in ctx.states:
        decodes = decode_bank(model, st, bank["bank"], counter)
        r1, r2 = e1_e2_from_decodes(st, decodes)
        e1.append(r1)
        e2.append(r2)
    seconds["E1_E2_shared_bank"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    e3 = [_map_slice(model, st, bank["bank"], oracle=False,
                     counter=counter) for st in ctx.states]
    seconds["E3"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    e4 = [_map_slice(model, st, bank["bank"], oracle=True,
                     counter=counter) for st in ctx.states]
    seconds["E4"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    jvp = [jvp_slice(model, st, probes["probes"], counter)
           for st in ctx.states]
    seconds["JVP"] = time.perf_counter() - t0

    per_est = {"E0": e0, "E1": e1, "E2": e2, "E3": e3, "E4": e4}
    aggregate, thresholds = aggregate_estimators(per_est)
    decision = decision_fields(aggregate)

    d1 = {
        "spec": "EXEC SS10.6 D1 (SEQREF-TDIAG v0.1, locked 2026-08-15); "
                "evidence only -- no verdict, no routing",
        "z_diag": {"rule": "PCG64(0) standard_normal (128, 13824) "
                           "float64 -> float32; one shared bank",
                   "generator": Z_DIAG_GENERATOR, "seed": Z_DIAG_SEED,
                   "shape": [Z_DIAG_N, tg.ffr.FLOW_DIM_REAL],
                   "dtype": "float32", "construction_dtype": "float64",
                   "bank_sha256": bank["bank_sha256"],
                   "manifest_sha256": bank["manifest_sha256"]},
        "aggregation_rule": ("per-slice PSNR/NMSE_u, then the arithmetic "
                             "mean over the SAME slices for every "
                             "estimator; D1 introduces no exclusions -- "
                             "metric invalidity is a typed ERROR"),
        "materiality_rule": ("mean_PSNR >= E0 + 2.0 dB OR mean_NMSE_u "
                             "<= 0.5 * E0 (frozen bands, EXEC SS10.6)"),
        "e0_r0_equivalence": {
            "rule": "recomputed E0 must equal the R0 step-500 per-slice "
                    "record EXACTLY before E1-E4 are interpreted",
            "checked_slices": len(e0), "equal": True},
        "estimators": {
            "E0": {"label": "z=0 decode (R0 reference)",
                   "per_slice": e0},
            "E1": {"label": "posterior mean: complex mean of the 128 "
                            "physical-u decodes BEFORE image formation; "
                            "reconstruct once",
                   "shared_bank_with": "E2", "per_slice": e1},
            "E2": {"label": "coordinate-wise re/im median over the SAME "
                            "128 physical-u decodes; reconstruct once",
                   "shared_bank_with": "E1", "per_slice": e2},
            "E3": {"label": "MAP-like multi-start representative",
                   "optimizer": {"name": "Adam", "lr": MAP_LR,
                                 "steps": MAP_STEPS,
                                 "starts": MAP_N_STARTS,
                                 "start_set": "z=0 + Z_DIAG[0:7]",
                                 "objective": "maximize log p_Z(z) + "
                                              "log|det J_f(u(z)|c)|",
                                 "winner": "highest final total log "
                                           "density; ties to the lowest "
                                           "start index",
                                 "trajectory_checkpoints":
                                     list(MAP_TRAJ_CHECKPOINTS)},
                   "per_slice": e3},
            "E4": {"label": "oracle multi-start (ground truth enters; "
                            "diagnostic ONLY)",
                   "routing": "diagnostic_only -- never a reconstruction "
                              "or routing estimator",
                   "optimizer": {"name": "Adam", "lr": MAP_LR,
                                 "steps": MAP_STEPS,
                                 "starts": MAP_N_STARTS,
                                 "start_set": "z=0 + Z_DIAG[0:7]",
                                 "objective": "minimize ||u(z) - "
                                              "u_true||^2 (physical u)",
                                 "winner": "lowest final squared-u "
                                           "error; ties to the lowest "
                                           "start index",
                                 "trajectory_checkpoints":
                                     list(MAP_TRAJ_CHECKPOINTS)},
                   "per_slice": e4}},
        "aggregate": aggregate,
        "thresholds": thresholds,
        "decision": decision,
        "jvp": {"map": "Jacobian of the physical-u decode map at z=0 "
                       "(float32 forward-mode JVP; q_j = ||J(0)v_j||^2)",
                "generator": "PCG64", "seed": JVP_SEED,
                "n_probes": JVP_N_PROBES, "dtype": "float32",
                "probes_sha256": probes["probes_sha256"],
                "manifest_sha256": probes["manifest_sha256"],
                "per_slice": jvp,
                "mean_sqrt_mean_q": float(np.mean(
                    [r["sqrt_mean_q"] for r in jvp]))},
        "runtime": {"note": "descriptive provenance, never scientific "
                            "routing evidence",
                    "per_estimator_seconds": {k: float(v) for k, v in
                                              seconds.items()},
                    "number_of_decodes": {
                        "E0_via_r0_path": int(e0_decodes),
                        "estimators_own_decode_calls":
                            int(counter.get("n", 0))},
                    "nonfinite_count": {
                        "E3": int(sum(r["nonfinite_count"] for r in e3)),
                        "E4": int(sum(r["nonfinite_count"] for r in e4))}}}
    logger.info("[SEQREF-TDIAG] D1 complete: usable material "
                "improvement=%s | oracle material improvement=%s "
                "(diagnostic only) | mean PSNR E0=%.6f E1=%.6f E2=%.6f "
                "E3=%.6f E4=%.6f",
                str(decision["usable_estimator_material_improvement"]),
                str(decision["oracle_material_improvement"]),
                aggregate["E0"]["mean_psnr"], aggregate["E1"]["mean_psnr"],
                aggregate["E2"]["mean_psnr"], aggregate["E3"]["mean_psnr"],
                aggregate["E4"]["mean_psnr"])
    return d1
