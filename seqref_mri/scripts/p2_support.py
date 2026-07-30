# SEQREF-P2SUP v0.1 -- P2 measured-support validity (EXEC v0.4 §8, A3)
# LIFETIME: KEEP
#
# Purpose
#   Verify, per slice, that the residual Delta x = x_norm - x0_prepared carries
#   no measured-support energy beyond numerical tolerance, and that the
#   prepared zero-filled image is the object the specification says it is.
#
# ORDERING (A3, load-bearing)
#   Delta x DEPENDS on x0_prepared, so the x0 contract assertion runs FIRST.
#   Computing support statistics from a Delta x whose x0 has not been asserted
#   would let an ERROR-class defect contaminate the scientific quantities
#   before being classified. The order is:
#     1. prepare y, x_norm, cond_in, amax, ops
#     2. independently reconstruct x0
#     3. assert the reconstruction agrees with cond_in
#     4. only then set x0_prepared = cond_in
#     5. Delta x = x_norm - x0_prepared
#     6. run the support gate
#
# x0 CONSTRUCTION (A3 LOCK 2, pipeline-matched)
#   The live contract exposes RAW y and the per-volume divisor amax; there is
#   NO persisted object named y_norm. train_base._prepare builds x0 as
#     x0_c = ops[i].A_adjoint(y)          # adjoint on RAW y
#     cond_in = complex_to_two_channel(x0_c) / a
#   so the independent assembly replicates that OPERATION ORDER -- adjoint on
#   raw y, divide in the IMAGE domain -- rather than the linearity-equivalent
#   rearrangement F^H(M(y/a)). It uses the locked primitives (fastmri_data
#   ifft2c and the column mask) but NOT A_adjoint, which is the
#   x0-construction helper A3 forbids reusing.
#
# MEASUREMENT STATE (recorded, A3)
#   fastmri_data.__getitem__ stores y ALREADY MASKED (y = k96 * mask), so
#   applying M again in the reconstruction is IDEMPOTENT, not semantically
#   necessary. It is applied anyway so the assembly states its own support.
#
# DTYPE PATH (A3, by the EXISTING registered rule)
#   _prepare returns float32, so the complete relevant operator path does not
#   genuinely run in float64 and the FLOAT32 thresholds govern the verdict. The
#   float64-operator result over the SAME prepared inputs is computed and
#   recorded as NON-BLOCKING operator-path sensitivity evidence. Agreement
#   shows robustness to operator precision ON THESE INPUTS; it does NOT show
#   the quantity is input-limited.
#
# Verdict semantics (A3): PASS -> facts, exit 0. BLOCK (a DATA premise failed:
#   degenerate k_i, or a support-condition failure) -> valid facts published
#   FIRST, then exit 1. ERROR (code/specification wrong, including the x0
#   contract mismatch) -> distinctly identified error record, exit 2.
#
# CONVENTION: logger.error + raise on every failure path. No fallback, no mock,
#   no placeholder, no silent pass.
#
# Changelog
#   v0.1 (2026-07-30) Created under Amendment A3. The near-zero switch replaces
#     BOTH ordinary ratio tests, since both denominators are residual-derived
#     and unstable in the same regime; equality at R_RESID_MIN enters the
#     near-zero branch. relative_max divides by max|F dx|; k_i is reserved to
#     the near-zero branch. The x0 assertion precedes every Delta x quantity.

from __future__ import annotations

import argparse
import logging
import os
import resource
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

from preflight_parents import (EXIT_BLOCK, EXIT_ERROR, EXIT_PASS,  # noqa: E402
                              REQUIRED_PREPARE_KEYS, StageBlock, StageError,
                              attach_semantic_hash, environment_record,
                              guard_run_mode, hash_project_code, publish_error,
                              publish_stage, require_finite, verify_parents)
from seqref_mri.src.fastmri_data import (CELL_HW, FastMRISliceDataset,  # noqa
                                         fft2c, ifft2c)
from seqref_mri.scripts.train_base import _collate, _prepare  # noqa: E402

SCRIPT_ID = "SEQREF-P2SUP"
SCRIPT_VERSION = "v0.1"
FACTS_SCHEMA = "seqref-p2-facts/1"
FACTS_PREFIX = "support_facts"
ERROR_PREFIX = "support_error"
SMOKE_FACTS_PREFIX = "smoke_support_facts"
SMOKE_ERROR_PREFIX = "smoke_support_error"

# EXEC v0.4 §13 -- registered, no value introduced here.
R_RESID_MIN = 1e-10               # BLOCKING switch: ||F dx||^2 / S_ref^2
R_X0_MIN = 1e-10                  # ERROR: x0 denominator floor
RHO_M_MAX_F32 = 1e-8              # ordinary rho_M bound, float32 path
RHO_M_MAX_F64 = 1e-10             # ordinary rho_M bound, genuine float64 only
REL_MAX_MAX = 1e-5                # ordinary relative-max bound
ABS_LEAK_F32 = 1e-7               # near-zero absolute leakage, x k_i
ABS_LEAK_F64 = 1e-10              # near-zero absolute leakage, x k_i
X0_ASSERT_RTOL = 1e-6             # ERROR: x0_rel_error bound
P2_BOUNDARY_BAND_DECADES = 1      # NON-VERDICT diagnostic band
# NON-VERDICT diagnostic floor for the operator-path RELATIVE difference. A
# bare abs(a-b)/b denominator is unstable when b is near zero: a harmless 1e-12
# absolute difference at rho ~ 1e-13 would emit a ~10x "relative difference"
# into a facts record, and a diagnostic that misleads is worse than none. This
# constant is NON-BLOCKING and must be registered in EXEC v0.4 §13 alongside
# the others before the authoritative run.
P2_PATH_DIFF_REL_FLOOR = 1e-12

# The registered FLOAT64 rule (EXEC §8 P2): float64 is reported only if the
# COMPLETE relevant operator path genuinely runs in float64. _prepare returns
# float32, so it does not, and the float32 pair governs the verdict.
VERDICT_DTYPE_PATH = "float32"
RHO_M_MAX = RHO_M_MAX_F32
ABS_LEAK = ABS_LEAK_F32
DTYPE_PATH_RULE = (
    "EXEC v0.4 §8 P2 FLOAT64 rule: float64 is reported only if the COMPLETE "
    "relevant operator path genuinely runs in float64. train_base._prepare "
    "returns float32, so it does not; the float32 thresholds "
    "(RHO_M_MAX_F32, ABS_LEAK_F32) govern the verdict. The float64-operator "
    "result over the SAME prepared inputs is non-blocking sensitivity "
    "evidence about OPERATOR precision only.")

logger = logging.getLogger(SCRIPT_ID)


# ---------------------------------------------------------------------------
# Pure operator helpers (unit-testable without the dataset)
# ---------------------------------------------------------------------------

def _mask_k(k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """M applied to a (H, W) k-space tensor with the locked (W,) column mask,
    exactly as MaskedFourierOperator._m does: plain multiply after casting the
    boolean mask to the k-space dtype, broadcasting over rows."""
    if mask.dtype != torch.bool:
        raise StageError("MASK_DTYPE",
                         f"mask dtype must be bool, got {mask.dtype}")
    if mask.dim() != 1 or mask.shape[-1] != k.shape[-1]:
        raise StageError("MASK_SHAPE",
                         f"mask must be 1-D of width {k.shape[-1]}, got "
                         f"{tuple(mask.shape)}")
    return k * mask.to(device=k.device, dtype=k.dtype)


def reconstruct_x0(y_raw: torch.Tensor, mask: torch.Tensor, amax: float,
                   *, dtype: torch.dtype) -> torch.Tensor:
    """Independent, pipeline-matched x0: adjoint on RAW y, then divide in the
    IMAGE domain. Returns a (2, H, W) real tensor.

    Does NOT call _prepare or MaskedFourierOperator.A_adjoint. Uses the locked
    primitive ifft2c and the locked column mask.
    """
    if amax <= 0.0 or not np.isfinite(amax):
        raise StageError("AMAX_INVALID",
                         f"per-volume divisor is {amax!r}; it must be finite "
                         f"and strictly positive (no fallback)")
    y = y_raw.to(dtype)
    x0_c = ifft2c(_mask_k(y, mask))              # image domain, RAW units
    two = torch.stack([x0_c.real, x0_c.imag], dim=-3)
    return two / amax                             # D2: divide in image domain


def x0_discrepancy(x0_recon: np.ndarray, x0_prepared: np.ndarray) -> dict:
    """Per-pixel COMPLEX MAGNITUDE of the difference, and the relative figure
    against max_pixel |x0_recon|."""
    d = x0_recon - x0_prepared
    mag = np.sqrt(d[0] ** 2 + d[1] ** 2)
    abs_err = float(mag.max())
    denom = float(np.sqrt(x0_recon[0] ** 2 + x0_recon[1] ** 2).max())
    require_finite({"x0_abs_error": abs_err, "x0_denominator": denom},
                   "P2 x0 discrepancy")
    if denom <= 0.0:
        raise StageError("X0_DENOMINATOR_NON_POSITIVE",
                         "max_pixel |x0_recon| is not strictly positive; the "
                         "relative error is undefined")
    flat = int(np.argmax(mag))
    return {"x0_abs_error": abs_err, "x0_rel_error": abs_err / denom,
            "x0_recon_max_magnitude": denom,
            "x0_worst_pixel": [int(flat // mag.shape[-1]),
                               int(flat % mag.shape[-1])]}


def support_quantities(dx_c: torch.Tensor, x_true_c: torch.Tensor,
                       mask: torch.Tensor, *, dtype: torch.dtype) -> dict:
    """All P2 residual quantities on one slice, at the given operator dtype.
    Reductions are performed in NumPy float64 regardless of operator dtype."""
    k = fft2c(dx_c.to(dtype))
    mk = _mask_k(k, mask)
    kt = fft2c(x_true_c.to(dtype))
    mkt = _mask_k(kt, mask)

    kn = k.detach().cpu().numpy().astype(np.complex128, copy=False)
    mkn = mk.detach().cpu().numpy().astype(np.complex128, copy=False)
    mktn = mkt.detach().cpu().numpy().astype(np.complex128, copy=False)

    e_k = float((np.abs(kn) ** 2).sum())
    e_mk = float((np.abs(mkn) ** 2).sum())
    max_k = float(np.abs(kn).max())
    max_mk = float(np.abs(mkn).max())
    k_i = float(np.abs(mktn).max())
    require_finite({"E_Fdx": e_k, "E_MFdx": e_mk, "max_Fdx": max_k,
                    "max_MFdx": max_mk, "k_i": k_i},
                   "P2 support quantities")
    return {"E_Fdx": e_k, "E_MFdx": e_mk, "max_Fdx": max_k,
            "max_MFdx": max_mk, "k_i": k_i}


def classify_slice(q: dict, s_ref_sq: float) -> dict:
    """Per-slice verdict structure (A3 LOCK 1).

    The near-zero branch REPLACES BOTH ordinary ratio checks. Equality at
    R_RESID_MIN enters the near-zero branch.
    """
    ratio = q["E_Fdx"] / s_ref_sq
    require_finite({"residual_energy_ratio": ratio}, "P2 residual ratio")
    k_i = q["k_i"]
    if not np.isfinite(k_i) or k_i <= 0.0:
        return {"residual_energy_ratio": ratio, "branch": None,
                "k_i_degenerate": True, "passed": False}

    out = {"residual_energy_ratio": ratio, "k_i_degenerate": False,
           "absolute_allowance": ABS_LEAK * k_i}
    if ratio > R_RESID_MIN:
        out["branch"] = "ordinary"
        # max|F dx| == 0 cannot arise here: the energy switch fires first.
        if q["max_Fdx"] <= 0.0:
            raise StageError("ORDINARY_BRANCH_ZERO_MAX",
                             f"max|F dx| is {q['max_Fdx']!r} on the ORDINARY "
                             f"branch (residual ratio {ratio!r} > "
                             f"{R_RESID_MIN}); the energy switch should have "
                             f"fired first")
        rho_m = q["E_MFdx"] / q["E_Fdx"]
        rel_max = q["max_MFdx"] / q["max_Fdx"]
        require_finite({"rho_M": rho_m, "relative_max": rel_max},
                       "P2 ordinary-branch ratios")
        out.update({
            "rho_M": rho_m, "relative_max": rel_max,
            "ordinary_allowance": REL_MAX_MAX * q["max_Fdx"],
            "rho_M_applicable": True, "relative_max_applicable": True,
            "absolute_leakage_applicable": False,
            "rho_M_pass": bool(rho_m <= RHO_M_MAX),
            "relative_max_pass": bool(rel_max <= REL_MAX_MAX),
            "absolute_leakage_pass": None,
            "diagnostic_only": [],
        })
        out["passed"] = out["rho_M_pass"] and out["relative_max_pass"]
    else:
        out["branch"] = "near_zero"
        rho_m = (q["E_MFdx"] / q["E_Fdx"]) if q["E_Fdx"] > 0.0 else None
        rel_max = (q["max_MFdx"] / q["max_Fdx"]) if q["max_Fdx"] > 0.0 else None
        out.update({
            "rho_M": rho_m, "relative_max": rel_max,
            "ordinary_allowance": (REL_MAX_MAX * q["max_Fdx"]
                                   if q["max_Fdx"] > 0.0 else None),
            "rho_M_applicable": False, "relative_max_applicable": False,
            "absolute_leakage_applicable": True,
            "rho_M_pass": None, "relative_max_pass": None,
            "absolute_leakage_pass": bool(q["max_MFdx"]
                                          <= ABS_LEAK * q["k_i"]),
            "diagnostic_only": ["rho_M", "relative_max"],
            "diagnostic_note": "recorded for diagnosis; on a near-zero slice "
                               "these do NOT affect the verdict because both "
                               "denominators are residual-derived and "
                               "unstable in the same regime",
        })
        out["passed"] = out["absolute_leakage_pass"]

    # NON-VERDICT boundary diagnostic. Operands must be finite and strictly
    # positive, so a zero residual-energy ratio falls OUTSIDE the band by
    # construction -- intended, and stated rather than derived.
    if ratio > 0.0 and np.isfinite(ratio):
        decades = float(abs(np.log10(ratio / R_RESID_MIN)))
        out["boundary_distance_decades"] = decades
        out["boundary_band_member"] = bool(decades
                                           <= P2_BOUNDARY_BAND_DECADES)
    else:
        out["boundary_distance_decades"] = None
        out["boundary_band_member"] = False
    if out["boundary_band_member"]:
        oa, aa = out.get("ordinary_allowance"), out["absolute_allowance"]
        out["allowance_ratio"] = (oa / aa if (oa is not None and aa > 0.0)
                                  else None)
        out["allowance_ratio_note"] = (
            "NON-VERDICT diagnostic. It does not alter P2 and does not block "
            "downstream stages. Any concern about an observed disparity "
            "requires an EXPLICIT SPECIFICATION AMENDMENT and must never be "
            "read retrospectively as changing a completed P2 verdict.")
    return out


# ---------------------------------------------------------------------------

def _peak_rss_bytes() -> int:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru * 1024) if sys.platform.startswith("linux") else int(ru)


def _evaluate(parents: dict, data_root: str, batch: int,
              smoke: int | None) -> list[dict]:
    ds = FastMRISliceDataset(data_root, split="train", mode="eval")
    if len(ds) != parents["p0s"]["population_size"]:
        raise StageError("POPULATION_CHANGED",
                         f"dataset now holds {len(ds)} slices but P0S froze "
                         f"its subset against "
                         f"{parents['p0s']['population_size']}")
    indices = parents["subset_indices"]
    if smoke is not None:
        indices = indices[:smoke]
        logger.warning("SMOKE MODE: %d of %d frozen indices; NOT "
                       "authoritative", len(indices), parents["subset_size"])
    if not indices:
        raise StageError("EMPTY_SUBSET", "the frozen subset selection is empty")
    torch.set_num_threads(1)
    loader = DataLoader(Subset(ds, indices), batch_size=batch, shuffle=False,
                        num_workers=0, collate_fn=_collate)
    s2 = parents["s_ref_squared"]

    rows: list[dict] = []
    for b in loader:
        p = _prepare(b, "cpu", test0=False)
        missing = [k for k in REQUIRED_PREPARE_KEYS if k not in p]
        if missing:
            raise StageError("PREPARE_CONTRACT_CHANGED",
                             f"_prepare() returned no {missing}")
        y, x_norm, cond_in = p["y"], p["x_norm"], p["cond_in"]
        amax = p["amax"]
        if cond_in.shape != x_norm.shape or x_norm.shape[1] != 2:
            raise StageError("STATE_SHAPE_UNEXPECTED",
                             f"x_norm {tuple(x_norm.shape)} / cond_in "
                             f"{tuple(cond_in.shape)} are not the expected "
                             f"two-channel states")
        masks = b["mask"]
        for j, meta in enumerate(b["meta"]):
            m = masks[j]
            a = float(amax[j].item())
            row = {
                "file": meta["file"],
                "slice_index": int(meta["slice_index"]),
                "split": meta["split"], "mode": meta["mode"],
                "epoch": None, "test0": False,
                "mask_seed": int(meta["mask_seed"]),
                "mask_n_columns": int(m.sum().item()),
                "mask_width": int(m.shape[-1]),
                "mask_indices_sha_source": "mask_seed (canonical_mask_seed) "
                                           "with the exact selected column "
                                           "indices recorded below",
                "mask_selected_columns": [int(c) for c in
                                          torch.nonzero(m).flatten().tolist()],
                "file_attr_max": a,
            }

            # --- STEP 2/3: independent x0, ASSERTED BEFORE any Delta x ------
            prep0 = cond_in[j].detach().cpu().numpy().astype(np.float64,
                                                             copy=False)
            r32 = reconstruct_x0(y[j], m, a, dtype=torch.complex64)
            r32n = r32.detach().cpu().numpy().astype(np.float64, copy=False)
            d32 = x0_discrepancy(r32n, prep0)
            r64 = reconstruct_x0(y[j], m, a, dtype=torch.complex128)
            r64n = r64.detach().cpu().numpy().astype(np.float64, copy=False)
            d64 = x0_discrepancy(r64n, prep0)
            dpath = x0_discrepancy(r64n, r32n)   # DIRECT fp64-vs-fp32
            e_x0 = float((r32n ** 2).sum())
            require_finite({"E_x0_recon": e_x0}, "P2 x0 energy")
            row.update({
                "x0_prepared_source_key": "cond_in",
                "x0_recon_energy_over_S_ref_sq": e_x0 / s2,
                "x0_abs_error": d32["x0_abs_error"],
                "x0_rel_error": d32["x0_rel_error"],
                "x0_worst_pixel": d32["x0_worst_pixel"],
                "x0_abs_error_f64": d64["x0_abs_error"],
                "x0_rel_error_f64": d64["x0_rel_error"],
                "x0_worst_pixel_f64": d64["x0_worst_pixel"],
                # PROXY: difference of two scalar max-errors, each measured
                # against x0_prepared and possibly at DIFFERENT pixels. By the
                # reverse triangle inequality it LOWER-BOUNDS the direct
                # discrepancy and can read near zero while the two
                # reconstructions differ pointwise. Kept for continuity; it is
                # NOT the operator-path sensitivity measure.
                "x0_prepared_error_abs_difference_between_paths":
                    abs(d64["x0_abs_error"] - d32["x0_abs_error"]),
                # DIRECT operator-path sensitivity: max complex magnitude of
                # (x0_recon_f64 - x0_recon_f32), and its scale-relative form
                # against max|x0_recon_f64|.
                "x0_path_abs_difference": dpath["x0_abs_error"],
                "x0_path_rel_difference": dpath["x0_rel_error"],
                "x0_path_worst_pixel": dpath["x0_worst_pixel"],
            })
            if e_x0 / s2 <= R_X0_MIN:
                raise StageError(
                    "X0_RECONSTRUCTION_CONTRACT_MISMATCH",
                    f"||x0_recon||^2 / S_ref^2 = {e_x0 / s2!r} <= {R_X0_MIN}; "
                    f"the relative-error denominator has no footing",
                    detail=row)
            if not (d32["x0_rel_error"] <= X0_ASSERT_RTOL):
                raise StageError(
                    "X0_RECONSTRUCTION_CONTRACT_MISMATCH",
                    f"independently reconstructed x0 differs from the "
                    f"prepared cond_in by a relative {d32['x0_rel_error']:.6e} "
                    f"> {X0_ASSERT_RTOL} on slice {meta['file']}#"
                    f"{meta['slice_index']}; the prepared object is not what "
                    f"the specification says it is. This is a CONTRACT fault, "
                    f"not a data verdict.", detail=row)

            # --- STEP 4/5/6: only now form Delta x and the support gate -----
            x0_prepared = cond_in[j]
            dx = x_norm[j] - x0_prepared
            dx_c = torch.complex(dx[0], dx[1])
            xt_c = torch.complex(x_norm[j][0], x_norm[j][1])
            q32 = support_quantities(dx_c, xt_c, m, dtype=torch.complex64)
            q64 = support_quantities(dx_c, xt_c, m, dtype=torch.complex128)
            v = classify_slice(q32, s2)
            rho64 = (q64["E_MFdx"] / q64["E_Fdx"]) if q64["E_Fdx"] > 0.0 \
                else None
            row.update(q32)
            row.update(v)
            row.update({
                "rho_M_f64_diagnostic": rho64,
                "operator_path_abs_difference":
                    (abs(rho64 - v["rho_M"])
                     if (rho64 is not None and v.get("rho_M") is not None)
                     else None),
                # Symmetric, floored denominator: max(|a|, |b|, floor).
                "operator_path_rel_difference":
                    (abs(rho64 - v["rho_M"])
                     / max(abs(rho64), abs(v["rho_M"]),
                           P2_PATH_DIFF_REL_FLOOR)
                     if (rho64 is not None and v.get("rho_M") is not None)
                     else None),
                "operator_path_rel_denominator_rule":
                    "max(|rho_M_f64|, |rho_M_f32|, P2_PATH_DIFF_REL_FLOOR)",
                "max_MFdx_f64": q64["max_MFdx"],
                "k_i_f64": q64["k_i"],
                "verdict_dtype_path": VERDICT_DTYPE_PATH,
            })
            rows.append(row)
    if len(rows) != len(indices):
        raise StageError("SUBSET_SIZE_MISMATCH",
                         f"collected {len(rows)} entries, expected "
                         f"{len(indices)}")
    for k, idx in enumerate(indices):
        rows[k]["dataset_index"] = int(idx)
    return rows


def _gate(rows: list[dict]) -> None:
    degen = [r for r in rows if r.get("k_i_degenerate")]
    if degen:
        f = degen[0]
        raise StageBlock(
            "LEAKAGE_REFERENCE_DEGENERATE",
            f"{len(degen)} slice(s) have k_i = max|M F x_true| not finite and "
            f"strictly positive; no epsilon and no substitute reference is "
            f"permitted", observed=f.get("k_i"), threshold="k_i > 0",
            first_failing={"dataset_index": f["dataset_index"],
                           "file": f["file"],
                           "slice_index": f["slice_index"],
                           "k_i": f.get("k_i")}, n_failing=len(degen))
    failing = [r for r in rows if not r.get("passed")]
    if failing:
        f = failing[0]
        raise StageBlock(
            "MEASURED_SUPPORT_INVALID",
            f"{len(failing)} slice(s) fail the measured-support conditions on "
            f"the {VERDICT_DTYPE_PATH} verdict path; the residual carries "
            f"measured-support energy beyond the registered tolerance",
            observed={"rho_M": f.get("rho_M"),
                      "relative_max": f.get("relative_max"),
                      "max_MFdx": f.get("max_MFdx"),
                      "absolute_allowance": f.get("absolute_allowance"),
                      "branch": f.get("branch")},
            threshold={"RHO_M_MAX": RHO_M_MAX, "REL_MAX_MAX": REL_MAX_MAX,
                       "ABS_LEAK": ABS_LEAK},
            first_failing={"dataset_index": f["dataset_index"],
                           "file": f["file"],
                           "slice_index": f["slice_index"]},
            n_failing=len(failing))


def compute_margins(rows: list[dict]) -> dict:
    """NON-VERDICT threshold margins, oriented so a value near 1 means
    the population only just satisfies the condition. Pure, so the
    null / unbounded / not_applicable branches are testable without a
    dataset."""
    def _mx(key, only=None):
        v = [r[key] for r in rows
             if isinstance(r.get(key), float)
             and (only is None or r.get(only) is True)]
        return max(v) if v else None

    def _margin(num, den):
        """Returns (value, status). null is never used to mean two things:
        the status field always says WHY a value is absent."""
        if den is None:
            return None, "not_applicable"
        if den == 0.0:
            return None, "unbounded"
        return num / den, "finite"

    nz = [r for r in rows if r.get("absolute_leakage_applicable") is True]
    leak = [(r["absolute_allowance"] / r["max_MFdx"]) for r in nz
            if isinstance(r.get("max_MFdx"), float) and r["max_MFdx"] > 0.0]
    n_leak_unbounded = sum(1 for r in nz if r.get("max_MFdx") == 0.0)
    if not nz:
        leak_status = "not_applicable"
    elif not leak:
        leak_status = "fully_unbounded"
    elif n_leak_unbounded:
        leak_status = "partly_unbounded"
    else:
        leak_status = "finite"
    m_x0, s_x0 = _margin(X0_ASSERT_RTOL, _mx("x0_rel_error"))
    m_rho, s_rho = _margin(RHO_M_MAX, _mx("rho_M", "rho_M_applicable"))
    m_rel, s_rel = _margin(REL_MAX_MAX,
                           _mx("relative_max", "relative_max_applicable"))
    return {
        "definition": "margin = distance to the registered threshold, "
                      "expressed so that a value near 1 means the population "
                      "only just satisfies the condition. NON-VERDICT: "
                      "computed from quantities already recorded, and "
                      "inspected during smoke review.",
        "undefined_rule": "a margin whose denominator is zero or absent is "
                          "recorded as null, NEVER as JSON infinity or NaN. "
                          "The companion *_status field states WHY, so a null "
                          "never has to be disambiguated from branch counts.",
        "status_vocabulary": ["finite", "unbounded", "not_applicable"],
        "x0_contract": m_x0, "x0_contract_status": s_x0,
        "x0_contract_formula": "X0_ASSERT_RTOL / max_i(x0_rel_error_i)",
        "ordinary_rho_M": m_rho, "ordinary_rho_M_status": s_rho,
        "ordinary_rho_M_formula": "RHO_M_MAX_F32 / max_i(rho_M_i over "
                                  "ordinary/applicable slices)",
        "relative_max": m_rel, "relative_max_status": s_rel,
        "relative_max_formula": "REL_MAX_MAX / max_i(relative_max_i over "
                                "ordinary/applicable slices)",
        "near_zero_leakage": (min(leak) if leak else None),
        "near_zero_leakage_status": leak_status,
        "near_zero_leakage_status_vocabulary": ["finite", "partly_unbounded",
                                                "fully_unbounded",
                                                "not_applicable"],
        "near_zero_leakage_formula": "min_i[(ABS_LEAK_F32 * k_i) / "
                                     "max|M F dx_i|] over applicable "
                                     "near-zero slices with POSITIVE leakage",
        "near_zero_leakage_applicable_slices": len(nz),
        "near_zero_leakage_unbounded_slices": n_leak_unbounded,
    }


def _build_facts(parents, rows, verdict, reason, block, repo_dir, script,
                 argv, t0, smoke) -> dict:
    thresholds = {"R_RESID_MIN": R_RESID_MIN, "R_X0_MIN": R_X0_MIN,
                  "RHO_M_MAX_F32": RHO_M_MAX_F32,
                  "RHO_M_MAX_F64": RHO_M_MAX_F64,
                  "REL_MAX_MAX": REL_MAX_MAX,
                  "ABS_LEAK_F32": ABS_LEAK_F32, "ABS_LEAK_F64": ABS_LEAK_F64,
                  "X0_ASSERT_RTOL": X0_ASSERT_RTOL,
                  "P2_BOUNDARY_BAND_DECADES": P2_BOUNDARY_BAND_DECADES,
                  "P2_PATH_DIFF_REL_FLOOR": P2_PATH_DIFF_REL_FLOOR,
                  "S_ref": parents["s_ref"],
                  "S_ref_squared": parents["s_ref_squared"]}

    def _worst(key, largest=True):
        cand = [r for r in rows if isinstance(r.get(key), (int, float))]
        if not cand:
            return None
        r = (max if largest else min)(cand, key=lambda z: z[key])
        return {"dataset_index": r["dataset_index"], "file": r["file"],
                "slice_index": r["slice_index"], key: r[key]}

    k_vals = [r["k_i"] for r in rows if isinstance(r.get("k_i"), float)]
    rho_vals = [r["rho_M"] for r in rows
                if isinstance(r.get("rho_M"), float)]
    margins = compute_margins(rows)
    summary = {
        "n_slices": len(rows), "smoke": smoke is not None,
        "margins": margins,
        "branch_counts": {
            "ordinary": sum(1 for r in rows if r.get("branch") == "ordinary"),
            "near_zero": sum(1 for r in rows if r.get("branch") == "near_zero")},
        "failure_counts": {
            "rho_M": sum(1 for r in rows if r.get("rho_M_pass") is False),
            "relative_max": sum(1 for r in rows
                                if r.get("relative_max_pass") is False),
            "absolute_leakage": sum(1 for r in rows
                                    if r.get("absolute_leakage_pass") is False),
            "k_i_degenerate": sum(1 for r in rows
                                  if r.get("k_i_degenerate"))},
        "worst_rho_M": _worst("rho_M"),
        "worst_relative_max": _worst("relative_max"),
        "worst_x0_rel_error": _worst("x0_rel_error"),
        "worst_x0_rel_error_f64": _worst("x0_rel_error_f64"),
        "worst_operator_path_abs_difference":
            _worst("operator_path_abs_difference"),
        "worst_x0_path_abs_difference": _worst("x0_path_abs_difference"),
        "worst_x0_path_rel_difference": _worst("x0_path_rel_difference"),
        "min_k_i": (min(k_vals) if k_vals else None),
        "median_k_i": (float(np.median(k_vals)) if k_vals else None),
        "median_rho_M": (float(np.median(rho_vals)) if rho_vals else None),
        "boundary_band_members": sum(1 for r in rows
                                     if r.get("boundary_band_member")),
        "aggregates_are_non_verdict": True,
        "aggregate_note": "every value in this summary is NON-VERDICT; the "
                          "verdict is the per-slice structure",
        "dtype_path": {
            "selected": VERDICT_DTYPE_PATH, "rule": DTYPE_PATH_RULE,
            "prepared_input_dtype": "float32",
            "fft_operand_dtype": "complex64 (verdict) / complex128 "
                                 "(sensitivity)",
            "fft_result_dtype": "complex64 (verdict) / complex128 "
                                "(sensitivity)",
            "reduction_accumulator_dtype": "complex128 -> float64 (NumPy)",
            "selected_thresholds": {"RHO_M_MAX": RHO_M_MAX,
                                    "ABS_LEAK": ABS_LEAK},
            "float64_verification": "float64 verification not performed: the "
                                    "complete relevant operator path does not "
                                    "genuinely run in float64"},
        "x0_construction": {
            "raw_measurement_source_key": "y",
            "per_volume_scale_source": "amax (HDF5 file-level attr 'max')",
            "mask_source": "batch['mask'] -- 1-D (W,) boolean column mask",
            "stored_measurement_already_masked": True,
            "reapplying_M_is_idempotent": True,
            "idempotency_note": "fastmri_data.__getitem__ stores "
                                "y = k96 * mask, so applying M again is "
                                "IDEMPOTENT rather than semantically "
                                "necessary; it is applied so the assembly "
                                "states its own support",
            "operation_order": "x0_c = ifft2c(M y_raw); x0 = "
                               "two_channel(x0_c) / a -- adjoint on RAW y, "
                               "divide in the IMAGE domain, matching "
                               "train_base._prepare",
            "not_used": "MaskedFourierOperator.A_adjoint and _prepare are NOT "
                        "called by the assertion path",
            "linearity_note": "F^H(M(y/a)) is mathematically equal but is a "
                              "different registered operation order and is "
                              "NOT the asserted form",
            "reconstruction_dtype": "float32 (pipeline-matched); float64 "
                                    "recorded as non-blocking sensitivity",
            "output_layout": "(2, H, W) real, channel 0 = Re, channel 1 = Im",
            "prepared_object_layout": "(B, 2, H, W) from _prepare['cond_in']"},
        "fft_convention": "fastmri_data.fft2c/ifft2c -- centred orthonormal: "
                          "fftshift(fft2(ifftshift(x), norm='ortho'))",
        "mask_broadcast": "boolean (W,) cast to the k-space dtype and "
                          "multiplied, broadcasting over rows -- identical to "
                          "MaskedFourierOperator._m",
        "decoder_note": "exact-DC decoder consistency belongs to P3; P2 does "
                        "NOT test the branch-aware decoder",
    }
    semantic = {"schema": FACTS_SCHEMA, "stage": "P2",
                "thresholds": thresholds, "verdict": verdict,
                "slices": rows,
                "summary": summary,
                "dtype_path_selection": summary["dtype_path"],
                "parents": {"p0_facts_sha256": parents["p0"]["facts_sha256"],
                            "p0s_facts_sha256":
                                parents["p0s"]["facts_sha256"],
                            "subset_manifest_sha256":
                                parents["p0s"]["subset_manifest_sha256"],
                            "contract_hash": parents["p0"]["contract_hash"]},
                "code": hash_project_code(repo_dir, script)["project_local"]}

    facts = {
        "schema": FACTS_SCHEMA,
        "script": {"id": SCRIPT_ID, "version": SCRIPT_VERSION,
                   "lifetime": "KEEP"},
        "stage": "P2", "artefact_type": "stage_facts",
        "run_mode": ("smoke" if smoke is not None else "authoritative"),
        "authoritative": smoke is None,
        "stage_description": "measured-support validity",
        "thresholds": thresholds, "verdict": verdict,
        "verdict_reason": reason, "summary": summary, "slices": rows,
        "parents": parents,
        "code": hash_project_code(repo_dir, script),
        "run": {**environment_record(repo_dir, argv),
                "runtime_seconds": time.time() - t0,
                "peak_memory_bytes": _peak_rss_bytes()},
        "hash_note": "the authoritative artefact SHA is the SHA-256 of THIS "
                     "FILE'S bytes, in the sidecar; semantic_sha256 covers "
                     "scientific content only and is not self-referential",
        "overwrite_policy": {
            "authoritative_never_overwritten": True,
            "rerun_behaviour": "a SEQUENTIAL rerun against an occupied authoritative prefix leaves <prefix>.json untouched and writes <prefix>.<utc-stamp>.json plus its own sidecar alongside it (preflight_io.publish); the stamp is taken from run.utc",
            "concurrent_behaviour": "a CONCURRENT same-prefix publication is refused at the atomic claim before any file is written -- distinct from the sequential rerun path",
            "consumer_rule": "downstream stages consume <prefix>.json only; a timestamped record is evidence of a rerun and is never the authoritative artefact"},
        "verify_before_use": ["P3 must verify this file against its sidecar "
                              "before consuming the support verdict"],
    }
    if block is not None:
        facts.update(block.as_record())
        semantic.update(block.as_record())
    return attach_semantic_hash(facts, semantic)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="SEQREF-P2SUP v0.1 -- P2 measured-support validity")
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--p0-facts", required=True)
    ap.add_argument("--p0s-facts", required=True)
    ap.add_argument("--p0s-script", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=None,
                    help="EPHEMERAL: first N frozen indices, smoke_ prefix; "
                         "never authoritative")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    t0 = time.time()
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    facts_prefix = SMOKE_FACTS_PREFIX if args.smoke else FACTS_PREFIX
    error_prefix = SMOKE_ERROR_PREFIX if args.smoke else ERROR_PREFIX
    script = os.path.abspath(__file__)
    parents = None
    rows: list[dict] = []

    try:
        if args.smoke is not None and args.smoke <= 0:
            raise StageError("BAD_SMOKE_SIZE",
                             f"--smoke must be a positive int, got "
                             f"{args.smoke!r}")
        run_mode = guard_run_mode(args.out_dir, args.smoke is not None)
        logger.info("%s run_mode=%s out_dir=%s", SCRIPT_ID, run_mode,
                    args.out_dir)
        parents = verify_parents(args.repo_dir, args.p0_facts, args.p0s_facts,
                                 args.p0s_script)
        rows = _evaluate(parents, args.data_root, args.batch, args.smoke)
        _gate(rows)
        reason = (f"all {len(rows)} slices satisfy the applicable "
                  f"measured-support conditions on the {VERDICT_DTYPE_PATH} "
                  f"verdict path")
        facts = _build_facts(parents, rows, "PASS", reason, None,
                             args.repo_dir, script, raw_argv, t0, args.smoke)
        path, sha = publish_stage(facts, args.out_dir, facts_prefix, "P2")
        logger.info("P2 PASS n=%d ordinary=%d near_zero=%d facts=%s "
                    "file_sha256=%s semantic_sha256=%s", len(rows),
                    facts["summary"]["branch_counts"]["ordinary"],
                    facts["summary"]["branch_counts"]["near_zero"], path, sha,
                    facts["semantic_sha256"])
        if args.smoke is not None:
            logger.warning("SMOKE run -- NOT authoritative; delete %s after "
                           "inspection", path)
        return EXIT_PASS

    except StageBlock as blk:
        logger.error("P2 BLOCK -- %s", blk.reason)
        try:
            facts = _build_facts(parents, rows, "BLOCK", blk.reason, blk,
                                 args.repo_dir, script, raw_argv, t0,
                                 args.smoke)
            path, sha = publish_stage(facts, args.out_dir, facts_prefix, "P2")
            logger.error("P2 BLOCK record published: %s (%s)", path, sha)
        except Exception:
            logger.exception("P2 BLOCK could not be published; the verdict "
                             "must not survive only as a log line")
            return EXIT_ERROR
        return EXIT_BLOCK
    except StageError as exc:
        logger.error("P2 ERROR [%s] -- %s", exc.error_code, exc.reason)
        publish_error(exc, args.out_dir, error_prefix, "P2",
                      parents=(parents or {}).get("p0"),
                      code={"script": script}, run={"argv": raw_argv})
        return EXIT_ERROR
    except Exception as exc:
        # Failure boundary. Anything not already typed -- an OSError during
        # dataset access, a library exception inside the FFT, a malformed
        # batch -- becomes a DETERMINISTIC ERROR with an exit code and, where
        # the output path is trustworthy, an auditable record.
        # KeyboardInterrupt and SystemExit derive from BaseException and are
        # deliberately NOT caught. An ordinary exception must never reach the
        # caller as a traceback: a traceback has no exit-code contract and
        # leaves no artefact behind.
        logger.exception("%s UNEXPECTED ERROR", SCRIPT_ID)
        wrapped = StageError(
            "UNEXPECTED_RUNTIME_ERROR",
            f"{type(exc).__name__}: {exc}",
            detail={"exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "raised_after_parent_verification": parents is not None},
            # Before parent verification the out-dir context is not yet
            # trustworthy, so no record is presented as valid.
            write_record=parents is not None)
        publish_error(wrapped, args.out_dir, error_prefix, "P2",
                      parents=(parents or {}).get("p0"),
                      code={"script": script}, run={"argv": raw_argv})
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
