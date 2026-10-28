# SEQREF-MRI-V05LTI17BD v0.2 -- scripts.v05lt_i17b_dcdiag
# LIFETIME: DIAGNOSTIC
# =============================================================================
# Purpose: per-measure diagnostic for finding I1.7b-1 (196 item 23): on the
#   SAME six validation slices as T9 F12 (validation view positions 0-5),
#   with a fresh seed-2026 build (not a production checkpoint), report for
#   R0+DC (and R1+DC for comparison) per slice: the four DC measures, the
#   six-slice fixture tolerances, value / tolerance ratios, numerators and
#   denominators, failed-measure names, absolute complex-image change, the
#   unmeasured k-space norm relative to the ground truth's, and PSNR / SSIM
#   before and after DC.
# Read-only: no production artefact, no results/ write (stdout only; the
#   owner tees it into results/v05lt/diagnostics). Validation data only; no
#   holdout contact; no training. Tolerances are the six-slice fixture
#   values recomputed by the production calibrate_dc and compared with the
#   T9 values; nothing here changes a registered tolerance.
# Uses: v05lt_eval v0.5 production functions; v05lt_fixtures._rt_data (host
#   data loader used by the T9 child) -- fixture helper, read-only.
# Exit: 0 = diagnostic completed (findings are data, not PASS/FAIL);
#   2 = the diagnostic itself failed (typed, logged).
# Interpretation: rows are compared with the T9 tolerances ONLY if
#   FLOORS_EQUAL_T9 all_true = true (all four floors bit-identical to T9).
#   The diagnostic is evidence only: not an acceptance gate; it never
#   changes a registered tolerance.
# Changelog (v0.2, 2026-10-22; reviewer reporting correction):
#   * ROW lines also print the slice identity, mse_before and mse_after.
#   * FLOORS_EQUAL_T9 adds all_true and the interpretation rule.
# Changelog (v0.1, 2026-10-22): created (reviewer request, T9 review).
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

from seqref_mri.scripts import v05lt_common as C

logger = logging.getLogger("seqref_mri.v05lt_i17b_dcdiag")
__version__ = "0.2"
__abbr__ = "SEQREF-MRI-V05LTI17BD"

INDICES = list(range(6))              # = T9 F12 host sample
T9_FLOORS = {"a": 3.9603606179481267e-16, "b": 7.357850261595835e-16,
             "c_rms": 3.7958943622102876e-16, "c_max": 5.41260417710494e-16}


def _parts(EV, x, x_dc, y, m) -> dict:
    """Numerators and denominators of the registered measures (194 §8)."""
    torch, fdm, _ = EV._dc_ops()
    kx, kd = fdm.fft2c(x), fdm.fft2c(x_dc)
    return {
        "a": (float((m * (kd - y)).abs().max()), float((m * y).abs().max())),
        "b": (float(torch.linalg.vector_norm((1 - m) * (kd - kx))),
              float(torch.linalg.vector_norm((1 - m) * kx))),
        "c_rms": (float(torch.linalg.vector_norm(x_dc - x)),
                  float(torch.linalg.vector_norm(x))),
        "c_max": (float((x_dc - x).abs().max()), float(x.abs().max()))}


def _slice_report(EV, M, st, x, tol, gt_unmeasured: float) -> dict:
    m, y = EV._dc_mask(st), EV.y_norm(st)
    x_dc = EV.hard_dc(x, y, m)
    meas = EV.dc_measures(x, x_dc, y, m, f"diag {st['identity']}")
    parts = _parts(EV, x, x_dc, y, m)
    for k in EV.DC_MEASURES:                      # same formula, same value
        num, den = parts[k]
        C.require(num / den == meas[k], "DIAG_MEASURE_MISMATCH",
                  f"{k}: {num}/{den} != {meas[k]}")
    ratio = {k: meas[k] / tol[k] for k in EV.DC_MEASURES}
    before = M.psnr_record(x.abs(), st["x_true_mag"])
    after = M.psnr_record(x_dc.abs(), st["x_true_mag"])
    return {
        "identity": st["identity"],
        "measures": meas,
        "ratio_to_tolerance": ratio,
        "numerators": {k: parts[k][0] for k in EV.DC_MEASURES},
        "denominators": {k: parts[k][1] for k in EV.DC_MEASURES},
        "failed_measures": [k for k in EV.DC_MEASURES if meas[k] > tol[k]],
        "identity_within_tolerance": EV.dc_identity(meas, tol),
        "abs_change": {
            "complex_max": float((x_dc - x).abs().max()),
            "complex_l2": float(np.linalg.norm((x_dc - x).numpy())),
            "magnitude_max": float((x_dc.abs() - x.abs()).abs().max())},
        "unmeasured_norm_over_gt": parts["b"][1] / gt_unmeasured,
        "psnr_before": before["psnr"], "psnr_after": after["psnr"],
        "psnr_diff_db": after["psnr"] - before["psnr"],
        "mse_before": before["mse"], "mse_after": after["mse"],
        "ssim_before": M.ssim_value(x.abs(), st["x_true_mag"]),
        "ssim_after": M.ssim_value(x_dc.abs(), st["x_true_mag"])}


def run(repo: Path, data_root: Path) -> dict:
    import torch
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.scripts import v05lt_fixtures as F
    from seqref_mri.src import free_flow_runtime as ffr
    from seqref_mri.tdiag import estimators as est
    out = {"abbr": __abbr__, "version": __version__,
           "eval_version": EV.__version__, "indices": INDICES,
           "torch": torch.__version__, "threads": torch.get_num_threads(),
           "deterministic": bool(torch.are_deterministic_algorithms_enabled())}
    with tempfile.TemporaryDirectory() as t:
        data = F._rt_data(True, repo, str(data_root), Path(t))
        states = EV.build_states(data["vv"], INDICES, data["p4"],
                                 data["s_ref"])
    cal = EV.calibrate_dc(states)
    tol = EV.check_tolerances(cal["tolerances"])
    out["calibration"] = {k: cal[k] for k in ("floors", "tolerances",
                                              "argmax_identity")}
    out["calibration"]["min_denominators"] = cal["provenance"][
        "min_denominators"]
    out["floors_equal_T9"] = {k: cal["floors"][k] == T9_FLOORS[k]
                              for k in EV.DC_MEASURES}
    model = ffr.build_model(init_seed=2026)
    model.eval()
    bank = est.z_diag_bank()["bank"]
    rows = []
    with torch.no_grad():
        for st in states:
            from seqref_mri.src import fastmri_data as fdm
            gt = st["x_true_c"].to(torch.complex128)
            m = EV._dc_mask(st)
            gt_un = float(torch.linalg.vector_norm((1 - m) * fdm.fft2c(gt)))
            x0, _ = EV.r0_image(model, st)
            dec = EV.decode_bank(model, st, bank)
            x1, _ = EV.r1_image(st, dec)
            rows.append({"r0_dc": _slice_report(EV, M, st, x0, tol, gt_un),
                         "r1_dc": _slice_report(EV, M, st, x1, tol, gt_un)})
    out["rows"] = rows
    for key in ("r0_dc", "r1_dc"):
        fails = {}
        for r in rows:
            for k in r[key]["failed_measures"]:
                fails[k] = fails.get(k, 0) + 1
        out[f"{key}_summary"] = {
            "n_identity": sum(r[key]["identity_within_tolerance"]
                              for r in rows),
            "n": len(rows), "failed_measure_counts": fails,
            "max_ratio": {k: max(r[key]["ratio_to_tolerance"][k]
                                 for r in rows) for k in EV.DC_MEASURES},
            "max_abs_psnr_diff_db": max(abs(r[key]["psnr_diff_db"])
                                        for r in rows)}
    return out


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=f"{__abbr__} v{__version__}")
    ap.add_argument("--repo", default=str(C.Paths().repo))
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args(argv)
    paths = C.Paths(repo=Path(a.repo))
    data_root = Path(a.data_root) if a.data_root else paths.data_root
    logger.info("%s v%s repo=%s data_root=%s", __abbr__, __version__,
                a.repo, data_root)
    try:
        res = run(Path(a.repo), data_root)
    except C.V05LTError as exc:
        logger.error("diagnostic failed: %s", exc)
        print(f"RESULT ERROR {exc.code}")
        return 2
    except Exception as exc:  # noqa: BLE001 -- logged, reported, exit 2
        logger.exception("diagnostic raised unexpectedly")
        print(f"RESULT ERROR {type(exc).__name__}")
        return 2
    print(f"\n=== {__abbr__} v{__version__} -- paste back ===")
    print("CALIBRATION " + json.dumps(res["calibration"], sort_keys=True))
    eq = res["floors_equal_T9"]
    all_true = all(eq.values())
    print("FLOORS_EQUAL_T9 " + json.dumps(dict(eq, all_true=all_true)))
    print("INTERPRETATION " + ("rows compared with the T9 six-slice "
                               "tolerances" if all_true else
                               "floors differ from T9: rows are NOT a "
                               "comparison with the T9 tolerances"))
    for i, r in enumerate(res["rows"]):
        for key in ("r0_dc", "r1_dc"):
            d = r[key]
            print(f"ROW {i} {key} " + json.dumps(
                {k: d[k] for k in ("identity", "failed_measures",
                                   "ratio_to_tolerance",
                                   "measures", "numerators", "denominators",
                                   "abs_change", "unmeasured_norm_over_gt",
                                   "psnr_before", "psnr_after",
                                   "psnr_diff_db", "mse_before",
                                   "mse_after", "ssim_before",
                                   "ssim_after")}, sort_keys=True))
    for key in ("r0_dc", "r1_dc"):
        print(f"SUMMARY {key} " + json.dumps(res[f"{key}_summary"],
                                              sort_keys=True))
    print(f"BACKEND torch={res['torch']} threads={res['threads']} "
          f"deterministic={res['deterministic']} eval={res['eval_version']}")
    finite = all(math.isfinite(v) for r in res["rows"] for key in r
                 for v in r[key]["measures"].values())
    print(f"RESULT DONE rows={len(res['rows'])} finite={finite}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
