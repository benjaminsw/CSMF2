# SEQREF-P1REP v0.1 -- P1 target representation branch (EXEC v0.4 §8, A3)
# LIFETIME: KEEP
#
# Purpose
#   Rule the target-representation branch -- REAL, COMPLEX or AMBIGUOUS -- on
#   the frozen 256-slice P0S subset, and record the per-slice evidence the
#   ruling rests on. P3 builds the coordinate map from this branch, so a wrong
#   ruling here is not recoverable downstream.
#
# What this stage does NOT do
#   It does not redraw the subset, does not consume P2's verdict, and does not
#   test the decoder. P1 and P2 are independent and may run in parallel.
#
# Verdict semantics (A3)
#   PASS  branch ruled REAL or COMPLEX; facts published; exit 0.
#   BLOCK a premise about the DATA failed -- a real-channel-degenerate slice,
#         or an AMBIGUOUS population. Valid facts are published FIRST, then
#         exit 1. A BLOCK never disappears as a log line.
#   ERROR the code or specification is wrong. A distinctly identified error
#         record is written where the output path is trustworthy, then exit 2.
#
# Locked conventions
#   * norms are SUMS over channel elements, not means; reduction over (H,W)
#     per slice in NumPy float64;
#   * comparisons exactly as written: <= for REAL, >= for COMPLEX;
#   * finiteness asserted on E_re, E_im, both maxima and both ratios BEFORE
#     any comparison -- NaN > tol evaluates False;
#   * empty subset -> ERROR, never a verdict;
#   * the median convention is INHERITED from the P0S implementation and
#     PROVEN by reproduction (preflight_parents), never re-invented;
#   * conjugate-symmetry violation is recorded for every slice (A3 core), with
#     the index pairing and the normalisation stated. Non-blocking.
#
# CONVENTION: logger.error + raise on every failure path. No fallback, no mock,
#   no placeholder, no silent pass.
#
# Changelog
#   v0.1 (2026-07-30) Created under Amendment A3. R_REAL_MIN is evaluated
#     BEFORE any median or branch quantity, as the registered BLOCK premise
#     about the data population. Mutual exclusivity AND exhaustiveness of the
#     three outcomes are asserted rather than assumed. Conjugate symmetry is
#     recorded in core rather than left optional.

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
from seqref_mri.src.fastmri_data import FastMRISliceDataset  # noqa: E402
from seqref_mri.scripts.train_base import _collate, _prepare  # noqa: E402

SCRIPT_ID = "SEQREF-P1REP"
SCRIPT_VERSION = "v0.1"
FACTS_SCHEMA = "seqref-p1-facts/1"
FACTS_PREFIX = "representation_facts"
ERROR_PREFIX = "representation_error"
SMOKE_FACTS_PREFIX = "smoke_representation_facts"
SMOKE_ERROR_PREFIX = "smoke_representation_error"

# EXEC v0.4 §13 -- registered, no value introduced here.
R_REAL_MIN = 1e-10            # BLOCKING: ||Re x||^2 / S_ref^2
RHO_E_REAL_MAX = 1e-10        # REAL condition on rho_imag_E
RHO_MAX_REAL_MAX = 1e-5       # REAL condition on rho_imag_max
RHO_E_COMPLEX_MIN = 1e-6      # COMPLEX condition on median rho_imag_E
RHO_MAX_COMPLEX_MIN = 1e-3    # COMPLEX condition on median rho_imag_max

logger = logging.getLogger(SCRIPT_ID)


# ---------------------------------------------------------------------------
# Pure per-slice quantities (unit-testable without the dataset)
# ---------------------------------------------------------------------------

def slice_representation_metrics(re: np.ndarray, im: np.ndarray) -> dict:
    """re, im: (H, W) float64. Sums over channel elements, not means."""
    if re.shape != im.shape:
        raise StageError("CHANNEL_SHAPE_MISMATCH",
                         f"real {re.shape} and imaginary {im.shape} channels "
                         f"differ in shape")
    e_re = float((re ** 2).sum())
    e_im = float((im ** 2).sum())
    max_re = float(np.abs(re).max())
    max_im = float(np.abs(im).max())
    require_finite({"E_re": e_re, "E_im": e_im, "max_abs_re": max_re,
                    "max_abs_im": max_im}, "P1 slice energies")
    if e_re <= 0.0 or max_re <= 0.0:
        # The ratio denominators. The R_REAL_MIN gate is what BLOCKS on a
        # degenerate real channel; reaching a division here means the gate was
        # evaluated out of order, which is a code fault.
        raise StageError("REAL_CHANNEL_DENOMINATOR_NON_POSITIVE",
                         f"real-channel denominators must be strictly "
                         f"positive before any ratio is formed "
                         f"(E_re={e_re!r}, max|Re|={max_re!r}); the "
                         f"R_REAL_MIN gate must be evaluated first")
    rho_e = e_im / e_re
    rho_max = max_im / max_re          # no stabilising constant
    require_finite({"rho_imag_E": rho_e, "rho_imag_max": rho_max},
                   "P1 slice ratios")
    return {"E_re": e_re, "E_im": e_im, "max_abs_re": max_re,
            "max_abs_im": max_im, "rho_imag_E": rho_e,
            "rho_imag_max": rho_max}


def conjugate_symmetry_violation(x_complex: np.ndarray) -> dict:
    """Non-blocking cross-check on F x, recorded ABSOLUTE and SCALE-RELATIVE.

    Pairing: the DFT is taken UNSHIFTED (ifftshift applied to the image, as in
    fastmri_data.fft2c, but without the trailing fftshift), so the conjugate
    partner of frequency index (u, v) is ((-u) mod N, (-v) mod N). A real
    image satisfies K[-u, -v] == conj(K[u, v]).
    Normalisation: the relative figure divides by max|K| over the same slice.
    """
    n0, n1 = x_complex.shape
    k = np.fft.fft2(np.fft.ifftshift(x_complex), norm="ortho")
    i0 = (-np.arange(n0)) % n0
    i1 = (-np.arange(n1)) % n1
    partner = np.conj(k[np.ix_(i0, i1)])
    viol = np.abs(k - partner)
    abs_v = float(viol.max())
    scale = float(np.abs(k).max())
    require_finite({"conj_violation_abs": abs_v, "k_max": scale},
                   "P1 conjugate-symmetry cross-check")
    rel_v = abs_v / scale if scale > 0.0 else None
    return {"conj_symmetry_violation_abs": abs_v,
            "conj_symmetry_violation_rel": rel_v,
            "conj_symmetry_k_max": scale}


def classify_branch(rho_e: np.ndarray, rho_max: np.ndarray) -> tuple[str, dict]:
    """REAL / COMPLEX / AMBIGUOUS with mutual exclusivity AND exhaustiveness
    asserted. Violation is an ERROR, not a verdict."""
    if rho_e.size == 0 or rho_max.size == 0:
        raise StageError("EMPTY_SUBSET",
                         "no slices were evaluated; an empty subset is an "
                         "ERROR, never a verdict")
    if not (np.all(np.isfinite(rho_e)) and np.all(np.isfinite(rho_max))):
        raise StageError("NON_FINITE_RATIO",
                         "non-finite ratio reached the branch classifier")
    n_real_e = int((rho_e <= RHO_E_REAL_MAX).sum())
    n_real_max = int((rho_max <= RHO_MAX_REAL_MAX).sum())
    is_real = bool(n_real_e == rho_e.size and n_real_max == rho_max.size)
    med_e = float(np.median(rho_e))
    med_max = float(np.median(rho_max))
    is_complex = bool(med_e >= RHO_E_COMPLEX_MIN
                      or med_max >= RHO_MAX_COMPLEX_MIN)
    if is_real and is_complex:
        raise StageError("BRANCH_NOT_MUTUALLY_EXCLUSIVE",
                         f"both REAL and COMPLEX conditions hold "
                         f"(median rho_E={med_e!r}, median "
                         f"rho_max={med_max!r}); the registered thresholds "
                         f"cannot both be satisfied and the specification or "
                         f"the implementation is wrong")
    ruling = "REAL" if is_real else ("COMPLEX" if is_complex else "AMBIGUOUS")
    if ruling not in ("REAL", "COMPLEX", "AMBIGUOUS"):
        raise StageError("BRANCH_NOT_EXHAUSTIVE",
                         f"classifier produced {ruling!r}")
    detail = {"median_rho_imag_E": med_e, "median_rho_imag_max": med_max,
              "n_satisfying_real_rho_E": n_real_e,
              "n_satisfying_real_rho_max": n_real_max,
              "n_slices": int(rho_e.size),
              "real_conditions_all_hold": is_real,
              "complex_condition_holds": is_complex}
    return ruling, detail


# ---------------------------------------------------------------------------

def _peak_rss_bytes() -> int:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru * 1024) if sys.platform.startswith("linux") else int(ru)


def _collect(parents: dict, data_root: str, batch: int,
             smoke: int | None) -> list[dict]:
    ds = FastMRISliceDataset(data_root, split="train", mode="eval")
    n_pop = len(ds)
    if n_pop != parents["p0s"]["population_size"]:
        raise StageError("POPULATION_CHANGED",
                         f"dataset now holds {n_pop} slices but P0S froze its "
                         f"subset against a population of "
                         f"{parents['p0s']['population_size']}")
    indices = parents["subset_indices"]
    if smoke is not None:
        indices = indices[:smoke]
        logger.warning("SMOKE MODE: %d of %d frozen indices; this run is NOT "
                       "authoritative", len(indices),
                       parents["subset_size"])
    if not indices:
        raise StageError("EMPTY_SUBSET",
                         "the frozen subset selection is empty")
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
                             f"_prepare() returned no {missing}; the "
                             f"preparation contract has changed")
        x = p["x_norm"]
        if x.ndim != 4 or x.shape[1] != 2:
            raise StageError("STATE_SHAPE_UNEXPECTED",
                             f"x_norm has shape {tuple(x.shape)}; the full "
                             f"two-channel state is required")
        xn = x.detach().cpu().numpy().astype(np.float64, copy=False)
        for j, meta in enumerate(b["meta"]):
            re, im = xn[j, 0], xn[j, 1]
            e_re = float((re ** 2).sum())
            require_finite({"E_re": e_re}, "P1 real-channel energy")
            row = {"file": meta["file"],
                   "slice_index": int(meta["slice_index"]),
                   "split": meta["split"], "mode": meta["mode"],
                   "mask_seed": int(meta["mask_seed"]),
                   "file_attr_max": float(meta["file_attr_max"]),
                   "E_re": e_re, "E_re_over_S_ref_sq": e_re / s2,
                   "finite_E_re": True}
            rows.append(row)
            row["_re"], row["_im"] = re, im
    if len(rows) != len(indices):
        raise StageError("SUBSET_SIZE_MISMATCH",
                         f"collected {len(rows)} entries, expected "
                         f"{len(indices)}")
    for k, idx in enumerate(indices):
        rows[k]["dataset_index"] = int(idx)
    return rows


def _gate_real_channel(rows: list[dict]) -> None:
    """R_REAL_MIN, evaluated BEFORE any median or branch quantity.

    Registered in EXEC v0.4 §8/§13 as a BLOCK: the declared premise is that the
    inspected population contains no real-channel-degenerate slice, which is a
    statement about the DATA, not about the code."""
    failing = [r for r in rows if r["E_re_over_S_ref_sq"] <= R_REAL_MIN]
    if failing:
        first = failing[0]
        raise StageBlock(
            "REAL_CHANNEL_DEGENERATE",
            f"{len(failing)} slice(s) have ||Re x||^2 / S_ref^2 <= "
            f"{R_REAL_MIN}; the declared premise that the inspected "
            f"population contains no real-channel-degenerate slice does not "
            f"hold, so no branch quantity may be formed",
            observed=first["E_re_over_S_ref_sq"], threshold=R_REAL_MIN,
            first_failing={"dataset_index": first["dataset_index"],
                           "file": first["file"],
                           "slice_index": first["slice_index"],
                           "E_re": first["E_re"],
                           "E_re_over_S_ref_sq": first["E_re_over_S_ref_sq"]},
            n_failing=len(failing))


def compute_margins(rows: list[dict]) -> dict:
    """NON-VERDICT threshold margins, oriented so a value near 1 means
    the population only just satisfies the condition: > 1 is headroom,
    < 1 is failure where the condition applies. Pure, so the edge cases
    are testable without a dataset."""
    e_rel = [r["E_re_over_S_ref_sq"] for r in rows
             if isinstance(r.get("E_re_over_S_ref_sq"), float)]
    return {
        "definition": "margin = distance to the registered threshold, "
                      "expressed so that a value near 1 means the population "
                      "only just satisfies the condition. NON-VERDICT: "
                      "computed from quantities already recorded, and "
                      "inspected during smoke review.",
        "undefined_rule": "a margin whose denominator is zero or absent is "
                          "recorded as null, never as a number",
        "status_vocabulary": ["finite", "unbounded", "not_applicable"],
        "status_rule": "finite = a real quotient; unbounded = the denominator "
                       "is exactly zero, so headroom is mathematically "
                       "unbounded; not_applicable = no slice contributes a "
                       "denominator. The value is null for BOTH non-finite "
                       "statuses -- JSON infinity and NaN are never emitted "
                       "-- so the STATUS field, not the null, carries the "
                       "reason.",
        "real_channel": (min(e_rel) / R_REAL_MIN if e_rel else None),
        "real_channel_status": ("finite" if e_rel else "not_applicable"),
        "real_channel_formula": "min_i(E_re,i / S_ref^2) / R_REAL_MIN",
        "real_channel_note": "R_REAL_MIN is a registered positive constant, "
                             "so this denominator can never be zero; only "
                             "not_applicable (an empty population, itself an "
                             "ERROR) is reachable. A PASS artefact has this "
                             "margin STRICTLY greater than 1, because the "
                             "gate BLOCKs at <= R_REAL_MIN; a value of "
                             "exactly 1 can appear only in a BLOCK artefact.",
    }


def _build_facts(parents: dict, rows: list[dict], ruling: str | None,
                 detail: dict | None, verdict: str, reason: str,
                 block: StageBlock | None, repo_dir: str, script: str,
                 argv, t0: float, smoke: int | None) -> dict:
    slices = []
    for r in rows:
        rec = {k: v for k, v in r.items() if not k.startswith("_")}
        slices.append(rec)
    thresholds = {"R_REAL_MIN": R_REAL_MIN,
                  "RHO_E_REAL_MAX": RHO_E_REAL_MAX,
                  "RHO_MAX_REAL_MAX": RHO_MAX_REAL_MAX,
                  "RHO_E_COMPLEX_MIN": RHO_E_COMPLEX_MIN,
                  "RHO_MAX_COMPLEX_MIN": RHO_MAX_COMPLEX_MIN,
                  "S_ref": parents["s_ref"],
                  "S_ref_squared": parents["s_ref_squared"]}

    def _worst(key, largest=True):
        vals = [r for r in rows if key in r]
        if not vals:
            return None
        r = (max if largest else min)(vals, key=lambda z: z[key])
        return {"dataset_index": r["dataset_index"], "file": r["file"],
                "slice_index": r["slice_index"], key: r[key]}

    margins = compute_margins(rows)
    summary = {
        "n_slices": len(rows),
        "margins": margins,
        "smoke": smoke is not None,
        "ruling": ruling,
        "branch_detail": detail,
        "worst_rho_imag_E": _worst("rho_imag_E"),
        "worst_rho_imag_max": _worst("rho_imag_max"),
        "min_E_re_over_S_ref_sq": _worst("E_re_over_S_ref_sq", largest=False),
        "max_conj_symmetry_violation_abs":
            _worst("conj_symmetry_violation_abs"),
        "median_convention": parents["median_convention"],
        "dtype_provenance": {
            "prepared_input_dtype": "float32 (_prepare returns float32)",
            "reduction_accumulator_dtype": "float64 (NumPy)",
            "reduction_scope": "sums over channel elements per slice, "
                               "reduced over (H, W)"},
        "conjugate_symmetry_note":
            "recorded for every slice as a NON-BLOCKING cross-check; the "
            "pairing is ((-u) mod N, (-v) mod N) on the UNSHIFTED DFT and the "
            "relative figure divides by max|K| on the same slice",
    }
    semantic = {"schema": FACTS_SCHEMA, "stage": "P1", "thresholds": thresholds,
                "verdict": verdict, "ruling": ruling, "branch_detail": detail,
                "slices": slices,
                "summary": {k: v for k, v in summary.items()
                            if k != "median_convention"},
                "median_convention": parents["median_convention"],
                "parents": {"p0_facts_sha256":
                            parents["p0"]["facts_sha256"],
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
        "stage": "P1",
        "artefact_type": "stage_facts",
        "run_mode": ("smoke" if smoke is not None else "authoritative"),
        "authoritative": smoke is None,
        "stage_description": "target representation branch",
        "thresholds": thresholds,
        "verdict": verdict,
        "verdict_reason": reason,
        "ruling": ruling,
        "summary": summary,
        "slices": slices,
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
                              "before consuming the branch ruling"],
    }
    if block is not None:
        facts.update(block.as_record())
        semantic.update(block.as_record())
    return attach_semantic_hash(facts, semantic)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="SEQREF-P1REP v0.1 -- P1 target representation branch")
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--p0-facts", required=True)
    ap.add_argument("--p0s-facts", required=True)
    ap.add_argument("--p0s-script", required=True,
                    help="path to p0s_normalisation_scale.py; the median "
                         "convention is inherited from it and proven by "
                         "reproduction")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", type=int, default=8,
                    help="loader batch size; affects speed only")
    ap.add_argument("--smoke", type=int, default=None,
                    help="EPHEMERAL: evaluate only the first N frozen indices "
                         "and publish under a smoke_ prefix; never "
                         "authoritative")
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
        rows = _collect(parents, args.data_root, args.batch, args.smoke)

        # BLOCK gate FIRST, before any median or branch quantity is formed.
        _gate_real_channel(rows)

        # Only now are the ratio denominators known to be non-degenerate.
        for r in rows:
            re, im = r.pop("_re"), r.pop("_im")
            r.update(slice_representation_metrics(re, im))
            r.update(conjugate_symmetry_violation(re + 1j * im))
            r["satisfies_real_rho_E"] = bool(r["rho_imag_E"]
                                             <= RHO_E_REAL_MAX)
            r["satisfies_real_rho_max"] = bool(r["rho_imag_max"]
                                               <= RHO_MAX_REAL_MAX)
            r["finite_E_im"] = True
            r["finite_ratios"] = True

        ruling, detail = classify_branch(
            np.asarray([r["rho_imag_E"] for r in rows], dtype=np.float64),
            np.asarray([r["rho_imag_max"] for r in rows], dtype=np.float64))

        if ruling == "AMBIGUOUS":
            raise StageBlock(
                "REPRESENTATION_AMBIGUOUS",
                f"neither the REAL conditions (all slices rho_imag_E <= "
                f"{RHO_E_REAL_MAX} AND rho_imag_max <= {RHO_MAX_REAL_MAX}) "
                f"nor the COMPLEX conditions (median rho_imag_E >= "
                f"{RHO_E_COMPLEX_MIN} OR median rho_imag_max >= "
                f"{RHO_MAX_COMPLEX_MIN}) hold; the branch is undetermined "
                f"and P3 has no map to build",
                observed=detail, threshold={"real": [RHO_E_REAL_MAX,
                                                    RHO_MAX_REAL_MAX],
                                            "complex": [RHO_E_COMPLEX_MIN,
                                                        RHO_MAX_COMPLEX_MIN]},
                first_failing=None, n_failing=len(rows))

        reason = (f"branch ruled {ruling}: median rho_imag_E="
                  f"{detail['median_rho_imag_E']:.6e}, median rho_imag_max="
                  f"{detail['median_rho_imag_max']:.6e} over "
                  f"{detail['n_slices']} slices")
        facts = _build_facts(parents, rows, ruling, detail, "PASS", reason,
                             None, args.repo_dir, script, raw_argv, t0,
                             args.smoke)
        path, sha = publish_stage(facts, args.out_dir, facts_prefix, "P1")
        logger.info("P1 PASS ruling=%s median_rho_E=%.6e median_rho_max=%.6e "
                    "n=%d facts=%s file_sha256=%s semantic_sha256=%s", ruling,
                    detail["median_rho_imag_E"], detail["median_rho_imag_max"],
                    detail["n_slices"], path, sha, facts["semantic_sha256"])
        if args.smoke is not None:
            logger.warning("SMOKE run -- NOT authoritative; delete %s after "
                           "inspection", path)
        return EXIT_PASS

    except StageBlock as blk:
        logger.error("P1 BLOCK -- %s", blk.reason)
        try:
            facts = _build_facts(parents, rows, None, None, "BLOCK",
                                 blk.reason, blk, args.repo_dir, script,
                                 raw_argv, t0, args.smoke)
            path, sha = publish_stage(facts, args.out_dir, facts_prefix, "P1")
            logger.error("P1 BLOCK record published: %s (%s)", path, sha)
        except Exception:
            logger.exception("P1 BLOCK could not be published; the verdict "
                             "must not survive only as a log line")
            return EXIT_ERROR
        return EXIT_BLOCK
    except StageError as exc:
        logger.error("P1 ERROR [%s] -- %s", exc.error_code, exc.reason)
        publish_error(exc, args.out_dir, error_prefix, "P1",
                      parents=(parents or {}).get("p0"),
                      code={"script": script},
                      run={"argv": raw_argv})
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
        publish_error(wrapped, args.out_dir, error_prefix, "P1",
                      parents=(parents or {}).get("p0"),
                      code={"script": script}, run={"argv": raw_argv})
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
