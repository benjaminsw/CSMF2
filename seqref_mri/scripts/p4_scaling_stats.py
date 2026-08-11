# =============================================================================
# SEQREF-P4CS v0.2 -- scripts.p4_scaling_stats
# LIFETIME: KEEP
# Purpose: P4 census/support layer under A5 (Route C). Traverses the
#   REGISTERED MASK-STATISTICS FRAME (EXEC §13: full training split,
#   train_slices=None, epoch set {0}, hash-bound generator), accumulates the
#   sparse per-file free-column weights w_i(c) in EXACT integer arithmetic,
#   evaluates the structural guard, the zero/denominator cases and the
#   counting invariants, classifies every column (never-free /
#   under-supported / eligible) against N_EFF_MIN = 900 in integer form, and
#   evaluates the frozen non-gating Kish prediction. Publishes
#   seqref-p4-stats/1: CENSUS/SUPPORT/ELIGIBILITY ONLY. Per-location scaling
#   statistics and the branch vote are ABSENT BY SCHEMA (not nulled, not
#   placeholders); they arrive with the statistics layer as schema /2.
# Taxonomy (LOCK 2): PASS(0) / ERROR(2). NO BLOCK branch exists -- every
#   gate tests a construction, contract or counting invariant. EXIT_BLOCK is
#   unreachable by design, as under A4 for P3.
# Mask derivation: the mask depends only on (width, seed); kspace is NOT
#   read. Seeds are derived two independent ways from the hash-bound module
#   (public canonical_mask_seed AND the dataset's own _mask_seed call) and
#   must agree, else ERROR.
# CONVENTION: every failure path -> logger.error + typed raise (StageError).
#   No fallback, no mock, no placeholder, no silent pass.
# Changelog
#   v0.2 (2026-08-08) Pre-execution review fix (P4ST v0.1 gap analysis):
#     the registered frame declares the generator HASH-BOUND to SHA
#     610cc1d1..., but v0.1 only RECORDED the executing generator's hash.
#     Added GENERATOR_SOURCE_SHA256 pin + enforce_generator_pin(): a
#     mismatch is GENERATOR_HASH_MISMATCH (ERROR). No artefacts exist, so
#     no invalidation is triggered.
#   v0.1 (2026-08-08) Created under A5 for the combined A4+A5+§9 landing.
#     First P4 implementation target per the post-landing ruling: the
#     census/support layer BEFORE any scaling statistics or branch vote.
# =============================================================================
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

from preflight_io import canonical_hash, file_sha256  # noqa: E402
from preflight_parents import (EXIT_ERROR, EXIT_PASS, StageError,  # noqa: E402
                               attach_semantic_hash, environment_record,
                               guard_run_mode, hash_project_code,
                               publish_error, publish_stage, verify_parents)
from preflight_parents_p3 import (bind_mask_seed_provenance,  # noqa: E402
                                  dataset_provenance, verify_p1_p2)
from seqref_mri.src import fastmri_data as fdm  # noqa: E402

SCRIPT_ID = "SEQREF-P4CS"
SCRIPT_VERSION = "v0.2"
FACTS_SCHEMA = "seqref-p4-stats/1"
FACTS_PREFIX = "scaling_stats"
ERROR_PREFIX = "p4_error"
SMOKE_FACTS_PREFIX = "smoke_scaling_stats"
SMOKE_ERROR_PREFIX = "smoke_p4_error"

logger = logging.getLogger(SCRIPT_ID)

# ---- registered constants (EXEC §13, A5 Route C) ----------------------------
N_EFF_MIN = 900                 # ceil((kappa*-1)/(4*epsilon^2)); ERROR gate
EPSILON = 0.05                  # relative precision design value
KAPPA_STAR = 10                 # fourth-moment DESIGN BOUND, fixed a priori
GRID_WIDTH = 96                 # EXEC 3.1
EPOCH_SET = (0,)                # registered frame
REGISTERED_TRAIN_FILES = 973    # P4FS v0.2 structure evidence
REGISTERED_TRAIN_SLICES = 34742

# Parent pins (identical binding as P3; P4 is downstream of P2 per §9.6).
P1_FACTS_SHA256 = ("1d0f760043c7e46ce5da338d81eb053e2d9"
                   "e0135a25063512ab9395bb18aa3a2")
P2_FACTS_SHA256 = ("8c22a025853187816e63b0121da39e2f74c"
                   "dd60126368b8559690f02877aab31")
P1_SEMANTIC_SHA256 = ("3823e4489cb3eac6177b23f17db3aa5437a"
                      "c197ef5de92f3597e57f3a92e45d7")
P2_SEMANTIC_SHA256 = ("77da087e853dbe5eed1547c97fc328ee406"
                      "b0be6ac15d976aaf47867cb219000")

# The registered frame declares the mask generator HASH-BOUND (EXEC §13).
# A pin that is recorded but not enforced is documentation, not a binding:
# the executing generator's source hash must EQUAL this value or the run
# is ERROR.
GENERATOR_SOURCE_SHA256 = ("610cc1d1d7968deebc88f645270e1baefb6589cb56841b"
                           "dd327450ca1069cb44")

# Frozen NON-GATING prediction (A5 §0): evaluated by the census, never
# revised by it, never a second acceptance condition.
FROZEN_PREDICTION = {"point_estimate": 958, "anticipated_lo": 950,
                     "anticipated_hi": 965,
                     "falsification_floor": 950,
                     "predicted_under_supported_empty": True}


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without the dataset)
# ---------------------------------------------------------------------------

def centre_columns(width: int) -> frozenset:
    """Derived from the hash-bound generator's own mask_counts, never
    hard-coded: the centred fully-sampled block of make_cartesian_mask."""
    n_center, _ = fdm.mask_counts(width)
    start = (width - n_center) // 2
    return frozenset(range(start, start + n_center))


def expected_acquired_count(width: int) -> int:
    return fdm.mask_counts(width)[1]


def free_columns_of(mask: np.ndarray) -> tuple:
    return tuple(int(c) for c in np.flatnonzero(~mask))


def accumulate_weights(rows) -> tuple[dict, int]:
    """rows: iterable of (relpath, free_columns). Returns
    ({column: {relpath: w_i(c)}}, n_rows). Sparse: only nonzero pairs."""
    weights: dict[int, dict[str, int]] = {}
    n = 0
    for rel, free in rows:
        n += 1
        for c in free:
            per_file = weights.setdefault(c, {})
            per_file[rel] = per_file.get(rel, 0) + 1
    return weights, n


def validate_weight(col: int, rel: str, w) -> None:
    """ZERO/DENOMINATOR clause: every recorded weight must be a positive
    (non-negative by construction, positive if present) finite integer."""
    if isinstance(w, bool) or not isinstance(w, (int, np.integer)):
        logger.error("[P4] w_i(%d) for %s is not an integer: %r", col, rel, w)
        raise StageError("WEIGHT_NOT_INTEGER",
                         f"w_i({col}) for {rel} is not an integer: {w!r}")
    if not np.isfinite(w) or w <= 0:
        logger.error("[P4] w_i(%d) for %s is non-finite or non-positive: %r",
                     col, rel, w)
        raise StageError("WEIGHT_NON_FINITE_OR_NONPOSITIVE",
                         f"w_i({col}) for {rel} = {w!r} is impossible for a "
                         f"recorded sparse entry; indicates a defect")


def column_statistics(weights: dict, width: int) -> list:
    """Exact integer per-column statistics for ALL columns 0..width-1."""
    cols = []
    for c in range(width):
        per_file = weights.get(c, {})
        n_free_raw = 0
        sum_w2 = 0
        for rel, w in per_file.items():
            validate_weight(c, rel, w)
            n_free_raw += int(w)
            sum_w2 += int(w) * int(w)
        n_free_files = len(per_file)
        kish = (n_free_raw * n_free_raw / sum_w2) if sum_w2 > 0 else None
        cols.append({"column": c,
                     "n_free_raw": n_free_raw,
                     "n_free_files": n_free_files,
                     "sum_w2": sum_w2,
                     "n_eff_kish": kish,
                     "kish_note": (None if sum_w2 > 0 else
                                   "Kish NOT evaluated: never-free column"),
                     "weights": dict(per_file)})
    return cols


def classify_column(n_free_raw: int, sum_w2: int) -> str:
    """A5 taxonomy in EXACT INTEGER form: kish >= N_EFF_MIN is evaluated as
    n_free_raw^2 >= N_EFF_MIN * sum_w2 (no float division in the gate)."""
    if n_free_raw == 0:
        return "never-free"
    if n_free_raw * n_free_raw < N_EFF_MIN * sum_w2:
        return "under-supported"
    return "eligible"


def structural_guard(columns: list, centre: frozenset,
                     authoritative: bool) -> dict:
    """Construction check, not a data BLOCK. The centre-must-be-never-free
    leg is population-independent and gates ALWAYS; the non-centre-positive
    leg gates only on the full frame (a small smoke population may
    legitimately miss a column)."""
    centre_free = [c["column"] for c in columns
                   if c["column"] in centre and c["n_free_raw"] > 0]
    if centre_free:
        logger.error("[P4] centre columns observed FREE: %s -- contradicts "
                     "the registered mask-family construction", centre_free)
        raise StageError("CENTRE_COLUMN_OBSERVED_FREE",
                         f"centre columns {centre_free} have n_free_raw > 0; "
                         f"the observation contradicts mask_counts/"
                         f"make_cartesian_mask construction")
    noncentre_zero = [c["column"] for c in columns
                      if c["column"] not in centre and c["n_free_raw"] == 0]
    if noncentre_zero and authoritative:
        logger.error("[P4] non-centre columns never free over the FULL "
                     "frame: %s", noncentre_zero)
        raise StageError("NONCENTRE_COLUMN_NEVER_FREE",
                         f"columns {noncentre_zero} have n_free_raw = 0 over "
                         f"the full registered frame; contradicts the "
                         f"registered mask family")
    return {"centre_columns": sorted(centre),
            "centre_never_free_ok": not centre_free,
            "noncentre_zero_columns": noncentre_zero,
            "noncentre_positive_gated": authoritative,
            "noncentre_positive_note": (
                "gated: full frame" if authoritative else
                "RECORDED ONLY: smoke population may legitimately miss a "
                "column; gates on the authoritative run")}


def counting_invariants(columns: list) -> dict:
    """The §13 safeguard block, evaluated for every column. Any failure is
    ERROR: an impossible count indicates a construction defect."""
    checks = {"non_negative_integers": True,
              "files_le_raw": True,
              "kish_bounds": True,
              "cauchy_schwarz": True}
    for rec in columns:
        raw, files, w2 = rec["n_free_raw"], rec["n_free_files"], rec["sum_w2"]
        for name, v in (("n_free_raw", raw), ("n_free_files", files),
                        ("sum_w2", w2)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                checks["non_negative_integers"] = False
                bad = (rec["column"], name, v)
                break
        else:
            bad = None
        if bad is not None:
            logger.error("[P4] counting invariant failed at %s", bad)
            raise StageError("COUNTING_INVARIANT_VIOLATED",
                             f"column {bad[0]}: {bad[1]} = {bad[2]!r} is "
                             f"not a non-negative integer")
        if files > raw:
            checks["files_le_raw"] = False
        if raw * raw > files * w2:
            checks["cauchy_schwarz"] = False
        if raw > 0:
            kish_ok = (raw * raw >= w2 and           # kish >= 1
                       raw * raw <= files * w2 and   # kish <= n_free_files
                       files <= REGISTERED_TRAIN_FILES)
            if not kish_ok:
                checks["kish_bounds"] = False
        if not all(checks.values()):
            logger.error("[P4] counting invariant failed at column %d: %s",
                         rec["column"], checks)
            raise StageError("COUNTING_INVARIANT_VIOLATED",
                             f"column {rec['column']}: {checks}")
    return dict(checks, all_ok=True,
                n_files_bound=REGISTERED_TRAIN_FILES)


def evaluate_frozen_prediction(columns: list, centre: frozenset) -> dict:
    """The frozen prediction is EVALUATED and RECORDED; it is evidence
    evaluation, never a second acceptance condition (A5 §0)."""
    kishs = [c["n_eff_kish"] for c in columns
             if c["column"] not in centre and c["n_free_raw"] > 0]
    if not kishs:
        # No positive-support non-centre column: nothing to evaluate. The
        # structural guard has already decided whether that is an ERROR.
        return {"gating": False, "evaluated": False,
                "reason": "no positive-support non-centre column in this "
                          "population"}
    kmin, kmax = min(kishs), max(kishs)
    all_pass = all(c["n_free_raw"] * c["n_free_raw"]
                   >= N_EFF_MIN * c["sum_w2"]
                   for c in columns if c["n_free_raw"] > 0)
    falsified = kmin < FROZEN_PREDICTION["falsification_floor"]
    return {"gating": False, "evaluated": True,
            "prediction": dict(FROZEN_PREDICTION),
            "n_noncentre_positive_columns": len(kishs),
            "kish_min": kmin, "kish_max": kmax,
            "kish_median": float(np.median(kishs)),
            "all_columns_pass_gate": all_pass,
            "min_within_anticipated_range":
                FROZEN_PREDICTION["anticipated_lo"] <= kmin <=
                FROZEN_PREDICTION["anticipated_hi"],
            "finding": ("falsified_model_recorded" if falsified
                        else "consistent_with_prediction"),
            "note": "a falsified model triggers documented diagnostic "
                    "review; it never alters N_EFF_MIN, the taxonomy, or "
                    "the verdict"}


def census_core(rows, width: int, authoritative: bool) -> dict:
    """Full census pipeline over (relpath, free_columns) rows. Pure: no I/O,
    no dataset. Every gate is a construction/counting ERROR, never BLOCK."""
    centre = centre_columns(width)
    weights, n_rows = accumulate_weights(rows)
    columns = column_statistics(weights, width)
    guard = structural_guard(columns, centre, authoritative)
    invariants = counting_invariants(columns)
    for rec in columns:
        rec["class"] = classify_column(rec["n_free_raw"], rec["sum_w2"])
    classes = {"never-free": 0, "under-supported": 0, "eligible": 0}
    under = []
    for rec in columns:
        classes[rec["class"]] += 1
        if rec["class"] == "under-supported":
            under.append(rec["column"])
    if under and authoritative:
        logger.error("[P4] under-supported columns over the full frame: %s",
                     under)
        raise StageError("UNDER_SUPPORTED_COLUMN",
                         f"columns {under}: n_free_raw^2 < N_EFF_MIN*sum_w2 "
                         f"-- underdetermined scaling contract (A5); a "
                         f"narrow gate failure is reported under the "
                         f"registered rule, never repaired by relaxing it")
    prediction = evaluate_frozen_prediction(columns, centre)
    files = sorted({rel for per in weights.values() for rel in per})
    return {"columns": columns, "classes": classes,
            "under_supported_columns": under,
            "under_supported_gated": authoritative,
            "structural_guard": guard, "counting_invariants": invariants,
            "frozen_prediction": prediction,
            "files": files, "n_rows": n_rows,
            "centre_columns": sorted(centre)}


# ---------------------------------------------------------------------------
# Frame traversal (live generator, hash-bound; kspace never read)
# ---------------------------------------------------------------------------

def enforce_generator_pin(seed_prov: dict) -> None:
    """The frame's hash-binding is a GATE, not a record: the executing
    generator's source hash (read by bind_mask_seed_provenance from the
    running module, never asserted) must equal the registered pin."""
    got = (seed_prov or {}).get("mask_seed_source_sha256")
    if not (seed_prov or {}).get("resolved") or got != GENERATOR_SOURCE_SHA256:
        logger.error("[P4] generator hash %s != registered pin %s",
                     got, GENERATOR_SOURCE_SHA256)
        raise StageError("GENERATOR_HASH_MISMATCH",
                         f"the executing mask generator hashes to {got}, "
                         f"but the registered frame is hash-bound to "
                         f"{GENERATOR_SOURCE_SHA256}; the generator changed "
                         f"(or provenance is unbound) and the census frame "
                         f"is no longer the registered one")


def traverse_frame(data_root: str, smoke):
    """Traverse the REGISTERED MASK-STATISTICS FRAME. The mask is a pure
    function of (width, seed), so the census reads NO kspace: it enumerates
    the dataset's own index (file, slice), derives each seed TWO ways from
    the hash-bound module -- the public canonical_mask_seed and the
    dataset's own _mask_seed (the exact call __getitem__ makes) -- and
    requires agreement. Returns (rows, dataset) with rows of
    (relpath, free_columns)."""
    ds = fdm.FastMRISliceDataset(data_root, split="train", mode="train")
    ds.set_epoch(EPOCH_SET[0])
    index = ds.index if smoke is None else ds.index[:smoke]
    n_acquired_expected = expected_acquired_count(fdm.CELL_HW)
    rows = []
    seen = set()
    for path, s in index:
        rel = path.relative_to(ds.data_root).as_posix()
        key = (rel, int(s))
        if key in seen:
            logger.error("[P4] duplicate frame row %s", key)
            raise StageError("DUPLICATE_FRAME_ROW",
                             f"frame row {key} appears twice; the "
                             f"realisation count is derived, never chosen, "
                             f"so duplication is a construction defect")
        seen.add(key)
        seed_public = fdm.canonical_mask_seed(fdm.TRAIN_BASE_SEED, rel,
                                              int(s), epoch=EPOCH_SET[0])
        seed_dataset = ds._mask_seed(path, int(s))
        if seed_public != seed_dataset:
            logger.error("[P4] seed derivation mismatch at %s: public=%d "
                         "dataset=%d", key, seed_public, seed_dataset)
            raise StageError("SEED_DERIVATION_MISMATCH",
                             f"canonical_mask_seed and dataset._mask_seed "
                             f"disagree at {key}; the hash-bound generator "
                             f"is internally inconsistent")
        mask = fdm.make_cartesian_mask(fdm.CELL_HW, seed_public)
        if int(mask.sum()) != n_acquired_expected:
            logger.error("[P4] acquired count %d != %d at %s",
                         int(mask.sum()), n_acquired_expected, key)
            raise StageError("ACQUIRED_COUNT_MISMATCH",
                             f"acquired count varies at {key}; "
                             f"make_cartesian_mask guarantees exact counts, "
                             f"so this indicates a generator defect")
        rows.append((rel, free_columns_of(mask)))
    return rows, ds


def check_population(rows: list, files: list, authoritative: bool) -> dict:
    """The realisation count is DERIVED from the population, never chosen.
    On the full frame the derived counts must equal the registered
    population; a mismatch means the inputs changed (§9.6 territory)."""
    observed = {"n_slices": len(rows), "n_files": len(files)}
    ok = (observed["n_slices"] == REGISTERED_TRAIN_SLICES and
          observed["n_files"] == REGISTERED_TRAIN_FILES)
    if authoritative and not ok:
        logger.error("[P4] frame population %s != registered (%d slices, "
                     "%d files)", observed, REGISTERED_TRAIN_SLICES,
                     REGISTERED_TRAIN_FILES)
        raise StageError("FRAME_POPULATION_MISMATCH",
                         f"derived frame population {observed} does not "
                         f"equal the registered {REGISTERED_TRAIN_SLICES} "
                         f"slices / {REGISTERED_TRAIN_FILES} files; the "
                         f"dataset or extraction changed")
    return {**observed,
            "registered": {"n_slices": REGISTERED_TRAIN_SLICES,
                           "n_files": REGISTERED_TRAIN_FILES},
            "gated": authoritative,
            "note": ("gated: full frame" if authoritative else
                     "RECORDED ONLY: smoke traverses a prefix of the "
                     "frame by design")}


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

def _p4_local_sha(repo_dir: str, relpath: str) -> str | None:
    path = os.path.join(repo_dir, relpath)
    return file_sha256(path) if os.path.isfile(path) else None


def _build_facts(parents, p1p2, census, population, grid_width,
                 verdict, reason, repo_dir, script, argv, t0, smoke,
                 seed_prov, dataset_prov) -> dict:
    centre = census["centre_columns"]
    # Artefact column records: full sparse w_i(c) as {file_index: w} over
    # the artefact's own files table, plus the exact sufficient statistics.
    file_index = {rel: i for i, rel in enumerate(census["files"])}
    columns_out = []
    for rec in census["columns"]:
        columns_out.append({
            "column": rec["column"], "class": rec["class"],
            "n_free_raw": rec["n_free_raw"],
            "n_free_files": rec["n_free_files"],
            "sum_w2": rec["sum_w2"],
            "n_eff_kish": rec["n_eff_kish"],
            "kish_note": rec["kish_note"],
            "w_i_sparse": {str(file_index[rel]): int(w)
                           for rel, w in sorted(rec["weights"].items())}})
    thresholds = {"N_EFF_MIN": N_EFF_MIN, "EPSILON": EPSILON,
                  "KAPPA_STAR": KAPPA_STAR,
                  "GRID_WIDTH": grid_width,
                  "EPOCH_SET": list(EPOCH_SET),
                  "REGISTERED_TRAIN_FILES": REGISTERED_TRAIN_FILES,
                  "REGISTERED_TRAIN_SLICES": REGISTERED_TRAIN_SLICES,
                  "centre_columns_derived": centre,
                  "n_acquired_expected":
                      expected_acquired_count(grid_width),
                  "gate_form": "exact integer: n_free_raw^2 vs "
                               "N_EFF_MIN*sum_w2; no float division in any "
                               "gate"}
    frame = {"population": "full training split",
             "train_slices": None, "epoch_set": list(EPOCH_SET),
             "mask_mode": "train",
             "generator": "seqref_mri/src/fastmri_data.py (SEQREF-I1 v0.3)",
             "generator_source_sha256":
                 _p4_local_sha(repo_dir, "seqref_mri/src/fastmri_data.py"),
             "base_seed": fdm.TRAIN_BASE_SEED,
             "seed_tuple_serialization":
                 "train: '{base_seed}|{epoch}|{relpath}|{slice_index}' "
                 "(UTF-8 -> SHA-256 -> first 8 bytes big-endian)",
             "realisation_count": ("DERIVED = n_slices x |epochs|; never a "
                                   "chosen parameter"),
             "kspace_read": False,
             "mask_source": "mask is a pure function of (width, seed); "
                            "seeds derived via canonical_mask_seed AND the "
                            "dataset's own _mask_seed, agreement required"}
    summary = {"verdict": verdict, "verdict_reason": reason,
               "smoke": smoke is not None,
               "smoke_slices": smoke,
               "n_rows": census["n_rows"],
               "classes": census["classes"],
               "n_files_observed": len(census["files"])}
    parent_ids = {
        "p0_facts_sha256": parents["p0"]["facts_sha256"] if parents else None,
        "p0s_facts_sha256": parents["p0s"]["facts_sha256"] if parents else None,
        "subset_manifest_sha256":
            parents["p0s"]["subset_manifest_sha256"] if parents else None,
        "contract_hash": parents["p0"]["contract_hash"] if parents else None,
        "p1_facts_sha256": p1p2["p1"]["facts_sha256"] if p1p2 else None,
        "p1_semantic_sha256": p1p2["p1"]["semantic_sha256"] if p1p2 else None,
        "p2_facts_sha256": p1p2["p2"]["facts_sha256"] if p1p2 else None,
        "p2_semantic_sha256": p1p2["p2"]["semantic_sha256"] if p1p2 else None,
        "p1_ruling": p1p2["p1"]["ruling"] if p1p2 else None}
    code = hash_project_code(repo_dir, script)
    p4_local = {"p4_local": [
        {"relpath": "seqref_mri/src/fastmri_data.py",
         "sha256": _p4_local_sha(repo_dir, "seqref_mri/src/fastmri_data.py")}]}
    semantic = {"schema": FACTS_SCHEMA, "stage": "P4",
                "thresholds": thresholds, "verdict": verdict,
                "frame": frame, "population": population,
                "columns": [{k: v for k, v in c.items()}
                            for c in columns_out],
                "structural_guard": census["structural_guard"],
                "counting_invariants": census["counting_invariants"],
                "frozen_prediction": census["frozen_prediction"],
                "summary": summary, "parents": parent_ids,
                "code": code["project_local"] + p4_local["p4_local"]}
    facts = {
        "schema": FACTS_SCHEMA,
        "script": {"id": SCRIPT_ID, "version": SCRIPT_VERSION,
                   "lifetime": "KEEP"},
        "stage": "P4",
        "artefact_type": "stage_facts",
        "run_mode": ("smoke" if smoke is not None else "authoritative"),
        "authoritative": smoke is None,
        "stage_description": "census/support layer (A5 Route C): sparse "
                             "w_i(c), exact per-column statistics, "
                             "structural guard, counting invariants, "
                             "eligibility classification, frozen-prediction "
                             "evaluation",
        "schema_scope": {
            "covers": ["census", "support", "eligibility"],
            "absent_by_schema": ["per_location_scaling_statistics",
                                 "branch_vote"],
            "note": "seqref-p4-stats/1 carries the census/support layer "
                    "ONLY. Scaling statistics and the branch vote are "
                    "ABSENT BY SCHEMA -- not null, not placeholder. They "
                    "arrive with the statistics layer as schema /2, which "
                    "a /1 consumer must reject rather than reinterpret."},
        "thresholds": thresholds,
        "verdict": verdict,
        "verdict_reason": reason,
        "frame": frame,
        "population": population,
        "files": census["files"],
        "columns": columns_out,
        "class_counts": census["classes"],
        "under_supported_columns": census["under_supported_columns"],
        "structural_guard": census["structural_guard"],
        "counting_invariants": census["counting_invariants"],
        "frozen_prediction": census["frozen_prediction"],
        "mask_seed_provenance": seed_prov or {"resolved": False},
        "dataset_provenance": dataset_prov or {},
        "summary": summary,
        "parents": {"p0_p0s": parents, "p1_p2": (
            {k: v for k, v in p1p2.items() if k != "p2_by_index"}
            if p1p2 else None)},
        "code": {**code, **p4_local},
        "run": {**environment_record(repo_dir, argv),
                "runtime_seconds": time.time() - t0},
        "hash_note": "the authoritative artefact SHA is the SHA-256 of "
                     "THIS FILE'S bytes, in the sidecar; semantic_sha256 "
                     "covers scientific content only",
    }
    return attach_semantic_hash(facts, semantic)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=f"{SCRIPT_ID} {SCRIPT_VERSION} -- P4 census/support "
                    "layer (A5 Route C)")
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--p0-facts", required=True)
    ap.add_argument("--p0s-facts", required=True)
    ap.add_argument("--p0s-script", required=True)
    ap.add_argument("--p1-facts", required=True)
    ap.add_argument("--p2-facts", required=True)
    ap.add_argument("--out-dir", required=True,
                    help="P4 output directory; for a smoke run this must be "
                         "an EPHEMERAL directory, never the parents' directory")
    ap.add_argument("--smoke", type=int, default=None,
                    help="EPHEMERAL: first N frame rows, smoke_ prefix; "
                         "never authoritative")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    t0 = time.time()
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    # `is not None`, not truthiness: --smoke 0 is invalid input, but the
    # error record for it must still carry the SMOKE prefix.
    facts_prefix = SMOKE_FACTS_PREFIX if args.smoke is not None \
        else FACTS_PREFIX
    error_prefix = SMOKE_ERROR_PREFIX if args.smoke is not None \
        else ERROR_PREFIX
    script = os.path.abspath(__file__)
    parents = p1p2 = None
    try:
        if args.smoke is not None and args.smoke <= 0:
            raise StageError("BAD_SMOKE_SIZE",
                             f"--smoke must be a positive int, got "
                             f"{args.smoke!r}")
        run_mode = guard_run_mode(args.out_dir, args.smoke is not None)
        logger.info("%s run_mode=%s out_dir=%s", SCRIPT_ID, run_mode,
                    args.out_dir)
        parents = verify_parents(args.repo_dir, args.p0_facts,
                                 args.p0s_facts, args.p0s_script)
        p1p2 = verify_p1_p2(
            args.p1_facts, args.p2_facts,
            expected_p1_sha=P1_FACTS_SHA256,
            expected_p2_sha=P2_FACTS_SHA256,
            expected_p1_semantic_sha=P1_SEMANTIC_SHA256,
            expected_p2_semantic_sha=P2_SEMANTIC_SHA256)
        seed_prov = bind_mask_seed_provenance(args.repo_dir)
        enforce_generator_pin(seed_prov)
        rows, ds = traverse_frame(args.data_root, args.smoke)
        census = census_core(rows, fdm.CELL_HW, args.smoke is None)
        population = check_population(rows, census["files"],
                                      args.smoke is None)
        dataset_prov = dataset_provenance(type(ds), ds)
        reason = (f"the census traversed the registered frame "
                  f"({census['n_rows']} rows, {len(census['files'])} files), "
                  f"all counting invariants and the structural guard hold, "
                  f"and the A5 support classification is computed in exact "
                  f"integer arithmetic; classes={census['classes']}")
        facts = _build_facts(parents, p1p2, census, population, fdm.CELL_HW,
                             "PASS", reason, args.repo_dir, script, raw_argv,
                             t0, args.smoke, seed_prov, dataset_prov)
        path, sha = publish_stage(facts, args.out_dir, facts_prefix, "P4")
        logger.info("P4 census PASS n=%d files=%d classes=%s facts=%s "
                    "file_sha256=%s semantic_sha256=%s", census["n_rows"],
                    len(census["files"]), census["classes"], path, sha,
                    facts["semantic_sha256"])
        if args.smoke is not None:
            logger.warning("SMOKE run -- NOT authoritative; delete %s after "
                           "inspection", path)
        return EXIT_PASS

    # NO BLOCK HANDLER. Every gate in this stage tests a construction,
    # contract or counting invariant (LOCK 2); EXIT_BLOCK is unreachable by
    # design. A future data premise must reintroduce the handler explicitly.
    except StageError as exc:
        logger.error("P4 ERROR [%s] -- %s", exc.error_code, exc.reason)
        publish_error(exc, args.out_dir, error_prefix, "P4",
                      parents=(parents or {}).get("p0"),
                      code={"script": script}, run={"argv": raw_argv})
        return EXIT_ERROR
    except Exception as exc:
        # Failure boundary. KeyboardInterrupt/SystemExit are deliberately
        # NOT caught. An ordinary exception must never surface as a bare
        # traceback: no exit-code contract, no artefact.
        logger.exception("%s UNEXPECTED ERROR", SCRIPT_ID)
        wrapped = StageError(
            "UNEXPECTED_RUNTIME_ERROR", f"{type(exc).__name__}: {exc}",
            detail={"exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "raised_after_parent_verification": parents is not None},
            write_record=parents is not None)
        publish_error(wrapped, args.out_dir, error_prefix, "P4",
                      parents=(parents or {}).get("p0"),
                      code={"script": script}, run={"argv": raw_argv})
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
