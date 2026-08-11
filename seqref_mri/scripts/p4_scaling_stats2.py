# =============================================================================
# SEQREF-P4S2 v0.1 -- scripts.p4_scaling_stats2
# LIFETIME: KEEP
# Purpose: P4 statistics layer, schema seqref-p4-stats/2, registered in EXEC
#   §13 (P4 /2 STATISTICS LAYER, 2026-08-09) and clarified in Concept §3.6
#   (D4 statistics semantics, preregistered clarification before observation,
#   2026-08-09) BEFORE any /2 code was built or any /2 quantity observed.
#   The stage INHERITS the eligible-column set from the authoritative
#   seqref-p4-stats/1 parent, pinned by BOTH hashes (byte and semantic);
#   it never re-derives eligibility. It traverses the REGISTERED
#   MASK-STATISTICS FRAME in dataset index (manifest) order, reads k-space
#   through the registered Construction-A preparation, normalises by
#   meta.file_attr_max (D2) on the registered /2 float64 arithmetic path,
#   and accumulates per-(r,c,channel) count/mean/M2 over ALL free
#   observations at eligible locations (ddof=0). It then computes the
#   pooled mean_global(ch)/sigma_global(ch) over the SAME population,
#   floor(ch) = 1e-2*sigma_global(ch) (D4 factor UNCHANGED), STRICT-<
#   floor hits on raw_std, per_location_scale = max(raw_std, floor), the
#   integer branch rule PER-LOCATION iff 20*n_floor_hits <= n_eligible
#   (5% threshold UNCHANGED), the branch-selected applied affine pair, the
#   pre-vote validity of BOTH candidate scale families, and the C7 affine
#   round-trip over the ACTUALLY SELECTED pair at <= 1e-12.
# Parent binding: the /1 artefact is verified against the registered pins
#   P4S1_FILE_SHA256 / P4S1_SEMANTIC_SHA256; a mismatch is ERROR. On an
#   authoritative run the stage's own sparse w_i(c) must equal the parent
#   table EXACTLY and the transpose invariant count(r,c) == n_free_raw(c)
#   must hold; both gates are authoritative-only and never fire at smoke
#   scale. Per-slice mask identity/seed consistency against the registered
#   generator is checked WHILE accumulating (guardrail against slice<->mask
#   binding drift), not only at the aggregate level.
# Taxonomy (LOCK 2): PASS(0) / ERROR(2). NO BLOCK branch exists -- every
#   gate tests a construction, contract or counting invariant. EXIT_BLOCK is
#   unreachable by design, as under A4 for P3 and A5 for the /1 stage.
# Smoke doctrine (as /1): a smoke run records and exercises but never
#   gates the population, full-coverage, parity or transpose checks; its
#   branch decision is a smoke-scale evaluation, never the verdict.
# CONVENTION: every failure path -> logger.error + typed raise (StageError).
#   No fallback, no mock, no placeholder, no silent pass.
# Changelog
#   v0.1 (2026-08-09) Created against the frozen /2 registration: EXEC §13
#     P4 /2 block (dual-hash parent pin, applied affine pair, pre-vote
#     validity, integer branch, C7 on the selected pair, arithmetic path,
#     per-slice consistency, authoritative-only parity) and Concept D4
#     statistics-semantics clarification. Consumes schema /1; never
#     overwrites it; publishes p4/scaling_statistics.json.
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

from preflight_io import canonical_hash, file_sha256  # noqa: E402
from preflight_parents import (EXIT_ERROR, EXIT_PASS, StageError,  # noqa: E402
                               attach_semantic_hash, environment_record,
                               guard_run_mode, hash_project_code,
                               publish_error, publish_stage)
from preflight_parents_p3 import (bind_mask_seed_provenance,  # noqa: E402
                                  dataset_provenance)
from seqref_mri.src import fastmri_data as fdm  # noqa: E402

SCRIPT_ID = "SEQREF-P4S2"
SCRIPT_VERSION = "v0.1"
FACTS_SCHEMA = "seqref-p4-stats/2"
PARENT_SCHEMA = "seqref-p4-stats/1"
FACTS_PREFIX = "scaling_statistics"
ERROR_PREFIX = "p4s2_error"
SMOKE_FACTS_PREFIX = "smoke_scaling_statistics"
SMOKE_ERROR_PREFIX = "smoke_p4s2_error"

logger = logging.getLogger(SCRIPT_ID)

# ---- registered constants (EXEC §13 P4 /2 block + Concept D4) --------------
GRID_HW = fdm.CELL_HW             # 96 x 96 cell (EXEC 3.1)
EPOCH_SET = (0,)                  # registered mask-statistics frame
FLOOR_FACTOR = 1e-2               # D4 variance floor factor -- UNCHANGED
BRANCH_DENOM = 20                 # 0.05 = 1/20; integer branch form
C7_RTOL = 1e-12                   # affine round-trip bound (ERROR class)
DDOF = 0                          # population std, registered
REGISTERED_TRAIN_FILES = 973      # P4FS v0.2 structure evidence (as /1)
REGISTERED_TRAIN_SLICES = 34742
REGISTERED_ELIGIBLE_COLUMNS = 88  # /1 census: every non-centre column

# /1 parent pins (EXEC §13 P4 /2 block): BOTH hashes, registered pre-build.
P4S1_FILE_SHA256 = ("9ec63dfdc384d3ff8541e3346bd214fa02916defa1065f"
                    "bc442d4da0110025c4")
P4S1_SEMANTIC_SHA256 = ("a4d64709549765bca09031a9d951e3c1a90d016a94"
                        "06518144e129210e69bac6")

# The registered frame declares the mask generator HASH-BOUND (EXEC §13);
# same binding as the /1 stage: the executing generator's source hash must
# EQUAL this value or the run is ERROR.
GENERATOR_SOURCE_SHA256 = ("610cc1d1d7968deebc88f645270e1baefb6589cb56841b"
                           "dd327450ca1069cb44")

CHANNELS = ("re", "im")


# ---------------------------------------------------------------------------
# /1 parent: dual-hash pin + eligible-set inheritance
# ---------------------------------------------------------------------------

def load_p4s1_parent(path: str) -> dict:
    """Verify the authoritative /1 parent against BOTH registered pins and
    extract the inherited eligible set and sparse weight table. Every
    mismatch is ERROR: the /2 statistics layer is defined only over the
    eligible set certified by THAT exact parent artefact."""
    if not os.path.isfile(path):
        logger.error("[P4S2] /1 parent not found: %s", path)
        raise StageError("PARENT_NOT_FOUND",
                         f"the /1 parent artefact does not exist at {path}")
    file_sha = file_sha256(path)
    if file_sha != P4S1_FILE_SHA256:
        logger.error("[P4S2] /1 parent file hash %s != pin %s", file_sha,
                     P4S1_FILE_SHA256)
        raise StageError("PARENT_FILE_HASH_MISMATCH",
                         f"the /1 parent hashes to {file_sha}, but the "
                         f"registered pin is {P4S1_FILE_SHA256}; the parent "
                         f"is not the artefact the /2 layer was registered "
                         f"against")
    sidecar_path = path + ".sha256"
    sidecar_present = os.path.isfile(sidecar_path)
    if sidecar_present:
        with open(sidecar_path, "r", encoding="utf-8") as fh:
            sidecar = fh.read().split()[0].strip()
        if sidecar != file_sha:
            logger.error("[P4S2] /1 sidecar %s != computed file hash %s",
                         sidecar, file_sha)
            raise StageError("PARENT_SIDECAR_MISMATCH",
                             f"the /1 sidecar records {sidecar} but the "
                             f"file computes to {file_sha}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            art = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.error("[P4S2] /1 parent unparsable: %s", exc)
        raise StageError("PARENT_STRUCTURE_INVALID",
                         f"the /1 parent at {path} is not valid JSON: {exc}")
    if art.get("schema") != PARENT_SCHEMA:
        logger.error("[P4S2] /1 parent schema %r != %r", art.get("schema"),
                     PARENT_SCHEMA)
        raise StageError("PARENT_SCHEMA_MISMATCH",
                         f"expected schema {PARENT_SCHEMA}, got "
                         f"{art.get('schema')!r}; a /2 consumer must reject "
                         f"any other schema rather than reinterpret it")
    if not (art.get("authoritative") and art.get("run_mode") == "authoritative"
            and art.get("verdict") == "PASS"):
        logger.error("[P4S2] /1 parent is not an authoritative PASS: "
                     "authoritative=%r run_mode=%r verdict=%r",
                     art.get("authoritative"), art.get("run_mode"),
                     art.get("verdict"))
        raise StageError("PARENT_NOT_AUTHORITATIVE_PASS",
                         "the /2 layer inherits only from an authoritative "
                         "PASS /1 artefact")
    sem = art.get("semantic_sha256")
    if sem != P4S1_SEMANTIC_SHA256:
        logger.error("[P4S2] /1 parent semantic hash %s != pin %s", sem,
                     P4S1_SEMANTIC_SHA256)
        raise StageError("PARENT_SEMANTIC_HASH_MISMATCH",
                         f"the /1 parent's embedded semantic_sha256 is "
                         f"{sem}, but the registered pin is "
                         f"{P4S1_SEMANTIC_SHA256}")
    files = art.get("files")
    columns = art.get("columns")
    if not isinstance(files, list) or not isinstance(columns, list) \
            or len(columns) != GRID_HW:
        logger.error("[P4S2] /1 parent structure invalid: files/columns")
        raise StageError("PARENT_STRUCTURE_INVALID",
                         "the /1 parent lacks a files table or a full "
                         "per-column record list")
    centre = centre_columns(GRID_HW)
    eligible = sorted(c["column"] for c in columns
                      if c.get("class") == "eligible")
    noncentre = [c for c in range(GRID_HW) if c not in centre]
    if eligible != noncentre or len(eligible) != REGISTERED_ELIGIBLE_COLUMNS:
        logger.error("[P4S2] inherited eligible set %s != expected non-centre "
                     "set of %d columns", eligible, REGISTERED_ELIGIBLE_COLUMNS)
        raise StageError("PARENT_ELIGIBLE_SET_UNEXPECTED",
                         f"the inherited eligible set has "
                         f"{len(eligible)} columns (expected "
                         f"{REGISTERED_ELIGIBLE_COLUMNS}, exactly the "
                         f"non-centre columns); the /2 registration fixes "
                         f"96 x 88 = 8,448 eligible locations")
    # Parent sparse weight table mapped back to relpaths.
    parent_w: dict[int, dict[str, int]] = {}
    parent_n_free_raw: dict[int, int] = {}
    for c in columns:
        col = int(c["column"])
        parent_n_free_raw[col] = int(c["n_free_raw"])
        parent_w[col] = {files[int(k)]: int(v)
                         for k, v in (c.get("w_i_sparse") or {}).items()}
    return {"path": path, "file_sha256": file_sha, "semantic_sha256": sem,
            "sidecar_present": sidecar_present, "files": files,
            "eligible_columns": eligible, "parent_w": parent_w,
            "parent_n_free_raw": parent_n_free_raw,
            "grandparents": art.get("parents")}


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without the dataset)
# ---------------------------------------------------------------------------

def centre_columns(width: int) -> frozenset:
    """Derived from the hash-bound generator's own mask_counts, never
    hard-coded (identical binding as the /1 stage)."""
    n_center, _ = fdm.mask_counts(width)
    start = (width - n_center) // 2
    return frozenset(range(start, start + n_center))


def expected_acquired_count(width: int) -> int:
    return fdm.mask_counts(width)[1]


def free_columns_of(mask: np.ndarray) -> tuple:
    return tuple(int(c) for c in np.flatnonzero(~mask))


def new_accumulator(width: int) -> dict:
    """Per-location two-channel Welford state over the full grid. Only
    eligible locations are ever updated; the rest must remain count 0."""
    return {"count": np.zeros((width, width, 2), dtype=np.int64),
            "mean": np.zeros((width, width, 2), dtype=np.float64),
            "M2": np.zeros((width, width, 2), dtype=np.float64)}


def accumulate_observations(acc: dict, free_cols: tuple,
                            vals: np.ndarray) -> None:
    """Welford update over one slice's free-column observations.
    vals: complex128 (H, n_free) -- F x_norm at the free columns. Channel
    order is the registered (re, im). Vectorised per channel; ddof=0 is
    realised later as sqrt(M2/n)."""
    if vals.shape != (GRID_HW, len(free_cols)):
        logger.error("[P4S2] observation block shape %s != (%d, %d)",
                     vals.shape, GRID_HW, len(free_cols))
        raise StageError("OBSERVATION_BLOCK_SHAPE",
                         f"vals shape {vals.shape} does not match "
                         f"({GRID_HW}, {len(free_cols)})")
    if not np.isfinite(vals.real).all() or not np.isfinite(vals.imag).all():
        logger.error("[P4S2] non-finite observation in free block")
        raise StageError("NON_FINITE_OBSERVATION",
                         "a free-coefficient observation is non-finite")
    cols = np.asarray(free_cols, dtype=np.int64)
    for ch, comp in ((0, vals.real), (1, vals.imag)):
        n = acc["count"][:, cols, ch]
        delta = comp - acc["mean"][:, cols, ch]
        n += 1
        acc["mean"][:, cols, ch] += delta / n
        acc["M2"][:, cols, ch] += delta * (comp - acc["mean"][:, cols, ch])
        acc["count"][:, cols, ch] = n


def accumulate_weights(weights: dict, rel: str, free_cols: tuple) -> None:
    """Sparse per-file free-column weights w_i(c), exact integers (the
    stage's OWN census, later compared against the /1 parent table on an
    authoritative run)."""
    for c in free_cols:
        per_file = weights.setdefault(int(c), {})
        per_file[rel] = per_file.get(rel, 0) + 1


def pooled_channel_moments(count: np.ndarray, mean: np.ndarray,
                           M2: np.ndarray, eligible: list,
                           ch: int) -> tuple:
    """Pooled first/second moments over ALL free observations at ALL
    eligible locations in channel ch -- one population, the SAME pool for
    the mean and the std (parallel combination of per-location sufficient
    statistics, Chan et al.; ddof=0)."""
    idx_r, idx_c = np.meshgrid(np.arange(GRID_HW),
                               np.asarray(eligible, dtype=np.int64),
                               indexing="ij")
    n = count[idx_r, idx_c, ch].astype(np.float64)
    m = mean[idx_r, idx_c, ch]
    m2 = M2[idx_r, idx_c, ch]
    observed = n > 0
    N = float(n[observed].sum())
    if N <= 0.0:
        logger.error("[P4S2] channel %s has zero pooled observations", ch)
        raise StageError("EMPTY_CHANNEL_POPULATION",
                         f"channel {ch}: no eligible location was observed; "
                         f"the pooled global moments are undefined")
    mean_g = float((n[observed] * m[observed]).sum() / N)
    m2_g = float((m2[observed]
                  + n[observed] * (m[observed] - mean_g) ** 2).sum())
    sigma_g = float(np.sqrt(m2_g / N))
    return mean_g, sigma_g, N


def branch_decision(n_floor_hits: int, n_eligible: int) -> dict:
    """The registered INTEGER branch rule: PER-LOCATION iff
    20*n_floor_hits <= n_eligible (0.05 = 1/20; no float division)."""
    if not isinstance(n_floor_hits, int) or not isinstance(n_eligible, int) \
            or n_floor_hits < 0 or n_eligible <= 0:
        logger.error("[P4S2] branch operands invalid: hits=%r eligible=%r",
                     n_floor_hits, n_eligible)
        raise StageError("BRANCH_OPERANDS_INVALID",
                         f"n_floor_hits={n_floor_hits!r}, "
                         f"n_eligible={n_eligible!r}")
    lhs = BRANCH_DENOM * n_floor_hits
    per_location = lhs <= n_eligible
    return {"rule": "PER-LOCATION iff 20*n_floor_hits <= n_eligible",
            "n_floor_hits": n_floor_hits, "n_eligible": n_eligible,
            "lhs_20x_hits": lhs, "rhs_n_eligible": n_eligible,
            "selected": "PER-LOCATION" if per_location
                        else "GLOBAL PER-CHANNEL"}


def floor_hit_mask(raw_std: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """STRICT-< floor-hit semantics (Concept D4 clarification): a location
    is a floor-hit if raw_std(r,c,ch) < floor(ch) for EITHER channel.
    Equality is NOT a hit."""
    return (raw_std[:, :, 0] < floor[0]) | (raw_std[:, :, 1] < floor[1])


def per_location_scales(raw_std: np.ndarray, floor: np.ndarray
                        ) -> np.ndarray:
    """Candidate per-location scale: max(raw_std(r,c,ch), floor(ch))."""
    return np.maximum(raw_std, floor)


def c7_roundtrip(applied_mean: np.ndarray, applied_scale: np.ndarray
                 ) -> dict:
    """C7 implementation-validity round-trip (ERROR class, gates ALWAYS):
    the float64 affine op x -> (x - applied_mean)/applied_scale and its
    exact inverse, over the ACTUALLY SELECTED pair. Metric:
        max|x_rt - x| / max(1, max|x|)  <=  C7_RTOL
    evaluated on deterministic probes built from the PUBLISHED parameters
    only: x in {m, m + s, m - 2s}. Near-zero stabilisation via max(1, .)."""
    if applied_mean.shape != applied_scale.shape or applied_mean.size == 0:
        logger.error("[P4S2] C7 operand shapes invalid: %s vs %s",
                     applied_mean.shape, applied_scale.shape)
        raise StageError("C7_OPERANDS_INVALID",
                         f"applied_mean {applied_mean.shape} vs applied_scale "
                         f"{applied_scale.shape}")
    if not np.isfinite(applied_mean).all() \
            or not np.isfinite(applied_scale).all() \
            or not (applied_scale > 0).all():
        logger.error("[P4S2] C7 operands non-finite or non-positive")
        raise StageError("C7_OPERANDS_INVALID",
                         "the selected applied pair contains a non-finite "
                         "mean or a non-positive scale")
    worst = 0.0
    for probe in (applied_mean, applied_mean + applied_scale,
                  applied_mean - 2.0 * applied_scale):
        y = (probe - applied_mean) / applied_scale
        x_rt = y * applied_scale + applied_mean
        err = np.abs(x_rt - probe) / np.maximum(1.0, np.abs(probe))
        worst = max(worst, float(err.max()))
    ok = worst <= C7_RTOL
    if not ok:
        logger.error("[P4S2] C7 round-trip %g > %g", worst, C7_RTOL)
        raise StageError("C7_ROUNDTRIP_VIOLATED",
                         f"the selected affine pair round-trips at "
                         f"{worst:.3e}, above the registered {C7_RTOL:.0e}; "
                         f"the scaling implementation is invalid")
    return {"op": "x -> (x - applied_mean)/applied_scale and its exact "
                  "inverse, float64",
            "metric": "max|x_rt - x| / max(1, max|x|)",
            "tolerance": C7_RTOL,
            "probes": ["applied_mean", "applied_mean + applied_scale",
                       "applied_mean - 2*applied_scale"],
            "max_rel_err": worst, "ok": ok}


def finalize_statistics(acc: dict, eligible: list,
                        authoritative: bool) -> dict:
    """From the accumulated sufficient statistics to the full /2 record:
    raw_std (ddof=0), pooled globals, floor, STRICT-< floor hits,
    per_location_scale, pre-vote validity of BOTH candidate families, the
    integer branch decision and the branch-selected applied affine pair.
    Gates: finiteness and positivity ALWAYS; full eligible coverage only
    on the authoritative run (a smoke prefix legitimately leaves eligible
    locations unobserved)."""
    centre = centre_columns(GRID_HW)
    n_eligible_locations = GRID_HW * len(eligible)
    count, mean, M2 = acc["count"], acc["mean"], acc["M2"]
    if not (count[:, :, 0] == count[:, :, 1]).all():
        logger.error("[P4S2] channel count mismatch between re and im")
        raise StageError("CHANNEL_COUNT_MISMATCH",
                         "re/im observation counts differ at a location; "
                         "both channels are observed by the same mask, so "
                         "this is an accumulation defect")
    centre_observed = [(r, c) for c in sorted(centre)
                       for r in range(GRID_HW) if count[r, c, 0] > 0]
    if centre_observed:
        logger.error("[P4S2] centre locations observed: %s",
                     centre_observed[:8])
        raise StageError("CENTRE_COLUMN_OBSERVED_FREE",
                         f"centre locations {centre_observed[:8]} carry "
                         f"observations; contradicts the registered "
                         f"mask-family construction")
    idx_r, idx_c = np.meshgrid(np.arange(GRID_HW),
                               np.asarray(eligible, dtype=np.int64),
                               indexing="ij")
    n_obs = count[idx_r, idx_c, 0]
    observed = n_obs > 0
    n_locations_observed = int(observed.sum())
    full_coverage = n_locations_observed == n_eligible_locations
    if authoritative and not full_coverage:
        missing = [(int(r), int(c)) for r, c in
                   zip(idx_r[~observed], idx_c[~observed])]
        logger.error("[P4S2] %d eligible locations unobserved on the full "
                     "frame, e.g. %s", n_eligible_locations
                     - n_locations_observed, missing[:8])
        raise StageError("ELIGIBLE_LOCATION_UNOBSERVED",
                         f"{n_eligible_locations - n_locations_observed} "
                         f"inherited eligible locations have no observation "
                         f"over the full frame, e.g. {missing[:8]}")
    # raw_std (population, ddof=0) at observed eligible locations.
    raw_std = np.full((GRID_HW, len(eligible), 2), np.nan)
    loc_mean = np.full((GRID_HW, len(eligible), 2), np.nan)
    for ch in (0, 1):
        m = mean[idx_r, idx_c, ch][observed]
        m2 = M2[idx_r, idx_c, ch][observed]
        n = n_obs[observed].astype(np.float64)
        loc_mean[:, :, ch][observed] = m
        raw_std[:, :, ch][observed] = np.sqrt(m2 / n)
    channels = {}
    for ch, name in enumerate(CHANNELS):
        mean_g, sigma_g, N = pooled_channel_moments(count, mean, M2,
                                                    eligible, ch)
        channels[name] = {"mean_global": mean_g, "sigma_global": sigma_g,
                          "floor": FLOOR_FACTOR * sigma_g,
                          "pooled_observations": N}
    floor = np.array([channels["re"]["floor"], channels["im"]["floor"]])
    sigma_global = np.array([channels["re"]["sigma_global"],
                             channels["im"]["sigma_global"]])
    mean_global = np.array([channels["re"]["mean_global"],
                            channels["im"]["mean_global"]])
    if not np.isfinite(raw_std[observed]).all() \
            or not np.isfinite(loc_mean[observed]).all():
        logger.error("[P4S2] non-finite per-location statistic")
        raise StageError("NON_FINITE_STATISTIC",
                         "a raw_std or location mean is non-finite")
    # STRICT-< floor hit on raw_std; equality is NOT a hit.
    floor_hit = np.zeros((GRID_HW, len(eligible)), dtype=bool)
    floor_hit[observed] = floor_hit_mask(raw_std, floor)[observed]
    n_floor_hits = int(floor_hit.sum())
    # Candidate per-location scales: max(raw_std, floor(ch)).
    per_location_scale = np.where(
        observed[..., None], per_location_scales(raw_std, floor), np.nan)
    # PRE-VOTE VALIDITY: BOTH candidate families must be numerically valid
    # before the branch vote (ERROR class).
    problems = []
    if not (np.isfinite(sigma_global).all() and (sigma_global > 0).all()):
        problems.append(f"sigma_global={sigma_global.tolist()}")
    if not np.isfinite(mean_global).all():
        problems.append(f"mean_global={mean_global.tolist()}")
    pls_obs = per_location_scale[observed]
    if not (np.isfinite(pls_obs).all() and (pls_obs > 0).all()):
        problems.append("per_location_scale has a non-finite or "
                        "non-positive entry")
    if problems:
        logger.error("[P4S2] pre-vote validity failed: %s", problems)
        raise StageError("PRE_VOTE_VALIDITY_FAILURE",
                         f"a candidate scale family is numerically invalid "
                         f"BEFORE the branch vote: {problems}")
    decision = branch_decision(n_floor_hits,
                               n_eligible_locations if authoritative
                               else n_locations_observed)
    # Branch-selected applied affine pair (registered D4 semantics).
    if decision["selected"] == "PER-LOCATION":
        applied_mean = np.where(observed[..., None], loc_mean, np.nan)
        applied_scale = per_location_scale
    else:
        applied_mean = np.where(observed[..., None], mean_global, np.nan)
        applied_scale = np.where(observed[..., None], sigma_global, np.nan)
    return {"channels": channels, "floor": floor,
            "sigma_global": sigma_global, "mean_global": mean_global,
            "raw_std": raw_std, "loc_mean": loc_mean,
            "per_location_scale": per_location_scale,
            "floor_hit": floor_hit, "n_floor_hits": n_floor_hits,
            "observed": observed, "n_obs_count": n_obs,
            "n_locations_observed": n_locations_observed,
            "n_eligible_locations": n_eligible_locations,
            "full_coverage": full_coverage,
            "decision": decision,
            "applied_mean": applied_mean, "applied_scale": applied_scale}


# ---------------------------------------------------------------------------
# Parity against the /1 parent (AUTHORITATIVE RUN ONLY; never gated at
# smoke scale -- a smoke prefix legitimately covers only part of the
# parent table)
# ---------------------------------------------------------------------------

def compare_weights_vs_parent(own_w: dict, parent: dict) -> dict:
    """Exact sparse w_i(c) equality between this stage's own census and the
    /1 parent table. Any difference means the frame traversed now is not
    the frame the /1 census certified -> ERROR."""
    diffs = []
    all_cols = sorted(set(own_w) | set(parent["parent_w"]))
    for c in all_cols:
        own = own_w.get(c, {})
        par = parent["parent_w"].get(c, {})
        if own != par:
            diffs.append({"column": c,
                          "own_total": sum(own.values()),
                          "parent_total": sum(par.values()),
                          "own_files": len(own), "parent_files": len(par)})
    if diffs:
        logger.error("[P4S2] sparse w_i(c) parity failed at %d columns, "
                     "e.g. %s", len(diffs), diffs[:4])
        raise StageError("PARENT_WEIGHTS_MISMATCH",
                         f"the stage's own sparse w_i(c) differs from the "
                         f"/1 parent table at {len(diffs)} columns, e.g. "
                         f"{diffs[:4]}; the traversed frame is not the "
                         f"certified one")
    return {"evaluated": True, "identical": True,
            "columns_compared": len(all_cols)}


def transpose_invariant(acc: dict, parent: dict, eligible: list) -> dict:
    """count(r,c) == n_free_raw(c) for every eligible location, against the
    PARENT's column totals. Rows of a column share one observation count
    (1-D column mask broadcast), so every row is checked."""
    bad = []
    for c in eligible:
        expected = parent["parent_n_free_raw"][c]
        col_counts = acc["count"][:, c, 0]
        if not (col_counts == expected).all():
            rows = [int(r) for r in range(GRID_HW)
                    if col_counts[r] != expected]
            bad.append({"column": c, "expected": expected,
                        "rows_differing": rows[:8]})
    if bad:
        logger.error("[P4S2] transpose invariant failed: %s", bad[:4])
        raise StageError("TRANSPOSE_INVARIANT_VIOLATED",
                         f"count(r,c) != n_free_raw(c) at {len(bad)} "
                         f"columns, e.g. {bad[:4]}")
    return {"evaluated": True, "holds": True,
            "columns_checked": len(eligible),
            "rows_per_column": GRID_HW}


def own_count_weight_consistency(acc: dict, own_w: dict,
                                 eligible: list) -> dict:
    """INTERNAL invariant, gated ALWAYS (construction): the stage's own
    per-column weight sums equal its own Welford counts. Distinct from the
    authoritative-only parent parity -- this catches accumulation defects
    at any scale, including smoke."""
    bad = []
    for c in eligible:
        w_sum = sum(own_w.get(c, {}).values())
        col_counts = acc["count"][:, c, 0]
        if not (col_counts == w_sum).all():
            bad.append({"column": c, "weight_sum": w_sum,
                        "count_min": int(col_counts.min()),
                        "count_max": int(col_counts.max())})
    if bad:
        logger.error("[P4S2] own count/weight inconsistency: %s", bad[:4])
        raise StageError("OWN_COUNT_WEIGHT_MISMATCH",
                         f"the stage's own w_i(c) sums disagree with its "
                         f"own per-location counts at {len(bad)} columns, "
                         f"e.g. {bad[:4]}; accumulation defect")
    return {"evaluated": True, "holds": True,
            "note": "internal own-accumulation consistency; gated at every "
                    "scale including smoke"}


# ---------------------------------------------------------------------------
# Frame traversal with per-slice mask identity/seed consistency
# ---------------------------------------------------------------------------

def enforce_generator_pin(seed_prov: dict) -> None:
    """Same binding as the /1 stage: the frame's hash-binding is a GATE,
    not a record."""
    got = (seed_prov or {}).get("mask_seed_source_sha256")
    if not (seed_prov or {}).get("resolved") or got != GENERATOR_SOURCE_SHA256:
        logger.error("[P4S2] generator hash %s != registered pin %s",
                     got, GENERATOR_SOURCE_SHA256)
        raise StageError("GENERATOR_HASH_MISMATCH",
                         f"the executing mask generator hashes to {got}, "
                         f"but the registered frame is hash-bound to "
                         f"{GENERATOR_SOURCE_SHA256}")


def traverse_frame(data_root: str, smoke, eligible: list):
    """Traverse the REGISTERED MASK-STATISTICS FRAME in dataset index
    (manifest) order, reading k-space through the registered Construction-A
    preparation. PER-SLICE, WHILE ACCUMULATING (guardrail against
    slice<->mask binding drift):
      * seed derived two ways (public canonical_mask_seed AND the dataset's
        own _mask_seed) and both must equal the item's recorded mask_seed;
      * the regenerated mask must equal the mask the dataset APPLIED;
      * the acquired count must equal the generator's exact count.
    The modelled free coefficient is F x_norm at the free columns, with
    x_norm = x_true / file_attr_max (registered D2 rule) executed on the
    /2 float64 arithmetic path (EXEC §13 P4 /2 arithmetic path)."""
    eligible_set = frozenset(eligible)
    ds = fdm.FastMRISliceDataset(data_root, split="train", mode="train")
    ds.set_epoch(EPOCH_SET[0])
    n_items = len(ds.index) if smoke is None else smoke
    n_acquired_expected = expected_acquired_count(GRID_HW)
    acc = new_accumulator(GRID_HW)
    weights: dict[int, dict[str, int]] = {}
    files_seen = set()
    per_slice = {"rows": 0, "seed_agreement": True,
                 "mask_identity": True, "acquired_count": True}
    seen = set()
    for i in range(n_items):
        path, s = ds.index[i]
        rel = path.relative_to(ds.data_root).as_posix()
        key = (rel, int(s))
        if key in seen:
            logger.error("[P4S2] duplicate frame row %s", key)
            raise StageError("DUPLICATE_FRAME_ROW",
                             f"frame row {key} appears twice; the "
                             f"realisation count is derived, never chosen, "
                             f"so duplication is a construction defect")
        seen.add(key)
        item = ds[i]
        meta = item["meta"]
        seed_public = fdm.canonical_mask_seed(fdm.TRAIN_BASE_SEED, rel,
                                              int(s), epoch=EPOCH_SET[0])
        seed_dataset = ds._mask_seed(path, int(s))
        seed_item = int(meta["mask_seed"])
        if seed_public != seed_dataset or seed_public != seed_item:
            logger.error("[P4S2] seed disagreement at (%s, %d): public=%d "
                         "dataset=%d item=%d", rel, s, seed_public,
                         seed_dataset, seed_item)
            raise StageError("SEED_DERIVATION_MISMATCH",
                             f"canonical_mask_seed, dataset._mask_seed and "
                             f"the item's recorded mask_seed disagree at "
                             f"({rel}, {s}); the slice<->mask binding is "
                             f"broken")
        mask_np = np.asarray(item["mask"].numpy(), dtype=bool)
        mask_regen = fdm.make_cartesian_mask(GRID_HW, seed_public)
        if not np.array_equal(mask_np, mask_regen):
            logger.error("[P4S2] mask identity failed at (%s, %d)", rel, s)
            raise StageError("MASK_BINDING_MISMATCH",
                             f"the mask the dataset applied at ({rel}, {s}) "
                             f"differs from the freshly regenerated mask "
                             f"for the verified seed; slice<->mask binding "
                             f"drift detected WHILE accumulating")
        if int(mask_np.sum()) != n_acquired_expected:
            logger.error("[P4S2] acquired count %d != %d at (%s, %d)",
                         int(mask_np.sum()), n_acquired_expected, rel, s)
            raise StageError("ACQUIRED_COUNT_MISMATCH",
                             f"acquired count varies at ({rel}, {s})")
        free = free_columns_of(mask_np)
        outside = [c for c in free if c not in eligible_set]
        if outside:
            logger.error("[P4S2] free columns %s outside the inherited "
                         "eligible set at (%s, %d)", outside, rel, s)
            raise StageError("FREE_COLUMN_OUTSIDE_INHERITANCE",
                             f"columns {outside} are free at ({rel}, {s}) "
                             f"but are not in the inherited eligible set; "
                             f"the observation contradicts the /1-certified "
                             f"frame")
        # Registered preparation: Construction A already applied inside
        # __getitem__; D2 division executed on the /2 float64 path.
        amax = float(meta["file_attr_max"])
        xt = item["x_true"].to(torch.float64)
        x_norm = torch.complex(xt[0] / amax, xt[1] / amax)   # (96,96) c128
        u = fdm.fft2c(x_norm).numpy()                        # F x_norm
        vals = u[:, np.asarray(free, dtype=np.int64)]
        accumulate_observations(acc, free, vals)
        accumulate_weights(weights, rel, free)
        files_seen.add(rel)
        per_slice["rows"] += 1
    return {"acc": acc, "weights": weights,
            "files": sorted(files_seen), "per_slice": per_slice,
            "dataset": ds}


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

def _local_sha(repo_dir: str, relpath: str) -> str | None:
    path = os.path.join(repo_dir, relpath)
    return file_sha256(path) if os.path.isfile(path) else None


def check_population(trav: dict, authoritative: bool) -> dict:
    """Derived, never chosen. Gated on the full frame only (as /1)."""
    observed = {"n_slices": trav["per_slice"]["rows"],
                "n_files": len(trav["files"])}
    ok = (observed["n_slices"] == REGISTERED_TRAIN_SLICES and
          observed["n_files"] == REGISTERED_TRAIN_FILES)
    if authoritative and not ok:
        logger.error("[P4S2] frame population %s != registered (%d, %d)",
                     observed, REGISTERED_TRAIN_SLICES,
                     REGISTERED_TRAIN_FILES)
        raise StageError("FRAME_POPULATION_MISMATCH",
                         f"derived frame population {observed} does not "
                         f"equal the registered {REGISTERED_TRAIN_SLICES} "
                         f"slices / {REGISTERED_TRAIN_FILES} files")
    return {**observed,
            "registered": {"n_slices": REGISTERED_TRAIN_SLICES,
                           "n_files": REGISTERED_TRAIN_FILES},
            "gated": authoritative,
            "note": ("gated: full frame" if authoritative else
                     "RECORDED ONLY: smoke traverses a prefix of the "
                     "frame by design")}


def _location_records(stats: dict, eligible: list) -> list:
    """Row-major (row, column) records for every OBSERVED eligible
    location. Unobserved eligible locations are absent (smoke prefix by
    design); on the authoritative run coverage is gated full, so absence
    never appears there."""
    out = []
    obs = stats["observed"]
    for r in range(GRID_HW):
        for j, c in enumerate(eligible):
            if not obs[r, j]:
                continue
            out.append({
                "row": r, "column": c,
                "count": int(stats["n_obs_count"][r, j]),
                "mean_re": float(stats["loc_mean"][r, j, 0]),
                "mean_im": float(stats["loc_mean"][r, j, 1]),
                "raw_std_re": float(stats["raw_std"][r, j, 0]),
                "raw_std_im": float(stats["raw_std"][r, j, 1]),
                "per_location_scale_re":
                    float(stats["per_location_scale"][r, j, 0]),
                "per_location_scale_im":
                    float(stats["per_location_scale"][r, j, 1]),
                "floor_hit": bool(stats["floor_hit"][r, j]),
                "applied_mean_re": float(stats["applied_mean"][r, j, 0]),
                "applied_mean_im": float(stats["applied_mean"][r, j, 1]),
                "applied_scale_re": float(stats["applied_scale"][r, j, 0]),
                "applied_scale_im": float(stats["applied_scale"][r, j, 1])})
    return out


def _build_facts(parent, trav, stats, c7, population, consistency,
                 verdict, reason, repo_dir, script, argv, t0, smoke,
                 seed_prov, dataset_prov) -> dict:
    eligible = parent["eligible_columns"]
    thresholds = {
        "FLOOR_FACTOR": FLOOR_FACTOR,
        "floor_rule": "floor(ch) = 1e-2 * sigma_global(ch)  [D4 factor "
                      "UNCHANGED]",
        "floor_hit_rule": "STRICT <: raw_std(r,c,ch) < floor(ch); equality "
                          "is NOT a hit; a location is a floor-hit if "
                          "EITHER channel hits",
        "BRANCH_DENOM": BRANCH_DENOM,
        "branch_rule": "PER-LOCATION iff 20*n_floor_hits <= n_eligible "
                       "(0.05 = 1/20; integer form, no float division; "
                       "5% threshold UNCHANGED)",
        "C7_RTOL": C7_RTOL,
        "DDOF": DDOF,
        "GRID_WIDTH": GRID_HW,
        "EPOCH_SET": list(EPOCH_SET),
        "REGISTERED_TRAIN_FILES": REGISTERED_TRAIN_FILES,
        "REGISTERED_TRAIN_SLICES": REGISTERED_TRAIN_SLICES,
        "REGISTERED_ELIGIBLE_COLUMNS": REGISTERED_ELIGIBLE_COLUMNS,
        "n_eligible_locations": stats["n_eligible_locations"],
        "N_EFF_MIN_context": 900,
        "applied_affine_pair": {
            "PER-LOCATION": "applied_mean = mean(r,c,ch); applied_scale = "
                            "per_location_scale(r,c,ch)",
            "GLOBAL PER-CHANNEL": "applied_mean = mean_global(ch); "
                                  "applied_scale = sigma_global(ch) at "
                                  "every eligible location",
            "note": "the selected branch's pairs are published as applied; "
                    "the unselected branch's statistics remain recorded, "
                    "never published as applied"}}
    frame = {"population": "full training split",
             "train_slices": None, "epoch_set": list(EPOCH_SET),
             "mask_mode": "train",
             "generator": "seqref_mri/src/fastmri_data.py (SEQREF-I1 v0.3)",
             "generator_source_sha256":
                 _local_sha(repo_dir, "seqref_mri/src/fastmri_data.py"),
             "base_seed": fdm.TRAIN_BASE_SEED,
             "seed_tuple_serialization":
                 "train: '{base_seed}|{epoch}|{relpath}|{slice_index}' "
                 "(UTF-8 -> SHA-256 -> first 8 bytes big-endian); the "
                 "generator signature parameter order is NOT the "
                 "serialization order",
             "realisation_count": ("DERIVED = n_slices x |epochs|; never a "
                                   "chosen parameter"),
             "kspace_read": True,
             "arithmetic_path": {
                 "device": "cpu", "torch_threads": 1, "dtype": "float64",
                 "traversal_order": "dataset index (manifest) order",
                 "preparation": "registered Construction A via "
                                "FastMRISliceDataset.__getitem__; D2 "
                                "division x_true/file_attr_max executed in "
                                "float64; registered fft2c (centred "
                                "orthonormal) for F x_norm"},
             "mask_source": "per-slice, WHILE accumulating: two-way seed "
                            "derivation agreement AND applied-mask identity "
                            "against the regenerated mask, plus exact "
                            "acquired count"}
    inheritance = {
        "parent_schema": PARENT_SCHEMA,
        "parent_file_sha256": parent["file_sha256"],
        "parent_semantic_sha256": parent["semantic_sha256"],
        "parent_sidecar_present": parent["sidecar_present"],
        "parent_file_basename": os.path.basename(parent["path"]),
        "eligible_columns": eligible,
        "n_eligible_columns": len(eligible),
        "n_eligible_locations": stats["n_eligible_locations"],
        "rule": "the eligible set is INHERITED from the pinned /1 parent; "
                "it is never re-derived, and the stage's own sparse w_i(c) "
                "must equal the parent table exactly on an authoritative "
                "run"}
    branch = dict(stats["decision"])
    branch["smoke_scale"] = smoke is not None
    branch["gating_note"] = (
        "the verdict-relevant decision: full eligible coverage gated"
        if smoke is None else
        "SMOKE-SCALE evaluation over the observed prefix only; recorded "
        "and exercised, never the verdict")
    locations = _location_records(stats, eligible)
    summary = {"verdict": verdict, "verdict_reason": reason,
               "smoke": smoke is not None, "smoke_slices": smoke,
               "n_rows": trav["per_slice"]["rows"],
               "n_files_observed": len(trav["files"]),
               "n_locations_emitted": len(locations),
               "n_floor_hits": stats["n_floor_hits"],
               "branch_selected": stats["decision"]["selected"],
               "c7_max_rel_err": c7["max_rel_err"]}
    parents_rec = {"p4_s1": {
                       "schema": PARENT_SCHEMA,
                       "file_sha256": parent["file_sha256"],
                       "semantic_sha256": parent["semantic_sha256"],
                       "verdict": "PASS",
                       "sidecar_present": parent["sidecar_present"]},
                   "grandparents_via_p4_s1_record": parent["grandparents"]}
    code = hash_project_code(repo_dir, script)
    p4s2_local = {"p4s2_local": [
        {"relpath": "seqref_mri/src/fastmri_data.py",
         "sha256": _local_sha(repo_dir, "seqref_mri/src/fastmri_data.py")}]}
    semantic = {"schema": FACTS_SCHEMA, "stage": "P4",
                "thresholds": thresholds, "verdict": verdict,
                "frame": frame, "inheritance": inheritance,
                "population": population, "channels": stats["channels"],
                "branch": branch, "locations": locations,
                "consistency": consistency, "summary": summary,
                "parents": parents_rec,
                "code": code["project_local"] + p4s2_local["p4s2_local"]}
    facts = {
        "schema": FACTS_SCHEMA,
        "script": {"id": SCRIPT_ID, "version": SCRIPT_VERSION,
                   "lifetime": "KEEP"},
        "stage": "P4",
        "artefact_type": "stage_facts",
        "run_mode": ("smoke" if smoke is not None else "authoritative"),
        "authoritative": smoke is None,
        "stage_description": "statistics layer (schema /2): per-location "
                             "count/mean/M2 over all free observations at "
                             "inherited eligible locations, pooled "
                             "globals, STRICT-< floor hits, integer branch "
                             "decision, branch-selected applied affine "
                             "pair, C7 round-trip",
        "schema_scope": {
            "covers": ["per_location_scaling_statistics", "branch_vote",
                       "applied_affine_pair"],
            "parent_schema": PARENT_SCHEMA,
            "note": "seqref-p4-stats/2 is the statistics layer. The /1 "
                    "artefact (census/support/eligibility) is FROZEN and "
                    "never overwritten; /2 inherits its eligible set under "
                    "a dual-hash pin."},
        "thresholds": thresholds,
        "verdict": verdict,
        "verdict_reason": reason,
        "frame": frame,
        "inheritance": inheritance,
        "population": population,
        "channels": stats["channels"],
        "branch": branch,
        "locations_order": "row-major over (row, column); observed "
                           "eligible locations only (authoritative: all "
                           "8,448, coverage gated)",
        "locations": locations,
        "c7": c7,
        "consistency": consistency,
        "files": trav["files"],
        "summary": summary,
        "parents": parents_rec,
        "mask_seed_provenance": seed_prov or {"resolved": False},
        "dataset_provenance": dataset_prov or {},
        "code": {**code, **p4s2_local},
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
        description=f"{SCRIPT_ID} {SCRIPT_VERSION} -- P4 statistics layer "
                    "(schema seqref-p4-stats/2)")
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--p4-stats1", required=True,
                    help="authoritative /1 artefact (scaling_stats.json); "
                         "verified against BOTH registered pins")
    ap.add_argument("--out-dir", required=True,
                    help="/2 output directory; for a smoke run this must "
                         "be an EPHEMERAL directory, never the parent's")
    ap.add_argument("--smoke", type=int, default=None,
                    help="EPHEMERAL: first N frame rows, smoke_ prefix; "
                         "never authoritative")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    t0 = time.time()
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    # `is not None`, not truthiness (as /1): --smoke 0 is invalid input,
    # but its error record must still carry the SMOKE prefix.
    facts_prefix = SMOKE_FACTS_PREFIX if args.smoke is not None \
        else FACTS_PREFIX
    error_prefix = SMOKE_ERROR_PREFIX if args.smoke is not None \
        else ERROR_PREFIX
    script = os.path.abspath(__file__)
    parent = None
    try:
        if args.smoke is not None and args.smoke <= 0:
            raise StageError("BAD_SMOKE_SIZE",
                             f"--smoke must be a positive int, got "
                             f"{args.smoke!r}")
        run_mode = guard_run_mode(args.out_dir, args.smoke is not None)
        logger.info("%s %s run_mode=%s out_dir=%s", SCRIPT_ID,
                    SCRIPT_VERSION, run_mode, args.out_dir)
        # Registered arithmetic path: CPU, single torch thread, float64.
        torch.set_num_threads(1)
        parent = load_p4s1_parent(args.p4_stats1)
        eligible = parent["eligible_columns"]
        logger.info("%s /1 parent pinned: file=%s semantic=%s eligible=%d "
                    "columns (%d locations)", SCRIPT_ID,
                    parent["file_sha256"][:12], parent["semantic_sha256"][:12],
                    len(eligible), GRID_HW * len(eligible))
        seed_prov = bind_mask_seed_provenance(args.repo_dir)
        enforce_generator_pin(seed_prov)
        trav = traverse_frame(args.data_root, args.smoke, eligible)
        acc, own_w = trav["acc"], trav["weights"]
        consistency = {"per_slice": {
            **trav["per_slice"],
            "rule": "two-way seed agreement + applied-mask identity + "
                    "exact acquired count, checked WHILE accumulating"}}
        consistency["internal_invariants"] = {
            "channel_count_equal": "gated inside finalize_statistics",
            "centre_never_observed": "gated inside finalize_statistics",
            "own_count_weight": own_count_weight_consistency(
                acc, own_w, eligible)}
        stats = finalize_statistics(acc, eligible, args.smoke is None)
        population = check_population(trav, args.smoke is None)
        if args.smoke is None:
            consistency["parity_vs_p4_s1"] = {
                "w_i_sparse_exact": compare_weights_vs_parent(own_w, parent),
                "transpose_invariant": transpose_invariant(acc, parent,
                                                           eligible),
                "note": "AUTHORITATIVE-ONLY gates; never fire at smoke "
                        "scale"}
        else:
            consistency["parity_vs_p4_s1"] = {
                "evaluated": False,
                "reason": "smoke traverses a frame prefix by design; "
                          "sparse w_i(c) equality and the transpose "
                          "invariant against the parent gate on the "
                          "authoritative run only"}
        # C7 over the ACTUALLY SELECTED applied pair (gates at any scale:
        # implementation validity, not a data gate).
        sel = stats["observed"]
        c7 = c7_roundtrip(stats["applied_mean"][sel],
                          stats["applied_scale"][sel])
        dataset_prov = dataset_provenance(type(trav["dataset"]),
                                          trav["dataset"])
        reason = (f"the statistics layer traversed the registered frame "
                  f"({trav['per_slice']['rows']} rows, "
                  f"{len(trav['files'])} files) with per-slice mask/seed "
                  f"consistency, accumulated ddof=0 statistics at "
                  f"{stats['n_locations_observed']} inherited eligible "
                  f"locations, and decided "
                  f"{stats['decision']['selected']} by the integer rule "
                  f"(20*{stats['n_floor_hits']} <= "
                  f"{stats['decision']['rhs_n_eligible']}: "
                  f"{stats['decision']['lhs_20x_hits']} <= "
                  f"{stats['decision']['rhs_n_eligible']}); C7 round-trip "
                  f"{c7['max_rel_err']:.3e} <= 1e-12")
        facts = _build_facts(parent, trav, stats, c7, population,
                             consistency, "PASS", reason, args.repo_dir,
                             script, raw_argv, t0, args.smoke, seed_prov,
                             dataset_prov)
        path, sha = publish_stage(facts, args.out_dir, facts_prefix, "P4")
        logger.info("P4 /2 PASS rows=%d files=%d locations=%d hits=%d "
                    "branch=%s facts=%s file_sha256=%s semantic_sha256=%s",
                    trav["per_slice"]["rows"], len(trav["files"]),
                    stats["n_locations_observed"], stats["n_floor_hits"],
                    stats["decision"]["selected"], path, sha,
                    facts["semantic_sha256"])
        if args.smoke is not None:
            logger.warning("SMOKE run -- NOT authoritative; delete %s "
                           "after inspection", path)
        return EXIT_PASS

    # NO BLOCK HANDLER (LOCK 2, as /1): every gate tests a construction,
    # contract or counting invariant; EXIT_BLOCK is unreachable by design.
    except StageError as exc:
        logger.error("P4 /2 ERROR [%s] -- %s", exc.error_code, exc.reason)
        publish_error(exc, args.out_dir, error_prefix, "P4",
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
                    "raised_after_parent_verification": parent is not None},
            write_record=parent is not None)
        publish_error(wrapped, args.out_dir, error_prefix, "P4",
                      code={"script": script}, run={"argv": raw_argv})
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
