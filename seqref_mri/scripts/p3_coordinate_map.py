#!/usr/bin/env python3
# SEQREF-P3CM v0.4.2 -- P3 PER-REALISATION coordinate maps, decoder
#                        validity, identity audit (COMPLEX branch, A4)
# LIFETIME: KEEP
#
# Purpose
#   Build and audit the free-coordinate map the flow will parameterise, and
#   prove the exact-DC decoder reproduces the anchor and preserves the measured
#   data. IMPL verifies this artefact against its sidecar, then RE-DERIVES
#   each realisation's map from the recorded acquired_columns under the
#   published enumeration rule and verifies map_sha256. It must never assume
#   a global map exists, and it must never skip the hash check.
#
# Verdict semantics (A4)
#   PASS  every realisation's map audited, decoder validated; facts
#         published; exit 0.
#   ERROR the code or specification is wrong. Construction, contract and
#         identity failures are ERROR: a broken scatter routine is not a
#         scientific finding about fastMRI.
#   BLOCK DOES NOT EXIST in this stage. Under A4 the census has no
#         scientific BLOCK outcome -- per-slice mask realisation is the
#         registered regime, and every remaining gate tests a construction,
#         contract or identity. Exit code 1 is UNREACHABLE by design;
#         that is deliberate and recorded, not a gap.
#
# CHANGELOG
# - v0.4.2 (2026-08-07): the FIFTH stale string. The Purpose block still
#   said "IMPL consumes the map this stage publishes; it must never rebuild
#   it" -- the v0.3.x consumption rule, contradicting the /2 re-derivation
#   contract fixed in v0.4.1. Found by the companion-file review, missed by
#   the v0.4.1 pass: a reminder that string audits must cover the header
#   prose, not only the artefact fields. Documentation only; no semantics.
# - v0.4.1 (2026-08-07): DOCUMENTATION-SYNC PATCH, no semantic change. Four
#   stale strings survived the v0.4 rewrite and contradicted it: (1)
#   verify_before_use still told IMPL to load coordinate_map.free_coordinates
#   and warned against rebuilding -- under A4 the consumer MUST re-derive
#   each map from the recorded acquired_columns and verify map_sha256, and
#   the string now says so with the reason; (2) the Verdict-semantics header
#   still described a scientific BLOCK on the falsified global-map premise;
#   (3) the mask_census docstring referenced census["outcome"] == "BLOCK",
#   a value no code path can produce; (4) the argparse description still
#   advertised v0.3.1. Also: the registered basis-probe scope note (ONE
#   realisation, permutation-agnostic, link 2 the per-realisation
#   permutation validator), removal of the duplicated max_MFdx_p3 row key,
#   and cleanup of a stale "BLOCK path" comment in _build_facts. Script SHA
#   changes; gates, thresholds, artefact schema and census semantics do not.
# - v0.4 (2026-08-03): REWRITTEN TO THE A4 PER-REALISATION SPECIFICATION. The
#   v0.3.x stage implemented the premise A4 falsified: it returned BLOCK when
#   column sets varied at a fixed acquired count, which is now the EXPECTED
#   REGIME (EXEC §8 P3), so the deployed stage would have blocked on every run
#   of the design it is specified to implement. The failure shape is recorded
#   plainly: internal-consistency review checked the amendment against itself
#   and never checked the IMPLEMENTATION against the amended specification --
#   the same shape as the concept's own §3.3-versus-§3.4 contradiction that
#   opened this work.
#     * census: NO scientific BLOCK branch remains. Varying column sets at a
#       fixed acquired count are RECORDED, not gated. Acquired-count variation
#       becomes ERROR: the count is fixed by construction (mask_counts raises),
#       so a violation is a broken generator contract, not a data verdict.
#     * one deterministic map PER MASK REALISATION, each audited by both links,
#       C1, C2 and C6; flow_dim_real required INVARIANT across the population.
#     * decoder validity uses THAT slice's map.
#     * artefact: canonical rule + PER-REALISATION BINDINGS (mask hash, map
#       hash, audit result), not 256 coordinate lists. Each binding carries
#       enough identity to prove sample <-> mask <-> map correspondence, and a
#       consumer RE-DERIVES the map and hash from the recorded mask.
#     * schema bumped to seqref-p3-facts/2. Readers MUST reject /1 rather than
#       interpret a global-map artefact as per-realisation facts.
#     * P3 now returns PASS or ERROR only; exit code 1 (BLOCK) is UNREACHABLE
#       and that is deliberate, recorded, and NOT a gap.
#   The v0.3.1 smoke artefact remains historical evidence bound to the OLD
#   implementation. It is neither rewritten nor reinterpreted.
# - v0.3.1 (2026-08-02): mask_census now RETURNS a completed census and the
#   caller raises the StageBlock. Raising from inside meant `census = ...` never
#   executed, so the BLOCK handler published an EMPTY census and every piece of
#   evidence about what varied -- distinct column sets, per-slice seeds -- was
#   built and then discarded. Also: the BLOCK facts path crashed when rows had been
#   COLLECTED but not validated: it indexed c3a_* directly and left the
#   private tensor handles in place, so the first real BLOCK exited 2 with no
#   artefact. Rows are now stripped and every gated getter tolerates an
#   unvalidated row; the mask-variation BLOCK carries per-slice seeds and
#   distinct column sets so the finding is diagnosable from the artefact.
# - v0.3 (2026-07-30): rewritten against the ACTUAL frozen API after repository
#   inspection. v0.2's every frozen call site was contract-incompatible and
#   would have died before reading data. Now wired exactly as P1/P2 are:
#   guard_run_mode(out_dir, smoke) -> verify_parents(repo, p0, p0s, p0s_script)
#   -> collect -> gate -> _build_facts -> publish_stage, with the identical
#   StageBlock/StageError boundary.
# - _prepare is called as _prepare(batch, "cpu", test0=False): the device is a
#   POSITIONAL second argument. Every earlier draft guessed a two-argument form.
# - Slice identities come from batch meta, as P2 obtains them; P0S carries
#   canonical_sorted_indices (plain ints), not identity records.
# - M4 mask-seed provenance recorded from meta["mask_seed"] per slice.
# - The removed v0.3 identity-residual gate is documented as a PRE-EXECUTION
#   REVIEW CORRECTION, not a result: it reduces algebraically to C3a.
#
# CONVENTION: logger.error + raise on every failure path. No fallback, no mock,
#   no placeholder, no silent pass.

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

from preflight_io import canonical_hash, file_sha256  # noqa: E402
from preflight_parents import (EXIT_ERROR, EXIT_PASS,  # noqa: E402
                               REQUIRED_PREPARE_KEYS, StageError,
                               attach_semantic_hash, environment_record,
                               guard_run_mode, hash_project_code, publish_error,
                               publish_stage, require_finite, verify_parents)
from preflight_parents_p3 import (bind_mask_seed_provenance,  # noqa: E402
                                  dataset_provenance, hash_p3_local_code,
                                  p2_field, verify_p1_p2)
import residual_decoder as dec  # noqa: E402
from seqref_mri.src.fastmri_data import FastMRISliceDataset  # noqa: E402
from seqref_mri.scripts.train_base import _collate, _prepare  # noqa: E402

SCRIPT_ID = "SEQREF-P3CM"
SCRIPT_VERSION = "v0.4.2"
FACTS_SCHEMA = "seqref-p3-facts/2"
LEGACY_FACTS_SCHEMA = "seqref-p3-facts/1"   # v0.3.x global-map format
FACTS_PREFIX = "coordinate_map"
ERROR_PREFIX = "coordinate_map_error"
SMOKE_FACTS_PREFIX = "smoke_coordinate_map"
SMOKE_ERROR_PREFIX = "smoke_coordinate_map_error"

# EXEC v0.4 §13 -- registered before execution. Shared values are INCIDENTAL:
# separate keys, separate facts fields, separate report rows, never
# interchangeable.
P3_DECODE_TOL = 1e-5            # ERROR-class: C3a zero-state decode vs anchor
P3_FIXITY_TOL = 1e-5            # ERROR-class: normalised measured fixity
P3_ROUNDTRIP_TOL = 1e-5         # ERROR-class: free-coordinate round trip
P3_BASIS_OFFTARGET_TOL = 1e-5   # ERROR-class: basis-probe off-target leakage
P3_PROBE_SEED = 0               # per-slice seed derived by canonical hash
P3_PATH_DIFF_REL_FLOOR = 1e-12  # NON-VERDICT floor, mirrors P2's rule exactly

# C3a is EXPECTED NON-ZERO. The primary decoder assembles in NORMALISED k-space
# while cond_in came from raw-assembly-then-image-division: identical
# mathematics, different fp32 operation order. Roughly 1e-7, two decades inside
# P3_DECODE_TOL. Registered so a passing 1e-7 is not misread as a near-miss.
P3_DECODE_EXPECTED_REL = 1e-7

# EXEC v0.4 §12 -- the CLOSED parent artefacts P3 was specified against. These
# are enforced, not merely recorded: sidecar consistency proves byte integrity,
# schema proves shape, but only these prove this is the artefact whose verdict
# EXEC closed. An unpinned load would accept any PASSing P1/P2.
P1_FACTS_SHA256 = ("1d0f760043c7e46ce5da338d81eb053e2d9"
                   "e0135a25063512ab9395bb18aa3a2")
P2_FACTS_SHA256 = ("8c22a025853187816e63b0121da39e2f74c"
                   "dd60126368b8559690f02877aab31")
P1_SEMANTIC_SHA256 = ("3823e4489cb3eac6177b23f17db3aa5437a"
                      "c197ef5de92f3597e57f3a92e45d7")
P2_SEMANTIC_SHA256 = ("77da087e853dbe5eed1547c97fc328ee406"
                      "b0be6ac15d976aaf47867cb219000")

logger = logging.getLogger(SCRIPT_ID)


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without the dataset)
# ---------------------------------------------------------------------------

def rel_diff(a: float, b: float) -> float:
    """Symmetric, floored -- P2's registered rule, reused verbatim."""
    return abs(a - b) / max(abs(a), abs(b), P3_PATH_DIFF_REL_FLOOR)


def margin_of(threshold: float, observed):
    """Oriented so > 1 is headroom. Distinguishes 'no applicable slice' from
    'zero observed', which are different facts."""
    if observed is None:
        return None, "not_applicable"
    if not np.isfinite(observed):
        return None, "unbounded"
    if observed == 0.0:
        return None, "unbounded"
    return threshold / observed, "finite"


def enumerate_free_independent(columns, height, width) -> list[list[int]]:
    """INDEPENDENT enumeration: pure-Python full-grid scan with a membership
    test. Deliberately a different implementation from the vectorised argwhere
    in residual_decoder, so the two derivations cannot share a defect."""
    acquired = set(int(c) for c in columns)
    return [[r, c] for r in range(height) for c in range(width)
            if c not in acquired]


def link1_ordered_list(cmap, independent) -> dict:
    """GATE: the published ordered list equals the independent enumeration."""
    published = cmap.ordered_coordinates()
    equal = published == independent
    first = None
    if not equal:
        for i, (a, b) in enumerate(zip(published, independent)):
            if a != b:
                first = {"index": i, "published": a, "independent": b}
                break
        first = first or {"index": min(len(published), len(independent)),
                          "published_len": len(published),
                          "independent_len": len(independent)}
        logger.error("link 1 FAILED: %s", first)
    return {"published_vs_independent_equal": equal,
            "first_coordinate_mismatch": first, "n_published": len(published),
            "n_independent": len(independent)}


def link2_unique_oracle(cmap, independent) -> dict:
    """GATE: production scatter equals an independently constructed grid.

    u[k] = (k+1) + i*(n_free-k): every component is an integer below 2^24 and
    therefore EXACTLY representable in float32, so the comparison is BITWISE.
    Valid only at grid-assembly stage, before any transform.
    """
    n = cmap.n_free_complex
    k = np.arange(n, dtype=np.float64)
    u = torch.from_numpy(((k + 1.0) + 1j * (n - k)).astype(np.complex64))
    produced = dec.scatter_unmeasured(u, cmap).numpy()
    expected = np.zeros((cmap.height, cmap.width), dtype=np.complex64)
    for i, (r, c) in enumerate(independent):
        expected[r, c] = np.complex64(complex(i + 1, n - i))
    exact = bool(np.array_equal(produced.real, expected.real)
                 and np.array_equal(produced.imag, expected.imag))
    first = None
    if not exact:
        bad = np.argwhere(produced != expected)
        if bad.size:
            r, c = int(bad[0][0]), int(bad[0][1])
            first = {"row": r, "col": c,
                     "produced": [float(produced[r, c].real),
                                  float(produced[r, c].imag)],
                     "expected": [float(expected[r, c].real),
                                  float(expected[r, c].imag)]}
        logger.error("link 2 FAILED at %s", first)
    off = produced.copy()
    off[cmap.free_rows, cmap.free_cols] = 0
    off_max = float(np.max(np.abs(off))) if off.size else 0.0
    return {"unique_oracle_bitwise_equal": exact,
            "unique_oracle_max_abs_error":
                0.0 if exact else float(np.max(np.abs(produced - expected))),
            "unique_oracle_first_mismatch": first,
            "off_support_max_abs": off_max,
            "off_support_exactly_zero": off_max == 0.0,
            "pattern": "u[k]=(k+1)+i*(n_free-k); integers < 2^24, exact in fp32"}


def c1_c2_audit(cmap) -> dict:
    """C1 completeness and C2 no-double-assignment over the whole grid."""
    seen = np.zeros((cmap.height, cmap.width), dtype=np.int32)
    acquired = np.zeros(cmap.width, dtype=bool)
    if cmap.mask_columns:
        acquired[np.asarray(cmap.mask_columns, dtype=np.int64)] = True
    seen[:, acquired] += 1
    seen[cmap.free_rows, cmap.free_cols] += 1
    for cls in (cmap.conjugate_filled, cmap.determined_from_partner,
                cmap.self_conjugate_real):
        for r, c in cls:
            seen[r, c] += 1
    unassigned = int(np.count_nonzero(seen == 0))
    doubled = int(np.count_nonzero(seen > 1))
    if unassigned or doubled:
        logger.error("C1/C2 FAILED: unassigned=%d doubled=%d", unassigned,
                     doubled)
    return {"c1_complete": unassigned == 0, "c2_no_double_assignment":
            doubled == 0, "n_unassigned": unassigned,
            "n_double_assigned": doubled, "class_counts": cmap.class_counts(),
            "completion_classes_measured_absent": {
                "conjugate_filled_from_free": len(cmap.conjugate_filled) == 0,
                "determined_from_acquired_partner":
                    len(cmap.determined_from_partner) == 0,
                "self_conjugate_real": len(cmap.self_conjugate_real) == 0}}


def c6_dimensions(cmap, n_acquired_columns, height, width) -> dict:
    """Three n_free counts by three genuinely different routes, then compared.
    An earlier draft's 'mask count' was the formula respelled; the full-grid
    count replaces it."""
    n_formula = height * (width - n_acquired_columns)
    acquired = np.zeros(width, dtype=bool)
    if cmap.mask_columns:
        acquired[np.asarray(cmap.mask_columns, dtype=np.int64)] = True
    n_full_grid = int(np.count_nonzero(
        ~np.broadcast_to(acquired[None, :], (height, width))))
    n_enumerated = len(cmap.ordered_coordinates())
    agree = n_formula == n_full_grid == n_enumerated
    if not agree:
        logger.error("C6 FAILED: formula=%d full_grid=%d enumerated=%d",
                     n_formula, n_full_grid, n_enumerated)
    flow_dim = 2 * n_enumerated
    return {"n_acquired": cmap.n_acquired, "n_free_formula": n_formula,
            "n_free_full_grid_count": n_full_grid,
            "n_free_enumerated": n_enumerated, "n_free_counts_agree": agree,
            "flow_dim_real": flow_dim, "bytes_per_state_fp32": flow_dim * 4,
            "bytes_per_state_fp64": flow_dim * 8,
            "bytes_per_batch_fp32_b8": flow_dim * 4 * 8,
            "feasibility_note": "state size only; architecture-dependent "
                                "parameters and activations are NOT estimated "
                                "here and IMPL must do that separately"}


def basis_probes(cmap) -> dict:
    """Through-FFT location behaviour and off-target leakage -- a different
    question from the scatter oracle, which tests the permutation itself.

    REGISTERED SCOPE (A4): this audit runs on ONE realisation, chosen
    deterministically (first insertion order). The probe measures an identity
    of the FFT PAIR, fft2c(ifft2c(.)) == (.), which holds regardless of which
    coordinates are free, so it is permutation-agnostic. Per-realisation
    permutation validity is link 2's unique-valued oracle, which exercises
    every free coordinate of every realisation bitwise; extending these
    probes to all realisations would buy redundancy, not coverage."""
    n = cmap.n_free_complex
    positions = sorted({0, n // 2, n - 1})
    records, worst_off, worst_tgt = [], 0.0, 0.0
    for pos in positions:
        for component in ("real", "imaginary"):
            u = torch.zeros(n, dtype=torch.complex64)
            u[pos] = (1 + 0j) if component == "real" else (0 + 1j)
            k_back = dec.fft2c(dec.ifft2c(dec.scatter_unmeasured(u, cmap)))
            r, c = int(cmap.free_rows[pos]), int(cmap.free_cols[pos])
            tgt = float(torch.abs(k_back[r, c] - u[pos]).item())
            off = k_back.clone()
            off[r, c] = 0
            off_max = float(torch.max(torch.abs(off)).item())
            worst_off, worst_tgt = max(worst_off, off_max), max(worst_tgt, tgt)
            peak = int(torch.argmax(torch.abs(k_back)).item())
            records.append({
                "basis_probe_coordinate_index": pos,
                "basis_probe_component": component,
                "expected_fourier_location": [r, c],
                "observed_peak_location": [peak // cmap.width,
                                           peak % cmap.width],
                "target_value_error": tgt,
                "max_off_target_magnitude": off_max,
                "basis_probe_pass": bool(off_max <= P3_BASIS_OFFTARGET_TOL
                                         and tgt <= P3_BASIS_OFFTARGET_TOL)})
    ok = all(r["basis_probe_pass"] for r in records)
    if not ok:
        logger.error("basis probes FAILED: worst off-target %.3e target %.3e",
                     worst_off, worst_tgt)
    return {"probes": records, "all_pass": ok, "worst_off_target": worst_off,
            "worst_target_error": worst_tgt,
            "threshold": P3_BASIS_OFFTARGET_TOL}


def mask_census(live: list[dict], p2_by_index: dict, authoritative: bool,
                expected_size: int, width: int) -> dict:
    """R1. Returns a COMPLETED census. Under A4 there is NO BLOCK outcome.

    Acquired-count variation raises ERROR from inside this function: the count
    is fixed by construction, so a violation is a broken generator contract.
    Varying column sets at a fixed count are RECORDED and are NOT gated.

    An earlier revision raised StageBlock from inside this function, so the
    assignment `census = mask_census(...)` never happened and the BLOCK handler
    published an EMPTY census -- every piece of evidence about what varied was
    built here and then discarded. The census is now a finished result before
    control flow changes.

    There is NO DATA-premise outcome: under A4 the census cannot produce a
    BLOCK, and census["outcome"] is always "PASS" on return. Drift between
    the live and persisted mask is raised here as StageError: that is a code,
    data or provenance defect, not a verdict, and it publishes no facts.
    """
    persisted, mismatches = {}, []
    for row in live:
        idx = row["dataset_index"]
        rec = p2_by_index.get(idx)
        if rec is None:
            raise StageError("P2_SLICE_RECORD_MISSING",
                             f"P2 has no per-slice record for dataset_index "
                             f"{idx}; the frozen subsets do not agree")
        cols = tuple(sorted(int(c) for c in
                            p2_field(rec, "selected_columns", f"slice {idx}")))
        persisted[idx] = cols
        if cols != row["selected_columns"]:
            mismatches.append({"dataset_index": idx,
                               "live": list(row["selected_columns"]),
                               "persisted": list(cols)})
        if int(p2_field(rec, "mask_seed", f"slice {idx}")) != row["mask_seed"]:
            mismatches.append({"dataset_index": idx, "field": "mask_seed",
                               "live": row["mask_seed"],
                               "persisted": p2_field(rec, "mask_seed",
                                                     f"slice {idx}")})

    live_sets = {r["selected_columns"] for r in live}
    persisted_sets = set(persisted.values())
    all_sets = live_sets | persisted_sets
    all_counts = {len(s) for s in all_sets}

    census = {
        "n_slices_compared": len(live),
        "full_frozen_population_compared": len(live) == expected_size,
        "all_compared_live_persisted_equal": len(mismatches) == 0,
        "n_live_persisted_mismatches": len(mismatches),
        "live_persisted_mismatches": mismatches[:8],
        "n_unique_persisted_mask_sets": len(persisted_sets),
        "n_unique_live_mask_sets": len(live_sets),
        "n_unique_acquired_counts": len(all_counts),
        "mask_width": width,
        "persisted_columns_field": "mask_selected_columns",
    }

    if mismatches:
        logger.error("live mask realisation differs from the persisted P2 mask "
                     "on %d slice(s)", len(mismatches))
        raise StageError("MASK_LIVE_PERSISTED_MISMATCH",
                         f"the live eval-mode mask does not reproduce the "
                         f"persisted P2 mask on {len(mismatches)} slice(s); "
                         f"this is code, data or provenance drift, not a data "
                         f"verdict", detail=census)
    if authoritative and not census["full_frozen_population_compared"]:
        raise StageError("SUBSET_SIZE_MISMATCH",
                         f"authoritative run compared {len(live)} slices, "
                         f"expected {expected_size}")
    distinct = sorted(all_sets)
    census["distinct_column_sets_sample"] = [list(s) for s in distinct[:4]]
    census["per_slice_mask_seeds"] = [
        {"dataset_index": r["dataset_index"], "file": r["file"],
         "slice_index": r["slice_index"], "mask_seed": r["mask_seed"],
         "n_columns": len(r["selected_columns"])} for r in live[:8]]
    census["n_unique_mask_seeds"] = len({r["mask_seed"] for r in live})
    census["n_distinct_sets"] = len(all_sets)

    # ---- ACQUIRED-COUNT INVARIANCE: ERROR, not BLOCK (A4).
    # mask_counts(96) fixes n_total = 24 and make_cartesian_mask RAISES if the
    # realised count differs, so invariance is guaranteed BY CONSTRUCTION. A
    # violation therefore means the generator contract is broken -- a code or
    # configuration defect, not a scientific finding about fastMRI.
    if len(all_counts) > 1:
        logger.error("acquired column count varies across the frozen subset: "
                     "%s", sorted(all_counts))
        raise StageError(
            "MASK_ACQUIRED_COUNT_VARIES",
            f"acquired column count varies across the frozen subset "
            f"({sorted(all_counts)}). The count is fixed by construction, so "
            f"this is a broken generator contract, not a data verdict.",
            detail=census)

    n_acq = next(iter(all_counts))
    census.update({
        "outcome": "PASS",
        "n_acquired_columns": n_acq,
        "acquired_count_invariant": True,
        "acquired_count_basis": "STRUCTURAL -- mask_counts(96) fixes n_total "
                                "and make_cartesian_mask raises on mismatch; "
                                "this census CORROBORATES the invariant rather "
                                "than establishing it",
        # A4: varying column sets at a fixed acquired count are the EXPECTED
        # REGIME. Recorded as a census observation, NOT gated. The v0.3.x
        # implementation returned BLOCK here, which would have blocked every
        # run of the design this stage is now specified to implement.
        "column_sets_vary": len(all_sets) > 1,
        "column_set_variation_is_expected": True,
        "column_set_variation_note":
            "per-slice mask realisation is the registered regime (EXEC §8 P3, "
            "concept §3.3 A4). One map is built and audited PER REALISATION; "
            "no global map exists and none is attempted.",
        "global_map_applicable": False,
        "global_map_note": "A4 retired the single global map. This field is "
                           "retained and set False so a reader of a /2 "
                           "artefact cannot mistake its absence for an "
                           "unevaluated check.",
        "no_block_branch": True,
        "no_block_branch_note":
            "the census has NO scientific BLOCK outcome under A4. An empty "
            "BLOCK set is the CORRECT result, not a gap in the taxonomy.",
    })
    logger.info("R1 census PASS: %d slices · acquired count %d (invariant) · "
                "%d distinct column sets · %d distinct seeds — set variation "
                "is the EXPECTED regime, not a block",
                len(live), n_acq, len(all_sets), census["n_unique_mask_seeds"])
    return census


def _p3_local_sha(repo_dir: str, relpath: str) -> str | None:
    """SHA of a P3-local source file, or None if absent (recorded as absent,
    never defaulted). Used to bind the enumeration RULE to the code that
    implements it: a rule named in prose binds nothing."""
    path = os.path.join(repo_dir, relpath)
    return file_sha256(path) if os.path.isfile(path) else None


def build_realisation_maps(rows: list[dict], height: int,
                           width: int) -> tuple[dict, list[dict]]:
    """ONE deterministic map PER MASK REALISATION (A4).

    Realisations are keyed by their acquired column set. Two slices with the
    SAME set legitimately share a map -- that is a property of the data, and
    neither sharing nor distinctness is assumed. Every slice receives a
    binding carrying enough identity to prove sample <-> mask <-> map
    correspondence, and the recorded mask is sufficient for a consumer to
    RE-DERIVE the map and its hash under the published enumeration rule.
    """
    maps: dict[tuple, dict] = {}
    bindings: list[dict] = []
    for r in rows:
        key = r["selected_columns"]
        if key not in maps:
            cmap = dec.build_coordinate_map(list(key), height, width)
            maps[key] = {
                "cmap": cmap,
                "mask_sha256": canonical_hash({"width": width,
                                               "selected_columns": list(key)}),
                "map_sha256": cmap.payload()["map_payload_sha256"],
                "n_free_complex": cmap.n_free_complex,
                "flow_dim_real": cmap.flow_dim_real,
                "n_acquired_columns": len(key),
                "n_slices": 0,
            }
        entry = maps[key]
        entry["n_slices"] += 1
        bindings.append({
            "dataset_index": r["dataset_index"], "file": r["file"],
            "slice_index": r["slice_index"], "split": r.get("split"),
            "mask_seed": r["mask_seed"],
            "acquired_columns": list(key),
            "mask_sha256": entry["mask_sha256"],
            "map_sha256": entry["map_sha256"],
            "n_free_complex": entry["n_free_complex"],
            "flow_dim_real": entry["flow_dim_real"],
        })
    logger.info("built %d distinct realisation map(s) over %d slices",
                len(maps), len(rows))
    return maps, bindings


def audit_realisations(maps: dict, height: int,
                       width: int) -> tuple[dict, dict, dict, dict]:
    """Audit EVERY realisation: both links, C1, C2, C6. Worst case reported.

    Dimensional invariance across realisations is a RETAINED GATE: the flow's
    input dimension is fixed at model-construction time, so a realisation with
    a different flow_dim_real would break the model contract.
    """
    l1_fail = l2_fail = c12_fail = c6_fail = None
    dims_seen: set[int] = set()
    free_seen: set[int] = set()
    n = 0
    first_dims = None
    for key, entry in maps.items():
        cmap = entry["cmap"]
        independent = enumerate_free_independent(list(key), height, width)
        a = link1_ordered_list(cmap, independent)
        b = link2_unique_oracle(cmap, independent)
        c = c1_c2_audit(cmap)
        d = c6_dimensions(cmap, len(key), height, width)
        if not a["published_vs_independent_equal"] and l1_fail is None:
            l1_fail = {"columns": list(key), **a}
        if not (b["unique_oracle_bitwise_equal"]
                and b["off_support_exactly_zero"]) and l2_fail is None:
            l2_fail = {"columns": list(key), **b}
        if not (c["c1_complete"] and c["c2_no_double_assignment"]) \
                and c12_fail is None:
            c12_fail = {"columns": list(key), **c}
        if not d["n_free_counts_agree"] and c6_fail is None:
            c6_fail = {"columns": list(key), **d}
        dims_seen.add(d["flow_dim_real"])
        free_seen.add(d["n_free_enumerated"])
        first_dims = first_dims or d
        n += 1

    link1 = {"n_realisations_audited": n,
             "all_published_vs_independent_equal": l1_fail is None,
             "first_failing_realisation": l1_fail}
    link2 = {"n_realisations_audited": n,
             "all_unique_oracle_bitwise_equal": l2_fail is None,
             "first_failing_realisation": l2_fail}
    audit = {"n_realisations_audited": n,
             "all_c1_complete_and_c2_disjoint": c12_fail is None,
             "first_failing_realisation": c12_fail}
    dims = {**(first_dims or {}),
            "n_realisations_audited": n,
            "distinct_flow_dim_real": sorted(dims_seen),
            "distinct_n_free_enumerated": sorted(free_seen),
            "flow_dim_invariant": len(dims_seen) <= 1,
            "all_n_free_counts_agree": c6_fail is None,
            "first_failing_realisation": c6_fail,
            "invariance_note":
                "flow_dim_real must be IDENTICAL across realisations: the "
                "flow's input dimension is fixed at model-construction time. "
                "Which coordinates are free varies per realisation; HOW MANY "
                "must not."}
    if not dims["flow_dim_invariant"]:
        logger.error("flow_dim_real varies across realisations: %s",
                     sorted(dims_seen))
    logger.info("audited %d realisation(s): links %s/%s · C1C2 %s · C6 %s · "
                "flow_dim %s", n,
                link1["all_published_vs_independent_equal"],
                link2["all_unique_oracle_bitwise_equal"],
                audit["all_c1_complete_and_c2_disjoint"],
                dims["all_n_free_counts_agree"], sorted(dims_seen))
    return link1, link2, audit, dims


def constraint_audit(maps: dict, energy_sum: np.ndarray,
                     energy_cnt: np.ndarray, conj: np.ndarray,
                     pair_indices: dict, height: int, width: int) -> dict:
    """Structural checks ERROR. Empirical patterns are NON-VERDICT.

    Under the COMPLEX branch with a column mask no exact algebraic rule
    relating unmeasured coefficients is derivable from the operator: there is
    no Hermitian symmetry, so self-conjugate Fourier locations remain
    two-real-dimensional and no unmeasured coefficient is determined by an
    acquired partner. The derivation is recorded; the structural checks
    confirm no realisation's map contradicts it.

    A4: the structural checks run PER REALISATION. Empirical coordinate energy
    is accumulated by PHYSICAL GRID LOCATION, because packed index k denotes a
    different Fourier location on every realisation.
    """
    structural = {"realisations_checked": 0, "free_overlaps_acquired_support": 0,
                  "duplicate_map_entries": 0, "omitted_unmeasured_locations": 0,
                  "deterministic_completion_relations_counted_free": 0,
                  "self_conjugate_restricted_coordinates": 0}
    first_fail = None
    for key, entry in maps.items():
        cmap = entry["cmap"]
        acquired = set(int(c) for c in cmap.mask_columns)
        free = [tuple(rc) for rc in cmap.ordered_coordinates()]
        per = {
            "free_overlaps_acquired_support":
                sum(1 for rc in free if rc[1] in acquired),
            "duplicate_map_entries": len(free) - len(set(free)),
            "omitted_unmeasured_locations":
                (height * (width - len(acquired))) - len(free),
            "deterministic_completion_relations_counted_free":
                len(cmap.conjugate_filled) + len(cmap.determined_from_partner),
            "self_conjugate_restricted_coordinates":
                len(cmap.self_conjugate_real),
        }
        structural["realisations_checked"] += 1
        for k, v in per.items():
            structural[k] += v
        if any(v != 0 for v in per.values()) and first_fail is None:
            first_fail = {"columns": list(key), **per}
    ok = first_fail is None
    if not ok:
        logger.error("structural exact-constraint audit FAILED on a "
                     "realisation: %s", first_fail)

    empirical = {"all_empirical_fields_non_verdict": True,
                 "n_realisations": len(maps),
                 "keying": "PHYSICAL GRID LOCATION (r, c), never packed index",
                 "keying_reason":
                     "packed index k denotes a different Fourier location on "
                     "every realisation, so pooling by k would average unlike "
                     "quantities -- the defect A5 corrects in D4",
                 "n_unmeasured_pairs_tested":
                     int(sum(pi["n_pairs"] for pi in pair_indices.values())),
                 "duplicate_trace_candidates": None,
                 "duplicate_trace_note":
                     "NOT computed under A4. A duplicate-trace test compares "
                     "coordinate columns by packed index, which is not a "
                     "stable identity across realisations. Recorded as absent "
                     "with its reason rather than reported as zero."}
    modelled = energy_cnt > 0
    if modelled.any():
        mean_energy = np.zeros_like(energy_sum)
        mean_energy[modelled] = energy_sum[modelled] / energy_cnt[modelled]
        vals = mean_energy[modelled]
        zero_all = int(np.count_nonzero(energy_sum[modelled] == 0.0))
        empirical.update({
            "n_locations_modelled_at_least_once": int(modelled.sum()),
            "n_locations_never_modelled": int((~modelled).size - modelled.sum()),
            "min_observations_per_modelled_location": int(energy_cnt[modelled].min()),
            "max_observations_per_modelled_location": int(energy_cnt[modelled].max()),
            "empirical_zero_energy_location_count": zero_all,
            "min_location_mean_energy": float(vals.min()),
            "median_location_mean_energy": float(np.median(vals)),
            "max_location_mean_energy": float(vals.max())})
    if conj.size:
        empirical.update({
            "conjugate_pair_violation_min": float(conj.min()),
            "conjugate_pair_violation_median": float(np.median(conj)),
            "conjugate_pair_violation_p95": float(np.percentile(conj, 95)),
            "conjugate_pair_violation_max": float(conj.max())})
    empirical["requires_amendment_review"] = bool(
        empirical.get("empirical_zero_energy_location_count", 0) > 0)
    if empirical["requires_amendment_review"]:
        logger.error("empirical zero-energy locations found (NON-VERDICT): "
                     "%s -- flagged for amendment review, NOT blocking",
                     empirical.get("empirical_zero_energy_location_count"))
    return {
        "derivation": "COMPLEX branch, column mask: no Hermitian symmetry, so "
                      "no exact algebraic rule reduces the free-coordinate "
                      "count. Self-conjugate Fourier locations remain "
                      "two-real-dimensional. Checked PER REALISATION.",
        "structural_checks": structural, "structural_pass": ok,
        "first_failing_realisation": first_fail,
        "empirical_diagnostics": empirical,
        "empirical_note": "a location observed zero across the inspected "
                          "subset is a FINITE-SAMPLE observation, not an "
                          "algebraic constraint: it flags for amendment "
                          "review and never blocks"}


def _peak_rss_bytes() -> int:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru * 1024) if sys.platform.startswith("linux") else int(ru)


# ---------------------------------------------------------------------------
# Collection and per-slice validity
# ---------------------------------------------------------------------------

def _collect(parents: dict, data_root: str, batch: int,
             smoke: int | None) -> tuple[list[dict], tuple[int, int], dict]:
    """Prepare every frozen slice and harvest the live mask from batch['mask'],
    which is where P2 read it. The operator object is never excavated."""
    ds = FastMRISliceDataset(data_root, split="train", mode="eval")
    dataset_prov = dataset_provenance(FastMRISliceDataset, ds)
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

    rows: list[dict] = []
    grid = None
    for b in loader:
        p = _prepare(b, "cpu", test0=False)
        missing = [k for k in REQUIRED_PREPARE_KEYS if k not in p]
        if missing:
            raise StageError("PREPARE_CONTRACT_CHANGED",
                             f"_prepare() returned no {missing}")
        y, x_norm, cond_in, amax = p["y"], p["x_norm"], p["cond_in"], p["amax"]
        if cond_in.shape != x_norm.shape or x_norm.shape[1] != 2:
            raise StageError("STATE_SHAPE_UNEXPECTED",
                             f"x_norm {tuple(x_norm.shape)} / cond_in "
                             f"{tuple(cond_in.shape)} are not the expected "
                             f"two-channel states")
        h, w = int(x_norm.shape[-2]), int(x_norm.shape[-1])
        if grid is None:
            grid = (h, w)
        elif grid != (h, w):
            raise StageError("GRID_SHAPE_VARIES",
                             f"prepared grid {(h, w)} differs from {grid} "
                             f"within the frozen subset")
        masks = b["mask"]
        for j, meta in enumerate(b["meta"]):
            m = masks[j]
            if m.dtype != torch.bool or m.dim() != 1 or int(m.shape[-1]) != w:
                raise StageError("MASK_SHAPE",
                                 f"batch mask must be 1-D bool of width {w}, "
                                 f"got {tuple(m.shape)} {m.dtype}")
            rows.append({
                "file": meta["file"], "slice_index": int(meta["slice_index"]),
                "split": meta["split"], "mode": meta["mode"],
                "epoch": None, "test0": False,
                "mask_seed": int(meta["mask_seed"]),
                "mask_width": int(m.shape[-1]),
                "mask_n_columns": int(m.sum().item()),
                "selected_columns": tuple(int(c) for c in
                                          torch.nonzero(m).flatten().tolist()),
                "prepared_shapes": {"y": list(p["y"].shape[1:]),
                                    "x_norm": list(x_norm.shape[1:]),
                                    "cond_in": list(cond_in.shape[1:]),
                                    "tgt_norm": list(p["tgt_norm"].shape[1:]),
                                    "mask": list(m.shape)},
                "_y": y[j].detach().clone(),
                "_x_norm": x_norm[j].detach().clone(),
                "_cond_in": cond_in[j].detach().clone(),
                "_amax": float(amax[j].item()),
            })
    if len(rows) != len(indices):
        raise StageError("SUBSET_SIZE_MISMATCH",
                         f"collected {len(rows)} entries, expected "
                         f"{len(indices)}")
    for k, idx in enumerate(indices):
        rows[k]["dataset_index"] = int(idx)
    return rows, grid, dataset_prov


def _two_channel_to_complex(t: torch.Tensor) -> torch.Tensor:
    if t.dim() != 3 or t.shape[0] != 2:
        raise StageError("STATE_LAYOUT_UNEXPECTED",
                         f"expected a (2, H, W) real state, got "
                         f"{tuple(t.shape)}")
    return torch.complex(t[0], t[1])


def _slice_validity(row: dict, cmap, pair_index: dict, s_ref: float,
                    p2_rec: dict) -> np.ndarray:
    """C3a, C3c under three probes, fp64 sensitivity, P2 continuity.

    Returns the slice's free-coordinate trace for the empirical diagnostics.
    Mutates `row` with the recorded measurements.
    """
    y = row.pop("_y")
    x_norm = _two_channel_to_complex(row.pop("_x_norm"))
    cond_in = _two_channel_to_complex(row.pop("_cond_in"))
    amax = row.pop("_amax")
    if not np.isfinite(amax) or amax <= 0.0:
        raise StageError("AMAX_INVALID",
                         f"per-volume divisor is {amax!r}; it must be finite "
                         f"and strictly positive (no fallback)")
    if torch.is_complex(y):
        y_c = y
    elif y.dim() == 3 and y.shape[0] == 2:
        y_c = torch.complex(y[0], y[1])
    else:
        raise StageError("MEASUREMENT_LAYOUT_UNEXPECTED",
                         f"y has layout {tuple(y.shape)} {y.dtype}; expected "
                         f"complex or a (2, H, W) real pair")

    n = cmap.n_free_complex
    zero = torch.zeros(n, dtype=y_c.dtype)

    # ---- C3a: primary decoder at the zero state vs the LIVE anchor.
    x_zero = dec.decode_normalised(y_c, amax, zero, cmap)
    denom = float(torch.max(torch.abs(cond_in)).item())
    require_finite({"c3a_denominator": denom}, "P3 C3a denominator")
    if denom <= 0.0:
        raise StageError("C3A_DENOMINATOR_NON_POSITIVE",
                         "max_pixel |cond_in| is not strictly positive; the "
                         "relative anchor error is undefined")
    c3a_abs = float(torch.max(torch.abs(x_zero - cond_in)).item())

    # ---- raw-path lineage. NEVER gates: P3's decoder is deliberately an
    # independent implementation, so equivalent-but-different fp32 call paths
    # may differ without a defect.
    x_zero_raw = dec.decode_raw_path(y_c, amax, zero, cmap)
    raw_abs = float(torch.max(torch.abs(x_zero_raw - cond_in)).item())
    raw_bitwise = bool(torch.equal(x_zero_raw.real, cond_in.real)
                       and torch.equal(x_zero_raw.imag, cond_in.imag))
    path_abs = float(torch.max(torch.abs(x_zero - x_zero_raw)).item())

    # ---- residual, computed live
    dx = x_norm - cond_in
    k_dx = dec.fft2c(dx)
    resid_l2 = float(torch.linalg.vector_norm(k_dx).item())
    m = dec.column_mask_tensor(cmap, y_c.device, k_dx.dtype)
    max_m_f_dx = float(torch.max(torch.abs(k_dx * m)).item())
    require_finite({"kspace_residual_l2": resid_l2, "max_MFdx_p3": max_m_f_dx},
                   "P3 residual quantities")

    # ---- probes. The per-slice seed derives from the slice IDENTITY, not from
    # loader or iteration order, so a reordering cannot change the probe.
    seed_material = {"p3_probe_seed": P3_PROBE_SEED,
                     "dataset_index": row["dataset_index"],
                     "file": row["file"], "slice_index": row["slice_index"]}
    slice_seed = int(canonical_hash(seed_material)[:16], 16) % (2 ** 63 - 1)
    rng = np.random.default_rng(slice_seed)
    sigma_a = s_ref / np.sqrt(2.0 * n)
    sigma_b = resid_l2 / np.sqrt(2.0 * n)
    require_finite({"sigma_A": float(sigma_a), "sigma_B": float(sigma_b)},
                   "P3 probe scales")
    if sigma_b <= 0.0:
        raise StageError("PROBE_SCALE_NON_POSITIVE",
                         f"the residual-matched probe scale is {sigma_b!r}; a "
                         f"zero-energy residual leaves scale B undefined")

    def gauss(sigma):
        return torch.from_numpy(
            (rng.normal(0.0, sigma, n)
             + 1j * rng.normal(0.0, sigma, n)).astype(np.complex64))

    u_true = dec.gather_unmeasured(k_dx, cmap)
    probes = {"slice_probe_seed": slice_seed, "sigma_A": float(sigma_a),
              "sigma_B": float(sigma_b),
              "probe_scale_rule_A": "S_ref / sqrt(2 n_free)",
              "probe_scale_rule_B": "||F dx||_2 / sqrt(2 n_free), per slice"}
    for label, u in (("scale_A", gauss(float(sigma_a))),
                     ("scale_B", gauss(float(sigma_b))), ("true", u_true)):
        x_cand = dec.decode_normalised(y_c, amax, u, cmap)
        fa, fr = dec.measured_fixity(x_cand, y_c, amax, cmap)
        _, rfr = dec.raw_fixity(x_cand, y_c, amax, cmap)
        u_back = dec.gather_unmeasured(dec.fft2c(x_cand), cmap)
        uden = float(torch.max(torch.abs(u)).item())
        require_finite({f"{label}_roundtrip_denominator": uden},
                       f"P3 {label} round trip")
        if uden <= 0.0:
            raise StageError("ROUNDTRIP_DENOMINATOR_NON_POSITIVE",
                             f"max|u| is {uden!r} on probe {label}; the "
                             f"relative round-trip error is undefined")
        probes[label] = {
            "realized_probe_l2": float(torch.linalg.vector_norm(u).item()),
            "fixity_abs": fa, "fixity_rel": fr, "raw_fixity_rel": rfr,
            "unmeasured_roundtrip_rel":
                float(torch.max(torch.abs(u_back - u)).item()) / uden}

    # ---- E2: GENUINE precision sensitivity. Distinct from the operation-order
    # diagnostic above; both are NON-VERDICT and separately recorded.
    y64 = dec.to_precision(y_c, True)
    c64 = dec.to_precision(cond_in, True)
    zero64 = torch.zeros(n, dtype=torch.complex128)
    x_zero64 = dec.decode_normalised(y64, amax, zero64, cmap)
    c3a_abs64 = float(torch.max(torch.abs(x_zero64 - c64)).item())
    u64 = dec.to_precision(u_true, True)
    x_true64 = dec.decode_normalised(y64, amax, u64, cmap)
    f64_abs, f64_rel = dec.measured_fixity(x_true64, y64, amax, cmap)
    u_back64 = dec.gather_unmeasured(dec.fft2c(x_true64), cmap)
    uden64 = float(torch.max(torch.abs(u64)).item())
    require_finite({"c3a_abs_f64": c3a_abs64, "fixity_abs_f64": f64_abs,
                    "fixity_rel_f64": f64_rel, "roundtrip_denominator_f64":
                        uden64}, "P3 fp64 sensitivity")
    if uden64 <= 0.0:
        raise StageError("ROUNDTRIP_DENOMINATOR_NON_POSITIVE",
                         "max|u_true| is not strictly positive on the fp64 path")
    rt64 = float(torch.max(torch.abs(u_back64 - u64)).item()) / uden64

    # ---- P2 continuity. NON-VERDICT throughout, and the ONLY like-for-like
    # comparison is P3's max|M F dx| against P2's persisted max_MFdx: same
    # quantity, same domain, same norm.
    #
    # max|decode(u_true) - x_norm| is recorded alongside it but is NOT a
    # prediction of that scalar and must never be plotted against it on an
    # identity line. It equals max_image |F^H(M F dx)|, an IMAGE-domain
    # supremum, whereas max_MFdx is a FOURIER-domain supremum on measured
    # support. A unitary transform preserves the L2 norm, not the maximum, so
    # the two have no reason to agree. It is the image-domain consequence of
    # the measured-support leakage P2 already accepted.
    obs_recon_abs = float(torch.max(torch.abs(
        dec.decode_normalised(y_c, amax, u_true, cmap) - x_norm)).item())
    p2_leak = float(p2_field(p2_rec, "max_MFdx", f"slice {row['dataset_index']}"))
    p2_rer = float(p2_field(p2_rec, "residual_energy_ratio",
                            f"slice {row['dataset_index']}"))
    p3_rer = (resid_l2 ** 2) / (s_ref ** 2)

    row.update({
        "c3a_decode_zero_abs": c3a_abs,
        "c3a_decode_zero_rel": c3a_abs / denom,
        "c3a_denominator": denom,
        "c3a_pass": bool((c3a_abs / denom) <= P3_DECODE_TOL),
        "raw_decode_zero_bitwise_equal": raw_bitwise,
        "raw_decode_zero_abs": raw_abs,
        "raw_decode_zero_rel": raw_abs / denom,
        "normalized_vs_raw_path_abs": path_abs,
        "normalized_vs_raw_path_rel": path_abs / denom,
        "operation_order_diagnostic_verdict_affecting": False,
        "kspace_residual_l2": resid_l2,
        "probes": probes,
        "c3a_rel_f32": c3a_abs / denom, "c3a_rel_f64": c3a_abs64 / denom,
        "c3a_precision_abs_difference": abs(c3a_abs - c3a_abs64),
        "c3a_precision_rel_difference": rel_diff(c3a_abs, c3a_abs64),
        "fixity_rel_f32": probes["true"]["fixity_rel"],
        "fixity_rel_f64": f64_rel,
        "fixity_precision_rel_difference":
            rel_diff(probes["true"]["fixity_abs"], f64_abs),
        "roundtrip_rel_f32": probes["true"]["unmeasured_roundtrip_rel"],
        "roundtrip_rel_f64": rt64,
        "roundtrip_precision_rel_difference":
            rel_diff(probes["true"]["unmeasured_roundtrip_rel"], rt64),
        "precision_denominator_rule":
            "max(|a|, |b|, P3_PATH_DIFF_REL_FLOOR)",
        "precision_scope":
            "complex128 FFT/decoder arithmetic over float32-PREPARED inputs; "
            "NON-VERDICT OPERATOR sensitivity, NOT an end-to-end float64 data "
            "path. Mirrors the P2 float64 rule exactly.",
        "precision_verdict_affecting": False,
        # like-for-like: Fourier-domain supremum on measured support, both sides
        "max_MFdx_p3": max_m_f_dx,
        "max_MFdx_p2": p2_leak,
        "max_MFdx_abs_diff": abs(max_m_f_dx - p2_leak),
        "max_MFdx_rel_diff": rel_diff(max_m_f_dx, p2_leak),
        "residual_energy_ratio_p3": p3_rer,
        "residual_energy_ratio_p2": p2_rer,
        "residual_energy_ratio_abs_diff": abs(p3_rer - p2_rer),
        "residual_energy_ratio_rel_diff": rel_diff(p3_rer, p2_rer),
        # separate image-domain consequence; NOT comparable to max_MFdx
        "observed_true_reconstruction_abs": obs_recon_abs,
        "observed_true_reconstruction_domain": "image",
        "observed_true_reconstruction_note":
            "= max_image |F^H(M F dx)|; the image-domain consequence of the "
            "measured-support leakage P2 accepted. It is NOT a predictor of "
            "max_MFdx and must not be compared to it on an identity line.",
        "anchor_source_p2": p2_field(p2_rec, "x0_source_key",
                                     f"slice {row['dataset_index']}"),
        "anchor_source_p3": "cond_in",
        "p2_continuity_verdict_affecting": False,
    })
    return dec.gather_unmeasured(k_dx, cmap).numpy(), k_dx


def _gate(rows: list[dict], link1: dict, link2: dict, audit: dict, dims: dict,
          basis: dict, constraints: dict) -> None:
    """Construction / contract / identity gates. Failure is ERROR, per the
    LOCK 2 precedent: a wrong scatter order or a broken decoder is a defect in
    this stage, not a scientific finding about the dataset."""
    checks = {
        "link1_ordered_list_all_realisations":
            link1["all_published_vs_independent_equal"],
        "link2_unique_oracle_all_realisations":
            link2["all_unique_oracle_bitwise_equal"],
        "c1_c2_all_realisations": audit["all_c1_complete_and_c2_disjoint"],
        "c6_dimensions_all_realisations": dims["all_n_free_counts_agree"],
        "flow_dim_invariant_across_realisations": dims["flow_dim_invariant"],
        "basis_probes": basis["all_pass"],
        "structural_constraints": constraints["structural_pass"],
        "c3a": all(r["c3a_pass"] for r in rows),
        "c3c_fixity": all(r["probes"][l]["fixity_rel"] <= P3_FIXITY_TOL
                          for r in rows
                          for l in ("scale_A", "scale_B", "true")),
        "unmeasured_roundtrip":
            all(r["probes"][l]["unmeasured_roundtrip_rel"] <= P3_ROUNDTRIP_TOL
                for r in rows for l in ("scale_A", "scale_B", "true")),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise StageError("P3_CONSTRUCTION_GATE_FAILED",
                         f"construction gates failed: {failed}. These are "
                         f"contract or implementation defects, not data "
                         f"verdicts.", detail={"checks": checks})


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

def _build_facts(parents, p3par, rows, maps, bindings, census, link1, link2,
                 audit, dims,
                 basis, constraints, grid, verdict, reason, block, repo_dir,
                 script, argv, t0, smoke, diag_seconds, seed_prov,
                 dataset_prov) -> dict:
    # Defensive: any row that somehow reached facts-building unvalidated
    # would still carry the private tensor handles and none of the c3a_*
    # measurements. Strip the handles (canonical_bytes would reject tensors)
    # and never index a measurement key directly.
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    validated = [r for r in rows if "c3a_decode_zero_rel" in r]

    thresholds = {"P3_DECODE_TOL": P3_DECODE_TOL,
                  "P3_FIXITY_TOL": P3_FIXITY_TOL,
                  "P3_ROUNDTRIP_TOL": P3_ROUNDTRIP_TOL,
                  "P3_BASIS_OFFTARGET_TOL": P3_BASIS_OFFTARGET_TOL,
                  "P3_PROBE_SEED": P3_PROBE_SEED,
                  "P3_PATH_DIFF_REL_FLOOR": P3_PATH_DIFF_REL_FLOOR,
                  "P3_DECODE_EXPECTED_REL": P3_DECODE_EXPECTED_REL,
                  "shared_values_are_incidental": True,
                  "S_ref": parents["s_ref"] if parents else None,
                  "S_ref_squared": parents["s_ref_squared"] if parents else None}

    def _worst(getter, label, threshold):
        cand = [r for r in validated if isinstance(getter(r), (int, float))]
        if not cand:
            return None
        r = max(cand, key=getter)
        obs = getter(r)
        val, status = margin_of(threshold, obs)
        return {"gate": label, "dataset_index": r["dataset_index"],
                "file": r["file"], "slice_index": r["slice_index"],
                "observed": obs, "threshold": threshold, "margin": val,
                "margin_status": status}

    gated = [
        ("c3a_decode_zero", P3_DECODE_TOL,
         lambda r: r.get("c3a_decode_zero_rel")),
        ("c3c_fixity", P3_FIXITY_TOL,
         lambda r: max((r.get("probes", {}).get(l, {}).get("fixity_rel")
                        for l in ("scale_A", "scale_B", "true")), default=None)
         if r.get("probes") else None),
        ("unmeasured_roundtrip", P3_ROUNDTRIP_TOL,
         lambda r: max((r.get("probes", {}).get(l, {})
                        .get("unmeasured_roundtrip_rel")
                        for l in ("scale_A", "scale_B", "true")), default=None)
         if r.get("probes") else None),
    ]
    worst, margins = [], {}
    for label, thr, get in gated:
        if thr is None:
            continue
        w = _worst(get, label, thr)
        if w:
            worst.append(w)
            margins[label] = {"value": w["margin"], "status":
                              w["margin_status"], "observed": w["observed"],
                              "threshold": thr}
    if basis:
        val, status = margin_of(P3_BASIS_OFFTARGET_TOL,
                                basis["worst_off_target"])
        margins["basis_offtarget"] = {"value": val, "status": status,
                                      "observed": basis["worst_off_target"],
                                      "threshold": P3_BASIS_OFFTARGET_TOL}

    summary = {
        "n_slices": len(rows), "n_slices_validated": len(validated),
        "slices_validated": len(validated) == len(rows),
        "smoke": smoke is not None,
        "branch": "COMPLEX", "margins": margins, "worst_slices": worst,
        "grid_shape": list(grid) if grid else None,
        "c3a_expected_rel": P3_DECODE_EXPECTED_REL,
        "c3a_expected_note":
            "C3a is EXPECTED NON-ZERO at roughly 1e-7. The primary decoder "
            "assembles in NORMALISED k-space while cond_in was produced by "
            "raw-assembly-then-image-division: identical mathematics, "
            "different fp32 operation order. Two decades of margin is a PASS, "
            "not a near-miss, and must not be 'fixed' by reverting the "
            "registered decoder order.",
        "raw_path_bitwise_equal_count":
            sum(1 for r in rows if r.get("raw_decode_zero_bitwise_equal")),
        "identity_residual_note":
            "The v0.3 true-reconstruction identity residual was REMOVED before "
            "execution: it reduces algebraically to the zero-state anchor "
            "discrepancy (C3a) and carried no independent evidence. This is a "
            "PRE-EXECUTION REVIEW CORRECTION, not a result.",
        "true_probe_scope_note":
            "the true-coefficient probe is BLIND to a shared gather/scatter "
            "permutation, since scatter_P(gather_P(K)) = (1-M)K; the ordered "
            "list identity and the unique-valued oracle are the map validators",
        "p2_continuity_note":
            "P3-vs-P2 leakage and residual-ratio agreement is DIAGNOSTIC "
            "continuity evidence; no P3 gate depends on it",
        "diagnostic_runtime_seconds": diag_seconds,
    }

    code = hash_project_code(repo_dir, script)
    p3_code = hash_p3_local_code(repo_dir)
    parent_ids = {
        "p0_facts_sha256": parents["p0"]["facts_sha256"] if parents else None,
        "p0s_facts_sha256": parents["p0s"]["facts_sha256"] if parents else None,
        "subset_manifest_sha256":
            parents["p0s"]["subset_manifest_sha256"] if parents else None,
        "contract_hash": parents["p0"]["contract_hash"] if parents else None,
        "p1_facts_sha256": p3par["p1"]["facts_sha256"] if p3par else None,
        "p1_semantic_sha256": p3par["p1"]["semantic_sha256"] if p3par else None,
        "p2_facts_sha256": p3par["p2"]["facts_sha256"] if p3par else None,
        "p2_semantic_sha256": p3par["p2"]["semantic_sha256"] if p3par else None,
        "p1_ruling": p3par["p1"]["ruling"] if p3par else None,
    }

    map_audit = {"link1_ordered_list_identity": link1,
                 "link2_unique_valued_oracle": link2, "c1_c2": audit,
                 "c6_dimensions": dims, "basis_probes": basis}
    # A4: RULE + PER-REALISATION BINDINGS, not one map and not 256 coordinate
    # lists. Each binding carries the identity needed to prove
    # sample <-> mask <-> map correspondence, and the recorded acquired
    # columns are sufficient for a consumer to RE-DERIVE the map and its hash
    # under the published enumeration rule.
    realisations = [
        {"acquired_columns": list(k), "mask_sha256": e["mask_sha256"],
         "map_sha256": e["map_sha256"], "n_free_complex": e["n_free_complex"],
         "flow_dim_real": e["flow_dim_real"], "n_slices": e["n_slices"]}
        for k, e in sorted((maps or {}).items())]
    coordinate_map = {
        "format": "rule_plus_per_realisation_bindings",
        "map_serialized": False,
        "map_serialized_note":
            "NO global coordinate list is persisted. Under A4 the map is per "
            "realisation, and 256 lists would be ~14 MB of duplicated "
            "derivable data. The ENUMERATION RULE plus per-realisation hashes "
            "is the consumable contract.",
        "enumeration_rule": dec.P3_FLATTEN_ORDER,
        "complex_packing_order": dec.P3_COMPLEX_PACKING_ORDER,
        "enumeration_rule_code_sha256": _p3_local_sha(repo_dir,
                                                     "seqref_mri/src/residual_decoder.py"),
        "grid_shape": list(grid) if grid else None,
        "n_realisations": len(realisations),
        "realisations": realisations,
        "consumer_contract":
            "IMPL re-derives each map from the RECORDED acquired_columns under "
            "the enumeration rule above and REQUIRES the derived map_sha256 to "
            "equal the recorded value; a mismatch is ERROR. IMPL must never "
            "invent its own map or assume a global one.",
    }

    semantic = {
        "schema": FACTS_SCHEMA, "stage": "P3", "thresholds": thresholds,
        "verdict": verdict, "branch": "COMPLEX",
        "mask_census": census, "coordinate_map": coordinate_map,
        "per_slice_bindings": bindings or [],
        "map_audit": map_audit, "constraint_audit": constraints,
        "slices": rows,
        "summary": {k: v for k, v in summary.items()
                    if k != "diagnostic_runtime_seconds"},
        "parents": parent_ids,
        "code": code["project_local"] + p3_code["p3_local"],
    }

    facts = {
        "schema": FACTS_SCHEMA,
        "script": {"id": SCRIPT_ID, "version": SCRIPT_VERSION,
                   "lifetime": "KEEP"},
        "stage": "P3",
        "artefact_type": "stage_facts",
        "run_mode": ("smoke" if smoke is not None else "authoritative"),
        "authoritative": smoke is None,
        "stage_description": "coordinate map, decoder validity, identity audit",
        "branch": "COMPLEX",
        "thresholds": thresholds,
        "verdict": verdict,
        "verdict_reason": reason,
        "mask_census": census,
        "mask_seed_provenance": {
            **(seed_prov or {"resolved": False}),
            **(dataset_prov or {}),
            "seed_field": "meta['mask_seed']",
            "per_slice_seed_map_sha256":
                canonical_hash([[r["file"], r["slice_index"], r["mask_seed"]]
                                for r in rows]) if rows else None,
            "live_equals_persisted": census.get(
                "all_compared_live_persisted_equal"),
            "census_scope_note":
                "the census proves the LIVE seeds equal P2's PERSISTED seeds; "
                "the binding above shows what produced them. Neither alone is "
                "sufficient provenance.",
        },
        "decoder_contract": {
            "primary_decoder_order": "normalized_kspace_assembly",
            "sensitivity_decoder_order": "raw_assembly_then_image_division",
            "decoder_input_units": "normalized_fourier",
            "measurement_units": "raw_kspace",
            "decoder_output_units": "normalized_image",
            "flatten_order": dec.P3_FLATTEN_ORDER,
            "complex_packing_order": dec.P3_COMPLEX_PACKING_ORDER,
            "d2_steps_3_4_5_active": False,
            "d2_steps_3_4_5_reason": "conjugate fill, determined-from-partner "
                                     "and self-conjugate-real are INACTIVE "
                                     "under COMPLEX; recorded as explicitly "
                                     "empty classes, never omitted",
            "fourier": dec.fourier_provenance()},
        "coordinate_map": coordinate_map,
        "per_slice_bindings": bindings or [],
        "schema_compatibility": {
            "schema": FACTS_SCHEMA,
            "supersedes": LEGACY_FACTS_SCHEMA,
            "incompatible_with_legacy": True,
            "note": "seqref-p3-facts/1 was emitted by the v0.3.x GLOBAL-MAP "
                    "implementation. A /1 artefact must be REJECTED by a /2 "
                    "consumer, never reinterpreted as per-realisation facts. "
                    "The v0.3.1 smoke artefact remains readable only as "
                    "historical evidence about that implementation."},
        "map_audit": map_audit,
        "c3b_anchor_identity": {
            "x_anchor": "x0", "p3_anchor_source_key": "cond_in",
            "x_det_constructed": False,
            "x_det_reason": "x_det does not exist under the COMPLEX branch "
                            "(EXEC §7.1)",
            "verified_by": "C3a against the LIVE cond_in, not by declaration"},
        "c3d": {"status": "inapplicable",
                "reason": "Hermitian / imaginary-energy validity has no "
                          "content for a complex target; recorded INAPPLICABLE "
                          "and explicitly NOT as a pass"},
        "constraint_audit": constraints,
        "summary": summary,
        "slices": rows,
        "parents": {"p0_p0s": parents, "p1_p2": (
            {k: v for k, v in p3par.items() if k != "p2_by_index"}
            if p3par else None)},
        "code": {**code, **p3_code},
        "run": {**environment_record(repo_dir, argv),
                "runtime_seconds": time.time() - t0,
                "peak_memory_bytes": _peak_rss_bytes()},
        "hash_note": "the authoritative artefact SHA is the SHA-256 of THIS "
                     "FILE'S bytes, in the sidecar; semantic_sha256 covers "
                     "scientific content only and is not self-referential",
        "verify_before_use": ["IMPL must verify this file against its "
                              "sidecar, then RE-DERIVE each realisation's "
                              "map from the recorded acquired_columns under "
                              "the published enumeration rule and REQUIRE "
                              "the derived map_sha256 to equal the recorded "
                              "value; a mismatch is ERROR. Re-derivation is "
                              "mandatory, not optional: the hashes are the "
                              "divergence check P3 exists to provide"],
    }
    # `block` is retained in the signature and is always None under A4: the
    # stage has no scientific BLOCK outcome. Kept rather than deleted so a
    # future amendment reintroducing a data premise restores the record path
    # explicitly rather than by accident.
    if block is not None:
        raise StageError("UNEXPECTED_BLOCK_RECORD",
                         "a BLOCK record was constructed, but P3 under A4 has "
                         "no scientific BLOCK outcome")
    return attach_semantic_hash(facts, semantic)


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="SEQREF-P3CM v0.4.2 -- P3 per-realisation coordinate "
                    "maps and decoder validity")
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--p0-facts", required=True)
    ap.add_argument("--p0s-facts", required=True)
    ap.add_argument("--p0s-script", required=True)
    ap.add_argument("--p1-facts", required=True)
    ap.add_argument("--p2-facts", required=True)
    ap.add_argument("--out-dir", required=True,
                    help="P3 output directory; for a smoke run this must be "
                         "an EPHEMERAL directory, never the parents' directory")
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
    parents = p3par = None
    rows: list[dict] = []
    census = {}
    cmap = link1 = link2 = audit = dims = basis = constraints = None
    grid = None
    diag_seconds = None
    seed_prov = dataset_prov = None

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
        p3par = verify_p1_p2(
            args.p1_facts, args.p2_facts,
            expected_p1_sha=P1_FACTS_SHA256,
            expected_p2_sha=P2_FACTS_SHA256,
            expected_p1_semantic_sha=P1_SEMANTIC_SHA256,
            expected_p2_semantic_sha=P2_SEMANTIC_SHA256)

        # M4 bindings are resolved BEFORE any data is read, so an unbindable
        # seed rule fails on a name rather than after a full collection pass.
        seed_prov = bind_mask_seed_provenance(args.repo_dir)

        rows, grid, dataset_prov = _collect(parents, args.data_root,
                                            args.batch, args.smoke)
        height, width = grid

        # R1 FIRST. Under A4 the census has NO BLOCK branch: varying column
        # sets at a fixed acquired count are the EXPECTED regime, and
        # acquired-count variation raises ERROR from inside mask_census.
        census = mask_census(rows, p3par["p2_by_index"], args.smoke is None,
                             parents["subset_size"], width)

        # ---- ONE MAP PER MASK REALISATION (A4). Distinct realisations are
        # keyed by their column set; identical sets legitimately share a map,
        # which is a property of the data and not assumed either way.
        maps, bindings = build_realisation_maps(rows, height, width)
        link1, link2, audit, dims = audit_realisations(maps, height, width)
        # Basis probes run on ONE realisation by registered design: the FFT
        # pair identity is permutation-agnostic, and link 2 validates every
        # realisation's permutation bitwise (see the basis_probes docstring).
        basis = basis_probes(next(iter(maps.values()))["cmap"])
        cmap = None   # NO global map exists under A4

        # Conjugate-pair indices are per realisation and are built ONCE each,
        # not once per slice: the vectorised diagnostic is otherwise dominated
        # by index construction.
        pair_indices = {k: dec.conjugate_pair_index(e["cmap"])
                        for k, e in maps.items()}

        # EMPIRICAL COORDINATE ENERGY IS ACCUMULATED BY PHYSICAL GRID LOCATION,
        # not by packed index. Under A4 index k denotes a different Fourier
        # location on every realisation, so pooling traces by k would average
        # unlike quantities -- the same defect A5 corrects in D4's scaling
        # statistics. Accumulators are (H, W); counts differ per location
        # because a location contributes only when its column is free.
        energy_sum = np.zeros((height, width), dtype=np.float64)
        energy_cnt = np.zeros((height, width), dtype=np.int64)

        t_diag = 0.0
        conj = []
        for r in rows:
            entry = maps[r["selected_columns"]]
            rcmap = entry["cmap"]
            trace, k_dx = _slice_validity(
                r, rcmap, pair_indices[r["selected_columns"]],
                parents["s_ref"], p3par["p2_by_index"][r["dataset_index"]])
            r["map_sha256"] = entry["map_sha256"]
            r["mask_sha256"] = entry["mask_sha256"]
            mag2 = np.abs(np.asarray(trace)) ** 2
            energy_sum[rcmap.free_rows, rcmap.free_cols] += mag2
            energy_cnt[rcmap.free_rows, rcmap.free_cols] += 1
            td = time.time()
            conj.append(dec.conjugate_pair_violation(
                k_dx, pair_indices[r["selected_columns"]]))
            t_diag += time.time() - td
        diag_seconds = t_diag
        constraints = constraint_audit(
            maps, energy_sum, energy_cnt,
            np.concatenate(conj) if conj else np.zeros(0),
            pair_indices, height, width)

        _gate(rows, link1, link2, audit, dims, basis, constraints)

        reason = (f"the coordinate map audits clean on both links and the "
                  f"exact-DC decoder reproduces the anchor and preserves the "
                  f"measured data across {len(rows)} slices; "
                  f"flow_dim_real={dims['flow_dim_real']}")
        facts = _build_facts(parents, p3par, rows, maps, bindings, census,
                             link1, link2,
                             audit, dims, basis, constraints, grid, "PASS",
                             reason, None, args.repo_dir, script, raw_argv, t0,
                             args.smoke, diag_seconds, seed_prov, dataset_prov)
        path, sha = publish_stage(facts, args.out_dir, facts_prefix, "P3")
        logger.info("P3 PASS n=%d n_free=%d flow_dim=%d facts=%s "
                    "file_sha256=%s semantic_sha256=%s", len(rows),
                    dims["n_free_enumerated"], dims["flow_dim_real"], path, sha,
                    facts["semantic_sha256"])
        if args.smoke is not None:
            logger.warning("SMOKE run -- NOT authoritative; delete %s after "
                           "inspection", path)
        return EXIT_PASS

    # NO `except StageBlock` HANDLER. Under A4 this stage has no scientific
    # BLOCK outcome: every remaining gate tests a CONSTRUCTION, CONTRACT or
    # IDENTITY, and EXIT_BLOCK is UNREACHABLE. That is deliberate and recorded,
    # not an omission. Should a future amendment reintroduce a data premise
    # here, the handler and its publish-facts-first discipline must come back
    # with it; the v0.3.1 implementation is preserved under
    # v0.3_superseded/ as the reference for that path.
    except StageError as exc:
        logger.error("P3 ERROR [%s] -- %s", exc.error_code, exc.reason)
        publish_error(exc, args.out_dir, error_prefix, "P3",
                      parents=(parents or {}).get("p0"),
                      code={"script": script}, run={"argv": raw_argv})
        return EXIT_ERROR
    except Exception as exc:
        # Failure boundary. KeyboardInterrupt and SystemExit derive from
        # BaseException and are deliberately NOT caught. An ordinary exception
        # must never reach the caller as a traceback: a traceback has no
        # exit-code contract and leaves no artefact behind.
        logger.exception("%s UNEXPECTED ERROR", SCRIPT_ID)
        wrapped = StageError(
            "UNEXPECTED_RUNTIME_ERROR", f"{type(exc).__name__}: {exc}",
            detail={"exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "raised_after_parent_verification": parents is not None},
            write_record=parents is not None)
        publish_error(wrapped, args.out_dir, error_prefix, "P3",
                      parents=(parents or {}).get("p0"),
                      code={"script": script}, run={"argv": raw_argv})
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
