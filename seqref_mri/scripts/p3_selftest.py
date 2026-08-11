#!/usr/bin/env python3
# SEQREF-P3ST v0.5.1 -- P3 self-test (fixtures + REAL frozen-API integration)
# LIFETIME: KEEP
#
# CHANGELOG
# - v0.5.1 (2026-08-07): FIXTURE BUG, found by the first v0.5 run (137/138).
#   test_facts_builder_a4 spliced the fake block record over `reason`
#   (positional index 14), leaving `block` (index 15) None, so the
#   UNEXPECTED_BLOCK_RECORD guard never fired and the check failed. The
#   _build_facts calls in that fixture are now KEYWORD invocations so a
#   positional-index mistake cannot recur. No stage defect was demonstrated;
#   the failing check failing LOUDLY is the discipline working. Check counts
#   unchanged; the registry stands as confirmed by the v0.5 run.
# - v0.5 (2026-08-07): REWRITTEN TO THE A4 PER-REALISATION SPECIFICATION.
#   The v0.4.1 suite was BLOCK-era code: it asserted the falsified premise.
#   Fixtures REMOVED BY NAME (not renamed -- the falsified premise must not
#   survive in the suite's vocabulary):
#     * test_block_plot_safe          (BLOCK artefacts no longer exist)
#     * test_block_facts_with_collected_rows   (_build_facts now REJECTS a
#                                       block record; the defensive strip it
#                                       tested is kept as a fixture)
#     * test_varying_mask_block_publication    (inverted: varying sets now
#                                       PASS end to end)
#     * census.varying_sets_BLOCK / census.varying_counts_BLOCK
#   A4 inversions:
#     * varying fixed-count column sets -> census PASS, recorded as the
#       EXPECTED REGIME; the e2e fixture drives real main() to EXIT_PASS and
#       verifies the /2 artefact, the bindings and the re-derivation identity.
#     * acquired-count variation -> StageError MASK_ACQUIRED_COUNT_VARIES,
#       exercised both as a unit injection and end to end through main().
#     * a StageBlock raised inside main() now surfaces as
#       UNEXPECTED_RUNTIME_ERROR (no handler exists) with NO facts published.
#     * _build_facts RAISES UNEXPECTED_BLOCK_RECORD if a block record is ever
#       constructed -- the guard is exercised.
#   New coverage: per-realisation map building (distinct masks -> distinct
#   maps, shared masks -> shared maps, canonical mask hashes, binding
#   identity), the retained flow_dim invariance gate OBSERVED TO FIRE, schema
#   /2 constants, /1 reader rejection by name, map-less /2 plot safety.
#   REGISTRY DISCIPLINE: EXPECTED_COUNTS for v0.5 was derived by STATIC
#   COUNT of this source, NOT carried forward from v0.4.1. The coverage
#   audit fails loudly on any mismatch, so the first real run either
#   confirms the registry or blocks the suite until it is corrected from
#   that run's actual counts. A green suite with a stale registry is
#   impossible by construction.
# - v0.4.1 (2026-08-02): registry corrected from a real run. Total was 119.
# - v0.4 (2026-08-02): adds the COVERAGE AUDIT (per-fixture registered
#   counts; a skipped check is a failing check).
# - v0.3.x (2026-08-02): fired-flag discipline; BLOCK-path fixtures that
#   exercised the old taxonomy's publication boundary.
# - v0.2 (2026-07-30): real frozen-API integration section.
#
# USAGE
#   python -m seqref_mri.scripts.p3_selftest --repo-dir . \
#       --log-file seqref_mri/results/_diag/p3_selftest.log

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

import preflight_io as pio  # noqa: E402
import preflight_parents as ppar  # noqa: E402
import preflight_parents_p3 as pp3  # noqa: E402
import residual_decoder as dec  # noqa: E402
from seqref_mri.scripts import p3_coordinate_map as p3  # noqa: E402
from seqref_mri.scripts import p3_plots as plots  # noqa: E402

logger = logging.getLogger("SEQREF-P3ST")
RESULTS: list[tuple[str, bool, str]] = []

# Per-fixture expected check counts. A green suite proves nothing about checks
# that never EXECUTED: several sit behind `if errs:` / `if pub:` guards, and a
# skipped one merely shrinks the total silently. Registering the count makes a
# skip loud. v0.5 values are a STATIC COUNT of this source -- the first real
# run must confirm them, and any mismatch fails the coverage audit, which is
# the designed behaviour (never carry a registry forward across a rewrite).
EXPECTED_COUNTS = {
    "test_fft_conventions": 4,
    "test_map_and_oracles": 12,
    "test_decoder": 8,
    "test_conjugate_diagnostic": 3,
    "test_census_taxonomy_a4": 22,
    "test_plot_reader": 8,
    "test_frozen_api_integration": 25,
    "test_parent_pinning": 7,
    "test_facts_builder_a4": 7,
    "test_varying_mask_pass_publication": 12,
    "test_semantic_hash_sensitivity": 3,
    "test_main_failure_boundary": 26,
}


def check(name: str, ok, detail: str = "") -> None:
    ok = bool(ok)
    RESULTS.append((name, ok, detail))
    (logger.info if ok else logger.error)("%-56s %s %s", name,
                                          "PASS" if ok else "FAIL", detail)


def raises(exc_types, fn, *a, **k) -> bool:
    try:
        fn(*a, **k)
        return False
    except exc_types:
        return True


def raises_code(exc_type, code: str, fn, *a, **k) -> bool:
    """Fired-flag discipline for typed injections: the raise must happen AND
    carry the expected code -- an injection that does not fire is reported as
    itself, not silently tested as something else."""
    try:
        fn(*a, **k)
        return False
    except exc_type as exc:
        return getattr(exc, "error_code", None) == code


# ===========================================================================
# A. FFT conventions -- the v0.1 endswith collapse must be impossible
# ===========================================================================

def test_fft_conventions() -> None:
    x = torch.randn(8, 8, dtype=torch.complex64)
    outs = {n: dec.reference_fft2(x, n) for n in dec.FFT_CONVENTIONS}
    names = list(outs)
    check("fft.four_conventions_distinct",
          all(not torch.allclose(outs[a], outs[b])
              for i, a in enumerate(names) for b in names[i + 1:]),
          "the endswith defect would collapse these to two")
    check("fft.rejects_unregistered_name",
          raises(ValueError, dec.reference_fft2, x, "made_up"))
    prov = dec.fourier_provenance()
    check("fft.pair_inherited_from_frozen_module",
          prov["fft_module"] == "seqref_mri.src.fastmri_data"
          and prov["fft_convention_inherited"] is True,
          "P3 registers no convention of its own")
    check("fft.stage_registers_no_convention_constant",
          not any(n.startswith("P3_FFT") for n in dir(p3)),
          "a stage-level convention constant is the defect class itself")


# ===========================================================================
# B. Map, oracles, payload
# ===========================================================================

def test_map_and_oracles() -> None:
    H = W = 8
    cols = [0, 1, 4]
    cmap = dec.build_coordinate_map(cols, H, W)
    ind = p3.enumerate_free_independent(cols, H, W)

    check("map.link1_passes_on_correct_map",
          p3.link1_ordered_list(cmap, ind)["published_vs_independent_equal"])
    l2 = p3.link2_unique_oracle(cmap, ind)
    check("map.link2_passes_on_correct_map",
          l2["unique_oracle_bitwise_equal"] and l2["off_support_exactly_zero"])
    a = p3.c1_c2_audit(cmap)
    check("map.c1_c2_pass", a["c1_complete"] and a["c2_no_double_assignment"])
    d = p3.c6_dimensions(cmap, len(cols), H, W)
    check("map.three_n_free_counts_agree", d["n_free_counts_agree"],
          f"{d['n_free_formula']}/{d['n_free_full_grid_count']}/"
          f"{d['n_free_enumerated']}")

    perm = np.arange(cmap.n_free_complex)
    perm[0], perm[1] = perm[1], perm[0]
    bad = dec.CoordinateMap(height=H, width=W, mask_columns=cmap.mask_columns,
                            free_rows=cmap.free_rows[perm],
                            free_cols=cmap.free_cols[perm])
    check("map.link1_fires_on_permutation",
          not p3.link1_ordered_list(bad, ind)["published_vs_independent_equal"])
    check("map.link2_fires_on_permutation",
          not p3.link2_unique_oracle(bad, ind)["unique_oracle_bitwise_equal"])

    n = cmap.n_free_complex
    k = np.arange(n, dtype=np.float64)
    swapped = torch.from_numpy(((n - k) + 1j * (k + 1.0)).astype(np.complex64))
    produced = dec.scatter_unmeasured(swapped, cmap).numpy()
    expected = np.zeros((H, W), dtype=np.complex64)
    for i, (r, c) in enumerate(ind):
        expected[r, c] = np.complex64(complex(i + 1, n - i))
    check("map.oracle_catches_real_imag_swap",
          not np.array_equal(produced, expected))

    payload = cmap.payload()
    check("map.payload_round_trip",
          dec.CoordinateMap.from_payload(payload).ordered_coordinates()
          == cmap.ordered_coordinates() and payload["map_serialized"] is True)
    corrupt = dict(payload, map_payload_sha256="0" * 64)
    check("map.payload_sha_mismatch_rejected",
          raises(ValueError, dec.CoordinateMap.from_payload, corrupt))
    check("map.payload_carries_full_ordered_list",
          len(payload["free_coordinates"]) == cmap.n_free_complex,
          "a hash of a list that is not in the artefact is not a map")
    check("map.rejects_duplicate_columns",
          raises(ValueError, dec.build_coordinate_map, [1, 1, 2], H, W))
    check("map.rejects_out_of_range_column",
          raises(ValueError, dec.build_coordinate_map, [0, W], H, W))


# ===========================================================================
# C. Decoder, probes, and the demonstrated N1 redundancy
# ===========================================================================

def test_decoder() -> None:
    H = W = 8
    cols = [0, 1, 4]
    cmap = dec.build_coordinate_map(cols, H, W)
    torch.manual_seed(0)
    x_true = torch.randn(H, W, dtype=torch.complex64)
    amax = 2.0
    m = dec.column_mask_tensor(cmap, x_true.device, torch.complex64)
    y_raw = dec.fft2c(x_true) * m * amax
    cond_in = dec.ifft2c(y_raw * m) / amax
    zero = torch.zeros(cmap.n_free_complex, dtype=torch.complex64)

    x0 = dec.decode_normalised(y_raw, amax, zero, cmap)
    c3a = (float(torch.max(torch.abs(x0 - cond_in)).item())
           / float(torch.max(torch.abs(cond_in)).item()))
    check("decoder.c3a_reproduces_anchor", c3a <= p3.P3_DECODE_TOL,
          f"rel={c3a:.3e}")

    u = torch.randn(cmap.n_free_complex, dtype=torch.complex64)
    x_cand = dec.decode_normalised(y_raw, amax, u, cmap)
    _, fr = dec.measured_fixity(x_cand, y_raw, amax, cmap)
    check("decoder.fixity_under_nonzero_probe", fr <= p3.P3_FIXITY_TOL,
          f"rel={fr:.3e}")
    u_back = dec.gather_unmeasured(dec.fft2c(x_cand), cmap)
    rt = (float(torch.max(torch.abs(u_back - u)).item())
          / float(torch.max(torch.abs(u)).item()))
    check("decoder.free_coordinate_round_trip", rt <= p3.P3_ROUNDTRIP_TOL,
          f"rel={rt:.3e}")
    check("decoder.basis_probes_pass", p3.basis_probes(cmap)["all_pass"])
    check("decoder.rejects_real_input",
          raises(ValueError, dec.decode_normalised, y_raw.real, amax, zero, cmap))
    check("decoder.rejects_wrong_u_length",
          raises(ValueError, dec.scatter_unmeasured,
                 torch.zeros(3, dtype=torch.complex64), cmap))

    # N1 -- the removed v0.3 gate reduces to C3a. Demonstrated, not gated.
    k_dx = dec.fft2c(x_true - cond_in)
    u_true = dec.gather_unmeasured(k_dx, cmap)
    lhs = (dec.decode_normalised(y_raw, amax, u_true, cmap) - x_true
           - (-dec.ifft2c(k_dx * m)))
    rhs = dec.decode_normalised(y_raw, amax, zero, cmap) - cond_in
    check("n1.identity_residual_equals_c3a",
          float(torch.max(torch.abs(lhs - rhs)).item()) < 1e-5,
          "pre-execution correction: the v0.3 gate had no independent content")

    perm = np.arange(cmap.n_free_complex)
    perm[0], perm[1] = perm[1], perm[0]
    pm = dec.CoordinateMap(height=H, width=W, mask_columns=cmap.mask_columns,
                           free_rows=cmap.free_rows[perm],
                           free_cols=cmap.free_cols[perm])
    blind = float(torch.max(torch.abs(
        dec.decode_normalised(y_raw, amax, dec.gather_unmeasured(k_dx, pm), pm)
        - dec.decode_normalised(y_raw, amax, u_true, cmap))).item())
    check("n1.true_probe_blind_to_shared_permutation", blind < 1e-5,
          "scatter_P(gather_P(K)) = (1-M)K -- the permutation cancels")


# ===========================================================================
# D. Vectorised conjugate diagnostic vs a slow reference
# ===========================================================================

def test_conjugate_diagnostic() -> None:
    H = W = 8
    cmap = dec.build_coordinate_map([0, 1, 4], H, W)
    idx = dec.conjugate_pair_index(cmap)
    torch.manual_seed(1)
    k = torch.randn(H, W, dtype=torch.complex64)
    fast = dec.conjugate_pair_violation(k, idx)

    acquired = set(cmap.mask_columns)
    den = float(torch.max(torch.abs(k)).item())
    slow, n_pairs = [], 0
    for r, c in cmap.ordered_coordinates():
        pr, pc = (-r) % H, (-c) % W
        if pc in acquired or (pr * W + pc) < (r * W + c):
            continue
        n_pairs += 1
        slow.append(float(torch.abs(k[r, c] - torch.conj(k[pr, pc])).item()) / den)
    slow = np.asarray(slow)
    check("conj.pair_count_equal", len(fast) == len(slow) == idx["n_pairs"],
          f"fast={len(fast)} slow={len(slow)} idx={idx['n_pairs']}")
    check("conj.vectorised_matches_slow_reference",
          len(fast) == len(slow)
          and float(np.max(np.abs(np.sort(fast) - np.sort(slow)))) < 1e-6,
          "performance fix must not change the measurement")
    check("conj.rejects_zero_denominator",
          raises(ValueError, dec.conjugate_pair_violation,
                 torch.zeros(H, W, dtype=torch.complex64), idx))


# ===========================================================================
# E. Census taxonomy (A4) and per-realisation map building
# ===========================================================================

def _p2rec(idx, cols, seed=7):
    return {"dataset_index": idx, "file": f"f{idx}.h5", "slice_index": idx,
            "mask_selected_columns": list(cols), "mask_seed": seed,
            "mask_n_columns": len(cols), "mask_width": 8, "max_MFdx": 1e-7,
            "k_i": 1.0, "relative_max": 1e-6, "residual_energy_ratio": 0.1,
            "x0_prepared_source_key": "cond_in", "x0_source_key": "cond_in",
            "x0_rel_error": 0.0}


def _live(idx, cols, seed=7):
    return {"dataset_index": idx, "file": f"f{idx}.h5", "slice_index": idx,
            "split": "train", "selected_columns": tuple(cols),
            "mask_seed": seed}


def test_census_taxonomy_a4() -> None:
    A, B, wide = (0, 1, 4), (0, 1, 5), (0, 1, 4, 6)
    same_live = [_live(i, A) for i in range(4)]
    same_p2 = {i: _p2rec(i, A) for i in range(4)}
    cen = p3.mask_census(same_live, same_p2, False, 4, 8)
    check("census.uniform_masks_pass",
          cen["outcome"] == "PASS" and cen["acquired_count_invariant"] is True)
    check("census.global_map_retired_even_when_uniform",
          cen["global_map_applicable"] is False
          and cen["no_block_branch"] is True,
          "A4: no global map exists even when every mask is identical")
    check("census.rename_applied",
          "all_compared_live_persisted_equal" in cen
          and "full_frozen_population_compared" in cen)

    varied_live = [_live(0, A), _live(1, B), _live(2, A), _live(3, A)]
    varied_p2 = {0: _p2rec(0, A), 1: _p2rec(1, B), 2: _p2rec(2, A),
                 3: _p2rec(3, A)}
    cv = p3.mask_census(varied_live, varied_p2, False, 4, 8)
    check("census.varying_sets_PASS_expected_regime",
          cv["outcome"] == "PASS" and cv["column_sets_vary"] is True
          and cv["column_set_variation_is_expected"] is True
          and cv["n_distinct_sets"] == 2,
          "A4 inversion: this was the MASK_SETS_VARY_FIXED_COUNT block")
    check("census.census_returned_not_raised",
          cv.get("per_slice_mask_seeds") and cv.get("distinct_column_sets_sample")
          and cv.get("n_unique_mask_seeds") is not None,
          "the census is a finished result before control flow changes")

    cnt_live = [_live(0, A), _live(1, wide), _live(2, A), _live(3, A)]
    cnt_p2 = {0: _p2rec(0, A), 1: _p2rec(1, wide), 2: _p2rec(2, A),
              3: _p2rec(3, A)}
    check("census.varying_counts_is_ERROR",
          raises_code(ppar.StageError, "MASK_ACQUIRED_COUNT_VARIES",
                      p3.mask_census, cnt_live, cnt_p2, False, 4, 8),
          "A4: a broken generator contract is ERROR, not BLOCK")

    drift_live = [_live(0, B)] + [_live(i, A) for i in range(1, 4)]
    check("census.live_persisted_drift_is_ERROR",
          raises_code(ppar.StageError, "MASK_LIVE_PERSISTED_MISMATCH",
                      p3.mask_census, drift_live, same_p2, False, 4, 8))
    seed_live = [dict(_live(i, A), mask_seed=99) for i in range(4)]
    check("census.mask_seed_drift_is_ERROR",
          raises(ppar.StageError, p3.mask_census, seed_live, same_p2, False, 4, 8),
          "M4 provenance participates in the census, not only the columns")
    check("census.authoritative_short_population_is_ERROR",
          raises(ppar.StageError, p3.mask_census, same_live, same_p2, True, 256, 8))

    # ---- per-realisation map building (A4 contract term 1 / 6a)
    rows = [dict(_live(i, A if i % 2 == 0 else B)) for i in range(4)]
    maps, bindings = p3.build_realisation_maps(rows, 8, 8)
    check("realisations.distinct_masks_distinct_maps",
          len(maps) == 2
          and len({e["map_sha256"] for e in maps.values()}) == 2
          and len(bindings) == 4)
    check("realisations.shared_mask_shared_map",
          maps[A]["n_slices"] == 2 and maps[B]["n_slices"] == 2
          and bindings[0]["map_sha256"] == bindings[2]["map_sha256"],
          "sharing is a property of the data, not assumed either way")
    check("realisations.mask_hash_is_canonical",
          maps[A]["mask_sha256"] == pio.canonical_hash(
              {"width": 8, "selected_columns": list(A)}),
          "one canonicalisation rule for structured content")
    need = {"dataset_index", "file", "slice_index", "mask_seed",
            "acquired_columns", "mask_sha256", "map_sha256", "flow_dim_real"}
    check("realisations.bindings_carry_identity",
          all(need <= set(b) for b in bindings),
          "enough identity to prove sample <-> mask <-> map correspondence")

    link1, link2, audit, dims = p3.audit_realisations(maps, 8, 8)
    check("realisations.audit_all_pass_on_valid_maps",
          link1["all_published_vs_independent_equal"]
          and link2["all_unique_oracle_bitwise_equal"]
          and audit["all_c1_complete_and_c2_disjoint"]
          and dims["all_n_free_counts_agree"]
          and dims["n_realisations_audited"] == 2)
    check("realisations.flow_dim_invariant_on_uniform_count",
          dims["flow_dim_invariant"] is True
          and dims["distinct_flow_dim_real"] == [maps[A]["flow_dim_real"]])

    mixed = [dict(_live(0, A)), dict(_live(1, wide))]
    mmaps, _ = p3.build_realisation_maps(mixed, 8, 8)
    _, _, _, mdims = p3.audit_realisations(mmaps, 8, 8)
    check("realisations.flow_dim_invariance_FIRES",
          mdims["flow_dim_invariant"] is False
          and len(mdims["distinct_flow_dim_real"]) == 2,
          "the retained dimensional gate must be observed to fail")

    # ---- constraint audit under the /2 signature
    pair_indices = {k: dec.conjugate_pair_index(e["cmap"])
                    for k, e in maps.items()}
    base = p3.constraint_audit(maps, np.zeros((8, 8)),
                               np.zeros((8, 8), dtype=np.int64),
                               np.zeros(0), pair_indices, 8, 8)
    check("constraints.structural_pass_on_valid_maps", base["structural_pass"])
    check("constraints.empirical_non_verdict",
          base["empirical_diagnostics"]["all_empirical_fields_non_verdict"])
    check("constraints.duplicate_trace_absent_with_reason",
          base["empirical_diagnostics"]["duplicate_trace_candidates"] is None
          and "NOT computed under A4"
          in base["empirical_diagnostics"]["duplicate_trace_note"],
          "recorded as absent with its reason, never reported as zero")
    check("constraints.self_conjugate_not_restricted_under_complex",
          base["structural_checks"]["self_conjugate_restricted_coordinates"] == 0,
          "without Hermitian symmetry these stay two-real-dimensional")

    esum = np.zeros((8, 8))
    ecnt = np.zeros((8, 8), dtype=np.int64)
    for j, (r, c) in enumerate(maps[A]["cmap"].ordered_coordinates()[:3]):
        ecnt[r, c] = 2
        esum[r, c] = 0.0 if j == 0 else 0.5 * j
    meas = p3.constraint_audit(maps, esum, ecnt, np.asarray([0.1, 0.5, 0.9]),
                               pair_indices, 8, 8)
    emp = meas["empirical_diagnostics"]
    check("constraints.empirical_values_measured",
          emp["empirical_zero_energy_location_count"] == 1
          and emp["conjugate_pair_violation_median"] is not None
          and emp["min_observations_per_modelled_location"] == 2,
          "v0.1 hardcoded requires_amendment_review = false")
    check("constraints.zero_coordinate_flags_review_not_error",
          emp["requires_amendment_review"] is True
          and meas["structural_pass"] is True,
          "a finite-sample zero is not an algebraic constraint")


# ===========================================================================
# F. Plot reader: /2 acceptance, /1 rejection BY NAME, map-less plot safety
# ===========================================================================

def _write_pair(base, name, facts):
    p = os.path.join(base, name)
    body = pio.canonical_bytes(facts)
    with open(p, "wb") as fh:
        fh.write(body)
    with open(p + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{pio.file_sha256(p)}  {name}\n")
    return p


def test_plot_reader() -> None:
    good = {"schema": plots.EXPECTED_SCHEMA, "stage": plots.EXPECTED_STAGE,
            "artefact_type": plots.EXPECTED_TYPE, "verdict": "PASS"}
    with tempfile.TemporaryDirectory() as td:
        for name, mut, reject in (
                ("legacy_v1_schema", {"schema": plots.LEGACY_SCHEMA}, True),
                ("wrong_schema", {"schema": "seqref-p2-facts/1"}, True),
                ("wrong_stage", {"stage": "P2"}, True),
                ("error_record", {"artefact_type": "error"}, True),
                ("good", {}, False)):
            p = _write_pair(td, f"{name}.json", {**good, **mut})
            check(f"plots.reader_{name}",
                  raises(ValueError, plots.load_verified, p) == reject)
        p = _write_pair(td, "legacy_named.json",
                        {**good, "schema": plots.LEGACY_SCHEMA})
        try:
            plots.load_verified(p)
            check("plots.reader_legacy_rejection_is_explicit", False, "no raise")
        except ValueError as exc:
            check("plots.reader_legacy_rejection_is_explicit",
                  "SUPERSEDED" in str(exc) and plots.LEGACY_SCHEMA in str(exc),
                  "the /2 contract requires rejection, never reinterpretation")
        p = os.path.join(td, "tampered.json")
        with open(p, "wb") as fh:
            fh.write(b'{"schema":"x"}')
        with open(p + ".sha256", "w", encoding="utf-8") as fh:
            fh.write("0" * 64 + "  tampered.json\n")
        check("plots.reader_sidecar_mismatch",
              raises(Exception, plots.load_verified, p))

        # The /2 analogue of the old block-plot-safety fixture: an artefact
        # with NO serialised map and no slices must still be plottable.
        mapless = {**good, "run_mode": "smoke",
                   "mask_census": {"outcome": "PASS", "no_block_branch": True},
                   "coordinate_map": {
                       "format": "rule_plus_per_realisation_bindings",
                       "grid_shape": [8, 8], "realisations": [],
                       "n_realisations": 0},
                   "map_audit": {}, "constraint_audit": {},
                   "slices": [], "summary": {"worst_slices": []}}
        try:
            n = plots.plot_all(mapless, Path(td) / "figs")
            check("plots.mapless_v2_facts_plot_safe", len(n) >= 1,
                  f"{len(n)} figure(s) from a map-less /2 artefact")
        except Exception as exc:  # noqa: BLE001
            check("plots.mapless_v2_facts_plot_safe", False,
                  f"{type(exc).__name__}: {exc}")


# ===========================================================================
# G. REAL frozen-API integration -- the gap that let v0.4 through
# ===========================================================================

def test_frozen_api_integration() -> None:
    sigs = {
        "guard_run_mode": ["out_dir", "smoke"],
        "verify_parents": ["repo_dir", "p0_facts_path", "p0s_facts_path",
                           "p0s_script_path"],
        "attach_semantic_hash": ["facts", "semantic_payload"],
        "hash_project_code": ["repo_dir", "script_path"],
        "environment_record": ["repo_dir", "argv"],
        "publish_stage": ["facts", "out_dir", "prefix", "stage"],
        "publish_error": ["exc", "out_dir", "prefix", "stage"],
    }
    for name, expected in sigs.items():
        fn = getattr(ppar, name, None)
        if fn is None:
            check(f"api.{name}_exists", False, "absent from preflight_parents")
            continue
        actual = [p for p, v in inspect.signature(fn).parameters.items()
                  if v.kind in (v.POSITIONAL_OR_KEYWORD, v.POSITIONAL_ONLY)]
        check(f"api.{name}_signature", actual[:len(expected)] == expected,
              f"actual={actual}")
    check("api.no_acquire_release_claim",
          not hasattr(ppar, "acquire_claim") and not hasattr(ppar, "release_claim"),
          "publication_claim is a context manager owned by publish_stage")
    check("api.publication_claim_is_context_manager",
          hasattr(ppar, "publication_claim"))
    check("api.stage_does_not_manage_claims",
          "publication_claim" not in inspect.getsource(p3.main)
          and "acquire_claim" not in inspect.getsource(p3.main),
          "acquiring manually would trip PUBLICATION_CLAIM_HELD against P3")
    check("api.prepare_signature_is_batch_device_test0",
          list(inspect.signature(
              __import__("seqref_mri.scripts.train_base", fromlist=["_prepare"])
              ._prepare).parameters)[:3] == ["batch", "device", "test0"]
          if hasattr(__import__("seqref_mri.scripts.train_base",
                                fromlist=["_prepare"]), "_prepare") else False,
          "the device is a POSITIONAL second argument")
    check("api.stage_has_no_block_machinery",
          not hasattr(p3, "StageBlock") and not hasattr(p3, "EXIT_BLOCK"),
          "A4: the stage imports neither; exit 1 is unreachable by design")
    check("api.stage_schema_constants_are_v2",
          p3.FACTS_SCHEMA == "seqref-p3-facts/2"
          and p3.LEGACY_FACTS_SCHEMA == "seqref-p3-facts/1"
          and plots.EXPECTED_SCHEMA == "seqref-p3-facts/2")

    with tempfile.TemporaryDirectory() as td:
        check("api.guard_accepts_clean_smoke_dir",
              ppar.guard_run_mode(td, True) == "smoke")
        open(os.path.join(td, "support_facts.json"), "w").close()
        check("api.guard_refuses_smoke_into_authoritative",
              raises(ppar.StageError, ppar.guard_run_mode, td, True))

    with tempfile.TemporaryDirectory() as td:
        facts = {"schema": "seqref-p3-facts/2", "stage": "P3",
                 "artefact_type": "stage_facts", "verdict": "PASS",
                 "run": {"utc": pio.utc_stamp()}}
        ppar.attach_semantic_hash(facts, {"schema": "seqref-p3-facts/2",
                                          "verdict": "PASS"})
        check("api.attach_semantic_hash_two_args",
              isinstance(facts.get("semantic_sha256"), str)
              and "included_keys" in facts.get("semantic_scope", {}))
        path, sha = ppar.publish_stage(facts, td, "selftest_p3", "P3")
        check("api.publish_stage_writes_and_verifies",
              pio.verify_sidecar(path) == sha)
        check("api.no_claim_residue",
              not [n for n in os.listdir(td) if n.endswith(".claim")])
        exc = ppar.StageError("SELFTEST_ERROR", "fixture", detail={"x": 1},
                              write_record=True)
        ep = ppar.publish_error(exc, td, "selftest_p3_error", "P3")
        check("api.publish_error_writes_typed_record",
              ep is not None
              and json.load(open(ep))["artefact_type"] == "error")
        untrusted = ppar.StageError("SELFTEST_UNTRUSTED", "fixture",
                                    write_record=False)
        check("api.untrusted_error_writes_nothing",
              ppar.publish_error(untrusted, td, "selftest_p3_error2", "P3")
              is None,
              "the ABSENCE of a record is the signal when parents are unverified")

    check("api.p3_local_code_hash_covers_new_modules",
          {f["relpath"] for f in pp3.hash_p3_local_code(_REPO)["p3_local"]}
          == set(pp3.P3_CODE_FILES),
          "the frozen CODE_HASH_FILES cannot name P3's modules")
    check("api.p2_field_spellings_frozen",
          pp3.P2_SLICE_KEYS["max_MFdx"] == "max_MFdx"
          and pp3.P2_SLICE_KEYS["selected_columns"] == "mask_selected_columns",
          "read from the authoritative artefact, not guessed")
    check("api.p2_field_missing_is_typed_error",
          raises(ppar.StageError, pp3.p2_field, {}, "max_MFdx", "fixture"))

    seed = pp3.bind_mask_seed_provenance(_REPO)
    check("api.mask_seed_provenance_bound",
          seed.get("resolved") is True
          and seed.get("mask_seed_source_sha256")
          and seed.get("seed_tuple_fields_from_source") is not None,
          "M4 must be READ from the executing code, not declared")
    from seqref_mri.src.fastmri_data import FastMRISliceDataset as _DS
    dsp = pp3.dataset_provenance(_DS)
    sig = dsp.get("dataset_init_signature") or ""
    check("api.dataset_signature_recorded",
          "split" in sig and "mode" in sig, f"signature={sig}")


def test_parent_pinning() -> None:
    """The registered P1/P2 identities must be ENFORCED, not merely stored."""
    base = {"schema": "seqref-p1-facts/1", "stage": "P1",
            "artefact_type": "stage_facts", "verdict": "PASS",
            "ruling": "COMPLEX", "semantic_sha256": "s" * 64,
            "slices": [{"dataset_index": 0, "mask_selected_columns": [0],
                        "mask_seed": 1, "max_MFdx": 1e-7}]}
    p2base = dict(base, schema="seqref-p2-facts/1", stage="P2")
    with tempfile.TemporaryDirectory() as td:
        p1p = _write_pair(td, "p1.json", base)
        p2p = _write_pair(td, "p2.json", p2base)
        real1, real2 = pio.file_sha256(p1p), pio.file_sha256(p2p)

        out = pp3.verify_p1_p2(p1p, p2p, expected_p1_sha=real1,
                               expected_p2_sha=real2,
                               expected_p1_semantic_sha="s" * 64,
                               expected_p2_semantic_sha="s" * 64)
        check("pin.correct_identities_pass",
              out["identity_pinning"]["all_identities_pinned"] is True)
        check("pin.wrong_p1_byte_sha_rejected",
              raises(ppar.StageError, pp3.verify_p1_p2, p1p, p2p,
                     expected_p1_sha="0" * 64, expected_p2_sha=real2))
        check("pin.wrong_p2_byte_sha_rejected",
              raises(ppar.StageError, pp3.verify_p1_p2, p1p, p2p,
                     expected_p1_sha=real1, expected_p2_sha="0" * 64))
        check("pin.wrong_semantic_sha_rejected",
              raises(ppar.StageError, pp3.verify_p1_p2, p1p, p2p,
                     expected_p1_sha=real1, expected_p2_sha=real2,
                     expected_p1_semantic_sha="0" * 64))

        stripped = _write_pair(td, "p1_nosem.json",
                               {k: v for k, v in base.items()
                                if k != "semantic_sha256"})
        check("pin.missing_semantic_sha_rejected",
              raises(ppar.StageError, pp3.verify_p1_p2, stripped, p2p,
                     expected_p1_sha=pio.file_sha256(stripped),
                     expected_p2_sha=real2))
        real_p1 = _write_pair(td, "p1_real.json", dict(base, ruling="REAL"))
        check("pin.non_complex_ruling_rejected",
              raises(ppar.StageError, pp3.verify_p1_p2, real_p1, p2p,
                     expected_p1_sha=pio.file_sha256(real_p1),
                     expected_p2_sha=real2))

    src = inspect.getsource(p3.main)
    check("pin.stage_passes_all_four_identities",
          all(k in src for k in ("expected_p1_sha", "expected_p2_sha",
                                 "expected_p1_semantic_sha",
                                 "expected_p2_semantic_sha")),
          "an unpinned load would accept any PASSing P1/P2")


# ===========================================================================
# G2. Facts builder under /2 (the defensive strip, retained; the BLOCK guard)
# ===========================================================================

_STUB_PARENTS = {
    "p0": {"path": "/fixture/p0", "facts_sha256": "0" * 64,
           "contract_hash": "c" * 64, "source_manifest_sha256": "m" * 64,
           "git": {}},
    "p0s": {"path": "/fixture/p0s", "facts_sha256": "1" * 64,
            "subset_manifest_sha256": "s" * 64,
            "population_manifest_sha256": "p" * 64, "population_size": 4,
            "git": {}},
    "s_ref": 1.0, "s_ref_squared": 1.0, "median_convention": {},
    "subset_indices": [0, 1, 2, 3], "subset_size": 4, "dataset": {},
}
_STUB_P3PAR = {
    "p1": {"facts_sha256": "a" * 64, "semantic_sha256": "b" * 64,
           "ruling": "COMPLEX", "schema": "seqref-p1-facts/1", "stage": "P1",
           "artefact_type": "stage_facts", "verdict": "PASS"},
    "p2": {"facts_sha256": "c" * 64, "semantic_sha256": "d" * 64,
           "schema": "seqref-p2-facts/1", "stage": "P2",
           "artefact_type": "stage_facts", "verdict": "PASS",
           "n_slice_records": 4},
    "p2_by_index": {}, "identity_pinning": {"all_identities_pinned": True},
    "branch": "COMPLEX",
}


def _collected_rows(sets=((0, 1, 4), (0, 1, 5)), n=4):
    """Rows as _collect leaves them: metadata plus the PRIVATE tensor handles,
    none of the c3a_* measurements. canonical_bytes would reject the handles."""
    return [{"dataset_index": i, "file": f"f{i}.h5", "slice_index": i,
             "split": "train", "mask_seed": 3 + i, "mask_width": 8,
             "mask_n_columns": len(sets[i % 2]),
             "selected_columns": tuple(sets[i % 2]),
             "prepared_shapes": {"y": [8, 8]},
             "_y": torch.zeros(2, 2), "_x_norm": torch.zeros(2, 2, 2),
             "_cond_in": torch.zeros(2, 2, 2), "_amax": 1.0}
            for i in range(n)]


def test_facts_builder_a4() -> None:
    rows = _collected_rows()
    maps, bindings = p3.build_realisation_maps(rows, 8, 8)
    pair_indices = {k: dec.conjugate_pair_index(e["cmap"])
                    for k, e in maps.items()}
    link1, link2, audit, dims = p3.audit_realisations(maps, 8, 8)
    basis = p3.basis_probes(maps[(0, 1, 4)]["cmap"])
    constraints = p3.constraint_audit(maps, np.zeros((8, 8)),
                                      np.zeros((8, 8), dtype=np.int64),
                                      np.zeros(0), pair_indices, 8, 8)
    census = {"n_slices_compared": 4, "all_compared_live_persisted_equal": True,
              "outcome": "PASS", "no_block_branch": True}
    script = os.path.abspath(p3.__file__)
    # KEYWORD invocation, deliberately: v0.5 built the call as a positional
    # tuple and the BLOCK-guard check then spliced the fake record over
    # `reason` (index 14), leaving `block` (index 15) as None -- the guard
    # correctly never fired and the check failed. A positional-index mistake
    # must not be able to recur silently.
    kwargs = dict(parents=_STUB_PARENTS, p3par=_STUB_P3PAR, rows=rows,
                  maps=maps, bindings=bindings, census=census, link1=link1,
                  link2=link2, audit=audit, dims=dims, basis=basis,
                  constraints=constraints, grid=(8, 8), verdict="PASS",
                  reason="fixture", block=None, repo_dir=_REPO, script=script,
                  argv=[], t0=0.0, smoke=4, diag_seconds=None,
                  seed_prov={"resolved": True},
                  dataset_prov={"dataset_class": "fixture"})
    facts = p3._build_facts(**kwargs)
    check("facts.schema_is_v2",
          facts["schema"] == "seqref-p3-facts/2"
          and facts["script"]["version"] == p3.SCRIPT_VERSION)
    check("facts.rule_plus_bindings_format",
          facts["coordinate_map"]["format"]
          == "rule_plus_per_realisation_bindings"
          and facts["coordinate_map"]["map_serialized"] is False
          and facts["coordinate_map"]["n_realisations"] == 2)
    check("facts.per_slice_bindings_published",
          len(facts["per_slice_bindings"]) == 4
          and {b["map_sha256"] for b in facts["per_slice_bindings"]}
          == {r["map_sha256"]
              for r in facts["coordinate_map"]["realisations"]})
    check("facts.legacy_marked_incompatible",
          facts["schema_compatibility"]["incompatible_with_legacy"] is True
          and facts["schema_compatibility"]["supersedes"]
          == "seqref-p3-facts/1")
    check("facts.private_handles_stripped_unvalidated_recorded",
          not any(k.startswith("_") for r in facts["slices"] for k in r)
          and facts["summary"]["slices_validated"] is False
          and facts["summary"]["n_slices_validated"] == 0,
          "the defensive strip stays; collected != validated is stated")
    check("facts.serialises_canonically",
          json.loads(pio.canonical_bytes(facts))["schema"]
          == "seqref-p3-facts/2")
    check("facts.block_record_rejected_under_A4",
          raises_code(ppar.StageError, "UNEXPECTED_BLOCK_RECORD",
                      p3._build_facts,
                      **{**kwargs, "block": {"block_code": "FIXTURE"}}),
          "the guard must fire if a BLOCK record is ever constructed")


# ===========================================================================
# G3. END-TO-END: varying sets PASS through the real main() under A4
# ===========================================================================

def _coherent_rows(sets=((0, 1, 4), (0, 1, 5)), n=4, H=8, W=8):
    """Fully coherent fake slices: _y / _x_norm / _cond_in / _amax consistent
    so the REAL _slice_validity, gates and facts builder all execute."""
    rows, p2idx = [], {}
    for i in range(n):
        cols = sets[i % len(sets)]
        cmap = dec.build_coordinate_map(list(cols), H, W)
        rng = np.random.default_rng(1000 + i)
        x_true = torch.from_numpy(
            (rng.normal(size=(H, W))
             + 1j * rng.normal(size=(H, W))).astype(np.complex64))
        amax = 2.0
        m = dec.column_mask_tensor(cmap, x_true.device, torch.complex64)
        y_raw = dec.fft2c(x_true) * m * amax
        cond_in = dec.ifft2c(y_raw * m) / amax
        rows.append({
            "dataset_index": i, "file": f"f{i}.h5", "slice_index": i,
            "split": "train", "mode": "eval", "epoch": None, "test0": False,
            "mask_seed": 100 + i, "mask_width": W,
            "mask_n_columns": len(cols), "selected_columns": tuple(cols),
            "prepared_shapes": {"y": [H, W]},
            "_y": y_raw,
            "_x_norm": torch.stack([x_true.real, x_true.imag]),
            "_cond_in": torch.stack([cond_in.real, cond_in.imag]),
            "_amax": amax})
        p2idx[i] = _p2rec(i, cols, seed=100 + i)
    return rows, p2idx


def _run_main_with_rows(out_dir, rows, p2idx, subset_size):
    """Drive the REAL main() past parent verification with stubbed verifiers
    and a stubbed _collect; census, maps, slice validity, gates, facts and
    publication all run for real. Section G tests the verifiers themselves."""
    argv = ["--repo-dir", _REPO, "--data-root", out_dir,
            "--p0-facts", "x", "--p0s-facts", "x", "--p0s-script", "x",
            "--p1-facts", "x", "--p2-facts", "x", "--out-dir", out_dir,
            "--smoke", str(len(rows))]
    saved = {"vp": p3.verify_parents, "vpp": p3.verify_p1_p2,
             "seed": p3.bind_mask_seed_provenance, "col": p3._collect}
    p3.verify_parents = lambda *a, **k: dict(_STUB_PARENTS,
                                             subset_size=subset_size)
    p3.verify_p1_p2 = lambda *a, **k: dict(_STUB_P3PAR, p2_by_index=p2idx)
    p3.bind_mask_seed_provenance = lambda *a, **k: {"resolved": True}
    p3._collect = lambda *a, **k: ([dict(r) for r in rows], (8, 8),
                                   {"dataset_class": "fixture"})
    try:
        return p3.main(argv)
    finally:
        p3.verify_parents, p3.verify_p1_p2 = saved["vp"], saved["vpp"]
        p3.bind_mask_seed_provenance = saved["seed"]
        p3._collect = saved["col"]


def test_varying_mask_pass_publication() -> None:
    """END-TO-END A4 inversion: real main(), real census, real publisher.

    The v0.3.x fixture proved a varying-set census BLOCKED and published.
    Under A4 the same population must PASS, publish a /2 artefact, and every
    binding must re-derive to the recorded hashes. This fixture is the one
    that would have caught the stale MASK_SETS_VARY_FIXED_COUNT gate: it
    executes the design the stage is specified to implement.
    """
    rows, p2idx = _coherent_rows()
    with tempfile.TemporaryDirectory() as td:
        rc = _run_main_with_rows(td, rows, p2idx, subset_size=4)
        check("e2e.varying_sets_exit_PASS", rc == ppar.EXIT_PASS,
              "A4: per-slice realisation is the registered regime")
        pub = [n for n in os.listdir(td)
               if n.startswith("smoke_coordinate_map.")
               and n.endswith(".json")]
        check("e2e.pass_artefact_published", len(pub) == 1, f"{pub}")
        if not pub:
            return
        path = os.path.join(td, pub[0])
        check("e2e.sidecar_verifies", bool(pio.verify_sidecar(path)))
        f = json.load(open(path))
        check("e2e.schema_is_v2",
              f.get("schema") == "seqref-p3-facts/2"
              and (f.get("schema_compatibility") or {})
              .get("incompatible_with_legacy") is True)
        cen = f.get("mask_census") or {}
        check("e2e.census_records_expected_regime",
              cen.get("outcome") == "PASS"
              and cen.get("column_sets_vary") is True
              and cen.get("no_block_branch") is True
              and cen.get("n_distinct_sets") == 2)
        cm = f.get("coordinate_map") or {}
        check("e2e.two_realisations_published",
              cm.get("n_realisations") == 2
              and cm.get("format") == "rule_plus_per_realisation_bindings")
        bindings = f.get("per_slice_bindings") or []
        rel_sha = {r.get("map_sha256") for r in cm.get("realisations") or []}
        check("e2e.bindings_rederive_identity",
              len(bindings) == 4
              and all(b["mask_sha256"] == pio.canonical_hash(
                  {"width": 8, "selected_columns": b["acquired_columns"]})
                  for b in bindings)
              and {b["map_sha256"] for b in bindings} == rel_sha,
              "a consumer re-derives mask and map hashes from the record alone")
        dims = (f.get("map_audit") or {}).get("c6_dimensions") or {}
        check("e2e.flow_dim_invariant",
              dims.get("flow_dim_invariant") is True
              and len(dims.get("distinct_flow_dim_real") or []) == 1)
        ma = f.get("map_audit") or {}
        check("e2e.map_audit_links_pass",
              (ma.get("link1_ordered_list_identity") or {})
              .get("all_published_vs_independent_equal") is True
              and (ma.get("link2_unique_valued_oracle") or {})
              .get("all_unique_oracle_bitwise_equal") is True)
        check("e2e.slices_validated_true",
              (f.get("summary") or {}).get("slices_validated") is True
              and (f.get("summary") or {}).get("n_slices") == 4)
        check("e2e.no_block_fields",
              f.get("block_code") is None and f.get("verdict") == "PASS")
        check("e2e.no_claim_residue",
              not [n for n in os.listdir(td) if n.endswith(".claim")])


def test_semantic_hash_sensitivity() -> None:
    """The semantic hash must move with SCIENCE and code, and stay still under
    runtime metadata."""
    def payload(code_sha):
        return {"schema": "seqref-p3-facts/2", "verdict": "PASS",
                "code": [{"relpath": "x.py", "sha256": code_sha}]}
    a, b = {}, {}
    ppar.attach_semantic_hash(a, payload("aa"))
    ppar.attach_semantic_hash(b, payload("bb"))
    check("semantic.code_hash_change_moves_semantic_hash",
          a["semantic_sha256"] != b["semantic_sha256"])
    c, d = {"run": {"utc": "t1"}}, {"run": {"utc": "t2"}}
    ppar.attach_semantic_hash(c, payload("aa"))
    ppar.attach_semantic_hash(d, payload("aa"))
    check("semantic.runtime_metadata_does_not_move_it",
          c["semantic_sha256"] == d["semantic_sha256"],
          "two scientifically identical reruns must agree")
    check("semantic.scope_is_recorded",
          "included_keys" in c["semantic_scope"]
          and "excluded" in c["semantic_scope"])


# ===========================================================================
# H. Real main() failure boundary (A4: BLOCK is unreachable, seen to be so)
# ===========================================================================

def _run_main_with(out_dir, *, fail_at, exc):
    """Drive the REAL main() past parent verification, then fail. Returns
    (exit_code, injection_fired): an injection that does not fire must be
    reported as such, not silently tested as something else."""
    argv = ["--repo-dir", _REPO, "--data-root", out_dir,
            "--p0-facts", "x", "--p0s-facts", "x", "--p0s-script", "x",
            "--p1-facts", "x", "--p2-facts", "x", "--out-dir", out_dir,
            "--smoke", "2"]
    saved = {"vp": p3.verify_parents, "vpp": p3.verify_p1_p2,
             "seed": p3.bind_mask_seed_provenance, "target": getattr(p3, fail_at)}
    p3.verify_parents = lambda *a, **k: dict(_STUB_PARENTS)
    p3.verify_p1_p2 = lambda *a, **k: dict(_STUB_P3PAR)
    p3.bind_mask_seed_provenance = lambda *a, **k: {"resolved": True}

    fired = {"value": False}

    def boom(*a, **k):
        fired["value"] = True
        raise exc
    setattr(p3, fail_at, boom)
    try:
        return p3.main(argv), fired["value"]
    finally:
        p3.verify_parents, p3.verify_p1_p2 = saved["vp"], saved["vpp"]
        p3.bind_mask_seed_provenance = saved["seed"]
        setattr(p3, fail_at, saved["target"])


def _jsons(td, prefix):
    return [n for n in os.listdir(td)
            if n.startswith(prefix) and n.endswith(".json")]


def test_main_failure_boundary() -> None:
    # --- UNTRUSTED: parents unverifiable -> exit 2, NO artefact at all.
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "out")
        os.makedirs(out)
        missing = os.path.join(td, "absent.json")
        rc = p3.main(["--repo-dir", _REPO, "--data-root", td,
                      "--p0-facts", missing, "--p0s-facts", missing,
                      "--p0s-script", missing, "--p1-facts", missing,
                      "--p2-facts", missing, "--out-dir", out, "--smoke", "2"])
        check("main.untrusted_parent_exit_error", rc == ppar.EXIT_ERROR,
              f"rc={rc}")
        check("main.untrusted_parent_writes_no_artefact",
              not _jsons(out, ""), "the ABSENCE of a record is the signal")
        check("main.untrusted_no_claim_residue",
              not [n for n in os.listdir(out) if n.endswith(".claim")])
        check("main.bad_smoke_size_is_error",
              p3.main(["--repo-dir", _REPO, "--data-root", td,
                       "--p0-facts", missing, "--p0s-facts", missing,
                       "--p0s-script", missing, "--p1-facts", missing,
                       "--p2-facts", missing, "--out-dir", out,
                       "--smoke", "0"]) == ppar.EXIT_ERROR)

    # --- TRUSTED, untyped: a downstream RuntimeError -> typed ERROR record.
    with tempfile.TemporaryDirectory() as td:
        rc, fired = _run_main_with(
            td, fail_at="_collect",
            exc=RuntimeError("simulated post-parent failure"))
        check("main.untyped_injection_fired", fired,
              "the injected failure must be the one under test")
        errs = _jsons(td, "smoke_coordinate_map_error")
        facts = _jsons(td, "smoke_coordinate_map.")
        check("main.trusted_failure_exit_error", rc == ppar.EXIT_ERROR,
              f"rc={rc}")
        check("main.trusted_failure_writes_error_record", len(errs) == 1,
              f"found {errs}")
        check("main.trusted_failure_publishes_no_facts", not facts,
              f"found {facts}")
        check("main.trusted_no_claim_residue",
              not [n for n in os.listdir(td) if n.endswith(".claim")])
        if errs:
            rec = json.load(open(os.path.join(td, errs[0])))
            check("main.error_record_typed",
                  rec.get("artefact_type") == "error"
                  and rec.get("verdict") == "ERROR")
            check("main.error_code_is_unexpected_runtime",
                  rec.get("error_code") == "UNEXPECTED_RUNTIME_ERROR",
                  f"got {rec.get('error_code')}")
            check("main.error_detail_records_exception_type",
                  (rec.get("detail") or {}).get("exception_type")
                  == "RuntimeError")
            check("main.error_record_sidecar_verifies",
                  bool(pio.verify_sidecar(os.path.join(td, errs[0]))))

    # --- TRUSTED, typed: a StageError code must SURVIVE, not be replaced.
    with tempfile.TemporaryDirectory() as td:
        rc, fired = _run_main_with(
            td, fail_at="_collect",
            exc=ppar.StageError("SELFTEST_CONSTRUCTION", "fixture",
                                detail={"k": 1}, write_record=True))
        check("main.typed_injection_fired", fired)
        errs = _jsons(td, "smoke_coordinate_map_error")
        check("main.typed_construction_error_exit", rc == ppar.EXIT_ERROR)
        if errs:
            rec = json.load(open(os.path.join(td, errs[0])))
            check("main.typed_error_code_preserved",
                  rec.get("error_code") == "SELFTEST_CONSTRUCTION",
                  f"got {rec.get('error_code')} -- a typed code must not be "
                  f"replaced by UNEXPECTED_RUNTIME_ERROR")

    # --- A4 INVERSION: a StageBlock surfacing is a RUNTIME DEFECT, not a
    # verdict. No handler exists, so the generic boundary wraps it, publishes
    # an ERROR record and NO facts. The old publish-facts-first BLOCK path is
    # gone with the taxonomy; this fixture proves it.
    with tempfile.TemporaryDirectory() as td:
        rc, fired = _run_main_with(
            td, fail_at="_collect",
            exc=ppar.StageBlock("SELFTEST_PREMISE", "fixture", observed=2,
                                threshold=1, n_failing=2))
        check("main.block_injection_fired", fired)
        errs = _jsons(td, "smoke_coordinate_map_error")
        facts = _jsons(td, "smoke_coordinate_map.")
        check("main.block_is_unreachable_exit_error", rc == ppar.EXIT_ERROR,
              "A4: no StageBlock handler exists; a BLOCK surfacing is a defect")
        check("main.block_surfaces_as_error_record", len(errs) == 1,
              f"{errs}")
        check("main.block_publishes_NO_facts", not facts,
              "the publish-facts-first BLOCK path is gone with the taxonomy")
        if errs:
            rec = json.load(open(os.path.join(td, errs[0])))
            check("main.block_wrapped_as_unexpected_runtime",
                  rec.get("error_code") == "UNEXPECTED_RUNTIME_ERROR")
            check("main.block_exception_type_recorded",
                  (rec.get("detail") or {}).get("exception_type")
                  == "StageBlock")

    # --- A4 reclassified ERROR, end to end: acquired-count variation is a
    # broken generator contract. Real census raises; the typed code must
    # survive to the record. Fired-flag: the record's code IS the proof the
    # census path executed.
    rows, p2idx = _coherent_rows(sets=((0, 1, 4), (0, 1, 4, 6)), n=2)
    with tempfile.TemporaryDirectory() as td:
        rc = _run_main_with_rows(td, rows, p2idx, subset_size=2)
        errs = _jsons(td, "smoke_coordinate_map_error")
        facts = _jsons(td, "smoke_coordinate_map.")
        check("main.count_varies_exit_error", rc == ppar.EXIT_ERROR)
        check("main.count_varies_typed_record", len(errs) == 1, f"{errs}")
        check("main.count_varies_publishes_no_facts", not facts)
        if errs:
            rec = json.load(open(os.path.join(td, errs[0])))
            check("main.count_varies_code_preserved",
                  rec.get("error_code") == "MASK_ACQUIRED_COUNT_VARIES",
                  "the reclassified ERROR injection is seen to fire end to end")


# ===========================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="SEQREF-P3ST v0.5.1 -- P3 self-test (A4 per-realisation)")
    ap.add_argument("--repo-dir", default=_REPO)
    ap.add_argument("--log-file", default="p3_selftest.log")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.log_file, mode="w",
                                      encoding="utf-8")],
        format="%(asctime)s %(levelname)s %(message)s")

    counts: dict[str, int] = {}
    for fn in (test_fft_conventions, test_map_and_oracles, test_decoder,
               test_conjugate_diagnostic, test_census_taxonomy_a4,
               test_plot_reader, test_frozen_api_integration,
               test_parent_pinning, test_facts_builder_a4,
               test_varying_mask_pass_publication,
               test_semantic_hash_sensitivity, test_main_failure_boundary):
        logger.info("--- %s ---", fn.__name__)
        before = len(RESULTS)
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            logger.exception("fixture %s raised", fn.__name__)
            check(f"{fn.__name__}.completed", False,
                  f"{type(exc).__name__}: {exc}")
        counts[fn.__name__] = len(RESULTS) - before

    # Coverage audit BEFORE the pass/fail summary: a check that did not run is
    # not a check that passed.
    coverage_ok = True
    for name, expected in EXPECTED_COUNTS.items():
        ran = counts.get(name, 0)
        if ran != expected:
            coverage_ok = False
            logger.error("COVERAGE: %s ran %d checks, expected %d -- a guarded "
                         "check was skipped or the registry is stale",
                         name, ran, expected)
        else:
            logger.info("coverage %-46s %d/%d", name, ran, expected)
    RESULTS.append(("selftest.coverage_matches_registry", coverage_ok,
                    f"executed={sum(counts.values())} "
                    f"registered={sum(EXPECTED_COUNTS.values())}"))

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    logger.info("SEQREF-P3ST: %d/%d checks passed (coverage_ok=%s)",
                passed, total, coverage_ok)
    for name, ok, detail in RESULTS:
        if not ok:
            logger.error("FAILED: %s %s", name, detail)
    print(json.dumps({"passed": passed, "total": total,
                      "coverage_ok": coverage_ok, "per_fixture": counts,
                      "failed": [n for n, ok, _ in RESULTS if not ok]},
                     indent=2))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
