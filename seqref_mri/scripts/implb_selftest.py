# SEQREF-IMPLBT v0.3 -- scripts.implb_selftest
# LIFETIME: KEEP
# Purpose: self-test for SEQREF-IMPLB v0.3 (IMPL-B spline-bound
#   calibration, schema seqref-implb-facts/1). Pure-core fixtures: no
#   dataset, no h5, no k-space. The suite pins the EXACT percentile
#   interpolation rule (independent exact-arithmetic reference), the
#   interleaved re/im packing order and count, the grid-keyed P4 /2
#   applied-pair gather (physical (r, c), never packed index), the
#   binding-identity and mandatory map re-derivation gates, the P4
#   applied-pair validity gates, the dual-hash parent pins, the
#   MANDATORY parent sidecar gates, the generator pin, the
#   PASS|ERROR-only taxonomy, and the publication no-overwrite
#   machinery.
# REGISTRY DISCIPLINE: EXPECTED_COUNTS is a STATIC count of this source,
#   re-derived per rewrite, never carried forward. The coverage audit
#   fails loudly on any mismatch: a green suite with a stale registry is
#   impossible by construction.
# CONVENTION: fixture failures are reported, never hidden; the suite
#   exits nonzero unless every check passes AND coverage matches the
#   registry.
# Changelog
#   v0.1 (2026-08-12) Created with SEQREF-IMPLB v0.1 against the
#     registered IMPL-B contract (IMPLSPEC v0.1 + 2026-08-11 percentile
#     amendment, incorporated from EXEC §13), before any calibration
#     observation. Never executed.
#   v0.2 (2026-08-12) Aligned with SEQREF-IMPLB v0.2 (pre-execution
#     review remediation): new test_parent_sidecars fixture -- missing
#     and wrong sidecar are ERROR for BOTH the P3 and the P4 /2 parent;
#     coverage registry re-derived (73 checks across 12 fixtures). No
#     smoke fixtures exist because IMPL-B v0.2 has no smoke mode.
#   v0.3 (2026-08-12) Aligned with SEQREF-IMPLB v0.3: new
#     test_failure_boundary fixture drives main() with verified-parent
#     stubs and an injected ordinary RuntimeError, proving
#     UNEXPECTED_RUNTIME_ERROR + EXIT_ERROR + typed error record with
#     verified sidecar, no facts artefact and no claim residue (P4S2T
#     main()-integration precedent). --repo-dir must now resolve to the
#     installation repo (it cannot silently lie about the artefact
#     paths). Coverage registry re-derived (80 checks, 13 fixtures).
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from fractions import Fraction

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

from preflight_io import (canonical_hash, file_sha256,  # noqa: E402
                          verify_sidecar)
from preflight_parents import StageError  # noqa: E402
import residual_decoder as dec  # noqa: E402
from seqref_mri.scripts import implb_calibration as implb  # noqa: E402
from seqref_mri.src import fastmri_data as fdm  # noqa: E402

SCRIPT_ID = "SEQREF-IMPLBT"
SCRIPT_VERSION = "v0.3"
logger = logging.getLogger(SCRIPT_ID)

_RESULTS: list[tuple[str, str, bool, str]] = []
_CURRENT = ["<none>"]

P3_ART = os.path.join(_REPO, "seqref_mri", "results", "_diag", "p3",
                      "coordinate_map.json")
P4S2_ART = os.path.join(_REPO, "seqref_mri", "results", "_diag", "p4",
                        "scaling_statistics.json")


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((_CURRENT[0], name, bool(cond), detail))


def expect_error(name: str, code: str, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except StageError as exc:
        check(name, exc.error_code == code,
              f"got {exc.error_code}, want {code}")
        return
    except Exception as exc:  # noqa: BLE001 -- reported, not hidden
        check(name, False, f"wrong exception {type(exc).__name__}: {exc}")
        return
    check(name, False, f"no error raised, want {code}")


def _binding_for(mask: np.ndarray, height: int = 96,
                 width: int = 96) -> tuple[dict, object]:
    """A valid synthetic P3-style binding for a REAL generated mask, built
    by the same rules the P3 stage used."""
    cols = [int(c) for c in np.nonzero(mask)[0]]
    cmap = dec.build_coordinate_map(cols, height, width)
    binding = {"dataset_index": 7, "file": "f.h5", "slice_index": 3,
               "split": "train", "mask_seed": 123456789,
               "acquired_columns": cols,
               "mask_sha256": canonical_hash(
                   {"width": width, "selected_columns": cols}),
               "map_sha256": cmap.payload()["map_payload_sha256"],
               "n_free_complex": cmap.n_free_complex,
               "flow_dim_real": cmap.flow_dim_real}
    return binding, cmap


def _row_for(binding: dict, order: int = 0) -> dict:
    return {"corpus_order": order,
            "dataset_index": binding["dataset_index"],
            "file": binding["file"],
            "slice_index": binding["slice_index"],
            "split": binding["split"],
            "mask_seed": binding["mask_seed"],
            "live_columns": tuple(binding["acquired_columns"])}


def _ref_percentile_linear(values, p: float) -> float:
    """Independent EXACT-arithmetic reference for np.percentile
    method=\"linear\": sort; rank = p/100 * (n - 1); linear interpolation
    between floor and ceil ranks. Fractions over the exact float values,
    so the reference shares no implementation with NumPy."""
    xs = sorted(Fraction(v) for v in values)
    n = len(xs)
    rank = Fraction(p) / Fraction(100) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return float(xs[lo] + frac * (xs[hi] - xs[lo]))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def test_percentile_exact() -> None:
    _CURRENT[0] = "test_percentile_exact"
    rng = np.random.default_rng(20260812)
    for n in (7, 101, 1001):
        vals = rng.normal(0.0, 1.0, n).astype(np.float64)
        qb = implb.compute_q_b(vals, expected_count=n)
        ref = _ref_percentile_linear(np.abs(vals), 99.9)
        check(f"p99.9 linear matches exact reference at n={n}",
              abs(qb["q"] - ref) <= 1e-12 * max(1.0, abs(ref)),
              f"q={qb['q']!r} ref={ref!r}")
    hand = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    qb = implb.compute_q_b(hand, expected_count=5)
    # exact rank = (999/1000) * 4 = 999/250 = 3.996; float64 evaluation
    # lands at 3.9960000000000004, so the check is against the EXACT
    # value within float64 round-off, never a float literal equality.
    check("hand-computed p99.9 on arange(5)",
          abs(qb["q"] - float(Fraction(999, 250))) <= 1e-12,
          f"q={qb['q']!r}")
    check("method literal recorded", qb["method"] == "linear"
          and "method=\"linear\"" in qb["method_literal"])


def test_compute_q_b_gates() -> None:
    _CURRENT[0] = "test_compute_q_b_gates"
    vals = np.arange(10, dtype=np.float64) + 1.0
    expect_error("wrong count -> CORPUS_SIZE_MISMATCH",
                 "CORPUS_SIZE_MISMATCH", implb.compute_q_b, vals,
                 expected_count=11)
    bad = vals.copy()
    bad[3] = np.nan
    expect_error("NaN corpus -> CORPUS_NON_FINITE", "CORPUS_NON_FINITE",
                 implb.compute_q_b, bad, expected_count=10)
    expect_error("2-D corpus -> CORPUS_LAYOUT_UNEXPECTED",
                 "CORPUS_LAYOUT_UNEXPECTED", implb.compute_q_b,
                 vals.reshape(2, 5), expected_count=10)
    qb = implb.compute_q_b(vals, expected_count=10)
    check("B == margin * q (float64)",
          qb["B"] == float(np.float64(1.1) * np.float64(qb["q"])))
    d = qb["diagnostics"]
    check("diagnostics coherent",
          d["max_abs_u_scaled"] == float(np.abs(vals).max())
          and 0.0 <= d["fraction_abs_u_scaled_beyond_B"] <= 1.0
          and d["count_abs_u_scaled_beyond_B"]
          == int((np.abs(vals) > qb["B"]).sum()))
    zero = np.zeros(10, dtype=np.float64)
    expect_error("degenerate q -> Q_INVALID", "Q_INVALID",
                 implb.compute_q_b, zero, expected_count=10)


def test_packing_order_count() -> None:
    _CURRENT[0] = "test_packing_order_count"
    cmap = dec.build_coordinate_map([0, 1], 3, 4)   # free cols {2,3}
    check("synthetic dims", cmap.n_free_complex == 6
          and cmap.flow_dim_real == 12)
    re = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    im = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    packed = implb.pack_scalar_corpus(re, im)
    check("interleaved re/im per complex coordinate",
          packed.tolist() == [1.0, 10.0, 2.0, 20.0, 3.0, 30.0,
                              4.0, 40.0, 5.0, 50.0, 6.0, 60.0])
    check("pack count == flow_dim_real",
          packed.shape == (cmap.flow_dim_real,)
          and packed.dtype == np.float64)
    expect_error("mismatched shapes -> STATE_LAYOUT_UNEXPECTED",
                 "STATE_LAYOUT_UNEXPECTED", implb.pack_scalar_corpus,
                 re, im[:4])


def test_grid_keyed_gather() -> None:
    _CURRENT[0] = "test_grid_keyed_gather"
    # Two realisations whose packed orders differ; scales distinct per
    # physical (r, c), so an index-keyed gather would produce wrong
    # numbers and a shuffled location table must change nothing.
    cmap_a = dec.build_coordinate_map([0], 3, 4)    # free {1,2,3}
    cmap_b = dec.build_coordinate_map([1], 3, 4)    # free {0,2,3}
    locations = [{"row": r, "column": c,
                  "applied_mean_re": 0.1 * r, "applied_mean_im": -0.1 * c,
                  "applied_scale_re": 1.0 + r + 0.5 * c,
                  "applied_scale_im": 2.0 + 0.25 * r + c}
                 for r in range(3) for c in range(4)]
    idx_ordered = implb.build_location_index(locations)
    idx_shuffled = implb.build_location_index(locations[::-1])
    u_a = torch.tensor([complex(1.0 + k, 2.0 * k) for k in range(9)],
                       dtype=torch.complex64)
    re_o, im_o = implb.standardise_free(u_a, cmap_a, idx_ordered)
    re_s, im_s = implb.standardise_free(u_a, cmap_a, idx_shuffled)
    check("location-table order irrelevant",
          np.array_equal(re_o, re_s) and np.array_equal(im_o, im_s))
    exp_re = np.empty(9)
    for k in range(9):
        r, c = int(cmap_a.free_rows[k]), int(cmap_a.free_cols[k])
        exp_re[k] = ((1.0 + k) - 0.1 * r) / (1.0 + r + 0.5 * c)
    check("gather keyed by physical (r, c), exact",
          np.array_equal(re_o, exp_re),
          f"got {re_o[:3]}, want {exp_re[:3]}")
    # Same vector gathered through the OTHER realisation's map reads the
    # locations of THAT map's columns -- proof the key is (r, c), not k.
    u_b = torch.tensor([complex(5.0 - k, 1.0) for k in range(9)],
                       dtype=torch.complex64)
    re_b, _ = implb.standardise_free(u_b, cmap_b, idx_ordered)
    exp_b = np.empty(9)
    for k in range(9):
        r, c = int(cmap_b.free_rows[k]), int(cmap_b.free_cols[k])
        exp_b[k] = ((5.0 - k) - 0.1 * r) / (1.0 + r + 0.5 * c)
    check("second realisation reads its own columns",
          np.array_equal(re_b, exp_b))
    dup = locations + [locations[0]]
    expect_error("duplicate location -> PARENT_STRUCTURE_INVALID",
                 "PARENT_STRUCTURE_INVALID", implb.build_location_index,
                 dup)


def test_standardise_free() -> None:
    _CURRENT[0] = "test_standardise_free"
    cmap = dec.build_coordinate_map([0], 1, 3)      # free (0,1), (0,2)
    loc = implb.build_location_index([
        {"row": 0, "column": 1, "applied_mean_re": 1.0,
         "applied_mean_im": -1.0, "applied_scale_re": 2.0,
         "applied_scale_im": 4.0},
        {"row": 0, "column": 2, "applied_mean_re": 0.0,
         "applied_mean_im": 0.0, "applied_scale_re": 1.0,
         "applied_scale_im": 1.0}])
    u = torch.tensor([5.0 + 7.0j, 3.0 - 2.0j], dtype=torch.complex64)
    re_s, im_s = implb.standardise_free(u, cmap, loc)
    check("exact affine standardisation",
          re_s.tolist() == [2.0, 3.0] and im_s.tolist() == [2.0, -2.0])
    expect_error("wrong length -> STATE_LAYOUT_UNEXPECTED",
                 "STATE_LAYOUT_UNEXPECTED", implb.standardise_free,
                 torch.tensor([1.0 + 0.0j], dtype=torch.complex64),
                 cmap, loc)
    expect_error("NaN coefficient -> U_NON_FINITE", "U_NON_FINITE",
                 implb.standardise_free,
                 torch.tensor([np.nan + 0.0j, 1.0 + 0.0j],
                              dtype=torch.complex64), cmap, loc)
    expect_error("real tensor -> STATE_LAYOUT_UNEXPECTED",
                 "STATE_LAYOUT_UNEXPECTED", implb.standardise_free,
                 torch.tensor([1.0, 2.0]), cmap, loc)


def test_applied_pair_validity() -> None:
    _CURRENT[0] = "test_applied_pair_validity"
    base = {"row": 0, "column": 1, "applied_mean_re": 0.5,
            "applied_mean_im": -0.25, "applied_scale_re": 2.0,
            "applied_scale_im": 4.0}
    idx = implb.build_location_index([base])
    got = implb.applied_pair(idx, 0, 1)
    check("valid pair returns float64",
          got == (np.float64(0.5), np.float64(2.0), np.float64(-0.25),
                  np.float64(4.0))
          and all(isinstance(v, np.float64) for v in got))
    expect_error("missing location -> P4_LOCATION_MISSING",
                 "P4_LOCATION_MISSING", implb.applied_pair, idx, 9, 9)
    for field, value in (("applied_scale_re", 0.0),
                         ("applied_scale_im", -1.0),
                         ("applied_scale_re", np.nan),
                         ("applied_scale_im", np.inf),
                         ("applied_mean_re", np.nan),
                         ("applied_mean_im", np.inf)):
        forged = dict(base)
        forged[field] = value
        bad = implb.build_location_index([forged])
        expect_error(f"{field}={value!r} -> P4_APPLIED_PAIR_INVALID",
                     "P4_APPLIED_PAIR_INVALID",
                     implb.applied_pair, bad, 0, 1)


def test_binding_identity() -> None:
    _CURRENT[0] = "test_binding_identity"
    mask = fdm.make_cartesian_mask(96, 424242)
    binding, cmap = _binding_for(mask)
    row = _row_for(binding)
    got = implb.verify_binding_identity(row, binding, 96, 96)
    check("valid binding re-derives the map",
          got.n_free_complex == 6912 and got.flow_dim_real == 13824
          and got.payload()["map_payload_sha256"]
          == binding["map_sha256"])
    for field in ("dataset_index", "mask_seed", "file", "slice_index"):
        forged = dict(binding)
        forged[field] = ("zzz" if isinstance(binding[field], str)
                         else binding[field] + 1)
        expect_error(f"tampered {field} -> BINDING_IDENTITY_MISMATCH",
                     "BINDING_IDENTITY_MISMATCH",
                     implb.verify_binding_identity, row, forged, 96, 96)
    forged = dict(binding)
    cols = list(binding["acquired_columns"])
    forged["acquired_columns"] = sorted(
        set(cols) - {max(cols)} | {max(cols) + 1 if max(cols) < 95 else 0})
    expect_error("tampered columns -> MASK_LIVE_BINDING_MISMATCH",
                 "MASK_LIVE_BINDING_MISMATCH",
                 implb.verify_binding_identity, row, forged, 96, 96)
    forged = dict(binding)
    forged["mask_sha256"] = "0" * 64
    expect_error("tampered mask hash -> MASK_HASH_MISMATCH",
                 "MASK_HASH_MISMATCH",
                 implb.verify_binding_identity, row, forged, 96, 96)
    forged = dict(binding)
    forged["map_sha256"] = "0" * 64
    expect_error("tampered map hash -> MAP_HASH_MISMATCH",
                 "MAP_HASH_MISMATCH",
                 implb.verify_binding_identity, row, forged, 96, 96)
    forged = dict(binding)
    forged["n_free_complex"] = 6911
    expect_error("tampered dims -> BINDING_DIMENSION_MISMATCH",
                 "BINDING_DIMENSION_MISMATCH",
                 implb.verify_binding_identity, row, forged, 96, 96)
    no_centre = sorted(set(range(0, 72, 3)) - set(range(44, 52)))[:24]
    while len(no_centre) < 24:
        no_centre = sorted(set(no_centre) | {max(no_centre) + 1})
    forged = dict(binding)
    forged["acquired_columns"] = no_centre
    forged_row = _row_for(forged)
    expect_error("centre missing -> BINDING_CENTRE_NOT_ACQUIRED",
                 "BINDING_CENTRE_NOT_ACQUIRED",
                 implb.verify_binding_identity, forged_row, forged,
                 96, 96)
    short = no_centre[:23]
    forged = dict(binding)
    forged["acquired_columns"] = short
    forged_row = _row_for(forged)
    expect_error("23 columns -> BINDING_ACQUIRED_COUNT_UNEXPECTED",
                 "BINDING_ACQUIRED_COUNT_UNEXPECTED",
                 implb.verify_binding_identity, forged_row, forged,
                 96, 96)


def test_parent_loaders() -> None:
    _CURRENT[0] = "test_parent_loaders"
    p3_present = os.path.isfile(P3_ART)
    check("authoritative P3 artefact present in repo", p3_present,
          f"missing {P3_ART}" if not p3_present else "")
    p4_present = os.path.isfile(P4S2_ART)
    check("authoritative P4 /2 artefact present in repo", p4_present,
          f"missing {P4S2_ART}" if not p4_present else "")
    if p3_present:
        p3 = implb.load_p3_parent(P3_ART)
        check("P3 dual-hash pin passes on the real artefact", True)
        check("P3 bindings complete and dimensionally exact",
              len(p3["bindings"]) == 256
              and all(int(b["n_free_complex"]) == 6912
                      and int(b["flow_dim_real"]) == 13824
                      and len(b["acquired_columns"]) == 24
                      for b in p3["bindings"]))
    if p4_present:
        p4 = implb.load_p4s2_parent(P4S2_ART)
        check("P4 /2 dual-hash pin passes on the real artefact", True)
        check("P4 /2 branch PER-LOCATION with 8,448 locations",
              p4["branch"] == "PER-LOCATION"
              and len(p4["location_index"]) == 8448)
    expect_error("missing P3 parent -> PARENT_NOT_FOUND",
                 "PARENT_NOT_FOUND", implb.load_p3_parent,
                 P3_ART + ".absent")
    expect_error("missing P4 /2 parent -> PARENT_NOT_FOUND",
                 "PARENT_NOT_FOUND", implb.load_p4s2_parent,
                 P4S2_ART + ".absent")
    if p3_present:
        expect_error("P3 wrong file pin -> PARENT_FILE_HASH_MISMATCH",
                     "PARENT_FILE_HASH_MISMATCH", implb.load_p3_parent,
                     P3_ART, expected_file_sha="0" * 64)
        expect_error("P3 wrong semantic pin -> "
                     "PARENT_SEMANTIC_HASH_MISMATCH",
                     "PARENT_SEMANTIC_HASH_MISMATCH",
                     implb.load_p3_parent, P3_ART,
                     expected_semantic_sha="0" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            forged = os.path.join(tmp, "coordinate_map.json")
            with open(P3_ART, encoding="utf-8") as fh:
                payload = json.load(fh)
            payload["per_slice_bindings"][0]["n_free_complex"] = 6911
            with open(forged, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            expect_error("tampered P3 parent -> "
                         "PARENT_FILE_HASH_MISMATCH",
                         "PARENT_FILE_HASH_MISMATCH",
                         implb.load_p3_parent, forged)
    if p4_present:
        expect_error("P4 /2 wrong file pin -> PARENT_FILE_HASH_MISMATCH",
                     "PARENT_FILE_HASH_MISMATCH",
                     implb.load_p4s2_parent, P4S2_ART,
                     expected_file_sha="0" * 64)
        expect_error("P4 /2 wrong semantic pin -> "
                     "PARENT_SEMANTIC_HASH_MISMATCH",
                     "PARENT_SEMANTIC_HASH_MISMATCH",
                     implb.load_p4s2_parent, P4S2_ART,
                     expected_semantic_sha="0" * 64)


def test_parent_sidecars() -> None:
    _CURRENT[0] = "test_parent_sidecars"
    for art, loader, label in ((P3_ART, implb.load_p3_parent, "P3"),
                               (P4S2_ART, implb.load_p4s2_parent,
                                "P4 /2")):
        present = os.path.isfile(art)
        check(f"{label} artefact present for sidecar fixtures", present,
              f"missing {art}" if not present else "")
        if not present:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            lone = os.path.join(tmp, os.path.basename(art))
            with open(art, "rb") as fh:
                blob = fh.read()
            with open(lone, "wb") as fh:
                fh.write(blob)
            expect_error(f"{label} missing sidecar -> "
                         "PARENT_SIDECAR_MISSING",
                         "PARENT_SIDECAR_MISSING", loader, lone)
            with open(lone + ".sha256", "w", encoding="utf-8") as fh:
                fh.write("0" * 64 + "  " + os.path.basename(art) + "\n")
            expect_error(f"{label} wrong sidecar -> "
                         "PARENT_SIDECAR_MISMATCH",
                         "PARENT_SIDECAR_MISMATCH", loader, lone)


def test_generator_pin() -> None:
    _CURRENT[0] = "test_generator_pin"
    implb.enforce_generator_pin(
        {"resolved": True,
         "mask_seed_source_sha256": implb.GENERATOR_SOURCE_SHA256})
    check("registered generator hash passes", True)
    expect_error("wrong generator hash -> GENERATOR_HASH_MISMATCH",
                 "GENERATOR_HASH_MISMATCH", implb.enforce_generator_pin,
                 {"resolved": True, "mask_seed_source_sha256": "0" * 64})
    expect_error("unresolved provenance -> GENERATOR_HASH_MISMATCH",
                 "GENERATOR_HASH_MISMATCH", implb.enforce_generator_pin,
                 {"resolved": False})


def test_failure_boundary() -> None:
    """Drive main() with verified-parent stubs and an injected ordinary
    RuntimeError AFTER parent verification (P4S2T main()-integration
    precedent: module-attribute stubs, dataset-free). Proves the
    registered failure boundary: EXIT_ERROR, UNEXPECTED_RUNTIME_ERROR,
    a typed error record with verified sidecar, no facts artefact, no
    claim residue."""
    _CURRENT[0] = "test_failure_boundary"
    parents_stub = {"p0": {"facts_sha256": "0" * 64},
                    "p0s": {"facts_sha256": "0" * 64},
                    "subset_indices": [0], "subset_size": 1, "s_ref": 1.0}
    parent_stub = {"file_sha256": "0" * 64, "semantic_sha256": "0" * 64,
                   "sidecar_verified": True, "bindings": [],
                   "branch": "PER-LOCATION", "locations_order": None,
                   "location_index": {}, "parents_record": None}
    saved = {}
    for name in ("verify_parents", "load_p3_parent", "load_p4s2_parent",
                 "bind_mask_seed_provenance", "_collect"):
        saved[name] = getattr(implb, name)
    try:
        implb.verify_parents = lambda *a, **k: parents_stub
        implb.load_p3_parent = lambda path, **k: dict(parent_stub)
        implb.load_p4s2_parent = lambda path, **k: dict(parent_stub)
        implb.bind_mask_seed_provenance = lambda repo: {
            "resolved": True, "mask_seed_source_sha256":
            implb.GENERATOR_SOURCE_SHA256}
        def _boom(*a, **k):
            raise RuntimeError("injected ordinary failure")
        implb._collect = _boom
        with tempfile.TemporaryDirectory() as tmp:
            rc = implb.main(["--repo-dir", _REPO, "--data-root", "/unused",
                             "--p0-facts", "/unused",
                             "--p0s-facts", "/unused",
                             "--p0s-script", "/unused",
                             "--p3-facts", "/unused",
                             "--p4-stats2", "/unused",
                             "--out-dir", tmp])
            names = os.listdir(tmp)
            recs = [n for n in names
                    if n.startswith("implb_error") and n.endswith(".json")]
            payload = {}
            if recs:
                with open(os.path.join(tmp, recs[0]),
                          encoding="utf-8") as fh:
                    payload = json.load(fh)
            check("ordinary runtime fault -> EXIT_ERROR",
                  rc == implb.EXIT_ERROR)
            check("error code UNEXPECTED_RUNTIME_ERROR",
                  payload.get("error_code") == "UNEXPECTED_RUNTIME_ERROR")
            check("record is typed error, never stage facts",
                  payload.get("artefact_type") == "error"
                  and bool(payload.get("not_stage_facts")))
            check("no facts artefact published",
                  not any(n.startswith("implb_facts") for n in names))
            sidecar_ok = False
            if recs:
                try:
                    verify_sidecar(os.path.join(tmp, recs[0]))
                    sidecar_ok = True
                except (OSError, RuntimeError):
                    sidecar_ok = False
            check("error record sidecar verifies",
                  len(recs) == 1 and sidecar_ok)
            check("no claim residue",
                  not any(n.startswith(".implb") for n in names))
            check("raised_after_parent_verification recorded",
                  (payload.get("detail") or {})
                  .get("raised_after_parent_verification") is True)
    finally:
        for name, fn in saved.items():
            setattr(implb, name, fn)


def test_no_block_taxonomy() -> None:
    _CURRENT[0] = "test_no_block_taxonomy"
    with open(implb.__file__, encoding="utf-8") as fh:
        source = fh.read()
    check("no StageBlock anywhere in the stage module",
          "StageBlock" not in source)
    check("no BLOCK exit or verdict in the stage module",
          "EXIT_BLOCK" not in source and '"BLOCK"' not in source)


def test_publication() -> None:
    _CURRENT[0] = "test_publication"
    facts = {"schema": "seqref-implb-facts/1", "stage": "IMPL-B",
             "verdict": "PASS", "summary": {"q": 1.0},
             "run": {"utc": "2026-08-12T00:00:00+0000"}}
    with tempfile.TemporaryDirectory() as tmp:
        path, sha = implb.publish_implb(facts, tmp, "implb_facts")
        check("artefact written", os.path.isfile(path))
        check("sidecar verifies", file_sha256(path) == sha
              and os.path.isfile(path + ".sha256"))
        facts2 = dict(facts)
        facts2["summary"] = {"q": 2.0}
        facts2["run"] = {"utc": "2026-08-12T00:00:01+0000"}
        path2, _ = implb.publish_implb(facts2, tmp, "implb_facts")
        check("authoritative untouched by rerun",
              file_sha256(os.path.join(tmp, "implb_facts.json")) == sha)
        siblings = [n for n in os.listdir(tmp)
                    if n.startswith("implb_facts.")
                    and n.endswith(".json") and n != "implb_facts.json"]
        check("timestamped sibling written",
              len(siblings) == 1 and path2.endswith(siblings[0]))
        claim = os.path.join(tmp, ".implb_facts.claim")
        with open(claim, "w") as fh:
            fh.write("stage=IMPL-B pid=0 token=fixture utc=now\n")
        expect_error("concurrent claim refused", "PUBLICATION_CLAIM_HELD",
                     implb.publish_implb, facts, tmp, "implb_facts")
    with tempfile.TemporaryDirectory() as tmp2:
        with open(os.path.join(tmp2, "implb_facts.json.sha256"),
                  "w") as fh:
            fh.write("0" * 64 + "  implb_facts.json\n")
        try:
            implb.publish_implb(facts, tmp2, "implb_facts")
            check("orphan sidecar -> ERROR", False, "no error raised")
        except StageError as exc:
            check("orphan sidecar -> ERROR", True, exc.error_code)
        except RuntimeError as exc:
            check("orphan sidecar -> ERROR", "unpaired" in str(exc),
                  str(exc)[:80])
    with tempfile.TemporaryDirectory() as tmp3:
        with open(os.path.join(tmp3, "implb_facts.tmp999"), "w") as fh:
            fh.write("residue")
        expect_error("stale temporary -> STALE_TEMPORARY_FOUND",
                     "STALE_TEMPORARY_FOUND", implb.publish_implb,
                     facts, tmp3, "implb_facts")


EXPECTED_COUNTS = {
    "test_percentile_exact": 5,
    "test_compute_q_b_gates": 6,
    "test_packing_order_count": 4,
    "test_grid_keyed_gather": 4,
    "test_standardise_free": 4,
    "test_applied_pair_validity": 8,
    "test_binding_identity": 11,
    "test_parent_loaders": 13,
    "test_parent_sidecars": 6,
    "test_failure_boundary": 7,
    "test_generator_pin": 3,
    "test_no_block_taxonomy": 2,
    "test_publication": 7,
}

FIXTURES = [test_percentile_exact, test_compute_q_b_gates,
            test_packing_order_count, test_grid_keyed_gather,
            test_standardise_free, test_applied_pair_validity,
            test_binding_identity, test_parent_loaders,
            test_parent_sidecars, test_failure_boundary,
            test_generator_pin, test_no_block_taxonomy, test_publication]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=f"{SCRIPT_ID} {SCRIPT_VERSION}")
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args(argv)
    handlers = [logging.StreamHandler()]
    if os.path.realpath(args.repo_dir) != os.path.realpath(_REPO):
        # The P3/P4 artefact paths are derived from the installation
        # path (_REPO), so a mismatched --repo-dir would silently lie
        # about which repository is under test.
        print(f"ERROR: --repo-dir {args.repo_dir} does not resolve to "
              f"the installation repo {_REPO}", file=sys.stderr)
        return 2
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode="w"))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    for fx in FIXTURES:
        try:
            fx()
        except Exception as exc:  # noqa: BLE001 -- reported, not hidden
            logger.exception("fixture %s crashed", fx.__name__)
            _RESULTS.append((_CURRENT[0], "fixture completed without "
                                          "crash",
                             False, f"{type(exc).__name__}: {exc}"))
    per_fixture: dict[str, int] = {}
    failed = []
    for fixture, name, ok, detail in _RESULTS:
        per_fixture[fixture] = per_fixture.get(fixture, 0) + 1
        if not ok:
            failed.append({"fixture": fixture, "check": name,
                           "detail": detail})
    passed = sum(1 for r in _RESULTS if r[2])
    total = len(_RESULTS)
    coverage_ok = per_fixture == EXPECTED_COUNTS
    for fixture, count in sorted(per_fixture.items()):
        logger.info("coverage %-36s %d/%d", fixture, count,
                    EXPECTED_COUNTS.get(fixture, -1))
    if not coverage_ok:
        logger.error("COVERAGE MISMATCH: registry %s != actual %s -- "
                     "the registry is stale; re-derive it from THIS "
                     "source", EXPECTED_COUNTS, per_fixture)
    for f in failed:
        logger.error("FAIL %s::%s -- %s", f["fixture"], f["check"],
                     f["detail"])
    logger.info("%s %s: %d/%d checks passed (coverage_ok=%s)",
                SCRIPT_ID, SCRIPT_VERSION, passed, total, coverage_ok)
    print(json.dumps({"script": SCRIPT_ID, "version": SCRIPT_VERSION,
                      "passed": passed, "total": total,
                      "coverage_ok": coverage_ok,
                      "per_fixture": per_fixture,
                      "failed": failed}, indent=2))
    return 0 if (passed == total and coverage_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
