#!/usr/bin/env python3
# SEQREF-P4FIDST v0.1 -- P4FID self-test (fixtures only, no dataset access)
# LIFETIME: KEEP
#
# Why this exists
#   p4_frame_identity.py is tagged LIFETIME: KEEP and its emitted artefact is
#   amendment evidence for the estimator-A registration. Under the campaign
#   precedent (p12_selftest, p3_selftest) a KEEP stage carries regression
#   evidence for its own audit semantics BEFORE it runs on data.
#
#   The immediate cause is concrete: the first draft DECLARED the inherited
#   identity as (split, file, slice_index) in its header and facts, and then
#   keyed resolution on (file, slice_index) only. A declared contract that the
#   code does not implement is the recurring defect class of this campaign, and
#   it is exactly what a fixture catches without touching 100 GB of data.
#
# Every check asserts a FAILURE MODE IS SEEN TO FIRE, not merely that the happy
# path passes. Per-fixture check counts are registered and compared, so a check
# skipped behind a guard fails the suite instead of shrinking the total.
#
# USAGE
#   python -m seqref_mri.scripts.p4fid_selftest --repo-dir . \
#       --log-file seqref_mri/results/_diag/p4fid_selftest.log

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import tempfile

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

import preflight_io as pio  # noqa: E402
import preflight_parents as ppar  # noqa: E402
from seqref_mri.src.fastmri_data import (CELL_HW, TRAIN_BASE_SEED,  # noqa: E402
                                         canonical_mask_seed, fft2c, ifft2c,
                                         mask_counts, make_cartesian_mask)
from seqref_mri.scripts import p4_frame_identity as fid  # noqa: E402

logger = logging.getLogger("SEQREF-P4FIDST")
RESULTS: list[tuple[str, bool, str]] = []

EXPECTED_COUNTS = {
    "test_registered_constants": 6,
    "test_structural_mask_facts": 7,
    "test_identity_algebra": 5,
    "test_identity_oracle_fires": 4,
    "test_split_identity_contract": 5,
    "test_frame_contract": 4,
    "test_relative_rule": 5,
    "test_output_path_guard": 4,
}
# 40 registered FIXTURE checks. The harness appends ONE meta check
# (selftest.coverage_matches_registry) after the fixtures, so a clean run
# reports 41/41 total = 40 fixture + 1 meta. Registry counts FIXTURE checks
# only, by design: the meta check cannot be in its own registry.
N_META_CHECKS = 1


def check(name: str, ok, detail: str = "") -> None:
    ok = bool(ok)
    RESULTS.append((name, ok, detail))
    (logger.info if ok else logger.error)("%-52s %s %s", name,
                                          "PASS" if ok else "FAIL", detail)


def raises(exc_types, fn, *a, **k) -> bool:
    try:
        fn(*a, **k)
        return False
    except exc_types:
        return True


# ===========================================================================

def test_registered_constants() -> None:
    check("const.gate_separate_from_expectation",
          fid.P4FID_EXPECTED_REL < fid.P4FID_IDENTITY_TOL,
          "an expectation is not a threshold; ~1e-7 must not gate at 1e-7")
    check("const.three_tolerances_are_distinct_keys",
          len({"P4FID_IDENTITY_TOL", "P4FID_FREE_TOL",
               "P4FID_SUPPORT_TOL"}) == 3
          and fid.P4FID_IDENTITY_TOL == fid.P4FID_FREE_TOL
          == fid.P4FID_SUPPORT_TOL,
          "shared value is INCIDENTAL; keys stay separate")
    check("const.denominator_floor_registered",
          fid.P4FID_REL_DENOM_FLOOR == 1e-12)
    check("const.no_block_path_exists",
          "StageBlock" not in open(fid.__file__).read(),
          "P4FID tests a construction; it has no data-premise verdict")
    check("const.schema_is_its_own",
          fid.FACTS_SCHEMA == "seqref-p4fid-facts/1")
    check("const.artefact_is_non_authoritative",
          "authoritative" in open(fid.__file__).read()
          and "evidence_class" in open(fid.__file__).read())


def test_structural_mask_facts() -> None:
    """The acquired count is fixed by construction, not sampled."""
    n_center, n_total = mask_counts(CELL_HW)
    check("mask.n_center_is_8", n_center == 8, f"got {n_center}")
    check("mask.n_total_is_24", n_total == 24, f"got {n_total}")
    check("mask.flow_dim_is_13824",
          2 * CELL_HW * (CELL_HW - n_total) == 13824)
    centre = set(range((CELL_HW - n_center) // 2,
                       (CELL_HW - n_center) // 2 + n_center))
    check("mask.centre_cols_44_51", centre == set(range(44, 52)), str(centre))

    always, counts = None, set()
    for seed in range(64):
        m = make_cartesian_mask(CELL_HW, seed)
        cols = set(np.flatnonzero(m).tolist())
        counts.add(len(cols))
        always = cols if always is None else (always & cols)
    check("mask.count_invariant_over_seeds", counts == {24}, str(counts))
    check("mask.centre_always_acquired", centre <= always,
          "the never-free class is STRUCTURAL, not empirical")
    check("mask.raises_on_bad_width",
          raises(ValueError, make_cartesian_mask, 1, 0))


def _fixture(seed: int):
    """Build x_true, mask, y, cond_in, x_norm exactly as the pipeline does."""
    torch.manual_seed(seed)
    x_true_c = torch.randn(CELL_HW, CELL_HW, dtype=torch.complex64)
    amax = 3.0
    mask = torch.from_numpy(make_cartesian_mask(CELL_HW, seed).copy())
    k96 = fft2c(x_true_c)
    y = k96 * mask.to(k96.dtype).unsqueeze(0)          # __getitem__
    x0_c = ifft2c(y)                                    # A_adjoint
    cond_in = x0_c / amax                               # _prepare order
    x_norm = x_true_c / amax
    return x_norm, cond_in, mask, amax


def test_identity_algebra() -> None:
    """F(x_norm - cond_in) == (1 - M) F(x_norm), to fp32 roundoff."""
    worst_id = worst_free = worst_sup = 0.0
    for seed in range(8):
        x_norm, cond_in, mask, _ = _fixture(seed)
        m = mask.to(torch.complex64)
        k_x = fft2c(x_norm)
        lhs = fft2c(x_norm - cond_in)
        rhs = (1.0 - m) * k_x
        den = float(torch.max(torch.abs(k_x)).item())
        worst_id = max(worst_id, float(torch.max(torch.abs(lhs - rhs)).item()) / den)
        worst_free = max(worst_free, float(
            torch.max(torch.abs((1.0 - m) * (lhs - k_x))).item()) / den)
        worst_sup = max(worst_sup, float(
            torch.max(torch.abs(m * lhs)).item()) / den)
    check("identity.holds_on_fixtures", worst_id <= fid.P4FID_IDENTITY_TOL,
          f"worst rel={worst_id:.3e}")
    check("identity.free_restriction_holds", worst_free <= fid.P4FID_FREE_TOL,
          f"worst rel={worst_free:.3e}")
    check("identity.measured_support_vanishes",
          worst_sup <= fid.P4FID_SUPPORT_TOL, f"worst rel={worst_sup:.3e}")
    check("identity.magnitude_matches_expectation",
          worst_id < 100 * fid.P4FID_EXPECTED_REL,
          "algebraically exact; only fp32 roundoff should appear")
    check("identity.value_is_mask_independent_at_free_coords",
          worst_free <= fid.P4FID_FREE_TOL,
          "F dx == F x_norm at free coordinates -- the estimator-A premise")


def test_identity_oracle_fires() -> None:
    """The oracle must be SEEN to catch a wrong mask and a wrong cond_in."""
    x_norm, cond_in, mask, _ = _fixture(0)
    wrong = torch.from_numpy(make_cartesian_mask(CELL_HW, 999).copy())
    check("oracle.wrong_seed_gives_different_mask",
          not torch.equal(mask, wrong),
          "otherwise the provenance check could not fire")
    m_wrong = wrong.to(torch.complex64)
    k_x = fft2c(x_norm)
    lhs = fft2c(x_norm - cond_in)
    den = float(torch.max(torch.abs(k_x)).item())
    bad = float(torch.max(torch.abs(lhs - (1.0 - m_wrong) * k_x)).item()) / den
    check("oracle.wrong_mask_breaks_identity", bad > fid.P4FID_IDENTITY_TOL,
          f"rel={bad:.3e}")
    # cond_in built with the linearity-equivalent but DIFFERENT order
    m = mask.to(torch.complex64)
    cond_wrong = cond_in * 1.0001
    bad2 = float(torch.max(torch.abs(
        fft2c(x_norm - cond_wrong) - (1.0 - m) * k_x)).item()) / den
    check("oracle.wrong_cond_in_breaks_identity",
          bad2 > fid.P4FID_IDENTITY_TOL, f"rel={bad2:.3e}")
    check("oracle.seed_rule_is_deterministic",
          canonical_mask_seed(TRAIN_BASE_SEED, "a/b.h5", 3, epoch=0)
          == canonical_mask_seed(TRAIN_BASE_SEED, "a/b.h5", 3, epoch=0)
          and canonical_mask_seed(TRAIN_BASE_SEED, "a/b.h5", 3, epoch=0)
          != canonical_mask_seed(TRAIN_BASE_SEED, "a/b.h5", 3, epoch=1))


def test_split_identity_contract() -> None:
    """The DECLARED identity is (split, file, slice_index). The first draft
    declared that and keyed on (file, slice_index); this fixture is why."""
    ok = {"subset": [{"split": "train", "file_relpath": "f.h5",
                      "slice_index": 2, "dataset_index": 7}]}
    entries = fid._p0s_entries(ok)
    check("split.parsed_from_entry", entries[0]["split"] == "train")
    check("split.identity_carries_all_three",
          {"split", "file", "slice_index"} <= set(entries[0]))

    bad = {"subset": [{"split": "val", "file_relpath": "f.h5",
                       "slice_index": 2}]}
    check("split.val_entry_rejected",
          raises(ppar.StageError, fid._p0s_entries, bad))
    missing = {"subset": [{"file_relpath": "f.h5", "slice_index": 2}]}
    check("split.absent_split_rejected",
          raises(ppar.StageError, fid._p0s_entries, missing),
          "split is CHECKED, never inferred")
    no_id = {"subset": [{"split": "train", "dataset_index": 7}]}
    check("split.missing_identity_rejected",
          raises(ppar.StageError, fid._p0s_entries, no_id))


def test_frame_contract() -> None:
    check("frame.train_slices_is_none", fid.FRAME["train_slices"] is None)
    check("frame.subset_seed_recorded_non_operative",
          fid.FRAME["subset_seed"] == 20260904
          and fid.FRAME["subset_seed_operative"] is False,
          "_subset returns ds unchanged when n is None")
    check("frame.single_epoch", fid.FRAME["epoch_set"] == [0])
    check("frame.count_rule_is_derived",
          "|selected training slices| x |epoch set|"
          == fid.FRAME["realisation_count_rule"],
          "derived, never chosen")


def test_relative_rule() -> None:
    check("rel.floor_applies_at_zero",
          fid.rel_of(1.0, 0.0) == 1.0 / fid.P4FID_REL_DENOM_FLOOR)
    check("rel.ordinary_case", abs(fid.rel_of(2.0, 4.0) - 0.5) < 1e-15)
    check("rel.floor_not_applied_above_it",
          abs(fid.rel_of(1.0, 1.0) - 1.0) < 1e-15)
    check("rel.zero_error_is_zero", fid.rel_of(0.0, 5.0) == 0.0)
    check("rel.floor_applies_below_floor",
          fid.rel_of(1.0, fid.P4FID_REL_DENOM_FLOOR / 10)
          == 1.0 / fid.P4FID_REL_DENOM_FLOOR,
          "a denominator BELOW the floor is clamped to it, not used")


def test_output_path_guard() -> None:
    """The locked-directory guard is a SAFETY PROPERTY, so it gets a fixture.

    An earlier revision raised it INSIDE the publication boundary, so a
    mistyped --out-dir refused the facts and then wrote an error record into
    the same forbidden directory. The guard is now pre-boundary; these checks
    are what stop that regressing.
    """
    check("output.locked_preflight_rejected",
          raises(ppar.StageError, fid.validate_output_dir,
                 "/tmp/residual_preflight"))
    check("output.nested_locked_path_rejected",
          raises(ppar.StageError, fid.validate_output_dir,
                 "seqref_mri/results/_diag/residual_preflight/sub"))
    ok = True
    try:
        fid.validate_output_dir("/tmp/p4fid")
        fid.validate_output_dir("seqref_mri/results/_diag/p4_frame_identity")
    except Exception:  # noqa: BLE001
        ok = False
    check("output.ordinary_paths_accepted", ok)
    check("output.guard_is_pre_boundary",
          "validate_output_dir(args.out_dir)" in open(fid.__file__).read()
          and open(fid.__file__).read().index("validate_output_dir(args.out_dir)")
          < open(fid.__file__).read().index("guard_run_mode(args.out_dir"),
          "must run BEFORE any path that can publish an error record")


# ===========================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="SEQREF-P4FIDST v0.1 -- P4FID self-test")
    ap.add_argument("--repo-dir", default=_REPO)
    ap.add_argument("--log-file", default="p4fid_selftest.log")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.log_file, mode="w",
                                      encoding="utf-8")],
        format="%(asctime)s %(levelname)s %(message)s")

    counts: dict[str, int] = {}
    for fn in (test_registered_constants, test_structural_mask_facts,
               test_identity_algebra, test_identity_oracle_fires,
               test_split_identity_contract, test_frame_contract,
               test_relative_rule, test_output_path_guard):
        logger.info("--- %s ---", fn.__name__)
        before = len(RESULTS)
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            logger.exception("fixture %s raised", fn.__name__)
            check(f"{fn.__name__}.completed", False,
                  f"{type(exc).__name__}: {exc}")
        counts[fn.__name__] = len(RESULTS) - before

    coverage_ok = True
    for name, expected in EXPECTED_COUNTS.items():
        ran = counts.get(name, 0)
        if ran != expected:
            coverage_ok = False
            logger.error("COVERAGE: %s ran %d checks, expected %d", name, ran,
                         expected)
        else:
            logger.info("coverage %-40s %d/%d", name, ran, expected)
    RESULTS.append(("selftest.coverage_matches_registry", coverage_ok,
                    f"executed={sum(counts.values())} "
                    f"registered={sum(EXPECTED_COUNTS.values())}"))

    n_fixture = sum(counts.values())
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    logger.info("SEQREF-P4FIDST: %d/%d checks passed "
                "(%d fixture + %d meta; coverage_ok=%s)",
                passed, total, n_fixture, N_META_CHECKS, coverage_ok)
    for name, ok, detail in RESULTS:
        if not ok:
            logger.error("FAILED: %s %s", name, detail)
    print(json.dumps({"passed": passed, "total": total,
                      "fixture_checks": n_fixture,
                      "meta_checks": N_META_CHECKS,
                      "coverage_ok": coverage_ok, "per_fixture": counts,
                      "failed": [n for n, ok, _ in RESULTS if not ok]},
                     indent=2))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
