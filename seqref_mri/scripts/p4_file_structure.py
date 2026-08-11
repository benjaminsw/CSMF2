#!/usr/bin/env python3
# SEQREF-P4FS v0.2 -- training-split file structure check (index only)
# LIFETIME: DIAGNOSTIC
#
# CHANGELOG
# - v0.2 (2026-08-03): shortest_files no longer truncated to 8 -- A5 requires
#   ALL identities attaining the minimum. p_slice_miss renamed
#   p_column_acquired: it is the probability a non-centre column is ACQUIRED on
#   one slice, and the old name inverted that. The probability note now states
#   its approximation as bounded (deterministic SHA-derived masks TREATED as
#   independent draws for one column) rather than claiming independence.
#   Adds the Kish effective-units block over files. Version label corrected in
#   the argparse description and the JSON "meaning" field re-framed from
#   "worst-case ICC = 1" to the registered within-file extreme -- "worst-case"
#   reads as a complete dependence bound, which it is not.
# - v0.1 (2026-08-02): created. Its emitted log is SUPERSEDED and must not be
#   preserved as amendment evidence: the code was edited after that run without
#   a version bump, so the v0.1 log describes a file that no longer exists.
#
# Purpose
#   Test the FROZEN A5 prediction about distinct-file coverage by measuring the
#   slices-per-file distribution of the training split. A file fails to
#   contribute a free observation at column c only if EVERY slice of that file
#   has c acquired, which has probability p_miss = (n_random / n_noncentre)^s
#   for a file of s slices. That probability is governed by the MINIMUM slice
#   count, not the mean, and the minimum is not known in advance.
#
# ROLE, REGISTERED BEFORE THIS RUNS (A5 §4)
#   MAY     explain a failed prediction.
#   MAY NOT alter epsilon, kappa*, N_SCALE_MIN_FILES, the registered frame or
#           the coverage taxonomy. A short file RAISES THE PROBABILITY that the
#           prediction is wrong; it does NOT license changing the gate.
#   CLASS   structure evidence, NON-VERDICT.
#
# Reads the dataset INDEX only: HDF5 headers for slice counts, no k-space, no
# masks, no statistics. LIFETIME DIAGNOSTIC because it produces no artefact any
# later stage consumes -- its output is read once and recorded in the amendment.
#
# CONVENTION: logger.error + raise on every failure path. No fallback, no mock,
#   no placeholder, no silent pass.
#
# USAGE
#   python -m seqref_mri.scripts.p4_file_structure \
#       --data-root seqref_mri/data/fastmri

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

from seqref_mri.src.fastmri_data import (CELL_HW,  # noqa: E402
                                         FastMRISliceDataset, mask_counts)

logger = logging.getLogger("SEQREF-P4FS")

# Registered A5 values, quoted here for the derived probabilities ONLY. This
# script cannot change them and does not evaluate the gate.
N_SCALE_MIN_FILES = 900
EPSILON = 0.05
KAPPA_STAR = 10


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="SEQREF-P4FS v0.2 -- training file structure check")
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    # eval mode: the index is built identically in both modes and no mask is
    # drawn here. Using eval avoids the set_epoch() requirement for a check
    # that has nothing to do with masks.
    ds = FastMRISliceDataset(args.data_root, split="train", mode="eval")

    per_file: Counter[str] = Counter()
    for path, _ in ds.index:
        per_file[os.path.relpath(str(path), args.data_root)] += 1
    if not per_file:
        logger.error("no files found in the training split index")
        raise RuntimeError("empty training split index")

    counts = sorted(per_file.values())
    n_files = len(counts)
    n_slices = sum(counts)
    smin, smax = counts[0], counts[-1]
    smed = counts[n_files // 2] if n_files % 2 else \
        (counts[n_files // 2 - 1] + counts[n_files // 2]) / 2

    n_center, n_total = mask_counts(CELL_HW)
    n_random = n_total - n_center
    n_noncentre = CELL_HW - n_center
    p_column_acquired = n_random / n_noncentre   # a non-centre column is
    #                                            acquired on one slice

    def p_file_miss(s: float) -> float:
        return p_column_acquired ** s

    shortest = sorted(f for f, c in per_file.items() if c == smin)

    # KISH EFFECTIVE UNITS over files, at the REGISTERED ICC = 1 WITHIN-FILE
    # EXTREME with files treated as the cluster units. At that extreme every
    # slice of a file repeats one observation, so a SLICE-WEIGHTED estimator
    # behaves as a weighted mean of n_files units with weights proportional to
    # file size, and its effective sample size is Kish's (sum w)^2 / sum w^2.
    # This bounds WITHIN-file dependence only; it establishes nothing about
    # independence BETWEEN files.
    #
    # PROXY, NOT THE GATED QUANTITY. This uses TOTAL slices per file. The gate
    # applies per column over FREE slices, w_i(c). Kish is invariant to a
    # COMMON multiplier, which does not cover file-specific thinning, so the
    # per-column values are NOT guaranteed to match this figure.
    sum_s = float(sum(counts))
    sum_s2 = float(sum(c * c for c in counts))
    kish = (sum_s * sum_s) / sum_s2
    hist = dict(sorted(Counter(counts).items()))

    # Expected number of files missing a given column, under the diagnostic
    # approximation that a file's slice masks behave as independent draws for
    # that column. DIAGNOSTIC ONLY: it evaluates no gate and enters no verdict.
    expected_missing = sum(p_file_miss(c) for c in counts)

    out = {
        "script": {"id": "SEQREF-P4FS", "version": "v0.2",
                   "lifetime": "DIAGNOSTIC"},
        "class": "structure evidence, NON-VERDICT",
        "role_note": ("may EXPLAIN a failed prediction; may NOT alter "
                      "epsilon, kappa*, N_SCALE_MIN_FILES, the frame or the "
                      "taxonomy"),
        "split": "train", "data_root": os.path.abspath(args.data_root),
        "n_files": n_files, "n_slices": n_slices,
        "slices_per_file": {"min": smin, "median": smed, "max": smax,
                            "mean": n_slices / n_files},
        "histogram_slice_count_to_n_files": hist,
        "shortest_files": shortest,
        "n_files_at_minimum": len(shortest),
        "mask_structure": {
            "n_center": n_center, "n_total": n_total, "n_random": n_random,
            "n_noncentre": n_noncentre,
            "p_column_acquired_per_slice": p_column_acquired},
        "derived_miss_probabilities": {
            "at_min_slices": p_file_miss(smin),
            "at_median_slices": p_file_miss(smed),
            "at_mean_slices": p_file_miss(n_slices / n_files),
            "expected_files_missing_a_given_column": expected_missing,
            "note": ("computed under the DIAGNOSTIC APPROXIMATION that the "
                     "deterministic SHA-derived slice masks behave as "
                     "independent draws for a given column. The seeds are "
                     "distinct and deterministic; nothing here PROVES "
                     "probabilistic independence, and this figure does not "
                     "evaluate the eligibility gate.")},
        "kish_effective_units_over_files": {
            "value": kish,
            "n_files": n_files,
            "ratio_to_n_files": kish / n_files,
            "rule": "(sum s_i)^2 / sum s_i^2 over files",
            "meaning": ("effective independent units for a SLICE-WEIGHTED "
                        "estimator at the REGISTERED ICC = 1 WITHIN-FILE "
                        "EXTREME, with files treated as the cluster units; "
                        "<= n_files, equal only for equal file sizes. This "
                        "bounds WITHIN-file dependence only and establishes "
                        "nothing about dependence BETWEEN files."),
            "basis": "TOTAL slices per file",
            "is_the_gated_quantity": False,
            "note": ("PROXY ONLY. The gate applies per column over FREE "
                     "slices w_i(c); Kish is invariant to a COMMON "
                     "multiplier, which does not cover file-specific "
                     "thinning, so per-column values may differ and MUST be "
                     "computed by the census rather than substituted from "
                     "here. Whether this quantity gates anything at all "
                     "depends on the A5 estimator route, not settled here.")},
        "registered_values_quoted_not_set_here": {
            "N_SCALE_MIN_FILES": N_SCALE_MIN_FILES, "epsilon": EPSILON,
            "kappa_star": KAPPA_STAR,
            "files_available": n_files,
            "gate_as_fraction_of_population": N_SCALE_MIN_FILES / n_files,
            "max_files_that_may_miss_a_column":
                n_files - N_SCALE_MIN_FILES},
        "frozen_prediction": {
            "never_free": "columns 44-51",
            "eligible": "the other 88 columns",
            "under_supported": "EMPTY",
            "status": "NOT evaluated here -- the coverage census evaluates it"},
    }
    logger.info("files=%d slices=%d  slices/file min=%d median=%s max=%d",
                n_files, n_slices, smin, smed, smax)
    logger.info("p(file misses a column) at min=%.3e  at median=%.3e",
                p_file_miss(smin), p_file_miss(smed))
    logger.info("a column may be missed by at most %d files before the gate "
                "fails (%d available, gate %d)",
                n_files - N_SCALE_MIN_FILES, n_files, N_SCALE_MIN_FILES)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
