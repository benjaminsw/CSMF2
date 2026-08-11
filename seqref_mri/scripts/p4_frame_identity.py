#!/usr/bin/env python3
# SEQREF-P4FID v0.1 -- P4 frame identity + mask-provenance diagnostic
# LIFETIME: KEEP
#
# Purpose
#   Establish, by measurement, the algebraic identity the estimator-A decision
#   rests on, under the REGISTERED TRAINING-MASK FRAME:
#
#       F(x_norm - cond_in)  ==  (1 - M) . F(x_norm)
#
#   and, restricted to the free coordinates of that sample's mask,
#
#       F(x_norm - cond_in)  ==  F(x_norm).
#
#   If the identity holds, the value of a modelled coefficient at a free
#   location does not depend on WHICH other columns the mask selected. That
#   licenses concept D4's free-conditioned statistics (estimator A) to be
#   accumulated over the registered deterministic frame WITHOUT an additional
#   MASK-RESAMPLING SCHEDULE. Each frame member's deterministic mask STILL
#   defines which locations contribute: the sample-specific inclusion rule
#   remains, and only a separate resampling schedule disappears. If the
#   identity FAILS, the P4 sampling contract changes and N_SCALE_MIN must not
#   be derived against this frame.
#
# What this does NOT establish
#   * It does NOT choose between estimator A (free-conditioned, concept D4)
#     and estimator B (unconditional). The identity says the VALUE is
#     mask-independent; it says nothing about which SAMPLING POPULATION D4
#     intends. A remains the approved estimator; B would require a concept
#     amendment.
#   * It does NOT verify the FFT convention. fft2c appears on BOTH sides of
#     the identity, so a wrong convention CANCELS and is invisible here. The
#     independence this diagnostic actually provides is over the MASK and the
#     cond_in CONSTRUCTION, not the transform.
#   * It makes NO claim beyond the 256 registered slice identities at epoch 0.
#
# Independence structure (the point of the oracle)
#   PRODUCTION  x_norm, cond_in via the live train_base._prepare.
#   ORACLE      the mask is REGENERATED from the registered seed rule
#               (canonical_mask_seed -> make_cartesian_mask) and required to
#               equal batch["mask"] EXACTLY; the right-hand side is built from
#               that regenerated mask, NOT from _prepare's returned operator
#               objects. Otherwise the check would prove _prepare consistent
#               with itself.
#
# Run-mode terminology (resolved before the full run)
#   guard_run_mode(out_dir, smoke) is a BINARY OUTPUT-ISOLATION guard: it keeps
#   smoke_* artefacts out of a directory holding non-smoke ones and the
#   reverse. Its "authoritative" branch means NON-SMOKE OUTPUT ISOLATION and
#   NOT scientific authority. P4FID's non-smoke run takes that branch while
#   remaining a NON-AUTHORITATIVE diagnostic: facts carry
#   run_mode = "diagnostic", authoritative = false, and an evidence_class
#   stating what the artefact is evidence FOR.
#   To keep the distinction structural rather than merely documented, P4FID
#   REFUSES the locked preflight artefact directory BEFORE entering any
#   publication path, so neither facts nor an error record can land beside the
#   authoritative P0-P4 records.

# Verdict semantics
#   PASS   identity, free-restriction, measured-support and mask provenance
#          all within registered tolerance; facts published; exit 0.
#   ERROR  parent/identity mismatch, mask regeneration mismatch, or any
#          registered tolerance exceeded. NO scientific BLOCK verdict is
#          available in this stage: it tests a CONSTRUCTION, not a premise
#          about the data population.
#
# CONVENTION: logger.error + raise on every failure path. No fallback, no
#   mock, no placeholder, no silent pass.
#
# USAGE
#   python -m seqref_mri.scripts.p4_frame_identity \
#     --repo-dir . --data-root seqref_mri/data/fastmri \
#     --p0-facts   .../p0_facts.json \
#     --p0s-facts  .../normalisation_scale_facts.json \
#     --p0s-script seqref_mri/scripts/_diag/p0s_normalisation_scale.py \
#     --out-dir seqref_mri/results/_diag/p4_frame_identity

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

from preflight_io import canonical_hash, file_sha256  # noqa: E402
from preflight_parents import (EXIT_ERROR, EXIT_PASS,  # noqa: E402
                               StageError, attach_semantic_hash,
                               environment_record, guard_run_mode,
                               hash_project_code, publish_error, publish_stage,
                               require_finite, verify_parents)
from seqref_mri.src.fastmri_data import (CELL_HW, TRAIN_BASE_SEED,  # noqa: E402
                                         FastMRISliceDataset,
                                         canonical_mask_seed, fft2c,
                                         make_cartesian_mask)
from seqref_mri.scripts.train_base import _collate, _prepare  # noqa: E402

SCRIPT_ID = "SEQREF-P4FID"
SCRIPT_VERSION = "v0.1"
FACTS_SCHEMA = "seqref-p4fid-facts/1"
FACTS_PREFIX = "frame_identity"
ERROR_PREFIX = "frame_identity_error"
SMOKE_FACTS_PREFIX = "smoke_frame_identity"
SMOKE_ERROR_PREFIX = "smoke_frame_identity_error"

# --------------------------------------------------------------------------
# REGISTERED FRAME (A4). Deterministic, not sampled: every mask realisation
# follows from the seed rule below, so the realisation count is DERIVED as
# |training slices| x |epoch set|, never chosen.
# --------------------------------------------------------------------------
FRAME = {
    "training_population": "full training split",
    "train_slices": None,
    "subset_seed": 20260904,
    "subset_seed_operative": False,     # _subset returns ds unchanged when
                                        # n is None, so the seed does not bind
    "epoch_set": [0],
    "mask_generator": "seqref_mri.src.fastmri_data (SEQREF-I1 v0.3)",
    "base_seed": TRAIN_BASE_SEED,
    "seed_tuple": "base_seed|epoch|relpath|slice_index",
    "mask_mode": "train",
    "realisation_count_rule": "|selected training slices| x |epoch set|",
    "structural_note": (
        "acquired count is STRUCTURAL, not sampled: mask_counts(96) fixes "
        "n_total = 24 and make_cartesian_mask raises if the realised count "
        "differs, so count invariance is guaranteed by construction and this "
        "frame CORROBORATES it rather than establishing it"),
}

# --------------------------------------------------------------------------
# REGISTERED CONSTANTS. Separate keys even where values coincide; the shared
# 1e-5 is INCIDENTAL and the three tolerances are never interchangeable.
# --------------------------------------------------------------------------
P4FID_EXPECTED_REL     = 1e-7    # NON-GATING expectation, not a threshold.
# The identity is algebraically EXACT given cond_in = F^H(M y)/a and
# y = M F x_true (mask idempotent). What is measured is fp32 FFT round-trip
# error alone. Registered so a passing ~1e-7 reads as the expected result and
# not as a near-miss.
P4FID_IDENTITY_TOL     = 1e-5    # ERROR: identity_rel_max
P4FID_FREE_TOL         = 1e-5    # ERROR: free_identity_rel_max
P4FID_SUPPORT_TOL      = 1e-5    # ERROR: measured_support_rel_max
P4FID_REL_DENOM_FLOOR  = 1e-12   # zero-safe relative denominator floor

# NOT COMPARABLE TO P2. P2's relative_max = 6.90977e-6 normalised by
# max|F dx|; this diagnostic normalises by max|F x_norm|, which is larger.
# Same underlying quantity, different denominator, different magnitude
# (~1e-7 here vs ~7e-6 there). Recorded so the difference is not read as a
# discrepancy. P2's masks were EVAL-mode; these are TRAIN-mode at epoch 0.
P2_RELATIVE_MAX_NOT_COMPARABLE = True

EXPECTED_SUBSET_SIZE = 256

# The locked EXEC §9 artefact directory. P4FID is a NON-AUTHORITATIVE
# diagnostic and must never publish here, whatever guard_run_mode's binary
# branch is called.
LOCKED_PREFLIGHT_DIRNAME = "residual_preflight"

P4FID_CODE_FILES = ["seqref_mri/scripts/p4_frame_identity.py"]

logger = logging.getLogger(SCRIPT_ID)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def rel_of(abs_err: float, denom: float) -> float:
    """Zero-safe relative rule, floored. Registered, not improvised."""
    return abs_err / max(denom, P4FID_REL_DENOM_FLOOR)


def two_channel_to_complex(t: torch.Tensor) -> torch.Tensor:
    if t.dim() != 3 or t.shape[0] != 2:
        raise StageError("STATE_LAYOUT_UNEXPECTED",
                         f"expected (2, H, W) real state, got {tuple(t.shape)}")
    return torch.complex(t[0], t[1])


def validate_output_dir(out_dir: str) -> None:
    """Refuse the locked EXEC §9 preflight directory.

    Called BEFORE any code path that can publish, including the error path.
    An earlier revision raised this inside the publication boundary, so a
    mistyped --out-dir refused the facts but still wrote an ERROR RECORD into
    the locked directory -- violating the very guarantee the guard states.
    """
    out_abs = os.path.abspath(out_dir)
    if LOCKED_PREFLIGHT_DIRNAME in out_abs.split(os.sep):
        raise StageError(
            "DIAGNOSTIC_INTO_LOCKED_DIR",
            f"refusing to write ANY artefact -- facts or error record -- into "
            f"the locked EXEC §9 preflight directory ({out_abs}). "
            f"guard_run_mode's non-smoke branch means output isolation, not "
            f"scientific authority; this stage is a NON-AUTHORITATIVE "
            f"diagnostic and must not sit beside the authoritative P0-P4 "
            f"records.",
            write_record=False)


def _peak_rss_bytes() -> int:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru * 1024) if sys.platform.startswith("linux") else int(ru)


# --------------------------------------------------------------------------
# Sample inheritance -- IDENTITIES, not bare indices
# --------------------------------------------------------------------------

def _p0s_entries(p0s_facts: dict) -> list[dict]:
    """Read the frozen subset as (file, slice_index) IDENTITIES.

    A bare dataset index is NOT portable provenance: it is meaningful only
    against a specific split and a specific index construction. The stable
    identity is (split, relpath, slice_index). The recorded index is retained
    and CROSS-CHECKED, never trusted alone.
    """
    for key in ("subset", "entries", "subset_entries", "slices"):
        block = p0s_facts.get(key)
        if isinstance(block, list) and block and isinstance(block[0], dict):
            entries = block
            break
    else:
        raise StageError("P0S_SUBSET_ENTRIES_MISSING",
                         f"P0S facts carry no per-entry subset list; "
                         f"present keys: {sorted(p0s_facts)}")
    out = []
    for i, e in enumerate(entries):
        rel = next((e[k] for k in ("file_relpath", "file", "relpath", "path")
                    if k in e), None)
        sl = next((e[k] for k in ("slice_index", "slice", "slice_idx")
                   if k in e), None)
        idx = next((e[k] for k in ("dataset_index", "index", "idx")
                    if k in e), None)
        if rel is None or sl is None:
            raise StageError("P0S_ENTRY_IDENTITY_MISSING",
                             f"P0S entry {i} lacks a file/slice identity; "
                             f"present keys: {sorted(e)}")
        # SPLIT IS PART OF THE IDENTITY AND IS CHECKED, NOT INFERRED. P0S used
        # the train split in EVAL mode; that it did so must be verified from
        # the record, because an identity resolved against the wrong split
        # would silently name a different slice.
        split = e.get("split")
        if split is None:
            split = p0s_facts.get("dataset", {}).get("split") \
                if isinstance(p0s_facts.get("dataset"), dict) else None
        if split != "train":
            raise StageError(
                "P0S_ENTRY_SPLIT_MISMATCH",
                f"P0S entry {i} has split={split!r}; expected 'train'. The "
                f"split is part of the inherited identity and is not assumed.")
        out.append({"position": i, "split": split, "file": str(rel),
                    "slice_index": int(sl),
                    "recorded_dataset_index":
                        None if idx is None else int(idx)})
    return out


def resolve_identities(ds: FastMRISliceDataset, entries: list[dict],
                       data_root: str) -> list[dict]:
    """Resolve each inherited identity to a UNIQUE dataset index in THIS
    dataset, and cross-check the recorded index where P0S carried one."""
    if ds.split != "train":
        raise StageError("DATASET_SPLIT_MISMATCH",
                         f"dataset split is {ds.split!r}; the inherited "
                         f"identities are 'train'")
    lookup: dict[tuple[str, str, int], list[int]] = {}
    for i, (path, sl) in enumerate(ds.index):
        rel = os.path.relpath(str(path), data_root).replace(os.sep, "/")
        lookup.setdefault((ds.split, rel, int(sl)), []).append(i)

    resolved, mismatches = [], []
    for e in entries:
        key = (e["split"], e["file"], e["slice_index"])
        hits = lookup.get(key, [])
        if len(hits) != 1:
            raise StageError(
                "IDENTITY_RESOLUTION_FAILED",
                f"identity {key} resolved to {len(hits)} dataset entries; "
                f"exactly one is required")
        idx = hits[0]
        rec = e["recorded_dataset_index"]
        if rec is not None and rec != idx:
            mismatches.append({"split": e["split"], "file": e["file"],
                               "slice_index": e["slice_index"],
                               "recorded": rec, "resolved": idx})
        resolved.append({**e, "dataset_index": idx,
                         "index_agrees": rec is None or rec == idx})
    if mismatches:
        logger.error("recorded dataset indices disagree with resolution on "
                     "%d identities: %s", len(mismatches), mismatches[:4])
        raise StageError(
            "IDENTITY_INDEX_MISMATCH",
            f"{len(mismatches)} inherited identities resolve to a different "
            f"dataset index than P0S recorded; the index is not portable and "
            f"the identity resolution disagrees with it")
    logger.info("resolved %d inherited identities uniquely; recorded indices "
                "agree", len(resolved))
    return resolved


# --------------------------------------------------------------------------
# Per-slice measurement
# --------------------------------------------------------------------------

def measure(ident: dict, batch: dict, prep: dict, epoch: int) -> dict:
    """One slice. PRODUCTION x_norm / cond_in; ORACLE mask and RHS."""
    x_norm = two_channel_to_complex(prep["x_norm"][0])
    cond_in = two_channel_to_complex(prep["cond_in"][0])

    # ---- ORACLE: regenerate the mask from the registered seed rule.
    # Taken from fastmri_data directly, NOT from prep["ops"], so the check
    # is not _prepare compared against itself.
    seed = canonical_mask_seed(TRAIN_BASE_SEED, ident["file"],
                               ident["slice_index"], epoch=epoch)
    mask_oracle = torch.from_numpy(
        make_cartesian_mask(CELL_HW, seed).copy())          # (W,) bool
    mask_batch = batch["mask"][0].to(torch.bool)
    mask_equal = bool(torch.equal(mask_oracle, mask_batch))
    meta_seed = int(batch["meta"][0]["mask_seed"])
    seed_equal = meta_seed == seed
    if not mask_equal or not seed_equal:
        logger.error("mask provenance mismatch on %s slice %d: "
                     "mask_equal=%s seed_equal=%s (oracle %d, meta %d)",
                     ident["file"], ident["slice_index"], mask_equal,
                     seed_equal, seed, meta_seed)
        raise StageError(
            "MASK_REGENERATION_MISMATCH",
            f"the epoch-{epoch} train mask regenerated from the registered "
            f"seed rule does not match the batch mask for "
            f"{ident['file']} slice {ident['slice_index']}")

    n_acquired = int(mask_oracle.sum().item())
    m = mask_oracle.to(torch.complex64)                     # broadcasts (W,)

    # ---- identity
    k_x = fft2c(x_norm)                                     # F x_norm
    lhs = fft2c(x_norm - cond_in)                           # F dx
    rhs = (1.0 - m) * k_x                                   # (1 - M) F x_norm

    denom = float(torch.max(torch.abs(k_x)).item())
    require_finite({"denominator_max_abs_F_x_norm": denom},
                   "P4FID relative denominator")
    if denom <= 0.0:
        raise StageError("DENOMINATOR_NON_POSITIVE",
                         f"max|F x_norm| = {denom!r}; the relative metrics "
                         f"are undefined for this slice")

    identity_abs = float(torch.max(torch.abs(lhs - rhs)).item())
    free_abs = float(torch.max(torch.abs((1.0 - m) * (lhs - k_x))).item())
    support_abs = float(torch.max(torch.abs(m * lhs)).item())
    require_finite({"identity_abs": identity_abs, "free_abs": free_abs,
                    "support_abs": support_abs}, "P4FID measurements")

    return {
        "split": ident["split"], "file": ident["file"],
        "slice_index": ident["slice_index"],
        "dataset_index": ident["dataset_index"], "epoch": epoch,
        "mask_seed": seed, "mask_seed_matches_meta": seed_equal,
        "mask_matches_oracle": mask_equal,
        "n_acquired_columns": n_acquired,
        "acquired_columns": [int(c) for c in
                             torch.nonzero(mask_oracle).flatten().tolist()],
        "denominator_max_abs_F_x_norm": denom,
        "identity_abs_max": identity_abs,
        "identity_rel_max": rel_of(identity_abs, denom),
        "free_identity_abs_max": free_abs,
        "free_identity_rel_max": rel_of(free_abs, denom),
        "measured_support_abs_max": support_abs,
        "measured_support_rel_max": rel_of(support_abs, denom),
    }


# --------------------------------------------------------------------------
# Stage body
# --------------------------------------------------------------------------

def run_stage(args, parents: dict, t0: float) -> dict:
    torch.set_num_threads(1)
    epoch = FRAME["epoch_set"][0]

    with open(args.p0s_facts, "r", encoding="utf-8") as fh:
        p0s_facts = json.load(fh)
    entries = _p0s_entries(p0s_facts)
    if args.smoke is None and len(entries) != EXPECTED_SUBSET_SIZE:
        raise StageError("SUBSET_SIZE_MISMATCH",
                         f"P0S carries {len(entries)} identities, expected "
                         f"{EXPECTED_SUBSET_SIZE}")

    # TRAIN mode -- the frame's regime. P0S froze these identities under
    # eval mode; inheriting the IDENTITIES into train mode is a deliberate
    # scope extension, recorded rather than assumed. The epoch-0 train masks
    # are NEW deterministic realisations, not inherited ones.
    ds = FastMRISliceDataset(args.data_root, split="train", mode="train")
    ds.set_epoch(epoch)

    # The FRAME is measured from the live dataset, never hard-coded. The
    # registered frame and the diagnostic sample are DIFFERENT sizes and are
    # recorded as such: the frame is the whole training population x one
    # epoch; the sample is the 256 inherited identities.
    if FRAME["train_slices"] is not None:
        raise StageError("FRAME_TRAIN_SLICES_NOT_NULL",
                         f"registered frame requires train_slices=None, got "
                         f"{FRAME['train_slices']!r}")
    if len(FRAME["epoch_set"]) != 1:
        raise StageError("FRAME_EPOCH_SET_SIZE",
                         f"registered frame requires exactly one epoch, got "
                         f"{FRAME['epoch_set']!r}")
    frame_live = {
        "frame_population_size": len(ds),
        "frame_epoch_count": len(FRAME["epoch_set"]),
        "frame_realisation_count": len(ds) * len(FRAME["epoch_set"]),
        "train_slices_is_none": True,
        "subset_seed_operative": False,
        "dataset_split": ds.split, "dataset_mode": ds.mode,
        "dataset_epoch": ds.epoch,
        "measured_from_live_dataset": True,
        "note": "the realisation count is DERIVED as population x epochs, "
                "never chosen; it is measured here and not hard-coded",
    }
    logger.info("frame: %d training slices x %d epoch = %d realisations",
                frame_live["frame_population_size"],
                frame_live["frame_epoch_count"],
                frame_live["frame_realisation_count"])

    identities = resolve_identities(ds, entries, args.data_root)
    if args.smoke is not None:
        identities = identities[: args.smoke]
        logger.warning("SMOKE MODE: %d identities; NOT authoritative",
                       len(identities))

    rows = []
    for ident in identities:
        batch = _collate([ds[ident["dataset_index"]]])
        prep = _prepare(batch, "cpu", test0=False)
        rows.append(measure(ident, batch, prep, epoch))

    counts = {r["n_acquired_columns"] for r in rows}
    if len(counts) != 1:
        raise StageError("ACQUIRED_COUNT_VARIES",
                         f"acquired column count is structurally fixed by "
                         f"mask_counts(); observed {sorted(counts)} -- the "
                         f"generator contract is violated")

    worst = {
        "identity": max(rows, key=lambda r: r["identity_rel_max"]),
        "free": max(rows, key=lambda r: r["free_identity_rel_max"]),
        "support": max(rows, key=lambda r: r["measured_support_rel_max"]),
    }
    checks = {
        "identity": worst["identity"]["identity_rel_max"] <= P4FID_IDENTITY_TOL,
        "free_restriction":
            worst["free"]["free_identity_rel_max"] <= P4FID_FREE_TOL,
        "measured_support":
            worst["support"]["measured_support_rel_max"] <= P4FID_SUPPORT_TOL,
        "mask_provenance": all(r["mask_matches_oracle"]
                               and r["mask_seed_matches_meta"] for r in rows),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        logger.error("registered tolerances exceeded: %s", failed)
        raise StageError("P4FID_TOLERANCE_EXCEEDED",
                         f"frame identity checks failed: {failed}. Estimator "
                         f"A must NOT be frozen and N_SCALE_MIN must NOT be "
                         f"derived against this frame.",
                         detail={"checks": checks})

    return _facts(args, parents, rows, worst, checks, counts, epoch, t0,
                  frame_live)


def _facts(args, parents, rows, worst, checks, counts, epoch, t0,
           frame_live) -> dict:
    def w(key, metric, threshold):
        r = worst[key]
        obs = r[metric]
        return {"file": r["file"], "slice_index": r["slice_index"],
                "observed": obs, "threshold": threshold,
                "margin": (threshold / obs) if obs > 0 else None,
                "margin_status": "finite" if obs > 0 else "unbounded"}

    thresholds = {
        "P4FID_IDENTITY_TOL": P4FID_IDENTITY_TOL,
        "P4FID_FREE_TOL": P4FID_FREE_TOL,
        "P4FID_SUPPORT_TOL": P4FID_SUPPORT_TOL,
        "P4FID_EXPECTED_REL": P4FID_EXPECTED_REL,
        "P4FID_REL_DENOM_FLOOR": P4FID_REL_DENOM_FLOOR,
        "shared_values_are_incidental": True,
    }
    summary = {
        "n_slices": len(rows), "epoch": epoch,
        "n_acquired_columns": sorted(counts)[0],
        "checks": checks,
        "worst_identity": w("identity", "identity_rel_max", P4FID_IDENTITY_TOL),
        "worst_free": w("free", "free_identity_rel_max", P4FID_FREE_TOL),
        "worst_support": w("support", "measured_support_rel_max",
                           P4FID_SUPPORT_TOL),
        "median_identity_rel": float(np.median(
            [r["identity_rel_max"] for r in rows])),
        "expected_magnitude_note":
            "the identity is algebraically EXACT given cond_in = F^H(M y)/a "
            "and y = M F x_true; what is measured is fp32 FFT round-trip "
            "error alone, expected near P4FID_EXPECTED_REL = 1e-7. A passing "
            "1e-7 is the EXPECTED result, not a near-miss.",
        "p2_comparability_note":
            "measured_support_rel_max is NOT comparable to P2's relative_max "
            "= 6.90977e-6. P2 normalised by max|F dx|; this normalises by "
            "max|F x_norm|, which is larger. Same underlying quantity, "
            "different denominator. P2's masks were EVAL-mode; these are "
            "TRAIN-mode at epoch 0.",
        "scope_note":
            "no claim beyond the 256 registered slice identities at epoch 0. "
            "This is a CONSTRUCTION diagnostic: it establishes the "
            "implementation identity and mask provenance, and licenses the "
            "estimator-A registration. It is NOT authoritative P4 scaling "
            "statistics and carries no dataset-population claim.",
        "estimator_note":
            "the identity shows the modelled VALUE at a free coordinate does "
            "not depend on which other columns were selected. It does NOT "
            "choose between estimator A (free-conditioned, concept D4) and "
            "estimator B (unconditional): that is a SAMPLING-POPULATION "
            "decision, and A remains the approved estimator.",
        "fft_scope_note":
            "fft2c appears on BOTH sides of the identity, so a wrong "
            "transform convention CANCELS and is invisible here. The "
            "independence provided is over the MASK and the cond_in "
            "CONSTRUCTION, not the transform.",
    }
    facts = {
        "schema": FACTS_SCHEMA,
        "script": {"id": SCRIPT_ID, "version": SCRIPT_VERSION,
                   "lifetime": "KEEP"},
        "stage": "P4FID",
        "artefact_type": "stage_facts",
        "run_mode": "smoke" if args.smoke else "diagnostic",
        "authoritative": False,
        "evidence_class": "amendment evidence for the implementation identity "
                          "and the estimator-A registration; NON-AUTHORITATIVE",
        "run_mode_note":
            "guard_run_mode is a BINARY OUTPUT-ISOLATION guard; its "
            "non-smoke branch means NON-SMOKE OUTPUT ISOLATION, not "
            "scientific authority. This artefact is a diagnostic in both "
            "branches, and publication into the locked EXEC §9 preflight "
            "directory is REFUSED structurally, not merely discouraged.",
        "verdict": "PASS",
        "thresholds": thresholds,
        "frame": {**FRAME, **frame_live},
        "frame_vs_sample_note":
            "REGISTERED FRAME = full training population x one epoch "
            "(frame_realisation_count). DIAGNOSTIC SAMPLE = the 256 inherited "
            "identities. These are different sizes and are never conflated: "
            "the identity is measured on the sample; the frame is what P4 "
            "will accumulate over.",
        "sample": {
            "inheritance": "the 256 P0S file/slice IDENTITIES, resolved "
                           "against split='train' and loaded in train mode "
                           "at epoch 0",
            "identity_fields": ["split", "file", "slice_index"],
            "index_portability_note":
                "a bare dataset index is NOT portable provenance; it is "
                "resolved from the identity and CROSS-CHECKED against the "
                "P0S record, never trusted alone",
            "new_realisations_note":
                "the slice IDENTITIES are inherited; the epoch-0 TRAIN masks "
                "are NEW deterministic realisations derived here, and P0S "
                "froze these identities under EVAL mode. The extension to "
                "train mode is deliberate and recorded.",
            "n_resolved": len(rows),
        },
        "parents": {
            "p0_facts_sha256": parents["p0"]["facts_sha256"],
            "p0s_facts_sha256": parents["p0s"]["facts_sha256"],
            "subset_manifest_sha256": parents["p0s"].get(
                "subset_manifest_sha256"),
            "contract_hash": parents["p0"].get("contract_hash"),
            "s_ref": parents.get("s_ref"),
        },
        "slices": rows,
        "summary": summary,
        "code": {**hash_project_code(args.repo_dir, os.path.abspath(__file__)),
                 "bindings": {
                     "fastmri_data": file_sha256(os.path.join(
                         args.repo_dir, "seqref_mri/src/fastmri_data.py")),
                     "train_base": file_sha256(os.path.join(
                         args.repo_dir, "seqref_mri/scripts/train_base.py")),
                     "note": "recorded EXPLICITLY rather than relying on "
                             "hash_project_code's frozen file list to cover "
                             "them; both supply the production path and the "
                             "oracle inputs",
                 },
                 "p4fid_local": [
                     {"relpath": p,
                      "sha256": file_sha256(os.path.join(args.repo_dir, p))}
                     for p in P4FID_CODE_FILES
                     if os.path.isfile(os.path.join(args.repo_dir, p))]},
        "run": {**environment_record(args.repo_dir, sys.argv[1:]),
                "runtime_seconds": time.time() - t0,
                "peak_memory_bytes": _peak_rss_bytes(),
                "device": "cpu", "torch_threads": 1},
    }
    semantic = {
        "schema": FACTS_SCHEMA, "stage": "P4FID", "verdict": "PASS",
        "thresholds": thresholds, "frame": {**FRAME, **frame_live},
        "sample": facts["sample"],
        "parents": facts["parents"], "slices": rows, "summary": summary,
        "code": facts["code"],
    }
    return attach_semantic_hash(facts, semantic)


# --------------------------------------------------------------------------

def _parent_ids(parents) -> dict | None:
    """BOTH parents in an ERROR record. P0 alone would omit the subset and
    scale provenance the diagnostic actually inherits from P0S."""
    if not parents:
        return None
    return {"p0": parents.get("p0"), "p0s": parents.get("p0s"),
            "s_ref": parents.get("s_ref")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="SEQREF-P4FID v0.1 -- P4 frame identity diagnostic")
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--p0-facts", required=True)
    ap.add_argument("--p0s-facts", required=True)
    ap.add_argument("--p0s-script", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    t0 = time.time()
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    facts_prefix = SMOKE_FACTS_PREFIX if args.smoke else FACTS_PREFIX
    error_prefix = SMOKE_ERROR_PREFIX if args.smoke else ERROR_PREFIX
    parents = None

    # PRE-BOUNDARY. Validated before the try/except that can publish, so no
    # artefact of any kind -- facts or error record -- can reach a forbidden
    # directory. Special-casing the code inside the handler would leave the
    # ordering fragile; this makes the path unreachable.
    try:
        validate_output_dir(args.out_dir)
    except StageError as exc:
        logger.error("P4FID ERROR [%s] -- %s", exc.error_code, exc.reason)
        logger.error("no artefact written: the output directory is forbidden, "
                     "so the error record has nowhere legitimate to go")
        return EXIT_ERROR

    try:
        if args.smoke is not None and args.smoke <= 0:
            raise StageError("BAD_SMOKE_SIZE",
                             f"--smoke must be positive, got {args.smoke!r}")
        guard_run_mode(args.out_dir, args.smoke is not None)
        parents = verify_parents(args.repo_dir, args.p0_facts, args.p0s_facts,
                                 args.p0s_script)
        facts = run_stage(args, parents, t0)
        path, sha = publish_stage(facts, args.out_dir, facts_prefix, "P4FID")
        logger.info("P4FID PASS n=%d worst identity_rel=%.3e facts=%s sha=%s",
                    facts["summary"]["n_slices"],
                    facts["summary"]["worst_identity"]["observed"], path, sha)
        print(json.dumps({"verdict": "PASS", "path": path,
                          "worst_identity_rel":
                              facts["summary"]["worst_identity"]["observed"],
                          "worst_support_rel":
                              facts["summary"]["worst_support"]["observed"]},
                         indent=2))
        return EXIT_PASS
    except StageError as exc:
        logger.error("P4FID ERROR [%s] -- %s", exc.error_code, exc.reason)
        publish_error(exc, args.out_dir, error_prefix, "P4FID",
                      parents=_parent_ids(parents),
                      code={"script": os.path.abspath(__file__)},
                      run={"argv": raw_argv})
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 -- registered failure boundary
        logger.exception("%s UNEXPECTED ERROR", SCRIPT_ID)
        wrapped = StageError(
            "UNEXPECTED_RUNTIME_ERROR", f"{type(exc).__name__}: {exc}",
            detail={"exception_type": type(exc).__name__,
                    "raised_after_parent_verification": parents is not None},
            write_record=parents is not None)
        publish_error(wrapped, args.out_dir, error_prefix, "P4FID",
                      parents=_parent_ids(parents),
                      code={"script": os.path.abspath(__file__)},
                      run={"argv": raw_argv})
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
