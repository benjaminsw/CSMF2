# SEQREF-P0S v0.1 -- shared normalisation scale + frozen subset (EXEC v0.4 §8)
# LIFETIME: DIAGNOSTIC
#
# Purpose
#   Compute and freeze S_ref, the reference scale every A2 anomaly floor is
#   expressed relative to, and freeze the 256-slice preflight subset that P1,
#   P2 and P3 consume without redrawing.
#
# Live-code re-verification (NOT the same as verifying the parent record)
#   A verified P0 sidecar proves the parent ARTEFACT is intact. It does not
#   prove the code about to be executed is still the code P0 approved: P0 could
#   have passed, _prepare could then have changed, and the sidecar would still
#   verify. P0S therefore recomputes the entity-level contract hash under the
#   locked profile and re-runs both prepare-binding assertions against LIVE
#   code, requiring equality with the P0-recorded hash. A mismatch RAISES with
#   nothing written.
#
#   CRITICALLY, the SCOPE of that recheck does not come from the parent facts.
#   Validating live code against whatever scope P0 happened to record proves
#   only that the code matches the parent's own list -- a truncated parent
#   would validate against its own truncation. That is the self-defined-scope
#   defect already fixed for the declaration, relocated into the parent
#   artefact. The parent record is therefore first required to MATCH the
#   A2-required scope exactly (normalisation_profile.EXPECTED_CONTRACT and
#   EXPECTED_ASSERTIONS), and the live recheck is then driven from those
#   profile constants, not from the facts. The parent supplies the expected
#   HASH and recorded binding names; it never supplies the scope.
#
#   The parent's source binding must also be FULL-CONTENT and must cover the
#   LOCKED roots (parent_expectations.py): a technically valid P0 run over
#   different directories is not an admissible parent.
#
# Canonical arithmetic
#   CPU only, torch.set_num_threads(1), energy reduction in NumPy float64.
#   A frozen reference artefact must not vary with hardware: CUDA and CPU
#   reductions differ in order, and multi-threaded CPU reductions can differ
#   between thread counts. 256 slices -- reproducibility costs nothing here.
#
# NOT metadata-only
#   P0 was a metadata gate and said so structurally. P0S is the first
#   STATISTICAL stage: it imports torch, opens HDF5 through the dataset, and
#   computes numbers from the data. It does NOT inherit P0's guarantee, and its
#   facts record says so.
#
# Normalisation is BORROWED, never reimplemented
#   P0S obtains x_norm by calling the live train_base.py::_prepare(...,
#   test0=False) -- the exact division path P0 pinned by contract hash. A
#   reimplementation could differ subtly from the code under contract, which is
#   precisely the failure Amendment A2 exists to prevent. The returned contract
#   keys are asserted before any statistic is computed.
#
# Locked sampling (EXEC v0.4 §8 P0S)
#   population : every (file, slice) entry of the TRAIN split in the dataset's
#                own deterministic index order (files sorted, slices ascending)
#   draw       : numpy Generator(PCG64(0)).choice(N, 256, replace=False)
#   canonical  : selected indices SORTED ascending -- the manifest hash must
#                not depend on shuffle order. Draw order is recorded too, so
#                exactly what the RNG produced stays auditable.
#   seed 0 is DISTINCT from train_base's subset_seed=20260904 and unrelated to
#   it; both facts are recorded so no later reader infers a relationship.
#   _subset() is deliberately NOT reused: it takes its seed from the training
#   config, preserves draw order, and records no provenance.
#
#   Sampling is uniform over SLICES. Volumes with more slices therefore
#   contribute proportionally more entries. This is intentional: S_ref
#   describes the inspected slice population, not a volume-balanced one.
#
# Dataset mode
#   split="train", mode="eval" -- NOT mode="train". Training mode draws fresh
#   masks per epoch and requires set_epoch(); the preflight must be
#   deterministic, and P1/P2 inherit this frozen mask realisation (EXEC §8 P2
#   requires mode=eval, epoch=null). S_ref itself is mask-independent, since
#   x_norm = x_true / a, but the realisation binds downstream conditioning and
#   measured coefficients. Recorded as a stated deviation from how
#   run_training opens the same split.
#
# Validity gate -- ORDER IS LOAD-BEARING (concept §3.1a)
#   1. compute all full-state norms;
#   2. every e_i = ||x_norm,i||^2_2 must be finite and non-negative;
#   3. S_ref = median_i ||x_norm,i||_2   (FULL two-channel state);
#   4. S_ref must be finite and STRICTLY POSITIVE;
#      -- 2 and 4 hold BEFORE any ratio is formed, so NaN or zero-scale cases
#         are never tallied as ordinary degeneracy;
#   5. RELATIVE LOW-ENERGY DIAGNOSTIC (recorded, CANNOT block):
#        relative_ratio_i = e_i / S_ref^2
#        report fraction(relative_ratio_i < P0S_STATE_MIN)
#      S_ref IS the population median, so no more than half the population can
#      lie below it and this fraction is structurally bounded by 0.50. The
#      original ">0.50 -> BLOCK" rule was therefore UNREACHABLE, and worse:
#      when the population really is majority-degenerate the median collapses
#      with it, so those slices score ratio ~ 1 and are not flagged at all.
#      Diagnostic only; useful for spotting a MINORITY of empty slices.
#      R_REAL_MIN is P1's real-branch DENOMINATOR guard and is NOT used here:
#      it would make P0S validity phase-sensitive and misreport a valid,
#      predominantly imaginary state as degenerate.
#   6. ABSOLUTE POPULATION-DEGENERACY GATE (reachable, blocking):
#        D      = number of real state elements = 2 * 96 * 96, DERIVED from
#                 the tensor shape and cross-checked, never hardcoded blindly
#        mse_i  = e_i / D
#        absolute_degenerate_i <=> mse_i < P0S_ABS_STATE_MSE_MIN
#        absolute_degenerate_fraction > 0.50 -> BLOCK; equality does NOT block.
#
#   WHY AN ABSOLUTE FLOOR DOES NOT CONTRADICT A2. A2 retired absolute
#   constants because per-volume scaling made absolute ENERGY meaningless.
#   That reasoning does not extend here: dividing by the volume's magnitude
#   maximum bounds the normalised state to |x_norm| <= 1 BY CONSTRUCTION for
#   every volume, so this representation has a common ceiling raw energy never
#   had. The floor is a NUMERICAL-EMPTY-STATE guard on a bounded
#   representation, not a claim about medically meaningful low signal.
#
#   That boundedness is an inference from Construction A, not a measurement,
#   so P0S RECORDS max_i max|x_norm,i| and RAISES if it materially exceeds 1:
#   the floor's premise would be false, which is a specification problem, not
#   a verdict about the data.
#
# Verdict semantics
#   PASS   gate satisfied; S_ref and the frozen subset are published.
#   BLOCK  majority-degenerate subset: a recorded scientific verdict, exit 1.
#          S_ref would be meaningless, so no number is carried forward.
#   RAISE  parent artefact unverifiable, contract keys absent, non-finite or
#          negative e_i, S_ref non-positive, duplicate or out-of-range index.
#          Nothing is written.
#
# CONVENTION: every failure path -> logger.error + raise. No fallback, no mock,
#   no placeholder, no silent pass.
#
# Changelog
#   v0.1 (2026-07-29) Created under Amendment A2 and its P0S clarification.
#     The relative >0.50 blocking rule was found UNREACHABLE during testing --
#     S_ref is the population median, so the fraction below it is bounded by
#     0.50 by construction, and a degenerate majority collapses the median
#     with it. Replaced by an absolute normalised-state floor, with the
#     relative fraction retained as a diagnostic.

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

from preflight_io import (canonical_hash, check_pairing,  # noqa: E402
                          file_sha256, git_state, publish, utc_stamp,
                          verify_sidecar)
from contract_hash import (contract_hash, check_prepare_binding,  # noqa: E402
                           PROCEDURE_ID, ASSERT_PROCEDURE_ID)
from normalisation_profile import (EXPECTED_ASSERTIONS,  # noqa: E402
                                   EXPECTED_CONTRACT, PROFILE_ID)
from parent_expectations import (EXPECTATIONS_ID,  # noqa: E402
                                 EXPECTED_SOURCE_FILE_COUNT,
                                 EXPECTED_SOURCE_ROOTS,
                                 validate_parent_scope,
                                 validate_parent_source_binding)
from seqref_mri.src.fastmri_data import FastMRISliceDataset  # noqa: E402
from seqref_mri.scripts.train_base import _collate, _prepare  # noqa: E402

SCRIPT_ID = "SEQREF-P0S"
SCRIPT_VERSION = "v0.1"
FACTS_SCHEMA = "seqref-p0s-facts/1"
FACTS_PREFIX = "normalisation_scale_facts"

SUBSET_SIZE = 256
SUBSET_SEED = 0
TRAINING_SUBSET_SEED = 20260904          # train_base default; UNRELATED
P0S_STATE_MIN = 1e-10                    # relative diagnostic only
P0S_ABS_STATE_MSE_MIN = 1e-12            # blocking; normalised RMS < 1e-6
DEGENERATE_BLOCK_FRACTION = 0.50
STATE_MAX_TOL = 1.0 + 1e-3               # |x_norm| <= 1 by construction
EXPECTED_D = 2 * 96 * 96                 # G0 3.1 + two-channel state

REQUIRED_PREPARE_KEYS = ("y", "x_norm", "cond_in", "tgt_norm", "amax", "ops")
P0_FACTS_SCHEMA = "seqref-p0-facts/2"

logger = logging.getLogger(SCRIPT_ID)


def draw_subset(n_population: int) -> tuple[list[int], list[int]]:
    """Locked sampler. Returns (draw_order, canonical_sorted)."""
    if n_population < SUBSET_SIZE:
        logger.error("population has %d entries, fewer than the locked subset "
                     "size %d", n_population, SUBSET_SIZE)
        raise ValueError("population smaller than subset size")
    rng = np.random.Generator(np.random.PCG64(SUBSET_SEED))
    idx = rng.choice(n_population, size=SUBSET_SIZE, replace=False)
    draw_order = [int(i) for i in idx]
    if len(set(draw_order)) != SUBSET_SIZE:
        logger.error("sampler returned duplicate indices despite "
                     "replace=False -- refusing to proceed")
        raise RuntimeError("duplicate indices in subset draw")
    if min(draw_order) < 0 or max(draw_order) >= n_population:
        logger.error("sampler returned an index outside [0, %d)",
                     n_population)
        raise RuntimeError("subset index out of range")
    return draw_order, sorted(draw_order)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="SEQREF-P0S v0.1 -- shared normalisation scale and frozen "
                    "256-slice preflight subset (STATISTICAL stage)")
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--data-root", required=True,
                    help="dataset root, e.g. seqref_mri/data/fastmri")
    ap.add_argument("--p0-facts", required=True,
                    help="path to the PASSED p0_facts.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", type=int, default=8,
                    help="loader batch size; affects speed only, not results")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    # ---- parent artefact: verify, bind, and record ITS git state -----------
    p0_sha = verify_sidecar(args.p0_facts)
    with open(args.p0_facts, "rb") as fh:
        p0 = json.load(fh)
    if p0.get("schema") != P0_FACTS_SCHEMA:
        logger.error("parent schema is %r, expected %r", p0.get("schema"),
                     P0_FACTS_SCHEMA)
        raise ValueError("parent schema mismatch")
    if p0.get("stage") != "P0":
        logger.error("parent stage is %r, expected 'P0'", p0.get("stage"))
        raise ValueError("parent stage mismatch")
    if p0.get("verdict") != "PASS":
        logger.error("P0 verdict is %r; P0S runs only after a PASS",
                     p0.get("verdict"))
        raise RuntimeError("P0 did not pass")

    p0_cv = p0.get("contract_verification", {})
    if p0_cv.get("reproduced") is not True:
        logger.error("parent contract_verification.reproduced is %r; a P0 that "
                     "did not reproduce its contract is not an admissible "
                     "parent", p0_cv.get("reproduced"))
        raise RuntimeError("parent contract not reproduced")

    # --- parent source binding: full-content AND the LOCKED roots ---------
    p0_source_binding = p0.get("source_binding", {})
    validate_parent_source_binding(p0_source_binding)      # raises on failure
    p0_manifest_field = p0_source_binding["field"]
    p0_manifest_sha = p0_source_binding[p0_manifest_field]

    # --- the PARENT RECORD must itself cover the A2-required scope ---------
    # (before any live source is read: a truncated parent must not become the
    #  basis of the live check)
    validate_parent_scope(p0_cv, p0.get("assertion_verification", {}),
                          EXPECTED_CONTRACT, EXPECTED_ASSERTIONS)

    # --- LIVE code must still be the code P0 approved ----------------------
    # Scope comes from the LOCKED PROFILE, not from the parent facts. The
    # parent supplies only the expected hash.
    live_files = []
    for spec in EXPECTED_CONTRACT:
        path = os.path.join(args.repo_dir, spec["relpath"])
        if not os.path.isfile(path):
            logger.error("contract file missing from the tree: %s", path)
            raise FileNotFoundError(path)
        with open(path, "rb") as fh:
            live_files.append({
                "relpath": spec["relpath"], "source_bytes": fh.read(),
                "entities": list(spec["entities"])})
    live = contract_hash(live_files)
    if live["contract_hash"] != p0_cv["live_hash"]:
        logger.error("LIVE CODE DIVERGED FROM P0: contract hash is %s, P0 "
                     "recorded %s. The parent record is intact but no longer "
                     "describes the code P0S is about to execute. Re-run P0.",
                     live["contract_hash"], p0_cv["live_hash"])
        raise RuntimeError("live code diverged from the approved contract")

    # Assertions are driven from EXPECTED_ASSERTIONS, so the required set is
    # fixed by the profile; the parent supplies only the recorded binding name.
    p0_bindings = {r.get("id"): r.get("binding")
                   for r in p0.get("assertion_verification", {})
                   .get("results", [])}
    live_assertions = []
    for aid, relpath, function, callee in EXPECTED_ASSERTIONS:
        path = os.path.join(args.repo_dir, relpath)
        if not os.path.isfile(path):
            logger.error("assertion target file missing: %s", path)
            raise FileNotFoundError(path)
        with open(path, "rb") as fh:
            src = fh.read()
        a = check_prepare_binding(src, relpath, function, callee)  # raises
        a["id"] = aid
        if a["binding"] != p0_bindings.get(aid):
            logger.error("assertion %s: binding changed (P0 recorded %r, live "
                         "%r)", aid, p0_bindings.get(aid), a["binding"])
            raise ValueError(f"live assertion binding changed: {aid}")
        live_assertions.append(a)

    p0_git = p0.get("code", {}).get("git", {})
    logger.info("parent P0 verified: sha=%s convention=%s git_dirty=%s "
                "live_contract=%s assertions=%d", p0_sha[:16],
                p0.get("declaration", {}).get("convention_label"),
                p0_git.get("dirty"), live["contract_hash"][:16],
                len(live_assertions))

    check_pairing(args.out_dir, FACTS_PREFIX)

    # ---- dataset: TRAIN split in EVAL mode (deterministic) -----------------
    ds = FastMRISliceDataset(args.data_root, split="train", mode="eval")
    n_population = len(ds)
    population_manifest = [
        {"relpath": os.path.relpath(str(f), args.data_root).replace(
            os.sep, "/"), "slice": int(s)} for f, s in ds.index]
    population_manifest_hash = canonical_hash(population_manifest)

    draw_order, canonical = draw_subset(n_population)

    # Canonical arithmetic: CPU, single-threaded. See header.
    device = "cpu"
    torch.set_num_threads(1)
    loader = DataLoader(Subset(ds, canonical), batch_size=args.batch,
                        shuffle=False, num_workers=0, collate_fn=_collate)

    entries: list[dict] = []
    e_values: list[float] = []
    max_abs_values: list[float] = []
    cmag_values: list[float] = []
    d_state: int | None = None
    for batch in loader:
        p = _prepare(batch, device, test0=False)
        missing = [k for k in REQUIRED_PREPARE_KEYS if k not in p]
        if missing:
            logger.error("_prepare() returned no %s -- the contract keys P0S "
                         "depends on are absent; the preparation contract has "
                         "changed", missing)
            raise KeyError(f"_prepare missing keys: {missing}")
        x = p["x_norm"]
        if x.ndim != 4:
            logger.error("x_norm has shape %s; expected (B, C, H, W) with the "
                         "full two-channel state", tuple(x.shape))
            raise ValueError("unexpected x_norm rank")
        # FULL two-channel state energy, in float64: the same quantity S_ref
        # is built from (concept §3.1a step 5).
        # Canonical arithmetic: move to NumPy float64 and reduce there, so
        # the frozen numbers do not depend on torch's reduction strategy.
        xn = x.detach().cpu().numpy().astype(np.float64, copy=False)
        e = (xn ** 2).reshape(xn.shape[0], -1).sum(axis=1)
        amax_state = np.abs(xn).reshape(xn.shape[0], -1).max(axis=1)
        # Boundedness premise is about COMPLEX MAGNITUDE (Construction A:
        # |x_true| <= file_attr_max), not per-element |real|. Measure both.
        if xn.shape[1] != 2:
            logger.error("state has %d channels; the two-channel complex "
                         "state is required to form a magnitude",
                         xn.shape[1])
            raise ValueError("unexpected channel count")
        cmag = np.sqrt(xn[:, 0] ** 2 + xn[:, 1] ** 2)
        cmag_max = cmag.reshape(cmag.shape[0], -1).max(axis=1)
        xf = xn
        if d_state is None:
            d_state = int(xf[0].size)
        elif d_state != int(xf[0].size):
            logger.error("state element count changed mid-pass: %d then %d",
                         d_state, int(xf[0].size))
            raise RuntimeError("inconsistent state dimension")
        for j, meta in enumerate(batch["meta"]):
            entries.append({
                "file": meta["file"], "slice_index": int(meta["slice_index"]),
                "split": meta["split"], "mode": meta["mode"],
                "mask_seed": int(meta["mask_seed"]),
                "file_attr_max": float(meta["file_attr_max"]),
                "e_i": float(e[j]),
                "max_abs_state": float(amax_state[j]),
                "max_complex_magnitude": float(cmag_max[j]),
            })
            e_values.append(float(e[j]))
            max_abs_values.append(float(amax_state[j]))
            cmag_values.append(float(cmag_max[j]))

    if len(entries) != SUBSET_SIZE:
        logger.error("collected %d entries, expected %d", len(entries),
                     SUBSET_SIZE)
        raise RuntimeError("subset size mismatch after loading")
    for k, idx in enumerate(canonical):
        entries[k]["dataset_index"] = idx

    # ---- validity gate, IN ORDER ------------------------------------------
    e_arr = np.asarray(e_values, dtype=np.float64)
    if not np.all(np.isfinite(e_arr)):
        bad = [entries[i] for i in np.flatnonzero(~np.isfinite(e_arr))]
        logger.error("non-finite state energy on %d slice(s), first: %s",
                     len(bad), bad[0])
        raise RuntimeError("non-finite e_i")
    if np.any(e_arr < 0.0):
        logger.error("negative state energy encountered -- impossible for a "
                     "sum of squares; the computation path is wrong")
        raise RuntimeError("negative e_i")

    s_ref = float(np.median(np.sqrt(e_arr)))
    if not np.isfinite(s_ref) or s_ref <= 0.0:
        logger.error("S_ref is %r; it must be finite and strictly positive "
                     "before any ratio is formed", s_ref)
        raise RuntimeError("S_ref non-positive or non-finite")

    # ---- boundedness premise of the ABSOLUTE floor -----------------------
    max_abs_arr = np.asarray(max_abs_values, dtype=np.float64)
    cmag_arr = np.asarray(cmag_values, dtype=np.float64)
    element_max = float(max_abs_arr.max())
    state_max = float(cmag_arr.max())       # the actual construction bound
    if not np.isfinite(state_max) or not np.isfinite(element_max):
        logger.error("non-finite state magnitude encountered")
        raise RuntimeError("non-finite state magnitude")
    if state_max > STATE_MAX_TOL:
        logger.error("max complex magnitude = %.6g exceeds 1 (tol %.6g). "
                     "The absolute "
                     "state floor P0S_ABS_STATE_MSE_MIN presumes the "
                     "normalised state is bounded by construction (|x_true| "
                     "<= file_attr_max). That premise does not hold here, so "
                     "the floor has no footing: this is a SPECIFICATION "
                     "problem, not a verdict about the data. Amend concept "
                     "v0.4 §3.1a before rerunning.", state_max, STATE_MAX_TOL)
        raise RuntimeError("normalised state exceeds its construction bound")

    if d_state is None:
        logger.error("state dimension was never determined")
        raise RuntimeError("state dimension unknown")
    if d_state != EXPECTED_D:
        logger.error("state has %d real elements, expected %d (2 x 96 x 96, "
                     "G0 3.1 + two-channel state). A state-shape change must "
                     "be an explicit amendment, not a silent rescaling of the "
                     "per-element floor.", d_state, EXPECTED_D)
        raise RuntimeError("unexpected state dimension")

    # ---- 5. RELATIVE low-energy DIAGNOSTIC (cannot block) -----------------
    ratios = e_arr / (s_ref ** 2)
    relative_low = ratios < P0S_STATE_MIN
    relative_low_fraction = float(relative_low.mean())

    # ---- 6. ABSOLUTE population-degeneracy GATE (reachable, blocking) -----
    mse = e_arr / float(d_state)
    absolute_degenerate = mse < P0S_ABS_STATE_MSE_MIN
    absolute_degenerate_fraction = float(absolute_degenerate.mean())

    for k in range(SUBSET_SIZE):
        entries[k]["ratio"] = float(ratios[k])
        entries[k]["relative_low_energy"] = bool(relative_low[k])
        entries[k]["mse"] = float(mse[k])
        entries[k]["absolute_degenerate"] = bool(absolute_degenerate[k])

    def _ids(flags):
        return [{"dataset_index": entries[i]["dataset_index"],
                 "file": entries[i]["file"],
                 "slice_index": entries[i]["slice_index"],
                 "e_i": entries[i]["e_i"], "mse": entries[i]["mse"],
                 "ratio": entries[i]["ratio"]}
                for i in np.flatnonzero(flags)]

    q90 = float(np.quantile(e_arr, 0.90))
    spread_q90 = (s_ref ** 2) / q90 if q90 > 0.0 else None

    if absolute_degenerate_fraction > DEGENERATE_BLOCK_FRACTION:
        verdict = "BLOCK"
        reason = (f"absolute_degenerate_fraction "
                  f"{absolute_degenerate_fraction:.4f} > "
                  f"{DEGENERATE_BLOCK_FRACTION}; most normalised states are "
                  f"numerically empty, so S_ref describes a degenerate "
                  f"population and is not carried forward")
    else:
        verdict = "PASS"
        reason = (f"absolute_degenerate_fraction "
                  f"{absolute_degenerate_fraction:.4f} <= "
                  f"{DEGENERATE_BLOCK_FRACTION}; S_ref is finite, strictly "
                  f"positive and computed on a population that is not "
                  f"numerically empty")

    subset_manifest = [{"dataset_index": e["dataset_index"],
                        "file": e["file"], "slice_index": e["slice_index"]}
                       for e in entries]

    facts = {
        "schema": FACTS_SCHEMA,
        "script": {"id": SCRIPT_ID, "version": SCRIPT_VERSION,
                   "lifetime": "DIAGNOSTIC"},
        "stage": "P0S",
        "stage_description": "shared normalisation scale + frozen subset",
        "statistics_computed": True,
        "metadata_only": False,
        "metadata_only_note": (
            "P0S is a STATISTICAL, torch/HDF5-reading stage and does NOT "
            "inherit P0's metadata-only guarantee"),
        "run": {"utc": utc_stamp(), "python": sys.version.split()[0],
                "numpy": np.__version__, "torch": torch.__version__,
                "device": device,
                "torch_threads": torch.get_num_threads(),
                "arithmetic_path": "energy reduction in NumPy float64 on CPU, "
                                   "single-threaded; canonical so the frozen "
                                   "record does not vary with hardware",
                "argv": sys.argv[1:] if argv is None else list(argv)},
        "code": {"git": git_state(args.repo_dir),
                 "repo_dir": os.path.abspath(args.repo_dir),
                 "script_path": os.path.abspath(__file__),
                 "script_sha256": file_sha256(os.path.abspath(__file__))},
        "parent_artifact_shas": {
            "p0_facts": p0_sha,
            "p0_facts_path": os.path.abspath(args.p0_facts),
            "p0_source_manifest_field": p0_manifest_field,
            "p0_source_manifest_sha256": p0_manifest_sha,
            "p0_source_binding_full_content": True,
            "p0_source_roots": p0_source_binding.get("roots_canonical"),
            "p0_source_n_files": p0_source_binding.get("n_files"),
            "p0_git": p0_git,
            "provenance_note": (
                "P0 was executed from the working tree recorded above. P0S's "
                "own commit is recorded separately under 'code'; this record "
                "does NOT imply P0 was run from P0S's commit."),
        },
        "dataset": {
            "data_root": args.data_root,
            "split": "train", "mode": "eval", "epoch": None, "test0": False,
            "mode_rationale": (
                "eval mode gives a deterministic mask realisation; train mode "
                "draws fresh masks per epoch and requires set_epoch(). S_ref "
                "is mask-independent (x_norm = x_true / a), but the "
                "realisation binds downstream conditioning and measured "
                "coefficients, and EXEC §8 P2 requires mode=eval, epoch=null. "
                "This is a stated deviation from how run_training opens the "
                "same split."),
            "population_size": n_population,
            "population_manifest_sha256": population_manifest_hash,
        },
        "sampling": {
            "subset_size": SUBSET_SIZE,
            "seed": SUBSET_SEED,
            "generator": "numpy.random.Generator",
            "bit_generator": "PCG64",
            "numpy_version": np.__version__,
            "call": f"Generator(PCG64({SUBSET_SEED})).choice("
                    f"{n_population}, size={SUBSET_SIZE}, replace=False)",
            "draw_order_indices": draw_order,
            "canonical_sorted_indices": canonical,
            "canonical_order_rationale": (
                "downstream stages consume the SORTED list so the manifest "
                "hash does not depend on shuffle order; draw order is kept so "
                "exactly what the RNG produced remains auditable"),
            "uniqueness_and_range_asserted": True,
            "uniform_over": "slices",
            "weighting_note": (
                "sampling is uniform over SLICES, so volumes with more slices "
                "contribute proportionally more entries. Intentional: S_ref "
                "describes the inspected slice population, not a "
                "volume-balanced population."),
            "training_subset_seed": TRAINING_SUBSET_SEED,
            "relationship_to_training_subset": "none",
            "subset_helper_note": (
                "train_base._subset() is deliberately NOT reused: it takes "
                "its seed from the training config, preserves draw order, and "
                "records no provenance"),
            "subset_manifest_sha256": canonical_hash(subset_manifest),
        },
        "normalisation_source": {
            "callee": "seqref_mri/scripts/train_base.py::_prepare",
            "test0": False,
            "required_keys": list(REQUIRED_PREPARE_KEYS),
            "note": "normalisation is BORROWED from the contract-pinned "
                    "division path, never reimplemented",
        },
        "s_ref": {
            "definition": "median_i ||x_norm,i||_2 over the frozen subset, "
                          "FULL two-channel complex state (real and imaginary "
                          "together)",
            "value": s_ref,
            "value_squared": s_ref ** 2,
            "dtype": "float64",
            "used_for": ["P0S_STATE_MIN", "R_REAL_MIN", "R_RESID_MIN",
                         "R_FREE_MIN"],
            "not_used_for": "coefficient-wise maximum leakage (P2 uses k_i); "
                            "and the absolute degeneracy gate, which must not "
                            "reference the median it validates",
            "valid_for_downstream": verdict == "PASS",
            "valid_for_downstream_note": (
                "a BLOCK record still reports the measured S_ref for "
                "diagnosis, but it MUST NOT be consumed. P1/P2/P3 require "
                "verdict == PASS AND s_ref.valid_for_downstream == true."),
        },
        "live_code_verification": {
            "blocking": True,
            "contract_procedure": PROCEDURE_ID,
            "assertion_procedure": ASSERT_PROCEDURE_ID,
            "profile": PROFILE_ID,
            "parent_expectations": EXPECTATIONS_ID,
            "scope_source": "normalisation_profile.EXPECTED_CONTRACT / "
                            "EXPECTED_ASSERTIONS -- NOT the parent facts. The "
                            "parent supplies the expected hash and recorded "
                            "binding names only.",
            "expected_source_roots": list(EXPECTED_SOURCE_ROOTS),
            "expected_source_file_count": EXPECTED_SOURCE_FILE_COUNT,
            "p0_recorded_hash": p0_cv["live_hash"],
            "live_hash": live["contract_hash"],
            "reproduced": True,
            "assertions": live_assertions,
            "note": "a verified parent SIDECAR proves the record is intact; "
                    "this proves the executing code is still the code P0 "
                    "approved",
        },
        "relative_low_energy_diagnostic": {
            "blocking": False,
            "constant": "P0S_STATE_MIN", "value": P0S_STATE_MIN,
            "test": "e_i / S_ref^2 < P0S_STATE_MIN, with e_i the FULL "
                    "two-channel state energy -- the same quantity S_ref is "
                    "built from. R_REAL_MIN is P1's real-branch denominator "
                    "guard and is NOT used here.",
            "structural_bound": (
                "S_ref IS the population median, so this fraction cannot "
                "exceed 0.50 and MUST NOT determine PASS/BLOCK. A "
                "majority-degenerate population collapses the median with it, "
                "leaving those slices at ratio ~ 1 and unflagged."),
            "count": int(relative_low.sum()),
            "fraction": relative_low_fraction,
            "slices": _ids(relative_low),
            "ratio_min": float(ratios.min()),
            "ratio_median": float(np.median(ratios)),
        },
        "absolute_degeneracy_gate": {
            "blocking": True,
            "constant": "P0S_ABS_STATE_MSE_MIN",
            "value": P0S_ABS_STATE_MSE_MIN,
            "D": d_state, "D_expected": EXPECTED_D,
            "test": "mse_i = e_i / D < P0S_ABS_STATE_MSE_MIN "
                    "(normalised RMS amplitude below 1e-6)",
            "block_boundary": f"absolute_degenerate_fraction > "
                              f"{DEGENERATE_BLOCK_FRACTION} "
                              f"(equality does NOT block)",
            "count": int(absolute_degenerate.sum()),
            "fraction": absolute_degenerate_fraction,
            "slices": _ids(absolute_degenerate),
            "mse_min": float(mse.min()), "mse_median": float(np.median(mse)),
            "mse_max": float(mse.max()),
            "rationale": (
                "a NUMERICAL-EMPTY-STATE guard, not a claim about medically "
                "meaningful low signal. It is reachable because the threshold "
                "does not reference the median being validated."),
            "reconciliation_with_A2": (
                "A2 retired absolute constants because per-volume scaling "
                "made absolute ENERGY meaningless. That does not extend here: "
                "dividing by the volume magnitude maximum bounds the "
                "normalised state to |x_norm| <= 1 by construction, giving "
                "this representation a common ceiling raw energy never had."),
            "boundedness_premise": {
                "claim": "|x_norm| <= 1 for every slice (Construction A)",
                "measured_max_complex_magnitude": state_max,
                "measured_max_abs_real_element": element_max,
                "checked_quantity": "max sqrt(re^2 + im^2) -- the quantity "
                                    "Construction A bounds; the per-element "
                                    "maximum is recorded alongside but is a "
                                    "weaker statement",
                "tolerance": STATE_MAX_TOL,
                "verified": True,
                "note": "measured, not assumed; exceeding the bound RAISES, "
                        "because the floor's premise would be false -- a "
                        "specification problem, not a data verdict",
            },
        },
        "spread_diagnostic": {
            "blocking": False,
            "statistic": "S_ref^2 / quantile_0.90(e_i)",
            "value": spread_q90,
            "q90_e": q90,
            "note": "q90 rather than max, which one energetic but valid slice "
                    "would dominate. Recorded only; NO threshold is attached "
                    "until one is justified on locked, candidate-blind data.",
        },
        "energy_summary": {
            "e_min": float(e_arr.min()), "e_median": float(np.median(e_arr)),
            "e_max": float(e_arr.max()),
        },
        "entries": entries,
        "verdict": verdict,
        "verdict_reason": reason,
        "downstream": (
            "P1 and P2 may run IN PARALLEL, both consuming this frozen subset "
            "and S_ref; neither redraws the subset nor consumes the other's "
            "verdict"
            if verdict == "PASS"
            else "HALT: S_ref is not carried forward; amend concept v0.4 "
                 "§3.1a and EXEC v0.4 §8 before any further stage"),
        "verify_before_use": [
            "P1, P2 and P3 must verify this file against its sidecar and the "
            "P0 sidecar before running",
        ],
        "hash_note": (
            "the authoritative artefact SHA is the SHA-256 of THIS FILE'S "
            "bytes, recorded in the sidecar; no self-referential hash is "
            "embedded"),
        "overwrite_policy": (
            "authoritative file is never overwritten in place; reruns write a "
            "new timestamped record alongside it"),
    }

    path, artifact_sha = publish(facts, args.out_dir, FACTS_PREFIX)

    logger.info("P0S verdict=%s S_ref=%.6e abs_degenerate=%d/%d (%.4f) "
                "rel_low=%d (%.4f, diagnostic) max|x_norm|=%.4f "
                "population=%d facts=%s file_sha256=%s", verdict, s_ref,
                int(absolute_degenerate.sum()), SUBSET_SIZE,
                absolute_degenerate_fraction, int(relative_low.sum()),
                relative_low_fraction, state_max, n_population, path,
                artifact_sha)
    logger.info("  subset_manifest_sha256=%s population_manifest_sha256=%s",
                facts["sampling"]["subset_manifest_sha256"][:16],
                population_manifest_hash[:16])
    if verdict == "PASS":
        logger.info("P0S PASS -- %s", reason)
        return 0
    logger.error("P0S BLOCK -- %s", reason)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
