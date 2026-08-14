# SEQREF-TINY v0.1 -- scripts.tiny_gate
# LIFETIME: KEEP
# =============================================================================
# Purpose: TINY / Class-B tiny-batch model-compatibility gate driver
#          (B1-B3) for the registered free-coordinate conditional NSF.
#          This is a GATE DRIVER, not a second trainer: the production
#          model, objective and optimizer step are reused unchanged from
#          SEQREF-IMPLR (free_flow_runtime) and SEQREF-IMPLT
#          (train_free_flow). This stage owns only: parent verification,
#          the registered 8-slice selection, endpoint metrics, the B1-B3
#          decisions, evidence assembly and publication.
# Locked package (EXEC SS10.3/10.4/10.5, TINY pre-implementation locks
#   2026-08-13; SS9.1 artefact registration):
#   * population  : FULL train split, FastMRISliceDataset(mode="eval");
#                   NO P0S exclusion -- accidental overlap with the frozen
#                   256 is permitted and recorded, never prevented.
#   * selection   : numpy Generator(PCG64(0)).choice(N, 8, replace=False);
#                   draw order IS the fixed training/evaluation order.
#   * model       : build_model(init_seed=0), spline_b from the pinned
#                   IMPL-B artefact; batch = all 8 slices every step.
#   * training    : 500 fixed steps, Adam lr 1e-4, betas (0.9, 0.999),
#                   eps 1e-8, weight_decay 0, no schedule; float32.
#   * B1          : (batch-mean NLL@0 - batch-mean NLL@500) / 13824 >= 0.10
#                   (production nll_objective reduction; divided ONCE).
#   * B2          : z=0 primary; per-slice PSNR then mean (data_range=1.0,
#                   normalised magnitude space); PASS iff
#                   final-initial >= 2.0 dB AND final >= anchor-0.5 dB.
#   * B3          : z=0 production inverse in UNSTANDARDISED physical free
#                   coordinates (P4 scaling undone); NMSE_u per slice then
#                   mean; R_FREE_MIN=1e-10 exclusions (vs S_ref^2) applied
#                   BEFORE the mean; PASS iff final <= 0.5 * initial.
#   * secondary   : posterior mean over ONE fixed latent bank (n=8, seed
#                   0), identical bank at both endpoints; computed and
#                   reported, NEVER gates (EXEC SS10.1).
#   * endpoints   : metrics at step 0 (untouched model) and step 500 only;
#                   no checkpoint selection, no best-of-500.
#   * taxonomy    : PASS exit 0 | BLOCK exit 1 (valid run, thresholds
#                   failed; facts WITH the measurements still published;
#                   NO threshold revision) | ERROR exit 2 (typed error
#                   record; never masquerades as scientific evidence).
# Publication: seqref_mri/results/_diag/tiny/tiny_facts.json
#   (schema seqref-tiny-facts/1) under the campaign claim/publish/sidecar
#   machinery; reruns write a stamped sibling, never overwrite.
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * Introduced for the TINY stage after the 2026-08-13 EXEC locks.
#   * Review-repair round (2026-08-13, pre-fixtures; NO contract change):
#     (1) IMPL parent byte-hash propagated from the verified loader return
#     instead of a nonexistent facts field (would have KeyError'd at facts
#     assembly after 500 steps); (2) secondary estimator now averages the
#     complex reconstructions FIRST and takes magnitude once -- the
#     registered production posterior-mean convention (train_base
#     ._posterior_mean); previously mean-of-magnitudes; (3) selection
#     identities are portable: split + data-root-relative POSIX path +
#     slice_index (+ dataset_index recorded), no machine-specific root
#     spelling inside the manifest hash; (4) B3 gates the registered
#     inequality final <= 0.5*initial DIRECTLY (zero-initial safe); the
#     ratio is report-only and null when initial == 0; (5) endpoint drift
#     checks compare exact per-slice anchor-PSNR vectors and exact
#     exclusion identity sets, not means/counts.
#   * Overlap-recording repair (2026-08-13, pre-execution; NO contract
#     change): the EXEC lock "accidental overlap with the frozen P0S 256
#     is permitted and recorded" was only asserted in a descriptive
#     string; the driver now extracts the frozen P0S index set from the
#     VERIFIED P0S artefact (sampling.canonical_sorted_indices) and
#     RECORDS the observed overlap (count, dataset indices, identities)
#     in the selection block. Observation only -- no exclusion, no
#     redraw.
#   * Provenance hardening (2026-08-13, pre-execution; NO rule change):
#     seqref_mri/scripts/tiny_selftest.py added to TINY_LOCAL_FILES so
#     the fixtures harness hash travels inside the TINY evidence code
#     record.
# Update summary:
#   v0.1 lands the gate driver: PCG64 selection, parent chain through
#   IMPL (with code-pin re-verification), endpoint metric engine (z=0
#   primary + fixed-bank secondary), B1-B3 evaluation and the
#   PASS/BLOCK/ERROR publication taxonomy. The 2026-08-13 review-repair
#   round fixed two definite defects (IMPL byte-hash propagation,
#   secondary estimator convention) and three robustness/provenance
#   items (portable identities, direct B3 inequality, exact drift
#   invariants) found in pre-execution review; the scientific contract
#   is unchanged. A second 2026-08-13 repair (pre-execution) made P0S
#   overlap actually RECORDED (count, dataset indices, identities from
#   the verified P0S artefact) instead of merely asserted -- observation
#   only, no exclusion, no redraw.
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

from preflight_io import (canonical_hash, file_sha256,  # noqa: E402
                          verify_sidecar)
from preflight_parents import (StageError, EXIT_PASS, EXIT_BLOCK,  # noqa: E402
                               EXIT_ERROR, verify_parents,
                               hash_project_code, environment_record,
                               attach_semantic_hash, publish_stage,
                               publish_error)
import residual_decoder as dec  # noqa: E402
from seqref_mri.src.fastmri_data import FastMRISliceDataset  # noqa: E402
from seqref_mri.src import fastmri_data as fdm  # noqa: E402
from seqref_mri.src import free_flow_runtime as ffr  # noqa: E402
from seqref_mri.src.metrics import psnr_per_sample  # noqa: E402
from seqref_mri.scripts import train_free_flow as tff  # noqa: E402
from seqref_mri.scripts.train_base import _collate, _prepare  # noqa: E402

SCRIPT_ID = "SEQREF-TINY"
SCRIPT_VERSION = "v0.1"
STAGE = "TINY"
FACTS_SCHEMA = "seqref-tiny-facts/1"
FACTS_PREFIX = "tiny_facts"
ERROR_PREFIX = "tiny_error"
logger = logging.getLogger(SCRIPT_ID)

# --- Locked TINY package (EXEC 2026-08-13 pre-implementation locks) ------
TINY_BATCH = 8                    # SS10.4 batch
TINY_SELECTION_SEED = 0           # SS10.4 selection seed
TINY_MODEL_INIT_SEED = 0          # SS10.4 model init seed
TINY_STEPS = 500                  # SS10.4 fixed steps
TINY_LR = 1e-4                    # SS10.4 Adam lr
TINY_BETAS = (0.9, 0.999)         # SS10.4
TINY_EPS = 1e-8                   # SS10.4
TINY_WEIGHT_DECAY = 0.0           # SS10.4 no weight decay
B1_NLL_DROP_PER_DIM_MIN = 0.10    # SS10.3 B1 (absolute, per free real dim)
B2_PSNR_DELTA_MIN_DB = 2.0        # SS10.3 B2 clause 1
B2_ANCHOR_FLOOR_DB = -0.5         # SS10.3 B2 clause 2 (final >= anchor-0.5)
B3_NMSE_RATIO_MAX = 0.5           # SS10.3 B3 (final <= 50% initial)
R_FREE_MIN = 1e-10                # SS13/SS10.5 exclusion vs S_ref^2
LATENT_BANK_N = 8                 # SS10.1 secondary bank
LATENT_BANK_SEED = 0              # SS10.1 secondary latent seed
TRACE_CHECKPOINTS = tuple(range(0, TINY_STEPS + 1, 50))  # 0,50,...,500
DATA_RANGE = 1.0                  # SS7.1: normalised-space scoring

# --- IMPL facts dual-pin (the authoritative Class-A artefact this stage --
#     builds on; same doctrine as the ffr parent loaders) ------------------
IMPL_FACTS_SCHEMA = "seqref-impl-facts/1"
IMPL_FACTS_FILE_SHA256 = ("b73edc241111d0ed821ad89fc0f9d9e43c76fefe714592b1"
                          "6e3a8c931d3adfc4")
IMPL_FACTS_SEMANTIC_SHA256 = ("b71d41196e173855acfa2f9c6e7404ec0a86dad98fcb"
                              "388b60d9791168bef77b")

TINY_LOCAL_FILES = [
    "seqref_mri/scripts/tiny_gate.py",
    "seqref_mri/scripts/tiny_selftest.py",
    "seqref_mri/src/free_flow_runtime.py",
    "seqref_mri/scripts/train_free_flow.py",
    "seqref_mri/src/conditioner.py",
    "seqref_mri/src/flows/nsf_layer.py",
    "seqref_mri/src/fastmri_data.py",
    "seqref_mri/src/residual_decoder.py",
    "seqref_mri/src/metrics.py",
    "seqref_mri/src/preflight_parents_p3.py",
    "seqref_mri/scripts/train_base.py",
]


def _fail(code: str, message: str, **kwargs) -> StageError:
    logger.error("[%s] %s: %s", SCRIPT_ID, code, message)
    return StageError(code, message, **kwargs)


# ---------------------------------------------------------------------------
# Parent chain: P0/P0S via the campaign verifier; P3 / P4 /2 / IMPL-B via
# the runtime's dual-pin loaders; IMPL facts via this stage's own dual-pin
# loader (ffr predates the IMPL artefact). The IMPL loader additionally
# re-verifies that the executing runtime/trainer/selftest files match the
# exact code pins recorded inside the Class-A facts.
# ---------------------------------------------------------------------------

def _load_impl_parent(path: str) -> tuple[dict, str]:
    """Dual-pin IMPL Class-A parent. Returns (parsed facts, VERIFIED file
    sha256): the artefact JSON carries no top-level copy of its own byte
    hash (external/sidecar provenance), so the verified value is returned
    alongside for downstream evidence assembly."""
    sha = file_sha256(path)
    if sha != IMPL_FACTS_FILE_SHA256:
        raise _fail("PARENT_FILE_MISMATCH",
                    f"IMPL facts file sha256 {sha} != registered pin "
                    f"{IMPL_FACTS_FILE_SHA256}")
    verify_sidecar(path)
    with open(path, "r", encoding="utf-8") as fh:
        art = json.load(fh)
    if art.get("schema") != IMPL_FACTS_SCHEMA:
        raise _fail("PARENT_SCHEMA_MISMATCH",
                    f"IMPL facts schema {art.get('schema')!r} != "
                    f"{IMPL_FACTS_SCHEMA!r}")
    if art.get("semantic_sha256") != IMPL_FACTS_SEMANTIC_SHA256:
        raise _fail("PARENT_SEMANTIC_MISMATCH",
                    f"IMPL facts semantic sha256 "
                    f"{art.get('semantic_sha256')} != registered pin "
                    f"{IMPL_FACTS_SEMANTIC_SHA256}")
    if art.get("verdict") != "PASS":
        raise _fail("PARENT_NOT_PASS",
                    f"IMPL facts verdict {art.get('verdict')!r} != PASS; "
                    f"TINY requires a closed Class-A gate")
    for rec in art.get("code", {}).get("impl_local", []):
        rel = rec["relpath"]
        on_disk = file_sha256(os.path.join(_REPO, rel))
        if on_disk != rec["sha256"]:
            raise _fail("CODE_PIN_MISMATCH",
                        f"executing file {rel} sha256 {on_disk} != the "
                        f"Class-A pinned {rec['sha256']}; the Class-A "
                        f"validity verdict does not cover this tree state")
    logger.info("[%s] IMPL parent pinned: file %s | semantic %s | code "
                "pins re-verified (%d files)",
                SCRIPT_ID, sha[:12], art["semantic_sha256"][:12],
                len(art.get("code", {}).get("impl_local", [])))
    return art, sha


def _s_ref_from_p0s(p0s_facts_path: str) -> float:
    """S_ref from the ALREADY-VERIFIED P0S artefact (verify_parents ran
    first). SS10.5 exclusions divide by S_ref^2."""
    with open(p0s_facts_path, "r", encoding="utf-8") as fh:
        art = json.load(fh)
    rec = art.get("s_ref")
    if not isinstance(rec, dict) or "value" not in rec:
        raise _fail("PARENT_FIELD_MISSING",
                    "verified P0S artefact lacks the s_ref.value field")
    if rec.get("valid_for_downstream") is not True:
        raise _fail("PARENT_SREF_INVALID",
                    f"P0S s_ref.valid_for_downstream is "
                    f"{rec.get('valid_for_downstream')!r}; the scale must "
                    f"not be consumed")
    s_ref = float(rec["value"])
    if not (np.isfinite(s_ref) and s_ref > 0.0):
        raise _fail("PARENT_SREF_INVALID",
                    f"P0S s_ref value {s_ref!r} is not finite and > 0")
    return s_ref


# ---------------------------------------------------------------------------
# Registered selection: PCG64(0).choice(N, 8, replace=False) over the FULL
# train population; draw order fixed; no P0S exclusion.
# ---------------------------------------------------------------------------

def _select_batch(ds: FastMRISliceDataset) -> dict:
    n = len(ds)
    if n < TINY_BATCH:
        raise _fail("SELECTION_POPULATION_TOO_SMALL",
                    f"train population {n} < TINY_BATCH {TINY_BATCH}")
    rng = np.random.Generator(np.random.PCG64(TINY_SELECTION_SEED))
    draw = rng.choice(n, TINY_BATCH, replace=False)
    draw_idx = [int(i) for i in draw]
    if len(set(draw_idx)) != TINY_BATCH:
        raise _fail("SELECTION_NOT_UNIQUE",
                    f"the registered draw produced duplicates: {draw_idx}")
    if min(draw_idx) < 0 or max(draw_idx) >= n:
        raise _fail("SELECTION_OUT_OF_RANGE",
                    f"draw indices {draw_idx} outside [0, {n})")
    # Portable identity (production convention, fastmri_data.__getitem__):
    # split + path RELATIVE to data_root (POSIX) + slice_index; the
    # dataset_index is recorded/cross-checked, never treated as portable
    # identity. No machine-specific data-root spelling enters the manifest.
    identities = []
    for i in draw_idx:
        path, slice_index = ds.index[i]
        rel = Path(path).relative_to(ds.data_root).as_posix()
        identities.append({
            "split": "train",
            "file": rel,
            "slice_index": int(slice_index),
            "dataset_index": i})
    canonical_sorted = sorted(
        ({k: v for k, v in ident.items() if k != "dataset_index"}
         for ident in identities),
        key=lambda r: (r["file"], r["slice_index"]))
    manifest_sha = canonical_hash({
        "rule": "PCG64(0).choice(N, 8, replace=False); draw order fixed",
        "population": n,
        "draw_order_indices": draw_idx,
        "ordered_identities": identities})
    logger.info("[%s] selection: N=%d draw=%s manifest=%s", SCRIPT_ID, n,
                draw_idx, manifest_sha[:12])
    return {"population": n,
            "draw_order_indices": draw_idx,
            "ordered_identities": identities,
            "canonical_sorted_identities": canonical_sorted,
            "manifest_sha256": manifest_sha,
            "p0s_overlap_rule": "no exclusion; accidental overlap with the "
                                "frozen P0S 256 is permitted and recorded"}


def _p0s_indices_from_artefact(p0s_facts_path: str) -> set[int]:
    """Frozen P0S 256 subset dataset indices from the ALREADY-VERIFIED P0S
    artefact (verify_parents ran first). Consumed ONLY to RECORD accidental
    overlap with the TINY draw -- never to exclude or redraw (EXEC
    2026-08-13 selection lock)."""
    with open(p0s_facts_path, "r", encoding="utf-8") as fh:
        art = json.load(fh)
    rec = art.get("sampling")
    idx = rec.get("canonical_sorted_indices") if isinstance(rec, dict) else None
    if (not isinstance(idx, list) or not idx
            or any(not isinstance(i, int) or isinstance(i, bool)
                   for i in idx)):
        raise _fail("PARENT_FIELD_MISSING",
                    "verified P0S artefact lacks a valid "
                    "sampling.canonical_sorted_indices index list")
    return set(idx)


def _record_p0s_overlap(selection: dict, p0s_indices: set[int]) -> None:
    """EXEC lock: accidental overlap with the frozen P0S 256 is PERMITTED
    AND RECORDED. This records the OBSERVED overlap in the selection
    block; it never excludes, never redraws, never changes the draw."""
    overlap = sorted(i for i in selection["draw_order_indices"]
                     if i in p0s_indices)
    selection["p0s_overlap"] = {
        "count": len(overlap),
        "dataset_indices": overlap,
        "identities": [ident for ident in selection["ordered_identities"]
                       if ident["dataset_index"] in p0s_indices],
        "policy": "observed and recorded; no exclusion, no redraw"}
    logger.info("[%s] P0S overlap observed: %d/%d selected slices also in "
                "the frozen P0S subset (recorded, no exclusion)",
                SCRIPT_ID, len(overlap), TINY_BATCH)


# ---------------------------------------------------------------------------
# Per-slice state: map, standardisation, targets, u_true, exclusion energy.
# Maps are DERIVED from each slice's live mask through the registered P3
# production builder (the TINY draw is independent of the P0S 256, so P3
# bindings cannot be assumed); P4 /2 scaling comes from the pinned
# location_index. Identity cross-check against the selection manifest runs
# BEFORE any metric or optimizer work on the slice.
# ---------------------------------------------------------------------------

def _build_slice_states(batch: dict, selection: dict, p4: dict,
                        s_ref: float) -> list[dict]:
    prep = _prepare(batch, "cpu", test0=False)
    missing = [k for k in tff.REQUIRED_PREPARE_KEYS if k not in prep]
    if missing:
        raise _fail("PREPARE_KEYS_MISSING",
                    f"_prepare result lacks {missing}")
    states = []
    for j, meta in enumerate(batch["meta"]):
        ident = selection["ordered_identities"][j]
        if (str(meta["file"]) != ident["file"]
                or int(meta["slice_index"]) != ident["slice_index"]):
            raise _fail("SELECTION_IDENTITY_MISMATCH",
                        f"loader item {j} is "
                        f"({meta['file']}, {meta['slice_index']}) but the "
                        f"registered draw expects "
                        f"({ident['file']}, {ident['slice_index']})")
        mask = batch["mask"][j]
        acq = np.flatnonzero(mask.to(torch.bool).cpu().numpy())
        cmap = dec.build_coordinate_map([int(c) for c in acq],
                                        ffr.GRID_H, ffr.GRID_W)
        if 2 * int(cmap.free_rows.shape[0]) != ffr.FLOW_DIM_REAL:
            raise _fail("MAP_DIM_MISMATCH",
                        f"slice {j}: map free dim "
                        f"{2 * int(cmap.free_rows.shape[0])} != "
                        f"{ffr.FLOW_DIM_REAL}")
        vecs = ffr.standardisation_vectors(cmap, p4["location_index"])
        x1 = prep["x_norm"][j:j + 1]
        c1 = prep["cond_in"][j:j + 1]
        # u_true in the UNSTANDARDISED physical free representation --
        # the same dx path encode_target standardises (B3 contract).
        dx = torch.complex(x1[0, 0], x1[0, 1]) - torch.complex(c1[0, 0],
                                                               c1[0, 1])
        k_dx = fdm.fft2c(dx)
        u_true = np.asarray(
            dec.gather_unmeasured(k_dx.unsqueeze(0), cmap)
            .detach().to(torch.complex128).cpu().numpy())[0]
        target = ffr.encode_target(x1, c1, cmap, vecs)      # (1, 13824) f64
        energy = float(np.sum(np.abs(u_true) ** 2))
        ratio = energy / (s_ref ** 2)
        x_true_c = torch.complex(x1[0, 0], x1[0, 1])        # (96,96) c64
        states.append({
            "identity": ident,
            "cmap": cmap, "vecs": vecs,
            "cond": c1,                                     # (1,2,96,96)
            "mask": mask.unsqueeze(0),                      # (1,96)
            "y": prep["y"][j:j + 1],                        # (1,96,96) c64
            "amax": prep["amax"][j:j + 1],                  # (1,)
            "x_true_mag": x_true_c.abs(),                   # (96,96)
            "anchor_mag": torch.complex(c1[0, 0], c1[0, 1]).abs(),
            "target": target,
            "u_true": u_true,
            "u_true_energy": energy,
            "u_true_ratio": ratio,
            "excluded": bool(ratio <= R_FREE_MIN)})
    return states


# ---------------------------------------------------------------------------
# Endpoint metric engine. Called exactly twice: step 0 (untouched model)
# and step 500. No checkpoint selection anywhere in this stage.
# ---------------------------------------------------------------------------

def _nll(model, targets, cond, mask) -> float:
    try:
        return float(tff.nll_objective(model, targets, cond, mask))
    except ffr.FreeFlowError as exc:
        raise _fail("TINY_NLL_NON_FINITE", f"{type(exc).__name__}: {exc}")
    except ValueError as exc:
        # The NSF backend raises a bare ValueError("non-finite output in
        # RQ spline") before the runtime gates; map it, re-raise others.
        if "non-finite" in str(exc).lower():
            raise _fail("TINY_NLL_NON_FINITE",
                        f"{type(exc).__name__}: {exc}")
        raise


def _decode_z(model, z, st) -> tuple:
    """One production z decode for slice state st -> (COMPLEX image
    (96,96), unstandardised free vector c128). The complex image is
    returned (not its magnitude) so the secondary estimator can average
    states first -- the registered posterior-mean convention."""
    x_hat = ffr.decode_to_image(model, z, st["cond"], st["mask"],
                                st["y"], st["amax"], st["cmap"],
                                st["vecs"])                 # (1,96,96) c
    us = model.decode_scalars(z, st["cond"], st["mask"])    # (1,13824) f32
    us_np = np.asarray(us.detach().to(torch.float64).cpu().numpy())
    re_s, im_s = ffr.unpack_scalars(us_np)
    u_hat = ffr.unstandardise_free(re_s, im_s, st["cmap"], st["vecs"])[0]
    return x_hat[0], u_hat


def _bank_mean_mag(xs: list[torch.Tensor]) -> torch.Tensor:
    """Posterior-mean magnitude over a latent bank: average the COMPLEX
    reconstructions first, take magnitude ONCE (the production convention,
    train_base._posterior_mean: mean state -> complex -> .abs()). NOT the
    mean of per-sample magnitudes -- for complex posteriors those differ
    (E|x| >= |E x|)."""
    if not xs:
        raise _fail("TINY_BANK_EMPTY",
                    "posterior-mean bank is empty; the fixed bank of "
                    f"{LATENT_BANK_N} latents is registered and mandatory")
    acc = torch.zeros_like(xs[0])
    for x in xs:
        acc += x
    return (acc / len(xs)).abs()


def _psnr(mag_hat: torch.Tensor, mag_true: torch.Tensor) -> float:
    val = float(psnr_per_sample(mag_hat.view(1, 1, *mag_hat.shape),
                                mag_true.view(1, 1, *mag_true.shape),
                                data_range=DATA_RANGE)[0])
    if not np.isfinite(val):
        raise _fail("TINY_METRIC_NON_FINITE",
                    f"PSNR is non-finite ({val!r})")
    return val


def _nmse(u_hat: np.ndarray, u_true: np.ndarray) -> float:
    num = float(np.sum(np.abs(u_hat - u_true) ** 2))
    den = float(np.sum(np.abs(u_true) ** 2))
    val = num / den                       # den > 0: excluded slices never
    if not np.isfinite(val):              # reach here (R_FREE_MIN gate)
        raise _fail("TINY_METRIC_NON_FINITE",
                    f"NMSE_u is non-finite ({val!r})")
    return val


def _endpoint_metrics(model, states: list[dict], s_ref: float,
                      latents: torch.Tensor) -> dict:
    """PRIMARY z=0 metrics (gating) + SECONDARY fixed-bank posterior mean
    (report-only), per slice then mean. Exclusions decided once from
    u_true energy -- identical at both endpoints by construction."""
    model.eval()
    z0 = torch.zeros(1, ffr.FLOW_DIM_REAL)
    targets = torch.from_numpy(np.concatenate(
        [st["target"] for st in states], axis=0).astype(np.float32))
    cond = torch.cat([st["cond"] for st in states], dim=0)
    mask = torch.cat([st["mask"] for st in states], dim=0)
    nll = _nll(model, targets, cond, mask)
    per_slice = []
    agg_num_z0 = 0.0        # aggregate-ratio NMSE accumulators (reported
    agg_num_pm = 0.0        # additionally per SS10.5; never gate B3)
    agg_den = 0.0
    with torch.no_grad():
        for j, st in enumerate(states):
            x0_c, u0 = _decode_z(model, z0, st)
            mag0 = x0_c.abs()
            # secondary: same fixed bank; COMPLEX states averaged first,
            # magnitude once (registered convention); the free-vector
            # mean is linear-space and unchanged.
            xs_bank = []
            u_acc = np.zeros_like(st["u_true"])
            for b in range(LATENT_BANK_N):
                x_b, u_b = _decode_z(model, latents[b:b + 1], st)
                xs_bank.append(x_b)
                u_acc += u_b
            mag_pm = _bank_mean_mag(xs_bank)
            u_pm = u_acc / LATENT_BANK_N
            rec = {
                "identity": st["identity"],
                "psnr_z0": _psnr(mag0, st["x_true_mag"]),
                "psnr_anchor": _psnr(st["anchor_mag"], st["x_true_mag"]),
                "psnr_posterior_mean": _psnr(mag_pm, st["x_true_mag"]),
                "nmse_u_z0": None, "nmse_u_posterior_mean": None,
                "u_true_ratio": st["u_true_ratio"],
                "excluded": st["excluded"]}
            if not st["excluded"]:
                rec["nmse_u_z0"] = _nmse(u0, st["u_true"])
                rec["nmse_u_posterior_mean"] = _nmse(u_pm, st["u_true"])
                agg_num_z0 += float(np.sum(np.abs(u0 - st["u_true"]) ** 2))
                agg_num_pm += float(np.sum(np.abs(u_pm - st["u_true"])
                                             ** 2))
                agg_den += float(np.sum(np.abs(st["u_true"]) ** 2))
            per_slice.append(rec)
    included = [r for r in per_slice if not r["excluded"]]
    if not included:
        raise _fail("TINY_ALL_SLICES_EXCLUDED",
                    "every slice fell under the R_FREE_MIN exclusion; no "
                    "valid B3 measurement can be formed")
    if not (np.isfinite(agg_den) and agg_den > 0.0):
        raise _fail("TINY_METRIC_NON_FINITE",
                    f"aggregate NMSE denominator {agg_den!r} is not "
                    f"finite and > 0")
    mean = lambda key, rows: float(np.mean([r[key] for r in rows]))  # noqa: E731
    return {
        "nll_batch_mean": nll,
        "per_slice": per_slice,
        "mean_psnr_z0": mean("psnr_z0", per_slice),
        "mean_psnr_anchor": mean("psnr_anchor", per_slice),
        "mean_psnr_posterior_mean": mean("psnr_posterior_mean", per_slice),
        "excluded_count": len(per_slice) - len(included),
        "excluded_identities": [r["identity"] for r in per_slice
                                if r["excluded"]],
        "mean_nmse_u_z0": mean("nmse_u_z0", included),
        "mean_nmse_u_posterior_mean": mean("nmse_u_posterior_mean",
                                           included),
        "aggregate_nmse_u_z0": agg_num_z0 / agg_den,
        "aggregate_nmse_u_posterior_mean": agg_num_pm / agg_den,
    }


# ---------------------------------------------------------------------------
# B1-B3 evaluation (EXEC SS10.3; thresholds frozen 2026-08-13, never
# revised against an observed result).
# ---------------------------------------------------------------------------

def _evaluate_gates(m0: dict, m500: dict) -> dict:
    b1 = (m0["nll_batch_mean"] - m500["nll_batch_mean"]) / ffr.FLOW_DIM_REAL
    b1_pass = bool(np.isfinite(b1) and b1 >= B1_NLL_DROP_PER_DIM_MIN)
    b2_delta = m500["mean_psnr_z0"] - m0["mean_psnr_z0"]
    b2_c1 = bool(b2_delta >= B2_PSNR_DELTA_MIN_DB)
    b2_c2 = bool(m500["mean_psnr_z0"]
                 >= m500["mean_psnr_anchor"] + B2_ANCHOR_FLOOR_DB)
    b2_pass = b2_c1 and b2_c2
    # B3 gates the REGISTERED inequality final <= 0.5 * initial directly.
    # This stays mathematically defined at initial == 0 (final must then
    # also be 0); a ratio form would divide by zero. The ratio is
    # report-only and null when initial == 0.
    b3_initial = m0["mean_nmse_u_z0"]
    b3_final = m500["mean_nmse_u_z0"]
    b3_limit = B3_NMSE_RATIO_MAX * b3_initial
    b3_pass = bool(np.isfinite(b3_initial) and np.isfinite(b3_final)
                   and b3_final <= b3_limit)
    b3_ratio = (float(b3_final / b3_initial)
                if np.isfinite(b3_initial) and b3_initial > 0.0 else None)
    failed = []
    if not b1_pass:
        failed.append("B1")
    if not b2_pass:
        failed.append("B2")
    if not b3_pass:
        failed.append("B3")
    return {"b1_nll_drop_per_dim": b1,
            "b1_threshold": B1_NLL_DROP_PER_DIM_MIN, "b1_pass": b1_pass,
            "b2_psnr_delta_db": b2_delta,
            "b2_delta_threshold_db": B2_PSNR_DELTA_MIN_DB,
            "b2_clause_delta_pass": b2_c1,
            "b2_final_psnr_db": m500["mean_psnr_z0"],
            "b2_anchor_psnr_db": m500["mean_psnr_anchor"],
            "b2_anchor_floor_db": B2_ANCHOR_FLOOR_DB,
            "b2_clause_anchor_pass": b2_c2,
            "b2_pass": b2_pass,
            "b3_initial_nmse_u": b3_initial,
            "b3_final_nmse_u": b3_final,
            "b3_limit": b3_limit,
            "b3_nmse_ratio": b3_ratio,
            "b3_ratio_note": "report-only diagnostic; the gate is the "
                             "registered inequality final <= threshold * "
                             "initial; ratio is null when initial == 0",
            "b3_threshold": B3_NMSE_RATIO_MAX, "b3_pass": b3_pass,
            "failed_gates": failed,
            "verdict": "PASS" if not failed else "BLOCK"}


def _assert_endpoint_stability(m0: dict, m500: dict) -> None:
    """Endpoint invariants, EXACT (not count/mean-level): the data-only
    per-slice anchor PSNR vector and the R_FREE_MIN exclusion identity set
    are endpoint-independent by construction; any drift means the metric
    path is state-contaminated. Equal counts or equal means are NOT
    sufficient -- different sets/vectors can share them."""
    anchor0 = [r["psnr_anchor"] for r in m0["per_slice"]]
    anchor500 = [r["psnr_anchor"] for r in m500["per_slice"]]
    if anchor0 != anchor500:
        raise _fail("ANCHOR_DRIFT",
                    "the per-slice data-only anchor PSNR vector differs "
                    "between endpoints; the metric path is "
                    "state-contaminated")
    if m0["excluded_identities"] != m500["excluded_identities"]:
        raise _fail("EXCLUSION_DRIFT",
                    "the R_FREE_MIN exclusion identity set changed between "
                    "endpoints; it is defined once from u_true energy and "
                    "cannot drift")


# ---------------------------------------------------------------------------
# Facts assembly (seqref-tiny-facts/1) + entry point
# ---------------------------------------------------------------------------

def _path_free(rec):
    if isinstance(rec, dict):
        return {k: _path_free(v) for k, v in rec.items() if k != "path"}
    if isinstance(rec, list):
        return [_path_free(v) for v in rec]
    return rec


def _code_record() -> dict:
    code = dict(hash_project_code(_REPO, os.path.abspath(__file__)))
    hashed = []
    for rel in TINY_LOCAL_FILES:
        path = os.path.join(_REPO, rel)
        if not os.path.isfile(path):
            raise _fail("CODE_HASH_FILE_MISSING",
                        f"project-local file required for the TINY code "
                        f"hash is missing: {rel}")
        hashed.append({"relpath": rel, "sha256": file_sha256(path)})
    code["tiny_local"] = hashed
    code["tiny_local_note"] = (
        "the TINY stage's own driver plus every module the production "
        "path executes; the frozen project hash block covers the "
        "preflight core")
    return code


def _build_facts(selection: dict, m0: dict, m500: dict, trace: dict,
                 gates: dict, parents: dict, p3: dict, p4: dict,
                 implb: dict, impl: dict, impl_file_sha: str,
                 s_ref: float) -> dict:
    parents_rec = {
        "parents_id": parents.get("parents_id"),
        "p0": _path_free(parents.get("p0")),
        "p0s": _path_free(parents.get("p0s")),
        "p3_coordinate_map": _path_free({k: v for k, v in p3.items()
                                         if k != "bindings"}),
        "p4_scaling_statistics": _path_free({k: v for k, v in p4.items()
                                             if k != "location_index"}),
        "implb_calibration": _path_free(implb),
        "impl_class_a": {"schema": impl["schema"],
                         "file_sha256": impl_file_sha,
                         "semantic_sha256": impl["semantic_sha256"],
                         "verdict": impl["verdict"]},
        "s_ref": {"value": s_ref, "source": "verified P0S artefact",
                  "used_for": "R_FREE_MIN exclusion ratios only"},
    }
    locked_config = {
        "population": "FULL train split, FastMRISliceDataset mode=eval; "
                      "no P0S exclusion (overlap permitted and recorded)",
        "selection": "numpy Generator(PCG64(0)).choice(N, 8, "
                     "replace=False); draw order = training/evaluation "
                     "order",
        "batch": TINY_BATCH, "steps": TINY_STEPS,
        "model_init_seed": TINY_MODEL_INIT_SEED,
        "optimizer": f"Adam(lr={TINY_LR}, betas={TINY_BETAS}, "
                     f"eps={TINY_EPS}, weight_decay={TINY_WEIGHT_DECAY}); "
                     f"no schedule",
        "precision": "float32 production forward/training; float64 "
                     "metrics",
        "b1_rule": "(batch-mean NLL@0 - batch-mean NLL@500) / 13824 "
                   ">= 0.10; production nll_objective reduction, divided "
                   "exactly once",
        "b2_rule": "z=0 per-slice PSNR then mean (data_range=1.0); "
                   "final-initial >= 2.0 dB AND final >= anchor-0.5 dB",
        "b3_rule": "z=0 production inverse, unstandardised physical free "
                   "coordinates; per-slice NMSE_u then mean over included; "
                   "final <= 0.5*initial",
        "secondary": f"posterior mean over one fixed latent bank "
                     f"(n={LATENT_BANK_N}, seed={LATENT_BANK_SEED}), "
                     f"identical bank at both endpoints; report-only",
        "endpoints": "step 0 (untouched model) and step 500 only; no "
                     "checkpoint selection",
    }
    verdict = gates["verdict"]
    facts = {
        "schema": FACTS_SCHEMA,
        "script": {"id": SCRIPT_ID, "version": SCRIPT_VERSION,
                   "lifetime": "KEEP"},
        "stage": STAGE,
        "artefact_type": "stage_facts",
        "run_mode": "authoritative",
        "authoritative": True,
        "locked_config": locked_config,
        "selection": selection,
        "nll_trace": trace,
        "endpoints": {"initial": m0, "final": m500},
        "gates": gates,
        "verdict": verdict,
        "verdict_reason": (
            "B1-B3 all meet their registered thresholds; tiny-batch "
            "model compatibility established; PILOT remains a separate "
            "stage" if verdict == "PASS" else
            "a valid TINY run completed but gates "
            f"{gates['failed_gates']} failed their registered thresholds; "
            "tiny-batch compatibility failure -- a hard stop before the "
            "pilot, NOT automatically a coding bug; no threshold "
            "revision"),
        "summary": {
            "gates": ["B1", "B2", "B3"],
            "gates_passed": 3 - len(gates["failed_gates"]),
            "gates_failed": list(gates["failed_gates"]),
            "exit_rule": "PASS exit 0 | BLOCK exit 1 (facts published "
                         "with measurements; no threshold revision) | "
                         "ERROR exit 2 (typed error record, no facts)",
            "plots": "none (machine-readable evidence only)"},
        "parents": parents_rec,
        "dataset_provenance": {
            "split": "train", "mode": "eval",
            "population": selection["population"],
            "selection_rule": "PCG64(0) draw, registered 2026-08-13",
            "map_rule": "per-slice coordinate maps derived from live "
                        "masks through the registered P3 production "
                        "builder; identity cross-checked against the "
                        "selection manifest BEFORE any metric or "
                        "optimizer work"},
        "code": _code_record(),
        "run": {**environment_record(_REPO, sys.argv),
                "hash_note": "file sha256 + sidecar; semantic sha256 "
                             "over the path-free semantic payload (run/ "
                             "excluded as volatile)"},
    }
    semantic = {k: v for k, v in facts.items() if k != "run"}
    attach_semantic_hash(facts, semantic)
    return facts


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SEQREF-TINY v0.1 tiny-batch model-compatibility gate "
                    "(B1-B3); publishes tiny/tiny_facts.json "
                    "(seqref-tiny-facts/1) on PASS and BLOCK alike; ERROR "
                    "writes a typed error record instead")
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", default=os.path.join(
        _REPO, "seqref_mri", "results", "_diag", "tiny"))
    p.add_argument("--p0-facts", default=None)
    p.add_argument("--p0s-facts", default=None)
    p.add_argument("--p0s-script", default=None)
    p.add_argument("--p3-facts", default=None)
    p.add_argument("--p4-stats2", default=None)
    p.add_argument("--implb-facts", default=None)
    p.add_argument("--impl-facts", default=None)
    p.add_argument("--log-file", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode="w",
                                            encoding="utf-8"))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s",
                        handlers=handlers, force=True)
    if os.path.realpath(args.repo_dir) != os.path.realpath(_REPO):
        print(f"--repo-dir {args.repo_dir!r} does not resolve to the "
              f"installation {_REPO!r}; refusing to run against a copy",
              file=sys.stderr)
        return EXIT_ERROR
    torch.set_num_threads(1)
    parents = None
    try:
        required = (args.p0_facts, args.p0s_facts, args.p0s_script,
                    args.p3_facts, args.p4_stats2, args.implb_facts,
                    args.impl_facts)
        if not all(required):
            raise _fail(
                "PARENT_INPUT_MISSING",
                "the TINY gate requires --p0-facts, --p0s-facts, "
                "--p0s-script, --p3-facts, --p4-stats2, --implb-facts "
                "and --impl-facts so the complete parent chain is "
                "verified, not assumed",
                detail={}, write_record=False)
        parents = verify_parents(_REPO, args.p0_facts, args.p0s_facts,
                                 args.p0s_script)
        p3 = ffr.load_p3_parent(args.p3_facts)
        p4 = ffr.load_p4s2_parent(args.p4_stats2)
        implb = ffr.load_implb_parent(args.implb_facts)
        impl, impl_file_sha = _load_impl_parent(args.impl_facts)
        s_ref = _s_ref_from_p0s(args.p0s_facts)

        ds = FastMRISliceDataset(args.data_root, split="train",
                                 mode="eval")
        selection = _select_batch(ds)
        _record_p0s_overlap(selection,
                            _p0s_indices_from_artefact(args.p0s_facts))
        loader = DataLoader(Subset(ds, selection["draw_order_indices"]),
                            batch_size=TINY_BATCH, shuffle=False,
                            collate_fn=_collate)
        batch = next(iter(loader))
        states = _build_slice_states(batch, selection, p4, s_ref)

        model = ffr.build_model(spline_b=implb["spline_b"],
                                init_seed=TINY_MODEL_INIT_SEED)
        latents = torch.randn(
            LATENT_BANK_N, ffr.FLOW_DIM_REAL,
            generator=torch.Generator().manual_seed(LATENT_BANK_SEED))
        targets = torch.from_numpy(np.concatenate(
            [st["target"] for st in states], axis=0).astype(np.float32))
        cond = torch.cat([st["cond"] for st in states], dim=0)
        mask = torch.cat([st["mask"] for st in states], dim=0)

        # Endpoint 1: the UNTOUCHED step-0 model.
        m0 = _endpoint_metrics(model, states, s_ref, latents)
        trace = {"0": m0["nll_batch_mean"]}

        optimizer = torch.optim.Adam(model.parameters(), lr=TINY_LR,
                                     betas=TINY_BETAS, eps=TINY_EPS,
                                     weight_decay=TINY_WEIGHT_DECAY)
        model.train()
        for step in range(1, TINY_STEPS + 1):
            try:
                tff.train_step(model, optimizer, targets, cond, mask)
            except ffr.FreeFlowError as exc:
                raise _fail("TINY_NLL_NON_FINITE",
                            f"step {step}: {type(exc).__name__}: {exc}")
            except ValueError as exc:
                if "non-finite" in str(exc).lower():
                    raise _fail("TINY_NLL_NON_FINITE",
                                f"step {step}: {type(exc).__name__}: "
                                f"{exc}")
                raise
            if step in TRACE_CHECKPOINTS:
                trace[str(step)] = _nll(model, targets, cond, mask)

        # Endpoint 2: after exactly 500 steps. Nothing in between gates.
        m500 = _endpoint_metrics(model, states, s_ref, latents)
        _assert_endpoint_stability(m0, m500)

        gates = _evaluate_gates(m0, m500)
        facts = _build_facts(selection, m0, m500, trace, gates, parents,
                             p3, p4, implb, impl, impl_file_sha, s_ref)
        path, sha = publish_stage(facts, args.out_dir, FACTS_PREFIX,
                                  STAGE)
        if gates["verdict"] == "PASS":
            logger.info("[%s] verdict PASS; published %s sha256=%s",
                        SCRIPT_ID, path, sha)
            return EXIT_PASS
        logger.info("[%s] verdict BLOCK (failed gates %s); a valid TINY "
                    "run completed and its measurements are published at "
                    "%s sha256=%s; NO threshold revision",
                    SCRIPT_ID, gates["failed_gates"], path, sha)
        return EXIT_BLOCK
    except StageError as exc:
        logger.error("[%s] %s: %s", SCRIPT_ID, exc.error_code, exc.reason)
        publish_error(exc, args.out_dir, ERROR_PREFIX, STAGE,
                      parents=parents)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 -- the registered boundary
        logger.exception("[%s] unexpected runtime failure", SCRIPT_ID)
        wrapped = StageError(
            "UNEXPECTED_RUNTIME_ERROR",
            f"{type(exc).__name__}: {exc}",
            detail={"exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "raised_after_parent_verification":
                        parents is not None},
            write_record=parents is not None)
        logger.error("[%s] %s: %s", SCRIPT_ID, wrapped.error_code,
                     wrapped.reason)
        publish_error(wrapped, args.out_dir, ERROR_PREFIX, STAGE,
                      parents=parents)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
