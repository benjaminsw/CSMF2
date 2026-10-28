# SEQREF-MRI-V05LTEVAL v0.9 -- scripts.v05lt_eval
# LIFETIME: KEEP
# =============================================================================
# Purpose: V05LT evaluator (194 §8, AM3, AM4; 195 §6) -- delivery I1.7(a)
#   (SEQREF-MRI-V05LTI17 v0.2):
#   * readouts on fixed slice states (tiny_gate._build_slice_states):
#       R0 -- z = 0 through tiny_gate._decode_z (as the shared validation);
#       R1 -- the registered posterior mean: ONE batched decode of the
#             128-row bank (d3._decode_bank_conditioned), complex mean of
#             the physical free vectors BEFORE image formation, ONE image
#             formation (estimators.image_from_u) -- the V02G v0.2 path;
#       prefix stability 16/32/64/128 from the SAME decodes of the PRIMARY
#       bank only (no extra decodes; banks B/C report k = 128); per slice: production PSNR + raw MSE + cap/zero flags
#       (v05lt_metrics.psnr_record), SSIM (float64), NMSE_u (eligible
#       slices on validation; all slices under the V02 holdout rule);
#   * schedule and reuse (plan_campaign): epoch 0 once per seed (both arms
#       must carry the SAME epoch-0 model-state hash); primary bank at
#       epoch 0 / 3 / selected / final; banks B/C and prefix stability at
#       selected only; checkpoints with the same model-state hash are
#       evaluated once (roles share the result). The registered campaign
#       without checkpoint coincidences gives 21 checkpoints / 409,728
#       decodes; coincidences reduce these totals;
#   * F10: strict input verification (verify_f10_inputs) and the D3-subset
#       PM reproduction on the V02 checkpoints ONLY (f10_reproduce), run in
#       a separate process under the historical V02G deterministic backend
#       (v02_readout_common.pin_deterministic_backend). Holdout contact is
#       the exception registered in 195 §6; no V05LT checkpoint is touched.
#   * failure classification (pre-pin item 18, EVALUATOR part only): errors
#       from the reused decode/image seams, from data loading/collation
#       (EVAL_DATA_LOAD) and from the shared metric calls made here are
#       mapped to typed failed_validity codes; non-finite values ->
#       EVAL_NON_FINITE. The shared metrics module's own validate() and
#       the trainer are NOT changed here (separate item-18 follow-up).
#   * evaluate_states() restores the model's previous train/eval mode
#       (try/finally); evaluator isolation itself is pending (I1.7(c)).
#   * (b) hard DC (194 §8 + AM1): x_DC = F^-1(M.y + (1 - M).F(x)) on the
#       COMPLEX image (production fft2c/ifft2c, column mask, y / amax
#       scaling, evaluation dtype complex128), magnitude afterwards;
#       measures (a)-(c); calibration on complex validation ground truth
#       with each image's own acquired values -> per-measure floor = max
#       over slices, tolerance = 10 x floor, with provenance; zero /
#       non-finite denominators or floors stop (DC_ZERO_DENOMINATOR,
#       DC_NON_FINITE, DC_FLOOR_INVALID) -- no epsilon; AM1 corruption
#       tests (pre-DC repair, post-DC detection; DC_DETECTION_FAILED);
#       four readouts R0, R1, R0+DC, R1+DC on identical states, R1+DC from
#       the existing R1 decodes (primary bank, k = 128);
#   * (b) zero-filled reference (AM3-D): image_from_u(u = 0); PSNR, SSIM,
#       eligible-slice NMSE_u (= 1 by construction, checked); no NLL;
#       non-finite -> EPOCH0_NONFINITE (AM3-E3); rows carry slice
#       identities; summary slice -> volume mean -> equal volume weights
#       (194 §8.2); strict-JSON, atomic write-once pair (staging + rename;
#       ZF_EXISTS / ZF_STAGING_PRESENT / ZF_INCOMPLETE) that refuses
#       results/v05lt/zf without the gate-2a flag;
#   * (b) holdout NMSE_u (AM4-B): no threshold exclusion; exactly zero
#       free-space energy -> VALIDITY_NMSE_UNDEFINED; plain mean.
#   * (c1) epoch0_evaluate (AM3-C): all-slice R0/SSIM/NMSE_u/NLL and
#       midpoint R0, R1, R0+DC, R1+DC in eval()/no-grad, RNG states saved
#       and restored, parameters checked unchanged, NaN/Inf ->
#       EPOCH0_NONFINITE, per-stage timing (addition C); panels (194
#       §10.1) reuse the readout images (no extra decodes), fixed
#       model-independent scales, DC columns only when not identical;
#       write_epoch0_eval: write-once directory (strict JSON, PNG + .npz,
#       SHA256SUMS), registered path refused unless production=True;
#       publish_noreplace: renameat2(RENAME_NOREPLACE), no fallback
#       (item 22), also used by the ZF writer.
#   Later delivery: (c2) item 18 trainer / shared-metrics classification.
#   Registered DC calibration values are produced at preflight (P5) and the
#   registered ZF pass at gate 2a -- not by this delivery or its fixtures.
# Production entry is DISABLED in this version: main() refuses
#   (EVAL_PRODUCTION_DISABLED); fixtures call the functions in temporary
#   folders (and F10 on the V02 checkpoints only).
# Fixture-only arguments: `bank_override`, `cross_check_code=False`,
#   `hash_checkpoints=False`, `v02g_contracts=False` (synthetic evidence
#   sets, whose content cannot carry V02G's real constants); production
#   passes none. Declared, not a fallback.
#   * post-run evaluation of trained checkpoints (v0.8; I1.8 b0, G2):
#     load_trained_checkpoint (file sha256, weights schema, saved meta,
#     model-state hash -> VerifiedCheckpoint; v0.9),
#     posteval_evaluate (one task of plan_campaign = one distinct model
#     state, deduplicated across its roles; midpoint readouts R0/R1/DC on
#     the task's banks and prefixes; panels) and write_posteval (write-once,
#     renameat2 no-overwrite, registered results/v05lt/posteval/<model-
#     state> refused without production=True). The record binds the output
#     to the checkpoint (model-state and file hash, epoch, roles), banks
#     (array sha256 = registered), masks (convention + sha256 of the
#     evaluated masks), slice population (identities sha256, eligibility
#     manifest sha256), code (file sha256 of eval/metrics/common + versions)
#     and backend (torch/NumPy/Python versions, threads, deterministic).
# CONVENTION: logger.error + V05LTError(code); no fallback, no placeholder.
# Changelog (v0.9, 2026-10-24; b0 review fixes):
#   * load_trained_checkpoint validates the weights schema and the saved
#     meta (seed, arm, epoch, model-state) against the committed values and
#     returns a VerifiedCheckpoint; posteval_evaluate binds epoch and run
#     identity to it (no caller-built checkpoint dictionary).
#   * write_posteval requires panels = declared cases exactly; load_posteval
#     verifies structure, panel files per case and panel hashes; malformed
#     documents -> POSTEVAL_FILE_INVALID (no raw exceptions).
# Changelog (v0.8, 2026-10-24; I1.8 b0, G2):
#   * Post-run trained-checkpoint evaluation: verified loader, evaluation
#     with full provenance, write-once publication. Epoch-0, ZF and readout
#     functions unchanged.
# Changelog (v0.7, 2026-10-22; I1.7(c1) review fixes):
#   * Bug fix: writers no longer resolve the final destination component
#     (dest_path); any existing entry incl. a dangling symlink is refused
#     (lexists) at destination and staging; RENAME_NOREPLACE retained.
#   * Duplicate panel case numbers / identities refused (PANEL_CASE_DUPLICATE).
#   * PNG labels: case identity, numeric scale endpoints, column and bar
#     names (5x7 bitmap font; unsupported chars / overflow refused).
# Changelog (v0.6, 2026-10-22; I1.7(c1), SEQREF-MRI-V05LTI17C v0.1):
#   * publish_noreplace (renameat2 NOREPLACE); ZF writer publishes with it,
#     so an empty directory created at the destination is never replaced.
#   * epoch0_evaluate, build_panel / panel_files / png_gray,
#     write_epoch0_eval; evaluate_states keep_images for panel cases.
#   * Epoch-0 stages map non-finite failures to EPOCH0_NONFINITE and other
#     exceptions to EPOCH0_EVAL_FAILED (typed).
# Changelog (v0.5, 2026-10-22; I1.7(b) review corrections):
#   * ZF rows carry volume/file/slice/eligible/midpoint; zf_summary now
#     volume-weighted (194 §8.2) with NMSE null volumes (AM2-C).
#   * ZF writer: allow_nan=False; data + sidecar staged and renamed as one
#     directory; interruption/refusal behaviour defined; verify_zf added.
#   * Tolerances validated (finite float > 0) before identity, AM1 tests
#     and four-readout evaluation (DC_TOLERANCE_INVALID).
#   * Measure (b) kept as registered in 194 §8 (denominator
#     ||(1 - M) . F(x)||_2) -- unchanged.
# Changelog (v0.4, 2026-10-22; I1.7(b), SEQREF-MRI-V05LTI17 v0.2):
#   * Hard DC, measures (a)-(c), calibration with provenance and explicit
#     stops, AM1 corruption tests, four readouts (R0, R1, R0+DC, R1+DC).
#   * Zero-filled reference rows/summary and gate-2a-guarded writer.
#   * Holdout NMSE_u path (AM4-B); build_states adds complex ground truth
#     and complex conditioning input, checked bitwise against TINY's state.
# Changelog (v0.3, 2026-10-22; T7 verifier fix -- no evidence changed):
#   * Facts semantic check follows the HISTORICAL procedure exactly: payload
#     = published facts minus V02G's own SEMANTIC_EXCLUDED and minus the
#     post-hash `semantic_scope` (written by attach_semantic_hash AFTER
#     hashing, with the generic preflight_parents list -- a known two-list
#     inconsistency, recorded, never used for the expected keys);
#     preflight_io.canonical_hash. The recomputed hash must equal both the
#     stored and the registered value; included_keys must equal the
#     procedure's key list exactly (no broadened exclusions accepted).
#   * historical_semantic_procedure(): on the host, the exclusion list and
#     the hashing functions are taken from the 6ffce4e5 sources (git show)
#     and must be identical to the working tree (F10_PROCEDURE_DRIFT).
#   * Registry gains `semantic_excluded` (checked against the code).
# Changelog (v0.2, 2026-10-22; reviewer corrections before the host run):
#   * evaluate_states: equal-length states/eligibility required
#     (EVAL_LENGTH_MISMATCH); model mode restored; prefixes on the primary
#     bank only.
#   * plan_campaign: one shared epoch-0 hash per seed (PLAN_EPOCH0_PAIRING);
#     budget described as the no-coincidence plan.
#   * typed classification extended to data loading/collation and metric
#     calls (EVAL_DATA_LOAD, EVAL_NON_FINITE, EVAL_METRIC_INVALID).
#   * train-record wording: sidecar + checkpoint state hashes bound to the
#     pinned facts (no independently approved train-record file hash).
# Changelog (NEW in v0.1, 2026-10-21):
#   * Introduced under I1.7(a): R0/R1 readouts with prefix stability,
#     schedule/reuse planner, strict F10 input verifier, F10 reproduction,
#     typed failure classification for the evaluator.
# =============================================================================
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

from seqref_mri.scripts import v05lt_common as C
from seqref_mri.scripts import v05lt_metrics as M

logger = logging.getLogger("seqref_mri.v05lt_eval")

__version__ = "0.9"
__abbr__ = "SEQREF-MRI-V05LTEVAL"

BANK_ROWS = 128
PREFIXES = (16, 32, 64, 128)
ROLES = ("epoch0", "epoch3", "selected", "final")
EXTRA_BANKS = ("B", "C")
NMSE_RULES = ("validation_eligible", "v02_holdout_all")

# ---------------------------------------------------------------------------
# F10 registry: approved values and where they are registered.
#   facts/semantic/commit/sibling: V02RD v0.3 (LudWiki 155); anchors, bank,
#   manifest and V02H pins: V02G constants (v02_readout_diag, unchanged
#   since 6ffce4e5 except fixture-only v0.3 additions -- documented diff
#   v05lt_i17_v02g_pin_diff.txt 1a87393d...); v02_facts: H8; P0S: H5;
#   holdout manifest file: 195 §3; P4/IMPL-B: production loader pins.
#   The train record has NO independently approved file hash. It is
#   verified only through (i) its sidecar (file bytes = recorded sha256)
#   and (ii) its checkpoint state hashes, which must equal those bound in
#   the pinned V02G facts (frozen_inputs.checkpoint_state_sha256_verified);
#   checkpoint files are hashed against the record's file_sha256.
# ---------------------------------------------------------------------------
REG_F10 = {
    "facts_rel": "seqref_mri/results/v02/readout_diag/v02_readout_facts.json",
    "facts_sha256": "0d9ed10c480641cdd1ffe2e260e51fbba9d694dcde79afbb0322740cf257fc98",
    "facts_semantic_sha256": "92b2ee1ff449d7a4e561945d6de98667f343bd52674cab017a98b1ed6a06df59",
    "facts_schema": "seqref-v02rd-facts/1",
    "facts_commit": "6ffce4e555f90a48daa5fbff8b841adc3dfd7b18",
    "v02_facts_rel": "seqref_mri/results/v02/eval/v02_facts.json",
    "v02_facts_sha256": "907e877c8b4bde0bf979c8add2115c5fd899031fdd259621cc6ae8320710bc0b",
    "manifest_dir_rel": "seqref_mri/results/v02/manifests",
    "holdout_manifest_file_sha256": "0f2f96073fad6bdaad1314ae1319b643a1b67c06424e757f1fea3cfc1e002019",
    "holdout_manifest_sha256": "ce27550cdcdfa627372a53d51df6c7a97e4c6cff0f5a80622e0fa110b0eea165",
    "d3_manifest_sha256": "66647202aead2edbbbea338194f464a44dc77b63e65824a56ff2998f40879bcd",
    "train_record_rel": "seqref_mri/results/v02/train/v02_train_record.json",
    "run_root_rel": "seqref_mri/results/v02/train",
    "p0s_rel": "seqref_mri/results/_diag/residual_preflight/normalisation_scale_facts.json",
    "p0s_sha256": "2070cc72c977096f731811513d81b2abdb5dababb69edb4e997a5f8283b7c693",
    "p4_rel": "seqref_mri/results/_diag/p4/scaling_statistics.json",
    "p4_sha256": "db0bf2c2e0cd3bd5e1ed86bf208198199f16df5b2fbbe0cc72903c8791f85684",
    "implb_rel": "seqref_mri/results/_diag/impl/implb_facts.json",
    "implb_sha256": "fb7a5de70cb4b1e747728d4833a02dc135bdd4e0bd8ad9509e3ac5492e4e1b2c",
    "v02h_rel": "seqref_mri/results/_diag/v02_readout_selftest/v02_readout_selftest.json",
    "v02h_sha256": "c9adf1978a120b2ef4f2553fbf29adb48b43227f71e80191eb344527f1dad3d9",
    "bank_sha256": "f8a6a34b6b474286359e4c99341de256b62bc0dff1e6bc8cf52e72f0461dc8d6",
    "bank_manifest_sha256": "c4d2c0ec7c908f7c20a4f287eeffbdf966924a5118b6ba734b0a329942ab7e5f",
    "steps": [0, 3258],
    "anchors": {"step0": {"psnr": 30.794137818416605, "nmse_u": 1.012275318293053},
                "final": {"psnr": 30.786611478861552, "nmse_u": 1.0141107234837616}},
    "psnr_tol_db": 1e-3,
    "nmse_tol_rel": 1e-4,
    "n_d3": 32,
    # V02G v0.2 SEMANTIC_EXCLUDED (6ffce4e5; identical at HEAD -- pin diff
    # 1a87393d... is fixture-only): the keys removed before hashing.
    "semantic_excluded": ["run", "timing", "r1_sampling_diagnostics",
                          "semantic_sha256", "artifact_sha256"],
}
# Keys added to the facts AFTER semantic hashing (attach_semantic_hash),
# hence never part of the payload.
POST_HASH_KEYS = ("semantic_sha256", "semantic_scope")
PROCEDURE_SOURCES = (
    ("seqref_mri/scripts/v02_readout_diag.py", ("SEMANTIC_EXCLUDED",
                                                "_publish_facts")),
    ("seqref_mri/src/preflight_parents.py", ("attach_semantic_hash",)),
    ("seqref_mri/src/preflight_io.py", ("canonical_bytes", "canonical_hash")),
)
REG_KEYS = tuple(REG_F10)


# ---------------------------------------------------------------------------
# Production seams (lazy) and typed failure classification (item 18)
# ---------------------------------------------------------------------------
def _seams():
    try:
        from seqref_mri.tdiag import _bootstrap  # noqa: F401 -- registered path setup
        import torch
        from seqref_mri.scripts import tiny_gate as tg
        from seqref_mri.tdiag import d3, estimators
        from seqref_mri.src import free_flow_runtime as ffr
        from seqref_mri.src.preflight_parents import StageError
    except ImportError as exc:
        C.fail("EVAL_IMPORT_FAILED", str(exc))
    return torch, tg, d3, estimators, ffr, StageError


def _classified(where: str, fn, *args, **kwargs):
    """Run a reused seam; any exception becomes a typed failed_validity
    error with a stable code (no silent pass, no generic error)."""
    torch, tg, d3, est, ffr, StageError = _seams()
    try:
        return fn(*args, **kwargs)
    except C.V05LTError:
        raise
    except StageError as exc:
        code = str(getattr(exc, "error_code", "STAGE_ERROR"))
        C.fail("EVAL_NON_FINITE" if "NON_FINITE" in code
              else f"EVAL_SEAM_{code}", f"{where}: {code}: {exc}")
    except ffr.FreeFlowError as exc:
        code = str(getattr(exc, "code", "FREEFLOW_ERROR"))
        C.fail("EVAL_NON_FINITE" if "NON_FINITE" in code
              else f"EVAL_SEAM_{code}", f"{where}: {code}: {exc}")
    except (ValueError, TypeError, RuntimeError, KeyError, IndexError) as exc:
        code = "EVAL_NON_FINITE" if "finite" in str(exc).lower() \
            else "EVAL_SEAM_ERROR"
        C.fail(code, f"{where}: {type(exc).__name__}: {exc}")


def _finite_array(a, where: str) -> None:
    if not np.all(np.isfinite(a)):
        C.fail("EVAL_NON_FINITE", f"{where}: non-finite values")


# ---------------------------------------------------------------------------
# States and readouts
# ---------------------------------------------------------------------------
def build_states(view, indices, p4: dict, s_ref: float) -> list:
    """Fixed slice states for view positions (evaluation masks)."""
    torch, tg, *_ = _seams()
    from seqref_mri.scripts.train_base import _collate
    states = []
    for start in range(0, len(indices), M.BATCH):
        part = list(indices[start:start + M.BATCH])
        try:
            batch = _collate([view[i] for i in part])
            ents = [view.entry(i) for i in part]
        except C.V05LTError:
            raise
        except Exception as exc:  # noqa: BLE001 -- typed, logged, re-raised
            C.fail("EVAL_DATA_LOAD", f"positions {part[:4]}...: "
                   f"{type(exc).__name__}: {exc}")
        sel = {"ordered_identities": [{"file": f, "slice_index": s}
                                      for f, s in ents]}
        part_states = _classified("build_states", tg._build_slice_states,
                                  batch, sel, p4, s_ref)
        from seqref_mri.scripts.train_base import _prepare
        prep = _classified("build_states prepare", _prepare, batch, "cpu",
                           test0=False)
        for j, st in enumerate(part_states):
            x1, c1 = prep["x_norm"][j], st["cond"][0]
            st["x_true_c"] = torch.complex(x1[0], x1[1])
            st["cond_c"] = torch.complex(c1[0], c1[1])
            if not (torch.equal(st["x_true_c"].abs(), st["x_true_mag"]) and
                    torch.equal(prep["cond_in"][j], c1)):
                C.fail("EVAL_STATE_MISMATCH", f"{st['identity']}: complex "
                       "ground truth / conditioning != TINY state")
        states += part_states
    return states


def iter_states(view, indices, p4: dict, s_ref: float, chunk: int = 32):
    """Streaming build_states (bounded memory for all-slice calibration)."""
    idx = list(indices)
    for i in range(0, len(idx), chunk):
        yield from build_states(view, idx[i:i + chunk], p4, s_ref)


def _nmse_or_none(u_hat, st, eligible: bool, rule: str):
    C.require(rule in NMSE_RULES, "EVAL_NMSE_RULE", rule)
    if rule == "validation_eligible" and not eligible:
        return None
    if rule == "v02_holdout_all" and \
            float(np.sum(np.abs(np.asarray(st["u_true"])) ** 2)) == 0.0:
        C.fail("VALIDITY_NMSE_UNDEFINED", f"{st.get('identity')}: zero "
               "free-space energy; NMSE_u undefined, never dropped (AM4-B)")
    return M.nmse_value(u_hat, st["u_true"])


def holdout_nmse_mean(values: list) -> dict:
    """AM4-B: plain mean of per-slice NMSE_u over ALL holdout slices
    (no threshold exclusion, no None allowed)."""
    C.require(len(values) > 0 and all(v is not None and math.isfinite(v)
                                      for v in values),
              "VALIDITY_NMSE_UNDEFINED", "holdout NMSE_u requires a finite "
              "value for every slice")
    return {"n": len(values), "mean": float(np.mean(values))}


def _metric(where: str, fn, *args):
    """Shared metric call; METRIC_* errors re-typed for the evaluator."""
    try:
        return fn(*args)
    except C.V05LTError as exc:
        if not exc.code.startswith("METRIC_"):
            raise
        txt = str(exc).lower()
        nonfinite = any(w in txt for w in ("finite", "nan", "inf"))
        C.fail("EVAL_NON_FINITE" if nonfinite else "EVAL_METRIC_INVALID",
               f"{where}: {exc.code}: {exc}")
    except Exception as exc:  # noqa: BLE001 -- typed, logged, re-raised
        C.fail("EVAL_METRIC_INVALID", f"{where}: {type(exc).__name__}: {exc}")


def _image_record(mag, st, u_hat, eligible: bool, rule: str) -> dict:
    rec = _metric("PSNR", M.psnr_record, mag, st["x_true_mag"])
    rec["ssim"] = _metric("SSIM", M.ssim_value, mag, st["x_true_mag"])
    rec["nmse_u"] = _metric("NMSE_u", _nmse_or_none, u_hat, st, eligible,
                            rule)
    return rec


def r0_image(model, st) -> tuple:
    """R0 complex image and its free vector (z = 0)."""
    torch, tg, d3, est, ffr, _ = _seams()
    z0 = torch.zeros(1, ffr.FLOW_DIM_REAL)
    with torch.no_grad():
        x0, u0 = _classified("R0 decode", tg._decode_z, model, z0, st)
    _finite_array(x0.abs().numpy(), "R0 image")
    return x0, u0


def r0_readout(model, st, eligible: bool = True,
               rule: str = "validation_eligible") -> dict:
    x0, u0 = r0_image(model, st)
    return _image_record(x0.abs(), st, u0, eligible, rule)


def decode_bank(model, st, bank: np.ndarray) -> np.ndarray:
    """One batched decode of all bank rows -> (rows, n_free) complex128."""
    torch, tg, d3, *_ = _seams()
    if bank.shape[0] != BANK_ROWS:
        C.fail("D3_BANK_LAYOUT_UNEXPECTED",
              f"R1 expects {BANK_ROWS} bank rows, got {bank.shape[0]}")
    with torch.no_grad():
        dec = _classified("R1 bank decode", d3._decode_bank_conditioned,
                          model, st, bank, st["cond"], st["mask"])
    if dec.shape[0] != BANK_ROWS:
        C.fail("D3_BANK_LAYOUT_UNEXPECTED",
              f"decoded {dec.shape[0]} rows, expected {BANK_ROWS}")
    _finite_array(dec, "R1 decodes")
    return dec


def r1_from_decodes(st, dec: np.ndarray, k: int, eligible: bool = True,
                    rule: str = "validation_eligible") -> dict:
    """Posterior mean of the first k decodes (k = 128: all, exactly as
    d3._pm_metrics / V02G v0.2 _r1_measure: dec.mean(axis=0))."""
    img, u_mean = r1_image(st, dec, k)
    return _image_record(img.abs(), st, u_mean, eligible, rule)


def r1_image(st, dec: np.ndarray, k: int = BANK_ROWS) -> tuple:
    """R1 complex image (ONE image formation after the u-space mean)."""
    torch, tg, d3, est, *_ = _seams()
    C.require(k in PREFIXES and k <= dec.shape[0], "EVAL_PREFIX_INVALID",
              str(k))
    u_mean = dec.mean(axis=0) if k == dec.shape[0] else dec[:k].mean(axis=0)
    img = _classified("R1 image formation", est.image_from_u, st, u_mean)
    _finite_array(img.abs().numpy(), "R1 image")
    return img, u_mean


# ---------------------------------------------------------------------------
# Hard data consistency (194 §8, AM1) -- I1.7(b)
# ---------------------------------------------------------------------------
DC_DTYPE = "complex128"               # evaluation precision (decode output)
DC_MEASURES = ("a", "b", "c_rms", "c_max")
DC_TOL_FACTOR = 10.0
DC_CORRUPT_SCALE = 1.0e-3             # AM1 deterministic corruption amplitude


def _dc_ops():
    torch, tg, *_ = _seams()
    try:
        from seqref_mri.src import fastmri_data as fdm
        from seqref_mri.src import residual_decoder as rdec
    except ImportError as exc:
        C.fail("EVAL_IMPORT_FAILED", str(exc))
    return torch, fdm, rdec


def _dc_mask(st):
    torch, fdm, rdec = _dc_ops()
    return rdec.column_mask_tensor(st["cmap"], "cpu", torch.complex128)


def y_norm(st):
    """Production acquired data in the decoder's normalised space: y / amax
    (as residual_decoder.decode_normalised), at the evaluation dtype."""
    torch, *_ = _dc_ops()
    return (st["y"][0] / st["amax"][0]).to(torch.complex128)


def hard_dc(x, y, m):
    """x_DC = F^-1(M.y + (1 - M).F(x)) on the COMPLEX image (194 §8)."""
    torch, fdm, _ = _dc_ops()
    C.require(torch.is_complex(x) and x.dtype == torch.complex128,
              "DC_DTYPE", f"image dtype {x.dtype} != {DC_DTYPE}")
    return fdm.ifft2c(m * y + (1 - m) * fdm.fft2c(x))


def _ratio(num: float, den: float, name: str, where: str) -> float:
    if not (math.isfinite(num) and math.isfinite(den)):
        C.fail("DC_NON_FINITE", f"{where} {name}: num={num!r} den={den!r}",
               "numerical_failure")
    if den == 0.0:
        C.fail("DC_ZERO_DENOMINATOR", f"{where} {name}: zero denominator "
               "(validity finding; amendment required, no default)")
    return num / den


def dc_measures(x, x_dc, y, m, where: str = "") -> dict:
    """(a) max|M(F x_DC - y)| / max|M y|; (b) ||(1-M)(F x_DC - F x)|| /
    ||(1-M) F x||; (c) relative RMS and relative max image change."""
    torch, fdm, _ = _dc_ops()
    kx, kd = fdm.fft2c(x), fdm.fft2c(x_dc)
    return {
        "a": _ratio(float((m * (kd - y)).abs().max()),
                    float((m * y).abs().max()), "a", where),
        "b": _ratio(float(torch.linalg.vector_norm((1 - m) * (kd - kx))),
                    float(torch.linalg.vector_norm((1 - m) * kx)), "b", where),
        "c_rms": _ratio(float(torch.linalg.vector_norm(x_dc - x)),
                        float(torch.linalg.vector_norm(x)), "c_rms", where),
        "c_max": _ratio(float((x_dc - x).abs().max()),
                        float(x.abs().max()), "c_max", where)}


def check_tolerances(tol: dict) -> dict:
    """Every registered measure present; each tolerance a finite float > 0."""
    C.require(isinstance(tol, dict) and set(tol) >= set(DC_MEASURES),
              "DC_TOLERANCE_MISSING", str(sorted(tol) if isinstance(
                  tol, dict) else type(tol)))
    for k in DC_MEASURES:
        v = tol[k]
        if not (isinstance(v, float) and math.isfinite(v) and v > 0.0):
            C.fail("DC_TOLERANCE_INVALID", f"{k}: {v!r} (finite float > 0 "
                   "required)")
    return tol


def dc_identity(meas: dict, tol: dict) -> bool:
    check_tolerances(tol)
    return all(meas[k] <= tol[k] for k in DC_MEASURES)


def calibrate_dc(states: list) -> dict:
    """Numerical floors on COMPLEX validation ground truth: DC with each
    image's own acquired values (y_cal = M.F(x_true)). Floor = max over
    slices per measure; tolerance = 10 x floor. Zero / non-finite floor ->
    DC_FLOOR_INVALID (no epsilon). Values are recorded at preflight (P5)."""
    torch, fdm, _ = _dc_ops()
    n_states = 0
    floors = {k: (-1.0, None) for k in DC_MEASURES}
    den_min = {"a": math.inf, "b": math.inf, "c_rms": math.inf,
               "c_max": math.inf}
    for st in states:             # any iterable (e.g. iter_states), one pass
        n_states += 1
        x = st["x_true_c"].to(torch.complex128)
        m = _dc_mask(st)
        kx = fdm.fft2c(x)
        y_cal = m * kx
        meas = dc_measures(x, hard_dc(x, y_cal, m), y_cal, m,
                           f"calibration {st['identity']}")
        dens = {"a": float((m * y_cal).abs().max()),
                "b": float(torch.linalg.vector_norm((1 - m) * kx)),
                "c_rms": float(torch.linalg.vector_norm(x)),
                "c_max": float(x.abs().max())}
        for k in DC_MEASURES:
            den_min[k] = min(den_min[k], dens[k])
            if meas[k] > floors[k][0]:
                floors[k] = (meas[k], st["identity"])
    C.require(n_states > 0, "DC_CALIBRATION_EMPTY", "no states")
    out = {"floors": {}, "tolerances": {}, "argmax_identity": {}}
    for k in DC_MEASURES:
        f, ident = floors[k]
        if not math.isfinite(f) or f <= 0.0:
            C.fail("DC_FLOOR_INVALID", f"measure {k}: floor {f!r} "
                   "(validity finding; amendment required, no default)")
        out["floors"][k] = f
        out["tolerances"][k] = DC_TOL_FACTOR * f
        out["argmax_identity"][k] = ident
    out["provenance"] = {
        "procedure": "194 §8: complex validation GT, acquired coefficients "
                     "replaced with their own values; floor = max over "
                     "slices; tolerance = 10 x floor",
        "n_slices": n_states, "dtype": DC_DTYPE,
        "fft": "fastmri_data.fft2c/ifft2c (ortho, centred)",
        "mask": "residual_decoder.column_mask_tensor (column, broadcast "
                "over rows)",
        "scaling": "x_norm = x_true / amax (train_base._prepare); "
                   "readouts use y / amax",
        "min_denominators": den_min,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "deterministic": bool(torch.are_deterministic_algorithms_enabled())}
    return out


def dc_corruption_tests(st, tol: dict) -> dict:
    """AM1-A on one state, deterministic (no RNG): corruption = scale x
    max|M y| on every acquired coefficient, alternating sign by column.
    Pre-DC repair: residual > tol_a before, (a) <= tol_a after, (c)
    between corrupted input and DC output > tol. Post-DC detection:
    corrupting the DC output gives (a) > tol_a."""
    torch, fdm, _ = _dc_ops()
    check_tolerances(tol)
    m = _dc_mask(st)
    y = y_norm(st)
    x = st["x_true_c"].to(torch.complex128)
    w = m.shape[-1]
    sign = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(w)],
                        dtype=torch.float64).to(torch.complex128)
    delta = DC_CORRUPT_SCALE * float((m * y).abs().max()) * sign * m
    # Pre-DC: input whose acquired coefficients are wrong.
    x_bad = fdm.ifft2c(m * (y + delta) + (1 - m) * fdm.fft2c(x))
    before = dc_measures(x_bad, x_bad, y, m, "AM1 pre before")["a"]
    x_fix = hard_dc(x_bad, y, m)
    after = dc_measures(x_bad, x_fix, y, m, "AM1 pre after")
    pre = {"a_before": before, "a_after": after["a"],
           "c_rms": after["c_rms"], "c_max": after["c_max"]}
    pre["pass"] = bool(before > tol["a"] and after["a"] <= tol["a"]
                       and after["c_rms"] > tol["c_rms"]
                       and after["c_max"] > tol["c_max"])
    # Post-DC: corrupt the DC output; the detector must fire.
    x_dc = hard_dc(x, y, m)
    x_dc_bad = fdm.ifft2c(fdm.fft2c(x_dc) + delta)
    post_a = dc_measures(x, x_dc_bad, y, m, "AM1 post")["a"]
    post = {"a": post_a, "pass": bool(post_a > tol["a"])}
    if not pre["pass"]:
        C.fail("DC_REPAIR_FAILED", f"{st['identity']}: {pre}")
    if not post["pass"]:
        C.fail("DC_DETECTION_FAILED", f"{st['identity']}: {post}")
    return {"pre_dc_repair": pre, "post_dc_detection": post}


def u_from_image(st, x):
    """Free vector of a complex image in the u_true convention
    (tiny_gate: gather_unmeasured(fft2c(x - cond)))."""
    torch, fdm, rdec = _dc_ops()
    k = fdm.fft2c(x - st["cond_c"].to(x.dtype))
    return np.asarray(rdec.gather_unmeasured(k.unsqueeze(0), st["cmap"])
                      .to(torch.complex128).numpy())[0]


def dc_readout(st, x, tol: dict, eligible: bool, rule: str) -> dict:
    """R+DC record: metrics of |x_DC|, the DC measures and identity flag."""
    m = _dc_mask(st)
    y = y_norm(st)
    x_dc = _classified("DC", hard_dc, x, y, m)
    _finite_array(x_dc.abs().numpy(), "DC image")
    meas = dc_measures(x, x_dc, y, m, f"readout {st['identity']}")
    rec = _image_record(x_dc.abs(), st, u_from_image(st, x_dc), eligible,
                        rule)
    rec["dc"] = meas
    rec["dc_identity"] = dc_identity(meas, tol)
    return rec


def dc_identity_summary(rows: list) -> dict:
    """Per DC readout: identity on ALL slices -> reported as identity
    results (194 §8)."""
    out = {}
    for key in ("r0_dc", "r1_dc"):
        flags = [r[key]["dc_identity"] for r in rows if key in r]
        out[key] = {"n": len(flags), "n_identity": int(sum(flags)),
                    "identity_all": bool(flags) and all(flags)}
    return out


# ---------------------------------------------------------------------------
# Zero-filled reference (AM3-D) -- I1.7(b)
# ---------------------------------------------------------------------------
ZF_SCHEMA = "seqref-v05lt-zf/1"
ZF_REGISTERED_DIR = ("results", "v05lt", "zf")


def zf_row(st, info: dict, rule: str = "validation_eligible") -> dict:
    """ZF = estimators.image_from_u(u = 0); PSNR, SSIM, NMSE_u; no NLL.
    info = eligibility entry {eligible, midpoint, volume_id}. Non-finite ->
    EPOCH0_NONFINITE (AM3-E3). Eligible NMSE_u must be exactly 1."""
    torch, tg, d3, est, *_ = _seams()
    C.require(isinstance(info, dict) and {"eligible", "midpoint",
                                          "volume_id"} <= set(info),
              "ZF_ROW_INFO", f"{st.get('identity')}: {info!r}")
    eligible = bool(info["eligible"])
    u0 = np.zeros_like(st["u_true"])
    img = _classified("ZF image formation", est.image_from_u, st, u0)
    mag = img.abs()
    if not bool(torch.isfinite(mag).all()):
        C.fail("EPOCH0_NONFINITE", f"ZF {st['identity']}: non-finite image",
               "numerical_failure")
    try:
        rec = _image_record(mag, st, u0, eligible, rule)
    except C.V05LTError as exc:
        if exc.code == "EVAL_NON_FINITE":
            C.fail("EPOCH0_NONFINITE", f"ZF {st['identity']}: {exc}",
                   "numerical_failure")
        raise
    if rec["nmse_u"] is not None and rec["nmse_u"] != 1.0:
        C.fail("ZF_PIPELINE_CHECK", f"{st['identity']}: eligible ZF NMSE_u "
               f"{rec['nmse_u']!r} != 1 (AM3-D)")
    # ZF + DC is identical to ZF by construction (AM3-D) and is not
    # measured: ZF has no unmeasured energy, so measure (b)'s denominator
    # is roundoff of zero and the ratio is meaningless (finding I1.7b-1).
    rec["zf_dc"] = "identical to ZF by construction (AM3-D); not measured"
    rec.update({"volume_id": info["volume_id"],
                "file": st["identity"]["file"],
                "slice_index": int(st["identity"]["slice_index"]),
                "eligible": eligible, "midpoint": bool(info["midpoint"])})
    return rec


def _zf_agg(rows: list) -> dict:
    """194 §8.2: per-slice -> volume mean -> equal-weight mean over volumes;
    NMSE_u over eligible slices only, volumes without one -> null and
    outside the NMSE population (AM2-C)."""
    from seqref_mri.scripts import v05lt_eligibility as E
    ent = [(r["volume_id"], r["file"], r["slice_index"]) for r in rows]
    key = lambda r: (r["file"], r["slice_index"])  # noqa: E731
    out = {"slices": len(rows), "volumes": len({r["volume_id"]
                                                for r in rows})}
    for m in ("psnr", "ssim"):
        vm = E.volume_means({key(r): r[m] for r in rows}, ent)
        out[m] = float(np.mean(list(vm.values())))
    el = {key(r): r["eligible"] for r in rows}
    vm = E.volume_means({key(r): r["nmse_u"] for r in rows if r["eligible"]},
                        ent, el)
    pop = E.nmse_population(vm)
    out["nmse_u"] = float(np.mean([vm[v] for v in pop])) if pop else None
    out["nmse_population_volumes"] = len(pop)
    out["nmse_null_volumes"] = sorted(v for v, x in vm.items() if x is None)
    for flag in ("psnr_capped", "zero_error"):
        ids = [f"{r['file']}|{r['slice_index']}" for r in rows if r[flag]]
        out[flag] = {"count": len(ids), "identities": ids}
    return out


def zf_summary(rows: list) -> dict:
    """All-slice and midpoint scopes, volume-weighted (computed once)."""
    C.require(len(rows) > 0, "ZF_EMPTY", "no ZF rows")
    C.require(len({(r["file"], r["slice_index"]) for r in rows}) ==
              len(rows), "AGG_DUPLICATE_SLICE", "duplicate ZF slice")
    mid = [r for r in rows if r["midpoint"]]
    return {"all_slices": _zf_agg(rows),
            "midpoint": _zf_agg(mid) if mid else None}


ZF_FILE = "zf_reference.json"


# ---------------------------------------------------------------------------
# A: atomic no-overwrite publication (item 22) -- I1.7(c1)
# ---------------------------------------------------------------------------
RENAME_NOREPLACE = 1                  # linux/fs.h
AT_FDCWD = -100


def _renameat2():
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    fn = getattr(libc, "renameat2", None)
    if fn is None:
        C.fail("PUBLISH_NOREPLACE_UNAVAILABLE", "libc has no renameat2; "
               "no-overwrite publication impossible (no fallback)")
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                   ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    return fn, ctypes


def publish_noreplace(src: Path, dst: Path) -> None:
    """Atomically rename src -> dst, refusing if dst exists in ANY form
    (file, empty or non-empty directory, symlink): renameat2 with
    RENAME_NOREPLACE. Unsupported kernel/filesystem -> stop; never falls
    back to a replacing rename."""
    import errno
    fn, ctypes = _renameat2()
    r = fn(AT_FDCWD, os.fsencode(str(src)), AT_FDCWD, os.fsencode(str(dst)),
           RENAME_NOREPLACE)
    if r != 0:
        e = ctypes.get_errno()
        if e == errno.EEXIST:
            C.fail("ARTIFACT_EXISTS", f"{dst}: destination exists "
                   "(RENAME_NOREPLACE)")
        if e in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
            C.fail("PUBLISH_NOREPLACE_UNAVAILABLE", f"{dst}: "
                   f"{os.strerror(e)} (no fallback)")
        C.fail("PUBLISH_FAILED", f"{src} -> {dst}: {os.strerror(e)}")
    C.fsync_dir(Path(dst).parent)


def _zf_staging(out_dir: Path) -> Path:
    return out_dir.with_name(f".{out_dir.name}.staging")


def dest_path(p) -> Path:
    """Normalise a publication destination WITHOUT following its final
    component: parent resolved, name kept. A symlink (dangling or not) at
    the destination is therefore seen as an existing entry, never
    followed."""
    p = Path(p)
    C.require(p.name not in ("", ".", ".."), "PUBLISH_DEST_INVALID", str(p))
    return Path(os.path.abspath(p.parent)).resolve() / p.name


def entry_exists(p: Path) -> bool:
    """True for ANY directory entry, incl. dangling symlinks (lstat)."""
    return os.path.lexists(p)


def verify_zf(out_dir: Path) -> str:
    """Complete pair check: data file + sidecar, sidecar hash == data hash."""
    out_dir = Path(out_dir)
    p, sc = out_dir / ZF_FILE, out_dir / (ZF_FILE + ".sha256")
    if not (p.is_file() and sc.is_file()):
        C.fail("ZF_INCOMPLETE", f"{out_dir}: data or sidecar missing")
    got = C.sha256_file(p)
    want = sc.read_text().split()[0] if sc.read_text().split() else ""
    C.require(got == want, "ZF_INCOMPLETE", f"{out_dir}: sidecar {want[:12]} "
              f"!= data {got[:12]}")
    return got


def write_zf(out_dir: Path, rows: list, summary: dict, context: dict, *,
             gate_2a: bool = False, hook=None) -> str:
    """Write-once ZF record as an atomic pair. Both files are written and
    fsynced in a staging sibling directory, which is then renamed to
    out_dir. Interruption leaves either nothing at out_dir or the complete
    pair; publication uses renameat2(RENAME_NOREPLACE), so an empty
    directory created at out_dir in the meantime is never replaced.
    Refusals (never overwrite, never delete): out_dir exists
    (ZF_EXISTS; verify_zf tells complete from incomplete); a staging
    directory left by an interrupted write (ZF_STAGING_PRESENT -- owner
    inspects and moves it). The registered results/v05lt/zf (or below) is
    refused unless gate_2a=True. Strict JSON: NaN/Inf refused."""
    out_dir = dest_path(out_dir)
    parts = out_dir.parts
    registered = any(tuple(parts[i:i + 3]) == ZF_REGISTERED_DIR
                     for i in range(len(parts) - 2))
    if registered and not gate_2a:
        C.fail("ZF_GATE_REQUIRED", f"{out_dir}: registered ZF output is "
               "written only at gate 2a")
    if entry_exists(out_dir):
        C.fail("ZF_EXISTS", f"{out_dir}: an entry already exists "
               "(write-once; symlinks are never followed)")
    stg = _zf_staging(out_dir)
    if entry_exists(stg):
        C.fail("ZF_STAGING_PRESENT", f"{stg}: left by an interrupted write; "
               "owner inspection required (not deleted)")
    doc = {"schema": ZF_SCHEMA, "context": context, "summary": summary,
           "rows": rows}
    try:
        data = (json.dumps(doc, sort_keys=True, allow_nan=False) +
                "\n").encode()
    except (TypeError, ValueError) as exc:
        C.fail("ZF_JSON_INVALID", f"{type(exc).__name__}: {exc}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stg.mkdir()
    sha = C.write_durable_new(stg / ZF_FILE, data)
    C.write_durable_new(stg / (ZF_FILE + ".sha256"),
                        f"{sha}  {ZF_FILE}\n".encode())
    if hook is not None:
        hook("zf:pair_staged")
    try:
        publish_noreplace(stg, out_dir)       # refuses ANY existing out_dir
    except C.V05LTError as exc:
        if exc.code == "ARTIFACT_EXISTS":
            C.fail("ZF_EXISTS", f"{out_dir}: appeared during the write; "
                   f"staging kept at {stg}")
        raise
    C.require(verify_zf(out_dir) == sha, "ZF_INCOMPLETE", str(out_dir))
    return sha


# ---------------------------------------------------------------------------
# Four readouts
# ---------------------------------------------------------------------------
def evaluate_states(model, states: list, eligible: list, banks: dict,
                    prefixes=(BANK_ROWS,), rule: str = "validation_eligible",
                    decode_counter: dict | None = None,
                    prefix_bank: str = "primary",
                    dc_tol: dict | None = None,
                    keep_images: dict | None = None) -> list:
    """R0 and R1 on the SAME states (identical slices and masks). With
    dc_tol: also R0+DC and R1+DC (R1+DC from the SAME primary-bank decodes,
    k = 128; no extra decodes). banks:
    {name: (128, d) float32}. Prefix stability only for `prefix_bank`
    (registered: the primary bank); other banks report k = 128 only.
    The model's previous train/eval mode is restored on exit.
    keep_images: {identity_key: {}} -- for those slices the complex R0 and
    R1 (primary, k = 128) images are stored in place (panels reuse the
    readout decodes; no extra decodes)."""
    if len(states) != len(eligible):
        C.fail("EVAL_LENGTH_MISMATCH", f"{len(states)} states vs "
               f"{len(eligible)} eligibility flags")
    C.require(BANK_ROWS in prefixes, "EVAL_PREFIX_INVALID", str(prefixes))
    if dc_tol is not None:
        check_tolerances(dc_tol)
    was_training = model.training
    model.eval()
    rows = []
    try:
        for st, el in zip(states, eligible):
            x0, u0 = r0_image(model, st)
            row = {"identity": st["identity"], "eligible": bool(el),
                   "r0": _image_record(x0.abs(), st, u0, el, rule),
                   "r1": {}}
            if dc_tol is not None:
                row["r0_dc"] = dc_readout(st, x0, dc_tol, el, rule)
            keep = (keep_images or {}).get(_ident_key(st["identity"]))
            if keep is not None:
                keep["r0"] = x0
            for name, bank in banks.items():
                dec = decode_bank(model, st, bank)
                if decode_counter is not None:
                    decode_counter[name] = decode_counter.get(name, 0) \
                        + dec.shape[0]
                ks = prefixes if name == prefix_bank else (BANK_ROWS,)
                row["r1"][name] = {str(k): r1_from_decodes(st, dec, k, el,
                                                           rule)
                                   for k in ks}
                if (dc_tol is not None or keep is not None) and \
                        name == prefix_bank:
                    x1, _ = r1_image(st, dec, BANK_ROWS)
                    if dc_tol is not None:
                        row["r1_dc"] = dc_readout(st, x1, dc_tol, el, rule)
                    if keep is not None:
                        keep["r1"] = x1
            rows.append(row)
    finally:
        model.train(was_training)
    return rows


# ---------------------------------------------------------------------------
# Schedule and reuse (F11)
# ---------------------------------------------------------------------------
def plan_campaign(runs: list, n_slices: int = 97) -> dict:
    """runs: [{"seed", "arm", "roles": {role: {"epoch", "model_state"}}}]
    -> evaluation tasks deduplicated by model-state hash. Epoch 0 is
    shared by both arms of a seed (one task per seed). Primary bank at
    every distinct checkpoint; banks B/C and prefix stability only at a
    checkpoint holding the role 'selected'."""
    tasks, epoch0_by_seed = {}, {}
    for r in runs:
        roles = r["roles"]
        C.require(set(roles) == set(ROLES), "PLAN_ROLES", str(sorted(roles)))
        C.require(roles["epoch0"]["epoch"] == 0 and roles["epoch3"]["epoch"]
                  == 3, "PLAN_EPOCHS", str(roles))
        e0 = roles["epoch0"]["model_state"]
        prev = epoch0_by_seed.setdefault(r["seed"], e0)
        if prev != e0:
            C.fail("PLAN_EPOCH0_PAIRING", f"seed {r['seed']}: arms carry "
                   f"different epoch-0 states {prev[:12]} / {e0[:12]}")
        for role in ROLES:
            ms = roles[role]["model_state"]
            C.require(isinstance(ms, str) and len(ms) == 64, "PLAN_STATE",
                      f"{role}: {ms!r}")
            t = tasks.setdefault(ms, {"model_state": ms, "epoch":
                                      roles[role]["epoch"], "uses": [],
                                      "banks": ["primary"], "prefixes":
                                      [BANK_ROWS]})
            C.require(t["epoch"] == roles[role]["epoch"],
                      "PLAN_STATE_EPOCH_CONFLICT", f"{ms[:12]}")
            t["uses"].append({"seed": r["seed"], "arm": r["arm"],
                              "role": role})
            if role == "selected":
                t["banks"] = ["primary", *EXTRA_BANKS]
                t["prefixes"] = list(PREFIXES)
                t["prefix_bank"] = "primary"
    out = sorted(tasks.values(), key=lambda t: (t["epoch"], t["model_state"]))
    decodes = sum(len(t["banks"]) * BANK_ROWS * n_slices for t in out)
    return {"tasks": out, "n_tasks": len(out), "decodes": decodes,
            "prefix_extra_decodes": 0,
            "note": "totals for these inputs; checkpoint coincidences "
                    "(shared model states) reduce them"}


# ---------------------------------------------------------------------------
# F10 -- strict input verification
# ---------------------------------------------------------------------------
def _sha(p: Path) -> str:
    return C.sha256_file(p)


def _sidecar(p: Path, code: str) -> str:
    side = Path(str(p) + ".sha256")
    if not p.is_file() or not side.is_file():
        C.fail(code, f"{p} or its sidecar is missing")
    rec = side.read_text().split()
    got = _sha(p)
    if not rec or rec[0] != got:
        C.fail(code, f"{p.name}: file {got[:12]} != sidecar "
              f"{(rec or ['<empty>'])[0][:12]}")
    return got


def _json(p: Path, code: str) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        C.fail(code, f"{p}: {exc}")


def check_registry(reg: dict) -> None:
    missing = [k for k in REG_KEYS if k not in reg or reg[k] in (None, "",
                                                                  [], {})]
    if missing:
        C.fail("F10_REGISTRY_INCOMPLETE", f"missing/empty: {missing}")
    for side in ("step0", "final"):
        a = reg["anchors"].get(side, {})
        for m in ("psnr", "nmse_u"):
            v = a.get(m)
            if not (isinstance(v, float) and math.isfinite(v)):
                C.fail("F10_REGISTRY_INCOMPLETE", f"anchor {side}.{m}: {v!r}")


def check_registry_against_code(reg: dict) -> dict:
    """The registry must equal the constants in the code that defined
    them (V02G v0.2 path; production loader pins)."""
    from seqref_mri.scripts import v02_readout_diag as g
    from seqref_mri.src import free_flow_runtime as ffr
    pairs = {
        "anchors.step0.psnr": (reg["anchors"]["step0"]["psnr"], g.PM_STEP0_PSNR),
        "anchors.step0.nmse_u": (reg["anchors"]["step0"]["nmse_u"], g.PM_STEP0_NMSE),
        "anchors.final.psnr": (reg["anchors"]["final"]["psnr"], g.PM_FINAL_PSNR),
        "anchors.final.nmse_u": (reg["anchors"]["final"]["nmse_u"], g.PM_FINAL_NMSE),
        "bank_sha256": (reg["bank_sha256"], g.PM_BANK_SHA256),
        "bank_manifest_sha256": (reg["bank_manifest_sha256"], g.PM_BANK_MANIFEST_SHA256),
        "holdout_manifest_sha256": (reg["holdout_manifest_sha256"], g.HOLDOUT_MANIFEST_SHA256),
        "d3_manifest_sha256": (reg["d3_manifest_sha256"], g.D3_SUBSET_MANIFEST_SHA256),
        "v02h_sha256": (reg["v02h_sha256"], g.V02H_EVIDENCE_SHA256),
        "steps": (list(reg["steps"]), [g.STEP0, g.FINAL_STEP]),
        "psnr_tol_db": (reg["psnr_tol_db"], g.PSNR_REPRO_TOL_DB),
        "nmse_tol_rel": (reg["nmse_tol_rel"], g.NMSE_REPRO_TOL_REL),
        "n_d3": (reg["n_d3"], g.N_D3_SUBSET),
        "facts_schema": (reg["facts_schema"], g.FACTS_SCHEMA),
        "semantic_excluded": (list(reg["semantic_excluded"]),
                              list(g.SEMANTIC_EXCLUDED)),
        "p4_sha256": (reg["p4_sha256"], ffr.P4S2_FILE_SHA256),
        "implb_sha256": (reg["implb_sha256"], ffr.IMPLB_FILE_SHA256),
    }
    bad = [k for k, (a, b) in pairs.items() if a != b]
    if bad:
        C.fail("F10_REGISTRY_CONFLICT", f"registry != code constants: {bad}")
    return {k: "equal" for k in pairs}


def _segment(text: str, name: str, where: str) -> str:
    """Source text of a top-level assignment or function `name`."""
    import ast
    lines = text.splitlines()
    for node in ast.parse(text).body:
        hit = (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == name) or (
            isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets))
        if hit:
            return "\n".join(lines[node.lineno - 1:node.end_lineno])
    C.fail("F10_PROCEDURE_MISSING", f"{where}: {name} not found")


def historical_semantic_procedure(repo: Path, commit: str,
                                  reg: dict = REG_F10) -> dict:
    """The semantic-hash procedure as it existed at `commit` (the commit
    that produced the authoritative facts): exclusion list and hashing
    functions from `git show commit:path`; each must be identical to the
    working tree; the exclusion list must equal the registry."""
    import ast
    import subprocess
    seg = {}
    for rel, names in PROCEDURE_SOURCES:
        r = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{rel}"],
                           capture_output=True)
        if r.returncode != 0:
            C.fail("F10_PROCEDURE_MISSING", f"{commit}:{rel}: "
                   f"{r.stderr.decode(errors='replace').strip()}")
        old = r.stdout.decode()
        new = (Path(repo) / rel).read_text()
        for n in names:
            a, b = _segment(old, n, f"{commit}:{rel}"), _segment(new, n, rel)
            if a != b:
                C.fail("F10_PROCEDURE_DRIFT", f"{rel}::{n} differs between "
                       f"{commit[:12]} and the working tree")
            seg[f"{rel}::{n}"] = hashlib.sha256(a.encode()).hexdigest()
    excl = ast.literal_eval(_segment(
        subprocess.run(["git", "-C", str(repo), "show",
                        f"{commit}:{PROCEDURE_SOURCES[0][0]}"],
                       capture_output=True).stdout.decode(),
        "SEMANTIC_EXCLUDED", commit).split("=", 1)[1].strip())
    if list(excl) != list(reg["semantic_excluded"]):
        C.fail("F10_PROCEDURE_MISMATCH", f"{commit[:12]} SEMANTIC_EXCLUDED "
               f"{list(excl)} != registry {reg['semantic_excluded']}")
    return {"commit": commit, "semantic_excluded": list(excl),
            "segments_identical_to_working_tree": seg}


def _semantic(facts: dict, excluded, code: str) -> dict:
    """Historical procedure: payload = facts minus `excluded` (V02G) minus
    the post-hash keys; preflight_io.canonical_hash; included_keys must be
    exactly the payload keys; recomputed = stored."""
    from seqref_mri.src.preflight_io import canonical_hash
    expect = sorted(k for k in facts if k not in set(excluded)
                    and k not in POST_HASH_KEYS)
    scope = facts.get("semantic_scope") or {}
    inc = scope.get("included_keys")
    if not isinstance(inc, list) or inc != expect:
        C.fail(code, f"included_keys {inc} != procedure keys {expect}")
    got = canonical_hash({k: facts[k] for k in expect})
    if got != facts.get("semantic_sha256"):
        C.fail(code, f"recomputed semantic {got[:12]} != stored "
               f"{str(facts.get('semantic_sha256'))[:12]}")
    recorded = list(scope.get("excluded") or [])
    return {"semantic_sha256": got, "payload_keys": expect,
            "procedure_excluded": list(excluded),
            "scope_excluded_field": recorded,
            "known_inconsistency": (sorted(set(recorded) ^ set(excluded))
                                    if set(recorded) != set(excluded) else [])}


def _anchor_eq(rec_psnr, rec_nmse, reg_anchor, where: str) -> None:
    if rec_psnr != reg_anchor["psnr"] or rec_nmse != reg_anchor["nmse_u"]:
        C.fail("F10_ANCHOR_MISMATCH", f"{where}: ({rec_psnr!r}, {rec_nmse!r})"
              f" != registered ({reg_anchor['psnr']!r}, "
              f"{reg_anchor['nmse_u']!r})")


def _manifest(p: Path, kind: str, stored_reg: str, file_reg: str | None):
    from seqref_mri.scripts import v02_manifests as v02m
    fsha = _sidecar(p, "F10_MANIFEST_SIDECAR")
    if file_reg is not None and fsha != file_reg:
        C.fail("F10_MANIFEST_FILE_MISMATCH", f"{p.name}: {fsha[:12]}")
    man = _json(p, "F10_MANIFEST_SCHEMA")
    if man.get("schema") != "seqref-v02-manifest/1" or man.get("kind") != kind:
        C.fail("F10_MANIFEST_SCHEMA", f"{p.name}: schema/kind "
              f"{man.get('schema')!r}/{man.get('kind')!r}")
    body = {k: v for k, v in man.items() if k != "manifest_sha256"}
    rec = v02m.manifest_sha256(body)
    if rec != man.get("manifest_sha256"):
        C.fail("F10_MANIFEST_CONTENT", f"{p.name}: recomputed {rec[:12]} != "
              f"stored {str(man.get('manifest_sha256'))[:12]}")
    if man["manifest_sha256"] != stored_reg:
        C.fail("F10_MANIFEST_MISMATCH", f"{p.name}: stored "
              f"{man['manifest_sha256'][:12]} != registered {stored_reg[:12]}")
    return man, fsha


PATH_KEYS = ("facts", "v02_facts", "manifest_dir", "train_record",
             "run_root", "p0s", "p4", "implb", "v02h")


def f10_paths(repo: Path, reg: dict = REG_F10) -> dict:
    missing = [k for k in PATH_KEYS if not reg.get(f"{k}_rel")]
    if missing:
        C.fail("F10_REGISTRY_INCOMPLETE", f"path keys {missing}")
    return {k: Path(repo) / reg[f"{k}_rel"] for k in PATH_KEYS}


def verify_f10_inputs(paths: dict, reg: dict = REG_F10, *,
                      cross_check_code: bool = True,
                      hash_checkpoints: bool = True,
                      v02g_contracts: bool = True,
                      bank_override: dict | None = None,
                      procedure_repo: Path | None = None) -> dict:
    """Strict F10 input verification (acceptance requirements 1, 3, 4).
    Every check fails with a typed code; nothing is defaulted."""
    from seqref_mri.scripts import v02_readout_diag as g
    check_registry(reg)
    out = {"registry_vs_code": (check_registry_against_code(reg)
                                if cross_check_code else "not checked")}
    # (1) authoritative facts: pinned path + full hash + sidecar + content
    fp = paths["facts"]
    fsha = _sidecar(fp, "F10_FACTS_SIDECAR")
    if fsha != reg["facts_sha256"]:
        C.fail("F10_FACTS_MISMATCH", f"{fp.name}: {fsha[:12]}")
    facts = _json(fp, "F10_FACTS_SCHEMA")
    if facts.get("schema") != reg["facts_schema"]:
        C.fail("F10_FACTS_SCHEMA", repr(facts.get("schema")))
    if procedure_repo is not None:
        out["procedure"] = historical_semantic_procedure(
            procedure_repo, reg["facts_commit"], reg)
    else:
        out["procedure"] = "registry list (fixture; no git procedure check)"
    semrec = _semantic(facts, reg["semantic_excluded"], "F10_FACTS_SEMANTIC")
    sem = semrec["semantic_sha256"]
    if sem != reg["facts_semantic_sha256"]:
        C.fail("F10_FACTS_SEMANTIC", f"{sem[:12]} != registered")
    out["facts_semantic"] = semrec
    if (facts.get("code") or {}).get("commit") != reg["facts_commit"]:
        C.fail("F10_FACTS_PROVENANCE", repr((facts.get("code") or {})
                                           .get("commit")))
    rp = (facts.get("reproduction") or {}).get("d3_pm") or {}
    for side in ("step0", "final"):
        r = rp.get(side)
        if r is None:
            C.fail("F10_ANCHOR_MISMATCH", f"facts reproduction.d3_pm.{side} "
                  "missing")
        _anchor_eq(r["psnr"]["anchor"], r["nmse_u"]["anchor"],
                   reg["anchors"][side], f"facts d3_pm.{side}")
    fi = facts.get("frozen_inputs") or {}
    for key, regk in (("holdout_manifest_sha256", "holdout_manifest_sha256"),
                      ("d3_subset_manifest_sha256", "d3_manifest_sha256")):
        if fi.get(key) != reg[regk]:
            C.fail("F10_FACTS_INPUTS", f"frozen_inputs.{key}")
    pb = fi.get("pm_bank") or {}
    if pb.get("bank_sha256") != reg["bank_sha256"] \
            or pb.get("manifest_sha256") != reg["bank_manifest_sha256"]:
        C.fail("F10_FACTS_INPUTS", "frozen_inputs.pm_bank")
    if (fi.get("v02h_evidence") or {}).get("sha256") != reg["v02h_sha256"]:
        C.fail("F10_FACTS_INPUTS", "frozen_inputs.v02h_evidence")
    ckst = fi.get("checkpoint_state_sha256_verified") or {}
    steps = [int(s) for s in reg["steps"]]
    for s in steps:
        if not isinstance(ckst.get(str(s)), str):
            C.fail("F10_FACTS_INPUTS", f"checkpoint state for step {s}")
    out["facts"] = {"path": str(fp), "sha256": fsha, "semantic": sem,
                    "commit": reg["facts_commit"]}
    # v02_facts.json (H8): same anchors
    vp = paths["v02_facts"]
    vsha = _sha(vp) if vp.is_file() else C.fail("F10_V02FACTS_MISMATCH",
                                               f"{vp} missing")
    if vsha != reg["v02_facts_sha256"]:
        C.fail("F10_V02FACTS_MISMATCH", vsha[:12])
    _sidecar(vp, "F10_V02FACTS_SIDECAR")
    pm = ((_json(vp, "F10_V02FACTS_MISMATCH").get("secondary_monitoring")
           or {}).get("d3_monitor") or {}).get("pm") or {}
    for side in ("step0", "final"):
        r = pm.get(side) or {}
        _anchor_eq(r.get("pm_psnr_mean"), r.get("pm_nmse_u_mean"),
                   reg["anchors"][side], f"v02_facts d3_monitor.pm.{side}")
    # (3) manifests: bytes vs sidecar; canonical recomputation vs stored
    #     field vs registered constant; V02G content contracts
    md = paths["manifest_dir"]
    hold, hsha = _manifest(md / "v02_holdout_manifest.json", "holdout",
                           reg["holdout_manifest_sha256"],
                           reg["holdout_manifest_file_sha256"])
    d3m, dsha = _manifest(md / "v02_d3_monitor_manifest.json", "d3_monitor",
                          reg["d3_manifest_sha256"], None)
    if v02g_contracts:      # V02G content contracts use V02G's constants
        try:
            g._verify_holdout(hold)
            g._verify_d3_subset(d3m, hold)
        except g.V02GError as exc:
            C.fail("F10_MANIFEST_CONTRACT", f"{exc.error_code}: {exc.reason}")
    if len(hold.get("entries", [])) != 199:
        C.fail("F10_MANIFEST_CONTRACT", "holdout population != 199")
    if len(d3m["entries"]) != reg["n_d3"]:
        C.fail("F10_MANIFEST_CONTRACT", "D3 subset size")
    out["manifests"] = {"holdout": {"file": hsha, "stored":
                                    hold["manifest_sha256"]},
                        "d3": {"file": dsha, "stored": d3m["manifest_sha256"]}}
    # (4) train record + checkpoints
    tp = paths["train_record"]
    tsha = _sidecar(tp, "F10_TRAIN_RECORD_SIDECAR")
    tr = _json(tp, "F10_TRAIN_RECORD_MISMATCH")
    if tr.get("schema") != "seqref-v02-train-record/1" \
            or int(tr.get("steps", -1)) != steps[-1]:
        C.fail("F10_TRAIN_RECORD_MISMATCH", "schema/steps")
    recs = {int(c["step"]): c for c in tr.get("checkpoints", [])}
    ck = {}
    for s in steps:
        c = recs.get(s)
        if c is None:
            C.fail("F10_TRAIN_RECORD_MISMATCH", f"no checkpoint record {s}")
        if c.get("state_sha256") != ckst[str(s)]:
            C.fail("F10_TRAIN_RECORD_MISMATCH", f"step {s}: state hash != "
                  "pinned facts")
        f = paths["run_root"] / c["file"]
        if not f.is_file():
            C.fail("F10_CHECKPOINT_MISSING", str(f))
        if hash_checkpoints:
            got = _sha(f)
            if got != c["file_sha256"]:
                C.fail("F10_CHECKPOINT_MISMATCH", f"{f.name}: {got[:12]}")
        ck[str(s)] = {"file": c["file"], "file_sha256": c["file_sha256"],
                      "state_sha256": c["state_sha256"],
                      "file_hashed": hash_checkpoints}
    out["train_record"] = {"sha256": tsha, "checkpoints": ck}
    # (4) parents against approved values
    par = {}
    for name in ("p0s", "p4", "implb", "v02h"):
        p = paths[name]
        if not p.is_file():
            C.fail("F10_PARENT_MISSING", str(p))
        got = _sha(p)
        if got != reg[f"{name}_sha256"]:
            C.fail("F10_PARENT_MISMATCH", f"{name}: {got[:12]}")
        par[name] = got
    try:
        g._check_v02h_content(_json(paths["v02h"], "F10_PARENT_MISMATCH"))
    except g.V02GError as exc:
        C.fail("F10_PARENT_CONTENT", f"{exc.error_code}: {exc.reason}")
    out["parents"] = par
    # (4) bank regenerated against approved values
    if bank_override is not None:
        b = bank_override
    else:
        torch, tg, d3, est, *_ = _seams()
        b = est.z_diag_bank()
    if b["bank_sha256"] != reg["bank_sha256"] \
            or b["manifest_sha256"] != reg["bank_manifest_sha256"] \
            or tuple(b["bank"].shape) != (BANK_ROWS, b["bank"].shape[1]):
        C.fail("F10_BANK_MISMATCH", f"{b['bank_sha256'][:12]}/"
              f"{b['manifest_sha256'][:12]} {tuple(b['bank'].shape)}")
    out["bank"] = {"bank_sha256": b["bank_sha256"],
                   "manifest_sha256": b["manifest_sha256"]}
    out["result"] = "INPUT_VERIFICATION_PASS"
    logger.info("[%s] F10 input verification PASS", __abbr__)
    return out


# ---------------------------------------------------------------------------
# F10 -- reproduction (separate process, historical V02G backend)
# ---------------------------------------------------------------------------
def f10_reproduce(repo: Path, data_root: Path, reg: dict = REG_F10) -> dict:
    """D3-subset PM reproduction with the V05LT R1 functions on the V02
    checkpoints ONLY. Call in a fresh process: pins the V02G backend
    first (1 thread, deterministic algorithms, TF32 off, CUBLAS)."""
    from seqref_mri.scripts import v02_readout_common as v02c
    from seqref_mri.scripts import v02_readout_diag as g
    from seqref_mri.scripts import v02_eval as v02e
    torch, tg, d3, est, ffr, StageError = _seams()
    backend = v02c.pin_deterministic_backend(torch)
    paths = f10_paths(repo, reg)
    ver = verify_f10_inputs(paths, reg, procedure_repo=repo)
    env = g._env()
    try:
        p4 = ffr.load_p4s2_parent(str(paths["p4"]))
        implb = ffr.load_implb_parent(str(paths["implb"]))
        ffr.require_spline_b(implb["spline_b"])
        s_ref = tg._s_ref_from_p0s(str(paths["p0s"]))
    except (ffr.FreeFlowError, StageError) as exc:
        C.fail("F10_PARENT_LOAD", f"{type(exc).__name__}: {exc}")
    md = paths["manifest_dir"]
    try:
        manifests = [v02e._load_manifest(md, f"v02_epoch{e}_manifest.json",
                                         "train_epoch")
                     for e in range(3)]
        record = v02e._load_train_record(str(paths["train_record"]),
                                         manifests)
        d3man = v02e._load_manifest(md, "v02_d3_monitor_manifest.json",
                                    "d3_monitor")
        recs = {int(c["step"]): c for c in record["checkpoints"]}
        models = {s: v02e._load_checkpoint(env, paths["run_root"], recs[s],
                                           implb["spline_b"])
                  for s in reg["steps"]}
        ds = env.FastMRISliceDataset(str(data_root), split="val",
                                     mode="eval")   # registered F10 exception
        states = v02e._collect_states(env, ds, d3man["entries"], p4, s_ref,
                                      reg["n_d3"])
    except g.V02GError as exc:
        C.fail("F10_V02G_LOAD", f"{exc.error_code}: {exc.reason}")
    except v02e.V02Error as exc:
        C.fail("F10_V02E_LOAD", str(exc))
    bank = est.z_diag_bank()["bank"]
    out = {"backend": backend, "input_verification": ver["result"],
           "n_slices": len(states), "endpoints": {}}
    for side, step in (("step0", reg["steps"][0]), ("final", reg["steps"][1])):
        model = models[step]
        model.eval()
        ps, ns = [], []
        for st in states:
            dec = decode_bank(model, st, bank)
            r = r1_from_decodes(st, dec, BANK_ROWS, True, "v02_holdout_all")
            ps.append(r["psnr"])
            ns.append(r["nmse_u"])
        mp, mn = float(np.mean(ps)), float(np.mean(ns))
        a = reg["anchors"][side]
        dp, rn = mp - a["psnr"], (mn - a["nmse_u"]) / a["nmse_u"]
        out["endpoints"][side] = {
            "psnr": {"measured": mp, "anchor": a["psnr"], "delta_db": dp,
                     "bit_exact": mp == a["psnr"]},
            "nmse_u": {"measured": mn, "anchor": a["nmse_u"], "delta_rel": rn,
                       "bit_exact": mn == a["nmse_u"]}}
        if not math.isfinite(dp) or abs(dp) > reg["psnr_tol_db"]:
            C.fail("F10_REPRO_PSNR", f"{side}: {mp!r} vs {a['psnr']!r}")
        if not math.isfinite(rn) or abs(rn) > reg["nmse_tol_rel"]:
            C.fail("F10_REPRO_NMSE", f"{side}: {mn!r} vs {a['nmse_u']!r}")
    out["result"] = "F10_REPRODUCTION_PASS"
    logger.info("[%s] F10 reproduction PASS %s", __abbr__,
                json.dumps(out["endpoints"]))
    return out


# ---------------------------------------------------------------------------
# I1.7(c1): epoch-0 evaluation (AM3-C), panels (194 §10.1), publication
# ---------------------------------------------------------------------------
E0_EVAL_SCHEMA = "seqref-v05lt-epoch0-eval/1"
E0_EVAL_FILE = "epoch0_eval.json"
PANEL_SCHEMA = "seqref-v05lt-panel/1"
PANEL_PCT = 99.5
PANEL_SEP = 2                          # separator columns (value 0)


def _check_case_numbers(cases: list) -> None:
    nums = [c["case"] for c in cases]
    keys = [(str(c["file"]), int(c["slice_index"])) for c in cases]
    if len(set(nums)) != len(nums) or len(set(keys)) != len(keys):
        C.fail("PANEL_CASE_DUPLICATE", f"case numbers {nums}")


def _ident_key(ident) -> tuple:
    return (str(ident["file"]), int(ident["slice_index"]))


def _is_nonfinite_error(exc) -> bool:
    """Typed non-finite failures from the readout/metric/decode seams."""
    code, txt = exc.code.upper(), str(exc).lower()
    return ("NON_FINITE" in code or "NONFINITE" in code or
            exc.taxonomy == "numerical_failure" or
            (code.startswith("METRIC_") and
             any(w in txt for w in ("finite", "nan", "inf"))))


def _epoch0_stage(where: str, fn, *a, **k):
    """AM3-E3: any NaN/Inf at epoch 0 -> EPOCH0_NONFINITE (stop)."""
    try:
        return fn(*a, **k)
    except C.V05LTError as exc:
        if _is_nonfinite_error(exc):
            C.fail("EPOCH0_NONFINITE", f"{where}: {exc.code}: {exc}",
                   "numerical_failure")
        raise
    except Exception as exc:  # noqa: BLE001 -- typed, logged, re-raised
        txt = f"{type(exc).__name__}: {exc}"
        if "NON_FINITE" in txt.upper() or "non-finite" in txt.lower():
            C.fail("EPOCH0_NONFINITE", f"{where}: {txt}", "numerical_failure")
        C.fail("EPOCH0_EVAL_FAILED", f"{where}: {txt}")


def _rng_states() -> dict:
    import random
    torch = _seams()[0]
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone()}


def _rng_equal(a: dict, b: dict) -> bool:
    torch = _seams()[0]
    na, nb = a["numpy"], b["numpy"]
    return (a["python"] == b["python"] and na[0] == nb[0] and
            np.array_equal(na[1], nb[1]) and tuple(na[2:]) == tuple(nb[2:])
            and torch.equal(a["torch"], b["torch"]))


def _rng_restore(st: dict) -> None:
    import random
    torch = _seams()[0]
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])


def _p995(a: np.ndarray, what: str) -> float:
    v = float(np.percentile(a, PANEL_PCT))
    if not (math.isfinite(v) and v > 0.0):
        C.fail("PANEL_SCALE_INVALID", f"{what}: p{PANEL_PCT} = {v!r}")
    return v


# 5x7 bitmap font (public-domain style glyphs), rows top->bottom
FONT5X7 = {
    " ": "00000 00000 00000 00000 00000 00000 00000",
    "A": "01110 10001 10001 11111 10001 10001 10001",
    "B": "11110 10001 10001 11110 10001 10001 11110",
    "C": "01110 10001 10000 10000 10000 10001 01110",
    "D": "11110 10001 10001 10001 10001 10001 11110",
    "E": "11111 10000 10000 11110 10000 10000 11111",
    "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01110 10001 10000 10111 10001 10001 01111",
    "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "01110 00100 00100 00100 00100 00100 01110",
    "J": "00111 00010 00010 00010 00010 10010 01100",
    "K": "10001 10010 10100 11000 10100 10010 10001",
    "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001",
    "N": "10001 11001 10101 10011 10001 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110",
    "P": "11110 10001 10001 11110 10000 10000 10000",
    "Q": "01110 10001 10001 10001 10101 10010 01101",
    "R": "11110 10001 10001 11110 10100 10010 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110",
    "T": "11111 00100 00100 00100 00100 00100 00100",
    "U": "10001 10001 10001 10001 10001 10001 01110",
    "V": "10001 10001 10001 10001 10001 01010 00100",
    "W": "10001 10001 10001 10101 10101 10101 01010",
    "X": "10001 10001 01010 00100 01010 10001 10001",
    "Y": "10001 10001 01010 00100 00100 00100 00100",
    "Z": "11111 00001 00010 00100 01000 10000 11111",
    "0": "01110 10001 10011 10101 11001 10001 01110",
    "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111",
    "3": "11111 00010 00100 00010 00001 10001 01110",
    "4": "00010 00110 01010 10010 11111 00010 00010",
    "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110",
    "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110",
    "9": "01110 10001 10001 01111 00001 00010 01100",
    ".": "00000 00000 00000 00000 00000 01100 01100",
    ",": "00000 00000 00000 00000 01100 00100 01000",
    "-": "00000 00000 00000 11111 00000 00000 00000",
    "+": "00000 00100 00100 11111 00100 00100 00000",
    "_": "00000 00000 00000 00000 00000 00000 11111",
    ":": "00000 01100 01100 00000 01100 01100 00000",
    "=": "00000 00000 11111 00000 11111 00000 00000",
    "/": "00000 00001 00010 00100 01000 10000 00000",
    "|": "00100 00100 00100 00100 00100 00100 00100",
    "[": "01110 01000 01000 01000 01000 01000 01110",
    "]": "01110 00010 00010 00010 00010 00010 01110",
}


COL_LABELS = {"observed": "OBSERVED", "r0": "R0", "r1": "R1", "gt": "GT",
              "r0_dc": "R0+DC", "r1_dc": "R1+DC",
              "err_observed": "|OBS-GT|", "err_r0": "|R0-GT|",
              "err_r1": "|R1-GT|", "err_r0_dc": "|R0+DC-GT|",
              "err_r1_dc": "|R1+DC-GT|"}
BAR_W = 24


def render_text(text: str, width: int) -> np.ndarray:
    """7-row bitmap of upper-cased text (5x7 glyphs, 1-px spacing), padded
    or refused to `width` -- never silently truncated."""
    text = text.upper()
    bad = sorted({ch for ch in text if ch not in FONT5X7})
    C.require(not bad, "PANEL_LABEL_CHARS", f"{bad} in {text!r}")
    C.require(6 * len(text) <= width, "PANEL_LABEL_WIDTH",
              f"{text!r} needs {6 * len(text)} px > {width}")
    out = np.zeros((7, width))
    for k, ch in enumerate(text):
        rows = FONT5X7[ch].split()
        for r, bits in enumerate(rows):
            for c, b in enumerate(bits):
                if b == "1":
                    out[r, 6 * k + c] = 1.0
    return out


def png_gray(img: np.ndarray) -> bytes:
    """Minimal 8-bit greyscale PNG (zlib; no plotting dependency)."""
    import struct
    import zlib
    a = np.ascontiguousarray(img, dtype=np.uint8)
    C.require(a.ndim == 2, "PANEL_PNG", f"ndim {a.ndim}")
    h, w = a.shape
    raw = b"".join(b"\x00" + a[i].tobytes() for i in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def build_panel(case: dict, st, images: dict, dc_row: dict) -> dict:
    """One panel (194 §10.1): Observed (ZF), R0, R1, GT, error maps; R0+DC /
    R1+DC (+ errors) only if DC is not identical within tolerance. Fixed
    scales from model-independent data only: images [0, p99.5(|GT|)],
    errors [0, p99.5(|Observed - GT|)]."""
    torch, tg, d3, est, *_ = _seams()
    C.require({"r0", "r1"} <= set(images), "PANEL_IMAGES_MISSING",
              f"case {case['case']}")
    gt = st["x_true_mag"].to(torch.float64).numpy()
    obs = est.image_from_u(st, np.zeros_like(st["u_true"])).abs() \
        .to(torch.float64).numpy()
    cols = {"observed": obs, "r0": images["r0"].abs().numpy(),
            "r1": images["r1"].abs().numpy(), "gt": gt}
    errs = {"err_observed": np.abs(obs - gt),
            "err_r0": np.abs(cols["r0"] - gt),
            "err_r1": np.abs(cols["r1"] - gt)}
    notes = []
    m, y = _dc_mask(st), y_norm(st)
    for key, src in (("r0_dc", "r0"), ("r1_dc", "r1")):
        if dc_row[key]["dc_identity"]:
            notes.append(f"{key}: DC identical within tolerance")
        else:
            xd = hard_dc(images[src], y, m).abs().numpy()
            cols[key] = xd
            errs[f"err_{key}"] = np.abs(xd - gt)
    for k, v in {**cols, **errs}.items():
        if not np.all(np.isfinite(v)):
            C.fail("EPOCH0_NONFINITE", f"panel case {case['case']} {k}",
                   "numerical_failure")
    img_hi = _p995(gt, f"case {case['case']} |GT|")
    err_hi = _p995(np.abs(obs - gt), f"case {case['case']} |Obs-GT|")
    return {"case": case, "columns": cols, "errors": errs, "notes": notes,
            "scales": {"image": [0.0, img_hi], "error": [0.0, err_hi]}}


def panel_files(panel: dict, stem: str) -> dict:
    """PNG + .npz. PNG: header lines (case identity; numeric scale
    endpoints), a label above every column, the fixed-scale tiles and two
    shared colour bars (IMG, ERR; top = scale maximum, bottom = 0)."""
    import io
    h = next(iter(panel["columns"].values())).shape[0]
    img_hi, err_hi = panel["scales"]["image"][1], panel["scales"]["error"][1]
    names = list(panel["columns"]) + list(panel["errors"])
    tiles = [np.clip(v / img_hi, 0, 1) for v in panel["columns"].values()]
    tiles += [np.clip(v / err_hi, 0, 1) for v in panel["errors"].values()]
    bar = np.linspace(1.0, 0.0, h)[:, None] * np.ones((1, BAR_W))
    blocks = list(zip([COL_LABELS[n] for n in names] + ["IMG", "ERR"],
                      tiles + [bar, bar]))
    cols = []
    for k, (lab, t) in enumerate(blocks):
        w = t.shape[1]
        cols.append(np.vstack([render_text(lab, w), np.zeros((2, w)), t]))
        if k < len(blocks) - 1:
            cols.append(np.zeros((h + 9, PANEL_SEP)))
    body = np.hstack(cols)
    W = body.shape[1]
    c = panel["case"]
    l1 = (f"CASE {c['case']} {Path(str(c['file'])).name} SLICE "
          f"{c['slice_index']}")
    l2 = f"IMG [0, {img_hi:.4E}]  ERR [0, {err_hi:.4E}]  FIXED SCALES"
    head = np.vstack([render_text(l1, W), np.zeros((2, W)),
                      render_text(l2, W), np.zeros((3, W))])
    png = png_gray(np.round(np.vstack([head, body]) * 255.0))
    buf = io.BytesIO()
    np.savez(buf, **{k: v for k, v in panel["columns"].items()},
             **{k: v for k, v in panel["errors"].items()},
             scale_image=np.array(panel["scales"]["image"]),
             scale_error=np.array(panel["scales"]["error"]))
    layout = names + ["bar_image", "bar_error"]
    return {f"{stem}.png": png, f"{stem}.npz": buf.getvalue(),
            "_meta": {"layout_left_to_right": layout,
                      "scales": panel["scales"], "notes": panel["notes"],
                      "case": panel["case"]}}


def epoch0_evaluate(model, data: dict, bank: np.ndarray, dc_tol: dict,
                    cases: list, *, all_indices=None, mid_indices=None,
                    decode_counter: dict | None = None) -> tuple:
    """AM3-C epoch-0 readouts on validation only, eval()/no-grad, RNG
    states saved and restored (AM3-B in-process rule). Production passes
    all_indices = mid_indices = None (all 3,461 / the 97 midpoints);
    explicit subsets are fixture-only. Returns (record, panels); writes
    nothing. Timing (addition C) recorded per stage."""
    import time
    torch = _seams()[0]
    check_tolerances(dc_tol)
    vv, elig = data["vv"], data["eligibility"]
    C.require(len(cases) > 0, "PANEL_CASES_EMPTY", "no panel cases")
    _check_case_numbers(cases)
    rng0 = _rng_states()
    was_training = model.training
    params0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    t = {}
    model.eval()
    try:
        with torch.no_grad():
            t0 = time.perf_counter()
            from seqref_mri.scripts import v05lt_metrics as Mx
            rows = _epoch0_stage("all-slice validation", Mx.validate, model,
                                 vv, elig, data["p4"], data["s_ref"],
                                 all_indices)
            summ = Mx.summarise(rows)
            t["all_slice_s"] = time.perf_counter() - t0
            if mid_indices is None:
                mid_indices = [i for i in range(len(vv))
                               if elig[vv.entry(i)]["midpoint"]]
            t0 = time.perf_counter()
            states = _classified("build_states", build_states, vv,
                                 mid_indices, data["p4"], data["s_ref"])
            el = [bool(elig[vv.entry(i)]["eligible"]) for i in mid_indices]
            by_key = {_ident_key(st["identity"]): st for st in states}
            keep = {}
            for c in cases:
                k = (str(c["file"]), int(c["slice_index"]))
                if k not in by_key:
                    C.fail("PANEL_CASE_MISSING", f"case {c['case']} {k} not "
                           "among the evaluated midpoint slices")
                keep[k] = {}
            cnt = decode_counter if decode_counter is not None else {}
            mrows = _epoch0_stage(
                "midpoint readouts", evaluate_states, model, states, el,
                {"primary": bank}, (BANK_ROWS,), decode_counter=cnt,
                dc_tol=dc_tol, keep_images=keep)
            t["midpoint_readouts_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            row_by_key = {_ident_key(r["identity"]): r for r in mrows}
            panels = [build_panel(c, by_key[(str(c["file"]),
                                             int(c["slice_index"]))],
                                  keep[(str(c["file"]),
                                        int(c["slice_index"]))],
                                  row_by_key[(str(c["file"]),
                                              int(c["slice_index"]))])
                      for c in cases]
            t["panels_s"] = time.perf_counter() - t0
    finally:
        model.train(was_training)
        rng1 = _rng_states()
        rng_consumed = not _rng_equal(rng0, rng1)
        _rng_restore(rng0)
    after = model.state_dict()
    C.require(all(torch.equal(params0[k], after[k]) for k in params0),
              "EPOCH0_STATE_CHANGED", "evaluation changed parameters/buffers")
    record = {
        "schema": E0_EVAL_SCHEMA, "eval_version": __version__,
        "all_slice": {"summary": summ, "rows": rows},
        "midpoint": {"rows": mrows, "dc_identity": dc_identity_summary(mrows),
                     "n": len(mrows),
                     "decodes": cnt.get("primary", 0)},
        "dc_tolerances": dc_tol,
        "cases": [c["case"] for c in cases],
        "timing": t,
        "rng": {"consumed_during_eval": rng_consumed,
                "restored": True},
        "torch_threads": torch.get_num_threads()}
    return record, panels


def _e0_registered(out_dir: Path) -> bool:
    parts = dest_path(out_dir).parts
    return any(parts[i:i + 2] == ("v05lt", "epoch0") and
               parts[i + 3] == "eval" and parts[i + 2].startswith("seed_")
               for i in range(len(parts) - 3))


def write_epoch0_eval(out_dir: Path, record: dict, panels: list,
                      context: dict, *, production: bool = False,
                      hook=None) -> dict:
    """Write-once epoch-0 evaluation directory: record JSON (strict),
    per-case PNG + .npz, SHA256SUMS; staged and published with
    renameat2(RENAME_NOREPLACE). The registered
    results/v05lt/epoch0/seed_<s>/eval is refused unless production=True
    (set only by the gated epoch-0 stage, I1.9)."""
    out_dir = dest_path(out_dir)
    if _e0_registered(out_dir) and not production:
        C.fail("EPOCH0_EVAL_GATE_REQUIRED", f"{out_dir}: registered epoch-0 "
               "evaluation is written only at the gated epoch-0 stage")
    if entry_exists(out_dir):
        C.fail("ARTIFACT_EXISTS", f"{out_dir}: an entry already exists "
               "(write-once; symlinks are never followed)")
    _check_case_numbers([p["case"] for p in panels])
    stg = out_dir.with_name(f".{out_dir.name}.staging")
    if entry_exists(stg):
        C.fail("STAGING_PRESENT", f"{stg}: left by an interrupted write; "
               "owner inspection required (not deleted)")
    files, metas = {}, []
    for pnl in panels:
        f = panel_files(pnl, f"panel_case{pnl['case']['case']}")
        metas.append(f.pop("_meta"))
        clash = set(f) & set(files)
        C.require(not clash, "PANEL_CASE_DUPLICATE", str(sorted(clash)))
        files.update(f)
    doc = dict(record, context=context, panels=metas,
               panel_sha256={k: C.sha256_bytes(v) for k, v in files.items()})
    try:
        files[E0_EVAL_FILE] = (json.dumps(doc, sort_keys=True,
                                          allow_nan=False) + "\n").encode()
    except (TypeError, ValueError) as exc:
        C.fail("EPOCH0_EVAL_JSON_INVALID", f"{type(exc).__name__}: {exc}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stg.mkdir()
    shas = {n: C.write_durable_new(stg / n, b) for n, b in sorted(files.items())}
    C.write_durable_new(stg / "SHA256SUMS", "".join(
        f"{h}  {n}\n" for n, h in sorted(shas.items())).encode())
    if hook is not None:
        hook("epoch0_eval:staged")
    publish_noreplace(stg, out_dir)
    for n, h in shas.items():
        C.require(C.sha256_file(out_dir / n) == h, "EPOCH0_EVAL_INCOMPLETE",
                  n)
    return shas


# ---------------------------------------------------------------------------
# Post-run evaluation of trained checkpoints (I1.8 b0, G2)
# ---------------------------------------------------------------------------
POSTEVAL_SCHEMA = "seqref-v05lt-posteval/1"
POSTEVAL_FILE = "posteval.json"
REGISTERED_BANK_ARRAY = {
    "primary": C.ZDIAG_BANK_SHA256,
    "B": C.BANK_ARRAY_SHA256["v05lt_bank_20261023.npy"],
    "C": C.BANK_ARRAY_SHA256["v05lt_bank_20261024.npy"]}
MASK_CONVENTION = ("validation evaluation masks: canonical_mask_seed"
                   "(EVAL_BASE_SEED, file, slice), no epoch (F02)")
# Trainer weights file (v05lt_train.WEIGHTS_SCHEMA; not imported: the
# trainer is not a dependency of the evaluator). Equality is checked in F
# group PEV.
WEIGHTS_SCHEMA = "seqref-v05lt-weights/1"
_WEIGHTS_META = {"seed": int, "arm": str, "lambda": (int, float),
                 "epoch": int, "step": int, "model_state_sha256": str,
                 "roles": list}


@dataclasses.dataclass(frozen=True)
class VerifiedCheckpoint:
    """Produced only by load_trained_checkpoint: identity read from the
    hash-verified weights file (never from the caller)."""
    file_name: str
    file_sha256: str
    model_state_sha256: str
    seed: int
    arm: str
    epoch: int
    roles: tuple

    def as_record(self) -> dict:
        return dict(dataclasses.asdict(self), roles=list(self.roles))


def load_trained_checkpoint(path: Path, file_sha256: str,
                            model_state_sha256: str, *, seed: int, arm: str,
                            epoch: int, model_factory=None):
    """Committed weights_e<NNN>.pt -> (model in eval mode,
    VerifiedCheckpoint). Verified: file sha256 = committed; schema =
    WEIGHTS_SCHEMA with exactly {schema, model, meta}; meta fields and
    types; meta seed/arm/epoch/model-state = the committed values passed by
    the caller (from the run's commit record); recomputed model-state hash
    = meta = committed. The global torch CPU RNG is saved and restored
    around build_model. model_factory: fixture-only (stub architecture);
    production passes none and the production build_model is used."""
    torch = _seams()[0]
    from seqref_mri.src import free_flow_runtime as ffr
    path = Path(path)
    got = C.sha256_file(path)
    C.require(got == file_sha256, "POSTEVAL_FILE_MISMATCH",
              f"{path.name}: {got} != committed {file_sha256}")
    rng = torch.get_rng_state()
    try:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 -- converted to a typed failure
            C.fail("POSTEVAL_FILE_INVALID",
                   f"{path.name}: {type(exc).__name__}: {exc}")
        C.require(isinstance(obj, dict) and set(obj) ==
                  {"schema", "model", "meta"} and obj["schema"] ==
                  WEIGHTS_SCHEMA and isinstance(obj["model"], dict) and
                  isinstance(obj["meta"], dict),
                  "POSTEVAL_FILE_INVALID", f"{path.name}: weights schema")
        meta = obj["meta"]
        bad = [k for k, t in _WEIGHTS_META.items()
               if k not in meta or isinstance(meta[k], bool) or
               not isinstance(meta[k], t)]
        C.require(not bad and all(isinstance(r, str) for r in meta["roles"]),
                  "POSTEVAL_FILE_INVALID", f"{path.name}: meta fields {bad}")
        want = {"seed": seed, "arm": arm, "epoch": epoch,
                "model_state_sha256": model_state_sha256}
        diff = {k: (meta[k], v) for k, v in want.items() if meta[k] != v}
        C.require(not diff, "POSTEVAL_META_MISMATCH",
                  f"{path.name}: saved != committed {diff}")
        ms = C.model_state_sha256(obj["model"])
        C.require(ms == model_state_sha256, "POSTEVAL_STATE_MISMATCH",
                  f"{path.name}: {ms} != {model_state_sha256}")
        model = (ffr.build_model(init_seed=seed) if model_factory is None
                 else model_factory(seed))
        model.load_state_dict(obj["model"], strict=True)
        model.eval()
        C.require(C.model_state_sha256(model) == ms,
                  "POSTEVAL_LOAD_MISMATCH", path.name)
    finally:
        torch.set_rng_state(rng)
    return model, VerifiedCheckpoint(
        file_name=path.name, file_sha256=got, model_state_sha256=ms,
        seed=meta["seed"], arm=meta["arm"], epoch=meta["epoch"],
        roles=tuple(meta["roles"]))


def _array_sha(a) -> str:
    return C.sha256_bytes(np.ascontiguousarray(a, dtype=np.float32)
                          .tobytes(order="C"))


def code_identity() -> dict:
    here = Path(__file__).resolve().parent
    return {"files": {n: C.sha256_file(here / n) for n in (
        "v05lt_eval.py", "v05lt_metrics.py", "v05lt_common.py")},
        "eval_version": __version__, "metrics_version": M.__version__}


def backend_identity() -> dict:
    torch = _seams()[0]
    return {"torch": torch.__version__, "numpy": np.__version__,
            "python": sys.version.split()[0],
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic": bool(
                torch.are_deterministic_algorithms_enabled())}


def posteval_evaluate(model, data: dict, task: dict, banks: dict,
                      dc_tol: dict, cases: list, *,
                      checkpoint: VerifiedCheckpoint,
                      eligibility_sha256: str, mid_indices=None,
                      decode_counter: dict | None = None) -> tuple:
    """One plan_campaign task (one distinct model state; roles listed in
    task['uses'], evaluated once). checkpoint: the VerifiedCheckpoint
    returned by load_trained_checkpoint (epoch and run identity are taken
    from it, never from a caller dictionary). Production passes
    mid_indices=None (the 97 midpoints); subsets are fixture-only. Returns (record,
    panels); writes nothing."""
    import time
    torch = _seams()[0]
    check_tolerances(dc_tol)
    ms = C.model_state_sha256(model)
    C.require(isinstance(checkpoint, VerifiedCheckpoint),
              "POSTEVAL_CHECKPOINT_UNVERIFIED", "checkpoint must come from "
              f"load_trained_checkpoint, got {type(checkpoint).__name__}")
    C.require(ms == task["model_state"] == checkpoint.model_state_sha256,
              "POSTEVAL_STATE_MISMATCH",
              f"model {ms[:12]} task {task['model_state'][:12]} checkpoint "
              f"{checkpoint.model_state_sha256[:12]}")
    C.require(checkpoint.epoch == task["epoch"] and task["epoch"] >= 1,
              "POSTEVAL_EPOCH_MISMATCH", f"{checkpoint.epoch} vs "
              f"{task['epoch']}")
    runs = {(u["seed"], u["arm"]) for u in task["uses"]}
    C.require(runs == {(checkpoint.seed, checkpoint.arm)},
              "POSTEVAL_RUN_MISMATCH", f"task uses {sorted(runs)} vs "
              f"checkpoint {(checkpoint.seed, checkpoint.arm)}")
    C.require(set(banks) == set(task["banks"]), "POSTEVAL_BANK_MISMATCH",
              f"{sorted(banks)} != task {sorted(task['banks'])}")
    bank_sha = {}
    for name, b in banks.items():
        C.require(name in REGISTERED_BANK_ARRAY and tuple(b.shape) ==
                  C.BANK_SHAPE and b.dtype == np.float32,
                  "POSTEVAL_BANK_MISMATCH", f"{name}: layout")
        bank_sha[name] = _array_sha(b)
        C.require(bank_sha[name] == REGISTERED_BANK_ARRAY[name],
                  "POSTEVAL_BANK_MISMATCH", f"{name}: {bank_sha[name]} != "
                  "registered")
    C.require(len(cases) > 0, "PANEL_CASES_EMPTY", "no panel cases")
    _check_case_numbers(cases)
    vv, elig = data["vv"], data["eligibility"]
    if mid_indices is None:
        mid_indices = [i for i in range(len(vv))
                       if elig[vv.entry(i)]["midpoint"]]
    rng0 = _rng_states()
    was_training = model.training
    params0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    t = {}
    model.eval()
    try:
        with torch.no_grad():
            t0 = time.perf_counter()
            states = _classified("build_states", build_states, vv,
                                 mid_indices, data["p4"], data["s_ref"])
            el = [bool(elig[vv.entry(i)]["eligible"]) for i in mid_indices]
            mh = hashlib.sha256()
            for st in states:
                mh.update(np.ascontiguousarray(
                    st["mask"].numpy()).tobytes(order="C"))
            ids = [[str(st["identity"]["file"]),
                    int(st["identity"]["slice_index"])] for st in states]
            by_key = {_ident_key(st["identity"]): st for st in states}
            keep = {}
            for c in cases:
                k = (str(c["file"]), int(c["slice_index"]))
                if k not in by_key:
                    C.fail("PANEL_CASE_MISSING", f"case {c['case']} {k} not "
                           "among the evaluated midpoint slices")
                keep[k] = {}
            cnt = decode_counter if decode_counter is not None else {}
            mrows = _classified(
                "posteval readouts", evaluate_states, model, states, el,
                banks, tuple(task["prefixes"]), decode_counter=cnt,
                prefix_bank=task.get("prefix_bank", "primary"),
                dc_tol=dc_tol, keep_images=keep)
            t["midpoint_readouts_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            row_by_key = {_ident_key(r["identity"]): r for r in mrows}
            panels = [build_panel(c, by_key[k], keep[k], row_by_key[k])
                      for c in cases
                      for k in [(str(c["file"]), int(c["slice_index"]))]]
            t["panels_s"] = time.perf_counter() - t0
    finally:
        model.train(was_training)
        rng1 = _rng_states()
        rng_consumed = not _rng_equal(rng0, rng1)
        _rng_restore(rng0)
    after = model.state_dict()
    C.require(all(torch.equal(params0[k], after[k]) for k in params0),
              "POSTEVAL_STATE_CHANGED", "evaluation changed parameters")
    record = {
        "schema": POSTEVAL_SCHEMA, "eval_version": __version__,
        "checkpoint": checkpoint.as_record(),
        "uses": sorted(task["uses"], key=lambda u: (u["seed"], u["arm"],
                                                    u["role"])),
        "banks": bank_sha, "prefixes": list(task["prefixes"]),
        "prefix_bank": task.get("prefix_bank", "primary"),
        "masks": {"convention": MASK_CONVENTION,
                  "sha256": mh.hexdigest()},
        "population": {"scope": "midpoint", "n": len(mrows),
                       "identities_sha256": C.sha256_bytes(
                           C.canonical_json(ids)),
                       "eligibility_manifest_sha256": eligibility_sha256},
        "code": code_identity(), "backend": backend_identity(),
        "midpoint": {"rows": mrows, "dc_identity": dc_identity_summary(mrows),
                     "decodes": dict(cnt)},
        "dc_tolerances": dc_tol, "cases": [c["case"] for c in cases],
        "timing": t, "rng": {"consumed_during_eval": rng_consumed,
                             "restored": True}}
    return record, panels


def _posteval_registered(out_dir: Path) -> bool:
    parts = dest_path(out_dir).parts
    return any(parts[i:i + 2] == ("v05lt", "posteval")
               for i in range(len(parts) - 1))


def write_posteval(out_dir: Path, record: dict, panels: list,
                   context: dict, *, production: bool = False,
                   hook=None) -> dict:
    """Write-once post-run evaluation directory (record JSON, panels,
    SHA256SUMS), staged and published with renameat2(RENAME_NOREPLACE);
    the registered results/v05lt/posteval/... is refused unless
    production=True (gated launcher stage, I1.9). The directory name must
    be the record's model-state hash (one directory per distinct
    checkpoint: deduplication across roles)."""
    out_dir = dest_path(out_dir)
    if _posteval_registered(out_dir) and not production:
        C.fail("POSTEVAL_GATE_REQUIRED", f"{out_dir}: registered post-run "
               "evaluation is written only at the gated stage")
    C.require(out_dir.name == record["checkpoint"]["model_state_sha256"],
              "POSTEVAL_DIR_NAME", f"{out_dir.name} != model-state hash")
    if entry_exists(out_dir):
        C.fail("ARTIFACT_EXISTS", f"{out_dir}: an entry already exists "
               "(write-once; symlinks are never followed)")
    _check_case_numbers([p["case"] for p in panels])
    declared = record.get("cases")
    got = sorted(p["case"]["case"] for p in panels)
    C.require(isinstance(declared, list) and len(declared) > 0 and
              len(set(declared)) == len(declared) and
              got == sorted(declared), "POSTEVAL_PANELS_MISMATCH",
              f"panels {got} must match the declared cases {declared} "
              "exactly")
    stg = out_dir.with_name(f".{out_dir.name}.staging")
    if entry_exists(stg):
        C.fail("STAGING_PRESENT", f"{stg}: left by an interrupted write; "
               "owner inspection required (not deleted)")
    files, metas = {}, []
    for pnl in panels:
        f = panel_files(pnl, f"panel_case{pnl['case']['case']}")
        metas.append(f.pop("_meta"))
        clash = set(f) & set(files)
        C.require(not clash, "PANEL_CASE_DUPLICATE", str(sorted(clash)))
        files.update(f)
    doc = dict(record, context=context, panels=metas,
               panel_sha256={k: C.sha256_bytes(v) for k, v in files.items()})
    try:
        files[POSTEVAL_FILE] = (json.dumps(doc, sort_keys=True,
                                           allow_nan=False) + "\n").encode()
    except (TypeError, ValueError) as exc:
        C.fail("POSTEVAL_JSON_INVALID", f"{type(exc).__name__}: {exc}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stg.mkdir()
    shas = {n: C.write_durable_new(stg / n, b) for n, b in sorted(files.items())}
    C.write_durable_new(stg / "SHA256SUMS", "".join(
        f"{h}  {n}\n" for n, h in sorted(shas.items())).encode())
    if hook is not None:
        hook("posteval:staged")
    publish_noreplace(stg, out_dir)
    for n, h in shas.items():
        C.require(C.sha256_file(out_dir / n) == h, "POSTEVAL_INCOMPLETE", n)
    return shas


def _panel_names(case_no) -> set:
    return {f"panel_case{case_no}.png", f"panel_case{case_no}.npz"}


def load_posteval(out_dir: Path) -> dict:
    """Verify a published post-run directory -> record. Checks: SHA256SUMS
    well formed and = the directory's files; every hash; JSON and required
    structure; schema; directory name = model-state hash; panel files =
    exactly PNG + NPZ per declared case, and panel_sha256 = SHA256SUMS.
    Malformed content -> POSTEVAL_FILE_INVALID, missing or altered files
    -> POSTEVAL_INCOMPLETE (never a raw exception)."""
    out_dir = Path(out_dir)
    sums = out_dir / "SHA256SUMS"
    C.require(sums.is_file(), "POSTEVAL_INCOMPLETE", f"{out_dir}: SHA256SUMS")
    listed = {}
    try:
        lines = sums.read_text().splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        C.fail("POSTEVAL_FILE_INVALID", f"SHA256SUMS: {exc}")
    for line in lines:
        h, sep, n = line.partition("  ")
        C.require(sep and len(h) == 64 and n and "/" not in n and
                  n not in listed, "POSTEVAL_FILE_INVALID",
                  f"SHA256SUMS line {line!r}")
        listed[n] = h
    present = {p.name for p in out_dir.iterdir()} - {"SHA256SUMS"}
    C.require(present == set(listed), "POSTEVAL_INCOMPLETE",
              f"listed {sorted(listed)[:3]} present {sorted(present)[:3]}")
    for n, h in listed.items():
        C.require((out_dir / n).is_file() and
                  C.sha256_file(out_dir / n) == h, "POSTEVAL_INCOMPLETE", n)
    C.require(POSTEVAL_FILE in listed, "POSTEVAL_INCOMPLETE", POSTEVAL_FILE)
    try:
        doc = json.loads((out_dir / POSTEVAL_FILE).read_bytes())
    except ValueError as exc:
        C.fail("POSTEVAL_FILE_INVALID", f"{POSTEVAL_FILE}: {exc}")
    ck = doc.get("checkpoint") if isinstance(doc, dict) else None
    cases = doc.get("cases") if isinstance(doc, dict) else None
    psha = doc.get("panel_sha256") if isinstance(doc, dict) else None
    C.require(isinstance(doc, dict) and doc.get("schema") == POSTEVAL_SCHEMA
              and isinstance(ck, dict) and
              isinstance(ck.get("model_state_sha256"), str) and
              isinstance(cases, list) and cases and
              all(isinstance(c, int) and not isinstance(c, bool)
                  for c in cases) and
              isinstance(psha, dict) and isinstance(doc.get("panels"), list)
              and isinstance(doc.get("midpoint"), dict),
              "POSTEVAL_FILE_INVALID", f"{out_dir}: record structure")
    C.require(out_dir.name == ck["model_state_sha256"],
              "POSTEVAL_FILE_INVALID", f"{out_dir.name} != model-state hash")
    nums = list(cases)
    C.require(len(set(nums)) == len(nums) and len(doc["panels"]) ==
              len(nums), "POSTEVAL_FILE_INVALID", f"cases {nums}")
    want = set().union(*(_panel_names(n) for n in nums))
    C.require(set(listed) - {POSTEVAL_FILE} == want and set(psha) == want,
              "POSTEVAL_INCOMPLETE", f"panel files {sorted(set(listed))} "
              f"vs declared cases {nums}")
    C.require(all(psha[n] == listed[n] for n in want),
              "POSTEVAL_FILE_INVALID", "panel_sha256 != SHA256SUMS")
    return doc


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    argparse.ArgumentParser(description="V05LT evaluator").parse_args(argv)
    try:
        C.fail("EVAL_PRODUCTION_DISABLED", "production evaluation is entered "
               "only through the gated stages (epoch-0 stage, launcher, "
               "gate 2a); this version exposes functions to fixtures only")
    except C.V05LTError as exc:
        print(f"EVALUATOR REFUSED {exc.code}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
