# SEQREF-V02H v0.6 -- scripts.v02_readout_selftest
# LIFETIME: KEEP
# =============================================================================
# Purpose: readout selftest fixture (SEQREF-V02H v0.1 spec r3, 2026-09-02,
#          LudWiki id 146 DRAFT/PRE-EXECUTION; replaces V02AMEND v0.7 SS21
#          items 5-7). Measures whether the R2 measured-region DC overwrite
#          in the production readout seam is a NO-OP (Outcome A) or
#          LOAD-BEARING (Outcome B) on the frozen 32-slice D3 monitor
#          subset, and reproduces the registered PM/C0 anchors. The R2
#          question concerns the production R0 z=0 readout: the
#          structural decision is made on the z=0 decode ONLY, and the
#          FINAL checkpoint alone decides R2 retention; step 0 is
#          measured and recorded as interpretive context (spec r3 SS9)
#          and can neither block nor reverse the outcome. Evidence
#          only, NO verdict field; the R2 keep/remove decision is an
#          input to the V02AMEND v1.0 assembly, never taken here.
# Invocation modes (v02_selftest precedent):
#   * --mode fixtures (default): the full fixture suite on synthetic data
#     through the REAL production decode assembly; publishes NOTHING
#   * --mode authoritative: the host run on the frozen inputs (hash-
#     verified manifests/train-record/checkpoints, pinned parents, the
#     frozen D3 32-slice subset) + publication of the facts artefact
#     (schema seqref-v02h-selftest-facts/1) via the claim-guarded
#     publication machinery
# Taxonomy: exit 0 = fixtures PASS / authoritative facts complete;
#   exit 2 = ERROR. The structural AMBIGUOUS outcome is the ONE exit-2
#   path that publishes FIRST: the complete evidence artifact (both
#   statistics per slice) materialises before the stage exits 2
#   (spec r3 SS8); true implementation/provenance failures remain
#   immediate no-publish ERRORs. There is no exit 1 and no skip: an
#   unexercised registered fixture or an ERROR path that fails to
#   fire is itself an ERROR. No fallback, no mock, no placeholder, no
#   silent pass. The stage issues NO scientific verdict (evidence-only,
#   V02AMEND v0.7 SS6 carried).
# Frozen decision constants (V02H spec r3 SS8; K FROZEN by the owner
#   2026-09-02; floor_rel is a pre-registered INDEPENDENT engineering
#   tolerance, NOT DCPROBE-derived -- a floor failure is an ERROR and
#   the floor is never loosened):
#   * R2_NONTRIVIAL(slice, endpoint) <=>
#       pre_rms_rel > K x floor_rms_rel  AND  pre_max_rel > K x floor_max_rel
#     exactly one statistic above -> AMBIGUOUS (structural, exit 2)
#   * ZERO-FLOOR SUBSTITUTION (per metric, spec r3 SS8): an exactly-zero
#     measured floor is replaced by the pre-registered positive
#     floor_rel=1e-6 before the K-comparison; a nonzero measured floor,
#     however small, is used as measured
#   * floors in the SAME relative metrics as the pre-projection
#     statistics; DCPROBE (1e-5 validity / 3.29e-07 measured, ABSOLUTE
#     scale) is a qualitative cross-check only, never a substitution
#   * decision endpoint: the FINAL checkpoint alone decides R2
#     retention; per-endpoint routing = all slices False ->
#     OUTCOME_A_R2_NO_OP, all True -> OUTCOME_B_R2_NONTRIVIAL, any
#     mixed/AMBIGUOUS -> the structural R2_OUTCOME_AMBIGUOUS state:
#     the COMPLETE evidence artifact (both statistics per slice) is
#     published FIRST, then the stage exits 2; the lock decision
#     returns to the owner; no silent tie-break (spec r3 SS8); step 0
#     is interpretive context only and a step-0/final divergence is
#     RECORDED, never an ERROR
# Streams: PCG64(4) keeps its registered role EXACTLY (the reserved
#   V02AMEND paired bootstrap, registry audit PASS 2026-09-02): its
#   single consumption is the VOLUME-level paired index matrix
#   integers(0, 199, size=(10000, 199)) -- one matrix, identical
#   resampled indices for R0 and Rk. V02H only PRE-VALIDATES that
#   machinery on synthetic paired arrays (fixtures mode); this stage
#   consumes the stream for NO scientific bootstrap (the R2 decision
#   needs none). Synthetic fixture data uses a SEPARATE fixed test-only
#   seed (FIXTURE_DATA_SEED, PCG64(20260910)), fixtures mode only,
#   explicitly OUTSIDE the campaign RNG registry. The PM reproduction
#   bank is the registered PCG64(0) Z_DIAG bank, regenerated via
#   estimators.z_diag_bank() and SHA-pinned against the registered
#   constants -- no new stream consumption.
# Deterministic execution (spec r3 deterministic-pinning section): both
#   modes pin single-thread + torch.use_deterministic_algorithms(True)
#   + TF32 disabled (matmul/cudnn) + cuDNN deterministic/benchmark-off,
#   with CUBLAS_WORKSPACE_CONFIG EXPLICITLY pinned -- a conflicting
#   pre-existing value REFUSES (BACKEND_CONFIG_CONFLICT), never a
#   setdefault that silently keeps a foreign configuration; the exact
#   backend state is RECORDED in facts, and unexpected rerun variance
#   is a DEFECT -- mechanised by the SEMANTIC_RERUN_MISMATCH
#   publication guard. The readout path is CPU-only (map_location="cpu"
#   throughout the inherited V02E loaders), so the CUDA flags are
#   inert-but-recorded and the PM/C0 anchors remain reproducible.
# Structural verification (spec r3 SS10): the R2 overwrite seam is
#   verified by the frozen TWO-STAGE contract -- STAGE 1 (pre-inverse-
#   FFT, bitwise): the acquired entries of the composed k-space are
#   EXACTLY the DC write and the unmeasured entries are BITWISE the
#   probe; STAGE 2 (post round-trip, tolerance-based): BOTH regions
#   return within floor_rel. Construction-by-formula is NOT a test of
#   the contract: corrupted constructions REFUSE
#   (R2_OVERWRITE_CONTRACT_VIOLATION /
#   R2_ROUNDTRIP_CONTRACT_VIOLATION), exercised by the corrupted-DC
#   negative fixtures in f06.
# Reuse seams (verbatim, zero reimplementation):
#   * v02e._load_sidecar_json / _load_manifest / _load_train_record /
#     _load_checkpoint / _collect_states / _measure_condition /
#     _no_verdict_scan -- frozen-input verification + D3 machinery
#   * tg._build_slice_states / tg._decode_z / tg._s_ref_from_p0s --
#     production slice-state construction and the production complex
#     decode seam (COMPLEX image, never magnitude)
#   * dec.decode_normalised / dec.build_coordinate_map /
#     dec.column_mask_tensor + fdm.fft2c/ifft2c -- the registered
#     exact-DC readout assembly and frozen Fourier pair
#   * d3.derangement / estimators.z_diag_bank -- PM reproduction
#   * v02m.d3_monitor_positions -- frozen D3 draw re-derivation
#   * preflight_parents.attach_semantic_hash / publish_stage --
#     publication
# Environment-bound imports (torch, the dataset, the model machinery)
#   are LAZY and TOTAL via _env()/_env_array(): ImportError propagates
#   as ERROR; there is no fallback path. Pure logic stays importable
#   without the data environment so the fixture suite can exercise it
#   in isolation.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from seqref_mri.scripts import v02_eval as v02e
from seqref_mri.scripts import v02_manifests as v02m

logger = logging.getLogger("seqref_mri.v02_readout_selftest")

__version__ = "0.6"
__abbr__ = "SEQREF-V02H"

FACTS_SCHEMA = "seqref-v02h-selftest-facts/1"
FACTS_PREFIX = "v02_readout_selftest"  # the registered KEEP-artifact
                                       # identity (spec r3: Output
                                       # results/_diag/v02_readout_
                                       # selftest/v02_readout_selftest.
                                       # json); the SCHEMA stays
                                       # seqref-v02h-selftest-facts/1
STAGE = "SEQREF-V02H"
EXIT_PASS = 0
EXIT_ERROR = 2

# ---------------------------------------------------------------------------
# Frozen decision constants (quoted from the V02H spec r3; never CLI-tunable)
# ---------------------------------------------------------------------------
K_SEPARATION = 10.0           # FROZEN (owner 2026-09-02, spec r3 SS8)
FLOOR_REL = 1e-6              # pre-registered INDEPENDENT engineering
                              # tolerance on BOTH relative floor metrics;
                              # a floor failure is an ERROR, never loosened
PSNR_REPRO_TOL_DB = 1e-3      # accepted tolerance (V02AMEND v0.7 SS21 item 4)
NMSE_REPRO_TOL_REL = 1e-4     # accepted tolerance (item 4), relative

STEP0, FINAL_STEP = 0, 3258            # the two locked endpoints; the
                                       # FINAL checkpoint alone decides
                                       # R2 retention (spec r3 SS8/SS9)
FLOW_DIM = 13824                       # == ffr.FLOW_DIM_REAL (quoted)
N_FIXTURE_SLICES = 32                  # == v02m.D3_SUBSET_N (frozen D3 set)
BOOTSTRAP_SEED = 4                     # PCG64(4), the reserved V02AMEND
                                       # paired-bootstrap stream
BOOTSTRAP_B = 10000
BOOTSTRAP_N_VOLUMES = 199              # == v02m.N_HOLDOUT_VOLUMES; the
                                       # amendment bootstrap resamples
                                       # VOLUMES, never slices
# Quoted V02AMEND acceptance constants (owner-accepted 2026-09-02). The
# registered rule has TWO conjuncts per statistic -- the practical-
# effect threshold on the OBSERVED MEAN and the uncertainty threshold
# on the CI bound, never one substituted for the other:
#   D1 pass <=> mean paired dPSNR >= +0.50 dB AND CI95 lower > 0
#   D2 pass <=> mean ratio <= 0.95            AND CI95 upper < 1.0
# Consumed by _d1d2_acceptance, exercised by the bootstrap-machinery
# FIXTURES only -- this stage takes no D1/D2 decision.
D1_PAIRED_PSNR_MIN_DB = 0.50           # D1 practical-effect threshold
D1_CI_NULL = 0.0                       # D1 uncertainty threshold: CI95
                                       # lower STRICTLY above null
D2_RATIO_MAX = 0.95                    # D2 practical-effect threshold
D2_CI_NULL = 1.0                       # D2 uncertainty threshold: CI95
                                       # upper STRICTLY below null
FIXTURE_DATA_SEED = 20260910           # fixed TEST-ONLY seed for the
                                       # synthetic fixture content;
                                       # fixtures mode only, explicitly
                                       # OUTSIDE the campaign RNG
                                       # registry (no scientific path
                                       # consumes it)
STREAM_CONSUMPTION_ORDER = [
    "paired_bootstrap_index_matrix: PCG64(4).integers(0, 199, "
    "size=(10000, 199)) -- one volume-level matrix, identical "
    "resampled indices for R0 and Rk; the single and only consumption "
    "of the reserved stream"]

OUTCOME_A = "OUTCOME_A_R2_NO_OP"
OUTCOME_B = "OUTCOME_B_R2_NONTRIVIAL"
ROUTE_AMBIGUOUS = "R2_OUTCOME_AMBIGUOUS"  # the structural AMBIGUOUS
                                          # state (spec r3 SS8): RMS and
                                          # max-guard disagree or slices
                                          # are mixed -> the COMPLETE
                                          # evidence artifact (both
                                          # statistics per slice) is
                                          # published FIRST, then the
                                          # stage exits 2; the lock
                                          # decision returns to the
                                          # owner; no silent tie-break

# ---------------------------------------------------------------------------
# Registered PM/C0 reproduction anchors (quoted verbatim from the
# authoritative v02_facts.json, schema seqref-v02-facts/1,
# secondary_monitoring blocks). These verify the PM and likelihood
# machinery ONLY: no registered D3-subset z=0 absolute anchors exist,
# so this stage does NOT verify z=0 R0 metric reproduction; the
# full-holdout R0 identity is V02G scope, and no new anchor is
# manufactured here.
# ---------------------------------------------------------------------------
PM_STEP0_PSNR = 30.794137818416605
PM_STEP0_NMSE = 1.012275318293053
PM_FINAL_PSNR = 30.786611478861552
PM_FINAL_NMSE = 1.0141107234837616
C0_NLL_BATCH_FINAL = 17985.7734375    # exact tie convention (float ==)
PM_BANK_SHA256 = ("f8a6a34b6b474286359e4c99341de256b62bc0dff1e6bc8cf52e"
                  "72f0461dc8d6")
PM_BANK_MANIFEST_SHA256 = ("c4d2c0ec7c908f7c20a4f287eeffbdf966924a5118b"
                           "6ba734b0a329942ab7e5f")
D3_SUBSET_MANIFEST_SHA256 = ("66647202aead2edbbbea338194f464a44dc77b63e"
                             "65824a56ff2998f40879bcd")

# ---------------------------------------------------------------------------
# Synthetic-fixture mask (fixtures mode only): a FIXED representative 96-
# # column 4x-Cartesian-plus-ACS pattern. Scientifically inert -- the
# fixture exercises the measurement chain, never mask realism; the
# authoritative mode uses each slice's REAL mask via tg._build_slice_states.
# ---------------------------------------------------------------------------
FIXTURE_MASK_COLS = sorted(set(range(0, 96, 4)) | set(range(44, 52)))
FIXTURE_N_FREE = 96 * (96 - len(FIXTURE_MASK_COLS))     # 6336

# Golden pins (2026-09-10; the bootstrap pin against NumPy PCG64(4),
# the content pin against the test-only PCG64(20260910); computed from
# the registered generator calls, embedded verbatim -- a generator or
# consumption-order change FAILS the pin, never silently re-pins).
GOLDEN_BOOT_IDX_SHA256 = ("6124ac52087fcbacceb907a32a226ec1d4210dd4ad"
                          "1abcc58da021c1a9a5233c")
GOLDEN_SYNTH_CONTENT_SHA256 = ("166acbf911da95c9a35a296239b9a66aa8e51ed"
                               "c1ff6dd046f560b4f53301502")

# The pinned deterministic cuBLAS workspace configuration. Torch accepts
# exactly two values for deterministic cuBLAS behaviour (":4096:8" and
# ":16:8"); any OTHER pre-existing value conflicts with the pinned
# deterministic contract and REFUSES -- never a setdefault that would
# silently keep a foreign configuration.
CUBLAS_CONFIG = ":4096:8"
CUBLAS_ACCEPTED = (":4096:8", ":16:8")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise v02m.V02Error(f"{code}: {message}")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Lazy, total environment imports (no fallback path; ImportError = ERROR).
# _env_array(): the array-level seam (torch + the registered decoder/Fourier
#   pair) -- needed by BOTH modes.
# _env(): adds the model/dataset/parent machinery -- authoritative only.
# ---------------------------------------------------------------------------

def _env_array() -> SimpleNamespace:
    try:
        from seqref_mri.tdiag import _bootstrap  # noqa: F401 --
        # registered path bootstrap (same contract as SEQREF-V02E v0.2):
        # legacy top-level names resolve to the SAME module objects
        import torch
        from seqref_mri.src import residual_decoder as dec
        from seqref_mri.src.fastmri_data import fft2c, ifft2c
    except ImportError as exc:
        _fail("ENV_IMPORT_FAILED",
              f"an array-level dependency is not importable in this "
              f"environment: {exc}")
    return SimpleNamespace(torch=torch, dec=dec, fft2c=fft2c,
                           ifft2c=ifft2c)


def _env() -> SimpleNamespace:
    """Full environment (the registered scientific host), built on the
    array seam. Every dependency is imported lazily and totally."""
    aenv = _env_array()
    try:
        from torch.utils.data import DataLoader, Subset
        from preflight_io import utc_stamp
        from preflight_parents import (StageError, attach_semantic_hash,
                                       publish_stage)
        from seqref_mri.src import free_flow_runtime as ffr
        from seqref_mri.src.fastmri_data import FastMRISliceDataset
        from seqref_mri.scripts import tiny_gate as tg
        from seqref_mri.scripts.train_base import _collate
        from seqref_mri.scripts.v02_train import state_sha256
        from seqref_mri.tdiag import d2b, d3, estimators
    except ImportError as exc:
        _fail("ENV_IMPORT_FAILED",
              f"an environment-bound dependency is not importable in "
              f"this environment: {exc}")
    # The namespace provisions EVERY attribute the reused V02E seams
    # touch -- mirroring v02e._env() (DataLoader/Subset/collate for
    # _iter_state_batches, d2b for _encode, d3/estimators/tg/ffr/
    # state_sha256/StageError for the measurement and checkpoint
    # paths) -- plus the array seam (dec/fft2c/ifft2c) for the readout
    # measurement itself.
    return SimpleNamespace(
        torch=aenv.torch, dec=aenv.dec, fft2c=aenv.fft2c, ifft2c=aenv.ifft2c,
        DataLoader=DataLoader, Subset=Subset, collate=_collate,
        utc_stamp=utc_stamp, StageError=StageError,
        attach_semantic_hash=attach_semantic_hash,
        publish_stage=publish_stage, ffr=ffr,
        FastMRISliceDataset=FastMRISliceDataset, tg=tg,
        state_sha256=state_sha256, d2b=d2b, d3=d3, estimators=estimators)


# ---------------------------------------------------------------------------
# Deterministic execution (spec r3 deterministic-pinning section). Both
# modes pin the backend BEFORE any tensor work; the exact state is
# recorded in facts. Unexpected rerun variance is a DEFECT, mechanised
# by the SEMANTIC_RERUN_MISMATCH publication guard.
# ---------------------------------------------------------------------------

def _pin_deterministic_backend(torch) -> dict:
    """Pin and record the deterministic backend state. The readout path
    is CPU-only (the inherited V02E loaders map checkpoints to CPU), so
    the CUDA flags are inert-but-recorded; the CPU kernels used
    (linear/RQ-spline/FFT) are unaffected by the deterministic flag, so
    the registered PM/C0 anchors remain reproducible. If any op ever
    lacks a deterministic implementation, torch raises and the run
    ERRORs -- no fallback. The cuBLAS workspace configuration is an
    EXPLICIT pin/refusal: an unset variable is assigned CUBLAS_CONFIG,
    a pre-existing torch-accepted value is kept, and any other
    pre-existing value is a configuration conflict -- ERROR, because
    the pinned configuration could not be guaranteed."""
    cublas_pre = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas_pre is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CONFIG
        cublas_source = "assigned"
    elif cublas_pre in CUBLAS_ACCEPTED:
        cublas_source = "preexisting_accepted"
    else:
        _fail("BACKEND_CONFIG_CONFLICT",
              f"CUBLAS_WORKSPACE_CONFIG is pre-set to {cublas_pre!r}, "
              f"which is not one of the deterministic cuBLAS workspace "
              f"configurations torch accepts {CUBLAS_ACCEPTED!r}; the "
              f"pinned configuration cannot be guaranteed and a foreign "
              f"environment is never silently kept or overridden")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {"torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "num_threads": torch.get_num_threads(),
            "deterministic_algorithms":
                torch.are_deterministic_algorithms_enabled(),
            "tf32_matmul_allowed":
                torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn_allowed": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config":
                os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "cublas_source": cublas_source,
            "device_path": "cpu (checkpoints map_location=cpu; the "
                           "inherited V02E readout path)",
            "rerun_variance": "a scientifically different rerun is a "
                              "DEFECT: the SEMANTIC_RERUN_MISMATCH "
                              "publication guard refuses to publish "
                              "alongside a divergent artefact"}


# ---------------------------------------------------------------------------
# Streams (see header): PCG64(4) keeps its registered V02AMEND paired-
# bootstrap role; the synthetic fixture content lives on a separate fixed
# test-only seed outside the registry.
# ---------------------------------------------------------------------------

def _boot_index_matrix() -> np.ndarray:
    """The single and only consumption of the reserved PCG64(4) stream
    in this stage: the VOLUME-level paired bootstrap index matrix --
    one matrix, identical resampled indices for R0 and Rk (the
    V02AMEND paired-bootstrap semantics)."""
    return np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, BOOTSTRAP_N_VOLUMES,
        size=(BOOTSTRAP_B, BOOTSTRAP_N_VOLUMES))


def _verify_stream_pins(boot_idx: np.ndarray) -> dict:
    b_sha = hashlib.sha256(boot_idx.tobytes(order="C")).hexdigest()
    if b_sha != GOLDEN_BOOT_IDX_SHA256:
        _fail("STREAM_PIN_MISMATCH",
              f"paired bootstrap index matrix sha256 {b_sha[:12]}... "
              f"!= the embedded golden pin "
              f"{GOLDEN_BOOT_IDX_SHA256[:12]}...; a generator or "
              f"consumption-order change fails the pin, never re-pins")
    return {"generator": "PCG64", "seed": BOOTSTRAP_SEED,
            "role": "the reserved V02AMEND paired-bootstrap stream; "
                    "V02H PRE-VALIDATES the machinery on synthetic "
                    "paired arrays (fixtures mode) and consumes the "
                    "stream for NO scientific bootstrap in this stage",
            "consumption_order": list(STREAM_CONSUMPTION_ORDER),
            "bootstrap_index_matrix_sha256": b_sha}


def _draw_synthetic_content() -> dict:
    """FIXTURES MODE ONLY: 32 synthetic slices for the REAL decode
    assembly, drawn from the fixed test-only seed (OUTSIDE the campaign
    RNG registry). Draw order per slice (locked): y_re, y_im,
    amax-normal, u_re, u_im (ONE decode point per slice -- the R2
    decision is z=0-only, so the fixture is single-decode)."""
    rng = np.random.Generator(np.random.PCG64(FIXTURE_DATA_SEED))
    y_stack = np.empty((N_FIXTURE_SLICES, 96, 96), dtype=np.complex128)
    amax_arr = np.empty(N_FIXTURE_SLICES, dtype=np.float64)
    u_stack = np.empty((N_FIXTURE_SLICES, FIXTURE_N_FREE),
                       dtype=np.complex128)
    for j in range(N_FIXTURE_SLICES):
        re = rng.standard_normal((96, 96))
        im = rng.standard_normal((96, 96))
        y_stack[j] = re + 1j * im
        amax_arr[j] = float(np.exp(rng.standard_normal()))
        ure = rng.standard_normal(FIXTURE_N_FREE)
        uim = rng.standard_normal(FIXTURE_N_FREE)
        u_stack[j] = (ure + 1j * uim) * (0.01 * amax_arr[j])
    return {"y": y_stack, "amax": amax_arr, "u": u_stack}


def _content_sha256(content: dict) -> str:
    return hashlib.sha256(content["y"].tobytes(order="C")
                          + content["amax"].tobytes(order="C")
                          + content["u"].tobytes(order="C")).hexdigest()


def _verify_content_pin(content: dict) -> str:
    sha = _content_sha256(content)
    if sha != GOLDEN_SYNTH_CONTENT_SHA256:
        _fail("STREAM_PIN_MISMATCH",
              f"synthetic fixture content sha256 {sha[:12]}... != the "
              f"embedded golden pin "
              f"{GOLDEN_SYNTH_CONTENT_SHA256[:12]}...")
    return sha


# ---------------------------------------------------------------------------
# Measurement core (array-level; model-free -- BOTH modes feed it through
# the same code path: fixtures via the real dec.decode_normalised on
# synthetic content, authoritative via tg._decode_z on the real model).
# All tensors are (96, 96) complex; y_raw complex, amax a positive scalar.
# ---------------------------------------------------------------------------

def _three_stats(aenv, delta_m, ref_m, support, where: str) -> dict:
    """The three registered DC residual statistics over the acquired
    support OMEGA (spec r3 SS7): delta_DC = the MAXIMUM COEFFICIENT-WISE
    ABSOLUTE RESIDUAL (headline, absolute scale -- the DCPROBE
    quantity), dc_rms_rel (PRIMARY), dc_max_rel (guard). support is the
    boolean ACQUIRED mask (from the registered column mask, never from
    |y| > 0: an acquired coefficient whose value is exactly zero is
    still acquired). delta_m/ref_m are full-grid complex tensors
    already masked to the measured region."""
    torch = aenv.torch
    if not bool(support.any().item()):
        _fail("STATS_DENOMINATOR_INVALID",
              f"{where}: the acquired support is empty; the relative "
              f"statistics are undefined and never coerced")
    dv = delta_m[support].to(torch.complex128)
    rv = ref_m[support].to(torch.complex128)
    ref_rms = float(torch.sqrt(torch.mean(torch.abs(rv) ** 2)).item())
    ref_max = float(torch.max(torch.abs(rv)).item())
    for name, den in (("rms", ref_rms), ("max", ref_max)):
        if not math.isfinite(den) or den <= 0.0:
            _fail("STATS_DENOMINATOR_INVALID",
                  f"{where}: measured-region reference {name} "
                  f"denominator is not finite and positive ({den!r})")
    dc_abs = float(torch.max(torch.abs(dv)).item())
    rms = float(torch.sqrt(torch.mean(torch.abs(dv) ** 2)).item())
    for name, v in (("delta_dc", dc_abs), ("rms", rms)):
        if not math.isfinite(v):
            _fail("STATS_NON_FINITE",
                  f"{where}: statistic {name} is non-finite ({v!r})")
    return {"delta_dc_abs": dc_abs,
            "rms_rel": rms / ref_rms, "max_rel": dc_abs / ref_max}


def _compose_overwrite(aenv, y_meas, k_probe, m, support, where: str,
                       tag: str):
    """Compose the exact-DC k-space write k_out = y_meas + (1-M).k_probe
    WITH the frozen STAGE-1 structural verification (spec r3 SS10,
    pre-inverse-FFT, bitwise): the acquired entries of the composition
    must be EXACTLY the DC write y_meas, and the unmeasured entries
    must be BITWISE the probe k_probe. Construction-by-formula is NOT
    a test of the contract: a non-binary mask value, DC content leaking
    into the unmeasured region, a broadcasting change, or a dtype
    demotion must FIRE here (ERROR), never pass silently."""
    torch = aenv.torch
    k_out = y_meas + (1.0 - m) * k_probe
    if not bool(torch.equal(k_out[support],
                            y_meas.to(k_out.dtype)[support])):
        _fail("R2_OVERWRITE_CONTRACT_VIOLATION",
              f"{where}/{tag}: the acquired entries of the composed "
              f"k-space are not exactly the DC write; the overwrite "
              f"contract is breached BEFORE the inverse FFT")
    if not bool(torch.equal(k_out[~support], k_probe[~support])):
        _fail("R2_OVERWRITE_CONTRACT_VIOLATION",
              f"{where}/{tag}: the unmeasured entries of the composed "
              f"k-space are not bitwise the probe; the overwrite "
              f"contract is breached BEFORE the inverse FFT")
    return k_out


def _check_roundtrip_regions(aenv, k_back, y_meas, k_probe, support,
                             where: str, tag: str) -> dict:
    """The frozen STAGE-2 structural verification (spec r3 SS10, post
    round-trip, tolerance-based, BOTH regions): after the inverse/
    forward FFT pair, the acquired region must return within floor_rel
    of the DC write y_meas and the unmeasured region within floor_rel
    of the probe k_probe -- measured in the SAME relative metrics as
    the floor gate. Any breach is an ERROR (the round-trip cannot be
    trusted at the pre-registered tolerance), never a loosening.
    Returns the per-region statistics as evidence."""
    yw = y_meas.to(k_back.dtype)
    acq = _three_stats(aenv, k_back - yw, yw, support,
                       f"{where}/{tag} roundtrip acquired")
    unm = _three_stats(aenv, k_back - k_probe, k_probe, ~support,
                       f"{where}/{tag} roundtrip unmeasured")
    for region, st in (("acquired", acq), ("unmeasured", unm)):
        for key in ("rms_rel", "max_rel"):
            v = float(st[key])
            if not math.isfinite(v) or v > FLOOR_REL:
                _fail("R2_ROUNDTRIP_CONTRACT_VIOLATION",
                      f"{where}/{tag}: the post-round-trip {region} "
                      f"region {key}={v!r} exceeds the pre-registered "
                      f"engineering tolerance floor_rel={FLOOR_REL!r}; "
                      f"the overwrite round-trip cannot be trusted and "
                      f"the tolerance is never loosened")
    return {"acquired": acq, "unmeasured": unm}


def _measure_readout(aenv, image, y_raw, amax, cmap, where: str) -> dict:
    """pre/floor/post DC residual measurement for ONE decoded complex
    image (spec r3 SS6/SS7), with the frozen TWO-STAGE structural
    verification of the overwrite seam (spec r3 SS10):
      pre   -- probe of the decoded image: fft2c(image) vs y/amax on the
               measured region (what an explicit R2 overwrite would
               actually correct)
      floor -- round-trip floor on the exactly-DC reconstruction with
               the SAME unmeasured content (the probe-chain noise
               floor), constructed through the VERIFIED seam
      post  -- the R2 overwrite seam k_post = M.(y/amax) +
               (1-M).fft2c(image), constructed AND verified: STAGE 1
               (pre-inverse-FFT, bitwise) proves the acquired entries
               are exactly the DC write and the unmeasured entries are
               bitwise the probe; STAGE 2 (post round-trip, tolerance)
               proves BOTH regions return within floor_rel. post and
               floor share the exact-DC construction BY DESIGN and
               their agreement is itself evidence that the overwrite
               reaches the floor"""
    dec, fft2c, ifft2c = aenv.dec, aenv.fft2c, aenv.ifft2c
    m = dec.column_mask_tensor(cmap, image.device, image.dtype)  # (W,)
    support = m.to(aenv.torch.bool).broadcast_to(image.shape)
    y_meas = (y_raw / amax) * m
    k_probe = fft2c(image)
    pre = _three_stats(aenv, (k_probe - y_meas) * m, y_meas, support,
                       f"{where} pre")
    k_exact = _compose_overwrite(aenv, y_meas, k_probe, m, support,
                                 where, "floor")
    floor = _three_stats(aenv, (fft2c(ifft2c(k_exact)) - k_exact) * m,
                         y_meas, support, f"{where} floor")
    k_post = _compose_overwrite(aenv, y_meas, k_probe, m, support,
                                where, "r2_overwrite")
    k_back = fft2c(ifft2c(k_post))
    post = _check_roundtrip_regions(aenv, k_back, y_meas, k_probe,
                                    support, where, "r2_overwrite")
    return {"pre": pre, "floor": floor, "post": post}


def _check_floor(floor_stats: dict, where: str) -> None:
    """The pre-registered floor gate (spec r3 SS8): BOTH relative floor
    metrics must be <= FLOOR_REL. A floor failure is an ERROR and the
    floor is NEVER loosened."""
    for key in ("rms_rel", "max_rel"):
        v = float(floor_stats[key])
        if not math.isfinite(v) or v > FLOOR_REL:
            _fail("FLOOR_EXCEEDS_TOLERANCE",
                  f"{where}: floor {key}={v!r} exceeds the pre-registered "
                  f"engineering tolerance floor_rel={FLOOR_REL!r}")


def _slice_decision(pre: dict, floor: dict) -> object:
    """The exact spec r3 SS8 rule on the z=0 decode statistics:
    True  <=> pre_rms_rel > K x floor_rms_rel AND
               pre_max_rel > K x floor_max_rel
    False <=> neither statistic exceeds; None (AMBIGUOUS) <=> exactly one
    exceeds (RMS/guard disagreement is structural, never coerced).
    The registered ZERO-FLOOR SUBSTITUTION (spec r3 SS8) applies PER
    METRIC: an EXACTLY-ZERO measured floor is replaced by the
    pre-registered positive floor_rel before the K-comparison -- a zero
    comparison threshold would make the rule trivially nontrivial; a
    nonzero measured floor, however small, is used as measured."""
    fl_rms = float(floor["rms_rel"])
    fl_max = float(floor["max_rel"])
    floor_rms = FLOOR_REL if fl_rms == 0.0 else fl_rms
    floor_max = FLOOR_REL if fl_max == 0.0 else fl_max
    hi_rms = float(pre["rms_rel"]) > K_SEPARATION * floor_rms
    hi_max = float(pre["max_rel"]) > K_SEPARATION * floor_max
    if hi_rms and hi_max:
        return True
    if (not hi_rms) and (not hi_max):
        return False
    return None


def _route_outcome(decisions: list, where: str) -> str:
    """Slice/endpoint routing (spec r3 SS8): all False -> Outcome A; all
    True -> Outcome B; ANY mixed or AMBIGUOUS element -> the structural
    ROUTE_AMBIGUOUS state. Routing does NOT raise on ambiguity: the
    complete evidence artifact (both statistics per slice) is published
    FIRST and the stage exits 2 only after successful publication
    (_publish_and_enforce); the lock decision returns to the owner and
    no silent tie-break is permitted. An empty decision list is a true
    implementation failure and remains an immediate no-publish ERROR."""
    if not decisions:
        _fail("ROUTE_INPUT_INVALID",
              f"{where}: an empty decision list cannot be routed; the "
              f"frozen fixture set has {N_FIXTURE_SLICES} slices")
    if all(d is False for d in decisions):
        return OUTCOME_A
    if all(d is True for d in decisions):
        return OUTCOME_B
    return ROUTE_AMBIGUOUS


def _ambiguity_record(endpoint_readout: dict) -> dict:
    """The AMBIGUOUS structural record for the facts artifact (spec r3
    SS8): BOTH statistics (the RMS primary and the max guard) for every
    slice whose decision is AMBIGUOUS, plus the full decision tally --
    the owner takes the lock decision from THIS record, with no silent
    tie-break."""
    per = endpoint_readout["per_slice"]
    return {"n_true": sum(s["decision"] is True for s in per),
            "n_false": sum(s["decision"] is False for s in per),
            "n_ambiguous": sum(s["decision"] is None for s in per),
            "disagreeing_slices": [
                {"identity": s["identity"],
                 "pre_rms_rel": s["stats"]["pre"]["rms_rel"],
                 "pre_max_rel": s["stats"]["pre"]["max_rel"],
                 "floor_rms_rel": s["stats"]["floor"]["rms_rel"],
                 "floor_max_rel": s["stats"]["floor"]["max_rel"]}
                for s in per if s["decision"] is None],
            "note": "R2_NONTRIVIAL = AMBIGUOUS: RMS and max-guard "
                    "disagree or per-slice decisions are mixed; both "
                    "statistics are published; the lock decision "
                    "returns to the owner; no silent tie-break "
                    "(spec r3 SS8)"}


def _endpoint_interpretation(route0: str, route1: str) -> dict:
    """Step-0 vs final interpretive record (spec r3 SS9): the FINAL
    checkpoint alone decides R2 retention; step 0 is interpretive
    context only. A divergence is RECORDED, never an ERROR and never a
    reversal of the final-endpoint outcome."""
    return {"decision_endpoint": "final",
            "decision_route": route1,
            "step0_route": route0,
            "endpoint_divergence": bool(route0 != route1),
            "note": "the final checkpoint alone decides R2 retention "
                    "(spec r3 SS8/SS9); step 0 is interpretive context "
                    "and can neither block nor reverse the outcome"}


def _paired_bootstrap(r0: np.ndarray, rk: np.ndarray,
                      idx: np.ndarray) -> dict:
    """The V02AMEND paired volume bootstrap machinery (pre-validated
    here on synthetic arrays; the scientific consumption is
    V02G/V02AMEND scope, NOT this stage):
      D1 -- per-resample mean of the PAIRED difference (rk - r0) over
            resampled volumes; CI95 of the resampled means
      D2 -- per-resample ratio of resampled arithmetic means
            mean(rk[idx]) / mean(r0[idx]); CI95
    ONE index matrix is applied IDENTICALLY to both sides (pairing).
    A zero resampled r0 mean makes D2 undefined: ERROR, never coerced
    (the V02E denominator guard, mirrored)."""
    r0 = np.asarray(r0, dtype=np.float64)
    rk = np.asarray(rk, dtype=np.float64)
    if r0.shape != (BOOTSTRAP_N_VOLUMES,) or rk.shape != r0.shape:
        _fail("BOOTSTRAP_INPUT_INVALID",
              f"paired arrays must both be ({BOOTSTRAP_N_VOLUMES},), "
              f"got {r0.shape} and {rk.shape}")
    if not np.isfinite(r0).all() or not np.isfinite(rk).all():
        _fail("BOOTSTRAP_INPUT_INVALID",
              "paired arrays contain non-finite values")
    if idx.shape != (BOOTSTRAP_B, BOOTSTRAP_N_VOLUMES):
        _fail("BOOTSTRAP_INPUT_INVALID",
              f"index matrix must be ({BOOTSTRAP_B}, "
              f"{BOOTSTRAP_N_VOLUMES}), got {idx.shape}")
    diff_b = (rk - r0)[idx].mean(axis=1)             # D1, paired
    r0_b = r0[idx].mean(axis=1)
    if np.any(r0_b == 0.0):
        _fail("BOOTSTRAP_DENOMINATOR_ZERO",
              "a resampled R0 mean is exactly 0.0; the D2 ratio is "
              "undefined for that resample and no coercion is "
              "permitted")
    ratio_b = rk[idx].mean(axis=1) / r0_b            # D2, paired idx
    return {"generator": "PCG64", "seed": BOOTSTRAP_SEED,
            "B": int(BOOTSTRAP_B), "unit": "volume", "paired": True,
            "d1_mean_paired_difference": {
                "mean": float(diff_b.mean()),
                "ci95": [float(v) for v in
                         np.percentile(diff_b, [2.5, 97.5])]},
            "d2_ratio_of_resampled_means": {
                "mean": float(ratio_b.mean()),
                "ci95": [float(v) for v in
                         np.percentile(ratio_b, [2.5, 97.5])]}}


def _d1d2_acceptance(d1: dict, d2: dict) -> dict:
    """The registered V02AMEND acceptance rule (owner-accepted
    2026-09-02), TWO conjuncts per statistic -- the practical-effect
    threshold on the OBSERVED MEAN and the uncertainty threshold on the
    CI bound, never one substituted for the other:
      D1 pass <=> mean paired dPSNR >= +0.50 dB AND CI95 lower > 0
      D2 pass <=> mean ratio <= 0.95            AND CI95 upper < 1.0
    (mean thresholds inclusive; CI null exclusions strict). V02H takes
    NO D1/D2 decision -- this function exists so the fixture exercises
    the EXACT conjuncts the amendment will apply, against machinery
    outputs and against hand-built boundary inputs."""
    d1_mean, d1_lo = float(d1["mean"]), float(d1["ci95"][0])
    d2_mean, d2_hi = float(d2["mean"]), float(d2["ci95"][1])
    d1_practical = d1_mean >= D1_PAIRED_PSNR_MIN_DB
    d1_uncertain = d1_lo > D1_CI_NULL
    d2_practical = d2_mean <= D2_RATIO_MAX
    d2_uncertain = d2_hi < D2_CI_NULL
    return {"d1": {"practical_effect_ok": d1_practical,
                   "uncertainty_ok": d1_uncertain,
                   "pass": d1_practical and d1_uncertain},
            "d2": {"practical_effect_ok": d2_practical,
                   "uncertainty_ok": d2_uncertain,
                   "pass": d2_practical and d2_uncertain}}


# ---------------------------------------------------------------------------
# PM/C0 reproduction gates (spec r3 SS12; anchors quoted above). PM and
# likelihood machinery ONLY -- not a z=0 R0 metric reproduction.
# ---------------------------------------------------------------------------

def _check_reproduction(pm0_psnr: float, pm0_nmse: float,
                        pmf_psnr: float, pmf_nmse: float,
                        c0_nll_final: float) -> dict:
    """PM absolutes toleranced (PSNR 1e-3 dB; NMSE_u 1e-4 relative) and
    the C0 final batch NLL an EXACT tie (float ==). Any breach is an
    ERROR, never a re-pin."""
    out = {}
    for label, meas, anchor in (
            ("step0", pm0_psnr, PM_STEP0_PSNR),
            ("final", pmf_psnr, PM_FINAL_PSNR)):
        delta = float(meas) - anchor
        out[f"pm_psnr_{label}"] = {"measured": float(meas),
                                   "anchor": anchor, "delta_db": delta,
                                   "tol_db": PSNR_REPRO_TOL_DB}
        if not math.isfinite(delta) or abs(delta) > PSNR_REPRO_TOL_DB:
            _fail("REPRO_PSNR_MISMATCH",
                  f"PM {label} psnr {float(meas)!r} deviates from the "
                  f"registered anchor {anchor!r} by {delta!r} dB "
                  f"(tol {PSNR_REPRO_TOL_DB!r})")
    for label, meas, anchor in (
            ("step0", pm0_nmse, PM_STEP0_NMSE),
            ("final", pmf_nmse, PM_FINAL_NMSE)):
        rel = (float(meas) - anchor) / anchor
        out[f"pm_nmse_u_{label}"] = {"measured": float(meas),
                                     "anchor": anchor, "delta_rel": rel,
                                     "tol_rel": NMSE_REPRO_TOL_REL}
        if not math.isfinite(rel) or abs(rel) > NMSE_REPRO_TOL_REL:
            _fail("REPRO_NMSE_MISMATCH",
                  f"PM {label} nmse_u {float(meas)!r} deviates from the "
                  f"registered anchor {anchor!r} by {rel!r} relative "
                  f"(tol {NMSE_REPRO_TOL_REL!r})")
    out["c0_nll_batch_final"] = {"measured": float(c0_nll_final),
                                 "anchor": C0_NLL_BATCH_FINAL,
                                 "tie": "exact (float ==)"}
    if float(c0_nll_final) != C0_NLL_BATCH_FINAL:
        _fail("REPRO_C0_NLL_MISMATCH",
              f"C0 final batch NLL {float(c0_nll_final)!r} != the "
              f"registered exact tie {C0_NLL_BATCH_FINAL!r}; the tie "
              f"convention is exact and never loosened")
    return out


def _verify_pm_bank(bank_rec: dict) -> dict:
    """The PM reproduction bank must regenerate to the registered SHAs
    (the PCG64(0) Z_DIAG bank, reused verbatim -- no new stream)."""
    if bank_rec["bank_sha256"] != PM_BANK_SHA256 \
            or bank_rec["manifest_sha256"] != PM_BANK_MANIFEST_SHA256:
        _fail("PM_BANK_MISMATCH",
              f"regenerated Z_DIAG bank sha "
              f"{bank_rec['bank_sha256'][:12]}.../manifest "
              f"{bank_rec['manifest_sha256'][:12]}... != the registered "
              f"constants {PM_BANK_SHA256[:12]}.../"
              f"{PM_BANK_MANIFEST_SHA256[:12]}...")
    return {"rule": "np.random.Generator(np.random.PCG64(0))."
                    "standard_normal(size=(128, 13824), dtype=float64)"
                    ".astype(float32) -- the registered TDIAG bank, "
                    "reused verbatim",
            "bank_sha256": bank_rec["bank_sha256"],
            "manifest_sha256": bank_rec["manifest_sha256"]}


def _verify_d3_subset(d3man: dict, holdman: dict) -> None:
    """Re-derive the frozen D3 draw and re-bind it to the verified
    holdout manifest (mirrored from SEQREF-V02E v0.2 run(); draw drift
    is an ERROR, never a redraw), plus the registered manifest-SHA
    cross-check against the v02_facts pin."""
    if int(d3man.get("n", -1)) != N_FIXTURE_SLICES \
            or len(d3man.get("entries", [])) != N_FIXTURE_SLICES:
        _fail("MANIFEST_CONTENT_MISMATCH",
              f"D3 monitor manifest declares n={d3man.get('n')} / "
              f"{len(d3man.get('entries', []))} entries; locked "
              f"{N_FIXTURE_SLICES}")
    positions = [int(p) for p in v02m.d3_monitor_positions()]
    drawn = [int(e["holdout_position"]) for e in d3man["entries"]]
    if drawn != positions or [int(e["draw_rank"])
                              for e in d3man["entries"]] != list(
                                  range(N_FIXTURE_SLICES)):
        _fail("D3_DRAW_MISMATCH",
              "the D3 monitor manifest does not reproduce the frozen "
              "PCG64(2).choice(199, 32, replace=False) draw in order")
    for e in d3man["entries"]:
        h = holdman["entries"][int(e["holdout_position"])]
        if e["file"] != h["file"] \
                or int(e["slice_index"]) != int(h["slice_index"]):
            _fail("D3_SUBSET_NOT_IN_HOLDOUT",
                  f"D3 entry {e['file']!r}/{e['slice_index']} is not the "
                  f"holdout entry at position {e['holdout_position']}")
    if d3man.get("manifest_sha256") != D3_SUBSET_MANIFEST_SHA256:
        _fail("D3_SUBSET_MANIFEST_DRIFT",
              f"D3 monitor manifest_sha256 "
              f"{str(d3man.get('manifest_sha256'))[:12]}... != the "
              f"registered v02_facts pin "
              f"{D3_SUBSET_MANIFEST_SHA256[:12]}...")


# ---------------------------------------------------------------------------
# Authoritative driver: the production complex decode seam over the frozen
# D3 32-slice fixture set at one endpoint.
# ---------------------------------------------------------------------------

def _decode_or_fail(env, model, z, st, where: str):
    """tg._decode_z verbatim (the production complex seam); StageError
    remapped to the single error identity. There is NO magnitude-space
    shortcut: the complex image is measured."""
    try:
        return env.tg._decode_z(model, z, st)
    except env.StageError as exc:
        _fail("DECODE_FAILURE", f"{where}: {exc.error_code}: {exc.reason}")


def _measure_endpoint_readout(env, model, states: list,
                              endpoint_label: str) -> dict:
    """For every slice: decode the production R0 z=0 latent through the
    production seam (the R2 question is z=0-only), measure
    pre/floor/post, gate the floor, decide per slice."""
    torch = env.torch
    z0 = torch.zeros(1, FLOW_DIM)
    per_slice = []
    with torch.no_grad():
        for j, st in enumerate(states):
            where = f"{endpoint_label} slice {j} {st['identity']!r}"
            image, _u_hat = _decode_or_fail(env, model, z0, st, where)
            meas = _measure_readout(env, image, st["y"][0], st["amax"],
                                    st["cmap"], where)
            _check_floor(meas["floor"], where)
            decision = _slice_decision(meas["pre"], meas["floor"])
            per_slice.append({"identity": st["identity"],
                              "decision": decision, "stats": meas})
            logger.info("[%s] %s: pre_rms_rel=%.3e floor_rms_rel=%.3e "
                        "decision=%s", __abbr__, where,
                        meas["pre"]["rms_rel"], meas["floor"]["rms_rel"],
                        decision)
    route = _route_outcome([s["decision"] for s in per_slice],
                           endpoint_label)
    return {"endpoint": endpoint_label, "route": route,
            "per_slice": per_slice}


# ---------------------------------------------------------------------------
# Facts assembly + claim-guarded publication (rerun guard mirrored from
# SEQREF-V02E v0.2: a scientifically different rerun may NOT publish
# silently alongside an existing authoritative artefact).
# ---------------------------------------------------------------------------

def _guard_rerun(facts: dict, out_dir: str) -> None:
    auth = Path(out_dir) / f"{FACTS_PREFIX}.json"
    if not auth.exists():
        return
    sidecar = Path(str(auth) + ".sha256")
    if not sidecar.exists() or _file_sha256(auth) != \
            sidecar.read_text().split()[0].strip():
        _fail("RERUN_PRIOR_ARTEFACT_MISMATCH",
              "an authoritative v02_readout_selftest.json exists but "
              "fails its own sidecar verification; no sibling is "
              "published next to an unverifiable artefact")
    prior = json.loads(auth.read_text())
    if prior.get("semantic_sha256") != facts["semantic_sha256"]:
        _fail("SEMANTIC_RERUN_MISMATCH",
              f"the existing authoritative {FACTS_PREFIX}.json has "
              f"semantic {str(prior.get('semantic_sha256'))[:12]}... != "
              f"this run's {facts['semantic_sha256'][:12]}...; a "
              f"scientifically different rerun may NOT publish silently "
              f"alongside")


def _publish_facts(env, facts: dict, out_dir: str) -> tuple:
    semantic = {k: v for k, v in facts.items() if k != "run"}
    env.attach_semantic_hash(facts, semantic)
    _guard_rerun(facts, out_dir)
    try:
        return env.publish_stage(facts, out_dir, FACTS_PREFIX, STAGE)
    except env.StageError as exc:
        _fail("PUBLICATION_FAILURE", f"{exc.error_code}: {exc.reason}")


def _publish_and_enforce(env, facts: dict, out_dir: str) -> tuple:
    """Publish the COMPLETE facts artifact FIRST; only then enforce the
    structural outcome (spec r3 SS8 ambiguity handling: exit 2 WITH
    BOTH STATISTICS PUBLISHED; the lock decision returns to the owner;
    no silent tie-break). True implementation/provenance failures are
    untouched: they remain immediate no-publish ERRORs raised before
    this point. Only the structural AMBIGUOUS outcome publishes before
    exiting 2."""
    path, sha = _publish_facts(env, facts, out_dir)
    if facts["outcome"]["label"] == ROUTE_AMBIGUOUS:
        _fail("R2_OUTCOME_AMBIGUOUS",
              f"the final-endpoint per-slice R2_NONTRIVIAL decisions "
              f"are mixed or AMBIGUOUS (RMS/max-guard disagreement); "
              f"the complete evidence artifact with both statistics "
              f"per slice is published at {path} (sha256 {sha}); the "
              f"lock decision returns to the owner; no silent "
              f"tie-break")
    return path, sha


def run(cfg: dict) -> dict:
    """The authoritative run. cfg keys: data_root, manifest_dir,
    train_record, run_root, p0s_facts, p4_stats2, implb_facts, out_dir.
    Every scientific constant is locked in this module and its frozen
    inputs, never in cfg."""
    env = _env()
    det_state = _pin_deterministic_backend(env.torch)
    t0 = time.time()
    man_dir = Path(cfg["manifest_dir"])

    manifests = [v02e._load_manifest(man_dir, f"v02_epoch{ep}_manifest.json",
                                     "train_epoch")
                 for ep in range(v02m.N_EPOCHS)]
    holdman = v02e._load_manifest(man_dir, "v02_holdout_manifest.json",
                                  "holdout")
    d3man = v02e._load_manifest(man_dir, "v02_d3_monitor_manifest.json",
                                "d3_monitor")
    _verify_d3_subset(d3man, holdman)

    record = v02e._load_train_record(cfg["train_record"], manifests)
    try:
        p4 = env.ffr.load_p4s2_parent(cfg["p4_stats2"])
        implb = env.ffr.load_implb_parent(cfg["implb_facts"])
        env.ffr.require_spline_b(implb["spline_b"])
        s_ref = env.tg._s_ref_from_p0s(cfg["p0s_facts"])
    except (env.ffr.FreeFlowError, env.StageError) as exc:
        _fail("PARENT_VERIFICATION_FAILED", f"{type(exc).__name__}: {exc}")
    logger.info("[%s] parents pinned: P4/2 %s | IMPL-B %s | s_ref=%.6g",
                __abbr__, p4["file_sha256"][:12],
                implb["file_sha256"][:12], s_ref)

    try:
        ds_val = env.FastMRISliceDataset(cfg["data_root"], split="val",
                                         mode="eval")
    except Exception as exc:  # the dataset ctor is a typed, logging
        _fail("DATA_PREMISE_FAILURE", f"{type(exc).__name__}: {exc}")

    ckpt_recs = {int(c["step"]): c for c in record["checkpoints"]}
    models = {step: v02e._load_checkpoint(env, Path(cfg["run_root"]),
                                          ckpt_recs[step],
                                          implb["spline_b"])
              for step in (STEP0, FINAL_STEP)}

    d3_states = v02e._collect_states(env, ds_val, d3man["entries"], p4,
                                     s_ref, N_FIXTURE_SLICES)

    # PCG64(4), registered paired-bootstrap role: the volume-level
    # paired index matrix is drawn and pin-verified BEFORE any
    # measurement (machinery pre-validation; this stage consumes the
    # stream for NO scientific bootstrap).
    boot_idx = _boot_index_matrix()
    streams = _verify_stream_pins(boot_idx)

    # R2 readout measurement at BOTH locked endpoints (z=0 only).
    readout = {
        "step0": _measure_endpoint_readout(env, models[STEP0], d3_states,
                                           "step0"),
        "final": _measure_endpoint_readout(env, models[FINAL_STEP],
                                           d3_states, "final")}
    # The FINAL checkpoint alone decides R2 retention; step 0 is
    # interpretive context (spec r3 SS9).
    interpretation = _endpoint_interpretation(readout["step0"]["route"],
                                              readout["final"]["route"])
    outcome = interpretation["decision_route"]

    # PM/C0 reproduction (registered PCG64(0) bank, SHA-pinned).
    bank_rec = env.estimators.z_diag_bank()
    bank_block = _verify_pm_bank(bank_rec)
    p = env.d3.derangement(N_FIXTURE_SLICES)
    pm0 = v02e._measure_condition(env, models[STEP0], d3_states, p, 0,
                                  bank_rec["bank"], "V02H-step0-C0")
    c0f = v02e._measure_condition(env, models[FINAL_STEP], d3_states, p, 0,
                                  bank_rec["bank"], "V02H-final-C0")
    repro = _check_reproduction(
        float(np.mean([s["pm_psnr"] for s in pm0["per_slice"]])),
        float(np.mean([s["pm_nmse_u"] for s in pm0["per_slice"]])),
        float(np.mean([s["pm_psnr"] for s in c0f["per_slice"]])),
        float(np.mean([s["pm_nmse_u"] for s in c0f["per_slice"]])),
        float(c0f["nll_batch"]))

    facts = {
        "schema": FACTS_SCHEMA,
        "script": f"{__abbr__} v{__version__}",
        "spec": "SEQREF-V02H v0.1 spec r3 (2026-09-02, LudWiki id 146 "
                "DRAFT/PRE-EXECUTION)",
        "mode": "authoritative",
        "config": {
            "k_separation": K_SEPARATION,
            "floor_rel": FLOOR_REL,
            "floor_wording": "pre-registered INDEPENDENT engineering "
                             "tolerance, NOT DCPROBE-derived; a floor "
                             "failure is an ERROR and the floor is never "
                             "loosened",
            "psnr_repro_tol_db": PSNR_REPRO_TOL_DB,
            "nmse_repro_tol_rel": NMSE_REPRO_TOL_REL,
            "endpoints": [STEP0, FINAL_STEP],
            "decision_endpoint": "final (step 0 is interpretive "
                                 "context only, spec r3 SS9)",
            "n_fixture_slices": N_FIXTURE_SLICES,
            "decode_points": ["z0 (the production R0 latent; the R2 "
                              "decision is z=0-only)"],
            "rule": "R2_NONTRIVIAL <=> pre_rms_rel > K x floor_rms_rel "
                    "AND pre_max_rel > K x floor_max_rel; exactly one "
                    "statistic above -> AMBIGUOUS (structural, exit 2)",
            "delta_dc_definition": "maximum coefficient-wise absolute "
                                   "residual over the acquired support "
                                   "(headline, absolute scale)",
            "support_definition": "the acquired column mask (never "
                                  "|y| > 0: an exactly-zero acquired "
                                  "coefficient is still acquired)"},
        "fixture_set": {
            "source": "v02_d3_monitor_manifest.json (frozen PCG64(2) "
                      "draw, re-derived and re-bound to the verified "
                      "holdout manifest)",
            "manifest_sha256": d3man["manifest_sha256"],
            "registered_pin": D3_SUBSET_MANIFEST_SHA256,
            "identities": [st["identity"] for st in d3_states]},
        "streams": {"v02h_stream": streams, "pm_bank": bank_block},
        "readout": readout,
        "outcome": {
            "label": outcome,
            "interpretation": interpretation,
            "routing_rule": "per endpoint: all slices False -> "
                            "OUTCOME_A_R2_NO_OP; all True -> "
                            "OUTCOME_B_R2_NONTRIVIAL; any mixed or "
                            "ambiguous -> the structural "
                            "R2_OUTCOME_AMBIGUOUS state: the COMPLETE "
                            "evidence artifact (both statistics per "
                            "slice) is published FIRST, then the stage "
                            "exits 2; the lock decision returns to the "
                            "owner; no silent tie-break (spec r3 SS8). "
                            "The FINAL endpoint alone decides R2 "
                            "retention",
            "note": "evidence only; the R2 keep/remove decision is an "
                    "input to the V02AMEND v1.0 assembly, never taken "
                    "here"},
        "pm_c0_reproduction": {
            "scope": "PM and likelihood machinery reproduction ONLY; "
                     "no registered D3-subset z=0 absolute anchors "
                     "exist, so this stage does NOT verify z=0 R0 "
                     "metric reproduction -- true R0 metric "
                     "reproduction on the full holdout is V02G scope, "
                     "and no new anchor is manufactured here",
            "checks": repro},
        "deterministic_execution": det_state,
        "validity": {
            "taxonomy": "exit 0 = facts artefact complete; exit 2 = "
                        "ERROR. The structural AMBIGUOUS outcome is "
                        "the ONE exit-2 path that publishes FIRST: "
                        "the complete evidence artifact (both "
                        "statistics per slice) materialises before "
                        "the stage exits 2 (spec r3 SS8); true "
                        "implementation/provenance failures remain "
                        "immediate no-publish ERRORs; no scientific "
                        "BLOCK, no exit 1, no skip",
            "manifest_sidecars_verified": [
                f"v02_epoch{ep}_manifest.json"
                for ep in range(v02m.N_EPOCHS)]
                + ["v02_holdout_manifest.json",
                   "v02_d3_monitor_manifest.json"],
            "train_record_sha256_verified": True,
            "checkpoint_state_sha256_verified": {
                str(step): ckpt_recs[step]["state_sha256"]
                for step in (STEP0, FINAL_STEP)},
            "d3_draw_rederived_match": True,
            "non_finite_events": 0},
        "run": {"utc": env.utc_stamp(), "argv": sys.argv,
                "wall_clock_s": time.time() - t0},
    }
    if outcome == ROUTE_AMBIGUOUS:
        facts["outcome"]["ambiguity"] = _ambiguity_record(
            readout["final"])
    v02e._no_verdict_scan(facts, "facts")
    path, sha = _publish_and_enforce(env, facts, str(Path(cfg["out_dir"])))
    logger.info("[%s] facts published %s sha256=%s; outcome %s",
                __abbr__, path, sha[:12], outcome)
    return facts


# ---------------------------------------------------------------------------
# Fixture suite (--mode fixtures). Every registered fixture is exercised;
# every exercised ERROR path must logger.error + V02Error carrying the
# expected code from the NAMED module's branch; a silent pass is an ERROR
# of the suite itself (V02PLAN SS7 row 10, carried). Observations are
# (module abbr, code) pairs: two modules may share a code string, and
# each branch must fire in its own module.
# ---------------------------------------------------------------------------
_OBSERVED: set = set()

# Harness's own failure surface, out of the coverage universe (V02S
# precedent): SELFTEST_FIXTURE_FAILED is the suite reporting itself.
_HARNESS_CODES = {"SELFTEST_FIXTURE_FAILED"}

# Written justifications for V02H codes the fixtures cannot reach; f10
# proves extracted == observed + deferred with no stale deferrals.
DEFERRED_JUSTIFICATIONS: dict = {
    "ENV_IMPORT_FAILED":
        "environment-bound: fires only where a registered dependency is "
        "genuinely absent; fabricating an ImportError would exercise "
        "the fixture, not the import boundary (V02S deferral precedent)",
    "DECODE_FAILURE":
        "model-bound: wraps tg._decode_z StageError conversion against "
        "the live flow model on the registered host; a fabricated model "
        "would exercise the fixture, not the seam",
    "PARENT_VERIFICATION_FAILED":
        "parent-load-bound: fires inside ffr.load_p4s2_parent / "
        "load_implb_parent / tg._s_ref_from_p0s against the pinned "
        "parent artefacts on the host",
    "DATA_PREMISE_FAILURE":
        "dataset-bound: the FastMRISliceDataset ctor against the live "
        "fastMRI traversal on the registered host",
    "PUBLICATION_FAILURE":
        "claim-guard internals: publish_stage's own StageError paths "
        "(claim collision, post-publish verify) fire only on a real "
        "claim conflict; the surrounding rerun-guard branches ARE "
        "exercised in f09"}


def _expect_error(fn, code: str, abbr: str) -> None:
    try:
        fn()
    except v02m.V02Error as exc:
        got = str(exc).split(":", 1)[0]
        if got != code:
            _fail("SELFTEST_FIXTURE_FAILED",
                  f"{abbr}: expected {code}, got {got} ({exc})")
        _OBSERVED.add((abbr, code))
        return
    _fail("SELFTEST_FIXTURE_FAILED",
          f"{abbr}: {code} did not fire; a silent pass in the selftest "
          f"is an ERROR (V02PLAN SS7 row 10)")


def _check(cond: bool, what: str) -> None:
    if not cond:
        _fail("SELFTEST_FIXTURE_FAILED", what)


# ---------------------------------------------------------------------------
# f01 -- frozen-input loader refusals (SEQREF-V02E loaders, verbatim).
# ---------------------------------------------------------------------------

def _write_json_pair(path: Path, doc: dict, *, sidecar_sha=None,
                     write_sidecar=True) -> None:
    path.write_text(json.dumps(doc))
    if write_sidecar:
        sha = sidecar_sha if sidecar_sha is not None else _file_sha256(path)
        Path(str(path) + ".sha256").write_text(sha + "\n")


def _valid_manifest() -> dict:
    body = {"schema": "seqref-v02-manifest/1", "kind": "holdout",
            "n_volumes": 1,
            "entries": [{"file": "f0", "slice_index": 0,
                         "dataset_index": 0, "holdout_position": 0}]}
    man = dict(body)
    man["manifest_sha256"] = v02m.manifest_sha256(body)
    return man


def _valid_train_record(man: dict) -> dict:
    return {"schema": "seqref-v02-train-record/1", "steps": FINAL_STEP,
            "checkpoints": [{"step": s, "file": f"ckpt_{s}.pt",
                             "file_sha256": "0" * 64,
                             "state_sha256": "1" * 64}
                            for s in v02e.CHECKPOINT_STEPS],
            "manifest_sha256": [man["manifest_sha256"]]}


def f01_loader_refusals() -> dict:
    abbr = "SEQREF-V02E"
    man = _valid_manifest()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "man.json"
        _write_json_pair(p, man, write_sidecar=False)
        _expect_error(lambda: v02e._load_manifest(Path(td), "man.json",
                                                  "holdout"),
                      "MANIFEST_HASH_MISMATCH", abbr)
        _write_json_pair(p, man, sidecar_sha="f" * 64)
        _expect_error(lambda: v02e._load_manifest(Path(td), "man.json",
                                                  "holdout"),
                      "MANIFEST_HASH_MISMATCH", abbr)
        bad_schema = dict(man, schema="seqref-v02-manifest/0")
        _write_json_pair(p, bad_schema)
        _expect_error(lambda: v02e._load_manifest(Path(td), "man.json",
                                                  "holdout"),
                      "MANIFEST_HASH_MISMATCH", abbr)
        _write_json_pair(p, man)
        _expect_error(lambda: v02e._load_manifest(Path(td), "man.json",
                                                  "train_epoch"),
                      "MANIFEST_SCHEMA_INVALID", abbr)
        drift = dict(man, manifest_sha256="e" * 64)
        _write_json_pair(p, drift)
        _expect_error(lambda: v02e._load_manifest(Path(td), "man.json",
                                                  "holdout"),
                      "MANIFEST_HASH_MISMATCH", abbr)
        rec = _valid_train_record(man)
        rp = Path(td) / "rec.json"
        bad_steps = dict(rec, steps=100)
        _write_json_pair(rp, bad_steps)
        _expect_error(lambda: v02e._load_train_record(str(rp), [man]),
                      "TRAIN_RECORD_MISMATCH", abbr)
        bad_sched = dict(rec, checkpoints=rec["checkpoints"][:-1])
        _write_json_pair(rp, bad_sched)
        _expect_error(lambda: v02e._load_train_record(str(rp), [man]),
                      "CHECKPOINT_SCHEDULE_MISMATCH", abbr)
        bad_chain = dict(rec, manifest_sha256=["d" * 64])
        _write_json_pair(rp, bad_chain)
        _expect_error(lambda: v02e._load_train_record(str(rp), [man]),
                      "TRAIN_RECORD_MISMATCH", abbr)
        _write_json_pair(rp, rec)
        got = v02e._load_train_record(str(rp), [man])
        _check(int(got["steps"]) == FINAL_STEP,
               "the valid train record failed to load")
    return {"branches": 8, "module": abbr}


# ---------------------------------------------------------------------------
# f02 -- the single PCG64(4) stream: determinism + golden pins.
# ---------------------------------------------------------------------------

def f02_stream_determinism() -> dict:
    boot_idx = _boot_index_matrix()
    streams = _verify_stream_pins(boot_idx)
    boot_idx2 = _boot_index_matrix()
    _check(np.array_equal(boot_idx, boot_idx2),
           "PCG64(4) bootstrap stream re-instantiation is not bitwise "
           "deterministic")
    tampered_idx = boot_idx.copy()
    tampered_idx[0, 0] += 1
    _expect_error(lambda: _verify_stream_pins(tampered_idx),
                  "STREAM_PIN_MISMATCH", __abbr__)
    content = _draw_synthetic_content()
    sha = _verify_content_pin(content)
    content2 = _draw_synthetic_content()
    _check(_content_sha256(content2) == sha,
           "test-only synthetic content stream re-instantiation is not "
           "bitwise deterministic")
    tampered_content = {"y": content["y"].copy(),
                        "amax": content["amax"].copy(),
                        "u": content["u"].copy()}
    tampered_content["u"][0, 0] += 1.0
    _expect_error(lambda: _verify_content_pin(tampered_content),
                  "STREAM_PIN_MISMATCH", __abbr__)
    return {"streams": streams, "synthetic_content_sha256": sha,
            "fixture_data_seed": {"seed": FIXTURE_DATA_SEED,
                                  "role": "test-only, fixtures mode "
                                          "only, outside the campaign "
                                          "RNG registry"}}


# ---------------------------------------------------------------------------
# f03 -- the measurement chain on synthetic slices through the REAL
# production decode assembly (dec.decode_normalised, the exact-DC
# readout), incl. the positive-injection proof that the SS8 rule fires
# TRUE on a genuinely inconsistent measured region.
# ---------------------------------------------------------------------------

def _fixture_cmap(aenv):
    return aenv.dec.build_coordinate_map(FIXTURE_MASK_COLS, 96, 96)


def _fixture_decode(aenv, y_np, amax, u_np, cmap):
    """The REAL production assembly on synthetic content: y complex64 /
    u complex128 / amax float32 -- the production dtype mix."""
    torch = aenv.torch
    y_t = torch.from_numpy(np.ascontiguousarray(y_np)).to(torch.complex64)
    u_t = torch.from_numpy(np.ascontiguousarray(u_np))
    amax_t = torch.tensor([float(amax)], dtype=torch.float32)
    return aenv.dec.decode_normalised(y_t, amax_t, u_t, cmap)


def f03_measurement_chain_synthetic() -> dict:
    aenv = _env_array()
    boot_idx = _boot_index_matrix()
    _verify_stream_pins(boot_idx)
    content = _draw_synthetic_content()
    _verify_content_pin(content)
    cmap = _fixture_cmap(aenv)
    torch = aenv.torch
    decisions = []
    max_floor_rms = 0.0
    max_pre_rms = 0.0
    for j in range(N_FIXTURE_SLICES):
        image = _fixture_decode(aenv, content["y"][j],
                                content["amax"][j],
                                content["u"][j], cmap)
        y_t = torch.from_numpy(
            np.ascontiguousarray(content["y"][j])).to(torch.complex64)
        meas = _measure_readout(aenv, image, y_t,
                                float(content["amax"][j]), cmap,
                                f"fixture slice {j}")
        _check_floor(meas["floor"], f"fixture slice {j}")
        max_floor_rms = max(max_floor_rms, meas["floor"]["rms_rel"])
        max_pre_rms = max(max_pre_rms, meas["pre"]["rms_rel"])
        decisions.append(_slice_decision(meas["pre"], meas["floor"]))
    route = _route_outcome(decisions, "fixtures-synthetic")
    _check(route == OUTCOME_A,
           f"the exactly-DC synthetic fixture must route {OUTCOME_A}, "
           f"got {route}")
    # Positive injection: amplify the measured region by 2e-3 -> the
    # SS8 rule MUST fire True (the fixture proves it can discriminate).
    image = _fixture_decode(aenv, content["y"][0], content["amax"][0],
                            content["u"][0], cmap)
    m = aenv.dec.column_mask_tensor(cmap, image.device, image.dtype)
    k_bad = aenv.fft2c(image) * (1.0 + 2e-3 * m)
    img_bad = aenv.ifft2c(k_bad)
    y_t = torch.from_numpy(np.ascontiguousarray(content["y"][0])).to(
        torch.complex64)
    meas_bad = _measure_readout(aenv, img_bad, y_t,
                                float(content["amax"][0]), cmap,
                                "injection")
    _check_floor(meas_bad["floor"], "injection")
    inj = _slice_decision(meas_bad["pre"], meas_bad["floor"])
    _check(inj is True,
           f"the 2e-3 measured-region injection must decide True, got "
           f"{inj} (pre_rms_rel={meas_bad['pre']['rms_rel']:.3e} vs "
           f"floor_rms_rel={meas_bad['floor']['rms_rel']:.3e})")
    return {"n_slices": N_FIXTURE_SLICES,
            "decode_points_per_slice": 1,
            "route": route, "max_floor_rms_rel": max_floor_rms,
            "max_pre_rms_rel": max_pre_rms,
            "injection_decision": True,
            "injection_pre_rms_rel": meas_bad["pre"]["rms_rel"]}


# ---------------------------------------------------------------------------
# f04 -- the V02AMEND paired volume bootstrap machinery, pre-validated on
# synthetic paired arrays with KNOWN ground truth: stream shape, pairing
# (identical indices both sides), D1/D2 estimator arithmetic, CI
# boundary/crossing behaviour, bitwise determinism, and every refusal.
# ---------------------------------------------------------------------------

FIXTURE_PAIRED_SEED = 20260911         # fixed TEST-ONLY seed for the
                                       # synthetic paired arrays;
                                       # outside the campaign RNG
                                       # registry


def f04_bootstrap_machinery() -> dict:
    idx = _boot_index_matrix()
    _verify_stream_pins(idx)
    _check(idx.shape == (BOOTSTRAP_B, BOOTSTRAP_N_VOLUMES),
           f"paired index matrix must be ({BOOTSTRAP_B}, "
           f"{BOOTSTRAP_N_VOLUMES}) over volumes")
    rng = np.random.Generator(np.random.PCG64(FIXTURE_PAIRED_SEED))
    psnr_r0 = 30.0 + 0.01 * rng.standard_normal(BOOTSTRAP_N_VOLUMES)
    psnr_rk = psnr_r0 + 0.6            # known exact +0.6 dB paired shift
    nmse_r0 = 1.0 + 0.01 * rng.standard_normal(BOOTSTRAP_N_VOLUMES)
    nmse_rk = 0.9 * nmse_r0            # known exact 0.9 ratio

    # Estimator arithmetic: exact recovery of the known ground truth.
    d1 = _paired_bootstrap(psnr_r0, psnr_rk, idx)
    _check(abs(d1["d1_mean_paired_difference"]["mean"] - 0.6) < 1e-12,
           "D1 must recover the exact paired +0.6 dB shift")
    d2 = _paired_bootstrap(nmse_r0, nmse_rk, idx)
    _check(abs(d2["d2_ratio_of_resampled_means"]["mean"] - 0.9) < 1e-12,
           "D2 must recover the exact 0.9 ratio of resampled means")
    null = _paired_bootstrap(psnr_r0, psnr_r0.copy(), idx)
    _check(null["d1_mean_paired_difference"]["ci95"] == [0.0, 0.0],
           "boundary: an exact-null pair must give a degenerate "
           "[0.0, 0.0] D1 CI")

    # The registered acceptance rule, exercised conjunct by conjunct
    # against MACHINERY outputs. The practical-effect threshold (on the
    # observed mean) and the uncertainty threshold (on the CI bound) are
    # exercised SEPARATELY -- including the discriminating cases where
    # the CI bound sits BETWEEN the null and the effect threshold, which
    # the v0.3 CI-vs-effect-threshold test classified wrongly. Every
    # wide-noise draw is demeaned so the paired mean stays EXACTLY at
    # its constructed value (the conjunct under test is isolated).
    def _demeaned():
        w = rng.standard_normal(BOOTSTRAP_N_VOLUMES)
        return w - w.mean()

    cases = {}
    # (a) clear pass: +0.6 dB / 0.9 ratio, tight CIs.
    cases["clear_pass"] = _d1d2_acceptance(
        d1["d1_mean_paired_difference"],
        d2["d2_ratio_of_resampled_means"])
    _check(cases["clear_pass"]["d1"]["pass"]
           and cases["clear_pass"]["d2"]["pass"],
           "the clear-pass case must pass both D1 and D2")
    # (b) D1 uncertainty failure alone: constructed paired shift
    # exactly +0.6 dB (practical PASS) but a wide CI crossing the null
    # (uncertainty FAIL).
    w = _demeaned()
    d1_wide = _paired_bootstrap(psnr_r0, psnr_r0 + 0.6 + 6.0 * w, idx)
    cases["d1_uncertainty_fail"] = _d1d2_acceptance(
        d1_wide["d1_mean_paired_difference"],
        d2["d2_ratio_of_resampled_means"])
    _check(cases["d1_uncertainty_fail"]["d1"]["practical_effect_ok"]
           and not cases["d1_uncertainty_fail"]["d1"]["uncertainty_ok"]
           and not cases["d1_uncertainty_fail"]["d1"]["pass"],
           "D1 mean +0.6 dB with a null-crossing CI must fail on the "
           "uncertainty conjunct alone")
    # (c) D1 practical-effect failure alone: mean +0.3 dB (practical
    # FAIL) with a tight CI entirely above the null (uncertainty PASS).
    w = _demeaned()
    d1_tight = _paired_bootstrap(psnr_r0, psnr_r0 + 0.3 + 0.02 * w, idx)
    cases["d1_practical_fail"] = _d1d2_acceptance(
        d1_tight["d1_mean_paired_difference"],
        d2["d2_ratio_of_resampled_means"])
    _check(not cases["d1_practical_fail"]["d1"]["practical_effect_ok"]
           and cases["d1_practical_fail"]["d1"]["uncertainty_ok"]
           and not cases["d1_practical_fail"]["d1"]["pass"],
           "D1 mean +0.3 dB with a tight above-null CI must fail on "
           "the practical-effect conjunct alone")
    # (d) D1 discriminating case: constructed paired shift exactly
    # +0.55 dB with the CI lower bound BETWEEN the null and +0.50 dB --
    # the registered rule PASSES (mean >= +0.50 AND CI lower > 0); the
    # v0.3 CI-lower-vs-+0.50 test would have REJECTED it.
    w = _demeaned()
    d1_disc = _paired_bootstrap(psnr_r0, psnr_r0 + 0.55 + 0.8 * w, idx)
    lo = d1_disc["d1_mean_paired_difference"]["ci95"][0]
    _check(0.0 < lo < D1_PAIRED_PSNR_MIN_DB,
           f"fixture construction: the D1 discriminating CI lower "
           f"{lo!r} must sit between the null and +0.50 dB")
    cases["d1_discriminating"] = _d1d2_acceptance(
        d1_disc["d1_mean_paired_difference"],
        d2["d2_ratio_of_resampled_means"])
    _check(cases["d1_discriminating"]["d1"]["pass"],
           "the registered D1 rule must PASS a mean >= +0.50 dB whose "
           "CI lower sits between the null and +0.50 dB")
    # (e) D2 uncertainty failure alone: constructed ratio 0.9
    # (practical PASS) but a wide CI crossing 1.0 (uncertainty FAIL).
    w = _demeaned()
    d2_wide = _paired_bootstrap(nmse_r0, nmse_r0 * (0.9 + 3.0 * w), idx)
    cases["d2_uncertainty_fail"] = _d1d2_acceptance(
        d1["d1_mean_paired_difference"],
        d2_wide["d2_ratio_of_resampled_means"])
    _check(cases["d2_uncertainty_fail"]["d2"]["practical_effect_ok"]
           and not cases["d2_uncertainty_fail"]["d2"]["uncertainty_ok"]
           and not cases["d2_uncertainty_fail"]["d2"]["pass"],
           "D2 mean ratio 0.9 with a null-crossing CI must fail on the "
           "uncertainty conjunct alone")
    # (f) D2 practical-effect failure alone: constructed ratio 0.97
    # (practical FAIL) with a tight CI entirely below 1.0 (uncertainty
    # PASS).
    w = _demeaned()
    d2_tight = _paired_bootstrap(nmse_r0, nmse_r0 * (0.97 + 0.02 * w),
                                 idx)
    cases["d2_practical_fail"] = _d1d2_acceptance(
        d1["d1_mean_paired_difference"],
        d2_tight["d2_ratio_of_resampled_means"])
    _check(not cases["d2_practical_fail"]["d2"]["practical_effect_ok"]
           and cases["d2_practical_fail"]["d2"]["uncertainty_ok"]
           and not cases["d2_practical_fail"]["d2"]["pass"],
           "D2 mean ratio 0.97 with a tight below-null CI must fail on "
           "the practical-effect conjunct alone")
    # (g) D2 discriminating case: mean ratio 0.94 with the CI upper
    # bound BETWEEN 0.95 and 1.0 -- the registered rule PASSES (mean
    # <= 0.95 AND CI upper < 1.0); the v0.3 CI-upper-vs-0.95 test would
    # have REJECTED it.
    w = _demeaned()
    d2_disc = _paired_bootstrap(nmse_r0, nmse_r0 * (0.94 + 0.15 * w),
                                idx)
    hi = d2_disc["d2_ratio_of_resampled_means"]["ci95"][1]
    _check(D2_RATIO_MAX < hi < D2_CI_NULL,
           f"fixture construction: the D2 discriminating CI upper "
           f"{hi!r} must sit between 0.95 and 1.0")
    cases["d2_discriminating"] = _d1d2_acceptance(
        d1["d1_mean_paired_difference"],
        d2_disc["d2_ratio_of_resampled_means"])
    _check(cases["d2_discriminating"]["d2"]["pass"],
           "the registered D2 rule must PASS a mean ratio <= 0.95 "
           "whose CI upper sits between 0.95 and 1.0")
    # (h) both conjuncts fail: the exact-null pair.
    cases["null_both_fail"] = _d1d2_acceptance(
        null["d1_mean_paired_difference"],
        null["d2_ratio_of_resampled_means"])
    _check(not cases["null_both_fail"]["d1"]["pass"]
           and not cases["null_both_fail"]["d1"]["practical_effect_ok"]
           and not cases["null_both_fail"]["d1"]["uncertainty_ok"],
           "the null pair must fail BOTH D1 conjuncts")

    # Boundary strictness on hand-built inputs (no bootstrap): the mean
    # thresholds are INCLUSIVE (>= +0.50 / <= 0.95), the CI null
    # exclusions STRICT (> 0 / < 1.0).
    bnd = _d1d2_acceptance({"mean": D1_PAIRED_PSNR_MIN_DB,
                            "ci95": [0.0, 0.6]},
                           {"mean": D2_RATIO_MAX, "ci95": [0.8, 1.0]})
    _check(bnd["d1"]["practical_effect_ok"]
           and not bnd["d1"]["uncertainty_ok"]
           and bnd["d2"]["practical_effect_ok"]
           and not bnd["d2"]["uncertainty_ok"],
           "boundary: mean exactly at the threshold must satisfy the "
           "practical conjunct; a CI bound exactly AT the null must "
           "fail the uncertainty conjunct")

    # Pairing proof: ONE index matrix is applied IDENTICALLY to both
    # sides; an independent resampling of the rk side changes the
    # per-resample differences bitwise.
    paired_diff = (psnr_rk - psnr_r0)[idx].mean(axis=1)
    unpaired_diff = (psnr_rk[idx] - psnr_r0[idx[:, ::-1]]).mean(axis=1)
    _check(not np.array_equal(paired_diff, unpaired_diff),
           "pairing broken: identical indices for R0/Rk are required")
    _check(_paired_bootstrap(psnr_r0, psnr_rk, idx) == d1,
           "the paired bootstrap is not bitwise deterministic on "
           "identical inputs")

    _expect_error(lambda: _paired_bootstrap(np.zeros(32), psnr_rk, idx),
                  "BOOTSTRAP_INPUT_INVALID", __abbr__)
    bad = nmse_r0.copy()
    bad[3] = np.nan
    _expect_error(lambda: _paired_bootstrap(bad, nmse_rk, idx),
                  "BOOTSTRAP_INPUT_INVALID", __abbr__)
    _expect_error(lambda: _paired_bootstrap(
        nmse_r0, nmse_rk, idx[:, :32]), "BOOTSTRAP_INPUT_INVALID",
        __abbr__)
    _expect_error(lambda: _paired_bootstrap(
        np.zeros(BOOTSTRAP_N_VOLUMES), nmse_rk, idx),
        "BOOTSTRAP_DENOMINATOR_ZERO", __abbr__)
    return {"ground_truth": {"d1_shift_db": 0.6, "d2_ratio": 0.9},
            "d1": d1["d1_mean_paired_difference"],
            "d2": d2["d2_ratio_of_resampled_means"],
            "registered_rule": {
                "d1": "pass <=> mean paired dPSNR >= +0.50 dB AND "
                      "CI95 lower > 0",
                "d2": "pass <=> mean ratio <= 0.95 AND CI95 upper "
                      "< 1.0"},
            "acceptance_cases": cases,
            "quoted_thresholds": {"d1_paired_psnr_min_db":
                                  D1_PAIRED_PSNR_MIN_DB,
                                  "d1_ci_null": D1_CI_NULL,
                                  "d2_ratio_max": D2_RATIO_MAX,
                                  "d2_ci_null": D2_CI_NULL},
            "branches": 4}


# ---------------------------------------------------------------------------
# f05 -- deterministic backend pinning (spec r3 deterministic-pinning
# section): flags set, state recorded, bitwise rerun determinism of the
# registered Fourier pair and a representative dense op chain.
# ---------------------------------------------------------------------------

def f05_deterministic_backend() -> dict:
    aenv = _env_array()
    state = _pin_deterministic_backend(aenv.torch)
    torch = aenv.torch
    _check(torch.get_num_threads() == 1, "num_threads not pinned to 1")
    _check(torch.are_deterministic_algorithms_enabled(),
           "deterministic algorithms not enabled")
    _check(not torch.backends.cuda.matmul.allow_tf32,
           "TF32 matmul not disabled")
    _check(not torch.backends.cudnn.allow_tf32,
           "TF32 cuDNN not disabled")
    _check(torch.backends.cudnn.deterministic,
           "cuDNN deterministic not set")
    _check(not torch.backends.cudnn.benchmark,
           "cuDNN benchmark not disabled")
    # CUBLAS_WORKSPACE_CONFIG explicit pin/refusal (no setdefault): a
    # conflicting pre-existing value REFUSES, a torch-accepted
    # pre-existing value is KEPT, an unset variable is ASSIGNED the pin.
    key = "CUBLAS_WORKSPACE_CONFIG"
    saved = os.environ.get(key)
    try:
        os.environ[key] = ":1024:8"
        _expect_error(lambda: _pin_deterministic_backend(torch),
                      "BACKEND_CONFIG_CONFLICT", __abbr__)
        os.environ[key] = ":16:8"
        st = _pin_deterministic_backend(torch)
        _check(st["cublas_workspace_config"] == ":16:8"
               and st["cublas_source"] == "preexisting_accepted",
               "a torch-accepted pre-existing CUBLAS configuration "
               "must be kept, not overridden")
        del os.environ[key]
        st = _pin_deterministic_backend(torch)
        _check(os.environ.get(key) == CUBLAS_CONFIG
               and st["cublas_source"] == "assigned",
               "an unset CUBLAS configuration must be assigned the "
               "pinned value")
    finally:
        if saved is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = saved
    state = _pin_deterministic_backend(torch)   # re-pin under the
                                                # restored environment
    g = torch.Generator().manual_seed(0)
    x = torch.randn(96, 96, generator=g,
                    dtype=torch.float32).to(torch.complex64)
    k1 = aenv.fft2c(aenv.ifft2c(x))
    k2 = aenv.fft2c(aenv.ifft2c(x))
    _check(torch.equal(k1, k2),
           "the registered Fourier pair shows rerun variance")
    a = torch.randn(256, 256, generator=g, dtype=torch.float64)
    b = torch.randn(256, 256, generator=g, dtype=torch.float64)
    _check(torch.equal(a @ b, a @ b),
           "dense matmul shows rerun variance")
    return {"state": state, "bitwise_rerun_identical": True}


# ---------------------------------------------------------------------------
# f06 -- the SS8 rule arithmetic, floor gate, statistic guards, routing,
# and the frozen SS10 TWO-STAGE structural verification of the overwrite
# seam (stage-1 pre-inverse-FFT bitwise + stage-2 post round-trip
# tolerance on BOTH regions), incl. the corrupted-DC negative fixtures.
# ---------------------------------------------------------------------------

def f06_rule_arithmetic() -> dict:
    aenv = _env_array()
    torch = aenv.torch
    fl = {"rms_rel": 1e-7, "max_rel": 1e-7}
    _check(_slice_decision({"rms_rel": 11e-7, "max_rel": 11e-7}, fl)
           is True, "both statistics above K x floor must decide True")
    _check(_slice_decision({"rms_rel": 9e-7, "max_rel": 9e-7}, fl)
           is False, "neither statistic above must decide False")
    _check(_slice_decision({"rms_rel": 1e-6, "max_rel": 1e-6}, fl)
           is False, "the K x floor boundary is STRICT (>); equality "
                     "must decide False")
    _check(_slice_decision({"rms_rel": 11e-7, "max_rel": 9e-7}, fl)
           is None, "RMS above / guard below must be AMBIGUOUS (None)")
    _check(_slice_decision({"rms_rel": 9e-7, "max_rel": 11e-7}, fl)
           is None, "guard above / RMS below must be AMBIGUOUS (None)")
    # The registered ZERO-FLOOR SUBSTITUTION (spec r3 SS8): an EXACTLY-
    # ZERO measured floor is replaced by floor_rel=1e-6 PER METRIC
    # before the K-comparison; a nonzero measured floor, however small,
    # is used as measured (proven by the cases above: with fl=1e-7 the
    # threshold is 1e-6, so 11e-7 decides True -- a substitution of the
    # small nonzero floor would have made it False).
    zfl = {"rms_rel": 0.0, "max_rel": 0.0}
    _check(_slice_decision({"rms_rel": 5e-6, "max_rel": 5e-6}, zfl)
           is False, "zero floor: 5e-6 is NOT above K x 1e-6; a raw "
                     "zero threshold would have decided True")
    _check(_slice_decision({"rms_rel": 1.1e-5, "max_rel": 1.1e-5}, zfl)
           is True, "zero floor: 1.1e-5 must exceed K x 1e-6")
    thr = K_SEPARATION * FLOOR_REL      # the exact float threshold;
                                        # NOTE 1e-5 != 10.0*1e-6 in
                                        # binary floating point
    _check(_slice_decision({"rms_rel": thr, "max_rel": thr}, zfl)
           is False, "zero floor: the K x floor_rel boundary is STRICT "
                     "(>); equality with the computed threshold must "
                     "decide False")
    _check(_slice_decision({"rms_rel": 5e-6, "max_rel": 11e-7},
                           {"rms_rel": 0.0, "max_rel": 1e-7})
           is None, "the substitution is PER METRIC: rms substituted "
                    "(5e-6 not above K x 1e-6) / max measured (11e-7 "
                    "above K x 1e-7) must be AMBIGUOUS (None)")
    _expect_error(lambda: _check_floor(
        {"rms_rel": 2e-6, "max_rel": 1e-7}, "t"),
        "FLOOR_EXCEEDS_TOLERANCE", __abbr__)
    _expect_error(lambda: _check_floor(
        {"rms_rel": 1e-7, "max_rel": 2e-6}, "t"),
        "FLOOR_EXCEEDS_TOLERANCE", __abbr__)
    zeros = torch.zeros(4, 4, dtype=torch.complex64)
    full = torch.ones(4, 4, dtype=torch.bool)
    empty = torch.zeros(4, 4, dtype=torch.bool)
    _expect_error(lambda: _three_stats(aenv, zeros, zeros, empty, "t"),
                  "STATS_DENOMINATOR_INVALID", __abbr__)
    _expect_error(lambda: _three_stats(aenv, zeros, zeros, full, "t"),
                  "STATS_DENOMINATOR_INVALID", __abbr__)
    ref = torch.ones(4, 4, dtype=torch.complex64)
    nans = torch.full((4, 4), float("nan"), dtype=torch.float32).to(
        torch.complex64)
    _expect_error(lambda: _three_stats(aenv, nans, ref, full, "t"),
                  "STATS_NON_FINITE", __abbr__)
    # Mask-based support: an acquired coefficient whose value is EXACTLY
    # ZERO is still selected (the v0.1 |ref|>0 selection dropped it).
    delta = torch.zeros(4, 4, dtype=torch.complex64)
    delta[0, 0] = 1e-3
    ref_zero = torch.ones(4, 4, dtype=torch.complex64)
    ref_zero[0, 0] = 0.0
    st = _three_stats(aenv, delta, ref_zero, full, "t")
    _check(abs(st["delta_dc_abs"] - 1e-3) < 1e-9,
           "the maximum coefficient-wise residual at an exactly-zero "
           "acquired coefficient must be selected via the mask")
    _check(_route_outcome([False] * N_FIXTURE_SLICES, "t") == OUTCOME_A,
           "all-False routing broken")
    _check(_route_outcome([True] * N_FIXTURE_SLICES, "t") == OUTCOME_B,
           "all-True routing broken")
    mixed = [False] * (N_FIXTURE_SLICES - 1) + [True]
    _check(_route_outcome(mixed, "t") == ROUTE_AMBIGUOUS,
           "mixed decisions must RETURN the structural AMBIGUOUS "
           "state (publish-then-exit-2 downstream), not raise")
    amb = [False] * (N_FIXTURE_SLICES - 1) + [None]
    _check(_route_outcome(amb, "t") == ROUTE_AMBIGUOUS,
           "an AMBIGUOUS slice decision must RETURN the structural "
           "AMBIGUOUS state, not raise")
    _expect_error(lambda: _route_outcome([], "t"),
                  "ROUTE_INPUT_INVALID", __abbr__)
    # Final-decides interpretive record: divergence is RECORDED, never
    # an ERROR, and never reverses the final-endpoint decision.
    interp = _endpoint_interpretation(OUTCOME_A, OUTCOME_B)
    _check(interp["decision_route"] == OUTCOME_B
           and interp["endpoint_divergence"] is True,
           "the final endpoint must decide even under divergence")
    interp2 = _endpoint_interpretation(OUTCOME_A, OUTCOME_A)
    _check(interp2["decision_route"] == OUTCOME_A
           and interp2["endpoint_divergence"] is False,
           "concordant endpoints must record no divergence")

    # SS10 two-stage structural verification of the overwrite seam.
    # Stage 1 (pre-inverse-FFT, bitwise) and stage 2 (post round-trip,
    # tolerance, BOTH regions) must FIRE on corrupted constructions --
    # construction-by-formula is not a test of the contract.
    g = torch.Generator().manual_seed(7)
    k_probe = torch.randn(96, 96, generator=g,
                          dtype=torch.float64).to(torch.complex128)
    y_full = torch.randn(96, 96, generator=g,
                         dtype=torch.float64).to(torch.complex128)
    m = torch.zeros(96, dtype=torch.float64)
    m[torch.tensor(FIXTURE_MASK_COLS)] = 1.0
    support = m.to(torch.bool).broadcast_to(k_probe.shape)
    y_meas = y_full * m
    # Clean construction: stage 1 passes and returns the exact write.
    k_ok = _compose_overwrite(aenv, y_meas, k_probe, m, support,
                              "t", "clean")
    _check(bool(torch.equal(k_ok[support], y_meas[support]))
           and bool(torch.equal(k_ok[~support], k_probe[~support])),
           "the clean overwrite construction must pass stage 1")
    # Corrupted-DC negative fixture: UNMASKED y (DC content leaks into
    # the unmeasured region) -> stage 1 must FIRE.
    _expect_error(lambda: _compose_overwrite(
        aenv, y_full, k_probe, m, support, "t", "dc-leak"),
        "R2_OVERWRITE_CONTRACT_VIOLATION", __abbr__)
    # Non-binary mask value at an UNMEASURED column -> stage 1 FIRES on
    # the unmeasured-region check.
    m_bad = m.clone()
    m_bad[10] = 0.5                     # column 10 is not acquired
    _expect_error(lambda: _compose_overwrite(
        aenv, y_full * m_bad, k_probe, m_bad, support, "t",
        "nonbinary-mask"), "R2_OVERWRITE_CONTRACT_VIOLATION", __abbr__)
    # Non-binary mask value at an ACQUIRED column -> stage 1 FIRES on
    # the acquired-region check (the write is not exactly the DC data).
    m_bad2 = m.clone()
    m_bad2[0] = 0.5                     # column 0 is acquired
    _expect_error(lambda: _compose_overwrite(
        aenv, y_full * m_bad2, k_probe, m_bad2, support, "t",
        "acquired-half"), "R2_OVERWRITE_CONTRACT_VIOLATION", __abbr__)
    # Clean round-trip: stage 2 passes on BOTH regions.
    k_back_ok = aenv.fft2c(aenv.ifft2c(k_ok))
    post_ok = _check_roundtrip_regions(aenv, k_back_ok, y_meas, k_probe,
                                       support, "t", "clean")
    _check(post_ok["acquired"]["rms_rel"] <= FLOOR_REL
           and post_ok["acquired"]["max_rel"] <= FLOOR_REL
           and post_ok["unmeasured"]["rms_rel"] <= FLOOR_REL
           and post_ok["unmeasured"]["max_rel"] <= FLOOR_REL,
           "the clean round-trip must pass stage 2 on both regions")
    # Corrupted round-trip: one tampered ACQUIRED coefficient -> stage 2
    # must FIRE.
    k_bad = k_back_ok.clone()
    k_bad[0, 0] += 1e-3                 # column 0 is acquired
    _expect_error(lambda: _check_roundtrip_regions(
        aenv, k_bad, y_meas, k_probe, support, "t", "acq-tamper"),
        "R2_ROUNDTRIP_CONTRACT_VIOLATION", __abbr__)
    # One tampered UNMEASURED coefficient -> stage 2 must FIRE.
    k_bad2 = k_back_ok.clone()
    k_bad2[0, 10] += 1e-3               # column 10 is not acquired
    _expect_error(lambda: _check_roundtrip_regions(
        aenv, k_bad2, y_meas, k_probe, support, "t", "unm-tamper"),
        "R2_ROUNDTRIP_CONTRACT_VIOLATION", __abbr__)
    return {"branches": 29,
            "ss10_contract": {
                "stage1": "pre-inverse-FFT, bitwise: acquired entries "
                          "exactly the DC write; unmeasured entries "
                          "bitwise the probe",
                "stage2": "post round-trip, tolerance floor_rel on "
                          "BOTH regions",
                "negative_fixtures": ["dc-leak (unmasked y)",
                                      "nonbinary mask, unmeasured col",
                                      "nonbinary mask, acquired col",
                                      "round-trip tamper, acquired",
                                      "round-trip tamper, unmeasured"]}}


# ---------------------------------------------------------------------------
# f07 -- reproduction gates: anchor-exact PASS + every refusal branch.
# ---------------------------------------------------------------------------

def f07_reproduction_gates() -> dict:
    ok = _check_reproduction(PM_STEP0_PSNR, PM_STEP0_NMSE, PM_FINAL_PSNR,
                             PM_FINAL_NMSE, C0_NLL_BATCH_FINAL)
    _check(ok["c0_nll_batch_final"]["measured"] == C0_NLL_BATCH_FINAL,
           "the anchor-exact reproduction must pass")
    _expect_error(lambda: _check_reproduction(
        PM_STEP0_PSNR + 2e-3, PM_STEP0_NMSE, PM_FINAL_PSNR, PM_FINAL_NMSE,
        C0_NLL_BATCH_FINAL), "REPRO_PSNR_MISMATCH", __abbr__)
    _expect_error(lambda: _check_reproduction(
        PM_STEP0_PSNR, PM_STEP0_NMSE, PM_FINAL_PSNR - 2e-3, PM_FINAL_NMSE,
        C0_NLL_BATCH_FINAL), "REPRO_PSNR_MISMATCH", __abbr__)
    _expect_error(lambda: _check_reproduction(
        PM_STEP0_PSNR, PM_STEP0_NMSE * (1.0 + 2e-4), PM_FINAL_PSNR,
        PM_FINAL_NMSE, C0_NLL_BATCH_FINAL), "REPRO_NMSE_MISMATCH",
        __abbr__)
    _expect_error(lambda: _check_reproduction(
        PM_STEP0_PSNR, PM_STEP0_NMSE, PM_FINAL_PSNR,
        PM_FINAL_NMSE * (1.0 - 2e-4), C0_NLL_BATCH_FINAL),
        "REPRO_NMSE_MISMATCH", __abbr__)
    _expect_error(lambda: _check_reproduction(
        PM_STEP0_PSNR, PM_STEP0_NMSE, PM_FINAL_PSNR, PM_FINAL_NMSE,
        float(np.nextafter(C0_NLL_BATCH_FINAL, np.inf))),
        "REPRO_C0_NLL_MISMATCH", __abbr__)
    good = {"bank_sha256": PM_BANK_SHA256,
            "manifest_sha256": PM_BANK_MANIFEST_SHA256}
    _verify_pm_bank(good)
    _expect_error(lambda: _verify_pm_bank(
        {"bank_sha256": "0" * 64,
         "manifest_sha256": PM_BANK_MANIFEST_SHA256}),
        "PM_BANK_MISMATCH", __abbr__)
    return {"branches": 7}


# ---------------------------------------------------------------------------
# f08 -- D3 fixture-set verification: fabricated valid manifests PASS;
# every tamper class refuses.
# ---------------------------------------------------------------------------

def _fabricated_d3_manifests() -> tuple:
    positions = [int(p) for p in v02m.d3_monitor_positions()]
    holdman = {"entries": [{"file": f"h{i}", "slice_index": i}
                           for i in range(v02m.N_HOLDOUT_VOLUMES)]}
    d3man = {"n": N_FIXTURE_SLICES,
             "manifest_sha256": D3_SUBSET_MANIFEST_SHA256,
             "entries": [{"holdout_position": p, "draw_rank": r,
                          "file": f"h{p}", "slice_index": p}
                         for r, p in enumerate(positions)]}
    return d3man, holdman


def f08_d3_subset_verification() -> dict:
    d3man, holdman = _fabricated_d3_manifests()
    _verify_d3_subset(d3man, holdman)          # the valid case must pass
    bad_n = dict(d3man, n=31)
    _expect_error(lambda: _verify_d3_subset(bad_n, holdman),
                  "MANIFEST_CONTENT_MISMATCH", __abbr__)
    tampered = dict(d3man, entries=[dict(e) for e in d3man["entries"]])
    tampered["entries"][3] = dict(tampered["entries"][3],
                                  holdout_position=999)
    _expect_error(lambda: _verify_d3_subset(tampered, holdman),
                  "D3_DRAW_MISMATCH", __abbr__)
    unbound = dict(d3man, entries=[dict(e) for e in d3man["entries"]])
    unbound["entries"][5] = dict(unbound["entries"][5],
                                 file="somewhere_else")
    _expect_error(lambda: _verify_d3_subset(unbound, holdman),
                  "D3_SUBSET_NOT_IN_HOLDOUT", __abbr__)
    drift = dict(d3man, manifest_sha256="0" * 64)
    _expect_error(lambda: _verify_d3_subset(drift, holdman),
                  "D3_SUBSET_MANIFEST_DRIFT", __abbr__)
    return {"branches": 4,
            "draw_positions_first4": [int(p) for p in
                                      v02m.d3_monitor_positions()[:4]]}


# ---------------------------------------------------------------------------
# f09 -- publication machinery in a throwaway directory (semantic hash,
# claim-guarded publish, both rerun-guard refusals, the no-verdict
# invariant). The directory is deleted; NOTHING is published.
# ---------------------------------------------------------------------------

def _pub_env() -> SimpleNamespace:
    try:
        from seqref_mri.tdiag import _bootstrap  # noqa: F401 -- the
        # registered path bootstrap (same contract as _env)
        from preflight_parents import (StageError, attach_semantic_hash,
                                       publish_stage)
    except ImportError as exc:
        _fail("ENV_IMPORT_FAILED",
              f"publication machinery not importable: {exc}")
    return SimpleNamespace(
        StageError=StageError, attach_semantic_hash=attach_semantic_hash,
        publish_stage=publish_stage)


def _mini_facts(payload_x: int) -> dict:
    return {"schema": FACTS_SCHEMA,
            "script": f"{__abbr__} v{__version__}",
            "mode": "fixtures-probe",
            "payload": {"x": payload_x},
            "run": {"utc": "fixture"}}


def f09_publication_machinery() -> dict:
    env = _pub_env()
    with tempfile.TemporaryDirectory() as td:
        path, sha = _publish_facts(env, _mini_facts(1), td)
        _check(Path(path).exists()
               and Path(str(path) + ".sha256").exists(),
               "claim-guarded publication did not materialise file + "
               "sidecar")
        _check(_file_sha256(Path(path)) == sha,
               "published file sha does not match the returned sha")
        # Identical-semantic rerun: permitted (mirrors V02E).
        _publish_facts(env, _mini_facts(1), td)
        # Corrupt the sidecar: the rerun guard must refuse.
        Path(str(path) + ".sha256").write_text("0" * 64 + "\n")
        _expect_error(lambda: _publish_facts(env, _mini_facts(1), td),
                      "RERUN_PRIOR_ARTEFACT_MISMATCH", __abbr__)
        # Restore the sidecar, alter the semantic payload: refuse.
        Path(str(path) + ".sha256").write_text(
            _file_sha256(Path(path)) + "\n")
        _expect_error(lambda: _publish_facts(env, _mini_facts(2), td),
                      "SEMANTIC_RERUN_MISMATCH", __abbr__)
    _expect_error(lambda: v02e._no_verdict_scan(
        {"a": {"verdict": "x"}}, "t"), "FACTS_SCHEMA_VIOLATION",
        "SEQREF-V02E")
    # AMBIGUOUS structural outcome (spec r3 SS8): the complete evidence
    # artifact publishes FIRST; the stage exits 2 only AFTER successful
    # publication. A non-ambiguous outcome publishes and returns.
    with tempfile.TemporaryDirectory() as td2:
        amb_facts = _mini_facts(3)
        amb_facts["outcome"] = {"label": ROUTE_AMBIGUOUS}
        _expect_error(lambda: _publish_and_enforce(env, amb_facts, td2),
                      "R2_OUTCOME_AMBIGUOUS", __abbr__)
        pub = Path(td2) / f"{FACTS_PREFIX}.json"
        _check(pub.exists() and Path(str(pub) + ".sha256").exists(),
               "AMBIGUOUS: the evidence artifact + sidecar must be "
               "published BEFORE the exit-2 error")
        _check(json.loads(pub.read_text())["outcome"]["label"]
               == ROUTE_AMBIGUOUS,
               "the published artifact must carry the AMBIGUOUS "
               "structural record")
    with tempfile.TemporaryDirectory() as td3:
        ok_facts = _mini_facts(4)
        ok_facts["outcome"] = {"label": OUTCOME_A}
        path3, _sha3 = _publish_and_enforce(env, ok_facts, td3)
        _check(Path(path3).exists(),
               "a non-ambiguous outcome must publish and return "
               "normally (no error)")
    return {"branches": 8, "published_and_deleted": True}


# ---------------------------------------------------------------------------
# f10 -- no silent pass: mechanical coverage of THIS module's ERROR codes.
# The universe is EXTRACTED from this module's source at runtime (every
# _fail("CODE") site), never curated by hand; each code is OBSERVED (a
# fixture fired that exact branch in THIS module) or DEFERRED with a
# written justification above.
# ---------------------------------------------------------------------------

def f10_no_silent_pass(results: list) -> dict:
    src = Path(__file__).read_text()
    codes = set(re.findall(r'_fail\("([A-Z][A-Z0-9_]{4,})"', src))
    codes -= _HARNESS_CODES
    _check(codes, "mechanical extraction found no ERROR codes; the "
                  "module source or the _fail convention changed")
    observed = {c for (abbr, c) in _OBSERVED if abbr == __abbr__}
    deferred = set(DEFERRED_JUSTIFICATIONS)
    unknown_obs = observed - codes
    stale_def = deferred - codes
    unjustified = deferred - set(DEFERRED_JUSTIFICATIONS)
    missing = codes - observed - deferred
    _check(not unknown_obs,
           f"observed codes absent from the source universe: "
           f"{sorted(unknown_obs)}")
    _check(not stale_def,
           f"stale deferrals (code no longer in the source): "
           f"{sorted(stale_def)}")
    _check(not unjustified, "a deferral lacks a written justification")
    _check(not missing,
           f"ERROR codes neither exercised nor deferred: "
           f"{sorted(missing)}; an unexercised registered ERROR path is "
           f"an ERROR (SS7 row 10)")
    _check(all(r["result"] == "PASS" for r in results),
           "a suite row did not PASS")
    return {"coverage_ok": True,
            "extracted": sorted(codes), "observed": sorted(observed),
            "deferred": sorted(deferred),
            "deferral_justifications": DEFERRED_JUSTIFICATIONS}


FIXTURES = [
    ("f01_loader_refusals", f01_loader_refusals),
    ("f02_stream_determinism", f02_stream_determinism),
    ("f03_measurement_chain_synthetic", f03_measurement_chain_synthetic),
    ("f04_bootstrap_machinery", f04_bootstrap_machinery),
    ("f05_deterministic_backend", f05_deterministic_backend),
    ("f06_rule_arithmetic", f06_rule_arithmetic),
    ("f07_reproduction_gates", f07_reproduction_gates),
    ("f08_d3_subset_verification", f08_d3_subset_verification),
    ("f09_publication_machinery", f09_publication_machinery),
]


def run_suite() -> dict:
    _OBSERVED.clear()
    # The deterministic backend is pinned BEFORE any fixture tensor work
    # (spec r3 deterministic-pinning section); f05 verifies and records.
    det_state = _pin_deterministic_backend(_env_array().torch)
    results = []
    for name, fn in FIXTURES:
        t0 = time.time()
        evidence = fn()          # any V02Error propagates as suite ERROR
        results.append({"name": name, "result": "PASS",
                        "elapsed_s": time.time() - t0,
                        "evidence": evidence})
        logger.info("[%s] %s: PASS", __abbr__, name)
    evidence10 = f10_no_silent_pass(results)
    results.append({"name": "f10_no_silent_pass", "result": "PASS",
                    "elapsed_s": 0.0, "evidence": evidence10})
    logger.info("[%s] f10_no_silent_pass: PASS", __abbr__)
    return {"schema": "seqref-v02h-selftest-report/1",
            "script": f"{__abbr__} v{__version__}",
            "spec": "SEQREF-V02H v0.1 spec r3 (2026-09-02)",
            "deterministic_execution": det_state,
            "results": results,
            "coverage": evidence10,
            "golden_pins": "pinned 2026-09-10 against NumPy PCG64(4) in "
                           "the campaign sandbox and embedded as "
                           "constants; a generator or stream-order "
                           "change fails the pin, never silently re-pins",
            "taxonomy": "exit 0 = all fixtures PASS + coverage_ok; "
                        "exit 2 = ERROR; no exit 1, no skip (SS7 row 10)"}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"{__abbr__} v{__version__} -- readout selftest "
                    f"fixture (R2 measured-region DC overwrite: no-op "
                    f"or load-bearing; V02AMEND v0.7 SS21 items 5-7)")
    ap.add_argument("--mode", choices=["fixtures", "authoritative"],
                    default="fixtures")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--manifest-dir", default=None)
    ap.add_argument("--train-record", default=None)
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--p0s-facts", default=None)
    ap.add_argument("--p4-stats2", default=None)
    ap.add_argument("--implb-facts", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()
    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode="w"))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s "
                               "%(message)s")
    try:
        if args.mode == "fixtures":
            report = run_suite()
            logger.info("[%s] fixtures mode: %d rows PASS, coverage_ok "
                        "= True; NOTHING published", __abbr__,
                        len(report["results"]))
        else:
            missing = [k for k in ("data_root", "manifest_dir",
                                   "train_record", "run_root",
                                   "p0s_facts", "p4_stats2",
                                   "implb_facts", "out_dir")
                       if getattr(args, k) is None]
            if missing:
                logger.error("[%s] authoritative mode requires %s",
                             __abbr__, ", ".join("--" + k.replace(
                                 "_", "-") for k in missing))
                return EXIT_ERROR
            run(vars(args))
    except v02m.V02Error:
        return EXIT_ERROR
    except Exception:  # noqa: BLE001 -- the registered boundary: no
        logger.exception("[%s] unexpected runtime failure", __abbr__)
        return EXIT_ERROR            # exception may pass silently / exit 1
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())

# =============================================================================
# Changelog
#   * v0.6 (2026-09-10, reviewer NOT-READY follow-up 5, pre-execution
#     spec-alignment correction): the AMBIGUOUS path now follows the
#     frozen spec r3 SS8 semantics -- endpoint routing RETURNS the
#     structural R2_OUTCOME_AMBIGUOUS state instead of raising pre-
#     publication; run() assembles the COMPLETE facts (incl. a new
#     outcome.ambiguity record: decision tally + BOTH statistics for
#     every disagreeing slice), publishes the evidence artifact FIRST
#     via _publish_and_enforce, and exits 2 only AFTER successful
#     publication (the lock decision returns to the owner; no silent
#     tie-break). True implementation/provenance failures (incl. an
#     empty decision list) remain immediate no-publish ERRORs. f09
#     proves AMBIGUOUS -> artifact + sidecar materialise before the
#     R2_OUTCOME_AMBIGUOUS error, and a non-ambiguous outcome publishes
#     and returns normally; f06 asserts routing returns the state.
#     Also carries the pre-execution artifact-identity repair
#     (FACTS_PREFIX = the registered v02_readout_selftest).
#   * v0.5 (2026-09-10, reviewer NOT-READY follow-up 4, pre-execution
#     spec-alignment correction): the registered ZERO-FLOOR SUBSTITUTION
#     (spec r3 SS8) is now applied PER METRIC in _slice_decision -- an
#     EXACTLY-ZERO measured floor is replaced by the pre-registered
#     positive floor_rel=1e-6 before the K=10 comparison (a raw zero
#     threshold would trivialize the rule); a nonzero measured floor,
#     however small, is used as measured. f06 proves both directions:
#     zero substitutes (False below K x 1e-6, True above, STRICT at the
#     1e-5 boundary), per-metric independence (one-sided substitution
#     routes AMBIGUOUS), and nonzero retention (the pre-existing fl=1e-7
#     cases). The branch is dormant on the real complex128 readout path
#     (measured floor ~3e-16) but the rule is frozen, so it is exact.
#     Also carries the v0.4 release-version repair (__version__ now
#     matches the header; a same-file edit race had dropped it).
#   * v0.4 (2026-09-10, reviewer NOT-READY follow-up 3, pre-execution
#     spec-alignment corrections): f04 now exercises the REGISTERED
#     two-conjunct D1/D2 acceptance rule via _d1d2_acceptance -- the
#     practical-effect threshold on the observed mean (>= +0.50 dB /
#     <= 0.95) and the uncertainty threshold on the CI bound (> 0 /
#     < 1.0) separately, incl. the two discriminating cases the v0.3
#     CI-vs-effect-threshold test classified wrongly and a boundary-
#     strictness matrix; the frozen TWO-STAGE R2 structural
#     verification (spec r3 SS10) is implemented and tested -- stage-1
#     pre-inverse-FFT bitwise checks (acquired exactly the DC write,
#     unmeasured bitwise the probe) and stage-2 post round-trip
#     tolerance checks on BOTH regions, with corrupted-DC negative
#     fixtures (new codes R2_OVERWRITE_CONTRACT_VIOLATION /
#     R2_ROUNDTRIP_CONTRACT_VIOLATION, exercised in f06; the post block
#     is now per-region evidence); CUBLAS_WORKSPACE_CONFIG is an
#     explicit pin/refusal (a conflicting pre-existing value fails
#     BACKEND_CONFIG_CONFLICT, exercised in f05; an accepted value is
#     kept, an unset value assigned) -- no setdefault.
#   * v0.3 (2026-09-10, reviewer NOT-READY follow-up 2, pre-execution
#     spec-alignment corrections): added the deterministic-execution
#     contract -- single-thread + torch.use_deterministic_algorithms(True)
#     + TF32 disabled (matmul/cudnn) + cuDNN deterministic/benchmark-off
#     + CUBLAS_WORKSPACE_CONFIG, pinned in BOTH modes before any tensor
#     work, the exact backend state RECORDED in facts
#     (deterministic_execution), rerun variance a DEFECT via the
#     SEMANTIC_RERUN_MISMATCH guard; PCG64(4) restored to its registered
#     V02AMEND role EXACTLY -- the volume-level PAIRED bootstrap index
#     matrix integers(0, 199, size=(10000, 199)), identical indices for
#     R0/Rk, pre-validated on synthetic paired arrays (D1 paired-difference
#     CI, D2 ratio-of-resampled-means CI, boundary/crossing/pairing/
#     determinism proofs, golden pin re-pinned); the mis-specified
#     (10000, 32) slice bootstrap on pre_rms_rel is REMOVED (the R2
#     decision needs no bootstrap). Fixture rows renumbered f04-f10.
#   * v0.2 (2026-09-10, reviewer NOT-READY follow-up, pre-execution
#     spec-alignment corrections): the FINAL checkpoint alone now decides
#     R2 retention (step 0 is interpretive context; divergence recorded,
#     never an ERROR); the R2_NONTRIVIAL decision is z=0-ONLY (the four
#     extra latent draws and the worst-case aggregation are removed);
#     PCG64(4) is bootstrap-only (the fixture z-bank is gone; synthetic
#     fixture data moved to the fixed test-only seed PCG64(20260910),
#     explicitly outside the campaign RNG registry; both golden pins
#     re-pinned); delta_DC is now the MAXIMUM COEFFICIENT-WISE absolute
#     residual (was |complex mean|); the acquired support OMEGA comes
#     from the registered column mask (was |y| > 0); the reproduction
#     block is honestly renamed PM/C0 reproduction (no z=0 R0 claim;
#     true R0 metric reproduction stays V02G scope).
#   * v0.1 (2026-09-10): introduced under SEQREF-V02H v0.1 spec r3
#     (2026-09-02, LudWiki id 146 DRAFT/PRE-EXECUTION; V02AMEND v0.7 SS21
#     items 5-7 replacement). First consumer of the reserved PCG64(4)
#     stream (registry audit PASS 2026-09-02). Implements: the single
#     PCG64(4) stream with golden pins; the pre/floor/post DC residual
#     measurement through the REAL production complex decode seam
#     (tg._decode_z) and the REAL exact-DC assembly
#     (dec.decode_normalised); the exact spec SS8 rule (K=10 FROZEN,
#     floor_rel=1e-6 independent engineering tolerance, never loosened);
#     outcome routing with structural AMBIGUOUS = exit 2; PM/C0
#     reproduction anchors (PM absolutes toleranced, C0 batch NLL exact
#     tie, Z_DIAG bank SHA pins); the frozen D3 32-slice fixture-set
#     verification (draw re-derivation + holdout re-binding + the
#     registered manifest-SHA pin); the slice bootstrap; the seven-row
#     fixture suite with mechanical no-silent-pass coverage. Authoritative
#     mode publishes seqref-v02h-selftest-facts/1 via the claim-guarded
#     machinery with the V02E-mirrored rerun guard. NO scientific
#     verdict: the R2 keep/remove decision is V02AMEND v1.0 scope.
# =============================================================================
