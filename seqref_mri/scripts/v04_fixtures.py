# =============================================================================
# SEQREF-MRI-V04F v0.13 -- scripts.v04_fixtures
# LIFETIME: KEEP
# Purpose: V04 fixture checks (V04PLAN v1.0 article 178 §5): f01 (V04G1
#          pipeline bit-exactness), f02/f03 (V04L objective), f04 (V04P
#          probe), f05/f06 (V04T training driver), f07/f08 (V04E
#          evaluation facts artifact) + f11-f16 (V04M model) + f17 (V04X
#          cross-flow readiness refusal); the V04C classification and
#          V04X cross-flow smokes run as supplementary groups.
#          Scope: the gate, objective, probe, training-driver, evaluation,
#          classification, model and cross-flow modules + f09
#          (evaluation determinism, V03 implementation-level precedent)
#          + f10 (repository-source static audit) -- the registered suite
#          f01-f17 is complete.
#          Synthetic CPU fixtures only (178 §5 convention); f14's authoritative
#          form runs over the 100 real probe batches -- the run here uses
#          synthetic batches and is PRELIMINARY evidence (data not present in
#          this environment), the authoritative f14 runs at the fixture gate.
# CONVENTION: PASS(0) / ERROR(2). Every failure -> logger.error + typed code.
#             No fallback / mock / placeholder / silent pass. Hooks are
#             observational only (record activations; never modify them).
# Changelog (v0.12 -> v0.13, f10 scope correction, 178 §5):
#   * f10 file enumeration changed from filesystem rglob("*.py") to
#     git ls-files -- "*.py" (repository-controlled source files only),
#     with a work-tree-root guard (rev-parse --show-toplevel must equal
#     the fixture root) and typed failure on git launch/decode errors.
#     Frozen 178 §5 specifies the three forbidden-pattern classes and
#     AUDIT_PATTERN_FOUND; it does not mandate filesystem traversal.
#     rglob scanned installed third-party environments in the monorepo
#     (4357 hits, zero in chain code -- STOP-4). P1/P2/P3 logic, the
#     required-path scope guard and the unreadable-file guard are
#     UNCHANGED. Pin-tree equivalence: rglob 203 vs tracked 201 --
#     outcome-equivalent (0 hits both scopes); the two walked-only
#     files are the documented root authoring stubs (logger.error+raise,
#     never imported by the chain). ls-files reads the index: staged
#     files are covered; untracked files are disclosed as out of scope.
#   * f01-f09 + f11-f17 unchanged.
# Changelog (v0.10 -> v0.11, f09 registered, advisor GO Option 2):
#   * NEW f09 (178 §5): evaluation determinism -- mirrors the
#     registered V03 implementation-level f09 (v03_eval selftest):
#     the synthetic aggregate construction is built twice through
#     v04_eval.build_arm_aggregates (the only randomness is the
#     fixed-seed PCG64(3) bootstrap); canonical bytes and semantic
#     hash must be bit-identical, else EVAL_DETERMINISM_MISMATCH.
#     Resolves independent-audit finding 1: the frozen f09 text
#     carries no real-data qualifier and 178 §10 requires no
#     separate determinism rerun -- the earlier deferral to the
#     authoritative-evaluation stage was over-conservative.
#   * Supplementary post-training real-data sibling rerun recorded
#     as a NON-GATING, non-registered stage obligation (advisor
#     Option 2): it adds no acceptance criterion and changes no
#     frozen requirement.
#   * Suite summary line updated to f01-f17 -- the registered
#     fixture suite is now complete. f01-f08 + f10-f17 unchanged.
# Changelog (v0.9 -> v0.10, f10 static audit, 178 §5):
#   * NEW f10: repo-wide static audit over every *.py in the
#     repository (self-inclusive; the task-marker tokens are built
#     dynamically so this file cannot self-trigger) for the frozen
#     silencing-pattern classes -- P1 except-swallow (the suite is
#     only a null statement, any exception type: the failure leaves
#     no trace), P2 silent retry (the suite is only a loop-continue
#     under a BROAD clause: bare / Exception / BaseException; a
#     narrow typed except in a candidate-probe loop with a terminal
#     typed raise is enumeration, not a silent retry), P3 marker-
#     gated return (task-marker comment on the return line or on the
#     immediately preceding non-blank line). Any hit ->
#     AUDIT_PATTERN_FOUND. The mechanical proxies are the
#     implementation-defined static encoding of the frozen class
#     names, disclosed here and in the execution record.
#   * Scope guard: the scan must cover the eight V04 modules, the
#     RealNVP layer module and this fixture file, else
#     AUDIT_SCOPE_INCOMPLETE (a scan of the wrong tree would be a
#     silent pass). Unreadable source files likewise fail the guard.
#   * Result at introduction: 203 files scanned, 0 hits. f09 remains
#     PENDING (authoritative evaluation stage). f01-f08 + f11-f17
#     unchanged.
# Changelog (v0.8 -> v0.9, V04X fixtures, advisor GO f17 + X01-X13):
#   * NEW f17 (178 §5): cross-flow readiness refusal -- an audit
#     invoked with a deliberately failed 173 §8.1 item (the frozen
#     example: a mismatched evaluation-case manifest, realised as a
#     reordered V03 case set) exits 2 with CROSSFLOW_GATE_FAILURE; the
#     audit record carries the failed item, an exploratory-at-most
#     claim and NO confirmatory DeltaDelta.
#   * NEW supplementary V04X cross-flow smoke (non-registered; advisor
#     V04X plan X01-X13): green preflight (explicit non_authoritative
#     record, 178 §12a), V03 pin refusal, preflight case-mismatch,
#     missing / duplicated / reordered identities refused without
#     canonicalisation, NLL-units and data_range comparability
#     refusals, missing deviation registration, V04R-not-registered
#     refusal, all-green audit PASS with exact DeltaDelta arithmetic
#     (binary-exact vectors: +0.75 dB / -0.25 nats, degenerate CIs),
#     bootstrap bitwise determinism, NLL higher-is-better sign
#     orientation, gate-FAIL records carry no DeltaDelta, per-seed
#     spread invariance (only within-architecture ensemble means
#     enter -- seeds never pair across architectures),
#     bitwise-identical double runs from fixed scratch roots.
#   * f09/f10 remain PENDING. f01-f08 + f11-f16 unchanged.
# Changelog (v0.7 -> v0.8, V04C fixtures, advisor GO C01-C16 + sweep):
#   * NEW supplementary V04C routing smoke (non-registered; advisor
#     V04C plan): the frozen 163 §15 matrix with V04 class names on
#     synthetic rows -- all five classes, boundary touches at 0 /
#     +/-0.5 dB / +1% (strict exclusion: a CI containing a boundary is
#     a straddle; the advisor's illustrative C01/C02/C04 CIs crossed
#     the +/-0.5 boundary and were corrected to boundary-clean CIs
#     matching the registered v03_classify selftest convention), A5
#     density-CI precedence, S_inst >=2/3 checked first, incomplete
#     arm -> V04-E, coexistence precedence cases.
#   * Cartesian sweep (4 reconstruction x 3 density x 3 stability
#     states) against an independent fixture-side oracle restating the
#     frozen matrix, plus V03C port-equivalence: every routed state
#     matches v03_classify.classify_arm with V03- renamed to V04-.
#   * Facts-integrity battery through v04_classify.load_verified_facts:
#     sidecar drift / missing, wrong schema, semantic drift, injected
#     verdict / classification keys (INJECTED_VERDICT_FIELD), pre-made
#     verdict under an unexpected top-level key, malformed parent sha
#     (PARENT_REFERENCE_INVALID), arm-coverage / stability-sum refusals,
#     invalid density evidence (FACTS_NON_FINITE_METRIC / missing delta
#     block), manifest identity / count / drift refusals.
#   * Sensitivity invariance (R1/NMSE-only mutations leave the
#     classification detail unchanged) and a 199-volume synthetic
#     end-to-end run through the real entry point v04_classify.run:
#     expected class V04-A, routing-record schema + closed key set +
#     semantic-hash self-consistency, bitwise-identical double run.
#   * f09/f10/f17 remain PENDING. f01-f08 + f11-f16 unchanged.
# Changelog (v0.6 -> v0.7, V04E fixtures, advisor GO + integrity extensions):
#   * NEW f07 (178 §5): verdict-injection refusal -- a complete synthetic
#     facts artifact built through the production v04_eval._build_facts
#     (6 synthetic completed runs, 199 synthetic identities) scans clean,
#     then injected verdict/classification keys (top-level, nested in
#     per_arm, deep in per_volume) are refused with
#     INJECTED_VERDICT_FIELD.
#   * NEW f08 (178 §5): facts schema refusals on the synthetic 199-volume
#     facts -- non-finite metric (FACTS_NON_FINITE_METRIC), n != 199
#     (FACTS_VOLUME_COUNT_MISMATCH), identity misalignment vs the holdout
#     manifest (FACTS_IDENTITY_MISMATCH).
#   * NEW supplementary V04E integrity smoke (non-registered; advisor
#     extension checks): checkpoint sha drift + corrupt state_dict
#     (CHECKPOINT_HASH_MISMATCH), duplicate volume IDs
#     (FACTS_DUPLICATE_VOLUME_ID), Z_DIAG bank pin (realised bank ==
#     registered f8a6a34b...1dc8d6; tamper -> BANK_HASH_MISMATCH),
#     bootstrap determinism + PCG64(3) stream pin, malformed parent
#     references (PARENT_REFERENCE_INVALID), six-run completeness
#     fail-closed (RUN_SET_INCOMPLETE / RUN_RECORD_INVALID branches),
#     V03 facts pin + recorded-CI-settings verification
#     (V03_FACTS_PIN_MISMATCH / CI_SETTINGS_MISMATCH / V03_FACTS_INVALID),
#     and a 2-volume synthetic end-to-end _evaluate_checkpoint proof
#     (finite metrics, bitwise-identical sibling evaluation).
#   * f09/f10 remain PENDING (f09 = full-eval sibling rerun, needs the
#     authoritative eval path; f10 = static audit, repo-wide stage).
#     f01-f06 + f11-f16 unchanged.
# Changelog (v0.5 -> v0.6, V04G1 fixtures, advisor GO + failure injection):
#   * NEW f01 (178 §5): pipeline bit-exactness logic on synthetic
#     artifacts -- pass / drift / missing / unexpected-file refusals via
#     MANIFEST_CONTENT_MISMATCH + reference-table tamper (GATE_INPUT_DRIFT)
#     and missing (GATE_INPUT_MISSING) raises. Authoritative f01 needs
#     the frozen V02/V03 artifacts (absent here); runs at the gate.
#   * NEW supplementary V04G1 refusal-injection checks (non-registered;
#     advisor item 4): all-six-green aggregate PASS + record consumable
#     by the V04T fail-closed loader; per-gate injected failures record
#     FAIL with typed codes (a: drift; b: NaN batch; c: tampered probe
#     record; d: NaN round-trip; e: tampered s_max cancellation; f: FiLM
#     wiring violation); aggregation aborts GATE_FAILURE with the record
#     written; run_gates schema guard; V04T loader refuses a FAIL record.
#   * v04_probe bumped to v0.2 (prepare_probe_batches extracted for gate
#     reuse; estimator/record unchanged). f02-f06 + f11-f16 unchanged.
# Changelog (v0.4 -> v0.5, V04T fixtures, advisor GO Option 2 + A-D):
#   * NEW f05 (178 §5/§9): checkpoint-selection rule on synthetic
#     validation histories -- argmin, earliest-epoch tie-break, empty /
#     non-finite / missing-checkpoint / mismatched-L_val refusals --
#     plus checkpoint-identity checks (safeguard A: round-trip filename
#     parse, arm/seed/attempt/init_seed/lambda fields).
#   * NEW f06 (178 §5/§8): rerun-ledger policy on synthetic ledgers --
#     attempt counting, max-3 refusal, numerical-failure + completed
#     terminal, sidecar tamper detection, attempt-scoped filenames,
#     seed-major order gate, error classification; interruption/resume
#     cases (safeguard B): interrupted pair blocks the chain, retry
#     permitted, prior entries byte-identical after append (append-only
#     proof), interrupted-with-partial-checkpoints case; fail-closed
#     prerequisite gates (safeguard D): probe/gate record hash + schema +
#     field gates, all-PASS enforcement, commit-pin vs repository HEAD
#     (real scratch git repo + non-repo refusal).
#   * NEW supplementary V04T driver smoke (non-registered; loud failure):
#     2-step synthetic end-to-end run (fresh disposable V04M init, own
#     generator + scratch dir) -- checkpoint write + sha + identity,
#     ledger append, frozen selection rule, bitwise determinism across
#     two repetitions; own 2-step budget, exempt from the locked 3258
#     production assertion (safeguard C exemption). f02-f04 + f11-f16
#     unchanged.
# Changelog (v0.3 -> v0.4, V04P fixture):
#   * NEW f04 (178 §5): probe validity branches -- zero / NaN / sub-floor
#     norms, lambda non-finite / <= 0 (finite negative density input) /
#     outside [1e-6, 1e6] -- each asserted PROBE_VALIDITY_FAILURE with
#     branch-attribution markers; happy path lambda_2' = 0.5*(2/1) == 1.0
#     exactly; global_grad_norm no-gradient guard; advisor-endorsed
#     no-parameter-update check (SHA-256 parameter digest bitwise-identical
#     before/after real-path measurement on synthetic batches; gradients
#     left cleared). f02/f03 + f11-f16 unchanged.
# Changelog (v0.2 -> v0.3, V04L fixtures):
#   * NEW f02 (lam=0 bit-exact value + gradient digest + R0-branch-cold
#     proof) and f03 (L_rec MSE contract, finite, gradient flows) per
#     178 §5; synthetic rec batch mirrors the registered V03 selftest.
#   * NEW supplementary lam>0 smoke diagnostic (non-registered; loud
#     failure). f11-f16 (amended f15, art. 180) unchanged.
# Changelog (v0.1 -> v0.2, SEQREF-MRI-V04AMEND-A1 v1.0 FROZEN, art. 180):
#   * f15 amended per article 180 (supersedes 178 §5 f15 (i),(ii),(iv) +
#     §7(f) ref): (i') conditioner-output, (ii') FiLM-internal pre-output,
#     (iii) verbatim gradient-path, (iv') K=3-clone unblocking (FiLM-modulated
#     response + grad conditioning-dependence), (v') shuffled control on the
#     unblocked clone, (vi') FiLM-frozen negative control; 3 sub-seeds; abs
#     noise floor 1e-9; FiLM-exclusivity wiring assertion. f11-f14/f16 and
#     failure semantics (CONDITIONING_DEAD, exit 2) unchanged.
# =============================================================================
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from seqref_mri.src import free_flow_runtime as ffr
from seqref_mri.src import residual_decoder as dec
from seqref_mri.src.base_experts import CondRealNVP
from seqref_mri.src.conditioner import Conditioner
from seqref_mri.src.fastmri_data import (FastMRISliceDataset,
                                         make_cartesian_mask)
from seqref_mri.src.flows import v03_objective as v03l
from seqref_mri.src.free_flow_runtime import (
    COND_IN_CHANNELS, FLOW_DIM_REAL, GRID_H, GRID_W, H_DIM, MASK_BITS)
from seqref_mri.scripts import v02_manifests as v02m
from seqref_mri.scripts import v03_classify as v03c
from seqref_mri.scripts import v03_eval as v03e
from seqref_mri.scripts import v03_train as v03t
from seqref_mri.scripts import v04_classify as v04c
from seqref_mri.scripts import v04_crossflow as v04x
from seqref_mri.scripts import v04_eval as v04e
from seqref_mri.scripts import v04_objective as v04o
from seqref_mri.scripts import v04_probe as v04p
from seqref_mri.scripts import v04_train as v04t
from seqref_mri.scripts.train_base import _collate
from seqref_mri.scripts.v04_model import (
    V04M_D_IN, V04M_HIDDEN, V04M_N_LAYERS, V04M_S_MAX, build_v04_model)

logger = logging.getLogger(__name__)
__version__ = "0.12"
__abbr__ = "SEQREF-MRI-V04F"

# Fixture determinism (recorded constants; fixtures must be reproducible)
FIXTURE_SEED = 20261011
F13_SEED = 20261012
F13_FD_EPS = 1e-3
F13_REL_TOL = 1e-3
F13_DENOM_FLOOR = 1e-2
F15_REL_EPS = 1e-6
F15_DENOM_FLOOR = 1e-12
F15_ABS_FLOOR = 1e-9          # A1 numerical-noise floor (mean |delta|)
F15_CLONE_STEPS = 3           # A1 K=3 Adam unblocking steps on clones
F15_CLONE_LR = 1e-4           # A1 Adam lr (PyTorch defaults otherwise)
F15_SUBSEEDS = (20261011, 20261012, 20261013)  # A1 repeatability
F02F03_BATCH = 2
F02F03_MASK_SEED_BASE = FIXTURE_SEED + 100  # synthetic Cartesian mask seeds
F16_CANCEL_TOL = 1e-5
F12_ROUNDTRIP_TOL = 1e-4
F14_BATCHES = 4            # PRELIMINARY: synthetic; authoritative = 100 probe batches
F14_BATCH_SIZE = 32
F04_MEASURE_BATCHES = 2    # synthetic batches for the measurement-path check
F04_E2E_BATCHES = 2        # patched N_PROBE_BATCHES for the _run seam test
F04_E2E_DATA_SEED = 20261013
F04_E2E_CENTRE = range(44, 52)   # always-acquired cols (p4_scaling_stats2)
# FIXED workspace root (never mkdtemp): the probe record embeds the P4
# parent path, so a random root would make the record sha256 -- and the
# suite log -- vary per run and break the byte-determinism comparison.
F04_E2E_ROOT = Path("/tmp/v04p_f04e2e_workspace")
V04T_SMOKE_SEED = 20261014  # driver-smoke generator seed (supplementary)
V04T_SMOKE_STEPS = 2        # smoke's own budget; exempt from the locked 3258
F07F08_SEED = 20261016      # synthetic facts generator seed (f07/f08)
V04E_SMOKE_SEED = 20261017  # integrity-smoke generator seed (supplementary)
V04E_SMOKE_MASK_SEED_BASE = V04E_SMOKE_SEED + 200  # synthetic mask seeds


class FixtureFailure(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[SEQREF-MRI-V04F] %s: %s", code, message)
    raise FixtureFailure(code, message)


def _synthetic_batch(gen: torch.Generator, batch: int):
    """Synthetic free-coordinate state + conditioning, V03 shapes/dtypes."""
    u = torch.randn(batch, FLOW_DIM_REAL, generator=gen)          # f32
    cond_in = torch.randn(batch, COND_IN_CHANNELS, 96, 96, generator=gen)
    mask = (torch.rand(batch, MASK_BITS, generator=gen) > 0.5).to(torch.float32)
    return u, cond_in, mask


def _post_outputs(layer, x1, h):
    """Observational capture of the raw (pre-clamp) post output."""
    captured = {}
    handle = layer.post.register_forward_hook(
        lambda mod, inp, out: captured.setdefault("raw", out.detach()))
    try:
        s, t = layer._st(x1, h)
    finally:
        handle.remove()
    if "raw" not in captured:
        _fail("HOOK_CAPTURE_FAILED", "post forward hook captured nothing")
    return captured["raw"], s, t


# -- f02/f03: V04L objective fixtures (178 §5) --------------------------------
def _synthetic_rec_batch(gen: torch.Generator, batch: int = F02F03_BATCH):
    """Synthetic conditioning + decode inputs for the objective fixtures.
    Mirrors the registered V03 selftest pattern: synthetic 4x-style
    Cartesian masks -> per-sample coordinate maps; fixture-standardisation
    vectors (mean 0, scale 1); synthetic y_raw/amax/tgt_norm/u_scaled."""
    maps, masks = [], []
    for i in range(batch):
        m = make_cartesian_mask(GRID_W, seed=F02F03_MASK_SEED_BASE + i)
        cols = [int(c) for c in np.flatnonzero(m)]
        cmap = dec.build_coordinate_map(cols, GRID_H, GRID_W)
        n = cmap.n_free_complex
        vecs = {"mean_re": np.zeros(n), "scale_re": np.ones(n),
                "mean_im": np.zeros(n), "scale_im": np.ones(n),
                "n_free_complex": n}               # fixture-synthetic
        maps.append((cmap, vecs))
        masks.append(torch.as_tensor(m, dtype=torch.bool))
    mask = torch.stack(masks)                      # (B,96) bool
    cond_in = torch.randn(batch, COND_IN_CHANNELS, GRID_H, GRID_W,
                          generator=gen)
    y_raw = torch.complex(torch.randn(batch, GRID_H, GRID_W, generator=gen),
                          torch.randn(batch, GRID_H, GRID_W,
                                      generator=gen)).to(torch.complex64)
    amax = torch.rand(batch, generator=gen) + 0.5
    tgt_norm = torch.rand(batch, 1, GRID_H, GRID_W, generator=gen)
    u_scaled = torch.randn(batch, FLOW_DIM_REAL, generator=gen)
    return u_scaled, cond_in, mask, y_raw, amax, tgt_norm, maps


def _grad_digest(model) -> str:
    """Incremental SHA-256 over (name, presence, raw bytes) in fixed
    parameter order (V03 f02b pattern -- raw f32 bytes are bitwise truth)."""
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        g = p.grad
        h.update(n.encode())
        h.update(b"\x01" if g is not None else b"\x00")
        if g is not None:
            h.update(g.detach().numpy().tobytes())
    return h.hexdigest()


def f02_lambda0_equivalence(model, gen) -> None:
    u, cond_in, mask, y_raw, amax, tgt_norm, maps = _synthetic_rec_batch(gen)
    nll_ref = v04o.density_only(model, u, cond_in, mask)
    nll_l0 = v04o.combined_loss(model, u, cond_in, mask, lam=0.0,
                                y_raw=y_raw, amax=amax, tgt_norm=tgt_norm,
                                maps=maps)
    if float(nll_l0) != float(nll_ref):
        _fail("LAMBDA0_PATH_MISMATCH",
              "lam=0 value differs from the density-only path")
    nll_bare = v04o.combined_loss(model, u, cond_in, mask, lam=0.0,
                                  y_raw=None, amax=None, tgt_norm=None,
                                  maps=[])
    if float(nll_bare) != float(nll_ref):
        _fail("LAMBDA0_PATH_MISMATCH",
              "lam=0 with empty maps differs; the R0 branch was entered")
    model.zero_grad(set_to_none=True)
    v04o.density_only(model, u, cond_in, mask).backward()
    ref_digest = _grad_digest(model)
    model.zero_grad(set_to_none=True)
    v04o.combined_loss(model, u, cond_in, mask, lam=0.0,
                       y_raw=None, amax=None, tgt_norm=None,
                       maps=[]).backward()
    if _grad_digest(model) != ref_digest:
        _fail("LAMBDA0_PATH_MISMATCH",
              "lam=0 parameter gradients differ from the density-only path")
    model.zero_grad(set_to_none=True)
    logger.info("f02 PASS: lam=0 bit-exact value + gradients; "
                "R0 branch cold")


def f03_lrec_contract(model, gen) -> None:
    _, cond_in, mask, y_raw, amax, tgt_norm, maps = _synthetic_rec_batch(gen)
    mag = torch.rand(2, GRID_H, GRID_W, dtype=torch.float64, generator=gen)
    if float(v03l.mse_normalised(mag, mag.clone())) != 0.0:
        _fail("LOSS_CONTRACT_VIOLATION", "MSE(x,x) != 0")
    if float(v03l.mse_normalised(mag, mag + 0.1)) <= 0.0:
        _fail("LOSS_CONTRACT_VIOLATION", "MSE(x,y) <= 0 for x != y")
    model.zero_grad(set_to_none=True)
    rec = v04o.rec_term(model, cond_in, mask, y_raw, amax, tgt_norm, maps)
    if not torch.isfinite(rec):
        _fail("LOSS_CONTRACT_VIOLATION", "L_rec is non-finite")
    rec.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        _fail("LOSS_CONTRACT_VIOLATION", "L_rec produced no gradients")
    gnorm = float(torch.sqrt(sum((g ** 2).sum() for g in grads)))
    if not np.isfinite(gnorm) or gnorm == 0.0:
        _fail("LOSS_CONTRACT_VIOLATION",
              f"L_rec gradient norm is {gnorm!r}; must be finite and > 0")
    model.zero_grad(set_to_none=True)
    logger.info("f03 PASS: MSE contract ok; L_rec=%.6f finite; "
                "grad norm=%.6e", float(rec), gnorm)


def lambda_positive_smoke(model, gen) -> None:
    """Supplementary diagnostic (NOT a registered fixture): combined loss
    and gradient norm finite at lam=1.0. Violation -> LOSS_CONTRACT_VIOLATION."""
    u, cond_in, mask, y_raw, amax, tgt_norm, maps = _synthetic_rec_batch(gen)
    model.zero_grad(set_to_none=True)
    total = v04o.combined_loss(model, u, cond_in, mask, lam=1.0,
                               y_raw=y_raw, amax=amax, tgt_norm=tgt_norm,
                               maps=maps)
    if not torch.isfinite(total):
        _fail("LOSS_CONTRACT_VIOLATION", "combined lam=1.0 non-finite")
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        _fail("LOSS_CONTRACT_VIOLATION",
              "combined lam=1.0 produced no gradients")
    gnorm = float(torch.sqrt(sum((g ** 2).sum() for g in grads)))
    if not np.isfinite(gnorm) or gnorm == 0.0:
        _fail("LOSS_CONTRACT_VIOLATION",
              f"combined lam=1.0 gradient norm is {gnorm!r}")
    model.zero_grad(set_to_none=True)
    logger.info("lam>0 smoke (supplementary, non-registered): "
                "combined(1.0)=%.6f grad norm=%.6e", float(total), gnorm)


# -- f04: probe validity branches + no-parameter-update check (178 §5) -----
def _f04_e2e_write_volume(path: Path, n_slices: int) -> None:
    """Synthetic single-coil volume satisfying every FastMRISliceDataset
    gate: 'kspace' complex64 (n_slices, 96, 96), 'reconstruction_esc'
    present, file attr 'max' finite > 0. Fixed-seed values -> identical
    slice VALUES across runs (the batch-list digest covers only
    (file, slice_index) strings; gradients see values, not h5 metadata)."""
    rng = np.random.default_rng(F04_E2E_DATA_SEED)
    shape = (n_slices, GRID_H, GRID_W)
    kspace = (rng.standard_normal(shape, dtype=np.float32)
              + 1j * rng.standard_normal(shape, dtype=np.float32)
              ).astype(np.complex64)
    with h5py.File(path, "w") as h:
        h.create_dataset("kspace", data=kspace)
        h.create_dataset("reconstruction_esc",
                         data=np.zeros(shape, dtype=np.float32))
        h.attrs["max"] = 1.0


def _f04_e2e_write_p4(path: Path) -> tuple:
    """Synthetic P4 /2 artifact passing every ffr.load_p4s2_parent gate:
    schema seqref-p4-stats/2, stage P4, authoritative PASS, PER-LOCATION
    branch, exactly REGISTERED_P4_LOCATIONS=8448 records = the 96x96 grid
    minus the always-acquired centre columns 44..51 (the p4_scaling_stats2
    eligibility rule), identity affine pairs (mean 0.0, scale 1.0). The
    embedded semantic hash is fixture-defined (canonical locations view);
    the loader checks it for equality against the patched pin only. Bytes
    are canonical_json, so the file hash is deterministic. Returns
    (file_sha256, semantic_sha256)."""
    locations = [{"row": r, "column": c,
                  "applied_mean_re": 0.0, "applied_mean_im": 0.0,
                  "applied_scale_re": 1.0, "applied_scale_im": 1.0}
                 for r in range(GRID_H) for c in range(GRID_W)
                 if c not in F04_E2E_CENTRE]
    if len(locations) != ffr.REGISTERED_P4_LOCATIONS:
        _fail("PROBE_E2E_FIXTURE_INVALID",
              f"synthetic location count {len(locations)} != "
              f"{ffr.REGISTERED_P4_LOCATIONS}")
    sem = hashlib.sha256(v02m.canonical_json(
        {"locations": locations})).hexdigest()
    art = {"schema": ffr.P4S2_FACTS_SCHEMA, "stage": "P4",
           "authoritative": True, "run_mode": "authoritative",
           "verdict": "PASS", "branch": {"selected": "PER-LOCATION"},
           "semantic_sha256": sem, "locations": locations}
    payload = v02m.canonical_json(art)
    path.write_bytes(payload)
    file_sha = hashlib.sha256(payload).hexdigest()
    (path.parent / (path.name + ".sha256")).write_text(
        f"{file_sha}  {path.name}\n")
    return file_sha, sem


def _f04_end_to_end_run() -> None:
    """V04P-FIX coverage closure (reviewer safeguard 1): the PRODUCTION
    v04_probe._run seam end-to-end on a synthetic train split + synthetic
    P4 /2 parent. f04 previously exercised measure_probe_gradients and
    the abort branches in memory but never _run, which let the v0.2
    double-hexdigest record-seam defect escape every verification layer.
    Asserts: exit 0; record + sidecar exist and are recompute-consistent;
    batch_list_sha256 is a 64-char lowercase hex string equal to an
    INDEPENDENTLY recomputed digest; lambda internal consistency from the
    record's own per-batch arrays; parameter_sha256 equals the fresh
    registered init; weight_state / parents / schema / locked fields exact.
    Pin seam: load_p4s2_parent binds its pin defaults at DEFINITION time,
    so the registered pins are swapped through __kwdefaults__ (patching
    the module constants would be a dead write) and restored in finally;
    every loader gate still runs in full against the synthetic artifact.
    N_PROBE_BATCHES is a runtime global lookup and is patched directly."""
    root = F04_E2E_ROOT
    if root.exists():
        shutil.rmtree(root)
    data_root = root / "data"
    split_dir = data_root / "knee_singlecoil_train" / "singlecoil_train"
    split_dir.mkdir(parents=True)
    p4_path = root / "p4" / "scaling_statistics.json"
    p4_path.parent.mkdir(parents=True)
    out_dir = root / "out"
    try:
        _f04_e2e_write_volume(split_dir / "synthetic_volume.h5",
                              F04_E2E_BATCHES * v02m.BATCH_SIZE)
        file_sha, sem_sha = _f04_e2e_write_p4(p4_path)
        kw = ffr.load_p4s2_parent.__kwdefaults__
        saved = (kw["expected_file_sha"], kw["expected_semantic_sha"],
                 v04p.N_PROBE_BATCHES)
        kw["expected_file_sha"] = file_sha
        kw["expected_semantic_sha"] = sem_sha
        v04p.N_PROBE_BATCHES = F04_E2E_BATCHES
        try:
            rc = v04p._run(str(data_root), str(p4_path), str(out_dir))
        finally:
            kw["expected_file_sha"] = saved[0]
            kw["expected_semantic_sha"] = saved[1]
            v04p.N_PROBE_BATCHES = saved[2]
        if rc != 0:
            _fail("PROBE_E2E_RUN_FAILED",
                  f"production v04_probe._run returned {rc}, expected 0")

        rec_path = out_dir / "v04_probe_record.json"
        side_path = out_dir / "v04_probe_record.json.sha256"
        if not rec_path.is_file() or not side_path.is_file():
            _fail("PROBE_E2E_RECORD_MISSING",
                  "production _run returned 0 but the record or its "
                  "sidecar is missing")
        rec_bytes = rec_path.read_bytes()
        rec = json.loads(rec_bytes.decode("utf-8"))
        rec_sha = hashlib.sha256(rec_bytes).hexdigest()
        if side_path.read_text() != f"{rec_sha}  {rec_path.name}\n":
            _fail("PROBE_E2E_SIDECAR_MISMATCH",
                  "record sidecar does not match '<sha256>  <name>' of "
                  "the record bytes")
        bls = rec.get("batch_list_sha256")
        if not isinstance(bls, str) or not re.fullmatch(r"[0-9a-f]{64}",
                                                        bls):
            _fail("PROBE_E2E_BATCHLIST_FORMAT",
                  f"batch_list_sha256 is {bls!r}, not a 64-char lowercase "
                  "hex string")

        # Independent digest: fixture-side iteration of the same dataset
        # with the production collate, hashing (file, slice_index) per
        # batch -- recomputed from the data, never from the record.
        ds = FastMRISliceDataset(str(data_root), split="train", mode="eval")
        loader = DataLoader(ds, batch_size=v02m.BATCH_SIZE, shuffle=False,
                            num_workers=0, collate_fn=_collate)
        if len(loader) != F04_E2E_BATCHES:
            _fail("PROBE_E2E_FIXTURE_INVALID",
                  f"synthetic split yields {len(loader)} batches, expected "
                  f"{F04_E2E_BATCHES}")
        indep = hashlib.sha256()
        for bi, batch in enumerate(loader):
            if bi >= F04_E2E_BATCHES:
                break
            indep.update(v02m.canonical_json(
                {"batch": bi,
                 "meta": [{"file": m["file"], "slice_index": m["slice_index"]}
                          for m in batch["meta"]]}))
        if bls != indep.hexdigest():
            _fail("PROBE_E2E_BATCHLIST_MISMATCH",
                  "record batch_list_sha256 does not equal the "
                  "independently recomputed digest")

        # Lambda internal consistency from the record's own arrays.
        gd = np.asarray(rec["g_density_per_batch"], dtype=np.float64)
        gr = np.asarray(rec["g_rec_per_batch"], dtype=np.float64)
        if gd.shape != (F04_E2E_BATCHES,) or gr.shape != (F04_E2E_BATCHES,):
            _fail("PROBE_E2E_RECORD_INVALID",
                  f"per-batch gradient arrays have shapes {gd.shape}/"
                  f"{gr.shape}, expected ({F04_E2E_BATCHES},)")
        gd_mean, gr_mean = float(gd.mean()), float(gr.mean())
        if gd_mean != rec["g_density_mean"] or gr_mean != rec["g_rec_mean"]:
            _fail("PROBE_E2E_RECORD_INVALID",
                  "record means do not match their per-batch arrays")
        lam = float(v04p.LAMBDA_PRIME_FACTOR * (gd_mean / gr_mean))
        if lam != rec["lambda_prime"]:
            _fail("PROBE_E2E_RECORD_INVALID",
                  f"record lambda_2'={rec['lambda_prime']!r} != "
                  f"{lam!r} recomputed from the record's own means")

        # Weight state: the measured model must be the fresh registered
        # init -- a second build_v04_model() digests identically.
        if rec["parameter_sha256"] != v04p.parameter_digest(
                build_v04_model()):
            _fail("PROBE_E2E_WEIGHT_STATE",
                  "record parameter_sha256 does not match a fresh "
                  "registered init digest")
        if rec["weight_state"] != {"kind": v04p.PROBE_WEIGHT_STATE,
                                   "model_init_seed":
                                   v04p.PROBE_INIT_SEED}:
            _fail("PROBE_E2E_RECORD_INVALID",
                  f"weight_state is {rec['weight_state']!r}")

        # Locked record fields (178 SS7(c)).
        locked = (("schema", v04p.SCHEMA),
                  ("script", f"{v04p.__abbr__} v{v04p.__version__}"),
                  ("spec", {"article": 176, "status": "FROZEN"}),
                  ("plan", {"article": 178, "status": "FROZEN"}),
                  ("n_batches", F04_E2E_BATCHES),
                  ("batch_size", v02m.BATCH_SIZE),
                  ("split", "train"), ("mode", "eval"),
                  ("role", "validation"),
                  ("validation_source_amendment", "A2"),
                  ("lambda_prime_factor", v04p.LAMBDA_PRIME_FACTOR),
                  ("lambda_band", [v04p.LAMBDA_BAND[0], v04p.LAMBDA_BAND[1]]),
                  ("norm_floor", v04p.NORM_FLOOR),
                  ("l_rec_scale", v04o.L_REC_SCALE))
        for key, want in locked:
            if rec.get(key) != want:
                _fail("PROBE_E2E_RECORD_INVALID",
                      f"record field {key!r} is {rec.get(key)!r}, "
                      f"expected {want!r}")
        par = (rec.get("parents") or {}).get("p4_stats2") or {}
        if (par.get("path") != str(p4_path)
                or par.get("file_sha256") != file_sha
                or par.get("semantic_sha256") != sem_sha):
            _fail("PROBE_E2E_RECORD_INVALID",
                  f"parents.p4_stats2 is {par!r}")
        logger.info("f04 end-to-end PASS: production _run exit 0; record "
                    "sha256 %s... sidecar-consistent; batch_list_sha256 "
                    "matches the independent digest; lambda_2'=%.6e "
                    "internally consistent; parameter digest == fresh "
                    "registered init", rec_sha[:12], rec["lambda_prime"])
    finally:
        if root.exists():
            shutil.rmtree(root)


def f04_probe_validity(model, gen) -> None:
    """178 §5 f04: every probe-failure branch aborts with the typed
    PROBE_VALIDITY_FAILURE (never a generic exception), the happy-path
    conversion is exact, the real measurement path is proven to leave
    the model bitwise unchanged (no parameter update, gradients cleared),
    and the production _run prepare->record seam is exercised end-to-end
    on synthetic inputs (V04P-FIX coverage closure, safeguard 1)."""
    # Failure branches. Marker = substring that must appear in the abort
    # message, attributing the abort to the intended gate (norm gates fire
    # before the lambda gates; lambda<=0 is tested with a finite negative
    # density input so it cannot be caught by a norm gate first).
    cases = [
        ("zero density norm",
         lambda: v04p.compute_lambda_prime(0.0, 1.0), "g_density"),
        ("NaN rec norm",
         lambda: v04p.compute_lambda_prime(1.0, float("nan")), "g_rec"),
        ("norm below 1e-12 floor",
         lambda: v04p.compute_lambda_prime(1.0, 5e-13), "floor"),
        ("lambda non-finite",
         lambda: v04p.compute_lambda_prime(1e300, 1e-11), "lambda"),
        ("lambda <= 0 (finite negative density input)",
         lambda: v04p.compute_lambda_prime(-1.0, 1.0), "non-positive"),
        ("lambda below band",
         lambda: v04p.compute_lambda_prime(1.0, 1e12), "band"),
        ("lambda above band",
         lambda: v04p.compute_lambda_prime(1e12, 1.0), "band"),
    ]
    for name, fn, marker in cases:
        try:
            fn()
        except v04p.V04PError as exc:
            if exc.code != "PROBE_VALIDITY_FAILURE":
                _fail("PROBE_FIXTURE_WRONG_CODE",
                      f"branch {name!r} raised {exc.code}, expected "
                      "PROBE_VALIDITY_FAILURE")
            if marker not in str(exc):
                _fail("PROBE_FIXTURE_WRONG_BRANCH",
                      f"branch {name!r} aborted without marker {marker!r}: "
                      f"{exc}")
            logger.info("f04 branch PASS: %s", name)
        else:
            _fail("PROBE_FIXTURE_NO_ABORT",
                  f"branch {name!r} did not abort; the §6 validity gate "
                  "is dead")

    # Happy path: lambda_2' = 0.5 * (2.0 / 1.0) = 1.0 exactly.
    got = v04p.compute_lambda_prime(2.0, 1.0)
    if got != 1.0:
        _fail("PROBE_FIXTURE_WRONG_VALUE",
              f"lambda_2'(2.0, 1.0) = {got!r} != 1.0")

    # global_grad_norm must refuse a gradient-free model (zero-norm guard).
    try:
        v04p.global_grad_norm(torch.nn.Linear(3, 2))
    except v04p.V04PError as exc:
        if exc.code != "PROBE_VALIDITY_FAILURE":
            _fail("PROBE_FIXTURE_WRONG_CODE",
                  f"no-grad guard raised {exc.code}")
    else:
        _fail("PROBE_FIXTURE_NO_ABORT",
              "global_grad_norm accepted a gradient-free model")

    # No-parameter-update check (advisor-endorsed): the real measurement
    # path on synthetic batches must leave parameters bitwise identical and
    # gradients cleared. The probe module constructs no optimizer at all;
    # an Adam step would change the digest.
    before = v04p.parameter_digest(model)
    batches = []
    for _ in range(F04_MEASURE_BATCHES):
        u, cond_in, mask, y_raw, amax, tgt_norm, maps = \
            _synthetic_rec_batch(gen)
        batches.append({"u_scaled": u, "cond_in": cond_in, "mask": mask,
                        "y_raw": y_raw, "amax": amax, "tgt_norm": tgt_norm,
                        "maps": maps})
    res = v04p.measure_probe_gradients(model, batches)
    if v04p.parameter_digest(model) != before:
        _fail("PROBE_STATE_MUTATION",
              "parameters changed during probe measurement")
    if any(p.grad is not None for p in model.parameters()):
        _fail("PROBE_STATE_MUTATION",
              "gradients left populated after probe measurement")
    for key in ("g_density_mean", "g_rec_mean"):
        v = res[key]
        if not np.isfinite(v) or v <= 0.0:
            _fail("PROBE_FIXTURE_WRONG_VALUE",
                  f"synthetic {key}={v!r}; expected finite and > 0")
    _f04_end_to_end_run()
    logger.info("f04 PASS: 7 abort branches + happy path (lambda_2'=1.0) + "
                "no-grad guard; measurement path bitwise parameter-neutral "
                "(g_d=%.6e g_r=%.6e over %d synthetic batches); production "
                "_run seam verified end-to-end",
                res["g_density_mean"], res["g_rec_mean"],
                F04_MEASURE_BATCHES)


# -- f01: pipeline bit-exactness (178 §5; V04G1 gate a) ----------------------
def _synthetic_pipeline_dir(tmp: Path):
    """Synthetic 'frozen pipeline' directory + matching reference table."""
    target = tmp / "pipeline"
    target.mkdir(parents=True)
    (target / "manifest_a.json").write_text("alpha-manifest\n")
    (target / "tensors.bin").write_bytes(b"\x01\x02\x03\x04")
    return target


def _reference_for(target: Path, tmp: Path) -> Path:
    from seqref_mri.scripts import v04_gate as v04g   # lazy (no top cycle)
    ref = {"schema": v04g.REF_SCHEMA, "artifacts": {}}
    for f in sorted(target.rglob("*")):
        if f.is_file() and f.suffix != ".sha256":
            ref["artifacts"][f.relative_to(target).as_posix()] = \
                hashlib.sha256(f.read_bytes()).hexdigest()
    rp = tmp / "reference.json"
    v04t._write_with_sidecar(rp, v04t.canonical_json(ref))
    return rp


def f01_manifest_bitexact() -> None:
    """178 §5 f01: frozen manifest/pipeline bit-exactness. Synthetic
    artifacts prove the logic; the authoritative form (frozen V02/V03
    artifacts + tensor SHAs) runs at the gate -- those artifacts are not
    present in this environment."""
    from seqref_mri.scripts import v04_gate as v04g   # lazy (no top cycle)
    tmp = Path(tempfile.mkdtemp(prefix="v04g1_f01_"))
    target = _synthetic_pipeline_dir(tmp)
    rp = _reference_for(target, tmp)
    ok = v04g.gate_a_pipeline_bitexact(rp, target)
    if ok["status"] != "PASS" or ok["artifacts_verified"] != 2:
        _fail("MANIFEST_CONTENT_MISMATCH",
              f"matching pipeline refused: {ok}")
    logger.info("f01 PASS: exact-match accept (2 artifacts)")

    drift_dir = _synthetic_pipeline_dir(tmp / "d1")
    (drift_dir / "manifest_a.json").write_text("ALPHA-manifest\n")
    r = v04g.gate_a_pipeline_bitexact(rp, drift_dir)
    if r["status"] != "FAIL" or r["code"] != "MANIFEST_CONTENT_MISMATCH" \
            or r["drift"][0]["reason"] != "sha256 drift":
        _fail("MANIFEST_CONTENT_MISMATCH", f"drift not detected: {r}")
    missing_dir = _synthetic_pipeline_dir(tmp / "d2")
    (missing_dir / "tensors.bin").unlink()
    r = v04g.gate_a_pipeline_bitexact(rp, missing_dir)
    if r["status"] != "FAIL" or r["drift"][0]["reason"] != "missing":
        _fail("MANIFEST_CONTENT_MISMATCH", f"missing not detected: {r}")
    extra_dir = _synthetic_pipeline_dir(tmp / "d3")
    (extra_dir / "unexpected.txt").write_text("intruder\n")
    r = v04g.gate_a_pipeline_bitexact(rp, extra_dir)
    if r["status"] != "FAIL" or r["drift"][0]["reason"] \
            != "unexpected file":
        _fail("MANIFEST_CONTENT_MISMATCH", f"extra not detected: {r}")
    logger.info("f01 PASS: drift / missing / unexpected-file refusals")

    # reference-table input gates (typed raises, input-level)
    tampered = tmp / "reference_tampered.json"
    v04t._write_with_sidecar(
        tampered, v04t.canonical_json({"schema": v04g.REF_SCHEMA,
                                       "artifacts": {"x": "y"}}))
    tampered.write_bytes(v04t.canonical_json(
        {"schema": v04g.REF_SCHEMA, "artifacts": {"x": "z"}}))
    try:
        v04g.gate_a_pipeline_bitexact(tampered, target)
    except v04g.V04G1Error as exc:
        if exc.code != "GATE_INPUT_DRIFT":
            _fail("GATE_FIXTURE_WRONG_CODE",
                  f"tampered reference raised {exc.code}")
        logger.info("f01 branch PASS: reference tamper -> GATE_INPUT_DRIFT")
    else:
        _fail("GATE_INPUT_DRIFT", "tampered reference accepted")
    try:
        v04g.gate_a_pipeline_bitexact(tmp / "absent.json", target)
    except v04g.V04G1Error as exc:
        if exc.code != "GATE_INPUT_MISSING":
            _fail("GATE_FIXTURE_WRONG_CODE",
                  f"missing reference raised {exc.code}")
        logger.info("f01 branch PASS: missing reference -> "
                    "GATE_INPUT_MISSING")
    else:
        _fail("GATE_INPUT_MISSING", "missing reference accepted")


# -- V04G1 refusal injection (supplementary, non-registered) ----------------
def v04g1_gate_refusal(model, gen) -> None:
    """Failure-injection proof (advisor item 4): the gate records FAIL
    with typed codes when any prerequisite fails, the aggregate aborts
    GATE_FAILURE with the record written, and an all-green run is the
    only PASS path. Synthetic inputs; NOT a registered fixture."""
    from seqref_mri.scripts import v04_gate as v04g   # lazy (no top cycle)
    tmp = Path(tempfile.mkdtemp(prefix="v04g1_refusal_"))

    target = _synthetic_pipeline_dir(tmp)
    rp = _reference_for(target, tmp)
    fresh = build_v04_model()
    good_probe = _good_probe_record()
    good_probe["parameter_sha256"] = v04p.parameter_digest(fresh)
    gp = tmp / "probe_good.json"
    v04t._write_with_sidecar(gp, v04t.canonical_json(good_probe))
    u, cond_in, mask, y_raw, amax, tgt_norm, maps = _synthetic_rec_batch(
        gen)
    batch = {"u_scaled": u, "cond_in": cond_in, "mask": mask}

    # all-green evaluation (each gate evaluated once, real measurements)
    green = {
        "a": v04g.gate_a_pipeline_bitexact(rp, target),
        "b": v04g.gate_b_lambda0_parity(model, batch),
        "c": v04g.gate_c_probe_record(str(gp)),
        "d": v04g.gate_d_invertibility(model, gen),
        "e": v04g.gate_e_logdet(model, gen),
        "f": v04g.gate_f_conditioning(model, gen),
    }
    bad = {k: v for k, v in green.items() if v["status"] != "PASS"}
    if bad:
        _fail("GATE_FIXTURE_WRONG_VALUE",
              f"green evaluation not all PASS: {bad}")
    rec = v04g.run_gates(dict(green), tmp / "pass_run",
                         form="preliminary-synthetic")
    if rec["form"] != "preliminary-synthetic":
        _fail("GATE_FIXTURE_WRONG_VALUE", "form field wrong")
    # cross-module: the V04T fail-closed loader must ACCEPT the PASS record
    v04t.load_gate_record(str(tmp / "pass_run" / "v04_gate_record.json"))
    logger.info("v04g1 refusal PASS path: all six gates green; record "
                "accepted by the V04T loader")

    # per-gate injection (sabotaged inputs -> FAIL + typed code)
    bad_batch = {"u_scaled": u.clone(), "cond_in": cond_in, "mask": mask}
    bad_batch["u_scaled"][0, 0] = float("nan")
    tampered_probe = tmp / "probe_tampered.json"
    v04t._write_with_sidecar(tampered_probe, v04t.canonical_json(
        good_probe))
    # body overwritten after the sidecar was written -> hash mismatch
    tampered_probe.write_bytes(v04t.canonical_json(
        dict(good_probe, lambda_prime=3.0)))
    bad_tuples = [(u.clone(), cond_in, mask)]
    bad_tuples[0][0][0, 0] = float("nan")
    sabotaged_e = copy.deepcopy(model)
    # fault injection simulating a defective ldj: the returned forward
    # log-det is offset by +1 while the hook-recomputed path is unchanged
    _orig_enc = sabotaged_e.encode_free
    sabotaged_e.encode_free = lambda *a, **k: (
        lambda r: (r[0], r[1] + 1.0))(_orig_enc(*a, **k))
    sabotaged_f = copy.deepcopy(model)
    sabotaged_f.flow.layers[0].concat = object()  # wiring violation
    injections = [
        ("a", lambda: v04g.gate_a_pipeline_bitexact(rp, drift_dir_for(tmp)),
         "MANIFEST_CONTENT_MISMATCH"),
        ("b", lambda: v04g.gate_b_lambda0_parity(model, bad_batch),
         "LAMBDA0_STEP_EXCEPTION:ValueError"),
        ("c", lambda: v04g.gate_c_probe_record(str(tampered_probe)),
         "PROBE_RECORD_HASH_MISMATCH"),
        ("d", lambda: v04g.gate_d_invertibility(model, gen,
                                                batches=bad_tuples),
         "INVERTIBILITY_EXCEPTION:ValueError"),
        ("e", lambda: v04g.gate_e_logdet(sabotaged_e, gen),
         "LOGDET_MISMATCH"),
        ("f", lambda: v04g.gate_f_conditioning(sabotaged_f, gen),
         "CONDITIONING_DEAD"),
    ]
    injected = {}
    for name, fn, code in injections:
        r = fn()
        if r["status"] != "FAIL" or r["code"] != code:
            _fail("GATE_FIXTURE_WRONG_VALUE",
                  f"injection {name}: got status={r['status']} "
                  f"code={r.get('code')}, expected FAIL/{code}")
        injected[name] = r
        logger.info("v04g1 refusal branch PASS: gate %s -> FAIL %s",
                    name, code)

    # aggregation: one real FAIL result among the green five -> the
    # record is written AND the run aborts GATE_FAILURE
    gates_fail = dict(green)
    gates_fail["c"] = injected["c"]
    try:
        v04g.run_gates(gates_fail, tmp / "fail_run",
                       form="preliminary-synthetic")
    except v04g.V04G1Error as exc:
        if exc.code != "GATE_FAILURE":
            _fail("GATE_FIXTURE_WRONG_CODE",
                  f"aggregate abort raised {exc.code}")
    else:
        _fail("GATE_FAILURE", "a FAIL gate did not abort the aggregate")
    fail_rec = json.loads(
        (tmp / "fail_run" / "v04_gate_record.json").read_text())
    if fail_rec["gates"]["c"]["status"] != "FAIL":
        _fail("GATE_FIXTURE_WRONG_VALUE",
              "FAIL record does not record the failed gate")
    if not (tmp / "fail_run" / "v04_gate_record.json.sha256").is_file():
        _fail("GATE_FIXTURE_WRONG_VALUE", "FAIL record lacks sidecar")
    # cross-module: the V04T fail-closed loader must REFUSE a FAIL record
    try:
        v04t.load_gate_record(str(tmp / "fail_run" / "v04_gate_record.json"))
    except v04t.V04TError as exc:
        if exc.code != "GATE_NOT_PASSED":
            _fail("GATE_FIXTURE_WRONG_CODE",
                  f"V04T loader raised {exc.code}, expected "
                  "GATE_NOT_PASSED")
    else:
        _fail("GATE_NOT_PASSED", "V04T accepted a FAIL gate record")
    # aggregate schema guard
    try:
        v04g.run_gates({k: v for k, v in green.items() if k != "f"},
                       tmp / "schema_run", form="preliminary-synthetic")
    except v04g.V04G1Error as exc:
        if exc.code != "GATE_SCHEMA_MISMATCH":
            _fail("GATE_FIXTURE_WRONG_CODE",
                  f"schema guard raised {exc.code}")
    else:
        _fail("GATE_SCHEMA_MISMATCH", "incomplete gate set accepted")
    logger.info("v04g1 refusal PASS (supplementary, non-registered): 6 "
                "injections + aggregate abort + record-written proof + "
                "V04T cross-refusal + schema guard")


def drift_dir_for(tmp: Path) -> Path:
    """Sabotaged copy of the synthetic pipeline for gate-a injection."""
    src = tmp / "pipeline"
    dst = tmp / "pipeline_drifted"
    dst.mkdir(exist_ok=True)
    for f in src.iterdir():
        dst.joinpath(f.name).write_bytes(f.read_bytes())
    (dst / "manifest_a.json").write_text("SABOTAGED\n")
    return dst


# -- f05: checkpoint selection + identity (178 §5/§9; safeguard A) ---------
def _ck_entry(epoch, l_val, arm=1, seed=101, attempt=1, lam=2.0):
    return {"epoch": epoch,
            "file": v04t.checkpoint_name(arm, seed, attempt, epoch),
            "file_sha256": f"fixture-sha-{epoch}", "arm": arm,
            "seed": seed, "attempt": attempt, "lambda": lam,
            "init_seed": seed, "val_NLL": l_val, "val_L_rec": 0.0,
            "L_val": l_val}


def f05_checkpoint_selection() -> None:
    """178 §5 f05: argmin L_val + earliest-epoch tie-break on synthetic
    histories; malformed inputs -> SELECTION_RULE_MISMATCH. Plus the
    checkpoint-identity gate (safeguard A)."""
    hist = [{"epoch": 1, "L_val": 3.0}, {"epoch": 2, "L_val": 2.5},
            {"epoch": 3, "L_val": 2.7}]
    cks = [_ck_entry(e["epoch"], e["L_val"]) for e in hist]
    sel = v04t.select_checkpoint(hist, cks)
    if sel["epoch"] != 2 or sel["L_val"] != 2.5:
        _fail("SELECTION_RULE_MISMATCH",
              f"argmin selected epoch {sel['epoch']}, expected 2")
    hist_tie = [{"epoch": 1, "L_val": 2.5}, {"epoch": 2, "L_val": 2.5},
                {"epoch": 3, "L_val": 2.5}]
    sel_tie = v04t.select_checkpoint(
        hist_tie, [_ck_entry(e["epoch"], e["L_val"]) for e in hist_tie])
    if sel_tie["epoch"] != 1:
        _fail("SELECTION_RULE_MISMATCH",
              f"tie broke to epoch {sel_tie['epoch']}, expected earliest "
              "(1)")
    logger.info("f05 PASS: argmin + earliest-epoch tie-break")

    bad_hist = list(hist)
    bad_hist[1] = {"epoch": 2, "L_val": float("nan")}
    cases = [
        ("empty history",
         lambda: v04t.select_checkpoint([], cks)),
        ("non-finite L_val",
         lambda: v04t.select_checkpoint(bad_hist, cks)),
        ("selected epoch missing checkpoint",
         lambda: v04t.select_checkpoint(hist, [cks[0], cks[2]])),
        ("checkpoint L_val mismatch",
         lambda: v04t.select_checkpoint(
             hist, [_ck_entry(1, 3.0), _ck_entry(2, 9.9),
                    _ck_entry(3, 2.7)])),
    ]
    for name, fn in cases:
        try:
            fn()
        except v04t.V04TError as exc:
            if exc.code != "SELECTION_RULE_MISMATCH":
                _fail("SELECTION_FIXTURE_WRONG_CODE",
                      f"f05 case {name!r} raised {exc.code}")
            logger.info("f05 branch PASS: %s", name)
        else:
            _fail("SELECTION_RULE_MISMATCH",
                  f"f05 case {name!r} accepted; the rule is dead")

    # Safeguard A: checkpoint identity -- accept correct, refuse wrong.
    good = _ck_entry(1, 3.0)
    v04t.assert_checkpoint_identity(good, 1, 101, 1, 2.0)   # no exception
    id_cases = [
        ("arm field", {**good, "arm": 0}),
        ("init_seed field", {**good, "init_seed": 202}),
        ("lambda field", {**good, "lambda": 0.0}),
        ("filename arm", {**good,
                          "file": v04t.checkpoint_name(0, 101, 1, 1)}),
        ("unparseable filename", {**good, "file": "ck.pt"}),
    ]
    for name, entry in id_cases:
        try:
            v04t.assert_checkpoint_identity(entry, 1, 101, 1, 2.0)
        except v04t.V04TError as exc:
            if exc.code != "CHECKPOINT_IDENTITY_MISMATCH":
                _fail("SELECTION_FIXTURE_WRONG_CODE",
                      f"identity case {name!r} raised {exc.code}")
            logger.info("f05 identity branch PASS: %s", name)
        else:
            _fail("CHECKPOINT_IDENTITY_MISMATCH",
                  f"identity case {name!r} accepted")
    logger.info("f05 PASS: 4 selection refusals + 5 identity refusals")


# -- f06: rerun-ledger policy + fail-closed prerequisites (178 §5/§8) ------
def _led_entry(arm, seed, attempt, status, steps=1, error=None):
    return {"arm": arm, "lambda": 0.0 if arm == 0 else 2.0, "seed": seed,
            "attempt": attempt, "status": status, "error_code": error,
            "steps_completed": steps, "wall_clock_s": 1.0,
            "record_file": "r.json", "record_sha256": "x",
            "checkpoints": []}


def _good_probe_record() -> dict:
    return {"schema": v04t.PROBE_SCHEMA, "n_batches": 100,
            "batch_size": 32, "split": "train", "mode": "eval",
            "validation_source_amendment": "A2",
            "l_rec_scale": v04o.L_REC_SCALE,
            "weight_state": {"kind": "init",
                             "model_init_seed": v04p.PROBE_INIT_SEED},
            "lambda_prime_factor": v04p.LAMBDA_PRIME_FACTOR,
            "batch_list_sha256": "ab" * 32, "lambda_prime": 2.0}


def _good_gate_record() -> dict:
    return {"schema": v04t.GATE_SCHEMA,
            "gates": {g: {"status": "PASS"} for g in v04t.GATE_NAMES}}


def _expect_v04t(code: str, name: str, fn) -> None:
    try:
        fn()
    except v04t.V04TError as exc:
        if exc.code != code:
            _fail("LEDGER_FIXTURE_WRONG_CODE",
                  f"f06 case {name!r} raised {exc.code}, expected {code}")
        logger.info("f06 branch PASS: %s", name)
    else:
        _fail(code, f"f06 case {name!r} accepted; the gate is dead")


def f06_rerun_ledger_policy() -> None:
    """178 §5 f06: rerun-ledger bookkeeping (V03T v0.5 policy verbatim) +
    interruption/resume cases (safeguard B) + fail-closed prerequisite
    gates (safeguard D). Synthetic ledgers/records only."""
    tmp = Path(tempfile.mkdtemp(prefix="v04t_f06_"))

    # (a) attempt counting + terminal rules + max-3 enforcement
    lp = tmp / "v04_run_ledger.json"
    if v04t.next_attempt(v04t.load_ledger(lp), 1, 101) != 1:
        _fail("RERUN_POLICY_VIOLATION", "fresh ledger attempt != 1")
    for a in (1, 2, 3):
        v04t.append_entry(lp, _led_entry(1, 101, a, "interrupted",
                                         error="INFRASTRUCTURE_INTERRUPTION"))
    _expect_v04t("RERUN_POLICY_VIOLATION", "max-3 enforcement",
                 lambda: v04t.next_attempt(v04t.load_ledger(lp), 1, 101))
    for status in ("numerical_failure", "completed"):
        lp2 = tmp / f"ledger_{status}.json"
        v04t.append_entry(lp2, _led_entry(1, 202, 1, status))
        _expect_v04t("RERUN_POLICY_VIOLATION", f"{status} terminal",
                     lambda lp2=lp2, status=status: v04t.next_attempt(
                         v04t.load_ledger(lp2), 1, 202))

    # (b) tamper detection (sidecar chain)
    bad = tmp / "ledger_tampered.json"
    v04t.append_entry(bad, _led_entry(1, 101, 1, "interrupted"))
    led_obj = json.loads(bad.read_text())
    led_obj["entries"][0]["status"] = "completed"      # silent mutation
    bad.write_bytes(v04t.canonical_json(led_obj))
    _expect_v04t("LEDGER_TAMPERED", "tampered ledger sidecar chain",
                 lambda: v04t.load_ledger(bad))

    # (c) attempt-scoped evidence naming (no overwrite across attempts)
    if "attempt1" not in v04t.run_record_name(1, 101, 1) \
            or v04t.run_record_name(1, 101, 1) == v04t.run_record_name(
                1, 101, 2) \
            or v04t.checkpoint_name(1, 101, 1, 3) == v04t.checkpoint_name(
                1, 101, 2, 3):
        _fail("RERUN_POLICY_VIOLATION",
              "evidence filenames collide across attempts")

    # (d) locked seed-major order gate (2 arms x 3 seeds)
    fresh = {"schema": v04t.LEDGER_SCHEMA, "entries": []}
    if v04t.expected_next_run(fresh) != (0, 101):
        _fail("ORDER_VIOLATION", "fresh ledger next run != (arm 0, 101)")
    v04t.enforce_order(fresh, 0, 101)                       # allowed
    for a, s in ((1, 101), (1, 303), (0, 202)):
        _expect_v04t("ORDER_VIOLATION", f"out-of-order ({a},{s})",
                     lambda a=a, s=s: v04t.enforce_order(fresh, a, s))
    lp_ord = tmp / "ledger_order.json"
    v04t.append_entry(lp_ord, _led_entry(0, 101, 1, "completed",
                                         steps=v04t.TOTAL_STEPS))
    if v04t.expected_next_run(v04t.load_ledger(lp_ord)) != (1, 101):
        _fail("ORDER_VIOLATION", "completed pair did not advance to (1,101)")
    v04t.append_entry(lp_ord, _led_entry(1, 101, 1, "interrupted",
                                         error="INFRASTRUCTURE_INTERRUPTION"))
    led_ord = v04t.load_ledger(lp_ord)
    if v04t.expected_next_run(led_ord) != (1, 101):
        _fail("ORDER_VIOLATION",
              "interrupted pair must block advance and demand its retry")
    v04t.enforce_order(led_ord, 1, 101)                     # retry allowed
    _expect_v04t("ORDER_VIOLATION", "advance past interrupted pair",
                 lambda: v04t.enforce_order(led_ord, 0, 202))
    v04t.append_entry(lp_ord, _led_entry(1, 101, 2, "numerical_failure",
                                         error="NUMERICAL_FAILURE"))
    if v04t.expected_next_run(v04t.load_ledger(lp_ord)) != (0, 202):
        _fail("ORDER_VIOLATION",
              "terminal numerical failure must advance the chain")

    # (e) error classification (V03T v0.5 taxonomy verbatim)
    if v04t.classify_objective_error("U_NON_FINITE") != "numerical_failure" \
            or v04t.classify_objective_error("LOSS_CONTRACT_VIOLATION") \
            != "failed_validity" \
            or v04t.classify_gradient_error("PROBE_VALIDITY_FAILURE") \
            != "failed_validity":
        _fail("LEDGER_FIXTURE_WRONG_VALUE",
              "error classification mapping wrong")
    logger.info("f06 PASS: attempt/terminal/tamper/naming/order/"
                "classification branches")

    # (B) interruption + resumption: retry permitted after interruption,
    # chain blocked at the pair until terminal, append-only preserved
    # byte-identically. Uses the FIRST global pair (0, 101): the chain
    # gate always points at the first unsettled pair in seed-major order.
    lp_b = tmp / "ledger_resume.json"
    v04t.append_entry(lp_b, _led_entry(0, 101, 1, "interrupted",
                                       steps=40,
                                       error="INFRASTRUCTURE_INTERRUPTION"))
    led_b = v04t.load_ledger(lp_b)
    if v04t.next_attempt(led_b, 0, 101) != 2:
        _fail("RERUN_POLICY_VIOLATION",
              "valid infrastructure resumption refused")
    if v04t.expected_next_run(led_b) != (0, 101):
        _fail("ORDER_VIOLATION", "interrupted pair did not block the chain")
    prior_bytes = v04t.canonical_json(led_b["entries"])   # pre-retry bytes
    v04t.append_entry(lp_b, _led_entry(0, 101, 2, "completed",
                                       steps=v04t.TOTAL_STEPS))
    led_b2 = v04t.load_ledger(lp_b)
    if v04t.canonical_json(led_b2["entries"][:-1]) != prior_bytes:
        _fail("RERUN_POLICY_VIOLATION",
              "append mutated prior ledger entries; append-only broken")
    if v04t.expected_next_run(led_b2) != (1, 101):
        _fail("ORDER_VIOLATION",
              "terminal retry must advance the chain to (1, 101)")
    _expect_v04t("RERUN_POLICY_VIOLATION", "restart after completed",
                 lambda: v04t.next_attempt(led_b2, 0, 101))
    # interrupted checkpoint write: partial checkpoint list on the
    # interrupted attempt does not leak into the retry's identity
    lp_c = tmp / "ledger_partial_ck.json"
    e = _led_entry(1, 303, 1, "interrupted",
                   error="INFRASTRUCTURE_INTERRUPTION")
    e["checkpoints"] = ["deadbeef"]        # partial write remnant
    v04t.append_entry(lp_c, e)
    led_c = v04t.load_ledger(lp_c)
    if v04t.next_attempt(led_c, 1, 303) != 2:
        _fail("RERUN_POLICY_VIOLATION",
              "interrupted (partial checkpoint) pair must be retryable")
    logger.info("f06 PASS (B): interruption/resume + append-only proof")

    # (D) fail-closed prerequisites: probe record, gate record, commit pin
    gp = tmp / "probe_good.json"
    v04t._write_with_sidecar(
        gp, v04t.canonical_json(_good_probe_record()))
    rec = v04t.load_probe_record(str(gp))
    if v04t.lambda_for_arm(rec, 0) != 0.0 \
            or v04t.lambda_for_arm(rec, 1) != 2.0:
        _fail("PROBE_RECORD_LAMBDAS_INVALID", "lambda_for_arm mapping")
    for name, mutate in (
            ("schema", lambda r: r.update(schema="wrong/9")),
            ("validation source", lambda r: r.update(split="val")),
            ("lambda band", lambda r: r.update(lambda_prime=1e9)),
            ("l_rec_scale", lambda r: r.update(l_rec_scale=1.0)),
            ("n_batches", lambda r: r.update(n_batches=99)),
            ("weight state seed",
             lambda r: r["weight_state"].update(model_init_seed=1)),
            ("batch-list hash",
             lambda r: r.update(batch_list_sha256="short"))):
        bad_rec = json.loads(v04t.canonical_json(_good_probe_record()))
        mutate(bad_rec)
        bp = tmp / f"probe_bad_{name.replace(' ', '_')}.json"
        v04t._write_with_sidecar(bp, v04t.canonical_json(bad_rec))
        _expect_v04t("PROBE_RECORD_CONFIG_MISMATCH" if name not in
                     ("schema", "lambda band") else
                     ("PROBE_RECORD_SCHEMA_MISMATCH" if name == "schema"
                      else "PROBE_RECORD_LAMBDAS_INVALID"),
                     f"probe record bad {name}",
                     lambda bp=bp: v04t.load_probe_record(str(bp)))
    tampered = tmp / "probe_tampered.json"
    v04t._write_with_sidecar(
        tampered, v04t.canonical_json(_good_probe_record()))
    tampered.write_bytes(v04t.canonical_json(
        dict(_good_probe_record(), extra="x")))
    _expect_v04t("PROBE_RECORD_HASH_MISMATCH", "probe record tampered",
                 lambda: v04t.load_probe_record(str(tampered)))
    _expect_v04t("TRAIN_PREREQ_MISSING", "probe record missing",
                 lambda: v04t.load_probe_record(str(tmp / "absent.json")))

    gg = tmp / "gate_good.json"
    v04t._write_with_sidecar(gg, v04t.canonical_json(_good_gate_record()))
    v04t.load_gate_record(str(gg))                          # accepted
    bad_gate = _good_gate_record()
    bad_gate["gates"]["c"] = {"status": "FAIL"}
    bg = tmp / "gate_fail.json"
    v04t._write_with_sidecar(bg, v04t.canonical_json(bad_gate))
    _expect_v04t("GATE_NOT_PASSED", "gate record with a FAIL gate",
                 lambda: v04t.load_gate_record(str(bg)))
    bad_schema = _good_gate_record()
    bad_schema["schema"] = "wrong/9"
    bs = tmp / "gate_schema.json"
    v04t._write_with_sidecar(bs, v04t.canonical_json(bad_schema))
    _expect_v04t("GATE_RECORD_SCHEMA_MISMATCH", "gate record bad schema",
                 lambda: v04t.load_gate_record(str(bs)))
    _expect_v04t("TRAIN_PREREQ_MISSING", "gate record missing",
                 lambda: v04t.load_gate_record(str(tmp / "absent.json")))

    # commit pin: real scratch git repo accepts its own HEAD; anything
    # else refuses. (Environment git -- sandbox scratch only.)
    repo = tmp / "scratch_repo"
    repo.mkdir()
    (repo / "f.txt").write_text("pin-me\n")
    for cmd in (["git", "init", "-q"],
                ["git", "add", "f.txt"],
                ["git", "-c", "user.email=fx@example.invalid", "-c",
                 "user.name=fixture", "commit", "-q", "-m", "pin"]):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          check=True, capture_output=True,
                          text=True).stdout.strip()
    if v04t.verify_commit_pin(head, repo) != head:
        _fail("COMMIT_PIN_MISMATCH", "correct pin refused")
    _expect_v04t("COMMIT_PIN_MISMATCH", "wrong pin",
                 lambda: v04t.verify_commit_pin("0" * 40, repo))
    _expect_v04t("COMMIT_PIN_MISMATCH", "non-repository root",
                 lambda: v04t.verify_commit_pin(head, tmp / "not_a_repo"))
    _expect_v04t("COMMIT_PIN_MISMATCH", "empty pin",
                 lambda: v04t.verify_commit_pin("", repo))
    logger.info("f06 PASS (D): probe/gate/commit-pin fail-closed gates "
                "(7 probe + 4 gate + 3 pin branches)")


# -- f07/f08: V04E facts-artifact fixtures (178 §5) -------------------------
def _expect_v04e(code: str, name: str, fn) -> None:
    try:
        fn()
    except v04e.V04EError as exc:
        if exc.code != code:
            _fail("EVAL_FIXTURE_WRONG_CODE",
                  f"{name}: raised {exc.code}, expected {code}")
        logger.info("%s", f"branch PASS: {name}")
    else:
        _fail(code, f"{name}: accepted; the refusal is dead")


def _synthetic_facts(n_volumes: int = None):
    """A complete synthetic facts artifact through the PRODUCTION builder
    v04_eval._build_facts: 6 completed synthetic runs (2 arms x 3 seeds),
    deterministic per-volume metrics over synthetic identities,
    format-valid synthetic parent shas. No model, no data."""
    n = v02m.N_HOLDOUT_VOLUMES if n_volumes is None else n_volumes
    ids = [{"file": f"vol{i:03d}.h5", "slice_index": i}
           for i in range(n)]
    rng = np.random.Generator(np.random.PCG64(F07F08_SEED))
    runs, selections, per_metrics = {}, {}, {}
    for seed, arm in v04t.GLOBAL_ORDER:
        pair = (arm, seed)
        base = rng.standard_normal(n)
        runs[pair] = {"status": "completed", "attempts": 1,
                      "error_code": None, "wall_clock_s": 1.0,
                      "record_file": v04t.run_record_name(arm, seed, 1),
                      "record_sha256": hashlib.sha256(
                          f"rec-{arm}-{seed}".encode()).hexdigest()}
        selections[pair] = {
            "file": v04t.checkpoint_name(arm, seed, 1, 2),
            "file_sha256": hashlib.sha256(
                f"ck-{arm}-{seed}".encode()).hexdigest(),
            "epoch": 2, "L_val": 1.0}
        scale = 30.0 + arm + seed / 1000.0
        per_metrics[pair] = {
            "identities": ids,
            "nll": (100.0 + base).tolist(),
            "psnr_r0": (scale + base).tolist(),
            "psnr_r1": (scale + 0.5 + base).tolist(),
            "nmse_u_r0": (0.10 + 0.01 * base).tolist(),
            "nmse_u_r1": (0.09 + 0.01 * base).tolist()}
    cfg = {"code_commit": "fixture", "code_branch": "fixture",
           "probe_record": "probe.json", "gate_record": "gate.json",
           "v03_facts": "v03_facts.json", "manifest_dir": "man",
           "p4_stats2": "p4.json", "p0s_facts": "p0s.json"}
    state = {"probe": {"lambda_prime": 2.0},
             "runs": runs, "selections": selections,
             "per_metrics": per_metrics,
             "bank_sha256": v04e.Z_DIAG_BANK_SHA256,
             "bank_manifest_sha256": "ab" * 32,
             "parent_shas": {k: hashlib.sha256(k.encode()).hexdigest()
                             for k in ("holdout_manifest", "p4_stats2",
                                       "p0s_facts", "v03_facts",
                                       "probe_record", "gate_record")}}
    return v04e._build_facts(cfg, state), ids


def f07_verdict_injection() -> None:
    """178 §5 f07: a precomputed verdict anywhere in the facts artifact
    is refused with INJECTED_VERDICT_FIELD; the stage is evidence-only
    by preregistration (V04C owns classification, never V04E)."""
    facts, ids = _synthetic_facts()
    v04e.scan_for_verdict(facts)                 # clean artifact passes
    v04e.validate_facts(facts, ids)
    v04e.validate_parent_references(facts["parent_references"])
    nested = copy.deepcopy(facts)
    nested["per_arm"][1]["aggregates"]["delta_vs_lambda0"][
        "delta_psnr_r0"]["verdict"] = "GO"
    deep = copy.deepcopy(facts)
    deep["per_arm_seed"][0]["per_volume"]["classification"] = "X"
    cases = [("top-level verdict key", {**facts, "verdict": "GO"}),
             ("nested classification key", nested),
             ("deep per-volume classification key", deep)]
    for name, bad in cases:
        _expect_v04e("INJECTED_VERDICT_FIELD", f"f07 {name}",
                     lambda bad=bad: v04e.scan_for_verdict(bad))
    logger.info("f07 PASS: verdict-injection refusal (3 injection "
                "branches; clean artifact accepted)")


def f08_facts_schema() -> None:
    """178 §5 f08: facts schema refusals -- non-finite metric, n != 199,
    identity misalignment vs the holdout manifest (typed codes)."""
    facts, ids = _synthetic_facts()
    v04e.validate_facts(facts, ids)              # clean artifact passes
    bad_nan = copy.deepcopy(facts)
    bad_nan["per_arm_seed"][0]["per_volume"]["nll"][5] = float("nan")
    _expect_v04e("FACTS_NON_FINITE_METRIC", "f08 non-finite metric",
                 lambda: v04e.validate_facts(bad_nan, ids))
    bad_count = copy.deepcopy(facts)
    pv = bad_count["per_arm_seed"][0]["per_volume"]
    pv["identities"] = ids[:v02m.N_HOLDOUT_VOLUMES - 1]
    _expect_v04e("FACTS_VOLUME_COUNT_MISMATCH", "f08 volume count 198",
                 lambda: v04e.validate_facts(bad_count, ids))
    bad_ident = copy.deepcopy(facts)
    bad_ident["per_arm_seed"][0]["per_volume"]["identities"][7] = {
        "file": "volX.h5", "slice_index": 7}
    _expect_v04e("FACTS_IDENTITY_MISMATCH", "f08 identity misalignment",
                 lambda: v04e.validate_facts(bad_ident, ids))
    logger.info("f08 PASS: facts schema refusals (3 branches; clean "
                "199-volume artifact accepted)")


# -- V04E integrity smoke (supplementary, non-registered) -------------------
def _synthetic_eval_state(gen: torch.Generator, i: int) -> dict:
    """One synthetic holdout-volume state with the full V02 production
    field set; fixture-synthetic tensors, identity standardisation."""
    m = make_cartesian_mask(GRID_W, seed=V04E_SMOKE_MASK_SEED_BASE + i)
    cols = [int(c) for c in np.flatnonzero(m)]
    cmap = dec.build_coordinate_map(cols, GRID_H, GRID_W)
    n = cmap.n_free_complex
    vecs = {"mean_re": np.zeros(n), "scale_re": np.ones(n),
            "mean_im": np.zeros(n), "scale_im": np.ones(n),
            "n_free_complex": n}                 # fixture-synthetic
    rng = np.random.Generator(np.random.PCG64(V04E_SMOKE_SEED + 100 + i))
    u_true = (rng.standard_normal(n)
              + 1j * rng.standard_normal(n)).astype(np.complex128)
    cond = torch.randn(1, COND_IN_CHANNELS, GRID_H, GRID_W,
                       generator=gen)
    mask = torch.as_tensor(m, dtype=torch.bool).view(1, -1)
    y = torch.complex(torch.randn(1, GRID_H, GRID_W, generator=gen),
                      torch.randn(1, GRID_H, GRID_W,
                                  generator=gen)).to(torch.complex64)
    amax = torch.rand(1, generator=gen) + 0.5
    x_true_mag = torch.rand(GRID_H, GRID_W, generator=gen) + 0.1
    target = np.concatenate([u_true.real, u_true.imag])[None, :]
    return {"identity": {"file": f"synthetic_vol{i:03d}.h5",
                         "slice_index": i},
            "cond": cond, "mask": mask, "y": y, "amax": amax,
            "cmap": cmap, "vecs": vecs, "u_true": u_true,
            "x_true_mag": x_true_mag, "target": target,
            "u_true_energy": float(np.sum(np.abs(u_true) ** 2))}


def _write_synthetic_run_set(root: Path, statuses: dict) -> None:
    """A synthetic 6-run ledger + hash-pinned run records through the
    production V04T writers (append-only ledger + sidecar chain)."""
    root.mkdir(parents=True, exist_ok=True)
    lp = root / "v04_run_ledger.json"
    for seed, arm in v04t.GLOBAL_ORDER:
        status = statuses.get((arm, seed), "completed")
        rf = v04t.run_record_name(arm, seed, 1)
        rec = {"schema": v04t.RECORD_SCHEMA, "arm": arm, "seed": seed,
               "attempt": 1, "lambda": 0.0 if arm == 0 else 2.0,
               "status": status,
               "validation_history": [{"epoch": 1, "L_val": 1.0}],
               "checkpoints": []}
        v04t._write_with_sidecar(root / rf, v04t.canonical_json(rec))
        entry = _led_entry(arm, seed, 1, status)
        entry["record_file"] = rf
        entry["record_sha256"] = hashlib.sha256(
            (root / rf).read_bytes()).hexdigest()
        v04t.append_entry(lp, entry)


def v04e_integrity_smoke() -> None:
    """Supplementary (non-registered) V04E implementation-integrity
    checks (advisor GO extensions): checkpoint sha drift, duplicate
    volume IDs, readout-bank hash, bootstrap determinism, malformed
    parent references, six-run completeness, V03 facts pin + recorded
    CI settings, synthetic end-to-end evaluation determinism."""
    tmp = Path(tempfile.mkdtemp(prefix="v04e_integrity_"))
    gen = torch.Generator().manual_seed(V04E_SMOKE_SEED)
    env = v04e._env()

    # (1) checkpoint loading: green path + sha drift + corrupt state_dict
    model = build_v04_model()
    ck = tmp / "ck.pt"
    torch.save(model.state_dict(), ck)
    sel = {"file": "ck.pt", "epoch": 1, "L_val": 1.0,
           "file_sha256": hashlib.sha256(ck.read_bytes()).hexdigest()}
    loaded = v04e._load_selected_model(env, {"run_dir": str(tmp)}, sel)
    if v04p.parameter_digest(loaded) != v04p.parameter_digest(model):
        _fail("CHECKPOINT_IDENTITY_MISMATCH",
              "v04e smoke: loaded checkpoint parameters diverge from "
              "the saved state_dict")
    ck.write_bytes(ck.read_bytes() + b"drift")
    _expect_v04e("CHECKPOINT_HASH_MISMATCH",
                 "v04e smoke checkpoint sha drift",
                 lambda: v04e._load_selected_model(
                     env, {"run_dir": str(tmp)}, sel))
    sd = model.state_dict()
    corrupt = dict(sd)
    corrupt.pop(next(iter(corrupt)))
    torch.save(corrupt, ck)
    sel2 = {"file": "ck.pt", "epoch": 1, "L_val": 1.0,
            "file_sha256": hashlib.sha256(ck.read_bytes()).hexdigest()}
    _expect_v04e("CHECKPOINT_HASH_MISMATCH",
                 "v04e smoke corrupt state_dict (strict load)",
                 lambda: v04e._load_selected_model(
                     env, {"run_dir": str(tmp)}, sel2))
    logger.info("v04e smoke (1) PASS: checkpoint green path + 2 refusal "
                "branches")

    # (2) duplicate volume identifiers (count preserved, order broken
    # only by the duplication)
    facts, ids = _synthetic_facts()
    dup = copy.deepcopy(facts)
    pv = dup["per_arm_seed"][0]["per_volume"]
    pv["identities"] = [ids[0], ids[0]] + ids[2:]
    _expect_v04e("FACTS_DUPLICATE_VOLUME_ID",
                 "v04e smoke duplicate volume id",
                 lambda: v04e.validate_facts(dup, ids))
    logger.info("v04e smoke (2) PASS: duplicate volume ID refusal")

    # (3) readout-bank hash: the realised Z_DIAG bank IS the registered
    # bank (full sha pinned at implementation; prefix/suffix quoted in
    # 176 §12); a tampered bank is refused.
    bank_rec = v04e._load_zdiag_bank(env)
    if not (v04e.Z_DIAG_BANK_SHA256.startswith("f8a6a34b")
            and v04e.Z_DIAG_BANK_SHA256.endswith("1dc8d6")):
        _fail("BANK_HASH_MISMATCH",
              "the pinned bank sha does not carry the registered "
              "f8a6a34b...1dc8d6 prefix/suffix")
    tampered = bank_rec["bank"].copy()
    tampered[0, 0] += 1.0
    tampered_sha = hashlib.sha256(
        np.ascontiguousarray(tampered).tobytes(order="C")).hexdigest()
    _expect_v04e("BANK_HASH_MISMATCH", "v04e smoke tampered bank",
                 lambda: v04e._check_bank(
                     {"bank_sha256": tampered_sha}))
    logger.info("v04e smoke (3) PASS: bank pin (realised == registered) "
                "+ tamper refusal")

    # (4) bootstrap determinism + PCG64(3) stream pin (V03E selftest
    # block verbatim)
    vals = [float(i) for i in range(1, 51)]
    if v04e.bootstrap_ci(vals, B=500) != v04e.bootstrap_ci(vals, B=500):
        _fail("BOOTSTRAP_INPUT_INVALID",
              "v04e smoke: bootstrap_ci is not deterministic")
    if v04e.bootstrap_relative([v + 1.0 for v in vals], vals, B=500) \
            != v04e.bootstrap_relative([v + 1.0 for v in vals], vals,
                                       B=500):
        _fail("BOOTSTRAP_INPUT_INVALID",
              "v04e smoke: bootstrap_relative is not deterministic")
    a_pin = np.array([2.0, 4.0, 8.0])
    c_pin = np.array([1.0, 2.0, 3.0])
    got = v04e.bootstrap_relative(a_pin, c_pin, B=100)
    rng = np.random.Generator(np.random.PCG64(v04e.BOOTSTRAP_SEED))
    idx = rng.integers(0, 3, size=(100, 3))
    exp_am = c_pin[idx].mean(axis=1)
    exp = (a_pin[idx].mean(axis=1) - exp_am) / np.abs(exp_am)
    elo, ehi = np.percentile(exp, v04e.CI_PERCENTILES)
    exp_rel = ((float(a_pin.mean()) - float(c_pin.mean()))
               / abs(float(c_pin.mean())))
    if got["ci95"] != [float(elo), float(ehi)] \
            or got["relative"] != exp_rel:
        _fail("BOOTSTRAP_INPUT_INVALID",
              "v04e smoke: bootstrap_relative does not follow the "
              "pinned PCG64(3) per-replicate-anchor stream")
    logger.info("v04e smoke (4) PASS: bootstrap determinism + stream pin")

    # (5) malformed parent-artifact references
    good_refs = {"a": {"path": "x.json", "sha256": "ab" * 32}}
    v04e.validate_parent_references(good_refs)
    v04e.validate_parent_references(
        {"train_records": {"r1": {"path": "r.json",
                                  "sha256": "cd" * 32}}})
    ref_cases = [
        ("missing sha256", {"a": {"path": "x.json"}}),
        ("short sha256", {"a": {"path": "x.json", "sha256": "ab"}}),
        ("non-hex sha256", {"a": {"path": "x.json",
                                  "sha256": "zz" * 32}}),
        ("missing path", {"a": {"sha256": "ab" * 32}}),
        ("nested malformed", {"train_records": {"r1": {"path": "r.json",
                                                       "sha256": "x"}}}),
    ]
    for name, bad in ref_cases:
        _expect_v04e("PARENT_REFERENCE_INVALID",
                     f"v04e smoke parent ref {name}",
                     lambda bad=bad: v04e.validate_parent_references(bad))
    logger.info("v04e smoke (5) PASS: parent-reference format gate "
                "(5 branches)")

    # (6) six-run completeness (fail-closed ledger gates)
    good_dir = tmp / "runs_good"
    _write_synthetic_run_set(good_dir, {})
    runs = v04e.load_run_set(good_dir)
    if len(runs) != len(v04t.GLOBAL_ORDER):
        _fail("RUN_SET_INCOMPLETE",
              f"v04e smoke: good run set loaded {len(runs)} pairs")
    miss_dir = tmp / "runs_missing"
    _write_synthetic_run_set(miss_dir, {})
    (miss_dir / v04t.run_record_name(1, 303, 1)).unlink()
    _expect_v04e("RUN_SET_INCOMPLETE", "v04e smoke missing record file",
                 lambda: v04e.load_run_set(miss_dir))
    unsettled_dir = tmp / "runs_unsettled"
    _write_synthetic_run_set(unsettled_dir, {(1, 303): "interrupted"})
    _expect_v04e("RUN_SET_INCOMPLETE", "v04e smoke unsettled pair",
                 lambda: v04e.load_run_set(unsettled_dir))
    drift_dir = tmp / "runs_drift"
    _write_synthetic_run_set(drift_dir, {})
    rf = drift_dir / v04t.run_record_name(0, 101, 1)
    rf.write_bytes(rf.read_bytes() + b" ")   # ledger pin now stale
    _expect_v04e("RUN_RECORD_INVALID", "v04e smoke record drift vs "
                 "ledger pin",
                 lambda: v04e.load_run_set(drift_dir))
    logger.info("v04e smoke (6) PASS: six-run completeness (green + 3 "
                "refusal branches)")

    # (7) V03 facts pin + recorded-CI-settings verification (178 §10)
    ci_block = {"mean": 1.0, "ci95": [0.9, 1.1], "B": v04e.BOOTSTRAP_B,
                "seed": v04e.BOOTSTRAP_SEED, "method": "percentile"}
    v03f = {"schema": v04e.V03_FACTS_SCHEMA,
            "per_arm": [{"arm": 0, "aggregates":
                         {"means_ci": {"nll": ci_block},
                          "delta_vs_lambda0": None}}]}
    vf = tmp / "v03_facts.json"
    v04t._write_with_sidecar(vf, v04t.canonical_json(v03f))
    own_sha = hashlib.sha256(vf.read_bytes()).hexdigest()
    v04e.load_v03_facts_reference(str(vf), expected_sha=own_sha)
    _expect_v04e("V03_FACTS_PIN_MISMATCH",
                 "v04e smoke V03 facts wrong pin",
                 lambda: v04e.load_v03_facts_reference(str(vf)))
    bad_ci = tmp / "v03_facts_badci.json"
    v03f_bad = copy.deepcopy(v03f)
    v03f_bad["per_arm"][0]["aggregates"]["means_ci"]["nll"]["B"] = 5000
    v04t._write_with_sidecar(bad_ci, v04t.canonical_json(v03f_bad))
    bad_sha = hashlib.sha256(bad_ci.read_bytes()).hexdigest()
    _expect_v04e("CI_SETTINGS_MISMATCH",
                 "v04e smoke V03 facts divergent CI settings",
                 lambda: v04e.load_v03_facts_reference(
                     str(bad_ci), expected_sha=bad_sha))
    bad_schema = tmp / "v03_facts_badschema.json"
    v04t._write_with_sidecar(
        bad_schema, v04t.canonical_json({**v03f, "schema": "wrong/9"}))
    bsc_sha = hashlib.sha256(bad_schema.read_bytes()).hexdigest()
    _expect_v04e("V03_FACTS_INVALID", "v04e smoke V03 facts bad schema",
                 lambda: v04e.load_v03_facts_reference(
                     str(bad_schema), expected_sha=bsc_sha))
    logger.info("v04e smoke (7) PASS: V03 facts pin + CI-settings "
                "verification (green + 3 refusal branches)")

    # (8) synthetic end-to-end evaluation determinism: 2 synthetic
    # volumes through the production _evaluate_checkpoint (d2b encode,
    # tiny_gate z=0 decode, d3 posterior mean over the registered bank)
    # twice; per-volume metrics must be bitwise identical.
    states = [_synthetic_eval_state(gen, i) for i in range(2)]
    p1 = v04e._evaluate_checkpoint(env, build_v04_model(), states,
                                   bank_rec["bank"])
    p2 = v04e._evaluate_checkpoint(env, build_v04_model(), states,
                                   bank_rec["bank"])
    if p1 != p2:
        _fail("EVAL_DETERMINISM_MISMATCH",
              "v04e smoke: sibling evaluations diverge")
    for key in ("nll", "psnr_r0", "psnr_r1", "nmse_u_r0", "nmse_u_r1"):
        if not all(math.isfinite(v) for v in p1[key]):
            _fail("FACTS_NON_FINITE_METRIC",
                  f"v04e smoke: end-to-end {key} carries a non-finite "
                  "value")
    if p1["identities"] != [st["identity"] for st in states]:
        _fail("FACTS_IDENTITY_MISMATCH",
              "v04e smoke: per-volume identities out of order")
    logger.info("v04e smoke (8) PASS: end-to-end determinism, 2 "
                "synthetic volumes, bitwise-identical sibling run "
                "(nll[0]=%.6f psnr_r0[0]=%.6f)",
                p1["nll"][0], p1["psnr_r0"][0])
    logger.info("v04e integrity smoke PASS: 8 check groups green "
                "(supplementary, non-registered)")


# -- V04C routing smoke (supplementary, non-registered) ---------------------
def _expect_v04c(code: str, name: str, fn) -> None:
    """Typed-refusal check for the classification layer: V04CError and
    the translated V04EError carry .code; the V02 manifest loader's
    V02Error embeds the code as the message prefix."""
    try:
        fn()
    except (v04c.V04CError, v04e.V04EError) as exc:
        if exc.code != code:
            _fail("CLASSIFY_FIXTURE_WRONG_CODE",
                  f"{name}: raised {exc.code}, expected {code}")
        logger.info("%s", f"branch PASS: {name}")
    except v02m.V02Error as exc:
        if not str(exc).startswith(code + ":"):
            _fail("CLASSIFY_FIXTURE_WRONG_CODE",
                  f"{name}: raised {exc}, expected {code}")
        logger.info("%s", f"branch PASS: {name}")
    else:
        _fail(code, f"{name}: accepted; the refusal is dead")


def _v04c_arm_row(psnr=(0.8, (0.55, 1.05)),
                  nll=(0.005, (-0.01, 0.008)), nf: int = 0,
                  nfv: int = 0, status: str = "complete",
                  arm: int = 1) -> dict:
    """Synthetic V04 per-arm facts row for routing tests. Boundary-clean
    defaults (the CI excludes 0 and both +/-0.5 dB cell boundaries; the
    density CI lies entirely below +1%) matching the registered
    v03_classify selftest convention. nf/nfv count failed seeds; the
    completed count is the remainder of the 3 locked seeds."""
    nc = v04c.N_SEEDS - nf - nfv
    row = {"arm": arm, "lambda": {0: 0.0, 1: 2.0}[arm],
           "ensemble_status": status,
           "stability": {"seeds_completed": nc,
                         "seeds_numerical_failure": nf,
                         "seeds_failed_validity": nfv}}
    if status == "complete":
        row["aggregates"] = {
            "means_ci": {"nll": {"mean": 1.0, "ci95": [0.9, 1.1]}},
            "delta_vs_lambda0": None if arm == 0 else {
                "delta_psnr_r0": {"mean": psnr[0],
                                  "ci95": [psnr[1][0], psnr[1][1]]},
                "delta_nll_relative": {
                    "relative": nll[0],
                    "ci95": [nll[1][0], nll[1][1]]}}}
    return row


def _v04c_matrix_oracle(rec: str, dens: str, s_inst: bool,
                        status: str) -> str:
    """Fixture-side restatement of the frozen 163 §15 matrix (classes
    renamed per 176 §15) -- the independent expectation the
    implementation is checked against."""
    if s_inst:
        return "V04-D"
    if status != "complete":
        return "V04-E"
    if rec == "degradation":
        return "V04-D"
    if rec == "straddle" or dens == "straddle":
        return "V04-E"
    if rec == "improvement" and dens == "preserved":
        return "V04-A"
    if rec == "improvement" and dens == "degraded":
        return "V04-B"
    if rec == "no_change" and dens == "preserved":
        return "V04-C"
    return "V04-E"


def v04c_routing_smoke() -> None:
    """V04C routing fixtures (advisor V04C plan C01-C16 + Cartesian
    sweep + facts-tamper battery + V03C port-equivalence + end-to-end
    bitwise double run). Synthetic values only; no model, no data."""
    # (1)+(2) C01-C13 + coexistence precedence: the frozen matrix on
    # boundary-clean and boundary-touching synthetic rows.
    cases = [
        ("C01", _v04c_arm_row(), "V04-A"),
        ("C02", _v04c_arm_row(nll=(0.03, (0.02, 0.04))), "V04-B"),
        ("C03", _v04c_arm_row(psnr=(0.05, (-0.05, 0.15))), "V04-C"),
        ("C04", _v04c_arm_row(psnr=(-0.8, (-1.05, -0.55))), "V04-D"),
        ("C05", _v04c_arm_row(nf=2, status="incomplete"), "V04-D"),
        ("C06", _v04c_arm_row(nll=(0.005, (-0.01, 0.01))), "V04-E"),
        ("C07", _v04c_arm_row(nll=(0.02, (0.005, 0.03))), "V04-E"),
        ("C07b", _v04c_arm_row(nll=(0.02, (0.01, 0.03))), "V04-E"),
        ("C08", _v04c_arm_row(psnr=(0.6, (0.3, 0.9))), "V04-E"),
        ("C08b", _v04c_arm_row(psnr=(-0.6, (-0.9, -0.3))), "V04-E"),
        ("C10", _v04c_arm_row(psnr=(0.8, (0.0, 1.05))), "V04-E"),
        ("C10b", _v04c_arm_row(psnr=(-0.8, (-1.05, 0.0))), "V04-E"),
        ("C11", _v04c_arm_row(psnr=(0.5, (0.05, 0.45))), "V04-A"),
        ("C11b", _v04c_arm_row(psnr=(0.5, (0.1, 0.9))), "V04-E"),
        ("C12", _v04c_arm_row(psnr=(-0.5, (-0.9, -0.55))), "V04-D"),
        ("C12b", _v04c_arm_row(psnr=(-0.5, (-0.9, -0.5))), "V04-E"),
        ("C13", _v04c_arm_row(nfv=1, status="incomplete"), "V04-E"),
        # instability / straddle / material change coexisting: the
        # registered precedence decides, never the case order.
        ("P1", _v04c_arm_row(nf=2, status="incomplete"), "V04-D"),
        ("P2", _v04c_arm_row(psnr=(-0.8, (-1.05, -0.55)),
                             nll=(0.005, (-0.01, 0.01))), "V04-D"),
        ("P3", _v04c_arm_row(psnr=(0.6, (0.3, 0.9)),
                             nll=(0.03, (0.02, 0.04))), "V04-E"),
        ("P4", _v04c_arm_row(psnr=(0.05, (-0.05, 0.15)),
                             nll=(0.03, (0.02, 0.04))), "V04-E"),
    ]
    classes = {f"V04-{c}" for c in "ABCDE"}
    for name, row, want in cases:
        got = v04c.route_arm(row)
        if got["class"] != want:
            _fail("CLASSIFY_FIXTURE_MISMATCH",
                  f"{name}: class {got['class']} != expected {want}")
        if got["class"] not in classes:
            _fail("CLASSIFY_FIXTURE_MISMATCH",
                  f"{name}: class {got['class']} is not a V04 class")
        rec = dens = None
        if row["ensemble_status"] == "complete":
            d = row["aggregates"]["delta_vs_lambda0"]
            rec = v03c.classify_reconstruction(
                d["delta_psnr_r0"]["mean"],
                tuple(d["delta_psnr_r0"]["ci95"]))
            dens = v03c.classify_density(
                d["delta_nll_relative"]["relative"],
                tuple(d["delta_nll_relative"]["ci95"]))
        oracle = _v04c_matrix_oracle(
            rec, dens, v04c.classify_stability(row),
            row["ensemble_status"])
        if got["class"] != oracle:
            _fail("CLASSIFY_FIXTURE_MISMATCH",
                  f"{name}: class {got['class']} != matrix oracle "
                  f"{oracle}")
    logger.info("v04c smoke (1) PASS: C01-C13 boundary cases + "
                "coexistence precedence, %d cases", len(cases))

    # (3) V03C port-equivalence: the predicates are the registered V03
    # implementations, and every routed state matches
    # v03_classify.classify_arm with the class prefix renamed.
    if v04c.classify_reconstruction is not v03c.classify_reconstruction \
            or v04c.classify_density is not v03c.classify_density:
        _fail("CLASSIFY_FIXTURE_MISMATCH",
              "V04C decision predicates are not the registered V03C "
              "implementations")
    if (v04c.REC_IMPROVE_DB, v04c.REC_DEGRADE_DB,
            v04c.NLL_REL_TOLERANCE) != (0.5, -0.5, 0.01):
        _fail("CLASSIFY_FIXTURE_MISMATCH",
              "inherited threshold values drifted from 176 §14")
    for name, row, want in cases:
        v03_row = copy.deepcopy(row)
        v03_row["arm"] = 2
        v03_row["lambda"] = 0.5
        renamed = v03c.classify_arm(v03_row)["class"].replace(
            "V03-", "V04-")
        if renamed != v04c.route_arm(row)["class"]:
            _fail("CLASSIFY_FIXTURE_MISMATCH",
                  f"{name}: V04C route diverges from the renamed V03C "
                  f"route ({renamed})")
    logger.info("v04c smoke (2) PASS: V03C predicate identity + "
                "renamed-route equivalence, %d cases", len(cases))

    # (4) Cartesian sweep: 4 reconstruction x 3 density x 3 stability
    # states; every combination yields exactly one frozen class, equal
    # to the fixture-side oracle, and each representative state really
    # produces its predicate label.
    rec_states = {"improvement": (0.8, (0.55, 1.05)),
                  "no_change": (0.05, (-0.05, 0.15)),
                  "degradation": (-0.8, (-1.05, -0.55)),
                  "straddle": (0.6, (0.3, 0.9))}
    dens_states = {"preserved": (0.005, (-0.01, 0.008)),
                   "degraded": (0.03, (0.02, 0.04)),
                   "straddle": (0.005, (-0.01, 0.01))}
    n_combo = 0
    for rl, psnr in rec_states.items():
        for dl, nll in dens_states.items():
            for nf in (0, 1, 2):
                status = "complete" if nf == 0 else "incomplete"
                row = _v04c_arm_row(psnr=psnr, nll=nll, nf=nf,
                                    status=status)
                if status == "complete":
                    d = row["aggregates"]["delta_vs_lambda0"]
                    if v03c.classify_reconstruction(
                            d["delta_psnr_r0"]["mean"],
                            tuple(d["delta_psnr_r0"]["ci95"])) != rl \
                            or v03c.classify_density(
                                d["delta_nll_relative"]["relative"],
                                tuple(d["delta_nll_relative"]["ci95"])
                                ) != dl:
                        _fail("CLASSIFY_FIXTURE_MISMATCH",
                              f"sweep representative ({rl}, {dl}) "
                              f"mislabelled by the frozen predicates")
                got = v04c.route_arm(row)["class"]
                want = _v04c_matrix_oracle(rl, dl, nf >= 2, status)
                if got != want or got not in classes:
                    _fail("CLASSIFY_FIXTURE_MISMATCH",
                          f"sweep ({rl}, {dl}, nf={nf}): {got} != "
                          f"oracle {want}")
                n_combo += 1
    logger.info("v04c smoke (3) PASS: Cartesian sweep, %d state "
                "combinations, all singly-classified", n_combo)

    # (5) Facts-integrity battery through the real loading path
    # (C09/C14 + one-field-at-a-time tampering).
    tmp = Path(tempfile.mkdtemp(prefix="v04c_smoke_"))
    base_facts, ids = _synthetic_facts()

    def _seal(facts: dict) -> dict:
        sealed = copy.deepcopy(facts)
        semantic = {k: v for k, v in sealed.items()
                    if k not in v04e._SEMANTIC_EXCLUDE}
        sealed["semantic_sha256"] = v04e._canonical_hash()(semantic)
        sealed["semantic_scope"] = {
            "included_keys": sorted(semantic.keys()),
            "excluded": list(v04e._SEMANTIC_EXCLUDE),
            "canonicalisation": "preflight_io.canonical_hash"}
        return sealed

    def _manifest(man_dir: Path, man_ids: list) -> None:
        entries = [{"file": e["file"],
                    "slice_index": int(e["slice_index"]),
                    "volume_ordinal": i}
                   for i, e in enumerate(man_ids)]
        man = {"schema": v02m.SCHEMA, "kind": "holdout",
               "selection": {"rule": "synthetic fixture identities",
                             "split": "val", "mode": "eval"},
               "n_volumes": len(entries), "entries": entries}
        man["manifest_sha256"] = v02m.manifest_sha256(man)
        v04e._write_with_sidecar(man_dir / v04e.HOLDOUT_MANIFEST, man)

    def _write_raw(sub: str, sealed: dict, man_ids: list = None):
        d = tmp / sub
        man_dir = d / "man"
        man_dir.mkdir(parents=True)
        _manifest(man_dir, ids if man_ids is None else man_ids)
        fp = d / "v04_facts.json"
        v04e._write_with_sidecar(fp, sealed)
        return fp, man_dir

    def _load(sub: str, facts: dict, man_ids: list = None):
        fp, man_dir = _write_raw(sub, _seal(facts), man_ids)
        return v04c.load_verified_facts(fp, man_dir)

    rows, facts_sha, manifest_sha = _load("green", base_facts)
    if sorted(rows) != [0, 1]:
        _fail("CLASSIFY_FIXTURE_MISMATCH",
              "green load: rows do not cover arms {0, 1}")
    if v04c.route_overall(
            [v04c.route_arm(rows[a]) for a in v04c.CLASS_ARMS]
            )["class"] != "V04-A":
        _fail("CLASSIFY_FIXTURE_MISMATCH",
              "green load: synthetic facts must route to V04-A")
    logger.info("v04c smoke (4) PASS: green synthetic facts load + "
                "route (199 identities, V04-A)")

    bad = copy.deepcopy(base_facts)
    bad["verdict"] = "V04-A"
    _expect_v04c("INJECTED_VERDICT_FIELD", "C14 top-level verdict key",
                 lambda: _load("t_verdict", bad))
    bad = copy.deepcopy(base_facts)
    bad["per_arm"][1]["classification"] = "V04-A"
    _expect_v04c("INJECTED_VERDICT_FIELD", "C14 nested classification "
                 "key", lambda: _load("t_class", bad))
    bad = copy.deepcopy(base_facts)
    bad["overall"] = {"class": "V04-A"}
    _expect_v04c("FACTS_RECORD_INVALID",
                 "pre-made verdict under an unexpected top-level key",
                 lambda: _load("t_overall", bad))
    bad = copy.deepcopy(base_facts)
    bad["schema"] = "seqref-v03-facts/1"
    _expect_v04c("FACTS_RECORD_INVALID", "wrong facts schema",
                 lambda: _load("t_schema", bad))
    drift = _seal(base_facts)
    drift["per_arm"][1]["aggregates"]["delta_vs_lambda0"][
        "delta_psnr_r0"]["mean"] = 9.9
    _expect_v04c("FACTS_RECORD_INVALID", "semantic drift (no rehash)",
                 lambda: v04c.load_verified_facts(
                     *_write_raw("t_drift", drift)))
    fp, man_dir = _write_raw("t_sidecar", _seal(base_facts))
    fp.write_bytes(fp.read_bytes() + b" ")
    _expect_v04c("CLASSIFY_INPUT_DRIFT", "facts sidecar drift",
                 lambda: v04c.load_verified_facts(fp, man_dir))
    fp, man_dir = _write_raw("t_missing", _seal(base_facts))
    fp.with_suffix(fp.suffix + ".sha256").unlink()
    _expect_v04c("CLASSIFY_INPUT_MISSING", "facts sidecar missing",
                 lambda: v04c.load_verified_facts(fp, man_dir))
    bad = copy.deepcopy(base_facts)
    bad["parent_references"]["v03_facts"]["sha256"] = "zz" * 32
    _expect_v04c("PARENT_REFERENCE_INVALID", "malformed parent sha",
                 lambda: _load("t_parent", bad))
    bad = copy.deepcopy(base_facts)
    bad["per_arm"] = [r for r in bad["per_arm"] if r["arm"] == 0]
    _expect_v04c("FACTS_RECORD_INVALID", "parity arm missing",
                 lambda: _load("t_armdrop", bad))
    bad = copy.deepcopy(base_facts)
    bad["per_arm"][1]["arm"] = 2
    _expect_v04c("FACTS_RECORD_INVALID", "arm label tampered",
                 lambda: _load("t_armlab", bad))
    bad = copy.deepcopy(base_facts)
    bad["per_arm"][1]["stability"]["seeds_failed_validity"] = 1
    _expect_v04c("FACTS_RECORD_INVALID", "stability counts sum != 3",
                 lambda: _load("t_stab", bad))
    drift = _seal(base_facts)
    drift["per_arm"][1]["aggregates"]["delta_vs_lambda0"][
        "delta_nll_relative"] = {"relative": float("nan"),
                                 "ci95": [float("nan"), float("nan")]}
    _expect_v04c("FACTS_NON_FINITE_METRIC",
                 "C09 non-finite density evidence",
                 lambda: v04c.load_verified_facts(
                     *_write_raw("t_nan", drift)))
    bad = copy.deepcopy(base_facts)
    del bad["per_arm"][1]["aggregates"]["delta_vs_lambda0"][
        "delta_nll_relative"]
    _expect_v04c("FACTS_RECORD_INVALID", "C09 density block missing",
                 lambda: _load("t_noden", bad))
    other_ids = [{"file": f"oth{i:03d}.h5", "slice_index": i}
                 for i in range(len(ids))]
    _expect_v04c("FACTS_IDENTITY_MISMATCH",
                 "manifest identities != facts identities",
                 lambda: _load("t_mident", base_facts, other_ids))
    _expect_v04c("MANIFEST_CONTENT_MISMATCH",
                 "manifest diverges from the locked 199 volumes",
                 lambda: _load("t_mcount", base_facts, ids[:-1]))
    fp, man_dir = _write_raw("t_mdrift", _seal(base_facts))
    mpath = man_dir / v04e.HOLDOUT_MANIFEST
    mpath.write_bytes(mpath.read_bytes() + b" ")
    _expect_v04c("MANIFEST_HASH_MISMATCH", "manifest sidecar drift",
                 lambda: v04c.load_verified_facts(fp, man_dir))
    logger.info("v04c smoke (5) PASS: facts-integrity battery, 16 "
                "tamper branches refused with typed codes")

    # (6) C15: R1/NMSE-only mutations never alter the classification.
    sens = copy.deepcopy(base_facts)
    d1 = sens["per_arm"][1]["aggregates"]["delta_vs_lambda0"]
    d1["delta_psnr_r1"]["mean"] = -7.7
    d1["delta_psnr_r1"]["ci95"] = [-9.0, -6.0]
    d1["delta_nmse_u_r0"]["mean"] = 5.5
    d1["delta_nmse_u_r1"]["ci95"] = [4.0, 7.0]
    rows_a, sha_a, _ = _load("c15_base", base_facts)
    rows_b, sha_b, _ = _load("c15_sens", sens)
    if sha_a == sha_b:
        _fail("CLASSIFY_FIXTURE_MISMATCH",
              "C15: the sensitivity-mutated facts must hash differently")
    det_a = v04c.route_arm(rows_a[1])
    det_b = v04c.route_arm(rows_b[1])
    if det_a != det_b:
        _fail("CLASSIFY_FIXTURE_MISMATCH",
              "C15: R1/NMSE-only changes altered the classification "
              "detail")
    logger.info("v04c smoke (6) PASS: C15 sensitivity invariance "
                "(overall %s unchanged)", det_a["class"])

    # (7) C16: end-to-end through the real entry point, bitwise
    # double run, record schema + closed key set + semantic hash. The
    # routing record pins its input by PATH + sha (V03C parity), so the
    # input lives at a fixed scratch location: a stable input identity
    # makes the record bytes (and this suite's logs) reproducible
    # across invocations; the location is recreated fresh each run.
    c16_root = Path(tempfile.gettempdir()) / "seqref_v04c_c16_fixed"
    if c16_root.exists():
        shutil.rmtree(c16_root)
    (c16_root / "man").mkdir(parents=True)
    _manifest(c16_root / "man", ids)
    fp = c16_root / "v04_facts.json"
    v04e._write_with_sidecar(fp, _seal(base_facts))
    man_dir = c16_root / "man"
    cfg = {"facts": str(fp), "manifest_dir": str(man_dir),
           "code_commit": "fixture", "code_branch": "fixture"}
    out_a, out_b = tmp / "c16_out_a", tmp / "c16_out_b"
    if v04c.run({**cfg, "out_dir": str(out_a)}) != 0 \
            or v04c.run({**cfg, "out_dir": str(out_b)}) != 0:
        _fail("CLASSIFY_FIXTURE_MISMATCH",
              "C16: v04_classify.run returned nonzero")
    rec_a = (out_a / "v04_routing_record.json").read_bytes()
    rec_b = (out_b / "v04_routing_record.json").read_bytes()
    if rec_a != rec_b:
        _fail("CLASSIFY_FIXTURE_MISMATCH",
              "C16: repeated classification is not bit-identical")
    if (out_a / "v04_routing_record.json.sha256").read_text() \
            != (out_b / "v04_routing_record.json.sha256").read_text():
        _fail("CLASSIFY_FIXTURE_MISMATCH",
              "C16: sidecars diverge between repeated runs")
    record = json.loads(rec_a.decode("utf-8"))
    if record.get("schema") != v04c.ROUTING_SCHEMA:
        _fail("CLASSIFY_FIXTURE_MISMATCH",
              f"C16: record schema {record.get('schema')!r}")
    if set(record) != {"schema", "script", "spec", "plan",
                       "inherited_rules", "code", "inputs", "thresholds",
                       "per_arm", "overall", "interpretation",
                       "semantic_sha256", "semantic_scope"}:
        _fail("CLASSIFY_FIXTURE_MISMATCH",
              "C16: routing-record key set diverges from the closed "
              "schema (no wall-clock fields permitted)")
    if record["overall"]["class"] != "V04-A" \
            or record["overall"]["primary_arm"] != 1:
        _fail("CLASSIFY_FIXTURE_MISMATCH",
              "C16: overall must be V04-A on the locked parity arm 1")
    if record["interpretation"]["conclusion"] \
            != v04c.INTERPRETATION_MAP["V04-A"]:
        _fail("CLASSIFY_FIXTURE_MISMATCH",
              "C16: interpretation is not the registered 176 §4 text")
    if record["inputs"]["facts_sha256"] != hashlib.sha256(
            fp.read_bytes()).hexdigest():
        _fail("CLASSIFY_FIXTURE_MISMATCH",
              "C16: recorded facts sha != the sidecar-pinned input")
    sem = {k: v for k, v in record.items()
           if k not in v04c._SEMANTIC_EXCLUDE}
    if v04e._canonical_hash()(sem) != record["semantic_sha256"]:
        _fail("CLASSIFY_FIXTURE_MISMATCH",
              "C16: record semantic_sha256 is not self-consistent")
    logger.info("v04c smoke (7) PASS: C16 end-to-end entry-point run, "
                "bitwise-identical double run (record sha256 %s...)",
                hashlib.sha256(rec_a).hexdigest()[:12])
    logger.info("v04c routing smoke PASS: 7 check groups green "
                "(supplementary, non-registered)")


# -- V04T driver smoke (supplementary, non-registered) ----------------------
def v04t_driver_smoke() -> None:
    """2-step synthetic end-to-end driver run: fresh disposable V04M init,
    own generator + scratch dir, checkpoint write + sha + identity, ledger
    append with an interruption/resume pair, frozen selection rule,
    bitwise determinism across two repetitions. Own 2-step budget --
    EXEMPT from the locked 3258-step production assertion (safeguard C
    exemption). NOT a seventh experimental run; NOT lambda evidence."""
    lam = 0.5                     # smoke-local value, NOT lambda_2'
    outcomes = []
    for rep in (1, 2):
        tmp = Path(tempfile.mkdtemp(prefix=f"v04t_smoke{rep}_"))
        gen = torch.Generator().manual_seed(V04T_SMOKE_SEED)
        model = build_v04_model(init_seed=V04T_SMOKE_SEED)  # disposable
        opt = torch.optim.Adam(model.parameters(), lr=v04t.LEARNING_RATE)
        led_path = tmp / "v04_run_ledger.json"
        # attempt 1: one step, then simulated infrastructure interruption
        u, cond_in, mask, y_raw, amax, tgt_norm, maps = \
            _synthetic_rec_batch(gen)
        opt.zero_grad(set_to_none=True)
        loss1 = v04o.combined_loss(model, u, cond_in, mask, lam=lam,
                                   y_raw=y_raw, amax=amax,
                                   tgt_norm=tgt_norm, maps=maps)
        if not torch.isfinite(loss1):
            _fail("LOSS_CONTRACT_VIOLATION", "smoke step 1 non-finite")
        loss1.backward()
        opt.step()
        v04t.append_entry(led_path, _led_entry(
            1, 101, 1, "interrupted", steps=1,
            error="INFRASTRUCTURE_INTERRUPTION"))
        if v04t.next_attempt(v04t.load_ledger(led_path), 1, 101) != 2:
            _fail("RERUN_POLICY_VIOLATION", "smoke resume refused")
        # attempt 2: full restart (V03T policy), two steps to completion
        model = build_v04_model(init_seed=V04T_SMOKE_SEED)  # fresh restart
        opt = torch.optim.Adam(model.parameters(), lr=v04t.LEARNING_RATE)
        gen2 = torch.Generator().manual_seed(V04T_SMOKE_SEED)
        history, cks, val_hist = [], [], []
        for ep in range(V04T_SMOKE_STEPS):
            u, cond_in, mask, y_raw, amax, tgt_norm, maps = \
                _synthetic_rec_batch(gen2)
            opt.zero_grad(set_to_none=True)
            nll = v04o.density_only(model, u, cond_in, mask)
            rec = v04o.rec_term(model, cond_in, mask, y_raw, amax,
                                tgt_norm, maps)
            loss = nll + lam * rec
            if not torch.isfinite(loss):
                _fail("LOSS_CONTRACT_VIOLATION",
                      f"smoke attempt 2 epoch {ep + 1} non-finite")
            loss.backward()
            opt.step()
            history.append({"step": ep + 1, "epoch": ep,
                            "L_total": float(loss)})
            ck = tmp / v04t.checkpoint_name(1, 101, 2, ep + 1)
            torch.save(model.state_dict(), ck)
            entry = {"epoch": ep + 1, "file": ck.name,
                     "file_sha256": v04t._file_sha256(ck), "arm": 1,
                     "seed": 101, "attempt": 2, "lambda": lam,
                     "init_seed": 101, "val_NLL": float(nll),
                     "val_L_rec": float(rec), "L_val": float(loss)}
            v04t.assert_checkpoint_identity(entry, 1, 101, 2, lam)
            cks.append(entry)
            val_hist.append({"epoch": ep + 1, "L_val": float(loss)})
        if len(history) != V04T_SMOKE_STEPS:
            _fail("STEP_COUNT_MISMATCH",
                  f"smoke completed {len(history)} steps; its own budget "
                  f"is {V04T_SMOKE_STEPS}")
        v04t.append_entry(led_path, _led_entry(
            1, 101, 2, "completed", steps=V04T_SMOKE_STEPS))
        led = v04t.load_ledger(led_path)
        if [e["status"] for e in led["entries"]] != ["interrupted",
                                                     "completed"]:
            _fail("RERUN_POLICY_VIOLATION",
                  "smoke ledger must read interrupted -> completed")
        _expect_v04t("RERUN_POLICY_VIOLATION", "smoke restart refusal",
                     lambda: v04t.next_attempt(led, 1, 101))
        sel = v04t.select_checkpoint(val_hist, cks)
        outcomes.append(([h["L_total"] for h in history], sel["epoch"],
                         sel["file_sha256"]))
    if outcomes[0] != outcomes[1]:
        _fail("LOSS_CONTRACT_VIOLATION",
              "driver smoke not bitwise-deterministic across repetitions")
    logger.info("v04t driver smoke PASS (supplementary, non-registered): "
                "2-step run, checkpoint+identity, ledger resume pair, "
                "selection epoch %d, bitwise-deterministic; L_total %s",
                outcomes[0][1], ["%.6f" % v for v in outcomes[0][0]])


# -- f11: mask alternation ----------------------------------------------------
def f11_mask_alternation(model) -> None:
    if len(model.flow.layers) != V04M_N_LAYERS:
        _fail("MASK_CONTRACT_VIOLATION",
              f"{len(model.flow.layers)} layers != {V04M_N_LAYERS}")
    for i, layer in enumerate(model.flow.layers):
        if layer.d_in != V04M_D_IN or layer.d_in + layer.d_out != FLOW_DIM_REAL:
            _fail("MASK_CONTRACT_VIOLATION",
                  f"layer {i}: split {layer.d_in}/{layer.d_out} does not "
                  f"partition {FLOW_DIM_REAL}")
        if layer.flip != bool(i % 2):
            _fail("MASK_CONTRACT_VIOLATION",
                  f"layer {i}: flip={layer.flip} != parity {bool(i % 2)}")
        # transformed half: second half when flip=False (even i), first when odd
        x = torch.arange(FLOW_DIM_REAL, dtype=torch.float32).unsqueeze(0)
        h = torch.zeros(1, H_DIM)
        y, _ = layer(x, h)
        unchanged = y[0, :V04M_D_IN] if not layer.flip else y[0, V04M_D_IN:]
        expected = x[0, :V04M_D_IN] if not layer.flip else x[0, V04M_D_IN:]
        if not bool((unchanged == expected).all()):
            _fail("MASK_CONTRACT_VIOLATION",
                  f"layer {i}: identity half moved under the mask")
    logger.info("f11 PASS: 6 layers, contiguous half-split d_in=%d, "
                "alternating flip parity", V04M_D_IN)


# -- f12: invertibility round-trip (full model) -------------------------------
def f12_invertibility(model, gen) -> None:
    u, cond_in, mask = _synthetic_batch(gen, 4)
    z, _ = model.encode_free(u, cond_in, mask)
    u_hat = model.decode_scalars(z, cond_in, mask)
    err = (u_hat - u).abs().max().item()
    if not err <= F12_ROUNDTRIP_TOL:
        _fail("INVERTIBILITY_FAILURE",
              f"round-trip max abs err {err:.3e} > {F12_ROUNDTRIP_TOL:.1e}")
    logger.info("f12 PASS: round-trip max abs err %.3e <= %.0e", err,
                F12_ROUNDTRIP_TOL)


# -- f13: full-Jacobian log-det validation (8-dim fixture instance) -----------
def f13_jacobian_logdet() -> None:
    torch.manual_seed(F13_SEED)
    cond = Conditioner(in_channels=COND_IN_CHANNELS, width=64, h_dim=4,
                       use_v2=False, y_residual_alpha_init=0.0)
    fixture = CondRealNVP(dim=8, hidden=32, h_dim=4, conditioner=cond,
                          n_layers=2, use_film=True,
                          post_init_std=0.1)  # fixture-only nonzero init
    fixture = fixture.double()
    gen = torch.Generator().manual_seed(F13_SEED + 1)
    x = torch.randn(1, 8, generator=gen, dtype=torch.float64)
    h = torch.randn(1, 4, generator=gen, dtype=torch.float64)
    z, ldj = fixture.encode(x, h)
    if not torch.isfinite(ldj).all():
        _fail("LOGDET_MISMATCH", "analytic ldj non-finite on fixture")
    # complete 8x8 numerical Jacobian, central differences
    J = torch.zeros(8, 8, dtype=torch.float64)
    for j in range(8):
        dx = torch.zeros(1, 8, dtype=torch.float64)
        dx[0, j] = F13_FD_EPS
        zp, _ = fixture.encode(x + dx, h)
        zm, _ = fixture.encode(x - dx, h)
        J[:, j] = ((zp - zm) / (2 * F13_FD_EPS))[0]
    sign, logabsdet = torch.linalg.slogdet(J)
    if not torch.isfinite(logabsdet) or int(sign.item()) not in (-1, 1):
        _fail("LOGDET_MISMATCH",
              f"numerical Jacobian singular/non-finite (sign={sign.item()}, "
              f"logabsdet={logabsdet.item()})")
    ana = float(ldj[0].item())
    num = float(logabsdet.item())
    dev = abs(num - ana) / max(abs(ana), F13_DENOM_FLOOR)
    if not dev <= F13_REL_TOL:
        _fail("LOGDET_MISMATCH",
              f"normalised log-det deviation {dev:.3e} > {F13_REL_TOL:.0e} "
              f"(analytic {ana:.6f}, numerical {num:.6f})")
    logger.info("f13 PASS: 8x8 FD slogdet %.6f vs analytic %.6f, "
                "normalised dev %.2e <= %.0e", num, ana, dev, F13_REL_TOL)


# -- f14: numerical conditioning at init (PRELIMINARY, synthetic batches) -----
def f14_numerical_conditioning(model, gen) -> None:
    for b in range(F14_BATCHES):
        u, cond_in, mask = _synthetic_batch(gen, F14_BATCH_SIZE)
        h = model.condition(cond_in, mask)
        x = u.to(torch.float32)
        for i, layer in enumerate(model.flow.layers):
            x1, _ = layer._split(x)
            raw, s, t = _post_outputs(layer, x1, h)
            if not torch.isfinite(raw).all() or not torch.isfinite(s).all() \
                    or not torch.isfinite(t).all():
                _fail("CONDITIONING_FAILURE",
                      f"layer {i} batch {b}: non-finite raw/s/t outputs")
            if bool((s.abs() > V04M_S_MAX).any()):
                _fail("CONDITIONING_FAILURE",
                      f"layer {i} batch {b}: |s| exceeds s_max={V04M_S_MAX} "
                      "(tanh clamp broken)")
            if layer.last_s_clamp_frac != 0.0:
                _fail("CONDITIONING_FAILURE",
                      f"layer {i} batch {b}: last_s_clamp_frac="
                      f"{layer.last_s_clamp_frac} != 0.0 exactly at init")
            x, _ = layer(x, h)
    logger.info("f14 PASS (PRELIMINARY, synthetic): raw outputs finite, "
                "|s| <= %.1f, saturation exactly 0.0 at init over %d batches",
                V04M_S_MAX, F14_BATCHES)


# -- f15: conditioning-path connectivity (AMENDED prong set, article 180) -----
# Amended per SEQREF-MRI-V04AMEND-A1 v1.0 FROZEN (LudWiki article 180), which
# supersedes 178 §5 f15 prongs (i), (ii), (iv) and the §7(f) reference.
def _rel_abs_change(ref: torch.Tensor, other: torch.Tensor):
    """A1 measurement convention: mean |d|/max(|ref|, 1e-12) + mean |d|."""
    d = (ref - other).abs()
    rel = (d / ref.abs().clamp_min(F15_DENOM_FLOOR)).mean().item()
    return rel, d.mean().item()


def _responds(rel: float, ab: float) -> bool:
    return rel > F15_REL_EPS and ab > F15_ABS_FLOOR


def _silent(rel: float, ab: float) -> bool:
    return rel <= F15_REL_EPS and ab <= F15_ABS_FLOOR


def _capture_mid_inputs(model, u, cond_in, mask):
    """FiLM-modulated activations gamma(h)*pre + beta(h) (= mid inputs)."""
    captured, handles = [], []
    for layer in model.flow.layers:
        handles.append(layer.mid.register_forward_hook(
            lambda mod, inp, out: captured.append(inp[0].detach())))
    try:
        with torch.no_grad():
            model.encode_free(u, cond_in, mask)
    finally:
        for hd in handles:
            hd.remove()
    if len(captured) != V04M_N_LAYERS:
        _fail("CONDITIONING_DEAD",
              f"mid-input hooks captured {len(captured)} layers != "
              f"{V04M_N_LAYERS}")
    return torch.cat([c.flatten(1) for c in captured], dim=1)


def _capture_film_hidden(model, u, cond_in, mask):
    """FiLM-internal pre-output activations (before the zero-init final affine)."""
    captured, handles = [], []
    for layer in model.flow.layers:
        handles.append(layer.film.mlp[-2].register_forward_hook(
            lambda mod, inp, out: captured.append(out.detach())))
    try:
        with torch.no_grad():
            model.encode_free(u, cond_in, mask)
    finally:
        for hd in handles:
            hd.remove()
    if len(captured) != V04M_N_LAYERS:
        _fail("CONDITIONING_DEAD",
              f"film-hidden hooks captured {len(captured)} layers != "
              f"{V04M_N_LAYERS}")
    return torch.cat([c.flatten(1) for c in captured], dim=1)


def _post_grads(model, u, cond_in, mask):
    """One fresh backward pass; cloned post.weight grads; model left clean."""
    model.zero_grad(set_to_none=True)
    lp = model.log_prob_free(u, cond_in, mask)
    if not torch.isfinite(lp).all():
        _fail("CONDITIONING_DEAD", "non-finite log_prob in gradient measurement")
    (-lp.mean()).backward()
    grads = []
    for i, layer in enumerate(model.flow.layers):
        gi = layer.post.weight.grad
        if gi is None or not torch.isfinite(gi).all():
            _fail("CONDITIONING_DEAD",
                  f"layer {i}: post.weight grad missing/non-finite")
        grads.append(gi.clone())
    model.zero_grad(set_to_none=True)
    return grads


def _assert_clone_equal(clone, ref, tag: str) -> None:
    for (n1, p1), (n2, p2) in zip(clone.named_parameters(),
                                  ref.named_parameters()):
        if n1 != n2 or not torch.equal(p1, p2):
            _fail("CONDITIONING_DEAD",
                  f"{tag}: clone parameter mismatch at {n1}")


def _clone_unblock(clone, u, cond_in, mask) -> None:
    """K=3 Adam steps (lr 1e-4, PyTorch defaults) on the fixture batch."""
    clone.train()
    opt = torch.optim.Adam(
        [p for p in clone.parameters() if p.requires_grad], lr=F15_CLONE_LR)
    for _ in range(F15_CLONE_STEPS):
        opt.zero_grad(set_to_none=True)
        loss = -clone.log_prob_free(u, cond_in, mask).mean()
        if not torch.isfinite(loss):
            _fail("CONDITIONING_DEAD",
                  "non-finite clone loss during unblocking")
        loss.backward()
        opt.step()
    clone.eval()


def f15_conditioning_connectivity(model) -> None:
    model.eval()
    # A1 wiring obligation: conditioning must enter the trunk only via FiLM,
    # otherwise the (vi') zero-expectation is invalid.
    for i, layer in enumerate(model.flow.layers):
        if not (getattr(layer, "use_film", False)
                and hasattr(layer, "film")
                and getattr(layer, "concat", None) is None):
            _fail("CONDITIONING_DEAD",
                  f"layer {i}: conditioning path not FiLM-exclusive "
                  "(negative-control assumption invalid)")
    for sub in F15_SUBSEEDS:
        gen_s = torch.Generator().manual_seed(sub)
        u1, cond_a, mask_a = _synthetic_batch(gen_s, 8)
        _, cond_b, mask_b = _synthetic_batch(gen_s, 8)

        # prong (i'): conditioner-output responsiveness
        rel1, ab1 = _rel_abs_change(model.condition(cond_a, mask_a),
                                    model.condition(cond_b, mask_b))
        if not _responds(rel1, ab1):
            _fail("CONDITIONING_DEAD",
                  f"sub-seed {sub} prong (i'): conditioner output h "
                  f"unresponsive (rel {rel1:.3e}, abs {ab1:.3e})")

        # prong (ii'): FiLM-internal pre-output responsiveness
        rel2, ab2 = _rel_abs_change(
            _capture_film_hidden(model, u1, cond_a, mask_a),
            _capture_film_hidden(model, u1, cond_b, mask_b))
        if not _responds(rel2, ab2):
            _fail("CONDITIONING_DEAD",
                  f"sub-seed {sub} prong (ii'): FiLM-internal activations "
                  f"unresponsive (rel {rel2:.3e}, abs {ab2:.3e})")

        # prong (iii): gradient path (verbatim frozen prong)
        g0 = _post_grads(model, u1, cond_a, mask_a)
        if not any(float(gi.abs().sum()) > 0.0 for gi in g0):
            _fail("CONDITIONING_DEAD",
                  f"sub-seed {sub} prong (iii): dL/dpost.weight exactly zero "
                  "on every layer")

        # prong (iv'): unblocking-connectivity on a disposable clone
        clone = copy.deepcopy(model)
        _assert_clone_equal(clone, model, "prong (iv')")
        _clone_unblock(clone, u1, cond_a, mask_a)
        rel_a, ab_a = _rel_abs_change(
            _capture_mid_inputs(clone, u1, cond_a, mask_a),
            _capture_mid_inputs(clone, u1, cond_b, mask_b))
        if not _responds(rel_a, ab_a):
            _fail("CONDITIONING_DEAD",
                  f"sub-seed {sub} prong (iv')(a): FiLM-modulated activations "
                  f"unresponsive after {F15_CLONE_STEPS} steps "
                  f"(rel {rel_a:.3e}, abs {ab_a:.3e})")
        ga = torch.cat([x.flatten()
                        for x in _post_grads(clone, u1, cond_a, mask_a)])
        gb = torch.cat([x.flatten()
                        for x in _post_grads(clone, u1, cond_b, mask_b)])
        rel_b, ab_b = _rel_abs_change(ga, gb)
        if not _responds(rel_b, ab_b):
            _fail("CONDITIONING_DEAD",
                  f"sub-seed {sub} prong (iv')(b): dL/dpost.weight "
                  f"conditioning-independent after unblocking "
                  f"(rel {rel_b:.3e}, abs {ab_b:.3e})")

        # prong (v'): shuffled-conditioning control on the unblocked clone
        perm = torch.randperm(8, generator=gen_s)
        rel_s, ab_s = _rel_abs_change(
            _capture_mid_inputs(clone, u1, cond_a, mask_a),
            _capture_mid_inputs(clone, u1, cond_a[perm], mask_a[perm]))
        if not _responds(rel_s, ab_s):
            _fail("CONDITIONING_DEAD",
                  f"sub-seed {sub} prong (v'): shuffled-conditioning control "
                  f"shows no change (rel {rel_s:.3e}, abs {ab_s:.3e})")

        # prong (vi'): FiLM-frozen negative control (expect exact silence)
        nc = copy.deepcopy(model)
        _assert_clone_equal(nc, model, "prong (vi')")
        for layer in nc.flow.layers:
            for prm in layer.film.parameters():
                prm.requires_grad_(False)
        _clone_unblock(nc, u1, cond_a, mask_a)
        rel_n, ab_n = _rel_abs_change(
            _capture_mid_inputs(nc, u1, cond_a, mask_a),
            _capture_mid_inputs(nc, u1, cond_b, mask_b))
        if not _silent(rel_n, ab_n):
            _fail("CONDITIONING_DEAD",
                  f"sub-seed {sub} prong (vi'): FiLM-frozen control responds "
                  f"(rel {rel_n:.3e}, abs {ab_n:.3e}) -- conditioning enters "
                  "outside the FiLM path")
        nga = torch.cat([x.flatten()
                         for x in _post_grads(nc, u1, cond_a, mask_a)])
        ngb = torch.cat([x.flatten()
                         for x in _post_grads(nc, u1, cond_b, mask_b)])
        rel_g, ab_g = _rel_abs_change(nga, ngb)
        if not _silent(rel_g, ab_g):
            _fail("CONDITIONING_DEAD",
                  f"sub-seed {sub} prong (vi'): FiLM-frozen control gradient "
                  f"responds (rel {rel_g:.3e}, abs {ab_g:.3e})")
        logger.info("f15 sub-seed %d PASS: (i') %.2e (ii') %.2e (iii) nonzero "
                    "(iv') act %.2e grad %.2e (v') %.2e (vi') silent %.1e",
                    sub, rel1, rel2, rel_a, rel_b, rel_s,
                    max(rel_n, rel_g))
    logger.info("f15 PASS (amended, article 180): all prongs x %d sub-seeds",
                len(F15_SUBSEEDS))


# -- f16: forward/inverse log-det cancellation ---------------------------------
def f16_logdet_cancellation(model, gen) -> None:
    u, cond_in, mask = _synthetic_batch(gen, 4)
    h = model.condition(cond_in, mask)
    s_fwd, s_inv = [], []
    handles = []
    for layer in model.flow.layers:
        def make(store, s_max=layer.s_max):
            def hook(mod, inp, out):
                s = s_max * torch.tanh(out.chunk(2, dim=-1)[0])
                store.append(s.detach().sum(dim=-1))
            return hook
        handles.append(layer.post.register_forward_hook(make(s_fwd)))
    try:
        _, ldj_fwd = model.encode_free(u, cond_in, mask)
    finally:
        for hd in handles:
            hd.remove()
    z, _ = model.encode_free(u, cond_in, mask)
    handles = []
    for layer in model.flow.layers:
        def make2(store, s_max=layer.s_max):
            def hook(mod, inp, out):
                s = s_max * torch.tanh(out.chunk(2, dim=-1)[0])
                store.append(s.detach().sum(dim=-1))
            return hook
        handles.append(layer.post.register_forward_hook(make2(s_inv)))
    try:
        model.decode_scalars(z, cond_in, mask)
    finally:
        for hd in handles:
            hd.remove()
    if len(s_fwd) != V04M_N_LAYERS or len(s_inv) != V04M_N_LAYERS:
        _fail("LOGDET_MISMATCH",
              f"hook capture incomplete: fwd {len(s_fwd)}, inv {len(s_inv)}")
    # inverse log-det = -sum(s) over layers applied in reverse
    ldj_inv = -torch.stack(list(reversed(s_inv))).sum(dim=0)
    resid = (ldj_fwd + ldj_inv).abs().max().item()
    if not resid <= F16_CANCEL_TOL:
        _fail("LOGDET_MISMATCH",
              f"fwd/inv log-det cancellation residual {resid:.3e} > "
              f"{F16_CANCEL_TOL:.0e}")
    logger.info("f16 PASS: fwd/inv cancellation residual %.3e <= %.0e",
                resid, F16_CANCEL_TOL)


# ---------------------------------------------------------------------------
# V04X cross-flow: f17 (registered, 178 §5) + supplementary X01-X13 smoke
# ---------------------------------------------------------------------------

def _expect_v04x(code: str, name: str, fn) -> None:
    try:
        fn()
    except (v04x.V04XError, v04e.V04EError, v04c.V04CError) as exc:
        if exc.code != code:
            _fail("CROSSFLOW_FIXTURE_MISMATCH",
                  f"{name}: expected {code}, got {exc.code}: {exc}")
        return
    except v02m.V02Error as exc:
        if not str(exc).startswith(code):
            _fail("CROSSFLOW_FIXTURE_MISMATCH",
                  f"{name}: expected {code}, got V02Error: {exc}")
        return
    _fail("CROSSFLOW_FIXTURE_MISMATCH",
          f"{name}: accepted; the refusal is dead")


def _v04x_ids(n: int = None) -> list:
    n = v02m.N_HOLDOUT_VOLUMES if n is None else n
    return [{"file": f"vol{i:03d}.h5", "slice_index": i}
            for i in range(n)]


def _v04x_metric_rows(vec) -> list:
    """A flat vector broadcasts to identical per-seed copies (the
    ensemble mean is then exact); a list of 3 vectors gives explicit
    per-seed values (the ensemble-mean invariance test)."""
    if vec and isinstance(vec[0], list):
        return [list(v) for v in vec]
    return [list(vec) for _ in range(3)]


def _v04x_v03_facts(ids: list, arm_metrics: dict) -> dict:
    """Synthetic V03 facts through the PRODUCTION v03_eval._build_facts:
    4 arms x 3 seeds completed, deterministic vectors, format-valid
    synthetic shas. No model, no data."""
    runs, selections, per_metrics = {}, {}, {}
    for si, seed in enumerate(v03t.SEEDS):
        for arm in v03t.ARMS:
            pair = (arm, seed)
            runs[pair] = {"status": "completed", "attempts": 1,
                          "error_code": None, "wall_clock_s": 1.0}
            selections[pair] = {"file": "f.pt",
                                "file_sha256": "a" * 64,
                                "epoch": 2, "L_val": 1.0}
            row = {"identities": ids}
            for k, vec in arm_metrics[arm].items():
                row[k] = _v04x_metric_rows(vec)[si]
            per_metrics[pair] = row
    cfg = {"code_commit": "fixture", "code_branch": "fixture",
           "probe_record": "probe.json", "gate_record": "gate.json"}
    state = {"probe": {"lambdas": {"0.1": 0.1, "0.5": 0.5,
                                   "0.9": 0.9}},
             "runs": runs, "selections": selections,
             "per_metrics": per_metrics}
    return v03e._build_facts(cfg, state)


def _v04x_v04_facts(ids: list, arm_metrics: dict) -> dict:
    """Synthetic V04 facts through the PRODUCTION v04_eval._build_facts
    (same construction as _synthetic_facts, explicit vectors)."""
    runs, selections, per_metrics = {}, {}, {}
    for si, seed in enumerate(v04t.SEEDS):
        for arm in v04t.ARMS:
            pair = (arm, seed)
            runs[pair] = {"status": "completed", "attempts": 1,
                          "error_code": None, "wall_clock_s": 1.0,
                          "record_file": v04t.run_record_name(
                              arm, seed, 1),
                          "record_sha256": hashlib.sha256(
                              f"rec-{arm}-{seed}".encode()).hexdigest()}
            selections[pair] = {
                "file": v04t.checkpoint_name(arm, seed, 1, 2),
                "file_sha256": hashlib.sha256(
                    f"ck-{arm}-{seed}".encode()).hexdigest(),
                "epoch": 2, "L_val": 1.0}
            row = {"identities": ids}
            for k, vec in arm_metrics[arm].items():
                row[k] = _v04x_metric_rows(vec)[si]
            per_metrics[pair] = row
    cfg = {"code_commit": "fixture", "code_branch": "fixture",
           "probe_record": "probe.json", "gate_record": "gate.json",
           "v03_facts": "v03_facts.json", "manifest_dir": "man",
           "p4_stats2": "p4.json", "p0s_facts": "p0s.json"}
    state = {"probe": {"lambda_prime": 2.0},
             "runs": runs, "selections": selections,
             "per_metrics": per_metrics,
             "bank_sha256": v04e.Z_DIAG_BANK_SHA256,
             "bank_manifest_sha256": "ab" * 32,
             "parent_shas": {k: hashlib.sha256(k.encode()).hexdigest()
                             for k in ("holdout_manifest", "p4_stats2",
                                       "p0s_facts", "v03_facts",
                                       "probe_record", "gate_record")}}
    return v04e._build_facts(cfg, state)


def _v04x_seal_v04(facts: dict) -> dict:
    """The production semantic seal (v04_eval run() adds it after
    _build_facts; the fixture mirrors the artifact, not the raw
    builder output)."""
    sealed = copy.deepcopy(facts)
    semantic = {k: v for k, v in sealed.items()
                if k not in v04e._SEMANTIC_EXCLUDE}
    sealed["semantic_sha256"] = v04e._canonical_hash()(semantic)
    sealed["semantic_scope"] = {
        "included_keys": sorted(semantic.keys()),
        "excluded": list(v04e._SEMANTIC_EXCLUDE),
        "canonicalisation": "preflight_io.canonical_hash"}
    return sealed


def _v04x_green_metrics(n_v03: int, n_v04: int) -> tuple:
    """Binary-exact per-volume vectors: Delta_V03 PSNR R0 = +0.5 dB,
    Delta_V04 = +1.25 dB (DeltaDelta = +0.75 dB); Delta_V03 NLL =
    +0.5 nats, Delta_V04 = +0.25 nats (higher-is-better; DeltaDelta =
    -0.25)."""
    def base(psnr, nll, n):
        return {"nll": [nll] * n, "psnr_r0": [psnr] * n,
                "psnr_r1": [psnr + 1.0] * n, "nmse_u_r0": [0.125] * n,
                "nmse_u_r1": [0.0625] * n}
    v03 = {0: base(30.0, 100.0, n_v03),
           1: base(30.25, 100.0, n_v03),
           2: base(30.5, 99.5, n_v03),
           3: base(30.75, 100.0, n_v03)}
    v04 = {0: base(28.0, 100.0, n_v04),
           1: base(29.25, 99.75, n_v04)}
    return v03, v04


def _v04x_chain_state(manifest_sha: str, **tampers) -> dict:
    """A green governance attestation
    (seqref-v04-crossflow-chain-state/1); tampers are dotted-path
    overrides for the refusal branches."""
    state = {
        "schema": v04x.CHAIN_STATE_SCHEMA,
        "activation": {"v03_overall": "decisive_null",
                       "activated_port": "realnvp",
                       "nice_status": "parked"},
        "chains": {
            "v03": {"spec": {"article": 163, "status": "FROZEN"},
                    "plan": {"article": 164, "status": "FROZEN"},
                    "exec": {"article": 169, "status": "registered"},
                    "result": {"article": 170, "status":
                               "registered"}},
            "v04": {"spec": {"article": 176, "status": "FROZEN"},
                    "plan": {"article": 178, "status": "FROZEN"},
                    "exec": {"article": 179, "status": "registered"},
                    "result": {"article": 181, "status":
                               "registered"}}},
        "deviations": [{"id": "affine-coupling-replaces-splines",
                        "scope": "RealNVP (V04)",
                        "registered_in": {"article": 173,
                                          "section": "8"},
                        "timing": "pre-results"}],
        "statistics": {"delta_delta_lock": {"article": 176,
                                            "section": "13"},
                       "frozen_timing": "pre-results"},
        "holdout_tuning": "none",
        "shared": {"data_range": 1.0,
                   "likelihood": v04x.LIKELIHOOD_CONTRACT,
                   "holdout_manifest_sha256": manifest_sha}}
    for path, value in tampers.items():
        node = state
        keys = path.split(".")
        for k in keys[:-1]:
            node = node[int(k)] if isinstance(node, list) else node[k]
        last = keys[-1]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
    return state


def _v04x_workspace(root: Path, *, v03_ids=None, v04_ids=None,
                    v03_metrics=None, v04_metrics=None,
                    chain_tampers=None) -> dict:
    """A deterministic fixed-root workspace (rmtree + recreate): the
    pinned holdout manifest, synthetic V03/V04 facts and a chain-state
    attestation, all with sidecars. Records pin their inputs by path,
    so the fixed root keeps suite runs cross-invocation identical."""
    if root.exists():
        shutil.rmtree(root)
    man_dir = root / "man"
    man_dir.mkdir(parents=True)
    ids = _v04x_ids()
    entries = [{"file": e["file"], "slice_index": int(e["slice_index"]),
                "volume_ordinal": i} for i, e in enumerate(ids)]
    man = {"schema": v02m.SCHEMA, "kind": "holdout",
           "selection": {"rule": "synthetic fixture identities",
                         "split": "val", "mode": "eval"},
           "n_volumes": len(entries), "entries": entries}
    man["manifest_sha256"] = v02m.manifest_sha256(man)
    v04e._write_with_sidecar(man_dir / v04x.HOLDOUT_MANIFEST, man)
    man_sha = hashlib.sha256(
        (man_dir / v04x.HOLDOUT_MANIFEST).read_bytes()).hexdigest()
    v03_ids = ids if v03_ids is None else v03_ids
    v04_ids = ids if v04_ids is None else v04_ids
    gm03, gm04 = _v04x_green_metrics(len(v03_ids), len(v04_ids))
    cfg = {"v03_facts": str(root / "v03_facts.json"),
           "v04_facts": str(root / "v04_facts.json"),
           "manifest_dir": str(man_dir),
           "chain_state": str(root / "chain_state.json"),
           "out": str(root / "out"),
           "commit": "fixture", "branch": "fixture"}
    v03_sha = v04x._write_with_sidecar(
        Path(cfg["v03_facts"]),
        _v04x_v03_facts(v03_ids,
                        gm03 if v03_metrics is None else v03_metrics))
    v04_sha = v04x._write_with_sidecar(
        Path(cfg["v04_facts"]),
        _v04x_seal_v04(_v04x_v04_facts(
            v04_ids, gm04 if v04_metrics is None else v04_metrics)))
    v04x._write_with_sidecar(
        Path(cfg["chain_state"]),
        _v04x_chain_state(man_sha, **(chain_tampers or {})))
    return {"cfg": cfg, "v03_sha": v03_sha, "v04_sha": v04_sha,
            "manifest_sha": man_sha, "ids": ids}


def _v04x_record(cfg: dict, name: str) -> dict:
    return json.loads((Path(cfg["out"]) / name).read_text())


def _v04x_assert_gate_failure(rec: dict, failed_items: list,
                              name: str) -> None:
    if rec["readiness"]["verdict"] != v04x.GATE_FAILURE:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              f"{name}: verdict {rec['readiness']['verdict']!r} != "
              f"{v04x.GATE_FAILURE}")
    got = [i["item"] for i in rec["readiness"]["items"]
           if i["status"] != "PASS"]
    if got != failed_items:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              f"{name}: failed items {got} != {failed_items}")
    if "delta_delta" in rec:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              f"{name}: a confirmatory DeltaDelta leaked into a "
              f"gate-failed record")
    if not rec["claim"].startswith("exploratory at most"):
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              f"{name}: claim {rec['claim']!r} is not "
              f"exploratory-at-most")


def f17_crossflow_readiness_refusal() -> None:
    """178 §5 f17: V04X invoked with a deliberately failed 173 §8.1
    item (the frozen example: a mismatched evaluation-case manifest,
    realised as a reordered V03 case set) refuses the registered
    claim -- CROSSFLOW_GATE_FAILURE, exploratory label at most, no
    confirmatory DeltaDelta; the within-architecture V04 result is
    unaffected."""
    root = Path(tempfile.gettempdir()) / "seqref_v04x_f17_fixed"
    ws = _v04x_workspace(root,
                         v03_ids=list(reversed(_v04x_ids())))
    rc = v04x.run_audit(ws["cfg"], v03_facts_sha=ws["v03_sha"])
    if rc != 2:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              f"f17: audit returned {rc}, want 2")
    rec = _v04x_record(ws["cfg"], v04x.AUDIT_RECORD)
    _v04x_assert_gate_failure(rec, ["shared_settings"], "f17")
    shared = next(i for i in rec["readiness"]["items"]
                  if i["item"] == "shared_settings")
    if not any("CROSSFLOW_CASE_MISMATCH" in e
               for e in shared["evidence"]):
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              "f17: the case-mismatch refusal is not in the "
              "shared_settings evidence")
    if "unaffected" not in rec["claim"]:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              "f17: the claim must state that the within-architecture "
              "V04 result is unaffected (178 SS12a)")
    v04x._verify_sidecar_file(
        Path(ws["cfg"]["out"]) / v04x.AUDIT_RECORD)
    logger.info("[%s] f17 PASS: cross-flow readiness refusal "
                "(CROSSFLOW_GATE_FAILURE, exploratory at most, no "
                "DeltaDelta)", __abbr__)


def v04x_crossflow_smoke() -> None:
    """Supplementary (non-registered) V04X smoke, advisor plan
    X01-X13: preflight/audit lifecycle on synthetic artifacts, every
    refusal branch, exact DeltaDelta arithmetic, determinism."""
    root = Path(tempfile.gettempdir()) / "seqref_v04x_smoke_fixed"
    ids = _v04x_ids()

    # X01 green preflight: non-authoritative record, 4/4 PASS.
    ws = _v04x_workspace(root / "x01")
    rc = v04x.run_preflight(ws["cfg"], v03_facts_sha=ws["v03_sha"])
    if rc != 0:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              f"X01: preflight returned {rc}, want 0")
    rec = _v04x_record(ws["cfg"], v04x.PREFLIGHT_RECORD)
    if rec["schema"] != v04x.PREFLIGHT_SCHEMA \
            or rec["status"] != "non_authoritative" \
            or "NOT 173" not in rec["notice"] \
            or rec["result"] != "PASS" \
            or [c["check"] for c in rec["checks"]] != [
                "v03_facts_pin", "v04_facts_integrity",
                "manifest_identity", "case_alignment"] \
            or any(c["status"] != "PASS" for c in rec["checks"]):
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              f"X01: malformed green preflight record: {rec}")
    if v04x._semantic_sha(rec) != rec["semantic_sha256"]:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              "X01: preflight semantic_sha256 not self-consistent")
    v04x._verify_sidecar_file(
        Path(ws["cfg"]["out"]) / v04x.PREFLIGHT_RECORD)
    logger.info("[%s] v04x smoke X01 PASS: green preflight "
                "(non-authoritative)", __abbr__)

    # X02 V03 sha-pin refusal (preflight records the FAIL, exit 2).
    rc = v04x.run_preflight(ws["cfg"], v03_facts_sha="0" * 64)
    rec = _v04x_record(ws["cfg"], v04x.PREFLIGHT_RECORD)
    if rc != 2 or rec["result"] != "FAIL" \
            or rec["checks"][0]["status"] != "FAIL" \
            or rec["checks"][0]["code"] != "V03_FACTS_PIN_MISMATCH" \
            or rec["checks"][-1]["status"] != "not_run":
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              f"X02: pin refusal not recorded as designed: "
              f"{rec['checks']}")
    logger.info("[%s] v04x smoke X02 PASS: V03 pin refusal",
                __abbr__)

    # X03 preflight case mismatch (reordered V03 case set).
    ws = _v04x_workspace(root / "x03",
                         v03_ids=list(reversed(ids)))
    rc = v04x.run_preflight(ws["cfg"], v03_facts_sha=ws["v03_sha"])
    rec = _v04x_record(ws["cfg"], v04x.PREFLIGHT_RECORD)
    if rc != 2 or rec["result"] != "FAIL" \
            or rec["checks"][-1]["status"] != "FAIL" \
            or rec["checks"][-1]["code"] != "CROSSFLOW_CASE_MISMATCH":
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              f"X03: case mismatch not refused: {rec['checks']}")
    logger.info("[%s] v04x smoke X03 PASS: preflight case mismatch",
                __abbr__)

    # X04 identity defects (missing / duplicated / reordered): refused
    # at the audit without canonicalisation -> gate failure.
    dup = copy.deepcopy(ids)
    dup[5] = dict(dup[6])
    swap = copy.deepcopy(ids)
    swap[3], swap[7] = swap[7], swap[3]
    for tag, bad in (("missing", ids[:-1]), ("duplicated", dup),
                     ("reordered", swap)):
        ws = _v04x_workspace(root / f"x04_{tag}", v03_ids=bad)
        rc = v04x.run_audit(ws["cfg"], v03_facts_sha=ws["v03_sha"])
        rec = _v04x_record(ws["cfg"], v04x.AUDIT_RECORD)
        if rc != 2:
            _fail("CROSSFLOW_FIXTURE_MISMATCH",
                  f"X04 {tag}: audit returned {rc}, want 2")
        _v04x_assert_gate_failure(rec, ["shared_settings"],
                                  f"X04 {tag}")
    logger.info("[%s] v04x smoke X04 PASS: identity defects refused "
                "(missing/duplicated/reordered, no canonicalisation)",
                __abbr__)

    # X05 NLL comparability (173 SS7): attested-likelihood tamper and
    # data_range tamper (re-sealed) both fail the shared-settings item.
    ws = _v04x_workspace(
        root / "x05_likelihood",
        chain_tampers={"shared.likelihood":
                       "per-slice NLL in bits (wrong units)"})
    rc = v04x.run_audit(ws["cfg"], v03_facts_sha=ws["v03_sha"])
    rec = _v04x_record(ws["cfg"], v04x.AUDIT_RECORD)
    if rc != 2:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              "X05 likelihood: audit accepted a units tamper")
    _v04x_assert_gate_failure(rec, ["shared_settings"],
                              "X05 likelihood")
    ws = _v04x_workspace(root / "x05_range")
    facts = json.loads(Path(ws["cfg"]["v04_facts"]).read_text())
    facts["evaluation"]["data_range"] = 2.0
    v04e._write_with_sidecar(Path(ws["cfg"]["v04_facts"]),
                             _v04x_seal_v04(facts))
    rc = v04x.run_audit(ws["cfg"], v03_facts_sha=ws["v03_sha"])
    rec = _v04x_record(ws["cfg"], v04x.AUDIT_RECORD)
    if rc != 2:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              "X05 data_range: audit accepted a data_range tamper")
    _v04x_assert_gate_failure(rec, ["shared_settings"],
                              "X05 data_range")
    logger.info("[%s] v04x smoke X05 PASS: NLL comparability refusals",
                __abbr__)

    # X06 missing deviation registration (173 SS8).
    ws = _v04x_workspace(root / "x06",
                         chain_tampers={"deviations": []})
    rc = v04x.run_audit(ws["cfg"], v03_facts_sha=ws["v03_sha"])
    rec = _v04x_record(ws["cfg"], v04x.AUDIT_RECORD)
    if rc != 2:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              "X06: audit accepted an empty deviations register")
    _v04x_assert_gate_failure(rec, ["deviations_registered"], "X06")
    rec_x06 = rec
    logger.info("[%s] v04x smoke X06 PASS: missing deviation "
                "registration refused", __abbr__)

    # X07 V04R not registered (178 SS12a lifecycle gate).
    ws = _v04x_workspace(
        root / "x07",
        chain_tampers={"chains.v04.result.status": "draft"})
    rc = v04x.run_audit(ws["cfg"], v03_facts_sha=ws["v03_sha"])
    rec = _v04x_record(ws["cfg"], v04x.AUDIT_RECORD)
    if rc != 2:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              "X07: audit passed without a registered V04R")
    _v04x_assert_gate_failure(rec, ["evidence_chains"], "X07")
    logger.info("[%s] v04x smoke X07 PASS: V04R-not-registered "
                "refusal", __abbr__)

    # X08 + X09 all-green audit PASS with exact DeltaDelta arithmetic
    # (binary-exact vectors): DeltaDelta PSNR R0 = +0.75 dB, DeltaDelta
    # NLL = -0.25 nats (higher-is-better), degenerate CIs.
    ws = _v04x_workspace(root / "x08")
    rc = v04x.run_audit(ws["cfg"], v03_facts_sha=ws["v03_sha"])
    if rc != 0:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              f"X08: green audit returned {rc}, want 0")
    rec = _v04x_record(ws["cfg"], v04x.AUDIT_RECORD)
    if rec["readiness"]["verdict"] != "PASS" \
            or len(rec["readiness"]["items"]) != 6 \
            or any(i["status"] != "PASS"
                   for i in rec["readiness"]["items"]) \
            or rec["claim"] != "registered cross-flow comparison":
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              f"X08: green readiness malformed: {rec['readiness']}")
    dd = rec["delta_delta"]
    if dd["n_volumes"] != v04x.N_VOLUMES \
            or dd["arms"] != {"v03_anchor": 0, "v03_parity": 2,
                              "v04_anchor": 0, "v04_parity": 1} \
            or dd["psnr_r0"]["delta_v03_mean"] != 0.5 \
            or dd["psnr_r0"]["delta_v04_mean"] != 1.25 \
            or dd["psnr_r0"]["dd"]["mean"] != 0.75 \
            or dd["psnr_r0"]["dd"]["ci95"] != [0.75, 0.75] \
            or dd["psnr_r0"]["dd_ci95_excludes_zero"] is not True \
            or dd["nll"]["delta_v03_mean"] != 0.5 \
            or dd["nll"]["delta_v04_mean"] != 0.25 \
            or dd["nll"]["dd"]["mean"] != -0.25 \
            or dd["nll"]["dd"]["ci95"] != [-0.25, -0.25] \
            or dd["nll"]["dd_ci95_excludes_zero"] is not True:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              f"X09: exact DeltaDelta arithmetic broken: {dd}")
    if dd["bootstrap"]["seed"] != v04e.BOOTSTRAP_SEED \
            or dd["bootstrap"]["B"] != v04e.BOOTSTRAP_B \
            or dd["bootstrap"]["method"] != "percentile" \
            or "v04_eval.bootstrap_ci" not in dd["bootstrap"]["source"] \
            or "uncertainty_scope" not in rec \
            or "three seeds" not in rec["uncertainty_scope"]:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              "X08: bootstrap provenance / uncertainty scope missing")
    if v04x._semantic_sha(rec) != rec["semantic_sha256"]:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              "X08: audit record semantic_sha256 not self-consistent")
    v04x._verify_sidecar_file(
        Path(ws["cfg"]["out"]) / v04x.AUDIT_RECORD)
    green_dd = dd
    logger.info("[%s] v04x smoke X08+X09 PASS: green audit, exact "
                "DeltaDelta (+0.75 dB / -0.25 nats)", __abbr__)

    # X10 bootstrap bitwise determinism on the frozen procedure.
    r3 = v04x._per_arm_rows(json.loads(
        Path(ws["cfg"]["v03_facts"]).read_text()))
    r4 = v04x._per_arm_rows(json.loads(
        Path(ws["cfg"]["v04_facts"]).read_text()))
    a1 = v04x.compute_delta_delta(r3, r4)
    a2 = v04x.compute_delta_delta(r3, r4)
    if set(a1) != set(a2) or not all(
            np.array_equal(a1[k], a2[k]) for k in a1):
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              "X10: compute_delta_delta is not bitwise deterministic")
    if v04x.paired_bootstrap(a1["dd_psnr_r0"]) \
            != v04x.paired_bootstrap(a2["dd_psnr_r0"]) \
            or v04x.paired_bootstrap(a1["dd_nll"]) \
            != v04x.paired_bootstrap(a2["dd_nll"]):
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              "X10: paired_bootstrap is not bitwise deterministic")
    logger.info("[%s] v04x smoke X10 PASS: bootstrap bitwise "
                "determinism", __abbr__)

    # X11 NLL sign orientation (higher-is-better): a parity arm with
    # WORSE NLL yields a negative improvement delta.
    gm03, gm04 = _v04x_green_metrics(len(ids), len(ids))
    gm04[1]["nll"] = [100.5] * len(ids)
    ws = _v04x_workspace(root / "x11", v04_metrics=gm04)
    rc = v04x.run_audit(ws["cfg"], v03_facts_sha=ws["v03_sha"])
    rec = _v04x_record(ws["cfg"], v04x.AUDIT_RECORD)
    if rc != 0 or rec["delta_delta"]["nll"]["delta_v04_mean"] != -0.5 \
            or rec["delta_delta"]["nll"]["dd"]["mean"] != -1.0:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              "X11: NLL sign orientation wrong (higher-is-better "
              "violated)")
    logger.info("[%s] v04x smoke X11 PASS: NLL sign orientation",
                __abbr__)

    # X12 a failed gate emits no confirmatory DeltaDelta (checked on
    # the X06 record; f17 asserts the same on the frozen example).
    if "delta_delta" in rec_x06 \
            or rec_x06["readiness"]["verdict"] != v04x.GATE_FAILURE:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              "X12: gate-FAIL record carries a DeltaDelta")
    logger.info("[%s] v04x smoke X12 PASS: gate FAIL -> no "
                "confirmatory DeltaDelta", __abbr__)

    # Seeds never pair across architectures (176 SS13): per-seed
    # spreads with identical ensemble means leave the DeltaDelta block
    # bitwise unchanged.
    sp03, _ = _v04x_green_metrics(len(ids), len(ids))
    n = len(ids)
    sp03[0]["psnr_r0"] = [[29.5] * n, [30.0] * n, [30.5] * n]
    sp03[2]["psnr_r0"] = [[30.0] * n, [30.5] * n, [31.0] * n]
    sp03[0]["nll"] = [[99.5] * n, [100.0] * n, [100.5] * n]
    sp03[2]["nll"] = [[99.0] * n, [99.5] * n, [100.0] * n]
    ws = _v04x_workspace(root / "x_spread", v03_metrics=sp03)
    rc = v04x.run_audit(ws["cfg"], v03_facts_sha=ws["v03_sha"])
    rec = _v04x_record(ws["cfg"], v04x.AUDIT_RECORD)
    if rc != 0 or rec["delta_delta"] != green_dd:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              "seed-spread invariance broken: more than the "
              "within-architecture ensemble means entered DeltaDelta")
    logger.info("[%s] v04x smoke PASS: seeds never pair across "
                "architectures (ensemble-mean invariance)", __abbr__)

    # X13 bitwise-identical double runs (fixed scratch roots; the
    # records pin their inputs by path).
    ws = _v04x_workspace(root / "x13")
    out = Path(ws["cfg"]["out"])
    rc1 = v04x.run_audit(ws["cfg"], v03_facts_sha=ws["v03_sha"])
    b1 = (out / v04x.AUDIT_RECORD).read_bytes()
    s1 = (out / (v04x.AUDIT_RECORD + ".sha256")).read_bytes()
    rc2 = v04x.run_audit(ws["cfg"], v03_facts_sha=ws["v03_sha"])
    b2 = (out / v04x.AUDIT_RECORD).read_bytes()
    s2 = (out / (v04x.AUDIT_RECORD + ".sha256")).read_bytes()
    if rc1 != 0 or rc2 != 0 or b1 != b2 or s1 != s2:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              "X13: audit double run is not bitwise identical")
    rc1 = v04x.run_preflight(ws["cfg"], v03_facts_sha=ws["v03_sha"])
    p1 = (out / v04x.PREFLIGHT_RECORD).read_bytes()
    rc2 = v04x.run_preflight(ws["cfg"], v03_facts_sha=ws["v03_sha"])
    p2 = (out / v04x.PREFLIGHT_RECORD).read_bytes()
    if rc1 != 0 or rc2 != 0 or p1 != p2:
        _fail("CROSSFLOW_FIXTURE_MISMATCH",
              "X13: preflight double run is not bitwise identical")
    logger.info("[%s] v04x smoke X13 PASS: bitwise-identical double "
                "runs", __abbr__)

    logger.info("[%s] v04x cross-flow smoke PASS (supplementary, "
                "non-registered): X01-X13 + seed-pairing invariance",
                __abbr__)


def f09_evaluation_determinism() -> None:
    """f09 (178 §5): evaluation determinism -- sibling rerun bit-exact.
    Mirrors the registered V03 implementation-level f09 (v03_eval
    selftest): the aggregate construction (the only randomness is the
    fixed-seed PCG64(3) bootstrap) yields bit-identical canonical
    bytes and semantic hashes across a double build. Synthetic values
    only; no model, no data. The post-training real-data sibling
    rerun is a separate supplementary, non-gating, non-registered
    stage obligation (advisor GO Option 2), not part of this fixture.
    """
    def _synth_aggregates() -> dict:
        per0 = [{k: [float(i + s) for i in range(1, 21)]
                 for k in v04e.METRIC_KEYS} for s in (0, 1, 2)]
        per1 = [{k: [float(i + s) + 0.5 for i in range(1, 21)]
                 for k in v04e.METRIC_KEYS} for s in (0, 1, 2)]
        ens0, agg0 = v04e.build_arm_aggregates(per0, None)
        ens1, agg1 = v04e.build_arm_aggregates(per1, ens0)
        return {"schema": v04e.FACTS_SCHEMA,
                "per_arm": [
                    {"arm": 0, "per_volume_ensemble": ens0,
                     "aggregates": agg0},
                    {"arm": 1, "per_volume_ensemble": ens1,
                     "aggregates": agg1}]}

    f1, f2 = _synth_aggregates(), _synth_aggregates()
    if v04e.canonical_json(f1) != v04e.canonical_json(f2):
        logger.error("f09: canonical bytes diverge across rebuilds")
        _fail("EVAL_DETERMINISM_MISMATCH",
              "f09 canonical bytes diverge across rebuilds")
    if v04e._semantic_sha(f1) != v04e._semantic_sha(f2):
        logger.error("f09: semantic hash diverges across rebuilds")
        _fail("EVAL_DETERMINISM_MISMATCH",
              "f09 semantic hash diverges across rebuilds")
    logger.info("f09 PASS: evaluation determinism (synthetic "
                "aggregates double-build, canonical bytes + semantic "
                "hash bit-exact)")


def f10_static_audit() -> None:
    """f10 (178 §5): repository-source static audit for the frozen silencing-
    pattern classes; any hit -> AUDIT_PATTERN_FOUND. Mechanical
    encoding (implementation-defined static proxies, disclosed in the
    header changelog and the execution record): P1 except-swallow (an
    except suite whose only statement is the null statement -- any
    exception type), P2 silent retry (an except suite whose only
    statement is a loop-continue under a broad clause: bare /
    Exception / BaseException; a narrow typed except in a candidate-
    probe loop with a terminal typed raise is enumeration, not a
    silent retry), P3 marker-gated return (a task-marker comment on
    the return line or on the immediately preceding non-blank line).
    The scan covers every git-tracked *.py (index-inclusive, so staged
    files are covered; untracked files are disclosed as out of scope
    until tracked) and is self-inclusive; the marker tokens are
    built dynamically so this file cannot self-trigger. Scope guard:
    the V04 modules, the RealNVP layer module and this file must be
    covered, else AUDIT_SCOPE_INCOMPLETE.
    """
    root = Path(__file__).resolve().parents[2]
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True, check=False)
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", "*.py"],
            capture_output=True, check=False)
        top_s = top.stdout.decode("utf-8").strip()
        names = [p.decode("utf-8")
                 for p in result.stdout.split(b"\0") if p]
    except (OSError, UnicodeDecodeError) as exc:
        logger.error("f10: git enumeration failed at %s: %s", root, exc)
        _fail("AUDIT_SCOPE_INCOMPLETE",
              f"f10: git enumeration failed at {root}: {exc}")
    if top.returncode != 0 or Path(top_s).resolve() != root:
        logger.error("f10: %s is not a git work-tree root (toplevel=%r)",
                     root, top_s)
        _fail("AUDIT_SCOPE_INCOMPLETE",
              f"f10: {root} is not a git work-tree root "
              f"(git rev-parse --show-toplevel returned {top_s!r})")
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")
        logger.error("f10: git ls-files failed at %s: %s", root, err)
        _fail("AUDIT_SCOPE_INCOMPLETE",
              f"f10: git ls-files failed at {root}: {err}")
    rel = sorted(names)
    files = [root / r for r in rel]
    rel = set(rel)
    required = tuple(
        f"seqref_mri/scripts/v04_{m}.py"
        for m in ("model", "objective", "probe", "train", "gate",
                  "eval", "classify", "crossflow", "fixtures")) + (
        "seqref_mri/src/flows/realnvp_layer.py",)
    missing = [r for r in required if r not in rel]
    if missing:
        logger.error("f10 scope guard: required paths unscanned: %s",
                     missing)
        _fail("AUDIT_SCOPE_INCOMPLETE",
              f"f10: required paths missing from scan: {missing}")
    markers = ("TO" + "DO", "FIX" + "ME")
    broad = re.compile(r"(Base)?Exception(\s+as\s+\w+)?")
    hits: list[tuple[str, int, str]] = []
    for path in files:
        rpath = path.relative_to(root).as_posix()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            logger.error("f10: unreadable source file %s: %s", rpath, exc)
            _fail("AUDIT_SCOPE_INCOMPLETE",
                  f"f10: unreadable source file {rpath}: {exc}")
        n = len(lines)
        for i, ln in enumerate(lines):
            s = ln.strip()
            indent = len(ln) - len(ln.lstrip())
            if re.match(r"except\b", s):
                clause = s[6:].strip()
                if clause.endswith(":"):
                    clause = clause[:-1].strip()
                suite: list[str] = []
                j = i + 1
                while j < n:
                    l2 = lines[j]
                    if not l2.strip():
                        j += 1
                        continue
                    if len(l2) - len(l2.lstrip()) <= indent:
                        break
                    st = l2.strip()
                    if not st.startswith("#"):
                        suite.append(st.split("#", 1)[0].strip())
                    j += 1
                if suite == ["pass"]:
                    hits.append((rpath, i + 1, "P1 except-swallow"))
                elif (suite == ["continue"] and
                      (clause == "" or broad.fullmatch(clause))):
                    hits.append((rpath, i + 1, "P2 silent-retry"))
            if "return" in s and any(m in s for m in markers):
                hits.append((rpath, i + 1, "P3 marker-gated return"))
            if s.startswith("#") and any(m in s for m in markers):
                j = i + 1
                while j < n and not lines[j].strip():
                    j += 1
                if j < n and "return" in lines[j]:
                    hits.append((rpath, i + 1, "P3 marker-gated return"))
    if hits:
        for h in hits:
            logger.error("f10 forbidden pattern: %s:%d %s", *h)
        _fail("AUDIT_PATTERN_FOUND",
              f"f10: {len(hits)} forbidden-pattern hit(s); first: "
              f"{hits[0][0]}:{hits[0][1]} {hits[0][2]}")
    logger.info("f10 PASS: %d git-tracked files scanned, 0 forbidden-"
                "pattern hits (P1/P2/P3)", len(files))


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    logger.info("fixture environment: torch %s, numpy %s",
                torch.__version__, np.__version__)
    gen = torch.Generator().manual_seed(FIXTURE_SEED)
    results: dict[str, str] = {}
    try:
        model = build_v04_model()
        n_params = sum(p.numel() for p in model.parameters())
    except Exception as exc:
        logger.error("V04M construction failed: %s", exc)
        return 2
    for name, fn in (("f01", f01_manifest_bitexact),
                     ("f02", lambda: f02_lambda0_equivalence(model, gen)),
                     ("f03", lambda: f03_lrec_contract(model, gen)),
                     ("f04", lambda: f04_probe_validity(model, gen)),
                     ("f05", f05_checkpoint_selection),
                     ("f06", f06_rerun_ledger_policy),
                     ("f07", f07_verdict_injection),
                     ("f08", f08_facts_schema),
                     ("f09", f09_evaluation_determinism),
                     ("f10", f10_static_audit),
                     ("f11", lambda: f11_mask_alternation(model)),
                     ("f12", lambda: f12_invertibility(model, gen)),
                     ("f13", f13_jacobian_logdet),
                     ("f14", lambda: f14_numerical_conditioning(model, gen)),
                     ("f15", lambda: f15_conditioning_connectivity(model)),
                     ("f16", lambda: f16_logdet_cancellation(model, gen)),
                     ("f17", f17_crossflow_readiness_refusal)):
        try:
            fn()
            results[name] = "PASS"
        except FixtureFailure as exc:
            results[name] = f"FAIL ({exc.code})"
    try:
        lambda_positive_smoke(model, gen)
        results["smoke_lam>0"] = "PASS (supplementary)"
    except FixtureFailure as exc:
        results["smoke_lam>0"] = f"FAIL ({exc.code})"
    try:
        v04t_driver_smoke()
        results["smoke_v04t_driver"] = "PASS (supplementary)"
    except FixtureFailure as exc:
        results["smoke_v04t_driver"] = f"FAIL ({exc.code})"
    try:
        v04g1_gate_refusal(model, gen)
        results["smoke_v04g1_refusal"] = "PASS (supplementary)"
    except FixtureFailure as exc:
        results["smoke_v04g1_refusal"] = f"FAIL ({exc.code})"
    try:
        v04e_integrity_smoke()
        results["smoke_v04e_integrity"] = "PASS (supplementary)"
    except FixtureFailure as exc:
        results["smoke_v04e_integrity"] = f"FAIL ({exc.code})"
    try:
        v04c_routing_smoke()
        results["smoke_v04c_routing"] = "PASS (supplementary)"
    except FixtureFailure as exc:
        results["smoke_v04c_routing"] = f"FAIL ({exc.code})"
    try:
        v04x_crossflow_smoke()
        results["smoke_v04x_crossflow"] = "PASS (supplementary)"
    except FixtureFailure as exc:
        results["smoke_v04x_crossflow"] = f"FAIL ({exc.code})"
    for name, res in results.items():
        logger.info("fixture %s: %s", name, res)
    n_fail = sum(1 for r in results.values() if not r.startswith("PASS"))
    if n_fail:
        logger.error("V04 fixture subset f01-f08, f11-f17: %d FAILURES — "
                     "%s (f14 is PRELIMINARY, synthetic batches)", n_fail,
                     {k: v for k, v in results.items() if v != "PASS"})
        return 2
    logger.info("V04 fixture suite f01-f17: ALL PASS (f14 "
                "PRELIMINARY — synthetic batches, authoritative run at "
                "the fixture gate; f01 logic-verified on synthetic "
                "artifacts, authoritative form at the gate; lam>0 smoke "
                "+ v04t driver smoke + v04g1 refusal + v04e integrity "
                "smoke + v04c routing smoke + v04x cross-flow smoke "
                "supplementary, non-registered). V04M params=%d",
                n_params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
