# =============================================================================
# SEQREF-MRI-V04E v0.1 -- scripts.v04_eval
# LIFETIME: KEEP
# Purpose: frozen V04 evaluation + facts artifact (V04SPEC v1.0 article 176
#          SS12-SS14 FROZEN; V04PLAN v1.0 article 178 SS10-SS11 FROZEN).
#          Runs AFTER the six training runs + the reliability gate, on the
#          199-volume holdout -- the FIRST V04 module legally touching
#          split="val". Faithful port of V03E v0.3 to the T-RealNVP chain:
#          the measurement primitives (d2b encode, tiny_gate decode, d3
#          posterior mean) are model-interface-generic and consumed
#          verbatim; NLL conventions identical to V03 (per-volume
#          per-slice NLL in nats, same p4 statistics, same coordinate map,
#          exact analytic log-dets -- 176 SS13 comparability clause).
# Stage order (locked):
#   A. setup + frozen-input pinning: parents (p4 stats2, p0s), holdout
#      manifest (exactly 199 volumes), probe record (lambda_2'), gate
#      record (all six gates PASS), V03 facts pin (178 SS10: sha256
#      da72ae73...f5274d; the CI settings recorded there must match the
#      locked bootstrap conventions), six-run ledger fully settled.
#   B. Checkpoint selection (176 SS13 / 178 SS9, per arm-seed): min
#      validation L_val, earliest-epoch tie, computed from the training
#      logs -- never holdout (fixture f05 -> SELECTION_RULE_MISMATCH).
#      The V04T-selected checkpoint is consumed; no reselection here.
#   C. Evaluation: 6 selected checkpoints x 199 volumes; per-volume NLL,
#      R0 (z=0) PSNR/NMSE_u, R1 (posterior mean over the registered
#      128-latent Z_DIAG bank f8a6a34b...1dc8d6, u-space mean
#      convention); data_range = 1.0.
#   D. Ensemble + aggregates (176 SS13): per-volume metrics averaged
#      across the 3 matched seeds per arm (metric-first averaging order,
#      never image-first); 95% bootstrap CIs over the 199 volumes (V02G
#      conventions: ONE PCG64(3) stream, B=10000, percentile, paired
#      volume-index draws); deltas vs the lambda=0 anchor on
#      ensemble-mean per-volume values.
#   E. Facts artifact seqref-v04-facts/1 (178 SS11) + SHA-256 sidecar +
#      semantic hash. Evidence-only: no verdict/classification field
#      (fixture f07 -> INJECTED_VERDICT_FIELD); schema refusals per
#      fixture f08 (non-finite metric, n != 199, identity misalignment
#      vs the holdout manifest). Deterministic by construction (f09):
#      the only randomness is the fixed-seed bootstrap and the
#      registered bank. V04E emits NO classification (that is V04C) and
#      NO cross-flow DD (that is V04X, after V04R registration).
# CONVENTION: logger.error + typed raise (V04EError). PASS(0)/ERROR(2).
#   No fallback, no mock, no placeholder, no silent pass.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from seqref_mri.src import free_flow_runtime as ffr
from seqref_mri.src.fastmri_data import FastMRISliceDataset
from seqref_mri.scripts import v02_eval as v02e
from seqref_mri.scripts import v02_manifests as v02m
from seqref_mri.scripts import v04_train as v04t
from seqref_mri.scripts.train_base import _collate
from seqref_mri.scripts.v02_train import state_sha256
from seqref_mri.scripts.v02_manifests import V02Error, canonical_json
from seqref_mri.scripts.v04_model import build_v04_model

logger = logging.getLogger("seqref_mri.v04_eval")

__version__ = "0.1"
__abbr__ = "SEQREF-MRI-V04E"

FACTS_SCHEMA = "seqref-v04-facts/1"
EVAL_RECORD_SCHEMA = "seqref-v04-eval-record/1"
RUN_RECORD_SCHEMA = v04t.RECORD_SCHEMA          # seqref-v04-train-record/1
BOOTSTRAP_SEED = 3                   # V02G convention (176 SS13, quoted)
BOOTSTRAP_B = 10000                  # V02G convention (176 SS13, quoted)
CI_PERCENTILES = (2.5, 97.5)         # 95% percentile interval
HOLDOUT_MANIFEST = "v02_holdout_manifest.json"
VERDICT_KEYS = ("verdict", "classification")
_SEMANTIC_EXCLUDE = ("run", "semantic_sha256", "semantic_scope")
# 178 SS10: the frozen V03 facts artifact is pin-verified at load and the
# CI settings recorded there must match the locked bootstrap conventions.
V03_FACTS_SCHEMA = "seqref-v03-facts/1"
V03_FACTS_SHA256 = ("da72ae736550bc0ca2cedb10676f238cd7598b83054bd923d"
                    "278e02f48f5274d")
# 176 SS12: the registered 128-latent Z_DIAG bank (quoted prefix/suffix
# f8a6a34b...1dc8d6); full sha256 pinned at implementation from the
# deterministic PCG64(0) construction (tdiag.estimators.z_diag_bank).
Z_DIAG_BANK_SHA256 = ("f8a6a34b6b474286359e4c99341de256b62bc0dff1e6bc"
                      "8cf52e72f0461dc8d6")
_HEX64 = frozenset("0123456789abcdef")


class V04EError(Exception):
    """Typed evaluation failure: .code is the machine-readable reason."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V04EError(code, message)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _register_src_path() -> Path:
    """The preflight modules import their siblings top-level (the host
    convention keeps src/ on sys.path); register the same path here so
    the module is import-safe in any environment. Deterministic, not a
    fallback: the path is derived from this file's own location."""
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def _canonical_hash():
    _register_src_path()
    try:
        from preflight_io import canonical_hash
    except ImportError as exc:
        _fail("ENV_IMPORT_FAILED",
              f"preflight_io is not importable in this environment: "
              f"{exc}")
    return canonical_hash


def _verify_sidecar_file(path: Path) -> str:
    side = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not side.is_file():
        _fail("EVAL_INPUT_MISSING",
              f"frozen input or sidecar missing at {path}")
    recorded = side.read_text().split()[0]
    digest = _file_sha256(path)
    if digest != recorded:
        _fail("EVAL_INPUT_DRIFT",
              f"{path.name}: sha256 {digest[:12]}... != sidecar "
              f"{recorded[:12]}...; frozen-input drift")
    return digest


def _semantic_sha(payload: dict) -> str:
    return _canonical_hash()(
        {k: v for k, v in payload.items() if k not in _SEMANTIC_EXCLUDE})


# ---------------------------------------------------------------------------
# Pure helpers (fixture domain -- no model, no data)
# ---------------------------------------------------------------------------

def ensemble_mean(per_seed: list) -> np.ndarray:
    """176 SS13 ensemble: per-volume values averaged across seeds (the
    registered averaging order: metric per seed first, mean across seeds
    second -- never image-first). Every seed vector must share one
    shape (one value per volume)."""
    if not per_seed:
        _fail("ENSEMBLE_INPUT_INVALID", "no seed vectors to average")
    arrs = [np.asarray(v, dtype=np.float64) for v in per_seed]
    shape = arrs[0].shape
    if any(a.shape != shape for a in arrs):
        _fail("ENSEMBLE_INPUT_INVALID",
              f"seed vector shapes diverge: {[a.shape for a in arrs]}")
    for a in arrs:
        if not np.isfinite(a).all():
            _fail("ENSEMBLE_INPUT_INVALID",
                  "a seed vector carries non-finite values")
    return np.mean(np.stack(arrs, axis=0), axis=0)


def ensemble_gate(completed_seeds: list) -> str:
    """176 SS13: the ensemble exists only as the 3 matched-seed
    ensemble over the locked seeds; an arm missing any seed is
    explicitly incomplete -- never a silently smaller ensemble."""
    if set(int(s) for s in completed_seeds) == set(v04t.SEEDS):
        return "complete"
    return "incomplete"


def bootstrap_ci(values, *, seed: int = BOOTSTRAP_SEED,
                 B: int = BOOTSTRAP_B) -> dict:
    """95% bootstrap CI over volumes (V02G conventions): ONE PCG64(seed)
    stream, B resampled index draws, resampled arithmetic means,
    percentile interval. Deterministic by construction."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size == 0:
        _fail("BOOTSTRAP_INPUT_INVALID",
              "bootstrap input must be a non-empty 1-D array")
    if not np.isfinite(v).all():
        _fail("BOOTSTRAP_INPUT_NON_FINITE",
              "bootstrap input contains non-finite values")
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, v.size, size=(B, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, CI_PERCENTILES)
    return {"mean": float(v.mean()), "ci95": [float(lo), float(hi)],
            "B": B, "seed": seed, "method": "percentile",
            "rule": "ONE PCG64(seed) stream; B resampled volume-index "
                    "draws; arithmetic means; percentile 2.5/97.5"}


def bootstrap_relative(arm_values, anchor_values, *,
                       seed: int = BOOTSTRAP_SEED,
                       B: int = BOOTSTRAP_B) -> dict:
    """Relative change of the arm mean vs the anchor mean with a 95%
    CI: ONE shared resample-index matrix applied to both vectors
    (paired volumes); each replicate divides by ITS OWN resampled
    anchor mean -- the percentile interval of the registered
    relative-change statistic, never of a fixed-denominator
    hybrid."""
    a = np.asarray(arm_values, dtype=np.float64)
    c = np.asarray(anchor_values, dtype=np.float64)
    if a.shape != c.shape or a.ndim != 1 or a.size == 0:
        _fail("BOOTSTRAP_INPUT_INVALID",
              "arm/anchor vectors must be matching non-empty 1-D "
              "arrays")
    if not (np.isfinite(a).all() and np.isfinite(c).all()):
        _fail("BOOTSTRAP_INPUT_NON_FINITE",
              "arm/anchor vectors contain non-finite values")
    anchor_mean = float(c.mean())
    if anchor_mean == 0.0:
        _fail("BOOTSTRAP_INPUT_INVALID",
              "anchor mean is zero; relative change is undefined")
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, a.size, size=(B, a.size))
    anchor_means = c[idx].mean(axis=1)
    if (anchor_means == 0.0).any():
        _fail("BOOTSTRAP_INPUT_INVALID",
              "a resampled anchor mean is exactly zero; the relative "
              "change is undefined for that replicate")
    rel = (a[idx].mean(axis=1) - anchor_means) / np.abs(anchor_means)
    lo, hi = np.percentile(rel, CI_PERCENTILES)
    return {"relative": (float(a.mean()) - anchor_mean)
            / abs(anchor_mean),
            "ci95": [float(lo), float(hi)], "B": B, "seed": seed,
            "method": "percentile",
            "rule": "ONE PCG64(seed) stream; ONE shared index matrix "
                    "for arm and anchor (paired volumes); relative "
                    "change per resample vs that resample's own "
                    "anchor mean; percentile 2.5/97.5"}


def scan_for_verdict(node, path: str = "facts") -> None:
    """Fixture f07: the facts artifact is evidence-only. Any verdict /
    classification key anywhere is refused."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k in VERDICT_KEYS:
                _fail("INJECTED_VERDICT_FIELD",
                      f"a {k!r} key entered the facts at {path}.{k}; "
                      f"the stage is evidence-only by preregistration")
            scan_for_verdict(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            scan_for_verdict(v, f"{path}[{i}]")


def _scan_finite(node, path: str) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            _scan_finite(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _scan_finite(v, f"{path}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        _fail("FACTS_NON_FINITE_METRIC",
              f"non-finite metric value at {path}")
    elif isinstance(node, (np.floating,)) and not np.isfinite(node):
        _fail("FACTS_NON_FINITE_METRIC",
              f"non-finite metric value at {path}")


def validate_facts(facts: dict, holdout_identities: list) -> None:
    """Fixture f08: schema refusals on the constructed facts artifact.
    Per-volume metric blocks must cover exactly the 199 holdout
    volumes, in manifest identity order, with finite values only;
    duplicate volume identifiers are an explicit refusal; every
    numeric field anywhere must be finite."""
    n = len(holdout_identities)
    for entry in facts.get("per_arm_seed", []):
        pv = entry.get("per_volume")
        if pv is None:
            continue                       # failed arm-seed: stability
        ids = pv.get("identities", [])
        seen = set()
        for e in ids:
            key = (e.get("file"), int(e.get("slice_index", -1)))
            if key in seen:
                _fail("FACTS_DUPLICATE_VOLUME_ID",
                      f"arm {entry.get('arm')} seed {entry.get('seed')}: "
                      f"duplicate volume identity {key!r}")
            seen.add(key)
        if len(ids) != n:
            _fail("FACTS_VOLUME_COUNT_MISMATCH",
                  f"arm {entry.get('arm')} seed {entry.get('seed')}: "
                  f"{len(ids)} per-volume identities != {n}")
        for got, want in zip(ids, holdout_identities):
            if got.get("file") != want.get("file") \
                    or int(got.get("slice_index")) \
                    != int(want.get("slice_index")):
                _fail("FACTS_IDENTITY_MISMATCH",
                      f"arm {entry.get('arm')} seed "
                      f"{entry.get('seed')}: per-volume identity "
                      f"{got!r} != holdout manifest entry {want!r}")
    _scan_finite({k: v for k, v in facts.items()
                  if k not in ("schema", "spec", "plan", "code",
                               "environment")}, "facts")


def validate_parent_references(refs: dict, path: str =
                               "parent_references") -> None:
    """Immutable parent-artifact references (178 SS11): every reference
    carries a well-formed sha256; file-backed references carry a
    non-empty path; generated banks carry the construction manifest
    hash instead. Nested groups (train_records) recurse."""
    if not isinstance(refs, dict) or not refs:
        _fail("PARENT_REFERENCE_INVALID",
              f"{path} must be a non-empty mapping of references")
    for name, entry in refs.items():
        if not isinstance(entry, dict):
            _fail("PARENT_REFERENCE_INVALID",
                  f"{path}.{name}: not a reference object")
        if "sha256" not in entry \
                and all(isinstance(v, dict) for v in entry.values()):
            validate_parent_references(entry, f"{path}.{name}")
            continue
        s = entry.get("sha256")
        if not isinstance(s, str) or len(s) != 64 \
                or any(c not in _HEX64 for c in s):
            _fail("PARENT_REFERENCE_INVALID",
                  f"{path}.{name}: malformed sha256 {s!r}")
        if "manifest_sha256" not in entry:
            p = entry.get("path")
            if not isinstance(p, str) or not p:
                _fail("PARENT_REFERENCE_INVALID",
                      f"{path}.{name}: file-backed reference lacks a "
                      f"path")


# ---------------------------------------------------------------------------
# Production environment (lazy, total) and frozen-input loading
# ---------------------------------------------------------------------------

def _env() -> SimpleNamespace:
    _register_src_path()
    try:
        from preflight_parents import StageError
        from seqref_mri.scripts import tiny_gate as tg
        from seqref_mri.tdiag import d2b, d3, estimators
    except ImportError as exc:
        _fail("ENV_IMPORT_FAILED",
              f"an environment-bound dependency is not importable in "
              f"this environment: {exc}")
    return SimpleNamespace(
        torch=torch, DataLoader=DataLoader, Subset=Subset,
        StageError=StageError, ffr=ffr,
        FastMRISliceDataset=FastMRISliceDataset, tg=tg, collate=_collate,
        d2b=d2b, d3=d3, estimators=estimators,
        state_sha256=state_sha256)


def load_v03_facts_reference(path: str, *,
                             expected_sha: str = V03_FACTS_SHA256
                             ) -> dict:
    """178 SS10: the frozen V03 facts artifact, pin-verified at load
    (sha256 da72ae73...f5274d). The CI settings recorded in it must
    match the locked V02G bootstrap conventions -- V04E's bootstrap is
    verified at implementation against the frozen record, not against
    memory."""
    p = Path(path)
    _verify_sidecar_file(p)
    digest = _file_sha256(p)
    if digest != expected_sha:
        _fail("V03_FACTS_PIN_MISMATCH",
              f"V03 facts sha256 {digest[:12]}... != the locked pin "
              f"{expected_sha[:12]}...; the cross-flow parent drifted")
    facts = json.loads(p.read_text())
    if facts.get("schema") != V03_FACTS_SCHEMA:
        _fail("V03_FACTS_INVALID",
              f"V03 facts schema {facts.get('schema')!r} != "
              f"{V03_FACTS_SCHEMA!r}")
    seen = 0
    for arm in facts.get("per_arm", []):
        aggs = arm.get("aggregates") or {}
        blocks = list((aggs.get("means_ci") or {}).values())
        blocks += list((aggs.get("delta_vs_lambda0") or {}).values())
        for b in blocks:
            if not isinstance(b, dict) or "B" not in b:
                continue
            seen += 1
            if int(b.get("B", -1)) != BOOTSTRAP_B \
                    or int(b.get("seed", -1)) != BOOTSTRAP_SEED \
                    or b.get("method") != "percentile":
                _fail("CI_SETTINGS_MISMATCH",
                      f"V03 facts arm {arm.get('arm')}: recorded CI "
                      f"settings (B={b.get('B')}, seed={b.get('seed')}, "
                      f"method={b.get('method')!r}) diverge from the "
                      f"locked V02G conventions (B={BOOTSTRAP_B}, "
                      f"seed={BOOTSTRAP_SEED}, percentile)")
    if seen == 0:
        _fail("V03_FACTS_INVALID",
              "the V03 facts artifact records no bootstrap CI settings; "
              "the 178 SS10 verification has nothing to check against")
    return facts


def _load_run_record(path: Path, arm: int, seed: int,
                     ledger_sha: str) -> dict:
    """One V04T run record, pinned twice: the file must match the
    LEDGER's recorded sha256 (the append-only chain of custody), and
    its own sidecar; schema and arm/seed identity are gated."""
    if not path.is_file():
        _fail("RUN_SET_INCOMPLETE",
              f"run record {path.name} (arm {arm} seed {seed}) missing")
    if _file_sha256(path) != ledger_sha:
        _fail("RUN_RECORD_INVALID",
              f"{path.name}: sha256 != the ledger pin; the run "
              f"evidence drifted")
    _verify_sidecar_file(path)
    rec = json.loads(path.read_text())
    if rec.get("schema") != RUN_RECORD_SCHEMA:
        _fail("RUN_RECORD_INVALID",
              f"{path.name}: schema {rec.get('schema')!r} != "
              f"{RUN_RECORD_SCHEMA!r}")
    if int(rec.get("arm", -1)) != arm or int(rec.get("seed", -1)) != seed:
        _fail("RUN_RECORD_INVALID",
              f"{path.name}: declares arm {rec.get('arm')} seed "
              f"{rec.get('seed')}, expected arm {arm} seed {seed}")
    return rec


def load_run_set(run_dir: Path) -> dict:
    """The 6 arm-seed run set from the append-only ledger: every
    (arm, seed) pair must be settled (terminal); completed pairs yield
    a hash-verified run record, failed pairs feed stability. Fail-closed:
    any unsettled pair aborts the evaluation before holdout contact."""
    ledger_path = run_dir / "v04_run_ledger.json"
    if not ledger_path.is_file():
        _fail("RUN_SET_INCOMPLETE",
              f"run ledger missing at {ledger_path}")
    try:
        ledger = v04t.load_ledger(ledger_path)
    except v04t.V04TError as exc:
        _fail("RUN_SET_INCOMPLETE", f"ledger: {exc.code}: {exc}")
    settled = {}
    for e in ledger["entries"]:
        settled[(int(e["arm"]), int(e["seed"]))] = e
    runs = {}
    for seed, arm in v04t.GLOBAL_ORDER:
        pair = (arm, seed)          # GLOBAL_ORDER is (seed, arm)
        if pair not in settled \
                or settled[pair]["status"] not in v04t.TERMINAL_STATUSES:
            _fail("RUN_SET_INCOMPLETE",
                  f"arm {pair[0]} seed {pair[1]} is not settled in the "
                  f"ledger (status "
                  f"{settled.get(pair, {}).get('status')!r})")
        e = settled[pair]
        entry = {"arm": pair[0], "seed": pair[1],
                 "status": e["status"],
                 "attempts": sum(
                     1 for x in ledger["entries"]
                     if int(x["arm"]) == pair[0]
                     and int(x["seed"]) == pair[1]),
                 "error_code": e.get("error_code"),
                 "wall_clock_s": e.get("wall_clock_s"),
                 "record_file": e.get("record_file"),
                 "record_sha256": e.get("record_sha256")}
        if e["status"] == "completed":
            entry["record"] = _load_run_record(
                run_dir / e["record_file"], pair[0], pair[1],
                e["record_sha256"])
        runs[pair] = entry
    return runs


# ---------------------------------------------------------------------------
# Stage C: per arm-seed holdout evaluation (R0 + R1)
# ---------------------------------------------------------------------------

def _evaluate_checkpoint(env: SimpleNamespace, model, states: list,
                         bank: np.ndarray) -> dict:
    """Per-volume NLL, R0 (z=0) and R1 (posterior mean over the
    registered Z_DIAG bank, u-space mean convention) metrics for one
    selected checkpoint through the production primitives (model-
    interface-generic: d2b encode, tiny_gate decode, d3 posterior
    mean)."""
    model.eval()
    per = {"identities": [], "nll": [], "psnr_r0": [], "psnr_r1": [],
           "nmse_u_r0": [], "nmse_u_r1": []}
    with torch.no_grad():
        for j, st in enumerate(states):
            where = f"holdout slice {j} {st['identity']!r}"
            _z, ldj, log_pz = v02e._encode(env, model, st, where)
            nll = float(-(log_pz + ldj))
            psnr0, nmse0 = v02e._z0_metrics(env, model, st, where)
            psnr1, nmse1 = env.d3._pm_metrics(model, st, bank,
                                              st["cond"], st["mask"])
            for name, v in (("nll", nll), ("psnr_r0", psnr0),
                            ("psnr_r1", psnr1), ("nmse_u_r0", nmse0),
                            ("nmse_u_r1", nmse1)):
                if not math.isfinite(float(v)):
                    _fail("FACTS_NON_FINITE_METRIC",
                          f"{where}: {name} is non-finite ({v!r})")
            per["identities"].append(st["identity"])
            per["nll"].append(nll)
            per["psnr_r0"].append(float(psnr0))
            per["psnr_r1"].append(float(psnr1))
            per["nmse_u_r0"].append(float(nmse0))
            per["nmse_u_r1"].append(float(nmse1))
    return per


def _load_selected_model(env: SimpleNamespace, cfg: dict,
                         selection: dict):
    """The V04T-selected checkpoint: file sha256 pinned to the run
    record, strict state_dict load into the registered architecture
    (fresh V04M init, then fully overwritten)."""
    path = Path(cfg["run_dir"]) / selection["file"]
    if not path.is_file():
        _fail("CHECKPOINT_HASH_MISMATCH",
              f"selected checkpoint {selection['file']} missing")
    if _file_sha256(path) != selection["file_sha256"]:
        _fail("CHECKPOINT_HASH_MISMATCH",
              f"{selection['file']}: sha256 != the run-record pin")
    try:
        payload = torch.load(path, map_location="cpu",
                             weights_only=True)
    except Exception as exc:  # torch.load raises several runtime types
        _fail("CHECKPOINT_HASH_MISMATCH",
              f"{selection['file']}: unreadable: "
              f"{type(exc).__name__}: {exc}")
    model = build_v04_model()
    try:
        model.load_state_dict(payload, strict=True)
    except (RuntimeError, KeyError) as exc:
        _fail("CHECKPOINT_HASH_MISMATCH",
              f"{selection['file']}: state_dict load failed: "
              f"{type(exc).__name__}: {exc}")
    return model


def _check_bank(bank_rec: dict) -> None:
    """The readout bank is a frozen input: the realised Z_DIAG bank's
    sha256 must equal the pinned full hash (registered prefix/suffix
    f8a6a34b...1dc8d6, 176 SS12)."""
    if bank_rec.get("bank_sha256") != Z_DIAG_BANK_SHA256:
        _fail("BANK_HASH_MISMATCH",
              f"Z_DIAG bank sha256 "
              f"{str(bank_rec.get('bank_sha256'))[:12]}... != the "
              f"pinned {Z_DIAG_BANK_SHA256[:12]}...; the readout bank "
              f"is not the registered one")


def _load_zdiag_bank(env: SimpleNamespace) -> dict:
    rec = env.estimators.z_diag_bank()
    _check_bank(rec)
    return rec


# ---------------------------------------------------------------------------
# Stage D: ensemble + aggregates (pure -- fixture domain)
# ---------------------------------------------------------------------------

METRIC_KEYS = ("nll", "psnr_r0", "psnr_r1", "nmse_u_r0", "nmse_u_r1")
DELTA_KEYS = ("psnr_r0", "psnr_r1", "nmse_u_r0", "nmse_u_r1")


def build_arm_aggregates(seed_pers: list,
                         anchor_ens: dict | None) -> tuple:
    """176 SS13/SS14 aggregates for one arm: ensemble-mean per-volume
    values across the completed seeds, 95% bootstrap CIs per metric,
    and deltas vs the lambda=0 anchor ensemble (None when the anchor
    is unavailable -- recorded, never coerced)."""
    if len(seed_pers) != len(v04t.SEEDS):
        _fail("ENSEMBLE_INPUT_INVALID",
              f"the matched-seed ensemble requires exactly "
              f"{len(v04t.SEEDS)} completed seed vectors, got "
              f"{len(seed_pers)}; incomplete arms must be gated "
              f"before aggregation (ensemble_gate)")
    ens = {k: ensemble_mean([p[k] for p in seed_pers]).tolist()
           for k in METRIC_KEYS}
    aggs = {k: bootstrap_ci(ens[k]) for k in METRIC_KEYS}
    deltas = None
    if anchor_ens is not None:
        deltas = {}
        for k in DELTA_KEYS:
            diff = (np.asarray(ens[k], dtype=np.float64)
                    - np.asarray(anchor_ens[k], dtype=np.float64))
            deltas[f"delta_{k}"] = bootstrap_ci(diff)
        deltas["delta_nll_relative"] = bootstrap_relative(
            ens["nll"], anchor_ens["nll"])
    return ens, {"means_ci": aggs, "delta_vs_lambda0": deltas}


# ---------------------------------------------------------------------------
# Orchestration: stage runner, record/facts writers, run, CLI
# ---------------------------------------------------------------------------

def _execute_stages(results: list, stages: list) -> bool:
    """Fail-fast stage runner (the V03G1/V03E convention): on the first
    failed stage, record the failure, mark every later stage
    "not_run", stop."""
    for i, (sid, fn) in enumerate(stages):
        try:
            details = fn()
        except V04EError as exc:
            results.append({"stage": sid, "status": "fail",
                            "code": exc.code, "message": str(exc)})
            for sid2, _fn2 in stages[i + 1:]:
                results.append({"stage": sid2, "status": "not_run"})
            return False
        except Exception as exc:
            logger.exception("[%s] unexpected error in stage %s",
                             __abbr__, sid)
            results.append({"stage": sid, "status": "fail",
                            "code": type(exc).__name__,
                            "message": str(exc)})
            for sid2, _fn2 in stages[i + 1:]:
                results.append({"stage": sid2, "status": "not_run"})
            return False
        results.append({"stage": sid, "status": "pass",
                        "details": details})
    return True


def _write_with_sidecar(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(payload))
    digest = _file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n")
    logger.info("[%s] written: %s (sha256 %s...)", __abbr__, path,
                digest[:12])
    return digest


def _parents(env: SimpleNamespace, cfg: dict, state: dict) -> None:
    try:
        state["p4"] = ffr.load_p4s2_parent(cfg["p4_stats2"])
        state["s_ref"] = env.tg._s_ref_from_p0s(cfg["p0s_facts"])
    except Exception as exc:  # FreeFlowError/StageError family
        logger.error("[%s] PARENT_VERIFICATION_FAILED: %s: %s",
                     __abbr__, type(exc).__name__, exc)
        raise V04EError("PARENT_VERIFICATION_FAILED",
                        f"{type(exc).__name__}: {exc}") from exc


def run(cfg: dict) -> int:
    """The frozen evaluation. cfg keys: data_root, manifest_dir,
    run_dir, p4_stats2, p0s_facts, v03_facts, probe_record,
    gate_record, code_commit, code_branch, out_dir."""
    env = _env()
    env.torch.set_num_threads(1)
    t0 = time.time()
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    state: dict = {}
    results: list = []

    def _setup():
        _parents(env, cfg, state)
        man_dir = Path(cfg["manifest_dir"])
        holdman = v02e._load_manifest(man_dir, HOLDOUT_MANIFEST,
                                      "holdout")
        if int(holdman["n_volumes"]) != v02m.N_HOLDOUT_VOLUMES \
                or len(holdman["entries"]) != v02m.N_HOLDOUT_VOLUMES:
            _fail("MANIFEST_CONTENT_MISMATCH",
                  "holdout manifest diverges from the locked "
                  f"{v02m.N_HOLDOUT_VOLUMES} volumes")
        state["holdman"] = holdman
        state["identities"] = [
            {"file": e["file"], "slice_index": int(e["slice_index"])}
            for e in holdman["entries"]]
        try:
            state["probe"] = v04t.load_probe_record(
                cfg["probe_record"])
        except v04t.V04TError as exc:
            _fail(exc.code, f"probe record: {exc}")
        try:
            state["gate"] = v04t.load_gate_record(cfg["gate_record"])
        except v04t.V04TError as exc:
            _fail(exc.code, f"gate record: {exc}")
        state["v03_facts"] = load_v03_facts_reference(cfg["v03_facts"])
        state["runs"] = load_run_set(Path(cfg["run_dir"]))
        state["parent_shas"] = {
            "holdout_manifest": _file_sha256(man_dir / HOLDOUT_MANIFEST),
            "p4_stats2": _file_sha256(Path(cfg["p4_stats2"])),
            "p0s_facts": _file_sha256(Path(cfg["p0s_facts"])),
            "v03_facts": _file_sha256(Path(cfg["v03_facts"])),
            "probe_record": _file_sha256(Path(cfg["probe_record"])),
            "gate_record": _file_sha256(Path(cfg["gate_record"]))}
        return {"holdout_volumes": v02m.N_HOLDOUT_VOLUMES,
                "note": "setup + frozen inputs pinned; zero holdout "
                        "contact so far -- split=\"val\" is first "
                        "touched inside stage holdout_evaluation"}

    def _stage_b():
        selections = {}
        for pair, entry in state["runs"].items():
            if entry["status"] != "completed":
                continue
            rec = entry["record"]
            try:
                selections[pair] = v04t.select_checkpoint(
                    rec["validation_history"], rec["checkpoints"])
            except v04t.V04TError as exc:
                _fail(exc.code, f"arm {pair[0]} seed {pair[1]}: {exc}")
        state["selections"] = selections
        return {f"arm{pair[0]}_seed{pair[1]}": s
                for pair, s in selections.items()}

    def _stage_c():
        try:
            ds_val = env.FastMRISliceDataset(
                cfg["data_root"], split="val", mode="eval")
        except Exception as exc:  # typed, logging dataset ctor
            _fail("DATA_PREMISE_FAILURE",
                  f"{type(exc).__name__}: {exc}")
        state["states"] = v02e._collect_states(
            env, ds_val, state["holdman"]["entries"], state["p4"],
            state["s_ref"], v02m.N_HOLDOUT_VOLUMES)
        bank_rec = _load_zdiag_bank(env)
        state["bank_sha256"] = bank_rec["bank_sha256"]
        state["bank_manifest_sha256"] = bank_rec["manifest_sha256"]
        per_metrics = {}
        for pair, sel in state["selections"].items():
            logger.info("[%s] stage C: evaluating arm %d seed %d "
                        "(epoch %d)", __abbr__, pair[0], pair[1],
                        sel["epoch"])
            model = _load_selected_model(env, cfg, sel)
            per_metrics[pair] = _evaluate_checkpoint(
                env, model, state["states"], bank_rec["bank"])
        state["per_metrics"] = per_metrics
        return {"evaluated": len(per_metrics),
                "bank_sha256": bank_rec["bank_sha256"],
                "bank_manifest_sha256":
                    bank_rec["manifest_sha256"]}

    def _stage_de():
        facts = _build_facts(cfg, state)
        scan_for_verdict(facts)
        validate_facts(facts, state["identities"])
        validate_parent_references(facts["parent_references"])
        semantic = {k: v for k, v in facts.items()
                    if k not in _SEMANTIC_EXCLUDE}
        facts["semantic_sha256"] = _canonical_hash()(semantic)
        facts["semantic_scope"] = {
            "included_keys": sorted(semantic.keys()),
            "excluded": list(_SEMANTIC_EXCLUDE),
            "canonicalisation": "preflight_io.canonical_hash"}
        sha = _write_with_sidecar(out / "v04_facts.json", facts)
        state["facts_sha256"] = sha
        return {"facts_sha256": sha}

    stages = [("setup_and_frozen_inputs", _setup),
              ("checkpoint_selection", _stage_b),
              ("holdout_evaluation", _stage_c),
              ("ensemble_aggregates_facts", _stage_de)]
    ok = _execute_stages(results, stages)
    record = {
        "schema": EVAL_RECORD_SCHEMA,
        "script": f"{__abbr__} v{__version__}",
        "spec": {"article": 176, "status": "FROZEN"},
        "plan": {"article": 178, "status": "FROZEN",
                 "amendments": ["A1"]},
        "inputs": {k: cfg[k] for k in
                   ("data_root", "manifest_dir", "run_dir",
                    "p4_stats2", "p0s_facts", "v03_facts",
                    "probe_record", "gate_record")},
        "stages": results,
        "overall": "PASS" if ok else "ERROR",
        "facts_sha256": state.get("facts_sha256"),
        "wall_clock_s": time.time() - t0,
        "environment": {"python": sys.version.split()[0],
                        "torch": torch.__version__,
                        "numpy": np.__version__,
                        "platform": sys.platform},
    }
    _write_with_sidecar(out / "v04_eval_record.json", record)
    if not ok:
        failed = next(r for r in results if r["status"] == "fail")
        logger.error("[%s] EVALUATION ERROR: stage %s failed (%s); "
                     "validity abort -- never a scientific outcome",
                     __abbr__, failed["stage"], failed["code"])
        return 2
    print(f"PASS {__abbr__} v{__version__}: {len(state['per_metrics'])} "
          f"arm-seed checkpoints evaluated on "
          f"{v02m.N_HOLDOUT_VOLUMES} volumes; v04_facts.json + sidecar")
    return 0


def _build_facts(cfg: dict, state: dict) -> dict:
    """seqref-v04-facts/1 per 178 SS11 (pure construction from stage
    products; scans run on the returned dict). Mirrors
    seqref-v03-facts/1; evidence-only -- no verdict fields (f07)."""
    probe = state["probe"]
    lambdas = {"0": 0.0, "1": float(probe["lambda_prime"])}
    shas = state["parent_shas"]
    train_refs = {}
    per_arm_seed = []
    for seed, arm in v04t.GLOBAL_ORDER:
        pair = (arm, seed)          # GLOBAL_ORDER is (seed, arm)
        entry = state["runs"][pair]
        train_refs[f"arm{pair[0]}_seed{pair[1]}"] = {
            "path": entry["record_file"],
            "sha256": entry["record_sha256"]}
        row = {"arm": pair[0], "seed": pair[1],
               "status": entry["status"],
               "run": {"attempts": entry["attempts"],
                       "error_code": entry["error_code"],
                       "wall_clock_s": entry["wall_clock_s"]}}
        if entry["status"] == "completed":
            sel = state["selections"][pair]
            row["checkpoint"] = {
                "file": sel["file"], "file_sha256":
                    sel["file_sha256"],
                "epoch": sel["epoch"], "L_val": sel["L_val"]}
            row["per_volume"] = state["per_metrics"][pair]
        per_arm_seed.append(row)
    per_arm = []
    anchor_ens = None
    for arm in v04t.ARMS:
        arm_pairs = [(a, s) for s, a in v04t.GLOBAL_ORDER if a == arm]
        completed = [pair for pair in arm_pairs
                     if pair in state["per_metrics"]]
        stab = {
            "seeds_completed": len(completed),
            "seeds_numerical_failure": sum(
                1 for pair in arm_pairs
                if state["runs"][pair]["status"]
                == "numerical_failure"),
            "seeds_failed_validity": sum(
                1 for pair in arm_pairs
                if state["runs"][pair]["status"]
                == "failed_validity")}
        row = {"arm": arm, "lambda": lambdas[str(arm)],
               "stability": stab,
               "ensemble_status": ensemble_gate(
                   [pair[1] for pair in completed])}
        if row["ensemble_status"] == "complete":
            ens, aggs = build_arm_aggregates(
                [state["per_metrics"][pair] for pair in completed],
                None if arm == 0 else anchor_ens)
            row["per_volume_ensemble"] = ens
            row["aggregates"] = aggs
            if arm == 0:
                anchor_ens = ens
        per_arm.append(row)
    return {
        "schema": FACTS_SCHEMA,
        "script": f"{__abbr__} v{__version__}",
        "spec": {"article": 176, "status": "FROZEN"},
        "plan": {"article": 178, "status": "FROZEN",
                 "amendments": ["A1"]},
        "code": {"commit": cfg["code_commit"],
                 "branch": cfg["code_branch"]},
        "seeds": list(v04t.SEEDS),
        "arms": {"lambdas": lambdas,
                 "probe_record": cfg["probe_record"],
                 "gate_record": cfg["gate_record"]},
        "evaluation": {
            "holdout_manifest": HOLDOUT_MANIFEST,
            "n_volumes": v02m.N_HOLDOUT_VOLUMES,
            "data_range": 1.0,
            "readouts": ["R0 (z=0)", "R1 (posterior mean over the "
                         "registered 128-latent Z_DIAG bank, u-space "
                         "mean convention)"],
            "bootstrap": {"seed": BOOTSTRAP_SEED, "B": BOOTSTRAP_B,
                          "method": "percentile",
                          "pairing": "paired volume-index draws"},
            "uncertainty_scope": "the paired bootstrap over the 199 "
                                 "holdout volumes measures evaluation-"
                                 "volume variability conditional on the "
                                 "fitted three-seed ensembles; it is not "
                                 "a CI over possible training "
                                 "initialisations (176 SS13 registered "
                                 "limitation; restated in V04R)"},
        "v03_facts_reference": {
            "path": cfg["v03_facts"], "sha256": V03_FACTS_SHA256,
            "ci_settings": "verified identical to the locked V02G "
                           "conventions at load (178 SS10)"},
        "parent_references": {
            "v03_facts": {"path": cfg["v03_facts"],
                          "sha256": shas["v03_facts"]},
            "holdout_manifest": {
                "path": str(Path(cfg["manifest_dir"])
                            / HOLDOUT_MANIFEST),
                "sha256": shas["holdout_manifest"]},
            "p4_stats2": {"path": cfg["p4_stats2"],
                          "sha256": shas["p4_stats2"]},
            "p0s_facts": {"path": cfg["p0s_facts"],
                          "sha256": shas["p0s_facts"]},
            "probe_record": {"path": cfg["probe_record"],
                             "sha256": shas["probe_record"]},
            "gate_record": {"path": cfg["gate_record"],
                            "sha256": shas["gate_record"]},
            "z_diag_bank": {
                "sha256": state["bank_sha256"],
                "manifest_sha256": state["bank_manifest_sha256"],
                "rule": "PCG64(0) standard_normal (128, 13824) f64 -> "
                        "f32 (176 SS12 registered bank)"},
            "train_records": train_refs},
        "per_arm_seed": per_arm_seed,
        "per_arm": per_arm,
        "environment": {"python": sys.version.split()[0],
                        "torch": torch.__version__,
                        "numpy": np.__version__,
                        "platform": sys.platform},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"{__abbr__} v{__version__} -- frozen V04 "
                    f"evaluation + facts artifact (176 SS12-14, "
                    f"178 SS10-11)")
    ap.add_argument("--mode", required=True, choices=("run",))
    ap.add_argument("--data-root")
    ap.add_argument("--manifest-dir")
    ap.add_argument("--run-dir")
    ap.add_argument("--p4-stats2")
    ap.add_argument("--p0s-facts")
    ap.add_argument("--v03-facts")
    ap.add_argument("--probe-record")
    ap.add_argument("--gate-record")
    ap.add_argument("--commit")
    ap.add_argument("--branch")
    ap.add_argument("--out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    required = ("data_root", "manifest_dir", "run_dir", "p4_stats2",
                "p0s_facts", "v03_facts", "probe_record", "gate_record",
                "commit", "branch", "out")
    missing = [k for k in required if getattr(args, k) is None]
    if missing:
        logger.error("[%s] missing required arguments: %s", __abbr__,
                     ", ".join("--" + m.replace("_", "-")
                               for m in missing))
        return 2
    cfg = {k: getattr(args, k) for k in required if k != "out"}
    cfg["out_dir"] = args.out
    cfg["code_commit"] = args.commit
    cfg["code_branch"] = args.branch
    try:
        return run(cfg)
    except V04EError as exc:
        logger.error("[%s] ERROR: %s", __abbr__, exc)
        return 2
    except Exception:
        logger.exception("[%s] unexpected exception", __abbr__)
        return 2


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Changelog
#   v0.1 (2026-10-12): initial V04E (advisor GO -- frozen core + synthetic
#   integrity tests). Faithful port of V03E v0.3 to the T-RealNVP chain
#   (176 SS12-14 / 178 SS10-11): 6-run (2 arms x 3 seeds) ledger-driven
#   evaluation; R0 (z=0) + R1 (registered 128-latent Z_DIAG bank, u-space
#   mean) readouts through the model-interface-generic production
#   primitives (d2b encode, tiny_gate decode, d3 posterior mean); V02G
#   bootstrap conventions (ONE PCG64(3), B=10000, percentile, paired
#   volume-index draws); facts artifact seqref-v04-facts/1 mirroring
#   seqref-v03-facts/1 with immutable parent references, the V03 facts
#   sha256 pin (da72ae73...f5274d) + recorded-CI-settings verification
#   (178 SS10), the pinned Z_DIAG bank hash (f8a6a34b...1dc8d6), a
#   duplicate-volume-ID guard and a parent-reference format gate
#   (advisor extension checks); no verdict fields (f07), schema refusals
#   (f08). Authoritative --mode run implemented (fail-closed: parents,
#   manifest, probe/gate records, V03 facts pin, settled six-run ledger
#   all required before holdout contact) but NOT executed -- the six
#   training runs do not exist yet.
# ---------------------------------------------------------------------------
