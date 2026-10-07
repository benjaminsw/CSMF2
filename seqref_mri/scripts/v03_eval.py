# SEQREF-V03E v0.3 -- scripts.v03_eval
# LIFETIME: KEEP
# =============================================================================
# Purpose: frozen V03 evaluation + facts artifact (V03SPEC v0.3
#          SS12/SS13/SS14, article 163 FROZEN; V03PLAN v0.3 SS9-SS11,
#          article 164 FROZEN + Amendments A1/A2/A3; A3 = article 166
#          FROZEN). Runs AFTER training, on the 199-volume holdout --
#          this is the FIRST module legally touching split="val".
# Stage order (locked):
#   A. V02 R0/z=0 holdout anchor reproduction (Amendment A3, deferred
#      from the SS17 gate): the frozen v0.2 checkpoints' holdout NLL
#      (per-slice, step 0/3258) and z=0 PSNR/NMSE_u means reproduce the
#      V02E anchors BIT-EXACTLY. This is the first holdout action and it
#      gates everything: on failure the run is an evaluation-stage
#      validity abort (eval record written, NO facts artifact, no V03
#      number is trusted or emitted), exit 2.
#   B. Checkpoint selection (SPEC SS13, per arm-seed): min validation
#      L_val, earliest-epoch tie, computed from the training logs --
#      never manual (fixture f05 -> SELECTION_RULE_MISMATCH).
#   C. Evaluation: 12 selected checkpoints x 199 volumes; per-volume
#      NLL, R0 (z=0) PSNR/NMSE_u, R1 (posterior mean over the
#      registered 128-latent Z_DIAG bank, u-space mean convention).
#   D. Ensemble + aggregates (SPEC SS13): per-volume metrics averaged
#      across the 3 matched seeds per arm; 95% bootstrap CIs over the
#      199 volumes (V02G conventions: ONE PCG64(3) stream, B=10000,
#      percentile); deltas vs the lambda=0 anchor on ensemble-mean
#      per-volume values.
#   E. Facts artifact seqref-v03-facts/1 (PLAN SS11) + SHA-256 sidecar
#      + semantic hash. Evidence-only: no verdict/classification field
#      (fixture f07 -> INJECTED_VERDICT_FIELD); schema refusals per
#      fixture f08 (non-finite metric, n != 199, identity misalignment
#      vs the holdout manifest). Deterministic by construction (f09):
#      the only randomness is the fixed-seed bootstrap and the
#      registered bank.
# CONVENTION: logger.error + typed raise (V03EError). PASS(0)/ERROR(2).
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
from seqref_mri.scripts import v03_train as v03t
from seqref_mri.scripts.train_base import _collate
from seqref_mri.scripts.v02_train import state_sha256
from seqref_mri.scripts.v02_manifests import V02Error, canonical_json

logger = logging.getLogger("seqref_mri.v03_eval")

__version__ = "0.3"
__abbr__ = "SEQREF-V03E"

FACTS_SCHEMA = "seqref-v03-facts/1"
EVAL_RECORD_SCHEMA = "seqref-v03-eval-record/1"
RUN_RECORD_SCHEMA = "seqref-v03-train-record/1"
BOOTSTRAP_SEED = 3                   # V02G convention (quoted)
BOOTSTRAP_B = 10000                  # V02G convention (quoted)
CI_PERCENTILES = (2.5, 97.5)         # 95% percentile interval
HOLDOUT_MANIFEST = "v02_holdout_manifest.json"
VERDICT_KEYS = ("verdict", "classification")
_SEMANTIC_EXCLUDE = ("run", "semantic_sha256", "semantic_scope")


class V03EError(Exception):
    """Typed evaluation failure: .code is the machine-readable reason."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V03EError(code, message)


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


def _v02_code(exc: V02Error, fallback: str) -> str:
    head = str(exc).split(":", 1)[0].strip()
    import re as _re
    return head if _re.fullmatch(r"[A-Z0-9_]+", head) else fallback


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
# Pure helpers (selftest domain -- no model, no data)
# ---------------------------------------------------------------------------

def select_checkpoint(validation_history: list,
                      checkpoints: list) -> dict:
    """SPEC SS13 selection rule, per arm-seed (fixture f05): the
    checkpoint with the lowest validation L_val; ties break to the
    earliest epoch (strict < keeps the first occurrence). The selected
    epoch must carry a checkpoint whose recorded L_val matches."""
    if not validation_history:
        _fail("SELECTION_RULE_MISMATCH",
              "empty validation history; the rule needs at least one "
              "epoch")
    best_ep, best_val = None, None
    for e in validation_history:
        lv = e.get("L_val")
        ep = e.get("epoch")
        if lv is None or not math.isfinite(float(lv)):
            _fail("SELECTION_RULE_MISMATCH",
                  f"epoch {ep}: L_val {lv!r} is not finite; the "
                  f"argmin rule is undefined")
        if best_val is None or float(lv) < best_val:
            best_ep, best_val = int(ep), float(lv)
    ck = {int(c["epoch"]): c for c in checkpoints}
    if best_ep not in ck:
        _fail("SELECTION_RULE_MISMATCH",
              f"selected epoch {best_ep} carries no checkpoint")
    c = ck[best_ep]
    if float(c.get("L_val", float("nan"))) != best_val:
        _fail("SELECTION_RULE_MISMATCH",
              f"checkpoint epoch {best_ep} records L_val "
              f"{c.get('L_val')!r} != the validation history "
              f"{best_val!r}")
    return {"epoch": best_ep, "L_val": best_val,
            "file": c["file"], "file_sha256": c["file_sha256"]}


def ensemble_mean(per_seed: list) -> np.ndarray:
    """SPEC SS13 ensemble: per-volume values averaged across seeds.
    Every seed vector must share one shape (one value per volume)."""
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
    """SPEC SS13: the ensemble exists only as the 3 matched-seed
    ensemble over the locked seeds; an arm missing any seed is
    explicitly incomplete -- never a silently smaller ensemble."""
    if set(int(s) for s in completed_seeds) == set(v03t.SEEDS):
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
    every numeric field anywhere must be finite."""
    n = len(holdout_identities)
    for entry in facts.get("per_arm_seed", []):
        pv = entry.get("per_volume")
        if pv is None:
            continue                       # failed arm-seed: stability
        ids = pv.get("identities", [])
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


def require_bitwise_equal(measured, anchor, what: str) -> None:
    """A3 stage: bit-exact float64 reproduction of the V02E holdout
    NLL endpoint arrays."""
    m = np.asarray(measured, dtype=np.float64)
    a = np.asarray(anchor, dtype=np.float64)
    if m.shape != a.shape:
        _fail("V02_ANCHOR_REPRODUCTION_MISMATCH",
              f"{what}: measured shape {m.shape} != anchor shape "
              f"{a.shape}")
    if not np.isfinite(m).all():
        _fail("V02_ANCHOR_REPRODUCTION_MISMATCH",
              f"{what}: the recomputed values carry a non-finite entry")
    if m.tobytes() != a.tobytes():
        idx = int(np.argmax(m != a))
        _fail("V02_ANCHOR_REPRODUCTION_MISMATCH",
              f"{what}: first divergence at slice {idx}: measured "
              f"{m.flat[idx]!r} != anchor {a.flat[idx]!r}")


def require_bitwise_scalar(measured: float, anchor: float,
                           what: str) -> None:
    """A3 stage: bit-exact reproduction of a scalar anchor (z=0 metric
    means at the endpoint checkpoints)."""
    m = np.float64(measured)
    a = np.float64(anchor)
    if not (np.isfinite(m) and np.isfinite(a)):
        _fail("V02_ANCHOR_REPRODUCTION_MISMATCH",
              f"{what}: non-finite value (measured {measured!r}, "
              f"anchor {anchor!r})")
    if m.tobytes() != a.tobytes():
        _fail("V02_ANCHOR_REPRODUCTION_MISMATCH",
              f"{what}: measured {measured!r} != anchor {anchor!r}")


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


def _load_run_record(path: Path, arm: int, seed: int,
                     ledger_sha: str) -> dict:
    """One V03T run record, pinned twice: the file must match the
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
    """The 12 arm-seed run set from the append-only ledger: every
    (arm, seed) pair must be settled (terminal); completed pairs yield
    a hash-verified run record, failed pairs feed stability."""
    ledger_path = run_dir / "v03_run_ledger.json"
    if not ledger_path.is_file():
        _fail("RUN_SET_INCOMPLETE",
              f"run ledger missing at {ledger_path}")
    try:
        ledger = v03t.load_ledger(ledger_path)
    except v03t.V03TError as exc:
        _fail("RUN_SET_INCOMPLETE", f"ledger: {exc.code}: {exc}")
    settled = {}
    for e in ledger["entries"]:
        settled[(int(e["arm"]), int(e["seed"]))] = e
    runs = {}
    for seed, arm in v03t.GLOBAL_ORDER:
        pair = (arm, seed)          # GLOBAL_ORDER is (seed, arm)
        if pair not in settled \
                or settled[pair]["status"] not in v03t.TERMINAL_STATUSES:
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
                 "wall_clock_s": e.get("wall_clock_s")}
        if e["status"] == "completed":
            entry["record"] = _load_run_record(
                run_dir / e["record_file"], pair[0], pair[1],
                e["record_sha256"])
        runs[pair] = entry
    return runs


def load_v02_holdout_anchors(path: Path) -> dict:
    """The frozen V02E holdout anchors for the A3 deferred check:
    sidecar integrity, schema, semantic self-consistency, per-slice
    NLL endpoint arrays (n=199, finite), and the endpoint per-
    checkpoint z=0 means."""
    _verify_sidecar_file(path)
    facts = json.loads(path.read_text())
    if facts.get("schema") != v02e.FACTS_SCHEMA:
        _fail("ANCHOR_RECORD_INVALID",
              f"facts schema {facts.get('schema')!r} != "
              f"{v02e.FACTS_SCHEMA!r}")
    if _semantic_sha(facts) != facts.get("semantic_sha256"):
        _fail("ANCHOR_RECORD_INVALID",
              "facts semantic_sha256 does not match the recomputed "
              "content hash; the frozen anchors drifted")
    hold = facts.get("endpoint_measurements", {}).get("holdout", {})
    anchors = {}
    for key in ("nll_step0", "nll_final"):
        arr = hold.get(key)
        if not isinstance(arr, list) \
                or len(arr) != v02m.N_HOLDOUT_VOLUMES:
            _fail("ANCHOR_RECORD_INVALID",
                  f"endpoint_measurements.holdout.{key} must be a list "
                  f"of {v02m.N_HOLDOUT_VOLUMES} per-slice values")
        a = np.asarray(arr, dtype=np.float64)
        if not np.isfinite(a).all():
            _fail("ANCHOR_RECORD_INVALID",
                  f"endpoint_measurements.holdout.{key} carries a "
                  f"non-finite anchor value")
        anchors[key] = arr
    per_ckpt = {int(c["step"]): c
                for c in hold.get("per_checkpoint", [])}
    for step in (v02e.STEP0, v02e.FINAL_STEP):
        c = per_ckpt.get(step)
        if c is None:
            _fail("ANCHOR_RECORD_INVALID",
                  f"no per-checkpoint anchor at step {step}")
        for key in ("nll_mean", "z0_psnr_mean", "z0_nmse_u_mean"):
            v = c.get(key)
            if v is None or not math.isfinite(float(v)):
                _fail("ANCHOR_RECORD_INVALID",
                      f"per-checkpoint step {step}: {key} {v!r} is "
                      f"not finite")
    anchors["per_checkpoint"] = per_ckpt
    return anchors


# ---------------------------------------------------------------------------
# Stage A: V02 R0/z=0 holdout anchor reproduction (Amendment A3)
# ---------------------------------------------------------------------------

def _holdout_nll_z0(env: SimpleNamespace, model, states: list):
    """Per-slice holdout NLL + z=0 PSNR/NMSE_u for one frozen
    checkpoint through the V02E production primitives."""
    n = len(states)
    nll = np.empty(n, dtype=np.float64)
    psnr = np.empty(n, dtype=np.float64)
    nmse = np.empty(n, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for j, st in enumerate(states):
            _z, ldj, log_pz = v02e._encode(env, model, st,
                                           f"holdout slice {j}")
            nll[j] = float(-(log_pz + ldj))
            psnr[j], nmse[j] = v02e._z0_metrics(env, model, st,
                                                f"holdout slice {j}")
    if not (np.isfinite(nll).all() and np.isfinite(psnr).all()
            and np.isfinite(nmse).all()):
        _fail("V02_ANCHOR_REPRODUCTION_MISMATCH",
              "the recomputed holdout metrics carry non-finite values")
    return nll, psnr, nmse


def stage_a_anchor_reproduction(env: SimpleNamespace, cfg: dict,
                                state: dict) -> dict:
    """Amendment A3 (deferred from the SS17 gate): the frozen v0.2
    checkpoints' holdout NLL (per-slice, steps 0/3258) and z=0
    PSNR/NMSE_u means reproduce the V02E anchors bit-exactly. FIRST
    holdout action; gates every V03 number."""
    logger.info("[%s] stage A: first holdout contact -- A3 deferred "
                "V02 anchor reproduction", __abbr__)
    try:
        ds_val = env.FastMRISliceDataset(
            cfg["data_root"], split="val", mode="eval")
    except Exception as exc:  # typed, logging dataset ctor
        _fail("DATA_PREMISE_FAILURE",
              f"{type(exc).__name__}: {exc}")
    state["states"] = v02e._collect_states(
        env, ds_val, state["holdman"]["entries"], state["p4"],
        state["s_ref"], v02m.N_HOLDOUT_VOLUMES)
    anchors = load_v02_holdout_anchors(Path(cfg["v02_facts"]))
    measured = {}
    for step, key in ((v02e.STEP0, "nll_step0"),
                      (v02e.FINAL_STEP, "nll_final")):
        nll, psnr, nmse = _holdout_nll_z0(env, state["models"][step],
                                          state["states"])
        require_bitwise_equal(nll, anchors[key], f"holdout {key}")
        c = anchors["per_checkpoint"][step]
        require_bitwise_scalar(float(nll.mean()), c["nll_mean"],
                               f"holdout nll_mean step {step}")
        require_bitwise_scalar(float(psnr.mean()), c["z0_psnr_mean"],
                               f"holdout z0_psnr_mean step {step}")
        require_bitwise_scalar(float(nmse.mean()), c["z0_nmse_u_mean"],
                               f"holdout z0_nmse_u_mean step {step}")
        measured[str(step)] = {
            "nll_mean": float(nll.mean()),
            "z0_psnr_mean": float(psnr.mean()),
            "z0_nmse_u_mean": float(nmse.mean())}
    return {"status": "pass",
            "detail": "V02E holdout anchors reproduced bit-exactly "
                      "(per-slice NLL endpoints + z=0 means at steps "
                      "0/3258)",
            "measured": measured}


# ---------------------------------------------------------------------------
# Stage C: per arm-seed holdout evaluation (R0 + R1)
# ---------------------------------------------------------------------------

def _evaluate_checkpoint(env: SimpleNamespace, model, states: list,
                         bank: np.ndarray) -> dict:
    """Per-volume NLL, R0 (z=0) and R1 (posterior mean over the
    registered Z_DIAG bank, u-space mean convention) metrics for one
    selected checkpoint through the production primitives."""
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
    """The selected checkpoint: file sha256 pinned to the run record,
    strict state_dict load into the registered architecture."""
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
    model = ffr.build_model()
    try:
        model.load_state_dict(payload, strict=True)
    except (RuntimeError, KeyError) as exc:
        _fail("CHECKPOINT_HASH_MISMATCH",
              f"{selection['file']}: state_dict load failed: "
              f"{type(exc).__name__}: {exc}")
    return model


# ---------------------------------------------------------------------------
# Stage D: ensemble + aggregates (pure -- selftest domain)
# ---------------------------------------------------------------------------

METRIC_KEYS = ("nll", "psnr_r0", "psnr_r1", "nmse_u_r0", "nmse_u_r1")
DELTA_KEYS = ("psnr_r0", "psnr_r1", "nmse_u_r0", "nmse_u_r1")


def build_arm_aggregates(seed_pers: list,
                         anchor_ens: dict | None) -> tuple:
    """SPEC SS13/SS14 aggregates for one arm: ensemble-mean per-volume
    values across the completed seeds, 95% bootstrap CIs per metric,
    and deltas vs the lambda=0 anchor ensemble (None when the anchor
    is unavailable -- recorded, never coerced)."""
    if len(seed_pers) != len(v03t.SEEDS):
        _fail("ENSEMBLE_INPUT_INVALID",
              f"the matched-seed ensemble requires exactly "
              f"{len(v03t.SEEDS)} completed seed vectors, got "
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
    """Fail-fast stage runner (the V03G1 convention): on the first
    failed stage, record the failure, mark every later stage
    "not_run", stop."""
    for i, (sid, fn) in enumerate(stages):
        try:
            details = fn()
        except V03EError as exc:
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
        implb = ffr.load_implb_parent(cfg["implb_facts"])
        ffr.require_spline_b(implb["spline_b"])
        state["implb"] = implb
        state["s_ref"] = env.tg._s_ref_from_p0s(cfg["p0s_facts"])
    except Exception as exc:  # FreeFlowError/StageError family
        logger.error("[%s] PARENT_VERIFICATION_FAILED: %s: %s",
                     __abbr__, type(exc).__name__, exc)
        raise V03EError("PARENT_VERIFICATION_FAILED",
                        f"{type(exc).__name__}: {exc}") from exc


def run(cfg: dict) -> int:
    """The frozen evaluation. cfg keys: data_root, manifest_dir,
    run_dir, train_record, run_root, p4_stats2, implb_facts,
    p0s_facts, v02_facts, probe_record, gate_record, code_commit,
    code_branch, out_dir."""
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
        state["manifests"] = [
            v02e._load_manifest(man_dir, f"v02_epoch{ep}_manifest.json",
                                "train_epoch")
            for ep in range(v02m.N_EPOCHS)]
        try:
            record = v02e._load_train_record(cfg["train_record"],
                                             state["manifests"])
        except V02Error as exc:
            _fail(_v02_code(exc, "TRAIN_RECORD_MISMATCH"), str(exc))
        state["v02_record"] = record
        ckpt_recs = {int(c["step"]): c for c in record["checkpoints"]}
        state["models"] = {}
        for step in (v02e.STEP0, v02e.FINAL_STEP):
            try:
                state["models"][step] = v02e._load_checkpoint(
                    env, Path(cfg["run_root"]), ckpt_recs[step],
                    state["implb"]["spline_b"])
            except V02Error as exc:
                _fail("CHECKPOINT_HASH_MISMATCH", f"step {step}: {exc}")
        try:
            state["probe"] = v03t.load_probe_record(
                cfg["probe_record"])
        except v03t.V03TError as exc:
            _fail("PROBE_RECORD_INVALID", f"{exc.code}: {exc}")
        gate = json.loads(_verify_gate(cfg["gate_record"]))
        if gate.get("overall") != "PASS":
            _fail("GATE_RECORD_INVALID",
                  "the parent reproduction gate record does not show "
                  "overall PASS; evaluation without a passed gate is "
                  "a validity abort")
        state["gate"] = gate
        state["runs"] = load_run_set(Path(cfg["run_dir"]))
        return {"holdout_volumes": v02m.N_HOLDOUT_VOLUMES,
                "note": "setup + frozen inputs pinned; zero holdout "
                        "contact so far -- split=\"val\" is first "
                        "touched inside stage "
                        "v02_anchor_reproduction"}

    def _stage_a():
        return stage_a_anchor_reproduction(env, cfg, state)

    def _stage_b():
        selections = {}
        for pair, entry in state["runs"].items():
            if entry["status"] != "completed":
                continue
            rec = entry["record"]
            selections[pair] = select_checkpoint(
                rec["validation_history"], rec["checkpoints"])
        state["selections"] = selections
        return {f"arm{pair[0]}_seed{pair[1]}": s
                for pair, s in selections.items()}

    def _stage_c():
        bank_rec = env.estimators.z_diag_bank()
        state["bank_sha256"] = bank_rec["bank_sha256"]
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
        semantic = {k: v for k, v in facts.items()
                    if k not in _SEMANTIC_EXCLUDE}
        facts["semantic_sha256"] = _canonical_hash()(semantic)
        facts["semantic_scope"] = {
            "included_keys": sorted(semantic.keys()),
            "excluded": list(_SEMANTIC_EXCLUDE),
            "canonicalisation": "preflight_io.canonical_hash"}
        sha = _write_with_sidecar(out / "v03_facts.json", facts)
        state["facts_sha256"] = sha
        return {"facts_sha256": sha}

    stages = [("setup_and_frozen_inputs", _setup),
              ("v02_anchor_reproduction", _stage_a),
              ("checkpoint_selection", _stage_b),
              ("holdout_evaluation", _stage_c),
              ("ensemble_aggregates_facts", _stage_de)]
    ok = _execute_stages(results, stages)
    record = {
        "schema": EVAL_RECORD_SCHEMA,
        "script": f"{__abbr__} v{__version__}",
        "spec": {"article": 163, "status": "FROZEN"},
        "plan": {"article": 164, "status": "FROZEN",
                 "amendments": ["A1", "A2", "A3"]},
        "a3_deferred_check": "V02 R0/z=0 holdout anchor reproduction "
                             "(article 166): stage "
                             "v02_anchor_reproduction -- FIRST holdout "
                             "action, gates every V03 number",
        "inputs": {k: cfg[k] for k in
                   ("data_root", "manifest_dir", "run_dir",
                    "train_record", "run_root", "p4_stats2",
                    "implb_facts", "p0s_facts", "v02_facts",
                    "probe_record", "gate_record")},
        "stages": results,
        "overall": "PASS" if ok else "ERROR",
        "facts_sha256": state.get("facts_sha256"),
        "wall_clock_s": time.time() - t0,
        "environment": {"python": sys.version.split()[0],
                        "torch": torch.__version__,
                        "platform": sys.platform},
    }
    _write_with_sidecar(out / "v03_eval_record.json", record)
    if not ok:
        failed = next(r for r in results if r["status"] == "fail")
        logger.error("[%s] EVALUATION ERROR: stage %s failed (%s); "
                     "validity abort -- never a scientific outcome",
                     __abbr__, failed["stage"], failed["code"])
        return 2
    print(f"PASS {__abbr__} v{__version__}: A3 anchor reproduction "
          f"bit-exact; {len(state['per_metrics'])} arm-seed "
          f"checkpoints evaluated on 199 volumes; v03_facts.json + "
          f"sidecar")
    return 0


def _verify_gate(path_str: str) -> str:
    p = Path(path_str)
    _verify_sidecar_file(p)
    return p.read_text()


def _build_facts(cfg: dict, state: dict) -> dict:
    """seqref-v03-facts/1 per PLAN SS11 (pure construction from stage
    products; scans run on the returned dict)."""
    probe = state["probe"]
    lambdas = {str(a): (0.0 if a == 0 else
                        probe["lambdas"][alpha])
               for a, alpha in ((1, "0.1"), (2, "0.5"), (3, "0.9"))}
    lambdas["0"] = 0.0
    per_arm_seed = []
    for seed, arm in v03t.GLOBAL_ORDER:
        pair = (arm, seed)          # GLOBAL_ORDER is (seed, arm)
        entry = state["runs"][pair]
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
    for arm in (0, 1, 2, 3):
        arm_pairs = [(a, s) for s, a in v03t.GLOBAL_ORDER if a == arm]
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
        "spec": {"article": 163, "status": "FROZEN"},
        "plan": {"article": 164, "status": "FROZEN",
                 "amendments": ["A1", "A2", "A3"]},
        "code": {"commit": cfg["code_commit"],
                 "branch": cfg["code_branch"]},
        "seeds": list(v03t.SEEDS),
        "arms": {"lambdas": lambdas,
                 "probe_record": cfg["probe_record"],
                 "gate_record": cfg["gate_record"]},
        "a3_anchor_reproduction": "PASS (bit-exact vs V02E holdout "
                                  "anchors; see v03_eval_record.json)",
        "per_arm_seed": per_arm_seed,
        "per_arm": per_arm,
        "environment": {"python": sys.version.split()[0],
                        "torch": torch.__version__,
                        "platform": sys.platform},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"{__abbr__} v{__version__} -- frozen V03 "
                    f"evaluation + facts artifact (V03SPEC SS12-14, "
                    f"Amendment A3)")
    ap.add_argument("--mode", required=True,
                    choices=("selftest", "run"))
    ap.add_argument("--data-root")
    ap.add_argument("--manifest-dir")
    ap.add_argument("--run-dir")
    ap.add_argument("--train-record")
    ap.add_argument("--run-root")
    ap.add_argument("--p4-stats2")
    ap.add_argument("--implb-facts")
    ap.add_argument("--p0s-facts")
    ap.add_argument("--v02-facts")
    ap.add_argument("--probe-record")
    ap.add_argument("--gate-record")
    ap.add_argument("--commit")
    ap.add_argument("--branch")
    ap.add_argument("--out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.mode == "selftest":
        return _selftest()
    required = ("data_root", "manifest_dir", "run_dir", "train_record",
                "run_root", "p4_stats2", "implb_facts", "p0s_facts",
                "v02_facts", "probe_record", "gate_record", "commit",
                "branch", "out")
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
    except V03EError as exc:
        logger.error("[%s] ERROR: %s", __abbr__, exc)
        return 2
    except Exception:
        logger.exception("[%s] unexpected exception", __abbr__)
        return 2


# ---------------------------------------------------------------------------
# Selftest (pure fixtures -- no model, no data)
# ---------------------------------------------------------------------------

def _expect_fail(fn, code: str, what: str) -> None:
    try:
        fn()
    except V03EError as exc:
        if exc.code != code:
            _fail("EVAL_SELFTEST_WRONG_CODE",
                  f"{what}: expected {code}, got {exc.code}")
        return
    _fail("EVAL_SELFTEST_NO_FAILURE",
          f"{what}: expected {code}, no failure raised")


def _selftest() -> int:
    import tempfile as _tf

    tmp = Path(_tf.mkdtemp(prefix="v03e_selftest_"))

    # f05: checkpoint selection (SPEC SS13) -- argmin, earliest-epoch
    # tie, and every refusal branch.
    hist = [{"epoch": e, "L_val": v}
            for e, v in ((1, 3.0), (2, 2.0), (3, 2.5))]
    ckpts = [{"epoch": e, "L_val": v, "file": f"ckpt_e{e}.pt",
              "file_sha256": f"{e}" * 64}
             for e, v in ((1, 3.0), (2, 2.0), (3, 2.5))]
    sel = select_checkpoint(hist, ckpts)
    if sel["epoch"] != 2 or sel["L_val"] != 2.0 \
            or sel["file"] != "ckpt_e2.pt":
        _fail("EVAL_SELFTEST_WRONG_VALUE", f"f05 argmin: {sel}")
    tie_hist = [{"epoch": e, "L_val": 2.0} for e in (1, 2)]
    tie_ck = [{"epoch": e, "L_val": 2.0, "file": f"f{e}.pt",
               "file_sha256": "a" * 64} for e in (1, 2)]
    if select_checkpoint(tie_hist, tie_ck)["epoch"] != 1:
        _fail("EVAL_SELFTEST_WRONG_VALUE", "f05 earliest-epoch tie")
    _expect_fail(lambda: select_checkpoint([], ckpts),
                 "SELECTION_RULE_MISMATCH", "f05 empty history")
    _expect_fail(lambda: select_checkpoint(
        [{"epoch": 1, "L_val": float("nan")}], ckpts),
        "SELECTION_RULE_MISMATCH", "f05 non-finite L_val")
    _expect_fail(lambda: select_checkpoint(hist, ckpts[:1]),
                 "SELECTION_RULE_MISMATCH", "f05 missing checkpoint")
    bad_ck = [dict(c) for c in ckpts]
    bad_ck[1]["L_val"] = 2.1
    _expect_fail(lambda: select_checkpoint(hist, bad_ck),
                 "SELECTION_RULE_MISMATCH",
                 "f05 checkpoint L_val mismatch")
    logger.info("[%s] f05 PASS: checkpoint selection rule", __abbr__)

    # Ensemble mean (SPEC SS13).
    em = ensemble_mean([[1.0, 2.0], [3.0, 4.0]])
    if em.tolist() != [2.0, 3.0]:
        _fail("EVAL_SELFTEST_WRONG_VALUE",
              f"ensemble mean {em.tolist()}")
    _expect_fail(lambda: ensemble_mean([]),
                 "ENSEMBLE_INPUT_INVALID", "ensemble empty")
    _expect_fail(lambda: ensemble_mean([[1.0], [1.0, 2.0]]),
                 "ENSEMBLE_INPUT_INVALID", "ensemble shape divergence")
    _expect_fail(lambda: ensemble_mean(
        [[1.0, float("inf")], [1.0, 2.0]]),
        "ENSEMBLE_INPUT_INVALID", "ensemble non-finite")

    # Matched-seed completeness (SPEC SS13): exactly the 3 locked
    # seeds, or the arm is explicitly incomplete.
    if ensemble_gate([101, 202, 303]) != "complete" \
            or ensemble_gate([202, 101, 303]) != "complete" \
            or ensemble_gate([101, 202]) != "incomplete" \
            or ensemble_gate([101]) != "incomplete" \
            or ensemble_gate([]) != "incomplete":
        _fail("EVAL_SELFTEST_WRONG_VALUE", "ensemble gate branches")
    _expect_fail(lambda: build_arm_aggregates(
        [{k: [1.0, 2.0] for k in METRIC_KEYS} for _ in range(2)],
        None), "ENSEMBLE_INPUT_INVALID",
        "2-seed ensemble must be refused")

    # Bootstrap CIs (V02G conventions): determinism + refusal branches.
    vals = [float(i) for i in range(1, 51)]
    ci1 = bootstrap_ci(vals, B=500)
    ci2 = bootstrap_ci(vals, B=500)
    if ci1 != ci2:
        _fail("EVAL_SELFTEST_WRONG_VALUE",
              "bootstrap_ci is not deterministic")
    if ci1["B"] != 500 or ci1["seed"] != BOOTSTRAP_SEED \
            or not (ci1["ci95"][0] <= ci1["mean"] <= ci1["ci95"][1]):
        _fail("EVAL_SELFTEST_WRONG_VALUE", f"bootstrap_ci {ci1}")
    _expect_fail(lambda: bootstrap_ci([]),
                 "BOOTSTRAP_INPUT_INVALID", "bootstrap empty")
    _expect_fail(lambda: bootstrap_ci([[1.0, 2.0]]),
                 "BOOTSTRAP_INPUT_INVALID", "bootstrap 2-D input")
    _expect_fail(lambda: bootstrap_ci([1.0, float("nan")]),
                 "BOOTSTRAP_INPUT_NON_FINITE", "bootstrap non-finite")
    rel1 = bootstrap_relative([v + 1.0 for v in vals], vals, B=500)
    rel2 = bootstrap_relative([v + 1.0 for v in vals], vals, B=500)
    if rel1 != rel2:
        _fail("EVAL_SELFTEST_WRONG_VALUE",
              "bootstrap_relative is not deterministic")
    _expect_fail(lambda: bootstrap_relative([1.0], [0.0]),
                 "BOOTSTRAP_INPUT_INVALID", "zero anchor mean")
    _expect_fail(lambda: bootstrap_relative([1.0, 2.0], [1.0]),
                 "BOOTSTRAP_INPUT_INVALID", "relative shape mismatch")
    _expect_fail(lambda: bootstrap_relative(
        [1.0, float("inf")], [1.0, 2.0]),
        "BOOTSTRAP_INPUT_NON_FINITE", "relative non-finite")
    # Each replicate divides by ITS OWN resampled anchor mean (the
    # registered relative-change statistic), pinned numerically
    # against the PCG64(3) index stream.
    a_pin = np.array([2.0, 4.0, 8.0])
    c_pin = np.array([1.0, 2.0, 3.0])
    got = bootstrap_relative(a_pin, c_pin, B=100)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    idx = rng.integers(0, 3, size=(100, 3))
    exp_am = c_pin[idx].mean(axis=1)
    exp = (a_pin[idx].mean(axis=1) - exp_am) / np.abs(exp_am)
    elo, ehi = np.percentile(exp, CI_PERCENTILES)
    exp_rel = ((float(a_pin.mean()) - float(c_pin.mean()))
               / abs(float(c_pin.mean())))
    if got["ci95"] != [float(elo), float(ehi)] \
            or got["relative"] != exp_rel:
        _fail("EVAL_SELFTEST_WRONG_VALUE",
              "bootstrap_relative does not use the per-replicate "
              "anchor mean")

    # f07: verdict-injection refusal (the stage is evidence-only).
    scan_for_verdict({"a": {"b": [1.0, {"c": 2.0}]}})
    _expect_fail(lambda: scan_for_verdict({"a": {"verdict": "GO"}}),
                 "INJECTED_VERDICT_FIELD", "f07 nested verdict")
    _expect_fail(lambda: scan_for_verdict(
        {"per_arm": [{"classification": "X"}]}),
        "INJECTED_VERDICT_FIELD", "f07 classification in list")
    logger.info("[%s] f07 PASS: verdict-injection refusal", __abbr__)

    # f08: facts schema refusals (synthetic 3-volume identities; the
    # count gate is against the passed holdout identity list).
    ids = [{"file": f"vol{i}.h5", "slice_index": i} for i in range(3)]
    pv = {"identities": ids, "nll": [1.0, 2.0, 3.0],
          "psnr_r0": [30.0, 31.0, 32.0],
          "psnr_r1": [31.0, 32.0, 33.0],
          "nmse_u_r0": [0.1, 0.1, 0.1],
          "nmse_u_r1": [0.09, 0.09, 0.09]}
    validate_facts({"schema": FACTS_SCHEMA,
                    "per_arm_seed": [
                        {"arm": 0, "seed": 101, "status": "completed",
                         "per_volume": pv},
                        {"arm": 1, "seed": 101,
                         "status": "numerical_failure"}]}, ids)
    _expect_fail(lambda: validate_facts(
        {"per_arm_seed": [{"arm": 0, "seed": 101,
                           "per_volume": {**pv,
                                          "identities": ids[:2]}}]},
        ids), "FACTS_VOLUME_COUNT_MISMATCH", "f08 volume count")
    _expect_fail(lambda: validate_facts(
        {"per_arm_seed": [
            {"arm": 0, "seed": 101,
             "per_volume": {**pv,
                            "identities": [ids[0], ids[1],
                                           {"file": "volX.h5",
                                            "slice_index": 2}]}}]},
        ids), "FACTS_IDENTITY_MISMATCH", "f08 identity order")
    _expect_fail(lambda: validate_facts(
        {"per_arm_seed": [
            {"arm": 0, "seed": 101,
             "per_volume": {**pv,
                            "nll": [1.0, float("nan"), 3.0]}}]},
        ids), "FACTS_NON_FINITE_METRIC", "f08 non-finite metric")
    logger.info("[%s] f08 PASS: facts schema refusals", __abbr__)

    # Facts construction with an incomplete arm: arm 3 is missing
    # seed 303 -> ensemble_status "incomplete" and NO scientific
    # aggregates; complete arms keep the full 3-seed ensemble.
    ids3 = [{"file": f"v{i}.h5", "slice_index": i} for i in range(3)]
    cfg_s = {"probe_record": "p", "gate_record": "g",
             "code_commit": "c", "code_branch": "b"}
    state_s = {"probe": {"lambdas": {"0.1": 0.1, "0.5": 0.5,
                                     "0.9": 0.9}},
               "runs": {}, "selections": {}, "per_metrics": {}}
    for seed, arm in v03t.GLOBAL_ORDER:
        done = (arm, seed) != (3, 303)
        state_s["runs"][(arm, seed)] = {
            "status": "completed" if done else "numerical_failure",
            "attempts": 1, "error_code": None, "wall_clock_s": 1.0}
        if done:
            state_s["selections"][(arm, seed)] = {
                "file": "f.pt", "file_sha256": "a" * 64,
                "epoch": 2, "L_val": 1.0}
            state_s["per_metrics"][(arm, seed)] = {
                "identities": ids3,
                **{k: [float(i + 1) for i in range(3)]
                   for k in METRIC_KEYS}}
    facts_s = _build_facts(cfg_s, state_s)
    scan_for_verdict(facts_s)
    validate_facts(facts_s, ids3)
    arms_by_id = {r["arm"]: r for r in facts_s["per_arm"]}
    if arms_by_id[3]["ensemble_status"] != "incomplete" \
            or "per_volume_ensemble" in arms_by_id[3] \
            or "aggregates" in arms_by_id[3]:
        _fail("EVAL_SELFTEST_WRONG_VALUE",
              f"incomplete arm leaked aggregates: {arms_by_id[3]}")
    if arms_by_id[0]["ensemble_status"] != "complete" \
            or arms_by_id[0]["aggregates"]["delta_vs_lambda0"] \
            is not None \
            or arms_by_id[2]["aggregates"]["delta_vs_lambda0"] is None:
        _fail("EVAL_SELFTEST_WRONG_VALUE",
              "complete-arm aggregates malformed")
    logger.info("[%s] ensemble completeness PASS: incomplete arm is "
                "explicit, aggregates 3-seed only", __abbr__)

    # A3 bit-exact comparators.
    require_bitwise_equal([1.0, 2.0], [1.0, 2.0], "x")
    require_bitwise_scalar(1.5, 1.5, "x")
    _expect_fail(lambda: require_bitwise_equal(
        [1.0, math.nextafter(2.0, 3.0)], [1.0, 2.0], "x"),
        "V02_ANCHOR_REPRODUCTION_MISMATCH", "one-ulp divergence")
    _expect_fail(lambda: require_bitwise_equal(
        [1.0, float("nan")], [1.0, 2.0], "x"),
        "V02_ANCHOR_REPRODUCTION_MISMATCH", "non-finite measured")
    _expect_fail(lambda: require_bitwise_equal(
        [1.0], [1.0, 2.0], "x"),
        "V02_ANCHOR_REPRODUCTION_MISMATCH", "shape divergence")
    _expect_fail(lambda: require_bitwise_scalar(
        1.5, 1.5 + 1e-15, "x"),
        "V02_ANCHOR_REPRODUCTION_MISMATCH", "scalar divergence")
    _expect_fail(lambda: require_bitwise_scalar(
        float("inf"), 1.0, "x"),
        "V02_ANCHOR_REPRODUCTION_MISMATCH", "non-finite scalar")

    # f09: evaluation determinism -- the aggregate construction (the
    # only randomness is the fixed-seed PCG64(3) bootstrap) yields
    # bit-identical canonical bytes and semantic hashes.
    def _synth_aggregates() -> dict:
        per0 = [{k: [float(i + s) for i in range(1, 21)]
                 for k in METRIC_KEYS} for s in (0, 1, 2)]
        per1 = [{k: [float(i + s) + 0.5 for i in range(1, 21)]
                 for k in METRIC_KEYS} for s in (0, 1, 2)]
        ens0, agg0 = build_arm_aggregates(per0, None)
        ens1, agg1 = build_arm_aggregates(per1, ens0)
        return {"schema": FACTS_SCHEMA,
                "per_arm": [
                    {"arm": 0, "per_volume_ensemble": ens0,
                     "aggregates": agg0},
                    {"arm": 1, "per_volume_ensemble": ens1,
                     "aggregates": agg1}]}

    f1, f2 = _synth_aggregates(), _synth_aggregates()
    if canonical_json(f1) != canonical_json(f2):
        _fail("EVAL_SELFTEST_WRONG_VALUE",
              "f09 canonical bytes diverge across rebuilds")
    if _semantic_sha(f1) != _semantic_sha(f2):
        _fail("EVAL_SELFTEST_WRONG_VALUE",
              "f09 semantic hash diverges across rebuilds")
    logger.info("[%s] f09 PASS: evaluation determinism", __abbr__)

    # Run-set loading: the 12-pair ledger, attempt counting, the
    # (seed, arm) GLOBAL_ORDER convention, and refusal branches.
    def _write_ledger(d: Path, entries: list) -> None:
        d.mkdir(parents=True, exist_ok=True)
        lp = d / "v03_run_ledger.json"
        lp.write_bytes(canonical_json(
            {"schema": v03t.LEDGER_SCHEMA, "entries": entries}))
        lp.with_suffix(".json.sha256").write_text(
            f"{_file_sha256(lp)}  {lp.name}\n")

    run_dir = tmp / "runs"
    run_dir.mkdir()
    entries = []
    for seed, arm in v03t.GLOBAL_ORDER:
        status = "completed" if (arm, seed) == (0, 101) \
            else "numerical_failure"
        e = {"arm": arm, "seed": seed, "status": status,
             "error_code": None if status == "completed"
             else "U_NON_FINITE",
             "wall_clock_s": 1.0}
        if status == "completed":
            name = v03t.run_record_name(arm, seed, 1)
            rp = run_dir / name
            rp.write_bytes(canonical_json(
                {"schema": RUN_RECORD_SCHEMA, "arm": arm, "seed": seed,
                 "validation_history": [], "checkpoints": []}))
            rp.with_suffix(".json.sha256").write_text(
                f"{_file_sha256(rp)}  {rp.name}\n")
            e["record_file"] = name
            e["record_sha256"] = _file_sha256(rp)
        entries.append(e)
    entries.append(dict(entries[0]))     # a rerun attempt of pair 0
    _write_ledger(run_dir, entries)
    runs = load_run_set(run_dir)
    if set(runs) != {(a, s) for s, a in v03t.GLOBAL_ORDER}:
        _fail("EVAL_SELFTEST_WRONG_VALUE",
              "run-set keys diverge from the locked (arm, seed) set")
    if runs[(0, 101)]["status"] != "completed" \
            or runs[(0, 101)]["attempts"] != 2 \
            or runs[(0, 101)]["record"]["arm"] != 0 \
            or runs[(1, 101)]["status"] != "numerical_failure":
        _fail("EVAL_SELFTEST_WRONG_VALUE",
              f"run-set content {runs[(0, 101)]}")
    _expect_fail(lambda: load_run_set(tmp / "no_ledger"),
                 "RUN_SET_INCOMPLETE", "missing ledger file")

    def _copy_record(d: Path) -> None:
        name = v03t.run_record_name(0, 101, 1)
        (d / name).write_bytes((run_dir / name).read_bytes())
        (d / (name + ".sha256")).write_text(
            (run_dir / (name + ".sha256")).read_text())

    _write_ledger(tmp / "runs_missing",
                  [e for e in entries
                   if not (e["arm"] == 3 and e["seed"] == 303)])
    _copy_record(tmp / "runs_missing")
    _expect_fail(lambda: load_run_set(tmp / "runs_missing"),
                 "RUN_SET_INCOMPLETE", "unsettled pair")
    running = [dict(e) for e in entries]
    running[1]["status"] = "running"
    _write_ledger(tmp / "runs_running", running)
    _copy_record(tmp / "runs_running")
    _expect_fail(lambda: load_run_set(tmp / "runs_running"),
                 "RUN_SET_INCOMPLETE", "non-terminal pair")
    drift = [dict(e) for e in entries]
    for e in drift:
        if e["arm"] == 0 and e["seed"] == 101:
            e["record_sha256"] = "0" * 64
    _write_ledger(tmp / "runs_drift", drift)
    (tmp / "runs_drift" /
     v03t.run_record_name(0, 101, 1)).write_bytes(
        (run_dir / v03t.run_record_name(0, 101, 1)).read_bytes())
    _expect_fail(lambda: load_run_set(tmp / "runs_drift"),
                 "RUN_RECORD_INVALID", "record sha vs ledger pin")

    # Fail-fast orchestration: pass/pass, typed failure, unexpected
    # failure; later stages must be "not_run".
    results: list = []
    if not _execute_stages(results, [("s1", lambda: {"x": 1}),
                                     ("s2", lambda: {"y": 2})]):
        _fail("EVAL_SELFTEST_WRONG_VALUE", "all-pass orchestration")
    if [r["status"] for r in results] != ["pass", "pass"]:
        _fail("EVAL_SELFTEST_WRONG_VALUE",
              f"all-pass statuses {results}")
    results = []

    def _boom():
        raise V03EError("SELECTION_RULE_MISMATCH", "synthetic")

    if _execute_stages(results, [("s1", lambda: {}), ("s2", _boom),
                                 ("s3", lambda: {})]):
        _fail("EVAL_SELFTEST_WRONG_VALUE",
              "failing orchestration returned pass")
    if [r["status"] for r in results] != ["pass", "fail", "not_run"] \
            or results[1]["code"] != "SELECTION_RULE_MISMATCH":
        _fail("EVAL_SELFTEST_WRONG_VALUE",
              f"fail-fast statuses {results}")
    results = []

    def _generic():
        raise ValueError("synthetic")

    _execute_stages(results, [("s1", _generic), ("s2", lambda: {})])
    if results[0]["code"] != "ValueError" \
            or results[1]["status"] != "not_run":
        _fail("EVAL_SELFTEST_WRONG_VALUE",
              f"unexpected-error statuses {results}")
    logger.info("[%s] orchestration PASS: fail-fast + not_run",
                __abbr__)

    # Artifact writer: file + sidecar round-trip.
    digest = _write_with_sidecar(tmp / "out" / "rec.json",
                                 {"schema": "x/1", "v": 1})
    wp = tmp / "out" / "rec.json"
    if not wp.is_file() or _file_sha256(wp) != digest \
            or (tmp / "out" / "rec.json.sha256"
                ).read_text().split()[0] != digest:
        _fail("EVAL_SELFTEST_WRONG_VALUE", "sidecar chain")

    # f10: static audit over this module.
    v03t.static_audit([Path(__file__)])
    logger.info("[%s] f10 PASS: static audit clean", __abbr__)

    print(f"PASS {__abbr__} v{__version__} selftest (f05, f07, f08, "
          f"f09, f10)")
    return 0


# ---------------------------------------------------------------------------
# Changelog
# v0.2 (2026-10-06): review fixes. (1) Matched-seed completeness
#   enforced: ensemble_gate marks arms missing any locked seed
#   "incomplete"; aggregates emitted only for the full 3-seed
#   ensemble. (2) bootstrap_relative replicates divide by each
#   resample's own anchor mean. (3) holdout-state construction moved
#   into stage v02_anchor_reproduction (first split="val" contact).
# v0.1 (2026-10-06): initial implementation under V03SPEC v0.3 (163,
#   FROZEN) SS12-SS14 as amended by SEQREF-MRI-V03AMEND-A3 (166) and
#   V03PLAN v0.3 (164, FROZEN) SS9-SS11 + Amendments A1/A2/A3. Five
#   stages: setup + frozen inputs (first split="val" contact),
#   A3-deferred V02 R0/z=0 holdout anchor reproduction (bit-exact,
#   gates every V03 number), checkpoint selection (f05 ->
#   SELECTION_RULE_MISMATCH), 12-checkpoint holdout evaluation (NLL,
#   R0 z=0, R1 posterior mean over the registered Z_DIAG bank),
#   ensemble + bootstrap aggregates + seqref-v03-facts/1 (f07 verdict
#   refusal, f08 schema refusals, f09 determinism). Fail-fast stage
#   runner; seqref-v03-eval-record/1 + SHA-256 sidecars on every
#   artifact; PASS(0)/ERROR(2). Selftest: f05/f07/f08/f09/f10 + pure-
#   helper branches + run-set chain of custody + orchestration +
#   writer round-trip. No model, no data in fixtures.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
