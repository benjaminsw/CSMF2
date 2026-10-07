# SEQREF-V03G1 v0.2 -- scripts.v03_gate
# LIFETIME: KEEP
# =============================================================================
# Purpose: V02-parent reproduction gate (V03SPEC v0.3 SS17, article 163
#          FROZEN, as amended by SEQREF-MRI-V03AMEND-A3, article 166;
#          V03PLAN v0.3 SS6 item 8 / Amendment A3, article 164 FROZEN).
#          BLOCKING: no authoritative multi-seed training before this
#          gate passes (SPEC SS17).
# Checks (SPEC SS17 + Amendment A3; all zero-holdout-contact):
#   1. pipeline_bitexactness (fixture f01): the frozen V02 manifests are
#      rebuilt from data_root via the production builder and SHA-256
#      compared file-by-file; the p4_stats2 tensor parent is dual-pinned
#      via ffr.load_p4s2_parent. The rebuild reads dataset INDEX
#      METADATA only -- no k-space tensor is touched.
#   2a. checkpoint_identity (A3): the frozen v0.2 checkpoints at steps
#      0/3258 verify against the V02T train-record pins (file sha256 +
#      reloaded state sha256) through the V02E production loader.
#   2b. train_nll_bitexactness (A3): the full-train monitoring-freeze
#      NLL endpoints (step 0 / 3258) are recomputed through the V02E
#      production measurement (split="train", mode="eval") and compared
#      BIT-EXACTLY to the V02E facts anchors
#      endpoint_measurements.train.nll_step0 / nll_final.
#   3. lambda0_step_parity (SPEC SS17.3): one optimisation step on the
#      first epoch-0 batch (fixed batch, fixed registered init seed,
#      deterministic mode) through (a) the frozen V02 train_step path
#      and (b) the V03 lambda=0 path (v03l.combined_loss lam=0.0, the
#      untouched V02 NLL tensor per fixture f02); the loss values must
#      agree within 1e-6 relative.
# Deferred (Amendment A3): the R0/z=0 holdout anchor reproduction is
#   V03E scope (evaluation stage, where holdout contact is legal). This
#   module NEVER constructs a split="val" dataset.
# Failure of any check = pre-run validity abort (SPEC SS17/SS18
#   category 1): the gate record is written with the failed check and
#   every later check "not_run"; exit 2; never a scientific outcome.
# Environment-bound imports (tiny_gate, TDIAG, preflight parents) are
#   LAZY via _env(); pure logic stays importable without the data
#   environment so the selftest can exercise it in isolation.
# CONVENTION: logger.error + typed raise (V03G1Error). PASS(0)/ERROR(2).
#   No fallback, no mock, no placeholder, no silent pass.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from seqref_mri.src import free_flow_runtime as ffr
from seqref_mri.src.fastmri_data import FastMRISliceDataset
from seqref_mri.src.flows import v03_objective as v03l
from seqref_mri.scripts import v02_eval as v02e
from seqref_mri.scripts import v02_manifests as v02m
from seqref_mri.scripts.train_base import _collate, _prepare
from seqref_mri.scripts.train_free_flow import (targets_from_prepared,
                                                train_step)
from seqref_mri.scripts.v02_train import (LEARNING_RATE,
                                          derive_cmap_from_mask,
                                          state_sha256)
from seqref_mri.scripts.v02_manifests import (N_EPOCHS, V02Error,
                                            canonical_json)

logger = logging.getLogger("seqref_mri.v03_gate")

__version__ = "0.2"
__abbr__ = "SEQREF-V03G1"

GATE_SCHEMA = "seqref-v03-gate-record/1"
PARITY_TOL = 1e-6                    # locked (SPEC SS17.3)
REL_FLOOR = 1e-12                    # quoted; relative-denominator floor
EXPECTED_MANIFEST_FILES = (
    "v02_epoch0_manifest.json",
    "v02_epoch1_manifest.json",
    "v02_epoch2_manifest.json",
    "v02_holdout_manifest.json",
    "v02_d3_monitor_manifest.json",
)
_MANIFEST_KIND = {
    "v02_epoch0_manifest.json": "train_epoch",
    "v02_epoch1_manifest.json": "train_epoch",
    "v02_epoch2_manifest.json": "train_epoch",
    "v02_holdout_manifest.json": "holdout",
    "v02_d3_monitor_manifest.json": "d3_monitor",
}
# preflight semantic-hash scope (mirrors attach_semantic_hash: the
# payload excludes runtime/host fields and the self-referential fields).
_SEMANTIC_EXCLUDE = ("run", "semantic_sha256", "semantic_scope")


class V03G1Error(Exception):
    """Typed gate failure: .code is the machine-readable reason."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V03G1Error(code, message)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _register_src_path() -> Path:
    """The preflight modules import their siblings top-level (the host
    convention keeps src/ on sys.path); register the same path here so
    the gate is import-safe in any environment. Deterministic, not a
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
    """Extract the typed code prefix from a V02Error ("CODE: message");
    fall back to a gate-level code when the shape is unexpected."""
    head = str(exc).split(":", 1)[0].strip()
    if re.fullmatch(r"[A-Z0-9_]+", head):
        return head
    return fallback


# ---------------------------------------------------------------------------
# Pure verification helpers (selftest domain -- no data, no model)
# ---------------------------------------------------------------------------

def _verify_sidecar_file(path: Path) -> str:
    """Frozen-input integrity: file + sidecar exist, sha256 matches.
    Returns the verified digest."""
    side = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not side.is_file():
        _fail("GATE_INPUT_MISSING",
              f"frozen input or sidecar missing at {path}")
    recorded = side.read_text().split()[0]
    digest = _file_sha256(path)
    if digest != recorded:
        _fail("GATE_INPUT_DRIFT",
              f"{path.name}: sha256 {digest[:12]}... != sidecar "
              f"{recorded[:12]}...; frozen-input drift")
    return digest


def compare_manifest_sets(frozen: dict, rebuilt: dict) -> None:
    """f01: the rebuilt manifest set must equal the frozen set exactly
    (same file names, same sha256 each)."""
    if set(frozen) != set(rebuilt):
        _fail("MANIFEST_CONTENT_MISMATCH",
              f"manifest file set diverges: only-frozen "
              f"{sorted(set(frozen) - set(rebuilt))}, only-rebuilt "
              f"{sorted(set(rebuilt) - set(frozen))}")
    bad = [name for name in sorted(frozen)
           if frozen[name] != rebuilt[name]]
    if bad:
        _fail("MANIFEST_CONTENT_MISMATCH",
              f"rebuilt manifests diverge from the frozen files: {bad}")


def load_facts_anchors(path: Path, expected_n: int) -> dict:
    """Frozen V02E facts anchors for check 2b: sidecar integrity,
    schema gate, semantic-hash self-consistency, endpoint shape and
    finiteness. Returns the two anchor arrays."""
    _verify_sidecar_file(path)
    facts = json.loads(path.read_text())
    if facts.get("schema") != v02e.FACTS_SCHEMA:
        _fail("ANCHOR_RECORD_INVALID",
              f"facts schema {facts.get('schema')!r} != "
              f"{v02e.FACTS_SCHEMA!r}")
    payload = {k: v for k, v in facts.items()
               if k not in _SEMANTIC_EXCLUDE}
    if _canonical_hash()(payload) != facts.get("semantic_sha256"):
        _fail("ANCHOR_RECORD_INVALID",
              "facts semantic_sha256 does not match the recomputed "
              "content hash; the frozen anchors drifted")
    train = facts.get("endpoint_measurements", {}).get("train", {})
    return _validate_anchor_arrays(train, expected_n)


def _validate_anchor_arrays(train: dict, expected_n: int) -> dict:
    """Anchor shape/finiteness gate (pure): both endpoint arrays must
    be lists of expected_n finite values."""
    anchors = {}
    for key in ("nll_step0", "nll_final"):
        arr = train.get(key) if isinstance(train, dict) else None
        if not isinstance(arr, list) or len(arr) != expected_n:
            _fail("ANCHOR_RECORD_INVALID",
                  f"endpoint_measurements.train.{key} must be a list of "
                  f"{expected_n} per-slice values")
        a = np.asarray(arr, dtype=np.float64)
        if not np.all(np.isfinite(a)):
            _fail("ANCHOR_RECORD_INVALID",
                  f"endpoint_measurements.train.{key} carries a "
                  f"non-finite anchor value")
        anchors[key] = arr
    return anchors


def require_bitwise_equal(measured, anchor, what: str) -> None:
    """Bit-exact float64 reproduction (A3 check 2b): same shape, all
    finite, identical bytes."""
    m = np.asarray(measured, dtype=np.float64)
    a = np.asarray(anchor, dtype=np.float64)
    if m.shape != a.shape:
        _fail("ANCHOR_REPRODUCTION_MISMATCH",
              f"{what}: measured shape {m.shape} != anchor shape "
              f"{a.shape}")
    if not np.all(np.isfinite(m)):
        _fail("ANCHOR_REPRODUCTION_MISMATCH",
              f"{what}: the recomputed values carry a non-finite entry")
    if m.tobytes() != a.tobytes():
        idx = int(np.argmax(m != a))
        _fail("ANCHOR_REPRODUCTION_MISMATCH",
              f"{what}: first divergence at slice {idx}: measured "
              f"{m.flat[idx]!r} != anchor {a.flat[idx]!r}")


def require_relative_within(loss_v03: float, loss_v02: float) -> float:
    """SPEC SS17.3: |v03 - v02| / max(|v02|, REL_FLOOR) <= PARITY_TOL.
    Returns the relative difference."""
    if not (math.isfinite(loss_v03) and math.isfinite(loss_v02)):
        _fail("PARITY_NON_FINITE",
              f"parity losses must be finite, got v03={loss_v03!r} "
              f"v02={loss_v02!r}")
    rel = abs(loss_v03 - loss_v02) / max(abs(loss_v02), REL_FLOOR)
    if rel > PARITY_TOL:
        _fail("PARITY_TOLERANCE_EXCEEDED",
              f"lambda=0 step parity: relative loss difference "
              f"{rel:.3e} > {PARITY_TOL:.0e} (v03={loss_v03!r}, "
              f"v02={loss_v02!r})")
    return rel


# ---------------------------------------------------------------------------
# Production environment (lazy, total) and the four gate checks
# ---------------------------------------------------------------------------

def _env() -> SimpleNamespace:
    """Environment-bound dependencies, imported lazily and totally (the
    V02E convention): ImportError propagates as ERROR; there is no
    fallback path."""
    _register_src_path()
    try:
        from preflight_parents import StageError
        from seqref_mri.scripts import tiny_gate as tg
        from seqref_mri.tdiag import d2b
    except ImportError as exc:
        _fail("ENV_IMPORT_FAILED",
              f"an environment-bound dependency is not importable in "
              f"this environment: {exc}")
    return SimpleNamespace(
        torch=torch, DataLoader=DataLoader, Subset=Subset,
        StageError=StageError, ffr=ffr,
        FastMRISliceDataset=FastMRISliceDataset, tg=tg, collate=_collate,
        d2b=d2b, state_sha256=state_sha256)


def _epoch_manifests(manifests: list) -> list:
    """Select the train_epoch manifests in epoch order -- the exact
    list the V02E production loader verifies the train record against
    (v02_eval builds [_load_manifest(epoch{ep}, train_epoch) for ep in
    range(N_EPOCHS)]). The gate state carries all five frozen
    manifests; the train record pins exactly the N_EPOCHS epoch
    hashes, so the holdout/d3 manifests must not enter the
    comparison.
    """
    epoch = [m for m in manifests if m.get("kind") == "train_epoch"]
    if len(epoch) != N_EPOCHS:
        _fail("MANIFEST_SCHEMA_INVALID",
              f"expected {N_EPOCHS} train_epoch manifests, got "
              f"{len(epoch)}")
    return epoch


def _check_pipeline(env: SimpleNamespace, cfg: dict, state: dict,
                    work_root: Path) -> dict:
    """Check 1 (SPEC SS17.1, fixture f01): frozen manifest/tensor
    bit-exactness. Rebuilds the manifests from data_root through the
    production builder (index metadata only) and SHA-compares every
    file; dual-pins the p4_stats2 tensor parent."""
    man_dir = Path(cfg["manifest_dir"])
    frozen, manifests = {}, []
    for name in EXPECTED_MANIFEST_FILES:
        _verify_sidecar_file(man_dir / name)
        try:
            manifests.append(
                v02e._load_manifest(man_dir, name, _MANIFEST_KIND[name]))
        except V02Error as exc:
            _fail(_v02_code(exc, "MANIFEST_CONTENT_MISMATCH"),
                  f"{name}: {exc}")
        frozen[name] = _file_sha256(man_dir / name)
    tmp = Path(tempfile.mkdtemp(prefix="v03g1_manifests_",
                                dir=work_root))
    try:
        try:
            v02m.build_all(cfg["data_root"], tmp)
        except V02Error as exc:
            _fail("PIPELINE_REPRODUCTION_FAILURE", str(exc))
        rebuilt = {}
        for name in EXPECTED_MANIFEST_FILES:
            p = tmp / name
            if not p.is_file():
                _fail("MANIFEST_CONTENT_MISMATCH",
                      f"the production rebuild did not produce {name}")
            rebuilt[name] = _file_sha256(p)
        compare_manifest_sets(frozen, rebuilt)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    try:
        p4 = ffr.load_p4s2_parent(cfg["p4_stats2"])
    except Exception as exc:  # FreeFlowError/StageError family
        logger.error("[%s] PARENT_VERIFICATION_FAILED: %s: %s",
                     __abbr__, type(exc).__name__, exc)
        raise V03G1Error("PARENT_VERIFICATION_FAILED",
                         f"{type(exc).__name__}: {exc}") from exc
    state["p4"] = p4
    state["manifests"] = manifests
    return {"manifest_sha256": frozen,
            "p4_file_sha256": p4["file_sha256"],
            "p4_semantic_sha256": p4["semantic_sha256"]}


def _check_checkpoint_identity(env: SimpleNamespace, cfg: dict,
                               state: dict) -> dict:
    """Check 2a (Amendment A3): frozen-checkpoint identity. The V02T
    train record is hash-verified against the verified epoch manifests
    (the train_epoch subset, mirroring the V02E production loader);
    the
    step-0/3258 checkpoints verify against its pins through the V02E
    production loader (file sha256 + reloaded state sha256)."""
    try:
        record = v02e._load_train_record(cfg["train_record"],
                                         _epoch_manifests(
                                             state["manifests"]))
    except V02Error as exc:
        _fail(_v02_code(exc, "TRAIN_RECORD_MISMATCH"), str(exc))
    try:
        implb = ffr.load_implb_parent(cfg["implb_facts"])
        ffr.require_spline_b(implb["spline_b"])
    except Exception as exc:  # FreeFlowError/StageError family
        logger.error("[%s] PARENT_VERIFICATION_FAILED: %s: %s",
                     __abbr__, type(exc).__name__, exc)
        raise V03G1Error("PARENT_VERIFICATION_FAILED",
                         f"{type(exc).__name__}: {exc}") from exc
    ckpt_recs = {int(c["step"]): c for c in record["checkpoints"]}
    models, pins = {}, {}
    for step in (v02e.STEP0, v02e.FINAL_STEP):
        if step not in ckpt_recs:
            _fail("CHECKPOINT_HASH_MISMATCH",
                  f"the train record carries no checkpoint at step "
                  f"{step}")
        try:
            models[step] = v02e._load_checkpoint(
                env, Path(cfg["run_root"]), ckpt_recs[step],
                implb["spline_b"])
        except V02Error as exc:
            _fail("CHECKPOINT_HASH_MISMATCH", f"step {step}: {exc}")
        pins[str(step)] = {
            "file": ckpt_recs[step]["file"],
            "file_sha256": ckpt_recs[step]["file_sha256"],
            "state_sha256": ckpt_recs[step]["state_sha256"]}
    state["models"] = models
    state["record"] = record
    return {"checkpoint_pins": pins, "spline_b": implb["spline_b"]}


def _check_train_nll(env: SimpleNamespace, cfg: dict,
                     state: dict) -> dict:
    """Check 2b (Amendment A3): full-train monitoring-freeze NLL
    endpoints recomputed through the V02E production measurement
    (split="train", mode="eval") and compared bit-exactly to the V02E
    facts anchors. Zero holdout contact."""
    anchors = load_facts_anchors(Path(cfg["v02_facts"]),
                                 v02m.N_TRAIN_SLICES)
    try:
        s_ref = env.tg._s_ref_from_p0s(cfg["p0s_facts"])
    except env.StageError as exc:
        logger.error("[%s] PARENT_VERIFICATION_FAILED: %s: %s",
                     __abbr__, type(exc).__name__, exc)
        raise V03G1Error("PARENT_VERIFICATION_FAILED",
                         f"{type(exc).__name__}: {exc}") from exc
    try:
        ds = env.FastMRISliceDataset(cfg["data_root"], split="train",
                                     mode="eval")
    except Exception as exc:  # the dataset ctor is a typed, logging
        _fail("DATA_PREMISE_FAILURE", f"{type(exc).__name__}: {exc}")
    try:
        tr = v02e._measure_train_endpoints(
            env, state["models"][v02e.STEP0],
            state["models"][v02e.FINAL_STEP], ds,
            state["manifests"][0]["entries"], state["p4"], s_ref)
    except V02Error as exc:
        _fail("TRAIN_NLL_REPRODUCTION_FAILURE", str(exc))
    require_bitwise_equal(tr["nll0"], anchors["nll_step0"], "nll_step0")
    require_bitwise_equal(tr["nll1"], anchors["nll_final"], "nll_final")
    return {"n_train_slices": v02m.N_TRAIN_SLICES,
            "nll_step0_mean": float(np.mean(tr["nll0"])),
            "nll_final_mean": float(np.mean(tr["nll1"])),
            "anchor_source": "v02_facts endpoint_measurements.train "
                             "(bit-exact)"}


def _check_parity(env: SimpleNamespace, cfg: dict,
                  state: dict) -> dict:
    """Check 3 (SPEC SS17.3): one lambda=0 optimisation step on the
    first epoch-0 batch, fixed registered init seed, deterministic
    mode. Path (a) = frozen V02 train_step; path (b) = V03
    combined_loss at lam=0.0 (the untouched V02 NLL tensor, fixture
    f02) + backward + Adam step. Losses must agree within 1e-6
    relative."""
    env.torch.use_deterministic_algorithms(True)
    try:
        ds = env.FastMRISliceDataset(cfg["data_root"], split="train",
                                     mode="train")
    except Exception as exc:  # the dataset ctor is a typed, logging
        _fail("DATA_PREMISE_FAILURE", f"{type(exc).__name__}: {exc}")
    ds.set_epoch(0)                       # fixed epoch-0 fresh masks
    loader = DataLoader(ds, batch_size=v02m.BATCH_SIZE, shuffle=False,
                        num_workers=0, collate_fn=_collate)
    batch = next(iter(loader))
    prep = _prepare(batch, "cpu", test0=False)
    targets, cond_ins, masks = [], [], []
    for j in range(len(batch["meta"])):
        cmap = derive_cmap_from_mask(batch["mask"][j])
        vecs = ffr.standardisation_vectors(cmap,
                                           state["p4"]["location_index"])
        one = {"x_norm": prep["x_norm"][j:j + 1],
               "cond_in": prep["cond_in"][j:j + 1]}
        targets.append(targets_from_prepared(one, cmap, vecs))
        cond_ins.append(prep["cond_in"][j:j + 1])
        masks.append(batch["mask"][j:j + 1])
    tgt = torch.cat(targets, 0)
    cond = torch.cat(cond_ins, 0)
    msk = torch.cat(masks, 0)
    # (a) frozen V02 training path: train_step does backward + Adam.
    model_a = ffr.build_model(init_seed=ffr.MODEL_INIT_SEED)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=LEARNING_RATE)
    loss_a = float(train_step(model_a, opt_a, tgt, cond, msk))
    # (b) V03 lambda=0 path: identical init, untouched NLL tensor.
    model_b = ffr.build_model(init_seed=ffr.MODEL_INIT_SEED)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=LEARNING_RATE)
    nll_b = v03l.combined_loss(model_b, tgt, cond, msk, lam=0.0,
                               y_raw=prep["y"], amax=prep["amax"],
                               tgt_norm=prep["tgt_norm"], maps=[])
    opt_b.zero_grad(set_to_none=True)
    nll_b.backward()
    opt_b.step()
    loss_b = float(nll_b)
    rel = require_relative_within(loss_b, loss_a)
    return {"loss_v02_train_step": loss_a,
            "loss_v03_lambda0": loss_b,
            "relative_difference": rel,
            "tolerance": PARITY_TOL,
            "init_seed": ffr.MODEL_INIT_SEED,
            "deterministic_algorithms": True}


# ---------------------------------------------------------------------------
# Orchestration: fail-fast check runner, gate record, run, CLI
# ---------------------------------------------------------------------------

def _execute_checks(results: list, checks: list) -> bool:
    """Fail-fast runner: on the first failed check, record the failure
    (typed code, or the exception type for unexpected errors), mark
    every later check "not_run", and stop. Returns True only when every
    check passed."""
    for i, (cid, fn) in enumerate(checks):
        try:
            details = fn()
        except V03G1Error as exc:
            results.append({"check": cid, "status": "fail",
                            "code": exc.code, "message": str(exc)})
            for cid2, _fn2 in checks[i + 1:]:
                results.append({"check": cid2, "status": "not_run"})
            return False
        except Exception as exc:
            logger.exception("[%s] unexpected error in check %s",
                             __abbr__, cid)
            results.append({"check": cid, "status": "fail",
                            "code": type(exc).__name__,
                            "message": str(exc)})
            for cid2, _fn2 in checks[i + 1:]:
                results.append({"check": cid2, "status": "not_run"})
            return False
        results.append({"check": cid, "status": "pass",
                        "details": details})
    return True


def _write_record(out: Path, record: dict) -> str:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "v03_gate_record.json"
    path.write_bytes(canonical_json(record))
    digest = _file_sha256(path)
    path.with_suffix(".json.sha256").write_text(
        f"{digest}  {path.name}\n")
    logger.info("[%s] gate record written: %s (sha256 %s...)", __abbr__,
                path, digest[:12])
    return digest


def run(cfg: dict) -> int:
    """The gate run. cfg keys: data_root, manifest_dir, train_record,
    run_root, p4_stats2, implb_facts, p0s_facts, v02_facts, out_dir.
    Every scientific constant is locked in this module, the SPEC/PLAN,
    and the frozen inputs -- never in cfg."""
    env = _env()
    env.torch.set_num_threads(1)
    t0 = time.time()
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    state: dict = {}
    results: list = []
    checks = [
        ("pipeline_bitexactness",
         lambda: _check_pipeline(env, cfg, state, out)),
        ("checkpoint_identity",
         lambda: _check_checkpoint_identity(env, cfg, state)),
        ("train_nll_bitexactness",
         lambda: _check_train_nll(env, cfg, state)),
        ("lambda0_step_parity",
         lambda: _check_parity(env, cfg, state)),
    ]
    ok = _execute_checks(results, checks)
    man_dir = Path(cfg["manifest_dir"])
    record = {
        "schema": GATE_SCHEMA,
        "script": f"{__abbr__} v{__version__}",
        "spec": {"article": 163, "status": "FROZEN",
                 "amendments": ["A3 (article 166)"]},
        "plan": {"article": 164, "status": "FROZEN",
                 "amendments": ["A1", "A2", "A3"]},
        "inputs": {
            "manifest_dir": str(man_dir),
            "manifest_sha256": {n: _file_sha256(man_dir / n)
                                for n in EXPECTED_MANIFEST_FILES
                                if (man_dir / n).is_file()},
            "train_record": {"path": cfg["train_record"]},
            "run_root": cfg["run_root"],
            "p4_stats2": cfg["p4_stats2"],
            "implb_facts": cfg["implb_facts"],
            "p0s_facts": cfg["p0s_facts"],
            "v02_facts": {"path": cfg["v02_facts"]}},
        "checks": results,
        "overall": "PASS" if ok else "ERROR",
        "deferred_to_v03e": "R0/z=0 holdout anchor reproduction "
                            "(Amendment A3); this module never "
                            "constructs a split=\"val\" dataset",
        "wall_clock_s": time.time() - t0,
        "environment": {"python": sys.version.split()[0],
                        "torch": torch.__version__,
                        "platform": sys.platform},
    }
    _write_record(out, record)
    if not ok:
        failed = next(r for r in results if r["status"] == "fail")
        logger.error("[%s] GATE ERROR: check %s failed (%s); recorded "
                     "as a pre-run validity abort -- never a scientific "
                     "outcome (SPEC SS17/SS18)", __abbr__,
                     failed["check"], failed["code"])
        return 2
    print(f"PASS {__abbr__} v{__version__} gate: 4/4 checks "
          f"(pipeline bit-exact, checkpoint identity, train-NLL "
          f"bit-exact, lambda=0 step parity <= {PARITY_TOL:.0e})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"{__abbr__} v{__version__} -- V02-parent "
                    f"reproduction gate (V03SPEC SS17 + Amendment A3)")
    ap.add_argument("--mode", required=True,
                    choices=("selftest", "run"))
    ap.add_argument("--data-root")
    ap.add_argument("--manifest-dir")
    ap.add_argument("--train-record")
    ap.add_argument("--run-root")
    ap.add_argument("--p4-stats2")
    ap.add_argument("--implb-facts")
    ap.add_argument("--p0s-facts")
    ap.add_argument("--v02-facts")
    ap.add_argument("--out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.mode == "selftest":
        return _selftest()
    missing = [k for k in ("data_root", "manifest_dir", "train_record",
                           "run_root", "p4_stats2", "implb_facts",
                           "p0s_facts", "v02_facts", "out")
               if getattr(args, k) is None]
    if missing:
        logger.error("[%s] missing required arguments: %s", __abbr__,
                     ", ".join("--" + m.replace("_", "-")
                               for m in missing))
        return 2
    cfg = {k: getattr(args, k) for k in
           ("data_root", "manifest_dir", "train_record", "run_root",
            "p4_stats2", "implb_facts", "p0s_facts", "v02_facts")}
    cfg["out_dir"] = args.out
    try:
        return run(cfg)
    except V03G1Error as exc:
        logger.error("[%s] ERROR: %s", __abbr__, exc)
        return 2
    except Exception:
        logger.exception("[%s] unexpected exception", __abbr__)
        return 2


# ---------------------------------------------------------------------------
# Selftest: fixture f01 (manifest/pipeline bit-exactness), the pure
# verification helpers (sidecar layer, facts anchors, bitwise/relative
# comparators), the fail-fast orchestration, and the f10 static audit.
# No model, no data.
# ---------------------------------------------------------------------------

def _expect_fail(fn, code: str, label: str) -> None:
    caught = None
    try:
        fn()
    except V03G1Error as exc:
        caught = exc
    if caught is None:
        _fail("PROBE_SELFTEST_WRONG_VALUE",
              f"{label}: expected {code}, no error raised")
    if caught.code != code:
        _fail("PROBE_SELFTEST_WRONG_CODE",
              f"{label}: expected {code}, got {caught.code}")
    logger.info("[%s] branch PASS: %s -> %s", __abbr__, label, code)


def _selftest() -> int:
    import tempfile as _tf

    tmp = Path(_tf.mkdtemp(prefix="v03g1_selftest_"))

    # f01: manifest-set bit-exactness comparison.
    frozen = {"a.json": "0" * 64, "b.json": "1" * 64}
    compare_manifest_sets(frozen, dict(frozen))
    drifted = dict(frozen)
    drifted["b.json"] = "2" * 64
    _expect_fail(lambda: compare_manifest_sets(frozen, drifted),
                 "MANIFEST_CONTENT_MISMATCH", "f01 content drift")
    _expect_fail(lambda: compare_manifest_sets(
        frozen, {"a.json": "0" * 64}),
        "MANIFEST_CONTENT_MISMATCH", "f01 missing rebuilt file")
    logger.info("[%s] f01 PASS: manifest-set bit-exactness", __abbr__)

    # Regression (v0.2): epoch-manifest selection mirrors the V02E
    # production loader; the train record pins exactly N_EPOCHS epoch
    # hashes, never the holdout/d3 manifests (gate attempt-1 abort).
    five = ([{"kind": "train_epoch", "manifest_sha256": f"e{i}"}
             for i in range(N_EPOCHS)]
            + [{"kind": "holdout", "manifest_sha256": "h"},
               {"kind": "d3_monitor", "manifest_sha256": "d"}])
    if [m["manifest_sha256"] for m in _epoch_manifests(five)] != \
            [f"e{i}" for i in range(N_EPOCHS)]:
        _fail("PROBE_SELFTEST_WRONG_VALUE",
              "epoch selection must keep the train_epoch order exactly")
    _expect_fail(lambda: _epoch_manifests(five[:N_EPOCHS - 1]),
                 "MANIFEST_SCHEMA_INVALID", "epoch count guard")
    logger.info("[%s] epoch-selection regression PASS", __abbr__)

    # Sidecar layer: missing input, missing sidecar, drift, ok.
    good = tmp / "good.json"
    good.write_bytes(canonical_json({"schema": "x/1", "v": 1}))
    good.with_suffix(".json.sha256").write_text(
        f"{_file_sha256(good)}  {good.name}\n")
    _verify_sidecar_file(good)
    _expect_fail(lambda: _verify_sidecar_file(tmp / "absent.json"),
                 "GATE_INPUT_MISSING", "missing input")
    noside = tmp / "noside.json"
    noside.write_bytes(canonical_json({"v": 2}))
    _expect_fail(lambda: _verify_sidecar_file(noside),
                 "GATE_INPUT_MISSING", "missing sidecar")
    badside = tmp / "badside.json"
    badside.write_bytes(canonical_json({"v": 3}))
    badside.with_suffix(".json.sha256").write_text(f"{'0' * 64}\n")
    _expect_fail(lambda: _verify_sidecar_file(badside),
                 "GATE_INPUT_DRIFT", "sidecar drift")

    # Facts anchors: happy path + every refusal branch.
    def _facts_file(name: str, mutate=None, semantic_ok: bool = True,
                    sidecar_ok: bool = True) -> Path:
        facts = {"schema": v02e.FACTS_SCHEMA,
                 "endpoint_measurements": {
                     "train": {"nll_step0": [1.0, 2.0],
                               "nll_final": [3.0, 4.0]}},
                 "run": {"utc": "synthetic"}}
        if mutate is not None:
            mutate(facts)
        payload = {k: v for k, v in facts.items()
                   if k not in _SEMANTIC_EXCLUDE}
        facts["semantic_sha256"] = (
            _canonical_hash()(payload) if semantic_ok else "0" * 64)
        p = tmp / name
        p.write_bytes(canonical_json(facts))
        digest = _file_sha256(p) if sidecar_ok else "0" * 64
        p.with_suffix(".json.sha256").write_text(
            f"{digest}  {p.name}\n")
        return p

    anchors = load_facts_anchors(_facts_file("facts_ok.json"), 2)
    if anchors["nll_step0"] != [1.0, 2.0] \
            or anchors["nll_final"] != [3.0, 4.0]:
        _fail("PROBE_SELFTEST_WRONG_VALUE", "facts anchor round-trip")
    _expect_fail(lambda: load_facts_anchors(
        _facts_file("facts_schema.json",
                    lambda f: f.update(schema="wrong/9")), 2),
        "ANCHOR_RECORD_INVALID", "facts wrong schema")
    _expect_fail(lambda: load_facts_anchors(
        _facts_file("facts_semantic.json", semantic_ok=False), 2),
        "ANCHOR_RECORD_INVALID", "facts semantic drift")
    _expect_fail(lambda: load_facts_anchors(
        _facts_file("facts_sidecar.json", sidecar_ok=False), 2),
        "GATE_INPUT_DRIFT", "facts sidecar drift")
    _expect_fail(lambda: load_facts_anchors(
        _facts_file("facts_len.json"), 3),
        "ANCHOR_RECORD_INVALID", "facts wrong length")
    # The finiteness/shape gate, unit-tested directly (a semantically
    # valid facts file can never carry NaN: canonical_hash forbids it).
    _expect_fail(lambda: _validate_anchor_arrays(
        {"nll_step0": [1.0, float("nan")], "nll_final": [3.0, 4.0]}, 2),
        "ANCHOR_RECORD_INVALID", "non-finite anchor value")
    _expect_fail(lambda: _validate_anchor_arrays({}, 2),
        "ANCHOR_RECORD_INVALID", "missing endpoint arrays")

    # Bit-exact comparator (A3 check 2b).
    require_bitwise_equal([1.0, 2.0], [1.0, 2.0], "x")
    _expect_fail(lambda: require_bitwise_equal(
        [1.0, math.nextafter(2.0, 3.0)], [1.0, 2.0], "x"),
        "ANCHOR_REPRODUCTION_MISMATCH", "one-ulp divergence")
    _expect_fail(lambda: require_bitwise_equal(
        [1.0, float("nan")], [1.0, 2.0], "x"),
        "ANCHOR_REPRODUCTION_MISMATCH", "non-finite measured")
    _expect_fail(lambda: require_bitwise_equal(
        [1.0, 2.0, 3.0], [1.0, 2.0], "x"),
        "ANCHOR_REPRODUCTION_MISMATCH", "shape divergence")

    # Relative comparator (SPEC SS17.3).
    if require_relative_within(1.0, 1.0) != 0.0:
        _fail("PROBE_SELFTEST_WRONG_VALUE", "identical losses rel != 0")
    require_relative_within(1.0 + 5e-7, 1.0)      # inside tolerance
    _expect_fail(lambda: require_relative_within(1.0 + 2e-6, 1.0),
                 "PARITY_TOLERANCE_EXCEEDED", "tolerance exceeded")
    _expect_fail(lambda: require_relative_within(
        float("nan"), 1.0),
        "PARITY_NON_FINITE", "non-finite parity loss")
    _expect_fail(lambda: require_relative_within(1e-13, 0.0),
                 "PARITY_TOLERANCE_EXCEEDED", "zero-anchor floor")
    if require_relative_within(0.0, 0.0) != 0.0:
        _fail("PROBE_SELFTEST_WRONG_VALUE", "zero/zero parity")

    # Fail-fast orchestration: pass/pass, typed failure, unexpected
    # failure; later checks must be "not_run".
    results: list = []
    if not _execute_checks(results, [("c1", lambda: {"x": 1}),
                                     ("c2", lambda: {"y": 2})]):
        _fail("PROBE_SELFTEST_WRONG_VALUE", "all-pass orchestration")
    if [r["status"] for r in results] != ["pass", "pass"]:
        _fail("PROBE_SELFTEST_WRONG_VALUE",
              f"all-pass statuses {results}")
    results = []

    def _boom():
        raise V03G1Error("MANIFEST_CONTENT_MISMATCH", "synthetic")

    if _execute_checks(results, [("c1", lambda: {"x": 1}),
                                 ("c2", _boom), ("c3", lambda: {})]):
        _fail("PROBE_SELFTEST_WRONG_VALUE",
              "failing orchestration returned pass")
    if [r["status"] for r in results] != ["pass", "fail", "not_run"] \
            or results[1]["code"] != "MANIFEST_CONTENT_MISMATCH":
        _fail("PROBE_SELFTEST_WRONG_VALUE",
              f"fail-fast statuses {results}")
    results = []

    def _generic():
        raise ValueError("synthetic")

    _execute_checks(results, [("c1", _generic), ("c2", lambda: {})])
    if results[0]["code"] != "ValueError" \
            or results[1]["status"] != "not_run":
        _fail("PROBE_SELFTEST_WRONG_VALUE",
              f"unexpected-error statuses {results}")
    logger.info("[%s] orchestration PASS: fail-fast + not_run",
                __abbr__)

    # Gate record writer: file + sidecar round-trip.
    rec = {"schema": GATE_SCHEMA, "overall": "PASS",
           "checks": [{"check": "c1", "status": "pass",
                       "details": {"x": 1}}]}
    digest = _write_record(tmp / "gate_out", rec)
    written = tmp / "gate_out" / "v03_gate_record.json"
    side = (tmp / "gate_out" / "v03_gate_record.json.sha256")
    if not written.is_file() or not side.is_file() \
            or side.read_text().split()[0] != digest \
            or _file_sha256(written) != digest:
        _fail("PROBE_SELFTEST_WRONG_VALUE", "record sidecar chain")

    # f10: static audit over this module.
    from seqref_mri.scripts import v03_train as v03t
    v03t.static_audit([Path(__file__)])
    logger.info("[%s] f10 PASS: static audit clean", __abbr__)

    print(f"PASS {__abbr__} v{__version__} selftest (f01, helpers, "
          f"epoch-selection, orchestration, f10)")
    return 0


# ---------------------------------------------------------------------------
# Changelog
# v0.1 (2026-10-06): initial implementation under V03SPEC v0.3 (163,
#   FROZEN) SS17 as amended by SEQREF-MRI-V03AMEND-A3 (166) and V03PLAN
#   v0.3 (164, FROZEN) Amendment A3 (SS6 item 8). Four checks: pipeline
#   bit-exactness (fixture f01), checkpoint identity (A3 check 2a),
#   train-NLL bit-exactness (A3 check 2b), lambda=0 step parity within
#   1e-6 relative (SS17.3). Fail-fast validity-abort gate record
#   (seqref-v03-gate-record/1) + SHA-256 sidecar; zero holdout contact;
#   R0/z=0 holdout anchor reproduction deferred to V03E per A3.
#   Selftest: f01 + pure-helper branches + fail-fast orchestration +
#   f10 static audit. No model, no data in fixtures.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
