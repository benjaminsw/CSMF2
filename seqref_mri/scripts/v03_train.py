# SEQREF-V03T v0.4 -- scripts.v03_train
# LIFETIME: KEEP
# =============================================================================
# Purpose: V03 training driver + append-only run ledger (V03SPEC v0.3
#          SS7/SS18, article 163 FROZEN; V03PLAN v0.3 SS8, article 164
#          FROZEN + Amendments A1/A2). ONE (arm, seed, attempt) per invocation;
#          the locked seed-major order (per seed: 0 -> l1 -> l2 -> l3) is
#          driven externally and enforced by the ledger, never by memory.
# Locked quantities (quoted, never CLI-tunable):
#   * arms {0,1,2,3} with lambda from the frozen probe record
#     (arm 0 -> 0.0; arms 1/2/3 -> lambdas at alpha 0.1/0.5/0.9);
#     seeds {101, 202, 303}; fresh init per run via
#     ffr.build_model(init_seed=seed)
#   * inherited V02 config: Adam lr=1e-4 constant, batch 32 (partial
#     batch kept), 3 epochs x 1086 = 3258 steps, float32 CPU, no
#     scheduler, no clipping; non-finite loss/gradients = NUMERICAL
#     failure, never a silent retry
#   * checkpoints at every epoch end (3 per run) + SHA-256 each; V03T
#     never selects a winner -- V03E applies the min-L_val rule (f05)
#   * validation: first 100 batches (Amendment A2: split="train",
#     mode="eval"), forward-only, per epoch; L_val = val_NLL +
#     lam * val_L_rec
#   * rerun policy (SPEC SS18): infrastructure interruption -> max 3
#     attempts total, full restart same seed; numerical failure and
#     failed_validity are TERMINAL, no restart; completed -> no restart.
#     Ledger is append-only and sidecar-chained; every attempt writes
#     attempt-scoped evidence files (no overwrite of prior attempts);
#     the locked seed-major global order is enforced via
#     expected_next_run (an interrupted pair blocks advance until
#     retried; a terminal pair advances the chain).
# Telemetry (reviewer-locked): per step -- step, epoch, L_density,
#   L_rec, lam*L_rec, L_total, combined gradient norm, lr. Per epoch --
#   val_NLL, val_L_rec, L_val, checkpoint hash, wall clock. Per run --
#   arm, lambda, seed, attempt, steps, status, environment.
#   ASYMMETRY (locked telemetry rule): per-step L_rec is recorded only
#   for lam > 0 (the loss computes it); for lam = 0 the R0 decode is not
#   executed per training step (cost) and the epoch-level val_L_rec
#   supplies the L_rec trajectory. Per-term gradient telemetry lives in
#   V03P -- V03T never runs separate per-term backward passes.
# Emits per run: v03_arm{A}_seed{S}_attempt{N}_train_record.json (schema
#   seqref-v03-train-record/1) + SHA-256 sidecar; ledger
#   v03_run_ledger.json (schema seqref-v03-run-ledger/1) + sidecar.
# CONVENTION: logger.error + typed raise (V03TError). PASS(0)/ERROR(2).
#   No fallback, no mock, no placeholder, no silent pass.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from seqref_mri.src import free_flow_runtime as ffr
from seqref_mri.src.fastmri_data import FastMRISliceDataset
from seqref_mri.src.flows import v03_objective as v03l
from seqref_mri.scripts import train_free_flow as tff
from seqref_mri.scripts.train_base import _collate, _prepare
from seqref_mri.scripts.train_free_flow import targets_from_prepared
from seqref_mri.scripts.v02_manifests import (BATCH_SIZE, N_EPOCHS,
                                              N_TRAIN_SLICES,
                                              canonical_json)
from seqref_mri.scripts.v02_train import derive_cmap_from_mask
from seqref_mri.scripts.v03_probe import (V03PError,
                                          global_grad_norm)

logger = logging.getLogger("seqref_mri.v03_train")

__version__ = "0.4"
__abbr__ = "SEQREF-V03T"

LEARNING_RATE = 1e-4                      # locked (V03PLAN SS8)
TOTAL_STEPS = 3258                        # 3 x ceil(34742/32), locked
STEPS_PER_EPOCH = 1086                    # locked
N_VAL_BATCHES = 100                       # locked (V03PLAN SS8 cadence)
ARMS = (0, 1, 2, 3)                       # locked (SPEC SS7)
SEEDS = (101, 202, 303)                   # locked (SPEC SS7)
ALPHA_KEYS = {1: "0.1", 2: "0.5", 3: "0.9"}   # locked (SPEC SS9)
MAX_ATTEMPTS = 3                          # locked (SPEC SS18: 2 reruns)
LEDGER_SCHEMA = "seqref-v03-run-ledger/1"
RECORD_SCHEMA = "seqref-v03-train-record/1"
PROBE_SCHEMA = "seqref-v03-probe-record/1"


class V03TError(RuntimeError):
    """Single typed identity for the V03 training driver."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V03TError(code, message)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_with_sidecar(path: Path, payload) -> str:
    """Write canonical bytes (or text) + SHA-256 sidecar; return digest."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    path.write_bytes(payload)
    digest = _file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n")
    return digest


# ---------------------------------------------------------------------------
# Static audit (fixture f10): forbidden patterns in the V03 modules.
# Tokens are assembled so this audit's own source never self-triggers.
# ---------------------------------------------------------------------------

FORBIDDEN_PATTERNS = (
    re.compile(r"except[^\n:]*:\s*pass\b"),
    re.compile(r"\b" + "TO" + "DO" + r"\b"),
    re.compile(r"\b" + "FIX" + "ME" + r"\b"),
)


def static_audit(paths) -> None:
    """Scan module sources for forbidden patterns (swallowing except-
    blocks with a bare pass body; deferred-work marker tokens). Any hit
    -> AUDIT_PATTERN_FOUND."""
    for p in paths:
        src = Path(p).read_text()
        for pat in FORBIDDEN_PATTERNS:
            m = pat.search(src)
            if m:
                _fail("AUDIT_PATTERN_FOUND",
                      f"{Path(p).name}: forbidden pattern {pat.pattern!r} "
                      f"matched {m.group(0)!r}")


# ---------------------------------------------------------------------------
# Run contract: frozen probe record + arm/seed validation
# ---------------------------------------------------------------------------

def load_probe_record(path: str) -> dict:
    """Hash-verified frozen probe record; schema and locked-field gates."""
    p = Path(path)
    side = p.with_suffix(p.suffix + ".sha256")
    if not p.is_file() or not side.is_file():
        _fail("PROBE_RECORD_MISSING",
              f"probe record or sidecar missing at {p}")
    recorded = side.read_text().split()[0]
    digest = _file_sha256(p)
    if digest != recorded:
        _fail("PROBE_RECORD_HASH_MISMATCH",
              f"probe record sha256 {digest[:12]}… != sidecar "
              f"{recorded[:12]}…; the frozen probe evidence drifted")
    rec = json.loads(p.read_text())
    if rec.get("schema") != PROBE_SCHEMA:
        _fail("PROBE_RECORD_SCHEMA_MISMATCH",
              f"schema {rec.get('schema')!r} != {PROBE_SCHEMA!r}")
    if int(rec.get("n_batches", -1)) != 100 \
            or int(rec.get("batch_size", -1)) != BATCH_SIZE:
        _fail("PROBE_RECORD_CONFIG_MISMATCH",
              "probe record n_batches/batch_size diverge from the locked "
              "100/32 contract")
    if rec.get("split") != "train" or rec.get("mode") != "eval" \
            or rec.get("validation_source_amendment") != "A2":
        _fail("PROBE_RECORD_CONFIG_MISMATCH",
              "probe record validation source diverges from Amendment A2 "
              '(split="train", mode="eval")')
    lambdas = rec.get("lambdas")
    if not isinstance(lambdas, dict) \
            or set(lambdas) != {"0.1", "0.5", "0.9"}:
        _fail("PROBE_RECORD_LAMBDAS_INVALID",
              f"lambdas field must cover exactly the locked alpha grid, "
              f"got {None if lambdas is None else sorted(lambdas)}")
    for k, v in lambdas.items():
        if not np.isfinite(v) or v <= 0 or not (1e-6 <= v <= 1e6):
            _fail("PROBE_RECORD_LAMBDAS_INVALID",
                  f"lambda[{k}]={v!r} violates the locked validity band; "
                  f"the probe record is not a valid run contract")
    return rec


def validate_arm_seed(arm: int, seed: int) -> None:
    if arm not in ARMS:
        _fail("ARM_INVALID", f"arm {arm!r} not in locked set {ARMS}")
    if seed not in SEEDS:
        _fail("SEED_INVALID", f"seed {seed!r} not in locked set {SEEDS}")


def lambda_for_arm(rec: dict, arm: int) -> float:
    """Arm 0 -> 0.0 (density-only anchor); arms 1/2/3 -> probe lambdas."""
    if arm == 0:
        return 0.0
    return float(rec["lambdas"][ALPHA_KEYS[arm]])


# ---------------------------------------------------------------------------
# Locked seed-major global order + attempt-scoped evidence naming
# ---------------------------------------------------------------------------

GLOBAL_ORDER = [(s, a) for s in SEEDS for a in ARMS]   # seed-major, locked
TERMINAL_STATUSES = ("completed", "numerical_failure", "failed_validity")


def run_record_name(arm: int, seed: int, attempt: int) -> str:
    return f"v03_arm{arm}_seed{seed}_attempt{attempt}_train_record.json"


def checkpoint_name(arm: int, seed: int, attempt: int, epoch: int) -> str:
    return f"v03_arm{arm}_seed{seed}_attempt{attempt}_epoch{epoch}.pt"


def expected_next_run(ledger: dict):
    """The single runnable (arm, seed) under the locked seed-major order.
    A pair is settled once it holds a TERMINAL entry (completed or a
    terminal failure); an interrupted pair blocks advancement until it is
    retried to a terminal state. Returns (arm, seed) or None if all 12
    pairs are settled."""
    for seed, arm in GLOBAL_ORDER:
        entries = [e for e in ledger["entries"]
                   if e["seed"] == seed and e["arm"] == arm]
        if not any(e["status"] in TERMINAL_STATUSES for e in entries):
            return arm, seed
    return None


def enforce_order(ledger: dict, arm: int, seed: int) -> None:
    nxt = expected_next_run(ledger)
    if nxt is None:
        _fail("ORDER_VIOLATION",
              "all 12 arm-seed pairs are settled; no further runs are "
              "permitted")
    if nxt != (arm, seed):
        _fail("ORDER_VIOLATION",
              f"locked seed-major order: the next runnable pair is arm "
              f"{nxt[0]} seed {nxt[1]}, not arm {arm} seed {seed}")


def classify_objective_error(code: str) -> str:
    """Map a V03L objective failure onto the ledger status taxonomy:
    U_NON_FINITE (non-finite unpacked target/readout) -> numerical_failure
    (instability evidence); LOSS_CONTRACT_VIOLATION and anything else ->
    failed_validity (structural contract breach, not instability). Both
    are TERMINAL -- neither permits a rerun."""
    if code == "U_NON_FINITE":
        return "numerical_failure"
    return "failed_validity"


def classify_gradient_error(code: str) -> str:
    """V03P gradient-norm failures are structural coverage failures (no
    parameter carries a gradient after backward): always failed_validity
    -- never numerical instability, never infrastructure. TERMINAL."""
    return "failed_validity"


# ---------------------------------------------------------------------------
# Append-only run ledger (sidecar-chained)
# ---------------------------------------------------------------------------

def load_ledger(path: Path):
    if not path.exists():
        return {"schema": LEDGER_SCHEMA, "entries": []}
    side = path.with_suffix(path.suffix + ".sha256")
    if not side.is_file():
        _fail("LEDGER_TAMPERED",
              f"ledger exists at {path} without its sidecar; append-only "
              f"chain broken")
    recorded = side.read_text().split()[0]
    if _file_sha256(path) != recorded:
        _fail("LEDGER_TAMPERED",
              "ledger content diverges from its sidecar; append-only "
              "chain broken")
    led = json.loads(path.read_text())
    if led.get("schema") != LEDGER_SCHEMA \
            or not isinstance(led.get("entries"), list):
        _fail("LEDGER_SCHEMA_MISMATCH",
              f"ledger schema {led.get('schema')!r} != {LEDGER_SCHEMA!r}")
    return led


def next_attempt(ledger: dict, arm: int, seed: int) -> int:
    """Rerun policy (SPEC SS18): completed/numerical_failure -> no
    restart; interruptions -> up to MAX_ATTEMPTS total, same seed."""
    prior = [e for e in ledger["entries"]
             if e["arm"] == arm and e["seed"] == seed]
    for e in prior:
        if e["status"] == "completed":
            _fail("RERUN_POLICY_VIOLATION",
                  f"arm {arm} seed {seed} already completed (attempt "
                  f"{e['attempt']}); no restart is permitted")
        if e["status"] == "numerical_failure":
            _fail("RERUN_POLICY_VIOLATION",
                  f"arm {arm} seed {seed} recorded a numerical failure "
                  f"(attempt {e['attempt']}); instability evidence is "
                  f"terminal, no restart is permitted")
    if len(prior) >= MAX_ATTEMPTS:
        _fail("RERUN_POLICY_VIOLATION",
              f"arm {arm} seed {seed} already used {len(prior)} attempts; "
              f"the locked maximum is {MAX_ATTEMPTS}")
    return len(prior) + 1


def append_entry(ledger_path: Path, entry: dict) -> str:
    """Append one entry, preserving every prior entry byte-identically
    (canonical JSON), and re-chain the sidecar."""
    led = load_ledger(ledger_path)
    led["entries"].append(entry)
    return _write_with_sidecar(ledger_path, canonical_json(led))


# ---------------------------------------------------------------------------
# Validation (forward-only, no updates)
# ---------------------------------------------------------------------------

def _validate(model, val_loader, p4, lam: float) -> dict:
    """Forward-only validation (Amendment A2 source: split="train",
    mode="eval" -- NEVER the 199-volume holdout). Explicit eval mode;
    training mode restored on exit."""
    nlls = np.empty(N_VAL_BATCHES, dtype=np.float64)
    recs = np.empty(N_VAL_BATCHES, dtype=np.float64)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if bi >= N_VAL_BATCHES:
                    break
                prep = _prepare(batch, "cpu", test0=False)
                targets, cond_ins, masks, maps = [], [], [], []
                for j in range(len(batch["meta"])):
                    cmap = derive_cmap_from_mask(batch["mask"][j])
                    vecs = ffr.standardisation_vectors(
                        cmap, p4["location_index"])
                    one = {"x_norm": prep["x_norm"][j:j + 1],
                           "cond_in": prep["cond_in"][j:j + 1]}
                    targets.append(targets_from_prepared(one, cmap, vecs))
                    cond_ins.append(prep["cond_in"][j:j + 1])
                    masks.append(batch["mask"][j:j + 1])
                    maps.append((cmap, vecs))
                tgt = torch.cat(targets, 0)
                cond = torch.cat(cond_ins, 0)
                msk = torch.cat(masks, 0)
                nlls[bi] = float(tff.nll_objective(model, tgt, cond, msk))
                recs[bi] = float(v03l.l_rec(model, cond, msk, prep["y"],
                                            prep["amax"], prep["tgt_norm"],
                                            maps))
    finally:
        if was_training:
            model.train()
    val_nll = float(nlls.mean())
    val_rec = float(recs.mean())
    if not (np.isfinite(val_nll) and np.isfinite(val_rec)):
        _fail("NUMERICAL_FAILURE",
              f"validation produced non-finite values (val_NLL={val_nll!r}"
              f", val_L_rec={val_rec!r})")
    return {"val_NLL": val_nll, "val_L_rec": val_rec,
            "L_val": val_nll + lam * val_rec}


# ---------------------------------------------------------------------------
# Single-run trainer: ONE (arm, seed, attempt)
# ---------------------------------------------------------------------------

def _run(data_root: str, p4_stats2: str, probe_record_path: str,
         out_dir: str, arm: int, seed: int) -> int:
    validate_arm_seed(arm, seed)
    rec_probe = load_probe_record(probe_record_path)
    lam = lambda_for_arm(rec_probe, arm)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / "v03_run_ledger.json"
    enforce_order(load_ledger(ledger_path), arm, seed)
    attempt = next_attempt(load_ledger(ledger_path), arm, seed)
    logger.info("[%s] arm %d (lambda=%.6g) seed %d attempt %d/%d",
                __abbr__, arm, lam, seed, attempt, MAX_ATTEMPTS)

    p4 = ffr.load_p4s2_parent(p4_stats2)
    model = ffr.build_model(init_seed=seed)     # fresh init, arm-seed
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ds_train = FastMRISliceDataset(data_root, split="train", mode="train")
    # Amendment A2: validation = train split in eval mode (deterministic
    # masks, training distribution). The official val split IS the
    # 199-volume holdout and is never touched before frozen evaluation.
    ds_valid = FastMRISliceDataset(data_root, split="train", mode="eval")
    val_loader = DataLoader(ds_valid, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0,
                            collate_fn=_collate)
    if len(val_loader) < N_VAL_BATCHES:
        _fail("VALIDATION_DATA_SHORT",
              f"validation split yields {len(val_loader)} batches; the "
              f"locked validation requires {N_VAL_BATCHES}")

    history, val_history, checkpoints = [], [], []
    step = 0
    t0 = time.time()
    status, error_code = "completed", None
    try:
        for ep in range(N_EPOCHS):
            ds_train.set_epoch(ep)
            loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=0,
                                collate_fn=_collate)
            for batch in loader:
                prep = _prepare(batch, "cpu", test0=False)
                targets, cond_ins, masks, maps = [], [], [], []
                for j in range(len(batch["meta"])):
                    cmap = derive_cmap_from_mask(batch["mask"][j])
                    vecs = ffr.standardisation_vectors(
                        cmap, p4["location_index"])
                    one = {"x_norm": prep["x_norm"][j:j + 1],
                           "cond_in": prep["cond_in"][j:j + 1]}
                    targets.append(targets_from_prepared(one, cmap, vecs))
                    cond_ins.append(prep["cond_in"][j:j + 1])
                    masks.append(batch["mask"][j:j + 1])
                    maps.append((cmap, vecs))
                tgt = torch.cat(targets, 0)
                cond = torch.cat(cond_ins, 0)
                msk = torch.cat(masks, 0)
                step += 1
                opt.zero_grad(set_to_none=True)
                nll = tff.nll_objective(model, tgt, cond, msk)
                if lam == 0.0:
                    loss = nll                    # untouched V02 path
                    l_rec_val, lam_rec_val = None, None
                else:
                    rec = v03l.l_rec(model, cond, msk, prep["y"],
                                     prep["amax"], prep["tgt_norm"], maps)
                    loss = nll + lam * rec
                    l_rec_val, lam_rec_val = float(rec), float(lam * rec)
                if not torch.isfinite(loss):
                    status, error_code = "numerical_failure", \
                        "NUMERICAL_FAILURE"
                    _fail("NUMERICAL_FAILURE",
                          f"step {step}: combined objective non-finite; "
                          f"instability evidence, no restart")
                loss.backward()
                gnorm = global_grad_norm(model)
                if not np.isfinite(gnorm):
                    status, error_code = "numerical_failure", \
                        "NUMERICAL_FAILURE"
                    _fail("NUMERICAL_FAILURE",
                          f"step {step}: combined gradient norm non-finite"
                          f"; instability evidence, no restart")
                opt.step()
                history.append({
                    "step": step, "epoch": ep,
                    "L_density": float(nll), "L_rec": l_rec_val,
                    "lam_L_rec": lam_rec_val, "L_total": float(loss),
                    "grad_norm": gnorm, "lr": LEARNING_RATE})
                if step % 100 == 0:
                    logger.info("[%s] step %d/%d L_total=%.4f",
                                __abbr__, step, TOTAL_STEPS, float(loss))
            if step != (ep + 1) * STEPS_PER_EPOCH:
                status, error_code = "interrupted", "STEP_COUNT_MISMATCH"
                _fail("STEP_COUNT_MISMATCH",
                      f"epoch {ep} ended at step {step}; expected "
                      f"{(ep + 1) * STEPS_PER_EPOCH}")
            vals = _validate(model, val_loader, p4, lam)
            ck = out / checkpoint_name(arm, seed, attempt,
                                     ep + 1)
            torch.save(model.state_dict(), ck)
            ck_sha = _file_sha256(ck)
            checkpoints.append({"epoch": ep + 1, "file": ck.name,
                                "file_sha256": ck_sha, **vals})
            val_history.append({"epoch": ep + 1, **vals})
            logger.info("[%s] epoch %d: %s", __abbr__, ep + 1, vals)
    except V03TError as exc:
        if status == "completed":
            status = ("numerical_failure"
                      if exc.code == "NUMERICAL_FAILURE"
                      else "interrupted")
            error_code = exc.code
        raise
    except v03l.V03LError as exc:
        # Objective-level failure (V03L): classify via the locked
        # taxonomy -- non-finite family -> numerical_failure; anything
        # else -> failed_validity. Both TERMINAL, never infrastructure.
        status = classify_objective_error(exc.code)
        error_code = exc.code
        logger.error("[%s] objective failure %s at step %d: %s",
                     __abbr__, exc.code, step, exc)
        raise V03TError(exc.code, f"objective failure: {exc}") from exc
    except V03PError as exc:
        # Gradient-norm helper failure (V03P): no gradient coverage
        # after backward is structural, not numerical instability ->
        # failed_validity, TERMINAL, never infrastructure.
        status = classify_gradient_error(exc.code)
        error_code = exc.code
        logger.error("[%s] gradient-norm failure %s at step %d: %s",
                     __abbr__, exc.code, step, exc)
        raise V03TError(exc.code, f"gradient-norm failure: {exc}") \
            from exc
    except Exception as exc:
        status, error_code = "interrupted", type(exc).__name__
        logger.exception("[%s] infrastructure interruption at step %d",
                         __abbr__, step)
        raise V03TError("INFRASTRUCTURE_INTERRUPTION",
                        f"{type(exc).__name__}: {exc}") from exc
    finally:
        record = {
            "schema": RECORD_SCHEMA,
            "script": f"{__abbr__} v{__version__}",
            "spec": {"article": 163, "status": "FROZEN"},
            "plan": {"article": 164, "status": "FROZEN",
                     "amendments": ["A1", "A2"]},
            "arm": arm, "lambda": lam, "seed": seed, "attempt": attempt,
            "status": status, "error_code": error_code,
            "steps_completed": step,
            "config": {"optimizer": "Adam", "lr": LEARNING_RATE,
                       "batch": BATCH_SIZE, "epochs": N_EPOCHS,
                       "steps": TOTAL_STEPS, "precision": "float32",
                       "device": "cpu", "schedule": None,
                       "gradient_clipping": None,
                       "init_seed": seed},
            "probe_record": {"path": str(probe_record_path),
                             "sha256": _file_sha256(
                                 Path(probe_record_path))},
            "history": history,
            "validation_history": val_history,
            "checkpoints": checkpoints,
            "wall_clock_s": time.time() - t0,
            "environment": {"python": platform.python_version(),
                            "torch": torch.__version__,
                            "platform": platform.platform()},
        }
        rpath = out / run_record_name(arm, seed, attempt)
        rsha = _write_with_sidecar(rpath, canonical_json(record))
        append_entry(ledger_path, {
            "arm": arm, "lambda": lam, "seed": seed, "attempt": attempt,
            "status": status, "error_code": error_code,
            "steps_completed": step,
            "wall_clock_s": record["wall_clock_s"],
            "record_file": rpath.name, "record_sha256": rsha,
            "checkpoints": [c["file_sha256"] for c in checkpoints]})
        logger.info("[%s] ledger entry appended: arm %d seed %d attempt "
                    "%d status %s", __abbr__, arm, seed, attempt, status)
    if step != TOTAL_STEPS:
        _fail("STEP_COUNT_MISMATCH",
              f"run completed {step} steps; the locked budget is "
              f"{TOTAL_STEPS}")
    print(f"PASS {__abbr__} v{__version__} arm {arm} seed {seed} "
          f"attempt {attempt}: {TOTAL_STEPS} steps, "
          f"{len(checkpoints)} checkpoints")
    return 0


# ---------------------------------------------------------------------------
# Selftest: fixture f06 (rerun-ledger bookkeeping), run-contract validator
# gates, and the f10 static audit over the V03 modules. No model, no data.
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="v03t_selftest_"))
    led_path = tmp / "v03_run_ledger.json"

    # f06a: attempt counting and policy gates.
    led = load_ledger(led_path)
    if next_attempt(led, 1, 101) != 1:
        _fail("RERUN_POLICY_VIOLATION", "fresh ledger attempt != 1")
    append_entry(led_path, {"arm": 1, "lambda": 2.0, "seed": 101,
                            "attempt": 1, "status": "interrupted",
                            "error_code": "INFRASTRUCTURE_INTERRUPTION",
                            "steps_completed": 40, "wall_clock_s": 1.0,
                            "record_file": "r.json", "record_sha256": "x",
                            "checkpoints": []})
    if next_attempt(load_ledger(led_path), 1, 101) != 2:
        _fail("RERUN_POLICY_VIOLATION", "after 1 interruption attempt != 2")
    append_entry(led_path, {"arm": 1, "lambda": 2.0, "seed": 101,
                            "attempt": 2, "status": "interrupted",
                            "error_code": "INFRASTRUCTURE_INTERRUPTION",
                            "steps_completed": 5, "wall_clock_s": 1.0,
                            "record_file": "r.json", "record_sha256": "x",
                            "checkpoints": []})
    append_entry(led_path, {"arm": 1, "lambda": 2.0, "seed": 101,
                            "attempt": 3, "status": "interrupted",
                            "error_code": "INFRASTRUCTURE_INTERRUPTION",
                            "steps_completed": 9, "wall_clock_s": 1.0,
                            "record_file": "r.json", "record_sha256": "x",
                            "checkpoints": []})
    for name, status in (("max attempts", "interrupted"),
                         ("numerical failure terminal", "numerical_failure"),
                         ("completed terminal", "completed")):
        lp = tmp / f"ledger_{status}.json"
        append_entry(lp, {"arm": 2, "lambda": 2.0, "seed": 202,
                          "attempt": 1, "status": status,
                          "error_code": None, "steps_completed": 1,
                          "wall_clock_s": 1.0, "record_file": "r.json",
                          "record_sha256": "x", "checkpoints": []})
        if status == "interrupted":
            for a in (2, 3):
                append_entry(lp, {"arm": 2, "lambda": 2.0, "seed": 202,
                                  "attempt": a, "status": "interrupted",
                                  "error_code": None, "steps_completed": 1,
                                  "wall_clock_s": 1.0,
                                  "record_file": "r.json",
                                  "record_sha256": "x", "checkpoints": []})
        try:
            next_attempt(load_ledger(lp), 2, 202)
        except V03TError as exc:
            if exc.code != "RERUN_POLICY_VIOLATION":
                _fail("PROBE_SELFTEST_WRONG_CODE",
                      f"{name}: raised {exc.code}")
            logger.info("[%s] f06 branch PASS: %s", __abbr__, name)
        else:
            _fail("RERUN_POLICY_VIOLATION",
                  f"{name}: policy gate is dead")
    # f06b: tampered ledger sidecar chain.
    bad = tmp / "ledger_tampered.json"
    append_entry(bad, {"arm": 1, "lambda": 2.0, "seed": 101, "attempt": 1,
                       "status": "interrupted", "error_code": None,
                       "steps_completed": 1, "wall_clock_s": 1.0,
                       "record_file": "r.json", "record_sha256": "x",
                       "checkpoints": []})
    led_obj = json.loads(bad.read_text())
    led_obj["entries"][0]["status"] = "completed"      # silent mutation
    bad.write_bytes(canonical_json(led_obj))
    try:
        load_ledger(bad)
    except V03TError as exc:
        if exc.code != "LEDGER_TAMPERED":
            _fail("PROBE_SELFTEST_WRONG_CODE",
                  f"tampered ledger raised {exc.code}")
        logger.info("[%s] f06 branch PASS: tamper detection", __abbr__)
    else:
        _fail("LEDGER_TAMPERED", "tampered ledger accepted")

    # f06c: attempt-scoped evidence naming -- a rerun can never overwrite
    # a prior attempt's record or checkpoints.
    if "attempt1" not in run_record_name(1, 101, 1) \
            or run_record_name(1, 101, 1) == run_record_name(1, 101, 2) \
            or checkpoint_name(1, 101, 1, 3) == checkpoint_name(1, 101, 2, 3):
        _fail("RERUN_POLICY_VIOLATION",
              "evidence filenames collide across attempts")
    logger.info("[%s] f06c PASS: attempt-scoped filenames", __abbr__)

    # f06d: locked seed-major global order gate.
    fresh = {"schema": LEDGER_SCHEMA, "entries": []}
    if expected_next_run(fresh) != (0, 101):
        _fail("ORDER_VIOLATION", "fresh ledger next run != (arm 0, 101)")
    enforce_order(fresh, 0, 101)                    # allowed, no exception
    for a, s in ((1, 101), (3, 303), (0, 202)):
        caught = None
        try:
            enforce_order(fresh, a, s)
        except V03TError as exc:
            caught = exc
        if caught is None or caught.code != "ORDER_VIOLATION":
            _fail("ORDER_VIOLATION",
                  f"out-of-order run arm {a} seed {s} not refused")
    lp_ord = tmp / "ledger_order.json"
    append_entry(lp_ord, {"arm": 0, "lambda": 0.0, "seed": 101,
                          "attempt": 1, "status": "completed",
                          "error_code": None, "steps_completed": TOTAL_STEPS,
                          "wall_clock_s": 1.0, "record_file": "r.json",
                          "record_sha256": "x", "checkpoints": []})
    if expected_next_run(load_ledger(lp_ord)) != (1, 101):
        _fail("ORDER_VIOLATION", "completed pair did not advance to (1,101)")
    append_entry(lp_ord, {"arm": 1, "lambda": 2.0, "seed": 101,
                          "attempt": 1, "status": "interrupted",
                          "error_code": "INFRASTRUCTURE_INTERRUPTION",
                          "steps_completed": 10, "wall_clock_s": 1.0,
                          "record_file": "r.json", "record_sha256": "x",
                          "checkpoints": []})
    led_ord = load_ledger(lp_ord)
    if expected_next_run(led_ord) != (1, 101):
        _fail("ORDER_VIOLATION",
              "interrupted pair must block advance and demand its retry")
    enforce_order(led_ord, 1, 101)                  # its retry: allowed
    caught = None
    try:
        enforce_order(led_ord, 2, 101)
    except V03TError as exc:
        caught = exc
    if caught is None or caught.code != "ORDER_VIOLATION":
        _fail("ORDER_VIOLATION",
              "advance past an interrupted pair not refused")
    append_entry(lp_ord, {"arm": 1, "lambda": 2.0, "seed": 101,
                          "attempt": 2, "status": "numerical_failure",
                          "error_code": "NUMERICAL_FAILURE",
                          "steps_completed": 10, "wall_clock_s": 1.0,
                          "record_file": "r.json", "record_sha256": "x",
                          "checkpoints": []})
    if expected_next_run(load_ledger(lp_ord)) != (2, 101):
        _fail("ORDER_VIOLATION",
              "terminal numerical failure must advance the chain")
    logger.info("[%s] f06d PASS: seed-major order gate (5 branches)",
                __abbr__)

    # f06e: objective-error classification (terminal, no rerun).
    if classify_objective_error("U_NON_FINITE") != "numerical_failure" \
            or classify_objective_error("LOSS_CONTRACT_VIOLATION") \
            != "failed_validity" \
            or classify_objective_error("MAP_BATCH_MISMATCH") \
            != "failed_validity":
        _fail("PROBE_SELFTEST_WRONG_VALUE",
              "objective-error classification mapping wrong")
    if classify_gradient_error("PROBE_VALIDITY_FAILURE") \
            != "failed_validity":
        _fail("PROBE_SELFTEST_WRONG_VALUE",
              "gradient-error classification mapping wrong")
    logger.info("[%s] f06e PASS: objective + gradient error "
                "classification", __abbr__)

    # Contract validator gates.
    good = {"schema": PROBE_SCHEMA, "n_batches": 100,
            "batch_size": BATCH_SIZE,
            "split": "train", "mode": "eval",
            "validation_source_amendment": "A2",
            "lambdas": {"0.1": 0.2, "0.5": 2.0, "0.9": 18.0}}
    gp = tmp / "probe_good.json"
    _write_with_sidecar(gp, canonical_json(good))
    rec = load_probe_record(str(gp))
    if lambda_for_arm(rec, 0) != 0.0 or lambda_for_arm(rec, 2) != 2.0:
        _fail("PROBE_RECORD_LAMBDAS_INVALID", "lambda_for_arm mapping")
    for name, mutate in (
            ("schema", lambda r: r.update(schema="wrong/9")),
            ("validation source", lambda r: r.update(split="val")),
            ("lambda band", lambda r: r["lambdas"].update({"0.5": 1e9})),
            ("n_batches", lambda r: r.update(n_batches=99))):
        bad_rec = json.loads(canonical_json(good))
        mutate(bad_rec)
        bp = tmp / f"probe_bad_{name.replace(' ', '_')}.json"
        _write_with_sidecar(bp, canonical_json(bad_rec))
        try:
            load_probe_record(str(bp))
        except V03TError:
            logger.info("[%s] contract branch PASS: bad %s", __abbr__,
                        name)
        else:
            _fail("PROBE_RECORD_SCHEMA_MISMATCH",
                  f"bad {name} probe record accepted")
    tampered = tmp / "probe_tampered.json"
    _write_with_sidecar(tampered, canonical_json(good))
    tampered.write_bytes(canonical_json(dict(good, n_batches=100,
                                             extra="x")))
    try:
        load_probe_record(str(tampered))
    except V03TError as exc:
        if exc.code != "PROBE_RECORD_HASH_MISMATCH":
            _fail("PROBE_SELFTEST_WRONG_CODE",
                  f"tampered probe record raised {exc.code}")
        logger.info("[%s] contract branch PASS: hash mismatch", __abbr__)
    else:
        _fail("PROBE_RECORD_HASH_MISMATCH", "tampered probe accepted")
    for arm, seed in ((4, 101), (0, 999), (-1, 101)):
        caught = None
        try:
            validate_arm_seed(arm, seed)
        except V03TError as exc:
            caught = exc
        if caught is None:
            _fail("ARM_INVALID", f"({arm}, {seed}) accepted")
    logger.info("[%s] contract branches PASS: arm/seed gates", __abbr__)

    # f10: static audit over the V03 modules + a synthetic offender.
    root = Path(__file__).resolve().parents[2]
    static_audit([root / "seqref_mri/src/flows/v03_objective.py",
                  root / "seqref_mri/scripts/v03_probe.py",
                  root / "seqref_mri/scripts/v03_train.py"])
    offender = tmp / "offender.py"
    offender.write_text("try:\n    x = 1\nexcept Exception:\n    pass\n")
    try:
        static_audit([offender])
    except V03TError as exc:
        if exc.code != "AUDIT_PATTERN_FOUND":
            _fail("PROBE_SELFTEST_WRONG_CODE",
                  f"offender raised {exc.code}")
        logger.info("[%s] f10 PASS: audit clean + offender refused",
                    __abbr__)
    else:
        _fail("AUDIT_PATTERN_FOUND", "offender file accepted")
    print(f"PASS {__abbr__} v{__version__} selftest (f06, contract, f10)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="v03_train")
    ap.add_argument("--mode", choices=("selftest", "run"), required=True)
    ap.add_argument("--data-root")
    ap.add_argument("--p4-stats2")
    ap.add_argument("--probe-record")
    ap.add_argument("--out")
    ap.add_argument("--arm", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        if args.mode == "selftest":
            return _selftest()
        if args.mode == "run":
            missing = [k for k, v in (
                ("data-root", args.data_root),
                ("p4-stats2", args.p4_stats2),
                ("probe-record", args.probe_record),
                ("out", args.out),
                ("arm", args.arm), ("seed", args.seed)) if v is None]
            if missing:
                _fail("RUN_ARGS_MISSING", f"--mode run requires {missing}")
            return _run(args.data_root, args.p4_stats2, args.probe_record,
                        args.out, args.arm, args.seed)
    except V03TError as exc:
        logger.error("[%s] ERROR: %s", __abbr__, exc)
        return 2
    except v03l.V03LError as exc:
        logger.error("[%s] objective ERROR: %s", __abbr__, exc)
        return 2
    except Exception:
        logger.exception("[%s] unexpected exception", __abbr__)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# Changelog
# v0.1 (2026-10-06): initial implementation under V03SPEC v0.3 (163,
#   FROZEN) SS7/SS18 / V03PLAN v0.3 (164, FROZEN, Amendment A1) SS8.
#   Five responsibilities per reviewer structure: run-contract loader
#   (hash-verified probe record, arm/seed gates, lambda mapping), single-
#   run trainer (fresh init per arm-seed, inherited V02 config, hybrid
#   objective), machine-readable history recorder, epoch checkpoint
#   manager (SHA-256, no winner selection), append-only sidecar-chained
#   run ledger enforcing max 3 attempts / same-seed restart / numerical-
#   failure terminal. Per-step telemetry incl. combined gradient norm;
#   per-epoch 100-batch forward-only validation with L_val. Fixtures:
#   f06 rerun policy (7 branches incl. tamper detection), run-contract
#   gates (6 branches), f10 static audit over V03 modules + offender
#   refusal. No per-term double backward (V03P owns per-term telemetry).
# v0.2 (2026-10-06): reviewer round -- 4 fixes + 1 strengthening.
#   (1) BLOCKER: attempt-scoped evidence filenames (run_record_name /
#   checkpoint_name) -- a rerun can no longer overwrite a prior attempt's
#   record or checkpoints. (2) BLOCKER: locked seed-major global order
#   enforced (expected_next_run / enforce_order, ORDER_VIOLATION;
#   interrupted pair blocks advance until retried, terminal pair
#   advances). (3) V03LError mapped onto the ledger taxonomy
#   (classify_objective_error: non-finite family -> numerical_failure,
#   else failed_validity; both terminal, no rerun) instead of falling
#   into the generic infrastructure catch. (4) Amendment A2: validation
#   source locked to TRAIN split in eval mode -- the official val split
#   is the 199-volume holdout; val-split validation would have been
#   pre-evaluation holdout contact. (5) _validate uses model.eval() and
#   restores training mode. f06 expanded: attempt-scoped naming, order
#   gate (5 branches), error classification.
# ---------------------------------------------------------------------------
