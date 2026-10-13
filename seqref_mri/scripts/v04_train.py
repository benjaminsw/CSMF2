# =============================================================================
# SEQREF-MRI-V04T v0.1 -- scripts.v04_train
# LIFETIME: KEEP
# Purpose: V04 training driver + append-only run ledger (V04SPEC v1.0
#          article 176 FROZEN SS7/SS18; V04PLAN v1.0 article 178 FROZEN
#          SS8/SS9). Resume/restart policy inherited VERBATIM from
#          V03T v0.5 (178 SS8) -- no reinterpretation. ONE (arm, seed,
#          attempt) per invocation; the locked seed-major order
#          (per seed: arm 0 -> arm 1) is driven externally and enforced
#          by the ledger, never by memory.
# Locked quantities (quoted, never CLI-tunable):
#   * arms {0, 1}: arm 0 -> lambda 0.0 (density-only anchor); arm 1 ->
#     lambda_2' from the frozen V04 probe record; seeds {101, 202, 303};
#     fresh V04M init per run via build_v04_model(init_seed=seed)
#   * inherited V02/V03 config: Adam lr=1e-4 PyTorch defaults, batch 32
#     (final partial batch kept), 3 epochs x 1086 = 3258 steps, float32
#     CPU, no scheduler, no clipping; non-finite loss/gradients =
#     NUMERICAL failure, never a silent retry
#   * checkpoints at every epoch end (3 per run) + SHA-256 each; every
#     checkpoint records arm/seed/attempt/epoch/lambda/init_seed identity
#     (safeguard A); selection rule = argmin L_val, earliest-epoch
#     tie-break (178 SS9, identical to V03; fixture f05)
#   * validation: first 100 batches (A2 convention: split="train",
#     mode="eval"), forward-only, per epoch; L_val = val_NLL +
#     lam * val_L_rec (V04L terms consumed, never reimplemented)
#   * rerun policy (V03T v0.5 verbatim): infrastructure interruption ->
#     max 3 attempts total, full restart same seed; numerical failure and
#     failed_validity TERMINAL, no restart; completed -> no restart.
#     Ledger append-only, sidecar-chained; attempt-scoped evidence files;
#     seed-major global order enforced via expected_next_run.
# Telemetry (mirrors V03T): per step -- step, epoch, L_density, L_rec,
#   lam*L_rec, L_total, combined gradient norm, lr. Per epoch -- val_NLL,
#   val_L_rec, L_val, checkpoint hash, wall clock. ASYMMETRY (locked):
#   per-step L_rec recorded only for lam > 0; lam = 0 does not execute
#   the R0 decode per training step.
# Post-training conditioning diagnostic (178 SS8, NON-GATING): same-z /
#   different-y output responsiveness on the final state; recorded in the
#   train record; never an acceptance threshold.
# GATE (fail-closed, safeguard D): --mode run refuses to start unless ALL
#   prerequisites verify -- (1) --commit-pin equals the repository HEAD;
#   (2) hash-verified V04 probe record (valid lambda_2'); (3) hash-
#   verified V04 gate record with all six gates PASS. Missing/invalid
#   prerequisite = typed abort; no fallback to an ordinary run.
# Emits per run: v04_arm{A}_seed{S}_attempt{N}_train_record.json (schema
#   seqref-v04-train-record/1) + SHA-256 sidecar; ledger
#   v04_run_ledger.json (schema seqref-v04-run-ledger/1) + sidecar.
# CONVENTION: logger.error + typed raise (V04TError). PASS(0)/ERROR(2).
#   No fallback, no mock, no placeholder, no silent pass.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from seqref_mri.src import free_flow_runtime as ffr
from seqref_mri.src.fastmri_data import FastMRISliceDataset
from seqref_mri.src.flows import v03_objective as v03l
from seqref_mri.scripts import v04_objective as v04o
from seqref_mri.scripts import v04_probe as v04p
from seqref_mri.scripts.train_base import _collate, _prepare
from seqref_mri.scripts.train_free_flow import targets_from_prepared
from seqref_mri.scripts.v02_manifests import (BATCH_SIZE, N_EPOCHS,
                                              canonical_json)
from seqref_mri.scripts.v02_train import derive_cmap_from_mask
from seqref_mri.scripts.v04_model import build_v04_model

logger = logging.getLogger("seqref_mri.v04_train")

__version__ = "0.1"
__abbr__ = "SEQREF-MRI-V04T"

LEARNING_RATE = 1e-4                      # locked (178 SS8)
TOTAL_STEPS = 3258                        # 3 x 1086, locked (178 SS8)
STEPS_PER_EPOCH = 1086                    # locked
N_VAL_BATCHES = 100                       # locked (178 SS8 cadence)
ARMS = (0, 1)                             # locked (176 SS7): 0, lambda_2'
SEEDS = (101, 202, 303)                   # locked (176 SS7)
MAX_ATTEMPTS = 3                          # locked (176 SS18)
LEDGER_SCHEMA = "seqref-v04-run-ledger/1"
RECORD_SCHEMA = "seqref-v04-train-record/1"
PROBE_SCHEMA = "seqref-v04-probe-record/1"
GATE_SCHEMA = "seqref-v04-gate-record/1"
GATE_NAMES = ("a", "b", "c", "d", "e", "f")   # 178 SS7 gates
DIAG_SEED_OFFSET = 7                      # diagnostic z generator offset


class V04TError(RuntimeError):
    """Single typed identity for the V04 training driver."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V04TError(code, message)


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
# Fail-closed prerequisites (safeguard D): commit pin + probe record +
# gate record. Missing/invalid prerequisite = typed abort, no fallback.
# ---------------------------------------------------------------------------

def verify_commit_pin(expected: str, repo_root: Path) -> str:
    """The pinned implementation commit must equal the repository HEAD.
    No git repository / no pin / mismatch -> COMMIT_PIN_MISMATCH."""
    if not expected:
        _fail("COMMIT_PIN_MISMATCH",
              "no commit pin supplied; authoritative training requires the "
              "pinned implementation commit (178 SS15)")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        _fail("COMMIT_PIN_MISMATCH",
              f"cannot resolve repository HEAD at {repo_root} ({exc}); the "
              "implementation commit is not pinned")
    if head != expected:
        _fail("COMMIT_PIN_MISMATCH",
              f"supplied pin {expected[:12]}... != repository HEAD "
              f"{head[:12]}...")
    return head


def load_probe_record(path: str) -> dict:
    """Hash-verified frozen V04 probe record; schema + locked-field gates."""
    p = Path(path)
    side = p.with_suffix(p.suffix + ".sha256")
    if not p.is_file() or not side.is_file():
        _fail("TRAIN_PREREQ_MISSING",
              f"probe record or sidecar missing at {p}")
    recorded = side.read_text().split()[0]
    digest = _file_sha256(p)
    if digest != recorded:
        _fail("PROBE_RECORD_HASH_MISMATCH",
              f"probe record sha256 {digest[:12]}... != sidecar "
              f"{recorded[:12]}...; the frozen probe evidence drifted")
    rec = json.loads(p.read_text())
    if rec.get("schema") != PROBE_SCHEMA:
        _fail("PROBE_RECORD_SCHEMA_MISMATCH",
              f"schema {rec.get('schema')!r} != {PROBE_SCHEMA!r}")
    if int(rec.get("n_batches", -1)) != 100 \
            or int(rec.get("batch_size", -1)) != BATCH_SIZE:
        _fail("PROBE_RECORD_CONFIG_MISMATCH",
              "probe record n_batches/batch_size diverge from the locked "
              "100/32 contract")
    if rec.get("l_rec_scale") != v04o.L_REC_SCALE:
        _fail("PROBE_RECORD_CONFIG_MISMATCH",
              f"probe record l_rec_scale {rec.get('l_rec_scale')!r} != "
              f"the pinned scale {v04o.L_REC_SCALE!r}")
    if rec.get("split") != "train" or rec.get("mode") != "eval" \
            or rec.get("validation_source_amendment") != "A2":
        _fail("PROBE_RECORD_CONFIG_MISMATCH",
              "probe record validation source diverges from the A2 "
              'convention (split="train", mode="eval")')
    ws = rec.get("weight_state") or {}
    if ws.get("kind") != "init" \
            or int(ws.get("model_init_seed", -1)) != v04p.PROBE_INIT_SEED:
        _fail("PROBE_RECORD_CONFIG_MISMATCH",
              f"probe weight state {ws!r} != the locked registered fresh "
              f"init (seed {v04p.PROBE_INIT_SEED})")
    if rec.get("lambda_prime_factor") != v04p.LAMBDA_PRIME_FACTOR:
        _fail("PROBE_RECORD_CONFIG_MISMATCH",
              "probe lambda factor diverges from the locked 0.5")
    blh = rec.get("batch_list_sha256")
    if not isinstance(blh, str) or len(blh) != 64:
        _fail("PROBE_RECORD_CONFIG_MISMATCH",
              "probe record lacks a valid batch_list_sha256")
    lam = rec.get("lambda_prime")
    if not isinstance(lam, (int, float)) or not np.isfinite(lam) \
            or lam <= 0 or not (v04p.LAMBDA_BAND[0] <= lam
                                <= v04p.LAMBDA_BAND[1]):
        _fail("PROBE_RECORD_LAMBDAS_INVALID",
              f"lambda_prime={lam!r} violates the locked validity band; "
              "the probe record is not a valid run contract")
    return rec


def load_gate_record(path: str) -> dict:
    """Hash-verified V04 reliability-gate record; all six gates must PASS."""
    p = Path(path)
    side = p.with_suffix(p.suffix + ".sha256")
    if not p.is_file() or not side.is_file():
        _fail("TRAIN_PREREQ_MISSING",
              f"gate record or sidecar missing at {p}")
    recorded = side.read_text().split()[0]
    digest = _file_sha256(p)
    if digest != recorded:
        _fail("GATE_RECORD_HASH_MISMATCH",
              f"gate record sha256 {digest[:12]}... != sidecar "
              f"{recorded[:12]}...")
    rec = json.loads(p.read_text())
    if rec.get("schema") != GATE_SCHEMA:
        _fail("GATE_RECORD_SCHEMA_MISMATCH",
              f"schema {rec.get('schema')!r} != {GATE_SCHEMA!r}")
    gates = rec.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(GATE_NAMES):
        _fail("GATE_RECORD_SCHEMA_MISMATCH",
              f"gate record must cover exactly gates {GATE_NAMES}, got "
              f"{None if gates is None else sorted(gates)}")
    for name in GATE_NAMES:
        if gates[name].get("status") != "PASS":
            _fail("GATE_NOT_PASSED",
                  f"reliability gate ({name}) status "
                  f"{gates[name].get('status')!r} != PASS; authoritative "
                  "training is not authorised (178 SS15 order)")
    return rec


# ---------------------------------------------------------------------------
# Run contract: arm/seed validation + lambda mapping
# ---------------------------------------------------------------------------

def validate_arm_seed(arm: int, seed: int) -> None:
    if arm not in ARMS:
        _fail("ARM_INVALID", f"arm {arm!r} not in locked set {ARMS}")
    if seed not in SEEDS:
        _fail("SEED_INVALID", f"seed {seed!r} not in locked set {SEEDS}")


def lambda_for_arm(rec: dict, arm: int) -> float:
    """Arm 0 -> 0.0 (density-only anchor); arm 1 -> probe lambda_2'."""
    if arm == 0:
        return 0.0
    return float(rec["lambda_prime"])


# ---------------------------------------------------------------------------
# Locked seed-major global order + attempt-scoped evidence naming
# ---------------------------------------------------------------------------

GLOBAL_ORDER = [(s, a) for s in SEEDS for a in ARMS]   # seed-major, locked
TERMINAL_STATUSES = ("completed", "numerical_failure", "failed_validity")


def run_record_name(arm: int, seed: int, attempt: int) -> str:
    return f"v04_arm{arm}_seed{seed}_attempt{attempt}_train_record.json"


def checkpoint_name(arm: int, seed: int, attempt: int, epoch: int) -> str:
    return f"v04_arm{arm}_seed{seed}_attempt{attempt}_epoch{epoch}.pt"


def parse_checkpoint_name(name: str):
    """Round-trip parse of the attempt-scoped checkpoint filename."""
    stem = name[:-3] if name.endswith(".pt") else name
    parts = stem.split("_")
    if len(parts) != 5 or parts[0] != "v04":
        return None
    try:
        return {"arm": int(parts[1][3:]), "seed": int(parts[2][4:]),
                "attempt": int(parts[3][7:]), "epoch": int(parts[4][5:])}
    except ValueError:
        return None


def assert_checkpoint_identity(entry: dict, arm: int, seed: int,
                               attempt: int, lam: float) -> None:
    """Safeguard A: every checkpoint records the correct seed, arm, epoch
    and configuration identity; the filename must round-trip to the same
    identity. Prevents accidental checkpoint reuse across arms/seeds."""
    parsed = parse_checkpoint_name(entry.get("file", ""))
    expect = {"arm": arm, "seed": seed, "attempt": attempt,
              "epoch": entry.get("epoch")}
    if parsed != expect:
        _fail("CHECKPOINT_IDENTITY_MISMATCH",
              f"checkpoint filename {entry.get('file')!r} parses to "
              f"{parsed}, expected {expect}")
    for key, want in (("arm", arm), ("seed", seed), ("attempt", attempt),
                      ("init_seed", seed)):
        if entry.get(key) != want:
            _fail("CHECKPOINT_IDENTITY_MISMATCH",
                  f"checkpoint entry {key}={entry.get(key)!r} != {want!r}")
    if float(entry.get("lambda", float("nan"))) != float(lam):
        _fail("CHECKPOINT_IDENTITY_MISMATCH",
              f"checkpoint entry lambda={entry.get('lambda')!r} != {lam!r}")


def expected_next_run(ledger: dict):
    """The single runnable (arm, seed) under the locked seed-major order.
    A pair is settled once it holds a TERMINAL entry; an interrupted pair
    blocks advancement until retried to a terminal state. Returns
    (arm, seed) or None if all 6 pairs are settled."""
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
              "all 6 arm-seed pairs are settled; no further runs are "
              "permitted")
    if nxt != (arm, seed):
        _fail("ORDER_VIOLATION",
              f"locked seed-major order: the next runnable pair is arm "
              f"{nxt[0]} seed {nxt[1]}, not arm {arm} seed {seed}")


def classify_objective_error(code: str) -> str:
    """V03T v0.5 taxonomy verbatim: U_NON_FINITE -> numerical_failure
    (instability evidence); anything else -> failed_validity (structural
    contract breach). Both TERMINAL -- neither permits a rerun."""
    if code == "U_NON_FINITE":
        return "numerical_failure"
    return "failed_validity"


def classify_gradient_error(code: str) -> str:
    """V04P gradient-norm failures are structural coverage failures:
    always failed_validity -- never numerical instability, never
    infrastructure. TERMINAL."""
    return "failed_validity"


# ---------------------------------------------------------------------------
# Append-only run ledger (sidecar-chained) -- V03T v0.5 policy verbatim
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
    """Rerun policy (V03T v0.5 verbatim): completed/numerical_failure ->
    no restart; interruptions -> up to MAX_ATTEMPTS total, same seed."""
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
# Checkpoint selection (178 SS9, identical to V03; fixture f05 domain)
# ---------------------------------------------------------------------------

def select_checkpoint(validation_history: list, checkpoints: list) -> dict:
    """Per arm-seed: the checkpoint with the lowest validation L_val;
    ties break to the earliest epoch (strict < keeps the first
    occurrence). The selected epoch must carry a checkpoint whose
    recorded L_val matches."""
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
                  "argmin rule is undefined")
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


# ---------------------------------------------------------------------------
# Post-training conditioning diagnostic (178 SS8, NON-GATING)
# ---------------------------------------------------------------------------

def post_training_conditioning_diagnostic(model, cond_a, mask_a, cond_b,
                                          mask_b, seed: int) -> dict:
    """Same-z / different-y output responsiveness on the final state.
    Recorded evidence only -- never an acceptance threshold (178 SS8)."""
    gen = torch.Generator().manual_seed(seed + DIAG_SEED_OFFSET)
    z = torch.randn(cond_a.shape[0], ffr.FLOW_DIM_REAL, generator=gen)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out_a = model.decode_scalars(z, cond_a, mask_a)
            out_b = model.decode_scalars(z, cond_b, mask_b)
    finally:
        if was_training:
            model.train()
    d = (out_a - out_b).abs()
    return {"rel_change": float(
                (d / out_a.abs().clamp_min(1e-12)).mean()),
            "abs_change": float(d.mean()),
            "gating": False}


# ---------------------------------------------------------------------------
# Validation (forward-only, no updates)
# ---------------------------------------------------------------------------

def _validate(model, val_loader, p4, lam: float) -> dict:
    """Forward-only validation (A2 source: split="train", mode="eval" --
    NEVER the 199-volume holdout). Explicit eval mode; training mode
    restored on exit."""
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
                nlls[bi] = float(v04o.density_only(model, tgt, cond, msk))
                recs[bi] = float(v04o.rec_term(model, cond, msk, prep["y"],
                                               prep["amax"],
                                               prep["tgt_norm"], maps))
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
# Single-run trainer: ONE (arm, seed, attempt). GATED -- prerequisites
# verify before any training state exists; no fallback to an ordinary run.
# ---------------------------------------------------------------------------

def _run(data_root: str, p4_stats2: str, probe_record_path: str,
         gate_record_path: str, commit_pin: str, out_dir: str,
         arm: int, seed: int) -> int:
    validate_arm_seed(arm, seed)
    repo_root = Path(__file__).resolve().parents[2]
    head = verify_commit_pin(commit_pin, repo_root)          # D(1)
    rec_probe = load_probe_record(probe_record_path)         # D(2)
    rec_gate = load_gate_record(gate_record_path)            # D(3)
    lam = lambda_for_arm(rec_probe, arm)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / "v04_run_ledger.json"
    enforce_order(load_ledger(ledger_path), arm, seed)
    attempt = next_attempt(load_ledger(ledger_path), arm, seed)
    logger.info("[%s] arm %d (lambda=%.6g) seed %d attempt %d/%d",
                __abbr__, arm, lam, seed, attempt, MAX_ATTEMPTS)

    p4 = ffr.load_p4s2_parent(p4_stats2)
    model = build_v04_model(init_seed=seed)     # fresh init, arm-seed (A)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ds_train = FastMRISliceDataset(data_root, split="train", mode="train")
    # A2 convention: validation = train split in eval mode (deterministic
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
                nll = v04o.density_only(model, tgt, cond, msk)
                if lam == 0.0:
                    loss = nll                    # untouched bit-exact path
                    l_rec_val, lam_rec_val = None, None
                else:
                    rec = v04o.rec_term(model, cond, msk, prep["y"],
                                        prep["amax"], prep["tgt_norm"],
                                        maps)
                    loss = nll + lam * rec
                    l_rec_val, lam_rec_val = float(rec), float(lam * rec)
                if not torch.isfinite(loss):
                    status, error_code = "numerical_failure", \
                        "NUMERICAL_FAILURE"
                    _fail("NUMERICAL_FAILURE",
                          f"step {step}: combined objective non-finite; "
                          f"instability evidence, no restart")
                loss.backward()
                gnorm = v04p.global_grad_norm(model)
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
            ck = out / checkpoint_name(arm, seed, attempt, ep + 1)
            torch.save(model.state_dict(), ck)
            ck_sha = _file_sha256(ck)
            entry = {"epoch": ep + 1, "file": ck.name,
                     "file_sha256": ck_sha, "arm": arm, "seed": seed,
                     "attempt": attempt, "lambda": lam, "init_seed": seed,
                     **vals}
            assert_checkpoint_identity(entry, arm, seed, attempt, lam)
            checkpoints.append(entry)
            val_history.append({"epoch": ep + 1, **vals})
            logger.info("[%s] epoch %d: %s", __abbr__, ep + 1, vals)
    except V04TError as exc:
        if status == "completed":
            status = ("numerical_failure"
                      if exc.code == "NUMERICAL_FAILURE"
                      else "interrupted")
            error_code = exc.code
        raise
    except (v03l.V03LError, v04o.V04LError) as exc:
        # Objective-level failure: classify via the locked taxonomy --
        # non-finite family -> numerical_failure; anything else ->
        # failed_validity. Both TERMINAL, never infrastructure.
        status = classify_objective_error(exc.code)
        error_code = exc.code
        logger.error("[%s] objective failure %s at step %d: %s",
                     __abbr__, exc.code, step, exc)
        raise V04TError(exc.code, f"objective failure: {exc}") from exc
    except v04p.V04PError as exc:
        # Gradient-norm helper failure: no gradient coverage after
        # backward is structural, not numerical instability ->
        # failed_validity, TERMINAL, never infrastructure.
        status = classify_gradient_error(exc.code)
        error_code = exc.code
        logger.error("[%s] gradient-norm failure %s at step %d: %s",
                     __abbr__, exc.code, step, exc)
        raise V04TError(exc.code, f"gradient-norm failure: {exc}") \
            from exc
    except Exception as exc:
        status, error_code = "interrupted", type(exc).__name__
        logger.exception("[%s] infrastructure interruption at step %d",
                         __abbr__, step)
        raise V04TError("INFRASTRUCTURE_INTERRUPTION",
                        f"{type(exc).__name__}: {exc}") from exc
    finally:
        diag = None
        if checkpoints:
            dgen = torch.Generator().manual_seed(seed)
            db = dgen  # conditioning pair for the non-gating diagnostic
            cond_a = torch.randn(2, v04p.ffr.COND_IN_CHANNELS, 96, 96,
                                 generator=db)
            mask_a = (torch.rand(2, 96, generator=db) > 0.5).float()
            cond_b = torch.randn(2, v04p.ffr.COND_IN_CHANNELS, 96, 96,
                                 generator=db)
            mask_b = (torch.rand(2, 96, generator=db) > 0.5).float()
            diag = post_training_conditioning_diagnostic(
                model, cond_a, mask_a, cond_b, mask_b, seed)
        record = {
            "schema": RECORD_SCHEMA,
            "script": f"{__abbr__} v{__version__}",
            "spec": {"article": 176, "status": "FROZEN"},
            "plan": {"article": 178, "status": "FROZEN",
                     "amendments": ["A1"]},
            "arm": arm, "lambda": lam, "seed": seed, "attempt": attempt,
            "status": status, "error_code": error_code,
            "steps_completed": step,
            "config": {"optimizer": "Adam", "lr": LEARNING_RATE,
                       "batch": BATCH_SIZE, "epochs": N_EPOCHS,
                       "steps": TOTAL_STEPS, "precision": "float32",
                       "device": "cpu", "schedule": None,
                       "gradient_clipping": None,
                       "init_seed": seed},
            "commit_pin": head,
            "probe_record": {"path": str(probe_record_path),
                             "sha256": _file_sha256(
                                 Path(probe_record_path))},
            "gate_record": {"path": str(gate_record_path),
                            "sha256": _file_sha256(
                                Path(gate_record_path))},
            "history": history,
            "validation_history": val_history,
            "checkpoints": checkpoints,
            "post_training_conditioning_diagnostic": diag,
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
    if step != TOTAL_STEPS:                                  # safeguard C
        _fail("STEP_COUNT_MISMATCH",
              f"run completed {step} steps; the locked budget is "
              f"{TOTAL_STEPS} (final partial batch kept)")
    print(f"PASS {__abbr__} v{__version__} arm {arm} seed {seed} "
          f"attempt {attempt}: {TOTAL_STEPS} steps, "
          f"{len(checkpoints)} checkpoints")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="v04_train")
    ap.add_argument("--mode", choices=("run",), required=True,
                    help="authoritative single run only; fixtures f05/f06 "
                         "live in scripts.v04_fixtures")
    ap.add_argument("--data-root")
    ap.add_argument("--p4-stats2")
    ap.add_argument("--probe-record")
    ap.add_argument("--gate-record")
    ap.add_argument("--commit-pin")
    ap.add_argument("--out")
    ap.add_argument("--arm", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        missing = [k for k, v in (
            ("data-root", args.data_root),
            ("p4-stats2", args.p4_stats2),
            ("probe-record", args.probe_record),
            ("gate-record", args.gate_record),
            ("commit-pin", args.commit_pin),
            ("out", args.out),
            ("arm", args.arm), ("seed", args.seed)) if v is None]
        if missing:
            _fail("RUN_ARGS_MISSING", f"--mode run requires {missing}")
        return _run(args.data_root, args.p4_stats2, args.probe_record,
                    args.gate_record, args.commit_pin, args.out,
                    args.arm, args.seed)
    except V04TError as exc:
        logger.error("[%s] ERROR: %s", __abbr__, exc)
        return 2
    except (v04o.V04LError, v03l.V03LError) as exc:
        logger.error("[%s] objective ERROR: %s", __abbr__, exc)
        return 2
    except Exception:
        logger.exception("[%s] unexpected exception", __abbr__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# Changelog
# v0.1 (2026-10-12): initial implementation under V04SPEC v1.0 (176,
#   FROZEN) SS7/SS18 / V04PLAN v1.0 (178, FROZEN) SS8/SS9 (advisor GO,
#   Option 2 + safeguards A-D). Structure mirrored from V03T v0.5:
#   run-contract loaders (hash-verified probe + gate records), single-run
#   trainer (fresh V04M init per arm-seed, inherited config, V04L hybrid
#   objective consumed via density_only/rec_term), per-step/per-epoch
#   telemetry with the locked L_rec asymmetry, epoch checkpoints with
#   SHA-256, append-only sidecar-chained ledger enforcing max 3 attempts
#   / same-seed restart / numerical-failure terminal / seed-major order.
#   V04 specifics: arms {0, lambda_2'} (2 x 3 = 6 runs); checkpoint
#   identity fields + round-trip filename check (A); fail-closed
#   prerequisites -- commit pin vs repository HEAD, valid probe record,
#   all-PASS gate record (D); total budget 3258 asserted with the final
#   partial batch kept (C); checkpoint selection rule (argmin L_val,
#   earliest-epoch tie) carried here as the single source (f05 domain;
#   V04E consumes it later); non-gating post-training conditioning
#   diagnostic (178 SS8). Authoritative --mode run implemented but never
#   executed at v0.1 (no probe/gate records or commit pin exist yet).
# ---------------------------------------------------------------------------
