# SEQREF-MRI-V03DLTT v0.5 -- scripts.v03dlt_train
# LIFETIME: KEEP
# =============================================================================
# Purpose: V03DLT longer-training diagnostic trainer (SEQREF-MRI-V03DLT v1.3,
#          Article 190, DESIGN FROZEN 2026-10-18 + AM1-AM3; V03DLTPLAN v1.2, Article
#          189, FROZEN, B1). ONE (arm, seed, attempt) per invocation,
#          30 epochs = 32,580 optimiser steps, T-NSF (V03 model, unchanged).
#          The run loop (_run_core) is a line-by-line copy of V03T v0.5
#          `_run` (pin 3393fbee); V03T helpers are IMPORTED UNCHANGED.
# Differences from V03T v0.5 `_run` (the ONLY ones):
#   * budget: 30 epochs x 1086 = 32,580 steps (locked; the production entry
#     refuses any other value -- B2.1). _run_core takes the budget as
#     explicit arguments so the fixture-only entry in v03dlt_fixtures can
#     run a small synthetic budget without patching this module;
#   * arms {0, 2} only and a dedicated 6-run seed-major order (190 A3):
#     validate_arm_seed_dlt / expected_next_run_dlt / enforce_order_dlt
#     replace V03T's 4-arm, 12-pair logic; next_attempt_dlt additionally
#     refuses any attempt after failed_validity (190 A4);
#   * arm-2 lambda asserted bit-equal to the registered 0.6130341651761398;
#   * provenance: commit-pin check, P4 parent hashes (as verified by the
#     pinned loader ffr.load_p4s2_parent) asserted against the registered
#     pins and recorded; CPU model and thread count recorded (never set);
#     record schema seqref-v03dlt-train-record/1; checkpoint identity
#     fields + assertion;
#   * campaign-root guard: out_dir must be an EXISTING V03DLT_<pin8>
#     directory (created once by the launcher --mode new); any path under
#     a V03 result directory is refused; attempt artifacts are never
#     overwritten; orphaned attempt files stop the run;
#   * KeyboardInterrupt classified as interrupted/KEYBOARD_INTERRUPT
#     (V03T v0.5 has no such branch, so a Ctrl-C would otherwise record
#     status "completed"; required by 189 B1/B2.8; mirrors V04T F1);
#   * SIGINT default handler installed at startup (AM1.1 of 184, ported).
#   Filenames and ledger stay in V03 format (v03_run_ledger.json inside the
#   campaign root) so V03T's ledger functions apply unchanged.
# Locked (never CLI-tunable): arms {0, 2}; seeds {101, 202, 303}; Adam lr
#   1e-4; batch 32 (final partial batch kept); float32 CPU; no schedule;
#   no clipping; max 3 attempts; mask epoch index = checkpoint epoch - 1.
# CONVENTION: logger.error + typed raise (V03TError, the V03T identity, so
#   the copied exception taxonomy is unchanged). PASS(0)/ERROR(2). No
#   fallback, no mock, no placeholder, no silent pass.
# Changelog (NEW in v0.1, 2026-10-18):
#   * Introduced under V03DLTPLAN v1.0 (Article 189) B1: line-by-line copy
#     of V03T v0.5 _run with the registered differences above; V03T
#     helpers imported unchanged; production budget asserted at 32,580.
#   * verify_commit_pin and assert_checkpoint_identity are written here
#     against the 189 B1 contract; the V04T originals (v04_train.py at
#     10cedfde) were not in the verified reference archive.
# Changelog (v0.2, 2026-10-18, review fix R1):
#   * check_campaign_root requires the exact name V03DLT_<commit_pin[:8]>
#     (was: any V03DLT_ prefix).
#   * Structural breaches inside the run loop (ARTIFACT_EXISTS,
#     CHECKPOINT_IDENTITY_MISMATCH) now record failed_validity (terminal,
#     no retry) instead of interrupted.
# Changelog (v0.3, 2026-10-18, review fix R2; 190 AM2):
#   * STEP_COUNT_MISMATCH -> failed_validity (terminal, no retry); was
#     V03T's inherited interrupted. Set renamed VALIDITY_CODES.
#   * _run_core refuses an inconsistent budget (n_epochs x steps_per_epoch
#     != total_steps) before any training or ledger write.
#   * STUDY constant and header now cite 190 v1.2 (AM1, AM2).
# Changelog (v0.4, 2026-10-18; 190 AM3, 189 B1/B2.4):
#   * assert_import_origin: every loaded seqref_mri module (incl. __main__
#     under -m) must be a file tracked at the pin under repo_root; checked
#     after the pin check and again after model/data construction; refused
#     (IMPORT_ORIGIN_VIOLATION) before any training or ledger write.
#   * Train record gains import_origin {repo_root, modules_checked, rule}.
#   * STUDY constant and header cite 190 v1.3 (AM1-AM3).
# Changelog (v0.5, 2026-10-18, review fix R3):
#   * Import origin tightened: module files and namespace-package
#     directories must lie inside <repo_root>/seqref_mri; a namespace
#     directory must contain at least one tracked file at the pin.
#   * Header cites V03DLTPLAN v1.2.
# =============================================================================
from __future__ import annotations

import argparse
import logging
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from seqref_mri.src import free_flow_runtime as ffr
from seqref_mri.src.fastmri_data import FastMRISliceDataset
from seqref_mri.src.flows import v03_objective as v03l
from seqref_mri.scripts import train_free_flow as tff
from seqref_mri.scripts import v03_train as v03t
from seqref_mri.scripts.train_base import _collate, _prepare
from seqref_mri.scripts.train_free_flow import targets_from_prepared
from seqref_mri.scripts.v02_manifests import BATCH_SIZE, canonical_json
from seqref_mri.scripts.v02_train import derive_cmap_from_mask
from seqref_mri.scripts.v03_probe import V03PError, global_grad_norm
from seqref_mri.scripts.v03_train import (
    MAX_ATTEMPTS, SEEDS, TERMINAL_STATUSES, V03TError, _file_sha256,
    _validate, _write_with_sidecar, append_entry, checkpoint_name,
    classify_gradient_error, classify_objective_error, lambda_for_arm,
    load_ledger, load_probe_record, run_record_name)

logger = logging.getLogger("seqref_mri.v03dlt_train")

__version__ = "0.5"
__abbr__ = "SEQREF-MRI-V03DLTT"

STUDY = "SEQREF-MRI-V03DLT v1.3 (Article 190, FROZEN + AM1-AM3)"
RECORD_SCHEMA_DLT = "seqref-v03dlt-train-record/1"
LEARNING_RATE = v03t.LEARNING_RATE        # 1e-4, inherited unchanged
STEPS_PER_EPOCH = v03t.STEPS_PER_EPOCH    # 1086, inherited unchanged
N_VAL_BATCHES = v03t.N_VAL_BATCHES        # 100, inherited unchanged
N_EPOCHS_DLT = 30                         # locked (190 A3)
TOTAL_STEPS_DLT = 32580                   # 30 x 1086, locked (190 A3)
ARMS_DLT = (0, 2)                         # locked (190 A3)
ORDER_DLT = [(s, a) for s in SEEDS for a in ARMS_DLT]   # seed-major, 6 runs
REGISTERED_LAMBDA_ARM2 = 0.6130341651761398             # 190 A3 (D3)
REGISTERED_P4_FILE_SHA256 = (
    "db0bf2c2e0cd3bd5e1ed86bf208198199f16df5b2fbbe0cc72903c8791f85684")
REGISTERED_P4_SEMANTIC_SHA256 = (
    "27bd5b693548d80b2bc837853469b5e62f3da6f0cb2184e5db85fde5c9c91008")
CAMPAIGN_PREFIX = "V03DLT_"               # campaign root V03DLT_<pin8>
LEDGER_NAME = "v03_run_ledger.json"       # V03 format inside the root
V03_RESULT_DIRNAMES = ("v03_train", "v03_probe", "v03_gate", "v03_eval",
                       "v03_classify", "v03_figures")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
# Contract breaches raised inside the run loop: terminal failed_validity,
# never a retryable interruption (190 AM2; review fixes R1, R2).
VALIDITY_CODES = frozenset({"ARTIFACT_EXISTS",
                            "CHECKPOINT_IDENTITY_MISMATCH",
                            "STEP_COUNT_MISMATCH"})


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V03TError(code, message)


# ---------------------------------------------------------------------------
# Registered differences: arms/order/attempts, budget, lambda, provenance,
# campaign-root guard, no overwrite
# ---------------------------------------------------------------------------

def validate_arm_seed_dlt(arm: int, seed: int) -> None:
    """190 A3: arms {0, 2} only (arms 1 and 3 refused); locked seeds."""
    if arm not in ARMS_DLT:
        _fail("ARM_INVALID", f"arm {arm!r} not in the V03DLT arms {ARMS_DLT}")
    if seed not in SEEDS:
        _fail("SEED_INVALID", f"seed {seed!r} not in locked set {SEEDS}")


def expected_next_run_dlt(ledger: dict):
    """The single runnable (arm, seed) under the 6-run seed-major order. A
    run is settled once it holds a TERMINAL entry (completed,
    numerical_failure or failed_validity); an unsettled run blocks every
    later run. Returns (arm, seed) or None when all six are settled."""
    for seed, arm in ORDER_DLT:
        entries = [e for e in ledger["entries"]
                   if e["seed"] == seed and e["arm"] == arm]
        if not any(e["status"] in TERMINAL_STATUSES for e in entries):
            return arm, seed
    return None


def enforce_order_dlt(ledger: dict, arm: int, seed: int) -> None:
    nxt = expected_next_run_dlt(ledger)
    if nxt is None:
        _fail("ORDER_VIOLATION",
              "all six V03DLT runs are settled; no further runs are "
              "permitted")
    if nxt != (arm, seed):
        _fail("ORDER_VIOLATION",
              f"6-run seed-major order: the next runnable run is arm "
              f"{nxt[0]} seed {nxt[1]}, not arm {arm} seed {seed}")


def next_attempt_dlt(ledger: dict, arm: int, seed: int) -> int:
    """V03T rule (completed / numerical_failure terminal; max 3 attempts)
    plus an explicit refusal after failed_validity (190 A4)."""
    for e in ledger["entries"]:
        if e["arm"] == arm and e["seed"] == seed \
                and e["status"] == "failed_validity":
            _fail("RERUN_POLICY_VIOLATION",
                  f"arm {arm} seed {seed} recorded failed_validity (attempt "
                  f"{e['attempt']}); validity failure is terminal, no "
                  f"restart is permitted")
    return v03t.next_attempt(ledger, arm, seed)


def assert_locked_budget(n_epochs: int, steps_per_epoch: int,
                         total_steps: int) -> None:
    """B2.1: the production budget is exactly 30 x 1086 = 32,580 steps;
    any other value is refused."""
    if (n_epochs, steps_per_epoch, total_steps) != (
            N_EPOCHS_DLT, STEPS_PER_EPOCH, TOTAL_STEPS_DLT) \
            or n_epochs * steps_per_epoch != total_steps:
        _fail("BUDGET_MISMATCH",
              f"budget {n_epochs} x {steps_per_epoch} = {total_steps} != the "
              f"locked {N_EPOCHS_DLT} x {STEPS_PER_EPOCH} = "
              f"{TOTAL_STEPS_DLT}")


def lambda_for_arm_dlt(rec_probe: dict, arm: int) -> float:
    """B2.2: arm 0 -> exactly 0.0; arm 2 -> the probe-record lambda at
    alpha 0.5, bit-equal to the registered value."""
    lam = lambda_for_arm(rec_probe, arm)
    if arm == 0 and lam != 0.0:
        _fail("LAMBDA_MISMATCH", f"arm 0 lambda {lam!r} is not 0.0")
    if arm == 2 and lam != REGISTERED_LAMBDA_ARM2:
        _fail("LAMBDA_MISMATCH",
              f"arm 2 lambda {lam!r} != registered "
              f"{REGISTERED_LAMBDA_ARM2!r} (190 A3)")
    return lam


def assert_p4_pins(p4: dict) -> None:
    """B2.7: the P4 parent, as verified by the pinned loader, must carry
    the registered file and semantic hashes (refusal before training)."""
    if p4.get("file_sha256") != REGISTERED_P4_FILE_SHA256 \
            or p4.get("semantic_sha256") != REGISTERED_P4_SEMANTIC_SHA256:
        _fail("P4_PIN_MISMATCH",
              f"P4 parent hashes file={p4.get('file_sha256')!r} "
              f"semantic={p4.get('semantic_sha256')!r} differ from the "
              f"registered pins (190 A3)")


def check_campaign_root(out_dir: str, commit_pin: str) -> Path:
    """B2.4 path guard: the EXISTING campaign root named exactly
    V03DLT_<commit_pin[:8]> (created once by the launcher); nothing under
    a V03 result directory."""
    out = Path(out_dir).resolve()
    hit = [d for d in V03_RESULT_DIRNAMES if d in out.parts]
    if hit:
        _fail("V03_EVIDENCE_PATH_REFUSED",
              f"output {out} lies under the V03 result directory {hit[0]}; "
              f"V03DLT never writes there")
    expected = f"{CAMPAIGN_PREFIX}{commit_pin[:8]}"
    if out.name != expected:
        _fail("CAMPAIGN_ROOT_INVALID",
              f"output {out} is not the campaign root {expected} for pin "
              f"{commit_pin}")
    if not out.is_dir():
        _fail("CAMPAIGN_ROOT_MISSING",
              f"campaign root {out} does not exist; it is created once by "
              f"the launcher (--mode new), never by the trainer")
    return out


def refuse_existing(path: Path) -> None:
    """Attempt artifacts are never overwritten (B2.4)."""
    for p in (path, path.with_suffix(path.suffix + ".sha256")):
        if p.exists():
            _fail("ARTIFACT_EXISTS",
                  f"{p} already exists; attempt artifacts are never "
                  f"overwritten")


def refuse_attempt_artifacts(out: Path, arm: int, seed: int,
                             attempt: int) -> None:
    """Second line of defence behind the launcher preflight (B4): no file
    of the attempt about to start may exist (hard-kill orphans)."""
    found = sorted(p.name for p in
                   out.glob(f"v03_arm{arm}_seed{seed}_attempt{attempt}_*"))
    if found:
        _fail("ORPHANED_ATTEMPT_ARTIFACTS",
              f"attempt {attempt} of arm {arm} seed {seed} already has "
              f"files {found}; stop for review, never auto-retry")


def _git(repo_root: Path, *args: str) -> str:
    try:
        res = subprocess.run(["git", "-C", str(repo_root), *args],
                             capture_output=True, text=True, check=False)
    except OSError as exc:
        _fail("COMMIT_PIN_UNVERIFIABLE", f"git not runnable: {exc}")
    if res.returncode != 0:
        _fail("COMMIT_PIN_UNVERIFIABLE",
              f"git {' '.join(args)} failed (rc={res.returncode}): "
              f"{res.stderr.strip()}")
    return res.stdout


def verify_commit_pin(commit_pin: str, repo_root: Path) -> str:
    """The running source is exactly the registered implementation pin:
    full 40-hex pin, HEAD equal to it, clean worktree (no modified or
    untracked files)."""
    if not isinstance(commit_pin, str) or not _HEX40.match(commit_pin):
        _fail("COMMIT_PIN_INVALID",
              f"commit pin {commit_pin!r} is not a full 40-hex SHA-1")
    head = _git(repo_root, "rev-parse", "HEAD").strip()
    if head != commit_pin:
        _fail("COMMIT_PIN_MISMATCH",
              f"HEAD {head} != registered pin {commit_pin}")
    dirty = _git(repo_root, "status", "--porcelain").strip()
    if dirty:
        _fail("WORKTREE_DIRTY",
              f"worktree {repo_root} is not clean:\n{dirty}")
    return head


def _seqref_modules() -> list[tuple[str, object]]:
    """Loaded seqref_mri modules, including __main__ when run with -m."""
    mods = []
    for name, mod in sorted(sys.modules.items()):
        if mod is None:
            continue
        real = name
        if name == "__main__":
            spec = getattr(mod, "__spec__", None)
            real = getattr(spec, "name", None) or ""
        if real == "seqref_mri" or real.startswith("seqref_mri."):
            mods.append((real, mod))
    return mods


def assert_import_origin(repo_root: Path) -> int:
    """190 AM3 / 189 B1, B2.4: every loaded seqref_mri module must come
    from a file TRACKED in the pin checkout under repo_root/seqref_mri.
    Untracked files (including those hidden by .git/info/exclude) and
    files outside the checkout are refused. 'Unmodified' is enforced by
    verify_commit_pin's clean-status check (modified tracked files are
    always reported, exclusions notwithstanding). Raises before any
    training or ledger write. Returns the number of modules checked."""
    root = repo_root.resolve()
    pkg_root = root / "seqref_mri"
    tracked = {p for p in
               _git(root, "ls-files", "-z", "--", "seqref_mri").split("\0")
               if p}
    if not tracked:
        _fail("IMPORT_ORIGIN_VIOLATION",
              f"no tracked seqref_mri files under {root}")
    mods = _seqref_modules()
    if not mods:
        _fail("IMPORT_ORIGIN_VIOLATION", "no seqref_mri modules loaded")
    bad = []
    for name, mod in mods:
        f = getattr(mod, "__file__", None)
        if f is None:                       # namespace package: directory
            dirs = [Path(p).resolve() for p in getattr(mod, "__path__", [])]
            if not dirs:
                bad.append((name, "no __file__ and no __path__"))
            for d in dirs:
                # Must lie inside <root>/seqref_mri AND contain at least one
                # tracked package file (review fix R3).
                try:
                    d.relative_to(pkg_root)
                    rel_d = d.relative_to(root).as_posix()
                except ValueError:
                    bad.append((name, f"package directory outside "
                                      f"{pkg_root}: {d}"))
                    continue
                if not any(t.startswith(rel_d + "/") for t in tracked):
                    bad.append((name, f"package directory has no tracked "
                                      f"files at the pin: {rel_d}"))
            continue
        rp = Path(f).resolve()
        try:
            rp.relative_to(pkg_root)
            rel = rp.relative_to(root).as_posix()
        except ValueError:
            bad.append((name, f"outside {pkg_root}: {rp}"))
            continue
        if rel not in tracked:
            bad.append((name, f"not tracked at the pin: {rel}"))
    if bad:
        _fail("IMPORT_ORIGIN_VIOLATION",
              "seqref_mri modules not loaded from tracked pin files: "
              + "; ".join(f"{n} ({why})" for n, why in bad))
    logger.info("[%s] import origin verified: %d seqref_mri modules from "
                "tracked files under %s", __abbr__, len(mods), root)
    return len(mods)


def assert_checkpoint_identity(entry: dict, arm: int, seed: int,
                               attempt: int, lam: float) -> None:
    """Each checkpoint entry is bound to its run identity and filename."""
    expected = {"arm": arm, "seed": seed, "attempt": attempt,
                "lambda": lam, "init_seed": seed,
                "file": checkpoint_name(arm, seed, attempt, entry["epoch"])}
    bad = {k: (entry.get(k), v) for k, v in expected.items()
           if entry.get(k) != v}
    if bad:
        _fail("CHECKPOINT_IDENTITY_MISMATCH",
              f"checkpoint entry identity differs (got, expected): {bad}")


def cpu_model() -> str:
    """CPU model from /proc/cpuinfo (Linux host); recorded, never used."""
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError as exc:
        _fail("CPU_MODEL_UNREADABLE", f"/proc/cpuinfo unreadable: {exc}")
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    _fail("CPU_MODEL_UNREADABLE", "/proc/cpuinfo has no 'model name' line")


# ---------------------------------------------------------------------------
# Run loop: line-by-line copy of V03T v0.5 _run (registered differences
# marked "V03DLT:"; everything else verbatim).
# ---------------------------------------------------------------------------

def _run_core(data_root: str, p4_stats2: str, probe_record_path: str,
              commit_pin: str, out_dir: str, arm: int, seed: int, *,
              n_epochs: int, steps_per_epoch: int, total_steps: int,
              repo_root: Path | None = None) -> int:
    validate_arm_seed_dlt(arm, seed)                         # V03DLT: arms
    if n_epochs < 1 or steps_per_epoch < 1 \
            or n_epochs * steps_per_epoch != total_steps:   # V03DLT: AM2
        _fail("BUDGET_MISMATCH",
              f"inconsistent budget {n_epochs} x {steps_per_epoch} != "
              f"{total_steps}; refused before any training or ledger write")
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    head = verify_commit_pin(commit_pin, repo_root)          # V03DLT: pin
    assert_import_origin(repo_root)                          # V03DLT: AM3
    rec_probe = load_probe_record(probe_record_path)
    lam = lambda_for_arm_dlt(rec_probe, arm)                 # V03DLT: lambda
    cpu = cpu_model()                                        # V03DLT: prov.
    out = check_campaign_root(out_dir, head)                 # V03DLT: guard
    ledger_path = out / LEDGER_NAME
    enforce_order_dlt(load_ledger(ledger_path), arm, seed)   # V03DLT: order
    attempt = next_attempt_dlt(load_ledger(ledger_path), arm, seed)
    refuse_attempt_artifacts(out, arm, seed, attempt)        # V03DLT: guard
    logger.info("[%s] arm %d (lambda=%.6g) seed %d attempt %d/%d",
                __abbr__, arm, lam, seed, attempt, MAX_ATTEMPTS)

    p4 = ffr.load_p4s2_parent(p4_stats2)
    assert_p4_pins(p4)                                       # V03DLT: P4
    model = ffr.build_model(init_seed=seed)     # fresh init, arm-seed
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ds_train = FastMRISliceDataset(data_root, split="train", mode="train")
    # Amendment A2: validation = train split in eval mode (deterministic
    # masks, training distribution). The official val split IS the
    # 199-volume holdout and is never touched by this study (190 A7).
    ds_valid = FastMRISliceDataset(data_root, split="train", mode="eval")
    val_loader = DataLoader(ds_valid, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0,
                            collate_fn=_collate)
    if len(val_loader) < N_VAL_BATCHES:
        _fail("VALIDATION_DATA_SHORT",
              f"validation split yields {len(val_loader)} batches; the "
              f"locked validation requires {N_VAL_BATCHES}")
    # V03DLT: AM3 -- re-check after model/data construction, which may
    # import further seqref_mri modules lazily.
    n_modules = assert_import_origin(repo_root)

    history, val_history, checkpoints = [], [], []
    step = 0
    t0 = time.time()
    status, error_code = "completed", None
    try:
        for ep in range(n_epochs):                           # V03DLT: budget
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
                                __abbr__, step, total_steps, float(loss))
            if step != (ep + 1) * steps_per_epoch:           # V03DLT: budget
                status, error_code = "failed_validity", \
                    "STEP_COUNT_MISMATCH"                    # V03DLT: AM2
                _fail("STEP_COUNT_MISMATCH",
                      f"epoch {ep} ended at step {step}; expected "
                      f"{(ep + 1) * steps_per_epoch}")
            vals = _validate(model, val_loader, p4, lam)
            ck = out / checkpoint_name(arm, seed, attempt,
                                     ep + 1)
            refuse_existing(ck)                              # V03DLT: guard
            torch.save(model.state_dict(), ck)
            ck_sha = _file_sha256(ck)
            entry = {"epoch": ep + 1, "file": ck.name,       # V03DLT: ident.
                     "file_sha256": ck_sha, "arm": arm, "seed": seed,
                     "attempt": attempt, "lambda": lam, "init_seed": seed,
                     **vals}
            assert_checkpoint_identity(entry, arm, seed, attempt, lam)
            checkpoints.append(entry)
            val_history.append({"epoch": ep + 1, **vals})
            logger.info("[%s] epoch %d: %s", __abbr__, ep + 1, vals)
    except V03TError as exc:
        if status == "completed":
            if exc.code == "NUMERICAL_FAILURE":
                status = "numerical_failure"
            elif exc.code in VALIDITY_CODES:                 # V03DLT: AM2
                status = "failed_validity"
            else:
                status = "interrupted"
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
    except KeyboardInterrupt:                                # V03DLT: SIGINT
        # Ctrl+C / forwarded SIGINT: an infrastructure interruption, never
        # a settled run (189 B1/B2.8; mirrors V04T F1).
        status, error_code = "interrupted", "KEYBOARD_INTERRUPT"
        logger.error("[%s] keyboard interruption at step %d", __abbr__, step)
        raise
    except Exception as exc:
        status, error_code = "interrupted", type(exc).__name__
        logger.exception("[%s] infrastructure interruption at step %d",
                         __abbr__, step)
        raise V03TError("INFRASTRUCTURE_INTERRUPTION",
                        f"{type(exc).__name__}: {exc}") from exc
    finally:
        record = {
            "schema": RECORD_SCHEMA_DLT,                     # V03DLT: schema
            "script": f"{__abbr__} v{__version__}",
            "spec": {"article": 190, "status": "FROZEN"},
            "plan": {"article": 189, "status": "FROZEN"},
            "arm": arm, "lambda": lam, "seed": seed, "attempt": attempt,
            "status": status, "error_code": error_code,
            "steps_completed": step,
            "config": {"optimizer": "Adam", "lr": LEARNING_RATE,
                       "batch": BATCH_SIZE, "epochs": n_epochs,
                       "steps": total_steps, "precision": "float32",
                       "device": "cpu", "schedule": None,
                       "gradient_clipping": None,
                       "init_seed": seed,
                       "study": STUDY},
            "commit_pin": head,                              # V03DLT: prov.
            "import_origin": {"repo_root": str(Path(repo_root).resolve()),
                              "modules_checked": n_modules,
                              "rule": "tracked at pin; clean status"},
            "p4_stats2": {"path": str(p4_stats2),
                          "file_sha256": p4["file_sha256"],
                          "semantic_sha256": p4["semantic_sha256"]},
            "probe_record": {"path": str(probe_record_path),
                             "sha256": _file_sha256(
                                 Path(probe_record_path))},
            "history": history,
            "validation_history": val_history,
            "checkpoints": checkpoints,
            "wall_clock_s": time.time() - t0,
            "environment": {"python": platform.python_version(),
                            "torch": torch.__version__,
                            "platform": platform.platform(),
                            "cpu_model": cpu,
                            "threads": torch.get_num_threads()},
        }
        rpath = out / run_record_name(arm, seed, attempt)
        refuse_existing(rpath)                               # V03DLT: guard
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
    if step != total_steps:
        _fail("STEP_COUNT_MISMATCH",
              f"run completed {step} steps; the locked budget is "
              f"{total_steps} (final partial batch kept)")
    print(f"PASS {__abbr__} v{__version__} arm {arm} seed {seed} "
          f"attempt {attempt}: {total_steps} steps, "
          f"{len(checkpoints)} checkpoints")
    return 0


def _run(data_root: str, p4_stats2: str, probe_record_path: str,
         commit_pin: str, out_dir: str, arm: int, seed: int) -> int:
    """Production entry: the locked 30 x 1086 = 32,580-step budget only."""
    assert_locked_budget(N_EPOCHS_DLT, STEPS_PER_EPOCH, TOTAL_STEPS_DLT)
    return _run_core(data_root, p4_stats2, probe_record_path, commit_pin,
                     out_dir, arm, seed, n_epochs=N_EPOCHS_DLT,
                     steps_per_epoch=STEPS_PER_EPOCH,
                     total_steps=TOTAL_STEPS_DLT)


def _disposition_name(h) -> str:
    if h is signal.SIG_IGN:
        return "SIG_IGN"
    if h is signal.SIG_DFL:
        return "SIG_DFL"
    if h is signal.default_int_handler:
        return "default_int_handler"
    return "other" if h is not None else "unknown"


def install_sigint() -> str:
    """Make SIGINT raise KeyboardInterrupt regardless of the inherited
    disposition (a launcher background job inherits SIG_IGN); return the
    inherited disposition's name (184 AM1.1, ported)."""
    inherited = _disposition_name(signal.getsignal(signal.SIGINT))
    signal.signal(signal.SIGINT, signal.default_int_handler)
    return inherited


def main() -> int:
    inherited = install_sigint()
    ap = argparse.ArgumentParser(prog="v03dlt_train")
    ap.add_argument("--mode", choices=("run",), required=True,
                    help="single 30-epoch diagnostic run; no budget option; "
                         "fixtures live in scripts.v03dlt_fixtures")
    ap.add_argument("--data-root")
    ap.add_argument("--p4-stats2")
    ap.add_argument("--probe-record")
    ap.add_argument("--commit-pin")
    ap.add_argument("--out")
    ap.add_argument("--arm", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    logger.info("[%s] SIGINT disposition inherited=%s; installed "
                "default_int_handler", __abbr__, inherited)
    try:
        missing = [k for k, v in (
            ("data-root", args.data_root),
            ("p4-stats2", args.p4_stats2),
            ("probe-record", args.probe_record),
            ("commit-pin", args.commit_pin),
            ("out", args.out),
            ("arm", args.arm), ("seed", args.seed)) if v is None]
        if missing:
            _fail("RUN_ARGS_MISSING", f"--mode run requires {missing}")
        return _run(args.data_root, args.p4_stats2, args.probe_record,
                    args.commit_pin, args.out, args.arm, args.seed)
    except KeyboardInterrupt:
        logger.error("[%s] keyboard interruption", __abbr__)
        return 2
    except V03TError as exc:
        logger.error("[%s] ERROR: %s", __abbr__, exc)
        return 2
    except v03l.V03LError as exc:
        logger.error("[%s] objective ERROR: %s", __abbr__, exc)
        return 2
    except Exception:
        logger.exception("[%s] unexpected exception", __abbr__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
