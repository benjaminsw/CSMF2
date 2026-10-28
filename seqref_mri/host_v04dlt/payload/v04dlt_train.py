# =============================================================================
# SEQREF-MRI-V04DLTT v0.2 -- scripts.v04dlt_train
# LIFETIME: KEEP
# Purpose: V04DLT longer-training diagnostic trainer (SEQREF-MRI-V04DLT v1.0,
#          Article 184, DESIGN FROZEN 2026-10-17, B1). ONE (arm, seed,
#          attempt) per invocation, 30 epochs = 32,580 optimiser steps.
#          The run loop (_run_core) is a line-by-line copy of V04T v0.3
#          `_run`; every other helper is IMPORTED UNCHANGED from
#          seqref_mri.scripts.v04_train (prerequisite loaders, commit-pin
#          check, ledger functions, order/attempt policy, checkpoint
#          naming/identity, selection rule, _validate, the non-gating
#          conditioning diagnostic, error classifiers, _write_with_sidecar).
# Differences from V04T v0.3 `_run` (the ONLY ones, Article 184 B1):
#   * budget: 30 epochs x 1086 = 32,580 steps (locked here; the production
#     entry refuses any other value -- B2.1); _run_core takes the budget
#     as explicit arguments so the fixture-only entry in v04dlt_fixtures
#     can run a small synthetic budget without patching this module;
#   * record `script` / `config` fields (study id, budget, thread count --
#     recorded, never set);
#   * campaign-root guard: out_dir must be an EXISTING V04DLT_<pin8>
#     directory (created once by the launcher --mode new); any path under
#     the V04 evidence root V04TR_10cedfde is refused; attempt artifacts
#     (records, checkpoints and their sidecars) are never overwritten.
#   Filenames stay in V04 format; the ledger is v04_run_ledger.json inside
#   the campaign root (Article 184 B1/B2.3), so the unchanged V04 parser,
#   identity checks and load_ledger apply.
# Locked (never CLI-tunable): arms {0, lambda_2'}, seeds {101, 202, 303},
#   Adam lr 1e-4, batch 32 (final partial batch kept), float32 CPU, no
#   schedule, no clipping, max 3 attempts, seed-major order, mask epoch
#   index = checkpoint epoch - 1 (set_epoch(ep), ep = 0..29).
# CONVENTION: logger.error + typed raise (V04TError, the V04T identity, so
#   the copied exception taxonomy is unchanged). PASS(0)/ERROR(2). No
#   fallback, no mock, no placeholder, no silent pass.
# Changelog (NEW in v0.1, 2026-10-17):
#   * Introduced under V04DLT v1.0 (Article 184) B1: line-by-line copy of
#     V04T v0.3 _run with the three registered differences above; helpers
#     imported unchanged; production budget asserted at 32,580 steps.
# Changelog (v0.2, 2026-10-17, review fix R1):
#   * main() first installs signal.default_int_handler for SIGINT
#     (install_sigint): a launcher background job inherits SIGINT as
#     ignored, and Python then never raises KeyboardInterrupt.
#   * The inherited disposition is logged; training code is unchanged.
# =============================================================================
from __future__ import annotations

import argparse
import logging
import platform
import signal
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
from seqref_mri.scripts import v04_train as v04t
from seqref_mri.scripts.train_base import _collate, _prepare
from seqref_mri.scripts.train_free_flow import targets_from_prepared
from seqref_mri.scripts.v02_manifests import BATCH_SIZE, canonical_json
from seqref_mri.scripts.v02_train import derive_cmap_from_mask
from seqref_mri.scripts.v04_model import build_v04_model
from seqref_mri.scripts.v04_train import (
    MAX_ATTEMPTS, RECORD_SCHEMA, V04TError, _file_sha256,
    _validate, _write_with_sidecar, append_entry, assert_checkpoint_identity,
    checkpoint_name, classify_gradient_error, classify_objective_error,
    enforce_order, lambda_for_arm, load_gate_record, load_ledger,
    load_probe_record, next_attempt, post_training_conditioning_diagnostic,
    run_record_name, validate_arm_seed, verify_commit_pin)

logger = logging.getLogger("seqref_mri.v04dlt_train")

__version__ = "0.2"
__abbr__ = "SEQREF-MRI-V04DLTT"

STUDY = "SEQREF-MRI-V04DLT v1.0 (Article 184)"
LEARNING_RATE = v04t.LEARNING_RATE        # 1e-4, inherited unchanged
STEPS_PER_EPOCH = 1086                    # locked (V04T, 184 A3)
N_EPOCHS_DLT = 30                         # locked (184 A3)
TOTAL_STEPS_DLT = 32580                   # 30 x 1086, locked (184 A3)
CAMPAIGN_PREFIX = "V04DLT_"               # campaign root V04DLT_<pin8>
V04_EVIDENCE_DIRNAME = "V04TR_10cedfde"   # registered V04 evidence root


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V04TError(code, message)


# ---------------------------------------------------------------------------
# Registered differences: locked budget, campaign-root guard, no overwrite
# ---------------------------------------------------------------------------

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


def check_campaign_root(out_dir: str) -> Path:
    """B2.3 path guard: an EXISTING V04DLT_<pin8> campaign root (created
    once by the launcher); nothing under the V04 evidence root."""
    out = Path(out_dir).resolve()
    if V04_EVIDENCE_DIRNAME in out.parts:
        _fail("V04_EVIDENCE_PATH_REFUSED",
              f"output {out} lies under the registered V04 evidence root "
              f"{V04_EVIDENCE_DIRNAME}; V04DLT never writes there")
    if not out.name.startswith(CAMPAIGN_PREFIX):
        _fail("CAMPAIGN_ROOT_INVALID",
              f"output {out} is not a {CAMPAIGN_PREFIX}<pin8> campaign root")
    if not out.is_dir():
        _fail("CAMPAIGN_ROOT_MISSING",
              f"campaign root {out} does not exist; it is created once by "
              f"the launcher (--mode new), never by the trainer")
    return out


def refuse_existing(path: Path) -> None:
    """Attempt artifacts are never overwritten (B2.3)."""
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
                   out.glob(f"v04_arm{arm}_seed{seed}_attempt{attempt}_*"))
    if found:
        _fail("ORPHANED_ATTEMPT_ARTIFACTS",
              f"attempt {attempt} of arm {arm} seed {seed} already has "
              f"files {found}; stop for review, never auto-retry")


# ---------------------------------------------------------------------------
# Run loop: line-by-line copy of V04T v0.3 _run (registered differences
# marked "V04DLT:"; everything else verbatim).
# ---------------------------------------------------------------------------

def _run_core(data_root: str, p4_stats2: str, probe_record_path: str,
              gate_record_path: str, commit_pin: str, out_dir: str,
              arm: int, seed: int, *, n_epochs: int, steps_per_epoch: int,
              total_steps: int) -> int:
    validate_arm_seed(arm, seed)
    repo_root = Path(__file__).resolve().parents[2]
    head = verify_commit_pin(commit_pin, repo_root)          # D(1)
    rec_probe = load_probe_record(probe_record_path)         # D(2)
    rec_gate = load_gate_record(gate_record_path)            # D(3)
    lam = lambda_for_arm(rec_probe, arm)
    out = check_campaign_root(out_dir)                       # V04DLT: guard
    ledger_path = out / "v04_run_ledger.json"
    enforce_order(load_ledger(ledger_path), arm, seed)
    attempt = next_attempt(load_ledger(ledger_path), arm, seed)
    refuse_attempt_artifacts(out, arm, seed, attempt)        # V04DLT: guard
    logger.info("[%s] arm %d (lambda=%.6g) seed %d attempt %d/%d",
                __abbr__, arm, lam, seed, attempt, MAX_ATTEMPTS)

    p4 = ffr.load_p4s2_parent(p4_stats2)
    model = build_v04_model(init_seed=seed)     # fresh init, arm-seed (A)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ds_train = FastMRISliceDataset(data_root, split="train", mode="train")
    # A2 convention: validation = train split in eval mode (deterministic
    # masks, training distribution). The official val split IS the
    # 199-volume holdout and is never touched by this study.
    ds_valid = FastMRISliceDataset(data_root, split="train", mode="eval")
    val_loader = DataLoader(ds_valid, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0,
                            collate_fn=_collate)
    if len(val_loader) < v04t.N_VAL_BATCHES:
        _fail("VALIDATION_DATA_SHORT",
              f"validation split yields {len(val_loader)} batches; the "
              f"locked validation requires {v04t.N_VAL_BATCHES}")

    history, val_history, checkpoints = [], [], []
    step = 0
    t0 = time.time()
    status, error_code = "completed", None
    try:
        for ep in range(n_epochs):                           # V04DLT: budget
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
                                __abbr__, step, total_steps, float(loss))
            if step != (ep + 1) * steps_per_epoch:           # V04DLT: budget
                status, error_code = "interrupted", "STEP_COUNT_MISMATCH"
                _fail("STEP_COUNT_MISMATCH",
                      f"epoch {ep} ended at step {step}; expected "
                      f"{(ep + 1) * steps_per_epoch}")
            vals = _validate(model, val_loader, p4, lam)
            ck = out / checkpoint_name(arm, seed, attempt, ep + 1)
            refuse_existing(ck)                              # V04DLT: guard
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
    except KeyboardInterrupt:
        # Ctrl+C / forwarded SIGINT: an infrastructure interruption, never
        # a settled run (V04T F1 repair, inherited).
        status, error_code = "interrupted", "KEYBOARD_INTERRUPT"
        logger.error("[%s] keyboard interruption at step %d", __abbr__, step)
        raise
    except Exception as exc:
        status, error_code = "interrupted", type(exc).__name__
        logger.exception("[%s] infrastructure interruption at step %d",
                         __abbr__, step)
        raise V04TError("INFRASTRUCTURE_INTERRUPTION",
                        f"{type(exc).__name__}: {exc}") from exc
    finally:
        diag = None
        diag_interrupted = False
        if checkpoints:
            try:
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
            except KeyboardInterrupt:
                # V04T F3 repair (inherited): Ctrl+C during the non-gating
                # diagnostic never rewrites the training status; evidence
                # is written, then the interrupt is re-raised below.
                logger.error("[%s] keyboard interruption during "
                             "post-training diagnostic (non-gating)",
                             __abbr__)
                diag = {"error": "KeyboardInterrupt", "gating": False}
                diag_interrupted = True
            except Exception as exc:
                # V04T F2 repair (inherited): a diagnostic failure never
                # prevents record/ledger writes.
                logger.error("[%s] post-training conditioning diagnostic "
                             "failed (non-gating): %s: %s", __abbr__,
                             type(exc).__name__, exc)
                diag = {"error": f"{type(exc).__name__}: {exc}",
                        "gating": False}
        record = {
            "schema": RECORD_SCHEMA,
            "script": f"{__abbr__} v{__version__}",         # V04DLT: script
            "spec": {"article": 176, "status": "FROZEN"},
            "plan": {"article": 178, "status": "FROZEN",
                     "amendments": ["A1"]},
            "arm": arm, "lambda": lam, "seed": seed, "attempt": attempt,
            "status": status, "error_code": error_code,
            "steps_completed": step,
            "config": {"optimizer": "Adam", "lr": LEARNING_RATE,  # V04DLT
                       "batch": BATCH_SIZE, "epochs": n_epochs,
                       "steps": total_steps, "precision": "float32",
                       "device": "cpu", "schedule": None,
                       "gradient_clipping": None,
                       "init_seed": seed,
                       "study": STUDY,
                       "torch_num_threads": torch.get_num_threads()},
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
        refuse_existing(rpath)                               # V04DLT: guard
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
        if diag_interrupted:
            raise KeyboardInterrupt
    if step != total_steps:                                  # safeguard C
        _fail("STEP_COUNT_MISMATCH",
              f"run completed {step} steps; the locked budget is "
              f"{total_steps} (final partial batch kept)")
    print(f"PASS {__abbr__} v{__version__} arm {arm} seed {seed} "
          f"attempt {attempt}: {total_steps} steps, "
          f"{len(checkpoints)} checkpoints")
    return 0


def _run(data_root: str, p4_stats2: str, probe_record_path: str,
         gate_record_path: str, commit_pin: str, out_dir: str,
         arm: int, seed: int) -> int:
    """Production entry: the locked 30 x 1086 = 32,580-step budget only."""
    assert_locked_budget(N_EPOCHS_DLT, STEPS_PER_EPOCH, TOTAL_STEPS_DLT)
    return _run_core(data_root, p4_stats2, probe_record_path,
                     gate_record_path, commit_pin, out_dir, arm, seed,
                     n_epochs=N_EPOCHS_DLT, steps_per_epoch=STEPS_PER_EPOCH,
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
    """R1: make SIGINT raise KeyboardInterrupt regardless of the inherited
    disposition; return the inherited disposition's name."""
    inherited = _disposition_name(signal.getsignal(signal.SIGINT))
    signal.signal(signal.SIGINT, signal.default_int_handler)
    return inherited


def main() -> int:
    inherited = install_sigint()
    ap = argparse.ArgumentParser(prog="v04dlt_train")
    ap.add_argument("--mode", choices=("run",), required=True,
                    help="single 30-epoch diagnostic run; no budget option; "
                         "fixtures live in scripts.v04dlt_fixtures")
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
    logger.info("[%s] SIGINT disposition inherited=%s; installed "
                "default_int_handler", __abbr__, inherited)
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
    except KeyboardInterrupt:
        logger.error("[%s] keyboard interruption", __abbr__)
        return 2
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
