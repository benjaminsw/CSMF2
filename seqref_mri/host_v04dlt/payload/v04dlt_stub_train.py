# =============================================================================
# SEQREF-MRI-V04DLTSTUB v0.3 -- scripts.v04dlt_stub_train  (FIXTURE-ONLY)
# LIFETIME: KEEP
# Purpose: synthetic training stub for launcher fixture B2.9 of
#          SEQREF-MRI-V04DLT v1.0 (Article 184). NEVER a production trainer:
#          it trains nothing, writes small synthetic checkpoints, and runs
#          only inside a fixture scratch directory marked by the file
#          V04DLT_STUB_SCRATCH. It refuses the real fastMRI data root, any
#          *.h5 source, and any output root outside its scratch directory.
# Behaviour per (arm, seed, attempt) comes from <scratch>/stub_plan.json:
#   succeed [delay_s] | numerical | infra | sleep (until SIGINT ->
#   interrupted/KEYBOARD_INTERRUPT) | no_entry_on_sigint (until SIGINT ->
#   exit without a ledger entry). SIGINT count, start and finally markers
#   are written to the scratch directory (never into the campaign root).
# Ledger/record FORMAT and order/attempt policy use the pinned V04T
#   functions (load_ledger, enforce_order, next_attempt, append_entry,
#   _write_with_sidecar, checkpoint_name, run_record_name). Control flow is
#   NOT identical to V04T: the stub writes its record and ledger entry
#   AFTER its finally block (V04T writes them inside its finally).
# Signals: first calls the production v04dlt_train.install_sigint() (R1)
#   and records the inherited disposition (stub_sigdisp_<key>.txt), then
#   installs its own counting handler so B2.9 can prove exactly one SIGINT.
# CONVENTION: logger.error + typed exit 2 on refusal. No silent pass.
# Changelog (NEW in v0.1, 2026-10-17): introduced for V04DLT B2.9.
# Changelog (v0.2, 2026-10-17): calls production install_sigint() first and
#   records the inherited SIGINT disposition; header corrected (record and
#   ledger are written after, not inside, the finally block).
# Changelog (v0.3, 2026-10-17): 'full' mode writes valid 30-epoch /
#   32,580-step record metadata and tiny checkpoints (B2.10 builder) for the
#   B2.9 launcher -> summary handoff; a non-completed 'full' run is refused.
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

import torch

from seqref_mri.scripts import v04_train as v04t
from seqref_mri.scripts import v04dlt_train as dlt
from seqref_mri.scripts import v04dlt_fixtures as fx
from seqref_mri.scripts.v02_manifests import canonical_json

logger = logging.getLogger("seqref_mri.v04dlt_stub_train")
__version__ = "0.3"
__abbr__ = "SEQREF-MRI-V04DLTSTUB"
MARKER = "V04DLT_STUB_SCRATCH"
REAL_DATA_ROOT = Path("/home/benjamin/CSMFII/seqref_mri/data/fastmri")
STATE = {"sigints": 0}


def _refuse(code: str, message: str) -> int:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    return 2


def _on_sigint(signum, frame):
    STATE["sigints"] += 1
    if STATE["sigints"] == 1:
        raise KeyboardInterrupt


def main() -> int:
    inherited = dlt.install_sigint()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="v04dlt_stub_train")
    for opt in ("--mode", "--data-root", "--p4-stats2", "--probe-record",
                "--gate-record", "--commit-pin", "--out"):
        ap.add_argument(opt, required=True)
    ap.add_argument("--arm", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    scratch = out.parent.parent
    if not (scratch / MARKER).is_file():
        return _refuse("STUB_OUTPUT_OUTSIDE_SCRATCH",
                       f"{out} is not inside a marked fixture scratch")
    data = Path(a.data_root).resolve()
    if data == REAL_DATA_ROOT or REAL_DATA_ROOT in data.parents:
        return _refuse("STUB_REAL_DATA_REFUSED", str(data))
    if scratch not in data.parents or any(data.rglob("*.h5")):
        return _refuse("STUB_DATA_REFUSED",
                       f"{data} is outside scratch or holds *.h5 sources")
    led_path = out / "v04_run_ledger.json"
    led = v04t.load_ledger(led_path)
    v04t.enforce_order(led, a.arm, a.seed)
    att = v04t.next_attempt(led, a.arm, a.seed)
    key = f"{a.arm}_{a.seed}_{att}"
    plan = json.loads((scratch / "stub_plan.json").read_text())
    act = plan.get(key, plan.get("default", ["succeed", 0]))
    (scratch / f"stub_sigdisp_{key}.txt").write_text(f"{inherited}\n")
    signal.signal(signal.SIGINT, _on_sigint)
    lam = 0.0 if a.arm == 0 else 2.0
    ck = out / v04t.checkpoint_name(a.arm, a.seed, att, 1)
    status, code, cks, full = "completed", None, [], None
    try:
        if act[0] == "full":
            full = fx._b210_record(a.arm, a.seed, att, "completed",
                                   dlt.TOTAL_STEPS_DLT, dlt.N_EPOCHS_DLT,
                                   dlt.TOTAL_STEPS_DLT, None, pin=a.commit_pin)
            for c in full["checkpoints"]:
                (out / c["file"]).write_bytes(fx._ck_bytes(c["file"]))
            cks = full["checkpoints"]
        else:
            torch.save({"w": torch.zeros(1)}, ck)
            cks.append({"epoch": 1, "file": ck.name,
                        "file_sha256": v04t._file_sha256(ck), "arm": a.arm,
                        "seed": a.seed, "attempt": att, "lambda": lam,
                        "init_seed": a.seed, "val_NLL": 1.0, "val_L_rec": 1.0,
                        "L_val": 1.0})
        (scratch / f"stub_started_{key}.txt").write_text(f"{time.time()}\n")
        if act[0] in ("succeed", "full"):
            time.sleep(float(act[1]) if len(act) > 1 else 0.0)
        elif act[0] == "numerical":
            status, code = "numerical_failure", "NUMERICAL_FAILURE"
        elif act[0] == "infra":
            status, code = "interrupted", "INFRASTRUCTURE_INTERRUPTION"
        elif act[0] in ("sleep", "no_entry_on_sigint"):
            while True:
                time.sleep(0.05)
        else:
            return _refuse("STUB_PLAN_INVALID", f"{key}: {act!r}")
    except KeyboardInterrupt:
        status, code = "interrupted", "KEYBOARD_INTERRUPT"
        if act[0] == "no_entry_on_sigint":
            (scratch / f"stub_sigint_{key}.txt").write_text(
                f"{STATE['sigints']}\n")
            return 2
    finally:
        (scratch / f"stub_finally_{key}.txt").write_text(f"{status}\n")
    if full is not None and status != "completed":
        return _refuse("STUB_FULL_NOT_COMPLETED", f"{key}: {status}")
    rec = full or {
           "schema": v04t.RECORD_SCHEMA, "script": f"{__abbr__} v{__version__}",
           "arm": a.arm, "lambda": lam, "seed": a.seed, "attempt": att,
           "status": status, "error_code": code, "steps_completed": 1,
           "history": [], "checkpoints": cks,
           "validation_history": [{"epoch": 1, "val_NLL": 1.0,
                                   "val_L_rec": 1.0, "L_val": 1.0}]}
    rpath = out / v04t.run_record_name(a.arm, a.seed, att)
    rsha = v04t._write_with_sidecar(rpath, canonical_json(rec))
    v04t.append_entry(led_path, {
        "arm": a.arm, "lambda": rec["lambda"], "seed": a.seed, "attempt": att,
        "status": status, "error_code": code,
        "steps_completed": rec["steps_completed"],
        "wall_clock_s": rec.get("wall_clock_s", 0.0), "record_file": rpath.name,
        "record_sha256": rsha, "checkpoints": [c["file_sha256"] for c in cks]})
    (scratch / f"stub_sigint_{key}.txt").write_text(f"{STATE['sigints']}\n")
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
