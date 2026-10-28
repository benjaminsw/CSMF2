# =============================================================================
# SEQREF-MRI-V04DLTF v0.3 -- scripts.v04dlt_fixtures
# LIFETIME: KEEP
# Purpose: required fixtures B2.0-B2.10 of SEQREF-MRI-V04DLT v1.0 (Article
#          184, DESIGN FROZEN 2026-10-17) for the longer-training diagnostic
#          trainer (V04DLTT), summary (V04DLTS) and launcher (V04DLTL).
#          Synthetic CPU data only, each fixture in its own scratch
#          directory; no real data root, no V04 evidence written, no holdout.
# Synthetic stand-ins (disclosed, fixture-confined, never production):
#   * synthetic single-coil h5 volumes and a synthetic P4 /2 artifact
#     (helpers reused unchanged from v04_fixtures, the f04 precedent);
#   * synthetic probe/gate records (the v04_fixtures f06 helpers);
#   * the registered P4 pins are swapped through load_p4s2_parent
#     __kwdefaults__ and V04T budget constants are swapped on the V04T
#     module (B2.0 only), each restored in `finally` (f04 precedent);
#   * B2.6 injects failures by wrapping objective functions inside the
#     fixture, restored in `finally`;
#   * B2.9 runs COPIES of the launcher. Default copy: only the trainer-
#     command line differs (fixture-only stub v04dlt_stub_train.py).
#     Signal cases: the trainer line points at a fixture-written shim that
#     imports the REAL v04dlt_train and calls its REAL main(), with only
#     _run replaced by a blocking stand-in (no data, no training). Hazard,
#     pending and control cases set SIGINT to SIG_IGN in the shim before
#     main() (simulating a `python &` spawn, which inherits SIG_IGN); the
#     control additionally disables install_sigint to show the hazard. Boundary cases: additionally the PY constant
#     points at a fixture wrapper that holds the launcher inside preflight.
#     Each copy asserts that only its declared lines differ.
#   * B2.10 builds tiny synthetic campaigns (checkpoint files of a few
#     bytes; valid 30-epoch / 32,580-step record metadata written directly,
#     no training) and synthetic V04 reference records; the summary's
#     registered V04 hash constants are swapped on the module for the
#     fixture's scope only and restored in `finally`.
# CONVENTION: PASS(0)/ERROR(2); every failure -> logger.error + typed code.
#   No fallback, no silent pass.
# Changelog (NEW in v0.1, 2026-10-17):
#   * Introduced under V04DLT v1.0 B2: B2.0 equivalence vs V04T v0.3 _run,
#     B2.1 budget, B2.2 lambda, B2.3 path guard, B2.4 two-step smoke via a
#     fixture-only entry, B2.5 30-epoch mask contract, B2.6 failure paths,
#     B2.7 mode/partial batch, B2.8 V04 regression suite, B2.9 launcher.
# Changelog (v0.2, 2026-10-17, review R1-R4):
#   * B2.9 adds real-main SIGINT startup, early-cancel (pending forward),
#     inherited-ignore control and launch-boundary CANCEL/STOP cases.
#   * B2.10 (new): summary on synthetic campaigns -- n = 0/1/2/3, terminal
#     and incomplete runs, missing evidence, inconsistent records, hashes,
#     overwrite refusal.
#   * B2.9 clean-up kills every recorded inhibitor, also after a failure.
# Changelog (v0.3, 2026-10-17, review R5/R6):
#   * B2.10 builder reproduces the launcher's full output (both run logs,
#     manifest sidecars); new manifest-sidecar and run-log cases.
#   * B2.9 handoff: a launcher-produced 2-session campaign -> summary.
# =============================================================================
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from seqref_mri.src import free_flow_runtime as ffr
from seqref_mri.src.fastmri_data import (TRAIN_BASE_SEED, FastMRISliceDataset,
                                         canonical_mask_seed,
                                         make_cartesian_mask, mask_counts)
from seqref_mri.scripts import v02_manifests as v02m
from seqref_mri.scripts import v04_fixtures as v04f
from seqref_mri.scripts import v04_objective as v04o
from seqref_mri.scripts import v04_train as v04t
from seqref_mri.scripts import v04dlt_train as dlt
from seqref_mri.scripts import v04dlt_summary as dls

logger = logging.getLogger("seqref_mri.v04dlt_fixtures")
__version__ = "0.3"
__abbr__ = "SEQREF-MRI-V04DLTF"

REPO = Path(__file__).resolve().parents[2]
REAL_DATA_ROOT = Path("/home/benjamin/CSMFII/seqref_mri/data/fastmri")
SEED = 101                       # first locked seed (pair order arm 0, arm 1)
B20_SLICES = 69                  # 32 + 32 + 5: three steps/epoch, partial batch
B20_EPOCHS = 2
B20_STEPS_PER_EPOCH = 3
B20_N_VAL = 2
SMALL_SLICES = 33                # 32 + 1: two steps/epoch, partial batch
SMALL_STEPS = 2
MASK_SAMPLE = (0, 17, 68)        # B2.5 fixed slice sample
HIST_KEYS = ("step", "epoch", "L_density", "L_rec", "lam_L_rec", "L_total",
             "grad_norm", "lr")
RUN_KEYS = ("status", "error_code", "steps_completed", "attempt", "lambda")
LEDGER_KEYS = ("status", "attempt", "steps_completed")


class FixtureFailure(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise FixtureFailure(code, message)


# ---------------------------------------------------------------------------
# Synthetic workspace (data + P4 + probe/gate records), confined to scratch
# ---------------------------------------------------------------------------

def _write_record(path: Path, rec: dict) -> None:
    payload = v02m.canonical_json(rec)
    path.write_bytes(payload)
    sha = v04t._file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha}  {path.name}\n")


def _workspace(tmp: Path, n_slices: int) -> dict:
    data_root = tmp / "data"
    split = data_root / "knee_singlecoil_train" / "singlecoil_train"
    split.mkdir(parents=True)
    v04f._f04_e2e_write_volume(split / "synthetic_volume.h5", n_slices)
    p4 = tmp / "p4" / "scaling_statistics.json"
    p4.parent.mkdir(parents=True)
    p4_file, p4_sem = v04f._f04_e2e_write_p4(p4)
    rec_dir = tmp / "records"
    rec_dir.mkdir()
    probe = rec_dir / "v04_probe_record.json"
    gate = rec_dir / "v04_gate_record.json"
    _write_record(probe, v04f._good_probe_record())
    _write_record(gate, v04f._good_gate_record())
    return {"tmp": tmp, "data_root": str(data_root), "p4": str(p4),
            "p4_pins": (p4_file, p4_sem), "probe": str(probe),
            "gate": str(gate),
            "lam": float(v04f._good_probe_record()["lambda_prime"])}


@contextlib.contextmanager
def _p4_pins(ws: dict):
    kw = ffr.load_p4s2_parent.__kwdefaults__
    saved = (kw["expected_file_sha"], kw["expected_semantic_sha"])
    kw["expected_file_sha"], kw["expected_semantic_sha"] = ws["p4_pins"]
    try:
        yield
    finally:
        kw["expected_file_sha"], kw["expected_semantic_sha"] = saved


@contextlib.contextmanager
def _patched(module, **values):
    saved = {k: getattr(module, k) for k in values}
    for k, v in values.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(module, k, v)


def _head() -> str:
    res = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False)
    if res.returncode != 0:
        _fail("FIXTURE_GIT_UNAVAILABLE",
              f"git rev-parse HEAD failed at {REPO}: {res.stderr.strip()}")
    return res.stdout.strip()


def _campaign_root(tmp: Path) -> Path:
    root = tmp / "V04DLT_fixture"
    root.mkdir()
    return root


def _core(ws: dict, root: Path, arm: int, *, n_epochs: int,
          steps_per_epoch: int, n_val: int) -> int:
    with _p4_pins(ws), _patched(v04t, N_VAL_BATCHES=n_val):
        return dlt._run_core(ws["data_root"], ws["p4"], ws["probe"],
                             ws["gate"], _head(), str(root), arm, SEED,
                             n_epochs=n_epochs,
                             steps_per_epoch=steps_per_epoch,
                             total_steps=n_epochs * steps_per_epoch)


def _expect(code: str, name: str, fn, exc_type=v04t.V04TError) -> None:
    try:
        fn()
    except exc_type as exc:
        got = getattr(exc, "code", type(exc).__name__)
        if got != code:
            _fail("FIXTURE_WRONG_CODE",
                  f"case {name!r} raised {got}, expected {code}")
        logger.info("branch PASS: %s -> %s", name, code)
    else:
        _fail(code, f"case {name!r} was accepted; the guard is dead")


def _record(root: Path, arm: int, attempt: int = 1) -> dict:
    p = root / v04t.run_record_name(arm, SEED, attempt)
    side = p.with_suffix(p.suffix + ".sha256")
    if not p.is_file() or not side.is_file():
        _fail("FIXTURE_RECORD_MISSING", f"record or sidecar missing: {p}")
    if side.read_text() != f"{v04t._file_sha256(p)}  {p.name}\n":
        _fail("FIXTURE_SIDECAR_MISMATCH", f"sidecar mismatch for {p}")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# B2.0 equivalence: new loop vs V04T v0.3 _run, same small synthetic budget
# ---------------------------------------------------------------------------

def _equal(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return (a == b) or (np.isnan(a) and np.isnan(b))
    return a == b


def b20_equivalence() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="v04dlt_b20_"))
    try:
        ws = _workspace(tmp, B20_SLICES)
        ref_root = tmp / "v04t_reference"
        new_root = _campaign_root(tmp)
        head = _head()
        ledger_after_arm0 = None
        with _p4_pins(ws), _patched(
                v04t, N_EPOCHS=B20_EPOCHS,
                STEPS_PER_EPOCH=B20_STEPS_PER_EPOCH,
                TOTAL_STEPS=B20_EPOCHS * B20_STEPS_PER_EPOCH,
                N_VAL_BATCHES=B20_N_VAL):
            for arm in (0, 1):
                if v04t._run(ws["data_root"], ws["p4"], ws["probe"],
                             ws["gate"], head, str(ref_root), arm,
                             SEED) != 0:
                    _fail("B20_REFERENCE_FAILED", f"V04T arm {arm} != 0")
        for arm in (0, 1):
            if _core(ws, new_root, arm, n_epochs=B20_EPOCHS,
                     steps_per_epoch=B20_STEPS_PER_EPOCH,
                     n_val=B20_N_VAL) != 0:
                _fail("B20_NEW_FAILED", f"V04DLTT arm {arm} != 0")
            if arm == 0:
                ledger_after_arm0 = v04t.load_ledger(
                    new_root / "v04_run_ledger.json")
        for arm in (0, 1):
            ra, rb = _record(ref_root, arm), _record(new_root, arm)
            if len(ra["history"]) != len(rb["history"]) or \
                    len(ra["history"]) != B20_EPOCHS * B20_STEPS_PER_EPOCH:
                _fail("B20_HISTORY_MISMATCH",
                      f"arm {arm}: history lengths {len(ra['history'])}/"
                      f"{len(rb['history'])}")
            for ha, hb in zip(ra["history"], rb["history"]):
                for k in HIST_KEYS:
                    if not _equal(ha[k], hb[k]):
                        _fail("B20_HISTORY_MISMATCH",
                              f"arm {arm} step {ha['step']} {k}: "
                              f"{ha[k]!r} != {hb[k]!r}")
            if ra["validation_history"] != rb["validation_history"]:
                _fail("B20_VALIDATION_MISMATCH", f"arm {arm}")
            for k in RUN_KEYS:
                if not _equal(ra[k], rb[k]):
                    _fail("B20_RUN_FIELD_MISMATCH",
                          f"arm {arm} {k}: {ra[k]!r} != {rb[k]!r}")
            if ra["post_training_conditioning_diagnostic"] != \
                    rb["post_training_conditioning_diagnostic"]:
                _fail("B20_DIAGNOSTIC_MISMATCH", f"arm {arm}")
            if len(ra["checkpoints"]) != B20_EPOCHS:
                _fail("B20_CHECKPOINT_MISMATCH", f"arm {arm} count")
            for ca, cb in zip(ra["checkpoints"], rb["checkpoints"]):
                sa = torch.load(ref_root / ca["file"], weights_only=True)
                sb = torch.load(new_root / cb["file"], weights_only=True)
                if set(sa) != set(sb):
                    _fail("B20_CHECKPOINT_MISMATCH",
                          f"arm {arm} epoch {ca['epoch']}: key sets differ")
                for key in sa:
                    if not torch.equal(sa[key], sb[key]):
                        _fail("B20_CHECKPOINT_MISMATCH",
                              f"arm {arm} epoch {ca['epoch']} tensor {key}")
        la = v04t.load_ledger(ref_root / "v04_run_ledger.json")["entries"]
        lb = v04t.load_ledger(new_root / "v04_run_ledger.json")["entries"]
        if [[e[k] for k in LEDGER_KEYS] for e in la] != \
                [[e[k] for k in LEDGER_KEYS] for e in lb]:
            _fail("B20_LEDGER_MISMATCH", "ledger status/attempt/steps differ")
        if lb[0] != ledger_after_arm0["entries"][0]:
            _fail("B23_LEDGER_NOT_APPEND_ONLY",
                  "earlier ledger entry changed after a later append")
        rb1 = _record(new_root, 1)
        rb0 = _record(new_root, 0)
        if rb1["lambda"] != ws["lam"] or rb0["lambda"] != 0.0:
            _fail("B22_LAMBDA_MISMATCH",
                  f"record lambdas arm0={rb0['lambda']!r} "
                  f"arm1={rb1['lambda']!r}, probe {ws['lam']!r}")
        if rb0["config"]["epochs"] != B20_EPOCHS or \
                rb0["script"] != f"{dlt.__abbr__} v{dlt.__version__}":
            _fail("B20_PROVENANCE_MISMATCH", "new record script/config")
        logger.info("B2.0 PASS: histories, validation, checkpoint tensors, "
                    "run fields, diagnostics and ledger transitions equal "
                    "V04T v0.3 (%d steps x 2 arms); B2.2 lambdas exact; "
                    "B2.3 ledger append-only", len(rb0["history"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=False)


# ---------------------------------------------------------------------------
# B2.1 budget / B2.2 lambda / B2.3 path guard
# ---------------------------------------------------------------------------

def b21_budget() -> None:
    dlt.assert_locked_budget(30, 1086, 32580)
    for bad in ((3, 1086, 3258), (30, 1086, 32581), (31, 1086, 33666),
                (30, 1085, 32550)):
        _expect("BUDGET_MISMATCH", f"budget {bad}",
                lambda b=bad: dlt.assert_locked_budget(*b))
    if (dlt.N_EPOCHS_DLT, dlt.STEPS_PER_EPOCH, dlt.TOTAL_STEPS_DLT) != \
            (30, 1086, 32580):
        _fail("BUDGET_MISMATCH", "module budget constants drifted")
    for opt in ("--epochs", "--steps", "--budget", "--total-steps"):
        res = subprocess.run(
            [sys.executable, "-m", "seqref_mri.scripts.v04dlt_train",
             "--mode", "run", opt, "3"], cwd=REPO, capture_output=True,
            text=True, check=False)
        if res.returncode != 2 or "unrecognized arguments" not in res.stderr:
            _fail("BUDGET_OPTION_EXPOSED",
                  f"production CLI accepted {opt} (rc {res.returncode})")
    logger.info("B2.1 PASS: locked 30 x 1086 = 32,580; other budgets and "
                "budget options refused")


def b22_lambda() -> None:
    rec = v04f._good_probe_record()
    if dlt.lambda_for_arm(rec, 0) != 0.0:
        _fail("B22_LAMBDA_MISMATCH", "arm 0 lambda != 0.0")
    if dlt.lambda_for_arm(rec, 1) != float(rec["lambda_prime"]):
        _fail("B22_LAMBDA_MISMATCH", "arm 1 lambda != probe lambda_prime")
    logger.info("B2.2 PASS: arm 0 exactly 0, arm 1 bit-equal to the probe "
                "record (record values checked in B2.0)")


def b23_path_guard() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="v04dlt_b23_"))
    try:
        _expect("CAMPAIGN_ROOT_MISSING", "missing root",
                lambda: dlt.check_campaign_root(str(tmp / "V04DLT_absent")))
        (tmp / "results_other").mkdir()
        _expect("CAMPAIGN_ROOT_INVALID", "wrong name",
                lambda: dlt.check_campaign_root(str(tmp / "results_other")))
        bad = tmp / "V04TR_10cedfde" / "V04DLT_x"
        bad.mkdir(parents=True)
        _expect("V04_EVIDENCE_PATH_REFUSED", "under V04 evidence",
                lambda: dlt.check_campaign_root(str(bad)))
        root = _campaign_root(tmp)
        f = root / "v04_arm0_seed101_attempt1_epoch1.pt"
        f.write_bytes(b"x")
        _expect("ARTIFACT_EXISTS", "existing artifact",
                lambda: dlt.refuse_existing(f))
        g = root / "x.json"
        g.with_suffix(".json.sha256").write_text("y\n")
        _expect("ARTIFACT_EXISTS", "existing sidecar",
                lambda: dlt.refuse_existing(g))
        _expect("ORPHANED_ATTEMPT_ARTIFACTS", "orphan before start",
                lambda: dlt.refuse_attempt_artifacts(root, 0, SEED, 1))
        ws = _workspace(tmp / "ws", SMALL_SLICES)
        _expect("ORPHANED_ATTEMPT_ARTIFACTS", "trainer refuses orphan",
                lambda: _core(ws, root, 0, n_epochs=1,
                              steps_per_epoch=SMALL_STEPS, n_val=1))
        if f.read_bytes() != b"x":
            _fail("B23_OVERWRITE", "orphan file was modified")
        logger.info("B2.3 PASS: missing/invalid/V04-evidence roots, existing "
                    "artifacts and orphans refused; nothing overwritten")
    finally:
        shutil.rmtree(tmp, ignore_errors=False)


# ---------------------------------------------------------------------------
# B2.4 two-step smoke through the FIXTURE-ONLY entry point
# ---------------------------------------------------------------------------

def fixture_entry(ws: dict, root: Path, scratch: Path, *, n_epochs: int,
                  steps_per_epoch: int, n_val: int) -> int:
    """Fixture-only entry (B2.4): refuses any real data root, any h5 source
    and any output root outside its scratch directory."""
    scratch = scratch.resolve()
    data = Path(ws["data_root"]).resolve()
    if data == REAL_DATA_ROOT or REAL_DATA_ROOT in data.parents:
        _fail("FIXTURE_REAL_DATA_REFUSED", f"real data root {data}")
    if scratch not in data.parents:
        _fail("FIXTURE_DATA_OUTSIDE_SCRATCH", f"data root {data}")
    for h5 in data.rglob("*.h5"):
        if scratch not in h5.resolve().parents:
            _fail("FIXTURE_DATA_OUTSIDE_SCRATCH", f"h5 source {h5}")
    if scratch not in root.resolve().parents:
        _fail("FIXTURE_OUTPUT_OUTSIDE_SCRATCH", f"output root {root}")
    return _core(ws, root, 0, n_epochs=n_epochs,
                 steps_per_epoch=steps_per_epoch, n_val=n_val)


def b24_two_step_smoke() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="v04dlt_b24_"))
    try:
        ws = _workspace(tmp, SMALL_SLICES)
        root = _campaign_root(tmp)
        bad_ws = dict(ws, data_root=str(REAL_DATA_ROOT))
        _expect("FIXTURE_REAL_DATA_REFUSED", "real data root",
                lambda: fixture_entry(bad_ws, root, tmp, n_epochs=1,
                                      steps_per_epoch=SMALL_STEPS, n_val=1),
                FixtureFailure)
        outside = Path(tempfile.mkdtemp(prefix="V04DLT_outside_"))
        try:
            _expect("FIXTURE_OUTPUT_OUTSIDE_SCRATCH", "output outside",
                    lambda: fixture_entry(ws, outside, tmp, n_epochs=1,
                                          steps_per_epoch=SMALL_STEPS,
                                          n_val=1), FixtureFailure)
        finally:
            shutil.rmtree(outside, ignore_errors=False)
        if fixture_entry(ws, root, tmp, n_epochs=1,
                         steps_per_epoch=SMALL_STEPS, n_val=1) != 0:
            _fail("B24_SMOKE_FAILED", "two-step smoke returned non-zero")
        rec = _record(root, 0)
        led = v04t.load_ledger(root / "v04_run_ledger.json")
        if rec["steps_completed"] != SMALL_STEPS or \
                [e["status"] for e in led["entries"]] != ["completed"]:
            _fail("B24_SMOKE_FAILED", "steps/ledger status unexpected")
        sel = v04t.select_checkpoint(rec["validation_history"],
                                     rec["checkpoints"])
        if sel["epoch"] != 1:
            _fail("B24_SMOKE_FAILED", f"selection epoch {sel['epoch']}")
        logger.info("B2.4 PASS: two-step smoke via fixture-only entry; "
                    "record/ledger/sidecars written; selection applied; "
                    "real data and outside roots refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=False)


# ---------------------------------------------------------------------------
# B2.5 mask contract, all 30 epochs
# ---------------------------------------------------------------------------

def b25_mask_contract() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="v04dlt_b25_"))
    try:
        ws = _workspace(tmp, B20_SLICES)
        ds = FastMRISliceDataset(ws["data_root"], split="train",
                                 mode="train")
        n_center, n_total = mask_counts(96)
        if (n_center, n_total) != (8, 24):
            _fail("B25_MASK_COUNTS", f"mask_counts(96)={n_center, n_total}")
        centre = slice((96 - n_center) // 2, (96 - n_center) // 2 + n_center)
        for k in range(1, dlt.N_EPOCHS_DLT + 1):      # checkpoint epoch k
            ds.set_epoch(k - 1)                        # mask index k - 1
            for i in MASK_SAMPLE:
                path, sl = ds.index[i]
                rel = Path(path).relative_to(ds.data_root).as_posix()
                want = make_cartesian_mask(96, canonical_mask_seed(
                    TRAIN_BASE_SEED, rel, int(sl), k - 1))
                got1 = ds[i]["mask"].numpy().astype(bool)
                got2 = ds[i]["mask"].numpy().astype(bool)
                if not np.array_equal(got1, want):
                    _fail("B25_MASK_NOT_CANONICAL", f"epoch {k} slice {i}")
                if not np.array_equal(got1, got2):
                    _fail("B25_MASK_NOT_DETERMINISTIC",
                          f"epoch {k} slice {i}")
                if int(got1.sum()) != n_total or not got1[centre].all():
                    _fail("B25_MASK_COUNTS", f"epoch {k} slice {i}")
        logger.info("B2.5 PASS: epochs 1-30 canonical (index k-1), exact "
                    "counts 24/8, deterministic; epochs 1-3 = V04 indices "
                    "0-2")
    finally:
        shutil.rmtree(tmp, ignore_errors=False)


# ---------------------------------------------------------------------------
# B2.6 failure paths / B2.7 mode + partial batch
# ---------------------------------------------------------------------------

def _injected(after: int, action):
    """Wrap v04o.density_only: call through for the first `after` calls,
    then apply `action(out)` (fixture-scoped injection)."""
    orig = v04o.density_only
    calls = {"n": 0}

    def wrapped(*a, **k):
        calls["n"] += 1
        out = orig(*a, **k)
        if calls["n"] > after:
            return action(out)
        return out
    return orig, wrapped


def _fail_case(name: str, action, expect_code: str, expect_status: str,
               after: int = 0, exc_type=v04t.V04TError) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="v04dlt_b26_"))
    try:
        ws = _workspace(tmp, SMALL_SLICES)
        root = _campaign_root(tmp)
        orig, wrapped = _injected(after, action)
        out = io.StringIO()
        v04o.density_only = wrapped
        try:
            with contextlib.redirect_stdout(out):
                _expect(expect_code, name,
                        lambda: _core(ws, root, 0, n_epochs=1,
                                      steps_per_epoch=SMALL_STEPS, n_val=1),
                        exc_type)
        finally:
            v04o.density_only = orig
        if "PASS" in out.getvalue():
            _fail("B26_PASS_LINE", f"{name}: PASS line printed on failure")
        rec = _record(root, 0)
        led = v04t.load_ledger(root / "v04_run_ledger.json")["entries"]
        if rec["status"] != expect_status or led[-1]["status"] != \
                expect_status:
            _fail("B26_STATUS_MISMATCH",
                  f"{name}: status {rec['status']}/{led[-1]['status']} != "
                  f"{expect_status}")
        terminal = expect_status in ("numerical_failure", "failed_validity")
        led_full = v04t.load_ledger(root / "v04_run_ledger.json")
        if terminal and expect_status == "numerical_failure":
            _expect("RERUN_POLICY_VIOLATION", f"{name}: no restart",
                    lambda: v04t.next_attempt(led_full, 0, SEED))
        if expect_status == "interrupted" and \
                v04t.next_attempt(led_full, 0, SEED) != 2:
            _fail("B26_RETRY_POLICY", f"{name}: retry not permitted")
    finally:
        shutil.rmtree(tmp, ignore_errors=False)


def _raise(exc):
    def act(_out):
        raise exc
    return act


def b26_failure_paths() -> None:
    _fail_case("non-finite loss", lambda out: out * float("nan"),
               "NUMERICAL_FAILURE", "numerical_failure")
    _fail_case("objective error",
               _raise(v04o.V04LError("V04L_INTERFACE_MISMATCH", "injected")),
               "V04L_INTERFACE_MISMATCH", "failed_validity")
    _fail_case("injected exception", _raise(RuntimeError("injected")),
               "INFRASTRUCTURE_INTERRUPTION", "interrupted")
    _fail_case("KeyboardInterrupt in training", _raise(KeyboardInterrupt()),
               "KeyboardInterrupt", "interrupted", after=1,
               exc_type=KeyboardInterrupt)
    # KeyboardInterrupt during the non-gating diagnostic: training status
    # stays completed, evidence written, interrupt re-raised.
    tmp = Path(tempfile.mkdtemp(prefix="v04dlt_b26d_"))
    try:
        ws = _workspace(tmp, SMALL_SLICES)
        root = _campaign_root(tmp)
        orig = dlt.post_training_conditioning_diagnostic

        def ki(*a, **k):
            raise KeyboardInterrupt
        dlt.post_training_conditioning_diagnostic = ki
        try:
            _expect("KeyboardInterrupt", "KeyboardInterrupt in diagnostic",
                    lambda: _core(ws, root, 0, n_epochs=1,
                                  steps_per_epoch=SMALL_STEPS, n_val=1),
                    KeyboardInterrupt)
        finally:
            dlt.post_training_conditioning_diagnostic = orig
        rec = _record(root, 0)
        if rec["status"] != "completed" or \
                rec["post_training_conditioning_diagnostic"] != \
                {"error": "KeyboardInterrupt", "gating": False}:
            _fail("B26_STATUS_MISMATCH", "diagnostic interrupt record")
        # fourth attempt refused; out-of-order refused
        root2 = tmp / "V04DLT_attempts"
        root2.mkdir()
        lp = root2 / "v04_run_ledger.json"
        for a in (1, 2, 3):
            v04t.append_entry(lp, v04f._led_entry(
                0, SEED, a, "interrupted", error="INFRASTRUCTURE_INTERRUPTION"))
        _expect("RERUN_POLICY_VIOLATION", "fourth attempt",
                lambda: _core(ws, root2, 0, n_epochs=1,
                              steps_per_epoch=SMALL_STEPS, n_val=1))
        root3 = tmp / "V04DLT_order"
        root3.mkdir()
        _expect("ORDER_VIOLATION", "out-of-order pair",
                lambda: _core(ws, root3, 1, n_epochs=1,
                              steps_per_epoch=SMALL_STEPS, n_val=1))
    finally:
        shutil.rmtree(tmp, ignore_errors=False)
    logger.info("B2.6 PASS: numerical_failure terminal, failed_validity, "
                "interrupted + retry, KeyboardInterrupt in training and in "
                "diagnostic (evidence written, no PASS line), fourth attempt "
                "and out-of-order refused")


def b27_mode_partial_batch() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="v04dlt_b27_"))
    try:
        ws = _workspace(tmp, SMALL_SLICES)
        root = _campaign_root(tmp)
        seen = []
        orig_val = dlt._validate
        orig_diag = dlt.post_training_conditioning_diagnostic

        def val(model, *a, **k):
            seen.append(("val_in", model.training))
            out = orig_val(model, *a, **k)
            seen.append(("val_out", model.training))
            return out

        def diag(model, *a, **k):
            seen.append(("diag_in", model.training))
            out = orig_diag(model, *a, **k)
            seen.append(("diag_out", model.training))
            return out
        dlt._validate, dlt.post_training_conditioning_diagnostic = val, diag
        try:
            if _core(ws, root, 0, n_epochs=1, steps_per_epoch=SMALL_STEPS,
                     n_val=1) != 0:
                _fail("B27_RUN_FAILED", "mode run returned non-zero")
        finally:
            dlt._validate, dlt.post_training_conditioning_diagnostic = \
                orig_val, orig_diag
        if seen != [("val_in", True), ("val_out", True),
                    ("diag_in", True), ("diag_out", True)]:
            _fail("B27_MODE_NOT_RESTORED", f"mode trace {seen}")
        rec = _record(root, 0)
        if rec["steps_completed"] != -(-SMALL_SLICES // v02m.BATCH_SIZE):
            _fail("B27_PARTIAL_BATCH_DROPPED",
                  f"{rec['steps_completed']} steps for {SMALL_SLICES} slices")
        logger.info("B2.7 PASS: train mode restored after validation and "
                    "diagnostic; final partial batch counted (%d slices -> "
                    "%d steps)", SMALL_SLICES, rec["steps_completed"])
    finally:
        shutil.rmtree(tmp, ignore_errors=False)


def b28_regression() -> None:
    res = subprocess.run([sys.executable, "-m",
                          "seqref_mri.scripts.v04_fixtures"], cwd=REPO,
                         capture_output=True, text=True, check=False)
    if res.returncode != 0 or "f01-f17: ALL PASS" not in res.stderr:
        tail = (res.stderr or "")[-2000:]
        _fail("B28_REGRESSION_FAILED",
              f"v04_fixtures rc {res.returncode}; tail: {tail}")
    logger.info("B2.8 PASS: registered suite f01-f17 ALL PASS")


# ---------------------------------------------------------------------------
# B2.9 launcher fixture: a COPY of the production launcher whose trainer
# line is the only change, driving the fixture-only stub in scratch.
# ---------------------------------------------------------------------------

LAUNCHER = REPO / "seqref_mri" / "scripts" / "v04dlt_launch.sh"
STUB = REPO / "seqref_mri" / "scripts" / "v04dlt_stub_train.py"
TRAINER_LINE = 'TRAINER_CMD=("$PY" -m seqref_mri.scripts.v04dlt_train)'
B29_TIMEOUT = 300


def _launcher_copy(scratch: Path, trainer: str = None, py: str = None) -> Path:
    """Copy of the production launcher; only the trainer line (always) and
    the PY constant (boundary cases) may differ -- asserted."""
    src = LAUNCHER.read_text().splitlines(keepends=True)
    hits = [i for i, ln in enumerate(src) if ln.rstrip("\n") == TRAINER_LINE]
    if len(hits) != 1:
        _fail("B29_TRAINER_LINE", f"trainer line found {len(hits)} times")
    dst = list(src)
    dst[hits[0]] = (f'TRAINER_CMD=({trainer})\n' if trainer
                    else f'TRAINER_CMD=("$PY" "{STUB}")\n')
    allowed = list(hits)
    if py:
        ph = [i for i, ln in enumerate(src) if ln.startswith("readonly PY=")]
        if len(ph) != 1:
            _fail("B29_PY_LINE", f"PY constant found {len(ph)} times")
        dst[ph[0]] = f'readonly PY="{py}"\n'
        allowed = sorted(allowed + ph)
    diff = [i for i, (x, y) in enumerate(zip(src, dst)) if x != y]
    if len(src) != len(dst) or diff != allowed:
        _fail("B29_COPY_DIFFERS", f"copy differs on lines {diff}")
    out = scratch / "v04dlt_launch_copy.sh"
    out.write_text("".join(dst))
    out.chmod(0o755)
    return out


class _Campaign:
    def __init__(self, scratch: Path, name: str):
        self.scratch = scratch
        self.results = scratch / f"results_{name}"
        self.results.mkdir()
        self.pin = _head()
        self.root = self.results / f"V04DLT_{self.pin[:8]}"
        self.data = scratch / "data"
        self.data.mkdir(exist_ok=True)

    def args(self, mode: str) -> list:
        return [str(self.scratch / "v04dlt_launch_copy.sh"), "--mode", mode,
                "--commit-pin", self.pin, "--repo", str(REPO),
                "--data-root", str(self.data),
                "--results-root", str(self.results)]


def _env(tmux: bool = True) -> dict:
    import os
    env = dict(os.environ)
    env.pop("TMUX", None)
    if tmux:
        env["TMUX"] = "/tmp/v04dlt-fixture-tmux,0,0"
    return env


def _run_launcher(c: _Campaign, mode: str, *, tmux: bool = True):
    return subprocess.run(c.args(mode), env=_env(tmux), capture_output=True,
                          text=True, timeout=B29_TIMEOUT, check=False)


def _start_launcher(c: _Campaign, mode: str):
    return subprocess.Popen(c.args(mode), env=_env(True),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)


def _plan(scratch: Path, plan: dict) -> None:
    (scratch / "stub_plan.json").write_text(json.dumps(plan))


def _tree(root: Path, skip=(".launch.lock",)) -> dict:
    return {p.name: v04t._file_sha256(p) for p in sorted(root.iterdir())
            if p.is_file() and p.name not in skip}


def _wait_for(path: Path, what: str, timeout: float = 60.0) -> None:
    import time
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > timeout:
            _fail("B29_TIMEOUT", f"waiting for {what}")
        time.sleep(0.05)


def _expect_rc(res, rc: int, code: str, name: str) -> None:
    text = (res.stdout or "") + (res.stderr or "")
    if res.returncode != rc or (code and code not in text):
        _fail("B29_UNEXPECTED_OUTCOME",
              f"{name}: rc {res.returncode} (want {rc}), want code {code}; "
              f"tail: {text[-1500:]}")
    logger.info("B2.9 branch PASS: %s -> rc %d %s", name, rc, code)


def _sessions(root: Path) -> list:
    return sorted(p.name[len("launcher_session_"):-len(".log")]
                  for p in root.glob("launcher_session_*.log"))


def _check_sha_files(root: Path) -> None:
    for side in sorted(root.glob("*.sha256")):
        res = subprocess.run(["sha256sum", "-c", "--quiet", side.name],
                             cwd=root, capture_output=True, text=True,
                             check=False)
        if res.returncode != 0:
            _fail("B29_SHA_CHECK_FAILED", f"{side.name}: {res.stdout}")


def _ledger_prefix_ok(root: Path) -> None:
    live = v04t.load_ledger(root / "v04_run_ledger.json")
    meta = {k: v for k, v in live.items() if k != "entries"}
    for snap in sorted(root.glob("ledger_snapshot_*.json")):
        s = json.loads(snap.read_text())
        side = (root / (snap.name + ".sha256")).read_text().split()
        if side[1] != snap.name:
            _fail("B29_SNAPSHOT_SIDECAR", f"{snap.name} sidecar names {side}")
        n = len(s["entries"])
        if s["entries"] != live["entries"][:n] or \
                {k: v for k, v in s.items() if k != "entries"} != meta:
            _fail("B29_PREFIX", f"{snap.name} not a prefix")


def _kill_inhibitor(root: Path, session: str) -> None:
    import os
    import re
    import signal as sg
    log = (root / f"launcher_session_{session}.log").read_text()
    m = re.search(r"inhibitor held \(pid (\d+)\)", log)
    if m:
        pid = int(m.group(1))
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            logger.info("inhibitor pid %s already gone", pid)
            return
        if b"sleep infinity" not in cmd:     # never kill a recycled pid
            logger.info("pid %s is not an inhibitor (%r); left alone", pid, cmd[:80])
            return
        try:
            os.kill(pid, sg.SIGKILL)
        except ProcessLookupError:
            logger.info("inhibitor pid %s already gone", pid)


def _kill_all_inhibitors(scratch: Path) -> None:
    """Hygiene: a failed or timed-out case can leave a launcher's inhibitor
    (systemd-inhibit on the host) running; kill every recorded one."""
    for log in scratch.rglob("launcher_session_*.log"):
        _kill_inhibitor(log.parent, log.name[len("launcher_session_"):-4])


def _lock_holders(lock: Path) -> list:
    import os
    found = []
    for d in Path("/proc").glob("[0-9]*/fd"):
        try:
            for fd in d.iterdir():
                if os.readlink(fd) == str(lock):
                    found.append((d.parent.name, (d.parent / "cmdline")
                                  .read_text().replace("\0", " ")[:160]))
        except OSError as exc:
            logger.debug("proc scan skipped %s: %s", d, exc)
    return found


def _wait_lock_free(lock: Path, timeout: float = 30.0) -> None:
    import time
    t0 = time.time()
    while subprocess.run(["flock", "-n", str(lock), "true"],
                         check=False).returncode != 0:
        if time.time() - t0 > timeout:
            _fail("B29_LOCK_NOT_RELEASED",
                  f"lock still held by {_lock_holders(lock)}")
        time.sleep(0.1)


def b29_launcher() -> None:
    import os
    import signal as sg
    import time
    scratch = Path(tempfile.mkdtemp(prefix="v04dlt_b29_"))
    try:
        (scratch / "V04DLT_STUB_SCRATCH").write_text("fixture scratch\n")
        _launcher_copy(scratch)
        _plan(scratch, {"default": ["succeed", 0]})
        # -- modes and tmux ---------------------------------------------------
        a = _Campaign(scratch, "a")
        _expect_rc(_run_launcher(a, "new", tmux=False), 2, "NOT_IN_TMUX",
                   "outside tmux")
        if a.root.exists():
            _fail("B29_ROOT_CREATED", "root created despite NOT_IN_TMUX")
        _expect_rc(_run_launcher(a, "resume"), 2, "CAMPAIGN_ROOT_MISSING",
                   "resume without root")
        # -- session 1: pair 1 then STOP (current run finishes) ----------------
        _plan(scratch, {"0_101_1": ["succeed", 3], "default": ["succeed", 0]})
        p = _start_launcher(a, "new")
        _wait_for(scratch / "stub_started_0_101_1.txt", "session-1 stub")
        note = b"pause for daytime use\n"
        (a.root / "STOP").write_bytes(note)
        o, e = p.communicate(timeout=B29_TIMEOUT)
        if p.returncode != 0 or "STOP_REQUESTED" not in e:
            _fail("B29_STOP", f"session 1 rc {p.returncode}: {e[-1500:]}")
        s1 = _sessions(a.root)
        disp = (scratch / "stub_sigdisp_0_101_1.txt").read_text().strip()
        if disp not in ("SIG_IGN", "SIG_DFL", "default_int_handler"):
            _fail("B29_PRECONDITION", f"stub SIGINT disposition {disp!r}")
        logger.info("B2.9: stub spawned via the launcher inherited SIGINT %s",
                    disp)
        led = v04t.load_ledger(a.root / "v04_run_ledger.json")["entries"]
        if [(x["arm"], x["seed"], x["status"]) for x in led] != \
                [(0, 101, "completed")]:
            _fail("B29_STOP", f"session 1 ledger {led}")
        if (a.root / f"stop_request_{s1[0]}.txt").read_bytes() != note or \
                "stop_sha256=" not in (a.root /
                                       f"stop_seen_{s1[0]}.txt").read_text():
            _fail("B29_STOP", "STOP bytes/metadata not preserved separately")
        man1 = (a.root / f"campaign_manifest_{s1[0]}.sha256").read_text()
        if f"stop_request_{s1[0]}.txt" not in man1 or \
                "v04_run_ledger.json" in man1:
            _fail("B29_MANIFEST", "session 1 manifest content")
        _check_sha_files(a.root)
        _ledger_prefix_ok(a.root)
        # -- resume refused while STOP present; tree unchanged ------------------
        before = _tree(a.root)
        _expect_rc(_run_launcher(a, "resume"), 2, "STOP_PRESENT",
                   "resume with STOP")
        if _tree(a.root) != before:
            _fail("B29_VERIFY_WROTE", "failed verification changed files")
        # -- ack-stop: edited STOP refused; exact STOP acknowledged -------------
        (a.root / "STOP").write_bytes(b"edited\n")
        _expect_rc(_run_launcher(a, "ack-stop"), 2, "STOP_CHANGED",
                   "ack-stop edited STOP")
        if not (a.root / "STOP").exists():
            _fail("B29_ACK", "edited STOP was renamed")
        (a.root / "STOP").write_bytes(note)
        _expect_rc(_run_launcher(a, "ack-stop"), 0, "ACK_STOP_PASS",
                   "ack-stop")
        if (a.root / "STOP").exists() or \
                not list(a.root.glob("stop_acknowledged_*.txt")):
            _fail("B29_ACK", "STOP not renamed on acknowledgement")
        # -- session 2: resume, pair 2, STOP; earlier files untouched -----------
        time.sleep(1.1)
        before = _tree(a.root)
        _plan(scratch, {"1_101_1": ["succeed", 3], "default": ["succeed", 0]})
        p = _start_launcher(a, "resume")
        _wait_for(scratch / "stub_started_1_101_1.txt", "session-2 stub")
        (a.root / "STOP").write_bytes(note)
        o, e = p.communicate(timeout=B29_TIMEOUT)
        if p.returncode != 0 or "STOP_REQUESTED" not in e:
            _fail("B29_SESSION2", f"rc {p.returncode}: {e[-1500:]}")
        after = _tree(a.root)
        for name, h in before.items():
            if name in ("v04_run_ledger.json", "v04_run_ledger.json.sha256"):
                continue
            if after.get(name) != h:
                _fail("B29_RESUME_MODIFIED", f"{name} changed or removed")
        s12 = _sessions(a.root)
        if len(s12) != 2:
            _fail("B29_SESSION2", f"sessions {s12}")
        _check_sha_files(a.root)          # BOTH session manifests verify
        _ledger_prefix_ok(a.root)
        _expect_rc(_run_launcher(a, "ack-stop"), 0, "ACK_STOP_PASS",
                   "ack-stop 2")
        # -- tampered earlier entry / metadata detected (scratch copies) --------
        for tag, mutate in (("entry", lambda L: L["entries"][0].update(
                                {"steps_completed": 999})),
                            ("meta", lambda L: L.update({"extra": 1}))):
            t = _Campaign(scratch, f"tamper_{tag}")
            shutil.copytree(a.root, t.root)
            lp = t.root / "v04_run_ledger.json"
            L = json.loads(lp.read_text())
            mutate(L)
            v04t._write_with_sidecar(lp, v02m.canonical_json(L))
            _expect_rc(_run_launcher(t, "resume"), 2,
                       "LEDGER_PREFIX_MISMATCH", f"tampered {tag}")
        # -- orphan detection (scratch copy): nothing deleted ---------------------
        o_c = _Campaign(scratch, "orphan")
        shutil.copytree(a.root, o_c.root)
        orphan = o_c.root / "v04_arm0_seed202_attempt1_epoch1.pt"
        orphan.write_bytes(b"orphan")
        before = _tree(o_c.root)
        _expect_rc(_run_launcher(o_c, "resume"), 2,
                   "ORPHANED_ATTEMPT_ARTIFACTS", "orphan")
        if _tree(o_c.root) != before:
            _fail("B29_VERIFY_WROTE", "orphan verification changed files")
        # -- cancellation by SIGTERM; lock held meanwhile -------------------------
        time.sleep(1.1)
        _plan(scratch, {"0_202_1": ["sleep"], "default": ["succeed", 0]})
        p = _start_launcher(a, "resume")
        _wait_for(scratch / "stub_started_0_202_1.txt", "cancel stub")
        held = _tree(a.root)
        _expect_rc(_run_launcher(a, "resume"), 2, "LOCK_HELD",
                   "second resume")
        _expect_rc(_run_launcher(a, "ack-stop"), 2, "LOCK_HELD",
                   "ack-stop while held")
        _expect_rc(_run_launcher(a, "new"), 2, "LOCK_HELD", "second new")
        if _tree(a.root) != held:
            _fail("B29_LOCK_WROTE", "refused instances changed files")
        os.kill(p.pid, sg.SIGTERM)
        o, e = p.communicate(timeout=B29_TIMEOUT)
        if p.returncode != 2 or "CANCELLED" not in e:
            _fail("B29_CANCEL", f"SIGTERM rc {p.returncode}: {e[-1500:]}")
        if (scratch / "stub_sigint_0_202_1.txt").read_text().strip() != "1" \
                or (scratch / "stub_finally_0_202_1.txt").read_text() \
                .strip() != "interrupted":
            _fail("B29_CANCEL", "stub did not receive exactly one SIGINT")
        led = v04t.load_ledger(a.root / "v04_run_ledger.json")["entries"]
        if (led[-1]["status"], led[-1]["error_code"]) != \
                ("interrupted", "KEYBOARD_INTERRUPT"):
            _fail("B29_CANCEL", f"ledger {led[-1]}")
        _check_sha_files(a.root)
        # -- cancellation by SIGINT to the launcher's process group ---------------
        g = _Campaign(scratch, "group")
        _plan(scratch, {"0_101_1": ["sleep"], "default": ["succeed", 0]})
        for f in scratch.glob("stub_*_0_101_1.txt"):
            f.rename(f.with_suffix(".txt.prev"))
        p = _start_launcher(g, "new")
        _wait_for(scratch / "stub_started_0_101_1.txt", "group stub")
        os.killpg(os.getpgid(p.pid), sg.SIGINT)
        o, e = p.communicate(timeout=B29_TIMEOUT)
        if p.returncode != 2 or "CANCELLED" not in e or \
                (scratch / "stub_sigint_0_101_1.txt").read_text().strip() \
                != "1":
            _fail("B29_CANCEL_GROUP", f"rc {p.returncode}: {e[-1500:]}")
        # -- trainer exits without the expected ledger entry ----------------------
        n = _Campaign(scratch, "noentry")
        for f in scratch.glob("stub_*_0_101_1.txt"):
            f.unlink()
        _plan(scratch, {"0_101_1": ["no_entry_on_sigint"],
                        "default": ["succeed", 0]})
        p = _start_launcher(n, "new")
        _wait_for(scratch / "stub_started_0_101_1.txt", "no-entry stub")
        os.kill(p.pid, sg.SIGTERM)
        o, e = p.communicate(timeout=B29_TIMEOUT)
        if p.returncode != 2 or "CANCEL_LEDGER_MISMATCH" not in e:
            _fail("B29_CANCEL_MISMATCH", f"rc {p.returncode}: {e[-1500:]}")
        # -- hard kill: no manifest -> next resume stops; nothing deleted ---------
        time.sleep(1.1)
        for f in scratch.glob("stub_*_0_202_2.txt"):
            f.unlink()
        _plan(scratch, {"0_202_2": ["sleep"], "default": ["succeed", 0]})
        p = _start_launcher(a, "resume")
        _wait_for(scratch / "stub_started_0_202_2.txt", "hard-kill stub")
        killed = _sessions(a.root)[-1]
        os.killpg(os.getpgid(p.pid), sg.SIGKILL)
        p.communicate(timeout=B29_TIMEOUT)
        stub_pid = None
        res = subprocess.run(["pgrep", "-f", str(STUB)], capture_output=True,
                             text=True, check=False)
        for pid in res.stdout.split():
            os.kill(int(pid), sg.SIGKILL)
            stub_pid = pid
        _kill_inhibitor(a.root, killed)
        t0 = time.time()
        while subprocess.run(["pgrep", "-f", str(STUB)], capture_output=True,
                             check=False).returncode == 0:
            if time.time() - t0 > 30:
                _fail("B29_TIMEOUT", "killed stub still running")
            time.sleep(0.05)
        _wait_lock_free(a.root / ".launch.lock")
        if (a.root / f"campaign_manifest_{killed}.sha256").exists():
            _fail("B29_HARD_KILL", "manifest written despite SIGKILL")
        before = _tree(a.root)
        _expect_rc(_run_launcher(a, "resume"), 2, "MISSING_SESSION_MANIFEST",
                   "resume after hard kill")
        if _tree(a.root) != before:
            _fail("B29_VERIFY_WROTE", "hard-kill verification changed files")
        logger.info("B2.9 PASS: one-line launcher copy; tmux/new/resume/"
                    "ack-stop modes; STOP finishes the current run and is "
                    "preserved by hash; resume verification writes nothing; "
                    "two-session manifests and snapshot prefixes verify; "
                    "tamper, orphan, hard kill (stub %s) detected; LOCK_HELD "
                    "x3; SIGTERM and group SIGINT forward exactly one SIGINT;"
                    " CANCEL_LEDGER_MISMATCH", stub_pid)
    finally:
        res = subprocess.run(["pgrep", "-f", str(STUB)], capture_output=True,
                             text=True, check=False)
        for pid in res.stdout.split():
            os.kill(int(pid), sg.SIGKILL)
        _kill_all_inhibitors(scratch)
        shutil.rmtree(scratch, ignore_errors=False)


# ---------------------------------------------------------------------------
# B2.9 (v0.2) signal cases: REAL trainer main() startup through the launcher
# spawn path; pending cancellation; inherited-ignore control; launch
# boundary (CANCEL / STOP during preflight starts no run).
# ---------------------------------------------------------------------------

SHIM_SRC = '''# fixture-written shim (V04DLTF B2.9): REAL v04dlt_train.main(), _run replaced
import json, os, signal, sys, time
from pathlib import Path
SCRATCH = Path(sys.argv[1]); ARGS = sys.argv[2:]
inh = signal.getsignal(signal.SIGINT)
plan = json.loads((SCRATCH / "shim_plan.json").read_text())
(SCRATCH / "shim_pid.txt").write_text(f"{os.getpid()}\\n")
(SCRATCH / "shim_inherited.txt").write_text(
    ("SIG_IGN" if inh is signal.SIG_IGN else "default_int_handler"
     if inh is signal.default_int_handler else repr(inh)) + "\\n")
if plan.get("force_ignore"):     # simulate the `python &` inherited-ignore spawn
    signal.signal(signal.SIGINT, signal.SIG_IGN)
(SCRATCH / "shim_ready.txt").write_text("1\\n")
from seqref_mri.scripts import v04dlt_train as t
if plan.get("disable_install"):
    t.install_sigint = lambda: "disabled-by-fixture"
def _blocking_run(*a, **k):
    ok = signal.getsignal(signal.SIGINT) is signal.default_int_handler
    (SCRATCH / "shim_in_run.txt").write_text(f"{ok}\\n")
    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        (SCRATCH / "shim_ki.txt").write_text("1\\n")
        raise
t._run = _blocking_run
time.sleep(float(plan.get("pre_main_delay", 0.0)))
sys.argv = ["v04dlt_train"] + ARGS
sys.exit(t.main())
'''

PYWRAP_SRC = '''#!/usr/bin/env bash
# fixture-written PY wrapper (V04DLTF B2.9): holds the launcher in preflight
for a in "$@"; do
  if [[ "$a" == preflight ]]; then
    : > "{scratch}/preflight_entered"
    while [[ ! -e "{scratch}/preflight_release" ]]; do sleep 0.05; done
    break
  fi
done
exec "{py}" "$@"
'''


def _case_scratch(base: Path, name: str) -> Path:
    s = base / name
    s.mkdir()
    (s / "V04DLT_STUB_SCRATCH").write_text("fixture scratch\n")
    _plan(s, {"default": ["succeed", 0]})
    return s


def _shim_case(base: Path, name: str, plan: dict) -> tuple:
    s = _case_scratch(base, name)
    (s / "real_shim.py").write_text(SHIM_SRC)
    (s / "shim_plan.json").write_text(json.dumps(plan))
    _launcher_copy(s, trainer=f'"$PY" "{s / "real_shim.py"}" "{s}"')
    return s, _Campaign(s, name)


def _forward_count(text: str) -> int:
    return text.count("forwarded one SIGINT to trainer pid")


def _kill_pidfile(path: Path) -> None:
    import os
    import signal as sg
    if path.is_file():
        try:
            os.kill(int(path.read_text().strip()), sg.SIGKILL)
        except ProcessLookupError:
            logger.info("pid in %s already gone", path.name)


def b29_signals() -> None:
    import os
    import signal as sg
    import time
    base = Path(tempfile.mkdtemp(prefix="v04dlt_b29s_"))
    try:
        # -- R1: real main() through the launcher spawn path -------------------
        s, c = _shim_case(base, "real", {"pre_main_delay": 0.0})
        p = _start_launcher(c, "new")
        _wait_for(s / "shim_in_run.txt", "real-main shim inside _run")
        os.kill(p.pid, sg.SIGTERM)
        o, e = p.communicate(timeout=B29_TIMEOUT)
        inh = (s / "shim_inherited.txt").read_text().strip()
        logger.info("B2.9: real trainer spawned via the launcher inherited "
                    "SIGINT %s", inh)
        if (s / "shim_in_run.txt").read_text().strip() != "True" or \
                not (s / "shim_ki.txt").is_file() or _forward_count(e) != 1:
            _fail("B29_REAL_SIGINT", f"real main did not receive exactly one "
                  f"SIGINT as KeyboardInterrupt; tail: {e[-1500:]}")
        _expect_rc(_Done(p.returncode, o, e), 2,
                   "CANCEL_LEDGER_MISMATCH", "real main SIGINT (stand-in _run "
                   "writes no ledger entry)")
        # -- R1 under the hazard: inherited SIG_IGN, fix restores SIGINT -----
        s, c = _shim_case(base, "hazard", {"pre_main_delay": 0.0,
                                           "force_ignore": True})
        p = _start_launcher(c, "new")
        _wait_for(s / "shim_in_run.txt", "hazard shim inside _run")
        os.kill(p.pid, sg.SIGTERM)
        o, e = p.communicate(timeout=B29_TIMEOUT)
        if (s / "shim_in_run.txt").read_text().strip() != "True" or \
                not (s / "shim_ki.txt").is_file() or _forward_count(e) != 1:
            _fail("B29_HAZARD_SIGINT", f"fix did not restore SIGINT under "
                  f"inherited SIG_IGN; tail: {e[-1500:]}")
        _expect_rc(_Done(p.returncode, o, e), 2, "CANCEL_LEDGER_MISMATCH",
                   "inherited SIG_IGN: install_sigint restores one SIGINT")
        # -- R1: cancellation before the handler exists is held pending -------
        s, c = _shim_case(base, "early", {"pre_main_delay": 3.0,
                                          "force_ignore": True})
        p = _start_launcher(c, "new")
        _wait_for(s / "shim_ready.txt", "early shim started (SIGINT ignored)")
        os.kill(p.pid, sg.SIGTERM)
        o, e = p.communicate(timeout=B29_TIMEOUT)
        if "cancellation pending" not in e or _forward_count(e) != 1 or \
                not (s / "shim_ki.txt").is_file():
            _fail("B29_EARLY_CANCEL", f"pending forward failed; tail: "
                  f"{e[-1500:]}")
        _expect_rc(_Done(p.returncode, o, e), 2, "CANCEL_LEDGER_MISMATCH",
                   "early cancel forwarded after handler install")
        # -- control: without install_sigint the SIGINT is ignored -------------
        s, c = _shim_case(base, "control", {"pre_main_delay": 0.0,
                                            "force_ignore": True,
                                            "disable_install": True})
        p = _start_launcher(c, "new")
        _wait_for(s / "shim_in_run.txt", "control shim inside _run")
        os.kill(int((s / "shim_pid.txt").read_text()), sg.SIGINT)
        time.sleep(1.5)
        if (s / "shim_ki.txt").exists() or \
                (s / "shim_in_run.txt").read_text().strip() != "False":
            _fail("B29_CONTROL", "SIGINT was not ignored without the fix; "
                  "the control does not demonstrate the hazard")
        _kill_pidfile(s / "shim_pid.txt")
        o, e = p.communicate(timeout=B29_TIMEOUT)
        _expect_rc(_Done(p.returncode, o, e), 2, "TRAINER_NO_LEDGER_ENTRY",
                   "control: inherited SIG_IGN swallows SIGINT")
        # -- R2: CANCEL and STOP during preflight start no run -----------------
        for name, act, rc, code, note in (
                ("bcancel", "sigterm", 2, "CANCELLED",
                 "cancellation at the launch boundary"),
                ("bstop", "stop", 0, "STOP_REQUESTED",
                 "STOP present at the launch boundary")):
            s = _case_scratch(base, name)
            w = s / "pywrap.sh"
            w.write_text(PYWRAP_SRC.format(scratch=s, py=sys.executable))
            w.chmod(0o755)
            _launcher_copy(s, py=str(w))
            c = _Campaign(s, name)
            p = _start_launcher(c, "new")
            _wait_for(s / "preflight_entered", f"{name}: preflight hold")
            if act == "sigterm":
                os.kill(p.pid, sg.SIGTERM)
            else:
                (c.root / "STOP").write_bytes(b"stop during preflight\n")
            time.sleep(0.5)
            (s / "preflight_release").write_text("go\n")
            o, e = p.communicate(timeout=B29_TIMEOUT)
            started = list(s.glob("stub_started_*.txt"))
            runlogs = list(c.root.glob("runlog_*"))
            if started or runlogs or note not in e:
                _fail("B29_BOUNDARY", f"{name}: trainer started={started} "
                      f"runlogs={runlogs}; tail: {e[-1500:]}")
            _expect_rc(_Done(p.returncode, o, e), rc, code,
                       f"{name}: request during preflight starts no run")
        logger.info("B2.9 signal PASS: real main() via launcher spawn path "
                    "gets exactly one SIGINT; under inherited SIG_IGN the fix "
                    "restores it; pending "
                    "cancellation forwarded after handler install; control "
                    "shows the hazard; CANCEL/STOP during preflight start "
                    "no run")
    finally:
        for f in base.glob("*/shim_pid.txt"):
            _kill_pidfile(f)
        res = subprocess.run(["pgrep", "-f", str(STUB)], capture_output=True,
                             text=True, check=False)
        for pid in res.stdout.split():
            os.kill(int(pid), sg.SIGKILL)
        _kill_all_inhibitors(base)
        shutil.rmtree(base, ignore_errors=False)


class _Done:
    def __init__(self, returncode, stdout, stderr):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def b29_handoff() -> None:
    """R5/R6: a campaign produced by the launcher copy (stub 'full' mode,
    valid 30-epoch metadata, two sessions with STOP/ack-stop) is passed
    directly to the summary."""
    import os
    import signal as sg
    import time
    scratch = Path(tempfile.mkdtemp(prefix="v04dlt_b29h_"))
    try:
        (scratch / "V04DLT_STUB_SCRATCH").write_text("fixture scratch\n")
        _launcher_copy(scratch)
        _plan(scratch, {"0_101_1": ["full", 3], "default": ["full", 0]})
        c = _Campaign(scratch, "h")
        p = _start_launcher(c, "new")
        _wait_for(scratch / "stub_started_0_101_1.txt", "handoff session 1")
        (c.root / "STOP").write_bytes(b"pause\n")
        o, e = p.communicate(timeout=B29_TIMEOUT)
        _expect_rc(_Done(p.returncode, o, e), 0, "STOP_REQUESTED",
                   "handoff session 1")
        _expect_rc(_run_launcher(c, "ack-stop"), 0, "ACK_STOP_PASS",
                   "handoff ack-stop")
        time.sleep(1.1)
        _expect_rc(_run_launcher(c, "resume"), 0, "CAMPAIGN_COMPLETE",
                   "handoff session 2")
        if len(_sessions(c.root)) != 2 or \
                len(list(c.root.glob("campaign_manifest_*.sha256.sha256"))) != 2:
            _fail("B29_HANDOFF", "expected 2 sessions with manifest sidecars")
        ref = _b210_v04_ref(scratch / "v04ref")
        pins = {"V04_LEDGER_SHA": v04t._file_sha256(ref / "v04_run_ledger.json"),
                "V04_RECORD_SHA": {(a, s): v04t._file_sha256(
                    ref / v04t.run_record_name(a, s, 1))
                    for a in dls.ARMS for s in dls.SEEDS}}
        with _patched(dls, **pins):
            neg = scratch / "neg" / c.root.name
            shutil.copytree(c.root, neg)
            if dls.build(c.root, ref) != 0:
                _fail("B29_HANDOFF", "summary rc != 0")
            sm = _summary_ok(c.root)
            if sm["paired"]["completed_matched_pairs"] != 3:
                _fail("B29_HANDOFF", f"pairs {sm['paired']}")
            s2 = _sessions(neg)[-1]
            _remove(neg, *[f"runlog_arm1_seed303_attempt1_{s2}.{x}.log{y}"
                           for x in ("stdout", "stderr")
                           for y in ("", ".sha256")])
            _expect_refusal(neg, ref, "MISSING_RUN_LOG",
                            "launcher campaign, both run logs removed")
        logger.info("B2.9 handoff PASS: launcher-produced 2-session campaign "
                    "(manifest sidecars, both run logs) accepted by the "
                    "summary, 3 pairs; removed run logs refused")
    finally:
        res = subprocess.run(["pgrep", "-f", str(STUB)], capture_output=True,
                             text=True, check=False)
        for pid in res.stdout.split():
            os.kill(int(pid), sg.SIGKILL)
        _kill_all_inhibitors(scratch)
        shutil.rmtree(scratch, ignore_errors=False)


def b29_all() -> None:
    b29_launcher()
    b29_signals()
    b29_handoff()


# ---------------------------------------------------------------------------
# B2.10 summary fixture: tiny synthetic campaigns, valid 30-epoch metadata
# ---------------------------------------------------------------------------

B210_PIN = "b2100c0de" + "0" * 31             # synthetic 40-hex pin
B210_SESSIONS = ("20261017T000001Z", "20261017T000002Z")


def _vals(arm: int, seed: int, e: int) -> dict:
    nll = 1.0 + 0.05 * arm + 0.001 * seed + 1.0 / e
    rec = 0.5 + 0.002 * seed + 0.2 / e
    lam = 0.0 if arm == 0 else dls.LAMBDA_2P
    return {"val_NLL": nll, "val_L_rec": rec, "L_val": nll + lam * rec}


def _hist(arm: int, seed: int, n: int, spe: int) -> list:
    lam = 0.0 if arm == 0 else dls.LAMBDA_2P
    out = []
    for i in range(1, n + 1):
        d = 2.0 + 0.001 * seed + 1.0 / i
        r = None if arm == 0 else 0.6 + 0.5 / i
        out.append({"step": i, "epoch": (i - 1) // spe, "L_density": d,
                    "L_rec": r, "lam_L_rec": None if r is None else lam * r,
                    "L_total": d + (0.0 if r is None else lam * r),
                    "grad_norm": 1.0 + 1.0 / i, "lr": v04t.LEARNING_RATE})
    return out


def _ck_bytes(name: str) -> bytes:
    return f"synthetic checkpoint {name}\n".encode()


def _b210_record(arm, seed, att, status, steps, k, nh, code,
                 epochs=None, total=None, spe=None, pin=B210_PIN) -> dict:
    epochs = dls.N_EPOCHS if epochs is None else epochs
    total = dls.TOTAL_STEPS if total is None else total
    spe = dls.STEPS_PER_EPOCH if spe is None else spe
    lam = 0.0 if arm == 0 else dls.LAMBDA_2P
    cks = []
    for e in range(1, k + 1):
        name = v04t.checkpoint_name(arm, seed, att, e)
        cks.append({"epoch": e, "file": name,
                    "file_sha256": hashlib.sha256(_ck_bytes(name)).hexdigest(),
                    "arm": arm, "seed": seed, "attempt": att, "lambda": lam,
                    "init_seed": seed, **_vals(arm, seed, e)})
    return {"schema": v04t.RECORD_SCHEMA, "script": "B2.10 synthetic",
            "arm": arm, "lambda": lam, "seed": seed, "attempt": att,
            "status": status, "error_code": code, "steps_completed": steps,
            "config": {"epochs": epochs, "steps": total,
                       "torch_num_threads": 4, "study": dlt.STUDY},
            "commit_pin": pin,
            "history": _hist(arm, seed, nh, spe),
            "validation_history": [{"epoch": e, **_vals(arm, seed, e)}
                                   for e in range(1, k + 1)],
            "checkpoints": cks, "wall_clock_s": 3600.0 * (1 + arm),
            "environment": {"python": "3.12.3", "torch": "synthetic",
                            "platform": "synthetic"}}


def _completed(arm, seed, att=1):
    return (arm, seed, att, "completed", dls.TOTAL_STEPS, dls.N_EPOCHS,
            dls.TOTAL_STEPS, None)


def _write_sha(root: Path, name: str) -> None:
    (root / f"{name}.sha256").write_text(
        f"{v04t._file_sha256(root / name)}  {name}\n")


def _b210_campaign(base: Path, name: str, runs: list, *, mut_rec=None,
                   mut_ent=None, last_prefix_short: bool = False) -> Path:
    """Seal a synthetic campaign exactly as trainer + launcher would:
    checkpoints, records (+sidecar), append-only ledger (+sidecar), and two
    launcher sessions (log + sidecar, snapshot + sidecar, manifest)."""
    root = base / name / f"V04DLT_{B210_PIN[:8]}"
    root.mkdir(parents=True)
    recs = [_b210_record(*r) for r in runs]
    if mut_rec:
        mut_rec(recs)
    lp = root / "v04_run_ledger.json"
    entries = []
    for r, rec in zip(runs, recs):
        arm, seed, att = r[0], r[1], r[2]
        for c in rec["checkpoints"]:
            (root / c["file"]).write_bytes(_ck_bytes(c["file"]))
        rp = root / v04t.run_record_name(arm, seed, att)
        rsha = v04t._write_with_sidecar(rp, v02m.canonical_json(rec))
        entries.append({"arm": arm, "lambda": rec["lambda"], "seed": seed,
                        "attempt": att, "status": rec["status"],
                        "error_code": rec["error_code"],
                        "steps_completed": rec["steps_completed"],
                        "wall_clock_s": rec["wall_clock_s"],
                        "record_file": rp.name, "record_sha256": rsha,
                        "checkpoints": [c["file_sha256"]
                                        for c in rec["checkpoints"]]})
    if mut_ent:
        mut_ent(entries)
    for en in entries:
        v04t.append_entry(lp, en)
    led = v04t.load_ledger(lp)
    n = len(led["entries"])
    cuts = (n // 2, n - 1 if last_prefix_short else n)
    prev = 0
    for s, cut in zip(B210_SESSIONS, cuts):
        for en in led["entries"][prev:cut]:     # launcher: both logs per run
            for stream in ("stdout", "stderr"):
                rl = (f"runlog_arm{en['arm']}_seed{en['seed']}_attempt"
                      f"{en['attempt']}_{s}.{stream}.log")
                (root / rl).write_text(f"synthetic {stream}\n")
                _write_sha(root, rl)
        prev = cut
        log = f"launcher_session_{s}.log"
        (root / log).write_text(f"synthetic session {s}\n")
        _write_sha(root, log)
        snap = f"ledger_snapshot_{s}.json"
        (root / snap).write_bytes(v02m.canonical_json(
            {**{k: v for k, v in led.items() if k != "entries"},
             "entries": led["entries"][:cut]}))
        _write_sha(root, snap)
        excl = ("v04_run_ledger.json", "v04_run_ledger.json.sha256", "STOP",
                ".launch.lock")
        lines = [f"{v04t._file_sha256(f)}  {f.name}\n"
                 for f in sorted(root.iterdir())
                 if f.is_file() and f.name not in excl
                 and not f.name.startswith("campaign_manifest_")]
        man = root / f"campaign_manifest_{s}.sha256"
        man.write_text("".join(lines))
        _write_sha(root, man.name)              # launcher: manifest sidecar
    return root


def _b210_v04_ref(base: Path) -> Path:
    ref = base / "V04TR_ref" / "train"
    ref.mkdir(parents=True)
    lp = ref / "v04_run_ledger.json"
    for seed in dls.SEEDS:
        for arm in dls.ARMS:
            rec = _b210_record(arm, seed, 1, "completed", dls.V04_STEPS,
                               dls.V04_EPOCHS, dls.V04_STEPS, None,
                               epochs=dls.V04_EPOCHS, total=dls.V04_STEPS)
            rp = ref / v04t.run_record_name(arm, seed, 1)
            rsha = v04t._write_with_sidecar(rp, v02m.canonical_json(rec))
            v04t.append_entry(lp, {"arm": arm, "seed": seed, "attempt": 1,
                                   "status": "completed",
                                   "record_file": rp.name,
                                   "record_sha256": rsha})
    return ref


def _drop_manifest_line(root: Path, name: str) -> None:
    for m in root.glob("campaign_manifest_*.sha256"):
        if m.name.endswith(".sha256.sha256"):
            continue
        lines = [ln for ln in m.read_text().splitlines(keepends=True)
                 if ln.split()[1] != name]
        m.write_text("".join(lines))
        if (root / f"{m.name}.sha256").exists():   # re-seal: isolate the case
            _write_sha(root, m.name)


def _remove(root: Path, *names: str) -> None:
    for nm in names:
        (root / nm).unlink()
        _drop_manifest_line(root, nm)


def _expect_refusal(root: Path, ref: Path, code: str, name: str) -> None:
    try:
        dls.build(root, ref)
    except (dls.V04DLTSError, v04t.V04TError) as exc:
        if exc.code != code:
            _fail("B210_WRONG_REFUSAL", f"{name}: {exc.code} != {code}")
        if (root / "summary").exists() and code != "ARTIFACT_EXISTS":
            _fail("B210_WROTE_ON_REFUSAL", f"{name}: summary dir created")
        logger.info("B2.10 branch PASS: %s -> %s", name, code)
        return
    _fail("B210_NOT_REFUSED", f"{name}: build passed; expected {code}")


def _summary_ok(root: Path) -> dict:
    out = root / "summary"
    for side in sorted(out.glob("*.sha256")):
        res = subprocess.run(["sha256sum", "-c", "--quiet", side.name],
                             cwd=out, capture_output=True, text=True,
                             check=False)
        if res.returncode != 0:
            _fail("B210_OUTPUT_SIDECAR", f"{side.name}: {res.stdout}")
    summ = json.loads((out / "v04dlt_summary.json").read_text())
    man = json.loads((out / "plot_manifest.json").read_text())
    if len(man["plots"]) != 8 or any(
            v04t._file_sha256(out / pl["file"]) != pl["sha256"]
            for pl in man["plots"]):
        _fail("B210_PLOTS", "plot set or plot hashes wrong")
    if man["summary_sha256"] != v04t._file_sha256(out / "v04dlt_summary.json"):
        _fail("B210_PLOTS", "manifest summary hash wrong")
    return summ


def b210_summary() -> None:
    base = Path(tempfile.mkdtemp(prefix="v04dlt_b210_"))
    try:
        ref = _b210_v04_ref(base)
        pins = {"V04_LEDGER_SHA": v04t._file_sha256(ref / "v04_run_ledger.json"),
                "V04_RECORD_SHA": {(a, s): v04t._file_sha256(
                    ref / v04t.run_record_name(a, s, 1))
                    for a in dls.ARMS for s in dls.SEEDS}}
        full = [_completed(a, s) for s in dls.SEEDS for a in dls.ARMS]
        with _patched(dls, **pins):
            # -- n = 3 ------------------------------------------------------------
            r3 = _b210_campaign(base, "n3", full)
            pristine = base / "pristine"
            shutil.copytree(r3, pristine / r3.name)
            if dls.build(r3, ref) != 0:
                _fail("B210_N3", "build rc != 0")
            sm = _summary_ok(r3)
            pr = sm["paired"]
            nck = sum(1 for k in sm["input_hash_register"]
                      if k.startswith("checkpoint/"))
            if pr["completed_matched_pairs"] != 3 or \
                    pr["cross_seed"]["arm0_val_NLL_epoch30"]["sd"] is None or \
                    nck != 180 or any(r["selected_epoch"] != 30
                                      for r in sm["run_table"]):
                _fail("B210_N3", f"pairs {pr['completed_matched_pairs']}, "
                      f"checkpoints registered {nck}")
            _expect_refusal(r3, ref, "ARTIFACT_EXISTS", "rebuild")
            # -- n = 2: terminal numerical failure ------------------------------
            runs = full[:5] + [(1, 303, 1, "numerical_failure", 5000, 4, 4999,
                                "NUMERICAL_FAILURE")]
            r = _b210_campaign(base, "n2", runs)
            dls.build(r, ref)
            sm = _summary_ok(r)
            row = [x for x in sm["run_table"] if (x["arm"], x["seed"]) ==
                   (1, 303)][0]
            if sm["paired"]["completed_matched_pairs"] != 2 or \
                    row["exclusion_cause"] != "numerical_failure (terminal)" \
                    or "selected_epoch" in row:
                _fail("B210_N2", f"{sm['paired']['seeds']} {row}")
            # -- n = 1: interrupted (attempts remain) + 3 attempts exhausted ----
            runs = [_completed(0, 101), _completed(1, 101),
                    (0, 202, 1, "interrupted", 1500, 1, 1499,
                     "KEYBOARD_INTERRUPT"), _completed(1, 202),
                    _completed(0, 303)] + [
                (1, 303, a, "interrupted", 1086 * a, a, 1086 * a,
                 "INFRASTRUCTURE_INTERRUPTION") for a in (1, 2, 3)]
            r = _b210_campaign(base, "n1", runs)
            dls.build(r, ref)
            sm = _summary_ok(r)
            causes = {(x["arm"], x["seed"]): x["exclusion_cause"]
                      for x in sm["run_table"]}
            x = sm["paired"]["cross_seed"]["arm0_val_NLL_epoch30"]
            if sm["paired"]["seeds"] != [101] or x["n"] != 1 or \
                    x["sd"] is not None or \
                    causes[(0, 202)] != "incomplete (interrupted, attempts " \
                                        "remain)" or \
                    causes[(1, 303)] != "infrastructure interruption, 3 " \
                                        "attempts exhausted":
                _fail("B210_N1", f"{sm['paired']['seeds']} {x} {causes}")
            # -- n = 0: failed_validity, not attempted --------------------------
            runs = [_completed(0, 101),
                    (1, 101, 1, "failed_validity", 10, 0, 9,
                     "V04L_INTERFACE_MISMATCH"), _completed(0, 202)]
            r = _b210_campaign(base, "n0", runs)
            dls.build(r, ref)
            sm = _summary_ok(r)
            causes = {(x["arm"], x["seed"]): x["exclusion_cause"]
                      for x in sm["run_table"]}
            if sm["paired"]["completed_matched_pairs"] != 0 or \
                    "cross_seed" in sm["paired"] or \
                    causes[(1, 101)] != "failed_validity (terminal)" or \
                    causes[(1, 303)] != "not_attempted":
                _fail("B210_N0", f"{sm['paired']} {causes}")
            # -- overwrite refusal before reading -------------------------------
            r = base / "pre" / r3.name
            shutil.copytree(pristine / r3.name, r)
            (r / "summary").mkdir()
            _expect_refusal(r, ref, "ARTIFACT_EXISTS", "pre-existing summary")
            # -- missing evidence (manifest lines dropped to isolate the check) -
            s1, s2 = B210_SESSIONS
            rec0 = v04t.run_record_name(0, 101, 1)
            ck0 = v04t.checkpoint_name(0, 101, 1, 7)
            cases = (
                ("no manifest", lambda r: [
                    (r / f"campaign_manifest_{s2}.sha256{y}").unlink()
                    for y in ("", ".sha256")], "MISSING_SESSION_EVIDENCE"),
                ("no log sidecar", lambda r: _remove(
                    r, f"launcher_session_{s1}.log.sha256"),
                 "MISSING_SESSION_EVIDENCE"),
                ("no manifest sidecar", lambda r: (
                    r / f"campaign_manifest_{s2}.sha256.sha256").unlink(),
                 "MISSING_SESSION_EVIDENCE"),
                ("both run logs removed together", lambda r: _remove(
                    r, *[f"runlog_arm1_seed303_attempt1_{s2}.{x}.log{y}"
                         for x in ("stdout", "stderr") for y in ("", ".sha256")]),
                 "MISSING_RUN_LOG"),
                ("stderr log removed", lambda r: _remove(
                    r, f"runlog_arm0_seed101_attempt1_{s1}.stderr.log",
                    f"runlog_arm0_seed101_attempt1_{s1}.stderr.log.sha256"),
                 "MISSING_RUN_LOG"),
                ("run logs moved to another session", lambda r: [
                    (_drop_manifest_line(r, f"{b}_{s1}.{x}.log{y}"),
                     (r / f"{b}_{s1}.{x}.log{y}").rename(
                         r / f"{b}_{s2}.{x}.log{y}"))
                    for b in ("runlog_arm0_seed101_attempt1",)
                    for x in ("stdout", "stderr") for y in ("", ".sha256")]
                 and [_write_sha(r, f"runlog_arm0_seed101_attempt1_{s2}.{x}.log")
                      for x in ("stdout", "stderr")],
                 "MISSING_RUN_LOG"),
                ("run log not in its manifest", lambda r: _drop_manifest_line(
                    r, f"runlog_arm0_seed101_attempt1_{s1}.stdout.log"),
                 "RUN_LOG_NOT_IN_MANIFEST"),
                ("no snapshot", lambda r: _remove(
                    r, f"ledger_snapshot_{s2}.json",
                    f"ledger_snapshot_{s2}.json.sha256"),
                 "MISSING_SESSION_EVIDENCE"),
                ("no sessions", lambda r: [
                    (_remove(r, f.name) if not f.name.startswith(
                        "campaign_manifest_") else f.unlink())
                    for f in sorted(r.iterdir()) if f.name.startswith(
                        ("launcher_session_", "ledger_snapshot_",
                         "campaign_manifest_")) and f.exists()],
                 "NO_SESSION_EVIDENCE"),
                ("orphan manifest", lambda r: shutil.copyfile(
                    r / f"campaign_manifest_{s2}.sha256",
                    r / "campaign_manifest_20261017T000009Z.sha256"),
                 "ORPHANED_SESSION_EVIDENCE"),
                ("no record sidecar", lambda r: _remove(r, f"{rec0}.sha256"),
                 "RECORD_SIDECAR_MISSING"),
                ("no ledger sidecar", lambda r: (
                    r / "v04_run_ledger.json.sha256").unlink(),
                 "LEDGER_TAMPERED"),
                ("no run-log sidecar", lambda r: _remove(
                    r, f"runlog_arm0_seed101_attempt1_{s1}.stdout.log.sha256"),
                 "MISSING_RUN_LOG_SIDECAR"),
                ("checkpoint missing", lambda r: _remove(r, ck0),
                 "CHECKPOINT_MISSING"),
                ("checkpoint bytes", lambda r: (
                    _drop_manifest_line(r, ck0),
                    (r / ck0).write_bytes(b"tampered\n")),
                 "CHECKPOINT_HASH_MISMATCH"),
                ("unregistered record", lambda r: shutil.copyfile(
                    r / rec0, r / v04t.run_record_name(0, 101, 2)),
                 "UNREGISTERED_RECORD"),
                ("record bytes", lambda r: (
                    _drop_manifest_line(r, rec0),
                    _drop_manifest_line(r, f"{rec0}.sha256"),
                    (r / rec0).write_bytes((r / rec0).read_bytes() + b" "),
                    _write_sha(r, rec0)),
                 "RECORD_HASH_MISMATCH"),
            )
            for i, (name, mutate, code) in enumerate(cases):
                r = base / f"miss{i}" / r3.name
                shutil.copytree(pristine / r3.name, r)
                mutate(r)
                _expect_refusal(r, ref, code, name)
            r = _b210_campaign(base, "short", full, last_prefix_short=True)
            _expect_refusal(r, ref, "LEDGER_NOT_SNAPSHOTTED",
                            "ledger entry outside any session snapshot")
            # -- inconsistent records (sealed consistently: only semantics) -----

            def _set(i, **kv):
                return lambda xs: xs[i].update(kv)

            def _drop_last_ck(rs):
                rs[0]["checkpoints"].pop()
                rs[0]["validation_history"].pop()

            def _short_steps(rs):
                rs[0]["steps_completed"] -= 1
                rs[0]["history"].pop()

            def _lam(rs):
                rs[1]["lambda"] = 2.0
                for c in rs[1]["checkpoints"]:
                    c["lambda"] = 2.0
            semantic = (
                ("status differs", None, _set(0, status="interrupted"),
                 "RECORD_LEDGER_MISMATCH"),
                ("checkpoint ref differs", None,
                 lambda es: es[0]["checkpoints"].__setitem__(0, "0" * 64),
                 "RECORD_LEDGER_MISMATCH"),
                ("steps differ", None, _set(0, steps_completed=1),
                 "RECORD_LEDGER_MISMATCH"),
                ("lambda wrong", _lam, None, "RECORD_LAMBDA_MISMATCH"),
                ("completed 32579 steps", _short_steps, None,
                 "RECORD_COVERAGE_INVALID"),
                ("completed 29 epochs", _drop_last_ck, None,
                 "RECORD_COVERAGE_INVALID"),
                ("pin differs", _set(2, commit_pin="f" * 40), None,
                 "RECORD_PIN_MISMATCH"),
                ("config steps", lambda rs: rs[0]["config"].update(
                    steps=dls.V04_STEPS), None, "RECORD_CONFIG_MISMATCH"),
                ("identity", _set(0, seed=202), None,
                 "RECORD_IDENTITY_MISMATCH"),
            )
            for i, (name, mr, me, code) in enumerate(semantic):
                r = _b210_campaign(base, f"sem{i}", full, mut_rec=mr,
                                   mut_ent=me)
                _expect_refusal(r, ref, code, name)
        logger.info("B2.10 PASS: summary n=3/2/1/0 (terminal, incomplete, "
                    "exhausted, not attempted); outputs + plot hashes verify; "
                    "overwrite refused; %d missing-evidence and %d "
                    "inconsistent-record cases refused before writing",
                    len(cases) + 1, len(semantic))
    finally:
        shutil.rmtree(base, ignore_errors=False)



def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    argv = sys.argv[1:] if argv is None else argv
    logger.info("fixture environment: torch %s, numpy %s, threads %d",
                torch.__version__, np.__version__, torch.get_num_threads())
    groups = [("B2.0", b20_equivalence), ("B2.1", b21_budget),
              ("B2.2", b22_lambda), ("B2.3", b23_path_guard),
              ("B2.4", b24_two_step_smoke), ("B2.5", b25_mask_contract),
              ("B2.6", b26_failure_paths), ("B2.7", b27_mode_partial_batch),
              ("B2.8", b28_regression)]
    groups.append(("B2.9", b29_all))
    groups.append(("B2.10", b210_summary))
    if argv:
        groups = [g for g in groups if g[0] in argv]
        if not groups:
            logger.error("no fixture group matches %s", argv)
            return 2
    results = {}
    for name, fn in groups:
        try:
            fn()
            results[name] = "PASS"
        except FixtureFailure as exc:
            results[name] = f"FAIL ({exc.code})"
        except Exception as exc:  # noqa: BLE001 -- recorded, never swallowed
            logger.exception("%s raised unexpectedly", name)
            results[name] = f"ERROR ({type(exc).__name__})"
    for name, res in results.items():
        logger.info("fixture %s: %s", name, res)
    bad = {k: v for k, v in results.items() if v != "PASS"}
    if bad:
        logger.error("V04DLT fixtures: %d not PASS: %s", len(bad), bad)
        return 2
    logger.info("V04DLT fixtures %s: ALL PASS", "/".join(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
