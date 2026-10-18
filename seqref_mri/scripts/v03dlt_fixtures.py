# SEQREF-MRI-V03DLTF v0.2 -- scripts.v03dlt_fixtures
# LIFETIME: KEEP
# =============================================================================
# Purpose: required fixtures B2.0-B2.11 of SEQREF-MRI-V03DLT v1.4 (Article
#          190, DESIGN FROZEN 2026-10-18 + AM1-AM4) and V03DLTPLAN v1.3
#          (Article 189, B2) for the NSF longer-training trainer (V03DLTT),
#          launcher (V03DLTL) and summary (V03DLTS). Synthetic CPU data only,
#          each fixture in its own scratch directory; no real data root, no
#          V03 evidence written, no holdout.
# Adapted from SEQREF-MRI-V04DLTF v0.3 (pin 9190735; verified archive
#   a473c4af...). V03 has no end-to-end synthetic helpers, so the synthetic
#   volume, P4 /2 artefact and probe record writers are written here (not
#   imported from any V04 module).
# Synthetic stand-ins (disclosed, fixture-confined, never production):
#   * synthetic single-coil h5 volumes (kspace + reconstruction_esc + 'max'),
#     a synthetic P4 /2 artefact (8,448 PER-LOCATION records, mean 0, scale
#     1) and a synthetic V03 probe record whose alpha-0.5 lambda equals the
#     registered lambda_2;
#   * the registered P4 pins are swapped through load_p4s2_parent
#     __kwdefaults__ and on the V03DLTT module, and budget / validation
#     constants on V03T and V03DLTT, each restored in `finally`;
#   * B2.8 injects failures by wrapping tff.nll_objective, global_grad_norm
#     or assert_checkpoint_identity inside the fixture, restored in finally;
#   * B2.4 import-origin cases insert fake module objects into sys.modules
#     (removed in finally); no file is ever written into the repository;
#   * B2.10 runs COPIES of the launcher. Per 190 AM4 a copy differs from the
#     production launcher only in the trainer-command line and the lines of
#     the marked HOST CONSTANTS block (isolated synthetic evidence, hashes,
#     branch, exclusion-list hash, PY, CKPT_BYTES, and a `sleep infinity`
#     stand-in for systemd-inhibit); every other line is asserted
#     byte-identical. Group B2.10H (host only, not in the default run) keeps
#     the production HOST CONSTANTS block and replaces only the trainer line;
#   * B2.11 builds tiny synthetic campaigns (checkpoint files of a few
#     bytes; valid 30-epoch / 32,580-step record metadata written directly,
#     no training) and synthetic V03 reference records; the summary's
#     registered V03 hash constants are swapped on the module for the
#     fixture's scope only and restored in `finally`.
# CONVENTION: PASS(0)/ERROR(2); every failure -> logger.error + typed code.
#   No fallback, no silent pass.
# Changelog (v0.2, 2026-10-18; review of package d1a1414b):
#   * B2.11: 10 ledger-history refusals (retry after numerical_failure /
#     failed_validity / completed, attempt gap / start / fourth attempt,
#     out-of-order, run past an unsettled run, arm 1, completed + code).
#   * B2.11: 14 further record refusals (history fields / finiteness incl.
#     numerical_failure / arm terms / epoch, P4 semantic + path, wall clock
#     inf / negative, monitor identity / finiteness / checkpoint agreement).
# Changelog (NEW in v0.1, 2026-10-18):
#   * Introduced under V03DLTPLAN v1.3 B2: B2.0 equivalence vs V03T v0.5 _run
#     (synthetic arm-1 predecessor), B2.1 budget, B2.2 lambda/arm, B2.3
#     order/attempts, B2.4 path guard + import origin, B2.5 smoke, B2.6 mask
#     contract, B2.7 P4 binding, B2.8 failure paths incl. AM2, B2.9 mode /
#     partial batch / V03 selftests, B2.10 launcher (AM4), B2.11 summary
#     with the 190 A5 population cases.
# =============================================================================
from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import h5py
import numpy as np
import torch

from seqref_mri.src import free_flow_runtime as ffr
from seqref_mri.src.fastmri_data import (TRAIN_BASE_SEED, FastMRISliceDataset,
                                         canonical_mask_seed, fft2c,
                                         make_cartesian_mask, mask_counts)
from seqref_mri.src.flows import v03_objective as v03l
from seqref_mri.scripts import train_free_flow as tff
from seqref_mri.scripts import v02_manifests as v02m
from seqref_mri.scripts import v03_train as v03t
from seqref_mri.scripts import v03dlt_train as dlt
from seqref_mri.scripts import v03dlt_summary as dls
from seqref_mri.scripts.v03_eval import select_checkpoint
from seqref_mri.scripts.v03_probe import V03PError

logger = logging.getLogger("seqref_mri.v03dlt_fixtures")
__version__ = "0.2"
__abbr__ = "SEQREF-MRI-V03DLTF"

REPO = Path(__file__).resolve().parents[2]
REAL_DATA_ROOT = Path("/home/benjamin/CSMFII/seqref_mri/data/fastmri")
SEED = 101                       # first locked seed (order arm 0, arm 2)
SMALL_SLICES = 33                # 32 + 1: two steps/epoch, partial batch
SMALL_STEPS = 2
B20_EPOCHS = 2
B20_N_VAL = 1
MASK_SAMPLE = (0, 17, 32)        # B2.6 fixed slice sample
KSPACE_HW = 112                  # synthetic k-space grid (>= 96 crop)
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
# Synthetic workspace (data + P4 + probe record), confined to scratch
# ---------------------------------------------------------------------------

def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_volume(path: Path, n_slices: int) -> None:
    """Synthetic single-coil volume: smooth complex images -> centred
    orthonormal k-space (the dataset's own FFT convention)."""
    rng = np.random.Generator(np.random.PCG64(20261018))
    yy, xx = np.mgrid[0:KSPACE_HW, 0:KSPACE_HW].astype(np.float32)
    imgs = np.zeros((n_slices, KSPACE_HW, KSPACE_HW), dtype=np.complex64)
    for s in range(n_slices):
        im = np.zeros((KSPACE_HW, KSPACE_HW), dtype=np.float32)
        for _ in range(4):
            cy, cx = rng.uniform(30, KSPACE_HW - 30, size=2)
            r = rng.uniform(6, 18)
            im += rng.uniform(0.3, 1.0) * np.exp(
                -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
        phase = np.exp(1j * rng.uniform(-0.3, 0.3)).astype(np.complex64)
        imgs[s] = (im * phase).astype(np.complex64)
    k = fft2c(torch.from_numpy(imgs)).numpy().astype(np.complex64)
    esc = np.abs(imgs).astype(np.float32)
    with h5py.File(path, "w") as h:
        h.create_dataset("kspace", data=k)
        h.create_dataset("reconstruction_esc", data=esc)
        h.attrs["max"] = float(esc.max())


def _write_p4(path: Path) -> tuple:
    """Synthetic P4 /2 artefact accepted by the pinned loader structure:
    authoritative PASS, PER-LOCATION, all 8,448 non-centre locations."""
    width = 96
    n_center, _ = mask_counts(width)
    start = (width - n_center) // 2
    centre = set(range(start, start + n_center))
    locs = [{"row": r, "column": c, "applied_mean_re": 0.0,
             "applied_mean_im": 0.0, "applied_scale_re": 1.0,
             "applied_scale_im": 1.0}
            for r in range(width) for c in range(width) if c not in centre]
    if len(locs) != ffr.REGISTERED_P4_LOCATIONS:
        _fail("FIXTURE_P4_LOCATIONS",
              f"{len(locs)} != {ffr.REGISTERED_P4_LOCATIONS}")
    sem = _sha_bytes(v02m.canonical_json({"synthetic_p4": locs}))
    art = {"schema": ffr.P4S2_FACTS_SCHEMA, "stage": "P4",
           "authoritative": True, "run_mode": "authoritative",
           "verdict": "PASS", "semantic_sha256": sem,
           "branch": {"selected": "PER-LOCATION"}, "locations": locs,
           "parents": {"note": "synthetic fixture artefact"}}
    payload = v02m.canonical_json(art)
    path.write_bytes(payload)
    fsha = _sha_bytes(payload)
    Path(str(path) + ".sha256").write_text(f"{fsha}  {path.name}\n")
    return fsha, sem


def _probe_record(lam2: float = None) -> dict:
    return {"schema": v03t.PROBE_SCHEMA, "n_batches": 100,
            "batch_size": v02m.BATCH_SIZE, "split": "train", "mode": "eval",
            "validation_source_amendment": "A2",
            "l_rec_scale": v03l.L_REC_SCALE,
            "lambdas": {"0.1": 0.06811490724179332,
                        "0.5": dlt.REGISTERED_LAMBDA_ARM2
                        if lam2 is None else lam2,
                        "0.9": 5.51730748658526}}


def _workspace(tmp: Path, n_slices: int, lam2: float = None) -> dict:
    data_root = tmp / "data"
    split = data_root / "knee_singlecoil_train" / "singlecoil_train"
    split.mkdir(parents=True)
    _write_volume(split / "synthetic_volume.h5", n_slices)
    p4 = tmp / "p4" / "scaling_statistics.json"
    p4.parent.mkdir(parents=True)
    pins = _write_p4(p4)
    rec_dir = tmp / "records"
    rec_dir.mkdir()
    probe = rec_dir / "v03_probe_record.json"
    v03t._write_with_sidecar(probe, v02m.canonical_json(_probe_record(lam2)))
    return {"tmp": tmp, "data_root": str(data_root), "p4": str(p4),
            "p4_pins": pins, "probe": str(probe)}


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


@contextlib.contextmanager
def _p4_pins(ws: dict, *, trainer_pins: bool = True):
    kw = ffr.load_p4s2_parent.__kwdefaults__
    saved = (kw["expected_file_sha"], kw["expected_semantic_sha"])
    kw["expected_file_sha"], kw["expected_semantic_sha"] = ws["p4_pins"]
    try:
        if trainer_pins:
            with _patched(dlt, REGISTERED_P4_FILE_SHA256=ws["p4_pins"][0],
                          REGISTERED_P4_SEMANTIC_SHA256=ws["p4_pins"][1]):
                yield
        else:
            yield
    finally:
        kw["expected_file_sha"], kw["expected_semantic_sha"] = saved


def _git(*args: str) -> str:
    res = subprocess.run(["git", "-C", str(REPO), *args],
                         capture_output=True, text=True, check=False)
    if res.returncode != 0:
        _fail("FIXTURE_GIT_UNAVAILABLE",
              f"git {' '.join(args)} failed at {REPO}: {res.stderr.strip()}")
    return res.stdout.strip()


def _head() -> str:
    return _git("rev-parse", "HEAD")


def _campaign_root(tmp: Path) -> Path:
    root = tmp / f"{dlt.CAMPAIGN_PREFIX}{_head()[:8]}"
    root.mkdir()
    return root


def _core(ws: dict, root: Path, arm: int, *, n_epochs: int,
          steps_per_epoch: int, n_val: int, total: int = None,
          trainer_pins: bool = True) -> int:
    total = n_epochs * steps_per_epoch if total is None else total
    try:
        with _p4_pins(ws, trainer_pins=trainer_pins), \
                _patched(v03t, N_VAL_BATCHES=n_val), \
                _patched(dlt, N_VAL_BATCHES=n_val):
            return dlt._run_core(ws["data_root"], ws["p4"], ws["probe"],
                                 _head(), str(root), arm, SEED,
                                 n_epochs=n_epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 total_steps=total, repo_root=REPO)
    finally:
        gc.collect()          # free the model/optimiser between cases


def _drop_tracebacks(exc: BaseException) -> None:
    """Clear the traceback of an exception and its whole cause/context
    chain (their frames keep each case's model and optimiser alive), then
    collect garbage."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        exc.__traceback__ = None
        exc = exc.__cause__ or exc.__context__
    gc.collect()


def _expect(code: str, name: str, fn, exc_type=v03t.V03TError) -> None:
    try:
        fn()
    except exc_type as exc:
        got = getattr(exc, "code", type(exc).__name__)
        _drop_tracebacks(exc)             # frames hold the model alive
        if got != code:
            _fail("FIXTURE_WRONG_CODE",
                  f"case {name!r} raised {got}, expected {code}")
        logger.info("branch PASS: %s -> %s", name, code)
    else:
        _fail(code, f"case {name!r} was accepted; the guard is dead")


def _record(root: Path, arm: int, attempt: int = 1, seed: int = SEED) -> dict:
    p = root / v03t.run_record_name(arm, seed, attempt)
    side = p.with_suffix(p.suffix + ".sha256")
    if not p.is_file() or not side.is_file():
        _fail("FIXTURE_RECORD_MISSING", f"record or sidecar missing: {p}")
    if side.read_text() != f"{v03t._file_sha256(p)}  {p.name}\n":
        _fail("FIXTURE_SIDECAR_MISMATCH", f"sidecar mismatch for {p}")
    return json.loads(p.read_text())


def _entry(arm, seed, att, status, steps=1, code=None, lam=None) -> dict:
    lam = (0.0 if arm == 0 else dlt.REGISTERED_LAMBDA_ARM2) \
        if lam is None else lam
    return {"arm": arm, "lambda": lam, "seed": seed, "attempt": att,
            "status": status, "error_code": code, "steps_completed": steps,
            "wall_clock_s": 1.0, "record_file": "r.json",
            "record_sha256": "x", "checkpoints": []}


# ---------------------------------------------------------------------------
# B2.0 equivalence: new loop vs V03T v0.5 _run, same small synthetic budget
# ---------------------------------------------------------------------------

def _equal(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return (a == b) or (np.isnan(a) and np.isnan(b))
    return a == b


def b20_equivalence() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="v03dlt_b20_"))
    try:
        ws = _workspace(tmp, SMALL_SLICES)
        ref_root = tmp / "v03t_reference"
        new_root = _campaign_root(tmp)
        total = B20_EPOCHS * SMALL_STEPS
        with _p4_pins(ws, trainer_pins=False), _patched(
                v03t, N_EPOCHS=B20_EPOCHS, STEPS_PER_EPOCH=SMALL_STEPS,
                TOTAL_STEPS=total, N_VAL_BATCHES=B20_N_VAL):
            if v03t._run(ws["data_root"], ws["p4"], ws["probe"],
                         str(ref_root), 0, SEED) != 0:
                _fail("B20_REFERENCE_FAILED", "V03T arm 0 != 0")
            # order satisfied, not bypassed: synthetic arm-1 predecessor
            v03t.append_entry(ref_root / "v03_run_ledger.json",
                              _entry(1, SEED, 1, "completed", steps=total,
                                     lam=0.06811490724179332))
            if v03t._run(ws["data_root"], ws["p4"], ws["probe"],
                         str(ref_root), 2, SEED) != 0:
                _fail("B20_REFERENCE_FAILED", "V03T arm 2 != 0")
        ledger_after_arm0 = None
        for arm in (0, 2):
            if _core(ws, new_root, arm, n_epochs=B20_EPOCHS,
                     steps_per_epoch=SMALL_STEPS, n_val=B20_N_VAL) != 0:
                _fail("B20_NEW_FAILED", f"V03DLTT arm {arm} != 0")
            if arm == 0:
                ledger_after_arm0 = v03t.load_ledger(
                    new_root / dlt.LEDGER_NAME)
        for arm in (0, 2):
            ra, rb = _record(ref_root, arm), _record(new_root, arm)
            if ra["steps_completed"] != total or len(ra["history"]) != total:
                _fail("B20_REFERENCE_NOT_TRAINED",
                      f"reference arm {arm}: {ra['steps_completed']} steps")
            if len(rb["history"]) != len(ra["history"]):
                _fail("B20_HISTORY_MISMATCH", f"arm {arm} lengths")
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
            if len(ra["checkpoints"]) != B20_EPOCHS or \
                    len(rb["checkpoints"]) != B20_EPOCHS:
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
                del sa, sb
            for c in ra["checkpoints"]:                 # free scratch disk
                (ref_root / c["file"]).unlink()
        la = [e for e in v03t.load_ledger(ref_root / "v03_run_ledger.json")
              ["entries"] if e["arm"] in (0, 2)]
        lb = v03t.load_ledger(new_root / dlt.LEDGER_NAME)["entries"]
        if [[e[k] for k in LEDGER_KEYS] for e in la] != \
                [[e[k] for k in LEDGER_KEYS] for e in lb]:
            _fail("B20_LEDGER_MISMATCH", "ledger status/attempt/steps differ")
        if lb[0] != ledger_after_arm0["entries"][0]:
            _fail("B24_LEDGER_NOT_APPEND_ONLY",
                  "earlier ledger entry changed after a later append")
        rb0, rb2 = _record(new_root, 0), _record(new_root, 2)
        if rb0["lambda"] != 0.0 or rb2["lambda"] != dlt.REGISTERED_LAMBDA_ARM2:
            _fail("B22_LAMBDA_MISMATCH",
                  f"record lambdas arm0={rb0['lambda']!r} "
                  f"arm2={rb2['lambda']!r}")
        if rb0["config"]["epochs"] != B20_EPOCHS or \
                rb0["script"] != f"{dlt.__abbr__} v{dlt.__version__}" or \
                rb0["schema"] != dlt.RECORD_SCHEMA_DLT:
            _fail("B20_PROVENANCE_MISMATCH", "new record script/config/schema")
        logger.info("B2.0 PASS: histories, validation, checkpoint tensors, run "
                    "fields and ledger transitions equal V03T v0.5 (%d steps x "
                    "2 arms; V03T order satisfied via a synthetic arm-1 "
                    "predecessor); lambdas exact; ledger append-only", total)
    finally:
        shutil.rmtree(tmp, ignore_errors=False)


# ---------------------------------------------------------------------------
# B2.1 budget / B2.2 lambda + arm / B2.3 order + attempts
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
            [sys.executable, "-m", "seqref_mri.scripts.v03dlt_train",
             "--mode", "run", opt, "3"], cwd=REPO, capture_output=True,
            text=True, check=False)
        if res.returncode != 2 or "unrecognized arguments" not in res.stderr:
            _fail("BUDGET_OPTION_EXPOSED",
                  f"production CLI accepted {opt} (rc {res.returncode})")
    logger.info("B2.1 PASS: locked 30 x 1086 = 32,580; other budgets and "
                "budget options refused")


def b22_lambda() -> None:
    rec = _probe_record()
    if dlt.lambda_for_arm_dlt(rec, 0) != 0.0:
        _fail("B22_LAMBDA_MISMATCH", "arm 0 lambda != 0.0")
    if dlt.lambda_for_arm_dlt(rec, 2) != 0.6130341651761398:
        _fail("B22_LAMBDA_MISMATCH", "arm 2 lambda not bit-equal")
    _expect("LAMBDA_MISMATCH", "probe lambda_2 differs",
            lambda: dlt.lambda_for_arm_dlt(_probe_record(0.6130341651762),
                                           2))
    for arm in (1, 3):
        _expect("ARM_INVALID", f"arm {arm}",
                lambda a=arm: dlt.validate_arm_seed_dlt(a, SEED))
    _expect("SEED_INVALID", "seed 404",
            lambda: dlt.validate_arm_seed_dlt(0, 404))
    logger.info("B2.2 PASS: arm 0 exactly 0; arm 2 bit-equal to "
                "0.6130341651761398; other lambda, arms 1/3 and seed refused "
                "(record values checked in B2.0)")


def b23_order_attempts() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="v03dlt_b23_"))
    try:
        want = [(101, 0), (101, 2), (202, 0), (202, 2), (303, 0), (303, 2)]
        if dlt.ORDER_DLT != want:
            _fail("B23_ORDER", f"ORDER_DLT {dlt.ORDER_DLT}")
        fresh = {"schema": v03t.LEDGER_SCHEMA, "entries": []}
        if dlt.expected_next_run_dlt(fresh) != (0, 101):
            _fail("B23_ORDER", "fresh ledger next run != arm 0 seed 101")
        for a, s in ((2, 101), (0, 202), (2, 303)):
            _expect("ORDER_VIOLATION", f"out of order arm {a} seed {s}",
                    lambda a=a, s=s: dlt.enforce_order_dlt(fresh, a, s))
        lp = tmp / "order.json"
        v03t.append_entry(lp, _entry(0, 101, 1, "completed"))
        if dlt.expected_next_run_dlt(v03t.load_ledger(lp)) != (2, 101):
            _fail("B23_ORDER", "completed run did not advance to arm 2")
        # three interrupted attempts: fourth refused; later runs blocked
        lp = tmp / "exhaust.json"
        for att in (1, 2, 3):
            v03t.append_entry(lp, _entry(0, 101, att, "interrupted",
                                         code="INFRASTRUCTURE_INTERRUPTION"))
        led = v03t.load_ledger(lp)
        _expect("RERUN_POLICY_VIOLATION", "fourth attempt",
                lambda: dlt.next_attempt_dlt(led, 0, 101))
        if dlt.expected_next_run_dlt(led) != (0, 101):
            _fail("B23_ORDER", "exhausted run did not block later runs")
        _expect("ORDER_VIOLATION", "run after exhausted",
                lambda: dlt.enforce_order_dlt(led, 2, 101))
        for status in ("completed", "numerical_failure", "failed_validity"):
            lp = tmp / f"term_{status}.json"
            v03t.append_entry(lp, _entry(0, 101, 1, status))
            _expect("RERUN_POLICY_VIOLATION", f"retry after {status}",
                    lambda lp=lp: dlt.next_attempt_dlt(
                        v03t.load_ledger(lp), 0, 101))
        lp = tmp / "all.json"
        for s, a in want:
            v03t.append_entry(lp, _entry(a, s, 1, "completed"))
        led = v03t.load_ledger(lp)
        if dlt.expected_next_run_dlt(led) is not None:
            _fail("B23_ORDER", "six settled runs still report a next run")
        _expect("ORDER_VIOLATION", "seventh run",
                lambda: dlt.enforce_order_dlt(led, 0, 101))
        # trainer-level refusals (before any training)
        ws = _workspace(tmp / "ws", SMALL_SLICES)
        root = _campaign_root(tmp)
        for att in (1, 2, 3):
            v03t.append_entry(root / dlt.LEDGER_NAME, _entry(
                0, SEED, att, "interrupted",
                code="INFRASTRUCTURE_INTERRUPTION"))
        _expect("RERUN_POLICY_VIOLATION", "trainer fourth attempt",
                lambda: _core(ws, root, 0, n_epochs=1,
                              steps_per_epoch=SMALL_STEPS, n_val=1))
        root2 = tmp / "o" / f"{dlt.CAMPAIGN_PREFIX}{_head()[:8]}"
        root2.mkdir(parents=True)
        _expect("ORDER_VIOLATION", "trainer out-of-order arm 2 first",
                lambda: _core(ws, root2, 2, n_epochs=1,
                              steps_per_epoch=SMALL_STEPS, n_val=1))
        logger.info("B2.3 PASS: six-run order; fourth attempt refused; retry "
                    "after completed/numerical_failure/failed_validity "
                    "refused; exhaustion blocks later runs; trainer refusals")
    finally:
        shutil.rmtree(tmp, ignore_errors=False)


# ---------------------------------------------------------------------------
# B2.4 path guard + import origin (190 AM3)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _fake_module(name: str, *, file=None, path=None):
    if name in sys.modules:
        _fail("B24_FAKE_MODULE_CLASH", f"{name} already loaded")
    m = types.ModuleType(name)
    m.__file__ = file
    if path is not None:
        m.__path__ = list(path)
    sys.modules[name] = m
    try:
        yield
    finally:
        del sys.modules[name]


def b24_path_guard() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="v03dlt_b24_"))
    pin = _head()
    try:
        _expect("CAMPAIGN_ROOT_MISSING", "missing root",
                lambda: dlt.check_campaign_root(
                    str(tmp / f"V03DLT_{pin[:8]}"), pin))
        (tmp / "results_other").mkdir()
        _expect("CAMPAIGN_ROOT_INVALID", "wrong name",
                lambda: dlt.check_campaign_root(str(tmp / "results_other"),
                                                pin))
        (tmp / "V03DLT_deadbeef").mkdir()
        _expect("CAMPAIGN_ROOT_INVALID", "other pin prefix",
                lambda: dlt.check_campaign_root(str(tmp / "V03DLT_deadbeef"),
                                                pin))
        for d in dlt.V03_RESULT_DIRNAMES:
            bad = tmp / d / f"V03DLT_{pin[:8]}"
            bad.mkdir(parents=True)
            _expect("V03_EVIDENCE_PATH_REFUSED", f"under {d}",
                    lambda b=bad: dlt.check_campaign_root(str(b), pin))
        root = _campaign_root(tmp)
        f = root / v03t.checkpoint_name(0, SEED, 1, 1)
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
            _fail("B24_OVERWRITE", "orphan file was modified")
        # -- import origin (AM3): fake module objects only, no repo writes ----
        n = dlt.assert_import_origin(REPO)
        pkg = REPO / "seqref_mri"
        cases = (
            ("untracked module file inside seqref_mri",
             dict(file=str(pkg / "scripts" / "fx_untracked_never_tracked.py"))),
            ("module file outside the checkout",
             dict(file=str(tmp / "elsewhere" / "seqref_mri" / "x.py"))),
            ("tracked-named module from another checkout",
             dict(file=str(tmp / "other" / "seqref_mri" / "scripts" /
                           "v03_train.py"))),
            ("namespace dir inside repo but outside seqref_mri",
             dict(path=[str(REPO / "fx_ns_outside")])),
            ("namespace dir in seqref_mri with no tracked files",
             dict(path=[str(pkg / "fx_ns_empty")])),
        )
        for i, (name, kw) in enumerate(cases):
            with _fake_module(f"seqref_mri.fx_origin_case{i}", **kw):
                _expect("IMPORT_ORIGIN_VIOLATION", name,
                        lambda: dlt.assert_import_origin(REPO))
        with _fake_module("seqref_mri.fx_origin_ok",
                          path=[str(pkg / "scripts")]):
            if dlt.assert_import_origin(REPO) != n + 1:
                _fail("B24_IMPORT_ORIGIN",
                      "namespace dir with tracked files not accepted")
        if dlt.assert_import_origin(REPO) != n:
            _fail("B24_IMPORT_ORIGIN", "fake modules not removed")
        logger.info("B2.4 PASS: missing/invalid/V03-evidence roots, existing "
                    "artifacts and orphans refused, nothing overwritten; "
                    "import origin: %d real modules accepted, 5 violations "
                    "refused, tracked namespace dir accepted", n)
    finally:
        shutil.rmtree(tmp, ignore_errors=False)


# ---------------------------------------------------------------------------
# B2.5 two-step smoke through the FIXTURE-ONLY entry point
# ---------------------------------------------------------------------------

def fixture_entry(ws: dict, root: Path, scratch: Path, *, n_epochs: int,
                  steps_per_epoch: int, n_val: int) -> int:
    """Fixture-only entry (B2.5): refuses any real data root, any h5 source
    outside scratch and any output root outside its scratch directory."""
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


def b25_two_step_smoke() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="v03dlt_b25_"))
    try:
        ws = _workspace(tmp, SMALL_SLICES)
        root = _campaign_root(tmp)
        bad_ws = dict(ws, data_root=str(REAL_DATA_ROOT))
        _expect("FIXTURE_REAL_DATA_REFUSED", "real data root",
                lambda: fixture_entry(bad_ws, root, tmp, n_epochs=1,
                                      steps_per_epoch=SMALL_STEPS, n_val=1),
                FixtureFailure)
        outside = Path(tempfile.mkdtemp(prefix="V03DLT_outside_"))
        try:
            _expect("FIXTURE_OUTPUT_OUTSIDE_SCRATCH", "output outside",
                    lambda: fixture_entry(ws, outside, tmp, n_epochs=1,
                                          steps_per_epoch=SMALL_STEPS,
                                          n_val=1), FixtureFailure)
        finally:
            shutil.rmtree(outside, ignore_errors=False)
        if fixture_entry(ws, root, tmp, n_epochs=1,
                         steps_per_epoch=SMALL_STEPS, n_val=1) != 0:
            _fail("B25_SMOKE_FAILED", "two-step smoke returned non-zero")
        rec = _record(root, 0)
        led = v03t.load_ledger(root / dlt.LEDGER_NAME)
        if rec["steps_completed"] != SMALL_STEPS or \
                [e["status"] for e in led["entries"]] != ["completed"]:
            _fail("B25_SMOKE_FAILED", "steps/ledger status unexpected")
        sel = select_checkpoint(rec["validation_history"], rec["checkpoints"])
        if sel["epoch"] != 1:
            _fail("B25_SMOKE_FAILED", f"selection epoch {sel['epoch']}")
        p4 = rec["p4_stats2"]
        env = rec["environment"]
        io_blk = rec["import_origin"]
        if (p4["path"], p4["file_sha256"], p4["semantic_sha256"]) != \
                (ws["p4"], *ws["p4_pins"]) or rec["commit_pin"] != _head() \
                or not env.get("cpu_model") or not isinstance(
                    env.get("threads"), int) or \
                io_blk["modules_checked"] < 10:
            _fail("B27_RECORD_PROVENANCE",
                  f"p4 {p4}, env {env}, import_origin {io_blk}")
        logger.info("B2.5 PASS: two-step smoke via fixture-only entry; record/"
                    "ledger/sidecars written; selection applied; P4 path and "
                    "hashes, pin, CPU/threads and import origin recorded; real "
                    "data and outside roots refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=False)


# ---------------------------------------------------------------------------
# B2.6 mask contract, all 30 epochs
# ---------------------------------------------------------------------------

def b26_mask_contract() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="v03dlt_b26_"))
    try:
        ws = _workspace(tmp, SMALL_SLICES)
        ds = FastMRISliceDataset(ws["data_root"], split="train", mode="train")
        n_center, n_total = mask_counts(96)
        if (n_center, n_total) != (8, 24):
            _fail("B26_MASK_COUNTS", f"mask_counts(96)={n_center, n_total}")
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
                    _fail("B26_MASK_NOT_CANONICAL", f"epoch {k} slice {i}")
                if not np.array_equal(got1, got2):
                    _fail("B26_MASK_NOT_DETERMINISTIC",
                          f"epoch {k} slice {i}")
                if int(got1.sum()) != n_total or not got1[centre].all():
                    _fail("B26_MASK_COUNTS", f"epoch {k} slice {i}")
        logger.info("B2.6 PASS: epochs 1-30 canonical (index k-1), exact "
                    "counts 24/8, deterministic; epochs 1-3 use V03 indices "
                    "0-2 through the same pinned dataset code")
    finally:
        shutil.rmtree(tmp, ignore_errors=False)


# ---------------------------------------------------------------------------
# B2.7 P4 binding
# ---------------------------------------------------------------------------

def b27_p4_binding() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="v03dlt_b27_"))
    try:
        ws = _workspace(tmp, SMALL_SLICES)
        with _p4_pins(ws, trainer_pins=False):
            p4 = ffr.load_p4s2_parent(ws["p4"])
        _expect("P4_PIN_MISMATCH", "loaded P4 vs registered pins",
                lambda: dlt.assert_p4_pins(p4))
        root = _campaign_root(tmp)
        _expect("P4_PIN_MISMATCH", "trainer refuses unregistered P4",
                lambda: _core(ws, root, 0, n_epochs=1,
                              steps_per_epoch=SMALL_STEPS, n_val=1,
                              trainer_pins=False))
        if (root / dlt.LEDGER_NAME).exists() or list(root.iterdir()):
            _fail("B27_WROTE_ON_REFUSAL", "P4 refusal left files")
        Path(ws["p4"]).write_bytes(Path(ws["p4"]).read_bytes() + b" ")
        with _p4_pins(ws, trainer_pins=False):
            _expect("PARENT_FILE_HASH_MISMATCH", "tampered P4 file",
                    lambda: ffr.load_p4s2_parent(ws["p4"]),
                    ffr.FreeFlowError)
        logger.info("B2.7 PASS: P4 hashes asserted against registered pins "
                    "(refusal before any ledger write); tampered file "
                    "refused by the pinned loader; record fields in B2.5")
    finally:
        shutil.rmtree(tmp, ignore_errors=False)


# ---------------------------------------------------------------------------
# B2.8 failure paths (incl. 190 AM2)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _swap(module, name: str, value):
    orig = getattr(module, name)
    setattr(module, name, value)
    try:
        yield orig
    finally:
        setattr(module, name, orig)


def _nll_injected(after: int, action):
    orig = tff.nll_objective
    calls = {"n": 0}

    def wrapped(*a, **k):
        calls["n"] += 1
        out = orig(*a, **k)
        if calls["n"] > after:
            return action(out)
        return out
    return wrapped


def _raise(exc):
    def act(_out):
        raise exc
    return act


def _check_failure(root: Path, name: str, expect_status: str,
                   expect_error: str) -> None:
    rec = _record(root, 0)
    led_full = v03t.load_ledger(root / dlt.LEDGER_NAME)
    led = led_full["entries"]
    if rec["status"] != expect_status or led[-1]["status"] != expect_status \
            or rec["error_code"] != expect_error:
        _fail("B28_STATUS_MISMATCH",
              f"{name}: status {rec['status']}/{led[-1]['status']} error "
              f"{rec['error_code']} != {expect_status}/{expect_error}")
    if expect_status in ("numerical_failure", "failed_validity"):
        _expect("RERUN_POLICY_VIOLATION", f"{name}: no restart",
                lambda: dlt.next_attempt_dlt(led_full, 0, SEED))
    elif dlt.next_attempt_dlt(led_full, 0, SEED) != 2:
        _fail("B28_RETRY_POLICY", f"{name}: retry not permitted")


def _fail_case(name: str, patch, expect_code: str, expect_status: str,
               expect_error: str = None, *, steps_per_epoch: int = None,
               exc_type=v03t.V03TError) -> None:
    spe = SMALL_STEPS if steps_per_epoch is None else steps_per_epoch
    tmp = Path(tempfile.mkdtemp(prefix="v03dlt_b28_"))
    try:
        ws = _workspace(tmp, SMALL_SLICES)
        root = _campaign_root(tmp)
        out = io.StringIO()
        ctx = patch(root) if patch else contextlib.nullcontext()
        with ctx, contextlib.redirect_stdout(out):
            _expect(expect_code, name,
                    lambda: _core(ws, root, 0, n_epochs=1,
                                  steps_per_epoch=spe, n_val=1),
                    exc_type)
        if "PASS" in out.getvalue():
            _fail("B28_PASS_LINE", f"{name}: PASS line printed on failure")
        _check_failure(root, name, expect_status,
                       expect_error or expect_code)
    finally:
        shutil.rmtree(tmp, ignore_errors=False)


def b28_failure_paths() -> None:
    nan = float("nan")
    _fail_case("non-finite loss",
               lambda r: _swap(tff, "nll_objective",
                               _nll_injected(0, lambda o: o * nan)),
               "NUMERICAL_FAILURE", "numerical_failure")
    _fail_case("non-finite gradient",
               lambda r: _swap(dlt, "global_grad_norm",
                               lambda m: float("inf")),
               "NUMERICAL_FAILURE", "numerical_failure")
    _fail_case("objective U_NON_FINITE",
               lambda r: _swap(tff, "nll_objective", _nll_injected(
                   0, _raise(v03l.V03LError("U_NON_FINITE", "injected")))),
               "U_NON_FINITE", "numerical_failure")
    for code in ("LOSS_CONTRACT_VIOLATION", "MAP_BATCH_MISMATCH"):
        _fail_case(f"objective {code}",
                   lambda r, c=code: _swap(tff, "nll_objective", _nll_injected(
                       0, _raise(v03l.V03LError(c, "injected")))),
                   code, "failed_validity")

    def no_grad(_m):
        raise V03PError("PROBE_VALIDITY_FAILURE", "injected: no gradients")
    _fail_case("gradient coverage", lambda r: _swap(
        dlt, "global_grad_norm", no_grad), "PROBE_VALIDITY_FAILURE",
        "failed_validity")
    # --- 190 AM2 ------------------------------------------------------------
    _fail_case("STEP_COUNT_MISMATCH (one batch fewer than the contract)",
               None, "STEP_COUNT_MISMATCH", "failed_validity",
               steps_per_epoch=SMALL_STEPS + 1)

    def plant(root):
        ck = root / v03t.checkpoint_name(0, SEED, 1, 1)

        def act(out):
            if not ck.exists():
                ck.write_bytes(b"planted during training\n")
            return out
        return _swap(tff, "nll_objective", _nll_injected(0, act))
    _fail_case("ARTIFACT_EXISTS during training", plant, "ARTIFACT_EXISTS",
               "failed_validity")
    orig_ident = dlt.assert_checkpoint_identity

    def bad_identity(entry, *a):
        return orig_ident(dict(entry, seed=999), *a)
    _fail_case("CHECKPOINT_IDENTITY_MISMATCH", lambda r: _swap(
        dlt, "assert_checkpoint_identity", bad_identity),
        "CHECKPOINT_IDENTITY_MISMATCH", "failed_validity")
    tmp = Path(tempfile.mkdtemp(prefix="v03dlt_b28b_"))
    try:
        ws = _workspace(tmp, SMALL_SLICES)
        root = _campaign_root(tmp)
        _expect("BUDGET_MISMATCH", "inconsistent budget",
                lambda: _core(ws, root, 0, n_epochs=1,
                              steps_per_epoch=SMALL_STEPS, n_val=1,
                              total=SMALL_STEPS + 1))
        if list(root.iterdir()):
            _fail("B28_WROTE_ON_REFUSAL", "budget refusal left files")
    finally:
        shutil.rmtree(tmp, ignore_errors=False)
    # --- infrastructure ------------------------------------------------------
    _fail_case("injected exception",
               lambda r: _swap(tff, "nll_objective", _nll_injected(
                   0, _raise(RuntimeError("injected")))),
               "INFRASTRUCTURE_INTERRUPTION", "interrupted", "RuntimeError")
    _fail_case("KeyboardInterrupt in training",
               lambda r: _swap(tff, "nll_objective", _nll_injected(
                   1, _raise(KeyboardInterrupt()))),
               "KeyboardInterrupt", "interrupted", "KEYBOARD_INTERRUPT",
               exc_type=KeyboardInterrupt)
    logger.info("B2.8 PASS: non-finite loss/gradient and U_NON_FINITE -> "
                "numerical_failure; structural objective codes and gradient "
                "coverage -> failed_validity; AM2 STEP_COUNT_MISMATCH, "
                "ARTIFACT_EXISTS, CHECKPOINT_IDENTITY_MISMATCH -> "
                "failed_validity, retry refused; inconsistent budget refused "
                "with no ledger entry; injected exception -> interrupted with "
                "retry; KeyboardInterrupt -> evidence, no PASS line")


# ---------------------------------------------------------------------------
# B2.9 mode, partial batch, V03 regression selftests
# ---------------------------------------------------------------------------

def b29_mode_partial_regression() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="v03dlt_b29_"))
    try:
        ws = _workspace(tmp, SMALL_SLICES)
        root = _campaign_root(tmp)
        seen = []
        orig_val = dlt._validate

        def val(model, *a, **k):
            seen.append(("val_in", model.training))
            out = orig_val(model, *a, **k)
            seen.append(("val_out", model.training))
            return out
        with _swap(dlt, "_validate", val):
            if _core(ws, root, 0, n_epochs=1, steps_per_epoch=SMALL_STEPS,
                     n_val=1) != 0:
                _fail("B29_RUN_FAILED", "mode run returned non-zero")
        if seen != [("val_in", True), ("val_out", True)]:
            _fail("B29_MODE_NOT_RESTORED", f"mode trace {seen}")
        rec = _record(root, 0)
        if rec["steps_completed"] != -(-SMALL_SLICES // v02m.BATCH_SIZE):
            _fail("B29_PARTIAL_BATCH_DROPPED",
                  f"{rec['steps_completed']} steps for {SMALL_SLICES} slices")
    finally:
        shutil.rmtree(tmp, ignore_errors=False)
    for mod in ("seqref_mri.scripts.v03_train", "seqref_mri.scripts.v03_probe",
                "seqref_mri.scripts.v03_eval"):
        res = subprocess.run([sys.executable, "-m", mod, "--mode", "selftest"],
                             cwd=REPO, capture_output=True, text=True,
                             check=False)
        if res.returncode != 0 or "PASS" not in (res.stdout + res.stderr):
            _fail("B29_REGRESSION_FAILED",
                  f"{mod} selftest rc {res.returncode}; tail: "
                  f"{(res.stdout + res.stderr)[-1500:]}")
        logger.info("B2.9 regression PASS: %s selftest", mod)
    logger.info("B2.9 PASS: train mode restored after validation; final "
                "partial batch counted (%d slices -> %d steps); V03T, V03P, "
                "V03E selftests pass", SMALL_SLICES, SMALL_STEPS)


# ---------------------------------------------------------------------------
# B2.10 launcher fixture (190 AM4): copies differ only in the trainer line
# and the marked HOST CONSTANTS block.
# ---------------------------------------------------------------------------

LAUNCHER = REPO / "seqref_mri" / "scripts" / "v03dlt_launch.sh"
STUB = REPO / "seqref_mri" / "scripts" / "v03dlt_stub_train.py"
TRAINER_LINE = 'TRAINER_CMD=("$PY" -m seqref_mri.scripts.v03dlt_train)'
BLOCK_BEGIN = "# >>> HOST CONSTANTS"
BLOCK_END = "# <<< HOST CONSTANTS"
B210_TIMEOUT = 300
FIXTURE_CKPT_BYTES = 4096


def _block_bounds(lines: list) -> tuple:
    b = [i for i, ln in enumerate(lines) if ln.startswith(BLOCK_BEGIN)]
    e = [i for i, ln in enumerate(lines) if ln.startswith(BLOCK_END)]
    if len(b) != 1 or len(e) != 1 or not b[0] < e[0]:
        _fail("B210_BLOCK_MARKERS", f"HOST CONSTANTS markers {b} / {e}")
    return b[0], e[0]


def _evidence(scratch: Path) -> dict:
    """Isolated synthetic stand-ins for the host evidence (AM4)."""
    res = scratch / "evidence"
    out = {"RES": res}
    for rel, key in (("_diag/p4/scaling_statistics.json", "P4"),
                     ("v03_probe/v03_probe_record.json", "PROBE"),
                     ("v03_gate/v03_gate_record.json", "GATE")):
        p = res / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = v02m.canonical_json({"synthetic": key})
        p.write_bytes(payload)
        Path(str(p) + ".sha256").write_text(
            f"{_sha_bytes(payload)}  {p.name}\n")
        out[key] = _sha_bytes(payload)
    ev = []
    for name in ["v03_run_ledger.json"] + [
            v03t.run_record_name(a, s, 1) for a in (0, 2) for s in v03t.SEEDS]:
        p = res / "v03_train" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = v02m.canonical_json({"synthetic": name})
        p.write_bytes(payload)
        ev.append(f'  "v03_train/{name} {_sha_bytes(payload)}"')
    out["V03_EVIDENCE"] = ev
    return out


def _exclude_sha() -> str:
    p = Path(_git("rev-parse", "--git-path", "info/exclude"))
    p = p if p.is_absolute() else REPO / p
    if not p.is_file():
        _fail("B210_EXCLUDE_MISSING", f"{p} missing")
    return v03t._file_sha256(p)


def _fixture_block(src_block: list, scratch: Path, py: str) -> list:
    ev = _evidence(scratch)
    mods = [ln for ln in src_block if ln.startswith('  "seqref_mri/')]
    if len(mods) != 10:
        _fail("B210_BLOCK_PARSE", f"{len(mods)} registered-module lines")
    branch = _git("branch", "--show-current")
    lines = [
        f'readonly PY="{py}"',
        f'readonly RES="{ev["RES"]}"',
        'readonly P4="$RES/_diag/p4/scaling_statistics.json"',
        f'readonly P4_SHA="{ev["P4"]}"',
        'readonly PROBE="$RES/v03_probe/v03_probe_record.json"',
        f'readonly PROBE_SHA="{ev["PROBE"]}"',
        'readonly GATE="$RES/v03_gate/v03_gate_record.json"',
        f'readonly GATE_SHA="{ev["GATE"]}"',
        f'readonly BRANCH="{branch}"',
        f'readonly EXCLUDE_SHA="{_exclude_sha()}"',
        "readonly -a V03_EVIDENCE=(", *ev["V03_EVIDENCE"], ")",
        "readonly -a REGISTERED_MODULES=(", *[m.rstrip("\n") for m in mods], ")",
        f"readonly CKPT_BYTES={FIXTURE_CKPT_BYTES}",
        "readonly INHIBIT_CMD=(sleep infinity)   # fixture stand-in (AM4)",
    ]
    return [ln + "\n" for ln in lines]


def _launcher_copy(scratch: Path, trainer: str = None, py: str = None,
                   host: bool = False) -> Path:
    """AM4: copy of the production launcher; only the trainer line and (in
    isolated mode) the HOST CONSTANTS block interior may differ -- asserted."""
    src = LAUNCHER.read_text().splitlines(keepends=True)
    hits = [i for i, ln in enumerate(src) if ln.rstrip("\n") == TRAINER_LINE]
    if len(hits) != 1:
        _fail("B210_TRAINER_LINE", f"trainer line found {len(hits)} times")
    b, e = _block_bounds(src)
    if not e < hits[0]:
        _fail("B210_TRAINER_LINE", "trainer line inside the constants block")
    new_trainer = (f'TRAINER_CMD=({trainer})\n' if trainer
                   else f'TRAINER_CMD=("$PY" "{STUB}")\n')
    if host:
        if py:
            _fail("B210_HOST_PY", "host mode keeps the production block")
        block = src[b + 1:e]
    else:
        block = _fixture_block(src[b + 1:e], scratch, py or sys.executable)
    dst = src[:b + 1] + block + src[e:]
    off = len(block) - (e - b - 1)
    dst[hits[0] + off] = new_trainer
    # assert: outside the block interior and the trainer line, identical
    head_ok = dst[:b + 1] == src[:b + 1]
    tail_src = src[e:]
    tail_dst = dst[b + 1 + len(block):]
    k = hits[0] - e
    tail_ok = len(tail_src) == len(tail_dst) and all(
        x == y for i, (x, y) in enumerate(zip(tail_src, tail_dst)) if i != k)
    if not (head_ok and tail_ok):
        _fail("B210_COPY_DIFFERS",
              "launcher copy differs outside the trainer line and the "
              "HOST CONSTANTS block")
    out = scratch / "v03dlt_launch_copy.sh"
    out.write_text("".join(dst))
    out.chmod(0o755)
    return out


class _Campaign:
    def __init__(self, scratch: Path, name: str):
        self.scratch = scratch
        self.results = scratch / f"results_{name}"
        self.results.mkdir()
        self.pin = _head()
        self.root = self.results / f"{dlt.CAMPAIGN_PREFIX}{self.pin[:8]}"
        self.data = scratch / "data"
        self.data.mkdir(exist_ok=True)

    def args(self, mode: str) -> list:
        return [str(self.scratch / "v03dlt_launch_copy.sh"), "--mode", mode,
                "--commit-pin", self.pin, "--repo", str(REPO),
                "--data-root", str(self.data),
                "--results-root", str(self.results)]


class _Done:
    def __init__(self, returncode, stdout, stderr):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _env(tmux: bool = True) -> dict:
    env = dict(os.environ)
    env.pop("TMUX", None)
    if tmux:
        env["TMUX"] = "/tmp/v03dlt-fixture-tmux,0,0"
    return env


def _run_launcher(c: _Campaign, mode: str, *, tmux: bool = True):
    return subprocess.run(c.args(mode), env=_env(tmux), capture_output=True,
                          text=True, timeout=B210_TIMEOUT, check=False)


def _start_launcher(c: _Campaign, mode: str):
    return subprocess.Popen(c.args(mode), env=_env(True),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)


def _plan(scratch: Path, plan: dict) -> None:
    (scratch / "stub_plan.json").write_text(json.dumps(plan))


def _tree(root: Path, skip=(".launch.lock",)) -> dict:
    return {p.name: v03t._file_sha256(p) for p in sorted(root.iterdir())
            if p.is_file() and p.name not in skip}


def _wait_for(path: Path, what: str, timeout: float = 120.0) -> None:
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > timeout:
            _fail("B210_TIMEOUT", f"waiting for {what}")
        time.sleep(0.05)


def _expect_rc(res, rc: int, code: str, name: str) -> None:
    text = (res.stdout or "") + (res.stderr or "")
    if res.returncode != rc or (code and code not in text):
        _fail("B210_UNEXPECTED_OUTCOME",
              f"{name}: rc {res.returncode} (want {rc}), want code {code}; "
              f"tail: {text[-1500:]}")
    logger.info("B2.10 branch PASS: %s -> rc %d %s", name, rc, code)


def _sessions(root: Path) -> list:
    return sorted(p.name[len("launcher_session_"):-len(".log")]
                  for p in root.glob("launcher_session_*.log"))


def _check_sha_files(root: Path) -> None:
    for side in sorted(root.glob("*.sha256")):
        res = subprocess.run(["sha256sum", "-c", "--quiet", side.name],
                             cwd=root, capture_output=True, text=True,
                             check=False)
        if res.returncode != 0:
            _fail("B210_SHA_CHECK_FAILED", f"{side.name}: {res.stdout}")


def _ledger_prefix_ok(root: Path) -> None:
    live = v03t.load_ledger(root / dlt.LEDGER_NAME)
    meta = {k: v for k, v in live.items() if k != "entries"}
    for snap in sorted(root.glob("ledger_snapshot_*.json")):
        s = json.loads(snap.read_text())
        side = (root / (snap.name + ".sha256")).read_text().split()
        if side[1] != snap.name:
            _fail("B210_SNAPSHOT_SIDECAR", f"{snap.name} sidecar names {side}")
        n = len(s["entries"])
        if s["entries"] != live["entries"][:n] or \
                {k: v for k, v in s.items() if k != "entries"} != meta:
            _fail("B210_PREFIX", f"{snap.name} not a prefix")


def _kill_inhibitor(root: Path, session: str) -> None:
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
            logger.info("pid %s is not an inhibitor (%r); left alone", pid,
                        cmd[:80])
            return
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            logger.info("inhibitor pid %s already gone", pid)


def _kill_all_inhibitors(scratch: Path) -> None:
    for log in scratch.rglob("launcher_session_*.log"):
        _kill_inhibitor(log.parent, log.name[len("launcher_session_"):-4])


def _kill_stubs() -> None:
    res = subprocess.run(["pgrep", "-f", str(STUB)], capture_output=True,
                         text=True, check=False)
    for pid in res.stdout.split():
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            logger.info("stub pid %s already gone", pid)


def _lock_holders(lock: Path) -> list:
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
    t0 = time.time()
    while subprocess.run(["flock", "-n", str(lock), "true"],
                         check=False).returncode != 0:
        if time.time() - t0 > timeout:
            _fail("B210_LOCK_NOT_RELEASED",
                  f"lock still held by {_lock_holders(lock)}")
        time.sleep(0.1)


def _new_scratch(prefix: str) -> Path:
    s = Path(tempfile.mkdtemp(prefix=prefix))
    (s / "V03DLT_STUB_SCRATCH").write_text("fixture scratch\n")
    _plan(s, {"default": ["succeed", 0]})
    return s


def b210_launcher() -> None:
    scratch = _new_scratch("v03dlt_b210_")
    try:
        _launcher_copy(scratch)
        # -- modes and tmux ---------------------------------------------------
        a = _Campaign(scratch, "a")
        _expect_rc(_run_launcher(a, "new", tmux=False), 2, "NOT_IN_TMUX",
                   "outside tmux")
        if a.root.exists():
            _fail("B210_ROOT_CREATED", "root created despite NOT_IN_TMUX")
        _expect_rc(_run_launcher(a, "resume"), 2, "CAMPAIGN_ROOT_MISSING",
                   "resume without root")
        # -- session 1: run 1 then STOP (current run finishes) -----------------
        _plan(scratch, {"0_101_1": ["succeed", 3], "default": ["succeed", 0]})
        p = _start_launcher(a, "new")
        _wait_for(scratch / "stub_started_0_101_1.txt", "session-1 stub")
        note = b"pause for daytime use\n"
        (a.root / "STOP").write_bytes(note)
        o, e = p.communicate(timeout=B210_TIMEOUT)
        if p.returncode != 0 or "STOP_REQUESTED" not in e:
            _fail("B210_STOP", f"session 1 rc {p.returncode}: {e[-1500:]}")
        if "campaign disk allowance PASS for 6 unsettled run(s)" not in e:
            _fail("B210_DISK", "campaign allowance for 6 runs not logged")
        s1 = _sessions(a.root)
        disp = (scratch / "stub_sigdisp_0_101_1.txt").read_text().strip()
        if disp not in ("SIG_IGN", "SIG_DFL", "default_int_handler"):
            _fail("B210_PRECONDITION", f"stub SIGINT disposition {disp!r}")
        led = v03t.load_ledger(a.root / dlt.LEDGER_NAME)["entries"]
        if [(x["arm"], x["seed"], x["status"]) for x in led] != \
                [(0, 101, "completed")]:
            _fail("B210_STOP", f"session 1 ledger {led}")
        if (a.root / f"stop_request_{s1[0]}.txt").read_bytes() != note or \
                "stop_sha256=" not in (a.root /
                                       f"stop_seen_{s1[0]}.txt").read_text():
            _fail("B210_STOP", "STOP bytes/metadata not preserved separately")
        man1 = (a.root / f"campaign_manifest_{s1[0]}.sha256").read_text()
        if f"stop_request_{s1[0]}.txt" not in man1 or \
                dlt.LEDGER_NAME in man1:
            _fail("B210_MANIFEST", "session 1 manifest content")
        _check_sha_files(a.root)
        _ledger_prefix_ok(a.root)
        # -- resume refused while STOP present; tree unchanged ------------------
        before = _tree(a.root)
        _expect_rc(_run_launcher(a, "resume"), 2, "STOP_PRESENT",
                   "resume with STOP")
        if _tree(a.root) != before:
            _fail("B210_VERIFY_WROTE", "failed verification changed files")
        # -- ack-stop: edited STOP refused; exact STOP acknowledged -------------
        (a.root / "STOP").write_bytes(b"edited\n")
        _expect_rc(_run_launcher(a, "ack-stop"), 2, "STOP_CHANGED",
                   "ack-stop edited STOP")
        if not (a.root / "STOP").exists():
            _fail("B210_ACK", "edited STOP was renamed")
        (a.root / "STOP").write_bytes(note)
        _expect_rc(_run_launcher(a, "ack-stop"), 0, "ACK_STOP_PASS",
                   "ack-stop")
        if (a.root / "STOP").exists() or \
                not list(a.root.glob("stop_acknowledged_*.txt")):
            _fail("B210_ACK", "STOP not renamed on acknowledgement")
        # -- session 2: resume, run 2 (failed_validity settles it), STOP --------
        time.sleep(1.1)
        before = _tree(a.root)
        _plan(scratch, {"2_101_1": ["validity"], "0_202_1": ["succeed", 3],
                        "default": ["succeed", 0]})
        p = _start_launcher(a, "resume")
        _wait_for(scratch / "stub_started_0_202_1.txt", "session-2 stub")
        (a.root / "STOP").write_bytes(note)
        o, e = p.communicate(timeout=B210_TIMEOUT)
        if p.returncode != 0 or "STOP_REQUESTED" not in e or \
                "campaign disk allowance PASS for 5 unsettled run(s)" not in e \
                or "pair arm 2 seed 101 settled: failed_validity" not in e:
            _fail("B210_SESSION2", f"rc {p.returncode}: {e[-1500:]}")
        after = _tree(a.root)
        for name, h in before.items():
            if name in (dlt.LEDGER_NAME, dlt.LEDGER_NAME + ".sha256"):
                continue
            if after.get(name) != h:
                _fail("B210_RESUME_MODIFIED", f"{name} changed or removed")
        if len(_sessions(a.root)) != 2:
            _fail("B210_SESSION2", f"sessions {_sessions(a.root)}")
        _check_sha_files(a.root)
        _ledger_prefix_ok(a.root)
        _expect_rc(_run_launcher(a, "ack-stop"), 0, "ACK_STOP_PASS",
                   "ack-stop 2")
        # -- tampered earlier entry / metadata detected (scratch copies) --------
        for tag, mutate in (("entry", lambda L: L["entries"][0].update(
                                {"steps_completed": 999})),
                            ("meta", lambda L: L.update({"extra": 1}))):
            t = _Campaign(scratch, f"tamper_{tag}")
            shutil.copytree(a.root, t.root)
            lp = t.root / dlt.LEDGER_NAME
            L = json.loads(lp.read_text())
            mutate(L)
            v03t._write_with_sidecar(lp, v02m.canonical_json(L))
            _expect_rc(_run_launcher(t, "resume"), 2,
                       "LEDGER_PREFIX_MISMATCH", f"tampered {tag}")
        # -- orphan detection (scratch copy): nothing deleted ---------------------
        o_c = _Campaign(scratch, "orphan")
        shutil.copytree(a.root, o_c.root)
        (o_c.root / v03t.checkpoint_name(2, 202, 1, 1)).write_bytes(b"orphan")
        before = _tree(o_c.root)
        _expect_rc(_run_launcher(o_c, "resume"), 2,
                   "ORPHANED_ATTEMPT_ARTIFACTS", "orphan")
        if _tree(o_c.root) != before:
            _fail("B210_VERIFY_WROTE", "orphan verification changed files")
        # -- cancellation by SIGTERM; lock held meanwhile -------------------------
        time.sleep(1.1)
        _plan(scratch, {"2_202_1": ["sleep"], "default": ["succeed", 0]})
        p = _start_launcher(a, "resume")
        _wait_for(scratch / "stub_started_2_202_1.txt", "cancel stub")
        held = _tree(a.root)
        _expect_rc(_run_launcher(a, "resume"), 2, "LOCK_HELD",
                   "second resume")
        _expect_rc(_run_launcher(a, "ack-stop"), 2, "LOCK_HELD",
                   "ack-stop while held")
        _expect_rc(_run_launcher(a, "new"), 2, "LOCK_HELD", "second new")
        if _tree(a.root) != held:
            _fail("B210_LOCK_WROTE", "refused instances changed files")
        os.kill(p.pid, signal.SIGTERM)
        o, e = p.communicate(timeout=B210_TIMEOUT)
        if p.returncode != 2 or "CANCELLED" not in e:
            _fail("B210_CANCEL", f"SIGTERM rc {p.returncode}: {e[-1500:]}")
        if (scratch / "stub_sigint_2_202_1.txt").read_text().strip() != "1" \
                or (scratch / "stub_finally_2_202_1.txt").read_text() \
                .strip() != "interrupted":
            _fail("B210_CANCEL", "stub did not receive exactly one SIGINT")
        led = v03t.load_ledger(a.root / dlt.LEDGER_NAME)["entries"]
        if (led[-1]["status"], led[-1]["error_code"]) != \
                ("interrupted", "KEYBOARD_INTERRUPT"):
            _fail("B210_CANCEL", f"ledger {led[-1]}")
        _check_sha_files(a.root)
        # -- cancellation by SIGINT to the launcher's process group ---------------
        g = _Campaign(scratch, "group")
        _plan(scratch, {"0_101_1": ["sleep"], "default": ["succeed", 0]})
        for f in scratch.glob("stub_*_0_101_1.txt"):
            f.rename(f.with_suffix(".txt.prev"))
        p = _start_launcher(g, "new")
        _wait_for(scratch / "stub_started_0_101_1.txt", "group stub")
        os.killpg(os.getpgid(p.pid), signal.SIGINT)
        o, e = p.communicate(timeout=B210_TIMEOUT)
        if p.returncode != 2 or "CANCELLED" not in e or \
                (scratch / "stub_sigint_0_101_1.txt").read_text().strip() \
                != "1":
            _fail("B210_CANCEL_GROUP", f"rc {p.returncode}: {e[-1500:]}")
        # -- trainer exits without the expected ledger entry ----------------------
        n = _Campaign(scratch, "noentry")
        for f in scratch.glob("stub_*_0_101_1.txt*"):
            f.unlink()
        _plan(scratch, {"0_101_1": ["no_entry_on_sigint"],
                        "default": ["succeed", 0]})
        p = _start_launcher(n, "new")
        _wait_for(scratch / "stub_started_0_101_1.txt", "no-entry stub")
        os.kill(p.pid, signal.SIGTERM)
        o, e = p.communicate(timeout=B210_TIMEOUT)
        if p.returncode != 2 or "CANCEL_LEDGER_MISMATCH" not in e:
            _fail("B210_CANCEL_MISMATCH", f"rc {p.returncode}: {e[-1500:]}")
        # -- hard kill: no manifest -> next resume stops; nothing deleted ---------
        time.sleep(1.1)
        for f in scratch.glob("stub_*_2_202_2.txt"):
            f.unlink()
        _plan(scratch, {"2_202_2": ["sleep"], "default": ["succeed", 0]})
        p = _start_launcher(a, "resume")
        _wait_for(scratch / "stub_started_2_202_2.txt", "hard-kill stub")
        killed = _sessions(a.root)[-1]
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        p.communicate(timeout=B210_TIMEOUT)
        _kill_stubs()
        _kill_inhibitor(a.root, killed)
        t0 = time.time()
        while subprocess.run(["pgrep", "-f", str(STUB)], capture_output=True,
                             check=False).returncode == 0:
            if time.time() - t0 > 30:
                _fail("B210_TIMEOUT", "killed stub still running")
            time.sleep(0.05)
        _wait_lock_free(a.root / ".launch.lock")
        if (a.root / f"campaign_manifest_{killed}.sha256").exists():
            _fail("B210_HARD_KILL", "manifest written despite SIGKILL")
        before = _tree(a.root)
        _expect_rc(_run_launcher(a, "resume"), 2, "MISSING_SESSION_MANIFEST",
                   "resume after hard kill")
        if _tree(a.root) != before:
            _fail("B210_VERIFY_WROTE", "hard-kill verification changed files")
        logger.info("B2.10 PASS: AM4 launcher copy; tmux/new/resume/ack-stop; "
                    "campaign disk allowance 6 then 5; failed_validity settles "
                    "a run; STOP preserved by hash; resume verification writes "
                    "nothing; tamper, orphan, hard kill detected; LOCK_HELD x3;"
                    " SIGTERM and group SIGINT forward exactly one SIGINT; "
                    "CANCEL_LEDGER_MISMATCH")
    finally:
        _kill_stubs()
        _kill_all_inhibitors(scratch)
        shutil.rmtree(scratch, ignore_errors=False)


RCWRAP_SRC = '''#!/usr/bin/env bash
# fixture-written wrapper (V03DLTF B2.10): runs the stub, then forces rc {rc}
"{py}" "{stub}" "$@"
exit {rc}
'''


def b210_exit_and_disk() -> None:
    """Review fixes R1/R2: campaign disk allowance and exit-code/ledger
    agreement."""
    base = Path(tempfile.mkdtemp(prefix="v03dlt_b210x_"))
    try:
        for name, plan, rc, code in (
                ("completed but rc 3", {"default": ["succeed", 0]}, 3,
                 "TRAINER_EXIT_MISMATCH"),
                ("numerical_failure but rc 0", {"default": ["numerical"]}, 0,
                 "TRAINER_EXIT_MISMATCH")):
            s = base / f"x{rc}"
            s.mkdir()
            (s / "V03DLT_STUB_SCRATCH").write_text("fixture scratch\n")
            _plan(s, plan)
            w = s / "rcwrap.sh"
            w.write_text(RCWRAP_SRC.format(py=sys.executable, stub=STUB, rc=rc))
            w.chmod(0o755)
            _launcher_copy(s, trainer=f'bash "{w}"')
            c = _Campaign(s, "x")
            _expect_rc(_run_launcher(c, "new"), 2, code, name)
            _kill_all_inhibitors(s)
        # campaign allowance: an impossible requirement is refused at start
        s = base / "disk"
        s.mkdir()
        (s / "V03DLT_STUB_SCRATCH").write_text("fixture scratch\n")
        _plan(s, {"default": ["succeed", 0]})
        cp = _launcher_copy(s)
        txt = cp.read_text().replace(
            f"readonly CKPT_BYTES={FIXTURE_CKPT_BYTES}\n",
            "readonly CKPT_BYTES=99999999999999\n", 1)
        cp.write_text(txt)
        c = _Campaign(s, "d")
        res = _run_launcher(c, "new")
        _expect_rc(res, 2, "DISK_SPACE_SHORT", "campaign allowance short")
        if "(campaign allowance)" not in res.stderr or \
                list(s.glob("stub_started_*")):
            _fail("B210_DISK", "allowance not checked before the first run")
        _kill_all_inhibitors(s)
        logger.info("B2.10 PASS: TRAINER_EXIT_MISMATCH both directions; "
                    "campaign disk allowance refused before any run")
    finally:
        _kill_stubs()
        _kill_all_inhibitors(base)
        shutil.rmtree(base, ignore_errors=False)


SHIM_SRC = '''# fixture-written shim (V03DLTF B2.10): REAL v03dlt_train.main(), _run replaced
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
from seqref_mri.scripts import v03dlt_train as t
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
sys.argv = ["v03dlt_train"] + ARGS
sys.exit(t.main())
'''

PYWRAP_SRC = '''#!/usr/bin/env bash
# fixture-written PY wrapper (V03DLTF B2.10): holds the launcher in preflight
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
    (s / "V03DLT_STUB_SCRATCH").write_text("fixture scratch\n")
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
    if path.is_file():
        try:
            os.kill(int(path.read_text().strip()), signal.SIGKILL)
        except ProcessLookupError:
            logger.info("pid in %s already gone", path.name)


def b210_signals() -> None:
    base = Path(tempfile.mkdtemp(prefix="v03dlt_b210s_"))
    try:
        s, c = _shim_case(base, "real", {"pre_main_delay": 0.0})
        p = _start_launcher(c, "new")
        _wait_for(s / "shim_in_run.txt", "real-main shim inside _run")
        os.kill(p.pid, signal.SIGTERM)
        o, e = p.communicate(timeout=B210_TIMEOUT)
        if (s / "shim_in_run.txt").read_text().strip() != "True" or \
                not (s / "shim_ki.txt").is_file() or _forward_count(e) != 1:
            _fail("B210_REAL_SIGINT", f"real main did not receive exactly one "
                  f"SIGINT as KeyboardInterrupt; tail: {e[-1500:]}")
        _expect_rc(_Done(p.returncode, o, e), 2, "CANCEL_LEDGER_MISMATCH",
                   "real main SIGINT (stand-in _run writes no ledger entry)")
        s, c = _shim_case(base, "hazard", {"pre_main_delay": 0.0,
                                           "force_ignore": True})
        p = _start_launcher(c, "new")
        _wait_for(s / "shim_in_run.txt", "hazard shim inside _run")
        os.kill(p.pid, signal.SIGTERM)
        o, e = p.communicate(timeout=B210_TIMEOUT)
        if (s / "shim_in_run.txt").read_text().strip() != "True" or \
                not (s / "shim_ki.txt").is_file() or _forward_count(e) != 1:
            _fail("B210_HAZARD_SIGINT", f"fix did not restore SIGINT under "
                  f"inherited SIG_IGN; tail: {e[-1500:]}")
        _expect_rc(_Done(p.returncode, o, e), 2, "CANCEL_LEDGER_MISMATCH",
                   "inherited SIG_IGN: install_sigint restores one SIGINT")
        s, c = _shim_case(base, "early", {"pre_main_delay": 3.0,
                                          "force_ignore": True})
        p = _start_launcher(c, "new")
        _wait_for(s / "shim_ready.txt", "early shim started (SIGINT ignored)")
        os.kill(p.pid, signal.SIGTERM)
        o, e = p.communicate(timeout=B210_TIMEOUT)
        if "cancellation pending" not in e or _forward_count(e) != 1 or \
                not (s / "shim_ki.txt").is_file():
            _fail("B210_EARLY_CANCEL", f"pending forward failed; tail: "
                  f"{e[-1500:]}")
        _expect_rc(_Done(p.returncode, o, e), 2, "CANCEL_LEDGER_MISMATCH",
                   "early cancel forwarded after handler install")
        s, c = _shim_case(base, "control", {"pre_main_delay": 0.0,
                                            "force_ignore": True,
                                            "disable_install": True})
        p = _start_launcher(c, "new")
        _wait_for(s / "shim_in_run.txt", "control shim inside _run")
        os.kill(int((s / "shim_pid.txt").read_text()), signal.SIGINT)
        time.sleep(1.5)
        if (s / "shim_ki.txt").exists() or \
                (s / "shim_in_run.txt").read_text().strip() != "False":
            _fail("B210_CONTROL", "SIGINT was not ignored without the fix; "
                  "the control does not demonstrate the hazard")
        _kill_pidfile(s / "shim_pid.txt")
        o, e = p.communicate(timeout=B210_TIMEOUT)
        _expect_rc(_Done(p.returncode, o, e), 2, "TRAINER_NO_LEDGER_ENTRY",
                   "control: inherited SIG_IGN swallows SIGINT")
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
                os.kill(p.pid, signal.SIGTERM)
            else:
                (c.root / "STOP").write_bytes(b"stop during preflight\n")
            time.sleep(0.5)
            (s / "preflight_release").write_text("go\n")
            o, e = p.communicate(timeout=B210_TIMEOUT)
            started = list(s.glob("stub_started_*.txt"))
            runlogs = list(c.root.glob("runlog_*"))
            if started or runlogs or note not in e:
                _fail("B210_BOUNDARY", f"{name}: trainer started={started} "
                      f"runlogs={runlogs}; tail: {e[-1500:]}")
            _expect_rc(_Done(p.returncode, o, e), rc, code,
                       f"{name}: request during preflight starts no run")
        logger.info("B2.10 signal PASS: real main() via launcher spawn path "
                    "gets exactly one SIGINT; inherited SIG_IGN restored; "
                    "pending cancellation forwarded; control shows the hazard; "
                    "CANCEL/STOP during preflight start no run")
    finally:
        for f in base.glob("*/shim_pid.txt"):
            _kill_pidfile(f)
        _kill_stubs()
        _kill_all_inhibitors(base)
        shutil.rmtree(base, ignore_errors=False)


def b210_handoff() -> None:
    """A campaign produced by the launcher copy (stub 'full' mode, valid
    30-epoch metadata, two sessions with STOP/ack-stop) passed directly to
    the summary."""
    scratch = _new_scratch("v03dlt_b210h_")
    try:
        _launcher_copy(scratch)
        _plan(scratch, {"0_101_1": ["full", 3], "default": ["full", 0]})
        c = _Campaign(scratch, "h")
        p = _start_launcher(c, "new")
        _wait_for(scratch / "stub_started_0_101_1.txt", "handoff session 1")
        (c.root / "STOP").write_bytes(b"pause\n")
        o, e = p.communicate(timeout=B210_TIMEOUT)
        _expect_rc(_Done(p.returncode, o, e), 0, "STOP_REQUESTED",
                   "handoff session 1")
        _expect_rc(_run_launcher(c, "ack-stop"), 0, "ACK_STOP_PASS",
                   "handoff ack-stop")
        time.sleep(1.1)
        _expect_rc(_run_launcher(c, "resume"), 0, "CAMPAIGN_COMPLETE",
                   "handoff session 2")
        if len(_sessions(c.root)) != 2 or \
                len(list(c.root.glob("campaign_manifest_*.sha256.sha256"))) != 2:
            _fail("B210_HANDOFF", "expected 2 sessions with manifest sidecars")
        ref = _b211_v03_ref(scratch / "v03ref")
        with _patched(dls, **_v03_pins(ref)):
            neg = scratch / "neg" / c.root.name
            shutil.copytree(c.root, neg)
            if dls.build(c.root, ref) != 0:
                _fail("B210_HANDOFF", "summary rc != 0")
            sm = _summary_ok(c.root)
            if sm["counts"] != {"n_run": 6, "n_arm0": 3, "n_arm2": 3,
                                "n_pair": 3, "n_repro": 6}:
                _fail("B210_HANDOFF", f"counts {sm['counts']}")
            s2 = _sessions(neg)[-1]
            _remove(neg, *[f"runlog_arm2_seed303_attempt1_{s2}.{x}.log{y}"
                           for x in ("stdout", "stderr")
                           for y in ("", ".sha256")])
            _expect_refusal(neg, ref, "MISSING_RUN_LOG",
                            "launcher campaign, both run logs removed")
        logger.info("B2.10 handoff PASS: launcher-produced 2-session campaign "
                    "accepted by the summary (all populations n=3/6); removed "
                    "run logs refused")
    finally:
        _kill_stubs()
        _kill_all_inhibitors(scratch)
        shutil.rmtree(scratch, ignore_errors=False)


def b210_all() -> None:
    b210_launcher()
    b210_exit_and_disk()
    b210_signals()
    b210_handoff()


def b210_host() -> None:
    """AM4 host run (before the pin): the production HOST CONSTANTS block,
    only the trainer line replaced by the stub. Verifies the real evidence,
    branch, exclusion-list, module-hash and disk checks, then stops after
    the first run. Host only; not part of the default run."""
    scratch = _new_scratch("v03dlt_b210host_")
    try:
        _launcher_copy(scratch, host=True)
        _plan(scratch, {"0_101_1": ["succeed", 3], "default": ["succeed", 0]})
        c = _Campaign(scratch, "host")
        p = _start_launcher(c, "new")
        _wait_for(scratch / "stub_started_0_101_1.txt", "host-run stub")
        (c.root / "STOP").write_bytes(b"host run: stop after run 1\n")
        o, e = p.communicate(timeout=B210_TIMEOUT)
        if p.returncode != 0 or "STOP_REQUESTED" not in e or \
                "campaign disk allowance PASS for 6 unsettled run(s)" not in e:
            _fail("B210H_FAILED", f"rc {p.returncode}: {e[-2000:]}")
        logger.info("B2.10H PASS: production HOST CONSTANTS -- real P4, probe, "
                    "gate (+sidecars), V03 evidence, branch, exclusion list, "
                    "module hashes and the 6-run disk allowance verified; one "
                    "stub run; STOP honoured")
    finally:
        _kill_stubs()
        _kill_all_inhibitors(scratch)
        shutil.rmtree(scratch, ignore_errors=False)


# ---------------------------------------------------------------------------
# B2.11 summary fixture: tiny synthetic campaigns, valid 30-epoch metadata
# ---------------------------------------------------------------------------

B211_PIN = "b2110c0de" + "0" * 31             # synthetic 40-hex pin
B211_SESSIONS = ("20261018T000001Z", "20261018T000002Z")


def _vals(arm: int, seed: int, e: int) -> dict:
    nll = 1.0 + 0.05 * arm + 0.001 * seed + 1.0 / e
    rec = 0.5 + 0.002 * seed + 0.2 / e
    lam = 0.0 if arm == 0 else dls.LAMBDA_2
    return {"val_NLL": nll, "val_L_rec": rec, "L_val": nll + lam * rec}


def _hist(arm: int, seed: int, n: int, spe: int) -> list:
    lam = 0.0 if arm == 0 else dls.LAMBDA_2
    out = []
    for i in range(1, n + 1):
        d = 2.0 + 0.001 * seed + 1.0 / i
        r = None if arm == 0 else 0.6 + 0.5 / i
        out.append({"step": i, "epoch": (i - 1) // spe, "L_density": d,
                    "L_rec": r, "lam_L_rec": None if r is None else lam * r,
                    "L_total": d + (0.0 if r is None else lam * r),
                    "grad_norm": 1.0 + 1.0 / i, "lr": v03t.LEARNING_RATE})
    return out


def _ck_bytes(name: str) -> bytes:
    return f"synthetic checkpoint {name}\n".encode()


def _b211_record(arm, seed, att, status, steps, k, nh, code,
                 epochs=None, total=None, spe=None, pin=B211_PIN,
                 schema=None) -> dict:
    epochs = dls.N_EPOCHS if epochs is None else epochs
    total = dls.TOTAL_STEPS if total is None else total
    spe = dls.STEPS_PER_EPOCH if spe is None else spe
    lam = 0.0 if arm == 0 else dls.LAMBDA_2
    cks = []
    for e in range(1, k + 1):
        name = v03t.checkpoint_name(arm, seed, att, e)
        cks.append({"epoch": e, "file": name,
                    "file_sha256": hashlib.sha256(_ck_bytes(name)).hexdigest(),
                    "arm": arm, "seed": seed, "attempt": att, "lambda": lam,
                    "init_seed": seed, **_vals(arm, seed, e)})
    return {"schema": dlt.RECORD_SCHEMA_DLT if schema is None else schema,
            "script": "B2.11 synthetic",
            "arm": arm, "lambda": lam, "seed": seed, "attempt": att,
            "status": status, "error_code": code, "steps_completed": steps,
            "config": {"epochs": epochs, "steps": total, "init_seed": seed,
                       "study": dlt.STUDY},
            "commit_pin": pin,
            "p4_stats2": {"path": "synthetic/scaling_statistics.json",
                          "file_sha256": dls.P4_SHA,
                          "semantic_sha256":
                              dlt.REGISTERED_P4_SEMANTIC_SHA256},
            "import_origin": {"repo_root": "synthetic", "modules_checked": 20,
                              "rule": "tracked at pin; clean status"},
            "history": _hist(arm, seed, nh, spe),
            "validation_history": [{"epoch": e, **_vals(arm, seed, e)}
                                   for e in range(1, k + 1)],
            "checkpoints": cks, "wall_clock_s": 3600.0 * (1 + arm),
            "environment": {"python": "3.12.3", "torch": "synthetic",
                            "platform": "synthetic",
                            "cpu_model": "synthetic", "threads": 4}}


def _completed(arm, seed, att=1):
    return (arm, seed, att, "completed", dls.TOTAL_STEPS, dls.N_EPOCHS,
            dls.TOTAL_STEPS, None)


def _write_sha(root: Path, name: str) -> None:
    (root / f"{name}.sha256").write_text(
        f"{v03t._file_sha256(root / name)}  {name}\n")


def _b211_campaign(base: Path, name: str, runs: list, *, mut_rec=None,
                   mut_ent=None, last_prefix_short: bool = False) -> Path:
    """Seal a synthetic campaign exactly as trainer + launcher would."""
    root = base / name / f"{dlt.CAMPAIGN_PREFIX}{B211_PIN[:8]}"
    root.mkdir(parents=True)
    recs = [_b211_record(*r) for r in runs]
    if mut_rec:
        mut_rec(recs)
    lp = root / dlt.LEDGER_NAME
    entries = []
    for r, rec in zip(runs, recs):
        arm, seed, att = r[0], r[1], r[2]
        for c in rec["checkpoints"]:
            (root / c["file"]).write_bytes(_ck_bytes(c["file"]))
        rp = root / v03t.run_record_name(arm, seed, att)
        rsha = v03t._write_with_sidecar(rp, v02m.canonical_json(rec))
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
        v03t.append_entry(lp, en)
    led = v03t.load_ledger(lp)
    n = len(led["entries"])
    cuts = (n // 2, n - 1 if last_prefix_short else n)
    prev = 0
    for s, cut in zip(B211_SESSIONS, cuts):
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
        excl = (dlt.LEDGER_NAME, dlt.LEDGER_NAME + ".sha256", "STOP",
                ".launch.lock")
        lines = [f"{v03t._file_sha256(f)}  {f.name}\n"
                 for f in sorted(root.iterdir())
                 if f.is_file() and f.name not in excl
                 and not f.name.startswith("campaign_manifest_")]
        man = root / f"campaign_manifest_{s}.sha256"
        man.write_text("".join(lines))
        _write_sha(root, man.name)              # launcher: manifest sidecar
    return root


def _b211_v03_ref(base: Path) -> Path:
    ref = base / "v03_train"
    ref.mkdir(parents=True)
    lp = ref / "v03_run_ledger.json"
    for seed in dls.SEEDS:
        for arm in dls.ARMS:
            rec = _b211_record(arm, seed, 1, "completed", dls.V03_STEPS,
                               dls.V03_EPOCHS, dls.V03_STEPS, None,
                               epochs=dls.V03_EPOCHS, total=dls.V03_STEPS,
                               schema="seqref-v03-train-record/1")
            rp = ref / v03t.run_record_name(arm, seed, 1)
            rsha = v03t._write_with_sidecar(rp, v02m.canonical_json(rec))
            v03t.append_entry(lp, {"arm": arm, "seed": seed, "attempt": 1,
                                   "status": "completed",
                                   "record_file": rp.name,
                                   "record_sha256": rsha})
    return ref


def _v03_pins(ref: Path) -> dict:
    return {"V03_LEDGER_SHA": v03t._file_sha256(ref / "v03_run_ledger.json"),
            "V03_RECORD_SHA": {(a, s): v03t._file_sha256(
                ref / v03t.run_record_name(a, s, 1))
                for a in dls.ARMS for s in dls.SEEDS}}


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
    except (dls.V03DLTSError, v03t.V03TError) as exc:
        if exc.code != code:
            _fail("B211_WRONG_REFUSAL", f"{name}: {exc.code} != {code}")
        if (root / "summary").exists() and code != "ARTIFACT_EXISTS":
            _fail("B211_WROTE_ON_REFUSAL", f"{name}: summary dir created")
        logger.info("B2.11 branch PASS: %s -> %s", name, code)
        return
    _fail("B211_NOT_REFUSED", f"{name}: build passed; expected {code}")


def _summary_ok(root: Path) -> dict:
    out = root / "summary"
    for side in sorted(out.glob("*.sha256")):
        res = subprocess.run(["sha256sum", "-c", "--quiet", side.name],
                             cwd=out, capture_output=True, text=True,
                             check=False)
        if res.returncode != 0:
            _fail("B211_OUTPUT_SIDECAR", f"{side.name}: {res.stdout}")
    summ = json.loads((out / "v03dlt_summary.json").read_text())
    man = json.loads((out / "plot_manifest.json").read_text())
    if len(man["plots"]) != 8 or any(
            v03t._file_sha256(out / pl["file"]) != pl["sha256"]
            for pl in man["plots"]):
        _fail("B211_PLOTS", "plot set or plot hashes wrong")
    if man["summary_sha256"] != v03t._file_sha256(out / "v03dlt_summary.json"):
        _fail("B211_PLOTS", "manifest summary hash wrong")
    return summ


def _row(sm: dict, arm: int, seed: int) -> dict:
    return [x for x in sm["run_table"] if (x["arm"], x["seed"]) ==
            (arm, seed)][0]


def b211_summary() -> None:
    base = Path(tempfile.mkdtemp(prefix="v03dlt_b211_"))
    try:
        ref = _b211_v03_ref(base)
        full = [_completed(a, s) for s, a in dlt.ORDER_DLT]
        with _patched(dls, **_v03_pins(ref)):
            # -- n_pair = 3 -----------------------------------------------------
            r3 = _b211_campaign(base, "n3", full)
            pristine = base / "pristine"
            shutil.copytree(r3, pristine / r3.name)
            if dls.build(r3, ref) != 0:
                _fail("B211_N3", "build rc != 0")
            sm = _summary_ok(r3)
            nck = sum(1 for k in sm["input_hash_register"]
                      if k.startswith("checkpoint/"))
            a0 = sm["arm_summaries"]["arm0"]["val_NLL_epoch30"]
            if sm["counts"] != {"n_run": 6, "n_arm0": 3, "n_arm2": 3,
                                "n_pair": 3, "n_repro": 6} or nck != 180 or \
                    a0["n"] != 3 or a0["sd"] is None or \
                    sm["paired"]["cross_seed_epoch30"][
                        "difference_val_NLL_epoch30"]["n"] != 3 or \
                    any(r["selected_epoch"] != 30 for r in sm["run_table"]) \
                    or sm["incomplete_runs"]:
                _fail("B211_N3", f"counts {sm['counts']}, checkpoints {nck}")
            _expect_refusal(r3, ref, "ARTIFACT_EXISTS", "rebuild")
            # -- n_pair = 2 and n_arm0 != n_pair: terminal numerical failure ----
            runs = full[:5] + [(2, 303, 1, "numerical_failure", 5000, 4, 4999,
                                "NUMERICAL_FAILURE")]
            r = _b211_campaign(base, "n2", runs)
            dls.build(r, ref)
            sm = _summary_ok(r)
            row = _row(sm, 2, 303)
            if sm["counts"] != {"n_run": 6, "n_arm0": 3, "n_arm2": 2,
                                "n_pair": 2, "n_repro": 6} or \
                    sm["populations"]["P-arm0"]["seeds"] != [101, 202, 303] \
                    or sm["paired"]["seeds"] != [101, 202] or \
                    row["incomplete_cause"] != "numerical_failure" or \
                    "selected_epoch" in row or \
                    "arm2_seed303" not in sm["populations"]["P-repro"]["runs"]:
                _fail("B211_N2", f"{sm['counts']} {row}")
            # -- n_pair = 1: exhausted run blocks later runs ---------------------
            runs = [_completed(0, 101), _completed(2, 101), _completed(0, 202)] \
                + [(2, 202, a, "interrupted", 1086 * (a + 2), a + 2,
                    1086 * (a + 2), "INFRASTRUCTURE_INTERRUPTION")
                   for a in (1, 2, 3)]
            r = _b211_campaign(base, "n1", runs)
            dls.build(r, ref)
            sm = _summary_ok(r)
            causes = {(x["arm"], x["seed"]): x["incomplete_cause"]
                      for x in sm["run_table"]}
            x = sm["paired"]["cross_seed_epoch30"]["difference_val_NLL_epoch30"]
            a0 = sm["arm_summaries"]["arm0"]["val_NLL_epoch30"]
            if sm["paired"]["seeds"] != [101] or x["n"] != 1 or \
                    x["sd"] is not None or \
                    x["sd_note"] != "not available (n < 2)" or \
                    a0["n"] != 2 or a0["sd"] is None or \
                    causes[(2, 202)] != "incomplete_attempts_exhausted" or \
                    causes[(0, 303)] != "not_attempted" or \
                    causes[(2, 303)] != "not_attempted" or \
                    "arm2_seed202" not in sm["populations"]["P-repro"]["runs"] \
                    or sm["counts"]["n_run"] != 4:
                _fail("B211_N1", f"{sm['counts']} {x} {causes}")
            # -- n_pair = 0: failed_validity; run stopped before epoch 3 --------
            runs = [_completed(0, 101),
                    (2, 101, 1, "failed_validity", 10, 0, 9,
                     "STEP_COUNT_MISMATCH"), _completed(0, 202),
                    (2, 202, 1, "interrupted", 1500, 1, 1500,
                     "KEYBOARD_INTERRUPT")]
            r = _b211_campaign(base, "n0", runs)
            dls.build(r, ref)
            sm = _summary_ok(r)
            causes = {(x["arm"], x["seed"]): x["incomplete_cause"]
                      for x in sm["run_table"]}
            pr = sm["paired"]
            if pr["n"] != 0 or "note" not in pr or \
                    "cross_seed_epoch30" in pr or \
                    sm["arm_summaries"]["arm2"]["n"] != 0 or \
                    sm["arm_summaries"]["arm2"]["val_NLL_epoch30"]["mean"] \
                    is not None or \
                    causes[(2, 101)] != "failed_validity" or \
                    causes[(2, 202)] != "interrupted_attempts_remain" or \
                    causes[(2, 303)] != "not_attempted" or \
                    "arm2_seed202" in sm["populations"]["P-repro"]["runs"] or \
                    "arm2_seed202" not in sm["populations"]["P-run"]["runs"] \
                    or "arm2_seed101" not in sm["populations"]["P-run"]["runs"]:
                _fail("B211_N0", f"{pr} {causes} {sm['populations']}")
            # -- overwrite refusal before reading -------------------------------
            r = base / "pre" / r3.name
            shutil.copytree(pristine / r3.name, r)
            (r / "summary").mkdir()
            _expect_refusal(r, ref, "ARTIFACT_EXISTS", "pre-existing summary")
            # -- missing evidence (manifest lines dropped to isolate the check) -
            s1, s2 = B211_SESSIONS
            rec0 = v03t.run_record_name(0, 101, 1)
            ck0 = v03t.checkpoint_name(0, 101, 1, 7)
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
                    r, *[f"runlog_arm2_seed303_attempt1_{s2}.{x}.log{y}"
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
                ("no sessions", lambda r: [
                    (_remove(r, f.name) if not f.name.startswith(
                        "campaign_manifest_") else f.unlink())
                    for f in sorted(r.iterdir()) if f.name.startswith(
                        ("launcher_session_", "ledger_snapshot_",
                         "campaign_manifest_")) and f.exists()],
                 "NO_SESSION_EVIDENCE"),
                ("run log not in its manifest", lambda r: _drop_manifest_line(
                    r, f"runlog_arm0_seed101_attempt1_{s1}.stdout.log"),
                 "RUN_LOG_NOT_IN_MANIFEST"),
                ("no snapshot", lambda r: _remove(
                    r, f"ledger_snapshot_{s2}.json",
                    f"ledger_snapshot_{s2}.json.sha256"),
                 "MISSING_SESSION_EVIDENCE"),
                ("orphan manifest", lambda r: shutil.copyfile(
                    r / f"campaign_manifest_{s2}.sha256",
                    r / "campaign_manifest_20261018T000009Z.sha256"),
                 "ORPHANED_SESSION_EVIDENCE"),
                ("no record sidecar", lambda r: _remove(r, f"{rec0}.sha256"),
                 "RECORD_SIDECAR_MISSING"),
                ("no ledger sidecar", lambda r: (
                    r / (dlt.LEDGER_NAME + ".sha256")).unlink(),
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
                    r / rec0, r / v03t.run_record_name(0, 101, 2)),
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
            r = _b211_campaign(base, "short", full, last_prefix_short=True)
            _expect_refusal(r, ref, "LEDGER_NOT_SNAPSHOTTED",
                            "ledger entry outside any session snapshot")
            # V03 reference evidence changed
            bad_ref = base / "badref" / "v03_train"
            shutil.copytree(ref, bad_ref)
            p = bad_ref / v03t.run_record_name(2, 202, 1)
            p.write_bytes(p.read_bytes() + b" ")
            r = base / "refchk" / r3.name
            shutil.copytree(pristine / r3.name, r)
            _expect_refusal(r, bad_ref, "V03_EVIDENCE_CHANGED",
                            "V03 reference record changed")
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
                rs[1]["lambda"] = 0.6130341651762
                for c in rs[1]["checkpoints"]:
                    c["lambda"] = 0.6130341651762
            semantic = (
                ("status differs", None, _set(5, status="failed_validity",
                                            error_code="STEP_COUNT_MISMATCH"),
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
                    steps=dls.V03_STEPS), None, "RECORD_CONFIG_MISMATCH"),
                ("identity", _set(0, seed=202), None,
                 "RECORD_IDENTITY_MISMATCH"),
                ("schema", _set(0, schema="seqref-v03-train-record/1"), None,
                 "RECORD_SCHEMA_MISMATCH"),
                ("P4 hash", lambda rs: rs[0]["p4_stats2"].update(
                    file_sha256="0" * 64), None, "RECORD_P4_MISMATCH"),
                ("wall clock", _set(0, wall_clock_s=None), None,
                 "RECORD_WALL_CLOCK_INVALID"),
            )
            for i, (name, mr, me, code) in enumerate(semantic):
                r = _b211_campaign(base, f"sem{i}", full, mut_rec=mr,
                                   mut_ent=me)
                _expect_refusal(r, ref, code, name)
            # -- v0.2: record content (history, P4, wall clock, monitor) -----
            nf = full[:5] + [(2, 303, 1, "numerical_failure", 5000, 4, 4999,
                              "NUMERICAL_FAILURE")]

            def _h(i, j, **kv):
                return lambda rs: rs[i]["history"][j].update(kv)

            def _hdel(rs):
                del rs[0]["history"][10]["L_density"]

            def _vh(i, **kv):
                return lambda rs: rs[i]["validation_history"][0].update(kv)
            content = (
                ("history missing L_density", full, _hdel,
                 "RECORD_HISTORY_INVALID"),
                ("history grad_norm NaN", full, _h(0, 5, grad_norm=float("nan")),
                 "RECORD_HISTORY_INVALID"),
                ("numerical_failure record holds inf L_total", nf,
                 _h(5, 4998, L_total=float("inf")), "RECORD_HISTORY_INVALID"),
                ("arm 0 history with L_rec", full, _h(0, 0, L_rec=0.5),
                 "RECORD_HISTORY_INVALID"),
                ("arm 2 history L_rec null", full, _h(1, 0, L_rec=None),
                 "RECORD_HISTORY_INVALID"),
                ("history epoch wrong", full, _h(0, 1086, epoch=0),
                 "RECORD_HISTORY_INVALID"),
                ("history lr wrong", full, _h(0, 3, lr=1e-3),
                 "RECORD_HISTORY_INVALID"),
                ("P4 semantic hash", full, lambda rs: rs[0]["p4_stats2"]
                 .update(semantic_sha256="0" * 64), "RECORD_P4_MISMATCH"),
                ("P4 path missing", full, lambda rs: rs[0]["p4_stats2"]
                 .pop("path"), "RECORD_P4_MISMATCH"),
                ("wall clock inf", full, _set(0, wall_clock_s=float("inf")),
                 "RECORD_WALL_CLOCK_INVALID"),
                ("wall clock negative", full, _set(0, wall_clock_s=-1.0),
                 "RECORD_WALL_CLOCK_INVALID"),
                ("L_val identity", full, _vh(1, L_val=9.0),
                 "RECORD_VALIDATION_INVALID"),
                ("monitor NaN", full, _vh(0, val_L_rec=float("nan")),
                 "RECORD_VALIDATION_INVALID"),
                ("checkpoint monitor differs", full, lambda rs: rs[0][
                    "checkpoints"][0].update(val_NLL=9.0),
                 "RECORD_VALIDATION_INVALID"),
            )
            for i, (name, runs, mr, code) in enumerate(content):
                r = _b211_campaign(base, f"con{i}", runs, mut_rec=mr)
                _expect_refusal(r, ref, code, name)
            # -- v0.2: ledger history replayed before settlement -------------
            nfail = (0, 101, 1, "numerical_failure", 5000, 4, 4999,
                     "NUMERICAL_FAILURE")
            fval = (0, 101, 1, "failed_validity", 10, 0, 9,
                    "STEP_COUNT_MISMATCH")

            def _intr(a, s, att):
                return (a, s, att, "interrupted", 2172, 2, 2172,
                        "INFRASTRUCTURE_INTERRUPTION")
            rest = full[1:]
            history = (
                ("completed retry after numerical_failure",
                 [nfail, _completed(0, 101, 2)] + rest, None,
                 "LEDGER_RETRY_AFTER_TERMINAL"),
                ("completed retry after failed_validity",
                 [fval, _completed(0, 101, 2)] + rest, None,
                 "LEDGER_RETRY_AFTER_TERMINAL"),
                ("second attempt after completed",
                 [_completed(0, 101), _completed(0, 101, 2)] + rest, None,
                 "LEDGER_RETRY_AFTER_TERMINAL"),
                ("attempt gap 1 -> 3",
                 [_intr(0, 101, 1), _completed(0, 101, 3)] + rest, None,
                 "LEDGER_ATTEMPT_INVALID"),
                ("first attempt numbered 2",
                 [_completed(0, 101, 2)] + rest, None,
                 "LEDGER_ATTEMPT_INVALID"),
                ("fourth attempt",
                 [_intr(0, 101, a) for a in (1, 2, 3)]
                 + [_completed(0, 101, 4)], None, "LEDGER_ATTEMPT_INVALID"),
                ("arm 2 before arm 0", [full[1], full[0]] + full[2:], None,
                 "LEDGER_ORDER_INVALID"),
                ("run past an unsettled run",
                 [full[0], _intr(2, 101, 1), full[2]], None,
                 "LEDGER_ORDER_INVALID"),
                ("arm 1", [_completed(1, 101)], None, "LEDGER_RUN_INVALID"),
                ("completed with error code", full, _set(
                    0, error_code="NUMERICAL_FAILURE"),
                 "LEDGER_STATUS_INVALID"),
            )
            for i, (name, runs, me, code) in enumerate(history):
                r = _b211_campaign(base, f"led{i}", runs, mut_ent=me)
                _expect_refusal(r, ref, code, name)
        logger.info("B2.11 PASS: populations n_pair=3/2/1/0 incl. n_arm0 != "
                    "n_pair, P-repro membership (>= 3 epochs, incomplete "
                    "included; < 3 epochs excluded), SD only n >= 2, n = 0 "
                    "omitted; causes numerical_failure / failed_validity / "
                    "exhausted / attempts remain / not attempted; outputs + "
                    "plot hashes verify; overwrite refused; %d missing-evidence"
                    ", %d inconsistent-record and %d ledger-history cases "
                    "refused before writing",
                    len(cases) + 2, len(semantic) + len(content), len(history))
    finally:
        shutil.rmtree(base, ignore_errors=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    argv = sys.argv[1:] if argv is None else argv
    logger.info("fixture environment: torch %s, numpy %s, threads %d, repo "
                "%s, HEAD %s", torch.__version__, np.__version__,
                torch.get_num_threads(), REPO, _head())
    groups = [("B2.0", b20_equivalence), ("B2.1", b21_budget),
              ("B2.2", b22_lambda), ("B2.3", b23_order_attempts),
              ("B2.4", b24_path_guard), ("B2.5", b25_two_step_smoke),
              ("B2.6", b26_mask_contract), ("B2.7", b27_p4_binding),
              ("B2.8", b28_failure_paths),
              ("B2.9", b29_mode_partial_regression),
              ("B2.10", b210_all), ("B2.11", b211_summary)]
    host_only = [("B2.10H", b210_host)]
    if argv:
        groups = [g for g in groups + host_only if g[0] in argv]
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
        logger.error("V03DLT fixtures: %d not PASS: %s", len(bad), bad)
        return 2
    logger.info("V03DLT fixtures %s: ALL PASS", "/".join(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
