# SEQREF-MRI-V04T-FIX-TEST v0.3 -- targeted synthetic tests for the F1/F2/F3 repair
# LIFETIME: KEEP
# Host reproduction: V04T_ORIG=<v0.1 path> V04T_CAND=<v0.3 path> \
#   python3 v04t_fix_targeted_tests.py   (defaults below)
# v0.3: reviewer round -- F3: Ctrl+C DURING the non-gating diagnostic
#   (BaseException) bypassed the v0.2 except-Exception wrapper and skipped
#   record/ledger writes; candidate v0.3 catches it explicitly, preserves
#   the training-determined status (non-gating), records the interruption
#   in the diagnostic payload, writes all evidence, then re-raises
#   (non-zero exit). New combined-failure tests F3a/F3b/F3c/F3d. REG now
#   uses SHARED probe/gate fixture paths for orig and candidate so the
#   full raw record (including provenance objects) is compared minus only
#   {wall_clock_s, script}; the exclusion-boundary check covers every
#   field with no nested exemptions.
# v0.2: reviewer round -- regression exclusion boundary made explicit and
#   machine-verified: ONLY wall_clock_s (timing, differs even v0.1-vs-v0.1)
#   and "script" (the intended version stamp = the repair) are excluded from
#   the raw record comparison, and ONLY wall_clock_s + record_sha256 (the
#   chained hash of the differing record file) from the ledger comparison;
#   environment, statuses, all numerical telemetry and probe/gate content
#   hashes are compared raw. New check REG_exclusion_boundary_exact asserts
#   the raw diff set is exactly that boundary and nothing more.
# Purpose: prove, on fully synthetic stubs (no MRI data, no repo):
#   F1a: Ctrl+C BEFORE a checkpoint -> interrupted record+ledger+sidecars, non-zero exit
#   F1b: Ctrl+C AFTER a checkpoint -> same, checkpoints preserved
#   F1c: the interrupted pair permits the legitimate next attempt (policy restart)
#   F2a: diagnostic EXECUTION failure -> logged + recorded, record/ledger preserved
#   F2b: diagnostic SETUP failure -> same
#   F3a: TERMINAL training failure + diagnostic failure -> original
#        status/error preserved, evidence written
#   F3b: Ctrl+C during training + diagnostic failure -> interrupted
#        status/KEYBOARD_INTERRUPT preserved, evidence written
#   F3c: Ctrl+C DURING diagnostic after successful training -> completed
#        status preserved, KeyboardInterrupt recorded in payload,
#        evidence written, non-zero exit
#   F3d: Ctrl+C DURING diagnostic after terminal failure -> terminal
#        status preserved (no downgrade), evidence written, non-zero exit
#   REG: successful-path record/ledger/checkpoint bytes identical orig vs candidate
# Original under test: registered v04_train.py v0.1 (db215925...4254).
# Candidate under test: v04_train v0.3 candidate (F1/F2/F3 patch only).
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import torch

# configurable for host reproduction (KEEP): point both at the real files
ORIG_PATH = os.environ.get("V04T_ORIG", "/mnt/agents/temp/v04_train.py")
CAND_PATH = os.environ.get(
    "V04T_CAND", "/mnt/agents/output/v04_train_v0.3_candidate.py")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (" -- " + str(detail) if detail and not ok else ""))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Stub seqref_mri package (synthetic; no repo import)
# ---------------------------------------------------------------------------

def make_stubs():
    class _Err(RuntimeError):
        def __init__(self, code, msg=""):
            self.code = code
            super().__init__(f"{code}: {msg}")

    mods = {}
    for name in ["seqref_mri", "seqref_mri.src", "seqref_mri.src.flows",
                 "seqref_mri.scripts"]:
        m = types.ModuleType(name)
        m.__path__ = []
        mods[name] = m

    ffr = types.ModuleType("seqref_mri.src.free_flow_runtime")
    ffr.FLOW_DIM_REAL = 4
    ffr.COND_IN_CHANNELS = 1
    ffr.load_p4s2_parent = lambda p: {"location_index": {}}
    ffr.standardisation_vectors = lambda cmap, idx: {}
    mods["seqref_mri.src.free_flow_runtime"] = ffr

    data = types.ModuleType("seqref_mri.src.fastmri_data")

    class FastMRISliceDataset(torch.utils.data.Dataset):
        N_ITEMS = 4  # test-tunable: steps_per_epoch * BATCH_SIZE
        def __init__(self, root, split="train", mode="train"):
            self.split, self.mode = split, mode
        def set_epoch(self, ep):
            self.epoch = ep
        def __len__(self):
            return type(self).N_ITEMS
        def __getitem__(self, i):
            return {"file": "synthetic.h5", "slice_index": i,
                    "mask": torch.ones(3)}
    data.FastMRISliceDataset = FastMRISliceDataset
    mods["seqref_mri.src.fastmri_data"] = data

    v03l = types.ModuleType("seqref_mri.src.flows.v03_objective")
    v03l.V03LError = type("V03LError", (_Err,), {})
    mods["seqref_mri.src.flows.v03_objective"] = v03l

    v04o = types.ModuleType("seqref_mri.scripts.v04_objective")
    v04o.V04LError = type("V04LError", (_Err,), {})
    v04o.L_REC_SCALE = 1e7
    v04o.state = {"calls": 0, "raise_at": None, "raise_exc": None}

    def density_only(model, tgt, cond, msk):
        v04o.state["calls"] += 1
        if v04o.state["raise_at"] == v04o.state["calls"]:
            raise v04o.state["raise_exc"]
        return sum(p.sum() for p in model.parameters()) * 0.0 + 1.0
    def rec_term(model, cond, msk, y, amax, tgt_norm, maps):
        return sum(p.sum() for p in model.parameters()) * 0.0 + 0.5
    v04o.density_only = density_only
    v04o.rec_term = rec_term
    mods["seqref_mri.scripts.v04_objective"] = v04o

    v04p = types.ModuleType("seqref_mri.scripts.v04_probe")
    v04p.V04PError = type("V04PError", (_Err,), {})
    v04p.PROBE_INIT_SEED = 20260813
    v04p.LAMBDA_PRIME_FACTOR = 0.5
    v04p.LAMBDA_BAND = (1e-6, 1e6)
    v04p.global_grad_norm = lambda model: 1.0
    v04p.ffr = ffr
    mods["seqref_mri.scripts.v04_probe"] = v04p

    tb = types.ModuleType("seqref_mri.scripts.train_base")
    def _collate(items):
        return {"meta": [{"file": it["file"], "slice_index": it["slice_index"]}
                         for it in items],
                "mask": torch.stack([it["mask"] for it in items])}
    def _prepare(batch, device, test0):
        n = len(batch["meta"])
        return {"x_norm": torch.zeros(n, 4), "cond_in": torch.zeros(n, 4),
                "y": torch.zeros(n), "amax": torch.ones(n),
                "tgt_norm": torch.zeros(n, 4)}
    tb._collate = _collate
    tb._prepare = _prepare
    mods["seqref_mri.scripts.train_base"] = tb

    tff = types.ModuleType("seqref_mri.scripts.train_free_flow")
    tff.targets_from_prepared = lambda one, cmap, vecs: torch.zeros(1, 4)
    mods["seqref_mri.scripts.train_free_flow"] = tff

    v02m = types.ModuleType("seqref_mri.scripts.v02_manifests")
    v02m.BATCH_SIZE = 2
    v02m.N_EPOCHS = 3
    v02m.canonical_json = lambda obj: json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    mods["seqref_mri.scripts.v02_manifests"] = v02m

    v02t = types.ModuleType("seqref_mri.scripts.v02_train")
    v02t.derive_cmap_from_mask = lambda mask: object()
    mods["seqref_mri.scripts.v02_train"] = v02t

    v04m = types.ModuleType("seqref_mri.scripts.v04_model")

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)
        def decode_scalars(self, z, cond, mask):
            return z.sum(dim=1, keepdim=True)
    def build_v04_model(init_seed):
        torch.manual_seed(init_seed)
        return TinyModel()
    v04m.build_v04_model = build_v04_model
    mods["seqref_mri.scripts.v04_model"] = v04m
    return mods


def load_module(path, name):
    stubs = make_stubs()
    saved = dict(sys.modules)
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k in stubs:
            sys.modules.pop(k, None)
        sys.modules.update({k: v for k, v in saved.items() if k not in sys.modules})
    mod.__stubs__ = stubs
    mod.verify_commit_pin = lambda expected, repo_root: expected  # not a repair target
    return mod


def write_contract_records(tmp, mod):
    probe = {"schema": "seqref-v04-probe-record/1", "n_batches": 100,
             "batch_size": 2, "l_rec_scale": 1e7, "split": "train",
             "mode": "eval", "validation_source_amendment": "A2",
             "weight_state": {"kind": "init", "model_init_seed": 20260813},
             "lambda_prime_factor": 0.5, "batch_list_sha256": "a" * 64,
             "lambda_prime": 2.4348938608414534}
    gate = {"schema": "seqref-v04-gate-record/1",
            "gates": {g: {"status": "PASS"} for g in "abcdef"}}
    paths = {}
    for name, rec in (("probe.json", probe), ("gate.json", gate)):
        p = tmp / name
        p.write_text(json.dumps(rec))
        Path(str(p) + ".sha256").write_text(f"{sha(p)}  {p.name}\n")
        paths[name] = str(p)
    return paths


def tiny_budget(mod, steps_per_epoch=1, epochs=1, val_batches=1):
    mod.STEPS_PER_EPOCH = steps_per_epoch
    mod.TOTAL_STEPS = steps_per_epoch * epochs
    mod.N_VAL_BATCHES = val_batches
    mod.N_EPOCHS = epochs  # module-level binding used by the training loop
    (mod.__stubs__["seqref_mri.src.fastmri_data"]
        .FastMRISliceDataset.N_ITEMS) = steps_per_epoch * 2  # 2 = stub BATCH_SIZE


def invoke(mod, tmp, arm, seed, out, recs=None):
    recs = recs or write_contract_records(tmp, mod)
    argv = ["v04_train", "--mode", "run", "--data-root", str(tmp),
            "--p4-stats2", str(tmp / "p4.json"),
            "--probe-record", recs["probe.json"],
            "--gate-record", recs["gate.json"],
            "--commit-pin", "0" * 40, "--out", str(out),
            "--arm", str(arm), "--seed", str(seed)]
    (tmp / "p4.json").write_text("{}")
    old = sys.argv
    sys.argv = argv
    try:
        return mod.main()
    except KeyboardInterrupt:
        return 130  # v0.1 lets it escape main(); the shell would see 130
    finally:
        sys.argv = old


def read_ledger(out):
    p = Path(out) / "v04_run_ledger.json"
    side = Path(str(p) + ".sha256")
    sidecar_ok = side.is_file() and sha(p) == side.read_text().split()[0]
    return (json.loads(p.read_text()) if p.is_file() else None), sidecar_ok


def read_record(out, arm, seed, attempt):
    p = Path(out) / f"v04_arm{arm}_seed{seed}_attempt{attempt}_train_record.json"
    if not p.is_file():
        return None, False
    side = Path(str(p) + ".sha256")
    ok = side.is_file() and sha(p) == side.read_text().split()[0]
    return json.loads(p.read_text()), ok


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------

def main():
    import tempfile
    orig = load_module(ORIG_PATH, "v04_train_orig")
    cand = load_module(CAND_PATH, "v04_train_cand")

    # --- F1a: Ctrl+C BEFORE any checkpoint ---------------------------------
    for mod, tag in ((orig, "orig"), (cand, "cand")):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            tiny_budget(mod, steps_per_epoch=1, epochs=1)
            st = mod.__stubs__["seqref_mri.scripts.v04_objective"].state
            st.update({"calls": 0, "raise_at": 1,
                       "raise_exc": KeyboardInterrupt()})
            rc = invoke(mod, tmp, 0, 101, tmp / "out")
            rec, side_ok = read_record(tmp / "out", 0, 101, 1)
            led, led_side_ok = read_ledger(tmp / "out")
            if tag == "orig":  # defect demonstration
                check("F1a_defect_orig_records_completed",
                      rec and rec["status"] == "completed"
                      and rec["steps_completed"] < 3258 or rec is None,
                      f"orig status={rec and rec['status']}")
            else:
                check("F1a_exit_nonzero", rc == 2, f"rc={rc}")
                check("F1a_record_interrupted",
                      rec and rec["status"] == "interrupted"
                      and rec["error_code"] == "KEYBOARD_INTERRUPT"
                      and side_ok, f"{rec and rec['status']}")
                check("F1a_ledger_interrupted",
                      led and led["entries"][0]["status"] == "interrupted"
                      and led_side_ok, f"{led and led['entries']}")

    # --- F1b: Ctrl+C AFTER epoch-1 checkpoint ------------------------------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tiny_budget(cand, steps_per_epoch=1, epochs=2)
        st = cand.__stubs__["seqref_mri.scripts.v04_objective"].state
        st.update({"calls": 0, "raise_at": 3, "raise_exc": KeyboardInterrupt()})
        # call 1 = epoch-0 train step; call 2 = epoch-0 validation; call 3 =
        # epoch-1 train step -> interruption lands AFTER the epoch-1 checkpoint
        rc = invoke(cand, tmp, 0, 101, tmp / "out")
        rec, side_ok = read_record(tmp / "out", 0, 101, 1)
        led, led_side_ok = read_ledger(tmp / "out")
        ck = list((tmp / "out").glob("v04_arm0_seed101_attempt1_epoch*.pt"))
        check("F1b_exit_nonzero", rc == 2, f"rc={rc}")
        check("F1b_interrupted_with_checkpoint",
              rec and rec["status"] == "interrupted" and side_ok
              and len(rec["checkpoints"]) == 1 and len(ck) == 1
              and led_side_ok, f"{rec and rec['status']}")
        check("F1b_diagnostic_still_ran",
              rec and isinstance(rec["post_training_conditioning_diagnostic"],
                                 dict))

    # --- F1c: interrupted pair permits the legitimate next attempt ---------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tiny_budget(cand, steps_per_epoch=1, epochs=1)
        st = cand.__stubs__["seqref_mri.scripts.v04_objective"].state
        st.update({"calls": 0, "raise_at": 1, "raise_exc": KeyboardInterrupt()})
        invoke(cand, tmp, 0, 101, tmp / "out")            # attempt 1: interrupted
        st.update({"calls": 0, "raise_at": None, "raise_exc": None})
        rc = invoke(cand, tmp, 0, 101, tmp / "out")       # attempt 2: completes
        rec2, side2 = read_record(tmp / "out", 0, 101, 2)
        led, led_side = read_ledger(tmp / "out")
        statuses = [e["status"] for e in led["entries"]] if led else None
        nxt = cand.expected_next_run(led) if led else None
        check("F1c_retry_completes",
              rc == 0 and rec2 and rec2["attempt"] == 2
              and rec2["status"] == "completed" and side2,
              f"rc={rc} statuses={statuses}")
        check("F1c_ledger_chain_and_order",
              statuses == ["interrupted", "completed"] and led_side
              and nxt == (1, 101), f"statuses={statuses} next={nxt}")

    # --- F2a: diagnostic EXECUTION failure preserves record/ledger ---------
    for mod, tag in ((orig, "orig"), (cand, "cand")):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            tiny_budget(mod, steps_per_epoch=1, epochs=1)
            st = mod.__stubs__["seqref_mri.scripts.v04_objective"].state
            st.update({"calls": 0, "raise_at": None, "raise_exc": None})
            def boom(*a, **k):
                raise RuntimeError("synthetic diagnostic failure")
            mod.post_training_conditioning_diagnostic = boom
            rc = invoke(mod, tmp, 0, 101, tmp / "out")
            rec, side_ok = read_record(tmp / "out", 0, 101, 1)
            led, led_side = read_ledger(tmp / "out")
            if tag == "orig":  # defect demonstration: record/ledger lost
                check("F2a_defect_orig_loses_record",
                      rec is None or led is None
                      or len(led["entries"]) == 0,
                      f"orig rec={'present' if rec else 'MISSING'}")
            else:
                check("F2a_completed_despite_diag_failure",
                      rc == 0 and rec and rec["status"] == "completed"
                      and side_ok and led and led["entries"][0]["status"]
                      == "completed" and led_side, f"rc={rc}")
                check("F2a_diag_error_recorded",
                      rec and isinstance(
                          rec["post_training_conditioning_diagnostic"], dict)
                      and "error" in rec["post_training_conditioning_diagnostic"]
                      and "RuntimeError" in rec[
                          "post_training_conditioning_diagnostic"]["error"])

    # --- F2b: diagnostic SETUP failure preserves record/ledger -------------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tiny_budget(cand, steps_per_epoch=1, epochs=1)
        st = cand.__stubs__["seqref_mri.scripts.v04_objective"].state
        st.update({"calls": 0, "raise_at": None, "raise_exc": None})
        cand.__stubs__["seqref_mri.scripts.v04_probe"].ffr.COND_IN_CHANNELS = None
        rc = invoke(cand, tmp, 0, 101, tmp / "out")
        rec, side_ok = read_record(tmp / "out", 0, 101, 1)
        led, led_side = read_ledger(tmp / "out")
        check("F2b_setup_failure_preserved",
              rc == 0 and rec and rec["status"] == "completed" and side_ok
              and "error" in (rec["post_training_conditioning_diagnostic"] or {})
              and led_side, f"rc={rc}")

    # --- F3 combined failures (fresh instances; F2b mutated a stub) -------
    # call map at steps_per_epoch=2, epochs=2, val_batches=1: epoch1 calls
    # 1-3 -> checkpoint 1; epoch2 calls 4-6 -> checkpoint 2. raise_at=4
    # therefore fails AFTER one checkpoint, so the diagnostic always runs.
    def f3_instance():
        mod = load_module(CAND_PATH, f"v04_train_cand_f3_{len(RESULTS)}")
        tiny_budget(mod, steps_per_epoch=2, epochs=2)
        st = mod.__stubs__["seqref_mri.scripts.v04_objective"].state
        st.update({"calls": 0, "raise_at": None, "raise_exc": None})
        return mod

    # F3a: TERMINAL training failure (U_NON_FINITE -> numerical_failure)
    # followed by diagnostic failure -> original status/error preserved.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mod = f3_instance()
        v04e = mod.__stubs__["seqref_mri.scripts.v04_objective"].V04LError
        st = mod.__stubs__["seqref_mri.scripts.v04_objective"].state
        st.update({"raise_at": 4,
                   "raise_exc": v04e("U_NON_FINITE", "synthetic")})
        def boom(*a, **k):
            raise RuntimeError("synthetic diagnostic failure")
        mod.post_training_conditioning_diagnostic = boom
        rc = invoke(mod, tmp, 0, 101, tmp / "out")
        rec, side_ok = read_record(tmp / "out", 0, 101, 1)
        led, led_side = read_ledger(tmp / "out")
        check("F3a_terminal_status_preserved",
              rc == 2 and rec and rec["status"] == "numerical_failure"
              and rec["error_code"] == "U_NON_FINITE" and side_ok
              and led and led["entries"][0]["status"] == "numerical_failure"
              and led_side, f"rc={rc}")
        check("F3a_diag_error_recorded",
              rec and "RuntimeError" in (
                  rec["post_training_conditioning_diagnostic"] or {})
              .get("error", ""))

    # F3b: Ctrl+C DURING TRAINING followed by diagnostic failure ->
    # interrupted/KEYBOARD_INTERRUPT preserved, evidence written.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mod = f3_instance()
        st = mod.__stubs__["seqref_mri.scripts.v04_objective"].state
        st.update({"raise_at": 4, "raise_exc": KeyboardInterrupt()})
        def boom(*a, **k):
            raise RuntimeError("synthetic diagnostic failure")
        mod.post_training_conditioning_diagnostic = boom
        rc = invoke(mod, tmp, 0, 101, tmp / "out")
        rec, side_ok = read_record(tmp / "out", 0, 101, 1)
        led, led_side = read_ledger(tmp / "out")
        check("F3b_interrupted_status_preserved",
              rc == 2 and rec and rec["status"] == "interrupted"
              and rec["error_code"] == "KEYBOARD_INTERRUPT" and side_ok
              and led and led["entries"][0]["status"] == "interrupted"
              and led_side, f"rc={rc}")
        check("F3b_diag_error_recorded",
              rec and "RuntimeError" in (
                  rec["post_training_conditioning_diagnostic"] or {})
              .get("error", ""))

    # F3c: Ctrl+C DURING THE DIAGNOSTIC after successful training ->
    # completed status preserved (non-gating), KeyboardInterrupt recorded
    # in the payload, evidence written, non-zero exit.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mod = f3_instance()
        def boom(*a, **k):
            raise KeyboardInterrupt
        mod.post_training_conditioning_diagnostic = boom
        rc = invoke(mod, tmp, 0, 101, tmp / "out")
        rec, side_ok = read_record(tmp / "out", 0, 101, 1)
        led, led_side = read_ledger(tmp / "out")
        check("F3c_completed_status_preserved",
              rc == 2 and rec and rec["status"] == "completed"
              and rec["error_code"] in (None, "") and side_ok
              and led and led["entries"][0]["status"] == "completed"
              and led_side, f"rc={rc}")
        check("F3c_diag_interrupt_recorded",
              rec and (rec["post_training_conditioning_diagnostic"] or {})
              .get("error") == "KeyboardInterrupt")

    # F3d: Ctrl+C DURING THE DIAGNOSTIC after a TERMINAL failure ->
    # terminal status NOT downgraded, evidence written, non-zero exit.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mod = f3_instance()
        v04e = mod.__stubs__["seqref_mri.scripts.v04_objective"].V04LError
        st = mod.__stubs__["seqref_mri.scripts.v04_objective"].state
        st.update({"raise_at": 4,
                   "raise_exc": v04e("U_NON_FINITE", "synthetic")})
        def boom(*a, **k):
            raise KeyboardInterrupt
        mod.post_training_conditioning_diagnostic = boom
        rc = invoke(mod, tmp, 0, 101, tmp / "out")
        rec, side_ok = read_record(tmp / "out", 0, 101, 1)
        led, led_side = read_ledger(tmp / "out")
        check("F3d_terminal_not_downgraded",
              rc == 2 and rec and rec["status"] == "numerical_failure"
              and rec["error_code"] == "U_NON_FINITE" and side_ok
              and led and led["entries"][0]["status"] == "numerical_failure"
              and led_side, f"rc={rc}")
        check("F3d_diag_interrupt_recorded",
              rec and (rec["post_training_conditioning_diagnostic"] or {})
              .get("error") == "KeyboardInterrupt")

    # --- REG: successful-path equivalence orig vs candidate ----------------
    # fresh module instances (earlier phases monkeypatch diagnostic stubs);
    # SHARED probe/gate fixture paths for both runs, so the FULL raw record
    # (including provenance objects: path + sha256) is compared minus only
    # {wall_clock_s, script} -- no nested exemptions.
    orig_r = load_module(ORIG_PATH, "v04_train_orig_reg")
    cand_r = load_module(CAND_PATH, "v04_train_cand_reg")
    with tempfile.TemporaryDirectory() as shared_td:
        shared_recs = write_contract_records(Path(shared_td), None)
        outs = {}
        for mod, tag in ((orig_r, "orig"), (cand_r, "cand")):
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                tiny_budget(mod, steps_per_epoch=2, epochs=2)
                st = mod.__stubs__["seqref_mri.scripts.v04_objective"].state
                st.update({"calls": 0, "raise_at": None, "raise_exc": None})
                rc = invoke(mod, tmp, 0, 101, tmp / "out",
                            recs=shared_recs)
                rec, _ = read_record(tmp / "out", 0, 101, 1)
                led, _ = read_ledger(tmp / "out")
                ck = sorted((tmp / "out").glob("*.pt"))
                outs[tag] = {"rc": rc, "rec": rec, "led": led,
                             "ck_sha": [sha(p) for p in ck]}
    a, b = outs["orig"], outs["cand"]

    def norm_rec(r):
        # excluded: wall_clock_s (timing, differs even v0.1-vs-v0.1) and
        # "script" (the intended version stamp -- the repair itself).
        # EVERYTHING else compared raw, including environment and the
        # full probe/gate provenance objects (shared fixture paths).
        return {k: v for k, v in r.items()
                if k not in ("wall_clock_s", "script")}
    check("REG_exit_and_status_equal",
          a["rc"] == b["rc"] == 0
          and a["rec"]["status"] == b["rec"]["status"] == "completed")
    check("REG_record_identical",
          norm_rec(a["rec"]) == norm_rec(b["rec"]))
    check("REG_ledger_identical",
          # record_sha256 legitimately differs: the record embeds per-tempdir
          # paths (normalized above); everything else must be byte-equal
          [{k: v for k, v in e.items()
            if k not in ("wall_clock_s", "record_sha256")}
           for e in a["led"]["entries"]]
          == [{k: v for k, v in e.items()
               if k not in ("wall_clock_s", "record_sha256")}
              for e in b["led"]["entries"]])
    check("REG_checkpoint_bytes_identical",
          a["ck_sha"] == b["ck_sha"] and len(a["ck_sha"]) == 2,
          f"{len(a['ck_sha'])} checkpoints")

    # --- REG exclusion boundary: machine-verified, nothing else differs ----
    # FULL raw field comparison -- shared fixture paths mean provenance
    # objects (path + sha256) must be identical, not exempted.
    rec_diff = {k for k in set(a["rec"]) | set(b["rec"])
                if a["rec"].get(k) != b["rec"].get(k)}
    led_diff = set()
    for ea, eb in zip(a["led"]["entries"], b["led"]["entries"]):
        for k in set(ea) | set(eb):
            if ea.get(k) != eb.get(k):
                led_diff.add(k)
    check("REG_exclusion_boundary_exact",
          rec_diff <= {"wall_clock_s", "script"}
          and "script" in rec_diff
          and a["rec"]["script"].endswith(" v0.1")
          and b["rec"]["script"].endswith(" v0.3")
          and led_diff <= {"wall_clock_s", "record_sha256"}
          and "record_sha256" in led_diff,
          f"rec_diff={sorted(rec_diff)} led_diff={sorted(led_diff)}")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks PASS")
    return 2 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
