# SEQREF-MRI-DIAGJSHF1IT v0.5 -- synthetic end-to-end integration test of F1 prepare() (DIAG-I2 v0.2)
# LIFETIME: KEEP
# =============================================================================
# SYNTHETIC DATA ONLY. The parent builds fixtures with the pinned code
# (v05lt_fixtures volumes, untrained production architecture saved in the
# pinned weights format, rows from pinned validate(); ZF rows are synthetic
# values, labelled). A child process then runs F1 exactly as on the host
# from start() on: write guard on its empty OUT, disk check, prepare() with
# the real adapter and pinned loaders, manifest. Only bind_production()
# (launch chain, completion, backend, load_run) is replaced by a fixture
# context, because the sandbox has no campaign; that part stays host-only.
# Usage: python -B test_diag_jsh_f1_integration.py PINNED_REPO_ROOT
# Changelog (v0.5, 2026-11-10): F1 v0.6 -- the fixture binding's bound
#   inputs now have the pinned _bound() shape ({"path": Path, "sha256"})
#   and pass through F1.record_inputs (the production conversion); P16
#   manifest input paths are str; P17 raw Path in the binding -> STOP
#   DIAG_MANIFEST_TYPE at the early check (before STAGE 5, no manifest);
#   P18 non-serialisable field generated later -> STOP DIAG_MANIFEST_TYPE
#   at the final check (after STAGE 5, no manifest); P19 unexpected
#   bound-input structure -> DIAG_BINDING; P15 version 0.6.
# Changelog (v0.4, 2026-11-09): F1 v0.5 -- P10b early timing refusal
#   before the full replay (no STAGE 5, no manifest); P15 manifest version
#   0.5 with the early-refusal projection recorded.
# Changelog (v0.3, 2026-11-09): command v0.4 checks (all pipeline statuses,
#   summary inside the allowance, reserve passed to Python) and a scaled
#   termination test of run_capped with TERM-ignoring / TERM-honouring
#   children.
# Changelog (v0.2, 2026-11-09): F1 v0.3 (enforced budget, start(out, A,
#   reserve), overall deadline); modes "budget" (manifest refused before
#   write) and "unbudgeted" (unregistered write inside OUT halts 77).
# Changelog (v0.1, 2026-11-09): first integration test.
# =============================================================================
from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

PIN = Path(sys.argv[1]).resolve()
CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(PIN), str(CODE)]
logging.basicConfig(level=logging.CRITICAL)
SEED_ARMS = [(s, a) for s in (2026, 2027, 2028) for a in ("lam0", "lam2")]


def child(fx: Path, out: Path, mode: str) -> int:
    import time
    import diag_jsh_adapter as A
    import diag_jsh_core as K
    import diag_jsh_f1_prepare as F1
    if mode == "budget":
        F1.F1_BUDGET = 9000          # log fits, manifest does not
    try:
        budget = F1.start(out, A, 10 ** 18 if mode == "disk" else 0)
    except K.DiagError as exc:
        print(f"RESULT DIAG_F1 STOP {exc.code}")
        return 1
    if mode == "write_code":
        open(CODE / "diag_jsh_core.py", "a").close()   # must halt (77)
        return 0
    if mode == "unbudgeted":
        open(out / "extra.txt", "w").close()           # must halt (77)
        return 0
    import torch
    torch.set_num_threads(1)
    from seqref_mri.scripts import v05lt_common as C
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.scripts import v05lt_prodstages as PS
    from seqref_mri.scripts import v05lt_fixtures as FX
    meta = pickle.loads((fx / "meta.pkl").read_bytes())
    sdoc = C.load_split_manifest(meta["mp"], meta["msha"], meta["expect"])
    _, vv = C.build_split_views(meta["root"], sdoc, FX.SYN_HOLDOUT,
                                meta["expect"])
    elig = M.eligibility_index(json.loads((fx / "elig.json").read_text()))
    paths = C.Paths(repo=PIN)
    stages = PS.ProductionStages(paths, paths.data_root,
                                 fixture={"root": str(fx / "fxroot")})
    pin_code = PS.source_identity(PIN)

    def load_selected(k):
        sel = json.loads((fx / f"sel_{k[0]}_{k[1]}.json").read_text())
        sel["rows"] = json.loads(Path(sel["rows_file"]).read_text())
        if mode == "tamper" and k == (2027, "lam2"):
            sel["rows"]["rows"][3]["psnr"] += 0.01
        return sel
    raw_inputs = {k: {"path": fx / f"{k}.input", "sha256": "0" * 64}
                  for k in F1.INPUT_KEYS}      # pinned _bound() shape
    inputs = (raw_inputs if mode == "pathleak"
              else F1.record_inputs(raw_inputs, K))
    if mode == "late_type":
        real_retained = A.retained_slices
        A.retained_slices = lambda *a: real_retained(*a) + [Path("/late")]  # not JSON
    ctx = {"binding": {"pin": "SYNTHETIC", "launch_chain": "SYNTHETIC",
                       "launch_inputs_sha256": "SYNTHETIC",
                       "completion_evidence_sha256": "SYNTHETIC",
                       "backend": "SYNTHETIC (1 thread)", "inputs": inputs},
           "data": {"vv": vv, "eligibility": elig, "p4": FX._unit_p4(),
                    "s_ref": 1.0},
           "zf_rows": json.loads((fx / "zf.json").read_text()),
           "zf_record_sha256": "SYNTHETIC",
           "load_selected": load_selected,
           "loaded_modules": lambda: A.check_loaded(stages, pin_code,
                                                    root=PIN)}
    t0 = time.monotonic()
    if mode == "projected":
        F1.SUMMARY_ALLOWANCE_S = 10 ** 9        # projection cannot fit
    lim = {"reserve": 0, "budget": F1.F1_BUDGET, "max_wall_s": 3600.0,
           "deadline": t0 - 1 if mode == "deadline" else t0 + 3480}
    try:
        sha = F1.prepare(ctx, out, lim, {"synthetic": True}, A, K, budget)
    except K.DiagError as exc:
        print(f"RESULT DIAG_F1 STOP {exc.code}")
        return 1
    print(f"RESULT DIAG_F1 PASS manifest_sha256={sha}")
    return 0


def parent() -> int:
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import diag_jsh_adapter as A
    import diag_jsh_f1_prepare as F1
    from seqref_mri.scripts import v05lt_common as C
    from seqref_mri.scripts import v05lt_eligibility as E
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_fixtures as FX
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.scripts import v05lt_train as T
    from seqref_mri.src import free_flow_runtime as ffr
    n_ok = n_fail = 0

    def ok(cond, label, detail=""):
        nonlocal n_ok, n_fail
        n_ok, n_fail = n_ok + bool(cond), n_fail + (not cond)
        print(("ok   " if cond else "FAIL ") + label +
              (f"  [{detail}]" if detail else ""))
    tmp = Path(tempfile.mkdtemp(prefix="diagf1it_"))
    fx = tmp / "fixture"
    fx.mkdir()
    vols = {"file9101.h5": 2, "file9201.h5": 2, "file9202.h5": 1,
            "file9203.h5": 3, "file9204.h5": 2}
    val = ["file9201.h5", "file9202.h5", "file9203.h5", "file9204.h5"]
    root, mp, msha, ex = FX.synth_split(fx, vols, ["file9101.h5"], val)
    sdoc = C.load_split_manifest(mp, msha, ex)
    _, vv = C.build_split_views(root, sdoc, FX.SYN_HOLDOUT, ex)
    edoc = E.build_manifest(E.compute_records(vv, 1.0),
                            {"validation": sdoc["validation"]}, {"cases": []},
                            {"p": 1})
    (fx / "elig.json").write_text(json.dumps(edoc))
    elig = M.eligibility_index(edoc)
    (fx / "meta.pkl").write_bytes(pickle.dumps(
        {"root": root, "mp": mp, "msha": msha, "expect": ex}))

    class Cfg:
        def __init__(self, seed, arm):
            self.seed, self.arm = seed, arm
    for i, (seed, arm) in enumerate(SEED_ARMS):
        model = ffr.build_model(init_seed=seed + 10 * i)
        ms = C.model_state_sha256(model)
        ck = fx / f"run_{seed}_{arm}" / "ckpt"
        ck.mkdir(parents=True)
        wp = ck / "weights_e001.pt"
        torch.save({"schema": EV.WEIGHTS_SCHEMA,
                    "model": {k: v.detach() for k, v in
                              model.state_dict().items()},
                    "meta": {"seed": seed, "arm": arm, "lambda": 0.0,
                             "epoch": 1, "step": 1, "model_state_sha256": ms,
                             "roles": ["best_psnr"]}}, wp)
        rows = M.validate(model, vv, elig, FX._unit_p4(), 1.0)
        doc = T.val_rows_doc(Cfg(seed, arm), 1, ms, rows)
        rp = ck / "val_e001.json"
        rp.write_text(json.dumps(doc))
        run = {"seed": seed, "arm": arm, "ckpt_dir": str(ck),
               "checkpoints": {"selected": {"epoch": 1,
                                            "model_state_sha256": ms}},
               "retained": {"best_psnr": {"name": wp.name,
                                          "sha256": A.sha256_file(wp)},
                            "val_rows_e001": {"name": rp.name,
                                              "sha256": A.sha256_file(rp)}},
               "rows": {1: doc},
               "epochs": {1: {"validation": M.summarise(rows)}}}
        sel = A.selected_checkpoint(run)
        sel.pop("rows")
        (fx / f"sel_{seed}_{arm}.json").write_text(json.dumps(sel))
        if i == 0:
            zf = [{"volume_id": r["volume_id"], "file": r["file"],
                   "slice_index": r["slice_index"],
                   "psnr": 20.0 + 0.5 * j}            # SYNTHETIC ZF values
                  for j, r in enumerate(rows)]
            (fx / "zf.json").write_text(json.dumps(zf))
    ok(True, "P0 synthetic fixtures built (6 checkpoints, 8 slices, 4 volumes)")

    def run_child(mode):
        out = tmp / f"out_{mode}"
        out.mkdir()
        p = subprocess.run([sys.executable, "-B", __file__, str(PIN), "child",
                            str(fx), str(out), mode], capture_output=True,
                           text=True, timeout=3000)
        return p, out
    p, out = run_child("pass")
    mf = out / "diag_jsh_manifest.json"
    ok(p.returncode == 0 and "PASS" in p.stdout and mf.is_file(),
       "P1 F1 prepare() end to end under the guard: PASS, manifest written",
       p.stdout.strip()[-90:] + p.stderr.strip()[-200:])
    if mf.is_file():
        man = json.loads(mf.read_text())
        side = (out / "diag_jsh_manifest.json.sha256").read_text().split()[0]
        ok(side == A.sha256_file(mf), "P2 manifest sidecar = manifest sha256")
        rp_ = man["reproduction"]["per_checkpoint"]
        ok(len(rp_) == 6 and all(v["slices"] == 8 and
                                 v["bitwise_equal_slices"] == 8
                                 for v in rp_.values()),
           "P3 all six checkpoints fully replayed (every slice)")

        def nulls(o, path=""):
            if o is None:
                return [path]
            if isinstance(o, dict):
                return [x for k, v in o.items() for x in nulls(v, f"{path}.{k}")]
            if isinstance(o, list):
                return [x for i, v in enumerate(o) for x in nulls(v, f"{path}[{i}]")]
            return []
        ok(not nulls(man), "P4 no null field in the manifest", str(nulls(man))[:80])
        cks = man["checkpoints"]
        ok(all(Path(c["rows_file"]).is_file() and Path(c["weights"]).is_file()
               and len(c["selection_evidence"]["per_epoch_psnr_sha256"]) == 64
               for c in cks.values()),
           "P5 resolvable row/weights paths and hashed selection evidence")
        ok(man["main_run_limits"]["status"].startswith("PENDING F2") and
           len(man["retained_slices"]) == 4 and
           set(man["subgroups"]) == {"H-D", "H-F"},
           "P6 main-run limits flagged pending F2; S-M parity, subgroups, "
           "retained slices recorded")
        used = A.tree_bytes(out)
        ok(sorted(x.name for x in out.iterdir()) ==
           ["diag_jsh_manifest.json", "diag_jsh_manifest.json.sha256",
            "f1.log"] and used <= man["preparation_limits"]["f1_budget_bytes"],
           "P7 only the three supported files written; within the budget",
           f"{used} bytes")
        er = man["preparation_limits"].get("early_refusal", {})
        ok(man["f1"]["version"] == "0.6" and er.get("t_boot_est_s", 0) > 0 and
           er.get("projected_s", 1e18) <= er.get("remaining_s", -1),
           "P15 manifest records F1 version 0.6 and the early-refusal "
           "projection (fits)", f"{man['f1']['version']} {er}")
        ins_ = man.get("inputs", {})
        ok(set(ins_) == set(F1.INPUT_KEYS) and
           all(isinstance(v["path"], str) and
               v["path"] == str(fx / f"{k}.input") for k, v in ins_.items()),
           "P16 bound-input paths recorded as str through record_inputs "
           "(pinned _bound() shape)", str(ins_)[:80])
        ok(man["sm_parity"]["label"].startswith("magnitude-spectrum error")
           and "NOT ADOPTED" in man["sm_parity"]["criterion"],
           "P7b S-M: frozen fallback label, criterion not adopted")
        ok(all(d["used"] <= d["free"] for d in
               man["preparation_limits"]["disk_checks"]) and
           len(man["preparation_limits"]["disk_checks"]) == 6 and
           man["preparation_limits"]["replay_limit_s"] > 0,
           "P8 disk checked after each of the six checkpoints")
    for mode, code, label in (
            ("disk", "DIAG_DISK", "P9 insufficient space refused before "
             "preparation"),
            ("deadline", "DIAG_DEADLINE", "P10 hard wall cap enforced"),
            ("tamper", "DIAG_REPRODUCTION_MISMATCH", "P11 a stored-row "
             "deviation in run 5 of 6 stops F1"),
            ("budget", "DIAG_DISK", "P11b manifest larger than the remaining "
             "budget refused before writing")):
        p, out = run_child(mode)
        ok(p.returncode == 1 and f"STOP {code}" in p.stdout and
           not (out / "diag_jsh_manifest.json").exists(),
           label + " (no manifest)", p.stdout.strip()[-60:])
    p, out = run_child("pathleak")
    flog = (out / "f1.log").read_text() if (out / "f1.log").exists() else ""
    ok(p.returncode == 1 and "STOP DIAG_MANIFEST_TYPE" in p.stdout and
       "early manifest fields" in flog and "STAGE 5" not in flog and
       not (out / "diag_jsh_manifest.json").exists(),
       "P17 raw Path in the binding: STOP DIAG_MANIFEST_TYPE at the early "
       "check (before STAGE 5, no manifest)", p.stdout.strip()[-70:])
    p, out = run_child("late_type")
    flog = (out / "f1.log").read_text() if (out / "f1.log").exists() else ""
    ok(p.returncode == 1 and "STOP DIAG_MANIFEST_TYPE" in p.stdout and
       "STAGE 2b" in flog and "STAGE 5" in flog and
       "manifest: TypeError" in flog and
       not (out / "diag_jsh_manifest.json").exists(),
       "P18 non-serialisable field generated later: STOP DIAG_MANIFEST_TYPE "
       "at the final check (after STAGE 5, no manifest)",
       p.stdout.strip()[-70:])
    import diag_jsh_core as K_
    import diag_jsh_f1_prepare as F1_
    good = {k: {"path": Path("/x") / k, "sha256": "a" * 64}
            for k in F1_.INPUT_KEYS}
    bad = []
    for label, mut in (("extra key", lambda d: d["zf"].update(size=1)),
                       ("path int", lambda d: d["zf"].update(path=7)),
                       ("short sha", lambda d: d["zf"].update(sha256="ab")),
                       ("missing", lambda d: d.pop("zf"))):
        d = {k: dict(v) for k, v in good.items()}
        mut(d)
        try:
            F1_.record_inputs(d, K_)
            bad.append(label)
        except K_.DiagError as exc:
            if exc.code != "DIAG_BINDING":
                bad.append(label)
    conv = F1_.record_inputs(good, K_)
    ok(not bad and all(isinstance(v["path"], str) for v in conv.values()),
       "P19 record_inputs: Path converted; extra key / non-path / short "
       "sha / missing input refused (DIAG_BINDING)", str(bad))
    p, out = run_child("projected")
    flog = (out / "f1.log").read_text() if (out / "f1.log").exists() else ""
    ok(p.returncode == 1 and "STOP DIAG_DEADLINE_PROJECTED" in p.stdout and
       "STAGE 4 projection" in flog and "STAGE 5" not in flog and
       not (out / "diag_jsh_manifest.json").exists(),
       "P10b projected time > remaining: STOP before the full replay (no "
       "STAGE 5, no manifest)", p.stdout.strip()[-70:])
    p, out = run_child("unbudgeted")
    ok(p.returncode == 77 and "unbudgeted" in p.stderr and
       not (out / "extra.txt").exists(),
       "P12b unregistered write inside OUT halts (77), nothing created")
    p, out = run_child("write_code")
    ok(p.returncode == 77 and "DIAG_WRITE_OUTSIDE" in p.stderr,
       "P12 diagnostic code protected after start() (write halts, 77)")
    # command-log cap: the script's own pipeline and size rule, extracted
    # verbatim (the host-only parts of the script are not run here)
    src = (CODE / "diag_jsh_f1_cmd.sh").read_text()
    pipe = 'head -c "$CMDLOG_MAX" | tee "$LOGF"'
    rule = '[[ $LSZ -ge 0 && $LSZ -lt $CMDLOG_MAX ]] || FAIL+=('
    ok(pipe in src and rule in src and
       "RC=${ps[0]}; HRC=${ps[1]}; TRC=${ps[2]}" in src and
       'FAIL+=("head rc $HRC")' in src and 'FAIL+=("tee rc $TRC")' in src and
       '--reserve-bytes "$((RES + CMDLOG_MAX))"' in src and
       'head -c "$REM" | tee -a "$LOGF"' in src and
       "TERM_LEAD=90" in src and "KILL_GRACE=60" in src,
       "P14a command: all three pipeline statuses checked, summary inside "
       "the remaining allowance, log allowance reserved for Python")
    frag = ("CMDLOG_MAX=1000; LOGF=" + str(tmp / "c.log") + "; FAIL=()\n"
            "python3 -c \"print('x' * 5000)\" 2>&1 | " + pipe + " >/dev/null\n"
            "LSZ=$(stat -c %s \"$LOGF\")\n" + rule +
            '"cap reached"); echo "${#FAIL[@]} $LSZ"')
    q = subprocess.run(["bash", "-c", frag], capture_output=True, text=True)
    ok(q.stdout.split() == ["1", "1000"],
       "P14b log over the allowance is cut at the cap and flagged STOP",
       q.stdout.strip())
    import shlex
    import time as _t

    def capped(behaviour):
        body = ("import signal, time\n" + ("signal.signal(signal.SIGTERM, "
                "signal.SIG_IGN)\n" if behaviour == "ignore" else "") +
                "time.sleep(30)")
        sh = (f"source {str(CODE / 'diag_jsh_f1_cmd.sh')!r}; TERM_LEAD=3; "
              f"KILL_GRACE=2; run_capped 8 {sys.executable} -c "
              f"{shlex.quote(body)}")
        t0 = _t.monotonic()
        q = subprocess.run(["bash", "-c", sh], capture_output=True, text=True,
                           timeout=60)
        return q.returncode, _t.monotonic() - t0
    rc, dt = capped("ignore")
    ok(rc == 137 and 6.5 <= dt < 8.0,
       "P14c scaled cap (WALL 8 s): TERM-ignoring child KILLed at WALL-1 s, "
       "before the cap", f"rc {rc}, {dt:.2f} s")
    rc, dt = capped("honour")
    ok(rc == 124 and 4.5 <= dt < 6.0,
       "P14d TERM-honouring child stops at WALL - TERM_LEAD", f"rc {rc}, "
       f"{dt:.2f} s")
    q = subprocess.run(["bash", "-c", f"source {str(CODE / 'diag_jsh_f1_cmd.sh')!r}"
                        "; echo SOURCED"], capture_output=True, text=True)
    ok(q.stdout.strip() == "SOURCED" and q.returncode == 0,
       "P14e sourcing the command defines functions only (runs nothing)")
    import shutil
    shutil.rmtree(tmp)
    ok(not tmp.exists(), "P13 fixture folder removed")
    print(f"RESULT {n_ok}/{n_ok + n_fail} checks passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "child":
        sys.exit(child(Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]))
    sys.exit(parent())
