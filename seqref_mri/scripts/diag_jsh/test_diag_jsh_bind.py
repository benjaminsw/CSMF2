# SEQREF-MRI-DIAGJSHBINDTEST v0.1 -- guarded production-binding test: F1 bind_from() under the strict write guard on a pinned fixture campaign
# LIFETIME: KEEP
# =============================================================================
# Runs the SAME binding code F1 uses in production (diag_jsh_f1_prepare
# v0.4 bind_from: chain -> completion -> backend -> bound inputs -> HEAD ->
# data -> ZF; then the six pinned load_run + selected checkpoints and the
# loaded-module rule) on a launched fixture campaign built by the pinned
# fixture code (_prd_layout + Campaign(fixture=True).run("new")), with the
# F1 strict write guard installed first (F1.start). Each case is its own
# process (the guard is process-wide).
# Differences from production (declared, test-only):
#   * the Campaign and stages are the pinned FIXTURE ones (fixture=True;
#     production uses fixture=False with C.Paths()); bind_production's thin
#     constructor itself is therefore not executed here;
#   * pin = the stand-in worktree HEAD (a68346d), not 7455da80;
#   * F1.V05LT and the bound repository = the stand-in worktree
#     (F.REPO_ROOT; production: paths.repo = /home/benjamin/CSMFII).
# Cases:
#   B1 success: binding fields, data, ZF, loaded-module rule; NO write
#      anywhere (campaign tree unchanged; OUT holds only f1.log); no halt
#   B1c the pinned load_run refuses a fixture-trained run (production
#      identity check in force): the six selected checkpoints cannot be
#      loaded from a fixture campaign, so that success path stays host-only
#      (covered on synthetic data by the F1 integration test, test double)
#   B2 negative control: a write into the campaign tree during binding ->
#      guard halt, exit 77, file not created
#   B3 HEAD != pin -> DIAG_PIN_MISMATCH;  B4 unapproved chain sha256 ->
#      DIAG_BINDING;  B5 backend changed after training -> refused by the
#      completion re-check first (epoch-0 records hold the backend
#      context): DIAG_NOT_COMPLETE
# The child process ends with os._exit(0) after reporting: the pinned
# fixture registers an atexit temp-folder cleanup that the strict guard
# would (correctly) halt; it is fixture housekeeping, not binding code.
# Usage: python -B test_diag_jsh_bind.py
# Changelog (v0.1, 2026-11-09): first guarded binding test.
# =============================================================================
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

CODE = Path(__file__).resolve().parent
PINWT = Path("/home/claude/v06lt_dev/pin_a68346d")
RES = []


def ok(cond, name, detail=""):
    RES.append(bool(cond))
    print(("ok   " if cond else "FAIL ") + name +
          (f"  [{detail}]" if detail else ""))


CHILD = r'''
import json, os, sys, traceback
from pathlib import Path
CODE, PINWT, OUT, CAMP, MODE = sys.argv[1:6]
sys.path[:0] = [PINWT, CODE]
import torch                              # registered backend (12/12) for
torch.set_num_threads(12)                 # the fixture campaign too, as on
torch.set_num_interop_threads(12)         # the production host
from seqref_mri.scripts import v05lt_fixtures as F
from seqref_mri.scripts import v05lt_launch as L
P = F._prd_layout(Path(CAMP))
cd = P["tmp"] / "camp"
L.Campaign(cd, P["st"], chain=P["chain"], fixture=True).run("new")
stages = P["stages"](root=cd)
chain = P["chain"]
if MODE == "badsha":
    chain = (chain[0], "0" * 64, chain[2])
camp = L.Campaign(cd, stages, chain=chain, fixture=True)


def tree(d):
    out = []
    for dp, dns, fns in os.walk(d):
        for n in sorted(dns + fns):
            q = os.path.join(dp, n)
            st = os.lstat(q)
            out.append([os.path.relpath(q, d), st.st_size, st.st_mtime_ns])
    return sorted(out)


before = tree(P["tmp"])
import diag_jsh_f1_prepare as F1
F1.V05LT = Path(F.REPO_ROOT)              # guard git allowance = the bound repo
import diag_jsh_adapter as A
import diag_jsh_core as K
Path(OUT).mkdir()
budget = F1.start(Path(OUT), A, 0)        # strict guard + budget + log FIRST
if MODE == "nobackend":
    torch.set_num_threads(2)              # backend changed after training
if MODE == "write":
    real = stages._bound
    def leaky():
        open(os.path.join(str(cd), "stray.txt"), "w").write("x")
        return real()
    stages._bound = leaky
pin = P["head"] if MODE != "badpin" else "0" * 40
try:
    ctx = F1.bind_from(camp, stages, F.REPO_ROOT, chain[0], chain[1], A, K,
                       pin=pin)          # repo = the pinned repository, as
                                         # paths.repo in production
    mods = ctx["loaded_modules"]()
    try:                                  # pinned load_run on a FIXTURE run
        ctx["load_selected"](A.SEED_ARMS[0])
        sel_code = None
    except K.DiagError as e:
        sel_code = e.code
    b = ctx["binding"]
    res = {"result": "PASS", "pin": b["pin"], "head": P["head"],
           "li": b["launch_inputs_sha256"], "li_expect": P["li_sha"],
           "ev": b["completion_evidence_sha256"], "backend": b["backend"],
           "n_mods": len(mods), "sel_code": sel_code,
           "n_data": len(ctx["data"]), "n_zf": len(ctx["zf_rows"])}
except K.DiagError as e:
    res = {"result": "STOP", "code": e.code}
except Exception as e:
    res = {"result": "UNEXPECTED", "err": f"{type(e).__name__}: {e}",
           "tb": traceback.format_exc()[-600:]}
res["tree_unchanged"] = tree(P["tmp"]) == before
res["out"] = sorted(os.listdir(OUT))
res["stray"] = os.path.lexists(os.path.join(str(cd), "stray.txt"))
os.write(1, ("RESJSON " + json.dumps(res) + "\n").encode())
os._exit(0)   # skip the fixture's atexit temp cleanup (not binding code)
'''


def case(mode: str):
    td = Path(tempfile.mkdtemp(prefix="bind_"))
    p = subprocess.run([sys.executable, "-B", "-c", CHILD, str(CODE),
                        str(PINWT), str(td / "out"), str(td / "c"), mode],
                       capture_output=True, text=True, timeout=900)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESJSON ")]
    res = json.loads(line[-1][8:]) if line else {}
    return p.returncode, res, p.stdout[-400:] + p.stderr[-800:], td


def main() -> int:
    t0 = time.monotonic()
    try:
        rc, r, tail, td = case("ok")
        ok(rc == 0 and r.get("result") == "PASS" and r["pin"] == r["head"]
           and r["li"] == r["li_expect"] and len(r["ev"]) == 64 and
           r["n_mods"] > 0 and r["n_data"] > 0 and r["n_zf"] > 0,
           "B1 bind_from on the pinned fixture campaign under the strict "
           "guard: chain, completion evidence, backend, bound inputs, HEAD = "
           "pin, data, ZF, loaded-module rule",
           json.dumps({k: r.get(k) for k in ("result", "code", "err",
                                             "n_mods")}))
        sc = r.get("sel_code")
        ok(sc == "DIAG_INPUT_INVALID",
           "B1c pinned load_run refuses a fixture-trained run "
           "(LOAD_RUN_IDENTITY: production identity check in force; the "
           "selected-checkpoint success path is host-only)", str(sc))
        ok(rc == 0 and r.get("tree_unchanged") and r.get("out") == ["f1.log"],
           "B1 no write: campaign tree unchanged; OUT holds only f1.log; no "
           "guard halt", f"rc={rc} out={r.get('out')}" + (
               "" if rc == 0 else " " + tail[-300:]))
        rc, r, tail, td = case("write")
        stray = list((td / "c").rglob("stray.txt"))
        ok(rc == 77 and not stray,
           "B2 negative control: a write into the campaign tree during "
           "binding halts the run (exit 77) before the file exists",
           f"rc={rc} stray={len(stray)}")
        for mode, want, lab in (("badpin", "DIAG_PIN_MISMATCH", "B3 HEAD != "
                                 "pin refused"),
                                ("badsha", "DIAG_BINDING", "B4 unapproved "
                                 "launch-chain sha256 refused"),
                                ("nobackend", "DIAG_NOT_COMPLETE", "B5 backend "
                                 "changed after training (threads 12 -> 2) "
                                 "refused -- caught first by the completion "
                                 "re-check (epoch-0 context)")):
            rc, r, tail, td = case(mode)
            ok(rc == 0 and r.get("result") == "STOP" and
               r.get("code") == want and r.get("tree_unchanged"),
               f"{lab} ({want}); nothing written", f"rc={rc} {r.get('code')}"
               + ("" if rc == 0 else " " + tail[-300:]))
    except Exception:                          # logged, counted as failure
        traceback.print_exc()
        RES.append(False)
    n = sum(RES)
    print(f"RESULT {n}/{len(RES)} checks passed "
          f"({time.monotonic() - t0:.0f} s)")
    return 0 if n == len(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
