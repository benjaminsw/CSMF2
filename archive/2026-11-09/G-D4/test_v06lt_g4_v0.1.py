# SEQREF-MRI-V06LTG4T v0.1 -- G-D4 tests: driver analyse()/main() on a completed pinned synthetic campaign; wrapper checks
# LIFETIME: DIAGNOSTIC
# =============================================================================
# Part A (driver, pinned stand-in worktree pin_a68346d on sys.path):
#   A1 success: analyse() == pinned run_analysis; output and sidecar verified;
#      bytes identical to an independent pinned run (determinism)
#   A2 configuration other than the approved hash -> STATS_CONFIG_UNAPPROVED
#   A3 sidecar tampered after a run -> G4_OUTPUT_MISMATCH (re-check logic)
#   A4 existing output -> ARTIFACT_EXISTS (write-once)
#   A5 production=True with fixture layout -> ANALYSIS_FIXTURE_LAYOUT;
#      fixture configuration in production -> STATS_FIXTURE_CONFIG
#   A6 no numeric value in any log record (root at DEBUG) or on stdout/stderr
#      during the analysis; NUMERIC detector itself tested; a leaked value is
#      withheld from the console handler and refused (G4_LOG_VALUES)
#   A7 main() success path: bind/finish replaced by TEST DOUBLES returning the
#      synthetic context (production bind needs the host campaign); stdout =
#      exactly one RESULT line with no decimal value; exit 0; typed refusal
#      -> exit 1; unexpected error -> exit 2 and logged
# Part B (wrapper, copy with V05/PIN/PY/DRIVER_SHA substituted, temporary git
#   repo, STUB driver = test double writing the two outputs):
#   B1 PASS; B2 extra file written -> exit 2 MTIME_CHECK FAIL; B3 symlinked
#   analysis folder -> exit 2; B4 dangling-symlink output -> exit 2;
#   B5 .tmp leftover -> exit 2; B6 driver hash mismatch -> exit 2;
#   B7 failing sort/comm (PATH shim) -> exit 2 MTIME_CHECK ERROR;
#   B8 refusal (stub exit 1) -> exit 1; B9 sidecar mismatch -> exit 2
# Changelog (v0.1, 2026-11-09): first G-D4 test set.
# =============================================================================
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN = Path("/home/claude/v06lt_dev/pin_a68346d")
sys.path.insert(0, str(PIN))
sys.path.insert(0, str(HERE))
import v06lt_v05lt_analysis_g4_driver as D            # noqa: E402
from seqref_mri.scripts import v05lt_common as C      # noqa: E402
from seqref_mri.scripts import v05lt_analysis as A    # noqa: E402
from seqref_mri.scripts import v05lt_fixtures as F    # noqa: E402

RES = []
log = logging.getLogger("g4test")


def ok(cond, name, detail=""):
    RES.append((bool(cond), name, detail))
    print(("PASS " if cond else "FAIL ") + name + (f" [{detail}]" if detail and not cond else ""))


def code_of(fn):
    try:
        fn()
    except C.V05LTError as e:
        return e.code
    except Exception as e:                     # logged, reported as failure
        log.error("unexpected %s", traceback.format_exc())
        return f"UNEXPECTED {type(e).__name__}"
    return None


def setup(td: Path):
    camp = F._syn_campaign(td / "c")
    lay = camp["layout"]
    an = td / "c" / "camp" / "analysis"
    an.mkdir()
    lay["difficult_sha256"] = A.select_difficult(lay, lay["difficult"], n=1)
    cp = an / "config.json"
    csha = A.write_config(lay, cp, B=200, fixture_override=True)
    return camp, lay, an, cp, csha


def part_a(td: Path):
    camp, lay, an, cp, csha = setup(td)
    # A1 + A6
    o1 = an / "v05lt_analysis.json"
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        r = D.analyse(lay, cp, csha, o1, production=False)
    ref = an / "ref.json"
    sref = A.run_analysis(lay, cp, csha, ref)
    side = Path(str(o1) + ".sha256").read_text()
    ok(r["sha256"] == hashlib.sha256(o1.read_bytes()).hexdigest() and
       side == f"{r['sha256']}  {o1.name}\n" and r["bytes"] == o1.stat().st_size,
       "A1 success: output sha, sidecar and size verified")
    ok(sref == r["sha256"] and ref.read_bytes() == o1.read_bytes(),
       "A1 identical bytes to an independent pinned run_analysis")
    ok(not [x for x in r["log_records"] if D.NUMERIC.search(x[2])],
       "A6 no numeric value in log records during the analysis",
       f"{len(r['log_records'])} records")
    ok(not D.NUMERIC.search(out.getvalue() + err.getvalue()),
       "A6 nothing numeric on stdout/stderr during the analysis",
       repr((out.getvalue() + err.getvalue())[:200]))
    ok(D.NUMERIC.search("psnr 31.25") and D.NUMERIC.search("d=-0.5") and
       not D.NUMERIC.search("v0.2 e012 n=296 sha 3a.b"),
       "A6 NUMERIC detects decimals, ignores counts and version tags")
    # A6 detector applied by analyse(): a decimal logged during the run
    real = A.run_analysis

    def leaky(*a, **k):
        logging.getLogger("seqref_mri.x").debug("value 0.731")
        return real(*a, **k)
    A.run_analysis = leaky
    cons = io.StringIO()
    ch = logging.StreamHandler(cons)
    ch.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(ch)
    try:
        c = code_of(lambda: D.analyse(lay, cp, csha, an / "leak.json",
                                      production=False))
    finally:
        A.run_analysis = real
        logging.getLogger().removeHandler(ch)
    ok(c == "G4_LOG_VALUES", "A6 a DEBUG record with a value -> G4_LOG_VALUES", c)
    ok("0.731" not in cons.getvalue() and "G4_LOG_VALUES" in cons.getvalue(),
       "A6 the value is withheld from the console handler", cons.getvalue())
    # A2
    c = code_of(lambda: D.analyse(lay, cp, "0" * 64, an / "a2.json",
                                  production=False))
    ok(c == "STATS_CONFIG_UNAPPROVED", "A2 unapproved configuration refused", c)
    ok(not (an / "a2.json").exists(), "A2 no output after refusal")
    # A3 sidecar re-check (tamper by wrapping run_analysis)
    def tamper(layout, cfg, sha, o, **k):
        s = real(layout, cfg, sha, o, **k)
        p = Path(str(o) + ".sha256")
        os.chmod(p, 0o644)
        p.write_text("f" * 64 + "  " + Path(o).name + "\n")
        return s
    A.run_analysis = tamper
    try:
        c = code_of(lambda: D.analyse(lay, cp, csha, an / "a3.json",
                                      production=False))
    finally:
        A.run_analysis = real
    ok(c == "G4_OUTPUT_MISMATCH", "A3 sidecar mismatch detected", c)
    # A4
    c = code_of(lambda: D.analyse(lay, cp, csha, o1, production=False))
    ok(c == "ARTIFACT_EXISTS", "A4 existing output refused (write-once)", c)
    # A5
    c = code_of(lambda: D.analyse(lay, cp, csha, an / "a5.json",
                                  production=True))
    ok(c == "ANALYSIS_FIXTURE_LAYOUT", "A5 fixture layout refused in production", c)
    c = code_of(lambda: A.write_config(lay, an / "c5.json", production=True,
                                       B=200, fixture_override=True))
    ok(c == "STATS_FIXTURE_CONFIG", "A5 fixture configuration refused in production", c)
    # A7 main() with TEST DOUBLES for bind/finish (production binding is host-only)
    rel = "out7/v05lt_analysis.json"
    (td / "r7" / "out7").mkdir(parents=True)

    class P:
        repo = td / "r7"
    ctx = {"paths": P, "layout": lay, "config": cp, "ev_sha": "e" * 64,
           "out_dir": td / "r7" / "out7"}
    rb, rf, ra = D.bind, D.finish, D.analyse
    D.bind = lambda *a: ctx
    D.finish = lambda c: None
    D.analyse = lambda la, cfg, sha, o, production: ra(la, cfg, sha, o,
                                                       production=False)
    try:
        def run(argv):
            o = io.StringIO()
            with contextlib.redirect_stdout(o):
                rc = D.main(argv)
            return rc, o.getvalue()
        args = ["chain", "c" * 64, csha, "1" * 64, "2" * 64, "3" * 64, rel]
        rc, so = run(args)
        lines = so.strip().splitlines()
        ok(rc == 0 and len(lines) == 1 and lines[0].startswith(
            "RESULT V05LT_ANALYSIS PASS analysis_sha256=") and
           "scope=validation_only" in lines[0] and not D.NUMERIC.search(so),
           "A7 main success: one RESULT line, no decimal values, exit 0", so)
        rc, so = run(args)
        ok(rc == 1 and so.strip() == "RESULT V05LT_ANALYSIS REFUSED ARTIFACT_EXISTS",
           "A7 main typed refusal -> exit 1", so)
        rc, so = run(args[:6] + ["elsewhere/v05lt_analysis.json"])
        ok(rc == 1 and "G4_OUTPUT_PATH" in so, "A7 output outside the approved folder refused", so)
        D.bind = lambda *a: 1 / 0
        rc, so = run(args)
        ok(rc == 2 and "FAILED UNEXPECTED ZeroDivisionError" in so,
           "A7 unexpected error -> exit 2 (logged)", so)
    finally:
        D.bind, D.finish, D.analyse = rb, rf, ra
        logging.getLogger().handlers.clear()


STUB = '''import sys, hashlib, os
from pathlib import Path
mode = os.environ.get("G4_STUB", "pass")
out = Path(sys.argv[7])
if mode == "refuse":
    print("RESULT V05LT_ANALYSIS REFUSED STATS_CONFIG_UNAPPROVED"); sys.exit(1)
b = b'{"x": 1}\\n'
out.write_bytes(b)
s = hashlib.sha256(b).hexdigest() if mode != "badside" else "f" * 64
Path(str(out) + ".sha256").write_text(s + "  " + out.name + "\\n")
if mode == "extra":
    Path("seqref_mri/stray.txt").write_text("x")
print("RESULT V05LT_ANALYSIS PASS"); sys.exit(0)
'''


def wrapper_env(td: Path, mode="pass", shim=None):
    v05 = td / "V05"
    sh = HERE / "v06lt_v05lt_analysis_g4_v0.2.sh"
    if not v05.exists():
        (v05 / "seqref_mri/results/v05lt/analysis").mkdir(parents=True)
        (v05 / "v06lt_evidence/V05LT_analysis").mkdir(parents=True)
        (v05 / "seqref_mri/keep.txt").write_text("k")
        g = ["git", "-C", str(v05)]
        subprocess.run(g + ["init", "-q"], check=True)
        subprocess.run(g + ["add", "-A"], check=True)
        subprocess.run(g + ["-c", "user.email=t@t", "-c", "user.name=t",
                            "commit", "-qm", "x"], check=True)
        drv = v05 / "v06lt_evidence/V05LT_analysis/v06lt_v05lt_analysis_g4_driver.py"
        drv.write_text(STUB)
    drv = v05 / "v06lt_evidence/V05LT_analysis/v06lt_v05lt_analysis_g4_driver.py"
    head = subprocess.run(["git", "-C", str(v05), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    s = sh.read_text()
    s = s.replace("V05=/home/benjamin/CSMFII", f"V05={v05}")
    s = s.replace("PIN=7455da8027f33a6cca4dbf72f04e72420e938f45", f"PIN={head}")
    s = s.replace("PY=$V05/seqref_mri/.venv_mri/bin/python", f"PY={sys.executable}")
    import re
    s = re.sub(r"DRIVER_SHA=\S+", "DRIVER_SHA=" + hashlib.sha256(drv.read_bytes()).hexdigest(), s)
    s = s.replace("MIN_FREE=$((2 * 1024 * 1024 * 1024))", "MIN_FREE=1")
    w = td / "w.sh"
    w.write_text(s)
    env = dict(os.environ, G4_STUB=mode)
    if shim:
        env["PATH"] = f"{shim}:{env['PATH']}"
    return v05, w, env


def run_w(w, env):
    p = subprocess.run(["bash", str(w), "chain", *["a" * 64] * 5, "300"],
                       capture_output=True, text=True, env=env, timeout=120)
    return p.returncode, p.stdout + p.stderr


def clean(v05):
    a = v05 / "seqref_mri/results/v05lt/analysis"
    for p in list(a.iterdir()):
        if p.is_symlink() or p.is_file():
            p.unlink()
    for p in (v05 / "v06lt_evidence/V05LT_analysis").glob("*.log"):
        p.unlink()
    (v05 / "seqref_mri/stray.txt").unlink(missing_ok=True)


def part_b(td: Path):
    v05, w, env = wrapper_env(td)
    a = v05 / "seqref_mri/results/v05lt/analysis"
    rc, o = run_w(w, env)
    ok(rc == 0 and "MTIME_CHECK OK" in o and "sidecar OK" in o and
       "GIT_STATUS unchanged" in o, "B1 wrapper PASS path", o[-400:])
    ok(not list((v05 / "v06lt_evidence/V05LT_analysis").glob(".g4_*")),
       "B1 marker and list files removed")
    rc, o = run_w(w, env)
    ok(rc == 2 and "output exists" in o, "B1b second run refused (write-once)", o[-200:])
    clean(v05)
    v05, w, env = wrapper_env(td, "extra")
    rc, o = run_w(w, env)
    ok(rc == 2 and "MTIME_CHECK FAIL" in o and "stray.txt" in o,
       "B2 extra write detected -> exit 2", o[-300:])
    clean(v05)
    real = td / "realdir"
    real.mkdir()
    a.rename(td / "adir_saved")
    a.symlink_to(real)
    rc, o = run_w(w, dict(env, G4_STUB="pass"))
    ok(rc == 2 and "symlink" in o, "B3 symlinked analysis folder refused", o[-200:])
    a.unlink()
    (td / "adir_saved").rename(a)
    (a / "v05lt_analysis.json").symlink_to(td / "nowhere")
    rc, o = run_w(w, dict(env, G4_STUB="pass"))
    ok(rc == 2 and "output exists" in o, "B4 dangling-symlink output refused", o[-200:])
    clean(v05)
    (a / ".tmp.v05lt_analysis.json.123").write_text("x")
    rc, o = run_w(w, dict(env, G4_STUB="pass"))
    ok(rc == 2 and "leftover" in o, "B5 pinned .tmp leftover refused", o[-200:])
    clean(v05)
    s = w.read_text()
    w2 = td / "w_bad.sh"
    import re
    w2.write_text(re.sub(r"DRIVER_SHA=[0-9a-f]{64}", "DRIVER_SHA=" + "0" * 64, s))
    rc, o = run_w(w2, dict(env, G4_STUB="pass"))
    ok(rc == 2 and "driver hash" in o, "B6 driver hash mismatch refused", o[-200:])
    for tool in ("sort", "comm"):
        shim = td / f"shim_{tool}"
        shim.mkdir(exist_ok=True)
        (shim / tool).write_text("#!/bin/sh\nexit 3\n")
        (shim / tool).chmod(0o755)
        v05, w, env = wrapper_env(td, "pass", shim)
        rc, o = run_w(w, env)
        ok(rc == 2 and "MTIME_CHECK ERROR" in o and tool in o,
           f"B7 failing {tool} -> exit 2, no PASS", o[-300:])
        clean(v05)
    v05, w, env = wrapper_env(td, "refuse")
    rc, o = run_w(w, env)
    ok(rc == 1 and "REFUSED STATS_CONFIG_UNAPPROVED" in o, "B8 refusal -> exit 1", o[-200:])
    clean(v05)
    v05, w, env = wrapper_env(td, "badside")
    rc, o = run_w(w, env)
    ok(rc == 2 and "sidecar MISMATCH" in o, "B9 sidecar mismatch -> exit 2", o[-200:])


def main():
    logging.basicConfig(level=logging.WARNING)
    for part in (part_a, part_b):
        with tempfile.TemporaryDirectory() as td:
            try:
                part(Path(td))
            except Exception:
                log.error("part %s crashed:\n%s", part.__name__, traceback.format_exc())
                RES.append((False, f"{part.__name__} crashed", ""))
    n = sum(r[0] for r in RES)
    print(f"G4 TESTS {n}/{len(RES)}")
    return 0 if n == len(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
