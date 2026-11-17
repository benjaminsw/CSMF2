# SEQREF-DUNC-X2B v0.2 -- dunc_x2_bank.py: byte regeneration of the primary Z_DIAG bank (224 X2; plan SEQREF-DUNC-X2 v0.2)
# LIFETIME: DIAGNOSTIC
"""Regenerates the primary Z_DIAG bank from the frozen recipe, independently of
V05LT code (NumPy + standard library only), and compares its SHA-256 with the
registered value. Writes x2_bank.log, x2_bank.log.sha256 and RESULT.txt into a
NEW directory; the bank is never written to disk.

Result rule: the log holds the CHECKS verdict only. The final X2_RESULT is
decided after the log is closed, its sidecar written and re-verified, and
RESULT.txt written and read back. It is printed last on stdout and stored in
RESULT.txt. PASS is impossible unless all evidence files exist and verify.
Validity stop: on a primary hash mismatch the mismatching hash and the two
diagnostic hashes are logged and the run stops; no control is generated.

Production mode : no arguments. Constants below are used; nothing is overridable.
Self-test mode  : --selftest with ALL overrides (mock repo, mock seed != 0, mock
                  expected hash). Output is marked NOT X2 EVIDENCE. The
                  interpreter path/hash check is SKIPPED in self-test mode only.
                  --fault {log_open,sidecar,result} injects one evidence-write
                  failure; accepted in self-test mode only (test hook, not a
                  fallback).

Exit codes: 0 PASS, 1 FAIL or ABORTED. No fallback, placeholder or dummy value.
Changelog:
  v0.2 (2026-11-17) review fixes: (1) log-file creation failure is caught and
    reported as FAIL with the result line; (2) final PASS depends on the log
    sidecar being written and re-verified and on RESULT.txt; the log carries
    X2_CHECKS only; (3) primary mismatch is an enforced validity stop (no
    controls); self-test fault hook added for the three evidence-write paths.
  v0.1 (2026-11-17) created from plan X2 v0.2 (R1 controls, R2 diagnostics).
Update summary: recipe, constants, pre-checks and controls unchanged from v0.1.
"""
import argparse
import hashlib
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone

VERSION = "SEQREF-DUNC-X2B v0.2"
PIN = "7455da8027f33a6cca4dbf72f04e72420e938f45"
REPO = "/home/benjamin/CSMFII"
INTERPRETER = REPO + "/seqref_mri/.venv_mri/bin/python"
INTERPRETER_SHA256 = "e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f"
PYTHON_VERSION = "3.12.3"
NUMPY_VERSION = "1.26.4"
OUT_BASE = REPO + "/v06lt_evidence/DIAG_UNC"
SEED = 0
SHAPE = (128, 13824)
REGISTERED_SHA256 = "f8a6a34b6b474286359e4c99341de256b62bc0dff1e6bc8cf52e72f0461dc8d6"

log = logging.getLogger("dunc_x2")


class Abort(Exception):
    pass


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise Abort(f"git {' '.join(args)} failed (exit {r.returncode}): {r.stderr.strip()}")
    return r.stdout


def prechecks(cfg):
    """Everything here runs BEFORE any file or directory is created."""
    facts = {}
    if sys.byteorder != "little":
        raise Abort(f"byte order is {sys.byteorder}, little-endian required")
    pyv = ".".join(str(v) for v in sys.version_info[:3])
    if pyv != PYTHON_VERSION:
        raise Abort(f"Python {pyv} != {PYTHON_VERSION}")
    try:
        import numpy as np
    except Exception as exc:  # logged by caller
        raise Abort(f"NumPy import failed: {exc!r}")
    if np.__version__ != NUMPY_VERSION:
        raise Abort(f"NumPy {np.__version__} != {NUMPY_VERSION}")
    head = git(cfg["repo"], "rev-parse", "HEAD").strip()
    if head != cfg["pin"]:
        raise Abort(f"HEAD {head} is not the pin {cfg['pin']}")
    if git(cfg["repo"], "status", "--porcelain").strip():
        raise Abort("Git status is not clean")
    if cfg["selftest"]:
        facts["interpreter_check"] = "SKIPPED (self-test mode)"
    else:
        if os.path.abspath(sys.executable) != INTERPRETER:
            raise Abort(f"interpreter {sys.executable} is not {INTERPRETER}")
        real = os.path.realpath(sys.executable)
        got = file_sha256(real)
        if got != INTERPRETER_SHA256:
            raise Abort(f"interpreter binary sha256 {got} != registered")
        facts["interpreter_check"] = f"OK {real} {got}"
    if not os.path.isdir(cfg["out_base"]):
        raise Abort(f"output base directory missing: {cfg['out_base']}")
    facts.update(python=pyv, numpy=np.__version__, head=head)
    return np, facts


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def fault(cfg, name):
    if cfg.get("fault") == name:
        raise OSError(f"injected fault: {name} (self-test only)")


def checks(cfg, np, facts, script, script_sha):
    """All logged checks. Returns True if every check passed."""
    tag = "X2_SELFTEST_CHECKS" if cfg["selftest"] else "X2_CHECKS"
    log.info("%s", VERSION)
    if cfg["selftest"]:
        log.warning("SELF-TEST MODE -- mock repo, mock seed, mock expected hash -- NOT X2 EVIDENCE")
    log.info("start_utc %s", utc())
    log.info("script %s sha256 %s", script, script_sha)
    log.info("python %s | numpy %s | byteorder %s", facts["python"], facts["numpy"], sys.byteorder)
    log.info("interpreter check: %s", facts["interpreter_check"])
    log.info("HEAD %s (pin, clean)", facts["head"])
    log.info("recipe: Generator(PCG64(%d)).standard_normal(size=%s, dtype=float64).astype(float32); "
             "identity = sha256(bank.tobytes(order='C'))", cfg["seed"], SHAPE)
    log.info("expected sha256 %s", cfg["expected"])

    # --- primary: one generator, one call, no other draw
    g = np.random.Generator(np.random.PCG64(cfg["seed"]))
    a64 = g.standard_normal(size=SHAPE, dtype=np.float64)
    del g
    bank = a64.astype(np.float32)
    if bank.shape != SHAPE or bank.dtype != np.float32:
        log.error("primary bank has shape %s dtype %s", bank.shape, bank.dtype)
        raise RuntimeError("primary bank shape/dtype")
    if not bool(np.isfinite(bank).all()):
        log.error("primary bank contains non-finite values")
        raise RuntimeError("primary bank non-finite")
    h = sha_bytes(bank.tobytes(order="C"))
    log.info("primary sha256 %s", h)
    # --- R2 diagnostics (no draw; logged on match and on mismatch)
    log.info("diag float64 pre-cast sha256 %s", sha_bytes(a64.tobytes(order="C")))
    log.info("diag first row sha256 %s", sha_bytes(bank[0].tobytes(order="C")))
    del a64
    if h != cfg["expected"]:
        log.error("PRIMARY: MISMATCH -- VALIDITY STOP: no control generated, no retry, no alternative recipe")
        log.info("end_utc %s", utc())
        log.error("%s: FAIL", tag)
        return False
    log.info("PRIMARY: MATCH")

    # --- R1 negative controls: own generators, only after a primary match
    ok = True
    c_seed = np.random.Generator(np.random.PCG64(cfg["seed"] + 1)).standard_normal(
        size=SHAPE, dtype=np.float64).astype(np.float32)
    c_f32 = np.random.Generator(np.random.PCG64(cfg["seed"])).standard_normal(
        size=SHAPE, dtype=np.float32)
    controls = [
        (f"control (a) seed {cfg['seed'] + 1}", sha_bytes(c_seed.tobytes(order="C"))),
        ("control (b) direct float32 draw", sha_bytes(c_f32.tobytes(order="C"))),
        ("control (c) transposed byte order", sha_bytes(bank.T.tobytes(order="C"))),
    ]
    if cfg.get("control_equal_test"):
        controls.append(("control (t) self-test duplicate of the primary", h))
    for name, ch in controls:
        if ch == cfg["expected"]:
            log.error("%s sha256 %s EQUALS the expected hash -- comparison not discriminating", name, ch)
            ok = False
        else:
            log.info("%s sha256 %s differs (as required)", name, ch)

    head2 = git(cfg["repo"], "rev-parse", "HEAD").strip()
    dirty = git(cfg["repo"], "status", "--porcelain").strip()
    if head2 != cfg["pin"] or dirty:
        log.error("post-check: HEAD %s / clean=%s", head2, not dirty)
        ok = False
    else:
        log.info("post-check: HEAD is the pin, status clean")
    log.info("end_utc %s", utc())
    if ok:
        log.info("%s: PASS (final X2_RESULT is decided after evidence verification; see RESULT.txt)", tag)
    else:
        log.error("%s: FAIL", tag)
    return ok


def run(cfg):
    np, facts = prechecks(cfg)
    script = os.path.realpath(__file__)
    script_sha = file_sha256(script)
    tag = "X2_SELFTEST_RESULT" if cfg["selftest"] else "X2_RESULT"

    out_dir = os.path.join(cfg["out_base"], "x2_bank_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    try:
        os.mkdir(out_dir)  # fails if it exists; an existing directory is never touched
    except OSError as exc:
        raise Abort(f"cannot create new output directory {out_dir}: {exc!r}")
    print(f"X2_OUT={out_dir}")
    log_path = os.path.join(out_dir, "x2_bank.log")
    side_path = log_path + ".sha256"
    result_path = os.path.join(out_dir, "RESULT.txt")
    problems = []

    # 1. log file (creation failure is caught: FAIL, not a traceback)
    fh = None
    try:
        fault(cfg, "log_open")
        fh = logging.FileHandler(log_path, mode="x", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        log.addHandler(fh)
    except Exception as exc:
        log.error("cannot create the log file %s: %r", log_path, exc)
        problems.append("log file not created")

    # 2. checks (only with a working log file)
    checks_ok = False
    if fh is not None:
        try:
            checks_ok = checks(cfg, np, facts, script, script_sha)
        except Exception as exc:
            log.error("unexpected error: %r", exc)
            log.error("%s: FAIL", "X2_SELFTEST_CHECKS" if cfg["selftest"] else "X2_CHECKS")
        finally:
            log.removeHandler(fh)
            fh.close()
        if not checks_ok:
            problems.append("checks failed (see x2_bank.log)")

    # 3. sidecar: written, then re-verified against the closed log
    log_sha = None
    if fh is not None:
        try:
            log_sha = file_sha256(log_path)
            fault(cfg, "sidecar")
            with open(side_path, "x", encoding="utf-8") as sf:
                sf.write(f"{log_sha}  x2_bank.log\n")
                sf.flush()
                os.fsync(sf.fileno())
            with open(side_path, "r", encoding="utf-8") as sf:
                parts = sf.read().split()
            if parts != [log_sha, "x2_bank.log"] or file_sha256(log_path) != log_sha:
                raise OSError("sidecar does not verify against the log")
        except Exception as exc:
            log.error("log sidecar failure: %r", exc)
            problems.append("log sidecar not written or not verified")

    # 4. final result: RESULT.txt written and read back; only then PASS
    final = "PASS" if not problems else "FAIL (nothing is deleted; do not rerun)"
    text = (f"{tag}: {final}\n{VERSION}\nutc {utc()}\nlog_sha256 {log_sha}\n"
            + "".join(f"problem: {p}\n" for p in problems))
    try:
        fault(cfg, "result")
        with open(result_path, "x", encoding="utf-8") as rf:
            rf.write(text)
            rf.flush()
            os.fsync(rf.fileno())
        with open(result_path, "r", encoding="utf-8") as rf:
            if rf.read() != text:
                raise OSError("RESULT.txt read-back differs")
    except Exception as exc:
        log.error("cannot write RESULT.txt: %r", exc)
        problems.append("RESULT.txt not written or not verified")
        final = "FAIL (nothing is deleted; do not rerun)"
    for p in problems:
        log.error("problem: %s", p)
    (log.info if not problems else log.error)("%s: %s", tag, final)
    return 0 if not problems else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--repo")
    ap.add_argument("--pin")
    ap.add_argument("--out-base")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--expected")
    ap.add_argument("--fault", choices=["log_open", "sidecar", "result"])
    ap.add_argument("--control-equal-test", action="store_true")
    a = ap.parse_args()
    over = [a.repo, a.pin, a.out_base, a.seed, a.expected]
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")
    try:
        if a.selftest:
            if any(v is None for v in over):
                raise Abort("--selftest needs --repo --pin --out-base --seed --expected")
            if a.seed == SEED:
                raise Abort("self-test must not use the registered seed")
            if a.expected == REGISTERED_SHA256:
                raise Abort("self-test must not use the registered hash")
            cfg = dict(selftest=True, repo=a.repo, pin=a.pin, out_base=a.out_base,
                       seed=a.seed, expected=a.expected, fault=a.fault,
                       control_equal_test=a.control_equal_test)
        else:
            if any(v is not None for v in over) or a.fault or a.control_equal_test:
                raise Abort("overrides and test hooks are refused in production mode")
            cfg = dict(selftest=False, repo=REPO, pin=PIN, out_base=OUT_BASE,
                       seed=SEED, expected=REGISTERED_SHA256, fault=None,
                       control_equal_test=False)
        rc = run(cfg)
    except Abort as exc:
        log.error("X2_ABORT: %s", exc)
        log.error("X2_RESULT: ABORTED (nothing is deleted; do not rerun)")
        rc = 1
    except Exception as exc:
        log.error("unexpected error outside the run: %r", exc)
        log.error("X2_RESULT: FAIL (nothing is deleted; do not rerun)")
        rc = 1
    print(f"X2_EXIT={rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
