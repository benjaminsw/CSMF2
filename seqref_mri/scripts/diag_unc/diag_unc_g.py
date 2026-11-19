# SEQREF-DUNC-GR v0.2.1 -- diag_unc_g.py: DIAG-UNC step G (provenance, reproduction gates, stage-1 records, stage-2 analysis, outputs; plan SEQREF-DUNC-F2G v0.2.2)
# LIFETIME: KEEP
"""Step G of DIAG-UNC (223 v1.0 + AM1). One run, stages G0-G5 of plan F2G v0.2.2 (LudWiki 229 U58).

  G0  pre-checks: interpreter; production tree == pin, branch, clean; code == commit C1 == F2 manifest;
      manifest == commit C2 == --f2-sha; C2^ == C1; C2 adds the manifest only; F1 evidence == manifest;
      free space; no other project process; new run directory; watchdog started.
  G1  bind: every registered input re-hashed against the F2 manifest; 30 role files hashed; stored rows
      of every epoch of the six runs hashed, recorded and checked (216 §3 NLL identity); bank regenerated.
  G2  bootstrap index matrix regenerated in one call and verified against the F2 identity.
  G3  stage 1, per model state and midpoint slice: own values -> reproduction gate -> (only if passed)
      diag_unc_core.slice_stats -> one write-once record. One index per state, one final index.
  G4  stage 2, from the verified records and the stored rows only: D-GAP, run_registered, cross rows.
  G5  tables, figure data, manifest, RESULT.txt.

Modes (each needs its own recorded approval and run card):
  default            the G run (run directory v06lt_evidence/DIAG_UNC/g_*)
  --bind-only        G0-G2 only; no checkpoint load, no slice read, no decode. NON-SCIENTIFIC; not a G run;
                     registers nothing (run directory gbind_*; result BIND_OK).
  --recovery-of DIR  stage 2 and outputs again from the verified stage-1 records of a G run that stopped
                     in G4 or G5 with no typed code (run directory grec_*). Never after a reproduction,
                     bootstrap, provenance or resource stop; no checkpoint, no MRI file is opened; the
                     budget is what the original run left of the registered stop limit.

Two Git identities, verified independently (plan F2-4 item 6): the production working tree stays at the
pin on seqref-v05lt and is never switched; C1 (code) and C2 (manifest) are commit objects, never checked
out; <run_dir>/code holds exported copies. The F2 manifest is read from <run_dir>/code/manifest_f2.json
(one fixed place; no path argument).
Layout: <run_dir>/code/{diag_unc_g.py, diag_unc_f1.py, diag_unc_adapter.py, diag_unc_core.py,
diag_unc_stats.py, diag_unc_guard.py, manifest_f2.json}. Evidence: <run_dir>/g.log, g.log.sha256,
RESULT.txt, on a stop <run_dir>/stop_record.json (or stop_record_watchdog.json) and, through the write
guard, <run_dir>/out/{reserve/, records/, maps/, tables/, figdata/, results.json,
manifest_g.pending.json, manifest_g.json}.

Reuse of the archived F1 runner (imported after its hash is verified; not edited): Access, load_role,
build_state, slice_values, load_ledger, closure_gate, the gate/need/info helpers, Stop/Abort, small
utilities and the registered constants. G has its own pre-checks, binding (against the F2 manifest),
orchestration, stage 1, stage 2, outputs and closing. See the reuse audit of the review package.

Resource enforcement (plan §4): every file under out/ goes through WriteGuard (ceiling and reserve of the
F2 stop limit; cumulative ledger). g.log, RESULT.txt and the watchdog evidence live outside the guard
root, so that a stop caused by the disk limit can still be recorded; a reservation file of
EVIDENCE_RESERVE bytes is written through the guard at start and the log is capped below it, so their
bytes are counted in advance. Wall time: a predictive check before every step
(elapsed + the longest step of that kind so far must fit) and a watchdog thread with an absolute
deadline for a hung step: it writes <run_dir>/stop_record_watchdog.json and RESULT.txt (guaranteed
evidence, written first and directly), then flushes the log and writes its sidecar (best effort: a
step that hangs while holding the log lock blocks this part) and ends the process with os._exit(1).
A fail-safe thread ends the process LOCK_WAIT_S + FAILSAFE_S after the deadline at the latest.
Closing (the last writes: pending manifest or stop record, log sidecar, manifest, RESULT.txt) is bounded
too: a run that completed its stages starts closing only if CLOSE_BOUND_S of the limit is left, else it
stops with DIAG_UNC_RESOURCE_LIMIT; a closing that is not finished CLOSE_BOUND_S after it began is ended
by the same watchdog thread ("closing fail-safe"). If RESULT.txt was by then written, synced and read
back equal, the run's own result stands and the process ends at once with its exit code. Otherwise the
fail-safe writes its stop record and an ABORTED RESULT.txt; if a RESULT.txt exists that was not
verified, it writes RESULT_INVALID.txt instead, which declares that file and any manifest_g.json
invalid. So a PASS run always ends inside the limit; a stopped run ends at most CLOSE_BOUND_S after
its stop (plus the lock wait and fail-safe if blocked). A RESULT.txt is valid only if the run
directory holds neither stop_record_watchdog.json nor RESULT_INVALID.txt.

Exposure: the log and the terminal carry identities, counts, deviations, pass flags, times and bytes.
No NLL, PSNR, score, interval or label is logged. Scientific values exist only under out/.

Self-test mode: --selftest CONFIG.json (mock repository, synthetic data, stand-in model); refused for the
production repository or pin; outputs marked NOT G EVIDENCE. A test hook, not a fallback.
Exit codes: 0 PASS / BIND_OK, 1 FAIL or ABORTED, 3 HOLD_RESOURCE or HOLD_PROVENANCE.
Every failure is logged. No fallback, placeholder, mock or dummy return in production mode.

Changelog:
  v0.2.1 (2026-11-19) after the review of v0.2, closing path only: the run's own result stands at the closing deadline only if RESULT.txt was
       written, synced and read back equal (not if the file merely exists), and then the process ends at the deadline with no grace period;
       an unverified RESULT.txt is declared invalid by RESULT_INVALID.txt and the run is ABORTED.
  v0.2 (2026-11-19) after the review of v0.1 (LudWiki 230 U59), three findings:
       1. closing phase bounded: closing starts only if CLOSE_BOUND_S of the limit is left; a closing fail-safe ends a closing that hangs;
       2. stage-1 records are strict JSON: an undefined within-slice correlation is null plus an explicit "undefined" list, never NaN;
       3. every retained map and reference file is re-verified against its registered hash before the outputs are written.
  v0.1 (2026-11-18) first version, from plan SEQREF-DUNC-F2G v0.2.2.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

T0 = time.monotonic()
VERSION = "SEQREF-DUNC-GR v0.2.1"
PLAN = "SEQREF-DUNC-F2G v0.2.2 (8ab33005e2dac06ac6282a7d9e7c7921e63274afa9a70ad83f22618215337bdb)"
PIN = "7455da8027f33a6cca4dbf72f04e72420e938f45"
REPO = "/home/benjamin/CSMFII"
INTERPRETER = REPO + "/seqref_mri/.venv_mri/bin/python"
DIAG_DIR = "seqref_mri/scripts/diag_unc/"
ME, F1_NAME, MANIFEST_NAME = "diag_unc_g.py", "diag_unc_f1.py", "manifest_f2.json"
MODULES = ("diag_unc_adapter.py", "diag_unc_core.py", "diag_unc_stats.py", "diag_unc_guard.py", F1_NAME, ME)
F2_SCHEMA, REC_SCHEMA = "seqref-dunc-f2-manifest/1", "seqref-dunc-g-record/1"
GIB = 2 ** 30
OWNER = {"wall_s": 12 * 3600.0, "disk_bytes": 20 * GIB, "reserve_bytes": 64 * GIB}     # 223 §1 (224 U20), frozen ceilings
EVIDENCE_RESERVE = 4 * 2 ** 20            # counted in advance for g.log, its sidecar, RESULT.txt and the watchdog evidence
LOG_CAP = EVIDENCE_RESERVE - 2 ** 18
LOCK_WAIT_S = 30.0                        # watchdog: longest wait for the write lock (plan §4)
CLOSE_BOUND_S = 60.0                      # closing: longest time for the last writes of a run (plan §4; review of GR v0.1, finding 1)
FAILSAFE_S = 30.0                         # watchdog: the process ends at most this long after the lock wait, whatever blocks
N_FIXED = 4                               # 216 §6 / 223 §9
ROLE_SHORT = {"epoch3": "e3", "selected": "sel", "final": "fin", "epoch10": "e10", "epoch30": "e30"}
RUN_PREFIX = {"g": "g_", "bind": "gbind_", "recovery": "grec_"}
TAGS = {"g": "G", "bind": "G_BIND", "recovery": "G_RECOVERY"}
NO_RECOVERY = "not eligible for recovery"
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")

log = logging.getLogger("dunc_g")
WLOCK = threading.Lock()                  # held around every guarded write; the watchdog waits for it at most LOCK_WAIT_S
STATE = {"stage": "G0", "state": None, "slice": None, "mode": "g", "selftest": False, "run_dir": None, "stage1_complete": False,
         "durations": {}, "limit_s": None, "log_path": None, "closing": False, "closing_t": None, "done": False, "rc": 1, "result_verified": False}


class Early(Exception):
    """Failure before the F1 helpers are imported: result ABORTED (pre-check) unless a result is given."""

    def __init__(self, msg, result="ABORTED", code="none"):
        super().__init__(msg)
        self.result, self.code = result, code


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo, *args, binary=False):
    r = subprocess.run(["git", "--no-optional-locks", "-C", repo, *args], capture_output=True)
    if r.returncode != 0:
        raise Early("git %s failed (exit %d): %s" % (" ".join(args), r.returncode, r.stderr.decode(errors="replace").strip()))
    return r.stdout if binary else r.stdout.decode(errors="replace")


def elapsed():
    return time.monotonic() - T0


def jsonable(o):
    """Results as plain JSON: tuple keys joined with '|', arrays as lists, non-finite floats refused."""
    np = sys.modules["numpy"]
    if isinstance(o, dict):
        return {("|".join(str(x) for x in k) if isinstance(k, tuple) else str(k)): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        raise ValueError("non-finite value in a result")
    return o


def cjson(o):
    return json.dumps(o, sort_keys=True, indent=1, allow_nan=False).encode("utf-8")


# ---------------------------------------------------------------- wall time: predictive check and watchdog (plan §4)
def tick(F, kind):
    """Before a step of this kind: stop if the elapsed time plus the longest such step so far exceeds the limit."""
    worst = STATE["durations"].get(kind, 0.0)
    if elapsed() + worst > STATE["limit_s"]:
        raise F.Stop("ABORTED", "DIAG_UNC_RESOURCE_LIMIT", "wall time: elapsed %.1f s + longest %s step %.1f s exceeds the registered stop limit %.3f s "
                     "(predictive check; %s)" % (elapsed(), kind, worst, STATE["limit_s"], NO_RECOVERY))
    if STATE["log_path"] and os.path.getsize(STATE["log_path"]) > LOG_CAP:
        raise F.Stop("ABORTED", "DIAG_UNC_RESOURCE_LIMIT", "the log exceeds its pre-counted size %d bytes (%s)" % (LOG_CAP, NO_RECOVERY))
    return time.monotonic()


def tock(kind, t):
    d = time.monotonic() - t
    STATE["durations"][kind] = max(STATE["durations"].get(kind, 0.0), d)
    return d


def _direct_write(path, data):
    """Write-once file outside the guard root (bytes counted in advance by the evidence reservation)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def begin_closing():
    if not STATE["closing"]:
        STATE["closing_t"] = time.monotonic()
        STATE["closing"] = True


def watchdog(limit_s, lock_wait_s, close_bound_s):
    """Absolute deadlines for a hung step and for a hung closing. Runs in a daemon thread; ends the process when one is passed."""
    while True:
        if STATE["done"]:
            return
        closing = STATE["closing"]
        deadline = (STATE["closing_t"] + close_bound_s) if closing else (T0 + limit_s)
        left = deadline - time.monotonic()
        if left > 0:
            time.sleep(min(left, 0.5))
            continue
        if closing and STATE["result_verified"]:             # RESULT.txt written, synced and read back equal: the run's own result stands;
            sys.stdout.flush()                               # the process ends at the closing deadline with its exit code (no lock wait, no grace)
            os._exit(STATE["rc"])
        cancel = threading.Event()

        def failsafe(c=cancel):                              # whatever blocks below (a held log lock, a dead file system): the process still ends
            if not c.wait(lock_wait_s + FAILSAFE_S):
                os._exit(1)
        threading.Thread(target=failsafe, daemon=True, name="dunc-watchdog-failsafe").start()
        got = WLOCK.acquire(timeout=lock_wait_s)
        if STATE["done"] or STATE["closing"] != closing:     # the run finished, or it reached its closing meanwhile: that phase has its own deadline
            cancel.set()
            if got:
                WLOCK.release()
            continue
        break
    mechanism = "closing fail-safe" if closing else "watchdog"
    result_path = os.path.join(STATE["run_dir"], "RESULT.txt")
    if closing and STATE["result_verified"]:                 # verified while this thread waited for the lock
        sys.stdout.flush()
        os._exit(STATE["rc"])
    unverified = os.path.exists(result_path)                 # a RESULT.txt that exists but was not verified proves nothing
    try:
        run_dir, tag = STATE["run_dir"], TAGS[STATE["mode"]] + ("_SELFTEST" if STATE["selftest"] else "")
        parts = []
        out_dir = os.path.join(run_dir, "out")
        for dp, _d, names in os.walk(out_dir) if os.path.isdir(out_dir) else ():
            parts += [os.path.relpath(os.path.join(dp, n), run_dir) for n in names if n.endswith(".dunc-part")]
        rec = {"version": VERSION, "utc": utc(), "selftest": STATE["selftest"], "mode": STATE["mode"], "result": "ABORTED",
               "typed_code": "DIAG_UNC_RESOURCE_LIMIT", "mechanism": mechanism,
               "reason": ("the closing writes were not finished %.0f s after closing began" % close_bound_s) if closing
               else "the registered wall-time stop limit was reached inside a step",
               "limit_s": limit_s, "close_bound_s": close_bound_s, "closing_elapsed_s": (time.monotonic() - STATE["closing_t"]) if closing else None,
               "main_stop_record_present": os.path.exists(os.path.join(run_dir, "stop_record.json")), "result_txt_present_unverified": unverified,
               "elapsed_s": elapsed(), "stage_reached": STATE["stage"], "state": STATE["state"], "slice": STATE["slice"],
               "stage1_complete": STATE["stage1_complete"], "write_lock_obtained": bool(got), "lock_wait_s": lock_wait_s,
               "unfinished_part_files": sorted(parts), "recovery": NO_RECOVERY}
        problems = []
        try:                                                 # 1. the stop record (guaranteed part of the evidence)
            data = cjson(rec)
            p = os.path.join(run_dir, "stop_record_watchdog.json")
            _direct_write(p, data)
            _direct_write(p + ".sha256", ("%s  stop_record_watchdog.json\n" % sha_bytes(data)).encode())
        except Exception as exc:                             # nothing can be raised to anyone here: record what failed and go on
            problems.append("stop record: %r" % (exc,))
        try:                                                 # 2. RESULT.txt (guaranteed part); it cannot name the log hash, see g.log.sha256
            text = ("%s_RESULT: ABORTED (nothing is deleted; do not rerun)\n%s\nmode %s\ntyped_code DIAG_UNC_RESOURCE_LIMIT\nreason %s: %s at %s; %s\nutc %s\n"
                    "log_sha256 see g.log.sha256 (written by the watchdog after this file, if the log could be flushed)\n"
                    % (tag, VERSION, STATE["mode"], mechanism, "closing not finished in time" if closing else "wall-time stop limit reached inside a step",
                       STATE["stage"], NO_RECOVERY, utc()) + "".join("problem: %s\n" % p for p in problems))
            if unverified:
                text = ("RESULT.txt and out/manifest_g.json (if present) of this run directory are INVALID: RESULT.txt exists but its write was not "
                        "verified before the closing deadline. The result of this run is:\n" + text)
            _direct_write(os.path.join(run_dir, "RESULT_INVALID.txt" if unverified else "RESULT.txt"), text.encode())
        except Exception as exc:
            problems.append("RESULT.txt: %r" % (exc,))
        try:                                                 # 3. the log (best effort: a step that hangs while holding the log lock blocks this part)
            log.error("G_WATCHDOG (%s): %s at %s (state %s, slice %s); write lock obtained: %s; DIAG_UNC_RESOURCE_LIMIT; %s", mechanism,
                      ("closing not finished %.0f s after it began" % close_bound_s) if closing else ("wall-time stop limit %.3f s reached" % limit_s),
                      STATE["stage"], STATE["state"], STATE["slice"], bool(got), NO_RECOVERY)
            for p in problems:
                log.error("G_WATCHDOG problem: %s", p)
            log.error("%s_RESULT: ABORTED (nothing is deleted; do not rerun)", tag)
            for h in logging.getLogger().handlers:
                h.flush()
                if isinstance(h, logging.FileHandler):
                    os.fsync(h.stream.fileno())
            lp = STATE["log_path"]
            if lp:
                _direct_write(lp + ".sha256", ("%s  %s\n" % (sha_file(lp), os.path.basename(lp))).encode())
        except Exception as exc:
            problems.append("log: %r" % (exc,))
    finally:
        try:                                                 # last channel for anything that could not be recorded above
            for p in locals().get("problems", []):
                sys.stderr.write("G_WATCHDOG problem: %s\n" % p)
            sys.stderr.flush()
            sys.stdout.flush()
        finally:
            os._exit(1)


# ---------------------------------------------------------------- G0: pre-checks (before any project module is imported)
def g0_prechecks(cfg):
    if sys.byteorder != "little":
        raise Early("byte order is %s, little-endian required" % sys.byteorder)
    for k in ("c1", "c2"):
        if not isinstance(cfg[k], str) or not HEX40.fullmatch(cfg[k]):
            raise Early("argument %s is not a full lower-case hex commit identity" % k)
    if not isinstance(cfg["f2_sha"], str) or not HEX64.fullmatch(cfg["f2_sha"]):
        raise Early("argument f2_sha is not a sha256")
    script = os.path.realpath(__file__)
    code_dir = os.path.dirname(script)
    run_dir = os.path.dirname(code_dir)
    want, have = set(MODULES) | {MANIFEST_NAME}, set(os.listdir(code_dir))
    if os.path.basename(script) != ME or have != want:
        raise Early("code directory must hold exactly %s, found %s" % (sorted(want), sorted(have)))
    if set(os.listdir(run_dir)) != {os.path.basename(code_dir)}:
        raise Early("run directory is not fresh: %s" % sorted(os.listdir(run_dir)))
    repo, mode = cfg["repo"], cfg["mode"]
    if not cfg["selftest"] and os.path.abspath(sys.executable) != INTERPRETER:
        raise Early("interpreter %s is not %s" % (sys.executable, INTERPRETER))
    if not run_dir.startswith(repo + "/v06lt_evidence/DIAG_UNC/" + RUN_PREFIX[mode]):
        raise Early("run directory %s is not under v06lt_evidence/DIAG_UNC/%s* (mode %s)" % (run_dir, RUN_PREFIX[mode], mode))
    # ---- identity 1: the production working tree (never switched)
    head = git(repo, "rev-parse", "HEAD").strip()
    if head != cfg["pin"]:
        raise Early("production tree: HEAD %s is not the pin" % head)
    if git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() != cfg["branch"]:
        raise Early("production tree: branch is not %s" % cfg["branch"])
    if git(repo, "status", "--porcelain").strip() or git(repo, "diff", "--cached", "--name-only").strip():
        raise Early("production tree: Git status is not clean or something is staged")
    # ---- identity 2: the diagnostic archive (commit objects only; nothing is checked out)
    c1, c2 = cfg["c1"], cfg["c2"]
    for c in (c1, c2):
        if git(repo, "rev-parse", "--verify", c + "^{commit}").strip() != c:
            raise Early("archive commit %s does not resolve to itself" % c)
    if git(repo, "rev-parse", "--verify", c2 + "^").strip() != c1:
        raise Early("the parent of the manifest commit C2 is not the code commit C1")
    if git(repo, "diff", "--name-only", c1, c2).split() != [DIAG_DIR + MANIFEST_NAME]:
        raise Early("C2 does not add exactly %s%s to C1" % (DIAG_DIR, MANIFEST_NAME))
    mpath = os.path.join(code_dir, MANIFEST_NAME)
    with open(mpath, "rb") as fh:
        mraw = fh.read()
    if not (sha_bytes(mraw) == cfg["f2_sha"] == sha_bytes(git(repo, "show", "%s:%s%s" % (c2, DIAG_DIR, MANIFEST_NAME), binary=True))):
        raise Early("F2 manifest: export, commit C2 and --f2-sha differ", "HOLD_PROVENANCE", "DIAG_UNC_INPUT_INVALID")
    MF = json.loads(mraw.decode("utf-8"))
    if not (MF.get("schema") == F2_SCHEMA and MF.get("code_commit") == c1 and MF.get("pin") == cfg["pin"]
            and bool(MF.get("selftest")) == cfg["selftest"] and set(MF.get("code") or {}) == set(MODULES)):
        raise Early("F2 manifest: schema, code_commit == C1, pin, self-test flag or code table do not match this run",
                    "HOLD_PROVENANCE", "DIAG_UNC_INPUT_INVALID")
    ident = {}
    for name in MODULES:
        exp = sha_file(os.path.join(code_dir, name))
        got = sha_bytes(git(repo, "show", "%s:%s%s" % (c1, DIAG_DIR, name), binary=True))
        if not (exp == got == MF["code"][name]):
            raise Early("%s: export %s / commit C1 %s / F2 manifest %s differ" % (name, exp[:12], got[:12], MF["code"][name][:12]),
                        "HOLD_PROVENANCE", "DIAG_UNC_INPUT_INVALID")
        ident[name] = exp
    # ---- registered stop limit (never above the owner ceilings)
    sl = MF.get("stop_limit") or {}
    if not (all(isinstance(sl.get(k), (int, float)) and not isinstance(sl.get(k), bool) for k in ("wall_s", "disk_bytes", "reserve_bytes"))
            and 0 < sl["wall_s"] <= OWNER["wall_s"] and EVIDENCE_RESERVE < sl["disk_bytes"] <= OWNER["disk_bytes"] and sl["reserve_bytes"] >= OWNER["reserve_bytes"]
            and (MF.get("tier") or {}).get("decision") == "both"):
        raise Early("F2 manifest: stop limit absent or outside the owner ceilings, or the tier decision is not 'both'", "HOLD_PROVENANCE", "DIAG_UNC_INPUT_INVALID")
    # ---- F1 evidence named by the manifest
    f1 = MF.get("f1") or {}
    f1_dir = os.path.join(repo, str(f1.get("run_dir")))
    f1_man, f1_res = os.path.join(f1_dir, "out", "manifest_f1.json"), os.path.join(f1_dir, "RESULT.txt")
    ok = os.path.isfile(f1_man) and os.path.isfile(f1_res) and sha_file(f1_man) == f1.get("manifest_sha256")
    if ok:
        with open(f1_res, "r", encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        ok = lines[0] == f1.get("result_line") and lines[0].endswith(": PASS") and ("manifest_sha256 %s" % f1["manifest_sha256"]) in lines
    if not ok:
        raise Early("F1 evidence: manifest_f1.json or RESULT.txt of %s is absent or differs from the F2 manifest" % f1.get("run_dir"),
                    "HOLD_PROVENANCE", "DIAG_UNC_INPUT_INVALID")
    free = int(cfg["free_fn"](run_dir))
    if free - sl["disk_bytes"] < sl["reserve_bytes"]:
        raise Early("free space %d bytes minus the registered disk limit %d is below the reserve %d" % (free, sl["disk_bytes"], sl["reserve_bytes"]),
                    "HOLD_RESOURCE", "DIAG_UNC_RESOURCE_LIMIT")
    if cfg["proc_check"]:
        mine, others, p = {os.getpid()}, [], os.getpid()
        while p > 1:                                         # this process and all its ancestors (shells, tmux, tee wrappers of the run card)
            with open("/proc/%d/stat" % p, "rb") as fh:
                p = int(fh.read().rsplit(b")", 1)[1].split()[1])
            mine.add(p)
        for pid in (p for p in os.listdir("/proc") if p.isdigit()):
            try:
                with open("/proc/%s/cmdline" % pid, "rb") as fh:
                    cmd = fh.read().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue                                     # the process ended while the table was read
            if int(pid) not in mine and (repo + "/seqref_mri") in cmd:
                others.append("%s: %s" % (pid, cmd[:160]))
        if others:
            raise Early("another process works in the project tree (G is never concurrent with V06LT or G-U1): %s" % others, "HOLD_RESOURCE", "none")
    return {"script": script, "script_sha": ident[ME], "code_dir": code_dir, "run_dir": run_dir, "head": head, "ident": ident, "free_start": free,
            "MF": MF, "manifest_sha": cfg["f2_sha"]}


def import_f1(code_dir):
    spec = importlib.util.spec_from_file_location("diag_unc_f1", os.path.join(code_dir, F1_NAME))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["diag_unc_f1"] = mod
    spec.loader.exec_module(mod)
    return mod


def production_closure(F, cfg):
    """The 41 pinned closure files: working tree == pin == registered (same check as F1 S0)."""
    for rel, want in sorted(cfg["closure"].items()):
        tree, pinned = sha_file(os.path.join(cfg["repo"], rel)), sha_bytes(git(cfg["repo"], "show", "%s:%s" % (cfg["pin"], rel), binary=True))
        if not (tree == pinned == want):
            raise F.Abort("%s: working tree %s / pin %s / registered %s differ" % (rel, tree[:12], pinned[:12], want[:12]))


# ---------------------------------------------------------------- guarded writes
def W(out, rel, data):
    with WLOCK:
        return out.write(rel, data)


def read_verified(root, rel, want=None):
    """A stage-1 file and its sidecar, read directly (the original directory of a recovery is never written)."""
    p = os.path.join(root, rel)
    try:
        with open(p, "rb") as fh:
            data = fh.read()
        with open(p + ".sha256", "r") as fh:
            side = (fh.read().split() or [""])[0]
    except OSError as exc:
        raise Early("stage-1 file %s or its sidecar cannot be read (%s)" % (rel, type(exc).__name__), "FAIL", "DIAG_UNC_INPUT_INVALID")
    sha = sha_bytes(data)
    if sha != side or (want is not None and sha != want):
        raise Early("stage-1 file %s does not match its sidecar or its index entry" % rel, "FAIL", "DIAG_UNC_INPUT_INVALID")
    return data, sha


# ---------------------------------------------------------------- G1: bind against the F2 manifest
ROW_FIELDS = ("psnr", "nll_per_dim", "nll_prior_per_dim", "nll_logdet_per_dim", "nll_decomp_abs_dev")


def rows_file(F, np, p, arm, seed, e, state, entries, identities, want_sha, is_role, tol, d):
    """One stored validation-row file: hash, sidecar, schema, identity, order, finiteness and the NLL identity of 216 §3.
    Returns (sha256, (n_rows, 5) float64 array of ROW_FIELDS). Any failure stops the run before a decode."""
    tag = "(%s,%d) val_e%03d.json" % (arm, seed, e)
    j, sha = F.load_json(p)
    rows = j.get("rows") or []
    ok = (sha == F.sidecar(p) and identities.get(os.path.abspath(p), sha) == sha and j.get("schema") == F.ROWS_SCHEMA and j.get("arm") == arm
          and j.get("seed") == seed and j.get("epoch") == e and j.get("model_state_sha256") == state
          and j.get("n_rows") == len(entries) == len(rows) and [(r.get("file"), r.get("slice_index")) for r in rows] == entries
          and (not is_role or want_sha == sha) and all(F.fin(r.get(f)) for r in rows for f in ROW_FIELDS))
    if not ok:
        F.need("ROWS", False, "%s == sidecar%s; schema, arm, seed, epoch; model state == ledger commit; %d rows in the registered order; "
               "PSNR, NLL, both NLL components and the stored decomposition deviation finite" % (tag, " == F2 manifest" if is_role else "", len(entries)))
    a = np.array([[r[f] for f in ROW_FIELDS] for r in rows], dtype=np.float64)
    tau = tol["nll_identity_rel"] * np.maximum(1.0, d * np.abs(a[:, 1]))
    dev = np.abs(a[:, 1] - (a[:, 2] + a[:, 3]))
    bad = np.nonzero((a[:, 4] < 0) | (a[:, 4] > tau) | (dev > tau / d + tol["nll_identity_abs"]))[0]
    if len(bad):
        q = int(bad[0])
        F.gate("NLLID", False, "%s row %d (%s|%d): NLL identity of 216 §3 violated: stored decomposition deviation / tau = %.3e, |NLL - (B + J)| / "
               "(tau/d + 1e-12) = %.3e (both must be <= 1; tau = 1e-6 max(1, d |NLL|), d = %d); %d row(s) affected"
               % (tag, q, entries[q][0], entries[q][1], a[q, 4] / tau[q], dev[q] / (tau[q] / d + tol["nll_identity_abs"]), int(d), len(bad)))
        raise F.Stop("HOLD_PROVENANCE", "DIAG_UNC_INPUT_INVALID", "NLL identity (216 §3) violated in %s, first at row %d; %d row(s)" % (tag, q, len(bad)))
    return sha, a

def g1_bind(F, cfg, ctx, MF, R, repo, acc, data_dir, B, with_view):
    A, s, np = ctx["A"], ctx["seams"], ctx["np"]
    need, info, sidecar, load_json = F.need, F.info, F.sidecar, F.load_json
    I, V = MF["inputs"], B["verified"]
    n_vol, n_boot = MF["bootstrap"]["shape"][1], MF["bootstrap"]["shape"][0]
    # ---- registered inputs
    an_path = os.path.join(R, "analysis", "v05lt_analysis.json")
    doc, an_sha = load_json(an_path)
    need("IN", an_sha == I["analysis_sha256"] == sidecar(an_path), "analysis document sha256 == F2 manifest == sidecar")
    identities = dict((doc.get("inputs") or {}).get("identities") or {})
    del doc
    li_path = os.path.join(R, "launch", "v05lt_launch_inputs.json")
    need("IN", sha_file(li_path) == I["launch_inputs_sha256"], "launch inputs sha256 == F2 manifest")
    el_path = os.path.join(R, "inputs", "v05lt_nmse_eligibility.json")
    doc, el_sha = load_json(el_path)
    need("IN", el_sha == I["eligibility_sha256"] == sidecar(el_path), "eligibility manifest sha256 == F2 manifest == sidecar")
    el_order_sha = (doc.get("provenance") or {}).get("validation_order_sha256")
    slices = [(x.get("file"), x.get("slice_index"), x.get("midpoint"), x.get("eligible"), x.get("volume_id")) for x in doc.get("slices", [])]
    del doc
    inputs = A.load_inputs(repo)
    paths = inputs["paths"]
    sp_path = str(paths.inputs / "v05lt_split_manifest.json")
    need("IN", sha_file(sp_path) == I["split_sha256"] == sidecar(sp_path), "split manifest sha256 == F2 manifest == sidecar")
    need("IN", inputs["p4"]["file_sha256"] == I["p4_sha256"] == sha_file(str(inputs["p4_path"])) and inputs["p4"]["sidecar_verified"] is True,
         "P4 scaling artefact verified by the production loader; file sha256 == F2 manifest")
    need("IN", sha_file(str(paths.p0s)) == I["p0s_sha256"] and inputs["s_ref"] == F.S_REF, "P0S artefact sha256 == F2 manifest; S_ref == registered value")
    n_files = len(inputs["split_doc"]["train"]) + len(inputs["split_doc"]["validation"])
    need("IN", len(inputs["split_doc"]["validation"]) == n_vol and len(inputs["holdout"]) > 0
         and not (inputs["holdout"] & {r["volume_id"] for r in inputs["split_doc"]["train"] + inputs["split_doc"]["validation"]}),
         "split manifest: %d training-directory volumes (%d validation); holdout-ID manifest verified (%d IDs, JSON only), disjoint"
         % (n_files, n_vol, len(inputs["holdout"])))
    need("IN", os.path.abspath(str(paths.data_root / F.TRAIN_DIR)) == data_dir and os.path.isdir(data_dir), "the production dataset resolves to %s" % F.TRAIN_DIR)
    V.update(analysis_sha256=an_sha, eligibility_sha256=el_sha)
    # ---- population
    entries = [(f, i) for f, i, _m, _e, _v in slices]
    if with_view:
        view = A.validation_view(inputs)
        need("ACC", len(acc.opens) == 2 * n_files and not acc.reads, "index construction: %d HDF5 opens == 2 x %d files, 0 array reads" % (len(acc.opens), n_files))
        need("POP", [view.entry(i) for i in range(len(view))] == entries and view.order_lines_sha256() == el_order_sha,
             "validation view: %d positions, identical in order to the eligibility manifest; order hash == eligibility provenance" % len(entries))
        B["view"] = view
    else:
        need("ACC", not acc.opens and not acc.reads, "recovery: no HDF5 file opened; the validation order is taken from the verified eligibility manifest")
    mid_pos = [q for q, x in enumerate(slices) if x[2] is True]
    mid = [entries[q] for q in mid_pos]
    vols = [slices[q][4] for q in mid_pos]
    pop = MF["population"]
    need("POP", len(mid) == n_vol == pop["n_midpoint"] and len(set(vols)) == n_vol and mid == [tuple(x) for x in pop["identities"]]
         and el_order_sha == pop["validation_order_sha256"] and all(isinstance(slices[q][3], bool) for q in mid_pos),
         "%d registered midpoint slices, one per validation volume, identical in order to the F2 manifest; validation order hash == F2 manifest" % len(mid))
    order = sorted(range(n_vol), key=lambda j: vols[j])      # bootstrap index j == j-th volume of the sorted validation volume list (223 §3.4)
    B.update(inputs=inputs, entries=entries, mid=mid, mid_pos=mid_pos, vols=vols, order=order, n_files=n_files, n_vol=n_vol, n_boot=n_boot,
             elig={entries[q]: slices[q][3] for q in mid_pos})
    # ---- G-U1 records: identity against the manifest; R1 references and the DC tolerance (whitelisted fields only)
    pe_root = os.path.join(R, "posteval")
    dirs = sorted(d for d in os.listdir(pe_root) if os.path.isdir(os.path.join(pe_root, d)))
    need("GU1", dirs == sorted(I["posteval"]), "post-evaluation records on disk == records of the F2 manifest (%d)" % len(dirs))
    recs, tol = {}, MF["tolerances"]
    for d in dirs:
        p = os.path.join(pe_root, d, "posteval.json")
        j, sha = load_json(p)
        rows = (j.get("midpoint") or {}).get("rows") or []
        ref = {}
        for r in rows:
            ident = r.get("identity") or {}
            r1 = (((r.get("r1") or {}).get("primary") or {}).get(str(F.BANK_ROWS)) or {})
            ref[(ident.get("file"), ident.get("slice_index"))] = (r1.get("psnr"), r1.get("nmse_u"), r.get("eligible"))
        ta = (j.get("dc_tolerances") or {}).get("a")
        need("GU1", sha == I["posteval"][d] and j.get("schema") == F.POSTEVAL_SCHEMA and (j.get("checkpoint") or {}).get("model_state_sha256") == d
             and list(ref) == mid and ta == tol["dc_a"]
             and all(F.fin(v[0]) and isinstance(v[2], bool) and (F.fin(v[1]) if v[2] else v[1] is None) and v[2] == B["elig"][k] for k, v in ref.items()),
             "record %s: sha256 == F2 manifest; rows == the registered midpoint slices in order; R1 primary/128 reference finite; "
             "DC tolerance (a) == F2 manifest" % d[:12])
        recs[d] = {"ref": ref}
        del j, rows
    V["posteval"] = dict(I["posteval"])
    # ---- ledgers (verified by the F1 function) and final_weights.json
    led = {}
    for arm, seed in F.RUNS:
        le = led[(arm, seed)] = F.load_ledger(R, arm, seed, identities)
        k = "%s|%d" % (arm, seed)
        fwp = os.path.join(R, "%d_%s" % (seed, arm), "final", "final_weights.json")
        need("LED", sha_file(le["path"]) == I["ledgers"][k] and sha_file(fwp) == I["final_weights"][k] == sidecar(fwp),
             "(%s,%d) ledger and final_weights.json sha256 == F2 manifest" % (arm, seed))
    V["ledgers"], V["final_weights"] = dict(I["ledgers"]), dict(I["final_weights"])
    # ---- the 30 roles and 28 states of the manifest: files hashed here; each state is re-hashed when its file is loaded (G3)
    roles = []
    for r in MF["roles"]:
        path = os.path.join(R, r["path"])
        le = led[(r["arm"], r["seed"])]
        want_epoch = le["n"] if r["role"] == "final" else (le["selected"] if r["role"] == "selected" else int(r["role"][5:]))
        ok = (r["role"] in ROLE_SHORT and r["epoch"] == want_epoch and r["epoch"] in le["by_epoch"]
              and le["by_epoch"][r["epoch"]]["body"]["model_state_sha256"] == r["model_state_sha256"]
              and os.path.isfile(path) and sidecar(path) == r["file_sha256"] and (r.get("posteval") is None or r["posteval"] in recs)
              and (r["tier"] == 1) == (r["role"] in F.ROLES_T1) and (r["tier"] == 1) == (r.get("posteval") is not None))
        need("ROLE", ok, "(%s,%d,%s) %s: epoch %d == ledger; state == the ledger commit of that epoch; sidecar == F2 manifest"
             % (r["arm"], r["seed"], r["role"], r["path"], r["epoch"]))
        need("ROLE", sha_file(path) == r["file_sha256"], "(%s,%d,%s) file sha256 == F2 manifest (%s)" % (r["arm"], r["seed"], r["role"], r["file_sha256"][:16]))
        roles.append(dict(r, path=path, key="%s|%d|%s" % (r["arm"], r["seed"], r["role"])))
    states = [st["model_state_sha256"] for st in MF["states"]]
    by_state = {st: sorted(r["key"] for r in roles if r["model_state_sha256"] == st) for st in states}
    need("ROLE", len(roles) == 30 and len({r["key"] for r in roles}) == 30 and len(states) == len(set(states)) == 28
         and {r["model_state_sha256"] for r in roles} == set(states) and all(sorted(st["roles"]) == by_state[st["model_state_sha256"]] for st in MF["states"])
         and all(len({q["model_state_sha256"] for q in roles if q["path"] == r["path"]}) == 1 for r in roles)
         and [st["index"] for st in MF["states"]] == list(range(28)),
         "30 roles, %d distinct files (a file shared by two roles holds one state), 28 distinct model states in the registered processing order; "
         "state table == role table" % len({r["path"] for r in roles}))
    V["roles"] = MF["roles"]
    B.update(recs=recs, led=led, roles=roles, states=states)
    info("ROLE", "%d roles bound to %d states (Tier 1: %d roles; Tier 2: %d roles); every file hashed" % (len(roles), len(states),
         sum(1 for r in roles if r["tier"] == 1), sum(1 for r in roles if r["tier"] == 2)))
    # ---- stored rows of every epoch of the six runs: hashed, recorded, checked (216 §3 identity), reduced to what stage 2 needs
    vol_of = [x[4] for x in slices]
    vol_ids = sorted(set(vol_of))
    vpos = np.array([vol_ids.index(v) for v in vol_of])
    vcount = np.bincount(vpos, minlength=len(vol_ids)).astype(float)
    role_epochs = {(r["arm"], r["seed"], r["epoch"]) for r in roles}
    B["rows"], B["hist"], V["history_rows"], n_rows_files, d = {}, {}, {}, 0, float(F.FLOW_DIM)
    for arm, seed in F.RUNS:
        le, hist = led[(arm, seed)], []
        for e in range(1, le["n"] + 1):
            p = os.path.join(R, "%d_%s" % (seed, arm), "ckpt", "val_e%03d.json" % e)
            key, is_role = "%s|%d|e%03d" % (arm, seed, e), (arm, seed, e) in role_epochs
            need("ROWS", os.path.isfile(p), "(%s,%d) val_e%03d.json present" % (arm, seed, e))
            sha, a = rows_file(F, np, p, arm, seed, e, le["by_epoch"][e]["body"]["model_state_sha256"], entries, identities,
                               I["stored_rows"].get(key) if is_role else None, is_role, tol, d)
            V["history_rows"][key] = sha
            n_rows_files += 1
            m = a[mid_pos][:, :4]
            allv = np.stack([np.bincount(vpos, weights=a[:, c], minlength=len(vol_ids)) / vcount for c in range(4)], axis=1).mean(axis=0)
            hist.append({"epoch": e, "midpoint": m.mean(axis=0).tolist(), "all_slices": allv.tolist()})
            if is_role:
                B["rows"][(arm, seed, e)] = m[order]         # (n_vol, 4) in sorted-volume order: PSNR R0, NLL, B, J
                B.setdefault("rows_mid", {})[(arm, seed, e)] = {mid[jj]: (float(m[jj, 0]), float(m[jj, 1])) for jj in range(n_vol)}
        B["hist"]["%s|%d" % (arm, seed)] = {"epochs": hist, "roles": {r["role"]: r["epoch"] for r in roles if (r["arm"], r["seed"]) == (arm, seed)},
                                           "columns": ["psnr_r0_db", "nll_per_dim", "B_nll_prior_per_dim", "J_nll_logdet_per_dim"]}
    need("ROWS", set(B["rows"]) == role_epochs and set(I["stored_rows"]) <= set(V["history_rows"]),
         "stored rows: %d files (every epoch of the six runs) hashed and recorded; schema, order, state and finiteness checked; NLL identity (216 §3) "
         "holds in every row; the %d role-epoch files == F2 manifest" % (n_rows_files, len(role_epochs)))
    V["history_rows_sha256"] = sha_bytes(cjson(V["history_rows"]))
    # ---- bank
    rec = A.z_bank()
    bank = rec["bank"]
    need("BANK", bank.shape == (F.BANK_ROWS, F.FLOW_DIM) and str(bank.dtype) == "float32" and rec["bank_sha256"] == I["z_diag_sha256"] == sha_bytes(bank.tobytes(order="C"))
         and rec["manifest_sha256"] == I["z_manifest_sha256"], "Z_DIAG bank regenerated: (128, 13824) float32, raw and manifest hashes == F2 manifest")
    B["bank"] = bank
    need("ACC", not acc.reads and not acc.loads, "G1 completed with 0 array reads and 0 torch.load")


# ---------------------------------------------------------------- stage-1 record files: strict JSON
UNDEFINED_MEASURES = ("rho_r0", "rho_r1")                    # 223 §5 U7 (i): the only values of a record that may be undefined


def encode_undefined(stats_doc):
    """In memory the registered core marks an undefined within-slice correlation as NaN (and reports the event). In the file it is null,
    listed explicitly; any other non-finite value is refused by the strict writer (allow_nan=False). Returns [[subset, measure], ...]."""
    undefined = []
    for name in sorted(stats_doc["subsets"]):
        for m in UNDEFINED_MEASURES:
            v = stats_doc["subsets"][name][m]
            if isinstance(v, float) and math.isnan(v):
                stats_doc["subsets"][name][m] = None
                undefined.append([name, m])
    return undefined


def decode_undefined(body):
    """Inverse of encode_undefined; the listed entries, and only they, must be null."""
    sub = body["stats"]["subsets"]
    listed = {(n, m) for n, m in body["undefined"]}
    nulls = {(n, m) for n in sub for m in UNDEFINED_MEASURES if sub[n][m] is None}
    if nulls != listed or len(listed) != len(body["undefined"]):
        raise Early("stage-1 record %s|%s: the null correlations %s are not the listed undefined ones %s"
                    % (body.get("file"), body.get("slice_index"), sorted(nulls), sorted(listed)), "FAIL", "DIAG_UNC_INPUT_INVALID")
    for n, m in listed:
        sub[n][m] = float("nan")
    return body["stats"]


def strict_loads(data):
    def refuse(token):
        raise Early("stage-1 file holds the non-standard JSON token %s" % token, "FAIL", "DIAG_UNC_INPUT_INVALID")
    return json.loads(data.decode("utf-8"), parse_constant=refuse)


# ---------------------------------------------------------------- G3: stage 1
def fixed_slices(F, np, B, MF):
    fs = MF["fixed_slices"]
    n = int(fs["n"])
    F.need("FIX", n == N_FIXED and B["n_vol"] >= n, "fixed-slice rule: %d slices drawn from %d volumes" % (n, B["n_vol"]))
    sv = sorted(B["vols"])
    draw = np.random.default_rng(int(fs["seed"])).choice(B["n_vol"], n, replace=False)     # 216 §6: one call, own generator
    picked = [sv[int(q)] for q in draw]
    return [{"draw_index": int(q), "volume_id": v, "position": B["vols"].index(v), "file": B["mid"][B["vols"].index(v)][0],
             "slice_index": B["mid"][B["vols"].index(v)][1]} for q, v in zip(draw, picked)]


def g3_stage1(F, cfg, ctx, MF, B, acc, out):
    A, C, np, torch, core, fdm = ctx["A"], ctx["C"], ctx["np"], ctx["torch"], ctx["core"], ctx["fdm"]
    tol, n_vol = MF["tolerances"], B["n_vol"]
    acc.slice_keys = {(os.path.join(acc.data_dir, os.path.basename(f)), i) for f, i in B["mid"]}
    B["slice_states"] = {}
    for ident in B["mid"]:                                   # each midpoint slice is read exactly once in the run
        STATE["slice"] = "%s|%d" % ident
        t = tick(F, "state_build")
        B["slice_states"][ident], _dt = F.build_state(ctx, B, ident)
        tock("state_build", t)
    masks_sha = sha_bytes(b"".join(B["slice_states"][x]["mask"][0].numpy().tobytes() for x in B["mid"]))
    F.need("POP", masks_sha == MF["population"]["regenerated_masks_sha256"], "%d midpoint slice states built; identity, regenerated mask bytes and eligibility "
           "equal for every slice; sha256 of the concatenated mask bytes == F2 manifest" % len(B["mid"]))
    F.need("ACC", sorted(acc.reads) == sorted(acc.slice_keys) and len(acc.opens) == 2 * B["n_files"] + n_vol,
           "slice reads: %d array reads == each midpoint slice once; %d further HDF5 opens" % (len(acc.reads), len(acc.opens) - 2 * B["n_files"]))
    B["fixed"] = fixed_slices(F, np, B, MF)
    fixed_pos = {x["position"] for x in B["fixed"]}
    F.info("FIX", "fixed slices for the maps (216 §6 rule, seed %d): positions %s" % (MF["fixed_slices"]["seed"], sorted(fixed_pos)))
    band_grid = np.asarray(core.band_grid(F.HW))
    events, index_shas = [], []
    for si, state in enumerate(B["states"]):
        STATE["state"], STATE["slice"] = si, None
        rs = [r for r in B["roles"] if r["model_state_sha256"] == state]
        model = None
        loaded = set()
        for r in rs:                                         # every distinct role file is verified by loading it once; the state is decoded once
            if r["path"] in loaded:
                F.info("CK", "(%s,%d,%s) uses the file already verified for this state (%s)" % (r["arm"], r["seed"], r["role"], os.path.basename(r["path"])))
                continue
            t = tick(F, "load")
            mdl, _dt = F.load_role(ctx, acc, r)
            tock("load", t)
            loaded.add(r["path"])
            model = mdl if model is None else model
            del mdl
        entries, devs, partial = [], {r["key"]: [] for r in rs}, True
        try:
            for vi, ident in enumerate(B["mid"]):
                STATE["slice"] = "%s|%d" % ident
                t = tick(F, "slice")
                st, elig = B["slice_states"][ident], B["elig"][ident]
                vals, _T, imgs, (dec, x0, u0) = F.slice_values(ctx, model, st, elig, keep_images=True)
                bad = []
                for r in rs:                                 # the reproduction gate of this slice, before it is scored (plan §5, Q1)
                    s_psnr, s_nll = B["rows_mid"][(r["arm"], r["seed"], r["epoch"])][ident]
                    rec = {"file": ident[0], "slice_index": ident[1], "d_r0_psnr_db": abs(vals["r0_psnr"] - s_psnr), "d_nll_per_dim": abs(vals["nll"] - s_nll),
                           "dc_direct_max": vals["dc_direct"], "dc_a_max": vals["dc_a"], "dc_a_tol": tol["dc_a"]}
                    fails = [n for n, ok in (("r0_psnr", rec["d_r0_psnr_db"] <= tol["r0_psnr_db"]), ("nll", rec["d_nll_per_dim"] <= tol["nll_abs"]),
                                             ("dc_direct", rec["dc_direct_max"] <= tol["dc_a"]), ("dc_a", rec["dc_a_max"] <= tol["dc_a"])) if not ok]
                    if r["tier"] == 1:
                        g_psnr, g_nmse, _e = B["recs"][r["posteval"]]["ref"][ident]
                        rec["d_r1_psnr_db"] = abs(vals["r1_psnr"] - g_psnr)
                        if not rec["d_r1_psnr_db"] <= tol["r1_psnr_db"]:
                            fails.append("r1_psnr")
                        if elig:
                            if g_nmse == 0.0:
                                raise F.Stop("FAIL", "DIAG_UNC_ZERO_DENOMINATOR", "registered R1 NMSE_u reference is zero at %s|%d" % ident)
                            rec["d_r1_nmse_rel"] = abs(vals["r1_nmse"] - g_nmse) / abs(g_nmse)
                            if not rec["d_r1_nmse_rel"] <= tol["r1_nmse_rel"]:
                                fails.append("r1_nmse")
                        else:
                            rec["d_r1_nmse_rel"] = "not_applicable_ineligible_slice"
                    else:
                        rec["d_r1_psnr_db"] = rec["d_r1_nmse_rel"] = "not_applicable_no_registered_r1_anchor"
                    rec["pass"] = not fails
                    devs[r["key"]].append(rec)
                    if fails:
                        bad.append((r["key"], fails, rec))
                if bad:
                    for key, fails, rec in bad:
                        F.gate("REPRO", False, "%s %s|%d: outside tolerance: %s (deviations: R0 PSNR %.3e dB, NLL %.3e, R1 PSNR %s, R1 NMSE rel %s, DC direct %.3e "
                               "and pinned (a) %.3e vs %.3e)" % (key, ident[0], ident[1], fails, rec["d_r0_psnr_db"], rec["d_nll_per_dim"], rec["d_r1_psnr_db"],
                                                               rec["d_r1_nmse_rel"], rec["dc_direct_max"], rec["dc_a_max"], rec["dc_a_tol"]))
                    raise F.Stop("FAIL", "DIAG_UNC_REPRODUCTION_MISMATCH", "first violation at state %d, %s|%d: %s; this slice is not scored; no further slice "
                                 "or state is processed; %s" % (si, ident[0], ident[1], [(k, f) for k, f, _r in bad], NO_RECOVERY))
                rows, cols = A.free_coordinates(st)
                loc, scale = A.loc_scale_block(st)
                x_true = np.asarray(st["x_true_c"].to(torch.complex128).numpy()).reshape(-1)
                ref_energy = float((fdm.fft2c(st["x_true_c"].to(torch.complex128)).abs() ** 2).sum())
                x = np.stack(imgs)
                x_r0 = np.asarray(x0.detach().numpy()).reshape(-1)
                ev = []
                srec = core.slice_stats(dec, A.observed_u(st), u0, loc, scale, band_grid[rows, cols], x, x_true, x_r0, ref_energy,
                                        case_id="%s|%d" % ident, events=ev)
                events += [{"code": e[0], "state_index": si, "case": e[1], "subset": e[2], "measure": e[3]} for e in ev]
                body = {"schema": REC_SCHEMA, "version": VERSION, "selftest": cfg["selftest"], "state_index": si, "model_state_sha256": state,
                        "roles": [r["key"] for r in rs], "volume_id": B["vols"][vi], "file": ident[0], "slice_index": ident[1], "position": vi,
                        "r1_psnr_db": vals["r1_psnr"], "stats": json.loads(core.record_to_bytes(srec).decode("utf-8"))}
                body["undefined"] = encode_undefined(body["stats"])          # 223 §5 U7 (i): explicit, never a NaN token
                if sorted((x[2], x[3]) for x in ev) != sorted(map(tuple, body["undefined"])):
                    raise F.Stop("FAIL", "DIAG_UNC_INPUT_INVALID", "state %d, %s|%d: undefined correlations of the record differ from the reported events"
                                 % (si, ident[0], ident[1]))
                rel = "records/s%02d/v%02d.rec" % (si, vi)
                try:
                    data = json.dumps(body, sort_keys=True, allow_nan=False).encode("utf-8")
                except ValueError as exc:
                    raise F.Stop("FAIL", "DIAG_UNC_NONFINITE", "state %d, %s|%d: a non-finite value other than an undefined within-slice correlation "
                                 "in the stage-1 record (%s)" % (si, ident[0], ident[1], exc))
                ent = {"file": rel, "sha256": W(out, rel, data), "bytes": len(data), "position": vi, "volume_id": B["vols"][vi], "identity": [ident[0], ident[1]],
                       "gate": "pass"}
                if vi in fixed_pos:                          # SD and error maps of the fixed slices, float32, for figures only (223 §9)
                    t_mag, m = np.abs(x_true), np.abs(x)
                    maps = np.stack([m.std(axis=0, ddof=1), np.abs(np.abs(x_r0) - t_mag), np.abs(np.abs(x.mean(axis=0)) - t_mag)]).astype("<f4")
                    ent["maps"] = {"file": "maps/s%02d_v%02d.f32" % (si, vi), "layout": "3 x %d float32 little-endian: M-space SD, |m_R0 - t|, ||mean x| - t|" % maps.shape[1]}
                    ent["maps"]["sha256"] = W(out, ent["maps"]["file"], maps.tobytes(order="C"))
                    if si == 0:
                        ent["reference"] = {"file": "maps/ref_v%02d.f32" % vi, "layout": "%d float32 little-endian: reference magnitude" % maps.shape[1]}
                        ent["reference"]["sha256"] = W(out, ent["reference"]["file"], t_mag.astype("<f4").tobytes(order="C"))
                entries.append(ent)
                del dec, x0, u0, imgs, x, srec, body, data, vals
                tock("slice", t)
            ok = C.model_state_sha256(model) == state and model.training is False
            F.need("IMM", ok, "state %d: model state hash and eval mode unchanged by decoding" % si)
            partial = False
        finally:
            del model
            name = "records/s%02d/reproduction%s.json" % (si, ".partial" if partial else "")
            try:                                             # complete, or partial after a stop: written exactly once per state
                repro_sha = W(out, name, cjson({"version": VERSION, "selftest": cfg["selftest"], "partial": partial, "state_index": si,
                                               "model_state_sha256": state, "tolerances": tol,
                                               "note": "absolute deviations |own - registered| and pass flags only; no compared value is stored", "roles": devs}))
            except Exception as exc:
                log.error("cannot write %s: %r", name, exc)
                raise
        STATE["slice"] = None
        t = tick(F, "index")
        idx_doc = {"schema": "seqref-dunc-g-state-index/1", "version": VERSION, "selftest": cfg["selftest"], "state_index": si, "model_state_sha256": state,
                   "roles": [r["key"] for r in rs], "n_records": len(entries), "records": entries, "reproduction": {"file": name, "sha256": repro_sha}}
        index_shas.append({"state_index": si, "model_state_sha256": state, "file": "records/s%02d/state_index.json" % si,
                           "sha256": W(out, "records/s%02d/state_index.json" % si, cjson(idx_doc))})
        tock("index", t)
        log.info("[G3] progress: state %d/%d done; roles %s; %d records; all gates passed; elapsed %.0f s; new disk use %d bytes",
                 si + 1, len(B["states"]), [r["key"] for r in rs], len(entries), elapsed(), out.written)
    STATE["state"] = None
    final = {"schema": "seqref-dunc-g-index/1", "version": VERSION, "selftest": cfg["selftest"], "n_states": len(index_shas),
             "n_records": len(index_shas) * n_vol, "states": index_shas, "fixed_slices": B["fixed"], "typed_events": events}
    B["index_sha"] = W(out, "records/index.json", cjson(final))
    STATE["stage1_complete"] = True
    B.pop("slice_states")                                    # not needed by stage 2: the decoded inputs leave memory here
    F.gate("S1", True, "stage 1 complete: %d states x %d slices = %d records, every reproduction gate passed; final index sha256 %s; %d non-fatal event(s)"
           % (len(index_shas), n_vol, len(index_shas) * n_vol, B["index_sha"], len(events)))


# ---------------------------------------------------------------- G4: stage 2 (verified records and stored rows only)
def load_records(F, ctx, B, rec_root):
    """Verify the final index, the 28 state indexes and every record; return {state: [records in sorted-volume order]}, r1 PSNR, band errors."""
    core, np = ctx["core"], ctx["np"]
    data, sha = read_verified(rec_root, "records/index.json", B.get("index_sha"))
    final = json.loads(data.decode("utf-8"))
    n_vol = B["n_vol"]
    F.need("IDX", final.get("schema") == "seqref-dunc-g-index/1" and [s["model_state_sha256"] for s in final["states"]] == B["states"]
           and final["n_records"] == len(B["states"]) * n_vol, "final index: %d states in the registered order, %d records" % (len(B["states"]), final["n_records"]))
    recs, n = {}, 0
    for s in final["states"]:
        t = tick(F, "verify_state")
        sdata, _ = read_verified(rec_root, s["file"], s["sha256"])
        sidx = json.loads(sdata.decode("utf-8"))
        ents = sidx["records"]
        ok = (sidx["model_state_sha256"] == s["model_state_sha256"] and sidx["state_index"] == s["state_index"] and len(ents) == n_vol
              and [e["position"] for e in ents] == list(range(n_vol)) and [tuple(e["identity"]) for e in ents] == B["mid"] and all(e["gate"] == "pass" for e in ents))
        F.need("IDX", ok, "state %d index: %d records in the registered order, every gate passed" % (s["state_index"], len(ents)))
        lst = []
        for e in ents:
            rdata, _ = read_verified(rec_root, e["file"], e["sha256"])
            body = strict_loads(rdata)
            if not (body["schema"] == REC_SCHEMA and body["model_state_sha256"] == s["model_state_sha256"] and body["position"] == e["position"]
                    and body["volume_id"] == B["vols"][e["position"]] == e["volume_id"] and F.fin(body["r1_psnr_db"])):
                F.need("IDX", False, "state %d record %d: identity == index == registered population" % (s["state_index"], e["position"]))
            lst.append((core.record_from_bytes(json.dumps(decode_undefined(body)).encode("utf-8")), body["r1_psnr_db"]))
            n += 1
        recs[s["model_state_sha256"]] = [lst[j] for j in B["order"]]
        tock("verify_state", t)
    F.gate("IDX", n == len(B["states"]) * n_vol, "stage-1 evidence verified: final index, %d state indexes and %d records equal their sidecars and index "
           "entries (final index sha256 %s)" % (len(B["states"]), n, sha))
    B["index_sha"], B["fixed"], B["events"] = sha, final["fixed_slices"], final["typed_events"]
    return recs


def paired_effect(np, stats, get, weights, seeds, idx):
    """223 §3.4 additive rule (as diag_unc_stats.additive_effect): per-volume paired differences, then means; one matrix."""
    diff = np.array([sum(w * get(role, seed) for role, w in weights.items()) for seed in seeds])
    per_seed = diff.mean(axis=1)
    lo, hi = np.percentile(diff[:, idx].mean(axis=2).mean(axis=0), [2.5, 97.5])            # as diag_unc_stats: 95% percentile, linear
    ci = (float(lo), float(hi))
    return {"point": float(per_seed.mean()), "per_seed": [float(v) for v in per_seed], "ci": ci, "label": stats.sign_label(ci), "undefined_draws": 0}


def g4_stage2(F, cfg, ctx, MF, B, idx, rec_root):
    np, stats, core = ctx["np"], ctx["stats"], ctx["core"]
    tol, seeds = MF["tolerances"], sorted({r["seed"] for r in B["roles"]})
    recs = load_records(F, ctx, B, rec_root)
    t = tick(F, "population")
    role_of = {(r["arm"] + "/" + ROLE_SHORT[r["role"]], r["seed"]): r for r in B["roles"]}
    names = stats.registered_roles(True)
    F.need("POPU", sorted(role_of) == sorted((n, s) for n in names for s in seeds) and seeds == [2026, 2027, 2028],
           "the 30 roles are the registered checkpoint roles of both tiers x three seeds")
    sv = sorted(B["vols"])
    pop = stats.Population(sv)
    series = {}
    for (name, seed), r in sorted(role_of.items()):
        rows = B["rows"][(r["arm"], r["seed"], r["epoch"])]
        lst = recs[r["model_state_sha256"]]
        for v, (rec, _r1) in zip(sv, lst):
            pop.add(name, seed, v, rec)
        series[(name, seed)] = {"psnr_r0": rows[:, 0], "nll": rows[:, 1], "B": rows[:, 2], "J": rows[:, 3], "psnr_r1": np.array([x[1] for x in lst]),
                                "band_err_r0": np.array([x[0]["band_err_r0"] for x in lst]),
                                "band_err_r1": np.array([x[0]["subsets"]["full"]["band_err_r1"] for x in lst])}
    pop.seal()
    del recs
    tock("population", t)
    contrasts = dict(stats.TIER1_CONTRASTS)
    contrasts.update(stats.TIER2_CONTRASTS)
    F.need("CON", {"tier1": sorted(stats.TIER1_CONTRASTS), "tier2": sorted(stats.TIER2_CONTRASTS)} == {k: sorted(v) for k, v in MF["contrasts"].items()},
           "contrast names of the statistics module == F2 manifest (%d + %d)" % (len(stats.TIER1_CONTRASTS), len(stats.TIER2_CONTRASTS)))
    # ---- D-GAP (223 §4) and the matched band error (223 §5 U4), every contrast
    t = tick(F, "dgap")
    tier2_names = {n for n in names if n.split("/")[1] in ("e10", "e30")}
    dgap, labels = {}, {}
    for cname, w in contrasts.items():
        eff = {q: paired_effect(np, stats, lambda role, seed, q=q: series[(role, seed)][q], w, seeds, idx) for q in ("nll", "B", "J", "psnr_r0", "psnr_r1")}
        lab = {"nll": stats.nll_label(eff["nll"]["ci"]), "psnr_r0": stats.psnr_label(eff["psnr_r0"]["ci"]), "psnr_r1": stats.psnr_label(eff["psnr_r1"]["ci"]),
               "practical_r0": stats.practical_label(eff["psnr_r0"]["ci"]), "practical_r1": stats.practical_label(eff["psnr_r1"]["ci"])}
        lab["gap_pattern"] = bool(stats.gap_pattern(lab["nll"], lab["psnr_r0"]))
        ci, pt = eff["nll"]["ci"], eff["nll"]["point"]
        if (ci[1] < 0 or ci[0] > 0) and abs(pt) >= tol["share_min_abs_dnll"]:      # 216 §3 share rule
            share = {"J_over_NLL": eff["J"]["point"] / pt, "B_over_NLL": eff["B"]["point"] / pt}
        else:
            share = "share not reported (216 §3: the dNLL interval must exclude 0 and |dNLL| >= %g nats/dim)" % tol["share_min_abs_dnll"]
        band = {k: [paired_effect(np, stats, lambda role, seed, k=k, b=b: series[(role, seed)][k][:, b], w, seeds, idx) for b in range(core.N_BANDS)]
                for k in ("band_err_r0", "band_err_r1")}
        dgap[cname] = {"effects": eff, "labels": lab, "share": share, "band_error": band,
                       "r1_provenance": ("new diagnostic measurement - no registered R1 anchor (Tier 2 role in this contrast)" if set(w) & tier2_names
                                         else "D-UNC re-decoded values verified against the registered G-U1 rows")}
        labels[cname] = {"nll": lab["nll"], "psnr_r0": lab["psnr_r0"]}
    tock("dgap", t)
    # ---- the registered analysis (step-B module, unchanged)
    t = tick(F, "run_registered")
    res = stats.run_registered(pop, idx, tier2=True, dgap=labels, validity_ok=True)
    tock("run_registered", t)
    # ---- X5, D-GAP part (223 §8.1; the statistics module evaluates the probabilistic measures only)
    x5c = [c for c in stats.TIER2_CONTRASTS if c.startswith(("C-OBJ@", "X-FIX"))]
    x5_hits = sorted("%s:%s" % (c, q) for c in x5c for q in ("nll", "psnr_r0", "psnr_r1") if dgap[c]["effects"][q]["label"] in ("below0", "above0"))
    cross = list(res["cross"] or [])
    if x5_hits and "X5" not in cross:
        cross = [x for x in cross if x != "X4"] + ["X5"]
    # ---- per-checkpoint levels (point and per-seed values only: no interval or label is registered for a single checkpoint)
    t = tick(F, "checkpoint_levels")
    levels, k_full, counts = {}, pop.records[(names[0], seeds[0])][0]["subsets"]["full"]["K"], stats.multiplicities(idx)
    for name in names:
        one, lv = {name: 1.0}, {}
        for m in stats.additive_measures(k_full):
            e = stats.additive_effect(pop, one, m, "full", idx)
            lv[m] = {"point": e["point"], "per_seed": e["per_seed"]}
        for m in stats.nonlinear_measures(k_full):
            e = stats.nonlinear_effect(pop, one, m, "full", idx, counts)
            lv[m] = {"point": e["point"], "per_seed": e["per_seed"]}
        for q in ("nll", "B", "J", "psnr_r0", "psnr_r1"):
            ps = [float(series[(name, s)][q].mean()) for s in seeds]
            lv[q] = {"point": float(np.mean(ps)), "per_seed": ps}
        for k in ("band_err_r0", "band_err_r1"):
            for b, bn in enumerate(("low", "mid", "high")):
                ps = [float(series[(name, s)][k][:, b].mean()) for s in seeds]
                lv["%s_%s" % (k, bn)] = {"point": float(np.mean(ps)), "per_seed": ps}
        levels[name] = lv
    tock("checkpoint_levels", t)
    return {"contrasts": res["contrasts"], "checkpoints": res["checkpoints"], "levels": levels, "dgap": dgap,
            "cross": {"rows": cross, "from_statistics_module": res["cross"], "x5_dgap_hits": x5_hits,
                      "note": "X5 is assigned if a probabilistic measure (statistics module) or the NLL-PSNR pair (D-GAP: dNLL, dPSNR R0, dPSNR R1) of "
                              "C-OBJ@10/@30 or X-FIX has an interval excluding 0"},
            "roles": {"%s|%d" % k: {"key": r["key"], "epoch": r["epoch"], "tier": r["tier"], "model_state_sha256": r["model_state_sha256"]} for k, r in role_of.items()},
            "volumes_sorted": sv, "seeds": seeds, "contrast_weights": contrasts, "k_full": k_full}


# ---------------------------------------------------------------- G5: tables and figure data
STATEMENT = {
    "population": "97 registered midpoint validation slices (one per validation volume) of V05LT; holdout excluded",
    "seeds": "2026, 2027, 2028; all effects paired by seed; intervals conditional on these three trained seeds",
    "interval": "95% percentile bootstrap (linear), unit = volume, B = 10,000, one index matrix (seed 20261114) reused for every quantity (223 §3.4)",
    "multiplicity": "no multiplicity adjustment; intervals are exploratory; an interval including 0 is never read as equivalence or no change",
    "validation_reuse": "the validation population was also used for checkpoint selection and for the plateau stop",
    "selection_limitation": "'selected' is the maximum of about 50 validation R0-PSNR checks on this population, so its PSNR is biased upward in both "
                            "arms; S-GAP dPSNR is biased downward and S-CTL upward; a lower final PSNR is not by itself evidence of harm (223 §1)",
    "tier2_r1": "R1 values of epochs 10 and 30 are new diagnostic measurements with no registered R1 reproduction anchor",
    "practical_reference": "0.5 dB is the V05LT practical reference (194 §9.2), context only, not a D-UNC decision rule",
    "sign_labels": "below0 / above0 / unresolved / not computed refer to the 95% interval of the paired effect",
}


def csv_bytes(header, rows, contrast_note, selftest):
    buf = io.StringIO()
    buf.write("# %s; self-test (NOT G EVIDENCE): %s\n" % (VERSION, selftest))
    for k, v in sorted(dict(STATEMENT, contrasts=contrast_note).items()):
        buf.write("# %s: %s\n" % (k, v))
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(["" if v is None else (repr(v) if isinstance(v, float) else v) for v in r])
    return buf.getvalue().encode("utf-8")


def eff_cells(e):
    ci = e.get("ci")
    return [e.get("point"), None if ci is None else ci[0], None if ci is None else ci[1]] + list(e.get("per_seed") or [None, None, None]) + [e.get("label")]


EFF_HEAD = ["point", "ci_lo", "ci_hi", "seed_2026", "seed_2027", "seed_2028", "label"]


def g5_outputs(F, cfg, ctx, MF, B, S, out):
    stats, core = ctx["stats"], ctx["core"]
    C, L, D = S["contrasts"], S["levels"], S["dgap"]
    cn = sorted(C)
    note = "all %d registered contrasts (223 §1): %s" % (len(cn), ", ".join(cn))
    cells = ["%s_%s" % c for c in stats.CELLS]
    bands = ("low", "mid", "high")

    def eff(c, m, sub="full"):
        return C[c]["effects"][(m, sub)]

    def stab(c, m):
        return C[c]["stability"].get(m)

    def measure_table(measures):
        rows = [[c, m] + eff_cells(eff(c, m)) + [eff(c, m)["undefined_draws"], stab(c, m)] for c in cn for m in measures]
        rows += [["CHECKPOINT " + r, m, L[r][m]["point"], None, None] + list(L[r][m]["per_seed"] or [None] * 3) + ["level (no interval registered)", None, None]
                 for r in sorted(L) for m in measures]
        return ["contrast", "measure"] + EFF_HEAD + ["undefined_draws", "mc_stability"], rows

    T = {}
    T["t01_dgap"] = (["contrast", "quantity"] + EFF_HEAD + ["direction_label", "practical_0.5dB", "gap_pattern", "share_J_over_NLL", "share_B_over_NLL", "r1_provenance"],
                     [[c, q] + eff_cells(D[c]["effects"][q])
                      + [D[c]["labels"].get(q), D[c]["labels"].get({"psnr_r0": "practical_r0", "psnr_r1": "practical_r1"}.get(q)), D[c]["labels"]["gap_pattern"],
                         D[c]["share"]["J_over_NLL"] if isinstance(D[c]["share"], dict) else D[c]["share"],
                         D[c]["share"]["B_over_NLL"] if isinstance(D[c]["share"], dict) else None, D[c]["r1_provenance"]]
                      for c in cn for q in ("nll", "B", "J", "psnr_r0", "psnr_r1")]
                     + [["CHECKPOINT " + r, q, L[r][q]["point"], None, None] + L[r][q]["per_seed"] + ["level (stored rows / verified re-decoded R1)"] + [None] * 6
                        for r in sorted(L) for q in ("nll", "B", "J", "psnr_r0", "psnr_r1")])
    T["t02_spread_width"] = measure_table(["sd_U", "sd_M"] + ["width_" + c for c in cells])
    T["t03a_coverage_checkpoint"] = (["checkpoint", "space", "level", "attainable_pi", "coverage", "signed_error", "ci_lo", "ci_hi", "seed_2026", "seed_2027", "seed_2028",
                                      "absolute_calibration_label", "reference_0.05", "tie_fraction", "ties_material", "mean_width"],
                                     [[r, sp, lv, k["pi"], k["coverage"], k["signed"], k["ci"][0], k["ci"][1]] + k["per_seed_signed"]
                                      + [k["label"], k["reference"], k["tie_fraction"], k["ties_material"], L[r]["width_%s_%s" % (sp, lv)]["point"]]
                                      for r in sorted(S["checkpoints"]) for (sp, lv), k in sorted(S["checkpoints"][r]["calibration"].items())])
    T["t03b_coverage_contrast"] = measure_table(["abscov_" + c for c in cells])
    T["t04_scores"] = measure_table(["ucrps", "mcrps", "energy"])
    h5, r5 = measure_table(["band_crps_" + b for b in bands])
    r5 += [[c, "%s_%s" % (k, bands[b])] + eff_cells(D[c]["band_error"][k][b]) + [0, "not applicable (no sample subset)"] for c in cn
           for k in ("band_err_r0", "band_err_r1") for b in range(3)]
    r5 += [["CHECKPOINT " + r, "%s_%s" % (k, b), L[r]["%s_%s" % (k, b)]["point"], None, None] + L[r]["%s_%s" % (k, b)]["per_seed"] + ["level (no interval registered)", None, None]
           for r in sorted(L) for k in ("band_err_r0", "band_err_r1") for b in bands]
    T["t05_band"] = (h5, r5)
    T["t06_ssr"] = measure_table(["ssr"])
    T["t07_unc_error"] = measure_table(["rho_r0", "rho_r1", "rho_vol"])
    subsets = sorted({sub for c in cn for (_m, sub) in C[c]["effects"]})
    T["t08a_mc_subsets"] = (["contrast", "measure", "subset"] + EFF_HEAD + ["mc_stability"],
                            [[c, m, sub] + eff_cells(e) + [stab(c, m)] for c in cn for (m, sub), e in sorted(C[c]["effects"].items())])
    T["t08b_mc_batch_means"] = (["contrast", "measure"] + ["group_%d" % g for g in range(8)] + ["mean", "se_mc", "bootstrap_half_width_K128"],
                                [[c, m] + list(bm["effects"]) + [bm["mean"], bm["se"],
                                                                 None if eff(c, m)["ci"] is None else (eff(c, m)["ci"][1] - eff(c, m)["ci"][0]) / 2.0]
                                 for c in cn for m, bm in sorted(C[c]["batch_means"].items())])
    T["t09_matrix"] = (["contrast", "nll_label", "psnr_r0_label", "psnr_r1_label", "practical_r0", "gap_pattern", "rows_assigned", "unresolved", "deciding_effects",
                        "report", "row_truth"],
                       [[c, D[c]["labels"]["nll"], D[c]["labels"]["psnr_r0"], D[c]["labels"]["psnr_r1"], D[c]["labels"]["practical_r0"], D[c]["labels"]["gap_pattern"],
                         json.dumps(C[c]["interpretation"]["rows"]), json.dumps(jsonable(C[c]["interpretation"]["unresolved"])),
                         json.dumps(C[c]["interpretation"]["deciding"]), C[c]["interpretation"]["report"], json.dumps(jsonable(C[c]["interpretation"]["truth"]))] for c in cn])
    cross_c = ["C-TIME0", "C-TIME2", "X-TIME", "S-CTL0", "S-CTL2", "X-CTL", "C-OBJ", "C-OBJ@3"] + sorted(c for c in cn if c.startswith(("C-OBJ@1", "C-OBJ@30", "F-TRAJ", "X-FIX")))
    cross_m = ["ucrps", "mcrps", "energy", "ssr"] + ["abscov_" + c for c in cells] + ["width_" + c for c in cells]
    T["t10_cross"] = (["contrast", "quantity"] + EFF_HEAD + ["cross_rows_assigned", "x5_dgap_hits"],
                      [[c, q] + eff_cells(D[c]["effects"][q]) + [json.dumps(S["cross"]["rows"]), json.dumps(S["cross"]["x5_dgap_hits"])]
                       for c in cross_c for q in ("nll", "B", "J", "psnr_r0", "psnr_r1")]
                      + [[c, m] + eff_cells(eff(c, m)) + [json.dumps(S["cross"]["rows"]), json.dumps(S["cross"]["x5_dgap_hits"])] for c in cross_c for m in cross_m])
    meta = {"version": VERSION, "selftest": cfg["selftest"], "plan": PLAN, "manifest_f2_sha256": B["manifest_sha"], "index_sha256": B["index_sha"],
            "statement": dict(STATEMENT, contrasts=note), "bootstrap_sha256": MF["bootstrap"]["sha256"]}
    results = jsonable(dict(S, meta=meta, histories=B["hist"], fixed_slices=B["fixed"], typed_events=B["events"]))
    FD = {
        "fd_histories": {"runs": B["hist"], "note": "per-epoch means of the stored validation rows; midpoint = the 97 midpoint slices; all_slices = mean over "
                                                    "volumes of per-volume slice means; no decoding"},
        "fd_dgap": {c: {"effects": D[c]["effects"], "labels": D[c]["labels"], "share": D[c]["share"], "r1_provenance": D[c]["r1_provenance"]} for c in cn},
        "fd_rankhist": {r: S["checkpoints"][r]["histogram"] for r in sorted(S["checkpoints"])},
        "fd_coverage": {r: {"calibration": S["checkpoints"][r]["calibration"], "width": {c: L[r]["width_" + c] for c in cells}} for r in sorted(S["checkpoints"])},
        "fd_crps": {c: {m: {"effect": eff(c, m), "mc_stability": stab(c, m)} for m in ("ucrps", "mcrps", "energy")} for c in cn},
        "fd_band": {c: {"band_crps": {b: eff(c, "band_crps_" + b) for b in bands}, "band_error": D[c]["band_error"]} for c in cn},
        "fd_maps": {"fixed_slices": B["fixed"], "files": B["map_files"], "note": "float32 maps written in stage 1; scales are fixed from model-independent "
                                                                                "data in the G-FIG plan before any map is viewed"},
        "fd_convergence": {c: {m: {sub: eff(c, m, sub) for sub in subsets if (m, sub) in C[c]["effects"]} for m in ("ucrps", "mcrps", "energy")} for c in cn},
        "fd_deciles": {r: {"bins": S["checkpoints"][r]["deciles"], "columns": ["mean_sd", "mean_r0_error", "mean_r1_error"]} for r in sorted(S["checkpoints"])},
    }
    inv = {"tables": sorted(T), "figdata": sorted(FD), "required": MF["output_inventory"]}
    F.need("INV", sorted(MF["output_inventory"]["tables"]) == sorted(T) and sorted(MF["output_inventory"]["figdata"]) == sorted(FD),
           "output inventory == F2 manifest (plan §9): %d tables, %d figure-data files" % (len(T), len(FD)))
    written = {}
    t = tick(F, "outputs")
    written["results.json"] = W(out, "results.json", cjson(results))
    for name, (header, rows) in sorted(T.items()):
        written["tables/%s.csv" % name] = W(out, "tables/%s.csv" % name, csv_bytes(header, rows, note, cfg["selftest"]))
    for name, body in sorted(FD.items()):
        written["figdata/%s.json" % name] = W(out, "figdata/%s.json" % name, cjson(jsonable({"meta": meta, "data": body})))
    tock("outputs", t)
    F.gate("OUT", len(written) == 1 + len(T) + len(FD), "outputs written once, each with a sha256 sidecar: results.json, %d tables, %d figure-data files "
           "(no value is logged)" % (len(T), len(FD)))
    return written, inv


# ---------------------------------------------------------------- recovery eligibility (plan §8)
def recovery_gate(F, cfg, guard_mod, MF):
    od = cfg["recovery_of"]
    res_p, stop_p, wd_p = os.path.join(od, "RESULT.txt"), os.path.join(od, "stop_record.json"), os.path.join(od, "stop_record_watchdog.json")
    ok = (os.path.isdir(od) and os.path.basename(od).startswith(RUN_PREFIX["g"]) and od.startswith(cfg["repo"] + "/v06lt_evidence/DIAG_UNC/")
          and os.path.isfile(res_p) and os.path.isfile(stop_p) and not os.path.exists(wd_p) and os.path.isfile(os.path.join(od, "out", "records", "index.json")))
    sr = {}
    if ok:
        with open(res_p, "r", encoding="utf-8") as fh:
            first = fh.read().split("\n")[0]
        sr = json.loads(read_verified(od, "stop_record.json")[0].decode("utf-8"))
        ok = (first.split(": ")[1].split(" ")[0] in ("FAIL", "ABORTED") and sr.get("stage_reached") in ("G4", "G5") and sr.get("typed_code") == "none"
              and sr.get("stage1_complete") is True and sr.get("manifest_f2_sha256") == cfg["f2_sha"] and bool(sr.get("selftest")) == cfg["selftest"]
              and sr.get("mode") == "g" and isinstance(sr.get("elapsed_s"), float))
    if not ok:
        F.gate("REC", False, "the original run %s is not eligible for a stage-2 recovery (required: a G run with the same F2 manifest that stopped in G4 or G5 "
               "with no typed code, a complete stage 1 and no watchdog stop)" % od)
        raise F.Stop("HOLD_PROVENANCE", "DIAG_UNC_INPUT_INVALID", "recovery refused: the original run is not eligible (plan §8)")
    base, mine = os.path.dirname(od), os.path.basename(STATE["run_dir"])
    earlier = []
    for d in sorted(os.listdir(base)):
        mark = os.path.join(base, d, "RECOVERY_OF.txt")
        if d != mine and d.startswith(RUN_PREFIX["recovery"]) and os.path.isfile(mark):
            with open(mark, "r", encoding="utf-8") as fh:
                if fh.read().strip() == os.path.basename(od):
                    earlier.append(d)
    if earlier:
        F.gate("REC", False, "a recovery of %s was already started (%s); a second one needs a new protocol review (plan §8)" % (os.path.basename(od), earlier))
        raise F.Stop("HOLD_PROVENANCE", "DIAG_UNC_INPUT_INVALID", "recovery refused: an earlier recovery of this run exists (plan §8)")
    _direct_write(os.path.join(STATE["run_dir"], "RECOVERY_OF.txt"), (os.path.basename(od) + "\n").encode())
    with open(os.path.join(od, "out", guard_mod.LEDGER_NAME), "r") as fh:
        used = int(json.load(fh)["written"])
    sl = MF["stop_limit"]
    left_s, left_b = sl["wall_s"] - sr["elapsed_s"], int(sl["disk_bytes"]) - used
    F.gate("REC", True, "original run %s: stopped at %s with no typed code, stage 1 complete; it used %.1f s and %d bytes of the registered stop limit; "
           "this recovery may use %.1f s and %d bytes" % (os.path.basename(od), sr["stage_reached"], sr["elapsed_s"], used, left_s, left_b))
    if left_s <= 0 or left_b <= EVIDENCE_RESERVE:
        raise F.Stop("HOLD_RESOURCE", "DIAG_UNC_RESOURCE_LIMIT", "nothing is left of the registered stop limit for a recovery")
    return left_s, left_b, {"run_dir": os.path.relpath(od, cfg["repo"]), "result_line": first, "stop_record_sha256": sha_file(stop_p),
                            "elapsed_s": sr["elapsed_s"], "disk_bytes": used, "approval": cfg["approval"]}


# ---------------------------------------------------------------- orchestration
def checks(cfg, pre, F):
    """Returns ((result, code, reason), write guard, pending manifest or None)."""
    MF, mode, tagc = pre["MF"], cfg["mode"], TAGS[cfg["mode"]]
    log.info("%s | plan %s | mode %s", VERSION, PLAN, mode)
    if cfg["selftest"]:
        log.warning("SELF-TEST MODE -- mock repository, synthetic data, stand-in model -- NOT G EVIDENCE")
    if mode == "bind":
        log.warning("BIND-ONLY MODE -- G0-G2; no checkpoint load, no slice read, no decode -- NON-SCIENTIFIC, NOT A G RUN, REGISTERS NOTHING")
    if mode == "recovery":
        log.warning("RECOVERY MODE -- stage 2 and outputs from the verified stage-1 records of %s; approval reference: %s", cfg["recovery_of"], cfg["approval"])
    log.info("start_utc %s", utc())
    log.info("script %s sha256 %s", pre["script"], pre["script_sha"])
    log.info("production tree: HEAD %s (pin, branch %s, clean, nothing staged)", pre["head"], cfg["branch"])
    log.info("diagnostic archive: code commit C1 %s, manifest commit C2 %s (parent C1; adds the manifest only); exports == C1 == F2 manifest: %s",
             cfg["c1"], cfg["c2"], {k: v[:16] for k, v in sorted(pre["ident"].items())})
    log.info("F2 manifest sha256 %s; F1 evidence %s (manifest %s) verified", pre["manifest_sha"], MF["f1"]["run_dir"], MF["f1"]["manifest_sha256"])
    log.info("registered stop limit %s; free space at start %d bytes", MF["stop_limit"], pre["free_start"])
    import h5py
    import numpy as np
    import torch
    log.info("python %s | numpy %s | torch %s | h5py %s | torch threads %d", sys.version.split()[0], np.__version__, torch.__version__,
             h5py.__version__, torch.get_num_threads())
    repo = cfg["repo"]
    R = os.path.join(repo, "seqref_mri", "results", "v05lt")
    sys.path.insert(0, pre["code_dir"])
    sys.path.insert(0, repo)
    data_dir = os.path.join(repo, "seqref_mri", "data", "fastmri", F.TRAIN_DIR)
    out_dir = os.path.join(pre["run_dir"], "out")
    os.mkdir(out_dir)
    B, out, result, extra = {"verified": {}, "manifest_sha": pre["manifest_sha"], "map_files": []}, None, None, {}
    acc = F.Access(h5py, torch, np, data_dir)
    acc.on()
    try:
        if cfg["closure"] is None:                          # production: the registered closure table of the (hash-verified) F1 runner
            cfg["closure"] = dict(F.CLOSURE)
        production_closure(F, cfg)
        import diag_unc_adapter as A
        import diag_unc_core as core
        import diag_unc_stats as stats
        import diag_unc_guard as guard_mod
        sl = MF["stop_limit"]
        limit_s, limit_b = float(sl["wall_s"]), int(sl["disk_bytes"])
        STATE["limit_s"] = limit_s
        if mode == "recovery":
            limit_s, limit_b, extra["recovery_of"] = recovery_gate(F, cfg, guard_mod, MF)
            STATE["limit_s"] = limit_s
        threading.Thread(target=watchdog, args=(limit_s, cfg["lock_wait_s"], cfg["close_bound_s"]), daemon=True, name="dunc-watchdog").start()
        out = guard_mod.WriteGuard(out_dir, ceiling=limit_b, reserve=int(sl["reserve_bytes"]), free_fn=lambda: cfg["free_fn"](out_dir))
        W(out, "reserve/evidence.reserve", b"\0" * EVIDENCE_RESERVE)
        log.info("limits of this run: wall %.3f s, new disk %d bytes, free-space reserve %d bytes; %d bytes counted in advance for the log, RESULT.txt "
                 "and stop evidence; watchdog started", limit_s, limit_b, sl["reserve_bytes"], EVIDENCE_RESERVE)
        s = A.seams()
        from seqref_mri.scripts import v05lt_metrics as M
        ctx = {"A": A, "C": s.C, "EV": s.EV, "M": M, "torch": torch, "np": np, "core": core, "stats": stats, "fdm": s.fdm, "seams": s}
        cl = dict(cfg, archive_commit=cfg["c1"])
        pre_f1 = dict(pre, ident={k: v for k, v in pre["ident"].items()})
        F.closure_gate(cl, pre_f1, "after import")
        F.need("CONST", (F.PIN, F.REPO, F.INTERPRETER, F.DIAG_DIR) == (PIN, REPO, INTERPRETER, DIAG_DIR) and F.N_VOLUMES == 97
               and {k: MF["tolerances"][k] for k in ("r0_psnr_db", "nll_abs", "r1_psnr_db", "r1_nmse_rel")}
               == {"r0_psnr_db": F.TOL_PSNR_DB, "nll_abs": F.TOL_NLL, "r1_psnr_db": F.TOL_PSNR_DB, "r1_nmse_rel": F.TOL_NMSE_REL}
               and MF["fixed_slices"]["seed"] == stats.DIAG_UNC_SEED + 1 and MF["bootstrap"]["seed"] == stats.DIAG_UNC_SEED,
               "constants of this script, the F1 runner, the statistics module and the F2 manifest agree (pin, paths, reproduction tolerances, seeds)")
        if F.FAILS:
            raise F.Stop("FAIL", "DIAG_UNC_INPUT_INVALID", "import closure or constants: failed gates %s" % sorted(set(F.FAILS)))
        STATE["stage"] = "G1"
        t = tick(F, "bind")
        g1_bind(F, cfg, ctx, MF, R, repo, acc, data_dir, B, with_view=(mode != "recovery"))
        ctx["bank"] = B["bank"]
        tock("bind", t)
        STATE["stage"] = "G2"
        t = tick(F, "bootstrap")
        idx = stats.bootstrap_matrix(n_boot=B["n_boot"], n_volumes=B["n_vol"])
        boot_sha = stats.verify_matrix(idx, MF["bootstrap"]["sha256"], n_volumes=B["n_vol"], n_boot=B["n_boot"])
        tock("bootstrap", t)
        F.gate("BOOT", True, "bootstrap index matrix regenerated in one call: shape %s, sha256 %s == F2 manifest" % (list(idx.shape), boot_sha))
        if mode == "bind":
            result = ("BIND_OK", "none", "G0-G2 completed; nothing was loaded, read or decoded")
        else:
            rec_root = out_dir
            if mode == "g":
                STATE["stage"] = "G3"
                g3_stage1(F, cfg, ctx, MF, B, acc, out)
            else:
                rec_root, STATE["stage1_complete"] = os.path.join(cfg["recovery_of"], "out"), True
            STATE["stage"], STATE["state"], STATE["slice"] = "G4", None, None
            if cfg.get("fault") == "stage2":
                raise RuntimeError("injected fault: stage 2 (self-test only)")
            S = g4_stage2(F, cfg, ctx, MF, B, idx, rec_root)
            STATE["stage"] = "G5"
            # the map files of the fixed slices, from the verified state indexes
            t = tick(F, "verify_maps")
            n_maps = n_refs = 0
            for st in json.loads(read_verified(rec_root, "records/index.json", B["index_sha"])[0].decode("utf-8"))["states"]:
                for e in json.loads(read_verified(rec_root, st["file"], st["sha256"])[0].decode("utf-8"))["records"]:
                    if "maps" in e:                          # bytes == sidecar == state-index entry, and the registered size (review of GR v0.1, finding 3)
                        npix = F.HW * F.HW
                        mdata, _ = read_verified(rec_root, e["maps"]["file"], e["maps"]["sha256"])
                        n_maps += 1
                        sizes_ok = len(mdata) == 3 * npix * 4
                        if "reference" in e:
                            rdata, _ = read_verified(rec_root, e["reference"]["file"], e["reference"]["sha256"])
                            n_refs += 1
                            sizes_ok = sizes_ok and len(rdata) == npix * 4
                        if not sizes_ok:
                            raise F.Stop("FAIL", "DIAG_UNC_INPUT_INVALID", "map file of state %d, position %d has the wrong size" % (st["state_index"], e["position"]))
                        B["map_files"].append({"state_index": st["state_index"], "position": e["position"], "maps": e["maps"], "reference": e.get("reference"),
                                               "root": "this run" if mode == "g" else extra["recovery_of"]["run_dir"]})
            tock("verify_maps", t)
            F.need("MAPS", n_maps == len(B["states"]) * len(B["fixed"]) and n_refs == len(B["fixed"]) and {m["position"] for m in B["map_files"]}
                   == {x["position"] for x in B["fixed"]}, "retained maps verified before the outputs: %d map files (%d states x %d fixed slices) and %d "
                   "reference files equal their sidecars and their state-index entries" % (n_maps, len(B["states"]), len(B["fixed"]), n_refs))
            if cfg.get("fault") == "outputs":
                raise RuntimeError("injected fault: outputs (self-test only)")
            written, inv = g5_outputs(F, cfg, ctx, MF, B, S, out)
            extra.update(outputs=written, inventory=inv)
            result = ("PASS", "none", "all stages completed")
        # ---- post-checks (every mode)
        t = tick(F, "post")
        F.closure_gate(cl, pre_f1, "at end")
        F.gate("POST", git(repo, "rev-parse", "HEAD").strip() == cfg["pin"] and not git(repo, "status", "--porcelain").strip(),
               "production tree: HEAD is the pin, status clean")
        bad = [k for k, want in sorted(B["verified"]["history_rows"].items())
               if sha_file(os.path.join(R, "%s_%s" % (k.split("|")[1], k.split("|")[0]), "ckpt", "val_%s.json" % k.split("|")[2])) != want]
        bad += [d for d, want in B["verified"]["posteval"].items() if sha_file(os.path.join(R, "posteval", d, "posteval.json")) != want]
        F.gate("POST", not bad, "registered JSON inputs unchanged since G1 (%d row files and %d G-U1 records re-hashed; changed: %s)"
               % (len(B["verified"]["history_rows"]), len(B["verified"]["posteval"]), bad or "none"))
        n_loads = len({r["path"] for r in B["roles"]}) if mode == "g" else 0
        n_opens = {"g": 2 * B["n_files"] + B["n_vol"], "bind": 2 * B["n_files"], "recovery": 0}[mode]
        F.gate("ACC", not acc.forbidden and len(acc.loads) == n_loads == len(set(acc.loads)) and len(acc.reads) == (B["n_vol"] if mode == "g" else 0)
               and len(acc.opens) == n_opens, "totals: %d HDF5 opens, %d array reads, %d torch.load (each distinct role file once), 0 numpy.load, %d forbidden"
               % (len(acc.opens), len(acc.reads), len(acc.loads), len(acc.forbidden)))
        tock("post", t)
        if F.FAILS:
            raise F.Stop("FAIL", "DIAG_UNC_INPUT_INVALID", "failed gates: %s" % sorted(set(F.FAILS)))
        if elapsed() + cfg["close_bound_s"] > STATE["limit_s"]:
            raise F.Stop("ABORTED", "DIAG_UNC_RESOURCE_LIMIT", "all stages completed, but %.1f s of the registered limit %.3f s are left and closing needs up "
                         "to %.0f s: the run is not closed as PASS (%s)" % (STATE["limit_s"] - elapsed(), STATE["limit_s"], cfg["close_bound_s"], NO_RECOVERY))
        begin_closing()                                      # from here the closing deadline applies (CLOSE_BOUND_S), enforced by the watchdog thread
        B["manifest"] = {
            "schema": "seqref-dunc-g-manifest/1", "version": VERSION, "plan": PLAN, "mode": mode, "utc": utc(), "selftest": cfg["selftest"],
            "validity": "PENDING -- certifies nothing. Written before the log was closed. See manifest_g.json and RESULT.txt.",
            "pin": cfg["pin"], "code_commit": cfg["c1"], "manifest_commit": cfg["c2"], "manifest_f2_sha256": pre["manifest_sha"], "code": pre["ident"],
            "inputs_verified": B["verified"], "bootstrap_sha256": boot_sha, "stage1_index_sha256": B.get("index_sha"), "fixed_slices": B.get("fixed"),
            "typed_events": B.get("events", []), "limits": {"wall_s": limit_s, "disk_bytes": limit_b, "reserve_bytes": sl["reserve_bytes"]},
            "used": {"elapsed_s": elapsed(), "disk_bytes_before_manifest": out.written, "longest_steps_s": STATE["durations"]},
            "access": {"hdf5_opens": len(acc.opens), "array_reads": len(acc.reads), "torch_loads": len(acc.loads)},
            "exposure": "log and terminal: identities, counts, deviations, pass flags, times and bytes only; scientific values only under out/",
            "completion": "223 §9 deliverables still pending after this run: G-FIG (figures) and the result report with the synthesis diagram", **extra}
        sha = W(out, "manifest_g.pending.json", cjson(jsonable(B["manifest"])))
        F.info("END", "pending manifest written (out/manifest_g.pending.json sha256 %s); elapsed %.1f s of %.3f s; new disk use %d bytes of %d"
               % (sha, elapsed(), limit_s, out.written, limit_b))
    except F.Stop as exc:
        log.error("G_STOP at %s: %s", STATE["stage"], exc)
        result = (exc.result, exc.code, exc.reason)
        if STATE["stage"] in ("G0", "G1") and exc.result == "FAIL" and exc.code == "DIAG_UNC_INPUT_INVALID":
            result = ("HOLD_PROVENANCE", exc.code, exc.reason)               # invalid registered input found while binding (plan Q5)
    except F.Abort as exc:
        log.error("G_ABORT at %s: %s", STATE["stage"], exc)
        result = ("ABORTED", "none", str(exc))
    except Early as exc:
        log.error("G_STOP at %s: %s", STATE["stage"], exc)
        result = (exc.result, exc.code, str(exc))
    except Exception as exc:
        code = getattr(exc, "code", None)
        log.error("error at %s: %r", STATE["stage"], exc)
        log.error("%s", traceback.format_exc())
        if code == "DIAG_UNC_RESOURCE_LIMIT":
            result = ("ABORTED", code, "%s: %s; %s" % (type(exc).__name__, exc, NO_RECOVERY))      # write guard: disk ceiling or free-space reserve
        elif isinstance(code, str) and code.startswith("DIAG_UNC_"):
            result = ("FAIL", code, "%s: %s" % (type(exc).__name__, exc))                           # typed stop of the step-B modules
        elif STATE["stage"] == "G3" and re.search(r"non[-_ ]?finite|\bnan\b|\binf\b", str(exc), re.I):
            result = ("FAIL", "DIAG_UNC_NONFINITE", "%s: %s" % (type(exc).__name__, exc))
        else:
            result = ("ABORTED", "none", "untyped error at %s: %s: %s" % (STATE["stage"], type(exc).__name__, exc))
    finally:
        acc.off()
    if result[0] not in ("PASS", "BIND_OK"):
        begin_closing()
        rec = {"version": VERSION, "utc": utc(), "selftest": cfg["selftest"], "mode": mode, "result": result[0], "typed_code": result[1], "reason": result[2],
               "mechanism": "main thread", "stage_reached": STATE["stage"], "state": STATE["state"], "slice": STATE["slice"],
               "stage1_complete": STATE["stage1_complete"], "elapsed_s": elapsed(), "manifest_f2_sha256": pre["manifest_sha"],
               "failed_gates": sorted(set(F.FAILS)), "verified_so_far": {k: (v if not isinstance(v, dict) else len(v)) for k, v in B.get("verified", {}).items()},
               "files_present": sorted(f for f in os.listdir(out_dir) if not f.startswith(".")),
               "recovery": ("may be requested under plan §8 (new recorded approval, reviewed fix)" if STATE["stage"] in ("G4", "G5") and result[1] == "none"
                            and STATE["stage1_complete"] and mode == "g" else NO_RECOVERY)}
        try:                                                 # outside the guard root: a disk-limit stop must still leave its record (bytes counted in advance)
            rec["disk_bytes"] = out.written if out is not None else None
            data, p = cjson(jsonable(rec)), os.path.join(pre["run_dir"], "stop_record.json")
            with WLOCK:
                _direct_write(p, data)
                _direct_write(p + ".sha256", ("%s  stop_record.json\n" % sha_bytes(data)).encode())
            log.error("stop record written: stop_record.json sha256 %s", sha_bytes(data))
        except Exception as exc:
            log.error("cannot write stop_record.json: %r", exc)
    try:
        import resource
        F.info("RES", "peak RSS %.0f MiB at the end" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0))
    except Exception as exc:
        log.error("[RES] resource report failed: %r", exc)
    log.info("end_utc %s", utc())
    (log.info if result[0] in ("PASS", "BIND_OK") else log.error)("%s%s_CHECKS: %s (typed code %s): %s", tagc, "_SELFTEST" if cfg["selftest"] else "", *result)
    return result, out, (B.get("manifest") if result[0] in ("PASS", "BIND_OK") else None)


def fault(cfg, name):
    if cfg.get("fault") == name:
        raise OSError("injected fault: %s (self-test only)" % name)


def run(cfg):
    STATE.update(mode=cfg["mode"], selftest=cfg["selftest"])
    pre = g0_prechecks(cfg)
    tag = TAGS[cfg["mode"]] + ("_SELFTEST_RESULT" if cfg["selftest"] else "_RESULT")
    run_dir = pre["run_dir"]
    STATE["run_dir"] = run_dir
    print("G_OUT=%s" % run_dir)
    log_path, problems = os.path.join(run_dir, "g.log"), []
    side_path, result_path = log_path + ".sha256", os.path.join(run_dir, "RESULT.txt")
    fh, result, out, pending = None, ("ABORTED", "none", "log file not created"), None, None
    try:
        fault(cfg, "log_open")
        fh = logging.FileHandler(log_path, mode="x", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logging.getLogger().addHandler(fh)                   # root: adapter, step-B, F1-helper and production messages are kept too
        STATE["log_path"] = log_path
    except Exception as exc:
        log.error("cannot create the log file %s: %r", log_path, exc)
        problems.append("log file not created")
    if fh is not None:
        try:
            F = import_f1(pre["code_dir"])
            result, out, pending = checks(cfg, pre, F)
        except Exception as exc:                             # checks() handles its own errors; this is a last line of defence
            log.error("unexpected error outside the stages: %r", exc)
            log.error("%s", traceback.format_exc())
            result = ("ABORTED", "none", "unexpected error outside the stages")
        finally:
            begin_closing()
            with WLOCK:                                      # a watchdog that already holds the lock finishes its own evidence first
                logging.getLogger().removeHandler(fh)
                fh.close()
    good = ("PASS", "BIND_OK")
    log_sha = None
    if fh is not None:
        try:
            log_sha = sha_file(log_path)
            fault(cfg, "sidecar")
            with open(side_path, "x", encoding="utf-8") as sf:
                sf.write("%s  g.log\n" % log_sha)
                sf.flush()
                os.fsync(sf.fileno())
            with open(side_path, "r", encoding="utf-8") as sf:
                parts = sf.read().split()
            if parts != [log_sha, "g.log"] or sha_file(log_path) != log_sha:
                raise OSError("sidecar does not verify against the log")
        except Exception as exc:
            log.error("log sidecar failure: %r", exc)
            problems.append("log sidecar not written or not verified")
    if problems and result[0] in good + ("HOLD_RESOURCE", "HOLD_PROVENANCE"):
        result = ("FAIL", "none", "evidence handling failed")
    manifest_sha = None
    if result[0] in good:                                    # only now: the log is closed and its sidecar verified
        try:
            fault(cfg, "manifest")
            pending.update(validity="valid only together with a RESULT.txt in the run directory whose first line is '%s: %s' and that names this file's "
                                    "sha256, and only if the run directory holds neither stop_record_watchdog.json nor RESULT_INVALID.txt; otherwise it "
                                    "certifies nothing" % (tag, result[0]), log_sha256=log_sha, result=result[0])
            manifest_sha = W(out, "manifest_g.json", cjson(jsonable(pending)))
        except Exception as exc:
            log.error("cannot write manifest_g.json: %r", exc)
            problems.append("manifest_g.json not written")
            result = ("FAIL", "none", "manifest_g.json not written")
    final = result[0] if result[0] in good else "%s (nothing is deleted; do not rerun)" % result[0]
    text = ("%s: %s\n%s\nmode %s\ntyped_code %s\nreason %s\nutc %s\nlog_sha256 %s\n" % (tag, final, VERSION, cfg["mode"], result[1], result[2], utc(), log_sha)
            + ("manifest_sha256 %s\n" % manifest_sha if result[0] in good else "") + "".join("problem: %s\n" % p for p in problems))
    STATE["rc"] = {"PASS": 0, "BIND_OK": 0, "HOLD_RESOURCE": 3, "HOLD_PROVENANCE": 3}.get(result[0], 1)
    try:
        fault(cfg, "result")
        with open(result_path, "x", encoding="utf-8") as rf:
            rf.write(text)
            rf.flush()
            os.fsync(rf.fileno())
        with open(result_path, "r", encoding="utf-8") as rf:
            if rf.read() != text:
                raise OSError("RESULT.txt read-back differs")
        STATE["result_verified"] = True                      # only from here does the run's own result stand at the closing deadline
    except Exception as exc:
        log.error("cannot write RESULT.txt: %r", exc)
        problems.append("RESULT.txt not written or not verified")
        final = "FAIL (nothing is deleted; do not rerun)"
        result = ("FAIL", "none", "RESULT.txt not written")
        if manifest_sha is not None:                         # a manifest exists without its RESULT.txt: mark it, delete nothing
            try:
                W(out, "MANIFEST_INVALID.txt", ("manifest_g.json (sha256 %s) is INVALID: RESULT.txt could not be written or verified; this run is FAIL. "
                                                "%s %s\n" % (manifest_sha, VERSION, utc())).encode("utf-8"))
                log.error("out/MANIFEST_INVALID.txt written")
            except Exception as exc2:
                log.error("cannot write MANIFEST_INVALID.txt: %r", exc2)
    STATE["rc"] = {"PASS": 0, "BIND_OK": 0, "HOLD_RESOURCE": 3, "HOLD_PROVENANCE": 3}.get(result[0], 1)
    for p in problems:
        log.error("problem: %s", p)
    (log.info if result[0] in good else log.error)("%s: %s", tag, final)
    STATE["done"] = True                                     # nothing is left but the exit: the closing fail-safe no longer acts
    return STATE["rc"]


def main():
    sys.dont_write_bytecode = True                           # no __pycache__ is written into the pinned tree or the run directory
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-commit")
    ap.add_argument("--manifest-commit")
    ap.add_argument("--f2-sha")
    ap.add_argument("--bind-only", action="store_true")
    ap.add_argument("--recovery-of", metavar="RUN_DIR")
    ap.add_argument("--recovery-approval", metavar="REFERENCE")
    ap.add_argument("--selftest", metavar="CONFIG.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")
    rc, tag = 1, "G"
    try:
        if a.bind_only and a.recovery_of:
            raise Early("--bind-only and --recovery-of exclude each other")
        if bool(a.recovery_of) != bool(a.recovery_approval):
            raise Early("a recovery needs both --recovery-of and --recovery-approval (the recorded approval reference)")
        mode = "bind" if a.bind_only else ("recovery" if a.recovery_of else "g")
        tag = TAGS[mode]
        cfg = dict(selftest=False, mode=mode, repo=REPO, pin=PIN, branch="seqref-v05lt", c1=a.code_commit, c2=a.manifest_commit, f2_sha=a.f2_sha, closure=None,
                   free_fn=lambda p: shutil.disk_usage(p).free, fault=None, proc_check=True, lock_wait_s=LOCK_WAIT_S, close_bound_s=CLOSE_BOUND_S,
                   recovery_of=os.path.abspath(a.recovery_of) if a.recovery_of else None, approval=a.recovery_approval)
        if a.selftest:
            if a.code_commit or a.manifest_commit or a.f2_sha:
                raise Early("self-test takes its identities from the config file only")
            with open(a.selftest, "r", encoding="utf-8") as fh:
                o = json.load(fh)
            must, opt = {"repo", "pin", "c1", "c2", "f2_sha", "closure"}, {"fault", "free_bytes", "proc_check", "lock_wait_s", "close_bound_s"}
            if set(o) - (must | opt) or not must <= set(o):
                raise Early("self-test config: keys must be exactly %s (+ optional %s)" % (sorted(must), sorted(opt)))
            if os.path.realpath(o["repo"]) == os.path.realpath(REPO) or o["pin"] == PIN:
                raise Early("self-test must not use the production repository or pin")
            if o.get("fault") not in (None, "log_open", "sidecar", "manifest", "result", "stage2", "outputs"):
                raise Early("self-test config: unknown fault")
            cfg.update({k: o[k] for k in must})
            cfg.update(selftest=True, repo=os.path.abspath(o["repo"]), fault=o.get("fault"), proc_check=bool(o.get("proc_check", False)),
                       lock_wait_s=float(o.get("lock_wait_s", LOCK_WAIT_S)), close_bound_s=float(o.get("close_bound_s", CLOSE_BOUND_S)))
            if "free_bytes" in o:
                cfg["free_fn"] = lambda p, v=int(o["free_bytes"]): v
        elif not (a.code_commit and a.manifest_commit and a.f2_sha):
            raise Early("production mode needs --code-commit, --manifest-commit and --f2-sha (run card)")
        rc = run(cfg)
    except Early as exc:
        log.error("G_STOP before the stages: %s", exc)
        log.error("%s_RESULT: %s (nothing is deleted; do not rerun)", tag, exc.result)
        rc = 3 if exc.result.startswith("HOLD") else 1
    except Exception as exc:
        log.error("unexpected error outside the run: %r", exc)
        log.error("%s", traceback.format_exc())
        log.error("%s_RESULT: FAIL (nothing is deleted; do not rerun)", tag)
        rc = 1
    print("G_EXIT=%d" % rc)
    sys.stdout.flush()
    sys.exit(rc)


if __name__ == "__main__":
    main()
