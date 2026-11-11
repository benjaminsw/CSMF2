# SEQREF-MRI-DIAGJSHG v0.6 -- DIAG-JSH stage G: single real-data main run (frozen mini-spec v1.0 §1-§8)
# LIFETIME: KEEP
# =============================================================================
# Plan SEQREF-MRI-DIAGJSHF2 v0.1 (F2b), reviewer edits 1-5 (217 J17).
# One run, bound to the F1 manifest (d3e8176e...) by hash and to the F2
# resource record (diag_jsh_f2_resources.py) by code hash:
#   1. binding through the pinned V05LT code (F1 bind_production, reused --
#      not the F1 preparation run); every binding field must equal the
#      manifest (pin, launch chain, launch inputs, completion evidence,
#      backend, bound inputs, ZF record, core/adapter/F1 code hashes);
#   2. selected checkpoints re-derived by the pinned load_run and compared
#      field by field with the manifest; population and case order equal;
#   3. each of the six checkpoints decoded ONCE (adapter replay); the
#      stored rows' NLL identity (§3) and the §1.3 reproduction check
#      (exact identities, frozen limits) per checkpoint; regenerated masks
#      equal the manifest; spectra (§4) per slice from the same decode;
#   4. stratifiers recomputed and required to equal the manifest's thirds,
#      boundaries and hashes; retained slices equal the manifest (the
#      recorded membership and slices are used, never re-chosen);
#   5. the registered G-D4 V05LT analysis (v05lt_analysis.json, bound by
#      sha256 and sidecar in the F2 resource record) is the stage document
#      written by the pinned v05lt_analysis.run_analysis (schema
#      seqref-v05lt-analysis-stage/1); its "analysis" member is the pinned
#      v05lt_stats.analyse result (seqref-v05lt-analysis/1), whose
#      comparisons supply the registered crossed-bootstrap result
#      between|selected|r0|psnr, reported alongside
#      the JSH estimate with its own estimand and resampling label (§1.4,
#      §2); never pooled;
#   6. DIAG-I3 v0.3 (unchanged): volume matrices, analysis (§2-§5), table,
#      figures; then the imported pinned modules are verified against the pin
#      (F1 check_loaded) and recorded; write-once outputs with sha256
#      sidecars; a run record g_run.json (+ sidecar) with identities, limits,
#      timings, loaded modules and output hashes. The deadline is checked
#      after the outputs, before the run record and again before PASS.
# Limits: Python deadline = start + min(T, MAX_WALL_S - 120 s), T from the
#   F2 resource record (never extended); write budget and log allowance
#   from the same record; the reserve is checked before every write.
# Writes: matplotlib is initialised in a RESTRICTED INITIALISATION PHASE,
#   enforced by this module's own audit hook before the adapter guard:
#   * write-opens and file operations only on paths inside OUT/mplconfig (or
#     /dev/null); a write-open on an integer descriptor is allowed only if
#     /proc/self/fd/N resolves inside OUT/mplconfig;
#   * subprocesses only `fc-list --help` / `fc-list --format=...`, with no
#     explicit env, and only if the executable that would run (explicit
#     `executable`, else argv[0] resolved on PATH) is the fc-list resolved at
#     phase start; any other process creation halts (exit 77);
#   * no os.link / os.symlink; at most INIT_MAX_OPENS write-opens in the
#     whole phase -- every write-open counts once, whatever its path, so
#     renaming, replacing, deleting or reusing a path cannot reset or evade
#     the allowance (halt 77 beyond); /dev/null excepted;
#   * fontconfig's user cache (written natively by fc-list) is redirected to
#     a read-only folder OUT/mplconfig/fontconfig-ro (XDG_CACHE_HOME), so
#     fc-list cannot write its cache (as a non-root user; root bypasses the
#     mode, and then the bytes land inside OUT and are counted);
#   * BOUNDED EXCEPTION to the per-write budget check (no write size is known
#     at open): before the phase, free >= reserve + budget is required; during
#     it each file is kernel-capped at INIT_FILE_CAP (RLIMIT_FSIZE, installed
#     by init_plotting; EFBIG, not a kill), so the SIMULTANEOUS FILE
#     CONTENTS covered by audited opens in the phase are
#     <= INIT_MAX_OPENS x INIT_FILE_CAP (16 MiB), whatever number of
#     descriptors is open at once and whatever happens to their names. This
#     does NOT bound cumulative bytes written (rewriting a file writes more
#     bytes without growing it) nor native writes (C-level writes, the
#     fc-list child, descriptors opened before the phase -- none to a regular
#     file in production: the run log is attached later and the standard
#     streams are the command's pipes); the bytes under
#     OUT/mplconfig are also checked against INIT_TOTAL_CAP (8 MiB) at every
#     audited event (halt 77); after the phase the cache must be
#     <= INIT_TOTAL_CAP and its bytes are charged to the budget (32 MiB).
#   The phase ends before the adapter guard is installed.
# Complement to the adapter guard (after the phase, for the whole run): the
#   adapter's guard accepts any integer descriptor for a write-open and checks
#   only argv for subprocesses; this module's hook additionally requires that
#   a descriptor opened for writing resolves (/proc/self/fd) inside OUT and,
#   once the budget is attached, to a registered file; and that the program
#   Popen would run (executable, else argv[0] on PATH) is the git / uname
#   resolved at start. The adapter is unchanged (its hash is bound in the
#   F1 manifest). From the guard
#   on, every write is registered and budgeted (strict guard). At the end the bytes under OUT must equal the
#   budgeted bytes (DIAG_UNBUDGETED_BYTES otherwise; catches native writes
#   inside OUT, not outside).
# Exits: 0 PASS; 1 typed STOP; 2 usage / unexpected (logged with traceback);
#   77 write guard; 78 log allowance. Partial outputs are kept (write-once).
# No fallback, placeholder, mock or dummy value in this file.
# Changelog (v0.6, 2026-11-11; G attempt 1 STOP DIAG_GD4, 217 J19): the
#   G-D4 loader reads the registered stage document (outer schema
#   seqref-v05lt-analysis-stage/1, then "analysis" with schema
#   seqref-v05lt-analysis/1 and the same config_sha256) -- v0.5 expected
#   the inner schema at the top level; the selected comparison's family,
#   metric, population and per-seed entries are checked; the configuration
#   hash is reported; header wording of the 16 MiB bound corrected (J18).
# Changelog (v0.5, 2026-11-11; review of v0.4): the allowance counts every
#   write-open in the phase (was: distinct path names, which renaming a
#   still-open file evaded).
# Changelog (v0.4, 2026-11-11; review of v0.3; also found in testing: an
#   import inside the audit hook re-entered the hook when the adapter was not
#   yet imported -- the adapter is now bound outside the hook): hard aggregate bound in the
#   initialisation phase (at most 4 distinct files, each kernel-capped);
#   os.link/os.symlink refused; fontconfig cache made read-only; complement
#   to the adapter guard for descriptor writes and the executable actually
#   run (same two classes as the v0.2 findings).
# Changelog (v0.3, 2026-11-11; review of v0.2): initialisation phase -- an
#   integer descriptor is checked through /proc/self/fd (was: any descriptor
#   allowed); the executable that would run is checked against the fc-list
#   resolved at phase start, explicit env refused (was: argv[0] name only);
#   per-file and total caches caps during the phase (bounded exception).
# Changelog (v0.2, 2026-11-11; review of v0.1): imported pinned modules
#   verified and recorded before PASS; registered G-D4 result bound by hash
#   and reported alongside (own estimand and resampling label); matplotlib
#   initialisation in an enforced restricted phase (was: unguarded); deadline
#   checked after the outputs, before the run record and before PASS.
# Changelog (v0.1, 2026-11-11): first G runner; manifest/resource binding,
#   single decode per checkpoint, recorded membership, budgeted plotting
#   cache, final byte reconciliation, run record.
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

__abbr__ = "SEQREF-MRI-DIAGJSHG"
__version__ = "0.6"
DIAG_ROOT = Path("/home/benjamin/CSMFII/v06lt_evidence/DIAG_JSH")
V05LT = Path("/home/benjamin/CSMFII")
CODE_FILES = ("diag_jsh_core.py", "diag_jsh_adapter.py",
              "diag_jsh_f1_prepare.py", "diag_jsh_i3.py", "diag_jsh_g_run.py",
              "diag_jsh_f2_resources.py")
WALL_MARGIN_S = 120             # Python deadline before MAX_WALL_S (as F1)
MIN_WALL_S = 300
MPL_DIR = "mplconfig"
GD4_ID = "between|selected|r0|psnr"         # registered V05LT comparison
GD4_STAGE_SCHEMA = "seqref-v05lt-analysis-stage/1"  # v05lt_analysis.RESULT_SCHEMA
GD4_ANALYSIS_SCHEMA = "seqref-v05lt-analysis/1"     # v05lt_stats.RESULT_SCHEMA
GD4_SEEDS = 3                                       # 2026, 2027, 2028
INIT_EXIT = 77                               # same code as the write guard
_INIT = {"dir": None, "fc": None, "opens": []}    # phase active if dir set
_POST = {"root": None, "exe": {}}            # complement active if root set
INIT_FILE_CAP = 4 * 2 ** 20                  # per file, kernel (RLIMIT_FSIZE)
INIT_MAX_OPENS = 4                           # write-opens in the phase
INIT_TOTAL_CAP = 8 * 2 ** 20                 # OUT/mplconfig total
FC_RO = "fontconfig-ro"                      # read-only XDG_CACHE_HOME
logger = logging.getLogger("seqref_diag.jsh_g")


def _args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--chain", required=True)
    p.add_argument("--chain-sha256", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reserve-bytes", type=int, required=True)
    p.add_argument("--max-wall-s", type=float, required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--manifest-sha256", required=True)
    for f in CODE_FILES:
        p.add_argument(f"--sha-{f.split('.')[0]}", required=True)
    return p.parse_args(argv)


def _init_halt(what: str, detail) -> None:
    sys.stderr.write(f"DIAG_INIT_PHASE {what} {detail!r}\n")
    sys.stderr.flush()
    os._exit(INIT_EXIT)


def _target(p):
    """Absolute real path of a write target (descriptor via /proc), or
    None if it cannot be resolved."""
    if isinstance(p, int) and not isinstance(p, bool):
        try:                                 # descriptor: its destination
            p = os.readlink(f"/proc/self/fd/{p}")
        except OSError:
            return None
    if isinstance(p, bytes):
        p = os.fsdecode(p)
    if not isinstance(p, (str, os.PathLike)):
        return None
    p = os.fspath(p)
    if not os.path.isabs(p):
        return None
    return os.path.realpath(p)


def _under(q, d) -> bool:
    return q is not None and (q == d or q.startswith(d + os.sep))


def _init_inside(p) -> bool:
    q = _target(p)
    return q == "/dev/null" or _under(q, _INIT["dir"])


def _fc_list_ok(executable, argv, env) -> bool:
    """argv is fc-list --help / --format=..., no explicit env, and the
    executable that would run is the fc-list resolved at phase start (if
    none was found then, nothing named fc-list can run: allowed only
    without an explicit executable)."""
    a = [os.fsdecode(x) if isinstance(x, bytes) else str(x)
         for x in (argv if isinstance(argv, (list, tuple)) else [])]
    if not (len(a) == 2 and os.path.basename(a[0]) == "fc-list" and
            (a[1] == "--help" or a[1].startswith("--format=")) and
            env is None):
        return False
    exe = a[0] if executable is None else (
        os.fsdecode(executable) if isinstance(executable, bytes)
        else str(executable))                # Popen passes argv[0] if unset
    found = (exe if os.sep in exe else shutil.which(exe))
    if found is None:
        return _INIT["fc"] is None and exe == "fc-list"
    return (_INIT["fc"] is not None and
            os.path.realpath(found) == _INIT["fc"])


def _init_bytes() -> int:
    n = 0
    for dp, _dn, fn in os.walk(_INIT["dir"]):
        for f in fn:
            try:
                n += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return n


def _post_audit(event: str, args) -> None:
    """Complement to the adapter guard (descriptor writes, real executable)."""
    A = _ADAPTER[0]
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")
                   ) or (isinstance(flags, int) and bool(flags & A._WFLAGS))
        if writing and isinstance(path, int) and not isinstance(path, bool):
            q = _target(path)
            files = A._GUARD["files"]
            if q in A._SINKS:
                return
            if not _under(q, _POST["root"]) or (files is not None and
                                                q not in files):
                _init_halt("DIAG_G_GUARD descriptor write", (path, q))
    elif event == "subprocess.Popen":
        ex, argv = (tuple(args) + (None, None))[:2]
        a = [os.fsdecode(x) if isinstance(x, bytes) else str(x)
             for x in (argv if isinstance(argv, (list, tuple)) else [])]
        name = os.path.basename(a[0]) if a else None
        want = _POST["exe"].get(name)
        exe = a[0] if ex is None else (os.fsdecode(ex) if isinstance(ex, bytes)
                                       else str(ex))
        found = exe if os.sep in exe else shutil.which(exe)
        if want is None or found is None or os.path.realpath(found) != want:
            _init_halt("DIAG_G_GUARD executable", (ex, argv))


def _init_audit(event: str, args) -> None:
    """Restricted initialisation phase (while _INIT["dir"]); afterwards the
    complement to the adapter guard (while _POST["root"])."""
    if _INIT["dir"] is None:
        if _POST["root"] is not None:
            _post_audit(event, args)
        return
    A = _ADAPTER[0]
    if event in ("os.link", "os.symlink"):
        _init_halt(event, args)
    if event in ("open",) or event in A._PATH_EVENTS:
        if _init_bytes() > INIT_TOTAL_CAP:
            _init_halt("cache total > INIT_TOTAL_CAP", INIT_TOTAL_CAP)
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")
                   ) or (isinstance(flags, int) and bool(flags & A._WFLAGS))
        if writing and not _init_inside(path):
            _init_halt("write outside OUT/mplconfig", path)
        if writing:
            q = _target(path)
            if q != "/dev/null":
                _INIT["opens"].append(q)     # every write-open counts once
                if len(_INIT["opens"]) > INIT_MAX_OPENS:
                    _init_halt("more than INIT_MAX_OPENS write-opens",
                               _INIT["opens"])
    elif event in A._PATH_EVENTS:
        for i in A._PATH_EVENTS[event]:
            if i < len(args) and not _init_inside(args[i]):
                _init_halt(event, args[i])
    elif event == "subprocess.Popen":
        ex, argv, _cwd, env = (tuple(args) + (None,) * 4)[:4]
        if not _fc_list_ok(ex, argv, env):
            _init_halt(event, (ex, argv, env is not None))
    elif event in A._PROC_EVENTS:
        _init_halt(event, args)


_HOOKED = []
_ADAPTER = []      # bound OUTSIDE the hook: an import inside an audit hook
#                    raises an "import" event and would re-enter the hook


def _hook() -> None:
    if not _ADAPTER:
        import diag_jsh_adapter
        _ADAPTER.append(diag_jsh_adapter)
    if not _HOOKED:
        sys.addaudithook(_init_audit)
        _HOOKED.append(True)


def begin_post(root: Path) -> None:
    """Start the complement for the run root (after the phase)."""
    ex = {}
    for n in ("git", "uname"):
        w = shutil.which(n)
        ex[n] = os.path.realpath(w) if w else None
    _POST["exe"] = ex
    _POST["root"] = os.path.realpath(str(root))
    _hook()


def begin_init(d: Path) -> None:
    """Start the restricted phase for directory d (hook added once)."""
    _hook()
    _INIT["opens"] = []
    _INIT["dir"] = os.path.realpath(str(d))


def init_plotting(out: Path) -> list:
    """matplotlib (Agg) and its font cache inside OUT/mplconfig, under the
    restricted phase; returns the files created there."""
    import resource
    import signal
    d = out / MPL_DIR
    os.environ["MPLCONFIGDIR"] = str(d)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["TMPDIR"] = str(out)
    old_xdg = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(d / FC_RO)
    fc = shutil.which("fc-list")
    _INIT["fc"] = os.path.realpath(fc) if fc else None
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    old_sig = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    cap = (INIT_FILE_CAP if hard == resource.RLIM_INFINITY
           else min(INIT_FILE_CAP, hard))
    resource.setrlimit(resource.RLIMIT_FSIZE, (cap, hard))
    begin_init(d)
    try:
        os.mkdir(d)
        os.mkdir(d / FC_RO, 0o555)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401
        from matplotlib import font_manager
        font_manager.fontManager.findfont("DejaVu Sans")
    finally:
        _INIT["dir"] = None                  # phase ends (hook inert)
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
        signal.signal(signal.SIGXFSZ, old_sig)
        if old_xdg is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = old_xdg
    return [(str(p.relative_to(out)), p.stat().st_size)
            for p in sorted(out.rglob("*")) if p.is_file()]


def start(out: Path, A, K, reserve: int, R: dict):
    """Restricted matplotlib phase -> guard -> budget -> log."""
    budget_bytes = R["g_budget_bytes"]
    free = shutil.disk_usage(out).free
    K.require(free >= reserve + budget_bytes, "DIAG_DISK",
              f"start: free {free} < reserve {reserve} + budget "
              f"{budget_bytes}")
    pre = init_plotting(out)
    K.require(all(f.startswith(MPL_DIR + "/") for f, _ in pre),
              "DIAG_INIT_PHASE", f"files outside {MPL_DIR}: {pre}")
    K.require(sum(n for _, n in pre) <= INIT_TOTAL_CAP and
              sum(n for _, n in pre) <= budget_bytes, "DIAG_INIT_PHASE",
              f"cache {sum(n for _, n in pre)} B > caps")
    begin_post(out)
    A.install_write_guard(out, git_repo=V05LT)
    budget = A.WriteBudget(out, reserve, budget_bytes)
    budget.take(sum(n for _, n in pre), "matplotlib cache (pre-guard, OUT)")
    h = A.BudgetLogHandler(budget, out / "g.log", R["log_allowance_bytes"])
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[h], force=True)
    logger.info("%s v%s start; pre-guard matplotlib files %s", __abbr__,
                __version__, pre)
    return budget, pre


def load_manifest(path: str, want: str, A, K) -> dict:
    """Manifest bytes = registered sha256 = its sidecar; strict JSON."""
    p = Path(path)
    K.require(p.is_file() and not p.is_symlink(), "DIAG_MANIFEST",
              f"{p} missing or a symlink")
    body = p.read_bytes()
    sha = A.sha256_bytes(body)
    K.require(sha == want, "DIAG_MANIFEST", f"manifest sha256 {sha} != {want}")
    side = Path(str(p) + ".sha256")
    K.require(side.is_file() and side.read_text().split()[:1] == [sha],
              "DIAG_MANIFEST", "sidecar missing or different")

    def no_const(x):
        K.fail("DIAG_MANIFEST", f"non-finite constant {x}")
    man = json.loads(body, parse_constant=no_const)
    K.require(man.get("schema") == "SEQREF-MRI-DIAG-JSH-MAN/0.2" and
              str(man.get("spec", "")).endswith(
                  "8665d790f4ab96d1cb526c442826da06b81a672c387a170a3c08adc2f"
                  "4018336"), "DIAG_MANIFEST", "schema or spec")
    return man


def _schema_of(x):
    """Schema of a JSON object, else its type name (for STOP messages)."""
    return x.get("schema") if isinstance(x, dict) else type(x).__name__


def load_gd4(R: dict, A, K) -> dict:
    """Registered V05LT analysis (G-D4): bytes = registered sha256 = its
    sidecar; the pinned stage document (outer GD4_STAGE_SCHEMA) carrying the
    pinned analyse() result under "analysis" (GD4_ANALYSIS_SCHEMA, same
    configuration hash); the registered comparison GD4_ID only, labelled."""
    g = R["gd4"]
    p = Path(g["path"])
    K.require(p.is_file() and not p.is_symlink(), "DIAG_GD4",
              f"{p} missing or a symlink")
    body = p.read_bytes()
    sha = A.sha256_bytes(body)
    K.require(sha == g["sha256"], "DIAG_GD4", f"sha256 {sha} != {g['sha256']}")
    side = Path(str(p) + ".sha256")
    K.require(side.is_file() and side.read_text().split()[:1] == [sha],
              "DIAG_GD4", "sidecar missing or different")
    doc = json.loads(body, parse_constant=lambda c: K.fail(
        "DIAG_GD4", f"non-finite constant {c}"))
    K.require(isinstance(doc, dict) and doc.get("schema") == GD4_STAGE_SCHEMA,
              "DIAG_GD4", f"schema {_schema_of(doc)!r} (expected "
              f"{GD4_STAGE_SCHEMA})")
    an = doc.get("analysis")
    K.require(isinstance(an, dict) and an.get("schema") == GD4_ANALYSIS_SCHEMA,
              "DIAG_GD4", f"analysis schema {_schema_of(an)!r} (expected "
              f"{GD4_ANALYSIS_SCHEMA})")
    cfg = doc.get("config_sha256")
    K.require(isinstance(cfg, str) and len(cfg) == 64 and
              all(ch in "0123456789abcdef" for ch in cfg) and
              an.get("config_sha256") == cfg, "DIAG_GD4",
              "configuration hash missing or differs between the stage "
              "document and its analysis")
    comps = an.get("comparisons")
    K.require(isinstance(comps, list), "DIAG_GD4", "analysis.comparisons "
              "is not a list")
    hits = [c for c in comps if isinstance(c, dict) and c.get("id") == GD4_ID]
    K.require(len(hits) == 1, "DIAG_GD4", f"{len(hits)} entries {GD4_ID}")
    c = hits[0]
    K.require(c.get("family") == "between" and c.get("metric") == "psnr" and
              c.get("population") == "val", "DIAG_GD4",
              f"{GD4_ID}: family/metric/population")
    ps = c.get("per_seed")
    K.require(isinstance(ps, list) and len(ps) == GD4_SEEDS and
              all(isinstance(x, dict) for x in ps), "DIAG_GD4",
              f"{GD4_ID}: per_seed")
    ci = c.get("ci")
    K.require(c.get("status") == "ok" and isinstance(c.get("estimate"), float)
              and math.isfinite(c["estimate"]) and isinstance(ci, list) and
              len(ci) == 2 and all(isinstance(x, float) and math.isfinite(x)
                                   for x in ci), "DIAG_GD4",
              f"{GD4_ID}: status/estimate/ci")
    return {"source": "registered V05LT validation analysis (G-D4)",
            "sha256": sha, "comparison_id": GD4_ID,
            "document_schema": [GD4_STAGE_SCHEMA, GD4_ANALYSIS_SCHEMA],
            "config_sha256": cfg,
            "estimate": c["estimate"], "ci95": ci,
            "per_seed": c.get("per_seed"), "support": c.get("support"),
            "practical": c.get("practical"),
            "estimand": g["estimand"], "resampling": g["resampling"],
            "note": "reported alongside the JSH estimate; never pooled; "
                    "neither substitutes for the other"}


def check_resources(R: dict, man_sha: str, a, K) -> dict:
    """F2 resource record bound to this manifest; T never extended."""
    K.require(R["manifest_sha256"] == man_sha, "DIAG_RESOURCES",
              "resource record is bound to another manifest")
    T, C = R["T_s"], R["max_wall_s"]
    K.require(isinstance(T, float) and math.isfinite(T) and T > 0 and
              a.max_wall_s == C and C - WALL_MARGIN_S >= T,
              "DIAG_RESOURCES", f"T {T}, registered cap {C}, "
              f"given {a.max_wall_s}")
    return {"T_s": T, "max_wall_s": C}


def same(K, what: str, got, want) -> None:
    K.require(got == want, "DIAG_MANIFEST_MISMATCH",
              f"{what}: run {got!r} != manifest {want!r}")


def run(ctx: dict, man: dict, man_sha: str, R: dict, out: Path, lim: dict,
        code: dict, A, K, I3, budget, pre: list) -> str:
    from seqref_mri.src import free_flow_runtime as ffr
    t_all = time.monotonic()
    hard = lim["deadline"]
    b = ctx["binding"]
    for k in ("pin", "launch_chain", "launch_inputs_sha256",
              "completion_evidence_sha256", "backend", "inputs"):
        same(K, k, b[k], man[k])
    same(K, "zf_record_sha256", ctx["zf_record_sha256"],
         man["zf_record_sha256"])
    for f, h in man["code_sha256"].items():
        same(K, f"code {f}", code.get(f), h)
    gd4 = load_gd4(R, A, K)
    logger.info("STAGE G1 binding equals the manifest; registered G-D4 "
                "result bound (%s)", gd4["sha256"])

    data = ctx["data"]
    vv, elig, p4, s_ref = (data["vv"], data["eligibility"], data["p4"],
                           data["s_ref"])
    sels = {k: ctx["load_selected"](k) for k in A.SEED_ARMS}
    for k, s in sels.items():
        rid = f"{k[0]}_{k[1]}"
        for x in ("epoch", "model_state_sha256", "weights", "weights_sha256",
                  "rows_file", "rows_sha256", "selection_evidence"):
            same(K, f"{rid}.{x}", s[x], man["checkpoints"][rid][x])
    docs = {k: s["rows"] for k, s in sels.items()}
    entries = A.check_population(docs, ctx["zf_rows"])
    n = len(entries)
    K.require(n == len(vv), "DIAG_POPULATION_MISMATCH",
              f"{n} rows vs {len(vv)} view slices")
    pop = man["population"]
    same(K, "slices", n, pop["slices"])
    same(K, "case_order_sha256", A.canonical_sha(entries),
         pop["case_order_sha256"])
    same(K, "slice_ids_sha256", docs[A.SEED_ARMS[0]]["slice_ids_sha256"],
         pop["slice_ids_sha256"])
    retained_keys = {(r["volume_id"], r["file"], r["slice_index"])
                     for r in man["retained_slices"]}
    A.check_deadline(hard, "inputs")
    logger.info("STAGE G2 checkpoints and population equal the manifest "
                "(%d slices)", n)

    per_slice, repro, hfs, imgs = {}, {}, [], {}
    t_loads, t_reps, disk = [], [], []
    max_decomp = max_spec = 0.0
    for k in A.SEED_ARMS:
        rid = f"{k[0]}_{k[1]}"
        stored = {(r["volume_id"], r["file"], r["slice_index"]): r
                  for r in docs[k]["rows"]}
        vals = {}

        def on_slice(key, row, spec, xh, xt, k=k, stored=stored, vals=vals):
            nonlocal max_spec
            vals[key] = I3.slice_values(stored[key], spec)
            for c, f in (("sc_cells", "sc_full"), ("sm_cells", "sm_full")):
                full = spec[f]
                r = abs(float(spec[c].sum()) - full) / full if full else 0.0
                max_spec = max(max_spec, r)
            if k[0] == 2026 and key in retained_keys:
                d = imgs.setdefault(key, {"volume_id": key[0],
                                          "slice_index": key[2]})
                d[k[1]] = xh
                if "x_true" in d:
                    K.require(bool((d["x_true"] == xt).all()),
                              "DIAG_REFERENCE_MISMATCH", f"{key} reference")
                d["x_true"] = xt
        t0 = time.monotonic()
        model, _vc = A.load_model(sels[k])
        t_loads.append(time.monotonic() - t0)
        t0 = time.monotonic()
        rep = A.replay(model, vv, elig, p4, s_ref, on_slice=on_slice,
                       deadline=hard)
        t_reps.append(time.monotonic() - t0)
        del model
        repro[rid] = A.reproduce(docs[k], rep, n)     # also §3 identity
        same(K, f"{rid} regenerated masks",
             A.canonical_sha(sorted(rep["mask_regen"].items())),
             pop["masks_regenerated_sha256"])
        K.require(list(vals) == list(entries), "DIAG_POPULATION_MISMATCH",
                  f"{rid}: decoded case order")
        per_slice[k] = vals
        hfs.append(rep["hf"])
        max_decomp = max(max_decomp, max(r["nll_decomp_abs_dev"]
                                         for r in docs[k]["rows"]))
        disk.append(budget.check(0, f"after {rid}"))
        logger.info("STAGE G3 %s decoded once; reproduction %s", rid,
                    repro[rid])
    _ = ffr.FLOW_DIM_REAL                             # pinned d_free in use
    hf = A.check_reference_equal(hfs)

    strat = A.stratifiers(entries, ctx["zf_rows"], hf)
    for s in ("H-D", "H-F"):
        ms = man["subgroups"][s]
        same(K, f"{s} thirds", strat[s]["thirds"], ms["thirds"])
        same(K, f"{s} boundaries", [[strat[s]["values"][g[0]],
                                     strat[s]["values"][g[-1]]]
                                    for g in strat[s]["thirds"]],
             ms["boundaries"])
        same(K, f"{s} sha256", strat[s]["sha256"], ms["sha256"])
    retained = A.retained_slices(entries, elig)
    same(K, "retained slices", retained, man["retained_slices"])
    ret_imgs = []
    for r in man["retained_slices"]:
        key = (r["volume_id"], r["file"], r["slice_index"])
        K.require(key in imgs and set(imgs[key]) >= {"lam0", "lam2", "x_true"},
                  "DIAG_RETAINED", f"{key}: images incomplete")
        ret_imgs.append(imgs[key])
    A.check_deadline(hard, "analysis")
    logger.info("STAGE G4 recorded subgroups and retained slices reproduced")

    t0 = time.monotonic()
    vols, M = I3.volume_matrices(
        {(s, a): per_slice[(s, a)] for s, a in A.SEED_ARMS}, entries)
    res = I3.analyse(vols, M, strat)
    res["reconstruction"]["registered_G-D4"] = gd4
    t_boot = time.monotonic() - t0
    A.check_deadline(hard, "outputs")
    ident = {k: {"epoch": sels[k]["epoch"],
                 "model_state_sha256": sels[k]["model_state_sha256"]}
             for k in A.SEED_ARMS}
    nsl = {v: sum(1 for e in entries if e[0] == v) for v in vols}
    table = I3.volume_table(vols, M, ident, nsl, strat)
    rel = I3.reliability(repro, max_decomp, max_spec, res["D-JAC"])
    t0 = time.monotonic()
    figs = I3.figures(res, M, vols, strat, ret_imgs)
    t_plot = time.monotonic() - t0
    mods = ctx["loaded_modules"]()           # pin check of imported modules
    logger.info("STAGE G5 %d imported pinned modules verified", len(mods))
    A.check_deadline(hard, "write")
    shas = I3.write_outputs(budget, out, res, table, figs, rel)
    A.check_deadline(hard, "after outputs")
    logger.info("STAGE G5 outputs written: %s", sorted(shas))

    rec = {"schema": "SEQREF-MRI-DIAG-JSH-GRUN/0.1",
           "g": {"abbr": __abbr__, "version": __version__,
                 "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime())},
           "manifest_sha256": man_sha, "resources": R, "code_sha256": code,
           "limits": {k: lim[k] for k in ("reserve", "T_s", "max_wall_s")},
           "timings_s": {"load": t_loads, "replay": t_reps,
                         "analysis": t_boot, "plot": t_plot,
                         "total_before_record": time.monotonic() - t_all},
           "disk_checks": disk, "pre_guard_files": pre,
           "loaded_pinned_modules": mods, "registered_gd4_sha256":
           gd4["sha256"],
           "outputs_sha256": shas}
    body = (json.dumps(rec, sort_keys=True, indent=1, allow_nan=False) +
            "\n").encode()
    sha = A.sha256_bytes(body)
    side = f"{sha}  g_run.json\n".encode()
    A.check_deadline(hard, "before run record")
    budget.check(len(body) + len(side), "before run record")
    budget.write_new(out / "g_run.json", body, "run record")
    budget.write_new(out / "g_run.json.sha256", side, "run record sidecar")
    logger.info("STAGE G6 run record written; reconciling bytes")
    for hd in logging.getLogger().handlers:
        hd.flush()
    on_disk = A.tree_bytes(out)
    K.require(on_disk == budget.used, "DIAG_UNBUDGETED_BYTES",
              f"{on_disk} bytes under OUT != budgeted {budget.used}")
    A.check_deadline(hard, "before PASS")
    return sha


def main(argv=None) -> int:
    a = _args(argv)
    code_dir = Path(__file__).resolve().parent
    if code_dir != DIAG_ROOT / "code":
        print("STOP usage: run from DIAG_JSH/code", file=sys.stderr)
        return 2
    sys.path.insert(0, str(code_dir))
    import diag_jsh_adapter as A
    import diag_jsh_core as K
    import diag_jsh_f1_prepare as F1
    import diag_jsh_f2_resources as F2R
    out = Path(a.out)
    try:
        F1.check_out(out, DIAG_ROOT / "out")
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    if a.max_wall_s < MIN_WALL_S or a.reserve_bytes < 0:
        print(f"STOP usage: MAX_WALL_S >= {MIN_WALL_S}, reserve >= 0",
              file=sys.stderr)
        return 2
    t0 = time.monotonic()
    try:
        R = dict(F2R.RESOURCES)
        budget, pre = start(out, A, K, a.reserve_bytes, R)
        lr = check_resources(R, a.manifest_sha256, a, K)
        lim = {"reserve": a.reserve_bytes, **lr,
               "deadline": t0 + min(lr["T_s"], a.max_wall_s - WALL_MARGIN_S)}
        want = {f: getattr(a, f"sha_{f.split('.')[0]}") for f in CODE_FILES}
        code = A.check_code(code_dir, want)
        man = load_manifest(a.manifest, a.manifest_sha256, A, K)
        import diag_jsh_i3 as I3
        ctx = F1.bind_production(a, A, K)
        sha = run(ctx, man, a.manifest_sha256, R, out, lim, code, A, K, I3,
                  budget, pre)
    except K.DiagError as exc:
        logger.error("STOP %s", exc)
        print(f"RESULT DIAG_G STOP {exc.code}")
        return 1
    except Exception as exc:  # noqa: BLE001 -- logged, never swallowed
        logger.error("UNEXPECTED %s:\n%s", type(exc).__name__,
                     traceback.format_exc())
        print(f"RESULT DIAG_G FAILED UNEXPECTED {type(exc).__name__}")
        return 2
    print(f"RESULT DIAG_G PASS run_record_sha256={sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
