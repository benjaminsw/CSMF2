# SEQREF-MRI-DIAGJSHADP v0.3.1 -- DIAG-JSH real-data adapter (DIAG-I2; frozen mini-spec v1.0 §1.1-§1.3, §5, §6, §8)
# LIFETIME: KEEP
# =============================================================================
# Feeds the pinned V05LT data, checkpoints and decodes into the scientific
# core (diag_jsh_core). Every pinned call is the V05LT implementation itself
# (imported from the bound V05LT tree; nothing re-implemented except the
# per-slice loop, which mirrors v05lt_metrics.validate step by step):
#   * write guard (§8): audit hook; any write outside the diagnostic root
#     halts the process (exit 77, DIAG_WRITE_OUTSIDE); installed before the
#     pinned imports;
#   * source identity (§1.1): pinned ProductionStages._check_loaded on every
#     loaded module + a per-module hash record for the manifest; diagnostic
#     code files checked against their reviewed hashes;
#   * inputs: run verified by pinned v05lt_results.load_run (ledger chain,
#     retained hashes, row files, re-derived selection); selected checkpoint
#     = the registered "selected" role (best validation PSNR); weights
#     loaded by pinned v05lt_eval.load_trained_checkpoint (file sha256 BEFORE
#     torch.load, meta, model-state sha256); ZF by pinned load_zf;
#   * reproduction (§1.3): R0 replay of every slice with pinned slice states
#     (tiny_gate._build_slice_states), NLL (log_prob_free, nll_components),
#     decode (_decode_z) and PSNR (psnr_record); masks regenerated with the
#     pinned rule and compared by sha256 of the C-order bool bytes; exact
#     identities and the frozen numerical limits via core.check_replay;
#   * population (§1.3), stratifiers and thirds (§5), retained slices (§6).
# Failures: logged DiagError(code) (core.fail); pinned V05LTError is re-raised
#   as DiagError with the pinned code in the message. No fallback, no
#   placeholder, no mock: test fixtures are synthetic data run through the
#   real pinned code and are labelled as such in the test file.
# Changelog (v0.3, 2026-11-09; F1 review 2):
#   * WriteBudget: bytes reserved BEFORE every write (budget and live free
#     space vs reserve); strict guard: only budget-registered files may be
#     opened for writing inside the root, no other file operation;
#     RLIMIT_FSIZE backstop; BudgetLogHandler (allowance; halt 78);
#   * S-M: zero-MSE slices counted, not compared.
# Changelog (v0.2, 2026-11-09; F1 review):
#   * write guard fail-closed: absolute resolvable write paths only (the
#     "open" audit event has no dir_fd), process creation refused except
#     read-only git; root must be reached without symlinks;
#   * check_disk / check_deadline helpers; replay() enforces a deadline;
#   * selected_checkpoint requires the row-file name/hash and records a
#     resolvable path and hashed selection evidence (no values).
# Changelog (v0.1, 2026-11-09):
#   * first adapter: write guard, source checks, selected-checkpoint binding,
#     full R0 replay with spectral records, population checks, H-D/H-F
#     thirds, retained-slice rule; tests in test_diag_jsh_adapter.py.
# =============================================================================
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

import diag_jsh_core as K

__abbr__ = "SEQREF-MRI-DIAGJSHADP"
__version__ = "0.3.1"
logger = logging.getLogger("seqref_diag.jsh_adapter")

WRITE_EXIT = 77                        # write guard halt (distinct from 75)
SEED_ARMS = tuple((s, a) for s in K.SEEDS for a in ("lam0", "lam2"))
SELECTED_ROLE = "best_psnr"            # retained role of the selected weights


# -- pinned-call wrapper ------------------------------------------------------
def pinned(code: str, where: str, fn, *a, **k):
    """Run a pinned call; V05LTError -> DiagError(code) with the pinned code
    kept in the message; any other exception logged and re-raised typed."""
    from seqref_mri.scripts import v05lt_common as C
    try:
        return fn(*a, **k)
    except K.DiagError:
        raise
    except C.V05LTError as exc:
        K.fail(code, f"{where}: {exc.code}: {exc}")
    except Exception as exc:  # noqa: BLE001 -- logged, re-raised typed
        logger.exception("%s: unexpected %s", where, type(exc).__name__)
        K.fail(code, f"{where}: {type(exc).__name__}: {exc}")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def canonical_sha(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode())


# -- write guard (§8) ------------------------------------------------------------
# Fail-closed policy (v0.2):
#   * a write event must name an ABSOLUTE path whose resolved target lies in
#     the root; relative paths are refused because the "open" audit event
#     does not carry dir_fd (descriptor-relative writes cannot be resolved);
#     integer descriptors are allowed only for already-open files;
#   * the only path allowed outside the root is /dev/null;
#   * process creation is refused, except `git -C <repo> rev-parse|ls-tree`
#     (read-only; pinned binding) and exactly `uname -p` (read-only;
#     platform.processor() during imports);
#   * native (C-level) writes bypass audit hooks: not covered (residual; the
#     pinned code paths used here make none, F1 checks the bytes in its root).
_GUARD = {"root": None, "git_repo": None, "files": None}  # files: strict
_WFLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_PATH_EVENTS = {"os.mkdir": (0,), "os.remove": (0,), "os.rmdir": (0,),
                "os.truncate": (0,), "os.chmod": (0,), "os.chown": (0,),
                "os.utime": (0,), "os.rename": (0, 1), "os.replace": (0, 1),
                "os.symlink": (1,), "os.link": (0, 1), "os.mkfifo": (0,),
                "os.mknod": (0,), "shutil.rmtree": (0,),
                "shutil.copyfile": (1,), "shutil.move": (0, 1),
                "shutil.copytree": (1,), "shutil.chown": (0,)}
_PROC_EVENTS = ("os.system", "os.exec", "os.spawn", "os.posix_spawn",
                "os.fork", "os.forkpty", "os.startfile", "pty.spawn")
_GIT_READ = ("rev-parse", "ls-tree")
_READ_ONLY_ARGV = {("uname", "-p")}     # platform.processor() at import
_SINKS = ("/dev/null",)                 # writes consume nothing (imports)


def _inside(p) -> bool:
    if isinstance(p, int) and not isinstance(p, bool):
        return True                     # already-open descriptor
    if isinstance(p, bytes):
        p = os.fsdecode(p)
    if not isinstance(p, (str, os.PathLike)):
        return False
    p = os.fspath(p)
    if not os.path.isabs(p):
        return False                    # cannot resolve dir_fd-relative
    q = os.path.realpath(p)
    if q in _SINKS:
        return True
    root = _GUARD["root"]
    return q == root or q.startswith(root + os.sep)


def _halt(event: str, path) -> None:
    logger.error("DIAG_WRITE_OUTSIDE: %s %r outside %s", event, path,
                 _GUARD["root"])
    for h in logging.getLogger().handlers + logger.handlers:
        try:
            h.flush()
        except Exception:  # noqa: BLE001 -- halting anyway
            pass
    sys.stderr.write(f"DIAG_WRITE_OUTSIDE {event} {path!r}\n")
    sys.stderr.flush()
    os._exit(WRITE_EXIT)


def _git_ok(args) -> bool:
    a = [os.fsdecode(x) if isinstance(x, bytes) else str(x)
         for x in (args or [])]
    if tuple(a) in _READ_ONLY_ARGV:
        return True
    return (len(a) >= 4 and os.path.basename(a[0]) == "git" and
            a[1] == "-C" and _GUARD["git_repo"] is not None and
            os.path.realpath(a[2]) == _GUARD["git_repo"] and
            a[3] in _GIT_READ)


def _audit(event: str, args) -> None:
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")
                   ) or (isinstance(flags, int) and bool(flags & _WFLAGS))
        if writing and not _inside(path):
            _halt(event, path)
        if writing and _GUARD["files"] is not None and not (
                isinstance(path, int) or os.path.realpath(os.fspath(path)) in
                _GUARD["files"] or os.path.realpath(os.fspath(path)) in _SINKS):
            _halt(event + " (unbudgeted)", path)
    elif event in _PATH_EVENTS:
        if _GUARD["files"] is not None:      # strict: no other file operation
            _halt(event + " (strict)", args[0] if args else args)
        for i in _PATH_EVENTS[event]:
            if i < len(args) and not _inside(args[i]):
                _halt(event, args[i])
    elif event == "subprocess.Popen":
        if not _git_ok(args[1] if len(args) > 1 else None):
            _halt(event, args[1] if len(args) > 1 else args)
    elif event in _PROC_EVENTS:
        _halt(event, args)


def install_write_guard(root, git_repo=None) -> str:
    """Restrict every write of this process to `root` (§8). Once only. The
    root must be an existing directory reached without symlinks."""
    K.require(_GUARD["root"] is None, "DIAG_GUARD", "guard already installed")
    ap = os.path.abspath(os.fspath(root))
    r = os.path.realpath(ap)
    K.require(os.path.isdir(r) and r == ap, "DIAG_GUARD",
              f"root {ap} missing or reached through a symlink ({r})")
    _GUARD["root"] = r
    if git_repo is not None:
        _GUARD["git_repo"] = os.path.realpath(os.fspath(git_repo))
    sys.addaudithook(_audit)
    logger.info("write guard on %s", r)
    return r


# -- resource limits (§8) -----------------------------------------------------------
LOG_EXIT = 78                         # log allowance exhausted (halt)


class WriteBudget:
    """Enforced F1 write budget (v0.3). Every byte F1 writes is reserved
    here BEFORE it is written: used + n <= budget and live free space
    (measured before the write) >= reserve + (budget - used). Attaching the budget makes the guard
    strict: inside the root only files registered here may be opened for
    writing, and no other file operation (mkdir, rename, remove, ...) is
    allowed. RLIMIT_FSIZE = budget is a kernel backstop per file (the
    kernel may write partially up to the limit; write_new checks the size).
    Not covered: native (C-level) writes, which bypass audit hooks."""

    def __init__(self, root, reserve: int, budget: int):
        import resource
        import signal
        K.require(_GUARD["root"] == os.path.realpath(os.fspath(root)),
                  "DIAG_GUARD", "budget root != guard root")
        K.require(reserve >= 0 and budget > 0, "DIAG_DISK", "limits")
        self.root, self.reserve, self.budget = _GUARD["root"], reserve, budget
        self.used, self.log = 0, []
        _GUARD["files"] = set()
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)   # EFBIG, not a kill
        soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
        lim = budget if hard == resource.RLIM_INFINITY else min(budget, hard)
        resource.setrlimit(resource.RLIMIT_FSIZE, (lim, hard))
        self.check(0, "budget attached")

    def why_not(self, n: int):
        """(reason or None, free) -- no logging (safe inside the handler)."""
        import shutil
        free = shutil.disk_usage(self.root).free
        if n < 0 or self.used + n > self.budget:
            return f"{self.used} + {n} > budget {self.budget}", free
        # free is measured BEFORE the n bytes are written: after the write
        # free - n must still cover reserve + (budget - used - n), i.e.
        # free >= reserve + (budget - used)  (v0.3.1 fix)
        if free < self.reserve + (self.budget - self.used):
            return (f"free {free} < reserve {self.reserve} + unused budget "
                    f"{self.budget - self.used}"), free
        return None, free

    def check(self, n: int, where: str) -> dict:
        why, free = self.why_not(n)
        K.require(why is None, "DIAG_DISK", f"{where}: {why}")
        return {"where": where, "used": self.used, "free": free}

    def take(self, n: int, where: str) -> None:
        self.check(n, where)
        self.used += n

    def register(self, path) -> str:
        p = os.path.realpath(os.fspath(path))
        K.require(p.startswith(self.root + os.sep) and not os.path.exists(p),
                  "DIAG_GUARD", f"{p}: outside root or exists")
        _GUARD["files"].add(p)
        return p

    def write_new(self, path, data: bytes, where: str) -> str:
        """Write-once file: reserve, register, create exclusively, write."""
        self.take(len(data), where)
        p = self.register(path)
        with open(p, "xb") as fh:
            fh.write(data)
        K.require(os.path.getsize(p) == len(data), "DIAG_DISK",
                  f"{where}: size after write")
        return p


class BudgetLogHandler(logging.Handler):
    """Log file whose every record is reserved in the budget first, within
    its own allowance; exhausting either halts the process (exit 78)."""

    def __init__(self, budget: WriteBudget, path, allowance: int):
        super().__init__()
        self.b, self.allowance, self.n = budget, allowance, 0
        self.fh = open(budget.register(path), "ab", buffering=0)

    def emit(self, record):
        data = (self.format(record) + "\n").encode("utf-8", "replace")
        why = (f"log allowance {self.allowance} reached"
               if self.n + len(data) > self.allowance
               else self.b.why_not(len(data))[0])
        if why is not None:                 # no logging here (recursion)
            sys.stderr.write(f"DIAG_LOG_BUDGET {why}\n")
            sys.stderr.flush()
            os._exit(LOG_EXIT)
        self.b.used += len(data)
        self.fh.write(data)
        self.n += len(data)


def tree_bytes(root) -> int:
    return sum(p.stat().st_size for p in Path(root).rglob("*")
               if p.is_file() and not p.is_symlink())


def check_disk(root, reserve: int, budget: int, where: str) -> dict:
    """Live free space >= reserve + remaining budget, and the bytes already
    written under root <= budget. Called before preparation, after each
    checkpoint and before every space-consuming write."""
    import shutil
    used = tree_bytes(root)
    free = shutil.disk_usage(root).free
    K.require(used <= budget, "DIAG_DISK", f"{where}: written {used} > "
              f"budget {budget}")
    K.require(free >= reserve + (budget - used), "DIAG_DISK",
              f"{where}: free {free} < reserve {reserve} + remaining "
              f"budget {budget - used}")
    return {"where": where, "used": used, "free": free}


def check_deadline(deadline: float, where: str) -> None:
    import time
    K.require(time.monotonic() <= deadline, "DIAG_DEADLINE",
              f"{where}: preparation time limit reached")


# -- source identity (§1.1) -----------------------------------------------------
def check_loaded(stages, pin_code: dict, root=None, modules=None) -> list:
    """Pinned loaded-module rule, then the per-module record for the
    manifest: [{module, path, sha256}] for every seqref_mri module."""
    pinned("DIAG_PIN_MISMATCH", "loaded modules", stages._check_loaded,
           pin_code, root, modules)
    table = sys.modules if modules is None else modules
    rec = []
    for name in sorted(table):
        f = getattr(table[name], "__file__", None)
        if f and (name == "seqref_mri" or name.startswith("seqref_mri.")):
            rec.append({"module": name, "path": os.path.realpath(f),
                        "sha256": sha256_file(f)})
    return rec


def check_code(code_dir, expected: dict) -> dict:
    """Diagnostic code files = their reviewed sha256 (and loaded from there)."""
    code_dir = Path(code_dir).resolve()
    got = {}
    for name, sha in sorted(expected.items()):
        p = code_dir / name
        K.require(p.is_file() and not p.is_symlink(), "DIAG_CODE_MISMATCH",
                  f"{p}: missing or not a regular file")
        got[name] = sha256_file(p)
        K.require(got[name] == sha, "DIAG_CODE_MISMATCH",
                  f"{name}: {got[name]} != reviewed {sha}")
    for mod in (K, sys.modules[__name__]):
        f = Path(mod.__file__).resolve()
        K.require(f.parent == code_dir and f.name in expected,
                  "DIAG_CODE_MISMATCH", f"{mod.__name__} loaded from {f}")
    return got


# -- masks (§1.3) -----------------------------------------------------------------
def mask_identity(item: dict, f: str, s: int) -> tuple:
    """(regenerated sha, loaded sha): pinned canonical_mask_seed +
    make_cartesian_mask at the loaded width, C-order bool bytes (as
    v05lt_prodstages.expected_masks_sha256). The loaded mask seed must equal
    the registered seed."""
    from seqref_mri.src import fastmri_data as FD
    seed = FD.canonical_mask_seed(FD.EVAL_BASE_SEED, f, int(s))
    w = int(item["y"].shape[-1])
    m = FD.make_cartesian_mask(w, seed).astype(bool)
    K.require(int(item["meta"]["mask_seed"]) == seed, "DIAG_MASK_SEED",
              f"{f}|{s}: loaded seed {item['meta']['mask_seed']} != {seed}")
    got = np.asarray(item["mask"]).astype(bool).reshape(-1)
    return (sha256_bytes(np.ascontiguousarray(m).tobytes(order="C")),
            sha256_bytes(np.ascontiguousarray(got).tobytes(order="C")))


# -- inputs -----------------------------------------------------------------------
def selected_checkpoint(run: dict) -> dict:
    """Registered selected checkpoint of a load_run() result: epoch and
    model state from the re-derived selection; weights = retained
    best_psnr = weights_e<NNN>.pt; rows = that epoch's verified row file."""
    K.require(isinstance(run, dict) and "checkpoints" in run,
              "DIAG_INPUT_INVALID", "run record")
    sel = run["checkpoints"]["selected"]
    e, ms = sel["epoch"], sel["model_state_sha256"]
    w = run["retained"].get(SELECTED_ROLE)
    K.require(isinstance(w, dict) and w.get("name") == f"weights_e{e:03d}.pt",
              "DIAG_SELECTION_MISMATCH", f"{run.get('arm')}/{run.get('seed')}:"
              f" retained {SELECTED_ROLE} {w} != epoch {e}")
    doc = run["rows"].get(e)
    K.require(isinstance(doc, dict) and doc.get("model_state_sha256") == ms
              and doc.get("epoch") == e, "DIAG_SELECTION_MISMATCH",
              f"{run.get('arm')}/{run.get('seed')}: rows of epoch {e} not "
              "bound to the selected model state")
    vr = run["retained"].get(f"val_rows_e{e:03d}")
    K.require(isinstance(vr, dict) and isinstance(vr.get("name"), str) and
              isinstance(vr.get("sha256"), str) and len(vr["sha256"]) == 64,
              "DIAG_SELECTION_MISMATCH", f"{run.get('arm')}/{run.get('seed')}"
              f": retained row file of epoch {e} missing")
    ep = run.get("epochs")
    K.require(isinstance(ep, dict) and e in ep, "DIAG_SELECTION_MISMATCH",
              "epoch records missing")
    psnr = {str(k): ep[k]["validation"]["all_slices"]["psnr"]
            for k in sorted(ep)}
    best = max(sorted(ep), key=lambda k: (ep[k]["validation"]["all_slices"]
                                          ["psnr"], -k))
    K.require(best == e, "DIAG_SELECTION_MISMATCH", f"selected epoch {e} "
              f"!= first best-PSNR epoch {best}")
    ck = Path(run["ckpt_dir"])
    return {"seed": run["seed"], "arm": run["arm"], "epoch": e,
            "model_state_sha256": ms, "weights": str(ck / w["name"]),
            "weights_sha256": w["sha256"], "rows_file": str(ck / vr["name"]),
            "rows_sha256": vr["sha256"], "rows": doc,
            "selection_evidence": {
                "rule": "selected = first epoch with the highest all-slice "
                        "validation PSNR (pinned load_run re-derivation)",
                "epochs": len(ep),
                "per_epoch_psnr_sha256": canonical_sha(psnr)}}


def load_selected(stages, seed: int, arm: str) -> dict:
    """Pinned load_run (full run verification) -> selected checkpoint."""
    from seqref_mri.scripts import v05lt_results as RS
    rdir = stages.run_dir(seed, arm)
    K.require(Path(rdir).is_dir(), "DIAG_INPUT_MISSING", f"{rdir}")
    man = pinned("DIAG_INPUT_INVALID", "manifests", stages.manifests)
    run = pinned("DIAG_INPUT_INVALID", f"load_run {seed}_{arm}", RS.load_run,
                 rdir, arm, seed, man, RS.Identities())
    return selected_checkpoint(run)


def load_model(sel: dict):
    """Pinned load_trained_checkpoint: file sha256 checked before
    torch.load, meta and model-state sha256 verified after."""
    from seqref_mri.scripts import v05lt_eval as EV
    K.require(Path(sel["weights"]).is_file(), "DIAG_INPUT_MISSING",
              sel["weights"])
    model, vc = pinned("DIAG_CHECKPOINT_MISMATCH", sel["weights"],
                       EV.load_trained_checkpoint, Path(sel["weights"]),
                       sel["weights_sha256"], sel["model_state_sha256"],
                       seed=sel["seed"], arm=sel["arm"], epoch=sel["epoch"])
    return model, vc


def load_zf(stages) -> dict:
    from seqref_mri.scripts import v05lt_results as RS
    ins = pinned("DIAG_INPUT_INVALID", "bound inputs", stages._bound)
    man = pinned("DIAG_INPUT_INVALID", "manifests", stages.manifests)
    d = Path(ins["zf"]["path"]).parent
    return pinned("DIAG_INPUT_INVALID", "ZF", RS.load_zf, d,
                  ins["zf"]["sha256"], man, RS.Identities())


# -- population (§1.3 item 1) ------------------------------------------------------
def entries_of(doc: dict) -> list:
    return [(r["volume_id"], r["file"], r["slice_index"]) for r in doc["rows"]]


def check_population(docs: dict, zf_rows: list) -> list:
    """All six row documents: identical slice_ids_sha256 (one ordered
    population, §1.4); ZF rows cover exactly that (file, slice) set with the
    same volume IDs. -> entries in validation order."""
    K.require(set(docs) == set(SEED_ARMS), "DIAG_POPULATION_MISMATCH",
              f"runs {sorted(docs)}")
    shas = {k: d["slice_ids_sha256"] for k, d in docs.items()}
    K.require(len(set(shas.values())) == 1, "DIAG_POPULATION_MISMATCH",
              f"slice_ids_sha256 differ: {shas}")
    ent = entries_of(docs[SEED_ARMS[0]])
    for k, d in docs.items():
        K.require(entries_of(d) == ent, "DIAG_POPULATION_MISMATCH",
                  f"{k}: case order")
    want = {(f, s): v for v, f, s in ent}
    got = {(r["file"], r["slice_index"]): r["volume_id"] for r in zf_rows}
    K.require(len(got) == len(zf_rows) and got == want,
              "DIAG_POPULATION_MISMATCH", "ZF rows != validation population "
              f"({len(zf_rows)} rows, {len(set(got) ^ set(want))} differing "
              "slices)")
    return ent


def stored_view(doc: dict, mask_regen: dict) -> dict:
    """Stored side of check_replay; the NLL identity is checked per slice
    (§3) before anything is aggregated."""
    from seqref_mri.src import free_flow_runtime as ffr
    out = {}
    for r in doc["rows"]:
        K.check_nll_identity(r, ffr.FLOW_DIM_REAL)
        k = (r["volume_id"], r["file"], r["slice_index"])
        K.require(k not in out, "DIAG_POPULATION_MISMATCH", f"duplicate {k}")
        out[k] = {"ckpt": doc["model_state_sha256"],
                  "mask": mask_regen[(r["file"], r["slice_index"])],
                  "psnr": r["psnr"], "nll_per_dim": r["nll_per_dim"]}
    return out


# -- R0 replay (§1.3 item 2, §4 inputs) -------------------------------------------
def replay(model, vv, eligibility: dict, p4: dict, s_ref: float,
           positions=None, on_slice=None, deadline=None) -> dict:
    """Per-slice R0 replay through the pinned pieces, in the batches and view
    order of v05lt_metrics.validate. Returns {"rows": {key: replay row},
    "mask_regen": {(f, s): sha}, "hf": {(f, s): float}, "sm_rel_dev": max
    |S-M pixel MSE - pinned MSE| / pinned MSE, "ckpt": model-state sha}.
    on_slice(key, row, spec, x_hat_c128, x_true_c128) receives each slice
    (no images are kept here)."""
    import torch
    from seqref_mri.scripts import v05lt_common as C
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.scripts import tiny_gate as tg
    from seqref_mri.scripts.train_base import _collate, _prepare
    from seqref_mri.src import free_flow_runtime as ffr
    idx = list(range(len(vv))) if positions is None else list(positions)
    K.require(len(idx) > 0, "DIAG_POPULATION_EMPTY", "no positions")
    ckpt = C.model_state_sha256(model)
    was = model.training
    model.eval()
    rows, regen, hf, smdev, smzero = {}, {}, {}, 0.0, 0
    z0 = torch.zeros(1, ffr.FLOW_DIM_REAL)
    try:
        with torch.no_grad():
            for start in range(0, len(idx), M.BATCH):
                if deadline is not None:
                    check_deadline(deadline, f"replay position {start}")
                part = idx[start:start + M.BATCH]
                items = [pinned("DIAG_DATA_LOAD", f"view {i}", vv.__getitem__,
                                i) for i in part]
                batch = pinned("DIAG_DATA_LOAD", f"positions {part[:4]}",
                               _collate, items)
                ents = [vv.entry(i) for i in part]
                sel = {"ordered_identities": [{"file": f, "slice_index": s}
                                              for f, s in ents]}
                states = pinned("DIAG_STATE_FAILED", "slice states",
                                tg._build_slice_states, batch, sel, p4, s_ref)
                prep = pinned("DIAG_STATE_FAILED", "prepare", _prepare, batch,
                              "cpu", test0=False)
                tgt = torch.from_numpy(np.concatenate(
                    [st["target"] for st in states], 0).astype(np.float32))
                cond = torch.cat([st["cond"] for st in states], 0)
                msk = torch.cat([st["mask"] for st in states], 0)
                d = int(tgt.shape[1])
                K.require(d == ffr.FLOW_DIM_REAL, "DIAG_DFREE", str(d))
                lp = pinned("DIAG_NLL_FAILED", "log_prob_free",
                            model.log_prob_free, tgt, cond, msk)
                prior, ldj = pinned("DIAG_NLL_FAILED", "nll components",
                                    M.nll_components, model, tgt, cond, msk)
                lp64 = lp.to(torch.float64)
                dev = ((prior + ldj).to(torch.float64) - lp64).abs()
                nll = (-lp64 / d).tolist()
                npr = (-prior.to(torch.float64) / d).tolist()
                nld = (-ldj.to(torch.float64) / d).tolist()
                for j, (st, (f, s)) in enumerate(zip(states, ents)):
                    K.require((f, s) in eligibility, "DIAG_POPULATION_MISMATCH",
                              f"{f}|{s} not in eligibility")
                    el = eligibility[(f, s)]
                    K.require(bool(el["eligible"]) == (not st["excluded"]),
                              "DIAG_POPULATION_MISMATCH",
                              f"{f}|{s}: eligibility flag")
                    regen[(f, s)], loaded = mask_identity(items[j], f, s)
                    x1 = prep["x_norm"][j]
                    x_true = torch.complex(x1[0], x1[1])
                    K.require(bool(torch.equal(x_true.abs(), st["x_true_mag"])),
                              "DIAG_STATE_MISMATCH", f"{f}|{s}: reference "
                              "image != slice-state magnitude")
                    x0, _u0 = pinned("DIAG_DECODE_FAILED", f"R0 {f}|{s}",
                                     tg._decode_z, model, z0, st)
                    rec = pinned("DIAG_METRIC_FAILED", f"PSNR {f}|{s}",
                                 M.psnr_record, x0.abs(), st["x_true_mag"])
                    xh = np.asarray(x0.detach().cpu().numpy(), np.complex128)
                    xt = np.asarray(x_true.detach().cpu().numpy(),
                                    np.complex128)
                    m = np.asarray(st["mask"].reshape(-1).cpu().numpy())
                    spec = K.spectral_record(xh, xt, m.astype(np.int64))
                    if rec["mse"] == 0.0:
                        smzero += 1         # relative deviation undefined
                    else:
                        smdev = max(smdev, abs(spec["sm_pixel_mse"] -
                                               rec["mse"]) / rec["mse"])
                    hf[(f, s)] = K.hf_fraction(xt)
                    key = (el["volume_id"], f, s)
                    row = {"ckpt": ckpt, "mask": loaded,
                           "psnr": float(rec["psnr"]),
                           "nll_per_dim": float(nll[j]),
                           "nll_prior_per_dim": float(npr[j]),
                           "nll_logdet_per_dim": float(nld[j]),
                           "nll_decomp_abs_dev": float(dev[j])}
                    for v in row.values():
                        K.require(not isinstance(v, float) or math.isfinite(v),
                                  "DIAG_NON_FINITE", f"{f}|{s}: replay {v!r}")
                    rows[key] = row
                    if on_slice is not None:
                        on_slice(key, row, spec, xh, xt)
    finally:
        model.train(was)
    return {"rows": rows, "mask_regen": regen, "hf": hf, "sm_rel_dev": smdev,
            "sm_zero_mse_slices": smzero,
            "ckpt": ckpt}


def reproduce(doc: dict, rep: dict, expected: int) -> dict:
    """§1.3 reproduction check for one checkpoint (exact identities, frozen
    numerical limits). rep = replay() over the full population."""
    stored = stored_view(doc, rep["mask_regen"])
    return K.check_replay(stored, rep["rows"], expected)


def reproduce_prefix(doc: dict, rep: dict) -> dict:
    """Early check (F1 step 4): a replay of the first m view positions vs
    the first m stored rows. Never a substitute for reproduce()."""
    m = len(rep["rows"])
    K.require(m >= 1, "DIAG_POPULATION_EMPTY", "no replayed slices")
    head = {**doc, "rows": doc["rows"][:m]}
    return K.check_replay(stored_view(head, rep["mask_regen"]), rep["rows"], m)


# -- stratifiers, thirds, retained slices (§5, §6) ------------------------------------
def check_reference_equal(hfs: list) -> dict:
    """H-F is arm-independent: identical per-slice values from every
    checkpoint's replay (same reference images)."""
    K.require(len(hfs) >= 1, "DIAG_REFERENCE_MISMATCH", "no replays")
    for h in hfs[1:]:
        K.require(h == hfs[0], "DIAG_REFERENCE_MISMATCH",
                  "reference HF fractions differ between checkpoints")
    return hfs[0]


def stratifiers(entries: list, zf_rows: list, hf: dict) -> dict:
    """H-D (volume mean ZF PSNR; lower = harder, first third hardest) and
    H-F (volume mean HF fraction; first third highest)."""
    zp = {(r["file"], r["slice_index"]): r["psnr"] for r in zf_rows}
    hd = K.volume_means({k: float(v) for k, v in zp.items()}, entries)
    hfv = K.volume_means(hf, entries)
    g_hd = K.thirds(hd, descending=False)
    g_hf = K.thirds(hfv, descending=True)
    return {"H-D": {"values": hd, "thirds": g_hd,
                    "sha256": canonical_sha({"values": hd, "thirds": g_hd})},
            "H-F": {"values": hfv, "thirds": g_hf,
                    "sha256": canonical_sha({"values": hfv, "thirds": g_hf})}}


def retained_slices(entries: list, eligibility: dict) -> list:
    """§6: sorted volume IDs; default_rng(DIAG_SEED + 1).choice(V, 4,
    replace=False); the registered midpoint slice of each drawn volume."""
    vols = sorted({v for v, _, _ in entries})
    K.require(len(vols) >= 4, "DIAG_RETAINED", f"{len(vols)} volumes")
    pick = np.random.default_rng(K.DIAG_SEED + 1).choice(len(vols), 4,
                                                         replace=False)
    out = []
    for i in pick.tolist():
        mids = [(f, s) for v, f, s in entries if v == vols[i] and
                eligibility[(f, s)]["midpoint"]]
        K.require(len(mids) == 1, "DIAG_RETAINED",
                  f"{vols[i]}: {len(mids)} midpoint slices")
        out.append({"volume_id": vols[i], "file": mids[0][0],
                    "slice_index": mids[0][1]})
    return out
