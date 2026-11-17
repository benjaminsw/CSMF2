# SEQREF-DUNC-F11A v0.1.1 -- test_diag_unc_fixture11a.py: DIAG-UNC fixture 11a (source and metadata bindings; plan SEQREF-DUNC-IMPL v0.9.1)
# LIFETIME: KEEP
"""Fixture 11a of DIAG-UNC (223 §10 item 11): pinned-source identity, coefficient order, per-slice
band map, mask regeneration and role bindings. Opens NO MRI file and NO checkpoint, decodes nothing,
computes no metric.

Layout expected by the run card:  <run_dir>/code/{this file, diag_unc_core.py, diag_jsh_core.py}
Evidence written to <run_dir>:    fixture11a.log, fixture11a.log.sha256, RESULT.txt

Result rule: the log holds the CHECKS verdict only; the final F11A_RESULT is decided after the log
is closed, its sidecar written and re-verified and RESULT.txt written and read back.

Production mode : no arguments; every constant below is used; nothing can be overridden.
Self-test mode  : --selftest CONFIG.json (mock repository and synthetic records). Output is marked
                  NOT FIXTURE-11A EVIDENCE. The config may replace: repo, pin, diag_commit, the
                  diag_jsh_core hash, the split-manifest, analysis-document and v05lt_eval hashes,
                  and may inject one evidence-write fault. Test hook only, not a fallback.

Exposure: JSON files are parsed, the whitelisted fields are copied and the rest is dropped before
any logging. Post-evaluation rows, r0/r1, losses, selection values, eligibility energies/ratios and
analysis outcomes are never used, compared or printed. Checkpoint files are identified by their
.sha256 sidecars only. h5py.File, torch.load and numpy.load are replaced by raising functions for
the duration of the checks (tripwire; instrumentation, not a mock).

Exit codes: 0 PASS, 1 FAIL or ABORTED. Every failure is logged. No fallback, placeholder or dummy.
Reported-only items (never gates): A2 band counts, A4 midpoint cross-check, per-slice mask and map
hashes.

The function bound() is copied unchanged from dunc_audit2.py v0.1.2 (SEQREF-DUNC-AUD2, archived at
ch2-diag f3794c3316a8dbfcf4480392e840cf388658c2e4, sha256 c60fcb98...78d0).

Changelog:
  v0.1.1 (2026-11-17) C7b: the validation-order hash is a gate again, as IMPL v0.9.1 requires. Its
    definition was verified in the pinned v05lt_eligibility.py (sha256 1449d74e...d6d1):
    provenance.validation_order_sha256 = SplitView.order_lines_sha256() of the validation view,
    i.e. sha256 over "<relpath>|<slice>\n" lines in view order, and the manifest rows are written
    in that order. v0.1 reported this comparison without gating it (reviewer: not acceptable
    without a plan amendment).
  v0.1 (2026-11-17) created from plan IMPL v0.9.1 with the two review clarifications
    (package-parent entries in the import check; role occurrences, states, two named coincidences).
Update summary: every other check is unchanged from v0.1.
"""
import argparse
import ast
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

VERSION = "SEQREF-DUNC-F11A v0.1.1"
PIN = "7455da8027f33a6cca4dbf72f04e72420e938f45"
DIAG_COMMIT = "d69b946f59abb014bf7b6cd21c842e54aa278a1f"
REPO = "/home/benjamin/CSMFII"
INTERPRETER = REPO + "/seqref_mri/.venv_mri/bin/python"

V05LT_MODULES = {   # module name -> (path in the pinned tree, registered sha256)
    "seqref_mri.src.fastmri_data": ("seqref_mri/src/fastmri_data.py", "610cc1d1d7968deebc88f645270e1baefb6589cb56841bdd327450ca1069cb44"),
    "seqref_mri.src.residual_decoder": ("seqref_mri/src/residual_decoder.py", "fd65e3a229e3df4f0226e276c92c916f5bd32cc77ed3655d68fb8a14608990d6"),
    "seqref_mri.src.preflight_io": ("seqref_mri/src/preflight_io.py", "ab01fefbf0c46ce08b8746e97250f420de3f219bc8a25a49e514bc4af3b19942"),
    "seqref_mri.scripts.v02_manifests": ("seqref_mri/scripts/v02_manifests.py", "5ef4c55121df0e5fa151c920bcafb75c55121c222afb7f0d73b19cdbc44749e9"),
    "seqref_mri.scripts.v05lt_common": ("seqref_mri/scripts/v05lt_common.py", "f35664fcb92607d0b0919ea80e55a90c99bbea9d6584b329c0d7688d9e3712a1"),
}
PACKAGE_PARENTS = ("seqref_mri", "seqref_mri.src", "seqref_mri.scripts")   # package entries, not scientific modules
DIAG_MODULES = {    # module name -> (path at DIAG_COMMIT, archived sha256)
    "diag_unc_core": ("seqref_mri/scripts/diag_unc/diag_unc_core.py", "a0d3b4a4079ff290a93098eb7a1c53173df596080a6b2b51469dc0f757c18b06"),
    "diag_jsh_core": ("seqref_mri/scripts/diag_jsh/diag_jsh_core.py", "85410c77472cf876021a49bd72254ac98461227887178e24cdb6d0ce5295c0fb"),
}
EVAL_SRC = ("seqref_mri/scripts/v05lt_eval.py", "10b4bc65d465e9c6d0a016c2d27977cd9f485ffb851936d9a15ad6a325898554")
SPLIT_SHA = "f7d21fb97e030a2e8227233cd5d1f171a120b765060ae83cafc728721428cf5d"
ANALYSIS_SHA = "5b267d3aa7ccb48bf3d7ccbfa2e2497d7f02df845aff8786c0fbe9b9dff2afa2"
Z_DIAG = "f8a6a34b6b474286359e4c99341de256b62bc0dff1e6bc8cf52e72f0461dc8d6"

HW, N_ACQ, N_FREE = 96, 24, 6912
CENTRE = set(range(44, 52))
N_VOLUMES, N_VAL_SLICES = 97, 3461
EVAL_BASE_SEED = 20261001
FLATTEN_ORDER = "row_major_row_then_ascending_unmeasured_column"
LEDGER_SCHEMA, POSTEVAL_SCHEMA = "seqref-v05lt-ledger/1", "seqref-v05lt-posteval/1"
RUNS = [(a, s) for a in ("lam0", "lam2") for s in (2026, 2027, 2028)]
ROLES = ("epoch3", "epoch10", "epoch30", "selected", "final")
ROLES_T1 = ("epoch3", "selected", "final")
EXPECT_SELECTED = {("lam0", 2026): 9, ("lam0", 2027): 10, ("lam0", 2028): 13,
                   ("lam2", 2026): 55, ("lam2", 2027): 48, ("lam2", 2028): 55}
EXPECT = {"role_occurrences": 30, "distinct_states": 28, "tier1_role_occurrences": 18,
          "tier1_distinct_states": 17, "posteval_records": 17, "rule_bindings": 29}
EXPECT_COINCIDENCES = {frozenset({("lam2", 2028, "selected"), ("lam2", 2028, "final")}),
                       frozenset({("lam0", 2027, "selected"), ("lam0", 2027, "epoch10")})}
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")

log = logging.getLogger("dunc_f11a")
FAILS = []
TRIP = []


class Abort(Exception):
    pass


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
        raise Abort("git %s failed (exit %d): %s" % (" ".join(args), r.returncode, r.stderr.decode(errors="replace").strip()))
    return r.stdout if binary else r.stdout.decode(errors="replace")


def gate(cid, ok, msg):
    if ok:
        log.info("[%s] PASS  %s", cid, msg)
    else:
        log.error("[%s] FAIL  %s", cid, msg)
        FAILS.append(cid)
    return bool(ok)


def info(cid, msg):
    log.info("[%s] INFO  %s", cid, msg)


def sidecar(path):
    """First 64-hex token of <path>.sha256, or None."""
    sc = path + ".sha256"
    if not os.path.isfile(sc):
        return None
    with open(sc, "r", encoding="utf-8", errors="replace") as fh:
        m = HEX64.search(fh.read())
    return m.group(0) if m else None


def load_json(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return json.loads(raw.decode("utf-8")), sha_bytes(raw)


# ---------------------------------------------------------------- pre-checks (nothing is created before these pass)
def prechecks(cfg):
    if sys.byteorder != "little":
        raise Abort("byte order is %s, little-endian required" % sys.byteorder)
    script = os.path.realpath(__file__)
    code_dir = os.path.dirname(script)
    run_dir = os.path.dirname(code_dir)
    want = {os.path.basename(script), "diag_unc_core.py", "diag_jsh_core.py"}
    have = set(os.listdir(code_dir))
    if have != want:
        raise Abort("code directory must hold exactly %s, found %s" % (sorted(want), sorted(have)))
    if set(os.listdir(run_dir)) != {os.path.basename(code_dir)}:
        raise Abort("run directory is not fresh: %s" % sorted(os.listdir(run_dir)))
    repo = cfg["repo"]
    if not cfg["selftest"]:
        if os.path.abspath(sys.executable) != INTERPRETER:
            raise Abort("interpreter %s is not %s" % (sys.executable, INTERPRETER))
        if not run_dir.startswith(repo + "/v06lt_evidence/DIAG_UNC/fixture11a_"):
            raise Abort("run directory %s is not under v06lt_evidence/DIAG_UNC/fixture11a_*" % run_dir)
    head = git(repo, "rev-parse", "HEAD").strip()
    if head != cfg["pin"]:
        raise Abort("HEAD %s is not the pin" % head)
    if git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() != cfg["branch"]:
        raise Abort("branch is not %s" % cfg["branch"])
    if git(repo, "status", "--porcelain").strip():
        raise Abort("Git status is not clean")
    if git(repo, "diff", "--cached", "--name-only").strip():
        raise Abort("something is staged")
    ident = {}
    for name, (rel, want_sha) in cfg["v05lt_modules"].items():
        tree, pinned = sha_file(os.path.join(repo, rel)), sha_bytes(git(repo, "show", "%s:%s" % (cfg["pin"], rel), binary=True))
        if not (tree == pinned == want_sha):
            raise Abort("%s: working tree %s / pin %s / registered %s differ" % (rel, tree[:12], pinned[:12], want_sha[:12]))
        ident[name] = tree
    for name, (rel, want_sha) in cfg["diag_modules"].items():
        exp, arch = sha_file(os.path.join(code_dir, name + ".py")), sha_bytes(git(repo, "show", "%s:%s" % (cfg["diag_commit"], rel), binary=True))
        if not (exp == arch == want_sha):
            raise Abort("%s: export %s / archive %s / registered %s differ" % (name, exp[:12], arch[:12], want_sha[:12]))
        ident[name] = exp
    eval_src = git(repo, "show", "%s:%s" % (cfg["pin"], EVAL_SRC[0]), binary=True)
    if sha_bytes(eval_src) != cfg["eval_sha"]:
        raise Abort("%s at the pin does not have the registered sha256" % EVAL_SRC[0])
    return {"script": script, "script_sha": sha_file(script), "code_dir": code_dir, "run_dir": run_dir,
            "head": head, "ident": ident, "eval_src": eval_src}


# ---------------------------------------------------------------- tripwire
def tripwire_on(mods):
    saved = []

    def make(label):
        def raiser(*_a, **_k):
            TRIP.append(label)
            log.error("[TRIP] %s was called: data or checkpoint access is forbidden in fixture 11a", label)
            raise RuntimeError("tripwire: %s" % label)
        return raiser
    for mod, attr in mods:
        saved.append((mod, attr, getattr(mod, attr)))
        setattr(mod, attr, make("%s.%s" % (mod.__name__, attr)))
    return saved


def tripwire_off(saved):
    for mod, attr, fn in saved:
        setattr(mod, attr, fn)


# ---------------------------------------------------------------- ledger (rule copied from dunc_audit2.py v0.1.2)
def bound(by_epoch, epoch, key, file_sha, state_sha):
    """Apply the binding rule. Returns (True, detail) or (False, reason)."""
    c = by_epoch.get(epoch)
    if c is None:
        return False, "no commit record for epoch %s" % epoch
    b = c["body"]
    ent = (b.get("retained") or {}).get(key)
    if not isinstance(ent, dict):
        return False, "retained[%s] absent in the epoch-%d commit" % (key, epoch)
    if ent.get("sha256") != file_sha:
        return False, "retained[%s].sha256 differs from the file sha256" % key
    if b.get("model_state_sha256") != state_sha:
        return False, "commit model_state_sha256 differs from the registered state"
    if key not in (b.get("roles_this_epoch") or []):
        return False, "%s not in roles_this_epoch %s" % (key, b.get("roles_this_epoch"))
    prev_ret = (by_epoch[epoch - 1]["body"].get("retained") or {}) if (epoch - 1) in by_epoch else {}
    if key.startswith("milestone_"):
        if key in prev_ret:
            return False, "milestone entry already existed in the previous commit"
    elif isinstance(prev_ret.get(key), dict) and prev_ret[key].get("sha256") == file_sha:
        return False, "the same rolling-label entry already existed in the previous commit"
    later = [e for e in by_epoch if e > epoch and key.startswith("milestone_")
             and (by_epoch[e]["body"].get("retained") or {}).get(key, {}).get("sha256") != file_sha]
    if later:
        return False, "retained[%s] changes in later commits %s" % (key, later[:3])
    return True, "commit seq %s, epoch %d, retained[%s] name=%r" % (c.get("seq"), epoch, key, ent.get("name"))


def load_ledger(R, arm, seed, identities):
    """Verified ledger -> dict with whitelisted fields only, or None (failures are logged as gates)."""
    tag = "(%s,%d)" % (arm, seed)
    p = os.path.join(R, "%d_%s" % (seed, arm), "ledger.jsonl")
    if not gate("C8", os.path.isfile(p), "%s ledger.jsonl present" % tag):
        return None
    with open(p, "rb") as fh:
        blob = fh.read()
    if not gate("C8", identities.get(os.path.abspath(p)) == sha_bytes(blob), "%s ledger sha256 == analysis inputs.identities" % tag):
        return None
    raw = [x for x in blob.split(b"\n") if x.strip()]
    full = [json.loads(x.decode("utf-8")) for x in raw]
    n = len(full)

    def canon(o):
        return json.dumps({k: v for k, v in o.items() if k != "self"}, sort_keys=True, separators=(",", ":")).encode()
    n_self = sum(1 for o in full if o.get("self") == sha_bytes(canon(o)))
    n_prev = sum(1 for i in range(1, n) if full[i].get("prev") == sha_bytes(raw[i - 1]))
    kinds = [o.get("kind") for o in full]
    ok = all([
        gate("C8", n >= 3 and all(o.get("schema") == LEDGER_SCHEMA for o in full), "%s %d records, schema %s" % (tag, n, LEDGER_SCHEMA)),
        gate("C8", [o.get("seq") for o in full] == list(range(n)), "%s seq contiguous from 0" % tag),
        gate("C8", n >= 3 and kinds[0] == "run_start" and kinds[-1] == "run_end", "%s run_start first, run_end last" % tag),
        gate("C8", n_self == n, "%s self hash verifies for %d/%d records" % (tag, n_self, n)),
        gate("C8", n >= 2 and n_prev == n - 1, "%s prev hash verifies for %d/%d links" % (tag, n_prev, n - 1)),
    ])
    by_epoch, sel = {}, None
    for o in full:                                           # whitelist: nothing else of a record is kept
        if o.get("kind") != "commit":
            continue
        b = o.get("body") or {}
        ret = {k: {"name": v.get("name"), "sha256": v.get("sha256")} for k, v in (b.get("retained") or {}).items() if isinstance(v, dict)}
        by_epoch[b.get("epoch")] = {"seq": o.get("seq"), "body": {"model_state_sha256": b.get("model_state_sha256"),
                                                                 "roles_this_epoch": list(b.get("roles_this_epoch") or []), "retained": ret}}
        sel = ((b.get("selection") or {}).get("best_psnr") or {}).get("epoch")      # epoch only; the value is never read
    del full
    epochs = sorted(e for e in by_epoch if isinstance(e, int))
    ok = gate("C8", bool(epochs) and epochs == list(range(1, len(epochs) + 1)) and len(epochs) == kinds.count("commit"),
              "%s one commit per epoch 1..%d" % (tag, len(epochs))) and ok
    if not ok:
        return None
    return {"by_epoch": by_epoch, "n": len(epochs), "sel": sel, "raw_sha": [sha_bytes(x) for x in raw], "kinds": kinds}


# ---------------------------------------------------------------- the checks
def checks(cfg, pre):
    tag = "F11A_SELFTEST_CHECKS" if cfg["selftest"] else "F11A_CHECKS"
    repo, R = cfg["repo"], os.path.join(cfg["repo"], "seqref_mri", "results", "v05lt")
    log.info("%s", VERSION)
    if cfg["selftest"]:
        log.warning("SELF-TEST MODE -- mock repository and synthetic records -- NOT FIXTURE-11A EVIDENCE")
    log.info("start_utc %s", utc())
    log.info("script %s sha256 %s", pre["script"], pre["script_sha"])
    log.info("HEAD %s (pin, branch %s, clean, nothing staged); diagnostic commit %s", pre["head"], cfg["branch"], cfg["diag_commit"])

    import h5py
    import numpy as np
    import torch
    log.info("python %s | numpy %s | torch %s | h5py %s | byteorder %s", sys.version.split()[0], np.__version__, torch.__version__,
             h5py.__version__, sys.byteorder)
    saved = tripwire_on([(h5py, "File"), (torch, "load"), (np, "load")])
    try:
        return _checks(cfg, pre, repo, R, np, tag)
    finally:
        tripwire_off(saved)


def _checks(cfg, pre, repo, R, np, tag):
    before = set(sys.modules)
    sys.path.insert(0, pre["code_dir"])
    sys.path.insert(0, repo)
    import importlib
    mods = {}
    for name in list(cfg["v05lt_modules"]) + list(cfg["diag_modules"]):
        mods[name] = importlib.import_module(name)
    fdm, rdec, pio = mods["seqref_mri.src.fastmri_data"], mods["seqref_mri.src.residual_decoder"], mods["seqref_mri.src.preflight_io"]
    V, C = mods["seqref_mri.scripts.v02_manifests"], mods["seqref_mri.scripts.v05lt_common"]
    core, jsh = mods["diag_unc_core"], mods["diag_jsh_core"]

    # ---- C1a / C1b source identity
    for name, (rel, _sha) in cfg["v05lt_modules"].items():
        f = os.path.realpath(mods[name].__file__)
        gate("C1a", f == os.path.realpath(os.path.join(repo, rel)) and sha_file(f) == pre["ident"][name],
             "%s loaded from the pinned tree, sha256 %s (== pin == registered)" % (name, pre["ident"][name][:16]))
    for name in cfg["diag_modules"]:
        f = os.path.realpath(mods[name].__file__)
        gate("C1b", f == os.path.join(pre["code_dir"], name + ".py") and sha_file(f) == pre["ident"][name],
             "%s loaded from the run directory export, sha256 %s (== archive %s == registered)" % (name, pre["ident"][name][:16], cfg["diag_commit"][:7]))
    gate("C1a", C.canonical_json is V.canonical_json, "v05lt_common.canonical_json is v02_manifests.canonical_json")

    # ---- MASK_CONVENTION from the pinned source text (no import)
    conv = None
    for node in ast.parse(pre["eval_src"].decode("utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MASK_CONVENTION" for t in node.targets):
            conv = ast.literal_eval(node.value)
    gate("C7b", isinstance(conv, str) and bool(conv), "MASK_CONVENTION read from the pinned v05lt_eval.py source text")

    # ---- registered inputs (whitelisted fields only)
    inp = os.path.join(R, "inputs")
    sp_path, el_path = os.path.join(inp, "v05lt_split_manifest.json"), os.path.join(inp, "v05lt_nmse_eligibility.json")
    doc, sp_sha = load_json(sp_path)
    gate("IN", sp_sha == cfg["split_sha"] == sidecar(sp_path), "split manifest sha256 == registered == sidecar")
    val_rows = [{"volume_id": r.get("volume_id"), "file": r.get("file"), "n_slices": r.get("n_slices")} for r in doc.get("validation", [])]
    del doc
    doc, el_sha = load_json(el_path)
    gate("IN", el_sha == sidecar(el_path), "eligibility manifest sha256 == sidecar")
    prov = doc.get("provenance", {})
    el_order_sha, el_split_sha = prov.get("validation_order_sha256"), prov.get("split_manifest_sha256")
    slices = [{"file": s.get("file"), "slice_index": s.get("slice_index"), "volume_id": s.get("volume_id"), "midpoint": s.get("midpoint")}
              for s in doc.get("slices", [])]
    del doc, prov
    gate("IN", el_split_sha == sp_sha, "eligibility provenance.split_manifest_sha256 == split manifest")
    an_path = os.path.join(R, "analysis", "v05lt_analysis.json")
    doc, an_sha = load_json(an_path)
    gate("IN", an_sha == cfg["analysis_sha"] == sidecar(an_path), "analysis document sha256 == registered (196, G-D4) == sidecar")
    identities = dict((doc.get("inputs") or {}).get("identities") or {})
    del doc                                                   # analysis outcomes dropped unread
    gate("IN", bool(identities) and all(isinstance(v, str) and HEX64.fullmatch(v) for v in identities.values()),
         "analysis inputs.identities is a path -> sha256 table (%d entries)" % len(identities))

    # ---- C7a population and masks
    mids = [s for s in slices if s["midpoint"] is True]
    gate("C7a", len(slices) == N_VAL_SLICES and all(isinstance(s["file"], str) and isinstance(s["slice_index"], int)
                                                     and not isinstance(s["slice_index"], bool) for s in slices),
         "eligibility manifest lists %d validation slices (expected %d)" % (len(slices), N_VAL_SLICES))
    by_vol = {r["volume_id"]: r for r in val_rows}
    gate("C7a", len(val_rows) == N_VOLUMES == len(by_vol), "split manifest lists %d validation volumes" % len(val_rows))
    gate("C7a", len(mids) == N_VOLUMES and sorted(s["volume_id"] for s in mids) == sorted(by_vol)
         and all(by_vol[s["volume_id"]]["file"] == s["file"] for s in mids if s["volume_id"] in by_vol),
         "%d midpoint slices, exactly one per validation volume, file names equal to the split manifest" % len(mids))
    gate("C7a", fdm.EVAL_BASE_SEED == EVAL_BASE_SEED, "fastmri_data.EVAL_BASE_SEED == %d" % EVAL_BASE_SEED)
    gate("C2", rdec.P3_FLATTEN_ORDER == FLATTEN_ORDER, "P3_FLATTEN_ORDER == %r" % FLATTEN_ORDER)

    # ---- band grids (C4 grid parity, C5 boundary)
    B = np.asarray(core.band_grid(HW))
    BJ = np.asarray(jsh.band_index(HW))
    k = np.arange(HW, dtype=np.int64) - HW // 2
    r2 = k[:, None] ** 2 + k[None, :] ** 2

    def grid_ok(g):
        g = np.asarray(g)
        return g.shape == (HW, HW) and g.dtype.kind in "iu" and bool(np.array_equal(g, BJ))

    def boundary_ok(g):
        g = np.asarray(g)
        return g.shape == (HW, HW) and bool(np.all(g[r2 < 512] == 0) and np.all(g[(r2 >= 512) & (r2 < 2048)] == 1) and np.all(g[r2 >= 2048] == 2))
    gate("C4", grid_ok(B), "step-B band_grid == DIAG-JSH band_index on the full %dx%d grid" % (HW, HW))
    gate("C5", int((r2 == 512).sum()) > 0 and int((r2 == 2048).sum()) > 0 and boundary_ok(B) and boundary_ok(BJ),
         "boundary radii: r2 == 512 -> mid (%d points), r2 == 2048 -> high (%d points)" % (int((r2 == 512).sum()), int((r2 == 2048).sum())))

    def pairs_ok(rows, cols, mask):
        ref = np.argwhere(~np.broadcast_to(np.asarray(mask, dtype=bool)[None, :], (HW, HW)))
        rows, cols = np.asarray(rows), np.asarray(cols)
        return rows.shape == cols.shape == (N_FREE,) and bool(np.array_equal(rows, ref[:, 0]) and np.array_equal(cols, ref[:, 1]))

    def bandvec_ok(vec, rows, cols):
        vec = np.asarray(vec)
        return vec.shape == (N_FREE,) and bool(np.array_equal(vec, BJ[np.asarray(rows), np.asarray(cols)]))

    # ---- per slice: C7a mask, C2 order, C4 band map
    masks, per_slice, n_bad = [], [], {"mask": 0, "order": 0, "band": 0}
    for s in mids:
        seed = fdm.canonical_mask_seed(fdm.EVAL_BASE_SEED, s["file"], s["slice_index"])
        m = fdm.make_cartesian_mask(HW, seed)
        m2 = fdm.make_cartesian_mask(HW, seed)
        acq = [int(c) for c in np.flatnonzero(m)]
        ok_m = (isinstance(m, np.ndarray) and m.dtype == np.bool_ and m.shape == (HW,) and len(acq) == N_ACQ
                and CENTRE.issubset(acq) and m.tobytes() == m2.tobytes())
        n_bad["mask"] += not ok_m
        cmap = rdec.build_coordinate_map(acq, HW, HW)
        rows, cols = np.asarray(cmap.free_rows), np.asarray(cmap.free_cols)
        ok_o = (pairs_ok(rows, cols, m) and cmap.n_free_complex == N_FREE and len(set(cols.tolist())) == HW - N_ACQ
                and not (set(cols.tolist()) & CENTRE) and cmap.flatten_order == FLATTEN_ORDER)
        n_bad["order"] += not ok_o
        band = B[rows, cols]
        counts = core.check_band_map(band, N_FREE)
        ok_b = bandvec_ok(band, rows, cols) and int(counts.sum()) == N_FREE
        n_bad["band"] += not ok_b
        masks.append(m)
        per_slice.append((s, seed, acq, [int(x) for x in counts], cmap.payload()["map_payload_sha256"],
                          pio.canonical_hash({"width": HW, "selected_columns": acq})))
    gate("C7a", n_bad["mask"] == 0, "97 masks: bool (96,), 24 acquired incl. 44-51, regenerated byte-identical (%d bad)" % n_bad["mask"])
    gate("C2", n_bad["order"] == 0, "97 coordinate maps: 6912 pairs == argwhere(~mask) row-major, 72 columns, none of 44-51 (%d bad)" % n_bad["order"])
    gate("C4", n_bad["band"] == 0, "97 per-slice band maps == DIAG-JSH grid at (free_rows, free_cols); counts sum to 6912 (%d bad)" % n_bad["band"])
    for (s, seed, acq, counts, map_sha, m_sha) in per_slice:     # A2 + per-slice records (reported, not gates)
        info("A2", "%s|%d seed=%d bands low/mid/high=%s map=%s mask=%s" % (s["file"], s["slice_index"], seed, counts, map_sha[:16], m_sha[:16]))

    # ---- C6 mutations: each must be detected by the checker named
    if mids and n_bad["order"] == 0:
        m0 = masks[0]
        ref = np.argwhere(~np.broadcast_to(m0[None, :], (HW, HW)))
        rows0, cols0 = ref[:, 0], ref[:, 1]
        vec0 = BJ[rows0, cols0]
        kk = np.arange(HW, dtype=np.int64)

        def grid_from(kv, strict=True):
            q = kv[:, None] ** 2 + kv[None, :] ** 2
            return ((q >= 512).astype(np.int64) + (q >= 2048).astype(np.int64)) if strict else ((q > 512).astype(np.int64) + (q > 2048).astype(np.int64))
        cm = np.lexsort((rows0, cols0))                       # column-major order of the same pairs
        i, j = 0, int(np.flatnonzero(vec0 != vec0[0])[0])
        sw = vec0.copy()
        sw[i], sw[j] = vec0[j], vec0[i]
        muts = [
            ("transposed coordinates (rows<->cols)", "pairs", pairs_ok(cols0, rows0, m0)),
            ("grid shifted by one (k - 47)", "grid", grid_ok(grid_from(kk - (HW // 2 - 1)))),
            ("uncentred grid (k = 0..95)", "grid", grid_ok(grid_from(kk))),
            ("column-major packing", "pairs", pairs_ok(rows0[cm], cols0[cm], m0)),
            ("non-strict boundary (<=)", "boundary", boundary_ok(grid_from(k, strict=False))),
            ("two packed indices of different bands swapped", "bandvec", bandvec_ok(sw, rows0, cols0)),
        ]
        for name, checker, passed in muts:
            gate("C6", not passed, "mutation detected by the %s check: %s" % (checker, name))
        info("C6", "the band grid is symmetric under transposition, so a rows<->cols swap cannot change band values; it is caught by the C2 pair check")
        gate("C6", pairs_ok(rows0, cols0, m0) and grid_ok(grid_from(k)) and boundary_ok(grid_from(k)) and bandvec_ok(vec0, rows0, cols0),
             "the same checkers accept the unmutated objects")
    else:
        gate("C6", False, "mutation tests not run (no valid coordinate map)")

    # ---- post-evaluation records (whitelisted fields only)
    recs, pe_root = [], os.path.join(R, "posteval")
    for d in sorted(os.listdir(pe_root)) if os.path.isdir(pe_root) else []:
        p = os.path.join(pe_root, d, "posteval.json")
        if not os.path.isfile(p):
            continue
        j, fsha = load_json(p)
        ck, pop, mk = j.get("checkpoint") or {}, j.get("population") or {}, j.get("masks") or {}
        recs.append({"dir": d, "file_sha": fsha, "registered": identities.get(os.path.abspath(p)), "schema": j.get("schema"),
                     "ck": {x: ck.get(x) for x in ("arm", "seed", "epoch", "file_name", "file_sha256", "model_state_sha256")} | {"roles": list(ck.get("roles") or [])},
                     "uses": [(u.get("arm"), u.get("seed"), u.get("role")) for u in (j.get("uses") or [])],
                     "masks": {"convention": mk.get("convention"), "sha256": mk.get("sha256")},
                     "pop": {x: pop.get(x) for x in ("scope", "n", "identities_sha256", "eligibility_manifest_sha256")},
                     "primary": (j.get("banks") or {}).get("primary"), "fixture": (j.get("context") or {}).get("fixture")})
        del j
    n = len(recs)

    def allrec(cid, label, fn):
        good = sum(1 for r in recs if fn(r))
        gate(cid, n > 0 and good == n, "%s: %d/%d post-evaluation records" % (label, good, n))
    gate("C8", n == EXPECT["posteval_records"], "post-evaluation records on disk: %d (expected %d)" % (n, EXPECT["posteval_records"]))
    allrec("C8", "file sha256 == analysis inputs.identities", lambda r: r["registered"] is not None and r["registered"] == r["file_sha"])
    allrec("C8", "schema, directory == checkpoint.model_state_sha256, context.fixture false",
           lambda r: r["schema"] == POSTEVAL_SCHEMA and r["dir"] == r["ck"]["model_state_sha256"] and r["fixture"] is False)
    allrec("C8", "banks.primary == registered Z_DIAG hash", lambda r: r["primary"] == Z_DIAG)
    allrec("C7b", "population scope midpoint, n == 97, eligibility_manifest_sha256 == eligibility file",
           lambda r: r["pop"]["scope"] == "midpoint" and r["pop"]["n"] == N_VOLUMES and r["pop"]["eligibility_manifest_sha256"] == el_sha)

    # ---- C7b order and aggregate mask hash
    lines_sha = sha_bytes("".join("%s|%d\n" % (s["file"], s["slice_index"]) for s in slices).encode("utf-8"))
    order_all = gate("C7b", isinstance(el_order_sha, str) and lines_sha == el_order_sha,
                     "validation order verified: sha256 over %d '<file>|<slice>' lines == provenance.validation_order_sha256" % len(slices))
    ids = [[str(s["file"]), int(s["slice_index"])] for s in mids]
    ids_sha = C.sha256_bytes(C.canonical_json(ids))
    order_mid = gate("C7b", n > 0 and all(r["pop"]["identities_sha256"] == ids_sha for r in recs),
                     "midpoint order verified: sha256(canonical_json(identities)) == population.identities_sha256 in %d/%d records"
                     % (sum(1 for r in recs if r["pop"]["identities_sha256"] == ids_sha), n))
    order_ok = order_all and order_mid
    if order_ok and n_bad["mask"] == 0:
        agg = hashlib.sha256()
        for m in masks:
            agg.update(np.ascontiguousarray(m[None, :]).tobytes(order="C"))
        agg = agg.hexdigest()
        gate("C7b", all(r["masks"]["sha256"] == agg for r in recs),
             "aggregate mask hash %s == masks.sha256 in %d/%d records" % (agg[:16], sum(1 for r in recs if r["masks"]["sha256"] == agg), n))
    else:
        gate("C7b", False, "aggregate mask hash NOT interpreted: slice order or masks not verified; no other order is tried")
    allrec("C7b", "masks.convention == MASK_CONVENTION of the pinned source", lambda r: conv is not None and r["masks"]["convention"] == conv)

    # ---- C8 role bindings
    role_state, role_file = {}, {}
    ledgers = {}
    for arm, seed in RUNS:
        tagr = "(%s,%d)" % (arm, seed)
        rd = os.path.join(R, "%d_%s" % (seed, arm))
        le = ledgers[(arm, seed)] = load_ledger(R, arm, seed, identities)
        if le is None:
            continue
        be = le["by_epoch"]
        for e, role in ((3, "epoch3"), (10, "epoch10"), (30, "epoch30")):
            fsha = sidecar(os.path.join(rd, "ckpt", "weights_e%03d.pt" % e))
            st = (be.get(e) or {}).get("body", {}).get("model_state_sha256")
            ok, why = bound(be, e, "milestone_%03d" % e, fsha, st) if fsha and st else (False, "sidecar or commit state missing")
            ok = ok and ("name=%r" % ("weights_e%03d.pt" % e)) in why
            if gate("C8", ok, "%s %s: sidecar sha256 bound to the epoch-%d state by the ledger rule (%s)" % (tagr, role, e, why)):
                role_state[(arm, seed, role)], role_file[(arm, seed, role)] = st, fsha
        sel = le["sel"]
        if gate("C8", sel == EXPECT_SELECTED[(arm, seed)], "%s selected epoch %s == %d (223)" % (tagr, sel, EXPECT_SELECTED[(arm, seed)])):
            name = "weights_e%03d.pt" % sel
            fsha = sidecar(os.path.join(rd, "ckpt", name))
            last = be[le["n"]]["body"]["retained"].get("best_psnr") or {}
            st = be[sel]["body"]["model_state_sha256"]
            if gate("C8", fsha is not None and last.get("sha256") == fsha and last.get("name") == name and isinstance(st, str),
                    "%s selected: %s sidecar sha256 == last commit retained[best_psnr]" % (tagr, name)):
                role_state[(arm, seed, "selected")], role_file[(arm, seed, "selected")] = st, fsha
        fwp = os.path.join(rd, "final", "final_weights.json")
        if gate("C8", os.path.isfile(fwp), "%s final_weights.json present" % tagr):
            j, fw_sha = load_json(fwp)
            src = j.get("source") or {}
            fw = {"run": (j.get("run") or {}).get("arm"), "seed": (j.get("run") or {}).get("seed"), "epoch": j.get("epoch"),
                  "state": j.get("model_state_sha256"), "wfile": j.get("weights_file"), "wsha": j.get("weights_sha256"),
                  "cseq": src.get("commit_seq"), "csha": src.get("commit_line_sha256"), "eseq": src.get("run_end_seq"), "esha": src.get("run_end_line_sha256")}
            del j
            nl = len(le["raw_sha"])

            def line_ok(seq, sha, kind):
                return isinstance(seq, int) and not isinstance(seq, bool) and 0 <= seq < nl and le["raw_sha"][seq] == sha and le["kinds"][seq] == kind
            cstate = next((c["body"]["model_state_sha256"] for e, c in be.items() if c["seq"] == fw["cseq"]), None)
            cepoch = next((e for e, c in be.items() if c["seq"] == fw["cseq"]), None)
            ok = (fw_sha == sidecar(fwp) and (fw["run"], fw["seed"]) == (arm, seed) and fw["wfile"] == "weights_final.pt"
                  and fw["wsha"] is not None and fw["wsha"] == sidecar(os.path.join(rd, "final", "weights_final.pt"))
                  and line_ok(fw["cseq"], fw["csha"], "commit") and line_ok(fw["eseq"], fw["esha"], "run_end")
                  and cstate is not None and fw["state"] == cstate and fw["epoch"] == cepoch == le["n"])
            if gate("C8", ok, "%s final: final_weights.json sidecar, weights sidecar, ledger line hashes and last-commit state all agree" % tagr):
                role_state[(arm, seed, "final")], role_file[(arm, seed, "final")] = fw["state"], fw["wsha"]

    # rule test on the bindings registered by the verified records (as dunc_audit2.py v0.1.2)
    tested = failed = 0
    for r in recs:
        ck = r["ck"]
        le = ledgers.get((ck["arm"], ck["seed"]))
        if le is None or r["registered"] != r["file_sha"] or not re.fullmatch(r"weights_e\d{3}\.pt", str(ck["file_name"])):
            continue
        keys = [x for x in ck["roles"] if x in (le["by_epoch"].get(ck["epoch"], {}).get("body", {}).get("roles_this_epoch") or [])]
        if not keys:
            failed += 1
            log.error("[C8] rule test (%s,%s,epoch %s): none of the checkpoint labels is in roles_this_epoch", ck["arm"], ck["seed"], ck["epoch"])
        for key in keys:
            tested += 1
            ok, why = bound(le["by_epoch"], ck["epoch"], key, ck["file_sha256"], ck["model_state_sha256"])
            if not ok:
                failed += 1
                log.error("[C8] rule test (%s,%s,epoch %s,%s): %s", ck["arm"], ck["seed"], ck["epoch"], key, why)
    gate("C8", failed == 0 and tested == EXPECT["rule_bindings"],
         "ledger rule reproduces %d/%d bindings registered by verified records (expected %d of %d)" % (tested - failed, tested, EXPECT["rule_bindings"], EXPECT["rule_bindings"]))

    # Tier 1 roles <-> records
    by_use = {}
    for r in recs:
        for u in r["uses"]:
            by_use.setdefault(u, []).append(r)
    t1 = [(a, s, role) for a, s in RUNS for role in ROLES_T1]
    bad = [u for u in t1 if len(by_use.get(u, [])) != 1 or by_use[u][0]["ck"]["model_state_sha256"] != role_state.get(u)
           or (by_use[u][0]["ck"]["arm"], by_use[u][0]["ck"]["seed"]) != u[:2]]
    gate("C8", not bad, "each Tier 1 role is used by exactly one record whose checkpoint state equals the role's state; not satisfied: %s" % (bad or "none"))
    gate("C8", set(by_use) == set(t1) and all(r["uses"] for r in recs), "the records' uses are exactly the 18 Tier 1 roles and every record has a use")
    same_file = [u for u in t1 if len(by_use.get(u, [])) == 1 and by_use[u][0]["ck"]["file_sha256"] == role_file.get(u)]
    info("C8", "%d of 18 Tier 1 records evaluated the role's own file; the others evaluated another file holding the same state (224 U9)" % len(same_file))

    # counts, printed first, then compared
    occ = [(a, s, role) for a, s in RUNS for role in ROLES]
    states = {u: role_state.get(u) for u in occ}
    got = {"role_occurrences": sum(1 for v in states.values() if v), "distinct_states": len({v for v in states.values() if v}),
           "tier1_role_occurrences": sum(1 for u in t1 if states.get(u)), "tier1_distinct_states": len({states[u] for u in t1 if states.get(u)}),
           "posteval_records": n, "rule_bindings": tested - failed}
    groups = {}
    for u, v in states.items():
        if v:
            groups.setdefault(v, set()).add(u)
    coinc = {frozenset(g) for g in groups.values() if len(g) > 1}
    info("C8", "counts: %s" % json.dumps(got, sort_keys=True))
    info("C8", "coincident roles: %s" % sorted(sorted(g) for g in coinc))
    for key in ("role_occurrences", "distinct_states", "tier1_role_occurrences", "tier1_distinct_states", "posteval_records"):
        gate("C8", got[key] == EXPECT[key], "%s = %d (expected %d)" % (key, got[key], EXPECT[key]))
    gate("C8", coinc == EXPECT_COINCIDENCES,
         "exactly the two named coincidences: lam2/2028 selected == final, lam0/2027 selected == epoch10")

    # ---- A4 (reported only)
    agree = sum(1 for s in mids if s["volume_id"] in by_vol and isinstance(by_vol[s["volume_id"]]["n_slices"], int)
                and V.midpoint_slice(by_vol[s["volume_id"]]["n_slices"]) == s["slice_index"])
    info("A4", "flagged midpoint equals v02_manifests.midpoint_slice(n_slices) for %d of %d volumes (not a gate)" % (agree, len(mids)))

    # ---- import closure, tripwire, Git state
    allowed = set(cfg["v05lt_modules"]) | set(cfg["diag_modules"])
    new = sorted(m for m in set(sys.modules) - before if m.split(".")[0] in ("seqref_mri", "diag_unc_core", "diag_jsh_core")
                 or m.startswith(("diag_unc_", "diag_jsh_")))
    extra = [m for m in new if m not in allowed and m not in PACKAGE_PARENTS]
    info("C1", "project modules imported: %s" % new)
    gate("C1a", not extra and allowed <= set(new), "imported project modules are exactly the approved list (package entries %s allowed); extra: %s"
         % (list(PACKAGE_PARENTS), extra or "none"))
    gate("TRIP", not TRIP, "tripwire: h5py.File, torch.load and numpy.load were never called (%d calls)" % len(TRIP))
    head2, dirty = git(repo, "rev-parse", "HEAD").strip(), git(repo, "status", "--porcelain").strip()
    gate("C1a", head2 == cfg["pin"] and not dirty, "post-check: HEAD is the pin, status clean")
    log.info("end_utc %s", utc())
    ok = not FAILS
    if ok:
        log.info("%s: PASS (final result is decided after evidence verification; see RESULT.txt)", tag)
    else:
        log.error("%s: FAIL (%d failed gates: %s)", tag, len(FAILS), sorted(set(FAILS)))
    return ok


# ---------------------------------------------------------------- evidence handling and main
def fault(cfg, name):
    if cfg.get("fault") == name:
        raise OSError("injected fault: %s (self-test only)" % name)


def run(cfg):
    pre = prechecks(cfg)
    tag = "F11A_SELFTEST_RESULT" if cfg["selftest"] else "F11A_RESULT"
    run_dir = pre["run_dir"]
    print("F11A_OUT=%s" % run_dir)
    log_path, problems = os.path.join(run_dir, "fixture11a.log"), []
    side_path, result_path = log_path + ".sha256", os.path.join(run_dir, "RESULT.txt")
    fh = None
    try:
        fault(cfg, "log_open")
        fh = logging.FileHandler(log_path, mode="x", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        log.addHandler(fh)
    except Exception as exc:
        log.error("cannot create the log file %s: %r", log_path, exc)
        problems.append("log file not created")
    checks_ok = False
    if fh is not None:
        try:
            checks_ok = checks(cfg, pre)
        except Exception as exc:
            log.error("unexpected error: %r", exc)
            log.error("%s: FAIL", "F11A_SELFTEST_CHECKS" if cfg["selftest"] else "F11A_CHECKS")
        finally:
            log.removeHandler(fh)
            fh.close()
        if not checks_ok:
            problems.append("checks failed (see fixture11a.log)")
    log_sha = None
    if fh is not None:
        try:
            log_sha = sha_file(log_path)
            fault(cfg, "sidecar")
            with open(side_path, "x", encoding="utf-8") as sf:
                sf.write("%s  fixture11a.log\n" % log_sha)
                sf.flush()
                os.fsync(sf.fileno())
            with open(side_path, "r", encoding="utf-8") as sf:
                parts = sf.read().split()
            if parts != [log_sha, "fixture11a.log"] or sha_file(log_path) != log_sha:
                raise OSError("sidecar does not verify against the log")
        except Exception as exc:
            log.error("log sidecar failure: %r", exc)
            problems.append("log sidecar not written or not verified")
    final = "PASS" if not problems else "FAIL (nothing is deleted; do not rerun)"
    text = "%s: %s\n%s\nutc %s\nlog_sha256 %s\n" % (tag, final, VERSION, utc(), log_sha) + "".join("problem: %s\n" % p for p in problems)
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
    ap.add_argument("--selftest", metavar="CONFIG.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")
    try:
        cfg = dict(selftest=False, repo=REPO, pin=PIN, branch="seqref-v05lt", diag_commit=DIAG_COMMIT, v05lt_modules=dict(V05LT_MODULES),
                   diag_modules=dict(DIAG_MODULES), eval_sha=EVAL_SRC[1], split_sha=SPLIT_SHA, analysis_sha=ANALYSIS_SHA, fault=None)
        if a.selftest:
            with open(a.selftest, "r", encoding="utf-8") as fh:
                o = json.load(fh)
            allowed = {"repo", "pin", "branch", "diag_commit", "diag_jsh_sha", "eval_sha", "split_sha", "analysis_sha", "fault"}
            if set(o) - allowed or not {"repo", "pin", "diag_commit"} <= set(o):
                raise Abort("self-test config: keys must include repo, pin, diag_commit and be within %s" % sorted(allowed))
            if os.path.realpath(o["repo"]) == os.path.realpath(REPO) or o["pin"] == PIN:
                raise Abort("self-test must not use the production repository or pin")
            if o.get("fault") not in (None, "log_open", "sidecar", "result"):
                raise Abort("self-test config: unknown fault")
            cfg.update(selftest=True, repo=os.path.abspath(o["repo"]), pin=o["pin"], diag_commit=o["diag_commit"], fault=o.get("fault"))
            for key in ("branch", "eval_sha", "split_sha", "analysis_sha"):
                if key in o:
                    cfg[key] = o[key]
            if "diag_jsh_sha" in o:
                cfg["diag_modules"]["diag_jsh_core"] = (DIAG_MODULES["diag_jsh_core"][0], o["diag_jsh_sha"])
        rc = run(cfg)
    except Abort as exc:
        log.error("F11A_ABORT: %s", exc)
        log.error("F11A_RESULT: ABORTED (nothing is deleted; do not rerun)")
        rc = 1
    except Exception as exc:
        log.error("unexpected error outside the run: %r", exc)
        log.error("F11A_RESULT: FAIL (nothing is deleted; do not rerun)")
        rc = 1
    print("F11A_EXIT=%d" % rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
