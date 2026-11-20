# SEQREF-DLATB1 v0.3.3 -- dlat_b1_v03.py: D-LAT B1 pass 2, read-only host check (LudWiki 236)
# LIFETIME: DIAGNOSTIC
"""Read-only host inspection for D-LAT blocker B1 (article 235 §9, record 236).

The registered D-UNC F2 manifest is the authoritative checkpoint inventory: 30 roles, 29 files,
28 distinct model states. Nothing is found by file-name search and then trusted.

Checks (standard library only):
  G  git identity; sha256 of source files at the pin
  M  F2 and G manifests: file sha256, schema, pin, and that G was run from exactly this F2;
     the RESULT.txt conditions named in manifest_g.validity are observed, not re-validated
  K  every role: path inside the runs root, regular non-symlink file, FULL file sha256 equal to
     the manifest, arm/seed/epoch consistent with the path
  T  state mapping: the 30 roles resolve to the registered 28 model-state identities
  L  V05LT ledgers read as text: registered file sha256, canonical lines, self-hash, sequence
     and previous-record chain; each role file against the WHOLE ledger history (every commit
     record, not only the last one), so a checkpoint superseded by a later commit is not a failure
  R  stored validation rows: file sha256 against both manifests; per-slice B and J key names
  C  D-UNC code of the G run against the F2 manifest, on disk and at the code commit
  X  extra checkpoint files not in the manifest (listed only)
  S  array-type files under the D-UNC evidence folder (names and sizes only)
  H  host resources

Never: import torch, numpy, h5py or a project module; deserialise a checkpoint (bytes are read
for hashing only); open an MRI file; call Ledger.recover, ArtifactStore.recover or the
ArtifactStore constructor; write anywhere except its own new output folder.

Limits, repeated in the output:
  * model-state sha256 is NOT recomputed (needs the checkpoint loaded): the check shows that the
    recorded identities agree with the manifest and that the files are byte-identical to the
    registered ones; checkpoint usability is NOT tested;
  * the G run is NOT re-validated;
  * manifest fields final_weights and posteval are recorded, not interpreted.
No fallback, placeholder, mock or dummy return: every miss is a FAIL or UNRESOLVED item.

Exit status: 0 = no FAIL and no UNRESOLVED; 1 = at least one FAIL or UNRESOLVED; 2 = internal error.

Changelog:
  v0.3.3 (2026-11-20) review correction (reporting only): a sha256 that merely occurs in the ledger
       text, outside the retained sets of the commit records, is INFO, not PASS: a bare
       occurrence does not show that the checkpoint was registered. (This replaces the
       "PASS with that wording" of the v0.3.2 line below.)
  v0.3.2 (2026-11-20) review correction: L3 reads the whole ledger history. A role file named with
       its sha256 by any commit record is a PASS (with a note whether the last commit retains
       it); a different sha256 for that name anywhere in the history is a FAIL; a sha256 found
       only outside the retained sets is a PASS with that wording; a ckpt/ file the ledger never
       mentions is UNRESOLVED (final files: INFO). File identity itself rests on K2.
  v0.3.1 (2026-11-20) bug fix after review: forbidden modules already present when the interpreter
       starts (Z0) are reported apart from those loaded during the check (Z1); both stay FAIL.
       An unavailable GPU is INFO (not needed for a read-only check).
  v0.3 (2026-11-20) F2 manifest is the checkpoint inventory (exact 30-role / 28-state check,
       final checkpoints included); ledger chain verified with the canonical JSON rule of
       v02_manifests; F2-G relationship and RESULT.txt conditions; stored rows by manifest hash;
       path containment and symlink refusal. Replaces the schema-free matching of v0.2.
  v0.2 (2026-11-20) first script version (not run on the host).
"""
import argparse
import datetime
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys

__version__ = "0.3.3"
LOG = logging.getLogger("dlat_b1")

PIN = "7455da8027f33a6cca4dbf72f04e72420e938f45"
F2_SCHEMA = "seqref-dunc-f2-manifest/1"
G_SCHEMA = "seqref-dunc-g-manifest/1"
LEDGER_SCHEMA = "seqref-v05lt-ledger/1"
GENESIS = "GENESIS"
EXPECT_F2_SHA = "d1956bf370dc6c1cc82caf9804dc9731cbec970c0f8205265c14bdf72e3a6215"  # reviewed copy, 236
EXPECT_G_SHA = "3b21b4ed27e6c4f6f7a0ef349a98ea298eb5eb1a2cb885b9e56393c0e4102348"   # reviewed copy, 236
N_ROLES, N_FILES, N_STATES = 30, 29, 28
G_RUN = "g_20261119T102151Z"
F2_COPIES = ("gbind_20261119T093846Z/code/manifest_f2.json", "staging/f2/manifest_f2.json")
ROW_KEYS = (b'"nll_prior_per_dim"', b'"nll_logdet_per_dim"')
ARRAY_EXT = (".npy", ".npz", ".pt", ".h5", ".bin")
FORBIDDEN_MODULES = ("torch", "numpy", "h5py", "seqref_mri")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CHUNK = 1 << 20
# forbidden modules already loaded when this file starts to run (before the check does anything)
PRELOADED = sorted(m for m in sys.modules if m.split(".")[0] in FORBIDDEN_MODULES)

# path at the pin -> (expected sha256 or None, origin of the expectation)
PIN_FILES = {
    "seqref_mri/tdiag/d3.py": ("73110e7d36a1ebdcb51926c4bbde81f88392f578e1aa860736c509706041245e", "registered 224 U3"),
    "seqref_mri/src/free_flow_runtime.py": ("8e171b2c521d77c8ad22b21d02537be160d15166aecb9e0319fd0a25b69823c9", "registered 224 U3"),
    "seqref_mri/scripts/v05lt_eval.py": ("10b4bc65d465e9c6d0a016c2d27977cd9f485ffb851936d9a15ad6a325898554", "registered 224 U3"),
    "seqref_mri/scripts/v05lt_metrics.py": ("87eaa5a4e85b6b70072c1c76fcd3e73ce104c3cdb04ebef973c346ff90b18a74", "registered 224 U3"),
    "seqref_mri/src/base_experts.py": ("c3d1adcafd57c71b2e176f4aa29b72e0b809539ef5006eae4f4162bbe406b952", "reviewed copy 236 LE7"),
    "seqref_mri/src/fastmri_data.py": ("610cc1d1d7968deebc88f645270e1baefb6589cb56841bdd327450ca1069cb44", "reviewed copy 236 LE7"),
    "seqref_mri/src/residual_decoder.py": ("fd65e3a229e3df4f0226e276c92c916f5bd32cc77ed3655d68fb8a14608990d6", "reviewed copy 236 LE7"),
    "seqref_mri/scripts/tiny_gate.py": ("3ddd73ac99612e4af8d69664539acaef2ec4e9621deae530327fe0a8cd1e88d8", "reviewed copy 236 LE7"),
    "seqref_mri/scripts/train_base.py": ("2ae948663dbd53df32e032c0ef8c54e23666106529ca1906a354fcee7ba6ffc5", "reviewed copy 236 LE7"),
    "seqref_mri/scripts/v05lt_common.py": ("f35664fcb92607d0b0919ea80e55a90c99bbea9d6584b329c0d7688d9e3712a1", "reviewed copy 236 LE7"),
    "seqref_mri/tdiag/estimators.py": ("05561e729c7b5a7e3f6914990af2808918e372bec1ed0710c97ef3de867dfc28", "reviewed copy 236 LE7"),
    "seqref_mri/src/flows/nsf_layer.py": ("37c1f0960dda0415f998503ed80fb4bc7ae2f69b87ab2812ee8cd3731524f7f2", "reviewed copy"),
    "seqref_mri/scripts/v02_manifests.py": ("5ef4c55121df0e5fa151c920bcafb75c55121c222afb7f0d73b19cdbc44749e9", "reviewed copy"),
    "seqref_mri/src/conditioner.py": (None, "not reviewed; hash recorded only"),
    "seqref_mri/src/forward_operator.py": (None, "not reviewed; hash recorded only"),
    "seqref_mri/scripts/train_free_flow.py": (None, "not reviewed; hash recorded only"),
}

ITEMS = []


class B1Error(Exception):
    """Internal failure of the check itself. Always logged."""

    def __init__(self, message):
        super().__init__(message)
        LOG.error("B1_ERROR: %s", message)


def item(check, status, text, **detail):
    if status not in ("PASS", "FAIL", "UNRESOLVED", "INFO"):
        raise B1Error("unknown status %r" % (status,))
    ITEMS.append({"check": check, "status": status, "text": text, "detail": detail})
    (LOG.error if status == "FAIL" else LOG.warning if status == "UNRESOLVED" else LOG.info)(
        "[%s] %s: %s", status, check, text)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_path(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:  # bytes only; nothing is deserialised
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj):
    """The rule of seqref_mri/scripts/v02_manifests.canonical_json, restated (no project import)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def safe_file(root, rel):
    """root/rel as a regular file reached through no symlink and lying inside root; else ValueError."""
    if not isinstance(rel, str) or not rel or os.path.isabs(rel) or "\\" in rel:
        raise ValueError("not a plain relative path: %r" % (rel,))
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError("path has an empty, '.' or '..' component: %r" % (rel,))
    cur = root
    for part in parts:
        cur = os.path.join(cur, part)
        try:
            st = os.lstat(cur)
        except OSError as exc:
            raise ValueError("missing: %s (%s)" % (cur, exc.strerror))
        if stat.S_ISLNK(st.st_mode):
            raise ValueError("symlink refused: %s" % cur)
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("not a regular file: %s" % cur)
    real_root = os.path.realpath(root)
    if os.path.commonpath([real_root, os.path.realpath(cur)]) != real_root:
        raise ValueError("resolves outside the root: %s" % cur)
    return cur


def walk(root):
    def on_error(exc):
        item("walk", "FAIL", "cannot list %s: %s" % (getattr(exc, "filename", "?"), exc))
    for d, _dirs, files in os.walk(root, followlinks=False, onerror=on_error):
        for name in sorted(files):
            p = os.path.join(d, name)
            if not os.path.islink(p):
                yield p


def git(repo, *args):
    try:
        r = subprocess.run(["git", "-C", repo] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise B1Error("git cannot be run: %s" % exc)
    return r.returncode, r.stdout, " / ".join(r.stderr.decode("utf-8", "replace").split("\n")[:2]).strip()


def load_json(path, check):
    """-> (document, sha256 of the file bytes) or (None, None) with a FAIL item."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        return json.loads(data), sha256_bytes(data)
    except (OSError, ValueError) as exc:
        item(check, "FAIL", "%s: unreadable or not JSON (%s)" % (path, exc))
        return None, None


# ---------------------------------------------------------------- G: git and sources
def check_git(repo):
    rc, out, err = git(repo, "rev-parse", "HEAD")
    if rc != 0:
        item("G1 head", "FAIL", "not a git repository or no HEAD: %s" % err)
        return
    head = out.decode().strip()
    rc, out, _ = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    item("G1 head", "INFO", "HEAD %s on %s" % (head, out.decode().strip() if rc == 0 else "?"), head=head)
    rc, out, _ = git(repo, "status", "--porcelain")
    n = len([ln for ln in out.decode("utf-8", "replace").splitlines() if ln.strip()]) if rc == 0 else None
    item("G2 worktree", "INFO", "git status --porcelain lines: %s (recorded, not judged)" % n, lines=n)
    rc, _, err = git(repo, "cat-file", "-e", PIN + "^{commit}")
    if rc != 0:
        item("G3 pin", "FAIL", "pin commit %s not found: %s" % (PIN, err))
        return
    item("G3 pin", "PASS", "pin commit present: %s" % PIN)
    for path, (expect, origin) in sorted(PIN_FILES.items()):
        rc, out, err = git(repo, "show", "%s:%s" % (PIN, path))
        if rc != 0:
            item("G4 source@pin", "FAIL", "%s: not readable at the pin (%s)" % (path, err), path=path)
            continue
        got = sha256_bytes(out)
        if expect is None:
            item("G4 source@pin", "INFO", "%s  %s  (%s)" % (got, path, origin), path=path, sha256=got)
        elif got == expect:
            item("G4 source@pin", "PASS", "%s  %s  (= %s)" % (got, path, origin), path=path, sha256=got)
        else:
            item("G4 source@pin", "FAIL", "%s: sha256 %s != expected %s (%s)" % (path, got, expect, origin),
                 path=path, sha256=got, expected=expect)


# ---------------------------------------------------------------- M: manifests
def check_manifests(f2_path, g_path, dunc_root):
    """-> (f2 document or None, g document or None)."""
    f2, f2_sha = load_json(f2_path, "M1 F2 manifest")
    g, g_sha = load_json(g_path, "M2 G manifest")
    if f2 is not None:
        ok = f2_sha == EXPECT_F2_SHA
        item("M1 F2 manifest", "PASS" if ok else "FAIL", "file sha256 %s (%s the reviewed copy)" % (f2_sha, "=" if ok else "!="),
             path=f2_path, sha256=f2_sha)
        ok = f2.get("schema") == F2_SCHEMA and f2.get("pin") == PIN and f2.get("selftest") is False
        item("M1 F2 manifest", "PASS" if ok else "FAIL", "schema %r, pin %r, selftest %r"
             % (f2.get("schema"), f2.get("pin"), f2.get("selftest")))
        for rel in F2_COPIES:
            p = os.path.join(dunc_root, rel)
            if os.path.isfile(p) and not os.path.islink(p):
                same = sha256_path(p) == f2_sha
                item("M1 F2 manifest", "PASS" if same else "FAIL", "copy %s is %s" % (rel, "identical" if same else "DIFFERENT"))
            else:
                item("M1 F2 manifest", "INFO", "copy %s not present (not required)" % rel)
    if g is not None:
        ok = g_sha == EXPECT_G_SHA
        item("M2 G manifest", "PASS" if ok else "FAIL", "file sha256 %s (%s the reviewed copy)" % (g_sha, "=" if ok else "!="),
             path=g_path, sha256=g_sha)
        ok = (g.get("schema") == G_SCHEMA and g.get("pin") == PIN and g.get("mode") == "g" and g.get("selftest") is False)
        item("M2 G manifest", "PASS" if ok else "FAIL", "schema %r, pin %r, mode %r, selftest %r; recorded result %r"
             % (g.get("schema"), g.get("pin"), g.get("mode"), g.get("selftest"), g.get("result")))
    if f2 is not None and g is not None:
        iv = g.get("inputs_verified", {})
        rel = {"G names this F2 file sha256": g.get("manifest_f2_sha256") == f2_sha,
               "same code commit": g.get("code_commit") == f2.get("code_commit"),
               "same code hashes": g.get("code") == f2.get("code"),
               "same roles": iv.get("roles") == f2.get("roles"),
               "same ledger hashes": iv.get("ledgers") == f2.get("inputs", {}).get("ledgers"),
               "F2 stored rows are among the G history rows":
                   all(iv.get("history_rows", {}).get(k) == v
                       for k, v in f2.get("inputs", {}).get("stored_rows", {}).items())}
        bad = [k for k, v in rel.items() if not v]
        item("M3 F2-G relationship", "PASS" if not bad else "FAIL",
             "all %d relations hold" % len(rel) if not bad else "relations that fail: %s" % bad)
        # conditions named in manifest_g.validity: observed here, the G run is not re-validated
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(g_path)))
        res = os.path.join(run_dir, "RESULT.txt")
        if not os.path.isfile(res) or os.path.islink(res):
            item("M4 G result file", "UNRESOLVED", "no RESULT.txt in %s" % run_dir)
        else:
            with open(res, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            first = text.split("\n", 1)[0].strip()
            ok = first == "G_RESULT: PASS" and g_sha in text
            item("M4 G result file", "PASS" if ok else "FAIL",
                 "first line %r; names the manifest_g sha256: %s (observed; G is not re-validated here)" % (first, g_sha in text))
        stops = [n for n in ("stop_record_watchdog.json", "RESULT_INVALID.txt") if os.path.lexists(os.path.join(run_dir, n))]
        item("M4 G result file", "PASS" if not stops else "FAIL", "invalidation records in the run folder: %s" % (stops or "none"))
    return f2, g


# ---------------------------------------------------------------- runs root
def resolve_runs_root(explicit, search_root, run_names):
    """The folder that holds the six run folders. Explicit, or the single candidate; never a choice."""
    def has_all(d):
        return all(os.path.isdir(os.path.join(d, n)) and not os.path.islink(os.path.join(d, n)) for n in run_names)
    if explicit:
        ok = os.path.isdir(explicit) and has_all(explicit)
        item("K0 runs root", "PASS" if ok else "FAIL", "--runs-root %s %s the %d run folders"
             % (explicit, "holds" if ok else "does not hold", len(run_names)))
        return os.path.realpath(explicit) if ok else None
    if not os.path.isdir(search_root):
        item("K0 runs root", "FAIL", "search root missing: %s" % search_root)
        return None
    cands = [d for d, _s, _f in os.walk(search_root, followlinks=False) if has_all(d)]
    if len(cands) == 1:
        item("K0 runs root", "PASS", "single folder holding the %d run folders: %s" % (len(run_names), cands[0]))
        return os.path.realpath(cands[0])
    item("K0 runs root", "UNRESOLVED", "%d candidate folders under %s; pass --runs-root explicitly: %s"
         % (len(cands), search_root, cands[:10]))
    return None


# ---------------------------------------------------------------- K: role files
def role_fields_ok(r):
    """Shape and path/field consistency of one role record -> list of problems."""
    probs = []
    for k in ("arm", "seed", "epoch", "role", "tier", "path", "file_sha256", "model_state_sha256"):
        if k not in r:
            probs.append("field %s missing" % k)
    if probs:
        return probs
    for k in ("file_sha256", "model_state_sha256"):
        if not isinstance(r[k], str) or not HEX64.match(r[k]):
            probs.append("%s is not 64 lower-case hex" % k)
    parts = str(r["path"]).split("/")
    if len(parts) != 3 or parts[0] != "%s_%s" % (r["seed"], r["arm"]):
        probs.append("path %r does not start with <seed>_<arm>/" % (r["path"],))
    elif r["role"] == "final":
        if parts[1:] != ["final", "weights_final.pt"]:
            probs.append("final role with path %r" % (r["path"],))
    elif parts[1:] != ["ckpt", "weights_e%03d.pt" % int(r["epoch"])]:
        probs.append("path %r does not match ckpt/weights_e<epoch>.pt for epoch %r" % (r["path"], r["epoch"]))
    return probs


def check_roles(f2, runs_root):
    """Full sha256 of every role file -> {relative path: sha256 of the bytes on disk}."""
    roles = f2.get("roles")
    if not isinstance(roles, list) or not all(isinstance(r, dict) for r in roles):
        item("K1 roles", "FAIL", "manifest has no list of role records")
        return {}
    bad_shape = 0
    for i, r in enumerate(roles):
        probs = role_fields_ok(r)
        if probs:
            bad_shape += 1
            item("K1 roles", "FAIL", "role %d (%s): %s" % (i, r.get("path"), "; ".join(probs)))
    paths = sorted({r["path"] for r in roles if "path" in r})
    ok = len(roles) == N_ROLES and len(paths) == N_FILES and bad_shape == 0
    item("K1 roles", "PASS" if ok else "FAIL", "%d role records (registered %d), %d distinct files (registered %d), %d malformed"
         % (len(roles), N_ROLES, len(paths), N_FILES, bad_shape))
    expect = {}
    for r in roles:
        if "path" in r and "file_sha256" in r:
            if expect.setdefault(r["path"], r["file_sha256"]) != r["file_sha256"]:
                item("K1 roles", "FAIL", "%s: two different file sha256 in the manifest" % r["path"])
    if runs_root is None:
        item("K2 file sha256", "UNRESOLVED", "runs root not resolved; no role file was hashed")
        return {}
    got, good = {}, 0
    for i, rel in enumerate(paths, 1):
        try:
            p = safe_file(runs_root, rel)
            size = os.path.getsize(p)
            h = sha256_path(p)
        except (ValueError, OSError) as exc:
            item("K2 file sha256", "FAIL", "%s: %s" % (rel, exc), path=rel)
            continue
        got[rel] = h
        if h == expect[rel]:
            good += 1
            item("K2 file sha256", "PASS", "[%d/%d] %s  %s  (%d bytes; = manifest)" % (i, len(paths), h, rel, size),
                 path=rel, sha256=h, bytes=size)
        else:
            item("K2 file sha256", "FAIL", "%s: file sha256 %s != manifest %s" % (rel, h, expect[rel]), path=rel, sha256=h)
        side = p + ".sha256"
        if os.path.isfile(side) and not os.path.islink(side):
            with open(side, "r", encoding="ascii", errors="replace") as fh:
                parts = fh.read().split()
            same = len(parts) == 2 and parts[0] == h and parts[1] == os.path.basename(p)
            item("K3 sidecar", "PASS" if same else "FAIL", "%s.sha256 %s the file" % (rel, "agrees with" if same else "DISAGREES with"))
        else:
            item("K3 sidecar", "INFO", "%s: no sidecar (not required by the manifest)" % rel)
    item("K4 summary", "PASS" if good == len(paths) == N_FILES else "FAIL",
         "checkpoint FILE sha256: %d of %d role files are byte-identical to the registered ones" % (good, len(paths)))
    return got


# ---------------------------------------------------------------- T: state mapping
def check_states(f2, file_hashes):
    roles, states = f2.get("roles", []), f2.get("states", [])
    by_state, by_file, probs = {}, {}, []
    for r in roles:
        try:
            label = "%s|%s|%s" % (r["arm"], r["seed"], r["role"])
            by_state.setdefault(r["model_state_sha256"], []).append(label)
            if by_file.setdefault(r["file_sha256"], r["model_state_sha256"]) != r["model_state_sha256"]:
                probs.append("file %s is given two model states" % r["file_sha256"])
        except (KeyError, TypeError):
            probs.append("a role record lacks arm, seed, role or a hash")
    labels = [x for v in by_state.values() for x in v]
    if len(set(labels)) != len(labels):
        probs.append("a role label arm|seed|role occurs twice")
    if len(by_state) != N_STATES:
        probs.append("%d distinct model states in the roles, registered %d" % (len(by_state), N_STATES))
    try:
        listed = {s["model_state_sha256"]: sorted(s["roles"]) for s in states}
        if [s["index"] for s in states] != list(range(len(states))) or len(listed) != len(states):
            probs.append("state indices are not 0..n-1 or a state hash repeats")
    except (KeyError, TypeError):
        listed = None
        probs.append("state list is malformed")
    if listed is not None and listed != {k: sorted(v) for k, v in by_state.items()}:
        probs.append("the state list does not equal the roles grouped by model state")
    if f2.get("decode_counts", {}).get("states") != N_STATES:
        probs.append("decode_counts.states is %r" % (f2.get("decode_counts", {}).get("states"),))
    item("T1 state mapping", "PASS" if not probs else "FAIL",
         "%d roles resolve to %d registered model-state identities" % (len(labels), len(by_state)) if not probs
         else "; ".join(probs))
    n_ok = sum(1 for r in roles if file_hashes.get(r.get("path")) == r.get("file_sha256"))
    states_ok = {r["model_state_sha256"] for r in roles
                 if file_hashes.get(r.get("path")) == r.get("file_sha256") and "model_state_sha256" in r}
    full = not probs and n_ok == len(roles) == N_ROLES and len(states_ok) == N_STATES
    item("T2 states on disk", "PASS" if full else ("UNRESOLVED" if not file_hashes else "FAIL"),
         "%d of %d roles and %d of %d model states are backed by a file with the registered sha256"
         % (n_ok, len(roles), len(states_ok), N_STATES))
    item("T3 model-state sha256", "INFO", "NOT recomputed by this check (needs the checkpoint loaded); "
         "identity rests on byte-identical files plus the registered mapping")
    item("T4 checkpoint usability", "INFO", "NOT tested by this check")


# ---------------------------------------------------------------- L: ledgers (text only)
def verify_ledger_bytes(data):
    """The complete-line rules of v05lt_common.Ledger._scan, read-only -> (records, problems)."""
    probs = []
    end = data.rfind(b"\n") + 1
    if data[end:]:
        probs.append("%d bytes after the last newline (torn tail)" % len(data[end:]))
    recs, prev = [], GENESIS
    for n, line in enumerate(data[:end].split(b"\n")[:-1]):
        try:
            rec = json.loads(line)
        except ValueError:
            probs.append("line %d is not JSON" % n)
            break
        if not isinstance(rec, dict):
            probs.append("line %d is not an object" % n)
            break
        if canonical_json(rec) != line:
            probs.append("line %d is not canonical" % n)
        body = {k: v for k, v in rec.items() if k != "self"}
        if rec.get("self") != sha256_bytes(canonical_json(body)):
            probs.append("line %d self-hash mismatch (record altered)" % n)
        if rec.get("schema") != LEDGER_SCHEMA or rec.get("seq") != n or rec.get("prev") != prev:
            probs.append("line %d breaks the chain (seq %r, prev %r)" % (n, rec.get("seq"), rec.get("prev")))
        recs.append(rec)
        prev = sha256_bytes(line)
    if not recs and not probs:
        probs.append("no record")
    return recs, probs


def check_ledgers(f2, runs_root, file_hashes):
    reg = f2.get("inputs", {}).get("ledgers")
    if not isinstance(reg, dict) or not reg:
        item("L1 ledger", "FAIL", "manifest has no registered ledger hashes")
        return
    if runs_root is None:
        item("L1 ledger", "UNRESOLVED", "runs root not resolved; no ledger was read")
        return
    for key, want in sorted(reg.items()):
        arm, _, seed = key.partition("|")
        run = "%s_%s" % (seed, arm)
        run_dir = os.path.join(runs_root, run)
        match = []
        if os.path.isdir(run_dir):
            for p in walk(run_dir):
                if p.endswith(".jsonl") and os.path.getsize(p) < (1 << 30) and sha256_path(p) == want:
                    match.append(p)
        if len(match) != 1:
            item("L1 ledger", "FAIL", "%s: %d .jsonl files with the registered sha256 %s" % (run, len(match), want))
            continue
        with open(match[0], "rb") as fh:
            data = fh.read()
        rel = os.path.relpath(match[0], runs_root)
        item("L1 ledger", "PASS", "%s: sha256 = registered %s" % (rel, want), path=rel)
        recs, probs = verify_ledger_bytes(data)
        item("L2 ledger chain", "PASS" if not probs else "FAIL",
             "%s: %d records, canonical lines, self-hashes and chain hold" % (rel, len(recs)) if not probs
             else "%s: %s" % (rel, "; ".join(probs[:5])))
        commits = [r for r in recs if r.get("kind") == "commit"]
        history = {}                                   # name -> [(seq, sha256)] over EVERY commit record
        for r in commits:
            for v in (r.get("body", {}).get("retained", {}) or {}).values():
                if isinstance(v, dict):
                    history.setdefault(v.get("name"), []).append((r.get("seq"), v.get("sha256")))
        last = ({v.get("name") for v in commits[-1].get("body", {}).get("retained", {}).values() if isinstance(v, dict)}
                if commits else set())
        item("L3 ledger history", "INFO", "%s: %d commit records; the last one retains %d files: %s"
             % (rel, len(commits), len(last), sorted(str(n) for n in last)))
        text = data.decode("utf-8", "replace")
        for path, h in sorted(file_hashes.items()):
            if not path.startswith(run + "/"):
                continue
            base = os.path.basename(path)
            same = [q for q, x in history.get(base, []) if x == h]
            other = [(q, x) for q, x in history.get(base, []) if x != h]
            if other:
                item("L3 ledger history", "FAIL", "%s: the ledger names this file with another sha256 (seq, sha256) %s; "
                     "file on disk %s" % (path, other[:3], h))
            elif same:
                item("L3 ledger history", "PASS", "%s: named with the sha256 of the file on disk by commit record(s) seq %s; "
                     "retained by the last commit: %s" % (path, same if len(same) <= 6 else [same[0], "...", same[-1]],
                                                         "yes" if base in last else "no (superseded later; not a failure)"))
            elif h in text:
                item("L3 ledger history", "INFO", "%s: its sha256 occurs in the ledger text, but in no retained set of a "
                     "commit record; this does not show that the file was registered there "
                     "(file identity rests on the registered manifest hash, K2)" % path)
            else:
                item("L3 ledger history", "INFO" if "/final/" in path else "UNRESOLVED",
                     "%s: neither its name with this sha256 nor the sha256 itself occurs in %s "
                     "(file identity rests on the registered manifest hash, K2)" % (path, rel))


# ---------------------------------------------------------------- R: stored validation rows
def count_keys(path):
    counts, carry = [0] * len(ROW_KEYS), b""
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            buf = carry + chunk
            for i, key in enumerate(ROW_KEYS):
                counts[i] += buf.count(key)
            carry = buf[-32:]
            for i, key in enumerate(ROW_KEYS):  # a key inside the overlap is counted with the next chunk
                counts[i] -= carry.count(key)
    for i, key in enumerate(ROW_KEYS):
        counts[i] += carry.count(key)
    return counts


def check_rows(f2, g, runs_root):
    stored = f2.get("inputs", {}).get("stored_rows", {})
    history = (g or {}).get("inputs_verified", {}).get("history_rows", {})
    table = dict(history)
    table.update(stored)
    if not stored:
        item("R1 stored rows", "FAIL", "manifest has no stored-row hashes")
        return
    if runs_root is None:
        item("R1 stored rows", "UNRESOLVED", "runs root not resolved; no row file was read")
        return
    good = bad = 0
    for key, want in sorted(table.items()):
        try:
            arm, seed, ep = key.split("|")
            p = safe_file(runs_root, "%s_%s/ckpt/val_%s.json" % (seed, arm, ep))  # F2 history_rows rule
            h = sha256_path(p)
        except (ValueError, OSError) as exc:
            bad += 1
            item("R1 stored rows", "FAIL", "%s: %s" % (key, exc))
            continue
        if h != want:
            bad += 1
            item("R1 stored rows", "FAIL", "%s: sha256 %s != registered %s" % (key, h, want))
            continue
        good += 1
        if key in stored:
            c = count_keys(p)
            item("R2 B and J keys", "PASS" if min(c) > 0 else "UNRESOLVED",
                 "%s: %d x nll_prior_per_dim, %d x nll_logdet_per_dim (key names counted; no value read)" % (key, c[0], c[1]))
    item("R1 stored rows", "PASS" if bad == 0 else "FAIL",
         "%d of %d registered row files have the registered sha256 (%d at the 28 states' epochs, %d in the G history)"
         % (good, len(table), len(stored), len(history)))


# ---------------------------------------------------------------- C: D-UNC code
def check_code(repo, f2, g_path):
    code, commit = f2.get("code"), f2.get("code_commit")
    if not isinstance(code, dict) or not code:
        item("C1 D-UNC code", "FAIL", "manifest has no code hashes")
        return
    code_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(g_path))), "code")
    for name, want in sorted(code.items()):
        try:
            h = sha256_path(safe_file(code_dir, name))
            item("C1 D-UNC code", "PASS" if h == want else "FAIL", "run copy %s: sha256 %s the manifest" % (name, "=" if h == want else "!="))
        except (ValueError, OSError) as exc:
            item("C1 D-UNC code", "FAIL", "run copy %s: %s" % (name, exc))
    rc, out, err = git(repo, "ls-tree", "-r", "--name-only", str(commit))
    if rc != 0:
        item("C2 D-UNC code@commit", "FAIL", "code commit %s not readable: %s" % (commit, err))
        return
    tree = out.decode("utf-8", "replace").splitlines()
    for name, want in sorted(code.items()):
        hits = [t for t in tree if t.split("/")[-1] == name]
        if len(hits) != 1:
            item("C2 D-UNC code@commit", "UNRESOLVED", "%s: %d files of that name at %s; none chosen" % (name, len(hits), commit))
            continue
        rc, blob, err = git(repo, "show", "%s:%s" % (commit, hits[0]))
        ok = rc == 0 and sha256_bytes(blob) == want
        item("C2 D-UNC code@commit", "PASS" if ok else "FAIL", "%s at %s: sha256 %s the manifest" % (hits[0], str(commit)[:12], "=" if ok else "!="))


# ---------------------------------------------------------------- X, S, H
def check_extra(f2, runs_root):
    if runs_root is None:
        return
    known = {r.get("path") for r in f2.get("roles", [])}
    extra = sorted(os.path.relpath(p, runs_root) for p in walk(runs_root)
                   if p.endswith(".pt") and os.path.relpath(p, runs_root) not in known)
    item("X1 extra checkpoints", "INFO", "%d .pt files under the runs root are not in the manifest "
         "(not hashed; not part of the D-LAT state population)" % len(extra), files=extra[:200])


def check_retention(dunc_root):
    if not os.path.isdir(dunc_root):
        item("S1 retention", "FAIL", "D-UNC root missing: %s" % dunc_root)
        return
    rows = [(os.path.relpath(p, dunc_root), os.path.getsize(p)) for p in walk(dunc_root) if p.endswith(ARRAY_EXT)]
    for rel, size in rows:
        item("S1 retention", "INFO", "%s  %d bytes (not opened)" % (rel, size))
    item("S2 summary", "INFO", "%d array-type files, %d bytes in total, under %s" % (len(rows), sum(s for _, s in rows), dunc_root))


def check_resources(repo, out_dir):
    for label, p in (("repository", repo), ("output", out_dir)):
        du = shutil.disk_usage(p)
        item("H1 disk", "INFO", "%s filesystem: %d bytes free of %d" % (label, du.free, du.total), free=du.free)
    try:
        with open("/proc/meminfo", "r") as fh:
            item("H2 memory", "INFO", "; ".join(ln.strip() for ln in fh if ln.startswith(("MemTotal", "MemAvailable"))))
    except OSError as exc:
        item("H2 memory", "UNRESOLVED", "/proc/meminfo unreadable: %s" % exc)
    item("H3 cpu", "INFO", "os.cpu_count() = %s" % os.cpu_count())
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if r.returncode == 0:
            item("H4 gpu", "INFO", r.stdout.decode("utf-8", "replace").strip().replace("\n", " | "))
        else:
            item("H4 gpu", "INFO", "no GPU reported: nvidia-smi exit %d: %s" % (r.returncode, r.stderr.decode("utf-8", "replace").strip()))
    except (OSError, subprocess.TimeoutExpired) as exc:
        item("H4 gpu", "INFO", "no GPU reported: nvidia-smi not usable (%s)" % exc)


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="SEQREF-DLATB1 v0.3: D-LAT B1 pass 2, read-only host check")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--runs-root", default=None, help="folder holding the six run folders; default: the single such "
                                                      "folder under <repo>/seqref_mri/results/v05lt, else UNRESOLVED")
    ap.add_argument("--dunc-root", default=None, help="default: <repo>/v06lt_evidence/DIAG_UNC")
    ap.add_argument("--f2", default=None, help="default: <dunc-root>/%s/code/manifest_f2.json" % G_RUN)
    ap.add_argument("--g", default=None, help="default: <dunc-root>/%s/out/manifest_g.json" % G_RUN)
    ap.add_argument("--out-root", default=None, help="default: <repo>/seqref_mri/results/_diag/dlat_b1")
    a = ap.parse_args(argv)
    del ITEMS[:]
    repo = os.path.realpath(a.repo)
    search_root = os.path.join(repo, "seqref_mri", "results", "v05lt")
    dunc_root = os.path.realpath(a.dunc_root or os.path.join(repo, "v06lt_evidence", "DIAG_UNC"))
    f2_path = a.f2 or os.path.join(dunc_root, G_RUN, "code", "manifest_f2.json")
    g_path = a.g or os.path.join(dunc_root, G_RUN, "out", "manifest_g.json")
    out_root = os.path.realpath(a.out_root or os.path.join(repo, "seqref_mri", "results", "_diag", "dlat_b1"))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = os.path.join(out_root, "dlat_b1_v03_" + stamp)
    for scanned in (os.path.realpath(search_root), dunc_root, os.path.realpath(a.runs_root) if a.runs_root else None):
        if scanned and os.path.commonpath([scanned, out_dir]) == scanned:
            print("refusing: output folder %s lies inside a scanned folder %s" % (out_dir, scanned), file=sys.stderr)
            return 2
    try:
        os.makedirs(out_dir, exist_ok=False)  # never overwrites an earlier report
    except OSError as exc:
        print("cannot create the output folder %s: %s" % (out_dir, exc), file=sys.stderr)
        return 2
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(out_dir, "report.log"), mode="x", encoding="utf-8")])
    LOG.info("SEQREF-DLATB1 v%s | read-only | no project import, no checkpoint load, no MRI file, no inference", __version__)
    LOG.info("repo=%s runs_root=%s dunc_root=%s f2=%s g=%s out=%s", repo, a.runs_root, dunc_root, f2_path, g_path, out_dir)
    code = 2
    try:
        check_git(repo)
        f2, g = check_manifests(f2_path, g_path, dunc_root)
        if f2 is None:
            item("K0 runs root", "UNRESOLVED", "no F2 manifest: the checkpoint, state, ledger, row and code checks did not run")
        else:
            run_names = sorted({str(r.get("path", "")).split("/")[0] for r in f2.get("roles", []) if isinstance(r, dict)})
            runs_root = resolve_runs_root(a.runs_root, search_root, run_names) if run_names and all(run_names) else None
            hashes = check_roles(f2, runs_root)
            check_states(f2, hashes)
            check_ledgers(f2, runs_root, hashes)
            check_rows(f2, g, runs_root)
            check_code(repo, f2, g_path)
            check_extra(f2, runs_root)
        check_retention(dunc_root)
        check_resources(repo, out_dir)
        item("Z0 startup modules", "PASS" if not PRELOADED else "FAIL",
             "forbidden modules already present at interpreter start: %s%s"
             % (sorted({m.split(".")[0] for m in PRELOADED}) or "none",
                " (start Python with -I -S)" if PRELOADED else ""))
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in FORBIDDEN_MODULES and m not in PRELOADED)
        item("Z1 imports", "PASS" if not loaded else "FAIL", "forbidden modules loaded during the check: %s" % (loaded or "none"))
        counts = {s: sum(1 for it in ITEMS if it["status"] == s) for s in ("PASS", "FAIL", "UNRESOLVED", "INFO")}
        code = 0 if counts["FAIL"] == 0 and counts["UNRESOLVED"] == 0 else 1
        LOG.info("RESULT %s | exit %d", counts, code)
        summary = {"script": "SEQREF-DLATB1 v%s" % __version__, "lifetime": "DIAGNOSTIC", "utc": stamp, "pin": PIN,
                   "repo": repo, "dunc_root": dunc_root, "f2": f2_path, "g": g_path,
                   "limits": ["model-state sha256 not recomputed", "checkpoint usability not tested",
                              "G run not re-validated", "final_weights and posteval fields not interpreted"],
                   "counts": counts, "exit": code, "items": ITEMS}
        with open(os.path.join(out_dir, "summary.json"), "x", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1, sort_keys=True)
    except Exception as exc:  # the check itself failed: logged, exit 2, no summary claimed
        LOG.exception("B1_INTERNAL_ERROR: %s: %s", type(exc).__name__, exc)
        code = 2
    logging.shutdown()
    for h in list(logging.getLogger().handlers):  # closed handlers must not be reused
        logging.getLogger().removeHandler(h)
    try:
        with open(os.path.join(out_dir, "SHA256SUMS"), "x", encoding="ascii") as fh:
            for name in ("report.log", "summary.json"):
                p = os.path.join(out_dir, name)
                if os.path.isfile(p):
                    fh.write("%s  %s\n" % (sha256_path(p), name))
    except OSError as exc:
        print("B1_ERROR: cannot write SHA256SUMS: %s" % exc, file=sys.stderr)
        code = 2
    print("output folder: %s | exit %d" % (out_dir, code))
    return code


if __name__ == "__main__":
    sys.exit(main())
