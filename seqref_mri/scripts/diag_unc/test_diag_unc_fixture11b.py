# SEQREF-DUNC-F11B v0.1 -- test_diag_unc_fixture11b.py: DIAG-UNC fixture 11b (production-tensor and adapter binding; plan SEQREF-DUNC-IMPL v0.10.1)
# LIFETIME: KEEP
"""Fixture 11b of DIAG-UNC (223 §10 item 11): binds diag_unc_adapter.py to the pinned V05LT
production tensors on exactly two registered validation midpoint slices and one checkpoint.
Criteria C3 and C9-C12 of IMPL v0.8 §4 (carried unchanged by IMPL v0.10.1). No metric is computed.

Layout expected by the run card:  <run_dir>/code/{this file, diag_unc_adapter.py, diag_unc_core.py}
Evidence written to <run_dir>:    fixture11b.log, fixture11b.log.sha256, RESULT.txt

Production mode : three mandatory arguments bound by the run card, nothing else can be changed:
                    --archive-commit      ch2-diag commit holding this file and the adapter
                    --ckpt-file-sha256    registered sha256 of the checkpoint file
                    --ckpt-state-sha256   registered model-state sha256 of the checkpoint
Self-test mode  : --selftest CONFIG.json (mock repository, synthetic HDF5 files, stand-in model).
                  Output is marked NOT FIXTURE-11B EVIDENCE. Test hook only, not a fallback.

Access (instrumented, not mocked; the real functions are always called):
  h5py.File          counted; allowed only read-only inside the registered training directory.
                     Expected: 2 per training-directory file (two production indexes, metadata
                     only) + 4 (each probe slice is read twice: once inside
                     v05lt_eval.build_states, once for the fixture's independent reference).
  h5py.Dataset read  allowed only for '/kspace'[slice] of the two probe identities; expected 4.
  torch.load         allowed once, on the registered checkpoint path.
  numpy.load         forbidden.
The production split 'val' (V02 holdout) is never requested; a path containing 'singlecoil_val'
is refused. Holdout volumes are known only through the registered holdout-ID JSON manifest.

Exposure: logged are hashes, identities, shapes, dtypes, counts, booleans and maximum-magnitude
norms. No pixel, k-space, coefficient or weight value, no PSNR/NLL/CRPS/coverage/loss. JSON files
are parsed, whitelisted fields are copied and the rest is dropped. Arrays stay in memory.

Result rule: the log holds the CHECKS verdict only; the final F11B_RESULT is decided after the log
is closed, its sidecar written and re-verified and RESULT.txt written and read back.
Exit codes: 0 PASS, 1 FAIL or ABORTED. Every failure is logged. No fallback, placeholder or dummy.
Reported-only items (never gates): C10a, C10c precondition when false, C10d status, timing/memory.
A zero or non-finite denominator, scale or norm is a FAIL (INVALID); no epsilon is added.

bound() and load_ledger() are taken from test_diag_unc_fixture11a.py v0.1.1 (bound() unchanged from
dunc_audit2.py v0.1.2; load_ledger() with the gate label changed only).

Changelog:
  v0.1 (2026-11-18) first version, from plan IMPL v0.10.1 and the import closure of 225 U44.
"""
import argparse
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

VERSION = "SEQREF-DUNC-F11B v0.1"
PIN = "7455da8027f33a6cca4dbf72f04e72420e938f45"
CORE_COMMIT = "d69b946f59abb014bf7b6cd21c842e54aa278a1f"
REPO = "/home/benjamin/CSMFII"
INTERPRETER = REPO + "/seqref_mri/.venv_mri/bin/python"
DIAG_DIR = "seqref_mri/scripts/diag_unc/"
CORE_SHA = "a0d3b4a4079ff290a93098eb7a1c53173df596080a6b2b51469dc0f757c18b06"
ADAPTER_SHA = "c1715841bf6805568b88f9195acbe27933a7e1fec9acc8bb22a33ff555a8505e"

EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
CLOSURE = {   # 225 §7 (U44): path in the pinned tree -> registered sha256 (41 modules)
    "seqref_mri/__init__.py": EMPTY,
    "seqref_mri/scripts/__init__.py": EMPTY,
    "seqref_mri/src/__init__.py": EMPTY,
    "seqref_mri/src/flows/__init__.py": EMPTY,
    "seqref_mri/scripts/tiny_gate.py": "3ddd73ac99612e4af8d69664539acaef2ec4e9621deae530327fe0a8cd1e88d8",
    "seqref_mri/scripts/train_base.py": "2ae948663dbd53df32e032c0ef8c54e23666106529ca1906a354fcee7ba6ffc5",
    "seqref_mri/scripts/train_free_flow.py": "32ec331ff79f3daf951f584b972889db909d313b893e9eaaa5c3697f1e165e3e",
    "seqref_mri/scripts/v02_eval.py": "6971d7ad82b36b77df266a49ef9c8d0a4737c197f529095c3e06f995d5156aa5",
    "seqref_mri/scripts/v02_manifests.py": "5ef4c55121df0e5fa151c920bcafb75c55121c222afb7f0d73b19cdbc44749e9",
    "seqref_mri/scripts/v02_readout_common.py": "c8fe6674a3bdb811c29c1dcdbc785e947995ded943754bfa500aa3467bece0d8",
    "seqref_mri/scripts/v02_readout_diag.py": "d19a02847cb3a22c9f9fa9153dfe695e79527b17583ba4f7149748b92f646db2",
    "seqref_mri/scripts/v02_train.py": "ff65e0a7de4c61e3cce610592130de4cb8a9119f94fa2920dd218f863b22c822",
    "seqref_mri/scripts/v05lt_common.py": "f35664fcb92607d0b0919ea80e55a90c99bbea9d6584b329c0d7688d9e3712a1",
    "seqref_mri/scripts/v05lt_eligibility.py": "1449d74e48b56af4b4687413849cb57b6f4f6d21917da1b91c0e03568249d6d1",
    "seqref_mri/scripts/v05lt_eval.py": "10b4bc65d465e9c6d0a016c2d27977cd9f485ffb851936d9a15ad6a325898554",
    "seqref_mri/scripts/v05lt_metrics.py": "87eaa5a4e85b6b70072c1c76fcd3e73ce104c3cdb04ebef973c346ff90b18a74",
    "seqref_mri/src/base_experts.py": "c3d1adcafd57c71b2e176f4aa29b72e0b809539ef5006eae4f4162bbe406b952",
    "seqref_mri/src/conditioner.py": "7dc7176125140613dfdc71457ac8bfb1c407bd86ed99f0176e0826bbda9669cc",
    "seqref_mri/src/contract_hash.py": "bf8379cb08a78fea419ee42d75990032fe17750f661943a440a1853cf1797b19",
    "seqref_mri/src/fastmri_data.py": "610cc1d1d7968deebc88f645270e1baefb6589cb56841bdd327450ca1069cb44",
    "seqref_mri/src/flows/nice_layer.py": "0222f0d1614b673b8d37dbcba5534a8b1cf1d7e0a384e30e7a518556afef723b",
    "seqref_mri/src/flows/nsf_layer.py": "37c1f0960dda0415f998503ed80fb4bc7ae2f69b87ab2812ee8cd3731524f7f2",
    "seqref_mri/src/flows/realnvp_image_layer.py": "8c570a271afb6977532d869195e024568bd4b72d5d00a8f76e65e1229377f19d",
    "seqref_mri/src/flows/realnvp_layer.py": "d07f977d392e1eddf50fd698b4547daa7fbb27536083708a41e72d9ac5e08c7a",
    "seqref_mri/src/forward_operator.py": "3cbc5162da9b727e82457a7c8f93121097c8228701f884a2190c681e4e1d4b5d",
    "seqref_mri/src/free_flow_runtime.py": "8e171b2c521d77c8ad22b21d02537be160d15166aecb9e0319fd0a25b69823c9",
    "seqref_mri/src/metrics.py": "051581fd82ebbe41e5adace1286a7bf4db920589dea5be629dd9d5219ef0de15",
    "seqref_mri/src/normalisation_profile.py": "e85c262d3c64c1c253692c5e104d2ec9028c35b1cdf38af4d9ed6a37d97adfe7",
    "seqref_mri/src/parent_expectations.py": "f0332ffbc2deb78f45f12a82987fcd243018cbe8d611450cf7546863fb95ea2d",
    "seqref_mri/src/preflight_io.py": "ab01fefbf0c46ce08b8746e97250f420de3f219bc8a25a49e514bc4af3b19942",
    "seqref_mri/src/preflight_parents.py": "61700c6f43ecd3bd7a05ff02a3eb642719f6bc2ea754134bf6d930575dfb5d50",
    "seqref_mri/src/residual_decoder.py": "fd65e3a229e3df4f0226e276c92c916f5bd32cc77ed3655d68fb8a14608990d6",
    "seqref_mri/src/train_utils.py": "eb98bcc3037411dd40d19bf1153a6094178a96a85549b777a1030cce3de3a625",
    "seqref_mri/tdiag/__init__.py": "acbc5f1996f3468bfbe08f4d90498d7900301495dcaf1c451e7cb6af55abf9ec",
    "seqref_mri/tdiag/_bootstrap.py": "3aab8f9e279838661a6165d52f1339cdfd9cf1de885571d5512be706e89b8e65",
    "seqref_mri/tdiag/d2a.py": "7ea5648a9ea725a71bd07a6f9d3393c29b79e3dbd9db8ae4f3a0ff0bac111eb1",
    "seqref_mri/tdiag/d2b.py": "ceed40052aa1359fae7178f2543fdcca03bb04827f6f674d228cd1ed66ebc23d",
    "seqref_mri/tdiag/d3.py": "73110e7d36a1ebdcb51926c4bbde81f88392f578e1aa860736c509706041245e",
    "seqref_mri/tdiag/estimators.py": "05561e729c7b5a7e3f6914990af2808918e372bec1ed0710c97ef3de867dfc28",
    "seqref_mri/tdiag/invariants.py": "ff6894fa3b0ae08c0617ead62457f7b33d012ad1c991e39f4b28aeb63f328d79",
    "seqref_mri/tdiag/replay.py": "374f8d352702aa7b4589285fad45c765c01fdfc5988e9ebae5bef1a03c02dc8e",
}
BARE_ALIASES = ("contract_hash", "normalisation_profile", "parent_expectations", "preflight_io",
                "preflight_parents", "residual_decoder")     # files of seqref_mri/src also loaded under a bare name
REQUIRED_SEAMS = ("seqref_mri.scripts.v05lt_common", "seqref_mri.scripts.v05lt_eval", "seqref_mri.scripts.tiny_gate",
                  "seqref_mri.scripts.train_base", "seqref_mri.tdiag.d3", "seqref_mri.tdiag.estimators",
                  "seqref_mri.src.free_flow_runtime", "seqref_mri.src.fastmri_data", "seqref_mri.src.residual_decoder",
                  "seqref_mri.src.preflight_io", "residual_decoder")

SPLIT_SHA = "f7d21fb97e030a2e8227233cd5d1f171a120b765060ae83cafc728721428cf5d"
ANALYSIS_SHA = "5b267d3aa7ccb48bf3d7ccbfa2e2497d7f02df845aff8786c0fbe9b9dff2afa2"
P4_SHA = "db0bf2c2e0cd3bd5e1ed86bf208198199f16df5b2fbbe0cc72903c8791f85684"
P0S_SHA = "2070cc72c977096f731811513d81b2abdb5dababb69edb4e997a5f8283b7c693"
S_REF = 15.62704352715211
Z_DIAG = "f8a6a34b6b474286359e4c99341de256b62bc0dff1e6bc8cf52e72f0461dc8d6"
Z_MANIFEST = "c4d2c0ec7c908f7c20a4f287eeffbdf966924a5118b6ba734b0a329942ab7e5f"
A2_LOG_REL = "v06lt_evidence/DIAG_UNC/fixture11a_20261117T141116Z/fixture11a.log"
A2_LOG_SHA = "9c2275aabefc0a8eb615f6428ae81d995ef37fa086565d3f62f1a16c66336651"
TRAIN_DIR = "knee_singlecoil_train/singlecoil_train"
PROBES = (   # rule A3 on the 97 A2 rows of fixture 11a (224 U41): minimum and maximum high-band count
    {"label": "min-high", "file": TRAIN_DIR + "/file1002115.h5", "slice": 16, "seed": 8933651549672418648,
     "bands": [1178, 3745, 1989], "map16": "a3001e9d970b9848", "mask16": "017b2d61dfbf3ab8"},
    {"label": "max-high", "file": TRAIN_DIR + "/file1000521.h5", "slice": 17, "seed": 10470870892658987006,
     "bands": [943, 3560, 2409], "map16": "6e3c83739ad5719e", "mask16": "45d149db94819dd1"},
)
CKPT = {"arm": "lam0", "seed": 2026, "epoch": 3, "role": "epoch3", "key": "milestone_003", "name": "weights_e003.pt"}

HW, N_ACQ, N_FREE, FLOW_DIM, BANK_ROWS = 96, 24, 6912, 13824, 128
N_VOLUMES, N_VAL_SLICES = 97, 3461
EVAL_BASE_SEED = 20261001
LEDGER_SCHEMA = "seqref-v05lt-ledger/1"
T64_ABS, T64_REL, T32_REL = 1e-12, 1e-10, 1e-5
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
HEX40 = re.compile(r"[0-9a-f]{40}")

log = logging.getLogger("dunc_f11b")
FAILS = []


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


def positive(v):
    return isinstance(v, float) and math.isfinite(v) and v > 0.0


# ---------------------------------------------------------------- pre-checks (nothing is created before these pass)
def prechecks(cfg):
    if sys.byteorder != "little":
        raise Abort("byte order is %s, little-endian required" % sys.byteorder)
    for key, rx in (("archive_commit", HEX40), ("ckpt_file_sha", HEX64), ("ckpt_state_sha", HEX64)):
        if not isinstance(cfg[key], str) or not rx.fullmatch(cfg[key]):
            raise Abort("argument %s is not a full lower-case hex identity" % key)
    script = os.path.realpath(__file__)
    code_dir = os.path.dirname(script)
    run_dir = os.path.dirname(code_dir)
    me = os.path.basename(script)
    want = {me, "diag_unc_adapter.py", "diag_unc_core.py"}
    have = set(os.listdir(code_dir))
    if me != "test_diag_unc_fixture11b.py" or have != want:
        raise Abort("code directory must hold exactly %s, found %s" % (sorted(want), sorted(have)))
    if set(os.listdir(run_dir)) != {os.path.basename(code_dir)}:
        raise Abort("run directory is not fresh: %s" % sorted(os.listdir(run_dir)))
    repo = cfg["repo"]
    if not cfg["selftest"]:
        if os.path.abspath(sys.executable) != INTERPRETER:
            raise Abort("interpreter %s is not %s" % (sys.executable, INTERPRETER))
    if not run_dir.startswith(repo + "/v06lt_evidence/DIAG_UNC/fixture11b_"):
        raise Abort("run directory %s is not under v06lt_evidence/DIAG_UNC/fixture11b_*" % run_dir)
    head = git(repo, "rev-parse", "HEAD").strip()
    if head != cfg["pin"]:
        raise Abort("HEAD %s is not the pin" % head)
    if git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() != cfg["branch"]:
        raise Abort("branch is not %s" % cfg["branch"])
    if git(repo, "status", "--porcelain").strip():
        raise Abort("Git status is not clean")
    if git(repo, "diff", "--cached", "--name-only").strip():
        raise Abort("something is staged")
    for rel, want_sha in sorted(cfg["closure"].items()):
        tree, pinned = sha_file(os.path.join(repo, rel)), sha_bytes(git(repo, "show", "%s:%s" % (cfg["pin"], rel), binary=True))
        if not (tree == pinned == want_sha):
            raise Abort("%s: working tree %s / pin %s / registered %s differ" % (rel, tree[:12], pinned[:12], want_sha[:12]))
    arch = git(repo, "rev-parse", "--verify", cfg["archive_commit"] + "^{commit}").strip()
    if arch != cfg["archive_commit"]:
        raise Abort("archive commit does not resolve to itself")
    ident = {}
    for name, commit, want_sha in (("diag_unc_core.py", cfg["core_commit"], cfg["core_sha"]),
                                   ("diag_unc_adapter.py", arch, cfg["adapter_sha"]), (me, arch, None)):
        exp = sha_file(os.path.join(code_dir, name))
        got = sha_bytes(git(repo, "show", "%s:%s%s" % (commit, DIAG_DIR, name), binary=True))
        if exp != got or (want_sha is not None and exp != want_sha):
            raise Abort("%s: export %s / archive %s (%s) / registered %s differ" % (name, exp[:12], got[:12], commit[:7], (want_sha or "n/a")[:12]))
        ident[name] = exp
    a2_path = os.path.join(repo, cfg["a2_log_rel"])
    if not os.path.isfile(a2_path) or sha_file(a2_path) != cfg["a2_log_sha"]:
        raise Abort("fixture 11a log %s is missing or does not have the registered sha256" % cfg["a2_log_rel"])
    with open(a2_path, "r", encoding="utf-8") as fh:
        a2_lines = [x.rstrip("\n") for x in fh if "[A2]" in x]
    for p in PROBES:
        line = "INFO [A2] INFO  %s|%d seed=%d bands low/mid/high=%s map=%s mask=%s" % (p["file"], p["slice"], p["seed"], p["bands"], p["map16"], p["mask16"])
        if a2_lines.count(line) != 1:
            raise Abort("probe %s is not registered by exactly one identical A2 row of the fixture 11a log" % p["label"])
    return {"script": script, "script_sha": ident[me], "code_dir": code_dir, "run_dir": run_dir, "head": head,
            "ident": ident, "a2_rows": len(a2_lines)}


# ---------------------------------------------------------------- access instrumentation (the real functions are always called)
class Guard:
    def __init__(self, h5py, torch, np, data_dir, probe_keys, ckpt_path):
        self.h5py, self.torch, self.np = h5py, torch, np
        self.data_dir, self.probe_keys, self.ckpt = data_dir, set(probe_keys), ckpt_path
        self.opens, self.reads, self.loads, self.forbidden, self.saved = [], [], [], [], []

    def forbid(self, what):
        self.forbidden.append(what)
        log.error("[ACC] forbidden access refused: %s", what)
        raise RuntimeError("access guard: %s" % what)

    def _swap(self, obj, attr, new):
        self.saved.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, new)

    def on(self):
        g, real_file, D = self, self.h5py.File, self.h5py.Dataset
        real_get, real_load = D.__getitem__, self.torch.load

        def file_open(name, mode="r", *a, **k):
            p = os.path.abspath(os.fspath(name)) if isinstance(name, (str, os.PathLike)) else None
            if (p is None or a or k or mode != "r" or os.path.dirname(p) != g.data_dir or not p.endswith(".h5")
                    or "singlecoil_val" in p or "singlecoil_val" in os.path.realpath(p)):
                g.forbid("h5py.File(%r, mode=%r)" % (p if p else type(name).__name__, mode))
            g.opens.append(p)
            return real_file(name, mode)

        def getitem(ds, args, *a, **k):
            key = (os.path.abspath(ds.file.filename), ds.name, int(args)) if type(args) is int else None
            if key is None or a or k or key[1] != "/kspace" or (key[0], key[2]) not in g.probe_keys:
                g.forbid("array read %s[%s] of %s" % (ds.name, type(args).__name__ if key is None else key[2], os.path.basename(ds.file.filename)))
            g.reads.append((key[0], key[2]))
            return real_get(ds, args)

        def denied(label):
            def fn(*_a, **_k):
                g.forbid(label)
            return fn

        def load(f, *a, **k):
            p = os.path.abspath(os.fspath(f)) if isinstance(f, (str, os.PathLike)) else None
            if p != g.ckpt or g.loads:
                g.forbid("torch.load(%r) (call %d)" % (p if p else type(f).__name__, len(g.loads) + 1))
            g.loads.append(p)
            return real_load(f, *a, **k)

        self._swap(self.h5py, "File", file_open)
        self._swap(D, "__getitem__", getitem)
        for attr in ("read_direct", "__array__", "__iter__"):
            self._swap(D, attr, denied("h5py.Dataset.%s" % attr))
        self._swap(self.torch, "load", load)
        self._swap(self.np, "load", denied("numpy.load"))

    def off(self):
        for obj, attr, old in reversed(self.saved):
            setattr(obj, attr, old)
        self.saved = []


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
    if not gate("CK", os.path.isfile(p), "%s ledger.jsonl present" % tag):
        return None
    with open(p, "rb") as fh:
        blob = fh.read()
    if not gate("CK", identities.get(os.path.abspath(p)) == sha_bytes(blob), "%s ledger sha256 == analysis inputs.identities" % tag):
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
        gate("CK", n >= 3 and all(o.get("schema") == LEDGER_SCHEMA for o in full), "%s %d records, schema %s" % (tag, n, LEDGER_SCHEMA)),
        gate("CK", [o.get("seq") for o in full] == list(range(n)), "%s seq contiguous from 0" % tag),
        gate("CK", n >= 3 and kinds[0] == "run_start" and kinds[-1] == "run_end", "%s run_start first, run_end last" % tag),
        gate("CK", n_self == n, "%s self hash verifies for %d/%d records" % (tag, n_self, n)),
        gate("CK", n >= 2 and n_prev == n - 1, "%s prev hash verifies for %d/%d links" % (tag, n_prev, n - 1)),
    ])
    by_epoch = {}
    for o in full:                                           # whitelist: nothing else of a record is kept
        if o.get("kind") != "commit":
            continue
        b = o.get("body") or {}
        ret = {k: {"name": v.get("name"), "sha256": v.get("sha256")} for k, v in (b.get("retained") or {}).items() if isinstance(v, dict)}
        by_epoch[b.get("epoch")] = {"seq": o.get("seq"), "body": {"model_state_sha256": b.get("model_state_sha256"),
                                                                 "roles_this_epoch": list(b.get("roles_this_epoch") or []), "retained": ret}}
    del full
    epochs = sorted(e for e in by_epoch if isinstance(e, int))
    ok = gate("CK", bool(epochs) and epochs == list(range(1, len(epochs) + 1)) and len(epochs) == kinds.count("commit"),
              "%s one commit per epoch 1..%d" % (tag, len(epochs))) and ok
    return {"by_epoch": by_epoch, "n": len(epochs)} if ok else None


# ---------------------------------------------------------------- import closure (by file, any module name)
def closure_gate(cfg, pre, stage):
    repo, prefixes = cfg["repo"] + os.sep, tuple(os.path.realpath(p) + os.sep for p in {sys.prefix, sys.base_prefix})
    exported = {os.path.join(pre["code_dir"], n): n for n in pre["ident"]}
    names, extra, bad_hash, bare, cache, run_mods = [], [], [], [], {}, set()
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not isinstance(f, str):
            continue
        f = os.path.realpath(f)
        if not f.startswith(repo) or f.startswith(prefixes):
            continue
        if f in exported:
            run_mods.add(exported[f])
            if sha_file(f) != pre["ident"][exported[f]]:
                bad_hash.append(name)
            continue
        rel = os.path.relpath(f, cfg["repo"])
        if rel not in cfg["closure"]:
            extra.append("%s (%s)" % (name, rel))
            continue
        if rel not in cache:
            cache[rel] = sha_file(f)
        if cache[rel] != cfg["closure"][rel]:
            bad_hash.append(name)
        names.append(name)
        if not name.startswith("seqref_mri"):
            bare.append((name, rel))
    bad_bare = [n for n, rel in bare if n not in BARE_ALIASES or rel != "seqref_mri/src/%s.py" % n]
    info("C1", "%s: %d project module names over %d pinned files: %s" % (stage, len(names), len(cache), names))
    gate("C1a", not extra and not bad_hash, "%s: every loaded project file is in the registered closure (225 §7) with its registered sha256; "
         "outside: %s; hash differs: %s" % (stage, extra or "none", bad_hash or "none"))
    gate("C1a", not bad_bare, "%s: bare-name modules %s are permitted aliases of seqref_mri/src files; not permitted: %s"
         % (stage, [n for n, _ in bare], bad_bare or "none"))
    gate("C1a", all(n in sys.modules for n in REQUIRED_SEAMS), "%s: the production seam modules are loaded" % stage)
    gate("C1b", run_mods == set(pre["ident"]), "%s: adapter, step-B core and this script are loaded from the run directory export %s"
         % (stage, sorted(run_mods)))


# ---------------------------------------------------------------- the checks
def checks(cfg, pre):
    tag = "F11B_SELFTEST_CHECKS" if cfg["selftest"] else "F11B_CHECKS"
    log.info("%s", VERSION)
    if cfg["selftest"]:
        log.warning("SELF-TEST MODE -- mock repository, synthetic data, stand-in model -- NOT FIXTURE-11B EVIDENCE")
    log.info("start_utc %s", utc())
    log.info("script %s sha256 %s", pre["script"], pre["script_sha"])
    log.info("HEAD %s (pin, branch %s, clean, nothing staged)", pre["head"], cfg["branch"])
    log.info("exports: adapter %s and this script == archive %s; diag_unc_core %s == archive %s", pre["ident"]["diag_unc_adapter.py"],
             cfg["archive_commit"], pre["ident"]["diag_unc_core.py"], cfg["core_commit"])
    log.info("pinned closure: %d files, working tree == pin == registered; fixture 11a log %s bound (%d A2 rows, both probes registered)",
             len(cfg["closure"]), cfg["a2_log_sha"][:16], pre["a2_rows"])
    log.info("checkpoint role (%s, %d, %s): file sha256 %s, model-state sha256 %s (run-card arguments)", CKPT["arm"], CKPT["seed"],
             CKPT["role"], cfg["ckpt_file_sha"], cfg["ckpt_state_sha"])

    import h5py
    import numpy as np
    import torch
    log.info("python %s | numpy %s | torch %s | h5py %s | byteorder %s | torch threads %d", sys.version.split()[0], np.__version__,
             torch.__version__, h5py.__version__, sys.byteorder, torch.get_num_threads())
    repo = cfg["repo"]
    sys.path.insert(0, pre["code_dir"])
    sys.path.insert(0, repo)
    data_dir = os.path.join(repo, "seqref_mri", "data", "fastmri", TRAIN_DIR)
    ckpt_path = os.path.join(repo, "seqref_mri", "results", "v05lt", "%d_%s" % (CKPT["seed"], CKPT["arm"]), "ckpt", CKPT["name"])
    guard = Guard(h5py, torch, np, data_dir, [(os.path.join(data_dir, os.path.basename(p["file"])), p["slice"]) for p in PROBES], ckpt_path)
    guard.on()
    try:
        return _checks(cfg, pre, guard, np, torch, tag, data_dir, ckpt_path)
    finally:
        guard.off()


def _checks(cfg, pre, guard, np, torch, tag, data_dir, ckpt_path):
    t_start = time.monotonic()
    repo, R = cfg["repo"], os.path.join(cfg["repo"], "seqref_mri", "results", "v05lt")
    import diag_unc_adapter as A
    import diag_unc_core as core
    s = A.seams()
    C, EV, tg, d3, ffr, fdm, dec = s.C, s.EV, s.tg, s.d3, s.ffr, s.fdm, s.dec
    tb, rdec, pio = (sys.modules[n] for n in ("seqref_mri.scripts.train_base", "seqref_mri.src.residual_decoder", "seqref_mri.src.preflight_io"))
    closure_gate(cfg, pre, "after import")
    gate("C1a", A.__version__ == "0.1" and tg.dec is sys.modules["residual_decoder"] and tg.ffr is ffr and d3.tg is tg,
         "adapter v%s; tiny_gate.dec is the bare-name residual_decoder; tiny_gate.ffr and d3.tg are the modules the adapter uses" % A.__version__)

    def tmax(t):
        return float(torch.max(torch.abs(t)))

    def bitwise(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

    def t64(a, b):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        return (a.shape == b.shape and bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
                and bool(np.all(np.abs(a - b) <= T64_ABS + T64_REL * np.abs(b))))

    # ---- C3 layout (slice independent)
    re0 = np.arange(N_FREE, dtype=np.float64) + 0.25
    im0 = -(np.arange(N_FREE, dtype=np.float64) + 0.5)
    packed = ffr.pack_scalars(re0, im0)
    back = ffr.unpack_scalars(packed)
    block = np.concatenate([re0, im0])
    gate("C3", packed.shape == (FLOW_DIM,) and packed.dtype == np.float64 and bitwise(packed[0::2], re0) and bitwise(packed[1::2], im0),
         "pack_scalars: scalar[2k] == Re[k] and scalar[2k+1] == Im[k] for all %d coordinates (distinct value per coordinate)" % N_FREE)
    gate("C3", bitwise(back[0], re0) and bitwise(back[1], im0), "unpack_scalars(pack_scalars(u)) == u bitwise")
    gate("C3", bitwise(A.interleaved_to_block(packed), block) and bitwise(A.block_to_interleaved(block), packed)
         and bitwise(A.block_to_interleaved(A.interleaved_to_block(packed)), packed),
         "adapter conversion interleaved <-> block [Re|Im] is exact and reversible")
    gate("C3", not bitwise(block, packed), "mutation detected: the block layout read as interleaved differs from the production layout")
    gate("C3", ffr.PACKING_ORDER_LITERAL == "interleaved_real_imag_per_complex_coordinate" and ffr.FLOW_DIM_REAL == FLOW_DIM
         and ffr.N_FREE_COMPLEX == N_FREE, "production constants: interleaved packing, %d real scalars, %d complex coefficients" % (FLOW_DIM, N_FREE))

    # ---- registered inputs (whitelisted fields only)
    inp = os.path.join(R, "inputs")
    el_path = os.path.join(inp, "v05lt_nmse_eligibility.json")
    doc, el_sha = load_json(el_path)
    gate("IN", el_sha == sidecar(el_path), "eligibility manifest sha256 == sidecar")
    prov = doc.get("provenance", {})
    el_order_sha, el_split_sha = prov.get("validation_order_sha256"), prov.get("split_manifest_sha256")
    slices = [(s_.get("file"), s_.get("slice_index"), s_.get("midpoint")) for s_ in doc.get("slices", [])]
    del doc, prov
    an_path = os.path.join(R, "analysis", "v05lt_analysis.json")
    doc, an_sha = load_json(an_path)
    gate("IN", an_sha == cfg["analysis_sha"] == sidecar(an_path), "analysis document sha256 == registered (196, G-D4) == sidecar")
    identities = dict((doc.get("inputs") or {}).get("identities") or {})
    del doc                                                   # analysis outcomes dropped unread
    inputs = A.load_inputs(repo)
    paths = inputs["paths"]
    sp_path = str(paths.inputs / "v05lt_split_manifest.json")
    gate("IN", sha_file(sp_path) == cfg["split_sha"] == sidecar(sp_path) == el_split_sha,
         "split manifest sha256 == registered == sidecar == eligibility provenance (also verified by the production loader)")
    gate("IN", inputs["p4"]["file_sha256"] == cfg["p4_sha"] == sha_file(str(inputs["p4_path"])) and inputs["p4"]["sidecar_verified"] is True,
         "P4 scaling artefact verified by the production loader: file sha256 %s == registered" % inputs["p4"]["file_sha256"][:16])
    gate("IN", sha_file(str(paths.p0s)) == cfg["p0s_sha"] and inputs["s_ref"] == S_REF, "P0S artefact sha256 == registered; S_ref == registered value")
    n_files = len(inputs["split_doc"]["train"]) + len(inputs["split_doc"]["validation"])
    gate("IN", len(inputs["split_doc"]["validation"]) == N_VOLUMES and len(inputs["holdout"]) > 0
         and not (inputs["holdout"] & {r["volume_id"] for r in inputs["split_doc"]["train"] + inputs["split_doc"]["validation"]}),
         "split manifest: %d training-directory volumes (%d validation); holdout-ID manifest verified (%d IDs, JSON only), disjoint"
         % (n_files, len(inputs["split_doc"]["validation"]), len(inputs["holdout"])))
    gate("IN", os.path.abspath(str(paths.data_root / TRAIN_DIR)) == data_dir and os.path.isdir(data_dir)
         and not os.path.isdir(str(paths.data_root / "singlecoil_train")),
         "data root %s; the production dataset resolves to %s" % (paths.data_root, TRAIN_DIR))

    # ---- phase A: production indexes (metadata opens only) and identity -> position mapping
    view = A.validation_view(inputs)
    gate("ACC", len(guard.opens) == 2 * n_files and len(set(guard.opens)) == n_files and not guard.reads,
         "index construction: %d HDF5 opens == 2 x %d training-directory files, %d array reads" % (len(guard.opens), n_files, len(guard.reads)))
    entries = [view.entry(i) for i in range(len(view))]
    gate("MAP", len(entries) == N_VAL_SLICES == len(slices) and entries == [(f, i) for f, i, _m in slices],
         "validation view: %d positions, identical in order to the eligibility manifest slices" % len(entries))
    gate("MAP", view.order_lines_sha256() == el_order_sha, "validation view order hash == eligibility provenance.validation_order_sha256")
    idents = [(p["file"], p["slice"]) for p in PROBES]
    positions = A.positions_for(view, idents)
    gate("MAP", [entries[q] for q in positions] == idents and all(slices[q][2] is True for q in positions) and len(set(positions)) == 2,
         "probe identities map to view positions %s; both are flagged midpoint in the eligibility manifest" % positions)
    gate("ACC", not guard.reads and len(guard.opens) == 2 * n_files, "identity mapping completed with 0 array reads and no further HDF5 open")
    for p in PROBES:
        seed = fdm.canonical_mask_seed(fdm.EVAL_BASE_SEED, p["file"], p["slice"])
        gate("ID", fdm.EVAL_BASE_SEED == EVAL_BASE_SEED and seed == p["seed"], "%s: canonical evaluation mask seed == fixture 11a A2 seed" % p["label"])

    # ---- checkpoint binding (before the file is loaded)
    le = load_ledger(R, CKPT["arm"], CKPT["seed"], identities)
    side = sidecar(ckpt_path)
    ok_ck = False
    if le is not None:
        ok, why = bound(le["by_epoch"], CKPT["epoch"], CKPT["key"], cfg["ckpt_file_sha"], cfg["ckpt_state_sha"])
        ok_ck = gate("CK", ok and ("name=%r" % CKPT["name"]) in why, "run-card file and state sha256 bound by the ledger rule (%s)" % why)
    ok_ck = gate("CK", side is not None and side == cfg["ckpt_file_sha"], "checkpoint sidecar sha256 == run-card file sha256") and ok_ck
    if not ok_ck:
        raise Abort("checkpoint binding failed; no slice is read and the checkpoint is not loaded")

    # ---- phase B: the two probe slices (production states + the fixture's independent reference)
    states = A.build_states(view, positions, inputs)
    gate("C9", len(states) == 2, "v05lt_eval.build_states returned 2 states (its own EVAL_STATE_MISMATCH check passed)")
    batch = tb._collate([view[q] for q in positions])
    prep = tb._prepare(batch, "cpu", test0=False)
    want_reads = sorted(guard.probe_keys) * 2
    gate("ACC", sorted(guard.reads) == sorted(want_reads) and len(guard.opens) == 2 * n_files + 4,
         "slice reads: %d array reads == each probe slice twice; %d further HDF5 opens" % (len(guard.reads), len(guard.opens) - 2 * n_files))
    band_grid = np.asarray(core.band_grid(HW))
    per = []
    for j, p in enumerate(PROBES):
        lab, st, meta = p["label"], states[j], batch["meta"][j]
        gate("ID", st["identity"] == {"file": p["file"], "slice_index": p["slice"]} and meta["file"] == p["file"]
             and meta["slice_index"] == p["slice"] and meta["split"] == "train" and meta["mode"] == "eval" and meta["mask_seed"] == p["seed"],
             "%s: state and item identity %s|%d, split train, mode eval, mask seed == A2" % (lab, p["file"], p["slice"]))
        m = st["mask"][0].numpy()
        acq = [int(c) for c in np.flatnonzero(m)]
        rows, cols = A.free_coordinates(st)
        counts = [int(x) for x in core.check_band_map(band_grid[rows, cols], N_FREE)]
        gate("ID", m.dtype == np.bool_ and m.shape == (HW,) and len(acq) == N_ACQ
             and m.tobytes() == fdm.make_cartesian_mask(HW, p["seed"]).tobytes()
             and pio.canonical_hash({"width": HW, "selected_columns": acq})[:16] == p["mask16"]
             and st["cmap"].payload()["map_payload_sha256"][:16] == p["map16"] and counts == p["bands"],
             "%s: live mask, coordinate map and band counts %s == fixture 11a A2 row (mask %s, map %s)" % (lab, counts, p["mask16"], p["map16"]))
        gate("ID", tuple(st["cond"].shape) == (1, 2, HW, HW) and st["cond"].dtype == torch.float32 and tuple(st["y"].shape) == (1, HW, HW)
             and st["y"].dtype == torch.complex64 and tuple(st["amax"].shape) == (1,) and st["target"].shape == (1, FLOW_DIM)
             and st["target"].dtype == np.float64 and tuple(st["x_true_c"].shape) == (HW, HW) and st["x_true_c"].dtype == torch.complex64,
             "%s: state shapes and dtypes (cond f32 (1,2,96,96), y c64 (1,96,96), target f64 (1,13824), x c64 (96,96))" % lab)
        x_c = torch.complex(prep["x_norm"][j, 0], prep["x_norm"][j, 1])
        c_c = torch.complex(prep["cond_in"][j, 0], prep["cond_in"][j, 1])
        gate("ID", torch.equal(prep["y"][j], st["y"][0]) and torch.equal(prep["amax"][j:j + 1], st["amax"]) and torch.equal(x_c, st["x_true_c"])
             and torch.equal(batch["mask"][j], st["mask"][0]), "%s: the second read of the slice reproduces y, amax, x and mask bitwise" % lab)

        # C9
        gate("C9", prep["cond_in"][j].dtype == st["cond"].dtype and torch.equal(prep["cond_in"][j], st["cond"][0]) and torch.equal(c_c, st["cond_c"]),
             "%s: prep['cond_in'] == state cond bitwise" % lab)

        # C10a (reported only)
        k_c = fdm.fft2c(c_c)
        e_free_c, den_c = tmax(dec.gather_unmeasured(k_c.unsqueeze(0), st["cmap"])), tmax(k_c)
        if positive(den_c) and math.isfinite(e_free_c):
            info("C10a", "%s: e = max|gather_unmeasured(fft2c(cond_in))| = %.6e; e / max|fft2c(cond_in)| = %.6e (reported, not a gate)" % (lab, e_free_c, e_free_c / den_c))
        else:
            gate("C10a", False, "%s: INVALID zero or non-finite norm of fft2c(cond_in)" % lab)

        # C10b
        u = A.observed_u(st)
        ref1 = np.asarray(dec.gather_unmeasured(fdm.fft2c(x_c - c_c).unsqueeze(0), st["cmap"]).detach().to(torch.complex128).numpy())[0]
        ref2 = EV.u_from_image(st, st["x_true_c"])
        gate("C10b", u.shape == (N_FREE,) and u.dtype == np.complex128 and bitwise(u, ref1) and bitwise(u, ref2) and bool(np.all(np.isfinite(u))),
             "%s: adapter U == gather_unmeasured(fft2c(x - cond_in)) bitwise (independent prepare, and v05lt_eval.u_from_image); (6912,) complex128" % lab)
        loc_index = inputs["p4"]["location_index"]
        mean_p, scale_p = np.empty(FLOW_DIM), np.empty(FLOW_DIM)
        for k in range(N_FREE):
            rec = loc_index[(int(rows[k]), int(cols[k]))]
            mean_p[2 * k], mean_p[2 * k + 1] = float(rec["applied_mean_re"]), float(rec["applied_mean_im"])
            scale_p[2 * k], scale_p[2 * k + 1] = float(rec["applied_scale_re"]), float(rec["applied_scale_im"])
        valid = bool(np.all(np.isfinite(mean_p)) and np.all(np.isfinite(scale_p)) and np.all(scale_p > 0))
        gate("C10b", valid, "%s: the %d registered P4 applied pairs at the slice's coordinates are finite with scale > 0 (INVALID otherwise)" % (lab, N_FREE))
        z_adp = A.standardise_interleaved(u, st)
        if valid:
            u_il = np.empty(FLOW_DIM)
            u_il[0::2], u_il[1::2] = u.real, u.imag
            ref_p4 = (u_il - mean_p) / scale_p
            loc_b, scale_b = A.loc_scale_block(st)
            z_core = A.block_to_interleaved(core.standardise(u, loc_b, scale_b))
            gate("C10b", z_adp.shape == (FLOW_DIM,) and z_adp.dtype == np.float64 and t64(z_adp, ref_p4),
                 "%s: adapter standardised U == (U - mean) / scale from the registered P4 pairs under T64 (bitwise: %s)" % (lab, bitwise(z_adp, ref_p4)))
            gate("C10b", t64(z_adp, st["target"][0]) and t64(z_core, z_adp),
                 "%s: adapter standardised U == production training target and == step-B standardise after block -> interleaved, under T64 "
                 "(bitwise: %s, %s)" % (lab, bitwise(z_adp, st["target"][0]), bitwise(z_core, z_adp)))
            re_s, im_s = ffr.unpack_scalars(z_adp)
            gate("C3", bitwise(z_adp[0::2], re_s) and bitwise(z_adp[1::2], im_s) and bitwise(A.block_to_interleaved(np.concatenate([re_s, im_s])), z_adp),
                 "%s: the standardised vector of the slice is interleaved (unpack and block conversion agree bitwise)" % lab)

        # C10c
        x128, c128 = st["x_true_c"].to(torch.complex128), st["cond_c"].to(torch.complex128)
        M = rdec.column_mask_tensor(st["cmap"], "cpu", torch.complex128)
        f_x, f_c, yn = fdm.fft2c(x128), fdm.fft2c(c128), EV.y_norm(st)
        d = fdm.ifft2c(M * (yn - f_x) - (1 - M) * f_c)
        x_hat = A.image_from_u(st, u)
        S, kmax = tmax(x128), tmax(f_x)
        r_id, r_eq, e_free, e_acq, d_max = tmax(x_hat - x128 - d), tmax(x_hat - x128), tmax((1 - M) * f_c), tmax(M * (yn - f_x)), tmax(d)
        finite = all(math.isfinite(v) for v in (r_id, r_eq, e_free, e_acq, d_max))
        if not (positive(S) and positive(kmax) and finite and tuple(x_hat.shape) == (HW, HW) and x_hat.dtype == torch.complex128):
            gate("C10c", False, "%s: INVALID zero or non-finite norm, or unexpected image shape/dtype" % lab)
        else:
            t32, tau = T32_REL * S, T32_REL * kmax
            gate("C10c", r_id <= t32, "%s: max|x_hat - x - d| = %.6e <= T32 = %.6e (S = max|x| = %.6e)" % (lab, r_id, t32, S))
            P = e_free <= tau and e_acq <= tau
            info("C10c", "%s: precondition P %s: e_free = %.6e, e_acq = %.6e, tau32 = %.6e, max|d| = %.6e" % (lab, "HOLDS" if P else "DOES NOT HOLD", e_free, e_acq, tau, d_max))
            if P:
                gate("C10c", r_eq <= t32, "%s: P holds, reconstruction equality max|x_hat - x| = %.6e <= T32 = %.6e" % (lab, r_eq, t32))
            else:
                info("C10c", "%s: P does not hold: RECORDED, not a failure; max|x_hat - x| = %.6e is not gated" % (lab, r_eq))
            # C10d (reported only)
            u_whole = np.asarray(dec.gather_unmeasured(fdm.fft2c(x_c).unsqueeze(0), st["cmap"]).detach().to(torch.complex128).numpy())[0]
            diff = float(np.max(np.abs(u_whole - u)))
            if math.isfinite(diff):
                info("C10d", "%s: whole-image U mutation: max|U_whole - U| = %.6e vs T32 = %.6e -> %s (reported, not a gate)"
                     % (lab, diff, t32, "DISCRIMINATING" if diff > t32 else "NON-DISCRIMINATING"))
            else:
                gate("C10d", False, "%s: INVALID non-finite mutation difference" % lab)
        per.append((lab, st))
        del x128, c128, f_x, f_c, yn, d, x_hat, k_c, ref1, ref2, z_adp

    # ---- C12 bank
    rec = A.z_bank()
    bank = rec["bank"]
    raw_sha = sha_bytes(np.ascontiguousarray(bank).tobytes(order="C"))
    gate("C12", isinstance(bank, np.ndarray) and bank.shape == (BANK_ROWS, FLOW_DIM) and bank.dtype == np.float32 and bank.flags["C_CONTIGUOUS"],
         "estimators.z_diag_bank(): shape (128, 13824), float32, C-contiguous")
    gate("C12", raw_sha == Z_DIAG == rec["bank_sha256"] == C.ZDIAG_BANK_SHA256, "bank raw C-order sha256 %s == registered X2 bank hash" % raw_sha[:16])
    gate("C12", rec["manifest_sha256"] == Z_MANIFEST == C.ZDIAG_MANIFEST_SHA256, "bank manifest sha256 %s == registered" % rec["manifest_sha256"][:16])
    mut = bank.copy()
    mut[0, 0] = np.nextafter(mut[0, 0], np.float32(np.inf))
    gate("C12", sha_bytes(mut.tobytes(order="C")) != Z_DIAG, "mutation detected: a bank differing in one element by one float32 step has another hash")
    del mut

    # ---- checkpoint (the only torch.load) and C11
    rng0 = EV._rng_states()
    model, vc = A.load_checkpoint(ckpt_path, cfg["ckpt_file_sha"], cfg["ckpt_state_sha"], seed=CKPT["seed"], arm=CKPT["arm"], epoch=CKPT["epoch"])
    state0 = C.model_state_sha256(model)
    gate("CK", guard.loads == [ckpt_path] and (vc.file_name, vc.file_sha256, vc.model_state_sha256, vc.seed, vc.arm, vc.epoch)
         == (CKPT["name"], cfg["ckpt_file_sha"], cfg["ckpt_state_sha"], CKPT["seed"], CKPT["arm"], CKPT["epoch"]) and state0 == cfg["ckpt_state_sha"]
         and model.training is False,
         "v05lt_eval.load_trained_checkpoint: one torch.load; file, state, seed, arm, epoch verified; recomputed state == registered; eval mode")
    info("CK", "roles stored in the checkpoint: %s" % list(vc.roles))
    z0 = torch.zeros(1, ffr.FLOW_DIM_REAL)
    for lab, st in per:
        t0 = time.monotonic()
        dec_adp = A.decode_bank(model, st, bank)
        with torch.no_grad():
            dec_ref = d3._decode_bank_conditioned(model, st, bank, st["cond"], st["mask"])
        gate("C11", isinstance(dec_adp, np.ndarray) and dec_adp.shape == (BANK_ROWS, N_FREE) and dec_adp.dtype == np.complex128
             and bool(np.all(np.isfinite(dec_adp))) and bitwise(dec_adp, dec_ref),
             "%s: adapter bank decode == d3._decode_bank_conditioned bitwise; (128, 6912) complex128, finite" % lab)
        del dec_adp, dec_ref
        with torch.no_grad():
            x0_ref, u0_ref = tg._decode_z(model, z0, st)
            u0_bank = d3._decode_bank_conditioned(model, st, np.zeros((1, FLOW_DIM), dtype=np.float32), st["cond"], st["mask"])[0]
        x0_adp, u0_adp = A.decode_r0(model, st)
        x0_img = A.image_from_u(st, u0_ref)
        gate("C11", u0_adp.shape == (N_FREE,) and u0_adp.dtype == np.complex128 and bitwise(u0_adp, u0_ref) and bitwise(u0_bank, u0_ref),
             "%s: R0 (z = 0) coefficients: adapter == tiny_gate._decode_z == one-row d3 bank route, bitwise" % lab)
        gate("C11", tuple(x0_adp.shape) == (HW, HW) and x0_adp.dtype == x0_ref.dtype and torch.equal(x0_adp, x0_ref) and torch.equal(x0_img, x0_ref),
             "%s: R0 image: adapter == tiny_gate._decode_z == image_from_u(R0 coefficients), bitwise" % lab)
        info("C11", "%s: decode section took %.1f s" % (lab, time.monotonic() - t0))
        del x0_ref, u0_ref, u0_bank, x0_adp, u0_adp, x0_img
    gate("IMM", C.model_state_sha256(model) == state0 and model.training is False and EV._rng_equal(rng0, EV._rng_states()),
         "model state hash, eval mode and the Python/NumPy/torch RNG states are unchanged by loading and decoding")
    del model, per, states, batch, prep, bank

    # ---- access totals, import closure, Git state
    gate("ACC", not guard.forbidden and len(guard.opens) == 2 * n_files + 4 and sorted(guard.reads) == sorted(want_reads)
         and guard.loads == [ckpt_path],
         "totals: %d HDF5 opens (2 x %d metadata + 4), %d array reads (2 probe slices, twice each), %d torch.load, 0 numpy.load, %d forbidden"
         % (len(guard.opens), n_files, len(guard.reads), len(guard.loads), len(guard.forbidden)))
    closure_gate(cfg, pre, "at end")
    head2, dirty = git(repo, "rev-parse", "HEAD").strip(), git(repo, "status", "--porcelain").strip()
    gate("C1a", head2 == cfg["pin"] and not dirty, "post-check: HEAD is the pin, status clean")
    try:
        import resource
        info("RES", "elapsed %.1f s; peak RSS %.0f MiB (reported only; not the X1(b) measurement)"
             % (time.monotonic() - t_start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0))
    except Exception as exc:
        log.error("[RES] resource report failed: %r", exc)
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
    tag = "F11B_SELFTEST_RESULT" if cfg["selftest"] else "F11B_RESULT"
    run_dir = pre["run_dir"]
    print("F11B_OUT=%s" % run_dir)
    log_path, problems = os.path.join(run_dir, "fixture11b.log"), []
    side_path, result_path = log_path + ".sha256", os.path.join(run_dir, "RESULT.txt")
    fh = None
    try:
        fault(cfg, "log_open")
        fh = logging.FileHandler(log_path, mode="x", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logging.getLogger().addHandler(fh)                   # root: adapter and production error messages are kept too
    except Exception as exc:
        log.error("cannot create the log file %s: %r", log_path, exc)
        problems.append("log file not created")
    checks_ok = False
    if fh is not None:
        try:
            checks_ok = checks(cfg, pre)
        except Abort as exc:
            log.error("F11B_STOP: %s", exc)
            log.error("%s: FAIL", "F11B_SELFTEST_CHECKS" if cfg["selftest"] else "F11B_CHECKS")
        except Exception as exc:
            log.error("unexpected error: %r", exc)
            log.error("%s", traceback.format_exc())
            log.error("%s: FAIL", "F11B_SELFTEST_CHECKS" if cfg["selftest"] else "F11B_CHECKS")
        finally:
            logging.getLogger().removeHandler(fh)
            fh.close()
        if not checks_ok:
            problems.append("checks failed (see fixture11b.log)")
    log_sha = None
    if fh is not None:
        try:
            log_sha = sha_file(log_path)
            fault(cfg, "sidecar")
            with open(side_path, "x", encoding="utf-8") as sf:
                sf.write("%s  fixture11b.log\n" % log_sha)
                sf.flush()
                os.fsync(sf.fileno())
            with open(side_path, "r", encoding="utf-8") as sf:
                parts = sf.read().split()
            if parts != [log_sha, "fixture11b.log"] or sha_file(log_path) != log_sha:
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
    sys.dont_write_bytecode = True                           # no __pycache__ is written into the pinned tree
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-commit")
    ap.add_argument("--ckpt-file-sha256")
    ap.add_argument("--ckpt-state-sha256")
    ap.add_argument("--selftest", metavar="CONFIG.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")
    try:
        cfg = dict(selftest=False, repo=REPO, pin=PIN, branch="seqref-v05lt", core_commit=CORE_COMMIT, core_sha=CORE_SHA,
                   adapter_sha=ADAPTER_SHA, archive_commit=a.archive_commit, ckpt_file_sha=a.ckpt_file_sha256,
                   ckpt_state_sha=a.ckpt_state_sha256, closure=dict(CLOSURE), split_sha=SPLIT_SHA, analysis_sha=ANALYSIS_SHA,
                   p4_sha=P4_SHA, p0s_sha=P0S_SHA, a2_log_rel=A2_LOG_REL, a2_log_sha=A2_LOG_SHA, fault=None)
        if a.selftest:
            if a.archive_commit or a.ckpt_file_sha256 or a.ckpt_state_sha256:
                raise Abort("self-test takes its identities from the config file only")
            with open(a.selftest, "r", encoding="utf-8") as fh:
                o = json.load(fh)
            need = {"repo", "pin", "core_commit", "archive_commit", "ckpt_file_sha", "ckpt_state_sha", "closure", "split_sha",
                    "analysis_sha", "p4_sha", "p0s_sha", "a2_log_sha"}
            if set(o) - (need | {"fault"}) or not need <= set(o):
                raise Abort("self-test config: keys must be exactly %s (+ optional fault)" % sorted(need))
            if os.path.realpath(o["repo"]) == os.path.realpath(REPO) or o["pin"] == PIN:
                raise Abort("self-test must not use the production repository or pin")
            if o.get("fault") not in (None, "log_open", "sidecar", "result"):
                raise Abort("self-test config: unknown fault")
            cfg.update(o)
            cfg.update(selftest=True, repo=os.path.abspath(o["repo"]), fault=o.get("fault"))
        elif not (a.archive_commit and a.ckpt_file_sha256 and a.ckpt_state_sha256):
            raise Abort("production mode needs --archive-commit, --ckpt-file-sha256 and --ckpt-state-sha256 (run card)")
        rc = run(cfg)
    except Abort as exc:
        log.error("F11B_ABORT: %s", exc)
        log.error("F11B_RESULT: ABORTED (nothing is deleted; do not rerun)")
        rc = 1
    except Exception as exc:
        log.error("unexpected error outside the run: %r", exc)
        log.error("F11B_RESULT: FAIL (nothing is deleted; do not rerun)")
        rc = 1
    print("F11B_EXIT=%d" % rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
