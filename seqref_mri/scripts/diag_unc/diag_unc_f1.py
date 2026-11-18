# SEQREF-DUNC-F1R v0.2 -- diag_unc_f1.py: DIAG-UNC step F1 (binding, resource probe, Tier 1 reproduction, manifest; plan SEQREF-DUNC-F1 v0.2.1, LudWiki 227)
# LIFETIME: KEEP
"""Step F1 of DIAG-UNC (223 v1.0 + AM1, §2 and §3.3). One run, stages S0-S4 of plan F1 v0.2.1.

  S0  pre-checks (abort; nothing but the run directory given by the run card exists)
  S1  bind and deduplicate: analysis document, 17 G-U1 post-evaluation records, six ledgers,
      final_weights.json, the 18 Tier 1 role files and the 12 Tier 2 milestone files (hashes only),
      stored validation rows, bank. 18 roles -> 17 states expected; anything else stops the run
      before the probe (HOLD_PROVENANCE, DIAG_UNC_INPUT_INVALID).
  S2  resource probe: exactly the two fixture-11b slices (two slice reads, two states) and the
      (lam0, 2026, epoch3) checkpoint; G and F1 projections; stop before S3 if a registered limit or
      the F1 planning cap is exceeded (HOLD_RESOURCE, DIAG_UNC_RESOURCE_LIMIT). Score values exist in
      memory only. The other 95 slice states are built at the start of S3 and timed there.
  S3  Tier 1 reproduction, 17 states x 97 midpoint slices, fail-fast at the first violation.
  S4  post-checks; manifest_f1.pending.json. The manifest proper is written only after the log is
      closed and its sidecar verified (see "Manifest validity").

Layout expected by the run card:
  <run_dir>/code/{diag_unc_f1.py, diag_unc_adapter.py, diag_unc_core.py, diag_unc_stats.py, diag_unc_guard.py}
Evidence: <run_dir>/f1.log, f1.log.sha256, RESULT.txt and, through the write guard (its root must be
a new directory, so it cannot be the run directory itself; plan amendment PA1),
<run_dir>/out/{probe.json, reproduction.json, manifest_f1.pending.json, manifest_f1.json | stop_record.json}.

Manifest validity: S4 writes manifest_f1.pending.json (marked pending, certifies nothing). After the
log is closed and its sidecar verified, and only if the result is PASS, manifest_f1.json is written
with the log sha256; RESULT.txt then records the manifest sha256. manifest_f1.json is valid only
together with a RESULT.txt whose first line is 'F1_RESULT: PASS' and that names its sha256. If
RESULT.txt cannot be written after the manifest exists, out/MANIFEST_INVALID.txt is written; nothing
is deleted.

Role -> file binding (registered records only; nothing is inferred):
  epoch3, selected : ckpt/<file_name> of the G-U1 record that carries the use; file sha256 == record ==
                     sidecar; bound to the state by the ledger rule (dunc_audit2 v0.1.2) for every
                     checkpoint label present in roles_this_epoch.
  final            : final/<weights_file> of final_weights.json (223 §1: the published final-weights
                     file); final_weights.json == sidecar, its line hashes == the verified ledger,
                     its state == the G-U1 record of the use == the last commit; file sha256 ==
                     weights_sha256 == sidecar. The G-U1 record may have evaluated another file
                     holding the same state; that file is named in the log and not used.
Every role file is hashed and loaded through v05lt_eval.load_trained_checkpoint (18 loads); a state
shared by two roles is decoded once and its deviations are written under both roles.

Reproduction (223 §3.3), per slice, metric values formed by the pinned functions on adapter decodes:
  R0 PSNR vs stored row psnr (0.001 dB); NLL vs stored row nll_per_dim (1e-4 nats/dim);
  R1 PSNR / NMSE_u vs G-U1 r1.primary["128"] (0.001 dB; 1e-4 relative);
  acquired-coefficient preservation, two gates per image (R0 image and each of the 128 sample
  images), both against the registered dc_tolerances.a of the G-U1 record:
    direct : max|M(F x - y)| / max|M y| of the image itself (dc_measures(x, x, y, m)["a"]) -- the image
             as decoded must already carry the acquired data;
    pinned : measure (a) of v05lt_eval.dc_measures on the hard-DC projection of the image (194 §8).
G-U1 rows must be in the registered midpoint order. A ledger entry's file name is compared for
milestone labels only: the verified rule (dunc_audit2 v0.1.2, 224 U12) binds rolling labels
(best_psnr, best_nll) by file sha256 and state, and the name of a rolling entry is not registered.
The NLL is model.log_prob_free(target_f32, cond, mask) as in v05lt_metrics.validate; the call is made
here, one slice at a time (validate uses batches of 32), and is not covered by fixture 11b.

Exposure: written and logged are identities, counts, absolute deviations, pass flags, times, memory
and byte counts. No PSNR, NLL, R1 or score value and no difference between roles is written or
logged; compared values stay in memory. JSON inputs are parsed and whitelisted fields kept.

Self-test mode: --selftest CONFIG.json (mock repository, synthetic data, stand-in model); refused
for the production repository or pin; output marked NOT F1 EVIDENCE. A test hook, not a fallback.
Exit codes: 0 PASS, 1 FAIL or ABORTED, 3 HOLD_RESOURCE or HOLD_PROVENANCE.
Every failure is logged. No fallback, placeholder, mock or dummy return in production mode.

bound() is copied unchanged from test_diag_unc_fixture11b.py v0.2 (dunc_audit2.py v0.1.2).

Stage-2 timing proxy: diag_unc_stats.run_registered reads only the retained per-slice records, whose
size does not depend on the number of coefficients or pixels (those costs are measured on the two
real probe slices, at 6,912 coefficients and 9,216 pixels). The proxy therefore uses small synthetic
arrays; a gate requires the synthetic record to have exactly the structure of the real probe record.

Changelog:
  v0.2 (2026-11-18) code review of v0.1 (HOLD): the probe reads and builds only its two slices, the
    other 95 states are built and timed in S3; the direct acquired-coefficient residual is a gate;
    G-U1 row order is required again; the synthetic proxy record must equal the real record in
    structure (gate PROXY); the manifest is written in two phases so that a late evidence failure
    cannot leave a manifest that reads as PASS. No tolerance, limit or binding rule changed.
  v0.1 (2026-11-18) first version, from plan SEQREF-DUNC-F1 v0.2.1. Not approved; kept as review evidence.
"""
import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

VERSION = "SEQREF-DUNC-F1R v0.2"
PLAN = "SEQREF-DUNC-F1 v0.2.1 (986d449b96aa5e5c0b15edcbada9c68490f5d1736059d76f5981dc8af8197a63)"
PIN = "7455da8027f33a6cca4dbf72f04e72420e938f45"
REPO = "/home/benjamin/CSMFII"
INTERPRETER = REPO + "/seqref_mri/.venv_mri/bin/python"
DIAG_DIR = "seqref_mri/scripts/diag_unc/"
ME = "diag_unc_f1.py"
EXPORTS = {   # exported module -> registered sha256 (adapter: 225 U45; step-B modules: 224 U33)
    "diag_unc_adapter.py": "c1715841bf6805568b88f9195acbe27933a7e1fec9acc8bb22a33ff555a8505e",
    "diag_unc_core.py": "a0d3b4a4079ff290a93098eb7a1c53173df596080a6b2b51469dc0f757c18b06",
    "diag_unc_stats.py": "b8706513a7da0517f477a81a49b4b7d2055e1451b47bec7dcc2c9b63a5de8f55",
    "diag_unc_guard.py": "b3dc9f69226f97275712b1bef5cfe9e0fdf1716f944c9ee137ca7223fe2e5bc5",
}

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
REQUIRED_SEAMS = ("seqref_mri.scripts.v05lt_common", "seqref_mri.scripts.v05lt_eval", "seqref_mri.scripts.v05lt_metrics",
                  "seqref_mri.scripts.tiny_gate", "seqref_mri.tdiag.d3", "seqref_mri.tdiag.estimators",
                  "seqref_mri.src.free_flow_runtime", "seqref_mri.src.fastmri_data", "residual_decoder")

SPLIT_SHA = "f7d21fb97e030a2e8227233cd5d1f171a120b765060ae83cafc728721428cf5d"
ANALYSIS_SHA = "5b267d3aa7ccb48bf3d7ccbfa2e2497d7f02df845aff8786c0fbe9b9dff2afa2"   # 196, G-D4; verified 224 U12
P4_SHA = "db0bf2c2e0cd3bd5e1ed86bf208198199f16df5b2fbbe0cc72903c8791f85684"
P0S_SHA = "2070cc72c977096f731811513d81b2abdb5dababb69edb4e997a5f8283b7c693"
S_REF = 15.62704352715211
Z_DIAG = "f8a6a34b6b474286359e4c99341de256b62bc0dff1e6bc8cf52e72f0461dc8d6"
Z_MANIFEST = "c4d2c0ec7c908f7c20a4f287eeffbdf966924a5118b6ba734b0a329942ab7e5f"
TRAIN_DIR = "knee_singlecoil_train/singlecoil_train"
PROBES = ((TRAIN_DIR + "/file1002115.h5", 16), (TRAIN_DIR + "/file1000521.h5", 17))   # fixture 11b probes (224 U41, 226 U50)
PROBE_ROLE = ("lam0", 2026, "epoch3")

RUNS = tuple((a, s) for a in ("lam0", "lam2") for s in (2026, 2027, 2028))
ROLES_T1 = ("epoch3", "selected", "final")
EPOCHS_T2 = (10, 30)
EXPECT_STATES = 17                                             # 223 §1, 224 U9
EXPECT_SHARED = (("lam2", 2028, "selected"), ("lam2", 2028, "final"))
N_VOLUMES, N_VAL_SLICES, BANK_ROWS, N_FREE, FLOW_DIM, HW = 97, 3461, 128, 6912, 13824, 96
LEDGER_SCHEMA, POSTEVAL_SCHEMA, ROWS_SCHEMA = "seqref-v05lt-ledger/1", "seqref-v05lt-posteval/1", "seqref-v05lt-val-rows/1"
TOL_PSNR_DB, TOL_NLL, TOL_NMSE_REL = 1e-3, 1e-4, 1e-4          # 223 §3.3
GIB = 2 ** 30
LIMITS = {"g_wall_s": 12 * 3600.0, "g_disk_bytes": 20 * GIB, "reserve_bytes": 64 * GIB,     # 223 §1 (224 U20), frozen
          "f1_cap_s": 12 * 3600.0}                                                         # F1 planning cap (plan A5), not frozen
MARGIN = 1.5
ALLOW_TIME_S, ALLOW_TIME_FRAC, ALLOW_DISK = 600.0, 0.10, GIB // 2                           # unmeasured allowances (plan A4)
RESULTS = ("PASS", "FAIL", "ABORTED", "HOLD_RESOURCE", "HOLD_PROVENANCE")
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
HEX40 = re.compile(r"[0-9a-f]{40}")
WEIGHTS_E = re.compile(r"weights_e\d{3}\.pt")

log = logging.getLogger("dunc_f1")
FAILS = []


class Abort(Exception):
    """Pre-check or structural failure with no typed code: result ABORTED."""


class Stop(Exception):
    """Typed stop: result FAIL, HOLD_RESOURCE or HOLD_PROVENANCE."""

    def __init__(self, result, code, reason):
        super().__init__("%s (%s): %s" % (result, code, reason))
        self.result, self.code, self.reason = result, code, reason


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


def need(cid, ok, msg):
    """A binding gate that must hold before the next step can be meaningful."""
    if not gate(cid, ok, msg):
        raise Stop("FAIL", "DIAG_UNC_INPUT_INVALID", "[%s] %s" % (cid, msg))


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


def fin(v):
    return isinstance(v, float) and math.isfinite(v)


def cjson(o):
    return json.dumps(o, sort_keys=True, indent=1).encode("utf-8")


# ---------------------------------------------------------------- pre-checks (S0)
def prechecks(cfg):
    if sys.byteorder != "little":
        raise Abort("byte order is %s, little-endian required" % sys.byteorder)
    if not isinstance(cfg["archive_commit"], str) or not HEX40.fullmatch(cfg["archive_commit"]):
        raise Abort("argument archive_commit is not a full lower-case hex identity")
    script = os.path.realpath(__file__)
    code_dir = os.path.dirname(script)
    run_dir = os.path.dirname(code_dir)
    want = {ME} | set(EXPORTS)
    have = set(os.listdir(code_dir))
    if os.path.basename(script) != ME or have != want:
        raise Abort("code directory must hold exactly %s, found %s" % (sorted(want), sorted(have)))
    if set(os.listdir(run_dir)) != {os.path.basename(code_dir)}:
        raise Abort("run directory is not fresh: %s" % sorted(os.listdir(run_dir)))
    repo = cfg["repo"]
    if not cfg["selftest"] and os.path.abspath(sys.executable) != INTERPRETER:
        raise Abort("interpreter %s is not %s" % (sys.executable, INTERPRETER))
    if not run_dir.startswith(repo + "/v06lt_evidence/DIAG_UNC/f1_"):
        raise Abort("run directory %s is not under v06lt_evidence/DIAG_UNC/f1_*" % run_dir)
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
    for name, want_sha in [(n, s) for n, s in sorted(EXPORTS.items())] + [(ME, None)]:
        exp = sha_file(os.path.join(code_dir, name))
        got = sha_bytes(git(repo, "show", "%s:%s%s" % (arch, DIAG_DIR, name), binary=True))
        if exp != got or (want_sha is not None and exp != want_sha):
            raise Abort("%s: export %s / archive %s / registered %s differ" % (name, exp[:12], got[:12], (want_sha or "n/a")[:12]))
        ident[name] = exp
    free = int(cfg["free_fn"](run_dir))
    if free < cfg["limits"]["reserve_bytes"]:
        raise Abort("free space %d bytes is below the registered reserve %d" % (free, cfg["limits"]["reserve_bytes"]))
    return {"script": script, "script_sha": ident[ME], "code_dir": code_dir, "run_dir": run_dir, "head": head,
            "ident": ident, "free_start": free}


# ---------------------------------------------------------------- access instrumentation (the real functions are always called)
class Access:
    def __init__(self, h5py, torch, np, data_dir):
        self.h5py, self.torch, self.np, self.data_dir = h5py, torch, np, data_dir
        self.slice_keys, self.ckpts = set(), set()
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
            if key is None or a or k or key[1] != "/kspace" or (key[0], key[2]) not in g.slice_keys:
                g.forbid("array read %s[%s] of %s" % (ds.name, type(args).__name__ if key is None else key[2], os.path.basename(ds.file.filename)))
            g.reads.append((key[0], key[2]))
            return real_get(ds, args)

        def denied(label):
            def fn(*_a, **_k):
                g.forbid(label)
            return fn

        def load(f, *a, **k):
            p = os.path.abspath(os.fspath(f)) if isinstance(f, (str, os.PathLike)) else None
            if p not in g.ckpts or p in g.loads:
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


# ---------------------------------------------------------------- ledger (rule copied from dunc_audit2.py v0.1.2 via fixture 11b v0.2)
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
    """Verified ledger -> whitelisted fields only. Any failure is a binding failure (Stop)."""
    tag = "(%s,%d)" % (arm, seed)
    p = os.path.join(R, "%d_%s" % (seed, arm), "ledger.jsonl")
    need("LED", os.path.isfile(p), "%s ledger.jsonl present" % tag)
    with open(p, "rb") as fh:
        blob = fh.read()
    need("LED", identities.get(os.path.abspath(p)) == sha_bytes(blob), "%s ledger sha256 == analysis inputs.identities" % tag)
    raw = [x for x in blob.split(b"\n") if x.strip()]
    full = [json.loads(x.decode("utf-8")) for x in raw]
    n = len(full)

    def canon(o):
        return json.dumps({k: v for k, v in o.items() if k != "self"}, sort_keys=True, separators=(",", ":")).encode()
    n_self = sum(1 for o in full if o.get("self") == sha_bytes(canon(o)))
    n_prev = sum(1 for i in range(1, n) if full[i].get("prev") == sha_bytes(raw[i - 1]))
    kinds = [o.get("kind") for o in full]
    need("LED", n >= 3 and all(o.get("schema") == LEDGER_SCHEMA for o in full) and [o.get("seq") for o in full] == list(range(n))
         and kinds[0] == "run_start" and kinds[-1] == "run_end" and n_self == n and n_prev == n - 1,
         "%s %d records: schema, contiguous seq, run_start first, run_end last, self %d/%d, prev %d/%d" % (tag, n, n_self, n, n_prev, n - 1))
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
    need("LED", bool(epochs) and epochs == list(range(1, len(epochs) + 1)) and len(epochs) == kinds.count("commit")
         and isinstance(sel, int) and not isinstance(sel, bool) and 1 <= sel <= len(epochs),
         "%s one commit per epoch 1..%d; selected epoch recorded in the last commit" % (tag, len(epochs)))
    return {"by_epoch": by_epoch, "n": len(epochs), "raw_sha": [sha_bytes(x) for x in raw], "kinds": kinds, "selected": sel, "path": p}


# ---------------------------------------------------------------- import closure (by file, any module name; as fixture 11b v0.2)
def closure_gate(cfg, pre, stage):
    repo, prefixes = cfg["repo"] + os.sep, tuple(os.path.realpath(p) + os.sep for p in {sys.prefix, sys.base_prefix})
    exported = {os.path.join(pre["code_dir"], n): n for n in pre["ident"]}
    names, extra, bad_hash, bare, cache, run_mods, rel_ok, rel_bad = [], [], [], [], {}, set(), [], []
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not isinstance(f, str):
            continue
        if not os.path.isabs(f):                             # not a path the import system loaded (e.g. torch.ops: '_ops.py')
            project = name.split(".")[0] in ("seqref_mri",) + BARE_ALIASES or name.startswith("diag_")
            (rel_bad if project or os.path.lexists(f) else rel_ok).append("%s (%s)" % (name, f))
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
    gate("C1a", not rel_bad, "%s: modules with a non-absolute __file__ %s are not project modules and name no existing file; "
         "not accepted: %s" % (stage, rel_ok, rel_bad or "none"))
    gate("C1a", not bad_bare, "%s: bare-name modules %s are permitted aliases of seqref_mri/src files; not permitted: %s"
         % (stage, [n for n, _ in bare], bad_bare or "none"))
    gate("C1a", all(n in sys.modules for n in REQUIRED_SEAMS), "%s: the production seam modules are loaded" % stage)
    gate("C1b", run_mods == set(pre["ident"]), "%s: this script, the adapter and the step-B modules are loaded from the run directory export %s"
         % (stage, sorted(run_mods)))


# ---------------------------------------------------------------- S1: bind and deduplicate
def s1_bind(cfg, A, s, R, repo, acc, data_dir, B):
    C, EV, fdm = s.C, s.EV, s.fdm
    n_vol, n_val = cfg["n_volumes"], cfg["n_val_slices"]
    V = B["verified"]                                         # identities verified so far (for stop_record / manifest)
    # ---- registered inputs
    an_path = os.path.join(R, "analysis", "v05lt_analysis.json")
    doc, an_sha = load_json(an_path)
    need("IN", an_sha == cfg["analysis_sha"] == sidecar(an_path), "analysis document sha256 == registered (196, G-D4; 224 U12) == sidecar")
    V["analysis_sha256"] = an_sha
    identities = dict((doc.get("inputs") or {}).get("identities") or {})
    masks_reg = (doc.get("inputs") or {}).get("masks_sha256")
    caps = set((doc.get("analysis") or {}).get("caps") or {})         # key names only; analysis outcomes are dropped unread
    del doc
    li_path = os.path.join(R, "launch", "v05lt_launch_inputs.json")
    li_doc, li_sha = load_json(li_path)
    li_reg = [v for v in (identities.get(os.path.abspath(li_path)), sidecar(li_path)) if v is not None]
    need("IN", bool(li_reg) and all(v == li_sha for v in li_reg),
         "launch inputs sha256 == its registered identity (analysis inputs.identities and/or sidecar; %d reference(s))" % len(li_reg))
    el_reg = ((li_doc.get("inputs") or {}).get("eligibility") or {}).get("sha256")
    del li_doc
    el_path = os.path.join(R, "inputs", "v05lt_nmse_eligibility.json")
    doc, el_sha = load_json(el_path)
    need("IN", el_sha == sidecar(el_path) == el_reg, "eligibility manifest sha256 == sidecar == launch-inputs registration")
    prov = doc.get("provenance", {})
    el_order_sha, el_split_sha = prov.get("validation_order_sha256"), prov.get("split_manifest_sha256")
    slices = [(x.get("file"), x.get("slice_index"), x.get("midpoint"), x.get("eligible"), x.get("volume_id")) for x in doc.get("slices", [])]
    del doc, prov
    inputs = A.load_inputs(repo)
    paths = inputs["paths"]
    sp_path = str(paths.inputs / "v05lt_split_manifest.json")
    need("IN", sha_file(sp_path) == cfg["split_sha"] == sidecar(sp_path) == el_split_sha,
         "split manifest sha256 == registered == sidecar == eligibility provenance")
    need("IN", inputs["p4"]["file_sha256"] == cfg["p4_sha"] == sha_file(str(inputs["p4_path"])) and inputs["p4"]["sidecar_verified"] is True,
         "P4 scaling artefact verified by the production loader; file sha256 == registered")
    need("IN", sha_file(str(paths.p0s)) == cfg["p0s_sha"] and inputs["s_ref"] == S_REF, "P0S artefact sha256 == registered; S_ref == registered value")
    n_files = len(inputs["split_doc"]["train"]) + len(inputs["split_doc"]["validation"])
    need("IN", len(inputs["split_doc"]["validation"]) == n_vol and len(inputs["holdout"]) > 0
         and not (inputs["holdout"] & {r["volume_id"] for r in inputs["split_doc"]["train"] + inputs["split_doc"]["validation"]}),
         "split manifest: %d training-directory volumes (%d validation); holdout-ID manifest verified (%d IDs, JSON only), disjoint"
         % (n_files, len(inputs["split_doc"]["validation"]), len(inputs["holdout"])))
    need("IN", os.path.abspath(str(paths.data_root / TRAIN_DIR)) == data_dir and os.path.isdir(data_dir),
         "the production dataset resolves to %s" % TRAIN_DIR)
    V.update(eligibility_sha256=el_sha, split_sha256=cfg["split_sha"], p4_sha256=cfg["p4_sha"], p0s_sha256=cfg["p0s_sha"], launch_inputs_sha256=li_sha)
    # ---- validation view (metadata opens only) and the registered midpoint population
    view = A.validation_view(inputs)
    need("ACC", len(acc.opens) == 2 * n_files and not acc.reads, "index construction: %d HDF5 opens == 2 x %d files, 0 array reads" % (len(acc.opens), n_files))
    entries = [view.entry(i) for i in range(len(view))]
    need("POP", len(entries) == n_val == len(slices) and entries == [(f, i) for f, i, _m, _e, _v in slices]
         and view.order_lines_sha256() == el_order_sha,
         "validation view: %d positions, identical in order to the eligibility manifest; order hash == eligibility provenance" % len(entries))
    mid_pos = [q for q, x in enumerate(slices) if x[2] is True]
    mid = [entries[q] for q in mid_pos]
    vols = [slices[q][4] for q in mid_pos]
    need("POP", len(mid) == n_vol and len(set(vols)) == n_vol and all(isinstance(slices[q][3], bool) for q in mid_pos)
         and all(p in mid for p in PROBES) and fdm.EVAL_BASE_SEED == 20261001,
         "%d registered midpoint slices, one per validation volume; both fixture-11b probe slices among them; EVAL_BASE_SEED registered" % len(mid))
    B.update(inputs=inputs, view=view, entries=entries, mid=mid, mid_pos=mid_pos, vols=vols, n_files=n_files,
             elig={entries[q]: slices[q][3] for q in mid_pos})
    V["population"] = {"n_midpoint": len(mid), "validation_order_sha256": el_order_sha}
    # ---- G-U1 post-evaluation records (17), bound through analysis inputs.identities
    pe_root = os.path.join(R, "posteval")
    dirs = sorted(d for d in os.listdir(pe_root) if os.path.isdir(os.path.join(pe_root, d)))
    reg_dirs = sorted(os.path.basename(os.path.dirname(k)) for k in identities if "/posteval/" in k and k.endswith("/posteval.json"))
    need("GU1", dirs == reg_dirs and all(os.path.isfile(os.path.join(pe_root, d, "posteval.json")) for d in dirs),
         "post-evaluation records on disk == records registered as analysis inputs (%d on disk, %d registered)" % (len(dirs), len(reg_dirs)))
    recs, by_use, masks = {}, {}, set()
    for d in dirs:
        p = os.path.join(pe_root, d, "posteval.json")
        j, sha = load_json(p)
        ck, pop, m = j.get("checkpoint") or {}, j.get("population") or {}, j.get("midpoint") or {}
        rows = m.get("rows") or []
        tol = j.get("dc_tolerances") or {}
        ok = (identities.get(os.path.abspath(p)) == sha and j.get("schema") == POSTEVAL_SCHEMA and d == ck.get("model_state_sha256")
              and (j.get("context") or {}).get("fixture") is False and (j.get("context") or {}).get("launch_inputs_sha256") == li_sha
              and (j.get("banks") or {}).get("primary") == Z_DIAG and pop.get("scope") == "midpoint" and pop.get("n") == n_vol
              and pop.get("eligibility_manifest_sha256") == el_sha and len(rows) == n_vol
              and (m.get("decodes") or {}).get("primary") == n_vol * BANK_ROWS and BANK_ROWS in (j.get("prefixes") or [])
              and fin(tol.get("a")) and tol["a"] > 0 and isinstance(ck.get("file_name"), str)
              and all(isinstance(ck.get(k), str) and HEX64.fullmatch(ck[k]) for k in ("file_sha256", "model_state_sha256")))
        need("GU1", ok, "record %s: registered identity, schema, directory == model state, non-fixture, launch inputs, Z_DIAG, population, "
             "%d rows, primary decodes, tolerance (a) valid" % (d[:12], n_vol))
        ref = {}
        for r in rows:                                       # whitelist: identity, eligibility and the two R1 reference values only
            ident = r.get("identity") or {}
            r1 = (((r.get("r1") or {}).get("primary") or {}).get(str(BANK_ROWS)) or {})
            ref[(ident.get("file"), ident.get("slice_index"))] = (r1.get("psnr"), r1.get("nmse_u"), r.get("eligible"))
        need("GU1", len(ref) == len(rows) and list(ref) == mid and all(fin(v[0]) and isinstance(v[2], bool) and (fin(v[1]) if v[2] else v[1] is None)
                                             and v[2] == B["elig"][k] for k, v in ref.items()),
             "record %s: row identities == the registered midpoint slices in the registered order; R1 primary/128 reference "
             "present and finite; eligibility flags == eligibility manifest" % d[:12])
        masks.add(str((j.get("masks") or {}).get("sha256")))
        recs[d] = {"dir": d, "sha": sha, "arm": ck.get("arm"), "seed": ck.get("seed"), "epoch": ck.get("epoch"), "file_name": ck["file_name"],
                   "file_sha": ck["file_sha256"], "state": ck["model_state_sha256"], "labels": list(ck.get("roles") or []),
                   "tol_a": float(tol["a"]), "ref": ref}
        for u in j.get("uses") or []:
            by_use.setdefault((u.get("arm"), u.get("seed"), u.get("role")), []).append(d)
        del j, rows
    need("GU1", len(masks) == 1 and next(iter(masks)) == masks_reg, "masks.sha256 is one value across the records == analysis inputs.masks_sha256")
    V["posteval"] = {d: r["sha"] for d, r in recs.items()}
    # ---- ledgers and final_weights.json
    led, fwj = {}, {}
    for arm, seed in RUNS:
        le = led[(arm, seed)] = load_ledger(R, arm, seed, identities)
        fwp = os.path.join(R, "%d_%s" % (seed, arm), "final", "final_weights.json")
        fw, fw_sha = load_json(fwp)
        src = fw.get("source") or {}
        ok = fw_sha == sidecar(fwp)
        for key, seqk, kind in (("commit_line_sha256", "commit_seq", "commit"), ("run_end_line_sha256", "run_end_seq", "run_end")):
            i = src.get(seqk)
            ok = (ok and isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(le["raw_sha"])
                  and src.get(key) == le["raw_sha"][i] and le["kinds"][i] == kind)
        last = le["by_epoch"][le["n"]]
        ok = (ok and fw.get("epoch") == le["n"] and last["seq"] == src.get("commit_seq")
              and fw.get("model_state_sha256") == last["body"]["model_state_sha256"]
              and isinstance(fw.get("weights_file"), str) and os.path.basename(fw["weights_file"]) == fw["weights_file"]
              and isinstance(fw.get("weights_sha256"), str) and HEX64.fullmatch(fw["weights_sha256"]) is not None)
        need("LED", ok, "(%s,%d) final_weights.json == sidecar; its commit and run_end line hashes == the verified ledger; epoch %s and "
             "model state == the last commit" % (arm, seed, fw.get("epoch")))
        fwj[(arm, seed)] = {"epoch": fw["epoch"], "state": fw["model_state_sha256"], "file": fw["weights_file"], "file_sha": fw["weights_sha256"], "sha": fw_sha}
        V.setdefault("ledgers", {})["%s|%d" % (arm, seed)] = sha_file(le["path"])
        V.setdefault("final_weights", {})["%s|%d" % (arm, seed)] = fw_sha
    # ---- the 18 Tier 1 roles: use -> record -> file -> state
    roles = []
    for arm, seed in RUNS:
        run_dir, le, fw = os.path.join(R, "%d_%s" % (seed, arm)), led[(arm, seed)], fwj[(arm, seed)]
        for role in ROLES_T1:
            tag = "(%s,%d,%s)" % (arm, seed, role)
            hits = by_use.get((arm, seed, role), [])
            need("ROLE", len(hits) == 1 and ("%s|%d|%s|r1|mid" % (arm, seed, role)) in caps,
                 "%s bound to exactly one G-U1 record and registered in the analysis document" % tag)
            r = recs[hits[0]]
            want_epoch = {"epoch3": 3, "selected": le["selected"], "final": le["n"]}[role]
            ok = r["arm"] == arm and r["seed"] == seed and r["epoch"] == want_epoch
            if role == "final":
                path, fsha, state, src = os.path.join(run_dir, "final", fw["file"]), fw["file_sha"], fw["state"], "final_weights.json"
                ok = ok and r["state"] == fw["state"]
                why = "published final-weights file; the G-U1 record evaluated %r" % r["file_name"]
            else:
                path, fsha, state, src = os.path.join(run_dir, "ckpt", r["file_name"]), r["file_sha"], r["state"], "G-U1 record"
                label = "milestone_003" if role == "epoch3" else "best_psnr"
                keys = [k for k in r["labels"] if k in le["by_epoch"][want_epoch]["body"]["roles_this_epoch"]] if ok else []
                ok = ok and WEIGHTS_E.fullmatch(r["file_name"]) is not None and label in r["labels"] and bool(keys)
                why = "no label of the record is in roles_this_epoch"
                whys = []
                for k in keys:                               # every label of the record that the commit carries must bind file and state
                    good, w = bound(le["by_epoch"], want_epoch, k, fsha, state)
                    whys.append("%s: %s" % (k, w))
                    ok = ok and good and (not k.startswith("milestone_") or ("name=%r" % r["file_name"]) in w)   # name fixed for milestone entries only
                why = "; ".join(whys) or why
            ok = ok and os.path.isfile(path) and sidecar(path) == fsha
            need("ROLE", ok, "%s -> %s: record arm/seed/epoch %s, file and state registered together in %s, sidecar == registered file sha256 (%s)"
                 % (tag, os.path.relpath(path, R), want_epoch, src, why))
            got = sha_file(path)
            need("ROLE", got == fsha, "%s file sha256 == registered (%s)" % (tag, fsha[:16]))
            roles.append({"arm": arm, "seed": seed, "role": role, "epoch": want_epoch, "path": path, "file_sha256": fsha,
                          "model_state_sha256": state, "bound_by": src, "posteval": r["dir"]})
    states = []
    for r in roles:
        if r["model_state_sha256"] not in states:
            states.append(r["model_state_sha256"])
        r["state_index"] = states.index(r["model_state_sha256"])
    B.update(recs=recs, led=led, roles=roles, states=states)
    V["roles"] = [{k: (os.path.relpath(v, R) if k == "path" else v) for k, v in r.items()} for r in roles]
    info("ROLE", "%d Tier 1 roles, %d file identities verified, %d distinct model states" % (len(roles), len({r["path"] for r in roles}), len(states)))
    # ---- deduplication is a provenance statement (plan C3): anything but the registered configuration stops before the probe
    shared = sorted((r["arm"], r["seed"], r["role"]) for st in states for r in roles
                    if r["model_state_sha256"] == st and sum(1 for q in roles if q["model_state_sha256"] == st) > 1)
    if len(states) != EXPECT_STATES or shared != sorted(EXPECT_SHARED) or len({r["path"] for r in roles}) != len(roles):
        gate("DEDUP", False, "expected %d states with exactly %s sharing one state and %d distinct files; found %d states, shared roles %s, %d files"
             % (EXPECT_STATES, list(EXPECT_SHARED), len(roles), len(states), shared, len({r["path"] for r in roles})))
        raise Stop("HOLD_PROVENANCE", "DIAG_UNC_INPUT_INVALID", "deduplication differs from the registered configuration: %d states, shared roles %s"
                   % (len(states), shared))
    gate("DEDUP", True, "18 roles -> %d states; only %s share a state (two files, both verified); the role table keeps all 18 rows" % (len(states), list(EXPECT_SHARED)))
    # ---- stored validation rows (Tier 1 role epochs and the Tier 2 epochs); no difference between files is formed
    def rows_of(arm, seed, epoch, state, keep):
        p = os.path.join(R, "%d_%s" % (seed, arm), "ckpt", "val_e%03d.json" % epoch)
        j, sha = load_json(p)
        rows = j.get("rows") or []
        ok = (sha == sidecar(p) and identities.get(os.path.abspath(p), sha) == sha and j.get("schema") == ROWS_SCHEMA and j.get("arm") == arm
              and j.get("seed") == seed and j.get("epoch") == epoch and j.get("model_state_sha256") == state == led[(arm, seed)]["by_epoch"][epoch]["body"]["model_state_sha256"]
              and j.get("n_rows") == n_val == len(rows) and [(r.get("file"), r.get("slice_index")) for r in rows] == entries)
        out = {}
        if ok:
            for q in mid_pos:
                r = rows[q]
                ok = (ok and r.get("midpoint") is True and fin(r.get("psnr")) and fin(r.get("nll_per_dim")) and r.get("eligible") == B["elig"][entries[q]])
                out[entries[q]] = (r.get("psnr"), r.get("nll_per_dim"))
        need("ROWS", ok, "(%s,%d) val_e%03d.json == sidecar; schema, arm, seed, epoch; model state == role state == ledger commit; %d rows in "
             "the registered order; the %d midpoint rows carry finite R0 PSNR and NLL" % (arm, seed, epoch, n_val, n_vol))
        V.setdefault("stored_rows", {})["%s|%d|e%03d" % (arm, seed, epoch)] = sha
        return out if keep else None
    B["rows"] = {}
    for r in roles:
        key = (r["arm"], r["seed"], r["epoch"])
        if key not in B["rows"]:
            B["rows"][key] = rows_of(r["arm"], r["seed"], r["epoch"], r["model_state_sha256"], True)
    # ---- Tier 2: files hashed and bound by the ledger rule; never loaded or decoded in F1
    tier2 = []
    for arm, seed in RUNS:
        for e in EPOCHS_T2:
            le = led[(arm, seed)]
            path = os.path.join(R, "%d_%s" % (seed, arm), "ckpt", "weights_e%03d.pt" % e)
            need("T2", os.path.isfile(path) and e in le["by_epoch"], "(%s,%d,epoch%d) milestone file and commit record present" % (arm, seed, e))
            state, fsha = le["by_epoch"][e]["body"]["model_state_sha256"], sha_file(path)
            good, why = bound(le["by_epoch"], e, "milestone_%03d" % e, fsha, state)
            need("T2", good and ("name=%r" % os.path.basename(path)) in why and sidecar(path) == fsha,
                 "(%s,%d,epoch%d) file sha256 == sidecar; file <-> state bound by the ledger rule (%s)" % (arm, seed, e, why))
            rows_of(arm, seed, e, state, False)
            tier2.append({"arm": arm, "seed": seed, "role": "epoch%d" % e, "epoch": e, "path": os.path.relpath(path, R), "file_sha256": fsha,
                          "model_state_sha256": state, "bound_by": "ledger rule"})
    B["tier2"] = tier2
    B["n_states_both"] = len(set(states) | {t["model_state_sha256"] for t in tier2})
    V["tier2_roles"] = tier2
    info("T2", "%d Tier 2 files bound, not loaded; Tier 1 + Tier 2: %d roles, %d distinct states" % (len(tier2), len(roles) + len(tier2), B["n_states_both"]))
    # ---- bank
    rec = A.z_bank()
    bank = rec["bank"]
    need("BANK", bank.shape == (BANK_ROWS, FLOW_DIM) and str(bank.dtype) == "float32" and rec["bank_sha256"] == Z_DIAG == sha_bytes(bank.tobytes(order="C"))
         and rec["manifest_sha256"] == Z_MANIFEST, "Z_DIAG bank regenerated: (128, 13824) float32, raw and manifest hashes == registered")
    B["bank"] = bank
    V["z_diag_sha256"] = Z_DIAG
    need("ACC", not acc.reads and not acc.loads and len(acc.opens) == 2 * n_files, "S1 completed with 0 array reads, 0 torch.load and no further HDF5 open")


# ---------------------------------------------------------------- one slice: own values (memory only) through the pinned functions
def slice_values(ctx, model, st, eligible, keep_images=False):
    """Own R0/NLL/R1/DC values of one slice and state. Returns (values, timings, sample images or None, decodes).
    The values never leave memory; only their deviations from the registered references are written."""
    A, EV, M, torch, np = ctx["A"], ctx["EV"], ctx["M"], ctx["torch"], ctx["np"]
    T, t = {}, time.perf_counter()
    x0, u0 = A.decode_r0(model, st)
    T["r0_decode"], t = time.perf_counter() - t, time.perf_counter()
    dec = A.decode_bank(model, st, ctx["bank"])
    T["bank_decode"], t = time.perf_counter() - t, time.perf_counter()
    with torch.no_grad():
        tgt = torch.from_numpy(np.ascontiguousarray(st["target"], dtype=np.float32))
        lp = model.log_prob_free(tgt, st["cond"], st["mask"])
    nll = float((-lp.to(torch.float64) / int(tgt.shape[1]))[0])
    T["nll"], t = time.perf_counter() - t, time.perf_counter()
    x1, u_mean = EV.r1_image(st, dec, BANK_ROWS)
    vals = {"r0_psnr": M.psnr_record(x0.abs(), st["x_true_mag"])["psnr"], "nll": nll,
            "r1_psnr": M.psnr_record(x1.abs(), st["x_true_mag"])["psnr"],
            "r1_nmse": M.nmse_value(u_mean, st["u_true"]) if eligible else None}
    T["metrics"], t = time.perf_counter() - t, time.perf_counter()
    m, y = EV._dc_mask(st), EV.y_norm(st)
    where = "%s|%s" % (st["identity"]["file"], st["identity"]["slice_index"])
    a_max = direct = 0.0
    imgs = [] if keep_images else None
    for k in range(-1, dec.shape[0]):                        # -1: the R0 image; 0..127: every sample image
        x = x0 if k < 0 else A.image_from_u(st, dec[k])
        if k >= 0 and keep_images:
            imgs.append(np.asarray(x.detach().numpy()).reshape(-1))
        a_max = max(a_max, EV.dc_measures(x, EV.hard_dc(x, y, m), y, m, where)["a"])
        direct = max(direct, EV.dc_measures(x, x, y, m, where)["a"])
    vals.update(dc_a=a_max, dc_direct=direct)
    T["images_dc"] = time.perf_counter() - t
    if not (bool(np.all(np.isfinite(dec))) and all(v is None or math.isfinite(v) for v in vals.values())):
        raise Stop("FAIL", "DIAG_UNC_NONFINITE", "non-finite decode or value at %s" % where)
    return vals, T, imgs, (dec, x0, u0)


def load_role(ctx, acc, r):
    """Hash-verified load of one role file through the production factory; returns (model, seconds)."""
    A, C = ctx["A"], ctx["C"]
    t = time.perf_counter()
    acc.ckpts.add(os.path.abspath(r["path"]))
    model, vc = A.load_checkpoint(r["path"], r["file_sha256"], r["model_state_sha256"], seed=r["seed"], arm=r["arm"], epoch=r["epoch"])
    ok = (vc.file_sha256 == r["file_sha256"] and vc.model_state_sha256 == r["model_state_sha256"] == C.model_state_sha256(model)
          and (vc.seed, vc.arm, vc.epoch) == (r["seed"], r["arm"], r["epoch"]) and model.training is False)
    need("CK", ok, "(%s,%d,%s) %s loaded by v05lt_eval.load_trained_checkpoint: file, state, seed, arm, epoch verified; recomputed state == "
         "registered; eval mode" % (r["arm"], r["seed"], r["role"], os.path.basename(r["path"])))
    return model, time.perf_counter() - t


def synthetic_record(core, np, ri, si, v):
    """Deterministic synthetic stage-1 record (arithmetic pattern, no random generator). Timing proxy only."""
    K, C_, P = BANK_ROWS, 12, 16
    ph = 0.37 * ri + 0.11 * si + 0.013 * v
    k, c, p = np.arange(K)[:, None], np.arange(C_)[None, :], np.arange(P)[None, :]
    u = (1.0 + 0.05 * ri) * (np.sin(0.7 * k + 1.3 * c + ph) + 1j * np.cos(0.9 * k + 0.5 * c + 2 * ph))
    x = (1.0 + 0.03 * si) * (np.sin(0.4 * k + 0.8 * p + ph) + 1j * np.cos(1.1 * k + 0.3 * p + ph)) + 2.0
    cc, pp = np.arange(C_), np.arange(P)
    return core.slice_stats(u, 0.3 * np.sin(cc + ph) + 0.2j * np.cos(cc - ph), 0.1 * np.cos(cc + ph) + 0.1j, np.zeros(2 * C_), np.ones(2 * C_),
                            cc % 3, x, 2.0 + 0.2 * np.sin(pp + ph) + 0.1j * np.cos(pp), 2.1 + 0.1 * np.cos(pp + ph) + 0.1j, 1.0 + 0.01 * v,
                            case_id=(ri, si, v))


def stage2_proxy(core, stats, np, idx, n_vol, tier2):
    roles = stats.registered_roles(tier2)
    pop = stats.Population(list(range(n_vol)))
    for ri, role in enumerate(roles):
        for si, seed in enumerate((2026, 2027, 2028)):
            for v in range(n_vol):
                pop.add(role, seed, v, synthetic_record(core, np, ri, si, v))
    pop.seal()
    t = time.perf_counter()
    stats.run_registered(pop, idx, tier2=tier2)              # result discarded: synthetic input, timing only
    return time.perf_counter() - t


def project(t_slice, t_load, t_stage2, rec_bytes, states, roles, n_vol):
    """Plan F1 v0.2.1 §4 S2: unadjusted W1 and R first; allowances from them only; margin applied once."""
    w1 = t_slice * states * n_vol + t_load * states
    rb = rec_bytes * roles * n_vol
    at, ad = max(ALLOW_TIME_S, ALLOW_TIME_FRAC * w1), max(ALLOW_DISK, rb)
    return {"states": states, "roles": roles, "W1_s": w1, "R_bytes": rb, "t_stage2_s": t_stage2, "allowance_time_s": at,
            "allowance_disk_bytes": ad, "margin": MARGIN, "T_s": MARGIN * (w1 + t_stage2 + at), "D_bytes": int(math.ceil(MARGIN * (rb + ad)))}


def within(p, free, lim):
    return p["T_s"] <= lim["g_wall_s"] and p["D_bytes"] <= lim["g_disk_bytes"] and free - p["D_bytes"] >= lim["reserve_bytes"]


# ---------------------------------------------------------------- S2: states, probe, projections, gate
def build_state(ctx, B, ident):
    """One slice state through the adapter, checked against the registered population. Returns (state, seconds)."""
    A, np, fdm = ctx["A"], ctx["np"], ctx["fdm"]
    f, i = ident
    t = time.perf_counter()
    st = A.build_states(B["view"], [B["mid_pos"][B["mid"].index(ident)]], B["inputs"])[0]
    dt = time.perf_counter() - t
    seed = fdm.canonical_mask_seed(fdm.EVAL_BASE_SEED, f, i)
    mask = st["mask"][0].numpy()
    if not (st["identity"] == {"file": f, "slice_index": i} and mask.dtype == np.bool_ and mask.tobytes() == fdm.make_cartesian_mask(HW, seed).tobytes()
            and (not st["excluded"]) == B["elig"][ident]):
        gate("POP", False, "%s|%d: state identity, regenerated mask bytes or eligibility differ from the registered ones" % (f, i))
        raise Stop("FAIL", "DIAG_UNC_POPULATION_MISMATCH", "slice state %s|%d does not match the registered population" % (f, i))
    return st, dt


def signature(o):
    """Structure of a stage-1 record: keys, container lengths and value types. No value."""
    if isinstance(o, dict):
        return {str(k): signature(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0])) if k != "case_id"}
    if isinstance(o, (list, tuple)):
        return [type(o).__name__, len(o)] + [signature(v) for v in o]
    if hasattr(o, "shape") and hasattr(o, "dtype"):
        return ["ndarray", list(o.shape)]
    return "number" if isinstance(o, (int, float)) and not isinstance(o, bool) else type(o).__name__


def s2_probe(cfg, ctx, B, acc, out, R):
    A, C, np, torch, core, stats, fdm = ctx["A"], ctx["C"], ctx["np"], ctx["torch"], ctx["core"], ctx["stats"], ctx["fdm"]
    n_vol, lim, data_dir = cfg["n_volumes"], cfg["limits"], acc.data_dir
    key_of = lambda ident: (os.path.join(data_dir, os.path.basename(ident[0])), ident[1])
    acc.slice_keys = {key_of(p) for p in PROBES}             # the probe may read its two slices only
    B["slice_states"], B["t_state"] = {}, {}
    for p in PROBES:
        B["slice_states"][p], B["t_state"][p] = build_state(ctx, B, p)
    need("ACC", sorted(acc.reads) == sorted(acc.slice_keys) and len(acc.opens) == 2 * B["n_files"] + len(PROBES),
         "probe: %d array reads == the two registered probe slices, once each; %d further HDF5 opens" % (len(acc.reads), len(acc.opens) - 2 * B["n_files"]))
    states, t_state = B["slice_states"], [B["t_state"][p] for p in PROBES]
    probe_role = next(r for r in B["roles"] if (r["arm"], r["seed"], r["role"]) == PROBE_ROLE)
    model, t_load = load_role(ctx, acc, probe_role)
    band_grid = np.asarray(core.band_grid(HW))
    per = []
    for f, i in PROBES:
        st = states[(f, i)]
        t0 = time.perf_counter()
        vals, T, imgs, (dec, x0, u0) = slice_values(ctx, model, st, B["elig"][(f, i)], keep_images=True)
        t_repro = time.perf_counter() - t0
        t = time.perf_counter()
        rows, cols = A.free_coordinates(st)
        loc, scale = A.loc_scale_block(st)
        x_true = np.asarray(st["x_true_c"].to(torch.complex128).numpy()).reshape(-1)
        ref_energy = float((fdm.fft2c(st["x_true_c"].to(torch.complex128)).abs() ** 2).sum())
        rec = core.slice_stats(dec, A.observed_u(st), u0, loc, scale, band_grid[rows, cols], np.stack(imgs), x_true,
                               np.asarray(x0.detach().numpy()).reshape(-1), ref_energy, case_id="%s|%d" % (f, i))
        T["slice_stats"], t = time.perf_counter() - t, time.perf_counter()
        nbytes = len(core.record_to_bytes(rec))              # serialised in memory only; never written (plan A3)
        T["serialise"] = time.perf_counter() - t
        sig_real = signature(rec)
        del rec, vals, imgs, dec, x0, u0
        per.append({"file": f, "slice_index": i, "sections_s": T, "record_bytes": nbytes, "t_repro_s": t_repro,
                    "t_all_s": sum(T.values()), "t_decode_s": T["r0_decode"] + T["bank_decode"]})
    t_state_max = max(t_state)
    need("PROXY", signature(synthetic_record(core, np, 0, 0, 0)) == sig_real,
         "the synthetic record of the stage-2 timing proxy has exactly the structure (keys, container lengths, value types) of the real probe "
         "record; run_registered reads these records only, so its cost does not depend on the coefficient or pixel count")
    t_slice = t_state_max + max(p["t_all_s"] for p in per)
    rec_bytes = max(p["record_bytes"] for p in per)
    idx = stats.bootstrap_matrix(n_boot=cfg["n_boot"], n_volumes=n_vol)
    boot = {"sha256": stats.verify_matrix(idx, stats.matrix_identity(idx), n_volumes=n_vol, n_boot=cfg["n_boot"]), "seed": stats.DIAG_UNC_SEED,
            "generator": "numpy.random.Generator(PCG64(seed)).integers(0, n_volumes, size=(n_boot, n_volumes), dtype=int64)",
            "shape": [int(idx.shape[0]), int(idx.shape[1])], "dtype": "<i8", "numpy": np.__version__,
            "identity": "sha256 of the raw little-endian int64 bytes, C order, no header"}
    t2 = {"tier1": stage2_proxy(core, stats, np, idx, n_vol, False), "tier1_tier2": stage2_proxy(core, stats, np, idx, n_vol, True)}
    del idx
    free = int(cfg["free_fn"](out.root))
    proj = {"tier1": project(t_slice, t_load, t2["tier1"], rec_bytes, len(B["states"]), len(B["roles"]), n_vol),
            "tier1_tier2": project(t_slice, t_load, t2["tier1_tier2"], rec_bytes, B["n_states_both"], len(B["roles"]) + len(B["tier2"]), n_vol)}
    for k in proj:
        proj[k]["within_registered_limits"] = within(proj[k], free, lim)
    f1_s3 = MARGIN * (max(p["t_repro_s"] for p in per) * len(B["states"]) * n_vol + t_load * len(B["roles"]))
    import resource
    probe = {"version": VERSION, "utc": utc(), "selftest": cfg["selftest"],
             "workload": {"slices": [[f, i] for f, i in PROBES], "checkpoint": "%s|%d|%s" % PROBE_ROLE, "n_slices": 2, "n_checkpoints": 1,
                          "slice_reads": len(acc.reads), "states_built": len(states)},
             "measured": {"t_load_s": t_load, "t_state_build_max_s": t_state_max, "t_state_build_mean_s": sum(t_state) / len(t_state),
                          "per_slice": per, "t_slice_s": t_slice, "record_bytes_max": rec_bytes,
                          "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, "free_bytes": free},
             "proxy_synthetic": {"t_stage2_s": t2, "n_volumes": n_vol, "n_boot": cfg["n_boot"], "record_structure_equals_real_probe_record": True,
                                 "note": "diag_unc_stats.run_registered on deterministic synthetic records: registered roles, seeds, volumes, "
                                 "subsets and bootstrap size; record structure equal to the real probe record (gate PROXY). The per-slice cost at "
                                 "6,912 coefficients and 9,216 pixels is in 'measured', not here. A proxy, not a measurement of the real run"},
             "unmeasured_allowance": {"time_s": "max(%g, %g x W1)" % (ALLOW_TIME_S, ALLOW_TIME_FRAC), "disk_bytes": "max(%d, R)" % ALLOW_DISK,
                                      "note": "plotting, output writing; not measured in F1"},
             "limits": lim, "projection_probe": proj, "f1_s3_projection_s": f1_s3, "bootstrap_index_matrix": boot,
             "note": "times, memory, byte counts, identities and projections only; no score value is stored"}
    B["probe"], B["boot"] = probe, boot
    B["probe_sha"] = out.write("probe.json", cjson(probe))
    info("S2", "probe record written (out/probe.json sha256 %s): t_load %.2f s, t_slice %.2f s, record %d bytes, stage-2 proxy %.1f / %.1f s, "
         "peak RSS %.0f MiB, free %.1f GiB" % (B["probe_sha"], t_load, t_slice, rec_bytes, t2["tier1"], t2["tier1_tier2"],
                                             probe["measured"]["peak_rss_bytes"] / 2 ** 20, free / GIB))
    for k in ("tier1", "tier1_tier2"):
        info("S2", "projection for G, %s: wall %.2f h, new disk %.3f GiB, within the registered limits: %s"
             % (k, proj[k]["T_s"] / 3600.0, proj[k]["D_bytes"] / GIB, proj[k]["within_registered_limits"]))
    if not gate("RES", proj["tier1"]["within_registered_limits"], "Tier 1 projection within the registered limits (12 h, 20 GiB, 64 GiB reserve)"):
        raise Stop("HOLD_RESOURCE", "DIAG_UNC_RESOURCE_LIMIT", "g_tier1_projection: wall %.2f h, disk %d bytes, free %d bytes"
                   % (proj["tier1"]["T_s"] / 3600.0, proj["tier1"]["D_bytes"], free))
    if not gate("RES", f1_s3 <= lim["f1_cap_s"], "F1 planning cap: projected S3 time %.2f h <= %.2f h (not G's ceiling)" % (f1_s3 / 3600.0, lim["f1_cap_s"] / 3600.0)):
        raise Stop("HOLD_RESOURCE", "DIAG_UNC_RESOURCE_LIMIT", "f1_planning_cap: projected S3 time %.2f h" % (f1_s3 / 3600.0))
    return model, probe_role


# ---------------------------------------------------------------- S3: Tier 1 reproduction (fail-fast)
def s3_reproduce(cfg, ctx, B, acc, out, model0, role0):
    C = ctx["C"]
    repro = {"version": VERSION, "selftest": cfg["selftest"], "partial": True, "tolerances": {"r0_psnr_db": TOL_PSNR_DB, "nll_abs_nats_per_dim": TOL_NLL,
             "r1_psnr_db": TOL_PSNR_DB, "r1_nmse_rel": TOL_NMSE_REL, "dc_direct and dc_a": "dc_tolerances.a of the G-U1 record"},
             "note": "absolute deviations |own - registered| and pass flags only; no compared value is stored", "roles": {}}
    timing = {"t_load_s": {}, "t_decode_max_s": {}}
    done = 0

    def flush():
        B["repro_sha"] = out.write("reproduction.json", cjson(repro))
    try:
        t_start = time.perf_counter()
        acc.slice_keys = {(os.path.join(acc.data_dir, os.path.basename(f)), i) for f, i in B["mid"]}
        for ident in B["mid"]:                               # the 95 slices not read by the probe; each slice is read exactly once in the run
            if ident not in B["slice_states"]:
                B["slice_states"][ident], B["t_state"][ident] = build_state(ctx, B, ident)
        B["masks_sha"] = sha_bytes(b"".join(B["slice_states"][x]["mask"][0].numpy().tobytes() for x in B["mid"]))
        gate("POP", len(B["slice_states"]) == len(B["mid"]), "%d midpoint slice states (2 from the probe, %d built now); identity, regenerated mask bytes "
             "and eligibility equal for every slice (sha256 of the concatenated mask bytes, registered order: %s)"
             % (len(B["slice_states"]), len(B["mid"]) - len(PROBES), B["masks_sha"]))
        need("ACC", sorted(acc.reads) == sorted(acc.slice_keys) and len(acc.opens) == 2 * B["n_files"] + len(B["mid"]),
             "slice reads: %d array reads == each midpoint slice once; %d further HDF5 opens" % (len(acc.reads), len(acc.opens) - 2 * B["n_files"]))
        timing["t_state_build_max_s"] = max(B["t_state"].values())
        for si, state in enumerate(B["states"]):
            rs = [r for r in B["roles"] if r["model_state_sha256"] == state]
            model = None
            for r in rs:                                     # every role file is verified by loading it; the state is decoded once
                key = "%s|%d|%s" % (r["arm"], r["seed"], r["role"])
                if r is role0:
                    mdl, timing["t_load_s"][key] = model0, B["probe"]["measured"]["t_load_s"]
                else:
                    mdl, timing["t_load_s"][key] = load_role(ctx, acc, r)
                model = mdl if model is None else model
                del mdl
                repro["roles"][key] = []
            t_dec = 0.0
            for ident in B["mid"]:
                st, elig = B["slice_states"][ident], B["elig"][ident]
                vals, T, _i, _d = slice_values(ctx, model, st, elig)
                del _d
                t_dec = max(t_dec, T["r0_decode"] + T["bank_decode"])
                bad = []
                for r in rs:
                    key = "%s|%d|%s" % (r["arm"], r["seed"], r["role"])
                    s_psnr, s_nll = B["rows"][(r["arm"], r["seed"], r["epoch"])][ident]
                    pe = B["recs"][r["posteval"]]
                    g_psnr, g_nmse, _e = pe["ref"][ident]
                    rec = {"file": ident[0], "slice_index": ident[1], "d_r0_psnr_db": abs(vals["r0_psnr"] - s_psnr),
                           "d_nll_per_dim": abs(vals["nll"] - s_nll), "d_r1_psnr_db": abs(vals["r1_psnr"] - g_psnr),
                           "dc_direct_max": vals["dc_direct"], "dc_a_max": vals["dc_a"], "dc_a_tol": pe["tol_a"]}
                    fails = [n for n, ok in (("r0_psnr", rec["d_r0_psnr_db"] <= TOL_PSNR_DB), ("nll", rec["d_nll_per_dim"] <= TOL_NLL),
                                             ("r1_psnr", rec["d_r1_psnr_db"] <= TOL_PSNR_DB), ("dc_direct", rec["dc_direct_max"] <= pe["tol_a"]),
                                             ("dc_a", rec["dc_a_max"] <= pe["tol_a"])) if not ok]
                    if elig:
                        if g_nmse == 0.0:
                            raise Stop("FAIL", "DIAG_UNC_ZERO_DENOMINATOR", "registered R1 NMSE_u reference is zero at %s|%d" % ident)
                        rec["d_r1_nmse_rel"] = abs(vals["r1_nmse"] - g_nmse) / abs(g_nmse)
                        if not rec["d_r1_nmse_rel"] <= TOL_NMSE_REL:
                            fails.append("r1_nmse")
                    else:
                        rec["d_r1_nmse_rel"] = "not_applicable_ineligible_slice"
                    rec["pass"] = not fails
                    repro["roles"][key].append(rec)
                    if fails:
                        bad.append((key, fails, rec))
                done += 1
                if bad:
                    for key, fails, rec in bad:
                        gate("REPRO", False, "%s %s|%d: outside tolerance: %s (deviations: R0 PSNR %.3e dB, NLL %.3e, R1 PSNR %.3e dB, R1 NMSE rel %s, DC direct %.3e and pinned (a) %.3e vs %.3e)"
                             % (key, ident[0], ident[1], fails, rec["d_r0_psnr_db"], rec["d_nll_per_dim"], rec["d_r1_psnr_db"], rec["d_r1_nmse_rel"],
                                rec["dc_direct_max"], rec["dc_a_max"], rec["dc_a_tol"]))
                    raise Stop("FAIL", "DIAG_UNC_REPRODUCTION_MISMATCH", "first violation at state %d, %s|%d: %s; no further slice or state is processed"
                               % (si, ident[0], ident[1], [(k, f) for k, f, _r in bad]))
            ok = C.model_state_sha256(model) == state and model.training is False
            del model
            timing["t_decode_max_s"][state] = t_dec
            need("IMM", ok, "state %d: model state hash and eval mode unchanged by decoding" % si)
            log.info("[S3] progress: state %d/%d done; roles %s; slices %d/%d; elapsed %.0f s", si + 1, len(B["states"]),
                     ["%s|%d|%s" % (r["arm"], r["seed"], r["role"]) for r in rs], len(B["mid"]), len(B["mid"]), time.perf_counter() - t_start)
        repro["partial"] = False
    finally:
        repro["slice_states_completed"] = done
        try:
            flush()                                          # complete, or partial after a stop (plan A1): written exactly once
        except Exception as exc:
            log.error("cannot write reproduction.json: %r", exc)
            raise
    n = sum(len(v) for v in repro["roles"].values())
    gate("REPRO", n == len(B["roles"]) * len(B["mid"]) and all(x["pass"] for v in repro["roles"].values() for x in v),
         "Tier 1 reproduction: %d role-slice records over %d states x %d slices, all within the 223 §3.3 tolerances (out/reproduction.json sha256 %s)"
         % (n, len(B["states"]), len(B["mid"]), B["repro_sha"]))
    return timing


# ---------------------------------------------------------------- S4: second projection, post-checks, manifest
def s4_close(cfg, pre, ctx, B, acc, out, timing, R):
    lim, n_vol, pm = cfg["limits"], cfg["n_volumes"], B["probe"]["measured"]
    dec_probe = max(p["t_decode_s"] for p in pm["per_slice"])
    non_decode = max(p["t_all_s"] - p["t_decode_s"] for p in pm["per_slice"]) + max(pm["t_state_build_max_s"], timing["t_state_build_max_s"])
    dec_s3, load_s3 = max(timing["t_decode_max_s"].values()), max(timing["t_load_s"].values())
    t_slice2, t2 = non_decode + dec_s3, B["probe"]["proxy_synthetic"]["t_stage2_s"]
    free = int(cfg["free_fn"](out.root))
    proj2 = {"tier1": project(t_slice2, load_s3, t2["tier1"], pm["record_bytes_max"], len(B["states"]), len(B["roles"]), n_vol),
             "tier1_tier2": project(t_slice2, load_s3, t2["tier1_tier2"], pm["record_bytes_max"], B["n_states_both"],
                                    len(B["roles"]) + len(B["tier2"]), n_vol)}
    p1 = B["probe"]["projection_probe"]
    use = {}
    for k in proj2:
        proj2[k]["within_registered_limits"] = within(proj2[k], free, lim)
        larger = proj2[k] if proj2[k]["T_s"] >= p1[k]["T_s"] else p1[k]
        use[k] = {"source": "s3_informed" if larger is proj2[k] else "probe", "T_s": larger["T_s"], "D_bytes": max(proj2[k]["D_bytes"], p1[k]["D_bytes"]),
                  "ratio_s3_to_probe_T": proj2[k]["T_s"] / p1[k]["T_s"],
                  "within_registered_limits": p1[k]["within_registered_limits"] and proj2[k]["within_registered_limits"]}
        info("S4", "projection for G, %s: probe %.2f h, S3-informed %.2f h (ratio %.3f); F2 uses the larger; within the registered limits: %s"
             % (k, p1[k]["T_s"] / 3600.0, proj2[k]["T_s"] / 3600.0, use[k]["ratio_s3_to_probe_T"], use[k]["within_registered_limits"]))
    info("S4", "S3 measurements: slowest decode (bank + R0) %.2f s (probe %.2f s); slowest checkpoint load %.2f s (probe %.2f s); slowest state "
         "build over all slices %.2f s (probe %.2f s)" % (dec_s3, dec_probe, load_s3, pm["t_load_s"], timing["t_state_build_max_s"], pm["t_state_build_max_s"]))
    # ---- post-checks
    closure_gate(cfg, pre, "at end")
    repo = cfg["repo"]
    gate("POST", git(repo, "rev-parse", "HEAD").strip() == cfg["pin"] and not git(repo, "status", "--porcelain").strip(), "HEAD is the pin, status clean")
    V, bad = B["verified"], []
    for path, want in ([(os.path.join(R, "analysis", "v05lt_analysis.json"), V["analysis_sha256"]),
                        (os.path.join(R, "inputs", "v05lt_nmse_eligibility.json"), V["eligibility_sha256"])]
                       + [(os.path.join(R, "posteval", d, "posteval.json"), s_) for d, s_ in V["posteval"].items()]
                       + [(os.path.join(R, "%s_%s" % (k.split("|")[1], k.split("|")[0]), "ledger.jsonl"), s_) for k, s_ in V["ledgers"].items()]
                       + [(os.path.join(R, "%s_%s" % (k.split("|")[1], k.split("|")[0]), "final", "final_weights.json"), s_) for k, s_ in V["final_weights"].items()]
                       + [(os.path.join(R, "%s_%s" % (k.split("|")[1], k.split("|")[0]), "ckpt", "val_%s.json" % k.split("|")[2]), s_)
                          for k, s_ in V["stored_rows"].items()]):
        if sha_file(path) != want:
            bad.append(os.path.relpath(path, R))
    gate("POST", not bad, "registered JSON inputs unchanged since S1 (%d files re-hashed; changed: %s)"
         % (2 + len(V["posteval"]) + len(V["ledgers"]) + len(V["final_weights"]) + len(V["stored_rows"]), bad or "none"))
    want_loads = sorted(os.path.abspath(r["path"]) for r in B["roles"])
    gate("ACC", not acc.forbidden and sorted(acc.loads) == want_loads and sorted(acc.reads) == sorted(acc.slice_keys)
         and len(acc.opens) == 2 * B["n_files"] + n_vol,
         "totals: %d HDF5 opens (2 x %d metadata + %d), %d array reads (each midpoint slice once), %d torch.load (each Tier 1 role file once), "
         "0 numpy.load, %d forbidden" % (len(acc.opens), B["n_files"], n_vol, len(acc.reads), len(acc.loads), len(acc.forbidden)))
    if FAILS:
        raise Stop("FAIL", "DIAG_UNC_INPUT_INVALID", "failed gates: %s" % sorted(set(FAILS)))
    ceil = lim["g_wall_s"]
    manifest = {
        "schema": "seqref-dunc-f1-manifest/1", "version": VERSION, "plan": PLAN, "utc": utc(), "selftest": cfg["selftest"],
        "status": "F1 manifest draft for F2 review; registers nothing; the Tier 2 decision and the stop limit belong to F2",
        "validity": "PENDING -- certifies nothing. Written before the log was closed. See manifest_f1.json and RESULT.txt.",
        "pin": cfg["pin"], "archive_commit": cfg["archive_commit"], "code": pre["ident"], "inputs": V,
        "states": [{"index": i, "model_state_sha256": st, "roles": ["%s|%d|%s" % (r["arm"], r["seed"], r["role"]) for r in B["roles"] if r["model_state_sha256"] == st]}
                   for i, st in enumerate(B["states"])],
        "population": {"n_midpoint": n_vol, "identities": [[f, i] for f, i in B["mid"]], "regenerated_masks_sha256": B["masks_sha"]},
        "decode_counts": {"tier1": {"states": len(B["states"]), "bank_decodes": len(B["states"]) * n_vol * BANK_ROWS, "r0_decodes": len(B["states"]) * n_vol},
                          "tier1_tier2": {"states": B["n_states_both"], "bank_decodes": B["n_states_both"] * n_vol * BANK_ROWS,
                                          "r0_decodes": B["n_states_both"] * n_vol}},
        "reproduction": {"file": "reproduction.json", "sha256": B["repro_sha"], "result": "all role-slice records within the 223 §3.3 tolerances"},
        "probe": {"file": "probe.json", "sha256": B["probe_sha"]},
        "s3_measured": {"t_decode_max_s_by_state": timing["t_decode_max_s"], "t_load_s_by_role": timing["t_load_s"],
                        "t_state_build_max_s": timing["t_state_build_max_s"], "free_bytes_at_end": free},
        "projection": {"cost_model": "T = 1.5 x [W1 + t_stage2 + allowance_time], W1 = t_slice x states x n_slices + t_load x states; "
                                     "D = 1.5 x [R + allowance_disk], R = record_bytes x roles x n_slices; allowances from W1 and R only",
                       "labels": {"t_slice, t_load, record_bytes, free_bytes": "measured", "t_stage2": "proxy_synthetic",
                                  "allowance_time, allowance_disk": "unmeasured_allowance"},
                       "probe": p1, "s3_informed": proj2, "for_f2": use, "registered_limits": lim},
        "proposed_stop_limit": {"tier1_wall_s": min(use["tier1"]["T_s"], ceil), "tier1_tier2_wall_s": min(use["tier1_tier2"]["T_s"], ceil),
                                "note": "proposal only; never above the owner ceiling; registered at F2"},
        "bootstrap_index_matrix": B["boot"], "typed_events": [],
        "exposure": "identities, counts, deviations, pass flags, times, memory and byte counts only"}
    sha = out.write("manifest_f1.pending.json", cjson(manifest))
    info("S4", "pending manifest written (out/manifest_f1.pending.json sha256 %s); the manifest proper is written after this log is closed and "
         "verified; new disk use counted by the write guard so far: %d bytes" % (sha, out.written))
    return manifest


# ---------------------------------------------------------------- orchestration
def checks(cfg, pre):
    """Returns ((result, code, reason), write guard, pending manifest or None). Every stop writes out/stop_record.json with what
    exists; nothing is invented."""
    log.info("%s | plan %s", VERSION, PLAN)
    if cfg["selftest"]:
        log.warning("SELF-TEST MODE -- mock repository, synthetic data, stand-in model -- NOT F1 EVIDENCE")
    log.info("start_utc %s", utc())
    log.info("script %s sha256 %s", pre["script"], pre["script_sha"])
    log.info("HEAD %s (pin, branch %s, clean, nothing staged); pinned closure %d files, working tree == pin == registered", pre["head"], cfg["branch"], len(cfg["closure"]))
    log.info("exports == archive %s == registered: %s", cfg["archive_commit"], {k: v[:16] for k, v in sorted(pre["ident"].items())})
    log.info("registered limits %s; free space at start %d bytes", cfg["limits"], pre["free_start"])
    import h5py
    import numpy as np
    import torch
    log.info("python %s | numpy %s | torch %s | h5py %s | torch threads %d", sys.version.split()[0], np.__version__, torch.__version__,
             h5py.__version__, torch.get_num_threads())
    repo = cfg["repo"]
    R = os.path.join(repo, "seqref_mri", "results", "v05lt")
    sys.path.insert(0, pre["code_dir"])
    sys.path.insert(0, repo)
    data_dir = os.path.join(repo, "seqref_mri", "data", "fastmri", TRAIN_DIR)
    out_dir = os.path.join(pre["run_dir"], "out")
    os.mkdir(out_dir)
    stage, B, out, result = "S0", {"verified": {}}, None, None
    acc = Access(h5py, torch, np, data_dir)
    acc.on()
    try:
        import diag_unc_adapter as A
        import diag_unc_core as core
        import diag_unc_stats as stats
        import diag_unc_guard as guard_mod
        out = guard_mod.WriteGuard(out_dir, ceiling=cfg["limits"]["g_disk_bytes"], reserve=cfg["limits"]["reserve_bytes"],
                                   free_fn=lambda: cfg["free_fn"](out_dir))
        s = A.seams()
        from seqref_mri.scripts import v05lt_metrics as M
        ctx = {"A": A, "C": s.C, "EV": s.EV, "M": M, "torch": torch, "np": np, "core": core, "stats": stats, "fdm": s.fdm}
        closure_gate(cfg, pre, "after import")
        if FAILS:
            raise Stop("FAIL", "DIAG_UNC_INPUT_INVALID", "import closure: failed gates %s" % sorted(set(FAILS)))
        stage = "S1"
        s1_bind(cfg, A, s, R, repo, acc, data_dir, B)
        ctx["bank"] = B["bank"]
        stage = "S2"
        model0, role0 = s2_probe(cfg, ctx, B, acc, out, R)
        stage = "S3"
        timing = s3_reproduce(cfg, ctx, B, acc, out, model0, role0)
        del model0
        stage = "S4"
        B["manifest"] = s4_close(cfg, pre, ctx, B, acc, out, timing, R)
        result = ("PASS", "none", "all stages completed")
    except Stop as exc:
        log.error("F1_STOP at %s: %s", stage, exc)
        result = (exc.result, exc.code, exc.reason)
    except Abort as exc:
        log.error("F1_ABORT at %s: %s", stage, exc)
        result = ("ABORTED", "none", str(exc))
    except Exception as exc:
        code = getattr(exc, "code", None)
        log.error("error at %s: %r", stage, exc)
        log.error("%s", traceback.format_exc())
        if isinstance(code, str) and code.startswith("DIAG_UNC_"):
            result = ("FAIL", code, "%s: %s" % (type(exc).__name__, exc))                   # typed stop of the step-B modules
        elif stage == "S3" and re.search(r"non[-_ ]?finite|\bnan\b|\binf\b", str(exc), re.I):
            result = ("FAIL", "DIAG_UNC_NONFINITE", "%s: %s" % (type(exc).__name__, exc))    # production non-finite failure in a decode
        else:
            result = ("ABORTED", "none", "untyped error at %s: %s: %s" % (stage, type(exc).__name__, exc))
    finally:
        acc.off()
    if result[0] != "PASS":
        rec = {"version": VERSION, "utc": utc(), "selftest": cfg["selftest"], "result": result[0], "typed_code": result[1], "reason": result[2],
               "stage_reached": stage, "failed_gates": sorted(set(FAILS)), "verified_so_far": B.get("verified", {}),
               "files_present": sorted(f for f in os.listdir(out_dir) if not f.startswith("."))}
        if "probe" in B:
            rec["projection_probe"] = B["probe"]["projection_probe"]
        if "states" in B:
            rec["state_count"] = len(B["states"])
        try:
            if out is None:
                raise Abort("write guard not available")
            log.error("stop record written: out/stop_record.json sha256 %s", out.write("stop_record.json", cjson(rec)))
        except Exception as exc:
            log.error("cannot write stop_record.json: %r", exc)
    try:
        import resource
        info("RES", "peak RSS %.0f MiB at the end" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0))
    except Exception as exc:
        log.error("[RES] resource report failed: %r", exc)
    log.info("end_utc %s", utc())
    (log.info if result[0] == "PASS" else log.error)("%s: %s (typed code %s): %s", "F1_SELFTEST_CHECKS" if cfg["selftest"] else "F1_CHECKS", *result)
    return result, out, (B.get("manifest") if result[0] == "PASS" else None)


def fault(cfg, name):
    if cfg.get("fault") == name:
        raise OSError("injected fault: %s (self-test only)" % name)


def run(cfg):
    pre = prechecks(cfg)
    tag = "F1_SELFTEST_RESULT" if cfg["selftest"] else "F1_RESULT"
    run_dir = pre["run_dir"]
    print("F1_OUT=%s" % run_dir)
    log_path, problems = os.path.join(run_dir, "f1.log"), []
    side_path, result_path = log_path + ".sha256", os.path.join(run_dir, "RESULT.txt")
    fh, result, out, pending = None, ("ABORTED", "none", "log file not created"), None, None
    try:
        fault(cfg, "log_open")
        fh = logging.FileHandler(log_path, mode="x", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logging.getLogger().addHandler(fh)                   # root: adapter, step-B and production messages are kept too
    except Exception as exc:
        log.error("cannot create the log file %s: %r", log_path, exc)
        problems.append("log file not created")
    if fh is not None:
        try:
            result, out, pending = checks(cfg, pre)
        except Exception as exc:                             # checks() handles its own errors; this is a last line of defence
            log.error("unexpected error outside the stages: %r", exc)
            log.error("%s", traceback.format_exc())
            result = ("ABORTED", "none", "unexpected error outside the stages")
        finally:
            logging.getLogger().removeHandler(fh)
            fh.close()
    log_sha = None
    if fh is not None:
        try:
            log_sha = sha_file(log_path)
            fault(cfg, "sidecar")
            with open(side_path, "x", encoding="utf-8") as sf:
                sf.write("%s  f1.log\n" % log_sha)
                sf.flush()
                os.fsync(sf.fileno())
            with open(side_path, "r", encoding="utf-8") as sf:
                parts = sf.read().split()
            if parts != [log_sha, "f1.log"] or sha_file(log_path) != log_sha:
                raise OSError("sidecar does not verify against the log")
        except Exception as exc:
            log.error("log sidecar failure: %r", exc)
            problems.append("log sidecar not written or not verified")
    if problems and result[0] in ("PASS", "HOLD_RESOURCE", "HOLD_PROVENANCE"):
        result = ("FAIL", "none", "evidence handling failed")
    manifest_sha = None
    if result[0] == "PASS":                                  # only now: the log is closed and its sidecar verified
        try:
            fault(cfg, "manifest")
            pending.update(validity="valid only together with a RESULT.txt in the run directory whose first line is 'F1_RESULT: PASS' and that names "
                                    "this file's sha256; otherwise it certifies nothing", log_sha256=log_sha, result="PASS")
            manifest_sha = out.write("manifest_f1.json", cjson(pending))
        except Exception as exc:
            log.error("cannot write manifest_f1.json: %r", exc)
            problems.append("manifest_f1.json not written")
            result = ("FAIL", "none", "manifest_f1.json not written")
    final = result[0] if result[0] == "PASS" else "%s (nothing is deleted; do not rerun)" % result[0]
    text = ("%s: %s\n%s\ntyped_code %s\nreason %s\nutc %s\nlog_sha256 %s\n" % (tag, final, VERSION, result[1], result[2], utc(), log_sha)
            + ("manifest_sha256 %s\n" % manifest_sha if result[0] == "PASS" else "") + "".join("problem: %s\n" % p for p in problems))
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
        result = ("FAIL", "none", "RESULT.txt not written")
        if manifest_sha is not None:                         # a manifest exists without its RESULT.txt: mark it, delete nothing
            try:
                out.write("MANIFEST_INVALID.txt", ("manifest_f1.json (sha256 %s) is INVALID: RESULT.txt could not be written or verified; this run is "
                                                   "FAIL. %s %s\n" % (manifest_sha, VERSION, utc())).encode("utf-8"))
                log.error("out/MANIFEST_INVALID.txt written")
            except Exception as exc2:
                log.error("cannot write MANIFEST_INVALID.txt: %r", exc2)
    for p in problems:
        log.error("problem: %s", p)
    (log.info if result[0] == "PASS" else log.error)("%s: %s", tag, final)
    return {"PASS": 0, "HOLD_RESOURCE": 3, "HOLD_PROVENANCE": 3}.get(result[0], 1)


def main():
    sys.dont_write_bytecode = True                           # no __pycache__ is written into the pinned tree
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-commit")
    ap.add_argument("--selftest", metavar="CONFIG.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")
    try:
        cfg = dict(selftest=False, repo=REPO, pin=PIN, branch="seqref-v05lt", archive_commit=a.archive_commit, closure=dict(CLOSURE),
                   split_sha=SPLIT_SHA, analysis_sha=ANALYSIS_SHA, p4_sha=P4_SHA, p0s_sha=P0S_SHA, n_volumes=N_VOLUMES, n_val_slices=N_VAL_SLICES,
                   n_boot=10000, limits=dict(LIMITS), free_fn=lambda p: shutil.disk_usage(p).free, fault=None)
        if a.selftest:
            if a.archive_commit:
                raise Abort("self-test takes its identities from the config file only")
            with open(a.selftest, "r", encoding="utf-8") as fh:
                o = json.load(fh)
            must = {"repo", "pin", "archive_commit", "closure", "split_sha", "analysis_sha", "p4_sha", "p0s_sha", "n_volumes", "n_val_slices", "n_boot"}
            opt = {"fault", "limits", "free_bytes"}
            if set(o) - (must | opt) or not must <= set(o):
                raise Abort("self-test config: keys must be exactly %s (+ optional %s)" % (sorted(must), sorted(opt)))
            if os.path.realpath(o["repo"]) == os.path.realpath(REPO) or o["pin"] == PIN:
                raise Abort("self-test must not use the production repository or pin")
            if o.get("fault") not in (None, "log_open", "sidecar", "manifest", "result"):
                raise Abort("self-test config: unknown fault")
            if set(o.get("limits") or {}) - set(LIMITS):
                raise Abort("self-test config: unknown limit")
            cfg.update({k: o[k] for k in must})
            cfg.update(selftest=True, repo=os.path.abspath(o["repo"]), fault=o.get("fault"), limits=dict(LIMITS, **(o.get("limits") or {})))
            if "free_bytes" in o:
                cfg["free_fn"] = lambda p, v=int(o["free_bytes"]): v
        elif not a.archive_commit:
            raise Abort("production mode needs --archive-commit (run card)")
        rc = run(cfg)
    except Abort as exc:
        log.error("F1_ABORT: %s", exc)
        log.error("F1_RESULT: ABORTED (nothing is deleted; do not rerun)")
        rc = 1
    except Exception as exc:
        log.error("unexpected error outside the run: %r", exc)
        log.error("F1_RESULT: FAIL (nothing is deleted; do not rerun)")
        rc = 1
    print("F1_EXIT=%d" % rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
