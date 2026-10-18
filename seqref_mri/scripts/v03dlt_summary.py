# SEQREF-MRI-V03DLTS v0.2 -- scripts.v03dlt_summary
# LIFETIME: KEEP
# =============================================================================
# Purpose: summary table + locked 8-plot set for SEQREF-MRI-V03DLT v1.4
#          (Article 190, DESIGN FROZEN 2026-10-18 + AM1-AM4; A5, A6, A9) and
#          V03DLTPLAN v1.3 (Article 189, B3a). Reads ONLY the hash-verified
#          V03DLT ledger, train records and session evidence, plus the six
#          registered V03 reference train records and the L1-verified V03
#          ledger. Never loads a checkpoint (bytes are hashed, streamed), never
#          touches the val split, makes no verdict.
# Adapted from SEQREF-MRI-V04DLTS v0.3 (pin 9190735; verified archive
#   a473c4af...). NSF differences: arms {0, 2}; registered lambda_2 =
#   0.6130341651761398; V03 ledger/record format (V03T) and V03E
#   select_checkpoint; record checks add schema, P4 file hash and the
#   import-origin block; the four 190 A5 populations (P-run, P-arm, P-pair,
#   P-repro) replace V04DLTS's single matched-pair population, each output
#   naming its population, seeds and n; derived exhaustion and not-attempted
#   runs reported by cause; reproducibility restricted to P-repro.
# Input binding (B3a): before reading, verifies and records SHA-256 of the
#   V03DLT ledger + sidecar, every V03DLT train record + sidecar, every
#   sidecar/manifest in the campaign root (sha256sum semantics), every
#   ledger snapshot as an entry-level prefix of the live ledger, the six V03
#   reference records and the V03 ledger against their registered hashes,
#   and this script file.
# Required evidence: >= 1 launcher session; per session the log + sidecar,
#   ledger snapshot + sidecar, campaign manifest + sidecar; the latest
#   snapshot covers the whole ledger; every record has its sidecar; every
#   ledger attempt has both run logs (+ sidecars) listed in the manifest of
#   the session whose snapshot first contains it; every train record on disk
#   is in the ledger; every checkpoint referenced by a record exists and its
#   bytes hash to the recorded SHA-256.
# Registered limitation (B3a): an orphan run log (no ledger entry) is a
#   review stop for the operator, not a refusal here; a summary from such a
#   campaign is not final evidence.
# Outputs (fresh directory <campaign root>/summary, never overwritten):
#   v03dlt_summary.json (+sidecar); 8 PNGs (Agg, fixed dpi; host-dependent
#   bytes); plot_manifest.json (+sidecar).
# Fixed rules (A9): trailing rolling mean, window 250 steps, never crossing
#   a run boundary, first 249 steps raw only; no interpolation or
#   extrapolation; incomplete runs plotted to their last recorded point with
#   a terminal marker; SD only for n >= 2; n = 0 outputs omitted and the
#   omission stated; L_total / L_val never compared across arms.
# CONVENTION: logger.error + typed raise (V03DLTSError). PASS(0)/ERROR(2).
#   No fallback, no mock, no silent pass.
# Changelog (v0.2, 2026-10-18; review of package d1a1414b):
#   * Ledger history replayed before settlement (_validate_ledger_history):
#     run/status/error-code domain, 6-run seed-major order, contiguous
#     attempts, <= MAX_ATTEMPTS, no entry after a terminal status.
#   * Record validation: P4 semantic hash + path, finite wall clock, every
#     history step (fields, epoch, lr, finiteness, arm-0 nulls), every
#     validation epoch (finite, L_val identity, checkpoint agreement).
#   * settle() reports the last entry (the only completed one, post-check).
# Changelog (NEW in v0.1, 2026-10-18):
#   * Introduced under V03DLTPLAN v1.3 B3a, adapted from V04DLTS v0.3 as
#     listed above (same evidence binding, record validation, smoothing and
#     plot set; 190 A5 populations).
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
from pathlib import Path

import numpy as np

from seqref_mri.scripts import v03_train as v03t
from seqref_mri.scripts import v03dlt_train as dlt
from seqref_mri.scripts.v02_manifests import canonical_json
from seqref_mri.scripts.v03_eval import V03EError, select_checkpoint

logger = logging.getLogger("seqref_mri.v03dlt_summary")
__version__ = "0.2"
__abbr__ = "SEQREF-MRI-V03DLTS"

SCHEMA = "seqref-v03dlt-summary/1"
STUDY = "SEQREF-MRI-V03DLT v1.4 (Article 190, FROZEN + AM1-AM4)"
N_EPOCHS = dlt.N_EPOCHS_DLT                 # 30 (locked, 190 A3)
STEPS_PER_EPOCH = dlt.STEPS_PER_EPOCH      # 1086
TOTAL_STEPS = dlt.TOTAL_STEPS_DLT          # 32,580
LAMBDA_2 = dlt.REGISTERED_LAMBDA_ARM2      # 0.6130341651761398 (190 A3)
P4_SHA = dlt.REGISTERED_P4_FILE_SHA256
P4_SEMANTIC_SHA = dlt.REGISTERED_P4_SEMANTIC_SHA256
TERMINAL = ("completed", "numerical_failure", "failed_validity")
HIST_KEYS = ("step", "epoch", "L_density", "L_rec", "lam_L_rec", "L_total",
             "grad_norm", "lr")
STATUSES = ("completed", "interrupted", "numerical_failure", "failed_validity")
V03_EPOCHS = 3
V03_STEPS = 3258
SMOOTH_WINDOW = 250
DPI = 120
ARMS, SEEDS = dlt.ARMS_DLT, tuple(v03t.SEEDS)
MONITORS = ("val_NLL", "val_L_rec", "L_val")
STEP_KEYS = ("L_density", "L_total", "L_rec", "lam_L_rec", "grad_norm")
MASK_CONVENTION = ("canonical_mask_seed(TRAIN_BASE_SEED=20261000, relpath, "
                   "slice, epoch); checkpoint epoch k uses mask index k-1")
V03_LEDGER_SHA = "a0f996353daffedb5397b250d9320aabd00e7325c25dd21ff0d75b1f2a4c2256"
V03_RECORD_SHA = {
    (0, 101): "729e15b9e2dc22909d3891a66d33c7e8eb99636b9737b7603ad3646d5d0ce960",
    (0, 202): "a05a30d4299a68080e59bd02b76bc819b413eb36b4f44835bae5d3fb8e025cda",
    (0, 303): "f2ce82acde7265812f244d9f115ef2dccfc051b2f7d9a32734ab387d9f7137e6",
    (2, 101): "e831151a089d8203e800fc8323d81dd905e9dc97fd722915ff33373229971f54",
    (2, 202): "6b89e17f103d9366779fd5409eba7ee9bfb6a84e47d85a6ff48200fee918c8ce",
    (2, 303): "2883e4080278d69dd457eb145ed5062d4b830bade64e1cc074caa49b07569a27",
}
PLOTS = (
    ("plot1_monitor_nll_vs_epoch.png", "Training-split monitor: NLL vs epoch"),
    ("plot2_monitor_rec_vs_epoch.png",
     "Training-split monitor: reconstruction loss vs epoch"),
    ("plot3_monitor_lval_vs_epoch.png", "Training-split monitor: L_val vs epoch"),
    ("plot4_training_loss_vs_step.png", "Training loss vs step"),
    ("plot5_grad_norm_vs_step.png", "Global gradient norm vs step"),
    ("plot6_paired_difference_vs_epoch.png",
     "Training-split monitor: paired arm difference (arm 2 - arm 0) vs epoch"),
    ("plot7_reproducibility_epochs1to3.png",
     "Epoch 1-3 reproducibility: new vs V03 monitor values"),
    ("plot8_wall_clock_by_run.png", "Wall-clock runtime by arm and seed"),
)


class V03DLTSError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V03DLTSError(code, message)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:                # streamed: checkpoints are large
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write(path: Path, payload: bytes) -> str:
    for p in (path, path.with_suffix(path.suffix + ".sha256")):
        if p.exists():
            _fail("ARTIFACT_EXISTS", f"{p} exists; outputs are never overwritten")
    path.write_bytes(payload)
    digest = _sha(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n")
    return digest


# ---------------------------------------------------------------------------
# Input binding (B3a, B4)
# ---------------------------------------------------------------------------

def _check_sidecars(root: Path, register: dict) -> None:
    for side in sorted(root.glob("*.sha256")):
        for ln in side.read_text().splitlines():
            parts = ln.split()
            if len(parts) != 2:
                _fail("SIDECAR_MALFORMED", f"{side.name}: {ln!r}")
            target = root / parts[1]
            if not target.is_file() or _sha(target) != parts[0]:
                _fail("SIDECAR_MISMATCH", f"{side.name} -> {parts[1]}")
        register[f"campaign/{side.name}"] = _sha(side)


def _check_snapshots(root: Path, led: dict) -> None:
    meta = {k: v for k, v in led.items() if k != "entries"}
    for snap in sorted(root.glob("ledger_snapshot_*.json")):
        s = json.loads(snap.read_text())
        n = len(s.get("entries", []))
        if {k: v for k, v in s.items() if k != "entries"} != meta or \
                s["entries"] != led["entries"][:n]:
            _fail("LEDGER_PREFIX_MISMATCH", f"{snap.name}")


def _require_sidecar(root: Path, name: str, code: str) -> None:
    side = root / f"{name}.sha256"
    if not side.is_file():
        _fail(code, f"{name}: sidecar {side.name} missing")
    parts = side.read_text().split()
    if len(parts) != 2 or parts[1] != name:
        _fail("SIDECAR_MALFORMED", f"{side.name} does not name {name}")


def _check_sessions(root: Path, led: dict) -> None:
    """Every launcher session left its complete evidence set; every ledger
    attempt has both run logs tied to the session that first holds it."""
    pre = "launcher_session_"
    sessions = sorted(p.name[len(pre):-len(".log")]
                      for p in root.glob(f"{pre}*.log"))
    if not sessions:
        _fail("NO_SESSION_EVIDENCE", "no launcher_session_<S>.log in the root")
    for s in sessions:
        _require_sidecar(root, f"{pre}{s}.log", "MISSING_SESSION_EVIDENCE")
        snap = f"ledger_snapshot_{s}.json"
        if not (root / snap).is_file():
            _fail("MISSING_SESSION_EVIDENCE", f"session {s}: {snap} missing")
        _require_sidecar(root, snap, "MISSING_SESSION_EVIDENCE")
        if not (root / f"campaign_manifest_{s}.sha256").is_file():
            _fail("MISSING_SESSION_EVIDENCE",
                  f"session {s}: campaign_manifest_{s}.sha256 missing")
        _require_sidecar(root, f"campaign_manifest_{s}.sha256",
                         "MISSING_SESSION_EVIDENCE")
    known = set(sessions)
    for pat, a, b in (("ledger_snapshot_*.json", "ledger_snapshot_", ".json"),
                      ("campaign_manifest_*.sha256", "campaign_manifest_",
                       ".sha256")):
        for p in root.glob(pat):
            if p.name.endswith((".json.sha256", ".sha256.sha256")):
                continue                       # sidecars, not session files
            if p.name[len(a):-len(b)] not in known:
                _fail("ORPHANED_SESSION_EVIDENCE", f"{p.name} has no session log")
    last = json.loads((root / f"ledger_snapshot_{sessions[-1]}.json")
                      .read_text())
    if last.get("entries") != led["entries"]:
        _fail("LEDGER_NOT_SNAPSHOTTED",
              f"latest session {sessions[-1]} snapshot holds "
              f"{len(last.get('entries', []))} of {len(led['entries'])} "
              f"ledger entries")
    for p in root.glob("runlog_*.log"):
        _require_sidecar(root, p.name, "MISSING_RUN_LOG_SIDECAR")
    counts = [len(json.loads((root / f"ledger_snapshot_{s}.json").read_text())
                  ["entries"]) for s in sessions]
    if counts != sorted(counts):
        _fail("LEDGER_PREFIX_MISMATCH",
              f"snapshot entry counts decrease across sessions: {counts}")
    listed = {}
    for i, e in enumerate(led["entries"]):
        s = sessions[next(k for k, n in enumerate(counts) if n > i)]
        if s not in listed:
            listed[s] = {ln.split()[1] for ln in (
                root / f"campaign_manifest_{s}.sha256").read_text()
                .splitlines() if ln.strip()}
        for stream in ("stdout", "stderr"):
            nm = (f"runlog_arm{e['arm']}_seed{e['seed']}_attempt"
                  f"{e['attempt']}_{s}.{stream}.log")
            if not (root / nm).is_file():
                _fail("MISSING_RUN_LOG",
                      f"ledger entry {i + 1}: {nm} missing (session {s})")
            _require_sidecar(root, nm, "MISSING_RUN_LOG_SIDECAR")
            if nm not in listed[s] or f"{nm}.sha256" not in listed[s]:
                _fail("RUN_LOG_NOT_IN_MANIFEST",
                      f"{nm} (+sidecar) not in campaign_manifest_{s}.sha256")


def _validate_ledger_history(led: dict) -> None:
    """Replay the ledger under the trainer's own rules (190 A4): every
    entry is the next runnable run under the 6-run seed-major order, its
    attempt is the next contiguous attempt (<= MAX_ATTEMPTS), and no run
    has an entry after a terminal status. Refuses before any settlement."""
    entries = led.get("entries")
    if not isinstance(entries, list):
        _fail("LEDGER_HISTORY_INVALID", "ledger has no entry list")
    for i, e in enumerate(entries):
        where = f"ledger entry {i + 1}"
        arm, seed = e.get("arm"), e.get("seed")
        if arm not in ARMS or seed not in SEEDS or isinstance(arm, bool) \
                or isinstance(seed, bool):
            _fail("LEDGER_RUN_INVALID", f"{where}: arm {arm!r} seed {seed!r}")
        st, code = e.get("status"), e.get("error_code")
        if st not in STATUSES:
            _fail("LEDGER_STATUS_INVALID", f"{where}: status {st!r}")
        if (st == "completed") != (code is None) or \
                (code is not None and not (isinstance(code, str) and code)):
            _fail("LEDGER_STATUS_INVALID",
                  f"{where}: status {st!r} with error_code {code!r}")
        prior = [x for x in entries[:i]
                 if (x["arm"], x["seed"]) == (arm, seed)]
        if any(x["status"] in TERMINAL for x in prior):
            _fail("LEDGER_RETRY_AFTER_TERMINAL",
                  f"{where}: arm {arm} seed {seed} attempt "
                  f"{e.get('attempt')!r} follows terminal status "
                  f"{[x['status'] for x in prior]}")
        nxt = dlt.expected_next_run_dlt({"entries": entries[:i]})
        if nxt != (arm, seed):
            _fail("LEDGER_ORDER_INVALID",
                  f"{where}: arm {arm} seed {seed}; the next runnable run "
                  f"was {nxt}")
        att = e.get("attempt")
        if isinstance(att, bool) or att != len(prior) + 1:
            _fail("LEDGER_ATTEMPT_INVALID",
                  f"{where}: attempt {att!r}; expected {len(prior) + 1}")
        if att > v03t.MAX_ATTEMPTS:
            _fail("LEDGER_ATTEMPT_INVALID",
                  f"{where}: attempt {att} exceeds the locked maximum "
                  f"{v03t.MAX_ATTEMPTS}")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x)


def _validate_history(name: str, arm: int, hist: list) -> None:
    """Every recorded step carries the locked fields with finite values.
    The trainer stops before appending a non-finite step (V03DLTT), so no
    status, numerical_failure included, may hold a non-finite value."""
    for h in hist:
        st = h.get("step") if isinstance(h, dict) else None
        if not isinstance(h, dict) or set(h) != set(HIST_KEYS):
            _fail("RECORD_HISTORY_INVALID",
                  f"{name}: step {st!r} fields "
                  f"{sorted(h) if isinstance(h, dict) else h!r}")
        if h["epoch"] != (st - 1) // STEPS_PER_EPOCH or \
                h["lr"] != v03t.LEARNING_RATE:
            _fail("RECORD_HISTORY_INVALID",
                  f"{name}: step {st} epoch {h['epoch']!r} lr {h['lr']!r}")
        for k in ("L_density", "L_total", "grad_norm"):
            if not _finite(h[k]):
                _fail("RECORD_HISTORY_INVALID", f"{name}: step {st} {k} "
                      f"{h[k]!r} not a finite number")
        if arm == 0:
            if h["L_rec"] is not None or h["lam_L_rec"] is not None or \
                    h["L_total"] != h["L_density"]:
                _fail("RECORD_HISTORY_INVALID",
                      f"{name}: step {st} arm 0 carries a reconstruction term")
        elif not (_finite(h["L_rec"]) and _finite(h["lam_L_rec"])):
            _fail("RECORD_HISTORY_INVALID",
                  f"{name}: step {st} arm 2 L_rec {h['L_rec']!r} lam_L_rec "
                  f"{h['lam_L_rec']!r}")


def _validate_monitor(name: str, lam: float, vhist: list, cks: list) -> None:
    for v, c in zip(vhist, cks):
        ep = v.get("epoch")
        vals = [v.get(k) for k in MONITORS]
        if not all(_finite(x) for x in vals):
            _fail("RECORD_VALIDATION_INVALID",
                  f"{name}: epoch {ep} monitor values {vals!r}")
        if v["L_val"] != v["val_NLL"] + lam * v["val_L_rec"]:
            _fail("RECORD_VALIDATION_INVALID",
                  f"{name}: epoch {ep} L_val != val_NLL + lambda*val_L_rec")
        if any(c.get(k) != v[k] for k in MONITORS):
            _fail("RECORD_VALIDATION_INVALID",
                  f"{name}: epoch {ep} checkpoint monitor values differ "
                  f"from validation_history")


def _validate_record(e: dict, rec: dict, pin8: str) -> None:
    """The record agrees with its ledger entry, the locked budget, the
    registered lambda and the registered P4 file."""
    name = e["record_file"]
    if rec.get("schema") != dlt.RECORD_SCHEMA_DLT:
        _fail("RECORD_SCHEMA_MISMATCH",
              f"{name}: schema {rec.get('schema')!r} != "
              f"{dlt.RECORD_SCHEMA_DLT!r}")
    if (rec.get("arm"), rec.get("seed"), rec.get("attempt")) != \
            (e["arm"], e["seed"], e["attempt"]):
        _fail("RECORD_IDENTITY_MISMATCH", name)
    for k in ("status", "error_code", "steps_completed", "lambda"):
        if rec.get(k) != e.get(k):
            _fail("RECORD_LEDGER_MISMATCH",
                  f"{name}: {k} record {rec.get(k)!r} != ledger {e.get(k)!r}")
    cks = rec.get("checkpoints") or []
    if [c.get("file_sha256") for c in cks] != e.get("checkpoints"):
        _fail("RECORD_LEDGER_MISMATCH", f"{name}: checkpoint hashes differ")
    if rec["status"] not in STATUSES:
        _fail("RECORD_STATUS_UNKNOWN", f"{name}: {rec['status']!r}")
    want_lam = 0.0 if e["arm"] == 0 else LAMBDA_2
    if e["arm"] not in ARMS or e["seed"] not in SEEDS or \
            rec["lambda"] != want_lam:
        _fail("RECORD_LAMBDA_MISMATCH",
              f"{name}: arm {e['arm']} lambda {rec['lambda']!r} != {want_lam!r}")
    pin = rec.get("commit_pin")
    if not isinstance(pin, str) or len(pin) != 40 or pin[:8] != pin8:
        _fail("RECORD_PIN_MISMATCH", f"{name}: commit_pin {pin!r} vs root {pin8}")
    p4 = rec.get("p4_stats2") or {}
    if p4.get("file_sha256") != P4_SHA:
        _fail("RECORD_P4_MISMATCH",
              f"{name}: P4 file hash {p4.get('file_sha256')!r} != {P4_SHA}")
    if p4.get("semantic_sha256") != P4_SEMANTIC_SHA:
        _fail("RECORD_P4_MISMATCH",
              f"{name}: P4 semantic hash {p4.get('semantic_sha256')!r} != "
              f"{P4_SEMANTIC_SHA}")
    if not isinstance(p4.get("path"), str) or not p4["path"]:
        _fail("RECORD_P4_MISMATCH", f"{name}: P4 path {p4.get('path')!r}")
    wc = rec.get("wall_clock_s")
    if not _finite(wc) or wc < 0:
        _fail("RECORD_WALL_CLOCK_INVALID", f"{name}: wall_clock_s {wc!r}")
    cfg = rec.get("config") or {}
    if cfg.get("epochs") != N_EPOCHS or cfg.get("steps") != TOTAL_STEPS:
        _fail("RECORD_CONFIG_MISMATCH",
              f"{name}: config epochs/steps {cfg.get('epochs')}/"
              f"{cfg.get('steps')} != {N_EPOCHS}/{TOTAL_STEPS}")
    k = len(cks)
    for i, c in enumerate(cks, start=1):
        if (c.get("epoch"), c.get("file"), c.get("arm"), c.get("seed"),
                c.get("attempt"), c.get("lambda")) != \
                (i, v03t.checkpoint_name(e["arm"], e["seed"], e["attempt"], i),
                 e["arm"], e["seed"], e["attempt"], want_lam):
            _fail("RECORD_CHECKPOINT_INVALID", f"{name}: checkpoint {i}")
    if [int(v.get("epoch", -1)) for v in rec.get("validation_history", [])] \
            != list(range(1, k + 1)):
        _fail("RECORD_COVERAGE_INVALID",
              f"{name}: validation epochs != checkpoint epochs 1..{k}")
    hist = rec.get("history", [])
    if not isinstance(hist, list) or \
            [h.get("step") if isinstance(h, dict) else None for h in hist] \
            != list(range(1, len(hist) + 1)):
        _fail("RECORD_COVERAGE_INVALID", f"{name}: history steps not 1..n")
    _validate_history(name, e["arm"], hist)
    _validate_monitor(name, want_lam, rec["validation_history"], cks)
    steps = rec["steps_completed"]
    if rec["status"] == "completed":
        if steps != TOTAL_STEPS or k != N_EPOCHS or len(hist) != TOTAL_STEPS:
            _fail("RECORD_COVERAGE_INVALID",
                  f"{name}: completed with {steps} steps, {k} checkpoints, "
                  f"{len(hist)} history steps (locked {TOTAL_STEPS}/"
                  f"{N_EPOCHS}/{TOTAL_STEPS})")
    else:
        if not isinstance(steps, int) or not 0 <= steps <= TOTAL_STEPS or \
                k > steps // STEPS_PER_EPOCH or \
                not steps - 1 <= len(hist) <= steps:
            _fail("RECORD_COVERAGE_INVALID",
                  f"{name}: {rec['status']} with {steps} steps, {k} "
                  f"checkpoints, {len(hist)} history steps")


def load_inputs(root: Path, v03_train_dir: Path) -> dict:
    register = {"script/v03dlt_summary.py": _sha(Path(__file__).resolve())}
    lp = root / dlt.LEDGER_NAME
    if not lp.is_file():
        _fail("LEDGER_MISSING", f"{lp}")
    led = v03t.load_ledger(lp)                 # sidecar-verified (V03T)
    register[f"campaign/{dlt.LEDGER_NAME}"] = _sha(lp)
    _validate_ledger_history(led)              # v0.2: before anything else
    _check_sidecars(root, register)
    _check_snapshots(root, led)
    _check_sessions(root, led)
    pin8 = root.name[len(dlt.CAMPAIGN_PREFIX):]
    pins, records = set(), {}
    listed = {e["record_file"] for e in led["entries"]}
    for p in root.glob("*_train_record.json"):
        if p.name not in listed:
            _fail("UNREGISTERED_RECORD", f"{p.name} is not in the ledger")
    for e in led["entries"]:
        rp = root / e["record_file"]
        if not rp.is_file() or _sha(rp) != e["record_sha256"]:
            _fail("RECORD_HASH_MISMATCH", f"{e['record_file']}")
        _require_sidecar(root, rp.name, "RECORD_SIDECAR_MISSING")
        rec = json.loads(rp.read_text())
        _validate_record(e, rec, pin8)
        pins.add(rec["commit_pin"])
        for c in rec["checkpoints"]:
            cp = root / c["file"]
            if not cp.is_file():
                _fail("CHECKPOINT_MISSING", c["file"])
            if _sha(cp) != c["file_sha256"]:
                _fail("CHECKPOINT_HASH_MISMATCH", c["file"])
            register[f"checkpoint/{c['file']}"] = c["file_sha256"]
        records[(e["arm"], e["seed"], e["attempt"])] = rec
        register[f"campaign/{rp.name}"] = e["record_sha256"]
    if len(pins) > 1:
        _fail("RECORD_PIN_MISMATCH", f"records carry several pins {sorted(pins)}")
    v03_led = v03_train_dir / "v03_run_ledger.json"
    if not v03_led.is_file() or _sha(v03_led) != V03_LEDGER_SHA:
        _fail("V03_EVIDENCE_CHANGED", "V03 ledger hash")
    register["v03/v03_run_ledger.json"] = V03_LEDGER_SHA
    v03 = {}
    for (arm, seed), want in V03_RECORD_SHA.items():
        p = v03_train_dir / v03t.run_record_name(arm, seed, 1)
        if not p.is_file() or _sha(p) != want:
            _fail("V03_EVIDENCE_CHANGED", f"{p.name}")
        v03[(arm, seed)] = json.loads(p.read_text())
        register[f"v03/{p.name}"] = want
    return {"ledger": led, "records": records, "v03": v03,
            "register": register}


# ---------------------------------------------------------------------------
# Settlement, populations (190 A5), per-run table, arm / paired summaries
# ---------------------------------------------------------------------------

def settle(led: dict) -> dict:
    """Per run: final status (incl. derived exhaustion), attempts used, the
    attempt whose record is reported, and the cause for incomplete runs.
    Requires a ledger already passed by _validate_ledger_history, so a
    terminal entry (completed included) is always the run's last entry."""
    out = {}
    for seed, arm in dlt.ORDER_DLT:
        es = [e for e in led["entries"]
              if e["arm"] == arm and e["seed"] == seed]
        if not es:
            out[(arm, seed)] = {"status": "not_attempted", "attempts": 0,
                                "attempt": None, "cause": "not_attempted"}
            continue
        last = es[-1]
        if last["status"] == "completed":
            status, cause = "completed", None
        elif last["status"] in ("numerical_failure", "failed_validity"):
            status = cause = last["status"]
        elif len(es) >= v03t.MAX_ATTEMPTS:
            status = cause = "incomplete_attempts_exhausted"
        else:
            status, cause = "interrupted", "interrupted_attempts_remain"
        out[(arm, seed)] = {"status": status, "attempts": len(es),
                            "attempt": last["attempt"], "cause": cause}
    return out


def _vh(rec: dict) -> dict:
    return {int(v["epoch"]): v for v in rec["validation_history"]}


def _val(vh: dict, ep: int, key: str):
    v = vh.get(ep, {}).get(key)
    return None if v is None else float(v)


def _delta(vh, a, b, key):
    x, y = _val(vh, a, key), _val(vh, b, key)
    return None if x is None or y is None else y - x


def populations(inp: dict, st: dict) -> dict:
    """The four locked 190 A5 populations, each with members and n."""
    def rec(a, s):
        return inp["records"][(a, s, st[(a, s)]["attempt"])]
    run = [(a, s) for s, a in dlt.ORDER_DLT if st[(a, s)]["attempt"] is not None
           and len(rec(a, s)["history"]) >= 1]
    arm = {a: [s for s in SEEDS if st[(a, s)]["status"] == "completed"]
           for a in ARMS}
    pair = [s for s in SEEDS if all(st[(a, s)]["status"] == "completed"
                                    for a in ARMS)]
    repro = [(a, s) for a, s in run
             if rec(a, s)["steps_completed"] >= V03_STEPS
             and len(rec(a, s)["validation_history"]) >= V03_EPOCHS]
    return {
        "P-run": {"definition": "every run with at least one recorded step "
                                "(reported attempt), complete or incomplete",
                  "runs": [f"arm{a}_seed{s}" for a, s in run], "n": len(run)},
        "P-arm0": {"definition": "arm-0 runs with status completed",
                   "seeds": arm[0], "n": len(arm[0])},
        "P-arm2": {"definition": "arm-2 runs with status completed",
                   "seeds": arm[2], "n": len(arm[2])},
        "P-pair": {"definition": "seeds whose arm-0 and arm-2 runs both "
                                 "completed", "seeds": pair, "n": len(pair)},
        "P-repro": {"definition": "runs whose reported attempt recorded at "
                                  "least 3 epochs (3,258 steps), each "
                                  "compared with its own V03 reference",
                    "runs": [f"arm{a}_seed{s}" for a, s in repro],
                    "n": len(repro)},
        "_members": {"run": run, "arm": arm, "pair": pair, "repro": repro},
    }


def run_table(inp: dict, st: dict) -> list:
    rows = []
    for seed, arm in dlt.ORDER_DLT:
        s = st[(arm, seed)]
        row = {"arm": arm, "seed": seed, "status": s["status"],
               "attempts": s["attempts"], "incomplete_cause": s["cause"]}
        if s["attempt"] is not None:
            rec = inp["records"][(arm, seed, s["attempt"])]
            vh = _vh(rec)
            env = rec.get("environment", {})
            row.update({
                "attempt": s["attempt"], "lambda": rec["lambda"],
                "init_seed": rec["config"].get("init_seed"),
                "steps_completed": rec["steps_completed"],
                "error_code": rec["error_code"],
                "wall_clock_s": rec["wall_clock_s"],
                "last_recorded_epoch": max(vh) if vh else None,
                "commit_pin": rec["commit_pin"],
                "p4_file_sha256": rec["p4_stats2"]["file_sha256"],
                "environment": {k: env.get(k) for k in
                                ("python", "torch", "platform", "cpu_model",
                                 "threads")},
                "import_origin": rec.get("import_origin"),
                "checkpoint_sha256_by_epoch": {
                    str(c["epoch"]): c["file_sha256"]
                    for c in rec["checkpoints"]}})
            for k in MONITORS:
                row[f"{k}_by_epoch"] = [_val(vh, e, k) for e in sorted(vh)]
                if s["status"] == "completed":
                    row[f"{k}_epoch3"] = _val(vh, 3, k)
                    row[f"{k}_epoch27"] = _val(vh, 27, k)
                    row[f"{k}_epoch30"] = _val(vh, N_EPOCHS, k)
                    row[f"{k}_change_3_to_30"] = _delta(vh, 3, N_EPOCHS, k)
                    row[f"{k}_change_27_to_30"] = _delta(vh, 27, N_EPOCHS, k)
            if s["status"] == "completed":
                sel = select_checkpoint(rec["validation_history"],
                                        rec["checkpoints"])
                row["selected_epoch"] = sel["epoch"]
                row["selected_file_sha256"] = sel["file_sha256"]
        rows.append(row)
    return rows


def _mean_sd(xs: list) -> dict:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None,
                "note": "omitted (n = 0)"}
    sd = float(np.std(xs, ddof=1)) if n >= 2 else None
    return {"n": n, "mean": float(np.mean(xs)), "sd": sd,
            "sd_note": None if n >= 2 else "not available (n < 2)"}


def arm_summaries(inp: dict, st: dict, pops: dict) -> dict:
    """P-arm: per-arm summaries over that arm's completed runs."""
    out = {}
    for a in ARMS:
        seeds = pops["_members"]["arm"][a]
        vhs = [_vh(inp["records"][(a, s, st[(a, s)]["attempt"])])
               for s in seeds]
        d = {"population": f"P-arm{a}", "seeds": seeds, "n": len(seeds)}
        for k in MONITORS:
            d[f"{k}_epoch30"] = _mean_sd([_val(v, N_EPOCHS, k) for v in vhs])
            d[f"{k}_change_3_to_30"] = _mean_sd(
                [_delta(v, 3, N_EPOCHS, k) for v in vhs])
            d[f"{k}_change_27_to_30"] = _mean_sd(
                [_delta(v, 27, N_EPOCHS, k) for v in vhs])
        d["selected_epoch"] = _mean_sd([
            select_checkpoint(inp["records"][(a, s, st[(a, s)]["attempt"])]
                              ["validation_history"],
                              inp["records"][(a, s, st[(a, s)]["attempt"])]
                              ["checkpoints"])["epoch"] for s in seeds])
        if not seeds:
            d["note"] = "n = 0: arm summary omitted"
        out[f"arm{a}"] = d
    return out


def paired(inp: dict, st: dict, pops: dict) -> dict:
    """P-pair: arm 2 - arm 0 per seed and cross-seed (NLL and rec only)."""
    pairs = pops["_members"]["pair"]
    per_seed = {}
    for s in pairs:
        v0 = _vh(inp["records"][(0, s, st[(0, s)]["attempt"])])
        v2 = _vh(inp["records"][(2, s, st[(2, s)]["attempt"])])
        per_seed[str(s)] = {k: [None if _val(v2, e, k) is None or
                                _val(v0, e, k) is None else
                                _val(v2, e, k) - _val(v0, e, k)
                                for e in range(1, N_EPOCHS + 1)]
                            for k in ("val_NLL", "val_L_rec")}
    out = {"population": "P-pair", "seeds": pairs, "n": len(pairs),
           "per_seed_difference_by_epoch": per_seed,
           "comparability_rule": "NLL and reconstruction compared across "
                                 "arms; L_total and L_val include lambda and "
                                 "are not compared across arms"}
    if not pairs:
        out["note"] = "n = 0: paired outputs omitted"
        return out
    out["cross_seed_epoch30"] = {
        f"difference_{k}_epoch30": _mean_sd(
            [per_seed[str(s)][k][N_EPOCHS - 1] for s in pairs])
        for k in ("val_NLL", "val_L_rec")}
    return out


def _diff_stats(a: list, b: list) -> dict:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return {"n": 0, "exact": None, "max_abs": None, "max_rel": None}
    x = np.array([p[0] for p in pairs], dtype=np.float64)
    y = np.array([p[1] for p in pairs], dtype=np.float64)
    d = np.abs(x - y)
    rel = d / np.maximum(np.abs(y), 1e-300)
    return {"n": len(pairs), "exact": bool(np.all(x == y)),
            "max_abs": float(d.max()), "max_rel": float(rel.max())}


def reproducibility(inp: dict, st: dict, pops: dict) -> dict:
    members = pops["_members"]["repro"]
    out = {"population": "P-repro", "n": len(members),
           "runs_compared": [f"arm{a}_seed{s}" for a, s in members],
           "rule": "check, not a promise; non-gating; V03-recorded fields "
                   "only; no change may be made to force agreement",
           "runs": {}}
    for a, s in members:
        rec = inp["records"][(a, s, st[(a, s)]["attempt"])]
        ref = inp["v03"][(a, s)]
        va, vb = _vh(rec), _vh(ref)
        mon = {k: _diff_stats([_val(va, e, k) for e in range(1, 4)],
                              [_val(vb, e, k) for e in range(1, 4)])
               for k in MONITORS}
        ha = {h["step"]: h for h in rec["history"] if h["step"] <= V03_STEPS}
        hb = {h["step"]: h for h in ref["history"]}
        steps = sorted(set(ha) & set(hb))
        stp = {k: _diff_stats([ha[t].get(k) for t in steps],
                              [hb[t].get(k) for t in steps])
               for k in STEP_KEYS}
        en, eo = rec.get("environment", {}), ref.get("environment", {})
        out["runs"][f"arm{a}_seed{s}"] = {
            "epochs_1_3_monitor": mon, "steps_compared": len(steps),
            "per_step": stp,
            "environment": {k: {"new": en.get(k), "v03": eo.get(k),
                                "equal": en.get(k) == eo.get(k)}
                            for k in ("python", "torch", "platform")},
            "not_recorded_by_v03": {
                "threads": en.get("threads"), "cpu_model": en.get("cpu_model"),
                "commit_pin": rec["commit_pin"],
                "p4_file_sha256": rec["p4_stats2"]["file_sha256"],
                "v03": "unknown (not recorded by V03)"}}
    if not members:
        out["note"] = "n = 0: reproducibility comparison omitted"
    return out


def incomplete_runs(st: dict) -> list:
    return [{"arm": a, "seed": s, "cause": st[(a, s)]["cause"],
             "attempts": st[(a, s)]["attempts"]}
            for s, a in dlt.ORDER_DLT if st[(a, s)]["status"] != "completed"]


# ---------------------------------------------------------------------------
# Plots (A9): fixed set, Agg backend, fixed dpi
# ---------------------------------------------------------------------------

def trailing_mean(values: list, window: int = SMOOTH_WINDOW) -> list:
    """Trailing rolling mean within ONE run; None for the first window-1
    steps (shown raw only) and wherever the window holds a gap."""
    out = [None] * len(values)
    for i in range(window - 1, len(values)):
        w = values[i - window + 1:i + 1]
        if all(v is not None for v in w):
            out[i] = float(np.mean(w))
    return out


def _caption(*parts: str) -> str:
    return "Population: " + "; ".join(parts)


def _plots(out: Path, inp: dict, st: dict, pops: dict, pr: dict,
           repro: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    color = {0: "#1f77b4", 2: "#d62728"}
    style = {101: "-", 202: "--", 303: ":"}
    m = pops["_members"]
    recs = {(a, s): inp["records"][(a, s, st[(a, s)]["attempt"])]
            for a, s in m["run"]}
    run_txt = (f"traces P-run (n={pops['P-run']['n']}: "
               f"{', '.join(pops['P-run']['runs']) or 'none'})")
    files = {}

    def save(fig, name, title, caption):
        fig.suptitle(title, fontsize=10)
        fig.text(0.01, 0.005, caption, fontsize=6, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        path = out / name
        if path.exists():
            _fail("ARTIFACT_EXISTS", str(path))
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        files[name] = _sha(path)

    def epoch_curves(ax, key, arms, band=True):
        for a in arms:
            for s in SEEDS:
                if (a, s) not in recs:
                    continue
                vh = _vh(recs[(a, s)])
                eps = sorted(vh)
                ys = [vh[e][key] for e in eps]
                ax.plot(eps, ys, style[s], color=color[a], alpha=0.5,
                        label=f"arm {a} seed {s}")
                if st[(a, s)]["status"] != "completed" and eps:
                    ax.plot(eps[-1], ys[-1], "x", color=color[a],
                            markersize=9, label=f"arm {a} seed {s} incomplete")
            if not band:
                continue
            seeds = m["arm"][a]
            n = len(seeds)
            if n == 0:
                ax.text(0.02, 0.02 + 0.06 * (a // 2), f"arm {a}: band omitted "
                        f"(P-arm{a} n = 0)", transform=ax.transAxes, fontsize=7)
                continue
            series = [_vh(recs[(a, s)]) for s in seeds]
            eps = list(range(1, N_EPOCHS + 1))
            mean = [float(np.mean([v[e][key] for v in series])) for e in eps]
            ax.plot(eps, mean, "-", color=color[a], linewidth=2.2,
                    label=f"arm {a} mean (P-arm{a}, n={n}, seeds {seeds})")
            if n >= 2:
                sd = [float(np.std([v[e][key] for v in series], ddof=1))
                      for e in eps]
                ax.fill_between(eps, np.array(mean) - sd, np.array(mean) + sd,
                                color=color[a], alpha=0.15)
            else:
                ax.text(0.02, 0.02 + 0.06 * (a // 2),
                        f"arm {a}: SD not available (n < 2)",
                        transform=ax.transAxes, fontsize=7)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.legend(fontsize=6)

    band_txt = ", ".join(f"band P-arm{a} (n={pops[f'P-arm{a}']['n']}, seeds "
                         f"{pops[f'P-arm{a}']['seeds']})" for a in ARMS)
    for idx, key in ((0, "val_NLL"), (1, "val_L_rec")):
        fig, ax = plt.subplots(figsize=(8, 5))
        epoch_curves(ax, key, ARMS)
        save(fig, *PLOTS[idx], _caption(run_txt, band_txt))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for a, ax in zip(ARMS, axs):
        epoch_curves(ax, "L_val", (a,), band=False)
        for s in SEEDS:
            if (a, s) in recs and st[(a, s)]["status"] == "completed":
                sel = select_checkpoint(recs[(a, s)]["validation_history"],
                                        recs[(a, s)]["checkpoints"])
                ax.axvline(sel["epoch"], color=color[a], linestyle=style[s],
                           alpha=0.4)
        ax.set_title(f"arm {a} (selected epochs marked; completed runs)",
                     fontsize=8)
    save(fig, *PLOTS[2], _caption(run_txt))
    for idx, logy in ((3, False), (4, True)):
        panels = ([("L_density", ARMS, "NLL (both arms)")] +
                  [("L_total", (a,), f"L_total arm {a} (not compared "
                                     f"across arms)") for a in ARMS]) \
            if idx == 3 else [("grad_norm", ARMS, "global gradient norm")]
        fig, axs = plt.subplots(1, len(panels),
                                figsize=(15 if idx == 3 else 8, 5))
        axs = np.atleast_1d(axs)
        for ax, (key, arms, ttl) in zip(axs, panels):
            for a in arms:
                for s in SEEDS:
                    if (a, s) not in recs:
                        continue
                    h = recs[(a, s)]["history"]
                    xs = [r["step"] for r in h]
                    ys = [r[key] for r in h]
                    ax.plot(xs, ys, style[s], color=color[a], alpha=0.15,
                            linewidth=0.5)
                    sm = trailing_mean(ys)
                    ax.plot([x for x, v in zip(xs, sm) if v is not None],
                            [v for v in sm if v is not None], style[s],
                            color=color[a], label=f"arm {a} seed {s}")
                    if st[(a, s)]["status"] != "completed" and xs:
                        ax.plot(xs[-1], ys[-1], "x", color=color[a])
            if logy:
                ax.set_yscale("log")
            ax.set_title(ttl, fontsize=8)
            ax.set_xlabel("step")
            ax.legend(fontsize=6)
        save(fig, *PLOTS[idx], _caption(
            run_txt, f"trailing mean window {SMOOTH_WINDOW} steps"))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key in zip(axs, ("val_NLL", "val_L_rec")):
        for s, d in pr["per_seed_difference_by_epoch"].items():
            eps = [e + 1 for e, v in enumerate(d[key]) if v is not None]
            ax.plot(eps, [v for v in d[key] if v is not None],
                    style[int(s)], color="k", label=f"seed {s}")
        ax.axhline(0.0, color="grey", linewidth=0.8)
        if not pr["seeds"]:
            ax.text(0.02, 0.5, "omitted: P-pair n = 0", transform=ax.transAxes,
                    fontsize=8)
        ax.set_title(f"{key}: arm 2 - arm 0", fontsize=8)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=6)
    save(fig, *PLOTS[5], _caption(
        f"P-pair (n={pr['n']}, seeds {pr['seeds']})"))
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, key in zip(axs, MONITORS):
        for a, s in m["repro"]:
            new, old = _vh(recs[(a, s)]), _vh(inp["v03"][(a, s)])
            eps = [e for e in (1, 2, 3) if e in new and e in old]
            ax.plot(eps, [old[e][key] for e in eps], "o" + style[s],
                    color=color[a], alpha=0.4, label=f"V03 arm {a} seed {s}")
            ax.plot(eps, [new[e][key] for e in eps], "x" + style[s],
                    color=color[a], label=f"new arm {a} seed {s}")
            r = repro["runs"][f"arm{a}_seed{s}"]["epochs_1_3_monitor"][key]
            if r["max_abs"] is not None and eps:
                ax.annotate(f"max|d|={r['max_abs']:.2e}",
                            (eps[-1], new[eps[-1]][key]), fontsize=6)
        if not m["repro"]:
            ax.text(0.02, 0.5, "omitted: P-repro n = 0",
                    transform=ax.transAxes, fontsize=8)
        ax.set_title(key, fontsize=8)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=5)
    save(fig, *PLOTS[6], _caption(
        f"P-repro (n={repro['n']}: {', '.join(repro['runs_compared']) or 'none'})"))
    fig, ax = plt.subplots(figsize=(8, 5))
    labels, vals, cols = [], [], []
    for a, s in m["run"]:
        labels.append(f"a{a}\ns{s}")
        vals.append(float(recs[(a, s)]["wall_clock_s"]) / 3600.0)
        cols.append(color[a])
    ax.bar(range(len(vals)), vals, color=cols)
    ax.set_xticks(range(len(vals)), labels, fontsize=7)
    ax.set_ylabel("wall clock (h, host-dependent)")
    save(fig, *PLOTS[7], _caption(run_txt))
    return files


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build(root: Path, v03_train_dir: Path) -> int:
    if not root.is_dir() or not root.name.startswith(dlt.CAMPAIGN_PREFIX):
        _fail("CAMPAIGN_ROOT_INVALID", str(root))
    out = root / "summary"
    if out.exists():
        _fail("ARTIFACT_EXISTS", f"{out} exists; the summary is never rebuilt "
              f"in place")
    inp = load_inputs(root, v03_train_dir)
    st = settle(inp["ledger"])
    pops = populations(inp, st)
    table = run_table(inp, st)
    arms = arm_summaries(inp, st, pops)
    pr = paired(inp, st, pops)
    repro = reproducibility(inp, st, pops)
    out.mkdir()
    files = _plots(out, inp, st, pops, pr, repro)
    import matplotlib
    pop_public = {k: v for k, v in pops.items() if not k.startswith("_")}
    summary = {"schema": SCHEMA, "script": f"{__abbr__} v{__version__}",
               "study": STUDY,
               "readouts": "descriptive training-split monitor values only; "
                           "no thresholds, no verdicts, no holdout",
               "counts": {"n_run": pops["P-run"]["n"],
                          "n_arm0": pops["P-arm0"]["n"],
                          "n_arm2": pops["P-arm2"]["n"],
                          "n_pair": pops["P-pair"]["n"],
                          "n_repro": pops["P-repro"]["n"]},
               "populations": pop_public,
               "incomplete_runs": incomplete_runs(st),
               "run_table": table, "arm_summaries": arms, "paired": pr,
               "reproducibility": repro,
               "mask_convention": MASK_CONVENTION,
               "input_hash_register": inp["register"],
               "environment": {"python": platform.python_version(),
                               "matplotlib": matplotlib.__version__,
                               "numpy": np.__version__}}
    s_sha = _write(out / "v03dlt_summary.json", canonical_json(summary))
    manifest = {"schema": "seqref-v03dlt-plot-manifest/1",
                "generator": {"script": f"{__abbr__} v{__version__}",
                              "sha256": inp["register"][
                                  "script/v03dlt_summary.py"],
                              "backend": "Agg", "dpi": DPI,
                              "smoothing_window": SMOOTH_WINDOW},
                "plots": [{"file": n, "title": t, "sha256": files[n]}
                          for n, t in PLOTS],
                "summary_sha256": s_sha,
                "input_hash_register": inp["register"],
                "note": "PNG bytes are host-dependent; only the host-generated "
                        "set is registered"}
    _write(out / "plot_manifest.json", canonical_json(manifest))
    c = summary["counts"]
    print(f"PASS {__abbr__} v{__version__}: summary {s_sha[:12]}..., "
          f"{len(files)} plots, n_run={c['n_run']} n_arm0={c['n_arm0']} "
          f"n_arm2={c['n_arm2']} n_pair={c['n_pair']} n_repro={c['n_repro']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="v03dlt_summary")
    ap.add_argument("--campaign-root", required=True)
    ap.add_argument("--v03-train-dir", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return build(Path(a.campaign_root).resolve(),
                     Path(a.v03_train_dir).resolve())
    except (V03DLTSError, v03t.V03TError, V03EError) as exc:
        logger.error("[%s] ERROR: %s", __abbr__, exc)
        return 2
    except Exception:
        logger.exception("[%s] unexpected exception", __abbr__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
