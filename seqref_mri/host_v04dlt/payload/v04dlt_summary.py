# =============================================================================
# SEQREF-MRI-V04DLTS v0.3 -- scripts.v04dlt_summary
# LIFETIME: KEEP
# Purpose: summary table + locked 8-plot set for SEQREF-MRI-V04DLT v1.0
#          (Article 184, DESIGN FROZEN 2026-10-17; A5, A6, A9, B3a).
#          Reads ONLY the hash-verified V04DLT ledger and train records and
#          the six registered V04 train records + V04 ledger. Never loads a
#          checkpoint, never touches the val split, makes no verdict.
# Input binding (B3a): before reading, verifies and records SHA-256 of the
#   V04DLT ledger + sidecar, every V04DLT train record + sidecar, every
#   sidecar/manifest in the campaign root (sha256sum semantics), every
#   ledger snapshot as an entry-level prefix of the live ledger (B4), the six
#   V04 train records and the V04 ledger against their registered hashes,
#   and this script file.
# Outputs (fresh directory <campaign root>/summary, never overwritten):
#   v04dlt_summary.json (+sidecar): A5 table, A6 selections, paired and
#   cross-seed summaries over completed matched pairs (n stated; SD null for
#   n < 2; no paired summary for n = 0), the reproducibility comparison and
#   the input-hash register; 8 PNGs (Agg, fixed dpi; host-dependent bytes);
#   plot_manifest.json (+sidecar).
# Fixed rules (A9): trailing rolling mean, window 250 steps, never crossing
#   a run boundary, first 249 steps raw only; no interpolation or
#   extrapolation; incomplete runs plotted to their last recorded point with
#   a terminal marker; arm statistics over completed runs, paired statistics
#   over completed matched pairs; L_total / L_val never compared across arms.
# CONVENTION: logger.error + typed raise (V04DLTSError). PASS(0)/ERROR(2).
#   No fallback, no mock, no silent pass.
# Required evidence (v0.2): >= 1 launcher session; per session the log +
#   sidecar, ledger snapshot + sidecar, campaign manifest + sidecar; the latest
#   snapshot covers the whole ledger; every record has its sidecar; every
#   run log has its sidecar; every ledger attempt has BOTH run logs (+
#   sidecars) named for, and listed in the manifest of, the session whose
#   snapshot first contains it; every train record on disk is in the ledger;
#   every checkpoint referenced by a record exists and its BYTES hash to the
#   recorded SHA-256 (streamed; tensors never loaded).
# Record consistency (v0.2): record == ledger on identity, status,
#   error_code, lambda, steps and checkpoint hashes; lambda = 0 (arm 0) or
#   the registered lambda_2' (arm 1); one commit pin matching the root;
#   config epochs/steps = 30 / 32,580; checkpoint names/identities and
#   validation epochs contiguous from 1; completed = 30 epochs, 32,580
#   steps and 32,580 history steps; incomplete <= 32,580 steps with
#   checkpoints <= steps // 1086 and history n or n - 1 steps.
# Changelog (NEW in v0.1, 2026-10-17): introduced under V04DLT v1.0 B3a.
# Changelog (v0.2, 2026-10-17, review fixes R3/R4):
#   * Missing session, record or run-log evidence is refused, not skipped.
#   * Checkpoint bytes are hashed against the record (no tensor load).
#   * Records validated against the ledger and the locked 30-epoch budget.
# Changelog (v0.3, 2026-10-17, review fixes R5/R6):
#   * Manifest sidecars required and no longer mistaken for orphan manifests.
#   * Both run logs required per ledger attempt, tied to their session.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from pathlib import Path

import numpy as np

from seqref_mri.scripts import v04_train as v04t
from seqref_mri.scripts import v04dlt_train as dlt
from seqref_mri.scripts.v02_manifests import canonical_json

logger = logging.getLogger("seqref_mri.v04dlt_summary")
__version__ = "0.3"
__abbr__ = "SEQREF-MRI-V04DLTS"

SCHEMA = "seqref-v04dlt-summary/1"
N_EPOCHS = dlt.N_EPOCHS_DLT                 # 30 (locked, 184 A3)
STEPS_PER_EPOCH = dlt.STEPS_PER_EPOCH      # 1086
TOTAL_STEPS = dlt.TOTAL_STEPS_DLT          # 32,580
LAMBDA_2P = 2.4348938608414534             # registered lambda_2' (184)
STATUSES = ("completed", "interrupted", "numerical_failure", "failed_validity")
V04_EPOCHS = 3
V04_STEPS = 3258
SMOOTH_WINDOW = 250
DPI = 120
ARMS, SEEDS = (0, 1), (101, 202, 303)
MONITORS = ("val_NLL", "val_L_rec", "L_val")
STEP_KEYS = ("L_density", "L_total", "L_rec", "lam_L_rec", "grad_norm")
V04_LEDGER_SHA = "6962c0db0b4a2e3ce54de6fe8a8669e45c9e6987c80216ba0da1234480828f89"
V04_RECORD_SHA = {
    (0, 101): "f591a5efff4bf334e5ad6883c380da51d888276f700fb3c080d378d6b8d41481",
    (1, 101): "d129bde2728d22235744ba8e1dbd266ae2821807345ce1c2d2275c8e43b6ad6d",
    (0, 202): "5e08b27d2a7523045eee265b32159aa7f108a053418fff8bbd2c79e8e8b086c7",
    (1, 202): "b917eb9ee8bf00e0ce17de5e87da2b90f06dc87de6280daf6cefa4c37fdb9689",
    (0, 303): "f69cb2f6d2ced5c8ab98238c1fa0b4c0746a9851eb8742f456ae7e72fe20e125",
    (1, 303): "474750df0e5dd401e429397d3a34bc3becda74f711c159218ebb9b907df324b4",
}
PLOTS = (
    ("plot1_monitor_nll_vs_epoch.png", "Training-split monitor: NLL vs epoch"),
    ("plot2_monitor_rec_vs_epoch.png",
     "Training-split monitor: reconstruction loss vs epoch"),
    ("plot3_monitor_lval_vs_epoch.png", "Training-split monitor: L_val vs epoch"),
    ("plot4_training_loss_vs_step.png", "Training loss vs step"),
    ("plot5_grad_norm_vs_step.png", "Global gradient norm vs step"),
    ("plot6_paired_difference_vs_epoch.png",
     "Training-split monitor: paired arm difference (arm 1 - arm 0) vs epoch"),
    ("plot7_reproducibility_epochs1to3.png",
     "Epoch 1-3 reproducibility: new vs V04 monitor values"),
    ("plot8_wall_clock_by_run.png", "Wall-clock runtime by arm and seed"),
)


class V04DLTSError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V04DLTSError(code, message)


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
    """R3: every launcher session left its complete evidence set."""
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
    # R6: every ledger attempt has both run logs, tied to its session
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


def _validate_record(e: dict, rec: dict, pin8: str) -> None:
    """R4: the record agrees with its ledger entry and the locked budget."""
    name = e["record_file"]
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
    want_lam = 0.0 if e["arm"] == 0 else LAMBDA_2P
    if e["arm"] not in ARMS or e["seed"] not in SEEDS or \
            rec["lambda"] != want_lam:
        _fail("RECORD_LAMBDA_MISMATCH",
              f"{name}: arm {e['arm']} lambda {rec['lambda']!r} != {want_lam!r}")
    pin = rec.get("commit_pin")
    if not isinstance(pin, str) or len(pin) != 40 or pin[:8] != pin8:
        _fail("RECORD_PIN_MISMATCH", f"{name}: commit_pin {pin!r} vs root {pin8}")
    cfg = rec.get("config") or {}
    if cfg.get("epochs") != N_EPOCHS or cfg.get("steps") != TOTAL_STEPS:
        _fail("RECORD_CONFIG_MISMATCH",
              f"{name}: config epochs/steps {cfg.get('epochs')}/"
              f"{cfg.get('steps')} != {N_EPOCHS}/{TOTAL_STEPS}")
    k = len(cks)
    for i, c in enumerate(cks, start=1):
        if (c.get("epoch"), c.get("file"), c.get("arm"), c.get("seed"),
                c.get("attempt"), c.get("lambda")) != \
                (i, v04t.checkpoint_name(e["arm"], e["seed"], e["attempt"], i),
                 e["arm"], e["seed"], e["attempt"], want_lam):
            _fail("RECORD_CHECKPOINT_INVALID", f"{name}: checkpoint {i}")
    if [int(v.get("epoch", -1)) for v in rec.get("validation_history", [])] \
            != list(range(1, k + 1)):
        _fail("RECORD_COVERAGE_INVALID",
              f"{name}: validation epochs != checkpoint epochs 1..{k}")
    hist = rec.get("history", [])
    if [h.get("step") for h in hist] != list(range(1, len(hist) + 1)):
        _fail("RECORD_COVERAGE_INVALID", f"{name}: history steps not 1..n")
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


def load_inputs(root: Path, v04_train_dir: Path) -> dict:
    register = {"script/v04dlt_summary.py": _sha(Path(__file__).resolve())}
    lp = root / "v04_run_ledger.json"
    if not lp.is_file():
        _fail("LEDGER_MISSING", f"{lp}")
    led = v04t.load_ledger(lp)                 # sidecar-verified (V04T)
    register["campaign/v04_run_ledger.json"] = _sha(lp)
    _check_sidecars(root, register)
    _check_snapshots(root, led)
    _check_sessions(root, led)
    pin8 = root.name[len("V04DLT_"):]
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
    v04_led = v04_train_dir / "v04_run_ledger.json"
    if _sha(v04_led) != V04_LEDGER_SHA:
        _fail("V04_EVIDENCE_CHANGED", "V04 ledger hash")
    register["v04/v04_run_ledger.json"] = V04_LEDGER_SHA
    v04 = {}
    for (arm, seed), want in V04_RECORD_SHA.items():
        p = v04_train_dir / v04t.run_record_name(arm, seed, 1)
        if _sha(p) != want:
            _fail("V04_EVIDENCE_CHANGED", f"{p.name}")
        v04[(arm, seed)] = json.loads(p.read_text())
        register[f"v04/{p.name}"] = want
    return {"ledger": led, "records": records, "v04": v04,
            "register": register}


# ---------------------------------------------------------------------------
# Per-run table (A5), selections (A6), paired / cross-seed summaries
# ---------------------------------------------------------------------------

def settle(led: dict) -> dict:
    """Per pair: final status, attempts used, and the attempt whose record
    is plotted (the completed one; else the last attempt, incomplete)."""
    out = {}
    for seed in SEEDS:
        for arm in ARMS:
            es = [e for e in led["entries"]
                  if e["arm"] == arm and e["seed"] == seed]
            if not es:
                out[(arm, seed)] = {"status": "not_attempted", "attempts": 0,
                                    "attempt": None, "cause": "not_attempted"}
                continue
            last = es[-1]
            comp = [e for e in es if e["status"] == "completed"]
            if comp:
                cause = None
            elif last["status"] == "numerical_failure":
                cause = "numerical_failure (terminal)"
            elif last["status"] == "failed_validity":
                cause = "failed_validity (terminal)"
            elif len(es) >= v04t.MAX_ATTEMPTS:
                cause = "infrastructure interruption, 3 attempts exhausted"
            else:
                cause = "incomplete (interrupted, attempts remain)"
            pick = comp[0] if comp else last
            out[(arm, seed)] = {"status": "completed" if comp else
                                last["status"], "attempts": len(es),
                                "attempt": pick["attempt"], "cause": cause}
    return out


def _vh(rec: dict) -> dict:
    return {int(v["epoch"]): v for v in rec["validation_history"]}


def _val(vh: dict, ep: int, key: str):
    v = vh.get(ep, {}).get(key)
    return None if v is None else float(v)


def _delta(vh, a, b, key):
    x, y = _val(vh, a, key), _val(vh, b, key)
    return None if x is None or y is None else y - x


def run_table(inp: dict, st: dict) -> list:
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            s = st[(arm, seed)]
            row = {"arm": arm, "seed": seed, "status": s["status"],
                   "attempts": s["attempts"], "exclusion_cause": s["cause"]}
            if s["attempt"] is not None:
                rec = inp["records"][(arm, seed, s["attempt"])]
                vh = _vh(rec)
                row.update({
                    "attempt": s["attempt"], "lambda": rec["lambda"],
                    "steps_completed": rec["steps_completed"],
                    "error_code": rec["error_code"],
                    "wall_clock_s": rec["wall_clock_s"],
                    "torch_num_threads": rec["config"].get("torch_num_threads"),
                    "commit_pin": rec["commit_pin"]})
                for k in MONITORS:
                    row[f"{k}_epoch3"] = _val(vh, 3, k)
                    row[f"{k}_epoch27"] = _val(vh, 27, k)
                    row[f"{k}_epoch30"] = _val(vh, N_EPOCHS, k)
                    row[f"{k}_change_3_to_30"] = _delta(vh, 3, N_EPOCHS, k)
                    row[f"{k}_change_27_to_30"] = _delta(vh, 27, N_EPOCHS, k)
                if s["status"] == "completed":
                    sel = v04t.select_checkpoint(rec["validation_history"],
                                                 rec["checkpoints"])
                    row["selected_epoch"] = sel["epoch"]
                    row["selected_file_sha256"] = sel["file_sha256"]
            rows.append(row)
    return rows


def _mean_sd(xs: list) -> dict:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None}
    sd = float(np.std(xs, ddof=1)) if n >= 2 else None
    return {"n": n, "mean": float(np.mean(xs)), "sd": sd,
            "sd_note": None if n >= 2 else "not available (n < 2)"}


def paired(inp: dict, st: dict) -> dict:
    pairs = [s for s in SEEDS if st[(0, s)]["status"] == "completed"
             and st[(1, s)]["status"] == "completed"]
    per_seed = {}
    for s in pairs:
        v0 = _vh(inp["records"][(0, s, st[(0, s)]["attempt"])])
        v1 = _vh(inp["records"][(1, s, st[(1, s)]["attempt"])])
        per_seed[str(s)] = {k: [None if _val(v1, e, k) is None or
                                _val(v0, e, k) is None else
                                _val(v1, e, k) - _val(v0, e, k)
                                for e in range(1, N_EPOCHS + 1)]
                            for k in ("val_NLL", "val_L_rec")}
    out = {"completed_matched_pairs": len(pairs), "seeds": pairs,
           "per_seed_difference_by_epoch": per_seed}
    if not pairs:
        out["note"] = "n = 0: no paired summary; per-run table and failure " \
                      "accounting only"
        return out
    cross = {}
    for k in ("val_NLL", "val_L_rec"):
        cross[f"difference_{k}_epoch30"] = _mean_sd(
            [per_seed[str(s)][k][N_EPOCHS - 1] for s in pairs])
    for arm in ARMS:
        for k in MONITORS:
            vals = [_vh(inp["records"][(arm, s, st[(arm, s)]["attempt"])])
                    for s in pairs]
            cross[f"arm{arm}_{k}_epoch30"] = _mean_sd(
                [_val(v, N_EPOCHS, k) for v in vals])
            cross[f"arm{arm}_{k}_change_3_to_30"] = _mean_sd(
                [_delta(v, 3, N_EPOCHS, k) for v in vals])
            cross[f"arm{arm}_{k}_change_27_to_30"] = _mean_sd(
                [_delta(v, 27, N_EPOCHS, k) for v in vals])
    out["cross_seed"] = cross
    out["comparability_rule"] = ("NLL and reconstruction compared across "
                                 "arms; L_total and L_val include lambda and "
                                 "are not compared across arms")
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


def reproducibility(inp: dict, st: dict) -> dict:
    out = {"rule": "check, not a promise; non-gating; V04-recorded fields "
                   "only; no change may be made to force agreement",
           "runs": {}}
    for (arm, seed), v04rec in inp["v04"].items():
        s = st[(arm, seed)]
        if s["attempt"] is None:
            out["runs"][f"arm{arm}_seed{seed}"] = {"available": False}
            continue
        rec = inp["records"][(arm, seed, s["attempt"])]
        va, vb = _vh(rec), _vh(v04rec)
        mon = {k: _diff_stats([_val(va, e, k) for e in range(1, 4)],
                              [_val(vb, e, k) for e in range(1, 4)])
               for k in MONITORS}
        ha = {h["step"]: h for h in rec["history"] if h["step"] <= V04_STEPS}
        hb = {h["step"]: h for h in v04rec["history"]}
        steps = sorted(set(ha) & set(hb))
        stp = {k: _diff_stats([ha[t].get(k) for t in steps],
                              [hb[t].get(k) for t in steps])
               for k in STEP_KEYS}
        env_new, env_old = rec.get("environment", {}), v04rec.get(
            "environment", {})
        out["runs"][f"arm{arm}_seed{seed}"] = {
            "available": True, "epochs_1_3_monitor": mon,
            "steps_compared": len(steps), "per_step": stp,
            "environment": {k: {"new": env_new.get(k), "v04": env_old.get(k),
                                "equal": env_new.get(k) == env_old.get(k)}
                            for k in ("python", "torch", "platform")},
            "threads": {"new": rec["config"].get("torch_num_threads"),
                        "v04": "unknown (not recorded by V04)"},
            "cpu_model": {"v04": "unknown (not recorded by V04)"}}
    return out


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


def _plots(out: Path, inp: dict, st: dict, pr: dict, repro: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    color = {0: "#1f77b4", 1: "#d62728"}
    style = {101: "-", 202: "--", 303: ":"}
    recs = {(a, s): inp["records"][(a, s, st[(a, s)]["attempt"])]
            for a in ARMS for s in SEEDS if st[(a, s)]["attempt"] is not None}
    files = {}

    def save(fig, name, title):
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        path = out / name
        if path.exists():
            _fail("ARTIFACT_EXISTS", str(path))
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        files[name] = _sha(path)

    def epoch_curves(ax, key, arms):
        for a in arms:
            series = []
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
                elif st[(a, s)]["status"] == "completed":
                    series.append(vh)
            n = len(series)
            if n:
                eps = list(range(1, N_EPOCHS + 1))
                m = [float(np.mean([v[e][key] for v in series])) for e in eps]
                ax.plot(eps, m, "-", color=color[a], linewidth=2.2,
                        label=f"arm {a} mean (n={n})")
                if n >= 2:
                    sd = [float(np.std([v[e][key] for v in series], ddof=1))
                          for e in eps]
                    ax.fill_between(eps, np.array(m) - sd, np.array(m) + sd,
                                    color=color[a], alpha=0.15)
                else:
                    ax.text(0.02, 0.02 + 0.06 * a,
                            f"arm {a}: SD not available (n < 2)",
                            transform=ax.transAxes, fontsize=7)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.legend(fontsize=6)

    for idx, key in ((0, "val_NLL"), (1, "val_L_rec")):
        fig, ax = plt.subplots(figsize=(8, 5))
        epoch_curves(ax, key, ARMS)
        save(fig, *PLOTS[idx])
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for a, ax in zip(ARMS, axs):
        epoch_curves(ax, "L_val", (a,))
        for s in SEEDS:
            if st[(a, s)]["status"] == "completed":
                sel = v04t.select_checkpoint(recs[(a, s)]["validation_history"],
                                             recs[(a, s)]["checkpoints"])
                ax.axvline(sel["epoch"], color=color[a], linestyle=style[s],
                           alpha=0.4)
        ax.set_title(f"arm {a} (selected epochs marked)", fontsize=8)
    save(fig, *PLOTS[2])
    for idx, keys, logy in ((3, ("L_density", "L_total"), False),
                            (4, ("grad_norm",), True)):
        fig, axs = plt.subplots(1, len(keys) if idx == 4 else 3,
                                figsize=(15 if idx == 3 else 8, 5))
        axs = np.atleast_1d(axs)
        panels = ([("L_density", ARMS, "NLL (both arms)")] +
                  [("L_total", (a,), f"L_total arm {a}") for a in ARMS]) \
            if idx == 3 else [("grad_norm", ARMS, "global gradient norm")]
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
        save(fig, *PLOTS[idx])
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key in zip(axs, ("val_NLL", "val_L_rec")):
        for s, d in pr["per_seed_difference_by_epoch"].items():
            eps = [e + 1 for e, v in enumerate(d[key]) if v is not None]
            ax.plot(eps, [v for v in d[key] if v is not None],
                    style[int(s)], color="k", label=f"seed {s}")
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.set_title(f"{key}: arm 1 - arm 0 (completed matched pairs, "
                     f"n={pr['completed_matched_pairs']})", fontsize=8)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=6)
    save(fig, *PLOTS[5])
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, key in zip(axs, MONITORS):
        for (a, s), v04rec in inp["v04"].items():
            if (a, s) not in recs:
                continue
            new, old = _vh(recs[(a, s)]), _vh(v04rec)
            eps = [e for e in (1, 2, 3) if e in new and e in old]
            ax.plot(eps, [old[e][key] for e in eps], "o" + style[s],
                    color=color[a], alpha=0.4, label=f"V04 arm {a} seed {s}")
            ax.plot(eps, [new[e][key] for e in eps], "x" + style[s],
                    color=color[a], label=f"new arm {a} seed {s}")
            r = repro["runs"][f"arm{a}_seed{s}"]["epochs_1_3_monitor"][key]
            if r["max_abs"] is not None and eps:
                ax.annotate(f"max|d|={r['max_abs']:.2e}",
                            (eps[-1], new[eps[-1]][key]), fontsize=6)
        ax.set_title(key, fontsize=8)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=5)
    save(fig, *PLOTS[6])
    fig, ax = plt.subplots(figsize=(8, 5))
    labels, vals, cols = [], [], []
    for s in SEEDS:
        for a in ARMS:
            if (a, s) in recs:
                labels.append(f"a{a}\ns{s}")
                vals.append(recs[(a, s)]["wall_clock_s"] / 3600.0)
                cols.append(color[a])
    ax.bar(range(len(vals)), vals, color=cols)
    ax.set_xticks(range(len(vals)), labels, fontsize=7)
    ax.set_ylabel("wall clock (h, host-dependent)")
    save(fig, *PLOTS[7])
    return files


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build(root: Path, v04_train_dir: Path) -> int:
    if not root.is_dir() or not root.name.startswith("V04DLT_"):
        _fail("CAMPAIGN_ROOT_INVALID", str(root))
    out = root / "summary"
    if out.exists():
        _fail("ARTIFACT_EXISTS", f"{out} exists; the summary is never rebuilt "
              f"in place")
    inp = load_inputs(root, v04_train_dir)
    st = settle(inp["ledger"])
    table = run_table(inp, st)
    pr = paired(inp, st)
    repro = reproducibility(inp, st)
    out.mkdir()
    files = _plots(out, inp, st, pr, repro)
    import matplotlib
    summary = {"schema": SCHEMA, "script": f"{__abbr__} v{__version__}",
               "study": "SEQREF-MRI-V04DLT v1.0 (Article 184)",
               "readouts": "descriptive training-split monitor values only; "
                           "no thresholds, no verdicts, no holdout",
               "run_table": table, "paired": pr,
               "reproducibility": repro,
               "input_hash_register": inp["register"],
               "environment": {"python": platform.python_version(),
                               "matplotlib": matplotlib.__version__,
                               "numpy": np.__version__}}
    s_sha = _write(out / "v04dlt_summary.json", canonical_json(summary))
    manifest = {"schema": "seqref-v04dlt-plot-manifest/1",
                "generator": {"script": f"{__abbr__} v{__version__}",
                              "sha256": inp["register"][
                                  "script/v04dlt_summary.py"],
                              "backend": "Agg", "dpi": DPI,
                              "smoothing_window": SMOOTH_WINDOW},
                "plots": [{"file": n, "title": t, "sha256": files[n]}
                          for n, t in PLOTS],
                "summary_sha256": s_sha,
                "input_hash_register": inp["register"],
                "note": "PNG bytes are host-dependent; only the host-generated "
                        "set is registered"}
    _write(out / "plot_manifest.json", canonical_json(manifest))
    print(f"PASS {__abbr__} v{__version__}: summary {s_sha[:12]}..., "
          f"{len(files)} plots, completed matched pairs "
          f"{pr['completed_matched_pairs']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="v04dlt_summary")
    ap.add_argument("--campaign-root", required=True)
    ap.add_argument("--v04-train-dir", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return build(Path(a.campaign_root).resolve(),
                     Path(a.v04_train_dir).resolve())
    except (V04DLTSError, v04t.V04TError) as exc:
        logger.error("[%s] ERROR: %s", __abbr__, exc)
        return 2
    except Exception:
        logger.exception("[%s] unexpected exception", __abbr__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
