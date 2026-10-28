# SEQREF-MRI-V05LTL v0.6 -- scripts.v05lt_launch
# LIFETIME: KEEP
# =============================================================================
# Purpose: campaign launcher logic for SEQREF-MRI-V05LT (I1.9 stage c1, plan
#   SEQREF-MRI-V05LTI19 v0.1; 195 §4, §5, §8.3, §9.1 + PA5-C). Sequential,
#   seed-major stages (epoch0(s) -> s/lam0 -> s/lam2, s = 2026, 2027, 2028)
#   under a campaign lock and a per-run lock, with:
#   * launch chain: the final launch-chain file (pin, gate 2a, P1-P6
#     evidence; built only after those stages) must match the approved
#     sha256 recorded in V05LTEXEC; every listed file is re-hashed;
#     internal agreement alone is refused;
#   * explicit modes: `new` (nothing of the campaign may exist) and
#     `resume` (continue the campaign);
#   * run state from the run ledger only (read-only inspection):
#     completed / terminal / awaiting owner / session_error / refused /
#     interrupted (resumable within 3 sessions; the trainer itself enforces
#     the session limit and records `interrupted — awaiting owner decision`);
#   * owner decisions (item 19): a write-once record bound to run ID,
#     configuration sha256, latest ledger sequence and line sha256, session
#     count and status, approved by its sha256. `resume` only for a
#     recoverable session_error while < 3 sessions have started (consumes the
#     next session); `halt` otherwise; terminal statuses never resumed; a
#     record never authorises a later interruption (binding mismatch);
#   * advancement (PA5-C): a run advances only when it ended stopped /
#     budget exhausted AND the post-run stage reports its final weights and
#     post-run outputs published and verified; any failure halts;
#   * data-content verification (PA5-A) is called before every session
#     (epoch-0 stage and each training session) through the stage interface;
#   * every halt and decision is appended to the campaign ledger.
# Stage execution is injected (`Stages` interface below). Production
#   (v0.4) runs only with the production stage class
#   (v05lt_prodstages.ProductionStages, production mode, exact type) and a
#   launch chain verified against its approved sha256; any fixture stage
#   set or fixture-mode production stages are refused in production. In
#   practice production stays impossible until the pin, gate 2a, P1-P6, the
#   launch-inputs record and the recorded launch approval exist (195 §8.3).
#   Holdout stage (v0.6, I1.9 c4): a SEPARATE `holdout` command
#   (Campaign.holdout); `new` / `resume` never call it, so finishing
#   training never triggers it.
# CONVENTION: logger.error + V05LTError(code); no fallback, no placeholder.
# Changelog (v0.6, 2026-10-27; I1.9 c4, SEQREF-MRI-V05LTI19C4 v0.1):
#   * Campaign.holdout(): campaign lock held throughout; launch chain and
#     launch-inputs binding as for a launch; verified completion
#     (refinement 1: the six expected run identities, each with an
#     `advance` in the latest completed launch, current evidence re-verified
#     read-only -- run state + registered configuration, verify_final,
#     every listed post-run record by sha256 and load_posteval, each
#     seed's epoch-0 stage); holdout approval (distinct input) must name the
#     launcher's approved launch chain; the accepted c3 gate
#     (v05lt_holdout.verify_gate) called unchanged; the approved selection
#     bound to the verified campaign; c3 output guard and holdout lock
#     (inside the campaign lock); c3 Runner.
#   * Campaign-ledger mirror records holdout_start / holdout_done /
#     holdout_halt (references only; not campaign halts; no second retry
#     mechanism); mirror-write failure -> LAUNCH_HOLDOUT_MIRROR_FAILED;
#     re-entry after completion verifies index + records by hash against
#     the holdout ledger, writes a missing mirror, never re-evaluates or
#     opens holdout MRI data; unmirrored holdout halts reconciled.
#   * holdout_status(): read-only (no file created, no MRI opened); main()
#     gains `holdout` and prints the holdout state with `status`.
#   * _verify_chain() factored out of run() (behaviour unchanged).
#   * Revision 2 (review of revision 1): a re-entry verification failure
#     (altered index or record) is recorded as a c3 halt
#     HOLDOUT_OUTPUT_MISMATCH (stage `reentry`, failing task) in the
#     holdout ledger through c3's Runner._halt and mirrored by reference;
#     restoring the files alone stays blocked; continuation only through
#     the c3 decision bound to that halt (retry retains the failed attempt
#     under _failed/). Mirror-write failures stay a separate recoverable
#     typed error.
#   * Revision 3 (review of revision 2): an unexpected ordinary exception
#     during re-entry verification (e.g. a file-read PermissionError) is
#     recorded as the c3 halt HOLDOUT_STAGE_ERROR (stage `reentry`, task,
#     type, message, traceback) and mirrored; clearing the fault alone
#     never resumes. BaseException (process interruption) passes through.
# Changelog (v0.5, 2026-10-26; c2-wiring revision-1 review):
#   * Direct launcher-to-input binding assertion: with ProductionStages the
#     launch refuses (LAUNCH_INPUTS_UNBOUND) unless bind_chain() returned
#     and holds the sha256 of the chain's launch_inputs entry.
# Changelog (v0.4, 2026-10-26; I1.9 c2 production wiring, V05LTI19C2P
#   v0.1):
#   * Launch chain schema /2 with the explicit `launch_inputs` role
#     (refinement 1); after the chain verifies, stages.bind_chain() verifies
#     the launch-inputs record before anything runs (a refusal; nothing is
#     produced, so nothing is recorded -- as for the chain itself).
#   * Production enabled only for ProductionStages in production mode
#     (LAUNCH_STAGES_NOT_PRODUCTION / LAUNCH_FIXTURE_IN_PRODUCTION);
#     fixture campaigns refuse production-mode stages (refinement 2).
#   * Addition C (refinement 5): every stage call goes through one
#     boundary: typed errors keep their handling; any other exception is
#     logged with its traceback and recorded as the campaign halt
#     LAUNCH_STAGE_ERROR {stage, type, message, traceback}, which needs a
#     bound campaign decision before retry. expected_config() is covered:
#     typed failures -> LAUNCH_CONFIG_FAILED (campaign halt). Training keeps
#     LAUNCH_SESSION_ERROR (gated by the run ledger), now with traceback.
#     Process interruptions (BaseException) pass through unchanged.
#   * main(): production entry (--chain, --approved-sha256, --decision,
#     --campaign-decision) with ProductionStages; status mode read-only.
# Changelog (v0.3, 2026-10-25; c1 revision-2 review fix):
#   * Both epoch-0 verifications (before execution and after the stage
#     returns) run inside the stage handler: a verifier error records the
#     campaign halt LAUNCH_EPOCH0_FAILED with the seed and the original
#     code, so restoring the artefact alone cannot resume.
# Changelog (v0.2, 2026-10-25; c1 revision-1 review fixes):
#   * Campaign-stage halts (data / epoch-0 / post-run / run not completed)
#     are not recorded in any run ledger, so clearing the fault alone no
#     longer permits continuation: the unresolved halt must be resolved by a
#     campaign decision bound to that halt record (ledger seq + line
#     sha256, code, stage); stale or unrelated decisions are refused;
#     `halt` ends the campaign (LAUNCH_CAMPAIGN_HALTED).
#   * Launch-chain entries validated structurally before use (exact keys,
#     string types, sha256 form): malformed entries -> LAUNCH_CHAIN_INVALID.
#   * Epoch-0 stage returning unverified now halts (recorded), not a bare
#     refusal.
# Changelog (NEW in v0.1, 2026-10-25; I1.9 c1, SEQREF-MRI-V05LTI19 v0.1):
#   * Launch-chain manifest (make / verify; approved hash first).
#   * Campaign and per-run flock locks; seed-major order; new/resume modes.
#   * Read-only run-state inspection; owner-decision records and rules.
#   * Advancement on verified post-run outputs; campaign ledger; halts.
# =============================================================================
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import sys
import traceback
from pathlib import Path

from seqref_mri.scripts import v05lt_common as C
from seqref_mri.scripts import v05lt_prodstages as PS

logger = logging.getLogger("seqref_mri.v05lt_launch")

__version__ = "0.6"
__abbr__ = "SEQREF-MRI-V05LTL"

SEEDS = (2026, 2027, 2028)
ARMS = ("lam0", "lam2")
MAX_SESSIONS = 3
CHAIN_SCHEMA = "seqref-v05lt-launch-chain/2"
CHAIN_ROLES = ("pin", "gate2a", "P1", "P2", "P3", "P4", "P5", "P6",
               PS.LAUNCH_INPUTS_ROLE)
DECISION_SCHEMA = "seqref-v05lt-owner-decision/1"
DECISIONS = ("resume", "halt")
CAMPAIGN_DECISION_SCHEMA = "seqref-v05lt-campaign-decision/1"
CAMPAIGN_DECISIONS = ("retry", "halt")
# Halts whose cause is not held in any run ledger: continuing requires a
# campaign decision bound to the halt record (clearing the fault is not
# enough). Run-state halts are gated by the run ledger itself.
CAMPAIGN_HALTS = ("LAUNCH_EPOCH0_FAILED", "LAUNCH_DATA_UNVERIFIED",
                  "LAUNCH_POSTRUN_FAILED", "LAUNCH_RUN_NOT_COMPLETED",
                  "LAUNCH_STAGE_ERROR", "LAUNCH_CONFIG_FAILED")
TRACEBACK_MAX = 8000
# c4: campaign-ledger mirror kinds of the holdout stage (references to the
# authoritative holdout ledger; deliberately NOT "halt" records, so they
# never gate `resume` and create no second retry mechanism).
HOLDOUT_MIRROR_KINDS = ("holdout_start", "holdout_done", "holdout_halt")
LEDGER_NAME = "launch_ledger.jsonl"
LOCK_NAME = ".launch.lock"
# Trainer status strings (v05lt_rules / v05lt_train; checked by fixtures)
STOPPED = "stopped_plateau"
EXHAUSTED = "budget exhausted; plateau not established"
FIXTURE_END = "fixture_budget_end"
INTERRUPTED_STATUS = "interrupted — awaiting owner decision"
TERMINAL = ("numerical_failure", "failed_validity")
# Run-state classes
ABSENT, COMPLETED, TERMINAL_C, AWAITING, SESSION_ERROR, REFUSED, \
    INTERRUPTED, UNREADABLE = ("absent", "completed", "terminal",
                               "awaiting_owner", "session_error", "refused",
                               "interrupted", "unreadable")


def stage_order() -> list:
    """195 §4 seed-major order."""
    out = []
    for s in SEEDS:
        out.append(("epoch0", s, None))
        out += [("run", s, a) for a in ARMS]
    return out


def run_id(seed: int, arm: str) -> str:
    return f"{seed}_{arm}"           # 195 §5 run directory name


def _is_sha(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(
        c in "0123456789abcdef" for c in x)


def _safe_rel(p: str) -> str:
    C.require(isinstance(p, str) and p and not p.startswith("/") and
              all(x not in ("", ".", "..") for x in p.split("/")),
              "LAUNCH_CHAIN_PATH", f"unsafe relative path {p!r}")
    return p


# ---------------------------------------------------------------------------
# Launch chain (built after pin, gate 2a, P1-P6 and the launch-inputs record)
# ---------------------------------------------------------------------------
def make_launch_chain(out_path: Path, root: Path, roles: dict) -> str:
    """roles: {role: relpath} for exactly CHAIN_ROLES. Returns the sha256
    that the launch approval must record."""
    C.require(set(roles) == set(CHAIN_ROLES), "LAUNCH_CHAIN_ROLES",
              f"{sorted(roles)} != {list(CHAIN_ROLES)}")
    ents = []
    for role in CHAIN_ROLES:
        rel = _safe_rel(roles[role])
        f = Path(root) / rel
        C.require(f.is_file() and not f.is_symlink(), "LAUNCH_CHAIN_MISSING",
                  f"{role}: {rel}")
        ents.append({"role": role, "path": rel,
                     "sha256": C.sha256_file(f)})
    doc = {"schema": CHAIN_SCHEMA, "abbr": __abbr__, "version": __version__,
           "entries": ents}
    return C.write_durable_new(Path(out_path), C.canonical_json(doc))


def verify_launch_chain(path: Path, approved_sha256: str, root: Path) -> dict:
    C.require(isinstance(approved_sha256, str) and len(approved_sha256) == 64,
              "LAUNCH_CHAIN_UNAPPROVED", "approved sha256 required")
    p = Path(path)
    C.require(p.is_file() and not p.is_symlink(), "LAUNCH_CHAIN_MISSING",
              str(p))
    got = C.sha256_file(p)
    C.require(got == approved_sha256, "LAUNCH_CHAIN_UNAPPROVED",
              f"{p}: {got} != approved {approved_sha256}")
    try:
        doc = json.loads(p.read_bytes())
    except ValueError as exc:
        C.fail("LAUNCH_CHAIN_INVALID", f"{p}: {exc}")
    C.require(isinstance(doc, dict) and doc.get("schema") == CHAIN_SCHEMA and
              isinstance(doc.get("entries"), list), "LAUNCH_CHAIN_INVALID",
              str(p))
    for i, e in enumerate(doc["entries"]):
        C.require(isinstance(e, dict) and set(e) == {"role", "path", "sha256"}
                  and all(isinstance(e[k], str) for k in e) and
                  _is_sha(e["sha256"]), "LAUNCH_CHAIN_INVALID",
                  f"{p}: entry {i} malformed: {e!r:.120}")
    roles = [e["role"] for e in doc["entries"]]
    C.require(roles == list(CHAIN_ROLES), "LAUNCH_CHAIN_ROLES",
              f"{roles} != {list(CHAIN_ROLES)}")
    for e in doc["entries"]:
        f = Path(root) / _safe_rel(e["path"])
        C.require(f.is_file() and not f.is_symlink(), "LAUNCH_CHAIN_MISSING",
                  f"{e['role']}: {e.get('path')}")
        disk = C.sha256_file(f)
        C.require(disk == e.get("sha256"), "LAUNCH_CHAIN_HASH",
                  f"{e['role']}: disk {disk} != chain {e.get('sha256')}")
    return doc


# ---------------------------------------------------------------------------
# Locks (195 §5: campaign lock plus per-run lock)
# ---------------------------------------------------------------------------
class Lock:
    """Non-blocking exclusive flock; held for the context's lifetime."""

    def __init__(self, path: Path, what: str):
        self.path, self.what, self.fd = Path(path), what, None

    def __enter__(self):
        try:
            self.fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as exc:
            C.fail("LAUNCH_LOCK_FAILED", f"{self.what}: {self.path}: {exc}")
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            self.fd = None
            C.fail("LAUNCH_LOCK_HELD", f"{self.what} lock held: {self.path}")
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None
        return False


# ---------------------------------------------------------------------------
# Run-state inspection (read-only)
# ---------------------------------------------------------------------------
def config_sha256(config: dict) -> str:
    return C.sha256_bytes(C.canonical_json(config))


def inspect_run(run_dir: Path, *, fixture: bool = False) -> dict:
    """Classify a run from its ledger only; never writes."""
    d = Path(run_dir)
    if not d.exists():
        return {"class": ABSENT}
    led = C.Ledger(d / "ledger.jsonl")
    if not led.path.is_file():
        return {"class": UNREADABLE, "reason": "run directory without ledger"}
    recs, _, tail = led._scan()          # read-only; torn tail reported
    if not recs or recs[0]["kind"] != "run_start":
        return {"class": UNREADABLE, "reason": "ledger has no run_start"}
    head = recs[-1]
    out = {"config_sha256": config_sha256(recs[0]["body"]["config"]),
           "sessions": sum(1 for r in recs if r["kind"] == "session_start"),
           "head_seq": head["seq"],
           "head_sha256": C.sha256_bytes(C.canonical_json(head)),
           "torn_tail": bool(tail)}
    ends = [r for r in recs if r["kind"] == "run_end"]
    if ends:
        st = ends[-1]["body"].get("status")
        out["status"] = st
        if st in (STOPPED, EXHAUSTED) or (fixture and st == FIXTURE_END):
            out["class"] = COMPLETED
        elif st in TERMINAL:
            out["class"] = TERMINAL_C
        elif st == INTERRUPTED_STATUS:
            out["class"] = AWAITING
        else:
            out["class"] = UNREADABLE
            out["reason"] = f"unregistered run_end status {st!r}"
        return out
    starts = [i for i, r in enumerate(recs) if r["kind"] == "session_start"]
    after = recs[starts[-1] + 1:] if starts else recs
    if any(r["kind"] == "session_error" for r in after):
        out["class"] = SESSION_ERROR
    elif head["kind"] == "resume_refused":
        out["class"] = REFUSED
        out["code"] = head["body"].get("code")
    else:
        out["class"] = INTERRUPTED
    return out


# ---------------------------------------------------------------------------
# Owner decisions (item 19; refinement 2)
# ---------------------------------------------------------------------------
def _binding(rid: str, state: dict) -> dict:
    return {"run_id": rid, "config_sha256": state.get("config_sha256"),
            "ledger_seq": state.get("head_seq"),
            "ledger_line_sha256": state.get("head_sha256"),
            "sessions_started": state.get("sessions"),
            "status_class": state.get("class")}


def make_owner_decision(out_path: Path, rid: str, run_dir: Path,
                        decision: str, note: str, *,
                        fixture: bool = False) -> str:
    """Write-once owner decision bound to the run's CURRENT state."""
    C.require(decision in DECISIONS, "OWNER_DECISION_INVALID", decision)
    C.require(isinstance(note, str) and note.strip() != "",
              "OWNER_DECISION_INVALID", "a reason (note) is required")
    st = inspect_run(run_dir, fixture=fixture)
    C.require(st["class"] in (SESSION_ERROR, AWAITING, REFUSED),
              "OWNER_DECISION_NOT_APPLICABLE",
              f"{rid}: state {st['class']} needs no owner decision")
    doc = {"schema": DECISION_SCHEMA, "binding": _binding(rid, st),
           "decision": decision, "note": note, "time_utc": C.utc_now()}
    return C.write_durable_new(Path(out_path), C.canonical_json(doc))


def check_owner_decision(decision: tuple, rid: str, state: dict) -> str:
    """decision = (path, approved sha256). Returns the decision value."""
    C.require(decision is not None, "LAUNCH_OWNER_DECISION_REQUIRED",
              f"{rid}: state {state['class']} requires an owner decision")
    path, sha = decision
    p = Path(path)
    C.require(p.is_file() and C.sha256_file(p) == sha,
              "OWNER_DECISION_UNAPPROVED", f"{p}: hash != approved {sha}")
    try:
        doc = json.loads(p.read_bytes())
    except ValueError as exc:
        C.fail("OWNER_DECISION_INVALID", f"{p}: {exc}")
    C.require(isinstance(doc, dict) and doc.get("schema") == DECISION_SCHEMA
              and doc.get("decision") in DECISIONS, "OWNER_DECISION_INVALID",
              str(p))
    C.require(doc.get("binding") == _binding(rid, state),
              "OWNER_DECISION_STALE", f"{rid}: decision bound to "
              f"{doc.get('binding')} but the run is now {_binding(rid, state)}")
    dec = doc["decision"]
    if dec == "resume":
        C.require(state["class"] == SESSION_ERROR and
                  state["sessions"] < MAX_SESSIONS,
                  "OWNER_DECISION_NOT_ALLOWED", f"{rid}: resume allowed only "
                  f"for a session_error with < {MAX_SESSIONS} sessions started "
                  f"(state {state['class']}, sessions {state['sessions']})")
    return dec


# ---------------------------------------------------------------------------
# Campaign decisions (unresolved campaign-stage halts; review of rev 1)
# ---------------------------------------------------------------------------
def campaign_halt_state(ledger: "C.Ledger") -> dict:
    """{'unresolved': halt record or None, 'halted': bool} from the
    campaign ledger (read-only)."""
    unresolved, halted = None, False
    for r in ledger.records() if ledger.path.exists() else []:
        if r["kind"] == "halt" and r["body"].get("code") in CAMPAIGN_HALTS \
                and unresolved is None:
            unresolved = r
        elif r["kind"] == "halt_resolved" and unresolved is not None and \
                r["body"].get("halt_seq") == unresolved["seq"]:
            if r["body"].get("decision") == "halt":
                halted = True
            unresolved = None
    return {"unresolved": unresolved, "halted": halted}


def _halt_binding(rec: dict) -> dict:
    b = rec["body"]
    return {"halt_seq": rec["seq"],
            "halt_line_sha256": C.sha256_bytes(C.canonical_json(rec)),
            "halt_code": b.get("code"),
            "stage": {k: b.get(k) for k in ("run", "seed")}}


def make_campaign_decision(out_path: Path, root: Path, decision: str,
                           note: str) -> str:
    """Write-once decision bound to the campaign's CURRENT unresolved
    campaign-stage halt."""
    C.require(decision in CAMPAIGN_DECISIONS, "OWNER_DECISION_INVALID",
              decision)
    C.require(isinstance(note, str) and note.strip() != "",
              "OWNER_DECISION_INVALID", "a reason (note) is required")
    hs = campaign_halt_state(C.Ledger(Path(root) / LEDGER_NAME))
    C.require(hs["unresolved"] is not None, "OWNER_DECISION_NOT_APPLICABLE",
              f"{root}: no unresolved campaign-stage halt")
    doc = {"schema": CAMPAIGN_DECISION_SCHEMA,
           "binding": _halt_binding(hs["unresolved"]), "decision": decision,
           "note": note, "time_utc": C.utc_now()}
    return C.write_durable_new(Path(out_path), C.canonical_json(doc))


def check_campaign_decision(decision: tuple, halt: dict) -> str:
    C.require(decision is not None, "LAUNCH_HALT_UNRESOLVED",
              f"unresolved campaign halt {halt['body'].get('code')} at seq "
              f"{halt['seq']}: clearing the fault is not enough; a campaign "
              "decision bound to this halt is required")
    path, sha = decision
    p = Path(path)
    C.require(p.is_file() and C.sha256_file(p) == sha,
              "OWNER_DECISION_UNAPPROVED", f"{p}: hash != approved {sha}")
    try:
        doc = json.loads(p.read_bytes())
    except ValueError as exc:
        C.fail("OWNER_DECISION_INVALID", f"{p}: {exc}")
    C.require(isinstance(doc, dict) and
              doc.get("schema") == CAMPAIGN_DECISION_SCHEMA and
              doc.get("decision") in CAMPAIGN_DECISIONS,
              "OWNER_DECISION_INVALID", str(p))
    C.require(doc.get("binding") == _halt_binding(halt),
              "OWNER_DECISION_STALE", f"decision bound to "
              f"{doc.get('binding')}, unresolved halt is "
              f"{_halt_binding(halt)}")
    return doc["decision"]


# ---------------------------------------------------------------------------
# Stage interface (injected; fixtures implement it in c1)
# ---------------------------------------------------------------------------
class Stages:
    """Interface only. Every method raises V05LTError on failure.
      bind_chain(doc, root)         verify the launch-inputs record (v0.4)
      verify_data()                 PA5-A content check before a session
      epoch0_done(seed) -> bool     verified committed epoch-0 stage
      epoch0(seed)                  epoch-0 stage (+ its evaluation)
      expected_config(seed, arm) -> dict   registered run configuration
      run_dir(seed, arm) -> Path
      train(seed, arm, mode)        mode 'start' | 'resume'
      post_run(seed, arm) -> dict   {'verified': True, 'artefacts': {...}}
    """
    names = ("bind_chain", "verify_data", "epoch0_done", "epoch0",
             "expected_config", "run_dir", "train", "post_run")


class Halted(C.V05LTError):
    pass


class Campaign:
    def __init__(self, root: Path, stages, *, chain: tuple,
                 fixture: bool = False, hook=None):
        prod_cls = isinstance(stages, PS.ProductionStages)
        if not fixture:
            C.require(type(stages) is PS.ProductionStages,
                      "LAUNCH_STAGES_NOT_PRODUCTION", "production requires "
                      "v05lt_prodstages.ProductionStages (exact class), got "
                      f"{type(stages).__name__}")
            C.require(stages.production, "LAUNCH_FIXTURE_IN_PRODUCTION",
                      "production launch with fixture-mode stages refused")
        else:
            C.require(not (prod_cls and stages.production),
                      "LAUNCH_STAGES_MODE_MISMATCH", "fixture campaign with "
                      "production-mode stages refused")
        missing = [n for n in Stages.names if not callable(
            getattr(stages, n, None))]
        C.require(not missing, "LAUNCH_STAGES_INVALID", str(missing))
        self.root, self.stages, self.chain = Path(root), stages, chain
        self.fixture, self.hook = fixture, hook
        self.ledger = C.Ledger(self.root / LEDGER_NAME)

    def _log(self, kind: str, body: dict) -> None:
        self.ledger.append(kind, body)

    def _halt(self, code: str, msg: str, **body) -> Halted:
        """Record and log a halt; returns the exception for the caller to
        raise (`raise self._halt(...)`), so every halt site is explicit."""
        try:
            self._log("halt", {"code": code, "message": msg, **body})
        except Exception as rec_exc:  # noqa: BLE001 -- logged, halt raised
            logger.error("[%s] HALT_RECORD_FAILED: %s", __abbr__, rec_exc)
        logger.error("[failed_validity] %s: %s", code, msg)
        return Halted(code, msg)

    def _call(self, stage: str, fn, *a, **body):
        """Addition C (refinement 5): one boundary for every stage call.
        V05LT errors (incl. Halted) and BaseException pass through; any other
        exception is logged with its traceback and recorded as the campaign
        halt LAUNCH_STAGE_ERROR (needs a bound decision before retry)."""
        try:
            return fn(*a)
        except C.V05LTError:
            raise
        except Exception as exc:  # noqa: BLE001 -- recorded halt, raised
            tb = traceback.format_exc()
            logger.error("[%s] unexpected %s in stage %s:\n%s", __abbr__,
                         type(exc).__name__, stage, tb)
            raise self._halt("LAUNCH_STAGE_ERROR", f"{stage}: "
                             f"{type(exc).__name__}: {exc}", stage=stage,
                             type=type(exc).__name__,
                             exc_message=str(exc)[:500],
                             traceback=tb[-TRACEBACK_MAX:], **body)

    def _campaign_absent(self) -> bool:
        if self.ledger.path.exists():
            return False
        for kind, s, a in stage_order():
            if kind == "run" and Path(self._call(
                    "run_dir", self.stages.run_dir, s, a)).exists():
                return False
            if kind == "epoch0":
                try:
                    if self._call("epoch0_done", self.stages.epoch0_done, s):
                        return False
                except Halted:
                    raise
                except C.V05LTError as exc:
                    # an incomplete / mismatched epoch-0 output exists
                    logger.error("[%s] seed %d epoch-0 outputs exist but "
                                 "are not complete (%s)", __abbr__, s,
                                 exc.code)
                    return False
        return True

    def _verify_chain(self) -> tuple:
        """Launch chain against its approved sha256, then the launch-inputs
        record through stages.bind_chain (refinement 1 of c2 wiring).
        Refusals here produce nothing and are not recorded."""
        doc = verify_launch_chain(*self.chain)
        try:                        # launch-inputs record
            bound = self.stages.bind_chain(doc, self.chain[2])
        except C.V05LTError:
            raise
        except Exception as exc:  # noqa: BLE001 -- typed refusal
            logger.error("[%s] launch inputs: %s", __abbr__,
                         traceback.format_exc())
            C.fail("LAUNCH_INPUTS_ERROR", f"{type(exc).__name__}: {exc}")
        if isinstance(self.stages, PS.ProductionStages):
            want = [e["sha256"] for e in doc["entries"]
                    if e["role"] == PS.LAUNCH_INPUTS_ROLE]
            held = (self.stages.inputs or {}).get("record_sha256")
            C.require(isinstance(bound, dict) and len(want) == 1 and
                      bound.get("launch_inputs_sha256") == want[0] == held,
                      "LAUNCH_INPUTS_UNBOUND", "stages not bound to the "
                      "chain's launch-inputs record")
        return doc, bound

    # -- entry --------------------------------------------------------------
    def run(self, mode: str, decisions: dict | None = None,
            campaign_decision: tuple | None = None) -> dict:
        """mode 'new' | 'resume'; decisions: {run_id: (path, sha256)};
        campaign_decision: (path, sha256) for an unresolved campaign halt."""
        C.require(mode in ("new", "resume"), "LAUNCH_MODE_INVALID", mode)
        self.root.mkdir(parents=True, exist_ok=True)
        with Lock(self.root / LOCK_NAME, "campaign"):
            if self.hook is not None:
                self.hook("campaign:locked")
            doc, bound = self._verify_chain()
            if mode == "new":
                C.require(self._campaign_absent(), "ARTIFACT_EXISTS",
                          f"{self.root}: campaign outputs exist; resume mode "
                          "is explicit")
            else:
                C.require(self.ledger.path.exists(), "LAUNCH_NO_CAMPAIGN",
                          f"{self.root}: nothing to resume")
                self._resolve_campaign_halt(campaign_decision)
            self._log("launch_start", {
                "mode": mode, "chain_sha256": self.chain[1],
                "chain_entries": len(doc["entries"]), "fixture": self.fixture,
                "bound": bound if isinstance(bound, dict) else None,
                "abbr": __abbr__, "version": __version__})
            done = []
            for kind, s, a in stage_order():
                if kind == "epoch0":
                    self._epoch0(s)
                else:
                    self._run(s, a, (decisions or {}).get(run_id(s, a)))
                done.append(f"{kind}:{s}:{a}")
            self._log("campaign_complete", {"stages": done})
            return {"status": "complete", "stages": done}

    def _resolve_campaign_halt(self, decision) -> None:
        hs = campaign_halt_state(self.ledger)
        C.require(not hs["halted"], "LAUNCH_CAMPAIGN_HALTED",
                  f"{self.root}: campaign ended by an owner `halt` decision")
        halt = hs["unresolved"]
        if halt is None:
            return
        dec = check_campaign_decision(decision, halt)
        self._log("halt_resolved", {"halt_seq": halt["seq"],
                                    "decision": dec,
                                    "sha256": decision[1],
                                    "binding": _halt_binding(halt)})
        if dec == "halt":
            C.fail("LAUNCH_CAMPAIGN_HALTED", f"owner `halt` for "
                   f"{halt['body'].get('code')} at seq {halt['seq']}")
        logger.warning("[%s] campaign halt %s (seq %d) resolved by an owner "
                       "`retry` decision", __abbr__, halt["body"].get("code"),
                       halt["seq"])

    # -- stages ---------------------------------------------------------------
    def _epoch0(self, seed: int) -> None:
        try:                       # both verifications inside the handler
            if self._call("epoch0_done", self.stages.epoch0_done, seed,
                          seed=seed):
                return
            self._call("verify_data", self.stages.verify_data, seed=seed)
            self._call("epoch0", self.stages.epoch0, seed, seed=seed)
            verified = self._call("epoch0_done", self.stages.epoch0_done,
                                  seed, seed=seed)
        except Halted:
            raise
        except C.V05LTError as exc:
            raise self._halt("LAUNCH_EPOCH0_FAILED", f"seed {seed}: {exc.code}",
                             seed=seed, cause=exc.code)
        if not verified:
            raise self._halt("LAUNCH_EPOCH0_FAILED", f"seed {seed}: stage "
                             "returned but is not verified", seed=seed)
        self._log("epoch0_done", {"seed": seed})

    def _state(self, seed, arm) -> dict:
        rid = run_id(seed, arm)
        rdir = self._call("run_dir", self.stages.run_dir, seed, arm, run=rid)
        try:
            st = inspect_run(rdir, fixture=self.fixture)
        except C.V05LTError as exc:
            raise self._halt("LAUNCH_RUN_UNREADABLE", f"{rid}: {exc.code}",
                             run=rid, cause=exc.code)
        if st["class"] not in (ABSENT, UNREADABLE):
            try:
                want = config_sha256(self._call(
                    "expected_config", self.stages.expected_config, seed,
                    arm, run=rid))
            except Halted:
                raise
            except C.V05LTError as exc:
                raise self._halt("LAUNCH_CONFIG_FAILED", f"{rid}: "
                                 f"{exc.code}", run=rid, cause=exc.code)
            if st["config_sha256"] != want:
                raise self._halt("LAUNCH_RUN_CONFIG_MISMATCH", f"{rid}: run "
                                 "configuration is not the registered one", run=rid)
        return st

    def _run(self, seed: int, arm: str, decision) -> None:
        rid = run_id(seed, arm)
        rdir = Path(self._call("run_dir", self.stages.run_dir, seed, arm,
                               run=rid))
        rdir.parent.mkdir(parents=True, exist_ok=True)
        with Lock(rdir.parent / f".{rid}.lock", f"run {rid}"):
            st = self._state(seed, arm)
            cls = st["class"]
            mode = None
            if cls == ABSENT:
                mode = "start"
            elif cls == INTERRUPTED:
                mode = "resume"
            elif cls == SESSION_ERROR or cls in (AWAITING, REFUSED):
                try:
                    dec = check_owner_decision(decision, rid, st)
                except C.V05LTError as exc:
                    raise self._halt(exc.code, str(exc), run=rid, state=cls)
                self._log("owner_decision_applied", {
                    "run": rid, "decision": dec, "sha256": decision[1],
                    "binding": _binding(rid, st)})
                if dec == "halt":
                    raise self._halt("LAUNCH_HALTED_BY_OWNER", f"{rid}: owner "
                                     "decision halt", run=rid, state=cls)
                mode = "resume"
            elif cls == TERMINAL_C:
                raise self._halt("LAUNCH_RUN_TERMINAL", f"{rid}: {st['status']} "
                                 "(never resumed)", run=rid, status=st["status"])
            elif cls == UNREADABLE:
                raise self._halt("LAUNCH_RUN_UNREADABLE", f"{rid}: "
                                 f"{st.get('reason')}", run=rid)
            if mode is not None:
                self._train(seed, arm, mode)
                st = self._state(seed, arm)
                if st["class"] != COMPLETED:
                    raise self._halt("LAUNCH_RUN_NOT_COMPLETED", f"{rid}: state "
                                     f"{st['class']} after training", run=rid,
                                     state=st["class"])
            self._advance(seed, arm)

    def _train(self, seed, arm, mode) -> None:
        rid = run_id(seed, arm)
        try:
            self._call("verify_data", self.stages.verify_data, run=rid)
        except Halted:
            raise
        except C.V05LTError as exc:
            raise self._halt("LAUNCH_DATA_UNVERIFIED", f"{rid}: {exc.code}; session "
                             "refused before it started", run=rid, cause=exc.code)
        self._log("session_launch", {"run": rid, "mode": mode})
        try:
            self.stages.train(seed, arm, mode)
        except C.V05LTError as exc:
            logger.error("[%s] %s: trainer stopped with %s", __abbr__, rid,
                         exc.code)
            st = self._state(seed, arm)
            if st["class"] == COMPLETED and exc.code == "RESUME_RUN_ENDED":
                # durable final commit without run_end: the trainer records
                # run_end (recovered) and refuses; the run is complete.
                logger.warning("[%s] %s: run_end recovered; run complete",
                               __abbr__, rid)
                self._log("run_end_recovered", {"run": rid,
                                                "status": st["status"]})
                return
            raise self._halt("LAUNCH_RUN_FAILED", f"{rid}: {exc.code}", run=rid,
                             cause=exc.code, state=st["class"])
        except Exception as exc:  # noqa: BLE001 -- recorded by trainer; halt
            tb = traceback.format_exc()
            logger.error("[%s] session error in %s:\n%s", __abbr__, rid, tb)
            raise self._halt("LAUNCH_SESSION_ERROR", f"{rid}: "
                             f"{type(exc).__name__}", run=rid, stage="train",
                             type=type(exc).__name__,
                             exc_message=str(exc)[:500],
                             traceback=tb[-TRACEBACK_MAX:])

    def _advance(self, seed, arm) -> None:
        """PA5-C: final weights and post-run outputs verified, or halt."""
        rid = run_id(seed, arm)
        try:
            res = self._call("post_run", self.stages.post_run, seed, arm,
                             run=rid)
        except Halted:
            raise
        except C.V05LTError as exc:
            raise self._halt("LAUNCH_POSTRUN_FAILED", f"{rid}: {exc.code}",
                             run=rid, cause=exc.code)
        ok = isinstance(res, dict) and res.get("verified") is True and \
            isinstance(res.get("artefacts"), dict) and bool(res["artefacts"])
        if not ok:
            raise self._halt("LAUNCH_POSTRUN_FAILED", f"{rid}: post-run outputs "
                             "not verified", run=rid)
        self._log("advance", {"run": rid, "artefacts": res["artefacts"]})

    # -- holdout stage (I1.9 c4; separate command, never called by run) ----
    def verify_completion(self) -> tuple:
        """Refinement 1 (c4): verified completion, read-only. The latest
        launch ended in `campaign_complete` with no later halt and no
        unresolved or final campaign halt; each of the six expected run
        identities has an `advance` record in that launch; the current
        evidence of every run is re-verified against that record (run
        completed with the registered configuration, final weights, every
        listed post-run record) and each seed's epoch-0 stage is verified.
        Returns (evidence, evidence_sha256). Refusals are not recorded."""
        from seqref_mri.scripts import v05lt_eval as EV
        from seqref_mri.scripts import v05lt_final as FW
        C.require(self.ledger.path.is_file(), "LAUNCH_HOLDOUT_INCOMPLETE",
                  f"{self.root}: no campaign ledger")
        recs = self.ledger.records()
        hs = campaign_halt_state(self.ledger)
        C.require(not hs["halted"] and hs["unresolved"] is None,
                  "LAUNCH_HOLDOUT_INCOMPLETE", "campaign halt unresolved or "
                  "final (owner `halt`)")
        flow = [r for r in recs if r["kind"] in ("launch_start",
                                                 "campaign_complete", "halt")]
        C.require(bool(flow) and flow[-1]["kind"] == "campaign_complete",
                  "LAUNCH_HOLDOUT_INCOMPLETE", "latest campaign record is "
                  f"{flow[-1]['kind'] if flow else None}, not campaign_complete")
        done = flow[-1]
        start = max(r["seq"] for r in recs if r["kind"] == "launch_start")
        want_stages = [f"{k}:{s}:{a}" for k, s, a in stage_order()]
        C.require(done["body"].get("stages") == want_stages,
                  "LAUNCH_HOLDOUT_INCOMPLETE", "campaign_complete does not "
                  "list the registered stages")
        adv = {}
        for r in recs:
            if start < r["seq"] < done["seq"] and r["kind"] == "advance":
                adv[r["body"].get("run")] = r
        want = [run_id(s, a) for k, s, a in stage_order() if k == "run"]
        C.require(sorted(adv) == sorted(want), "LAUNCH_HOLDOUT_INCOMPLETE",
                  f"advanced runs {sorted(map(str, adv))} != expected run "
                  f"identities {sorted(want)}")
        st = self.stages
        ev = {"complete_seq": done["seq"], "launch_start_seq": start,
              "runs": {}, "epoch0": {}}
        try:
            for kind, s, a in stage_order():
                if kind == "epoch0":
                    C.require(st.epoch0_done(s) is True,
                              "LAUNCH_HOLDOUT_EVIDENCE", f"seed {s}: epoch-0 "
                              "stage not complete")
                    ck = st.epoch0_checkpoint(s)
                    ev["epoch0"][str(s)] = {
                        "file_sha256": ck["file_sha256"],
                        "model_state_sha256": ck["model_state_sha256"]}
                    continue
                rid = run_id(s, a)
                rdir = Path(st.run_dir(s, a))
                rs = inspect_run(rdir, fixture=self.fixture)
                C.require(rs["class"] == COMPLETED, "LAUNCH_HOLDOUT_EVIDENCE",
                          f"{rid}: run state {rs['class']}")
                C.require(rs["config_sha256"] == config_sha256(
                    st.expected_config(s, a)), "LAUNCH_HOLDOUT_EVIDENCE",
                    f"{rid}: configuration is not the registered one")
                arts = adv[rid]["body"].get("artefacts")
                C.require(isinstance(arts, dict), "LAUNCH_HOLDOUT_EVIDENCE",
                          f"{rid}: advance record without artefacts")
                fw = FW.verify_final(rdir, fixture=self.fixture)
                pe = sorted(k for k in arts if k.startswith("posteval/"))
                C.require(set(arts) == {"weights_final.pt",
                                        "final_weights.json", *pe} and pe,
                          "LAUNCH_HOLDOUT_EVIDENCE", f"{rid}: advance "
                          f"artefacts {sorted(arts)}")
                C.require(arts["weights_final.pt"] == fw["sha256"] and
                          arts["final_weights.json"] == fw["record_sha256"],
                          "LAUNCH_HOLDOUT_EVIDENCE", f"{rid}: final weights "
                          "differ from the advance record")
                states = []
                for k in pe:
                    ms = k.split("/", 1)[1]
                    C.require(_is_sha(ms), "LAUNCH_HOLDOUT_EVIDENCE",
                              f"{rid}: artefact name {k}")
                    d = Path(st.root) / k
                    C.require(d.is_dir() and not d.is_symlink(),
                              "LAUNCH_HOLDOUT_EVIDENCE", f"{rid}: {k} missing")
                    C.require(C.sha256_file(d / EV.POSTEVAL_FILE) == arts[k],
                              "LAUNCH_HOLDOUT_EVIDENCE", f"{rid}: {k} differs "
                              "from the advance record")
                    doc = EV.load_posteval(d)
                    C.require(doc.get("checkpoint", {}).get(
                        "model_state_sha256") == ms, "LAUNCH_HOLDOUT_EVIDENCE",
                        f"{rid}: {k} records another model state")
                    states.append(ms)
                C.require(fw["model_state_sha256"] in states,
                          "LAUNCH_HOLDOUT_EVIDENCE", f"{rid}: final weights "
                          "without a post-run record")
                ev["runs"][rid] = {
                    "advance_seq": adv[rid]["seq"],
                    "final_sha256": fw["sha256"],
                    "final_record_sha256": fw["record_sha256"],
                    "final_model_state_sha256": fw["model_state_sha256"],
                    "final_epoch": fw["epoch"], "final_path": str(fw["path"]),
                    "run_dir": str(rdir), "posteval_states": states}
        except C.V05LTError as exc:
            if exc.code == "LAUNCH_HOLDOUT_EVIDENCE":
                raise
            C.fail("LAUNCH_HOLDOUT_EVIDENCE", f"{exc.code}: {exc}")
        except Exception as exc:  # noqa: BLE001 -- typed refusal, logged
            logger.error("[%s] completion evidence:\n%s", __abbr__,
                         traceback.format_exc())
            C.fail("LAUNCH_HOLDOUT_EVIDENCE", f"{type(exc).__name__}: {exc}")
        return ev, C.sha256_bytes(C.canonical_json(ev))

    def _mirror(self, kind: str, body: dict) -> None:
        """Campaign-ledger mirror record (reference only); a write failure
        is a typed error -- the holdout ledger stays authoritative."""
        C.require(kind in HOLDOUT_MIRROR_KINDS, "LAUNCH_HOLDOUT_MIRROR_FAILED",
                  f"unknown mirror kind {kind}")
        try:
            if self.hook is not None:
                self.hook(f"holdout:mirror:{kind}")
            self._log(kind, body)
        except Exception as exc:  # noqa: BLE001 -- typed failure, logged
            logger.error("[%s] campaign mirror %s not written:\n%s", __abbr__,
                         kind, traceback.format_exc())
            C.fail("LAUNCH_HOLDOUT_MIRROR_FAILED", f"{kind}: "
                   f"{type(exc).__name__}: {exc} (holdout ledger is "
                   "authoritative; rerun the holdout command to re-enter)")

    def _mirrored(self, kind: str, key: str, value) -> bool:
        return any(r["kind"] == kind and r["body"].get(key) == value
                   for r in self.ledger.records())

    def _reconcile_halt(self, H, out: Path, approval_sha256: str) -> None:
        """An unresolved holdout halt not yet mirrored (an earlier mirror
        write failed) is mirrored now (`recovered`)."""
        if (out / H.LEDGER_NAME).exists():
            self._mirror_halt(H, out, approval_sha256, recovered=True)

    def _mirror_halt(self, H, out: Path, approval_sha256: str,
                     recovered: bool) -> None:
        """Mirror the holdout ledger's unresolved halt (reference only). A
        mirror-write failure is logged and reconciled on the next
        invocation; the halt itself stays in the authoritative ledger."""
        hs = H._halt_state(C.Ledger(out / H.LEDGER_NAME))
        if hs["unresolved"] is None:
            return
        b = H._halt_binding(hs["unresolved"])
        if self._mirrored("holdout_halt", "halt_line_sha256",
                          b["halt_line_sha256"]):
            return
        try:
            self._mirror("holdout_halt", {**b, "recovered": recovered,
                                          "approval": approval_sha256})
        except C.V05LTError as mexc:
            logger.error("[%s] holdout halt not mirrored (%s); reconciled on "
                         "the next invocation", __abbr__, mexc.code)

    def _reentry_verify(self, H, gate, out: Path, comp: dict,
                        where: dict) -> tuple:
        """Hash verification of a completed holdout output against the
        holdout ledger's completion record. `where['task']` names the task
        directory of a failing record (None for the index)."""
        A = H._mods()[0]
        RS = H._mods()[5]
        isha = comp["body"].get("index_sha256")
        idx = out / H.INDEX_NAME
        C.require(idx.is_file() and not idx.is_symlink() and
                  C.sha256_file(idx) == isha and C.read_sidecar(
                      Path(str(idx) + ".sha256"),
                      "LAUNCH_HOLDOUT_REENTRY") == isha,
                  "LAUNCH_HOLDOUT_REENTRY", f"{idx}: index differs from the "
                  "holdout ledger's completion record")
        doc = json.loads(idx.read_bytes())
        want = {f"{a}|{s}|{r}" for a in A.SELECTION_ARMS
                for s in A.SELECTION_SEEDS for r in RS.HOLDOUT_ROLES}
        C.require(doc.get("approval_sha256") == gate.approval_sha256 and
                  doc.get("manifest_content_sha256") ==
                  gate.manifest["content_sha256"] and
                  set(doc.get("records", {})) == want,
                  "LAUNCH_HOLDOUT_REENTRY", "index bindings / record set")
        for key, rec in sorted(doc["records"].items()):
            p = out / rec["path"]
            where["task"] = Path(rec["path"]).parent.name or None
            C.require(p.is_file() and not p.is_symlink() and
                      C.sha256_file(p) == rec["sha256"],
                      "LAUNCH_HOLDOUT_REENTRY", f"{key}: {rec['path']} "
                      "differs from the index")
        where["task"] = None
        return isha, doc, want

    def _reentry(self, H, gate, out: Path, ev_sha: str, data_root: Path):
        """Re-entry after completion: the holdout ledger's latest gate is
        followed by `complete`; the index and every record are verified by
        sha256 against it; a missing `holdout_done` mirror is written. No
        re-evaluation, no holdout MRI file opened. None -> run the stage.
        Revision 2: a verification failure is a c3 output mismatch -- a
        halt recorded in the holdout ledger (stage `reentry`, the failing
        task, cause) through c3's own Runner._halt and mirrored by
        reference, so restoring the files alone never resumes; continuation
        needs the c3 decision bound to that halt."""
        led = C.Ledger(out / H.LEDGER_NAME)
        if not led.path.exists():
            return None
        recs = led.records()
        hs = H._halt_state(led)
        comp = [r for r in recs if r["kind"] == "complete"]
        gates = [r for r in recs if r["kind"] == "gate_passed"]
        if hs["unresolved"] is not None or hs["halted"] or not comp or \
                gates[-1]["seq"] > comp[-1]["seq"]:
            return None
        C.require(set(hs["approvals"]) == {gate.approval_sha256},
                  "HOLDOUT_APPROVAL_CHANGED", f"{out}: completed under "
                  f"{sorted(a[:12] for a in hs['approvals'])}; one output "
                  "directory per approval")
        where = {"task": None}
        try:
            isha, doc, want = self._reentry_verify(H, gate, out, comp[-1],
                                                   where)
        except C.V05LTError as exc:
            logger.error("[%s] re-entry verification failed (%s); recording "
                         "a holdout halt", __abbr__, exc.code)
            hx = H.Runner(gate, out, data_root)._halt(
                "HOLDOUT_OUTPUT_MISMATCH", f"reentry: {exc.code}: {exc}",
                stage="reentry", task=where["task"], cause=exc.code)
            self._mirror_halt(H, out, gate.approval_sha256, recovered=False)
            raise hx from exc
        except Exception as exc:  # noqa: BLE001 -- recorded halt, raised
            tb = traceback.format_exc()
            logger.error("[%s] unexpected %s during re-entry verification "
                         "(task %s):\n%s", __abbr__, type(exc).__name__,
                         where["task"], tb)
            hx = H.Runner(gate, out, data_root)._halt(
                "HOLDOUT_STAGE_ERROR", f"reentry: {type(exc).__name__}: "
                f"{exc}", stage="reentry", task=where["task"],
                type=type(exc).__name__, exc_message=str(exc)[:500],
                traceback=tb[-TRACEBACK_MAX:])
            self._mirror_halt(H, out, gate.approval_sha256, recovered=False)
            raise hx from exc
        if not self._mirrored("holdout_done", "index_sha256", isha):
            self._mirror("holdout_done", {
                "approval_sha256": gate.approval_sha256,
                "index_sha256": isha, "records": len(want),
                "holdout_complete_seq": comp[-1]["seq"],
                "evidence_sha256": ev_sha, "recovered": True})
        logger.info("[%s] holdout stage already complete (index %s); "
                    "verified by hash, nothing re-evaluated", __abbr__,
                    isha[:12])
        return {"status": "complete", "reentry": True, "index_sha256": isha,
                "records": {k: {"path": str(out / v["path"]),
                                "sha256": v["sha256"]}
                            for k, v in doc["records"].items()}}

    def holdout(self, approval: tuple, *, analysis_chain: Path,
                manifest_path: Path, data_root: Path, out_dir: Path,
                root: Path, decision: tuple | None = None,
                launch_root: Path | None = None,
                hooks: dict | None = None) -> dict:
        """The holdout stage through the launcher (c4). approval = holdout
        approval (PATH, externally approved sha256) -- a distinct input from
        the launch approval (self.chain). Order: campaign lock -> launch
        chain + launch inputs -> verified completion -> approval names the
        approved launch chain -> c3 gate (unchanged) -> selection bound to
        the verified campaign -> c3 output guard -> holdout lock ->
        re-entry / halt reconciliation -> holdout_start -> c3 Runner ->
        holdout_done (or holdout_halt). No holdout MRI file is opened before
        the c3 gate is recorded."""
        from seqref_mri.scripts import v05lt_holdout as H
        C.require(type(self.stages) is PS.ProductionStages or (
            self.fixture and isinstance(self.stages, PS.ProductionStages)),
            "LAUNCH_HOLDOUT_STAGES", "the holdout stage needs "
            "v05lt_prodstages.ProductionStages")
        C.require(self.root.is_dir() and not self.root.is_symlink(),
                  "LAUNCH_NO_CAMPAIGN", f"{self.root}: no campaign")
        with Lock(self.root / LOCK_NAME, "campaign"):
            if self.hook is not None:
                self.hook("holdout:locked")
            self._verify_chain()
            ev, ev_sha = self.verify_completion()
            ap = H.load_approval(Path(approval[0]), approval[1],
                                 bool(self.stages.production))
            C.require(ap["launch_chain_sha256"] == self.chain[1],
                      "LAUNCH_HOLDOUT_CHAIN_MISMATCH", "holdout approval "
                      f"names launch chain {ap['launch_chain_sha256'][:12]}, "
                      f"launcher approved {self.chain[1][:12]}")
            gate = H.verify_gate(approval, root=root,
                                 launch_chain=self.chain[0],
                                 analysis_chain=analysis_chain,
                                 manifest_path=manifest_path,
                                 stages=self.stages, launch_root=launch_root)
            bind_selection(gate.selection, ev, Path(root))
            out = H.check_out_dir(out_dir, data_root, self.stages.paths,
                                  gate.production)
            if self.hook is not None:
                self.hook("holdout:bound")
            out.mkdir(parents=True, exist_ok=True)
            with Lock(out / H.LOCK_NAME, "holdout"):
                self._reconcile_halt(H, out, gate.approval_sha256)
                again = self._reentry(H, gate, out, ev_sha, data_root)
                if again is not None:
                    return again
                self._mirror("holdout_start", {
                    "approval_sha256": gate.approval_sha256,
                    "launch_chain_sha256": self.chain[1],
                    "analysis_chain_sha256": gate.analysis_chain_sha256,
                    "selection_sha256": gate.selection_sha256,
                    "evidence_sha256": ev_sha,
                    "decision_sha256": decision[1] if decision else None})
                try:
                    res = H.Runner(gate, out, data_root,
                                   hooks=hooks).run(decision)
                except H.Halted as hexc:
                    logger.error("[%s] holdout stage halted (%s); mirroring "
                                 "the halt reference", __abbr__, hexc.code)
                    self._mirror_halt(H, out, gate.approval_sha256,
                                      recovered=False)
                    raise
                self._mirror("holdout_done", {
                    "approval_sha256": gate.approval_sha256,
                    "index_sha256": res["index_sha256"],
                    "records": len(res["records"]),
                    "holdout_complete_seq": [r for r in C.Ledger(
                        out / H.LEDGER_NAME).records()
                        if r["kind"] == "complete"][-1]["seq"],
                    "evidence_sha256": ev_sha, "recovered": False})
                return res


def bind_selection(sel: dict, ev: dict, root: Path) -> None:
    """Refinement 1 (c4): the approved selection belongs to the verified
    campaign. Per run: final = the verified final-weights file (path, file
    sha256, model state, epoch); selected and epoch-3 model states among
    that run's verified post-run states, files inside that run directory.
    Per seed: epoch 0 = the verified epoch-0 checkpoint."""
    bad = []
    root = Path(root)
    for key, run in sorted(sel["runs"].items()):
        arm, seed = key.split("|")
        e = ev["runs"].get(run_id(int(seed), arm))
        if e is None:
            bad.append(f"{key}: not a verified run")
            continue
        fin = run["final"]
        if not (fin["sha256"] == e["final_sha256"] and
                fin["model_state_sha256"] == e["final_model_state_sha256"]
                and fin["epoch"] == e["final_epoch"] and
                (root / fin["file"]).resolve() ==
                Path(e["final_path"]).resolve()):
            bad.append(f"{key} final")
        rdir = Path(e["run_dir"]).resolve()
        for role in ("epoch3", "selected"):
            ck = run[role]
            if ck["model_state_sha256"] not in e["posteval_states"] or \
                    rdir not in (root / ck["file"]).resolve().parents:
                bad.append(f"{key} {role}")
    for s, ck in sorted(sel["epoch0"].items()):
        e0 = ev["epoch0"].get(str(s))
        if e0 is None or ck["sha256"] != e0["file_sha256"] or \
                ck["model_state_sha256"] != e0["model_state_sha256"]:
            bad.append(f"epoch0 {s}")
    C.require(not bad, "LAUNCH_HOLDOUT_SELECTION_UNBOUND", "approved "
              f"selection is not the verified campaign's: {bad[:6]}")


def holdout_status(root: Path, out_dir: Path) -> dict:
    """Addition C (c4): read-only holdout state -- campaign-ledger mirror
    records and the holdout ledger summary. Creates no file or directory
    and opens no MRI data."""
    from seqref_mri.scripts import v05lt_holdout as H
    out = {"mirror": {}, "holdout": None}
    led = C.Ledger(Path(root) / LEDGER_NAME)
    if led.path.is_file():
        for r in led.records():
            if r["kind"] in HOLDOUT_MIRROR_KINDS:
                out["mirror"][r["kind"]] = {"seq": r["seq"], **{
                    k: r["body"].get(k) for k in (
                        "approval_sha256", "index_sha256", "halt_code",
                        "halt_seq", "recovered") if k in r["body"]}}
    if (Path(out_dir) / H.LEDGER_NAME).is_file():
        out["holdout"] = H.status(Path(out_dir))
    return out


def status(root: Path, stages, *, fixture: bool = False) -> list:
    """Read-only overview of every stage."""
    out = []
    for kind, s, a in stage_order():
        if kind == "epoch0":
            try:
                out.append({"stage": f"epoch0:{s}",
                            "done": bool(stages.epoch0_done(s))})
            except C.V05LTError as exc:     # reported, not swallowed
                logger.error("[%s] status epoch0:%d: %s", __abbr__, s,
                             exc.code)
                out.append({"stage": f"epoch0:{s}", "done": False,
                             "error": exc.code})
        else:
            out.append({"stage": run_id(s, a), **inspect_run(
                stages.run_dir(s, a), fixture=fixture)})
    return out


def _pathsha(s: str, what: str) -> tuple:
    path, sep, sha = s.rpartition(":")
    C.require(bool(sep) and bool(path) and _is_sha(sha), "LAUNCH_ARGS_INVALID",
              f"{what}: expected PATH:SHA256, got {s!r}")
    return (Path(path), sha)


def main(argv=None) -> int:
    """Production entry (195 §8.3 step 5). Requires the launch chain and its
    approved sha256 (recorded in V05LTEXEC); decisions as PATH:SHA256."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=f"{__abbr__} v{__version__}")
    ap.add_argument("mode", choices=("new", "resume", "status", "holdout"))
    ap.add_argument("--chain", help="launch-chain file")
    ap.add_argument("--approved-sha256", help="approved launch-chain sha256")
    ap.add_argument("--decision", action="append", default=[],
                    help="RUN_ID=PATH:SHA256 (owner decision)")
    ap.add_argument("--campaign-decision", help="PATH:SHA256")
    ap.add_argument("--holdout-approval", help="holdout: PATH:SHA256 (the "
                    "approval sha256 recorded in 196; distinct from the "
                    "launch approval)")
    ap.add_argument("--analysis-chain", help="holdout: analysis-chain file")
    ap.add_argument("--holdout-decision", help="holdout: PATH:SHA256 "
                    "(decision bound to the unresolved holdout halt)")
    a = ap.parse_args(argv)
    try:
        paths = C.Paths()
        stages = PS.ProductionStages(paths, paths.data_root)
        if a.mode == "status":
            from seqref_mri.scripts import v05lt_holdout as H
            for row in status(paths.v05lt, stages):
                print(json.dumps(row, sort_keys=True))
            print(json.dumps({"holdout_state": holdout_status(
                paths.v05lt, H.registered_out(paths))}, sort_keys=True))
            return 0
        C.require(bool(a.chain) and bool(a.approved_sha256),
                  "LAUNCH_CHAIN_UNAPPROVED", "--chain and --approved-sha256 "
                  "are required (195 §8.3)")
        if a.mode == "holdout":
            from seqref_mri.scripts import v05lt_holdout as H
            C.require(bool(a.holdout_approval) and bool(a.analysis_chain),
                      "LAUNCH_ARGS_INVALID", "holdout needs "
                      "--holdout-approval PATH:SHA256 and --analysis-chain")
            camp = Campaign(paths.v05lt, stages,
                            chain=(Path(a.chain), a.approved_sha256,
                                   paths.repo), fixture=False)
            res = camp.holdout(
                _pathsha(a.holdout_approval, "--holdout-approval"),
                analysis_chain=Path(a.analysis_chain),
                manifest_path=H.registered_manifest(paths),
                data_root=paths.data_root, out_dir=H.registered_out(paths),
                root=paths.repo,
                decision=(_pathsha(a.holdout_decision, "--holdout-decision")
                          if a.holdout_decision else None))
            print(f"HOLDOUT RESULT {res['status']} index="
                  f"{res['index_sha256']} records={len(res['records'])}"
                  f"{' (re-entry)' if res.get('reentry') else ''}")
            return 0
        decs = {}
        for d in a.decision:
            rid, sep, rest = d.partition("=")
            C.require(bool(sep), "LAUNCH_ARGS_INVALID", f"--decision {d!r}")
            decs[rid] = _pathsha(rest, "--decision")
        cdec = (_pathsha(a.campaign_decision, "--campaign-decision")
                if a.campaign_decision else None)
        camp = Campaign(paths.v05lt, stages,
                        chain=(Path(a.chain), a.approved_sha256, paths.repo),
                        fixture=False)
        out = camp.run(a.mode, decs, cdec)
    except C.V05LTError as exc:
        print(f"LAUNCHER REFUSED {exc.code}")
        return 2
    except Exception as exc:  # noqa: BLE001 -- logged, never swallowed
        logger.error("[%s] UNEXPECTED %s:\n%s", __abbr__, type(exc).__name__,
                     traceback.format_exc())
        print(f"LAUNCHER FAILED UNEXPECTED {type(exc).__name__}")
        return 2
    print(f"LAUNCH RESULT {out['status']} stages={len(out['stages'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
