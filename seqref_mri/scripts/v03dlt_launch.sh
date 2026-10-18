#!/usr/bin/env bash
# SEQREF-MRI-V03DLTL v0.2 -- scripts/v03dlt_launch.sh
# LIFETIME: KEEP
# =============================================================================
# Purpose: V03DLT campaign launcher (SEQREF-MRI-V03DLT v1.3, Article 190,
#   DESIGN FROZEN 2026-10-18 + AM1-AM3; V03DLTPLAN v1.2, Article 189, B4).
#   Runs the six (arm, seed) pairs sequentially in the locked seed-major
#   order (190 A3), one uninterrupted v03dlt_train.py --mode run invocation
#   per attempt, inside tmux, under sleep inhibition and an exclusive launch
#   lock, with a fresh preflight before every run and ledger-policy retries
#   only.
# Adapted from SEQREF-MRI-V04DLTL v0.2 (pin 9190735; verified archive
#   a473c4af...). NSF differences: V03 ledger/record format and helpers
#   (V03T + v03dlt_train order/attempt functions), arms {0, 2}, no gate
#   argument to the trainer (the gate record is still hash-checked), V03
#   evidence bound by registered hashes, and the 190 AM3 checks: main
#   checkout on branch seqref-v03dlt, HEAD = pin, clean status, registered
#   .git/info/exclude hash. The trainer itself enforces import origin at
#   startup (refusal -> no ledger entry -> TRAINER_NO_LEDGER_ENTRY stop).
# Usage (each a separate owner action, from a tmux session):
#   v03dlt_launch.sh --mode new|resume|ack-stop --commit-pin <40-hex> \
#     --repo /home/benjamin/CSMFII --data-root <fastMRI root> \
#     --results-root /home/benjamin/CSMFII/seqref_mri/results
#   Campaign root = <results-root>/V03DLT_<pin8>.
# Modes: new (root must not exist; created once) | resume (root must exist;
#   read-only verification under the lock first) | ack-stop (acknowledge a
#   STOP request by hash; renames, never deletes; no training).
# STOP file <root>/STOP: the current run finishes normally, then the
#   launcher stops before the next. SIGINT/SIGTERM: ONE SIGINT forwarded to
#   the trainer (own session), but only once /proc/<pid>/status shows the
#   trainer CATCHES SIGINT; until then the cancellation is pending. The
#   launcher waits, then checks the ledger for interrupted/KEYBOARD_INTERRUPT
#   (else CANCEL_LEDGER_MISMATCH). STOP and cancellation are re-checked at
#   the launch boundary, immediately before the trainer is spawned.
# Evidence per session <S> (UTC stamp): launcher_session_<S>.log (+sidecar
#   after its last write), ledger_snapshot_<S>.json (+ own sidecar),
#   campaign_manifest_<S>.sha256 (+sidecar) -- written on a HANDLED exit
#   only; a hard kill prevents them and the next resume stops for review.
# tmux and systemd-inhibit reduce, but do not remove, the risk of loss on
#   shutdown, power loss or a logout policy that kills user processes.
# D2: CKPT_BYTES = 1,026,586,538 (measured on host 2026-10-18). Per attempt
#   the launcher requires 30 x CKPT_BYTES x 1.5 + 64 MiB (about 46.3 GB);
#   a new campaign requires six times that (about 277 GB). An empty value
#   refuses with D2_NOT_REGISTERED (kept as a guard, not a fallback).
# CONVENTION: every failure is logged with a typed code; exit 0 only for a
#   completed campaign or an acknowledged-STOP pause / ack-stop; exit 2
#   otherwise. No fallback, no silent pass. The trainer command below is
#   the ONLY line the B2.10 fixture replaces (plus PY); there is no option
#   to change it.
# Changelog (NEW in v0.1, 2026-10-18):
#   * Introduced under V03DLTPLAN v1.2 B1/B4, adapted from V04DLTL v0.2 as
#     listed above (same new/resume/ack-stop, lock, inhibition, STOP,
#     single-SIGINT cancellation, session evidence and hard-kill detection).
# Changelog (v0.2, 2026-10-18, review fixes R1/R2):
#   * Disk (B3): campaign-level allowance at session start for every
#     unsettled run (6 for a new campaign), plus the per-attempt re-check
#     before every run.
#   * Trainer exit code must agree with its ledger entry (rc 0 <=>
#     completed); any contradiction stops with TRAINER_EXIT_MISMATCH.
#   * Host constants delimited by markers so synthetic fixtures can supply
#     isolated inputs (see fixtures B2.10; 190 AM4).
#   * CKPT_BYTES registered from the host measurement (D2).
# =============================================================================
set -uo pipefail

readonly ABBR="SEQREF-MRI-V03DLTL"
readonly VERSION="0.2"
# >>> HOST CONSTANTS (evidence is hash-verified before every run) >>>>>>>>>>>>
readonly PY="/home/benjamin/CSMFII/seqref_mri/.venv_mri/bin/python"
readonly RES="/home/benjamin/CSMFII/seqref_mri/results"
readonly P4="$RES/_diag/p4/scaling_statistics.json"
readonly P4_SHA="db0bf2c2e0cd3bd5e1ed86bf208198199f16df5b2fbbe0cc72903c8791f85684"
readonly PROBE="$RES/v03_probe/v03_probe_record.json"
readonly PROBE_SHA="7706e68aaf8396462ccd932ffc377cf833e277b449ed6be7b9100aa7e23f3da8"
readonly GATE="$RES/v03_gate/v03_gate_record.json"
readonly GATE_SHA="74629e77788f85cf2c84e902c6497dfab3b004a3e29ae2b9e85b611094cd6351"
readonly BRANCH="seqref-v03dlt"
readonly EXCLUDE_SHA="d2fd48207fab9a37506c111978184c9b0ba896d760deb65a1c0d2da697a58967"
readonly -a V03_EVIDENCE=(
  "v03_train/v03_run_ledger.json a0f996353daffedb5397b250d9320aabd00e7325c25dd21ff0d75b1f2a4c2256"
  "v03_train/v03_arm0_seed101_attempt1_train_record.json 729e15b9e2dc22909d3891a66d33c7e8eb99636b9737b7603ad3646d5d0ce960"
  "v03_train/v03_arm0_seed202_attempt1_train_record.json a05a30d4299a68080e59bd02b76bc819b413eb36b4f44835bae5d3fb8e025cda"
  "v03_train/v03_arm0_seed303_attempt1_train_record.json f2ce82acde7265812f244d9f115ef2dccfc051b2f7d9a32734ab387d9f7137e6"
  "v03_train/v03_arm2_seed101_attempt1_train_record.json e831151a089d8203e800fc8323d81dd905e9dc97fd722915ff33373229971f54"
  "v03_train/v03_arm2_seed202_attempt1_train_record.json 6b89e17f103d9366779fd5409eba7ee9bfb6a84e47d85a6ff48200fee918c8ce"
  "v03_train/v03_arm2_seed303_attempt1_train_record.json 2883e4080278d69dd457eb145ed5062d4b830bade64e1cc074caa49b07569a27"
)
readonly -a REGISTERED_MODULES=(
  "seqref_mri/src/fastmri_data.py 610cc1d1d7968deebc88f645270e1baefb6589cb56841bdd327450ca1069cb44"
  "seqref_mri/src/free_flow_runtime.py 8e171b2c521d77c8ad22b21d02537be160d15166aecb9e0319fd0a25b69823c9"
  "seqref_mri/src/flows/v03_objective.py e2ab02d5e95b9b736b689c021fdd3cd07e1d46bb74f77ad80544aa9785b8cd26"
  "seqref_mri/scripts/v03_train.py 619b0f2ca061c42c86ee7004509584c04cf8ae7e641ed25d6a762227530cc558"
  "seqref_mri/scripts/v03_probe.py ac4997e7376e03cbf3cb6aaae0336153b2008780ab943c094f76431f44412026"
  "seqref_mri/scripts/v03_eval.py a013bab62712bf2f9101f5cc498858b46d8eae0b07f50d89fb8b937b26f0dab8"
  "seqref_mri/scripts/train_base.py 2ae948663dbd53df32e032c0ef8c54e23666106529ca1906a354fcee7ba6ffc5"
  "seqref_mri/scripts/train_free_flow.py 32ec331ff79f3daf951f584b972889db909d313b893e9eaaa5c3697f1e165e3e"
  "seqref_mri/scripts/v02_manifests.py 5ef4c55121df0e5fa151c920bcafb75c55121c222afb7f0d73b19cdbc44749e9"
  "seqref_mri/scripts/v02_train.py ff65e0a7de4c61e3cce610592130de4cb8a9119f94fa2920dd218f863b22c822"
)
readonly CKPT_BYTES=1026586538                # D2: v03_arm0_seed101_attempt1_epoch3.pt, measured on host 2026-10-18
readonly INHIBIT_CMD=(systemd-inhibit --what=sleep:idle --who=V03DLT --why="V03DLT training campaign" --mode=block sleep infinity)
# <<< HOST CONSTANTS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
TRAINER_CMD=("$PY" -m seqref_mri.scripts.v03dlt_train)

MODE="" PIN="" REPO="" DATA_ROOT="" RESULTS_ROOT=""
ROOT="" S="" LOG="" PRELOG="" IPID="" TPID="" CANCEL=0 FORWARDED=0 SESSION_OPEN=0 PENDING_LOGGED=0

say() { printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$ABBR" "$*" >&2; }
log() {
  local line; line="$(date -u +%FT%TZ) $ABBR $*"
  printf '%s\n' "$line" >&2
  if [[ -n "$LOG" ]]; then printf '%s\n' "$line" >> "$LOG"; else PRELOG+="$line"$'\n'; fi
}
die_early() { say "ERROR ${1:-HELPER_FAILED}: $2"; exit 2; }   # before any session file exists
helper_code() { sed -n 's/^ERROR \([A-Z_0-9]*\):.*/\1/p' <<<"$1" | head -1; }

helper() {  # embedded Python helper; reuses the pinned V03T + V03DLT ledger functions
  ( cd "$REPO" && PYTHONPATH="$REPO" "$PY" -P - "$@" <<'PY'
import hashlib, json, re, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
from seqref_mri.scripts import v03_train as v03t
from seqref_mri.scripts import v03dlt_train as dlt
from seqref_mri.scripts.v02_manifests import canonical_json

def out(*a): print(*a); sys.stdout.flush()
def fail(code, msg): print(f"ERROR {code}: {msg}", file=sys.stderr); sys.exit(2)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
ART = re.compile(r"^v03_arm(\d+)_seed(\d+)_attempt(\d+)_")
LED = dlt.LEDGER_NAME
EXCL = (LED, LED + ".sha256", "STOP", ".launch.lock")

def ledger(root):
    try:
        return v03t.load_ledger(root / LED)
    except v03t.V03TError as exc:
        fail(exc.code, str(exc))

def orphans(root, led):
    known = {(e["arm"], e["seed"], e["attempt"]) for e in led["entries"]}
    bad = []
    for p in sorted(root.iterdir()):
        m = ART.match(p.name)
        if m and tuple(int(g) for g in m.groups()) not in known:
            bad.append(p.name)
    if bad:
        fail("ORPHANED_ATTEMPT_ARTIFACTS", f"files with no ledger entry: {bad}")

def check_sidecars(root):
    for side in sorted(root.glob("*.sha256")):
        for ln in side.read_text().splitlines():
            parts = ln.split()
            if len(parts) != 2:
                fail("SIDECAR_MALFORMED", f"{side.name}: {ln!r}")
            target = root / parts[1]
            if not target.is_file() or sha(target) != parts[0]:
                fail("SIDECAR_MISMATCH", f"{side.name}: {parts[1]}")

def check_snapshots(root, led):
    meta = {k: v for k, v in led.items() if k != "entries"}
    for snap in sorted(root.glob("ledger_snapshot_*.json")):
        s = json.loads(snap.read_text())
        n = len(s.get("entries", []))
        if {k: v for k, v in s.items() if k != "entries"} != meta \
                or s["entries"] != led["entries"][:n]:
            fail("LEDGER_PREFIX_MISMATCH", f"{snap.name} is not an entry-level prefix of the live ledger")

def check_sessions(root):
    for lg in sorted(root.glob("launcher_session_*.log")):
        st = lg.name[len("launcher_session_"):-len(".log")]
        for need in (f"campaign_manifest_{st}.sha256", f"ledger_snapshot_{st}.json",
                     f"launcher_session_{st}.log.sha256"):
            if not (root / need).is_file():
                fail("MISSING_SESSION_MANIFEST", f"session {st} lacks {need} (hard kill?)")

cmd, root = sys.argv[1], Path(sys.argv[2])
if cmd == "verify":
    led = ledger(root)
    if (root / "STOP").exists():
        fail("STOP_PRESENT", "acknowledge the STOP request first (--mode ack-stop)")
    check_sessions(root); check_sidecars(root); check_snapshots(root, led); orphans(root, led)
    out("VERIFY_PASS", len(led["entries"]))
elif cmd == "next":
    led = ledger(root)
    nxt = dlt.expected_next_run_dlt(led)
    if nxt is None:
        out("DONE")
    else:
        arm, seed = nxt
        try:
            att = dlt.next_attempt_dlt(led, arm, seed)
        except v03t.V03TError as exc:
            out("BLOCKED", "EXHAUSTED_ATTEMPTS" if exc.code == "RERUN_POLICY_VIOLATION" else exc.code)
            sys.exit(0)
        out("RUN", arm, seed, att)
elif cmd == "remaining":
    led = ledger(root)
    settled = sum(1 for seed, arm in dlt.ORDER_DLT
                  if any(e["arm"] == arm and e["seed"] == seed
                         and e["status"] in v03t.TERMINAL_STATUSES
                         for e in led["entries"]))
    out("REMAINING", len(dlt.ORDER_DLT) - settled)
elif cmd == "preflight":
    led = ledger(root); orphans(root, led)
    arm, seed, att = (int(x) for x in sys.argv[3:6])
    hits = sorted(p.name for p in root.glob(f"v03_arm{arm}_seed{seed}_attempt{att}_*"))
    if hits:
        fail("ORPHANED_ATTEMPT_ARTIFACTS", f"attempt {att} already has {hits}")
    out("PREFLIGHT_LEDGER_PASS")
elif cmd == "latest":
    led = ledger(root)
    arm, seed, att = (int(x) for x in sys.argv[3:6])
    hit = [e for e in led["entries"] if (e["arm"], e["seed"], e["attempt"]) == (arm, seed, att)]
    out("NONE" if not hit else f"{hit[-1]['status']} {hit[-1]['error_code']}")
elif cmd == "stopseen":
    st = sys.argv[3]
    req, seen = root / f"stop_request_{st}.txt", root / f"stop_seen_{st}.txt"
    if not req.exists():
        shutil.copyfile(root / "STOP", req)
        seen.write_text(f"first_seen_utc={now()}\nstop_sha256={sha(req)}\nsession={st}\n")
    out("STOP_RECORDED", sha(req))
elif cmd == "ackstop":
    stop = root / "STOP"
    if not stop.is_file():
        fail("NO_STOP_REQUEST", "no STOP file to acknowledge")
    h = sha(stop)
    match = None
    for req in sorted(root.glob("stop_request_*.txt")):
        st = req.name[len("stop_request_"):-len(".txt")]
        seen = root / f"stop_seen_{st}.txt"
        if sha(req) == h and seen.is_file() and f"stop_sha256={h}" in seen.read_text():
            match = st
    if match is None:
        fail("STOP_CHANGED", "STOP does not match any preserved stop_request copy")
    a = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest, ack = root / f"stop_acknowledged_{a}.txt", root / f"stop_ack_{a}.txt"
    if dest.exists() or ack.exists():
        fail("ARTIFACT_EXISTS", f"{dest.name} or {ack.name} exists")
    stop.rename(dest)
    ack.write_text(f"acknowledged_utc={now()}\nrequest_session={match}\nstop_sha256={h}\n")
    ack.with_name(ack.name + ".sha256").write_text(f"{sha(ack)}  {ack.name}\n")
    out("ACK_STOP_PASS", match, h)
elif cmd == "snapshot":
    st = sys.argv[3]
    snap = root / f"ledger_snapshot_{st}.json"
    if snap.exists() or snap.with_name(snap.name + ".sha256").exists():
        fail("ARTIFACT_EXISTS", snap.name)
    live = root / LED
    if live.exists():
        led = ledger(root)
        shutil.copyfile(live, snap)
        want = (root / (LED + ".sha256")).read_text().split()[0]
        if sha(snap) != want:
            fail("SNAPSHOT_MISMATCH", "snapshot bytes != live ledger sidecar")
    else:
        snap.write_bytes(canonical_json({"schema": v03t.LEDGER_SCHEMA, "entries": []}))
    snap.with_name(snap.name + ".sha256").write_text(f"{sha(snap)}  {snap.name}\n")
    out("SNAPSHOT_PASS", sha(snap))
elif cmd == "manifest":
    st = sys.argv[3]
    man = root / f"campaign_manifest_{st}.sha256"
    if man.exists():
        fail("ARTIFACT_EXISTS", man.name)
    names = sorted(p.name for p in root.iterdir() if p.is_file()
                   and p.name not in EXCL and not p.name.startswith("campaign_manifest_"))
    man.write_text("".join(f"{sha(root / n)}  {n}\n" for n in names))
    man.with_name(man.name + ".sha256").write_text(f"{sha(man)}  {man.name}\n")
    out("MANIFEST_PASS", len(names), sha(man))
else:
    fail("HELPER_COMMAND_UNKNOWN", cmd)
PY
  )
}

# --- arguments ----------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --commit-pin) PIN="${2:-}"; shift 2 ;;
    --repo) REPO="${2:-}"; shift 2 ;;
    --data-root) DATA_ROOT="${2:-}"; shift 2 ;;
    --results-root) RESULTS_ROOT="${2:-}"; shift 2 ;;
    *) die_early ARGS_INVALID "unknown argument $1" ;;
  esac
done
[[ "$MODE" =~ ^(new|resume|ack-stop)$ ]] || die_early ARGS_INVALID "--mode new|resume|ack-stop required"
[[ "$PIN" =~ ^[0-9a-f]{40}$ ]] || die_early ARGS_INVALID "--commit-pin must be a 40-hex commit"
[[ -n "$REPO" && -d "$REPO" ]] || die_early ARGS_INVALID "--repo must be an existing directory"
[[ -n "$RESULTS_ROOT" && -d "$RESULTS_ROOT" ]] || die_early ARGS_INVALID "--results-root must be an existing directory"
REPO="$(cd "$REPO" && pwd)"
ROOT="$(cd "$RESULTS_ROOT" && pwd)/V03DLT_${PIN:0:8}"
case "$ROOT" in
  */v03_train/*|*/v03_probe/*|*/v03_gate/*|*/v03_eval/*|*/v03_classify/*|*/v03_figures/*)
    die_early V03_EVIDENCE_PATH_REFUSED "$ROOT" ;;
esac

if [[ "$MODE" != "ack-stop" ]]; then
  [[ -n "${TMUX:-}" ]] || die_early NOT_IN_TMUX "start the launcher inside a tmux session"
  [[ -n "$DATA_ROOT" && -d "$DATA_ROOT" ]] || die_early ARGS_INVALID "--data-root must be an existing directory"
  [[ -n "$CKPT_BYTES" ]] || die_early D2_NOT_REGISTERED "CKPT_BYTES (D2) not registered; measure on host before the pin"
fi
NEED_BYTES=0                                  # one attempt: 30 checkpoints x 1.5 + records
[[ -z "$CKPT_BYTES" ]] || NEED_BYTES=$(( CKPT_BYTES * 30 * 3 / 2 + 64 * 1024 * 1024 ))

# --- read-only code / evidence preflight ---------------------------------------
check_code() {
  local head; head="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)" || { echo "COMMIT_PIN_MISMATCH cannot resolve HEAD"; return 1; }
  [[ "$head" == "$PIN" ]] || { echo "COMMIT_PIN_MISMATCH HEAD $head != $PIN"; return 1; }
  local br; br="$(git -C "$REPO" branch --show-current 2>/dev/null)"
  [[ "$br" == "$BRANCH" ]] || { echo "BRANCH_MISMATCH '$br' != $BRANCH"; return 1; }
  local ex; ex="$(git -C "$REPO" rev-parse --git-path info/exclude 2>/dev/null)" || { echo "EXCLUDE_UNRESOLVED"; return 1; }
  [[ "$ex" = /* ]] || ex="$REPO/$ex"
  [[ "$(sha256sum "$ex" 2>/dev/null | cut -d' ' -f1)" == "$EXCLUDE_SHA" ]] || { echo "EXCLUDE_LIST_CHANGED $ex"; return 1; }
  [[ -z "$(git -C "$REPO" status --porcelain 2>&1)" ]] || { echo "WORKTREE_DIRTY $REPO"; return 1; }
  local item rel want got blob
  for item in "${REGISTERED_MODULES[@]}"; do
    rel="${item% *}"; want="${item#* }"
    got="$(sha256sum "$REPO/$rel" 2>/dev/null | cut -d' ' -f1)"
    blob="$(git -C "$REPO" show "HEAD:$rel" 2>/dev/null | sha256sum | cut -d' ' -f1)"
    [[ "$got" == "$want" && "$blob" == "$want" ]] || { echo "MODULE_HASH_MISMATCH $rel"; return 1; }
  done
  for rel in seqref_mri/scripts/v03dlt_train.py seqref_mri/scripts/v03dlt_launch.sh \
             seqref_mri/scripts/v03dlt_summary.py seqref_mri/scripts/v03dlt_fixtures.py \
             seqref_mri/scripts/v03dlt_stub_train.py; do
    git -C "$REPO" ls-files --error-unmatch -- "$rel" >/dev/null 2>&1 || { echo "MODULE_NOT_TRACKED $rel"; return 1; }
  done
  return 0
}
check_evidence() {
  local f want item
  for f in "$P4:$P4_SHA" "$PROBE:$PROBE_SHA" "$GATE:$GATE_SHA"; do
    want="${f##*:}"; f="${f%:*}"
    [[ "$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)" == "$want" ]] || { echo "EVIDENCE_HASH_MISMATCH $f"; return 1; }
  done
  for f in "$PROBE" "$GATE"; do
    ( cd "$(dirname "$f")" && sha256sum -c --quiet "$(basename "$f").sha256" >/dev/null 2>&1 ) || { echo "EVIDENCE_SIDECAR_MISMATCH $f"; return 1; }
  done
  for item in "${V03_EVIDENCE[@]}"; do
    f="$RES/${item% *}"; want="${item#* }"
    [[ "$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)" == "$want" ]] || { echo "V03_EVIDENCE_CHANGED $f"; return 1; }
  done
  return 0
}
check_space() {   # $1 = number of attempts the free space must cover (B3)
  local n="${1:-1}" need avail
  need=$(( NEED_BYTES * n ))
  avail="$(df --output=avail -B1 "$ROOT" 2>/dev/null | tail -1 | tr -d ' ')"
  [[ "$avail" =~ ^[0-9]+$ && "$avail" -ge "$need" ]] || { echo "DISK_SPACE_SHORT avail=${avail:-?} need=$need attempts=$n"; return 1; }
}

# --- lock ---------------------------------------------------------------------
acquire_lock() {
  exec 9>>"$ROOT/.launch.lock" || die_early LOCK_FAILED "$ROOT/.launch.lock"
  flock -n 9 || die_early LOCK_HELD "another launcher instance holds $ROOT/.launch.lock"
}

# --- handled exit: snapshot -> close log + sidecar -> manifest ---------------
exit_handled() {
  local code="$1" reason="$2" r
  trap '' INT TERM
  if [[ -n "$IPID" ]]; then kill "$IPID" 2>/dev/null; wait "$IPID" 2>/dev/null; IPID=""; fi
  if [[ "$SESSION_OPEN" -eq 1 ]]; then
    r="$(helper snapshot "$ROOT" "$S" 2>&1)" || { log "ERROR SNAPSHOT_FAILED: $r"; code=2; }
    log "handled exit: $reason (exit $code); $r"
    LOG_DONE="$LOG"; LOG=""
    ( cd "$ROOT" && sha256sum "$(basename "$LOG_DONE")" > "$(basename "$LOG_DONE").sha256" ) \
      || { say "ERROR LOG_SIDECAR_FAILED"; code=2; }
    r="$(helper manifest "$ROOT" "$S" 2>&1)" || { say "ERROR MANIFEST_FAILED: $r"; code=2; }
    say "$r"
  else
    say "handled exit before session start: $reason (exit $code)"
  fi
  exit "$code"
}

trainer_catches_sigint() {   # SigCgt bit for SIGINT (2) -> mask 0x2
  local m
  m="$(sed -n 's/^SigCgt:[[:space:]]*//p' "/proc/$TPID/status" 2>/dev/null)" || return 1
  [[ "$m" =~ ^[0-9a-fA-F]+$ ]] && (( (16#$m & 2) != 0 ))
}
trainer_alive() {           # alive and not a zombie
  local st
  kill -0 "$TPID" 2>/dev/null || return 1
  st="$(sed -E 's/^.*\) ([A-Za-z]).*$/\1/' "/proc/$TPID/stat" 2>/dev/null)"
  [[ "$st" != "Z" ]]
}
forward_if_ready() {
  [[ "$CANCEL" -eq 1 && "$FORWARDED" -eq 0 && -n "$TPID" ]] || return 0
  trainer_alive || return 0
  if trainer_catches_sigint; then
    kill -INT "$TPID" 2>/dev/null && FORWARDED=1 && log "forwarded one SIGINT to trainer pid $TPID"
  elif [[ "$PENDING_LOGGED" -eq 0 ]]; then
    PENDING_LOGGED=1
    log "cancellation pending: trainer pid $TPID has not yet installed its SIGINT handler"
  fi
}
on_signal() {
  log "signal $1 received: cancellation"
  CANCEL=1
  forward_if_ready
}

open_session() {
  S="$(date -u +%Y%m%dT%H%M%SZ)"
  LOG="$ROOT/launcher_session_$S.log"
  [[ ! -e "$LOG" ]] || die_early ARTIFACT_EXISTS "$LOG"
  printf '%s' "$PRELOG" > "$LOG" || die_early LOG_FAILED "$LOG"
  SESSION_OPEN=1
  log "session $S opened: mode=$MODE pin=$PIN repo=$REPO branch=$BRANCH data_root=$DATA_ROOT root=$ROOT launcher v$VERSION"
  trap 'on_signal INT' INT
  trap 'on_signal TERM' TERM
  setsid "${INHIBIT_CMD[@]}" >/dev/null 2>&1 9>&- &   # never inherits the launch lock
  IPID=$!
  sleep 1
  kill -0 "$IPID" 2>/dev/null || { IPID=""; log "ERROR INHIBIT_UNAVAILABLE: systemd-inhibit did not hold a sleep:idle lock"; exit_handled 2 INHIBIT_UNAVAILABLE; }
  log "sleep:idle inhibitor held (pid $IPID)"
}

# --- modes ----------------------------------------------------------------------
case "$MODE" in
  ack-stop)
    [[ -d "$ROOT" ]] || die_early CAMPAIGN_ROOT_MISSING "$ROOT"
    [[ -e "$ROOT/.launch.lock" ]] || die_early LOCK_FAILED "missing $ROOT/.launch.lock"
    acquire_lock
    r="$(helper ackstop "$ROOT" 2>&1)" || die_early "$(helper_code "$r")" "$r"
    say "$r"; exit 0 ;;
  new)
    if [[ -e "$ROOT" ]]; then
      if [[ -e "$ROOT/.launch.lock" ]] && ! flock -n "$ROOT/.launch.lock" true; then
        die_early LOCK_HELD "another launcher instance holds $ROOT/.launch.lock"
      fi
      die_early CAMPAIGN_ROOT_EXISTS "$ROOT exists; use --mode resume"
    fi
    r="$(check_code)" || die_early "${r%% *}" "$r"
    r="$(check_evidence)" || die_early "${r%% *}" "$r"
    mkdir "$ROOT" || die_early CAMPAIGN_ROOT_CREATE_FAILED "$ROOT"
    acquire_lock
    log "campaign root created: $ROOT" ;;
  resume)
    [[ -d "$ROOT" ]] || die_early CAMPAIGN_ROOT_MISSING "$ROOT; use --mode new"
    [[ -e "$ROOT/.launch.lock" ]] || die_early LOCK_FAILED "missing $ROOT/.launch.lock"
    acquire_lock
    r="$(helper verify "$ROOT" 2>&1)" || die_early "$(helper_code "$r")" "$r"
    PRELOG+="$(date -u +%FT%TZ) $ABBR resume verification (read-only, under lock): $r"$'\n'
    say "resume verification: $r" ;;
esac
open_session

# --- campaign-level disk allowance (B3): space for every unsettled run ---------
rem="$(helper remaining "$ROOT" 2>&1)" || { log "ERROR REMAINING_FAILED: $rem"; exit_handled 2 REMAINING_FAILED; }
read -r _ nrem <<<"$rem"
[[ "$nrem" =~ ^[0-6]$ ]] || { log "ERROR REMAINING_FAILED: $rem"; exit_handled 2 REMAINING_FAILED; }
if [[ "$nrem" -gt 0 ]]; then
  r="$(check_space "$nrem")" || { log "ERROR $r (campaign allowance)"; exit_handled 2 "${r%% *}"; }
  log "campaign disk allowance PASS for $nrem unsettled run(s)"
fi

# --- campaign loop -----------------------------------------------------------------
while :; do
  if [[ -e "$ROOT/STOP" ]]; then
    r="$(helper stopseen "$ROOT" "$S" 2>&1)" || { log "ERROR STOP_RECORD_FAILED: $r"; exit_handled 2 STOP_RECORD_FAILED; }
    log "STOP request recorded ($r); stopping before the next run"
    exit_handled 0 STOP_REQUESTED
  fi
  [[ "$CANCEL" -eq 0 ]] || exit_handled 2 CANCELLED
  nxt="$(helper next "$ROOT" 2>&1)" || { log "ERROR NEXT_FAILED: $nxt"; exit_handled 2 NEXT_FAILED; }
  read -r kind a1 a2 a3 <<<"$nxt"
  case "$kind" in
    DONE) log "all six pairs settled"; exit_handled 0 CAMPAIGN_COMPLETE ;;
    BLOCKED) log "ERROR $a1: a pair cannot settle; later pairs not attempted"; exit_handled 2 "$a1" ;;
    RUN) ;;
    *) log "ERROR NEXT_FAILED: $nxt"; exit_handled 2 NEXT_FAILED ;;
  esac
  ARM="$a1"; SEED="$a2"; ATT="$a3"
  r="$(check_code)" || { log "ERROR $r"; exit_handled 2 "${r%% *}"; }
  r="$(check_evidence)" || { log "ERROR $r"; exit_handled 2 "${r%% *}"; }
  r="$(check_space 1)" || { log "ERROR $r (per-attempt check)"; exit_handled 2 "${r%% *}"; }
  r="$(helper preflight "$ROOT" "$ARM" "$SEED" "$ATT" 2>&1)" || { log "ERROR $r"; exit_handled 2 PREFLIGHT_LEDGER_FAILED; }
  log "preflight PASS for arm $ARM seed $SEED attempt $ATT; starting trainer"
  OUT_LOG="$ROOT/runlog_arm${ARM}_seed${SEED}_attempt${ATT}_$S.stdout.log"
  ERR_LOG="$ROOT/runlog_arm${ARM}_seed${SEED}_attempt${ATT}_$S.stderr.log"
  [[ ! -e "$OUT_LOG" && ! -e "$ERR_LOG" ]] || { log "ERROR ARTIFACT_EXISTS run log"; exit_handled 2 ARTIFACT_EXISTS; }
  # launch boundary: a STOP or cancellation during preflight starts no run
  if [[ -e "$ROOT/STOP" ]]; then log "STOP present at the launch boundary; no run started"; continue; fi
  [[ "$CANCEL" -eq 0 ]] || { log "cancellation at the launch boundary; no run started"; exit_handled 2 CANCELLED; }
  ( cd "$REPO" && PYTHONPATH="$REPO" exec setsid "${TRAINER_CMD[@]}" --mode run --data-root "$DATA_ROOT" \
      --p4-stats2 "$P4" --probe-record "$PROBE" --commit-pin "$PIN" \
      --out "$ROOT" --arm "$ARM" --seed "$SEED" ) >"$OUT_LOG" 2>"$ERR_LOG" &
  TPID=$!
  FORWARDED=0 PENDING_LOGGED=0
  while trainer_alive; do
    forward_if_ready
    sleep 0.2
  done
  wait "$TPID"; rc=$?
  TPID=""
  ( cd "$ROOT" && sha256sum "$(basename "$OUT_LOG")" > "$(basename "$OUT_LOG").sha256" \
      && sha256sum "$(basename "$ERR_LOG")" > "$(basename "$ERR_LOG").sha256" ) \
    || { log "ERROR RUN_LOG_SIDECAR_FAILED"; exit_handled 2 RUN_LOG_SIDECAR_FAILED; }
  latest="$(helper latest "$ROOT" "$ARM" "$SEED" "$ATT" 2>&1)" || { log "ERROR LEDGER_READ_FAILED: $latest"; exit_handled 2 LEDGER_READ_FAILED; }
  log "trainer exited rc=$rc; ledger: $latest"
  read -r st code <<<"$latest"
  if [[ "$CANCEL" -eq 1 || "$code" == "KEYBOARD_INTERRUPT" ]]; then
    if [[ "$st" == "interrupted" && "$code" == "KEYBOARD_INTERRUPT" && "$CANCEL" -eq 1 ]]; then
      exit_handled 2 CANCELLED
    fi
    log "ERROR CANCEL_LEDGER_MISMATCH: cancel=$CANCEL ledger='$latest'"
    exit_handled 2 CANCEL_LEDGER_MISMATCH
  fi
  # exit code must agree with the ledger: rc 0 <=> completed (review fix R2)
  if [[ "$st" != "NONE" ]]; then
    if { [[ "$rc" -ne 0 && "$st" == "completed" ]] || [[ "$rc" -eq 0 && "$st" != "completed" ]]; }; then
      log "ERROR TRAINER_EXIT_MISMATCH: rc=$rc but ledger='$latest'; stop for review"
      exit_handled 2 TRAINER_EXIT_MISMATCH
    fi
  fi
  case "$st" in
    NONE) log "ERROR TRAINER_NO_LEDGER_ENTRY: stop for review"; exit_handled 2 TRAINER_NO_LEDGER_ENTRY ;;
    completed|numerical_failure|failed_validity) log "pair arm $ARM seed $SEED settled: $st" ;;
    interrupted) log "infrastructure interruption ($code): ledger-policy retry next" ;;
    *) log "ERROR LEDGER_STATUS_UNKNOWN: $latest"; exit_handled 2 LEDGER_STATUS_UNKNOWN ;;
  esac
done
