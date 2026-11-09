#!/usr/bin/env bash
# SEQREF-MRI-DIAGJSHF1CMD v0.4 -- reviewed host command for DIAG-JSH step F1 (preparation; reads real data)
# LIFETIME: KEEP
# =============================================================================
# Runs diag_jsh_f1_prepare.py with the V05LT interpreter (-B -E -s, PYTHON*
# removed) from the V05LT tree, after G-D1 (completion PASS) and G-D4.
# Before Python: HEAD = pin; no V05LT/V06LT process; DIAG_JSH, code/ and out/
# reached without symlinks; code files = reviewed sha256; live free space
# >= RESERVE + F1 budget + command-log allowance; then the empty run folder
# OUT is created. Python installs its write guard and enforced write budget
# on OUT before any write, with reserve = RESERVE + CMDLOG_MAX, so the
# command log's allowance stays reserved while Python runs.
# Hard wall cap MAX_WALL_S (>= 300): Python stops at MAX_WALL_S - 120 s;
# run_capped sends TERM at MAX_WALL_S - TERM_LEAD (90 s) and KILL
# KILL_GRACE (60 s) later, i.e. at MAX_WALL_S - 30 s.
# Command log: everything written to it (Python output and the final
# verification summary) is capped at CMDLOG_MAX in total; reaching the cap,
# or any non-zero status of timeout/Python, head or tee, is never a PASS.
# Afterwards (indicators, not proof): MTIME_CHECK and GIT_STATUS.
# Exit: 0 PASS; 1 typed STOP; 2 usage / environment / write-check failure;
#   77 write-guard halt; 78 log allowance; 124/137 wall cap reached.
# Usage: bash diag_jsh_f1_cmd.sh CHAIN CHAIN_SHA RESERVE_BYTES MAX_WALL_S \
#          CORE_SHA ADAPTER_SHA F1_SHA
# Sourcing the file defines the constants and run_capped only (used by the
#   termination test); it runs nothing.
# Changelog (v0.4, 2026-11-09; F1 review 3): statuses of all three pipeline
#   stages checked; final summary written inside the remaining log
#   allowance; command-log allowance reserved for Python; run_capped
#   function (tested with a TERM-ignoring child).
# Changelog (v0.3, 2026-11-09): true hard cap; command log capped.
# Changelog (v0.2, 2026-11-09): OUT created and validated; disk check.
# Changelog (v0.1, 2026-11-09): first F1 command.
# =============================================================================
V05=/home/benjamin/CSMFII
PIN=7455da8027f33a6cca4dbf72f04e72420e938f45
PY=$V05/seqref_mri/.venv_mri/bin/python
DJ=$V05/v06lt_evidence/DIAG_JSH
F1_BUDGET=$((16 * 1024 * 1024))     # = F1_BUDGET in diag_jsh_f1_prepare.py
CMDLOG_MAX=$((1024 * 1024))         # command-log allowance (all writes to it)
TERM_LEAD=90                        # TERM at WALL - TERM_LEAD
KILL_GRACE=60                       # KILL at WALL - TERM_LEAD + KILL_GRACE

run_capped() {                      # run_capped WALL cmd...  (hard cap)
  local wall=$1; shift
  timeout --signal=TERM --kill-after="$KILL_GRACE" "$((wall - TERM_LEAD))" "$@"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi   # sourced: defs only
set -u
die() { echo "STOP: $*" >&2; exit 2; }
CHAIN=${1:-}; CSHA=${2:-}; RES=${3:-}; WALL=${4:-}; S1=${5:-}; S2=${6:-}; S3=${7:-}
for h in "$CSHA" "$S1" "$S2" "$S3"; do
  [[ "$h" =~ ^[0-9a-f]{64}$ ]] || die "usage: $0 CHAIN CHAIN_SHA RESERVE_BYTES MAX_WALL_S CORE_SHA ADAPTER_SHA F1_SHA"
done
[[ "$RES" =~ ^[0-9]+$ && "$WALL" =~ ^[0-9]+$ && $WALL -ge 300 ]] || die "RESERVE_BYTES integer; MAX_WALL_S integer >= 300"
[[ -n "$CHAIN" && -x "$PY" ]] || die "chain or interpreter missing"
for d in "$DJ" "$DJ/code" "$DJ/out"; do
  [[ -d "$d" && "$(realpath -e -- "$d")" == "$d" ]] || die "$d missing or reached through a symlink"
done
cd "$V05" || die "cannot cd $V05"
[[ "$(git rev-parse HEAD)" == "$PIN" ]] || die "V05LT HEAD is not the pin"
if pgrep -f "seqref_mri.scripts.(v05lt|v06lt)_" >/dev/null; then
  die "a V05LT/V06LT process is running"
fi
for p in "diag_jsh_core.py $S1" "diag_jsh_adapter.py $S2" "diag_jsh_f1_prepare.py $S3"; do
  set -- $p
  [[ ! -L "$DJ/code/$1" && "$(sha256sum "$DJ/code/$1" | cut -d' ' -f1)" == "$2" ]] || die "$1 hash != reviewed"
done
FREE=$(df -B1 --output=avail "$DJ/out" | tail -1 | tr -d ' ') || die "df failed"
[[ "$FREE" =~ ^[0-9]+$ && $FREE -ge $((RES + F1_BUDGET + CMDLOG_MAX)) ]] || die "free $FREE < reserve $RES + F1 budget + command-log allowance"
STAMP=F1_$(date -u +%Y%m%dT%H%M%SZ)
OUT=$DJ/out/$STAMP
LOGF=$DJ/out/$STAMP.cmd.log
[[ ! -e "$OUT" && ! -e "$LOGF" ]] || die "exists: $OUT"
mkdir -- "$OUT" || die "cannot create $OUT"
MARK=$(mktemp "$DJ/out/.f1_mark.XXXXXX") || die "mktemp failed"
G0=$(git status --porcelain -- . ':(exclude)v06lt_evidence') || die "git status failed"
sleep 1
run_capped "$WALL" \
  env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP "$PY" -B -E -s \
  "$DJ/code/diag_jsh_f1_prepare.py" --chain "$CHAIN" --chain-sha256 "$CSHA" \
  --out "$OUT" --reserve-bytes "$((RES + CMDLOG_MAX))" --max-wall-s "$WALL" \
  --sha-diag_jsh_core "$S1" --sha-diag_jsh_adapter "$S2" \
  --sha-diag_jsh_f1_prepare "$S3" 2>&1 | head -c "$CMDLOG_MAX" | tee "$LOGF"
ps=("${PIPESTATUS[@]}"); RC=${ps[0]}; HRC=${ps[1]}; TRC=${ps[2]}
FAIL=()
[[ $HRC -eq 0 ]] || FAIL+=("head rc $HRC")
[[ $TRC -eq 0 ]] || FAIL+=("tee rc $TRC")
LSZ=$(stat -c %s "$LOGF" 2>/dev/null || echo -1)
[[ $LSZ -ge 0 && $LSZ -lt $CMDLOG_MAX ]] || FAIL+=("command log allowance reached ($LSZ bytes)")
NEW=$(find "$V05/seqref_mri" -newer "$MARK" -print 2>&1); FRC=$?
[[ $FRC -eq 0 ]] || FAIL+=("find rc $FRC")
G1=$(git status --porcelain -- . ':(exclude)v06lt_evidence' 2>&1); GRC=$?
[[ $GRC -eq 0 ]] || FAIL+=("git status rc $GRC")
rm -f "$MARK" || FAIL+=("cannot remove marker")
SUMMARY=$(
  echo "exit=$RC head=$HRC tee=$TRC"
  if [[ $FRC -ne 0 ]]; then echo "MTIME_CHECK ERROR (rc $FRC)"
  elif [[ -n "$NEW" ]]; then echo "MTIME_CHECK FAIL:"; printf '%s\n' "$NEW" | head -20
  else echo "MTIME_CHECK OK (indicator, not proof)"; fi
  if [[ $GRC -ne 0 ]]; then echo "GIT_STATUS ERROR (rc $GRC)"
  elif [[ "$G0" == "$G1" ]]; then echo "GIT_STATUS unchanged"
  else echo "GIT_STATUS CHANGED"; fi
)
REM=$(( CMDLOG_MAX - (LSZ < 0 ? CMDLOG_MAX : LSZ) ))
SB=$(printf '%s\n' "$SUMMARY" | wc -c)       # bytes, not characters
if (( SB > REM )); then
  FAIL+=("summary ($SB bytes) exceeds remaining log allowance ($REM)")
  printf '%s\n' "$SUMMARY"
else
  printf '%s\n' "$SUMMARY" | head -c "$REM" | tee -a "$LOGF"
  ps=("${PIPESTATUS[@]}")
  [[ ${ps[0]} -eq 0 && ${ps[1]} -eq 0 && ${ps[2]} -eq 0 ]] || FAIL+=("summary append rc ${ps[*]}")
fi
FSZ=$(stat -c %s "$LOGF" 2>/dev/null || echo -1)
[[ $FSZ -ge 0 && $FSZ -le $CMDLOG_MAX ]] || FAIL+=("final log size $FSZ > allowance")
HASH=$(sha256sum "$LOGF") || FAIL+=("log hash failed")
[[ -n "${HASH:-}" ]] && echo "$HASH"
(( ${#FAIL[@]} )) && { echo "STOP: verification/logging error: ${FAIL[*]}" >&2; exit 2; }
[[ $FRC -eq 0 && -z "$NEW" && "$G0" == "$G1" ]] || exit 2
exit "$RC"
