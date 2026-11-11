#!/usr/bin/env bash
# SEQREF-MRI-DIAGJSHGCMD v0.1 -- reviewed host command for DIAG-JSH stage G (single real-data main run)
# LIFETIME: KEEP
# =============================================================================
# Plan SEQREF-MRI-DIAGJSHF2 v0.1 (F2c). Same structure as the reviewed F1
# command v0.4 (diag_jsh_f1_cmd.sh), with these changes only:
#   * runs diag_jsh_g_run.py; checks six code files (core, adapter, F1
#     prepare, I3, G runner, F2 resource record) and the F1 manifest and its
#     sidecar against the registered sha256 before Python starts;
#   * G budget (G_BUDGET = g_budget_bytes of the F2 resource record) in the
#     free-space check: free >= RESERVE + G_BUDGET + CMDLOG_MAX;
#   * running-job check also matches diag_jsh_ (as the reviewed pre-checks);
#   * run folder out/G_<UTC stamp>/, command log out/G_<UTC stamp>.cmd.log.
# Unchanged from F1 v0.4: V05LT interpreter -B -E -s with PYTHON* removed,
# HEAD = pin, directories reached without symlinks, empty run folder, hard
# wall cap (Python deadline from the resource record, at most MAX_WALL_S -
# 120 s; TERM at MAX_WALL_S - 90 s; KILL 60 s later), command log capped at
# CMDLOG_MAX with all pipeline statuses checked, MTIME_CHECK and GIT_STATUS
# afterwards (indicators, not proof).
# Exit: 0 PASS; 1 typed STOP; 2 usage / environment / write-check failure;
#   77 write-guard halt; 78 log allowance; 124/137 wall cap reached.
# Usage: bash diag_jsh_g_cmd.sh CHAIN CHAIN_SHA RESERVE_BYTES MAX_WALL_S \
#          MANIFEST MANIFEST_SHA CORE_SHA ADAPTER_SHA F1_SHA I3_SHA G_SHA F2RES_SHA
# Sourcing the file defines the constants and run_capped only; it runs
#   nothing.
# Changelog (v0.1, 2026-11-11): first G command (from F1 command v0.4).
# =============================================================================
V05=/home/benjamin/CSMFII
PIN=7455da8027f33a6cca4dbf72f04e72420e938f45
PY=$V05/seqref_mri/.venv_mri/bin/python
DJ=$V05/v06lt_evidence/DIAG_JSH
G_BUDGET=$((32 * 1024 * 1024))      # = g_budget_bytes in diag_jsh_f2_resources.py
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
CHAIN=${1:-}; CSHA=${2:-}; RES=${3:-}; WALL=${4:-}; MANI=${5:-}; MSHA=${6:-}
S1=${7:-}; S2=${8:-}; S3=${9:-}; S4=${10:-}; S5=${11:-}; S6=${12:-}
for h in "$CSHA" "$MSHA" "$S1" "$S2" "$S3" "$S4" "$S5" "$S6"; do
  [[ "$h" =~ ^[0-9a-f]{64}$ ]] || die "usage: $0 CHAIN CHAIN_SHA RESERVE_BYTES MAX_WALL_S MANIFEST MANIFEST_SHA CORE_SHA ADAPTER_SHA F1_SHA I3_SHA G_SHA F2RES_SHA"
done
[[ "$RES" =~ ^[0-9]+$ && "$WALL" =~ ^[0-9]+$ && $WALL -ge 300 ]] || die "RESERVE_BYTES integer; MAX_WALL_S integer >= 300"
[[ -n "$CHAIN" && -x "$PY" ]] || die "chain or interpreter missing"
[[ -f "$MANI" && ! -L "$MANI" && -f "$MANI.sha256" && ! -L "$MANI.sha256" ]] || die "manifest or sidecar missing or a symlink"
[[ "$(sha256sum "$MANI" | cut -d' ' -f1)" == "$MSHA" ]] || die "manifest hash != registered"
[[ "$(cut -d' ' -f1 "$MANI.sha256")" == "$MSHA" ]] || die "manifest sidecar != registered"
for d in "$DJ" "$DJ/code" "$DJ/out"; do
  [[ -d "$d" && "$(realpath -e -- "$d")" == "$d" ]] || die "$d missing or reached through a symlink"
done
cd "$V05" || die "cannot cd $V05"
[[ "$(git rev-parse HEAD)" == "$PIN" ]] || die "V05LT HEAD is not the pin"
pgrep -f 'seqref_mri[.]scripts[.](v05lt|v06lt)_|diag_jsh_(f1_prepare|g_run)[.]py' >/dev/null; PRC=$?
[[ $PRC -eq 1 ]] || die "a V05LT/V06LT/DIAG-JSH process is running (pgrep rc $PRC)"
for p in "diag_jsh_core.py $S1" "diag_jsh_adapter.py $S2" "diag_jsh_f1_prepare.py $S3" \
         "diag_jsh_i3.py $S4" "diag_jsh_g_run.py $S5" "diag_jsh_f2_resources.py $S6"; do
  set -- $p
  [[ ! -L "$DJ/code/$1" && "$(sha256sum "$DJ/code/$1" | cut -d' ' -f1)" == "$2" ]] || die "$1 hash != reviewed"
done
FREE=$(df -B1 --output=avail "$DJ/out" | tail -1 | tr -d ' ') || die "df failed"
[[ "$FREE" =~ ^[0-9]+$ && $FREE -ge $((RES + G_BUDGET + CMDLOG_MAX)) ]] || die "free $FREE < reserve $RES + G budget + command-log allowance"
STAMP=G_$(date -u +%Y%m%dT%H%M%SZ)
OUT=$DJ/out/$STAMP
LOGF=$DJ/out/$STAMP.cmd.log
[[ ! -e "$OUT" && ! -e "$LOGF" ]] || die "exists: $OUT"
mkdir -- "$OUT" || die "cannot create $OUT"
MARK=$(mktemp "$DJ/out/.g_mark.XXXXXX") || die "mktemp failed"
G0=$(git status --porcelain -- . ':(exclude)v06lt_evidence') || die "git status failed"
sleep 1
run_capped "$WALL" \
  env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP "$PY" -B -E -s \
  "$DJ/code/diag_jsh_g_run.py" --chain "$CHAIN" --chain-sha256 "$CSHA" \
  --out "$OUT" --reserve-bytes "$((RES + CMDLOG_MAX))" --max-wall-s "$WALL" \
  --manifest "$MANI" --manifest-sha256 "$MSHA" \
  --sha-diag_jsh_core "$S1" --sha-diag_jsh_adapter "$S2" \
  --sha-diag_jsh_f1_prepare "$S3" --sha-diag_jsh_i3 "$S4" \
  --sha-diag_jsh_g_run "$S5" --sha-diag_jsh_f2_resources "$S6" \
  2>&1 | head -c "$CMDLOG_MAX" | tee "$LOGF"
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
