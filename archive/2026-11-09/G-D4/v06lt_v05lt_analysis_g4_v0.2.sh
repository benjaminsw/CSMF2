#!/usr/bin/env bash
# SEQREF-MRI-V06LTG4 v0.2 -- G-D4: V05LT registered validation analysis (PLAN 195 §9.2) via the pinned run_analysis
# LIFETIME: EPHEMERAL
# =============================================================================
# Runs v06lt_v05lt_analysis_g4_driver.py (hash checked here) with the V05LT
# interpreter (-B -E -s, PYTHON* removed) from the V05LT tree. The driver
# orchestrates only pinned functions (see its header); validation only.
# Output (write-once; approved location, narrowly scoped):
#   seqref_mri/results/v05lt/analysis/v05lt_analysis.json (+ .sha256)
#   and the pinned transient hard-link source .tmp.v05lt_analysis.json.<pid>
# Log: /home/benjamin/CSMFII/v06lt_evidence/V05LT_analysis/.
# Free-space floor 2 GiB: a PROPOSED allowance, not a measured budget.
#   Expected writes: the analysis document (MB scale: 296 comparisons, the
#   descriptive block with per-volume and per-epoch values), its sidecar
#   (< 100 B), the pinned .tmp file (the same bytes, hard-linked, removed
#   after linking: no additional space at the end) and this log (a few KB).
#   The actual sizes are recorded by the run for later reuse.
# Before the run: HEAD = pin; no V05LT / V06LT / DIAG-JSH process; analysis
#   folder and evidence folder reached without symlinks; outputs, sidecar,
#   any pinned .tmp leftover and the log absent (-e or -L); driver hash.
# After the run (indicators, not proof): MTIME_CHECK -- only the analysis
#   folder and the two outputs may be newer than the start; GIT_STATUS
#   (outputs and v06lt_evidence/ excluded) unchanged. Every command used to
#   build and compare these lists is checked; any failure -> exit 2.
# Hard wall cap MAX_WALL_S: TERM at MAX_WALL_S - 90 s, KILL 60 s later.
# Exit: 0 PASS; 1 REFUSED (typed code); 2 usage / environment / write-check
#   failure; 124/137 wall cap reached.
# Usage: bash v06lt_v05lt_analysis_g4_v0.2.sh CHAIN CHAIN_SHA CONFIG_SHA \
#          ELIG_SHA ZF_SHA DIFF_SHA MAX_WALL_S
# Changelog (v0.2, 2026-11-09; review of v0.1): checked list building and
#   comparison (find, sort, comm); symlink-free resolved folders; -e/-L on
#   outputs, leftovers and log; driver in a hashed file; free-space floor
#   described as a proposed allowance.
# Changelog (v0.1, 2026-11-09): first G-D4 command.
# =============================================================================
V05=/home/benjamin/CSMFII
PIN=7455da8027f33a6cca4dbf72f04e72420e938f45
PY=$V05/seqref_mri/.venv_mri/bin/python
EVD=$V05/v06lt_evidence/V05LT_analysis
ADIR=$V05/seqref_mri/results/v05lt/analysis
REL_OUT=seqref_mri/results/v05lt/analysis/v05lt_analysis.json
DRIVER=$EVD/v06lt_v05lt_analysis_g4_driver.py
DRIVER_SHA=8885726427e740c8dba5e6cbfa7bd656ee3aa7117826441bb2c36bd8619a51c5
MIN_FREE=$((2 * 1024 * 1024 * 1024))    # PROPOSED allowance (see header)
TERM_LEAD=90
KILL_GRACE=60

run_capped() {
  local wall=$1; shift
  timeout --signal=TERM --kill-after="$KILL_GRACE" "$((wall - TERM_LEAD))" "$@"
}
absent() { [[ ! -e "$1" && ! -L "$1" ]]; }
nolink() { [[ -d "$1" && "$(realpath -e -- "$1" 2>/dev/null)" == "$1" ]]; }

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi   # sourced: defs only
set -u
die() { echo "STOP: $*" >&2; exit 2; }
CHAIN=${1:-}; CSHA=${2:-}; CFG=${3:-}; ELIG=${4:-}; ZF=${5:-}; DIFF=${6:-}; WALL=${7:-}
for h in "$CSHA" "$CFG" "$ELIG" "$ZF" "$DIFF"; do
  [[ "$h" =~ ^[0-9a-f]{64}$ ]] || die "usage: $0 CHAIN CHAIN_SHA CONFIG_SHA ELIG_SHA ZF_SHA DIFF_SHA MAX_WALL_S"
done
[[ "$WALL" =~ ^[0-9]+$ && $WALL -ge 300 ]] || die "MAX_WALL_S integer >= 300"
[[ -n "$CHAIN" && -x "$PY" ]] || die "chain or interpreter missing"
cd "$V05" || die "cannot cd $V05"
[[ "$(git rev-parse HEAD)" == "$PIN" ]] || die "V05LT HEAD is not the pin"
if pgrep -f "seqref_mri.scripts.(v05lt|v06lt)_|diag_jsh_" >/dev/null; then
  die "a V05LT / V06LT / DIAG-JSH process is running"
fi
nolink "$ADIR" || die "$ADIR missing or reached through a symlink"
nolink "$EVD" || die "$EVD missing or reached through a symlink"
absent "$V05/$REL_OUT" && absent "$V05/$REL_OUT.sha256" || die "output exists: $REL_OUT (write-once)"
LEFT=$(find "$ADIR" -maxdepth 1 -name '.tmp.v05lt_analysis.json.*' -print 2>&1) || die "find failed"
[[ -z "$LEFT" ]] || die "pinned temporary leftover present: $LEFT"
[[ -f "$DRIVER" && ! -L "$DRIVER" ]] || die "driver missing: $DRIVER"
[[ "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == "$DRIVER_SHA" ]] || die "driver hash != reviewed"
FREE=$(df -B1 --output=avail "$ADIR" | tail -1 | tr -d ' ') || die "df failed"
[[ "$FREE" =~ ^[0-9]+$ && $FREE -ge $MIN_FREE ]] || die "free $FREE < $MIN_FREE"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOGF="$EVD/v05lt_analysis_g4_$STAMP.log"
absent "$LOGF" || die "log exists: $LOGF"
MARK=$(mktemp "$EVD/.g4_mark.XXXXXX") || die "mktemp failed"
GX=(-- . ':(exclude)v06lt_evidence' ":(exclude)$REL_OUT" ":(exclude)$REL_OUT.sha256")
G0=$(git status --porcelain "${GX[@]}") || die "git status failed"
sleep 1

run_capped "$WALL" env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP "$PY" -B -E -s \
  "$DRIVER" "$CHAIN" "$CSHA" "$CFG" "$ELIG" "$ZF" "$DIFF" "$REL_OUT" 2>&1 | tee "$LOGF"
ps=("${PIPESTATUS[@]}"); RC=${ps[0]}; TRC=${ps[1]}
FAIL=()
[[ $TRC -eq 0 ]] || FAIL+=("tee rc $TRC")
LST=$(mktemp "$EVD/.g4_lists.XXXXXX") || die "mktemp failed"
find "$V05/seqref_mri" -newer "$MARK" -print > "$LST.new" 2> "$LST.err" || FAIL+=("find rc $?")
[[ -s "$LST.err" ]] && FAIL+=("find stderr: $(head -c 200 "$LST.err")")
sort -u "$LST.new" > "$LST.new.s" || FAIL+=("sort (new) rc $?")
printf '%s\n' "$ADIR" "$V05/$REL_OUT" "$V05/$REL_OUT.sha256" | sort -u > "$LST.ok.s" || FAIL+=("sort (allowed) rc $?")
comm -23 "$LST.new.s" "$LST.ok.s" > "$LST.extra" || FAIL+=("comm rc $?")
EXTRA=$(cat "$LST.extra") || FAIL+=("read extra rc $?")
G1=$(git status --porcelain "${GX[@]}" 2>&1); GRC=$?
[[ $GRC -eq 0 ]] || FAIL+=("git status rc $GRC")
rm -f "$MARK" "$LST" "$LST.new" "$LST.err" "$LST.new.s" "$LST.ok.s" "$LST.extra" || FAIL+=("cannot remove temporary lists")
OUTCHK="absent"
if [[ -e "$V05/$REL_OUT" ]]; then
  S1=$(sha256sum "$V05/$REL_OUT" | cut -d' ' -f1) || FAIL+=("output hash rc $?")
  S2=$(cut -d' ' -f1 "$V05/$REL_OUT.sha256" 2>/dev/null)
  [[ -n "${S1:-}" && "$S1" == "${S2:-}" ]] && OUTCHK="sidecar OK $S1" || OUTCHK="sidecar MISMATCH"
fi
{
  echo "exit=$RC tee=$TRC output=$OUTCHK"
  if (( ${#FAIL[@]} )); then echo "MTIME_CHECK ERROR: ${FAIL[*]}"
  elif [[ -n "$EXTRA" ]]; then echo "MTIME_CHECK FAIL: paths other than the two outputs:"; printf '%s\n' "$EXTRA" | head -20
  else echo "MTIME_CHECK OK: only the analysis outputs are newer than the start (indicator, not proof)"; fi
  if [[ $GRC -ne 0 ]]; then echo "GIT_STATUS ERROR (rc $GRC)"
  elif [[ "$G0" == "$G1" ]]; then echo "GIT_STATUS unchanged (outputs excluded)"
  else echo "GIT_STATUS CHANGED"; fi
} | tee -a "$LOGF"
ps=("${PIPESTATUS[@]}"); [[ ${ps[1]} -eq 0 ]] || FAIL+=("log append rc ${ps[1]}")
HASH=$(sha256sum "$LOGF") || FAIL+=("log hash failed")
[[ -n "${HASH:-}" ]] && echo "$HASH"
(( ${#FAIL[@]} )) && { echo "STOP: verification/logging error: ${FAIL[*]} (no PASS)" >&2; exit 2; }
[[ -z "$EXTRA" && "$G0" == "$G1" ]] || exit 2
if [[ $RC -eq 0 ]]; then [[ "$OUTCHK" == sidecar\ OK* ]] || exit 2; fi
exit "$RC"
