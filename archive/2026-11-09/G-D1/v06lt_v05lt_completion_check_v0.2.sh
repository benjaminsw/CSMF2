#!/usr/bin/env bash
# SEQREF-MRI-V06LTV05DONE v0.2 -- read-only V05LT registered completion check (V06LT P1/P6 prerequisite)
# LIFETIME: DIAGNOSTIC
# =============================================================================
# What it runs: the PINNED V05LT code, from the V05LT tree, with the V05LT
#   interpreter (-B -E -s, PYTHON* variables removed):
#     Campaign._verify_chain()   launch chain vs its approved sha256, then the
#                                launch-inputs record (bind_chain: inputs,
#                                pin, backend) -- read-only;
#     Campaign.verify_completion()  the registered completion check (latest
#                                launch ended in campaign_complete, no
#                                unresolved/final halt, six advanced runs,
#                                each run re-verified: completed, registered
#                                config, final weights, post-run records;
#                                epoch-0 stage per seed) -- read-only.
#   This is the order the registered holdout path uses, WITHOUT its campaign
#   lock (the lock would create .launch.lock in the V05LT results tree).
# Prints only: PASS/REFUSED, the refusal code, the evidence sha256 and the
#   campaign_complete sequence number. Per-run epochs and post-run states are
#   NOT printed (no added outcome exposure; SPEC 211 §10.1).
# Writes: only this script's log in the authorised evidence folder
#   /home/benjamin/CSMFII/v06lt_evidence/V05LT_completion/ (outside
#   seqref_mri/). Afterwards two checks (indicators, not proof):
#   MTIME_CHECK  no surviving file or directory under seqref_mri/ has a
#                modification time newer than the start marker (kept outside
#                the tree). It detects newer SURVIVING files; directory mtimes
#                also catch most creations/deletions inside existing
#                directories, but it does not prove that nothing was written
#                or deleted (e.g. a write that restores its mtime).
#   GIT_STATUS   `git status` (excluding v06lt_evidence/) unchanged.
#   Run it only when no V05LT process is running.
# Changelog v0.2 (2026-11-05): failures of the final find, git status, log
#   append and log hash are checked (any -> exit 2, never a PASS exit);
#   check renamed MTIME_CHECK and described as an indicator, not proof;
#   no `set -e` (explicit checks only).
# Exit: 0 PASS; 1 REFUSED (typed code printed: not complete / evidence);
#       2 usage, environment or write-check failure.
# Usage: bash v06lt_v05lt_completion_check_v0.1.sh CHAIN_PATH APPROVED_SHA256
#   CHAIN_PATH      V05LT launch chain, as recorded in V05LTEXEC
#   APPROVED_SHA256 its approved sha256, as recorded in V05LTEXEC
# =============================================================================
set -u
V05=/home/benjamin/CSMFII
PIN=7455da8027f33a6cca4dbf72f04e72420e938f45
PY=$V05/seqref_mri/.venv_mri/bin/python
OUT=$V05/v06lt_evidence/V05LT_completion
die() { echo "STOP: $*" >&2; exit 2; }
CHAIN=${1:-}; ASHA=${2:-}
[[ -n "$CHAIN" && "$ASHA" =~ ^[0-9a-f]{64}$ ]] \
  || die "usage: $0 CHAIN_PATH APPROVED_SHA256 (both from V05LTEXEC)"
[[ -x "$PY" ]] || die "V05LT interpreter missing: $PY"
cd "$V05" || die "cannot cd $V05"
[[ "$(git rev-parse HEAD)" == "$PIN" ]] || die "V05LT HEAD is not the pin"
if pgrep -f "seqref_mri.scripts.v05lt_(launch|train|prodstages|eval)" >/dev/null; then
  die "a V05LT process is running; run this check only after it has stopped"
fi
mkdir -p "$OUT" || die "cannot create $OUT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOGF="$OUT/v05lt_completion_$STAMP.log"
[[ ! -e "$LOGF" ]] || die "log exists: $LOGF"
MARK=$(mktemp "${TMPDIR:-/tmp}/v05done_mark.XXXXXX") || die "mktemp failed"
G0=$(git status --porcelain -- . ':(exclude)v06lt_evidence') || die "git status failed"
sleep 1                            # marker strictly older than any new write

set +e
env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP "$PY" -B -E -s - \
  "$CHAIN" "$ASHA" <<'PY' 2>&1 | tee "$LOGF"
import json, logging, sys, traceback
from pathlib import Path
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
from seqref_mri.scripts import v05lt_common as C
from seqref_mri.scripts import v05lt_launch as L
from seqref_mri.scripts import v05lt_prodstages as PS
chain, asha = Path(sys.argv[1]), sys.argv[2]
try:
    paths = C.Paths()
    stages = PS.ProductionStages(paths, paths.data_root)
    camp = L.Campaign(paths.v05lt, stages, chain=(chain, asha, paths.repo),
                      fixture=False)
    camp._verify_chain()                       # read-only (no lock)
    ev, ev_sha = camp.verify_completion()      # registered check, read-only
    print(f"RESULT V05LT_COMPLETION PASS evidence_sha256={ev_sha} "
          f"complete_seq={ev['complete_seq']} runs={len(ev['runs'])} "
          f"epoch0_seeds={len(ev['epoch0'])}")
    sys.exit(0)
except C.V05LTError as exc:
    print(f"RESULT V05LT_COMPLETION REFUSED {exc.code}")
    sys.exit(1)
except Exception as exc:                       # logged, never swallowed
    logging.error("UNEXPECTED %s:\n%s", type(exc).__name__,
                  traceback.format_exc())
    print(f"RESULT V05LT_COMPLETION FAILED UNEXPECTED {type(exc).__name__}")
    sys.exit(2)
PY
ps=("${PIPESTATUS[@]}")
RC=${ps[0]}; TRC=${ps[1]}
FAIL=()
NEW=$(find "$V05/seqref_mri" -newer "$MARK" -print 2>&1); FRC=$?
[[ $FRC -eq 0 ]] || FAIL+=("find rc $FRC")
G1=$(git status --porcelain -- . ':(exclude)v06lt_evidence' 2>&1); GRC=$?
[[ $GRC -eq 0 ]] || FAIL+=("git status rc $GRC")
rm -f "$MARK" || FAIL+=("cannot remove marker $MARK")
{
  echo "exit=$RC tee=$TRC"
  if [[ $FRC -ne 0 ]]; then echo "MTIME_CHECK ERROR: find failed (rc $FRC)"
  elif [[ -n "$NEW" ]]; then
    echo "MTIME_CHECK FAIL: surviving paths under seqref_mri/ newer than the start:"
    printf '%s\n' "$NEW" | head -20
  else echo "MTIME_CHECK OK: no surviving path under seqref_mri/ newer than the start (indicator, not proof)"; fi
  if [[ $GRC -ne 0 ]]; then echo "GIT_STATUS ERROR: git status failed (rc $GRC)"
  elif [[ "$G0" == "$G1" ]]; then echo "GIT_STATUS unchanged"
  else echo "GIT_STATUS CHANGED"; fi
} | tee -a "$LOGF"
ps=("${PIPESTATUS[@]}")
[[ ${ps[1]} -eq 0 ]] || FAIL+=("log append (tee -a) rc ${ps[1]}")
HASH=$(sha256sum "$LOGF") || FAIL+=("log hash failed")
[[ -n "${HASH:-}" ]] && echo "$HASH"
[[ $TRC -eq 0 ]] || FAIL+=("log write (tee) rc $TRC")
if (( ${#FAIL[@]} )); then
  echo "STOP: verification/logging error: ${FAIL[*]} (no PASS)" >&2; exit 2
fi
[[ $FRC -eq 0 && -z "$NEW" && "$G0" == "$G1" ]] || exit 2
exit "$RC"
