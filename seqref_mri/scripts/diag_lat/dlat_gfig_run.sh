#!/usr/bin/env bash
# SEQREF-DLAT-GFIG v0.2 -- dlat_gfig_run.sh: render the figures, Table 1 and S1-S7 of the D-LAT G run ON THE HOST, from the run's own files
# LIFETIME: KEEP
#
# DRAFT FOR REVIEW. 'run' WRITES a new evidence folder: do not run it before the review of this
# script, its Git registration and the owner's authorisation, recorded in LudWiki.
# 'preflight' is read-only and writes nothing.
#
#   preflight -> read-only: user is not root; the G run folder exists and its g_report.json has the
#                registered sha256 and the verdict G: COMPLETE; the worktree is on branch ch2-diag
#                and its HEAD commit holds the renderer dlat_g_render.py at its reviewed sha256 and
#                this launcher exactly as it is being run (so both are registered in Git); the
#                interpreter exists and has matplotlib and numpy; free disk space.
#   run       -> 'preflight', then ONE rendering: the renderer is taken from that commit (not
#                from the worktree), reads the G run's out/ folder in place and writes ONLY into a
#                new folder gfig_<timestamp> beside the G run. The G run folder is checked to be
#                unchanged (file list, sizes, modification times) before and after.
# The renderer is presentation only (SEQREF-DLAT-GREN v0.1): it refuses any result file whose
# sha256 differs from g_report.json and computes no D-LAT measure, effect, interval, class or tag.
# Nothing is retried, resumed, forced or deleted (one temporary matplotlib cache folder excepted).
# Any failed check stops with an ERROR line and a non-zero exit. No fallback.
#
# Usage:  bash dlat_gfig_run.sh preflight
#         bash dlat_gfig_run.sh run
#
# Changelog:
#   v0.2 (2026-11-24) review of 17:09 UTC: C1 the bytecode scan is a full checked traversal and
#        fails closed (an incomplete scan is a failure of its own); the renderer is v0.2; renderer
#        and launcher are verified at the HEAD commit of ch2-diag by sha256 instead of a commit
#        hash written into this file (they are registered in one commit, whose hash cannot be
#        known in advance); the comparison with the sandbox rendering is removed (renderer v0.2
#        writes its version into every table, so no file could be equal).
#   v0.1 (2026-11-24) first version (draft for review; never registered, never run beyond preflight).
set -euo pipefail

WT="/home/benjamin/CSMFII/worktrees/ch2-diag"
DIR="seqref_mri/scripts/diag_lat"
BRANCH="ch2-diag"
RENDER_SHA="92913a896fa845489ae5c5db56ce914998d516d74554534fc76b9a1298c2cb86"   # dlat_g_render.py v0.2
SELF_NAME="dlat_gfig_run.sh"
EVID="/home/benjamin/CSMFII/v06lt_evidence/DIAG_LAT"
GRUN="$EVID/g_20261124T133246Z"                         # the single G run (LudWiki 257 LE104)
GREPORT_SHA="b4aeced199d558ec5c929836a843bae2f7f553d0a24846f00f335a047c3470d9"  # out/g_report.json
PY="/home/benjamin/CSMFII/seqref_mri/.venv_mri/bin/python"
MIN_FREE_BYTES=209715200                                # 200 MiB; the rendering writes about 5 MiB
ALLOW_ROOT=0

if [ "${DLAT_REHEARSAL:-0}" = "1" ]; then
  echo "#### REHEARSAL OVERRIDES ACTIVE -- NOT A HOST RUN ####"
  WT="${DLAT_WT:-$WT}"; EVID="${DLAT_EVID:-$EVID}"
  GRUN="${DLAT_GRUN:-$GRUN}"; GREPORT_SHA="${DLAT_GREPORT_SHA:-$GREPORT_SHA}"; PY="${DLAT_PY:-$PY}"
  ALLOW_ROOT="${DLAT_ALLOW_ROOT:-0}"   # the sandbox runs as root; never set on the host
fi

die() { echo "ERROR: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
listing() { find "$GRUN" -type f -printf '%P %s %T@\n' | LC_ALL=C sort | sha256sum | cut -d' ' -f1; }

preflight() {
  echo "== preflight $(date -u +%Y-%m-%dT%H:%M:%SZ) host $(hostname)"
  if [ "$(id -u)" = "0" ]; then
    [ "${ALLOW_ROOT:-0}" = "1" ] || die "do not run as root"
    echo "executing user: root (allowed in the rehearsal only)"
  else
    echo "executing user: $(id -un) (uid $(id -u)), not root"
  fi
  [ -d "$GRUN/out/figdata" ] || die "the G run folder $GRUN/out/figdata does not exist"
  local h; h=$(sha "$GRUN/out/g_report.json")
  [ "$h" = "$GREPORT_SHA" ] || die "g_report.json has sha256 $h, expected the registered $GREPORT_SHA"
  [ "$(tail -n 1 "$GRUN/out/g_verdict.txt")" = "G: COMPLETE" ] || die "the G run verdict is not 'G: COMPLETE'"
  echo "G run: $GRUN  g_report.json at the registered sha256 $GREPORT_SHA; verdict G: COMPLETE"
  [ "$(git -C "$WT" rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || die "the worktree $WT is not on branch $BRANCH"
  COMMIT=$(git -C "$WT" rev-parse HEAD)
  h=$(git -C "$WT" show "$COMMIT:$DIR/dlat_g_render.py" 2>/dev/null | sha256sum | cut -d' ' -f1 || true)
  [ "$h" = "$RENDER_SHA" ] || die "dlat_g_render.py at the HEAD commit $COMMIT has sha256 $h, expected the reviewed $RENDER_SHA (renderer v0.2 is not registered at HEAD)"
  echo "renderer: HEAD commit $COMMIT of $BRANCH holds dlat_g_render.py at its reviewed sha256 $RENDER_SHA"
  SELF_SHA=$(sha "$(readlink -f "$0")")
  h=$(git -C "$WT" show "$COMMIT:$DIR/$SELF_NAME" 2>/dev/null | sha256sum | cut -d' ' -f1 || true)
  [ "$h" = "$SELF_SHA" ] || die "this launcher (sha256 $SELF_SHA) is not the $SELF_NAME registered at the HEAD commit $COMMIT (sha256 there: $h); register it first"
  echo "launcher: this script, sha256 $SELF_SHA, is the one registered at the HEAD commit"
  [ -x "$PY" ] || die "the interpreter $PY does not exist or is not executable"
  # the packages are located and their versions read from their metadata; nothing is imported or written
  local v; v=$("$PY" -I -B -c "import sys, importlib.util as u, importlib.metadata as m; assert u.find_spec('matplotlib') and u.find_spec('numpy'); print(sys.version.split()[0], m.version('matplotlib'), m.version('numpy'))" 2>&1) \
    || die "matplotlib or numpy is not installed for the interpreter $PY: $v"
  echo "interpreter: $PY  python/matplotlib/numpy = $(echo "$v" | tail -n 1)"
  local free; free=$(df --output=avail -B1 "$EVID" | tail -n 1 | tr -d ' ')
  [ "$free" -ge "$MIN_FREE_BYTES" ] || die "free disk $free bytes below $MIN_FREE_BYTES"
  echo "resources: free disk $free bytes >= $MIN_FREE_BYTES"
  echo "PREFLIGHT PASSED"
}

case "${1:-}" in
  preflight) preflight ;;
  run)
    preflight
    TS=$(date -u +%Y%m%dT%H%M%SZ); RUN="$EVID/gfig_$TS"
    [ ! -e "$RUN" ] || die "the run folder $RUN already exists"
    mkdir -p "$RUN/code" "$RUN/logs"
    echo "run folder: $RUN"
    git -C "$WT" show "$COMMIT:$DIR/dlat_g_render.py" > "$RUN/code/dlat_g_render.py"
    [ "$(sha "$RUN/code/dlat_g_render.py")" = "$RENDER_SHA" ] || die "the snapshot of the renderer does not have the reviewed sha256"
    chmod -R a-w "$RUN/code"
    {
      echo "launcher SEQREF-DLAT-GFIG v0.2, step run"
      echo "launcher sha256 $SELF_SHA"
      echo "renderer commit $COMMIT sha256 $RENDER_SHA"
      echo "G run $GRUN g_report.json sha256 $GREPORT_SHA"
      echo "interpreter $PY"
      echo "started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$RUN/logs/run_context.txt"
    before=$(listing)
    MPL=$(mktemp -d)
    set +e
    MPLCONFIGDIR="$MPL" MPLBACKEND=Agg "$PY" -I -B "$RUN/code/dlat_g_render.py" \
      --g-out "$GRUN/out" --g-report-sha256 "$GREPORT_SHA" --out "$RUN/out" 2>&1 | tee "$RUN/logs/terminal.log"
    rc=${PIPESTATUS[0]}
    set -e
    rm -rf "$MPL"
    after=$(listing)
    echo "renderer finished $(date -u +%Y-%m-%dT%H:%M:%SZ) exit $rc" >> "$RUN/logs/run_context.txt"
    ok=1
    if [ "$rc" = "0" ] && [ -f "$RUN/out/render_verdict.txt" ] && [ "$(head -n 1 "$RUN/out/render_verdict.txt")" = "RENDER: COMPLETE" ]; then
      echo "renderer (exit 0 and the verdict RENDER: COMPLETE): PASS"
    else
      echo "renderer (exit $rc): FAIL"; ok=0
    fi
    if [ "$before" = "$after" ]; then echo "G run folder unchanged (file list, sizes, modification times): PASS"
    else echo "G run folder unchanged: FAIL"; ok=0; fi
    # full, checked traversal: an incomplete scan is a failure of its own and never reads as clean
    if scan=$(find "$RUN" \( -name '__pycache__' -o -name '*.pyc' -o -name '*.pyo' \) -print 2>"$RUN/logs/bytecode_scan_errors.txt"); then
      if [ -s "$RUN/logs/bytecode_scan_errors.txt" ]; then echo "bytecode scan incomplete (messages in logs/bytecode_scan_errors.txt): FAIL"; ok=0
      elif [ -n "$scan" ]; then echo "no bytecode written: FAIL ($(printf '%s\n' "$scan" | wc -l) entries)"; ok=0
      else echo "no bytecode written (complete traversal of the run folder): PASS"; fi
    else
      echo "bytecode scan incomplete (find failed; messages in logs/bytecode_scan_errors.txt): FAIL"; ok=0
    fi
    if [ "$ok" = "1" ]; then
      # evidence manifest: every file of out/ except the manifest itself, verdict and log included
      ( cd "$RUN/out" && find . -type f ! -name render_sha256.txt | LC_ALL=C sort | sed 's|^\./||' | xargs sha256sum > render_sha256.txt )
      echo "evidence manifest: $RUN/out/render_sha256.txt ($(wc -l < "$RUN/out/render_sha256.txt") files; the manifest itself is not listed)"
      echo "LAUNCHER VERDICT (run): PASS" | tee "$RUN/logs/launcher_verdict.txt"
    else
      echo "LAUNCHER VERDICT (run): FAIL" | tee "$RUN/logs/launcher_verdict.txt"
    fi
    ( cd "$RUN" && find . -type f ! -name run_sha256.txt | LC_ALL=C sort | sed 's|^\./||' | xargs sha256sum > run_sha256.txt )
    chmod -R a-w "$RUN"
    echo "every file of the run folder with its sha256: $RUN/run_sha256.txt  (sha256 $(sha "$RUN/run_sha256.txt"))"
    echo "The run folder stays read-only (evidence). Send back: this terminal output and the folder $RUN/out as one archive."
    [ "$ok" = "1" ] || exit 1
    ;;
  *) echo "usage: bash dlat_gfig_run.sh preflight|run" >&2; exit 2 ;;
esac
