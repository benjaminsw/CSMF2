#!/usr/bin/env bash
# SEQREF-MRI-V05LTLSH v0.3 -- v05lt_launch.sh (launcher wrapper, 195 §2)
# LIFETIME: KEEP
# Purpose: the launcher entry named in 195 §2 / §8.3 step 5. All logic is in
#   seqref_mri/scripts/v05lt_launch.py (SEQREF-MRI-V05LTL); this wrapper only
#   enters the repository, selects the registered virtual environment and
#   passes the arguments through. Errors are logged; the Python exit code is
#   returned unchanged. Production needs --chain and --approved-sha256
#   (launcher v0.4); it remains impossible until the pin, gate 2a, P1-P6,
#   the launch-inputs record and the recorded launch approval exist.
#   The separate `holdout` command (launcher v0.6) is passed through like
#   any other mode; it needs --holdout-approval PATH:SHA256 recorded in
#   V05LTEXEC and is never triggered by `new` / `resume`.
# Changelog (v0.3, 2026-10-27; V05LTI19C4 v0.1): header text only (the
#   `holdout` command documented); behaviour unchanged.
# Changelog (v0.2, 2026-10-26; V05LTI19C2P v0.1): header text only (the
#   c1 refusal wording replaced); behaviour unchanged.
# Changelog (NEW in v0.1, 2026-10-25; I1.9 c1, SEQREF-MRI-V05LTI19 v0.1):
#   * Thin wrapper; no gate or training logic here.
set -u
REPO="${REPO:-/home/benjamin/CSMFII}"
PY="$REPO/seqref_mri/.venv_mri/bin/python"
log() { echo "$(date -u +%FT%TZ) SEQREF-MRI-V05LTLSH $*" >&2; }
cd "$REPO" || { log "ERROR LAUNCH_WRAPPER_REPO: cannot enter $REPO"; exit 2; }
[ -x "$PY" ] || { log "ERROR LAUNCH_WRAPPER_PYTHON: $PY not executable"; exit 2; }
"$PY" -m seqref_mri.scripts.v05lt_launch "$@"
rc=$?
[ "$rc" -eq 0 ] || log "ERROR LAUNCH_EXIT: v05lt_launch exited $rc"
exit "$rc"
