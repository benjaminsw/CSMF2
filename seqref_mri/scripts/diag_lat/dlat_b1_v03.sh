#!/usr/bin/env bash
# SEQREF-DLATB1 v0.3.1 -- dlat_b1_v03.sh: launcher for the D-LAT B1 pass-2 read-only host check
# LIFETIME: DIAGNOSTIC
#
# Runs dlat_b1_v03.py (same folder) with the isolated system Python (-I -S: no site packages): no project import,
# no checkpoint load, no MRI file, no inference. Writes only a new folder under
#   /home/benjamin/CSMFII/seqref_mri/results/_diag/dlat_b1/
# Usage:
#   bash dlat_b1_v03.sh                       # runs root = the single folder holding the six runs
#   bash dlat_b1_v03.sh --runs-root <folder>  # if that folder is not unique (reported UNRESOLVED)
# Exit: 0 no FAIL/UNRESOLVED | 1 at least one FAIL or UNRESOLVED | 2 internal error
#
# Changelog:
#   v0.3.1 (2026-11-20) bug fix after review: Python started with -I -S so that no site package
#          (numpy and the like) is loaded at interpreter start.
#   v0.3 (2026-11-20) launches the v0.3 checker (F2 manifest as inventory); --manifest removed.
#   v0.2 (2026-11-20) first version (not run on the host).
set -euo pipefail
REPO=/home/benjamin/CSMFII
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY="$HERE/dlat_b1_v03.py"
if [ ! -f "$PY" ]; then
  echo "DLATB1 ERROR: $PY not found" >&2
  exit 2
fi
if [ ! -d "$REPO/.git" ]; then
  echo "DLATB1 ERROR: $REPO is not a git repository" >&2
  exit 2
fi
exec python3 -I -S "$PY" --repo "$REPO" "$@"
