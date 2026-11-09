#!/usr/bin/env bash
# SEQREF-MRI-ARCHPUSH v0.1 -- archive G-D4 / G-D1 host scripts and the DIAG-JSH sandbox snapshot to benjaminsw/CSMF2 (separate checkout, orphan branch)
# LIFETIME: KEEP
# =============================================================================
# Code-preservation rule (DOCTREE 165 §4). Run by the owner on the host.
#   * Separate fresh clone ARC (never the pinned V05LT tree /home/benjamin/CSMFII).
#   * New orphan branch BR (no shared history with CSMF2 main); refuses if BR
#     already exists on the remote; plain push, never --force.
#   * Host files are copied read-only from the evidence folders and checked
#     against their recorded sha256 before commit; the sandbox bundle is
#     checked against its SHA256SUMS.sandbox.
#   * This script copies itself into the commit (archive_cmd/).
#   * Push verified: remote branch head == local HEAD.
# Prints: commit SHA, remote verification, per-file sha256 and git blob id.
# Exit 0 OK; 2 any failure (nothing pushed unless every check passed).
# NOTE: benjaminsw/CSMF2 is PUBLIC; the archive (code, logs, host paths,
#   hashes) becomes public. No credentials, data, checkpoints or results.
# Usage: bash seqref_archive_push_v0.1.sh BUNDLE_ZIP BUNDLE_ZIP_SHA256
# Changelog (v0.1, 2026-11-09): first archive command.
# =============================================================================
set -euo pipefail
die() { echo "STOP: $*" >&2; exit 2; }
SELF=$(realpath -e -- "$0") || die "cannot resolve the script path"
ZIP=$(realpath -e -- "${1:-/nonexistent}" 2>/dev/null) || die "usage: $0 BUNDLE_ZIP BUNDLE_ZIP_SHA256"
ZSHA=${2:-}
[[ -f "$ZIP" && "$ZSHA" =~ ^[0-9a-f]{64}$ ]] || die "usage: $0 BUNDLE_ZIP BUNDLE_ZIP_SHA256"
[[ "$(sha256sum "$ZIP" | cut -d' ' -f1)" == "$ZSHA" ]] || die "bundle hash != $ZSHA"
REPO=https://github.com/benjaminsw/CSMF2.git
BR=seqref-archive
ARC=/home/benjamin/seqref_archive/CSMF2
D=archive/2026-11-09
EV=/home/benjamin/CSMFII/v06lt_evidence
declare -A HOST=(
  ["$EV/V05LT_analysis/v06lt_v05lt_analysis_g4_v0.2.sh"]=fe2785d9bba0947e4582b4a49d75b315a89a6f8583dd0abab8633c44ba0a7233
  ["$EV/V05LT_analysis/v06lt_v05lt_analysis_g4_driver.py"]=8885726427e740c8dba5e6cbfa7bd656ee3aa7117826441bb2c36bd8619a51c5
  ["$EV/V05LT_completion/v06lt_v05lt_completion_check_v0.2.sh"]=1b5acf700f1d535a3247e52686fc451e7fd0fa95b2daa967b73454c22083c617
)
declare -A DEST=(
  ["$EV/V05LT_analysis/v06lt_v05lt_analysis_g4_v0.2.sh"]=G-D4
  ["$EV/V05LT_analysis/v06lt_v05lt_analysis_g4_driver.py"]=G-D4
  ["$EV/V05LT_completion/v06lt_v05lt_completion_check_v0.2.sh"]=G-D1
)
for f in "${!HOST[@]}"; do
  [[ -f "$f" && ! -L "$f" ]] || die "missing: $f"
  [[ "$(sha256sum "$f" | cut -d' ' -f1)" == "${HOST[$f]}" ]] || die "hash != recorded: $f"
done
[[ ! -e "$ARC" && ! -L "$ARC" ]] || die "$ARC exists (use a fresh checkout)"
[[ -z "$(git ls-remote --heads "$REPO" "$BR")" ]] || die "remote branch $BR already exists"
mkdir -p "$(dirname "$ARC")"
git clone --quiet --no-checkout "$REPO" "$ARC"
cd "$ARC"
git switch --quiet --orphan "$BR"
git rm -rq --cached . 2>/dev/null || true
mkdir -p "$D"
unzip -q "$ZIP" -d "$D"
( cd "$D" && sha256sum -c --quiet SHA256SUMS.sandbox ) || die "bundle content check failed"
for f in "${!HOST[@]}"; do
  mkdir -p "$D/${DEST[$f]}"; cp -p -- "$f" "$D/${DEST[$f]}/"
done
mkdir -p "$D/archive_cmd"; cp -p -- "$SELF" "$D/archive_cmd/"
( cd "$D" && sha256sum $(find . -type f ! -name SHA256SUMS | sort) > SHA256SUMS )
for f in "${!HOST[@]}"; do
  grep -q "^${HOST[$f]}  ./${DEST[$f]}/$(basename "$f")\$" "$D/SHA256SUMS" || die "archived copy differs: $f"
done
git add -- "$D"
git commit --quiet \
  -m "SEQREF-MRI archive 2026-11-09: G-D4/G-D1 executed scripts, DIAG-JSH reviewed sandbox files (DOCTREE 165 §4)"
HEAD_SHA=$(git rev-parse HEAD)
git push --quiet -u origin "$BR"
REMOTE_SHA=$(git ls-remote --heads origin "$BR" | cut -f1)
[[ "$REMOTE_SHA" == "$HEAD_SHA" ]] || die "push not verified: remote $REMOTE_SHA != local $HEAD_SHA"
echo "RESULT ARCHIVE_PUSHED repo=benjaminsw/CSMF2 branch=$BR commit=$HEAD_SHA remote=$REMOTE_SHA path=$D"
echo "--- sha256 (archive) ---"; cat "$D/SHA256SUMS"
echo "--- git blobs ---"; git ls-tree -r HEAD -- "$D" | awk '{print $3"  "$4}'
