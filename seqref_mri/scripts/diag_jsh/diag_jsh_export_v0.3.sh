#!/usr/bin/env bash
# SEQREF-MRI-DIAGJSHEXP v0.3 -- export reviewed DIAG-JSH run files from a ch2-diag commit to v06lt_evidence/DIAG_JSH/code with CSMFII back on the V05LT pin
# LIFETIME: KEEP
# =============================================================================
# Plan SEQREF-MRI-DEVTREE v0.3, step 3. Run by the owner on the host, read
# from Git so it works on any branch:
#   bash <(git -C /home/benjamin/CSMFII show ch2-diag:seqref_mri/scripts/diag_jsh/diag_jsh_export.sh) COMMIT
# Steps (stops at the first failure; exit 2). Every inspection command's
# exit status is checked separately from its output:
#   0. no V05LT, V06LT or DIAG-JSH job running (before anything changes);
#   1. CSMFII clean (git status --porcelain empty; v06lt_evidence/ is
#      excluded locally);
#   2. COMMIT exists, is on ch2-diag and on origin/ch2-diag (pushed);
#   3. switch to seqref-v05lt (only after 0-2); HEAD = pin 7455da80...;
#      clean; the pinned source scan's view (every *.py under seqref_mri/
#      and every file under seqref_mri/configs/, skipping any path with a
#      part in {.venv_mri, results, data, __pycache__}, as SOURCE_EXCLUDE in
#      v05lt_prodstages) holds only files tracked at the pin and no symlink;
#   4. export the 5 run files from COMMIT to a temporary folder; each = its
#      blob in COMMIT = the reviewed sha256 below;
#   5. destination ancestors validated before anything is created
#      (CSMFII, v06lt_evidence, DIAG_JSH, out, code: real directories, not
#      symlinks); code/ created if absent; if present it must hold exactly
#      the 5 files, each a regular non-symlink file with the reviewed
#      content (then the deployed files are left unchanged; the helper
#      still fetches from origin and creates and removes its own temporary
#      folder in v06lt_evidence/). Never overwrites or deletes deployed files.
# Prints RESULT DIAGJSH_EXPORT PASS commit=... and the 5 hashes.
# Not done here: the F1 run itself (separate reviewed command and GO).
# Changelog (v0.3, 2026-11-10; reviewer finding on v0.2): the source-scan
#   comparison no longer uses process substitution (a failed producer was
#   accepted as an empty list); both sorted lists are written to files with
#   each command's status checked, then compared; wording of the
#   identical-repeat case corrected.
# Changelog (v0.2, 2026-11-10; reviewer findings on v0.1): running-job
#   check before switching; extra-file scan mirrors the pinned exclusions
#   (venv excluded) and refuses symlinks; command status checked
#   separately from output (no "|| true"); exported files must be regular
#   non-symlink files; destination ancestors validated before creation.
# Changelog (v0.1, 2026-11-10): first export helper.
# =============================================================================
set -euo pipefail
die() { echo "STOP: $*" >&2; exit 2; }
V05=/home/benjamin/CSMFII
PIN=7455da8027f33a6cca4dbf72f04e72420e938f45
PINBR=seqref-v05lt
DEVBR=ch2-diag
SRC=seqref_mri/scripts/diag_jsh
EV=$V05/v06lt_evidence
DJ=$EV/DIAG_JSH
JOBS='seqref_mri\.scripts\.(v05lt|v06lt)_|diag_jsh_'
C=${1:-}
[[ "$C" =~ ^[0-9a-f]{40}$ ]] || die "usage: bash <(git show $DEVBR:$SRC/diag_jsh_export.sh) FULL_40_CHAR_COMMIT"
REVIEWED=$(cat <<'TAB'
diag_jsh_core.py 85410c77472cf876021a49bd72254ac98461227887178e24cdb6d0ce5295c0fb
diag_jsh_adapter.py c566858189be4a3dee822ce4eef46b156986881a92cb6d4360b386ca785707c1
diag_jsh_f1_prepare.py 3472714752948d5fbf63d7a96ffa64db2b787d6f9a40b6720e944537242fd739
diag_jsh_f1_cmd.sh 0d67a2df4cbec089eccace3bd15885e0d1eb5c12e477d51cd88f519dfe9b1d93
diag_jsh_i3.py 70f6b57511f62b3f178fa53ffcda938740b584e491da139895f17396e9aa1dca
TAB
)
NAMES=$(cut -d' ' -f1 <<< "$REVIEWED" | sort)
realdir() { [[ -d "$1" && ! -L "$1" && "$(realpath -e -- "$1")" == "$1" ]]; }
realdir "$V05" || die "$V05 missing or reached through a symlink"
cd "$V05" || die "cannot cd $V05"

# 0. running jobs (this helper's own processes are excluded by name)
set +e; PG=$(pgrep -af "$JOBS"); PRC=$?; set -e
case $PRC in
  0) set +e; PG2=$(grep -v 'diag_jsh_export' <<< "$PG"); GRC=$?; set -e
     case $GRC in 0) printf '%s\n' "$PG2" >&2; die "a V05LT/V06LT/DIAG-JSH job is running (wait; do not stop it)" ;;
       1) ;; *) die "grep failed (rc $GRC)" ;; esac ;;
  1) ;;
  *) die "pgrep failed (rc $PRC)" ;;
esac
echo "JOBS none running"

# 1. clean
ST=$(git status --porcelain) || die "git status failed"
[[ -z "$ST" ]] || { printf '%s\n' "$ST" >&2; die "CSMFII not clean (commit on $DEVBR first)"; }

# 2. commit identity
T=$(git cat-file -t "$C" 2>/dev/null) || die "commit $C not found"
[[ "$T" == commit ]] || die "$C is not a commit"
git merge-base --is-ancestor "$C" "$DEVBR" || die "$C is not on $DEVBR"
git fetch --quiet origin "$DEVBR" || die "git fetch origin $DEVBR failed"
git merge-base --is-ancestor "$C" "origin/$DEVBR" || die "$C is not pushed (not on origin/$DEVBR)"
CD=$(git log -1 --format=%cI "$C") || die "git log failed"
echo "COMMIT $C ($CD)"

# 3. back on the pin
BRN=$(git branch --show-current) || die "git branch failed"
if [[ "$BRN" != "$PINBR" ]]; then git switch --quiet "$PINBR" || die "git switch $PINBR failed"; fi
H=$(git rev-parse HEAD) || die "git rev-parse failed"
[[ "$H" == "$PIN" ]] || die "HEAD $H != pin $PIN"
ST=$(git status --porcelain) || die "git status failed"
[[ -z "$ST" ]] || die "not clean after switching to $PINBR"
[[ ! -e "$SRC" && ! -L "$SRC" ]] || die "$SRC still present on $PINBR"
# work folder (inside v06lt_evidence, outside seqref_mri); created only after steps 0-2
realdir "$EV" || die "$EV missing or reached through a symlink"
WORK=$(mktemp -d "$EV/.diagjsh_export.XXXXXX") || die "mktemp failed"
trap 'rm -rf -- "$WORK"' EXIT
TMP=$WORK/files
mkdir -- "$TMP" || die "cannot create $TMP"
TRK=$(git ls-files -- seqref_mri) || die "git ls-files failed"
SCAN=$(find seqref_mri \( -name .venv_mri -o -name results -o -name data -o -name __pycache__ \) -prune \
       -o \( -type l -printf 'L %p\n' \) -o \( -type f \( -name '*.py' -o -path 'seqref_mri/configs/*' \) -printf 'F %p\n' \)) \
       || die "find over seqref_mri failed"
set +e; LINKS=$(grep '^L ' <<< "$SCAN"); GRC=$?; set -e
case $GRC in 0) printf '%s\n' "$LINKS" >&2; die "symlink in the pinned source scan" ;; 1) ;; *) die "grep failed (rc $GRC)" ;; esac
sed -n 's/^F //p' <<< "$SCAN" > "$WORK/scan.raw" || die "sed over the source scan failed"
LC_ALL=C sort "$WORK/scan.raw" > "$WORK/scan.lst" || die "sort of the source scan failed"
printf '%s\n' "$TRK" > "$WORK/trk.raw" || die "writing the tracked list failed"
LC_ALL=C sort "$WORK/trk.raw" > "$WORK/trk.lst" || die "sort of the tracked list failed"
EXTRA=$(LC_ALL=C comm -23 "$WORK/scan.lst" "$WORK/trk.lst") || die "comm failed"
[[ -z "$EXTRA" ]] || { printf '%s\n' "$EXTRA" >&2; die "file(s) not tracked at the pin in the pinned source scan (would cause PROD_PIN_MISMATCH)"; }
echo "PIN OK $PIN on $PINBR; source scan matches tracked files"

# 4. export to a temporary folder (inside v06lt_evidence, outside seqref_mri)
while read -r n h; do
  git show "$C:$SRC/$n" > "$TMP/$n" || die "$n not in $C"
  [[ "$(sha256sum "$TMP/$n" | cut -d' ' -f1)" == "$h" ]] || die "$n at $C != reviewed $h"
done <<< "$REVIEWED"

# 5. place (no overwrite, no delete); ancestors validated before creation
for d in "$DJ" "$DJ/out"; do
  if [[ -e "$d" || -L "$d" ]]; then realdir "$d" || die "$d is not a plain directory"
  else mkdir -- "$d" || die "cannot create $d"; realdir "$d" || die "$d check failed"; fi
done
CODE=$DJ/code
if [[ -e "$CODE" || -L "$CODE" ]]; then
  realdir "$CODE" || die "$CODE is not a plain directory"
  HAVE=$(find "$CODE" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort) || die "find over $CODE failed"
  [[ "$HAVE" == "$NAMES" ]] || { printf '%s\n' "$HAVE" >&2; die "$CODE holds other files (move them aside first)"; }
  while read -r n h; do
    [[ -f "$CODE/$n" && ! -L "$CODE/$n" ]] || die "$CODE/$n is not a regular file"
    cmp -s "$TMP/$n" "$CODE/$n" || die "$CODE/$n differs from $C (move $CODE aside first)"
  done <<< "$REVIEWED"
  echo "CODE already identical to $C (deployed files unchanged)"
else
  mkdir -- "$CODE" || die "cannot create $CODE"
  realdir "$CODE" || die "$CODE check failed"
  while read -r n h; do
    cp -- "$TMP/$n" "$CODE/$n" || die "copy $n failed"
    chmod a-w "$CODE/$n" || die "chmod $n failed"
  done <<< "$REVIEWED"
fi
while read -r n h; do
  [[ -f "$CODE/$n" && ! -L "$CODE/$n" ]] || die "placed $n is not a regular file"
  [[ "$(sha256sum "$CODE/$n" | cut -d' ' -f1)" == "$h" ]] || die "placed $n != reviewed"
done <<< "$REVIEWED"
echo "RESULT DIAGJSH_EXPORT PASS commit=$C pin=$PIN code=$CODE"
sha256sum "$CODE"/*
echo "Next: the reviewed F1 command (only after its GO). Afterwards: git switch $DEVBR"
