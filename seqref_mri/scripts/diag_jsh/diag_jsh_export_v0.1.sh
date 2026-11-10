#!/usr/bin/env bash
# SEQREF-MRI-DIAGJSHEXP v0.1 -- export reviewed DIAG-JSH run files from a ch2-diag commit to v06lt_evidence/DIAG_JSH/code with CSMFII back on the V05LT pin
# LIFETIME: KEEP
# =============================================================================
# Plan SEQREF-MRI-DEVTREE v0.3, step 3. Run by the owner on the host, read
# from Git so it works on any branch:
#   bash <(git -C /home/benjamin/CSMFII show ch2-diag:seqref_mri/scripts/diag_jsh/diag_jsh_export.sh) COMMIT
# Steps (stops at the first failure; exit 2):
#   1. CSMFII has no uncommitted tracked change and no untracked file
#      (v06lt_evidence/ is excluded locally);
#   2. COMMIT exists, is on branch ch2-diag, and is on origin/ch2-diag
#      (pushed); recorded;
#   3. switch to seqref-v05lt; HEAD must be the pin 7455da80...; no .py
#      file under seqref_mri/ other than tracked ones (the pinned source
#      fingerprint covers every *.py there);
#   4. export the 5 run files from COMMIT; each = its blob in COMMIT and =
#      the reviewed sha256 below;
#   5. place them in $DJ/code: created if absent; if present it must hold
#      exactly these 5 files with identical content (then nothing is
#      written). Never overwrites or deletes; creates $DJ/out if absent.
# Prints RESULT DIAGJSH_EXPORT PASS commit=... and the 5 hashes.
# Not done here: the F1 run itself (separate reviewed command and GO).
# Changelog (v0.1, 2026-11-10): first export helper.
# =============================================================================
set -euo pipefail
die() { echo "STOP: $*" >&2; exit 2; }
V05=/home/benjamin/CSMFII
PIN=7455da8027f33a6cca4dbf72f04e72420e938f45
PINBR=seqref-v05lt
DEVBR=ch2-diag
SRC=seqref_mri/scripts/diag_jsh
DJ=$V05/v06lt_evidence/DIAG_JSH
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
cd "$V05" || die "cannot cd $V05"
# 1. clean
[[ -z "$(git status --porcelain)" ]] || { git status --short >&2; die "CSMFII not clean (commit on $DEVBR first)"; }
# 2. commit identity
[[ "$(git cat-file -t "$C" 2>/dev/null)" == commit ]] || die "commit $C not found"
git merge-base --is-ancestor "$C" "$DEVBR" || die "$C is not on $DEVBR"
git fetch --quiet origin "$DEVBR" || die "git fetch origin $DEVBR failed"
git merge-base --is-ancestor "$C" "origin/$DEVBR" || die "$C is not pushed (not on origin/$DEVBR)"
echo "COMMIT $C ($(git log -1 --format=%cI "$C"))"
# 3. back on the pin
[[ "$(git branch --show-current)" == "$PINBR" ]] || git switch --quiet "$PINBR" || die "git switch $PINBR failed"
[[ "$(git rev-parse HEAD)" == "$PIN" ]] || die "HEAD != pin $PIN"
[[ -z "$(git status --porcelain)" ]] || die "not clean after switching to $PINBR"
EXTRA=$(git ls-files --others --ignored --exclude-standard -- seqref_mri | grep '\.py$' || true)
[[ -z "$EXTRA" ]] || { printf '%s\n' "$EXTRA" >&2; die "untracked/ignored .py under seqref_mri (breaks the pinned fingerprint)"; }
[[ ! -e "$SRC" ]] || die "$SRC still present on $PINBR"
echo "PIN OK $PIN on $PINBR"
# 4. export to a temporary folder outside seqref_mri, check hashes
[[ -d "$V05/v06lt_evidence" && ! -L "$V05/v06lt_evidence" ]] || die "v06lt_evidence missing or a symlink"
TMP=$(mktemp -d "$V05/v06lt_evidence/.diagjsh_export.XXXXXX") || die "mktemp failed"
trap 'rm -rf -- "$TMP"' EXIT
while read -r n h; do
  [[ -n "$n" ]] || continue
  git show "$C:$SRC/$n" > "$TMP/$n" || die "$n not in $C"
  [[ "$(sha256sum "$TMP/$n" | cut -d' ' -f1)" == "$h" ]] || die "$n at $C != reviewed $h"
done <<< "$REVIEWED"
# 5. place (no overwrite, no delete)
for d in "$DJ" "$DJ/out"; do
  [[ -d "$d" ]] || mkdir -- "$d" || die "cannot create $d"
  [[ "$(realpath -e -- "$d")" == "$d" ]] || die "$d reached through a symlink"
done
if [[ -e "$DJ/code" || -L "$DJ/code" ]]; then
  [[ -d "$DJ/code" && "$(realpath -e -- "$DJ/code")" == "$DJ/code" ]] || die "$DJ/code not a plain folder"
  [[ "$(ls -A "$DJ/code" | sort)" == "$(ls -A "$TMP" | sort)" ]] || die "$DJ/code holds other files (move them aside first)"
  diff -rq "$TMP" "$DJ/code" >/dev/null || die "$DJ/code differs from $C (move it aside first)"
  echo "CODE already identical to $C (nothing written)"
else
  mkdir -- "$DJ/code" || die "cannot create $DJ/code"
  while read -r n h; do [[ -n "$n" ]] || continue
    cp -- "$TMP/$n" "$DJ/code/$n"; done <<< "$REVIEWED"
  chmod a-w "$DJ/code"/* || die "chmod failed"
fi
while read -r n h; do [[ -n "$n" ]] || continue
  [[ "$(sha256sum "$DJ/code/$n" | cut -d' ' -f1)" == "$h" ]] || die "placed $n != reviewed"; done <<< "$REVIEWED"
echo "RESULT DIAGJSH_EXPORT PASS commit=$C pin=$PIN code=$DJ/code"
sha256sum "$DJ/code"/*
echo "Next: the reviewed F1 command (only after its GO). Afterwards: git switch $DEVBR"
