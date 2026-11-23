#!/usr/bin/env bash
# SEQREF-DLAT-F1ARCH v0.2 -- dlat_f1_archive.sh: Git registration of the reviewed D-LAT F1 package on ch2-diag (V12; DOCTREE 165 section 4)
# LIFETIME: KEEP
#
# DRAFT FOR REVIEW. 'commit' and 'push' MODIFY GIT: do not run them before the complete F1
# package is reviewed and the owner has authorised V12 (LudWiki 250). 'check' is read-only.
#
# What it does:
#   check            -> read-only: branch ch2-diag; local HEAD and the remote branch both at the
#                       registered start commit; every package file present in the worktree at its
#                       reviewed sha256; nothing staged. States for each file whether HEAD already
#                       holds the same bytes, other bytes, or no such file.
#   commit <sha256>  -> 'check', then ONE commit of exactly the package files that differ from HEAD
#                       and of this script (its reviewed sha256 is the argument and is verified on
#                       the copy in the worktree and on the copy being run). Afterwards: the
#                       commit changes only package files; every package file at HEAD has its
#                       reviewed sha256.
#   push <sha256>    -> HEAD must be exactly one ordinary commit above the start commit; pushes
#                       ch2-diag and verifies the remote by fetch and by ls-remote.
# The pinned V05LT checkout is not touched: this script works in the ch2-diag worktree only.
# Nothing is ever forced, amended, rebased or deleted. Any failed check stops with an ERROR line
# and a non-zero exit; a wrong staged set is unstaged before stopping. No fallback, no retry.
#
# Usage:  bash dlat_f1_archive.sh check
#         bash dlat_f1_archive.sh commit <reviewed sha256 of dlat_f1_archive.sh>
#         bash dlat_f1_archive.sh push   <reviewed sha256 of dlat_f1_archive.sh>
#
# Changelog:
#   v0.2 (2026-11-23) reviewed sha256 of diag_lat_f1.py v0.3, diag_lat_f1test.py v0.3 and dlat_f1_run.sh v0.2
#        (LudWiki 250 LE86). No command or check changed.
#   v0.1 (2026-11-23) first version (draft for review), on the pattern of dlat_p3_archive.sh v0.7.1;
#        one commit instead of two (the script's own sha256 is passed as an argument).
set -euo pipefail

WT="/home/benjamin/CSMFII/worktrees/ch2-diag"
BRANCH="ch2-diag"
DIR="seqref_mri/scripts/diag_lat"
START="fddffbd80cea2cc96c2ea1f75baccabf0565cc1f"      # head of ch2-diag, local and remote (LudWiki 245 LE65)
SELF_NAME="dlat_f1_archive.sh"
MSG="D-LAT F1 package: driver, binder, population stage, revised adapter/stats/tests, run card, launchers

Frozen specification SEQREF-MRI-DIAG-LAT v1.0 (LudWiki 235). Plans SEQREF-DLAT-F1 v0.2 and
SEQREF-DLAT-F1D v0.1; reviewed in LudWiki 247-250. Preparation code: no D-LAT evidence.

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Df3EoZnqFQN2xsHZ7sBL1x"

# The package: every file with its reviewed sha256 (this script excepted; its sha256 is the argument).
FILES=(diag_lat_core.py diag_lat_guard.py diag_lat_adapter.py diag_lat_stats.py diag_lat_population.py diag_lat_bind.py diag_lat_f1.py
       diag_lat_selftest.py diag_lat_p3test.py diag_lat_adptest.py diag_lat_poptest.py diag_lat_bindtest.py diag_lat_f1test.py
       dlat_f1_run_card_v0.2.json dlat_f1_run.sh)
declare -A SHA=(
  [diag_lat_core.py]=acfccb4099e204870c6c82ab8876ad9682aa87d465b0f845c6960a18a7697bc9
  [diag_lat_guard.py]=4c78796aa3185c54d53e73ac251fc0a8c7e59fd45509dfea490cf9221d3b76f3
  [diag_lat_adapter.py]=c6dc185af42b3e99e9859f74bb810e6382e3e7ed6246a9d33ff7b01d04a5a621
  [diag_lat_stats.py]=47f3c0358cd16fb179868fbebe6d68cd1d0c7370c098a71ae36fdd17d7ece831
  [diag_lat_population.py]=f695272d834feb8801dde769fa7d7d7e62edd27cd230f32bd96037df791085f9
  [diag_lat_bind.py]=2793db5283779d6d836cdd76704efaf415d689c7e289c97c273ba0f8af7697c2
  [diag_lat_f1.py]=ea0f177f65101b29dce170ac216e044dd7af068b62800df68169f56172370be0
  [diag_lat_selftest.py]=397e722542c855337a029dfff0451efca02c0ebe31a9659e299f30ed8f7361c4
  [diag_lat_p3test.py]=04b4ab02c18e97af6e784c52cef3e3a84d95fc08570b6f72d546460804960746
  [diag_lat_adptest.py]=ea48385ac5a29729eaa5fd376ae921b2a0bc7d0a041e8651e8fbc5338b89bdd6
  [diag_lat_poptest.py]=24b968710e96c1c27cd5a6c285699f0bf351ed8e82bba0a7cc513850a30226b9
  [diag_lat_bindtest.py]=ec824e144d9e59671f443d2cbd622705e8ee85eb9404d0a88e8ed0e4be841887
  [diag_lat_f1test.py]=83de4430ff08314cf2f3f4e316ba97a0eae12a46031b920ce9be256e1e8702f5
  [dlat_f1_run_card_v0.2.json]=c4325b7d9543ee5836a9c101e37c1e74b76ac3d5b21b68576ff4c78a930729eb
  [dlat_f1_run.sh]=3eff06550f646cd4a1abaeaf6a164da1a4ee022b30bf1cb7be7d70722ff1bbd0
)

if [ "${DLAT_REHEARSAL:-0}" = "1" ]; then
  echo "#### REHEARSAL OVERRIDES ACTIVE -- NOT A HOST RUN ####"
  WT="${DLAT_WT:-$WT}"; START="${DLAT_START:-$START}"
fi

die() { echo "ERROR: $*" >&2; exit 1; }
SELF=$(readlink -f "$0")

remote_head() {   # read-only: the commit the remote branch points at; stops if it cannot be read
  local out
  out=$(git ls-remote --heads origin "$BRANCH") || die "the remote branch $BRANCH could not be read (git ls-remote failed); nothing is changed"
  out=$(printf '%s\n' "$out" | awk -v r="refs/heads/$BRANCH" '$2==r{print $1}')
  [ -n "$out" ] || die "the remote has no branch $BRANCH"
  echo "$out"
}

enter() {
  [ -d "$WT" ] || die "worktree not found: $WT"
  cd "$WT"
  [ "$(git rev-parse --show-toplevel)" = "$(readlink -f "$WT")" ] || die "not the top of the ch2-diag worktree"
  [ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || die "branch is $(git rev-parse --abbrev-ref HEAD), expected $BRANCH"
  git cat-file -e "$START^{commit}" 2>/dev/null || die "the registered start commit $START is not in this repository"
}

manifest() { { for f in "${FILES[@]}"; do echo "$DIR/$f"; done; echo "$DIR/$SELF_NAME"; } | sort; }

changed_in() { git diff-tree --no-commit-id --name-only -r "$1" | sort; }

self_check() {   # $1 reviewed sha256 of this script: the copy in the worktree and the copy being run
  local got
  [ -n "$1" ] || die "the reviewed sha256 of $SELF_NAME is required as the second argument"
  [ -f "$DIR/$SELF_NAME" ] || die "missing file: $DIR/$SELF_NAME"
  got=$(sha256sum "$DIR/$SELF_NAME" | cut -d' ' -f1)
  [ "$got" = "$1" ] || die "$SELF_NAME in the worktree has sha256 $got, expected $1"
  [ "$(sha256sum "$SELF" | cut -d' ' -f1)" = "$1" ] || die "the script being run does not have the expected sha256 $1"
}

check() {
  local f got at_head r h n_new=0
  enter
  h=$(git rev-parse HEAD) || exit 1
  echo "worktree $WT  branch $BRANCH  HEAD $h"
  [ "$h" = "$START" ] || die "HEAD is $h, expected the registered start commit $START"
  r=$(remote_head) || exit 1
  [ "$r" = "$START" ] || die "the remote branch $BRANCH is at $r, expected the registered start commit $START; nothing is changed"
  echo "branch position: local HEAD and remote $BRANCH both at the registered start commit $START"
  for f in "${FILES[@]}"; do
    [ -f "$DIR/$f" ] || die "missing file: $DIR/$f"
    got=$(sha256sum "$DIR/$f" | cut -d' ' -f1)
    [ "$got" = "${SHA[$f]}" ] || die "sha256 mismatch for $f: got $got, reviewed ${SHA[$f]}"
    if git cat-file -e "HEAD:$DIR/$f" 2>/dev/null; then
      at_head=$(git show "HEAD:$DIR/$f" | sha256sum | cut -d' ' -f1)
      if [ "$at_head" = "$got" ]; then echo "OK (already archived at HEAD, unchanged)  $got  $DIR/$f"
      else echo "OK (reviewed version to archive; HEAD holds other bytes, sha256 $at_head)  $got  $DIR/$f"; n_new=$((n_new+1)); fi
    else
      echo "OK (reviewed version to archive; not at HEAD)  $got  $DIR/$f"; n_new=$((n_new+1))
    fi
  done
  [ -z "$(git diff --cached --name-only)" ] || die "something is already staged; unstage it first: $(git diff --cached --name-only | tr '\n' ' ')"
  echo "CHECK PASSED: ${#FILES[@]} package files at their reviewed sha256 ($n_new to archive); nothing staged"
}

case "${1:-}" in
  check)
    check ;;
  commit)
    check
    self_check "${2:-}"
    for f in "${FILES[@]}" "$SELF_NAME"; do git add -f -- "$DIR/$f"; done
    staged=$(git diff --cached --name-only | sort)
    [ -n "$staged" ] || die "nothing to commit: every package file is already identical at HEAD"
    extra=$(comm -23 <(printf '%s\n' "$staged") <(manifest))
    if [ -n "$extra" ]; then git reset -q; die "the staged set holds a file outside the package (staging undone; nothing committed): $(echo $extra)"; fi
    git commit -q -m "$MSG"
    echo "COMMITTED $(git rev-parse HEAD)"
    git show --stat --format='%H%n%an <%ae>%n%ad%n%s' HEAD
    [ "$(git rev-parse HEAD~1)" = "$START" ] || die "the commit is not directly above the registered start commit"
    [ -z "$(comm -23 <(changed_in HEAD) <(manifest))" ] || die "the commit changes a file outside the package: $(comm -23 <(changed_in HEAD) <(manifest) | tr '\n' ' ')"
    echo "COMMIT-SCOPE-OK: the commit changes only package files ($(changed_in HEAD | wc -l) files)"
    for f in "${FILES[@]}"; do
      got=$(git show "HEAD:$DIR/$f" | sha256sum | cut -d' ' -f1)
      [ "$got" = "${SHA[$f]}" ] || die "committed content of $f has sha256 $got, reviewed ${SHA[$f]}"
      echo "COMMITTED-OK  $got  $DIR/$f"
    done
    got=$(git show "HEAD:$DIR/$SELF_NAME" | sha256sum | cut -d' ' -f1)
    [ "$got" = "$2" ] || die "committed content of $SELF_NAME has sha256 $got, expected $2"
    echo "COMMITTED-OK  $got  $DIR/$SELF_NAME"
    echo "NEXT: bash $0 push $2" ;;
  push)
    enter
    self_check "${2:-}"
    h=$(git rev-parse HEAD) || exit 1
    [ "$(git rev-list --count "$START..HEAD")" = "1" ] && [ "$(git rev-parse HEAD~1)" = "$START" ] \
      && [ "$(git rev-list --parents -n 1 HEAD | wc -w)" = "2" ] || die "HEAD $h is not exactly one ordinary commit above the registered start commit $START"
    [ -z "$(comm -23 <(changed_in HEAD) <(manifest))" ] || die "the commit at HEAD changes a file outside the package"
    for f in "${FILES[@]}"; do
      [ "$(git show "HEAD:$DIR/$f" | sha256sum | cut -d' ' -f1)" = "${SHA[$f]}" ] || die "$f at HEAD does not have its reviewed sha256"
    done
    [ "$(git show "HEAD:$DIR/$SELF_NAME" | sha256sum | cut -d' ' -f1)" = "$2" ] || die "$SELF_NAME at HEAD does not have the expected sha256"
    [ -z "$(git status --porcelain --untracked-files=no)" ] || die "tracked files are modified or staged in the worktree; nothing is pushed"
    r=$(remote_head) || exit 1
    [ "$r" = "$START" ] || [ "$r" = "$h" ] || die "the remote branch $BRANCH is at $r: neither the start commit nor HEAD; nothing is pushed"
    git push origin "$BRANCH"
    git fetch -q origin "$BRANCH"
    [ "$h" = "$(git rev-parse "origin/$BRANCH")" ] || die "remote $BRANCH does not equal local HEAD after the push"
    r=$(remote_head) || exit 1
    [ "$r" = "$h" ] || die "the remote branch (git ls-remote) is at $r, not at local HEAD, after the push"
    echo "PUSHED-AND-VERIFIED repo $(git remote get-url origin) branch $BRANCH HEAD $h"
    echo "START-COMMIT $START"
    echo "F1-PACKAGE-COMMIT $h"
    for f in "${FILES[@]}"; do echo "ARCHIVED  ${SHA[$f]}  $DIR/$f"; done
    echo "ARCHIVED  $2  $DIR/$SELF_NAME" ;;
  *)
    die "usage: bash $0 check | commit <sha256> | push <sha256>" ;;
esac
