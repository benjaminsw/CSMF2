#!/usr/bin/env bash
# SEQREF-DLAT-GRENARCH v0.1 -- dlat_gren_archive.sh: Git registration of the D-LAT G renderer (renderer, synthetic test) on ch2-diag (DOCTREE 165 section 4)
# LIFETIME: KEEP
#
# DRAFT FOR REVIEW. 'commit' and 'push' MODIFY GIT: do not run them before the review of this
# script and the owner's authorisation of the registration, recorded in LudWiki. 'check' is read-only.
#
# Built from the registered dlat_g_archive.sh v0.3 (commit 6cf1701c, sha256 701fb76c...4ad9) by exact
# replacements (program mk_gren_archive.py): the header, the start commit, the file list, the commit
# message and the script name. No command or check changed.
#   check            -> read-only: branch ch2-diag; local HEAD and the remote branch both at the
#                       registered start commit (the G package commit); the two package files in
#                       the worktree at their reviewed sha256; nothing staged.
#   commit <sha256>  -> 'check', then ONE commit of exactly the package files and of this script.
#   push <sha256>    -> HEAD must be exactly one ordinary commit above the start commit; pushes
#                       ch2-diag and verifies the remote by fetch and by ls-remote.
# Both files are new: nothing registered is replaced. The pinned V05LT checkout is not touched.
# Nothing is ever forced, amended, rebased or deleted.
# Any failed check stops with an ERROR line and a non-zero exit. No fallback, no retry.
#
# Usage:  bash dlat_gren_archive.sh check
#         bash dlat_gren_archive.sh commit <reviewed sha256 of dlat_gren_archive.sh, all 64 characters>
#         bash dlat_gren_archive.sh push   <reviewed sha256 of dlat_gren_archive.sh, all 64 characters>
#
# Changelog:
#   v0.1 (2026-11-24) first version (draft for review), from dlat_g_archive.sh v0.3.
set -euo pipefail

WT="/home/benjamin/CSMFII/worktrees/ch2-diag"
BRANCH="ch2-diag"
DIR="seqref_mri/scripts/diag_lat"
START="6cf1701ca24c9cb18d94968b14c11ab3649f8159"      # head of ch2-diag, local and remote: the G package commit (LudWiki 256 LE103)
SELF_NAME="dlat_gren_archive.sh"
MSG="D-LAT G renderer v0.1 and its synthetic test v0.1

Report plan SEQREF-DLAT-G-REPORT-PLAN. Presentation only: the renderer reads the result files
named in g_report.json of the single G run, refuses any file whose sha256 differs, and computes
no D-LAT measure, effect, interval, class or tag (Figure 1 role means and S4 histogram counts are
formed at rendering and listed in the render report). Registration is not an authorisation to
render: rendering needs its own recorded authorisation.

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Df3EoZnqFQN2xsHZ7sBL1x"

# The package: every file with its reviewed sha256 (this script excepted; its sha256 is the argument).
FILES=(dlat_g_render.py dlat_g_rendertest.py)
declare -A SHA=(
  [dlat_g_render.py]=4802a4cf49cea0c374d8fc7ab79b2d0279b2abb1011bb9c66f7eae5f8c830fcf
  [dlat_g_rendertest.py]=96efc479a36f89876875d18e5d313028528882dddc534d3e275efd7b26aa0873
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
    echo "G-PACKAGE-COMMIT $h"
    for f in "${FILES[@]}"; do echo "ARCHIVED  ${SHA[$f]}  $DIR/$f"; done
    echo "ARCHIVED  $2  $DIR/$SELF_NAME" ;;
  *)
    die "usage: bash $0 check | commit <sha256> | push <sha256>" ;;
esac
