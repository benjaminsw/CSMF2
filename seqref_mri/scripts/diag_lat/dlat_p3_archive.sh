#!/usr/bin/env bash
# SEQREF-DLATP3-ARCH v0.3 -- dlat_p3_archive.sh: Git archival of the reviewed D-LAT P1/P2/P3 files on ch2-diag
# LIFETIME: KEEP
#
# Archives exactly six reviewed files (238 LE20, LE22, LE26) under seqref_mri/scripts/diag_lat/ on
# branch ch2-diag. Git only: runs no Python, reads no checkpoint or data, and never touches the
# pinned V05LT checkout (/home/benjamin/CSMFII, branch seqref-v05lt).
#
# Usage (in this order; each step prints what it did):
#   bash dlat_p3_archive.sh check    # verifies worktree, branch and the six sha256; changes nothing
#   bash dlat_p3_archive.sh commit   # check, then stages EXACTLY the six files and commits; the
#                                    # commit's changed-file list must equal the six-file manifest
#   bash dlat_p3_archive.sh scripts <sha256 of this script>
#                                    # separate commit of the two reviewed launcher scripts
#                                    # (dlat_p3_archive.sh, dlat_p3_run.sh), added with git add -f
#   bash dlat_p3_archive.sh push <sha256 of this script>
#                                    # refuses unless BOTH commits are verified (six files; two
#                                    # launcher scripts with their reviewed sha256); then pushes
#                                    # ch2-diag and verifies the remote
# Any failed check stops with an ERROR line and a non-zero exit. No fallback.
#
# Changelog:
#   v0.3 (2026-11-21) review of v0.2: 'push' now fails unless the separate scripts commit exists,
#        changes exactly the two launcher scripts, and both committed files have their reviewed
#        sha256 (the hash of this script is given on the command line). v0.2 preserved.
#   v0.2 (2026-11-21) review of v0.1 (C7): the staged set and the commit's changed-file list
#        must equal the six-file manifest exactly (a subset is refused and unstaged); new mode
#        'scripts' for a separate commit of the two launcher scripts with their final hashes;
#        'push' re-verifies the archive commit's changed-file list. v0.1 preserved.
#   v0.1 (2026-11-21) first version.
set -euo pipefail

WT="/home/benjamin/CSMFII/worktrees/ch2-diag"
BRANCH="ch2-diag"
DIR="seqref_mri/scripts/diag_lat"
MSG="D-LAT pre-freeze engineering: P1/P2/P3 reviewed files (238 LE20, LE22, LE26)

Synthetic-test code only; not frozen; no D-LAT evidence.
core v0.1, stats v0.2, guard v0.1, selftest v0.3, adapter v0.1, p3test v0.2.

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Df3EoZnqFQN2xsHZ7sBL1x"

FILES=(diag_lat_core.py diag_lat_stats.py diag_lat_guard.py diag_lat_selftest.py diag_lat_adapter.py diag_lat_p3test.py)
declare -A SHA=(
  [diag_lat_core.py]=acfccb4099e204870c6c82ab8876ad9682aa87d465b0f845c6960a18a7697bc9
  [diag_lat_stats.py]=1d7db11d38b11c3d8f67bcae086eb29daa5a5d903b5ce5e9446c43ce9b109831
  [diag_lat_guard.py]=4c78796aa3185c54d53e73ac251fc0a8c7e59fd45509dfea490cf9221d3b76f3
  [diag_lat_selftest.py]=0aeaa9e081ccceaf9069585ddf39bd12ecaa211ea4c4253bff599b772f91b794
  [diag_lat_adapter.py]=300c30587f8c55f8901a3c36bbcd84d594a9fe6624d67e359c57cec48a0760e2
  [diag_lat_p3test.py]=3e20063f18f94058e62053bac9753050b907dbbe2675cd9e69132833808de238
)

RUN_SH="dlat_p3_run.sh"
RUN_SH_SHA="28aabdcd486e808c9980d2fcc6e69f9847ab00c5292d42e8378dccd9f3c363cf"                              # reviewed dlat_p3_run.sh v0.3
MSG_SH="D-LAT P3: reviewed launcher scripts (archival v0.3, synthetic host run v0.3; run card SEQREF-DLATP3-RUN v0.3)

Git archival helper and synthetic host run script. No D-LAT evidence.

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Df3EoZnqFQN2xsHZ7sBL1x"

die() { echo "ERROR: $*" >&2; exit 1; }
SELF=$(readlink -f "$0")

scripts_manifest() { printf '%s\n' "$DIR/dlat_p3_archive.sh" "$DIR/$RUN_SH" | sort; }

scripts_commit() {   # $1 reviewed sha256 of dlat_p3_archive.sh: one commit changing exactly the two scripts
  local c got
  c=$(git log -1 --format=%H -- "$DIR/$RUN_SH")
  [ -n "$c" ] || die "the launcher scripts are not archived yet (run: bash $0 scripts <sha256>)"
  [ "$(changed_in "$c")" = "$(scripts_manifest)" ] || die "commit $c does not change exactly the two launcher scripts: $(changed_in "$c" | tr '\n' ' ')"
  got=$(git show "HEAD:$DIR/$RUN_SH" | sha256sum | cut -d' ' -f1)
  [ "$got" = "$RUN_SH_SHA" ] || die "archived $RUN_SH has sha256 $got, reviewed $RUN_SH_SHA"
  got=$(git show "HEAD:$DIR/dlat_p3_archive.sh" | sha256sum | cut -d' ' -f1)
  [ "$got" = "$1" ] || die "archived dlat_p3_archive.sh has sha256 $got, expected $1"
  [ "$(sha256sum "$SELF" | cut -d' ' -f1)" = "$1" ] || die "the script being run does not have the expected sha256 $1"
  echo "$c"
}

manifest() { for f in "${FILES[@]}"; do echo "$DIR/$f"; done | sort; }

changed_in() { git diff-tree --no-commit-id --name-only -r "$1" | sort; }

archive_commit() {   # the commit that last changed the six files must be one commit, changing exactly them
  local c
  c=$(git log -1 --format=%H -- "$DIR/diag_lat_p3test.py")
  [ -n "$c" ] || die "no commit holds $DIR/diag_lat_p3test.py"
  [ "$(changed_in "$c")" = "$(manifest)" ] || die "commit $c does not change exactly the six-file manifest: $(changed_in "$c" | tr '\n' ' ')"
  for f in "${FILES[@]}"; do
    [ "$(git show "$c:$DIR/$f" | sha256sum | cut -d' ' -f1)" = "${SHA[$f]}" ] || die "$f in commit $c does not have its reviewed sha256"
    [ "$(git show "HEAD:$DIR/$f" | sha256sum | cut -d' ' -f1)" = "${SHA[$f]}" ] || die "$f at HEAD does not have its reviewed sha256"
  done
  echo "$c"
}

check() {
  [ -d "$WT" ] || die "worktree not found: $WT"
  cd "$WT"
  [ "$(git rev-parse --show-toplevel)" = "$WT" ] || die "not the top of the ch2-diag worktree"
  [ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || die "branch is $(git rev-parse --abbrev-ref HEAD), expected $BRANCH"
  echo "worktree $WT  branch $BRANCH  HEAD $(git rev-parse HEAD)"
  for f in "${FILES[@]}"; do
    [ -f "$DIR/$f" ] || die "missing file: $DIR/$f"
    got=$(sha256sum "$DIR/$f" | cut -d' ' -f1)
    [ "$got" = "${SHA[$f]}" ] || die "sha256 mismatch for $f: got $got, reviewed ${SHA[$f]}"
    echo "OK  $got  $DIR/$f"
  done
  staged=$(git diff --cached --name-only)
  [ -z "$staged" ] || die "something is already staged; unstage it first: $staged"
  echo "CHECK PASSED: six files match their reviewed sha256; nothing staged"
}

case "${1:-}" in
  check)
    check ;;
  commit)
    check
    for f in "${FILES[@]}"; do git add -- "$DIR/$f"; done
    staged=$(git diff --cached --name-only | sort)
    want=$(manifest)
    if [ -z "$staged" ]; then die "nothing to commit: the six files are already identical at HEAD (run: bash $0 push)"; fi
    if [ "$staged" != "$want" ]; then
      git reset -q
      die "staged set differs from the six-file manifest (nothing committed, staging undone). Staged: $(echo $staged). A file already in Git, or an extra path, must be documented and reviewed first"
    fi
    git commit -q -m "$MSG"
    echo "COMMITTED $(git rev-parse HEAD)"
    git show --stat --format='%H%n%an <%ae>%n%ad%n%s' HEAD
    [ "$(changed_in HEAD)" = "$want" ] || die "the commit's changed-file list differs from the six-file manifest: $(changed_in HEAD | tr '\n' ' ')"
    echo "COMMIT-SCOPE-OK: changed-file list equals the six-file manifest exactly"
    for f in "${FILES[@]}"; do
      got=$(git show "HEAD:$DIR/$f" | sha256sum | cut -d' ' -f1)
      [ "$got" = "${SHA[$f]}" ] || die "committed content of $f has sha256 $got, reviewed ${SHA[$f]}"
      echo "COMMITTED-OK  $got  $DIR/$f"
    done
    echo "NEXT: bash $0 scripts <reviewed sha256 of dlat_p3_archive.sh>" ;;
  scripts)
    self_want="${2:-}"
    [ -n "$self_want" ] || die "usage: bash $0 scripts <reviewed sha256 of dlat_p3_archive.sh>"
    [ -d "$WT" ] || die "worktree not found: $WT"
    cd "$WT"
    [ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || die "branch is not $BRANCH"
    ac=$(archive_commit)
    echo "archive commit of the six files: $ac"
    [ -z "$(git diff --cached --name-only)" ] || die "something is already staged; unstage it first"
    [ -f "$DIR/dlat_p3_archive.sh" ] && [ -f "$DIR/$RUN_SH" ] || die "both launcher scripts must be in $DIR"
    got=$(sha256sum "$DIR/dlat_p3_archive.sh" | cut -d' ' -f1)
    [ "$got" = "$self_want" ] || die "dlat_p3_archive.sh in the worktree has sha256 $got, expected $self_want"
    [ "$(sha256sum "$SELF" | cut -d' ' -f1)" = "$self_want" ] || die "the script being run does not have the expected sha256 $self_want"
    got=$(sha256sum "$DIR/$RUN_SH" | cut -d' ' -f1)
    [ "$got" = "$RUN_SH_SHA" ] || die "$RUN_SH has sha256 $got, reviewed $RUN_SH_SHA"
    git add -f -- "$DIR/dlat_p3_archive.sh" "$DIR/$RUN_SH"
    want=$(scripts_manifest)
    staged=$(git diff --cached --name-only | sort)
    if [ "$staged" != "$want" ]; then git reset -q; die "staged set differs from the two launcher scripts (staging undone): $(echo $staged)"; fi
    git commit -q -m "$MSG_SH"
    [ "$(changed_in HEAD)" = "$want" ] || die "the scripts commit's changed-file list differs from the two launcher scripts"
    echo "COMMITTED-SCRIPTS $(git rev-parse HEAD)"
    echo "SCRIPT  $self_want  $DIR/dlat_p3_archive.sh"
    echo "SCRIPT  $RUN_SH_SHA  $DIR/$RUN_SH"
    echo "NEXT: bash $0 push $self_want" ;;
  push)
    [ -d "$WT" ] || die "worktree not found: $WT"
    cd "$WT"
    [ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || die "branch is not $BRANCH"
    self_want="${2:-}"
    [ -n "$self_want" ] || die "usage: bash $0 push <reviewed sha256 of dlat_p3_archive.sh>"
    ac=$(archive_commit)
    sc=$(scripts_commit "$self_want")
    [ -z "$(git status --porcelain --untracked-files=no)" ] || die "tracked files are modified or staged in the worktree; nothing is pushed"
    git push origin "$BRANCH"
    git fetch -q origin "$BRANCH"
    [ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$BRANCH")" ] || die "remote $BRANCH does not equal local HEAD after the push"
    echo "PUSHED-AND-VERIFIED repo $(git remote get-url origin) branch $BRANCH HEAD $(git rev-parse HEAD)"
    echo "ARCHIVE-COMMIT $ac (changes exactly the six-file manifest)"
    for f in "${FILES[@]}"; do echo "ARCHIVED  ${SHA[$f]}  $DIR/$f"; done
    echo "SCRIPTS-COMMIT $sc (changes exactly the two launcher scripts)"
    echo "ARCHIVED-SCRIPT  $self_want  $DIR/dlat_p3_archive.sh"
    echo "ARCHIVED-SCRIPT  $RUN_SH_SHA  $DIR/$RUN_SH" ;;
  *)
    die "usage: bash $0 check|commit|scripts <sha256>|push <sha256>" ;;
esac
