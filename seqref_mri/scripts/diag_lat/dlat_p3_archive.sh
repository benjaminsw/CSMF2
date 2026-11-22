#!/usr/bin/env bash
# SEQREF-DLATP3-ARCH v0.5 -- dlat_p3_archive.sh: Git archival of the reviewed D-LAT P1/P2/P3 files on ch2-diag
# LIFETIME: KEEP
#
# State on the host before v0.5: the six files are archived in commit 4ab895b9... (exactly the
# six-file manifest), the launcher scripts v0.4 in 2f1ec6a0.... After the failed host run
# (LudWiki 239 LE29) ONE of the six files changes: diag_lat_p3test.py v0.2 -> v0.3 (fix plan
# SEQREF-DLAT-P3FIX v0.1). v0.5 archives that one file in its own commit and then the two
# launcher scripts v0.5. Git only: runs no Python, reads no checkpoint or data, and never touches
# the pinned V05LT checkout (/home/benjamin/CSMFII, branch seqref-v05lt).
#
# Usage (in this order; each step prints what it did):
#   bash dlat_p3_archive.sh check    # worktree, branch, base commit, five unchanged files at HEAD,
#                                    # the new test file in the folder; changes nothing
#   bash dlat_p3_archive.sh update   # check, then ONE commit changing exactly diag_lat_p3test.py
#   bash dlat_p3_archive.sh scripts <sha256 of this script>
#                                    # separate commit of the two reviewed launcher scripts
#   bash dlat_p3_archive.sh push <sha256 of this script>
#                                    # refuses unless the base commit, the update commit and the
#                                    # scripts commit are all verified; then pushes and verifies
# Any failed check stops with an ERROR line and a non-zero exit. No fallback.
#
# Changelog:
#   v0.5 (2026-11-21) the six-file 'commit' mode is replaced by 'update' (one file); the archive
#        is verified as base commit (six files) + update commit (exactly the test file) + no
#        other change to the other five files; reviewed sha256 of the test file v0.3 and of
#        dlat_p3_run.sh v0.5. v0.4 preserved (archived in commit 2f1ec6a0...).
#   v0.4 (2026-11-21) reviewed sha256 of dlat_p3_run.sh v0.4; no logic changed.
#   v0.3 (2026-11-21) 'push' requires the verified scripts commit.
#   v0.2 (2026-11-21) exact six-file scope; separate scripts commit.
#   v0.1 (2026-11-21) first version.
set -euo pipefail

WT="/home/benjamin/CSMFII/worktrees/ch2-diag"
BRANCH="ch2-diag"
DIR="seqref_mri/scripts/diag_lat"
BASE="4ab895b988db17ea2d38092e7f86078da49e7122"       # archive commit of the six files (239 LE27)
UPD="diag_lat_p3test.py"                              # the one file that changes
MSG="D-LAT P3: diag_lat_p3test.py v0.3 (fix plan SEQREF-DLAT-P3FIX v0.1; 239 LE29)

Report step corrected after the failed synthetic host run; new regression fixture R7.
No existing fixture, tolerance or parameter changed. Synthetic-test code only; no D-LAT evidence.

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Df3EoZnqFQN2xsHZ7sBL1x"

FILES=(diag_lat_core.py diag_lat_stats.py diag_lat_guard.py diag_lat_selftest.py diag_lat_adapter.py diag_lat_p3test.py)
declare -A SHA=(
  [diag_lat_core.py]=acfccb4099e204870c6c82ab8876ad9682aa87d465b0f845c6960a18a7697bc9
  [diag_lat_stats.py]=1d7db11d38b11c3d8f67bcae086eb29daa5a5d903b5ce5e9446c43ce9b109831
  [diag_lat_guard.py]=4c78796aa3185c54d53e73ac251fc0a8c7e59fd45509dfea490cf9221d3b76f3
  [diag_lat_selftest.py]=0aeaa9e081ccceaf9069585ddf39bd12ecaa211ea4c4253bff599b772f91b794
  [diag_lat_adapter.py]=300c30587f8c55f8901a3c36bbcd84d594a9fe6624d67e359c57cec48a0760e2
  [diag_lat_p3test.py]=777685bea9b279af4382d83de917b4024d6cbf07f7702f602741fc01895a338f
)

RUN_SH="dlat_p3_run.sh"
RUN_SH_SHA="2a959822def3029f522ba8ca9557af791c1235dc6e3c3514478fa5db41a17fca"                              # reviewed dlat_p3_run.sh v0.5
MSG_SH="D-LAT P3: reviewed launcher scripts (archival v0.5, synthetic host run v0.5; run card SEQREF-DLATP3-RUN v0.5)

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

archive_commit() {   # base commit (six files) + one update commit (exactly the test file) + nothing else
  local c other f
  git cat-file -e "$BASE^{commit}" 2>/dev/null || die "base archive commit $BASE is not in this repository"
  [ "$(changed_in "$BASE")" = "$(manifest)" ] || die "base commit $BASE does not change exactly the six-file manifest"
  c=$(git log -1 --format=%H -- "$DIR/$UPD")
  [ "$c" != "$BASE" ] || die "$UPD v0.3 is not archived yet (run: bash $0 update)"
  [ "$(changed_in "$c")" = "$DIR/$UPD" ] || die "commit $c does not change exactly $DIR/$UPD: $(changed_in "$c" | tr '\n' ' ')"
  [ "$(git rev-list --count "$BASE..HEAD" -- "$DIR/$UPD")" = "1" ] || die "$DIR/$UPD was changed by more than one commit after the base commit"
  for f in "${FILES[@]}"; do
    [ "$f" = "$UPD" ] && continue
    other=$(git rev-list --count "$BASE..HEAD" -- "$DIR/$f")
    [ "$other" = "0" ] || die "$DIR/$f was changed after the base commit ($other commit(s))"
  done
  for f in "${FILES[@]}"; do
    [ "$(git show "HEAD:$DIR/$f" | sha256sum | cut -d' ' -f1)" = "${SHA[$f]}" ] || die "$f at HEAD does not have its reviewed sha256"
  done
  echo "$c"
}

check() {
  local f got
  [ -d "$WT" ] || die "worktree not found: $WT"
  cd "$WT"
  [ "$(git rev-parse --show-toplevel)" = "$WT" ] || die "not the top of the ch2-diag worktree"
  [ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || die "branch is $(git rev-parse --abbrev-ref HEAD), expected $BRANCH"
  echo "worktree $WT  branch $BRANCH  HEAD $(git rev-parse HEAD)"
  git cat-file -e "$BASE^{commit}" 2>/dev/null || die "base archive commit $BASE is not in this repository"
  [ "$(changed_in "$BASE")" = "$(manifest)" ] || die "base commit $BASE does not change exactly the six-file manifest"
  echo "base archive commit $BASE changes exactly the six-file manifest"
  for f in "${FILES[@]}"; do
    [ -f "$DIR/$f" ] || die "missing file: $DIR/$f"
    got=$(sha256sum "$DIR/$f" | cut -d' ' -f1)
    [ "$got" = "${SHA[$f]}" ] || die "sha256 mismatch for $f: got $got, reviewed ${SHA[$f]}"
    if [ "$f" != "$UPD" ]; then
      [ "$(git show "HEAD:$DIR/$f" | sha256sum | cut -d' ' -f1)" = "${SHA[$f]}" ] || die "$f at HEAD differs from its reviewed sha256"
      echo "OK (unchanged, archived)  $got  $DIR/$f"
    else
      echo "OK (new version to archive)  $got  $DIR/$f"
    fi
  done
  staged=$(git diff --cached --name-only)
  [ -z "$staged" ] || die "something is already staged; unstage it first: $staged"
  echo "CHECK PASSED: five files unchanged and archived; $UPD has its reviewed v0.3 sha256; nothing staged"
}

case "${1:-}" in
  check)
    check ;;
  update)
    check
    git add -- "$DIR/$UPD"
    staged=$(git diff --cached --name-only | sort)
    if [ -z "$staged" ]; then die "nothing to commit: $UPD is already identical at HEAD (run: bash $0 scripts <sha256>)"; fi
    if [ "$staged" != "$DIR/$UPD" ]; then git reset -q; die "staged set is not exactly $DIR/$UPD (nothing committed, staging undone): $(echo $staged)"; fi
    git commit -q -m "$MSG"
    echo "COMMITTED $(git rev-parse HEAD)"
    git show --stat --format='%H%n%an <%ae>%n%ad%n%s' HEAD
    [ "$(changed_in HEAD)" = "$DIR/$UPD" ] || die "the commit's changed-file list is not exactly $DIR/$UPD: $(changed_in HEAD | tr '\n' ' ')"
    echo "COMMIT-SCOPE-OK: changed-file list is exactly $DIR/$UPD"
    got=$(git show "HEAD:$DIR/$UPD" | sha256sum | cut -d' ' -f1)
    [ "$got" = "${SHA[$UPD]}" ] || die "committed content of $UPD has sha256 $got, reviewed ${SHA[$UPD]}"
    echo "COMMITTED-OK  $got  $DIR/$UPD"
    echo "NEXT: bash $0 scripts <reviewed sha256 of dlat_p3_archive.sh>" ;;
  scripts)
    self_want="${2:-}"
    [ -n "$self_want" ] || die "usage: bash $0 scripts <reviewed sha256 of dlat_p3_archive.sh>"
    [ -d "$WT" ] || die "worktree not found: $WT"
    cd "$WT"
    [ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || die "branch is not $BRANCH"
    ac=$(archive_commit)
    echo "base archive commit $BASE; update commit of $UPD: $ac"
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
    echo "ARCHIVE-COMMIT $BASE (changes exactly the six-file manifest)"
    echo "UPDATE-COMMIT $ac (changes exactly $DIR/$UPD)"
    for f in "${FILES[@]}"; do echo "ARCHIVED  ${SHA[$f]}  $DIR/$f"; done
    echo "SCRIPTS-COMMIT $sc (changes exactly the two launcher scripts)"
    echo "ARCHIVED-SCRIPT  $self_want  $DIR/dlat_p3_archive.sh"
    echo "ARCHIVED-SCRIPT  $RUN_SH_SHA  $DIR/$RUN_SH" ;;
  *)
    die "usage: bash $0 check|update|scripts <sha256>|push <sha256>" ;;
esac
