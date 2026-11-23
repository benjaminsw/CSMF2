#!/usr/bin/env bash
# SEQREF-DLATP3-ARCH v0.7.1 -- dlat_p3_archive.sh: Git archival of the revised D-LAT P2/P3 files on ch2-diag (authorisation V9)
# LIFETIME: KEEP
#
# DRAFT FOR REVIEW. DO NOT RUN 'update', 'scripts' or 'push' before the run card SEQREF-DLATP3-RUN
# v0.7.1 is reviewed and the owner has authorised V9 (LudWiki 243). 'check' changes nothing.
#
# State on the host before this registration: the six files are archived in commit 4ab895b9... (exactly the
# six-file manifest), diag_lat_p3test.py v0.3 in one later commit (eeb282f6...), the launcher
# scripts v0.5 in 78967965... (head). The controlled revision SEQREF-DLAT-P23REV v0.1 (LudWiki
# 242 LE52) changes THREE of the six files: diag_lat_stats.py v0.2 -> v0.3, diag_lat_selftest.py
# v0.3 -> v0.4, diag_lat_p3test.py v0.3 -> v0.4. This script archives those three files in ONE
# commit and then the two launcher scripts (this version) in a separate commit. Git only: runs no Python, reads
# no checkpoint or data, and never touches the pinned V05LT checkout (/home/benjamin/CSMFII,
# branch seqref-v05lt). Earlier commits and the scripts v0.5 stay in the history, unedited.
#
# Usage (in this order; each step prints what it did):
#   bash dlat_p3_archive.sh check    # worktree, branch, base commit, three unchanged files at HEAD,
#                                    # the three revised files in the folder; changes nothing
#   bash dlat_p3_archive.sh update   # check, then ONE commit changing exactly the three revised files
#   bash dlat_p3_archive.sh scripts <sha256 of this script>
#                                    # separate commit of the two reviewed launcher scripts
#   bash dlat_p3_archive.sh push <sha256 of this script>
#                                    # refuses unless the base commit, the revision commit and the
#                                    # scripts commit are all verified; then pushes and verifies
# Any failed check stops with an ERROR line and a non-zero exit. No fallback.
#
# The branch position is verified at every step (safeguard R1): the work starts from the
# registered commit of the scripts v0.5 (START) and adds exactly two commits on top of it, in a
# straight line: START -> revision commit -> scripts commit. HEAD must be START (before
# 'update'), START + the revision commit (before 'scripts') or START + both (before 'push');
# the remote branch must still be at START until the push (or already at the verified HEAD if
# the push was made before). Any other commit, locally or on the remote, stops the script.
#
# Changelog:
#   v0.7.1 (2026-11-22) correction required by the review of v0.7 (LudWiki 243 LE55): reviewed
#        sha256 of dlat_p3_run.sh v0.7.1 (provenance label corrected there); the version named
#        in the scripts commit message; two header comment lines that still described the work
#        as "v0.6". No executable line changed except the embedded hash and the commit message.
#        v0.7 preserved unchanged (not archived in Git; sha256 f1074a64...f6d5).
#   v0.7 (2026-11-22) safeguard R1 (LudWiki 243 LE54): the start commit of the branch (scripts
#        v0.5, registered as 78967965...cfb7) is resolved and required; HEAD must be START plus
#        exactly the commits this script makes, each with one parent; the remote branch is read
#        (git ls-remote, read-only) and must be at START before the push; no intervening commit
#        is accepted at any step. The two-commit scope checks and all hashes of v0.6 are kept;
#        reviewed sha256 of dlat_p3_run.sh v0.7. v0.6 preserved unchanged (not archived in Git;
#        sha256 f99db247...62dc).
#   v0.6 (2026-11-22) plan SEQREF-DLAT-P23REV-HOST v0.1 (LudWiki 243 LE53): 'update' commits
#        exactly the three revised files; the archive is verified as base commit (six files) +
#        the earlier one-file commit of the test file + one revision commit (exactly the three
#        files) + no change to core, guard and adapter; reviewed sha256 of the three revised
#        files and of dlat_p3_run.sh v0.6. v0.5 preserved (archived in commit 78967965...).
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
# START: the commit of the launcher scripts v0.5, head of ch2-diag before this registration.
# The record holds it in abbreviated form only (78967965...cfb7; 239 LE31). The script resolves
# the one commit with this prefix, requires the suffix, prints the full hash and compares with it.
START_PREFIX="78967965"
START_SUFFIX="cfb7"
REV=(diag_lat_stats.py diag_lat_selftest.py diag_lat_p3test.py)   # the three files that change
MSG="D-LAT P2/P3: controlled revision SEQREF-DLAT-P23REV v0.1 (242 LE52; candidate v0.6.2, not frozen)

diag_lat_stats.py v0.3: outcome rule (A14), seed tag (A16), fixed-epoch tag (A19), rules_version.
diag_lat_selftest.py v0.4 and diag_lat_p3test.py v0.4: tests of the revised rules (T15, R1b).
Pre-freeze engineering code; synthetic tests only; no D-LAT evidence.

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Df3EoZnqFQN2xsHZ7sBL1x"
# sha256 of the three files as archived BEFORE this revision (a state check, not a reviewed target)
declare -A OLD_SHA=(
  [diag_lat_stats.py]=1d7db11d38b11c3d8f67bcae086eb29daa5a5d903b5ce5e9446c43ce9b109831
  [diag_lat_selftest.py]=0aeaa9e081ccceaf9069585ddf39bd12ecaa211ea4c4253bff599b772f91b794
  [diag_lat_p3test.py]=777685bea9b279af4382d83de917b4024d6cbf07f7702f602741fc01895a338f
)

FILES=(diag_lat_core.py diag_lat_stats.py diag_lat_guard.py diag_lat_selftest.py diag_lat_adapter.py diag_lat_p3test.py)
declare -A SHA=(
  [diag_lat_core.py]=acfccb4099e204870c6c82ab8876ad9682aa87d465b0f845c6960a18a7697bc9
  [diag_lat_stats.py]=247d9c8bf1c28f9b063468c6a29fa97d45b9ed5ca33ed7339211153cfeef2bb3
  [diag_lat_guard.py]=4c78796aa3185c54d53e73ac251fc0a8c7e59fd45509dfea490cf9221d3b76f3
  [diag_lat_selftest.py]=7ae7ccdb5cfc9a7b27076cbe2989013bf85b83060f59cb4f70ca7d192edb3d72
  [diag_lat_adapter.py]=300c30587f8c55f8901a3c36bbcd84d594a9fe6624d67e359c57cec48a0760e2
  [diag_lat_p3test.py]=0e9236574e26ade509d62df166843bb83bc5494f95dfe17a422c46409971056d
)

RUN_SH="dlat_p3_run.sh"
RUN_SH_SHA="d8ba92ef24cfa94071f571763a55d8533b573bad1b519b0c959840e5220faa5c"                              # reviewed dlat_p3_run.sh v0.7.1
MSG_SH="D-LAT P2/P3: reviewed launcher scripts (archival v0.7.1, synthetic host validation v0.7.1; run card SEQREF-DLATP3-RUN v0.7.1)

Git archival helper and synthetic host run script. No D-LAT evidence.

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Df3EoZnqFQN2xsHZ7sBL1x"

die() { echo "ERROR: $*" >&2; exit 1; }
SELF=$(readlink -f "$0")

start_commit() {   # the one commit with the registered prefix and suffix; stops if none or more than one
  local s
  s=$(git rev-parse --verify --quiet "${START_PREFIX}^{commit}") || die "no unique commit with the registered prefix $START_PREFIX in this repository (missing or ambiguous)"
  case "$s" in "$START_PREFIX"*"$START_SUFFIX") ;; *) die "commit $s does not match the registered start commit ${START_PREFIX}...${START_SUFFIX}" ;; esac
  echo "$s"
}

remote_head() {   # read-only: the commit the remote branch points at; stops if it cannot be read
  local out
  out=$(git ls-remote --heads origin "$BRANCH") || die "the remote branch $BRANCH could not be read (git ls-remote failed); nothing is changed"
  out=$(printf '%s\n' "$out" | awk -v r="refs/heads/$BRANCH" '$2==r{print $1}')
  [ -n "$out" ] || die "the remote has no branch $BRANCH"
  echo "$out"
}

position() {   # how many commits HEAD is above START (0, 1 or 2) in a straight line; stops on anything else
  local s h n i
  # 'exit 1' after each nested call: bash clears errexit inside a command substitution, so a
  # failed inner call must be passed on explicitly or the function would continue on an empty value
  s=$(start_commit) || exit 1
  h=$(git rev-parse HEAD) || exit 1
  git merge-base --is-ancestor "$s" "$h" || die "HEAD $h does not descend from the registered start commit $s"
  n=$(git rev-list --count "$s..$h") || exit 1
  [ "$n" -le 2 ] || die "HEAD $h is $n commits above the registered start commit $s; at most the revision commit and the scripts commit are allowed: unexpected commits"
  i=0
  while [ "$i" -lt "$n" ]; do
    [ "$(git rev-list --parents -n 1 "HEAD~$i" | wc -w)" = "2" ] || die "commit $(git rev-parse "HEAD~$i") is a merge or has no parent: not a straight line from the start commit"
    i=$((i + 1))
  done
  [ "$(git rev-parse "HEAD~$n")" = "$s" ] || die "HEAD~$n is not the registered start commit $s"
  echo "$n"
}

require_position() {   # $1 wanted number of commits above START; also checks the remote branch
  local n r s h
  n=$(position) || exit 1
  s=$(start_commit) || exit 1
  h=$(git rev-parse HEAD) || exit 1
  [ "$n" = "$1" ] || die "HEAD $h is $n commit(s) above the registered start commit $s; this step needs exactly $1"
  r=$(remote_head) || exit 1
  if [ "$r" = "$s" ]; then
    echo "branch position: start commit $s; HEAD $h = start + $n commit(s); remote $BRANCH at the start commit"
  elif [ "$1" = "2" ] && [ "$r" = "$h" ]; then
    echo "branch position: start commit $s; HEAD $h = start + 2 commits; remote $BRANCH already at HEAD (pushed before)"
  else
    die "the remote branch $BRANCH is at $r, expected the registered start commit $s: unexpected remote state; nothing is changed"
  fi
}

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

is_rev() { local r; for r in "${REV[@]}"; do [ "$r" = "$1" ] && return 0; done; return 1; }

rev_manifest() { for f in "${REV[@]}"; do echo "$DIR/$f"; done | sort; }

archive_commit() {   # base (six files) + earlier one-file commit of the test file + ONE revision commit (exactly the three files) + nothing else
  local c f n want
  git cat-file -e "$BASE^{commit}" 2>/dev/null || die "base archive commit $BASE is not in this repository"
  [ "$(changed_in "$BASE")" = "$(manifest)" ] || die "base commit $BASE does not change exactly the six-file manifest"
  c=$(git log -1 --format=%H -- "$DIR/diag_lat_stats.py")
  [ "$c" != "$BASE" ] || die "the revised files are not archived yet (run: bash $0 update)"
  [ "$(changed_in "$c")" = "$(rev_manifest)" ] || die "commit $c does not change exactly the three revised files: $(changed_in "$c" | tr '\n' ' ')"
  for f in "${FILES[@]}"; do
    n=$(git rev-list --count "$BASE..HEAD" -- "$DIR/$f")
    if is_rev "$f"; then
      [ "$(git log -1 --format=%H -- "$DIR/$f")" = "$c" ] || die "$DIR/$f was last changed by another commit than the revision commit $c"
      want=1; [ "$f" = "diag_lat_p3test.py" ] && want=2      # the test file also has its earlier one-file commit (v0.3)
      [ "$n" = "$want" ] || die "$DIR/$f was changed by $n commit(s) after the base commit, expected $want"
    else
      [ "$n" = "0" ] || die "$DIR/$f was changed after the base commit ($n commit(s))"
    fi
  done
  f=$(git log --format=%H "$BASE..HEAD" -- "$DIR/diag_lat_p3test.py" | tail -1)
  [ "$(changed_in "$f")" = "$DIR/diag_lat_p3test.py" ] || die "the earlier commit $f of the test file does not change exactly $DIR/diag_lat_p3test.py"
  for f in "${FILES[@]}"; do
    [ "$(git show "HEAD:$DIR/$f" | sha256sum | cut -d' ' -f1)" = "${SHA[$f]}" ] || die "$f at HEAD does not have its reviewed sha256"
  done
  echo "$c"
}

check() {
  local f got at_head s0
  [ -d "$WT" ] || die "worktree not found: $WT"
  cd "$WT"
  [ "$(git rev-parse --show-toplevel)" = "$WT" ] || die "not the top of the ch2-diag worktree"
  [ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || die "branch is $(git rev-parse --abbrev-ref HEAD), expected $BRANCH"
  echo "worktree $WT  branch $BRANCH  HEAD $(git rev-parse HEAD)"
  s0=$(start_commit) || exit 1
  echo "registered start commit ${START_PREFIX}...${START_SUFFIX} resolves to $s0"
  require_position 0
  git cat-file -e "$BASE^{commit}" 2>/dev/null || die "base archive commit $BASE is not in this repository"
  [ "$(changed_in "$BASE")" = "$(manifest)" ] || die "base commit $BASE does not change exactly the six-file manifest"
  echo "base archive commit $BASE changes exactly the six-file manifest"
  for f in "${FILES[@]}"; do
    [ -f "$DIR/$f" ] || die "missing file: $DIR/$f"
    got=$(sha256sum "$DIR/$f" | cut -d' ' -f1)
    [ "$got" = "${SHA[$f]}" ] || die "sha256 mismatch for $f: got $got, reviewed ${SHA[$f]}"
    at_head=$(git show "HEAD:$DIR/$f" | sha256sum | cut -d' ' -f1)
    if ! is_rev "$f"; then
      [ "$at_head" = "${SHA[$f]}" ] || die "$f at HEAD differs from its reviewed sha256"
      echo "OK (unchanged, archived)  $got  $DIR/$f"
    elif [ "$at_head" = "${OLD_SHA[$f]}" ]; then
      echo "OK (revised version to archive; HEAD holds the earlier archived version)  $got  $DIR/$f"
    elif [ "$at_head" = "${SHA[$f]}" ]; then
      echo "OK (revised version already archived at HEAD)  $got  $DIR/$f"
    else
      die "$f at HEAD has sha256 $at_head: neither the earlier archived version nor the revised one"
    fi
  done
  staged=$(git diff --cached --name-only)
  [ -z "$staged" ] || die "something is already staged; unstage it first: $staged"
  echo "CHECK PASSED: core, guard and adapter unchanged and archived; the three revised files have their reviewed sha256; nothing staged"
}

case "${1:-}" in
  check)
    check ;;
  update)
    check
    for f in "${REV[@]}"; do git add -- "$DIR/$f"; done
    staged=$(git diff --cached --name-only | sort)
    if [ -z "$staged" ]; then die "nothing to commit: the three revised files are already identical at HEAD (run: bash $0 scripts <sha256>)"; fi
    if [ "$staged" != "$(rev_manifest)" ]; then git reset -q; die "staged set is not exactly the three revised files (nothing committed, staging undone): $(echo $staged)"; fi
    git commit -q -m "$MSG"
    echo "COMMITTED $(git rev-parse HEAD)"
    git show --stat --format='%H%n%an <%ae>%n%ad%n%s' HEAD
    [ "$(changed_in HEAD)" = "$(rev_manifest)" ] || die "the commit's changed-file list is not exactly the three revised files: $(changed_in HEAD | tr '\n' ' ')"
    echo "COMMIT-SCOPE-OK: changed-file list is exactly the three revised files"
    for f in "${REV[@]}"; do
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
    require_position 1
    ac=$(archive_commit)
    [ "$ac" = "$(git rev-parse HEAD)" ] || die "HEAD is not the revision commit $ac"
    echo "base archive commit $BASE; revision commit of the three files: $ac"
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
    require_position 2
    ac=$(archive_commit)
    sc=$(scripts_commit "$self_want")
    [ "$sc" = "$(git rev-parse HEAD)" ] || die "HEAD is not the scripts commit $sc"
    [ "$ac" = "$(git rev-parse HEAD~1)" ] || die "the commit below HEAD is not the revision commit $ac"
    [ -z "$(git status --porcelain --untracked-files=no)" ] || die "tracked files are modified or staged in the worktree; nothing is pushed"
    git push origin "$BRANCH"
    git fetch -q origin "$BRANCH"
    [ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$BRANCH")" ] || die "remote $BRANCH does not equal local HEAD after the push"
    rh=$(remote_head) || exit 1
    [ "$rh" = "$(git rev-parse HEAD)" ] || die "the remote branch (git ls-remote) is at $rh, not at local HEAD, after the push"
    echo "PUSHED-AND-VERIFIED repo $(git remote get-url origin) branch $BRANCH HEAD $(git rev-parse HEAD)"
    s0=$(start_commit) || exit 1
    echo "START-COMMIT $s0 (launcher scripts v0.5; registered as ${START_PREFIX}...${START_SUFFIX})"
    echo "ARCHIVE-COMMIT $BASE (changes exactly the six-file manifest)"
    echo "REVISION-COMMIT $ac (changes exactly the three revised files)"
    for f in "${FILES[@]}"; do echo "ARCHIVED  ${SHA[$f]}  $DIR/$f"; done
    echo "SCRIPTS-COMMIT $sc (changes exactly the two launcher scripts)"
    echo "ARCHIVED-SCRIPT  $self_want  $DIR/dlat_p3_archive.sh"
    echo "ARCHIVED-SCRIPT  $RUN_SH_SHA  $DIR/$RUN_SH" ;;
  *)
    die "usage: bash $0 check|update|scripts <sha256>|push <sha256>" ;;
esac
