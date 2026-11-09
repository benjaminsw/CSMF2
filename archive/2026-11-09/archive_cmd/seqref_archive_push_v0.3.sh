#!/usr/bin/env bash
# SEQREF-MRI-ARCHPUSH v0.3 -- archive G-D4 / G-D1 host scripts and the DIAG-JSH reviewed sandbox .py/.sh files to benjaminsw/CSMF2 (separate checkout, orphan branch)
# LIFETIME: KEEP
# =============================================================================
# Code-preservation rule (DOCTREE 165 §4). Run by the owner on the host.
#   * Input: SRC_DIR holding the sandbox .py/.sh files (versioned names, as
#     delivered); each is checked against the sha256 embedded below. No zip.
#   * Host files are copied read-only from the evidence folders and checked
#     against their recorded sha256.
#   * Locations (v0.3): SRC_DIR must be $EV/archive_src; the separate fresh
#     clone is $EV/archive_CSMF2. Both inside the project, outside
#     seqref_mri/ (the pinned source-identity scan covers every *.py there).
#   * Separate fresh clone ARC (never the pinned V05LT tree); new orphan
#     branch BR; refuses if BR exists on the remote; plain push, no --force;
#     push verified (remote head == local HEAD).
#   * This script copies itself into the commit (archive_cmd/). README and
#     SHA256SUMS are generated here.
#   * Logs (.log) are NOT archived (not delivered to the host); their sha256
#     stay recorded in 213/217.
# Prints: commit SHA, remote verification, per-file sha256 and git blob id.
# Exit 0 OK; 2 any failure (nothing pushed unless every check passed).
# NOTE: benjaminsw/CSMF2 is PUBLIC; archived code and host paths become public.
# Usage: bash seqref_archive_push_v0.3.sh /home/benjamin/CSMFII/v06lt_evidence/archive_src
# Changelog (v0.3, 2026-11-09): source folder and clone moved to
#   CSMFII/v06lt_evidence/ (owner decision); both checked to be there and
#   reached without symlinks; v0.2 archived as superseded.
# Changelog (v0.2, 2026-11-09): takes a folder of .py/.sh files with embedded
#   expected hashes instead of a zip bundle (no zip on the host); logs not
#   archived; README generated; v0.1 archived as superseded.
# Changelog (v0.1, 2026-11-09): first archive command (zip bundle; superseded).
# =============================================================================
set -euo pipefail
die() { echo "STOP: $*" >&2; exit 2; }
SELF=$(realpath -e -- "$0") || die "cannot resolve the script path"
SRC=$(realpath -e -- "${1:-/nonexistent}" 2>/dev/null) || die "usage: $0 SRC_DIR"
[[ -d "$SRC" ]] || die "usage: $0 SRC_DIR"
REPO=https://github.com/benjaminsw/CSMF2.git
BR=seqref-archive
EV=/home/benjamin/CSMFII/v06lt_evidence
ARC=$EV/archive_CSMF2
D=archive/2026-11-09
[[ -d "$EV" && "$(realpath -e -- "$EV")" == "$EV" ]] || die "$EV missing or reached through a symlink"
[[ "$SRC" == "$EV/archive_src" ]] || die "SRC_DIR must be $EV/archive_src (got $SRC)"
# name|archive subfolder|sha256  (sandbox files)
SANDBOX=$(cat <<'TAB'
  diag_jsh_adapter_v0.3.1.py|DIAG-JSH/code|c566858189be4a3dee822ce4eef46b156986881a92cb6d4360b386ca785707c1
  diag_jsh_core_v0.2.1.py|DIAG-JSH/code|85410c77472cf876021a49bd72254ac98461227887178e24cdb6d0ce5295c0fb
  diag_jsh_f1_cmd_v0.4.sh|DIAG-JSH/code|0d67a2df4cbec089eccace3bd15885e0d1eb5c12e477d51cd88f519dfe9b1d93
  diag_jsh_f1_prepare_v0.3.py|DIAG-JSH/superseded|b8ae5a23aa55c8abf9be3bc48fd03a9a72c250154f3935893afc0b7de713ed53
  diag_jsh_f1_prepare_v0.4.py|DIAG-JSH/code|840395a61ecd644d5fbaa814060f1181a601d72e98c4f0ad6e49a2cf019f8c87
  diag_jsh_i3_v0.1.py|DIAG-JSH/superseded|e908b2a035c7c98682b6e2d02ecdf5a2e72e2014c1a2c52296c7b75e482df57e
  diag_jsh_i3_v0.2.py|DIAG-JSH/code|9c7b40313aa76b0d483f5c6a1cf44697087f49c9648db5ff71871747d3a172d4
  regverify_v0.1.py|tools|39b044dcaaf080461d955fe1a152516e454342ee94cc42025c55b8b7d9b1a252
  seqref_archive_push_v0.2.sh|archive_cmd/superseded|53b3d564ceca9fc83bbbcbd8ba14372fe0fb63af76fc9de8e67efdc508949082
  seqref_archive_push_v0.1.sh|archive_cmd/superseded|f211be209f9448a301cce21fe206de1c8a58b4697a3305b83ef3ea35e760df2a
  test_diag_jsh_adapter_v0.3.1.py|DIAG-JSH/code|07a16a3c3f641793bd034b7ae9953025598ff761881fa441d36379c7c2cd40e4
  test_diag_jsh_bind_v0.1.py|DIAG-JSH/code|135af0c224e15964809c97c3192569e4adbebb89e08229cc3a032a708c23b6ba
  test_diag_jsh_core_v0.2.1.py|DIAG-JSH/code|5a84b856f662ccd3114c15228e9a943a07d5a7c314217cb51201ec119146d51e
  test_diag_jsh_f1_integration_v0.3.py|DIAG-JSH/code|5f7fddcea528a8755ac1380148c921da7905970bc7273db28afeb893a4f7039b
  test_diag_jsh_i3_v0.2.py|DIAG-JSH/code|8b9f49643414c71f507e925b362235be438c80cefd5c6b1930ed9eaaeaee7fcd
  test_v06lt_g4_v0.1.py|G-D4|bdf3c2938b867735c3679d97e47f5a0e4e6971496ac2a6b9ce9754e76958c8d4
TAB
)
# host path|archive subfolder|sha256
HOSTF=$(cat <<TAB
  $EV/V05LT_analysis/v06lt_v05lt_analysis_g4_v0.2.sh|G-D4|fe2785d9bba0947e4582b4a49d75b315a89a6f8583dd0abab8633c44ba0a7233
  $EV/V05LT_analysis/v06lt_v05lt_analysis_g4_driver.py|G-D4|8885726427e740c8dba5e6cbfa7bd656ee3aa7117826441bb2c36bd8619a51c5
  $EV/V05LT_completion/v06lt_v05lt_completion_check_v0.2.sh|G-D1|1b5acf700f1d535a3247e52686fc451e7fd0fa95b2daa967b73454c22083c617
TAB
)
chk() { [[ -f "$1" && ! -L "$1" && "$(sha256sum "$1" | cut -d' ' -f1)" == "$2" ]]; }
while IFS='|' read -r n sub h; do n=${n// /}; [[ -n "$n" ]] || continue
  chk "$SRC/$n" "$h" || die "missing or hash != expected: $SRC/$n"; done <<< "$SANDBOX"
while IFS='|' read -r f sub h; do f=${f// /}; [[ -n "$f" ]] || continue
  chk "$f" "$h" || die "missing or hash != recorded: $f"; done <<< "$HOSTF"
[[ ! -e "$ARC" && ! -L "$ARC" ]] || die "$ARC exists (use a fresh checkout)"
[[ -z "$(git ls-remote --heads "$REPO" "$BR")" ]] || die "remote branch $BR already exists"
mkdir -p "$(dirname "$ARC")"
git clone --quiet --no-checkout "$REPO" "$ARC"
cd "$ARC"
git switch --quiet --orphan "$BR"
while IFS='|' read -r n sub h; do n=${n// /}; [[ -n "$n" ]] || continue
  mkdir -p "$D/$sub"; cp -p -- "$SRC/$n" "$D/$sub/$n"; done <<< "$SANDBOX"
while IFS='|' read -r f sub h; do f=${f// /}; [[ -n "$f" ]] || continue
  mkdir -p "$D/$sub"; cp -p -- "$f" "$D/$sub/"; done <<< "$HOSTF"
mkdir -p "$D/archive_cmd"; cp -p -- "$SELF" "$D/archive_cmd/"
cat > "$D/README.md" <<'EOF2'
<!-- SEQREF-MRI-ARCHIVE-20261109 v0.2 -- archive snapshot: G-D4 / G-D1 executed scripts and DIAG-JSH reviewed sandbox files -->
<!-- LIFETIME: KEEP -->

# SEQREF-MRI archive snapshot, 2026-11-09

Preserved under the code-preservation rule (LudWiki DOCTREE 165 §4). Exact bytes; verify with `sha256sum -c SHA256SUMS`.

- `G-D4/` — host-executed V05LT registered validation analysis (V05LTEXEC 196): wrapper and driver from the host evidence folder; sandbox test. Retained execution evidence, not for automatic rerun.
- `G-D1/` — host-executed V05LT completion check v0.2 (V06LTEXEC 213 E18).
- `DIAG-JSH/code/` — files as reviewed on 2026-11-09 (DIAG-JSH-EXEC 217 §3), version in the file name: core v0.2.1, adapter v0.3.1, F1 prepare v0.4, F1 command v0.4, I3 v0.2 and tests. Not approved for host execution.
- `DIAG-JSH/superseded/` — F1 prepare v0.3, I3 v0.1.
- `tools/` — LudWiki readback-verification helper. `archive_cmd/` — the command that made this commit (and its superseded v0.1, v0.2).
- Test logs are not included; their sha256 are recorded in 213/217.

No credentials, data, checkpoints or V05LT results are included.
EOF2
( cd "$D" && sha256sum $(find . -type f ! -name SHA256SUMS | sort) > SHA256SUMS )
while IFS='|' read -r n sub h; do n=${n// /}; [[ -n "$n" ]] || continue
  grep -q "^$h  ./$sub/$n\$" "$D/SHA256SUMS" || die "archived copy differs: $n"; done <<< "$SANDBOX"
while IFS='|' read -r f sub h; do f=${f// /}; [[ -n "$f" ]] || continue
  grep -q "^$h  ./$sub/$(basename "$f")\$" "$D/SHA256SUMS" || die "archived copy differs: $f"; done <<< "$HOSTF"
git add -- "$D"
git commit --quiet -m "SEQREF-MRI archive 2026-11-09: G-D4/G-D1 executed scripts, DIAG-JSH reviewed sandbox files (DOCTREE 165 §4)"
HEAD_SHA=$(git rev-parse HEAD)
git push --quiet -u origin "$BR"
REMOTE_SHA=$(git ls-remote --heads origin "$BR" | cut -f1)
[[ "$REMOTE_SHA" == "$HEAD_SHA" ]] || die "push not verified: remote $REMOTE_SHA != local $HEAD_SHA"
echo "RESULT ARCHIVE_PUSHED repo=benjaminsw/CSMF2 branch=$BR commit=$HEAD_SHA remote=$REMOTE_SHA path=$D"
echo "--- sha256 (archive) ---"; cat "$D/SHA256SUMS"
echo "--- git blobs ---"; git ls-tree -r HEAD -- "$D" | awk '{print $3"  "$4}'
