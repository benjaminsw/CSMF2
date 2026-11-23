#!/usr/bin/env bash
# SEQREF-DLATP3-RUN v0.7.1 -- dlat_p3_run.sh: synthetic host validation of the revised D-LAT P2/P3 code: P3 fixtures AND P2 self-test (run card SEQREF-DLATP3-RUN v0.7.1)
# LIFETIME: KEEP
#
# DRAFT FOR REVIEW. DO NOT RUN before the run card is reviewed and the owner has authorised V10
# (LudWiki 243). SYNTHETIC TENSORS ONLY -- NOT D-LAT EVIDENCE. Uses no checkpoint, no MRI file,
# no stored row, no P4 file, nothing under results/v05lt.
#
# What it does:
#   preflight  -> refuses root; pinned checkout at the pin with no modified tracked file and no
#                 unexpected untracked importable file (C6); ch2-diag archive holds the six reviewed
#                 files; D-UNC adapter and the three archived D-UNC sources found by sha256;
#                 interpreter versions; available memory and free disk ENFORCED in bytes (C5)
#   snapshot   -> copies the six reviewed files from Git (git show, hash-checked) and the D-UNC
#                 adapter (hash-checked) into <run>/code, and the three D-UNC sources (hash-checked)
#                 into <run>/dunc, so exactly the reviewed bytes are executed
#   contain    -> <run>/code and <run>/dunc are made read-only; the script then PROVES by a write
#                 attempt that the executing user cannot create a file or a folder in them, and
#                 that no __pycache__ exists before the run
#   run        -> python -I -B <run>/code/diag_lat_p3test.py --mode host, then
#                 python -I -B <run>/code/diag_lat_selftest.py --dunc-archive <run>/dunc
#                 (both always run; neither is retried)
#   after      -> pinned checkout compared with its state before; no __pycache__ and no .pyc in
#                 the run folder; sha256 of every output file; ONE combined verdict
# Combined verdict: VALIDATION PASS only if the P3 test passes its gate, the self-test passes its
# gate, the pinned checkout is unchanged and no bytecode was written. Anything else is FAIL.
# The bytecode scan FAILS CLOSED: if the folder cannot be traversed completely, that is a
# failure (exit 92), never a proof that no bytecode exists.
# Any failed check stops with an ERROR line and a non-zero exit. No fallback, no retry.
#
# Usage:  bash dlat_p3_run.sh sources        # read-only: finds the D-UNC adapter and the three D-UNC
#                                            # sources of T19 by sha256 and prints their paths; creates
#                                            # nothing; can be run BEFORE the Git registration (V9)
#         bash dlat_p3_run.sh preflight      # checks only; creates nothing
#         bash dlat_p3_run.sh run            # preflight, then the run (needs V10)
#
# Two different memory controls (C5):
#   REQ_AVAIL_RAM_GIB  preflight requirement on the memory the operating system reports as
#                      available; the run is refused below it.
#   RAM_LIMIT_GIB      limit handed to the D-LAT memory guard, which compares the peak resident
#                      memory of the test process with it. The guard is a check made by the
#                      program; it is NOT a hard limit enforced by the operating system.
#
# Changelog:
#   v0.7.1 (2026-11-22) correction required by the review of v0.7 (LudWiki 243 LE55): the line
#        written into logs/run_context.txt named run card v0.6; it now names the run card that
#        governs the run, v0.7.1. One executable line changed; no check, gate, limit or command
#        changed. v0.7 preserved unchanged (not archived in Git; sha256 9471590f...4aaa).
#   v0.7 (2026-11-22) safeguard R2 (LudWiki 243 LE54): the bytecode scan no longer suppresses the
#        errors of find; a scan that cannot traverse the run folder completely returns exit 92 and
#        is a FAIL of the combined validation (detected bytecode stays exit 91); a failure to
#        hash every output file is reported and cannot leave the result at 0. New read-only
#        mode 'sources': the preflight checks the Git archive before it looks for the D-UNC
#        files, so it cannot answer before V9 whether the three sources of T19 are on the host;
#        'sources' runs only that lookup. Nothing else changed. v0.6 preserved unchanged (not
#        archived in Git; sha256 1d491d8d...8908).
#   v0.6 (2026-11-22) host validation of the revised P2/P3 code (plan SEQREF-DLAT-P23REV-HOST v0.1;
#        LudWiki 243 LE53): sha256 of diag_lat_stats.py v0.3, diag_lat_selftest.py v0.4 and
#        diag_lat_p3test.py v0.4; the P2 self-test is run in the same run folder against the three
#        archived D-UNC sources (found by sha256; T19 is never skipped) and is a required gate;
#        root is refused; the code snapshot is read-only and that is proved by a write attempt;
#        no-bytecode check before and after (exit 91); exact gates on both terminal summaries;
#        one combined verdict. Pin, tracked-file, allowlist, version, memory and disk checks
#        unchanged. v0.5 preserved (archived in commit 78967965...).
#   v0.5 (2026-11-21) after the failed host run of v0.4 (LudWiki 239 LE29; fix plan SEQREF-DLAT-P3FIX
#        v0.1): carries the sha256 of diag_lat_p3test.py v0.3; the summary shows the new fixture R7
#        and the provenance line; the result line explains exit status 4. No check of the
#        preflight was changed. v0.4 preserved (archived in commit 2f1ec6a0...).
#   v0.4 (2026-11-21) host preflight of v0.3 refused two untracked, Git-ignored files at the
#        repository root (v03_plot_results.py, v03_reconcile.py). They are added to the allowlist
#        with the sha256 the owner reported from the host. Nothing else changed. v0.3 preserved.
#   v0.3 (2026-11-21) review of v0.2 (R-U4): the folder exception is replaced by an allowlist of
#        two named files with registered sha256 (found by the owner's read-only inspection); the
#        tracked and untracked checks now also cover the top level of the repository and of
#        seqref_mri/, which are on the import path as well (author's finding). v0.2 preserved.
#   v0.2 (2026-11-21) review of v0.1 (C5, C6): available memory and free disk are enforced in
#        exact bytes; untracked importable files under the pinned src/tdiag/scripts are refused
#        (one documented folder is tolerated and listed with hashes); the before/after record
#        also lists ignored files; the script records its own sha256. v0.1 preserved.
#   v0.1 (2026-11-21) first version (draft for review).
set -euo pipefail

REPO="/home/benjamin/CSMFII"                         # pinned V05LT checkout (read only)
PIN="7455da8027f33a6cca4dbf72f04e72420e938f45"
PIN_BRANCH="seqref-v05lt"
WT="/home/benjamin/CSMFII/worktrees/ch2-diag"        # reviewed D-LAT files (Git archive)
DIR="seqref_mri/scripts/diag_lat"
PY="$REPO/seqref_mri/.venv_mri/bin/python"
RESULTS="$REPO/seqref_mri/results/diag_lat"
WANT_TORCH="2.5.1+cu121"; WANT_NUMPY="1.26.4"
REQ_AVAIL_RAM_GIB=16                                 # preflight: available memory required (C5)
RAM_LIMIT_GIB=16                                     # process-memory limit for the D-LAT memory guard (236 E5)
DISK_RESERVE_GIB=64                                  # registered disk reserve (236 E5), enforced in bytes
GIB=1073741824
# Untracked importable files are refused in the checked locations, except the files named here,
# and those only with exactly this sha256 (R-U4). They are the B1 host-check files v0.3.3
# registered in LudWiki 236; no pinned module imports from seqref_mri/scripts/diag_lat/.
declare -A ALLOWED_UNTRACKED=(
  [seqref_mri/scripts/diag_lat/dlat_b1_v03.py]=34e92911047fee027921c6ab53964d37ce9ee7529f1110c4aea7a8bb9922d789
  [seqref_mri/scripts/diag_lat/dlat_b1_v03_selftest.py]=c868b09f44f3ad6b9ae89c607e4230cd7ba4f8193ca7a97417437562a4645f6c
  # Git-ignored V03 helper scripts at the repository root (sha256 reported by the owner from the
  # host, 2026-11-21). No pinned module and no D-LAT file imports a module of either name.
  [v03_plot_results.py]=f05d67ef2cf771e75f6dc09576718830c601e215906abc9b6e0ab58c32db36a6
  [v03_reconcile.py]=bdf1a3883fd509200837948e08bce2d53d0fc35b3449c0c06f109e508f0d26cb
)
# Checked locations: the three pinned source folders (recursive) and, because the repository root
# and seqref_mri/src are put on the import path, the top level of the repository and of seqref_mri/.
TOP_LEVEL=(':(glob)*' ':(glob)seqref_mri/*')
DUNC_SHA="c1715841bf6805568b88f9195acbe27933a7e1fec9acc8bb22a33ff555a8505e"
# The three archived D-UNC sources that fixture T19 of the self-test compares against (the sha256
# registered in diag_lat_selftest.py). Found by sha256; a missing file stops the preflight.
DUNC_SRC=(diag_unc_core.py diag_unc_stats.py diag_unc_guard.py)
declare -A DUNC_SRC_SHA=(
  [diag_unc_core.py]=a0d3b4a4079ff290a93098eb7a1c53173df596080a6b2b51469dc0f757c18b06
  [diag_unc_stats.py]=b8706513a7da0517f477a81a49b4b7d2055e1451b47bec7dcc2c9b63a5de8f55
  [diag_unc_guard.py]=b3dc9f69226f97275712b1bef5cfe9e0fdf1716f944c9ee137ca7223fe2e5bc5
)
# Gates on the two terminal summaries (exact lines; anything else is a FAIL of the validation).
P3_GATE='^fixtures: PASS 8, FAIL 0, NOT RUN 0; checks 127; run COMPLETE$'
P3_PROV='^provenance: ok'
ST_GATE='^fixtures 17/17 passed; checks 254; all_pass=True; T19: run$'
PINNED_PATHS=(seqref_mri/src seqref_mri/tdiag seqref_mri/scripts)

FILES=(diag_lat_core.py diag_lat_stats.py diag_lat_guard.py diag_lat_selftest.py diag_lat_adapter.py diag_lat_p3test.py)
declare -A SHA=(
  [diag_lat_core.py]=acfccb4099e204870c6c82ab8876ad9682aa87d465b0f845c6960a18a7697bc9
  [diag_lat_stats.py]=247d9c8bf1c28f9b063468c6a29fa97d45b9ed5ca33ed7339211153cfeef2bb3
  [diag_lat_guard.py]=4c78796aa3185c54d53e73ac251fc0a8c7e59fd45509dfea490cf9221d3b76f3
  [diag_lat_selftest.py]=7ae7ccdb5cfc9a7b27076cbe2989013bf85b83060f59cb4f70ca7d192edb3d72
  [diag_lat_adapter.py]=300c30587f8c55f8901a3c36bbcd84d594a9fe6624d67e359c57cec48a0760e2
  [diag_lat_p3test.py]=0e9236574e26ade509d62df166843bb83bc5494f95dfe17a422c46409971056d
)

# Rehearsal overrides (author's sandbox only). Ignored unless DLAT_REHEARSAL=1; announced loudly.
if [ "${DLAT_REHEARSAL:-0}" = "1" ]; then
  echo "#### REHEARSAL OVERRIDES ACTIVE -- NOT A HOST RUN ####"
  REPO="${DLAT_REPO:-$REPO}"; WT="${DLAT_WT:-$WT}"; PY="${DLAT_PY:-$PY}"; RESULTS="${DLAT_RESULTS:-$RESULTS}"
  PIN="${DLAT_PIN:-$PIN}"; PIN_BRANCH="${DLAT_PIN_BRANCH:-$PIN_BRANCH}"; WANT_TORCH="${DLAT_TORCH:-$WANT_TORCH}"
  RAM_LIMIT_GIB="${DLAT_RAM_GIB:-$RAM_LIMIT_GIB}"; DISK_RESERVE_GIB="${DLAT_RESERVE_GIB:-$DISK_RESERVE_GIB}"
  REQ_AVAIL_RAM_GIB="${DLAT_AVAIL_GIB:-$REQ_AVAIL_RAM_GIB}"
  DUNC_ADAPTER="${DLAT_DUNC_ADAPTER:-}"
  MODE="sandbox"
else
  DUNC_ADAPTER=""
  MODE="host"
fi

die() { echo "ERROR: $*" >&2; exit 1; }
SELF=$(readlink -f "$0")

pinned_state() {   # HEAD, branch, every modified or untracked path, and every untracked file
  git -C "$REPO" rev-parse HEAD                      # (ignored ones included) under the pinned paths
  git -C "$REPO" rev-parse --abbrev-ref HEAD
  git -C "$REPO" status --porcelain=v1 --untracked-files=all
  echo "-- untracked, ignored included, in the checked locations:"
  git -C "$REPO" ls-files --others -- "${PINNED_PATHS[@]}" "${TOP_LEVEL[@]}" | sort
}

untracked_check() {   # C6 / R-U4: no unexpected untracked importable file in the checked locations
  local f got bad=0 tolerated=0
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    case "$f" in */__pycache__/*) continue ;; esac          # bytecode caches cannot replace a source module
    case "$f" in *.py|*.pyc|*.pyo|*.so|*.pth) ;; *) continue ;; esac
    if [ -n "${ALLOWED_UNTRACKED[$f]:-}" ]; then
      got=$(sha256sum "$REPO/$f" | cut -d' ' -f1)
      if [ "$got" = "${ALLOWED_UNTRACKED[$f]}" ]; then
        echo "allowed untracked file (registered sha256): $got  $f"; tolerated=$((tolerated+1))
      else
        echo "ERROR: allowlisted file $f has sha256 $got, registered ${ALLOWED_UNTRACKED[$f]}" >&2; bad=$((bad+1))
      fi
    else
      echo "ERROR: unexpected untracked importable file in the pinned checkout: $f" >&2; bad=$((bad+1))
    fi
  done < <(git -C "$REPO" ls-files --others -- "${PINNED_PATHS[@]}" "${TOP_LEVEL[@]}" | sort)
  [ "$bad" = "0" ] || die "$bad unexpected or changed untracked importable file(s) in the pinned checkout"
  echo "pinned checkout: no unexpected untracked importable file ($tolerated allowlisted, each with its registered sha256)"
}

find_dunc_adapter() {
  if [ -n "$DUNC_ADAPTER" ]; then echo "$DUNC_ADAPTER"; return; fi
  local hits
  hits=$(find "$REPO" -name diag_unc_adapter.py -not -path '*/results/*' -type f 2>/dev/null | sort)
  for h in $hits; do
    if [ "$(sha256sum "$h" | cut -d' ' -f1)" = "$DUNC_SHA" ]; then echo "$h"; return; fi
  done
  die "no diag_unc_adapter.py with the registered sha256 $DUNC_SHA found under $REPO"
}

find_dunc_source() {   # $1 file name -> path of a copy with the registered sha256, or stop
  local h
  while IFS= read -r h; do
    [ -n "$h" ] || continue
    if [ "$(sha256sum "$h" | cut -d' ' -f1)" = "${DUNC_SRC_SHA[$1]}" ]; then echo "$h"; return; fi
  done < <(find "$REPO" -name "$1" -not -path '*/results/*' -type f 2>/dev/null | sort)
  die "no $1 with the registered sha256 ${DUNC_SRC_SHA[$1]} found under $REPO (T19 is never skipped; return for review)"
}

no_bytecode() {   # $1 folder, $2 when: returns 91 if a bytecode cache or file exists, 92 if the scan itself failed
  local hits
  [ -d "$1" ] || { echo "ERROR: bytecode scan $2: $1 is not a folder" >&2; return 92; }
  # FAIL CLOSED: the errors of find are NOT suppressed; a non-zero status of find (a folder that
  # cannot be read or entered, a vanished entry) means the scan is incomplete and proves nothing.
  if ! hits=$(find "$1" \( -name __pycache__ -o -name '*.pyc' -o -name '*.pyo' \) -print); then
    echo "ERROR: bytecode scan $2 could not traverse $1 completely (find failed; see its message above): the scan proves nothing" >&2
    return 92
  fi
  hits=$(printf '%s\n' "$hits" | sed '/^$/d' | sort)
  if [ -n "$hits" ]; then
    echo "ERROR: bytecode found $2 in $1:" >&2; echo "$hits" >&2
    return 91
  fi
  echo "no-bytecode check ($2): complete traversal; no __pycache__, .pyc or .pyo under $1"
}

must_not_write() {   # $1 folder: PROVES that the executing user cannot create a file or folder in it
  local d="$1"
  [ ! -w "$d" ] || die "$d is still writable for user $(id -un) after chmod"
  if ( : > "$d/.dlat_write_probe" ) 2>/dev/null; then rm -f "$d/.dlat_write_probe"; die "a file could be created in $d: the snapshot is not read-only for user $(id -un)"; fi
  if mkdir "$d/__pycache__" 2>/dev/null; then rmdir "$d/__pycache__"; die "a __pycache__ folder could be created in $d: bytecode would not be contained"; fi
  echo "containment: user $(id -un) (uid $(id -u)) cannot create a file or a __pycache__ folder in $d (proved by attempt)"
}

preflight() {
  echo "== preflight $(date -u +%Y-%m-%dT%H:%M:%SZ) host $(hostname) mode $MODE"
  [ "$(id -u)" != "0" ] || die "this script refuses to run as root: a read-only snapshot does not stop root from writing bytecode"
  echo "executing user: $(id -un) (uid $(id -u)), not root"
  [ -d "$REPO/.git" ] || [ -f "$REPO/.git" ] || die "pinned checkout not found: $REPO"
  head_now=$(git -C "$REPO" rev-parse HEAD)
  [ "$head_now" = "$PIN" ] || die "pinned checkout HEAD is $head_now, expected the pin $PIN"
  br=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
  [ "$br" = "$PIN_BRANCH" ] || die "pinned checkout is on branch $br, expected $PIN_BRANCH"
  git -C "$REPO" diff --quiet HEAD -- "${PINNED_PATHS[@]}" || die "a tracked file under ${PINNED_PATHS[*]} differs from the pin"
  git -C "$REPO" diff --quiet HEAD -- ':(glob)*.py' ':(glob)seqref_mri/*.py' || die "a tracked .py file at the top level of the repository or of seqref_mri/ differs from the pin"
  n_py=$(git -C "$REPO" ls-tree -r --name-only "$PIN" -- "${PINNED_PATHS[@]}" | grep -c '\.py$')
  echo "pinned checkout: HEAD = pin $PIN, branch $br, $n_py tracked .py files under ${PINNED_PATHS[*]} identical to the pin"
  untracked_check
  [ -d "$WT" ] || die "ch2-diag worktree not found: $WT"
  ARCH_COMMIT=$(git -C "$WT" rev-parse HEAD)
  for f in "${FILES[@]}"; do
    got=$(git -C "$WT" show "HEAD:$DIR/$f" 2>/dev/null | sha256sum | cut -d' ' -f1)
    [ "$got" = "${SHA[$f]}" ] || die "archived $f at $ARCH_COMMIT has sha256 $got, reviewed ${SHA[$f]}"
  done
  echo "archive: ch2-diag commit $ARCH_COMMIT holds the six reviewed files"
  DUNC_PATH=$(find_dunc_adapter)
  [ "$(sha256sum "$DUNC_PATH" | cut -d' ' -f1)" = "$DUNC_SHA" ] || die "diag_unc_adapter.py at $DUNC_PATH does not have the registered sha256"
  echo "D-UNC adapter: $DUNC_PATH (sha256 registered)"
  declare -gA DUNC_SRC_PATH=()
  for f in "${DUNC_SRC[@]}"; do
    DUNC_SRC_PATH[$f]=$(find_dunc_source "$f")
    echo "D-UNC source for T19: ${DUNC_SRC_SHA[$f]}  ${DUNC_SRC_PATH[$f]}"
  done
  [ -x "$PY" ] || die "interpreter not found: $PY"
  vers=$("$PY" -I -B -c "import sys,torch,numpy,h5py;print(sys.version.split()[0],torch.__version__,numpy.__version__,h5py.__version__)")
  echo "interpreter: $PY  python/torch/numpy/h5py = $vers"
  set -- $vers
  [ "$2" = "$WANT_TORCH" ] || die "torch is $2, expected $WANT_TORCH"
  [ "$3" = "$WANT_NUMPY" ] || die "numpy is $3, expected $WANT_NUMPY"
  free_b=$(df -B1 --output=avail "$REPO" | tail -1 | tr -dc '0-9')
  need_b=$((DISK_RESERVE_GIB * GIB))
  [ -n "$free_b" ] || die "could not read the free disk space of $REPO"
  [ "$free_b" -ge "$need_b" ] || die "free disk $free_b bytes is below the reserve $need_b bytes (${DISK_RESERVE_GIB} GiB)"
  mem_b=$(awk '/^MemAvailable:/ {printf "%d", $2*1024}' /proc/meminfo)
  need_m=$((REQ_AVAIL_RAM_GIB * GIB))
  [ -n "$mem_b" ] && [ "$mem_b" -gt 0 ] || die "could not read MemAvailable from /proc/meminfo"
  [ "$mem_b" -ge "$need_m" ] || die "available memory $mem_b bytes is below the requirement $need_m bytes (${REQ_AVAIL_RAM_GIB} GiB)"
  echo "resources: free disk $free_b bytes >= reserve $need_b; available memory $mem_b bytes >= required $need_m"
  echo "process-memory limit handed to the D-LAT memory guard: ${RAM_LIMIT_GIB} GiB (a program check, not an OS limit)"
  echo "PREFLIGHT PASSED"
}

case "${1:-}" in
  sources)
    echo "== D-UNC sources $(date -u +%Y-%m-%dT%H:%M:%SZ) host $(hostname) (read-only; creates nothing)"
    DUNC_PATH=$(find_dunc_adapter)
    [ "$(sha256sum "$DUNC_PATH" | cut -d' ' -f1)" = "$DUNC_SHA" ] || die "diag_unc_adapter.py at $DUNC_PATH does not have the registered sha256"
    echo "D-UNC adapter: $DUNC_SHA  $DUNC_PATH"
    for f in "${DUNC_SRC[@]}"; do
      p=$(find_dunc_source "$f")
      echo "D-UNC source for T19: ${DUNC_SRC_SHA[$f]}  $p"
    done
    echo "SOURCES FOUND: the D-UNC adapter and the three D-UNC sources of T19, each at its registered sha256" ;;
  preflight)
    preflight ;;
  run)
    preflight
    RUN="$RESULTS/p3_synthetic_$(date -u +%Y%m%dT%H%M%SZ)"
    [ ! -e "$RUN" ] || die "run folder already exists: $RUN"
    mkdir -p "$RUN/code" "$RUN/dunc" "$RUN/logs" "$RUN/selftest"
    echo "run folder: $RUN"
    pinned_state > "$RUN/logs/pinned_before.txt"
    for f in "${FILES[@]}"; do
      git -C "$WT" show "$ARCH_COMMIT:$DIR/$f" > "$RUN/code/$f"
      [ "$(sha256sum "$RUN/code/$f" | cut -d' ' -f1)" = "${SHA[$f]}" ] || die "snapshot of $f does not have the reviewed sha256"
    done
    cp "$DUNC_PATH" "$RUN/code/diag_unc_adapter.py"
    [ "$(sha256sum "$RUN/code/diag_unc_adapter.py" | cut -d' ' -f1)" = "$DUNC_SHA" ] || die "snapshot of diag_unc_adapter.py does not have the registered sha256"
    for f in "${DUNC_SRC[@]}"; do
      cp "${DUNC_SRC_PATH[$f]}" "$RUN/dunc/$f"
      [ "$(sha256sum "$RUN/dunc/$f" | cut -d' ' -f1)" = "${DUNC_SRC_SHA[$f]}" ] || die "snapshot of $f does not have the registered sha256"
    done
    (cd "$RUN/code" && sha256sum *.py) > "$RUN/logs/code_sha256.txt"
    (cd "$RUN/dunc" && sha256sum *.py) > "$RUN/logs/dunc_sha256.txt"
    [ "$(ls "$RUN/code" | wc -l)" = "7" ] && [ "$(ls "$RUN/dunc" | wc -l)" = "3" ] || die "the snapshot does not hold exactly 7 + 3 files"
    chmod -R a-w "$RUN/code" "$RUN/dunc"
    must_not_write "$RUN/code"
    must_not_write "$RUN/dunc"
    no_bytecode "$RUN" "before the run" || exit $?
    { echo "run card SEQREF-DLATP3-RUN v0.7.1"; echo "run script sha256 $(sha256sum "$SELF" | cut -d' ' -f1)"; echo "mode $MODE"; echo "pin $PIN"; echo "archive commit $ARCH_COMMIT";
      echo "interpreter $PY"; echo "versions $vers"; echo "started $(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$RUN/logs/run_context.txt"
    set +e
    "$PY" -I -B "$RUN/code/diag_lat_p3test.py" --pinned-root "$REPO" --out "$RUN/out" --mode "$MODE" \
        --ram-limit-gib "$RAM_LIMIT_GIB" > "$RUN/logs/p3_terminal.log" 2>&1
    rc=$?
    set -e
    echo "exit $rc" >> "$RUN/logs/p3_terminal.log"
    echo "p3 finished $(date -u +%Y-%m-%dT%H:%M:%SZ) exit $rc" >> "$RUN/logs/run_context.txt"
    # P2 self-test: always run, whatever the P3 test returned; never retried. T19 uses the snapshot.
    set +e
    (cd "$RUN/selftest" && "$PY" -I -B "$RUN/code/diag_lat_selftest.py" --dunc-archive "$RUN/dunc" \
        --out "$RUN/selftest/selftest_report.json") > "$RUN/logs/selftest_terminal.log" 2>&1
    rc_st=$?
    set -e
    echo "exit $rc_st" >> "$RUN/logs/selftest_terminal.log"
    echo "selftest finished $(date -u +%Y-%m-%dT%H:%M:%SZ) exit $rc_st" >> "$RUN/logs/run_context.txt"
    pinned_state > "$RUN/logs/pinned_after.txt"
    pinned_ok=1
    if cmp -s "$RUN/logs/pinned_before.txt" "$RUN/logs/pinned_after.txt"; then
      echo "pinned checkout: unchanged by the run"
    else
      echo "ERROR: the pinned checkout changed during the run:" >&2
      diff "$RUN/logs/pinned_before.txt" "$RUN/logs/pinned_after.txt" >&2 || true
      pinned_ok=0
    fi
    bytecode_rc=0
    no_bytecode "$RUN" "after the run" || bytecode_rc=$?
    bytecode_ok=1; [ "$bytecode_rc" = "0" ] || bytecode_ok=0
    (cd "$RUN/code" && sha256sum -c --quiet "$RUN/logs/code_sha256.txt") || die "a file of the code snapshot changed during the run"
    (cd "$RUN/dunc" && sha256sum -c --quiet "$RUN/logs/dunc_sha256.txt") || die "a file of the D-UNC snapshot changed during the run"
    grep -E "^(R5|R7|T9|T10|T11|T12b|R1a|R1b) |^pinned preflight|^provenance:|^fixtures:|^peak resident|\[FAIL\]|^Traceback|Error" "$RUN/logs/p3_terminal.log" || true
    grep -E "^(T[0-9]+a?|LOG) +(PASS|FAIL)|^fixtures |\[FAIL\]|^Traceback" "$RUN/logs/selftest_terminal.log" || true
    p3_ok=0; st_ok=0
    if [ "$rc" = "0" ] && grep -qE "$P3_GATE" "$RUN/logs/p3_terminal.log" && grep -qE "$P3_PROV" "$RUN/logs/p3_terminal.log" \
       && [ -f "$RUN/out/p3_report.json" ] && [ -f "$RUN/out/p3_resource.json" ]; then p3_ok=1; fi
    if [ "$rc_st" = "0" ] && grep -qE "$ST_GATE" "$RUN/logs/selftest_terminal.log" && [ -f "$RUN/selftest/selftest_report.json" ]; then st_ok=1; fi
    st() { [ "$1" = "1" ] && echo PASS || echo FAIL; }
    final=0
    [ "$p3_ok" = "1" ] && [ "$st_ok" = "1" ] || final=1
    [ "$bytecode_ok" = "1" ] || final=$bytecode_rc
    [ "$pinned_ok" = "1" ] || final=90
    { echo "P3 test (exit $rc; gate: 8 fixtures PASS, 127 checks, COMPLETE, provenance ok): $(st $p3_ok)"
      echo "P2 self-test (exit $rc_st; gate: 17/17 fixtures, 254 checks, all_pass, T19 run): $(st $st_ok)"
      echo "pinned checkout unchanged: $(st $pinned_ok)"
      if [ "$bytecode_rc" = "92" ]; then echo "no bytecode written: FAIL (the scan could not be completed; nothing is proved)"; else echo "no bytecode written: $(st $bytecode_ok)"; fi
      if [ "$final" = "0" ]; then echo "COMBINED VALIDATION: PASS"; else echo "COMBINED VALIDATION: FAIL (exit $final) -- evidence preserved; no retry; return for review"; fi
    } | tee "$RUN/logs/validation_verdict.txt"
    echo "verdict exit $final" >> "$RUN/logs/run_context.txt"
    if (cd "$RUN" && find . -type f ! -name outputs_sha256.txt | sort | xargs sha256sum) > "$RUN/outputs_sha256.txt"; then
      echo "outputs and hashes: $RUN/outputs_sha256.txt"
    else
      echo "ERROR: not every output file could be listed and hashed; $RUN/outputs_sha256.txt is incomplete" >&2
      [ "$final" != "0" ] || final=1
    fi
    echo "RESULT exit $final  (0 = COMBINED VALIDATION PASS; 1 = the P3 test or the self-test did not pass its gate, or the outputs could not all be hashed; 90 = pinned checkout changed; 91 = bytecode written; 92 = bytecode scan could not be completed. Only 0 is a pass.)"
    echo "The folders $RUN/code and $RUN/dunc stay read-only (evidence)."
    echo "Paste back: this terminal output, $RUN/logs/p3_terminal.log, $RUN/logs/selftest_terminal.log, $RUN/logs/validation_verdict.txt, $RUN/out/p3_report.json, $RUN/out/p3_resource.json, $RUN/selftest/selftest_report.json, $RUN/outputs_sha256.txt"
    exit "$final" ;;
  *)
    die "usage: bash $0 sources|preflight|run" ;;
esac
