#!/usr/bin/env bash
# SEQREF-DLATP3-RUN v0.5 -- dlat_p3_run.sh: synthetic host run of the D-LAT P3 fixtures (run card SEQREF-DLATP3-RUN v0.5)
# LIFETIME: KEEP
#
# DRAFT FOR REVIEW. DO NOT RUN before the run card is reviewed and the owner has authorised V7
# (LudWiki 238). SYNTHETIC TENSORS ONLY -- NOT D-LAT EVIDENCE. Uses no checkpoint, no MRI file,
# no stored row, no P4 file, nothing under results/v05lt.
#
# What it does:
#   preflight  -> pinned checkout at the pin with no modified tracked file and no unexpected
#                 untracked importable file (C6); ch2-diag archive holds the six reviewed files;
#                 interpreter versions; available memory and free disk ENFORCED in bytes (C5)
#   snapshot   -> copies the six reviewed files from Git (git show, hash-checked) and the D-UNC
#                 adapter (hash-checked) into <run>/code, so exactly the reviewed bytes are executed
#   run        -> python -I -B <run>/code/diag_lat_p3test.py --mode host (no bytecode is written,
#                 so the pinned checkout is not altered)
#   after      -> pinned checkout compared with its state before; sha256 of every output file
# Any failed check stops with an ERROR line and a non-zero exit. No fallback, no retry.
#
# Usage:  bash dlat_p3_run.sh preflight      # checks only; creates nothing
#         bash dlat_p3_run.sh run            # preflight, then the run (needs V7)
#
# Two different memory controls (C5):
#   REQ_AVAIL_RAM_GIB  preflight requirement on the memory the operating system reports as
#                      available; the run is refused below it.
#   RAM_LIMIT_GIB      limit handed to the D-LAT memory guard, which compares the peak resident
#                      memory of the test process with it. The guard is a check made by the
#                      program; it is NOT a hard limit enforced by the operating system.
#
# Changelog:
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
PINNED_PATHS=(seqref_mri/src seqref_mri/tdiag seqref_mri/scripts)

FILES=(diag_lat_core.py diag_lat_stats.py diag_lat_guard.py diag_lat_selftest.py diag_lat_adapter.py diag_lat_p3test.py)
declare -A SHA=(
  [diag_lat_core.py]=acfccb4099e204870c6c82ab8876ad9682aa87d465b0f845c6960a18a7697bc9
  [diag_lat_stats.py]=1d7db11d38b11c3d8f67bcae086eb29daa5a5d903b5ce5e9446c43ce9b109831
  [diag_lat_guard.py]=4c78796aa3185c54d53e73ac251fc0a8c7e59fd45509dfea490cf9221d3b76f3
  [diag_lat_selftest.py]=0aeaa9e081ccceaf9069585ddf39bd12ecaa211ea4c4253bff599b772f91b794
  [diag_lat_adapter.py]=300c30587f8c55f8901a3c36bbcd84d594a9fe6624d67e359c57cec48a0760e2
  [diag_lat_p3test.py]=777685bea9b279af4382d83de917b4024d6cbf07f7702f602741fc01895a338f
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

preflight() {
  echo "== preflight $(date -u +%Y-%m-%dT%H:%M:%SZ) host $(hostname) mode $MODE"
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
  preflight)
    preflight ;;
  run)
    preflight
    RUN="$RESULTS/p3_synthetic_$(date -u +%Y%m%dT%H%M%SZ)"
    [ ! -e "$RUN" ] || die "run folder already exists: $RUN"
    mkdir -p "$RUN/code" "$RUN/logs"
    echo "run folder: $RUN"
    pinned_state > "$RUN/logs/pinned_before.txt"
    for f in "${FILES[@]}"; do
      git -C "$WT" show "$ARCH_COMMIT:$DIR/$f" > "$RUN/code/$f"
      [ "$(sha256sum "$RUN/code/$f" | cut -d' ' -f1)" = "${SHA[$f]}" ] || die "snapshot of $f does not have the reviewed sha256"
    done
    cp "$DUNC_PATH" "$RUN/code/diag_unc_adapter.py"
    [ "$(sha256sum "$RUN/code/diag_unc_adapter.py" | cut -d' ' -f1)" = "$DUNC_SHA" ] || die "snapshot of diag_unc_adapter.py does not have the registered sha256"
    (cd "$RUN/code" && sha256sum *.py) > "$RUN/logs/code_sha256.txt"
    { echo "run card SEQREF-DLATP3-RUN v0.5"; echo "run script sha256 $(sha256sum "$SELF" | cut -d' ' -f1)"; echo "mode $MODE"; echo "pin $PIN"; echo "archive commit $ARCH_COMMIT";
      echo "interpreter $PY"; echo "versions $vers"; echo "started $(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$RUN/logs/run_context.txt"
    set +e
    "$PY" -I -B "$RUN/code/diag_lat_p3test.py" --pinned-root "$REPO" --out "$RUN/out" --mode "$MODE" \
        --ram-limit-gib "$RAM_LIMIT_GIB" > "$RUN/logs/p3_terminal.log" 2>&1
    rc=$?
    set -e
    echo "exit $rc" >> "$RUN/logs/p3_terminal.log"
    echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ) exit $rc" >> "$RUN/logs/run_context.txt"
    pinned_state > "$RUN/logs/pinned_after.txt"
    if cmp -s "$RUN/logs/pinned_before.txt" "$RUN/logs/pinned_after.txt"; then
      echo "pinned checkout: unchanged by the run"
    else
      echo "ERROR: the pinned checkout changed during the run:" >&2
      diff "$RUN/logs/pinned_before.txt" "$RUN/logs/pinned_after.txt" >&2 || true
      rc=90
    fi
    (cd "$RUN" && find . -type f ! -name outputs_sha256.txt | sort | xargs sha256sum) > "$RUN/outputs_sha256.txt"
    grep -E "^(R5|R7|T9|T10|T11|T12b|R1a|R1b) |^pinned preflight|^provenance:|^fixtures:|^peak resident|\[FAIL\]|^Traceback|Error" "$RUN/logs/p3_terminal.log" || true
    echo "outputs and hashes: $RUN/outputs_sha256.txt"
    echo "RESULT exit $rc  (0 = complete and all PASS; 1 = a fixture FAILED or the program stopped; 2 = INCOMPLETE; 4 = PROVENANCE ERROR, not a pass; 90 = pinned checkout changed)"
    echo "Paste back: this terminal output, $RUN/logs/p3_terminal.log, $RUN/out/p3_report.json, $RUN/out/p3_resource.json, $RUN/outputs_sha256.txt"
    exit "$rc" ;;
  *)
    die "usage: bash $0 preflight|run" ;;
esac
