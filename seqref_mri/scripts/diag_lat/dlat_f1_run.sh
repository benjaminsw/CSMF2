#!/usr/bin/env bash
# SEQREF-DLAT-F1RUN v0.3 -- dlat_f1_run.sh: launcher of the D-LAT F1 preparation run and of its bind-only preflight (plan SEQREF-DLAT-F1 v0.2 with amendment AM-F1-1, criteria A1, A4, A12; driver diag_lat_f1.py v0.3)
# LIFETIME: KEEP
#
# DRAFT FOR REVIEW. DO NOT RUN 'bind' or 'run' before the amendment AM-F1-1 and run card v0.3 are
# reviewed, both are registered in Git and the owner has authorised the step: 'run' needs V15
# (LudWiki 253). The first F1 run (V13, run card v0.2) FAILED and stays a failure; this launcher
# starts a NEW run with the amended card. F1 writes NO measure, contrast, class or figure.
#
# What it does:
#   preflight -> refuses root; pinned checkout at the pin, no modified tracked file, no unexpected
#                untracked importable file; ONE ch2-diag commit holds every reviewed file at its
#                reviewed sha256; the registered D-UNC adapter and the registered manifest_f2.json
#                found by sha256; interpreter versions; available memory and free disk in bytes.
#                Read-only; creates nothing.
#   snapshot  -> the reviewed files are taken from that one Git commit (git show, hash-checked)
#                into <run>/code, the run card into <run>/card, the D-UNC adapter into <run>/dunc
#                and the manifest into <run>/input; all made read-only, and the script PROVES by a
#                write attempt that the executing user cannot create a file or folder there
#   bind      -> python -I -B diag_lat_f1.py --bind-only : manifest, checkpoint and stored-row
#                files by sha256, row order. No PyTorch, no checkpoint tensor, no inference.
#   run       -> python -I -B diag_lat_f1.py : the full F1 (stages D1-D9). Once; never retried.
#   after     -> pinned checkout and the V05LT results folder compared with their state before
#                (exit 90); no bytecode in the run folder, fail-closed scan (exit 91, 92); the
#                snapshot unchanged; the driver's own sha256 manifest verified; sha256 of every
#                file of the run folder; ONE verdict.
# Verdict PASS only if the driver exits 0, prints its PASS line, wrote its verdict file with the
# PASS line, its output manifest verifies, the pinned checkout and the results folder are
# unchanged and no bytecode was written. Anything else is FAIL; evidence is kept; no retry.
# TWO DIFFERENT TIME LIMITS (LudWiki 250 LE86):
#   F1 wall ceiling  3,600 s  the maximum authorised F1 run time (235 section 7). Enforced by the DRIVER
#                             after every model state, after the last stage and directly before the
#                             verdict: a run that crosses it is F1: FAIL, whatever else it computed.
#   TIMEOUT_S        5,400 s  (90 minutes; the owner's decision) an operating-system backstop that kills a
#                             process that hangs. The 1,800 s above the ceiling are a TERMINATION ALLOWANCE,
#                             so that the driver can reach its own check and write its failure evidence.
#                             They are NOT permissible run time: nothing between 3,600 s and 5,400 s can pass.
# Not in the plans' wording, DISCLOSED for review: that backstop; the
# results folder is compared by file list, size and modification time (the driver and the
# binder compare its registered files by sha256); the run's own folder, which lies inside the
# pinned checkout's folder, is left out of the before/after Git status (found in the rehearsal).
# No fallback, no retry. Any failed check stops with an ERROR line and a non-zero exit.
#
# Usage:  bash dlat_f1_run.sh preflight     # checks only; creates nothing
#         bash dlat_f1_run.sh bind          # preflight, snapshot, bind-only preflight (needs authorisation)
#         bash dlat_f1_run.sh run           # preflight, snapshot, the F1 run (needs V13)
#
# Changelog:
#   v0.3 (2026-11-23) F1 amendment AM-F1-1 (SEQREF-DLAT-F1AM v0.1; LudWiki 253): the run card is
#        dlat_f1_run_card_v0.3.json with its sha256 (two round-trip tolerances changed in the card;
#        nothing in the code); the version line of the run context. No check, gate, file list,
#        time limit or exit code changed. The seven code files are those of commit 9db28992.
#   v0.2 (2026-11-23) review of v0.1 and the owner's decision (LudWiki 250 LE86): TIMEOUT_S 4500 -> 5400
#        ("timeout can set to 90 minutes"), described as a termination allowance and separated from
#        the 3,600 s F1 ceiling in the header and in the terminal message; sha256 of driver v0.3;
#        the version line of the run context. No check, gate or exit code changed.
#   v0.1 (2026-11-23) first version (draft for review), on the pattern of dlat_p3_run.sh v0.7.1.
set -euo pipefail

REPO="/home/benjamin/CSMFII"                         # pinned V05LT checkout (read only)
PIN="7455da8027f33a6cca4dbf72f04e72420e938f45"
PIN_BRANCH="seqref-v05lt"
WT="/home/benjamin/CSMFII/worktrees/ch2-diag"        # reviewed D-LAT files (Git archive)
DIR="seqref_mri/scripts/diag_lat"
PY="$REPO/seqref_mri/.venv_mri/bin/python"
EVID="/home/benjamin/CSMFII/v06lt_evidence/DIAG_LAT" # 235 section 7, location
V05LT_RESULTS="$REPO/seqref_mri/results/v05lt"
WANT_TORCH="2.5.1+cu121"; WANT_NUMPY="1.26.4"
REQ_AVAIL_RAM_GIB=16                                 # preflight: available memory required
DISK_RESERVE_GIB=64                                  # registered disk reserve, enforced in bytes
TIMEOUT_S=5400                                       # backstop (90 min): F1 wall ceiling 3600 s + 1800 s termination allowance
GIB=1073741824
declare -A ALLOWED_UNTRACKED=(                        # as dlat_p3_run.sh v0.7.1 (LudWiki 236, 239)
  [seqref_mri/scripts/diag_lat/dlat_b1_v03.py]=34e92911047fee027921c6ab53964d37ce9ee7529f1110c4aea7a8bb9922d789
  [seqref_mri/scripts/diag_lat/dlat_b1_v03_selftest.py]=c868b09f44f3ad6b9ae89c607e4230cd7ba4f8193ca7a97417437562a4645f6c
  [v03_plot_results.py]=f05d67ef2cf771e75f6dc09576718830c601e215906abc9b6e0ab58c32db36a6
  [v03_reconcile.py]=bdf1a3883fd509200837948e08bce2d53d0fc35b3449c0c06f109e508f0d26cb
)
TOP_LEVEL=(':(glob)*' ':(glob)seqref_mri/*')
PINNED_PATHS=(seqref_mri/src seqref_mri/tdiag seqref_mri/scripts)
DUNC_SHA="c1715841bf6805568b88f9195acbe27933a7e1fec9acc8bb22a33ff555a8505e"
MANIFEST_SHA="d1956bf370dc6c1cc82caf9804dc9731cbec970c0f8205265c14bdf72e3a6215"   # registered D-UNC manifest_f2.json
CARD="dlat_f1_run_card_v0.3.json"
# The files the F1 run executes or reads, each at its reviewed sha256, all from ONE ch2-diag commit.
FILES=(diag_lat_f1.py diag_lat_bind.py diag_lat_adapter.py diag_lat_core.py diag_lat_guard.py diag_lat_population.py diag_lat_stats.py)
declare -A SHA=(
  [diag_lat_f1.py]=ea0f177f65101b29dce170ac216e044dd7af068b62800df68169f56172370be0
  [diag_lat_bind.py]=2793db5283779d6d836cdd76704efaf415d689c7e289c97c273ba0f8af7697c2
  [diag_lat_adapter.py]=c6dc185af42b3e99e9859f74bb810e6382e3e7ed6246a9d33ff7b01d04a5a621
  [diag_lat_core.py]=acfccb4099e204870c6c82ab8876ad9682aa87d465b0f845c6960a18a7697bc9
  [diag_lat_guard.py]=4c78796aa3185c54d53e73ac251fc0a8c7e59fd45509dfea490cf9221d3b76f3
  [diag_lat_population.py]=f695272d834feb8801dde769fa7d7d7e62edd27cd230f32bd96037df791085f9
  [diag_lat_stats.py]=47f3c0358cd16fb179868fbebe6d68cd1d0c7370c098a71ae36fdd17d7ece831
  [$CARD]=45142fac001e5d428523c8d7faf4f2540e7ce918f357ea09ac11e0ee93ad849c
)

# Rehearsal overrides (author's sandbox only). Ignored unless DLAT_REHEARSAL=1; announced loudly.
if [ "${DLAT_REHEARSAL:-0}" = "1" ]; then
  echo "#### REHEARSAL OVERRIDES ACTIVE -- NOT A HOST RUN ####"
  REPO="${DLAT_REPO:-$REPO}"; WT="${DLAT_WT:-$WT}"; PY="${DLAT_PY:-$PY}"; EVID="${DLAT_EVID:-$EVID}"
  PIN="${DLAT_PIN:-$PIN}"; PIN_BRANCH="${DLAT_PIN_BRANCH:-$PIN_BRANCH}"; WANT_TORCH="${DLAT_TORCH:-$WANT_TORCH}"
  WANT_NUMPY="${DLAT_NUMPY:-$WANT_NUMPY}"; DISK_RESERVE_GIB="${DLAT_RESERVE_GIB:-$DISK_RESERVE_GIB}"
  REQ_AVAIL_RAM_GIB="${DLAT_AVAIL_GIB:-$REQ_AVAIL_RAM_GIB}"; V05LT_RESULTS="$REPO/seqref_mri/results/v05lt"
  MODE="sandbox"
else
  MODE="host"
fi

die() { echo "ERROR: $*" >&2; exit 1; }
SELF=$(readlink -f "$0")

pinned_state() {   # HEAD, branch, every modified or untracked path, and every untracked file in the checked locations.
  # The evidence folder lies inside the pinned checkout's folder: THIS run's own folder ($RUN_REL, set by
  # execute) is left out of the status, and nothing else is. Other evidence folders stay in the comparison.
  git -C "$REPO" rev-parse HEAD
  git -C "$REPO" rev-parse --abbrev-ref HEAD
  if [ -n "${RUN_REL:-}" ]; then git -C "$REPO" status --porcelain=v1 --untracked-files=all -- . ":(exclude)$RUN_REL"
  else git -C "$REPO" status --porcelain=v1 --untracked-files=all; fi
  echo "-- untracked, ignored included, in the checked locations:"
  git -C "$REPO" ls-files --others -- "${PINNED_PATHS[@]}" "${TOP_LEVEL[@]}" | sort
}

results_state() {   # A4: every file of the V05LT results folder with size and modification time; fails if it cannot be listed
  [ -d "$V05LT_RESULTS" ] || { echo "(no folder $V05LT_RESULTS)"; return 0; }
  find "$V05LT_RESULTS" -type f -printf '%P\t%s\t%T@\n' | sort
}

untracked_check() {
  local f got bad=0 tolerated=0
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    case "$f" in */__pycache__/*) continue ;; esac
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

find_by_sha() {   # $1 file name, $2 sha256 -> path of a copy with that sha256 under $REPO, or stop
  local h
  while IFS= read -r h; do
    [ -n "$h" ] || continue
    if [ "$(sha256sum "$h" | cut -d' ' -f1)" = "$2" ]; then echo "$h"; return; fi
  done < <(find "$REPO" -name "$1" -not -path '*/results/*' -not -path "$EVID/*" -type f 2>/dev/null | sort)
  die "no $1 with the registered sha256 $2 found under $REPO"
}

no_bytecode() {   # $1 folder, $2 when: 91 if bytecode exists, 92 if the scan itself failed (fail closed)
  local hits
  [ -d "$1" ] || { echo "ERROR: bytecode scan $2: $1 is not a folder" >&2; return 92; }
  if ! hits=$(find "$1" \( -name __pycache__ -o -name '*.pyc' -o -name '*.pyo' \) -print); then
    echo "ERROR: bytecode scan $2 could not traverse $1 completely: the scan proves nothing" >&2
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
  [ "$(id -u)" != "0" ] || die "this script refuses to run as root: a read-only snapshot does not stop root from writing"
  echo "executing user: $(id -un) (uid $(id -u)), not root"
  [ -d "$REPO/.git" ] || [ -f "$REPO/.git" ] || die "pinned checkout not found: $REPO"
  head_now=$(git -C "$REPO" rev-parse HEAD)
  [ "$head_now" = "$PIN" ] || die "pinned checkout HEAD is $head_now, expected the pin $PIN"
  br=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
  [ "$br" = "$PIN_BRANCH" ] || die "pinned checkout is on branch $br, expected $PIN_BRANCH"
  git -C "$REPO" diff --quiet HEAD -- "${PINNED_PATHS[@]}" || die "a tracked file under ${PINNED_PATHS[*]} differs from the pin"
  git -C "$REPO" diff --quiet HEAD -- ':(glob)*.py' ':(glob)seqref_mri/*.py' || die "a tracked .py file at the top level of the repository or of seqref_mri/ differs from the pin"
  echo "pinned checkout: HEAD = pin $PIN, branch $br, tracked files under ${PINNED_PATHS[*]} identical to the pin"
  untracked_check
  [ -d "$WT" ] || die "ch2-diag worktree not found: $WT"
  ARCH_COMMIT=$(git -C "$WT" rev-parse HEAD)
  for f in "${FILES[@]}" "$CARD"; do
    if git -C "$WT" cat-file -e "$ARCH_COMMIT:$DIR/$f" 2>/dev/null; then got=$(git -C "$WT" show "$ARCH_COMMIT:$DIR/$f" | sha256sum | cut -d' ' -f1); else got="(not in that commit)"; fi
    [ "$got" = "${SHA[$f]}" ] || die "archived $f at $ARCH_COMMIT has sha256 $got, reviewed ${SHA[$f]} (are the F1 code and run card v0.3 registered in Git?)"
  done
  echo "archive: the ONE ch2-diag commit $ARCH_COMMIT holds the ${#FILES[@]} reviewed code files and the run card, each at its reviewed sha256"
  DUNC_PATH=$(find_by_sha diag_unc_adapter.py "$DUNC_SHA")
  echo "D-UNC adapter: $DUNC_SHA  $DUNC_PATH"
  MANIFEST_PATH=$(find_by_sha manifest_f2.json "$MANIFEST_SHA")
  echo "registered D-UNC manifest: $MANIFEST_SHA  $MANIFEST_PATH"
  [ -x "$PY" ] || die "interpreter not found: $PY"
  vers=$("$PY" -I -B -c "import sys,torch,numpy,h5py;print(sys.version.split()[0],torch.__version__,numpy.__version__,h5py.__version__)")
  echo "interpreter: $PY  python/torch/numpy/h5py = $vers"
  set -- $vers
  [ "$2" = "$WANT_TORCH" ] || die "torch is $2, expected $WANT_TORCH"
  [ "$3" = "$WANT_NUMPY" ] || die "numpy is $3, expected $WANT_NUMPY"
  [ -d "$EVID" ] || die "evidence folder not found: $EVID (it is not created by this script)"
  free_b=$(df -B1 --output=avail "$EVID" | tail -1 | tr -dc '0-9')
  need_b=$((DISK_RESERVE_GIB * GIB))
  [ -n "$free_b" ] || die "could not read the free disk space of $EVID"
  [ "$free_b" -ge "$need_b" ] || die "free disk $free_b bytes is below the reserve $need_b bytes (${DISK_RESERVE_GIB} GiB)"
  mem_b=$(awk '/^MemAvailable:/ {printf "%d", $2*1024}' /proc/meminfo)
  need_m=$((REQ_AVAIL_RAM_GIB * GIB))
  [ -n "$mem_b" ] && [ "$mem_b" -gt 0 ] || die "could not read MemAvailable from /proc/meminfo"
  [ "$mem_b" -ge "$need_m" ] || die "available memory $mem_b bytes is below the requirement $need_m bytes (${REQ_AVAIL_RAM_GIB} GiB)"
  echo "resources: free disk $free_b bytes >= reserve $need_b; available memory $mem_b bytes >= required $need_m"
  echo "PREFLIGHT PASSED"
}

execute() {   # $1 = bind | run
  local kind="$1" tag gate_line verdict_file extra=()
  if [ "$kind" = "bind" ]; then tag="bind"; gate_line="BIND: PASS"; verdict_file="bind_verdict.txt"; extra=(--bind-only)
  else tag="f1"; gate_line="F1: PASS"; verdict_file="f1_verdict.txt"; fi
  preflight
  RUN="$EVID/${tag}_$(date -u +%Y%m%dT%H%M%SZ)"
  [ ! -e "$RUN" ] || die "run folder already exists: $RUN"
  mkdir -p "$RUN/code" "$RUN/card" "$RUN/dunc" "$RUN/input" "$RUN/logs"
  echo "run folder: $RUN"
  RUN_REL=""
  case "$(readlink -f "$RUN")/" in "$(readlink -f "$REPO")"/*) RUN_REL=$(realpath --relative-to="$(readlink -f "$REPO")" "$RUN") ;; esac
  if [ -n "$RUN_REL" ]; then echo "the run folder lies inside the pinned checkout's folder: $RUN_REL is left out of the before/after status, nothing else"; fi
  pinned_state > "$RUN/logs/pinned_before.txt"
  results_state > "$RUN/logs/results_before.txt" || die "the V05LT results folder could not be listed completely"
  for f in "${FILES[@]}"; do
    git -C "$WT" show "$ARCH_COMMIT:$DIR/$f" > "$RUN/code/$f"
    [ "$(sha256sum "$RUN/code/$f" | cut -d' ' -f1)" = "${SHA[$f]}" ] || die "snapshot of $f does not have the reviewed sha256"
  done
  git -C "$WT" show "$ARCH_COMMIT:$DIR/$CARD" > "$RUN/card/$CARD"
  [ "$(sha256sum "$RUN/card/$CARD" | cut -d' ' -f1)" = "${SHA[$CARD]}" ] || die "snapshot of the run card does not have the reviewed sha256"
  cp "$DUNC_PATH" "$RUN/dunc/diag_unc_adapter.py"
  [ "$(sha256sum "$RUN/dunc/diag_unc_adapter.py" | cut -d' ' -f1)" = "$DUNC_SHA" ] || die "snapshot of diag_unc_adapter.py does not have the registered sha256"
  cp "$MANIFEST_PATH" "$RUN/input/manifest_f2.json"
  [ "$(sha256sum "$RUN/input/manifest_f2.json" | cut -d' ' -f1)" = "$MANIFEST_SHA" ] || die "snapshot of manifest_f2.json does not have the registered sha256"
  (cd "$RUN" && sha256sum code/* card/* dunc/* input/*) > "$RUN/logs/snapshot_sha256.txt"
  [ "$(find "$RUN/code" -type f | wc -l)" = "${#FILES[@]}" ] && [ "$(find "$RUN/card" "$RUN/dunc" "$RUN/input" -type f | wc -l)" = "3" ] || die "the snapshot does not hold exactly the reviewed files"
  chmod -R a-w "$RUN/code" "$RUN/card" "$RUN/dunc" "$RUN/input"
  for d in code card dunc input; do must_not_write "$RUN/$d"; done
  no_bytecode "$RUN" "before the run" || exit $?
  { echo "launcher SEQREF-DLAT-F1RUN v0.3, step $kind"; echo "F1 wall ceiling 3600 s (driver); timeout backstop $TIMEOUT_S s (termination allowance, not run time)"; echo "launcher sha256 $(sha256sum "$SELF" | cut -d' ' -f1)"; echo "mode $MODE"; echo "pin $PIN";
    echo "archive commit $ARCH_COMMIT"; echo "interpreter $PY"; echo "versions $vers"; echo "started $(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$RUN/logs/run_context.txt"
  [ "$kind" = "bind" ] || extra=(--dunc-dir "$RUN/dunc")
  set +e
  timeout --signal=TERM --kill-after=60 "$TIMEOUT_S" "$PY" -I -B "$RUN/code/diag_lat_f1.py" --run-card "$RUN/card/$CARD" --run-card-sha256 "${SHA[$CARD]}" \
      --repo "$REPO" --manifest "$RUN/input/manifest_f2.json" --out "$RUN/out" "${extra[@]}" > "$RUN/logs/terminal.log" 2>&1
  rc=$?
  set -e
  echo "exit $rc" >> "$RUN/logs/terminal.log"
  echo "driver finished $(date -u +%Y-%m-%dT%H:%M:%SZ) exit $rc" >> "$RUN/logs/run_context.txt"
  [ "$rc" != "124" ] && [ "$rc" != "137" ] || echo "ERROR: the driver was stopped by the timeout backstop after $TIMEOUT_S s" >&2
  pinned_state > "$RUN/logs/pinned_after.txt"
  results_rc=0
  results_state > "$RUN/logs/results_after.txt" || results_rc=1
  tree_ok=1
  if cmp -s "$RUN/logs/pinned_before.txt" "$RUN/logs/pinned_after.txt"; then echo "pinned checkout: unchanged by the run"
  else echo "ERROR: the pinned checkout changed during the run:" >&2; diff "$RUN/logs/pinned_before.txt" "$RUN/logs/pinned_after.txt" >&2 || true; tree_ok=0; fi
  if [ "$results_rc" = "0" ] && cmp -s "$RUN/logs/results_before.txt" "$RUN/logs/results_after.txt"; then echo "V05LT results folder: unchanged by the run (file list, sizes, modification times)"
  else echo "ERROR: the V05LT results folder changed during the run or could not be listed:" >&2; diff "$RUN/logs/results_before.txt" "$RUN/logs/results_after.txt" >&2 || true; tree_ok=0; fi
  bytecode_rc=0
  no_bytecode "$RUN" "after the run" || bytecode_rc=$?
  snap_ok=1
  (cd "$RUN" && sha256sum -c --quiet "$RUN/logs/snapshot_sha256.txt") || { echo "ERROR: a file of the snapshot changed during the run" >&2; snap_ok=0; }
  grep -E "^(F1|BIND): |ERROR|^Traceback" "$RUN/logs/terminal.log" | tail -40 || true
  drv_ok=0
  if [ "$rc" = "0" ] && grep -qxF "$gate_line" "$RUN/logs/terminal.log" && [ -f "$RUN/out/$verdict_file" ] \
     && [ "$(tail -n 1 "$RUN/out/$verdict_file")" = "$gate_line" ] && [ -f "$RUN/out/outputs_sha256.txt" ] \
     && (cd "$RUN/out" && sha256sum -c --quiet outputs_sha256.txt); then drv_ok=1; fi
  if [ "$kind" = "run" ] && [ "$drv_ok" = "1" ] && [ ! -f "$RUN/out/f1_report.json" ]; then echo "ERROR: f1_report.json is missing" >&2; drv_ok=0; fi
  st() { [ "$1" = "1" ] && echo PASS || echo FAIL; }
  final=0
  [ "$drv_ok" = "1" ] && [ "$snap_ok" = "1" ] || final=1
  [ "$bytecode_rc" = "0" ] || final=$bytecode_rc
  [ "$tree_ok" = "1" ] || final=90
  { echo "driver, step $kind (exit $rc; gate: exit 0, the line '$gate_line', its verdict file and its sha256 manifest): $(st $drv_ok)"
    echo "snapshot unchanged: $(st $snap_ok)"
    echo "pinned checkout and V05LT results folder unchanged: $(st $tree_ok)"
    if [ "$bytecode_rc" = "92" ]; then echo "no bytecode written: FAIL (the scan could not be completed; nothing is proved)"; elif [ "$bytecode_rc" = "0" ]; then echo "no bytecode written: PASS"; else echo "no bytecode written: FAIL"; fi
    if [ "$final" = "0" ]; then echo "LAUNCHER VERDICT ($kind): PASS"; else echo "LAUNCHER VERDICT ($kind): FAIL (exit $final) -- evidence preserved; no retry; return for review"; fi
  } | tee "$RUN/logs/launcher_verdict.txt"
  echo "verdict exit $final" >> "$RUN/logs/run_context.txt"
  if (cd "$RUN" && find . -type f ! -name run_sha256.txt | sort | xargs sha256sum) > "$RUN/run_sha256.txt"; then
    echo "every file of the run folder with its sha256: $RUN/run_sha256.txt"
  else
    echo "ERROR: not every file of the run folder could be hashed; $RUN/run_sha256.txt is incomplete" >&2
    [ "$final" != "0" ] || final=1
  fi
  echo "RESULT exit $final  (0 = PASS; 1 = the driver did not pass its gate, the snapshot changed or the files could not all be hashed; 90 = pinned checkout or results folder changed; 91 = bytecode written; 92 = bytecode scan incomplete)"
  echo "The snapshot folders of $RUN stay read-only (evidence). Send back: this terminal output and the folders $RUN/logs and $RUN/out."
  exit "$final"
}

case "${1:-}" in
  preflight) preflight ;;
  bind)      execute bind ;;
  run)       execute run ;;
  *)         die "usage: bash $0 preflight|bind|run" ;;
esac
