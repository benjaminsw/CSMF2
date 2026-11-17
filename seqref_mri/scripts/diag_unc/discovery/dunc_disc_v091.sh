# SEQREF-DUNC-DISC v0.9.1 -- read-only pinned-source discovery for the D-UNC real-data adapter (fixture 11) and X2
# LIFETIME: DIAGNOSTIC
#
# Reads tracked Python source only, through read-only Git commands (rev-parse, status, cat-file,
# show, grep, ls-tree). It never imports a module, never opens a checkpoint, MRI file, stored
# row or results file, never decodes, and never switches branch. It writes three new files in
# v06lt_evidence/DIAG_UNC/audit/ and nothing else.
#
# Run:  bash dunc_disc_v091.sh 2>&1 | tee /tmp/dunc_disc_v091_terminal.txt
# Outcomes (nothing is deleted in any case; do not rerun):
#   COMPLETE   (exit 0): the script ran correctly and every required search category produced
#                        candidate lines. It does NOT establish the coefficient order, mask
#                        construction, bank recipe or band-boundary rule: those are confirmed
#                        only by reviewing the lines that were found.
#   INCOMPLETE (exit 3): the script ran correctly, but one or more required categories
#                        produced no candidate line.
#   FAIL / ABORTED (exit 1): an integrity, Git, read, write or verification check failed.
#
# A search with no match is recorded as "0 matching lines" and is not a failure.
# A Git command, read or write that fails aborts the script.

set +e
PIN=7455da8027f33a6cca4dbf72f04e72420e938f45
REPO=/home/benjamin/CSMFII
AUDIT=$REPO/v06lt_evidence/DIAG_UNC/audit
ARCHIVE_REF=ch2-diag
JSH_DIR=seqref_mri/scripts/diag_jsh
SRC="seqref_mri/tdiag/d3.py seqref_mri/src/free_flow_runtime.py seqref_mri/scripts/v05lt_eval.py seqref_mri/scripts/v05lt_metrics.py"

# registered sha256 of the four pinned source files (224 U3, confirmed in U9)
reg_hash() {
  case "$1" in
    seqref_mri/tdiag/d3.py)                echo 73110e7d36a1ebdcb51926c4bbde81f88392f578e1aa860736c509706041245e ;;
    seqref_mri/src/free_flow_runtime.py)   echo 8e171b2c521d77c8ad22b21d02537be160d15166aecb9e0319fd0a25b69823c9 ;;
    seqref_mri/scripts/v05lt_eval.py)      echo 10b4bc65d465e9c6d0a016c2d27977cd9f485ffb851936d9a15ad6a325898554 ;;
    seqref_mri/scripts/v05lt_metrics.py)   echo 87eaa5a4e85b6b70072c1c76fcd3e73ce104c3cdb04ebef973c346ff90b18a74 ;;
    *) echo UNREGISTERED ;;
  esac
}

OUT=""
abort() {
  echo "DISC_ABORT: $*"
  if [ -n "$OUT" ] && [ -f "$OUT" ]; then echo "DISC_ABORT: $* ($(date -u +%Y-%m-%dT%H:%M:%SZ))" >> "$OUT"; fi
  echo "DISC_RESULT: ABORTED (nothing is deleted; do not rerun)"
  exit 1
}

# ---- checked helpers: every Git call and every write is tested ----
head_is_pin() {  # 0 = HEAD is the pin; 1 = it is not; 2 = git failed
  local h; h=$(git rev-parse HEAD) || return 2
  [ "$h" = "$PIN" ]
}
tree_is_clean() {  # 0 = clean; 1 = not clean; 2 = git failed
  local s; s=$(git status --short) || return 2
  [ -z "$s" ]
}
CONTENT=""
load() {  # $1 ref  $2 path -> CONTENT (aborts if the object cannot be read; an empty file is allowed)
  CONTENT=$(git show "$1:$2") || abort "git show failed: $1:$2"
}
nlines() { printf '%s\n' "$1" | grep -c ''; }
emit() { printf '%s\n' "$1" >> "$OUT" || abort "write to summary failed"; }
sec() { emit ""; emit "===== $* ====="; }
BLOCKS=0
SEC_MATCHES=0
kgrep() {  # $1 ref  $2 path  $3 label  $4 pattern ; adds to SEC_MATCHES ; never truncates
  local n m rc
  load "$1" "$2"
  n=$(printf '%s\n' "$CONTENT" | grep -c -E "$4"); rc=$?
  [ "$rc" -le 1 ] || abort "grep failed (exit $rc): $2 | $3"
  emit ""; emit "--- $2 | $3 | $n matching lines ---"
  if [ "$n" -gt 0 ]; then
    m=$(printf '%s\n' "$CONTENT" | grep -n -E -C2 "$4"); rc=$?
    [ "$rc" -eq 0 ] || abort "grep failed (exit $rc): $2 | $3"
    emit "$m"
  fi
  BLOCKS=$((BLOCKS + 1)); SEC_MATCHES=$((SEC_MATCHES + n))
}
tree_grep() {  # $1 ref  $2 options  $3 pattern -> TG (may be empty) ; aborts if git grep fails
  local rc
  TG=$(git grep $2 -I -E "$3" "$1" -- 'seqref_mri/*.py'); rc=$?
  [ "$rc" -le 1 ] || abort "git grep failed (exit $rc): $3"
  TG=$(printf '%s' "$TG" | sed "s/^$1://")
}

# ---- 1. pre-checks: nothing is created before these pass ----
cd "$REPO" || abort "cannot enter $REPO"
head_is_pin;   rc=$?; [ "$rc" = "0" ] || abort "HEAD is not the pin, or git rev-parse failed (code $rc)"
tree_is_clean; rc=$?; [ "$rc" = "0" ] || abort "Git status is not clean, or git status failed (code $rc)"
ARCHIVE_COMMIT=$(git rev-parse --verify --quiet "$ARCHIVE_REF^{commit}") || abort "archive branch $ARCHIVE_REF not found"
[ -d "$AUDIT" ] || abort "audit directory missing"
SRC_LINES_TOTAL=0
for f in $SRC; do
  git cat-file -e "$PIN:$f" || abort "pinned file missing: $f"
  got=$(set -o pipefail; git show "$PIN:$f" | sha256sum | cut -d' ' -f1) || abort "cannot hash $f"
  [ "$got" = "$(reg_hash "$f")" ] || abort "source identity mismatch: $f ($got)"
done
JSH_FILES=$(git ls-tree -r --name-only "$ARCHIVE_COMMIT" -- "$JSH_DIR") || abort "git ls-tree failed for $JSH_DIR"
JSH_PY=$(printf '%s\n' "$JSH_FILES" | grep -E '\.py$')
[ -n "$JSH_PY" ] || abort "no .py file under $JSH_DIR on $ARCHIVE_REF"

TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$AUDIT/dunc_disc_v091_$TS.txt
FULL=$AUDIT/dunc_disc_v091_${TS}_full_source.txt
SUMS=$OUT.sha256
if [ -e "$OUT" ] || [ -e "$FULL" ] || [ -e "$SUMS" ]; then OUT=""; abort "output file already exists"; fi
: > "$OUT" || { OUT=""; abort "cannot create summary file"; }
: > "$FULL" || abort "cannot create full-listing file"

# ---- 2. header and identities ----
SCRIPT_SHA=$(sha256sum "$0" | cut -d' ' -f1); [ -n "$SCRIPT_SHA" ] || abort "cannot hash the script"
emit "SEQREF-DUNC-DISC v0.9.1 | source text only | no import, no checkpoint, no MRI data, no stored rows, no metric value"
emit "start_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
emit "script_sha256: $SCRIPT_SHA"
emit "pin: $PIN"
emit "archive_ref: $ARCHIVE_REF = $ARCHIVE_COMMIT"
emit "note: this output lists candidate source lines found by keyword search. A match is not proof; the coefficient order, mask construction, bank recipe and band-boundary rule are confirmed only by reviewing these lines."

sec "A. pinned source identity (sha256 of git show at the pin; all equal to the registered values)"
for f in $SRC; do
  n=$(set -o pipefail; git show "$PIN:$f" | wc -l) || abort "cannot count lines: $f"
  [ "$n" -gt 0 ] || abort "pinned source file has no line: $f"
  last=$(set -o pipefail; git show "$PIN:$f" | tail -c 1 | od -An -c | tr -d ' ') || abort "cannot read the last byte: $f"
  if [ "$last" = '\n' ]; then endnl=yes; else endnl=no; fi
  SRC_LINES_TOTAL=$((SRC_LINES_TOTAL + n))
  emit "$(reg_hash "$f")  $f  newline_count=$n  ends_with_newline=$endnl"
done

sec "B. structure: import, def and class lines of the four pinned files"
P_STRUCT='^[[:space:]]*(import |from [A-Za-z_.]+ import |def |class |[A-Z][A-Z0-9_]+[[:space:]]*=)'
SEC_MATCHES=0
for f in $SRC; do
  load "$PIN" "$f"
  m=$(printf '%s\n' "$CONTENT" | grep -n -E "$P_STRUCT"); rc=$?
  [ "$rc" -le 1 ] || abort "grep failed (exit $rc): structure of $f"
  n=0; [ -n "$m" ] && n=$(nlines "$m")
  emit ""; emit "--- $f | structure | $n matching lines ---"; [ -n "$m" ] && emit "$m"
  SEC_MATCHES=$((SEC_MATCHES + n))
done
M_B=$SEC_MATCHES

sec "C. coefficient order, acquired/unacquired split and Fourier conventions (four pinned files)"
P_COEF='unacq|acquir|d_free|free_idx|free_mask|6912|13824|9216|flatten|reshape|view_as_real|view_as_complex|fftshift|ifftshift|fft2|ifft2|fftn|ifftn|masked_select|nonzero|argwhere|boolean_mask|permute|transpose|order='
SEC_MATCHES=0; for f in $SRC; do kgrep "$PIN" "$f" "coefficient order" "$P_COEF"; done; M_C=$SEC_MATCHES

sec "D. decode path, conditioning input and sample bank use (four pinned files)"
P_DEC='cond_in|prep\[|def prep|decode|inverse|\.sample|z_diag|Z_DIAG|posterior_mean|r1|R1|zero_latent|z *= *0|torch\.zeros|no_grad|load_state_dict|model_state|weights_final|final_weights|uses'
SEC_MATCHES=0; for f in $SRC; do kgrep "$PIN" "$f" "decode and cond_in" "$P_DEC"; done; M_D=$SEC_MATCHES

sec "E. mask construction (four pinned files)"
P_MASK='mask|center_fraction|accelerat|equispaced|random_mask|RandomMask|EquispacedMask|seed|rng|Generator|PCG64|manual_seed'
SEC_MATCHES=0; for f in $SRC; do kgrep "$PIN" "$f" "mask and seeds" "$P_MASK"; done; M_E=$SEC_MATCHES

sec "F. primary sample bank (X2) and masks: all tracked .py files under seqref_mri at the pin"
P_BANKNAME='Z_DIAG|z_diag'
P_BANKGEN='PCG64|standard_normal|randn|default_rng|SeedSequence|RandomState|manual_seed'
P_BANKCTX='Z_DIAG|z_diag|PCG64|standard_normal|randn|default_rng|SeedSequence|RandomState|manual_seed|13824|float32|dtype|astype|tobytes|byteorder|<f4|sha256|hashlib|seed|shape|128'
emit "--- F1. every line naming the bank or a random generator (file:line:text; nothing filtered out) ---"
tree_grep "$PIN" "-n" "$P_BANKNAME|$P_BANKGEN|13824"
M_F1=0; [ -n "$TG" ] && M_F1=$(nlines "$TG")
emit "$M_F1 matching lines"; [ -n "$TG" ] && emit "$TG"

emit ""; emit "--- F2. candidate files for the bank-generating procedure: files that name the bank AND construct a random generator ---"
tree_grep "$PIN" "-l" "$P_BANKNAME"; FILES_NAME=$TG
tree_grep "$PIN" "-l" "$P_BANKGEN";  FILES_GEN=$TG
CAND=""
for f in $FILES_NAME; do
  if printf '%s\n' "$FILES_GEN" | grep -q -x -F "$f"; then CAND="$CAND $f"; fi
done
N_CAND=0; for f in $CAND; do N_CAND=$((N_CAND + 1)); done
emit "files naming the bank: $( [ -n "$FILES_NAME" ] && nlines "$FILES_NAME" || echo 0 ); files constructing a generator: $( [ -n "$FILES_GEN" ] && nlines "$FILES_GEN" || echo 0 ); candidate files (both): $N_CAND"
if [ "$N_CAND" = "0" ]; then
  emit "NO CANDIDATE FILE FOUND: this search gives no candidate for the bank-generating procedure (X2 cannot be planned from this output)."
else
  for f in $CAND; do emit "candidate: $f"; done
  SEC_MATCHES=0
  for f in $CAND; do kgrep "$PIN" "$f" "bank generation: seed, shape, dtype, byte convention" "$P_BANKCTX"; done
  BLOCKS=$((BLOCKS - N_CAND))   # candidate blocks are counted separately below
fi

emit ""; emit "--- F3. files that define mask functions (file names only) ---"
tree_grep "$PIN" "-l" 'def [A-Za-z_]*mask|class [A-Za-z_]*Mask|center_fraction'
M_F3=0; [ -n "$TG" ] && M_F3=$(nlines "$TG")
emit "$M_F3 files"; [ -n "$TG" ] && emit "$TG"

emit ""; emit "--- F4. modules imported by the four files that live in this repository (import lines) ---"
M_F4=0
for f in $SRC; do
  load "$PIN" "$f"
  m=$(printf '%s\n' "$CONTENT" | grep -n -E '^[[:space:]]*(from|import) +(seqref_mri|src|tdiag|scripts|\.)'); rc=$?
  [ "$rc" -le 1 ] || abort "grep failed (exit $rc): imports of $f"
  if [ -n "$m" ]; then emit "$(printf '%s\n' "$m" | sed "s|^|$f:|")"; M_F4=$((M_F4 + $(nlines "$m"))); fi
done
emit "$M_F4 matching lines"

sec "G. DIAG-JSH code on $ARCHIVE_REF ($ARCHIVE_COMMIT): file list"
emit "$JSH_FILES"

sec "H. DIAG-JSH band rule and boundary handling (.py files under $JSH_DIR that mention a band)"
P_BAND='band|radius|radial|rho|sqrt|hypot|1 */ *3|2 */ *3|0\.333|0\.666|512|2048|4608|18432|fftshift|fftfreq|meshgrid|arange|< |<=|>=|digitize|searchsorted'
SEC_MATCHES=0; N_BANDFILES=0
for f in $JSH_PY; do
  load "$ARCHIVE_COMMIT" "$f"
  printf '%s\n' "$CONTENT" | grep -q -i -E 'band'; rc=$?
  [ "$rc" -le 1 ] || abort "grep failed (exit $rc): $f"
  if [ "$rc" = "0" ]; then kgrep "$ARCHIVE_COMMIT" "$f" "band rule" "$P_BAND"; N_BANDFILES=$((N_BANDFILES + 1)); fi
done
M_H=$SEC_MATCHES
emit ""; emit "files mentioning a band: $N_BANDFILES"

# ---- 3. full numbered listing of the four pinned files (second file; upload only if asked) ----
# Listed straight from git show (not from a shell variable), so trailing blank lines are kept.
printf '%s\n' "SEQREF-DUNC-DISC v0.9.1 | full numbered listing of the four pinned source files at $PIN" >> "$FULL" || abort "write to full listing failed"
for f in $SRC; do
  printf '\n===== %s =====\n' "$f" >> "$FULL" || abort "write to full listing failed: $f"
  ( set -o pipefail; git show "$PIN:$f" | cat -n >> "$FULL" ) || abort "listing failed: $f"
done
FULL_EXPECT=$((1 + 8 + SRC_LINES_TOTAL))      # newlines: header + (separator + title) per file + every source newline
FULL_GOT=$(wc -l < "$FULL") || abort "cannot count the full listing"
# each listed file, with its line numbers removed, must reproduce the pinned source
# (compared after giving both a final newline if the source has none)
LIST_OK=1
for f in $SRC; do
  ref=$(set -o pipefail; git show "$PIN:$f" | sed -e '$a\' | sha256sum | cut -d' ' -f1) || abort "cannot hash $f for the listing check"
  got=$(set -o pipefail; awk -v t="===== $f =====" '$0==t{on=1;next} /^=====/{on=0} on && /^ *[0-9]+\t/' "$FULL" | sed -E 's/^ *[0-9]+\t//' | sha256sum | cut -d' ' -f1) || abort "cannot re-read the listing of $f"
  [ "$ref" = "$got" ] || LIST_OK=0
done

# ---- 4. findings and post-checks ----
sec "Y. candidate evidence per category (counts of matching lines; a match is not proof; 0 is recorded, not a failure)"
emit "B structure lines: $M_B"
emit "C coefficient-order lines: $M_C"
emit "D decode and cond_in lines: $M_D"
emit "E mask and seed lines: $M_E"
emit "F1 bank or generator lines in the pinned tree: $M_F1"
emit "F2 candidate files for the bank-generating procedure: $N_CAND"
emit "F3 files defining mask functions: $M_F3"
emit "H DIAG-JSH files mentioning a band: $N_BANDFILES ; band-rule lines: $M_H"

OK=1
chk() { if eval "$2"; then emit "PASS  $1"; else emit "FAIL  $1"; OK=0; fi; }
sec "Z. post-checks"
chk "HEAD is the pin (git succeeded)"                 'head_is_pin'
chk "Git status clean (git succeeded)"                'tree_is_clean'
chk "sections A to H, Y and Z present"                '[ "$(grep -c "^===== [A-HYZ]\. " "$OUT")" = "10" ]'
chk "search blocks written: 12 + band files ($((12 + N_BANDFILES)))"  '[ "$BLOCKS" = "$((12 + N_BANDFILES))" ]'
chk "full listing complete ($FULL_EXPECT newlines expected, $FULL_GOT found)"  '[ "$FULL_GOT" = "$FULL_EXPECT" ]'
chk "full listing reproduces the four pinned files"   '[ "$LIST_OK" = "1" ]'
emit "end_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
emit "END OF SUMMARY"

# evidence hashes: generation must succeed, hold exactly two entries and verify
( cd "$AUDIT" && sha256sum "$(basename "$OUT")" "$(basename "$FULL")" > "$(basename "$SUMS")" ); rc=$?
if [ "$rc" != "0" ]; then echo "FAIL  sha256 file generation (exit $rc)"; OK=0; fi
if [ "$(grep -c '' "$SUMS" 2>/dev/null)" != "2" ]; then echo "FAIL  sha256 file does not hold exactly two entries"; OK=0; fi
( cd "$AUDIT" && sha256sum -c --quiet "$(basename "$SUMS")" ); rc=$?
if [ "$rc" != "0" ]; then echo "FAIL  sha256 verification of the two outputs (exit $rc)"; OK=0; fi
[ "$(tail -n 1 "$OUT")" = "END OF SUMMARY" ] || { echo "FAIL  summary does not end with its end marker"; OK=0; }

echo "summary:      $OUT ($(grep -c '' "$OUT") lines)"
echo "full listing: $FULL ($FULL_GOT lines)"
cat "$SUMS"
sed -n '/^===== Y\. /,$p' "$OUT"
MISSING=0
if [ "$N_CAND" = "0" ]; then echo "DISC_FINDING: no candidate file for the bank-generating procedure (see F2)"; MISSING=1; fi
if [ "$M_C" = "0" ]; then echo "DISC_FINDING: no coefficient-order line matched (see C)"; MISSING=1; fi
if [ "$M_D" = "0" ]; then echo "DISC_FINDING: no decode or cond_in line matched (see D)"; MISSING=1; fi
if [ "$M_E" = "0" ]; then echo "DISC_FINDING: no mask line matched (see E)"; MISSING=1; fi
if [ "$N_BANDFILES" = "0" ]; then echo "DISC_FINDING: no DIAG-JSH file mentions a band (see H)"; MISSING=1; fi
if [ "$OK" != "1" ]; then
  echo "DISC_RESULT: FAIL (nothing is deleted; do not rerun)"
  exit 1
elif [ "$MISSING" = "1" ]; then
  echo "DISC_RESULT: INCOMPLETE (the script ran correctly, but one or more required categories produced no candidate line; do not rerun)"
  exit 3
else
  echo "DISC_RESULT: COMPLETE (the script ran correctly and every required category produced candidate lines; the lines still have to be reviewed)"
  exit 0
fi
