# SEQREF-MRI-V05LTRELOC v0.2 -- copy registered V05LT inputs unchanged to results/v05lt/inputs (194 AM2-D)
# LIFETIME: KEEP
# Changelog v0.2: explicit file list; all sources, hashes, destination and log checked BEFORE any copy;
#   copy into a new staging dir, verify, then atomic rename to inputs/; log never overwritten;
#   bank manifest hashes verified; script kept as provenance (sha256 recorded in the log).
set -euo pipefail
ROOT=/home/benjamin/CSMFII/seqref_mri
R=$ROOT/results/v05lt
DEST=$R/inputs
LOG=$R/diagnostics/v05lt_inputs_reloc.log
MAN=/home/benjamin/v05lt_manifests
BNK=/home/benjamin/v05lt_banks
STAGE=$R/.inputs_staging_$$

err() { echo "ERROR $*" >&2; exit 2; }

# --- 0. log and destination must not exist (checked before anything is written)
mkdir -p "$R/diagnostics"
[ -e "$LOG" ]   && err "LOG_EXISTS $LOG"
[ -e "$DEST" ]  && err "DEST_EXISTS $DEST"
[ -e "$STAGE" ] && err "STAGE_EXISTS $STAGE"
set -o noclobber; : > "$LOG"; set +o noclobber
exec > >(tee -a "$LOG") 2>&1

echo "V05LTRELOC v0.2 start $(date -Is) host=$(hostname)"
echo "script_sha256 $(sha256sum "$0" | cut -d' ' -f1) $0"

# --- 1. explicit registered inputs: <expected sha256 or '-'> <source path>
#     '-' = not a registered identity (sidecars, run metadata): copied and checked equal to the original only
LIST=(
"f7d21fb97e030a2e8227233cd5d1f171a120b765060ae83cafc728721428cf5d $MAN/v05lt_split_manifest.json"
"b10954f86cd45db607c956c45664abb7fe373254ca6a15f5cf518ae6668082ec $MAN/v05lt_case_manifest.json"
"- $MAN/v05lt_split_manifest.json.sha256"
"- $MAN/v05lt_case_manifest.json.sha256"
"- $MAN/v05lt_manifests_run.json"
"41987536056f002c8bad1e5b24245d071d19301122fd0e1b1099c09449f9ed31 $BNK/v05lt_bank_20261023.npy"
"7e13f9f94ed61d5568e4dab2e92a887fdcebae6fb986fe96cd10eb88c882c4c3 $BNK/v05lt_bank_20261024.npy"
"5edcacb8519cc36965f202c7d47f6d6c72da1eb9a3ce2a54a211092f5b506ccc $BNK/v05lt_bank_20261023_manifest.json"
"bc9e05106c0e21bd8628be927064ad5b46f0051fb5dc5eead770698095372ff3 $BNK/v05lt_bank_20261024_manifest.json"
)

# --- 2. preflight: every source exists, registered hashes match, no duplicate basenames
echo "--- preflight (no copy yet)"
fail=0; declare -A seen
for e in "${LIST[@]}"; do
  set -- $e; exp=$1; src=$2; b=$(basename "$src")
  [ -n "${seen[$b]:-}" ] && { echo "ERROR DUP_BASENAME $b"; fail=1; }; seen[$b]=1
  if [ ! -f "$src" ]; then echo "ERROR SOURCE_MISSING $src"; fail=1; continue; fi
  got=$(sha256sum "$src" | cut -d' ' -f1)
  if [ "$exp" != "-" ] && [ "$got" != "$exp" ]; then echo "ERROR REG_MISMATCH $b got=$got exp=$exp"; fail=1
  else echo "PRE_OK $got $b ${exp/-/(unregistered)}"; fi
done
if [ $fail -ne 0 ]; then
  echo "--- source listings for diagnosis"; ls -la "$MAN" "$BNK" || true
  err "PREFLIGHT_FAILED (nothing copied)"
fi

# --- 3. copy into a fresh staging dir, verify each copy against its original
mkdir "$STAGE"
for e in "${LIST[@]}"; do
  set -- $e; src=$2; b=$(basename "$src")
  cp -p --no-clobber "$src" "$STAGE/$b"
  a=$(sha256sum "$src" | cut -d' ' -f1); c=$(sha256sum "$STAGE/$b" | cut -d' ' -f1)
  [ "$a" = "$c" ] || err "COPY_MISMATCH $b (staging left at $STAGE for inspection)"
  echo "COPY_OK $c $b"
done
n=$(find "$STAGE" -type f | wc -l)
[ "$n" -eq "${#LIST[@]}" ] || err "FILE_COUNT $n != ${#LIST[@]}"

# --- 4. atomic publish: staging -> inputs (same filesystem); inputs must still be absent
[ -e "$DEST" ] && err "DEST_APPEARED $DEST (staging left at $STAGE)"
mv -T "$STAGE" "$DEST"
chmod a-w "$DEST"/*

# --- 5. originals retained
for e in "${LIST[@]}"; do set -- $e; [ -f "$2" ] || err "ORIGINAL_GONE $2"; done
echo "--- inputs/"; ls -l "$DEST"
echo "RESULT PASS files=${#LIST[@]} $(date -Is)"
