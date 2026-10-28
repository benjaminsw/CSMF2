# SEQREF-MRI-V05LTI12 v0.1 -- I1.2 read-only source checks: V02 holdout NMSE exclusion rule; psnr_per_sample at MSE=0
# LIFETIME: DIAGNOSTIC
set -uo pipefail
cd /home/benjamin/CSMFII || { echo "ERROR NO_REPO"; exit 2; }
PIN=6ffce4e555f90a48daa5fbff8b841adc3dfd7b18
OUT=seqref_mri/results/v05lt/diagnostics/v05lt_i12_inspect.txt
[ -e "$OUT" ] && { echo "ERROR OUT_EXISTS $OUT"; exit 2; }
set -o noclobber
{
echo "V05LTI12 v0.1 $(date -Is) host=$(hostname) script_sha256=$(sha256sum "$0" | cut -d' ' -f1)"
echo "HEAD=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git cat-file -e "$PIN^{commit}" && echo "PIN_PRESENT $PIN" || echo "ERROR PIN_MISSING $PIN"

echo; echo "=== A. files at pin mentioning R_FREE_MIN / exclusion / _nmse (scripts, tdiag, src)"
git grep -n -E "R_FREE_MIN|excluded|_nmse\(|nmse_u|def _nmse" "$PIN" -- 'seqref_mri/scripts/*.py' 'seqref_mri/tdiag/*.py' 'seqref_mri/src/*.py' \
  | grep -v -E "^[^:]+:[^:]+:[0-9]+:\s*#" | head -200

echo; echo "=== B. V02 holdout evaluation entry points at pin (v02*/d3/estimators; holdout split use)"
git grep -n -E "split=.?\"?val|holdout|R_FREE_MIN|excluded" "$PIN" -- 'seqref_mri/scripts/v02*.py' 'seqref_mri/tdiag/d3.py' 'seqref_mri/tdiag/estimators.py' | head -200

echo; echo "=== C. psnr_per_sample source at pin"
git show "$PIN:seqref_mri/src/metrics.py" | awk '/def psnr_per_sample/{f=1} f{print NR": "$0} f&&/^def /&&!/psnr_per_sample/{exit}' | head -80

echo; echo "=== D. tiny_gate._psnr source at pin"
git show "$PIN:seqref_mri/scripts/tiny_gate.py" | awk '/def _psnr/{f=1} f{print NR": "$0} f&&/^def /&&!/_psnr/{exit}' | head -40

echo; echo "=== E. working tree vs pin for the two metric files"
for f in seqref_mri/src/metrics.py seqref_mri/scripts/tiny_gate.py; do
  a=$(git rev-parse "$PIN:$f"); b=$(git hash-object "$f" 2>/dev/null || echo MISSING)
  [ "$a" = "$b" ] && echo "SAME $f $a" || echo "DIFFERS $f pin=$a tree=$b"
done
echo; echo "RESULT DONE (read-only; no code executed beyond git/grep)"
} 2>&1 | tee "$OUT"
