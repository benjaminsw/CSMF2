# SEQREF-DUNC-DISC v0.4 -- read-only V05LT layout + key-name discovery (v0.2 + v0.3 combined; names, sizes, types only)
# LIFETIME: EPHEMERAL
set -o pipefail
cd /home/benjamin/CSMFII/seqref_mri || { echo "ERROR: cd failed" >&2; exit 1; }
R=results/v05lt
[ -d "$R" ] || { echo "ERROR: V05LT results directory not found" >&2; exit 1; }

echo "=== Directory layout ==="
find "$R" -maxdepth 3 -not -name 'val_e*.json' | sort | sed -n '1,150p'
echo "=== Validation-row directories ==="
find "$R" -name 'val_e*.json' -printf '%h\n' | sort | uniq -c
echo "=== Checkpoint files ==="
find "$R" -name '*.pt' -printf '%s %p\n' | sort -k2
echo "=== Training and checkpoint records ==="
grep -rlE 'selected_epoch|best_epoch|stop_epoch|state_sha256|model_sha256' \
    "$R" --include='*.json' --exclude='val_e*.json' 2>/dev/null | sed -n '1,40p'
echo "=== Source-pin evidence candidates ==="
grep -rlE '7455da80' "$R" ../v06lt_evidence \
    --include='*.json' --include='*.txt' --include='*.log' 2>/dev/null | sed -n '1,60p'

echo "=== Key names (no values) ==="
P=$(ls -d "$R"/posteval/*/ 2>/dev/null | sed -n '1p')
V=$(find "$R" -name 'val_e*.json' | sort | sed -n '1p')
python3 -I - "$R/launch/v05lt_pin_record.json" "$R/launch/v05lt_launch_inputs.json" \
  "$R/2026_lam0/final/final_weights.json" "$R/analysis/v05lt_analysis.json" \
  "${P}posteval.json" "$V" <<'EOF'
import json, sys
def walk(o, pre, d):
    if isinstance(o, dict):
        for k in sorted(o)[:60]:
            print(f"{pre}{k}: {type(o[k]).__name__}")
            if d < 2: walk(o[k], pre + "  ", d + 1)
    elif isinstance(o, list) and o:
        print(f"{pre}[list len={len(o)}, item={type(o[0]).__name__}]")
        if d < 2: walk(o[0], pre + "  ", d + 1)
for p in sys.argv[1:]:
    print(f"=== {p}")
    try:
        walk(json.load(open(p)), "  ", 0)
    except Exception as e:
        print(f"ERROR: {p}: {type(e).__name__}: {e}")
EOF
echo "=== Discovery finished ==="
