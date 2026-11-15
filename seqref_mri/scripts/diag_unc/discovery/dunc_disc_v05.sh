# SEQREF-DUNC-DISC v0.5 -- manifest structure, bank registration, role/sidecar formats (identifiers only; no metric values)
# LIFETIME: EPHEMERAL
set -o pipefail
cd /home/benjamin/CSMFII/seqref_mri || { echo "ERROR: cd failed" >&2; exit 1; }
echo "=== git ==="; git --no-optional-locks rev-parse --show-toplevel HEAD
echo "=== inputs dir ==="; find results/v05lt/inputs -maxdepth 2 -type f -printf '%s %p\n' | sort -k2
python3 -I - <<'EOF'
import json, os, glob
R = "results/v05lt"
BANK = "f8a6a34b6b474286359e4c99341de256b62bc0dff1e6bc8cf52e72f0461dc8d6"
def walk(o, pre, d, maxd=3):
    if isinstance(o, dict):
        for k in sorted(o)[:25]:
            v = o[k]
            ex = " e.g. %r" % v[:80] if isinstance(v, str) and d > 0 else ""
            print("%s%s: %s%s" % (pre, k, type(v).__name__, ex))
            if d < maxd: walk(v, pre + "  ", d + 1, maxd)
        if len(o) > 25: print("%s... (%d keys)" % (pre, len(o)))
    elif isinstance(o, list) and o:
        ex = " e.g. %r" % o[0][:80] if isinstance(o[0], str) else ""
        print("%s[list len=%d, item=%s%s]" % (pre, len(o), type(o[0]).__name__, ex))
        if d < maxd: walk(o[0], pre + "  ", d + 1, maxd)
def find(o, path, hits):
    if isinstance(o, dict):
        for k, v in o.items(): find(v, path + "/" + str(k), hits)
    elif isinstance(o, list):
        for i, v in enumerate(o[:2000]): find(v, path + "[]", hits)
    elif o == BANK: hits.add(path)
try:
    li = json.load(open(R + "/launch/v05lt_launch_inputs.json"))
    for name, ent in sorted(li["inputs"].items()):
        p = ent.get("path"); print("=== input %s path=%r" % (name, p))
        for c in (p, os.path.join("..", p), os.path.join(R, p)):
            if p and os.path.isfile(c) and c.endswith(".json") and name in ("case_manifest", "split_manifest", "eligibility"):
                print("  resolved=%s" % c); walk(json.load(open(c)), "  ", 0); break
except Exception as e:
    print("ERROR launch inputs: %s: %s" % (type(e).__name__, e))
print("=== records containing the primary-bank hash (key paths only) ===")
for p in sorted(glob.glob(R + "/launch/*.json") + glob.glob(R + "/preflight/*.json") + glob.glob(R + "/inputs/*.json")
                + glob.glob(R + "/posteval/*/posteval.json")[:1] + glob.glob(R + "/epoch0/*/eval/*.json")[:1]):
    try:
        hits = set(); find(json.load(open(p)), "", hits)
        if hits: print(p, sorted(hits))
    except Exception as e:
        print("ERROR %s: %s: %s" % (p, type(e).__name__, e))
print("=== posteval roles ===")
for p in sorted(glob.glob(R + "/posteval/*/posteval.json")):
    try:
        j = json.load(open(p)); c = j["checkpoint"]
        print(c.get("arm"), c.get("seed"), c.get("epoch"), c.get("file_name"), c.get("roles"),
              [(u.get("arm"), u.get("seed"), u.get("role")) for u in j.get("uses", [])],
              "dir==state" if os.path.basename(os.path.dirname(p)) == c.get("model_state_sha256") else "dir!=state")
    except Exception as e:
        print("ERROR %s: %s: %s" % (p, type(e).__name__, e))
print("=== formats ===")
try:
    print("sidecar:", repr(open(R + "/2026_lam0/ckpt/weights_e003.pt.sha256").read()[:160]))
    fw = json.load(open(R + "/2026_lam0/final/final_weights.json"))
    print("final_weights:", fw.get("weights_file"), fw.get("epoch"), fw.get("status"), fw.get("schema"))
    v = json.load(open(R + "/2026_lam0/ckpt/val_e001.json")); r = v["rows"][0]
    print("val row ids:", repr(r["volume_id"]), repr(r["file"]), r["slice_index"], "schema=%r" % v["schema"], "arm=%r" % v["arm"])
    a = json.load(open(R + "/analysis/v05lt_analysis.json"))
    k = sorted(a["inputs"]["identities"])[0]; print("identity value length:", len(a["inputs"]["identities"][k]))
    print("run_info keys:"); walk(a["inputs"]["run_info"], "  ", 0, 1)
    pr = json.load(open(R + "/launch/v05lt_pin_record.json"))
    print("pin: identity_method=%r n=%d" % (pr.get("identity_method"), len(pr["source_identity"])))
    for s in ("seqref_mri/tdiag/d3.py", "seqref_mri/src/free_flow_runtime.py", "seqref_mri/scripts/v05lt_eval.py", "seqref_mri/scripts/v05lt_metrics.py"):
        print("  in pin record:", s, s in pr["source_identity"])
except Exception as e:
    print("ERROR formats: %s: %s" % (type(e).__name__, e))
EOF
echo "=== Discovery finished ==="
