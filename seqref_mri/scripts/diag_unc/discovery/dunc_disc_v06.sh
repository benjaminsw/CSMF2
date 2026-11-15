# SEQREF-DUNC-DISC v0.6 -- ledger and posteval/analysis structure (key names, types, event names, epochs; no metric values)
# LIFETIME: EPHEMERAL
set -o pipefail
cd /home/benjamin/CSMFII/seqref_mri || { echo "ERROR: cd failed" >&2; exit 1; }
python3 -I - <<'EOF'
import json, glob, os
R = "results/v05lt"
SAFE = ("seq", "event", "type", "kind", "epoch", "schema", "status")
def walk(o, pre, d, maxd):
    if isinstance(o, dict):
        for k in sorted(o)[:40]:
            print("%s%s: %s" % (pre, k, type(o[k]).__name__))
            if d < maxd: walk(o[k], pre + "  ", d + 1, maxd)
        if len(o) > 40: print("%s... (%d keys)" % (pre, len(o)))
    elif isinstance(o, list) and o:
        print("%s[list len=%d, item=%s]" % (pre, len(o), type(o[0]).__name__))
        if d < maxd: walk(o[0], pre + "  ", d + 1, maxd)
def safe(o):
    out = {k: o[k] for k in SAFE if k in o and isinstance(o[k], (str, int))}
    b = o.get("body")
    if isinstance(b, dict):
        out.update({"body." + k: b[k] for k in SAFE if k in b and isinstance(b[k], (str, int))})
        if isinstance(b.get("retained"), dict): out["retained"] = sorted(b["retained"])
    return out
try:
    p = R + "/2026_lam0/ledger.jsonl"; seen = set()
    lines = open(p).read().splitlines(); print("=== %s lines=%d" % (p, len(lines)))
    for i, line in enumerate(lines, 1):
        o = json.loads(line); print(i, safe(o))
        sig = (o.get("event") or o.get("type") or o.get("kind"), tuple(sorted(o)))
        if sig not in seen:
            seen.add(sig); print("  --- structure of line %d" % i); walk(o, "    ", 0, 3)
    for q in sorted(glob.glob(R + "/*_lam*/ledger.jsonl")):
        print("ledger", q, "lines=%d" % sum(1 for _ in open(q)), "sidecar=%s" % os.path.isfile(q + ".sha256"))
except Exception as e:
    print("ERROR ledger: %s: %s" % (type(e).__name__, e))
try:
    q = sorted(glob.glob(R + "/posteval/*/posteval.json"))[0]; j = json.load(open(q))
    print("=== posteval midpoint structure"); walk(j["midpoint"], "  ", 0, 2)
    print("decodes:", j["midpoint"].get("decodes"), "population.n:", j["population"].get("n"),
          "scope:", j["population"].get("scope"), "prefixes:", j.get("prefixes"), "schema:", j.get("schema"))
    print("sidecar beside posteval.json:", os.path.isfile(q + ".sha256"), "| dir files:", sorted(os.listdir(os.path.dirname(q)))[:20])
    a = json.load(open(R + "/analysis/v05lt_analysis.json"))
    keys = sorted(a["analysis"]["caps"]); print("=== analysis caps keys: %d" % len(keys))
    print("r1|mid keys:", [k for k in keys if k.endswith("|r1|mid")])
    print("=== one caps entry"); walk(a["analysis"]["caps"][keys[0]], "  ", 0, 1)
    print("=== stream"); walk(a["analysis"]["stream"], "  ", 0, 2)
    ids = a["inputs"]["identities"]
    print("identities: posteval=%d ledger=%d final=%d" % (sum("/posteval/" in k for k in ids),
          sum(k.endswith("ledger.jsonl") for k in ids), sum("/final/" in k for k in ids)))
    print("inputs keys:", sorted(a["inputs"]))
except Exception as e:
    print("ERROR posteval/analysis: %s: %s" % (type(e).__name__, e))
EOF
echo "=== Discovery finished ==="
