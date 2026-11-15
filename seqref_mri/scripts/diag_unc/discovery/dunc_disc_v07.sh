# SEQREF-DUNC-DISC v0.7 -- ledger hash-chain semantics (match counts only; no ledger values)
# LIFETIME: EPHEMERAL
set -o pipefail
cd /home/benjamin/CSMFII/seqref_mri || { echo "ERROR: cd failed" >&2; exit 1; }
python3 -I - <<'EOF'
import json, glob, hashlib, sys
R = "results/v05lt"; err = 0
h = lambda b: hashlib.sha256(b).hexdigest()
def canon(o, sep): return json.dumps(o, sort_keys=True, separators=sep).encode()
for p in sorted(glob.glob(R + "/*_lam*/ledger.jsonl")):
    try:
        raw = open(p, "rb").read().split(b"\n")
        raw = [x for x in raw if x.strip()]; L = [json.loads(x) for x in raw]; n = len(L)
        print("===", p, "lines=%d" % n)
        print("  schema all ledger/1:", all(o.get("schema") == "seqref-v05lt-ledger/1" for o in L),
              "| seq contiguous:", [o.get("seq") for o in L] == list(range(n)),
              "| first:", L[0].get("kind"), "| last:", L[-1].get("kind"))
        print("  line0 prev: type=%s len=%s | self len=%s" % (type(L[0].get("prev")).__name__,
              len(L[0]["prev"]) if isinstance(L[0].get("prev"), str) else None, len(str(L[0].get("self")))))
        c = {}
        def t(name, ok): c[name] = c.get(name, 0) + bool(ok)
        for i in range(n):
            o = L[i]; body = {k: v for k, v in o.items() if k != "self"}
            for nm, sep in (("compact", (",", ":")), ("default", (", ", ": "))):
                t("self == sha256(canonical %s json without self)" % nm, o.get("self") == h(canon(body, sep)))
            if i:
                t("prev == previous self", o.get("prev") == L[i-1].get("self"))
                t("prev == sha256(previous raw line)", o.get("prev") == h(raw[i-1]))
                t("prev == sha256(previous raw line + newline)", o.get("prev") == h(raw[i-1] + b"\n"))
                for nm, sep in (("compact", (",", ":")), ("default", (", ", ": "))):
                    t("prev == sha256(canonical %s previous object)" % nm, o.get("prev") == h(canon(L[i-1], sep)))
        for k, v in c.items(): print("  %-58s %d/%d" % (k, v, n if k.startswith("self") else n - 1))
        fw = json.load(open(p.replace("ledger.jsonl", "final/final_weights.json")))["source"]
        for key, seqk in (("commit_line_sha256", "commit_seq"), ("run_end_line_sha256", "run_end_seq")):
            i = fw.get(seqk); v = fw.get(key)
            print("  final_weights %s (seq %s): == self %s | == sha256(raw line) %s | == sha256(raw+newline) %s" % (
                key, i, v == L[i].get("self"), v == h(raw[i]), v == h(raw[i] + b"\n")))
    except Exception as e:
        err = 1; print("ERROR %s: %s: %s" % (p, type(e).__name__, e))
sys.exit(err)
EOF
rc=$?; echo "=== Discovery finished rc=$rc ==="; exit $rc
