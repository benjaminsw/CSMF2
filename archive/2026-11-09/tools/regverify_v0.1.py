# SEQREF-MRI-REGVERIFY v0.1 -- compare a LudWiki readback (from the transcript) with a local file
# LIFETIME: EPHEMERAL
import json,hashlib,sys
p="/root/.claude/projects/-home-claude/aae3b5ac-3a04-5a76-befe-d0f05278fca8.jsonl"
aid,ts,f=sys.argv[1:4]
def walk(o):
    if isinstance(o,dict):
        for v in o.values(): yield from walk(v)
    elif isinstance(o,list):
        for v in o: yield from walk(v)
    elif isinstance(o,str): yield o
body=None
for line in open(p):
    if ts not in line: continue
    for s in walk(json.loads(line)):
        if s.lstrip().startswith('{') and f'"id": {aid}' in s and ts in s:
            try: body=json.loads(s)["body"]
            except Exception: pass
loc=open(f).read()
print(aid, ts, "found" if body is not None else "NOT FOUND", body==loc, hashlib.sha256(loc.encode()).hexdigest())
