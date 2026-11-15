# SEQREF-DUNC-AUD2 v0.1.1 -- dunc_audit2.py: read-only ledger binding (B5) and R1-readout reconciliation (step P) for DIAG-UNC
# LIFETIME: DIAGNOSTIC
"""Supplementary read-only audit after SEQREF-DUNC-AUD v0.4.2 (log dunc_audit_20261115T115114Z.log).
It does not replace or alter that log.

B5L  Tier 2 file<->state binding from the training ledger (schema seqref-v05lt-ledger/1, seen in
     DISC v0.6). Rule: the commit record whose body.epoch == NN binds file F to state S iff
       body.model_state_sha256 == S, body.retained["milestone_0NN"].sha256 == sha256(F),
       "milestone_0NN" is in body.roles_this_epoch, the key milestone_0NN is absent from the
       previous commit record (whatever its hash) and its entry is unchanged in every later one.
     (Rolling labels best_psnr / best_nll, used only when testing the rule on known bindings,
     exist in earlier commits by design; for them the previous entry must have a different hash.)
     The rule is first tested on the bindings already registered by verified posteval records;
     if it fails on any of them it is not used and the Tier 2 bindings stay UNRESOLVED.
P    The existing registered V05LT R1 validation readout: role coverage, analysis registration,
     midpoint population, primary-bank decode count, launch-input, mask and eligibility bindings.
     P1b: whether the analysis document itself ties each <arm>|<seed>|<role> entry to its
     posteval record (a model-state, posteval-file or checkpoint-file hash stored under that
     entry). Own record's hash -> CONFIRMED; only another record's hash -> FAIL; none ->
     UNRESOLVED (the link then rests on the registered inputs and the pinned analysis code).

Standard library only. No torch, numpy, checkpoint loading, MRI data or model evaluation.
Ledger losses, validation summaries and selection values, and posteval r0/r1 payloads, are never
printed, compared or aggregated; payloads are tested for presence only.
Writes exactly two new files (log + .sha256). No fallback, placeholder, mock or dummy path:
anything not established from a registered record is UNRESOLVED.

Outcome (worst of B5L, P, INT): exit 0 PASS, 3 INCOMPLETE, 1 FAIL, 2 FAIL (audit integrity).

Usage (host):
  python3 -I -B dunc_audit2.py --repo /home/benjamin/CSMFII

Changelog
- v0.1.1: binding rule rejects a milestone key present in the previous commit whatever its hash;
  new check P1b for a per-entry analysis-to-readout source identity (absent -> UNRESOLVED,
  pointing at another record -> FAIL); extra posteval directories not registered as analysis
  inputs are reported.
- v0.1: first reviewed version (not run on the host).
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback

VERSION = "SEQREF-DUNC-AUD2 v0.1.1"
PIN = "7455da8027f33a6cca4dbf72f04e72420e938f45"
Z_DIAG = "f8a6a34b6b474286359e4c99341de256b62bc0dff1e6bc8cf52e72f0461dc8d6"
ANALYSIS_SHA = "5b267d3aa7ccb48bf3d7ccbfa2e2497d7f02df845aff8786c0fbe9b9dff2afa2"   # 196, G-D4
LEDGER_SCHEMA, POSTEVAL_SCHEMA = "seqref-v05lt-ledger/1", "seqref-v05lt-posteval/1"
N_VOLUMES, N_SAMPLES = 97, 128
RUNS = [(s, a) for a in ("lam0", "lam2") for s in (2026, 2027, 2028)]
ROLES_T1 = ("epoch3", "selected", "final")
EPOCHS_T2 = (10, 30)
EXPECT_SELECTED = {("lam0", 2026): 9, ("lam0", 2027): 10, ("lam0", 2028): 13,
                   ("lam2", 2026): 55, ("lam2", 2027): 48, ("lam2", 2028): 55}
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
ITEMS = ("B5L", "P", "INT")
GOOD, BAD, OPEN = ("PASS", "CONFIRMED"), ("FAIL", "ERROR"), ("UNRESOLVED",)

TALLY = {i: {} for i in ITEMS}
LOG = None
SNAP = {}


def rec(step, item, status, msg):
    if item in TALLY:
        TALLY[item][status] = TALLY[item].get(status, 0) + 1
    line = "[%s] %-3s %-10s %s" % (step, item, status, msg)
    LOG.write(line + "\n")
    LOG.flush()
    print(line)


def tri(step, item, computed, registered, msg):
    rec(step, item, "UNRESOLVED" if registered is None else ("PASS" if computed == registered else "FAIL"), msg)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snap(path):
    if path not in SNAP:
        st = os.stat(path)
        SNAP[path] = (sha_file(path), st.st_size, st.st_mtime_ns)
    return SNAP[path][0]


def load_json(path):
    snap(path)
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def sidecar(path):
    sc = path + ".sha256"
    if not os.path.isfile(sc):
        return None
    snap(sc)
    with open(sc, "r", encoding="utf-8", errors="replace") as f:
        m = HEX64.search(f.read())
    return m.group(0) if m else None


def git(repo, *args):
    p = subprocess.run(["git", "--no-optional-locks", "-C", repo] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("git %s: %s" % (" ".join(args), p.stderr.decode(errors="replace").strip()))
    return p.stdout.decode(errors="replace").strip()


def resolve(repo, R, raw):
    raw = str(raw)
    cands = [raw] if os.path.isabs(raw) else [os.path.join(b, raw) for b in (repo, os.path.join(repo, "seqref_mri"), R)]
    hits = sorted(set(os.path.realpath(c) for c in cands if os.path.isfile(c)))
    return hits[0] if len(hits) == 1 else None


def step(name, item, fn, *a):
    try:
        return fn(*a)
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        rec(name, item, "ERROR", "%s: %s (line %d in %s)" % (type(e).__name__, e, tb.lineno, tb.name))
        return None


# ---------------------------------------------------------------- S0
def s0(repo, R, state):
    top = git(repo, "rev-parse", "--show-toplevel")
    ok = os.path.realpath(top) == os.path.realpath(repo)
    rec("S0", "INT", "PASS" if ok else "FAIL", "Git toplevel %s the --repo argument" % ("==" if ok else "!="))
    if not ok:
        raise RuntimeError("repository root not verified")
    head = git(repo, "rev-parse", "HEAD")
    rec("S0", "INT", "INFO", "HEAD=%s (%s the pin)" % (head, "==" if head == PIN else "!="))
    state["git"] = (head, hashlib.sha256(git(repo, "status", "--porcelain").encode()).hexdigest())


def s0_records(repo, R, state):
    ap = os.path.join(R, "analysis", "v05lt_analysis.json")
    ana = load_json(ap)
    for item in ("B5L", "P"):
        tri("S0", item, SNAP[ap][0], ANALYSIS_SHA, "analysis document sha256 == value registered in 196 (G-D4)")
        tri("S0", item, SNAP[ap][0], sidecar(ap), "analysis document sha256 == its sidecar")
    state["ana"], state["ident"] = ana, ana.get("inputs", {}).get("identities", {})
    lp = os.path.join(R, "launch", "v05lt_launch_inputs.json")
    state["li"], state["li_sha"] = load_json(lp), SNAP[lp][0]
    vs = {}
    for seed, arm in RUNS:
        d = os.path.join(R, "%d_%s" % (seed, arm), "ckpt")
        for f in sorted(os.listdir(d)):
            if re.fullmatch(r"val_e\d{3}\.json", f):
                st = load_json(os.path.join(d, f)).get("model_state_sha256")   # header field only
                if isinstance(st, str) and HEX64.fullmatch(st):
                    vs[(arm, seed, int(f[5:8]))] = st
    state["val_state"] = vs
    recs = []
    pe_root = os.path.join(R, "posteval")
    for d in sorted(os.listdir(pe_root)):
        p = os.path.join(pe_root, d, "posteval.json")
        if os.path.isfile(p):
            j = load_json(p)
            recs.append((d, p, j))
    state["posteval"] = recs


# ---------------------------------------------------------------- ledger
def load_ledger(R, arm, seed, state):
    """Returns {epoch: commit body} after integrity and structure checks, or None."""
    p = os.path.join(R, "%d_%s" % (seed, arm), "ledger.jsonl")
    tag = "(%s,%d)" % (arm, seed)
    if not os.path.isfile(p):
        rec("L1", "B5L", "UNRESOLVED", "%s ledger.jsonl not found" % tag)
        return None
    reg = state["ident"].get(os.path.abspath(p))
    tri("L1", "B5L", snap(p), reg, "%s ledger.jsonl sha256 == analysis.identities registration" % tag)
    if reg is None or reg != SNAP[p][0]:
        return None
    with open(p, "r", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f.read().splitlines() if x.strip()]
    ok = (len(lines) >= 3 and all(l.get("schema") == LEDGER_SCHEMA for l in lines)
          and [l.get("seq") for l in lines] == list(range(len(lines)))
          and all(lines[i].get("prev") == lines[i - 1].get("self") for i in range(1, len(lines)))
          and lines[0].get("kind") == "run_start" and lines[-1].get("kind") == "run_end")
    rec("L1", "B5L", "PASS" if ok else "FAIL",
        "%s %d lines: schema, seq contiguous, prev == previous self, run_start first, run_end last" % (tag, len(lines)))
    commits = [l for l in lines if l.get("kind") == "commit"]
    epochs = [c.get("body", {}).get("epoch") for c in commits]
    n_val = sum(1 for (a, s, _e) in state["val_state"] if (a, s) == (arm, seed))
    ok2 = bool(commits) and epochs == list(range(1, len(commits) + 1)) and len(commits) == n_val
    rec("L1", "B5L", "PASS" if ok2 else "FAIL",
        "%s commit records: one per epoch 1..%d, equal to the number of validation-row files (%d)" % (tag, len(commits), n_val))
    if not (ok and ok2):
        return None
    by_epoch = {c["body"]["epoch"]: c for c in commits}
    bad = sum(1 for e, c in by_epoch.items() if c["body"].get("model_state_sha256") != state["val_state"].get((arm, seed, e)))
    rec("L1", "B5L", "PASS" if bad == 0 else "FAIL",
        "%s commit model_state_sha256 == val_e<NNN> model_state_sha256 for every epoch (%d differ)" % (tag, bad))
    sel = commits[-1]["body"].get("selection", {}).get("best_psnr", {}).get("epoch")   # epoch only; value never read
    rec("L1", "B5L", "PASS" if sel == EXPECT_SELECTED[(arm, seed)] else "FAIL",
        "%s last commit selection.best_psnr.epoch == selected epoch quoted in 223 (%s)" % (tag, EXPECT_SELECTED[(arm, seed)]))
    return by_epoch if bad == 0 else None


def bound(by_epoch, epoch, key, file_sha, state_sha):
    """Apply the binding rule. Returns (True, detail) or (False, reason)."""
    c = by_epoch.get(epoch)
    if c is None:
        return False, "no commit record for epoch %s" % epoch
    b = c["body"]
    ent = (b.get("retained") or {}).get(key)
    if not isinstance(ent, dict):
        return False, "retained[%s] absent in the epoch-%d commit" % (key, epoch)
    if ent.get("sha256") != file_sha:
        return False, "retained[%s].sha256 differs from the file sha256" % key
    if b.get("model_state_sha256") != state_sha:
        return False, "commit model_state_sha256 differs from the registered state"
    if key not in (b.get("roles_this_epoch") or []):
        return False, "%s not in roles_this_epoch %s" % (key, b.get("roles_this_epoch"))
    prev_ret = (by_epoch[epoch - 1]["body"].get("retained") or {}) if (epoch - 1) in by_epoch else {}
    if key.startswith("milestone_"):
        if key in prev_ret:
            return False, "milestone entry already existed in the previous commit"
    elif isinstance(prev_ret.get(key), dict) and prev_ret[key].get("sha256") == file_sha:
        return False, "the same rolling-label entry already existed in the previous commit"
    later = [e for e in by_epoch if e > epoch and key.startswith("milestone_")
             and (by_epoch[e]["body"].get("retained") or {}).get(key, {}).get("sha256") != file_sha]
    if later:
        return False, "retained[%s] changes in later commits %s" % (key, later[:3])
    return True, "commit seq %s, epoch %d, retained[%s] name=%r" % (c.get("seq"), epoch, key, ent.get("name"))


def s_ledger(R, state):
    ledgers = {(a, s): load_ledger(R, a, s, state) for s, a in RUNS}
    # 1. test the rule on bindings already registered by verified posteval records
    tested = failed = 0
    for d, p, j in state["posteval"]:
        c = j.get("checkpoint", {})
        arm, seed, epoch, fn = c.get("arm"), c.get("seed"), c.get("epoch"), str(c.get("file_name"))
        if state["ident"].get(os.path.abspath(p)) != SNAP[p][0]:
            rec("L2", "B5L", "UNRESOLVED", "posteval record %s not confirmed by analysis.identities; not used to test the rule" % d[:12])
            continue
        le = ledgers.get((arm, seed))
        if le is None or not re.fullmatch(r"weights_e\d{3}\.pt", fn):
            continue                                   # weights_final.pt is not a retained ledger entry
        keys = [k for k in (c.get("roles") or []) if k in (le.get(epoch, {}).get("body", {}).get("roles_this_epoch") or [])]
        if not keys:
            failed += 1
            rec("L2", "B5L", "FAIL", "rule test (%s,%s,epoch %s): none of the checkpoint labels %s is in roles_this_epoch" % (arm, seed, epoch, c.get("roles")))
            continue
        for k in keys:
            tested += 1
            ok, why = bound(le, epoch, k, c.get("file_sha256"), c.get("model_state_sha256"))
            if not ok:
                failed += 1
                rec("L2", "B5L", "FAIL", "rule test (%s,%s,epoch %s,%s): %s" % (arm, seed, epoch, k, why))
    valid = tested >= 12 and failed == 0
    rec("L2", "B5L", "PASS" if valid else ("FAIL" if failed else "UNRESOLVED"),
        "ledger binding rule reproduces %d/%d bindings already registered by verified posteval records" % (tested - failed, tested))
    # 2. apply it to the Tier 2 files
    for seed, arm in RUNS:
        le = ledgers.get((arm, seed))
        for e in EPOCHS_T2:
            k = "(%s,%d,epoch%d)" % (arm, seed, e)
            path = os.path.join(R, "%d_%s" % (seed, arm), "ckpt", "weights_e%03d.pt" % e)
            st = state["val_state"].get((arm, seed, e))
            if not os.path.isfile(path) or st is None:
                rec("L3", "B5L", "UNRESOLVED", "%s: milestone file or registered model state missing" % k)
                continue
            fsha = snap(path)
            tri("L3", "B5L", fsha, sidecar(path), "%s file sha256 == its sidecar" % k)
            if le is None or not valid:
                rec("L3", "B5L", "UNRESOLVED", "%s: ledger not verified or rule not validated; binding not established" % k)
                continue
            ok, why = bound(le, e, "milestone_%03d" % e, fsha, st)
            name_ok = ok and ("name=%r" % os.path.basename(path)) in why
            rec("L3", "B5L", "CONFIRMED" if ok and name_ok else "UNRESOLVED",
                "%s file<->state binding by ledger commit record: %s%s" % (k, why, "" if name_ok or not ok else " (retained name differs from the file name)"))


# ---------------------------------------------------------------- step P
def s_p(repo, R, state):
    ana, li, recs = state["ana"], state["li"], state["posteval"]
    caps = ana.get("analysis", {}).get("caps", {})
    ent = li.get("inputs", {}).get("eligibility", {})
    ep = resolve(repo, R, ent.get("path"))
    mid = None
    if ep is None:
        rec("P2", "P", "UNRESOLVED", "eligibility manifest path does not resolve to one file")
    else:
        tri("P2", "P", snap(ep), ent.get("sha256"), "eligibility manifest sha256 == launch_inputs registration")
        mid = set((s["volume_id"], s["slice_index"]) for s in load_json(ep).get("slices", []) if s.get("midpoint") is True)
        rec("P2", "P", "PASS" if len(mid) == N_VOLUMES else "FAIL", "registered midpoint slices: %d" % len(mid))
    by_use, cnt = {}, {}
    def c(name, ok):
        cnt.setdefault(name, [0, 0])[0 if ok else 1] += 1
    idkeys, pop_sha, masks, unreg = set(), set(), set(), []
    for d, p, j in recs:
        reg = state["ident"].get(os.path.abspath(p))
        if reg is None:
            unreg.append(d[:12])
        else:
            c("file sha256 == analysis.identities", reg == SNAP[p][0])
        ck, pop, m = j.get("checkpoint", {}), j.get("population", {}), j.get("midpoint", {})
        for u in j.get("uses") or []:
            by_use.setdefault((u.get("arm"), u.get("seed"), u.get("role")), []).append(d)
        rows = m.get("rows", [])
        c("schema", j.get("schema") == POSTEVAL_SCHEMA)
        c("directory == model_state_sha256 == val_e<epoch> state",
          d == ck.get("model_state_sha256") == state["val_state"].get((ck.get("arm"), ck.get("seed"), ck.get("epoch"))))
        c("context.fixture is false", j.get("context", {}).get("fixture") is False)
        c("context.launch_inputs_sha256 == launch inputs file", j.get("context", {}).get("launch_inputs_sha256") == state["li_sha"])
        c("banks.primary == Z_DIAG", j.get("banks", {}).get("primary") == Z_DIAG)
        c("population scope midpoint, n == 97", pop.get("scope") == "midpoint" and pop.get("n") == N_VOLUMES)
        c("population.eligibility_manifest_sha256 == eligibility file", ep is not None and pop.get("eligibility_manifest_sha256") == SNAP[ep][0])
        c("97 midpoint rows", len(rows) == N_VOLUMES)
        c("every row carries r0 and r1 payloads (presence only)", bool(rows) and all(isinstance(r.get("r1"), dict) and r["r1"] and isinstance(r.get("r0"), dict) and r["r0"] for r in rows))
        c("decodes.primary == 97 x 128", m.get("decodes", {}).get("primary") == N_VOLUMES * N_SAMPLES)
        c("128 in prefixes", N_SAMPLES in (j.get("prefixes") or []))
        ids = [r.get("identity") for r in rows]
        idkeys |= set(k for i in ids if isinstance(i, dict) for k in i)
        if mid is not None and all(isinstance(i, dict) and "volume_id" in i and "slice_index" in i for i in ids) and ids:
            c("row identities == registered midpoint slices", set((i["volume_id"], i["slice_index"]) for i in ids) == mid and len(ids) == len(mid))
        pop_sha.add(str(pop.get("identities_sha256")))
        masks.add(str(j.get("masks", {}).get("sha256")))
    n = len(recs)
    if unreg or not n:
        rec("P2", "P", "UNRESOLVED", "posteval records without a registered identity in analysis.identities: %s" % (unreg or "no records"))
    for name, (good, bad) in cnt.items():
        full = n - len(unreg) if name.startswith("file sha256") else n
        rec("P2", "P", "PASS" if bad == 0 and good == full else "FAIL", "%s: %d/%d posteval records" % (name, good, full))
    if "row identities == registered midpoint slices" not in cnt:
        rec("P2", "P", "UNRESOLVED", "row identity fields are %s, not volume_id/slice_index; midpoint identity match not tested" % sorted(idkeys))
    rec("P2", "P", "PASS" if len(pop_sha) == 1 else "FAIL", "population.identities_sha256 identical across the %d records" % n)
    tri("P3", "P", next(iter(masks)) if len(masks) == 1 else "not-unique", ana.get("inputs", {}).get("masks_sha256"),
        "posteval masks.sha256 (single value across records) == analysis inputs.masks_sha256")
    miss = []
    for seed, arm in RUNS:
        for role in ROLES_T1:
            ok = len(by_use.get((arm, seed, role), [])) == 1 and ("%s|%d|%s|r1|mid" % (arm, seed, role)) in caps
            if not ok:
                miss.append((arm, seed, role))
    rec("P1", "P", "PASS" if not miss else "UNRESOLVED",
        "18 Tier 1 roles: each bound to exactly one posteval record and registered as <arm>|<seed>|<role>|r1|mid in the analysis document; missing=%s" % (miss or "none"))
    # P1b: per-entry source identity inside the analysis document (hash strings only; no metric value)
    own, known = {}, {}
    for d, p, j in recs:
        ck = j.get("checkpoint", {})
        hs = set(h for h in (ck.get("model_state_sha256"), ck.get("file_sha256"), SNAP[p][0]) if isinstance(h, str) and HEX64.fullmatch(h))
        for h in hs:
            known[h] = d
        own[d] = hs
    found = {}                                   # hash -> list of key paths (identities table excluded)
    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + (str(k),))
        elif isinstance(o, list):
            for v in o:
                walk(v, path)
        elif isinstance(o, str) and o in known:
            found.setdefault(o, []).append(path)
    walk({k: v for k, v in ana.items() if k != "inputs"}, ())
    walk({k: v for k, v in ana.get("inputs", {}).items() if k != "identities"}, ("inputs",))
    n_ok = n_bad = 0
    open_roles = []
    for seed, arm in RUNS:
        for role in ROLES_T1:
            ds = by_use.get((arm, seed, role), [])
            if len(ds) != 1:
                continue
            pre = "%s|%d|%s" % (arm, seed, role)
            under = [(h, pth) for h, pths in found.items() for pth in pths if any(str(x) == pre or str(x).startswith(pre + "|") for x in pth)]
            mine = [x for x in under if x[0] in own[ds[0]]]
            other = [x for x in under if x[0] not in own[ds[0]]]
            if other:
                n_bad += 1
                rec("P1b", "P", "FAIL", "%s: analysis entry carries a hash of another posteval record at %s" % (pre, "/".join(other[0][1])))
            elif mine:
                n_ok += 1
            else:
                open_roles.append(pre)
    if n_ok:
        rec("P1b", "P", "CONFIRMED", "%d/18 analysis role entries carry a hash of their own verified posteval record" % n_ok)
    if open_roles:
        rec("P1b", "P", "UNRESOLVED", "%d/18 analysis role entries hold no per-entry source identity; their link to the posteval record rests on "
                                      "the registered inputs (analysis.identities) and the pinned analysis code only: %s" % (len(open_roles), open_roles[:3]))
        rec("P1b", "P", "INFO", "posteval hashes elsewhere in the analysis document (outside inputs.identities): %d distinct, e.g. %s" % (
            len(found), sorted(set("/".join(pt[:3]) for pts in found.values() for pt in pts))[:5]))
    reg_dirs = set(os.path.basename(os.path.dirname(k)) for k in state["ident"] if "/posteval/" in k and k.endswith("/posteval.json"))
    have = set(d for d, _p, _j in recs)
    rec("P1b", "P", "PASS" if reg_dirs == have else ("FAIL" if reg_dirs - have else "UNRESOLVED"),
        "posteval records registered as analysis inputs vs on disk: %d registered, %d on disk, %d registered but missing, %d on disk but unregistered" % (
            len(reg_dirs), len(have), len(reg_dirs - have), len(have - reg_dirs)))
    rec("P1", "P", "INFO", "payload presence only: no r0/r1 value, cap count or analysis outcome was read or printed")


def s_after(repo, state):
    changed = 0
    for p, (sha, size, mt) in sorted(SNAP.items()):
        st = os.stat(p)
        if sha_file(p) != sha or st.st_size != size or st.st_mtime_ns != mt:
            changed += 1
            rec("S6", "INT", "FAIL", "CHANGED during audit: %s" % p)
    rec("S6", "INT", "PASS" if changed == 0 else "FAIL", "before/after sha256, size and mtime_ns over %d inspected files: %d changed" % (len(SNAP), changed))
    if "git" in state:
        now = (git(repo, "rev-parse", "HEAD"), hashlib.sha256(git(repo, "status", "--porcelain").encode()).hexdigest())
        changed += now != state["git"]
        rec("S6", "INT", "PASS" if now == state["git"] else "FAIL", "Git HEAD and porcelain status unchanged")
    else:
        rec("S6", "INT", "UNRESOLVED", "Git state not compared")
    return changed


def verdict(item):
    t = TALLY[item]
    if any(t.get(s) for s in BAD):
        return "FAIL"
    if any(t.get(s) for s in OPEN) or not any(t.get(s) for s in GOOD):
        return "INCOMPLETE"
    return "PASS"


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/benjamin/CSMFII")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    R = os.path.join(repo, "seqref_mri", "results", "v05lt")
    out_dir = os.path.abspath(a.out) if a.out else os.path.join(repo, "v06lt_evidence", "DIAG_UNC", "audit")
    if not os.path.isdir(R):
        print("ERROR: results directory not found: %s" % R, file=sys.stderr)
        return 1
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(out_dir, "dunc_audit2_%s.log" % ts)
    LOG = open(log_path, "x", encoding="utf-8")
    me = os.path.abspath(__file__)
    rec("S0", "INT", "INFO", "%s | script=%s sha256=%s | argv=%s | python=%s | utc=%s" % (
        VERSION, me, sha_file(me), sys.argv[1:], sys.version.split()[0], ts))
    state = {}
    step("S0", "INT", s0, repo, R, state)
    step("S0", "INT", s0_records, repo, R, state)
    if "posteval" in state:
        step("L", "B5L", s_ledger, R, state)
        step("P", "P", s_p, repo, R, state)
    else:
        rec("S0", "B5L", "UNRESOLVED", "registered records not loaded; ledger checks not run")
        rec("S0", "P", "UNRESOLVED", "registered records not loaded; readout checks not run")
    changed = step("S6", "INT", s_after, repo, state)
    v = {i: verdict(i) for i in ITEMS}
    for i in ITEMS:
        rec("S6", "-", "VERDICT", "%s = %s %s" % (i, v[i], dict(sorted(TALLY[i].items()))))
    if changed is None or changed or v["INT"] == "FAIL":
        overall, code = "FAIL (audit integrity)", 2
    elif "FAIL" in v.values():
        overall, code = "FAIL", 1
    elif "INCOMPLETE" in v.values():
        overall, code = "INCOMPLETE", 3
    else:
        overall, code = "PASS", 0
    rec("S6", "-", "OUTCOME", "%s (exit %d)" % (overall, code))
    LOG.close()
    digest = sha_file(log_path)
    with open(log_path + ".sha256", "x", encoding="utf-8") as f:
        f.write("%s  %s\n" % (digest, os.path.basename(log_path)))
    print("LOG %s\nSHA256 %s" % (log_path, digest))
    return code


if __name__ == "__main__":
    sys.exit(main())
