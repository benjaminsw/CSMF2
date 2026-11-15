# SEQREF-DUNC-AUD v0.4.2 -- dunc_audit.py: read-only B1/B3/B5 provenance audit for DIAG-UNC (V05LT)
# LIFETIME: DIAGNOSTIC
"""Read-only provenance audit for D-UNC freeze items B1, B3 and B5 (article 223 s14; record 224).

Standard library only. No torch, no numpy, no checkpoint loading, no MRI data, no model
evaluation, no contrast, no aggregation of NLL/PSNR. Metric fields are checked for presence,
numeric type and finiteness only; their values are never printed or combined.

Writes exactly two new files (log + .sha256 sidecar) in the output directory; nothing else.
No fallback, placeholder, mock or dummy path: anything that cannot be established from an
existing registered record is UNRESOLVED and makes its item INCOMPLETE.

Terminal outcome (worst of B1, B3, B5, INT):
  exit 0  PASS        every mandatory check verified
  exit 3  INCOMPLETE  no failure, but required evidence unresolved
  exit 1  FAIL        mismatch, validity failure or script error
  exit 2  FAIL        audit integrity failure (an inspected file or the Git state changed)

Usage (host):
  python3 -I -B dunc_audit.py --repo /home/benjamin/CSMFII

Changelog
- v0.4.2: a ledger line is never accepted as a file<->state binding. The ledger schema has not
  been reviewed, so co-occurrence of the two hashes in one line is logged as INFO (line number
  and the key paths holding each hash) and the Tier 2 binding stays UNRESOLVED (B5 INCOMPLETE)
  unless a verified posteval record binds the file.
- v0.4.1: posteval.json files verified against their registered identities (sidecar or
  analysis.identities; unregistered -> UNRESOLVED) before they are used to bind roles; the
  primary-bank file is hashed only when a record entry carries sha256 == Z_DIAG together
  with an explicit `path`; conflicting file<->state bindings across posteval records -> FAIL
  and are never used.
- v0.4: PASS / INCOMPLETE / FAIL outcomes per item with distinct exit codes; full Z_DIAG hash
  and registered-reference checks; exact population match against the split and eligibility
  manifests (97 volumes, registered slice counts, one registered midpoint per volume); roles
  bound through posteval `uses` and verified with checkpoint metadata and final_weights.json;
  Tier 2 file<->state binding reported separately (posteval record, else UNRESOLVED);
  Git root and pin commit verified before any Git read.
- v0.3: first reviewed version (not run on the host).
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback

VERSION = "SEQREF-DUNC-AUD v0.4.2"
PIN = "7455da8027f33a6cca4dbf72f04e72420e938f45"
VAL_SCHEMA = "seqref-v05lt-val-rows/1"
Z_DIAG = "f8a6a34b6b474286359e4c99341de256b62bc0dff1e6bc8cf52e72f0461dc8d6"  # registered primary bank
N_VOLUMES, N_SLICES = 97, 3461                     # 194 split; DISC v0.4/v0.5
RUNS = [(s, a) for a in ("lam0", "lam2") for s in (2026, 2027, 2028)]
ROLES_T1 = ("epoch3", "selected", "final")
EPOCHS_T2 = (10, 30)
# Already-exposed values quoted in 223 s1 (213 E16; 217 J17); consistency checks only.
EXPECT_SELECTED = {("lam0", 2026): 9, ("lam0", 2027): 10, ("lam0", 2028): 13,
                   ("lam2", 2026): 55, ("lam2", 2027): 48, ("lam2", 2028): 55}
EXPECT_FINAL = {("lam0", 2026): 49, ("lam0", 2027): 49, ("lam0", 2028): 50,
                ("lam2", 2026): 58, ("lam2", 2027): 49, ("lam2", 2028): 55}
# Preliminary provenance log prov_20261114T083034Z (sha256 of file content at the pin).
PRELIM = {
    "seqref_mri/tdiag/d3.py": "73110e7d36a1ebdcb51926c4bbde81f88392f578e1aa860736c509706041245e",
    "seqref_mri/src/free_flow_runtime.py": "8e171b2c521d77c8ad22b21d02537be160d15166aecb9e0319fd0a25b69823c9",
    "seqref_mri/scripts/v05lt_eval.py": "10b4bc65d465e9c6d0a016c2d27977cd9f485ffb851936d9a15ad6a325898554",
    "seqref_mri/scripts/v05lt_metrics.py": "87eaa5a4e85b6b70072c1c76fcd3e73ce104c3cdb04ebef973c346ff90b18a74",
}
NUM_KEYS = ("nll_per_dim", "nll_prior_per_dim", "nll_logdet_per_dim", "psnr")
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
ITEMS = ("B1", "B3", "B5", "INT")
GOOD, BAD, OPEN = ("PASS", "CONFIRMED", "RESOLVED"), ("FAIL", "ERROR"), ("UNRESOLVED",)
# INFO and DEFERRED lines never change a verdict; DEFERRED lines are listed in the summary.

TALLY = {i: {} for i in ITEMS}
DEFERRED = []
LOG = None
SNAP = {}          # path -> (sha256, size, mtime_ns) at first read
CKPT_SHA = {}      # checkpoint path -> sha256 (before)


def rec(step, item, status, msg):
    if item in TALLY:
        TALLY[item][status] = TALLY[item].get(status, 0) + 1
    if status == "DEFERRED":
        DEFERRED.append(msg)
    line = "[%s] %-3s %-10s %s" % (step, item, status, msg)
    LOG.write(line + "\n")
    LOG.flush()
    print(line)


def tri(step, item, result, msg, absent="UNRESOLVED"):
    """result in MATCH / MISMATCH / ABSENT -> PASS / FAIL / <absent>."""
    rec(step, item, {"MATCH": "PASS", "MISMATCH": "FAIL", "ABSENT": absent}[result], msg)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snap(path):
    """Record the before-identity of a file once; return its sha256."""
    if path not in SNAP:
        st = os.stat(path)
        SNAP[path] = (sha_file(path), st.st_size, st.st_mtime_ns)
    return SNAP[path][0]


def load_json(path):
    snap(path)
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def sidecar(path):
    """Registered sha256 from <path>.sha256, or None if the sidecar is absent or unparseable."""
    sc = path + ".sha256"
    if not os.path.isfile(sc):
        return None
    snap(sc)
    with open(sc, "r", encoding="utf-8", errors="replace") as f:
        m = HEX64.search(f.read())
    return m.group(0) if m else None


def cmp3(computed, registered):
    if registered is None:
        return "ABSENT"
    return "MATCH" if computed == registered else "MISMATCH"


def git(repo, *args, binary=False):
    p = subprocess.run(["git", "--no-optional-locks", "-C", repo] + list(args),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("git %s: %s" % (" ".join(args), p.stderr.decode(errors="replace").strip()))
    return p.stdout if binary else p.stdout.decode(errors="replace").strip()


def resolve(repo, R, raw):
    """Registered path -> existing file, or None unless exactly one candidate exists."""
    raw = str(raw)
    cands = [raw] if os.path.isabs(raw) else [os.path.join(b, raw) for b in (repo, os.path.join(repo, "seqref_mri"), R)]
    hits = sorted(set(os.path.realpath(c) for c in cands if os.path.isfile(c)))
    return hits[0] if len(hits) == 1 else None


def find_value(o, target, path, hits):
    if isinstance(o, dict):
        for k, v in o.items():
            find_value(v, target, path + "/" + str(k), hits)
    elif isinstance(o, list):
        for v in o:
            find_value(v, target, path + "[]", hits)
    elif o == target:
        hits.append(path)


def step(name, item, fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # logged and counted as a failure of the step's item
        tb = traceback.extract_tb(e.__traceback__)[-1]
        rec(name, item, "ERROR", "%s: %s (line %d in %s)" % (type(e).__name__, e, tb.lineno, tb.name))
        return None


def run_item(arm, role):
    """B5 owns the lam2 epoch-3 and final roles and all Tier 2 roles (223 s14); B1 the rest."""
    return "B5" if (arm == "lam2" and role in ("epoch3", "final")) or role.startswith("epoch1") or role.startswith("epoch30") else "B1"


# ---------------------------------------------------------------- S0
def s0_git(repo):
    top = git(repo, "rev-parse", "--show-toplevel")
    ok = os.path.realpath(top) == os.path.realpath(repo)
    rec("S0", "INT", "PASS" if ok else "FAIL", "Git toplevel %s the --repo argument" % ("==" if ok else "!="))
    if not ok:
        raise RuntimeError("repository root not verified; Git-dependent checks refused")
    git(repo, "cat-file", "-e", PIN + "^{commit}")
    rec("S0", "INT", "PASS", "pin commit %s exists in the repository" % PIN)
    head = git(repo, "rev-parse", "HEAD")
    status = git(repo, "status", "--porcelain")
    rec("S0", "INT", "INFO", "HEAD=%s (%s the pin) branch=%s porcelain_lines=%d" % (
        head, "==" if head == PIN else "!=", git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        len([l for l in status.splitlines() if l.strip()])))
    return head, hashlib.sha256(status.encode()).hexdigest()


# ---------------------------------------------------------------- S1
def s1_sources(repo, R, state):
    pin_rec = load_json(os.path.join(R, "launch", "v05lt_pin_record.json"))
    li = load_json(os.path.join(R, "launch", "v05lt_launch_inputs.json"))
    ana = load_json(os.path.join(R, "analysis", "v05lt_analysis.json"))
    state["launch_inputs"], state["analysis"] = li, ana
    state["identities"] = ana.get("inputs", {}).get("identities", {})
    rec("S1", "B1", "PASS" if pin_rec.get("commit") == PIN else "FAIL", "pin_record.commit == pin")
    rec("S1", "B1", "PASS" if li.get("pin", {}).get("commit") == PIN else "FAIL", "launch_inputs.pin.commit == pin")
    rec("S1", "B1", "INFO", "pin_record identity_method=%r; n(source_identity)=%d" % (
        pin_rec.get("identity_method"), len(pin_rec.get("source_identity", {}))))
    if not state.get("git_ok"):
        rec("S1", "B1", "UNRESOLVED", "source files not verified: Git root or pin not established in S0")
        return
    src_id, li_code = pin_rec.get("source_identity", {}), li.get("pin", {}).get("code", {})
    at_pin = {}
    for rel, prelim in PRELIM.items():
        sha = hashlib.sha256(git(repo, "show", "%s:%s" % (PIN, rel), binary=True)).hexdigest()
        blob_id = git(repo, "rev-parse", "%s:%s" % (PIN, rel))
        at_pin[rel] = sha
        reg = src_id.get(rel)
        if reg is None:
            rec("S1", "B1", "UNRESOLVED", "%s not listed in pin_record.source_identity" % rel)
        else:
            how = "sha256 of content" if reg == sha else ("git blob id" if reg == blob_id else None)
            rec("S1", "B1", "CONFIRMED" if how else "FAIL",
                "%s vs pin_record.source_identity (%s at pin)" % (rel, how or "no match"))
        sec = li_code.get(rel)
        if sec is None:
            rec("S1", "B1", "INFO", "%s not listed in launch_inputs.pin.code (secondary record)" % rel)
        else:
            rec("S1", "B1", "PASS" if sec in (sha, blob_id) else "FAIL", "%s vs launch_inputs.pin.code" % rel)
        rec("S1", "B1", "PASS" if sha == prelim else "FAIL", "%s sha256 at pin == preliminary provenance log" % rel)
        wt = os.path.join(repo, rel)
        rec("S1", "B1", "PASS" if os.path.isfile(wt) and snap(wt) == sha else "FAIL",
            "%s working-tree file == content at pin" % rel)
    state["at_pin"] = at_pin
    for base in ("v05lt_eval.py", "v05lt_metrics.py"):
        reg = ana.get("code", {}).get(base)
        if reg is None:
            rec("S1", "B1", "INFO", "analysis.code has no entry for %s (secondary record)" % base)
        else:
            rec("S1", "B1", "PASS" if reg == at_pin["seqref_mri/scripts/" + base] else "FAIL",
                "analysis.code[%s] == sha256 at pin" % base)


def s1_bank(repo, R, state):
    """Primary bank: registered hash references now; byte identity only if a record registers a file."""
    refs = []
    for name in ("v05lt_bank_20261023_manifest.json", "v05lt_bank_20261024_manifest.json"):
        p = os.path.join(R, "inputs", name)
        refs.append((name + " recipe_check.z_diag_sha256",
                     load_json(p).get("recipe_check", {}).get("z_diag_sha256") if os.path.isfile(p) else None))
    for s in (2026, 2027, 2028):
        p = os.path.join(R, "epoch0", "seed_%d" % s, "eval", "epoch0_eval.json")
        refs.append(("epoch0 seed %d context.bank_primary_sha256" % s,
                     load_json(p).get("context", {}).get("bank_primary_sha256") if os.path.isfile(p) else None))
    for label, v in refs:
        tri("S1", "B1", cmp3(Z_DIAG, v), "%s == registered Z_DIAG hash" % label)
    p1 = os.path.join(R, "preflight", "v05lt_p1_inputs.json")
    files = []
    if os.path.isfile(p1):
        j = load_json(p1)
        hits = []
        find_value(j, Z_DIAG, "", hits)
        rec("S1", "B1", "CONFIRMED" if hits else "UNRESOLVED", "preflight P1 record references the Z_DIAG hash at %s" % (hits or "no key"))
        for ent in j.get("evidence_inputs", []) if isinstance(j.get("evidence_inputs"), list) else []:
            # only an entry that itself pairs sha256 == Z_DIAG with an explicit `path` names the bank file
            if isinstance(ent, dict) and ent.get("sha256") == Z_DIAG and isinstance(ent.get("path"), str):
                f = resolve(repo, R, ent["path"])
                if f:
                    files.append(("path", f))
                else:
                    rec("S1", "B1", "UNRESOLVED", "preflight P1 entry registers a primary-bank path %r that does not resolve to one file" % ent["path"])
    else:
        rec("S1", "B1", "UNRESOLVED", "preflight P1 record not found")
    if files:
        for k, f in files:
            rec("S1", "B1", "CONFIRMED" if snap(f) == Z_DIAG else "FAIL",
                "primary bank file registered by an explicit %r entry: sha256 of %s == Z_DIAG hash" % (k, os.path.relpath(f, repo)))
    else:
        rec("S1", "B1", "DEFERRED", "primary Z_DIAG bank: no record entry pairs the Z_DIAG hash with an explicit file path; byte identity "
                                    "by regeneration from the frozen recipe is left to a separately approved fixture (223 s10)")


# ---------------------------------------------------------------- S4 (run before roles: supplies epoch-indexed states)
def s4_population(repo, R, state):
    li = state["launch_inputs"]
    reg = {}
    for name in ("split_manifest", "eligibility", "case_manifest"):
        ent = li.get("inputs", {}).get(name, {})
        p = resolve(repo, R, ent.get("path"))
        if p is None:
            rec("S4", "B3", "UNRESOLVED", "%s: registered path %r does not resolve to exactly one file" % (name, ent.get("path")))
            continue
        tri("S4", "B3", cmp3(snap(p), ent.get("sha256")), "%s file sha256 == launch_inputs registration" % name)
        tri("S4", "B3", cmp3(SNAP[p][0], sidecar(p)), "%s file sha256 == its sidecar" % name)
        reg[name] = (p, load_json(p))
    if "split_manifest" not in reg or "eligibility" not in reg:
        rec("S4", "B3", "UNRESOLVED", "registered population not established; row checks limited to internal consistency")
        return None
    sp_path, sp = reg["split_manifest"]
    el = reg["eligibility"][1]
    val = sp.get("validation", [])
    pop = {}                                      # volume_id -> (file, n_slices)
    for v in val:
        pop[v["volume_id"]] = (v["file"], v["n_slices"])
    n_sl = sum(n for _, n in pop.values())
    rec("S4", "B3", "PASS" if len(val) == N_VOLUMES and len(pop) == N_VOLUMES else "FAIL",
        "split manifest validation list: %d records, %d distinct volume ids (expected %d)" % (len(val), len(pop), N_VOLUMES))
    rec("S4", "B3", "PASS" if n_sl == N_SLICES and sp.get("counts", {}).get("validation_slices") == n_sl
        and sp.get("counts", {}).get("validation_volumes") == len(pop) else "FAIL",
        "registered slice counts sum to %d (expected %d) and agree with split manifest counts" % (n_sl, N_SLICES))
    for holder, label in ((el.get("provenance", {}), "eligibility.provenance"), (reg.get("case_manifest", (None, {}))[1], "case_manifest")):
        tri("S4", "B3", cmp3(SNAP[sp_path][0], holder.get("split_manifest_sha256")),
            "%s.split_manifest_sha256 == split manifest file sha256" % label, absent="INFO")
    ids = set((v, s) for v, (_, n) in pop.items() for s in range(n))
    el_ids = [(s["volume_id"], s["slice_index"]) for s in el.get("slices", [])]
    el_mid = [(s["volume_id"], s["slice_index"]) for s in el.get("slices", []) if s.get("midpoint") is True]
    rec("S4", "B3", "PASS" if len(el_ids) == len(set(el_ids)) and set(el_ids) == ids else "FAIL",
        "eligibility manifest slice identities == split manifest volumes x registered slice counts (%d)" % len(el_ids))
    ok_mid = len(el_mid) == N_VOLUMES and sorted(v for v, _ in el_mid) == sorted(pop)
    rec("S4", "B3", "PASS" if ok_mid else "FAIL",
        "registered midpoint slices: %d, exactly one per validation volume" % len(el_mid))
    return {"pop": pop, "ids": ids, "mid": set(el_mid)}


def s4_rows(R, state, P):
    ident = state.get("identities", {})
    run_info = state["analysis"].get("inputs", {}).get("run_info", {})
    val_state, sid_all = {}, set()
    for seed, arm in RUNS:
        d = os.path.join(R, "%d_%s" % (seed, arm), "ckpt")
        files = sorted(f for f in os.listdir(d) if re.fullmatch(r"val_e\d{3}\.json", f))
        epochs = [int(f[5:8]) for f in files]
        n = len(files)
        rec("S4", "B3", "PASS" if n and epochs == list(range(1, n + 1)) else "FAIL",
            "(%s,%d) validation-row files contiguous for epochs 1..%d" % (arm, seed, n))
        rec("S4", "B3", "PASS" if n == EXPECT_FINAL[(arm, seed)] else "FAIL",
            "(%s,%d) %d epochs == final epoch quoted in 223" % (arm, seed, n))
        tri("S4", "B3", cmp3(n, run_info.get("%s|%d" % (arm, seed), {}).get("last_epoch")),
            "(%s,%d) %d epochs == analysis run_info.last_epoch" % (arm, seed, n))
        fail = {"sidecar": 0, "identity": 0, "schema": 0, "header": 0, "n_rows": 0, "state": 0, "row_keys": 0,
                "types": 0, "nonfinite": 0, "population": 0, "midpoint": 0, "file_field": 0}
        unres = {"sidecar_absent": 0, "identity_absent": 0}
        sids = set()
        for f, e in zip(files, epochs):
            p = os.path.join(d, f)
            j = load_json(p)
            sha = SNAP[p][0]
            for key, regv in (("sidecar", sidecar(p)), ("identity", ident.get(os.path.abspath(p)))):
                c = cmp3(sha, regv)
                if c == "MISMATCH":
                    fail[key] += 1
                elif c == "ABSENT":
                    unres[key + "_absent"] += 1
            if j.get("schema") != VAL_SCHEMA:
                fail["schema"] += 1
            if j.get("epoch") != e or j.get("seed") != seed or j.get("arm") != arm:
                fail["header"] += 1
            rows = j.get("rows", [])
            if j.get("n_rows") != len(rows):
                fail["n_rows"] += 1
            st = j.get("model_state_sha256")
            if isinstance(st, str) and HEX64.fullmatch(st):
                val_state[(arm, seed, e)] = st
            else:
                fail["state"] += 1
            idl, midl = [], []
            for r in rows:
                if not all(k in r for k in NUM_KEYS + ("midpoint", "volume_id", "slice_index", "file")):
                    fail["row_keys"] += 1
                    continue
                for k in NUM_KEYS:
                    v = r[k]
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        fail["types"] += 1
                    elif not math.isfinite(v):
                        fail["nonfinite"] += 1
                if not isinstance(r["midpoint"], bool):
                    fail["types"] += 1
                key = (r["volume_id"], r["slice_index"])
                idl.append(key)
                if r["midpoint"] is True:
                    midl.append(key)
                if P and P["pop"].get(r["volume_id"], (None,))[0] != r["file"]:
                    fail["file_field"] += 1
            if P:
                if len(idl) != len(set(idl)) or set(idl) != P["ids"]:
                    fail["population"] += 1
                if len(midl) != len(set(midl)) or set(midl) != P["mid"]:
                    fail["midpoint"] += 1
            sids.add(str(j.get("slice_ids_sha256")))
        sid_all |= sids
        bad = {k: v for k, v in fail.items() if v}
        rec("S4", "B3", "FAIL" if bad else "PASS",
            "(%s,%d) %d files: schema, header, per-row B3 fields (presence, numeric type, finiteness)%s; failures=%s" % (
                arm, seed, n, ", exact registered population and midpoint identities" if P else "", bad or "none"))
        if any(unres.values()):
            rec("S4", "B3", "UNRESOLVED", "(%s,%d) registered file identity missing: %s" % (arm, seed, {k: v for k, v in unres.items() if v}))
        rec("S4", "B3", "PASS" if len(sids) == 1 else "FAIL", "(%s,%d) slice_ids_sha256 constant over epochs" % (arm, seed))
    rec("S4", "B3", "PASS" if len(sid_all) == 1 else "FAIL", "slice_ids_sha256 identical across the six runs")
    rec("S4", "B3", "INFO", "metric fields %s read for presence, type and finiteness only; no value printed or aggregated" % (NUM_KEYS,))
    state["val_state"] = val_state


# ---------------------------------------------------------------- S2 / S3
def find_ckpt(run_dir, file_name):
    hits = [c for c in (os.path.join(run_dir, "ckpt", file_name), os.path.join(run_dir, "final", file_name)) if os.path.isfile(c)]
    return hits[0] if len(hits) == 1 else None


def s2_roles(R, state):
    pe_root = os.path.join(R, "posteval")
    recs, by_use = [], {}
    ident = state.get("identities", {})
    prov = {"registered": 0, "unregistered": 0, "mismatched": 0}
    for d in sorted(os.listdir(pe_root)):
        p = os.path.join(pe_root, d, "posteval.json")
        if not os.path.isfile(p):
            rec("S2", "B1", "UNRESOLVED", "posteval directory without posteval.json: %s" % d)
            continue
        j = load_json(p)
        res = [cmp3(SNAP[p][0], x) for x in (sidecar(p), ident.get(os.path.abspath(p)))]
        prov["mismatched" if "MISMATCH" in res else ("registered" if "MATCH" in res else "unregistered")] += 1
        c = j.get("checkpoint", {})
        r = {"dir": d, "arm": c.get("arm"), "seed": c.get("seed"), "epoch": c.get("epoch"), "file_name": c.get("file_name"),
             "file_sha256": c.get("file_sha256"), "state": c.get("model_state_sha256"), "labels": c.get("roles") or [],
             "bank": j.get("banks", {}).get("primary"), "fixture": j.get("context", {}).get("fixture"),
             "n_mid": len(j.get("midpoint", {}).get("rows", [])), "code": j.get("code", {}).get("files", {}) or {}}
        recs.append(r)
        for u in j.get("uses") or []:
            by_use.setdefault((u.get("arm"), u.get("seed"), u.get("role")), []).append(r)
    n = len(recs)
    rec("S2", "B1", "FAIL" if prov["mismatched"] else ("UNRESOLVED" if prov["unregistered"] or not n else "CONFIRMED"),
        "posteval.json file sha256 vs registered identity (sidecar or analysis.identities): %s; role bindings below rest on these records" % prov)
    rec("S2", "B1", "PASS" if n and all(r["dir"] == r["state"] for r in recs) else "FAIL",
        "%d posteval records; directory name == checkpoint.model_state_sha256 for all" % n)
    rec("S2", "B1", "PASS" if n and all(r["bank"] == Z_DIAG for r in recs) else "FAIL",
        "banks.primary == registered Z_DIAG hash in all %d posteval records" % n)
    rec("S2", "B1", "PASS" if n and all(r["fixture"] is False for r in recs) else "FAIL", "context.fixture is false in all posteval records")
    rec("S2", "B1", "PASS" if n and all(r["n_mid"] == N_VOLUMES for r in recs) else "FAIL",
        "midpoint row count == %d in all posteval records (counts only; rows not read)" % N_VOLUMES)
    at_pin = state.get("at_pin")
    if at_pin:
        bad = sum(1 for r in recs for b, s in r["code"].items()
                  if "seqref_mri/scripts/" + b in at_pin and s != at_pin["seqref_mri/scripts/" + b])
        rec("S2", "B1", "PASS" if bad == 0 else "FAIL", "posteval code.files hashes == sha256 at pin (%d mismatches)" % bad)
    roles, post_bind = {}, {}
    conflicts = set()
    for r in recs:
        if post_bind.get(r["file_sha256"], r["state"]) != r["state"]:
            conflicts.add(r["file_sha256"])
        post_bind[r["file_sha256"]] = r["state"]
    for c in sorted(conflicts, key=str):
        post_bind[c] = None                      # never used as a binding
    rec("S2", "B1", "FAIL" if conflicts else "PASS",
        "posteval file<->state bindings consistent: %d file hashes are bound to more than one model state" % len(conflicts))
    state["conflicts"] = conflicts
    for seed, arm in RUNS:
        run_dir = os.path.join(R, "%d_%s" % (seed, arm))
        fw_path = os.path.join(run_dir, "final", "final_weights.json")
        fw = load_json(fw_path)
        tri("S2", run_item(arm, "final"), cmp3(SNAP[fw_path][0], sidecar(fw_path)), "(%s,%d) final_weights.json == its sidecar" % (arm, seed))
        for role in ROLES_T1:
            item, k = run_item(arm, role), (arm, seed, role)
            hits = by_use.get(k, [])
            if len(hits) != 1:
                rec("S2", item, "UNRESOLVED", "%s: %d posteval records carry this use" % (k, len(hits)))
                continue
            r = hits[0]
            ok = r["arm"] == arm and r["seed"] == seed
            if role == "epoch3":
                ok = ok and r["epoch"] == 3 and "milestone_003" in r["labels"]
                what = "checkpoint arm/seed, epoch 3, label milestone_003"
            elif role == "selected":
                ok = ok and r["epoch"] == EXPECT_SELECTED[(arm, seed)] and "best_psnr" in r["labels"]
                what = "checkpoint arm/seed, epoch %d quoted in 223, label best_psnr" % EXPECT_SELECTED[(arm, seed)]
            else:
                ok = (ok and r["epoch"] == EXPECT_FINAL[(arm, seed)] == fw.get("epoch")
                      and r["state"] == fw.get("model_state_sha256"))
                what = "checkpoint arm/seed, epoch %d quoted in 223, final_weights.json epoch and model state" % EXPECT_FINAL[(arm, seed)]
            rec("S2", item, "PASS" if ok else "FAIL", "%s use verified against: %s" % (k, what))
            if role == "final":   # 223 s1: the final role is the published final-weights file
                path, freg, src = find_ckpt(run_dir, str(fw.get("weights_file"))), fw.get("weights_sha256"), "final_weights.json"
                state_reg = fw.get("model_state_sha256")
                rec("S2", item, "INFO", "%s posteval evaluated %r; published file %r, status=%r" % (
                    k, r["file_name"], fw.get("weights_file"), fw.get("status")))
            else:
                path, freg, src, state_reg = find_ckpt(run_dir, str(r["file_name"])), r["file_sha256"], "posteval record", r["state"]
            if path is None:
                rec("S2", item, "UNRESOLVED", "%s: checkpoint file not found uniquely" % (k,))
                continue
            if role != "final" and freg in conflicts:
                rec("S2", item, "FAIL", "%s: its posteval file hash is bound to conflicting model states" % (k,))
                continue
            roles[k] = {"path": path, "epoch": r["epoch"], "state": state_reg, "file_reg": freg, "bind": src, "item": item}
            rec("S2", item, "RESOLVED", "%s -> %s (file and model state registered together in %s)" % (k, os.path.relpath(path, R), src))
        for e in EPOCHS_T2:
            k = (arm, seed, "epoch%d" % e)
            path = os.path.join(run_dir, "ckpt", "weights_e%03d.pt" % e)
            if os.path.isfile(path):
                roles[k] = {"path": path, "epoch": e, "state": None, "file_reg": None, "bind": None, "item": "B5"}
                rec("S2", "B5", "RESOLVED", "%s -> %s (milestone file present)" % (k, os.path.relpath(path, R)))
            else:
                rec("S2", "B5", "UNRESOLVED", "%s: milestone file missing" % (k,))
    state["roles"], state["post_bind"] = roles, post_bind


def s2_hash_ckpts(R, state):
    roles, ident = state["roles"], state.get("identities", {})
    by_path = {}
    for k, v in roles.items():
        by_path.setdefault(v["path"], []).append((k, v))
    total = 0
    for p in sorted(by_path):
        sha = snap(p)
        CKPT_SHA[p] = sha
        total += SNAP[p][1]
        item = "B5" if all(v["item"] == "B5" for _, v in by_path[p]) else "B1"
        regs = [("sidecar", sidecar(p)), ("analysis.identities", ident.get(os.path.abspath(p)))]
        regs += [(v["bind"], v["file_reg"]) for _, v in by_path[p] if v["file_reg"]]
        res = [(n, cmp3(sha, r)) for n, r in regs]
        status = "FAIL" if any(c == "MISMATCH" for _, c in res) else ("UNRESOLVED" if res[0][1] == "ABSENT" else "CONFIRMED")
        rec("S2", item, status, "file %s sha256=%s size=%d | %s" % (
            os.path.relpath(p, R), sha, SNAP[p][1], " ".join("%s=%s" % x for x in res)))
    rec("S5", "INT", "INFO", "D-UNC checkpoint files=%d total_bytes=%d" % (len(by_path), total))


def ledger_candidates(run_dir, file_sha, state_sha):
    """Ledger lines in which both hashes occur. Evidence for review only: never a binding,
    because the ledger schema has not been reviewed (v0.4.2)."""
    p = os.path.join(run_dir, "ledger.jsonl")
    out = []
    if not os.path.isfile(p):
        return out
    snap(p)
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if file_sha in line and state_sha in line:
                fk, sk = [], []
                try:
                    obj = json.loads(line)
                    find_value(obj, file_sha, "", fk)
                    find_value(obj, state_sha, "", sk)
                except ValueError:
                    pass
                out.append("line %d: file hash at %s, state hash at %s" % (i, fk or "?", sk or "?"))
    return out[:3]


def s3_identities(R, state):
    roles, vs, post = state["roles"], state.get("val_state", {}), state["post_bind"]
    for k in sorted(roles, key=str):
        arm, seed, role = k
        v = roles[k]
        reg = vs.get((arm, seed, v["epoch"]))
        if v["state"] is None:                       # Tier 2: state from the epoch-indexed validation record
            v["state"] = reg
            fsha = CKPT_SHA.get(v["path"])
            if reg is None or fsha is None:
                rec("S3", "B5", "UNRESOLVED", "%s: no registered model state or file hash" % (k,))
            elif fsha in state.get("conflicts", ()):
                rec("S3", "B5", "FAIL", "%s: file hash is bound to conflicting model states in the posteval records" % (k,))
            elif post.get(fsha) == reg:
                rec("S3", "B5", "CONFIRMED", "%s file<->state binding registered by a posteval record" % (k,))
            else:
                rec("S3", "B5", "UNRESOLVED", "%s: file sha256 and val_e%03d model state are each registered, but no verified "
                                              "record binds this file to that state; needs the F1 model-state hash" % (k, v["epoch"]))
                for c in ledger_candidates(os.path.dirname(os.path.dirname(v["path"])), fsha, reg):
                    rec("S3", "B5", "INFO", "%s ledger.jsonl co-occurrence, not accepted as a binding (schema unreviewed): %s" % (k, c))
        else:
            tri("S3", v["item"], cmp3(v["state"], reg), "%s registered model state == val_e%03d model_state_sha256" % (k, v["epoch"] or 0))

    def same(k1, k2, label):
        a, b = roles.get(k1), roles.get(k2)
        if not a or not b or not a["state"] or not b["state"]:
            rec("S3", "B5", "UNRESOLVED", "%s: a role or its registered model state is missing" % label)
            return
        rec("S3", "B5", "CONFIRMED", "%s: registered model state %s; file %s; file sha256 %s" % (
            label, "IDENTICAL" if a["state"] == b["state"] else "DISTINCT",
            "same path" if a["path"] == b["path"] else "different paths",
            "same" if CKPT_SHA.get(a["path"]) == CKPT_SHA.get(b["path"]) else "different"))
    same(("lam2", 2028, "selected"), ("lam2", 2028, "final"), "lam2/2028 selected vs final")
    same(("lam0", 2027, "selected"), ("lam0", 2027, "epoch10"), "lam0/2027 selected vs epoch 10")
    t1 = [roles[(a, s, r)]["state"] for s, a in RUNS for r in ROLES_T1 if (a, s, r) in roles]
    t2 = [roles[(a, s, "epoch%d" % e)]["state"] for s, a in RUNS for e in EPOCHS_T2 if (a, s, "epoch%d" % e) in roles]
    rec("S3", "B5", "INFO", "Tier 1: %d/18 roles resolved, %d unique registered model states (223 expects 17)" % (
        len(t1), len(set(x for x in t1 if x))))
    rec("S3", "B5", "INFO", "Tier 1+2: %d/30 roles resolved, %d unique registered model states (223 expects 28)" % (
        len(t1) + len(t2), len(set(x for x in t1 + t2 if x))))


# ---------------------------------------------------------------- S5 / S6
def s5_resources(repo, R, out_dir):
    for label, p in (("results", R), ("output", out_dir)):
        du = shutil.disk_usage(p)
        rec("S5", "INT", "INFO", "disk %s: total=%d used=%d free=%d bytes" % (label, du.total, du.used, du.free))
    tot = n = 0
    for root, _, fs in os.walk(R):
        for f in fs:
            if f.endswith(".pt"):
                tot += os.path.getsize(os.path.join(root, f))
                n += 1
    rec("S5", "INT", "INFO", "all .pt files under results/v05lt: n=%d total_bytes=%d" % (n, tot))
    prov = os.path.join(repo, "v06lt_evidence", "prov_20261114T083034Z_9S6jg3.log")
    if os.path.isfile(prov):
        rec("S5", "INT", "INFO", "earlier provenance log sha256=%s size=%d (identity only; archived separately)" % (snap(prov), SNAP[prov][1]))
    else:
        rec("S5", "INT", "INFO", "earlier provenance log not found at %s" % os.path.relpath(prov, repo))


def s6_after(repo, g):
    changed = 0
    for p, (sha, size, mt) in sorted(SNAP.items()):
        try:
            st = os.stat(p)
            if sha_file(p) != sha or st.st_size != size or st.st_mtime_ns != mt:
                changed += 1
                rec("S6", "INT", "FAIL", "CHANGED during audit: %s" % p)
        except Exception as e:
            changed += 1
            rec("S6", "INT", "FAIL", "cannot re-read %s: %s" % (p, e))
    rec("S6", "INT", "PASS" if changed == 0 else "FAIL",
        "before/after sha256, size and mtime_ns over %d inspected files: %d changed" % (len(SNAP), changed))
    if g:
        now = (git(repo, "rev-parse", "HEAD"), hashlib.sha256(git(repo, "status", "--porcelain").encode()).hexdigest())
        if now != g:
            changed += 1
        rec("S6", "INT", "PASS" if now == g else "FAIL", "Git HEAD and porcelain status unchanged")
    else:
        rec("S6", "INT", "UNRESOLVED", "Git state not compared (not established in S0)")
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
    log_path = os.path.join(out_dir, "dunc_audit_%s.log" % ts)
    LOG = open(log_path, "x", encoding="utf-8")   # refuses to overwrite
    me = os.path.abspath(__file__)
    rec("S0", "INT", "INFO", "%s | script=%s sha256=%s | argv=%s | python=%s | utc=%s" % (
        VERSION, me, sha_file(me), sys.argv[1:], sys.version.split()[0], ts))
    state = {}
    g = step("S0", "INT", s0_git, repo)
    state["git_ok"] = g is not None
    step("S1", "B1", s1_sources, repo, R, state)
    step("S1", "B1", s1_bank, repo, R, state)
    if "launch_inputs" in state:
        P = step("S4", "B3", s4_population, repo, R, state)
        step("S4", "B3", s4_rows, R, state, P)
    else:
        rec("S4", "B3", "UNRESOLVED", "launch inputs and analysis records not loaded; row checks not run")
    step("S2", "B1", s2_roles, R, state)
    if "roles" in state:
        step("S2", "B1", s2_hash_ckpts, R, state)
        step("S3", "B5", s3_identities, R, state)
    else:
        rec("S3", "B5", "UNRESOLVED", "roles not resolved; identity checks not run")
    step("S5", "INT", s5_resources, repo, R, out_dir)
    changed = step("S6", "INT", s6_after, repo, g)
    v = {i: verdict(i) for i in ITEMS}
    for i in ITEMS:
        rec("S6", "-", "VERDICT", "%s = %s %s" % (i, v[i], dict(sorted(TALLY[i].items()))))
    for m in DEFERRED:
        rec("S6", "-", "NOTE", "deferred (does not close or fail an item): %s" % m)
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
