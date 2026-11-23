# SEQREF-DLAT-BINDT v0.1 -- diag_lat_bindtest.py: synthetic tests of diag_lat_bind.py v0.1
# LIFETIME: KEEP
"""Tests of the D-LAT production binding. SYNTHETIC FILES ONLY -- NOT D-LAT EVIDENCE.
A synthetic results folder and a synthetic manifest with the structure of the registered D-UNC
F2 manifest are written into a new temporary folder; no checkpoint, no MRI data, no PyTorch.

  T-B1  a correct tree binds: 30 roles -> 28 states, shared states, 29 files, stored rows returned in order
  T-B2  manifest failures: hash, schema, pin, self-test flag, roles, states, population, stored-row keys
  T-B3  file failures: checkpoint bytes, sidecar, missing file, eligibility, launch inputs, stored rows
  T-B4  (optional, --real-manifest) the REAL manifest_f2.json binds at manifest level against the
        registered sha256; structure only, no file of the host is read

SELF-TEST EXCEPTION: this file writes synthetic and deliberately wrong inputs. None of that
exists in the binding module. Every failure is logged.

Changelog:
  v0.1 (2026-11-23) first version.
"""
import argparse
import copy
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import diag_lat_bind as B  # noqa: E402
import diag_lat_core as C  # noqa: E402
import diag_lat_stats as ST  # noqa: E402

__version__ = "0.1"
ABBR = "SEQREF-DLAT-BINDT"
LOG = logging.getLogger("diag_lat.bindtest")
BANNER = "SANDBOX SYNTHETIC TEST -- NOT HOST EVIDENCE -- NOT D-LAT EVIDENCE (synthetic files)"
SEEDS = (2026, 2027, 2028)
EPOCHS = {"epoch3": 3, "epoch10": 10, "epoch30": 30}
# selected and final epochs per (arm, seed); two coincidences as in the registered manifest:
# lam0|2027 selected == epoch 10 (one file); lam2|2028 selected == final (two files, one state)
SEL = {("lam0", 2026): 9, ("lam0", 2027): 10, ("lam0", 2028): 13, ("lam2", 2026): 55, ("lam2", 2027): 48, ("lam2", 2028): 55}
FIN = {("lam0", 2026): 49, ("lam0", 2027): 49, ("lam0", 2028): 50, ("lam2", 2026): 58, ("lam2", 2027): 49, ("lam2", 2028): 55}
N_SLICES_PER_VOL = 3
CHECKS = []


def check(tid, name, ok, note=""):
    CHECKS.append({"test": tid, "check": name, "pass": bool(ok), "note": str(note)})
    if not ok:
        LOG.error("%s FAIL: %s %s", tid, name, note)


def raises(tid, name, code, fn):
    try:
        fn()
    except C.DiagLatError as e:
        check(tid, name, e.code == code, "raised %s" % e.code)
    else:
        check(tid, name, False, "no error raised")


def h(b):
    return hashlib.sha256(b).hexdigest()


def put(path, data, with_sidecar=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    if with_sidecar:
        with open(path + ".sha256", "w") as fh:
            fh.write("%s  %s\n" % (h(data), os.path.basename(path)))
    return h(data)


def build(root):
    """Synthetic results folder and manifest. Returns (manifest dict, expected stored values)."""
    slices, ids = [], []
    for v in range(B.N_VOLUMES):
        f = "knee_singlecoil_train/singlecoil_train/file%07d.h5" % (1000002 + 7 * v)
        for i in range(N_SLICES_PER_VOL):
            mid = i == 1
            slices.append({"file": f, "slice_index": 10 + i, "midpoint": mid, "eligible": bool((v + i) % 2), "volume_id": "vol%04d" % ((v * 37) % 1000)})
            if mid:
                ids.append([f, 10 + i])
    order_sha = h(b"order")
    el = put(os.path.join(root, "inputs", "v05lt_nmse_eligibility.json"),
             json.dumps({"slices": slices, "provenance": {"validation_order_sha256": order_sha}}).encode())
    li = put(os.path.join(root, "launch", "v05lt_launch_inputs.json"), b'{"launch": true}', with_sidecar=False)
    roles, stored, expect = [], {}, {}
    for tier, names in ((1, ("epoch3", "selected", "final")), (2, ("epoch10", "epoch30"))):
        for arm in B.ARMS:
            for seed in SEEDS:
                for role in names:
                    e = EPOCHS.get(role) or (SEL if role == "selected" else FIN)[(arm, seed)]
                    state = h(("state|%s|%d|%d" % (arm, seed, e)).encode())
                    rel = ("%d_%s/final/weights_final.pt" if role == "final" else "%d_%s/ckpt/weights_e%03d.pt") % (
                        (seed, arm) if role == "final" else (seed, arm, e))
                    fsha = put(os.path.join(root, rel), ("weights|%s|%s" % (rel, state)).encode())
                    roles.append({"arm": arm, "epoch": e, "file_sha256": fsha, "model_state_sha256": state, "path": rel,
                                  "posteval": state if tier == 1 else None, "role": role, "seed": seed, "tier": tier})
                    key = "%s|%d|e%03d" % (arm, seed, e)
                    if key not in stored:
                        g = np.random.Generator(np.random.PCG64(int(h(key.encode())[:8], 16)))
                        vals = g.standard_normal((len(slices), 4))
                        rows = [{"file": s["file"], "slice_index": s["slice_index"], "psnr": float(30 + vals[q, 0]), "nll_per_dim": float(vals[q, 1]),
                                 "nll_prior_per_dim": float(vals[q, 2]), "nll_logdet_per_dim": float(vals[q, 3]), "nll_decomp_abs_dev": 0.0}
                                for q, s in enumerate(slices)]
                        stored[key] = put(os.path.join(root, "%d_%s" % (seed, arm), "ckpt", "val_e%03d.json" % e),
                                          json.dumps({"schema": B.ROWS_SCHEMA, "arm": arm, "seed": seed, "epoch": e, "model_state_sha256": state,
                                                      "n_rows": len(rows), "rows": rows}).encode())
                        expect[state] = np.array([[30 + vals[q, 0], vals[q, 1], vals[q, 2], vals[q, 3]]
                                                  for q, s in enumerate(slices) if s["midpoint"]])
    by_state = {}
    for r in roles:
        by_state.setdefault(r["model_state_sha256"], []).append("%s|%d|%s" % (r["arm"], r["seed"], r["role"]))
    states = [{"index": i, "model_state_sha256": s, "roles": sorted(v)} for i, (s, v) in enumerate(sorted(by_state.items()))]
    mf = {"schema": B.F2_SCHEMA, "selftest": False, "pin": B.PIN, "roles": roles, "states": states,
          "population": {"identities": ids, "n_midpoint": B.N_VOLUMES, "regenerated_masks_sha256": h(b"masks"), "validation_order_sha256": order_sha},
          "inputs": {"eligibility_sha256": el, "launch_inputs_sha256": li, "split_sha256": h(b"s"), "p4_sha256": h(b"p4"), "p0s_sha256": h(b"p0"),
                     "z_diag_sha256": h(b"z"), "z_manifest_sha256": h(b"zm"), "stored_rows": stored}}
    return mf, expect


def write_manifest(folder, mf, name="manifest_f2.json"):
    data = json.dumps(mf, indent=1).encode()
    p = os.path.join(folder, name)
    with open(p, "wb") as fh:
        fh.write(data)
    return p, h(data)


def main():
    ap = argparse.ArgumentParser(description="binding tests (synthetic; NOT D-LAT EVIDENCE)")
    ap.add_argument("--out", required=True, help="report file; must not exist yet")
    ap.add_argument("--real-manifest", help="optional: path of the real D-UNC manifest_f2.json (manifest-level check only)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.CRITICAL)
    if os.path.exists(args.out):
        LOG.error("report exists: %s", args.out)
        raise SystemExit("report already exists: %s" % args.out)
    check("T-B1", "this program runs with bytecode writing disabled (python -B)", sys.dont_write_bytecode)
    tmp = tempfile.mkdtemp(prefix="dlat_bindtest_")
    try:
        root = os.path.join(tmp, "results")
        mf, expect = build(root)
        mpath, msha = write_manifest(tmp, mf)
        bound = B.bind_manifest(mpath, msha)
        check("T-B1", "30 role assignments resolve to exactly 28 states; three seeds; ten registered roles",
              len(bound["role_map"]) == 30 and len(bound["states"]) == 28 and bound["seeds"] == list(SEEDS)
              and set(bound["role_map"]) == {(r, s) for r in B.ROLES for s in SEEDS} and len(B.ROLES) == 10)
        check("T-B1", "two shared states: one with one file (selected == epoch 10), one with two files (selected == final); 29 files",
              len(bound["shared_states"]) == 2 and bound["n_files"] == 29
              and bound["role_map"][("lam0/sel", 2027)] == bound["role_map"][("lam0/e10", 2027)]
              and bound["role_map"][("lam2/sel", 2028)] == bound["role_map"][("lam2/fin", 2028)]
              and sorted(len(bound["states"][s]["files"]) for s in bound["shared_states"]) == [1, 2])
        check("T-B1", "roles are the contrast roles of the statistics module; processing order == the manifest's state table",
              B.ROLES == tuple(sorted({r for w in ST.ALL_CONTRASTS.values() for r in w}))
              and bound["state_order"] == [t["model_state_sha256"] for t in mf["states"]])
        ver = B.verify_files(bound, root)
        check("T-B1", "files verified: 29 checkpoint files, 28 stored-row files, eligibility and launch inputs",
              ver["files_hashed"] == 29 + 28 + 2 and ver["n_rows_per_file"] == B.N_VOLUMES * N_SLICES_PER_VOL)
        worst = max(float(np.max(np.abs(np.stack([ver["stored"][s][n] for n in B.ROW_NAMES], axis=1) - expect[s]))) for s in expect)
        check("T-B1", "stored rows of the 28 states: 97 midpoint values each of R0_PSNR, NLL, B, J, in manifest slice order (exact)",
              set(ver["stored"]) == set(bound["states"]) and worst == 0.0
              and all(v.shape == (B.N_VOLUMES,) for st in ver["stored"].values() for v in st.values()))
        check("T-B1", "volume order is the order of the sorted volume ids; 97 distinct volumes",
              [ver["volume_ids"][q] for q in ver["volume_order"]] == sorted(ver["volume_ids"]) and len(set(ver["volume_ids"])) == B.N_VOLUMES)

        # ---- T-B2: manifest failures
        def bad_manifest(edit, name):
            m = copy.deepcopy(mf); edit(m)
            p, s = write_manifest(tmp, m, name)
            return lambda: B.bind_manifest(p, s)
        raises("T-B2", "a manifest whose sha256 is not the registered one", B.ORIGIN, lambda: B.bind_manifest(mpath, "0" * 64))
        raises("T-B2", "the default registered sha256 refuses the synthetic manifest", B.ORIGIN, lambda: B.bind_manifest(mpath))
        raises("T-B2", "a missing manifest", B.ORIGIN, lambda: B.bind_manifest(os.path.join(tmp, "none.json"), msha))
        PM = B.POPULATION_MISMATCH
        cases = (("wrong schema", B.ORIGIN, lambda m: m.update(schema="x")),
                 ("self-test manifest", B.ORIGIN, lambda m: m.update(selftest=True)),
                 ("another pin", B.ORIGIN, lambda m: m.update(pin="0" * 40)),
                 ("29 role assignments", PM, lambda m: m["roles"].pop()),
                 ("a duplicated role assignment", PM, lambda m: m["roles"].__setitem__(1, dict(m["roles"][0]))),
                 ("an unknown role name", PM, lambda m: m["roles"][0].update(role="epoch5")),
                 ("an absolute checkpoint path", PM, lambda m: m["roles"][0].update(path="/etc/passwd")),
                 ("a path leaving the results folder", PM, lambda m: m["roles"][0].update(path="../x.pt")),
                 ("30 distinct states (a shared state split)", PM, lambda m: [r for r in m["roles"] if r["role"] == "epoch10" and r["seed"] == 2027
                                                                            and r["arm"] == "lam0"][0].update(model_state_sha256=h(b"other"))),
                 ("one state under two seeds", PM, lambda m: m["roles"][3].update(model_state_sha256=m["roles"][0]["model_state_sha256"])),
                 ("one file with two hashes", PM, lambda m: [r for r in m["roles"] if r["role"] == "epoch10" and r["seed"] == 2027
                                                            and r["arm"] == "lam0"][0].update(file_sha256=h(b"x"))),
                 ("a state table that disagrees with the role table", PM, lambda m: m["states"][0].update(roles=["lam0|2026|final"])),
                 ("a state table out of order", PM, lambda m: m["states"].reverse()),
                 ("96 slice identities", PM, lambda m: m["population"]["identities"].pop()),
                 ("a duplicated slice identity", PM, lambda m: m["population"]["identities"].__setitem__(1, m["population"]["identities"][0])),
                 ("a stored-row entry missing", PM, lambda m: m["inputs"]["stored_rows"].pop(sorted(m["inputs"]["stored_rows"])[0])),
                 ("a missing input hash", B.ORIGIN, lambda m: m["inputs"].pop("z_diag_sha256")))
        for i, (name, code, edit) in enumerate(cases):
            raises("T-B2", "manifest refused -- " + name, code, bad_manifest(edit, "bad_%02d.json" % i))

        # ---- T-B3: file failures (each on a fresh copy of the tree)
        def bad_tree(edit, name):
            r2 = os.path.join(tmp, name)
            shutil.copytree(root, r2)
            edit(r2)
            return lambda: B.verify_files(bound, r2)
        first = sorted(bound["states"])[0]
        st0 = bound["states"][first]
        ck = st0["path"]
        rows_rel = os.path.join("%d_%s" % (st0["seed"], st0["arm"]), "ckpt", "val_e%03d.json" % st0["epoch"])

        def rewrite_rows(r2, edit):
            p = os.path.join(r2, rows_rel)
            j = json.load(open(p)); edit(j)
            json.dump(j, open(p, "w"))

        def rewrite_rows_registered(r2, edit):          # content changed AND hashes made consistent: only the content check can stop it
            rewrite_rows(r2, edit)
            p = os.path.join(r2, rows_rel)
            new = h(open(p, "rb").read())
            open(p + ".sha256", "w").write(new)
            bound["inputs"]["stored_rows"]["%s|%d|e%03d" % (st0["arm"], st0["seed"], st0["epoch"])] = new
        key0 = "%s|%d|e%03d" % (st0["arm"], st0["seed"], st0["epoch"])
        keep = bound["inputs"]["stored_rows"][key0]
        file_cases = (("a changed checkpoint file", B.ORIGIN, lambda r2: open(os.path.join(r2, ck), "ab").write(b"x")),
                      ("a checkpoint sidecar that names another hash", B.ORIGIN, lambda r2: open(os.path.join(r2, ck + ".sha256"), "w").write("0" * 64)),
                      ("a checkpoint without sidecar", B.ORIGIN, lambda r2: os.remove(os.path.join(r2, ck + ".sha256"))),
                      ("a missing checkpoint file", B.ORIGIN, lambda r2: os.remove(os.path.join(r2, ck))),
                      ("changed launch inputs", B.ORIGIN, lambda r2: open(os.path.join(r2, "launch", "v05lt_launch_inputs.json"), "ab").write(b" ")),
                      ("a changed eligibility manifest", B.ORIGIN, lambda r2: open(os.path.join(r2, "inputs", "v05lt_nmse_eligibility.json"), "ab").write(b" ")),
                      ("changed stored rows (hash)", B.ORIGIN, lambda r2: rewrite_rows(r2, lambda j: j["rows"][0].update(psnr=1.0))),
                      ("a missing stored-row file", B.ORIGIN, lambda r2: os.remove(os.path.join(r2, rows_rel))))
        for i, (name, code, edit) in enumerate(file_cases):
            raises("T-B3", "files refused -- " + name, code, bad_tree(edit, "tree_%02d" % i))
        content_cases = (("stored rows of another model state", PM, lambda j: j.update(model_state_sha256=h(b"zz"))),
                         ("stored rows of another epoch", PM, lambda j: j.update(epoch=j["epoch"] + 1)),
                         ("stored rows in another order", PM, lambda j: j["rows"].reverse()),
                         ("stored rows with a row missing", PM, lambda j: j["rows"].pop()),
                         ("a non-finite stored value on a midpoint slice", C.NONFINITE, lambda j: j["rows"][1].update(nll_per_dim=None)))
        for i, (name, code, edit) in enumerate(content_cases):
            raises("T-B3", "files refused (hashes consistent, content wrong) -- " + name, code,
                   bad_tree(lambda r2, e=edit: rewrite_rows_registered(r2, e), "ctree_%02d" % i))
            bound["inputs"]["stored_rows"][key0] = keep
        raises("T-B3", "a results folder that does not exist", B.ORIGIN, lambda: B.verify_files(bound, os.path.join(tmp, "nowhere")))
        check("T-B3", "the correct tree still verifies after the failing cases (nothing was changed in it)",
              B.verify_files(bound, root)["files_hashed"] == 59)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    real = None
    if args.real_manifest:
        rb = B.bind_manifest(args.real_manifest)                       # the registered sha256, the registered pin
        real = {"manifest_sha256": rb["manifest_sha256"], "roles": len(rb["role_map"]), "states": len(rb["states"]), "files": rb["n_files"],
                "seeds": rb["seeds"], "slices": len(rb["identities"]),
                "shared_states": {s: rb["states"][s]["roles"] for s in rb["shared_states"]},
                "probe_states": {r: rb["role_map"][(r, rb["seeds"][0])] for r in ("lam0/e3", "lam0/fin", "lam2/e3", "lam2/fin")}}
        check("T-B4", "the REAL D-UNC F2 manifest binds at manifest level: registered sha256 and pin, 30 roles -> 28 states, 29 files, 97 slices",
              rb["manifest_sha256"] == B.REGISTERED_F2_SHA256 and real["roles"] == 30 and real["states"] == 28 and real["files"] == 29
              and real["slices"] == 97 and rb["seeds"] == [2026, 2027, 2028], "structure only; no file of the host was read")
        check("T-B4", "its two shared states are lam0|2027 (selected == epoch 10) and lam2|2028 (selected == final)",
              sorted(sorted(v) for v in real["shared_states"].values()) == [["lam0|2027|epoch10", "lam0|2027|selected"], ["lam2|2028|final", "lam2|2028|selected"]])
    here = os.path.dirname(os.path.abspath(__file__))
    n_fail = sum(1 for c in CHECKS if not c["pass"])
    rep = {"banner": BANNER, "schema": "seqref-dlat-bindtest/1", "bindtest_version": __version__, "bind_version": B.__version__,
           "real_manifest": real, "files_sha256": {n: B.sha_file(os.path.join(here, n)) for n in ("diag_lat_bind.py", "diag_lat_bindtest.py")},
           "checks": CHECKS, "counts": {"pass": len(CHECKS) - n_fail, "fail": n_fail}, "result": "PASS" if n_fail == 0 else "FAIL"}
    with open(args.out, "x") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(BANNER)
    for t in ("T-B1", "T-B2", "T-B3", "T-B4"):
        rows = [c for c in CHECKS if c["test"] == t]
        if rows:
            print("%s  %d/%d" % (t, sum(c["pass"] for c in rows), len(rows)))
        for c in rows:
            if not c["pass"]:
                print("   FAIL %s %s" % (c["check"], c["note"]))
    print("RESULT %s (%d checks, %d failed)" % (rep["result"], len(CHECKS), n_fail))
    raise SystemExit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
