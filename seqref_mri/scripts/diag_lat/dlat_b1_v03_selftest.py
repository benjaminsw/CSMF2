# SEQREF-DLATB1T v0.3.3 -- dlat_b1_v03_selftest.py: synthetic fixtures for dlat_b1_v03.py
# LIFETIME: DIAGNOSTIC
"""Builds small synthetic trees (30 roles / 29 files / 28 states, six ledgers, row files, both
manifests, a git repository) in a temporary folder and runs the checker on them. Touches nothing
outside that folder. The registered constants of the checker (pin, expected manifest hashes,
source table) are replaced by the synthetic ones IN THIS PROCESS ONLY; the checker file has no
test switch.

A case is judged on its FAIL and UNRESOLVED items and on the checker's exit status.
Starting environment: the forbidden modules already loaded BEFORE this harness imports the checker
are recorded (BASELINE). If there are any, the checker's 'Z0 startup modules' item is a FAIL in
every case; the harness then sets that one item aside, says so, and requires that it names exactly
the baseline. 'Z1 imports' (modules loaded during the check) is never set aside.
Exit status: 0 = every case behaved as expected; 1 = at least one did not. Every failure is logged.

Changelog:
  v0.3.3 (2026-11-20) one case added with the checker v0.3.3 (16 cases): a role checkpoint whose
       sha256 occurs in the ledger only outside the retained sets must be reported as INFO
       (no PASS, no FAIL, no UNRESOLVED for that file). A case may now name an INFO text
       that must be present.
  v0.3.2 (2026-11-20) three ledger-history cases added with the checker v0.3.2 (15 cases):
       role checkpoints superseded by a later commit (must pass); a role checkpoint the ledger
       never mentions (UNRESOLVED); a name recorded with another sha256 in the history (FAIL).
  v0.3.1 (2026-11-20) bug fix after review: accounts for the starting environment (see above);
       the GPU item is no longer ignored (the checker now reports it as INFO); the valid case
       must exit 0 in a clean interpreter.
  v0.3 (2026-11-20) first version, with the checker v0.3.
"""
import copy
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile

BASELINE = sorted({m.split(".")[0] for m in sys.modules if m.split(".")[0] in ("torch", "numpy", "h5py", "seqref_mri")})
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("dlat_b1_v03", os.path.join(HERE, "dlat_b1_v03.py"))
B1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B1)

LOG = logging.getLogger("dlat_b1_selftest")
RUNS = [(arm, seed) for arm in ("lam0", "lam2") for seed in (2026, 2027, 2028)]
SRC = {"seqref_mri/src/a.py": b"a\n", "seqref_mri/scripts/b.py": b"b\n"}
CODE = {"diag_unc_adapter.py": b"adapter\n", "diag_unc_core.py": b"core\n"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def sh(cwd, *args):
    subprocess.run(list(args), cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def build(tmp, edit_f2=None, after=None, second_root=False, ledger_mode=None):
    """-> argv for the checker. edit_f2(f2) changes the manifest before it is hashed and bound to G;
    after(ctx) changes the finished tree. ledger_mode shapes the ledger of run 2026_lam0 only:
    'superseded' = a later commit retains only the epoch-30 file; 'omit' = no commit names the
    epoch-3 file; 'wrong_history' = an earlier commit names the epoch-3 file with another sha256;
    'hash_only' = no commit names the epoch-3 file, but a non-commit record carries its sha256."""
    repo = os.path.join(tmp, "repo")
    runs = os.path.join(repo, "seqref_mri", "results", "v05lt", "runs")
    dunc = os.path.join(repo, "v06lt_evidence", "DIAG_UNC")
    for rel, data in SRC.items():
        write(os.path.join(repo, rel), data)
    for name, data in CODE.items():
        write(os.path.join(repo, "seqref_mri", "scripts", "diag_unc", name), data)
        write(os.path.join(dunc, B1.G_RUN, "code", name), data)
    sh(repo, "git", "init", "-q")
    sh(repo, "git", "add", "seqref_mri/src", "seqref_mri/scripts")
    sh(repo, "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "synthetic")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, stdout=subprocess.PIPE).stdout.decode().strip()

    roles, ledgers, rows = [], {}, {}
    for arm, seed in RUNS:
        run = "%d_%s" % (seed, arm)
        sel = 10 if (arm, seed) == ("lam0", 2027) else 9          # one run: selected = epoch 10 (same file twice)
        plan = [("epoch3", 3, 1), ("selected", sel, 1), ("final", 49, 1), ("epoch10", 10, 2), ("epoch30", 30, 2)]
        retained = {}
        for role, epoch, tier in plan:
            final = role == "final"
            rel = "%s/final/weights_final.pt" % run if final else "%s/ckpt/weights_e%03d.pt" % (run, epoch)
            same_state = final and (arm, seed) == ("lam2", 2028)   # one run: final = the selected state, other bytes
            data = ("%s|%s|%s" % (run, "final-bytes" if final else epoch, "w")).encode()
            state = sha(("state|%s|%s" % (run, sel if same_state else ("final" if final else epoch))).encode())
            if same_state:
                epoch = sel
            write(os.path.join(runs, rel), data)
            if not final:
                write(os.path.join(runs, rel + ".sha256"), ("%s  %s\n" % (sha(data), os.path.basename(rel))).encode())
                retained["e%03d" % epoch] = {"name": os.path.basename(rel), "sha256": sha(data)}
            roles.append({"arm": arm, "seed": seed, "epoch": epoch, "role": role, "tier": tier, "path": rel,
                          "file_sha256": sha(data), "model_state_sha256": state, "posteval": None})
            row = json.dumps({"rows": [{"nll_prior_per_dim": 1.0, "nll_logdet_per_dim": 2.0}]}).encode()
            write(os.path.join(runs, run, "ckpt", "val_e%03d.json" % epoch), row)
            rows["%s|%d|e%03d" % (arm, seed, epoch)] = sha(row)
        lines, prev = [], B1.GENESIS
        plan_l = [("start", {"run": run}), ("commit", {"retained": retained, "superseded": [], "store": "ckpt"})]
        if run == "2026_lam0" and ledger_mode == "superseded":
            plan_l.append(("commit", {"retained": {"e030": retained["e030"]}, "store": "ckpt",
                                      "superseded": sorted(v["name"] for k, v in retained.items() if k != "e030")}))
        elif run == "2026_lam0" and ledger_mode in ("omit", "hash_only"):
            plan_l[1] = ("commit", {"retained": {k: v for k, v in retained.items() if k != "e003"}, "superseded": [], "store": "ckpt"})
            if ledger_mode == "hash_only":
                plan_l.append(("note", {"text": "sha256 seen", "sha256": retained["e003"]["sha256"]}))
        elif run == "2026_lam0" and ledger_mode == "wrong_history":
            plan_l.insert(1, ("commit", {"retained": {"e003": {"name": retained["e003"]["name"], "sha256": sha(b"other bytes")}},
                                         "superseded": [], "store": "ckpt"}))
        for n, (kind, body) in enumerate(plan_l):
            rec = {"schema": B1.LEDGER_SCHEMA, "seq": n, "prev": prev, "time_utc": "t", "kind": kind, "body": body}
            rec["self"] = sha(B1.canonical_json(rec))
            line = B1.canonical_json(rec)
            lines.append(line)
            prev = sha(line)
        led = b"\n".join(lines) + b"\n"
        write(os.path.join(runs, run, "ledger.jsonl"), led)
        ledgers["%s|%d" % (arm, seed)] = sha(led)
    by_state = {}
    for r in roles:
        by_state.setdefault(r["model_state_sha256"], []).append("%s|%s|%s" % (r["arm"], r["seed"], r["role"]))
    f2 = {"schema": B1.F2_SCHEMA, "pin": commit, "selftest": False, "roles": roles, "code_commit": commit,
          "code": {n: sha(d) for n, d in CODE.items()}, "decode_counts": {"states": len(by_state)},
          "states": [{"index": i, "model_state_sha256": k, "roles": v} for i, (k, v) in enumerate(sorted(by_state.items()))],
          "inputs": {"ledgers": ledgers, "stored_rows": rows}}
    if edit_f2:
        edit_f2(f2)
    f2_bytes = json.dumps(f2, indent=1, sort_keys=True).encode()
    g = {"schema": B1.G_SCHEMA, "pin": commit, "mode": "g", "selftest": False, "result": "PASS",
         "manifest_f2_sha256": sha(f2_bytes), "code_commit": commit, "code": f2["code"],
         "inputs_verified": {"roles": f2["roles"], "ledgers": f2["inputs"]["ledgers"], "history_rows": dict(rows)}}
    g_bytes = json.dumps(g, indent=1, sort_keys=True).encode()
    f2_path = os.path.join(dunc, B1.G_RUN, "code", "manifest_f2.json")
    g_path = os.path.join(dunc, B1.G_RUN, "out", "manifest_g.json")
    write(f2_path, f2_bytes)
    write(g_path, g_bytes)
    write(os.path.join(dunc, B1.G_RUN, "RESULT.txt"), ("G_RESULT: PASS\nmanifest_g sha256 %s\n" % sha(g_bytes)).encode())
    if second_root:
        for arm, seed in RUNS:
            os.makedirs(os.path.join(repo, "seqref_mri", "results", "v05lt", "copy", "%d_%s" % (seed, arm)))
    B1.PIN = commit
    B1.PIN_FILES = {p: (sha(d), "synthetic") for p, d in SRC.items()}
    B1.EXPECT_F2_SHA, B1.EXPECT_G_SHA = sha(f2_bytes), sha(g_bytes)
    ctx = {"repo": repo, "runs": runs, "dunc": dunc, "f2_path": f2_path, "g_path": g_path, "f2": f2}
    if after:
        after(ctx)
    return ["--repo", repo]


def problems():
    """-> (problems with the startup item set aside if the baseline explains it, problems as reported)."""
    raw = sorted({(it["check"], it["status"]) for it in B1.ITEMS if it["status"] in ("FAIL", "UNRESOLVED")})
    z0 = [it for it in B1.ITEMS if it["check"] == "Z0 startup modules"]
    explained = (bool(BASELINE) and len(z0) == 1 and z0[0]["status"] == "FAIL"
                 and sorted({m.split(".")[0] for m in B1.PRELOADED}) == BASELINE)
    return [p for p in raw if not (explained and p == ("Z0 startup modules", "FAIL"))], raw


# ---- mutations
def dup_state(f2):  # two roles of different files share one model state that the state list does not show
    f2["roles"][0]["model_state_sha256"] = f2["roles"][1]["model_state_sha256"]


def escape_path(f2):
    f2["roles"][0]["path"] = "../outside.pt"


def rm_final(c):
    os.remove(os.path.join(c["runs"], "2026_lam0", "final", "weights_final.pt"))


def wrong_bytes(c):
    with open(os.path.join(c["runs"], "2026_lam2", "ckpt", "weights_e030.pt"), "ab") as fh:
        fh.write(b"x")


def make_symlink(c):
    p = os.path.join(c["runs"], "2027_lam2", "ckpt", "weights_e003.pt")
    shutil.move(p, p + ".real")
    os.symlink(p + ".real", p)


def alter_ledger(c):
    p = os.path.join(c["runs"], "2028_lam0", "ledger.jsonl")
    with open(p, "rb") as fh:
        data = fh.read()
    write(p, data.replace(b'"run":"2028_lam0"', b'"run":"2028_lamX"'))


def alter_ledger_registered(c):
    """Altered record AND the registered ledger hash follows it: only the chain check can see it."""
    alter_ledger(c)
    with open(os.path.join(c["runs"], "2028_lam0", "ledger.jsonl"), "rb") as fh:
        new = sha(fh.read())
    f2 = copy.deepcopy(c["f2"])
    f2["inputs"]["ledgers"]["lam0|2028"] = new
    f2_bytes = json.dumps(f2, indent=1, sort_keys=True).encode()
    with open(c["g_path"], "rb") as fh:
        g = json.load(fh)
    g["manifest_f2_sha256"] = sha(f2_bytes)
    g["inputs_verified"]["ledgers"] = f2["inputs"]["ledgers"]
    g_bytes = json.dumps(g, indent=1, sort_keys=True).encode()
    write(c["f2_path"], f2_bytes)
    write(c["g_path"], g_bytes)
    write(os.path.join(c["dunc"], B1.G_RUN, "RESULT.txt"), ("G_RESULT: PASS\n%s\n" % sha(g_bytes)).encode())
    B1.EXPECT_F2_SHA, B1.EXPECT_G_SHA = sha(f2_bytes), sha(g_bytes)


def edit_f2_after(c):  # the F2 file on disk no longer is the registered one, nor the one G was run from
    with open(c["f2_path"], "ab") as fh:
        fh.write(b"\n")


def invalid_record(c):
    write(os.path.join(c["dunc"], B1.G_RUN, "RESULT_INVALID.txt"), b"x")


def row_changed(c):
    with open(os.path.join(c["runs"], "2026_lam0", "ckpt", "val_e003.json"), "ab") as fh:
        fh.write(b" ")


CASES = [  # name, build kwargs, checks that must appear with that status, exact (True) or at least (False)
    ("valid 30 roles / 29 files / 28 states", {}, [], True),
    ("duplicated state identity", {"edit_f2": dup_state}, [("T1 state mapping", "FAIL"), ("T2 states on disk", "FAIL")], True),
    ("role path leaves the runs root", {"edit_f2": escape_path}, [("K1 roles", "FAIL"), ("K2 file sha256", "FAIL")], False),
    ("missing final checkpoint", {"after": rm_final}, [("K2 file sha256", "FAIL"), ("K4 summary", "FAIL"), ("T2 states on disk", "FAIL")], True),
    ("checkpoint bytes changed", {"after": wrong_bytes}, [("K2 file sha256", "FAIL"), ("K3 sidecar", "FAIL"), ("K4 summary", "FAIL"),
                                                          ("L3 ledger history", "FAIL"), ("T2 states on disk", "FAIL")], True),
    ("role file is a symlink", {"after": make_symlink}, [("K2 file sha256", "FAIL"), ("K4 summary", "FAIL"), ("T2 states on disk", "FAIL")], True),
    ("ledger altered", {"after": alter_ledger}, [("L1 ledger", "FAIL")], True),
    ("ledger altered, registered hash follows", {"after": alter_ledger_registered}, [("L2 ledger chain", "FAIL")], True),
    ("F2 manifest differs from the registered one", {"after": edit_f2_after}, [("M1 F2 manifest", "FAIL"), ("M3 F2-G relationship", "FAIL")], True),
    ("invalidation record in the G run folder", {"after": invalid_record}, [("M4 G result file", "FAIL")], True),
    ("stored row file changed", {"after": row_changed}, [("R1 stored rows", "FAIL")], True),
    ("two candidate runs roots", {"second_root": True}, [("K0 runs root", "UNRESOLVED")], False),
    ("role checkpoints superseded by a later commit", {"ledger_mode": "superseded"}, [], True),
    ("role checkpoint never mentioned by the ledger", {"ledger_mode": "omit"}, [("L3 ledger history", "UNRESOLVED")], True),
    ("ledger history names a file with another sha256", {"ledger_mode": "wrong_history"}, [("L3 ledger history", "FAIL")], True),
    ("sha256 occurs in the ledger outside retained sets", {"ledger_mode": "hash_only"}, [], True,
     "2026_lam0/ckpt/weights_e003.pt: its sha256 occurs in the ledger text, but in no retained set"),
]


def main():
    LOG.setLevel(logging.INFO)
    LOG.propagate = False          # the checker resets the root handlers on every run
    LOG.addHandler(logging.StreamHandler(sys.stdout))
    saved = (B1.PIN, B1.PIN_FILES, B1.EXPECT_F2_SHA, B1.EXPECT_G_SHA)
    LOG.info("SEQREF-DLATB1T v0.3.3 | checker v%s | python %s", B1.__version__, sys.version.split()[0])
    if BASELINE:
        LOG.info("starting environment: %s already loaded before the checker was imported; the checker's "
                 "'Z0 startup modules' FAIL is set aside in every case (use python3 -I -S for a clean start)", BASELINE)
    else:
        LOG.info("starting environment: clean (no forbidden module loaded before the checker was imported)")
    failed = 0
    for case in CASES:
        name, kwargs, expect, exact = case[:4]
        must_info = case[4] if len(case) > 4 else None
        tmp = tempfile.mkdtemp(prefix="dlat_b1_selftest_")
        try:
            argv = build(tmp, **kwargs)
            logging.disable(logging.CRITICAL)      # the checker's own lines are in its report.log
            code = B1.main(argv)
            logging.disable(logging.NOTSET)
            got, raw = problems()
            ok = (got == sorted(expect)) if exact else all(e in got for e in expect)
            ok = ok and code == (1 if raw else 0)   # exit status must follow the reported items
            if must_info is not None:               # the named file must be INFO and must have no PASS item
                l3 = [it for it in B1.ITEMS if it["check"] == "L3 ledger history" and "2026_lam0/ckpt/weights_e003.pt" in it["text"]]
                ok = ok and len(l3) == 1 and l3[0]["status"] == "INFO" and must_info in l3[0]["text"]
            if not ok:
                failed += 1
                LOG.error("SELFTEST FAIL: %s | exit %s | got %s | expected %s%s", name, code, got, sorted(expect), "" if exact else " (at least)")
            else:
                LOG.info("selftest ok: %-46s exit %d -> %s", name, code, got or "no FAIL, no UNRESOLVED")
        except Exception as exc:
            logging.disable(logging.NOTSET)
            failed += 1
            LOG.exception("SELFTEST ERROR: %s: %s: %s", name, type(exc).__name__, exc)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            B1.PIN, B1.PIN_FILES, B1.EXPECT_F2_SHA, B1.EXPECT_G_SHA = saved
    LOG.info("SELFTEST RESULT: %d of %d cases as expected", len(CASES) - failed, len(CASES))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
