# SEQREF-DUNC-F2M v0.1 -- diag_unc_f2m.py: builds manifest_f2.json from the verified F1 evidence (plan SEQREF-DUNC-F2G v0.2.2, F2-3 / F2-4 step 3)
# LIFETIME: KEEP
"""F2 manifest builder. Run by the author AFTER the code archive commit C1 exists; reads no model and no MRI
file. Inputs: manifest_f1.json and reproduction.json of the F1 run (hashes given and checked), the six code
files of C1, the commit identity C1, the reviewed commands. Output: manifest_f2.json (canonical bytes) and
its sha256. The manifest never names its own commit (C2).

Decisions written here are the ones approved in plan F2G v0.2.2 and LudWiki 229 U58: both tiers under rule
(i) of 223 §1, checked again from the F1 projections; the stop limit is the unrounded F1 value. Nothing is
derived from a scientific result (none exists). Every error stops the build; nothing is defaulted.

Changelog: v0.1 (2026-11-18) first version.
"""
import argparse
import hashlib
import json
import sys

VERSION = "SEQREF-DUNC-F2M v0.1"
PLAN = "SEQREF-DUNC-F2G v0.2.2 (8ab33005e2dac06ac6282a7d9e7c7921e63274afa9a70ad83f22618215337bdb)"
MODULES = ("diag_unc_adapter.py", "diag_unc_core.py", "diag_unc_stats.py", "diag_unc_guard.py", "diag_unc_f1.py", "diag_unc_g.py")
Z_MANIFEST = "c4d2c0ec7c908f7c20a4f287eeffbdf966924a5118b6ba734b0a329942ab7e5f"     # registered (F1 runner constant; verified by F1 gate BANK)
TABLES = ["t01_dgap", "t02_spread_width", "t03a_coverage_checkpoint", "t03b_coverage_contrast", "t04_scores", "t05_band", "t06_ssr", "t07_unc_error",
          "t08a_mc_subsets", "t08b_mc_batch_means", "t09_matrix", "t10_cross"]
FIGDATA = ["fd_histories", "fd_dgap", "fd_rankhist", "fd_coverage", "fd_crps", "fd_band", "fd_maps", "fd_convergence", "fd_deciles"]
TIER1 = ["S-GAP0", "S-CTL0", "C-TIME0", "S-CTL2", "C-TIME2", "S-GAP2", "C-OBJ", "C-OBJ@3", "X-TIME", "X-CTL"]
TIER2 = ["C-OBJ@10", "C-OBJ@30"] + ["%s(%s)" % (p, t) for t in ("3→10", "10→30", "3→30") for p in ("F-TRAJ0", "F-TRAJ2", "X-FIX")]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(path, want):
    with open(path, "rb") as fh:
        raw = fh.read()
    if sha(raw) != want:
        raise SystemExit("ERROR: %s has sha256 %s, expected %s" % (path, sha(raw), want))
    return json.loads(raw.decode("utf-8"))


def build(f1_manifest, f1_manifest_sha, repro, repro_sha, code_dir, c1, f1_run_dir, f1_result_line, interpreter, commands, approvals, selftest=False,
          stop_override=None, tol_override=None):
    mf = read(f1_manifest, f1_manifest_sha)
    if mf.get("schema") != "seqref-dunc-f1-manifest/1" or mf.get("result") != "PASS" or bool(mf.get("selftest")) != bool(selftest):
        raise SystemExit("ERROR: the F1 manifest is not a PASS manifest of the expected kind")
    if mf["reproduction"]["sha256"] != repro_sha:
        raise SystemExit("ERROR: reproduction.json is not the file named by the F1 manifest")
    rp = read(repro, repro_sha)
    tols = {x["dc_a_tol"] for v in rp["roles"].values() for x in v}
    if rp.get("partial") is not False or len(tols) != 1:
        raise SystemExit("ERROR: reproduction.json is partial or carries more than one DC tolerance")
    V = mf["inputs"]
    code = {}
    for name in MODULES:
        with open("%s/%s" % (code_dir, name), "rb") as fh:
            code[name] = sha(fh.read())
    for name in ("diag_unc_adapter.py", "diag_unc_core.py", "diag_unc_stats.py", "diag_unc_guard.py", "diag_unc_f1.py"):
        if code[name] != mf["code"][name]:
            raise SystemExit("ERROR: %s differs from the file F1 ran (unchanged modules are required)" % name)
    roles = [dict(arm=r["arm"], seed=r["seed"], role=r["role"], epoch=r["epoch"], path=r["path"], file_sha256=r["file_sha256"],
                  model_state_sha256=r["model_state_sha256"], posteval=r["posteval"], tier=1) for r in V["roles"]]
    roles += [dict(arm=r["arm"], seed=r["seed"], role=r["role"], epoch=r["epoch"], path=r["path"], file_sha256=r["file_sha256"],
                   model_state_sha256=r["model_state_sha256"], posteval=None, tier=2) for r in V["tier2_roles"]]
    t1 = [s["model_state_sha256"] for s in mf["states"]]
    order = []                                               # processing order: the states that only Tier 2 uses first, so that the first load of a
    for r in roles:                                          # milestone file F1 did not load (epochs 10 and 30) happens at the start of stage 1
        if r["model_state_sha256"] not in t1 and r["model_state_sha256"] not in order:
            order.append(r["model_state_sha256"])
    order += t1
    states = [{"index": i, "model_state_sha256": st, "roles": sorted("%s|%d|%s" % (r["arm"], r["seed"], r["role"]) for r in roles if r["model_state_sha256"] == st)}
              for i, st in enumerate(order)]
    use, lim = mf["projection"]["for_f2"], mf["projection"]["registered_limits"]
    free = mf["s3_measured"]["free_bytes_at_end"]
    both = use["tier1_tier2"]
    if not (len(roles) == 30 and len(states) == 28 == mf["decode_counts"]["tier1_tier2"]["states"] and both["within_registered_limits"] is True
            and both["T_s"] <= lim["g_wall_s"] and both["D_bytes"] <= lim["g_disk_bytes"] and free - both["D_bytes"] >= lim["reserve_bytes"]
            and both["T_s"] == mf["proposed_stop_limit"]["tier1_tier2_wall_s"]):
        raise SystemExit("ERROR: 30 roles / 28 states or rule (i) of 223 §1 (Tier 1 + Tier 2 projection within the registered limits) does not hold")
    stop = {"wall_s": both["T_s"], "disk_bytes": both["D_bytes"], "reserve_bytes": lim["reserve_bytes"]}
    tol = {"r0_psnr_db": 1e-3, "nll_abs": 1e-4, "r1_psnr_db": 1e-3, "r1_nmse_rel": 1e-4, "dc_a": tols.pop(),
           "nll_identity_rel": 1e-6, "nll_identity_abs": 1e-12, "share_min_abs_dnll": 1e-3,
           "source": "223 §3.3 / 216 §1.3 (reproduction); the registered dc_tolerances.a of the G-U1 records; 216 §3 (identity, share rule)"}
    if stop_override or tol_override:                        # self-test only: the harness exercises limits and the share rule with other values
        if not selftest:
            raise SystemExit("ERROR: an override is a self-test facility")
        stop.update(stop_override or {})
        tol.update(tol_override or {})
    boot = mf["bootstrap_index_matrix"]
    return {
        "schema": "seqref-dunc-f2-manifest/1", "builder": VERSION, "plan": PLAN, "selftest": bool(selftest),
        "spec": "LudWiki 223 SEQREF-MRI-DIAG-UNC v1.0 (FROZEN) + AM1; LudWiki 216 SEQREF-MRI-DIAG-JSH v1.0 (FROZEN) §3 and §6 as restated in plan §6",
        "f1": {"run_dir": f1_run_dir, "manifest_sha256": f1_manifest_sha, "result_line": f1_result_line, "reproduction_sha256": repro_sha,
               "record": "LudWiki 229 U57"},
        "pin": mf["pin"], "code_commit": c1, "code": code, "interpreter": interpreter,
        "inputs": {k: V[k] for k in ("analysis_sha256", "launch_inputs_sha256", "eligibility_sha256", "split_sha256", "p4_sha256", "p0s_sha256", "z_diag_sha256",
                                     "posteval", "ledgers", "final_weights", "stored_rows")} | {"z_manifest_sha256": Z_MANIFEST},
        "history_rows": {"rule": "every ckpt/val_e<NNN>.json of the six runs, epochs 1..final of the verified ledgers; hashed, recorded and checked in G1 "
                                 "before any decode; a missing or changed file is HOLD_PROVENANCE"},
        "roles": roles, "states": states,
        "population": {"n_midpoint": mf["population"]["n_midpoint"], "identities": mf["population"]["identities"],
                       "regenerated_masks_sha256": mf["population"]["regenerated_masks_sha256"], "validation_order_sha256": V["population"]["validation_order_sha256"]},
        "tier": {"decision": "both", "rule": "223 §1 activation rule, case (i): the Tier 1 + Tier 2 projection is within the registered limits",
                 "projections": use, "owner_limits": lim, "free_bytes_at_f1": free, "note": "resource evidence only; never revisited after G starts"},
        "stop_limit": stop,
        "decode_counts": mf["decode_counts"]["tier1_tier2"],
        "bootstrap": {"seed": boot["seed"], "shape": boot["shape"], "dtype": boot["dtype"], "sha256": boot["sha256"], "generator": boot["generator"],
                      "numpy": boot["numpy"], "deviation": "D1 (LudWiki 228 U52): the same matrix was generated in the sandbox before F1; the F1 value is the registered one"},
        "contrasts": {"tier1": TIER1, "tier2": TIER2},
        "tolerances": tol,
        "fixed_slices": {"rule": "216 §6: sorted validation volume IDs; numpy.random.default_rng(seed).choice(V, n, replace=False); the midpoint slice of "
                                 "each drawn volume; derived in G", "seed": boot["seed"] + 1, "n": 4},
        "commands": commands,
        "output_inventory": {"tables": TABLES, "figdata": FIGDATA, "other": ["results.json", "records/", "maps/", "manifest_g.json"],
                             "pending_after_g": ["G-FIG: every mechanical figure of 223 §9", "result report with the D-JAC / D-SPEC / D-UNC synthesis diagram"]},
        "exposure": "log and terminal: identities, counts, deviations, pass flags, times and bytes only; scientific values only under out/; first viewing of "
                    "results is a separate reviewed step after the integrity checks",
        "approvals": approvals,
    }


def canonical(doc):
    return json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def main():
    ap = argparse.ArgumentParser()
    for a in ("f1-manifest", "f1-manifest-sha", "repro", "repro-sha", "code-dir", "code-commit", "f1-run-dir", "f1-result-line", "interpreter-json",
              "commands-json", "approvals-json", "out"):
        ap.add_argument("--" + a, required=True)
    a = ap.parse_args()
    load = lambda p: json.load(open(p, "r", encoding="utf-8"))
    doc = build(a.f1_manifest, a.f1_manifest_sha, a.repro, a.repro_sha, a.code_dir, a.code_commit, a.f1_run_dir, a.f1_result_line,
                load(a.interpreter_json), load(a.commands_json), load(a.approvals_json))
    raw = canonical(doc)
    with open(a.out, "xb") as fh:
        fh.write(raw)
    print("manifest_f2.json sha256 %s (%d bytes)" % (sha(raw), len(raw)))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR: %r" % (exc,), file=sys.stderr)
        raise
