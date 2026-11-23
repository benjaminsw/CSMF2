# SEQREF-DLAT-POPT v0.1 -- diag_lat_poptest.py: synthetic tests of diag_lat_population.py v0.1 (plan SEQREF-DLAT-POP v0.1)
# LIFETIME: KEEP
"""Tests of the D-LAT population stage. SYNTHETIC NUMBERS ONLY -- NOT D-LAT EVIDENCE.
No model, no checkpoint, no MRI data, no PyTorch. 97 synthetic volumes, the ten registered
roles, three seeds, 28 deduplicated model states (two roles share a state in two seeds), and
THE FULL REGISTERED BOOTSTRAP MATRIX (seed 20261120, 10,000 x 97, one call).

  T-P1  build: role map, deduplicated states, schema, order (C1)
  T-P2  classified results equal a direct run_registered call on an independently built population
  T-P3  reported quantities: 25 keys x 21 contrasts, no label; one effect and interval recomputed by hand
  T-P4  state means, per seed and pooled
  T-P5  E_cov from stored z_proj equals the core function on the unprojected codes
  T-P6  L6 on the seven read contrasts and three seeds
  T-P7  "not computed": propagation into contrasts, classes, main table, state means, L6, events
  T-P8  typed stops
  T-P9  full workload: timing, peak memory, determinism, JSON round trip, main table completeness

SELF-TEST EXCEPTION: this file builds synthetic and deliberately wrong inputs. None of that
exists in the population module. The identity of the bootstrap matrix is computed here and is
NOT yet the registered value: F1 records it (235 section 3.2). Every failure is logged.

Changelog:
  v0.1 (2026-11-23) first version.
"""
import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import diag_lat_adapter as A  # noqa: E402
import diag_lat_core as C  # noqa: E402
import diag_lat_guard as GD  # noqa: E402
import diag_lat_population as PO  # noqa: E402
import diag_lat_stats as ST  # noqa: E402

__version__ = "0.1"
ABBR = "SEQREF-DLAT-POPT"
LOG = logging.getLogger("diag_lat.poptest")
BANNER = "SANDBOX SYNTHETIC TEST -- NOT HOST EVIDENCE -- NOT D-LAT EVIDENCE (synthetic numbers)"
FIXTURE_SEED = 515151                     # fixtures only; not a registered seed
SEEDS = (2026, 2027, 2028)
N = ST.N_VOLUMES
LABELS = list(ST.STABILITY_SUBSETS)
VOLS = ["v%03d" % i for i in range(N)]
# two shared states: seed 2026 lam0/sel == lam0/e30; seed 2027 lam2/sel == lam2/fin  -> 30 - 2 = 28
SHARED = {("lam0/sel", 2026): ("lam0/e30", 2026), ("lam2/sel", 2027): ("lam2/fin", 2027)}
LEVEL = {"lam0/e3": 0.0, "lam0/e10": 0.05, "lam0/e30": 0.3, "lam0/sel": 0.1, "lam0/fin": 0.5,
         "lam2/e3": 0.0, "lam2/e10": 0.02, "lam2/e30": 0.04, "lam2/sel": 0.03, "lam2/fin": 0.1}
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


def role_map():
    return {(r, s): "%s@%d" % SHARED.get((r, s), (r, s)) for r in PO.ROLES for s in SEEDS}


def make_records(z_states):
    """records[state id] = 97 records with the full schema. Values: a role level, a volume term
    and a key term; P_swap mirrored (predicted to fall). z_states: {state id: (97, d) codes}
    for the states whose z_proj comes from real projections; the others get a direct draw."""
    g = np.random.Generator(np.random.PCG64(FIXTURE_SEED))
    tau = np.abs(g.standard_normal(N)) * 0.01
    q_tail = tau * 30.0
    proj = C.projection_matrix(g.standard_normal((64, C.M_PROJ)))
    recs = {}
    for sid in sorted(set(role_map().values())):
        role, seed = sid.split("@")[0], int(sid.split("@")[1])
        lv = LEVEL[role]
        noise = g.standard_normal((N, len(PO.SCALAR_KEYS), len(LABELS))) * 0.05
        vol = np.sin(np.arange(N) * (1.3 + SEEDS.index(seed)))
        zp = (z_states[sid] @ proj) if sid in z_states else g.standard_normal((N, C.M_PROJ))
        rows = []
        for v in range(N):
            rec, bank = {}, {lab: {} for lab in LABELS}
            for ki, k in enumerate(PO.SCALAR_KEYS):
                sign = -1.0 if k["name"] == "P_swap" else 1.0
                base = sign * lv * (1.0 + 0.1 * ki) + 0.3 * vol[v] + 0.01 * ki
                if k["name"] == "NLL":
                    base = 2.0 - lv + 0.3 * vol[v]
                if k["name"] == "R0_PSNR":
                    base = 30.0 - 3.0 * lv + 0.5 * vol[v]
                if k["name"] in ("tau", "q_tail"):
                    rec[k["name"]] = float(tau[v] if k["name"] == "tau" else q_tail[v])
                elif k["bank"]:
                    for li, lab in enumerate(LABELS):
                        bank[lab][k["name"]] = float(base + noise[v, ki, li])
                else:
                    rec[k["name"]] = float(base + noise[v, ki, 0])
            rec["z_proj"] = [float(x) for x in zp[v]]
            rec[A.BANK_FIELD] = bank
            rows.append(rec)
        recs[sid] = rows
    return recs, proj


def direct_population(recs, rmap):
    """An independent construction: role-level arrays set directly, without build_population."""
    pop = ST.Population(VOLS)
    for (role, seed), sid in rmap.items():
        for name in ("KS", "P_swap", "NLL", "R0_PSNR"):
            pop.set(role, seed, name, [r[name] for r in recs[sid]])
        for name in ("log_gamma_ratio", "d01", "a01", "TC_G", "phys_lgr", "phys_d01", "phys_a01"):
            for lab in LABELS:
                pop.set(role, seed, name, [r[A.BANK_FIELD][lab][name] for r in recs[sid]], lab)
    return pop.seal()


def hashlib_sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def dumps(obj):
    return json.dumps(obj, sort_keys=True, default=str)


def main():
    ap = argparse.ArgumentParser(description="population-stage tests (synthetic; NOT D-LAT EVIDENCE)")
    ap.add_argument("--out", required=True, help="report file; must not exist yet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.CRITICAL)      # expected typed stops are exercised on purpose
    if os.path.exists(args.out):
        LOG.error("report exists: %s", args.out)
        raise SystemExit("report already exists: %s" % args.out)
    check("T-P9", "this program runs with bytecode writing disabled (python -B)", sys.dont_write_bytecode)
    g = np.random.Generator(np.random.PCG64(FIXTURE_SEED + 7))
    rmap = role_map()
    z_states = {sid: g.standard_normal((N, 64)) for sid in ("lam0/fin@2026", "lam2/e3@2028")}
    recs, proj = make_records(z_states)
    t0 = time.perf_counter()
    idx = ST.bootstrap_matrix()                       # the single registered call (seed 20261120)
    sha = ST.matrix_identity(idx)
    t_matrix = time.perf_counter() - t0
    check("T-P9", "bootstrap matrix: registered seed, 10,000 x 97, int64", idx.shape == (ST.N_BOOT, N) and idx.dtype == np.int64
          and ST.DIAG_LAT_SEED == 20261120, "sha256 %s (computed here; registered by F1)" % sha)

    # ---- T-P1
    built = PO.build_population(recs, rmap, VOLS, SEEDS)
    pop = built["pop"]
    check("T-P1", "28 distinct states for 30 role-seed columns; ten registered roles; three seeds",
          len(built["states"]) == 28 and len(rmap) == 30 and len(PO.ROLES) == 10 and pop.seeds == list(SEEDS)
          and sorted(pop.roles) == sorted(PO.ROLES))
    check("T-P1", "86 population keys (18 flat + 17 x 4 subsets); z_proj kept per state as 97 x 8",
          len(pop.keys) == 86 and all(v.shape == (N, C.M_PROJ) for v in built["z_proj"].values()))
    check("T-P1", "a shared state gives both roles the same values; all roles kept (C1)",
          all(np.array_equal(pop.col(a[0], a[1], "KS"), pop.col(b[0], b[1], "KS")) for a, b in SHARED.items())
          and not np.array_equal(pop.col("lam0/sel", 2027, "KS"), pop.col("lam0/e30", 2027, "KS")))
    check("T-P1", "every volume in the registered order", pop.volume_ids == VOLS
          and all(pop.col(r, s, "q_z")[5] == recs[rmap[(r, s)]][5]["q_z"] for (r, s) in rmap))

    # ---- full run (T-P9 timing)
    t0 = time.perf_counter()
    res = PO.run_population(recs, rmap, VOLS, SEEDS, idx, sha, 28, [])
    t_run = time.perf_counter() - t0
    peak = GD.peak_rss_bytes()

    # ---- T-P2
    t0 = time.perf_counter()
    direct = ST.run_registered(direct_population(recs, rmap), idx, {"L2": "phys_lgr", "L3a": "phys_d01", "L3b": "phys_a01"})
    t_direct = time.perf_counter() - t0
    check("T-P2", "classified results (six measures, 21 contrasts, classes, tags, outcome) equal the direct call",
          dumps(res["classified"]) == dumps(direct), "%d bytes compared" % len(dumps(direct)))
    cl = res["classified"]["measures"]
    check("T-P2", "the synthetic design is read as constructed: six gap-selective, outcome A (interface check, not a finding)",
          all(cl[m]["class"] == ST.C_SELECTIVE for m in cl) and res["classified"]["outcome"]["outcome"] == "A")
    check("T-P2", "rules version names the frozen specification v1.0", "v1.0 FROZEN" in res["rules_version"])

    # ---- T-P3
    rep = res["reported"]
    check("T-P3", "25 reported keys x 21 contrasts; fields point, ci, per_seed and no direction label (Q-P2)",
          len(rep) == 25 and all(len(v["contrasts"]) == 21 and all(set(e) == {"point", "ci", "per_seed"} for e in v["contrasts"].values())
                                 for v in rep.values()))
    check("T-P3", "no classified key, no tail key and no stored input among the reported keys",
          not set(rep) & set(PO.CLASSIFIED_KEYS + PO.TAIL_KEYS + PO.INPUT_KEYS))
    worst = 0.0
    for key, sub, con in (("q_z", None, "S-GAP0"), ("PR", "full", "X-TIME"), ("phys_e1_sq", "full", "X-FIX(10→30)")):
        w = ST.ALL_CONTRASTS[con]
        per_seed = []
        for s in SEEDS:
            d = np.zeros(N)
            for role, wt in w.items():
                src = recs[rmap[(role, s)]]
                d += wt * np.array([(r[A.BANK_FIELD][sub][key] if sub else r[key]) for r in src])
            per_seed.append(d)
        pooled = np.mean(per_seed, axis=0)                       # mean over seeds first, then over volumes
        boot = np.array([pooled[row].mean() for row in idx])     # one replicate at a time
        lo, hi = np.percentile(boot, [2.5, 97.5])
        e = rep[key]["contrasts"][con]
        worst = max(worst, abs(e["point"] - pooled.mean()), abs(e["ci"][0] - lo), abs(e["ci"][1] - hi),
                    max(abs(a - b.mean()) for a, b in zip(e["per_seed"], per_seed)))
    check("T-P3", "effect, interval and per-seed effects of three reported keys equal a hand computation (1e-12)", worst <= 1e-12,
          "largest difference %.3e" % worst)

    # ---- T-P4
    sm = res["state_means"]
    ref = float(np.mean([r["q_z"] for r in recs[rmap[("lam0/fin", 2027)]]]))
    pooled = float(np.mean([np.mean([r[A.BANK_FIELD]["full"]["PR"] for r in recs[rmap[("lam2/sel", s)]]]) for s in SEEDS]))
    check("T-P4", "33 keys x 10 roles; a per-seed mean and a pooled mean equal the hand values; no interval field (Q-P3)",
          len(sm) == 33 and all(len(v["roles"]) == 10 for v in sm.values())
          and abs(sm["q_z"]["roles"]["lam0/fin"]["per_seed"]["2027"] - ref) <= 1e-12
          and abs(sm["PR"]["roles"]["lam2/sel"]["pooled"] - pooled) <= 1e-12
          and all(set(r) == {"per_seed", "pooled"} for v in sm.values() for r in v["roles"].values()))

    # ---- T-P5
    ec = res["e_cov"]
    d5 = max(abs(ec["per_state"][sid] - C.projected_cov_mismatch(z, proj)) for sid, z in z_states.items())
    check("T-P5", "E_cov from the 97 stored projections equals the core function on the unprojected codes (1e-10)", d5 <= 1e-10,
          "largest difference %.3e" % d5)
    check("T-P5", "28 per-state values; reference sqrt(9/96); every role and seed mapped to its state; no interval or contrast (Q-P4)",
          len(ec["per_state"]) == 28 and abs(ec["reference"] - math.sqrt(9.0 / 96.0)) <= 1e-15
          and set(ec) == {"reference", "n_volumes", "m", "per_state", "by_role_and_seed"}
          and all(ec["by_role_and_seed"][r][str(s)] == ec["per_state"][rmap[(r, s)]] for (r, s) in rmap))

    # ---- T-P6
    l6 = res["l6"]
    check("T-P6", "seven read contrasts x three seeds; share, six measures against PSNR and NLL, three tail correlations (Q-P1)",
          list(l6) == list(PO.READ_CONTRASTS) and len(l6) == 7
          and all(set(l6[c]) == {str(s) for s in SEEDS} and set(l6[c][str(s)]["measures"]) == set(ST.MEASURES)
                  and set(l6[c][str(s)]["tail"]) == {"vs_dKS", "vs_dNLL", "vs_dPSNR"}
                  and "share" in l6[c][str(s)]["share_nll_lower_psnr_lower"] for c in l6 for s in SEEDS))
    one = ST.l6_contrast(pop, "X-FIX(10→30)", 2028, idx, built["tau"])
    check("T-P6", "an interaction contrast equals a direct l6_contrast call", dumps(l6["X-FIX(10→30)"]["2028"]) == dumps(one))

    # ---- T-P7: TC_G not computed for one volume of the state shared by lam0/sel and lam0/e30 in seed 2026
    recs_nc = dict(recs)
    row = [json.loads(json.dumps(r)) for r in recs["lam0/e30@2026"]]
    row[11][A.BANK_FIELD]["full"]["TC_G"] = C.NOT_COMPUTED
    recs_nc["lam0/e30@2026"] = row
    rn = PO.run_population(recs_nc, rmap, VOLS, SEEDS, idx, sha, 28, [{"code": C.UNDEFINED_STATISTIC, "measure": "TC_G",
                                                                    "state": "lam0/e30@2026", "volume": "v011"}])
    l4 = rn["classified"]["measures"]["L4"]
    hit = [c for c, w in ST.ALL_CONTRASTS.items() if {"lam0/sel", "lam0/e30"} & set(w)]
    check("T-P7", "every contrast containing either role of the shared state is 'not computed' for L4, and no other",
          all((l4["effects"][c]["label"] == ST.NC) == (c in hit) for c in ST.ALL_CONTRASTS), "%d contrasts" % len(hit))
    check("T-P7", "the class of L4 follows the missing-contrast rule; the other five measures are unchanged",
          l4["class"] == ST.C_UNRES_NC and all(dumps(rn["classified"]["measures"][m]) == dumps(res["classified"]["measures"][m])
                                               for m in ST.MEASURES if m != "L4"), l4["class"])
    mt = {r["measure"]: r for r in rn["main_table"]["rows"]}
    check("T-P7", "main table: L4 on S-GAP0 is 'not computed' with no number; never zero",
          mt["L4"]["effects"]["S-GAP0"] == {"point": ST.NC, "ci": None, "per_seed": None, "label": ST.NC})
    smn = rn["state_means"]["TC_G"]["roles"]
    check("T-P7", "state means: both roles of the shared state 'not computed' in seed 2026 and pooled; other seeds computed",
          all(smn[r]["per_seed"]["2026"] == ST.NC and smn[r]["pooled"] == ST.NC and isinstance(smn[r]["per_seed"]["2027"], float)
              for r in ("lam0/sel", "lam0/e30")) and isinstance(smn["lam0/fin"]["pooled"], float))
    check("T-P7", "L6: L4 'not computed' for seed 2026 on S-GAP0, computed for seeds 2027 and 2028",
          rn["l6"]["S-GAP0"]["2026"]["measures"]["L4"]["vs_nll"]["rho"] == ST.NC
          and all(isinstance(rn["l6"]["S-GAP0"][s]["measures"]["L4"]["vs_nll"]["rho"], float) for s in ("2027", "2028")))
    ev = rn["events"]
    check("T-P7", "events: two role columns of one state listed with one volume each; the adapter event carried",
          sorted((e["role"], e["seed"], e["key"], e["subset"], e["volumes_not_computed"], e["state"]) for e in ev["not_computed_states"])
          == [("lam0/e30", 2026, "TC_G", "full", 1, "lam0/e30@2026"), ("lam0/sel", 2026, "TC_G", "full", 1, "lam0/e30@2026")]
          and ev["n_adapter_events"] == 1)
    check("T-P7", "reported quantities are unaffected by a missing TC_G", dumps(rn["reported"]) == dumps(res["reported"]))

    # ---- T-P8
    run = lambda **kw: PO.run_population(**{**dict(records=recs, role_map=rmap, volume_ids=VOLS, seeds=SEEDS, idx=idx,
                                                   registered_matrix_sha256=sha, expected_states=28, events=[]), **kw})
    PM = ST.POPULATION_MISMATCH
    raises("T-P8", "a wrong matrix hash stops before anything is computed", ST.BOOTSTRAP_IDENTITY,
           lambda: run(registered_matrix_sha256="0" * 64))
    raises("T-P8", "a changed matrix stops", ST.BOOTSTRAP_IDENTITY, lambda: run(idx=idx[:, ::-1].copy()))
    raises("T-P8", "a role missing from the role map", PM, lambda: run(role_map={k: v for k, v in rmap.items() if k != ("lam2/e10", 2027)}))
    raises("T-P8", "an unregistered role in the role map", PM, lambda: run(role_map={**rmap, ("lam9/e3", 2026): "lam0/e3@2026"}))
    raises("T-P8", "a state with records but in no role", PM, lambda: run(records={**recs, "orphan@2026": recs["lam0/e3@2026"]}))
    raises("T-P8", "a state in the role map without records", PM, lambda: run(records={k: v for k, v in recs.items() if k != "lam2/e3@2028"}))
    raises("T-P8", "a state mapped to two seeds", PM, lambda: run(role_map={**rmap, ("lam0/e3", 2027): "lam0/e3@2026"},
                                                                  records={k: v for k, v in recs.items() if k != "lam0/e3@2027"}))
    raises("T-P8", "96 records for 97 volumes", PM, lambda: run(records={**recs, "lam0/e3@2026": recs["lam0/e3@2026"][:-1]}))
    raises("T-P8", "an unexpected number of distinct states", PM, lambda: run(expected_states=30))
    bad = [json.loads(json.dumps(r)) for r in recs["lam0/e3@2026"]]; bad[3].pop("q_z")
    raises("T-P8", "a record that fails the schema", A.RECORD_SCHEMA, lambda: run(records={**recs, "lam0/e3@2026": bad}))
    bad = [json.loads(json.dumps(r)) for r in recs["lam0/e3@2026"]]; bad[3]["tau"] += 1e-9
    raises("T-P8", "tau differing between states", PM, lambda: run(records={**recs, "lam0/e3@2026": bad}))
    raises("T-P8", "unsorted volume ids", PM, lambda: run(volume_ids=VOLS[::-1]))
    raises("T-P8", "events that are not a list of dicts", C.INPUT, lambda: run(events="none"))

    # ---- T-P9
    again = PO.run_population(recs, rmap, VOLS, SEEDS, idx, sha, 28, [])
    text = dumps(res)
    check("T-P9", "deterministic: a second run gives an identical result", dumps(again) == text)
    check("T-P9", "the result survives a JSON round trip unchanged", dumps(json.loads(json.dumps(res))) == dumps(json.loads(text)))
    rows = res["main_table"]["rows"]
    need = {"measure", "key", "predicted_sign", "class", "fixed_epoch_tag", "seed_tag", "stability_summary", "coordinate_tag", "effects"}
    check("T-P9", "main table: six rows in the fixed order, every tag, effect with interval and per-seed effects on the four contrasts",
          [r["measure"] for r in rows] == ["L1", "L2", "L3a", "L3b", "L4", "L5"] and all(set(r) == need for r in rows)
          and all(list(r["effects"]) == list(PO.MAIN_TABLE_CONTRASTS) and all(len(e["per_seed"]) == 3 and len(e["ci"]) == 2
                                                                           for e in r["effects"].values()) for r in rows)
          and set(res["main_table"]["outcome"]) == {"outcome", "provisional", "open_measures", "open_contrasts"})
    check("T-P9", "result sections complete (C2)", set(res) == {"population_version", "stats_version", "adapter_version", "core_version",
          "rules_version", "identity", "classified", "main_table", "reported", "state_means", "e_cov", "l6", "tail", "events"})
    check("T-P9", "tail table: 97 values each, equal to the inputs", res["tail"]["tau"] == [r["tau"] for r in recs["lam0/e3@2026"]]
          and len(res["tail"]["q_tail"]) == N and res["tail"]["tau_constant"] is False)
    here = os.path.dirname(os.path.abspath(__file__))
    n_fail = sum(1 for c in CHECKS if not c["pass"])
    report = {"banner": BANNER, "schema": "seqref-dlat-poptest/1", "poptest_version": __version__,
              "population_version": PO.__version__, "stats_version": ST.__version__, "adapter_version": A.__version__,
              "core_version": C.__version__, "fixture_seed": FIXTURE_SEED, "numpy": np.__version__, "python": sys.version.split()[0],
              "bootstrap_matrix": {"seed": ST.DIAG_LAT_SEED, "shape": list(idx.shape), "sha256": sha,
                                   "note": "computed in this sandbox test; the registered identity is recorded by F1"},
              "workload": {"n_volumes": N, "n_boot": ST.N_BOOT, "states": 28, "role_seed_columns": 30, "contrasts": 21,
                           "seconds_matrix": t_matrix, "seconds_run_population": t_run, "seconds_run_registered_alone": t_direct,
                           "result_json_bytes": len(text), "peak_rss_bytes_after_first_run": peak,
                           "peak_rss_bytes_end": GD.peak_rss_bytes(), "cpu_count": os.cpu_count(),
                           "note": "sandbox CPU, synthetic numbers: an input to the F1 run card, not a host measurement"},
              "files_sha256": {n: hashlib_sha(os.path.join(here, n)) for n in sorted(os.listdir(here)) if n.endswith(".py")},
              "checks": CHECKS, "counts": {"pass": len(CHECKS) - n_fail, "fail": n_fail}, "result": "PASS" if n_fail == 0 else "FAIL"}
    with open(args.out, "x") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(BANNER)
    for t in ("T-P1", "T-P2", "T-P3", "T-P4", "T-P5", "T-P6", "T-P7", "T-P8", "T-P9"):
        rows = [c for c in CHECKS if c["test"] == t]
        print("%s  %d/%d" % (t, sum(c["pass"] for c in rows), len(rows)))
        for c in rows:
            if not c["pass"]:
                print("   FAIL %s %s" % (c["check"], c["note"]))
    print("workload: run_population %.1f s; run_registered alone %.1f s; peak RSS %.0f MiB; result %d bytes"
          % (t_run, t_direct, GD.peak_rss_bytes() / 2 ** 20, len(text)))
    print("RESULT %s (%d checks, %d failed)" % (report["result"], len(CHECKS), n_fail))
    raise SystemExit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
