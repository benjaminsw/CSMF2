# SEQREF-DLAT-GT v0.3 -- diag_lat_gtest.py: synthetic tests of diag_lat_g.py v0.3 (plan SEQREF-DLAT-F2-PLAN v0.3, gate F2.5)
# LIFETIME: KEEP
"""Tests of the G driver on the SYNTHETIC binding of the registered F1 test (diag_lat_f1test.SynBinding:
30 role-seed assignments on 28 state identities, synthetic slices and bank, the pinned model with
fixture parameters). No checkpoint, no MRI data. NOT D-LAT EVIDENCE.

  T-G     one complete run (28 states x 3 slices, 200 draws). The 28 state identities share two
          parameter settings, so slice_state_record is REALLY called once per (setting, slice),
          six times, and its result reused for the identities with the same weights; every other
          adapter call (six passes per slice-state, stored-row check) is real: G: COMPLETE,
          output files, record files, hashes; the population file equals a second call of the
          registered population stage on the records READ BACK from the record files
  T-EG1   rule E-G1: unit cases (boundary, affected states and contrasts, shared state, non-finite)
          and runs: with every slice-state flagged the population file is BYTE-IDENTICAL to the
          unflagged run; an injected residual flags exactly the expected state and contrasts
  T-STOP  typed stops with their criterion and evidence: runtime guard, wall limit, disk limit D,
          memory, registered arrays, environment, stored rows, non-finite residual, state identity
  T-MAN   manifest validation; the production entry point in child processes (-I -B)
  T-FIG   figure plan v0.2: the twelve data files and Table 1; every value equals its source in the
          population file, the stored rows or a registered function; provenance marks; rules 2 and 11
  T-PASS  flow passes of one record (132 encode, 132 decode, as the F1 probe record); the two calls
          the record function repeats return bit-identical values; no random generator state changes
  T-SRC   source bindings (v0.3): a synthetic D-UNC record folder in the registered layout (states in ANOTHER
          order than D-LAT) and three synthetic evidence files; values copied exactly into Figure 1 (c) and
          S6; a typed stop for every broken link of the chain; --check-sources in a child process without
          PyTorch; and, when --dunc-samples is given, the REAL host sample files read by the same functions
  T-FULL  FULL SIZE (28 states x 97 slices x 10,000 draws) on labelled SYNTHETIC records: real
          run-level bytes and the cumulative write-guard count against D (F2 decision Q3), the
          time of the population stage, peak memory

SELF-TEST EXCEPTION: this file builds synthetic and deliberately wrong inputs and, after T-G,
replaces adapter.slice_state_record (T-EG1, T-STOP) or GRun.record (T-FULL) by functions that
return cached or synthetic records, so that a run costs seconds. None of that is a fallback in
production code. Disclosed limits: the production binding (host data and checkpoints) is not
exercised; a real record is computed on 6 distinct synthetic slice-states only; T-FULL times the population
stage on this sandbox, not on the host; the registered projection matrix (a QR result) has other last bits on
this sandbox than on the host, so its registered identity cannot be reproduced here.

Changelog:
  v0.3 (2026-11-24) for driver v0.3: T-SRC; every manifest carries the part "sources"; T-FIG checks
        Figure 1 (c) and S6 against the bound sources. No v0.2 check removed (two are replaced by
        their positive form: Figure 1 (c) and the S6 synthetic panel are now written).
  v0.2 (2026-11-24) for driver v0.2: T-FIG, T-PASS; the report's result-file hashes; output schema v0.2
        in T-G and T-FULL; the E-G1 reading text. No v0.1 check removed.
  v0.1 (2026-11-24) first version.
"""
import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import diag_lat_adapter as A
import diag_lat_bind as BD
import diag_lat_core as C
import diag_lat_f1 as F
import diag_lat_f1test as FT
import diag_lat_g as G
import diag_lat_guard as GD
import diag_lat_p3test as P
import diag_lat_population as PO
import diag_lat_stats as ST
from diag_lat_core import DiagLatError

__version__ = "0.3"
ABBR = "SEQREF-DLAT-GT"
BANNER = "SANDBOX SYNTHETIC TEST -- NOT HOST EVIDENCE -- NOT D-LAT EVIDENCE (synthetic tensors)"
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS = FT.CHECKS
check, raises, variant = FT.check, FT.raises, FT.variant
RUN_LEVEL = ["g.log", "g_eg1.json", "g_main_table.json", "g_population.json", "g_report.json", "g_verdict.txt",
             "outputs_sha256.txt", "records_sha256.txt"] + list(G.FIG_NAMES)
RESULT_FILES = ["g_eg1.json", "g_main_table.json", "g_population.json"] + list(G.FIG_NAMES)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes(order="C")).hexdigest()


def arrays(n_vol, n_boot):
    """The three registered identities at the given sizes, computed here by the calls of 235 section 3.2."""
    idx = np.random.Generator(np.random.PCG64(20261120)).integers(0, n_vol, size=(n_boot, n_vol), dtype=np.int64)
    noise = np.random.Generator(np.random.PCG64(20261121)).standard_normal(size=(n_vol, C.N_ACQUIRED, 2))
    q, r = np.linalg.qr(np.random.Generator(np.random.PCG64(20261122)).standard_normal(size=(C.D_FREE, C.M_PROJ)))
    return {"bootstrap_sha256": sha(idx), "noise_sha256": sha(noise), "projection_sha256": sha(q * np.sign(np.diag(r)))}


SRC = {}                                        # the synthetic sources of the current binding (set by use_sources)


def wr(path, data, sidecar=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    digest = hashlib.sha256(data).hexdigest()
    if sidecar:
        with open(path + ".sha256", "w") as fh:
            fh.write(digest + "\n")
    return digest


def r1_value(sid, i):
    return 20.0 + int(sid[:6], 16) % 1000 / 100.0 + 0.03125 * i


def make_dunc(root, state_order, identities, volume_ids, f2_sha, mutate=None):
    """A SYNTHETIC D-UNC G output folder in the layout of diag_unc_g.py (manifest_g.json -> records/index.json ->
    state indexes -> record files, each with a sidecar). The states are written in REVERSED order. mutate(level, obj, k, i)
    may change an object before it is hashed. Returns the sha256 of manifest_g.json."""
    mutate = mutate or (lambda level, obj, k, i: obj)
    states = []
    for k, sid in enumerate(reversed(list(state_order))):
        ents = []
        for i, (ident, vol) in enumerate(zip(identities, volume_ids)):
            body = mutate("record", {"schema": "seqref-dunc-g-record/1", "version": "synthetic", "selftest": False, "state_index": k,
                                     "model_state_sha256": sid, "roles": [], "volume_id": vol, "file": ident[0], "slice_index": ident[1],
                                     "position": i, "r1_psnr_db": r1_value(sid, i), "stats": {}, "undefined": []}, k, i)
            rel = "records/s%02d/v%02d.rec" % (k, i)
            ents.append({"file": rel, "sha256": wr(os.path.join(root, rel), json.dumps(body, sort_keys=True).encode()), "bytes": 0, "position": i,
                         "volume_id": vol, "identity": [ident[0], ident[1]], "gate": "pass"})
        sidx = mutate("state", {"schema": "seqref-dunc-g-state-index/1", "version": "synthetic", "selftest": False, "state_index": k,
                                "model_state_sha256": sid, "roles": [], "n_records": len(ents), "records": ents, "reproduction": {}}, k, None)
        rel = "records/s%02d/state_index.json" % k
        states.append({"state_index": k, "model_state_sha256": sid, "file": rel, "sha256": wr(os.path.join(root, rel), json.dumps(sidx, sort_keys=True).encode())})
    final = mutate("index", {"schema": "seqref-dunc-g-index/1", "version": "synthetic", "selftest": False, "n_states": len(states),
                             "n_records": len(states) * len(identities), "states": states, "fixed_slices": [], "typed_events": []}, None, None)
    idx_sha = wr(os.path.join(root, "records", "index.json"), json.dumps(final, sort_keys=True).encode())
    man_g = mutate("manifest", {"schema": "seqref-dunc-g-manifest/1", "selftest": False, "manifest_f2_sha256": f2_sha, "stage1_index_sha256": idx_sha}, None, None)
    return wr(os.path.join(root, "manifest_g.json"), json.dumps(man_g, sort_keys=True).encode())


def make_sources(folder, state_order, identities, volume_ids, f2_sha, mutate=None):
    """The manifest part "sources" for a synthetic D-UNC folder and three synthetic evidence files."""
    out = {"dunc_g_out": os.path.join(folder, "dunc_out"), "note": "synthetic"}
    out["dunc_g_manifest_sha256"] = make_dunc(out["dunc_g_out"], state_order, identities, volume_ids, f2_sha, mutate)
    for name in G.EVIDENCE:
        out[name] = os.path.join(folder, "evidence", name + ".json")
        out[name + "_sha256"] = wr(out[name], json.dumps({"banner": "SYNTHETIC %s" % name, "fixtures": [{"id": name, "checks": [{"observed": 1.5, "tolerance": 2.0}]}]}).encode(),
                                   sidecar=False)
    return out


def use_sources(tmp, binding, f2_sha=None):
    SRC.clear()
    SRC.update(make_sources(tempfile.mkdtemp(prefix="src_", dir=tmp), binding.state_order, [tuple(x) for x in binding.identities],
                            list(binding.volume_ids), f2_sha or FT.h("synthetic manifest")))


def manifest(torch, n_vol=FT.N_SLICES, n_boot=200, **over):
    m = {"schema": G.MANIFEST_SCHEMA, "selftest": True, "rules_version": ST.RULES_VERSION,
         "specification": {"article": 235, "version": "v1.0 FROZEN", "sha256": "6afa3eda250946548b239df64344974d3c3502d23f6b4e22468ca61e6af97327"},
         "device": "cpu", "manifest_f2_sha256": FT.h("synthetic manifest"), "pin": "SYNTHETIC-NOT-A-PIN",
         "registered": {"n_states": 28, "n_volumes": n_vol, "n_boot": n_boot, "bank_rows": C.K_BANK},
         "arrays": arrays(n_vol, n_boot),
         "tolerances": {"stored_rows_nats_per_dim": 1e-4, "s1_s2_p1_relative": 1e-5},
         "e_g1": {"rule": "E-G1", "bank_roundtrip_flag_above": 1e-4, "statement": "synthetic", "provenance": "synthetic"},
         "limits": {"L_s": 21600.0, "D_bytes": 2 * GD.GIB, "t_stat_s": 25.0, "g_wall_ceiling_s": 21600, "disk_ceiling_bytes": 2 * GD.GIB,
                    "ram_bytes": 12 * GD.GIB, "reserve_bytes": GD.GIB},
         "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                         "torch_threads": int(torch.get_num_threads())},
         "f1_evidence": {"run": "synthetic", "report_sha256": FT.h("r"), "run_card_sha256": FT.h("c"), "commit": "synthetic",
                         "target_roundtrip_max_abs": 1e-5, "target_roundtrip_n": 1, "target_roundtrip_tolerance": 5e-5,
                         "peak_rss_bytes": 1},
         "figure_plan": {"name": "synthetic", "version": "synthetic", "sha256": FT.h("f"), "status": "synthetic"},
         "sources": dict(SRC), "review_status": "synthetic", "not_verified_before_run": "synthetic"}
    for k, v in over.items():
        m[k] = dict(m[k], **v) if isinstance(v, dict) else v
    return m


def run(tmp, binding, man):
    out = os.path.join(tmp, "run_%d" % len(CHECKS))
    rep, rc = G.run_g(binding, man, out, banner=BANNER)
    return rep, rc, out


def stops(tid, name, tmp, binding, man, code, criterion):
    """The run must end G: FAIL with this code under this criterion, evidence written, no result file."""
    try:
        rep, rc, out = run(tmp, binding, man)
    except DiagLatError as exc:
        return check(tid, name, False, "raised %s instead of reporting" % exc.code)
    got = rep.get("stop") or {}
    present = set(n for n in os.listdir(out) if not n.startswith(".") and not n.endswith(".sha256"))
    ok = (rc == 1 and got.get("code") == code and got.get("criterion") == criterion and rep["verdict"] == "G: FAIL"
          and rep["criteria"][criterion]["status"] == "FAIL" and {"g_report.json", "g_verdict.txt", "g.log", "outputs_sha256.txt"} <= present
          and not ({"g_population.json", "g_eg1.json", "g_main_table.json"} & present) and rep["outputs"]["result_files"] == {}
          and open(os.path.join(out, "g_verdict.txt")).read().endswith("G: FAIL\n"))
    check(tid, name, ok, "got %s at %s, exit %s, files %s" % (got.get("code"), got.get("criterion"), rc, sorted(present)))
    return rep, out


def read_back(out, binding):
    """Records from the record files (sidecar verified by hand), 'not computed' restored as the registered object."""
    def fix(o):
        if isinstance(o, dict):
            return {k: fix(v) for k, v in o.items()}
        if isinstance(o, list):
            return [fix(v) for v in o]
        return C.NOT_COMPUTED if o == C.NOT_COMPUTED else o
    records, events, bad = {}, [], 0
    for k, sid in enumerate(binding.state_order):
        rows = []
        for i in range(len(binding.volume_ids)):
            p = os.path.join(out, "records", "s%02d_%s" % (k, sid[:12]), "v%03d.json" % i)
            raw = open(p, "rb").read()
            bad += hashlib.sha256(raw).hexdigest() != open(p + ".sha256").read().strip()
            d = json.loads(raw)
            bad += not (d["schema"] == G.RECORD_SCHEMA and d["state"] == sid and d["slice_index"] == i)
            rows.append(fix(d["record"]))
            events.extend(d["events"])
        records[sid] = rows
    return records, events, bad


def child(args):
    return subprocess.run([sys.executable, "-I", "-B", os.path.join(HERE, "diag_lat_g.py")] + args, capture_output=True, text=True, timeout=600)


def main():
    ap = argparse.ArgumentParser(description="G driver tests (synthetic; NOT D-LAT EVIDENCE)")
    ap.add_argument("--pinned-root", required=True)
    ap.add_argument("--g-manifest", required=True, help="the draft production G manifest (read only)")
    ap.add_argument("--real-manifest", required=True, help="the registered D-UNC manifest_f2.json (read only)")
    ap.add_argument("--report", required=True)
    ap.add_argument("--dunc-samples", help="folder with the four REAL D-UNC sample files and sidecars from the host (read only)")
    ap.add_argument("--skip-full", action="store_true", help="development only: the report is then INCOMPLETE")
    args = ap.parse_args()
    logging.basicConfig(level=logging.CRITICAL)             # expected typed stops are exercised on purpose
    logging.getLogger().handlers[0].setLevel(logging.CRITICAL)   # the terminal stays quiet ...
    logging.getLogger().setLevel(logging.INFO)                    # ... while the driver's own log file gets its lines
    print(BANNER)
    P.pinned_preflight(args.pinned_root)
    s = P.need_pinned()
    torch = s.torch
    tmp = tempfile.mkdtemp(prefix="dlat_gtest_")
    t0 = time.monotonic()
    base = FT.SynBinding(s)
    use_sources(tmp, base)
    info = {}

    # ---- T-G: the complete run with real adapter calls; every record and diagnostic cached for the later runs
    cache, real_record = {}, A.slice_state_record

    def caching(model, st, bank, **kw):
        key = (id(model), st["identity"]["slice_index"])
        if key not in cache:                     # the 28 identities share two parameter settings: one real call per (setting, slice)
            cache[key] = real_record(model, st, bank, **kw)
        return cache[key]

    A.slice_state_record = caching
    try:
        t = time.monotonic()
        rep, rc, out = run(tmp, base, manifest(torch))
        info["t_g_real_s"] = time.monotonic() - t
    finally:
        A.slice_state_record = real_record
    check("T-G", "complete run: exit 0, G: COMPLETE, all ten criteria PASS", rc == 0 and rep["verdict"] == "G: COMPLETE"
          and all(rep["criteria"][c]["status"] == "PASS" for c in G.CRITERIA), json.dumps(rep.get("stop")))
    names = sorted(os.path.relpath(os.path.join(d, f), out) for d, _s, fs in os.walk(out) for f in fs
                   if not f.startswith(".") and not f.endswith(".sha256") and not d.startswith(os.path.join(out, "records")))
    check("T-G", "run-level files are exactly the output schema v0.2 (twelve files in figdata/), plus the records folder",
          names == sorted(RUN_LEVEL) and os.path.isdir(os.path.join(out, "records")), names)
    lines = [ln.split("  ") for ln in open(os.path.join(out, "outputs_sha256.txt")).read().splitlines()]
    check("T-G", "outputs_sha256.txt lists every run-level file at its hash", sorted(n for _h, n in lines) == sorted(set(RUN_LEVEL) - {"outputs_sha256.txt"})
          and all(BD.sha_file(os.path.join(out, n)) == hh for hh, n in lines))
    rlines = [ln.split("  ") for ln in open(os.path.join(out, "records_sha256.txt")).read().splitlines()]
    n_ss = 28 * FT.N_SLICES
    check("T-G", "%d record files, each listed at its hash in records_sha256.txt" % n_ss, len(rlines) == n_ss
          and all(BD.sha_file(os.path.join(out, n)) == hh for hh, n in rlines)
          and sum(len(f) for _d, _s, f in os.walk(os.path.join(out, "records"))) == 2 * n_ss)
    records, events, bad = read_back(out, base)
    for rows in records.values():
        for r in rows:
            try:
                A.validate_record(r, list(ST.STABILITY_SUBSETS))
            except DiagLatError:
                bad += 1
    check("T-G", "every record file: sidecar, schema, identity; every record passes the registered validate_record", bad == 0, bad)
    idx = ST.bootstrap_matrix(n_volumes=FT.N_SLICES, n_boot=200)
    again = PO.run_population(records, base.role_map, list(base.volume_ids), base.seeds, idx, ST.matrix_identity(idx), 28, events, n_boot=200)
    pop_bytes = open(os.path.join(out, "g_population.json"), "rb").read()
    check("T-G", "g_population.json equals the registered population stage called again on the records read back from disk",
          F.dumps(again) == pop_bytes)
    pop = json.loads(pop_bytes)
    check("T-G", "population file has exactly the keys run_population returns; six classified measures; 21 contrasts",
          set(pop) == set(again) and set(pop["classified"]["measures"]) == set(ST.MEASURES) and len(pop["identity"]["contrasts"]) == 21)
    on_disk = json.load(open(os.path.join(out, "g_report.json")))
    check("T-G", "report on disk: exact keys; manifest embedded; arrays verified; stop null", set(on_disk) == G.REPORT_KEYS
          and on_disk["manifest"]["content"]["schema"] == G.MANIFEST_SCHEMA and on_disk["arrays"]["verified_against_manifest"] is True
          and on_disk["stop"] is None and "run_card" not in on_disk)
    check("T-G", "report holds the sha256 of every result file, equal to the file on disk and to outputs_sha256.txt (R3)",
          sorted(on_disk["outputs"]["result_files"]) == sorted(RESULT_FILES)
          and all(BD.sha_file(os.path.join(out, n)) == hh for n, hh in on_disk["outputs"]["result_files"].items())
          and all(dict((n, hh) for hh, n in lines)[n] == hh for n, hh in on_disk["outputs"]["result_files"].items()))
    res = on_disk["resources"]
    check("T-G", "resources: 28 per-state entries with W_k; W_k never above L; final elapsed recorded", len(res["per_state"]) == 28
          and all(p["W_k_s"] <= res["L_s"] for p in res["per_state"]) and res["elapsed_final_s"] > 0 and res["peak_rss_bytes"] > 0)
    eg = json.load(open(os.path.join(out, "g_eg1.json")))
    flat = [r for sid in base.state_order for r in eg["residuals"][sid]]
    check("T-G", "E-G1 annotation: exact keys; %d residuals, all finite; maximum equals the largest; flags = residuals above 1e-4" % n_ss,
          set(eg) == G.EG1_KEYS and len(flat) == n_ss and all(np.isfinite(flat)) and eg["max"]["residual"] == max(flat)
          and eg["n_flagged"] == sum(r > 1e-4 for r in flat) == len(eg["flagged"]), eg["n_flagged"])
    info["real_bank_roundtrip_max"] = max(flat)
    tab = json.load(open(os.path.join(out, "g_main_table.json")))
    check("T-G", "main table: the registered table unchanged, review status and figure plan reference beside it, L4 annotation separate",
          tab["main_table"] == pop["main_table"] and tab["review_status"] == "synthetic" and tab["figure_plan"]["sha256"] == FT.h("f")
          and set(tab["numerical_annotation_L4"]["contrasts"]) == set(PO.MAIN_TABLE_CONTRASTS))
    check("T-G", "verdict file: ten criterion lines and G: COMPLETE", open(os.path.join(out, "g_verdict.txt")).read()
          == "".join("G%d: PASS\n" % k for k in range(1, 11)) + "G: COMPLETE\n")
    check("T-G", "cache holds one real record per (parameter setting, slice)", len(cache) == 2 * FT.N_SLICES, len(cache))
    check("T-EG1", "the annotation carries the reading text approved at F2.4, word for word", eg["reading"] == G.READING and G.READING ==
          "An affected L4 result is reported with the annotation 'numerical adequacy unresolved' beside its class, tags and stability "
          "label. The annotation withdraws and changes none of them and does not make an outcome provisional. A flag is not a finding "
          "of numerical invalidity: nothing is deleted or replaced. Whether a flagged L4 result is numerically adequate is decided by "
          "review, outside the frozen rules.")

    # ---- T-FIG: figure plan v0.2
    fig = {n: json.load(open(os.path.join(out, "figdata", "%s.json" % n))) for n in ("fig1", "fig2", "fig3", "fig4", "fig5", "s1", "s2", "s3",
                                                                                  "s4", "s5", "s6", "s7")}
    cm, sm, pop_sha = pop["classified"]["measures"], pop["state_means"], hashlib.sha256(pop_bytes).hexdigest()
    panels = {n: {p["panel"]: p for p in d["panels"]} for n, d in fig.items()}
    allp = [p for d in fig.values() for p in d["panels"]]
    check("T-FIG", "every file: schema, the figure plan by name, version and sha256, the review status", all(
        d["schema"] == G.FIG_SCHEMA and d["figure_plan"]["sha256"] == FT.h("f") and d["review_status"] == "synthetic" for d in fig.values()))
    check("T-FIG", "rule 10: every panel is marked 'existing result' or 'D-LAT measure'; every panel fed from the population file names it with its sha256",
          all(p["provenance"] in (G.EXISTING, G.MEASURE) for p in allp)
          and all(p["source"]["sha256"] == pop_sha for p in allp if (p.get("source") or {}).get("file") == "g_population.json")
          and sum(1 for p in allp if (p.get("source") or {}).get("file") == "g_population.json") >= 20)
    check("T-FIG", "Figure 1 consists of existing results only; (a), (b) equal the stored rows of every state; (c) holds the R1 PSNR of the bound D-UNC records",
          all(p["provenance"] == G.EXISTING for p in fig["fig1"]["panels"])
          and all(panels["fig1"]["a"]["data"][sid][k] == [float(v) for v in base.stored[sid][k]] for sid in base.state_order for k in ("NLL", "B", "J"))
          and all(panels["fig1"]["b"]["data"][sid]["R0_PSNR"] == [float(v) for v in base.stored[sid]["R0_PSNR"]] for sid in base.state_order)
          and all(panels["fig1"]["c"]["data"][sid]["R1_PSNR"] == [r1_value(sid, i) for i in range(FT.N_SLICES)] for sid in base.state_order)
          and set(panels["fig1"]["c"]["data"]) == set(base.state_order) and panels["fig1"]["c"]["source"]["sha256"] == SRC["dunc_g_manifest_sha256"]
          and "not_written" not in json.dumps(fig))
    seven = list(PO.READ_CONTRASTS)
    spec_panels = (("fig2", "a", "L1"), ("fig2", "b", "L2"), ("fig3", "a", "L3a"), ("fig3", "b", "L3b"), ("fig4", "a", "L4"), ("fig4", "b", "L5"))
    check("T-FIG", "contrast panels of Figures 2-4: the seven read contrasts, each equal to the classified effect (pooled, interval, per seed)",
          all(set(panels[f][p]["data"]) == set(seven) and len(seven) == 7 and all(panels[f][p]["data"][c] == cm[mid]["effects"][c] for c in seven) for f, p, mid in spec_panels))
    traj = (("fig2", "c", ("log_gamma_truth", "log_gamma_bank")), ("fig2", "d", ("q_z",)), ("fig3", "c", ("e0_sq", "e1_sq", "tr_sigma_per_dim")),
            ("fig4", "c", ("PR",)), ("fig4", "d", G.SENSITIVITIES))
    check("T-FIG", "trajectory panels equal the registered state means, by checkpoint role, per seed and pooled; epochs given per role and seed",
          all(panels[f][p]["data"][k] == sm[k] for f, p, keys in traj for k in keys)
          and all(fig["fig2"]["checkpoint_roles"][r][str(sd)]["epoch"] == base.states[base.role_map[(r, sd)]]["epoch"] for r in PO.ROLES for sd in base.seeds))
    check("T-FIG", "Figure 4: L4 labelled as a proxy with the E-G1 annotation beside it; PR beside the registered isotropic reference; sensitivities with their contrasts",
          "proxy" in panels["fig4"]["a"]["label"] and panels["fig4"]["a"]["numerical_annotation"]["summary"] == eg["l4_summary"]
          and panels["fig4"]["c"]["isotropic_reference"] == C.pr_isotropic_reference(C.K_BANK, C.D_FREE)
          and all(panels["fig4"]["d"]["contrasts"][k][c] == pop["reported"][k]["contrasts"][c] for k in G.SENSITIVITIES for c in seven))
    check("T-FIG", "Figure 5 (a), (b): six rows in the fixed order L1..L5, each with its own unit; rule 2: X-TIME and X-FIX(10→30) together; values equal the classified effects",
          panels["fig5"]["a"]["row_order"] == ["L1", "L2", "L3a", "L3b", "L4", "L5"] == list(panels["fig5"]["a"]["data"])
          and set(panels["fig5"]["a"]["units"]) == set(ST.MEASURES) and "rule_11" in fig["fig5"]
          and panels["fig5"]["b"]["contrasts"] == ["X-TIME", "X-FIX(10→30)"]
          and all(panels["fig5"][p]["data"][m][c] == cm[m]["effects"][c] for p in ("a", "b") for m in ST.MEASURES for c in panels["fig5"][p]["contrasts"]))
    built = PO.build_population(records, base.role_map, list(base.volume_ids), base.seeds)["pop"]
    ok = True
    for c in seven:
        for sd in base.seeds:
            for key, name in (("d_NLL", "NLL"), ("d_R0_PSNR", "R0_PSNR")):
                want = ST.seed_differences(built, ST.ALL_CONTRASTS[c], name, sd)
                ok = ok and panels["fig5"]["c"]["data"][c][str(sd)][key] == [float(v) for v in want]
    check("T-FIG", "Figure 5 (c): per-volume changes equal the registered seed_differences on the records read back from disk; (d) equals the L6 tables",
          ok and panels["fig5"]["d"]["data"] == pop["l6"])
    check("T-FIG", "S1: all 21 contrasts for the six classified and every reported key; S2: stability labels of L2, L3a, L3b, L4; S7: the events table",
          all(panels["s1"]["classified"]["data"][m] == cm[m]["effects"] and len(cm[m]["effects"]) == 21 for m in ST.MEASURES)
          and panels["s1"]["reported"]["data"] == pop["reported"]
          and all(panels["s2"][m]["data"] == cm[m]["stability_per_contrast"] for m in ("L2", "L3a", "L3b", "L4")) and len(panels["s2"]) == 4
          and panels["s7"]["events"]["data"] == pop["events"])
    check("T-FIG", "S3: physical complements beside the standardised effects with the coordinate tag; S4: tail and its L6 correlations; S5: skewness, kurtosis, E_cov",
          all(panels["s3"][m]["data"]["physical"] == cm[m]["physical_effects"] and panels["s3"][m]["coordinate_tag"] == cm[m]["coordinate_tag"]
              for m in ST.COORDINATE_TAGGED)
          and panels["s3"]["PR and phys_PR"]["data"]["phys_PR"]["contrasts"] == pop["reported"]["phys_PR"]["contrasts"]
          and panels["s4"]["tail"]["data"] == pop["tail"]
          and all(panels["s4"]["L6 tail correlations"]["data"][c][sd] == pop["l6"][c][sd]["tail"] for c in seven for sd in pop["l6"][c])
          and panels["s5"]["E_cov"]["data"] == pop["e_cov"] and panels["s5"]["skewness"]["data"]["state_means"] == sm["skewness"])
    s6 = {p["source"]["file"]: p for p in fig["s6"]["panels"] if (p.get("source") or {}).get("sha256")}
    check("T-FIG", "S6: step G fixtures equal the report's; the fixture tables of the three bound evidence files are copied exactly, each with path and sha256",
          panels["s6"]["step G"]["data"] == on_disk["fixtures"] and len(fig["s6"]["panels"]) == 5
          and all(s6[SRC[e]]["source"]["sha256"] == SRC[e + "_sha256"] and s6[SRC[e]]["provenance"] == G.EXISTING
                  and s6[SRC[e]]["data"]["fixtures"] == json.load(open(SRC[e]))["fixtures"] for e in G.EVIDENCE))
    check("T-SRC", "report: the binding names the verified D-UNC chain (2 + 28 x (1 + n) files) and the three evidence hashes",
          on_disk["binding"]["sources"]["files_verified"] == 2 + 28 * (1 + FT.N_SLICES) and on_disk["binding"]["sources"]["n_records"] == n_ss
          and on_disk["binding"]["sources"]["evidence"] == {e: SRC[e + "_sha256"] for e in G.EVIDENCE})

    # ---- T-SRC: every broken link of the chain is a typed stop (unit level; no model)
    ids3, vols3 = [tuple(x) for x in base.identities], list(base.volume_ids)
    f2s = FT.h("synthetic manifest")

    def broken(name, code, mutate=None, after=None, f2=f2s):
        d = tempfile.mkdtemp(prefix="bad_", dir=tmp)
        src = make_sources(d, base.state_order, ids3, vols3, f2s, mutate)
        if after:
            after(src)
        raises("T-SRC", name, code, lambda: G.bind_sources({"sources": src, "manifest_f2_sha256": f2}, base.state_order, ids3, vols3))

    def at(level, k, i, **change):
        def m(lv, obj, kk, ii):
            return dict(obj, **change) if lv == level and (k is None or kk == k) and (i is None or ii == i) else obj
        return m

    def swap_identity(lv, obj, kk, ii):
        if lv == "state" and kk == 2:
            ents = [dict(e) for e in obj["records"]]
            ents[0]["identity"], ents[1]["identity"] = ents[1]["identity"], ents[0]["identity"]
            return dict(obj, records=ents)
        return obj

    def tamper(rel):
        def f(src):
            with open(os.path.join(src["dunc_g_out"], rel), "ab") as fh:
                fh.write(b" ")
        return f
    good = make_sources(tempfile.mkdtemp(prefix="good_", dir=tmp), base.state_order, ids3, vols3, f2s)
    got = G.bind_sources({"sources": good, "manifest_f2_sha256": f2s}, base.state_order, ids3, vols3)
    check("T-SRC", "intact chain: every state matched by its hash although D-UNC lists the states in reversed order; values copied exactly",
          all(got["r1"][sid] == [r1_value(sid, i) for i in range(FT.N_SLICES)] for sid in base.state_order) and len(got["r1"]) == 28)
    broken("another root hash than the registered one: DIAG_LAT_ORIGIN", A.ORIGIN, after=lambda src: src.update(dunc_g_manifest_sha256=FT.h("x")))
    broken("the D-UNC manifest names another F2 manifest: DIAG_LAT_POPULATION_MISMATCH", ST.POPULATION_MISMATCH, f2=FT.h("other f2"))
    broken("the D-UNC manifest is a self-test: DIAG_LAT_POPULATION_MISMATCH", ST.POPULATION_MISMATCH, mutate=at("manifest", None, None, selftest=True))
    broken("final index changed after hashing: DIAG_LAT_ORIGIN", A.ORIGIN, after=tamper("records/index.json"))
    broken("a state index changed after hashing: DIAG_LAT_ORIGIN", A.ORIGIN, after=tamper("records/s05/state_index.json"))
    broken("a record file changed after hashing: DIAG_LAT_ORIGIN", A.ORIGIN, after=tamper("records/s07/v01.rec"))
    broken("a record file missing: DIAG_LAT_ORIGIN", A.ORIGIN, after=lambda src: os.remove(os.path.join(src["dunc_g_out"], "records/s03/v02.rec")))
    broken("a sidecar missing: DIAG_LAT_ORIGIN", A.ORIGIN, after=lambda src: os.remove(os.path.join(src["dunc_g_out"], "records/s03/v02.rec.sha256")))
    broken("the index counts 27 states: DIAG_LAT_POPULATION_MISMATCH", ST.POPULATION_MISMATCH, mutate=at("index", None, None, n_states=27))
    broken("a state index names another model state: DIAG_LAT_POPULATION_MISMATCH", ST.POPULATION_MISMATCH,
           mutate=at("state", 4, None, model_state_sha256=FT.h("other")))
    broken("two slice identities swapped in a state index: DIAG_LAT_POPULATION_MISMATCH", ST.POPULATION_MISMATCH, mutate=swap_identity)
    broken("a record with another position: DIAG_LAT_POPULATION_MISMATCH", ST.POPULATION_MISMATCH, mutate=at("record", 6, 1, position=2))
    broken("a record of another volume: DIAG_LAT_POPULATION_MISMATCH", ST.POPULATION_MISMATCH, mutate=at("record", 6, 1, volume_id="other"))
    broken("a record of another model state: DIAG_LAT_POPULATION_MISMATCH", ST.POPULATION_MISMATCH,
           mutate=at("record", 6, 1, model_state_sha256=FT.h("other")))
    broken("a record without a finite R1 PSNR: DIAG_LAT_NONFINITE", C.NONFINITE, mutate=at("record", 9, 0, r1_psnr_db=None))
    broken("an evidence report changed after registration: DIAG_LAT_ORIGIN", A.ORIGIN, after=lambda src: open(src["p3_report"], "ab").write(b" "))
    broken("an evidence report missing: DIAG_LAT_ORIGIN", A.ORIGIN, after=lambda src: os.remove(src["selftest_report"]))

    def no_fixtures(src):
        src["f1_report_sha256"] = wr(src["f1_report"], b'{"verdict": "x"}', sidecar=False)
    broken("an evidence report without a fixture table: DIAG_LAT_ORIGIN", A.ORIGIN, after=no_fixtures)
    other = FT.SynBinding.__new__(FT.SynBinding)
    other.__dict__.update(base.__dict__)
    raises("T-SRC", "the D-LAT binding has a state the D-UNC chain lacks: DIAG_LAT_POPULATION_MISMATCH", ST.POPULATION_MISMATCH,
           lambda: G.bind_sources({"sources": good, "manifest_f2_sha256": f2s}, [FT.h("new state")] + list(base.state_order)[1:], ids3, vols3))
    if args.dunc_samples:                                    # the REAL host files, read by the same functions
        sd = os.path.realpath(args.dunc_samples)
        root_sha = "3b21b4ed27e6c4f6f7a0ef349a98ea298eb5eb1a2cb885b9e56393c0e4102348"
        mg = G._json(G._read(os.path.join(sd, "manifest_g.json"), root_sha, "sample manifest", True), "m")
        fi = G._json(G._read(os.path.join(sd, "records", "index.json"), mg["stage1_index_sha256"], "sample index", True), "i")
        s0 = fi["states"][0]
        si = G._json(G._read(os.path.join(sd, s0["file"]), s0["sha256"], "sample state index", True), "s")
        e0 = si["records"][0]
        rc0 = G._json(G._read(os.path.join(sd, e0["file"]), e0["sha256"], "sample record", True), "r")
        reg = json.load(open(args.real_manifest))
        reg_states = sorted(x["model_state_sha256"] for x in reg["states"])
        check("T-SRC", "REAL host samples: registered root hash; the four schemas; manifest -> index -> state index -> record, each at the sha256 its parent names",
              (mg["schema"], fi["schema"], si["schema"], rc0["schema"]) == G.DUNC_SCHEMAS and mg["selftest"] is False)
        check("T-SRC", "REAL host samples: the D-UNC manifest names the registered F2 manifest; 28 states and 2,716 records; the states are those of the registered F2 manifest",
              mg["manifest_f2_sha256"] == BD.REGISTERED_F2_SHA256 and fi["n_states"] == 28 and fi["n_records"] == 2716
              and sorted(x["model_state_sha256"] for x in fi["states"]) == reg_states)
        idents = [tuple(x) for x in reg["population"]["identities"]]
        check("T-SRC", "REAL host samples: the state index lists the 97 registered slice identities in the registered order with positions 0..96 and every gate passed",
              [tuple(e["identity"]) for e in si["records"]] == idents and [e["position"] for e in si["records"]] == list(range(97))
              and all(e["gate"] == "pass" for e in si["records"]) and si["model_state_sha256"] == s0["model_state_sha256"])
        check("T-SRC", "REAL host samples: the record carries state, position, volume, slice identity and a finite float r1_psnr_db",
              rc0["model_state_sha256"] == s0["model_state_sha256"] and rc0["position"] == 0 and rc0["volume_id"] == e0["volume_id"]
              and (rc0["file"], rc0["slice_index"]) == idents[0] and isinstance(rc0["r1_psnr_db"], float) and np.isfinite(rc0["r1_psnr_db"]))
        info["real_sample_r1_psnr_db"] = rc0["r1_psnr_db"]
        info["real_sample_first_state_index_in_d_unc"] = s0["state_index"]
    else:
        print("  [SKIP] real D-UNC samples not given: the report is INCOMPLETE")
    check("T-FIG", "Table 1: six rows in the fixed order, the outcome and the exploratory-status sentence", tab["id"] == "table1"
          and [r["measure"] for r in tab["main_table"]["rows"]] == list(ST.MEASURES) and tab["main_table"]["outcome"] == pop["classified"]["outcome"]
          and tab["exploratory_status"] == pop["classified"]["note"])
    plain = json.dumps(fig)
    check("T-FIG", "no figure file holds a rescaled, ranked or common-scale quantity key (rule 11); none of the excluded outputs (plan section 5)",
          not any(w in plain for w in ("\"rank", "\"rescaled", "\"z_score", "\"common_scale", "umap", "UMAP", "\"covariance_matrix")))

    # ---- T-PASS: flow passes and the redundancy of the two repeated calls (R3)
    count = {"encode": 0, "decode": 0}
    originals = []
    for model in base.models.values():
        for name in ("encode", "decode"):
            real = getattr(model.flow, name)
            originals.append((model.flow, name, real))

            def wrapped(x, *a, _real=real, _name=name, **k):
                count[_name] += int(x.shape[0])
                return _real(x, *a, **k)
            setattr(model.flow, name, wrapped)
    try:
        g = G.GRun(manifest(torch), os.path.join(tmp, "unused_g"), "x")
        f1 = F.F1Run(FT.card(), os.path.join(tmp, "unused_f"), "x")
        for r_ in (g, f1):
            r_.b = base
            r_.d2_d3(); r_.d4_tensors(); r_.d5(); r_.slice_fixtures()
        sid0 = base.state_order[0]
        model0, _v = base.load_state(sid0)
        st0 = base._slices[0]
        t_state, n_state = torch.get_rng_state().clone(), np.random.get_state()[1].copy()
        count.update(encode=0, decode=0)
        rec_g, diag_g = g.record(model0, st0, 0, sid0)
        passes_g = dict(count)
        rng_same = bool(torch.equal(t_state, torch.get_rng_state())) and bool(np.array_equal(n_state, np.random.get_state()[1]))
        count.update(encode=0, decode=0)
        f1.probe_record(model0, st0, 0, sid0)
        passes_f = dict(count)
        e1, e2 = A.encode_truth(model0, st0), A.encode_truth(model0, st0)
        r1, r2 = A.decode_r0_std(model0, st0), A.decode_r0_std(model0, st0)
        rec_again, _d = g.record(model0, st0, 0, sid0)
    finally:
        for obj, name, real in originals:
            setattr(obj, name, real)
    info["passes_g_record"], info["passes_f1_probe_record"] = passes_g, passes_f
    check("T-PASS", "one G record: 132 encode and 132 decode passes (K = 128), the same as the F1 probe record on which t and L were measured",
          passes_g == {"encode": 132, "decode": 132} == passes_f, (passes_g, passes_f))
    check("T-PASS", "the repeated target encode returns bit-identical z*, NLL, B and J; the repeated R0 decode a bit-identical image",
          np.array_equal(e1["z_star"], e2["z_star"]) and all(e1[k] == e2[k] for k in ("NLL", "B", "J")) and np.array_equal(r1, r2)
          and rec_g["NLL"] == e1["NLL"])
    check("T-PASS", "a record changes neither the torch nor the NumPy global random state; a second call returns the identical record",
          rng_same and F.dumps(rec_again) == F.dumps(rec_g))

    # ---- from here on slice_state_record returns the cached real records (SELF-TEST EXCEPTION)
    inject = {}

    def cached(model, st, bank, **kw):
        rec, diag = cache[(id(model), st["identity"]["slice_index"])]
        key = (cached.calls // FT.N_SLICES, st["identity"]["slice_index"])
        cached.calls += 1
        return rec, (dict(diag, roundtrip_max_abs=inject[key]) if key in inject else diag)

    def fast(binding, man):
        cached.calls = 0
        return run(tmp, binding, man)

    A.slice_state_record = cached
    try:
        rep2, rc2, out2 = fast(base, manifest(torch))
        check("T-EG1", "a run on the cached records reproduces g_population.json byte for byte", rc2 == 0
              and open(os.path.join(out2, "g_population.json"), "rb").read() == pop_bytes)
        # every slice-state flagged: nothing in the population file changes
        rep3, rc3, out3 = fast(base, manifest(torch, e_g1={"bank_roundtrip_flag_above": 1e-30}))
        eg3 = json.load(open(os.path.join(out3, "g_eg1.json")))
        check("T-EG1", "threshold 1e-30: all %d flagged, G: COMPLETE (no stop), population file BYTE-IDENTICAL" % n_ss, rc3 == 0
              and rep3["verdict"] == "G: COMPLETE" and eg3["n_flagged"] == n_ss
              and open(os.path.join(out3, "g_population.json"), "rb").read() == pop_bytes)
        check("T-EG1", "threshold 1e-30: all 28 states affected, all 21 contrasts and the L4 summary 'numerical adequacy unresolved', outcome note set",
              len(eg3["affected_states"]) == 28 and all(c["label"] == G.UNRESOLVED for c in eg3["contrasts"].values())
              and eg3["l4_summary"] == G.UNRESOLVED and eg3["l4_outcome_note"] == G.OUTCOME_NOTE
              and rep3["numerical"]["l4_summary"] == G.UNRESOLVED)
        tab3 = json.load(open(os.path.join(out3, "g_main_table.json")))
        check("T-EG1", "threshold 1e-30: main table rows unchanged; only the annotation differs", tab3["main_table"] == tab["main_table"]
              and tab3["numerical_annotation_L4"]["summary"] == G.UNRESOLVED)
        # one injected residual: state of (lam2/e10, 2027), slice 1
        target = base.role_map[("lam2/e10", 2027)]
        k_t = base.state_order.index(target)
        inject[(k_t, 1)] = 5e-4
        rep4, rc4, out4 = fast(base, manifest(torch))
        eg4 = json.load(open(os.path.join(out4, "g_eg1.json")))
        want = {c for c, w in ST.ALL_CONTRASTS.items() if w.get("lam2/e10", 0.0) != 0.0}
        got = {c for c, v in eg4["contrasts"].items() if v["label"] == G.UNRESOLVED}
        check("T-EG1", "one injected residual 5e-4: one flag with its identity, one affected state, exactly the contrasts with lam2/e10 (%d)" % len(want),
              rc4 == 0 and eg4["n_flagged"] == 1 and eg4["flagged"][0]["state"] == target and eg4["flagged"][0]["slice_index"] == 1
              and eg4["flagged"][0]["residual"] == 5e-4 and eg4["affected_states"] == [target] and got == want and len(want) > 0, sorted(got))
        check("T-EG1", "that run: gap contrasts not affected -> L4 summary 'no numerical flag', no outcome note; population file BYTE-IDENTICAL",
              eg4["l4_summary"] == G.NO_FLAG and eg4["l4_outcome_note"] is None
              and open(os.path.join(out4, "g_population.json"), "rb").read() == pop_bytes)
        rec_file = json.load(open(os.path.join(out4, "records", "s%02d_%s" % (k_t, target[:12]), "v001.json")))
        check("T-EG1", "the flagged residual is in its record file's diagnostics and in the log", rec_file["diagnostics"]["roundtrip_max_abs"] == 5e-4
              and "E-G1 flag" in open(os.path.join(out4, "g.log")).read())
        inject.clear()
        inject[(0, 0)] = 1e-4
        _r, _c, out5 = fast(base, manifest(torch))
        check("T-EG1", "boundary: a residual of exactly 1e-4 is not flagged", json.load(open(os.path.join(out5, "g_eg1.json")))["n_flagged"]
              == eg["n_flagged"])
        inject.clear()
        inject[(3, 2)] = float("nan")
        cached.calls = 0
        _rep, out6 = stops("T-STOP", "non-finite bank round-trip residual: DIAG_LAT_NONFINITE under G7", tmp, base, manifest(torch), C.NONFINITE, "G7")
        cached.calls = 0
        check("T-STOP", "that stop: the record files of the three completed states are kept (9), the record of the stop is not written",
              sum(len(f) for _d, _s, f in os.walk(os.path.join(out6, "records"))) == 2 * (3 * FT.N_SLICES + 2))
        inject.clear()

        # ---- T-STOP
        cached.calls = 0
        stops("T-STOP", "runtime guard: L below W_1 stops after state 1 with DIAG_LAT_RESOURCE_LIMIT under G8", tmp, base,
              manifest(torch, limits={"L_s": 26.0, "t_stat_s": 25.0}), GD.RESOURCE_LIMIT, "G8")
        cached.calls = 0
        _rep, out7 = stops("T-STOP", "disk limit D: a D that holds only a few records stops with DIAG_LAT_RESOURCE_LIMIT under G8", tmp, base,
                           manifest(torch, limits={"D_bytes": 40000}), GD.RESOURCE_LIMIT, "G8")
        cached.calls = 0
        stops("T-STOP", "memory: a limit below the peak stops with DIAG_LAT_RESOURCE_LIMIT under G8", tmp, base,
              manifest(torch, limits={"ram_bytes": 2 ** 20}), GD.RESOURCE_LIMIT, "G8")
        for part, code in (("bootstrap_sha256", ST.BOOTSTRAP_IDENTITY), ("noise_sha256", F.REPRODUCTION_MISMATCH),
                           ("projection_sha256", F.REPRODUCTION_MISMATCH)):
            cached.calls = 0
            stops("T-STOP", "registered array %s differs from the manifest: %s under G5" % (part, code), tmp, base,
                  manifest(torch, arrays={part: FT.h("other")}), code, "G5")
        cached.calls = 0
        stops("T-STOP", "thread count differs from the F1 run: DIAG_LAT_REPRODUCTION_MISMATCH under G4", tmp, base,
              manifest(torch, environment={"torch_threads": int(torch.get_num_threads()) + 1}), F.REPRODUCTION_MISMATCH, "G4")
        cached.calls = 0
        stops("T-STOP", "torch version differs: DIAG_LAT_REPRODUCTION_MISMATCH under G4", tmp, base,
              manifest(torch, environment={"torch": "0.0"}), F.REPRODUCTION_MISMATCH, "G4")
        cached.calls = 0
        stops("T-STOP", "numpy version differs: DIAG_LAT_REPRODUCTION_MISMATCH under G1", tmp, base,
              manifest(torch, environment={"numpy": "0.0"}), F.REPRODUCTION_MISMATCH, "G1")
        shifted = {sid: dict(v) for sid, v in base.stored.items()}
        first = base.state_order[0]
        shifted[first]["NLL"] = shifted[first]["NLL"] + 1e-3
        cached.calls = 0
        stops("T-STOP", "stored NLL row off by 1e-3: DIAG_LAT_ADAPTER under G7 at the first record", tmp, variant(base, stored=shifted),
              manifest(torch), A.ADAPTER, "G7")

        class WrongLoad(FT.SynBinding):
            def __init__(self, b):
                self.__dict__.update(b.__dict__)

            def load_state(self, sid):
                model, ver = FT.SynBinding.load_state(self, sid)
                return model, dict(ver, model_state_sha256=FT.h("other state"))
        cached.calls = 0
        stops("T-STOP", "the loader returns another model state: DIAG_LAT_ORIGIN under G2", tmp, WrongLoad(base), manifest(torch), A.ORIGIN, "G2")
        cached.calls = 0
        stops("T-STOP", "29 states for 28 registered: DIAG_LAT_POPULATION_MISMATCH under G2", tmp, base,
              manifest(torch, registered={"n_states": 29}), ST.POPULATION_MISMATCH, "G2")
        cached.calls = 0
        stops("T-STOP", "self-test flag of manifest and binding differ: DIAG_LAT_RUN_CARD under G1", tmp, variant(base, selftest=False),
              manifest(torch), F.RUN_CARD, "G1")
        raises("T-STOP", "an existing output folder is refused", GD.GUARD_STATE, lambda: G.run_g(base, manifest(torch), out))
        cached.calls = 0
        _rep, out8 = stops("T-SRC", "run: a source with another root hash stops under G3 before any record (DIAG_LAT_ORIGIN)", tmp, base,
                           manifest(torch, sources={"dunc_g_manifest_sha256": FT.h("x")}), A.ORIGIN, "G3")
        check("T-SRC", "that stop: no record file was written and no model was loaded", not os.path.isdir(os.path.join(out8, "records")) and cached.calls == 0)
        cached.calls = 0
        stops("T-SRC", "run: an evidence report with another sha256 stops under G3 (DIAG_LAT_ORIGIN)", tmp, base,
              manifest(torch, sources={"selftest_report_sha256": FT.h("x")}), A.ORIGIN, "G3")
        check("T-STOP", "the D stop kept the record files written before it and wrote its failure evidence under the ceiling",
              0 < sum(len(f) for _d, _s, f in os.walk(os.path.join(out7, "records"))) < 2 * n_ss)
    finally:
        A.slice_state_record = real_record

    # ---- T-EG1 unit cases
    role_map, seeds, order = base.role_map, base.seeds, base.state_order
    ids = [("f%d" % i, i) for i in range(2)]
    spec = {"rule": "E-G1", "bank_roundtrip_flag_above": 1e-4, "statement": "s", "provenance": "p"}
    clean = {sid: [1e-5, 9.99e-5] for sid in order}
    e = G.eg1(clean, order, ids, role_map, seeds, spec)
    check("T-EG1", "unit: no residual above the threshold: no flag, every label 'no numerical flag'", e["n_flagged"] == 0
          and e["affected_states"] == [] and all(c["label"] == G.NO_FLAG for c in e["contrasts"].values()) and e["l4_summary"] == G.NO_FLAG
          and e["n_slice_states"] == 56 and e["max"]["residual"] == 9.99e-5)
    shared = role_map[("lam0/sel", 2027)]
    check("T-EG1", "unit precondition: lam0/sel and lam0/e10 of seed 2027 share one state", shared == role_map[("lam0/e10", 2027)])
    one = dict(clean)
    one[shared] = [1e-5, 1.0000001e-4]
    e = G.eg1(one, order, ids, role_map, seeds, spec)
    want = {c for c, w in ST.ALL_CONTRASTS.items() if w.get("lam0/sel", 0.0) != 0.0 or w.get("lam0/e10", 0.0) != 0.0}
    got = {c for c, v in e["contrasts"].items() if v["label"] == G.UNRESOLVED}
    check("T-EG1", "unit: a flagged shared state affects the contrasts of BOTH its roles; S-GAP0 among them -> L4 summary unresolved",
          got == want and "S-GAP0" in got and e["l4_summary"] == G.UNRESOLVED and e["l4_outcome_note"] == G.OUTCOME_NOTE
          and e["contrasts"]["S-GAP0"]["affected_role_seed"] == ["lam0/sel|2027"] and e["n_flagged"] == 1, sorted(got))
    x = role_map[("lam0/fin", 2028)]
    x_over = {c for c, w in ST.ALL_CONTRASTS.items() if w.get("lam0/fin", 0.0) != 0.0}
    e = G.eg1(dict(clean, **{x: [2e-4, 3e-4]}), order, ids, role_map, seeds, spec)
    check("T-EG1", "unit: two flags in one state: two flags, one affected state, maximum and its place reported", e["n_flagged"] == 2
          and e["affected_states"] == [x] and e["max"] == {"state": x, "state_index": order.index(x), "slice_index": 1, "identity": ["f1", 1], "residual": 3e-4}
          and {c for c, v in e["contrasts"].items() if v["label"] == G.UNRESOLVED} == x_over)
    for badv in (float("nan"), float("inf")):
        raises("T-EG1", "unit: residual %r is fatal (DIAG_LAT_NONFINITE)" % badv, C.NONFINITE,
               lambda badv=badv: G.eg1(dict(clean, **{x: [badv, 1e-5]}), order, ids, role_map, seeds, spec))
    check("T-EG1", "unit: the input residuals are returned unchanged", e["residuals"][order[0]] == clean[order[0]])
    # runtime guard arithmetic (235 section 7)
    check("T-STOP", "runtime guard: W_k = elapsed + (n - k) T_max + t_stat; equal to L passes", G.runtime_guard(100.0, 3, 28, 10.0, 25.0, 375.0) == 375.0)
    raises("T-STOP", "runtime guard: W_k above L stops", GD.RESOURCE_LIMIT, lambda: G.runtime_guard(100.0, 3, 28, 10.0, 25.0, 374.999))
    check("T-STOP", "runtime guard: after the last state W = elapsed + t_stat", G.runtime_guard(100.0, 28, 28, 10.0, 25.0, 125.0) == 125.0)

    # ---- T-MAN
    good = manifest(torch)
    for name, mut, code in (
            ("a missing key", lambda m: m.pop("e_g1"), G.G_MANIFEST), ("an unexpected key", lambda m: m.update(extra=1), G.G_MANIFEST),
            ("a missing limit", lambda m: m["limits"].pop("L_s"), G.G_MANIFEST), ("device cuda", lambda m: m.update(device="cuda"), F.REPRODUCTION_MISMATCH),
            ("L above its ceiling", lambda m: m["limits"].update(L_s=21600.5), G.G_MANIFEST),
            ("D above its ceiling", lambda m: m["limits"].update(D_bytes=2 * GD.GIB + 1), G.G_MANIFEST),
            ("D as a float", lambda m: m["limits"].update(D_bytes=1e6), G.G_MANIFEST),
            ("threshold zero", lambda m: m["e_g1"].update(bank_roundtrip_flag_above=0.0), G.G_MANIFEST),
            ("another rules version", lambda m: m.update(rules_version="x"), G.G_MANIFEST),
            ("a malformed array hash", lambda m: m["arrays"].update(noise_sha256="abc"), G.G_MANIFEST),
            ("another S1 tolerance", lambda m: m["tolerances"].update(s1_s2_p1_relative=1e-4), G.G_MANIFEST),
            ("cited target round trip above its tolerance", lambda m: m["f1_evidence"].update(target_roundtrip_max_abs=1e-4), G.G_MANIFEST),
            ("production with synthetic sizes", lambda m: m.update(selftest=False), G.G_MANIFEST),
            ("a relative source path", lambda m: m["sources"].update(p3_report="p3_report.json"), G.G_MANIFEST),
            ("a shortened source hash", lambda m: m["sources"].update(dunc_g_manifest_sha256="3b21b4ed"), G.G_MANIFEST),
            ("no sources part", lambda m: m.pop("sources"), G.G_MANIFEST),
            ("an empty review status", lambda m: m.update(review_status=" "), G.G_MANIFEST)):
        m = json.loads(json.dumps(good))
        mut(m)
        raises("T-MAN", "manifest with %s is refused" % name, code, lambda m=m: G.check_manifest(m))
    prod_raw = open(args.g_manifest, "rb").read()
    prod_sha = hashlib.sha256(prod_raw).hexdigest()
    prod = json.loads(prod_raw)
    try:
        G.check_manifest(prod)
        ok = True
    except DiagLatError as exc:
        ok = False
        info["production_manifest_error"] = exc.message
    check("T-MAN", "the draft production manifest passes check_manifest", ok)
    here = arrays(97, 10000)
    check("T-MAN", "bootstrap and noise identities computed here at the registered sizes equal those registered in V15",
          all(here[k] == prod["arrays"][k] for k in ("bootstrap_sha256", "noise_sha256")))
    info["projection_sha256_on_this_machine"] = here["projection_sha256"]      # the QR result depends on the LAPACK build: see the evidence note
    info["projection_equals_host_registered"] = here["projection_sha256"] == prod["arrays"]["projection_sha256"]
    v15 = {"L_s": 9210.59234748222, "D_bytes": 825533754, "t_stat_s": 24.774574849754572, "g_wall_ceiling_s": 21600,
           "disk_ceiling_bytes": 2 * GD.GIB, "ram_bytes": 16 * GD.GIB, "reserve_bytes": 64 * GD.GIB}
    check("T-MAN", "production manifest: L, D and t_stat are the V15 figures exactly; ceilings of 235 section 7; threshold 1e-4; not a self-test",
          prod["limits"] == v15 and prod["e_g1"]["bank_roundtrip_flag_above"] == 1e-4 and prod["selftest"] is False
          and prod["registered"] == F.REGISTERED and prod["manifest_f2_sha256"] == BD.REGISTERED_F2_SHA256 and prod["pin"] == BD.PIN, prod["limits"])
    check("T-MAN", "production manifest: registered array identities, environment and evidence of V15; figure plan sha256 of 235 section 5.1",
          prod["arrays"] == {"bootstrap_sha256": "718ece159e2cb81c995e616883a40db6adcfba90c0eb79cfccffa9c7a89de471",
                             "noise_sha256": "8059e9cbc4516e4c08de2a7a50a681cc623ace2a3fe77944d8f3ef2d9ec5b24f",
                             "projection_sha256": "dcfb9fbde82c9a1839006551491f885b4bf673ad9c5be2c599bacdd40ec64ea2"}
          and prod["environment"] == {"python": "3.12.3", "torch": "2.5.1+cu121", "numpy": "1.26.4", "torch_threads": 12}
          and prod["f1_evidence"]["report_sha256"] == "f75a009fccb48e00fe66e18e706b27a9e4eeb5f879906a24b54a02d64b0930d6"
          and prod["f1_evidence"]["run_card_sha256"] == "45142fac001e5d428523c8d7faf4f2540e7ce918f357ea09ac11e0ee93ad849c"
          and prod["figure_plan"]["sha256"] == "83ad15029858171d8f3a6bb6a37cca25b5469e9d6fdebd782bb28033df3aded1"
          and prod["specification"]["sha256"] == "6afa3eda250946548b239df64344974d3c3502d23f6b4e22468ca61e6af97327")
    srcs = prod["sources"]
    check("T-MAN", "production manifest: the four sources with complete hashes and absolute host paths; the bound F1 report is the cited V15 report",
          srcs["dunc_g_manifest_sha256"] == "3b21b4ed27e6c4f6f7a0ef349a98ea298eb5eb1a2cb885b9e56393c0e4102348"
          and srcs["dunc_g_out"] == "/home/benjamin/CSMFII/v06lt_evidence/DIAG_UNC/g_20261119T102151Z/out"
          and srcs["f1_report_sha256"] == prod["f1_evidence"]["report_sha256"]
          and srcs["p3_report_sha256"] == "8df65249ce934ad92082703c2e5784286cdd02ad9cd108bdb7e5ea15e6b1984d"
          and srcs["selftest_report_sha256"] == "0a346b2398edc65250c66cb926314336ab5501185126cc78e3a924eaa9d9843e"
          and all(srcs[e].startswith("/home/benjamin/CSMFII/") for e in G.EVIDENCE))
    # --check-sources in a child process on the synthetic results folder of the binding test (no PyTorch)
    import diag_lat_bindtest as BT
    crepo = os.path.join(tmp, "crepo")
    croot = os.path.join(crepo, *BD.RESULTS_SUBDIR)
    os.makedirs(croot)
    mf, _expect = BT.build(croot)
    mpath, msha = BT.write_manifest(tmp, mf)
    bound = BD.bind_manifest(mpath, f2_sha256=msha, n_states=28, n_volumes=97, pin=BD.PIN)
    vfiles = BD.verify_files(bound, croot)
    csrc = make_sources(tempfile.mkdtemp(prefix="csrc_", dir=tmp), list(bound["state_order"]), [tuple(x) for x in bound["identities"]],
                        list(vfiles["volume_ids"]), msha)
    cman = manifest(torch, n_vol=97, n_boot=10000, manifest_f2_sha256=msha, pin=BD.PIN, sources=csrc)
    cpath = os.path.join(tmp, "check_manifest.json")
    open(cpath, "wb").write(F.dumps(cman))
    code = ("man = json.load(open(sys.argv[1]))\nimport diag_lat_g as G\nG.check_manifest(man)\ninfo = G.check_sources(sys.argv[2], sys.argv[3], man)\n"
            "print('SOURCES: PASS', info['files_verified'], info['n_records'], 'torch' in sys.modules)")
    before = sorted(os.path.join(d, f) for d, _s, fs in os.walk(tmp) for f in fs)
    r = FT.snippet(code, cpath, crepo, mpath)
    check("T-SRC", "--check-sources path in a child process (-I -B): PASS on 2,746 files and 2,716 records, PyTorch not imported, no file created",
          r.returncode == 0 and "SOURCES: PASS 2746 2716 False" in r.stdout
          and before == sorted(os.path.join(d, f) for d, _s, fs in os.walk(tmp) for f in fs), (r.stdout[-200:], r.stderr[-300:]))
    os.remove(os.path.join(csrc["dunc_g_out"], "records", "s11", "v40.rec"))
    r = FT.snippet(code, cpath, crepo, mpath)
    check("T-SRC", "--check-sources with one record file removed: typed stop, non-zero exit", r.returncode != 0 and "DIAG_LAT_ORIGIN" in r.stderr, r.stderr[-200:])
    self_path = os.path.join(tmp, "selftest_manifest.json")
    open(self_path, "wb").write(F.dumps(good))
    empty = os.path.join(tmp, "empty_repo")
    os.makedirs(empty)
    common = ["--repo", empty, "--manifest", args.real_manifest, "--dunc-dir", empty]
    r = child(["--g-manifest", args.g_manifest, "--g-manifest-sha256", prod_sha, "--out", os.path.join(tmp, "c1"), "--checkpoint=x"] + common)
    check("T-MAN", "entry point: --checkpoint is refused with exit 3 before any import", r.returncode == 3 and "DIAG_LAT_REFUSED" in r.stderr, r.returncode)
    r = child(["--g-manifest", args.g_manifest, "--g-manifest-sha256", "0" * 64, "--out", os.path.join(tmp, "c2")] + common)
    check("T-MAN", "entry point: a wrong manifest sha256 stops with DIAG_LAT_G_MANIFEST, exit 1, nothing written",
          r.returncode == 1 and "G: FAIL (%s)" % G.G_MANIFEST in r.stdout and not os.path.exists(os.path.join(tmp, "c2")), r.stdout[-200:])
    r = child(["--g-manifest", self_path, "--g-manifest-sha256", BD.sha_file(self_path), "--out", os.path.join(tmp, "c3")] + common)
    check("T-MAN", "entry point: a self-test manifest is refused", r.returncode == 1 and "G: FAIL (%s)" % G.G_MANIFEST in r.stdout
          and not os.path.exists(os.path.join(tmp, "c3")), r.stdout[-200:])
    r = child(["--g-manifest", args.g_manifest, "--g-manifest-sha256", prod_sha, "--out", os.path.join(tmp, "c4")] + common)
    c4 = os.path.join(tmp, "c4")
    rep_c4 = json.load(open(os.path.join(c4, "g_report.json"))) if os.path.isfile(os.path.join(c4, "g_report.json")) else {}
    check("T-MAN", "entry point with the production manifest on a checkout without results: typed stop, exit 1, G: FAIL, no traceback, no result file",
          r.returncode == 1 and "G: FAIL" in r.stdout and "Traceback" not in r.stderr and "DIAG_LAT_" in r.stderr
          and not os.path.exists(os.path.join(c4, "g_population.json")), r.stdout[-200:])
    info["entry_point_stop"] = {"stdout": r.stdout.strip()[-120:], "report": rep_c4.get("stop")}

    # ---- T-FULL: full size on labelled SYNTHETIC records
    if not args.skip_full:
        old_n = FT.N_SLICES
        FT.N_SLICES = 97
        try:
            big = FT.SynBinding(s)
        finally:
            FT.N_SLICES = old_n
        big = variant(big, volume_ids=["v%03d" % v for v in range(97)])
        use_sources(tmp, big)
        labels = list(ST.STABILITY_SUBSETS)
        syn, syn_events = F.synthetic_population(big.role_map, big.state_order, 97, labels)
        diag0 = dict(cache[next(iter(cache))][1], events=[])
        pos = {"n": 0}

        def synthetic_record(self, model, st, i, sid):
            k = pos["n"] // 97
            pos["n"] += 1
            ev = [dict(e) for e in syn_events if e["state"] == str(sid) and i == 0]
            return syn[sid][i], dict(diag0, roundtrip_max_abs=3e-5 + 1e-7 * ((k * 97 + i) % 500), events=ev)

        real = G.GRun.record
        G.GRun.record = synthetic_record
        try:
            t = time.monotonic()
            repf, rcf, outf = run(tmp, big, manifest(torch, n_vol=97, n_boot=10000, limits={"D_bytes": 825533754, "L_s": 9210.59234748222,
                                                                                         "t_stat_s": 24.774574849754572}))
            info["t_full_s"] = time.monotonic() - t
        finally:
            G.GRun.record = real
        check("T-FULL", "full size on SYNTHETIC records: G: COMPLETE, 2,716 record files", rcf == 0 and repf["verdict"] == "G: COMPLETE"
              and repf["records"]["n"] == 2716 and repf["outputs"]["n_record_files"] == 2716, json.dumps(repf.get("stop")))
        guard = GD.WriteGuard(outf, 2 * GD.GIB, 0)
        written = guard.written
        on = sum(os.path.getsize(os.path.join(d, f)) for d, _s, fs in os.walk(outf) for f in fs)
        run_level = {n: os.path.getsize(os.path.join(outf, n)) for n in RUN_LEVEL}
        info.update(full_guard_written_bytes=written, full_bytes_on_disk=on, full_run_level_bytes=run_level,
                    full_population_stage_s=repf["resources"]["population_stage_s"], full_peak_rss_bytes=repf["resources"]["peak_rss_bytes"],
                    full_record_bytes_max=max(os.path.getsize(os.path.join(d, f)) for d, _s, fs in os.walk(os.path.join(outf, "records"))
                                              for f in fs if f.endswith(".json")))
        check("T-FULL", "Q3: cumulative write-guard count %d bytes with the final output schema is below the operative limit D = 825,533,754" % written,
              written < 825533754, written)
        info["full_run_level_total_bytes"] = sum(run_level.values())
        info["full_within_f1_allowance_R_fix"] = sum(run_level.values()) + 193 * len(run_level) <= 1034780
        info["full_within_f1_projection_R"] = written <= 13484924
        check("T-FULL", "full size: the twelve figure data files and Table 1 are written; Figure 5 (c) holds 97 values per contrast and seed",
              all(os.path.isfile(os.path.join(outf, n)) for n in G.FIG_NAMES)
              and all(len(v["d_NLL"]) == 97 for per in json.load(open(os.path.join(outf, "figdata", "fig5.json")))["panels"][2]["data"].values()
                      for v in per.values()))
        egf = json.load(open(os.path.join(outf, "g_eg1.json")))
        check("T-FULL", "full size: 2,716 residuals recorded, none flagged, one missing-value event carried into the population stage",
              egf["n_slice_states"] == 2716 and egf["n_flagged"] == 0
              and json.load(open(os.path.join(outf, "g_population.json")))["events"]["n_adapter_events"] == 1)
    else:
        print("  [SKIP] T-FULL not run: the report is INCOMPLETE")

    info["elapsed_s"] = time.monotonic() - t0
    n_fail = sum(1 for c in CHECKS if not c["pass"])
    complete = not args.skip_full and bool(args.dunc_samples)
    report = {"banner": BANNER, "test": ABBR, "version": __version__, "driver": G.__version__, "complete": complete,
              "driver_sha256": BD.sha_file(os.path.join(HERE, "diag_lat_g.py")), "production_manifest_sha256": prod_sha,
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                              "threads": int(torch.get_num_threads()), "platform": platform.platform()},
              "n_checks": len(CHECKS), "n_fail": n_fail, "info": info, "checks": CHECKS,
              "verdict": ("GTEST: PASS" if n_fail == 0 else "GTEST: FAIL") if complete else "GTEST: INCOMPLETE"}
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    print("%d checks, %d failed -- %s" % (len(CHECKS), n_fail, report["verdict"]))
    return 0 if n_fail == 0 and complete else 1


if __name__ == "__main__":
    sys.exit(main())
