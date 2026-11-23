# SEQREF-DLAT-F1T v0.3 -- diag_lat_f1test.py: synthetic tests of diag_lat_f1.py v0.3 (plan SEQREF-DLAT-F1D v0.1; criteria A1-A12 of SEQREF-DLAT-F1 v0.2; conditions C3, C5-C12)
# LIFETIME: KEEP
"""Tests of the F1 driver on a SYNTHETIC binding built from the pinned classes (as the P3 test
does): 30 role-seed assignments on 28 state identities, synthetic slices, a synthetic bank, the
pinned model with fixture parameters. No checkpoint, no MRI data. NOT D-LAT EVIDENCE.

Every criterion A1-A12 has a passing case (the complete run T-F1) and at least one failing case
that must stop with its typed code under that criterion. Bind-only (C6, C8) and the production
entry point run in child processes with -I -B on the synthetic results folder of diag_lat_bindtest.

SELF-TEST EXCEPTION: this file builds synthetic and deliberately wrong inputs. None of that is a
fallback in production code. The 28 synthetic states use TWO parameter settings (the pinned model
has 2.6e8 parameters; 28 copies do not fit in memory): the state identities are distinct, the
weights are not. Disclosed limits: the production binding's slice_states, bank and load_state
need the host's data and checkpoints and are not exercised here; S2, S3
and S4 have no failing case at driver level (S2 and S4 are tested in the P3 test and the core
self-test; an S3 failure needs duplicate volumes, which A3 refuses first); fixture 3 and
fixture 8 have no failing case here (core self-test).

Since C10 the production entry point accepts only the registered manifest hash, so it can no
longer be run on the synthetic results folder. Bind-only and the production binding are tested
there by calling bind_only() and run_f1() in child processes (-I -B) with a self-test card; the
entry point itself is tested with the production card, the REAL registered manifest_f2.json
(--real-manifest) and a checkout without results, where it must stop typed.

Changelog:
  v0.3 (2026-11-23) for driver v0.3: T-WALL (the final elapsed-time check: a run whose stages all
        complete but which ends after the wall ceiling is F1: FAIL under A9; the same run inside
        the ceiling is not stopped); T-F1 checks the recorded final elapsed time.
  v0.2 (2026-11-23) for driver v0.2: T-C10 (production card with another manifest hash or pin),
        T-C11 (failure report: separate file and schema, validated; six refused forms), card
        field of C12; the child-process tests rebuilt as described above. No other test changed.
  v0.1 (2026-11-23) first version.
"""
import argparse
import copy
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import diag_lat_adapter as A
import diag_lat_bind as BD
import diag_lat_bindtest as BT
import diag_lat_core as C
import diag_lat_f1 as F
import diag_lat_guard as GD
import diag_lat_p3test as P
import diag_lat_population as PO
import diag_lat_stats as ST
from diag_lat_core import DiagLatError

__version__ = "0.3"
ABBR = "SEQREF-DLAT-F1T"
BANNER = "SANDBOX SYNTHETIC TEST -- NOT HOST EVIDENCE -- NOT D-LAT EVIDENCE (synthetic tensors)"
SEEDS = [2026, 2027, 2028]
SHARED = {("lam0/sel", 2027): ("lam0/e10", 2027), ("lam2/sel", 2028): ("lam2/fin", 2028)}
N_SLICES = 3
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS = []


def check(tid, name, ok, note=""):
    CHECKS.append({"test": tid, "name": name, "pass": bool(ok), "note": str(note)})
    print("  [%s] %-5s %s%s" % ("PASS" if ok else "FAIL", tid, name, (" -- %s" % note) if note and not ok else ""))
    return bool(ok)


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def card(**over):
    c = {"schema": F.CARD_SCHEMA, "selftest": True, "rules_version": ST.RULES_VERSION, "device": "cpu",
         "manifest_f2_sha256": h("synthetic manifest"), "pin": "SYNTHETIC-NOT-A-PIN",
         "registered": {"n_states": 28, "n_volumes": N_SLICES, "n_boot": 200, "bank_rows": C.K_BANK},
         "tolerances": {"fx1_roundtrip_max_abs": 1e-5, "fx2ii_roundtrip_max_abs": 1e-5, "stored_rows_nats_per_dim": 1e-4,
                        "s1_s2_p1_relative": 1e-5, "fx7_relative": 1e-5, "fx3_l2_identity_abs": 1e-12},
         "fx7_validity": "synthetic", "not_verified_before_run": "synthetic", "ceilings": {"f1_wall_s": 3600, "g_wall_s": 21600, "disk_bytes": 2 * GD.GIB,
                                                   "reserve_bytes": GD.GIB, "ram_bytes": 12 * GD.GIB, "vram_bytes": 10 * GD.GIB}}
    for k, v in over.items():
        if isinstance(v, dict):
            c[k] = dict(c[k], **v)
        else:
            c[k] = v
    return c


class SynBinding:
    """SYNTHETIC binding with the registered structure (30 assignments -> 28 states)."""
    selftest = True
    pin = "SYNTHETIC-NOT-A-PIN"
    manifest_sha256 = h("synthetic manifest")
    files_hashed = 0

    def __init__(self, s):
        self.s = s
        self.seeds = list(SEEDS)
        self.role_map = {(r, sd): h("state %s@%d" % SHARED.get((r, sd), (r, sd))) for r in PO.ROLES for sd in SEEDS}
        self.state_order = sorted(set(self.role_map.values()))
        self.states = {}
        for k, sid in enumerate(self.state_order):
            role, seed = next((r, sd) for (r, sd), q in sorted(self.role_map.items()) if q == sid)
            self.states[sid] = {"arm": role.split("/")[0], "seed": seed, "epoch": k, "path": "synthetic/%d" % k,
                                "file_sha256": h("file " + sid), "setting": k % 2}
        self._slices = P.synthetic_states(s, N_SLICES)
        self.identities = [(st["identity"]["file"], st["identity"]["slice_index"]) for st in self._slices]
        self.volume_ids = [f for f, _i in self.identities]
        hh = hashlib.sha256()
        for st in self._slices:
            hh.update(st["mask"][0].numpy().tobytes())
        self.masks_sha256 = hh.hexdigest()
        self._bank = P.rng(1).standard_normal((C.K_BANK, C.D_FREE)).astype(np.float32)
        self.bank_sha256 = hashlib.sha256(self._bank.tobytes(order="C")).hexdigest()
        self.models, rows = {}, {}
        for k, (_label, seed, amp) in enumerate(P.SETTINGS):
            model = s.ffr.build_model()
            model.eval()
            P.set_parameters(s, model, seed, amp)
            self.models[k] = model
            enc = [A.encode_truth(model, st) for st in self._slices]
            psnr = [float(s.tg._psnr(s.EV.r0_image(model, st)[0].abs(), st["x_true_mag"])) for st in self._slices]
            rows[k] = {"R0_PSNR": np.array(psnr), "NLL": np.array([e["NLL"] for e in enc]),
                       "B": np.array([e["B"] for e in enc]), "J": np.array([e["J"] for e in enc])}
        self.stored = {sid: {n: v.copy() for n, v in rows[self.states[sid]["setting"]].items()} for sid in self.state_order}
        self.loads = 0

    def slice_states(self):
        return self._slices

    def bank(self):
        return self._bank

    def load_state(self, sid):
        self.loads += 1
        return self.models[self.states[sid]["setting"]], {"model_state_sha256": sid, "file_sha256": self.states[sid]["file_sha256"]}


def variant(base, **attrs):
    b = copy.copy(base)
    for k, v in attrs.items():
        setattr(b, k, v)
    return b


def with_slice(base, i, **fields):
    """A binding whose slice i is a shallow copy with some fields replaced."""
    sl = list(base._slices)
    sl[i] = dict(sl[i], **fields)
    return variant(base, _slices=sl)


def run(tmp, name, binding, c, **kw):
    return F.run_f1(binding, c, os.path.join(tmp, name), **kw)


def stops(tid, name, tmp, binding, c, code, criterion, **kw):
    """The run must end FAIL with this code under this criterion, with its evidence written."""
    out = os.path.join(tmp, "neg_%d" % len(CHECKS))
    try:
        rep, rc = F.run_f1(binding, c, out, **kw)
    except DiagLatError as exc:
        return check(tid, name, False, "raised %s instead of reporting" % exc.code)
    got = rep.get("stop") or {}
    files = all(os.path.isfile(os.path.join(out, f)) for f in ("f1_report.json", "f1_verdict.txt", "f1.log", "outputs_sha256.txt"))
    ok = (rc == 1 and got.get("code") == code and got.get("criterion") == criterion and rep["verdict"] == "F1: FAIL"
          and rep["criteria"][criterion]["status"] == "FAIL" and files
          and open(os.path.join(out, "f1_verdict.txt")).read().endswith("F1: FAIL\n"))
    return check(tid, name, ok, "got %s at %s, exit %s, files %s" % (got.get("code"), got.get("criterion"), rc, files))


def raises(tid, name, code, fn):
    try:
        fn()
    except DiagLatError as exc:
        return check(tid, name, exc.code == code, "got %s" % exc.code)
    except Exception as exc:
        return check(tid, name, False, "%s: %s" % (type(exc).__name__, exc))
    return check(tid, name, False, "no stop")


def child(args):
    return subprocess.run([sys.executable, "-I", "-B", os.path.join(HERE, "diag_lat_f1.py")] + args,
                          capture_output=True, text=True, timeout=600)


def snippet(code, *argv):
    """A child process (-I -B) that imports the driver and runs a few lines; argv[1:] = argv."""
    return subprocess.run([sys.executable, "-I", "-B", "-c", "import sys, json\nsys.path.insert(0, %r)\nimport diag_lat_f1 as F\n%s" % (HERE, code)]
                          + list(argv), capture_output=True, text=True, timeout=600)


def main():
    ap = argparse.ArgumentParser(description="F1 driver tests (synthetic; NOT D-LAT EVIDENCE)")
    ap.add_argument("--pinned-root", required=True)
    ap.add_argument("--real-manifest", required=True, help="the registered D-UNC manifest_f2.json (read only)")
    ap.add_argument("--run-card", required=True, help="the draft production run card (read only)")
    ap.add_argument("--report", required=True)
    ap.add_argument("--skip-full", action="store_true", help="development only: the report is then INCOMPLETE")
    args = ap.parse_args()
    logging.basicConfig(level=logging.CRITICAL)             # expected typed stops are exercised on purpose
    print(BANNER)
    cache_before = os.path.isdir(os.path.join(HERE, "__pycache__"))
    P.pinned_preflight(args.pinned_root)
    s = P.need_pinned()
    torch = s.torch
    tmp = tempfile.mkdtemp(prefix="dlat_f1test_")
    t0 = time.monotonic()
    base = SynBinding(s)
    info = {}

    # ---- T-F1: the complete run passes (A1-A12)
    if not args.skip_full:
        rep, rc = run(tmp, "full", base, card(), banner=BANNER)
        out = os.path.join(tmp, "full")
        check("T-F1", "complete run: exit 0, F1: PASS, all twelve criteria PASS", rc == 0 and rep["verdict"] == "F1: PASS"
              and all(rep["criteria"][c]["status"] == "PASS" for c in F.CRITERIA), json.dumps(rep.get("stop")))
        names = sorted(n for n in os.listdir(out) if not n.startswith(".") and not n.endswith(".sha256"))
        lines = [ln.split("  ") for ln in open(os.path.join(out, "outputs_sha256.txt")).read().splitlines()]
        check("T-F1", "outputs: report, verdict, log and manifest only; every manifest hash equals the file (A12)",
              names == ["f1.log", "f1_report.json", "f1_verdict.txt", "outputs_sha256.txt"]
              and sorted(n for _h, n in lines) == ["f1.log", "f1_report.json", "f1_verdict.txt"]
              and all(BD.sha_file(os.path.join(out, n)) == hh for hh, n in lines), names)
        on_disk = json.load(open(os.path.join(out, "f1_report.json")))
        check("T-F1", "report on disk has exactly the schema keys and no record, measure, contrast or class key (A11, C5)",
              set(on_disk) == F.REPORT_KEYS and F.check_report(on_disk))
        text = open(os.path.join(out, "f1_report.json")).read() + open(os.path.join(out, "f1.log")).read()
        check("T-F1", "no record key name appears as a JSON key in the report or the log (C5)",
              not any('"%s":' % k in text for k in F.REFUSED_KEYS))
        b = rep["binding"]
        check("T-F1", "C3: 30 assignments, 28 states, each loaded once; state table lists both shared states with two roles",
              b["n_role_assignments"] == 30 and b["n_states"] == 28 and base.loads == 28
              and sorted(len(x["roles"]) for x in b["states"]).count(2) == 2)
        pr, pj = rep["probe"], rep["projection"]
        check("T-F1", "probe: 4 states x 3 slices, 12 complete records at K = 128, timings and byte counts only (A9)",
              len(pr["states"]) == 4 and len(set(pr["states"])) == 4 and len(pr["record_s"]) == 12 and len(pr["record_bytes"]) == 12
              and pr["states"] == [base.role_map[(r, 2026)] for r in F.PROBE_ROLES] and pr["slices"] == [0, 1, 2])
        w = 28 * pr["s"] + 28 * N_SLICES * pr["t"] + pr["t_stat"]
        r = 28 * N_SLICES * pr["b"] + pr["R_fix"]
        check("T-F1", "W, R, L, D equal the formulas of 235 section 7 recomputed here; the final elapsed time is recorded and inside the ceiling",
              0 < pj["f1_elapsed_final_s"] < 3600 and abs(pj["W_s"] - w) < 1e-9 and pj["R_bytes"] == r and abs(pj["L_s"] - 1.5 * (w + max(600.0, 0.1 * w))) < 1e-9
              and abs(pj["D_bytes"] - 1.5 * (r + max(0.5 * GD.GIB, r))) < 1e-3 and pr["s"] == max(pr["load_s"]) and pr["t"] == max(pr["record_s"]))
        fx = rep["fixtures"]
        n_ss = 28 * N_SLICES
        check("T-F1", "fixtures 1-4, 6, 7 on every slice-state, fixture 2 (ii) on the 12 probe records, within the card tolerances (A6)",
              all(fx[k]["n"] == n_ss and fx[k]["max"] <= fx[k]["tolerance"] for k in
                  ("stored_NLL_abs_diff", "stored_B_abs_diff", "stored_J_abs_diff", "fx3_l2_identity_abs", "fx7_relative", "fx1_roundtrip_max_abs"))
              and fx["fx2ii_roundtrip_max_abs"]["n"] == 12 and fx["fx2ii_roundtrip_max_abs"]["max"] <= 1e-5
              and fx["fx8_beyond_bound_outside_tail"]["total"] == 0, {k: v.get("max") for k, v in fx.items()})
        check("T-F1", "swap and perturbation fixtures: S1 on every slice-state, S2, S4 and P1 on every slice, S3 pairs written (A7)",
              fx["S1_cond_over_tolerance"]["n"] == n_ss and fx["S1_cond_over_tolerance"]["max"] <= 1.0 and fx["S1_nll_abs_diff"]["max"] <= 1e-4
              and fx["P1_nll_abs_diff"]["n"] == n_ss and fx["S2_excluded_over_tolerance"]["n"] == N_SLICES
              and fx["P1_cond_over_tolerance"]["max"] == 0.0 and fx["S4_relative_l2"]["min"] > 0.0
              and fx["S3"]["pairs"] == [[list(base.identities[i]), list(base.identities[(i + 1) % N_SLICES])] for i in range(N_SLICES)])
        ar = rep["arrays"]
        idx = ST.bootstrap_matrix(n_volumes=N_SLICES, n_boot=200)
        check("T-F1", "three array identities recorded with shape, dtype, NumPy version and call; bootstrap identity recomputed (A8)",
              ar["bootstrap"]["matrix_identity"] == ST.matrix_identity(idx) and ar["noise"]["shape"] == [N_SLICES, C.N_ACQUIRED, 2]
              and ar["projection"]["shape"] == [C.D_FREE, C.M_PROJ] and ar["registered_sizes"] is False
              and all(set(ar[k]) >= {"sha256", "shape", "dtype", "numpy", "call"} for k in ("bootstrap", "noise", "projection")))
        proj = C.projection_matrix(np.random.Generator(np.random.PCG64(F.PROJ_SEED)).standard_normal(size=(C.D_FREE, C.M_PROJ)))
        check("T-F1", "the projection identity is that of the registered call (seed 20261122, full size)",
              ar["projection"]["sha256"] == C.array_identity(proj)["sha256"])
        check("T-F1", "device record: cpu, float32 parameters, thread count; memory and written bytes recorded (A5, A10)",
              rep["device"]["models"]["devices"] == ["cpu"] and rep["device"]["slice_state_tensors"] == ["cpu"]
              and rep["memory"]["peak_rss_bytes"] > 0 and rep["memory"]["disk_written_bytes"] >= 0)
        info["full"] = {"elapsed_s": time.monotonic() - t0, "s": pr["s"], "t": pr["t"], "t_max_over_min": pr["t_max_over_min"],
                        "t_stat": pr["t_stat"], "b": pr["b"], "R_fix": pr["R_fix"], "peak_rss_bytes": rep["memory"]["peak_rss_bytes"],
                        "fixture_max": {k: v.get("max") for k, v in fx.items() if isinstance(v, dict) and "max" in v},
                        "note": "sandbox CPU, 2 threads, synthetic tensors, 3 slices: not a host measurement"}

    # ---- A1
    saved = sys.dont_write_bytecode
    sys.dont_write_bytecode = False
    try:
        stops("T-A1", "interpreter without -B: DIAG_LAT_ORIGIN", tmp, base, card(), A.ORIGIN, "A1")
    finally:
        sys.dont_write_bytecode = saved
    stops("T-A1", "run card self-test flag differs from the binding's: DIAG_LAT_RUN_CARD", tmp, variant(base, selftest=False), card(), F.RUN_CARD, "A1")

    # ---- A2 (C3, C9)
    rm = dict(base.role_map)
    rm[("lam0/e3", 2026)] = rm[("lam0/e10", 2026)]
    stops("T-A2", "C3: 30 assignments resolving to 27 states: DIAG_LAT_POPULATION_MISMATCH", tmp, variant(base, role_map=rm), card(),
          ST.POPULATION_MISMATCH, "A2")
    rm = {k: v for k, v in base.role_map.items() if k != ("lam2/fin", 2028)}
    stops("T-A2", "a registered role missing: DIAG_LAT_POPULATION_MISMATCH", tmp, variant(base, role_map=rm), card(), ST.POPULATION_MISMATCH, "A2")

    class WrongState(SynBinding):
        def __init__(self):
            pass

        def load_state(self, sid):
            m, v = SynBinding.load_state(self, sid)
            return m, dict(v, model_state_sha256=h("another state"))

    class WrongFile(WrongState):
        def load_state(self, sid):
            m, v = SynBinding.load_state(self, sid)
            return m, dict(v, file_sha256=h("another file"))

    class Refusing(WrongState):
        def load_state(self, sid):
            raise DiagLatError(A.ORIGIN, "synthetic: the loader refused the file")

    for cls, name in ((WrongState, "C9: the loaded model state differs from the manifest"),
                      (WrongFile, "C9: the loaded file hash differs from the manifest"), (Refusing, "the loader refuses a file")):
        b = cls()
        b.__dict__.update(base.__dict__)
        stops("T-A2", "%s: DIAG_LAT_ORIGIN" % name, tmp, b, card(), A.ORIGIN, "A2")

    def failing_factory():
        raise DiagLatError(A.ORIGIN, "synthetic: a checkpoint file differs from the manifest")
    stops("T-A2", "the binding itself fails (production: bind_manifest or verify_files): reported under A2", tmp, failing_factory, card(),
          A.ORIGIN, "A2")

    # ---- A3
    stops("T-A3", "slices in another order: DIAG_LAT_POPULATION_MISMATCH", tmp, variant(base, _slices=base._slices[::-1]), card(),
          ST.POPULATION_MISMATCH, "A3")
    stops("T-A3", "mask bytes differ from the registered hash: DIAG_LAT_ORIGIN", tmp, variant(base, masks_sha256=h("x")), card(), A.ORIGIN, "A3")
    bank2 = base._bank.copy()
    bank2[0, 0] += 1.0
    stops("T-A3", "bank changed: DIAG_LAT_ORIGIN", tmp, variant(base, _bank=bank2), card(), A.ORIGIN, "A3")
    stops("T-A3", "bank with 64 rows: DIAG_LAT_ORIGIN", tmp, variant(base, _bank=base._bank[:64]), card(), A.ORIGIN, "A3")
    st2 = dict(base.stored)
    del st2[base.state_order[3]]
    stops("T-A3", "stored rows of one state missing: DIAG_LAT_POPULATION_MISMATCH", tmp, variant(base, stored=st2), card(), ST.POPULATION_MISMATCH, "A3")
    stops("T-A3", "two slices of one volume: DIAG_LAT_POPULATION_MISMATCH", tmp,
          variant(base, volume_ids=[base.volume_ids[0]] * 2 + base.volume_ids[2:]), card(), ST.POPULATION_MISMATCH, "A3")

    # ---- A4 (child processes)
    r = child(["--checkpoint", "x.pt", "--run-card", "c", "--run-card-sha256", "0", "--repo", ".", "--manifest", "m", "--out", os.path.join(tmp, "r1")])
    check("T-A4", "--checkpoint refused before any import, exit 3, nothing written", r.returncode == 3 and "DIAG_LAT_REFUSED" in r.stderr
          and not os.path.exists(os.path.join(tmp, "r1")))
    r = child(["--data-root=/x", "--run-card", "c", "--run-card-sha256", "0", "--repo", ".", "--manifest", "m", "--out", os.path.join(tmp, "r2")])
    check("T-A4", "--data-root refused, exit 3", r.returncode == 3 and not os.path.exists(os.path.join(tmp, "r2")))
    r = child(["--tolerance", "1", "--run-card", "c", "--run-card-sha256", "0", "--repo", ".", "--manifest", "m", "--out", os.path.join(tmp, "r3")])
    check("T-A4", "an unknown option is refused, exit 2, nothing written", r.returncode == 2 and not os.path.exists(os.path.join(tmp, "r3")))

    # ---- A5
    raises("T-A5", "run card device cuda: DIAG_LAT_REPRODUCTION_MISMATCH, no output folder", F.REPRODUCTION_MISMATCH,
           lambda: run(tmp, "cuda", base, card(device="cuda")))
    check("T-A5", "... and nothing was written", not os.path.exists(os.path.join(tmp, "cuda")))
    stops("T-A5", "a slice-state tensor on another device: DIAG_LAT_REPRODUCTION_MISMATCH", tmp,
          with_slice(base, 1, extra=torch.empty(1, device="meta")), card(), F.REPRODUCTION_MISMATCH, "A5")

    class Fake:
        def __init__(self, model, dtype):
            self._p = [torch.zeros(1, dtype=dtype)]

        def parameters(self):
            return iter(self._p)

    class OtherPrecision(WrongState):
        def load_state(self, sid):
            m, v = SynBinding.load_state(self, sid)
            return Fake(m, torch.float64), v
    b = OtherPrecision()
    b.__dict__.update(base.__dict__)
    stops("T-A5", "model parameters not float32: DIAG_LAT_REPRODUCTION_MISMATCH", tmp, b, card(), F.REPRODUCTION_MISMATCH, "A5")

    # ---- A6
    for name in ("NLL", "B", "J"):
        st2 = {k: dict(v) for k, v in base.stored.items()}
        first = base.state_order[0]
        st2[first][name] = st2[first][name] + np.array([0.0, 2e-4, 0.0])
        stops("T-A6", "stored %s differs by 2e-4 nats/dim on one slice: DIAG_LAT_ADAPTER" % name, tmp, variant(base, stored=st2), card(), A.ADAPTER, "A6")
    stops("T-A6", "fixture 1: a round-trip tolerance below the float32 residual stops: DIAG_LAT_ADAPTER", tmp, base,
          card(tolerances={"fx1_roundtrip_max_abs": 1e-12}), A.ADAPTER, "A6")
    stops("T-A6", "fixture 7: a tolerance below the residual stops: DIAG_LAT_ADAPTER", tmp, base, card(tolerances={"fx7_relative": 1e-13}), A.ADAPTER, "A6")
    stops("T-A6", "fixture 2 (ii): a tolerance below the residual stops at the first probe record: DIAG_LAT_ADAPTER", tmp, base,
          card(tolerances={"fx2ii_roundtrip_max_abs": 1e-12}), A.ADAPTER, "A6")

    # ---- A7
    stops("T-A7", "P1: a second division by amax (stored cond halved): DIAG_LAT_PERTURB_FIXTURE", tmp,
          with_slice(base, 1, cond=base._slices[1]["cond"] * 0.5), card(), A.PERTURB_FIXTURE, "A7")
    stops("T-A7", "S1: the stored truth image differs from the measurement by 1e-3 (fixture 7 tolerance opened): DIAG_LAT_SWAP_FIXTURE", tmp,
          with_slice(base, 0, x_true_c=base._slices[0]["x_true_c"] * (1.0 + 1e-3)), card(tolerances={"fx7_relative": 1e-1}), C.SWAP_FIXTURE, "A7")

    # ---- A8
    y = base._slices[2]["y"].clone()
    cols = np.flatnonzero(~base._slices[2]["mask"][0].numpy())
    y[0, 5, int(cols[0])] = 1.0
    stops("T-A8", "measurement non-zero outside the acquired cells: DIAG_LAT_BOOTSTRAP_IDENTITY", tmp, with_slice(base, 2, y=y), card(),
          ST.BOOTSTRAP_IDENTITY, "A8")
    stops("T-A8", "RMS(y) = 0 on one slice: DIAG_LAT_ZERO_DENOMINATOR", tmp, with_slice(base, 1, y=torch.zeros_like(base._slices[1]["y"])), card(),
          C.ZERO_DENOMINATOR, "A8")
    yn = base._slices[0]["y"].clone()
    yn[0, 0, int(np.flatnonzero(base._slices[0]["mask"][0].numpy())[0])] = float("nan")
    stops("T-A8", "non-finite measurement: DIAG_LAT_NONFINITE", tmp, with_slice(base, 0, y=yn), card(), C.NONFINITE, "A8")

    # ---- A9 (D8 called directly: the population stage on synthetic input)
    def d8(c, n_vol, n_boot, role_map, order, seeds):
        r = F.F1Run(c, None, "direct")
        r.b = variant(base, role_map=role_map, state_order=order, seeds=seeds)
        t = time.monotonic()
        r.idx = ST.bootstrap_matrix(n_volumes=n_vol, n_boot=n_boot)
        r.t_matrix = time.monotonic() - t
        r.idx_sha = ST.matrix_identity(r.idx)
        r.probe = {"load_s": [1.0, 2.0, 1.5, 1.2], "record_s": [3.0] * 11 + [4.0], "record_bytes": [9000] * 11 + [9500]}
        r.d8(12345)
        return r
    small = (N_SLICES, 200, base.role_map, base.state_order, base.seeds)
    r = d8(card(), *small)
    pr, pj = r.rep["probe"], r.rep["projection"]
    w = 28 * 2.0 + 28 * N_SLICES * 4.0 + pr["t_stat"]
    check("T-A9", "D8: s, t, b are the maxima; W = 28 s + n t + t_stat; the missing-value branch was run; status PASS",
          pr["s"] == 2.0 and pr["t"] == 4.0 and pr["b"] == 9500 and abs(pj["W_s"] - w) < 1e-9 and pr["missing_value_events"] == 1
          and abs(pr["t_max_over_min"] - 4.0 / 3.0) < 1e-12 and r.crit["A9"]["status"] == "PASS" and "SYNTHETIC" in pr["population"])
    raises("T-A9", "L above the G wall ceiling: DIAG_LAT_RESOURCE_LIMIT", GD.RESOURCE_LIMIT, lambda: d8(card(ceilings={"g_wall_s": 600}), *small))
    raises("T-A9", "D above the disk ceiling: DIAG_LAT_RESOURCE_LIMIT", GD.RESOURCE_LIMIT, lambda: d8(card(ceilings={"disk_bytes": GD.GIB // 2}), *small))
    t = time.monotonic()
    r = d8(card(selftest=True, registered={"n_volumes": 97, "n_boot": 10000}), 97, 10000, base.role_map, base.state_order, base.seeds)
    info["d8_registered_size"] = {"seconds": time.monotonic() - t, "t_stat": r.rep["probe"]["t_stat"], "R_fix": r.rep["probe"]["R_fix"],
                                  "bootstrap_matrix_sha256": r.idx_sha, "peak_rss_bytes": GD.peak_rss_bytes(),
                                  "note": "sandbox CPU: the population stage at the registered size (28 states, 97 volumes, 21 contrasts, "
                                          "10,000 draws) on SYNTHETIC records; not a host measurement"}
    check("T-A9", "D8 at the registered size (97 volumes, 10,000 draws) completes on synthetic records (C7)",
          r.crit["A9"]["status"] == "PASS" and r.rep["probe"]["t_stat"] > 0 and r.rep["probe"]["R_fix"] > 0)

    # ---- A10
    stops("T-A10", "peak memory above the limit after the first state: DIAG_LAT_RESOURCE_LIMIT", tmp, base, card(ceilings={"ram_bytes": 2 ** 20}),
          GD.RESOURCE_LIMIT, "A10")

    class OutOfMemory(WrongState):
        def load_state(self, sid):
            raise MemoryError("synthetic")
    b = OutOfMemory()
    b.__dict__.update(base.__dict__)
    stops("T-A10", "out of memory while loading a state is caught: DIAG_LAT_RESOURCE_LIMIT under A10", tmp, b, card(), GD.RESOURCE_LIMIT, "A10")

    class Slow(WrongState):
        def load_state(self, sid):
            time.sleep(1.2)
            return SynBinding.load_state(self, sid)
    b = Slow()
    b.__dict__.update(base.__dict__)
    stops("T-A10", "F1 wall ceiling exceeded after a state: DIAG_LAT_RESOURCE_LIMIT", tmp, b, card(ceilings={"f1_wall_s": 1}), GD.RESOURCE_LIMIT, "A10")

    # ---- the final elapsed-time check (stage methods replaced by TEST stand-ins that only take time)
    saved_methods = {n: getattr(F.F1Run, n) for n in ("d1", "d2_d3", "d4_tensors", "d5", "slice_fixtures", "d6_d7", "d8")}

    def late_d8(self, log_bytes):
        self.rep["projection"] = {}
        time.sleep(1.3)                          # the last stage ends after a 1 s ceiling; no check runs inside it
    try:
        for n in saved_methods:
            setattr(F.F1Run, n, (lambda self, *a: None))
        F.F1Run.d8 = late_d8
        stops("T-WALL", "every stage completes but the run ends after the wall ceiling: F1: FAIL, DIAG_LAT_RESOURCE_LIMIT under A9", tmp, base,
              card(ceilings={"f1_wall_s": 1}), GD.RESOURCE_LIMIT, "A9")
        rep, rc = run(tmp, "wall_ok", base, card())
        check("T-WALL", "the same run inside the ceiling is not stopped; its final elapsed time is recorded",
              rep["stop"] is None and 1.3 <= rep["projection"]["f1_elapsed_final_s"] < 3600)
    finally:
        for n, m in saved_methods.items():
            setattr(F.F1Run, n, m)

    # ---- A11
    good = {k: None for k in F.REPORT_KEYS}
    check("T-A11", "a report with exactly the schema keys passes", F.check_report(good))
    raises("T-A11", "an extra top-level key: DIAG_LAT_RECORD_SCHEMA", A.RECORD_SCHEMA, lambda: F.check_report(dict(good, extra=1)))
    raises("T-A11", "a missing top-level key: DIAG_LAT_RECORD_SCHEMA", A.RECORD_SCHEMA,
           lambda: F.check_report({k: None for k in F.REPORT_KEYS if k != "probe"}))
    for key in ("KS", "TC_G", "P_swap", "L4", "S-GAP0", "classified", "record", A.BANK_FIELD):
        raises("T-A11", "a nested key %r: DIAG_LAT_RECORD_SCHEMA" % key, A.RECORD_SCHEMA,
               lambda key=key: F.check_report(dict(good, probe={"x": [{"y": {key: 1.0}}]})))
    saved = F.REFUSED_KEYS
    F.REFUSED_KEYS = saved | {"banner"}
    try:
        out = os.path.join(tmp, "a11")
        rep, rc = F.run_f1(variant(base, selftest=False), card(), out)
        names = sorted(n for n in os.listdir(out) if not n.startswith(".") and not n.endswith(".sha256"))
        check("T-C11", "a run whose report fails its schema: no f1_report.json; f1_failure_report.json with the failure schema, F1: FAIL, exit 1",
              rc == 1 and names == ["f1.log", "f1_failure_report.json", "f1_verdict.txt", "outputs_sha256.txt"]
              and set(rep) == F.FAILURE_KEYS and rep["schema"] == F.FAILURE_SCHEMA and rep["verdict"] == "F1: FAIL"
              and rep["criteria"]["A11"] == {"status": "FAIL", "stop": A.RECORD_SCHEMA} and rep["stop"]["code"] == A.RECORD_SCHEMA
              and rep["criteria"]["A1"] == {"status": "FAIL", "stop": F.RUN_CARD}
              and json.load(open(os.path.join(out, "f1_failure_report.json"))) == rep
              and open(os.path.join(out, "f1_verdict.txt")).read().endswith("F1: FAIL\n"), names)
    finally:
        F.REFUSED_KEYS = saved

    crit = {c: {"status": F.NOT_REACHED, "observed": None, "limit": None, "stop": None} for c in F.CRITERIA}
    crit["A11"].update(status="FAIL", stop=A.RECORD_SCHEMA)
    fr = F.failure_report(crit, A.RECORD_SCHEMA, "m", "b", "s")
    check("T-C11", "a failure report has exactly its keys and validates", set(fr) == F.FAILURE_KEYS and F.check_failure_report(fr))
    for name, bad in (("verdict F1: PASS", dict(fr, verdict="F1: PASS")), ("the full report schema name", dict(fr, schema=F.REPORT_SCHEMA)),
                      ("an extra key", dict(fr, fixtures={})), ("a criterion with an observed value", dict(fr, criteria=crit)),
                      ("A11 not FAIL", dict(fr, criteria=dict(fr["criteria"], A11={"status": "PASS", "stop": None}))),
                      ("a stop without its criterion", dict(fr, stop={"code": "x", "message": "m"}))):
        raises("T-C11", "a failure report with %s is refused: DIAG_LAT_RECORD_SCHEMA" % name, A.RECORD_SCHEMA, lambda bad=bad: F.check_failure_report(bad))

    # ---- A12 and the run card
    os.makedirs(os.path.join(tmp, "exists"))
    raises("T-A12", "an existing output folder is refused: DIAG_LAT_GUARD_STATE", GD.GUARD_STATE, lambda: run(tmp, "exists", base, card()))
    good_card = card()
    path = os.path.join(tmp, "card.json")
    raw = json.dumps(good_card, indent=1).encode()
    open(path, "wb").write(raw)
    sha = hashlib.sha256(raw).hexdigest()
    check("T-CARD", "a run card with its sha256 loads", F.load_card(path, sha) == (good_card, sha))
    raises("T-CARD", "wrong sha256: DIAG_LAT_RUN_CARD", F.RUN_CARD, lambda: F.load_card(path, h("x")))
    raises("T-CARD", "missing file: DIAG_LAT_RUN_CARD", F.RUN_CARD, lambda: F.load_card(path + ".none", sha))
    bad = [("a missing key", {k: v for k, v in good_card.items() if k != "ceilings"}), ("an extra key", dict(good_card, extra=1)),
           ("a tolerance that is not a float", card(tolerances={"fx7_relative": 1})), ("a negative tolerance", card(tolerances={"fx7_relative": -1e-5})),
           ("an extra tolerance", card(tolerances={"other": 1e-5})), ("a ceiling that is not an integer", card(ceilings={"g_wall_s": 21600.0})),
           ("another rules version", card(rules_version="v0.9")), ("an S1 tolerance other than the adapter's", card(tolerances={"s1_s2_p1_relative": 1e-4})),
           ("non-registered sizes without the self-test flag", card(selftest=False)), ("another schema", card(schema="x")),
           ("an empty not-verified statement (C12)", card(not_verified_before_run=" ")), ("a manifest hash that is not a string", card(manifest_f2_sha256=1))]
    for name, c in bad:
        raises("T-CARD", "%s: DIAG_LAT_RUN_CARD" % name, F.RUN_CARD, lambda c=c: F.check_card(c))
    r = child(["--run-card", path, "--run-card-sha256", sha, "--repo", tmp, "--manifest", path, "--out", os.path.join(tmp, "st")])
    check("T-CARD", "the production entry point refuses a self-test run card, exit 1, nothing written",
          r.returncode == 1 and "DIAG_LAT_RUN_CARD" in r.stdout and not os.path.exists(os.path.join(tmp, "st")))

    # ---- C10: the production card
    prod_raw = open(args.run_card, "rb").read()
    prod_sha = hashlib.sha256(prod_raw).hexdigest()
    prod = json.loads(prod_raw)
    check("T-C10", "the draft production run card passes the driver's checks: registered manifest hash, pin, sizes, cpu",
          F.load_card(args.run_card, prod_sha)[0] == prod and prod["selftest"] is False and prod["manifest_f2_sha256"] == BD.REGISTERED_F2_SHA256
          and prod["pin"] == BD.PIN and prod["registered"] == F.REGISTERED and prod["device"] == "cpu"
          and BD.sha_file(args.real_manifest) == BD.REGISTERED_F2_SHA256)
    raises("T-C10", "a production card naming another manifest hash: DIAG_LAT_RUN_CARD", F.RUN_CARD,
           lambda: F.check_card(dict(prod, manifest_f2_sha256=h("another manifest"))))
    raises("T-C10", "a production card naming another pin: DIAG_LAT_RUN_CARD", F.RUN_CARD, lambda: F.check_card(dict(prod, pin="0" * 40)))

    # ---- bind-only (C6, C8) and the production binding on the synthetic results folder of the binding test (child processes)
    repo = os.path.join(tmp, "repo")
    root = os.path.join(repo, *BD.RESULTS_SUBDIR)
    os.makedirs(root)
    mf, _expect = BT.build(root)
    mpath, msha = BT.write_manifest(tmp, mf)
    scard = os.path.join(tmp, "card_syn.json")
    json.dump(card(manifest_f2_sha256=msha, pin=BD.PIN, registered={"n_volumes": 97, "n_boot": 10000}), open(scard, "w"))
    bind_code = ("card = json.load(open(sys.argv[1]))\nF.check_card(card)\nrep, rc = F.bind_only(sys.argv[2], sys.argv[3], card, sys.argv[4], 'synthetic')\n"
                 "print('BIND: PASS' if rc == 0 else 'BIND: FAIL')\nsys.exit(rc)")
    before = {os.path.join(d, f): os.stat(os.path.join(d, f)).st_mtime_ns for d, _s, fs in os.walk(repo) for f in fs}
    out = os.path.join(tmp, "bind1")
    r = snippet(bind_code, scard, repo, mpath, out)
    rep = json.load(open(os.path.join(out, "bind_report.json"))) if os.path.isfile(os.path.join(out, "bind_report.json")) else {}
    oc = rep.get("outcome") or {}
    check("T-BIND", "C8: bind-only on the results folder: exit 0, BIND: PASS, complete outcome reported", r.returncode == 0 and "BIND: PASS" in r.stdout
          and oc.get("n_role_assignments") == 30 and oc.get("n_states") == 28 and oc.get("n_checkpoint_files") == 29 and oc.get("files_hashed") == 59
          and oc.get("n_midpoint_slices") == 97 and oc.get("n_volumes") == 97 and len(oc.get("states", [])) == 28
          and len(oc.get("role_map", {})) == 30 and sorted(oc.get("volume_order", [])) == list(range(97)) and rep.get("stop") is None,
          r.stdout + r.stderr[-300:])
    check("T-BIND", "C6, C12: no PyTorch imported; the results folder is unchanged; the report is the only new content and says what it does not verify",
          oc.get("torch_imported") is False and rep.get("not_verified_by_this_preflight") == "synthetic"
          and before == {os.path.join(d, f): os.stat(os.path.join(d, f)).st_mtime_ns for d, _s, fs in os.walk(repo) for f in fs}
          and sorted(n for n in os.listdir(out) if not n.startswith(".") and not n.endswith(".sha256"))
          == ["bind_report.json", "bind_verdict.txt", "outputs_sha256.txt"])
    victim = os.path.join(root, mf["roles"][4]["path"])
    orig = open(victim, "rb").read()
    open(victim, "wb").write(orig + b"x")
    out = os.path.join(tmp, "bind2")
    r = snippet(bind_code, scard, repo, mpath, out)
    rep = json.load(open(os.path.join(out, "bind_report.json"))) if os.path.isfile(os.path.join(out, "bind_report.json")) else {}
    check("T-BIND", "C8: a changed checkpoint file: exit 1, BIND: FAIL, DIAG_LAT_ORIGIN in the report",
          r.returncode == 1 and "BIND: FAIL" in r.stdout and (rep.get("stop") or {}).get("code") == A.ORIGIN and rep.get("outcome") is None)
    open(victim, "wb").write(orig)
    out = os.path.join(tmp, "prod1")
    r = snippet("card = json.load(open(sys.argv[1]))\n"
                "def make():\n    b = F.ProductionBinding(sys.argv[2], sys.argv[3], card)\n    b.selftest = True\n    return b\n"
                "rep, rc = F.run_f1(make, card, sys.argv[4])\nprint(rep['verdict'])\nsys.exit(rc)", scard, repo, mpath, out)
    rep = json.load(open(os.path.join(out, "f1_report.json"))) if os.path.isfile(os.path.join(out, "f1_report.json")) else {}
    check("T-PROD", "the production binding binds the folder at hash level, then stops typed at the first pinned data call, which the sandbox "
                    "cannot serve: F1: FAIL under A3 with its evidence written (C12: these calls are not verified here)",
          r.returncode == 1 and "F1: FAIL" in r.stdout and (rep.get("stop") or {}).get("criterion") == "A3"
          and (rep.get("stop") or {}).get("code") in (A.ADAPTER, A.ORIGIN) and rep["criteria"]["A1"]["status"] == "PASS"
          and rep["binding"] is None, json.dumps(rep.get("stop")) + r.stderr[-300:])

    # ---- the production entry point: production card, the REAL registered manifest, a checkout without results
    empty = os.path.join(tmp, "empty_repo")
    os.makedirs(empty)
    common = ["--run-card", args.run_card, "--run-card-sha256", prod_sha, "--repo", empty, "--manifest", args.real_manifest]
    import shutil
    if shutil.disk_usage(tmp).free < prod["ceilings"]["reserve_bytes"]:
        # The sandbox has less free disk than the registered reserve: the unmodified entry point must refuse to write.
        out = os.path.join(tmp, "main_reserve")
        r = child(common + ["--out", out, "--bind-only"])
        check("T-MAIN", "entry point on a disk below the registered reserve: BIND: FAIL (DIAG_LAT_RESOURCE_LIMIT), no report written",
              r.returncode == 1 and "BIND: FAIL (DIAG_LAT_RESOURCE_LIMIT)" in r.stdout and not os.path.exists(os.path.join(out, "bind_report.json")))
        info["reserve"] = "sandbox free disk is below the 64 GiB reserve: the T-MAIN checks below call main() in a child with the free-space reading replaced (test only)"

    def child_main(argv):       # main() of the driver in a child process, the free-space reading of the write guard replaced (TEST ONLY)
        return snippet("import diag_lat_guard as GD\norig = GD.WriteGuard\n"
                       "F.GD.WriteGuard = lambda root, ceiling, reserve, free_fn=None: orig(root, ceiling, reserve, free_fn=lambda: 2 ** 42)\n"
                       "sys.argv = ['diag_lat_f1.py'] + sys.argv[1:]\nsys.exit(F.main())", *argv)
    out = os.path.join(tmp, "main_bind")
    r = child_main(common + ["--out", out, "--bind-only"])
    rep = json.load(open(os.path.join(out, "bind_report.json"))) if os.path.isfile(os.path.join(out, "bind_report.json")) else {}
    check("T-MAIN", "entry point, --bind-only, real manifest, no results folder: exit 1, BIND: FAIL, DIAG_LAT_ORIGIN reported, no PyTorch",
          r.returncode == 1 and "BIND: FAIL (DIAG_LAT_ORIGIN)" in r.stdout and (rep.get("stop") or {}).get("code") == A.ORIGIN, r.stdout + r.stderr[-300:])
    r = child_main(common + ["--out", out, "--bind-only"])
    check("T-MAIN", "an existing output folder is refused, exit 1", r.returncode == 1 and "BIND: FAIL (DIAG_LAT_GUARD_STATE)" in r.stdout)
    out = os.path.join(tmp, "main_full")
    r = child_main(common + ["--out", out])
    check("T-MAIN", "the full run without --dunc-dir is refused, exit 1, nothing written", r.returncode == 1 and not os.path.exists(out))
    r = child_main(common + ["--out", out, "--dunc-dir", HERE])
    rep = json.load(open(os.path.join(out, "f1_report.json"))) if os.path.isfile(os.path.join(out, "f1_report.json")) else {}
    check("T-MAIN", "entry point, full run, real manifest, no results folder: the binding fails, F1: FAIL under A2 with its evidence written",
          r.returncode == 1 and "F1: FAIL" in r.stdout and (rep.get("stop") or {}) .get("criterion") == "A2"
          and (rep.get("stop") or {}).get("code") == A.ORIGIN, json.dumps(rep.get("stop")) + r.stderr[-300:])
    r = child_main(["--run-card", args.run_card, "--run-card-sha256", h("x"), "--repo", empty, "--manifest", args.real_manifest, "--out", os.path.join(tmp, "m3")])
    check("T-MAIN", "entry point with a wrong run-card hash: exit 1, DIAG_LAT_RUN_CARD, nothing written",
          r.returncode == 1 and "DIAG_LAT_RUN_CARD" in r.stdout and not os.path.exists(os.path.join(tmp, "m3")))

    # ---- no bytecode
    check("T-ENV", "this test ran with -I -B and created no bytecode cache", bool(sys.flags.isolated) and sys.dont_write_bytecode
          and os.path.isdir(os.path.join(HERE, "__pycache__")) == cache_before)

    n_fail = sum(1 for c in CHECKS if not c["pass"])
    complete = not args.skip_full
    report = {"banner": BANNER, "test": ABBR, "version": __version__, "driver_version": F.__version__, "complete": complete,
              "n_checks": len(CHECKS), "n_fail": n_fail, "checks": CHECKS, "info": info, "elapsed_s": time.monotonic() - t0,
              "result": "PASS" if (n_fail == 0 and complete) else ("INCOMPLETE" if n_fail == 0 else "FAIL")}
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    print("%s: %d checks, %d failed -- %s (%.0f s)" % (ABBR, len(CHECKS), n_fail, report["result"], report["elapsed_s"]))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
