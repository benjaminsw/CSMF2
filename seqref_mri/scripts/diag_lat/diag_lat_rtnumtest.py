# SEQREF-DLAT-RTNUM-T v0.1 -- diag_lat_rtnumtest.py: synthetic test of the round-trip numerical diagnostic diag_lat_rtnum.py v0.1 (plan SEQREF-DLAT-RTNUM v0.3)
# LIFETIME: DIAGNOSTIC
"""SANDBOX SYNTHETIC TEST -- NOT HOST EVIDENCE -- NOT D-LAT EVIDENCE.

Synthetic slices and fixture parameters on the pinned model code (no checkpoint, no MRI data).
It tests the diagnostic's code paths, not the trained checkpoints.

Groups:
  T-RULE   the tolerance series (1, 2, 5 x 10^k at least twice the development maximum)
  T-STAT   residual statistics on known arrays; typed stops
  T-SLICE  first, middle, last
  T-CARD   the production run card and every refusal of check_card
  T-SMALL  a complete run on a reduced synthetic binding (three states, one for each seed)
  T-DET    a second run gives identical statistics
  T-INPUT  models, bank and slice tensors unchanged by a run
  T-SUM    every condition of the proposal, on constructed statistics (no model)
  T-MIS    an injected forward and inverse mismatch in the float64 flow is seen, and no tolerance is proposed
  T-F64    the float64 copy: a copy that is not float64, or differs in value, stops
  T-STOP   typed stops with evidence written
  T-SCHEMA a measure key or a non-finite number in the report: no report of statistics is written
  T-MAIN   the production entry point (child processes)
  T-FULL   the registered structure: 28 states, 84 slice-states, sets 30 and 54 (slow; --skip-full omits it
           and the report is then marked INCOMPLETE)
Not tested here (disclosed): the -I -B refusal (the same lines as the F1 driver's, tested there in a
child process); the production data calls (no data in the sandbox).
No fallback, placeholder, mock or dummy return in the code under test; the test replaces functions
only where a group says so, and restores them.

Changelog:
  v0.1 (2026-11-23) first version.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import diag_lat_adapter as A  # noqa: E402
import diag_lat_bind as BD  # noqa: E402
import diag_lat_core as C  # noqa: E402
import diag_lat_f1 as F1  # noqa: E402
import diag_lat_f1test as T  # noqa: E402
import diag_lat_guard as GD  # noqa: E402
import diag_lat_p3test as P  # noqa: E402
import diag_lat_rtnum as R  # noqa: E402
from diag_lat_core import DiagLatError  # noqa: E402

__version__ = "0.1"
BANNER = "SANDBOX SYNTHETIC TEST -- NOT HOST EVIDENCE -- NOT D-LAT EVIDENCE (synthetic tensors)"
CHECKS = []
SMALL_K = 16                        # bank rows of the reduced binding (speed); T-FULL uses the registered 128
FILES = ("rtnum_report.json", "rtnum_verdict.txt", "rtnum.log", "outputs_sha256.txt")


def check(tid, name, ok, note=""):
    CHECKS.append({"id": tid, "name": name, "ok": bool(ok), "note": str(note)})
    print("%s %-8s %s%s" % ("PASS" if ok else "FAIL", tid, name, (" [" + str(note) + "]") if (note and not ok) else ""))
    return bool(ok)


def raises(tid, name, code, fn):
    try:
        fn()
    except DiagLatError as exc:
        return check(tid, name, exc.code == code, "got %s" % exc.code)
    except Exception as exc:
        return check(tid, name, False, "%s: %s" % (type(exc).__name__, exc))
    return check(tid, name, False, "no stop")


def card(**over):
    c = {"schema": R.CARD_SCHEMA, "selftest": True, "plan": "synthetic", "device": "cpu",
         "manifest_f2_sha256": T.h("synthetic manifest"), "pin": "SYNTHETIC-NOT-A-PIN",
         "registered": {"n_states": 3, "n_volumes": T.N_SLICES, "bank_rows": SMALL_K},
         "population": {"development_seeds": [2026], "validation_seeds": [2027, 2028], "n_slices": 3,
                        "n_development_slice_states": 3, "n_validation_slice_states": 6},
         "limits": {"present_tolerance_max_abs": 1e-5, "f64_roundtrip_max_abs": 1e-9, "nll_f64_vs_stored_nats_per_dim": 1e-4,
                    "nll_f64_vs_f32_nats_per_dim": 1e-5, "j_f32_vs_f64_nats_per_dim": 1e-5,
                    "decoded_bank_f32_vs_f64_max_abs": 1e-4, "tolerance_ceiling_max_abs": 1e-4},
         "ceilings": {"wall_s": 3600, "disk_bytes": 2 * GD.GIB, "reserve_bytes": GD.GIB, "ram_bytes": 12 * GD.GIB}}
    for k, v in over.items():
        c[k] = dict(c[k], **v) if isinstance(v, dict) else v
    return c


def small(base):
    """Three states of the synthetic binding, the first of each seed (development 1, validation 2), and its first SMALL_K bank rows."""
    b = copy.copy(base)
    order = []
    for seed in T.SEEDS:
        order.append(next(sid for sid in base.state_order if base.states[sid]["seed"] == seed))
    b.state_order = order
    b.states = {sid: dict(base.states[sid]) for sid in order}
    b.stored = {sid: {n: v.copy() for n, v in base.stored[sid].items()} for sid in order}
    b._bank = np.ascontiguousarray(base._bank[:SMALL_K])
    b.bank_sha256 = hashlib.sha256(b._bank.tobytes(order="C")).hexdigest()
    return b


def stops(tid, name, tmp, binding, c, code, done=None):
    """The run must end RTNUM: FAIL with this code, with its evidence written."""
    out = os.path.join(tmp, "neg_%d" % len(CHECKS))
    try:
        rep, rc = R.run_rtnum(binding, c, out, banner=BANNER)
    except DiagLatError as exc:
        return check(tid, name, False, "raised %s instead of reporting" % exc.code)
    got = rep.get("stop") or {}
    files = all(os.path.isfile(os.path.join(out, f)) for f in FILES)
    ok = (rc == 1 and got.get("code") == code and rep["verdict"] == "RTNUM: FAIL" and rep["summary"] is None and files
          and open(os.path.join(out, "rtnum_verdict.txt")).read() == "RTNUM: FAIL\n"
          and (done is None or got.get("slice_states_done") == done))
    return check(tid, name, ok, "got %s, exit %s, files %s, done %s" % (got.get("code"), rc, files, got.get("slice_states_done")))


def strip_time(rep):
    return json.dumps(rep["slice_states"], sort_keys=True), json.dumps(rep["summary"], sort_keys=True)


def state_hash(binding):
    h = hashlib.sha256()
    for k in sorted(binding.models):
        for name, p in sorted(binding.models[k].named_parameters()):
            h.update(name.encode())
            h.update(p.detach().numpy().tobytes())
    h.update(binding._bank.tobytes())
    torch = A.seams().torch
    for st in binding._slices:
        for key in sorted(st):
            if isinstance(st[key], torch.Tensor):
                h.update(st[key].numpy().tobytes())
            elif isinstance(st[key], np.ndarray):
                h.update(st[key].tobytes())
    return h.hexdigest()


def fake_record(state, set_name, pos, bank32=4e-6, target32=3e-6, f64=1e-14, **over):
    def st(v):
        return {"max_abs": v, "rms": v / 3, "max_abs_over_reference_max": v / 5, "rms_over_reference_rms": v / 3, "row": 0,
                "coordinate": 0, "reference_at_max": 1.0, "error_at_max": v, "distance_to_spline_bound": C.SPLINE_B - 1.0,
                "float32_spacings_at_max": v / 1.19e-7, "latent_max_abs": 4.0}
    r = {"state": state, "state_index": 0, "set": set_name, "slice_position": pos,
         "bank_direction": {"f32": st(bank32), "f64": st(f64)}, "target_direction": {"f32": st(target32), "f64": st(f64)},
         "nll_f32_vs_stored_abs": 1e-7, "nll_f64_vs_stored_abs": 1e-7, "nll_f64_vs_f32_abs": 1e-8,
         "j_bank_f32_vs_f64_max_abs": 1e-7, "j_target_f32_vs_f64_abs": 1e-8,
         "decoded_bank_f32_vs_f64": {"max_abs": 1e-6, "rms": 1e-7}, "direct_f32_vs_registered_max_abs": 0.0}
    r.update(over)
    return r


def fake_set(dev=None, val=None):
    recs = [fake_record("d%d" % i, "development", i, **(dev or {})) for i in range(3)]
    recs += [fake_record("v%d" % i, "validation", i, **(val or {})) for i in range(6)]
    return recs


def child_main(args):
    return subprocess.run([sys.executable, "-I", "-B", os.path.join(HERE, "diag_lat_rtnum.py")] + args,
                          capture_output=True, text=True, timeout=900)


def snippet(code, *argv):
    return subprocess.run([sys.executable, "-I", "-B", "-c",
                           "import sys, json\nsys.path.insert(0, %r)\nimport diag_lat_rtnum as R\n%s" % (HERE, code)] + list(argv),
                          capture_output=True, text=True, timeout=900)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--pinned-root", required=True)
    ap.add_argument("--real-manifest", required=True, help="the registered D-UNC manifest_f2.json (read only)")
    ap.add_argument("--run-card", required=True, help="the draft production run card of the diagnostic (read only)")
    ap.add_argument("--report", required=True)
    ap.add_argument("--skip-full", action="store_true", help="development only: the report is then INCOMPLETE")
    args = ap.parse_args()
    print(BANNER)
    if not (sys.flags.isolated and sys.dont_write_bytecode):
        print("ERROR: run with -I -B")
        return 2
    t_start = time.monotonic()
    P.pinned_preflight(args.pinned_root)
    s = P.need_pinned()
    torch = s.torch
    tmp = tempfile.mkdtemp(prefix="rtnumtest_")
    lim = card()["limits"]

    # ---- T-RULE
    for dev, want in ((3.433e-5, 1e-4), (4.9e-6, 1e-5), (5e-6, 1e-5), (5.1e-6, 2e-5), (6e-5, 2e-4), (1e-6, 2e-6),
                      (2.4e-5, 5e-5), (2.6e-5, 1e-4), (9.72e-6, 2e-5), (4.9e-4, 1e-3)):
        got = R.tolerance_by_rule(dev)
        check("T-RULE", "development maximum %g -> T %g" % (dev, want), got == want, "got %r" % got)
    for bad in (0.0, -1e-6, float("nan"), float("inf"), 1):
        raises("T-RULE", "development maximum %r: DIAG_LAT_INPUT" % (bad,), C.INPUT, lambda bad=bad: R.tolerance_by_rule(bad))

    # ---- T-STAT
    ref = np.array([[1.0, -2.0, 5.5], [0.5, 4.0, -3.0]])
    e = np.array([[1e-6, -2e-6, 4e-6], [0.0, 1e-6, -3e-6]])
    st = R.residual_stats(e, ref)
    check("T-STAT", "maximum, position, reference and signed error at the maximum",
          st["max_abs"] == 4e-6 and (st["row"], st["coordinate"]) == (0, 2) and st["reference_at_max"] == 5.5 and st["error_at_max"] == 4e-6)
    check("T-STAT", "RMS and the two relative errors", math.isclose(st["rms"], math.sqrt(np.mean(e * e)), rel_tol=1e-12)
          and math.isclose(st["max_abs_over_reference_max"], 4e-6 / 5.5, rel_tol=1e-12)
          and math.isclose(st["rms_over_reference_rms"], math.sqrt(np.mean(e * e)) / math.sqrt(np.mean(ref * ref)), rel_tol=1e-12))
    check("T-STAT", "distance to the spline bound (negative beyond it) and the error in float32 spacings",
          math.isclose(st["distance_to_spline_bound"], C.SPLINE_B - 5.5, rel_tol=1e-12) and st["distance_to_spline_bound"] < 0
          and math.isclose(st["float32_spacings_at_max"], 4e-6 / float(np.spacing(np.float32(5.5))), rel_tol=1e-12))
    raises("T-STAT", "zero reference: DIAG_LAT_ZERO_DENOMINATOR", C.ZERO_DENOMINATOR, lambda: R.residual_stats(e, np.zeros_like(ref)))
    bad = e.copy()
    bad[0, 0] = np.nan
    raises("T-STAT", "non-finite residual: DIAG_LAT_NONFINITE", C.NONFINITE, lambda: R.residual_stats(bad, ref))
    raises("T-STAT", "float32 residual: DIAG_LAT_INPUT", C.INPUT, lambda: R.residual_stats(e.astype(np.float32), ref))
    raises("T-STAT", "unequal shapes: DIAG_LAT_INPUT", C.INPUT, lambda: R.residual_stats(e[:1], ref))

    # ---- T-SLICE
    check("T-SLICE", "97 slices -> 0, 48, 96; 3 slices -> 0, 1, 2", R.pick_slices(97) == [0, 48, 96] and R.pick_slices(3) == [0, 1, 2])
    raises("T-SLICE", "two slices: DIAG_LAT_POPULATION_MISMATCH", R.POPULATION, lambda: R.pick_slices(2))

    # ---- T-CARD
    with open(args.run_card, "rb") as fh:
        raw = fh.read()
    prod, prod_sha = json.loads(raw), hashlib.sha256(raw).hexdigest()
    try:
        R.check_card(prod)
        ok = True
    except DiagLatError as exc:
        ok = False
        print(exc)
    check("T-CARD", "the draft production run card is accepted", ok and prod["selftest"] is False)
    check("T-CARD", "it names the registered manifest, pin, sizes and population; the real manifest has the registered sha256",
          prod["manifest_f2_sha256"] == BD.REGISTERED_F2_SHA256 and prod["pin"] == BD.PIN and prod["registered"] == R.REGISTERED
          and prod["population"] == R.REGISTERED_POPULATION and BD.sha_file(args.real_manifest) == BD.REGISTERED_F2_SHA256)
    check("T-CARD", "its limits are those of plan v0.3 (1e-9, 1e-4, 1e-5, 1e-5, 1e-4, ceiling 1e-4, present 1e-5); wall 3600 s",
          prod["limits"] == lim and prod["ceilings"]["wall_s"] == 3600 and prod["ceilings"]["reserve_bytes"] == 64 * GD.GIB
          and prod["ceilings"]["ram_bytes"] == 16 * GD.GIB and prod["ceilings"]["disk_bytes"] == 2 * GD.GIB)
    c0, n0 = R.load_card(args.run_card, prod_sha)
    check("T-CARD", "load_card with the right sha256", c0 == prod and n0 == prod_sha)
    raises("T-CARD", "load_card with a wrong sha256: DIAG_LAT_RUN_CARD", R.RUN_CARD, lambda: R.load_card(args.run_card, T.h("x")))
    refusals = [("another manifest", dict(prod, manifest_f2_sha256=T.h("m")), R.RUN_CARD), ("another pin", dict(prod, pin="0" * 40), R.RUN_CARD),
                ("other sizes", dict(prod, registered=dict(prod["registered"], n_states=27)), R.RUN_CARD),
                ("another population", dict(prod, population=dict(prod["population"], development_seeds=[2027], validation_seeds=[2026, 2028])), R.RUN_CARD),
                ("device cuda", dict(prod, device="cuda"), R.MISMATCH),
                ("four slices", dict(prod, population=dict(prod["population"], n_slices=4)), R.RUN_CARD),
                ("overlapping seed sets", card(population={"validation_seeds": [2026, 2027]}), R.RUN_CARD),
                ("a missing limit", dict(prod, limits={k: v for k, v in prod["limits"].items() if k != "tolerance_ceiling_max_abs"}), R.RUN_CARD),
                ("an extra key", dict(prod, tolerance=1e-4), R.RUN_CARD), ("an integer limit", dict(prod, limits=dict(prod["limits"], f64_roundtrip_max_abs=1)), R.RUN_CARD),
                ("a negative limit", dict(prod, limits=dict(prod["limits"], j_f32_vs_f64_nats_per_dim=-1e-5)), R.RUN_CARD),
                ("a missing ceiling", dict(prod, ceilings={k: v for k, v in prod["ceilings"].items() if k != "wall_s"}), R.RUN_CARD),
                ("a wrong schema", dict(prod, schema="seqref-dlat-f1-runcard/1"), R.RUN_CARD), ("not a dict", [1], R.RUN_CARD)]
    for name, cc, code in refusals:
        raises("T-CARD", "%s: %s" % (name, code), code, lambda cc=cc: R.check_card(cc))

    # ---- T-SUM (constructed statistics; no model)
    sm = R.summarise(fake_set(), lim)
    check("T-SUM", "all conditions met: T by the rule from the development maximum (bank 4e-6 -> 1e-5, target 3e-6 -> 1e-5), proposed",
          sm["proposal"]["bank_direction"]["proposed_t"] == 1e-5 and sm["proposal"]["target_direction"]["proposed_t"] == 1e-5
          and all(sm["proposal"]["bank_direction"]["conditions"].values()) and sm["proposal"]["note"] == R.PROPOSAL_NOTE
          and sm["n_slice_states"] == {"development": 3, "validation": 6})
    sm = R.summarise(fake_set(dev={"bank32": 3.4e-5}, val={"bank32": 6e-5}), lim)
    check("T-SUM", "development 3.4e-5 -> T 1e-4, at the ceiling, validation 6e-5 inside: proposed; nine slice-states above the present 1e-5",
          sm["proposal"]["bank_direction"]["proposed_t"] == 1e-4 and sm["above_present_tolerance"]["all"]["bank_direction"] == 9
          and sm["above_present_tolerance"]["all"]["target_direction"] == 0)
    a = R.summarise(fake_set(dev={"bank32": 3.4e-5}, val={"bank32": 6e-5}), lim)["proposal"]["bank_direction"]["t_by_rule"]
    b = R.summarise(fake_set(dev={"bank32": 3.4e-5}, val={"bank32": 9e-4}), lim)["proposal"]["bank_direction"]
    check("T-SUM", "T is formed from the development set only (a larger validation residual does not change it) and then fails on validation",
          a == b["t_by_rule"] == 1e-4 and b["proposed_t"] is None and b["conditions"]["validation_all_within_t"] is False
          and b["conditions"]["t_within_ceiling"] is True)
    neg = [("development 6e-5 -> T 2e-4 above the ceiling", dict(dev={"bank32": 6e-5}), "t_within_ceiling"),
           ("a float64 residual of 1e-8 in the validation set", dict(val={"f64": 1e-8}), "f64_roundtrip_within_limit_all"),
           ("J difference 2e-5 on the bank", dict(val={"j_bank_f32_vs_f64_max_abs": 2e-5}), "j_within_limit_all"),
           ("J difference 2e-5 on the target", dict(dev={"j_target_f32_vs_f64_abs": 2e-5}), "j_within_limit_all"),
           ("decoded-bank difference 2e-4", dict(val={"decoded_bank_f32_vs_f64": {"max_abs": 2e-4, "rms": 1e-5}}), "decoded_bank_within_limit_all"),
           ("float64 likelihood 2e-4 from the stored row", dict(val={"nll_f64_vs_stored_abs": 2e-4}), "f64_path_valid_likelihood"),
           ("float64 likelihood 2e-5 from float32", dict(dev={"nll_f64_vs_f32_abs": 2e-5}), "f64_path_valid_likelihood"),
           ("the direct float32 path differs from the registered one", dict(val={"direct_f32_vs_registered_max_abs": 1e-12}), "f64_path_differs_by_dtype_only")]
    for name, kw, cond in neg:
        pr = R.summarise(fake_set(**kw), lim)["proposal"]
        failed = [k for k, v in pr["bank_direction"]["conditions"].items() if not v]
        shared = cond != "t_within_ceiling"
        check("T-SUM", "%s: no tolerance proposed (%s)" % (name, cond),
              pr["bank_direction"]["proposed_t"] is None and failed == [cond]
              and (pr["target_direction"]["proposed_t"] is None) == shared, "failed %s" % failed)
    raises("T-SUM", "an empty set: DIAG_LAT_POPULATION_MISMATCH", R.POPULATION,
           lambda: R.summarise([r for r in fake_set() if r["set"] == "development"], lim))

    # ---- T-F64
    class NoDouble(torch.nn.Linear):
        def double(self):
            return self

    class Shift(torch.nn.Linear):
        def double(self):
            m = super().double()
            with torch.no_grad():
                m.weight.add_(1e-12)
            return m

    class Holder:
        def __init__(self, flow):
            self.flow = flow
    raises("T-F64", "a copy that stays float32: DIAG_LAT_REPRODUCTION_MISMATCH", R.MISMATCH, lambda: R.make_f64_flow(Holder(NoDouble(3, 3))))
    raises("T-F64", "a copy whose values differ by 1e-12: DIAG_LAT_REPRODUCTION_MISMATCH", R.MISMATCH, lambda: R.make_f64_flow(Holder(Shift(3, 3))))
    lin = torch.nn.Linear(3, 3)
    f = R.make_f64_flow(Holder(lin))
    check("T-F64", "a sound copy: float64, equal in value, the original still float32",
          f.weight.dtype == torch.float64 and lin.weight.dtype == torch.float32 and bool(torch.equal(lin.weight.double(), f.weight)))

    # ---- synthetic binding
    base = T.SynBinding(s)
    sb = small(base)
    before = state_hash(base)

    # ---- T-SMALL
    out = os.path.join(tmp, "small")
    rep, rc = R.run_rtnum(sb, card(), out, banner=BANNER)
    recs = rep["slice_states"]
    check("T-SMALL", "the run completes: exit 0, RTNUM: COMPLETE, no stop", rc == 0 and rep["verdict"] == "RTNUM: COMPLETE" and rep["stop"] is None)
    check("T-SMALL", "nine slice-states (three states x three slices), development 3, validation 6",
          len(recs) == 9 and rep["summary"]["n_slice_states"] == {"development": 3, "validation": 6}
          and [r["slice_position"] for r in recs] == [0, 1, 2] * 3 and rep["population"]["slice_positions"] == [0, 1, 2])
    check("T-SMALL", "the four files are written, each with the write guard's sha256 sidecar, and the sha256 manifest verifies",
          sorted(n for n in os.listdir(out) if not n.startswith(".") and not n.endswith(".sha256")) == sorted(FILES)
          and all(open(os.path.join(out, n + ".sha256")).read().split()[0] == hashlib.sha256(open(os.path.join(out, n), "rb").read()).hexdigest() for n in FILES)
          and all(hashlib.sha256(open(os.path.join(out, ln.split("  ")[1]), "rb").read()).hexdigest() == ln.split("  ")[0]
                  for ln in open(os.path.join(out, "outputs_sha256.txt")).read().splitlines()))
    disk = json.load(open(os.path.join(out, "rtnum_report.json")))
    check("T-SMALL", "the report on disk passes the schema and equals the returned report", R.check_report(disk) and disk == json.loads(F1.dumps(rep)))
    check("T-SMALL", "float64 round trips within 1e-9 in both directions on every slice-state",
          all(r[d]["f64"]["max_abs"] <= 1e-9 for r in recs for d in R.DIRECTIONS),
          max(r[d]["f64"]["max_abs"] for r in recs for d in R.DIRECTIONS))
    check("T-SMALL", "float32 residuals are orders above the float64 ones",
          all(r[d]["f32"]["max_abs"] > 1e3 * r[d]["f64"]["max_abs"] for r in recs for d in R.DIRECTIONS))
    check("T-SMALL", "the direct float32 flow call with the shared h equals the registered decode exactly (difference 0)",
          all(r["direct_f32_vs_registered_max_abs"] == 0.0 for r in recs))
    check("T-SMALL", "float32 likelihood equals the stored row; float64 within 1e-5 of float32",
          all(r["nll_f32_vs_stored_abs"] <= 1e-12 and r["nll_f64_vs_f32_abs"] <= 1e-5 for r in recs),
          max(r["nll_f64_vs_f32_abs"] for r in recs))
    m0, st0 = sb.models[sb.states[sb.state_order[0]]["setting"]], sb._slices[0]
    u = A.decode_bank_std(m0, st0, sb._bank)
    z, _j = A.reencode_bank(m0, st0, u)
    fx2 = float(np.max(np.abs(z.astype(np.float64) - sb._bank.astype(np.float64))))
    enc = A.encode_truth(m0, st0)
    back = A.decode_bank_std(m0, st0, np.ascontiguousarray(np.asarray(enc["z_star"], dtype=np.float32)[None, :]))[0]
    fx1 = float(np.max(np.abs(back.astype(np.float64) - A.target_std(st0)[1].astype(np.float64))))
    check("T-SMALL", "bank residual = fixture 2 (ii) and target residual = fixture 1, to the last digit",
          recs[0]["bank_direction"]["f32"]["max_abs"] == fx2 and recs[0]["target_direction"]["f32"]["max_abs"] == fx1,
          "%r %r / %r %r" % (recs[0]["bank_direction"]["f32"]["max_abs"], fx2, recs[0]["target_direction"]["f32"]["max_abs"], fx1))
    check("T-SMALL", "timing of every state, the elapsed time and the peak memory are recorded",
          len(rep["timing"]["states"]) == 3 and rep["timing"]["elapsed_s"] > 0 and rep["memory"]["peak_rss_bytes"] > 0
          and all(set(x) == {"index", "load_s", "f64_copy_s", "measure_s"} for x in rep["timing"]["states"]))
    check("T-SMALL", "the proposal carries the note 'PROPOSED AMENDMENT ONLY' and the conditions for each direction",
          rep["summary"]["proposal"]["note"].startswith("PROPOSED AMENDMENT ONLY")
          and all(set(rep["summary"]["proposal"][d]) == {"development_max_f32", "validation_max_f32", "t_by_rule", "conditions", "proposed_t"}
                  for d in R.DIRECTIONS))
    text = open(os.path.join(out, "rtnum_report.json")).read()
    check("T-SMALL", "no measure key anywhere in the report; the banner says residual statistics only",
          not any(('"%s"' % k) in text for k in F1.REFUSED_KEYS) and disk["banner"] == BANNER and "NO MEASURE" in R.BANNER)

    # ---- T-DET, T-INPUT
    rep2, rc2 = R.run_rtnum(sb, card(), os.path.join(tmp, "small2"), banner=BANNER)
    check("T-DET", "a second run gives identical statistics and summary", rc2 == 0 and strip_time(rep2) == strip_time(rep))
    check("T-INPUT", "model parameters, bank and slice tensors are unchanged by two runs", state_hash(base) == before)

    # ---- T-MIS
    orig_make = R.make_f64_flow

    class Skewed:
        def __init__(self, flow):
            self.flow = flow

        def decode(self, z, h):
            return self.flow.decode(z, h) * (1.0 + 1e-6)

        def encode(self, u, h):
            return self.flow.encode(u, h)
    R.make_f64_flow = lambda model: Skewed(orig_make(model))
    try:
        repm, rcm = R.run_rtnum(sb, card(), os.path.join(tmp, "mis"), banner=BANNER)
    finally:
        R.make_f64_flow = orig_make
    pm = repm["summary"]["proposal"]
    check("T-MIS", "an injected mismatch between decode and encode of the float64 flow: the run completes, the float64 residual is above 1e-9, no tolerance is proposed",
          rcm == 0 and repm["summary"]["maxima"]["all"]["bank_direction"]["f64"]["value"] > 1e-9
          and pm["bank_direction"]["proposed_t"] is None and pm["target_direction"]["proposed_t"] is None
          and pm["bank_direction"]["conditions"]["f64_roundtrip_within_limit_all"] is False,
          repm["summary"]["maxima"]["all"]["bank_direction"]["f64"]["value"])
    check("T-MIS", "make_f64_flow is restored", R.make_f64_flow is orig_make)

    # ---- T-STOP
    first = sb.state_order[0]
    stops("T-STOP", "selftest card with a production-style binding flag: DIAG_LAT_RUN_CARD", tmp, T.variant(sb, selftest=False), card(), R.RUN_CARD, 0)
    stops("T-STOP", "four bound states for a card of three: DIAG_LAT_POPULATION_MISMATCH", tmp, sb, card(registered={"n_states": 4}), R.POPULATION, 0)
    stops("T-STOP", "a card with 4 development slice-states: DIAG_LAT_POPULATION_MISMATCH", tmp, sb,
          card(population={"n_development_slice_states": 4}), R.POPULATION, 0)
    stops("T-STOP", "a state whose seed is in neither set: DIAG_LAT_POPULATION_MISMATCH", tmp, sb,
          card(population={"validation_seeds": [2027], "n_validation_slice_states": 3}), R.POPULATION, 0)
    bad_row = small(base)
    bad_row.stored[first]["NLL"][1] = np.nan
    stops("T-STOP", "a non-finite stored likelihood row: DIAG_LAT_POPULATION_MISMATCH", tmp, bad_row, card(), R.POPULATION, 0)
    stops("T-STOP", "slice states out of the registered order: DIAG_LAT_POPULATION_MISMATCH", tmp,
          T.variant(sb, identities=list(reversed(sb.identities))), card(), R.POPULATION, 0)
    stops("T-STOP", "a bank with another sha256: DIAG_LAT_ORIGIN", tmp, T.variant(sb, bank_sha256=T.h("other bank")), card(), A.ORIGIN, 0)
    stops("T-STOP", "a bank of 8 rows: DIAG_LAT_ORIGIN", tmp, T.variant(sb, _bank=np.ascontiguousarray(sb._bank[:8])), card(), A.ORIGIN, 0)

    class WrongId:
        def __init__(self, b):
            self._b = b

        def __getattr__(self, name):                 # everything but load_state is the reduced binding's
            return getattr(self._b, name)

        def load_state(self, sid):
            model, ver = self._b.load_state(sid)
            return model, dict(ver, model_state_sha256=T.h("another state"))
    stops("T-STOP", "the loader returns another model state: DIAG_LAT_ORIGIN", tmp, WrongId(sb), card(), A.ORIGIN, 0)

    class Double(WrongId):
        def load_state(self, sid):
            model, ver = self._b.load_state(sid)
            return copy.deepcopy(model).double(), ver
    stops("T-STOP", "a model with float64 parameters: DIAG_LAT_REPRODUCTION_MISMATCH", tmp, Double(sb), card(), R.MISMATCH, 0)

    class Second(WrongId):
        def load_state(self, sid):
            model, ver = self._b.load_state(sid)
            if sid == self.state_order[1]:
                ver = dict(ver, file_sha256=T.h("another file"))
            return model, ver
    stops("T-STOP", "the second state's file identity is wrong: DIAG_LAT_ORIGIN after three slice-states, which stay in the report",
          tmp, Second(sb), card(), A.ORIGIN, 3)
    stops("T-STOP", "wall ceiling of 1 s: DIAG_LAT_RESOURCE_LIMIT after the first state", tmp, sb, card(ceilings={"wall_s": 1}), GD.RESOURCE_LIMIT, 3)
    stops("T-STOP", "memory ceiling of 1 MiB: DIAG_LAT_RESOURCE_LIMIT after the first state", tmp, sb, card(ceilings={"ram_bytes": 2 ** 20}), GD.RESOURCE_LIMIT, 3)
    exists = os.path.join(tmp, "exists")
    os.makedirs(exists)
    raises("T-STOP", "an existing output folder: DIAG_LAT_GUARD_STATE, nothing written", GD.GUARD_STATE, lambda: R.run_rtnum(sb, card(), exists))
    check("T-STOP", "the existing folder is left empty", os.listdir(exists) == [])
    orig_measure = R.measure

    def nan_measure(*a, **k):
        r = orig_measure(*a, **k)
        r["bank_direction"]["f32"]["rms"] = float("nan")
        return r

    def key_measure(*a, **k):
        r = orig_measure(*a, **k)
        r["tail"] = 0.5
        return r
    for name, fn, code in (("a non-finite number", nan_measure, C.NONFINITE), ("a measure key ('tail')", key_measure, A.RECORD_SCHEMA)):
        R.measure = fn
        outs = os.path.join(tmp, "schema_%s" % code)
        try:
            reps, rcs = R.run_rtnum(sb, card(), outs, banner=BANNER)
        finally:
            R.measure = orig_measure
        written = json.load(open(os.path.join(outs, "rtnum_report.json")))
        check("T-SCHEMA", "%s in the report: RTNUM: FAIL (%s); the file written holds no slice-state statistics and no summary" % (name, code),
              rcs == 1 and reps["verdict"] == "RTNUM: FAIL" and reps["stop"]["code"] == code and written["slice_states"] == []
              and written["summary"] is None and open(os.path.join(outs, "rtnum_verdict.txt")).read() == "RTNUM: FAIL\n")
    check("T-SCHEMA", "measure is restored", R.measure is orig_measure)
    raises("T-SCHEMA", "check_report refuses a missing top-level key", A.RECORD_SCHEMA, lambda: R.check_report({k: None for k in R.REPORT_KEYS if k != "summary"}))
    check("T-INPUT", "model parameters, bank and slice tensors are unchanged after all runs", state_hash(base) == before)

    # ---- T-MAIN
    empty = os.path.join(tmp, "empty_repo")
    os.makedirs(empty)
    dunc = os.path.join(tmp, "dunc")
    os.makedirs(dunc)
    r = child_main(["--checkpoint", "x"])
    check("T-MAIN", "--checkpoint is refused before any import: exit 3", r.returncode == 3 and "DIAG_LAT_REFUSED" in r.stderr)
    r = child_main(["--data-root=/x"])
    check("T-MAIN", "--data-root= is refused: exit 3", r.returncode == 3 and "DIAG_LAT_REFUSED" in r.stderr)
    common = ["--run-card", args.run_card, "--repo", empty, "--manifest", args.real_manifest, "--dunc-dir", dunc]
    r = child_main(common + ["--run-card-sha256", T.h("x"), "--out", os.path.join(tmp, "m1")])
    check("T-MAIN", "a wrong run-card hash: exit 1, DIAG_LAT_RUN_CARD, nothing written",
          r.returncode == 1 and "RTNUM: FAIL (DIAG_LAT_RUN_CARD)" in r.stdout and not os.path.exists(os.path.join(tmp, "m1")))
    st_card = os.path.join(tmp, "selftest_card.json")
    with open(st_card, "wb") as fh:
        fh.write(F1.dumps(card()))
    r = child_main(["--run-card", st_card, "--run-card-sha256", BD.sha_file(st_card), "--repo", empty, "--manifest", args.real_manifest,
                    "--dunc-dir", dunc, "--out", os.path.join(tmp, "m2")])
    check("T-MAIN", "a self-test card at the production entry point: exit 1, DIAG_LAT_RUN_CARD, nothing written",
          r.returncode == 1 and "RTNUM: FAIL (DIAG_LAT_RUN_CARD)" in r.stdout and not os.path.exists(os.path.join(tmp, "m2")))
    import shutil
    low_disk = shutil.disk_usage(tmp).free < prod["ceilings"]["reserve_bytes"]
    if low_disk:
        r = child_main(common + ["--run-card-sha256", prod_sha, "--out", os.path.join(tmp, "m3")])
        check("T-MAIN", "sandbox free disk below the registered reserve: the unmodified entry point refuses (DIAG_LAT_RESOURCE_LIMIT)",
              r.returncode == 1 and "RTNUM: FAIL (DIAG_LAT_RESOURCE_LIMIT)" in r.stdout, r.stdout[-300:] + r.stderr[-300:])
    out4 = os.path.join(tmp, "m4")
    argv = common + ["--run-card-sha256", prod_sha, "--out", out4]
    if low_disk:      # the free-space reading of the guard is replaced IN THE TEST CHILD ONLY; everything else is the entry point
        r = snippet("orig = R.GD.WriteGuard\n"
                    "R.GD.WriteGuard = lambda root, ceiling, reserve, free_fn=None: orig(root, ceiling, reserve, free_fn=lambda: 2 ** 42)\n"
                    "sys.argv = ['diag_lat_rtnum.py'] + sys.argv[1:]\nsys.exit(R.main())", *argv)
    else:
        r = child_main(argv)
    ok = r.returncode == 1 and "RTNUM: FAIL" in r.stdout and os.path.isfile(os.path.join(out4, "rtnum_report.json"))
    stop4 = json.load(open(os.path.join(out4, "rtnum_report.json")))["stop"] if ok else {}
    check("T-MAIN", "production card, real manifest, a checkout without results: the manifest binds, the missing files stop typed, the evidence is written",
          ok and stop4.get("slice_states_done") == 0 and str(stop4.get("code", "")).startswith("DIAG_LAT_"), "%s %s" % (stop4, r.stdout[-300:]))

    # ---- T-FULL
    if args.skip_full:
        print("T-FULL skipped: THE REPORT IS INCOMPLETE")
    else:
        fc = card(registered={"n_states": 28, "bank_rows": C.K_BANK}, population={"n_development_slice_states": 30, "n_validation_slice_states": 54})
        repf, rcf = R.run_rtnum(base, fc, os.path.join(tmp, "full"), banner=BANNER)
        check("T-FULL", "28 states, 128 bank rows: 84 slice-states, development 30, validation 54, RTNUM: COMPLETE",
              rcf == 0 and len(repf["slice_states"]) == 84 and repf["summary"]["n_slice_states"] == {"development": 30, "validation": 54}
              and len({r["state"] for r in repf["slice_states"]}) == 28 and base.loads >= 28)
        check("T-FULL", "float64 within 1e-9 on all 84; the direct float32 path equals the registered one on all 84",
              all(r[d]["f64"]["max_abs"] <= 1e-9 for r in repf["slice_states"] for d in R.DIRECTIONS)
              and all(r["direct_f32_vs_registered_max_abs"] == 0.0 for r in repf["slice_states"]))
        check("T-INPUT", "model parameters, bank and slice tensors are unchanged after the full run", state_hash(base) == before)
        print("T-FULL synthetic figures (NOT host evidence): %s" % json.dumps(
            {d: {s2: repf["summary"]["maxima"][s2][d]["f32"]["value"] for s2 in R.SETS} for d in R.DIRECTIONS}))
        print("T-FULL elapsed %.0f s, peak %.0f MiB (sandbox, 2 threads)" % (repf["timing"]["elapsed_s"], repf["memory"]["peak_rss_bytes"] / 2 ** 20))

    n_fail = sum(1 for c in CHECKS if not c["ok"])
    complete = not args.skip_full
    verdict = "PASS" if (n_fail == 0 and complete) else ("INCOMPLETE" if n_fail == 0 else "FAIL")
    here = {n: BD.sha_file(os.path.join(HERE, n)) for n in ("diag_lat_rtnum.py", "diag_lat_rtnumtest.py")}
    report = {"banner": BANNER, "test": __version__, "diagnostic": R.__version__, "checks": CHECKS, "n_checks": len(CHECKS), "n_failed": n_fail,
              "complete": complete, "verdict": verdict, "code_sha256": here, "run_card_sha256": prod_sha,
              "torch": torch.__version__, "numpy": np.__version__, "seconds": time.monotonic() - t_start}
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    print("%d checks, %d failed -- RTNUM TEST: %s" % (len(CHECKS), n_fail, verdict))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
