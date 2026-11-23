# SEQREF-DLAT-ADPT v0.1.1 -- diag_lat_adptest.py: synthetic tests of diag_lat_adapter.py v0.2 (record schema SEQREF-DLAT-ADPSCH v0.1)
# LIFETIME: KEEP
"""Tests T-S1 to T-S6 of the adapter record schema (LudWiki 247 LE73, LE74).

SYNTHETIC TENSORS ONLY -- NOT D-LAT EVIDENCE. No checkpoint, no MRI data. The model is the pinned
V05LT class with fixture parameters; the builders are those of diag_lat_p3test.py v0.5 (imported,
not copied). The archived adapter v0.1 is loaded from --v01 under another module name and is used
only as the regression reference.

  T-S1  schema: every key present and typed; malformed records are refused
  T-S2  statistics: classes, tags and outcome identical from v0.1 keys and from v0.2 keys
  T-S3  regression: the 11 values of v0.1 are bit-identical in v0.2 (same model, state, bank)
  T-S4  "not computed": TC_G on a constant coordinate, PR still computed; where the schema allows it
  T-S5  identities between stored values
  T-S6  z_proj: required, shape, finite, built from the projection matrix; E_cov from stored values

Not a fallback, placeholder, mock or dummy return anywhere. The fixture projection matrix is drawn
from the fixture seed below, which is NOT the registered seed 20261122. Every failure is logged.
Disclosed limit: the a01 "not computed" case (||mu - R0|| = 0 exactly) is not reachable with a
real model; it is tested in diag_lat_selftest.py at core level and here only at schema level.

Changelog:
  v0.1.1 (2026-11-23) documentation only: runs with diag_lat_p3test.py v0.5 and adapter v0.2.1. No check changed.
  v0.1 (2026-11-23) first version.
"""
import argparse
import importlib.util
import json
import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import diag_lat_adapter as A  # noqa: E402
import diag_lat_core as C  # noqa: E402
import diag_lat_p3test as P  # noqa: E402
import diag_lat_stats as ST  # noqa: E402

__version__ = "0.1.1"
ABBR = "SEQREF-DLAT-ADPT"
LOG = logging.getLogger("diag_lat.adptest")
BANNER = "SANDBOX SYNTHETIC TEST -- NOT HOST EVIDENCE -- NOT D-LAT EVIDENCE (synthetic tensors)"
FIXTURE_SEED = 424242                    # fixtures only; not a registered seed
V01_SHA256 = "300c30587f8c55f8901a3c36bbcd84d594a9fe6624d67e359c57cec48a0760e2"
V01_KEYS = ("KS", "P_swap", "NLL", "R0_PSNR")
V01_BANK = ("log_gamma_ratio", "d01", "a01", "TC_G", "phys_lgr", "phys_d01", "phys_a01")
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
    except TypeError as e:
        check(tid, name, code == "TypeError", "raised TypeError: %s" % e)
    else:
        check(tid, name, False, "no error raised")


def load_v01(path):
    got = P.sha_file(path)
    if got != V01_SHA256:
        LOG.error("adapter v0.1 at %s has sha256 %s, registered %s", path, got, V01_SHA256)
        raise SystemExit("adapter v0.1 hash mismatch")
    spec = importlib.util.spec_from_file_location("diag_lat_adapter_v01", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def l5_inputs(s, model, st, donor, noise):
    """The L5 values as the F1/G driver will compute them, from existing adapter and core calls."""
    own = A.encode_truth(model, st)
    sw = A.swap_input(st, donor)
    p_swap = C.swap_penalty(A.encode_truth(model, st, sw["cond_swap"], sw["mask"])["NLL"], own["NLL"])
    r0 = A.decode_r0_std(model, st).astype(np.float64)
    r0_sw = A.decode_r0_std(model, st, sw["cond_swap"], sw["mask"]).astype(np.float64)
    scale, mean = A.scale_mean_interleaved(st)
    shift = C.shift_norm(r0_sw, r0)
    shift_p = C.shift_norm(C.to_physical(r0_sw, scale, mean), C.to_physical(r0, scale, mean))
    y_acq = A.acquired_values(st)
    cond_p = A.perturbed_input(st, C.scale_noise(noise, y_acq))
    enc_p = A.encode_truth(model, st, cond_p, st["mask"])
    sens = C.l5b_sensitivities(r0, A.decode_r0_std(model, st, cond_p, st["mask"]).astype(np.float64),
                               own["z_star"].astype(np.float64), enc_p["z_star"].astype(np.float64),
                               own["NLL"], enc_p["NLL"])
    x0, _u0 = s.EV.r0_image(model, st)
    return {"p_swap": p_swap, "r0_psnr": float(s.tg._psnr(x0.abs(), st["x_true_mag"])),
            "swap_shift_r0": shift, "phys_swap_shift_r0": shift_p, "sensitivities": sens}, own


def model_tests(v01, pinned_root):
    P.pinned_preflight(pinned_root)
    s = P.need_pinned()
    g = np.random.Generator(np.random.PCG64(FIXTURE_SEED))
    states = P.synthetic_states(s, P.N_SLICES, offset=0)
    donors = s.d3.derangement(P.N_SLICES)
    model = s.ffr.build_model(); model.eval()
    P.set_parameters(s, model, 11, 0.01)
    bank = g.standard_normal((8, C.D_FREE)).astype(np.float32)
    proj = C.projection_matrix(g.standard_normal((C.D_FREE, C.M_PROJ)))
    check("T-S6", "fixture projection is orthonormal (max abs of P'P - I <= 1e-12)",
          float(np.max(np.abs(proj.T @ proj - np.eye(C.M_PROJ)))) <= 1e-12)
    subsets = P.TOY_SUBSETS
    recs, zs = [], []
    for i, st in enumerate(states):
        noise = np.random.Generator(np.random.PCG64(FIXTURE_SEED + 1 + i)).standard_normal((A.acquired_values(st).size, 2))
        kw, own = l5_inputs(s, model, st, states[donors[i]], noise)
        rec, diag = A.slice_state_record(model, st, bank, subsets=subsets, proj=proj, **kw)
        old, old_diag = v01.slice_state_record(model, st, bank, subsets=subsets, p_swap=kw["p_swap"], r0_psnr=kw["r0_psnr"])
        recs.append(rec); zs.append(own["z_star"].astype(np.float64))
        tag = "slice %d: " % i
        # T-S1
        names_flat = {k["name"] for k in A.RECORD_KEYS if not k["bank"]}
        names_bank = {k["name"] for k in A.RECORD_KEYS if k["bank"]}
        check("T-S1", tag + "record passes validate_record; 19 flat names, 17 bank names on every subset",
              A.validate_record(rec, subsets.keys()) and set(rec) == names_flat | {A.BANK_FIELD} and len(names_flat) == 19
              and len(names_bank) == 17 and all(set(rec[A.BANK_FIELD][lab]) == names_bank for lab in subsets))
        scal = [rec[n] for n in names_flat if n != "z_proj"] + [v for lab in subsets for v in rec[A.BANK_FIELD][lab].values()]
        check("T-S1", tag + "86 scalar values, every one a finite float (no not-computed value in this case)",
              len(scal) == 18 + 17 * len(subsets) and all(isinstance(v, float) and math.isfinite(v) for v in scal),
              "%d scalars with %d toy subsets" % (len(scal), len(subsets)))
        # T-S3
        same = all(rec[k] == old[k] for k in V01_KEYS) and all(
            rec[A.BANK_FIELD][lab][k] == old[v01.BANK_FIELD][lab][k] for lab in subsets for k in V01_BANK)
        check("T-S3", tag + "the 11 keys of v0.1 are bit-identical in v0.2", same)
        check("T-S3", tag + "tau and q_tail of the record equal the v0.1 diagnostics; other diagnostics unchanged",
              rec["tau"] == old_diag["tau"] and rec["q_tail"] == old_diag["q_tail"]
              and all(diag[k] == old_diag[k] for k in diag if k != "events") and "tau" not in diag and "q_tail" not in diag)
        # T-S5
        b = rec[A.BANK_FIELD]
        check("T-S5", tag + "log_gamma_ratio = log_gamma_truth - log_gamma_bank, standardised and physical (exact)",
              all(b[lab]["log_gamma_ratio"] == rec["log_gamma_truth"] - b[lab]["log_gamma_bank"]
                  and b[lab]["phys_lgr"] == rec["phys_log_gamma_truth"] - b[lab]["phys_log_gamma_bank"] for lab in subsets))
        check("T-S5", tag + "e0_sq_minus_e1_sq = e0_sq - e1_sq, standardised and physical (exact)",
              all(b[lab]["e0_sq_minus_e1_sq"] == rec["e0_sq"] - b[lab]["e1_sq"]
                  and b[lab]["phys_e0_sq_minus_e1_sq"] == rec["phys_e0_sq"] - b[lab]["phys_e1_sq"] for lab in subsets))
        check("T-S5", tag + "B = q_z / 2 + ln(2 pi) / 2 (1e-12)", abs(diag["B"] - (0.5 * rec["q_z"] + C.HALF_LOG_2PI)) <= 1e-12)
        check("T-S5", tag + "PR and phys_PR lie in (0, K - 1] on every subset",
              all(0.0 < b[lab]["PR"] <= len(subsets[lab]) - 1 + 1e-9 and 0.0 < b[lab]["phys_PR"] <= len(subsets[lab]) - 1 + 1e-9
                  for lab in subsets))
        # T-S6
        ref = np.array([float(np.dot(zs[-1], proj[:, j])) for j in range(C.M_PROJ)])
        check("T-S6", tag + "z_proj is a list of exactly 8 finite floats equal to z* @ proj (1e-10)",
              isinstance(rec["z_proj"], list) and len(rec["z_proj"]) == C.M_PROJ
              and all(isinstance(v, float) and math.isfinite(v) for v in rec["z_proj"])
              and float(np.max(np.abs(np.array(rec["z_proj"]) - ref))) <= 1e-10)
    st, kw0 = states[0], l5_inputs(s, model, states[0], states[donors[0]], np.ones((A.acquired_values(states[0]).size, 2)))[0]
    e_ref = C.projected_cov_mismatch(np.stack(zs), proj)
    e_rec = C.projected_cov_mismatch(np.array([r["z_proj"] for r in recs]), np.eye(C.M_PROJ))
    check("T-S6", "E_cov from the stored z_proj equals E_cov from the truth codes (1e-10)", abs(e_ref - e_rec) <= 1e-10,
          "%.12g vs %.12g" % (e_ref, e_rec))
    call = lambda **over: A.slice_state_record(model, st, bank, **{**dict(subsets=subsets, proj=proj, **kw0), **over})
    raises("T-S6", "proj with a wrong shape is refused", C.INPUT, lambda: call(proj=proj[:, :7]))
    raises("T-S6", "proj transposed is refused", C.INPUT, lambda: call(proj=proj.T))
    raises("T-S6", "a non-finite proj is refused", C.NONFINITE, lambda: call(proj=np.where(np.arange(proj.size).reshape(proj.shape) == 0, np.nan, proj)))
    raises("T-S6", "proj is a required argument (no default)", "TypeError",
           lambda: A.slice_state_record(model, st, bank, subsets=subsets, **kw0))
    raises("T-S1", "sensitivities with a missing key are refused", C.INPUT,
           lambda: call(sensitivities={k: 0.0 for k in A.SENSITIVITY_KEYS[:2]}))
    raises("T-S1", "a non-float pass-through value is refused by the schema", A.RECORD_SCHEMA, lambda: call(swap_shift_r0=None))
    # T-S4: a coordinate beyond the spline bound is copied unchanged, so a constant column has sigma = 0
    bank_c = bank.copy(); bank_c[:, 0] = 6.0
    rec_c, diag_c = A.slice_state_record(model, st, bank_c, subsets=subsets, proj=proj, **kw0)
    bc = rec_c[A.BANK_FIELD]
    check("T-S4", "constant coordinate: TC_G is NOT_COMPUTED on every subset, PR and phys_PR are still floats, event recorded",
          all(bc[lab]["TC_G"] is C.NOT_COMPUTED and isinstance(bc[lab]["PR"], float) and isinstance(bc[lab]["phys_PR"], float)
              for lab in subsets) and any(e["code"] == C.UNDEFINED_STATISTIC and e["measure"] == "TC_G" for e in diag_c["events"]))
    check("T-S4", "constant coordinate: no other value is NOT_COMPUTED",
          all(v is not C.NOT_COMPUTED for lab in subsets for k, v in bc[lab].items() if k != "TC_G"))
    return recs[0], subsets


def schema_tests(good, subsets):
    cp = lambda: json.loads(json.dumps(good))
    labels = list(subsets)

    def edited(fn):
        r = cp(); fn(r)
        return lambda: A.validate_record(r, labels)
    lab0 = labels[0]
    for name, fn in (
            ("a flat key missing (q_z)", lambda r: r.pop("q_z")),
            ("z_proj missing", lambda r: r.pop("z_proj")),
            ("an extra flat key", lambda r: r.update(extra=1.0)),
            ("a bank key missing (PR)", lambda r: r[A.BANK_FIELD][lab0].pop("PR")),
            ("an extra bank key", lambda r: r[A.BANK_FIELD][lab0].update(extra=1.0)),
            ("z_proj with 7 values", lambda r: r.update(z_proj=r["z_proj"][:7])),
            ("z_proj with 9 values", lambda r: r.update(z_proj=r["z_proj"] + [0.0])),
            ("z_proj with a non-finite value", lambda r: r.update(z_proj=[float("nan")] + r["z_proj"][1:])),
            ("z_proj with an integer", lambda r: r.update(z_proj=[1] + r["z_proj"][1:])),
            ("z_proj as a scalar", lambda r: r.update(z_proj=0.0)),
            ("NOT_COMPUTED in PR", lambda r: r[A.BANK_FIELD][lab0].update(PR=C.NOT_COMPUTED)),
            ("NOT_COMPUTED in e1_sq", lambda r: r[A.BANK_FIELD][lab0].update(e1_sq=C.NOT_COMPUTED)),
            ("NOT_COMPUTED in tau", lambda r: r.update(tau=C.NOT_COMPUTED)),
            ("a non-finite q_tail", lambda r: r.update(q_tail=float("inf")))):
        raises("T-S1", "record refused -- " + name, A.RECORD_SCHEMA, edited(fn))
    ok = cp()
    for k in ("a01", "phys_a01", "TC_G"):
        ok[A.BANK_FIELD][lab0][k] = C.NOT_COMPUTED
    check("T-S4", "NOT_COMPUTED is accepted in a01, phys_a01 and TC_G, and (above) nowhere else", A.validate_record(ok, labels))
    nc = [k["name"] for k in A.RECORD_KEYS if "NOT_COMPUTED" in k["type"]]
    check("T-S4", "exactly three schema entries allow NOT_COMPUTED", sorted(nc) == ["TC_G", "a01", "phys_a01"], nc)


def stats_tests():
    """T-S2: the statistics read the same values whether or not the new keys are present."""
    labels = list(ST.STABILITY_SUBSETS)
    new_flat = [k["name"] for k in A.RECORD_KEYS if not k["bank"] and k["name"] not in V01_KEYS and k["name"] != "z_proj"]
    new_bank = [k["name"] for k in A.RECORD_KEYS if k["bank"] and k["name"] not in V01_BANK]
    recs = {(role, seed): [P.synthetic_record(role, si, v, labels) for v in range(P.N_VOL)]
            for role in P.ROLES for si, seed in enumerate(P.SEEDS)}

    def population(v2):
        pop = ST.Population(["v%03d" % i for i in range(P.N_VOL)])
        for (role, seed), rows in sorted(recs.items()):
            for name in V01_KEYS + (tuple(new_flat) if v2 else ()):
                pop.set(role, seed, name, [r[name] for r in rows])
            for name in V01_BANK + (tuple(new_bank) if v2 else ()):
                for lab in labels:
                    pop.set(role, seed, name, [r[A.BANK_FIELD][lab][name] for r in rows], lab)
        return pop.seal()
    idx = np.random.Generator(np.random.PCG64(P.SELFTEST_SEED)).integers(0, P.N_VOL, size=(400, P.N_VOL), dtype=np.int64)
    tau = np.abs(np.sin(np.arange(P.N_VOL) * 0.9)) * 0.05
    out = []
    for v2 in (False, True):
        pop = population(v2)
        res = ST.run_registered(pop, idx, A.PHYSICAL_NAMES)
        l6 = {str(sd): ST.l6_contrast(pop, "S-GAP0", sd, idx, tau) for sd in P.SEEDS}
        out.append(json.dumps({"res": res, "l6": l6}, sort_keys=True, default=str))
    check("T-S2", "synthetic records carry the v0.2 names (%d new flat, %d new bank)" % (len(new_flat), len(new_bank)),
          len(new_flat) == 14 and len(new_bank) == 10)
    check("T-S2", "run_registered (classes, tags, outcome, every contrast) and L6 are identical with and without the new keys",
          out[0] == out[1], "compared as sorted JSON, %d bytes" % len(out[0]))
    check("T-S2", "PHYSICAL_NAMES is unchanged from v0.1", A.PHYSICAL_NAMES == {"L2": "phys_lgr", "L3a": "phys_d01", "L3b": "phys_a01"})


def main():
    ap = argparse.ArgumentParser(description="adapter v0.2 schema tests (synthetic; NOT D-LAT EVIDENCE)")
    ap.add_argument("--pinned-root", required=True, help="repository root that holds seqref_mri/ (pinned sources, read only)")
    ap.add_argument("--v01", required=True, help="path of the archived diag_lat_adapter.py v0.1")
    ap.add_argument("--out", required=True, help="report file; must not exist yet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.CRITICAL)      # expected typed stops are exercised on purpose
    if os.path.exists(args.out):
        LOG.error("report exists: %s", args.out)
        raise SystemExit("report already exists: %s" % args.out)
    v01 = load_v01(args.v01)
    good, subsets = model_tests(v01, args.pinned_root)
    schema_tests(good, subsets)
    stats_tests()
    here = os.path.dirname(os.path.abspath(__file__))
    n_fail = sum(1 for c in CHECKS if not c["pass"])
    rep = {"banner": BANNER, "schema": "seqref-dlat-adptest/1", "adptest_version": __version__, "adapter_version": A.__version__,
           "adapter_v01_sha256": V01_SHA256, "core_version": C.__version__, "stats_version": ST.__version__,
           "fixture_seed": FIXTURE_SEED, "pinned_preflight": P.CTX["pinned"]["status"],
           "files_sha256": {n: P.sha_file(os.path.join(here, n)) for n in sorted(os.listdir(here)) if n.endswith(".py")},
           "checks": CHECKS, "counts": {"pass": len(CHECKS) - n_fail, "fail": n_fail},
           "result": "PASS" if n_fail == 0 else "FAIL"}
    with open(args.out, "x") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(BANNER)
    for t in ("T-S1", "T-S2", "T-S3", "T-S4", "T-S5", "T-S6"):
        rows = [c for c in CHECKS if c["test"] == t]
        print("%s  %d/%d" % (t, sum(c["pass"] for c in rows), len(rows)))
        for c in rows:
            if not c["pass"]:
                print("   FAIL %s %s" % (c["check"], c["note"]))
    print("RESULT %s (%d checks, %d failed)" % (rep["result"], len(CHECKS), n_fail))
    raise SystemExit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
