# SEQREF-DLAT-RTNUM-D v0.1 -- diag_lat_rtnum.py: round-trip numerical diagnostic, float32 against float64 (plan SEQREF-DLAT-RTNUM v0.3, core only)
# LIFETIME: DIAGNOSTIC
"""Follows the failed F1 run V13 (LudWiki 251 LE88): fixture 2 (ii), bank re-encode residual
3.433e-5 against 1e-5. Question: float32 precision, numerical sensitivity of the trained maps,
or a forward and inverse mismatch (LudWiki 252 LE89-LE91).

THIS DIAGNOSTIC WRITES RESIDUAL STATISTICS ONLY: no D-LAT measure, contrast, class or figure.
It changes no tolerance and replaces no F1 check. A tolerance it names is a PROPOSED AMENDMENT
ONLY: it needs its own scientific justification and review, and it is no permission to rerun F1.

Population (fixed in the plan): every bound state x the first, middle and last slice x all bank
rows. Development set: the states of the run card's development seeds. Validation set: the
states of its validation seeds; no rule is formed from them.

Two precisions on identical inputs (review amendment A1):
  float32  the REGISTERED path: adapter calls encode_truth, decode_bank_std, reencode_bank
           (the calls of F1 fixtures 1 and 2 (ii)).
  float64  a float64 copy of the pinned flow only (copy.deepcopy(model.flow).double()); the
           conditioning vector h comes from the registered float32 conditioner and is converted;
           bank rows, target and h are the float32 values converted exactly. The whole pinned
           model cannot be converted (its mask branch casts to float32). What this path cannot
           show: a precision effect inside the conditioner.
Directions: bank e_z = f(g(z_k)) - z_k; target e_u = g(f(u)) - u.

The file holds no science of D-LAT. Arithmetic of this file (complete list): differences,
maxima, root mean squares and ratios of residuals; B + J for the float64 likelihood of the
target (core formulas); the tolerance series 1, 2, 5 x 10^k; comparisons with run-card limits.

Measurement values never stop the run: they are recorded and compared in the summary. Stops
(typed DiagLatError, logged, recorded in the report, verdict RTNUM: FAIL) are for binding,
identity, device, dtype, non-finite values, resource ceilings and the report schema only.
No fallback, placeholder, mock or dummy return.

DISCLOSED for review (not in the plan's wording):
  (1) one report file holds the statistics of all slice-states (the plan says "one JSON of
      these statistics per slice-state"): one write, one hash;
  (2) relative error is defined as max|e| / max|reference| and rms(e) / rms(reference);
      a coordinate-wise ratio is undefined where the reference is zero;
  (3) an extra check, direct_f32_vs_registered: the pinned float32 flow called directly with the
      shared h must give the registered decode exactly (difference 0); it shows that the
      float64 path differs from the registered one by the dtype only;
  (4) the float64 copy is compared with the float32 parameters value by value (exact);
  (5) the likelihood figures are written as differences only (float32 and float64 against the
      stored row, and against each other); no likelihood value is written;
  (6) latent_max_abs (largest |z| of the reference latent) is recorded beside each residual.
  (7) if the report fails its schema (a measure key, a non-finite number), the statistics are
      NOT written: the report file then holds the stop, the run card and the verdict
      RTNUM: FAIL only;
  (8) the wall and the memory ceiling are checked after every model state and once more
      directly before the verdict; a stop keeps the statistics collected so far in the report,
      without a summary.

Changelog:
  v0.1 (2026-11-23) first version.
"""
import sys

FORBIDDEN = ("--checkpoint", "--data-root")
EXIT_REFUSED = 3


def refuse_forbidden(argv):
    """Standard library only; runs before NumPy, PyTorch or any project module is imported."""
    hit = sorted({f for a in argv for f in FORBIDDEN if a == f or a.startswith(f + "=")})
    if hit:
        sys.stderr.write("ERROR DIAG_LAT_REFUSED: option(s) %s are not accepted: checkpoints and data come only "
                         "through the registered manifest\n" % ", ".join(hit))
        raise SystemExit(EXIT_REFUSED)


if __name__ == "__main__":
    refuse_forbidden(sys.argv[1:])

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import diag_lat_adapter as A  # noqa: E402
import diag_lat_bind as BD  # noqa: E402
import diag_lat_core as C  # noqa: E402
import diag_lat_f1 as F1  # noqa: E402   (ProductionBinding, MemoryLog, dumps, sha_bytes, REFUSED_KEYS: the registered F1 code)
import diag_lat_guard as GD  # noqa: E402
import diag_lat_stats as ST  # noqa: E402
from diag_lat_core import DiagLatError  # noqa: E402

__version__ = "0.1"
ABBR = "SEQREF-DLAT-RTNUM-D"
LOG = logging.getLogger("diag_lat.rtnum")

CARD_SCHEMA = "seqref-dlat-rtnum-runcard/1"
REPORT_SCHEMA = "seqref-dlat-rtnum-report/1"
RUN_CARD = F1.RUN_CARD
MISMATCH = F1.REPRODUCTION_MISMATCH
POPULATION = ST.POPULATION_MISMATCH
BANNER = "D-LAT ROUND-TRIP NUMERICAL DIAGNOSTIC -- RESIDUAL STATISTICS ONLY -- NO MEASURE -- NO TOLERANCE IS CHANGED"
PROPOSAL_NOTE = ("PROPOSED AMENDMENT ONLY. Not adopted. A value named here needs its own scientific justification and "
                 "review before it enters an F1 run card, and it is no permission to rerun F1 (LudWiki 252 LE91).")
CARD_KEYS = {"schema", "selftest", "plan", "device", "manifest_f2_sha256", "pin", "registered", "population", "limits", "ceilings"}
REG_KEYS = {"n_states", "n_volumes", "bank_rows"}
POP_KEYS = {"development_seeds", "validation_seeds", "n_slices", "n_development_slice_states", "n_validation_slice_states"}
LIMIT_KEYS = {"present_tolerance_max_abs", "f64_roundtrip_max_abs", "nll_f64_vs_stored_nats_per_dim",
              "nll_f64_vs_f32_nats_per_dim", "j_f32_vs_f64_nats_per_dim", "decoded_bank_f32_vs_f64_max_abs",
              "tolerance_ceiling_max_abs"}
CEIL_KEYS = {"wall_s", "disk_bytes", "reserve_bytes", "ram_bytes"}
REGISTERED = {"n_states": 28, "n_volumes": 97, "bank_rows": C.K_BANK}
REGISTERED_POPULATION = {"development_seeds": [2026], "validation_seeds": [2027, 2028], "n_slices": 3,
                         "n_development_slice_states": 30, "n_validation_slice_states": 54}
REPORT_KEYS = {"schema", "banner", "versions", "run_card", "environment", "binding", "population", "slice_states",
               "summary", "timing", "memory", "stop", "verdict"}
DIRECTIONS = ("bank_direction", "target_direction")
PRECISIONS = ("f32", "f64")
SETS = ("development", "validation")
SERIES = (1, 2, 5, 10)


def _stop(code, message):
    raise DiagLatError(code, message)            # DiagLatError logs at construction


# ---- run card -------------------------------------------------------------------------------
def load_card(path, sha256):
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        _stop(RUN_CARD, "cannot read the run card %s: %s" % (path, exc))
    got = F1.sha_bytes(raw)
    if got != sha256:
        _stop(RUN_CARD, "run card %s has sha256 %s, expected %s" % (path, got, sha256))
    try:
        card = json.loads(raw)
    except ValueError as exc:
        _stop(RUN_CARD, "run card is not JSON: %s" % exc)
    check_card(card)
    return card, got


def _ints(v):
    return isinstance(v, list) and len(v) > 0 and all(isinstance(x, int) and not isinstance(x, bool) for x in v) and len(set(v)) == len(v)


def check_card(card):
    """Exact key sets; no default. A production card must name the registered manifest, pin, sizes and population."""
    if not isinstance(card, dict) or set(card) != CARD_KEYS:
        got = set(card) if isinstance(card, dict) else set()
        _stop(RUN_CARD, "run card keys: missing %s, unexpected %s" % (sorted(CARD_KEYS - got), sorted(got - CARD_KEYS)))
    reg, pop, lim, ceil = card["registered"], card["population"], card["limits"], card["ceilings"]
    ok = (card["schema"] == CARD_SCHEMA and isinstance(card["selftest"], bool) and isinstance(card["plan"], str) and card["plan"].strip()
          and isinstance(card["device"], str) and isinstance(card["manifest_f2_sha256"], str) and isinstance(card["pin"], str)
          and isinstance(reg, dict) and set(reg) == REG_KEYS and isinstance(pop, dict) and set(pop) == POP_KEYS
          and isinstance(lim, dict) and set(lim) == LIMIT_KEYS and isinstance(ceil, dict) and set(ceil) == CEIL_KEYS
          and all(isinstance(v, float) and math.isfinite(v) and v > 0.0 for v in lim.values())
          and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in list(ceil.values()) + list(reg.values()))
          and _ints(pop["development_seeds"]) and _ints(pop["validation_seeds"])
          and not set(pop["development_seeds"]) & set(pop["validation_seeds"])
          and all(isinstance(pop[k], int) and not isinstance(pop[k], bool) and pop[k] > 0
                  for k in ("n_slices", "n_development_slice_states", "n_validation_slice_states")))
    if not ok:
        _stop(RUN_CARD, "run card: schema, a size, the population, a limit or a ceiling is missing or malformed")
    if card["device"] != "cpu":
        _stop(MISMATCH, "run card device %r: this code runs on cpu only; there is no transfer code and no fallback" % card["device"])
    if pop["n_slices"] != 3:
        _stop(RUN_CARD, "run card n_slices %r: the plan fixes three slices (first, middle, last)" % pop["n_slices"])
    if not card["selftest"]:
        if reg != REGISTERED or pop != REGISTERED_POPULATION:
            _stop(RUN_CARD, "run card sizes %r and population %r differ from the registered %r and %r"
                  % (reg, pop, REGISTERED, REGISTERED_POPULATION))
        if card["manifest_f2_sha256"] != BD.REGISTERED_F2_SHA256 or card["pin"] != BD.PIN:
            _stop(RUN_CARD, "run card names manifest %s and pin %s; the registered ones are %s and %s"
                  % (card["manifest_f2_sha256"], card["pin"], BD.REGISTERED_F2_SHA256, BD.PIN))


# ---- arithmetic -----------------------------------------------------------------------------
def pick_slices(n):
    """First, middle and last slice of the n bound slices (the probe slices of F1)."""
    idx = sorted({0, n // 2, n - 1})
    if len(idx) != 3:
        _stop(POPULATION, "%d bound slices do not give three distinct slices (first, middle, last)" % n)
    return idx


def residual_stats(e, ref):
    """Statistics of a residual e against its reference (both (rows, d) float64, same shape)."""
    e, ref = np.asarray(e), np.asarray(ref)
    if e.dtype != np.float64 or ref.dtype != np.float64 or e.shape != ref.shape or e.ndim != 2 or e.size == 0:
        _stop(C.INPUT, "residual %s %s and reference %s %s must be equal-shaped (rows, d) float64"
              % (e.dtype, e.shape, ref.dtype, ref.shape))
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(ref))):
        _stop(C.NONFINITE, "a residual or its reference contains a non-finite value")
    a = np.abs(e)
    row, col = (int(v) for v in np.unravel_index(int(np.argmax(a)), a.shape))
    ref_max, ref_rms = float(np.max(np.abs(ref))), float(np.sqrt(np.mean(ref * ref)))
    if ref_max == 0.0 or ref_rms == 0.0:
        _stop(C.ZERO_DENOMINATOR, "the reference of a residual is identically zero")
    at = float(ref[row, col])
    max_abs = float(a[row, col])
    rms = float(np.sqrt(np.mean(e * e)))
    return {"max_abs": max_abs, "rms": rms, "max_abs_over_reference_max": max_abs / ref_max,
            "rms_over_reference_rms": rms / ref_rms, "row": row, "coordinate": col, "reference_at_max": at,
            "error_at_max": float(e[row, col]), "distance_to_spline_bound": C.SPLINE_B - abs(at),
            "float32_spacings_at_max": max_abs / float(np.spacing(np.float32(abs(at)))),
            "latent_max_abs": None}


def tolerance_by_rule(development_max):
    """The smallest value of the series 1, 2, 5 x 10^k that is at least twice the development maximum."""
    if not (isinstance(development_max, float) and math.isfinite(development_max) and development_max > 0.0):
        _stop(C.INPUT, "the development maximum %r is not a positive finite number" % (development_max,))
    need = 2.0 * development_max
    k = int(math.floor(math.log10(need)))
    for kk in (k - 1, k, k + 1):
        for m in SERIES:
            cand = float("%de%d" % (m, kk))
            if cand >= need:
                return cand
    _stop(C.INPUT, "no value of the series covers %r" % need)


# ---- the float64 flow -----------------------------------------------------------------------
def make_f64_flow(model):
    """A float64 copy of the pinned flow. The model itself is not changed (checked)."""
    torch = A.seams().torch
    try:
        f64 = copy.deepcopy(model.flow).double()
        f64.eval()
    except Exception as exc:
        _stop(A.ADAPTER, "float64 copy of the flow: %s: %s" % (type(exc).__name__, exc))
    p32, p64 = dict(model.flow.named_parameters()), dict(f64.named_parameters())
    b32, b64 = dict(model.flow.named_buffers()), dict(f64.named_buffers())
    if set(p32) != set(p64) or set(b32) != set(b64) or not p64:
        _stop(MISMATCH, "the float64 copy of the flow does not hold the parameters and buffers of the original")
    with torch.no_grad():
        for name in sorted(p32):
            if p32[name].dtype != torch.float32 or p64[name].dtype != torch.float64:
                _stop(MISMATCH, "flow parameter %s: original %s, copy %s; expected float32 and float64"
                      % (name, p32[name].dtype, p64[name].dtype))
            if not bool(torch.equal(p32[name].to(torch.float64), p64[name])):
                _stop(MISMATCH, "flow parameter %s: the float64 copy differs in value from the original" % name)
        for name in sorted(b32):
            if b32[name].is_floating_point() and (b64[name].dtype != torch.float64
                                                  or not bool(torch.equal(b32[name].to(torch.float64), b64[name]))):
                _stop(MISMATCH, "flow buffer %s: the float64 copy is not the exact float64 image of the original" % name)
    return f64


def _np64(name, t, shape, dtype):
    if t.dtype != dtype or tuple(t.shape) != tuple(shape):
        _stop(MISMATCH, "%s: got %s %s, expected %s %s" % (name, t.dtype, tuple(t.shape), dtype, tuple(shape)))
    out = np.ascontiguousarray(t.detach().cpu().numpy()).astype(np.float64)
    if not np.all(np.isfinite(out)):
        _stop(C.NONFINITE, "%s contains a non-finite value" % name)
    return out


def measure(model, f64, st, bank, stored_nll):
    """All statistics of one slice-state. Nothing here stops on the size of a residual."""
    torch = A.seams().torch
    k, d = bank.shape
    # -- float32: the registered path (adapter calls of F1 fixtures 1 and 2 (ii))
    enc = A.encode_truth(model, st)
    _t64, t32 = A.target_std(st)
    zstar32 = np.ascontiguousarray(np.asarray(enc["z_star"], dtype=np.float32))
    if zstar32.shape != (d,) or t32.shape != (d,):
        _stop(C.INPUT, "target %s and its latent %s must be (%d,)" % (t32.shape, zstar32.shape, d))
    u32 = A.decode_bank_std(model, st, bank)
    z32, j32 = A.reencode_bank(model, st, u32)
    back32 = A.decode_bank_std(model, st, zstar32[None, :])
    # -- float64: the flow only; h from the registered float32 conditioner; identical input values
    with torch.no_grad():
        h32 = F1.pinned("condition", model.condition, st["cond"], st["mask"])
        if h32.dtype != torch.float32:
            _stop(MISMATCH, "the conditioning vector is %s, expected float32" % h32.dtype)
        direct = F1.pinned("flow.decode (float32, direct)", model.flow.decode, torch.from_numpy(bank), h32)
        h64 = h32.to(torch.float64)
        zb = torch.from_numpy(bank).to(torch.float64)
        u64_t = F1.pinned("flow.decode (float64, bank)", f64.decode, zb, h64)
        z64_t, ldj64_t = F1.pinned("flow.encode (float64, bank)", f64.encode, u64_t, h64)
        tt = torch.from_numpy(np.ascontiguousarray(t32[None, :])).to(torch.float64)
        zs64_t, ldjs64_t = F1.pinned("flow.encode (float64, target)", f64.encode, tt, h64)
        back64_t = F1.pinned("flow.decode (float64, target)", f64.decode, zs64_t, h64)
    direct_np = _np64("direct float32 decode", direct, (k, d), torch.float32)
    u64 = _np64("float64 decoded bank", u64_t, (k, d), torch.float64)
    z64 = _np64("float64 re-encoded bank", z64_t, (k, d), torch.float64)
    j64 = -_np64("float64 log-determinant (bank)", ldj64_t.reshape(-1), (k,), torch.float64) / d
    zs64 = _np64("float64 latent of the target", zs64_t, (1, d), torch.float64)
    js64 = -float(_np64("float64 log-determinant (target)", ldjs64_t.reshape(-1), (1,), torch.float64)[0]) / d
    back64 = _np64("float64 target round trip", back64_t, (1, d), torch.float64)
    nll64 = C.nll_prior_per_dim(C.q_of(zs64[0])) + js64
    if not math.isfinite(nll64):
        _stop(C.NONFINITE, "float64 likelihood of the target is not finite")
    bank64, u32_64 = bank.astype(np.float64), u32.astype(np.float64)
    t_ref = t32.astype(np.float64)[None, :]
    out = {"bank_direction": {"f32": residual_stats(z32.astype(np.float64) - bank64, bank64),
                              "f64": residual_stats(z64 - bank64, bank64)},
           "target_direction": {"f32": residual_stats(back32.astype(np.float64) - t_ref, t_ref),
                                "f64": residual_stats(back64 - t_ref, t_ref)}}
    bank_latent = float(np.max(np.abs(bank64)))
    for p in PRECISIONS:
        out["bank_direction"][p]["latent_max_abs"] = bank_latent
    out["target_direction"]["f32"]["latent_max_abs"] = float(np.max(np.abs(zstar32.astype(np.float64))))
    out["target_direction"]["f64"]["latent_max_abs"] = float(np.max(np.abs(zs64)))
    du = u32_64 - u64
    dj = np.abs(np.asarray(j32, dtype=np.float64) - j64)
    if not np.all(np.isfinite(dj)):
        _stop(C.NONFINITE, "a log-determinant difference is not finite")
    out.update({
        "nll_f32_vs_stored_abs": abs(enc["NLL"] - float(stored_nll)),
        "nll_f64_vs_stored_abs": abs(nll64 - float(stored_nll)),
        "nll_f64_vs_f32_abs": abs(nll64 - enc["NLL"]),
        "j_bank_f32_vs_f64_max_abs": float(np.max(dj)),
        "j_target_f32_vs_f64_abs": abs(enc["J"] - js64),
        "decoded_bank_f32_vs_f64": {"max_abs": float(np.max(np.abs(du))), "rms": float(np.sqrt(np.mean(du * du)))},
        "direct_f32_vs_registered_max_abs": float(np.max(np.abs(direct_np - u32_64)))})
    return out


# ---- summary --------------------------------------------------------------------------------
def summarise(records, limits):
    """Maxima by set, direction and precision; the conditions of plan section 3 (Q-R3, Q-R6, Q-R7);
    a tolerance by the rule from the DEVELOPMENT set only, checked on the validation set."""
    def worst(rows, get):
        if not rows:
            _stop(POPULATION, "a summary set is empty")
        r = max(rows, key=get)
        return {"value": get(r), "state": r["state"], "slice_position": r["slice_position"]}

    by = {s: [r for r in records if r["set"] == s] for s in SETS}
    out = {"n_slice_states": {s: len(by[s]) for s in SETS}, "maxima": {}, "above_present_tolerance": {}, "checks": {}, "proposal": {}}
    for s in SETS + ("all",):
        rows = records if s == "all" else by[s]
        out["maxima"][s] = {d: {p: worst(rows, lambda r, d=d, p=p: r[d][p]["max_abs"]) for p in PRECISIONS} for d in DIRECTIONS}
        out["above_present_tolerance"][s] = {
            d: sum(1 for r in rows if r[d]["f32"]["max_abs"] > limits["present_tolerance_max_abs"]) for d in DIRECTIONS}
    scal = {"nll_f32_vs_stored_abs": None, "nll_f64_vs_stored_abs": limits["nll_f64_vs_stored_nats_per_dim"],
            "nll_f64_vs_f32_abs": limits["nll_f64_vs_f32_nats_per_dim"],
            "j_bank_f32_vs_f64_max_abs": limits["j_f32_vs_f64_nats_per_dim"],
            "j_target_f32_vs_f64_abs": limits["j_f32_vs_f64_nats_per_dim"], "direct_f32_vs_registered_max_abs": 0.0}
    for name, lim in scal.items():
        w = worst(records, lambda r, name=name: r[name])
        out["checks"][name] = dict(w, limit=lim, within=None if lim is None else bool(w["value"] <= lim))
    w = worst(records, lambda r: r["decoded_bank_f32_vs_f64"]["max_abs"])
    out["checks"]["decoded_bank_f32_vs_f64_max_abs"] = dict(
        w, limit=limits["decoded_bank_f32_vs_f64_max_abs"], within=bool(w["value"] <= limits["decoded_bank_f32_vs_f64_max_abs"]))
    ck = out["checks"]
    shared = {
        "f64_path_valid_likelihood": bool(ck["nll_f64_vs_stored_abs"]["within"] and ck["nll_f64_vs_f32_abs"]["within"]),
        "f64_path_differs_by_dtype_only": bool(ck["direct_f32_vs_registered_max_abs"]["within"]),
        "f64_roundtrip_within_limit_all": bool(all(out["maxima"]["all"][d]["f64"]["value"] <= limits["f64_roundtrip_max_abs"] for d in DIRECTIONS)),
        "j_within_limit_all": bool(ck["j_bank_f32_vs_f64_max_abs"]["within"] and ck["j_target_f32_vs_f64_abs"]["within"]),
        "decoded_bank_within_limit_all": bool(ck["decoded_bank_f32_vs_f64_max_abs"]["within"])}
    for d in DIRECTIONS:
        dev = out["maxima"]["development"][d]["f32"]["value"]
        val = out["maxima"]["validation"][d]["f32"]["value"]
        t = tolerance_by_rule(dev)
        cond = dict(shared, t_within_ceiling=bool(t <= limits["tolerance_ceiling_max_abs"]), validation_all_within_t=bool(val <= t))
        out["proposal"][d] = {"development_max_f32": dev, "validation_max_f32": val, "t_by_rule": t, "conditions": cond,
                              "proposed_t": t if all(cond.values()) else None}
    out["proposal"]["note"] = PROPOSAL_NOTE
    return out


# ---- report ---------------------------------------------------------------------------------
def check_report(rep):
    """Exact top-level keys; no key of a D-LAT record, measure, contrast, class or figure; finite numbers only."""
    if set(rep) != REPORT_KEYS:
        _stop(A.RECORD_SCHEMA, "report keys: missing %s, unexpected %s" % (sorted(REPORT_KEYS - set(rep)), sorted(set(rep) - REPORT_KEYS)))

    def walk(o, where):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in F1.REFUSED_KEYS:
                    _stop(A.RECORD_SCHEMA, "report holds the key %r at %s: the diagnostic writes no measure" % (k, where))
                walk(v, "%s.%s" % (where, k))
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, where)
        elif isinstance(o, float) and not math.isfinite(o):
            _stop(C.NONFINITE, "report holds a non-finite number at %s" % where)
    walk(rep, "report")
    return True


# ---- the run --------------------------------------------------------------------------------
def _wall(t0, ceiling):
    el = time.monotonic() - t0
    if el > ceiling:
        _stop(GD.RESOURCE_LIMIT, "wall time %.0f s exceeds the ceiling %d s" % (el, ceiling))
    return el


def _collect(b, card, rep, records, timing, t0):
    reg, pop, lim = card["registered"], card["population"], card["limits"]
    if card["selftest"] is not bool(b.selftest):
        _stop(RUN_CARD, "run card selftest = %r but the binding's is %r" % (card["selftest"], b.selftest))
    if not (sys.flags.isolated and sys.dont_write_bytecode):
        _stop(A.ORIGIN, "the interpreter must run with -I -B (isolated %d, no bytecode %s)" % (sys.flags.isolated, sys.dont_write_bytecode))
    here = os.path.dirname(os.path.abspath(__file__))
    torch = A.seams().torch
    rep["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                          "torch_threads": int(torch.get_num_threads()), "platform": platform.platform(),
                          "isolated": bool(sys.flags.isolated), "no_bytecode": bool(sys.dont_write_bytecode),
                          "code_sha256": {n: BD.sha_file(os.path.join(here, n)) for n in sorted(os.listdir(here)) if n.endswith(".py")},
                          "pin": b.pin}
    # -- binding (the checks of F1 stages D2, D3 that this diagnostic depends on)
    order = list(b.state_order)
    if not (len(order) == len(set(order)) == reg["n_states"] and set(b.states) == set(order)):
        _stop(POPULATION, "%d bound states, expected exactly %d distinct" % (len(order), reg["n_states"]))
    n = reg["n_volumes"]
    if not (len(b.identities) == len(set(b.identities)) == n):
        _stop(POPULATION, "%d bound slices, expected %d distinct" % (len(b.identities), n))
    dev_seeds, val_seeds = set(pop["development_seeds"]), set(pop["validation_seeds"])
    set_of = {}
    for sid in order:
        seed = b.states[sid]["seed"]
        if seed not in dev_seeds | val_seeds:
            _stop(POPULATION, "state %s has seed %r, which is in neither set of the run card" % (str(sid)[:12], seed))
        set_of[sid] = "development" if seed in dev_seeds else "validation"
        row = b.stored.get(sid)
        if not (isinstance(row, dict) and "NLL" in row and np.asarray(row["NLL"]).shape == (n,) and np.all(np.isfinite(row["NLL"]))):
            _stop(POPULATION, "stored likelihood row of state %s is missing, incomplete or non-finite" % str(sid)[:12])
    counts = {s: 3 * sum(1 for sid in order if set_of[sid] == s) for s in SETS}
    if counts != {"development": pop["n_development_slice_states"], "validation": pop["n_validation_slice_states"]}:
        _stop(POPULATION, "the sets hold %r slice-states; the run card says %d and %d"
              % (counts, pop["n_development_slice_states"], pop["n_validation_slice_states"]))
    states = b.slice_states()
    ids = [(str(st["identity"]["file"]), int(st["identity"]["slice_index"])) for st in states]
    if ids != [(str(f), int(i)) for f, i in b.identities]:
        _stop(POPULATION, "the slice states are not the registered identities in the registered order")
    devs = sorted({str(v.device) for st in states for v in st.values() if isinstance(v, torch.Tensor)})
    if devs != [card["device"]]:
        _stop(MISMATCH, "slice-state tensors are on %s, the run card says %s" % (devs, card["device"]))
    bank = np.asarray(b.bank())
    d = int(A.seams().ffr.FLOW_DIM_REAL)
    bank_sha = F1.sha_bytes(np.ascontiguousarray(bank).tobytes(order="C"))
    if bank.shape != (reg["bank_rows"], d) or bank.dtype != np.float32 or bank_sha != b.bank_sha256:
        _stop(A.ORIGIN, "bank %s %s sha256 %s, registered (%d, %d) float32 %s"
              % (bank.dtype, bank.shape, bank_sha, reg["bank_rows"], d, b.bank_sha256))
    picked = pick_slices(n)
    rep["binding"] = {"manifest_sha256": b.manifest_sha256, "n_states": len(order), "n_slices_bound": n,
                      "bank_sha256": bank_sha, "bank_shape": list(bank.shape), "files_hashed": int(b.files_hashed),
                      "holdout": "not requested: validation view only"}
    rep["population"] = {"slice_positions": picked, "slices": [[ids[i][0], ids[i][1]] for i in picked],
                         "rule": "first, middle and last of the bound slices", "sets": {s: counts[s] for s in SETS},
                         "development_seeds": sorted(dev_seeds), "validation_seeds": sorted(val_seeds),
                         "states": [{"index": k, "model_state_sha256": str(sid), "arm": b.states[sid]["arm"],
                                     "seed": b.states[sid]["seed"], "epoch": b.states[sid]["epoch"], "set": set_of[sid]}
                                    for k, sid in enumerate(order)]}
    for k, sid in enumerate(order):
        t = time.monotonic()
        model, ver = b.load_state(sid)
        t_load = time.monotonic() - t
        if ver["model_state_sha256"] != sid or ver["file_sha256"] != b.states[sid]["file_sha256"]:
            _stop(A.ORIGIN, "state %s: the loader returned model state %s, file %s; the manifest names %s, %s"
                  % (str(sid)[:12], ver["model_state_sha256"], ver["file_sha256"], sid, b.states[sid]["file_sha256"]))
        dr = A.device_record(model)
        if dr["devices"] != [card["device"]] or dr["dtypes"] != ["torch.float32"]:
            _stop(MISMATCH, "state %s: parameters on %s with %s; the run card says %s float32"
                  % (str(sid)[:12], dr["devices"], dr["dtypes"], card["device"]))
        t = time.monotonic()
        f64 = make_f64_flow(model)
        t_copy = time.monotonic() - t
        t = time.monotonic()
        for i in picked:
            rec = measure(model, f64, states[i], bank, b.stored[sid]["NLL"][i])
            rec.update({"state": str(sid), "state_index": k, "set": set_of[sid], "slice_position": i})
            records.append(rec)
            LOG.info("state %2d %s %s slice %d: bank f32 %.3e f64 %.3e; target f32 %.3e f64 %.3e", k, str(sid)[:12], set_of[sid], i,
                     rec["bank_direction"]["f32"]["max_abs"], rec["bank_direction"]["f64"]["max_abs"],
                     rec["target_direction"]["f32"]["max_abs"], rec["target_direction"]["f64"]["max_abs"])
        after = A.device_record(model)
        if after["dtypes"] != ["torch.float32"]:
            _stop(MISMATCH, "state %s: the model's parameters are %s after the measurement" % (str(sid)[:12], after["dtypes"]))
        timing["states"].append({"index": k, "load_s": t_load, "f64_copy_s": t_copy, "measure_s": time.monotonic() - t})
        del model, f64
        _wall(t0, card["ceilings"]["wall_s"])                        # both ceilings after every state
        GD.check_memory(card["ceilings"]["ram_bytes"])
    if len(records) != reg["n_states"] * 3:
        _stop(POPULATION, "%d slice-state records, expected %d" % (len(records), reg["n_states"] * 3))
    rep["summary"] = summarise(records, lim)


def run_rtnum(binding, card, out_dir, card_sha256="(not from a file)", banner=BANNER):
    """binding: a binding object or a function that builds one (production). Returns (report, exit code).
    out_dir must not exist; it is created here. Exit 0 = the diagnostic completed (RTNUM: COMPLETE);
    this says nothing about a tolerance."""
    check_card(card)
    try:
        os.makedirs(out_dir, exist_ok=False)
    except OSError as exc:
        _stop(GD.GUARD_STATE, "cannot create the new output folder %s: %s" % (out_dir, exc))
    mem = F1.MemoryLog()
    mem.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(mem)
    t0 = time.monotonic()
    rep = {k: None for k in REPORT_KEYS}
    records, timing = [], {"states": [], "ceiling_s": card["ceilings"]["wall_s"], "elapsed_s": None}
    rep.update(schema=REPORT_SCHEMA, banner=banner, run_card={"sha256": card_sha256, "content": card},
               versions={"diagnostic": __version__, "f1_driver": F1.__version__, "adapter": A.__version__, "binder": BD.__version__,
                         "core": C.__version__, "guard": GD.__version__})
    try:
        guard = GD.WriteGuard(out_dir, card["ceilings"]["disk_bytes"], card["ceilings"]["reserve_bytes"])
        try:
            with GD.memory_guarded(card["ceilings"]["ram_bytes"]):
                b = binding() if callable(binding) else binding      # production: manifest and files bound here
                _collect(b, card, rep, records, timing, t0)
            timing["elapsed_s"] = _wall(t0, card["ceilings"]["wall_s"])      # final check before the verdict
            peak = GD.check_memory(card["ceilings"]["ram_bytes"])
            rep["memory"] = {"peak_rss_bytes": peak, "limit_bytes": card["ceilings"]["ram_bytes"],
                             "disk_written_bytes": guard.written, "reserve_bytes": card["ceilings"]["reserve_bytes"]}
        except DiagLatError as exc:
            rep["stop"] = {"code": exc.code, "message": exc.message, "slice_states_done": len(records)}
            LOG.error("the diagnostic stopped after %d slice-states: %s", len(records), exc)
        if timing["elapsed_s"] is None:
            timing["elapsed_s"] = time.monotonic() - t0
        rep["slice_states"], rep["timing"] = records, timing
        ok = rep["stop"] is None and rep["summary"] is not None
        rep["verdict"] = "RTNUM: COMPLETE" if ok else "RTNUM: FAIL"
        try:
            check_report(rep)
        except DiagLatError as exc:                  # the report is not written; a minimal failure report is
            ok = False
            LOG.error("the report failed its schema and is not written: %s", exc)
            rep = {k: None for k in REPORT_KEYS}
            rep.update(schema=REPORT_SCHEMA, banner=banner, run_card={"sha256": card_sha256, "content": card},
                       stop={"code": exc.code, "message": exc.message, "slice_states_done": len(records)},
                       slice_states=[], verdict="RTNUM: FAIL")
        files = {"rtnum_report.json": F1.dumps(rep), "rtnum_verdict.txt": (rep["verdict"] + "\n").encode(),
                 "rtnum.log": ("\n".join(mem.lines) + "\n").encode()}
        for name, data in files.items():
            guard.write(name, data)
        guard.write("outputs_sha256.txt", "".join("%s  %s\n" % (F1.sha_bytes(dd), nn) for nn, dd in sorted(files.items())).encode())
        return rep, (0 if ok else 1)
    finally:
        root.removeHandler(mem)


def main():
    ap = argparse.ArgumentParser(description="D-LAT round-trip numerical diagnostic (residual statistics only)", allow_abbrev=False)
    ap.add_argument("--run-card", required=True)
    ap.add_argument("--run-card-sha256", required=True)
    ap.add_argument("--repo", required=True, help="the pinned V05LT checkout (read-only)")
    ap.add_argument("--manifest", required=True, help="the registered D-UNC manifest_f2.json")
    ap.add_argument("--out", required=True, help="a new output folder; it must not exist")
    ap.add_argument("--dunc-dir", required=True, help="folder of the registered diag_unc_adapter.py (verified by sha256)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        card, card_sha = load_card(args.run_card, args.run_card_sha256)
        if card["selftest"]:
            _stop(RUN_CARD, "a self-test run card is not accepted by the production entry point")
        if not os.path.isdir(args.dunc_dir):
            _stop(A.ORIGIN, "--dunc-dir must be an existing folder")
        sys.path.append(os.path.realpath(args.dunc_dir))             # after the D-LAT code folder: nothing is shadowed
        sys.path.append(os.path.realpath(args.repo))                 # the pinned checkout (as the F1 driver does)
        rep, code = run_rtnum(lambda: F1.ProductionBinding(args.repo, args.manifest, card), card, args.out, card_sha)
        print(rep["verdict"])
        return code
    except DiagLatError as exc:
        LOG.error("the diagnostic did not start or could not write its evidence: %s", exc)
        print("RTNUM: FAIL (%s)" % exc.code)
        return 1
    except Exception as exc:                     # logged and re-raised: never swallowed
        LOG.error("unexpected error: %s: %s", type(exc).__name__, exc)
        raise


if __name__ == "__main__":
    sys.exit(main())
