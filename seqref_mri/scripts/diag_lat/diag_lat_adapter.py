# SEQREF-DLAT-ADP v0.1 -- diag_lat_adapter.py: D-LAT adapter to the pinned V05LT production seams (plan SEQREF-DLAT-P3 v0.2)
# LIFETIME: KEEP
"""Adapter between the D-LAT mathematics (diag_lat_core, P1) and the pinned V05LT classes.

PRE-FREEZE ENGINEERING FILE. Written against candidate v0.5.2 (not frozen) and plan
SEQREF-DLAT-P3 v0.2 with clarifications C2-C4 (LudWiki 238 LE23, LE24). Not D-LAT evidence.

Every model operation is a call into a pinned function, reached through the hash-verified
D-UNC adapter (diag_unc_adapter.seams()). The pinned tree is imported, never edited.

Seams used (module -> function):
  d3                 _encode_slice_conditioned            (target encode, own or explicit cond/mask)
  free_flow_runtime  FreeFlowModel.condition, .flow.encode, .decode_scalars, FLOW_DIM_REAL
  forward_operator   MaskedFourierOperator.A, .A_adjoint, complex_to_two_channel
  fastmri_data       fft2c
  diag_lat_core      every measure, check and typed stop

Arithmetic in this file (complete list; everything else is a pinned or core call):
  (a) J = -ldj / d and NLL = B + J            (encode_truth, reencode_bank; 235-candidate section 3)
  (b) cond' = two_channel(A_adjoint(y + n)) / amax, one division (perturbed_input; production _prepare order)
  (c) the interleaved layout of the scale and mean vectors (scale_mean_interleaved; pack_scalars order)
  (d) comparisons of the checks S2, S4 and P1 (maximum absolute differences, one relative L2 norm)
Items (a) and (d) go beyond the wording of plan v0.2 section 2 ("no arithmetic of its own except
the single division by amax and the layout of the scale and mean vectors"): DISCLOSED for review.

Layout: D-LAT stays in the production interleaved layout [Re0, Im0, Re1, Im1, ...]; the D-UNC
block re-ordering is not used. Device: the adapter adds no device transfer.
Importing this module imports no PyTorch and no project code; the first seam call does.
No fallback, placeholder, mock or dummy return. Every failure is logged and raised as a typed
DiagLatError. No decoded sample and no latent vector is written to disk by this module.

Typed codes added here: DIAG_LAT_ORIGIN, DIAG_LAT_ADAPTER, DIAG_LAT_RECORD_SCHEMA; from the
candidate: DIAG_LAT_JACOBIAN_FIXTURE, DIAG_LAT_PERTURB_FIXTURE.

Changelog:
  v0.1 (2026-11-21) first version: seams, swap and perturbation inputs, per-slice record,
        RECORD_KEYS schema with validate_record (C3), checks S2, S4 and P1.
"""
import hashlib
import importlib.util
import logging
import math

import numpy as np

import diag_lat_core as C
from diag_lat_core import DiagLatError

__version__ = "0.1"
ABBR = "SEQREF-DLAT-ADP"
LOG = logging.getLogger("diag_lat.adapter")

DUNC_ADAPTER_SHA256 = "c1715841bf6805568b88f9195acbe27933a7e1fec9acc8bb22a33ff555a8505e"
ORIGIN = "DIAG_LAT_ORIGIN"
ADAPTER = "DIAG_LAT_ADAPTER"
RECORD_SCHEMA = "DIAG_LAT_RECORD_SCHEMA"
JACOBIAN_FIXTURE = "DIAG_LAT_JACOBIAN_FIXTURE"      # 235-candidate section 6 (raised by the fixture file)
PERTURB_FIXTURE = "DIAG_LAT_PERTURB_FIXTURE"        # 235-candidate section 6, P1
SWAP_REL_TOL = 1e-5                 # S1 / S2 tolerance, provisional until F1 (235-candidate section 6)

# ---- record schema (E7 of 238 LE22; C3 of 238 LE23) ---------------------------------
# name, bank-dependent, subset requirement, expected type.
#   "float"            : finite Python float
#   "float|NOT_COMPUTED": finite Python float or the explicit status diag_lat_core.NOT_COMPUTED
RECORD_KEYS = (
    {"name": "KS", "bank": False, "subsets": "none", "type": "float"},
    {"name": "P_swap", "bank": False, "subsets": "none", "type": "float"},
    {"name": "NLL", "bank": False, "subsets": "none", "type": "float"},
    {"name": "R0_PSNR", "bank": False, "subsets": "none", "type": "float"},
    {"name": "log_gamma_ratio", "bank": True, "subsets": "every label", "type": "float"},
    {"name": "d01", "bank": True, "subsets": "every label", "type": "float"},
    {"name": "a01", "bank": True, "subsets": "every label", "type": "float|NOT_COMPUTED"},
    {"name": "TC_G", "bank": True, "subsets": "every label", "type": "float|NOT_COMPUTED"},
    {"name": "phys_lgr", "bank": True, "subsets": "every label", "type": "float"},
    {"name": "phys_d01", "bank": True, "subsets": "every label", "type": "float"},
    {"name": "phys_a01", "bank": True, "subsets": "every label", "type": "float|NOT_COMPUTED"},
)
PHYSICAL_NAMES = {"L2": "phys_lgr", "L3a": "phys_d01", "L3b": "phys_a01"}
BANK_FIELD = "bank"                 # record[BANK_FIELD][subset label][key name]

_DUA = None


def _stop(code, message):
    raise DiagLatError(code, message)            # DiagLatError logs at construction


def _value_ok(v, typ):
    if typ == "float|NOT_COMPUTED" and v is C.NOT_COMPUTED:
        return True
    return isinstance(v, float) and not isinstance(v, bool) and math.isfinite(v)


def validate_record(record, subset_labels):
    """Check one slice-state record against RECORD_KEYS: exact key set, structure and types."""
    labels = list(subset_labels)
    if not isinstance(record, dict):
        _stop(RECORD_SCHEMA, "record is %s, not a dict" % type(record).__name__)
    flat = [k for k in RECORD_KEYS if not k["bank"]]
    bank = [k for k in RECORD_KEYS if k["bank"]]
    want = {k["name"] for k in flat} | {BANK_FIELD}
    if set(record) != want:
        _stop(RECORD_SCHEMA, "record keys: missing %s, unexpected %s"
              % (sorted(want - set(record)), sorted(set(record) - want)))
    for k in flat:
        if not _value_ok(record[k["name"]], k["type"]):
            _stop(RECORD_SCHEMA, "%s = %r is not %s" % (k["name"], record[k["name"]], k["type"]))
    b = record[BANK_FIELD]
    if not isinstance(b, dict) or set(b) != set(labels):
        _stop(RECORD_SCHEMA, "bank part must be a dict with exactly the subset labels %s" % sorted(labels))
    names = {k["name"] for k in bank}
    for lab in labels:
        if not isinstance(b[lab], dict) or set(b[lab]) != names:
            got = set(b[lab]) if isinstance(b[lab], dict) else set()
            _stop(RECORD_SCHEMA, "subset %s: missing %s, unexpected %s"
                  % (lab, sorted(names - got), sorted(got - names)))
        for k in bank:
            if not _value_ok(b[lab][k["name"]], k["type"]):
                _stop(RECORD_SCHEMA, "subset %s: %s = %r is not %s" % (lab, k["name"], b[lab][k["name"]], k["type"]))
    return True


# ---- origin and seams -----------------------------------------------------------------
def verify_origin():
    """sha256 of diag_unc_adapter.py, checked BEFORE it is imported. Returns (path, sha256)."""
    spec = importlib.util.find_spec("diag_unc_adapter")
    if spec is None or not spec.origin:
        _stop(ORIGIN, "diag_unc_adapter.py is not on the import path")
    with open(spec.origin, "rb") as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    if got != DUNC_ADAPTER_SHA256:
        _stop(ORIGIN, "diag_unc_adapter.py at %s has sha256 %s, registered %s" % (spec.origin, got, DUNC_ADAPTER_SHA256))
    return spec.origin, got


def seams():
    """Pinned production modules, through the hash-verified D-UNC adapter (imported once)."""
    global _DUA
    if _DUA is None:
        verify_origin()
        import diag_unc_adapter as dua
        _DUA = dua
    try:
        return _DUA.seams()
    except _DUA.AdapterError as exc:
        _stop(ADAPTER, "D-UNC adapter: %s" % exc)


def _pinned(where, fn, *args, **kwargs):
    """Run a pinned call; any failure is logged with its location and raised as a typed stop."""
    try:
        return fn(*args, **kwargs)
    except DiagLatError:
        raise
    except Exception as exc:
        raise DiagLatError(ADAPTER, "%s: %s: %s" % (where, type(exc).__name__, exc)) from exc


def _finite(name, a):
    if not np.all(np.isfinite(a)):
        _stop(C.NONFINITE, "%s contains a non-finite value" % name)
    return a


def _own(st, cond, mask):
    return (st["cond"] if cond is None else cond), (st["mask"] if mask is None else mask)


def device_record(model):
    """Devices and dtypes of the parameters, and the thread count. No transfer is made."""
    torch = seams().torch
    params = list(model.parameters())
    return {"devices": sorted({str(p.device) for p in params}), "dtypes": sorted({str(p.dtype) for p in params}),
            "n_parameters": int(sum(p.numel() for p in params)), "torch": torch.__version__,
            "torch_threads": int(torch.get_num_threads()), "cuda_available": bool(torch.cuda.is_available())}


# ---- flow passes ------------------------------------------------------------------------
def encode_truth(model, st, cond=None, mask=None):
    """Encode the stored target under (cond, mask); the slice's own by default, explicit for L5.
    Convention (T9): ldj = log|det dz/du|, so log p(u) = log N(z) + ldj, J = -ldj/d, NLL = B + J."""
    s = seams()
    cond, mask = _own(st, cond, mask)
    z, ldj, log_pz = _pinned("d3._encode_slice_conditioned", s.d3._encode_slice_conditioned, model, st, cond, mask)
    d = int(z.size)
    if d != int(s.ffr.FLOW_DIM_REAL):
        _stop(C.INPUT, "encode returned %d scalars, expected %d" % (d, s.ffr.FLOW_DIM_REAL))
    b = C.nll_prior_per_dim(C.q_of(z))
    j = -float(ldj) / d
    nll = b + j
    if not (math.isfinite(j) and math.isfinite(nll)):
        _stop(C.NONFINITE, "encode_truth: J = %r, NLL = %r" % (j, nll))
    return {"z_star": z, "ldj": float(ldj), "log_pz": float(log_pz), "d": d, "B": b, "J": j, "NLL": nll}


def decode_bank_std(model, st, bank, cond=None, mask=None):
    """Bank rows (K, d) float32 -> the K standardised sample vectors (K, d) float32."""
    s = seams()
    cond, mask = _own(st, cond, mask)
    bank = np.asarray(bank)
    if bank.dtype != np.float32 or bank.ndim != 2 or bank.shape[1] != int(s.ffr.FLOW_DIM_REAL):
        _stop(C.INPUT, "bank must be (K, %d) float32, got %s %s" % (s.ffr.FLOW_DIM_REAL, bank.dtype, bank.shape))
    with s.torch.no_grad():
        us = _pinned("decode_scalars (bank)", model.decode_scalars,
                     s.torch.from_numpy(np.ascontiguousarray(bank)), cond, mask)
    out = np.ascontiguousarray(us.detach().cpu().numpy())
    if out.dtype != np.float32 or out.shape != bank.shape:
        _stop(C.INPUT, "decode returned %s %s for a bank of %s" % (out.dtype, out.shape, bank.shape))
    return _finite("decoded bank", out)


def reencode_bank(model, st, u_std, cond=None, mask=None):
    """Re-encode decoded vectors (K, d) float32 -> (z (K, d) float32, J (K,) float64 = -ldj/d).
    The pinned decode returns no log-determinant, so J(u_k) comes from this encode pass."""
    s = seams()
    cond, mask = _own(st, cond, mask)
    u = np.asarray(u_std)
    if u.dtype != np.float32 or u.ndim != 2:
        _stop(C.INPUT, "re-encode needs (K, d) float32, got %s %s" % (u.dtype, u.shape))
    with s.torch.no_grad():
        h = _pinned("condition", model.condition, cond, mask)
        z_t, ldj_t = _pinned("flow.encode (bank)", model.flow.encode, s.torch.from_numpy(np.ascontiguousarray(u)), h)
    z = np.ascontiguousarray(z_t.detach().cpu().numpy())
    ldj = ldj_t.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if z.shape != u.shape or ldj.shape != (u.shape[0],):
        _stop(C.INPUT, "re-encode returned z %s, ldj %s for input %s" % (z.shape, ldj.shape, u.shape))
    _finite("re-encoded z", z)
    return z, _finite("J of the bank", -ldj / u.shape[1])


def decode_r0_std(model, st, cond=None, mask=None):
    """R0 in standardised scalars (d,) float32: the pinned decode at z = 0."""
    s = seams()
    cond, mask = _own(st, cond, mask)
    with s.torch.no_grad():
        us = _pinned("decode_scalars (R0)", model.decode_scalars,
                     s.torch.zeros(1, int(s.ffr.FLOW_DIM_REAL)), cond, mask)
    return _finite("R0", np.ascontiguousarray(us.detach().cpu().numpy()[0]))


def conditioning_vector(model, st, cond=None, mask=None):
    """h = model.condition(cond, mask) as float64 (1-D); used by fixtures and the C2 check."""
    s = seams()
    cond, mask = _own(st, cond, mask)
    with s.torch.no_grad():
        h = _pinned("condition", model.condition, cond, mask)
    return _finite("h", h.detach().cpu().numpy().astype(np.float64).reshape(-1))


# ---- coordinates --------------------------------------------------------------------------
def target_std(st):
    """Stored target: (float64 (d,), float32 (d,)) standardised interleaved scalars.
    The float32 vector is the one the pinned encode passes to the flow."""
    t = np.asarray(st["target"])
    if t.dtype != np.float64 or t.ndim != 2 or t.shape[0] != 1:
        _stop(C.INPUT, "state target must be (1, d) float64, got %s %s" % (t.dtype, t.shape))
    return _finite("target", t[0]), np.ascontiguousarray(t[0], dtype=np.float32)


def scale_mean_interleaved(st):
    """Per-coordinate (scale, mean), each (d,) float64, in the production interleaved order
    of free_flow_runtime.pack_scalars: index 2k is the real part, 2k + 1 the imaginary part."""
    v = st["vecs"]
    n = int(v["n_free_complex"])
    scale = np.empty(2 * n, dtype=np.float64)
    mean = np.empty(2 * n, dtype=np.float64)
    scale[0::2], scale[1::2] = v["scale_re"], v["scale_im"]
    mean[0::2], mean[1::2] = v["mean_re"], v["mean_im"]
    if not (np.all(np.isfinite(scale)) and np.all(np.isfinite(mean)) and np.all(scale > 0.0)):
        _stop(C.INPUT, "standardisation vectors are non-finite or have a non-positive scale")
    return scale, mean


# ---- L5 inputs ------------------------------------------------------------------------------
def _operator(st):
    s = seams()
    from seqref_mri.src import forward_operator as fop          # pinned; already loaded by the seams
    mask1 = st["mask"][0]
    if mask1.dtype != s.torch.bool:
        _stop(C.INPUT, "state mask dtype is %s, expected bool" % mask1.dtype)
    return fop, _pinned("MaskedFourierOperator", fop.MaskedFourierOperator, mask1)


def acquired_cells(st):
    """(rows, cols) of the acquired cells in row-major order (row, then ascending acquired
    column) -- the order of the pinned coordinate map applied to the acquired set."""
    m = st["mask"][0].detach().cpu().numpy().astype(bool)
    rc = np.argwhere(np.broadcast_to(m[None, :], (m.size, m.size)))
    return np.ascontiguousarray(rc[:, 0]), np.ascontiguousarray(rc[:, 1])


def acquired_values(st):
    """Stored measurement y_i on the acquired cells, complex128 (n_cells,), unnormalised units."""
    r, c = acquired_cells(st)
    y = st["y"][0].detach().cpu().numpy()
    return _finite("y on the acquired cells", y[r, c].astype(np.complex128))


def swap_check_s2(st_i, cond_c, x_j):
    """S2, recipient-mask correctness: with k = fft2c(complex cond_swap) and s = max|k| over the
    recipient's acquired columns, max|k| <= 1e-5 s on the excluded columns and k equals
    fft2c(x_true_c[j]) within 1e-5 s on the acquired columns. Returns (excluded, acquired, tol)."""
    s = seams()
    k = _pinned("fft2c(cond_swap)", s.fdm.fft2c, cond_c).detach().cpu().numpy()
    k_ref = _pinned("fft2c(x_true_c[j])", s.fdm.fft2c, x_j).detach().cpu().numpy()
    acq = st_i["mask"][0].detach().cpu().numpy().astype(bool)
    tol = C.swap_tolerance_scale(k[:, acq], SWAP_REL_TOL)            # stops if the scale is 0
    excluded = float(np.max(np.abs(k[:, ~acq]))) if (~acq).any() else 0.0
    acquired = float(np.max(np.abs(k[:, acq] - k_ref[:, acq])))
    if not (math.isfinite(excluded) and math.isfinite(acquired)):
        _stop(C.NONFINITE, "swap check S2: non-finite residual")
    if excluded > tol or acquired > tol:
        _stop(C.SWAP_FIXTURE, "S2 failed: excluded columns max %.3e, acquired columns max difference %.3e, tolerance %.3e"
              % (excluded, acquired, tol))
    return excluded, acquired, tol


def swap_input(st_i, st_j, check_identity=True):
    """L5 (a), recipient-mask variant: the donor's object observed under the recipient's mask.
    y_swap = op_i.A(x_true_c[j]); cond_swap = two_channel(op_i.A_adjoint(y_swap)); no division.
    Runs S2 (recipient-mask correctness) and, unless check_identity is False (S1 self-swap
    fixture only), S4 (the content changed). The mask returned is the recipient's stored object."""
    s = seams()
    fop, op = _operator(st_i)
    x_j = st_j["x_true_c"]
    y_swap = _pinned("op_i.A", op.A, x_j)
    cond_c = _pinned("op_i.A_adjoint", op.A_adjoint, y_swap)
    cond_swap = _pinned("complex_to_two_channel", fop.complex_to_two_channel, cond_c).unsqueeze(0)
    own = st_i["cond"]
    if tuple(cond_swap.shape) != tuple(own.shape) or cond_swap.dtype != own.dtype:
        _stop(C.SWAP_FIXTURE, "cond_swap is %s %s, the stored cond is %s %s"
              % (cond_swap.dtype, tuple(cond_swap.shape), own.dtype, tuple(own.shape)))
    excluded, acquired, tol = swap_check_s2(st_i, cond_c, x_j)
    a = cond_swap.detach().cpu().numpy().astype(np.float64).ravel()
    b = own.detach().cpu().numpy().astype(np.float64).ravel()
    nb = float(np.linalg.norm(b))
    if nb == 0.0:
        _stop(C.ZERO_DENOMINATOR, "swap check S4: the recipient's own conditioning input is zero")
    rel = C.shift_norm(a, b) / nb
    if check_identity:
        C.check_swap_identity(cond_swap.detach().cpu().numpy(), own.detach().cpu().numpy())      # S4
    return {"y_swap": y_swap, "cond_swap": cond_swap, "mask": st_i["mask"],
            "s2_excluded_max": excluded, "s2_acquired_max_diff": acquired, "s2_tolerance": tol,
            "s4_relative_l2": rel, "max_abs_diff_to_own": float(np.max(np.abs(a - b))),
            "max_abs_own": float(np.max(np.abs(b)))}


def perturbed_input(st, noise):
    """L5 (b): cond' = two_channel(op_i.A_adjoint(y_i + n_i)) / amax_i -- one division, after the
    adjoint, as the pinned train_base._prepare does. noise: complex (n_cells,) on the acquired
    cells in the order of acquired_cells(), already scaled by diag_lat_core.scale_noise."""
    s = seams()
    fop, op = _operator(st)
    r, c = acquired_cells(st)
    n = np.asarray(noise)
    if not np.iscomplexobj(n) or n.shape != (r.size,):
        _stop(C.INPUT, "noise must be complex (%d,), got %s %s" % (r.size, n.dtype, n.shape))
    if not (np.all(np.isfinite(n.real)) and np.all(np.isfinite(n.imag))):
        _stop(C.NONFINITE, "L5(b): non-finite noise")
    y = st["y"]
    add = s.torch.zeros_like(y)
    add[0, s.torch.from_numpy(r), s.torch.from_numpy(c)] = s.torch.from_numpy(n).to(y.dtype)
    y_pert = y + add
    x0 = s.torch.stack([_pinned("op_i.A_adjoint", op.A_adjoint, y_pert[0])])
    cond = _pinned("complex_to_two_channel", fop.complex_to_two_channel, x0) / st["amax"].view(-1, 1, 1, 1)
    if tuple(cond.shape) != tuple(st["cond"].shape) or cond.dtype != st["cond"].dtype:
        _stop(C.INPUT, "cond' is %s %s, the stored cond is %s %s"
              % (cond.dtype, tuple(cond.shape), st["cond"].dtype, tuple(st["cond"].shape)))
    if not bool(s.torch.isfinite(cond).all()):
        _stop(C.NONFINITE, "L5(b): non-finite perturbed conditioning input")
    return cond


def perturb_check_p1(st, cond_zero):
    """P1: with zero noise the rebuilt conditioning input reproduces the stored one. Expected
    exactly 0; must not exceed the S1 tolerance 1e-5 x max|cond|. Guards against a second
    division by amax. Returns the maximum absolute difference."""
    own = st["cond"].detach().cpu().numpy().astype(np.float64)
    got = cond_zero.detach().cpu().numpy().astype(np.float64)
    if got.shape != own.shape:
        _stop(PERTURB_FIXTURE, "P1: shapes %s and %s differ" % (got.shape, own.shape))
    scale = float(np.max(np.abs(own)))
    if scale == 0.0:
        _stop(C.ZERO_DENOMINATOR, "P1: the stored conditioning input is zero")
    diff = float(np.max(np.abs(got - own)))
    if not math.isfinite(diff) or diff > SWAP_REL_TOL * scale:
        _stop(PERTURB_FIXTURE, "P1 failed: max |cond' - cond| = %.3e exceeds %.3e" % (diff, SWAP_REL_TOL * scale))
    return diff


# ---- tail ---------------------------------------------------------------------------------
def tail_check(model, st, z_star):
    """Tail set of the float32 target and the invariance check on z* (stops on a mismatch)."""
    _t64, t32 = target_std(st)
    mask, tau, q_tail = C.tail_set(t32)
    n = C.check_tail_invariance(np.asarray(z_star), t32, mask)
    return {"tau": tau, "q_tail": q_tail, "n_tail": n}


# ---- one record per slice and model state ---------------------------------------------------
def slice_state_record(model, st, bank, *, subsets, p_swap, r0_psnr):
    """One record with exactly RECORD_KEYS, plus diagnostics that are not part of the record.
    subsets: {label: index array} -- an argument; production passes diag_lat_core.STABILITY_SUBSETS
    and nothing else. p_swap and r0_psnr are passed through (L5 penalty; stored R0 PSNR row).
    Returns (record, diagnostics). Nothing is written to disk."""
    if not isinstance(subsets, dict) or not subsets:
        _stop(C.INPUT, "subsets must be a non-empty {label: indices} map")
    enc = encode_truth(model, st)
    z_star = enc["z_star"]
    tail = tail_check(model, st, z_star)
    u_true, _t32 = target_std(st)
    bank = np.asarray(bank)
    u_bank = decode_bank_std(model, st, bank)
    z_re, j_bank = reencode_bank(model, st, u_bank)
    r0 = decode_r0_std(model, st)
    events = []
    z64 = z_star.astype(np.float64)
    std = C.bank_measures(u_true, r0.astype(np.float64), z64, u_bank.astype(np.float64),
                          bank.astype(np.float64), j_bank, events, subsets=subsets)
    scale, mean = scale_mean_interleaved(st)
    ut_p = C.to_physical(u_true, scale, mean)
    r0_p = C.to_physical(r0.astype(np.float64), scale, mean)
    ub_p = C.to_physical(u_bank.astype(np.float64), scale, mean)
    rec_bank = {}
    for label, idx in subsets.items():
        ev = []
        l2p = C.l2_measures(ut_p, r0_p, z64, ub_p[idx], bank[idx].astype(np.float64))
        l3p = C.l3_measures(ut_p, r0_p, ub_p[idx], ev)
        for e in ev:
            e["subset"] = label
            e["coordinates"] = "physical"
        events.extend(ev)
        m = std[label]
        rec_bank[label] = {"log_gamma_ratio": m["log_gamma_ratio"], "d01": m["d01"], "a01": m["a01"],
                           "TC_G": m["TC_G"], "phys_lgr": l2p["log_gamma_ratio"],
                           "phys_d01": l3p["d01"], "phys_a01": l3p["a01"]}
    record = {"KS": C.l1_measures(z64)["KS"], "P_swap": p_swap, "NLL": enc["NLL"], "R0_PSNR": r0_psnr,
              BANK_FIELD: rec_bank}
    validate_record(record, subsets.keys())
    diagnostics = {"B": enc["B"], "J": enc["J"], "ldj": enc["ldj"], "log_pz": enc["log_pz"],
                   "tau": tail["tau"], "q_tail": tail["q_tail"], "n_tail": tail["n_tail"],
                   "roundtrip_max_abs": float(np.max(np.abs(z_re - bank))),
                   "mean_J_bank": float(j_bank.mean()), "events": events,
                   "bank_rows": int(bank.shape[0]), "subset_labels": list(subsets.keys())}
    return record, diagnostics
