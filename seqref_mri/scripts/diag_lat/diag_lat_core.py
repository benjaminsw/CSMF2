# SEQREF-DLAT-CORE v0.1 -- diag_lat_core.py: D-LAT per slice-state measures (pure NumPy)
# LIFETIME: KEEP
"""
STATUS: PRE-FREEZE ENGINEERING; NOT ACTIVATED (236 LE19). Written against the mini-spec
candidate v0.5.2 (sha256 a09c8f5e...8eda6), which is not frozen. Provisional until freeze.

Scope: mathematics only. No I/O, no model, no checkpoint, no MRI file, no random draw.
Imports: NumPy and the standard library only (no SciPy: normal CDF is math.erf).
All measures are computed in float64 from the arrays passed in (assumption A1).

No fallback, placeholder, mock or dummy return: every failure path logs an error and
raises DiagLatError with a typed code. A measure that cannot be computed in a valid
degenerate case is the explicit status NOT_COMPUTED with a recorded event, never NaN/0.

Stability subsets (ported unchanged from diag_unc_core.py v0.2, sha256 a0d3b4a4...8b06):
FOUR labels but only THREE distinct sample subsets -- "pre64" and "halfA" are the same
first 64 samples. They must not be read as independent confirmations.

Changelog
- v0.1 (2026-11-21): first version. L1 (KS, q_z, moments, tail set, projected covariance),
  L2 (secant gains), L3 (d01, a01, error terms), L4 (TC_G plug-in, PR), L5 arithmetic
  (swap penalty/shift, noise scaling, sensitivities), physical complements, section 3.1
  edge cases with typed codes, projection-matrix construction and array identity.
- v0.1, found by fixture T16 before release: zero variance is decided exactly (all K values
  of a coordinate equal), not by sd == 0 alone, which rounding of the mean can defeat.
"""
import hashlib
import logging
import math

import numpy as np

__version__ = "0.1"
ABBR = "SEQREF-DLAT-CORE"
log = logging.getLogger("diag_lat.core")

# ---- registered constants (candidate v0.5.2) --------------------------------------
D_FREE = 13824
K_BANK = 128
SPLINE_B = 5.159583485556914      # B_s: bound of the pinned spline (not the NLL term B)
M_PROJ = 8
EPSILON = 0.01
N_VOLUMES = 97
N_ACQUIRED = 2304
HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
NOT_COMPUTED = "NOT_COMPUTED"

# Ported from diag_unc_core.py v0.2 (line 76). pre64 == halfA: three distinct subsets.
STABILITY_SUBSETS = {
    "full": np.arange(128),
    "pre64": np.arange(64),
    "halfA": np.arange(0, 64),
    "halfB": np.arange(64, 128),
}

# ---- typed codes -------------------------------------------------------------------
NONFINITE = "DIAG_LAT_NONFINITE"
ZERO_DENOMINATOR = "DIAG_LAT_ZERO_DENOMINATOR"
UNDEFINED_STATISTIC = "DIAG_LAT_UNDEFINED_STATISTIC"      # non-fatal event
TAIL_INVARIANCE = "DIAG_LAT_TAIL_INVARIANCE"
SWAP_FIXTURE = "DIAG_LAT_SWAP_FIXTURE"
SWAP_IDENTITY = "DIAG_LAT_SWAP_IDENTITY"
INPUT = "DIAG_LAT_INPUT"


class DiagLatError(Exception):
    """Typed stop. Logged at construction; never swallowed by this module."""

    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        log.error("%s: %s", code, message)


def _event(events, code, measure, detail):
    ev = {"code": code, "measure": measure, "detail": detail}
    events.append(ev)
    log.warning("event %s [%s]: %s", code, measure, detail)


def _f64(name, a, ndim=None):
    a = np.asarray(a)
    if a.dtype.kind not in "fiu":
        raise DiagLatError(INPUT, f"{name}: dtype {a.dtype} is not real numeric")
    a = a.astype(np.float64, copy=False)
    if ndim is not None and a.ndim != ndim:
        raise DiagLatError(INPUT, f"{name}: expected {ndim} dimensions, got shape {a.shape}")
    if a.size == 0:
        raise DiagLatError(INPUT, f"{name}: empty array")
    if not np.all(np.isfinite(a)):
        raise DiagLatError(NONFINITE, f"{name}: {int((~np.isfinite(a)).sum())} non-finite value(s)")
    return a


def _finite_scalar(name, v):
    v = float(v)
    if not math.isfinite(v):
        raise DiagLatError(NONFINITE, f"{name} is not finite ({v!r})")
    return v


def _same(name_a, a, name_b, b):
    if a.shape != b.shape:
        raise DiagLatError(INPUT, f"shape mismatch: {name_a} {a.shape} vs {name_b} {b.shape}")


# ---- normal CDF (standard library) -------------------------------------------------
_erf = np.frompyfunc(math.erf, 1, 1)


def norm_cdf(x):
    """Standard normal CDF, 0.5 * (1 + erf(x / sqrt(2))), element-wise via math.erf."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.asarray(_erf(x / math.sqrt(2.0)), dtype=np.float64))


# ---- L1 ----------------------------------------------------------------------------
def ks_normal(x):
    """Exact two-sided Kolmogorov-Smirnov distance of the pooled entries of x to N(0,1)."""
    x = np.sort(_f64("ks input", x, 1))
    n = x.size
    f = norm_cdf(x)
    i = np.arange(1, n + 1, dtype=np.float64)
    return _finite_scalar("KS", max(np.max(i / n - f), np.max(f - (i - 1.0) / n)))


def pooled_moments(x):
    """Skewness and excess kurtosis as plain moment ratios (divide by n; no correction)."""
    x = _f64("moment input", x, 1)
    c = x - x.mean()
    m2 = float(np.mean(c * c))
    if m2 == 0.0:
        raise DiagLatError(ZERO_DENOMINATOR, "pooled moments: zero variance")
    skew = float(np.mean(c ** 3)) / m2 ** 1.5
    exkurt = float(np.mean(c ** 4)) / m2 ** 2 - 3.0
    return _finite_scalar("skewness", skew), _finite_scalar("excess kurtosis", exkurt)


def q_of(z):
    z = _f64("z", z, 1)
    return float(np.dot(z, z)) / z.size


def nll_prior_per_dim(q_z):
    """B = 0.5 * q_z + 0.5 * ln(2 pi) (per dimension)."""
    return 0.5 * _finite_scalar("q_z", q_z) + HALF_LOG_2PI


def l1_measures(z_star):
    z = _f64("z_star", z_star, 1)
    skew, exkurt = pooled_moments(z)
    q_z = q_of(z)
    return {"KS": ks_normal(z), "q_z": q_z, "B": nll_prior_per_dim(q_z),
            "skewness": skew, "excess_kurtosis": exkurt}


def tail_set(u_std_true_f32, b_s=SPLINE_B):
    """Tail set of the standardised target: |entry| > B_s, compared in float32 (the vector
    passed to the flow). Returns (boolean mask, tau, q_tail)."""
    u = np.asarray(u_std_true_f32)
    if u.dtype != np.float32 or u.ndim != 1:
        raise DiagLatError(INPUT, f"tail_set needs the 1-D float32 target, got {u.dtype} {u.shape}")
    if not np.all(np.isfinite(u)):
        raise DiagLatError(NONFINITE, "tail_set: non-finite target")
    mask = np.abs(u) > np.float32(b_s)
    u64 = u.astype(np.float64)
    tau = float(mask.sum()) / u.size
    q_tail = float(np.sum(u64[mask] ** 2)) / u.size
    return mask, tau, q_tail


def check_tail_invariance(z_star_f32, u_std_true_f32, mask):
    """On the tail set z* must equal the input exactly (pinned spline is the identity there)."""
    z = np.asarray(z_star_f32)
    u = np.asarray(u_std_true_f32)
    if z.dtype != np.float32 or u.dtype != np.float32:
        raise DiagLatError(INPUT, "tail invariance is checked on the float32 vectors")
    _same("z_star", z, "u_std_true", u)
    bad = int(np.count_nonzero(z[mask] != u[mask]))
    if bad:
        raise DiagLatError(TAIL_INVARIANCE, f"{bad} tail entr(y/ies) of z* differ from the input")
    return int(np.count_nonzero(mask))


def projection_matrix(raw):
    """Orthonormal projection from the registered raw draw: reduced QR in float64, sign of
    each column fixed so that the diagonal of R is positive."""
    raw = _f64("projection raw draw", raw, 2)
    if raw.shape[0] < raw.shape[1]:
        raise DiagLatError(INPUT, f"projection needs rows >= columns, got {raw.shape}")
    q, r = np.linalg.qr(raw, mode="reduced")
    diag = np.diag(r)
    if np.any(diag == 0.0):
        raise DiagLatError(ZERO_DENOMINATOR, "projection: rank-deficient draw (zero on diag R)")
    q = q * np.sign(diag)[None, :]
    return np.ascontiguousarray(q)


def array_identity(a):
    """sha256 of the little-endian C-order bytes, with shape and dtype (223 section 3.4 rule)."""
    a = np.ascontiguousarray(a)
    le = a.astype(a.dtype.newbyteorder("<"), copy=False)
    return {"sha256": hashlib.sha256(le.tobytes(order="C")).hexdigest(),
            "shape": list(a.shape), "dtype": le.dtype.str, "numpy": np.__version__}


def projected_cov_mismatch(z_stars, proj):
    """E_cov = ||S - I||_F / ||I||_F for the projected truth codes across slices
    (sample covariance, mean removed, N - 1 denominator). Descriptive only."""
    z = _f64("z_stars", z_stars, 2)
    p = _f64("projection", proj, 2)
    if z.shape[1] != p.shape[0]:
        raise DiagLatError(INPUT, f"projection {p.shape} does not match codes {z.shape}")
    n, m = z.shape[0], p.shape[1]
    if n < 2:
        raise DiagLatError(INPUT, "projected covariance needs at least 2 slices")
    y = z @ p
    c = y - y.mean(axis=0, keepdims=True)
    s = c.T @ c / (n - 1)
    return _finite_scalar("E_cov", np.linalg.norm(s - np.eye(m)) / math.sqrt(m))


def e_cov_reference(n, m):
    """Analytic reference for E_cov under exact N(0, I): root of E||S - I||_F^2 / m
    = sqrt((m + 1) / (n - 1)). ANALYTIC, not a draw (no unregistered random draw)."""
    if n < 2 or m < 1:
        raise DiagLatError(INPUT, f"e_cov_reference: n={n}, m={m}")
    return math.sqrt((m + 1.0) / (n - 1.0))


# ---- coordinates -------------------------------------------------------------------
def to_physical(u_std, scale, mean):
    """u = s (.) u_std + m, per coordinate; scale must be positive."""
    u = _f64("u_std", u_std)
    s = _f64("scale", scale, 1)
    m = _f64("mean", mean, 1)
    if u.shape[-1] != s.size or s.size != m.size:
        raise DiagLatError(INPUT, f"to_physical: shapes {u.shape}, {s.shape}, {m.shape}")
    if np.any(s <= 0.0):
        raise DiagLatError(INPUT, "to_physical: non-positive scale")
    return u * s + m


# ---- L2 ----------------------------------------------------------------------------
def l2_measures(u_true, r0, z_star, u_bank, z_bank):
    """Secant displacements. gamma_truth = ||u_true - R0|| / ||z*||;
    gamma_bank = mean_k ||u_k - R0|| / ||z_k||. Any zero norm stops."""
    u_true = _f64("u_true", u_true, 1); r0 = _f64("R0", r0, 1); z_star = _f64("z_star", z_star, 1)
    u_bank = _f64("u_bank", u_bank, 2); z_bank = _f64("z_bank", z_bank, 2)
    _same("u_true", u_true, "R0", r0); _same("u_bank", u_bank, "z_bank", z_bank)
    num_t = float(np.linalg.norm(u_true - r0)); den_t = float(np.linalg.norm(z_star))
    num_b = np.linalg.norm(u_bank - r0[None, :], axis=1); den_b = np.linalg.norm(z_bank, axis=1)
    if den_t == 0.0:
        raise DiagLatError(ZERO_DENOMINATOR, "L2: ||z*|| = 0")
    if num_t == 0.0:
        raise DiagLatError(ZERO_DENOMINATOR, "L2: ||u_true - R0|| = 0")
    if np.any(den_b == 0.0):
        raise DiagLatError(ZERO_DENOMINATOR, f"L2: ||z_k|| = 0 for {int((den_b == 0).sum())} sample(s)")
    g_t = num_t / den_t
    g_b = float(np.mean(num_b / den_b))
    if g_b == 0.0:
        raise DiagLatError(ZERO_DENOMINATOR, "L2: gamma_bank = 0 (all samples equal R0)")
    out = {"log_gamma_truth": math.log(g_t), "log_gamma_bank": math.log(g_b),
           "log_gamma_ratio": math.log(g_t) - math.log(g_b),
           "log_rmse_r0": math.log(num_t / math.sqrt(u_true.size))}
    for k, v in out.items():
        _finite_scalar(k, v)
    return out


# ---- L3 ----------------------------------------------------------------------------
def l3_measures(u_true, r0, u_bank, events):
    """d01 = ||mu - R0|| / sqrt(tr Sigma); a01 = cos(mu - R0, u_true - R0)."""
    u_true = _f64("u_true", u_true, 1); r0 = _f64("R0", r0, 1); u_bank = _f64("u_bank", u_bank, 2)
    _same("u_true", u_true, "R0", r0)
    k, d = u_bank.shape
    if k < 2 or d != u_true.size:
        raise DiagLatError(INPUT, f"L3: bank shape {u_bank.shape}")
    mu = u_bank.mean(axis=0)
    c = u_bank - mu[None, :]
    tr = float(np.sum(c * c)) / (k - 1)
    # exact criterion: identical samples have tr Sigma = 0 whatever rounding the mean introduces
    if tr == 0.0 or bool(np.all(u_bank == u_bank[0][None, :])):
        raise DiagLatError(ZERO_DENOMINATOR, "L3a: tr Sigma = 0 (all samples identical)")
    a = mu - r0; b = u_true - r0
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    out = {"d01": na / math.sqrt(tr), "e0_sq": nb * nb / d,
           "e1_sq": float(np.sum((u_true - mu) ** 2)) / d, "tr_sigma_per_dim": tr / d}
    out["e0_sq_minus_e1_sq"] = out["e0_sq"] - out["e1_sq"]
    if na == 0.0:
        out["a01"] = NOT_COMPUTED
        _event(events, UNDEFINED_STATISTIC, "a01", "||mu - R0|| = 0: d01 = 0 reported, a01 not computed")
    elif nb == 0.0:
        raise DiagLatError(ZERO_DENOMINATOR, "L3b: ||u_true - R0|| = 0")
    else:
        out["a01"] = float(np.dot(a, b)) / (na * nb)
    for kk, v in out.items():
        if v is not NOT_COMPUTED:
            _finite_scalar(kk, v)
    return out


# ---- L4 ----------------------------------------------------------------------------
def participation_ratio(u_bank):
    """PR = (tr Sigma)^2 / tr(Sigma^2), from the K x K Gram matrix (never d x d)."""
    u_bank = _f64("u_bank", u_bank, 2)
    k = u_bank.shape[0]
    if k < 2:
        raise DiagLatError(INPUT, "PR needs at least 2 samples")
    c = u_bank - u_bank.mean(axis=0, keepdims=True)
    g = c @ c.T / (k - 1)
    tr = float(np.trace(g)); tr2 = float(np.sum(g * g))
    if not (math.isfinite(tr) and tr > 0.0):
        raise DiagLatError(ZERO_DENOMINATOR, f"PR: tr Sigma = {tr!r}")
    return _finite_scalar("PR", tr * tr / tr2)


def pr_isotropic_reference(k, d):
    """Analytic approximation of PR for isotropic Gaussian samples (K - 1 denominator):
    n d / (n + d + 1), n = K - 1. ANALYTIC, not a draw."""
    n = k - 1
    if n < 1 or d < 1:
        raise DiagLatError(INPUT, f"pr_isotropic_reference: K={k}, d={d}")
    return n * d / (n + d + 1.0)


def l4_measures(u_bank, j_bank, events):
    """TC_G = mean_j log sigma_j - mean_k J(u_k): PLUG-IN Gaussian-marginal proxy. Not an
    upper bound at finite K; can be negative; never truncated. J(u_k) is an input
    (re-encoding belongs to the adapter). TC_G is unchanged by a diagonal affine map."""
    u_bank = _f64("u_bank", u_bank, 2); j = _f64("J bank", j_bank, 1)
    k = u_bank.shape[0]
    if j.size != k or k < 2:
        raise DiagLatError(INPUT, f"L4: bank {u_bank.shape}, J {j.shape}")
    sd = u_bank.std(axis=0, ddof=1)
    # exact criterion: a coordinate whose K values are all equal has sigma_j = 0, whatever
    # rounding the mean introduces (no epsilon); an underflowed SD of exactly 0 counts too
    n_zero = int(np.count_nonzero((u_bank.max(axis=0) == u_bank.min(axis=0)) | (sd == 0.0)))
    out = {"PR": participation_ratio(u_bank), "mean_J": float(j.mean())}
    if n_zero:
        out["TC_G"] = NOT_COMPUTED
        out["mean_log_sigma"] = NOT_COMPUTED
        _event(events, UNDEFINED_STATISTIC, "TC_G",
               f"sigma_j = 0 for {n_zero} of {sd.size} coordinate(s): TC_G not computed")
    else:
        mls = float(np.mean(np.log(sd)))
        out["mean_log_sigma"] = _finite_scalar("mean log sigma", mls)
        out["TC_G"] = _finite_scalar("TC_G", mls - out["mean_J"])
    return out


# ---- bank-dependent measures on the stability subsets --------------------------------
def bank_measures(u_true, r0, z_star, u_bank, z_bank, j_bank, events, subsets=None):
    """L2, L3, L4 on each stability subset. Returns {label: {...}}. Events carry the label."""
    subsets = STABILITY_SUBSETS if subsets is None else subsets
    u_bank = np.asarray(u_bank); z_bank = np.asarray(z_bank); j_bank = np.asarray(j_bank)
    out = {}
    for label, idx in subsets.items():
        if idx.max() >= u_bank.shape[0]:
            raise DiagLatError(INPUT, f"subset {label} exceeds the bank ({u_bank.shape[0]} samples)")
        ev = []
        res = {}
        res.update(l2_measures(u_true, r0, z_star, u_bank[idx], z_bank[idx]))
        res.update(l3_measures(u_true, r0, u_bank[idx], ev))
        res.update(l4_measures(u_bank[idx], j_bank[idx], ev))
        for e in ev:
            e["subset"] = label
        events.extend(ev)
        out[label] = res
    return out


# ---- L5 ----------------------------------------------------------------------------
def swap_penalty(nll_swap, nll_own):
    """P_swap = NLL(u_i | y~_j) - NLL(u_i | y_i), per dimension."""
    return _finite_scalar("NLL swap", nll_swap) - _finite_scalar("NLL own", nll_own)


def shift_norm(a, b):
    """Euclidean norm of a - b (swap shift of R0; either coordinate space)."""
    a = _f64("a", a, 1); b = _f64("b", b, 1)
    _same("a", a, "b", b)
    return float(np.linalg.norm(a - b))


def check_swap_identity(cond_swap, cond_own):
    """S4: the swapped conditioning input must differ from the recipient's own."""
    a = np.asarray(cond_swap); b = np.asarray(cond_own)
    _same("cond_swap", a, "cond_own", b)
    if np.array_equal(a, b):
        raise DiagLatError(SWAP_IDENTITY, "swapped input is identical to the recipient's own")


def swap_tolerance_scale(acquired_coefficients, rel=1e-5):
    """S2 tolerance = rel x largest acquired coefficient magnitude; a zero scale stops."""
    a = np.asarray(acquired_coefficients)
    if not np.all(np.isfinite(a)):
        raise DiagLatError(NONFINITE, "swap fixture: non-finite acquired coefficient")
    scale = float(np.max(np.abs(a))) if a.size else 0.0
    if scale == 0.0:
        raise DiagLatError(SWAP_FIXTURE, "largest acquired coefficient is 0: no zero tolerance is used")
    return rel * scale


def _rms_complex(c):
    return math.sqrt(float(np.mean(c.real ** 2 + c.imag ** 2)))


def scale_noise(noise_ri, y_acquired, eps=EPSILON):
    """Rescale one slice's registered draw (n_cells, 2) so that its REALISED RMS over the
    acquired cells equals eps x RMS(y_i) exactly. Returns the complex noise (n_cells,)."""
    n = np.asarray(noise_ri); y = np.asarray(y_acquired)
    if n.ndim != 2 or n.shape[1] != 2 or y.ndim != 1 or y.size != n.shape[0]:
        raise DiagLatError(INPUT, f"scale_noise: noise {n.shape}, y {y.shape}")
    if not np.iscomplexobj(y):
        raise DiagLatError(INPUT, "scale_noise: y must be complex")
    if not (np.all(np.isfinite(n)) and np.all(np.isfinite(y.real)) and np.all(np.isfinite(y.imag))):
        raise DiagLatError(NONFINITE, "L5(b): non-finite value in the draw or in y_i")
    c = n[:, 0].astype(np.float64) + 1j * n[:, 1].astype(np.float64)
    rms_n = _rms_complex(c); rms_y = _rms_complex(y.astype(np.complex128))
    if rms_n == 0.0:
        raise DiagLatError(ZERO_DENOMINATOR, "L5(b): realised RMS of the unscaled noise = 0")
    if rms_y == 0.0:
        raise DiagLatError(ZERO_DENOMINATOR, "L5(b): RMS(y_i) = 0")
    out = c * (eps * rms_y / rms_n)
    if not (np.all(np.isfinite(out.real)) and np.all(np.isfinite(out.imag))):
        raise DiagLatError(NONFINITE, "L5(b): non-finite scaled noise")
    return out


def l5b_sensitivities(r0, r0_pert, z_star, z_pert, nll, nll_pert, eps=EPSILON):
    """RMS change of R0, RMS change of z*, change of NLL, each divided by eps."""
    if not (math.isfinite(eps) and eps > 0.0):
        raise DiagLatError(INPUT, f"eps = {eps!r}")
    r0 = _f64("R0", r0, 1); z = _f64("z_star", z_star, 1)
    return {"dR0_rms_per_eps": shift_norm(r0_pert, r0) / math.sqrt(r0.size) / eps,
            "dz_rms_per_eps": shift_norm(z_pert, z) / math.sqrt(z.size) / eps,
            "dNLL_per_eps": (_finite_scalar("NLL pert", nll_pert) - _finite_scalar("NLL", nll)) / eps}
