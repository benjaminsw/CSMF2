# SEQREF-MRI-DIAGJSHCORE v0.2.1 -- DIAG-JSH scientific core (D-JAC, D-SPEC, D-HET rules of mini-spec v0.5)
# LIFETIME: KEEP
# =============================================================================
# Pure NumPy implementation of the frozen scientific rules of
# SEQREF-MRI-DIAG-JSH v0.4 (no model, data or V05LT import). The real-data
# adapter (DIAG-I2: pinned states, R0 decode, row loading) feeds arrays into
# these functions. Every refusal is a logged DiagError(code); no fallback,
# no placeholder, no silent substitution.
# Sections: constants (§1.2, §1.4, §3, §4) · identities (§1.3, §3) ·
#   spectral grid and cells (§4) · aggregation and thirds (§1.4, §5) ·
#   bootstrap draws and estimators (§1.4) · Spearman (§1.4) · labels (§2, §3).
# Changelog (v0.2, 2026-11-09; review of v0.1):
#   * input validation before any aggregation: matrices finite, 2-D, equal
#     shapes, SEEDS rows; draws int64 (b, V) with indices in range; finite
#     outputs required; Spearman refuses non-finite inputs;
#   * check_replay refuses empty populations and checks the expected count;
#   * check_nll_identity requires a positive integer d and a non-negative
#     stored deviation;
#   * subgroup_estimates requires exactly three nonempty, disjoint groups
#     covering 0..V-1 and matching per-third draws.
# Changelog (v0.1, 2026-11-09):
#   * first implementation of mini-spec v0.4 rules; synthetic tests in
#     test_diag_jsh_core.py.
# =============================================================================
from __future__ import annotations

import logging
import math
from statistics import fmean

import numpy as np

__abbr__ = "SEQREF-MRI-DIAGJSHCORE"
__version__ = "0.2.1"
logger = logging.getLogger("seqref_diag.jsh_core")

# -- frozen constants (mini-spec v0.4) ----------------------------------------
DIAG_SEED = 20261109            # §1.4
SEEDS = (2026, 2027, 2028)      # §1.1 (rows of every seed x volume matrix)
B = 10_000                      # §1.4
LEVELS = (2.5, 97.5)            # 95% percentile interval, linear (§1.4)
PRACTICAL_DB = 0.5              # registered (v05lt_stats, PLAN 194 §9.2)
DENSITY_BOUND = 0.01            # registered (+1 % NLL ratio)
NLL_TOL_REL = 1e-6              # registered NLL_DECOMP_TOL_REL (total)
CONTRIB_MIN = 1e-3              # nats/dim (§3)
PSNR_REPLAY_TOL = 1e-3          # dB, registered P2 limit (§1.3)
NLL_REPLAY_TOL = 1e-4           # nats/dim, registered P2 limit (§1.3)
SPEC_REL, SPEC_ABS = 1e-9, 1e-24  # §4 identity tolerance
UNDEF_MAX = 0.01                # §1.4 undefined-correlation fraction
N = 96                          # registered CELL_HW
HALF = N // 2                   # 48
# band limits on r^2 = kx^2 + ky^2: rho = r / (48*sqrt2); rho < 1/3 <=>
# r^2 < 4608/9 = 512; rho < 2/3 <=> r^2 < 4*4608/9 = 2048 (exact integers)
R2_LOW, R2_MID = 512, 2048
BANDS = ("low", "mid", "high")
SPLITS = ("acquired", "unacquired")


class DiagError(RuntimeError):
    def __init__(self, code: str, msg: str):
        super().__init__(f"{code}: {msg}")
        self.code = code


def fail(code: str, msg: str):
    logger.error("[%s] %s: %s", __abbr__, code, msg)
    raise DiagError(code, msg)


def require(cond: bool, code: str, msg: str) -> None:
    if not cond:
        fail(code, msg)


# -- identities (§1.3, §3) ----------------------------------------------------
def check_nll_identity(row: dict, d: int) -> None:
    """Stored per-slice components: deviation and per-dim recomposition."""
    require(isinstance(d, int) and not isinstance(d, bool) and d > 0,
            "DIAG_DIM_INVALID", f"d_free {d!r}")
    for k in ("nll_per_dim", "nll_prior_per_dim", "nll_logdet_per_dim",
              "nll_decomp_abs_dev"):
        v = row.get(k)
        require(isinstance(v, float) and math.isfinite(v), "DIAG_NLL_FIELD",
                f"{k}={v!r} ({row.get('file')}|{row.get('slice_index')})")
    require(row["nll_decomp_abs_dev"] >= 0.0, "DIAG_NLL_FIELD",
            f"negative stored deviation {row['nll_decomp_abs_dev']}")
    tot = NLL_TOL_REL * max(1.0, d * abs(row["nll_per_dim"]))
    require(row["nll_decomp_abs_dev"] <= tot, "DIAG_NLL_IDENTITY",
            f"stored deviation {row['nll_decomp_abs_dev']} > {tot}")
    rec = row["nll_prior_per_dim"] + row["nll_logdet_per_dim"]
    require(abs(row["nll_per_dim"] - rec) <= tot / d + 1e-12,
            "DIAG_NLL_IDENTITY", f"|nll - (B + J)| = "
            f"{abs(row['nll_per_dim'] - rec)} ({row.get('file')}|"
            f"{row.get('slice_index')})")


def check_replay(stored: dict, replay: dict, expected: int) -> dict:
    """§1.3: exact identities, then the frozen numerical tolerances.
    stored / replay: {key: {"ckpt", "mask", "psnr", "nll_per_dim"}} with
    key = (volume_id, file, slice_index) in validation order; expected =
    the registered population size."""
    require(isinstance(expected, int) and expected > 0, "DIAG_POPULATION_EMPTY",
            f"expected count {expected!r}")
    require(len(stored) == expected and len(replay) == expected,
            "DIAG_POPULATION_MISMATCH", f"{len(stored)} stored / "
            f"{len(replay)} replayed != expected {expected}")
    require(list(stored) == list(replay), "DIAG_POPULATION_MISMATCH",
            "case identities or order differ")
    dp = dn = 0.0
    bit = 0
    for k, s in stored.items():
        r = replay[k]
        require(r["ckpt"] == s["ckpt"], "DIAG_REPRODUCTION_MISMATCH",
                f"{k}: checkpoint identity")
        require(r["mask"] == s["mask"], "DIAG_REPRODUCTION_MISMATCH",
                f"{k}: mask identity")
        for v in (r["psnr"], s["psnr"], r["nll_per_dim"], s["nll_per_dim"]):
            require(isinstance(v, float) and math.isfinite(v),
                    "DIAG_NON_FINITE", f"{k}: {v!r}")
        ep, en = abs(r["psnr"] - s["psnr"]), abs(r["nll_per_dim"] -
                                                  s["nll_per_dim"])
        require(ep <= PSNR_REPLAY_TOL and en <= NLL_REPLAY_TOL,
                "DIAG_REPRODUCTION_MISMATCH",
                f"{k}: |dPSNR| {ep} dB, |dNLL| {en} nats/dim")
        dp, dn = max(dp, ep), max(dn, en)
        bit += int(ep == 0.0 and en == 0.0)
    return {"max_abs_psnr_dev": dp, "max_abs_nll_dev": dn,
            "bitwise_equal_slices": bit, "slices": len(stored)}


# -- spectral grid and cells (§4) ---------------------------------------------
def band_index(n: int = N) -> np.ndarray:
    require(n == N, "DIAG_GRID", f"cell {n} != registered {N}")
    k = np.arange(n) - HALF
    r2 = k[:, None] ** 2 + k[None, :] ** 2          # rows = ky, cols = kx
    return np.where(r2 < R2_LOW, 0, np.where(r2 < R2_MID, 1, 2))


def fft2c(x: np.ndarray) -> np.ndarray:
    """Registered centred orthonormal FFT (fastmri_data.fft2c), complex128."""
    x = np.asarray(x, dtype=np.complex128)
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)),
                                       norm="ortho"), axes=(-2, -1))


def _cells(energy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bi = band_index()
    acq = np.broadcast_to(np.asarray(mask, dtype=bool)[None, :], (N, N))
    out = np.zeros((3, 2), dtype=np.float64)
    for b in range(3):
        out[b, 0] = float(np.sum(energy[(bi == b) & acq], dtype=np.float64))
        out[b, 1] = float(np.sum(energy[(bi == b) & ~acq], dtype=np.float64))
    return out


def _identity(cells: np.ndarray, full: float, what: str) -> None:
    require(abs(float(cells.sum()) - full) <= SPEC_REL * full + SPEC_ABS,
            "DIAG_SPECTRAL_IDENTITY", f"{what}: cells {cells.sum()} vs "
            f"full {full}")


def spectral_record(x_hat: np.ndarray, x_true: np.ndarray,
                    mask: np.ndarray) -> dict:
    """S-C (complex) and S-M (magnitude) cell energies for one slice.
    x_hat, x_true: complex (N, N); mask: (N,) 0/1 over columns (kx)."""
    x_hat = np.asarray(x_hat, dtype=np.complex128)
    x_true = np.asarray(x_true, dtype=np.complex128)
    require(x_hat.shape == (N, N) and x_true.shape == (N, N),
            "DIAG_GRID", f"shapes {x_hat.shape}, {x_true.shape}")
    m = np.asarray(mask)
    require(m.shape == (N,) and set(np.unique(m)).issubset({0, 1}),
            "DIAG_MASK", f"mask shape {m.shape}")
    require(bool(np.isfinite(x_hat).all() and np.isfinite(x_true).all()),
            "DIAG_NON_FINITE", "image")
    ref = float(np.sum(np.abs(fft2c(x_true)) ** 2, dtype=np.float64))
    require(ref > 0.0, "DIAG_ZERO_REFERENCE", "zero reference energy")
    ec = np.abs(fft2c(x_hat) - fft2c(x_true)) ** 2
    cc, fc = _cells(ec, m), float(np.sum(ec, dtype=np.float64))
    _identity(cc, fc, "S-C")
    em = np.abs(fft2c(np.abs(x_hat) - np.abs(x_true))) ** 2
    cm, fm = _cells(em, m), float(np.sum(em, dtype=np.float64))
    _identity(cm, fm, "S-M")
    return {"sc_cells": cc, "sc_full": fc, "sc_contrib": cc / ref,
            "sm_cells": cm, "sm_full": fm,
            "sm_pixel_mse": fm / (N * N), "ref_energy": ref}


def hf_fraction(x_true: np.ndarray) -> float:
    """H-F per slice: high-band share of reference Fourier energy."""
    e = np.abs(fft2c(x_true)) ** 2
    tot = float(np.sum(e, dtype=np.float64))
    require(tot > 0.0, "DIAG_ZERO_REFERENCE", "zero reference energy")
    return float(np.sum(e[band_index() == 2], dtype=np.float64)) / tot


# -- aggregation and thirds (§1.4, §5) ----------------------------------------
def volume_means(values: dict, entries: list) -> dict:
    """Registered weighting (v05lt_eligibility.volume_means, all slices)."""
    acc: dict = {}
    for vid, f, s in entries:
        v = values[(f, s)]
        require(isinstance(v, float) and math.isfinite(v),
                "DIAG_NON_FINITE", f"{f}|{s}: {v!r}")
        acc.setdefault(vid, []).append(v)
    return {vid: float(np.mean(v)) for vid, v in acc.items()}


def thirds(values: dict, *, descending: bool) -> list:
    """Three groups of volume IDs, first = hardest (H-D: ascending ZF PSNR)
    or highest (H-F: descending HF fraction); ties by volume ID ascending;
    sizes floor(V/3) + (1 if r < V mod 3)."""
    for vid, v in values.items():
        require(isinstance(v, float) and math.isfinite(v),
                "DIAG_NON_FINITE", f"stratifier {vid}: {v!r}")
    V = len(values)
    require(V >= 3, "DIAG_THIRDS", f"{V} volumes")
    order = sorted(values, key=lambda k: ((-values[k] if descending
                                           else values[k]), k))
    base, r = divmod(V, 3)
    sizes = [base + (1 if i < r else 0) for i in range(3)]
    out, i = [], 0
    for n in sizes:
        out.append(order[i:i + n])
        i += n
    return out


# -- bootstrap draws and estimators (§1.4) ------------------------------------
def make_draws(V: int, strata: list, *, b: int = B,
               seed: int = DIAG_SEED) -> dict:
    """Generation order: v (b x V); then, per stratifier in the given order,
    one (b x n_t) array per third in third order. strata: [thirds_sizes]."""
    rng = np.random.default_rng(seed)
    v = rng.integers(0, V, size=(b, V), dtype=np.int64)
    st = []
    for sizes in strata:
        st.append([rng.integers(0, n, size=(b, n), dtype=np.int64)
                   for n in sizes])
    return {"v": v, "strata": st}


def _matrix(X, name: str) -> np.ndarray:
    X = np.asarray(X)
    require(X.ndim == 2 and X.shape[0] == len(SEEDS) and X.shape[1] >= 1,
            "DIAG_SHAPE", f"{name} shape {X.shape} != ({len(SEEDS)}, V)")
    require(np.issubdtype(X.dtype, np.number) and
            not np.iscomplexobj(X), "DIAG_SHAPE", f"{name} dtype {X.dtype}")
    X = X.astype(np.float64)
    require(bool(np.isfinite(X).all()), "DIAG_NON_FINITE",
            f"{name}: {int((~np.isfinite(X)).sum())} non-finite values")
    return X


def _pair(A, R) -> tuple:
    A, R = _matrix(A, "A"), _matrix(R, "R")
    require(A.shape == R.shape, "DIAG_SHAPE", f"{A.shape} vs {R.shape}")
    return A, R


def _draws(v, V: int, name: str = "v") -> np.ndarray:
    v = np.asarray(v)
    require(v.ndim == 2 and v.shape[0] >= 1 and v.shape[1] == V and
            v.dtype == np.int64, "DIAG_DRAWS", f"{name} {v.shape} {v.dtype} "
            f"for V={V}")
    require(int(v.min()) >= 0 and int(v.max()) < V, "DIAG_DRAWS",
            f"{name} indices outside 0..{V - 1}")
    return v


def _finite_out(x: float, what: str) -> float:
    require(math.isfinite(x), "DIAG_NON_FINITE", f"{what} = {x!r}")
    return x


def _pct(x: np.ndarray) -> tuple:
    x = np.asarray(x, dtype=np.float64)
    require(x.size >= 1 and bool(np.isfinite(x).all()), "DIAG_NON_FINITE",
            "bootstrap replicates")
    lo, hi = np.percentile(x, LEVELS, method="linear")
    return float(lo), float(hi)


def delta_estimate(A: np.ndarray, R: np.ndarray, v: np.ndarray) -> dict:
    """A, R: (seeds, V) volume means. Point = seed mean of volume-mean
    paired differences; interval from the shared volume draws."""
    A, R = _pair(A, R)
    v = _draws(v, A.shape[1])
    D = A - R
    point = _finite_out(fmean(D.ravel().tolist()), "delta estimate")
    boot = D[:, v].mean(axis=2).mean(axis=0)        # (b,)
    lo, hi = _pct(boot)
    return {"estimate": point, "ci": (lo, hi),
            "per_seed": [fmean(D[k].tolist()) for k in range(D.shape[0])]}


def ratio_estimate(A: np.ndarray, R: np.ndarray, v: np.ndarray) -> dict:
    """Registered NLL ratio r = (mean A - mean R) / |mean R| (ratio of
    means); refused when the reference mean is exactly 0."""
    A, R = _pair(A, R)
    v = _draws(v, A.shape[1])
    ra = fmean(R.ravel().tolist())
    require(ra != 0.0, "DIAG_RATIO_UNDEFINED", "reference mean NLL is 0")
    point = (fmean(A.ravel().tolist()) - ra) / abs(ra)
    ma = A[:, v].mean(axis=(0, 2))
    mr = R[:, v].mean(axis=(0, 2))
    require(bool((mr != 0.0).all()), "DIAG_RATIO_UNDEFINED",
            "reference mean NLL is 0 in a draw")
    lo, hi = _pct((ma - mr) / np.abs(mr))
    return {"estimate": point, "ci": (lo, hi)}


def subgroup_estimates(A: np.ndarray, R: np.ndarray, groups: list,
                       arrays: list) -> dict:
    """groups: three lists of volume positions (fixed membership);
    arrays: the stratifier's per-third draws. Returns per-third estimates
    and the first-minus-last contrast."""
    A, R = _pair(A, R)
    V = A.shape[1]
    require(isinstance(groups, (list, tuple)) and len(groups) == 3 and
            isinstance(arrays, (list, tuple)) and len(arrays) == 3,
            "DIAG_STRATA", "exactly three groups and three draw arrays")
    flat = []
    for g in groups:
        require(len(g) >= 1 and all(isinstance(i, (int, np.integer)) and
                                    not isinstance(i, bool) for i in g),
                "DIAG_STRATA", f"empty or non-integer group {g!r}")
        flat.extend(int(i) for i in g)
    require(sorted(flat) == list(range(V)), "DIAG_STRATA",
            "groups must be disjoint and cover every volume 0..V-1")
    b0 = None
    for g, arr in zip(groups, arrays):
        _draws(arr, len(g), "stratum draws")
        b0 = arr.shape[0] if b0 is None else b0
        require(arr.shape[0] == b0, "DIAG_STRATA", "draw counts differ")
    D = A - R
    pts, boots = [], []
    for g, arr in zip(groups, arrays):
        idx = np.asarray(g, dtype=np.int64)
        pts.append(fmean(D[:, idx].ravel().tolist()))
        boots.append(D[:, idx[arr]].mean(axis=2).mean(axis=0))
    res = {"thirds": [{"n": len(g), "estimate": p, "ci": _pct(bt)}
                      for g, p, bt in zip(groups, pts, boots)]}
    res["contrast_first_minus_last"] = {"estimate": pts[0] - pts[2],
                                        "ci": _pct(boots[0] - boots[2])}
    return res


# -- Spearman (§1.4) -----------------------------------------------------------
def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties) along the last axis."""
    order = np.argsort(x, axis=-1, kind="mergesort")
    xs = np.take_along_axis(x, order, axis=-1)
    n = x.shape[-1]
    r = np.empty_like(x, dtype=np.float64)
    pos = np.arange(1, n + 1, dtype=np.float64)
    for row in np.ndindex(x.shape[:-1]):
        s = xs[row]
        ranks = pos.copy()
        i = 0
        while i < n:
            j = i
            while j + 1 < n and s[j + 1] == s[i]:
                j += 1
            if j > i:
                ranks[i:j + 1] = (i + 1 + j + 1) / 2.0
            i = j + 1
        r[row][order[row]] = ranks
    return r


def _spearman_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rx, ry = _ranks(x), _ranks(y)
    rx = rx - rx.mean(axis=-1, keepdims=True)
    ry = ry - ry.mean(axis=-1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(-1) * (ry ** 2).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (rx * ry).sum(-1) / den
    rho[den == 0.0] = np.nan                         # constant vector
    return rho


def spearman_estimate(dx: np.ndarray, dy: np.ndarray, v: np.ndarray) -> dict:
    """dx, dy: (seeds, V) paired volume differences; averaged over seeds
    per volume first, then Spearman over volumes per draw (§1.4)."""
    dx, dy = _pair(dx, dy)
    v = _draws(v, dx.shape[1])
    x, y = dx.mean(axis=0), dy.mean(axis=0)
    point = float(_spearman_rows(x[None, :], y[None, :])[0])
    require(math.isfinite(point), "DIAG_CORRELATION_UNDEFINED",
            "constant vector in the observed data")
    rb = _spearman_rows(x[v], y[v])
    und = int(np.isnan(rb).sum())
    out = {"estimate": point, "undefined_draws": und, "draws": int(len(rb))}
    if und > UNDEF_MAX * len(rb):
        out["ci"] = None
        out["note"] = "not computed (> 1% undefined draws)"
    else:
        out["ci"] = _pct(rb[~np.isnan(rb)])
    return out


# -- labels (§2, §3) ------------------------------------------------------------
def direction_label(lo: float, hi: float) -> str:
    _finite_out(float(lo), "ci lo"); _finite_out(float(hi), "ci hi")
    require(lo <= hi, "DIAG_INTERVAL", f"lo {lo} > hi {hi}")
    if lo > 0:
        return "improvement supported"
    if hi < 0:
        return "deterioration supported"
    return "direction unresolved"


def practical_label(lo: float, hi: float) -> str:
    _finite_out(float(lo), "ci lo"); _finite_out(float(hi), "ci hi")
    require(lo <= hi, "DIAG_INTERVAL", f"lo {lo} > hi {hi}")
    if lo > PRACTICAL_DB:
        return "0.5 dB improvement supported"
    if hi < PRACTICAL_DB:
        return "0.5 dB improvement ruled out at this interval level"
    return "0.5 dB improvement unresolved"


def density_label(lo: float, hi: float) -> str:
    _finite_out(float(lo), "ci lo"); _finite_out(float(hi), "ci hi")
    require(lo <= hi, "DIAG_INTERVAL", f"lo {lo} > hi {hi}")
    """Registered density label on the NLL ratio interval."""
    if hi < DENSITY_BOUND:
        return "preserved"
    if lo > DENSITY_BOUND:
        return "worse"
    return "unclear"


def contributions(dnll: dict, dJ: float, dB: float) -> dict:
    """§3: shares only if the dNLL interval excludes 0 and |dNLL| >=
    CONTRIB_MIN; shares > 100 % allowed."""
    lo, hi = dnll["ci"]
    est = dnll["estimate"]
    for v, w in ((lo, "ci lo"), (hi, "ci hi"), (est, "dNLL"), (dJ, "dJ"),
                 (dB, "dB")):
        _finite_out(float(v), w)
    require(lo <= hi, "DIAG_INTERVAL", f"lo {lo} > hi {hi}")
    if (lo > 0 or hi < 0) and abs(est) >= CONTRIB_MIN:
        return {"reported": True, "J_share": dJ / est, "B_share": dB / est}
    return {"reported": False, "reason": "dNLL interval includes 0 or "
            f"|dNLL| < {CONTRIB_MIN} nats/dim"}
