# SEQREF-DUNC-CORE v0.2 -- D-UNC stage 1: per-slice uncertainty statistics (223 v1.0 §5, §6, §11)
# LIFETIME: KEEP
"""Stage 1 of the D-UNC extractor (SEQREF-MRI-DIAG-UNC v1.0 + AM1, LudWiki 223).

For one slice and one checkpoint, while its decoded samples are in memory, this
module computes every sample-dependent per-slice quantity on the full bank and on
the registered sample subsets of 223 §6. It reads no file and imports no V05LT code.

Step B (plan SEQREF-DUNC-IMPL v0.4): synthetic inputs only.

Changelog
- v0.2 (2026-11-16): physical Fourier band rule of 216 §4 (exact integer arithmetic); band-map
  integer dtype validated (typed error instead of an uncontrolled exception); JSON round-trip of
  stage-1 records for the disk-backed stage 2; DIAG_UNC_INPUT_INVALID per 223 AM1-2.
- v0.1 (2026-11-16): first version.
"""
import json
import logging
import math
from fractions import Fraction

import numpy as np

__version__ = "0.2"
LOG = logging.getLogger("diag_unc")

REGISTERED_K = 128
CELL_HW = 96            # registered square cell (216 §4)
N_FREE_COEF = 6912      # unacquired coefficients per slice (223 §3.2)
N_BANDS = 3
N_DECILES = 10
LEVELS = (("0.50", Fraction(1, 2)), ("0.80", Fraction(4, 5)),
          ("0.90", Fraction(9, 10)), ("0.95", Fraction(19, 20)))
FATAL_CODES = ("DIAG_UNC_REPRODUCTION_MISMATCH", "DIAG_UNC_POPULATION_MISMATCH",
               "DIAG_UNC_NONFINITE", "DIAG_UNC_ZERO_DENOMINATOR",
               "DIAG_UNC_RESOURCE_LIMIT", "DIAG_UNC_BOOTSTRAP_IDENTITY",
               "DIAG_UNC_INPUT_INVALID")
EVENT_UNDEFINED_CORRELATION = "DIAG_UNC_UNDEFINED_CORRELATION"  # non-fatal (223 §3.3)


class DiagUncError(Exception):
    """Typed validity failure. Always logged; never replaced by a substitute value."""

    def __init__(self, code, message):
        if code not in FATAL_CODES:
            raise ValueError("unknown D-UNC failure code: %r" % (code,))
        super().__init__("%s: %s" % (code, message))
        self.code = code
        LOG.error("%s: %s", code, message)


def _require(cond, code, message):
    if not cond:
        raise DiagUncError(code, message)


def _finite(name, a):
    _require(bool(np.all(np.isfinite(a))), "DIAG_UNC_NONFINITE", "%s contains a non-finite value" % name)


# ---------------------------------------------------------------- levels (U2)
def attainable(level, k_samples):
    """k_c and attainable nominal level pi_c for K samples; pi is None if not resolvable."""
    kc = math.floor((1 - level) / 2 * (k_samples + 1))
    if kc == 0:
        return 0, None
    return kc, float(Fraction(k_samples + 1 - 2 * kc, k_samples + 1))


# ---------------------------------------------------------------- subsets (§6)
def registered_subsets(k_samples):
    """Sample subsets of 223 §6 for the registered bank; only 'full' for any other K."""
    if k_samples != REGISTERED_K:
        return {"full": np.arange(k_samples)}, {}
    subsets = {"full": np.arange(128), "pre16": np.arange(16), "pre32": np.arange(32),
               "pre64": np.arange(64), "halfA": np.arange(0, 64), "halfB": np.arange(64, 128)}
    groups = {"g%d" % g: np.arange(16 * g, 16 * g + 16) for g in range(8)}
    return subsets, groups


# ---------------------------------------------------------------- primitives
def rank_counts(x, y):
    """L = #{k: X_k < y}, E = #{k: X_k = y}, per coordinate. x: (K, n); y: (n,)."""
    return (x < y).sum(axis=0), (x == y).sum(axis=0)


def rank_histogram(less, equal, k_samples):
    """K+1 bin masses; a tied observation gives 1/(E+1) to each of bins L..L+E (histogram only)."""
    diff = np.zeros(k_samples + 2)
    mass = 1.0 / (equal + 1.0)
    np.add.at(diff, less, mass)
    np.add.at(diff, less + equal + 1, -mass)
    return np.cumsum(diff)[:k_samples + 1]


def fair_crps(x_sorted, y):
    """Fair ensemble CRPS per coordinate by the sorted-sample identity. x_sorted: (K, n) ascending."""
    k = x_sorted.shape[0]
    coef = 2.0 * np.arange(k) - k + 1.0
    spread = (coef[:, None] * x_sorted).sum(axis=0) / (k * (k - 1.0))
    return np.abs(x_sorted - y).mean(axis=0) - spread


def pairwise_distances(x):
    """Euclidean distances between sample vectors, one sample against the rest per chunk. x: (K, n)."""
    k = x.shape[0]
    d = np.zeros((k, k))
    for i in range(k - 1):
        row = np.sqrt(((x[i + 1:] - x[i]) ** 2).sum(axis=1))
        d[i, i + 1:] = row
        d[i + 1:, i] = row
    return d


def fair_energy(dist, dist_y, idx):
    """Fair energy score on sample subset idx from the pairwise matrix and distances to the truth."""
    ks = len(idx)
    return float(dist_y[idx].mean() - dist[np.ix_(idx, idx)].sum() / (2.0 * ks * (ks - 1.0)))


def average_ranks(a):
    order = np.argsort(a, kind="stable")
    s = a[order]
    new = np.concatenate(([True], s[1:] != s[:-1]))
    group = np.cumsum(new) - 1
    count = np.bincount(group)
    end = np.cumsum(count)
    avg = end - (count - 1) / 2.0
    out = np.empty(len(a))
    out[order] = avg[group]
    return out


def spearman(a, b):
    """Spearman rho with average ranks; nan if either input is constant (undefined)."""
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra, rb = average_ranks(a), average_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra * ra).sum() * (rb * rb).sum()))


def decile_sums(sd, err_r0, err_r1):
    """Equal-count SD bins (lowest first, ties by index); rows: sum SD, sum R0 err, sum R1 err, count."""
    n = len(sd)
    order = np.argsort(sd, kind="stable")
    base, rem = divmod(n, N_DECILES)
    out = np.zeros((N_DECILES, 4))
    start = 0
    for b in range(N_DECILES):
        size = base + (1 if b < rem else 0)
        sel = order[start:start + size]
        out[b] = (sd[sel].sum(), err_r0[sel].sum(), err_r1[sel].sum(), size)
        start += size
    return out


def band_grid(hw=CELL_HW):
    """Band (0 low, 1 mid, 2 high) of every location of the centred hw x hw Fourier grid (216 §4).

    Index i maps to frequency k = i - hw/2 on both axes; rho = r / ((hw/2) * sqrt(2)); low rho in
    [0, 1/3), mid [1/3, 2/3), high [2/3, 1]. Evaluated exactly in integers: rho < a/3 <=> 9 r^2 < 2 a^2 (hw/2)^2,
    so a location exactly on a boundary goes to the upper band, as the half-open rule states.
    """
    _require(hw > 0 and hw % 2 == 0, "DIAG_UNC_INPUT_INVALID", "grid size must be a positive even integer")
    k = np.arange(hw, dtype=np.int64) - hw // 2
    r2 = k[:, None] ** 2 + k[None, :] ** 2            # rows = ky, columns = kx
    corner2 = 2 * (hw // 2) ** 2
    return (9 * r2 >= corner2).astype(np.int64) + (9 * r2 >= 4 * corner2).astype(np.int64)


def unacquired_band_map(column_mask):
    """Bands of the unacquired locations for a 1-D column mask (1 = acquired), as an (hw, n_unacquired) grid.

    The order in which the pinned V05LT code lists the free coefficients is NOT assumed here; the
    real-data adapter (fixture 11) must map this grid onto that order.
    """
    mask = np.asarray(column_mask)
    _require(mask.ndim == 1 and mask.dtype.kind in "biu" and bool(np.all((mask == 0) | (mask == 1))),
             "DIAG_UNC_INPUT_INVALID", "column mask must be a 1-D integer or boolean array of 0/1")
    return band_grid(len(mask))[:, mask == 0]


def check_band_map(band, n_coef):
    band = np.asarray(band)
    _require(band.dtype.kind in "iu", "DIAG_UNC_INPUT_INVALID", "band map is not an integer array")
    _require(band.shape == (n_coef,), "DIAG_UNC_INPUT_INVALID", "band map has the wrong length")
    _require(bool(np.all((band >= 0) & (band < N_BANDS))), "DIAG_UNC_INPUT_INVALID", "band map value outside 0..2")
    counts = np.bincount(band, minlength=N_BANDS)
    _require(int(counts.sum()) == n_coef, "DIAG_UNC_INPUT_INVALID", "band counts do not sum to the coefficient count")
    return counts


def standardise(u, loc, scale):
    """Complex coefficients (.., C) -> standardised real coordinates (.., 2C): [Re | Im]."""
    _require(bool(np.all(scale > 0)), "DIAG_UNC_INPUT_INVALID", "standardisation scale is not positive")
    real = np.concatenate([u.real, u.imag], axis=-1)
    return (real - loc) / scale


# ---------------------------------------------------------------- per subset
def _marginal(x, y):
    """Coverage counts, widths, histogram, ties, SD and fair CRPS for one space. x: (Ks, n) real."""
    ks, n = x.shape
    xs = np.sort(x, axis=0)
    less, equal = rank_counts(x, y)
    rank = less + 0.5 * equal
    cov = {}
    for name, level in LEVELS:
        kc, pi = attainable(level, ks)
        if pi is None:
            cov[name] = None
            continue
        covered = int(((rank >= kc) & (rank <= ks - kc)).sum())
        cov[name] = (covered, n, float((xs[ks - kc] - xs[kc - 1]).mean()))
    sd = x.std(axis=0, ddof=1)
    crps = fair_crps(xs, y)
    return {"cov": cov, "hist": rank_histogram(less, equal, ks), "ties": int((equal > 0).sum()), "n": n}, sd, crps


def _subset_record(idx, ustd, ytrue, band2, u, u_true, x, x_true, m_r0, t, dist, dist_y, ref_energy):
    ks = len(idx)
    rec = {"K": ks}
    mu, sd_u, crps_u = _marginal(ustd[idx], ytrue)
    m = np.abs(x[idx])
    mm, sd_m, crps_m = _marginal(m, t)
    rec["U"], rec["M"] = mu, mm
    rec["sd_mean"] = {"U": float(sd_u.mean()), "M": float(sd_m.mean())}
    rec["ucrps"] = float(crps_u.mean())
    rec["mcrps"] = float(crps_m.mean())
    rec["band_crps"] = [float(crps_u[band2 == b].mean()) for b in range(N_BANDS)]
    rec["energy"] = fair_energy(dist, dist_y, idx)
    xbar = x[idx].mean(axis=0)
    rec["sig2"] = float((np.abs(x[idx] - xbar) ** 2).mean(axis=1).sum() / (ks - 1.0))
    rec["e2"] = float((np.abs(xbar - x_true) ** 2).mean())
    err_r0 = np.abs(m_r0 - t)
    err_r1 = np.abs(np.abs(xbar) - t)
    rec["rho_r0"] = spearman(sd_m, err_r0)
    rec["rho_r1"] = spearman(sd_m, err_r1)
    rec["sd_slice"] = float(sd_m.mean())
    rec["mse_r0"] = float(((m_r0 - t) ** 2).mean())
    rec["deciles"] = decile_sums(sd_m, err_r0, err_r1)
    ubar = u[idx].mean(axis=0)
    band = band2[:u.shape[1]]
    rec["band_err_r1"] = [float((np.abs(ubar - u_true)[band == b] ** 2).sum() / ref_energy) for b in range(N_BANDS)]
    return rec


# ---------------------------------------------------------------- stage 1
def slice_stats(u, u_true, u_r0, loc, scale, band, x, x_true, x_r0, ref_energy,
                case_id=None, events=None, allow_other_k=False):
    """Stage-1 record for one slice and checkpoint.

    u       (K, C) complex   decoded unacquired coefficients, one row per bank sample
    u_true  (C,)   complex   reference coefficients;  u_r0 (C,) complex  z = 0 decode
    loc, scale (2C,) real    pinned standardisation (scale > 0);  band (C,) int in {0,1,2}
    x       (K, P) complex   images of the samples;  x_true, x_r0 (P,) complex
    ref_energy     float     sum over all Fourier locations of |F(x*)|^2
    events  list or None     receives non-fatal (code, case_id, measure) entries
    """
    u, x = np.asarray(u), np.asarray(x)
    _require(u.ndim == 2 and x.ndim == 2 and u.shape[0] == x.shape[0], "DIAG_UNC_INPUT_INVALID",
             "sample arrays must be (K, C) and (K, P) with one K")
    k, c = u.shape
    _require(allow_other_k or k == REGISTERED_K, "DIAG_UNC_INPUT_INVALID",
             "bank has %d samples; the registered bank has %d" % (k, REGISTERED_K))
    _require(k >= 2, "DIAG_UNC_INPUT_INVALID", "at least two samples are needed")
    _require(u_true.shape == (c,) and u_r0.shape == (c,) and loc.shape == (2 * c,) and scale.shape == (2 * c,)
             and x_true.shape == (x.shape[1],) and x_r0.shape == (x.shape[1],),
             "DIAG_UNC_INPUT_INVALID", "input shapes are inconsistent")
    for name, arr in (("u", u), ("u_true", u_true), ("u_r0", u_r0), ("loc", loc), ("scale", scale),
                      ("x", x), ("x_true", x_true), ("x_r0", x_r0)):
        _finite(name, arr)
    check_band_map(band, c)
    _require(bool(np.isfinite(ref_energy)), "DIAG_UNC_NONFINITE", "reference energy is not finite")
    _require(ref_energy > 0, "DIAG_UNC_ZERO_DENOMINATOR", "reference spectral energy is zero")

    ustd = standardise(u, loc, scale)
    ytrue = standardise(u_true, loc, scale)
    band2 = np.concatenate([band, band])
    t, m_r0 = np.abs(x_true), np.abs(x_r0)
    dist = pairwise_distances(ustd)
    dist_y = np.sqrt(((ustd - ytrue) ** 2).sum(axis=1))

    subsets, groups = registered_subsets(k)
    out = {"case_id": case_id, "subsets": {}, "groups": {}}
    for name, idx in subsets.items():
        out["subsets"][name] = _subset_record(idx, ustd, ytrue, band2, u, u_true, x, x_true, m_r0, t,
                                              dist, dist_y, ref_energy)
    for name, idx in groups.items():  # fair scores only (223 §6.3)
        xs = np.sort(ustd[idx], axis=0)
        ms = np.sort(np.abs(x[idx]), axis=0)
        out["groups"][name] = {"ucrps": float(fair_crps(xs, ytrue).mean()),
                               "mcrps": float(fair_crps(ms, t).mean()),
                               "energy": fair_energy(dist, dist_y, idx)}
    err0 = np.abs(u_r0 - u_true) ** 2
    out["band_err_r0"] = [float(err0[band == b].sum() / ref_energy) for b in range(N_BANDS)]
    for name, rec in out["subsets"].items():
        for measure in ("rho_r0", "rho_r1"):
            if math.isnan(rec[measure]):
                LOG.warning("%s: case %r, subset %s, %s", EVENT_UNDEFINED_CORRELATION, case_id, name, measure)
                if events is not None:
                    events.append((EVENT_UNDEFINED_CORRELATION, case_id, name, measure))
    return out


# ---------------------------------------------------------------- retained statistics on disk
def record_to_bytes(record):
    """Stage-1 record as JSON bytes. float64 values round-trip exactly (shortest repr)."""
    def enc(o):
        if isinstance(o, np.ndarray):
            return {"__nd__": o.tolist(), "shape": list(o.shape)}
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, tuple):
            return {"__tuple__": [enc(v) for v in o]}
        if isinstance(o, dict):
            return {str(k): enc(v) for k, v in o.items()}
        if isinstance(o, list):
            return [enc(v) for v in o]
        return o
    return json.dumps(enc(record), sort_keys=True).encode("utf-8")


def record_from_bytes(data):
    def dec(o):
        if isinstance(o, dict):
            if "__nd__" in o:
                return np.array(o["__nd__"], dtype=float).reshape(o["shape"])
            if "__tuple__" in o:
                return tuple(dec(v) for v in o["__tuple__"])
            return {k: dec(v) for k, v in o.items()}
        if isinstance(o, list):
            return [dec(v) for v in o]
        return o
    try:
        return dec(json.loads(data.decode("utf-8")))
    except (ValueError, UnicodeDecodeError) as exc:
        raise DiagUncError("DIAG_UNC_INPUT_INVALID", "retained-statistics file is not readable: %s" % exc)
