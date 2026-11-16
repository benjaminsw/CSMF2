# SEQREF-DUNC-FIX v0.5 -- D-UNC synthetic fixtures 1-10 and 12 (223 v1.0 + AM1 §10; plan SEQREF-DUNC-IMPL v0.4)
# LIFETIME: KEEP
"""Synthetic fixtures for the D-UNC extractor. Synthetic data only: no V05LT module is
imported, and no checkpoint, stored row or MRI data is read.

Registered before the first run (plan v0.3/v0.4): D1 fixture-12 tolerances, D3 chi-square
test, D4 fixture seed 20261116. Set in this file and declared in plan v0.4: Z_SE = 4
standard errors for the statistical checks of fixtures 1-3 and 8, and SSR_TOL = 0.02 for
'SSR close to 1'.

The registered bootstrap matrix (seed 20261114) is NOT generated here (223 §3.4: it is
first generated in F1). Identity tests use the same call with the fixture seed.

Run:  python3 -B test_diag_unc_fixtures.py   (exit 0 = all pass; 3 = all pass with open items; 1 = failure)

Changelog
- v0.5 (2026-11-16): every refused guard start is checked for an identical directory tree before and
  after (names, kinds and file contents), including directories that had no lock file. No other fixture changed.
- v0.4 (2026-11-16): guard fixtures for ledger creation — allowed in a new empty directory (also when
  four processes start there at once), refused when the ledger is missing from a directory that holds
  outputs, leftover partial files, unrelated files, or only a marked lock file; nothing is deleted on
  refusal. No other fixture changed.
- v0.3 (2026-11-16): guard fixtures for the reviewer's counterexample (two active guard objects on one
  directory), interleaved guards, four concurrent processes with and without a binding ceiling, a guard
  starting during another guard's write, recovery that leaves unrelated '.part' files alone, and a
  damaged or removed ledger at start and at write. No other fixture changed.
- v0.2 (2026-11-16): fixture 6 with the 216 §4 radius rule on the 96 x 96 grid; fixture 9 with the
  AM1-3 three-valued row logic, deciding effects, the Tier 1/Tier 2 contrast table and the AM1-4
  identity; fixture 10 with an instrumented generator (AM1-1) and the typed input error (AM1-2);
  fixture 12 on six registered roles and all ten Tier 1 contrasts, with tied data and the retained
  statistics written through the guard and reloaded; guard restart, interrupted-write and ledger tests;
  Python, NumPy and SciPy versions logged.
- v0.1 (2026-11-16): first version.
"""
import copy
import logging
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np

import diag_unc_core as core
import diag_unc_guard as guard
import diag_unc_stats as stats
from diag_unc_core import DiagUncError

FIXTURE_SEED = 20261116
ATOL_SLICE, RTOL_SLICE = 1e-12, 1e-10      # D1, per-slice quantities and histogram masses
ATOL_POP, RTOL_POP = 1e-10, 1e-8           # D1, population-level estimates and interval endpoints
CHI2_N, CHI2_BINS, CHI2_EXPECTED = 25800, 129, 200   # D3
CHI2_CRIT_128_P001 = 183.1864373344724     # upper 0.001 point of chi-square, 128 df (p > 0.001 <=> stat < crit)
Z_SE = 4.0
SSR_TOL = 0.02
OPEN_ITEMS = []     # anything that needs a reviewer decision; never counted as a pass
OBSERVATIONS = []   # facts the fixtures are required to report (223 AM1-1); not failures


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def rng_for(n):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([FIXTURE_SEED, n])))


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def close(a, b, atol, rtol):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False, float("inf")
    both_nan = np.isnan(a) & np.isnan(b)
    dev = np.where(both_nan, 0.0, np.abs(a - b))
    ok = bool(np.all((dev <= atol + rtol * np.abs(np.where(both_nan, 0.0, b))) & ~(np.isnan(a) ^ np.isnan(b))))
    return ok, float(np.nanmax(dev)) if dev.size else 0.0


def expect_error(code, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except DiagUncError as exc:
        check(exc.code == code, "expected %s, got %s" % (code, exc.code))
        return
    raise AssertionError("expected %s, but no error was raised" % code)


def cgauss(rng, size):
    return rng.standard_normal(size) + 1j * rng.standard_normal(size)


def unitary(rng, n):
    q, r = np.linalg.qr(cgauss(rng, (n, n)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def band_map(c):
    return np.minimum(np.arange(c) * core.N_BANDS // c, core.N_BANDS - 1)


def make_inputs(rng, k=128, c=12, n_acq=6, sigma_f=1.0, bias=0.0, q=None):
    """One synthetic slice: predictive N(mu, sigma_f^2) per coefficient, truth N(mu, 1) (+ bias)."""
    q = unitary(rng, n_acq + c) if q is None else q
    acq = cgauss(rng, n_acq)
    mu = cgauss(rng, c)
    u_true = mu + cgauss(rng, c) + bias
    u = mu + sigma_f * cgauss(rng, (k, c))
    img = lambda coef: q[:, :n_acq] @ acq + coef @ q[:, n_acq:].T
    loc = rng.standard_normal(2 * c)
    scale = 0.5 + rng.random(2 * c)
    return {"u": u, "u_true": u_true, "u_r0": mu, "loc": loc, "scale": scale, "band": band_map(c),
            "x": img(u), "x_true": img(u_true), "x_r0": img(mu),
            "ref_energy": float((np.abs(acq) ** 2).sum() + (np.abs(u_true) ** 2).sum())}, img


# ================================================================ fixture 1
def fixture_01():
    rng = rng_for(1)
    info = {}
    x, y = rng.standard_normal((128, CHI2_N)), rng.standard_normal(CHI2_N)
    less, equal = core.rank_counts(x, y)
    hist = core.rank_histogram(less, equal, 128)
    check(len(hist) == CHI2_BINS and abs(hist.sum() - CHI2_N) < 1e-9, "histogram shape or mass")
    chi2 = float(((hist - CHI2_EXPECTED) ** 2 / CHI2_EXPECTED).sum())
    info["chi2"] = chi2
    check(chi2 < CHI2_CRIT_128_P001, "rank histogram not uniform: chi2 = %.2f" % chi2)
    try:
        from scipy.stats import chi2 as chi2_dist
        info["chi2_p"] = float(chi2_dist.sf(chi2, 128))
        check((info["chi2_p"] > 0.001) == (chi2 < CHI2_CRIT_128_P001), "critical value inconsistent with scipy")
    except ImportError:
        info["chi2_p"] = "scipy not available; critical value used"
    rank = less + 0.5 * equal
    for name, level in core.LEVELS:
        kc, pi = core.attainable(level, 128)
        cov = float(((rank >= kc) & (rank <= 128 - kc)).mean())
        check(abs(cov - pi) <= Z_SE * math.sqrt(pi * (1 - pi) / CHI2_N), "coverage %s: %.4f vs %.4f" % (name, cov, pi))
    for k in (16, 128):
        xs = np.sort(rng.standard_normal((k, CHI2_N)), axis=0)
        crps = core.fair_crps(xs, rng.standard_normal(CHI2_N))
        se = crps.std(ddof=1) / math.sqrt(CHI2_N)
        check(abs(crps.mean() - 1 / math.sqrt(math.pi)) <= Z_SE * se, "fair CRPS biased at K = %d" % k)
        d, n_obs = 8, 1500
        closed = math.exp(math.lgamma((d + 1) / 2) - math.lgamma(d / 2))
        es = np.empty(n_obs)
        for i in range(n_obs):
            s, t = rng.standard_normal((k, d)), rng.standard_normal(d)
            es[i] = core.fair_energy(core.pairwise_distances(s), np.sqrt(((s - t) ** 2).sum(axis=1)), np.arange(k))
        check(abs(es.mean() - closed) <= Z_SE * es.std(ddof=1) / math.sqrt(n_obs), "fair energy biased at K = %d" % k)
        ssr = _ssr_synthetic(rng, k, sigma_f=1.0)
        info["ssr_K%d" % k] = ssr
        check(abs(ssr - 1) <= SSR_TOL, "SSR of a calibrated ensemble at K = %d is %.4f" % (k, ssr))
    return info


def _ssr_synthetic(rng, k, sigma_f, bias=0.0, n_vol=200, n_pix=64):
    mu = cgauss(rng, (n_vol, 1, n_pix))
    x = mu + sigma_f * cgauss(rng, (n_vol, k, n_pix))
    truth = mu[:, 0] + cgauss(rng, (n_vol, n_pix)) + bias
    xbar = x.mean(axis=1)
    sig2 = (np.abs(x - xbar[:, None]) ** 2).mean(axis=2).sum(axis=1) / (k - 1)
    e2 = (np.abs(xbar - truth) ** 2).mean(axis=1)
    return math.sqrt((k + 1) / k * sig2.mean()) / math.sqrt(e2.mean())


# ================================================================ fixture 2
def fixture_02():
    rng = rng_for(2)
    n, k = CHI2_N, 128
    y = rng.standard_normal(n)
    kc, pi = core.attainable(dict(core.LEVELS)["0.90"], k)
    se = math.sqrt(pi * (1 - pi) / n)
    res = {}
    for name, sig, shift in (("cal", 1.0, 0.0), ("over", 0.5, 0.0), ("under", 2.0, 0.0), ("shift", 1.0, 1.0)):
        x = shift + sig * rng.standard_normal((k, n))
        xs = np.sort(x, axis=0)
        less, equal = core.rank_counts(x, y)
        rank = less + 0.5 * equal
        res[name] = {"cov": float(((rank >= kc) & (rank <= k - kc)).mean()),
                     "width": float((xs[k - kc] - xs[kc - 1]).mean()), "crps": float(core.fair_crps(xs, y).mean())}
    check(res["over"]["cov"] < pi - Z_SE * se, "over-confident case not under-covering")
    check(res["under"]["cov"] > pi + Z_SE * se, "under-confident case not over-covering")
    check(res["shift"]["cov"] < pi - Z_SE * se, "shifted case not under-covering")
    check(res["over"]["width"] < res["cal"]["width"] < res["under"]["width"], "width ordering")
    check(all(res["cal"]["crps"] < res[c]["crps"] for c in ("over", "under", "shift")), "CRPS ordering")
    ssr = {"over": _ssr_synthetic(rng, k, 0.5), "under": _ssr_synthetic(rng, k, 2.0),
           "shift": _ssr_synthetic(rng, k, 1.0, bias=1.0)}
    check(ssr["over"] < 1 - SSR_TOL and ssr["shift"] < 1 - SSR_TOL and ssr["under"] > 1 + SSR_TOL, "SSR side")
    return {"coverage_0.90": {c: res[c]["cov"] for c in res}, "ssr": ssr}


# ================================================================ fixture 3
def fixture_03():
    rng = rng_for(3)
    d, n_obs, k, rho = 6, 1500, 128, 0.9
    chol = np.linalg.cholesky((1 - rho) * np.eye(d) + rho * np.ones((d, d)))
    out = {n: {"crps": np.empty(n_obs), "es": np.empty(n_obs)} for n in ("A", "B", "C")}
    for i in range(n_obs):
        y = chol @ rng.standard_normal(d)
        samples = {"A": rng.standard_normal((k, d)) @ chol.T,        # correct dependence
                   "B": rng.standard_normal((k, d)),                 # same marginals, independent
                   "C": 2.0 * rng.standard_normal((k, d)) @ chol.T}  # same dependence, wider marginals
        for name, s in samples.items():
            out[name]["crps"][i] = core.fair_crps(np.sort(s, axis=0), y).mean()
            out[name]["es"][i] = core.fair_energy(core.pairwise_distances(s), np.sqrt(((s - y) ** 2).sum(axis=1)),
                                                  np.arange(k))
    z = lambda a: a.mean() / (a.std(ddof=1) / math.sqrt(n_obs))
    z_crps_ab, z_es_ab = z(out["A"]["crps"] - out["B"]["crps"]), z(out["A"]["es"] - out["B"]["es"])
    z_es_ac = z(out["A"]["es"] - out["C"]["es"])
    check(abs(z_crps_ab) <= Z_SE, "marginal CRPS responds to dependence: z = %.2f" % z_crps_ab)
    check(z_es_ab < -Z_SE, "energy score does not respond to dependence: z = %.2f" % z_es_ab)
    check(abs(z_es_ac) > Z_SE, "energy score does not respond to a marginal-only change: z = %.2f" % z_es_ac)
    return {"z_crps_dependence": z_crps_ab, "z_energy_dependence": z_es_ab, "z_energy_marginal_only": z_es_ac}


# ================================================================ fixture 4
EXPECTED_KC = {16: (4, 1, 0, 0), 32: (8, 3, 1, 0), 64: (16, 6, 3, 1), 128: (32, 12, 6, 3)}


def fixture_04():
    rng = rng_for(4)
    for k, kcs in EXPECTED_KC.items():
        for (name, level), kc in zip(core.LEVELS, kcs):
            got_kc, pi = core.attainable(level, k)
            check(got_kc == kc, "k_c for K = %d, level %s" % (k, name))
            check((pi is None) if kc == 0 else abs(pi - (k + 1 - 2 * kc) / (k + 1)) < 1e-15, "pi_c K = %d %s" % (k, name))
    pis = [core.attainable(level, 128)[1] for _, level in core.LEVELS]
    check([round(p, 4) for p in pis] == [0.5039, 0.814, 0.907, 0.9535], "K = 128 attainable levels")
    # midrank ties and histogram masses
    x, y = np.array([[1.0], [2.0], [2.0], [3.0]]), np.array([2.0])
    less, equal = core.rank_counts(x, y)
    check((less[0], equal[0]) == (1, 2) and less[0] + 0.5 * equal[0] == 2.0, "midrank with ties")
    hist = core.rank_histogram(less, equal, 4)
    check(np.allclose(hist, [0, 1 / 3, 1 / 3, 1 / 3, 0]), "tie mass not spread over bins L..L+E")
    xt = np.round(rng.standard_normal((32, 500)), 1)
    yt = np.round(rng.standard_normal(500), 1)
    lt, et = core.rank_counts(xt, yt)
    ht = core.rank_histogram(lt, et, 32)
    check(abs(ht.sum() - 500) < 1e-9 and (et > 0).any() and ht.min() >= -1e-12, "histogram mass with ties")
    # order-statistic rule <=> rank rule (continuous data), every K and resolvable level
    for k in EXPECTED_KC:
        xc, yc = rng.standard_normal((k, 4000)), rng.standard_normal(4000)
        xs = np.sort(xc, axis=0)
        lc, ec = core.rank_counts(xc, yc)
        check(not ec.any(), "unexpected tie in continuous data")
        for _, level in core.LEVELS:
            kc, pi = core.attainable(level, k)
            if pi is None:
                continue
            by_rank = (lc >= kc) & (lc <= k - kc)
            by_order = (xs[kc - 1] < yc) & (yc < xs[k - kc])
            check(np.array_equal(by_rank, by_order), "order-statistic and rank rules differ at K = %d" % k)
    # tie-fraction flag and 'coverage unchanged by the histogram rule', through stage 1 and stage 2
    vols = list(range(5))
    events = []
    pop = stats.Population(vols)
    for v in vols:
        inp, _ = make_inputs(rng, k=128)
        inp["x"][:, :2] = 0.0
        inp["x_true"][:2] = 0.0
        rec = core.slice_stats(case_id=v, events=events, **inp)
        m = rec["subsets"]["full"]["M"]
        check(m["ties"] == 2 and abs(m["hist"].sum() - m["n"]) < 1e-9, "tie count or histogram mass")
        check(abs(m["hist"][0] - 2 / 129) < 1 and m["cov"]["0.50"][0] >= 2, "tied pixels not covered by midrank")
        pop.add("r", 0, v, rec)
    pop.seal()
    idx = stats.bootstrap_matrix(seed=FIXTURE_SEED, n_volumes=5, n_boot=200)
    cal = stats.checkpoint_calibration(pop, "r", "M", "0.90", idx)
    check(cal["ties_material"] and abs(cal["tie_fraction"] - 2 / 18) < 1e-12, "tie flag in M-space")
    check(not stats.checkpoint_calibration(pop, "r", "U", "0.90", idx)["ties_material"], "tie flag in U-space")
    k16 = core.slice_stats(allow_other_k=True, **make_inputs(rng, k=16)[0])["subsets"]["full"]
    check(k16["U"]["cov"]["0.90"] is None and k16["U"]["cov"]["0.95"] is None and k16["U"]["cov"]["0.80"] is not None,
          "levels not marked 'not resolvable' at K = 16")
    return {"attainable_K128": pis}


# ================================================================ fixture 5
def fixture_05():
    rng = rng_for(5)
    x, y = np.round(rng.standard_normal((128, 50)), 2), rng.standard_normal(50)
    direct = np.abs(x - y).mean(axis=0) - np.abs(x[:, None, :] - x[None, :, :]).sum(axis=(0, 1)) / (2 * 128 * 127)
    ok, dev = close(core.fair_crps(np.sort(x, axis=0), y), direct, 1e-12, 1e-10)
    check(ok, "sorted-sample CRPS differs from the double sum by %.3g" % dev)
    s, t = rng.standard_normal((128, 40)), rng.standard_normal(40)
    dist, dist_y = core.pairwise_distances(s), np.sqrt(((s - t) ** 2).sum(axis=1))
    devs = [dev]
    for idx in (np.arange(128), np.arange(16), np.arange(64, 128), np.arange(32, 48)):
        sub = s[idx]
        ref = np.linalg.norm(sub - t, axis=1).mean() - np.linalg.norm(sub[:, None] - sub[None], axis=2).sum() / (
            2 * len(idx) * (len(idx) - 1))
        ok, dev = close(core.fair_energy(dist, dist_y, idx), ref, 1e-12, 1e-10)
        check(ok, "chunked energy score differs from the direct one by %.3g" % dev)
        devs.append(dev)
    return {"max_deviation": max(devs)}


# ================================================================ fixture 6
def _centred_fft(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))


def fixture_06():
    rng = rng_for(6)
    inp, _ = make_inputs(rng)
    bad = dict(inp, scale=inp["scale"].copy())
    bad["scale"][3] = 0.0
    expect_error("DIAG_UNC_INPUT_INVALID", core.slice_stats, **bad)
    bad["scale"][3] = -1.0
    expect_error("DIAG_UNC_INPUT_INVALID", core.slice_stats, **bad)
    raw = np.concatenate([inp["u"].real, inp["u"].imag], axis=1)
    raw_y = np.concatenate([inp["u_true"].real, inp["u_true"].imag])
    l1, e1 = core.rank_counts(raw, raw_y)
    l2, e2 = core.rank_counts(core.standardise(inp["u"], inp["loc"], inp["scale"]),
                              core.standardise(inp["u_true"], inp["loc"], inp["scale"]))
    check(np.array_equal(l1, l2) and np.array_equal(e1, e2), "ranks change under standardisation")
    # physical band rule of 216 §4 on the registered 96 x 96 grid, against exact rational arithmetic
    from fractions import Fraction
    grid = core.band_grid()
    check(grid.shape == (96, 96) and grid.dtype.kind == "i", "band grid shape or dtype")
    ref = np.empty((96, 96), dtype=int)
    for i in range(96):
        for j in range(96):
            rho2 = Fraction((i - 48) ** 2 + (j - 48) ** 2, 2 * 48 * 48)     # rho^2, exact
            ref[i, j] = 0 if rho2 < Fraction(1, 9) else (1 if rho2 < Fraction(4, 9) else 2)
    check(np.array_equal(grid, ref), "band grid differs from the exact radius rule")
    counts = np.bincount(grid.ravel(), minlength=3)
    check(int(counts.sum()) == 96 * 96 and counts.min() > 0, "band counts do not sum to 96^2")
    # plane waves and a delta fall in their expected bands (centred orthonormal FFT, 216 §4)
    yy, xx = np.meshgrid(np.arange(96), np.arange(96), indexing="ij")
    for (ky, kx), band in (((0, 0), 0), ((10, 20), 0), ((16, 16), 1), ((0, 30), 1), ((-22, 5), 0), ((-25, 5), 1),
                           ((32, 32), 2), ((40, -40), 2), ((-48, -48), 2), ((0, 47), 2)):
        spec = np.abs(_centred_fft(np.exp(2j * np.pi * (ky * (yy - 48) + kx * (xx - 48)) / 96.0)))
        iy, ix = np.unravel_index(np.argmax(spec), spec.shape)
        check((iy - 48, ix - 48) == (ky, kx) and spec[iy, ix] > 95.9, "plane wave (%d, %d) not at its location" % (ky, kx))
        check(grid[iy, ix] == band, "plane wave (%d, %d) in band %d, expected %d" % (ky, kx, grid[iy, ix], band))
    delta = np.zeros((96, 96))
    delta[48, 48] = 1.0
    energy = np.abs(_centred_fft(delta)) ** 2
    ok, dev = close([energy[grid == b].sum() for b in range(3)], counts / 9216.0, 1e-14, 1e-12)
    check(ok, "delta energy per band is not the band's share of locations: %.3g" % dev)
    # unacquired locations for a column mask with 24 acquired columns: 6,912 coefficients
    mask = np.zeros(96, dtype=np.int64)
    mask[rng.choice(96, 24, replace=False)] = 1
    ub = core.unacquired_band_map(mask)
    check(ub.shape == (96, 72) and ub.size == core.N_FREE_COEF, "unacquired band map shape")
    ucounts = core.check_band_map(ub.ravel(), core.N_FREE_COEF)
    check(int(ucounts.sum()) == 6912, "band counts do not sum to 6,912")
    check(np.array_equal(ub, grid[:, mask == 0]), "unacquired bands are not the grid restricted to mask = 0")
    expect_error("DIAG_UNC_INPUT_INVALID", core.unacquired_band_map, mask.astype(float))
    expect_error("DIAG_UNC_INPUT_INVALID", core.unacquired_band_map, np.full(96, 2))
    expect_error("DIAG_UNC_INPUT_INVALID", core.check_band_map, np.full(6912, 3), 6912)
    expect_error("DIAG_UNC_INPUT_INVALID", core.check_band_map, band_map(6911), 6912)
    expect_error("DIAG_UNC_INPUT_INVALID", core.check_band_map, band_map(6912).astype(float), 6912)
    expect_error("DIAG_UNC_INPUT_INVALID", core.slice_stats, **dict(inp, band=inp["band"].astype(float)))
    # spectral partition identity and zero-reference refusal
    rec = core.slice_stats(**inp)
    total = float((np.abs(inp["u_r0"] - inp["u_true"]) ** 2).sum() / inp["ref_energy"])
    ok, dev = close(sum(rec["band_err_r0"]), total, 1e-14, 1e-12)
    check(ok, "spectral partition identity (R0) off by %.3g" % dev)
    ubar = inp["u"].mean(axis=0)
    ok, dev = close(sum(rec["subsets"]["full"]["band_err_r1"]),
                    float((np.abs(ubar - inp["u_true"]) ** 2).sum() / inp["ref_energy"]), 1e-14, 1e-12)
    check(ok, "spectral partition identity (R1) off by %.3g" % dev)
    expect_error("DIAG_UNC_ZERO_DENOMINATOR", core.slice_stats, **dict(inp, ref_energy=0.0))
    return {"band_counts_96x96": counts.tolist(), "unacquired_band_counts": ucounts.tolist()}


# ================================================================ fixture 7
def fixture_07():
    rng = rng_for(7)
    out = {}
    for k in (128, 16):
        vols = list(range(6))
        pop = stats.Population(vols)
        sig2, e2, usig2, ue2 = [], [], [], []
        for v in vols:
            inp, img = make_inputs(rng, k=k)
            pop.add("r", 0, v, core.slice_stats(allow_other_k=True, **inp))
            xbar, ubar = inp["x"].mean(axis=0), inp["u"].mean(axis=0)
            ok, dev = close(np.abs(xbar - img(ubar)), np.zeros(len(xbar)), 1e-12, 0)
            check(ok, "R1 differs from image_from_u(mean u) by %.3g" % dev)
            n_pix = inp["x"].shape[1]
            sig2.append((np.abs(inp["x"] - xbar) ** 2).sum() / n_pix / (k - 1))
            e2.append((np.abs(xbar - inp["x_true"]) ** 2).sum() / n_pix)
            usig2.append((np.abs(inp["u"] - ubar) ** 2).sum() / (k - 1))
            ue2.append((np.abs(ubar - inp["u_true"]) ** 2).sum())
        pop.seal()
        idx = stats.bootstrap_matrix(seed=FIXTURE_SEED, n_volumes=6, n_boot=100)
        got = stats.nonlinear_effect(pop, {"r": 1.0}, "ssr", "full", idx)["point"]
        hand = math.sqrt((k + 1) / k * np.mean(sig2)) / math.sqrt(np.mean(e2))
        ok, dev = close(got, hand, 1e-12, 1e-10)
        check(ok, "pooled SSR differs from the formula by %.3g (K = %d)" % (dev, k))
        uncorrected = math.sqrt(np.mean(sig2)) / math.sqrt(np.mean(e2))
        ok, _ = close(got / uncorrected, math.sqrt((k + 1) / k), 1e-12, 1e-10)
        check(ok, "finite-K correction missing")
        parseval = math.sqrt((k + 1) / k * np.mean(usig2)) / math.sqrt(np.mean(ue2))
        ok, dev = close(got, parseval, 1e-10, 1e-9)
        check(ok, "image SSR differs from the physical-U ratio by %.3g" % dev)
        out["ssr_K%d" % k] = got
    # zero denominator: sample mean equals the reference exactly in every volume
    pop = stats.Population([0, 1])
    events = []
    for v in (0, 1):
        inp, _ = make_inputs(rng)
        inp["x_true"] = np.round(4 * np.abs(inp["x_true"])) + 0j   # real integers: sums, means and magnitudes exact
        inp["x"] = np.tile(inp["x_true"], (128, 1))
        pop.add("r", 0, v, core.slice_stats(case_id=v, events=events, **inp))
    pop.seal()
    expect_error("DIAG_UNC_ZERO_DENOMINATOR", stats.nonlinear_effect, pop, {"r": 1.0}, "ssr", "full",
                 stats.bootstrap_matrix(seed=FIXTURE_SEED, n_volumes=2, n_boot=10))
    check(len(events) > 0 and all(e[0] == core.EVENT_UNDEFINED_CORRELATION for e in events),
          "constant SD map did not record the non-fatal undefined-correlation event")
    eff = stats.additive_effect(pop, {"r": 1.0}, "rho_r0", "full", stats.bootstrap_matrix(FIXTURE_SEED, 2, 10))
    check(eff["label"] == "not computed" and eff["point"] is None, "undefined within-slice rho was substituted")
    check(stats.additive_effect(pop, {"r": 1.0}, "ucrps", "full", stats.bootstrap_matrix(FIXTURE_SEED, 2, 10))["point"]
          is not None, "an unaffected measure was stopped by the undefined correlation")
    return out


# ================================================================ fixture 8
def fixture_08():
    rng = rng_for(8)
    n_rep, n_vol, n_grp, sigma, tau = 4000, 97, 8, 1.0, 0.3
    out = {}
    for name, t in (("independent", 0.0), ("shared_bank", tau)):
        se2 = np.empty(n_rep)
        for r in range(n_rep):
            vals = t * rng.standard_normal((1, n_grp)) + sigma * rng.standard_normal((n_vol, n_grp))
            se2[r] = stats.batch_means_se(vals.mean(axis=0))[1] ** 2
        known = (t * t + sigma * sigma / n_vol) / n_grp
        se_of_mean = known * math.sqrt(2.0 / (n_grp - 1) / n_rep)
        check(abs(se2.mean() - known) <= Z_SE * se_of_mean, "batch-means SE^2 %.5g vs known %.5g (%s)"
              % (se2.mean(), known, name))
        out[name] = {"mean_se2": float(se2.mean()), "known": known}
    check(out["shared_bank"]["mean_se2"] > 5 * out["independent"]["mean_se2"],
          "batch-means SE does not carry the cross-volume correlation of a shared bank")
    mean, se = stats.batch_means_se([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    check(mean == 4.5 and abs(se - math.sqrt(6.0) / math.sqrt(8.0)) < 1e-15, "batch-means arithmetic")
    same = {"pre64": "below0", "full": "below0", "halfA": "below0", "halfB": "below0", "pre16": "unresolved"}
    check(stats.stability_label(same) == "stable", "stable label")
    check(stats.stability_label(dict(same, halfB="unresolved")) == "Monte Carlo adequacy unresolved", "unstable half")
    check(stats.stability_label(dict(same, pre64="above0")) == "Monte Carlo adequacy unresolved", "unstable K = 64")
    check(stats.stability_label({"full": "below0"}) == "Monte Carlo adequacy unresolved", "missing subset")
    return out


# ================================================================ shared synthetic population
ROLES = {"lam0/e3": (1.7, 1.3), "lam0/sel": (1.25, 1.0), "lam0/fin": (0.9, 1.05),
         "lam2/e3": (1.6, 1.2), "lam2/sel": (1.1, 0.9), "lam2/fin": (1.0, 0.85)}
TIER2_ROLES = {"lam0/e10": (1.3, 1.0), "lam0/e30": (1.0, 1.0), "lam2/e10": (1.2, 0.95), "lam2/e30": (1.05, 0.9)}


def _population(rng, roles, seeds=(2026, 2027, 2028), n_vol=97, keep=None, ties=False, disk=None):
    """Synthetic population: roles {name: (sigma_f, bias)}; truth per volume shared by all roles and seeds.

    ties  put exact ties into the data: two zero pixels in every image (M-space) and, in every third
          volume, a reference coefficient equal to one bank sample (U-space).
    disk  a WriteGuard: the stage-1 records are written through it, read back and verified, and the
          population is built from the reloaded records only (stage 2 never sees the in-memory ones).
    """
    c, n_acq = 12, 6
    q = unitary(rng, n_acq + c)
    vols = list(range(n_vol))
    truth = [(cgauss(rng, n_acq), cgauss(rng, c)) for _ in vols]
    loc, scale = rng.standard_normal(2 * c), 0.5 + rng.random(2 * c)
    pop = stats.Population(vols)
    for role, (sigma_f, bias) in roles.items():
        for seed in seeds:
            recs = []
            for v in vols:
                acq, u_true = truth[v]
                mu = u_true + bias * cgauss(rng, c)
                u = mu + sigma_f * cgauss(rng, (128, c))
                if ties and v % 3 == 0:
                    u_true = u_true.copy()
                    u_true[0] = u[5, 0]
                img = lambda coef: q[:, :n_acq] @ acq + coef @ q[:, n_acq:].T
                x, x_true, x_r0 = img(u), img(u_true), img(mu)
                if ties:
                    x[:, :2], x_true[:2], x_r0[:2] = 0.0, 0.0, 0.0
                inp = {"u": u, "u_true": u_true, "u_r0": mu, "loc": loc, "scale": scale, "band": band_map(c),
                       "x": x, "x_true": x_true, "x_r0": x_r0,
                       "ref_energy": float((np.abs(acq) ** 2).sum() + (np.abs(u_true) ** 2).sum())}
                recs.append(core.slice_stats(case_id=(role, seed, v), **inp))
                if keep is not None:
                    keep[(role, seed, v)] = inp
            if disk is not None:
                name = "stage1/%s_%d.json" % (role.replace("/", "_"), seed)
                disk.write(name, core.record_to_bytes({"records": recs}))
                recs = core.record_from_bytes(disk.read(name))["records"]
            for v, rec in zip(vols, recs):
                pop.add(role, seed, v, rec)
    return pop.seal()


# ================================================================ fixture 9
def _labels(cov="unresolved", **over):
    lab = {"ucrps": "unresolved", "mcrps": "unresolved", "energy": "unresolved", "rho_r0": "unresolved"}
    for c in stats.CELLS:
        lab["abscov_%s_%s" % c] = cov
        lab["width_%s_%s" % c] = "unresolved"
    for key, val in over.items():
        key = key.replace("__", ".")
        check(key in lab, "unknown measure in test: " + key)
        lab[key] = val
    return lab


def fixture_09(shared):
    I, D, U = "improvement supported", "deterioration supported", "direction unresolved"
    b, a, u, nc = "below0", "above0", "unresolved", "not computed"
    check(stats.nll_label((-0.2, -0.01)) == I and stats.nll_label((0.01, 0.2)) == D, "NLL labels")
    check(stats.nll_label((-0.2, 0.0)) == U and stats.nll_label((0.0, 0.2)) == U, "NLL boundary at 0")
    check(stats.psnr_label((0.01, 0.3)) == I and stats.psnr_label((-0.3, -0.01)) == D, "PSNR labels")
    check(stats.psnr_label((0.0, 0.3)) == U, "PSNR boundary at 0")
    check(stats.practical_label((0.51, 0.9)) == "supported", "practical size supported")
    check(stats.practical_label((0.1, 0.49)) == "ruled out at this interval level", "practical size ruled out")
    check(stats.practical_label((0.5, 0.9)) == "unresolved" and stats.practical_label((0.1, 0.5)) == "unresolved",
          "practical-size boundary at 0.5 dB (AM1-3 a)")
    check(stats.gap_pattern(I, U) and stats.gap_pattern(I, D) and not stats.gap_pattern(I, I)
          and not stats.gap_pattern(U, U), "gap pattern")
    rows = lambda lab, **kw: stats.matrix_rows(I, U, lab, **kw)["rows"]
    # every row, with all eight cells present (AM1-3 c, d)
    check(rows(_labels(cov=b, ucrps=b, mcrps=b)) == [1], "row 1")
    check(rows(_labels(cov=b, ucrps=b)) == [2], "row 2")
    check(rows(_labels(ucrps=b, mcrps=b)) == [1, 3], "rows 1 and 3")
    check(rows(_labels(cov=b, ucrps=b, mcrps=b, abscov_M_0__90=u)) == [1, 3], "row 3 from one M cell")
    check(rows(_labels(abscov_U_0__90=b, width_U_0__90=a)) == [4], "row 4")
    check(rows(_labels(cov=b, ucrps=b, mcrps=b, width_U_0__90=a)) == [], "a cell bought by width blocks row 1")
    check(rows(_labels(cov=b, mcrps=b, width_U_0__90=a)) == [4], "row 4 pairs the cell with its own space")
    check(rows(_labels(cov=b, ucrps=b, width_U_0__90=a)) == [2], "row 4 not triggered when CRPS of that space improves")
    check(rows(_labels(energy=b)) == [5], "row 5")
    check(rows(_labels(rho_r0=a)) == [6], "row 6")
    check(rows(_labels()) == [7], "row 7")
    check(rows(_labels(ucrps=b)) == [2, 3], "rows 2 and 3 co-occur")
    check(rows(_labels(mcrps=b)) == [3], "row 3 pairs CRPS and coverage within M only")
    check(rows(_labels(rho_r0=b)) == [7], "association below 0 is not an improvement")
    full = _labels(cov=b, ucrps=b, mcrps=b, energy=b, rho_r0=a)
    check(stats.matrix_rows(I, I, full)["rows"] == [8], "row 8")
    check(stats.matrix_rows(U, I, full)["rows"] == [9], "row 9")
    check(stats.matrix_rows(D, U, full)["rows"] == [], "NLL deterioration: no row")
    check(rows(_labels(), validity_ok=False) == [7, 10], "row 10 (validity)")
    check(stats.matrix_rows(U, I, full, validity_ok=False)["rows"] == [9, 10], "rows 9 and 10")
    # three outcomes per row (AM1-3 b, e): assigned / ruled out / unresolved
    r = stats.matrix_rows(I, U, _labels(cov=b, mcrps=b, abscov_U_0__90=nc))
    check(r["truth"][1] is False and all(row != 1 for row, _ in r["unresolved"]),
          "row 1 must be ruled out when U-CRPS does not improve, even with a coverage interval missing")
    r = stats.matrix_rows(I, U, _labels(cov=b, ucrps=b, mcrps=b, abscov_U_0__90=nc, width_U_0__90=a))
    check(r["truth"][1] is None and (1, ["abscov_U_0.90"]) in r["unresolved"] and 1 not in r["rows"]
          and 10 in r["rows"] and r["report"] == stats.UNRESOLVED_REPORT,
          "row 1 must be unresolved when both CRPS improve and a needed coverage interval is missing")
    r = stats.matrix_rows(I, U, _labels(cov=b, ucrps=b, mcrps=b, abscov_U_0__90=nc))
    check(r["truth"][1] is True and r["truth"][3] is None and r["rows"] == [1, 10],
          "row 1 is decided when the width does not rise in the cell whose coverage is missing; row 3 is not")
    r = stats.matrix_rows(I, U, _labels(rho_r0=nc))
    check(r["truth"][7] is None and r["truth"][6] is None and 7 not in r["rows"] and 10 in r["rows"]
          and r["report"] == stats.UNRESOLVED_REPORT and dict(r["unresolved"])[7] == ["rho_r0"],
          "row 7 must never be assigned while one of its measures is not computed")
    r = stats.matrix_rows(I, U, _labels(ucrps=b, rho_r0=nc))
    check(r["truth"][7] is False and r["truth"][6] is False and r["unresolved"] == [] and r["rows"] == [2, 3]
          and r["report"] is None, "rows 6 and 7 are ruled out by an improved U-CRPS whatever is missing")
    r = stats.matrix_rows(I, U, _labels(energy=nc))
    check(r["truth"][5] is None and r["truth"][7] is None and r["rows"] == [10], "energy not computed")
    r = stats.matrix_rows(I, U, _labels(ucrps=nc, mcrps=nc, energy=nc, rho_r0=nc, cov=nc))
    check(r["rows"] == [10] and [row for row, _ in r["unresolved"]] == [1, 2, 3, 5, 6, 7] and r["truth"][4] is False,
          "scores not computed but widths valid and not rising: row 4 ruled out, the other rows unresolved")
    none = {k: nc for k in _labels()}
    r = stats.matrix_rows(I, U, none)
    check(r["rows"] == [10] and [row for row, _ in r["unresolved"]] == [1, 2, 3, 4, 5, 6, 7],
          "nothing computed: every row unresolved, none assigned")
    check(stats._and(True, None) is None and stats._and(False, None) is False and stats._or(True, None) is True
          and stats._or(False, None) is None and stats._not(None) is None, "three-valued connectives")
    # deciding effects and Monte Carlo adequacy (AM1-3 g)
    stable = {m: "stable" for m in _labels()}
    mc = "Monte Carlo adequacy unresolved"
    lab1 = _labels(cov=b, ucrps=b, mcrps=b)
    check(rows(lab1, stability=stable) == [1], "all stable")
    check(rows(lab1, stability=dict(stable, energy=mc)) == [1], "energy is not a deciding effect of row 1")
    check(rows(lab1, stability=dict(stable, ucrps=mc)) == [1, 10], "co-primary endpoint unstable")
    check(rows(lab1, stability=dict(stable, **{"width_M_0.95": mc})) == [1, 10], "a cell of row 1 unstable")
    check(rows(_labels(cov=b, ucrps=b), stability=dict(stable, **{"width_M_0.95": mc})) == [2],
          "width is not a deciding effect of row 2")
    check(rows(_labels(energy=b), stability=dict(stable, energy=mc)) == [5, 10], "energy decides row 5")
    check(rows(_labels(), stability=dict(stable, rho_r0=mc)) == [7, 10], "association decides row 7")
    # cross-objective rows (§8.1)
    dg = {"S-GAP0": (I, U), "C-TIME0": (I, U), "C-TIME2": (I, I)}
    check(stats.cross_rows(dg, [u, u]) == ["X1", "X4"], "X1 and X4")
    check(stats.cross_rows(dict(dg, **{"C-TIME2": (I, U)}), [b]) == ["X2", "X3"], "X2 and X3")
    check(stats.cross_rows(dg, [u], [a]) == ["X1", "X5"], "X5")
    check(stats.cross_rows(dg, [nc, u]) == ["X1", "X4"], "a measure that is not computed is not a difference")
    # registered contrast table
    t1, t2 = stats.TIER1_CONTRASTS, stats.TIER2_CONTRASTS
    check(sorted(t1) == sorted(["S-GAP0", "S-CTL0", "C-TIME0", "S-CTL2", "C-TIME2", "S-GAP2", "C-OBJ", "C-OBJ@3",
                                "X-TIME", "X-CTL"]), "Tier 1 contrast names")
    check(len(t2) == 11 and all(abs(sum(w.values())) < 1e-15 for w in list(t1.values()) + list(t2.values())),
          "Tier 2 contrast count or weights not summing to zero")
    check(t1["S-GAP0"] == {"lam0/fin": 1.0, "lam0/sel": -1.0} and t1["C-OBJ@3"] == {"lam2/e3": 1.0, "lam0/e3": -1.0}
          and t1["X-TIME"] == {"lam2/fin": 1.0, "lam2/e3": -1.0, "lam0/fin": -1.0, "lam0/e3": 1.0}
          and t2["X-FIX(3→30)"] == {"lam2/e30": 1.0, "lam2/e3": -1.0, "lam0/e30": -1.0, "lam0/e3": 1.0},
          "contrast definitions")
    check(len(stats.registered_roles(False)) == 6 and len(stats.registered_roles(True)) == 10, "registered roles")
    # C-TIME computed directly; AM1-4: point estimates add for additive and nonlinear aggregates alike;
    # interval endpoints do not; a nonlinear aggregate is not a mean of per-volume values
    pop, idx = shared["pop"], shared["idx"]
    info = {}
    for m in ("ucrps", "abscov_U_0.90", "ssr", "rho_vol"):
        fn = stats.additive_effect if m == "ucrps" else stats.nonlinear_effect
        eff = {c: fn(pop, t1[c], m, "full", idx) for c in ("C-TIME0", "S-CTL0", "S-GAP0")}
        ok, dev = close(eff["C-TIME0"]["point"], eff["S-CTL0"]["point"] + eff["S-GAP0"]["point"], 1e-12, 1e-10)
        check(ok, "point identity TIME = CTL + GAP off by %.3g for %s" % (dev, m))
        gap = eff["C-TIME0"]["ci"][1] - (eff["S-CTL0"]["ci"][1] + eff["S-GAP0"]["ci"][1])
        check(abs(gap) > 1e-6, "interval endpoints unexpectedly additive for " + m)
        info["upper_endpoint_gap_" + m] = float(gap)
        if m == "abscov_U_0.90":
            pooled = eff["S-GAP0"]["point"]
    kc, pi = core.attainable(dict(core.LEVELS)["0.90"], 128)
    per_volume = []
    for role in ("lam0/fin", "lam0/sel"):
        vals = []
        for seed in pop.seeds:
            cv = pop.col(role, seed, lambda r: r["subsets"]["full"]["U"]["cov"]["0.90"][0])
            n = pop.col(role, seed, lambda r: r["subsets"]["full"]["U"]["cov"]["0.90"][1])
            vals.append(np.abs(cv / n - pi).mean())
        per_volume.append(np.mean(vals))
    shortcut = per_volume[0] - per_volume[1]
    check(abs(shortcut - pooled) > 1e-4, "per-volume shortcut unexpectedly equals the pooled absolute coverage error")
    info.update({"abscov_pooled": pooled, "abscov_per_volume_shortcut": float(shortcut)})
    # entry point on a small population with all ten roles (Tier 2 on)
    rng = rng_for(9)
    small = _population(rng, dict(ROLES, **TIER2_ROLES), n_vol=8)
    idx8 = stats.bootstrap_matrix(seed=FIXTURE_SEED, n_volumes=8, n_boot=400)
    res = stats.run_registered(small, idx8, tier2=True)
    check(sorted(res["contrasts"]) == sorted(list(t1) + list(t2)) and len(res["checkpoints"]) == 10,
          "entry point did not return every registered contrast and checkpoint")
    check(all(v["interpretation"].startswith("not assigned") for v in res["contrasts"].values()) and res["cross"] is None,
          "rows assigned without D-GAP labels")
    pt = lambda c, m: res["contrasts"][c]["effects"][(m, "full")]["point"]
    for m in ("mcrps", "energy", "ssr", "abscov_M_0.80"):
        ok, dev = close(pt("F-TRAJ2(3→30)", m), pt("F-TRAJ2(3→10)", m) + pt("F-TRAJ2(10→30)", m), 1e-12, 1e-10)
        check(ok, "fixed-epoch trajectory identity off by %.3g for %s" % (dev, m))
        ok, dev = close(pt("X-FIX(3→10)", m), pt("F-TRAJ2(3→10)", m) - pt("F-TRAJ0(3→10)", m), 1e-12, 1e-10)
        check(ok, "X-FIX is not F-TRAJ2 - F-TRAJ0 for %s (%.3g)" % (m, dev))
        ok, dev = close(pt("X-TIME", m), pt("C-TIME2", m) - pt("C-TIME0", m), 1e-12, 1e-10)
        check(ok, "X-TIME is not C-TIME2 - C-TIME0 for %s (%.3g)" % (m, dev))
    gap_all = {c: {"nll": I, "psnr_r0": U} for c in list(t1) + list(t2)}
    res2 = stats.run_registered(small, idx8, tier2=True, dgap=gap_all)
    check(all(isinstance(v["interpretation"], dict) for v in res2["contrasts"].values())
          and isinstance(res2["cross"], list), "rows not assigned with D-GAP labels")
    tier1_only = _population(rng, ROLES, n_vol=8)
    check(len(stats.run_registered(tier1_only, idx8)["contrasts"]) == 10, "Tier 1 entry point")
    expect_error("DIAG_UNC_POPULATION_MISMATCH", stats.run_registered, tier1_only, idx8, True)
    expect_error("DIAG_UNC_POPULATION_MISMATCH", stats.run_registered, tier1_only, shared["idx"])
    expect_error("DIAG_UNC_POPULATION_MISMATCH", stats.analyse_contrast, tier1_only, {"lam9/fin": 1.0}, idx8)
    return info


# ================================================================ fixture 10
def ref_rank(a):
    return (a[None, :] < a[:, None]).sum(axis=1) + ((a[None, :] == a[:, None]).sum(axis=1) + 1) / 2.0


def ref_spearman(a, b):
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(np.corrcoef(ref_rank(a), ref_rank(b))[0, 1])


class CountingGenerator:
    """Wraps a NumPy generator and records every method call made on it."""

    def __init__(self, seed):
        self.seed = seed
        self.calls = []
        self._gen = np.random.Generator(np.random.PCG64(seed))

    def __getattr__(self, name):
        attr = getattr(self._gen, name)
        if not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return attr(*args, **kwargs)
        return wrapped


def fixture_10(shared):
    rng = rng_for(10)
    pop, idx = shared["pop"], shared["idx"]
    gap0 = stats.TIER1_CONTRASTS["S-GAP0"]
    # nonlinear aggregates per replicate: vectorised against an explicit loop (ties, repeats, constants)
    a, b = np.round(rng.standard_normal(97), 1), rng.standard_normal(97)
    got = stats.boot_spearman(a, b, idx)
    ref = np.array([ref_spearman(a[idx[i]], b[idx[i]]) for i in range(400)])
    ok, dev = close(got[:400], ref, 1e-12, 1e-10)
    check(ok, "bootstrap Spearman differs from the loop by %.3g" % dev)
    check(np.isnan(stats.boot_spearman(np.ones(97), b, idx)).all(), "constant vector not undefined")
    two = np.zeros(97)
    two[0] = 1.0
    frac = np.isnan(stats.boot_spearman(two, b, idx)).mean()
    check(0.3 < frac < 0.45, "undefined-draw fraction for a near-constant vector: %.3f" % frac)
    # no stale counts: a different matrix in the same call sequence gives its own result (cache removed)
    idx_b = stats.bootstrap_matrix(seed=FIXTURE_SEED + 7)
    r_a1 = stats.boot_spearman(a, b, idx)
    r_b = stats.boot_spearman(a, b, idx_b)
    r_b_ref = np.array([ref_spearman(a[idx_b[i]], b[idx_b[i]]) for i in range(200)])
    ok, dev = close(r_b[:200], r_b_ref, 1e-12, 1e-10)
    check(ok and not np.array_equal(r_a1, r_b) and not hasattr(stats, "_COUNT_CACHE"), "stale bootstrap counts")
    work = idx.copy()
    r_w1 = stats.boot_spearman(a, b, work)
    work[:] = idx_b                      # same object, new contents
    check(np.array_equal(stats.boot_spearman(a, b, work), r_b) and np.array_equal(r_w1, r_a1),
          "counts reused after the matrix contents changed")
    # bootstrap reuse: one matrix, unchanged by use, identical results on reuse
    before = stats.matrix_identity(idx)
    e1 = stats.nonlinear_effect(pop, gap0, "ssr", "full", idx)
    e2 = stats.nonlinear_effect(pop, gap0, "ssr", "pre64", idx)
    e3 = stats.nonlinear_effect(pop, gap0, "ssr", "full", idx)
    check(stats.matrix_identity(idx) == before and e1 == e3 and e1 != e2, "bootstrap matrix reuse")
    # index-matrix identity gate (same call, fixture seed; the registered matrix is first generated in F1)
    registered = stats.matrix_identity(stats.bootstrap_matrix(seed=FIXTURE_SEED))
    check(stats.verify_matrix(stats.bootstrap_matrix(seed=FIXTURE_SEED), registered) == registered, "identity")
    check(stats.matrix_identity(idx.astype(">i8")) == before, "identity depends on host byte order")
    # AM1-1: the single-call requirement, verified with an instrumented generator
    made = []

    def factory(seed):
        made.append(CountingGenerator(seed))
        return made[-1]
    inst = stats.bootstrap_matrix(seed=FIXTURE_SEED, generator_factory=factory)
    check(len(made) == 1 and made[0].seed == FIXTURE_SEED, "generator not created exactly once with the seed")
    check(made[0].calls == [("integers", (0, 97), {"size": (10000, 97), "dtype": np.int64})],
          "generation is not exactly one integers call with the registered arguments: %r" % (made[0].calls,))
    check(np.array_equal(inst, idx), "instrumented generation differs from the plain call")
    # the registered arguments are the defaults; checked from the signature, without generating that matrix
    import inspect
    defaults = {k: v.default for k, v in inspect.signature(stats.bootstrap_matrix).parameters.items()}
    check((defaults["seed"], defaults["n_volumes"], defaults["n_boot"]) == (20261114, 97, 10000)
          and stats.DIAG_UNC_SEED == 20261114, "default arguments are not the registered ones")
    # AM1-1: refusals
    expect_error("DIAG_UNC_BOOTSTRAP_IDENTITY", stats.verify_matrix, idx.astype(np.int32), registered)
    expect_error("DIAG_UNC_BOOTSTRAP_IDENTITY", stats.verify_matrix, idx.astype(float), registered)
    expect_error("DIAG_UNC_BOOTSTRAP_IDENTITY", stats.verify_matrix, idx[:9999], registered)
    expect_error("DIAG_UNC_BOOTSTRAP_IDENTITY", stats.verify_matrix, idx.T.copy(), registered)
    expect_error("DIAG_UNC_BOOTSTRAP_IDENTITY", stats.verify_matrix, stats.bootstrap_matrix(seed=FIXTURE_SEED + 1),
                 registered)
    gen = np.random.Generator(np.random.PCG64(FIXTURE_SEED))
    gen.random()
    expect_error("DIAG_UNC_BOOTSTRAP_IDENTITY", stats.verify_matrix,
                 gen.integers(0, 97, size=(10000, 97), dtype=np.int64), registered)
    gen = np.random.Generator(np.random.PCG64(FIXTURE_SEED))
    by_column = np.stack([gen.integers(0, 97, size=10000, dtype=np.int64) for _ in range(97)], axis=1)
    check(not np.array_equal(by_column, idx), "column-wise multi-call generation unexpectedly identical")
    expect_error("DIAG_UNC_BOOTSTRAP_IDENTITY", stats.verify_matrix, by_column, registered)
    gen = np.random.Generator(np.random.PCG64(FIXTURE_SEED))
    row_split = np.concatenate([gen.integers(0, 97, size=(5000, 97), dtype=np.int64),
                                gen.integers(0, 97, size=(5000, 97), dtype=np.int64)])
    row_split_identical = bool(np.array_equal(row_split, idx))
    if row_split_identical:   # AM1-1: reported as an observation, not a failure
        check(stats.verify_matrix(row_split, registered) == registered, "byte-identical matrix refused")
        OBSERVATIONS.append("fixture 10: a row-major split of the call into two calls returns a byte-identical matrix "
                            "(NumPy %s); the identity hash cannot distinguish it (223 AM1-1)" % np.__version__)
    else:
        expect_error("DIAG_UNC_BOOTSTRAP_IDENTITY", stats.verify_matrix, row_split, registered)
    # undefined bootstrap draws through the effect function: excluded and counted (<= 1%), or interval not computed
    und = {}
    for label, n_distinct in (("few", 5), ("many", 1)):
        fake = stats.Population(pop.volume_ids)
        for seed in pop.seeds:
            for vi, v in enumerate(pop.volume_ids):
                rec = copy.deepcopy(pop.records[("lam0/fin", seed)][vi])
                rec["subsets"]["full"]["sd_slice"] = 1.0 if vi < n_distinct else 0.0
                fake.add("lam0/fin", seed, v, rec)
        fake.seal()
        eff = stats.nonlinear_effect(fake, {"lam0/fin": 1.0}, "rho_vol", "full", idx)
        const = np.array([(idx[i] >= n_distinct).all() for i in range(idx.shape[0])])
        check(eff["undefined_draws"] == int(const.sum()), "undefined-draw count (%s)" % label)
        und[label] = eff
    check(0 < und["few"]["undefined_draws"] <= 100 and und["few"]["ci"] is not None, "<= 1% undefined draws not excluded")
    check(und["many"]["undefined_draws"] > 100 and und["many"]["ci"] is None and und["many"]["label"] == "not computed"
          and und["many"]["point"] is not None, "> 1% undefined draws: interval must be 'not computed'")
    # input-validation refusals (AM1-2), bank-hash and 128-decode refusals
    inp, _ = make_inputs(rng)
    for key in ("u", "u_true", "x", "x_r0", "loc"):
        bad = dict(inp, **{key: inp[key].copy()})
        bad[key].flat[1] = np.nan
        expect_error("DIAG_UNC_NONFINITE", core.slice_stats, **bad)
    expect_error("DIAG_UNC_NONFINITE", core.slice_stats, **dict(inp, ref_energy=float("inf")))
    expect_error("DIAG_UNC_INPUT_INVALID", core.slice_stats, **make_inputs(rng, k=64)[0])
    expect_error("DIAG_UNC_INPUT_INVALID", core.slice_stats, **make_inputs(rng, k=127)[0])
    expect_error("DIAG_UNC_INPUT_INVALID", core.slice_stats, **dict(inp, u_true=inp["u_true"][:-1]))
    expect_error("DIAG_UNC_INPUT_INVALID", core.slice_stats, **dict(inp, band=np.full(12, 5)))
    expect_error("DIAG_UNC_INPUT_INVALID", core.slice_stats, **dict(inp, x=inp["x"][:100]))
    expect_error("DIAG_UNC_INPUT_INVALID", core.record_from_bytes, b"{not json")
    expect_error("DIAG_UNC_REPRODUCTION_MISMATCH", stats.verify_hash, "0" * 64, "f" * 64, "primary bank")
    stats.verify_hash("a" * 64, "a" * 64, "primary bank")
    expect_error("DIAG_UNC_POPULATION_MISMATCH", stats.Population, [2, 1])
    p = stats.Population([0, 1])
    rec = core.slice_stats(**inp)
    expect_error("DIAG_UNC_POPULATION_MISMATCH", p.add, "r", 0, 1, rec)
    p.add("r", 0, 0, rec)
    expect_error("DIAG_UNC_POPULATION_MISMATCH", p.seal)
    expect_error("DIAG_UNC_POPULATION_MISMATCH", p.col, "r", 0, lambda r: 0.0)
    unknown_code_refused = False
    try:
        DiagUncError("DIAG_UNC_SOMETHING_ELSE", "x")
    except ValueError:
        unknown_code_refused = True
    check(unknown_code_refused, "unknown failure code accepted")
    check("DIAG_UNC_INPUT_INVALID" in core.FATAL_CODES and len(core.FATAL_CODES) == 7, "typed failure list (AM1-2)")
    return {"fixture_seed_matrix_sha256": registered, "row_major_split_identical": row_split_identical,
            "undefined_fraction_near_constant": float(frac),
            "undefined_draws": {k: v["undefined_draws"] for k, v in und.items()}}


# ================================================================ fixture 12
SUBSETS = {"full": np.arange(128), "pre16": np.arange(16), "pre32": np.arange(32), "pre64": np.arange(64),
           "halfA": np.arange(0, 64), "halfB": np.arange(64, 128)}
LEVEL_NAMES = ("0.50", "0.80", "0.90", "0.95")


def ref_slice(inp, idx):
    """Full-array reference for one slice and subset: direct formulas, independent of diag_unc_core.

    Ranks come from binary search in the sorted samples; tie masses are added one observation at a time.
    """
    ks = len(idx)
    kcs = EXPECTED_KC[ks]
    u, x = inp["u"][idx], inp["x"][idx]
    us = (np.concatenate([u.real, u.imag], axis=1) - inp["loc"]) / inp["scale"]
    ys = (np.concatenate([inp["u_true"].real, inp["u_true"].imag]) - inp["loc"]) / inp["scale"]
    t, m, m0 = np.abs(inp["x_true"]), np.abs(x), np.abs(inp["x_r0"])
    out = {}
    for space, s, y in (("U", us, ys), ("M", m, t)):
        srt = np.sort(s, axis=0)
        n = s.shape[1]
        less = np.array([np.searchsorted(srt[:, j], y[j], side="left") for j in range(n)])
        equal = np.array([np.searchsorted(srt[:, j], y[j], side="right") for j in range(n)]) - less
        hist = np.zeros(ks + 1)
        for j in range(n):
            for bin_ in range(less[j], less[j] + equal[j] + 1):
                hist[bin_] += 1.0 / (equal[j] + 1)
        out[space + "_hist"] = hist
        out[space + "_ties"] = int((equal > 0).sum())
        midrank = less + 0.5 * equal
        for name, kc in zip(LEVEL_NAMES, kcs):
            if kc == 0:
                out["cov_%s_%s" % (space, name)] = None
                continue
            out["cov_%s_%s" % (space, name)] = (int(((midrank >= kc) & (midrank <= ks - kc)).sum()), n,
                                                 float(np.mean(srt[ks - kc] - srt[kc - 1])))
        out["sd_" + space] = np.sqrt(((s - s.mean(axis=0)) ** 2).sum(axis=0) / (ks - 1))
        out["crps_" + space] = np.abs(s - y).mean(axis=0) - np.abs(s[:, None] - s[None]).sum(axis=(0, 1)) / (
            2.0 * ks * (ks - 1))
    band2 = np.concatenate([inp["band"], inp["band"]])
    out["band_crps"] = [float(np.mean(out["crps_U"][band2 == b])) for b in range(3)]
    out["energy"] = float(np.linalg.norm(us - ys, axis=1).mean()
                          - np.linalg.norm(us[:, None] - us[None], axis=2).sum() / (2.0 * ks * (ks - 1)))
    xbar = x.mean(axis=0)
    out["sig2"] = float((np.abs(x - xbar) ** 2).sum() / x.shape[1] / (ks - 1))
    out["e2"] = float((np.abs(xbar - inp["x_true"]) ** 2).sum() / x.shape[1])
    e0, e1 = np.abs(m0 - t), np.abs(np.abs(xbar) - t)
    out["rho_r0"], out["rho_r1"] = ref_spearman(out["sd_M"], e0), ref_spearman(out["sd_M"], e1)
    out["sd_slice"], out["mse_r0"] = float(np.mean(out["sd_M"])), float(np.mean((m0 - t) ** 2))
    order = np.lexsort((np.arange(len(t)), out["sd_M"]))
    out["deciles"] = np.array([(out["sd_M"][p].sum(), e0[p].sum(), e1[p].sum(), len(p))
                               for p in np.array_split(order, 10)])
    ubar = u.mean(axis=0)
    out["band_err_r1"] = [float((np.abs(ubar - inp["u_true"]) ** 2)[inp["band"] == b].sum() / inp["ref_energy"])
                          for b in range(3)]
    return out


def ref_slice_group(inp, idx):
    """Fair scores of one group of 16 by direct formulas: (U-CRPS, M-CRPS, energy score)."""
    ks = len(idx)
    u, x = inp["u"][idx], inp["x"][idx]
    us = (np.concatenate([u.real, u.imag], axis=1) - inp["loc"]) / inp["scale"]
    ys = (np.concatenate([inp["u_true"].real, inp["u_true"].imag]) - inp["loc"]) / inp["scale"]
    t, m = np.abs(inp["x_true"]), np.abs(x)
    crps = lambda s, y: float(np.mean(np.abs(s - y).mean(axis=0)
                                      - np.abs(s[:, None] - s[None]).sum(axis=(0, 1)) / (2.0 * ks * (ks - 1))))
    energy = float(np.linalg.norm(us - ys, axis=1).mean()
                   - np.linalg.norm(us[:, None] - us[None], axis=2).sum() / (2.0 * ks * (ks - 1)))
    return crps(us, ys), crps(m, t), energy


def ref_sign(ci):
    if ci is None:
        return "not computed"
    return "below0" if ci[1] < 0 else ("above0" if ci[0] > 0 else "unresolved")


class Tally:
    """Largest deviation and pass/fail per acceptance-table row."""

    def __init__(self):
        self.rows = {i: {"n": 0, "max_dev": 0.0, "fail": []} for i in range(1, 10)}

    def exact(self, row, a, b, what):
        self.rows[row]["n"] += 1
        if a != b:
            self.rows[row]["fail"].append("%s: %r != %r" % (what, a, b))

    def num(self, row, a, b, what, level):
        atol, rtol = (ATOL_SLICE, RTOL_SLICE) if level == "slice" else (ATOL_POP, RTOL_POP)
        ok, dev = close(a, b, atol, rtol)
        r = self.rows[row]
        r["n"] += 1
        r["max_dev"] = max(r["max_dev"], dev)
        if not ok:
            r["fail"].append("%s: deviation %.3g" % (what, dev))


def fixture_12(shared):
    """Streamed two-stage pipeline (stage 1 -> guarded files on disk -> stage 2) against a full-array reference."""
    pop, idx, raw = shared["pop"], shared["idx"], shared["raw"]
    roles, seeds, vols = list(ROLES), list(pop.seeds), pop.volume_ids
    subs = list(SUBSETS)
    tally = Tally()
    add_names = (["ucrps", "mcrps", "energy", "sd_U", "sd_M", "rho_r0", "rho_r1", "band_crps_low", "band_crps_mid",
                  "band_crps_high"] + ["width_%s_%s" % (s, n) for s in "UM" for n in LEVEL_NAMES])
    shape = (len(roles), len(seeds), len(subs), len(vols))
    add = {m: np.full(shape, np.nan) for m in add_names}
    cov = np.full((2, 4) + shape, np.nan)
    ncov = np.full((2, 4) + shape, np.nan)
    sig2, e2, sdv, mse = (np.zeros(shape) for _ in range(4))
    dec = np.zeros(shape[:3] + (10, 4))
    hist_full = {(ri, sp): np.zeros(129) for ri in range(len(roles)) for sp in "UM"}
    tie_tot = np.zeros((len(roles), 2))
    grp = np.zeros((3, len(roles), len(seeds), 8, len(vols)))
    n_tied = {"U": 0, "M": 0}
    # ---- rows 1-4: per-slice quantities, reloaded streamed record against the full-array reference
    for ri, role in enumerate(roles):
        for si, seed in enumerate(seeds):
            for vi, v in enumerate(vols):
                inp, rec = raw[(role, seed, v)], pop.records[(role, seed)][vi]
                tally.exact(1, tuple(rec["case_id"]), (role, seed, v), "case identity after reload")
                for bi, sub in enumerate(subs):
                    ref, got = ref_slice(inp, SUBSETS[sub]), rec["subsets"][sub]
                    tag = "%s/%s/%s/%s" % (role, seed, v, sub)
                    for sp_i, space in enumerate("UM"):
                        for li, name in enumerate(LEVEL_NAMES):
                            r_c, g_c = ref["cov_%s_%s" % (space, name)], got[space]["cov"][name]
                            tally.exact(1, g_c is None, r_c is None, tag + " resolvable " + name)
                            if r_c is None:
                                continue
                            tally.exact(1, (g_c[0], g_c[1]), (r_c[0], r_c[1]), tag + " counts " + space + name)
                            tally.num(3, g_c[2], r_c[2], tag + " width " + space + name, "slice")
                            cov[sp_i, li, ri, si, bi, vi], ncov[sp_i, li, ri, si, bi, vi] = r_c[0], r_c[1]
                            add["width_%s_%s" % (space, name)][ri, si, bi, vi] = r_c[2]
                        tally.exact(1, got[space]["ties"], ref[space + "_ties"], tag + " ties " + space)
                        tally.num(2, got[space]["hist"], ref[space + "_hist"], tag + " hist " + space, "slice")
                        tally.num(2, got[space]["hist"].sum(), got[space]["n"], tag + " hist mass " + space, "slice")
                        tally.num(3, got["sd_mean"][space], np.mean(ref["sd_" + space]), tag + " sd " + space, "slice")
                        if sub == "full":
                            hist_full[(ri, space)] += ref[space + "_hist"]
                            tie_tot[ri, sp_i] += ref[space + "_ties"]
                            n_tied[space] += ref[space + "_ties"]
                    for g_v, r_v, nm in ((got["ucrps"], np.mean(ref["crps_U"]), "ucrps"),
                                         (got["mcrps"], np.mean(ref["crps_M"]), "mcrps"),
                                         (got["energy"], ref["energy"], "energy"),
                                         (got["band_crps"], ref["band_crps"], "band crps"),
                                         (got["band_err_r1"], ref["band_err_r1"], "band error")):
                        tally.num(3, g_v, r_v, tag + " " + nm, "slice")
                    for nm in ("sig2", "e2", "rho_r0", "rho_r1", "sd_slice", "mse_r0", "deciles"):
                        tally.num(4, got[nm], ref[nm], tag + " " + nm, "slice")
                    tally.exact(4, math.isnan(got["rho_r0"]), math.isnan(ref["rho_r0"]), tag + " rho flag")
                    add["ucrps"][ri, si, bi, vi], add["mcrps"][ri, si, bi, vi] = np.mean(ref["crps_U"]), np.mean(ref["crps_M"])
                    add["energy"][ri, si, bi, vi] = ref["energy"]
                    add["sd_U"][ri, si, bi, vi], add["sd_M"][ri, si, bi, vi] = np.mean(ref["sd_U"]), np.mean(ref["sd_M"])
                    add["rho_r0"][ri, si, bi, vi], add["rho_r1"][ri, si, bi, vi] = ref["rho_r0"], ref["rho_r1"]
                    for b, bn in enumerate(("low", "mid", "high")):
                        add["band_crps_" + bn][ri, si, bi, vi] = ref["band_crps"][b]
                    sig2[ri, si, bi, vi], e2[ri, si, bi, vi] = ref["sig2"], ref["e2"]
                    sdv[ri, si, bi, vi], mse[ri, si, bi, vi] = ref["sd_slice"], ref["mse_r0"]
                    dec[ri, si, bi] += ref["deciles"]
                for g in range(8):
                    rg = ref_slice_group(inp, np.arange(16 * g, 16 * g + 16))
                    gg = rec["groups"]["g%d" % g]
                    for mi, nm in enumerate(("ucrps", "mcrps", "energy")):
                        tally.num(3, gg[nm], rg[mi], "%s/%s/%s/g%d %s" % (role, seed, v, g, nm), "slice")
                        grp[mi, ri, si, g, vi] = rg[mi]
    tally.exact(1, n_tied["U"] > 0 and n_tied["M"] > 0, True, "the fixture data must contain ties in both spaces")
    # ---- reference population level: explicit loop over the bootstrap replicates
    n_boot = idx.shape[0]
    kvec = np.array([len(SUBSETS[s]) for s in subs], dtype=float)
    add_stack = np.stack([add[m] for m in add_names])
    b_add = np.empty((n_boot,) + add_stack.shape[:-1])
    b_cov = np.empty((n_boot,) + cov.shape[:-1])
    b_ssr = np.empty((n_boot,) + shape[:3])
    b_rho = np.empty((n_boot,) + shape[:3])
    for i in range(n_boot):
        j = idx[i]
        b_add[i] = add_stack[..., j].mean(axis=-1)
        b_cov[i] = cov[..., j].sum(axis=-1) / ncov[..., j].sum(axis=-1)
        b_ssr[i] = np.sqrt((kvec + 1) / kvec * sig2[..., j].mean(axis=-1)) / np.sqrt(e2[..., j].mean(axis=-1))
        for ri in range(len(roles)):
            for si in range(len(seeds)):
                for bi in range(len(subs)):
                    b_rho[i, ri, si, bi] = ref_spearman(sdv[ri, si, bi, j], mse[ri, si, bi, j])
    pis = np.full((4, len(subs)), np.nan)
    for bi, sub in enumerate(subs):
        k = len(SUBSETS[sub])
        for li, kc in enumerate(EXPECTED_KC[k]):
            if kc:
                pis[li, bi] = (k + 1 - 2 * kc) / (k + 1)
    p_cov = cov.sum(axis=-1) / ncov.sum(axis=-1)                          # (2,4,R,S,Sub)
    p_abs, b_abs = np.abs(p_cov - pis[None, :, None, None, :]), np.abs(b_cov - pis[None, None, :, None, None, :])
    p_ssr = np.sqrt((kvec + 1) / kvec * sig2.mean(axis=-1)) / np.sqrt(e2.mean(axis=-1))
    p_rho = np.array([[[ref_spearman(sdv[ri, si, bi], mse[ri, si, bi]) for bi in range(len(subs))]
                       for si in range(len(seeds))] for ri in range(len(roles))])

    def ref_effect(weights, point_rs, boot_rs):
        """point_rs: (R, S); boot_rs: (B, R, S) -> point, per-seed, interval, label, undefined draws."""
        w = np.array([weights.get(r, 0.0) for r in roles])
        per_seed = (w[:, None] * point_rs).sum(axis=0)
        boot = (w[None, :, None] * boot_rs).sum(axis=1).mean(axis=1)
        if np.isnan(per_seed).any():
            return None, None, None, "not computed", int(np.isnan(boot).sum())
        und = int(np.isnan(boot).sum())
        ci = None if und > 0.01 * n_boot else tuple(np.percentile(boot[~np.isnan(boot)], [2.5, 97.5]))
        return float(per_seed.mean()), per_seed, ci, ref_sign(ci), und

    def compare(row_pop, row_lab, got, ref, what):
        point, per_seed, ci, label, und = ref
        tally.exact(row_lab, got["label"], label, what + " label")
        tally.exact(row_lab, got["undefined_draws"], und, what + " undefined draws")
        tally.exact(row_pop, got["point"] is None, point is None, what + " computed")
        if point is not None:
            tally.num(row_pop, got["point"], point, what + " point", "pop")
            tally.num(row_pop, got["per_seed"], per_seed, what + " per-seed", "pop")
        tally.exact(row_pop, got["ci"] is None, ci is None, what + " interval computed")
        if ci is not None:
            tally.num(row_pop, got["ci"], ci, what + " interval", "pop")

    def all_refs(weights):
        refs = {}
        for bi, sub in enumerate(subs):
            for mi, m in enumerate(add_names):
                if np.isnan(add[m][:, :, bi]).all():
                    continue
                refs[(m, sub)] = ref_effect(weights, add[m][:, :, bi].mean(axis=-1), b_add[:, mi, :, :, bi])
            refs[("ssr", sub)] = ref_effect(weights, p_ssr[:, :, bi], b_ssr[:, :, :, bi])
            refs[("rho_vol", sub)] = ref_effect(weights, p_rho[:, :, bi], b_rho[:, :, :, bi])
            for sp_i, space in enumerate("UM"):
                for li, name in enumerate(LEVEL_NAMES):
                    if not np.isnan(pis[li, bi]):
                        refs[("abscov_%s_%s" % (space, name), sub)] = ref_effect(
                            weights, p_abs[sp_i, li, :, :, bi], b_abs[:, sp_i, li, :, :, bi])
        return refs

    # ---- everything below comes from the single entry point
    I, U = "improvement supported", "direction unresolved"
    dgap = {c: {"nll": I, "psnr_r0": U} for c in stats.TIER1_CONTRASTS}     # synthetic 'gap pattern' labels
    result = stats.run_registered(pop, idx, dgap=dgap)
    tally.exact(6, sorted(result["contrasts"]), sorted(stats.TIER1_CONTRASTS), "registered contrast set")
    # ---- row 5: per checkpoint and seed (single-role weights), calibration, pooled deciles and histograms
    nonlinear = lambda m: m.startswith(("abscov", "ssr", "rho_vol"))
    for ri, role in enumerate(roles):
        got_all, refs = stats.analyse_contrast(pop, {role: 1.0}, idx), all_refs({role: 1.0})
        tally.exact(5, sorted(got_all["effects"]), sorted(refs), role + " measure set")
        for key in refs:
            if nonlinear(key[0]):
                compare(5, 8, got_all["effects"][key], refs[key], "%s %s/%s" % (role, key[0], key[1]))
        ck = result["checkpoints"][role]
        tot = dec[ri, :, 0].sum(axis=0)
        tally.num(5, ck["deciles"], tot[:, :3] / tot[:, 3:4], role + " pooled deciles", "pop")
        for sp_i, space in enumerate("UM"):
            tally.num(5, ck["histogram"][space], hist_full[(ri, space)], role + " pooled histogram " + space, "pop")
            for li, name in enumerate(LEVEL_NAMES):
                cal = ck["calibration"][(space, name)]
                signed = (p_cov[sp_i, li, ri, :, 0] - pis[li, 0]).mean()
                bsig = (b_cov[:, sp_i, li, ri, :, 0] - pis[li, 0]).mean(axis=1)
                ci = tuple(np.percentile(bsig, [2.5, 97.5]))
                tally.num(5, [cal["signed"], cal["coverage"]], [signed, signed + pis[li, 0]], role + " coverage", "pop")
                tally.num(5, cal["ci"], ci, role + " signed coverage interval", "pop")
                lab = {"below0": "over-confident supported", "above0": "under-confident supported"}.get(
                    ref_sign(ci), "miscalibration not established")
                tally.exact(8, cal["label"], lab, "%s calibration label %s %s" % (role, space, name))
                ref_refl = "within" if (-0.05 <= ci[0] and ci[1] <= 0.05) else (
                    "outside" if (ci[0] > 0.05 or ci[1] < -0.05) else "straddles")
                tally.exact(8, cal["reference"], ref_refl, "%s reference label %s %s" % (role, space, name))
                frac = tie_tot[ri, sp_i] / ncov[sp_i, li, ri, :, 0].sum()
                tally.num(5, cal["tie_fraction"], frac, role + " tie fraction " + space, "pop")
                tally.exact(8, cal["ties_material"], bool(frac > 0.001), role + " tie flag " + space)
    # ---- rows 6-8: the ten registered Tier 1 contrasts (paired contrasts and interactions)
    summary = {}
    for cname, weights in stats.TIER1_CONTRASTS.items():
        got_all, refs = result["contrasts"][cname], all_refs(weights)
        tally.exact(6, sorted(got_all["effects"]), sorted(refs), cname + " measure set")
        for key, ref in refs.items():
            row = 6 if key[1] == "full" else 7
            compare(row, 8, got_all["effects"][key], ref, "%s %s/%s" % (cname, key[0], key[1]))
        ref_stab = {}
        for m in sorted({k[0] for k in refs}):
            vals = [refs[(m, s)][3] if (m, s) in refs else None for s in ("pre64", "full", "halfA", "halfB")]
            ref_stab[m] = "stable" if None not in vals and len(set(vals)) == 1 else "Monte Carlo adequacy unresolved"
            tally.exact(8, got_all["stability"][m], ref_stab[m], "%s stability %s" % (cname, m))
        w = np.array([weights.get(r, 0.0) for r in roles])
        for mi, m in enumerate(("ucrps", "mcrps", "energy")):
            eff = (w[:, None, None] * grp[mi].mean(axis=-1)).sum(axis=0).mean(axis=0)
            se = math.sqrt(((eff - eff.mean()) ** 2).sum() / 7.0) / math.sqrt(8.0)
            bm = got_all["batch_means"][m]
            tally.num(7, bm["effects"] + [bm["mean"], bm["se"]], list(eff) + [eff.mean(), se], cname + " batch " + m, "pop")
        ref_interp = stats.matrix_rows(I, U, {m: refs[(m, "full")][3] for (m, s) in refs if s == "full"},
                                       stability=ref_stab)
        got_interp = got_all["interpretation"]
        for field in ("rows", "unresolved", "deciding", "report"):
            tally.exact(8, got_interp[field], ref_interp[field], "%s interpretation %s" % (cname, field))
        summary[cname] = {"rows": got_interp["rows"], "ucrps": got_all["effects"][("ucrps", "full")]["point"],
                          "ucrps_ci": got_all["effects"][("ucrps", "full")]["ci"]}
    xt = [all_refs(stats.TIER1_CONTRASTS["X-TIME"])[(m, "full")][3] for m in stats.PROBABILISTIC]
    tally.exact(8, result["cross"], stats.cross_rows({c: (I, U) for c in ("S-GAP0", "C-TIME0", "C-TIME2")}, xt),
                "cross-objective rows")
    # ---- row 9: write guard — ceiling, reserve, deleted intermediates, restart, interrupted write,
    #      several active guard objects and processes on one directory
    tmp = tempfile.mkdtemp(prefix="dunc_fix12_")
    led, per = guard.LEDGER_BYTES, guard.WriteGuard.charge

    def refused(fn, code, what):
        try:
            fn()
            tally.exact(9, "no error", code, what)
        except DiagUncError as exc:
            tally.exact(9, exc.code, code, what)

    def sub(name):
        path = os.path.join(tmp, name)
        os.mkdir(path)
        return path

    def tree(path):
        """Every entry under path with its kind and, for files, its bytes."""
        out = []
        for dirpath, dirnames, filenames in os.walk(path):
            rel = os.path.relpath(dirpath, path)
            out.extend((os.path.join(rel, d), "dir", b"") for d in dirnames)
            for name in filenames:
                with open(os.path.join(dirpath, name), "rb") as fh:
                    out.append((os.path.join(rel, name), "file", fh.read()))
        return sorted(out)

    def refused_unchanged(path, what):
        """A guard start that must be refused and must leave the directory tree exactly as it was."""
        snapshot = tree(path)
        refused(lambda: guard.WriteGuard(path, ceiling=10 ** 6, reserve=0), "DIAG_UNC_INPUT_INVALID", what)
        tally.exact(9, tree(path) == snapshot, True, what + ": directory changed by the refusal")

    try:
        tally.exact(9, per(400), 400 + guard.SIDECAR_BYTES + guard.LEDGER_REWRITE_BYTES, "charge per write")
        d1 = sub("ceiling")
        g = guard.WriteGuard(d1, ceiling=led + 1500, reserve=0)
        g.write("a.bin", b"x" * 400)
        g.delete("a.bin")
        tally.exact(9, g.written, led + per(400), "cumulative count after delete")
        refused(lambda: g.write("b.bin", b"x" * 800), "DIAG_UNC_RESOURCE_LIMIT", "ceiling with deleted intermediate")
        tally.exact(9, os.path.exists(os.path.join(d1, "b.bin")), False, "file created despite the ceiling")
        tally.exact(9, g.written, led + per(400), "refused write was charged")
        g.write("c.bin", b"x" * 400)
        tally.exact(9, g.read("c.bin"), b"x" * 400, "read back with sidecar check")
        # restart: a new guard on the same directory continues the count
        g_restart = guard.WriteGuard(d1, ceiling=led + 1500, reserve=0)
        tally.exact(9, g_restart.written, led + 2 * per(400), "count after restart")
        refused(lambda: g_restart.write("d.bin", b"x" * 200), "DIAG_UNC_RESOURCE_LIMIT", "ceiling after restart")
        tally.exact(9, os.path.exists(os.path.join(d1, "d.bin")), False, "file created after restart despite the ceiling")
        # independent control: the same sequence without the restart is refused at the same point
        gc = guard.WriteGuard(sub("control"), ceiling=led + 1500, reserve=0)
        gc.write("a.bin", b"x" * 400)
        gc.delete("a.bin")
        gc.write("c.bin", b"x" * 400)
        refused(lambda: gc.write("d.bin", b"x" * 200), "DIAG_UNC_RESOURCE_LIMIT", "ceiling, control without restart")
        gc.write("e.bin", b"x" * (1500 - 2 * per(400) - per(0)))        # exactly to the ceiling is allowed
        tally.exact(9, gc.written, led + 1500, "write exactly to the ceiling")
        refused(lambda: gc.write("f.bin", b""), "DIAG_UNC_RESOURCE_LIMIT", "empty file beyond the ceiling")
        # reserve, with a controlled free-space reading
        free = [led + 1500]
        g2 = guard.WriteGuard(sub("reserve"), ceiling=10 ** 9, reserve=1000, free_fn=lambda: free[0])
        g2.write("d.bin", b"x" * 400)
        refused(lambda: g2.write("e.bin", b"x" * (500 + led)), "DIAG_UNC_RESOURCE_LIMIT", "reserve")
        tally.exact(9, os.path.exists(os.path.join(g2.root, "e.bin")), False, "file created despite the reserve")
        free[0] = 10 ** 6
        g2.write("e.bin", b"x" * (500 + led))                           # the same write passes once space is free
        refused(lambda: g2.write("d.bin", b"y"), "DIAG_UNC_INPUT_INVALID", "overwrite")
        refused(lambda: g2.write("../x.bin", b"y"), "DIAG_UNC_INPUT_INVALID", "path outside the root")
        refused(lambda: g2.write(guard.LEDGER_NAME, b"y"), "DIAG_UNC_INPUT_INVALID", "write to the ledger name")
        refused(lambda: g2.write(guard.LOCK_NAME, b"y"), "DIAG_UNC_INPUT_INVALID", "write to the lock name")
        refused(lambda: g2.write("x" + guard.PART_SUFFIX, b"y"), "DIAG_UNC_INPUT_INVALID", "write to a partial-file name")
        refused(lambda: guard.WriteGuard(os.path.join(tmp, "missing")), "DIAG_UNC_INPUT_INVALID", "missing root")
        # ---- two guard objects active on one directory (reviewer's counterexample): the second write is refused
        d_two = sub("two_guards")
        ga = guard.WriteGuard(d_two, ceiling=900, reserve=0)
        gb = guard.WriteGuard(d_two, ceiling=900, reserve=0)
        ga.write("a.bin", b"x" * 350)
        refused(lambda: gb.write("b.bin", b"x" * 350), "DIAG_UNC_RESOURCE_LIMIT", "second active guard over the ceiling")
        tally.exact(9, os.path.exists(os.path.join(d_two, "b.bin")), False, "second guard wrote despite the ceiling")
        tally.exact(9, (ga.written, gb.written), (led + per(350), led + per(350)), "both guards read one ledger")
        # interleaved writes from two active guards are all counted
        d_il = sub("interleaved")
        ga, gb = guard.WriteGuard(d_il, ceiling=10 ** 6, reserve=0), guard.WriteGuard(d_il, ceiling=10 ** 6, reserve=0)
        for i in range(5):
            ga.write("a%d.bin" % i, b"x" * 100)
            gb.write("b%d.bin" % i, b"y" * 200)
        tally.exact(9, ga.written, led + 5 * per(100) + 5 * per(200), "interleaved writes of two guards")
        refused(lambda: gb.write("a0.bin", b"z"), "DIAG_UNC_INPUT_INVALID", "overwrite of the other guard's file")
        # overlapping guards with an interrupted write and a restart
        real_fsync, calls = guard._fsync, [0]

        def failing_fsync(fh):
            calls[0] += 1
            if calls[0] == 2:           # 1 = ledger update, 2 = the data file
                raise OSError("simulated interruption")
            real_fsync(fh)
        guard._fsync = failing_fsync
        try:
            ga.write("big.bin", b"z" * 5000)
            tally.exact(9, "no error", "OSError", "interrupted write")
        except OSError:
            tally.exact(9, "OSError", "OSError", "interrupted write")
        finally:
            guard._fsync = real_fsync
        expect = led + 5 * per(100) + 5 * per(200) + per(5000)
        tally.exact(9, sorted(n for n in os.listdir(d_il) if "big" in n or n.endswith(guard.PART_SUFFIX)), [],
                    "files left by the interrupted write")
        tally.exact(9, (ga.written, gb.written), (expect, expect), "interrupted write stays counted for both guards")
        gb.write("after.bin", b"q" * 10)
        tally.exact(9, guard.WriteGuard(d_il, ceiling=10 ** 6, reserve=0).written, expect + per(10),
                    "count after interruption, a further write and a restart")
        ga.write("big.bin", b"z" * 5000)                                # the name is free again
        # ---- several processes on one directory: no lost update, and the ceiling holds exactly
        import multiprocessing
        ctx = multiprocessing.get_context("fork")
        for label, ceiling, expected_ok in (("free", 10 ** 9, 100), ("capped", led + 30 * per(100) + 10, 30)):
            d_mp = sub("procs_" + label)
            if label == "capped":       # 'free': the four processes also race to create the ledger in a new directory
                guard.WriteGuard(d_mp, ceiling=ceiling, reserve=0)
            queue = ctx.Queue()
            procs = [ctx.Process(target=_guard_worker, args=(d_mp, ceiling, "p%d" % i, 25, 100, queue)) for i in range(4)]
            for p in procs:
                p.start()
            results = [queue.get(timeout=120) for _ in procs]
            for p in procs:
                p.join(timeout=120)
            ok = sum(r[0] for r in results)
            tally.exact(9, (ok, sum(r[1] for r in results), sum(r[2] for r in results)), (expected_ok, 100 - expected_ok, 0),
                        "%s: writes accepted, refused, other errors across 4 processes" % label)
            tally.exact(9, guard.WriteGuard(d_mp, ceiling=ceiling, reserve=0).written, led + expected_ok * per(100),
                        "%s: ledger after 4 concurrent processes" % label)
            tally.exact(9, len([n for n in os.listdir(d_mp) if n.endswith(".bin")]), expected_ok,
                        "%s: files on disk" % label)
        # ---- a guard starting while another guard is in the middle of a write must wait, and must not remove
        #      the partial file of that write
        import threading
        d_th = sub("in_progress")
        gw = guard.WriteGuard(d_th, ceiling=10 ** 6, reserve=0)
        entered, release, started = threading.Event(), threading.Event(), []
        calls[0] = 0

        def blocking_fsync(fh):
            calls[0] += 1
            if calls[0] == 2:
                entered.set()
                release.wait(timeout=60)
            real_fsync(fh)
        guard._fsync = blocking_fsync
        try:
            writer = threading.Thread(target=lambda: gw.write("slow.bin", b"s" * 1000))
            writer.start()
            tally.exact(9, entered.wait(timeout=60), True, "writer did not reach the data file")
            second = threading.Thread(target=lambda: started.append(guard.WriteGuard(d_th, ceiling=10 ** 6, reserve=0)))
            second.start()
            second.join(timeout=0.5)
            tally.exact(9, (second.is_alive(), started), (True, []), "a second guard started during a write")
            tally.exact(9, os.path.exists(os.path.join(d_th, "slow.bin" + guard.PART_SUFFIX)), True,
                        "partial file of the write in progress is missing")
            release.set()
            writer.join(timeout=60)
            second.join(timeout=60)
        finally:
            release.set()
            guard._fsync = real_fsync
        tally.exact(9, (len(started), started[0].removed_partial if started else None), (1, []),
                    "second guard removed the partial file of a write in progress")
        tally.exact(9, gw.read("slow.bin"), b"s" * 1000, "write in progress was damaged")
        # ---- recovery: only the guard's own partial files are removed; unrelated '.part' files are kept
        with open(os.path.join(d_th, "crash.bin" + guard.PART_SUFFIX), "wb") as fh:
            fh.write(b"partial")
        with open(os.path.join(d_th, "someone_elses.part"), "wb") as fh:
            fh.write(b"not ours")
        g4 = guard.WriteGuard(d_th, ceiling=10 ** 6, reserve=0)
        tally.exact(9, g4.removed_partial, ["crash.bin" + guard.PART_SUFFIX], "stale partial file not reported")
        tally.exact(9, os.path.exists(os.path.join(d_th, "crash.bin" + guard.PART_SUFFIX)), False, "stale partial file kept")
        tally.exact(9, os.path.exists(os.path.join(d_th, "someone_elses.part")), True, "unrelated .part file removed")
        tally.exact(9, g4.written, led + per(1000), "recovery changed the count")
        # ---- a damaged ledger is refused at start and at the next write, never reset
        g5 = guard.WriteGuard(d_th, ceiling=10 ** 6, reserve=0)
        for content in ("{broken", '{"written": -5}', '{"schema": "seqref-dunc-write-ledger/1", "written": 3}',
                        '{"written": 999999}', "[]"):
            with open(os.path.join(d_th, guard.LEDGER_NAME), "w") as fh:
                fh.write(content)
            refused(lambda: guard.WriteGuard(d_th), "DIAG_UNC_INPUT_INVALID", "damaged ledger at start: " + content)
            refused(lambda: g5.write("late.bin", b"y"), "DIAG_UNC_INPUT_INVALID", "damaged ledger at write: " + content)
            tally.exact(9, os.path.exists(os.path.join(d_th, "late.bin")), False, "write went ahead with a damaged ledger")
        os.remove(os.path.join(d_th, guard.LEDGER_NAME))
        refused(lambda: g5.write("late.bin", b"y"), "DIAG_UNC_INPUT_INVALID", "ledger removed while a guard is active")
        # ---- ledger creation: only in a genuinely new directory; a missing ledger never restarts the count
        d_new = sub("new_dir")
        g_new = guard.WriteGuard(d_new, ceiling=10 ** 6, reserve=0)
        tally.exact(9, (g_new.written, sorted(os.listdir(d_new))), (led, [guard.LEDGER_NAME, guard.LOCK_NAME]),
                    "first-time creation in an empty directory")
        with open(os.path.join(d_new, guard.LOCK_NAME)) as fh:
            tally.exact(9, fh.read(), guard.LOCK_MARK, "lock file not marked at creation")
        g_new.write("out.bin", b"o" * 500)                              # reviewer's sequence
        tally.exact(9, g_new.written, led + per(500), "count before the ledger is removed")
        os.remove(os.path.join(d_new, guard.LEDGER_NAME))
        refused_unchanged(d_new, "ledger missing from a directory with outputs")
        tally.exact(9, os.path.exists(os.path.join(d_new, guard.LEDGER_NAME)), False, "refusal recreated the ledger")
        snapshot = tree(d_new)
        refused(lambda: g_new.write("more.bin", b"m"), "DIAG_UNC_INPUT_INVALID", "write after the ledger was removed")
        tally.exact(9, tree(d_new) == snapshot, True, "refused write changed the directory")
        tally.exact(9, g_new.read("out.bin"), b"o" * 500, "existing output damaged by the refusal")
        d_del = sub("deleted_outputs")                                  # outputs deleted, then the ledger removed
        g_del = guard.WriteGuard(d_del, ceiling=10 ** 6, reserve=0)
        g_del.write("tmp.bin", b"t" * 700)
        g_del.delete("tmp.bin")
        os.remove(os.path.join(d_del, guard.LEDGER_NAME))
        tally.exact(9, os.listdir(d_del), [guard.LOCK_NAME], "directory should hold only the lock file")
        refused_unchanged(d_del, "ledger missing, only a marked lock file left")
        d_used = sub("unrelated_file")                                  # never guarded, but not empty: no lock file
        with open(os.path.join(d_used, "notes.txt"), "w") as fh:
            fh.write("existing evidence")
        refused_unchanged(d_used, "no ledger in a directory that already holds a file")
        tally.exact(9, os.listdir(d_used), ["notes.txt"], "a lock file was left in an unguarded directory")
        d_part = sub("leftover_part")                                   # leftover partial file, no ledger, no lock
        with open(os.path.join(d_part, "x.bin" + guard.PART_SUFFIX), "wb") as fh:
            fh.write(b"partial")
        refused_unchanged(d_part, "no ledger in a directory with a leftover partial file")
        d_sub = sub("only_subdir")
        os.mkdir(os.path.join(d_sub, "stage1"))
        refused_unchanged(d_sub, "no ledger in a directory that holds a subdirectory")
        d_lock = sub("unmarked_lock_and_file")                          # unmarked lock file plus another file
        open(os.path.join(d_lock, guard.LOCK_NAME), "w").close()
        with open(os.path.join(d_lock, "data.bin"), "wb") as fh:
            fh.write(b"d")
        refused_unchanged(d_lock, "no ledger, unmarked lock file and another file")
        d_bad = sub("damaged_ledger_start")                             # damaged ledger: refused, nothing changed
        guard.WriteGuard(d_bad, ceiling=10 ** 6, reserve=0).write("keep.bin", b"k" * 50)
        with open(os.path.join(d_bad, guard.LEDGER_NAME), "w") as fh:
            fh.write("{broken")
        with open(os.path.join(d_bad, "left.bin" + guard.PART_SUFFIX), "wb") as fh:
            fh.write(b"partial")
        refused_unchanged(d_bad, "damaged ledger with a leftover partial file")
        d_ok = sub("unmarked_lock_only")                                # an empty, unmarked lock file alone is a new directory
        open(os.path.join(d_ok, guard.LOCK_NAME), "w").close()
        tally.exact(9, guard.WriteGuard(d_ok, ceiling=10 ** 6, reserve=0).written, led, "new directory with an empty lock file")
        # ---- a tampered output is detected on read
        with open(os.path.join(d1, "c.bin"), "wb") as fh:
            fh.write(b"tampered")
        refused(lambda: g.read("c.bin"), "DIAG_UNC_REPRODUCTION_MISMATCH", "sidecar mismatch on read")
        tally.exact(9, (guard.CEILING_BYTES, guard.RESERVE_BYTES), (21474836480, 68719476736), "registered limits")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    failed = {r: v["fail"][:5] for r, v in tally.rows.items() if v["fail"]}
    table = {r: {"checks": v["n"], "max_dev": v["max_dev"], "pass": not v["fail"]} for r, v in tally.rows.items()}
    if failed:
        raise AssertionError("fixture 12 rows failed: %r" % failed)
    return {"acceptance_table": table, "tied_coordinates": n_tied, "disk": shared["disk_info"], "contrasts": summary}


def _guard_worker(root, ceiling, tag, n_writes, size, queue):
    """Child process: try n_writes writes; report (accepted, refused by the limit, other errors)."""
    logging.getLogger("diag_unc").handlers.clear()
    logging.getLogger("diag_unc").addHandler(logging.NullHandler())   # the parent counts the outcomes
    accepted = limited = other = 0
    try:
        g = guard.WriteGuard(root, ceiling=ceiling, reserve=0)
        for i in range(n_writes):
            try:
                g.write("%s_%03d.bin" % (tag, i), b"w" * size)
                accepted += 1
            except DiagUncError as exc:
                if exc.code == "DIAG_UNC_RESOURCE_LIMIT":
                    limited += 1
                else:
                    other += 1
    except Exception:
        other += 1
    queue.put((accepted, limited, other))


# ================================================================ runner
def main():
    handler = ListHandler()
    logging.getLogger("diag_unc").addHandler(handler)
    logging.getLogger("diag_unc").setLevel(logging.WARNING)
    logging.getLogger("diag_unc").propagate = False
    try:
        import scipy
        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = "not installed (chi-square critical value used without the cross-check)"
    print("SEQREF-DUNC-FIX v0.5 | core %s stats %s guard %s | fixture seed %d"
          % (core.__version__, stats.__version__, guard.__version__, FIXTURE_SEED))
    print("python %s | numpy %s | scipy %s | byte order %s" % (sys.version.split()[0], np.__version__, scipy_version,
                                                               sys.byteorder))
    shared = {}
    t0 = time.time()
    raw = {}
    disk_root = tempfile.mkdtemp(prefix="dunc_stage1_")
    try:
        disk = guard.WriteGuard(disk_root, ceiling=2 * 10 ** 9, reserve=0)
        shared["pop"] = _population(rng_for(12), ROLES, keep=raw, ties=True, disk=disk)
        files = sorted(n for n in os.listdir(os.path.join(disk_root, "stage1")) if n.endswith(".json"))
        shared["disk_info"] = {"files": len(files), "bytes_counted": disk.written}
    finally:
        shutil.rmtree(disk_root, ignore_errors=True)
    shared["raw"] = raw
    shared["idx"] = stats.bootstrap_matrix(seed=FIXTURE_SEED)
    print("fixture-seed bootstrap matrix sha256 %s" % stats.matrix_identity(shared["idx"]))
    print("shared synthetic population built in %.1f s (6 roles x 3 seeds x 97 volumes, K = 128, with ties; stage-1 "
          "records written through the guard and reloaded: %d files, %d bytes counted)"
          % (time.time() - t0, shared["disk_info"]["files"], shared["disk_info"]["bytes_counted"]))
    fixtures = [(1, fixture_01, ()), (2, fixture_02, ()), (3, fixture_03, ()), (4, fixture_04, ()),
                (5, fixture_05, ()), (6, fixture_06, ()), (7, fixture_07, ()), (8, fixture_08, ()),
                (9, fixture_09, (shared,)), (10, fixture_10, (shared,)), (12, fixture_12, (shared,))]
    failures = 0
    for number, fn, args in fixtures:
        t0 = time.time()
        try:
            info = fn(*args)
            print("fixture %2d PASS (%.1f s) %s" % (number, time.time() - t0, info))
        except Exception as exc:  # a failing fixture is reported, never skipped
            failures += 1
            logging.getLogger("diag_unc").error("fixture %d failed: %s", number, exc)
            print("fixture %2d FAIL (%.1f s) %s: %s" % (number, time.time() - t0, type(exc).__name__, exc))
    errors = sum(1 for r in handler.records if r.levelno >= logging.ERROR)
    warnings = sum(1 for r in handler.records if r.levelno == logging.WARNING)
    print("logged: %d error records (expected refusals and any failure), %d warning records (non-fatal events)"
          % (errors, warnings))
    print("fixture 11 (adapter) and X2 (bank regeneration) are not part of step B")
    for item in OBSERVATIONS:
        print("OBSERVATION: " + item)
    for item in OPEN_ITEMS:
        print("OPEN ITEM: " + item)
    status = "FAIL" if failures else ("PASS WITH OPEN ITEMS" if OPEN_ITEMS else "PASS")
    print("RESULT: %s (%d of %d fixtures failed; %d open items)" % (status, failures, len(fixtures), len(OPEN_ITEMS)))
    return 1 if failures else (3 if OPEN_ITEMS else 0)


if __name__ == "__main__":
    sys.exit(main())
