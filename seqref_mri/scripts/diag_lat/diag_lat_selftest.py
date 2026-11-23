# SEQREF-DLAT-TEST v0.5 -- diag_lat_selftest.py: synthetic fixtures for the D-LAT modules
# LIFETIME: KEEP
"""
STATUS: PRE-FREEZE ENGINEERING; NOT ACTIVATED (236 LE19). Plan SEQREF-DLAT-IMPL v0.2.

SELF-TEST EXCEPTION (stated here and nowhere else): this file uses deterministic synthetic
data and a NumPy stand-in for the masked Fourier operator. It is a test hook; production
code never falls into it. Every output of this file is NOT D-LAT EVIDENCE.
Self-test seed PCG64(987654321) is declared here; it is not a registered scientific seed.

P1: T1-T8, T12a, T13, T16 (core). P2: T14 (bootstrap), T15 (class rules, tags, outcome,
not-computed propagation, independent second implementation), T17 (L6), T18 (resource and
memory guards), T19 (ports against the archived D-UNC originals).
T19 needs the archived D-UNC files (diag_unc_core.py, diag_unc_stats.py, diag_unc_guard.py):
pass --dunc-archive DIR. Their sha256 is verified before import. Without the archive T19 is
reported NOT RUN and the run FAILS, unless --without-t19 is given explicitly.
NOT implemented: T9-T11, T12b, R1 (P3, on hold).

Changelog
- v0.5 (2026-11-23): for diag_lat_stats.py v0.4 (248 LE76): the end-to-end check expects
  RULES_VERSION to name the frozen specification v1.0 and its sha256, instead of "v0.6.2".
  No other fixture, tolerance or check changed.
- v0.4 (2026-11-22): revision SEQREF-DLAT-P23REV v0.1 (242 LE51) for diag_lat_stats.py v0.3
  and candidate v0.6.2. T15 only: outcome cases as a table (A14) plus an independent second
  implementation by enumeration of every completion of the missing labels; seed-tag cases
  (A16: four values, boundaries, cross-contrast combinations); fixed-epoch tag in its four
  situations and against an independent rule on all label combinations (A19); the string
  "not applicable" for L1 and L5 (A17); rules_version and open_contrasts. The case tables
  are module-level so that the negative control can run them on the v0.2 functions.
  No other fixture, tolerance or check changed.
- v0.3 (2026-11-21): T17 regression checks for the L6 cross-seed defect (a state not computed
  in seed 2026 must not suppress L6 for seeds 2027 and 2028 on the SAME contrast); the earlier
  'other seed unaffected' check used a contrast that did not contain the missing state and is
  replaced. T14 checks that the pooled rule still propagates across seeds.
- v0.2 (2026-11-21): P2 fixtures T14, T15, T17, T18, T19 added; command-line options
  --dunc-archive, --without-t19, --out; P1 fixtures unchanged.
- v0.1 (2026-11-21): P1 fixtures T1-T8, T12a, T13, T16; independent second implementation
  of TC_G (R3, core part); runtime benchmark; JSON and text evidence report.
"""
import hashlib
import json
import logging
import math
import itertools
import os
import statistics
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))    # works under python3 -I
import diag_lat_core as C
import diag_lat_guard as GD
import diag_lat_stats as ST

__version__ = "0.5"
SELFTEST_SEED = 987654321
BANNER = "NOT D-LAT EVIDENCE (synthetic self-test)"
# T6/T7b values observed at step B, per NumPy version; compared to 1e-12 on that version.
RECORDED = {"2.5.3": {"T6_mean_K64": -0.008072882337133483, "T6_mean_K128": -0.0033667118735011482,
                      "T7b_pr": 125.8501580963618}}

RESULTS = []


class Fx:
    def __init__(self, fid, title):
        self.fid, self.title, self.checks = fid, title, []

    def close(self, name, observed, expected, tol):
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):      # e.g. a "not computed" status
            self.true(name, False, f"observed {observed!r} is not a number")
            return
        res = abs(observed - expected)
        self.checks.append({"check": name, "observed": observed, "expected": expected,
                            "residual": res, "tolerance": tol, "pass": bool(res <= tol)})

    def true(self, name, cond, note=""):
        self.checks.append({"check": name, "observed": bool(cond), "expected": True,
                            "residual": None, "tolerance": "exact", "pass": bool(cond), "note": note})

    def raises(self, name, code, fn):
        try:
            fn()
        except C.DiagLatError as e:
            self.true(name, e.code == code, f"raised {e.code}")
        else:
            self.true(name, False, "no error raised")

    def info(self, name, value):
        self.checks.append({"check": name, "observed": value, "expected": None, "residual": None,
                            "tolerance": "recorded", "pass": True})

    def done(self):
        RESULTS.append({"id": self.fid, "title": self.title, "checks": self.checks,
                        "pass": all(c["pass"] for c in self.checks)})


def inv_cdf(p):
    nd = statistics.NormalDist()
    return np.array([nd.inv_cdf(float(v)) for v in p])


def digamma(x):
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1.0 / 12 - f * (1.0 / 120 - f / 252))


def whitened(k, d):
    """Deterministic K x d matrix with zero column means and exact sample covariance I."""
    kk = np.arange(1, k + 1, dtype=np.float64)[:, None]
    jj = np.arange(d, dtype=np.float64)[None, :]
    g = np.sin(0.37 * kk * (jj + 1.0)) + 0.11 * np.cos(1.3 * kk + jj)
    gc = g - g.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(gc, compute_uv=False)
    if not (k - 1 >= d and sv[-1] > 1e-6 * sv[0]):
        raise RuntimeError("fixture construction error: whitening matrix is rank deficient")
    w, v = np.linalg.eigh(gc.T @ gc / (k - 1))
    return gc @ (v @ np.diag(w ** -0.5) @ v.T)


def tc_g_independent(u, j):
    """Second implementation of TC_G (R3): explicit loops, no shared code with the core."""
    k, d = len(u), len(u[0])
    acc = 0.0
    for col in range(d):
        m = sum(u[r][col] for r in range(k)) / k
        var = sum((u[r][col] - m) ** 2 for r in range(k)) / (k - 1)
        acc += 0.5 * math.log(var)
    return acc / d - sum(j) / k


# ---- fixtures ----------------------------------------------------------------------
def t1():
    fx = Fx("T1", "KS, skewness, excess kurtosis")
    d = C.D_FREE
    grid = inv_cdf((np.arange(1, d + 1) - 0.5) / d)
    fx.close("KS of the N(0,1) quantile grid = 1/(2d)", C.ks_normal(grid), 1.0 / (2 * d), 1e-9)
    mu = 0.3
    fx.close("KS, grid shifted by 0.3, vs population 2*Phi(mu/2)-1 (bound 1/(2d))",
             C.ks_normal(grid + mu), 2 * float(C.norm_cdf(mu / 2)) - 1, 1.0 / (2 * d) + 1e-12)
    s = 1.5
    xs = math.sqrt(2 * math.log(s) * s * s / (s * s - 1))
    pop = float(C.norm_cdf(xs) - C.norm_cdf(xs / s))
    fx.close("KS, grid scaled by 1.5, vs population closed form (bound 1/(2d))",
             C.ks_normal(grid * s), pop, 1.0 / (2 * d) + 1e-12)
    # independent brute-force KS on a small irregular sample
    x = np.array([-2.2, -0.4, -0.4, 0.1, 0.9, 1.3, 3.0])
    xs_ = sorted(x.tolist()); n = len(xs_); ref = 0.0
    for i, v in enumerate(xs_):
        f = 0.5 * (1 + math.erf(v / math.sqrt(2)))
        ref = max(ref, (i + 1) / n - f, f - i / n)
    fx.close("KS equals an independent loop implementation (with ties)", C.ks_normal(x), ref, 1e-15)
    # two-component point-mass mixture: 0.3 at -1, 0.7 at 2 (closed-form everything)
    mix = np.array([-1.0] * 300 + [2.0] * 700)
    fx.close("KS of the two-point mixture = Phi(2) - 0.3", C.ks_normal(mix),
             float(C.norm_cdf(2.0)) - 0.3, 1e-12)
    p, a, b = 0.3, -1.0, 2.0
    m = p * a + (1 - p) * b
    m2 = p * (a - m) ** 2 + (1 - p) * (b - m) ** 2
    m3 = p * (a - m) ** 3 + (1 - p) * (b - m) ** 3
    m4 = p * (a - m) ** 4 + (1 - p) * (b - m) ** 4
    sk, ek = C.pooled_moments(mix)
    fx.close("skewness of the mixture (analytic)", sk, m3 / m2 ** 1.5, 1e-9)
    fx.close("excess kurtosis of the mixture (analytic)", ek, m4 / m2 ** 2 - 3, 1e-9)
    sk, ek = C.pooled_moments(grid)
    fx.close("skewness of the symmetric grid = 0", sk, 0.0, 1e-9)
    fx.info("excess kurtosis of the finite grid (recorded; < 0 by tail truncation)", ek)
    fx.done()


def t2():
    fx = Fx("T2", "q_z, tau, q_tail, B identity")
    d = 1000
    u = (np.sin(np.arange(d) * 0.7) * 2.0).astype(np.float32)
    tail_idx = [3, 500, 999]
    u[tail_idx] = np.array([6.0, -7.5, 5.2], dtype=np.float32)
    u[10] = np.float32(C.SPLINE_B)          # exactly on the bound: not in the tail set
    mask, tau, q_tail = C.tail_set(u)
    fx.true("tail count exact (3; entry equal to B_s excluded)", int(mask.sum()) == 3 and not mask[10])
    fx.close("tau = 3/d", tau, 3 / d, 0.0)
    fx.close("q_tail = sum of squares on the tail / d", q_tail,
             (36.0 + 56.25 + float(np.float32(5.2)) ** 2) / d, 1e-12)
    z = (0.5 * u).astype(np.float32); z[mask] = u[mask]
    q_z = C.q_of(z)
    fx.true("q_tail <= q_z", q_tail <= q_z)
    fx.true("tail invariance check passes on equal entries", C.check_tail_invariance(z, u, mask) == 3)
    z64 = z.astype(np.float64)
    direct = -float(np.mean(-0.5 * z64 ** 2 - 0.5 * math.log(2 * math.pi)))
    fx.close("B = 0.5 q_z + 0.5 ln(2 pi) equals the direct Gaussian value", C.nll_prior_per_dim(q_z), direct, 1e-12)
    fx.done()


def t3():
    fx = Fx("T3", "gamma_truth, gamma_bank, L2 identity (linear decoder)")
    d, k = 6, 16
    q, _ = np.linalg.qr(np.cos(np.outer(np.arange(1, d + 1), np.arange(1, d + 1)) * 0.9))
    c = 2.5
    zb = whitened(k, d) + 0.3
    zs = np.linspace(-1.0, 1.5, d)
    r0 = np.zeros(d)
    out = C.l2_measures(c * zs @ q.T, r0, zs, c * zb @ q.T, zb)
    fx.close("isotropic decoder u = cQz: log gamma_truth = log c", out["log_gamma_truth"], math.log(c), 1e-12)
    fx.close("isotropic decoder: log gamma_bank = log c", out["log_gamma_bank"], math.log(c), 1e-12)
    fx.close("isotropic decoder: log ratio = 0", out["log_gamma_ratio"], 0.0, 1e-12)
    a = np.array([3.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    zs2 = np.zeros(d); zs2[0] = 1.7                       # truth on the a = 3 axis
    zb2 = np.zeros((k, d)); zb2[:, 1] = np.linspace(0.5, 2.0, k)   # bank on an a = 0.5 axis
    out = C.l2_measures(a * zs2, r0, zs2, zb2 * a, zb2)
    fx.close("diagonal decoder: log ratio = log(3 / 0.5)", out["log_gamma_ratio"], math.log(6.0), 1e-12)
    out = C.l2_measures(c * zs @ q.T, r0, zs, c * zb @ q.T, zb)
    fx.close("identity log RMSE_R0 = log gamma_truth + 0.5 log q_z",
             out["log_rmse_r0"], out["log_gamma_truth"] + 0.5 * math.log(C.q_of(zs)), 1e-12)
    fx.done()


def t4():
    fx = Fx("T4", "d01, a01, error decomposition")
    d, k = 6, 128
    z = whitened(k, d)
    l = np.linalg.cholesky(0.5 * np.eye(d) + 0.5 * np.ones((d, d)) + np.diag(np.arange(d) * 0.3))
    b = np.array([0.4, -0.2, 0.1, 0.0, 0.3, -0.5])
    ub = z @ l.T + b
    ut = np.array([1.0, 0.2, -0.3, 0.5, 0.1, -0.9]); r0 = np.zeros(d)
    ev = []
    out = C.l3_measures(ut, r0, ub, ev)
    tr = float(np.trace(l @ l.T))
    fx.close("d01 = ||b|| / sqrt(tr L L^T)", out["d01"], float(np.linalg.norm(b)) / math.sqrt(tr), 1e-10)
    fx.close("a01 = cos(b, u_true)", out["a01"],
             float(b @ ut) / float(np.linalg.norm(b) * np.linalg.norm(ut)), 1e-10)
    fx.close("tr Sigma / d", out["tr_sigma_per_dim"], tr / d, 1e-10)
    lhs = float(np.mean(np.sum((ub - ut) ** 2, axis=1)))
    fx.close("mean_k ||u_k - u||^2 = d e1^2 + ((K-1)/K) tr Sigma", lhs,
             d * out["e1_sq"] + (k - 1) / k * d * out["tr_sigma_per_dim"], 1e-10)
    fx.close("e0^2 - e1^2", out["e0_sq_minus_e1_sq"],
             (float(ut @ ut) - float((ut - b) @ (ut - b))) / d, 1e-10)
    fx.true("no event recorded", ev == [])
    fx.done()


def _t5_case(rho, d=6, k=128):
    z = whitened(k, d)
    r = (1 - rho) * np.eye(d) + rho * np.ones((d, d))
    l = np.linalg.cholesky(r)
    u = z @ l.T
    logdet_r = (d - 1) * math.log(1 - rho) + math.log(1 + (d - 1) * rho)
    j = np.full(k, logdet_r / (2 * d))        # J = -(1/d) log|det d f/du|, f = L^-1
    return u, j, -logdet_r / (2 * d)


def t5():
    fx = Fx("T5", "TC_G value and sign (deterministic whitening / colouring)")
    u, j, expected = _t5_case(0.5)
    ev = []
    out = C.l4_measures(u, j, ev)
    fx.close("sample SDs equal 1 (mean log sigma = 0)", out["mean_log_sigma"], 0.0, 1e-9)
    fx.close("TC_G = -(1/2d) log det R, rho = 0.5", out["TC_G"], expected, 1e-9)
    fx.true("TC_G > 0 for rho = 0.5", out["TC_G"] > 0)
    fx.close("independent second implementation (R3) agrees", tc_g_independent(u.tolist(), j.tolist()),
             out["TC_G"], 1e-12)
    u0, j0, e0 = _t5_case(0.0)
    fx.close("R = I gives TC_G = 0", C.l4_measures(u0, j0, [])["TC_G"], e0, 1e-9)
    wrong = C.l4_measures(u, -j, [])["TC_G"]
    fx.true("NEGATIVE CONTROL: flipped sign of J is detected (differs from the expected value)",
            abs(wrong - expected) > 1e-3, f"|wrong - expected| = {abs(wrong - expected):.6f}")
    fx.done()


def t6(rng_reps=2000):
    fx = Fx("T6", "TC_G finite-K behaviour (self-test generator)")
    d = 16
    rec = RECORDED.get(np.__version__, {})
    for k in (64, 128):
        gen = np.random.Generator(np.random.PCG64(SELFTEST_SEED))
        vals = np.empty(rng_reps); zero_j = np.zeros(k)
        for r in range(rng_reps):
            vals[r] = C.l4_measures(gen.standard_normal((k, d)), zero_j, [])["TC_G"]
        n = k - 1
        analytic = 0.5 * (digamma(n / 2) - math.log(n / 2))
        mean = float(vals.mean())
        fx.close(f"K = {k}: mean TC_G over {rng_reps} repetitions vs analytic bias {analytic:.6f}",
                 mean, analytic, 1.5e-3)
        fx.info(f"K = {k}: standard error of that mean", float(vals.std(ddof=1) / math.sqrt(rng_reps)))
        fx.info(f"K = {k}: share of negative TC_G values (reported, never truncated)", float((vals < 0).mean()))
        key = f"T6_mean_K{k}"
        if key in rec:
            fx.close(f"K = {k}: equals the value recorded for NumPy {np.__version__}", mean, rec[key], 1e-12)
        else:
            fx.info(f"K = {k}: no recorded value for NumPy {np.__version__} (observed value recorded)", mean)
    fx.done()


def t7():
    fx = Fx("T7", "participation ratio")
    d, k = 6, 128
    z = whitened(k, d)
    fx.close("(a) equal eigenvalues: PR = d", C.participation_ratio(z * math.sqrt(2.0)), float(d), 1e-10)
    lam = np.zeros(d); lam[0] = 1.0
    u = z * np.sqrt(lam)
    fx.close("(a) lambda = (1,0,...,0): PR = 1", C.participation_ratio(u), 1.0, 1e-10)
    ev = []
    out = C.l4_measures(u, np.zeros(k), ev)
    fx.true("(a) sigma_j = 0: TC_G not computed, PR still computed, one event with the count",
            out["TC_G"] is C.NOT_COMPUTED and abs(out["PR"] - 1.0) < 1e-10 and len(ev) == 1
            and "5 of 6" in ev[0]["detail"], ev[0]["detail"] if ev else "")
    kk, dd = C.K_BANK, C.D_FREE
    pr1 = C.participation_ratio(np.random.Generator(np.random.PCG64(SELFTEST_SEED)).standard_normal((kk, dd)))
    pr2 = C.participation_ratio(np.random.Generator(np.random.PCG64(SELFTEST_SEED)).standard_normal((kk, dd)))
    fx.close("(b) isotropic K = 128, d = 13,824: reproducible on repetition", pr1, pr2, 1e-12)
    plan_ref = kk * dd / (kk + dd)
    fx.close("(b) within 5 % of K d / (K + d) (plan plausibility bound)", pr1, plan_ref, 0.05 * plan_ref)
    fx.info("(b) observed PR", pr1)
    fx.info("(b) core analytic reference n d / (n + d + 1), n = K - 1", C.pr_isotropic_reference(kk, dd))
    key = "T7b_pr"
    rec = RECORDED.get(np.__version__, {})
    if key in rec:
        fx.close(f"(b) equals the value recorded for NumPy {np.__version__}", pr1, rec[key], 1e-12)
    fx.done()


def t8():
    fx = Fx("T8", "physical complements; TC_G invariance")
    d, k = 6, 128
    u, j, _ = _t5_case(0.5)
    u = u + np.linspace(-0.2, 0.3, d)
    s = np.array([0.5, 2.0, 1.5, 0.8, 3.0, 1.1]); m = np.linspace(1.0, -1.0, d)
    up = C.to_physical(u, s, m)
    tc_std = C.l4_measures(u, j, [])["TC_G"]
    tc_phys = C.l4_measures(up, j + float(np.mean(np.log(s))), [])["TC_G"]
    fx.close("TC_G identical in both spaces (J shifts by mean log s)", tc_phys, tc_std, 1e-10)
    ut = np.array([1.0, 0.2, -0.3, 0.5, 0.1, -0.9]); r0 = np.linspace(0.1, -0.1, d)
    c = 2.0
    cs, cm = np.full(d, c), np.zeros(d)
    a = C.l3_measures(ut, r0, u, []); b = C.l3_measures(C.to_physical(ut, cs, cm), C.to_physical(r0, cs, cm),
                                                      C.to_physical(u, cs, cm), [])
    fx.close("uniform scale c: d01 unchanged", b["d01"], a["d01"], 1e-10)
    fx.close("uniform scale c: a01 unchanged", b["a01"], a["a01"], 1e-10)
    fx.close("uniform scale c: e0^2 scales by c^2", b["e0_sq"], c * c * a["e0_sq"], 1e-10)
    utp, r0p = C.to_physical(ut, s, m), C.to_physical(r0, s, m)
    ph = C.l3_measures(utp, r0p, up, [])
    mu = up.mean(axis=0); tr = float(np.sum(np.var(up, axis=0, ddof=1)))
    fx.close("general scale: physical d01 equals the direct formula", ph["d01"],
             float(np.linalg.norm(mu - r0p)) / math.sqrt(tr), 1e-10)
    fx.close("general scale: physical e0^2 = mean of s^2 (u - R0)^2", ph["e0_sq"],
             float(np.mean((s * (ut - r0)) ** 2)), 1e-10)
    zs = np.linspace(-1, 1.5, d); zb = whitened(k, d) + 0.3
    l2s = C.l2_measures(ut, r0, zs, u, zb)
    l2p = C.l2_measures(C.to_physical(ut, cs, cm), C.to_physical(r0, cs, cm), zs, C.to_physical(u, cs, cm), zb)
    fx.close("uniform scale c: log gamma_truth shifts by log c", l2p["log_gamma_truth"],
             l2s["log_gamma_truth"] + math.log(c), 1e-10)
    fx.close("uniform scale c: log ratio unchanged", l2p["log_gamma_ratio"], l2s["log_gamma_ratio"], 1e-10)
    fx.done()


def t12a():
    fx = Fx("T12a", "perturbation scaling and pathway (float64, analytic; NumPy stand-in operator)")
    n, eps, amax = 96, C.EPSILON, 3.7
    gen = np.random.Generator(np.random.PCG64(SELFTEST_SEED))
    x = gen.standard_normal((n, n)) + 1j * gen.standard_normal((n, n))
    cols = np.r_[np.arange(44, 52), np.arange(0, 32, 2)]            # 24 acquired columns
    mask = np.zeros(n, bool); mask[cols] = True
    A = lambda img: np.fft.fft2(img, norm="ortho") * mask[None, :]
    AH = lambda ksp: np.fft.ifft2(ksp * mask[None, :], norm="ortho")
    y = A(x)
    y_acq = y[:, mask].reshape(-1)
    fx.true("acquired cells = 2,304", y_acq.size == C.N_ACQUIRED)
    raw = gen.standard_normal((C.N_ACQUIRED, 2))
    nz = C.scale_noise(raw, y_acq, eps)
    rms = lambda c: math.sqrt(float(np.mean(np.abs(c) ** 2)))
    fx.close("realised RMS / (eps RMS(y)) = 1", rms(nz) / (eps * rms(y_acq)), 1.0, 1e-12)
    cond = AH(y) / amax
    yp = y.copy(); yp[:, mask] += nz.reshape(n, -1)
    condp = AH(yp) / amax
    y0 = y.copy(); y0[:, mask] += np.zeros_like(nz).reshape(n, -1)
    fx.true("zero noise gives the conditioning input exactly", np.array_equal(AH(y0) / amax, cond))
    lhs = math.sqrt(float(np.sum(np.abs((condp - cond) * amax) ** 2)))
    rhs = math.sqrt(float(np.sum(np.abs(nz) ** 2)))
    fx.close("single division: ||(cond' - cond) amax|| / ||n|| = 1 (Parseval)", lhs / rhs, 1.0, 1e-12)
    cond_dd = AH(yp) / amax / amax
    bad = math.sqrt(float(np.sum(np.abs((cond_dd - cond) * amax) ** 2))) / rhs
    fx.true("NEGATIVE CONTROL: a doubled division fails the same check", abs(bad - 1.0) > 1e-3,
            f"ratio = {bad:.6f}")
    fx.done()


def t13():
    fx = Fx("T13", "L1 projection (QR, sign rule, identity)")
    d, m = 200, C.M_PROJ
    raw = np.sin(np.outer(np.arange(1, d + 1), np.arange(1, m + 1)) * 0.31) + \
        0.2 * np.cos(np.arange(d)[:, None] * 1.7 + np.arange(m)[None, :])
    p = C.projection_matrix(raw)
    fx.close("orthonormal columns: max |P^T P - I|", float(np.max(np.abs(p.T @ p - np.eye(m)))), 0.0, 1e-12)
    r = p.T @ raw
    fx.true("diagonal of R positive", bool(np.all(np.diag(r) > 0)))
    fx.close("R upper triangular: max |strict lower part|", float(np.max(np.abs(np.tril(r, -1)))), 0.0, 1e-12)
    fx.true("same bytes on repetition (identity exact)",
            C.array_identity(p)["sha256"] == C.array_identity(C.projection_matrix(raw.copy()))["sha256"])
    p2 = C.projection_matrix(raw * np.array([1, -1, 1, -1, 1, 1, -1, 1.0]))
    fx.close("column sign of the draw flips the column of P (sign rule deterministic)",
             float(np.max(np.abs(p2 - p * np.array([1, -1, 1, -1, 1, 1, -1, 1.0])))), 0.0, 1e-12)
    z = whitened(97, m) @ p.T                          # codes whose projection has covariance I
    fx.close("E_cov = 0 for codes with exact projected covariance I", C.projected_cov_mismatch(z, p), 0.0, 1e-10)
    fx.close("E_cov = 1 for projected covariance 2 I", C.projected_cov_mismatch(z * math.sqrt(2.0), p), 1.0, 1e-10)
    fx.close("analytic reference sqrt((m+1)/(N-1))", C.e_cov_reference(97, m), math.sqrt(9 / 96), 1e-15)
    fx.done()


def t16():
    fx = Fx("T16", "edge cases of section 3.1 (core rows)")
    d, k = 6, 8
    z = whitened(k, d) + 0.2; ub = z * 1.3 + 0.1
    ut = np.linspace(0.5, 1.5, d); r0 = np.zeros(d); zs = np.linspace(-1, 1, d)
    nanv = zs.copy(); nanv[2] = np.nan
    fx.raises("non-finite value -> NONFINITE", C.NONFINITE, lambda: C.l1_measures(nanv))
    infb = ub.copy(); infb[0, 0] = np.inf
    fx.raises("non-finite bank -> NONFINITE", C.NONFINITE, lambda: C.l3_measures(ut, r0, infb, []))
    fx.raises("L2: ||z*|| = 0 -> ZERO_DENOMINATOR", C.ZERO_DENOMINATOR,
              lambda: C.l2_measures(ut, r0, np.zeros(d), ub, z))
    zz = z.copy(); zz[3] = 0.0
    fx.raises("L2: ||z_k|| = 0 -> ZERO_DENOMINATOR", C.ZERO_DENOMINATOR,
              lambda: C.l2_measures(ut, r0, zs, ub, zz))
    fx.raises("L2: ||u_true - R0|| = 0 -> ZERO_DENOMINATOR", C.ZERO_DENOMINATOR,
              lambda: C.l2_measures(r0, r0, zs, ub, z))
    fx.raises("L3a: tr Sigma = 0 -> ZERO_DENOMINATOR", C.ZERO_DENOMINATOR,
              lambda: C.l3_measures(ut, r0, np.tile(ut, (k, 1)), []))
    ev = []
    wb = whitened(k, d) + 0.4
    out = C.l3_measures(ut, wb.mean(axis=0), wb, ev)        # R0 set equal to the bank mean exactly
    fx.true("L3b: ||mu - R0|| = 0 -> d01 = 0 reported, a01 NOT_COMPUTED, one non-fatal event",
            out["d01"] == 0.0 and out["a01"] is C.NOT_COMPUTED and len(ev) == 1
            and ev[0]["code"] == C.UNDEFINED_STATISTIC)
    ub0 = ub.copy(); ub0[:, [1, 4]] = 0.7
    ev = []
    out = C.l4_measures(ub0, np.zeros(k), ev)
    fx.true("L4: sigma_j = 0 -> TC_G NOT_COMPUTED, count recorded (2 of 6), PR computed",
            out["TC_G"] is C.NOT_COMPUTED and "2 of 6" in ev[0]["detail"] and math.isfinite(out["PR"]))
    ev = []
    bm = C.bank_measures(ut, r0, zs, ub0, z, np.zeros(k), ev,
                         subsets={"full": np.arange(k), "halfA": np.arange(k // 2)})
    fx.true("bank_measures: the event carries its subset label",
            bm["full"]["TC_G"] is C.NOT_COMPUTED and {e["subset"] for e in ev} == {"full", "halfA"})
    y = (np.arange(1, 9) + 1j * np.arange(8)).astype(np.complex128)
    raw = np.sin(np.arange(16.0)).reshape(8, 2) + 0.3
    bad = raw.copy(); bad[1, 0] = np.nan
    fx.raises("L5(b): non-finite draw -> NONFINITE", C.NONFINITE, lambda: C.scale_noise(bad, y))
    ybad = y.copy(); ybad[2] = np.inf
    fx.raises("L5(b): non-finite y -> NONFINITE", C.NONFINITE, lambda: C.scale_noise(raw, ybad))
    fx.raises("L5(b): zero noise RMS -> ZERO_DENOMINATOR", C.ZERO_DENOMINATOR,
              lambda: C.scale_noise(np.zeros((8, 2)), y))
    fx.raises("L5(b): RMS(y) = 0 -> ZERO_DENOMINATOR", C.ZERO_DENOMINATOR,
              lambda: C.scale_noise(raw, np.zeros(8, complex)))
    fx.raises("L5(a) S2: zero tolerance scale -> SWAP_FIXTURE", C.SWAP_FIXTURE,
              lambda: C.swap_tolerance_scale(np.zeros(8, complex)))
    fx.raises("L5(a) S4: identical swapped input -> SWAP_IDENTITY", C.SWAP_IDENTITY,
              lambda: C.check_swap_identity(raw, raw.copy()))
    u32 = np.array([6.0, 0.5, -7.0, 1.0], np.float32); mask, _, _ = C.tail_set(u32)
    z32 = u32.copy(); z32[2] = np.float32(-7.0000005)
    fx.raises("L1: changed tail entry -> TAIL_INVARIANCE", C.TAIL_INVARIANCE,
              lambda: C.check_tail_invariance(z32, u32, mask))
    fx.raises("tail set on a float64 vector -> INPUT", C.INPUT, lambda: C.tail_set(u32.astype(np.float64)))
    fx.true("STABILITY_SUBSETS: four labels, three distinct subsets (pre64 == halfA)",
            len(C.STABILITY_SUBSETS) == 4 and np.array_equal(C.STABILITY_SUBSETS["pre64"], C.STABILITY_SUBSETS["halfA"])
            and len({v.tobytes() for v in C.STABILITY_SUBSETS.values()}) == 3)
    fx.done()


# ---- P2 fixtures ---------------------------------------------------------------------
DUNC_SHA = {"diag_unc_core.py": "a0d3b4a4079ff290a93098eb7a1c53173df596080a6b2b51469dc0f757c18b06",
            "diag_unc_stats.py": "b8706513a7da0517f477a81a49b4b7d2055e1451b47bec7dcc2c9b63a5de8f55",
            "diag_unc_guard.py": "b3dc9f69226f97275712b1bef5cfe9e0fdf1716f944c9ee137ca7223fe2e5bc5"}
DUNC_MATRIX_SHA = "9517d035ee691d17b2ae4fbc31958222d85ad803431546f5a0455fe40cda6443"
ROLES = ["%s/%s" % (a, st) for a in ("lam0", "lam2") for st in ("e3", "e10", "e30", "sel", "fin")]
SEEDS = (2026, 2027, 2028)


def toy_idx(v, n_boot=400):
    return np.random.Generator(np.random.PCG64(SELFTEST_SEED)).integers(0, v, size=(n_boot, v), dtype=np.int64)


def toy_population(level, v=12, names=("KS",), noise=0.01, subset=None, missing=None):
    """level: {role: value}. Per-volume value = level + small deterministic wiggle (same across roles,
    so paired differences are exact constants plus a seed-specific deterministic term)."""
    pop = ST.Population(["v%03d" % i for i in range(v)])
    for name in names:
        for role in ROLES:
            for si, seed in enumerate(SEEDS):
                base = np.sin(np.arange(v) * (1.3 + si))            # cancels in every paired difference
                wig = noise * np.cos(np.arange(v) * 0.7 + ROLES.index(role) + si)
                vals = list(level[role] + base + wig)
                if missing and (role, seed, name) in missing:
                    vals[missing[(role, seed, name)]] = C.NOT_COMPUTED
                pop.set(role, seed, name, vals, subset)
    return pop.seal()


def t14():
    fx = Fx("T14", "bootstrap mechanics")
    idx = ST.bootstrap_matrix(seed=SELFTEST_SEED, n_volumes=12, n_boot=400)
    fx.true("index matrix hash reproducible",
            ST.matrix_identity(idx) == ST.matrix_identity(ST.bootstrap_matrix(seed=SELFTEST_SEED, n_volumes=12, n_boot=400)))
    fx.true("verify_matrix accepts the recorded hash", ST.verify_matrix(idx, ST.matrix_identity(idx), 12, 400) == ST.matrix_identity(idx))
    fx.raises("verify_matrix: wrong hash -> BOOTSTRAP_IDENTITY", ST.BOOTSTRAP_IDENTITY, lambda: ST.verify_matrix(idx, "0" * 64, 12, 400))
    fx.raises("matrix_identity: int32 matrix -> BOOTSTRAP_IDENTITY", ST.BOOTSTRAP_IDENTITY,
              lambda: ST.matrix_identity(idx.astype(np.int32)))
    calls = []

    class Gen:
        def integers(self, *a, **k):
            calls.append((a, k)); return np.zeros(k["size"], dtype=np.int64)
    ST.bootstrap_matrix(generator_factory=lambda seed: Gen())
    fx.true("registered call: exactly one draw, integers(0, 97, size=(10000, 97), dtype=int64)",
            len(calls) == 1 and calls[0][0] == (0, 97) and calls[0][1] == {"size": (10000, 97), "dtype": np.int64})
    level = {r: 0.0 for r in ROLES}; level["lam0/fin"] = 0.25
    pop = toy_population(level, noise=0.0)
    e = ST.additive_effect(pop, ST.ALL_CONTRASTS["S-GAP0"], "KS", idx)
    fx.close("constant paired effect: point = 0.25", e["point"], 0.25, 1e-12)
    fx.close("constant paired effect: zero-width interval", e["ci"][1] - e["ci"][0], 0.0, 1e-12)
    fx.true("constant positive effect labelled above0", e["label"] == "above0")
    pop = toy_population(level, noise=0.01)
    e = ST.additive_effect(pop, ST.ALL_CONTRASTS["S-GAP0"], "KS", idx)
    d = np.array([pop.col("lam0/fin", s, "KS") - pop.col("lam0/sel", s, "KS") for s in SEEDS])
    fx.close("known paired effect recovered (mean over seeds of volume means)", e["point"], float(d.mean()), 1e-12)
    ref = np.array([d[:, row].mean() for row in idx])
    lo, hi = np.percentile(ref, [2.5, 97.5])
    fx.close("interval equals an explicit per-replicate loop (lower)", e["ci"][0], float(lo), 1e-12)
    fx.close("interval equals an explicit per-replicate loop (upper)", e["ci"][1], float(hi), 1e-12)
    xt = ST.additive_effect(pop, ST.ALL_CONTRASTS["X-TIME"], "KS", idx)
    d0 = np.mean([pop.col("lam0/fin", s, "KS") - pop.col("lam0/e3", s, "KS") for s in SEEDS])
    d2 = np.mean([pop.col("lam2/fin", s, "KS") - pop.col("lam2/e3", s, "KS") for s in SEEDS])
    fx.close("X-TIME = (lam2 change) - (lam0 change)", xt["point"], float(d2 - d0), 1e-12)
    fx.true("21 contrasts: 10 Tier 1 + 11 Tier 2", len(ST.TIER1_CONTRASTS) == 10 and len(ST.TIER2_CONTRASTS) == 11)
    fx.done()


def classify_independent(s, sg, ct, xt, c2):
    """Second implementation of the class rule (R3): written from 235-candidate section 5 and
    238 LE21 as a decision table on 'predicted side' flags; shares no code with diag_lat_stats."""
    pred, opp = ("above0", "below0") if s > 0 else ("below0", "above0")
    n = "not computed"
    table_gap = {(pred, pred): "GA", (opp, opp): "CON"}
    state = table_gap.get((sg, ct))
    if state is None:
        if n in (sg, ct):
            other = ct if sg == n else sg
            return "unresolved" if other == "unresolved" else "unresolved — measure not computed"
        return "unresolved"
    if state == "CON":
        return "contradicted"
    if xt == n:
        return "gap-associated; selectivity and shared status unresolved — measure not computed"
    if xt == opp:                                   # X-TIME = lam2 - lam0: lam0-favouring side is the opposite side
        return "gap-selective"
    return {pred: "shared", n: "gap-associated; shared status unresolved — measure not computed"}.get(
        c2, "gap-associated (not selective, not shared)")


_N = "not computed"
_LETTER_RANK = {"gap-selective": 2}


def letter_independent(classes):
    """Outcome letter on a list of classes (candidate section 5): A if one is gap-selective,
    else B if one is gap-associated (label starts with 'gap-associated' or is 'shared'), else C."""
    if "gap-selective" in classes:
        return "A"
    return "B" if any(c == "shared" or c.startswith("gap-associated") for c in classes) else "C"


def outcome_independent(ms):
    """Second implementation of the A14 rule BY ENUMERATION; shares no code with diag_lat_stats.
    ms: list of (s, S-GAP0, C-TIME0, X-TIME, C-TIME2). The outcome is provisional iff some way of
    filling in the 'not computed' labels gives another letter than the computed results do.
    A measure is open iff filling in ITS missing labels alone can change the letter."""
    done = ("above0", "below0", "unresolved")
    fills = [list(itertools.product(*[(done if v == _N else (v,)) for v in m[1:]])) for m in ms]
    now = [classify_independent(*m) for m in ms]
    letter = letter_independent(now)
    prov = any(letter_independent([classify_independent(m[0], *f) for m, f in zip(ms, combo)]) != letter
               for combo in itertools.product(*fills))
    open_ids = [i for i, m in enumerate(ms)
                if any(letter_independent(now[:i] + [classify_independent(m[0], *f)] + now[i + 1:]) != letter for f in fills[i])]
    return letter, prov, open_ids


def fixed_epoch_independent(s, sg, ct, ft):
    """Second implementation of the fixed-epoch tag (candidate v0.6.2 section 5, A19)."""
    pred, opp = ("above0", "below0") if s > 0 else ("below0", "above0")
    if sg == pred and ct == pred:
        return {_N: "not computed", pred: "corroborated at fixed epochs"}.get(ft, "not corroborated at fixed epochs")
    if any(v in (opp, "unresolved") for v in (sg, ct)):
        return "not applicable"
    return "not computed"


# Outcome cases (A14): measures as (s, S-GAP0, C-TIME0, X-TIME, C-TIME2); expected letter,
# provisional flag and the contrasts that keep the outcome open, per measure index.
_SEL = (1, "above0", "above0", "below0", "above0")
_SHARED = (1, "above0", "above0", "above0", "above0")
_PLAIN = (1, "above0", "above0", "unresolved", "unresolved")
_CON = (1, "below0", "below0", "below0", "above0")
_UNRES = (1, "above0", "unresolved", "below0", "above0")
OUTCOME_CASES = [
    ("A confirmed: one gap-selective, open classes elsewhere do not matter",
     [_SEL, (1, _N, "above0", _N, "above0"), _UNRES], "A", False, {}),
    ("B confirmed: shared, plain, case (iii) with only C-TIME2 missing, contradicted",
     [_SHARED, _PLAIN, (1, "above0", "above0", "unresolved", _N), _CON], "B", False, {}),
    ("B PROVISIONAL: gap-associated, X-TIME missing",
     [(1, "above0", "above0", _N, "above0"), _UNRES], "B", True, {0: ["X-TIME"]}),
    ("B PROVISIONAL: another measure lacks S-GAP0 and has X-TIME on the selective side",
     [_SHARED, (1, _N, "above0", "below0", "above0")], "B", True, {1: ["S-GAP0"]}),
    ("B CONFIRMED: the measure lacking S-GAP0 has X-TIME off the selective side (class open, letter fixed)",
     [_SHARED, (1, _N, "above0", "above0", "above0")], "B", False, {}),
    ("B CONFIRMED: the measure lacking S-GAP0 has X-TIME unresolved",
     [_SHARED, (1, _N, "above0", "unresolved", "above0")], "B", False, {}),
    ("B CONFIRMED: one gap contrast missing, the other on the opposite side (class 'unresolved — measure not computed')",
     [_SHARED, (1, _N, "below0", _N, _N)], "B", False, {}),
    ("B PROVISIONAL: only the measure that can still be gap-selective is named",
     [_SHARED, (1, _N, "above0", "above0", "above0"), (1, "above0", "above0", _N, _N)], "B", True, {2: ["X-TIME"]}),
    ("C confirmed: nothing gap-associated, nothing missing",
     [_UNRES, _CON], "C", False, {}),
    ("C PROVISIONAL: a missing S-GAP0 could give gap association (X-TIME unresolved: B possible, A not)",
     [_UNRES, (1, _N, "above0", "unresolved", "above0")], "C", True, {1: ["S-GAP0"]}),
    ("C CONFIRMED: one gap contrast missing, the other on the opposite side",
     [_UNRES, (1, "below0", _N, "below0", "above0")], "C", False, {}),
    ("C CONFIRMED: one gap contrast missing, the other unresolved (plain unresolved)",
     [(1, _N, "unresolved", _N, _N)], "C", False, {}),
    ("C PROVISIONAL: both gap contrasts and X-TIME missing",
     [(1, _N, _N, _N, _N)], "C", True, {0: ["S-GAP0", "C-TIME0", "X-TIME"]}),
    ("C PROVISIONAL: one measure can reach B only, another can reach A; both named with their contrasts",
     [(1, _N, "above0", "above0", "above0"), (1, "above0", _N, _N, "above0")], "C", True, {0: ["S-GAP0"], 1: ["C-TIME0", "X-TIME"]}),
    ("L5 (falls): B PROVISIONAL with C-TIME0 missing and X-TIME above 0 (selective side)",
     [(-1, "below0", "below0", "below0", "below0"), (-1, "below0", _N, "above0", "below0")], "B", True, {1: ["C-TIME0"]}),
    ("L5 (falls): B CONFIRMED with C-TIME0 missing and X-TIME below 0",
     [(-1, "below0", "below0", "below0", "below0"), (-1, "below0", _N, "below0", "below0")], "B", False, {}),
]

# Seed-tag cases (A16): (description, S-GAP0 (pooled, per seed), C-TIME0 (pooled, per seed), expected)
_OK, _OK2 = (0.2, [0.1, 0.2, 0.3]), (0.1, [0.05, 0.1, 0.15])
_MIX, _ZSEED, _P0, _NCE, _NCSEED = (0.2, [0.5, 0.3, -0.2]), (0.2, [0.3, 0.3, 0.0]), (0.0, [0.1, -0.1, 0.0]), (None, None), (0.2, [0.3, None, 0.1])
SEED_CASES = [
    ("all seeds share the pooled sign on both contrasts", _OK, _OK2, "seed-consistent"),
    ("negative pooled sign shared by all seeds", (-0.2, [-0.1, -0.2, -0.3]), _OK2, "seed-consistent"),
    ("one seed of opposite sign", _MIX, _OK2, "mixed-seed"),
    ("boundary: a per-seed effect of exactly 0 does not share the sign", _ZSEED, _OK2, "mixed-seed"),
    ("boundary: pooled effect exactly 0", _P0, _OK2, "no pooled direction"),
    ("boundary: pooled effect and every per-seed effect exactly 0", (0.0, [0.0, 0.0, 0.0]), _OK2, "no pooled direction"),
    ("a missing value alone: contrast not computed", _NCE, _OK2, "not computed"),
    ("a missing value alone: one per-seed effect not computed, the others agree", _NCSEED, _OK2, "not computed"),
    ("a computed disagreement together with a missing value on the same contrast", (0.2, [0.5, None, -0.1]), _OK2, "mixed-seed"),
    ("across contrasts: S-GAP0 consistent, C-TIME0 mixed", _OK, _MIX, "mixed-seed"),
    ("across contrasts: S-GAP0 consistent, C-TIME0 per-seed 0", _OK, _ZSEED, "mixed-seed"),
    ("across contrasts: S-GAP0 not computed, C-TIME0 mixed (first match is mixed-seed)", _NCE, _MIX, "mixed-seed"),
    ("across contrasts: S-GAP0 mixed, C-TIME0 not computed", _MIX, _NCE, "mixed-seed"),
    ("across contrasts: S-GAP0 pooled 0, C-TIME0 mixed", _P0, _MIX, "mixed-seed"),
    ("across contrasts: S-GAP0 mixed, C-TIME0 pooled 0", _MIX, _P0, "mixed-seed"),
    ("across contrasts: S-GAP0 not computed, C-TIME0 pooled 0", _NCE, _P0, "no pooled direction"),
    ("across contrasts: S-GAP0 pooled 0, C-TIME0 not computed", _P0, _NCE, "no pooled direction"),
    ("across contrasts: S-GAP0 pooled 0, C-TIME0 one per-seed effect not computed", _P0, _NCSEED, "no pooled direction"),
    ("across contrasts: S-GAP0 consistent, C-TIME0 pooled 0", _OK, _P0, "no pooled direction"),
    ("across contrasts: S-GAP0 consistent, C-TIME0 not computed", _OK, _NCE, "not computed"),
    ("across contrasts: S-GAP0 consistent, C-TIME0 one per-seed effect not computed", _OK, _NCSEED, "not computed"),
    ("across contrasts: both not computed", _NCE, _NCE, "not computed"),
]

# Fixed-epoch cases (A15, A19), the four situations: (description, s, S-GAP0, C-TIME0, X-TIME, C-TIME2, F-TRAJ0(10->30), expected)
FIXED_CASES = [
    ("gap-associated, F-TRAJ0 supported", 1, "above0", "above0", "above0", "above0", "above0", "corroborated at fixed epochs"),
    ("gap-associated (selectivity open), F-TRAJ0 unresolved", 1, "above0", "above0", _N, "above0", "unresolved", "not corroborated at fixed epochs"),
    ("gap-associated, F-TRAJ0 on the opposite side", 1, "above0", "above0", "below0", "above0", "below0", "not corroborated at fixed epochs"),
    ("L5 gap-selective, F-TRAJ0 below 0", -1, "below0", "below0", "above0", "unresolved", "below0", "corroborated at fixed epochs"),
    ("gap-associated, F-TRAJ0 not computed", 1, "above0", "above0", "above0", "above0", _N, "not computed"),
    ("gap association still possible: S-GAP0 missing, C-TIME0 supported", 1, _N, "above0", "below0", "above0", "above0", "not computed"),
    ("gap association still possible: both gap contrasts missing", 1, _N, _N, "below0", "above0", "above0", "not computed"),
    ("gap association still possible, F-TRAJ0 also missing", 1, "above0", _N, "below0", "above0", _N, "not computed"),
    ("excluded: S-GAP0 missing, C-TIME0 on the opposite side (class stays open)", 1, _N, "below0", "below0", "above0", "above0", "not applicable"),
    ("excluded: C-TIME0 missing, S-GAP0 unresolved", 1, "unresolved", _N, "below0", "above0", "above0", "not applicable"),
    ("excluded: unresolved (one supported, one unresolved)", 1, "above0", "unresolved", "below0", "above0", "above0", "not applicable"),
    ("excluded: contradicted", 1, "below0", "below0", "below0", "above0", "above0", "not applicable"),
    ("excluded, F-TRAJ0 not computed: still not applicable", 1, "below0", "below0", "below0", "above0", _N, "not applicable"),
]


def _labels(m, ft="unresolved"):
    return {"S-GAP0": m[1], "C-TIME0": m[2], "X-TIME": m[3], "C-TIME2": m[4], ST.FIXED_EPOCH: ft}


def _outcome_input(ms):
    return {"M%d" % i: {"class": ST.classify(m[0], _labels(m)), "predicted_sign": m[0], "labels": _labels(m)} for i, m in enumerate(ms)}


def t15():
    fx = Fx("T15", "class rules, tags, outcome, not-computed propagation")
    L = ("above0", "below0", "unresolved", ST.NC)
    n_all = n_ok = 0
    for s in (1, -1):
        for sg, ct, xt, c2 in itertools.product(L, repeat=4):
            n_all += 1
            n_ok += ST.classify(s, {"S-GAP0": sg, "C-TIME0": ct, "X-TIME": xt, "C-TIME2": c2}) == classify_independent(s, sg, ct, xt, c2)
    fx.true(f"core class rule equals the independent implementation on all {n_all} label combinations", n_ok == n_all)
    cl = lambda s, sg, ct, xt="unresolved", c2="unresolved": ST.classify(s, {"S-GAP0": sg, "C-TIME0": ct, "X-TIME": xt, "C-TIME2": c2})
    rows = [  # (description, s, S-GAP0, C-TIME0, X-TIME, C-TIME2, expected)
        ("rising measure, X-TIME below 0 -> gap-selective", 1, "above0", "above0", "below0", "above0", ST.C_SELECTIVE),
        ("rising measure, X-TIME above 0 is NOT selective; C-TIME2 supported -> shared", 1, "above0", "above0", "above0", "above0", ST.C_SHARED),
        ("L5 (falls), X-TIME above 0 -> gap-selective", -1, "below0", "below0", "above0", "unresolved", ST.C_SELECTIVE),
        ("L5 (falls), X-TIME below 0 is NOT selective; C-TIME2 below 0 -> shared", -1, "below0", "below0", "below0", "below0", ST.C_SHARED),
        ("I1: gap-associated, not selective, C-TIME2 unresolved", 1, "above0", "above0", "unresolved", "unresolved", ST.C_GA_PLAIN),
        ("I1: C-TIME2 on the opposite side is not 'shared'", 1, "above0", "above0", "unresolved", "below0", ST.C_GA_PLAIN),
        ("contradicted", 1, "below0", "below0", "below0", "above0", ST.C_CONTRADICTED),
        ("one supported, one unresolved -> unresolved", 1, "above0", "unresolved", "below0", "above0", ST.C_UNRESOLVED),
        ("one supported, one opposite -> unresolved", 1, "above0", "below0", "below0", "above0", ST.C_UNRESOLVED),
        ("I2: S-GAP0 missing, C-TIME0 includes 0 -> plain unresolved", 1, ST.NC, "unresolved", "below0", "above0", ST.C_UNRESOLVED),
        ("I2: C-TIME0 missing, S-GAP0 includes 0 -> plain unresolved", 1, "unresolved", ST.NC, "below0", "above0", ST.C_UNRESOLVED),
        ("I2: S-GAP0 missing, C-TIME0 supported -> unresolved, measure not computed", 1, ST.NC, "above0", "below0", "above0", ST.C_UNRES_NC),
        ("I2: C-TIME0 missing, S-GAP0 opposite -> unresolved, measure not computed", 1, "below0", ST.NC, "below0", "above0", ST.C_UNRES_NC),
        ("I2: both missing -> unresolved, measure not computed", 1, ST.NC, ST.NC, "below0", "above0", ST.C_UNRES_NC),
        ("I2: gap-associated, X-TIME missing -> selectivity and shared status unresolved", 1, "above0", "above0", ST.NC, "above0", ST.C_GA_SEL_UNRES),
        ("I2: gap-associated, not selective, C-TIME2 missing -> shared status unresolved", 1, "above0", "above0", "unresolved", ST.NC, ST.C_GA_SHARED_UNRES),
        ("I2: selective is established even if C-TIME2 is missing", 1, "above0", "above0", "below0", ST.NC, ST.C_SELECTIVE),
    ]
    for desc, s, sg, ct, xt, c2, exp in rows:
        fx.true(desc, cl(s, sg, ct, xt, c2) == exp, cl(s, sg, ct, xt, c2))
    # outcome (candidate v0.6.2 section 5, A14)
    for desc, ms, letter, prov, open_c in OUTCOME_CASES:
        exp = {"outcome": letter, "provisional": prov, "open_measures": ["M%d" % i for i in sorted(open_c)],
               "open_contrasts": {"M%d" % i: open_c[i] for i in sorted(open_c)}}
        got = ST.outcome(_outcome_input(ms))
        fx.true("outcome " + desc, got == exp, repr(got))
    n_all = n_ok = 0
    trip = [(s, sg, cg, xt, "unresolved") for s in (1, -1) for sg in L for cg in L for xt in L]
    for ms in [[m] for m in trip] + [[a, b2] for a in trip[:64] for b2 in trip[64:]] + [[a, b2] for a in trip[:64:5] for b2 in trip[:64]]:
        letter, prov, open_ids = outcome_independent(ms)
        got = ST.outcome(_outcome_input(ms))
        n_all += 1
        n_ok += (got["outcome"], got["provisional"], got["open_measures"]) == (letter, prov, ["M%d" % i for i in open_ids])
    fx.true(f"outcome rule equals the independent enumeration on {n_all} single measures and pairs (letter, provisional, open measures)",
            n_ok == n_all, f"{n_ok}/{n_all}")
    n_all = n_ok = 0
    for s, sg, cg, xt, c2 in itertools.product((1, -1), L, L, L, L):
        got = ST.outcome(_outcome_input([(s, sg, cg, xt, c2), (1, "above0", "above0", L[n_all % 4], c2)]))
        letter, prov, open_ids = outcome_independent([(s, sg, cg, xt, c2), (1, "above0", "above0", L[n_all % 4], c2)])
        n_all += 1
        n_ok += (got["outcome"], got["provisional"], got["open_measures"]) == (letter, prov, ["M%d" % i for i in open_ids])
    fx.true(f"outcome rule equals the independent enumeration with C-TIME2 varied ({n_all} pairs with a gap-associated partner)",
            n_ok == n_all, f"{n_ok}/{n_all}")
    fx.true("outcome: open_contrasts names only not-computed contrasts among S-GAP0, C-TIME0, X-TIME, and is empty iff confirmed",
            all((not o["provisional"]) == (o["open_contrasts"] == {}) and all(lab[c] == ST.NC and c in ("S-GAP0", "C-TIME0", "X-TIME") for mid, cs in o["open_contrasts"].items() for lab in [inp[mid]["labels"]] for c in cs)
                for inp in (_outcome_input(ms) for _, ms, _, _, _ in OUTCOME_CASES) for o in [ST.outcome(inp)]))
    bad = _outcome_input([_SHARED]); bad["M0"]["class"] = ST.C_UNRESOLVED
    fx.raises("outcome: a class that disagrees with its labels -> INPUT", C.INPUT, lambda: ST.outcome(bad))
    # tags
    eff = lambda e: {"point": e[0], "per_seed": e[1]}
    for desc, sg, cg, exp in SEED_CASES:
        got = ST.seed_tag({"S-GAP0": eff(sg), "C-TIME0": eff(cg)})
        fx.true("seed tag: %s -> %s" % (desc, exp), got == exp, repr(got))
    fx.true("seed tag: exactly the four registered values occur in the cases",
            {c[3] for c in SEED_CASES} == {"seed-consistent", "mixed-seed", "no pooled direction", "not computed"})
    fx.raises("seed tag: a non-finite effect -> NONFINITE", C.NONFINITE,
              lambda: ST.seed_tag({"S-GAP0": eff((0.2, [0.1, float("nan"), 0.3])), "C-TIME0": eff(_OK2)}))
    for desc, s, sg, cg, xt, c2, ft, exp in FIXED_CASES:
        lab = _labels((s, sg, cg, xt, c2), ft)
        got = ST.fixed_epoch_tag(ST.classify(s, lab), s, lab)
        fx.true("fixed-epoch tag: %s -> %s" % (desc, exp), got == exp, repr(got))
    n_all = n_ok = 0
    for s in (1, -1):
        for sg, cg, xt, c2, ft in itertools.product(L, repeat=5):
            lab = _labels((s, sg, cg, xt, c2), ft)
            n_all += 1
            n_ok += ST.fixed_epoch_tag(ST.classify(s, lab), s, lab) == fixed_epoch_independent(s, sg, cg, ft)
    fx.true(f"fixed-epoch tag equals the independent rule on all {n_all} label combinations", n_ok == n_all, f"{n_ok}/{n_all}")
    fx.raises("fixed-epoch tag: a class that disagrees with two supported gap contrasts -> INPUT", C.INPUT,
              lambda: ST.fixed_epoch_tag(ST.C_UNRESOLVED, 1, _labels(_SHARED, "above0")))
    fx.raises("fixed-epoch tag: unknown class -> INPUT", C.INPUT, lambda: ST.fixed_epoch_tag("not determined", 1, _labels(_SHARED)))
    ct = lambda a, b: ST.coordinate_tag(1, {"S-GAP0": a, "C-TIME0": b})
    fx.true("coordinate tag: same / opposite / not resolved (including not computed)",
            ct("above0", "above0") == "same direction in physical coordinates" and ct("below0", "below0") == "opposite in physical coordinates"
            and ct("above0", "unresolved") == "not resolved in physical coordinates" and ct("above0", ST.NC) == "not resolved in physical coordinates")
    st = lambda **k: ST.stability_label(k)
    fx.true("I4 per-contrast stability: identical on four labels -> stable; one differs or is missing -> unresolved",
            st(pre64="above0", full="above0", halfA="above0", halfB="above0") == "stable"
            and st(pre64="above0", full="above0", halfA="above0", halfB="unresolved") == "Monte Carlo adequacy unresolved"
            and st(pre64="above0", full="above0", halfA="above0") == "Monte Carlo adequacy unresolved")
    # end to end: population -> effects -> class, tags, propagation
    idx = toy_idx(12)
    lv = {r: 0.0 for r in ROLES}
    lv.update({"lam0/sel": 0.1, "lam0/fin": 0.5, "lam0/e10": 0.05, "lam0/e30": 0.3, "lam2/fin": 0.1})

    def build(missing=None, flip=1.0):
        pop = ST.Population(["v%03d" % i for i in range(12)])
        src = {}
        for name, sub in (("KS", None), ("P_swap", None)) + tuple((nm, sb) for nm in ("log_gamma_ratio", "d01", "a01", "TC_G", "phys_lgr", "phys_d01", "phys_a01") for sb in ST.STABILITY_SUBSETS):
            sign = -1.0 if name == "P_swap" else 1.0
            for role in ROLES:
                for si, seed in enumerate(SEEDS):
                    base = np.sin(np.arange(12) * (1.3 + si))
                    wig = 0.01 * np.cos(np.arange(12) * 0.7 + ROLES.index(role) + si)
                    lev = sign * lv[role] * (flip if (name == "TC_G" and sub == "halfB") else 1.0)
                    vals = list(lev + base + wig)
                    if missing and (role, seed, name, sub) in missing:
                        vals[3] = C.NOT_COMPUTED
                    pop.set(role, seed, name, vals, sub)
        return pop.seal()
    phys = {"L2": "phys_lgr", "L3a": "phys_d01", "L3b": "phys_a01"}
    res = ST.run_registered(build(), idx, phys)
    m = res["measures"]
    fx.true("end to end: all six measures gap-selective under a lam0-only rise (L5 mirrored); outcome A",
            all(m[k]["class"] == ST.C_SELECTIVE for k in m) and res["outcome"]["outcome"] == "A")
    fx.true("end to end: outcome A is confirmed with nothing open; the result names its rules version",
            res["outcome"] == {"outcome": "A", "provisional": False, "open_measures": [], "open_contrasts": {}}
            and res["rules_version"] == ST.RULES_VERSION and "v1.0 FROZEN" in ST.RULES_VERSION
            and "6afa3eda250946548b239df64344974d3c3502d23f6b4e22468ca61e6af97327" in ST.RULES_VERSION)
    fx.true("end to end: seed-consistent, corroborated at fixed epochs, stable, same direction in physical coordinates",
            all(m[k]["seed_tag"] == "seed-consistent" and m[k]["fixed_epoch_tag"] == "corroborated at fixed epochs" for k in m)
            and all(m[k]["stability_summary"] == "stable" for k in ("L2", "L3a", "L3b", "L4"))
            and all(m[k]["stability_summary"] == "not applicable" and m[k]["stability_per_contrast"] is None for k in ("L1", "L5"))
            and all(m[k]["coordinate_tag"] == "same direction in physical coordinates" for k in phys)
            and all(m[k]["coordinate_tag"] == "not applicable" for k in ("L1", "L4", "L5")))
    res = ST.run_registered(build(flip=-1.0), idx, phys)
    fx.true("I4: a sign change in one half makes L4 'Monte Carlo adequacy unresolved' (summary and per contrast), class unchanged",
            res["measures"]["L4"]["stability_summary"] == "Monte Carlo adequacy unresolved"
            and res["measures"]["L4"]["stability_per_contrast"]["S-GAP0"]["stability"] == "Monte Carlo adequacy unresolved"
            and res["measures"]["L4"]["class"] == ST.C_SELECTIVE and res["measures"]["L2"]["stability_summary"] == "stable")
    miss = {("lam2/e3", 2027, "TC_G", "full"): 1}
    res = ST.run_registered(build(missing=miss), idx, phys)
    l4 = res["measures"]["L4"]
    fx.true("propagation: one not-computed slice-state (lam2/e3) -> X-TIME and C-TIME2 not computed, S-GAP0 and C-TIME0 computed",
            l4["effects"]["X-TIME"]["label"] == ST.NC and l4["effects"]["C-TIME2"]["label"] == ST.NC
            and l4["effects"]["S-GAP0"]["label"] == "above0" and l4["effects"]["C-TIME0"]["label"] == "above0")
    fx.true("propagation: established gap association is kept; selectivity unresolved; other measures untouched",
            l4["class"] == ST.C_GA_SEL_UNRES and res["measures"]["L2"]["class"] == ST.C_SELECTIVE)
    miss = {("lam0/fin", 2026, n, sb): 1 for n in ("log_gamma_ratio", "d01", "a01", "TC_G") for sb in ("full",)}
    miss.update({("lam0/fin", 2026, "KS", None): 1, ("lam0/fin", 2026, "P_swap", None): 1})
    res = ST.run_registered(build(missing=miss), idx, phys)
    fx.true("propagation: lam0/fin not computed for every measure -> all classes 'unresolved — measure not computed'; outcome C PROVISIONAL",
            all(r["class"] == ST.C_UNRES_NC for r in res["measures"].values())
            and res["outcome"]["outcome"] == "C" and res["outcome"]["provisional"] is True and len(res["outcome"]["open_measures"]) == 6)
    fx.true("propagation: the same case names S-GAP0, C-TIME0 and X-TIME as open for each measure; seed and fixed-epoch tags 'not computed'",
            all(v == ["S-GAP0", "C-TIME0", "X-TIME"] for v in res["outcome"]["open_contrasts"].values()) and len(res["outcome"]["open_contrasts"]) == 6
            and all(r["seed_tag"] == ST.NC and r["fixed_epoch_tag"] == ST.NC for r in res["measures"].values()))
    miss_sel = {("lam0/sel", 2026, n, "full"): 1 for n in ("log_gamma_ratio", "d01", "a01", "TC_G")}
    miss_sel.update({("lam0/sel", 2026, "KS", None): 1, ("lam0/sel", 2026, "P_swap", None): 1})
    r2 = ST.run_registered(build(missing=miss_sel), idx, phys)
    fx.true("propagation: lam0/sel not computed -> only S-GAP0 missing; C-TIME0 supported, X-TIME selective: outcome C PROVISIONAL, S-GAP0 named, fixed-epoch tag 'not computed'",
            all(r["class"] == ST.C_UNRES_NC and r["fixed_epoch_tag"] == ST.NC and r["seed_tag"] == ST.NC for r in r2["measures"].values())
            and r2["outcome"] == {"outcome": "C", "provisional": True, "open_measures": sorted(ST.MEASURES),
                                  "open_contrasts": {k: ["S-GAP0"] for k in sorted(ST.MEASURES)}}, repr(r2["outcome"]))
    miss_30 = {("lam0/e30", 2028, n, "full"): 1 for n in ("log_gamma_ratio", "d01", "a01", "TC_G")}
    miss_30.update({("lam0/e30", 2028, "KS", None): 1, ("lam0/e30", 2028, "P_swap", None): 1})
    r3 = ST.run_registered(build(missing=miss_30), idx, phys)
    fx.true("propagation: lam0/e30 not computed -> classes and outcome A unchanged; fixed-epoch tag 'not computed' (A15)",
            all(r["class"] == ST.C_SELECTIVE and r["fixed_epoch_tag"] == ST.NC and r["seed_tag"] == "seed-consistent" for r in r3["measures"].values())
            and r3["outcome"]["outcome"] == "A" and r3["outcome"]["provisional"] is False)
    fx.true("'not computed' is never a number: point, per_seed and ci are None",
            res["measures"]["L1"]["effects"]["S-GAP0"] == {"point": None, "per_seed": None, "ci": None, "label": ST.NC, "undefined_draws": 0})
    fx.raises("population: wrong number of volumes -> POPULATION_MISMATCH", ST.POPULATION_MISMATCH,
              lambda: ST.Population(["a", "b"]).set("lam0/e3", 2026, "KS", [1.0]))
    fx.raises("population: unsorted volume list -> POPULATION_MISMATCH", ST.POPULATION_MISMATCH, lambda: ST.Population(["b", "a"]))
    fx.raises("population: non-finite value -> NONFINITE", C.NONFINITE,
              lambda: ST.Population(["a", "b"]).set("lam0/e3", 2026, "KS", [1.0, float("nan")]))
    fx.done()


def t17():
    fx = Fx("T17", "L6 (Spearman, undefined draws, share)")
    a = np.array([1.0, 2.0, 2.0, 4.0, 5.0, 7.0]); b = np.array([2.0, 1.0, 3.0, 3.0, 6.0, 5.0])
    ra = [1, 2.5, 2.5, 4, 5, 6]; rb = [2, 1, 3.5, 3.5, 6, 5]
    ma, mb = sum(ra) / 6, sum(rb) / 6
    hand = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    fx.close("Spearman rho with ties equals the hand value from average ranks", ST.spearman(a, b), hand, 1e-12)
    idx = toy_idx(6, 300)
    r = ST.spearman_result(a, np.full(6, 3.0), idx)
    fx.true("constant variable: rho 'not computed', no interval, reason recorded", r["rho"] == ST.NC and r["ci"] is None and r["reason"])
    bs = ST.boot_spearman(a, b, idx)
    ref = np.array([ST.spearman(a[row], b[row]) for row in idx])
    both = ~np.isnan(ref)
    fx.true("bootstrap rho: undefined exactly where a resampled variable is constant", np.array_equal(np.isnan(bs), np.isnan(ref)))
    fx.close("bootstrap rho equals rho recomputed on each resample (explicit loop)", float(np.max(np.abs(bs[both] - ref[both]))), 0.0, 1e-12)
    v = 97; idx97 = toy_idx(v, 2000)
    x = np.arange(v, dtype=float)
    y5 = np.zeros(v); y5[:5] = 1.0                      # P(resample has no 1) = (92/97)^97, about 0.6 %
    r = ST.spearman_result(x, y5, idx97)
    und = int(np.isnan(ST.boot_spearman(x, y5, idx97)).sum())
    fx.true(f"<= 1 % undefined draws ({und} of 2000): excluded, counted, interval computed",
            0 < und <= 20 and r["undefined_draws"] == und and r["ci"] is not None and r["reason"] is None)
    y1 = np.zeros(v); y1[0] = 1.0                       # about 37 % undefined
    r = ST.spearman_result(x, y1, idx97)
    fx.true(f"> 1 % undefined draws ({r['undefined_draws']} of 2000): interval NOT computed, point kept",
            r["undefined_draws"] > 20 and r["ci"] is None and isinstance(r["rho"], float) and r["reason"])
    dn = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, -2.0]); dp = np.array([-0.1, 0.2, -0.3, -0.4, 0.5, -0.6])
    sh = ST.share_both_lower(dn, dp, idx)
    fx.close("share of volumes with NLL lower and PSNR lower = 3/6", sh["share"], 0.5, 1e-15)
    fx.true("share interval lies in [0, 1] and contains the point", 0 <= sh["ci"][0] <= 0.5 <= sh["ci"][1] <= 1)
    # l6_contrast end to end, with a measure not computed and a constant tail share
    lv = {r_: 0.0 for r_ in ROLES}; lv["lam0/fin"] = 0.4
    pop = ST.Population(["v%03d" % i for i in range(12)])
    for name, sub in (("KS", None), ("P_swap", None), ("NLL", None), ("R0_PSNR", None)) + tuple((n, "full") for n in ("log_gamma_ratio", "d01", "a01", "TC_G")):
        for ri, role in enumerate(ROLES):
            for si, seed in enumerate(SEEDS):
                vals = list(lv[role] + 0.1 * np.sin(np.arange(12) * (0.9 + 0.2 * ri) + si + len(name)))
                if name == "TC_G" and role == "lam0/fin" and seed == 2026:
                    vals[0] = C.NOT_COMPUTED
                pop.set(role, seed, name, vals, sub)
    pop.seal()
    i12 = toy_idx(12, 300)
    out = ST.l6_contrast(pop, "S-GAP0", 2026, i12, tau=np.linspace(0, 0.01, 12))
    fx.true("l6_contrast: six measures x two targets, three tail correlations, share with interval",
            set(out["measures"]) == set(ST.MEASURES) and set(out["tail"]) == {"vs_dKS", "vs_dNLL", "vs_dPSNR"}
            and isinstance(out["measures"]["L1"]["vs_psnr"]["rho"], float) and out["share_nll_lower_psnr_lower"]["ci"] is not None)
    tau12 = np.linspace(0, 0.01, 12)
    fx.true("l6_contrast: L4 not computed in seed 2026 -> L6 for seed 2026 reports 'not computed' on S-GAP0",
            out["measures"]["L4"]["vs_psnr"]["rho"] == ST.NC and out["measures"]["L4"]["vs_nll"]["rho"] == ST.NC
            and isinstance(out["measures"]["L2"]["vs_psnr"]["rho"], float))
    for other in (2027, 2028):
        o = ST.l6_contrast(pop, "S-GAP0", other, i12, tau=tau12)["measures"]["L4"]
        d_m = pop.col("lam0/fin", other, "TC_G", "full") - pop.col("lam0/sel", other, "TC_G", "full")
        d_p = pop.col("lam0/fin", other, "R0_PSNR") - pop.col("lam0/sel", other, "R0_PSNR")
        fx.true(f"REGRESSION (cross-seed): the missing state of seed 2026 does not suppress L4 for seed {other} on the same contrast",
                isinstance(o["vs_psnr"]["rho"], float) and isinstance(o["vs_nll"]["rho"], float) and o["vs_psnr"]["ci"] is not None)
        fx.close(f"REGRESSION: seed {other} L4-vs-PSNR rho equals rho computed directly from that seed's values",
                 o["vs_psnr"]["rho"], ST.spearman(d_m, d_p), 1e-15)
    fx.true("seed_differences: seed 2026 not computed, seeds 2027 and 2028 computed, on the same contrast",
            ST.seed_differences(pop, ST.ALL_CONTRASTS["S-GAP0"], "TC_G", 2026, "full") is C.NOT_COMPUTED
            and all(isinstance(ST.seed_differences(pop, ST.ALL_CONTRASTS["S-GAP0"], "TC_G", sd, "full"), np.ndarray) for sd in (2027, 2028)))
    fx.true("pooled rule unchanged: the same missing state makes the pooled S-GAP0 effect of L4 'not computed'",
            ST.additive_effect(pop, ST.ALL_CONTRASTS["S-GAP0"], "TC_G", i12, "full")["label"] == ST.NC
            and ST.per_seed_differences(pop, ST.ALL_CONTRASTS["S-GAP0"], "TC_G", "full") is C.NOT_COMPUTED)
    fx.true("contrast without the missing state is unaffected in every seed (S-CTL0)",
            all(isinstance(ST.l6_contrast(pop, "S-CTL0", sd, i12, tau=tau12)["measures"]["L4"]["vs_nll"]["rho"], float) for sd in SEEDS))
    fx.raises("seed_differences: unknown seed -> INPUT", C.INPUT, lambda: ST.seed_differences(pop, ST.ALL_CONTRASTS["S-GAP0"], "KS", 1999))
    out = ST.l6_contrast(pop, "S-GAP0", 2027, i12, tau=np.zeros(12))
    fx.true("l6_contrast: constant tail share -> rho 'not computed'", all(t["rho"] == ST.NC for t in out["tail"].values()))
    fx.done()


def t18():
    fx = Fx("T18", "resource and memory guards")
    gib = GD.GIB
    with tempfile.TemporaryDirectory() as tmp:
        g = GD.WriteGuard(tmp, ceiling=GD.LEDGER_BYTES + GD.WriteGuard.charge(100) + 10, reserve=0, free_fn=lambda: 10 * gib)
        g.write("a.bin", b"x" * 100)
        fx.close("cumulative count = ledger + charge(100)", g.written, GD.LEDGER_BYTES + GD.WriteGuard.charge(100), 0)
        fx.raises("write above the ceiling -> RESOURCE_LIMIT", GD.RESOURCE_LIMIT, lambda: g.write("b.bin", b"y" * 100))
        fx.true("refused write left no file and did not change the count",
                not os.path.exists(os.path.join(tmp, "b.bin")) and g.written == GD.LEDGER_BYTES + GD.WriteGuard.charge(100))
        fx.raises("write-once: overwrite refused -> GUARD_STATE", GD.GUARD_STATE, lambda: g.write("a.bin", b"z"))
        fx.true("read verifies the sidecar", g.read("a.bin") == b"x" * 100)
        with open(os.path.join(tmp, "a.bin"), "wb") as fh:
            fh.write(b"tampered")
        fx.raises("changed file -> SIDECAR_MISMATCH", GD.SIDECAR_MISMATCH, lambda: g.read("a.bin"))
        fx.raises("path outside the root -> GUARD_STATE", GD.GUARD_STATE, lambda: g.write("../escape.bin", b"q"))
    with tempfile.TemporaryDirectory() as tmp:
        g = GD.WriteGuard(tmp, ceiling=gib, reserve=64 * gib, free_fn=lambda: 64 * gib + GD.LEDGER_BYTES)
        fx.raises("write that would leave less than the reserve free -> RESOURCE_LIMIT", GD.RESOURCE_LIMIT,
                  lambda: g.write("c.bin", b"x" * 100))
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "existing.txt"), "w").close()
        before = sorted(os.listdir(tmp))
        fx.raises("directory that is not new -> GUARD_STATE", GD.GUARD_STATE, lambda: GD.WriteGuard(tmp, ceiling=gib, reserve=0))
        fx.true("refusal left the directory exactly as found", sorted(os.listdir(tmp)) == before)
    try:
        GD.WriteGuard(tempfile.gettempdir())
        fx.true("ceiling and reserve are required parameters", False)
    except TypeError:
        fx.true("ceiling and reserve are required parameters (no default limit in the guard)", True)
    fx.true("peak resident memory is a positive number of bytes", GD.peak_rss_bytes() > 1024 * 1024)
    fx.raises("memory limit below the fixture's own use -> RESOURCE_LIMIT", GD.RESOURCE_LIMIT, lambda: GD.check_memory(1024))
    fx.true("memory within the limit returns the peak", GD.check_memory(10 ** 15) == GD.peak_rss_bytes() or GD.check_memory(10 ** 15) > 0)
    calls = {"n": 0}

    def oom():
        with GD.memory_guarded(10 ** 15):
            calls["n"] += 1
            raise MemoryError("synthetic out-of-memory")
    fx.raises("a raised out-of-memory error becomes RESOURCE_LIMIT", GD.RESOURCE_LIMIT, oom)
    fx.true("no retry: the guarded block ran exactly once", calls["n"] == 1)

    def over():
        with GD.memory_guarded(1024):
            calls["n"] += 1
    fx.raises("memory_guarded: peak above the limit at block end -> RESOURCE_LIMIT", GD.RESOURCE_LIMIT, over)
    fx.done()


def _guard_script(make, codes):
    """Run the same operations on a guard factory; return a comparable trace."""
    trace = []

    def step(name, fn):
        try:
            trace.append((name, "ok", fn()))
        except Exception as e:                       # typed errors are compared by their mapped code
            trace.append((name, "error", codes.get(getattr(e, "code", type(e).__name__), getattr(e, "code", type(e).__name__))))
    with tempfile.TemporaryDirectory() as tmp:
        free = {"v": 100 * 2 ** 30}
        g = make(tmp, 5000, 2 ** 30, lambda: free["v"])
        step("written after create", lambda: g.written)
        step("write a", lambda: g.write("a.bin", b"a" * 1000))
        step("written", lambda: g.written)
        step("overwrite a", lambda: g.write("a.bin", b"b"))
        step("write sub/b", lambda: g.write("sub/b.bin", b"b" * 2000))
        step("written", lambda: g.written)
        step("write above ceiling", lambda: g.write("c.bin", b"c" * 2000))
        step("written unchanged", lambda: g.written)
        step("advisory check small", lambda: g.check(10))
        step("delete a", lambda: g.delete("a.bin"))
        step("written after delete", lambda: g.written)
        step("read b", lambda: g.read("sub/b.bin"))
        step("reserved path", lambda: g.write(".." + os.sep + "x.bin", b"x"))
        free["v"] = 2 ** 30 + 100
        step("below reserve", lambda: g.write("d.bin", b"d" * 10))
        free["v"] = 100 * 2 ** 30
        g2 = make(tmp, 5000, 2 ** 30, lambda: free["v"])
        step("second guard sees the same count", lambda: g2.written)
        step("second guard write", lambda: g2.write("e.bin", b"e" * 100))
        step("first guard sees it", lambda: g.written)
        step("listing", lambda: sorted(n for n in os.listdir(tmp) if not n.startswith(".")))
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "old.txt"), "w").close()
        step("not-new directory", lambda: make(tmp, 5000, 0, lambda: 2 ** 40))
        step("listing after refusal", lambda: sorted(os.listdir(tmp)))
    return trace


def t19(archive):
    fx = Fx("T19", "ports against the archived D-UNC originals")
    for name, want in DUNC_SHA.items():
        got = hashlib.sha256(open(os.path.join(archive, name), "rb").read()).hexdigest()
        fx.true(f"archived {name} has its registered sha256", got == want, got)
    if not all(c["pass"] for c in fx.checks):
        fx.done(); return
    sys.path.insert(0, archive)
    logging.getLogger("diag_unc").setLevel(logging.CRITICAL)
    import diag_unc_core as UC
    import diag_unc_guard as UG
    import diag_unc_stats as US
    fx.true("port reproduces the registered D-UNC bootstrap matrix (seed 20261114)",
            ST.matrix_identity(ST.bootstrap_matrix(20261114)) == DUNC_MATRIX_SHA == US.matrix_identity(US.bootstrap_matrix()))
    fx.true("D-LAT matrix (seed 20261120) differs from the D-UNC matrix and equals the original function at that seed",
            ST.matrix_identity(ST.bootstrap_matrix()) == US.matrix_identity(US.bootstrap_matrix(20261120)) != DUNC_MATRIX_SHA)
    idx = toy_idx(31, 500)
    fx.true("matrix_identity identical", ST.matrix_identity(idx) == US.matrix_identity(idx))
    fx.true("multiplicities identical", np.array_equal(ST.multiplicities(idx), US.multiplicities(idx)))
    gen = np.random.Generator(np.random.PCG64(SELFTEST_SEED))
    boots = [gen.standard_normal(500), np.zeros(500), -np.abs(gen.standard_normal(500)) - 0.1, np.abs(gen.standard_normal(500)) + 0.1]
    fx.true("percentile interval and sign label identical (four cases incl. zero width, both signs)",
            all(ST._percentile_ci(b) == US._percentile_ci(b) and ST.sign_label(ST._percentile_ci(b)) == US.sign_label(US._percentile_ci(b))
                for b in boots) and ST.sign_label(None) == US.sign_label(None))
    cases = {"continuous": (gen.standard_normal(31), gen.standard_normal(31)),
             "ties": (np.round(gen.standard_normal(31)), np.round(2 * gen.standard_normal(31))),
             "undefined replicates": (np.arange(31.0), (np.arange(31) == 0).astype(float)),
             "constant": (np.arange(31.0), np.ones(31))}
    for name, (a, b) in cases.items():
        x, y = ST.boot_spearman(a, b, idx), US.boot_spearman(a, b, idx)
        same_nan = np.array_equal(np.isnan(x), np.isnan(y))
        fx.true(f"boot_spearman identical: {name} ({int(np.isnan(x).sum())} undefined)", same_nan and np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)]))
        p, q = ST.spearman(a, b), UC.spearman(a, b)
        fx.true(f"spearman identical: {name}", (math.isnan(p) and math.isnan(q)) or abs(p - q) <= 1e-15)
    fx.true("contrast tables identical: names, order, weights (Tier 1 and Tier 2)",
            list(ST.TIER1_CONTRASTS.items()) == list(US.TIER1_CONTRASTS.items())
            and list(ST.TIER2_CONTRASTS.items()) == list(US.TIER2_CONTRASTS.items()))
    fx.true("stability subset labels identical", ST.STABILITY_SUBSETS == US.STABILITY_SUBSETS
            and all(np.array_equal(C.STABILITY_SUBSETS[k], np.asarray(v)) for k, v in
                    {"full": np.arange(128), "pre64": np.arange(64), "halfA": np.arange(0, 64), "halfB": np.arange(64, 128)}.items()))
    labs = [{"pre64": "above0", "full": "above0", "halfA": "above0", "halfB": "above0"},
            {"pre64": "above0", "full": "above0", "halfA": "above0", "halfB": "below0"}, {"full": "above0"}]
    fx.true("stability_label identical", all(ST.stability_label(l) == US.stability_label(l) for l in labs))
    # paired additive effect: same per-volume values through both populations
    vols = ["v%03d" % i for i in range(31)]
    up, lp = US.Population(vols), ST.Population(vols)
    for ri, role in enumerate(ROLES):
        for si, seed in enumerate(SEEDS):
            vals = np.sin(np.arange(31) * (0.3 + 0.05 * ri) + si) + 0.2 * ri
            for i, vid in enumerate(vols):
                up.add(role, seed, vid, {"subsets": {"full": {"K": 128, "v": float(vals[i])}}})
            lp.set(role, seed, "v", vals, "full")
    up.seal(); lp.seal()
    worst = 0.0; same_label = True
    for cname, w in ST.ALL_CONTRASTS.items():
        diff = np.array([sum(wt * up.col(role, seed, lambda r: r["subsets"]["full"]["v"]) for role, wt in w.items()) for seed in up.seeds])
        o_point = float(diff.mean(axis=1).mean()); o_ci = US._percentile_ci(diff[:, idx].mean(axis=2).mean(axis=0))
        e = ST.additive_effect(lp, w, "v", idx, "full")
        worst = max(worst, abs(e["point"] - o_point), abs(e["ci"][0] - o_ci[0]), abs(e["ci"][1] - o_ci[1]))
        same_label &= e["label"] == US.sign_label(o_ci)
    fx.close("paired effect and interval equal the original pooling rule on all 21 contrasts", worst, 0.0, 1e-15)
    fx.true("sign labels equal on all 21 contrasts", same_label)
    cmap = {"DIAG_UNC_INPUT_INVALID": "STATE", "DIAG_LAT_GUARD_STATE": "STATE", "DIAG_UNC_RESOURCE_LIMIT": "LIMIT",
            "DIAG_LAT_RESOURCE_LIMIT": "LIMIT", "DIAG_UNC_REPRODUCTION_MISMATCH": "SIDECAR", "DIAG_LAT_SIDECAR_MISMATCH": "SIDECAR"}
    t_o = _guard_script(lambda root, c, r, f: UG.WriteGuard(root, ceiling=c, reserve=r, free_fn=f), cmap)
    t_p = _guard_script(lambda root, c, r, f: GD.WriteGuard(root, ceiling=c, reserve=r, free_fn=f), cmap)
    fx.true(f"write guard: identical trace on {len(t_o)} operations (counts, digests, refusals, listings)", t_o == t_p,
            "" if t_o == t_p else repr([(a, b) for a, b in zip(t_o, t_p) if a != b][:2]))
    fx.true("write guard trace contains accepted writes, both limit refusals and both state refusals",
            [s for _, s, _ in t_o].count("error") == 5 and sum(1 for _, s, v in t_o if v == "LIMIT") == 2)
    fx.true("accounting constants identical", (GD.SIDECAR_BYTES, GD.LEDGER_BYTES, GD.LEDGER_REWRITE_BYTES)
            == (UG.SIDECAR_BYTES, UG.LEDGER_BYTES, UG.LEDGER_REWRITE_BYTES))
    fx.done()


def benchmark():
    d, k = C.D_FREE, C.K_BANK
    gen = np.random.Generator(np.random.PCG64(SELFTEST_SEED))
    zs = gen.standard_normal(d).astype(np.float32)
    ub = gen.standard_normal((k, d)).astype(np.float32); zb = gen.standard_normal((k, d)).astype(np.float32)
    ut = gen.standard_normal(d).astype(np.float32); r0 = (0.1 * gen.standard_normal(d)).astype(np.float32)
    j = gen.standard_normal(k)
    reps = 20
    t = time.perf_counter()
    for _ in range(reps):
        C.ks_normal(zs)
    t_ks = (time.perf_counter() - t) / reps
    t = time.perf_counter()
    for _ in range(reps):
        C.l1_measures(zs); C.tail_set(ut); C.bank_measures(ut, r0, zs, ub, zb, j, [])
    t_all = (time.perf_counter() - t) / reps
    n_ss = 97 * 28
    return {"ks_seconds_per_call": t_ks, "core_seconds_per_slice_state_standardised": t_all,
            "slice_states": n_ss, "projected_ks_seconds_total": t_ks * n_ss,
            "projected_core_seconds_total_standardised": t_all * n_ss,
            "note": "synthetic float32 inputs, d = 13,824, K = 128, four subset labels; sandbox CPU; "
                    "physical complements roughly double the bank part; flow passes not included"}


def main(out_json=None, dunc_archive=None, without_t19=False):
    logging.basicConfig(level=logging.CRITICAL)      # expected typed stops are exercised on purpose
    counter = {"n": 0}

    class H(logging.Handler):
        def emit(self, rec):
            if rec.levelno >= logging.ERROR:
                counter["n"] += 1
    C.log.addHandler(H()); C.log.setLevel(logging.DEBUG)
    for lg in (ST.log, GD.LOG):
        lg.setLevel(logging.CRITICAL)
    t0 = time.perf_counter()
    for f in (t1, t2, t3, t4, t5, t6, t7, t8, t12a, t13, t16, t14, t15, t17, t18):
        f()
    t19_status = "run"
    if dunc_archive:
        t19(dunc_archive)
    elif without_t19:
        t19_status = "NOT RUN (--without-t19 given explicitly)"
    else:
        t19_status = "NOT RUN: no --dunc-archive given -> the run FAILS"
        fxm = Fx("T19", "ports against the archived D-UNC originals")
        fxm.true("archived D-UNC files supplied (--dunc-archive DIR)", False, "not supplied")
        fxm.done()
    n_raise = sum(1 for r in RESULTS for c in r["checks"] if str(c.get("note", "")).startswith("raised"))
    fxl = Fx("LOG", "every typed stop is logged as an error")
    fxl.true(f"error log records ({counter['n']}) >= typed stops exercised ({n_raise})", counter["n"] >= n_raise)
    fxl.done()
    bench = benchmark()
    ok = all(r["pass"] for r in RESULTS)
    here = os.path.dirname(os.path.abspath(__file__))
    files = {n: hashlib.sha256(open(os.path.join(here, n), "rb").read()).hexdigest()
             for n in ("diag_lat_core.py", "diag_lat_stats.py", "diag_lat_guard.py", "diag_lat_selftest.py")}
    rep = {"banner": BANNER, "selftest_version": __version__, "core_version": C.__version__,
           "stats_version": ST.__version__, "guard_version": GD.__version__, "t19": t19_status,
           "python": sys.version.split()[0], "numpy": np.__version__, "selftest_seed": SELFTEST_SEED,
           "fixtures": RESULTS, "all_pass": ok, "benchmark": bench, "files_sha256": files,
           "elapsed_seconds": time.perf_counter() - t0}
    print(BANNER)
    for r in RESULTS:
        print(f"{r['id']:5s} {'PASS' if r['pass'] else 'FAIL'}  {r['title']}")
        for c in r["checks"]:
            res = "" if c["residual"] is None else f" residual {c['residual']:.3e} (tol {c['tolerance']})"
            obs = f" value {c['observed']!r}" if c["tolerance"] == "recorded" else ""
            print(f"      [{'ok' if c['pass'] else 'FAIL'}] {c['check']}{res}{obs}")
    n_checks = sum(len(r["checks"]) for r in RESULTS)
    print(f"fixtures {sum(r['pass'] for r in RESULTS)}/{len(RESULTS)} passed; checks {n_checks}; all_pass={ok}; T19: {t19_status}")
    print("benchmark:", json.dumps(bench))
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(rep, fh, indent=1, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=BANNER)
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument("--dunc-archive", default=None, help="folder with the archived diag_unc_core/stats/guard.py (T19)")
    ap.add_argument("--without-t19", action="store_true", help="skip T19 explicitly (reported as NOT RUN)")
    a = ap.parse_args()
    sys.exit(main(a.out, a.dunc_archive, a.without_t19))
