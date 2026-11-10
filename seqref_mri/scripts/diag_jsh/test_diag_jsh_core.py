# SEQREF-MRI-DIAGJSHTEST v0.2.1 -- synthetic fixture tests for diag_jsh_core (mini-spec v0.5 §7)
# LIFETIME: KEEP
# =============================================================================
# Synthetic data only. Each check prints "ok"/"FAIL" with a label; exit 0
# only if every check passes. Covers §7 items 1-7 for the scientific core;
# the real-data adapter (DIAG-I2) is tested separately later.
# Changelog (v0.2, 2026-11-09): refusal tests R1-R8 for the review findings
#   (non-finite, shape/broadcast, draw indices, empty replay, NLL fields,
#   subgroup structure, labels); replay calls pass the expected count.
# Changelog (v0.1, 2026-11-09): first fixture set.
# =============================================================================
from __future__ import annotations

import logging
import math
import sys

import numpy as np

import diag_jsh_core as J

logging.basicConfig(level=logging.CRITICAL)
RES = []


def ok(cond, label, detail=""):
    RES.append(bool(cond))
    print(f"{'ok  ' if cond else 'FAIL'} {label}" + (f"  [{detail}]" if
                                                     detail and not cond
                                                     else ""))


def raises(code, fn, label):
    try:
        fn()
    except J.DiagError as e:
        ok(e.code == code, label, f"got {e.code}")
        return
    ok(False, label, "no error")


# 1. NLL identity, contribution rule, units ------------------------------------
d = 13824
good = {"nll_per_dim": 1.25, "nll_prior_per_dim": 1.5,
        "nll_logdet_per_dim": -0.25, "nll_decomp_abs_dev": 1e-6}
J.check_nll_identity(good, d)
ok(True, "1a valid row passes NLL identity")
bad = dict(good, nll_logdet_per_dim=-0.2499)
raises("DIAG_NLL_IDENTITY", lambda: J.check_nll_identity(bad, d),
       "1b recomposition off by 1e-4 refused")
dev = dict(good, nll_decomp_abs_dev=1.0)
raises("DIAG_NLL_IDENTITY", lambda: J.check_nll_identity(dev, d),
       "1c stored deviation above total tolerance refused")
raises("DIAG_NLL_FIELD", lambda: J.check_nll_identity(
    dict(good, nll_per_dim=float("nan")), d), "1d non-finite field refused")
c = J.contributions({"estimate": -0.01, "ci": (-0.02, -0.005)}, -0.012, 0.002)
ok(c["reported"] and abs(c["J_share"] - 1.2) < 1e-12 and
   abs(c["B_share"] + 0.2) < 1e-12, "1e shares > 100% allowed (J 120%, B -20%)")
ok(not J.contributions({"estimate": -0.01, "ci": (-0.02, 0.001)}, -0.01,
                       0.0)["reported"], "1f dNLL interval incl. 0 -> no share")
ok(not J.contributions({"estimate": -5e-4, "ci": (-6e-4, -4e-4)}, -5e-4,
                       0.0)["reported"], "1g |dNLL| < 1e-3 -> no share")

# 2. spectral partition ----------------------------------------------------------
bi = J.band_index()
cnt = [int((bi == b).sum()) for b in range(3)]
ok(sum(cnt) == 96 * 96 and min(cnt) > 0, "2a bands partition all 96^2 "
   "coefficients", str(cnt))
k = np.arange(96) - 48
r2 = k[:, None] ** 2 + k[None, :] ** 2
ok(bi[48, 48] == 0 and bi[0, 0] == 2 and
   bool(((r2 == 511) <= (bi == 0)).all()) and
   bool(((r2 == 512) <= (bi == 1)).all()) and
   bool(((r2 == 2047) <= (bi == 1)).all()) and
   bool(((r2 == 2048) <= (bi == 2)).all()),
   "2b DC low, corner high, half-open boundaries at r^2 = 512 / 2048")
rng = np.random.default_rng(1)
xt = rng.normal(size=(96, 96)) + 1j * rng.normal(size=(96, 96))
xh = xt + 0.1 * (rng.normal(size=(96, 96)) + 1j * rng.normal(size=(96, 96)))
mask = np.zeros(96, int)
mask[44:52] = 1
mask[::4] = 1
rec = J.spectral_record(xh, xt, mask)
ok(abs(rec["sc_cells"].sum() - rec["sc_full"]) <= 1e-9 * rec["sc_full"],
   "2c S-C six cells sum to the full spectrum error")
ok(abs(rec["sc_full"] - np.sum(np.abs(xh - xt) ** 2)) <= 1e-9 *
   rec["sc_full"], "2d orthonormal Parseval (complex)")
pm = float(np.mean((np.abs(xh) - np.abs(xt)) ** 2))
ok(abs(rec["sm_pixel_mse"] - pm) <= 1e-12 * max(pm, 1e-30),
   "2e S-M cell sum / 96^2 = magnitude pixel MSE")
# plane wave: single kx frequency -> known band & split
kx, ky = 30, 0                 # r^2 = 900 -> mid band
img = np.exp(2j * np.pi * (kx * np.arange(96)[None, :] +
                           ky * np.arange(96)[:, None]) / 96)
F = J.fft2c(img)
pos = np.unravel_index(np.argmax(np.abs(F)), F.shape)
ok(bi[pos] == 1 and (pos[1] - 48) == kx and (pos[0] - 48) == ky,
   "2f plane wave lands at its frequency in the mid band", str(pos))
e = J.spectral_record(img + xt, xt, mask)
col = pos[1]
split = 0 if mask[col] else 1
ok(abs(e["sc_cells"][1, split] - e["sc_full"]) <= 1e-9 * e["sc_full"],
   "2g plane-wave error entirely in its (band, split) cell")
z = J.spectral_record(xt, xt, mask)
ok(z["sc_full"] == 0.0, "2h zero error accepted (absolute tolerance)")
raises("DIAG_ZERO_REFERENCE", lambda: J.spectral_record(
    xh, np.zeros((96, 96)), mask), "2i zero reference energy refused")
raises("DIAG_GRID", lambda: J.spectral_record(xh[:95], xt[:95], mask),
       "2j wrong grid refused")
ok(abs(J.hf_fraction(img) - 0.0) < 1e-12 and
   abs(J.hf_fraction(np.exp(2j * np.pi * 47 * np.arange(96)[None, :] / 96 +
                            2j * np.pi * 47 * np.arange(96)[:, None] / 96))
       - 1.0) < 1e-9, "2k HF fraction 0 for mid-band wave, 1 for corner wave")
try:                                   # registered torch fft2c, if present
    import torch
    t = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(
        torch.from_numpy(xt), dim=(-2, -1)), norm="ortho"), dim=(-2, -1))
    ok(np.allclose(t.numpy(), J.fft2c(xt), rtol=0, atol=1e-10),
       "2l NumPy fft2c equals the registered torch convention")
except ImportError:
    print("skip 2l torch not installed (checked in DIAG-I2 against the "
          "pinned fft2c)")

# 3. pairing / replay checks --------------------------------------------------
st = {("v1", "f", 0): {"ckpt": "a", "mask": "m0", "psnr": 30.0,
                       "nll_per_dim": 1.0},
      ("v1", "f", 1): {"ckpt": "a", "mask": "m1", "psnr": 31.0,
                       "nll_per_dim": 1.1}}
rp = {kk: dict(vv) for kk, vv in st.items()}
rp[("v1", "f", 1)]["psnr"] += 5e-4
out = J.check_replay(st, rp, 2)
ok(out["bitwise_equal_slices"] == 1 and abs(out["max_abs_psnr_dev"] -
                                            5e-4) < 1e-12,
   "3a replay within 0.001 dB accepted; deviations recorded")
rp2 = {kk: dict(vv) for kk, vv in st.items()}
rp2[("v1", "f", 0)]["psnr"] += 2e-3
raises("DIAG_REPRODUCTION_MISMATCH", lambda: J.check_replay(st, rp2, 2),
       "3b replay PSNR off by 0.002 dB refused")
rp3 = {kk: dict(vv) for kk, vv in st.items()}
rp3[("v1", "f", 0)]["mask"] = "other"
raises("DIAG_REPRODUCTION_MISMATCH", lambda: J.check_replay(st, rp3, 2),
       "3c mask identity must be exact")
rp4 = dict(reversed(list(st.items())))
raises("DIAG_POPULATION_MISMATCH", lambda: J.check_replay(st, rp4, 2),
       "3d case order must be identical")

# 4. volume aggregation ---------------------------------------------------------
ent = [("A", "a", 0), ("A", "a", 1), ("B", "b", 0)]
vm = J.volume_means({("a", 0): 1.0, ("a", 1): 3.0, ("b", 0): 5.0}, ent)
ok(vm == {"A": 2.0, "B": 5.0}, "4a volume means (registered weighting)")

# 5. thirds ----------------------------------------------------------------------
vals = {f"v{i:02d}": float(v) for i, v in enumerate([5, 1, 3, 3, 2, 9, 7])}
t = J.thirds(vals, descending=False)
ok([len(g) for g in t] == [3, 2, 2] and t[0] == ["v01", "v04", "v02"],
   "5a V=7: sizes 3/2/2, hardest first, ties by volume ID", str(t))
t2 = J.thirds(vals, descending=True)
ok(t2[0] == ["v05", "v06", "v00"], "5b descending (highest HF first)",
   str(t2))
raises("DIAG_THIRDS", lambda: J.thirds({"a": 1.0, "b": 2.0},
                                       descending=False), "5c V < 3 refused")

# 6. bootstrap ------------------------------------------------------------------
V = 7
dr = J.make_draws(V, [[3, 2, 2], [3, 2, 2]], b=500)
dr2 = J.make_draws(V, [[3, 2, 2], [3, 2, 2]], b=500)
ok(np.array_equal(dr["v"], dr2["v"]) and
   all(np.array_equal(a, b) for s1, s2 in zip(dr["strata"], dr2["strata"])
       for a, b in zip(s1, s2)), "6a draws reproducible from DIAG_SEED")
ok(dr["v"].shape == (500, 7) and [a.shape for a in dr["strata"][0]] ==
   [(500, 3), (500, 2), (500, 2)], "6b generation order and shapes")
rs = np.random.default_rng(5)
R = rs.normal(30, 1, size=(3, V))
A = R + 0.2
de = J.delta_estimate(A, R, dr["v"])
ok(abs(de["estimate"] - 0.2) < 1e-12 and abs(de["ci"][0] - 0.2) < 1e-9 and
   abs(de["ci"][1] - 0.2) < 1e-9, "6c constant shift: estimate and CI = 0.2")
groups = [[0, 1, 2], [3, 4], [5, 6]]
A2 = R.copy()
A2[:, 0:3] += 1.0
sg = J.subgroup_estimates(A2, R, groups, dr["strata"][0])
ok(abs(sg["thirds"][0]["estimate"] - 1.0) < 1e-12 and
   abs(sg["thirds"][2]["estimate"]) < 1e-12 and
   abs(sg["contrast_first_minus_last"]["estimate"] - 1.0) < 1e-12 and
   [g["n"] for g in sg["thirds"]] == [3, 2, 2],
   "6d stratified subgroups: fixed sizes, first-minus-last contrast")
x = np.tile(np.arange(V, dtype=float), (3, 1))
sp = J.spearman_estimate(x, 2 * x + 1, dr["v"])
ok(abs(sp["estimate"] - 1.0) < 1e-12, "6e Spearman = 1 for monotone data")
d3 = J.make_draws(3, [], b=2000)            # P(constant resample) = 1/9
x3 = np.tile(np.arange(3, dtype=float), (3, 1))
sp3 = J.spearman_estimate(x3, x3 + 1, d3["v"])
ok(sp3["undefined_draws"] > 0.01 * 2000 and sp3["ci"] is None,
   "6f >1% undefined draws (constant resamples) -> interval not computed",
   str(sp3["undefined_draws"]))
d5 = J.make_draws(5, [], b=10000)           # P(constant) = 5^-4 = 0.16%
x5 = np.tile(np.arange(5, dtype=float), (3, 1))
sp5 = J.spearman_estimate(x5, x5 ** 2, d5["v"])
ok(0 < sp5["undefined_draws"] <= 0.01 * 10000 and sp5["ci"] is not None,
   "6f2 <=1% undefined: excluded from the percentiles, count reported",
   str(sp5["undefined_draws"]))
Vb = 60
drb = J.make_draws(Vb, [], b=2000)
xb = np.tile(np.arange(Vb, dtype=float), (3, 1))
yb = xb + np.random.default_rng(2).normal(0, 5, size=(3, Vb))
spb = J.spearman_estimate(xb, yb, drb["v"])
ok(spb["undefined_draws"] == 0 and spb["ci"] is not None and
   spb["ci"][0] <= spb["estimate"] <= spb["ci"][1],
   "6g defined draws: interval computed around the estimate")
ok(np.allclose(J._ranks(np.array([[3.0, 1.0, 3.0, 2.0]])),
               [[3.5, 1.0, 3.5, 2.0]]), "6h average ranks for ties")
raises("DIAG_CORRELATION_UNDEFINED", lambda: J.spearman_estimate(
    np.ones((3, V)), x, dr["v"]), "6i constant observed vector refused")
Rn = np.full((3, V), 2.0)
re_ = J.ratio_estimate(Rn * 1.005, Rn, dr["v"])
ok(abs(re_["estimate"] - 0.005) < 1e-12 and
   J.density_label(*re_["ci"]) == "preserved",
   "6j registered NLL ratio (ratio of means) and density label")
raises("DIAG_RATIO_UNDEFINED", lambda: J.ratio_estimate(
    Rn, np.zeros((3, V)), dr["v"]), "6k zero reference mean refused")

# 7. labels ------------------------------------------------------------------------
ok(J.direction_label(0.1, 0.4) == "improvement supported" and
   J.practical_label(0.1, 0.4) ==
   "0.5 dB improvement ruled out at this interval level",
   "7a small improvement supported, 0.5 dB ruled out")
ok(J.direction_label(-0.2, 0.7) == "direction unresolved" and
   J.practical_label(-0.2, 0.7) == "0.5 dB improvement unresolved",
   "7b unresolved direction and practical size")
ok(J.practical_label(0.6, 0.9) == "0.5 dB improvement supported" and
   J.direction_label(-0.4, -0.1) == "deterioration supported",
   "7c practical supported; deterioration supported")

# R. refusal tests (review of core v0.1) --------------------------------------
Rr = np.random.default_rng(9).normal(size=(3, 4))
vr = J.make_draws(4, [], b=100)["v"]
An = Rr.copy(); An[1, 2] = np.nan
raises("DIAG_NON_FINITE", lambda: J.delta_estimate(An, Rr, vr),
       "R1a delta_estimate refuses NaN input")
raises("DIAG_NON_FINITE", lambda: J.spearman_estimate(An, Rr, vr),
       "R1b spearman_estimate refuses NaN input (no finite rho from NaN)")
raises("DIAG_NON_FINITE", lambda: J.ratio_estimate(Rr, np.where(
    np.isnan(An), np.inf, An), vr), "R1c ratio_estimate refuses inf input")
raises("DIAG_SHAPE", lambda: J.subgroup_estimates(
    Rr, Rr[:1], [[0], [1], [2, 3]], J.make_draws(4, [[1, 1, 2]], b=50)
    ["strata"][0]), "R2a (3,4) vs (1,4): no silent broadcasting")
raises("DIAG_SHAPE", lambda: J.delta_estimate(Rr[:2], Rr[:2], vr),
       "R2b seed dimension must be 3")
raises("DIAG_SHAPE", lambda: J.delta_estimate(Rr, Rr[:, :3], vr),
       "R2c unequal volume dimension refused")
raises("DIAG_DRAWS", lambda: J.delta_estimate(Rr, Rr, vr + 1),
       "R3a draw index out of range refused")
raises("DIAG_DRAWS", lambda: J.delta_estimate(Rr, Rr, vr.astype(np.int32)),
       "R3b draw dtype must be int64")
raises("DIAG_DRAWS", lambda: J.delta_estimate(Rr, Rr, vr[:, :3]),
       "R3c draw width must equal V")
raises("DIAG_POPULATION_EMPTY", lambda: J.check_replay({}, {}, 0),
       "R4a empty replay population refused")
raises("DIAG_POPULATION_MISMATCH", lambda: J.check_replay({}, {}, 2),
       "R4b replay count must equal the expected population")
raises("DIAG_NLL_FIELD", lambda: J.check_nll_identity(
    dict(good, nll_decomp_abs_dev=-1e-9), d),
    "R5a negative stored deviation refused")
for bad_d, lab in ((0, "0"), (True, "bool"), (13824.0, "float")):
    raises("DIAG_DIM_INVALID", lambda bd=bad_d: J.check_nll_identity(
        good, bd), f"R5b dimension {lab} refused")
st3 = J.make_draws(4, [[1, 1, 2]], b=50)["strata"][0]
for grp, lab in (([[0], [1, 2, 3]], "two groups"),
                 ([[0], [], [1, 2, 3]], "empty group"),
                 ([[0, 1], [1], [2, 3]], "overlapping groups"),
                 ([[0], [1], [2]], "groups not covering all volumes")):
    raises("DIAG_STRATA", lambda g=grp: J.subgroup_estimates(
        Rr, Rr, g, st3 if len(g) == 3 else st3[:2]), f"R6 {lab} refused")
raises("DIAG_DRAWS", lambda: J.subgroup_estimates(
    Rr, Rr, [[0], [1], [2, 3]], [st3[0], st3[1], st3[1]]),
    "R6 per-third draw width must match group size")
raises("DIAG_STRATA", lambda: J.subgroup_estimates(
    Rr, Rr, [[0], [1], [2, 3]], [st3[0], st3[1][:10], st3[2]]),
    "R6 per-third draw counts must agree")
raises("DIAG_INTERVAL", lambda: J.direction_label(0.5, 0.1),
       "R7a interval with lo > hi refused")
raises("DIAG_NON_FINITE", lambda: J.practical_label(float("nan"), 1.0),
       "R7b non-finite interval refused")
raises("DIAG_NON_FINITE", lambda: J.contributions(
    {"estimate": float("nan"), "ci": (-1.0, -0.5)}, -0.1, 0.0),
    "R8 contributions refuse non-finite input")
raises("DIAG_INTERVAL", lambda: J.contributions(
    {"estimate": 0.01, "ci": (0.02, 0.001)}, 0.005, 0.005),
    "R9 contributions refuse a reversed interval (lo > hi)")

n_ok = sum(RES)
print(f"RESULT {n_ok}/{len(RES)} synthetic checks passed")
sys.exit(0 if n_ok == len(RES) else 1)
