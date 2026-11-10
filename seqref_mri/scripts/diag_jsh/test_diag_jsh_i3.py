# SEQREF-MRI-DIAGJSHI3TEST v0.3 -- synthetic-table tests for DIAG-JSH I3 (analysis, table, figures, outputs)
# LIFETIME: KEEP
# =============================================================================
# SYNTHETIC TABLES ONLY: per-slice values with planted effects (known
# answers), no model, no data. Checks the analysis against the planted
# truth and the core, the per-volume table (regenerates the estimates),
# the figures, the budgeted write-once outputs, and measures plot time and
# output bytes for the F2 main-run limits.
# Usage: python -B test_diag_jsh_i3.py
# Changelog (v0.3, 2026-11-09): 5f titles = complex error magnitude labels.
# Changelog (v0.2, 2026-11-09): 5c-5g -- estimate outside its interval and
#   missing intervals render (Figures 5a, 5b, 6a); interval endpoints drawn
#   exactly, unclipped; Figure S2 layout (reference, reconstruction, error
#   per arm) and fixed shared scales.
# Changelog (v0.1, 2026-11-09): first I3 test set.
# =============================================================================
from __future__ import annotations

import csv
import io
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(CODE)]
logging.basicConfig(level=logging.CRITICAL)

import numpy as np                                            # noqa: E402

import diag_jsh_core as K                                     # noqa: E402
import diag_jsh_i3 as I3                                      # noqa: E402

N_OK = N_FAIL = 0


def ok(cond, label, detail=""):
    global N_OK, N_FAIL
    N_OK, N_FAIL = N_OK + bool(cond), N_FAIL + (not cond)
    print(("ok   " if cond else "FAIL ") + label + (f"  [{detail}]" if detail
                                                  else ""))


def raises(code, fn, label):
    try:
        fn()
    except K.DiagError as e:
        ok(e.code == code, label, f"got {e.code}")
        return
    except Exception as e:  # noqa: BLE001 -- reported as a failed check
        ok(False, label, f"untyped {type(e).__name__}: {e}")
        return
    ok(False, label, "no error")


# -- synthetic population with planted effects ---------------------------------
rng = np.random.default_rng(11)
V = 12
vols = [f"v{i:02d}" for i in range(V)]
nsl = {v: 1 + i % 3 for i, v in enumerate(vols)}
entries = [(v, f"f/{v}.h5", s) for v in vols for s in range(nsl[v])]
zf_psnr = {v: 20.0 + 0.5 * i for i, v in enumerate(vols)}   # v00 hardest
hf_val = {v: float(rng.uniform(0.01, 0.05)) for v in vols}
strat = {"H-D": {"values": zf_psnr,
                 "thirds": K.thirds(zf_psnr, descending=False)},
         "H-F": {"values": hf_val, "thirds": K.thirds(hf_val,
                                                     descending=True)}}
hard = set(strat["H-D"]["thirds"][0])
G_PSNR, G_HARD = 0.3, 0.6        # planted dPSNR (all) + extra (hardest third)
D_J, D_B = -0.012, 0.002         # planted likelihood components


def slice_vals(seed, arm, v, s):
    r = np.random.default_rng(hash((seed, v, s)) % 2 ** 32)
    base = {"psnr": 30 + r.normal(0, 1), "B": 0.5 + r.normal(0, .01),
            "J": -1.0 + r.normal(0, .01), "ref_energy": 100 + r.uniform()}
    for c in I3.CELLS:
        base[f"sc_{c}"] = 1.0 + r.uniform()
        base[f"sm_{c}"] = 0.5 + r.uniform()
    if arm == "lam2":
        noise = np.random.default_rng(hash((seed, v, s, 2)) % 2 ** 32)
        base["psnr"] += G_PSNR + (G_HARD if v in hard else 0.0) + \
            noise.normal(0, 0.05)
        base["J"] += D_J + noise.normal(0, 0.001)
        base["B"] += D_B
        base["sc_high_unacquired"] -= 0.2
    base["nll"] = base["B"] + base["J"]
    for c in I3.CELLS:
        base[f"scc_{c}"] = base[f"sc_{c}"] / base["ref_energy"]
    return {k: float(x) for k, x in base.items()}


per = {(sd, a): {(v, f, s): slice_vals(sd, a, v, s) for v, f, s in entries}
       for sd in K.SEEDS for a in I3.ARMS}

# 1. slice records -----------------------------------------------------------------
spec = K.spectral_record(np.ones((96, 96)) * (1 + 0.5j),
                         np.ones((96, 96)) * (1 + 0.4j),
                         np.r_[np.ones(48), np.zeros(48)].astype(int))
row = {"psnr": 31.0, "nll_per_dim": -0.5, "nll_prior_per_dim": 0.5,
       "nll_logdet_per_dim": -1.0}
sv = I3.slice_values(row, spec)
ok(set(sv) == set(I3.ALL) and sv["sc_low_acquired"] ==
   float(spec["sc_cells"][0, 0]), "1a slice record has every registered field")
raises("DIAG_NON_FINITE", lambda: I3.slice_values({**row, "psnr": float("nan")},
                                                   spec), "1b non-finite refused")

# 2. volume matrices -------------------------------------------------------------------
vlist, M = I3.volume_matrices(per, entries)
vm = K.volume_means({(f, s): per[(2027, "lam2")][(v, f, s)]["psnr"]
                     for v, f, s in entries}, entries)
ok(vlist == vols and np.array_equal(M["psnr"]["lam2"][1],
                                    [vm[v] for v in vols]),
   "2a volume matrices = core volume_means (registered weighting)")
bad = {**per, (2026, "lam0"): dict(reversed(list(per[(2026, "lam0")].items())))}
raises("DIAG_POPULATION_MISMATCH", lambda: I3.volume_matrices(bad, entries),
       "2b case-order mismatch refused")
raises("DIAG_POPULATION_MISMATCH", lambda: I3.volume_matrices(
    {k: v for k, v in per.items() if k != (2028, "lam2")}, entries),
    "2c missing run refused")

# 3. analysis vs planted truth ------------------------------------------------------------
t0 = time.monotonic()
res = I3.analyse(vlist, M, strat)
t_an = time.monotonic() - t0
D = M["psnr"]["lam2"] - M["psnr"]["lam0"]
ok(abs(res["reconstruction"]["estimate"] - D.mean()) < 1e-12 and
   res["reconstruction"]["direction"] == "improvement supported",
   "3a dPSNR = seed-mean of volume-mean differences; improvement supported",
   f"{res['reconstruction']['estimate']:.3f} dB")
ok(res["reconstruction"]["practical_0p5dB"].startswith("0.5 dB improvement")
   and "note" in res["reconstruction"], "3b practical label and note present",
   res["reconstruction"]["practical_0p5dB"])
j = res["D-JAC"]
ok(abs(j["dNLL"]["estimate"] - (j["dB"]["estimate"] + j["dJ"]["estimate"]))
   < 1e-12 and abs(j["dJ"]["estimate"] - D_J) < 1e-3,
   "3c dNLL = dB + dJ; dJ = planted value")
c = j["contributions"]
ok(c["reported"] and abs(c["J_share"] - D_J / (D_J + D_B)) < 0.1 and
   c["J_share"] > 1.0, "3d Jacobian share > 100% when base offsets",
   f"J share {c['J_share']:.3f}")
ok(j["nll_ratio"]["density_label"] in ("preserved", "worse", "unclear"),
   "3e density label from the NLL ratio interval",
   j["nll_ratio"]["density_label"])
ok(all(isinstance(j[k]["undefined_draws"], int) and
       (j[k]["ci95"] is None or len(j[k]["ci95"]) == 2)
       for k in ("spearman_dJ_dPSNR", "spearman_dNLL_dPSNR")),
   "3f Spearman results with undefined-draw counts")
hd = res["D-HET"]["H-D"]["metrics"]["psnr"]
ok(res["D-HET"]["H-D"]["n_per_third"] == [4, 4, 4] and
   hd["first_minus_last"]["heterogeneity_supported"] and
   abs(hd["first_minus_last"]["estimate"] - G_HARD) < 0.1,
   "3g planted hardest-third gain found only via the direct contrast",
   f"{hd['first_minus_last']['estimate']:.3f}")
s = res["D-SPEC"]["S-C"]["sc_high_unacquired"]
ok(abs(s["estimate"] + 0.2) < 1e-9 and s["ci95"][1] < 0 and
   res["D-SPEC"]["S-M_label"].startswith("magnitude-spectrum error"),
   "3h planted high/unacquired S-C reduction; S-M fallback label")
ok(I3.analyse(vlist, M, strat) == res, "3i analysis deterministic (frozen draws)")
Mc = {**M, "J": {"lam0": M["J"]["lam0"], "lam2": M["J"]["lam0"] + 0.0}}
raises("DIAG_CORRELATION_UNDEFINED", lambda: I3.analyse(vlist, Mc, strat),
       "3j constant dJ: correlation undefined -> typed stop")

# 4. per-volume table ------------------------------------------------------------------------
ident = {(sd, a): {"epoch": 10 + i, "model_state_sha256": f"{i:064x}"}
         for i, (sd, a) in enumerate((sd, a) for sd in K.SEEDS
                                     for a in I3.ARMS)}
tab = I3.volume_table(vlist, M, ident, nsl, strat)
rows = list(csv.DictReader(io.StringIO(tab.decode())))
ok(len(rows) == 3 * V * 3 and set(r["arm"] for r in rows) ==
   {"lam0", "lam2", "lam2-lam0"}, "4a table: seed x volume x {λ0, λ2, Δ}")
dd = np.array([[float(r["psnr"]) for r in rows if r["arm"] == "lam2-lam0"
                and int(r["seed"]) == sd] for sd in K.SEEDS])
ok(abs(dd.mean() - res["reconstruction"]["estimate"]) < 1e-12,
   "4b dPSNR regenerated from the table alone")
ok(all(r["HD_third"] in "123" and r["HF_third"] in "123" for r in rows) and
   all(r["selected_epoch"] and len(r["model_state_sha256"]) == 64
       for r in rows if r["arm"] != "lam2-lam0"),
   "4c identity, stratifier values and membership in every row")

# 5. figures -----------------------------------------------------------------------------------
ret = [{"volume_id": vols[i], "slice_index": 0,
        "x_true": rng.standard_normal((96, 96)) + 0j,
        "lam0": rng.standard_normal((96, 96)) + 0j,
        "lam2": rng.standard_normal((96, 96)) + 0j} for i in range(4)]
t0 = time.monotonic()
figs = I3.figures(res, M, vlist, strat, ret)
t_plot = time.monotonic() - t0
ok(sorted(figs) == ["fig5a_likelihood_components.png",
                    "fig5b_spectral_cells.png", "fig6a_heterogeneity.png",
                    "figS1_associations.png", "figS2_registered_slices.png"]
   and all(b[:8] == b"\x89PNG\r\n\x1a\n" for b in figs.values()),
   "5a five registered figures rendered (PNG)", f"plot {t_plot:.2f} s")
ok(I3.figures(res, M, vlist, strat, ret).keys() == figs.keys(),
   "5b figures regenerate from results + table data (no decoding)")

import copy                                                   # noqa: E402
import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

bad = copy.deepcopy(res)                     # estimate outside its interval
for k in ("dNLL", "dB", "dJ"):
    e = bad["D-JAC"][k]["estimate"]
    bad["D-JAC"][k]["ci95"] = [e + 1.0, e + 2.0]
for m in I3.SC:
    e = bad["D-SPEC"]["S-C"][m]["estimate"]
    bad["D-SPEC"]["S-C"][m]["ci95"] = [e - 3.0, e - 2.0]
for st in ("H-D", "H-F"):
    h = bad["D-HET"][st]["metrics"]["psnr"]
    for th in h["thirds"]:
        th["ci95"] = [th["estimate"] + 0.5, th["estimate"] + 0.7]
    c = h["first_minus_last"]
    c["ci95"] = [c["estimate"] - 0.9, c["estimate"] - 0.4]
try:
    f2 = I3.figures(bad, M, vlist, strat, ret)
    err = ""
except Exception as exc:                     # reported, not swallowed
    f2, err = {}, f"{type(exc).__name__}: {exc}"
ok(len(f2) == 5, "5c estimate outside its percentile interval renders "
   "(5a, 5b, 6a; no negative-offset error)", err)
none = copy.deepcopy(res)
none["D-JAC"]["dJ"]["ci95"] = None
none["D-HET"]["H-D"]["metrics"]["psnr"]["thirds"][0]["ci95"] = None
try:
    f3 = I3.figures(none, M, vlist, strat, ret)
    err = ""
except Exception as exc:
    f3, err = {}, f"{type(exc).__name__}: {exc}"
ok(len(f3) == 5, "5d missing interval drawn as no line (no crash)", err)

fig, ax = plt.subplots()
drawn = I3._interval(ax, 1.0, [2.5, 7.25], vertical=False)
seg = ax.collections[0].get_segments()[0]
fig2, ax2 = plt.subplots()
vdrawn = I3._interval(ax2, 0.0, [-40.0, 1e6], vertical=True)
vseg = ax2.collections[0].get_segments()[0]
ok(drawn and seg[0].tolist() == [2.5, 1.0] and seg[1].tolist() == [7.25, 1.0]
   and vdrawn and vseg[0].tolist() == [0.0, -40.0] and
   vseg[1].tolist() == [0.0, 1e6] and
   not I3._interval(ax, 0, [float("nan"), 1.0], vertical=True),
   "5e endpoints drawn exactly from the interval, unclipped; non-finite "
   "endpoints not drawn")
plt.close(fig)
plt.close(fig2)

cap = {}
real_sub = plt.subplots


def spy(*a, **k):
    f, axs = real_sub(*a, **k)
    if a[:2] == (len(ret), 5):
        cap["axes"] = axs
        cap["fig"] = f
    return f, axs


plt.subplots = spy
real_close = plt.close
plt.close = lambda *a, **k: None             # keep S2 open for inspection
try:
    I3.figures(res, M, vlist, strat, ret)
finally:
    plt.subplots = real_sub
    plt.close = real_close
axs = cap.get("axes")
if axs is not None:
    titles = [a.get_title() for a in axs[0]]
    clims = [[a.images[0].get_clim() for a in row] for row in axs]
    vmax = max(float(np.abs(r[k]).max()) for r in ret
               for k in ("x_true", "lam0", "lam2"))
    emax = max(float(np.abs(r[a] - r["x_true"]).max())
               for r in ret for a in ("lam0", "lam2"))
    data_ok = all(np.array_equal(axs[i][2].images[0].get_array(),
                                 np.abs(ret[i]["lam0"] - ret[i]["x_true"])) and
                  np.array_equal(axs[i][4].images[0].get_array(),
                                 np.abs(ret[i]["lam2"] - ret[i]["x_true"]))
                  for i in range(len(ret)))
    real_close(cap["fig"])
else:
    titles, clims, data_ok, vmax, emax = [], [], False, 0, 0
ok(titles == ["reference |x|", "R0 λ₀ |x̂|", "λ₀ complex error |x̂ − x|",
              "R0 λ₂ |x̂|", "λ₂ complex error |x̂ − x|"] and
   data_ok, "5f Figure S2: reference, reconstruction and |reconstruction − "
   "reference| per arm (no λ₂ − λ₀ difference panel)", str(titles))
ok(clims and all(row[j] == (0, vmax) for row in clims for j in (0, 1, 3)) and
   all(row[j] == (0, emax) for row in clims for j in (2, 4)),
   "5g Figure S2 fixed scales shared across slices and arms (magnitude, "
   "error)")

# 6. budgeted outputs (guarded subprocess) ------------------------------------------------------
tmp = Path(tempfile.mkdtemp(prefix="diagi3_"))
payload = tmp / "payload.json"
payload.write_text(json.dumps({"res": res, "tab": tab.decode()}))


def run_writer(budget):
    root = Path(tempfile.mkdtemp(dir=tmp, prefix="o_"))
    body = (f"import sys, json; sys.path[:0]=[{str(CODE)!r}]\n"
            "from pathlib import Path\n"
            "import diag_jsh_adapter as A, diag_jsh_core as K, diag_jsh_i3 as I3\n"
            f"R = Path({str(root)!r}); A.install_write_guard(R)\n"
            f"B = A.WriteBudget(R, 0, {budget})\n"
            f"d = json.loads(Path({str(payload)!r}).read_text())\n"
            "import numpy as np\n"
            "figs = {'fig5a_likelihood_components.png': b'\\x89PNG' + b'0' * 100}\n"
            "try:\n"
            " s = I3.write_outputs(B, R, d['res'], d['tab'].encode(), figs, {'x': 1})\n"
            " print('USED', B.used, len(s))\n"
            "except K.DiagError as e:\n print('CODE', e.code)\n")
    p = subprocess.run([sys.executable, "-B", "-c", body], capture_output=True,
                       text=True, timeout=300)
    return p, root


p, root = run_writer(16 * 2 ** 20)
names = sorted(x.name for x in root.iterdir())
ok(p.returncode == 0 and "USED" in p.stdout and len(names) == 8 and
   all((root / f"{n}.sha256").is_file() for n in
       ("results.json", "per_volume.csv", "limitations.md")),
   "6a outputs written once with sidecars through the enforced budget",
   p.stdout.strip() + p.stderr.strip()[-120:])
p, root = run_writer(2000)
ok("CODE DIAG_DISK" in p.stdout and not (root / "results.json").exists(),
   "6b output larger than the budget refused before writing")
fig_bytes = sum(len(b) for b in figs.values())
out_bytes = (fig_bytes + len(tab) + len(json.dumps(res)) +
             len(I3.LIMITATIONS))
ok(True, "6c measured for F2 (synthetic V=12; scales with volumes)",
   f"analysis {t_an:.2f} s, plots {t_plot:.2f} s, outputs ≈ {out_bytes} B")
import shutil                                                 # noqa: E402
shutil.rmtree(tmp)
print(f"RESULT {N_OK}/{N_OK + N_FAIL} checks passed")
sys.exit(0 if N_FAIL == 0 else 1)
