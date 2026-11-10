# SEQREF-MRI-DIAGJSHI3 v0.3 -- DIAG-JSH main-run analysis and outputs (frozen mini-spec v1.0 §2-§6; core only)
# LIFETIME: KEEP
# =============================================================================
# Turns per-slice values of the six selected checkpoints into the registered
# outputs, using only the scientific core (diag_jsh_core v0.2.1):
#   * per-volume machine-readable table (identity, PSNR/NLL/B/J per arm and
#     paired lambda2 - lambda0 differences, spectral cells, stratifiers and
#     subgroup membership) -- plots can be regenerated without decoding;
#   * D-JAC (§3), reconstruction context (§2), D-SPEC (§4), D-HET (§5)
#     result tables (JSON) with the registered labels;
#   * Figure 5a (likelihood components), 5b (S-C cells), 6a (thirds and
#     extreme-third contrasts with counts), S1 (association plots), S2 (the
#     four registered slices: reference, reconstruction and complex
#     reconstruction error magnitude |x_hat - x_ref| per arm, on fixed
#     scales); limitations statement.
# Intervals are drawn from their own endpoints, independently of the point
#   estimate marker (a percentile interval need not contain the estimate);
#   endpoints are never clipped. A missing interval is drawn as no line.
# Every output is written through the enforced WriteBudget (write-once,
# sha256 sidecar). Intervals: exploratory, unadjusted, conditional on the
# three trained seeds. S-M is labelled "magnitude-spectrum error" (frozen
# fallback; parity criterion not adopted).
# NOT YET INCLUDED (flagged): the real-data main-run binding (manifest
#   re-check, decode via the adapter, reproduction re-check) -> DIAG-I3 v0.2
#   with its integration test; this version is tested on synthetic tables.
# Changelog (v0.3, 2026-11-09; review of v0.2): Figure S2 error panels
#   labelled 'complex error magnitude |x̂ − x|' (abs of the complex
#   difference, not a difference of magnitudes).
# Changelog (v0.2, 2026-11-09; review of v0.1): interval endpoints drawn
#   independently of the estimate (fixes 'xerr must not contain negative
#   values' when the estimate lies outside its percentile interval), in
#   Figures 5a, 5b and 6a, without clipping; Figure S2 now shows reference,
#   reconstruction and |reconstruction - reference| per arm on fixed scales
#   (was a lambda2 - lambda0 difference image).
# Changelog (v0.1, 2026-11-09): first core analysis/output module.
# =============================================================================
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math

import numpy as np

import diag_jsh_core as K

__abbr__ = "SEQREF-MRI-DIAGJSHI3"
__version__ = "0.3"
logger = logging.getLogger("seqref_diag.jsh_i3")

ARMS = ("lam0", "lam2")                       # Δ = lam2 - lam0 (§1.1)
SCALARS = ("psnr", "nll", "B", "J")
CELLS = [f"{b}_{s}" for b in K.BANDS for s in K.SPLITS]      # 6 cells
SC = [f"sc_{c}" for c in CELLS]                # complex error per cell
SCC = [f"scc_{c}" for c in CELLS]              # contribution per cell
SM = [f"sm_{c}" for c in CELLS]                # magnitude-spectrum error
HET_METRICS = list(SCALARS) + SC               # §5 per-third metrics
ALL = list(SCALARS) + SC + SCC + SM + ["ref_energy"]


# -- per-slice record from adapter outputs -------------------------------------
def slice_values(row: dict, spec: dict) -> dict:
    """Adapter replay row + core spectral_record -> flat per-slice values."""
    out = {"psnr": row["psnr"], "nll": row["nll_per_dim"],
           "B": row["nll_prior_per_dim"], "J": row["nll_logdet_per_dim"],
           "ref_energy": spec["ref_energy"]}
    for i, b in enumerate(K.BANDS):
        for j, s in enumerate(K.SPLITS):
            c = f"{b}_{s}"
            out[f"sc_{c}"] = float(spec["sc_cells"][i, j])
            out[f"scc_{c}"] = float(spec["sc_contrib"][i, j])
            out[f"sm_{c}"] = float(spec["sm_cells"][i, j])
    for k, v in out.items():
        K.require(isinstance(v, float) and math.isfinite(v),
                  "DIAG_NON_FINITE", f"{k}={v!r}")
    return out


# -- volume matrices -------------------------------------------------------------
def volume_matrices(per_slice: dict, entries: list) -> tuple:
    """per_slice: {(seed, arm): {(vid, f, s): values}} -> (vols, M) with
    M[metric][arm] a (3, V) array of volume means (registered weighting)."""
    K.require(set(per_slice) == {(s, a) for s in K.SEEDS for a in ARMS},
              "DIAG_POPULATION_MISMATCH", f"runs {sorted(per_slice)}")
    vols = sorted({v for v, _, _ in entries})
    M = {m: {a: np.zeros((len(K.SEEDS), len(vols))) for a in ARMS}
         for m in ALL}
    for (seed, arm), rows in per_slice.items():
        K.require(list(rows) == list(entries), "DIAG_POPULATION_MISMATCH",
                  f"{seed}/{arm}: case order")
        k = K.SEEDS.index(seed)
        for m in ALL:
            vm = K.volume_means({(f, s): rows[(v, f, s)][m]
                                 for v, f, s in entries}, entries)
            M[m][arm][k] = [vm[v] for v in vols]
    return vols, M


def _est(d: dict) -> dict:
    return {"estimate": d["estimate"], "ci95": list(d["ci"]),
            **({"per_seed": d["per_seed"]} if "per_seed" in d else {})}


# -- analysis (§2-§5) ------------------------------------------------------------------
def analyse(vols: list, M: dict, strat: dict) -> dict:
    """strat: {"H-D": {"thirds": [[vid...]x3]}, "H-F": {...}} (manifest)."""
    V = len(vols)
    pos = {v: i for i, v in enumerate(vols)}
    sizes = [[len(g) for g in strat[s]["thirds"]] for s in ("H-D", "H-F")]
    dr = K.make_draws(V, sizes)
    v = dr["v"]

    def d(m):
        return K.delta_estimate(M[m]["lam2"], M[m]["lam0"], v)
    dpsnr = d("psnr")
    lo, hi = dpsnr["ci"]
    recon = {**_est(dpsnr), "direction": K.direction_label(lo, hi),
             "practical_0p5dB": K.practical_label(lo, hi),
             "note": "a non-significant dPSNR never supports 'reconstruction "
                     "does not improve'"}
    dn, dB, dJ = d("nll"), d("B"), d("J")
    ratio = K.ratio_estimate(M["nll"]["lam2"], M["nll"]["lam0"], v)
    dpd = {m: M[m]["lam2"] - M[m]["lam0"] for m in ("psnr", "nll", "J")}
    jac = {"dNLL": _est(dn), "dB": _est(dB), "dJ": _est(dJ),
           "nll_ratio": {**_est(ratio),
                         "density_label": K.density_label(*ratio["ci"])},
           "contributions": K.contributions(dn, dJ["estimate"],
                                            dB["estimate"]),
           "spearman_dJ_dPSNR": K.spearman_estimate(dpd["J"], dpd["psnr"], v),
           "spearman_dNLL_dPSNR": K.spearman_estimate(dpd["nll"], dpd["psnr"],
                                                      v),
           "units": "nats per free dimension; negative = better likelihood"}
    for k in ("spearman_dJ_dPSNR", "spearman_dNLL_dPSNR"):
        s = jac[k]
        s["ci95"] = list(s.pop("ci")) if s["ci"] is not None else None
    spec = {"S-C": {m: _est(d(m)) for m in SC},
            "S-C_contribution": {m: _est(d(m)) for m in SCC},
            "S-M": {m: _est(d(m)) for m in SM},
            "S-M_label": "magnitude-spectrum error (not a PSNR-MSE "
                         "decomposition; parity criterion not adopted)"}
    het = {}
    for i, s in enumerate(("H-D", "H-F")):
        groups = [[pos[x] for x in g] for g in strat[s]["thirds"]]
        het[s] = {"n_per_third": [len(g) for g in groups],
                  "metrics": {}}
        for m in HET_METRICS:
            r = K.subgroup_estimates(M[m]["lam2"], M[m]["lam0"], groups,
                                     dr["strata"][i])
            het[s]["metrics"][m] = {
                "thirds": [{"n": t["n"], "estimate": t["estimate"],
                            "ci95": list(t["ci"])} for t in r["thirds"]],
                "first_minus_last": {
                    "estimate": r["contrast_first_minus_last"]["estimate"],
                    "ci95": list(r["contrast_first_minus_last"]["ci"]),
                    "heterogeneity_supported": bool(
                        r["contrast_first_minus_last"]["ci"][0] > 0 or
                        r["contrast_first_minus_last"]["ci"][1] < 0)}}
    return {"schema": "SEQREF-MRI-DIAG-JSH-RESULTS/0.1",
            "contrast": "lambda2 - lambda0 (objective comparison)",
            "intervals": "95% percentile, paired volume bootstrap B=10000, "
                         "DIAG_SEED 20261109; exploratory, unadjusted, "
                         "conditional on the three trained seeds",
            "volumes": V, "reconstruction": recon, "D-JAC": jac,
            "D-SPEC": spec, "D-HET": het}


# -- per-volume table ---------------------------------------------------------------
def volume_table(vols: list, M: dict, ident: dict, nslices: dict,
                 strat: dict) -> bytes:
    """CSV: one row per seed x volume x {lam0, lam2, lam2-lam0}."""
    member = {s: {v: t for t, g in enumerate(strat[s]["thirds"]) for v in g}
              for s in ("H-D", "H-F")}
    cols = (["seed", "arm", "selected_epoch", "model_state_sha256",
             "volume_id", "n_slices"] + ALL +
            ["HD_value", "HD_third", "HF_value", "HF_third"])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for k, seed in enumerate(K.SEEDS):
        for j, vid in enumerate(vols):
            for arm in ARMS + ("lam2-lam0",):
                if arm == "lam2-lam0":
                    vals = [repr(float(M[m]["lam2"][k, j] - M[m]["lam0"][k, j]))
                            for m in ALL]
                    idn = ["", ""]
                else:
                    vals = [repr(float(M[m][arm][k, j])) for m in ALL]
                    ck = ident[(seed, arm)]
                    idn = [ck["epoch"], ck["model_state_sha256"]]
                w.writerow([seed, arm, *idn, vid, nslices[vid], *vals,
                            repr(strat["H-D"]["values"][vid]),
                            member["H-D"][vid] + 1,
                            repr(strat["H-F"]["values"][vid]),
                            member["H-F"][vid] + 1])
    return buf.getvalue().encode()


def reliability(repro: dict, max_decomp: float, max_spec_rel: float,
                jac: dict) -> dict:
    return {"replay_deviations": repro,
            "max_stored_nll_decomposition_residual": max_decomp,
            "max_spectral_identity_rel_residual": max_spec_rel,
            "spearman_undefined_draws": {
                k: jac[k]["undefined_draws"] for k in
                ("spearman_dJ_dPSNR", "spearman_dNLL_dPSNR")}}


# -- figures -------------------------------------------------------------------------
def _fig_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                metadata={"Software": None})
    import matplotlib.pyplot as plt
    plt.close(fig)
    return buf.getvalue()


def _interval(ax, pos, ci, *, vertical: bool, cap: float = 0.12):
    """Draw [lo, hi] at `pos` from its own endpoints (no estimate offsets,
    no clipping). None / non-finite endpoints -> nothing drawn."""
    if ci is None or len(ci) != 2 or not all(
            c is not None and math.isfinite(c) for c in ci):
        return False
    lo, hi = float(ci[0]), float(ci[1])
    if vertical:
        ax.vlines(pos, lo, hi, color="k", lw=1.2, zorder=3)
        ax.hlines([lo, hi], pos - cap, pos + cap, color="k", lw=1.2, zorder=3)
    else:
        ax.hlines(pos, lo, hi, color="k", lw=1.2, zorder=3)
        ax.vlines([lo, hi], pos - cap, pos + cap, color="k", lw=1.2, zorder=3)
    return True


def _bars(ax, labels, ests, cis, per_seed=None, ref=None):
    y = np.arange(len(labels))
    for i, ci in enumerate(cis):
        _interval(ax, y[i], ci, vertical=False)
    ax.scatter(ests, y, marker="o", color="k", s=22, zorder=4)
    if per_seed:
        for i, ps in enumerate(per_seed):
            ax.scatter(ps, [y[i]] * len(ps), marker="x", color="0.5", s=18,
                       zorder=3)
    ax.axvline(0.0, color="0.6", lw=0.8)
    if ref is not None:
        ax.axvline(ref, color="0.6", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)


def figures(res: dict, M: dict, vols: list, strat: dict,
            retained: list | None) -> dict:
    """Figure 5a, 5b, 6a, S1 and (if images given) S2 as PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = {}
    jac = res["D-JAC"]
    fig, ax = plt.subplots(figsize=(5, 2.4))
    ks = ("dNLL", "dB", "dJ")
    _bars(ax, ["ΔNLL", "Δbase (B)", "ΔJacobian (J)"],
          [jac[k]["estimate"] for k in ks], [jac[k]["ci95"] for k in ks],
          [jac[k]["per_seed"] for k in ks])
    ax.set_xlabel("λ₂ − λ₀ (nats per free dim; negative = better)")
    ax.set_title("Figure 5a — likelihood components (95% CI; × = seeds)")
    out["fig5a_likelihood_components.png"] = _fig_bytes(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    sc = res["D-SPEC"]["S-C"]
    _bars(ax, [c.replace("_", " / ") for c in CELLS],
          [sc[m]["estimate"] for m in SC], [sc[m]["ci95"] for m in SC],
          [sc[m]["per_seed"] for m in SC])
    ax.set_xlabel("Δ band error energy Σ|E|² (λ₂ − λ₀; negative = better)")
    ax.set_title("Figure 5b — complex Fourier error by band and location")
    out["fig5b_spectral_cells.png"] = _fig_bytes(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    for ax, s in zip(axes, ("H-D", "H-F")):
        h = res["D-HET"][s]["metrics"]["psnr"]
        names = (["hardest", "middle", "easiest"] if s == "H-D" else
                 ["highest HF", "middle", "lowest HF"])
        pos = {v: i for i, v in enumerate(vols)}
        for t, g in enumerate(strat[s]["thirds"]):
            idx = [pos[v] for v in g]
            dv = (M["psnr"]["lam2"] - M["psnr"]["lam0"])[:, idx].mean(0)
            ax.scatter([t] * len(idx), dv, color="0.6", s=10, zorder=2)
            th = h["thirds"][t]
            _interval(ax, t + 0.15, th["ci95"], vertical=True, cap=0.06)
            ax.scatter([t + 0.15], [th["estimate"]], marker="o", color="k",
                       s=22, zorder=4)
        c = h["first_minus_last"]
        _interval(ax, 3.2, c["ci95"], vertical=True, cap=0.06)
        ax.scatter([3.2], [c["estimate"]], marker="s", color="k", s=22,
                   zorder=4)
        ax.set_xticks([0, 1, 2, 3.2])
        ax.set_xticklabels([f"{n}\n(n={h['thirds'][i]['n']})"
                            for i, n in enumerate(names)] +
                           ["first − last"], fontsize=8)
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_title({"H-D": "ZF difficulty", "H-F": "HF content"}[s])
    axes[0].set_ylabel("ΔPSNR (dB, λ₂ − λ₀)")
    fig.suptitle("Figure 6a — thirds (exploratory; dots = seed-mean volumes)",
                 fontsize=10)
    out["fig6a_heterogeneity.png"] = _fig_bytes(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    dps = (M["psnr"]["lam2"] - M["psnr"]["lam0"]).mean(0)
    for ax, m, k in zip(axes, ("J", "nll"),
                        ("spearman_dJ_dPSNR", "spearman_dNLL_dPSNR")):
        dx = (M[m]["lam2"] - M[m]["lam0"]).mean(0)
        ax.scatter(dx, dps, s=10, color="k")
        r = jac[k]
        ci = r["ci95"]
        ax.set_title(f"ρ = {r['estimate']:.3f}" +
                     (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else
                      " (CI not computed)"), fontsize=9)
        ax.set_xlabel(f"Δ{'J' if m == 'J' else 'NLL'} (seed-mean, volume)")
    axes[0].set_ylabel("ΔPSNR (dB)")
    fig.suptitle("Figure S1 — associations (descriptive, not causal)",
                 fontsize=10)
    out["figS1_associations.png"] = _fig_bytes(fig)

    if retained:
        n = len(retained)
        fig, axes = plt.subplots(n, 5, figsize=(10, 2 * n), squeeze=False)
        # One magnitude scale for reference and both reconstructions, one
        # error scale for both arms, shared across all slices (fixed).
        vmax = max(float(np.abs(r[k]).max()) for r in retained
                   for k in ("x_true",) + ARMS)
        emax = max(float(np.abs(r[a] - r["x_true"]).max())
                   for r in retained for a in ARMS)
        titles = ("reference |x|", "R0 λ₀ |x̂|", "λ₀ complex error |x̂ − x|",
                  "R0 λ₂ |x̂|", "λ₂ complex error |x̂ − x|")
        for i, r in enumerate(retained):
            ims = [np.abs(r["x_true"]),
                   np.abs(r["lam0"]), np.abs(r["lam0"] - r["x_true"]),
                   np.abs(r["lam2"]), np.abs(r["lam2"] - r["x_true"])]
            for j, (im, t) in enumerate(zip(ims, titles)):
                kw = ({"cmap": "gray", "vmin": 0, "vmax": vmax}
                      if j in (0, 1, 3) else
                      {"cmap": "magma", "vmin": 0, "vmax": emax})
                axes[i, j].imshow(im, **kw)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                if i == 0:
                    axes[i, j].set_title(t, fontsize=8)
            axes[i, 0].set_ylabel(f"{r['volume_id']}|{r['slice_index']}",
                                  fontsize=7)
        fig.suptitle("Figure S2 — four registered slices (seed 2026): "
                     f"magnitude 0–{vmax:.3g}; complex reconstruction error "
                     f"magnitude |x̂ − x| 0–{emax:.3g} (fixed scales)", fontsize=10)
        out["figS2_registered_slices.png"] = _fig_bytes(fig)
    return out


LIMITATIONS = """# DIAG-JSH limitations (registered statement)

- Objective comparison: λ₂ − λ₀ at the V05LT selected checkpoints; not a
  before/after sequential-refinement experiment.
- Intervals: 95% percentile, paired volume bootstrap; exploratory and
  unadjusted for multiplicity; conditional on the three trained seeds (seed
  variability is not resampled). The registered crossed V05LT intervals are
  reported separately and never pooled with these.
- Associations (Spearman) and subgroup results are descriptive and
  exploratory; heterogeneity is claimed only from the direct extreme-third
  contrast.
- D-JAC describes how the density changes; it does not establish why
  reconstruction does or does not change.
- S-M is a magnitude-spectrum error; it is not reported as a PSNR-MSE
  decomposition (parity criterion not adopted).
- R0 readout only; validation population only; holdout excluded.
"""


def write_outputs(budget, out, res: dict, table: bytes, figs: dict,
                  rel: dict) -> dict:
    """Write-once outputs + sha256 sidecars through the enforced budget."""
    files = {"results.json": (json.dumps({**res, "reliability": rel},
                                         sort_keys=True, indent=1,
                                         allow_nan=False) + "\n").encode(),
             "per_volume.csv": table,
             "limitations.md": LIMITATIONS.encode(), **figs}
    shas = {}
    for name, data in sorted(files.items()):
        sha = hashlib.sha256(data).hexdigest()
        side = f"{sha}  {name}\n".encode()
        budget.check(len(data) + len(side), f"before {name}")
        budget.write_new(out / name, data, name)
        budget.write_new(out / f"{name}.sha256", side, f"{name} sidecar")
        shas[name] = sha
    return shas
