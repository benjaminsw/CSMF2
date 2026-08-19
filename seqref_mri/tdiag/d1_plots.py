# SEQREF-TDIAG v0.1 -- tdiag.d1_plots
# LIFETIME: KEEP
# =============================================================================
# Purpose: D1 diagnostic figures (EXEC SS10.6 D1). FOUR figures, rendered
#          under the stage out-dir (results/_diag/diag/figures/):
#            1. estimator aggregate performance (mean PSNR / mean NMSE_u
#               for E0-E4, frozen-band threshold lines, per-slice points)
#            2. per-slice estimator deltas vs E0 (delta-PSNR and
#               NMSE-ratio heatmaps, rows = slices, cols = E1-E4)
#            3. E3/E4 optimization trajectories (small multiples by
#               slice; 8 starts light, winner highlighted)
#            4. JVP sensitivity (16 q_j per slice + sqrt(mean q), log y)
#          The figures are DESCRIPTIVE SUPPORT ONLY: they never become
#          registered evidence, never feed the decision fields and never
#          route. A rendering failure is logger.error + typed StageError
#          (D1_PLOT_FAILURE) -- no silent skip.
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass.
# Changelog (NEW in v0.1, D1 slice 2026-08-18):
#   * Introduced with the D1 slice under the 2026-08-15 EXEC SS10.6 lock.
# Update summary:
#   v0.1 D1 lands the four locked diagnostic figures for the estimator
#   slate, kept strictly outside the registered evidence path.
# =============================================================================
from __future__ import annotations

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seqref_mri.tdiag import _bootstrap  # noqa: F401

from preflight_parents import StageError  # noqa: E402

logger = logging.getLogger("SEQREF-TDIAG")

_EST = ("E0", "E1", "E2", "E3", "E4")
_DELTA_EST = ("E1", "E2", "E3", "E4")


def _fig_aggregate(d1: dict, path: str) -> None:
    agg, thr = d1["aggregate"], d1["thresholds"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(_EST))
    for ax, key, tline, tlabel in (
            (axes[0], "mean_psnr", thr["E0_plus_2db"],
             "E0 + 2.0 dB"),
            (axes[1], "mean_nmse_u", thr["E0_half_nmse_u"],
             "0.5 x E0 NMSE_u")):
        means = [agg[k][key] for k in _EST]
        ax.bar(x, means, color="0.75", edgecolor="k", width=0.6)
        for i, k in enumerate(_EST):
            pts = [r["psnr" if key == "mean_psnr" else "nmse_u"]
                   for r in d1["estimators"][k]["per_slice"]]
            ax.scatter(np.full(len(pts), x[i]), pts, s=14, c="C0",
                       zorder=3)
        ax.axhline(tline, color="C3", ls="--", lw=1.2, label=tlabel)
        ax.set_xticks(x, _EST)
        ax.set_title(key)
        ax.legend(fontsize=8)
    fig.suptitle("D1 estimator aggregate performance (bars = mean over "
                 "the same slices; points = per-slice)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_deltas(d1: dict, path: str) -> None:
    e0 = d1["estimators"]["E0"]["per_slice"]
    n = len(e0)
    dp = np.zeros((n, len(_DELTA_EST)))
    rn = np.zeros((n, len(_DELTA_EST)))
    for j, k in enumerate(_DELTA_EST):
        rows = d1["estimators"][k]["per_slice"]
        for i in range(n):
            dp[i, j] = rows[i]["psnr"] - e0[i]["psnr"]
            rn[i, j] = (rows[i]["nmse_u"] / e0[i]["nmse_u"]
                        if e0[i]["nmse_u"] > 0.0 else np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, mat, title, fmt in ((axes[0], dp, "delta PSNR vs E0 (dB)",
                                 "{:+.2f}"),
                                (axes[1], rn, "NMSE_u ratio vs E0",
                                 "{:.2f}")):
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r")
        ax.set_xticks(range(len(_DELTA_EST)), _DELTA_EST)
        ax.set_yticks(range(n), [f"s{i}" for i in range(n)])
        ax.set_title(title)
        for i in range(n):
            for j in range(len(_DELTA_EST)):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center",
                        va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("D1 per-slice estimator deltas vs E0")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_trajectories(d1: dict, path: str) -> None:
    e3 = d1["estimators"]["E3"]["per_slice"]
    e4 = d1["estimators"]["E4"]["per_slice"]
    n = len(e3)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.1 * n), squeeze=False)
    for i in range(n):
        for ax, rec, title in (
                (axes[i][0], e3[i], "E3 total log density"),
                (axes[i][1], e4[i], "E4 squared-u error")):
            for s in rec["starts"]:
                steps = sorted(int(k) for k in s["trajectory"])
                vals = [s["trajectory"][str(k)] for k in steps]
                kw = ({"lw": 2.0, "color": "C3", "zorder": 3}
                      if s["winner"] else
                      {"lw": 0.8, "color": "0.6", "alpha": 0.7})
                ax.plot(steps, vals, marker=".", ms=3, **kw)
            ax.set_title(f"slice {i}: {title} (winner highlighted)",
                         fontsize=9)
            ax.set_xlabel("step", fontsize=8)
    fig.suptitle("D1 E3/E4 optimization trajectories "
                 "(checkpoints 0..200)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_jvp(d1: dict, path: str) -> None:
    rows = d1["jvp"]["per_slice"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, r in enumerate(rows):
        ax.scatter(np.arange(len(r["q"])) + i * 0.06, r["q"], s=16,
                   label=f"slice {i}")
        ax.axhline(r["sqrt_mean_q"] ** 2, ls=":", lw=0.8,
                   color=f"C{i % 10}")
    ax.set_yscale("log")
    ax.set_xlabel("probe index")
    ax.set_ylabel("q_j = ||J(0) v_j||^2 (log)")
    ax.set_title("D1 JVP sensitivity at z=0 (dotted: mean q per slice)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_d1_figures(d1: dict, out_dir: str) -> list:
    """Render the four locked D1 figures under out_dir/figures/. Returns
    the written paths. Any rendering failure is a typed ERROR (the
    figures are descriptive, but a silent skip is never permitted)."""
    fig_dir = os.path.join(out_dir, "figures")
    try:
        os.makedirs(fig_dir, exist_ok=True)
        targets = [
            ("d1_estimator_aggregate.png", _fig_aggregate),
            ("d1_per_slice_deltas.png", _fig_deltas),
            ("d1_optimization_trajectories.png", _fig_trajectories),
            ("d1_jvp_sensitivity.png", _fig_jvp)]
        written = []
        for name, fn in targets:
            path = os.path.join(fig_dir, name)
            fn(d1, path)
            written.append(path)
    except StageError:
        raise
    except Exception as exc:  # noqa: BLE001 -- typed boundary
        logger.error("[SEQREF-TDIAG] D1_PLOT_FAILURE: %s: %s",
                     type(exc).__name__, exc)
        raise StageError(
            "D1_PLOT_FAILURE",
            f"D1 figure rendering failed: {type(exc).__name__}: {exc}",
            detail={"exception_type": type(exc).__name__,
                    "exception_message": str(exc)})
    logger.info("[SEQREF-TDIAG] D1 figures written: %s",
                [os.path.basename(p) for p in written])
    return written
