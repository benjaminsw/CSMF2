# SEQREF-TDIAG v0.1 -- tdiag.d3_plots
# LIFETIME: KEEP
# =============================================================================
# Purpose: D3 diagnostic figures (EXEC SS10.6 D3). THREE figures under
#          the stage out-dir (results/_diag/diag/figures/):
#            1. d3_sensitivity_scores.png -- grouped S_NLL/S_PSNR bars
#               for C1-C3 with the locked 0.01/0.25 band lines and the
#               C1 classification annotated (the campaign-level visual:
#               it directly mirrors the D3 decision rule).
#            2. d3_per_slice_psnr_deltas.png -- 8 x 3 SIGNED
#               delta_PSNR_z0 (Ck - C0) diverging heatmap centred at 0
#               (slice-level heterogeneity behind the aggregate score).
#            3. d3_z0_vs_pm_psnr.png -- per-condition SIGNED mean
#               delta_PSNR for z=0 vs the posterior mean side by side
#               (does the sensitivity survive posterior averaging?).
#          The figures are DESCRIPTIVE SUPPORT ONLY: they never become
#          evidence, never feed a band and never route. A rendering
#          failure is logger.error + typed StageError D3_PLOT_FAILURE
#          (all-or-nothing: no facts artefact accompanies a plot
#          failure).
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * D3 slice (2026-08-20, under the same SS10.6 lock; NO contract
#     change): module introduced with the three D3 figures.
# Update summary:
#   v0.1 D3 lands the grouped sensitivity-score bars with the locked
#   band lines, the per-slice signed delta heatmap and the z=0 vs
#   posterior-mean comparison panel.
# =============================================================================
from __future__ import annotations

import logging
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from seqref_mri.tdiag import _bootstrap  # noqa: F401,E402

from preflight_parents import StageError  # noqa: E402

logger = logging.getLogger("SEQREF-TDIAG")

_CONDITION_KEYS = ("C1", "C2", "C3")


def _fig_sensitivity(d3: dict, path: str) -> None:
    agg = d3["conditions_measured"]
    s_nll = [agg[c]["S_NLL"] for c in _CONDITION_KEYS]
    s_psnr = [agg[c]["S_PSNR"] for c in _CONDITION_KEYS]
    x = np.arange(len(_CONDITION_KEYS))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x - 0.19, s_nll, width=0.38, color="C0", label="S_NLL")
    ax.bar(x + 0.19, s_psnr, width=0.38, color="C1", label="S_PSNR (z=0)")
    for y, c, label in ((0.25, "C3", "0.25 strong band"),
                        (0.01, "C2", "0.01 negligible band")):
        ax.axhline(y, color=c, linestyle=":", lw=1.4, label=label)
    ax.set_xticks(x, ["C1 donor image+mask\n(ROUTING)",
                      "C2 own image+donor mask\n(attribution)",
                      "C3 donor image+own mask\n(attribution)"])
    ax.set_ylabel("sensitivity score S = |delta| / GAIN_REF")
    cls = d3["classification"]
    ax.set_title(f"D3 conditioner sensitivity -- C1: "
                 f"S_NLL={cls['S_NLL']:.4f} S_PSNR={cls['S_PSNR']:.4f} "
                 f"({cls['label']})", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_heatmap(d3: dict, path: str) -> None:
    mat = np.array([[rec[c]["delta_vs_c0"]["delta_z0_psnr"]
                     for c in _CONDITION_KEYS]
                    for rec in d3["per_slice"]], dtype=np.float64)
    vmax = float(np.max(np.abs(mat)))
    if vmax == 0.0:
        vmax = 1.0        # defined edge: all-zero deltas still render
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(3), _CONDITION_KEYS)
    ax.set_yticks(range(mat.shape[0]),
                  [f"s{i}" for i in range(mat.shape[0])])
    for i in range(mat.shape[0]):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center",
                    va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="signed delta_PSNR_z0 vs C0 [dB]")
    ax.set_title("D3 per-slice signed z=0 PSNR response to the donor "
                 "perturbation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_z0_vs_pm(d3: dict, path: str) -> None:
    agg = d3["conditions_measured"]
    z0 = [agg[c]["mean_delta_z0_psnr_vs_c0"] for c in _CONDITION_KEYS]
    pm = [agg[c]["mean_delta_pm_psnr_vs_c0"] for c in _CONDITION_KEYS]
    x = np.arange(len(_CONDITION_KEYS))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x - 0.19, z0, width=0.38, color="C0", label="z = 0")
    ax.bar(x + 0.19, pm, width=0.38, color="C4",
           label="posterior mean (128-bank)")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xticks(x, _CONDITION_KEYS)
    ax.set_ylabel("signed mean delta_PSNR vs C0 [dB]")
    ax.set_title("D3 z=0 vs posterior-mean PSNR response")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_d3_figures(d3: dict, out_dir: str) -> list:
    """Render the three D3 figures under out_dir/figures/. Returns the
    written paths. Any rendering failure is a typed ERROR (the figures
    are descriptive, but a silent skip is never permitted)."""
    fig_dir = os.path.join(out_dir, "figures")
    try:
        os.makedirs(fig_dir, exist_ok=True)
        targets = [
            ("d3_sensitivity_scores.png", _fig_sensitivity),
            ("d3_per_slice_psnr_deltas.png", _fig_heatmap),
            ("d3_z0_vs_pm_psnr.png", _fig_z0_vs_pm)]
        written = []
        for name, fn in targets:
            path = os.path.join(fig_dir, name)
            fn(d3, path)
            written.append(path)
    except StageError:
        raise
    except Exception as exc:  # noqa: BLE001 -- typed boundary
        logger.error("[SEQREF-TDIAG] D3_PLOT_FAILURE: %s: %s",
                     type(exc).__name__, exc)
        raise StageError(
            "D3_PLOT_FAILURE",
            f"D3 figure rendering failed: {type(exc).__name__}: {exc}",
            detail={"exception_type": type(exc).__name__,
                    "exception_message": str(exc)})
    logger.info("[SEQREF-TDIAG] D3 figures written: %s",
                [os.path.basename(p) for p in written])
    return written
