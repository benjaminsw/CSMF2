# SEQREF-TDIAG v0.1 -- tdiag.d2c_plots
# LIFETIME: KEEP
# =============================================================================
# Purpose: D2c diagnostic figures (EXEC SS10.6 D2c). THREE figures
#          under the stage out-dir (results/_diag/diag/figures/); raw
#          per-state NLL bars are deliberately NOT plotted (at this
#          scale the absolute values would obscure the quantity of
#          interest -- the per-dimension DIFFERENCE, review
#          2026-08-20):
#            1. d2c_holdout_delta_distribution.png -- sorted per-volume
#               delta_NLL/dim strip (ascending), with horizontal lines
#               at 0, G_hold, G_train, 0.25*G_train and 0.75*G_train
#               (the locked memorization/transfer reference levels).
#            2. d2c_train_vs_holdout_gain.png -- G_train vs G_hold
#               points/bars with R annotated and the locked band
#               boundaries 0.25/0.75*G_train shown (the campaign-level
#               visual: it directly mirrors the D2c decision rule).
#            3. d2c_delta_nll_vs_delta_psnr.png -- per-volume scatter
#               of delta_NLL/dim vs delta_PSNR (likelihood gain vs
#               reconstruction change on unseen volumes; extends the
#               D1/D2b density-fidelity question to the holdout set).
#          The figures are DESCRIPTIVE SUPPORT ONLY: they never become
#          evidence, never feed a band and never route. A rendering
#          failure is logger.error + typed StageError D2C_PLOT_FAILURE
#          (all-or-nothing: no facts artefact accompanies a plot
#          failure).
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * D2c slice (2026-08-20, under the same SS10.6 lock; NO contract
#     change): module introduced with the three D2c figures.
# Update summary:
#   v0.1 D2c lands the sorted per-volume delta distribution, the
#   train-vs-holdout gain panel and the likelihood-vs-reconstruction
#   scatter.
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


def _agg(d2c: dict) -> dict:
    return d2c["aggregate"]


def _fig_distribution(d2c: dict, path: str) -> None:
    agg = _agg(d2c)
    vals = np.sort(np.array(
        [r["delta"]["delta_nll_per_dim"] for r in d2c["per_slice"]],
        dtype=np.float64))
    g_train = agg["G_train"]
    g_hold = agg["G_hold"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.scatter(np.arange(1, len(vals) + 1), vals, s=28, zorder=3,
               color="C0", label="holdout volume (sorted)")
    for y, color, ls, label in (
            (0.0, "k", "-", "0"),
            (g_hold, "C0", "--", f"G_hold = {g_hold:.3f}"),
            (g_train, "C2", "-", f"G_train = {g_train:.3f}"),
            (0.25 * g_train, "C3", ":", "0.25 x G_train"),
            (0.75 * g_train, "C1", ":", "0.75 x G_train")):
        ax.axhline(y, color=color, linestyle=ls, label=label, lw=1.4)
    ax.set_xlabel("holdout volume rank (sorted by delta_NLL/dim)")
    ax.set_ylabel("delta_NLL / dim (step0 - step500)")
    ax.set_title(f"D2c holdout per-volume likelihood gain -- "
                 f"R = {agg['R']:.4f} ({agg['classification']['label']})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_gain(d2c: dict, path: str) -> None:
    agg = _agg(d2c)
    g_train = agg["G_train"]
    g_hold = agg["G_hold"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.bar([0, 1], [g_train, g_hold], width=0.55,
           color=["C2", "C0"])
    ax.set_xticks([0, 1], ["G_train\n(8 TINY slices)",
                           "G_hold\n(32 unseen volumes)"])
    for y, c, label in ((0.25 * g_train, "C3", "0.25 x G_train"),
                        (0.75 * g_train, "C1", "0.75 x G_train")):
        ax.axhline(y, color=c, linestyle=":", lw=1.4, label=label)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.annotate(f"R = {agg['R']:.4f}\n"
                f"{agg['classification']['label']}",
                xy=(0.5, max(g_train, g_hold)),
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("G = delta_NLL / dim")
    ax.set_title("D2c train vs holdout likelihood gain")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_scatter(d2c: dict, path: str) -> None:
    x = np.array([r["delta"]["delta_nll_per_dim"]
                  for r in d2c["per_slice"]], dtype=np.float64)
    y = np.array([r["delta"]["delta_psnr"] for r in d2c["per_slice"]],
                 dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(x, y, s=30, color="C0", zorder=3)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.axvline(0.0, color="k", lw=1.0)
    ax.set_xlabel("delta_NLL / dim (step0 - step500)")
    ax.set_ylabel("delta_PSNR (step500 - step0) [dB]")
    ax.set_title("D2c holdout: likelihood gain vs reconstruction change")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_d2c_figures(d2c: dict, out_dir: str) -> list:
    """Render the three D2c figures under out_dir/figures/. Returns the
    written paths. Any rendering failure is a typed ERROR (the figures
    are descriptive, but a silent skip is never permitted)."""
    fig_dir = os.path.join(out_dir, "figures")
    try:
        os.makedirs(fig_dir, exist_ok=True)
        targets = [
            ("d2c_holdout_delta_distribution.png", _fig_distribution),
            ("d2c_train_vs_holdout_gain.png", _fig_gain),
            ("d2c_delta_nll_vs_delta_psnr.png", _fig_scatter)]
        written = []
        for name, fn in targets:
            path = os.path.join(fig_dir, name)
            fn(d2c, path)
            written.append(path)
    except StageError:
        raise
    except Exception as exc:  # noqa: BLE001 -- typed boundary
        logger.error("[SEQREF-TDIAG] D2C_PLOT_FAILURE: %s: %s",
                     type(exc).__name__, exc)
        raise StageError(
            "D2C_PLOT_FAILURE",
            f"D2c figure rendering failed: {type(exc).__name__}: {exc}",
            detail={"exception_type": type(exc).__name__,
                    "exception_message": str(exc)})
    logger.info("[SEQREF-TDIAG] D2c figures written: %s",
                [os.path.basename(p) for p in written])
    return written
