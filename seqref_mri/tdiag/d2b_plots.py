# SEQREF-TDIAG v0.1 -- tdiag.d2b_plots
# LIFETIME: KEEP
# =============================================================================
# Purpose: D2b diagnostic figures (EXEC SS10.6 D2b). TWO figures under
#          the stage out-dir (results/_diag/diag/figures/); stacked bars
#          are deliberately NOT used (L_base and L_logdet may carry
#          opposite signs -- stacking would mislead):
#            1. d2b_aggregate_decomposition.png -- panel A: grouped
#               endpoint bars (L_base / L_logdet / NLL at step 0 and
#               step 500, prominent zero line); panel B: improvement
#               bars (dL_base / dL_logdet / dNLL) annotated with the
#               identity and identity_error.
#            2. d2b_per_slice_decomposition.png -- 8 slices x 3 columns
#               (dL_base / dL_logdet / dNLL) diverging heatmap centred
#               at zero, values printed in the cells (slice
#               heterogeneity is the D2a lesson; means alone are not
#               enough).
#          The figures are DESCRIPTIVE SUPPORT ONLY: they never become
#          evidence, never feed a band and never route. A rendering
#          failure is logger.error + typed StageError D2B_PLOT_FAILURE
#          (all-or-nothing: no facts artefact accompanies a plot
#          failure).
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * D2b slice (2026-08-19, under the same SS10.6 lock; NO contract
#     change): module introduced with the two D2b figures.
# Update summary:
#   v0.1 D2b lands the aggregate decomposition panels and the per-slice
#   diverging heatmap.
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


def _fig_aggregate(d2b: dict, path: str) -> None:
    agg = d2b["aggregate"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    labels = ["L_base", "L_logdet", "NLL"]
    v0 = [agg["step0"][k] for k in labels]
    v5 = [agg["step500"][k] for k in labels]
    x = np.arange(3)
    axa.bar(x - 0.2, v0, width=0.4, label="step 0", color="#4c78a8")
    axa.bar(x + 0.2, v5, width=0.4, label="step 500", color="#f58518")
    axa.axhline(0.0, color="0.2", lw=1.2)
    axa.set_xticks(x)
    axa.set_xticklabels(labels)
    axa.set_title("A: endpoint decomposition")
    axa.legend(fontsize=8)
    for xi, v in zip(x - 0.2, v0):
        axa.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                     xytext=(0, 3 if v >= 0 else -11), ha="center",
                     fontsize=7)
    for xi, v in zip(x + 0.2, v5):
        axa.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                     xytext=(0, 3 if v >= 0 else -11), ha="center",
                     fontsize=7)
    d = agg["delta"]
    dl = [d["delta_L_base"], d["delta_L_logdet"], d["delta_NLL"]]
    bars = axb.bar([0, 1, 2], dl, color=["#4c78a8", "#f58518",
                                         "#54a24b"])
    axb.axhline(0.0, color="0.2", lw=1.2)
    axb.set_xticks([0, 1, 2])
    axb.set_xticklabels(["dL_base", "dL_logdet", "dNLL"])
    axb.set_title("B: improvement decomposition (positive = gain)")
    for b_, v in zip(bars, dl):
        axb.annotate(f"{v:.1f}", (b_.get_x() + b_.get_width() / 2, v),
                     textcoords="offset points",
                     xytext=(0, 3 if v >= 0 else -11), ha="center",
                     fontsize=8)
    axb.annotate(f"dNLL = dL_base + dL_logdet\n"
                 f"identity_error = {d['identity_error']:.3g}",
                 xy=(0.02, 0.97), xycoords="axes fraction",
                 va="top", fontsize=8,
                 bbox=dict(boxstyle="round", fc="0.95", ec="0.7"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_per_slice(d2b: dict, path: str) -> None:
    rows = d2b["per_slice"]
    mat = np.array([[r["delta"]["delta_L_base"],
                     r["delta"]["delta_L_logdet"],
                     r["delta"]["delta_NLL"]] for r in rows],
                   dtype=np.float64)
    labels = [f"{r['identity']['file']}:{r['identity']['slice_index']}"
              for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    v = float(np.max(np.abs(mat))) if mat.size else 1.0
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["dL_base", "dL_logdet", "dNLL"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(mat[i, j]) > 0.6 * v
                    else "black")
    ax.set_title("D2b: per-slice improvement decomposition "
                 "(positive = gain)")
    fig.colorbar(im, ax=ax, label="delta (step0 - step500)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_d2b_figures(d2b: dict, out_dir: str) -> list:
    """Render the two D2b figures under out_dir/figures/. Returns the
    written paths. Any rendering failure is a typed ERROR (the figures
    are descriptive, but a silent skip is never permitted)."""
    fig_dir = os.path.join(out_dir, "figures")
    try:
        os.makedirs(fig_dir, exist_ok=True)
        targets = [
            ("d2b_aggregate_decomposition.png", _fig_aggregate),
            ("d2b_per_slice_decomposition.png", _fig_per_slice)]
        written = []
        for name, fn in targets:
            path = os.path.join(fig_dir, name)
            fn(d2b, path)
            written.append(path)
    except StageError:
        raise
    except Exception as exc:  # noqa: BLE001 -- typed boundary
        logger.error("[SEQREF-TDIAG] D2B_PLOT_FAILURE: %s: %s",
                     type(exc).__name__, exc)
        raise StageError(
            "D2B_PLOT_FAILURE",
            f"D2b figure rendering failed: {type(exc).__name__}: {exc}",
            detail={"exception_type": type(exc).__name__,
                    "exception_message": str(exc)})
    logger.info("[SEQREF-TDIAG] D2b figures written: %s",
                [os.path.basename(p) for p in written])
    return written
