# SEQREF-TDIAG v0.1 -- tdiag.d2a_plots
# LIFETIME: KEEP
# =============================================================================
# Purpose: D2a diagnostic figures (EXEC SS10.6 D2a). THREE figures,
#          rendered under the stage out-dir (results/_diag/diag/
#          figures/):
#            1. d2a_latent_movement.png  -- paired panels over the 8
#               slices: (A) ||z_true|| step0 -> step500, (B) log p_Z
#               percentile step0 -> step500; per-slice connected points,
#               Z_DIAG bank norm reference band (median, q05-q95).
#            2. d2a_bank_logp_ecdf.png  -- ECDF of the 128 Z_DIAG
#               log-density values with the 16 z_true markers overlaid
#               (marker style = step, colour = slice); percentile is the
#               registered quantity, so the ECDF makes it visually
#               direct.
#            3. d2a_topk_drift_heatmap.png -- signed Delta z for the top
#               20 coordinates by max-over-slices |Delta z| x 8 slices.
#          The figures are DESCRIPTIVE SUPPORT ONLY: they never become
#          evidence, never feed a band and never route. A rendering
#          failure is logger.error + typed StageError D2A_PLOT_FAILURE
#          (all-or-nothing: no facts artefact accompanies a plot
#          failure).
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * D2a slice (2026-08-19, under the same SS10.6 lock; NO contract
#     change): module introduced with the three D2a figures.
# Update summary:
#   v0.1 D2a lands the latent-movement panels, the bank log-density ECDF
#   with z_true markers and the top-K coordinate-drift heatmap.
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

_CMAP = plt.get_cmap("tab10")


def _slice_labels(d2a: dict) -> list:
    return [f"{s['identity']['file']}:{s['identity']['slice_index']}"
            for s in d2a["slices"]]


def _fig_movement(d2a: dict, path: str) -> None:
    slices = d2a["slices"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, s in enumerate(slices):
        c = _CMAP(i % 10)
        axa.plot([0, 500], [s["step0"]["norm_z"], s["step500"]["norm_z"]],
                 "-o", color=c, lw=1.2, ms=4)
        axb.plot([0, 500], [s["step0"]["percentile"]["percentile_percent"],
                            s["step500"]["percentile"]["percentile_percent"]],
                 "-o", color=c, lw=1.2, ms=4, label=_slice_labels(d2a)[i])
    ns = d2a["bank_reference"]["norm_summary"]
    axa.axhline(ns["median"], color="0.4", ls="--", lw=1,
                label="Z_DIAG norm median")
    axa.axhspan(ns["q05"], ns["q95"], color="0.4", alpha=0.15,
                label="Z_DIAG norm q05-q95")
    axa.set_xlabel("replay step")
    axa.set_ylabel(r"$\|z_{true}\|$")
    axa.set_title("A: true-latent norm")
    axa.legend(fontsize=7)
    axb.axhline(50.0, color="0.4", ls="--", lw=1)
    axb.set_ylim(-2, 102)
    axb.set_xlabel("replay step")
    axb.set_ylabel(r"percentile of $\log p_Z(z_{true})$ in Z_DIAG (%)")
    axb.set_title("B: bank log-density percentile")
    axb.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_ecdf(d2a: dict, path: str) -> None:
    logp = np.sort(np.asarray(
        d2a["bank_reference"]["logp_values"], dtype=np.float64))
    n = logp.shape[0]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.step(logp, 100.0 * np.arange(1, n + 1) / n, where="post",
            color="0.3", lw=1.4, label="Z_DIAG ECDF (n=128)")
    labels = _slice_labels(d2a)
    for i, s in enumerate(d2a["slices"]):
        c = _CMAP(i % 10)
        ax.plot(s["step0"]["log_pz"],
                s["step0"]["percentile"]["percentile_percent"],
                "o", color=c, ms=6, mfc="none", mew=1.4)
        ax.plot(s["step500"]["log_pz"],
                s["step500"]["percentile"]["percentile_percent"],
                "^", color=c, ms=6, label=labels[i])
    ax.plot([], [], "o", color="0.2", mfc="none", mew=1.4,
            label="z_true step0")
    ax.plot([], [], "^", color="0.2", label="z_true step500")
    ax.set_xlabel(r"$\log p_Z$")
    ax.set_ylabel("percentile (%)")
    ax.set_ylim(-2, 105)
    ax.set_title("D2a: true-latent density inside the Z_DIAG bank")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_topk(d2a: dict, path: str) -> None:
    g = d2a["global_topk_drift"]
    mat = np.asarray(g["delta_matrix"], dtype=np.float64)  # slices x K
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    v = float(np.max(np.abs(mat))) if mat.size else 1.0
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(len(g["coordinate_indices"])))
    ax.set_xticklabels([str(j) for j in g["coordinate_indices"]],
                       rotation=90, fontsize=6)
    ax.set_yticks(range(len(_slice_labels(d2a))))
    ax.set_yticklabels(_slice_labels(d2a), fontsize=7)
    ax.set_xlabel("coordinate index (top 20 by max |Delta z|)")
    ax.set_title(r"D2a: signed $z_{500}-z_0$ drift, top coordinates")
    fig.colorbar(im, ax=ax, label=r"$\Delta z$")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_d2a_figures(d2a: dict, out_dir: str) -> list:
    """Render the three D2a figures under out_dir/figures/. Returns the
    written paths. Any rendering failure is a typed ERROR (the figures
    are descriptive, but a silent skip is never permitted)."""
    fig_dir = os.path.join(out_dir, "figures")
    try:
        os.makedirs(fig_dir, exist_ok=True)
        targets = [
            ("d2a_latent_movement.png", _fig_movement),
            ("d2a_bank_logp_ecdf.png", _fig_ecdf),
            ("d2a_topk_drift_heatmap.png", _fig_topk)]
        written = []
        for name, fn in targets:
            path = os.path.join(fig_dir, name)
            fn(d2a, path)
            written.append(path)
    except StageError:
        raise
    except Exception as exc:  # noqa: BLE001 -- typed boundary
        logger.error("[SEQREF-TDIAG] D2A_PLOT_FAILURE: %s: %s",
                     type(exc).__name__, exc)
        raise StageError(
            "D2A_PLOT_FAILURE",
            f"D2a figure rendering failed: {type(exc).__name__}: {exc}",
            detail={"exception_type": type(exc).__name__,
                    "exception_message": str(exc)})
    logger.info("[SEQREF-TDIAG] D2a figures written: %s",
                [os.path.basename(p) for p in written])
    return written
