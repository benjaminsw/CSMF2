# =============================================================================
# STEP-1_1_1 v0.1 -- experiments.step_1_1_1.density_plots
# Purpose: end-of-run latent / cycle / reconstruction density visualisations
#          for the step_1_1_1 latent-shape investigation. Three functions,
#          each saving a single PNG.
# CONVENTION: NaN/shape/size errors -> logger.error + raise. No silent
#             fallback. No mock / dummy / placeholder.
# Changelog (NEW in v0.1):
#   * Introduced.
# Update summary:
#   v0.1 ships the three plots specified in the step_1_1_1 plan:
#     1. plot_latent_density          -- pooled z KDE vs N(0,1) reference
#     2. plot_cycle_density           -- per-pixel cycle error histogram
#     3. plot_reconstruction_panel    -- 4-row image grid (y / x / decode(z_enc) / decode(z_prior))
#   Together they show the direct visual cost of bad latent KS.
# =============================================================================
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_1_1"


def plot_latent_density(z: torch.Tensor,
                        save_path: Path,
                        *,
                        title: str = "Latent density vs N(0,1)",
                        lambda_used: float | None = None,
                        ) -> dict:
    """Pool all (B, D) latent values into a flat array, plot histogram +
    KDE alongside the standard normal reference. Report empirical mean /
    std / KS statistic in the legend.

    Returns a dict with the same scalars for the run report.
    """
    if z.dim() != 2:
        logger.error("[plot_latent_density] expected (B,D), got %s",
                     tuple(z.shape))
        raise ValueError(f"expected 2-D z, got shape {tuple(z.shape)}")
    z_flat = z.detach().cpu().numpy().reshape(-1).astype(np.float64)
    if not np.isfinite(z_flat).all():
        logger.error("[plot_latent_density] z contains non-finite values")
        raise ValueError("non-finite z")
    z_mean = float(z_flat.mean())
    z_std  = float(z_flat.std(ddof=1))
    ks_stat, _ = stats.kstest(z_flat, "norm")
    ks_stat = float(ks_stat)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    # histogram of empirical z
    ax.hist(z_flat, bins=120, density=True, alpha=0.45, color="#1f77b4",
            label=f"empirical z  (mean={z_mean:.3f}, std={z_std:.3f})")
    # reference N(0,1)
    xs = np.linspace(min(-5.0, z_flat.min()),
                     max( 5.0, z_flat.max()), 401)
    ax.plot(xs, stats.norm.pdf(xs), color="#d62728", lw=2.0,
            label="N(0, 1) reference")
    ax.set_xlabel("z value")
    ax.set_ylabel("density")
    suffix = f"   (lambda_moment={lambda_used:.3g})" \
        if lambda_used is not None else ""
    ax.set_title(f"{title}{suffix}\nKS statistic = {ks_stat:.4f}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("[plot_latent_density] saved -> %s  (ks=%.4f, mean=%.3f, "
                "std=%.3f)", save_path, ks_stat, z_mean, z_std)
    return {"ks": ks_stat, "mean": z_mean, "std": z_std, "n": int(z_flat.size)}


def plot_cycle_density(x_true: torch.Tensor,
                       x_cycle: torch.Tensor,
                       save_path: Path,
                       *,
                       title: str = "Cycle error density",
                       ) -> dict:
    """Histogram of per-pixel cycle error  (x_true - x_cycle).
    A sharp spike at 0 means the flow is numerically invertible. Wide
    tails predict poor reconstruction even when KS looks good.
    """
    if x_true.shape != x_cycle.shape:
        logger.error("[plot_cycle_density] shape mismatch true=%s cycle=%s",
                     tuple(x_true.shape), tuple(x_cycle.shape))
        raise ValueError(f"shape mismatch: {tuple(x_true.shape)} vs "
                         f"{tuple(x_cycle.shape)}")
    err = (x_true - x_cycle).detach().cpu().numpy().reshape(-1).astype(
        np.float64)
    if not np.isfinite(err).all():
        logger.error("[plot_cycle_density] cycle error has non-finite values")
        raise ValueError("non-finite cycle error")
    abs_max = float(np.abs(err).max())
    rms     = float(np.sqrt((err ** 2).mean()))
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    # symlog histogram so a sharp 0-spike + heavy tails are both visible
    ax.hist(err, bins=120, color="#2ca02c", alpha=0.7,
            label=f"max|err|={abs_max:.3e}  rms={rms:.3e}")
    ax.set_yscale("log")
    ax.set_xlabel("per-pixel error (x - decode(encode(x)))")
    ax.set_ylabel("count (log)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("[plot_cycle_density] saved -> %s  (max|err|=%.3e rms=%.3e)",
                save_path, abs_max, rms)
    return {"abs_max": abs_max, "rms": rms, "n": int(err.size)}


def plot_reconstruction_panel(y: torch.Tensor,
                              x_true: torch.Tensor,
                              x_recon_encoded: torch.Tensor,
                              x_recon_prior: torch.Tensor,
                              save_path: Path,
                              *,
                              title: str = "Reconstruction: encoded z vs N(0,1)-sampled z",
                              n_show: int = 8,
                              ) -> dict:
    """4-row image grid:
        row 1: y                  -- degraded input
        row 2: x_true             -- ground-truth clean image
        row 3: x_recon_encoded    -- decode( encode(x_true), y ): best case
        row 4: x_recon_prior      -- decode( z~N(0,1),      y ): realistic

    Gap between rows 3 and 4 is the direct visual cost of bad latent KS.
    """
    for name, t in [("y", y), ("x_true", x_true),
                    ("x_recon_encoded", x_recon_encoded),
                    ("x_recon_prior",   x_recon_prior)]:
        if t.dim() < 3:
            logger.error("[plot_reconstruction_panel] %s has dim<3: %s",
                         name, tuple(t.shape))
            raise ValueError(f"{name} must be (B, C, H, W) or (B, H, W), "
                             f"got {tuple(t.shape)}")
        if not torch.isfinite(t).all():
            logger.error("[plot_reconstruction_panel] %s has non-finite", name)
            raise ValueError(f"{name} has non-finite values")
    n = min(n_show, y.size(0))
    if n < 1:
        logger.error("[plot_reconstruction_panel] empty batch")
        raise ValueError("empty batch passed to reconstruction panel")

    def _to_img(t):
        # (B,1,H,W) -> (B,H,W). Clip to [0,1] for display only.
        if t.dim() == 4:
            t = t.squeeze(1)
        return t[:n].detach().cpu().clamp(0.0, 1.0).numpy()

    rows = [_to_img(y), _to_img(x_true),
            _to_img(x_recon_encoded), _to_img(x_recon_prior)]
    labels = ["y (degraded)", "x (true)",
              "decode(encode(x_true), y)",
              "decode(z~N(0,1), y)"]
    fig, axes = plt.subplots(4, n, figsize=(1.4 * n, 5.6), dpi=120)
    if n == 1:
        axes = axes.reshape(4, 1)
    for r in range(4):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(rows[r][c], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(labels[r], fontsize=8, rotation=0,
                              ha="right", va="center")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    # difference summary (encoded vs prior reconstructions)
    diff = (x_recon_encoded[:n] - x_recon_prior[:n]).detach().cpu().numpy()
    rel_diff = float(np.sqrt((diff ** 2).mean()))
    logger.info("[plot_reconstruction_panel] saved -> %s  rms(enc-prior)=%.3f",
                save_path, rel_diff)
    return {"n_shown": int(n), "rms_enc_minus_prior": rel_diff}
