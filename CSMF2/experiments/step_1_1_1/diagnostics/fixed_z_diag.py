# =============================================================================
# FZDY v0.1 -- experiments.step_1_1.diagnostics.fixed_z_diag
# Purpose: Phase-3 "fixed-z different-y" diagnostic. Decode ONE fixed latent z
#          under several different observations y; conditioning is working iff
#          the decoded x changes with y. Produces a grid PNG and a quantified
#          sensitivity metric over a bank of K fixed z's.
# CONVENTION: no fallback / mock / pass. Every bad input / non-finite tensor
#             -> logger.error + raise.
# Metric: per fixed z_k, sensitivity
#             s_k = mean_pixel(Var_i[ decode(z_k, y_i) ]) / mean_pixel(Var_i[y_i])
#         computed in PIXEL space (inverse_logit applied to decodes). Reported
#         as mean / min over the z-bank. passed = (mean >= tau).
# Pairs with the shuffle-gap signal: low s with FiLM "alive" == weak
#         conditioning even though FiLM weights move.
# Changelog (NEW in v0.1):
#   * Introduced. Grid (rows=y, col1=y, col2=decode(z,y)), K-z bank,
#     normalized sensitivity, informational pass/fail at tau.
# Update summary:
#   v0.1 implements FZDY scope C: core grid + quantified sensitivity gate +
#   shuffle-gap-corroborating metric + fixed-z bank (K seeds).
# =============================================================================
from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "FZDY"

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CSMF2.data.degrade import inverse_logit

_VAR_EPS = 1e-8


@torch.no_grad()
def fixed_z_different_y(model, y_batch: torch.Tensor, *,
                        epoch: int, out_dir, image_hw: tuple[int, int] = (28, 28),
                        n_y: int = 6, n_z: int = 3, tau: float = 0.05,
                        seed_base: int = 4242) -> dict:
    # model: a built expert exposing .cond(y)->h, .decode(z,h)->x_flat, .dim.
    # y_batch: (B,1,h,w) degraded observations in [0,1]. Uses the first n_y.
    # Returns {"fixed_z_different_y": {...}} for merge into the epoch record.
    if not (hasattr(model, "cond") and hasattr(model, "decode")
            and hasattr(model, "dim")):
        logger.error("[FZDY] model missing cond/decode/dim interface")
        raise AttributeError("model must expose cond(), decode(), dim")
    if y_batch.dim() != 4 or y_batch.size(1) != 1:
        logger.error("[FZDY] expected y (B,1,H,W), got %s", tuple(y_batch.shape))
        raise ValueError(f"expected y (B,1,H,W), got {tuple(y_batch.shape)}")
    if n_y < 2:
        logger.error("[FZDY] n_y must be >=2, got %d", n_y)
        raise ValueError(f"n_y must be >=2, got {n_y}")
    if y_batch.size(0) < n_y:
        logger.error("[FZDY] batch %d < n_y %d", y_batch.size(0), n_y)
        raise ValueError(f"batch {y_batch.size(0)} < n_y {n_y}")
    if n_z < 1:
        logger.error("[FZDY] n_z must be >=1, got %d", n_z)
        raise ValueError(f"n_z must be >=1, got {n_z}")
    H, W = image_hw
    dim = int(model.dim)
    if H * W != dim:
        logger.error("[FZDY] image_hw %s product != model.dim %d", image_hw, dim)
        raise ValueError(f"image_hw {image_hw} product != dim {dim}")

    device = y_batch.device
    dtype = y_batch.dtype
    y_sel = y_batch[:n_y]                              # (n_y,1,h,w)
    y_var = y_sel.flatten(1).var(dim=0, unbiased=False).mean()
    y_var = torch.clamp(y_var, min=_VAR_EPS)

    sens_per_z: list[float] = []
    first_decode_pix: torch.Tensor | None = None       # for the figure

    for k in range(n_z):
        g = torch.Generator(device=device).manual_seed(seed_base + k)
        z_k = torch.randn(1, dim, generator=g, device=device, dtype=dtype)
        xs = []
        for i in range(n_y):
            h_i = model.cond(y_sel[i:i + 1])           # (1, h_dim)
            x_i = model.decode(z_k, h_i)               # (1, dim), logit space
            xs.append(x_i)
        X = torch.cat(xs, dim=0)                        # (n_y, dim)
        if not torch.isfinite(X).all():
            logger.error("[FZDY] non-finite decode at z-index %d", k)
            raise ValueError(f"non-finite decode at z-index {k}")
        X_pix = inverse_logit(X)                        # (n_y, dim) in [0,1]
        out_disp = X_pix.var(dim=0, unbiased=False).mean()
        s_k = (out_disp / y_var).item()
        sens_per_z.append(s_k)
        if k == 0:
            first_decode_pix = X_pix.detach()

    sens_mean = float(sum(sens_per_z) / len(sens_per_z))
    sens_min = float(min(sens_per_z))
    passed = bool(sens_mean >= tau)

    # ---- figure (uses z_0) --------------------------------------------------
    out_dir = Path(out_dir)
    fig_dir = out_dir / "gen_diag"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"fixed_z_different_y_epoch_{epoch}.png"

    fig, axes = plt.subplots(n_y, 2, figsize=(4, 2 * n_y))
    if n_y == 1:
        axes = axes.reshape(1, 2)
    yh, yw = y_sel.shape[-2], y_sel.shape[-1]
    for i in range(n_y):
        y_img = y_sel[i, 0].detach().cpu().numpy()
        x_img = first_decode_pix[i].view(H, W).detach().cpu().numpy()
        axes[i, 0].imshow(y_img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 1].imshow(x_img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        if i == 0:
            axes[i, 0].set_title(f"y ({yh}x{yw})")
            axes[i, 1].set_title("decode(fixed z, y)")
    verdict = "PASS" if passed else "FAIL"
    fig.suptitle(f"FZDY epoch {epoch}  sens mean={sens_mean:.3f} "
                 f"min={sens_min:.3f}  tau={tau:.3f}  {verdict}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)
    logger.info("[FZDY] epoch %d: sens_mean=%.4f sens_min=%.4f passed=%s -> %s",
                epoch, sens_mean, sens_min, passed, fig_path)

    return {
        "fixed_z_different_y": {
            "n_y": n_y, "n_z": n_z, "tau": tau,
            "sensitivity_per_z": sens_per_z,
            "sensitivity_mean": sens_mean,
            "sensitivity_min": sens_min,
            "passed": passed,
            "fig_path": str(fig_path),
        }
    }
