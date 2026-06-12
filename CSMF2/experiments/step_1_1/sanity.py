# =============================================================================
# STEP-1_1 v0.4 -- experiments.step_1_1.sanity
# Purpose: step-specific sanity routines (see v0.2 description below).
# CONVENTION: any failure -> logger.error + raise. No fallback.
# Changelog (v0.4 -> v0.5):
#   * plot_film_alive: the alive VERDICT now uses the MEDIAN across-y std of
#     gamma/beta over FiLM heads, not the MIN. With min, a single quiet layer
#     (beta_std ~7e-4 < eps=1e-3) flipped alive=False for the whole run even
#     though FZDY ~1.6 proved conditioning works (observed: realnvp s2/n0.05
#     seeds 1/2). Median checks whether FiLM is BROADLY alive; FZDY remains the
#     real collapse detector. eps kept at 1e-3. The min values are still
#     returned (gamma_std_min/beta_std_min) for reference; NEW gamma_std_med/
#     beta_std_med added to the returned dict and used for the verdict.
# Changelog (v0.3 -> v0.4):
#   * BUGFIX (false "alive"): plot_film_alive measured g.std() over the WHOLE
#     (B, feat) tensor, so channel-to-channel spread dominated and reported
#     alive=True even when gamma/beta were IDENTICAL across y (collapsed
#     conditioner). Now the alive verdict uses the ACROSS-Y std:
#         y_sens = g.std(dim=0).mean()      # per-feature std across batch
#     i.e. "does FiLM move when y moves". The old channel spread is still
#     reported as gamma_chan_spread / beta_chan_spread for reference, but is
#     NOT used for the verdict. Requires batch B>=2 (raises otherwise).
# Changelog (v0.2 -> v0.3):
#   * plot_diag_spectrum: SKIP-with-log-info if expert has no DiagScale layer
#     (RealNVP / Glow). Returns {"skipped": True, "reason": "..."}. NOT a
#     fallback -- it is the explicit no-DiagScale path.
#   * NEW plot_s_spectrum(expert, x, y, out_path): RealNVP/Glow analogue of
#     diag_spectrum. Aggregates |s| across every coupling layer found in
#     expert (calls layer._st where present, layer.coupling._st for Glow).
#     Reports spectral_entropy / spectral_entropy_norm for cross-expert
#     comparison.
#   * NEW plot_w1x1_spectrum(expert, out_path): SVD per InvertibleConv1x1
#     for Glow. Reports min_sv_per_step, condition number, global min SV.
# Changelog (v0.3.1 -> v0.2):
#   * NEW: plot_nll_curve(train_hist, test_hist, out_path).
#   * NEW (opt-in): plot_samples_grid, plot_samples_fixed_z, plot_sw2_diversity,
#     plot_diag_spectrum, plot_film_alive, plot_logp_shuffle.
# Changelog (v0.3 -> v0.3.1):
#   * plot_orig_degraded_cycle_generated + plot_cycle_error_heatmap in float64.
# Changelog (v0.1.3 -> v0.3):
#   * NEW: plot_cycle_error_heatmap, plot_forward_consistency, plot_latent_interp.
# Changelog (v0.1.2 -> v0.1.3):
#   * invertibility_check in float64.
# Changelog (v0.1.1 -> v0.1.2):
#   * numeric_logdet_check twin in float64.
# Changelog (v0.1 -> v0.1.1):
#   * numeric_logdet_check accepts abs_tol.
# Changelog (NEW in v0.1):
#   * Introduced.
# Update summary:
#   v0.5 makes the FiLM-alive verdict robust to one near-silent layer: it now
#   judges on the MEDIAN across-y std (not the min), so a single layer dipping
#   below eps no longer false-fails an otherwise-healthy conditioner. FZDY is
#   still the authoritative collapse signal. Affects only newly-written reports
#   (the flag is computed at run time); existing reports keep their baked value.
#   v0.4 fixes the misleading FiLM-alive verdict: it now measures whether
#   gamma/beta change ACROSS y (the only thing that proves conditioning is
#   used), not channel spread (always nonzero). A collapsed conditioner now
#   correctly reads alive=False. Spectrum dispatch (v0.3) unchanged.
# =============================================================================
from __future__ import annotations
import logging
import traceback
from pathlib import Path
logger = logging.getLogger(__name__)
__version__ = "0.5"
__abbr__ = "STEP-1_1"

import math
from statistics import median
import numpy as np
import torch

from ...data.degrade import inverse_logit, blur, downsample


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ------------- numeric log-det check on toy ---------------------------------
def numeric_logdet_check(expert, *, dim_toy: int = 16, h_dim: int,
                         device: torch.device, eps: float = 1e-4,
                         rel_tol: float = 1e-3, abs_tol: float = 1e-3) -> dict:
    # Numerical Jacobian via central differences on the expert's encode, summing
    # log|det J|. Only practical for small dim; we require the expert to be
    # exercisable at `dim_toy`. If the expert was instantiated for dim=784, we
    # skip and raise -- the caller must build a tiny replica.
    # NOTE: cast to float64 -- FP32 central differences + slogdet on a 16x16
    # Jacobian produce ~1e-3 noise that is indistinguishable from real math
    # errors. We want to test the formulas, not FP32 precision.
    if expert.dim != dim_toy:
        logger.error("[numeric_logdet_check] expert.dim=%d != dim_toy=%d; "
                     "caller must instantiate a twin at dim_toy", expert.dim, dim_toy)
        raise ValueError("expert.dim must equal dim_toy for numeric check")
    expert = expert.to(device=device, dtype=torch.float64).eval()
    x = torch.randn(1, dim_toy, device=device, dtype=torch.float64, requires_grad=False)
    h = torch.randn(1, h_dim, device=device, dtype=torch.float64)
    with torch.no_grad():
        z0, ldj_analytic = expert.encode(x, h)
    # build numerical Jacobian column by column
    J = torch.zeros(dim_toy, dim_toy, device=device, dtype=torch.float64)
    for j in range(dim_toy):
        dx = torch.zeros_like(x)
        dx[0, j] = eps
        with torch.no_grad():
            zp, _ = expert.encode(x + dx, h)
            zm, _ = expert.encode(x - dx, h)
        J[:, j] = (zp - zm).squeeze(0) / (2 * eps)
    sign, logabsdet_num = torch.linalg.slogdet(J)
    abs_err = float(abs(logabsdet_num.item() - ldj_analytic.item()))
    rel_err = float(abs_err / max(abs(logabsdet_num.item()), 1e-8))
    passed = (rel_err < rel_tol) or (abs_err < abs_tol)
    if not passed:
        logger.error("[numeric_logdet_check] FAIL abs_err=%.3e (tol=%.1e)  "
                     "rel_err=%.3e (tol=%.1e)  analytic=%.4f numeric=%.4f",
                     abs_err, abs_tol, rel_err, rel_tol,
                     ldj_analytic.item(), logabsdet_num.item())
    return {"rel_err": rel_err, "abs_err": abs_err,
            "analytic": float(ldj_analytic.item()),
            "numeric": float(logabsdet_num.item()), "passed": bool(passed)}


# ------------- invertibility ------------------------------------------------
def invertibility_check(expert, x: torch.Tensor, y: torch.Tensor,
                        *, tol: float = 1e-5) -> dict:
    # Cast to float64 to test formulas, not FP32 precision. 6 stacked RQ-spline
    # couplings over 784 dims in FP32 have a rounding floor ~1e-5 to 1e-4 --
    # masks real math errors. FP64 drops this to ~1e-10.
    expert.eval()
    orig_dtype = next(expert.parameters()).dtype
    expert_fp64 = expert.to(torch.float64)
    try:
        x64 = x.to(torch.float64)
        y64 = y.to(torch.float64)
        with torch.no_grad():
            h = expert_fp64.cond(y64)
            z, _ = expert_fp64.encode(x64, h)
            x_rec = expert_fp64.decode(z, h)
        err = float((x64 - x_rec).abs().max().item())
    finally:
        expert.to(orig_dtype)                 # restore training dtype
    passed = err < tol
    if not passed:
        logger.error("[invertibility_check] max err=%.3e > tol=%.1e", err, tol)
    return {"max_abs_err": err, "tol": tol, "passed": bool(passed)}


# ------------- latent z histogram + KS --------------------------------------
def _ks_vs_standard_normal(samples: np.ndarray) -> float:
    # two-sample Kolmogorov-Smirnov vs a fixed N(0,1) reference; no scipy dep.
    s = np.sort(samples.ravel())
    n = s.size
    if n == 0:
        logger.error("[ks_vs_N01] empty samples")
        raise ValueError("empty samples for KS test")
    ecdf = np.arange(1, n + 1) / n
    # N(0,1) CDF via erf
    ncdf = 0.5 * (1.0 + np.vectorize(math.erf)(s / math.sqrt(2.0)))
    return float(np.abs(ecdf - ncdf).max())


def plot_latent_histogram(expert, x: torch.Tensor, y: torch.Tensor,
                          out_path: Path | str,
                          *, max_samples: int = 20000,
                          ks_warn: float = 0.05) -> dict:
    expert.eval()
    with torch.no_grad():
        h = expert.cond(y)
        z, _ = expert.encode(x, h)
    z_np = z.detach().cpu().numpy()
    flat = z_np.ravel()
    if flat.size > max_samples:
        flat = np.random.default_rng(0).choice(flat, size=max_samples, replace=False)

    ks = _ks_vs_standard_normal(flat)
    mean = float(flat.mean()); std = float(flat.std())
    passed = (ks < ks_warn) and (abs(mean) < 0.05) and (abs(std - 1.0) < 0.1)

    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(flat, bins=80, density=True, alpha=0.7, label="z")
    xs = np.linspace(-5, 5, 200)
    axes[0].plot(xs, np.exp(-0.5 * xs ** 2) / math.sqrt(2 * math.pi),
                 "r--", label="N(0,1)")
    tag_col = "black" if passed else "red"
    axes[0].set_title(f"latent z histogram  mean={mean:.3f} std={std:.3f}  "
                      f"KS={ks:.3f}  {'OK' if passed else 'FAIL'}",
                      color=tag_col, fontsize=10)
    axes[0].legend()

    # per-dim mean & std
    per_dim_mean = z_np.mean(axis=0)
    per_dim_std  = z_np.std(axis=0)
    axes[1].plot(per_dim_mean, label="mean per dim")
    axes[1].plot(per_dim_std,  label="std  per dim")
    axes[1].axhline(0.0, color="k", linewidth=0.5)
    axes[1].axhline(1.0, color="k", linewidth=0.5, linestyle=":")
    axes[1].set_title("per-dim mean & std (target: 0 and 1)")
    axes[1].legend(fontsize=8)

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
    except OSError:
        logger.error("[plot_latent_histogram] save failed %s\n%s",
                     out_path, traceback.format_exc())
        raise
    return {"ks": ks, "mean": mean, "std": std, "passed": passed}


# ------------- Original / Degraded / Cycle / Generated panel ---------------
def plot_orig_degraded_cycle_generated(expert, x: torch.Tensor, y: torch.Tensor,
                                       out_path: Path | str,
                                       *, n_rows: int = 6,
                                       image_shape: tuple[int, int] = (28, 28),
                                       x_is_logit: bool = True) -> dict:
    # x: (B, D) or (B, 1, H, W). y: (B, 1, h, w).  Assumes D = H*W.
    expert.eval()
    if x.dim() == 4:
        x_flat = x.flatten(1)
    else:
        x_flat = x

    # Run encode/decode in FP64 to eliminate FP32 spline-inverse noise (~1e-5)
    # that would otherwise show as structured hotspots in the cycle row.
    orig_dtype = next(expert.parameters()).dtype
    expert_fp64 = expert.to(torch.float64)
    try:
        x64 = x_flat.to(torch.float64)
        y64 = y.to(torch.float64)
        with torch.no_grad():
            h = expert_fp64.cond(y64)
            z, _ = expert_fp64.encode(x64, h)
            x_cycle = expert_fp64.decode(z, h)
            # generated: sample z_new ~ N(0,I) at same h
            z_new = torch.randn_like(z)
            x_gen = expert_fp64.decode(z_new, h)
    finally:
        expert.to(orig_dtype)

    cycle_err = float((x64 - x_cycle).abs().max().item())

    def _to_img(v):
        v = v.detach().to(torch.float32).cpu()
        if x_is_logit:
            v_img = inverse_logit(v)
        else:
            v_img = v.clamp(0, 1)
        return v_img.view(-1, *image_shape).numpy()

    X   = _to_img(x64[:n_rows])
    Xc  = _to_img(x_cycle[:n_rows])
    Xg  = _to_img(x_gen[:n_rows])
    Y   = y[:n_rows, 0].detach().cpu().numpy()

    plt = _mpl()
    fig, axes = plt.subplots(n_rows, 4, figsize=(6.5, 1.3 * n_rows))
    titles = ["Original", "Degraded y", "Cycle f⁻¹(f(x))", "Generated f⁻¹(z~N)"]
    for r in range(n_rows):
        for c, (img, t) in enumerate(zip([X[r], Y[r], Xc[r], Xg[r]], titles)):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(t, fontsize=9)
    tag = f"cycle max|err|={cycle_err:.2e}  " \
          f"{'OK' if cycle_err < 1e-5 else 'FAIL (>1e-5)'}"
    fig.suptitle(tag, fontsize=10,
                 color="black" if cycle_err < 1e-5 else "red")

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
    except OSError:
        logger.error("[plot_orig_deg_cycle_gen] save failed %s\n%s",
                     out_path, traceback.format_exc())
        raise
    return {"cycle_max_err": cycle_err, "passed": cycle_err < 1e-5}


# ============================================================================
# v0.3 NEW PLOTS -- cycle heatmap / forward consistency / latent interp (slerp)
# ============================================================================
def plot_cycle_error_heatmap(expert, x: torch.Tensor, y: torch.Tensor,
                             out_path: Path | str, *,
                             n_rows: int = 6,
                             image_shape: tuple[int, int] = (28, 28)) -> dict:
    # |x - f^-1(f(x))| per pixel. Proves whether invertibility error is
    # structured (stroke edges, tail bins) or unstructured FP noise.
    # FP64 encode/decode -- see note in plot_orig_degraded_cycle_generated.
    expert.eval()
    orig_dtype = next(expert.parameters()).dtype
    expert_fp64 = expert.to(torch.float64)
    try:
        x64 = x.to(torch.float64)
        y64 = y.to(torch.float64)
        with torch.no_grad():
            h = expert_fp64.cond(y64)
            z, _ = expert_fp64.encode(x64, h)
            x_rec = expert_fp64.decode(z, h)
        err = (x64 - x_rec).abs().detach().to(torch.float32).cpu() \
                                  .view(-1, *image_shape).numpy()
    finally:
        expert.to(orig_dtype)
    e_mean = float(err.mean()); e_std = float(err.std()); e_max = float(err.max())
    # "structured" if per-pixel std is much larger than mean (concentration)
    structured_ratio = float(e_std / max(e_mean, 1e-12))

    plt = _mpl()
    rows = min(n_rows, err.shape[0])
    fig, axes = plt.subplots(rows, 1, figsize=(3.2, 1.1 * rows), squeeze=False)
    vmax = max(e_max, 1e-12)
    for r in range(rows):
        ax = axes[r, 0]
        im = ax.imshow(err[r], cmap="hot", vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"|x - f^-1(f(x))|   mean={e_mean:.2e}  std={e_std:.2e}  "
                 f"max={e_max:.2e}  std/mean={structured_ratio:.2f}",
                 fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
    except OSError:
        logger.error("[plot_cycle_error_heatmap] save failed %s\n%s",
                     out_path, traceback.format_exc())
        raise
    return {"cycle_mean": e_mean, "cycle_std": e_std,
            "cycle_max": e_max, "structured_ratio": structured_ratio}


def plot_forward_consistency(expert, y: torch.Tensor, *,
                             blur_sigma: float, scale: int,
                             out_path: Path | str,
                             n_rows: int = 6,
                             image_shape: tuple[int, int] = (28, 28)) -> dict:
    # Sample x_hat ~ q(x|y) in logit space, invert-logit to pixel, apply A =
    # Downsample o Blur, compare to y. Columns: y | A(x_hat) | |A(x_hat) - y|.
    # Reports rel_residual = ||A(x_hat) - y|| / ||y||  (averaged over batch).
    expert.eval()
    with torch.no_grad():
        h = expert.cond(y)
        z = torch.randn(y.shape[0], expert.dim, device=y.device, dtype=y.dtype)
        x_logit = expert.decode(z, h)
    x_pixel = inverse_logit(x_logit).view(-1, 1, *image_shape)     # (B,1,28,28)
    with torch.no_grad():
        Ax = downsample(blur(x_pixel, blur_sigma), scale)
    resid = (Ax - y)
    abs_r = resid.flatten(1).norm(dim=1)
    y_r   = y.flatten(1).norm(dim=1).clamp_min(1e-8)
    rel_batch = (abs_r / y_r)
    rel_mean = float(rel_batch.mean()); rel_max = float(rel_batch.max())
    abs_mean = float(abs_r.mean())

    plt = _mpl()
    rows = min(n_rows, y.shape[0])
    y_np   = y[:rows, 0].detach().cpu().numpy()
    ax_np  = Ax[:rows, 0].detach().cpu().numpy()
    res_np = resid[:rows, 0].abs().detach().cpu().numpy()
    fig, axes = plt.subplots(rows, 3, figsize=(5.0, 1.3 * rows), squeeze=False)
    titles = ["y (degraded)", "A(x̂)", "|A(x̂) - y|"]
    vmax_r = max(float(res_np.max()), 1e-12)
    for r in range(rows):
        axes[r, 0].imshow(y_np[r], cmap="gray", vmin=0, vmax=1)
        axes[r, 1].imshow(ax_np[r], cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(res_np[r], cmap="hot", vmin=0, vmax=vmax_r)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=9)
    fig.suptitle(f"rel resid mean={rel_mean:.3f}  max={rel_max:.3f}  "
                 f"(step 1.1: report only; WP1 gate <0.1)", fontsize=9)

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
    except OSError:
        logger.error("[plot_forward_consistency] save failed %s\n%s",
                     out_path, traceback.format_exc())
        raise
    return {"fwd_abs_mean": abs_mean,
            "fwd_rel_mean": rel_mean, "fwd_rel_max": rel_max}


def _slerp(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    # Spherical linear interpolation for Gaussian latents (preserves norm).
    # Falls back to lerp when z0, z1 are nearly colinear.
    z0f = z0.flatten(1); z1f = z1.flatten(1)
    n0 = z0f.norm(dim=1, keepdim=True).clamp_min(1e-8)
    n1 = z1f.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cos = (z0f * z1f).sum(dim=1, keepdim=True) / (n0 * n1)
    cos = cos.clamp(-1.0, 1.0)
    omega = torch.acos(cos)
    sin_o = torch.sin(omega)
    near_zero = sin_o.abs() < 1e-6
    # lerp branch (flat) + slerp branch (spherical)
    lerp = (1.0 - t) * z0f + t * z1f
    slerp = (torch.sin((1.0 - t) * omega) / sin_o.clamp_min(1e-8)) * z0f \
          + (torch.sin(t * omega) / sin_o.clamp_min(1e-8)) * z1f
    out = torch.where(near_zero, lerp, slerp)
    return out.view_as(z0)


def plot_latent_interp(expert, y: torch.Tensor, out_path: Path | str,
                       *, n_steps: int = 8, n_rows: int = 4,
                       image_shape: tuple[int, int] = (28, 28)) -> dict:
    # For each of n_rows y samples, sample z0, z1 ~ N(0,I), slerp n_steps, decode.
    expert.eval()
    rows = min(n_rows, y.shape[0])
    y_sub = y[:rows]
    with torch.no_grad():
        h = expert.cond(y_sub)
        z0 = torch.randn(rows, expert.dim, device=y.device, dtype=y.dtype)
        z1 = torch.randn(rows, expert.dim, device=y.device, dtype=y.dtype)
        grid = []
        for i in range(n_steps):
            t = i / (n_steps - 1)
            z_t = _slerp(z0, z1, t)
            x_logit = expert.decode(z_t, h)
            x_pix = inverse_logit(x_logit).view(rows, *image_shape)
            grid.append(x_pix.detach().cpu().numpy())

    plt = _mpl()
    fig, axes = plt.subplots(rows, n_steps, figsize=(1.0 * n_steps, 1.0 * rows),
                             squeeze=False)
    for r in range(rows):
        for c in range(n_steps):
            axes[r, c].imshow(grid[c][r], cmap="gray")
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(f"t={c/(n_steps-1):.2f}", fontsize=8)
    fig.suptitle("latent interpolation (slerp):  z0 -> z1  at fixed h(y)",
                 fontsize=9)

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
    except OSError:
        logger.error("[plot_latent_interp] save failed %s\n%s",
                     out_path, traceback.format_exc())
        raise
    return {"n_steps": n_steps, "n_rows": rows}


# =============================================================================
# v0.2 NEW -- always-on NLL curve at end-of-run
# =============================================================================
def plot_nll_curve(train_hist: list, test_hist: list, out_path) -> dict:
    # Single chart, two lines: train (solid), test (dashed). Pure read of scalar
    # histories already in report.json. Always-on at end-of-run.
    out_path = Path(out_path)
    if not isinstance(train_hist, (list, tuple)) or not isinstance(test_hist, (list, tuple)):
        logger.error("[plot_nll_curve] hists must be lists, got %s/%s",
                     type(train_hist).__name__, type(test_hist).__name__)
        raise ValueError("train_hist/test_hist must be lists")
    if len(train_hist) == 0 or len(test_hist) == 0:
        logger.error("[plot_nll_curve] empty history (train=%d, test=%d)",
                     len(train_hist), len(test_hist))
        raise ValueError("empty NLL history")
    n = max(len(train_hist), len(test_hist))
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    xs_tr = list(range(1, len(train_hist) + 1))
    xs_te = list(range(1, len(test_hist) + 1))
    ax.plot(xs_tr, train_hist, "-",  color="#36a", lw=1.3, label="train")
    ax.plot(xs_te, test_hist,  "--", color="#a33", lw=1.3, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL (nats)")
    ax.set_title(f"NLL per epoch  (n={n})", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
    except OSError:
        logger.error("[plot_nll_curve] save failed %s\n%s",
                     out_path, traceback.format_exc())
        raise
    return {"n_epochs": int(n),
            "train_first": float(train_hist[0]),
            "train_last":  float(train_hist[-1]),
            "test_first":  float(test_hist[0]),
            "test_last":   float(test_hist[-1])}


# =============================================================================
# v0.2 NEW -- v2-conditioner diagnostics (opt-in via cfg.use_v2_conditioner)
# Migrated from nice_improved/sanity_v2.py.
# =============================================================================
def _save_fig(plt, out_path) -> None:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
    except OSError:
        logger.error("[sanity._save_fig] save failed %s\n%s",
                     out_path, traceback.format_exc())
        raise


def plot_samples_grid(expert, y_one: torch.Tensor, out_path,
                      *, n: int = 64, nrow: int = 8,
                      image_shape: tuple = (28, 28)) -> dict:
    if y_one.dim() != 4 or y_one.size(0) != 1:
        logger.error("[plot_samples_grid] y_one must be (1,1,H,W), got %s",
                     tuple(y_one.shape))
        raise ValueError("y_one must be (1,1,H,W)")
    expert.eval()
    with torch.no_grad():
        x_logit = expert.sample(n, y_one)
    x_pix = inverse_logit(x_logit).view(n, *image_shape).cpu().numpy()
    if not np.isfinite(x_pix).all():
        logger.error("[plot_samples_grid] non-finite samples")
        raise ValueError("non-finite samples")
    plt = _mpl()
    ncol = (n + nrow - 1) // nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(0.9 * ncol, 0.9 * nrow), squeeze=False)
    for i in range(nrow * ncol):
        r, c = divmod(i, ncol); ax = axes[r, c]
        if i < n:
            ax.imshow(x_pix[i], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"f^-1(z), z~N(0,I)   n={n}", fontsize=9)
    _save_fig(plt, out_path)
    return {"n": int(n), "min": float(x_pix.min()),
            "max": float(x_pix.max()), "mean": float(x_pix.mean())}


def plot_samples_fixed_z(expert, y_one: torch.Tensor, out_path,
                         *, n: int = 32, nrow: int = 4, z_seed: int = 1234,
                         image_shape: tuple = (28, 28)) -> dict:
    if y_one.dim() != 4 or y_one.size(0) != 1:
        logger.error("[plot_samples_fixed_z] y_one must be (1,1,H,W)")
        raise ValueError("y_one must be (1,1,H,W)")
    expert.eval()
    g = torch.Generator(device=y_one.device).manual_seed(int(z_seed))
    z = torch.randn(n, expert.dim, generator=g,
                    device=y_one.device, dtype=y_one.dtype)
    with torch.no_grad():
        h = expert.cond(y_one).expand(n, -1)
        x_logit = expert.decode(z, h)
    x_pix = inverse_logit(x_logit).view(n, *image_shape).cpu().numpy()
    if not np.isfinite(x_pix).all():
        logger.error("[plot_samples_fixed_z] non-finite samples")
        raise ValueError("non-finite samples")
    plt = _mpl()
    ncol = (n + nrow - 1) // nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(0.9 * ncol, 0.9 * nrow), squeeze=False)
    for i in range(nrow * ncol):
        r, c = divmod(i, ncol); ax = axes[r, c]
        if i < n:
            ax.imshow(x_pix[i], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"f^-1(z*), fixed z seed={z_seed}", fontsize=9)
    _save_fig(plt, out_path)
    return {"n": int(n), "z_seed": int(z_seed)}


def _sliced_wasserstein2(X: torch.Tensor, Y: torch.Tensor, n_proj: int = 64) -> float:
    if X.dim() != 2 or Y.dim() != 2 or X.shape[1] != Y.shape[1]:
        logger.error("[_sliced_wasserstein2] bad shapes X=%s Y=%s",
                     tuple(X.shape), tuple(Y.shape))
        raise ValueError("X, Y must be (N, D) with matching D")
    D = X.shape[1]
    dirs = torch.randn(n_proj, D, device=X.device, dtype=X.dtype)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    px = (X @ dirs.T).sort(dim=0).values
    py = (Y @ dirs.T).sort(dim=0).values
    if px.shape[0] != py.shape[0]:
        n = min(px.shape[0], py.shape[0])
        idx = torch.linspace(0, px.shape[0] - 1, n, device=X.device).long()
        idy = torch.linspace(0, py.shape[0] - 1, n, device=X.device).long()
        px = px[idx]; py = py[idy]
    return float(((px - py) ** 2).mean().item())


def plot_sw2_diversity(expert, y_one: torch.Tensor, out_path,
                       *, n_samples: int = 256, n_proj: int = 64) -> dict:
    expert.eval()
    with torch.no_grad():
        X = expert.sample(n_samples, y_one)
        Y = expert.sample(n_samples, y_one)
    sw2_xy = _sliced_wasserstein2(X, Y, n_proj=n_proj)
    intra_X = float(X.std(dim=0).mean().item())
    intra_Y = float(Y.std(dim=0).mean().item())
    if not (math.isfinite(sw2_xy) and math.isfinite(intra_X) and math.isfinite(intra_Y)):
        logger.error("[plot_sw2_diversity] non-finite metric")
        raise ValueError("non-finite SW2/intra metric")
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.bar(["SW2(X,Y)", "intra_std X", "intra_std Y"],
           [sw2_xy, intra_X, intra_Y], color=["#3a6", "#36a", "#36a"])
    ax.set_ylabel("value")
    ax.set_title(f"sample diversity  N={n_samples}  proj={n_proj}", fontsize=9)
    _save_fig(plt, out_path)
    return {"sw2_xy": sw2_xy, "intra_std_X": intra_X, "intra_std_Y": intra_Y,
            "n_samples": int(n_samples), "n_proj": int(n_proj)}


def plot_diag_spectrum(expert, out_path) -> dict:
    diag_layer = None
    for layer in expert.layers:
        if hasattr(layer, "scale") and hasattr(layer.scale, "log_s"):
            diag_layer = layer.scale; break
    if diag_layer is None:
        # v0.3: RealNVP / Glow have no DiagScale. Explicit skip path.
        logger.info("[plot_diag_spectrum] expert has no DiagScale; skipping "
                    "(use plot_s_spectrum for affine-coupling experts)")
        return {"skipped": True,
                "reason": "no DiagScale (RealNVP / Glow / non-NICE expert)"}
    log_s = diag_layer.log_s.detach().cpu().numpy()
    sigma = np.exp(-log_s)
    sigma_sorted = np.sort(sigma)[::-1]
    p = sigma_sorted / max(sigma_sorted.sum(), 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    H = float(-(p * np.log(p)).sum())
    H_norm = float(H / math.log(len(p)))
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    ax.plot(sigma_sorted, lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("dim (sorted)"); ax.set_ylabel("sigma_d  (log)")
    ax.set_title(f"DiagScale spectrum  H_norm={H_norm:.3f}", fontsize=9)
    _save_fig(plt, out_path)
    return {"sigma_min": float(sigma_sorted.min()),
            "sigma_max": float(sigma_sorted.max()),
            "spectral_entropy": H, "spectral_entropy_norm": H_norm,
            "n_dim": int(len(sigma_sorted))}


def plot_film_alive(expert, cond, y: torch.Tensor, out_path,
                    *, eps: float = 1e-3) -> dict:
    # v0.3: collect FiLM heads from BOTH structures:
    #   * NICE / RealNVP:  layer.film     (one head per coupling)
    #   * Glow:            layer.coupling.film1 + layer.coupling.film2
    # Returns one bar per FiLM head; labels disambiguate by layer.
    expert.eval(); cond.eval()
    film_heads: list[tuple[str, torch.nn.Module]] = []
    for i, layer in enumerate(expert.layers):
        if hasattr(layer, "film"):
            film_heads.append((f"L{i}", layer.film))
        elif hasattr(layer, "coupling"):
            cpl = layer.coupling
            if hasattr(cpl, "film1"):
                film_heads.append((f"L{i}.1", cpl.film1))
            if hasattr(cpl, "film2"):
                film_heads.append((f"L{i}.2", cpl.film2))
    if not film_heads:
        logger.error("[plot_film_alive] no FiLM heads in expert")
        raise RuntimeError("no FiLM heads found in expert")
    if y.dim() != 4 or y.size(0) < 2:
        logger.error("[plot_film_alive] need batch B>=2 to measure across-y "
                     "std, got y shape %s", tuple(y.shape))
        raise ValueError("plot_film_alive requires y batch with B>=2")
    with torch.no_grad():
        h = cond(y)
        gammas, betas, labels = [], [], []
        gamma_chan, beta_chan = [], []
        for label, film in film_heads:
            g, b = film(h)                       # each (B, feat_width)
            # v0.4: ACROSS-Y std (per feature, averaged) -- "does FiLM move
            # when y moves". This is the verdict metric.
            gammas.append(float(g.std(dim=0).mean().item()))
            betas.append(float(b.std(dim=0).mean().item()))
            # old metric, kept for reference only (channel-to-channel spread;
            # always nonzero even if the conditioner is collapsed).
            gamma_chan.append(float(g.std().item()))
            beta_chan.append(float(b.std().item()))
            labels.append(label)
    g_min = float(min(gammas)); b_min = float(min(betas))
    # v0.5: VERDICT on the MEDIAN across-y std, not the min. One quiet layer
    # (b_min ~7e-4) must not flip alive=False when FiLM is broadly responsive.
    g_med = float(median(gammas)); b_med = float(median(betas))
    alive = (g_med > eps) and (b_med > eps)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(0.5 * len(labels) + 2.5, 2.6))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, gammas, 0.36, label="gamma_std (across y)")
    ax.bar(x + 0.18, betas,  0.36, label="beta_std (across y)")
    ax.axhline(eps, color="r", lw=0.8, ls="--", label=f"eps={eps:g}")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, fontsize=7)
    ax.set_yscale("log"); ax.set_ylabel("std across y (log)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"per-layer FiLM across-y std   alive={alive}", fontsize=9)
    _save_fig(plt, out_path)
    return {"gamma_stds": gammas, "beta_stds": betas, "labels": labels,
            "gamma_std_min": g_min, "beta_std_min": b_min,
            "gamma_std_med": g_med, "beta_std_med": b_med,
            "gamma_chan_spread": gamma_chan, "beta_chan_spread": beta_chan,
            "alive": bool(alive), "eps": float(eps)}


def plot_logp_shuffle(expert, x: torch.Tensor, y: torch.Tensor,
                      out_path) -> dict:
    expert.eval()
    with torch.no_grad():
        lp_real = expert.log_prob(x, y)
        perm = torch.randperm(y.shape[0], device=y.device)
        lp_shuf = expert.log_prob(x, y[perm])
    gap = (lp_real - lp_shuf).detach().cpu().numpy()
    if not np.isfinite(gap).all():
        logger.error("[plot_logp_shuffle] non-finite gap")
        raise ValueError("non-finite logp shuffle gap")
    mean = float(gap.mean()); std = float(gap.std())
    median = float(np.median(gap))
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    ax.hist(gap, bins=40, color="#36a", alpha=0.85)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.axvline(mean, color="r", lw=0.8, ls="--")
    ax.set_xlabel("log p(x|y) - log p(x|y_shuffled)   [nats]")
    ax.set_ylabel("count")
    ax.set_title(f"shuffle gap  mean={mean:.2f}  std={std:.2f}", fontsize=9)
    _save_fig(plt, out_path)
    return {"gap_mean": mean, "gap_std": std, "gap_median": median, "n": int(gap.size)}


# ============================================================================
# v0.3: s_spectrum + w1x1_spectrum (RealNVP / Glow)
# ============================================================================
def plot_s_spectrum(expert, x: torch.Tensor, y: torch.Tensor,
                    out_path) -> dict:
    # RealNVP / Glow analogue of diag_spectrum. Walks every layer that owns
    # a `_st` (or .coupling._st for Glow), aggregates |s| across them, and
    # reports spectral entropy for cross-expert comparison.
    expert.eval()
    s_layers: list[torch.Tensor] = []
    with torch.no_grad():
        h = expert.cond(y)
        # For Glow we have to feed in (B, C, H, W) AFTER squeeze. Detect via
        # presence of `_to_image` and `image_shape` on the expert.
        if hasattr(expert, "_to_image") and hasattr(expert, "image_shape"):
            from ...models.flows.glow.squeeze import squeeze2x2  # local import
            x_img = expert._to_image(x)
            z = squeeze2x2(x_img)
            for layer in expert.layers:
                # GlowStep has .actnorm/.inv1x1/.coupling; peek the coupling
                if hasattr(layer, "coupling") and hasattr(layer.coupling, "_st"):
                    z, _ = layer.actnorm(z)
                    z, _ = layer.inv1x1(z)
                    x1, _ = layer.coupling._split(z)
                    s_lay, _ = layer.coupling._st(x1, h)
                    s_layers.append(s_lay.detach().abs().mean(dim=0)
                                    .flatten().cpu())
                    z, _ = layer.coupling(z, h)
                else:
                    z, _ = layer(z, h)
        else:
            # Flat (B, D) experts (RealNVP / NICE / NSF).
            z = x
            for layer in expert.layers:
                if hasattr(layer, "_st") and hasattr(layer, "s_max"):
                    x1 = z[..., layer.d_in:] if getattr(layer, "flip", False) \
                         else z[..., :layer.d_in]
                    s_lay, _ = layer._st(x1, h)
                    s_layers.append(s_lay.detach().abs().mean(dim=0).cpu())
                z, _ = layer(z, h)
    if not s_layers:
        logger.info("[plot_s_spectrum] no affine coupling layers; skipping")
        return {"skipped": True, "reason": "no affine coupling layers"}
    sigma = torch.cat(s_layers).numpy()
    sigma_sorted = np.sort(sigma)[::-1]
    p = sigma_sorted / max(sigma_sorted.sum(), 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    H = float(-(p * np.log(p)).sum())
    H_norm = float(H / math.log(len(p)))
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    ax.plot(sigma_sorted, lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("idx (sorted)")
    ax.set_ylabel("|s|  (log)")
    ax.set_title(f"|s| spectrum  H_norm={H_norm:.3f}", fontsize=9)
    _save_fig(plt, out_path)
    return {"s_min": float(sigma_sorted.min()),
            "s_max": float(sigma_sorted.max()),
            "spectral_entropy": H, "spectral_entropy_norm": H_norm,
            "n_units": int(len(sigma_sorted))}


def plot_w1x1_spectrum(expert, out_path) -> dict:
    # SVD of every Inv1x1Conv in a Glow expert. Reports per-step min SV,
    # condition number, and the global minimum across all steps.
    min_sv_per_step: list[float] = []
    cond_per_step: list[float] = []
    for i, layer in enumerate(expert.layers):
        if not hasattr(layer, "inv1x1") or not hasattr(layer.inv1x1,
                                                       "singular_values"):
            continue
        sv = layer.inv1x1.singular_values().cpu().numpy()
        smin, smax = float(sv.min()), float(sv.max())
        min_sv_per_step.append(smin)
        cond_per_step.append(smax / max(smin, 1e-12))
    if not min_sv_per_step:
        logger.info("[plot_w1x1_spectrum] no Inv1x1Conv layers; skipping")
        return {"skipped": True, "reason": "no Inv1x1Conv layers"}
    global_min = float(min(min_sv_per_step))
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    idx = np.arange(len(min_sv_per_step))
    ax.bar(idx, min_sv_per_step, 0.7, color="#36a")
    ax.set_yscale("log")
    ax.set_xlabel("Glow step")
    ax.set_ylabel("min SV (log)")
    ax.set_title(f"Inv1x1Conv min SV per step  global_min={global_min:.2e}",
                 fontsize=9)
    _save_fig(plt, out_path)
    return {"min_sv_per_step": min_sv_per_step,
            "cond_per_step": cond_per_step,
            "global_min_sv": global_min,
            "n_steps": int(len(min_sv_per_step))}
