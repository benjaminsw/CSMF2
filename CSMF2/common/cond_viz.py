# =============================================================================
# COND-GATE v0.4 -- common.cond_viz
# Purpose: diagnostic plots for the 11 COND-GATE checks. PNG saved to
#          step_X/plots/cond_gate/epoch_<N>.png (or similar).
# CONVENTION: matplotlib imported lazily so a missing matplotlib does not
#             break module import. Any IO error is logged + re-raised.
# Changelog (v0.3 -> v0.4):
#   * Added plot_h_st_response -- scatter of ||Δs||, ||Δt||, ||Δlogp|| vs
#     ||Δy|| with fitted linear slope annotated (matches numeric check #10).
#   * Added plot_null_control_overlay -- overlay of real-model NLL vs
#     h-ablated-copy NLL, with relative-gap annotation (matches check #11).
#   * All previous 8 plots unchanged.
# =============================================================================
from __future__ import annotations
import logging
import traceback
from pathlib import Path
logger = logging.getLogger(__name__)
__version__ = "0.4"
__abbr__ = "COND-GATE"

import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(plt, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
    except OSError:
        logger.error("[cond_viz] save failed %s\n%s", out_path, traceback.format_exc())
        raise


def _fit_slope(xs, ys):
    # Closed-form OLS slope (same formula as cond_diagnostics._linfit_slope,
    # duplicated here so cond_viz has no cross-module dependency).
    x = np.asarray(xs, dtype=float).ravel()
    y = np.asarray(ys, dtype=float).ravel()
    if x.size < 2 or y.size != x.size:
        return float("nan")
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom < 1e-30:
        return float("nan")
    return float(((x - xm) * (y - ym)).sum() / denom)


# ===== Original 8 plots ======================================================
def plot_h_hist(h, gamma, beta, out_path):
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, arr, title in zip(axes, [h, gamma, beta], ["h", "gamma", "beta"]):
        a = np.asarray(arr).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            ax.set_title(title + " (no finite)")
            continue
        ax.hist(a, bins=50)
        ax.set_title(f"{title}  mean={a.mean():.3f} std={a.std():.3f}")
    _save(plt, out_path)


def plot_h_diversity(pairwise_matrix, out_path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(np.asarray(pairwise_matrix), cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_title("pairwise ||h_i - h_j||")
    _save(plt, out_path)


def plot_grad_traj(cond_grad_history, film_grad_history, out_path, min_grad=None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cond_grad_history, label="conditioner", linewidth=2)
    fh = np.asarray(film_grad_history)
    if fh.ndim == 2:
        for i in range(fh.shape[1]):
            ax.plot(fh[:, i], label=f"FiLM[{i}]", alpha=0.6)
    elif fh.size > 0:
        ax.plot(fh, label="FiLM", alpha=0.6)
    if min_grad is not None:
        ax.axhline(min_grad, color="red", linestyle="--",
                   label=f"min_grad={min_grad:.0e}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("grad norm")
    ax.legend(fontsize=8)
    ax.set_title("gradient flow (must stay above min_grad)")
    _save(plt, out_path)


def plot_logp_shuffle(logp_real, logp_shuffled, out_path):
    plt = _mpl()
    r = np.asarray(logp_real).ravel()
    s = np.asarray(logp_shuffled).ravel()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([0, 1], [r.mean(), s.mean()], yerr=[r.std(), s.std()], color=["C0", "C3"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["logp(x|h)", "logp(x|shuffle h)"])
    ax.set_ylabel("mean logp")
    ax.set_title("conditioning effect on logp (bars must differ)")
    _save(plt, out_path)


def plot_nan_inf_traj(nan_inf_history, out_path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 3))
    if not nan_inf_history:
        logger.error("[plot_nan_inf_traj] empty history")
        ax.set_title("no data")
        _save(plt, out_path)
        return
    keys = sorted({k for d in nan_inf_history for k in d.keys()})
    for k in keys:
        ax.plot([d.get(k, 0) for d in nan_inf_history], label=k)
    ax.set_xlabel("epoch")
    ax.set_ylabel("count")
    ax.set_title("NaN/Inf per epoch (must stay 0)")
    ax.legend(fontsize=8)
    _save(plt, out_path)


def plot_determinism_traj(max_delta_h_history, tol, out_path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 3))
    vals = np.asarray(max_delta_h_history, dtype=float)
    vals = np.clip(vals, 1e-20, None)
    ax.plot(vals, marker="o")
    ax.axhline(tol, color="red", linestyle="--", label=f"tol={tol:.0e}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("max|delta h| (twin runs)")
    ax.set_title("(y, seed) determinism")
    ax.legend()
    _save(plt, out_path)


def plot_film_per_layer(per_layer_stats, out_path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 4))
    L = len(per_layer_stats)
    if L == 0:
        ax.set_title("no FiLM layers")
        _save(plt, out_path)
        return
    xs = np.arange(L)
    g_mean = [s["gamma"]["mean"] for s in per_layer_stats]
    g_std  = [s["gamma"]["std"]  for s in per_layer_stats]
    b_mean = [s["beta"]["mean"]  for s in per_layer_stats]
    b_std  = [s["beta"]["std"]   for s in per_layer_stats]
    ax.bar(xs - 0.2, g_mean, 0.4, yerr=g_std, label="gamma", capsize=3)
    ax.bar(xs + 0.2, b_mean, 0.4, yerr=b_std, label="beta",  capsize=3)
    ax.set_xlabel("FiLM layer index")
    ax.set_ylabel("mean +/- std")
    ax.set_xticks(xs)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend()
    ax.set_title("per-layer FiLM stats (flat bar = dead layer)")
    _save(plt, out_path)


def plot_gate_collapse(neff_history, entropy_history, argmax_hist, out_path):
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(neff_history, marker="o")
    axes[0].set_title("Neff per epoch")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("Neff")
    axes[1].plot(entropy_history, marker="o", color="C1")
    axes[1].set_title("gate entropy per epoch")
    axes[1].set_xlabel("epoch")
    ah = np.asarray(argmax_hist).ravel()
    axes[2].bar(np.arange(ah.size), ah)
    axes[2].set_title("argmax-expert histogram")
    axes[2].set_xlabel("expert k"); axes[2].set_ylabel("count")
    _save(plt, out_path)


# ===== NEW in v0.4 ===========================================================
def plot_h_st_response(dy_norms, ds_norms=None, dt_norms=None,
                       dlogp_norms=None, out_path=None,
                       min_slope=1e-3, require_s=True):
    # Scatter ||Δy|| vs ||Δs||, ||Δt||, ||Δlogp|| with fitted linear slope
    # annotated on each sub-plot. Slopes below min_slope are flagged red.
    plt = _mpl()
    series = []
    if ds_norms is not None and require_s:
        series.append(("Δs", ds_norms, "C0"))
    if dt_norms is not None:
        series.append(("Δt", dt_norms, "C2"))
    if dlogp_norms is not None:
        series.append(("Δlogp", dlogp_norms, "C3"))

    if not series:
        logger.error("[plot_h_st_response] no series provided")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_title("no data")
        _save(plt, out_path)
        return

    dy = np.asarray(dy_norms, dtype=float).ravel()
    fig, axes = plt.subplots(1, len(series),
                             figsize=(4.2 * len(series), 3.5), squeeze=False)
    for ax, (name, ys, col) in zip(axes[0], series):
        y = np.asarray(ys, dtype=float).ravel()
        slope = _fit_slope(dy, y)
        ax.scatter(dy, y, s=18, color=col, alpha=0.7)
        xs_line = np.linspace(dy.min(), dy.max(), 50)
        # intercept via y = slope*x + b ; b = mean(y) - slope*mean(x)
        b = y.mean() - slope * dy.mean() if slope == slope else 0.0
        ax.plot(xs_line, slope * xs_line + b, "k--", linewidth=1)
        bad = not (slope == slope and slope > min_slope)
        tag_col = "red" if bad else "black"
        tag = f"slope={slope:.3g}" + (f"  < {min_slope:.0e}  FAIL" if bad else "  OK")
        ax.set_title(f"||Δy|| vs ||{name}||\n{tag}", color=tag_col, fontsize=10)
        ax.set_xlabel("||Δy||"); ax.set_ylabel(f"||{name}||")
    _save(plt, out_path)


def plot_null_control_overlay(real_nll_history, ablated_nll_history,
                              out_path, min_relative_gap=0.10):
    # Two NLL curves + shaded gap + annotated RELATIVE gap (ablated-real)/|real|.
    plt = _mpl()
    r = np.asarray(real_nll_history, dtype=float).ravel()
    a = np.asarray(ablated_nll_history, dtype=float).ravel()
    n = min(r.size, a.size)
    if n == 0:
        logger.error("[plot_null_control_overlay] empty history")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_title("no data")
        _save(plt, out_path)
        return
    r, a = r[:n], a[:n]
    xs = np.arange(n)

    r_final = float(r[-min(5, n):].mean())
    a_final = float(a[-min(5, n):].mean())
    rel_gap = (a_final - r_final) / max(abs(r_final), 1e-12)
    bad = rel_gap < min_relative_gap

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, r, label="real (h used)", color="C2", linewidth=2)
    ax.plot(xs, a, label="h-ablated (null control)", color="C3", linewidth=2)
    ax.fill_between(xs, r, a, where=(a >= r), alpha=0.2, color="C3",
                    label="gap (ablated - real)")
    tag_col = "red" if bad else "black"
    tag = (f"relative gap (last-5-mean) = {rel_gap:.3f}"
           + (f"  < {min_relative_gap:.2f}  FAIL" if bad
              else f"  >= {min_relative_gap:.2f}  OK"))
    ax.set_title("null-control: real vs h-ablated NLL\n" + tag, color=tag_col,
                 fontsize=10)
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL (lower = better)")
    ax.legend(fontsize=8, loc="best")
    _save(plt, out_path)
