# =============================================================================
# COND-GATE v0.3 -- common.cond_viz
# Purpose: diagnostic plots for the 9 COND-GATE checks. Writes PNG to
#          step_X/plots/cond_gate/epoch_<N>.png or similar.
# CONVENTION: matplotlib imported lazily so that a missing matplotlib does not
#             break module import. Any IO error is logged + re-raised.
# Changelog (v0.2 -> v0.3):
#   * Added plot_film_per_layer (per-layer γ/β bar chart).
#   * Added plot_gate_collapse (Neff / entropy traj + argmax histogram).
#   * plot_grad_traj now supports multi-layer FiLM grad history.
# =============================================================================
from __future__ import annotations
import logging
import traceback
from pathlib import Path
logger = logging.getLogger(__name__)
__version__ = "0.3"
__abbr__ = "COND-GATE"

import numpy as np


def _mpl():
    # Lazy import so scaffolder-only installs don't need matplotlib.
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


def plot_grad_traj(cond_grad_history, film_grad_history, out_path):
    # cond_grad_history: list[float] length = epochs
    # film_grad_history: list[list[float]] shape (epochs, L) OR list[float]
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cond_grad_history, label="conditioner", linewidth=2)
    fh = np.asarray(film_grad_history)
    if fh.ndim == 2:
        for i in range(fh.shape[1]):
            ax.plot(fh[:, i], label=f"FiLM[{i}]", alpha=0.6)
    elif fh.size > 0:
        ax.plot(fh, label="FiLM", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("grad norm")
    ax.legend(fontsize=8)
    ax.set_title("gradient flow (must stay >> 0)")
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
    # nan_inf_history: list[dict] per epoch with keys like h_nan/h_inf/gamma_nan/...
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
    vals = np.clip(vals, 1e-20, None)   # log-safe
    ax.plot(vals, marker="o")
    ax.axhline(tol, color="red", linestyle="--", label=f"tol={tol:.0e}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("max|delta h| (twin runs)")
    ax.set_title("(y, seed) determinism")
    ax.legend()
    _save(plt, out_path)


def plot_film_per_layer(per_layer_stats, out_path):
    # per_layer_stats: list of {layer_index, gamma:{mean,std}, beta:{mean,std}}
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
