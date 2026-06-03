# =============================================================================
# STEP-1_1_1_1 v0.1.1 -- experiments.step_1_1_1_1.run
# Purpose: end-to-end MAP refinement on ANY trained conditional flow ckpt.
#   1. Resolve ckpt directory (primary: --ckpt-dir; helper: --best-params)
#   2. Load expert + cond from ckpt (architecture inferred from ckpt's cfg)
#   3. Iterate over n_test val images in batches, call refine()
#   4. Aggregate metrics (residual / prior / z_norm / PSNR before+after /
#      fwd_rel before+after); save metrics.json
#   5. Save 6 plots:
#        plots/reconstruction_panel.png   (4-row headline)
#        plots/loss_trajectory.png        (residual + prior + total over t)
#        plots/z_norm_trajectory.png      (mean ||z|| over t)
#        plots/residual_heatmap.png       (per-pixel A(x_hat)-y, before vs after)
#        plots/xhat_filmstrip.png         (x_hat at t=0,steps/4,steps/2,steps-1)
#        plots/psnr_scatter.png           (psnr_before vs psnr_after; y=x line)
# CONVENTION: every failure -> logger.error + raise. No fallback / placeholder.
# Changelog (v0.1 -> v0.1.1):
#   * Independent-experiment refactor. --ckpt-dir is now the primary
#     interface. --best-params + --best-params-expert + --train-results-root
#     is an OPTIONAL convenience helper, not a default.
#   * Removed cfg.expert -- expert architecture is read from the loaded
#     ckpt's report.json, not declared by the user.
#   * No default paths to step_1_1_1. The MAP experiment now works equally
#     well against checkpoints from step_1_1, step_1_1_1, future step_1_2
#     mixtures, or any other conditional-flow training step.
# Changelog (NEW in v0.1):
#   * Introduced.
# Update summary:
#   v0.1.1 makes step_1_1_1_1 a generic inference-time experiment. It
#   answers ONLY: "given any trained conditional flow, does MAP refinement
#   improve reconstruction?" -- without coupling to any specific training
#   regime. Reusable as-is for future training steps.
# =============================================================================
from __future__ import annotations
import argparse
import glob
import json
import logging
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import MAPCfg
from .map_refine import refine, degrade_diff
from ...data.degrade import (MNISTDegraded, dequantize_logit, inverse_logit,
                             blur, downsample)
from ...models.conditioner import Conditioner
from ...models.experts import build_expert

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
logger = logging.getLogger("CSMF2.step_1_1_1_1.run")
__version__ = "0.1.1"
__abbr__ = "STEP-1_1_1_1"


# ---------- locate the step_1_1_1 checkpoint --------------------------------


def _find_ckpt_dir(*, base: str, expert: str, noise_sigma: float,
                   latent_moment_lambda: float) -> Path:
    """Search step_1_1_1 results for a run matching (expert, noise, lam).
    Picks the most recent if multiple match. Raises if none match.
    """
    base_p = Path(base)
    if not base_p.exists():
        logger.error("[find_ckpt_dir] base dir not found: %s", base_p)
        raise FileNotFoundError(f"step_1_1_1 results dir not found: {base_p}")
    candidates = []
    for d in sorted(base_p.glob(f"{expert}_s*_n*_seed*_*"),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        rep_path = d / "report.json"
        if not rep_path.exists():
            continue
        try:
            rep = json.loads(rep_path.read_text())
        except json.JSONDecodeError:
            logger.warning("[find_ckpt_dir] could not parse %s, skipping",
                           rep_path)
            continue
        cfg = rep.get("cfg", {})
        if (cfg.get("expert") == expert
                and abs(cfg.get("noise_sigma", -1) - noise_sigma) < 1e-9
                and abs(cfg.get("latent_moment_lambda", -1)
                        - latent_moment_lambda) < 1e-9):
            candidates.append(d)
    if not candidates:
        logger.error("[find_ckpt_dir] no run matched expert=%s noise=%s "
                     "lambda=%s under %s",
                     expert, noise_sigma, latent_moment_lambda, base_p)
        raise FileNotFoundError(
            f"no step_1_1_1 run matched expert={expert} "
            f"noise={noise_sigma} lambda={latent_moment_lambda}")
    chosen = candidates[0]
    if len(candidates) > 1:
        logger.warning("[find_ckpt_dir] %d candidates matched; picked most "
                       "recent: %s", len(candidates), chosen)
    return chosen


def _resolve_ckpt_dir(cfg: MAPCfg) -> tuple[Path, dict]:
    """Returns (ckpt_dir, train_cfg). train_cfg is the cfg from the
    step_1_1_1 run.
    """
    if cfg.ckpt_dir is not None:
        d = Path(cfg.ckpt_dir)
        rep_path = d / "report.json"
        if not rep_path.exists():
            logger.error("[resolve_ckpt] report.json missing in %s", d)
            raise FileNotFoundError(f"report.json missing in {d}")
        train_cfg = json.loads(rep_path.read_text()).get("cfg", {})
        return d, train_cfg
    # best_params.json path -- convenience helper
    bp_path = Path(cfg.best_params)
    if not bp_path.exists():
        logger.error("[resolve_ckpt] best_params.json not found: %s", bp_path)
        raise FileNotFoundError(f"best_params.json not found: {bp_path}")
    bp = json.loads(bp_path.read_text())
    winners = bp.get("winners", {})
    bp_expert = cfg.best_params_expert
    if bp_expert not in winners:
        logger.error("[resolve_ckpt] expert=%r missing from winners keys=%s",
                     bp_expert, list(winners.keys()))
        raise KeyError(
            f"expert={bp_expert!r} missing from best_params winners")
    w = winners[bp_expert]
    if w.get("latent_moment_lambda") is None:
        logger.error("[resolve_ckpt] %s: latent_moment_lambda is null in "
                     "best_params (notes=%s)", bp_expert, w.get("notes"))
        raise ValueError(
            f"{bp_expert} has null latent_moment_lambda in best_params -- "
            f"no winner has been recorded yet. Either complete the sweep "
            f"or use --ckpt-dir to point at a checkpoint directly.")
    train_base = cfg.train_results_root
    d = _find_ckpt_dir(base=train_base, expert=bp_expert,
                       noise_sigma=w["noise_sigma"],
                       latent_moment_lambda=w["latent_moment_lambda"])
    rep_path = d / "report.json"
    train_cfg = json.loads(rep_path.read_text()).get("cfg", {})
    return d, train_cfg


# ---------- PSNR helper ------------------------------------------------------


def _psnr_per_image(x_hat: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """Compute per-image PSNR in dB. Inputs in [0,1], shape (B,1,28,28)."""
    if x_hat.shape != x_true.shape:
        logger.error("[psnr] shape mismatch: %s vs %s",
                     tuple(x_hat.shape), tuple(x_true.shape))
        raise ValueError("psnr shape mismatch")
    mse = ((x_hat - x_true) ** 2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


# ---------- plots ------------------------------------------------------------


def _plot_reconstruction_panel(y, x_true, x_initial, x_star, save_path,
                                title, n_show=8):
    n = min(n_show, y.size(0))
    fig, axes = plt.subplots(4, n, figsize=(1.4 * n, 5.6), dpi=120)
    if n == 1:
        axes = axes.reshape(4, 1)
    rows = [y, x_true, x_initial, x_star]
    labels = ["y (degraded)", "x (true)",
              "x_initial (z0 decode)", "x_star (after MAP)"]
    for r in range(4):
        for c in range(n):
            ax = axes[r, c]
            img = rows[r][c].squeeze(0).detach().cpu().clamp(0, 1).numpy()
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(labels[r], fontsize=8, rotation=0,
                              ha="right", va="center")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)


def _plot_loss_trajectory(loss_curves, save_path, title):
    # loss_curves: list of dict per-batch; each has 'residual','prior','loss'
    if not loss_curves:
        return
    res = np.mean([c["residual"] for c in loss_curves], axis=0)
    pri = np.mean([c["prior"]    for c in loss_curves], axis=0)
    tot = np.mean([c["loss"]     for c in loss_curves], axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    ax.plot(res, label="residual ||A(x_hat)-y||^2", color="#1f77b4", lw=2)
    ax.plot(pri, label="prior ||z||^2/D",           color="#d62728", lw=2)
    ax.plot(tot, label="total loss",                color="#2ca02c", lw=2,
            linestyle="--")
    ax.set_xlabel("MAP step"); ax.set_ylabel("value")
    ax.set_yscale("log"); ax.set_title(title)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)


def _plot_z_norm_trajectory(loss_curves, save_path, title, D_dim):
    if not loss_curves:
        return
    zn = np.mean([c["z_norm"] for c in loss_curves], axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 3.5), dpi=120)
    ax.plot(zn, color="#9467bd", lw=2, label="mean ||z||")
    ax.axhline(math.sqrt(D_dim), color="gray", linestyle=":",
               label=f"sqrt(D) = {math.sqrt(D_dim):.1f}")
    ax.set_xlabel("MAP step"); ax.set_ylabel("||z||")
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)


def _plot_residual_heatmap(y, x_initial, x_star, A_fn, save_path, title,
                            n_show=4):
    n = min(n_show, y.size(0))
    with torch.no_grad():
        r_before = (A_fn(x_initial[:n]) - y[:n]) ** 2
        r_after  = (A_fn(x_star[:n])    - y[:n]) ** 2
    vmax = float(max(r_before.max().item(), r_after.max().item(), 1e-12))
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.5), dpi=120)
    if n == 1:
        axes = axes.reshape(2, 1)
    for c in range(n):
        for r, src, lab in [(0, r_before, "before"), (1, r_after, "after")]:
            ax = axes[r, c]
            img = src[c].squeeze(0).cpu().numpy()
            im = ax.imshow(img, cmap="hot", vmin=0.0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(lab, fontsize=9, rotation=0, ha="right",
                              va="center")
    fig.suptitle(title + f"   (max sq err = {vmax:.4f})", fontsize=10)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)


def _plot_xhat_filmstrip(snapshots_per_batch, save_path, title, n_show=4):
    # snapshots_per_batch: list of lists. snapshots_per_batch[0] is the first
    # batch's list of {"step": t, "img": (B,1,28,28)}.
    if not snapshots_per_batch or not snapshots_per_batch[0]:
        return
    snaps = snapshots_per_batch[0]    # use first batch
    n_t = len(snaps)
    n   = min(n_show, snaps[0]["img"].size(0))
    fig, axes = plt.subplots(n, n_t, figsize=(1.4 * n_t, 1.4 * n), dpi=120)
    if n == 1:
        axes = axes.reshape(1, n_t)
    if n_t == 1:
        axes = axes.reshape(n, 1)
    for r in range(n):
        for c, snap in enumerate(snaps):
            ax = axes[r, c]
            img = snap["img"][r].squeeze(0).cpu().clamp(0, 1).numpy()
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={snap['step']}", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)


def _plot_psnr_scatter(psnr_before, psnr_after, save_path, title):
    pb = np.asarray(psnr_before); pa = np.asarray(psnr_after)
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=120)
    ax.scatter(pb, pa, s=20, alpha=0.6, color="#1f77b4")
    lo = float(min(pb.min(), pa.min())) - 1.0
    hi = float(max(pb.max(), pa.max())) + 1.0
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--",
            label="y = x (no change)")
    pct_up = float((pa > pb).mean() * 100.0)
    ax.set_xlabel("PSNR before MAP (dB)")
    ax.set_ylabel("PSNR after MAP (dB)")
    ax.set_title(f"{title}\n{pct_up:.1f}% improved")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)


# ---------- main run ---------------------------------------------------------


def run(cfg: MAPCfg) -> dict:
    t0 = time.time()
    out_dir = Path(cfg.out_root) / cfg.run_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = out_dir / "plots"; plots.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "run.log"); fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s :: %(message)s"))
    logging.getLogger("CSMF2").addHandler(fh)
    logger.info("MAP-refine run | tag=%s | cfg=%s", cfg.run_tag(), cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # 1. locate the trained checkpoint
    ckpt_dir, tcfg = _resolve_ckpt_dir(cfg)
    logger.info("[ckpt] loading from %s", ckpt_dir)
    logger.info("[ckpt] train cfg: expert=%s scale=%s noise=%s lam=%s "
                "use_v2=%s", tcfg.get("expert"), tcfg.get("scale"),
                tcfg.get("noise_sigma"), tcfg.get("latent_moment_lambda"),
                tcfg.get("use_v2_conditioner"))
    # v0.1.1: expert architecture is whatever the ckpt was trained for.
    # No user-side cfg.expert -- we read it from the ckpt's report.
    expert_name = tcfg.get("expert")
    if expert_name not in ("nice", "realnvp", "nsf", "glow"):
        logger.error("[ckpt] unknown expert in ckpt's cfg: %r", expert_name)
        raise ValueError(
            f"unknown expert in ckpt's cfg: {expert_name!r}; expected one "
            f"of {{nice, realnvp, nsf, glow}}")
    # When using --best-params + --best-params-expert, sanity-check that the
    # resolved ckpt matches what the user asked for.
    if cfg.best_params_expert is not None \
            and expert_name != cfg.best_params_expert:
        logger.error("[ckpt] best_params_expert=%r resolved to ckpt for %r",
                     cfg.best_params_expert, expert_name)
        raise ValueError(
            f"best_params_expert={cfg.best_params_expert!r} but resolved "
            f"ckpt was trained for {expert_name!r}")

    # 2. build cond + expert with the SAME architecture as training
    cond_kwargs = dict(width=tcfg.get("cond_width", 128),
                       h_dim=tcfg.get("h_dim", 256),
                       use_v2=bool(tcfg.get("use_v2_conditioner", True)))
    if tcfg.get("cond_y_residual_alpha_init", 0.0) > 0.0:
        cond_kwargs["y_residual_alpha_init"] = tcfg["cond_y_residual_alpha_init"]
        cond_kwargs["y_input_size"] = (28 // tcfg["scale"]) * (28 // tcfg["scale"])
    cond = Conditioner(**cond_kwargs).to(device)
    film_kwargs = {}
    if expert_name in ("nice", "realnvp", "glow"):
        film_kwargs = dict(film_hidden=tcfg.get("film_hidden", 128),
                           film_depth=tcfg.get("film_depth", 2),
                           film_use_gelu=bool(tcfg.get("film_use_gelu", True)))
    extra_kwargs: dict = {}
    if expert_name == "realnvp":
        # build_expert wants `n_layers`, not `n_couplings`. `s_max` is not a
        # build_expert kwarg (it's a CondRealNVP-internal constant).
        extra_kwargs = dict(n_layers=tcfg.get("realnvp_n_couplings", 6))
    elif expert_name == "glow":
        extra_kwargs = dict(
            n_layers=tcfg.get("glow_n_steps", 8),
            s_max=tcfg.get("glow_s_max", 2.0),
            image_shape=(tcfg.get("glow_image_c", 1),
                         tcfg.get("glow_image_h", 28),
                         tcfg.get("glow_image_w", 28)),
            inv1x1_seed_base=tcfg.get("seed", 0),
            film_gain_init=tcfg.get("glow_film_gain_init", 0.3))
    hidden_for_build = (tcfg.get("glow_coupling_hidden", 256)
                        if expert_name == "glow"
                        else tcfg.get("flow_hidden", 256))
    expert = build_expert(expert_name, dim=tcfg.get("dim", 784),
                          h_dim=tcfg.get("h_dim", 256),
                          conditioner=cond, hidden=hidden_for_build,
                          use_film=bool(tcfg.get("use_film", True)),
                          **film_kwargs, **extra_kwargs).to(device)

    # 3. load state_dicts. strict=True -- a mismatch means architecture
    # divergence between training and this run, which is a real bug.
    ckpt = torch.load(ckpt_dir / "ckpt.pt", map_location=device,
                      weights_only=False)
    if "expert" not in ckpt or "cond" not in ckpt:
        logger.error("[ckpt] missing 'expert' or 'cond' in keys=%s",
                     list(ckpt.keys()))
        raise KeyError("checkpoint missing 'expert' or 'cond'")
    expert.load_state_dict(ckpt["expert"], strict=True)
    cond.load_state_dict(ckpt["cond"], strict=True)
    expert.eval(); cond.eval()
    for p in expert.parameters(): p.requires_grad_(False)
    for p in cond.parameters():   p.requires_grad_(False)
    logger.info("[ckpt] loaded; expert + cond frozen")

    # 4. test data (the SEALED test split, used here for inference benchmarking)
    test_ds = MNISTDegraded(root=tcfg.get("data_root", "./mnist_data"),
                            split="test", scale=tcfg["scale"],
                            sigma=tcfg.get("blur_sigma", 1.0),
                            noise_sigma=tcfg["noise_sigma"])
    if cfg.n_test < len(test_ds):
        idx = list(range(cfg.n_test))
        test_ds = Subset(test_ds, idx)
    loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # 5. forward operator A: same blur+downsample as training, NO noise,
    # NO clamp. (Noise was random per-call; MAP needs deterministic A.)
    sigma_blur = tcfg.get("blur_sigma", 1.0)
    scale      = tcfg["scale"]
    def A_fn(x_img):
        return degrade_diff(x_img, sigma=sigma_blur, scale=scale,
                            blur_fn=blur, downsample_fn=downsample)

    # 6. iterate
    all_psnr_before, all_psnr_after = [], []
    all_fwd_rel_before, all_fwd_rel_after = [], []
    all_loss_curves = []         # one entry per batch (averaged inside refine)
    first_panel_done = False
    n_seen = 0
    snapshots_for_filmstrip = []   # first batch only

    for bi, (x_img, y_img) in enumerate(loader):
        x_img = x_img.to(device); y_img = y_img.to(device)

        # baseline: random z0, decode through trained flow (no MAP).
        # This is x_initial; how good is the model out of the box on this y?
        B = y_img.size(0)
        with torch.no_grad():
            h = cond(y_img)
            z0 = torch.randn(B, expert.dim, device=device, generator=gen,
                             dtype=h.dtype)
            x_init_logit = expert.decode(z0, h)
            x_initial = inverse_logit(x_init_logit).view(B, 1, 28, 28).clamp(
                0.0, 1.0)

        # MAP refinement (random init = realistic inference)
        x_star, z_star, log_batch = refine(
            expert=expert, cond=cond, y=y_img, A_fn=A_fn,
            inverse_logit=inverse_logit, dequantize_logit=dequantize_logit,
            steps=cfg.steps, lr=cfg.lr, lambda_prior=cfg.lambda_prior,
            init=cfg.init, x_true=x_img if cfg.init == "encoded" else None,
            device=device, gen=gen)

        # PSNR (using x_true since we have it)
        psnr_before = _psnr_per_image(x_initial, x_img).detach().cpu().numpy()
        psnr_after  = _psnr_per_image(x_star,    x_img).detach().cpu().numpy()
        all_psnr_before.extend(psnr_before.tolist())
        all_psnr_after.extend(psnr_after.tolist())

        # fwd_rel before/after = mean( ||A(x_hat) - y|| / ||y|| ) per image
        with torch.no_grad():
            num_b = (A_fn(x_initial) - y_img).flatten(1).norm(dim=1)
            den   = y_img.flatten(1).norm(dim=1).clamp_min(1e-12)
            num_a = (A_fn(x_star)    - y_img).flatten(1).norm(dim=1)
            fwd_rel_before = (num_b / den).cpu().numpy()
            fwd_rel_after  = (num_a / den).cpu().numpy()
        all_fwd_rel_before.extend(fwd_rel_before.tolist())
        all_fwd_rel_after.extend(fwd_rel_after.tolist())

        all_loss_curves.append({
            "residual": log_batch["residual"],
            "prior":    log_batch["prior"],
            "loss":     log_batch["loss"],
            "z_norm":   log_batch["z_norm"],
        })
        if not first_panel_done:
            snapshots_for_filmstrip.append(log_batch["x_snapshots"])
            # 1st batch -> plots
            _plot_reconstruction_panel(
                y_img, x_img, x_initial, x_star,
                plots / "reconstruction_panel.png",
                title=f"{expert_name}  init={cfg.init}  steps={cfg.steps}  "
                      f"lp={cfg.lambda_prior}")
            _plot_residual_heatmap(
                y_img, x_initial, x_star, A_fn,
                plots / "residual_heatmap.png",
                title=f"Residual heatmap  {expert_name}")
            _plot_xhat_filmstrip(
                snapshots_for_filmstrip,
                plots / "xhat_filmstrip.png",
                title=f"x_hat evolution  {expert_name}")
            first_panel_done = True

        n_seen += B
        logger.info("[batch %d] B=%d  psnr_before_mean=%.2f  "
                    "psnr_after_mean=%.2f  fwd_rel_before=%.3f  "
                    "fwd_rel_after=%.3f", bi, B,
                    float(np.mean(psnr_before)), float(np.mean(psnr_after)),
                    float(np.mean(fwd_rel_before)),
                    float(np.mean(fwd_rel_after)))

    # 7. aggregate
    psnr_before_arr = np.asarray(all_psnr_before)
    psnr_after_arr  = np.asarray(all_psnr_after)
    fwd_rel_before_arr = np.asarray(all_fwd_rel_before)
    fwd_rel_after_arr  = np.asarray(all_fwd_rel_after)

    aggregate = {
        "fwd_rel_before_mean":  float(fwd_rel_before_arr.mean()),
        "fwd_rel_after_mean":   float(fwd_rel_after_arr.mean()),
        "psnr_before_mean":     float(psnr_before_arr.mean()),
        "psnr_after_mean":      float(psnr_after_arr.mean()),
        "psnr_before_median":   float(np.median(psnr_before_arr)),
        "psnr_after_median":    float(np.median(psnr_after_arr)),
        "psnr_improved_pct":    float((psnr_after_arr > psnr_before_arr).mean()),
        "residual_final_mean":  float(np.mean(
            [c["residual"][-1] for c in all_loss_curves])),
        "residual_initial_mean": float(np.mean(
            [c["residual"][0]  for c in all_loss_curves])),
        "z_norm_final_mean":    float(np.mean(
            [c["z_norm"][-1] for c in all_loss_curves])),
        "n_images":             int(n_seen),
        "n_nonfinite_batches":  0,
        "wall_clock_s":         time.time() - t0,
    }
    logger.info("[aggregate] fwd_rel: %.3f -> %.3f   psnr: %.2f -> %.2f   "
                "improved %% = %.1f",
                aggregate["fwd_rel_before_mean"],
                aggregate["fwd_rel_after_mean"],
                aggregate["psnr_before_mean"],
                aggregate["psnr_after_mean"],
                aggregate["psnr_improved_pct"] * 100.0)

    # 8. remaining plots (loss + z_norm + psnr scatter)
    _plot_loss_trajectory(
        all_loss_curves, plots / "loss_trajectory.png",
        title=f"Loss components vs MAP step ({expert_name})")
    _plot_z_norm_trajectory(
        all_loss_curves, plots / "z_norm_trajectory.png",
        title=f"||z|| vs MAP step ({expert_name})",
        D_dim=tcfg.get("dim", 784))
    _plot_psnr_scatter(
        psnr_before_arr, psnr_after_arr,
        plots / "psnr_scatter.png",
        title=f"PSNR before vs after MAP  ({expert_name})")

    # 9. metrics.json
    metrics = {
        "version":    __version__,
        "abbr":       __abbr__,
        "cfg":        cfg.__dict__,
        "train_cfg":  tcfg,
        "ckpt_dir":   str(ckpt_dir),
        "aggregate":  aggregate,
        "psnr_before": all_psnr_before,
        "psnr_after":  all_psnr_after,
        "fwd_rel_before": all_fwd_rel_before,
        "fwd_rel_after":  all_fwd_rel_after,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("MAP-refine run DONE  out=%s", out_dir)
    return metrics


def _parse_args() -> MAPCfg:
    p = argparse.ArgumentParser(
        description="MAP refinement on any trained conditional flow ckpt. "
                    "Supply --ckpt-dir (primary) OR "
                    "--best-params + --best-params-expert + --train-results-root "
                    "(convenience helper).")
    # Primary interface
    p.add_argument("--ckpt-dir", type=str, default=None,
                   help="[PRIMARY] path to any directory containing ckpt.pt + "
                        "report.json. The experiment is agnostic to which "
                        "training step produced this checkpoint.")
    # Convenience helper -- all three required to use it
    p.add_argument("--best-params", type=str, default=None,
                   help="[HELPER] path to a best_params.json with 'winners' "
                        "entries. Used together with --best-params-expert "
                        "and --train-results-root to resolve a ckpt-dir.")
    p.add_argument("--best-params-expert",
                   choices=("nice", "realnvp", "nsf", "glow"),
                   default=None,
                   help="[HELPER] which winner key to look up in "
                        "best_params.json.")
    p.add_argument("--train-results-root", type=str, default=None,
                   help="[HELPER] root directory where the resolved ckpt "
                        "lives on disk (e.g. some_step/results).")
    # MAP optimisation knobs
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lambda-prior", type=float, default=1e-3)
    p.add_argument("--init", choices=("random", "encoded"), default="random",
                   help="z0 init mode. 'random'~N(0,1) is the realistic case; "
                        "'encoded' uses encode(x_true) as the ceiling.")
    p.add_argument("--n-test", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-root", type=str,
                   default="./CSMF2/experiments/step_1_1_1_1/results")
    a = p.parse_args()
    return MAPCfg(ckpt_dir=a.ckpt_dir,
                  best_params=a.best_params,
                  best_params_expert=a.best_params_expert,
                  train_results_root=a.train_results_root,
                  steps=a.steps, lr=a.lr, lambda_prior=a.lambda_prior,
                  init=a.init, n_test=a.n_test, batch_size=a.batch_size,
                  seed=a.seed, out_root=a.out_root)


if __name__ == "__main__":
    cfg = _parse_args()
    try:
        report = run(cfg)
    except Exception:
        logger.error("MAP-refine run FAILED\n%s", traceback.format_exc())
        sys.exit(1)
    sys.exit(0)
