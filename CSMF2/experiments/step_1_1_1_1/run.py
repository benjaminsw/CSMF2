# =============================================================================
# STEP-1_1_1_1 v0.1 -- experiments.step_1_1_1_1.run
# Purpose: run ONE latent-refinement arm (random_map | is_only | is_map) on a
#          frozen step_1_1 checkpoint, over n_images. Computes before/after
#          reconstruction metrics, writes report.json + the 4 core plots.
#          Run the three arms (same ckpt, same seed) then aggregate_arms.
# CONVENTION: no fallback / mock / dummy / pass. Bad input / non-finite ->
#             logger.error + raise. Flow is frozen; only z is optimised.
# Exit codes: 0 = ran + metrics written; 1 = crash. (No gate concept here --
#             this is a measurement experiment, not a gated training step.)
# Changelog (NEW in v0.1):
#   * Introduced. Loads ckpt (architecture-agnostic), builds A from the saved
#     cfg, runs the arm, records fwd_rel/residual/PSNR before+after, MAP
#     internals, IS internals, runtime; saves grid + objective + bar plots.
# Update summary:
#   v0.1 is the core MAP-ABL runner. random_map: 1 latent -> MAP; is_only:
#   K candidates -> pick best -> 0 steps; is_map: K -> best -> MAP. "before"
#   = the arm's initial latent (random draw, or selected candidate); "after"
#   = post-MAP (== before for is_only). PSNR uses the loader's clean x.
# =============================================================================
from __future__ import annotations
import argparse
import json
import logging
import math
import sys
import time
import traceback
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ...data.degrade import MNISTDegraded, inverse_logit, blur, downsample
from .config import MAPCfg
from .model_io import build_from_report
from .map_core import (map_objective, generate_candidates, select_best,
                       run_map)

logger = logging.getLogger("CSMF2.step_1_1_1_1.run")
__version__ = "0.1"
__abbr__ = "STEP-1_1_1_1"

_IMAGE_HW = (28, 28)


def _configure_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s :: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)


def _recon_metrics(z, h, expert, x_clean, y, *, blur_sigma, scale,
                   lambda_prior):
    """Per-image data/prior/residual/fwd_rel/PSNR for latent z (no grad)."""
    with torch.no_grad():
        total, data, prior, Ax = map_objective(
            z, h, expert, y, blur_sigma=blur_sigma, scale=scale,
            lambda_prior=lambda_prior)
        y_norm = y.flatten(1).norm(dim=1).clamp_min(1e-12)
        fwd_rel = (Ax - y).flatten(1).norm(dim=1) / y_norm
        x_hat = inverse_logit(expert.decode(z, h)).view(z.size(0), 1, *_IMAGE_HW)
        mse = (x_hat - x_clean).flatten(1).pow(2).mean(dim=1).clamp_min(1e-12)
        psnr = 10.0 * torch.log10(1.0 / mse)             # data in [0,1]
        z_norm = z.flatten(1).norm(dim=1)
    return {"data": data, "prior": prior, "total": total, "fwd_rel": fwd_rel,
            "psnr": psnr, "z_norm": z_norm, "x_hat": x_hat, "Ax": Ax}


def _mean(t) -> float:
    return float(t.mean())


def run(cfg: MAPCfg) -> dict:
    t0 = time.time()
    out_dir = Path(cfg.out_root) / cfg.run_tag()
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    _configure_logging(out_dir)
    logger.info("STEP-1_1_1_1 run | tag=%s | cfg=%s", cfg.run_tag(), cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    expert, cond, train_cfg = build_from_report(cfg.ckpt_dir, device)
    blur_sigma, scale = train_cfg.blur_sigma, train_cfg.scale

    # ---- data: same degradation as training; take n_images from val -------
    ds = MNISTDegraded(train_cfg.data_root, split="val",
                       sigma=blur_sigma, scale=scale,
                       noise_sigma=train_cfg.noise_sigma)
    loader = DataLoader(ds, batch_size=cfg.n_images, shuffle=False)
    x_clean, y = next(iter(loader))
    x_clean = x_clean.to(device); y = y.to(device)
    if x_clean.dim() == 3:
        x_clean = x_clean.unsqueeze(1)
    n = x_clean.size(0)
    logger.info("[run] refining %d images (arm=%s)", n, cfg.init)

    # ---- pick initial latent z0 per arm -----------------------------------
    is_block = None
    if cfg.init == "random_map":
        h = cond(y)
        z0 = torch.randn(n, int(expert.dim), generator=gen,
                         device=device, dtype=h.dtype)
    else:  # is_only / is_map
        z_cand, resid, h = generate_candidates(
            expert, cond, y, K=cfg.K, blur_sigma=blur_sigma, scale=scale,
            generator=gen)
        z0, best_r, mean_r, gap = select_best(z_cand, resid)
        is_block = {"best_of_K_residual": _mean(best_r),
                    "mean_candidate_residual": _mean(mean_r),
                    "candidate_rank_gap": _mean(gap),
                    "selected_z_norm": _mean(z0.flatten(1).norm(dim=1))}

    # ---- BEFORE metrics (initial latent) ----------------------------------
    before = _recon_metrics(z0, h, expert, x_clean, y, blur_sigma=blur_sigma,
                            scale=scale, lambda_prior=cfg.lambda_prior)

    # ---- MAP (skipped for is_only) ----------------------------------------
    if cfg.init == "is_only":
        z_final = z0
        opt_block = {"objective_curve": [], "grad_norms": [],
                     "n_steps": 0, "converged": True}
    else:
        opt_block = run_map(z0, h, expert, y, blur_sigma=blur_sigma,
                            scale=scale, lambda_prior=cfg.lambda_prior,
                            steps=cfg.map_steps, lr_z=cfg.lr_z,
                            conv_tol=cfg.conv_tol, log_every=cfg.log_every)
        z_final = opt_block.pop("z")

    after = _recon_metrics(z_final, h, expert, x_clean, y,
                           blur_sigma=blur_sigma, scale=scale,
                           lambda_prior=cfg.lambda_prior)

    # ---- plots -------------------------------------------------------------
    _plot_grid(y, x_clean, before["x_hat"], after["x_hat"], cfg,
               plots / "recon_grid.png")
    if opt_block["objective_curve"]:
        _plot_objective(opt_block["objective_curve"], cfg,
                        plots / "objective_curve.png")
    _plot_bars(before, after, cfg, plots / "residual_psnr_bars.png")

    runtime = time.time() - t0
    report = {
        "map_cfg": cfg.__dict__,
        "train_cfg": train_cfg.__dict__,
        "arm": cfg.init,
        "n_images": n,
        "metrics": {
            "fwd_rel_before": _mean(before["fwd_rel"]),
            "fwd_rel_after":  _mean(after["fwd_rel"]),
            "residual_before": _mean(before["data"]),
            "residual_after":  _mean(after["data"]),
            "psnr_before": _mean(before["psnr"]),
            "psnr_after":  _mean(after["psnr"]),
            "data_term":  _mean(after["data"]),
            "prior_term": _mean(after["prior"]),
            "total_objective": _mean(after["total"]),
            "latent_norm": _mean(after["z_norm"]),
        },
        "is_block": is_block,
        "optimization": opt_block,
        "runtime_s": runtime,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    logger.info("STEP-1_1_1_1 run DONE arm=%s fwd_rel %.4f->%.4f psnr %.2f->"
                "%.2f runtime=%.1fs out=%s", cfg.init,
                report["metrics"]["fwd_rel_before"],
                report["metrics"]["fwd_rel_after"],
                report["metrics"]["psnr_before"],
                report["metrics"]["psnr_after"], runtime, out_dir)
    return report


def _to_np(t, i):
    return t[i, 0].detach().cpu().clamp(0.0, 1.0).numpy()


def _plot_grid(y, x_clean, x_before, x_after, cfg, path, n_show=8):
    n = min(n_show, y.size(0))
    rows = [("y (degraded)", y), ("x (true)", x_clean),
            (f"init ({cfg.init})", x_before), ("after MAP", x_after)]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.4 * n, 5.6), dpi=120)
    if n == 1:
        axes = axes.reshape(len(rows), 1)
    for r, (label, t) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(_to_np(t, c), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, ha="right",
                              va="center")
    fig.suptitle(f"MAP reconstruction -- arm={cfg.init}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    logger.info("[run] saved %s", path)


def _plot_objective(curve, cfg, path):
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=120)
    ax.plot(range(len(curve)), curve, marker=".")
    ax.set_xlabel("MAP step"); ax.set_ylabel("mean objective")
    ax.set_title(f"MAP objective curve -- arm={cfg.init}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    logger.info("[run] saved %s", path)


def _plot_bars(before, after, cfg, path):
    labels = ["fwd_rel", "PSNR"]
    b = [_mean(before["fwd_rel"]), _mean(before["psnr"])]
    a = [_mean(after["fwd_rel"]), _mean(after["psnr"])]
    x = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), dpi=120)
    for ax, i, name in zip(axes, range(2), labels):
        ax.bar(["before", "after"], [b[i], a[i]],
               color=["#888", "#1f77b4"])
        ax.set_title(name); ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"before vs after -- arm={cfg.init}", fontsize=10)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    logger.info("[run] saved %s", path)


def _parse_args():
    p = argparse.ArgumentParser(description="MAP / IS+MAP latent refinement")
    p.add_argument("--ckpt-dir", required=True,
                   help="step_1_1 run dir containing ckpt.pt + report.json")
    p.add_argument("--init", choices=("random_map", "is_only", "is_map"),
                   default="is_map")
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--map-steps", type=int, default=100)
    p.add_argument("--lr-z", type=float, default=0.05)
    p.add_argument("--lambda-prior", type=float, default=1e-3)
    p.add_argument("--sigma-y", type=float, default=0.1)
    p.add_argument("--conv-tol", type=float, default=1e-4)
    p.add_argument("--n-images", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-root",
                   default="./CSMF2/experiments/step_1_1_1_1/results")
    p.add_argument("--log-every", type=int, default=10)
    a = p.parse_args()
    steps = 0 if a.init == "is_only" else a.map_steps
    cfg = MAPCfg(ckpt_dir=a.ckpt_dir, init=a.init, K=a.K, map_steps=steps,
                 lr_z=a.lr_z, lambda_prior=a.lambda_prior, sigma_y=a.sigma_y,
                 conv_tol=a.conv_tol, n_images=a.n_images, seed=a.seed,
                 out_root=a.out_root, log_every=a.log_every)
    return cfg


if __name__ == "__main__":
    cfg = _parse_args()
    try:
        run(cfg)
        sys.exit(0)
    except Exception:
        logger.error("STEP-1_1_1_1 run FAILED\n%s", traceback.format_exc())
        sys.exit(1)
