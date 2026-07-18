# SEQREF-TRNREF v0.9 -- train_refiner
# LIFETIME: KEEP
# Changelog (v0.8 -> v0.9, SPGATE V1 support -- spatial gate):
#   * refiner.gate_mode OPTIONAL in {scalar, spatial}; absent -> scalar
#     (existing hashes untouched); passed to CplRegRefiner v0.3.
#   * apply_gate(g, dx) single application point (scalar B, or spatial
#     B,1,H,W; shape-checked, raise on mismatch) replaces every
#     g.view(-1,1,1,1): loss, budget, recomposition, val, plots.
#   * _gate_stats trainer-level generalized statistics (GatedUpdate.g_stats
#     is scalar-only): adds g_frac_near_zero, g_std_within_image (median),
#     g_cov_10, g_cov_50 columns; scatter uses per-sample gate mean in both
#     modes. applied_residual_ratio_median column = SPGATE SS8 eligibility.
#   * Spatial best-checkpoint diagnostics: gate_heatmap_mean, gate_grid
#     (improved/neutral/harmed), dpsnr-vs-gstd/cov10 scatters, gate-vs-error
#     binned diagnostic (VALIDATION ONLY -- target never enters the gate).
#   * per_sample_best.npz artifact (per-sample vectors never enter
#     metrics.csv): dpsnr, applied/residual ratio (+median/p10/p90/frac>=.10),
#     g mean/std/coverages.
#   * Startup self-checks (spatial): g/dx shape; initial g_mean within
#     [0.8,1.2]*g_init; spatial param count -> status.json (with total).
# Changelog (v0.7 -> v0.8, DBG Phase C support -- parameterization scale):
#   * refiner.residual_scale_mode OPTIONAL in {none, train_residual_rms};
#     absent -> "none" (existing cfg hashes untouched); resolved mode always
#     logged. C's config sets train_residual_rms EXPLICITLY.
#   * train_residual_rms: scale = mean_i rms_pixels(x_true_i - x0_i) over
#     the FIXED train cache only (pixel count inferred, not hard-coded);
#     recorded in status.json (mode, value, n_train, formula). Never clamped
#     or tuned after seeing C results.
#   * Applied consistently: training loss, val metrics, conditioning-gap
#     shuffles, and the correction budget all use dx_eff = scale * dx_raw
#     with x1 RECOMPOSED as x0 + g*dx_eff. Startup self-check verifies
#     recomposition matches the model's own x1 at scale=1 (raise on
#     mismatch). Metrics gain dx_raw_rms / dx_eff_rms / applied_rms so
#     compensation (network inflating dx_raw to cancel the scale) is
#     visible. Epoch-0 row matters for C: the warm-start correction starts
#     shrunk by ~scale.
# Phase 6: train ONE candidate refiner (flavor realnvp|nice) against the same
# frozen base x0. Loss = Charbonnier(x1, x_true) + lambda_budget * budget,
# where budget uses the configured train.budget_form ('g_dx' LOCKED for the
# SEQREF-WARM campaign: penalty on the APPLIED correction (g*dx)^2).
# Keep-best on val_dpsnr with best-relative early stopping.
# x0/inputs precomputed once via base_io cache.
# Gate (approved split):
#   HARD PASS:   val_dpsnr > +0.3 dB, val_fwd_rel_x1 <= val_fwd_rel_x0,
#                base_frozen verified (grad_max_abs == 0)
#   DIAGNOSTIC:  y_gap_dpsnr > 0, atr_gap_dpsnr > 0 (evidence, required
#                non-contradictory for 3-seed promotion)
# No NLL, no lambda_rec. No fallback/mock/pass; failures logger.error + raise.
# Changelog (v0.6 -> v0.7, SEQREF-WARM duplicate-control safety):
#   * experiment.replicate REQUIRED (no silent default -- forgotten field
#     must fail loudly, not recreate collisions). CLI --replicate override
#     folded into cfg BEFORE cfg_hash (tag and hash always agree). Run dir
#     gains _rep{n}_; status.json records replicate.
#   * Early stopping REQUIRED: train.early_stop_patience (campaign 15) +
#     train.early_stop_min_delta_dpsnr (campaign 0.005 dB, pre-registered
#     PRIOR, not noise-derived; NOT the W0/W1 significance gate -- the
#     duplicate-control floor is measured separately). Best-relative rule:
#     improved iff val_dpsnr > best + min_delta; stale >= patience -> stop.
#     Keep-best retained. status.json: stopped_early, stop_epoch,
#     epochs_completed, best_epoch, best_val_dpsnr, both ES params.
#   * Header loss line corrected (said mean(dx^2); budget_form-aware
#     since v0.3, g_dx locked for this campaign).
# Changelog (v0.5 -> v0.6, SEQREF-WARM noise control, consolidated):
#   * initialization.mode REQUIRED in {scratch, warm_start} -- never inferred
#     from a missing warm_start block; scratch FORBIDS a warm_start block
#     (contradiction -> raise). scratch ws_audit = {mode, loaded_fraction 0.0,
#     source None}. Unblocks W0 arms (v0.2's "scratch deferred" now due).
#   * Training seed APPLIED (was recorded only): torch.manual_seed +
#     cuda.manual_seed_all before model build; cached-tensor train loader
#     gets its own seeded Generator. W0/W1 same-seed pairing now real;
#     residual CUDA nondeterminism is what duplicate controls measure.
#   * diagnostics.perm_seed REQUIRED (campaign 778): conditioning-gap
#     permutation built ONCE and reused across epochs and arms (was fresh
#     randperm per epoch -> diag_ok could flicker); length-validated.
#   * status.json records init_mode, recon_seed, diag_perm_seed.
# Changelog (v0.4 -> v0.5, SEQREF-WARM noise control E2):
#   * base.recon_seed REQUIRED (raise if absent): dedicated x0-reconstruction
#     seed passed to BOTH precompute_split calls, independent of the arm's
#     training seed. Every comparison arm (W0/W1/O1/O2, any train.seed) now
#     trains and is scored against the IDENTICAL cached x0, making cross-arm
#     dpsnr paired (base-eval sampling noise cancels). Previously rng_seed
#     (training seed) keyed the cache, so multi-seed arms drew different x0.
#     Campaign value: recon_seed 777 (EXEC SS7.2). No other logic changes.
# Changelog (TRNREF v0.4 -> v0.4-fseq, SEQREF-FSEQ W2):
#   * seqref_mri fork; dataset construction via degrade.make_degraded
#     with REQUIRED cell.dataset key ({mnist, fashion_mnist}; raise on
#     absent/unknown). No other logic changes.
# Changelog (v0.3 -> v0.4, SEQREF-REFINE2):
#   * Stage-2 support: optional cfg.stage1.run_dir loads a FROZEN trained
#     stage-1 refiner (base_io.FrozenStage1). Cached base tensors transform to
#     x_prev = x1 = clamp(stage1(inputs0, x0)), inputs = [y_up, x1, Aᵀr1]
#     (precompute_stage2, dual-sha cache). ALL "x0"-named columns/fields then
#     mean THE PREVIOUS STAGE; gate is vs x_prev (the sequential claim);
#     gate thresholds UNCHANGED (fixed _GATE_DPSNR, HARD/DIAG split).
#     Frozen check covers base AND stage-1 each epoch. Run dirs use
#     <flavor>_refine2_ tag; status.json adds stage + stage1 block.
#   * PRE-REGISTERED stage-2 two-tier gate (locked before any x2 numbers):
#     HARD >= +0.30 (+ fwd_rel<=prev, ssim>=prev, diag); MEANINGFUL >= +0.10
#     (+ per-sample mean>0, %improved>55%, fwd_rel<=prev, diag). Stage-1
#     gate unchanged.
# Changelog (v0.2 -> v0.3):
#   * Budget penalty INCENTIVE FIX: penalty now on the APPLIED correction.
#     Config key train.budget_form REQUIRED, in {"dx","g_dx"}:
#       dx    -> lam * mean(dx^2)          (v0.2 behaviour, kept for records)
#       g_dx  -> lam * mean((g*dx)^2)      (prices what touches the image;
#                removes the "shrink dx, max g" cheat that pinned the gate)
#     Absent/invalid budget_form -> raise (no silent default; the key also
#     separates cfg hashes so Run-3 dirs never collide with Run-1/2).
# Changelog (v0.1 -> v0.2):
#   * Warm-start exclusion policy threaded from config (warm_start.
#     exclude_patterns, default CPLREG DEFAULT_EXCLUDE); warm_start.
#     min_loaded_fraction now REQUIRED in config (raise if absent) since the
#     policy changes the numel baseline. Audit incl. exclusions -> status.json.
# Changelog (v0.1):
#   * Full approved tracking schema (metrics.csv / status.json), 4 core plots
#     + 3 extras, 7-row recon grid, seed0 summary print, HARD/DIAG gate eval.
from __future__ import annotations
import argparse
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from seqref_mri.src.degrade import make_degraded
from seqref_mri.src.metrics import psnr as _psnr, ssim as _ssim, fwd_rel as _fwd_rel
from seqref_mri.src.refiners.base_io import (FrozenBase, FrozenStage1,
                                               precompute_split,
                                               precompute_stage2)
from seqref_mri.src.refiners.coupling_regressor import (CplRegRefiner,
                                                          load_warm_start,
                                                          DEFAULT_EXCLUDE)
from seqref_mri.src.refiners.gated_update import GatedUpdate
from seqref_mri.src.train_utils import (setup_logger, seed_from_index,
                                          cfg_hash, write_json, sha256_file)

logger = setup_logger("seqref_mri.train_refiner")
__version__ = "0.9"

# Pre-registered gate constants (locked before any x2 numbers, Ben 2026-07-09)
_GATE_DPSNR = 0.3          # stage-1 & stage-2 HARD threshold (aggregate dB)
_GATE2_MEANINGFUL = 0.1    # stage-2 MEANINGFUL tier
_GATE2_PCT = 0.55          # stage-2 MEANINGFUL: % samples improved



def _load_cfg(path: str) -> dict:
    if not os.path.isfile(path):
        logger.error("[train_refiner] config not found: %s", path)
        raise FileNotFoundError(path)
    with open(path) as f:
        return yaml.safe_load(f)


def _charbonnier(x1, x_true, eps: float) -> torch.Tensor:
    return torch.sqrt((x1 - x_true) ** 2 + eps * eps).mean()


def _psnr_per_sample(x_hat, x_true) -> torch.Tensor:
    m = ((x_hat - x_true) ** 2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / m)


def apply_gate(g, dx):
    # Single application point for scalar (B,) and spatial (B,1,H,W) gates.
    # NO no_grad here: used inside the training loss and budget -- gradients
    # must flow through g * dx.
    if g.ndim == 1:
        g = g[:, None, None, None]
    if g.ndim != 4 or dx.ndim != 4:
        logger.error("[apply_gate] need 4-D g and dx, got g.ndim=%d "
                     "dx.ndim=%d", g.ndim, dx.ndim)
        raise ValueError("apply_gate: need 4-D g and dx")
    if g.shape[0] != dx.shape[0] or g.shape[1] != dx.shape[1]:
        logger.error("[apply_gate] batch/channel mismatch: %s vs %s",
                     tuple(g.shape), tuple(dx.shape))
        raise ValueError("apply_gate: batch/channel mismatch")
    if g.shape[2:] not in ((1, 1), dx.shape[2:]):
        logger.error("[apply_gate] spatial shape %s not broadcastable to "
                     "%s", tuple(g.shape[2:]), tuple(dx.shape[2:]))
        raise ValueError("apply_gate: spatial shapes incompatible")
    return g * dx


def _gate_stats(g, g_max):
    # Trainer-level generalized gate statistics: works for scalar (B,) and
    # spatial (B,1,H,W). GatedUpdate.g_stats is scalar-only; do not use it
    # for spatial gates. Column names match the historical scalar set plus
    # the spatial additions (zeros/harmless for scalar mode).
    gf = g.flatten() if g.ndim == 1 else g.flatten(1)
    if g.ndim == 1:
        within_std = 0.0
        per_sample_mean = g
    else:
        within_std = float(g.flatten(1).std(dim=1).median())
        per_sample_mean = g.flatten(1).mean(dim=1)
    return {"g_mean": float(gf.mean()), "g_std": float(gf.std()),
            "g_min": float(gf.min()), "g_max_val": float(gf.max()),
            "g_max_frac": float((gf > 0.95 * g_max).float().mean()),
            "g_frac_near_zero": float((gf < 0.05 * g_max).float().mean()),
            "g_std_within_image": within_std,
            "g_cov_10": float((gf > 0.10 * g_max).float().mean()),
            "g_cov_50": float((gf > 0.50 * g_max).float().mean()),
            }, per_sample_mean


@torch.no_grad()
def _forward_split(model, inputs, x0, bs, device, residual_scale=1.0):
    # Batched forward over a whole cached split. Returns x1, dx_eff, g (cpu).
    # v0.8: if residual_scale != 1, dx_eff = scale * dx_raw and x1 is
    # RECOMPOSED as x0 + g*dx_eff (verified against the model's own x1 at
    # startup). dx returned is dx_EFF; raw is dx_eff / scale.
    xs, dxs, gs = [], [], []
    for i in range(0, inputs.size(0), bs):
        inp = inputs[i:i + bs].to(device)
        x0b = x0[i:i + bs].to(device)
        x1, dx, g = model(inp, x0b)
        if residual_scale != 1.0:
            dx = residual_scale * dx
            x1 = x0b + apply_gate(g, dx)
        xs.append(x1.cpu()); dxs.append(dx.cpu()); gs.append(g.cpu())
    return torch.cat(xs), torch.cat(dxs), torch.cat(gs)


@torch.no_grad()
def _val_metrics(model, val, base, bs, device, psnr_x0, ssim_x0, fwd_x0,
                 val_perm, residual_scale=1.0):
    # val = dict(x_true, y, x0, inputs). Returns full per-epoch metric dict.
    # val_perm: FIXED permutation (built once from diagnostics.perm_seed) so
    # the conditioning-gap diagnostic is paired across epochs AND arms.
    model.eval()
    x_true, y, x0, inputs = val["x_true"], val["y"], val["x0"], val["inputs"]
    if val_perm.size(0) != inputs.size(0):
        logger.error("[train_refiner] val_perm length %d != val inputs %d",
                     val_perm.size(0), inputs.size(0))
        raise ValueError("val_perm length mismatch")
    x1, dx, g = _forward_split(model, inputs, x0, bs, device,
                               residual_scale=residual_scale)
    x1c = x1.clamp(0, 1)
    psnr_x1 = _psnr(x1c, x_true); ssim_x1 = _ssim(x1c, x_true)
    fwd_x1 = _fwd_rel(x1c, y, base.blur_sigma, base.scale)
    gs, _ = _gate_stats(g, model.g_max)
    delta_l2 = float(dx.flatten(1).norm(dim=1).mean())
    delta_linf = float(dx.flatten(1).abs().max(dim=1).values.mean())
    tgt = (x_true - x0)
    delta_tgt_l2 = float((dx - tgt).flatten(1).norm(dim=1).mean())

    # conditioning gaps: permute one input channel across the batch (fixed perm).
    perm = val_perm
    def _gap(channel: int) -> tuple[float, float]:
        shuf = inputs.clone()
        shuf[:, channel] = inputs[perm, channel]
        x1s, _, _ = _forward_split(model, shuf, x0, bs, device,
                                   residual_scale=residual_scale)
        return _psnr(x1s.clamp(0, 1), x_true)
    correct_dpsnr = psnr_x1 - psnr_x0
    shuf_y_psnr = _gap(0)      # channel 0 = y_up
    shuf_atr_psnr = _gap(2)    # channel 2 = Aᵀr0
    m = {"val_psnr_x0": psnr_x0, "val_psnr_x1": psnr_x1,
         "val_dpsnr": psnr_x1 - psnr_x0,
         "val_ssim_x0": ssim_x0, "val_ssim_x1": ssim_x1,
         "val_dssim": ssim_x1 - ssim_x0,
         "val_fwd_rel_x0": fwd_x0, "val_fwd_rel_x1": fwd_x1,
         "val_dfwd_rel": fwd_x1 - fwd_x0,
         "delta_l2_mean": delta_l2, "delta_linf_mean": delta_linf,
         "dx_raw_rms": float((dx / residual_scale).pow(2).mean().sqrt()),
         "dx_eff_rms": float(dx.pow(2).mean().sqrt()),
         "applied_rms": float(apply_gate(g, dx).pow(2).mean().sqrt()),
         "applied_residual_ratio_median": float(
             (apply_gate(g, dx).flatten(1).pow(2).mean(dim=1).sqrt()
              / max(residual_scale, 1e-12)).median()),
         "delta_target_l2": delta_tgt_l2,
         "correct_y_dpsnr": correct_dpsnr,
         "shuffled_y_dpsnr": shuf_y_psnr - psnr_x0,
         "y_gap_dpsnr": psnr_x1 - shuf_y_psnr,
         "correct_atr_dpsnr": correct_dpsnr,
         "shuffled_atr_dpsnr": shuf_atr_psnr - psnr_x0,
         "atr_gap_dpsnr": psnr_x1 - shuf_atr_psnr,
         **{k if k != "g_max_val" else "g_max": v for k, v in gs.items()}}
    return m, x1c, dx, g


_CSV_COLS = ["epoch", "train_loss", "val_loss", "val_psnr_x0", "val_psnr_x1",
             "val_dpsnr", "val_ssim_x0", "val_ssim_x1", "val_dssim",
             "val_fwd_rel_x0", "val_fwd_rel_x1", "val_dfwd_rel",
             "g_mean", "g_std", "g_min", "g_max", "g_max_frac",
             "g_frac_near_zero", "g_std_within_image", "g_cov_10", "g_cov_50",
             "delta_l2_mean", "delta_linf_mean", "delta_target_l2",
             "dx_raw_rms", "dx_eff_rms", "applied_rms",
             "applied_residual_ratio_median",
             "correct_y_dpsnr", "shuffled_y_dpsnr", "y_gap_dpsnr",
             "correct_atr_dpsnr", "shuffled_atr_dpsnr", "atr_gap_dpsnr",
             "base_grad_norm", "refiner_grad_norm"]


def _plots(hist, run_dir):
    ep = [h["epoch"] for h in hist]
    def _line(keys, labels, ylab, name):
        plt.figure(figsize=(6, 4))
        for k, l in zip(keys, labels):
            plt.plot(ep, [h[k] for h in hist], label=l)
        plt.xlabel("epoch"); plt.ylabel(ylab); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(run_dir, name), dpi=110); plt.close()
    _line(["val_psnr_x0", "val_psnr_x1", "val_dpsnr"],
          ["PSNR(x0)", "PSNR(x1)", "ΔPSNR"], "dB", "psnr_curve.png")
    _line(["val_fwd_rel_x0", "val_fwd_rel_x1"],
          ["fwd_rel(x0)", "fwd_rel(x1)"], "fwd_rel", "fwd_rel_curve.png")
    _line(["g_mean", "g_max_frac"], ["g_mean", "g_max_frac"], "gate",
          "gate_curve.png")
    _line(["correct_y_dpsnr", "shuffled_y_dpsnr",
           "correct_atr_dpsnr", "shuffled_atr_dpsnr"],
          ["correct-y ΔPSNR", "shuffled-y ΔPSNR",
           "correct-Aᵀr ΔPSNR", "shuffled-Aᵀr ΔPSNR"], "ΔPSNR (dB)",
          "conditioning_gap_curve.png")


def _spatial_gate_plots(dx, g, x_true, x0, x1, run_dir, dps, g_max):
    # SPGATE SS7 best-checkpoint diagnostics (spatial mode only).
    import numpy as np
    gm = g[:, 0]
    # 6. Mean gate heatmap (+ across-sample std)
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    for a, (img, t) in zip(ax, [(gm.mean(0), "mean g"),
                                (gm.std(0), "std g")]):
        im = a.imshow(img.numpy()); a.set_title(t, fontsize=8); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "gate_heatmap_mean.png"), dpi=110)
    plt.close()
    # 5. Spatial gate grid: representative improved / neutral / harmed
    order = dps.argsort()
    picks = [order[-1], order[len(order) // 2], order[0]]
    dxe = dx; app = apply_gate(g, dx)
    rows_lab = ["x0", "dx_eff", "gate", "g*dx_eff", "x1", "x_true"]
    fig, ax = plt.subplots(len(rows_lab), 3, figsize=(4.5, 8))
    for c, i in enumerate(picks):
        imgs = [x0[i, 0], dxe[i, 0], gm[i], app[i, 0], x1[i, 0],
                x_true[i, 0]]
        for r, img in enumerate(imgs):
            ax[r, c].imshow(img.numpy(), cmap="gray"); ax[r, c].axis("off")
        ax[0, c].set_title(f"{dps[i]:+.2f}dB", fontsize=7)
    for r, lab in enumerate(rows_lab):
        ax[r, 0].text(-0.4, 0.5, lab, transform=ax[r, 0].transAxes,
                      ha="right", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "gate_grid.png"), dpi=110); plt.close()
    # 8. Scatters: dPSNR vs within-image g std / applied-ratio / coverage_10
    ps_std = g.flatten(1).std(dim=1).numpy()
    ps_cov10 = (g.flatten(1) > 0.10 * g_max).float().mean(1).numpy()
    for xv, xlab, fname in [(ps_std, "per-sample gate spatial std",
                             "scatter_dpsnr_vs_gstd.png"),
                            (ps_cov10, "per-sample coverage_10",
                             "scatter_dpsnr_vs_cov10.png")]:
        plt.figure(figsize=(5, 4))
        plt.scatter(xv, dps, s=4, alpha=0.3)
        plt.xlabel(xlab); plt.ylabel("per-sample dPSNR (dB)")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, fname), dpi=110); plt.close()
    # 9. Gate-error diagnostic (VALIDATION ONLY; target never enters the
    # gate at inference -- this only correlates the learned map post hoc).
    err = (x_true - x0).abs()[:, 0].flatten().numpy()
    gflat = gm.flatten().numpy()
    idx = np.argsort(err); nb = 20
    bins = np.array_split(idx, nb)
    be = [err[b].mean() for b in bins]; bg = [gflat[b].mean() for b in bins]
    corr = float(np.corrcoef(err, gflat)[0, 1])
    plt.figure(figsize=(5, 4))
    plt.plot(be, bg, marker="o")
    plt.xlabel("|x_true - x0| (binned)"); plt.ylabel("mean gate value")
    plt.title(f"gate vs base error (corr {corr:.3f})", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "gate_vs_error.png"), dpi=110)
    plt.close()


def _save_per_sample_artifacts(dx, g, x_true, x0, x1, residual_scale,
                               run_dir, g_max):
    # Per-sample vectors at the BEST checkpoint -> npz (never metrics.csv).
    import numpy as np
    dps = (_psnr_per_sample(x1, x_true) - _psnr_per_sample(x0, x_true))
    app = apply_gate(g, dx)
    ratio = (app.flatten(1).pow(2).mean(1).sqrt()
             / max(residual_scale, 1e-12))
    out = {"per_sample_dpsnr": dps.numpy(),
           "per_sample_applied_residual_ratio": ratio.numpy(),
           "ratio_median": float(ratio.median()),
           "ratio_p10": float(ratio.quantile(0.10)),
           "ratio_p90": float(ratio.quantile(0.90)),
           "ratio_frac_ge_010": float((ratio >= 0.10).float().mean())}
    if g.ndim == 4:
        gf = g.flatten(1)
        out.update({"per_sample_g_mean": gf.mean(1).numpy(),
                    "per_sample_g_std": gf.std(1).numpy(),
                    "per_sample_cov10": (gf > 0.10 * g_max)
                    .float().mean(1).numpy(),
                    "per_sample_cov50": (gf > 0.50 * g_max)
                    .float().mean(1).numpy()})
    else:
        out["per_sample_g_mean"] = g.numpy()
    np.savez(os.path.join(run_dir, "per_sample_best.npz"), **out)
    logger.info("[train_refiner] per-sample artifacts: per_sample_best.npz "
                "(ratio median %.4f, frac>=0.10 %.3f)",
                out["ratio_median"], out["ratio_frac_ge_010"])


def _extra_plots(dx, g, x_true, x0, x1, run_dir, g_max):
    plt.figure(figsize=(6, 4))
    plt.hist(dx.flatten().numpy(), bins=100, alpha=0.6, label="Δx")
    plt.hist(apply_gate(g, dx).flatten().numpy(), bins=100,
             alpha=0.6, label="g·Δx")
    plt.legend(); plt.yscale("log"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "delta_hist.png"), dpi=110); plt.close()

    dps = (_psnr_per_sample(x1, x_true) - _psnr_per_sample(x0, x_true)).numpy()
    ps_g = (g if g.ndim == 1 else g.flatten(1).mean(dim=1)).numpy()
    plt.figure(figsize=(6, 4))
    plt.scatter(ps_g, dps, s=4, alpha=0.3)
    plt.xlabel("per-sample gate mean"); plt.ylabel("per-sample ΔPSNR (dB)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "scatter_dpsnr_vs_g.png"), dpi=110)
    plt.close()

    if g.ndim == 4:
        _spatial_gate_plots(dx, g, x_true, x0, x1, run_dir, dps,
                            g_max)

    worst = torch.argsort(torch.from_numpy(dps))[:8]
    fig, ax = plt.subplots(3, 8, figsize=(9, 3.5))
    for c, i in enumerate(worst):
        for r, img in enumerate([x0[i, 0], x1[i, 0], x_true[i, 0]]):
            ax[r, c].imshow(img, cmap="gray", vmin=0, vmax=1); ax[r, c].axis("off")
        ax[0, c].set_title(f"{dps[i]:.2f}dB", fontsize=7)
    for r, lab in enumerate(["x0", "x1", "x_true"]):
        ax[r, 0].text(-0.35, 0.5, lab, transform=ax[r, 0].transAxes,
                      ha="right", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "failure_cases.png"), dpi=110); plt.close()


def _recon_grid(val, x1, run_dir, k=8):
    y_up = torch.nn.functional.interpolate(val["y"][:k], size=(28, 28),
                                           mode="nearest")
    x0, xt, x1k = val["x0"][:k], val["x_true"][:k], x1[:k]
    rows = [y_up, x0, x1k, xt, (x1k - x0).abs(), (xt - x0).abs(),
            (xt - x1k).abs()]
    labels = ["y_up", "x0", "x1", "x_true", "|x1-x0|", "|xt-x0|", "|xt-x1|"]
    fig, ax = plt.subplots(7, k, figsize=(k + 1, 7))
    for r in range(7):
        for c in range(k):
            ax[r, c].imshow(rows[r][c, 0], cmap="gray", vmin=0, vmax=1)
            ax[r, c].axis("off")
        ax[r, 0].text(-0.3, 0.5, labels[r], transform=ax[r, 0].transAxes,
                      ha="right", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "recon_grid.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--replicate", type=int, default=None,
                    help="override experiment.replicate (folded into cfg "
                         "BEFORE hashing, so tag and hash agree)")
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    # v0.7: experiment.replicate REQUIRED (no silent default -- a forgotten
    # field must fail loudly, not recreate run-dir collisions). CLI override
    # is applied to cfg BEFORE cfg_hash so duplicates get distinct hashes.
    if args.replicate is not None:
        cfg.setdefault("experiment", {})["replicate"] = int(args.replicate)
    if "replicate" not in cfg.get("experiment", {}):
        logger.error("[train_refiner] experiment.replicate missing -- "
                     "REQUIRED (duplicate-control safety; use 0 for the "
                     "primary run, 1/2 for duplicates, or pass --replicate)")
        raise KeyError("experiment.replicate required")
    replicate = int(cfg["experiment"]["replicate"])
    seed_index = args.seed if args.seed is not None else int(cfg["train"]["seed"])
    rng_seed = seed_from_index(seed_index)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    r = cfg["refiner"]
    flavor = r["flavor"]
    base = FrozenBase(cfg["base"]["run_dir"], device)
    n_post = int(cfg["base"].get("n_post", 16))
    if "recon_seed" not in cfg["base"]:
        logger.error("[train_refiner] base.recon_seed missing -- REQUIRED "
                     "(dedicated x0 reconstruction seed, independent of "
                     "train.seed; campaign value 777, EXEC SS7.2)")
        raise KeyError("base.recon_seed required")
    recon_seed = int(cfg["base"]["recon_seed"])

    # v0.6: initialization.mode REQUIRED and explicit -- never inferred from
    # a missing warm_start block. Distinct cfg hashes for W0/W1 by declaration.
    init_cfg = cfg.get("initialization", {})
    init_mode = init_cfg.get("mode")
    if init_mode not in ("scratch", "warm_start"):
        logger.error("[train_refiner] initialization.mode REQUIRED, one of "
                     "{'scratch','warm_start'}; got %r", init_mode)
        raise ValueError("initialization.mode required: 'scratch'|'warm_start'")
    if init_mode == "scratch" and cfg.get("warm_start"):
        logger.error("[train_refiner] initialization.mode=scratch but a "
                     "warm_start block is present -- contradictory config")
        raise ValueError("scratch mode forbids a warm_start block")

    # v0.6: diagnostics.perm_seed REQUIRED -- fixed conditioning-gap
    # permutation, paired across epochs and arms (EXEC SS7.2).
    diag_cfg = cfg.get("diagnostics", {})
    if "perm_seed" not in diag_cfg:
        logger.error("[train_refiner] diagnostics.perm_seed missing -- "
                     "REQUIRED (fixed val permutation; campaign value 778)")
        raise KeyError("diagnostics.perm_seed required")
    diag_perm_seed = int(diag_cfg["perm_seed"])

    # v0.6: APPLY the training seed (previously recorded but never applied) --
    # model init and cached-tensor shuffle order become seed-paired across
    # arms; residual CUDA nondeterminism remains and is what duplicate
    # controls measure.
    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)
    stage1_cfg = cfg.get("stage1")
    stage = 2 if (stage1_cfg or {}).get("run_dir") else 1
    chash = cfg_hash(cfg)
    stage_tag = "refine2" if stage == 2 else "refine"
    run_dir = os.path.join(cfg["output"]["root"],
                           f"{flavor}_{stage_tag}_s{base.scale}_n"
                           f"{float(base.cfg['cell']['noise_sigma']):.2f}_"
                           f"seed{seed_index}_rep{replicate}_{chash}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info("[train_refiner] flavor=%s stage=%d seed=%d dir=%s", flavor,
                stage, seed_index, run_dir)

    cell = base.cfg["cell"]
    dk = dict(sigma=base.blur_sigma, scale=base.scale,
              noise_sigma=float(cell["noise_sigma"]))
    root = cell["data_root"]
    bs = int(cfg["train"]["batch_size"])
    tl = DataLoader(make_degraded(cell.get("dataset"), root, split="train", **dk), batch_size=bs,
                    shuffle=False, num_workers=2)
    vl = DataLoader(make_degraded(cell.get("dataset"), root, split="val", **dk), batch_size=bs,
                    shuffle=False, num_workers=2)
    cache_dir = os.path.join(cfg["output"]["root"], "_cache")
    trX, trY, trX0, trIn = precompute_split(base, tl, n_post=n_post,
                                            rng_seed=recon_seed,
                                            cache_dir=cache_dir,
                                            split_name="train", device=device)
    vaX, vaY, vaX0, vaIn = precompute_split(base, vl, n_post=n_post,
                                            rng_seed=recon_seed,
                                            cache_dir=cache_dir,
                                            split_name="val", device=device)
    stage1 = None
    if stage == 2:
        stage1 = FrozenStage1(stage1_cfg["run_dir"], device)
        trX0, trIn = precompute_stage2(stage1, base, trX, trY, trX0, trIn,
                                       batch_size=bs, cache_dir=cache_dir,
                                       split_name="train")
        vaX0, vaIn = precompute_stage2(stage1, base, vaX, vaY, vaX0, vaIn,
                                       batch_size=bs, cache_dir=cache_dir,
                                       split_name="val")
        logger.info("[train_refiner] STAGE 2: gating vs stage-1 x1 (%s)",
                    stage1.checkpoint_sha256[:12])
    val = {"x_true": vaX, "y": vaY, "x0": vaX0, "inputs": vaIn}
    psnr_x0 = _psnr(vaX0, vaX); ssim_x0 = _ssim(vaX0, vaX)
    fwd_x0 = _fwd_rel(vaX0, vaY, base.blur_sigma, base.scale)
    logger.info("[train_refiner] frozen base val: psnr=%.3f ssim=%.4f fwd=%.4f",
                psnr_x0, ssim_x0, fwd_x0)

    gate_mode = cfg["refiner"].get("gate_mode", "scalar")
    if gate_mode not in ("scalar", "spatial"):
        logger.error("[train_refiner] refiner.gate_mode must be scalar|"
                     "spatial, got %r", gate_mode)
        raise ValueError("invalid gate_mode")
    model = CplRegRefiner(flavor=flavor,
                          dim=int(r.get("dim", 784)),
                          h_dim=int(r.get("h_dim", 256)),
                          hidden=int(r.get("hidden", 256)),
                          n_layers=r.get("n_layers"),
                          cond_width=int(r.get("cond_width", 128)),
                          film_hidden=int(r.get("film_hidden", 128)),
                          film_depth=int(r.get("film_depth", 2)),
                          film_use_gelu=bool(r.get("film_use_gelu", True)),
                          s_max=float(r.get("s_max", 4.0)),
                          post_init_std=float(r.get("post_init_std", 1e-3)),
                          g_max=float(r.get("g_max", 0.5)),
                          g_init=float(r.get("g_init", 0.05)),
                          gate_mode=gate_mode).to(device)
    ws_audit = None
    if init_mode == "warm_start":
        ws = cfg.get("warm_start", {})
        if not ws.get("path"):
            logger.error("[train_refiner] initialization.mode=warm_start but "
                         "warm_start.path missing -- required")
            raise ValueError("warm_start.path required in warm_start mode")
        if "min_loaded_fraction" not in ws:
            logger.error("[train_refiner] warm_start.min_loaded_fraction "
                         "required (policy exclusions change the baseline)")
            raise ValueError("warm_start.min_loaded_fraction required")
        excl = tuple(ws.get("exclude_patterns", DEFAULT_EXCLUDE))
        ws_audit = load_warm_start(model, ws["path"],
                                   min_loaded_fraction=float(
                                       ws["min_loaded_fraction"]),
                                   exclude_patterns=excl)
        ws_audit["source"] = ws["path"]
        ws_audit["source_sha256"] = sha256_file(ws["path"])
        ws_audit["mode"] = "warm_start"
    else:  # scratch (W0): fresh init from the seeded RNG above, no loading.
        ws_audit = {"mode": "scratch", "loaded_fraction": 0.0, "source": None}
        logger.info("[train_refiner] initialization.mode=scratch -- fresh "
                    "init, no checkpoint loaded")

    tset = TensorDataset(trX, trX0, trIn)
    train_gen = torch.Generator().manual_seed(rng_seed)
    tload = DataLoader(tset, batch_size=bs, shuffle=True, drop_last=True,
                       generator=train_gen)
    # v0.6: fixed conditioning-gap permutation, built ONCE, reused all epochs.
    diag_gen = torch.Generator().manual_seed(diag_perm_seed)
    val_perm = torch.randperm(vaIn.size(0), generator=diag_gen)

    # v0.8: residual parameterization scale (DBG SS7.5). Key OPTIONAL:
    # absent resolves to "none" (current behaviour) so existing cfg hashes
    # are untouched; C's config sets train_residual_rms EXPLICITLY.
    rs_mode = cfg["refiner"].get("residual_scale_mode", "none")
    if rs_mode not in ("none", "train_residual_rms"):
        logger.error("[train_refiner] refiner.residual_scale_mode must be "
                     "'none' or 'train_residual_rms', got %r", rs_mode)
        raise ValueError("invalid residual_scale_mode")
    residual_scale = 1.0
    if rs_mode == "train_residual_rms":
        # scale = mean_i sqrt(mean_pixels((x_true_i - x0_i)^2)) over the
        # FIXED train cache only; pixel count inferred from the tensor.
        with torch.no_grad():
            res = (trX - trX0).flatten(1)
            residual_scale = float(res.pow(2).mean(dim=1).sqrt().mean())
        if not (residual_scale > 0):
            logger.error("[train_refiner] residual_scale <= 0 (%r)",
                         residual_scale)
            raise ValueError("residual_scale must be positive")
        # Recomposition self-check: x0 + g*dx must equal the model's own x1
        # at scale=1 on one val batch, else the trainer-level scaling would
        # not match the model's update rule.
        with torch.no_grad():
            inb = vaIn[:bs].to(device); x0b = vaX0[:bs].to(device)
            x1m, dxm, gm = model(inb, x0b)
            x1r = x0b + apply_gate(gm, dxm)
            dmax = float((x1m - x1r).abs().max())
        if dmax > 1e-5:
            logger.error("[train_refiner] recomposition mismatch %.3e -- "
                         "model update rule is not x0 + g*dx; residual "
                         "scaling cannot be applied at trainer level", dmax)
            raise ValueError("recomposition mismatch")
    logger.info("[train_refiner] residual_scale_mode=%s residual_scale=%.6f"
                " gate_mode=%s", rs_mode, residual_scale, gate_mode)
    if gate_mode == "spatial":
        with torch.no_grad():
            inb = vaIn[:bs].to(device); x0b = vaX0[:bs].to(device)
            _, dx0, g0 = model(inb, x0b)
        if g0.shape != dx0.shape:
            logger.error("[train_refiner] spatial gate shape %s != dx %s",
                         tuple(g0.shape), tuple(dx0.shape))
            raise ValueError("spatial gate/dx shape mismatch")
        g_init_cfg = float(r.get("g_init", 0.05))
        gm0 = float(g0.mean())
        if not (0.8 * g_init_cfg <= gm0 <= 1.2 * g_init_cfg):
            logger.error("[train_refiner] initial spatial g_mean %.5f "
                         "outside [0.8,1.2]*g_init (%.5f)", gm0, g_init_cfg)
            raise ValueError("spatial gate init mean out of tolerance")
        n_sp = sum(p.numel() for p in model.gate_spatial.parameters())
        logger.info("[train_refiner] spatial gate: init g_mean=%.5f, "
                    "params=%d (recorded in status.json)", gm0, n_sp)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    epochs = int(cfg["train"]["epochs"])
    # v0.7: early stopping REQUIRED (EXEC SS7 policy). Best-relative rule:
    # improved iff val_dpsnr > best_dpsnr + min_delta; else stale += 1;
    # stop when stale >= patience. Keep-best retained. min_delta is a
    # STOPPING threshold (pre-registered prior, not noise-derived) -- it is
    # NOT the W0/W1 significance gate; duplicates measure that floor.
    for k in ("early_stop_patience", "early_stop_min_delta_dpsnr"):
        if k not in cfg["train"]:
            logger.error("[train_refiner] train.%s missing -- REQUIRED "
                         "(campaign: patience 15, min_delta 0.005 dB)", k)
            raise KeyError(f"train.{k} required")
    es_patience = int(cfg["train"]["early_stop_patience"])
    es_min_delta = float(cfg["train"]["early_stop_min_delta_dpsnr"])
    grad_clip = float(cfg["train"].get("grad_clip", 5.0))
    ch_eps = float(cfg["train"].get("charbonnier_eps", 1e-3))
    lam_b = float(cfg["train"].get("delta_budget_lambda", 1e-3))
    if cfg["train"].get("budget_form") not in ("dx", "g_dx"):
        logger.error("[train_refiner] train.budget_form required, one of "
                     "{'dx','g_dx'}, got %r", cfg["train"].get("budget_form"))
        raise ValueError("train.budget_form required: 'dx' or 'g_dx'")
    budget_form = cfg["train"]["budget_form"]
    def _budget(dx, g):
        if budget_form == "g_dx":
            return lam_b * (apply_gate(g, dx) ** 2).mean()
        return lam_b * (dx ** 2).mean()

    best_dpsnr = -float("inf"); best_epoch = -1; hist = []
    stale = 0; stopped_early = False; stop_epoch = None
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        run = 0.0; nb = 0; gn_sum = 0.0
        for xt, x0b, inp in tload:
            xt, x0b, inp = xt.to(device), x0b.to(device), inp.to(device)
            opt.zero_grad()
            x1, dx, g = model(inp, x0b)
            if residual_scale != 1.0:
                dx = residual_scale * dx
                x1 = x0b + apply_gate(g, dx)
            loss = _charbonnier(x1, xt, ch_eps) + _budget(dx, g)
            if not torch.isfinite(loss):
                logger.error("[train_refiner] non-finite loss")
                raise ValueError("non-finite loss")
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            run += loss.item(); nb += 1; gn_sum += float(gn)
        train_loss = run / nb
        m, x1v, dxv, gv = _val_metrics(model, val, base, bs, device,
                                       psnr_x0, ssim_x0, fwd_x0, val_perm,
                                       residual_scale=residual_scale)
        with torch.no_grad():
            val_loss = float(_charbonnier(x1v, vaX, ch_eps) + _budget(dxv, gv))
        base_gn = base.grad_max_abs()
        if stage1 is not None:
            base_gn = max(base_gn, stage1.grad_max_abs())
        if base_gn != 0.0:
            logger.error("[train_refiner] frozen base/stage-1 has grads! "
                         "max_abs=%.3e", base_gn)
            raise RuntimeError("frozen base/stage-1 received gradients")
        row = {"epoch": ep, "train_loss": train_loss, "val_loss": val_loss,
               **m, "base_grad_norm": base_gn,
               "refiner_grad_norm": gn_sum / nb}
        hist.append(row)
        logger.info("[train_refiner] ep %d loss=%.5f dpsnr=%+.3f fwd=%.4f/%.4f"
                    " g=%.3f ygap=%+.3f atrgap=%+.3f", ep, train_loss,
                    m["val_dpsnr"], m["val_fwd_rel_x0"], m["val_fwd_rel_x1"],
                    m["g_mean"], m["y_gap_dpsnr"], m["atr_gap_dpsnr"])
        if m["val_dpsnr"] > best_dpsnr + es_min_delta:
            best_dpsnr = m["val_dpsnr"]; best_epoch = ep; stale = 0
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_dpsnr": best_dpsnr}, ckpt_path)
        else:
            stale += 1
            if stale >= es_patience:
                stopped_early = True; stop_epoch = ep
                logger.info("[train_refiner] EARLY STOP at ep %d: no "
                            "val_dpsnr improvement > %.4f dB for %d checks "
                            "(best %.4f @ ep %d)", ep, es_min_delta,
                            es_patience, best_dpsnr, best_epoch)
                break
    if best_epoch < 0:
        logger.error("[train_refiner] no best epoch recorded")
        raise RuntimeError("no keep-best checkpoint")

    # best-epoch artifacts
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    m, x1v, dxv, gv = _val_metrics(model, val, base, bs, device,
                                   psnr_x0, ssim_x0, fwd_x0, val_perm,
                                   residual_scale=residual_scale)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    with open(os.path.join(run_dir, "metrics.csv"), "w") as f:
        f.write(",".join(_CSV_COLS) + "\n")
        for h in hist:
            f.write(",".join(f"{h[c]:.6f}" if isinstance(h[c], float)
                             else str(h[c]) for c in _CSV_COLS) + "\n")
    _plots(hist, run_dir)
    _extra_plots(dxv, gv, vaX, vaX0, x1v, run_dir, model.g_max)
    _save_per_sample_artifacts(dxv, gv, vaX, vaX0, x1v, residual_scale,
                               run_dir, model.g_max)
    _recon_grid(val, x1v, run_dir)

    dps = _psnr_per_sample(x1v, vaX) - _psnr_per_sample(vaX0, vaX)
    diag_ok = (m["y_gap_dpsnr"] > 0 and m["atr_gap_dpsnr"] > 0)
    if stage == 1:
        hard_pass = (m["val_dpsnr"] > _GATE_DPSNR
                     and m["val_fwd_rel_x1"] <= m["val_fwd_rel_x0"])
        meaningful_pass = None
    else:
        # Stage-2 two-tier gate (pre-registered BEFORE any x2 numbers):
        #   HARD:       dpsnr >= +0.30 AND fwd_rel <= prev AND ssim >= prev AND diag
        #   MEANINGFUL: dpsnr >= +0.10 AND per-sample mean > 0 AND %improved > 55%
        #               AND fwd_rel <= prev AND diag
        hard_pass = (m["val_dpsnr"] >= _GATE_DPSNR
                     and m["val_fwd_rel_x1"] <= m["val_fwd_rel_x0"]
                     and m["val_ssim_x1"] >= m["val_ssim_x0"]
                     and diag_ok)
        meaningful_pass = (m["val_dpsnr"] >= _GATE2_MEANINGFUL
                           and float(dps.mean()) > 0.0
                           and float((dps > 0).float().mean()) > _GATE2_PCT
                           and m["val_fwd_rel_x1"] <= m["val_fwd_rel_x0"]
                           and diag_ok)
    write_json(os.path.join(run_dir, "status.json"), {
        "refiner_expert": flavor, "seed_index": seed_index,
        "rng_seed": rng_seed, "cfg_hash": chash,
        "init_mode": init_mode, "recon_seed": recon_seed,
        "diag_perm_seed": diag_perm_seed,
        "replicate": replicate,
        "gate_mode": gate_mode,
        "gate_spatial_params": (sum(p.numel() for p in
                                    model.gate_spatial.parameters())
                                if gate_mode == "spatial" else 0),
        "total_params": sum(p.numel() for p in model.parameters()),
        "residual_scale_mode": rs_mode,
        "residual_scale": residual_scale,
        "residual_scale_n_train": int(trX.size(0)),
        "residual_scale_formula": "mean_i rms_pixels(x_true_i - x0_i)",
        "stopped_early": stopped_early, "stop_epoch": stop_epoch,
        "epochs_completed": len(hist), "best_epoch": best_epoch,
        "best_val_dpsnr": best_dpsnr,
        "early_stop_patience": es_patience,
        "early_stop_min_delta_dpsnr": es_min_delta,
        "base_expert": base.expert,
        "base_checkpoint_path": base.checkpoint_path,
        "base_checkpoint_sha256": base.checkpoint_sha256,
        "base_cfg_hash": base.cfg_hash, "base_frozen": True,
        "base_grad_max_abs": base.grad_max_abs(),
        "x0_mode": "posterior_pixel_mean", "x0_n_post": n_post,
        "stage": stage,
        "stage1": (None if stage1 is None else
                   {"run_dir": stage1_cfg["run_dir"], "flavor": stage1.flavor,
                    "checkpoint_sha256": stage1.checkpoint_sha256,
                    "frozen": True}),
        "budget_form": budget_form,
        "warm_start": ws_audit,
        "best_epoch": best_epoch, "best_val_dpsnr": m["val_dpsnr"],
        "best_val_psnr_x0": m["val_psnr_x0"],
        "best_val_psnr_x1": m["val_psnr_x1"],
        "best_val_ssim_x0": m["val_ssim_x0"],
        "best_val_ssim_x1": m["val_ssim_x1"],
        "best_val_fwd_rel_x0": m["val_fwd_rel_x0"],
        "best_val_fwd_rel_x1": m["val_fwd_rel_x1"],
        "best_g_mean": m["g_mean"], "best_g_max_frac": m["g_max_frac"],
        "best_y_gap_dpsnr": m["y_gap_dpsnr"],
        "best_atr_gap_dpsnr": m["atr_gap_dpsnr"],
        "pct_samples_improved": float((dps > 0).float().mean()),
        "gate_hard_pass": bool(hard_pass), "gate_diag_ok": bool(diag_ok),
        "gate_meaningful_pass": (None if meaningful_pass is None
                                 else bool(meaningful_pass)),
        "refiner_checkpoint_sha256": sha256_file(ckpt_path),
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": device, "torch_version": torch.__version__,
        "train_time_sec": round(time.time() - t0, 1), "status": "done",
    })
    prev = "x0(base)" if stage == 1 else f"x1(stage1:{stage1.flavor})"
    print("=== seed0 summary ({} {}) ===".format(flavor, stage_tag))
    print(f"dPSNR mean {float(dps.mean()):+.4f} median {float(dps.median()):+.4f}"
          f"  %improved {float((dps > 0).float().mean()) * 100:.1f}%")
    print(f"fwd_rel prev {m['val_fwd_rel_x0']:.4f} -> new {m['val_fwd_rel_x1']:.4f}")
    print(f"g_mean {m['g_mean']:.4f}  g_max_frac {m['g_max_frac']:.4f}")
    print(f"y_gap {m['y_gap_dpsnr']:+.4f}  atr_gap {m['atr_gap_dpsnr']:+.4f}")
    tier = "" if meaningful_pass is None else f"   MEANINGFUL: {meaningful_pass}"
    print(f"HARD PASS: {hard_pass}{tier}   DIAGNOSTIC OK: {diag_ok}   vs {prev}")
    logger.info("[train_refiner] DONE best dpsnr=%+.3f @ep %d", best_dpsnr,
                best_epoch)


if __name__ == "__main__":
    main()
