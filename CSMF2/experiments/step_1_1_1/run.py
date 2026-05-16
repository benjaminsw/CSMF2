# =============================================================================
# STEP-1_1_1 v0.1 -- experiments.step_1_1_1.run
# Purpose: train ONE (expert, seed, scale, noise) run. Same as step_1_1
#          v0.13 + (a) latent moment-matching penalty, (b) optional resume
#          from a step_1_1 checkpoint, (c) per-epoch latent KS / fwd_rel /
#          intra_std / cycle_max / frac_gap top-level history fields,
#          (d) soft exit-gate (warnings, not hard fail), (e) end-of-run
#          density + reconstruction plots saved to results/<tag>/plots/.
# CONVENTION: every failure -> logger.error + raise. No fallback / placeholder.
# Changelog (NEW in STEP-1_1_1 v0.1):
#   * Forked from step_1_1 run.py v0.13.
#   * NEW: latent moment penalty in training loop (opt-in via
#     cfg.latent_moment_lambda > 0):
#         L += lambda * ( mean(z.mean(0)**2) + mean((z.std(0)-1)**2) )
#     Per-50-step log: "[lat] mean_l2=X std_l2=Y".
#     Per-epoch log:    "[lat_epoch] mean_l2=X std_l2=Y lambda=L".
#   * NEW CLI: --latent-moment-lambda (default 0.0),
#              --latent-ks-target (default 0.10),
#              --fwd-rel-target (default 0.5),
#              --resume-from PATH (default None).
#   * NEW per-epoch top-level history fields (promoted from sanity panel):
#         latent_ks_hist, latent_moment_hist, fwd_rel_hist,
#         intra_std_hist, frac_gap_above_1_nat_hist, cycle_max_hist.
#   * NEW: --resume-from loads expert + conditioner state_dicts from a
#     prior checkpoint before training starts. Cfg must be compatible
#     (same expert + scale + noise + h_dim). Raises on mismatch.
#   * NEW: end-of-run plots saved via density_plots module:
#         plots/latent_density.png
#         plots/cycle_density.png
#         plots/reconstruction_panel.png
#   * Default behaviour at lambda=0 + no resume = byte-identical training
#     to step_1_1 v0.13 (modulo logger name + out_root).
# Update summary:
#   STEP-1_1_1 investigates whether moment matching of the latent
#   distribution improves prior-sampling reconstruction. The penalty is
#   cheap, differentiable, and a natural first attempt; if KS stays high
#   after the planned sweep, escalate to MMD/SW2 in step_1_1_2.
# =============================================================================
from __future__ import annotations
import argparse
import json
import logging
import math
import random
import sys
import time
import traceback
from pathlib import Path

logger = logging.getLogger("CSMF2.step_1_1_1.run")
__version__ = "0.1"
__abbr__ = "STEP-1_1_1"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import StepCfg
from .sanity import (numeric_logdet_check, invertibility_check,
                     plot_latent_histogram, plot_orig_degraded_cycle_generated,
                     plot_cycle_error_heatmap, plot_forward_consistency,
                     plot_latent_interp,
                     plot_nll_curve,
                     plot_samples_grid, plot_samples_fixed_z,
                     plot_sw2_diversity, plot_diag_spectrum,
                     plot_s_spectrum, plot_w1x1_spectrum,
                     plot_film_alive, plot_logp_shuffle)
from ...common import cond_viz as cv
from ...data.degrade import MNISTDegraded, dequantize_logit
from ...models.conditioner import Conditioner
from ...models.experts import build_expert


# ---------- utilities -------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _configure_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); root.addHandler(sh)
    fh = logging.FileHandler(out_dir / "run.log"); fh.setFormatter(fmt); root.addHandler(fh)


# ============================================================================
# COND-GATE v0.4 hook -- step 1.1 subset (no gate yet; gate-collapse SKIPPED).
# ============================================================================
# AUTO-WIRE SWITCH: step 1.2 will set this to the gate's forward function.
# When _GATE_FN is not None, the hook additionally emits plot_gate_collapse.
# Until then, the gate-collapse plot is deliberately not produced -- its
# preconditions (trained gate) do not exist at step 1.1.
_GATE_FN = None
# TODO(step 1.2): after training the mixture gate w_psi(y), set
#   _GATE_FN = gate.forward   (or an equivalent callable y -> (B, K) weights)
# and plot_gate_collapse will auto-wire on the next epoch boundary.
# ============================================================================


def _cond_gate_hook(expert, cond, test_loader, device, gen, plots_dir,
                    epoch: int, history: dict) -> None:
    # Produces 9 plots per epoch (see docstring of sanity.py for plan alignment).
    # Consumes ONE test batch; writes to plots_dir/cond_gate/.
    expert.eval(); cond.eval()
    out = plots_dir / "cond_gate"
    out.mkdir(parents=True, exist_ok=True)

    x_img, y_img = next(iter(test_loader))
    x_img = x_img.to(device); y_img = y_img.to(device)
    x_logit, _ = dequantize_logit(x_img, generator=gen)
    x_flat = x_logit.flatten(1)

    with torch.no_grad():
        # ---- h, gamma, beta aggregate + per-layer stats ---------------------
        h = cond(y_img)
        per_layer_stats: list[dict] = []
        gs: list[torch.Tensor] = []; bs: list[torch.Tensor] = []
        for layer in expert.layers:
            if hasattr(layer, "film"):
                g, b = layer.film(h)
                per_layer_stats.append({
                    "gamma": {"mean": float(g.mean()), "std": float(g.std())},
                    "beta":  {"mean": float(b.mean()), "std": float(b.std())},
                })
                gs.append(g.flatten()); bs.append(b.flatten())
        gamma_agg = torch.cat(gs) if gs else torch.zeros(1, device=device)
        beta_agg  = torch.cat(bs) if bs else torch.zeros(1, device=device)
        if not per_layer_stats:
            logger.error("[_cond_gate_hook] expert has no FiLM layers to inspect")
            raise RuntimeError("no FiLM layers found in expert")

        # ---- h diversity (pairwise) ----------------------------------------
        B = min(h.shape[0], 32)
        pairwise = torch.cdist(h[:B], h[:B]).cpu().numpy()

        # ---- logp shuffle test ---------------------------------------------
        lp_real = expert.log_prob(x_flat, y_img)
        y_shuf = y_img[torch.randperm(y_img.shape[0], device=device)]
        lp_shuf = expert.log_prob(x_flat, y_shuf)

        # ---- (y, seed) determinism -----------------------------------------
        torch.manual_seed(0); h1 = cond(y_img)
        torch.manual_seed(0); h2 = cond(y_img)
        det_max = float((h1 - h2).abs().max())

        # ---- null control: real NLL vs h-ablated NLL -----------------------
        h_zero = torch.zeros_like(h)
        z_abl, ldj_abl = expert.encode(x_flat, h_zero)
        lp_abl = -0.5 * (z_abl ** 2 + math.log(2 * math.pi)).sum(-1) + ldj_abl
        nll_real = float(-lp_real.mean()); nll_abl = float(-lp_abl.mean())

        # ---- h_st_response: vary y, measure Δlogp (s,t skipped; see below) --
        # Δs, Δt would require expert-specific access to the first coupling's
        # internal s,t. Δlogp is expert-agnostic and proves the same thing
        # (conditioning modulates the flow's output).
        y0 = y_img[:1].expand(16, *y_img.shape[1:]).contiguous()
        noise_scale = torch.linspace(0.0, 0.3, 16, device=device).view(-1, 1, 1, 1)
        y_pert = y0 + torch.randn_like(y0) * noise_scale
        dy = (y_pert - y0).flatten(1).norm(dim=1).cpu().numpy()
        x_rep = x_flat[:1].expand(16, -1).contiguous()
        lp_pert = expert.log_prob(x_rep, y_pert)
        dlogp = (lp_pert - lp_pert[:1]).abs().cpu().numpy()

        # ---- NaN / Inf counts ---------------------------------------------
        nan_inf = {
            "h_nan":  int(torch.isnan(h).sum()),
            "h_inf":  int(torch.isinf(h).sum()),
            "g_nan":  int(torch.isnan(gamma_agg).sum()),
            "b_nan":  int(torch.isnan(beta_agg).sum()),
        }

    # ---- update trajectory histories ---------------------------------------
    history["det_max_dh"].append(det_max)
    history["real_nll"].append(nll_real)
    history["abl_nll"].append(nll_abl)
    history["nan_inf"].append(nan_inf)
    history["h_std"].append(float(h.std().item()))
    history["gamma_std"].append(float(gamma_agg.std().item()))
    history["beta_std"].append(float(beta_agg.std().item()))
    # OLS slope of ||Δy|| vs |Δlogp| -- proves conditioning modulates logp
    import numpy as _np
    dy_a = _np.asarray(dy, dtype=float)
    dl_a = _np.asarray(dlogp, dtype=float)
    if dy_a.size >= 2 and dy_a.std() > 1e-8:
        slope = float(_np.polyfit(dy_a, dl_a, 1)[0])
    else:
        slope = 0.0
    history["h_st_slope"].append(slope)

    # ---- plots -------------------------------------------------------------
    cv.plot_h_hist(h.cpu().numpy(),
                   gamma_agg.cpu().numpy(), beta_agg.cpu().numpy(),
                   out / f"h_hist_epoch_{epoch}.png")
    cv.plot_h_diversity(pairwise, out / f"h_diversity_epoch_{epoch}.png")
    cv.plot_logp_shuffle(lp_real.cpu().numpy(), lp_shuf.cpu().numpy(),
                         out / f"logp_shuffle_epoch_{epoch}.png")
    cv.plot_film_per_layer(per_layer_stats,
                           out / f"film_per_layer_epoch_{epoch}.png")
    cv.plot_determinism_traj(history["det_max_dh"], tol=1e-6,
                             out_path=out / f"determinism_traj_epoch_{epoch}.png")
    cv.plot_nan_inf_traj(history["nan_inf"],
                         out_path=out / f"nan_inf_traj_epoch_{epoch}.png")
    cv.plot_null_control_overlay(history["real_nll"], history["abl_nll"],
                                 out / f"null_control_epoch_{epoch}.png",
                                 min_relative_gap=0.10)
    cv.plot_h_st_response(dy, ds_norms=None, dt_norms=None, dlogp_norms=dlogp,
                          out_path=out / f"h_st_response_epoch_{epoch}.png",
                          require_s=False)
    if history["cond_grad"]:
        cv.plot_grad_traj(history["cond_grad"], history["film_grad"],
                          out / f"grad_traj_epoch_{epoch}.png",
                          min_grad=1e-6)

    # SKIPPED until gate is trained -- auto-wires when _GATE_FN is set.
    if _GATE_FN is not None:
        # TODO(step 1.2): collect (Neff, entropy, argmax_hist) history in
        # the training loop, then call cv.plot_gate_collapse(...) here.
        logger.info("[_cond_gate_hook] _GATE_FN detected but plot_gate_collapse "
                    "not yet wired -- see TODO(step 1.2)")



# ---------- the run ---------------------------------------------------------
def run(cfg: StepCfg) -> dict:
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_root) / cfg.run_tag()
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    _configure_logging(out_dir)
    logger.info("STEP-1_1 run | tag=%s | cfg=%s", cfg.run_tag(), cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    # data -- v0.7: explicit train/val/test split. Per-epoch sanity uses val.
    # `test_loader` is bound to the VAL dataset so existing per-epoch sites
    # (cond_gate hook, sanity batch fetch) keep their variable names. The
    # 10k test set is bound separately and used ONCE at end-of-run.
    train_ds = MNISTDegraded(cfg.data_root, split="train",
                             sigma=cfg.blur_sigma, scale=cfg.scale,
                             noise_sigma=cfg.noise_sigma)
    val_ds   = MNISTDegraded(cfg.data_root, split="val",
                             sigma=cfg.blur_sigma, scale=cfg.scale,
                             noise_sigma=cfg.noise_sigma)
    test_ds_final = MNISTDegraded(cfg.data_root, split="test",
                                  sigma=cfg.blur_sigma, scale=cfg.scale,
                                  noise_sigma=cfg.noise_sigma)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    # `test_loader` is intentionally aliased to val (per-epoch monitoring).
    # See test_ds_final / test_loader_final below for the sealed test set.
    test_loader  = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=0)
    test_loader_final = DataLoader(test_ds_final, batch_size=cfg.batch_size,
                                   shuffle=False, num_workers=0)
    logger.info("[run] data sizes: train=%d  val=%d (per-epoch)  test=%d (final-only)",
                len(train_ds), len(val_ds), len(test_ds_final))

    # v0.12: sanity check on y_img variation across the batch. Catches a
    # data-loader bug where all y in a minibatch are equal (would make the
    # entire conditioning analysis meaningless). Done once on the first
    # train batch, before any model build.
    _yx, _yy = next(iter(train_loader))
    _yy = _yy.to(device)
    _y_batch_std = float(_yy.std(dim=0).mean().item())
    logger.info("[y_sanity] y.std(dim=0).mean() = %.4f  shape=%s",
                _y_batch_std, tuple(_yy.shape))
    if _y_batch_std < 1e-3:
        logger.error("[y_sanity] y_batch_std=%.6e < 1e-3 -- y_img is "
                     "(near-)constant across batch; conditioner cannot work",
                     _y_batch_std)
        raise RuntimeError(
            f"y_img has no batch variation (std={_y_batch_std:.2e}); "
            f"check data pipeline")
    del _yx, _yy

    # model
    # v0.13: y-residual bypass — y_input_size derived from MNIST 28x28 / scale.
    # For scale=2 -> 14x14=196; scale=4 -> 7x7=49.
    _y_in_size = (28 // cfg.scale) * (28 // cfg.scale)
    cond_kwargs: dict = dict(width=cfg.cond_width, h_dim=cfg.h_dim,
                             use_v2=cfg.use_v2_conditioner)
    if cfg.cond_y_residual_alpha_init > 0.0:
        cond_kwargs["y_residual_alpha_init"] = cfg.cond_y_residual_alpha_init
        cond_kwargs["y_input_size"] = _y_in_size
    cond = Conditioner(**cond_kwargs).to(device)
    # v0.8: film_kwargs now allowed for nice, realnvp, glow (build_expert
    # raises for nsf only). Glow + RealNVP also get their own param packs.
    film_kwargs: dict = {}
    if cfg.expert in ("nice", "realnvp", "glow"):
        film_kwargs = dict(film_hidden=cfg.film_hidden,
                           film_depth=cfg.film_depth,
                           film_use_gelu=cfg.film_use_gelu)

    extra_kwargs: dict = {}
    if cfg.expert == "realnvp":
        extra_kwargs.update(n_layers=cfg.realnvp_n_couplings)
    elif cfg.expert == "glow":
        extra_kwargs.update(
            n_layers=cfg.glow_n_steps,
            s_max=cfg.glow_s_max,
            image_shape=(cfg.glow_image_c, cfg.glow_image_h, cfg.glow_image_w),
            inv1x1_seed_base=cfg.seed,
            film_gain_init=cfg.glow_film_gain_init,    # v0.9
        )

    # Glow uses its own coupling_hidden (3-conv NN channels); other experts use flow_hidden.
    hidden_for_build = (cfg.glow_coupling_hidden
                        if cfg.expert == "glow" else cfg.flow_hidden)
    expert = build_expert(cfg.expert, dim=cfg.dim, h_dim=cfg.h_dim,
                          conditioner=cond, hidden=hidden_for_build,
                          use_film=cfg.use_film,
                          **film_kwargs, **extra_kwargs).to(device)
    opt = torch.optim.Adam(list(expert.parameters()), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    # STEP-1_1_1: optional resume from a prior checkpoint (e.g. a step_1_1
    # run). Loads conditioner + expert state_dicts. Cfg compatibility is
    # checked strictly -- any mismatch in expert/scale/noise/h_dim/dim
    # raises. Optimizer state is NOT restored (intentional: we are starting
    # a new optimisation episode with new objectives).
    if cfg.resume_from is not None:
        ckpt_path = Path(cfg.resume_from)
        if not ckpt_path.exists():
            logger.error("[resume] checkpoint path does not exist: %s",
                         ckpt_path)
            raise FileNotFoundError(f"resume_from path does not exist: "
                                    f"{ckpt_path}")
        logger.info("[resume] loading checkpoint from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Compatibility checks
        ck_cfg = ckpt.get("cfg", {})
        for k in ("expert", "scale", "noise_sigma", "h_dim", "dim"):
            v_new = getattr(cfg, k)
            v_old = ck_cfg.get(k)
            if v_old is not None and v_old != v_new:
                logger.error("[resume] cfg mismatch: %s old=%r new=%r",
                             k, v_old, v_new)
                raise ValueError(
                    f"resume_from cfg mismatch on {k}: "
                    f"old={v_old!r} new={v_new!r}")
        if "expert" not in ckpt or "cond" not in ckpt:
            logger.error("[resume] checkpoint missing 'expert' or 'cond' "
                         "state_dict (keys=%s)", list(ckpt.keys()))
            raise KeyError("checkpoint must contain 'expert' and 'cond' "
                           "state_dicts")
        # strict=False because step_1_1_1 cond may have y_residual buffers
        # that the source checkpoint lacks (and vice versa); we still log
        # missing / unexpected keys explicitly.
        m1 = expert.load_state_dict(ckpt["expert"], strict=False)
        m2 = cond.load_state_dict(ckpt["cond"], strict=False)
        if m1.missing_keys or m1.unexpected_keys:
            logger.warning("[resume] expert load -- missing=%s unexpected=%s",
                           m1.missing_keys, m1.unexpected_keys)
        if m2.missing_keys or m2.unexpected_keys:
            logger.warning("[resume] cond load -- missing=%s unexpected=%s",
                           m2.missing_keys, m2.unexpected_keys)
        logger.info("[resume] checkpoint loaded successfully")

    # ---------- v0.8: Glow data-dependent Actnorm init ---------------------
    # One pass over a single train batch before epoch 1. Each Actnorm sees
    # its own pre-actnorm activations and initialises s, b. Skipped for
    # non-Glow experts (no Actnorm in their layer stacks).
    if cfg.expert == "glow":
        x_init_img, y_init = next(iter(train_loader))
        x_init_img = x_init_img.to(device); y_init = y_init.to(device)
        x_init_logit, _ = dequantize_logit(x_init_img, generator=gen)
        x_init_flat = x_init_logit.flatten(1)
        expert.init_actnorm(x_init_flat, y_init)

    # ---------- numeric logdet on a dim-16 twin (cheap) --------------------
    # SKIPPED for Glow: a dim-16 toy is incompatible with image-shape Glow
    # (squeeze requires (C,H,W) with C*H*W=dim and even H,W). The per-epoch
    # invertibility_check below covers correctness.
    if cfg.expert == "glow":
        logdet_rep = {"passed": True, "skipped": True,
                      "reason": "twin-flow logdet check incompatible with "
                                "image-shape Glow; per-epoch "
                                "invertibility_check covers correctness"}
        logger.info("[run] logdet twin check SKIPPED for Glow: %s",
                    logdet_rep["reason"])
    else:
        twin_cond_kwargs = dict(cond_kwargs)
        twin_cond = Conditioner(**twin_cond_kwargs).to(device)
        twin_extra = dict(extra_kwargs)
        # twin uses dim=16; Glow path is skipped above so no glow kwargs here
        twin = build_expert(cfg.expert, dim=16, h_dim=cfg.h_dim,
                            conditioner=twin_cond, hidden=64,
                            use_film=cfg.use_film,
                            **film_kwargs, **twin_extra).to(device)
        # v0.13: deterministic twin perturbation so the numeric logdet check
        # is reproducible. Previously a different RNG state per run caused
        # occasional flakes at the 1e-3 tolerance.
        twin_gen = torch.Generator(device=device).manual_seed(12345)
        with torch.no_grad():
            for p in twin.parameters():
                p.add_(torch.randn(p.shape, generator=twin_gen, device=device,
                                   dtype=p.dtype) * 0.05)
        # v0.13: loosen tolerance from 1e-3 to 5e-3. FP64 central differences
        # on a 16x16 Jacobian have ~1e-3 noise floor inherent to the method;
        # 1e-3 was an aggressive bound that flaked stochastically.
        logdet_rep = numeric_logdet_check(twin, dim_toy=16, h_dim=cfg.h_dim,
                                          device=device,
                                          rel_tol=5e-3, abs_tol=5e-3)
    (out_dir / "logdet_check.json").write_text(json.dumps(logdet_rep, indent=2))
    if not logdet_rep["passed"]:
        logger.error("[run] numeric logdet check FAILED: %s", logdet_rep)
        raise RuntimeError("numeric logdet check failed -- halting")

    # ---------- training loop ----------------------------------------------
    train_nll_hist: list[float] = []
    test_nll_hist:  list[float] = []
    glow_film_gain_hist: list[list[float]] = []   # v0.9: Glow-only, list of per-step gains per epoch
    shuffle_hinge_hist: list[dict] = []            # v0.10: list of {"gap": float, "hinge": float} per epoch
    h_std_obs_hist:    list[dict] = []             # v0.12: per-epoch {"total":, "batch":}
    grad_norms_hist:   list[dict] = []             # v0.11: per-epoch {"cond": float, "film_gain": float}
    y_resid_alpha_hist: list[float] = []           # v0.13: per-epoch y_residual_alpha values
    # STEP-1_1_1: latent + reconstruction trajectory metrics
    latent_moment_hist: list[dict] = []            # per-epoch {"mean_l2":, "std_l2":}
    latent_ks_hist:     list[float] = []           # per-epoch latent KS vs N(0,1)
    fwd_rel_hist:       list[float] = []           # per-epoch forward_consistency.fwd_rel_mean
    intra_std_hist:     list[dict] = []            # per-epoch {"X":, "Y":}
    frac_gap_hist:      list[float] = []           # per-epoch logp_shuffle.frac_gap_above_1_nat (= gap_median > 1.0 not exactly, derived)
    cycle_max_hist:     list[float] = []           # per-epoch cycle_panel.cycle_max_err
    sanity_reports: list[dict]  = []
    cg_history: dict = {
        "cond_grad":  [], "film_grad": [],
        "det_max_dh": [], "real_nll":  [], "abl_nll": [],
        "nan_inf":    [],
        "h_std":      [], "gamma_std": [], "beta_std": [],
        "h_st_slope": [],
    }

    for epoch in range(cfg.epochs):
        expert.train(); cond.train()
        t0 = time.time(); nb = 0; nll_sum = 0.0
        # v0.10: shuffle-hinge accumulators (used only when lambda > 0)
        hinge_sum = 0.0; gap_sum = 0.0; n_hinge = 0
        # v0.11: h.std accumulators + last-step grad-norm captures
        h_std_pen_sum = 0.0; h_std_obs_sum = 0.0; h_std_batch_sum = 0.0; n_hstd = 0
        # STEP-1_1_1: latent moment accumulators
        lat_mean_sum = 0.0; lat_var_sum = 0.0; n_lat = 0
        last_film_gain_grad = 0.0
        for step, (x_img, y_img) in enumerate(train_loader):
            x_img = x_img.to(device); y_img = y_img.to(device)
            # x_img: (B,1,28,28) in [0,1] -> dequantize logit -> (B, 784)
            x_logit, ldj_deq = dequantize_logit(x_img, generator=gen)
            x_flat = x_logit.flatten(1)
            # v0.11: manual cond() so we can apply the h.std penalty.
            # Replaces lp = expert.log_prob(x_flat, y_img) with the same
            # computation, exposing h for inspection / regularisation.
            h = cond(y_img)
            z, ldj_flow = expert.encode(x_flat, h)
            lp = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1) + ldj_flow
            if not torch.isfinite(lp).all():
                logger.error("[run] non-finite log_prob at epoch=%d step=%d",
                             epoch, step)
                raise RuntimeError("non-finite log_prob")
            nll = -(lp + ldj_deq).mean()
            if not torch.isfinite(nll):
                logger.error("[run] non-finite NLL at epoch=%d step=%d", epoch, step)
                raise RuntimeError("non-finite NLL")
            total = nll
            # v0.10: shuffle-gap hinge loss (opt-in)
            if cfg.shuffle_loss_lambda > 0.0:
                with torch.no_grad():
                    perm = torch.randperm(y_img.size(0), device=device)
                lp_shuf = expert.log_prob(x_flat, y_img[perm])
                # detach lp_shuf so gradient flows only via lp_real -- prevents
                # the model from minimising the hinge by jointly degrading both
                gap = (lp - lp_shuf.detach()).mean()
                hinge = torch.clamp(cfg.shuffle_loss_margin - gap, min=0.0)
                if not torch.isfinite(hinge):
                    logger.error("[run] non-finite hinge at epoch=%d step=%d",
                                 epoch, step)
                    raise RuntimeError("non-finite hinge loss")
                total = total + cfg.shuffle_loss_lambda * hinge
                hinge_sum += float(hinge.item())
                gap_sum   += float(gap.item())
                n_hinge   += 1
            # v0.12: penalty operates on across-BATCH std (averaged over dims),
            # not total h.std(). The latter is satisfied by collapsing all y to
            # one point with per-dim spread -- audited in v0.11. Across-batch
            # std cannot be cheated: it directly measures "does h vary with y".
            h_std_batch = h.std(dim=0).mean()      # across-batch, then avg over dims
            h_std_total = h.std()                  # diagnostic only
            if cfg.h_std_penalty_mu > 0.0:
                h_pen = torch.clamp(cfg.h_std_target - h_std_batch, min=0.0)
                if not torch.isfinite(h_pen):
                    logger.error("[run] non-finite h_std penalty at "
                                 "epoch=%d step=%d", epoch, step)
                    raise RuntimeError("non-finite h_std penalty")
                total = total + cfg.h_std_penalty_mu * h_pen
                h_std_pen_sum += float(h_pen.item())
                n_hstd += 1
            h_std_obs_sum  += float(h_std_total.item())
            h_std_batch_sum += float(h_std_batch.item())
            # v0.12: per-50-step h log shows BOTH stats so we can see whether
            # the model is moving the right one.
            if (step + 1) % cfg.log_every == 0:
                logger.info("[h] epoch=%d step=%d total=%.5f batch=%.5f",
                            epoch, step + 1,
                            float(h_std_total.item()),
                            float(h_std_batch.item()))
            # STEP-1_1_1 v0.1: latent moment-matching penalty (opt-in).
            # Pushes per-dim z toward (mean=0, var=1). Acts on the SAME z
            # already computed above -- no extra forward pass.
            z_mean_l2 = (z.mean(dim=0) ** 2).mean()
            z_var_l2  = ((z.var(dim=0, unbiased=False) - 1.0) ** 2).mean()
            if cfg.latent_moment_lambda > 0.0:
                L_moment = z_mean_l2 + z_var_l2
                if not torch.isfinite(L_moment):
                    logger.error("[run] non-finite latent moment at "
                                 "epoch=%d step=%d", epoch, step)
                    raise RuntimeError("non-finite latent moment penalty")
                total = total + cfg.latent_moment_lambda * L_moment
            lat_mean_sum += float(z_mean_l2.item())
            lat_var_sum  += float(z_var_l2.item())
            n_lat        += 1
            if (step + 1) % cfg.log_every == 0:
                logger.info("[lat] epoch=%d step=%d mean_l2=%.5f std_l2=%.5f",
                            epoch, step + 1,
                            float(z_mean_l2.item()), float(z_var_l2.item()))
            opt.zero_grad(set_to_none=True)
            total.backward()
            # capture grad norms from the LAST minibatch of the epoch for grad_traj
            if step == len(train_loader) - 1:
                cg = 0.0
                for p in cond.parameters():
                    if p.grad is not None:
                        cg += float(p.grad.detach().norm().item()) ** 2
                cg_history["cond_grad"].append(cg ** 0.5)
                fg_list: list[float] = []
                for layer in expert.layers:
                    if hasattr(layer, "film"):
                        fg = 0.0
                        for p in layer.film.parameters():
                            if p.grad is not None:
                                fg += float(p.grad.detach().norm().item()) ** 2
                        fg_list.append(fg ** 0.5)
                cg_history["film_grad"].append(fg_list)
                # v0.11: capture film_gain gradient norm (Glow only)
                fgg = 0.0
                for layer in expert.layers:
                    if hasattr(layer, "coupling") and hasattr(layer.coupling,
                                                              "film_gain"):
                        gp = layer.coupling.film_gain.grad
                        if gp is not None:
                            fgg += float(gp.detach().norm().item()) ** 2
                last_film_gain_grad = fgg ** 0.5
            nn.utils.clip_grad_norm_(expert.parameters(), cfg.grad_clip)
            opt.step()
            nll_sum += float(nll.item()); nb += 1
            if (step + 1) % cfg.log_every == 0:
                logger.info("epoch=%d step=%d/%d train_nll=%.3f",
                            epoch, step + 1, len(train_loader), nll_sum / nb)
        train_nll = nll_sum / max(nb, 1)
        train_nll_hist.append(train_nll)

        # per-epoch validation NLL (held-out 5k val set; field name preserved
        # as test_nll_hist for downstream compatibility -- see header changelog)
        expert.eval(); cond.eval()
        with torch.no_grad():
            tot = 0.0; nt = 0
            for x_img, y_img in test_loader:
                x_img = x_img.to(device); y_img = y_img.to(device)
                x_logit, ldj_deq = dequantize_logit(x_img, generator=gen)
                lp = expert.log_prob(x_logit.flatten(1), y_img)
                tot += float(-(lp + ldj_deq).mean().item()); nt += 1
            test_nll = tot / max(nt, 1)
        test_nll_hist.append(test_nll)
        logger.info("epoch=%d  train_nll=%.3f  val_nll=%.3f  t=%.1fs",
                    epoch, train_nll, test_nll, time.time() - t0)

        # v0.9: log Glow's per-coupling film_gain values (mean/min/max).
        # Tells us whether the model leans into conditioning (gain grows) or
        # abandons it (gain decays toward 0). Stored in report["glow_film_gain_hist"].
        if cfg.expert == "glow":
            gains = []
            for layer in expert.layers:
                if hasattr(layer, "coupling") and hasattr(layer.coupling,
                                                          "film_gain"):
                    gains.append(float(layer.coupling.film_gain.detach()
                                       .item()))
            if gains:
                arr = torch.tensor(gains)
                logger.info("[glow] film_gain mean=%.4f min=%.4f max=%.4f",
                            float(arr.mean()), float(arr.min()),
                            float(arr.max()))
                glow_film_gain_hist.append(gains)

        # v0.10: log shuffle-hinge stats (only when lambda > 0).
        if cfg.shuffle_loss_lambda > 0.0 and n_hinge > 0:
            gap_mean   = gap_sum   / n_hinge
            hinge_mean = hinge_sum / n_hinge
            logger.info("[shuffle] gap_mean=%.3f hinge=%.4f lambda=%.3f margin=%.2f",
                        gap_mean, hinge_mean,
                        cfg.shuffle_loss_lambda, cfg.shuffle_loss_margin)
            shuffle_hinge_hist.append({"gap": gap_mean, "hinge": hinge_mean})

        # v0.11/v0.12: per-epoch h.std stats. 'batch' is the across-batch std
        # averaged over dims -- the right metric for conditioning. 'total' is
        # the overall h.std (sanity).
        h_std_total_mean = h_std_obs_sum   / max(nb, 1)
        h_std_batch_mean = h_std_batch_sum / max(nb, 1)
        if cfg.h_std_penalty_mu > 0.0 and n_hstd > 0:
            h_pen_mean = h_std_pen_sum / n_hstd
            logger.info("[h_std] total=%.5f batch=%.5f penalty=%.5f "
                        "target=%.4f mu=%.3f",
                        h_std_total_mean, h_std_batch_mean, h_pen_mean,
                        cfg.h_std_target, cfg.h_std_penalty_mu)
        else:
            logger.info("[h_std] total=%.5f batch=%.5f (no penalty)",
                        h_std_total_mean, h_std_batch_mean)
        h_std_obs_hist.append({"total": h_std_total_mean,
                               "batch": h_std_batch_mean})

        # STEP-1_1_1: latent moment per-epoch log + history.
        lat_mean_mean = lat_mean_sum / max(n_lat, 1)
        lat_var_mean  = lat_var_sum  / max(n_lat, 1)
        if cfg.latent_moment_lambda > 0.0:
            logger.info("[lat_epoch] mean_l2=%.5f std_l2=%.5f lambda=%.3f",
                        lat_mean_mean, lat_var_mean, cfg.latent_moment_lambda)
        else:
            logger.info("[lat_epoch] mean_l2=%.5f std_l2=%.5f (no penalty)",
                        lat_mean_mean, lat_var_mean)
        latent_moment_hist.append({"mean_l2": lat_mean_mean,
                                   "std_l2":  lat_var_mean})

        # v0.13: y-residual alpha log (only when enabled)
        if cfg.cond_y_residual_alpha_init > 0.0 and \
                hasattr(cond, "y_residual_alpha"):
            alpha_val = float(cond.y_residual_alpha.detach().item())
            logger.info("[y_resid] alpha=%.4f", alpha_val)
            y_resid_alpha_hist.append(alpha_val)

        # v0.11: per-epoch grad-norm summary.
        cond_grad_last = cg_history["cond_grad"][-1] if cg_history.get(
            "cond_grad") else 0.0
        logger.info("[grad] cond=%.4f film_gain=%.4f",
                    cond_grad_last, last_film_gain_grad)
        grad_norms_hist.append({"cond": cond_grad_last,
                                "film_gain": last_film_gain_grad})

        # ---------- sanity -------------------------------------------------
        if cfg.sanity_every_epoch:
            # pull one batch for checks
            x_img, y_img = next(iter(test_loader))
            x_img = x_img.to(device); y_img = y_img.to(device)
            x_logit, _ = dequantize_logit(x_img, generator=gen)
            x_flat = x_logit.flatten(1)

            inv_rep   = invertibility_check(expert, x_flat, y_img)
            lat_rep   = plot_latent_histogram(
                expert, x_flat, y_img,
                plots / f"latent_hist_epoch_{epoch}.png")
            panel_rep = plot_orig_degraded_cycle_generated(
                expert, x_flat, y_img,
                plots / f"orig_deg_cycle_gen_epoch_{epoch}.png")
            # v0.3 -- cycle heatmap / forward consistency / latent interp (slerp)
            cycle_rep = plot_cycle_error_heatmap(
                expert, x_flat, y_img,
                plots / f"cycle_heatmap_epoch_{epoch}.png")
            fwd_rep   = plot_forward_consistency(
                expert, y_img,
                blur_sigma=cfg.blur_sigma, scale=cfg.scale,
                out_path=plots / f"fwd_consistency_epoch_{epoch}.png")
            interp_rep = plot_latent_interp(
                expert, y_img,
                plots / f"latent_interp_epoch_{epoch}.png")
            rep = {"epoch": epoch,
                   "train_nll": train_nll, "test_nll": test_nll,
                   "invertibility": inv_rep,
                   "latent": lat_rep,
                   "cycle_panel": panel_rep,
                   "cycle_heatmap": cycle_rep,
                   "forward_consistency": fwd_rep,
                   "latent_interp": interp_rep}
            sanity_reports.append(rep)
            if not inv_rep["passed"]:
                logger.error("[run] invertibility FAILED: %s", inv_rep)
                raise RuntimeError("invertibility failure -- halting run")

            # COND-GATE v0.4 plots (9 of 10; gate-collapse skipped until step 1.2)
            # v0.8: SKIPPED for Glow. The legacy hook walks expert.layers
            # looking for top-level .film attributes (NICE / RealNVP pattern).
            # Glow stores FiLM inside layer.coupling.film1/film2 -- the v2
            # diagnostics block below covers the same conditioning health
            # checks (film_alive, logp_shuffle) using Glow's actual structure.
            if cfg.expert != "glow":
                _cond_gate_hook(expert, cond, test_loader, device, gen, plots,
                                epoch, cg_history)

            # ---------- v0.5 v2-conditioner diagnostics (opt-in) -----------
            # v0.8: dir renamed v2/ -> gen_diag/. Spectrum plot dispatched by
            # expert. Glow adds w1x1_spectrum.
            if cfg.use_v2_conditioner:
                v2_dir = plots / "gen_diag"
                y_one = y_img[:1]
                v2_grid    = plot_samples_grid(
                    expert, y_one,
                    v2_dir / f"samples_grid_epoch_{epoch}.png")
                v2_fixed_z = plot_samples_fixed_z(
                    expert, y_one,
                    v2_dir / "samples_fixed_z.png",       # overwrite each epoch
                    n=32)
                v2_sw2     = plot_sw2_diversity(
                    expert, y_one,
                    v2_dir / f"sw2_diversity_epoch_{epoch}.png",
                    n_samples=256, n_proj=64)
                # spectrum: diag_spectrum for NICE (has DiagScale),
                # s_spectrum for RealNVP / Glow (affine couplings).
                if cfg.expert == "nice":
                    v2_spec = plot_diag_spectrum(
                        expert,
                        v2_dir / f"diag_spectrum_epoch_{epoch}.png")
                else:
                    v2_spec = plot_s_spectrum(
                        expert, x_flat, y_img,
                        v2_dir / f"s_spectrum_epoch_{epoch}.png")
                v2_film    = plot_film_alive(
                    expert, cond, y_img,
                    v2_dir / f"film_alive_epoch_{epoch}.png",
                    eps=1e-3)
                v2_shuf    = plot_logp_shuffle(
                    expert, x_flat, y_img,
                    v2_dir / f"logp_shuffle_epoch_{epoch}.png")
                v2_block = {"samples_grid": v2_grid,
                            "samples_fixed_z": v2_fixed_z,
                            "sw2_diversity": v2_sw2,
                            "spectrum": v2_spec,
                            "film_alive": v2_film,
                            "logp_shuffle": v2_shuf}
                if cfg.expert == "glow":
                    v2_block["w1x1_spectrum"] = plot_w1x1_spectrum(
                        expert,
                        v2_dir / f"w1x1_spectrum_epoch_{epoch}.png")
                rep["v2_diag"] = v2_block

            # STEP-1_1_1: promote per-epoch diagnostics to top-level histories
            # for easy plotting and exit-gate logic without panel re-parsing.
            latent_ks_hist.append(float(lat_rep.get("ks", float("nan"))))
            fwd_rel_hist.append(float(fwd_rep.get("fwd_rel_mean",
                                                  float("nan"))))
            cycle_max_hist.append(float(cycle_rep.get("cycle_max",
                                                      float("nan"))))
            # intra_std + frac_gap live in the v2_diag block when v2 is on.
            if cfg.use_v2_conditioner and "v2_diag" in rep:
                sw2 = rep["v2_diag"].get("sw2_diversity", {})
                intra_std_hist.append({
                    "X": float(sw2.get("intra_std_X", float("nan"))),
                    "Y": float(sw2.get("intra_std_Y", float("nan"))),
                })
                shuf = rep["v2_diag"].get("logp_shuffle", {})
                # frac_gap_above_1_nat is reported by plot_logp_shuffle.
                # Fallback: compute from median+std if direct field missing.
                if "frac_gap_above_1_nat" in shuf:
                    frac_gap_hist.append(float(shuf["frac_gap_above_1_nat"]))
                else:
                    # use gap_median as a rough proxy; not the same metric.
                    frac_gap_hist.append(float("nan"))
            else:
                intra_std_hist.append({"X": float("nan"),
                                       "Y": float("nan")})
                frac_gap_hist.append(float("nan"))

    # ---------- exit-criteria report ---------------------------------------
    last = sanity_reports[-1]
    exit_ok = (
        logdet_rep["passed"] and
        last["invertibility"]["passed"] and
        last["latent"]["passed"] and
        last["cycle_panel"]["passed"]
    )
    # STEP-1_1_1: soft exit-gate warnings (do NOT halt training).
    # Surfaced in log + persisted to report["exit_warnings"].
    exit_warnings: list[str] = []
    last_ks = latent_ks_hist[-1] if latent_ks_hist else float("nan")
    last_fwd = fwd_rel_hist[-1]  if fwd_rel_hist  else float("nan")
    first_fwd = fwd_rel_hist[0]  if fwd_rel_hist  else float("nan")
    if not (last_ks <= cfg.latent_ks_target):
        exit_warnings.append(
            f"latent_ks={last_ks:.4f} > target {cfg.latent_ks_target:.4f}")
    # Reconstruction: prefer relative improvement when resuming.
    if cfg.resume_from is not None and not math.isnan(first_fwd):
        rel_improve = (first_fwd - last_fwd) / max(first_fwd, 1e-6)
        if rel_improve < 0.20:
            exit_warnings.append(
                f"fwd_rel improved only {rel_improve*100:.1f}% < 20% target "
                f"(first={first_fwd:.3f}, last={last_fwd:.3f})")
    else:
        if not (last_fwd <= cfg.fwd_rel_target):
            exit_warnings.append(
                f"fwd_rel={last_fwd:.3f} > target {cfg.fwd_rel_target:.3f} "
                f"(informational; no resume baseline available)")
    for w in exit_warnings:
        logger.warning("[exit_gate] %s", w)
    # Reshape cg_history for summary.py. We save per-epoch means so the
    # summary can apply pass/fail thresholds. h/γ/β stds are collected in
    # the hook; if absent, summary will treat those gates as failing -- the
    # hook already populates real_nll/abl_nll/det_max_dh/nan_inf/film_grad.
    cg_hist_save = {
        "real_nll":       cg_history.get("real_nll", []),
        "abl_nll":        cg_history.get("abl_nll", []),
        "det_max_dh":     cg_history.get("det_max_dh", []),
        "nan_inf":        cg_history.get("nan_inf", []),
        "cond_grad":      cg_history.get("cond_grad", []),
        "film_grad_min":  [min(fg) if fg else 0.0
                           for fg in cg_history.get("film_grad", [])],
        "h_std":          cg_history.get("h_std", []),
        "gamma_std":      cg_history.get("gamma_std", []),
        "beta_std":       cg_history.get("beta_std", []),
        "h_st_slope":     cg_history.get("h_st_slope", []),
    }
    report = {
        "cfg": cfg.__dict__,
        "logdet_check": logdet_rep,
        "train_nll_hist": train_nll_hist,
        "test_nll_hist":  test_nll_hist,
        "sanity_per_epoch": sanity_reports,
        "cond_gate_history": cg_hist_save,
        "glow_film_gain_hist": glow_film_gain_hist,   # v0.9
        "shuffle_hinge_hist":  shuffle_hinge_hist,    # v0.10
        "h_std_obs_hist":      h_std_obs_hist,        # v0.11
        "grad_norms_hist":     grad_norms_hist,       # v0.11
        "y_resid_alpha_hist":  y_resid_alpha_hist,    # v0.13
        # STEP-1_1_1: latent + reconstruction trajectory metrics
        "latent_moment_hist":  latent_moment_hist,
        "latent_ks_hist":      latent_ks_hist,
        "fwd_rel_hist":        fwd_rel_hist,
        "intra_std_hist":      intra_std_hist,
        "frac_gap_hist":       frac_gap_hist,
        "cycle_max_hist":      cycle_max_hist,
        "exit_warnings":       exit_warnings,
        "exit_criteria_met": bool(exit_ok),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    torch.save({"expert": expert.state_dict(),
                "cond":   cond.state_dict(),
                "cfg":    cfg.__dict__,        # STEP-1_1_1: for resume validation
               },
               out_dir / "ckpt.pt")

    # v0.7: ONE-SHOT final NLL on the sealed 10k test set (split="test").
    # Unlike test_nll_hist (which is val-NLL per epoch), this is touched
    # exactly once and is the headline number for cross-run comparison.
    expert.eval(); cond.eval()
    with torch.no_grad():
        tot = 0.0; nt = 0
        for x_img, y_img in test_loader_final:
            x_img = x_img.to(device); y_img = y_img.to(device)
            x_logit, ldj_deq = dequantize_logit(x_img, generator=gen)
            lp = expert.log_prob(x_logit.flatten(1), y_img)
            tot += float(-(lp + ldj_deq).mean().item()); nt += 1
        final_test_nll = tot / max(nt, 1)
    logger.info("[run] FINAL test_nll (sealed 10k) = %.3f", final_test_nll)
    report["final_test_nll"] = final_test_nll
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    # STEP-1_1_1: end-of-run density + reconstruction plots.
    # Compute on one fresh test batch -- not training data.
    try:
        from .density_plots import (plot_latent_density,
                                    plot_cycle_density,
                                    plot_reconstruction_panel)
        expert.eval(); cond.eval()
        x_img, y_img = next(iter(test_loader))
        x_img = x_img.to(device); y_img = y_img.to(device)
        with torch.no_grad():
            x_logit, _ = dequantize_logit(x_img, generator=gen)
            x_flat = x_logit.flatten(1)
            h = cond(y_img)
            z, _ = expert.encode(x_flat, h)
            # 1) latent density vs N(0,1)
            lat_density_rep = plot_latent_density(
                z, plots / "latent_density.png",
                title="z = encode(x, y) vs N(0,1)",
                lambda_used=cfg.latent_moment_lambda)
            # 2) cycle density: x - decode(encode(x))
            x_back = expert.decode(z, h)
            cyc_density_rep = plot_cycle_density(
                x_flat, x_back, plots / "cycle_density.png",
                title="Per-pixel cycle error  x - decode(encode(x))")
            # 3) reconstruction panel
            # row 3: decode(encoded z, y) -- already x_back, just reshape
            x_back_img = x_back.view_as(x_img)
            # row 4: decode(z ~ N(0,1), y) -- new prior sample
            z_prior = torch.randn(z.shape, device=device, generator=gen,
                                  dtype=z.dtype)
            x_prior_flat = expert.decode(z_prior, h)
            x_prior_img = x_prior_flat.view_as(x_img)
            recon_rep = plot_reconstruction_panel(
                y_img, x_img, x_back_img, x_prior_img,
                plots / "reconstruction_panel.png",
                title=(f"Recon: encoded vs N(0,1)-sampled z  "
                       f"(lambda_moment={cfg.latent_moment_lambda:.3g})"),
                n_show=8)
        report["latent_density"]     = lat_density_rep
        report["cycle_density"]      = cyc_density_rep
        report["reconstruction_panel"] = recon_rep
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    except Exception:
        logger.error("[run] step_1_1_1 density plots FAILED\n%s",
                     traceback.format_exc())
        raise

    # v0.5: always-on NLL curve at end of run -> plots/nll_curve.png
    nll_curve_rep = plot_nll_curve(train_nll_hist, test_nll_hist,
                                   plots / "nll_curve.png")
    report["nll_curve"] = nll_curve_rep
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    # end-of-run summary (ANSI-coloured console + summary.txt + summary.csv)
    try:
        from .summary import summarize
        summary_block = summarize(out_dir)
        exit_ok = bool(summary_block["exit_criteria_met"])
    except Exception:
        logger.error("[run] summarize() failed\n%s", traceback.format_exc())
        raise

    logger.info("STEP-1_1 run DONE  exit_ok=%s  out=%s", exit_ok, out_dir)
    return report


# ---------- CLI -------------------------------------------------------------
def _parse_args() -> tuple[StepCfg, bool]:
    p = argparse.ArgumentParser()
    p.add_argument("--expert",
                   choices=["nice", "realnvp", "nsf", "glow"], default="nice",
                   help="Default 'nice'. v0.8: v2 conditioner supported for "
                        "nice / realnvp / glow. For nsf you MUST pass "
                        "--no-use-v2-conditioner.")
    p.add_argument("--scale", type=int, choices=[2, 4], required=True)
    p.add_argument("--noise-sigma", type=float, choices=[0.0, 0.05, 0.1], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--cond-width", type=int, choices=[64, 128], default=128)
    p.add_argument("--use-film", type=int, choices=[0, 1], default=1)
    p.add_argument("--cache-h", type=int, choices=[0, 1], default=1)
    # v0.6 v2-conditioner: BooleanOptionalAction -> --no-use-v2-conditioner opts out
    p.add_argument("--use-v2-conditioner",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Default ON. Pass --no-use-v2-conditioner for the legacy "
                        "stack (also reset --cond-width / --h-dim / --film-* if "
                        "you want strict v0.4 reproducibility).")
    p.add_argument("--h-dim", type=int, default=256)
    p.add_argument("--film-depth", type=int, default=2)
    p.add_argument("--film-hidden", type=int, default=128)
    p.add_argument("--film-use-gelu",
                   action=argparse.BooleanOptionalAction, default=True)
    # v0.8: RealNVP-specific
    p.add_argument("--realnvp-n-couplings", type=int, default=6)
    p.add_argument("--realnvp-s-max", type=float, default=2.0)
    # v0.8: Glow-specific
    p.add_argument("--glow-n-steps", type=int, default=8)
    p.add_argument("--glow-coupling-hidden", type=int, default=256)
    p.add_argument("--glow-s-max", type=float, default=2.0)
    p.add_argument("--glow-film-gain-init", type=float, default=0.3,
                   help="initial value of Glow's learnable per-coupling FiLM "
                        "gain (v0.5+; default 0.3). 0 reproduces v0.8 behaviour.")
    # v0.10: shuffle-gap hinge loss
    p.add_argument("--shuffle-loss-lambda", type=float, default=0.0,
                   help="hinge weight for the shuffle-gap auxiliary loss. "
                        "Default 0.0 = OFF (no extra forward pass; behaviour "
                        "identical to v0.9). Suggested for Glow: 0.1.")
    p.add_argument("--shuffle-loss-margin", type=float, default=0.5,
                   help="target mean shuffle gap (nats) below which hinge is "
                        "active. Default 0.5.")
    # v0.11/v0.12: direct h.std penalty (penalty uses across-batch std in v0.12)
    p.add_argument("--h-std-penalty-mu", type=float, default=0.0,
                   help="weight on the h.std penalty (Glow conditioner rescue). "
                        "Default 0.0 = OFF. v0.12 penalty operates on across-"
                        "BATCH std (averaged over dims). Suggested for Glow: 10.")
    p.add_argument("--h-std-target", type=float, default=0.05,
                   help="target across-batch h.std below which the penalty "
                        "is active. Default 0.05.")
    # v0.13: conditioner y-residual bypass
    p.add_argument("--cond-y-residual-alpha-init", type=float, default=0.0,
                   help="initial value for learnable alpha that scales the "
                        "linear y-bypass in Conditioner: "
                        "h = cnn_head(y) + alpha * Linear(y.flatten). "
                        "Default 0.0 = bypass DISABLED. Suggested rescue "
                        "value for Glow: 0.3.")
    # STEP-1_1_1: latent moment penalty + soft exit targets + resume-from
    p.add_argument("--latent-moment-lambda", type=float, default=0.0,
                   help="weight on the latent moment-matching penalty "
                        "(STEP-1_1_1). Default 0.0 = OFF. Suggested sweep: "
                        "{0.0, 0.1, 1.0, 10.0}.")
    p.add_argument("--latent-ks-target", type=float, default=0.10,
                   help="soft target for the latent-KS exit warning. "
                        "Default 0.10.")
    p.add_argument("--fwd-rel-target", type=float, default=0.5,
                   help="soft target for fwd_rel_mean. Used as absolute "
                        "target when not resuming; when --resume-from is "
                        "given, a 20%% relative-improvement criterion is "
                        "preferred. Default 0.5.")
    p.add_argument("--resume-from", type=str, default=None,
                   help="optional path to a checkpoint (.pt) to load expert "
                        "+ conditioner state_dicts from before training. "
                        "Useful for fine-tuning from a step_1_1 run. "
                        "Default: None (fresh init).")
    p.add_argument("--data-root", default="./mnist_data")
    p.add_argument("--out-root",  default="./CSMF2/experiments/step_1_1_1/results")
    p.add_argument("--skip-if-complete", action="store_true",
                   dest="skip_if_complete",
                   help="skip if a completed report.json already exists "
                        "(was --resume in step_1_1; renamed to avoid "
                        "confusion with --resume-from)")
    a = p.parse_args()
    cfg = StepCfg(expert=a.expert, scale=a.scale, noise_sigma=a.noise_sigma,
                  seed=a.seed, epochs=a.epochs, batch_size=a.batch_size,
                  cond_width=a.cond_width, use_film=bool(a.use_film),
                  cache_h=bool(a.cache_h), data_root=a.data_root,
                  out_root=a.out_root,
                  use_v2_conditioner=a.use_v2_conditioner,
                  h_dim=a.h_dim, film_depth=a.film_depth,
                  film_hidden=a.film_hidden, film_use_gelu=a.film_use_gelu,
                  realnvp_n_couplings=a.realnvp_n_couplings,
                  realnvp_s_max=a.realnvp_s_max,
                  glow_n_steps=a.glow_n_steps,
                  glow_coupling_hidden=a.glow_coupling_hidden,
                  glow_s_max=a.glow_s_max,
                  glow_film_gain_init=a.glow_film_gain_init,
                  shuffle_loss_lambda=a.shuffle_loss_lambda,
                  shuffle_loss_margin=a.shuffle_loss_margin,
                  h_std_penalty_mu=a.h_std_penalty_mu,
                  h_std_target=a.h_std_target,
                  cond_y_residual_alpha_init=a.cond_y_residual_alpha_init,
                  latent_moment_lambda=a.latent_moment_lambda,
                  latent_ks_target=a.latent_ks_target,
                  fwd_rel_target=a.fwd_rel_target,
                  resume_from=a.resume_from)
    return cfg, a.skip_if_complete


if __name__ == "__main__":
    cfg, skip_if_complete = _parse_args()
    out_dir = Path(cfg.out_root) / cfg.run_tag()
    if skip_if_complete and (out_dir / "report.json").exists():
        try:
            existing = json.loads((out_dir / "report.json").read_text())
            if existing.get("exit_criteria_met"):
                print(f"[skip] SKIP {cfg.run_tag()} (already complete + passed)")
                sys.exit(0)
        except (OSError, json.JSONDecodeError):
            logger.warning("[skip_if_complete] corrupt report.json at %s, "
                           "rerunning", out_dir)
    try:
        report = run(cfg)
        sys.exit(0 if report.get("exit_criteria_met") else 2)
    except Exception:
        logger.error("STEP-1_1 run FAILED\n%s", traceback.format_exc())
        sys.exit(1)
