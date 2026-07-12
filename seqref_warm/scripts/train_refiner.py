# SEQREF-TRNREF v0.4 -- train_refiner
# LIFETIME: KEEP
# Phase 6: train ONE candidate refiner (flavor realnvp|nice) against the same
# frozen base x0. Loss = Charbonnier(x1, x_true) + lambda_budget * mean(dx^2).
# Keep-best on val_dpsnr. x0/inputs precomputed once via base_io cache.
# Gate (approved split):
#   HARD PASS:   val_dpsnr > +0.3 dB, val_fwd_rel_x1 <= val_fwd_rel_x0,
#                base_frozen verified (grad_max_abs == 0)
#   DIAGNOSTIC:  y_gap_dpsnr > 0, atr_gap_dpsnr > 0 (evidence, required
#                non-contradictory for 3-seed promotion)
# No NLL, no lambda_rec. No fallback/mock/pass; failures logger.error + raise.
# Changelog (TRNREF v0.4 -> v0.4-fseq, SEQREF-FSEQ W2):
#   * seqref_warm fork; dataset construction via degrade.make_degraded
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

from seqref_warm.src.degrade import make_degraded
from seqref_warm.src.metrics import psnr as _psnr, ssim as _ssim, fwd_rel as _fwd_rel
from seqref_warm.src.refiners.base_io import (FrozenBase, FrozenStage1,
                                               precompute_split,
                                               precompute_stage2)
from seqref_warm.src.refiners.coupling_regressor import (CplRegRefiner,
                                                          load_warm_start,
                                                          DEFAULT_EXCLUDE)
from seqref_warm.src.refiners.gated_update import GatedUpdate
from seqref_warm.src.train_utils import (setup_logger, seed_from_index,
                                          cfg_hash, write_json, sha256_file)

logger = setup_logger("seqref_warm.train_refiner")
__version__ = "0.4"

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


@torch.no_grad()
def _forward_split(model, inputs, x0, bs, device):
    # Batched forward over a whole cached split. Returns x1, dx, g (cpu).
    xs, dxs, gs = [], [], []
    for i in range(0, inputs.size(0), bs):
        inp = inputs[i:i + bs].to(device)
        x0b = x0[i:i + bs].to(device)
        x1, dx, g = model(inp, x0b)
        xs.append(x1.cpu()); dxs.append(dx.cpu()); gs.append(g.cpu())
    return torch.cat(xs), torch.cat(dxs), torch.cat(gs)


@torch.no_grad()
def _val_metrics(model, val, base, bs, device, psnr_x0, ssim_x0, fwd_x0):
    # val = dict(x_true, y, x0, inputs). Returns full per-epoch metric dict.
    model.eval()
    x_true, y, x0, inputs = val["x_true"], val["y"], val["x0"], val["inputs"]
    x1, dx, g = _forward_split(model, inputs, x0, bs, device)
    x1c = x1.clamp(0, 1)
    psnr_x1 = _psnr(x1c, x_true); ssim_x1 = _ssim(x1c, x_true)
    fwd_x1 = _fwd_rel(x1c, y, base.blur_sigma, base.scale)
    gs = GatedUpdate.g_stats(g, model.g_max)
    delta_l2 = float(dx.flatten(1).norm(dim=1).mean())
    delta_linf = float(dx.flatten(1).abs().max(dim=1).values.mean())
    tgt = (x_true - x0)
    delta_tgt_l2 = float((dx - tgt).flatten(1).norm(dim=1).mean())

    # conditioning gaps: permute one input channel across the batch.
    perm = torch.randperm(inputs.size(0))
    def _gap(channel: int) -> tuple[float, float]:
        shuf = inputs.clone()
        shuf[:, channel] = inputs[perm, channel]
        x1s, _, _ = _forward_split(model, shuf, x0, bs, device)
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
             "delta_l2_mean", "delta_linf_mean", "delta_target_l2",
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


def _extra_plots(dx, g, x_true, x0, x1, run_dir):
    plt.figure(figsize=(6, 4))
    plt.hist(dx.flatten().numpy(), bins=100, alpha=0.6, label="Δx")
    plt.hist((g.view(-1, 1, 1, 1) * dx).flatten().numpy(), bins=100,
             alpha=0.6, label="g·Δx")
    plt.legend(); plt.yscale("log"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "delta_hist.png"), dpi=110); plt.close()

    dps = (_psnr_per_sample(x1, x_true) - _psnr_per_sample(x0, x_true)).numpy()
    plt.figure(figsize=(6, 4))
    plt.scatter(g.numpy(), dps, s=4, alpha=0.3)
    plt.xlabel("g"); plt.ylabel("per-sample ΔPSNR (dB)"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "scatter_dpsnr_vs_g.png"), dpi=110)
    plt.close()

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
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    seed_index = args.seed if args.seed is not None else int(cfg["train"]["seed"])
    rng_seed = seed_from_index(seed_index)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    r = cfg["refiner"]
    flavor = r["flavor"]
    base = FrozenBase(cfg["base"]["run_dir"], device)
    n_post = int(cfg["base"].get("n_post", 16))
    stage1_cfg = cfg.get("stage1")
    stage = 2 if (stage1_cfg or {}).get("run_dir") else 1
    chash = cfg_hash(cfg)
    stage_tag = "refine2" if stage == 2 else "refine"
    run_dir = os.path.join(cfg["output"]["root"],
                           f"{flavor}_{stage_tag}_s{base.scale}_n"
                           f"{float(base.cfg['cell']['noise_sigma']):.2f}_"
                           f"seed{seed_index}_{chash}")
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
                                            rng_seed=rng_seed,
                                            cache_dir=cache_dir,
                                            split_name="train", device=device)
    vaX, vaY, vaX0, vaIn = precompute_split(base, vl, n_post=n_post,
                                            rng_seed=rng_seed,
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
                          g_init=float(r.get("g_init", 0.05))).to(device)
    ws = cfg.get("warm_start", {})
    ws_audit = None
    if ws.get("path"):
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
    else:
        logger.error("[train_refiner] warm_start.path missing -- scratch runs "
                     "are deferred by plan (SEQREF-REFINE v0.2); refusing")
        raise ValueError("warm_start.path required (scratch A/B deferred)")

    tset = TensorDataset(trX, trX0, trIn)
    tload = DataLoader(tset, batch_size=bs, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    epochs = int(cfg["train"]["epochs"])
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
            return lam_b * ((g.view(-1, 1, 1, 1) * dx) ** 2).mean()
        return lam_b * (dx ** 2).mean()

    best_dpsnr = -float("inf"); best_epoch = -1; hist = []
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        run = 0.0; nb = 0; gn_sum = 0.0
        for xt, x0b, inp in tload:
            xt, x0b, inp = xt.to(device), x0b.to(device), inp.to(device)
            opt.zero_grad()
            x1, dx, g = model(inp, x0b)
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
                                       psnr_x0, ssim_x0, fwd_x0)
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
        if m["val_dpsnr"] > best_dpsnr:
            best_dpsnr = m["val_dpsnr"]; best_epoch = ep
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_dpsnr": best_dpsnr}, ckpt_path)
    if best_epoch < 0:
        logger.error("[train_refiner] no best epoch recorded")
        raise RuntimeError("no keep-best checkpoint")

    # best-epoch artifacts
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    m, x1v, dxv, gv = _val_metrics(model, val, base, bs, device,
                                   psnr_x0, ssim_x0, fwd_x0)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    with open(os.path.join(run_dir, "metrics.csv"), "w") as f:
        f.write(",".join(_CSV_COLS) + "\n")
        for h in hist:
            f.write(",".join(f"{h[c]:.6f}" if isinstance(h[c], float)
                             else str(h[c]) for c in _CSV_COLS) + "\n")
    _plots(hist, run_dir)
    _extra_plots(dxv, gv, vaX, vaX0, x1v, run_dir)
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
