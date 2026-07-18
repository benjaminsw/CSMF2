# SEQREF-TRNBASE v0.8 -- train_base
# LIFETIME: KEEP
# Phase 2: train one expert (NSF/RealNVP/NICE) as full reconstructor. Logit-space
# NLL, val early-stop, keep-best. Test-0 (sigma=0,scale=1) auto-segregated.
# No fallback/mock/pass. Failures logger.error + raise. recon_grid RAISES on
# decode failure (no silent skip) until f64 NSF decode is ported.
# Changelog (v0.7 -> v0.8, SEQREF-FSEQ W2):
#   * Dataset construction via degrade.make_degraded with REQUIRED
#     cell.dataset key ({mnist, fashion_mnist}; absent/unknown -> raise, no
#     silent default). Recorded cell block now includes the dataset key.
#     No other logic changes.
# Changelog (v0.6 -> v0.7, SEQREF-NICER3):
#   * Expert-specific model keys generalised to a map:
#     realnvp -> {s_max, post_init_std}; nice -> {use_permute, post_init_std}.
#     Keys present for the wrong expert -> explicit raise (unchanged policy).
#     use_permute cast to bool, numeric keys to float. Defaults preserved.
#   * s_clamp probe unchanged (realnvp only; nice writes nan columns).
# Changelog (v0.5 -> v0.6, SEQREF-SMAX):
#   * _build threads model.s_max / model.post_init_std for expert='realnvp'
#     ONLY; explicit raise if present for any other expert. Defaults preserved
#     when keys absent (s_max=2.0, post_init_std=0.0 -> exact v0.5 model).
#   * NEW per-epoch s_clamp probe (realnvp only): encode first val batch with
#     correct y, read per-layer last_s_clamp_frac. metrics.csv gains
#     s_clamp_mean,s_clamp_max (nan for other experts); status.json gains
#     s_clamp_mean_final + s_clamp_frac_layers_final at best epoch.
# Changelog (v0.4 -> v0.5):
#   * Gradient clipping (train.grad_clip); grad_norm tracked per epoch.
# Changelog (v0.3 -> v0.4):
#   * CCR conditioning pressure (shuffle-gap hinge + h_std penalty + y-residual).
# Changelog (v0.2 -> v0.3):
#   * recon_grid posterior PIXEL-mean; recon metrics in csv/status.
# Update summary:
#   v0.7 extends the SMAX hard-gating pattern to NICE for SEQREF-NICER3:
#   use_permute (R3, the A/B variable) and post_init_std (carry-over, both
#   arms) become config keys for expert='nice' only. RealNVP behaviour is
#   byte-identical to v0.6. No rec-loss term exists in this file.
from __future__ import annotations
import argparse
import logging
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from seqref_mri.src.degrade import (make_degraded, dequantize_logit,
                                      inverse_logit)
from seqref_mri.src.conditioner import Conditioner
from seqref_mri.src.base_experts import build_expert
from seqref_mri.src.metrics import mse as _mse, psnr as _psnr, fwd_rel as _fwd_rel
from seqref_mri.src.train_utils import (setup_logger, seed_from_index,
                                          cfg_hash, make_run_dir, write_json,
                                          sha256_file)

logger = setup_logger("seqref_mri.train_base")
__version__ = "0.7"

_FILM_KEYS = ("film_hidden", "film_depth", "film_use_gelu")
# Expert-specific model keys (SEQREF-SMAX + SEQREF-NICER3). A key listed for
# one expert raises if present in the config of any other expert.
_EXPERT_KEYS = {
    "realnvp": ("s_max", "post_init_std"),
    "nice":    ("use_permute", "post_init_std"),
}
_BOOL_KEYS = ("use_permute",)
_RECON_N_POST = 16


def _load_cfg(path: str) -> dict:
    if not os.path.isfile(path):
        logger.error("[train_base] config not found: %s", path)
        raise FileNotFoundError(path)
    with open(path) as f:
        return yaml.safe_load(f)


def _build(cfg: dict, device: str):
    m = cfg["model"]
    expert = cfg["expert"]
    dim = int(m["dim"])
    h_dim = int(m.get("h_dim", 128))
    scale = int(cfg["cell"]["scale"])
    # CCR: y-residual bypass (learnable direct y->h path the CNN cannot starve).
    ccr = cfg.get("ccr", {})
    cond_kwargs = dict(width=int(m.get("cond_width", 64)), h_dim=h_dim,
                       use_v2=bool(m.get("cond_use_v2", False)))
    alpha0 = float(ccr.get("cond_y_residual_alpha_init", 0.0))
    if alpha0 > 0.0:
        cond_kwargs["y_residual_alpha_init"] = alpha0
        cond_kwargs["y_input_size"] = (28 // scale) * (28 // scale)
    cond = Conditioner(**cond_kwargs).to(device)
    kw = {}
    if "n_layers" in m:
        kw["n_layers"] = int(m["n_layers"])
    if expert != "nsf":
        for k in _FILM_KEYS:
            if k in m:
                kw[k] = m[k]
    elif any(k in m for k in _FILM_KEYS):
        logger.error("[train_base] film_* keys not allowed for nsf: %s",
                     [k for k in _FILM_KEYS if k in m])
        raise ValueError("film_* keys not allowed for nsf")
    # SEQREF-SMAX/NICER3: expert-specific model keys, hard-gated.
    allowed = set(_EXPERT_KEYS.get(expert, ()))
    all_special = {k for keys in _EXPERT_KEYS.values() for k in keys}
    misplaced = [k for k in all_special if k in m and k not in allowed]
    if misplaced:
        logger.error("[train_base] keys %s not allowed for expert=%r "
                     "(allowed: %s)", misplaced, expert, sorted(allowed))
        raise ValueError(f"keys {misplaced} not allowed for expert={expert!r}")
    for k in allowed:
        if k in m:
            kw[k] = bool(m[k]) if k in _BOOL_KEYS else float(m[k])
    model = build_expert(expert, dim=dim, h_dim=h_dim, conditioner=cond,
                         hidden=int(m.get("hidden", 256)),
                         use_film=bool(m.get("use_film", True)), **kw).to(device)
    return model


@torch.no_grad()
def _posterior_pixel_mean(model, y, n_post: int):
    # For each y: sample n_post z~N(0,I), decode, inverse_logit (pixel), mean in
    # PIXEL space. Returns (x_hat (B,1,28,28), sample_pixel_std scalar). RAISES on
    # decode failure (no skip).
    B = y.size(0)
    h = model.cond(y)                                    # (B, h_dim)
    acc = torch.zeros(B, 1, 28, 28, device=y.device, dtype=y.dtype)
    sq = torch.zeros_like(acc)
    for _ in range(n_post):
        z = torch.randn(B, model.dim, device=y.device, dtype=y.dtype)
        try:
            x_logit = model.decode(z, h)
        except Exception:
            logger.error("[train_base] decode FAILED (NSF f64 decode not ported?)"
                         " -- raising, not skipping", exc_info=True)
            raise
        xp = inverse_logit(x_logit).view(B, 1, 28, 28).clamp(0, 1)
        acc += xp
        sq += xp * xp
    mean = acc / n_post
    var = (sq / n_post - mean * mean).clamp_min(0.0)
    return mean, float(var.sqrt().mean())


@torch.no_grad()
def _val_recon(model, loader, device, blur_sigma, scale, n_post):
    # Aggregate PSNR/MSE/fwd_rel/sample_pixel_std over the val set (pixel-mean x_hat).
    model.eval()
    tot_mse = tot_psnr = tot_fwd = tot_std = 0.0; n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        b = x.size(0)
        x_hat, sstd = _posterior_pixel_mean(model, y, n_post)
        tot_mse += _mse(x_hat, x) * b
        tot_psnr += _psnr(x_hat, x) * b
        tot_fwd += _fwd_rel(x_hat, y, blur_sigma, scale) * b
        tot_std += sstd * b
        n += b
    return (tot_mse / n, tot_psnr / n, tot_fwd / n, tot_std / n)


@torch.no_grad()
def _s_clamp_probe(model, loader, device, gen):
    # SEQREF-SMAX diagnostic (realnvp only): encode first val batch with CORRECT
    # y, then read per-layer last_s_clamp_frac (frac |s| > 0.99*s_max).
    # Runs standalone so the training shuffle pass cannot contaminate the stat.
    model.eval()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    x_logit, _ = dequantize_logit(x, generator=gen)
    h = model.cond(y)
    model.encode(x_logit.flatten(1), h)
    fracs = []
    for i, layer in enumerate(model.layers):
        if not hasattr(layer, "last_s_clamp_frac"):
            logger.error("[train_base] s_clamp probe: layer %d has no "
                         "last_s_clamp_frac (realnvp_layer < v0.3?)", i)
            raise AttributeError("layer missing last_s_clamp_frac")
        fracs.append(float(layer.last_s_clamp_frac))
    return fracs


def _nll(model, x_pix, y, gen) -> torch.Tensor:
    # x_pix: (B,1,28,28) in [0,1]. logit-dequantize, flatten, -(logprob+ldj).
    x_logit, ldj = dequantize_logit(x_pix, generator=gen)
    x_flat = x_logit.flatten(1)
    lp = model.log_prob(x_flat, y)
    return -(lp + ldj).mean()


def _loss_ccr(model, x_pix, y, gen, ccr, device):
    # NLL + CCR pressure. Manual cond() to expose h for the h_std penalty.
    # Returns (total, h_std_batch_float, gap_float_or_nan).
    x_logit, ldj_deq = dequantize_logit(x_pix, generator=gen)
    x_flat = x_logit.flatten(1)
    h = model.cond(y)
    z, ldj_flow = model.encode(x_flat, h)
    lp = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1) + ldj_flow
    if not torch.isfinite(lp).all():
        logger.error("[train_base] non-finite log_prob")
        raise RuntimeError("non-finite log_prob")
    total = -(lp + ldj_deq).mean()
    gap_f = float("nan")

    lam = float(ccr.get("shuffle_loss_lambda", 0.0))
    if lam > 0.0:
        with torch.no_grad():
            perm = torch.randperm(y.size(0), device=device)
        lp_shuf = model.log_prob(x_flat, y[perm])
        gap = (lp - lp_shuf.detach()).mean()   # detach: gradient only via lp
        hinge = torch.clamp(float(ccr.get("shuffle_loss_margin", 0.5)) - gap,
                            min=0.0)
        if not torch.isfinite(hinge):
            logger.error("[train_base] non-finite hinge")
            raise RuntimeError("non-finite hinge loss")
        total = total + lam * hinge
        gap_f = float(gap.item())

    h_std_batch = h.std(dim=0).mean()          # across-batch std, avg over dims
    mu = float(ccr.get("h_std_penalty_mu", 0.0))
    if mu > 0.0:
        h_pen = torch.clamp(float(ccr.get("h_std_target", 0.05)) - h_std_batch,
                            min=0.0)
        if not torch.isfinite(h_pen):
            logger.error("[train_base] non-finite h_std penalty")
            raise RuntimeError("non-finite h_std penalty")
        total = total + mu * h_pen
    if not torch.isfinite(total):
        logger.error("[train_base] non-finite total loss")
        raise ValueError("non-finite training loss")
    return total, float(h_std_batch.item()), gap_f


@torch.no_grad()
def _val_nll(model, loader, device, gen) -> float:
    model.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        b = x.size(0)
        tot += _nll(model, x, y, gen).item() * b
        n += b
    return tot / n


@torch.no_grad()
def _recon_grid(model, loader, device, out_png, blur_sigma, scale, n_post, k=8):
    # Rows: y_up | x_hat(posterior pixel-mean) | x_true | abs_err.
    model.eval()
    x, y = next(iter(loader))
    x, y = x[:k].to(device), y[:k].to(device)
    x_hat, _ = _posterior_pixel_mean(model, y, n_post)
    y_up = F.interpolate(y, size=(28, 28), mode="nearest")
    err = (x_hat - x).abs()
    rows = [y_up, x_hat, x, err]
    labels = ["y_up", "x_hat", "x_true", "abs_err"]
    fig, ax = plt.subplots(4, k, figsize=(k + 1, 4))
    for r in range(4):
        for c in range(k):
            ax[r, c].imshow(rows[r][c, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            ax[r, c].axis("off")
        ax[r, 0].text(-0.3, 0.5, labels[r], transform=ax[r, 0].transAxes,
                      ha="right", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close(fig)


def _nll_curve(hist, best_epoch, out_png):
    ep = [h["epoch"] for h in hist]
    plt.figure(figsize=(6, 4))
    plt.plot(ep, [h["train_nll"] for h in hist], label="train_nll")
    plt.plot(ep, [h["val_nll"] for h in hist], label="val_nll")
    plt.axvline(best_epoch, color="k", ls="--", lw=1, label=f"best@{best_epoch}")
    plt.xlabel("epoch"); plt.ylabel("nll"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=110); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="override seed_index")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    seed_index = args.seed if args.seed is not None else int(cfg["train"]["seed"])
    rng_seed = seed_from_index(seed_index)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cell = cfg["cell"]
    scale = int(cell["scale"]); blur_sigma = float(cell["blur_sigma"])
    noise_sigma = float(cell["noise_sigma"])
    test0 = (blur_sigma == 0.0 and scale == 1)

    chash = cfg_hash(cfg)
    run_dir = make_run_dir(cfg["output"]["root"], expert=cfg["expert"],
                           scale=scale, noise_sigma=noise_sigma,
                           seed_index=seed_index, cfg_hash_hex=chash, test0=test0)
    logger.info("[train_base] expert=%s seed=%d rng=%d test0=%s dir=%s",
                cfg["expert"], seed_index, rng_seed, test0, run_dir)

    root = cell["data_root"]
    dk = dict(sigma=blur_sigma, scale=scale, noise_sigma=noise_sigma)
    tr = make_degraded(cell.get("dataset"), root, split="train", **dk)
    va = make_degraded(cell.get("dataset"), root, split="val", **dk)
    bs = int(cfg["train"]["batch_size"])
    tl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)
    vl = DataLoader(va, batch_size=bs, shuffle=False, num_workers=2)

    model = _build(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    gen = torch.Generator(device=device).manual_seed(rng_seed)

    epochs = int(cfg["train"]["epochs"])
    min_delta = float(cfg["train"].get("early_stop_min_delta", 1e-3))
    grad_clip = float(cfg["train"].get("grad_clip", 5.0))
    ccr = cfg.get("ccr", {})
    ccr_on = float(ccr.get("shuffle_loss_lambda", 0.0)) > 0.0 or \
        float(ccr.get("h_std_penalty_mu", 0.0)) > 0.0
    probe_sclamp = (cfg["expert"] == "realnvp")
    best_val = float("inf"); best_epoch = -1; hist = []
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        run = 0.0; nb = 0; hstd_sum = 0.0; gap_sum = 0.0; ng = 0; gn_sum = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss, hstd_b, gap_f = _loss_ccr(model, x, y, gen, ccr, device)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            run += loss.item(); nb += 1; hstd_sum += hstd_b; gn_sum += float(gn)
            if gap_f == gap_f:  # not NaN
                gap_sum += gap_f; ng += 1
        tr_nll = run / nb
        h_std_ep = hstd_sum / nb
        gap_ep = (gap_sum / ng) if ng else float("nan")
        grad_ep = gn_sum / nb
        val = _val_nll(model, vl, device, gen)
        if probe_sclamp:
            sc_layers = _s_clamp_probe(model, vl, device, gen)
            sc_mean = sum(sc_layers) / len(sc_layers)
            sc_max = max(sc_layers)
        else:
            sc_layers = []
            sc_mean = float("nan"); sc_max = float("nan")
        hist.append({"epoch": ep, "train_nll": tr_nll, "val_nll": val,
                     "lr": opt.param_groups[0]["lr"],
                     "h_std_batch": h_std_ep, "shuffle_gap": gap_ep,
                     "grad_norm": grad_ep,
                     "s_clamp_mean": sc_mean, "s_clamp_max": sc_max,
                     "s_clamp_layers": sc_layers})
        logger.info("[train_base] ep %d train=%.3f val=%.3f h_std=%.4f gap=%.3f "
                    "grad=%.2f s_clamp=%.3f/%.3f", ep, tr_nll, val, h_std_ep,
                    gap_ep, grad_ep, sc_mean, sc_max)
        if val < best_val - min_delta:
            best_val = val; best_epoch = ep
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_nll": val}, ckpt_path)

    if best_epoch < 0:
        logger.error("[train_base] no epoch improved val by min_delta=%.4g", min_delta)
        raise RuntimeError("training produced no keep-best checkpoint")

    # reload best checkpoint, then compute recon metrics once (16x decode is costly)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    v_mse, v_psnr, v_fwd, v_sstd = _val_recon(model, vl, device, blur_sigma,
                                              scale, _RECON_N_POST)
    logger.info("[train_base] recon@best psnr=%.3f mse=%.5f fwd_rel=%.4f pstd=%.4f",
                v_psnr, v_mse, v_fwd, v_sstd)

    # artifacts
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    with open(os.path.join(run_dir, "config_hash.txt"), "w") as f:
        f.write(chash + "\n")
    with open(os.path.join(run_dir, "metrics.csv"), "w") as f:
        f.write("epoch,train_nll,val_nll,lr,h_std_batch,shuffle_gap,grad_norm,"
                "s_clamp_mean,s_clamp_max\n")
        for h in hist:
            f.write(f"{h['epoch']},{h['train_nll']:.6f},{h['val_nll']:.6f},"
                    f"{h['lr']:.6g},{h['h_std_batch']:.6f},{h['shuffle_gap']:.6f},"
                    f"{h['grad_norm']:.6f},{h['s_clamp_mean']:.6f},"
                    f"{h['s_clamp_max']:.6f}\n")
        f.write("# recon@best (posterior_pixel_mean):"
                f" val_psnr={v_psnr:.4f} val_mse={v_mse:.6f}"
                f" val_fwd_rel={v_fwd:.6f} sample_pixel_std={v_sstd:.6f}\n")
    _nll_curve(hist, best_epoch, os.path.join(run_dir, "nll_curve.png"))
    _recon_grid(model, vl, device, os.path.join(run_dir, "recon_grid.png"),
                blur_sigma, scale, _RECON_N_POST)

    write_json(os.path.join(run_dir, "status.json"), {
        "expert": cfg["expert"], "seed_index": seed_index, "rng_seed": rng_seed,
        "cfg_hash": chash, "test0": test0, "best_epoch": best_epoch,
        "best_val_nll": best_val, "checkpoint_path": ckpt_path,
        "best_checkpoint_sha256": sha256_file(ckpt_path),
        "cell": {"dataset": cell.get("dataset"), "scale": scale, "blur_sigma": blur_sigma, "noise_sigma": noise_sigma},
        "n_params": n_params, "device": device,
        "torch_version": torch.__version__,
        "recon_mode": "posterior_pixel_mean", "recon_n_post": _RECON_N_POST,
        "val_psnr": v_psnr, "val_mse": v_mse, "val_fwd_rel": v_fwd,
        "sample_pixel_std": v_sstd,
        "ccr_enabled": ccr_on, "ccr": ccr, "grad_clip": grad_clip,
        "h_std_batch_final": hist[best_epoch]["h_std_batch"],
        "shuffle_gap_final": hist[best_epoch]["shuffle_gap"],
        "grad_norm_final": hist[best_epoch]["grad_norm"],
        "s_clamp_mean_final": hist[best_epoch]["s_clamp_mean"],
        "s_clamp_frac_layers_final": hist[best_epoch]["s_clamp_layers"],
        "train_time_sec": round(time.time() - t0, 1), "status": "done",
    })
    logger.info("[train_base] DONE best_val=%.3f @epoch %d", best_val, best_epoch)


if __name__ == "__main__":
    main()
