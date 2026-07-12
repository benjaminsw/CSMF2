# SEQREF-FMPROBE v0.1 -- scripts/_diag/fm_ungated_probe.py
# LIFETIME: DIAGNOSTIC
# H1-vs-H2 probe for the Level-3 FM arms (no training): load a saved FM run,
# recompute Δx_FM (K=8 rollout from x1), then
#   * α-sweep: x2(α) = clamp(x1 + α·Δx_FM), α ∈ {0,0.05,0.1,0.25,0.5,1.0}
#   * per-α: aggregate ΔPSNR, fwd_rel, SSIM, %improved, per-sample mean/median
#   * |Δx| stats (mean/std/max/p95), g stats, cos(Δx_FM, x_true−x1)
#   * plots: alpha_sweep_dpsnr, alpha_sweep_fwd_ssim, dx_hist, probe_grid,
#            per_sample_scatter (ungated gain vs cosine)
# Decision rules (Ben 2026-07-09): α=1 dpsnr≈0 & tiny |Δx| -> H1;
#   α=1 (or any α) > +0.10 -> H2; fwd_rel blowup -> not measurement-faithful;
#   cos≈0 -> misaligned; cos>0.2 & ungated helps -> direction real.
# No fallback/mock/pass; missing files raise.
from __future__ import annotations
import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from seqref_warm.src.degrade import MNISTDegraded
from seqref_warm.src.metrics import psnr as _psnr, ssim as _ssim, fwd_rel as _fwd_rel
from seqref_warm.src.refiners.base_io import (FrozenBase, FrozenStage1,
                                               precompute_split,
                                               precompute_stage2)
from seqref_warm.src.refiners.flow_matching_refiner import FMRefiner
from seqref_warm.src.train_utils import setup_logger, seed_from_index

logger = setup_logger("seqref_warm.fm_ungated_probe")
_ALPHAS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
_K = 8


def _psnr_ps(x_hat, x_true):
    m = ((x_hat - x_true) ** 2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="FM run dir (fm_arm*_...)")
    args = ap.parse_args()
    with open(os.path.join(args.run, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_index = int(cfg["train"]["seed"])
    rng_seed = seed_from_index(seed_index)

    base = FrozenBase(cfg["base"]["run_dir"], device)
    stage1 = FrozenStage1(cfg["stage1"]["run_dir"], device)
    arm_b = cfg.get("arm_b_expert_channel")
    arm = "B" if arm_b else "A"
    cell = base.cfg["cell"]
    dk = dict(sigma=base.blur_sigma, scale=base.scale,
              noise_sigma=float(cell["noise_sigma"]))
    bs = int(cfg["train"]["batch_size"])
    vl = DataLoader(MNISTDegraded(cell["data_root"], split="val", **dk),
                    batch_size=bs, shuffle=False, num_workers=2)
    cache_dir = os.path.join(cfg["output"]["root"], "_cache")
    vaX, vaY, vaX0, vaIn0 = precompute_split(base, vl,
                                             n_post=int(cfg["base"]["n_post"]),
                                             rng_seed=rng_seed,
                                             cache_dir=cache_dir,
                                             split_name="val", device=device)
    vaX1, vaCond = precompute_stage2(stage1, base, vaX, vaY, vaX0, vaIn0,
                                     batch_size=bs, cache_dir=cache_dir,
                                     split_name="val")
    if arm == "B":
        rnvp = FrozenStage1(arm_b["run_dir"], device)
        vaXR, _ = precompute_stage2(rnvp, base, vaX, vaY, vaX0, vaIn0,
                                    batch_size=bs, cache_dir=cache_dir,
                                    split_name="val")
        vaCond = torch.cat([vaCond, vaXR], dim=1)

    r = cfg["refiner"]
    model = FMRefiner(cond_channels=vaCond.size(1),
                      hidden=int(r.get("hidden", 64)),
                      depth=int(r.get("depth", 4)),
                      t_embed_dim=int(r.get("t_embed_dim", 64)),
                      g_max=float(r.get("g_max", 0.5)),
                      g_init=float(r.get("g_init", 0.05))).to(device)
    ckpt = os.path.join(args.run, "checkpoint.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()

    dxs, gs = [], []
    with torch.no_grad():
        for i in range(0, vaCond.size(0), bs):
            c = vaCond[i:i + bs].to(device)
            x = vaX1[i:i + bs].to(device)
            _, dx, g = model(x, c, _K)
            dxs.append(dx.cpu()); gs.append(g.cpu())
    dx = torch.cat(dxs); g = torch.cat(gs)

    psnr_x1 = _psnr(vaX1, vaX)
    ssim_x1 = _ssim(vaX1, vaX)
    fwd_x1 = _fwd_rel(vaX1, vaY, base.blur_sigma, base.scale)
    ps_x1 = _psnr_ps(vaX1, vaX)
    oracle = vaX - vaX1
    cosv = F.cosine_similarity(dx.flatten(1), oracle.flatten(1), dim=1)
    dxa = dx.abs()
    rows = []
    for a in _ALPHAS:
        x2 = (vaX1 + a * dx).clamp(0, 1)
        dps = _psnr_ps(x2, vaX) - ps_x1
        rows.append({"alpha": a,
                     "agg_dpsnr": _psnr(x2, vaX) - psnr_x1,
                     "fwd": _fwd_rel(x2, vaY, base.blur_sigma, base.scale),
                     "ssim": _ssim(x2, vaX),
                     "pct": float((dps > 0).float().mean()) * 100,
                     "mean": float(dps.mean()), "median": float(dps.median())})
    ga = float(g.mean())
    x2g = (vaX1 + g.view(-1, 1, 1, 1) * dx).clamp(0, 1)
    gated_dpsnr = _psnr(x2g, vaX) - psnr_x1
    best = max(rows, key=lambda r_: r_["agg_dpsnr"])
    a1 = [r_ for r_ in rows if r_["alpha"] == 1.0][0]

    out = args.run
    plt.figure(figsize=(5.5, 4))
    plt.plot([r_["alpha"] for r_ in rows], [r_["agg_dpsnr"] for r_ in rows],
             "o-")
    plt.axhline(0, color="gray", lw=0.8)
    plt.axhline(0.1, color="k", ls=":", lw=0.8, label="+0.10 (H2 bar)")
    plt.xlabel("α"); plt.ylabel("aggregate ΔPSNR (dB)"); plt.legend()
    plt.title(f"Arm {arm}: α-sweep")
    plt.tight_layout(); plt.savefig(os.path.join(out, "alpha_sweep_dpsnr.png"),
                                    dpi=120); plt.close()
    fig, ax1 = plt.subplots(figsize=(5.5, 4))
    ax1.plot([r_["alpha"] for r_ in rows], [r_["fwd"] for r_ in rows], "o-",
             color="C0", label="fwd_rel")
    ax1.axhline(fwd_x1, color="C0", ls="--", lw=0.8)
    ax1.set_xlabel("α"); ax1.set_ylabel("fwd_rel", color="C0")
    ax2 = ax1.twinx()
    ax2.plot([r_["alpha"] for r_ in rows], [r_["ssim"] for r_ in rows], "s-",
             color="C3", label="ssim")
    ax2.axhline(ssim_x1, color="C3", ls="--", lw=0.8)
    ax2.set_ylabel("ssim", color="C3")
    plt.title(f"Arm {arm}: fwd_rel / ssim vs α")
    plt.tight_layout(); plt.savefig(os.path.join(out,
                                    "alpha_sweep_fwd_ssim.png"), dpi=120)
    plt.close(fig)
    plt.figure(figsize=(5.5, 4))
    plt.hist(dx.flatten().numpy(), bins=120)
    plt.yscale("log"); plt.xlabel("Δx_FM value"); plt.ylabel("count (log)")
    plt.title(f"Arm {arm}: Δx_FM histogram")
    plt.tight_layout(); plt.savefig(os.path.join(out, "dx_hist.png"), dpi=120)
    plt.close()
    k = 8
    y_up = F.interpolate(vaY[:k], size=(28, 28), mode="nearest")
    x2u = (vaX1 + dx).clamp(0, 1)
    grid_rows = [y_up, vaX1[:k], x2g[:k], x2u[:k], vaX[:k], dxa[:k],
                 (vaX[:k] - vaX1[:k]).abs()]
    labels = ["y_up", "x1", "x1+gΔ", "x1+Δ", "x_true", "|Δ|", "|xt−x1|"]
    fig, ax = plt.subplots(7, k, figsize=(k + 1, 7))
    for ri in range(7):
        for c in range(k):
            ax[ri, c].imshow(grid_rows[ri][c, 0], cmap="gray", vmin=0, vmax=1)
            ax[ri, c].axis("off")
        ax[ri, 0].text(-0.3, 0.5, labels[ri], transform=ax[ri, 0].transAxes,
                       ha="right", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(out, "probe_grid.png"),
                                    dpi=110); plt.close(fig)
    ung_gain = _psnr_ps(x2u, vaX) - ps_x1
    plt.figure(figsize=(5.5, 4))
    plt.scatter(ung_gain.numpy(), cosv.numpy(), s=4, alpha=0.3)
    plt.axhline(0, color="gray", lw=0.8); plt.axvline(0, color="gray", lw=0.8)
    plt.xlabel("per-sample ΔPSNR ungated (α=1)")
    plt.ylabel("cos(Δx_FM, x_true−x1)")
    plt.title(f"Arm {arm}: gain vs alignment")
    plt.tight_layout(); plt.savefig(os.path.join(out,
                                    "per_sample_scatter.png"), dpi=120)
    plt.close()

    print(f"=== FM ungated probe: Arm {arm} ({os.path.basename(args.run)}) ===")
    print(f"g: mean {ga:.4f} min {float(g.min()):.4f} max {float(g.max()):.4f}")
    print(f"|dx|: mean {float(dxa.mean()):.4e} p95 "
          f"{float(dxa.flatten().quantile(0.95)):.4e} max {float(dxa.max()):.4e}")
    print(f"cos(dx, x_true-x1): mean {float(cosv.mean()):+.4f} "
          f"median {float(cosv.median()):+.4f}")
    print(f"gated (official) dpsnr: {gated_dpsnr:+.4f}")
    print("alpha sweep (agg dpsnr | fwd | ssim | %improved | ps-mean):")
    for r_ in rows:
        print(f"  α={r_['alpha']:<5} {r_['agg_dpsnr']:+.4f} | {r_['fwd']:.4f} "
              f"| {r_['ssim']:.4f} | {r_['pct']:5.1f}% | {r_['mean']:+.4f}")
    print(f"x1 baselines: fwd {fwd_x1:.4f} ssim {ssim_x1:.4f}")
    print(f"best α = {best['alpha']} ({best['agg_dpsnr']:+.4f} dB)   "
          f"α=1: {a1['agg_dpsnr']:+.4f} dB")
    h2 = best["agg_dpsnr"] > 0.10
    tiny = float(dxa.mean()) < 1e-3
    verdict = ("H2 (correction real, gate suppressed)" if h2 else
               "H1 (no useful correction learned)" if (tiny or
               a1["agg_dpsnr"] <= 0.02) else
               "AMBIGUOUS (small but nonzero -- inspect plots)")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
