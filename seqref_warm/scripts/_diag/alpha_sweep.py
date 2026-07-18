# SEQREF-ASWEEP v0.1 -- alpha_sweep
# LIFETIME: DIAGNOSTIC
# DBG Phase A1b (no-training): free-alpha sweep on the SAVED W1 checkpoint.
# Loads the trained refiner, computes dx ONCE on the cached val split (dx is
# frozen -- unlike a retrained fixed-alpha arm, the body cannot rescale), then
# evaluates x(alpha) = x0 + alpha*dx over a grid. Answers: would the existing
# correction become useful if applied more gently?
#   Best alpha << 0.5  -> over-application contributes; B targets calibration
#   No alpha helps     -> correction direction/quality is primary, not gate
#   (per-sample spread -> a selective gate could help; current one doesn't)
# Also logs the ACTUAL loss components once (Charbonnier vs weighted budget)
# at the trained (g,dx) to measure -- not estimate -- budget dominance.
# Outputs: JSON + three curves (alpha vs dpsnr / fwd_rel / %improved) +
# per-sample dPSNR hists at best alpha and alpha=0.5 + harmful-tail stats.
# No fallback/mock/silent-pass. Failures: logger.error + raise.
from __future__ import annotations
import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from seqref_warm.src.degrade import make_degraded
from seqref_warm.src.refiners.base_io import FrozenBase, precompute_split
from seqref_warm.scripts.train_refiner import (_forward_split, _charbonnier,
                                               _psnr_per_sample)
from seqref_warm.scripts import train_refiner as TR

logger = logging.getLogger("seqref_warm.alpha_sweep")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

ALPHAS = [0.0, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]


def _aggregate_psnr(x, x_true):
    mse = torch.mean((x - x_true) ** 2)
    return float(10.0 * torch.log10(1.0 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="refine_w1_rnvp.yaml")
    ap.add_argument("--run-dir", required=True,
                    help="W1 run dir containing checkpoint.pt + config.yaml")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ckpt_path = os.path.join(args.run_dir, "checkpoint.pt")
    if not os.path.isfile(ckpt_path):
        logger.error("[alpha_sweep] checkpoint not found: %s", ckpt_path)
        raise FileNotFoundError(ckpt_path)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = FrozenBase(cfg["base"]["run_dir"], device)
    n_post = int(cfg["base"].get("n_post", 16))
    recon_seed = int(cfg["base"]["recon_seed"])
    cell = base.cfg["cell"]
    dk = dict(sigma=base.blur_sigma, scale=base.scale,
              noise_sigma=float(cell["noise_sigma"]))
    bs = int(cfg["train"]["batch_size"])
    vl = DataLoader(make_degraded(cell.get("dataset"), cell["data_root"],
                                  split="val", **dk),
                    batch_size=bs, shuffle=False, num_workers=2)
    cache_dir = os.path.join(cfg["output"]["root"], "_cache")
    vaX, vaY, vaX0, vaIn = precompute_split(base, vl, n_post=n_post,
                                            rng_seed=recon_seed,
                                            cache_dir=cache_dir,
                                            split_name="val", device=device)

    r = cfg["refiner"]
    model = TR.CplRegRefiner(flavor=r["flavor"], dim=int(r.get("dim", 784)),
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
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    logger.info("[alpha_sweep] loaded %s (best ep %s)", ckpt_path,
                ck.get("epoch"))

    with torch.no_grad():
        x1_trained, dx, g = _forward_split(model, vaIn, vaX0, bs, device)
        dx = dx.cpu(); g = g.cpu()
    x_true = vaX.cpu(); x0 = vaX0.cpu()
    psnr_x0_per = _psnr_per_sample(x0.clamp(0, 1), x_true)
    agg_x0 = _aggregate_psnr(x0.clamp(0, 1), x_true)

    # Measured loss components at the trained (g, dx) -- budget dominance.
    ch_eps = float(cfg["train"].get("charbonnier_eps", 1e-3))
    lam_b = float(cfg["train"].get("delta_budget_lambda", 1e-3))
    applied = (g.view(-1, 1, 1, 1) * dx) if g.dim() == 1 else g * dx
    charb = float(_charbonnier((x0 + applied).clamp(0, 1), x_true, ch_eps))
    budget = float(lam_b * torch.mean(applied ** 2))
    logger.info("[alpha_sweep] LOSS COMPONENTS at trained (g,dx): "
                "charbonnier=%.6f  weighted_budget=%.8f  ratio=%.1fx",
                charb, budget, charb / budget if budget > 0 else float("inf"))

    rows = []
    per_sample_by_alpha = {}
    for a in ALPHAS:
        xa = (x0 + a * dx).clamp(0, 1)
        per = _psnr_per_sample(xa, x_true) - psnr_x0_per
        agg = _aggregate_psnr(xa, x_true) - agg_x0
        fwd = TR._fwd_rel(xa, vaY.cpu(), base.blur_sigma, base.scale)
        pi = float((per > 0).float().mean() * 100.0)
        p5 = float(torch.quantile(per, 0.05))
        rows.append({"alpha": a, "agg_dpsnr": agg, "fwd_rel": float(fwd),
                     "pct_improved": pi, "per_sample_mean": float(per.mean()),
                     "p5_dpsnr": p5, "min_dpsnr": float(per.min())})
        per_sample_by_alpha[a] = per
        logger.info("[alpha_sweep] a=%.3f  agg_dpsnr=%+.4f  fwd_rel=%.4f  "
                    "%%improved=%.1f  p5=%+.3f  min=%+.3f", a, agg, fwd, pi,
                    p5, float(per.min()))

    best = max(rows, key=lambda r: r["agg_dpsnr"])
    logger.info("[alpha_sweep] BEST alpha=%.3f  agg_dpsnr=%+.4f  "
                "fwd_rel=%.4f  %%improved=%.1f", best["alpha"],
                best["agg_dpsnr"], best["fwd_rel"], best["pct_improved"])

    al = [r["alpha"] for r in rows]
    for key, ylab, fname in [("agg_dpsnr", "aggregate dPSNR (dB)",
                              "alpha_vs_dpsnr.png"),
                             ("fwd_rel", "fwd_rel", "alpha_vs_fwd_rel.png"),
                             ("pct_improved", "% samples improved",
                              "alpha_vs_pct_improved.png")]:
        plt.figure(figsize=(6, 4))
        plt.plot(al, [r[key] for r in rows], marker="o")
        if key == "agg_dpsnr":
            plt.axhline(0.0, color="gray", lw=0.8)
        if key == "fwd_rel":
            plt.axhline(0.2915, color="gray", lw=0.8, label="x0 (0.2915)")
            plt.legend()
        plt.xlabel("alpha"); plt.ylabel(ylab); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, fname), dpi=120); plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(per_sample_by_alpha[best["alpha"]].numpy(), bins=80, alpha=0.6,
             label=f"alpha={best['alpha']:.3f}")
    plt.hist(per_sample_by_alpha[0.50].numpy(), bins=80, alpha=0.6,
             label="alpha=0.500")
    plt.axvline(0.0, color="gray", lw=0.8)
    plt.xlabel("per-sample dPSNR (dB)"); plt.ylabel("count"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "per_sample_hist.png"), dpi=120)
    plt.close()

    out = {"run_dir": args.run_dir, "best_epoch": ck.get("epoch"),
           "alphas": rows, "best": best,
           "loss_components": {"charbonnier": charb,
                               "weighted_budget": budget,
                               "dominance_ratio": charb / budget
                               if budget > 0 else None},
           "g_stats": {"mean": float(g.mean()), "std": float(g.std()),
                       "min": float(g.min()), "max": float(g.max())}}
    with open(os.path.join(args.out_dir, "alpha_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    logger.info("[alpha_sweep] written: %s", args.out_dir)


if __name__ == "__main__":
    main()
