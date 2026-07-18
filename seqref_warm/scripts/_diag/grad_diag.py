# SEQREF-GRADDIAG v0.2 -- grad_diag
# LIFETIME: DIAGNOSTIC
# DBG Phase A1c (no-training): gradient-strength comparison at the saved W1
# checkpoint. Scalar loss ratio (15,646x) proved the budget negligible in the
# REPORTED loss but says nothing about gradient scale. Measures, per lambda
# in {0.01, 0.1, 1.0}:
#     ||grad_theta Charbonnier||  vs  ||grad_theta lambda*mean((g*dx)^2)||
# v0.2 (robustness lock for Phase B):
#   * THREE deterministic val batches (first three of the seeded cache),
#     median ratio reported -- single-batch ratios can mislead.
#   * cosine(grad_charb, grad_budget) per batch: strongly negative =>
#     budget actively resists larger corrections; ~0 => terms act on
#     mostly different parameter directions.
# NOTE: ratios are exact in lambda ONLY at this fixed checkpoint/batch
# (grad[lam*L] = lam*grad[L]); during training the geometry drifts.
# No fallback/mock/silent-pass. Failures: logger.error + raise.
from __future__ import annotations
import argparse
import json
import logging
import os

import torch
import yaml
from torch.utils.data import DataLoader

from seqref_warm.src.degrade import make_degraded
from seqref_warm.src.refiners.base_io import FrozenBase, precompute_split
from seqref_warm.scripts.train_refiner import _charbonnier
from seqref_warm.scripts import train_refiner as TR

logger = logging.getLogger("seqref_warm.grad_diag")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

LAMBDAS = [0.01, 0.1, 1.0]


def _grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm() ** 2)
    return total ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="refine_w1_rnvp.yaml")
    ap.add_argument("--run-dir", required=True, help="W1 run dir")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", required=True, help="JSON output path")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ckpt_path = os.path.join(args.run_dir, "checkpoint.pt")
    if not os.path.isfile(ckpt_path):
        logger.error("[grad_diag] checkpoint not found: %s", ckpt_path)
        raise FileNotFoundError(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = FrozenBase(cfg["base"]["run_dir"], device)
    n_post = int(cfg["base"].get("n_post", 16))
    recon_seed = int(cfg["base"]["recon_seed"])
    cell = base.cfg["cell"]
    dk = dict(sigma=base.blur_sigma, scale=base.scale,
              noise_sigma=float(cell["noise_sigma"]))
    vl = DataLoader(make_degraded(cell.get("dataset"), cell["data_root"],
                                  split="val", **dk),
                    batch_size=int(cfg["train"]["batch_size"]),
                    shuffle=False, num_workers=2)
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
    model.train()  # gradients required; no optimizer step is taken

    ch_eps = float(cfg["train"].get("charbonnier_eps", 1e-3))
    B = args.batch_size
    n_batches = 3
    import statistics as st

    def _grad_vec(model):
        return torch.cat([p.grad.detach().flatten()
                          for p in model.parameters() if p.grad is not None])

    per_batch = []
    for bi in range(n_batches):
        xb = vaX[bi * B:(bi + 1) * B].to(device)
        x0b = vaX0[bi * B:(bi + 1) * B].to(device)
        inb = vaIn[bi * B:(bi + 1) * B].to(device)
        if xb.size(0) == 0:
            logger.error("[grad_diag] val split exhausted at batch %d", bi)
            raise ValueError("not enough val samples for 3 batches")

        def forward_terms():
            x1, dx, g = model(inb, x0b)
            applied = g.view(-1, 1, 1, 1) * dx if g.dim() == 1 else g * dx
            charb = _charbonnier(x0b + applied, xb, ch_eps)
            raw_budget = torch.mean(applied ** 2)
            return charb, raw_budget

        model.zero_grad(set_to_none=True)
        charb, _ = forward_terms()
        charb.backward()
        gv_charb = _grad_vec(model)
        gn_charb = float(gv_charb.norm())

        model.zero_grad(set_to_none=True)
        _, raw_budget = forward_terms()
        raw_budget.backward()          # lambda-free; scale exactly by lambda
        gv_budget = _grad_vec(model)
        gn_raw = float(gv_budget.norm())
        cos = float(torch.nn.functional.cosine_similarity(
            gv_charb, gv_budget, dim=0))

        row = {"batch": bi, "charb_loss": float(charb),
               "charb_grad_norm": gn_charb,
               "cosine_charb_budget": cos,
               "lambdas": [{"lambda": lam,
                            "grad_norm": lam * gn_raw,
                            "charb_to_budget_grad_ratio":
                                gn_charb / (lam * gn_raw)
                                if gn_raw > 0 else float("inf")}
                           for lam in LAMBDAS]}
        per_batch.append(row)
        logger.info("[grad_diag] batch %d: ||gC||=%.5f cos=%.3f | " + " ".join(
            "lam=%.2f ratio=%.1fx" % (l["lambda"],
                                      l["charb_to_budget_grad_ratio"])
            for l in row["lambdas"]), bi, gn_charb, cos)

    medians = []
    for i, lam in enumerate(LAMBDAS):
        rs = [b["lambdas"][i]["charb_to_budget_grad_ratio"]
              for b in per_batch]
        medians.append({"lambda": lam, "median_ratio": st.median(rs),
                        "ratios": rs})
        logger.info("[grad_diag] lambda=%.2f median ratio = %.1fx  %s",
                    lam, st.median(rs),
                    ["%.1f" % r for r in rs])
    cosines = [b["cosine_charb_budget"] for b in per_batch]
    logger.info("[grad_diag] cosine(charb, budget) per batch: %s  median %.3f",
                ["%.3f" % c for c in cosines], st.median(cosines))

    results = {"n_batches": n_batches, "batch_size": B,
               "per_batch": per_batch, "median_ratios": medians,
               "cosines": cosines, "median_cosine": st.median(cosines)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("[grad_diag] written: %s", args.out)


if __name__ == "__main__":
    main()
