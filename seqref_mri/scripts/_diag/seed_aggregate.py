# SEQREF-SEEDAGG v0.1 -- scripts/_diag/seed_aggregate.py
# LIFETIME: DIAGNOSTIC
# Aggregate N refiner runs (one per seed) into the 3-seed claim verdict:
#   CLAIM PASS iff  mean aggregate dpsnr > +0.3  AND  >= 2/3 seeds dpsnr > 0
#                   AND no seed regresses fwd_rel (tol 1e-3)
# Optional --spread-check: loads base + refiner for the FIRST run, verifies
# Option-A exactly -- Var over posterior samples of x1_i = x0_i + g*dx (shared
# correction) must equal Var of x0_i (max |std diff| ~ 0). Never measured until
# now; asserted-by-design previously.
# Outputs: seed_aggregate.md + console table. No fallback/mock/pass; missing
# status.json or gate fields -> logger.error + raise.
from __future__ import annotations
import argparse
import json
import logging
import os

logger = logging.getLogger("seqref_mri.seed_aggregate")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

_CLAIM_MEAN = 0.3
_CLAIM_MIN_POS = 2 / 3
_FWD_TOL = 1e-3


def _load(run_dir: str) -> dict:
    p = os.path.join(run_dir, "status.json")
    if not os.path.isfile(p):
        logger.error("[seedagg] missing %s", p)
        raise FileNotFoundError(p)
    with open(p) as f:
        s = json.load(f)
    for k in ("seed_index", "best_val_dpsnr", "best_val_fwd_rel_x0",
              "best_val_fwd_rel_x1", "best_val_psnr_x0", "best_val_psnr_x1",
              "pct_samples_improved", "best_y_gap_dpsnr", "best_atr_gap_dpsnr",
              "best_g_mean", "refiner_expert"):
        if k not in s:
            logger.error("[seedagg] %s missing field %s", p, k)
            raise KeyError(k)
    return s


def _spread_check(run_dir: str, n_post: int = 16, batch: int = 64) -> float:
    # Option-A verification on one val batch: max |std(x1_i) - std(x0_i)|.
    import torch
    from torch.utils.data import DataLoader
    from seqref_mri.src.degrade import MNISTDegraded
    from seqref_mri.src.refiners.base_io import FrozenBase, refiner_inputs
    from seqref_mri.src.refiners.coupling_regressor import CplRegRefiner
    import yaml
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = FrozenBase(cfg["base"]["run_dir"], device)
    r = cfg["refiner"]
    model = CplRegRefiner(flavor=r["flavor"], dim=int(r.get("dim", 784)),
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
    ckpt = torch.load(os.path.join(run_dir, "checkpoint.pt"),
                      map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    cell = base.cfg["cell"]
    ds = MNISTDegraded(cell["data_root"], split="val", sigma=base.blur_sigma,
                       scale=base.scale,
                       noise_sigma=float(cell["noise_sigma"]))
    x, y = next(iter(DataLoader(ds, batch_size=batch, shuffle=False)))
    y = y.to(device)
    gen = torch.Generator(device=device).manual_seed(12345)
    with torch.no_grad():
        # per-sample posterior draws x0_i (f64 decode), shared correction
        y64 = y.to(torch.float64)
        h_base = base.model.cond(y64)
        draws = []
        from seqref_mri.src.degrade import inverse_logit
        for _ in range(n_post):
            z = torch.randn(y.size(0), base.model.dim, generator=gen,
                            device=device, dtype=torch.float64)
            xl = base.model.decode(z, h_base)
            draws.append(inverse_logit(xl).view(-1, 1, 28, 28).clamp(0, 1).float())
        x0_draws = torch.stack(draws)                       # (P,B,1,28,28)
        x0_mean = x0_draws.mean(0)
        inp = refiner_inputs(y, x0_mean, base.blur_sigma, base.scale)
        _, dx, g = model(inp, x0_mean)
        corr = (g.view(-1, 1, 1, 1) * dx).unsqueeze(0)      # shared (1,B,1,28,28)
        x1_draws = (x0_draws + corr).clamp(0, 1)
        # clamp can touch spread at the boundary; report both
        d_raw = float(((x0_draws + corr).std(0) - x0_draws.std(0)).abs().max())
        d_clamped = float((x1_draws.std(0) - x0_draws.std(0)).abs().max())
    logger.info("[seedagg] spread check: max|Δstd| raw=%.3e clamped=%.3e",
                d_raw, d_clamped)
    return d_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="refiner run dirs, one per seed")
    ap.add_argument("--out", default="seed_aggregate.md")
    ap.add_argument("--spread-check", action="store_true")
    args = ap.parse_args()

    rows = [_load(d) for d in args.runs]
    rows.sort(key=lambda s: s["seed_index"])
    flavors = {s["refiner_expert"] for s in rows}
    if len(flavors) != 1:
        logger.error("[seedagg] mixed flavors %s", flavors)
        raise ValueError("all runs must share one refiner_expert")
    dps = [s["best_val_dpsnr"] for s in rows]
    fwd_reg = [s["best_val_fwd_rel_x1"] - s["best_val_fwd_rel_x0"] for s in rows]
    mean_dpsnr = sum(dps) / len(dps)
    frac_pos = sum(d > 0 for d in dps) / len(dps)
    no_fwd_reg = all(r <= _FWD_TOL for r in fwd_reg)
    claim = (mean_dpsnr > _CLAIM_MEAN and frac_pos >= _CLAIM_MIN_POS
             and no_fwd_reg)

    lines = [f"# {flavors.pop()}_refine 3-seed aggregate", "",
             "| seed | agg dPSNR | psnr x0->x1 | fwd_rel x0->x1 | %improved |"
             " g_mean | y_gap | atr_gap |",
             "|--|--|--|--|--|--|--|--|"]
    for s in rows:
        lines.append(
            f"| {s['seed_index']} | {s['best_val_dpsnr']:+.3f} |"
            f" {s['best_val_psnr_x0']:.2f}->{s['best_val_psnr_x1']:.2f} |"
            f" {s['best_val_fwd_rel_x0']:.4f}->{s['best_val_fwd_rel_x1']:.4f} |"
            f" {100*s['pct_samples_improved']:.1f}% |"
            f" {s['best_g_mean']:.3f} | {s['best_y_gap_dpsnr']:+.3f} |"
            f" {s['best_atr_gap_dpsnr']:+.3f} |")
    lines += ["",
              f"mean aggregate dPSNR: **{mean_dpsnr:+.3f}** (claim bar +0.3)",
              f"seeds improving: {sum(d > 0 for d in dps)}/{len(dps)}",
              f"fwd_rel regressions (tol {_FWD_TOL}): "
              f"{[f'{r:+.4f}' for r in fwd_reg]}",
              f"**CLAIM GATE: {'PASS' if claim else 'FAIL'}**"]
    if args.spread_check:
        d = _spread_check(args.runs[0])
        lines.append(f"Option-A spread check (run[0]): max|Δstd| = {d:.3e} "
                     f"({'OK' if d < 1e-6 else 'VIOLATION'})")
    md = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
