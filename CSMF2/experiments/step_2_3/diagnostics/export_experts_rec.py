# =============================================================================
# NWS v0.4 -- CSMF2.experiments.step_2_3.diagnostics.export_experts_rec
# Purpose: Step 2.3-NWS Step 0. Re-score the FROZEN, trained 2.3-A experts over
#          the MNIST val split with the project's OWN scorer per_expert_rec
#          (z-bank, sum-over-pixels). Emit per_sample_rec.csv + recons.pt
#          (cached pixel-space x_hat_k, y, x_true, labels). No training.
# CONVENTION: No silent fallback. NSF spline inverse stored/scored in f64. Every
#          failure path -> logger.error + raise.
# CAVEAT (logged at runtime): MNISTDegraded.degrade uses generator=None, so the
#   val AWGN is a FRESH realisation, not the run's exact noise. rec values are
#   noise-realisation dependent; cross-expert STRUCTURE is the signal, and the
#   multi-seed confirm averages the realisation out.
# Changelog (NEW in v0.4):
#   * Introduced (replaces the build_val_loader/reconstruct_per_expert draft).
#     Loads 3 per-expert dirs via load_experts_trainable(train=False); builds the
#     z-bank from the run cfg (rec_z_bank_size/seed/mode); scores per_expert_rec
#     -> (B,K); derives best_rec/best_expert, nsf_rec_quartile, gap_*.
#   * Caches z-bank-mean pixel x_hat_k (decode -> inverse_logit) for soft-gate +
#     residual; attaches digit labels by dataset index (batch drops them).
# Update summary:
#   The one expensive GPU pass. Authoritative rec from per_expert_rec; x_hat is an
#   auxiliary z-bank-mean reconstruction used only by the downstream diagnostics.
# =============================================================================
from __future__ import annotations

import argparse
import logging
import os

from CSMF2.experiments.step_2_3.diagnostics.nws_common import (
    EXPERTS, QUARTILE_LABELS, SEED_RNG, results_dir, setup_logging, read_run_cfg,
    to_f64, _wire_load_experts_trainable, _wire_rec_and_zbank,
    _wire_mnist_degraded, _wire_degrade_ops,
)

logger = logging.getLogger(__name__)
__version__ = "0.4"
__abbr__ = "NWS"


def _labels_for(ds) -> list[int]:
    """Digit labels in dataset order (MNISTDegraded.__getitem__ drops them)."""
    targets = ds.base.targets
    idx = ds.subset_idx
    n = len(ds)
    return [int(targets[idx[i] if idx is not None else i]) for i in range(n)]


def export(per_expert_dirs: list[str], mixture_dir: str, seed_index: int,
           smoke: int, device: str) -> str:
    import torch
    import pandas as pd

    load_experts_trainable = _wire_load_experts_trainable()
    per_expert_rec, make_z_bank = _wire_rec_and_zbank()
    MNISTDegraded = _wire_mnist_degraded()
    inverse_logit, _, _ = _wire_degrade_ops()

    cfg = read_run_cfg(mixture_dir)
    blur_sigma = float(cfg["blur_sigma"]); scale = int(cfg["scale"])
    noise_sigma = float(cfg["noise_sigma"]); data_root = cfg["data_root"]
    zb_size = int(cfg["rec_z_bank_size"]); zb_seed = int(cfg["rec_z_bank_seed"])
    zb_mode = cfg.get("rec_z_bank_mode", "randn")
    logger.info("[export] cfg: blur=%.3f scale=%d noise=%.3f z-bank(size=%d seed=%d mode=%s)",
                blur_sigma, scale, noise_sigma, zb_size, zb_seed, zb_mode)
    logger.warning("[export] val AWGN is a fresh realisation (degrade generator=None); "
                   "rec is noise-realisation dependent -- structure is the signal.")

    if len(per_expert_dirs) != len(EXPERTS):
        logger.error("[export] %d dirs for %d experts", len(per_expert_dirs), len(EXPERTS))
        raise ValueError("per-expert dir count != expert_set")

    dev = torch.device(device)
    experts, _ref = load_experts_trainable(per_expert_dirs, dev, train=False)
    if len(experts) != len(EXPERTS):
        logger.error("[export] loaded %d experts, expected %d", len(experts), len(EXPERTS))
        raise RuntimeError("expert load count mismatch")

    dim = int(experts[0].dim)
    z_bank = make_z_bank(dim, zb_size, zb_mode, zb_seed, dev, torch.float32)
    S = z_bank.size(0)

    ds = MNISTDegraded(data_root, split="val", sigma=blur_sigma, scale=scale,
                       noise_sigma=noise_sigma)
    labels_all = _labels_for(ds)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)

    rows, cache = [], {f"x_hat_{e}": [] for e in EXPERTS}
    cache_y, cache_xt = [], []
    n_seen = 0
    with torch.no_grad():
        for x, y in dl:
            if smoke and n_seen >= smoke:
                break
            x = x.to(dev); y = y.to(dev)                 # x:(B,1,28,28) y:(B,1,14,14)
            B = y.size(0)

            rec = per_expert_rec(experts, y, z_bank,
                                 blur_sigma=blur_sigma, scale=scale)   # (B,K)
            rec = to_f64(rec).cpu().numpy()

            # auxiliary z-bank-mean pixel reconstruction per expert (for soft-gate/residual)
            for k, ex in enumerate(experts):
                h = ex.cond(y)
                acc = None
                for s in range(S):
                    z = z_bank[s:s + 1].expand(B, -1)
                    xp = to_f64(inverse_logit(ex.decode(z, h))).view(B, -1)
                    acc = xp if acc is None else acc + xp
                cache[f"x_hat_{EXPERTS[k]}"].append((acc / S).cpu())

            for i in range(B):
                row = {"sample_id": n_seen + i, "class": labels_all[n_seen + i],
                       "noise_sigma": noise_sigma, "scale": scale}
                for k, e in enumerate(EXPERTS):
                    row[f"rec_{e}"] = float(rec[i, k])
                rows.append(row)
            cache_y.append(to_f64(y).flatten(1).cpu())
            cache_xt.append(to_f64(x).flatten(1).cpu())
            n_seen += B

    if n_seen == 0:
        logger.error("[export] no samples scored -- empty val set")
        raise RuntimeError("empty val set")

    df = pd.DataFrame(rows)
    rec_cols = [f"rec_{e}" for e in EXPERTS]
    df["best_rec"] = df[rec_cols].min(axis=1)
    df["best_expert"] = df[rec_cols].idxmin(axis=1).str.replace("rec_", "", regex=False)
    try:
        df["nsf_rec_quartile"] = pd.qcut(df["rec_nsf"], 4, labels=list(QUARTILE_LABELS))
    except ValueError as e:
        logger.error("[export] qcut on rec_nsf failed (duplicate edges?): %s", e)
        raise
    df["gap_realnvp"] = (df["rec_realnvp"] - df["rec_nsf"]) / df["rec_nsf"]
    df["gap_nice_mix"] = (df["rec_nice_mix"] - df["rec_nsf"]) / df["rec_nsf"]

    out = results_dir(seed_index)
    csv_path = os.path.join(out, "per_sample_rec.csv")
    df.to_csv(csv_path, index=False)
    logger.info("[export] wrote %s (%d samples)", csv_path, n_seen)

    recons = {"y": torch.cat(cache_y), "x_true": torch.cat(cache_xt),
              "blur_sigma": blur_sigma, "scale": scale, "noise_sigma": noise_sigma}
    for e in EXPERTS:
        recons[f"x_hat_{e}"] = torch.cat(cache[f"x_hat_{e}"])
    pt_path = os.path.join(out, "recons.pt")
    torch.save(recons, pt_path)
    logger.info("[export] wrote %s", pt_path)
    return csv_path


def main() -> None:
    ap = argparse.ArgumentParser(description=f"{__abbr__} v{__version__} Step-0 export")
    ap.add_argument("--expert-dirs", nargs=3, required=True,
                    help="3 per-expert dirs from split_experts, expert_set order")
    ap.add_argument("--mixture-dir", required=True,
                    help="2.3-A mixture run dir (for cfg: blur/scale/noise/z-bank)")
    ap.add_argument("--seed-index", type=int, default=0, choices=sorted(SEED_RNG))
    ap.add_argument("--smoke", type=int, default=0, help="limit to ~N samples (0 = full)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    setup_logging()
    if args.smoke:
        logger.info("[export] SMOKE mode: ~%d samples", args.smoke)
    export(args.expert_dirs, args.mixture_dir, args.seed_index, args.smoke, args.device)


if __name__ == "__main__":
    main()
