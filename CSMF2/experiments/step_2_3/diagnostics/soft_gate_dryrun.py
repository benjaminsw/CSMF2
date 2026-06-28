# =============================================================================
# NWS v0.4 -- CSMF2.experiments.step_2_3.diagnostics.soft_gate_dryrun
# Purpose: Step 2.3-NWS addition 2. WITHOUT training, ask whether a
#          reconstruction-aware soft gate would beat NSF. Build weights
#          w_k ∝ exp(-rec_k / T) from frozen per_expert_rec, blend the cached
#          pixel x_hat_k, score ||pixel_A(x_mix) - y||^2 (sum-over-pixels, to
#          match per_expert_rec) over a T-curve vs NSF's own mean rec.
# CONVENTION: No silent fallback. Failure -> logger.error + raise. A blend that
#          never beats NSF => mixture-base is risky unless the gate goes near
#          one-hot (R2 Guard 1).
# NOTE: x_mix is a POINT-estimate blend of z-bank-mean reconstructions, scored
#   once (no z-bank), whereas per_expert_rec is a z-bank mean -- a small,
#   documented metric mismatch; both are sum-over-pixels in y-space.
# Changelog (NEW in v0.4):
#   * Introduced (real-surface version). Reads recons.pt (pixel x_hat_k, y,
#     blur/scale) + per_sample_rec.csv; sweeps T; uses pixel_A; emits
#     soft_gate_dryrun_report.json (best_soft_gate_rec feeds the sweep verdict)
#     + soft_gate_temperature_curve.png with the nsf_mean_rec baseline.
# Update summary:
#   The training-free "is this routable" check (RECMET #8) on frozen outputs,
#   on the same metric family as the 2.3-A argmin.
# =============================================================================
from __future__ import annotations

import argparse
import logging
import os

from CSMF2.experiments.step_2_3.diagnostics.nws_common import (
    EXPERTS, SEED_RNG, results_dir, plots_dir, save_report, setup_logging,
    to_f64, pixel_A,
)

logger = logging.getLogger(__name__)
__version__ = "0.4"
__abbr__ = "NWS"

T_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)


def _load(seed_index: int):
    import torch
    import pandas as pd
    out = results_dir(seed_index)
    pt, csv = os.path.join(out, "recons.pt"), os.path.join(out, "per_sample_rec.csv")
    for p in (pt, csv):
        if not os.path.exists(p):
            logger.error("[soft-gate] %s missing -- run export_experts_rec first", p)
            raise FileNotFoundError(p)
    return torch.load(pt, map_location="cpu"), pd.read_csv(csv)


def dryrun(seed_index: int) -> dict:
    import torch
    recons, df = _load(seed_index)
    blur_sigma = float(recons["blur_sigma"]); scale = int(recons["scale"])
    y = to_f64(recons["y"])                                  # (N, Dy)
    X = torch.stack([to_f64(recons[f"x_hat_{e}"]) for e in EXPERTS], dim=1)  # (N,K,784)
    R = torch.stack([torch.tensor(df[f"rec_{e}"].values, dtype=torch.float64)
                     for e in EXPERTS], dim=1)               # (N,K)
    nsf_mean_rec = float(df["rec_nsf"].mean())

    curve = []
    for T in T_GRID:
        w = torch.softmax(-R / T, dim=1)                     # (N,K)
        x_mix = (w.unsqueeze(-1) * X).sum(dim=1)             # (N,784)
        diff = pixel_A(x_mix, blur_sigma, scale) - y         # (N,Dy)
        score = float(diff.pow(2).sum(dim=1).mean())         # sum-pixels, mean-batch
        curve.append({"T": T, "soft_gate_rec": score, "beats_nsf_mean": score < nsf_mean_rec})

    best = min(curve, key=lambda r: r["soft_gate_rec"])
    report = {"cell": "s2/n0.05", "seed_index": seed_index, "nsf_mean_rec": nsf_mean_rec,
              "best_T": best["T"], "best_soft_gate_rec": best["soft_gate_rec"],
              "best_beats_nsf_mean": best["beats_nsf_mean"], "curve": curve}
    save_report(os.path.join(results_dir(seed_index), "soft_gate_dryrun_report.json"), report)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([r["T"] for r in curve], [r["soft_gate_rec"] for r in curve], marker="o")
    ax.axhline(nsf_mean_rec, ls="--", c="grey", label=f"NSF mean rec {nsf_mean_rec:.4g}")
    ax.set_xscale("log"); ax.set_xlabel("temperature T (log)")
    ax.set_ylabel("soft-gate mixture rec"); ax.legend()
    ax.set_title("Soft-gate temperature dry-run")
    fig.savefig(os.path.join(plots_dir(seed_index), "soft_gate_temperature_curve.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("[soft-gate] best T=%s rec=%.4g beats_nsf=%s",
                best["T"], best["soft_gate_rec"], best["beats_nsf_mean"])
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=f"{__abbr__} v{__version__} soft-gate dry-run")
    ap.add_argument("--seed-index", type=int, default=0, choices=sorted(SEED_RNG))
    args = ap.parse_args()
    setup_logging()
    dryrun(args.seed_index)


if __name__ == "__main__":
    main()
