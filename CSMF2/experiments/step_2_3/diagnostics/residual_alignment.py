# =============================================================================
# NWS v0.4 -- CSMF2.experiments.step_2_3.diagnostics.residual_alignment
# Purpose: Step 2.3-NWS bonus (best R2 bridge). For each weak expert, test
#          whether its correction direction points along NSF's leftover error:
#            r_nsf = y - A(x_nsf);  d_k = A(x_k - x_nsf) = A(x_k) - A(x_nsf);
#            cos(r_nsf, d_k).  Positive (esp. on NSF-worst samples) => usable as
#          an R2 corrector; ~0/negative => not useful. Renders example panels.
# CONVENTION: No silent fallback. Failure -> logger.error + raise. Cosines in f64.
# Changelog (NEW in v0.4):
#   * Introduced (real-surface version). Reads recons.pt (pixel x_hat_k, y) +
#     per_sample_rec.csv; uses pixel_A for A; cosine per weak expert, split by
#     NSF rec quartile (Q4 = NSF-worst). Emits residual_alignment_report.json +
#     residual_alignment_hist.png + near_winner_examples_grid.png.
# Update summary:
#   Most direct evidence for whether R2's residual heads have a signal to
#   correct, on frozen outputs, before building anything.
# =============================================================================
from __future__ import annotations

import argparse
import logging
import os

from CSMF2.experiments.step_2_3.diagnostics.nws_common import (
    NON_NSF, EXPERTS, SEED_RNG, results_dir, plots_dir, save_report,
    setup_logging, to_f64, pixel_A,
)

logger = logging.getLogger(__name__)
__version__ = "0.4"
__abbr__ = "NWS"


def _load(seed_index: int):
    import torch
    import pandas as pd
    out = results_dir(seed_index)
    pt, csv = os.path.join(out, "recons.pt"), os.path.join(out, "per_sample_rec.csv")
    for p in (pt, csv):
        if not os.path.exists(p):
            logger.error("[residual] %s missing -- run export_experts_rec first", p)
            raise FileNotFoundError(p)
    return torch.load(pt, map_location="cpu"), pd.read_csv(csv)


def _cosine(a, b, eps=1e-12):
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1) * b.norm(dim=1) + eps
    return num / den


def analyse(seed_index: int) -> dict:
    import numpy as np
    recons, df = _load(seed_index)
    blur_sigma = float(recons["blur_sigma"]); scale = int(recons["scale"])
    y = to_f64(recons["y"])
    Ax_nsf = pixel_A(to_f64(recons["x_hat_nsf"]), blur_sigma, scale)
    r_nsf = y - Ax_nsf
    q4 = (df["nsf_rec_quartile"].values == "Q4")

    summary, cos_store = {}, {}
    for e in NON_NSF:
        d_k = pixel_A(to_f64(recons[f"x_hat_{e}"]), blur_sigma, scale) - Ax_nsf
        cos = _cosine(r_nsf, d_k).cpu().numpy()
        cos_store[e] = cos
        cq4 = cos[q4] if q4.any() else np.array([])
        summary[e] = {"cos_mean": float(np.mean(cos)), "cos_median": float(np.median(cos)),
                      "cos_mean_nsf_worst_q4": float(np.mean(cq4)) if cq4.size else None,
                      "frac_positive_q4": float(np.mean(cq4 > 0)) if cq4.size else None}

    report = {"cell": "s2/n0.05", "seed_index": seed_index, "n_samples": int(len(df)),
              "per_expert": summary,
              "note": ("positive cosine on Q4 (NSF-worst) => correction direction may "
                       "help R2; ~0/negative => not useful as a corrector.")}
    save_report(os.path.join(results_dir(seed_index), "residual_alignment_report.json"), report)
    _plot_hist(cos_store, seed_index)
    _plot_examples(recons, df, seed_index)
    logger.info("[residual] cos_mean: %s",
                {e: round(summary[e]["cos_mean"], 3) for e in NON_NSF})
    return report


def _plot_hist(cos_store, seed_index) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for e, cos in cos_store.items():
        ax.hist(cos, bins=60, alpha=0.5, label=e)
    ax.axvline(0.0, c="k")
    ax.set_xlabel("cos(r_nsf, A(x_k - x_nsf))"); ax.set_ylabel("count"); ax.legend()
    ax.set_title("Residual alignment (R2 corrector bridge)")
    fig.savefig(os.path.join(plots_dir(seed_index), "residual_alignment_hist.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_examples(recons, df, seed_index, k=16) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    cand = df.index[df["best_expert"].isin(list(NON_NSF))].tolist()
    if not cand:
        logger.info("[residual] no non-NSF best_expert samples -- example grid skipped")
        return
    idx = cand[:k]
    cols = ["x_true"] + [f"x_hat_{e}" for e in EXPERTS]
    fig, axes = plt.subplots(len(idx), len(cols), figsize=(2 * len(cols), 2 * len(idx)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idx):
        for c, key in enumerate(cols):
            axes[r, c].imshow(recons[key][i].reshape(28, 28).cpu().numpy(), cmap="gray")
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(key.replace("x_hat_", ""), fontsize=8)
    fig.suptitle("Non-NSF near-winner examples (visual sanity)")
    fig.savefig(os.path.join(plots_dir(seed_index), "near_winner_examples_grid.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=f"{__abbr__} v{__version__} residual alignment")
    ap.add_argument("--seed-index", type=int, default=0, choices=sorted(SEED_RNG))
    args = ap.parse_args()
    setup_logging()
    analyse(args.seed_index)


if __name__ == "__main__":
    main()
