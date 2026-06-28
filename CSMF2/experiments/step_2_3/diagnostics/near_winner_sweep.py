# =============================================================================
# NWS v0.4 -- CSMF2.experiments.step_2_3.diagnostics.near_winner_sweep
# Purpose: Step 2.3-NWS Steps 1-2. Sweep near-winner tolerance tau over the
#          frozen per_sample_rec.csv, test STRUCTURE (digit/noise) + CONCENTRATION
#          (NSF-worst quartile), compute the oracle, pick the honest tau, emit the
#          NWS-SIGNAL / NWS-NO-SIGNAL verdict. Reads the CSV; no GPU.
# CONVENTION: No silent fallback. Failure -> logger.error + raise. Raising tau
#          RELAXES the threshold; it never CREATES complementarity.
# REFERENCE: verdict compares oracle / soft-gate to NSF's OWN per_expert_rec
#          mean (nsf_mean_rec). 0.156 is the V2 soft_fwd_rel (different metric),
#          reported for context only.
# Changelog (v0.2 -> v0.4):
#   * Reference switched from 0.156 to nsf_mean_rec (per_expert_rec is the
#     rec_argmin metric NWS re-credits; 0.156 is a different reconstruction).
#   * oracle_summary uses oracle_rec vs nsf_mean_rec. Plot 5 baseline = nsf_mean_rec.
#   * tau grid / structure / concentration / verbose verdict unchanged in spirit.
# Update summary:
#   Decision core: turns cached per_expert_rec into a tau value + SIGNAL/NO-SIGNAL
#   call that feeds the R1-vs-R2 choice, on the same metric as the 2.3-A argmin.
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import os

from CSMF2.experiments.step_2_3.diagnostics.nws_common import (
    EXPERTS, NON_NSF, TAUS, QUARTILE_LABELS, SEED_RNG, V2_SOFT_FWD_REL_CONTEXT,
    SIGNAL_TAU_MAX, results_dir, plots_dir, save_report, setup_logging,
    classify_verdict,
)

logger = logging.getLogger(__name__)
__version__ = "0.4"
__abbr__ = "NWS"


def _load_csv(seed_index: int):
    import pandas as pd
    path = os.path.join(results_dir(seed_index), "per_sample_rec.csv")
    if not os.path.exists(path):
        logger.error("[sweep] %s missing -- run export_experts_rec first", path)
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _near_win_mask(df, tau: float):
    thr = (1.0 + tau) * df["best_rec"]
    return {e: (df[f"rec_{e}"] <= thr) for e in EXPERTS}


def _class_concentration(df, mask_nonnsf) -> float:
    sub = df[mask_nonnsf]
    if len(sub) == 0:
        return 0.0
    counts = sub["class"].value_counts(normalize=True)
    n_classes = max(int(df["class"].nunique()), 1)
    return float(counts.max() - 1.0 / n_classes)


def sweep(seed_index: int) -> dict:
    df = _load_csv(seed_index)
    n = len(df)
    oracle_rec = float(df["best_rec"].mean())
    nsf_mean_rec = float(df["rec_nsf"].mean())

    per_tau, first_nonnsf_tau = [], None
    for tau in TAUS:
        mask = _near_win_mask(df, tau)
        counts = {e: int(mask[e].sum()) for e in EXPERTS}
        nonnsf_mask = mask["realnvp"] | mask["nice_mix"]
        n_nonnsf = int(nonnsf_mask.sum())
        q4 = df["nsf_rec_quartile"] == "Q4"
        q4_overlap = float((nonnsf_mask & q4).sum() / n_nonnsf) if n_nonnsf else 0.0
        clustering = _class_concentration(df, nonnsf_mask)
        digit_dist = (df[nonnsf_mask]["class"].value_counts().sort_index().to_dict()
                      if n_nonnsf else {})
        noise_dist = (df[nonnsf_mask]["noise_sigma"].value_counts().to_dict()
                      if n_nonnsf else {})
        per_tau.append({"tau": tau, "counts": counts, "n_nonnsf": n_nonnsf,
                        "q4_overlap": q4_overlap, "class_concentration": clustering,
                        "digit_dist": {int(k): int(v) for k, v in digit_dist.items()},
                        "noise_dist": {float(k): int(v) for k, v in noise_dist.items()}})
        if first_nonnsf_tau is None and n_nonnsf > 0:
            first_nonnsf_tau = tau

    base = per_tau[0]
    if base["tau"] == 0.0 and base["n_nonnsf"] > 0:
        logger.warning("[sweep] tau=0 has %d non-NSF near-wins (exact ties?) -- "
                       "expected ~0 to match 5000/0/0 argmin baseline", base["n_nonnsf"])

    chosen = None
    for r in per_tau:
        if (r["tau"] <= SIGNAL_TAU_MAX and r["n_nonnsf"] > 0
                and r["q4_overlap"] > 0.25 and r["class_concentration"] > 0.25):
            chosen = r["tau"]; break

    soft_gate_rec = None
    sg = os.path.join(results_dir(seed_index), "soft_gate_dryrun_report.json")
    if os.path.exists(sg):
        try:
            soft_gate_rec = float(json.load(open(sg))["best_soft_gate_rec"])
        except (OSError, KeyError, ValueError) as e:
            logger.error("[sweep] soft_gate_dryrun_report.json unreadable: %s", e)
            raise
    else:
        logger.info("[sweep] no soft-gate report yet -- verdict provisional")

    ref = chosen if chosen is not None else first_nonnsf_tau
    ref_row = next((r for r in per_tau if r["tau"] == ref), per_tau[-1])
    verdict = classify_verdict(
        first_nonnsf_tau=first_nonnsf_tau, q4_overlap=ref_row["q4_overlap"],
        class_concentration=ref_row["class_concentration"],
        oracle_rec=oracle_rec, nsf_mean_rec=nsf_mean_rec, soft_gate_rec=soft_gate_rec)

    report = {"cell": "s2/n0.05", "seed_index": seed_index,
              "rng_seed": SEED_RNG[seed_index], "n_samples": n,
              "metric": "per_expert_rec (z-bank, sum-over-pixels)",
              "v2_soft_fwd_rel_context": V2_SOFT_FWD_REL_CONTEXT,
              "oracle_rec": oracle_rec, "nsf_mean_rec": nsf_mean_rec,
              "first_nonnsf_tau": first_nonnsf_tau, "chosen_tau": chosen,
              "soft_gate_rec": soft_gate_rec, "per_tau": per_tau, "verdict": verdict}
    save_report(os.path.join(results_dir(seed_index), "near_winner_sweep_report.json"), report)
    save_report(os.path.join(results_dir(seed_index), "oracle_summary_report.json"),
                {"oracle_rec": oracle_rec, "nsf_mean_rec": nsf_mean_rec,
                 "oracle_beats_nsf_mean": oracle_rec < nsf_mean_rec,
                 "v2_soft_fwd_rel_context": V2_SOFT_FWD_REL_CONTEXT})

    import pandas as pd
    pd.DataFrame([{"tau": r["tau"], "n_nonnsf": r["n_nonnsf"],
                   "q4_overlap": r["q4_overlap"],
                   "class_concentration": r["class_concentration"],
                   **{f"count_{e}": r["counts"][e] for e in EXPERTS}}
                  for r in per_tau]).to_csv(
        os.path.join(results_dir(seed_index), "near_winner_sweep.csv"), index=False)

    _plots(df, per_tau, oracle_rec, nsf_mean_rec, chosen, seed_index)
    logger.info("[sweep] verdict=%s chosen_tau=%s oracle=%.4g nsf_mean=%.4g",
                verdict["label"], chosen, oracle_rec, nsf_mean_rec)
    return report


def _plots(df, per_tau, oracle_rec, nsf_mean_rec, chosen, seed_index) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    pdir = plots_dir(seed_index)
    pd_chosen = chosen if chosen is not None else per_tau[-1]["tau"]
    taus = [r["tau"] for r in per_tau]

    fig, ax = plt.subplots()
    for e in EXPERTS:
        ax.plot(taus, [r["counts"][e] for r in per_tau], marker="o", label=e)
    ax.axvline(SIGNAL_TAU_MAX, ls="--", c="grey", label=f"signal tau<= {SIGNAL_TAU_MAX}")
    ax.set_xlabel("tau"); ax.set_ylabel("near-winner count"); ax.legend()
    ax.set_title("Near-winner count vs tau")
    fig.savefig(os.path.join(pdir, "near_winner_count_vs_tau.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    for e in NON_NSF:
        fig, ax = plt.subplots()
        ax.hist(df[f"gap_{e}"].clip(-1, 5), bins=60)
        ax.axvline(0.0, c="k"); ax.set_xlabel("(rec_k - rec_nsf)/rec_nsf")
        ax.set_ylabel("count"); ax.set_title(f"Relative gap to NSF -- {e}")
        fig.savefig(os.path.join(pdir, f"relative_gap_hist_{e}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

    mask = _near_win_mask(df, pd_chosen)
    classes = sorted(df["class"].unique())
    grid = np.array([[int((mask[e] & (df["class"] == c)).sum()) for c in classes]
                     for e in NON_NSF])
    fig, ax = plt.subplots()
    im = ax.imshow(grid, aspect="auto")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(NON_NSF))); ax.set_yticklabels(list(NON_NSF))
    ax.set_xlabel("digit class"); ax.set_title(f"Non-NSF near-wins by digit (tau={pd_chosen})")
    fig.colorbar(im, ax=ax)
    fig.savefig(os.path.join(pdir, "near_winner_by_digit_heatmap.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    nonnsf = mask["realnvp"] | mask["nice_mix"]
    qcounts = [int((nonnsf & (df["nsf_rec_quartile"] == q)).sum()) for q in QUARTILE_LABELS]
    fig, ax = plt.subplots()
    ax.bar(list(QUARTILE_LABELS), qcounts)
    ax.set_xlabel("NSF rec quartile (Q1 best .. Q4 worst)")
    ax.set_ylabel("non-NSF near-winner count")
    ax.set_title(f"Concentration on NSF-weak samples (tau={pd_chosen})")
    fig.savefig(os.path.join(pdir, "nsf_weakness_concentration.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(["NSF mean", "oracle min_k"], [nsf_mean_rec, oracle_rec])
    ax.axhline(nsf_mean_rec, ls="--", c="grey", label="NSF mean rec")
    ax.set_ylabel("per_expert_rec"); ax.legend(); ax.set_title("Oracle vs NSF (same metric)")
    fig.savefig(os.path.join(pdir, "oracle_vs_nsf_bar.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=f"{__abbr__} v{__version__} tau sweep")
    ap.add_argument("--seed-index", type=int, default=0, choices=sorted(SEED_RNG))
    args = ap.parse_args()
    setup_logging()
    sweep(args.seed_index)


if __name__ == "__main__":
    main()
