# =============================================================================
# STEP-1_3 v0.3 -- experiments.step_1_3.aggregate_beta
# Purpose: collect RECGATE runs across beta under a results root, emit the
#          beta-sweep table + per-beta label table + core plots 1-6 + the
#          corrected Stage-1.3 verdict.
# CONVENTION: missing data / keys -> logger.error + raise. No fallback / mock.
# Plots: 1 Neff vs beta, 2 mean weight vs beta, 3 fwd_rel vs beta + NSF-only
#        baseline line, 4 mixture NLL vs beta, 5 score_argmin counts per beta,
#        6 rec_argmin counts per beta.
# Decision (v0.3 -- SAME-beta evaluation, dual labels):
#   For each beta i (one row):
#     diversity_i = max(mean_weight[i]) < COLLAPSE_THRESH
#     improve_i   = min(soft_i,hard_i) <  base_i - PARITY_EPS   (meaningful)
#     parity_i    = |min(soft_i,hard_i) - base_i| <= PARITY_EPS  (tie)
#   specialization = max non-top-expert rec_argmin share >= SPEC_MIN
#                    (rec_argmin is gate-independent; read from score_diag)
#   strict_success  = EXISTS i: diversity_i AND improve_i
#   research_signal = (EXISTS i: diversity_i AND parity_i) AND specialization
# Changelog (v0.2 -> v0.3):
#   * RECGATE-AGG v0.3: verdict no longer mixes evidence across beta rows.
#     diversity AND fwd-improvement must hold at the SAME beta. |dfwd|<=1e-3
#     counts as PARITY not a win. Emits TWO labels (strict_success,
#     research_signal); research_signal uses rec_argmin specialization so
#     parity-with-specialization is kept as a legit Stage-2.3 trigger.
#     Adds per-beta label table + CLI flags (--parity-eps/--spec-min/
#     --collapse-thresh). Fixes the false "SUCCEEDS" on the CB run.
# Changelog (v0.1 -> v0.2):
#   * Decision rule keys collapse on CROSS-SAMPLE usage concentration
#     (max mean weight >= 0.70), not per-sample Neff_mean.
# Changelog (NEW in v0.1):
#   * Introduced.
# Update summary:
#   v0.3 makes the verdict auditable and honest: a per-beta table shows which
#   beta (if any) is both diverse and reconstruction-competitive, and the two
#   labels separate "actually beats NSF" from "ties NSF but specializes".
# =============================================================================
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
__version__ = "0.3"
__abbr__ = "STEP-1_3"

# defaults (overridable via CLI)
PARITY_EPS = 1e-3        # |dfwd_rel| <= this => parity, not a win
SPEC_MIN = 0.01          # non-top rec_argmin share >= this => specialization
COLLAPSE_THRESH = 0.70   # max mean weight >= this => collapsed (no diversity)


def _load(results_root: Path):
    if not results_root.exists():
        logger.error("[aggregate_beta] %s missing", results_root)
        raise FileNotFoundError(results_root)
    runs = []
    for d in sorted(results_root.iterdir()):
        rpt = d / "report.json"
        if not rpt.exists():
            continue
        data = json.loads(rpt.read_text())
        runs.append(data)
    if not runs:
        logger.error("[aggregate_beta] no report.json under %s", results_root)
        raise ValueError("no runs found")
    runs.sort(key=lambda r: r["stage13_cfg"]["beta"])
    return runs


def aggregate(results_root: Path, *, parity_eps: float = PARITY_EPS,
              spec_min: float = SPEC_MIN,
              collapse_thresh: float = COLLAPSE_THRESH) -> dict:
    runs = _load(results_root)
    names = runs[0]["expert_names"]; K = len(names)
    betas = [r["stage13_cfg"]["beta"] for r in runs]
    neff = [r["gate"]["Neff_mean"] for r in runs]
    soft = [r["reconstruction"]["soft_fwd_rel"] for r in runs]
    hard = [r["reconstruction"]["hard_fwd_rel"] for r in runs]
    base = [r["reconstruction"]["nsf_only_fwd_rel"] for r in runs]
    mixnll = [r["reconstruction"]["mixture_NLL"] for r in runs]
    weights = [r["gate"]["mean_weight_per_expert"] for r in runs]
    sc_argmin = [r["score_diag"]["score_argmin_counts"] for r in runs]
    rec_argmin = [r["score_diag"]["rec_argmin_counts"] for r in runs]

    sep = "=" * 78
    lines = [sep, "RECGATE v0.3 -- beta sweep (alpha=1, gamma=0)", sep,
             f"  {'beta':>5} {'Neff':>7} {'soft_fwd':>9} {'hard_fwd':>9} "
             f"{'nsf_fwd':>8} {'mixNLL':>10}  usage(" + ",".join(names) + ")"]
    for i, b in enumerate(betas):
        wfmt = ",".join(f"{w:.2f}" for w in weights[i])
        lines.append(f"  {b:>5} {neff[i]:>7.3f} {soft[i]:>9.4f} {hard[i]:>9.4f} "
                     f"{base[i]:>8.4f} {mixnll[i]:>10.1f}  [{wfmt}]")

    # ---- SAME-beta per-row evaluation (no cross-row mixing) ---------------
    per_beta = []
    for i in range(len(betas)):
        best_fwd = min(soft[i], hard[i])
        delta = best_fwd - base[i]                      # <0 means better (lower)
        diversity = max(weights[i]) < collapse_thresh
        improve = delta < -parity_eps                   # meaningfully better
        parity = abs(delta) <= parity_eps               # tie within eps
        per_beta.append({"beta": betas[i], "diversity": diversity,
                         "improve": improve, "parity": parity,
                         "best_fwd": best_fwd, "delta_vs_nsf": delta,
                         "max_weight": max(weights[i])})

    # ---- specialization from rec_argmin (gate-independent) ----------------
    # share of samples won (absolute rec) by any NON-top expert, max over beta
    spec_share = 0.0
    for counts in rec_argmin:
        tot = sum(counts)
        if tot <= 0:
            continue
        top = max(range(K), key=lambda k: counts[k])
        non_top = (tot - counts[top]) / tot
        spec_share = max(spec_share, non_top)
    specialization = spec_share >= spec_min

    strict_betas = [pb["beta"] for pb in per_beta
                    if pb["diversity"] and pb["improve"]]
    research_betas = [pb["beta"] for pb in per_beta
                      if pb["diversity"] and pb["parity"]]
    strict_success = len(strict_betas) > 0
    research_signal = (len(research_betas) > 0) and specialization

    # ---- per-beta label table (auditable) ---------------------------------
    lines += ["", "Per-beta labels (same-row evaluation):",
              f"  {'beta':>5} {'diverse':>8} {'improve':>8} {'parity':>7} "
              f"{'dfwd_vs_nsf':>12} {'max_w':>7}"]
    for pb in per_beta:
        lines.append(f"  {pb['beta']:>5} {str(pb['diversity']):>8} "
                     f"{str(pb['improve']):>8} {str(pb['parity']):>7} "
                     f"{pb['delta_vs_nsf']:>+12.5f} {pb['max_weight']:>7.2f}")
    lines += ["",
              f"  specialization: non-top rec_argmin share = {spec_share:.4f} "
              f"(>= {spec_min} ? {specialization})  thresholds: parity_eps="
              f"{parity_eps}, collapse={collapse_thresh}"]

    # ---- dual verdict ------------------------------------------------------
    if strict_success:
        verdict = (f"STRICT SUCCESS -- beta(s) {strict_betas} are BOTH diverse "
                   f"(max weight < {collapse_thresh}) AND meaningfully beat "
                   f"NSF-only (dfwd < -{parity_eps}). -> proceed to Stage 2.3.")
    elif research_signal:
        verdict = (f"RESEARCH SIGNAL (not strict success) -- beta(s) "
                   f"{research_betas} are diverse AND at fwd PARITY with NSF-only "
                   f"(|dfwd| <= {parity_eps}), and rec_argmin shows non-NSF "
                   f"specialization (share {spec_share:.3f}). Parity-with-"
                   f"specialization is a legit Stage-2.3 trigger, NOT a strict win.")
    else:
        # explain WHY, without mixing rows
        any_div = any(pb["diversity"] for pb in per_beta)
        any_imp = any(pb["improve"] for pb in per_beta)
        verdict = (f"NO SUCCESS -- no single beta is both diverse and "
                   f"reconstruction-competitive. diversity_at_any_beta={any_div}, "
                   f"improvement_at_any_beta={any_imp}, specialization={specialization}"
                   f" (share {spec_share:.3f}). Diverse betas are worse than "
                   f"NSF-only; the parity beta (if any) is collapsed. Experts not "
                   f"yet complementary at a usable operating point.")
    lines += ["", f"DECISION: {verdict}", sep]
    text = "\n".join(lines); print(text)

    # ---- plots 1-6 ---------------------------------------------------------
    def _line(xs, ys, ylabel, title, path, baseline=None):
        fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=120)
        ax.plot(xs, ys, marker="o", label=ylabel)
        if baseline is not None:
            ax.axhline(baseline, color="r", ls="--", label="NSF-only baseline")
            ax.legend()
        ax.set_xlabel("beta"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

    def _stacked(counts, title, path):
        fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=120)
        bottoms = [0] * len(betas)
        xs = [str(b) for b in betas]
        for k in range(K):
            vals = [counts[i][k] for i in range(len(betas))]
            ax.bar(xs, vals, bottom=bottoms, label=names[k])
            bottoms = [bottoms[i] + vals[i] for i in range(len(betas))]
        ax.set_xlabel("beta"); ax.set_ylabel("argmin count")
        ax.set_title(title); ax.legend()
        fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

    p = results_root
    _line(betas, neff, "Neff", "1. Neff vs beta", p / "p1_neff_vs_beta.png")
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=120)
    for k in range(K):
        ax.plot(betas, [weights[i][k] for i in range(len(betas))],
                marker="o", label=names[k])
    ax.set_xlabel("beta"); ax.set_ylabel("mean weight"); ax.set_ylim(0, 1)
    ax.set_title("2. Mean expert weight vs beta"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(p / "p2_weight_vs_beta.png", bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=120)
    ax.plot(betas, soft, marker="o", label="soft_fwd_rel")
    ax.plot(betas, hard, marker="s", label="hard_fwd_rel")
    ax.axhline(base[0], color="r", ls="--", label="NSF-only baseline")
    ax.set_xlabel("beta"); ax.set_ylabel("fwd_rel"); ax.set_title("3. fwd_rel vs beta")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(p / "p3_fwdrel_vs_beta.png", bbox_inches="tight"); plt.close(fig)
    _line(betas, mixnll, "mixture NLL", "4. Mixture NLL vs beta",
          p / "p4_mixnll_vs_beta.png")
    _stacked(sc_argmin, "5. score argmin counts per beta", p / "p5_score_argmin.png")
    _stacked(rec_argmin, "6. rec argmin counts per beta", p / "p6_rec_argmin.png")

    out = {"betas": betas, "neff": neff, "soft_fwd_rel": soft,
           "hard_fwd_rel": hard, "nsf_only_fwd_rel": base, "mixture_NLL": mixnll,
           "mean_weight_per_expert": weights, "rec_argmin_counts": rec_argmin,
           "per_beta_labels": per_beta,
           "specialization_share": spec_share,
           "specialization": specialization,
           "strict_success": strict_success, "strict_betas": strict_betas,
           "research_signal": research_signal, "research_betas": research_betas,
           "thresholds": {"parity_eps": parity_eps, "spec_min": spec_min,
                          "collapse_thresh": collapse_thresh},
           "verdict": verdict}
    (p / "beta_summary.json").write_text(json.dumps(out, indent=2))
    (p / "beta_summary.txt").write_text(text)
    logger.info("[aggregate_beta] wrote beta_summary + plots 1-6")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="./CSMF2/experiments/step_1_3/results")
    ap.add_argument("--parity-eps", type=float, default=PARITY_EPS,
                    help="|dfwd_rel| <= this counts as parity, not a win")
    ap.add_argument("--spec-min", type=float, default=SPEC_MIN,
                    help="non-top rec_argmin share >= this => specialization")
    ap.add_argument("--collapse-thresh", type=float, default=COLLAPSE_THRESH,
                    help="max mean weight >= this => collapsed (no diversity)")
    a = ap.parse_args()
    aggregate(Path(a.results_root), parity_eps=a.parity_eps,
              spec_min=a.spec_min, collapse_thresh=a.collapse_thresh)
