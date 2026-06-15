# =============================================================================
# STEP-1_1_1 v0.1 -- experiments.step_1_1_1.aggregate_ablation
# Purpose: aggregate the latent-shape penalty (lambda_lat) ablation into a
#          PER-EXPERT self-comparison. No cross-expert "winner": for each of
#          {NICE, RealNVP, NSF} independently, compare every lambda cell
#          against that expert's own lambda=0 baseline and emit a verdict.
# CONVENTION: no silent fallback / mock / dummy / placeholder. Missing keys or
#             unreadable reports -> logger.error + raise.
# Changelog (NEW in v0.1):
#   * Introduced.
# Update summary:
#   v0.1 reads every cell's report.json under a results root, groups by expert,
#   and for each (expert, lambda != 0) cell classifies improves / neutral /
#   regresses versus that expert's lambda=0 cell. A KS-win whose conditioning
#   collapsed is forced to 'regresses' (the WIN-but-collapsed guard). Emits a
#   per-expert console table + four lambda-sweep PNGs (KS, FZDY, NLL, fwd_rel).
# =============================================================================
from __future__ import annotations
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_1_1"

# Decision thresholds (mirror the CCR Phase-0 backport cell rule).
KS_REL_IMPROVE = 0.10     # KS must drop >=10% vs baseline to count as improves
NLL_REL_TOL    = 0.05     # NLL may worsen by <=5%
FWD_REL_TOL    = 0.10     # fwd_rel may worsen by <=10%


def _require(d: dict, path: tuple[str, ...], cell: Path):
    """Walk nested keys; missing key -> logger.error + raise (no fallback)."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            logger.error("[aggregate_ablation] %s missing key path %s",
                         cell, "/".join(path))
            raise KeyError(f"{cell}: missing {'/'.join(path)}")
        cur = cur[k]
    return cur


def _read_cell(report_path: Path) -> dict:
    """Extract the ablation metric vector from one run's report.json."""
    try:
        rep = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("[aggregate_ablation] cannot read %s: %s",
                     report_path, exc)
        raise
    cell = report_path.parent
    per_epoch = _require(rep, ("sanity_per_epoch",), cell)
    if not per_epoch:
        logger.error("[aggregate_ablation] %s has empty sanity_per_epoch", cell)
        raise ValueError(f"{cell}: empty sanity_per_epoch")
    last = per_epoch[-1]
    fz = _require(last, ("fixed_z_different_y",), cell)
    return {
        "expert":      _require(rep, ("cfg", "expert"), cell),
        "seed":        _require(rep, ("cfg", "seed"), cell),
        "lambda":      float(_require(rep, ("cfg", "latent_moment_lambda"), cell)),
        "ks":          float(_require(last, ("latent", "ks"), cell)),
        "nll":         float(_require(rep, ("final_test_nll",), cell)),
        "fwd_rel":     _require(rep, ("summary", "fwd_rel_last"), cell),
        "fzdy":        fz.get("sensitivity_mean"),
        "fzdy_pass":   bool(fz.get("passed", False)),
        "cond_pass":   bool(_require(rep, ("summary", "conditioning",
                                           "conditioning_pass"), cell)),
        "exit_ok":     bool(_require(rep, ("exit_criteria_met",), cell)),
        "dir":         str(cell),
    }


def _verdict(cell: dict, base: dict) -> str:
    """improves / neutral / regresses versus that expert's lambda=0 cell.
    A KS-win with collapsed conditioning is forced to 'regresses'."""
    # safety first: conditioning collapse or worse exit -> regresses
    if not cell["cond_pass"]:
        return "regresses"          # WIN-but-collapsed guard
    # NLL is stored as a LOSS (lower better) and is typically NEGATIVE, so a
    # multiplicative (1+tol) ceiling would invert for negatives. Use an
    # additive tolerance on |base|: worse = higher than base + tol*|base|.
    nll_ok = cell["nll"] <= base["nll"] + NLL_REL_TOL * abs(base["nll"])
    fwd_ok = True
    if base["fwd_rel"] is not None and cell["fwd_rel"] is not None:
        fwd_ok = cell["fwd_rel"] <= base["fwd_rel"] * (1.0 + FWD_REL_TOL)
    if not (nll_ok and fwd_ok):
        return "regresses"
    ks_improved = cell["ks"] <= base["ks"] * (1.0 - KS_REL_IMPROVE)
    return "improves" if ks_improved else "neutral"


def _plot_sweep(per_expert: dict, metric: str, ylabel: str,
                out_path: Path, *, lower_better: bool) -> None:
    """One line per expert: metric vs lambda (mean over seeds)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=120)
    plotted = False
    for expert, cells in sorted(per_expert.items()):
        by_lambda: dict[float, list[float]] = defaultdict(list)
        for c in cells:
            v = c[metric]
            if v is not None:
                by_lambda[c["lambda"]].append(float(v))
        if not by_lambda:
            continue
        xs = sorted(by_lambda)
        ys = [sum(by_lambda[x]) / len(by_lambda[x]) for x in xs]
        ax.plot(xs, ys, marker="o", label=expert)
        plotted = True
    if not plotted:
        logger.error("[aggregate_ablation] no data for metric %s", metric)
        raise ValueError(f"no data to plot for metric {metric}")
    ax.set_xlabel("lambda_lat")
    ax.set_ylabel(ylabel)
    arrow = "lower better" if lower_better else "higher better"
    ax.set_title(f"{ylabel} vs lambda_lat  ({arrow})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("[aggregate_ablation] saved -> %s", out_path)


def aggregate(results_root: Path) -> dict:
    reports = sorted(results_root.glob("*/report.json"))
    if not reports:
        logger.error("[aggregate_ablation] no report.json under %s",
                     results_root)
        raise FileNotFoundError(f"no report.json under {results_root}")
    cells = [_read_cell(p) for p in reports]
    per_expert: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        per_expert[c["expert"]].append(c)

    panels: dict[str, list[dict]] = {}
    lines: list[str] = []
    sep = "=" * 70
    lines.append(sep)
    lines.append("LATENT-SHAPE PENALTY ABLATION -- per-expert self-comparison")
    lines.append("(each expert vs its OWN lambda=0 baseline; no cross-expert "
                 "winner)")
    lines.append(sep)

    for expert in sorted(per_expert):
        cells_e = per_expert[expert]
        # baseline = mean over seeds at lambda == 0
        base_cells = [c for c in cells_e if c["lambda"] == 0.0]
        if not base_cells:
            logger.error("[aggregate_ablation] expert %s has no lambda=0 "
                         "baseline cell", expert)
            raise ValueError(f"{expert}: missing lambda=0 baseline")
        base = {
            "ks":      sum(c["ks"] for c in base_cells) / len(base_cells),
            "nll":     sum(c["nll"] for c in base_cells) / len(base_cells),
            "fwd_rel": (sum(c["fwd_rel"] for c in base_cells) / len(base_cells)
                        if all(c["fwd_rel"] is not None for c in base_cells)
                        else None),
        }
        # group non-baseline cells by lambda, average seeds
        by_lambda: dict[float, list[dict]] = defaultdict(list)
        for c in cells_e:
            by_lambda[c["lambda"]].append(c)
        rows = []
        for lam in sorted(by_lambda):
            grp = by_lambda[lam]
            cell = {
                "lambda":    lam,
                "ks":        sum(c["ks"] for c in grp) / len(grp),
                "nll":       sum(c["nll"] for c in grp) / len(grp),
                "fwd_rel":   (sum(c["fwd_rel"] for c in grp) / len(grp)
                              if all(c["fwd_rel"] is not None for c in grp)
                              else None),
                "fzdy":      (sum(c["fzdy"] for c in grp) / len(grp)
                              if all(c["fzdy"] is not None for c in grp)
                              else None),
                "cond_pass": all(c["cond_pass"] for c in grp),
                "n_seeds":   len(grp),
            }
            cell["verdict"] = ("baseline" if lam == 0.0
                               else _verdict(cell, base))
            rows.append(cell)
        panels[expert] = rows

        # best lambda = an 'improves' cell with the lowest KS (else None)
        improving = [r for r in rows if r["verdict"] == "improves"]
        best = min(improving, key=lambda r: r["ks"]) if improving else None

        lines.append("")
        lines.append(f"[{expert}]   baseline KS={base['ks']:.4f}  "
                     f"NLL={base['nll']:.2f}")
        lines.append(f"  {'lambda':>7} {'KS':>8} {'NLL':>9} {'fwd_rel':>8} "
                     f"{'FZDY':>10} {'cond':>5} {'verdict':>10}")
        for r in rows:
            fwd = f"{r['fwd_rel']:.3f}" if r["fwd_rel"] is not None else "  n/a"
            fz  = f"{r['fzdy']:.2e}" if r["fzdy"] is not None else "      n/a"
            cp  = "ok" if r["cond_pass"] else "DEAD"
            lines.append(f"  {r['lambda']:>7.2f} {r['ks']:>8.4f} "
                         f"{r['nll']:>9.2f} {fwd:>8} {fz:>10} {cp:>5} "
                         f"{r['verdict']:>10}")
        if best is not None:
            lines.append(f"  -> best lambda for {expert}: {best['lambda']:.2f} "
                         f"(KS {base['ks']:.4f} -> {best['ks']:.4f})")
        else:
            lines.append(f"  -> no lambda improves over baseline for {expert} "
                         f"(keep lambda=0)")

    lines.append("")
    lines.append(sep)
    report_txt = "\n".join(lines)
    print(report_txt)

    plots_dir = results_root / "ablation_plots"
    _plot_sweep(per_expert, "ks",      "latent KS",
                plots_dir / "ks_vs_lambda.png",      lower_better=True)
    _plot_sweep(per_expert, "fzdy",    "FZDY sensitivity",
                plots_dir / "fzdy_vs_lambda.png",    lower_better=False)
    _plot_sweep(per_expert, "nll",     "final test NLL",
                plots_dir / "nll_vs_lambda.png",     lower_better=True)
    _plot_sweep(per_expert, "fwd_rel", "fwd_rel",
                plots_dir / "fwd_rel_vs_lambda.png", lower_better=True)

    out = {"panels": panels, "thresholds": {
        "ks_rel_improve": KS_REL_IMPROVE, "nll_rel_tol": NLL_REL_TOL,
        "fwd_rel_tol": FWD_REL_TOL}}
    (results_root / "ablation_summary.json").write_text(
        json.dumps(out, indent=2))
    (results_root / "ablation_summary.txt").write_text(report_txt)
    logger.info("[aggregate_ablation] wrote ablation_summary.{json,txt} + "
                "4 sweep plots under %s", plots_dir)
    return out


def _parse_args():
    p = argparse.ArgumentParser(
        description="Per-expert latent-shape penalty ablation aggregator")
    p.add_argument("--results-root",
                   default="./CSMF2/experiments/step_1_1_1/results",
                   help="dir containing <run_tag>/report.json cells")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    a = _parse_args()
    aggregate(Path(a.results_root))
