# =============================================================================
# STEP-1_1 v0.4 -- experiments.step_1_1.aggregate
# Purpose: scan results/ dir, load each report.json's "summary" block, produce:
#          - _aggregate.csv (tidy rows for plotting)
#          - _aggregate.md  (markdown table for write-up)
#          - stdout table with ANSI colour + step 1.2 unlock verdict.
# Usage:   python -m CSMF2.experiments.step_1_1.aggregate
# Changelog (NEW in v0.4):
#   * Introduced.
# =============================================================================
from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

logger = logging.getLogger(__name__)
__version__ = "0.4"
__abbr__ = "STEP-1_1"

_USE_COLOR = os.environ.get("NO_COLOR", "") == ""
_G = "\033[32m" if _USE_COLOR else ""
_R = "\033[31m" if _USE_COLOR else ""
_Y = "\033[33m" if _USE_COLOR else ""
_B = "\033[1m"  if _USE_COLOR else ""
_N = "\033[0m"  if _USE_COLOR else ""


def _load_summaries(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    if not results_root.exists():
        logger.error("[aggregate] %s does not exist", results_root)
        raise FileNotFoundError(results_root)
    for run_dir in sorted(results_root.iterdir()):
        rpt = run_dir / "report.json"
        if not rpt.exists():
            continue
        try:
            data = json.loads(rpt.read_text())
        except Exception:
            logger.error("[aggregate] could not read %s", rpt)
            continue
        s = data.get("summary")
        if s is None:
            continue
        s = dict(s)            # shallow copy; mutate ok
        s["run_tag"] = run_dir.name
        rows.append(s)
    return rows


def _group_key(r: dict) -> tuple:
    return (r.get("expert"), r.get("scale"), r.get("noise"))


def main(results_root: Path = None) -> int:
    results_root = results_root or Path("./CSMF2/experiments/step_1_1/results")
    rows = _load_summaries(results_root)
    if not rows:
        print(f"{_Y}no runs found in {results_root}{_N}")
        return 1

    # tidy CSV ----------------------------------------------------------------
    csv_path = results_root / "_aggregate.csv"
    fields = ["run_tag", "expert", "scale", "noise", "seed",
              "epochs_completed", "exit_criteria_met",
              "train_first", "train_last", "test_first", "test_last",
              "nll_improved", "cycle_max_last", "fwd_rel_last"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "run_tag": r["run_tag"],
                "expert": r.get("expert"),
                "scale":  r.get("scale"),
                "noise":  r.get("noise"),
                "seed":   r.get("seed"),
                "epochs_completed": r.get("epochs_completed"),
                "exit_criteria_met": int(bool(r.get("exit_criteria_met"))),
                "train_first": r["nll"]["train_first"],
                "train_last":  r["nll"]["train_last"],
                "test_first":  r["nll"]["test_first"],
                "test_last":   r["nll"]["test_last"],
                "nll_improved": int(bool(r["nll"]["improved"])),
                "cycle_max_last": r.get("cycle_max_last"),
                "fwd_rel_last":   r.get("fwd_rel_last"),
            })

    # grouped by (expert, scale, noise) --------------------------------------
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[_group_key(r)].append(r)

    # ---- markdown table -----------------------------------------------------
    md = ["| expert | scale | noise | pass/total | test NLL (last, mean ± std) | cycle_max |",
          "|---|---|---|---|---|---|"]
    print(f"\n{_B}STEP-1_1 SWEEP AGGREGATE ({len(rows)} runs){_N}")
    print("=" * 78)
    print(f"{'group':<22} {'pass/total':>12} {'NLL (mean±std)':>20} {'cycle_max':>14}")
    print("-" * 78)

    # expert -> set of (scale, noise) combos with ALL seeds passing
    per_expert_ok: dict[str, int] = defaultdict(int)
    per_expert_total: dict[str, int] = defaultdict(int)

    for k in sorted(groups.keys(), key=lambda t: (str(t[0]), t[1], t[2])):
        runs = groups[k]
        n_pass = sum(1 for r in runs if r.get("exit_criteria_met"))
        n_tot = len(runs)
        nlls = [r["nll"]["test_last"] for r in runs
                if r.get("nll", {}).get("test_last") is not None]
        cyc  = [r.get("cycle_max_last") for r in runs
                if r.get("cycle_max_last") is not None]
        nll_str = (f"{mean(nlls):.1f} ± {stdev(nlls):.1f}"
                   if len(nlls) >= 2 else
                   (f"{nlls[0]:.1f}" if nlls else "—"))
        cyc_str = (f"{max(cyc):.1e}" if cyc else "—")
        col = _G if n_pass == n_tot else (_Y if n_pass > 0 else _R)
        group_tag = f"{k[0]:<7} s{k[1]} n{k[2]}"
        print(f"{group_tag:<22} {col}{n_pass:>6}/{n_tot:<5}{_N} "
              f"{nll_str:>20} {cyc_str:>14}")
        md.append(f"| {k[0]} | {k[1]} | {k[2]} | {n_pass}/{n_tot} | "
                  f"{nll_str} | {cyc_str} |")

        per_expert_total[k[0]] += n_tot
        per_expert_ok[k[0]]    += n_pass

    print("-" * 78)
    experts_competent: list[str] = []
    for expert, tot in per_expert_total.items():
        ok = per_expert_ok[expert]
        frac = ok / max(tot, 1)
        mark = _G + "competent" + _N if frac >= 0.8 else _R + "not competent" + _N
        print(f"  {expert:<7}  {ok}/{tot} runs pass  ({frac:.0%})  -> {mark}")
        if frac >= 0.8:
            experts_competent.append(expert)

    print()
    n_comp = len(experts_competent)
    if n_comp >= 2:
        print(f"{_G}{_B}STEP 1.2 UNLOCK: YES{_N}  "
              f"(≥2 competent experts: {', '.join(experts_competent)})")
        unlock = 0
    else:
        print(f"{_R}{_B}STEP 1.2 UNLOCK: NO{_N}  "
              f"(only {n_comp} competent: {', '.join(experts_competent) or '—'})")
        unlock = 2

    # ---- save markdown ------------------------------------------------------
    md_path = results_root / "_aggregate.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"\nwrote {csv_path}")
    print(f"wrote {md_path}")
    return unlock


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-root",
                   default="./CSMF2/experiments/step_1_1/results")
    a = p.parse_args()
    sys.exit(main(Path(a.results_root)))
