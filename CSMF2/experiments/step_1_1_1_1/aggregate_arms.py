# =============================================================================
# STEP-1_1_1_1 v0.1 -- experiments.step_1_1_1_1.aggregate_arms
# Purpose: collect the three arm reports (random_map / is_only / is_map) under
#          a results root and emit the comparison table + the decision-rule
#          verdict that separates SELECTION benefit from OPTIMIZATION benefit
#          from COST. Reads each run dir's report.json["arm"]/["metrics"].
# CONVENTION: missing arms / keys -> logger.error + raise. No fallback / mock.
# Changelog (NEW in v0.1):
#   * Introduced. Point-by-point table over arms + IS-vs-1.2 decision rule.
# Update summary:
#   v0.1 compares fwd_rel / PSNR / residual / runtime across the three arms
#   and prints the verdict: keep IS+MAP, fall back to best-of-K, or skip to
#   Stage 1.2. Thresholds are relative (REL_GAIN) and reported alongside.
# =============================================================================
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_1_1_1"

REL_GAIN = 0.05      # >=5% relative change counts as a real difference

_ARMS = ("random_map", "is_only", "is_map")


def _load_arms(results_root: Path) -> dict:
    arms: dict[str, dict] = {}
    if not results_root.exists():
        logger.error("[aggregate_arms] %s does not exist", results_root)
        raise FileNotFoundError(results_root)
    for run_dir in sorted(results_root.iterdir()):
        rpt = run_dir / "report.json"
        if not rpt.exists():
            continue
        try:
            data = json.loads(rpt.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("[aggregate_arms] cannot read %s: %s", rpt, exc)
            raise
        arm = data.get("arm")
        if arm in _ARMS:
            arms[arm] = data            # last one wins if duplicated
    missing = [a for a in _ARMS if a not in arms]
    if missing:
        logger.error("[aggregate_arms] missing arm reports: %s "
                     "(run all three first)", missing)
        raise ValueError(f"missing arm reports: {missing}")
    return arms


def _rel_better(new: float, ref: float, *, lower_better: bool) -> bool:
    if ref == 0:
        return False
    delta = (ref - new) / abs(ref) if lower_better else (new - ref) / abs(ref)
    return delta >= REL_GAIN


def aggregate(results_root: Path) -> dict:
    arms = _load_arms(results_root)
    m = {a: arms[a]["metrics"] for a in _ARMS}

    lines, sep = [], "=" * 78
    lines.append(sep)
    lines.append("MAP-ABL three-arm comparison (selection vs optimization vs cost)")
    lines.append(sep)
    hdr = (f"  {'arm':<12} {'fwd_rel_after':>14} {'PSNR_after':>11} "
           f"{'residual_after':>15} {'runtime_s':>10}")
    lines.append(hdr)
    for a in _ARMS:
        lines.append(f"  {a:<12} {m[a]['fwd_rel_after']:>14.4f} "
                     f"{m[a]['psnr_after']:>11.2f} "
                     f"{m[a]['residual_after']:>15.2f} "
                     f"{arms[a]['runtime_s']:>10.1f}")
    # IS-only "after" == its initial selection (no MAP); random_map "before"
    # is the random sample. Use fwd_rel_after as the headline reconstruction.
    rnd = m["random_map"]["fwd_rel_after"]
    iso = m["is_only"]["fwd_rel_after"]
    ism = m["is_map"]["fwd_rel_after"]
    rnd_init = m["random_map"]["fwd_rel_before"]   # = random sample, no opt

    lines.append("")
    lines.append("Verdict (fwd_rel, lower better):")
    sel = _rel_better(iso, rnd_init, lower_better=True)
    optr = _rel_better(rnd, rnd_init, lower_better=True)
    is_beats_iso = _rel_better(ism, iso, lower_better=True)
    is_beats_rnd = _rel_better(ism, rnd, lower_better=True)

    lines.append(f"  selection helps      (is_only < random sample): {sel}")
    lines.append(f"  optimization helps   (random_map < random sample): {optr}")
    lines.append(f"  is_map < is_only:    {is_beats_iso}")
    lines.append(f"  is_map < random_map: {is_beats_rnd}")

    if is_beats_iso and is_beats_rnd:
        verdict = ("KEEP IS+MAP -- better init genuinely helps MAP beyond "
                   "both selection and random-init optimization.")
    elif not is_beats_iso and sel:
        verdict = ("MAP adds little over selection -- best-of-K sampling may "
                   "be enough; reconsider MAP's cost.")
    elif not is_beats_rnd:
        verdict = ("IS init not worth its cost -- is_map ~ random_map; "
                   "proceed to Stage 1.2 (mixture skeleton).")
    else:
        verdict = ("Mixed signal -- inspect per-image spread; consider "
                   "multi_start escalation (additional).")
    lines.append("")
    lines.append(f"DECISION: {verdict}")
    lines.append(sep)

    text = "\n".join(lines)
    print(text)
    out = {"arms": {a: arms[a]["metrics"] for a in _ARMS},
           "runtime_s": {a: arms[a]["runtime_s"] for a in _ARMS},
           "rel_gain_threshold": REL_GAIN, "verdict": verdict}
    (results_root / "arms_summary.json").write_text(json.dumps(out, indent=2))
    (results_root / "arms_summary.txt").write_text(text)
    logger.info("[aggregate_arms] wrote arms_summary.{json,txt}")
    return out


def _parse_args():
    p = argparse.ArgumentParser(description="MAP-ABL three-arm aggregator")
    p.add_argument("--results-root",
                   default="./CSMF2/experiments/step_1_1_1_1/results")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    a = _parse_args()
    aggregate(Path(a.results_root))
