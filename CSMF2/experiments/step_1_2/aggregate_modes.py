# =============================================================================
# STEP-1_2 v0.1 -- experiments.step_1_2.aggregate_modes
# Purpose: collect the single / uniform / learned reports under a results root
#          and emit the MIX-SKEL summary table + the skeleton pass condition.
#          Collapse (Neff->1 under pure NLL) is reported as a FAILURE OF THE
#          PURE-NLL GATE, not of the mixture code.
# CONVENTION: missing modes / keys -> logger.error + raise. No fallback / mock.
# Pass condition (mechanical, NOT gate-diversity):
#   mixture logp finite (guaranteed -- runs raise otherwise)
#   weights sum to 1   (guaranteed -- runs raise otherwise)
#   learned_nll <= uniform_nll  AND  learned_nll ~ single_nll (within ~1-2%)
#   collapse measured clearly (Neff trajectory present)
# Changelog (NEW in v0.1):
#   * Introduced. Summary table + pass/collapse verdict across the 3 modes.
# Update summary:
#   v0.1 prints  mode | NLL | Neff | entropy | <per-expert weights>  and the
#   verdict: skeleton mechanics OK + pure-NLL gate collapse confirmed ->
#   proceed to Stage 1.3 (reconstruction-aware gate).
# =============================================================================
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_2"

_MODES = ("single", "uniform", "learned")
_APPROX = 0.02     # learned ~ single tolerance (2%)


def _load_modes(results_root: Path) -> dict:
    modes: dict[str, dict] = {}
    if not results_root.exists():
        logger.error("[aggregate_modes] %s does not exist", results_root)
        raise FileNotFoundError(results_root)
    for run_dir in sorted(results_root.iterdir()):
        rpt = run_dir / "report.json"
        if not rpt.exists():
            continue
        try:
            data = json.loads(rpt.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("[aggregate_modes] cannot read %s: %s", rpt, exc)
            raise
        m = data.get("mode")
        if m in _MODES:
            modes[m] = data
    missing = [m for m in _MODES if m not in modes]
    if missing:
        logger.error("[aggregate_modes] missing mode reports: %s "
                     "(run all three first)", missing)
        raise ValueError(f"missing mode reports: {missing}")
    return modes


def aggregate(results_root: Path) -> dict:
    modes = _load_modes(results_root)
    names = modes["uniform"]["expert_names"]
    lines, sep = [], "=" * 78
    lines.append(sep)
    lines.append("MIX-SKEL v0.2 -- mode comparison (pure NLL, frozen experts)")
    lines.append(sep)
    head = (f"  {'mode':<9} {'NLL':>10} {'Neff':>7} {'entropy':>8}  "
            + "  ".join(f"{n+'_w':>9}" for n in names))
    lines.append(head)
    for m in _MODES:
        r = modes[m]
        nll = r["mode_nll"]
        g = r.get("gate")
        if g is None:                                  # single: no gate
            neff = ent = None
            ws = ["—"] * len(names)
            lines.append(f"  {m:<9} {nll:>10.2f} {'—':>7} {'—':>8}  "
                         + "  ".join(f"{w:>9}" for w in ws))
        else:
            neff = g["Neff_mean"]; ent = g["gate_entropy"]
            ws = g["mean_weight_per_expert"]
            lines.append(f"  {m:<9} {nll:>10.2f} {neff:>7.3f} {ent:>8.3f}  "
                         + "  ".join(f"{w:>9.3f}" for w in ws))

    # ---- pass condition ----------------------------------------------------
    single_nll = modes["single"]["mode_nll"]
    uniform_nll = modes["uniform"]["mode_nll"]
    learned_nll = modes["learned"]["mode_nll"]
    learned_neff = modes["learned"]["gate"]["Neff_mean"]
    beats_uniform = learned_nll <= uniform_nll
    approx_single = abs(learned_nll - single_nll) <= _APPROX * abs(single_nll)
    has_curve = bool(modes["learned"].get("curves", {}).get("Neff"))

    skeleton_ok = beats_uniform and approx_single and has_curve
    collapsed = learned_neff < 1.5

    lines.append("")
    lines.append("Pass condition (mechanical, not gate-diversity):")
    lines.append(f"  learned_nll <= uniform_nll      : {beats_uniform} "
                 f"({learned_nll:.2f} <= {uniform_nll:.2f})")
    lines.append(f"  learned_nll ~ single_best (<=2%) : {approx_single} "
                 f"({learned_nll:.2f} vs {single_nll:.2f})")
    lines.append(f"  Neff trajectory recorded         : {has_curve}")
    lines.append(f"  -> SKELETON MECHANICS: "
                 f"{'OK' if skeleton_ok else 'CHECK'}")
    lines.append("")
    if collapsed:
        lines.append(f"  pure_nll_gate_collapse: FAIL (expected) -- "
                     f"Neff={learned_neff:.3f} -> gate collapsed onto the "
                     f"best-likelihood expert. This is the intended trigger "
                     f"for Stage 1.3 (reconstruction-aware gate).")
    else:
        lines.append(f"  pure_nll_gate retained diversity (Neff="
                     f"{learned_neff:.3f}) -- unexpected under pure NLL; "
                     f"inspect before proceeding.")
    lines.append(sep)

    text = "\n".join(lines)
    print(text)
    out = {"single_nll": single_nll, "uniform_nll": uniform_nll,
           "learned_nll": learned_nll, "learned_Neff": learned_neff,
           "skeleton_ok": bool(skeleton_ok), "collapsed": bool(collapsed)}
    (results_root / "modes_summary.json").write_text(json.dumps(out, indent=2))
    (results_root / "modes_summary.txt").write_text(text)
    logger.info("[aggregate_modes] wrote modes_summary.{json,txt}")
    return out


def _parse_args():
    p = argparse.ArgumentParser(description="MIX-SKEL three-mode aggregator")
    p.add_argument("--results-root",
                   default="./CSMF2/experiments/step_1_2/results")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    a = _parse_args()
    aggregate(Path(a.results_root))
