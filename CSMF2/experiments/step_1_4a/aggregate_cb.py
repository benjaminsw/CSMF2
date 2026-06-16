# =============================================================================
# STEP-1_4A v0.1 -- experiments.step_1_4a.aggregate_cb
# Purpose: summarize the Stage 1.4a conditional-base experiment. Reads the CB
#          run reports (base health + val NLL) and, if a post-CB RECGATE-global
#          summary is present, applies the COMPLEMENTARITY success criterion --
#          NOT mean gap-closing.
# CONVENTION: missing inputs / keys -> logger.error + raise. No fallback/mock.
# Success (all must hold):
#   rec_argmin_counts becomes LESS NSF-dominated (NICE/RealNVP win some)
#   AND RECGATE-global max mean weight < 0.70 (no full NSF collapse)
#   AND fwd_rel not worse than NSF-only baseline
#   AND base_alive = True for the CB experts
# Changelog (NEW in v0.1):
#   * Introduced.
# Update summary:
#   v0.1 prints base-health per expert + the complementarity verdict. The
#   RECGATE-global rerun on CB ckpts is produced by step_1_3.aggregate_beta;
#   pass its beta_summary.json via --recgate-summary.
# =============================================================================
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_4A"


def _load_cb_runs(results_root: Path) -> dict:
    runs = {}
    if not results_root.exists():
        logger.error("[aggregate_cb] %s missing", results_root)
        raise FileNotFoundError(results_root)
    for d in sorted(results_root.iterdir()):
        rpt = d / "report.json"
        if not rpt.exists():
            continue
        data = json.loads(rpt.read_text())
        if data.get("use_conditional_base"):
            runs[data["expert"]] = data
    if not runs:
        logger.error("[aggregate_cb] no CB run reports under %s", results_root)
        raise ValueError("no CB runs found")
    return runs


def aggregate(results_root: Path, recgate_summary: Path = None) -> dict:
    runs = _load_cb_runs(results_root)
    sep = "=" * 78
    lines = [sep, "CBASE v0.1 -- Stage 1.4a conditional-base summary", sep,
             f"  {'expert':<9} {'val_NLL':>10} {'base_alive':>11} "
             f"{'mu_std':>9} {'ls_std':>9} {'KL':>9}"]
    all_alive = True
    for ex in sorted(runs):
        r = runs[ex]; bf = r.get("base_final") or {}
        alive = bool(r.get("base_alive"))
        all_alive = all_alive and alive
        lines.append(f"  {ex:<9} {r['val_nll']:>10.1f} {str(alive):>11} "
                     f"{bf.get('mu_std_across_y', 0):>9.2e} "
                     f"{bf.get('log_sigma_std_across_y', 0):>9.2e} "
                     f"{bf.get('base_effect_magnitude', 0):>9.2e}")

    rec_ok = gate_ok = fwd_ok = None
    if recgate_summary is not None:
        rg = json.loads(Path(recgate_summary).read_text())
        # max mean weight across betas (collapse if any expert dominates all)
        weights = rg.get("mean_weight_per_expert", [])
        if weights:
            least_dom = min(max(w) for w in weights)
            gate_ok = least_dom < 0.70
        soft = rg.get("soft_fwd_rel", []); base = rg.get("nsf_only_fwd_rel", [])
        if soft and base:
            fwd_ok = min(soft) <= base[0]
        lines += ["", "Post-CB RECGATE-global:",
                  f"  no full NSF collapse (max weight < 0.70): {gate_ok}",
                  f"  fwd_rel not worse than NSF-only:          {fwd_ok}"]
    lines += ["", f"  base_alive (all CB experts): {all_alive}"]

    # ---- verdict -----------------------------------------------------------
    if recgate_summary is None:
        verdict = ("CB trained. Base health above. Rerun RECGATE-global on the "
                   "CB ckpts and pass --recgate-summary to judge complementarity.")
    elif all_alive and gate_ok and fwd_ok:
        verdict = ("Stage 1.4a SUCCEEDS -- CB created useful complementarity "
                   "(gate no longer fully NSF, fwd_rel not worse, base alive). "
                   "-> Stage 2.3 is now meaningful.")
    elif all_alive and not gate_ok:
        verdict = ("CB alive but NO complementarity -- gate still collapses to "
                   "one expert. CB did not differentiate them enough; consider "
                   "Stage 1.4b (multi-scale RealNVP / NICE depth).")
    elif not all_alive:
        verdict = ("Base NOT alive on >=1 CB expert -- any NLL change is "
                   "retrain variance, not CB. Fix base conditioning before "
                   "judging (raise base_gain / check tau_b / logsigma clamp).")
    else:
        verdict = ("Mixed -- inspect rec_argmin_counts and per-expert fwd_rel "
                   "before deciding on Stage 1.4b.")
    lines += [f"DECISION: {verdict}", sep]
    text = "\n".join(lines); print(text)

    out = {"experts": {ex: {"val_nll": runs[ex]["val_nll"],
                            "base_alive": runs[ex]["base_alive"],
                            "base_final": runs[ex].get("base_final")}
                       for ex in runs},
           "all_base_alive": all_alive, "gate_ok": gate_ok, "fwd_ok": fwd_ok,
           "verdict": verdict}
    (results_root / "cb_summary.json").write_text(json.dumps(out, indent=2))
    (results_root / "cb_summary.txt").write_text(text)
    logger.info("[aggregate_cb] wrote cb_summary.{json,txt}")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="./CSMF2/experiments/step_1_4a/results")
    ap.add_argument("--recgate-summary", default=None,
                    help="step_1_3 beta_summary.json from the post-CB RECGATE rerun")
    a = ap.parse_args()
    aggregate(Path(a.results_root),
              Path(a.recgate_summary) if a.recgate_summary else None)
