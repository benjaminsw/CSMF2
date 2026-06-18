# =============================================================================
# STEP-1_4A v0.2 -- experiments.step_1_4a.summarize_depth  (RNVP-DEPTH v0.2)
# Purpose: collate the RealNVP depth sweep -- per-depth val_NLL (CB reports),
#          soft/hard fwd_rel + NSF-only baseline (RECGATE summaries), RealNVP
#          rec_argmin wins + tier (1.3a breakdowns) -- into one table + a
#          depth-vs-metric plot + a dual-label verdict.
# CONVENTION: missing inputs / length mismatch -> logger.error + raise.
#             No fallback / mock / dummy / pass.
# Decision (v0.2 -- fwd is the gate; rec_argmin WIN COUNT is context, NOT a
#   success criterion). Per depth (SAME depth), with PARITY_EPS=1e-3:
#     strict_i   = min(soft,hard) <  nsf - eps     (mixture reconstructs BETTER)
#     research_i = |min(soft,hard) - nsf| <= eps  AND tier not FLAT (parity+spec)
#   strict_success  = EXISTS depth strict_i
#   research_signal = EXISTS depth research_i
#   else density-only (NLL improved) or saturated.
# Changelog (v0.1 -> v0.2):
#   * Verdict fix: dropped the `wins_rose` shortcut (a rec_argmin win-count rise
#     no longer triggers success -- c16 rose to 356 wins while fwd_rel got
#     WORSE and tier stayed FLAT). Strict success now requires fwd to BEAT
#     NSF-only; research signal = fwd parity + non-FLAT tier. Win count is a
#     table column only. Added columns: dfwd_vs_nsf, tier, strict, research.
#     Mirrors the RECGATE-AGG v0.3 dual-label fix.
# Changelog (NEW in v0.1):
#   * Introduced. table + p_depth_vs_metric.png + decision.
# Update summary:
#   v0.2 prevents the same false-positive class fixed in aggregate_beta: a
#   descriptive count (wins) can no longer masquerade as success; only the
#   mixture actually reconstructing better than NSF-only counts.
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
__version__ = "0.2"
__abbr__ = "STEP-1_4A"

DEPTHS = [6, 8, 12, 16]


def _load_json(p):
    p = Path(p)
    if not p.exists():
        logger.error("[summarize_depth] missing %s", p)
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def _realnvp_idx(names):
    if "realnvp" not in names:
        logger.error("[summarize_depth] 'realnvp' not in expert_names %s", names)
        raise ValueError("realnvp not found in expert_names")
    return names.index("realnvp")


def summarize(depth_root: Path, recgate_summaries, breakdowns, out_root: Path):
    if not (len(recgate_summaries) == len(breakdowns) == len(DEPTHS)):
        logger.error("[summarize_depth] need %d recgate + %d breakdown files, "
                     "got %d / %d", len(DEPTHS), len(DEPTHS),
                     len(recgate_summaries), len(breakdowns))
        raise ValueError("one recgate summary and one breakdown per depth")
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)

    val_nll, soft, hard, base, rnvp_wins, top2, tiers = [], [], [], [], [], [], []
    for d, rg_path, bd_path in zip(DEPTHS, recgate_summaries, breakdowns):
        rg = _load_json(rg_path)
        bd = _load_json(bd_path)
        # best (lowest) soft/hard_fwd across beta for this depth + the NSF-only base
        soft.append(min(rg["soft_fwd_rel"]))
        hard.append(min(rg["hard_fwd_rel"]))
        base.append(rg["nsf_only_fwd_rel"][0])
        # RealNVP rec_argmin wins + top2_share + tier from the breakdown
        names = bd["experts"]; ri = _realnvp_idx(names)
        rnvp_wins.append(bd["win_totals"][names[ri]])
        top2.append(bd["top2_share"])
        tiers.append(bd.get("tier", "?"))
        # val_NLL: locate the depth's CB run report under depth_root
        nll = None
        for run_dir in sorted(Path(depth_root).iterdir()):
            rpt = run_dir / "report.json"
            if not rpt.exists():
                continue
            data = json.loads(rpt.read_text())
            if (data.get("expert") == "realnvp"
                    and data.get("cfg", {}).get("realnvp_n_couplings") == d):
                nll = data["val_nll"]; break
        if nll is None:
            logger.error("[summarize_depth] no realnvp report for depth %d "
                         "under %s", d, depth_root)
            raise FileNotFoundError(f"realnvp depth={d} report")
        val_nll.append(nll)

    # ---- per-depth flags (SAME-depth; win count is context, NOT a gate) ---
    PARITY_EPS = 1e-3
    SPEC_MIN_WINS = 150     # a non-FLAT tier on fewer wins is a small-N artifact
    per_depth = []
    for i, d in enumerate(DEPTHS):
        best_fwd = min(soft[i], hard[i])
        dfwd = best_fwd - base[i]                 # <0 means mixture beats NSF
        # non-FLAT tier only counts as specialization if backed by enough wins
        not_flat = (tiers[i] not in ("FLAT", "NONE", "?")
                    and rnvp_wins[i] >= SPEC_MIN_WINS)
        strict_i = dfwd < -PARITY_EPS             # reconstruction genuinely better
        parity_i = abs(dfwd) <= PARITY_EPS
        research_i = parity_i and not_flat        # ties NSF but specializes (real)
        per_depth.append({"depth": d, "dfwd": dfwd, "tier": tiers[i],
                          "wins": rnvp_wins[i], "strict": strict_i,
                          "research": research_i})

    # ---- table -------------------------------------------------------------
    sep = "=" * 78
    lines = [sep, "RNVP-DEPTH v0.2 -- RealNVP coupling-depth sweep (hidden fixed)",
             sep, f"  {'couplings':>9} {'val_NLL':>10} {'soft_fwd':>9} "
             f"{'nsf_fwd':>8} {'dfwd':>9} {'rnvp_wins':>10} {'tier':>16} "
             f"{'strict':>7} {'research':>9}"]
    for i, d in enumerate(DEPTHS):
        pd = per_depth[i]
        lines.append(f"  {d:>9} {val_nll[i]:>10.1f} {soft[i]:>9.4f} "
                     f"{base[i]:>8.4f} {pd['dfwd']:>+9.4f} {rnvp_wins[i]:>10} "
                     f"{tiers[i]:>16} {str(pd['strict']):>7} "
                     f"{str(pd['research']):>9}")

    # ---- decision (fwd is the gate; win count is NOT) ---------------------
    strict_depths = [pd["depth"] for pd in per_depth if pd["strict"]]
    research_depths = [pd["depth"] for pd in per_depth if pd["research"]]
    nll_drop = val_nll[0] - min(val_nll)
    nll_drop_meaningful = nll_drop > 0.005 * abs(val_nll[0])
    if strict_depths:
        verdict = (f"STRICT SUCCESS -- depth(s) {strict_depths}: the mixture "
                   f"reconstructs BETTER than NSF-only (dfwd < -{PARITY_EPS}). "
                   f"Adopt the best such depth; seed-1 confirm before locking.")
    elif research_depths:
        verdict = (f"RESEARCH SIGNAL (not strict) -- depth(s) {research_depths}: "
                   f"fwd at PARITY with NSF-only AND rec_argmin tier not FLAT "
                   f"(real specialization). A legit Stage-2.3 trigger, not a win.")
    elif nll_drop_meaningful:
        verdict = (f"DENSITY ONLY -- val_NLL improved ({val_nll[0]:.0f} -> "
                   f"{min(val_nll):.0f}) but NO depth reconstructs better than "
                   f"NSF-only and tiers stay FLAT. Better density, NOT "
                   f"complementarity. rec_argmin wins ({rnvp_wins[0]} -> "
                   f"{max(rnvp_wins)}) are context, not success. -> image RealNVP (3d).")
    else:
        verdict = ("SATURATED -- neither reconstruction nor density improved "
                   "meaningfully with depth. Flat vector RealNVP is the wall "
                   "-> image RealNVP (3d).")
    lines += ["", f"DECISION: {verdict}", sep]
    text = "\n".join(lines); print(text)

    # ---- depth-vs-metric plot ---------------------------------------------
    fig, ax1 = plt.subplots(figsize=(7.5, 4.4), dpi=120)
    ax1.set_xlabel("coupling layers")
    ax1.set_ylabel("val NLL", color="#185FA5")
    ax1.plot(DEPTHS, val_nll, marker="o", color="#185FA5", label="val NLL")
    ax1.tick_params(axis="y", labelcolor="#185FA5")
    ax2 = ax1.twinx()
    ax2.set_ylabel("RealNVP rec_argmin wins", color="#0F6E56")
    ax2.plot(DEPTHS, rnvp_wins, marker="s", color="#0F6E56",
             label="rec_argmin wins")
    ax2.tick_params(axis="y", labelcolor="#0F6E56")
    ax1.set_xticks(DEPTHS)
    ax1.set_title("RealNVP depth sweep: density vs competitiveness")
    fig.tight_layout(); fig.savefig(out_root / "p_depth_vs_metric.png",
                                    bbox_inches="tight")
    plt.close(fig)

    out = {"depths": DEPTHS, "val_nll": val_nll, "soft_fwd_rel": soft,
           "hard_fwd_rel": hard, "nsf_only_fwd_rel": base,
           "realnvp_rec_argmin_wins": rnvp_wins, "realnvp_top2_share": top2,
           "tiers": tiers, "per_depth": per_depth,
           "strict_depths": strict_depths, "research_depths": research_depths,
           "strict_success": bool(strict_depths),
           "research_signal": bool(research_depths),
           "verdict": verdict}
    (out_root / "depth_summary.json").write_text(json.dumps(out, indent=2))
    (out_root / "depth_summary.txt").write_text(text)
    logger.info("[summarize_depth] wrote depth_summary.{json,txt} + plot")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-root", default="./CSMF2/experiments/step_1_4a/results_depth")
    ap.add_argument("--recgate-summaries", nargs="+", required=True,
                    help="one beta_summary.json per depth, in 6/8/12/16 order")
    ap.add_argument("--breakdowns", nargs="+", required=True,
                    help="one rec_argmin_breakdown.json per depth, in 6/8/12/16 order")
    ap.add_argument("--out-root", default="./CSMF2/experiments/step_1_4a/results_depth")
    a = ap.parse_args()
    summarize(Path(a.depth_root), a.recgate_summaries, a.breakdowns,
              Path(a.out_root))
