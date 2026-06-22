# =============================================================================
# STEP-1_4B v0.1 -- experiments.step_1_4b.aggregate_consist  (RNVP-CONSIST v0.1)
# Purpose: join, per beta, the 3f fine-tune report (val_NLL, val_consist, base
#          health) with its RECGATE beta_summary (soft/hard/nsf fwd -> dfwd) and
#          1.3a breakdown (RealNVP wins, tier, top2_share), then emit one table
#          + a dual-label verdict WITH the base-gaming sub-case.
# CONVENTION: missing inputs for a beta -> logger.error + raise (no silent skip).
#   fwd is the success gate; win count is context (carried from RNVP-DEPTH v0.2).
# Verdict per beta:
#   STRICT      dfwd < -PARITY_EPS
#   RESEARCH    |dfwd|<=eps AND tier non-FLAT AND wins>=SPEC_MIN_WINS
#   NLL_WRECKED delta_val_nll > +NLL_GUARD_NATS  (beta too large)
#   BASE_GAMING val_consist improved vs beta=0 BUT dfwd not strict AND base
#               health spiked/clamp-saturated (sigma collapse / KL jump /
#               mu-std blowup / high clamp fraction) -> base cheated, not the flow
#   CONSIST_ONLY val_consist improved, dfwd flat, tier FLAT, base health OK
#   NO_MOVEMENT  dfwd~0, tier FLAT, val_consist ~ unchanged
# Changelog (v0.2 -> v0.3, RNVP-CONSIST v0.4):
#   * Sweep axis is TARGET_GRAD_RATIO (dirs consist_tgrNN_seed0, glob
#     /tmp/consist_tgr*.json, breakdown results_consist_tgrNN). Table shows tgr,
#     derived beta, realized grad_ratio. Plots x-axis = target_grad_ratio; Plot A
#     adds target=realized diagonal. Control row is tgr=0.
# Changelog (v0.1 -> v0.2):
#   * Plots: pA_grad_ratio (beta vs grad_ratio + useful band), pB_beta_tradeoff
#     (3-panel beta vs val_consistency/delta_NLL/dfwd), pC_grad_cosine (conflict).
#     Rows now carry grad_ratio/grad_cosine/grad_norms + delta_val_consist.
# Changelog (NEW in v0.1):
#   * Introduced. Mirrors summarize_depth v0.2 dual-label logic + base-gaming.
# Update summary:
#   v0.1 turns the beta sweep into one verdict; base-gaming is a first-class
#   outcome (not folded into consistency-only) so the base-frozen arm is triggered.
# =============================================================================
from __future__ import annotations
import argparse
import glob
import json
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # headless
import matplotlib.pyplot as plt

logger = logging.getLogger("CSMF2.step_1_4b.aggregate_consist")
__version__ = "0.3"
__abbr__ = "STEP-1_4B"

PARITY_EPS = 1e-3
SPEC_MIN_WINS = 150
NLL_GUARD_NATS = 100.0
# base-gaming thresholds (relative to the beta=0 control's base health)
KL_SPIKE_FACTOR = 2.0          # base_kl more than 2x the control
MU_STD_SPIKE_FACTOR = 2.0      # mu_std_across_y more than 2x the control
CLAMP_FRAC_MAX = 0.50          # >50% of logsigma at a clamp = saturation


def _tgr_from_name(p: str) -> float:
    m = re.search(r"consist_tgr(\d+\.\d+)", p)
    if not m:
        logger.error("[agg] cannot parse target_grad_ratio from %s", p)
        raise ValueError(f"cannot parse target_grad_ratio from {p}")
    return float(m.group(1))


def _load_recgate(path: str):
    d = json.loads(Path(path).read_text())
    soft = min(d["soft_fwd_rel"]); hard = min(d["hard_fwd_rel"])
    nsf = d["nsf_only_fwd_rel"][0]
    return soft, hard, nsf, min(soft, hard) - nsf


def _load_breakdown(bd_dir: Path):
    j = bd_dir / "rec_argmin_breakdown.json"
    if not j.exists():
        logger.error("[agg] missing breakdown json %s", j)
        raise FileNotFoundError(j)
    bd = json.loads(j.read_text())
    # realnvp is the non-dominant target expert (index 1 in the triple)
    wins = bd.get("winner_totals", [0, 0, 0])[1]
    return wins, bd.get("top2_share", 0.0), bd.get("tier", "?")


def _plots(rows, out_dir: Path):
    """The 3f decision plots, x-axis = target_grad_ratio (the swept knob; beta is
    DERIVED). A: tgr vs realized grad_ratio (did we land in the useful band?).
    B: 3-panel tgr vs val_consistency/delta_NLL/dfwd (did consistency help, did
    NLL survive, did dfwd move?). C: tgr vs grad_cosine (do the terms fight?).
    tgr=0 has no consistency grad -> excluded from the grad plots (ratio 0/cos NaN)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tgrs = [r["tgr"] for r in rows]
    pos = [r for r in rows if r["tgr"] > 0.0]       # grad diagnostics need tgr>0
    bp = [r["tgr"] for r in pos]

    # ---- Plot A: beta vs grad_ratio ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bp, [r["grad_ratio"] for r in pos], "o-", color="#1f77b4")
    for lo, hi, c, lbl in [(0.05, 0.30, "#2ca02c", "useful band"),
                           (1.0, 1.0, "#d62728", "competition (~1)")]:
        if lo == hi:
            ax.axhline(lo, ls="--", color=c, alpha=0.7, label=lbl)
        else:
            ax.axhspan(lo, hi, color=c, alpha=0.15, label=lbl)
    ax.set_xlabel("target_grad_ratio"); ax.set_ylabel("realized grad_ratio")
    ax.plot([min(bp), max(bp)] if bp else [0,1], [min(bp), max(bp)] if bp else [0,1],
            ls=":", color="gray", alpha=0.5, label="target=realized")
    ax.set_title("3f Plot A -- gradient pressure (target vs realized)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "pA_grad_ratio.png", dpi=120)
    plt.close(fig)

    # ---- Plot B: 3-panel trade-off ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(tgrs, [r["val_consist"] for r in rows], "o-", color="#1f77b4")
    axes[0].set_title("tgr vs val_consistency"); axes[0].set_xlabel("target_grad_ratio")
    axes[0].set_ylabel("val_consistency (lower=better)")
    axes[1].plot(tgrs, [r["delta_val_nll"] for r in rows], "o-", color="#ff7f0e")
    axes[1].axhline(100.0, ls="--", color="#d62728", alpha=0.7, label="+100 guard")
    axes[1].set_title("tgr vs delta_NLL"); axes[1].set_xlabel("target_grad_ratio")
    axes[1].set_ylabel("delta_val_NLL (>0 = worse)"); axes[1].legend(fontsize=8)
    axes[2].plot(tgrs, [r["dfwd"] for r in rows], "o-", color="#2ca02c")
    axes[2].axhline(0.0, ls="-", color="k", alpha=0.4)
    axes[2].axhline(-1e-3, ls="--", color="#2ca02c", alpha=0.7, label="strict (-1e-3)")
    axes[2].set_title("tgr vs dfwd_vs_nsf"); axes[2].set_xlabel("target_grad_ratio")
    axes[2].set_ylabel("dfwd (<0 = beats NSF)"); axes[2].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle("3f Plot B -- target_grad_ratio trade-off (consistency / NLL / dfwd)")
    fig.tight_layout(); fig.savefig(out_dir / "pB_beta_tradeoff.png", dpi=120)
    plt.close(fig)

    # ---- Plot C: beta vs grad_cosine ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bp, [r["grad_cosine"] for r in pos], "o-", color="#9467bd")
    ax.axhline(0.0, ls="-", color="k", alpha=0.4)
    ax.set_ylim(-1.05, 1.05)
    ax.text(0.02, 0.92, "agree", transform=ax.transAxes, fontsize=8, color="#2ca02c")
    ax.text(0.02, 0.06, "fight", transform=ax.transAxes, fontsize=8, color="#d62728")
    ax.set_xlabel("target_grad_ratio"); ax.set_ylabel("cos(grad_NLL, grad_consistency)")
    ax.set_title("3f Plot C -- gradient conflict")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "pC_grad_cosine.png", dpi=120)
    plt.close(fig)
    logger.info("[agg] wrote plots pA_grad_ratio / pB_beta_tradeoff / pC_grad_cosine "
                "to %s", out_dir)


def aggregate(result_root: str, recgate_glob: str, breakdown_root: str):
    recgate_paths = sorted(glob.glob(recgate_glob))
    if not recgate_paths:
        logger.error("[agg] no RECGATE summaries match %s", recgate_glob)
        raise FileNotFoundError(recgate_glob)

    # control (beta=0) base health, for gaming comparison
    ctrl = None
    rows = []
    for rp in recgate_paths:
        tgr = _tgr_from_name(rp)
        rep_path = Path(result_root) / f"consist_tgr{tgr:.2f}_seed0" / "report.json"
        if not rep_path.exists():
            logger.error("[agg] missing 3f report %s", rep_path)
            raise FileNotFoundError(rep_path)
        rep = json.loads(rep_path.read_text())
        soft, hard, nsf, dfwd = _load_recgate(rp)
        bd_dir = Path(breakdown_root) / f"results_consist_tgr{tgr:.2f}"
        wins, top2, tier = _load_breakdown(bd_dir)
        last = rep["history"][-1]
        row = {
            "tgr": tgr, "beta": rep.get("derived_beta", rep.get("beta", float("nan"))),
            "val_nll": rep["val_nll"],
            "delta_val_nll": rep["delta_val_nll"],
            "val_consist": rep["val_consist_after"],
            "delta_val_consist": rep.get("delta_val_consist", float("nan")),
            "soft": soft, "hard": hard, "nsf": nsf, "dfwd": dfwd,
            "wins": wins, "top2": top2, "tier": tier,
            "base_kl": last["base_kl"], "mu_std": last["mu_std_across_y"],
            "at_min": last["fraction_at_sigma_min"],
            "at_max": last["fraction_at_sigma_max"],
            "base_alive": last["base_alive"],
            "grad_ratio": rep.get("grad_ratio", last.get("grad_ratio", float("nan"))),
            "grad_cosine": rep.get("grad_cosine", last.get("grad_cosine", float("nan"))),
            "grad_norm_nll": rep.get("grad_norm_nll", float("nan")),
            "grad_norm_consistency": rep.get("grad_norm_consistency", float("nan")),
        }
        rows.append(row)
        if tgr == 0.0:
            ctrl = row

    rows.sort(key=lambda r: r["tgr"])
    if ctrl is None:
        logger.error("[agg] no tgr=0 control found -- cannot gauge gaming/baseline")
        raise RuntimeError("missing beta=0 control")

    # per-beta verdict
    for r in rows:
        not_flat = (r["tier"] not in ("FLAT", "NONE", "?")
                    and r["wins"] >= SPEC_MIN_WINS)
        strict = r["dfwd"] < -PARITY_EPS
        parity = abs(r["dfwd"]) <= PARITY_EPS
        research = parity and not_flat
        nll_wrecked = r["delta_val_nll"] > NLL_GUARD_NATS
        consist_improved = r["val_consist"] < ctrl["val_consist"] - 1e-9
        base_spiked = (r["base_kl"] > KL_SPIKE_FACTOR * max(ctrl["base_kl"], 1e-6)
                       or r["mu_std"] > MU_STD_SPIKE_FACTOR * max(ctrl["mu_std"], 1e-6)
                       or r["at_min"] > CLAMP_FRAC_MAX or r["at_max"] > CLAMP_FRAC_MAX
                       or not r["base_alive"])
        if nll_wrecked:
            v = "NLL_WRECKED"
        elif strict:
            v = "STRICT"
        elif research:
            v = "RESEARCH"
        elif consist_improved and not strict and base_spiked:
            v = "BASE_GAMING"
        elif consist_improved and r["tier"] in ("FLAT", "NONE", "?"):
            v = "CONSIST_ONLY"
        else:
            v = "NO_MOVEMENT"
        r["verdict"] = v

    # overall: best (most negative) dfwd among non-wrecked betas
    elig = [r for r in rows if r["verdict"] != "NLL_WRECKED"]
    strict_any = any(r["verdict"] == "STRICT" for r in rows)
    research_any = any(r["verdict"] == "RESEARCH" for r in rows)
    gaming_any = any(r["verdict"] == "BASE_GAMING" for r in rows)
    best = min(elig, key=lambda r: r["dfwd"]) if elig else None

    # ---- print table ------------------------------------------------------
    print("=" * 78)
    print(f"RNVP-CONSIST v{__version__} -- target_grad_ratio sweep (Stage 1.4b-3f), seed0")
    print("=" * 78)
    hdr = (f"{'tgr':>5} {'beta':>9} {'g_rat':>8} {'val_NLL':>9} {'dNLL':>7} "
           f"{'val_con':>9} {'dfwd':>9} {'wins':>5} {'tier':>10} "
           f"{'baseKL':>7} {'verdict':>13}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        gr = r.get("grad_ratio", float("nan"))
        print(f"{r['tgr']:5.2f} {r['beta']:9.3g} {gr:8.2g} {r['val_nll']:9.1f} "
              f"{r['delta_val_nll']:7.1f} {r['val_consist']:9.4f} "
              f"{r['dfwd']:+9.4f} {r['wins']:5d} {r['tier']:>10} "
              f"{r['base_kl']:7.1f} {r['verdict']:>13}")
    print("-" * len(hdr))

    if strict_any:
        verdict = ("STRICT SUCCESS -- reconstruction pressure beats NSF-only at "
                   f"tgr={best['tgr']:.2f} (beta={best['beta']:.3g}, "
                   f"dfwd={best['dfwd']:+.4f}). Carry RealNVP consistency into "
                   "full Stage 2.3.")
    elif research_any:
        verdict = ("RESEARCH SIGNAL -- fwd parity + specialization (non-FLAT, "
                   ">=150 wins). A legit Stage-2.3 trigger.")
    elif gaming_any:
        verdict = ("BASE-GAMING suspected -- consistency improved but via base "
                   "drift, not the flow (dfwd not strict, base health spiked). "
                   "RERUN the best tgr with --freeze-base to confirm; if frozen "
                   "does NOT improve consistency, it was the base shortcut.")
    else:
        verdict = ("NO MOVEMENT / CONSISTENCY-ONLY -- weak RealNVP-only "
                   "reconstruction pressure did not move dfwd (tiers FLAT). "
                   "Strongest evidence yet to go to full Stage 2.3 (NSF unfrozen) "
                   "or rethink whether NSF dominance at s2/n0.05 is structural.")
    print("VERDICT:", verdict)
    print("=" * 78)

    out = {"rows": rows, "best_tgr": (best["tgr"] if best else None),
           "best_derived_beta": (best["beta"] if best else None),
           "strict_success": strict_any, "research_signal": research_any,
           "base_gaming_any": gaming_any, "verdict": verdict,
           "PARITY_EPS": PARITY_EPS, "SPEC_MIN_WINS": SPEC_MIN_WINS,
           "NLL_GUARD_NATS": NLL_GUARD_NATS}
    out_path = Path(result_root) / "consist_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    (Path(result_root) / "consist_summary.txt").write_text(verdict + "\n")
    _plots(rows, Path(result_root) / "plots")
    logger.info("[agg] wrote %s", out_path)
    return out


def _parse():
    p = argparse.ArgumentParser(description="Aggregate 3f consistency beta sweep")
    p.add_argument("--result-root",
                   default="./CSMF2/experiments/step_1_4b/results_consistency")
    p.add_argument("--recgate-glob", default="/tmp/consist_b*.json")
    p.add_argument("--breakdown-root",
                   default="./CSMF2/experiments/step_1_3a")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    a = _parse()
    aggregate(a.result_root, a.recgate_glob, a.breakdown_root)
