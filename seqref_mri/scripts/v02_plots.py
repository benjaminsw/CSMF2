# SEQREF-V02F v0.3 -- scripts.v02_plots
# LIFETIME: KEEP
# =============================================================================
# Purpose: candidate v0.2 closure figures (V02PLAN v0.2 §10). Reads ONLY
#          the published v02_facts.json (schema seqref-v02-facts/1);
#          every figure input is hard-checked against the recorded
#          aggregates (atol 1e-9) -- a missing key, a schema mismatch or
#          a failed consistency check is ERROR, never a skipped panel.
#          Outputs are DIAGNOSTIC-lifetime PNGs.
# CONVENTION: logger.error + typed raise (V02Error). No fallback, no
#   mock, no placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * Introduced under V02PLAN v0.2 (LOCKED 2026-08-21).
# v0.2 (bug fix, reviewer blocker 3, 2026-08-22): plot_decomposition
#   renders an explicit "undefined" marker for the evaluator's
#   defined-null shares (mean dNLL == 0); a mixed null/non-null pair
#   is ERROR (FACTS_SHAPE_MISMATCH); no bare TypeError.
# v0.3 (bug fixes, reviewer blocker, 2026-08-22): main() gained the
#   registered unexpected-exception boundary (logger.exception + exit
#   2); holdout-trajectory suptitle now names the locked estimator
#   (ratio of resampled arithmetic means), not "mean of ratios".
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seqref_mri.scripts.v02_manifests import V02Error

logger = logging.getLogger("seqref_mri.v02_plots")

__version__ = "0.3"
__abbr__ = "SEQREF-V02F"

SCHEMA = "seqref-v02-facts/1"
ATOL = 1e-9
CHECKPOINT_STEPS = (0, 1086, 2172, 3258)
V1_FLOOR = 0.10                     # quoted from V02SPEC §5
V2_BANDS = (0.25, 0.75)             # quoted from V02SPEC §5
V3_PSNR_MIN_DB = 1.0                # quoted from V02SPEC §5
V3_NMSE_RATIO_MAX = 0.75            # quoted from V02SPEC §5


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V02Error(f"{code}: {message}")


def _need(node: dict, key: str, path: str):
    if key not in node:
        _fail("FACTS_KEY_MISSING", f"{path}.{key} is absent from "
              f"v02_facts.json; no panel is skipped silently")
    return node[key]


def _finite(arr, path: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all():
        _fail("FACTS_VALUE_INVALID", f"{path} is not a finite 1-D array")
    return a


def _check_mean(arr: np.ndarray, recorded: float, path: str) -> None:
    if abs(float(arr.mean()) - float(recorded)) > ATOL:
        _fail("FACTS_CONSISTENCY_FAILED",
              f"{path}: derived mean {arr.mean()!r} != recorded "
              f"{recorded!r}")


def load_facts(path: str) -> dict:
    facts = json.loads(Path(path).read_text())
    if facts.get("schema") != SCHEMA:
        _fail("FACTS_SCHEMA_MISMATCH",
              f"schema {facts.get('schema')!r} != {SCHEMA!r}")
    if "verdict" in facts:
        _fail("FACTS_VERDICT_PRESENT",
              "v02_facts.json carries a campaign verdict key; schema "
              "forbids it (V02SPEC §9)")
    return facts


def plot_gain_summary(facts: dict, out: Path) -> None:
    v = _need(facts, "v1_v2_v3", "facts")
    g_train = float(_need(v, "g_train", "v1_v2_v3"))
    g_hold = float(_need(v, "g_hold", "v1_v2_v3"))
    r = _need(v, "r", "v1_v2_v3")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(["G_train", "G_hold"], [g_train, g_hold],
                color=["#4C72B0", "#55A868"])
    axes[0].axhline(V1_FLOOR, color="red", ls="--",
                    label=f"V1 floor {V1_FLOOR}")
    axes[0].set_ylabel("mean ΔNLL / dim")
    axes[0].set_title("Likelihood gains (V1)")
    axes[0].legend()
    if r is None:
        axes[1].text(0.5, 0.5, "R = null\n(G_train = 0;\nnever coerced)",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Transfer ratio R (V2)")
    else:
        axes[1].bar(["R"], [float(r)], color="#8172B2")
        for band in V2_BANDS:
            axes[1].axhline(band, color="red", ls="--",
                            label=f"band {band}")
        axes[1].set_title("Transfer ratio R (V2)")
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "v02_gain_summary.png", dpi=150)
    plt.close(fig)


def plot_per_slice_delta(facts: dict, out: Path) -> None:
    em = _need(facts, "endpoint_measurements", "facts")
    for pop in ("train", "holdout"):
        blk = _need(em, pop, "endpoint_measurements")
        n0 = _finite(_need(blk, "nll_step0", pop), f"{pop}.nll_step0")
        n1 = _finite(_need(blk, "nll_final", pop), f"{pop}.nll_final")
        if n0.shape != n1.shape:
            _fail("FACTS_SHAPE_MISMATCH",
                  f"{pop}: step0 {n0.shape} vs final {n1.shape}")
        d = (n0 - n1) / 13824.0
        _check_mean(d, float(_need(blk, "mean_gain_per_dim", pop)),
                    f"{pop} gain")
        if pop == "train":
            d_train = d
        else:
            d_hold = d
    fig, ax = plt.subplots(figsize=(7, 4))
    for d, name, color in ((d_train, f"train (n={d_train.size})",
                            "#4C72B0"),
                           (d_hold, f"holdout (n={d_hold.size})",
                            "#C44E52")):
        xs = np.sort(d)
        ax.plot(xs, np.arange(1, xs.size + 1) / xs.size, label=name,
                color=color)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("per-slice ΔNLL / dim (step0 − final)")
    ax.set_ylabel("ECDF")
    ax.set_title("Per-slice likelihood gain distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "v02_per_slice_delta.png", dpi=150)
    plt.close(fig)


def plot_decomposition(facts: dict, out: Path) -> None:
    sm = _need(facts, "secondary_monitoring", "facts")
    d2b = _need(sm, "d2b_decomposition", "secondary_monitoring")
    pops, base_shares, ldj_shares, null_pops = [], [], [], []
    for pop in ("train", "holdout"):
        blk = _need(d2b, pop, "d2b_decomposition")
        b = _need(blk, "base_share_pct", pop)
        l = _need(blk, "logdet_share_pct", pop)
        if (b is None) != (l is None):
            _fail("FACTS_SHAPE_MISMATCH",
                  f"{pop}: exactly one decomposition share is null; the "
                  f"defined-null contract is both-or-neither")
        pops.append(pop)
        if b is None:
            # Defined null (mean endpoint NLL change exactly 0.0, e.g.
            # the V1-fail outcome): render an explicit "undefined"
            # marker like the R-null branch -- never float(None).
            null_pops.append(pop)
            base_shares.append(0.0)
            ldj_shares.append(0.0)
        else:
            base_shares.append(float(b))
            ldj_shares.append(float(l))
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(pops))
    ax.bar(x - 0.2, base_shares, width=0.4, label="L_base share %",
           color="#4C72B0")
    ax.bar(x + 0.2, ldj_shares, width=0.4, label="L_logdet share %",
           color="#DD8452")
    for i, pop in enumerate(pops):
        if pop in null_pops:
            ax.text(x[i], 5.0, "undefined\n(mean \u0394NLL = 0;\nnever "
                    "coerced)", ha="center", va="bottom", fontsize=8,
                    color="black")
    ax.axhline(100.0, color="black", lw=0.8)
    ax.set_xticks(x, pops)
    ax.set_title("D2b endpoint decomposition (same-encode byproduct)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "v02_decomposition.png", dpi=150)
    plt.close(fig)


def plot_holdout_trajectory(facts: dict, out: Path) -> None:
    em = _need(facts, "endpoint_measurements", "facts")
    hold = _need(em, "holdout", "endpoint_measurements")
    traj = _need(hold, "per_checkpoint", "holdout")
    if [int(t["step"]) for t in traj] != list(CHECKPOINT_STEPS):
        _fail("FACTS_CHECKPOINT_MISMATCH",
              "holdout per_checkpoint steps diverge from the locked "
              f"{list(CHECKPOINT_STEPS)}")
    steps = [int(t["step"]) for t in traj]
    psnr = [float(t["z0_psnr_mean"]) for t in traj]
    nmse = [float(t["z0_nmse_u_mean"]) for t in traj]
    v3 = _need(facts, "v1_v2_v3", "facts")
    v3_blk = _need(v3, "v3", "v1_v2_v3")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps, psnr, "o-", color="#4C72B0")
    axes[0].axhline(psnr[0] + V3_PSNR_MIN_DB, color="red", ls="--",
                    label=f"V3: step0 + {V3_PSNR_MIN_DB} dB")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("z=0 PSNR (dB)")
    axes[0].set_title("Holdout z=0 PSNR"); axes[0].legend()
    axes[1].plot(steps, nmse, "o-", color="#C44E52")
    axes[1].axhline(nmse[0] * V3_NMSE_RATIO_MAX, color="red", ls="--",
                    label=f"V3: step0 x {V3_NMSE_RATIO_MAX}")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("z=0 NMSE_u")
    axes[1].set_title("Holdout z=0 NMSE_u"); axes[1].legend()
    ratio = _need(v3_blk, "nmse_ratio_bootstrap_mean", "v3")
    fig.suptitle("V3 bootstrap NMSE ratio (mean over B resamples of the "
                 "ratio of resampled arithmetic means): "
                 f"{float(ratio):.4f}")
    fig.tight_layout()
    fig.savefig(out / "v02_holdout_trajectory.png", dpi=150)
    plt.close(fig)


def plot_d3_monitor(facts: dict, out: Path) -> None:
    sm = _need(facts, "secondary_monitoring", "facts")
    d3 = _need(sm, "d3_monitor", "secondary_monitoring")
    conds = _need(d3, "conditions", "d3_monitor")
    names = [str(c["condition"]) for c in conds]
    dnll = [float(c["mean_delta_nll"]) for c in conds]
    dpsnr = [float(c["mean_delta_psnr"]) for c in conds]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, dnll, color="#4C72B0")
    axes[0].set_title("D3 monitor: ΔNLL by condition (final state)")
    axes[1].bar(names, dpsnr, color="#55A868")
    axes[1].set_title("D3 monitor: ΔPSNR by condition (final state)")
    fig.tight_layout()
    fig.savefig(out / "v02_d3_monitor.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"{__abbr__} v{__version__} -- candidate v0.2 "
                    f"closure figures (facts-only, DIAGNOSTIC outputs)")
    ap.add_argument("--facts", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s "
                               "%(message)s")
    try:
        facts = load_facts(args.facts)
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        plot_gain_summary(facts, out)
        plot_per_slice_delta(facts, out)
        plot_decomposition(facts, out)
        plot_holdout_trajectory(facts, out)
        plot_d3_monitor(facts, out)
    except V02Error:
        return 2
    except Exception:  # noqa: BLE001 -- the registered boundary: no
        logger.exception("[%s] unexpected runtime failure", __abbr__)
        return 2                # exception may escape as exit 1
    logger.info("[%s] 5 figures written to %s", __abbr__, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
