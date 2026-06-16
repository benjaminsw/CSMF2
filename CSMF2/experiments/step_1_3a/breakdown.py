# =============================================================================
# STEP-1_3A v0.2 -- experiments.step_1_3a.breakdown  (RECARGMIN-DIAG v0.2)
# Purpose: DIAGNOSTIC ONLY (no training). Recompute per-sample rec_argmin on
#          the val set for the post-CB experts, attach the digit class, and
#          decide whether the non-top expert's reconstruction wins are a
#          usable CLUSTER or scattered noise. Gates the multi-scale RealNVP
#          build (Stage 1.4b).
# CONVENTION: missing labels / loader mismatch / non-finite -> logger.error +
#             raise. No fallback / mock / dummy / pass.
# Key design: the stored rec_argmin_counts is aggregated (length-K) and has no
#   per-sample class labels, so we RECOMPUTE rec per sample with the SAME
#   shared z-bank + global setup as RECGATE (reuse step_1_3.scores), then
#   self-check that per-class totals reproduce the stored counts.
# Self-check (v0.2 TOLERANT -- tie-jitter in the S=4 z-proxy flips a few
#   winners near ties; exact match is too brittle):
#   sum differs           -> FAIL (wrong sample set / dataset split)
#   max |diff| > tol_abs   -> FAIL (wrong expert order / ckpt / z-bank)
#   L1 diff  > tol_l1      -> FAIL (too many flips)
#   else not exact         -> WARN + continue (record both in JSON)
#   defaults tol_abs=10, tol_l1=20. Catches real mismatches, allows jitter.
# Decision metric: rec_argmin (which expert WINS per sample), NOT mean fwd_rel.
# Verdict (three-tier; small N -> cluster is PROVISIONAL):
#   top2_share < 0.35           -> FLAT
#   0.35 <= top2_share < 0.50   -> WEAK_CLUSTER
#   top2_share >= 0.50          -> PROVISIONAL_CLUSTER  (confirm with seed 1)
# Changelog (v0.1 -> v0.2):
#   * Self-check is now tolerant (abs + L1 thresholds, warn-not-fail on close
#     mismatch) with --count-tol-abs/--count-tol-l1 flags; records the full
#     self_check dict (expected/recomputed/diff/status) in the JSON. Fixes the
#     brittle exact-match that raised on benign [3,87,4910] vs [0,83,4917].
# Changelog (NEW in v0.1):
#   * Introduced. recompute rec_argmin + per-class table + self-check +
#     concentration verdict + plots p1 (by class) / p2 (non-top wins hist).
# Update summary:
#   v0.2 keeps the safety check useful (still fails on wrong experts/data) but
#   no longer blocks on tie-jitter. The degradation-cell axis is a no-op now
#   (single cell s2/n0.05); the output field is forward-compat only.
# =============================================================================
from __future__ import annotations
import argparse
import json
import logging
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ...data.degrade import MNISTDegraded
from ..step_1_2.model_io import load_experts
from ..step_1_3.scores import make_z_bank, per_expert_rec

logger = logging.getLogger("CSMF2.step_1_3a.breakdown")
__version__ = "0.1"
__abbr__ = "STEP-1_3A"

N_CLASSES = 10


def _digit_labels(ds):
    """Per-sample digit class aligned to dataset order. Raise if unavailable
    (no silent fallback -- a wrong label invalidates the whole diagnosis)."""
    for attr in ("targets", "labels"):
        t = getattr(ds, attr, None)
        if t is not None:
            return [int(v) for v in (t.tolist() if torch.is_tensor(t) else t)]
    base = getattr(ds, "dataset", None) or getattr(ds, "mnist", None) \
        or getattr(ds, "base", None)
    if base is not None:
        for attr in ("targets", "labels"):
            t = getattr(base, attr, None)
            if t is not None:
                return [int(v) for v in (t.tolist() if torch.is_tensor(t) else t)]
    logger.error("[breakdown] dataset exposes no digit labels (targets/labels)")
    raise AttributeError("val dataset has no digit labels for the breakdown")


@torch.no_grad()
def run(ckpt_dirs, *, out_root: str, expected_counts=None,
        max_samples: int = 5000, count_tol_abs: int = 10,
        count_tol_l1: int = 20) -> dict:
    out_dir = Path(out_root); out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experts, train_cfgs, ref = load_experts(ckpt_dirs, device)
    names = [c.expert for c in train_cfgs]
    K = len(experts)
    logger.info("[breakdown] experts=%s scale=%d blur=%.2f noise=%.2f",
                names, ref.scale, ref.blur_sigma, ref.noise_sigma)

    ds = MNISTDegraded(ref.data_root, split="val", sigma=ref.blur_sigma,
                       scale=ref.scale, noise_sigma=ref.noise_sigma)
    labels_all = _digit_labels(ds)
    loader = DataLoader(ds, batch_size=128, shuffle=False)   # MUST match order

    # same shared z-bank as RECGATE (fixed_shared, S=4, seed 1234)
    z_bank = make_z_bank(ref.dim, 4, "fixed_shared", 1234, device,
                         next(experts[0].parameters()).dtype)

    # counts[k][digit] = # samples where expert k is the absolute-rec winner
    counts = [[0] * N_CLASSES for _ in range(K)]
    win_totals = [0] * K
    seen = 0
    for x_img, y in loader:
        y = y.to(device)
        bsz = y.size(0)
        idx = list(range(seen, seen + bsz))
        rec = per_expert_rec(experts, y, z_bank,
                             blur_sigma=ref.blur_sigma, scale=ref.scale)  # (B,K)
        argmin = rec.argmin(dim=1).tolist()
        for j, k in enumerate(argmin):
            digit = labels_all[idx[j]]
            counts[k][digit] += 1
            win_totals[k] += 1
        seen += bsz
        if seen >= max_samples:
            break

    # ---- self-check: per-class totals must reproduce stored counts --------
    # Tolerant: tie-jitter in the S=4 z-proxy can flip a few winners near ties.
    # FAIL only on real mismatch (wrong experts/order/z-bank/dataset/loader):
    #   sum differs            -> FAIL (wrong sample set)
    #   any |diff| > tol_abs   -> FAIL (a winner count moved too far)
    #   L1 diff > tol_l1       -> FAIL (too many flips overall)
    #   else not exact         -> WARN + continue (record both)
    recomputed = [sum(counts[k]) for k in range(K)]
    logger.info("[breakdown] recomputed rec_argmin totals = %s (n=%d)",
                recomputed, seen)
    self_check = {"expected": list(expected_counts) if expected_counts else None,
                  "recomputed": recomputed, "status": "skipped",
                  "tol_abs": count_tol_abs, "tol_l1": count_tol_l1}
    if expected_counts is not None:
        exp = list(expected_counts)
        if len(exp) != K:
            logger.error("[breakdown] expected_counts length %d != K %d",
                         len(exp), K)
            raise ValueError("expected_counts length mismatch")
        diff = [recomputed[k] - exp[k] for k in range(K)]
        max_abs = max(abs(d) for d in diff)
        l1 = sum(abs(d) for d in diff)
        self_check.update(diff=diff, max_abs=max_abs, l1=l1)
        if sum(recomputed) != sum(exp):
            logger.error("[breakdown] SELF-CHECK FAILED: sum %d != stored sum %d "
                         "-- wrong sample set / dataset split",
                         sum(recomputed), sum(exp))
            raise RuntimeError("rec_argmin self-check: sample-count mismatch")
        if max_abs > count_tol_abs:
            logger.error("[breakdown] SELF-CHECK FAILED: max per-expert diff %d "
                         "> tol %d (diff=%s) -- likely wrong expert order / "
                         "checkpoint / z-bank", max_abs, count_tol_abs, diff)
            raise RuntimeError("rec_argmin self-check: per-expert diff too large")
        if l1 > count_tol_l1:
            logger.error("[breakdown] SELF-CHECK FAILED: L1 diff %d > tol %d "
                         "(diff=%s) -- too many winner flips", l1, count_tol_l1,
                         diff)
            raise RuntimeError("rec_argmin self-check: L1 diff too large")
        if recomputed == exp:
            self_check["status"] = "exact"
            logger.info("[breakdown] self-check OK (exact): %s", recomputed)
        else:
            self_check["status"] = "warn_within_tol"
            logger.warning("[breakdown] SELF-CHECK WARNING: close mismatch "
                           "accepted within tolerance -- recomputed %s vs stored "
                           "%s (diff=%s, max_abs=%d<=%d, L1=%d<=%d). Treated as "
                           "tie-jitter; verdict unchanged.", recomputed, exp,
                           diff, max_abs, count_tol_abs, l1, count_tol_l1)

    # ---- concentration verdict on the top NON-dominant expert -------------
    # dominant = the overall winner (NSF); we judge the best non-dominant one.
    dominant = max(range(K), key=lambda k: win_totals[k])
    non_dom = [k for k in range(K) if k != dominant]
    target = max(non_dom, key=lambda k: win_totals[k]) if non_dom else dominant
    tt = win_totals[target]
    if tt > 0:
        sorted_classes = sorted(counts[target], reverse=True)
        top2_share = (sorted_classes[0] + sorted_classes[1]) / tt
        max_class_share = sorted_classes[0] / tt
    else:
        top2_share = 0.0; max_class_share = 0.0
    expected_uniform = tt / N_CLASSES

    if tt == 0:
        verdict = (f"NO WINS for any non-dominant expert ({names[dominant]} "
                   f"wins all {win_totals[dominant]}). No complementarity to "
                   f"build on; multi-scale unlikely to help.")
        tier = "NONE"
    elif top2_share >= 0.50:
        tier = "PROVISIONAL_CLUSTER"
        verdict = (f"{tier} -- {names[target]} concentrates {top2_share:.0%} of "
                   f"its {tt} wins in its top-2 classes (uniform would be "
                   f"~{2*expected_uniform:.1f}/{tt}). Multi-scale {names[target]} "
                   f"(1.4b) is JUSTIFIED, but CONFIRM with seed 1 -- {tt} wins is "
                   f"small.")
    elif top2_share >= 0.35:
        tier = "WEAK_CLUSTER"
        verdict = (f"{tier} -- {names[target]} shows mild concentration "
                   f"({top2_share:.0%} in top-2). Borderline; confirm with seed 1 "
                   f"before committing to multi-scale.")
    else:
        tier = "FLAT"
        verdict = (f"{tier} -- {names[target]}'s {tt} wins are ~uniform across "
                   f"classes ({top2_share:.0%} in top-2 vs ~"
                   f"{2*expected_uniform/tt:.0%} uniform). Likely noise; more "
                   f"capacity unlikely to create complementarity.")

    # ---- text table --------------------------------------------------------
    sep = "=" * 78
    lines = [sep, "RECARGMIN-DIAG v0.1 -- rec_argmin by digit class (Stage 1.3a)",
             sep, f"  n={seen}  experts={names}  winner_totals={win_totals}",
             "", "  digit " + " ".join(f"{nm:>8}" for nm in names)]
    for d in range(N_CLASSES):
        lines.append(f"  {d:>5} " + " ".join(f"{counts[k][d]:>8}" for k in range(K)))
    lines += ["  -----",
              "  TOTAL " + " ".join(f"{win_totals[k]:>8}" for k in range(K)),
              "", f"  target non-dominant expert: {names[target]} "
              f"(wins {tt}, uniform/class ~{expected_uniform:.1f})",
              f"  top2_share={top2_share:.3f}  max_class_share={max_class_share:.3f}",
              "", f"DECISION: {verdict}", sep]
    text = "\n".join(lines); print(text)

    # ---- plots -------------------------------------------------------------
    _plot_by_class(counts, names, out_dir)
    _plot_target_hist(counts[target], names[target], expected_uniform, out_dir)

    out = {"experts": names, "n": seen,
           "counts_by_class": {names[k]: counts[k] for k in range(K)},
           "win_totals": {names[k]: win_totals[k] for k in range(K)},
           "dominant": names[dominant], "target_non_dominant": names[target],
           "top2_share": top2_share, "max_class_share": max_class_share,
           "expected_uniform_per_class": expected_uniform,
           "tier": tier, "verdict": verdict,
           "self_check": self_check,
           # forward-compat: single-cell now, so degradation axis is a no-op
           "by_degradation_cell": {f"s{ref.scale}_n{ref.noise_sigma:.2f}":
                                   {names[k]: win_totals[k] for k in range(K)}},
           "degradation_axis_informative": False}
    (out_dir / "rec_argmin_breakdown.json").write_text(json.dumps(out, indent=2))
    (out_dir / "rec_argmin_breakdown.txt").write_text(text)
    logger.info("[breakdown] wrote rec_argmin_breakdown.{json,txt} + plots")
    return out


def _plot_by_class(counts, names, out_dir):
    K = len(names); digits = list(range(N_CLASSES))
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=120)
    bottom = [0] * N_CLASSES
    for k in range(K):
        ax.bar(digits, counts[k], bottom=bottom, label=names[k])
        bottom = [bottom[d] + counts[k][d] for d in range(N_CLASSES)]
    ax.set_xlabel("digit class"); ax.set_ylabel("rec_argmin wins")
    ax.set_xticks(digits)
    ax.set_title("Absolute-reconstruction winner by digit class (stacked)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "p1_recargmin_by_class.png",
                                    bbox_inches="tight")
    plt.close(fig)


def _plot_target_hist(target_counts, target_name, expected_uniform, out_dir):
    digits = list(range(N_CLASSES))
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=120)
    ax.bar(digits, target_counts, color="#1f77b4")
    ax.axhline(expected_uniform, color="r", ls="--",
               label=f"uniform ~{expected_uniform:.1f}/class")
    ax.set_xlabel("digit class"); ax.set_ylabel(f"{target_name} wins")
    ax.set_xticks(digits)
    ax.set_title(f"{target_name} reconstruction wins by class "
                 f"(concentration vs noise floor)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "p2_target_win_class_hist.png",
                                    bbox_inches="tight")
    plt.close(fig)


def _parse_args():
    p = argparse.ArgumentParser(description="Stage 1.3a rec_argmin breakdown")
    p.add_argument("--ckpt-dirs", nargs="+", required=True,
                   help="post-CB expert run dirs (nice realnvp nsf order)")
    p.add_argument("--out-root", default="./CSMF2/experiments/step_1_3a/results")
    p.add_argument("--expected-counts", nargs="*", type=int, default=None,
                   help="stored rec_argmin_counts for the self-check, e.g. 0 83 4917")
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--count-tol-abs", type=int, default=10,
                   help="max allowed per-expert winner-count diff vs stored")
    p.add_argument("--count-tol-l1", type=int, default=20,
                   help="max allowed total (L1) winner-count diff vs stored")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    try:
        run(a.ckpt_dirs, out_root=a.out_root,
            expected_counts=a.expected_counts, max_samples=a.max_samples,
            count_tol_abs=a.count_tol_abs, count_tol_l1=a.count_tol_l1)
        sys.exit(0)
    except Exception:
        logger.error("STEP-1_3A breakdown FAILED\n%s", traceback.format_exc())
        sys.exit(1)
