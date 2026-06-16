# =============================================================================
# STEP-1_3 v0.2 -- experiments.step_1_3.run
# Purpose: one reconstruction-aware gate run (RECGATE v0.3). Frozen experts.
#          (1) frozen calibration (per-expert NLL mu/sigma + chosen rec norm),
#          (2) cache standardized hybrid scores over train, (3) train gate to
#          minimize sum_k w_k(y)*score_k, (4) eval gate: NLL / fwd_rel
#          (soft+hard) / PSNR / Neff / usage / argmin counts + NSF-only
#          baseline. Writes report.json and plots 7-9.
# CONVENTION: non-finite / sigma<floor / weight-sum error -> raise (exit 1).
#             No fallback / mock / dummy / pass. rec = deterministic proxy.
# Exit codes: 0 = ran + report; 1 = crash / safety violation.
# Changelog (v0.1 -> v0.2):
#   * Threads cfg.rec_norm into calibration_stats (default 'global'); report
#     calibration block now tolerates the rec_norm string field.
# Changelog (NEW in v0.1):
#   * Introduced.
# Update summary:
#   v0.2 fixes reconstruction routing: with rec_norm='global' the gate is
#   rewarded for lower ABSOLUTE residual, so rising beta should favor the best
#   reconstructor. Gate trains on a fixed cached score (experts frozen).
# =============================================================================
from __future__ import annotations
import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ...data.degrade import MNISTDegraded, dequantize_logit, inverse_logit, blur, downsample
from ..step_1_2.model_io import load_experts
from ..step_1_2.mixture import LearnedGlobalGate, mixture_logp, per_expert_logp, gate_metrics
from .config import Stage13Cfg
from .scores import (make_z_bank, per_expert_nll, per_expert_rec,
                     per_expert_recon_pixels, calibration_stats, hybrid_score)

logger = logging.getLogger("CSMF2.step_1_3.run")
__version__ = "0.1"
__abbr__ = "STEP-1_3"
_IMAGE_HW = (28, 28)


def _cfg_log(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


def _loader(ref, split, bs, shuffle):
    ds = MNISTDegraded(ref.data_root, split=split, sigma=ref.blur_sigma,
                       scale=ref.scale, noise_sigma=ref.noise_sigma)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def _A(x_pix, blur_sigma, scale):
    return downsample(blur(x_pix, blur_sigma), scale)


def run(cfg: Stage13Cfg) -> dict:
    out_dir = Path(cfg.out_root) / cfg.run_tag()
    plots = out_dir / "plots"; plots.mkdir(parents=True, exist_ok=True)
    _cfg_log(out_dir)
    logger.info("STEP-1_3 run | tag=%s | cfg=%s", cfg.run_tag(), cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    experts, train_cfgs, ref = load_experts(list(cfg.ckpt_dirs), device)
    K = len(experts); names = [c.expert for c in train_cfgs]
    dim = int(experts[0].dim)
    blur_sigma, scale = ref.blur_sigma, ref.scale
    y_in = (28 // scale) * (28 // scale)
    z_bank = make_z_bank(dim, cfg.z_bank_size, cfg.rec_z_mode,
                         cfg.z_bank_seed, device, next(experts[0].parameters()).dtype)

    # ---- (1) calibration: frozen per-expert mu/sigma -----------------------
    cal_loader = _loader(ref, "train", cfg.batch_size, shuffle=True)
    nll_acc, rec_acc = [], []
    for bi, (x_img, y) in enumerate(cal_loader):
        if bi >= cfg.calib_batches:
            break
        x_img = x_img.to(device); y = y.to(device)
        if x_img.dim() == 3:
            x_img = x_img.unsqueeze(1)
        x_logit, ldj_deq = dequantize_logit(x_img, generator=gen)
        nll_acc.append(per_expert_nll(experts, x_logit.flatten(1), y, ldj_deq))
        rec_acc.append(per_expert_rec(experts, y, z_bank,
                                      blur_sigma=blur_sigma, scale=scale))
    stats = calibration_stats(torch.cat(nll_acc), torch.cat(rec_acc),
                              min_sigma=cfg.min_sigma, rec_norm=cfg.rec_norm)
    logger.info("[calib] mu_nll=%s sigma_nll=%s mu_rec=%s sigma_rec=%s",
                stats["mu_nll"].tolist(), stats["sigma_nll"].tolist(),
                stats["mu_rec"].tolist(), stats["sigma_rec"].tolist())

    # ---- (2) cache standardized scores over train --------------------------
    tr_loader = _loader(ref, "train", cfg.batch_size, shuffle=True)
    Y_tr, SC_tr = [], []
    for bi, (x_img, y) in enumerate(tr_loader):
        if bi >= cfg.train_batches:
            break
        x_img = x_img.to(device); y = y.to(device)
        if x_img.dim() == 3:
            x_img = x_img.unsqueeze(1)
        x_logit, ldj_deq = dequantize_logit(x_img, generator=gen)
        nll_ke = per_expert_nll(experts, x_logit.flatten(1), y, ldj_deq)
        rec_ke = per_expert_rec(experts, y, z_bank, blur_sigma=blur_sigma,
                                scale=scale)
        score, _, _ = hybrid_score(nll_ke, rec_ke, stats,
                                   alpha=cfg.alpha, beta=cfg.beta)
        Y_tr.append(y.detach()); SC_tr.append(score.detach())
    Y_tr = torch.cat(Y_tr); SC_tr = torch.cat(SC_tr)
    n_tr = Y_tr.size(0)
    logger.info("[cache] train cached: %d samples", n_tr)

    # ---- (3) train gate: minimize sum_k w_k(y) * score_k -------------------
    gate = LearnedGlobalGate(y_in, K, cfg.gate_hidden, cfg.tau).to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=cfg.lr_gate)
    curve = []
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n_tr, device=device)
        gate.train(); run_loss, n = 0.0, 0
        for i in range(0, n_tr, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            yb = Y_tr[idx]; sb = SC_tr[idx]
            log_w = gate.log_weights(yb)
            loss = (log_w.exp() * sb).sum(dim=1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run_loss += float(loss) * yb.size(0); n += yb.size(0)
        curve.append(run_loss / n)
        if epoch % max(1, cfg.epochs // 10) == 0:
            logger.info("[gate] epoch %d score-loss=%.4f", epoch, run_loss / n)

    # ---- (4) eval on val ---------------------------------------------------
    gate.eval()
    val_loader = _loader(ref, "val", cfg.batch_size, shuffle=False)
    best_k = int(torch.argmin(stats["mu_nll"]))     # best-calib-NLL expert (NSF)
    agg = {"nll_mix": 0.0, "n": 0,
           "soft_num": 0.0, "soft_den": 0.0, "hard_num": 0.0, "hard_den": 0.0,
           "soft_mse": 0.0, "hard_mse": 0.0,
           "base_num": 0.0, "base_den": 0.0, "base_nll": 0.0}
    lp_all, logw_all, ldj_all = [], [], []
    score_all, nlln_all, recn_all = [], [], []
    sc_argmin = torch.zeros(K, dtype=torch.long)
    rec_argmin = torch.zeros(K, dtype=torch.long)
    nll_argmin = torch.zeros(K, dtype=torch.long)
    w_rows = []
    for x_img, y in val_loader:
        x_img = x_img.to(device); y = y.to(device)
        if x_img.dim() == 3:
            x_img = x_img.unsqueeze(1)
        x_clean = x_img
        x_logit, ldj_deq = dequantize_logit(x_img, generator=gen)
        nll_ke = per_expert_nll(experts, x_logit.flatten(1), y, ldj_deq)
        rec_ke = per_expert_rec(experts, y, z_bank, blur_sigma=blur_sigma, scale=scale)
        score, nll_norm, rec_norm = hybrid_score(nll_ke, rec_ke, stats,
                                                 alpha=cfg.alpha, beta=cfg.beta)
        lp_ke = per_expert_logp(experts, x_logit.flatten(1), y)
        log_w = gate.log_weights(y); w = log_w.exp()
        # mixture NLL under the gate
        lp_mix = mixture_logp(lp_ke, log_w, ldj_deq)
        agg["nll_mix"] += float((-lp_mix).sum()); agg["n"] += y.size(0)
        # reconstructions (shared z-bank pixel mean per expert)
        xk = per_expert_recon_pixels(experts, y, z_bank)          # (B,K,1,28,28)
        soft = (w.view(-1, K, 1, 1, 1) * xk).sum(dim=1)           # (B,1,28,28)
        hard = xk[torch.arange(y.size(0)), w.argmax(dim=1)]       # (B,1,28,28)
        for tag, xhat in (("soft", soft), ("hard", hard)):
            Ax = _A(xhat, blur_sigma, scale)
            agg[f"{tag}_num"] += float((Ax - y).flatten(1).norm(dim=1).sum())
            agg[f"{tag}_den"] += float(y.flatten(1).norm(dim=1).sum())
            agg[f"{tag}_mse"] += float((xhat - x_clean).flatten(1).pow(2).mean(dim=1).sum())
        # NSF-only baseline (best-calib expert)
        xb = xk[:, best_k]
        Axb = _A(xb, blur_sigma, scale)
        agg["base_num"] += float((Axb - y).flatten(1).norm(dim=1).sum())
        agg["base_den"] += float(y.flatten(1).norm(dim=1).sum())
        agg["base_nll"] += float(nll_ke[:, best_k].sum())
        # argmin counts (lower=better) + caches
        sc_argmin += torch.bincount(score.argmin(1).cpu(), minlength=K)
        rec_argmin += torch.bincount(rec_ke.argmin(1).cpu(), minlength=K)
        nll_argmin += torch.bincount(nll_ke.argmin(1).cpu(), minlength=K)
        lp_all.append(lp_ke.detach()); logw_all.append(log_w.detach())
        ldj_all.append(ldj_deq.detach())
        score_all.append(score.detach()); nlln_all.append(nll_norm.detach())
        recn_all.append(rec_norm.detach()); w_rows.append(w.detach())

    n = agg["n"]
    lp_ke = torch.cat(lp_all); log_w = torch.cat(logw_all)
    gm = gate_metrics(log_w, lp_ke, weight_sum_tol=cfg.weight_sum_tol)
    score_all = torch.cat(score_all); nlln_all = torch.cat(nlln_all)
    recn_all = torch.cat(recn_all); w_all = torch.cat(w_rows)

    soft_fwd = agg["soft_num"] / agg["soft_den"]
    hard_fwd = agg["hard_num"] / agg["hard_den"]
    base_fwd = agg["base_num"] / agg["base_den"]
    soft_psnr = 10.0 * torch.log10(torch.tensor(1.0 / max(agg["soft_mse"] / n, 1e-12)))
    hard_psnr = 10.0 * torch.log10(torch.tensor(1.0 / max(agg["hard_mse"] / n, 1e-12)))

    report = {
        "stage13_cfg": cfg.__dict__, "expert_names": names,
        "calibration": {k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in stats.items()},
        "score_diag": {
            "nll_norm_mean": nlln_all.mean(0).tolist(),
            "nll_norm_std": nlln_all.std(0).tolist(),
            "rec_norm_mean": recn_all.mean(0).tolist(),
            "rec_norm_std": recn_all.std(0).tolist(),
            "score_mean": score_all.mean(0).tolist(),
            "score_std": score_all.std(0).tolist(),
            "score_argmin_counts": sc_argmin.tolist(),
            "rec_argmin_counts": rec_argmin.tolist(),
            "nll_argmin_counts": nll_argmin.tolist(),
        },
        "gate": gm,
        "reconstruction": {
            "soft_fwd_rel": soft_fwd, "hard_fwd_rel": hard_fwd,
            "soft_PSNR": float(soft_psnr), "hard_PSNR": float(hard_psnr),
            "mixture_NLL": agg["nll_mix"] / n,
            "nsf_only_fwd_rel": base_fwd,
            "nsf_only_NLL": agg["base_nll"] / n,
            "baseline_expert": names[best_k],
        },
        "decision_helpers": {
            "neff_gt_1p5": bool(gm["Neff_mean"] > 1.5),
            "max_weight_lt_0p70": bool(max(gm["mean_weight_per_expert"]) < 0.70),
            "fwd_rel_improved_vs_nsf_only": bool(soft_fwd < base_fwd or hard_fwd < base_fwd),
            "best_expert_varies": bool(sum(1 for c in sc_argmin.tolist() if c > 0) > 1),
        },
        "score_loss_curve": curve,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    _plot_scatter(nlln_all, recn_all, names, plots / "p7_nllnorm_vs_recnorm.png")
    _plot_heatmap(w_all, names, plots / "p8_gate_weight_heatmap.png")
    _plot_score_hist(score_all, names, plots / "p9_score_hist.png")

    logger.info("STEP-1_3 DONE a=%.1f b=%.1f Neff=%.3f soft_fwd=%.4f "
                "nsf_only_fwd=%.4f usage=%s", cfg.alpha, cfg.beta,
                gm["Neff_mean"], soft_fwd, base_fwd,
                gm["mean_weight_per_expert"])
    return report


# ---- plots 7-9 -------------------------------------------------------------
def _plot_scatter(nlln, recn, names, path, max_pts=2000):
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=120)
    K = nlln.size(1)
    for k in range(K):
        x = nlln[:max_pts, k].cpu().numpy(); y = recn[:max_pts, k].cpu().numpy()
        ax.scatter(x, y, s=6, alpha=0.4, label=names[k])
    ax.set_xlabel("NLL_norm (standardized)"); ax.set_ylabel("rec_norm (standardized)")
    ax.set_title("Density vs reconstruction trade-off (per sample/expert)")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def _plot_heatmap(w_all, names, path, max_rows=200):
    fig, ax = plt.subplots(figsize=(5.0, 6.0), dpi=120)
    W = w_all[:max_rows].cpu().numpy()
    im = ax.imshow(W, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_xlabel("expert"); ax.set_ylabel("sample")
    ax.set_title("Gate weights per sample (routing vs averaging)")
    fig.colorbar(im, ax=ax, label="weight")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def _plot_score_hist(score_all, names, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=120)
    K = score_all.size(1)
    for k in range(K):
        ax.hist(score_all[:, k].cpu().numpy(), bins=60, alpha=0.5, label=names[k])
    ax.set_xlabel("hybrid score (lower=better)"); ax.set_ylabel("count")
    ax.set_title("Per-expert score distribution (did standardization equalize?)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def _parse_args():
    p = argparse.ArgumentParser(description="Stage 1.3 reconstruction-aware gate")
    p.add_argument("--ckpt-dirs", nargs="+", required=True)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--rec-z-mode", choices=("fixed_shared", "zero"),
                   default="fixed_shared")
    p.add_argument("--rec-norm", choices=("global", "per_expert"),
                   default="global")
    p.add_argument("--z-bank-size", type=int, default=4)
    p.add_argument("--z-bank-seed", type=int, default=1234)
    p.add_argument("--calib-batches", type=int, default=20)
    p.add_argument("--train-batches", type=int, default=200)
    p.add_argument("--gate-hidden", type=int, default=128)
    p.add_argument("--lr-gate", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-root", default="./CSMF2/experiments/step_1_3/results")
    a = p.parse_args()
    return Stage13Cfg(
        ckpt_dirs=tuple(a.ckpt_dirs), alpha=a.alpha, beta=a.beta, tau=a.tau,
        rec_z_mode=a.rec_z_mode, rec_norm=a.rec_norm,
        z_bank_size=a.z_bank_size,
        z_bank_seed=a.z_bank_seed, calib_batches=a.calib_batches,
        train_batches=a.train_batches, gate_hidden=a.gate_hidden,
        lr_gate=a.lr_gate, epochs=a.epochs, batch_size=a.batch_size,
        seed=a.seed, out_root=a.out_root)


if __name__ == "__main__":
    cfg = _parse_args()
    try:
        run(cfg); sys.exit(0)
    except Exception:
        logger.error("STEP-1_3 run FAILED\n%s", traceback.format_exc())
        sys.exit(1)
