# =============================================================================
# RESREF-R2 v0.1 -- experiments.step_1_4b.resref_r2
# Purpose: R2 = per-sample escalation of the PROMOTED R1 baseline (Option B).
#          Each NSF posterior sample gets its OWN correction Dx_i, instead of
#          R1's single shared field. This is a RISK/BENEFIT TRADE-OFF EXPERIMENT,
#          not a presumed upgrade: the supervised target pulls every sample
#          toward the same x_true, which structurally wants to COLLAPSE the
#          posterior. A spread-retention penalty fights that; the lambda_spread
#          SWEEP measures whether per-sample correction can beat R1 WITHOUT
#          damaging uncertainty.
# CONVENTION: no fallback / mock / pass. Bad input / failed pre-flight / non-
#             finite -> logger.error + raise. Reuses R1's adjoint, decode,
#             metrics, head arch (single source of truth -> from .resref_r1).
#
# DELTAS FROM R1 (the only differences):
#   inputs : PER-SAMPLE [y_up, x_sample_i, A^T r0_i], r0_i = y - A(x_sample_i)
#            (L4 mean-only lock is RELAXED on purpose -- that IS Option B)
#   output : x_final_i = x_sample_i + Dx_i           (per-sample, not shared)
#   loss   : MSE(Dx_i, x_true - x_sample_i)  +  lambda_spread * spread_penalty
#            spread_penalty = relu(Var_i[x_sample] - Var_i[x_final]).mean()
#            (asymmetric: punishes spread SHRINKAGE only, not growth)
#   L3     : FLIPS from an exact invariant (R1) to a MEASURED GATE --
#            spread_retention = Var_after / Var_before  must be >= tau (0.85)
#   PROMOTE: must beat R1 (not NSF) on fwd_rel/PSNR/SSIM AND keep spread >= tau,
#            over 3 seeds. (Calibration/coverage deferred to the next patch.)
#
# NOT in v0.1 (deferred, by decision): calibration/coverage metric. First prove
#   R2 can beat R1 while preserving spread; add coverage as the next patch.
# Changelog (NEW in v0.1):
#   * Introduced. Per-sample correction head; spread-penalised supervised loss;
#     spread-retention gate; lambda_spread sweep; promotion-vs-R1. Reuses R1
#     machinery wholesale.
# Update summary:
#   v0.1 is the R2-lite trade-off probe. A flat/failed sweep (no lambda beats R1
#   without collapse) is itself a result: it says R1 (shared correction) was the
#   right stopping point for this cell.
# =============================================================================
from __future__ import annotations
import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

logger = logging.getLogger("CSMF2.step_1_4b.resref_r2")
__version__ = "0.1"
__abbr__ = "RESREF-R2"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ...data.degrade import MNISTDegraded
# single source of truth: reuse R1's verified components
from .resref_r1 import (A_forward, A_adjoint, adjoint_preflight, R1Head,
                        _z_bank, nsf_mean_and_samples, _psnr, _ssim,
                        _gaussian_window, _EPS)
from ..step_1_1_1_1.model_io import build_from_report


def _features_persample(y, samples, blur_sigma, scale, image_hw):
    # y (B,1,M,M); samples (B,K,1,H,W) -> feat (B*K,3,H,W), per-sample inputs.
    B, K = samples.shape[:2]
    H, W = image_hw
    M = y.shape[-1]
    x_flat = samples.reshape(B * K, 1, H, W)
    y_rep = y.unsqueeze(1).expand(B, K, 1, M, M).reshape(B * K, 1, M, M)
    y_up = F.interpolate(y_rep, size=image_hw, mode="nearest")
    r0 = y_rep - A_forward(x_flat, blur_sigma, scale)        # per-sample residual
    atr0 = A_adjoint(r0, blur_sigma, scale, image_hw)
    feat = torch.cat([y_up, x_flat, atr0], dim=1)            # (B*K,3,H,W)
    return feat, B, K


def run(ckpt_dir: str, *, lambda_spread: float, tau_spread: float,
        r1_report: str | None, epochs: int, batch_size: int, lr: float,
        hidden: int, alpha_init: float, z_mode: str, z_bank_size: int,
        z_bank_seed: int, seed: int, n_val_batches: int, out_root: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    nsf, _cond, ref = build_from_report(ckpt_dir, device)
    if ref.expert != "nsf":
        logger.error("[RESREF-R2] expected NSF checkpoint, got %r", ref.expert)
        raise ValueError(f"R2 expects NSF, got {ref.expert!r}")
    nsf.eval()
    for p in nsf.parameters():
        p.requires_grad_(False)
    nsf = nsf.double()                # f64 NSF -> stable RQ-spline inverse
    blur_sigma, scale, noise_sigma = ref.blur_sigma, ref.scale, ref.noise_sigma
    dim = ref.dim
    image_hw = (28, 28)
    work_dtype = torch.float32

    adj_rel = adjoint_preflight(blur_sigma, scale, image_hw, device)  # L1
    z_bank = _z_bank(dim, z_bank_size, z_mode, z_bank_seed, device, torch.float64)

    r2 = R1Head(hidden=hidden, alpha_init=alpha_init).to(device).to(work_dtype)
    opt = torch.optim.Adam(r2.parameters(), lr=lr)

    tr = DataLoader(MNISTDegraded(ref.data_root, split="train", sigma=blur_sigma,
                                  scale=scale, noise_sigma=noise_sigma),
                    batch_size=batch_size, shuffle=True)
    va = DataLoader(MNISTDegraded(ref.data_root, split="val", sigma=blur_sigma,
                                  scale=scale, noise_sigma=noise_sigma),
                    batch_size=batch_size, shuffle=False)

    loss_hist: list[float] = []; mse_hist: list[float] = []; spr_hist: list[float] = []
    for epoch in range(epochs):
        r2.train(); rl = ml = sl = 0.0; nb = 0
        for x_img, y in tr:
            x_img = x_img.to(device).to(work_dtype); y = y.to(device).to(work_dtype)
            _, samples = nsf_mean_and_samples(nsf, y, z_bank, image_hw, work_dtype)
            feat, B, K = _features_persample(y, samples, blur_sigma, scale, image_hw)
            dx = (r2(feat)).reshape(B, K, 1, *image_hw)      # per-sample Dx
            x_final = samples + dx
            target = x_img.unsqueeze(1) - samples            # (B,K,1,H,W)
            mse = F.mse_loss(dx, target)
            v_before = samples.var(dim=1, unbiased=False)
            v_after = x_final.var(dim=1, unbiased=False)
            spread_pen = F.relu(v_before - v_after).mean()   # punish shrinkage only
            loss = mse + lambda_spread * spread_pen
            if not torch.isfinite(loss):
                logger.error("[RESREF-R2] non-finite loss epoch=%d", epoch)
                raise RuntimeError("non-finite loss")
            opt.zero_grad(); loss.backward(); opt.step()
            rl += float(loss.item()); ml += float(mse.item())
            sl += float(spread_pen.item()); nb += 1
        loss_hist.append(rl / max(nb, 1)); mse_hist.append(ml / max(nb, 1))
        spr_hist.append(sl / max(nb, 1))
        logger.info("[RESREF-R2] epoch %d  loss=%.6f mse=%.6f spread_pen=%.3e "
                    "alpha=%.4f", epoch, loss_hist[-1], mse_hist[-1],
                    spr_hist[-1], float(r2.alpha.detach()))

    # ---- eval + gates (val) -------------------------------------------------
    r2.eval()
    fr_nsf: list[float] = []; fr_r2: list[float] = []
    corr_e: list[float] = []; budget_e: list[float] = []
    psnr_nsf: list[float] = []; psnr_r2: list[float] = []
    ssim_nsf: list[float] = []; ssim_r2: list[float] = []
    vb_sum = 0.0; va_sum = 0.0; n_seen = 0
    ssim_win = _gaussian_window(11, 1.5, device, work_dtype)
    panel = None
    with torch.no_grad():
        for bi, (x_img, y) in enumerate(va):
            if bi >= n_val_batches:
                break
            x_img = x_img.to(device).to(work_dtype); y = y.to(device).to(work_dtype)
            x0_mean, samples = nsf_mean_and_samples(nsf, y, z_bank, image_hw, work_dtype)
            feat, B, K = _features_persample(y, samples, blur_sigma, scale, image_hw)
            dx = (r2(feat)).reshape(B, K, 1, *image_hw)
            x_final = samples + dx
            x_final_mean = x_final.mean(dim=1)               # point estimate (vs R1)

            yn = y.flatten(1).norm(dim=1).clamp_min(_EPS)
            fr_nsf += (A_forward(x0_mean, blur_sigma, scale).sub(y)
                       .flatten(1).norm(dim=1) / yn).tolist()
            fr_r2 += (A_forward(x_final_mean, blur_sigma, scale).sub(y)
                      .flatten(1).norm(dim=1) / yn).tolist()

            xt = x_img.clamp(0, 1)
            x0c = x0_mean.clamp(0, 1); xfc = x_final_mean.clamp(0, 1)
            psnr_nsf += _psnr(x0c, xt).tolist();  psnr_r2 += _psnr(xfc, xt).tolist()
            ssim_nsf += _ssim(x0c, xt, ssim_win).tolist()
            ssim_r2 += _ssim(xfc, xt, ssim_win).tolist()

            # L2 budget (per-sample, image space)
            corr_e += dx.flatten(2).pow(2).sum(2).mean(1).tolist()
            budget_e += (x_img.unsqueeze(1) - samples).flatten(2).pow(2).sum(2).mean(1).tolist()

            # spread retention (the measured L3 gate)
            vb = samples.var(dim=1, unbiased=False)
            vaf = x_final.var(dim=1, unbiased=False)
            vb_sum += float(vb.sum()); va_sum += float(vaf.sum())
            n_seen += vb.numel()

            if panel is None:
                panel = (y[:6].cpu(), x0_mean[:6].cpu(),
                         x_final_mean[:6].cpu(), x_img[:6].cpu())

    mean_fr_nsf = float(sum(fr_nsf) / len(fr_nsf))
    mean_fr_r2 = float(sum(fr_r2) / len(fr_r2))
    m_psnr_nsf = float(sum(psnr_nsf) / len(psnr_nsf))
    m_psnr_r2 = float(sum(psnr_r2) / len(psnr_r2))
    m_ssim_nsf = float(sum(ssim_nsf) / len(ssim_nsf))
    m_ssim_r2 = float(sum(ssim_r2) / len(ssim_r2))
    mean_corr = float(sum(corr_e) / len(corr_e))
    mean_budget = float(sum(budget_e) / len(budget_e))
    budget_ratio = mean_corr / (mean_budget + _EPS)
    spread_retention = (va_sum / n_seen) / ((vb_sum / n_seen) + _EPS)
    spread_pass = bool(spread_retention >= tau_spread)

    # ---- promotion vs R1 (if R1 report supplied) ---------------------------
    r1_cmp = None
    if r1_report:
        rp = json.loads(Path(r1_report).read_text())
        r1_fr, r1_ps, r1_ss = rp["fwd_rel_r1"], rp["psnr_r1"], rp["ssim_r1"]
        beats = bool(mean_fr_r2 < r1_fr and m_psnr_r2 > r1_ps and m_ssim_r2 > r1_ss)
        r1_cmp = {
            "r1_fwd_rel": r1_fr, "r1_psnr": r1_ps, "r1_ssim": r1_ss,
            "delta_fwd_rel_vs_r1": r1_fr - mean_fr_r2,   # +ve = R2 better
            "delta_psnr_vs_r1": m_psnr_r2 - r1_ps,
            "delta_ssim_vs_r1": m_ssim_r2 - r1_ss,
            "beats_r1_all_metrics": beats,
        }

    report = {
        "abbr": __abbr__, "version": __version__, "seed": seed,
        "lambda_spread": lambda_spread, "tau_spread": tau_spread,
        "ckpt_dir": ckpt_dir,
        "cell": {"scale": scale, "blur_sigma": blur_sigma, "noise_sigma": noise_sigma},
        "L1_adjoint_rel_err": adj_rel,
        "fwd_rel_nsf": mean_fr_nsf, "fwd_rel_r2": mean_fr_r2,
        "psnr_nsf": m_psnr_nsf, "psnr_r2": m_psnr_r2,
        "ssim_nsf": m_ssim_nsf, "ssim_r2": m_ssim_r2,
        "L2_budget_ratio": budget_ratio, "L2_within_budget": bool(budget_ratio <= 1.0),
        "spread_retention": spread_retention,
        "spread_gate_tau": tau_spread, "spread_gate_pass": spread_pass,
        "alpha_final": float(r2.alpha.detach()),
        "vs_r1": r1_cmp,
        "loss_hist": loss_hist, "mse_hist": mse_hist, "spread_pen_hist": spr_hist,
        "note": ("R2-lite trade-off run; promotion needs 3 seeds, beats R1 on "
                 "all metrics, AND spread_retention >= tau. Calibration deferred."),
    }

    out_dir = (Path(out_root) /
               f"resref_r2_s{scale}_n{noise_sigma:.2f}_lam{lambda_spread}_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resref_r2_report.json").write_text(json.dumps(report, indent=2))
    txt = _format_txt(report)
    (out_dir / "resref_r2_report.txt").write_text(txt)
    print(txt)
    if panel is not None:
        _panel_plot(out_dir, panel)
    torch.save(r2.state_dict(), out_dir / "r2_head.pt")
    logger.info("[RESREF-R2] wrote report + r2_head.pt -> %s", out_dir)
    return report


def _format_txt(r: dict) -> str:
    c = r["cell"]
    L = ["=" * 78,
         f"RESREF-R2 v{r['version']} -- per-sample (Option B) -- "
         f"lambda_spread={r['lambda_spread']} seed {r['seed']}",
         "=" * 78,
         f"  cell: scale={c['scale']} blur={c['blur_sigma']:.2f} "
         f"noise={c['noise_sigma']:.2f}   tau_spread={r['tau_spread']}",
         "",
         f"  L1 adjoint rel_err          = {r['L1_adjoint_rel_err']:.3e}  (pre-flight PASS)",
         "",
         f"  fwd_rel  NSF / R2           = {r['fwd_rel_nsf']:.4f} / {r['fwd_rel_r2']:.4f}",
         f"  PSNR     NSF / R2           = {r['psnr_nsf']:.3f} / {r['psnr_r2']:.3f} dB",
         f"  SSIM     NSF / R2           = {r['ssim_nsf']:.4f} / {r['ssim_r2']:.4f}",
         "",
         f"  L2 budget ratio             = {r['L2_budget_ratio']:.4f}"
         f"   within_budget={r['L2_within_budget']}",
         f"  SPREAD retention (Va/Vb)    = {r['spread_retention']:.4f}"
         f"   (tau {r['spread_gate_tau']})  PASS={r['spread_gate_pass']}",
         f"  alpha (final)               = {r['alpha_final']:.4f}"]
    if r["vs_r1"]:
        v = r["vs_r1"]
        L += ["",
              "  vs R1 (promotion target):",
              f"    d_fwd_rel = {v['delta_fwd_rel_vs_r1']:+.4f}   "
              f"d_PSNR = {v['delta_psnr_vs_r1']:+.3f} dB   "
              f"d_SSIM = {v['delta_ssim_vs_r1']:+.4f}",
              f"    beats_R1_all_metrics = {v['beats_r1_all_metrics']}   "
              f"spread_pass = {r['spread_gate_pass']}",
              f"    -> PROMOTE-candidate = "
              f"{v['beats_r1_all_metrics'] and r['spread_gate_pass']} "
              f"(needs 3 seeds)"]
    L += ["",
          "  NOTE: R2-lite trade-off. Promotion = beats R1 (all metrics) AND",
          "        spread_retention >= tau, over 3 seeds. Calibration deferred.",
          "=" * 78]
    return "\n".join(L)


def _panel_plot(out_dir: Path, panel) -> None:
    y, x0, xf, xt = panel
    n = y.size(0)
    titles = ["y (obs)", "NSF x0_mean", "R2 x_final", "x_true"]
    cols = [F.interpolate(y, size=(28, 28), mode="nearest"), x0, xf, xt]
    fig, axes = plt.subplots(n, 4, figsize=(5.5, 1.4 * n))
    for i in range(n):
        for j, col in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(col[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "p1_before_after_panel.png", dpi=120)
    plt.close(fig)


def _parse_args():
    p = argparse.ArgumentParser(description="RESREF-R2: per-sample correction (Option B)")
    p.add_argument("--ckpt-dir", required=True, help="frozen NSF result dir")
    p.add_argument("--lambda-spread", type=float, required=True,
                   help="spread-retention penalty weight; sweep {0,0.1,1.0,10.0}")
    p.add_argument("--tau-spread", type=float, default=0.85,
                   help="spread-retention gate (Var_after/Var_before >= tau)")
    p.add_argument("--r1-report", default=None,
                   help="path to R1 resref_r1_report.json for promotion-vs-R1")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--alpha-init", type=float, default=0.1)
    p.add_argument("--z-mode", choices=("fixed_shared", "zero"), default="fixed_shared")
    p.add_argument("--z-bank-size", type=int, default=4)
    p.add_argument("--z-bank-seed", type=int, default=1234)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-val-batches", type=int, default=8)
    p.add_argument("--out-root", default="./CSMF2/experiments/step_1_4b/results")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    a = _parse_args()
    try:
        run(a.ckpt_dir, lambda_spread=a.lambda_spread, tau_spread=a.tau_spread,
            r1_report=a.r1_report, epochs=a.epochs, batch_size=a.batch_size,
            lr=a.lr, hidden=a.hidden, alpha_init=a.alpha_init, z_mode=a.z_mode,
            z_bank_size=a.z_bank_size, z_bank_seed=a.z_bank_seed, seed=a.seed,
            n_val_batches=a.n_val_batches, out_root=a.out_root)
    except Exception:
        logger.error("RESREF-R2 run FAILED\n%s", traceback.format_exc())
        sys.exit(1)
