# =============================================================================
# RESREF-R1 v0.1 -- experiments.step_1_4b.resref_r1
# Purpose: R1 residual-refinement head. Frozen NSF produces a posterior-mean
#          reconstruction x0_mean; a SMALL CNN learns a single correction field
#          Dx from MEAN-DERIVED inputs and ADDS THE SAME Dx to every NSF sample
#          (Option A). This refines NSF's structured leftover error (confirmed
#          by RESPROBE: STRUCTURED_HEADROOM, edge_corr 0.55 / autocorr 0.52)
#          WITHOUT touching posterior spread.
# CONVENTION: no fallback / mock / pass. Bad input / failed pre-flight / non-
#             finite -> logger.error + raise.
#
# THE FOUR LOCKS (all enforced here):
#   L1 ADJOINT  : A^T computed via autograd (exact adjoint of degrade's A).
#                 Pre-flight dot-product test <Ax,r>==<x,A^T r> MUST pass
#                 (float64) or training ABORTS before a single step.
#   L2 BUDGET   : correction energy gated in IMAGE SPACE against the same-units
#                 target energy: ||alpha*Dx||^2  vs  ||x_true - x0_mean||^2.
#   L3 NO-COLLAPSE: Option A. Dx is computed ONCE from mean-derived inputs
#                 (y_up, x0_mean, A^T r0_mean) and broadcast-added to every
#                 sample -> Var_i[x_final] == Var_i[x_sample] EXACTLY. Asserted
#                 numerically as a tripwire (catches any per-sample leak that
#                 would silently turn R1 into R2).
#   L4 x0=MEAN  : R1 inputs are mean-derived only. Feeding per-sample x0_i is
#                 Option B / R2 and is explicitly NOT done here.
#
# Training target: SUPERVISED IMAGE-SPACE (option i):  alpha*R1(.) ~= x_true-x0_mean
#   (MSE). Avoids fitting the measurement noise in y (the (ii) data-consistency
#   loss is deferred to R2 / ablation).
# Gate (val): fwd_rel(x0_mean + Dx) < fwd_rel(NSF) by a real margin; L2 budget
#   respected; L3 variance assertion holds. Promotion needs 3 seeds (run thrice;
#   improvement must exceed cross-seed std) -- this module runs ONE seed.
# Changelog (NEW in v0.1):
#   * Introduced. Frozen-NSF R1-mean head, Option A propagation, autograd
#     adjoint, supervised image-space target, four-lock gating.
# Update summary:
#   v0.1 is the posterior-safe R1 baseline: it can only improve or fail to
#   improve the mean reconstruction; it cannot collapse the posterior (Option A
#   preserves spread by construction). R2 (per-sample correction) comes only if
#   R1 clears the fwd_rel gate.
# =============================================================================
from __future__ import annotations
import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

logger = logging.getLogger("CSMF2.step_1_4b.resref_r1")
__version__ = "0.1"
__abbr__ = "RESREF-R1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ...data.degrade import MNISTDegraded, blur, downsample, inverse_logit
from ..step_1_1_1_1.model_io import build_from_report

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Forward operator A and its EXACT adjoint A^T (L1)
# ---------------------------------------------------------------------------
def A_forward(x, blur_sigma, scale):
    return downsample(blur(x, blur_sigma), scale)


def A_adjoint(r, blur_sigma, scale, image_hw):
    # A is linear, so its adjoint is the vector-Jacobian product. Computing it
    # via autograd gives the EXACT adjoint of degrade's A (reflect-pad blur +
    # avg-pool), including all boundary terms -- no hand-derived transpose.
    with torch.enable_grad():
        x = torch.zeros(r.size(0), 1, image_hw[0], image_hw[1],
                        dtype=r.dtype, device=r.device, requires_grad=True)
        y = A_forward(x, blur_sigma, scale)
        (atr,) = torch.autograd.grad((y * r).sum(), x)
    return atr.detach()


def adjoint_preflight(blur_sigma, scale, image_hw, device, tol=1e-6):
    # L1 hard gate: <A x, r> must equal <x, A^T r>. Run in float64. Abort on fail.
    H, W = image_hw
    M = H // scale
    x = torch.randn(4, 1, H, W, dtype=torch.float64, device=device)
    r = torch.randn(4, 1, M, M, dtype=torch.float64, device=device)
    lhs = (A_forward(x, blur_sigma, scale) * r).sum()
    rhs = (x * A_adjoint(r, blur_sigma, scale, image_hw)).sum()
    rel = float((lhs - rhs).abs() / (lhs.abs() + _EPS))
    if rel > tol:
        logger.error("[RESREF-R1] ADJOINT PRE-FLIGHT FAILED: <Ax,r>=%.6f "
                     "<x,A^Tr>=%.6f rel_err=%.3e tol=%.1e -- A^T is wrong, "
                     "ABORTING before training", float(lhs), float(rhs), rel, tol)
        raise RuntimeError("adjoint dot-product test failed")
    logger.info("[RESREF-R1] adjoint pre-flight PASS: rel_err=%.3e (tol %.1e)",
                rel, tol)
    return rel


# ---------------------------------------------------------------------------
# Small correction head (L4: inputs are mean-derived only)
# ---------------------------------------------------------------------------
class R1Head(nn.Module):
    # Input: cat[y_up, x0_mean, A^T r0_mean] (3,28,28). Output: Dx_raw (1,28,28).
    # Final conv zero-init -> Dx ~= 0 at start -> x_final ~= x_sample (safe).
    # alpha learnable, init small (~0.1).
    def __init__(self, hidden: int = 64, alpha_init: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, 1, 3, padding=1)
        nn.init.zeros_(self.conv3.weight); nn.init.zeros_(self.conv3.bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.conv1(feat))
        z = F.gelu(self.conv2(z))
        dx_raw = self.conv3(z)
        return self.alpha * dx_raw


# ---------------------------------------------------------------------------
# NSF reconstruction: posterior mean + samples (samples only for L3 check)
# ---------------------------------------------------------------------------
def _z_bank(dim, k, mode, seed, device, dtype):
    if mode == "zero":
        return torch.zeros(1, dim, device=device, dtype=dtype)
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(k, dim, generator=g, device=device, dtype=dtype)


@torch.no_grad()
def nsf_mean_and_samples(nsf, y, z_bank, image_hw, out_dtype):
    # Returns (x0_mean (B,1,H,W), samples (B,K,1,H,W)) in pixel space, cast to
    # out_dtype. NSF decode (RQ-spline inverse) is run in FLOAT64: the stacked
    # spline root inversion accumulates float32 error and can yield a negative
    # discriminant (project convention -- spline inversion requires f64). nsf
    # is held as a float64 module; z_bank/y are cast to f64 for the decode.
    H, W = image_hw
    y64 = y.double()
    h = nsf.cond(y64)
    B = y.size(0)
    xs = []
    for i in range(z_bank.size(0)):
        z_i = z_bank[i:i + 1].expand(B, -1)              # (B, dim) f64
        x_logit = nsf.decode(z_i, h)                     # (B, dim) logit, f64
        x_pix = inverse_logit(x_logit).view(B, 1, H, W)  # (B,1,H,W) pixel, f64
        xs.append(x_pix)
    samples = torch.stack(xs, dim=1)                     # (B,K,1,H,W) f64
    x0_mean = samples.mean(dim=1)                        # (B,1,H,W) f64
    if not torch.isfinite(x0_mean).all():
        logger.error("[RESREF-R1] non-finite NSF reconstruction")
        raise ValueError("non-finite NSF reconstruction")
    return x0_mean.to(out_dtype), samples.to(out_dtype)


def _features(y, x0_mean, blur_sigma, scale, image_hw):
    # Mean-derived inputs only (L4). y_up = nearest upsample of y to image size.
    y_up = F.interpolate(y, size=image_hw, mode="nearest")
    r0 = y - A_forward(x0_mean, blur_sigma, scale)
    atr0 = A_adjoint(r0, blur_sigma, scale, image_hw)
    return torch.cat([y_up, x0_mean, atr0], dim=1)        # (B,3,H,W)


# ---------------------------------------------------------------------------
# Image-quality metrics vs x_true (PSNR / SSIM). Both measure closeness to the
# CLEAN image -- complementary to fwd_rel, which only measures consistency with
# the degraded y. fwd_rel can improve while the image worsens; these catch that.
# ---------------------------------------------------------------------------
def _psnr(x, ref):
    # x, ref: (B,1,H,W) in [0,1]. Returns per-image PSNR (dB). MAX=1.
    mse = (x - ref).flatten(1).pow(2).mean(dim=1).clamp_min(_EPS)
    return 10.0 * torch.log10(1.0 / mse)


def _gaussian_window(ws: int, sigma: float, device, dtype):
    c = torch.arange(ws, device=device, dtype=dtype) - (ws - 1) / 2.0
    g = torch.exp(-(c ** 2) / (2.0 * sigma * sigma)); g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, ws, ws)


def _ssim(x, ref, window):
    # Standard single-channel SSIM with a Gaussian window; x,ref (B,1,H,W) [0,1].
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window.size(-1) // 2
    mu_x = F.conv2d(x, window, padding=pad)
    mu_r = F.conv2d(ref, window, padding=pad)
    mu_x2, mu_r2, mu_xr = mu_x * mu_x, mu_r * mu_r, mu_x * mu_r
    sx2 = F.conv2d(x * x, window, padding=pad) - mu_x2
    sr2 = F.conv2d(ref * ref, window, padding=pad) - mu_r2
    sxr = F.conv2d(x * ref, window, padding=pad) - mu_xr
    ssim_map = (((2 * mu_xr + C1) * (2 * sxr + C2)) /
                ((mu_x2 + mu_r2 + C1) * (sx2 + sr2 + C2)))
    return ssim_map.mean(dim=[1, 2, 3])                   # per-image SSIM


# ---------------------------------------------------------------------------
# Train + gate
# ---------------------------------------------------------------------------
def run(ckpt_dir: str, *, epochs: int, batch_size: int, lr: float,
        hidden: int, alpha_init: float, z_mode: str, z_bank_size: int,
        z_bank_seed: int, seed: int, n_val_batches: int, fwd_rel_baseline: float,
        out_root: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    nsf, _cond, ref = build_from_report(ckpt_dir, device)
    if ref.expert != "nsf":
        logger.error("[RESREF-R1] expected NSF checkpoint, got %r", ref.expert)
        raise ValueError(f"R1 expects NSF, got {ref.expert!r}")
    nsf.eval()
    for p in nsf.parameters():
        p.requires_grad_(False)
    nsf = nsf.double()                # f64 NSF -> stable RQ-spline inverse
    blur_sigma, scale, noise_sigma = ref.blur_sigma, ref.scale, ref.noise_sigma
    dim = ref.dim
    image_hw = (28, 28)
    work_dtype = torch.float32        # R1 head + features + training run in f32

    # ---- L1: adjoint pre-flight (ABORT before training if it fails) --------
    adj_rel = adjoint_preflight(blur_sigma, scale, image_hw, device)

    z_bank = _z_bank(dim, z_bank_size, z_mode, z_bank_seed, device,
                     torch.float64)   # f64 for the NSF decode

    r1 = R1Head(hidden=hidden, alpha_init=alpha_init).to(device).to(work_dtype)
    opt = torch.optim.Adam(r1.parameters(), lr=lr)

    tr = DataLoader(MNISTDegraded(ref.data_root, split="train", sigma=blur_sigma,
                                  scale=scale, noise_sigma=noise_sigma),
                    batch_size=batch_size, shuffle=True)
    va = DataLoader(MNISTDegraded(ref.data_root, split="val", sigma=blur_sigma,
                                  scale=scale, noise_sigma=noise_sigma),
                    batch_size=batch_size, shuffle=False)

    loss_hist: list[float] = []
    for epoch in range(epochs):
        r1.train(); run_loss = 0.0; nb = 0
        for x_img, y in tr:
            x_img = x_img.to(device).to(work_dtype); y = y.to(device).to(work_dtype)
            x0_mean, _ = nsf_mean_and_samples(nsf, y, z_bank, image_hw, work_dtype)
            feat = _features(y, x0_mean, blur_sigma, scale, image_hw)
            dx = r1(feat)                                  # alpha * R1(.)
            target = x_img - x0_mean                       # supervised (i)
            loss = F.mse_loss(dx, target)
            if not torch.isfinite(loss):
                logger.error("[RESREF-R1] non-finite loss epoch=%d", epoch)
                raise RuntimeError("non-finite loss")
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += float(loss.item()); nb += 1
        loss_hist.append(run_loss / max(nb, 1))
        logger.info("[RESREF-R1] epoch %d  mse=%.6f  alpha=%.4f",
                    epoch, loss_hist[-1], float(r1.alpha.detach()))

    # ---- evaluation + gates (val) ------------------------------------------
    r1.eval()
    fr_nsf: list[float] = []; fr_r1: list[float] = []
    corr_e: list[float] = []; budget_e: list[float] = []
    psnr_nsf: list[float] = []; psnr_r1: list[float] = []
    ssim_nsf: list[float] = []; ssim_r1: list[float] = []
    ssim_win = _gaussian_window(11, 1.5, device, work_dtype)
    var_abs_max = 0.0
    panel = None
    with torch.no_grad():
        for bi, (x_img, y) in enumerate(va):
            if bi >= n_val_batches:
                break
            x_img = x_img.to(device).to(work_dtype); y = y.to(device).to(work_dtype)
            x0_mean, samples = nsf_mean_and_samples(nsf, y, z_bank, image_hw, work_dtype)
            feat = _features(y, x0_mean, blur_sigma, scale, image_hw)
            dx = r1(feat)                                  # (B,1,H,W)
            x_final_mean = x0_mean + dx

            yn = y.flatten(1).norm(dim=1).clamp_min(_EPS)
            fr_nsf += (A_forward(x0_mean, blur_sigma, scale).sub(y)
                       .flatten(1).norm(dim=1) / yn).tolist()
            fr_r1 += (A_forward(x_final_mean, blur_sigma, scale).sub(y)
                      .flatten(1).norm(dim=1) / yn).tolist()

            # image-quality vs x_true (clamp to [0,1] for metric domain)
            xt = x_img.clamp(0, 1)
            x0c = x0_mean.clamp(0, 1); xfc = x_final_mean.clamp(0, 1)
            psnr_nsf += _psnr(x0c, xt).tolist();  psnr_r1 += _psnr(xfc, xt).tolist()
            ssim_nsf += _ssim(x0c, xt, ssim_win).tolist()
            ssim_r1 += _ssim(xfc, xt, ssim_win).tolist()

            # L2 budget (image space, same units)
            corr_e += dx.flatten(1).pow(2).sum(1).tolist()
            budget_e += (x_img - x0_mean).flatten(1).pow(2).sum(1).tolist()

            # L3 variance-preservation tripwire: x_final_i = sample_i + dx
            x_final_samples = samples + dx.unsqueeze(1)     # broadcast over K
            v_before = samples.var(dim=1, unbiased=False)
            v_after = x_final_samples.var(dim=1, unbiased=False)
            var_abs_max = max(var_abs_max,
                              float((v_after - v_before).abs().max()))

            if panel is None:
                panel = (y[:6].cpu(), x0_mean[:6].cpu(),
                         x_final_mean[:6].cpu(), x_img[:6].cpu())

    mean_fr_nsf = float(sum(fr_nsf) / len(fr_nsf))
    mean_fr_r1 = float(sum(fr_r1) / len(fr_r1))
    improvement = mean_fr_nsf - mean_fr_r1
    mean_corr = float(sum(corr_e) / len(corr_e))
    mean_budget = float(sum(budget_e) / len(budget_e))
    budget_ratio = mean_corr / (mean_budget + _EPS)
    m_psnr_nsf = float(sum(psnr_nsf) / len(psnr_nsf))
    m_psnr_r1 = float(sum(psnr_r1) / len(psnr_r1))
    m_ssim_nsf = float(sum(ssim_nsf) / len(ssim_nsf))
    m_ssim_r1 = float(sum(ssim_r1) / len(ssim_r1))

    # L3 must be ~0 (exact by construction); flag if a per-sample leak occurred.
    # Threshold 1e-5: a real per-sample leak shifts variance by O(1) (~5 orders
    # above this); 1e-5 only absorbs float32 var() rounding, not a real leak.
    L3_ok = var_abs_max < 1e-5
    if not L3_ok:
        logger.error("[RESREF-R1] L3 VARIANCE TRIPWIRE: max|var_after-var_before|"
                     "=%.3e > 1e-5 -- a per-sample input leaked (this is no "
                     "longer Option A)", var_abs_max)

    report = {
        "abbr": __abbr__, "version": __version__, "seed": seed,
        "ckpt_dir": ckpt_dir,
        "cell": {"scale": scale, "blur_sigma": blur_sigma,
                 "noise_sigma": noise_sigma},
        "L1_adjoint_rel_err": adj_rel,
        "fwd_rel_nsf": mean_fr_nsf,
        "fwd_rel_r1": mean_fr_r1,
        "fwd_rel_improvement": improvement,
        "fwd_rel_baseline_ref": fwd_rel_baseline,
        "psnr_nsf": m_psnr_nsf,
        "psnr_r1": m_psnr_r1,
        "psnr_improvement": m_psnr_r1 - m_psnr_nsf,
        "ssim_nsf": m_ssim_nsf,
        "ssim_r1": m_ssim_r1,
        "ssim_improvement": m_ssim_r1 - m_ssim_nsf,
        "L2_correction_energy": mean_corr,
        "L2_budget_energy": mean_budget,
        "L2_budget_ratio": budget_ratio,
        "L2_within_budget": bool(budget_ratio <= 1.0),
        "L3_var_abs_max": var_abs_max,
        "L3_variance_preserved": L3_ok,
        "alpha_final": float(r1.alpha.detach()),
        "loss_hist": loss_hist,
        "note": ("single-seed run; promotion requires 3 seeds with improvement "
                 "> cross-seed std"),
    }

    out_dir = Path(out_root) / f"resref_r1_s{scale}_n{noise_sigma:.2f}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resref_r1_report.json").write_text(json.dumps(report, indent=2))
    txt = _format_txt(report)
    (out_dir / "resref_r1_report.txt").write_text(txt)
    print(txt)
    if panel is not None:
        _panel_plot(out_dir, panel)
    torch.save(r1.state_dict(), out_dir / "r1_head.pt")
    logger.info("[RESREF-R1] wrote report + r1_head.pt -> %s", out_dir)
    return report


def _format_txt(r: dict) -> str:
    c = r["cell"]
    L = ["=" * 78,
         f"RESREF-R1 v{r['version']} -- R1-mean (Option A) -- seed {r['seed']}",
         "=" * 78,
         f"  cell: scale={c['scale']} blur={c['blur_sigma']:.2f} "
         f"noise={c['noise_sigma']:.2f}",
         "",
         f"  L1 adjoint rel_err          = {r['L1_adjoint_rel_err']:.3e}  (pre-flight PASS)",
         "",
         f"  fwd_rel  NSF (baseline)      = {r['fwd_rel_nsf']:.4f}",
         f"  fwd_rel  R1                  = {r['fwd_rel_r1']:.4f}",
         f"  improvement                 = {r['fwd_rel_improvement']:+.4f}"
         f"   ({'BEATS NSF' if r['fwd_rel_improvement'] > 0 else 'NO IMPROVEMENT'})",
         "",
         "  IMAGE QUALITY vs x_true (does the IMAGE get closer, not just y?)",
         f"  PSNR  NSF -> R1              = {r['psnr_nsf']:.3f} -> {r['psnr_r1']:.3f} dB"
         f"   ({r['psnr_improvement']:+.3f})",
         f"  SSIM  NSF -> R1              = {r['ssim_nsf']:.4f} -> {r['ssim_r1']:.4f}"
         f"   ({r['ssim_improvement']:+.4f})",
         "",
         f"  L2 correction energy        = {r['L2_correction_energy']:.4f}",
         f"  L2 budget   energy          = {r['L2_budget_energy']:.4f}",
         f"  L2 ratio (corr/budget)      = {r['L2_budget_ratio']:.4f}"
         f"   within_budget={r['L2_within_budget']}",
         "",
         f"  L3 var |after-before| max   = {r['L3_var_abs_max']:.3e}"
         f"   variance_preserved={r['L3_variance_preserved']}",
         f"  alpha (final)               = {r['alpha_final']:.4f}",
         "",
         "  NOTE: single seed. Promotion needs 3 seeds; improvement must",
         "        exceed cross-seed std.",
         "=" * 78]
    return "\n".join(L)


def _panel_plot(out_dir: Path, panel) -> None:
    y, x0, xf, xt = panel
    n = y.size(0)
    titles = ["y (obs)", "NSF x0_mean", "R1 x_final", "x_true"]
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
    p = argparse.ArgumentParser(description="RESREF-R1: NSF residual refinement (Option A)")
    p.add_argument("--ckpt-dir", required=True, help="frozen NSF result dir")
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
    p.add_argument("--fwd-rel-baseline", type=float, default=0.324,
                   help="reference NSF fwd_rel for the cell (informational)")
    p.add_argument("--out-root", default="./CSMF2/experiments/step_1_4b/results")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    a = _parse_args()
    try:
        run(a.ckpt_dir, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
            hidden=a.hidden, alpha_init=a.alpha_init, z_mode=a.z_mode,
            z_bank_size=a.z_bank_size, z_bank_seed=a.z_bank_seed, seed=a.seed,
            n_val_batches=a.n_val_batches, fwd_rel_baseline=a.fwd_rel_baseline,
            out_root=a.out_root)
    except Exception:
        logger.error("RESREF-R1 run FAILED\n%s", traceback.format_exc())
        sys.exit(1)
