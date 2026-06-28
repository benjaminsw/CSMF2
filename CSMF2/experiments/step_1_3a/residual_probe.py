# =============================================================================
# RESPROBE v0.1 -- experiments.step_1_3a.residual_probe
# Purpose: decide whether frozen NSF's leftover reconstruction error has any
#          LEARNABLE STRUCTURE left for a residual head (R1) to fix -- BEFORE
#          building R1. Two orthogonal diagnostics, reported together, because
#          "flat" has two causes that demand opposite next moves:
#            (A) magnitude vs noise floor  -- is there headroom above the
#                irreducible noise, or has NSF already hit the floor?
#            (B) structure of the error    -- if there IS headroom, is the
#                residual predictable (edges / spatial smoothness) or white?
#          Branch:
#            SNR ~ 0                      -> AT_NOISE_FLOOR        -> climb data
#                                            staircase (Fashion / CIFAR-gray);
#                                            nothing to correct in THIS cell.
#            SNR >> 0 & structured        -> STRUCTURED_HEADROOM   -> R1 justified.
#            SNR >> 0 & NOT structured    -> STRUCTURELESS_ABOVE_FLOOR -> head
#                                            redesign (R2 / conditional base);
#                                            richer data likely won't help.
# CONVENTION: no fallback / mock / dummy / pass. Any non-finite tensor or bad
#             input -> logger.error + raise. Single frozen NSF expert only.
# Method (uses GROUND-TRUTH x to separate signal from noise EMPIRICALLY):
#   x0      = per_expert_recon_pixels(NSF, y, z_bank)   (B,1,28,28) pixel space
#   A(.)    = downsample(blur(.))                       measurement space
#   signal  = A(x_true) - A(x0)         <- NSF's measurement error, NOISE-FREE
#   noise   = y         - A(x_true)      <- the irreducible noise n (empirical)
#   SNR     = mean||signal||^2 / mean||noise||^2
#   dx      = x_true - x0                <- image-space error R1 would predict
#   structure of dx:  (b1) spatial concentration (CoV of per-pixel |dx|),
#                     (b2) edge correlation  corr(|dx|, |grad x_true|),
#                     (b3) lag-1 spatial autocorrelation of signed dx.
# Changelog (NEW in v0.1):
#   * Introduced. Single-NSF residual-structure probe; magnitude + structure
#     diagnostics; three-way verdict; rec_residual_probe.{json,txt} + plots.
# Update summary:
#   v0.1 gates R1: tells you whether constructed residual complementarity has
#   anything to construct from in the current cell, and -- if flat -- WHICH
#   kind of flat (climb-data vs redesign-head).
# =============================================================================
from __future__ import annotations
import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

logger = logging.getLogger("CSMF2.step_1_3a.residual_probe")
__version__ = "0.1"
__abbr__ = "RESPROBE"

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ...data.degrade import MNISTDegraded, blur, downsample
from ..step_1_1_1_1.model_io import build_from_report
from ..step_1_3.scores import make_z_bank, per_expert_recon_pixels

_EPS = 1e-8


def _A(x_pix, blur_sigma, scale):
    return downsample(blur(x_pix, blur_sigma), scale)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    # Pearson correlation over all elements of two equally-shaped tensors.
    a = a.flatten().double(); b = b.flatten().double()
    if a.numel() != b.numel():
        logger.error("[RESPROBE] _pearson shape mismatch %d vs %d",
                     a.numel(), b.numel())
        raise ValueError("pearson shape mismatch")
    am = a - a.mean(); bm = b - b.mean()
    denom = (am.norm() * bm.norm())
    return float((am @ bm) / (denom + _EPS))


def _spatial_autocorr(field: torch.Tensor) -> float:
    # Lag-1 spatial autocorrelation (mean of horizontal + vertical neighbour
    # Pearson) of a (B,1,H,W) signed field. White noise -> ~0; smooth/
    # structured error -> positive.
    if field.dim() != 4:
        logger.error("[RESPROBE] autocorr expects (B,1,H,W), got %s",
                     tuple(field.shape))
        raise ValueError("autocorr expects (B,1,H,W)")
    h = _pearson(field[..., :, :-1], field[..., :, 1:])   # horizontal lag-1
    v = _pearson(field[..., :-1, :], field[..., 1:, :])   # vertical   lag-1
    return 0.5 * (h + v)


def _edge_magnitude(x: torch.Tensor) -> torch.Tensor:
    # |grad x| via forward differences, zero-padded to keep (B,1,H,W).
    gx = torch.zeros_like(x); gy = torch.zeros_like(x)
    gx[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    gy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    return torch.sqrt(gx * gx + gy * gy + _EPS)


def run(ckpt_dir: str, *, split: str, n_batches: int, batch_size: int,
        z_mode: str, z_bank_size: int, z_bank_seed: int, seed: int,
        snr_floor: float, struct_thr: float, out_root: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    # Single frozen expert (NSF). build_from_report is the architecture-agnostic
    # loader that load_experts wraps; calling it directly skips the >=2-ckpt
    # mixture guard, which does not apply to a single-expert residual probe.
    expert, _cond, ref = build_from_report(ckpt_dir, device)
    experts = [expert]
    if ref.expert != "nsf":
        logger.error("[RESPROBE] expected an NSF checkpoint, got expert=%r",
                     ref.expert)
        raise ValueError(f"residual probe expects NSF, got {ref.expert!r}")
    name = ref.expert
    blur_sigma, scale, noise_sigma = ref.blur_sigma, ref.scale, ref.noise_sigma
    dim = ref.dim
    dtype = next(expert.parameters()).dtype
    z_bank = make_z_bank(dim, z_bank_size, z_mode, z_bank_seed, device, dtype)

    ds = MNISTDegraded(ref.data_root, split=split, sigma=blur_sigma,
                       scale=scale, noise_sigma=noise_sigma)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    sig_e: list[float] = []      # ||A(x_true) - A(x0)||^2  per image (signal)
    noi_e: list[float] = []      # ||y - A(x_true)||^2       per image (noise)
    fwd_rel: list[float] = []    # ||y - A(x0)|| / ||y||     per image (cont.)
    dx_all: list[torch.Tensor] = []      # (B,1,28,28) image-space error
    edge_all: list[torch.Tensor] = []    # (B,1,28,28) |grad x_true|

    with torch.no_grad():
        for bi, (x_img, y) in enumerate(loader):
            if bi >= n_batches:
                break
            x_img = x_img.to(device); y = y.to(device)   # both pixel space
            xk = per_expert_recon_pixels(experts, y, z_bank)   # (B,1,1,28,28)
            x0 = xk[:, 0]                                        # (B,1,28,28)
            if not torch.isfinite(x0).all():
                logger.error("[RESPROBE] non-finite reconstruction at batch %d", bi)
                raise ValueError("non-finite reconstruction")

            A_x0   = _A(x0,    blur_sigma, scale)
            A_xt   = _A(x_img, blur_sigma, scale)
            signal = A_xt - A_x0          # noise-free measurement error
            noise  = y    - A_xt          # empirical irreducible noise n
            r_noisy = y   - A_x0          # = signal + noise

            sig_e += signal.flatten(1).pow(2).sum(1).tolist()
            noi_e += noise.flatten(1).pow(2).sum(1).tolist()
            yn = y.flatten(1).norm(dim=1).clamp_min(_EPS)
            fwd_rel += (r_noisy.flatten(1).norm(dim=1) / yn).tolist()

            dx_all.append((x_img - x0).cpu())
            edge_all.append(_edge_magnitude(x_img).cpu())

    if not sig_e:
        logger.error("[RESPROBE] no batches consumed (split=%s empty?)", split)
        raise RuntimeError("no data consumed")

    dx = torch.cat(dx_all, dim=0)            # (N,1,28,28)
    edge = torch.cat(edge_all, dim=0)        # (N,1,28,28)
    n_img = dx.shape[0]

    # ---- diagnostic (A): magnitude vs noise floor --------------------------
    mean_sig = float(sum(sig_e) / len(sig_e))
    mean_noi = float(sum(noi_e) / len(noi_e))
    snr = mean_sig / (mean_noi + _EPS)
    M = (28 // scale) * (28 // scale)
    theo_floor = float(noise_sigma ** 2 * M)     # sigma^2 * #measurement pixels
    mean_fwd_rel = float(sum(fwd_rel) / len(fwd_rel))

    # ---- diagnostic (B): structure of the image-space error dx -------------
    absdx = dx.abs()
    per_pixel_mean = absdx.mean(dim=0)           # (1,28,28)
    mu = float(per_pixel_mean.mean())
    cov_pixels = float(per_pixel_mean.std() / (mu + _EPS))   # spatial CoV
    edge_corr = _pearson(absdx, edge)            # |dx| vs |grad x_true|
    autocorr = _spatial_autocorr(dx)             # lag-1 spatial autocorr of dx

    structured = bool(edge_corr > struct_thr or autocorr > struct_thr)

    # ---- three-way verdict --------------------------------------------------
    if snr < snr_floor:
        verdict = "AT_NOISE_FLOOR"
        action = ("NSF already reconstructs to ~noise floor in this cell; no "
                  "headroom to correct. Climb the data staircase "
                  "(Fashion-MNIST -> CIFAR-gray); do NOT build R1 here.")
    elif structured:
        verdict = "STRUCTURED_HEADROOM"
        action = ("NSF leaves structured, above-floor error. R1 (small "
                  "correction head conditioned on y, x0) is JUSTIFIED.")
    else:
        verdict = "STRUCTURELESS_ABOVE_FLOOR"
        action = ("Above-floor error but no spatial/edge structure (white). "
                  "Richer data likely won't help; reconsider head design "
                  "(R2 / conditional base) rather than R1-as-is.")

    report = {
        "abbr": __abbr__, "version": __version__,
        "expert": name, "ckpt_dir": ckpt_dir,
        "cell": {"scale": scale, "blur_sigma": blur_sigma,
                 "noise_sigma": noise_sigma, "n_images": n_img,
                 "measurement_pixels": M},
        "diag_A_magnitude": {
            "mean_signal_energy": mean_sig,      # ||A(x_true)-A(x0)||^2
            "mean_noise_energy_empirical": mean_noi,   # ||y-A(x_true)||^2
            "noise_energy_theoretical_sigma2M": theo_floor,
            "SNR_signal_over_noise": snr,
            "mean_fwd_rel": mean_fwd_rel,
            "snr_floor_threshold": snr_floor,
            "above_floor": bool(snr >= snr_floor),
        },
        "diag_B_structure": {
            "spatial_CoV_per_pixel_absdx": cov_pixels,
            "edge_correlation": edge_corr,        # corr(|dx|, |grad x_true|)
            "lag1_spatial_autocorr": autocorr,
            "struct_threshold": struct_thr,
            "structured": structured,
        },
        "verdict": verdict,
        "recommended_action": action,
    }

    # ---- write outputs (atomic) --------------------------------------------
    out_dir = Path(out_root) / f"resprobe_{name}_s{scale}_n{noise_sigma:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rec_residual_probe.json").write_text(json.dumps(report, indent=2))

    txt = _format_txt(report)
    (out_dir / "rec_residual_probe.txt").write_text(txt)
    print(txt)

    _plots(out_dir, per_pixel_mean, absdx, edge, dx)
    logger.info("[RESPROBE] wrote rec_residual_probe.{json,txt} + plots -> %s",
                out_dir)
    return report


def _format_txt(r: dict) -> str:
    a = r["diag_A_magnitude"]; b = r["diag_B_structure"]; c = r["cell"]
    L = []
    L.append("=" * 78)
    L.append(f"RESPROBE v{r['version']} -- NSF residual-structure probe")
    L.append("=" * 78)
    L.append(f"  cell: scale={c['scale']} blur={c['blur_sigma']:.2f} "
             f"noise={c['noise_sigma']:.2f}  n_images={c['n_images']}  "
             f"M(meas px)={c['measurement_pixels']}")
    L.append("")
    L.append("  (A) MAGNITUDE vs NOISE FLOOR")
    L.append(f"      signal ||A(x_true)-A(x0)||^2 (mean) = {a['mean_signal_energy']:.4f}")
    L.append(f"      noise  ||y-A(x_true)||^2     (mean) = {a['mean_noise_energy_empirical']:.4f}"
             f"   (theory sigma^2*M = {a['noise_energy_theoretical_sigma2M']:.4f})")
    L.append(f"      SNR = signal/noise                 = {a['SNR_signal_over_noise']:.4f}"
             f"   (floor {a['snr_floor_threshold']})  above_floor={a['above_floor']}")
    L.append(f"      mean fwd_rel (cont.)               = {a['mean_fwd_rel']:.4f}")
    L.append("")
    L.append("  (B) STRUCTURE of image-space error dx = x_true - x0")
    L.append(f"      spatial CoV of per-pixel |dx|      = {b['spatial_CoV_per_pixel_absdx']:.4f}")
    L.append(f"      edge corr  corr(|dx|,|grad x|)     = {b['edge_correlation']:.4f}")
    L.append(f"      lag-1 spatial autocorr of dx       = {b['lag1_spatial_autocorr']:.4f}"
             f"   (thr {b['struct_threshold']})  structured={b['structured']}")
    L.append("")
    L.append(f"  VERDICT: {r['verdict']}")
    L.append(f"  ACTION : {r['recommended_action']}")
    L.append("=" * 78)
    return "\n".join(L)


def _plots(out_dir: Path, per_pixel_mean, absdx, edge, dx) -> None:
    # p1: per-pixel mean |dx| heatmap (spatial concentration).
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    im = ax.imshow(per_pixel_mean[0].numpy(), cmap="magma")
    ax.set_title("per-pixel mean |dx|", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out_dir / "p1_abs_dx_map.png", dpi=110)
    plt.close(fig)

    # p2: |dx| vs |grad x_true| 2D hist (edge correlation).
    a = absdx.flatten().numpy(); e = edge.flatten().numpy()
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.hist2d(e, a, bins=60, cmap="viridis")
    ax.set_xlabel("|grad x_true|"); ax.set_ylabel("|dx|")
    ax.set_title("edge vs residual", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "p2_edge_vs_residual.png", dpi=110)
    plt.close(fig)


def _parse_args():
    p = argparse.ArgumentParser(description="NSF residual-structure probe (R1 gate)")
    p.add_argument("--ckpt-dir", required=True,
                   help="single frozen NSF result dir (the cell under test)")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--z-mode", choices=("fixed_shared", "zero"),
                   default="fixed_shared")
    p.add_argument("--z-bank-size", type=int, default=4)
    p.add_argument("--z-bank-seed", type=int, default=1234)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snr-floor", type=float, default=0.5,
                   help="SNR (signal/noise) below this = AT_NOISE_FLOOR")
    p.add_argument("--struct-thr", type=float, default=0.1,
                   help="edge-corr OR autocorr above this = structured")
    p.add_argument("--out-root", default="./CSMF2/experiments/step_1_3a/results")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    a = _parse_args()
    try:
        run(a.ckpt_dir, split=a.split, n_batches=a.n_batches,
            batch_size=a.batch_size, z_mode=a.z_mode,
            z_bank_size=a.z_bank_size, z_bank_seed=a.z_bank_seed,
            seed=a.seed, snr_floor=a.snr_floor, struct_thr=a.struct_thr,
            out_root=a.out_root)
    except Exception:
        logger.error("RESPROBE run FAILED\n%s", traceback.format_exc())
        sys.exit(1)
