# SEQREF-CONDDIAG v0.1 -- check_conditioning
# LIFETIME: DIAGNOSTIC
# R1 rescue diagnostics for a trained base run. Loads run_dir (config.yaml +
# checkpoint.pt), rebuilds the expert via train_base._build, and reports the
# common conditioning/recon stats + ldj. Writes JSON to results/_diag/.
# No fallback/mock/pass. Failures logger.error + raise. Decode RAISES on failure.
# Changelog (v0.1):
#   * h_std, same_z_diff_y_output_diff, correct_y_vs_shuffled_y_fwd_rel,
#     sample_pixel_std, val_psnr/mse/fwd_rel, ldj_mean/ldj_std.
#   * RealNVP/NICE per-layer s/t/m stats NOT included -- need layer-internal
#     hooks (realnvp_layer.py / nice_layer.py); flagged for a follow-up.
from __future__ import annotations
import argparse
import logging
import os

import torch
import yaml
from torch.utils.data import DataLoader

from seqref_warm.src.degrade import MNISTDegraded, dequantize_logit
from seqref_warm.src.metrics import mse as _mse, psnr as _psnr, fwd_rel as _fwd_rel
from seqref_warm.src.train_utils import setup_logger, write_json
from seqref_warm.scripts.train_base import _build, _posterior_pixel_mean

logger = setup_logger("seqref_warm.check_conditioning")
__version__ = "0.1"
_N_POST = 16


def _load_run(run_dir: str, device: str):
    cfg_path = os.path.join(run_dir, "config.yaml")
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    for p in (cfg_path, ckpt_path):
        if not os.path.isfile(p):
            logger.error("[conddiag] missing %s", p)
            raise FileNotFoundError(p)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = _build(cfg, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()
    return cfg, model


@torch.no_grad()
def diagnose(run_dir: str, device: str, out_json: str) -> dict:
    cfg, model = _load_run(run_dir, device)
    cell = cfg["cell"]
    scale = int(cell["scale"]); blur = float(cell["blur_sigma"])
    ds = MNISTDegraded(cell["data_root"], split="val", sigma=blur, scale=scale,
                       noise_sigma=float(cell["noise_sigma"]))
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    B = x.size(0)

    # h_std: across-batch conditioner spread
    h = model.cond(y)
    h_std = float(h.std(dim=0).mean())

    # same_z_diff_y_output_diff: fix z, vary y (shuffle) -> pixel-space output diff
    z = torch.randn(B, model.dim, device=device, dtype=x.dtype)
    perm = torch.randperm(B, device=device)
    h_shuf = model.cond(y[perm])
    xa = model.decode(z, h)
    xb = model.decode(z, h_shuf)
    same_z_diff_y = float((xa - xb).abs().mean())

    # recon: correct-y vs shuffled-y fwd_rel (posterior pixel-mean)
    x_hat, sstd = _posterior_pixel_mean(model, y, _N_POST)
    x_hat_shuf, _ = _posterior_pixel_mean(model, y[perm], _N_POST)
    fwd_correct = _fwd_rel(x_hat, y, blur, scale)
    fwd_shuffled = _fwd_rel(x_hat_shuf, y, blur, scale)

    # ldj (available from encode, no layer internals)
    x_logit, _ = dequantize_logit(x)
    _, ldj = model.encode(x_logit.flatten(1), h)

    rep = {
        "run_dir": run_dir, "expert": cfg["expert"],
        "h_std": h_std,
        "same_z_diff_y_output_diff": same_z_diff_y,
        "sample_pixel_std": sstd,
        "val_psnr": _psnr(x_hat, x), "val_mse": _mse(x_hat, x),
        "val_fwd_rel": fwd_correct,
        "correct_y_vs_shuffled_y_fwd_rel": {
            "correct": fwd_correct, "shuffled": fwd_shuffled,
            "uses_y": bool(fwd_correct < fwd_shuffled)},
        "ldj_mean": float(ldj.mean()), "ldj_std": float(ldj.std()),
        "note_missing": ("per-layer s/t (RealNVP) and m/diag_log_s (NICE) require "
                         "layer-internal hooks; not in v0.1"),
    }
    if same_z_diff_y < 1e-4:
        logger.error("[conddiag] same_z_diff_y_output_diff ~0 (%.2e): y IGNORED",
                     same_z_diff_y)
    write_json(out_json, rep)
    logger.info("[conddiag] %s: h_std=%.4f same_z_diff_y=%.4e uses_y=%s "
                "psnr=%.3f fwd_rel=%.4f", cfg["expert"], h_std, same_z_diff_y,
                rep["correct_y_vs_shuffled_y_fwd_rel"]["uses_y"],
                rep["val_psnr"], fwd_correct)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or os.path.join("./seqref_warm/results/_diag",
                                   f"conddiag_{os.path.basename(args.run_dir.rstrip('/'))}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    diagnose(args.run_dir, device, out)


if __name__ == "__main__":
    main()
