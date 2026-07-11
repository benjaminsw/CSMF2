# SEQREF-FSEQ-CELLBASE v0.1 -- scripts/_diag/cell_baselines.py
# LIFETIME: DIAGNOSTIC
# P2 cell-difficulty sanity (FSEQ v0.1 §4): classical y_up baselines under the
# candidate cell, computed with the SAME degradation implementation and val
# split as training (make_degraded -> _VisionDegraded; no shortcuts).
# Reports, per dataset in {fashion_mnist, mnist}: nearest-upsample PSNR/fwd_rel
# and bicubic-upsample PSNR/fwd_rel on the val split, plus cell metadata.
# Output: printed table + results/_diag/cell_baselines_s4_n0.10.json
# Decision rule (pre-agreed): Fashion baselines finite/sensible/not degenerate
# -> lock s4/n0.10, proceed to P3; nearly unrecoverable or trivially easy ->
# adjust the cell NOW, before training bases.
# No fallback/mock/pass; failures log+raise.
from __future__ import annotations
import argparse
import json
import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fashion_seqref.src.degrade import make_degraded
from fashion_seqref.src.metrics import psnr as _psnr, fwd_rel as _fwd_rel
from fashion_seqref.src.train_utils import setup_logger

logger = setup_logger("fashion_seqref.cell_baselines")

_DATASETS = ("fashion_mnist", "mnist")
_METHODS = ("nearest", "bicubic")


def _upsample(y: torch.Tensor, method: str, size: int = 28) -> torch.Tensor:
    if method == "nearest":
        return F.interpolate(y, size=(size, size), mode="nearest")
    if method == "bicubic":
        return F.interpolate(y, size=(size, size), mode="bicubic",
                             align_corners=False).clamp(0.0, 1.0)
    logger.error("[cell_baselines] unknown method %r", method)
    raise ValueError(f"unknown method {method!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./fashion_seqref/data")
    ap.add_argument("--blur-sigma", type=float, default=1.0)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--noise-sigma", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default=None,
                    help="default: results/_diag/cell_baselines_s<scale>_"
                         "n<noise>.json under ./fashion_seqref")
    args = ap.parse_args()
    out_path = args.out or (f"./fashion_seqref/results/_diag/cell_baselines_"
                            f"s{args.scale}_n{args.noise_sigma:.2f}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    report = {"cell": {"blur_sigma": args.blur_sigma, "scale": args.scale,
                       "noise_sigma": args.noise_sigma, "split": "val"},
              "datasets": {}}
    for ds_name in _DATASETS:
        ds = make_degraded(ds_name, args.data_root, split="val",
                           sigma=args.blur_sigma, scale=args.scale,
                           noise_sigma=args.noise_sigma)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2)
        xs, ys = [], []
        for x, y in dl:
            xs.append(x); ys.append(y)
        x = torch.cat(xs); y = torch.cat(ys)
        if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
            logger.error("[cell_baselines] non-finite data for %s", ds_name)
            raise ValueError(f"non-finite data for {ds_name}")
        entry = {"split_size": len(ds)}
        for m in _METHODS:
            x_hat = _upsample(y, m)
            entry[m] = {"psnr": _psnr(x_hat, x),
                        "fwd_rel": _fwd_rel(x_hat, y, args.blur_sigma,
                                            args.scale)}
        report["datasets"][ds_name] = entry

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    c = report["cell"]
    print(f"=== cell baselines: s{c['scale']} / n{c['noise_sigma']:.2f} / "
          f"blur {c['blur_sigma']} / split val ===")
    print(f"{'dataset':<15} {'n':>6}  {'nearest PSNR':>12} {'fwd_rel':>8}  "
          f"{'bicubic PSNR':>12} {'fwd_rel':>8}")
    for ds_name, e in report["datasets"].items():
        print(f"{ds_name:<15} {e['split_size']:>6}  "
              f"{e['nearest']['psnr']:>12.3f} {e['nearest']['fwd_rel']:>8.4f}  "
              f"{e['bicubic']['psnr']:>12.3f} {e['bicubic']['fwd_rel']:>8.4f}")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
