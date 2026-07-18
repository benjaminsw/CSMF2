# SEQREF-EVNOISE v0.1 -- eval_noise
# LIFETIME: DIAGNOSTIC
# Live-path step 1 (INDEX SS4 / EXEC SS7.2 Step A): eval-noise check.
# Re-evaluates ONE locked checkpoint R times with identical settings and NO
# retraining, to measure the PSNR spread caused purely by posterior
# reconstruction sampling (posterior_pixel_mean draws n_post fresh z per eval).
# Reports per-repeat PSNR/MSE/fwd_rel/pstd and the PSNR RANGE (max - min) --
# the campaign's single definition of measured variation.
# Optional --eval-seed seeds torch's global RNG before EACH repeat with
# (eval_seed + repeat_index); with a FIXED --eval-seed and --same-seed the
# spread should collapse to ~0, demonstrating the stabilisation fix.
# Reuses train_base._build/_val_recon directly -- no reimplementation drift.
# No fallback/mock/silent-pass. Failures: logger.error + raise.
from __future__ import annotations
import argparse
import json
import logging
import os

import torch
import yaml
from torch.utils.data import DataLoader

from seqref_mri.src.degrade import make_degraded
from seqref_mri.scripts.train_base import _build, _val_recon

logger = logging.getLogger("seqref_mri.eval_noise")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="the base config the checkpoint was trained with")
    ap.add_argument("--ckpt", required=True,
                    help="path to checkpoint.pt of the LOCKED run")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-post", type=int, default=None,
                    help="override n_post (default: 16, matching training)")
    ap.add_argument("--eval-seed", type=int, default=None,
                    help="seed torch RNG per repeat (eval_seed + i); "
                         "omit = unseeded, measuring the natural spread")
    ap.add_argument("--same-seed", action="store_true",
                    help="with --eval-seed: use the SAME seed every repeat "
                         "(spread should collapse to ~0)")
    ap.add_argument("--out", default=None,
                    help="optional JSON output path")
    args = ap.parse_args()

    if args.repeats < 3:
        logger.error("[eval_noise] repeats=%d < 3 -- protocol requires >=3",
                     args.repeats)
        raise ValueError("repeats must be >= 3 (EXEC SS7.2 Step A)")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if not os.path.isfile(args.ckpt):
        logger.error("[eval_noise] checkpoint not found: %s", args.ckpt)
        raise FileNotFoundError(args.ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cell = cfg["cell"]
    scale = int(cell["scale"]); blur_sigma = float(cell["blur_sigma"])
    n_post = args.n_post if args.n_post is not None else 16

    va = make_degraded(cell.get("dataset"), cell["data_root"], split="val",
                       sigma=blur_sigma, scale=scale,
                       noise_sigma=float(cell["noise_sigma"]))
    vl = DataLoader(va, batch_size=int(cfg["train"]["batch_size"]),
                    shuffle=False, num_workers=2)

    model = _build(cfg, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info("[eval_noise] ckpt=%s (epoch %s) n_post=%d repeats=%d "
                "eval_seed=%s same_seed=%s", args.ckpt, ckpt.get("epoch"),
                n_post, args.repeats, args.eval_seed, args.same_seed)

    rows = []
    for i in range(args.repeats):
        if args.eval_seed is not None:
            s = args.eval_seed if args.same_seed else args.eval_seed + i
            torch.manual_seed(s)
            if device == "cuda":
                torch.cuda.manual_seed_all(s)
        mse, psnr, fwd, pstd = _val_recon(model, vl, device, blur_sigma,
                                          scale, n_post)
        rows.append({"repeat": i, "psnr": psnr, "mse": mse,
                     "fwd_rel": fwd, "pstd": pstd})
        logger.info("[eval_noise] repeat %d: psnr=%.4f mse=%.5f fwd_rel=%.4f "
                    "pstd=%.4f", i, psnr, mse, fwd, pstd)

    psnrs = [r["psnr"] for r in rows]
    fwds = [r["fwd_rel"] for r in rows]
    rng_psnr = max(psnrs) - min(psnrs)
    rng_fwd = max(fwds) - min(fwds)
    logger.info("[eval_noise] PSNR range (max-min) = %.5f dB over %d repeats",
                rng_psnr, args.repeats)
    logger.info("[eval_noise] fwd_rel range        = %.5f", rng_fwd)

    result = {"ckpt": args.ckpt, "epoch": ckpt.get("epoch"),
              "n_post": n_post, "repeats": args.repeats,
              "eval_seed": args.eval_seed, "same_seed": args.same_seed,
              "rows": rows, "psnr_range": rng_psnr, "fwd_rel_range": rng_fwd}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("[eval_noise] written: %s", args.out)


if __name__ == "__main__":
    main()
