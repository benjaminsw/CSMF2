# =============================================================================
# SEQREF-TB-MRI v0.5 -- scripts.train_base
# LIFETIME: KEEP
# Purpose: MRI base-expert trainer (NICE / RealNVP / NSF) on the locked cell.
#   REBUILDS the MNIST train_base per the S1 ledger; nothing here imports
#   degrade or logit transforms.
# Locked conventions implemented:
#   * Data: fastmri_data v0.3 (construction A, exact-count mask, SHA-256
#     seeds; set_epoch enforced; make_train_loader -- no persistent workers).
#   * State: complex two-channel (B,2,96,96), flattened dim = 18432 for the
#     flows (EXEC 3.14).
#   * D1 conditioning input: normalized two-channel zero-filled state
#     [Re(x0), Im(x0)], x0 = A^H y (conditioner v0.6, in_channels=2).
#   * D2 normalization: state AND target divided by the per-file HDF5 `max`
#     attr (meta.file_attr_max). Metrics afterwards use data_range = 1.0 --
#     NEVER the raw attr again (double-application guard). Both the raw
#     attr and the normalized range are recorded in run facts.
#   * Loss: NLL = -log_prob(x_flat_norm | cond_input), mean over batch.
#   * Val: NLL + posterior-mean magnitude PSNR/SSIM (data_range=1.0) +
#     3.11 k-space consistency on the UN-normalized reconstruction vs y.
#     Posterior mean: P latin draws via model.cond + model.decode (batched;
#     _BaseExpert.sample() is UNUSED -- its (1,1,H,W) single-channel check
#     is incompatible with 2-channel conditioning; flagged for I4).
#   * Test-0 analogue: --test0 runs the FULL-MASK edge cell (mask = all
#     ones, x0 = x_true) -- SEPARATE from ordinary masked validation.
#   * Timing recorded per epoch, SPLIT: t_data / t_fb / t_val / t_sample.
# CONVENTION: logger.error + raise. No fallback/mock/silent pass.
# Exposes run_training(cfg) for the I3 pilot (importable, no subprocess).
# Changelog (v0.4 -> v0.5, NICEXT protocol fixes):
#   * Run-dir hash uses the source checkpoint SHA256 + epoch, not the
#     filename (basename "best.pt" collides across sources).
#   * Source checkpoint FULLY LOCKED: every unchanged NICEXT field
#     (train_slices, val_slices, batch, lr, seed_index, n_post,
#     subset_seed, test0) must match the source cfg; only epochs/output/
#     init metadata are exempt. Source epoch must be a non-negative int.
#   * --init-sha256: computed SHA must equal the declared SHA --
#     comparing proves the INTENDED checkpoint was used, not merely
#     which one was. (Mandatory with --init-from since the v0.5 patch.)
#   * --init-sha256 MANDATORY whenever --init-from is used (and vice
#     versa): optional verification gets skipped; smoke and formal both
#     carry the declared hash. TOCTOU closed: file re-hashed after
#     torch.load. Competence block lists qualifying_epochs. --epochs
#     must be a positive int (best_rec can never remain None).
#   * Two-threshold outcome recorded: best.pt saves init metadata;
#     facts carry best_epoch + best_val (full metrics) + a competence
#     block (floors 31.53/0.691; passed = any epoch meets BOTH) --
#     PSNR-only interpretation prevented at the record level.
# Changelog (v0.3 -> v0.4, NICEXT continuation support):
#   * --init-from CKPT: WEIGHTS-ONLY initialization from a saved best.pt.
#     Strict validation before loading: blob keys present, expert match,
#     architecture fields (n_layers/hidden/h_dim/cond_width/nsf_B) must
#     equal the current cfg -- no locked field is silently reset; strict
#     state-dict load. Full sha256 + source epoch recorded in facts under
#     "init"; absolute path NEVER persisted (basename only).
#   * Epoch identity preserved: training continues at source_epoch + 1
#     (e.g. ep29 best.pt -> epochs 30..30+added-1); set_epoch uses the
#     ABSOLUTE epoch, so the train-mask schedule extends rather than
#     repeats. --epochs means ADDED epochs when --init-from is given.
#   * Optimizer is FRESH Adam by design (checkpoints store no optimizer
#     state) -- recorded limitation, per the NICEXT declaration.
# Changelog (v0.2 -> v0.3, pre-deployment review fixes):
#   * CUDA synchronization at EVERY GPU timing boundary (t_fb, t_sample,
#     validation total, epoch total): async kernels no longer shift work
#     between timing buckets -- these numbers size the 3.15 budgets.
#   * Persisted facts no longer contain the absolute run_dir (stale after
#     report staging renames); only run_dir_rel is written. The absolute
#     path is returned in-memory under "_run_dir_abs_unpersisted".
#   * cfg is SANITIZED before persistence (facts.json AND best.pt): the
#     absolute out_root is dropped, out_root_rel="runs" recorded instead.
# Changelog (v0.1 -> v0.2, pre-deployment review fixes):
#   * Timing categories DISJOINT: t_data / t_fb / t_val_non_sample /
#     t_sample, plus t_epoch_total from an independent wall clock.
#   * Run facts record run_dir_rel (basename relative to out_root) so
#     reports whose staging directory is renamed do not hold stale
#     absolute paths; absolute run_dir kept for the live run only.
# Changelog (NEW in v0.1): Introduced.
# Update summary: minimal MRI base trainer implementing D1/D2 and the
#   locked cell; budgets/protocol values come from EXEC 3.15 after the
#   pilot -- this file takes them as arguments, never hard-codes them.
# =============================================================================
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from seqref_mri.src.fastmri_data import (FastMRISliceDataset,
                                         make_train_loader, CELL_HW,
                                         ACCELERATION, CENTER_FRACTION)
from seqref_mri.src.forward_operator import (MaskedFourierOperator,
                                             two_channel_to_complex,
                                             complex_to_two_channel)
from seqref_mri.src.metrics import psnr_per_sample, ssim_per_sample
from seqref_mri.src.conditioner import Conditioner
from seqref_mri.src.base_experts import build_expert
from seqref_mri.src.train_utils import (setup_logger, seed_from_index,
                                        cfg_hash, make_run_dir, write_json)

logger = setup_logger("seqref_mri.train_base")

__version__ = "0.5"
__abbr__ = "SEQREF-TB-MRI"

DIM = 2 * CELL_HW * CELL_HW          # 18432 -- flattened two-channel state
IN_CHANNELS = 2                       # D1 conditioning input channels
NORMALIZED_DATA_RANGE = 1.0           # D2: after /file_attr_max


def _fail(msg: str) -> None:
    logger.error("[train_base] %s", msg)
    raise RuntimeError(msg)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _subset(ds, n: int | None, seed: int):
    if n is None or n >= len(ds):
        return ds
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.choice(len(ds), size=n, replace=False)
    return Subset(ds, [int(i) for i in idx])


def _collate(items: list[dict]) -> dict:
    out = {k: torch.stack([it[k] for it in items])
           for k in ("x_true", "y", "mask", "target_mag")}
    out["file_attr_max"] = torch.tensor(
        [it["meta"]["file_attr_max"] for it in items], dtype=torch.float32)
    out["meta"] = [it["meta"] for it in items]
    return out


def _prepare(batch: dict, device: str, *, test0: bool) -> dict:
    # Returns normalized tensors + per-sample ops. D2: divide by attr max.
    y = batch["y"].to(device)                       # (B,96,96) c64
    x_true = batch["x_true"].to(device)             # (B,2,96,96)
    amax = batch["file_attr_max"].to(device)        # (B,)
    if test0:
        mask = torch.ones(y.shape[0], CELL_HW, dtype=torch.bool)
    else:
        mask = batch["mask"]
    ops = [MaskedFourierOperator(mask[i].to(device))
           for i in range(y.shape[0])]
    if test0:
        # full-mask edge: regenerate y under the full mask so y = F(x_true)
        xt_c = two_channel_to_complex(x_true)
        y = torch.stack([ops[i].A(xt_c[i]) for i in range(len(ops))])
    x0_c = torch.stack([ops[i].A_adjoint(y[i]) for i in range(len(ops))])
    a = amax.view(-1, 1, 1, 1)
    cond_in = complex_to_two_channel(x0_c) / a           # (B,2,96,96) D2-norm
    x_norm = x_true / a                                  # (B,2,96,96)
    tgt_norm = (batch["target_mag"].to(device) / amax.view(-1, 1, 1)
                ).unsqueeze(1)                            # (B,1,96,96)
    return {"y": y, "x_norm": x_norm, "cond_in": cond_in,
            "tgt_norm": tgt_norm, "amax": amax, "ops": ops}


def _posterior_mean(model, cond_in: torch.Tensor, n_post: int
                    ) -> torch.Tensor:
    # Batched posterior mean via cond+decode (sample() unused -- see header).
    h = model.cond(cond_in)                               # (B, h_dim)
    acc = None
    for _ in range(n_post):
        z = torch.randn(cond_in.shape[0], model.dim,
                        device=cond_in.device, dtype=cond_in.dtype)
        x = model.decode(z, h)
        acc = x if acc is None else acc + x
    return acc / n_post                                    # (B, DIM) norm'd


@torch.no_grad()
def _validate(model, loader, device: str, n_post: int, *, test0: bool
              ) -> dict:
    model.eval()
    device_s = device
    _sync(device_s)
    t_val0 = time.perf_counter()
    t_sample = 0.0
    nlls, psnrs, ssims, cons = [], [], [], []
    for batch in loader:
        p = _prepare(batch, device, test0=test0)
        nll = -model.log_prob(p["x_norm"].flatten(1), p["cond_in"])
        nlls.append(nll.cpu())
        _sync(device_s)
        t0 = time.perf_counter()
        xm = _posterior_mean(model, p["cond_in"], n_post)
        _sync(device_s)
        t_sample += time.perf_counter() - t0
        xm_state = xm.view(-1, 2, CELL_HW, CELL_HW)
        xm_c = two_channel_to_complex(xm_state)
        mag = xm_c.abs().unsqueeze(1)                      # normalized mag
        psnrs.append(psnr_per_sample(
            mag.cpu(), p["tgt_norm"].cpu(),
            data_range=NORMALIZED_DATA_RANGE).cpu())
        ssims.append(ssim_per_sample(
            mag.cpu(), p["tgt_norm"].cpu(),
            data_range=NORMALIZED_DATA_RANGE).cpu())
        # 3.11 consistency on the UN-normalized reconstruction
        xm_un = xm_c * p["amax"].view(-1, 1, 1)
        cons.extend(float(p["ops"][i].consistency(xm_un[i], p["y"][i]))
                    for i in range(len(p["ops"])))
    _sync(device_s)
    t_val_total = time.perf_counter() - t_val0
    return {"nll": float(torch.cat(nlls).mean()),
            "psnr": float(torch.cat(psnrs).mean()),
            "ssim": float(torch.cat(ssims).mean()),
            "consistency_mean": float(np.mean(cons)),
            "t_val_non_sample_s": t_val_total - t_sample,
            "t_sample_s": t_sample}


def _load_init_weights(model, cfg: dict) -> dict:
    # NICEXT continuation: weights-only, strictly validated.
    p = Path(cfg["init_from"])
    if not p.is_file():
        _fail(f"--init-from checkpoint missing: {p}")
    import hashlib
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    blob = torch.load(p, map_location="cpu")
    for key in ("model", "cfg", "epoch"):
        if key not in blob:
            _fail(f"--init-from checkpoint lacks '{key}'")
    scfg = blob["cfg"]
    if scfg.get("expert") != cfg["expert"]:
        _fail(f"--init-from expert {scfg.get('expert')!r} != {cfg['expert']!r}")
    # EVERY unchanged field locked -- architecture AND protocol; only
    # epochs / output location / init metadata are intentionally exempt.
    for k in ("n_layers", "hidden", "h_dim", "cond_width",
              "train_slices", "val_slices", "batch", "lr",
              "seed_index", "n_post", "subset_seed", "test0"):
        if scfg.get(k) != cfg.get(k):
            _fail(f"--init-from cfg[{k!r}]={scfg.get(k)!r} != current "
                  f"{cfg.get(k)!r} -- locked field mismatch")
    if cfg["expert"] == "nsf" and scfg.get("nsf_B") != cfg.get("nsf_B"):
        _fail("--init-from nsf_B mismatch")
    source_epoch = blob["epoch"]
    if type(source_epoch) is not int or source_epoch < 0:
        _fail(f"--init-from invalid source epoch: {source_epoch!r}")
    declared_sha = cfg["init_sha256"]
    import re
    if not re.fullmatch(r"[0-9a-fA-F]{64}", declared_sha):
        _fail("--init-sha256 must be exactly 64 hexadecimal characters")
    declared_sha = declared_sha.lower()
    if sha != declared_sha:
        _fail(f"--init-from SHA mismatch: computed {sha} != declared "
              f"{declared_sha} -- wrong checkpoint")
    # TOCTOU: re-hash after torch.load consumed the file
    if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
        _fail("--init-from checkpoint changed during loading")
    try:
        model.load_state_dict(blob["model"], strict=True)
    except Exception as e:
        _fail(f"--init-from strict state-dict load FAILED: {e!r}")
    logger.info("[train_base] init-from %s (epoch %d, sha %s...)",
                p.name, blob["epoch"], sha[:12])
    return {"init_from_name": p.name, "init_from_dir": p.parent.name,
            "init_from_sha256": sha, "init_from_epoch": source_epoch,
            "declared_sha_verified": True,
            "optimizer": "FRESH Adam (checkpoint stores no optimizer "
                         "state; declared limitation)"}


def run_training(cfg: dict) -> dict:
    if type(cfg.get("epochs")) is not int or cfg["epochs"] <= 0:
        _fail("--epochs must be a positive integer")
    if cfg.get("init_from") and not cfg.get("init_sha256"):
        _fail("--init-sha256 is required whenever --init-from is used")
    if cfg.get("init_sha256") and not cfg.get("init_from"):
        _fail("--init-sha256 requires --init-from")
    # cfg keys: data_root, expert, epochs, train_slices, val_slices, batch,
    # lr, seed_index, n_layers, hidden, h_dim, cond_width, nsf_B, n_post,
    # out_root, test0(bool), subset_seed
    rng_seed = seed_from_index(cfg["seed_index"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test0 = bool(cfg.get("test0", False))

    tr_full = FastMRISliceDataset(cfg["data_root"], split="train",
                                  mode="train")
    va_full = FastMRISliceDataset(cfg["data_root"], split="val", mode="eval")
    tr = _subset(tr_full, cfg.get("train_slices"), cfg["subset_seed"])
    va = _subset(va_full, cfg.get("val_slices"), cfg["subset_seed"] + 1)

    cond = Conditioner(in_channels=IN_CHANNELS, width=cfg["cond_width"],
                       h_dim=cfg["h_dim"])
    expert_kwargs = {"n_layers": cfg["n_layers"]}
    if cfg["expert"] == "nsf":
        if not (isinstance(cfg.get("nsf_B"), (int, float))
                and cfg["nsf_B"] > 0):
            _fail("expert=nsf requires nsf_B > 0 (D3 pilot-measured value)")
        # B injected via a thin rebuild: build_expert has no B passthrough,
        # so construct CondNSF directly with the pilot value.
        from seqref_mri.src.base_experts import CondNSF
        model = CondNSF(dim=DIM, h_dim=cfg["h_dim"], conditioner=cond,
                        hidden=cfg["hidden"], n_layers=cfg["n_layers"],
                        K=8, B=float(cfg["nsf_B"]), use_film=True)
    else:
        model = build_expert(cfg["expert"], dim=DIM, h_dim=cfg["h_dim"],
                             conditioner=cond, hidden=cfg["hidden"],
                             use_film=True, **expert_kwargs)
    init_rec = None
    if cfg.get("init_from"):
        init_rec = _load_init_weights(model, cfg)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    hash_cfg = {k: v for k, v in cfg.items()
                if k not in ("out_root", "init_from", "init_sha256")}
    if init_rec is not None:
        # path-free AND collision-free: SHA + epoch identify the source
        hash_cfg["init_from_sha256"] = init_rec["init_from_sha256"]
        hash_cfg["init_from_epoch"] = init_rec["init_from_epoch"]
    ch = cfg_hash(hash_cfg)
    # persisted config: NO absolute paths (stale after staging renames)
    persisted_cfg = {k: v for k, v in cfg.items()
                     if k not in ("out_root", "init_from", "init_sha256")}
    persisted_cfg["out_root_rel"] = "runs"
    run_dir = make_run_dir(cfg["out_root"], expert=cfg["expert"],
                           accel=ACCELERATION,
                           center_fraction=CENTER_FRACTION,
                           seed_index=cfg["seed_index"], cfg_hash_hex=ch,
                           test0=test0)
    logger.info("[train_base] %s | dim=%d params=%.2fM device=%s run=%s",
                cfg["expert"], DIM, n_params / 1e6, device, run_dir)

    va_loader = DataLoader(va, batch_size=cfg["batch"], shuffle=False,
                           collate_fn=_collate)
    tr_indices = (tr.indices if isinstance(tr, Subset)
                  else list(range(len(tr_full))))
    # epoch identity: continuation starts at source_epoch + 1; --epochs
    # means ADDED epochs in that case. set_epoch gets ABSOLUTE numbers.
    start_epoch = (init_rec["init_from_epoch"] + 1) if init_rec else 0
    history = []
    best_rec = None
    for epoch in range(start_epoch, start_epoch + cfg["epochs"]):
        _sync(device)
        t_epoch0 = time.perf_counter()
        tr_full.set_epoch(epoch)          # enforced fresh-mask policy
        sampler = torch.utils.data.SubsetRandomSampler(
            tr_indices, generator=torch.Generator().manual_seed(
                rng_seed + epoch))
        tr_loader = make_train_loader(tr_full, batch_size=cfg["batch"],
                                      sampler=sampler, collate_fn=_collate)
        model.train()
        t_data = t_fb = 0.0
        ep_nll = []
        t_mark = time.perf_counter()
        for batch in tr_loader:
            t_data += time.perf_counter() - t_mark
            _sync(device)
            t0 = time.perf_counter()
            p = _prepare(batch, device, test0=test0)
            nll = -model.log_prob(p["x_norm"].flatten(1), p["cond_in"]).mean()
            if not torch.isfinite(nll):
                _fail(f"non-finite training NLL at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            nll.backward()
            opt.step()
            _sync(device)
            t_fb += time.perf_counter() - t0
            ep_nll.append(float(nll))
            t_mark = time.perf_counter()
        val = _validate(model, va_loader, device, cfg["n_post"], test0=test0)
        rec = {"epoch": epoch, "train_nll_mean": float(np.mean(ep_nll)),
               "val": val, "t_data_s": t_data, "t_fb_s": t_fb,
               "t_val_non_sample_s": val["t_val_non_sample_s"],
               "t_sample_s": val["t_sample_s"],
               "t_epoch_total_s": (_sync(device),
                                   time.perf_counter() - t_epoch0)[1]}
        history.append(rec)
        logger.info("[train_base] ep%d nll=%.4f val_nll=%.4f psnr=%.3f "
                    "ssim=%.4f cons=%.3e | t(data/fb/val/sample)="
                    "%.1f/%.1f/%.1f/%.1f s", epoch, rec["train_nll_mean"],
                    val["nll"], val["psnr"], val["ssim"],
                    val["consistency_mean"], t_data, t_fb,
                    val["t_val_non_sample_s"], val["t_sample_s"])
        if best_rec is None or val["psnr"] > best_rec["val"]["psnr"]:
            best_rec = rec
            torch.save({"model": model.state_dict(),
                        "cfg": persisted_cfg,
                        "epoch": epoch,
                        "init": init_rec}, Path(run_dir) / "best.pt")

    facts = {"script": f"{__abbr__} v{__version__}", "cfg": persisted_cfg,
             "rng_seed": rng_seed, "n_params": n_params, "device": device,
             "normalization": {"scheme": "D2: state+target / file_attr_max",
                               "metric_data_range": NORMALIZED_DATA_RANGE,
                               "note": "raw file attrs recorded per sample "
                                       "in loader meta; range applied ONCE"},
             "history": history,
             "best_val_psnr": best_rec["val"]["psnr"],
             "best_epoch": best_rec["epoch"],
             "best_val": best_rec["val"],
             "competence": (lambda ce: {
                 "psnr_floor": 31.53, "ssim_floor": 0.691,
                 "passed": bool(ce), "qualifying_epochs": ce})(
                 [row["epoch"] for row in history
                  if row["val"]["psnr"] >= 31.53
                  and row["val"]["ssim"] >= 0.691]),
             "init": init_rec,
             "run_dir_rel": Path(run_dir).name}
    write_json(str(Path(run_dir) / "facts.json"), facts)
    # absolute path returned IN-MEMORY ONLY (never persisted -- stale after
    # report staging renames)
    return facts | {"_run_dir_abs_unpersisted": str(run_dir)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--expert", required=True,
                   choices=["nice", "realnvp", "nsf"])
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--train-slices", type=int, default=None)
    p.add_argument("--val-slices", type=int, default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed-index", type=int, default=0)
    p.add_argument("--n-layers", type=int, required=True)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--h-dim", type=int, default=128)
    p.add_argument("--cond-width", type=int, default=64)
    p.add_argument("--nsf-B", type=float, default=None)
    p.add_argument("--n-post", type=int, default=4)
    p.add_argument("--out-root", default="seqref_mri/results/bases")
    p.add_argument("--subset-seed", type=int, default=20260904)
    p.add_argument("--test0", action="store_true",
                   help="FULL-MASK edge cell (separate from masked val)")
    p.add_argument("--init-from", default=None,
                   help="weights-only continuation from a saved best.pt "
                        "(strictly validated; --epochs = ADDED epochs)")
    p.add_argument("--init-sha256", default=None,
                   help="declared full sha256 of the --init-from "
                        "checkpoint; MANDATORY whenever --init-from is "
                        "used (smoke and formal); run fails on mismatch")
    a = p.parse_args()
    run_training(vars(a) | {"data_root": a.data_root,
                            "train_slices": a.train_slices,
                            "val_slices": a.val_slices,
                            "seed_index": a.seed_index,
                            "n_layers": a.n_layers,
                            "cond_width": a.cond_width,
                            "nsf_B": a.nsf_B, "n_post": a.n_post,
                            "out_root": a.out_root,
                            "subset_seed": a.subset_seed,
                            "init_from": a.init_from,
                            "init_sha256": a.init_sha256})


if __name__ == "__main__":
    main()
