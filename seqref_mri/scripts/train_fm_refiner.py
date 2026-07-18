# SEQREF-TRNFM v0.1 -- train_fm_refiner
# LIFETIME: KEEP
# Level-3 flow-matching refiner training (SEQREF-FMREFINE v0.1). Frozen chain
# NSF -> NICE Run-3 supplies x1 (cached, dual-sha); Arm B adds a 4th cond
# channel xR = frozen RealNVP_refine output on x0 (cached likewise).
# Loss (§6): L_fm + 0.1*L_img(x2, gated K_train rollout) + 1e-3*mean((g·Δ)²).
# K-SWEEP LOGGED EVERY EPOCH: dpsnr at K=1/4/8 (K=8 official gate metric;
# K=1 ≈ Level-1 regression control -- approved tightening).
# Gates: PRE-REGISTERED module constants (stage-2 two-tier vs x1); early stop
# patience 15 on val_dpsnr (K=8); keep-best on val_dpsnr.
# No fallback/mock/pass; frozen chain asserted grad-free each epoch.
# Changelog (TRNFM v0.1-fseq -> v0.2-fseq, SEQREF-FSEQ W3 / P7):
#   * chained_channel config key: frozen STAGE-2 checkpoint applied to
#     x1 (inputs [y_up, x1, A^T r1]) as an extra conditioning channel --
#     distinct cache keys (train/val_chained); mutually exclusive with
#     arm_b_expert_channel (which applies a stage-1 refiner to x0 and
#     would be WRONG for a proposal trained on x1); sha recorded in
#     status.json. Raise on conflict/missing checkpoint.
# Changelog (TRNFM v0.1 -> v0.1-fseq, SEQREF-FSEQ W2):
#   * seqref_mri fork; dataset construction via degrade.make_degraded
#     with REQUIRED cell.dataset key ({mnist, fashion_mnist}; raise on
#     absent/unknown). No other logic changes.
# Changelog (v0.1):
#   * Initial. Reuses BASEIO caches, METRIC v0.2, GATEUPD; TRNREF-style
#     schema/plots/grid + k-sweep columns and curve.
from __future__ import annotations
import argparse
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from seqref_mri.src.degrade import make_degraded
from seqref_mri.src.metrics import psnr as _psnr, ssim as _ssim, fwd_rel as _fwd_rel
from seqref_mri.src.refiners.base_io import (FrozenBase, FrozenStage1,
                                               precompute_split,
                                               precompute_stage2)
from seqref_mri.src.refiners.flow_matching_refiner import FMRefiner
from seqref_mri.src.refiners.gated_update import GatedUpdate
from seqref_mri.src.train_utils import (setup_logger, seed_from_index,
                                          cfg_hash, write_json, sha256_file)

logger = setup_logger("seqref_mri.train_fm_refiner")
__version__ = "0.1"

# Pre-registered gate constants (stage-2 two-tier; NOT config knobs)
_GATE_HARD = 0.3
_GATE_MEANINGFUL = 0.1
_GATE_PCT = 0.55
_FWD_TOL = 0.0          # hard: fwd_rel(x2) <= fwd_rel(x1) exactly per plan
_K_SWEEP = (1, 4, 8)
_K_OFFICIAL = 8


def _charbonnier(a, b, eps):
    return torch.sqrt((a - b) ** 2 + eps * eps).mean()


def _psnr_per_sample(x_hat, x_true):
    m = ((x_hat - x_true) ** 2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / m)


@torch.no_grad()
def _forward_split(model, cond, x1, k, bs, device):
    xs, dxs, gs = [], [], []
    for i in range(0, cond.size(0), bs):
        c = cond[i:i + bs].to(device)
        x = x1[i:i + bs].to(device)
        x2, dx, g = model(x, c, k)
        xs.append(x2.cpu()); dxs.append(dx.cpu()); gs.append(g.cpu())
    return torch.cat(xs), torch.cat(dxs), torch.cat(gs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed_index = args.seed if args.seed is not None else int(cfg["train"]["seed"])
    rng_seed = seed_from_index(seed_index)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fm = cfg["flow_matching"]
    if fm.get("path") != "linear" or fm.get("integrator") != "euler":
        logger.error("[train_fm] v0.1 supports path=linear integrator=euler "
                     "only, got %s/%s", fm.get("path"), fm.get("integrator"))
        raise ValueError("unsupported flow_matching settings")
    k_train = int(fm.get("k_train", 4))

    base = FrozenBase(cfg["base"]["run_dir"], device)
    stage1 = FrozenStage1(cfg["stage1"]["run_dir"], device)
    arm_b = cfg.get("arm_b_expert_channel")
    chained = cfg.get("chained_channel")
    if arm_b and chained:
        logger.error("[train_fm] arm_b_expert_channel and chained_channel are "
                     "mutually exclusive")
        raise ValueError("arm_b_expert_channel and chained_channel are "
                         "mutually exclusive")
    arm = "B" if (arm_b or chained) else "A"
    n_post = int(cfg["base"].get("n_post", 16))
    chash = cfg_hash(cfg)
    run_dir = os.path.join(cfg["output"]["root"],
                           f"fm_arm{arm}_s{base.scale}_n"
                           f"{float(base.cfg['cell']['noise_sigma']):.2f}_"
                           f"seed{seed_index}_{chash}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    logger.info("[train_fm] arm=%s seed=%d dir=%s", arm, seed_index, run_dir)

    cell = base.cfg["cell"]
    dk = dict(sigma=base.blur_sigma, scale=base.scale,
              noise_sigma=float(cell["noise_sigma"]))
    bs = int(cfg["train"]["batch_size"])
    tl = DataLoader(make_degraded(cell.get("dataset"), cell["data_root"], split="train", **dk),
                    batch_size=bs, shuffle=False, num_workers=2)
    vl = DataLoader(make_degraded(cell.get("dataset"), cell["data_root"], split="val", **dk),
                    batch_size=bs, shuffle=False, num_workers=2)
    cache_dir = os.path.join(cfg["output"]["root"], "_cache")
    trX, trY, trX0, trIn0 = precompute_split(base, tl, n_post=n_post,
                                             rng_seed=rng_seed,
                                             cache_dir=cache_dir,
                                             split_name="train", device=device)
    vaX, vaY, vaX0, vaIn0 = precompute_split(base, vl, n_post=n_post,
                                             rng_seed=rng_seed,
                                             cache_dir=cache_dir,
                                             split_name="val", device=device)
    trX1, trCond = precompute_stage2(stage1, base, trX, trY, trX0, trIn0,
                                     batch_size=bs, cache_dir=cache_dir,
                                     split_name="train")
    vaX1, vaCond = precompute_stage2(stage1, base, vaX, vaY, vaX0, vaIn0,
                                     batch_size=bs, cache_dir=cache_dir,
                                     split_name="val")
    if chained:
        # W3: frozen STAGE-2 checkpoint applied to x1 (NOT x0) -- the NICE
        # proposal was trained on x1; feeding x0 would be the wrong regime.
        # Distinct split_name keys the cache separately from the x0 caches.
        nice2 = FrozenStage1(chained["run_dir"], device)
        trXN, _ = precompute_stage2(nice2, base, trX, trY, trX1, trIn0,
                                    batch_size=bs, cache_dir=cache_dir,
                                    split_name="train_chained")
        vaXN, _ = precompute_stage2(nice2, base, vaX, vaY, vaX1, vaIn0,
                                    batch_size=bs, cache_dir=cache_dir,
                                    split_name="val_chained")
        trCond = torch.cat([trCond, trXN], dim=1)
        vaCond = torch.cat([vaCond, vaXN], dim=1)
    if arm_b:
        rnvp = FrozenStage1(arm_b["run_dir"], device)
        trXR, _ = precompute_stage2(rnvp, base, trX, trY, trX0, trIn0,
                                    batch_size=bs, cache_dir=cache_dir,
                                    split_name="train")
        vaXR, _ = precompute_stage2(rnvp, base, vaX, vaY, vaX0, vaIn0,
                                    batch_size=bs, cache_dir=cache_dir,
                                    split_name="val")
        trCond = torch.cat([trCond, trXR], dim=1)
        vaCond = torch.cat([vaCond, vaXR], dim=1)
        logger.info("[train_fm] Arm B: xR channel from %s",
                    rnvp.checkpoint_sha256[:12])

    r = cfg["refiner"]
    model = FMRefiner(cond_channels=trCond.size(1),
                      hidden=int(r.get("hidden", 64)),
                      depth=int(r.get("depth", 4)),
                      t_embed_dim=int(r.get("t_embed_dim", 64)),
                      g_max=float(r.get("g_max", 0.5)),
                      g_init=float(r.get("g_init", 0.05))).to(device)
    if cfg["train"].get("budget_form") != "g_dx":
        logger.error("[train_fm] budget_form must be 'g_dx' (locked), got %r",
                     cfg["train"].get("budget_form"))
        raise ValueError("budget_form must be 'g_dx'")
    lam_img = float(cfg["train"].get("lambda_img", 0.1))
    lam_b = float(cfg["train"].get("lambda_budget", 1e-3))
    ch_eps = float(cfg["train"].get("charbonnier_eps", 1e-3))
    grad_clip = float(cfg["train"].get("grad_clip", 5.0))
    patience = int(cfg["train"].get("early_stop_patience", 15))
    epochs = int(cfg["train"]["epochs"])
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    torch.manual_seed(rng_seed)

    psnr_x1 = _psnr(vaX1, vaX); ssim_x1 = _ssim(vaX1, vaX)
    fwd_x1 = _fwd_rel(vaX1, vaY, base.blur_sigma, base.scale)
    logger.info("[train_fm] frozen chain x1 val: psnr=%.3f ssim=%.4f fwd=%.4f",
                psnr_x1, ssim_x1, fwd_x1)

    tload = DataLoader(TensorDataset(trX, trX1, trCond), batch_size=bs,
                       shuffle=True, drop_last=True)
    best_dpsnr = -float("inf"); best_epoch = -1; hist = []; stale = 0
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        run = 0.0; nb = 0; gn_sum = 0.0
        for xt_true, x1b, cb in tload:
            xt_true, x1b, cb = (xt_true.to(device), x1b.to(device),
                                cb.to(device))
            opt.zero_grad()
            t = torch.rand(x1b.size(0), device=device)
            x_t = (1 - t)[:, None, None, None] * x1b \
                + t[:, None, None, None] * xt_true
            v_tgt = xt_true - x1b
            l_fm = _charbonnier(model.velocity(x_t, t, cb), v_tgt, ch_eps)
            x2, dx, g = model(x1b, cb, k_train)
            l_img = _charbonnier(x2, xt_true, ch_eps)
            l_bud = ((g.view(-1, 1, 1, 1) * dx) ** 2).mean()
            loss = l_fm + lam_img * l_img + lam_b * l_bud
            if not torch.isfinite(loss):
                logger.error("[train_fm] non-finite loss")
                raise ValueError("non-finite loss")
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            run += loss.item(); nb += 1; gn_sum += float(gn)

        # frozen check
        fz = max(base.grad_max_abs(), stage1.grad_max_abs())
        if fz != 0.0:
            logger.error("[train_fm] frozen chain has grads! %.3e", fz)
            raise RuntimeError("frozen chain received gradients")

        model.eval()
        dps_k = {}
        for k in _K_SWEEP:
            x2v, dxv, gv = _forward_split(model, vaCond, vaX1, k, bs, device)
            dps_k[k] = _psnr(x2v.clamp(0, 1), vaX) - psnr_x1
            if k == _K_OFFICIAL:
                x2o, dxo, go = x2v.clamp(0, 1), dxv, gv
        ssim_x2 = _ssim(x2o, vaX)
        fwd_x2 = _fwd_rel(x2o, vaY, base.blur_sigma, base.scale)
        gs = GatedUpdate.g_stats(go, model.g_max)
        # conditioning gaps at official K
        perm = torch.randperm(vaCond.size(0))
        def _gap(ch):
            s = vaCond.clone(); s[:, ch] = vaCond[perm, ch]
            x2s, _, _ = _forward_split(model, s, vaX1, _K_OFFICIAL, bs, device)
            return _psnr(x2s.clamp(0, 1), vaX)
        y_gap = (psnr_x1 + dps_k[_K_OFFICIAL]) - _gap(0)
        atr_gap = (psnr_x1 + dps_k[_K_OFFICIAL]) - _gap(2)
        row = {"epoch": ep, "train_loss": run / nb,
               "dpsnr_k1": dps_k[1], "dpsnr_k4": dps_k[4],
               "val_dpsnr": dps_k[8],
               "val_psnr_x1": psnr_x1,
               "val_psnr_x2": psnr_x1 + dps_k[8],
               "val_ssim_x1": ssim_x1, "val_ssim_x2": ssim_x2,
               "val_fwd_rel_x1": fwd_x1, "val_fwd_rel_x2": fwd_x2,
               "g_mean": gs["g_mean"], "g_max_frac": gs["g_max_frac"],
               "y_gap_dpsnr": y_gap, "atr_gap_dpsnr": atr_gap,
               "refiner_grad_norm": gn_sum / nb}
        hist.append(row)
        logger.info("[train_fm] ep %d loss=%.5f dpsnr k1=%+.3f k4=%+.3f "
                    "k8=%+.3f fwd=%.4f g=%.3f ygap=%+.3f atr=%+.3f", ep,
                    row["train_loss"], dps_k[1], dps_k[4], dps_k[8], fwd_x2,
                    gs["g_mean"], y_gap, atr_gap)
        if dps_k[8] > best_dpsnr:
            best_dpsnr = dps_k[8]; best_epoch = ep; stale = 0
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_dpsnr": best_dpsnr}, ckpt_path)
        else:
            stale += 1
            if stale >= patience:
                logger.info("[train_fm] early stop at ep %d (patience %d)",
                            ep, patience)
                break
    if best_epoch < 0:
        logger.error("[train_fm] no keep-best checkpoint")
        raise RuntimeError("no keep-best checkpoint")

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()
    x2v, dxv, gv = _forward_split(model, vaCond, vaX1, _K_OFFICIAL, bs, device)
    x2c = x2v.clamp(0, 1)
    m_dpsnr = _psnr(x2c, vaX) - psnr_x1
    m_ssim = _ssim(x2c, vaX); m_fwd = _fwd_rel(x2c, vaY, base.blur_sigma,
                                               base.scale)
    dps = _psnr_per_sample(x2c, vaX) - _psnr_per_sample(vaX1, vaX)
    gs = GatedUpdate.g_stats(gv, model.g_max)
    perm = torch.randperm(vaCond.size(0))
    def _gapf(ch):
        s = vaCond.clone(); s[:, ch] = vaCond[perm, ch]
        x2s, _, _ = _forward_split(model, s, vaX1, _K_OFFICIAL, bs, device)
        return (psnr_x1 + m_dpsnr) - _psnr(x2s.clamp(0, 1), vaX)
    y_gap = _gapf(0); atr_gap = _gapf(2)
    diag_ok = y_gap > 0 and atr_gap > 0
    hard = (m_dpsnr >= _GATE_HARD and m_fwd <= fwd_x1 + _FWD_TOL
            and m_ssim >= ssim_x1 and diag_ok)
    meaningful = (m_dpsnr >= _GATE_MEANINGFUL and float(dps.mean()) > 0
                  and float((dps > 0).float().mean()) > _GATE_PCT
                  and m_fwd <= fwd_x1 + 1e-3 and diag_ok)

    # csv + plots + grid
    cols = list(hist[0].keys())
    with open(os.path.join(run_dir, "metrics.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for h in hist:
            f.write(",".join(f"{h[c]:.6f}" if isinstance(h[c], float)
                             else str(h[c]) for c in cols) + "\n")
    ep_ax = [h["epoch"] for h in hist]
    def _line(keys, labels, ylab, name):
        plt.figure(figsize=(6, 4))
        for k, l in zip(keys, labels):
            plt.plot(ep_ax, [h[k] for h in hist], label=l)
        plt.xlabel("epoch"); plt.ylabel(ylab); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(run_dir, name), dpi=110); plt.close()
    _line(["dpsnr_k1", "dpsnr_k4", "val_dpsnr"],
          ["ΔPSNR K=1 (~L1 control)", "ΔPSNR K=4", "ΔPSNR K=8 (official)"],
          "ΔPSNR (dB)", "ksweep_curve.png")
    _line(["val_fwd_rel_x1", "val_fwd_rel_x2"], ["fwd(x1)", "fwd(x2)"],
          "fwd_rel", "fwd_rel_curve.png")
    _line(["g_mean", "g_max_frac"], ["g_mean", "g_max_frac"], "gate",
          "gate_curve.png")
    _line(["y_gap_dpsnr", "atr_gap_dpsnr"], ["y_gap", "Aᵀr_gap"], "gap (dB)",
          "conditioning_gap_curve.png")
    k = 8
    import torch.nn.functional as F
    y_up = F.interpolate(vaY[:k], size=(28, 28), mode="nearest")
    rows = [y_up, vaX1[:k], x2c[:k], vaX[:k], (x2c[:k] - vaX1[:k]).abs(),
            (vaX[:k] - vaX1[:k]).abs(), (vaX[:k] - x2c[:k]).abs()]
    labels = ["y_up", "x1", "x2", "x_true", "|x2-x1|", "|xt-x1|", "|xt-x2|"]
    fig, ax = plt.subplots(7, k, figsize=(k + 1, 7))
    for ri in range(7):
        for c in range(k):
            ax[ri, c].imshow(rows[ri][c, 0], cmap="gray", vmin=0, vmax=1)
            ax[ri, c].axis("off")
        ax[ri, 0].text(-0.3, 0.5, labels[ri], transform=ax[ri, 0].transAxes,
                       ha="right", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "recon_grid.png"), dpi=110)
    plt.close(fig)

    write_json(os.path.join(run_dir, "status.json"), {
        "level": 3, "arm": arm, "seed_index": seed_index, "rng_seed": rng_seed,
        "cfg_hash": chash, "k_train": k_train, "k_official": _K_OFFICIAL,
        "base_checkpoint_sha256": base.checkpoint_sha256,
        "stage1_sha256": stage1.checkpoint_sha256,
        "arm_b_source_sha256": (rnvp.checkpoint_sha256 if arm_b else None),
        "chained_channel_source_sha256": (nice2.checkpoint_sha256 if chained
                                          else None),
        "chained_channel_run_dir": (chained["run_dir"] if chained else None),
        "chain_frozen": True, "budget_form": "g_dx",
        "best_epoch": best_epoch,
        "best_val_dpsnr": float(m_dpsnr),
        "best_dpsnr_k1": hist[best_epoch]["dpsnr_k1"],
        "best_dpsnr_k4": hist[best_epoch]["dpsnr_k4"],
        "val_psnr_x1": psnr_x1, "val_psnr_x2": psnr_x1 + float(m_dpsnr),
        "val_ssim_x1": ssim_x1, "val_ssim_x2": float(m_ssim),
        "val_fwd_rel_x1": fwd_x1, "val_fwd_rel_x2": float(m_fwd),
        "g_mean": gs["g_mean"], "g_max_frac": gs["g_max_frac"],
        "y_gap_dpsnr": float(y_gap), "atr_gap_dpsnr": float(atr_gap),
        "pct_samples_improved": float((dps > 0).float().mean()),
        "gate_hard_pass": bool(hard), "gate_meaningful_pass": bool(meaningful),
        "gate_diag_ok": bool(diag_ok),
        "refiner_checkpoint_sha256": sha256_file(ckpt_path),
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": device, "train_time_sec": round(time.time() - t0, 1),
        "status": "done",
    })
    print(f"=== seed{seed_index} summary (FM arm {arm}, K={_K_OFFICIAL}) ===")
    print(f"dPSNR agg {float(m_dpsnr):+.4f}  (K=1 {hist[best_epoch]['dpsnr_k1']:+.4f}"
          f"  K=4 {hist[best_epoch]['dpsnr_k4']:+.4f})")
    print(f"per-sample mean {float(dps.mean()):+.4f} median "
          f"{float(dps.median()):+.4f}  %improved "
          f"{float((dps > 0).float().mean()) * 100:.1f}%")
    print(f"fwd_rel x1 {fwd_x1:.4f} -> x2 {float(m_fwd):.4f}   ssim "
          f"{ssim_x1:.4f} -> {float(m_ssim):.4f}")
    print(f"g_mean {gs['g_mean']:.4f}  g_max_frac {gs['g_max_frac']:.4f}  "
          f"y_gap {float(y_gap):+.4f}  atr_gap {float(atr_gap):+.4f}")
    print(f"HARD: {hard}   MEANINGFUL: {meaningful}   DIAG: {diag_ok}   vs x1")


if __name__ == "__main__":
    main()
