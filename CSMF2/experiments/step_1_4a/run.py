# =============================================================================
# STEP-1_4A v0.3 -- experiments.step_1_4a.run
# Purpose: train one conditional-base expert (NICE-CB / RealNVP-CB / NSF-CB)
#          under NLL + Phase-4 CCR. v0.3 adds per-epoch validation, early
#          stopping on val-NLL plateau/overfit, and KEEP-BEST checkpointing
#          (best-val epoch, gated on base_alive) so the gate sees each expert
#          at its best point -- and saves NLL + CCR loss curves.
# CONVENTION: non-finite / sigma issues / no eligible best ckpt -> raise.
#             No fallback / mock / dummy / pass. Identity-safe base init.
# Exit codes: 0 = trained + report; 1 = crash.
# Changelog (v0.2 -> v0.3):
#   * Per-epoch val NLL; early stopping (patience on rel val improvement);
#     keep-best checkpoint (saves the best-val state, base_alive-gated, NOT
#     the last/overfit one); records train/val NLL + shuffle_gap + h_std
#     curves; plots p_nll_curve + p_ccr_curves. epochs is now a MAX (default
#     150); training usually stops earlier at the val plateau.
# Changelog (v0.1 -> v0.2):
#   * Plots 1-4: base trajectories + reconstruction panel.
# Changelog (NEW in v0.1):
#   * Introduced. NLL via CBExpert.log_prob; CCR shuffle + h_std terms;
#     per-epoch base_diagnostics; ckpt with base_net.
# Update summary:
#   v0.3 makes long runs safe + interpretable: stop at the val-NLL turn, keep
#   the best-val (base-alive) checkpoint, and plot the trajectory so the
#   stopping point is justified rather than a guessed epoch count.
# Update summary:
#   v0.1 swaps the fixed N(0,I) base for ConditionalBase and trains end-to-end.
#   Improvement is attributed to CB only if base_alive (else it's retrain
#   variance) -- the WIN-but-collapsed guard, base edition.
# =============================================================================
from __future__ import annotations
import argparse
import json
import logging
import math
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ...data.degrade import MNISTDegraded, dequantize_logit, inverse_logit
from ...models.conditioner import Conditioner
from ...models.experts import build_expert
from .config import CBCfg
from .cond_base import ConditionalBase, base_diagnostics
from .cb_expert import CBExpert

logger = logging.getLogger("CSMF2.step_1_4a.run")
__version__ = "0.1"
__abbr__ = "STEP-1_4A"


def _build(cfg: CBCfg, device):
    """Build conditioner + expert (mirror Step-1.1) + conditional base, wrap."""
    _y_in = (28 // cfg.scale) * (28 // cfg.scale)
    cond_kwargs = dict(width=cfg.cond_width, h_dim=cfg.h_dim,
                       use_v2=cfg.use_v2_conditioner)
    if cfg.cond_y_residual_alpha_init > 0.0:
        cond_kwargs["y_residual_alpha_init"] = cfg.cond_y_residual_alpha_init
        cond_kwargs["y_input_size"] = _y_in
    cond = Conditioner(**cond_kwargs).to(device)
    film_kwargs = {}
    if cfg.expert in ("nice", "realnvp"):
        film_kwargs = dict(film_hidden=cfg.film_hidden, film_depth=cfg.film_depth,
                           film_use_gelu=cfg.film_use_gelu)
    extra = {}
    if cfg.expert == "realnvp":
        extra.update(n_layers=cfg.realnvp_n_couplings)
    expert = build_expert(cfg.expert, dim=cfg.dim, h_dim=cfg.h_dim,
                          conditioner=cond, hidden=cfg.flow_hidden,
                          use_film=cfg.use_film, **film_kwargs, **extra).to(device)
    base = None
    model = expert
    if cfg.use_conditional_base:
        base = ConditionalBase(
            cfg.dim, cfg.h_dim, mu_hidden=cfg.base_mu_hidden,
            logsigma_hidden=cfg.base_logsigma_hidden,
            logsigma_min=cfg.base_logsigma_min,
            logsigma_max=cfg.base_logsigma_max,
            base_init=cfg.base_init, base_gain=cfg.base_gain).to(device)
        model = CBExpert(expert, base)
    return model, expert, cond, base


def _nll(model, x_flat, y, ldj_deq):
    """-(log p(x|y) + ldj_deq).mean() via the (CB)expert log_prob."""
    lp = model.log_prob(x_flat, y)
    if not torch.isfinite(lp).all():
        logger.error("[run] non-finite log_prob")
        raise RuntimeError("non-finite log_prob")
    return -(lp + ldj_deq).mean()


def _ccr_terms(model, cond, x_flat, y, cfg):
    """Phase-4 CCR: shuffle-gap hinge + h.std penalty (carried from Step 1.1)."""
    extra = x_flat.new_zeros(())
    logs = {}
    if cfg.shuffle_loss_lambda > 0.0:
        perm = torch.randperm(y.size(0), device=y.device)
        lp_real = model.log_prob(x_flat, y)
        lp_shuf = model.log_prob(x_flat, y[perm])
        gap = (lp_real - lp_shuf).mean()
        l_shuf = torch.clamp(cfg.shuffle_loss_margin - gap, min=0.0)
        extra = extra + cfg.shuffle_loss_lambda * l_shuf
        logs["shuffle_gap"] = float(gap)
    if cfg.h_std_penalty_mu > 0.0:
        h = cond(y)
        s = h.std(dim=0, unbiased=False).mean()
        l_hstd = torch.clamp(torch.tensor(cfg.h_std_target, device=y.device) - s,
                             min=0.0) ** 2
        extra = extra + cfg.h_std_penalty_mu * l_hstd
        logs["h_std"] = float(s)
    return extra, logs


def _eval_val_nll(model, val_loader, device, gen):
    model.eval()
    with torch.no_grad():
        nll_sum, n = 0.0, 0
        for x_img, y in val_loader:
            x_img = x_img.to(device); y = y.to(device)
            if x_img.dim() == 3:
                x_img = x_img.unsqueeze(1)
            x_logit, ldj_deq = dequantize_logit(x_img, generator=gen)
            lp = model.log_prob(x_logit.flatten(1), y)
            nll_sum += float((-(lp + ldj_deq)).sum()); n += y.size(0)
    return nll_sum / n


def run(cfg: CBCfg) -> dict:
    out_dir = Path(cfg.out_root) / cfg.run_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    logger.info("STEP-1_4A | tag=%s | cfg=%s", cfg.run_tag(), cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model, expert, cond, base = _build(cfg, device)
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_ds = MNISTDegraded(cfg.data_root, split="train", sigma=cfg.blur_sigma,
                             scale=cfg.scale, noise_sigma=cfg.noise_sigma)
    val_ds = MNISTDegraded(cfg.data_root, split="val", sigma=cfg.blur_sigma,
                           scale=cfg.scale, noise_sigma=cfg.noise_sigma)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # ---- training with per-epoch val + early stopping + keep-best ----------
    import copy
    base_per_epoch = []
    curves = {"train_nll": [], "val_nll": [], "shuffle_gap": [], "h_std": []}
    best = {"val_nll": float("inf"), "epoch": -1, "state": None}
    bad_epochs = 0
    stopped_epoch = cfg.epochs - 1
    for epoch in range(cfg.epochs):
        model.train()
        ep_nll, ep_gap, ep_hstd, nb = 0.0, 0.0, 0.0, 0
        for bi, (x_img, y) in enumerate(train_loader):
            x_img = x_img.to(device); y = y.to(device)
            if x_img.dim() == 3:
                x_img = x_img.unsqueeze(1)
            x_logit, ldj_deq = dequantize_logit(x_img, generator=gen)
            x_flat = x_logit.flatten(1)
            nll = _nll(model, x_flat, y, ldj_deq)
            extra, logs = _ccr_terms(model, cond, x_flat, y, cfg)
            loss = nll + extra
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            ep_nll += float(nll); nb += 1
            ep_gap += float(logs.get("shuffle_gap", 0.0))
            ep_hstd += float(logs.get("h_std", 0.0))
        # per-epoch val + base health
        val_nll = _eval_val_nll(model, val_loader, device, gen)
        alive = True
        if base is not None:
            yb = next(iter(val_loader))[1].to(device)
            bd = base_diagnostics(base, cond, yb, tau_b=cfg.base_tau)
            base_per_epoch.append(bd); alive = bool(bd["base_alive"])
        curves["train_nll"].append(ep_nll / nb)
        curves["val_nll"].append(val_nll)
        curves["shuffle_gap"].append(ep_gap / nb)
        curves["h_std"].append(ep_hstd / nb)
        # keep-best (only checkpoints where the base is alive are eligible)
        # sign-safe improvement: val must beat best by min_delta*|best| (NLL<0)
        prev = best["val_nll"]
        if math.isinf(prev):
            improved = True
        else:
            improved = val_nll < prev - cfg.early_stop_min_delta * abs(prev)
        eligible = (base is None) or alive
        if improved and eligible:
            best = {"val_nll": val_nll, "epoch": epoch,
                    "state": {"expert": copy.deepcopy(expert.state_dict()),
                              "cond": copy.deepcopy(cond.state_dict())}}
            if base is not None:
                best["state"]["base"] = copy.deepcopy(base.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        logger.info("[epoch %d] train_nll=%.2f val_nll=%.2f alive=%s "
                    "best=%.2f@%d bad=%d", epoch, ep_nll / nb, val_nll, alive,
                    best["val_nll"], best["epoch"], bad_epochs)
        if bad_epochs >= cfg.early_stop_patience:
            logger.info("[early-stop] no val improvement for %d epochs -> stop "
                        "at epoch %d (best epoch %d)", bad_epochs, epoch,
                        best["epoch"])
            stopped_epoch = epoch
            break

    if best["state"] is None:
        logger.error("[run] no eligible best checkpoint (val never improved "
                     "with base_alive) -- check base conditioning / tau_b")
        raise RuntimeError("no eligible best checkpoint (base may be dead)")

    # ---- restore BEST weights (not the last/overfit ones) ------------------
    expert.load_state_dict(best["state"]["expert"])
    cond.load_state_dict(best["state"]["cond"])
    if base is not None:
        base.load_state_dict(best["state"]["base"])
    val_nll = best["val_nll"]
    base_final = base_per_epoch[best["epoch"]] if base_per_epoch else None
    base_alive = bool(base_final["base_alive"]) if base_final else False

    # ---- save BEST ckpt + report ------------------------------------------
    ckpt = {"expert": expert.state_dict(), "cond": cond.state_dict()}
    if base is not None:
        ckpt["base"] = base.state_dict()
    torch.save(ckpt, out_dir / "ckpt.pt")
    report = {
        "cfg": cfg.__dict__,
        "expert": cfg.expert,
        "use_conditional_base": cfg.use_conditional_base,
        "val_nll": val_nll,                       # best-epoch val NLL
        "best_epoch": best["epoch"],
        "stopped_epoch": stopped_epoch,
        "early_stopped": stopped_epoch < cfg.epochs - 1,
        "base_final": base_final,
        "base_alive": base_alive,
        "base_per_epoch": base_per_epoch,
        "curves": curves,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    # ---- plots (CBASE v0.2 base/recon + v0.3 NLL/loss curves) --------------
    plots = out_dir / "plots"; plots.mkdir(parents=True, exist_ok=True)
    if base_per_epoch:
        _plot_base_trajectories(base_per_epoch, plots)
    _plot_recon_panel(model, val_loader, cfg, device, gen, plots)
    _plot_loss_curves(curves, best["epoch"], plots)

    logger.info("STEP-1_4A DONE %s best_val_nll=%.2f @epoch %d (stopped %d) "
                "base_alive=%s out=%s", cfg.expert, val_nll, best["epoch"],
                stopped_epoch, base_alive, out_dir)
    return report


def _plot_base_trajectories(base_per_epoch, plots):
    """Plots 1-3: mu-std, logsigma-std, KL(+alive) across epochs."""
    ep = list(range(len(base_per_epoch)))
    mu = [b["mu_std_across_y"] for b in base_per_epoch]
    ls = [b["log_sigma_std_across_y"] for b in base_per_epoch]
    kl = [b["base_effect_magnitude"] for b in base_per_epoch]
    alive = [b["base_alive"] for b in base_per_epoch]
    for vals, name, fn in ((mu, "mu_std_across_y", "p1_mu_std.png"),
                           (ls, "log_sigma_std_across_y", "p2_logsigma_std.png")):
        fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=120)
        ax.plot(ep, vals, marker="o")
        ax.set_xlabel("epoch"); ax.set_ylabel(name)
        ax.set_title(f"Base {name} trajectory"); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(plots / fn, bbox_inches="tight")
        plt.close(fig)
    # plot 3: KL trajectory, colour-coded by base_alive
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=120)
    ax.plot(ep, kl, color="#888", zorder=1)
    col = ["#2ca02c" if a else "#d62728" for a in alive]
    ax.scatter(ep, kl, c=col, zorder=2, label="green=alive / red=dead")
    ax.set_xlabel("epoch"); ax.set_ylabel("KL( N(mu,sigma) || N(0,I) )")
    ax.set_title("Base effect magnitude (KL to N(0,I)) + alive")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(plots / "p3_base_kl_alive.png",
                                    bbox_inches="tight")
    plt.close(fig)
    logger.info("[run] saved plots 1-3 (base trajectories)")


@torch.no_grad()
def _plot_recon_panel(model, val_loader, cfg, device, gen, plots, n_show=8):
    """Plot 4: Original x / Degraded y / x_hat(from y) / error map.
    x_hat = decode(eps, h(y)) with eps ~ N(0,I) (shared across rows)."""
    x_img, y = next(iter(val_loader))
    x_img = x_img.to(device); y = y.to(device)
    if x_img.dim() == 3:
        x_img = x_img.unsqueeze(1)
    n = min(n_show, x_img.size(0))
    x_img = x_img[:n]; y = y[:n]
    h = model.cond(y)
    eps = torch.randn(1, int(model.dim), generator=gen,
                      device=device, dtype=h.dtype).expand(n, -1)
    x_hat = inverse_logit(model.decode(eps, h)).view(n, 1, 28, 28).clamp(0, 1)
    err = (x_hat - x_img).abs()

    cols = [("Original x", x_img), ("Degraded y", y),
            ("x_hat (from y)", x_hat), ("|x_hat - x|", err)]
    fig, axes = plt.subplots(n, 4, figsize=(6.4, 1.5 * n), dpi=120)
    if n == 1:
        axes = axes.reshape(1, 4)
    for r in range(n):
        for c, (label, t) in enumerate(cols):
            ax = axes[r, c]
            img = t[r, 0].detach().cpu().numpy()
            cmap = "magma" if label.startswith("|") else "gray"
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(label, fontsize=9)
    tag = "CB" if cfg.use_conditional_base else "plain"
    fig.suptitle(f"{cfg.expert}-{tag}: reconstruction from y", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(plots / "p4_recon_panel.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("[run] saved plot 4 (reconstruction panel)")


def _plot_loss_curves(curves, best_epoch, plots):
    """CBASE v0.3: train/val NLL (with best-epoch marker) + CCR term curves."""
    ep = list(range(len(curves["val_nll"])))
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=120)
    ax.plot(ep, curves["train_nll"], marker=".", label="train NLL")
    ax.plot(ep, curves["val_nll"], marker=".", label="val NLL")
    if 0 <= best_epoch < len(ep):
        ax.axvline(best_epoch, color="g", ls="--", label=f"best (ep {best_epoch})")
    ax.set_xlabel("epoch"); ax.set_ylabel("NLL")
    ax.set_title("NLL trajectory (early-stop keeps best-val)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(plots / "p_nll_curve.png", bbox_inches="tight")
    plt.close(fig)
    # CCR term curves (shuffle gap should stay above margin; h_std healthy)
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=120)
    ax.plot(ep, curves["shuffle_gap"], marker=".", label="shuffle_gap")
    ax.plot(ep, curves["h_std"], marker=".", label="h_std")
    ax.set_xlabel("epoch"); ax.set_ylabel("CCR term")
    ax.set_title("CCR health: shuffle gap + h_std")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(plots / "p_ccr_curves.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("[run] saved NLL + CCR loss curves")


def _parse_args():
    p = argparse.ArgumentParser(description="Stage 1.4a conditional-base expert")
    p.add_argument("--expert", choices=("nice", "realnvp", "nsf"), required=True)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--noise-sigma", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-conditional-base", action="store_true",
                   help="train the plain (non-CB) baseline for before/after")
    p.add_argument("--no-use-v2-conditioner", action="store_true")
    p.add_argument("--out-root", default="./CSMF2/experiments/step_1_4a/results")
    a = p.parse_args()
    return CBCfg(expert=a.expert, scale=a.scale, noise_sigma=a.noise_sigma,
                 epochs=a.epochs, seed=a.seed,
                 use_conditional_base=not a.no_conditional_base,
                 use_v2_conditioner=not a.no_use_v2_conditioner,
                 out_root=a.out_root)


if __name__ == "__main__":
    cfg = _parse_args()
    try:
        run(cfg); sys.exit(0)
    except Exception:
        logger.error("STEP-1_4A run FAILED\n%s", traceback.format_exc())
        sys.exit(1)
