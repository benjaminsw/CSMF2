# =============================================================================
# STEP-1_2 v0.1 -- experiments.step_1_2.run
# Purpose: run ONE mixture mode (single | uniform | learned) over frozen
#          step_1_1 experts under PURE NLL. Records mode NLL + gate health +
#          per-expert diagnostics + (learned) training curves; writes
#          report.json and the 6 core plots. Run all three then aggregate.
# CONVENTION: non-finite logp / weight-sum error -> raise (exit 1). No
#             fallback / mock / dummy / pass. Experts frozen; only the learned
#             gate trains. single mode has NO gate -> gate fields are null.
# Exit codes: 0 = ran + report written; 1 = crash / safety violation.
# Changelog (NEW in v0.1):
#   * Introduced (MIX-SKEL v0.2 core + plot #8 NLL-vs-Neff).
# Update summary:
#   v0.1 evaluates single/uniform directly; trains the global gate for learned
#   (experts frozen). Collapse is recorded as pure_nll_gate_collapse, not a
#   code error. Numerical safety is enforced in mixture.gate_metrics (raises).
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

from ...data.degrade import MNISTDegraded, dequantize_logit
from .config import MixCfg
from .model_io import load_experts
from .mixture import (per_expert_logp, mixture_logp, per_expert_nll,
                      gate_metrics, UniformGate, LearnedGlobalGate)

logger = logging.getLogger("CSMF2.step_1_2.run")
__version__ = "0.1"
__abbr__ = "STEP-1_2"


def _configure_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


def _loaders(ref_cfg, batch_size):
    train_ds = MNISTDegraded(ref_cfg.data_root, split="train",
                             sigma=ref_cfg.blur_sigma, scale=ref_cfg.scale,
                             noise_sigma=ref_cfg.noise_sigma)
    val_ds = MNISTDegraded(ref_cfg.data_root, split="val",
                           sigma=ref_cfg.blur_sigma, scale=ref_cfg.scale,
                           noise_sigma=ref_cfg.noise_sigma)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False))


def _eval_pass(experts, gate, loader, device, gen, cfg):
    """One full pass: mixture NLL + accumulated gate metrics + per-expert NLL.
    Gate may be None (single mode handled by caller)."""
    nll_sum, n = 0.0, 0
    lp_all, logw_all, ldj_all = [], [], []
    for x_img, y in loader:
        x_img = x_img.to(device); y = y.to(device)
        if x_img.dim() == 3:
            x_img = x_img.unsqueeze(1)
        x_logit, ldj_deq = dequantize_logit(x_img, generator=gen)
        x_flat = x_logit.flatten(1)
        lp_ke = per_expert_logp(experts, x_flat, y)
        log_w = gate.log_weights(y)
        lp = mixture_logp(lp_ke, log_w, ldj_deq)
        nll_sum += float((-lp).sum()); n += lp.size(0)
        lp_all.append(lp_ke.detach()); logw_all.append(log_w.detach())
        ldj_all.append(ldj_deq.detach())
    lp_ke = torch.cat(lp_all); log_w = torch.cat(logw_all)
    ldj = torch.cat(ldj_all)
    gm = gate_metrics(log_w, lp_ke, weight_sum_tol=cfg.weight_sum_tol)
    pe = per_expert_nll(lp_ke, ldj)
    return nll_sum / n, gm, pe


def run(cfg: MixCfg) -> dict:
    out_dir = Path(cfg.out_root) / cfg.run_tag()
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    _configure_logging(out_dir)
    logger.info("STEP-1_2 run | tag=%s | cfg=%s", cfg.run_tag(), cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    experts, train_cfgs, ref = load_experts(list(cfg.ckpt_dirs), device)
    K = len(experts)
    expert_names = [c.expert for c in train_cfgs]
    train_loader, val_loader = _loaders(ref, cfg.batch_size)
    y_in = (28 // ref.scale) * (28 // ref.scale)

    curves = {"train_nll": [], "val_nll": [], "Neff": [], "entropy": [],
              "weight_mean": []}

    # ---- build gate + (learned) train -------------------------------------
    if cfg.mode == "single":
        gate = UniformGate(K).to(device)        # used only to compute per-expert
    elif cfg.mode == "uniform":
        gate = UniformGate(K).to(device)
    else:  # learned
        gate = LearnedGlobalGate(y_in, K, cfg.gate_hidden, cfg.tau).to(device)
        opt = torch.optim.Adam(gate.parameters(), lr=cfg.lr_gate)
        for epoch in range(cfg.epochs):
            gate.train()
            run_nll, n = 0.0, 0
            for x_img, y in train_loader:
                x_img = x_img.to(device); y = y.to(device)
                if x_img.dim() == 3:
                    x_img = x_img.unsqueeze(1)
                x_logit, ldj_deq = dequantize_logit(x_img, generator=gen)
                x_flat = x_logit.flatten(1)
                lp_ke = per_expert_logp(experts, x_flat, y)
                log_w = gate.log_weights(y)
                lp = mixture_logp(lp_ke, log_w, ldj_deq)
                loss = (-lp).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                run_nll += float((-lp).sum()); n += lp.size(0)
            gate.eval()
            v_nll, gm, _ = _eval_pass(experts, gate, val_loader, device,
                                      gen, cfg)
            curves["train_nll"].append(run_nll / n)
            curves["val_nll"].append(v_nll)
            curves["Neff"].append(gm["Neff_mean"])
            curves["entropy"].append(gm["gate_entropy"])
            curves["weight_mean"].append(gm["mean_weight_per_expert"])
            logger.info("[learned] epoch %d train_nll=%.2f val_nll=%.2f "
                        "Neff=%.3f entropy=%.3f", epoch, run_nll / n, v_nll,
                        gm["Neff_mean"], gm["gate_entropy"])

    # ---- final val pass ----------------------------------------------------
    gate.eval()
    val_nll, gm, pe = _eval_pass(experts, gate, val_loader, device, gen, cfg)

    # ---- mode NLL ----------------------------------------------------------
    if cfg.mode == "single":
        mode_nll = pe["single_nll"]               # best single expert; no gate
        gate_block = None                         # single has NO gate
    else:
        mode_nll = val_nll
        gate_block = gm

    # ---- pass condition + collapse flag ------------------------------------
    collapse = None
    pass_cond = None
    if cfg.mode == "learned":
        single_nll = pe["single_nll"]
        # 'uniform_nll' is not known in this run; aggregate_modes does the
        # cross-mode check. Here we record the within-run signals.
        approx_single = abs(mode_nll - single_nll) <= 0.02 * abs(single_nll)
        collapse = {
            "Neff_final": gm["Neff_mean"],
            "collapsed_onto_best": bool(approx_single and gm["Neff_mean"] < 1.5),
            "verdict": ("pure_nll_gate_collapse: FAIL (expected -- motivates "
                        "Stage 1.3)" if gm["Neff_mean"] < 1.5 else
                        "gate retained diversity (unexpected under pure NLL)"),
        }
        pass_cond = {"learned_approx_single_best": bool(approx_single)}

    report = {
        "mix_cfg": cfg.__dict__,
        "expert_names": expert_names,
        "mode": cfg.mode,
        "mode_nll": mode_nll,
        "gate": gate_block,                        # null for single
        "per_expert": pe,
        "curves": curves if cfg.mode == "learned" else None,
        "collapse": collapse,
        "pass_cond": pass_cond,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    # ---- plots (6 core) ----------------------------------------------------
    _plot_per_expert_nll(pe, expert_names, plots / "per_expert_nll.png")
    if gate_block is not None:
        _plot_gate_usage(gate_block, expert_names, plots / "gate_usage.png")
    if cfg.mode == "learned":
        _plot_curve(curves["Neff"], "Neff", plots / "neff_trajectory.png")
        _plot_curve(curves["entropy"], "gate entropy",
                    plots / "entropy_trajectory.png")
        _plot_nll_vs_neff(curves, plots / "nll_vs_neff.png")
    logger.info("STEP-1_2 run DONE mode=%s nll=%.2f%s out=%s", cfg.mode,
                mode_nll,
                (f" Neff={gm['Neff_mean']:.3f}" if gate_block else ""),
                out_dir)
    return report


# ---- plot helpers ----------------------------------------------------------
def _plot_per_expert_nll(pe, names, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=120)
    ax.bar(names, pe["per_expert_nll_mean"],
           yerr=pe["per_expert_nll_std"], color="#1f77b4", capsize=4)
    ax.set_ylabel("per-expert NLL (mean ± std)")
    ax.set_title("Per-expert NLL -- why one expert dominates")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    logger.info("[run] saved %s", path)


def _plot_gate_usage(gm, names, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=120)
    ax.bar(names, gm["mean_weight_per_expert"],
           yerr=gm["weight_std_per_expert"], color="#2ca02c", capsize=4)
    ax.set_ylabel("mean gate weight"); ax.set_ylim(0, 1)
    ax.set_title("Gate usage (mean weight per expert)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    logger.info("[run] saved %s", path)


def _plot_curve(vals, ylabel, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=120)
    ax.plot(range(len(vals)), vals, marker="o")
    ax.set_xlabel("epoch"); ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} trajectory"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    logger.info("[run] saved %s", path)


def _plot_nll_vs_neff(curves, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=120)
    sc = ax.scatter(curves["Neff"], curves["val_nll"],
                    c=range(len(curves["Neff"])), cmap="viridis")
    ax.set_xlabel("Neff (gate diversity)"); ax.set_ylabel("val NLL")
    ax.set_title("NLL vs Neff -- NLL improves AS the gate collapses")
    fig.colorbar(sc, ax=ax, label="epoch")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    logger.info("[run] saved %s", path)


def _parse_args():
    p = argparse.ArgumentParser(description="Stage 1.2 mixture skeleton")
    p.add_argument("--ckpt-dirs", nargs="+", required=True,
                   help="one trained step_1_1 run dir per expert (>=2)")
    p.add_argument("--mode", choices=("single", "uniform", "learned"),
                   default="learned")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--gate-hidden", type=int, default=128)
    p.add_argument("--lr-gate", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-root", default="./CSMF2/experiments/step_1_2/results")
    a = p.parse_args()
    return MixCfg(ckpt_dirs=tuple(a.ckpt_dirs), mode=a.mode, tau=a.tau,
                  gate_hidden=a.gate_hidden, lr_gate=a.lr_gate,
                  epochs=a.epochs, batch_size=a.batch_size, seed=a.seed,
                  out_root=a.out_root)


if __name__ == "__main__":
    cfg = _parse_args()
    try:
        run(cfg)
        sys.exit(0)
    except Exception:
        logger.error("STEP-1_2 run FAILED\n%s", traceback.format_exc())
        sys.exit(1)
