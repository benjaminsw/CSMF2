# =============================================================================
# CSMF2 v0.3 -- scripts.audit_conditioning
# Purpose: standalone diagnostic to confirm whether trained flow experts
#          actually USE the conditioner h. Loads a run's ckpt.pt + report.json,
#          runs four probes on a batch of (x, y) pairs:
#            (1) raw h statistics (mean, std, min, max) -- is h non-degenerate?
#            (2) h variation across different y in the batch -- does
#                conditioner produce DIFFERENT h for DIFFERENT y?
#            (3) sample-level shuffle gap:
#                  log p(x_i | y_i)  vs  log p(x_i | y_pi(i))
#                If close to zero, the flow ignores y.
#            (4) intervention test:
#                  log p(x | y)  vs  log p(x | h := mean(h))
#                Replaces conditioner output by its batch mean; if logp barely
#                changes, conditioning signal does not reach the flow.
# CONVENTION: read-only; never mutates the run. Failures -> raise.
# Changelog (v0.2 -> v0.3):
#   * BUGFIX: probe (5) now reports BOTH across-batch FiLM output std
#     (gamma_batch_std, beta_batch_std) and the legacy across-all-elements
#     std (gamma_total_std, beta_total_std). v0.2 mixed these. Audit on
#     v0.11 Glow showed huge total_std with batch_std ~= 0 -- FiLM had
#     developed per-dim offsets that vary across feat-dim but are identical
#     across batch. Misleading. New 'batch_std' is the right metric.
# Changelog (v0.1 -> v0.2):
#   * NEW probe (5): per-FiLM-head (gamma, beta) statistics.
# Changelog (NEW in v0.1):
#   * Introduced. Drops next to scripts/inspect_runs.py.
# Update summary:
#   Direct test of the hypothesis: "FiLM gamma/beta std > 0 but logp(x|y)
#   does not depend on y". If probes (2) and (3) disagree -- (2) says h
#   varies but (3) says logp is constant in y -- the contradiction is
#   localised to the flow body (couplings ignore h despite FiLM living).
# =============================================================================
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)
__version__ = "0.3"


def load_run(run_dir: Path, device: torch.device):
    from CSMF2.experiments.step_1_1.config import StepCfg
    from CSMF2.models.conditioner import Conditioner
    from CSMF2.models.experts import build_expert

    rep = json.loads((run_dir / "report.json").read_text())
    cfg_dict = rep["cfg"]
    cfg = StepCfg(**cfg_dict)

    cond_kwargs = dict(width=cfg.cond_width, h_dim=cfg.h_dim,
                       use_v2=cfg.use_v2_conditioner)
    if getattr(cfg, "cond_y_residual_alpha_init", 0.0) > 0.0:
        cond_kwargs["y_residual_alpha_init"] = cfg.cond_y_residual_alpha_init
        cond_kwargs["y_input_size"] = (28 // cfg.scale) * (28 // cfg.scale)
    cond = Conditioner(**cond_kwargs).to(device)
    film_kwargs = {}
    extra_kwargs = {}
    if cfg.expert in ("nice", "realnvp", "glow"):
        film_kwargs = dict(film_hidden=cfg.film_hidden,
                           film_depth=cfg.film_depth,
                           film_use_gelu=cfg.film_use_gelu)
    if cfg.expert == "realnvp":
        extra_kwargs.update(n_layers=cfg.realnvp_n_couplings)
    elif cfg.expert == "glow":
        extra_kwargs.update(
            n_layers=cfg.glow_n_steps, s_max=cfg.glow_s_max,
            image_shape=(cfg.glow_image_c, cfg.glow_image_h, cfg.glow_image_w),
            inv1x1_seed_base=cfg.seed)
    hidden = (cfg.glow_coupling_hidden if cfg.expert == "glow"
              else cfg.flow_hidden)
    expert = build_expert(cfg.expert, dim=cfg.dim, h_dim=cfg.h_dim,
                          conditioner=cond, hidden=hidden,
                          use_film=cfg.use_film,
                          **film_kwargs, **extra_kwargs).to(device)

    ckpt = torch.load(run_dir / "ckpt.pt", map_location=device)
    # try common ckpt schemas
    if "expert" in ckpt:
        expert.load_state_dict(ckpt["expert"])
        if "cond" in ckpt:
            cond.load_state_dict(ckpt["cond"])
    elif "state_dict" in ckpt:
        expert.load_state_dict(ckpt["state_dict"])
    else:
        expert.load_state_dict(ckpt)
    expert.eval(); cond.eval()
    return cfg, cond, expert


def get_data(cfg, device, n: int):
    from CSMF2.data.degrade import MNISTDegraded, dequantize_logit
    ds = MNISTDegraded(root=cfg.data_root, split="val",
                       scale=cfg.scale, sigma=cfg.blur_sigma,
                       noise_sigma=cfg.noise_sigma)
    loader = DataLoader(ds, batch_size=n, shuffle=False, num_workers=0)
    x_img, y_img = next(iter(loader))
    x_img = x_img.to(device); y_img = y_img.to(device)
    gen = torch.Generator(device=device).manual_seed(0)
    x_logit, _ = dequantize_logit(x_img, generator=gen)
    return x_logit.flatten(1), y_img


@torch.no_grad()
def audit(run_dir: Path, n: int = 256, device: str = "cpu") -> dict:
    dev = torch.device(device)
    cfg, cond, expert = load_run(run_dir, dev)
    x, y = get_data(cfg, dev, n)

    # ---- (1) raw h statistics ---------------------------------------------
    h = cond(y)                                  # (n, h_dim)
    h_stats = {
        "shape":  list(h.shape),
        "mean":   float(h.mean().item()),
        "std":    float(h.std().item()),
        "min":    float(h.min().item()),
        "max":    float(h.max().item()),
        "abs_mean": float(h.abs().mean().item()),
    }

    # ---- (2) h varies across y --------------------------------------------
    # per-feature std across batch (over examples). Compare to per-feature
    # std AFTER replacing y with y[0] (constant y). The ratio is the
    # "useful variation factor".
    h_std_per_dim   = h.std(dim=0)               # (h_dim,)
    h_const = cond(y[:1].expand_as(y))
    h_const_std = h_const.std(dim=0)             # should be ~0
    h_variation = {
        "std_real_y_mean":    float(h_std_per_dim.mean().item()),
        "std_real_y_min":     float(h_std_per_dim.min().item()),
        "std_const_y_mean":   float(h_const_std.mean().item()),   # ~0 expected
        "ratio_real_to_const": float(
            (h_std_per_dim.mean() / (h_const_std.mean() + 1e-12)).item()),
    }

    # ---- (3) sample-level shuffle gap -------------------------------------
    lp_real = expert.log_prob(x, y)              # (n,)
    perm = torch.randperm(n, device=dev)
    lp_shuf = expert.log_prob(x, y[perm])
    gap = (lp_real - lp_shuf)
    shuffle = {
        "gap_mean":   float(gap.mean().item()),
        "gap_std":    float(gap.std().item()),
        "gap_median": float(gap.median().item()),
        "gap_abs_mean": float(gap.abs().mean().item()),
        "frac_gap_above_1_nat":
            float((gap.abs() > 1.0).float().mean().item()),
    }

    # ---- (4) intervention: replace h by its mean --------------------------
    # We override the conditioner's output temporarily. Easiest: monkey-patch
    # cond.forward to return the same mean tensor regardless of input.
    h_mean_vec = h.mean(dim=0, keepdim=True).expand(n, -1).contiguous()
    original_forward = cond.forward
    def _const_forward(_y):
        return h_mean_vec
    cond.forward = _const_forward
    try:
        lp_const = expert.log_prob(x, y)
    finally:
        cond.forward = original_forward
    delta = (lp_real - lp_const)
    intervention = {
        "delta_mean":   float(delta.mean().item()),
        "delta_std":    float(delta.std().item()),
        "delta_abs_mean": float(delta.abs().mean().item()),
        "interpretation": (
            "if delta_abs_mean << 1 nat, replacing h by its mean barely "
            "changes logp -> flow ignores h"
        ),
    }

    # ---- (5) FiLM output stats per layer (v0.1.1 -- D1 verification) --------
    # For each FiLM head in the expert, compute std of (gamma, beta) across
    # the batch ONLY (not over feat-dim too). gamma_batch_std answers
    # "does FiLM produce DIFFERENT output for DIFFERENT y" -- the right
    # question. gamma_total_std (over both axes) is included as a sanity
    # check: trained FiLM tends to have large total_std even when batch_std
    # is zero, by developing a per-dim constant offset.
    film_stats: list[dict] = []
    h_for_film = cond(y)
    for i, layer in enumerate(expert.layers):
        if hasattr(layer, "film"):
            heads = [("film", layer.film)]
        elif hasattr(layer, "coupling"):
            cpl = layer.coupling
            heads = []
            if hasattr(cpl, "film1"):
                heads.append(("film1", cpl.film1))
            if hasattr(cpl, "film2"):
                heads.append(("film2", cpl.film2))
        else:
            continue
        for hname, fhead in heads:
            g, b = fhead(h_for_film)
            film_stats.append({
                "layer": i,
                "head":  hname,
                "gamma_batch_std": float(g.std(dim=0).mean().item()),
                "beta_batch_std":  float(b.std(dim=0).mean().item()),
                "gamma_total_std": float(g.std().item()),
                "beta_total_std":  float(b.std().item()),
                "gamma_abs":  float(g.abs().mean().item()),
                "beta_abs":   float(b.abs().mean().item()),
            })

    out = {
        "run_dir":     str(run_dir),
        "expert":      cfg.expert,
        "h_stats":     h_stats,
        "h_variation": h_variation,
        "shuffle":     shuffle,
        "intervention": intervention,
        "film_stats":  film_stats,
    }
    return out


def _format(report: dict) -> str:
    L = []
    L.append("=" * 70)
    L.append(f"AUDIT  expert={report['expert']}  run={report['run_dir']}")
    L.append("=" * 70)
    L.append("[1] raw h statistics:")
    for k, v in report["h_stats"].items():
        L.append(f"     {k:14s} = {v}")
    L.append("[2] h variation across y:")
    for k, v in report["h_variation"].items():
        L.append(f"     {k:24s} = {v}")
    L.append("[3] sample-level shuffle gap:")
    for k, v in report["shuffle"].items():
        L.append(f"     {k:20s} = {v}")
    L.append("[4] intervention (h := mean(h)):")
    for k, v in report["intervention"].items():
        L.append(f"     {k:18s} = {v}")
    L.append("[5] per-FiLM-head output stats (batch_std = across-y; total_std = legacy):")
    for fs in report.get("film_stats", []):
        L.append(f"     L{fs['layer']:>2d}.{fs['head']:6s}  "
                 f"g_batch={fs['gamma_batch_std']:.5f}  "
                 f"b_batch={fs['beta_batch_std']:.5f}  "
                 f"g_total={fs['gamma_total_std']:.3f}  "
                 f"b_total={fs['beta_total_std']:.3f}")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True,
                   help="run directory containing ckpt.pt and report.json")
    p.add_argument("-n", type=int, default=256,
                   help="batch size for the audit (default 256)")
    p.add_argument("--device", default="cpu",
                   help="cpu or cuda")
    p.add_argument("--out", default=None,
                   help="optional path to write the audit JSON")
    a = p.parse_args()
    report = audit(Path(a.run), n=a.n, device=a.device)
    text = _format(report)
    print(text)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
