# =============================================================================
# STEP-1_4B v0.1 -- experiments.step_1_4b.consist_finetune  (RNVP-CONSIST v0.1)
# Purpose: Stage 1.4b-3f. Warm-start the stable image-RealNVP-CB checkpoint and
#          fine-tune RealNVP (couplings + conditional base + conditioner) with
#              L = NLL(x|y) + beta * mean||A(x_hat) - y||^2,   x_hat = decode(mu(h),h)
#          NICE/NSF are NOT in this run (single-expert fine-tune). Tests whether
#          reconstruction PRESSURE moves reconstruction where capacity did not.
# CONVENTION: warm-start asserts ImageCondRealNVP + CBExpert; beta=0 must
#          reproduce the baseline NLL (fail-fast); non-finite -> raise. No
#          fallback / mock / dummy / pass.
# Keep-best: lowest VAL CONSISTENCY among base_alive epochs whose val_NLL has
#          NOT regressed more than nll_guard_nats past the warm-start baseline.
#          (best-NLL would discard the best-reconstruction epoch -- the point.)
# Base-gaming guard: base health (mu/logsigma std+mean, sigma clamp fractions,
#          KL) tracked every epoch so a consistency drop driven by the base
#          cheating (not the flow) is visible. Optional --freeze-base arm.
# Changelog (NEW in v0.1):
#   * Introduced. Warm-start + hybrid loss + keep-best-on-consistency + NLL
#     guard + base-health tracking + beta=0 reproduce-baseline assert.
# Changelog (v0.3 -> v0.4):
#   * Option 1 (units): consistency_term now noise-normalized
#     mean_pixels((A x_hat - y)^2)/sigma_eff^2, sigma_eff=max(noise_sigma,
#     sigma_floor). noise_sigma/sigma_floor/sigma_eff threaded + reported.
#   * Option 2 (strength): sweep TARGET_GRAD_RATIO, not raw beta. beta derived
#     on a warmup batch: beta = target_grad_ratio*||g_NLL||/||g_con||, capped at
#     beta_cap (warns if hit -> consistency grad ~0). target_grad_ratio=0 ->
#     pure-NLL control. Report carries target_grad_ratio, derived_beta,
#     realized_grad_ratio, warmup norms, sigma_eff. out_dir = consist_tgrNN_seedN.
#     CLI --consistency-beta -> --target-grad-ratio (+ --sigma-floor/--beta-cap).
# Changelog (v0.2 -> v0.3):
#   * Gradient diagnostics: per-epoch, on a FIXED val batch, two separate
#     backward passes measure grad_norm_nll, grad_norm_consistency (unweighted),
#     grad_norm_consistency_weighted=beta*., grad_ratio=beta*||g_con||/||g_nll||,
#     grad_cosine=cos(g_nll,g_con). Diagnostics only (NOT used to optimize);
#     written to history + report summary. beta=0 -> grad_ratio 0, cosine NaN.
# Changelog (v0.1 -> v0.2):
#   * beta=0 guard made ONE-SIDED: continued NLL-only fine-tuning legitimately
#     IMPROVES NLL (e.g. -3650 -> -3718 over 40 epochs); only fail if NLL
#     DEGRADES past +tol. The two-sided abs() check wrongly flagged the
#     improvement as 'harness wrong'. beta>0 NLL reference is now the beta=0
#     fine-tuned baseline, NOT the pre-fine-tune warm-start.
# Update summary:
#   v0.1 reuses the CB build/data/dequantize path; only the loss (adds the
#   consistency term) and the keep-best criterion differ from step_1_4a.run.
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

from ...data.degrade import MNISTDegraded, dequantize_logit
from ..step_1_1_1_1.model_io import build_from_report
from ..step_1_4a.cond_base import base_kl
from .consistency_loss import consistency_term

logger = logging.getLogger("CSMF2.step_1_4b.consist_finetune")
__version__ = "0.4"
__abbr__ = "STEP-1_4B"

_BASELINE_NLL = -3650.0          # image-RealNVP-CB seed0 reference (for beta=0 check)
_BETA0_NLL_TOL = 50.0            # beta=0 must not DEGRADE NLL by more than this (one-sided, nats)


def _assert_image_cb(model, ckpt_dir):
    """Warm-start must be a CBExpert wrapping ImageCondRealNVP."""
    inner = getattr(model, "expert", None)
    ok_cb = type(model).__name__ == "CBExpert"
    ok_img = inner is not None and type(inner).__name__ == "ImageCondRealNVP"
    if not (ok_cb and ok_img):
        logger.error("[3f] warm-start must be CBExpert(ImageCondRealNVP); got "
                     "%s(%s) from %s", type(model).__name__,
                     type(inner).__name__ if inner is not None else None, ckpt_dir)
        raise TypeError("3f warm-start is not CBExpert(ImageCondRealNVP)")


@torch.no_grad()
def _base_health(model, h):
    """mu/logsigma std+mean across batch, sigma clamp fractions, mean KL."""
    base = model.base
    mu, logsigma, sigma = base.params(h)
    kl = base_kl(mu, logsigma, sigma).mean().item()
    at_min = (logsigma <= base.logsigma_min + 1e-6).float().mean().item()
    at_max = (logsigma >= base.logsigma_max - 1e-6).float().mean().item()
    return {
        "mu_std_across_y": float(mu.std(dim=0).mean()),
        "mu_mean": float(mu.mean()),
        "log_sigma_std_across_y": float(logsigma.std(dim=0).mean()),
        "log_sigma_mean": float(logsigma.mean()),
        "sigma_min": float(sigma.min()), "sigma_max": float(sigma.max()),
        "base_kl": kl, "fraction_at_sigma_min": at_min,
        "fraction_at_sigma_max": at_max,
        "base_alive": bool(kl > 1.0),     # same alive notion as CB trainer
    }


def _flat_grad(params):
    """Concatenate the .grad of params into one vector (zeros where grad is None)."""
    chunks = []
    for p in params:
        if p.grad is None:
            chunks.append(torch.zeros(p.numel(), device=p.device))
        else:
            chunks.append(p.grad.detach().reshape(-1))
    return torch.cat(chunks)


def _grad_diagnostics(model, params, x_flat, y, ldj_deq, *, beta,
                      blur_sigma, scale, noise_sigma, sigma_floor):
    """Two SEPARATE backward passes on a fixed diagnostic batch to measure the
    NLL gradient and the (unweighted) consistency gradient independently, then:
      grad_norm_nll, grad_norm_consistency,
      grad_norm_consistency_weighted = beta * grad_norm_consistency,
      grad_ratio   = beta * ||g_con|| / ||g_nll||,
      grad_cosine  = cos(g_nll, g_con).
    The normal training step only yields the COMBINED grad, so these are
    computed here (once per epoch) at extra cost; NOT used to optimize.
    Also returns the RAW (beta-free) norms so a target_grad_ratio can derive beta.
    """
    h = model.cond(y)
    # --- grad of NLL alone ---
    model.zero_grad(set_to_none=True)
    nll = -(model.log_prob(x_flat, y) + ldj_deq).mean()
    nll.backward()
    g_nll = _flat_grad(params)
    # --- grad of UNWEIGHTED consistency alone ---
    model.zero_grad(set_to_none=True)
    con = consistency_term(model, model.cond(y), y, blur_sigma=blur_sigma,
                           scale=scale, noise_sigma=noise_sigma,
                           sigma_floor=sigma_floor)
    con.backward()
    g_con = _flat_grad(params)
    model.zero_grad(set_to_none=True)

    n_nll = float(g_nll.norm())
    n_con = float(g_con.norm())
    denom = max(n_nll, 1e-12)
    cos = float(torch.dot(g_nll, g_con) /
                (max(g_nll.norm() * g_con.norm(), torch.tensor(1e-12,
                 device=g_nll.device))))
    return {
        "grad_norm_nll": n_nll,
        "grad_norm_consistency": n_con,
        "grad_norm_consistency_weighted": beta * n_con,
        "grad_ratio": beta * n_con / denom,
        "grad_cosine": cos,
    }


@torch.no_grad()
def _eval(model, loader, device, gen, *, blur_sigma, scale, noise_sigma,
          sigma_floor):
    model.eval()
    nll_acc, con_acc, nb = 0.0, 0.0, 0
    last_h = None
    for x_img, y in loader:
        x_img, y = x_img.to(device), y.to(device)
        if x_img.dim() == 3:
            x_img = x_img.unsqueeze(1)
        x_logit, ldj_deq = dequantize_logit(x_img, generator=gen)
        x_flat = x_logit.flatten(1)
        h = model.cond(y)
        last_h = h
        nll = -(model.log_prob(x_flat, y) + ldj_deq).mean()
        con = consistency_term(model, h, y, blur_sigma=blur_sigma, scale=scale,
                               noise_sigma=noise_sigma, sigma_floor=sigma_floor)
        nll_acc += float(nll); con_acc += float(con); nb += 1
    return nll_acc / nb, con_acc / nb, last_h


def run(ckpt_dir: str, *, target_grad_ratio: float, lr: float, epochs: int,
        seed: int, out_root: str, freeze_base: bool = False,
        nll_guard_nats: float = 100.0, grad_clip: float = 5.0,
        sigma_floor: float = 0.05, beta_cap: float = 1e4) -> dict:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2026 + seed)

    # ---- warm-start (asserts CBExpert(ImageCondRealNVP)) ------------------
    model, cond, cfg = build_from_report(ckpt_dir, device)
    _assert_image_cb(model, ckpt_dir)
    # the full raw cfg dict (image-RealNVP-CB: realnvp_type='image' + image_* +
    # CB fields) is needed verbatim for the output report so RECGATE's loader
    # can rebuild the identical architecture. build_from_report returns only the
    # typed StepCfg, so read the raw dict straight from the warm-start report.
    _ws_report = json.loads((Path(ckpt_dir) / "report.json").read_text())
    raw_cfg = _ws_report["cfg"]
    # un-freeze the warm-started modules (loader froze them); NICE/NSF not present
    for p in model.parameters():
        p.requires_grad_(True)
    for p in cond.parameters():
        p.requires_grad_(True)
    if freeze_base:
        for p in model.base.parameters():
            p.requires_grad_(False)
        logger.info("[3f] base FROZEN (disambiguation arm)")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    blur_sigma, scale = cfg.blur_sigma, cfg.scale

    train_ds = MNISTDegraded(cfg.data_root, split="train", sigma=blur_sigma,
                             scale=scale, noise_sigma=cfg.noise_sigma)
    val_ds = MNISTDegraded(cfg.data_root, split="val", sigma=blur_sigma,
                           scale=scale, noise_sigma=cfg.noise_sigma)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False)
    gen = torch.Generator(device=device).manual_seed(2026 + seed)

    noise_sigma = cfg.noise_sigma
    sigma_eff = max(float(noise_sigma), float(sigma_floor))

    # ---- baseline (pre-fine-tune) val metrics ----------------------------
    nll0, con0, _ = _eval(model, val_loader, device, gen,
                          blur_sigma=blur_sigma, scale=scale,
                          noise_sigma=noise_sigma, sigma_floor=sigma_floor)
    logger.info("[3f] warm-start baseline: val_NLL=%.1f val_consist=%.4f "
                "(noise_sigma=%.4f sigma_floor=%.4f sigma_eff=%.4f)",
                nll0, con0, noise_sigma, sigma_floor, sigma_eff)

    # fixed diagnostic batch (first val batch); reused for warmup beta-derivation
    # AND per-epoch grad diagnostics.
    _dx, _dy = next(iter(val_loader))
    _dx = _dx.to(device); _dy = _dy.to(device)
    if _dx.dim() == 3:
        _dx = _dx.unsqueeze(1)
    _dx_logit, _d_ldj = dequantize_logit(_dx, generator=gen)
    _dx_flat = _dx_logit.flatten(1)

    # ---- Option 2: derive beta from target_grad_ratio --------------------
    # beta = target_grad_ratio * ||g_NLL|| / ||g_consistency||, measured on the
    # warmup batch (UNWEIGHTED consistency grad). target_grad_ratio=0 -> pure-NLL
    # control (beta=0). Cap absurd beta (a sign the consistency grad is ~0).
    if target_grad_ratio == 0.0:
        beta = 0.0
        warm_ratio = {"grad_norm_nll": float("nan"),
                      "grad_norm_consistency": float("nan"), "grad_ratio": 0.0,
                      "grad_cosine": float("nan")}
        logger.info("[3f] target_grad_ratio=0 -> pure-NLL control (beta=0)")
    else:
        warm_ratio = _grad_diagnostics(model, params, _dx_flat, _dy, _d_ldj,
                                       beta=1.0, blur_sigma=blur_sigma,
                                       scale=scale, noise_sigma=noise_sigma,
                                       sigma_floor=sigma_floor)
        gN, gC = warm_ratio["grad_norm_nll"], warm_ratio["grad_norm_consistency"]
        if not (gC > 0.0) or not math.isfinite(gC):
            logger.error("[3f] warmup ||g_consistency||=%.3e invalid -> cannot "
                         "derive beta", gC)
            raise RuntimeError("warmup consistency gradient norm invalid")
        beta = target_grad_ratio * gN / gC
        if beta > beta_cap:
            logger.warning("[3f] derived beta=%.3e exceeds cap %.1e (||g_NLL||="
                           "%.3e ||g_con||=%.3e) -> CAPPING. Consistency grad is "
                           "tiny; pressure may still be weak.", beta, beta_cap,
                           gN, gC)
            beta = beta_cap
        logger.info("[3f] target_grad_ratio=%.3f -> derived beta=%.4g "
                    "(||g_NLL||=%.3e ||g_con||=%.3e)", target_grad_ratio, beta,
                    gN, gC)

    out_dir = Path(out_root) / f"consist_tgr{target_grad_ratio:.2f}_seed{seed}" \
        f"{'_basefrozen' if freeze_base else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best = {"val_consist": float("inf"), "epoch": -1, "val_nll": nll0,
            "state": None}
    history = []
    for ep in range(epochs):
        model.train()
        for x_img, y in train_loader:
            x_img, y = x_img.to(device), y.to(device)
            if x_img.dim() == 3:
                x_img = x_img.unsqueeze(1)
            x_logit, ldj_deq = dequantize_logit(x_img, generator=gen)
            x_flat = x_logit.flatten(1)
            h = model.cond(y)
            nll = -(model.log_prob(x_flat, y) + ldj_deq).mean()
            if beta > 0.0:
                con = consistency_term(model, h, y, blur_sigma=blur_sigma,
                                       scale=scale, noise_sigma=noise_sigma,
                                       sigma_floor=sigma_floor)
                loss = nll + beta * con
            else:
                loss = nll                      # target_grad_ratio=0 == pure NLL
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()

        val_nll, val_con, val_h = _eval(model, val_loader, device, gen,
                                        blur_sigma=blur_sigma, scale=scale,
                                        noise_sigma=noise_sigma,
                                        sigma_floor=sigma_floor)
        health = _base_health(model, val_h)
        # gradient diagnostics on the fixed batch (uses the UNWEIGHTED consistency
        # grad; grad_ratio applies beta). Skipped for beta=0 (no consistency term
        # in the loss -> grad_ratio is 0 by definition, cosine undefined).
        if beta > 0.0:
            grads = _grad_diagnostics(model, params, _dx_flat, _dy, _d_ldj,
                                      beta=beta, blur_sigma=blur_sigma,
                                      scale=scale, noise_sigma=noise_sigma,
                                      sigma_floor=sigma_floor)
        else:
            grads = {"grad_norm_nll": float("nan"), "grad_norm_consistency": float("nan"),
                     "grad_norm_consistency_weighted": 0.0, "grad_ratio": 0.0,
                     "grad_cosine": float("nan")}
        model.train()                     # _grad_diagnostics left grads zeroed
        row = {"epoch": ep, "val_nll": val_nll, "val_consist": val_con,
               **health, **grads}
        history.append(row)
        logger.info("[3f] ep %d val_NLL=%.1f val_consist=%.4f base_kl=%.1f "
                    "alive=%s grad_ratio=%.3f grad_cos=%.3f", ep, val_nll, val_con,
                    health["base_kl"], health["base_alive"],
                    grads["grad_ratio"], grads["grad_cosine"])

        # keep-best on val_consist, guarded by NLL regression + base_alive
        nll_ok = val_nll <= _BASELINE_NLL + nll_guard_nats
        if (health["base_alive"] and nll_ok and val_con < best["val_consist"]):
            best = {"val_consist": val_con, "epoch": ep, "val_nll": val_nll,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}

    # ---- target_grad_ratio=0 control: continued NLL-only fine-tuning is allowed
    #      to IMPROVE density (40 more pure-NLL epochs); only fail if it
    #      meaningfully DEGRADES NLL (one-sided guard -- broken-harness symptom).
    if beta == 0.0:
        final_nll = history[-1]["val_nll"]
        if final_nll > nll0 + _BETA0_NLL_TOL:
            logger.error("[3f] tgr=0 control DEGRADED NLL: val_NLL %.1f vs "
                         "warm-start %.1f (tol +%.0f) -- harness is wrong",
                         final_nll, nll0, _BETA0_NLL_TOL)
            raise RuntimeError("tgr=0 control degraded baseline NLL")
        logger.info("[3f] tgr=0 control OK: val_NLL %.1f (warm-start %.1f); this "
                    "fine-tuned NLL is the reference baseline for tgr>0 rows",
                    final_nll, nll0)

    if best["state"] is None:
        logger.error("[3f] no acceptable epoch (NLL guard / base_alive never "
                     "satisfied) for beta=%.2f seed=%d", beta, seed)
        raise RuntimeError("no keep-best checkpoint passed the guards")

    # restore best weights into the live modules, then save in the SAME layout
    # build_from_report expects (expert/cond/base + report['cfg']), so RECGATE
    # and the 1.3a breakdown can load the fine-tuned model with no special path.
    model.load_state_dict(best["state"])
    inner = model.expert                      # ImageCondRealNVP (owns .cond)
    ckpt = {"expert": inner.state_dict(),
            "cond": inner.cond.state_dict(),
            "base": model.base.state_dict()}
    torch.save(ckpt, out_dir / "ckpt.pt")

    # report.json: carry the warm-start cfg dict forward (image-RealNVP-CB cfg,
    # incl. realnvp_type='image' + image_* + CB fields) so build_from_report
    # rebuilds the identical architecture; annotate the 3f fine-tune on top.
    report = {
        "cfg": raw_cfg,                       # full image-RealNVP-CB cfg (loadable)
        "abbr": "RNVP-CONSIST", "version": __version__,
        "stage": "1.4b-3f",
        "target_grad_ratio": target_grad_ratio,
        "derived_beta": beta, "beta": beta,           # 'beta' kept for back-compat
        "beta_cap": beta_cap,
        "warmup_grad_norm_nll": warm_ratio["grad_norm_nll"],
        "warmup_grad_norm_consistency": warm_ratio["grad_norm_consistency"],
        "noise_sigma": noise_sigma, "sigma_floor": sigma_floor,
        "sigma_eff": sigma_eff,
        "seed": seed, "lr": lr,
        "epochs": epochs, "freeze_base": freeze_base, "xhat_mode": "base_mean",
        "starting_checkpoint": str(ckpt_dir),
        "val_nll_before": nll0, "val_consist_before": con0,
        "val_nll": best["val_nll"], "val_nll_after": best["val_nll"],
        "val_consist_after": best["val_consist"],
        "delta_val_nll": best["val_nll"] - nll0,
        "delta_val_consist": best["val_consist"] - con0,
        "best_epoch": best["epoch"], "base_alive": True,
        # grad diagnostics summary = the LAST epoch's values (steady-state).
        # realized_grad_ratio = the grad_ratio actually achieved (vs target).
        "grad_norm_nll": history[-1]["grad_norm_nll"],
        "grad_norm_consistency": history[-1]["grad_norm_consistency"],
        "grad_norm_consistency_weighted": history[-1]["grad_norm_consistency_weighted"],
        "grad_ratio": history[-1]["grad_ratio"],
        "realized_grad_ratio": history[-1]["grad_ratio"],
        "grad_cosine": history[-1]["grad_cosine"],
        "history": history,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    logger.info("[3f] DONE tgr=%.2f derived_beta=%.4g seed=%d best_epoch=%d "
                "val_NLL=%.1f (delta %.1f) val_consist=%.4f realized_gr=%.3g -> %s",
                target_grad_ratio, beta, seed, best["epoch"], best["val_nll"],
                report["delta_val_nll"], best["val_consist"],
                history[-1]["grad_ratio"], out_dir)
    return report


def _parse():
    p = argparse.ArgumentParser(description="Stage 1.4b-3f RealNVP consistency fine-tune")
    p.add_argument("--ckpt-dir", required=True, help="image-RealNVP-CB warm-start dir")
    p.add_argument("--target-grad-ratio", type=float, required=True,
                   help="desired beta*||g_con||/||g_nll||; beta derived from a "
                        "warmup batch. 0 -> pure-NLL control.")
    p.add_argument("--consistency-lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--freeze-base", action="store_true",
                   help="disambiguation arm: freeze the conditional base")
    p.add_argument("--nll-guard-nats", type=float, default=100.0)
    p.add_argument("--sigma-floor", type=float, default=0.05,
                   help="sigma_eff = max(noise_sigma, sigma_floor); guards /0")
    p.add_argument("--beta-cap", type=float, default=1e4,
                   help="cap derived beta (huge beta => consistency grad ~0)")
    p.add_argument("--out-root", default="./CSMF2/experiments/step_1_4b/results_consistency")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    try:
        run(a.ckpt_dir, target_grad_ratio=a.target_grad_ratio,
            lr=a.consistency_lr, epochs=a.epochs, seed=a.seed,
            out_root=a.out_root, freeze_base=a.freeze_base,
            nll_guard_nats=a.nll_guard_nats, sigma_floor=a.sigma_floor,
            beta_cap=a.beta_cap)
        sys.exit(0)
    except Exception:
        logger.error("RNVP-CONSIST 3f FAILED\n%s", traceback.format_exc())
        sys.exit(1)
