# =============================================================================
# STEP-2_3 v0.3 -- experiments.step_2_3.run_phase_a  (S2.3-A joint trainer)
# Purpose: Stage 2.3-A -- the EXPERT-PRESSURE-FIRST falsifier. Experts UNFROZEN,
#          all-expert MEAN consistency, soft gate present + LIGHTLY regularized
#          (NOT gate-training), L0-normalized NLL + rec, GradNorm hook present
#          but OFF, per-term grad norms + grad_cosine LOGGED. Re-score by the
#          executable GO/STOP verdict (V1 RECARGMIN tier + V2 mixture-vs-NSF-only
#          + G2 Neff/max_weight). See S2.3-PLAN v0.4.
# CONVENTION: no fallback/mock/silent-pass. Every failure path logger.error+raise.
#          argmax is used ONLY for the rec_argmin REPORT (diagnostic), NEVER in
#          the training loss (L_rec = mean_k rec_k, all experts, every batch).
# Changelog (v0.2 -> v0.3, after lr=1e-4/clip=1.0 ALSO diverged at epoch 6):
#   * --rec-exclude CLI flag -> cfg.rec_exclude -> all_expert_consistency(exclude=).
#     Fix D: drop NSF from the rec term (its spline inverse drifts non-invertible
#     under rec pressure). NSF stays full in NLL + gate + mixture.
#   * Forward-path drift probe _drift_probe(): logged every log_every WHILE FINITE
#     (epochs 0..crash), per expert: h_absmax + per-COMPONENT param_absmax
#     (cond/flow/base). This is the evidence the inverse-only NSF probe + the
#     at-crash diagnostic both missed -- they fire when values are ALREADY NaN.
#     param_absmax growing epoch-over-epoch => training DRIFT (-> remove/freeze NSF
#     rec); params sane + disc marginally negative => float32 PRECISION (-> float64
#     inverse). Also emits a compact [2.3-A drift] line to the live log. No
#     training-design change: this run is still PURELY diagnostic.
#     NOTE: crash pinned at epoch 6 across lr 5e-4->1e-4 + clip 5.0->1.0 -- a 5x LR
#     drop did NOT move it, which leans toward fixed-schedule drift over step-size,
#     but the param_absmax trajectory is what confirms it. Do NOT pre-commit a fix.
# Changelog (v0.1 -> v0.2, after 30ep run diverged: NaN h in Conditioner):
#   * NaN-source diagnostic (C): wrap _mixture_nll/all_expert_consistency; on any
#     non-finite blow-up log epoch+it, then probe each expert's cond(y) to NAME
#     the culprit (index/expert/h_finite/h_absmax) BEFORE re-raising. The
#     conditioner's own guard stays the backstop; this adds the context it lacks.
#   * --lr and --grad-clip exposed as CLI flags (were cfg-default only) so each
#     run records its actual stability settings in report.json. Smoke diverged at
#     lr=5e-4/clip=5.0; the A+B stability retry is lr=1e-4 + clip=1.0.
# Changelog (NEW in v0.1):
#   * Introduced. train_phase_a(cfg) -> report dict. Loads K trainable CB experts
#     + soft gate; L0 warmup; joint loop (mixture NLL + mean-all-expert rec +
#     light entropy/load-balance); logs grad norms (incl. anti-collapse) +
#     grad_cosine; writes report.json; computes the GO/STOP verdict.
# Update summary:
#   v0.2 adds failure-attribution + stability CLI knobs after the v0.1 30-epoch
#   run hit non-finite h. No design change (no warmup-freeze, no per-expert LR):
#   the simple A+B (lower LR, tighter clip) stability retry is tried first, with
#   C naming the divergent expert if it recurs.
# =============================================================================
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import torch

from .config import Stage23Cfg
from .model_io import load_experts_trainable, build_gate
from .losses import (all_expert_consistency, gate_entropy_loss,
                     gate_load_balance_loss, stage_2_3a_verdict)
from .gradnorm import L0Normalizer, grad_norm_of, grad_vec, grad_cosine
from ..step_1_2.mixture import (per_expert_logp, mixture_logp, gate_metrics,
                                per_expert_nll)
from ..step_1_3.scores import per_expert_rec, make_z_bank
from ...data.degrade import MNISTDegraded, dequantize_logit

logger = logging.getLogger(__name__)
__version__ = "0.3"
__abbr__ = "STEP-2_3"


def _y_in(scale: int) -> int:
    return (28 // scale) * (28 // scale)


def _mixture_nll(experts, x_flat, y, gate, ldj_deq):
    """-mean log p(x|y) = -mean[ logsumexp_k(log_w_k + lp_k) + ldj_deq ].
    log_w from the SOFT gate; lp_k = expert_k.log_prob(x_flat, y)."""
    lp_ke = per_expert_logp(experts, x_flat, y)            # (B,K), grad-enabled
    log_w = gate.log_weights(y)                            # (B,K)
    logp = mixture_logp(lp_ke, log_w, ldj_deq)             # (B,)
    return -logp.mean(), lp_ke, log_w


def train_phase_a(cfg: Stage23Cfg, ckpt_dirs, device, *,
                  baseline_nsf_only_fwd_rel: float,
                  nll_baseline: float) -> dict:
    """Run 2.3-A. ckpt_dirs: the K CB-expert run dirs (NSF/RealNVP/[NICE-MIX]),
    order MUST match cfg.expert_set. baseline_* : the frozen Stage-1.3 reference
    the mixture must beat (V2) and the N0 NLL it must not critically regress."""
    if cfg.phase != "A":
        logger.error("[2.3-A] train_phase_a called with phase=%s", cfg.phase)
        raise ValueError("train_phase_a requires phase='A'")
    torch.manual_seed(2026 + cfg.seed)

    experts, ref = load_experts_trainable(ckpt_dirs, device, train=True)
    k = len(experts)
    gate = build_gate(_y_in(cfg.scale), k, cfg.gate_hidden, cfg.gate_tau, device)

    # all trainable params: experts (incl. conds + bases) + gate
    params = [p for m in experts for p in m.parameters() if p.requires_grad]
    params += [p for p in gate.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    # grad-norm instrumentation uses TWO surfaces, each measured against the
    # params the term actually trains (Phase-A rec = mean_k rec_k is gate-INDEP):
    #   expert surface -> ||g_nll||, ||g_rec|| (THE 3f ratio lives here)
    #   gate surface   -> ||g_nll||, ||g_entropy||, ||g_loadbalance||
    expert_params = [p for m in experts for p in m.parameters() if p.requires_grad]
    gate_params = [p for p in gate.parameters() if p.requires_grad]

    train_ds = MNISTDegraded(cfg.data_root, split="train", sigma=cfg.blur_sigma,
                             scale=cfg.scale, noise_sigma=cfg.noise_sigma)
    val_ds = MNISTDegraded(cfg.data_root, split="val", sigma=cfg.blur_sigma,
                           scale=cfg.scale, noise_sigma=cfg.noise_sigma)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size,
                                           shuffle=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size)

    # Level 1: L0-normalizer over {nll, rec} (GradNorm OFF in 2.3-A)
    l0 = L0Normalizer(["nll", "rec"], warmup_batches=cfg.l0_warmup_batches)

    # seeded dequant generator (mirrors step_1_3: reproducible dequant noise so
    # the NLL is baseline-comparable). MUST be on the data device -- dequantize_logit
    # builds the uniform noise on x.device, and torch.rand requires the generator's
    # device to match the target tensor's device.
    deq_gen = torch.Generator(device=device).manual_seed(2026 + cfg.seed)

    history = []
    for epoch in range(cfg.epochs):
        for m in experts:
            m.train()
        gate.train()
        for it, batch in enumerate(train_dl):
            x_flat, y, ldj_deq = _unpack(batch, device, deq_gen)

            # ---- NaN-source diagnostic (C): on any non-finite blow-up inside
            # the mixture/conditioner, log epoch/it + WHICH expert before the
            # deep raise propagates. The conditioner's own guard is the backstop;
            # this adds the context (expert index, epoch, iter) it cannot see.
            try:
                nll, lp_ke, log_w = _mixture_nll(experts, x_flat, y, gate, ldj_deq)
                rec_mean, rec_per = all_expert_consistency(
                    experts, [m.cond for m in experts], y,
                    blur_sigma=cfg.blur_sigma, scale=cfg.scale,
                    noise_sigma=cfg.noise_sigma,
                    expert_names=cfg.expert_set, exclude=cfg.rec_exclude)
            except (RuntimeError, ValueError) as exc:
                logger.error("[2.3-A] non-finite blow-up at epoch %d it %d: %s",
                             epoch, it, exc)
                # probe each expert individually to name the culprit
                for k, m in enumerate(experts):
                    try:
                        h_k = m.cond(y)
                        finite = bool(torch.isfinite(h_k).all())
                        logger.error("[2.3-A]   expert[%d]=%s h_finite=%s "
                                     "h_absmax=%.3e", k, cfg.expert_set[k], finite,
                                     float(h_k.detach().abs().max()) if finite else float("nan"))
                    except Exception as e2:
                        logger.error("[2.3-A]   expert[%d]=%s cond() FAILED: %s",
                                     k, cfg.expert_set[k], e2)
                raise

            # ---- L0 warmup: observe raw values, no backward until frozen ----
            if not l0.ready:
                l0.observe({"nll": float(nll.detach()),
                            "rec": float(rec_mean.detach())})
                continue

            terms = l0.normalize({"nll": nll, "rec": rec_mean})
            # main objective (GradNorm OFF -> fixed alpha/beta on normalized terms)
            obj = cfg.alpha_nll * terms["nll"] + cfg.beta_rec * terms["rec"]

            # anti-collapse: OUTSIDE the normalized objective, own fixed coef.
            # want HIGH entropy -> SUBTRACT lambda*H; penalize avg load imbalance.
            H = gate_entropy_loss(log_w)
            lb = gate_load_balance_loss(log_w)
            loss = obj - cfg.entropy_lambda * H + cfg.load_balance_lambda * lb

            if not torch.isfinite(loss).all():
                logger.error("[2.3-A] non-finite loss at epoch %d it %d",
                             epoch, it)
                raise RuntimeError("non-finite loss")

            # ---- grad-norm / grad_cosine LOGGING (v0.1, the 3f lesson) -------
            # Two surfaces, each measured where the term actually trains. Phase-A
            # rec = mean_k rec_k is GATE-INDEPENDENT (it trains experts, not the
            # gate), so the 3f ratio ||g_rec||/||g_nll|| is measured on EXPERT
            # params. Anti-collapse (entropy/load-balance) trains the gate, so its
            # norms are measured on GATE params. Computed on the live graph BEFORE
            # the real backward (retain_graph leaves loss.backward() valid).
            gradlog = None
            if it % cfg.log_every == 0:
                gradlog = _grad_diagnostics(
                    expert_terms={"nll": cfg.alpha_nll * terms["nll"],
                                  "rec": cfg.beta_rec * terms["rec"]},
                    expert_params=expert_params,
                    gate_terms={"nll": cfg.alpha_nll * terms["nll"],
                                "entropy": -cfg.entropy_lambda * H,
                                "load_balance": cfg.load_balance_lambda * lb},
                    gate_params=gate_params)

            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            if it % cfg.log_every == 0:
                driftlog = _drift_probe(experts, y, cfg.expert_set)
                rec = _log_step(epoch, it, nll, rec_mean, rec_per,
                                terms, H, lb, log_w, lp_ke, ldj_deq,
                                gradlog, cfg)
                rec["drift"] = driftlog
                # emit a compact one-line drift summary to the log (visible live,
                # not just in report.json) so divergence is watchable as it builds
                logger.error("[2.3-A drift] ep%d it%d | h_absmax=%s | "
                            "param_absmax=%s", epoch, it,
                            {k: round(v, 2) for k, v in driftlog["h_absmax"].items()},
                            {k: {c: round(x, 1) for c, x in d.items()}
                             for k, d in driftlog["param_absmax"].items()})
                history.append(rec)

    val = _evaluate(experts, gate, val_dl, device, cfg,
                    eval_gen=torch.Generator(device=device).manual_seed(7000 + cfg.seed))
    beats = val["soft_fwd_rel"] < baseline_nsf_only_fwd_rel
    # The canonical RECARGMIN tier (V1) comes from step_1_3a.breakdown run on the
    # EXPORTED per-expert dirs (see export_experts.py) -- not recomputed here, to
    # avoid diverging from the tier definition N3/N8 used. So the trainer writes a
    # DEFERRED verdict carrying V2/G2/NLL (which it CAN compute), and the final
    # GO/STOP is sealed by rerun_verdict() once the tier is known.
    verdict = {
        "verdict": "PENDING_BREAKDOWN",
        "reason": "V1 RECARGMIN tier requires step_1_3a.breakdown on exported "
                  "per-expert dirs; V2/G2/NLL computed below.",
        "V2_beats_nsf_only": bool(beats),
        "V2_soft_fwd_rel": val["soft_fwd_rel"],
        "V2_nsf_only_measured": val["nsf_only_fwd_rel_measured"],
        "V2_baseline_frozen_1_3": baseline_nsf_only_fwd_rel,
        "G2_neff_mean": val["Neff_mean"],
        "G2_max_weight": max(val["mean_weight_per_expert"]),
        "nll_now": val["mixture_nll"], "nll_baseline": nll_baseline,
        "numeric_ok": val["numeric_ok"]}

    report = {"abbr": __abbr__, "version": __version__, "phase": "A",
              "cfg": _cfg_dict(cfg), "expert_set": list(cfg.expert_set),
              "baseline_nsf_only_fwd_rel": baseline_nsf_only_fwd_rel,
              "nll_baseline": nll_baseline, "val": val, "verdict": verdict,
              "history_tail": history[-20:]}
    out = Path(cfg.out_root) / cfg.run_tag()
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    torch.save({"gate": gate.state_dict(),
                "experts": [m.state_dict() for m in experts]},
               out / "ckpt.pt")
    logger.info("[2.3-A] %s -> %s (%s)", cfg.run_tag(), verdict["verdict"],
                verdict["reason"])
    return report


def rerun_verdict(report_path, recargmin_tier: str, cfg: Stage23Cfg):
    """Seal the GO/STOP verdict after step_1_3a.breakdown supplies the canonical
    RECARGMIN tier (V1). Reads the 2.3-A report.json (which already holds V2/G2/
    NLL from training), combines with the tier via stage_2_3a_verdict, writes the
    sealed verdict back. Keeps the tier definition canonical (from breakdown)."""
    rp = Path(report_path)
    report = json.loads(rp.read_text())
    v = report["verdict"]
    sealed = stage_2_3a_verdict(
        recargmin_tier=recargmin_tier,
        beats_nsf_only=bool(v["V2_beats_nsf_only"]),
        neff_mean=float(v["G2_neff_mean"]), max_weight=float(v["G2_max_weight"]),
        nll_now=float(v["nll_now"]), nll_baseline=float(v["nll_baseline"]),
        numeric_ok=bool(v["numeric_ok"]),
        neff_min=cfg.neff_min, max_weight_max=cfg.max_weight_max,
        nll_regression_tol=cfg.nll_regression_tol)
    report["verdict_sealed"] = sealed
    report["val"]["recargmin_tier"] = recargmin_tier
    rp.write_text(json.dumps(report, indent=2))
    logger.info("[2.3-A] sealed verdict %s (%s) tier=%s",
                sealed["verdict"], sealed["reason"], recargmin_tier)
    return sealed


def _unpack(batch, device, gen):
    """Turn a MNISTDegraded (x, y) batch into (x_flat_logit, y, ldj_deq), EXACTLY
    as step_1_3/run.py does: x is pixel-space (B,1,28,28) in [0,1]; the experts
    train/score in LOGIT space, so dequantize_logit(x, generator=gen) -> (x_logit,
    ldj_deq), then flatten. The generator is SEEDED (per run seed) so the dequant
    noise -- and therefore the NLL -- is reproducible and baseline-comparable
    (an unseeded per-batch dequant makes NLL noisy; 1.3 passes gen for this).
    MNISTDegraded returns (x, y) ONLY -- there is no ldj_deq in the batch; it is
    produced here, never silently zeroed."""
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        logger.error("[2.3-A] expected (x, y) batch, got %s", type(batch))
        raise TypeError("MNISTDegraded batch must be (x, y)")
    x_img = batch[0].to(device)                       # (B,1,28,28) pixel [0,1]
    y = batch[1].to(device)                           # (B,1,h,w)
    x_logit, ldj_deq = dequantize_logit(x_img, generator=gen)
    x_flat = x_logit.flatten(1)                       # (B,784) logit space
    return x_flat, y, ldj_deq


@torch.no_grad()
def _drift_probe(experts, y, expert_names) -> dict:
    """Forward-path drift probe (v0.2): logged on each log-step WHILE FINITE, so
    we see what grows BEFORE a NaN crash (the inverse/conditioner guards fire too
    late -- at the crash, when everything is already NaN). The decisive signal is
    param_absmax per COMPONENT: if an expert's weights grow epoch-over-epoch the
    failure is training DRIFT (-> remove/freeze NSF rec); if params stay sane and
    only the spline discriminant goes marginally negative it is float32 PRECISION
    (-> float64 inverse). Components per CBExpert:
        cond = expert.expert.cond   (conditioner)
        base = expert.base          (conditional base)
        flow = expert.expert params MINUS the cond.* submodule (the couplings)
    h_absmax = max|cond(y)| -- grows if the conditioner output is diverging."""
    out = {"h_absmax": {}, "param_absmax": {}}
    for k, m in enumerate(experts):
        name = expert_names[k]
        # h_absmax (guard: cond may already be NaN -> report inf-flag, never raise)
        try:
            h = m.expert.cond(y)
            out["h_absmax"][name] = (float(h.abs().max())
                                     if torch.isfinite(h).all() else float("inf"))
        except Exception:
            out["h_absmax"][name] = float("inf")
        # per-component param_absmax
        cond_ids = {id(p) for p in m.expert.cond.parameters()}
        cond_mx = max((float(p.detach().abs().max())
                       for p in m.expert.cond.parameters()), default=0.0)
        base_mx = max((float(p.detach().abs().max())
                       for p in m.base.parameters()), default=0.0)
        flow_mx = max((float(p.detach().abs().max())
                       for p in m.expert.parameters() if id(p) not in cond_ids),
                      default=0.0)
        out["param_absmax"][name] = {"cond": cond_mx, "flow": flow_mx,
                                     "base": base_mx}
    return out


def _grad_diagnostics(*, expert_terms, expert_params, gate_terms, gate_params):
    """Per-term gradient norms on the correct surface for each term.

    The Phase-A all-expert consistency (rec = mean_k rec_k) is GATE-INDEPENDENT:
    it trains the experts, not the gate. So measuring ||g_rec|| against the gate
    reads ~0 BY DESIGN and tells you nothing. The 3f lesson -- reconstruction
    pressure drowned by NLL -- is about whether rec reaches the EXPERTS, so the
    decisive ratio ||g_rec||/||g_nll|| is computed on EXPERT params. Anti-collapse
    (entropy/load-balance) trains the gate, so its norms use GATE params.

    Returns separate expert/gate norm dicts, grad_cosine(nll,rec) on EXPERT params
    (where both live), and grad_ratio_rec_over_nll on EXPERT params (the 3f signal).
    autograd.grad(retain_graph=True) keeps the caller's loss.backward() valid."""
    expert_params = [p for p in expert_params if p.requires_grad]
    gate_params = [p for p in gate_params if p.requires_grad]
    e_norms, e_vecs = {}, {}
    for name, term in expert_terms.items():
        e_norms[name] = grad_norm_of(term, expert_params)
        e_vecs[name] = grad_vec(term, expert_params)
    g_norms = {}
    for name, term in gate_terms.items():
        try:
            g_norms[name] = grad_norm_of(term, gate_params)
        except RuntimeError:
            g_norms[name] = 0.0          # term genuinely independent of gate
    cos = (grad_cosine(e_vecs["nll"], e_vecs["rec"])
           if ("nll" in e_vecs and "rec" in e_vecs) else None)
    ratio = (e_norms.get("rec", 0.0) / e_norms["nll"]
             if e_norms.get("nll", 0.0) > 0 else None)
    return {"expert_grad_norms": e_norms,        # nll, rec  (THE 3f surface)
            "gate_grad_norms": g_norms,          # nll, entropy, load_balance
            "grad_cosine_nll_rec_experts": cos,
            "grad_ratio_rec_over_nll_experts": ratio}


def _log_step(epoch, it, nll, rec_mean, rec_per, terms, H, lb, log_w, lp_ke,
              ldj_deq, gradlog, cfg):
    """Per-log-step record: losses (raw+norm), gate health, and the REAL per-term
    gradient norms on TWO surfaces (expert_grad_norms incl. the 3f ratio
    rec/nll on EXPERT params; gate_grad_norms for nll/entropy/load_balance on GATE
    params), computed by _grad_diagnostics on the live graph BEFORE backward and
    passed in as `gradlog`. v0.1 instrumentation: grad norms LOGGED from the first
    run (GradNorm itself stays OFF; logging != balancing)."""
    with torch.no_grad():
        gm = gate_metrics(log_w, lp_ke, weight_sum_tol=1e-4)
        pe = per_expert_nll(lp_ke, ldj_deq)
    rec = {"epoch": epoch, "it": it,
           "nll_raw": float(nll.detach()), "rec_raw": float(rec_mean.detach()),
           "nll_norm": float(terms["nll"].detach()),
           "rec_norm": float(terms["rec"].detach()),
           "rec_per_expert": rec_per,
           "gate_entropy": float(H.detach()), "load_balance": float(lb.detach()),
           "Neff_mean": gm["Neff_mean"], "max_weight": max(gm["mean_weight_per_expert"]),
           "mean_weight_per_expert": gm["mean_weight_per_expert"],
           "per_expert_nll_mean": pe["per_expert_nll_mean"]}
    if gradlog is not None:
        rec.update(gradlog)        # grad_norms, grad_cosine_nll_rec, grad_ratio
    return rec


@torch.no_grad()
def _evaluate(experts, gate, val_dl, device, cfg, *, eval_gen):
    """Val pass: mixture NLL, gate health, and the V2 reconstruction residual.

    V2 residual uses step_1_3.scores.per_expert_rec EXACTLY (deterministic-proxy
    rec over a FIXED SHARED z-bank: rec_k = mean_s ||A(decode(z_s,h_k)) - y||^2),
    so it is directly comparable to the frozen Stage-1.3 NSF-only baseline (same
    metric, same z-bank). We then gate-weight per-expert rec into the MIXTURE
    residual: soft_rec = sum_k p_k(y) * rec_k (mean over batch). nsf_only_rec is
    the NSF column alone (NSF index in cfg.expert_set). V2 pass = soft_rec beats
    nsf_only_rec. NOTE: this is the z-bank PROXY residual (the canonical gate
    signal), NOT the mu-mean used by the training consistency term -- matching the
    baseline's definition is what makes the comparison valid."""
    for m in experts:
        m.eval()
    gate.eval()
    dim = experts[0].dim
    z_bank = make_z_bank(dim, cfg.rec_z_bank_size, cfg.rec_z_mode,
                         cfg.rec_z_bank_seed, device, next(experts[0].parameters()).dtype)
    try:
        nsf_idx = list(cfg.expert_set).index("nsf")
    except ValueError:
        logger.error("[2.3-A eval] 'nsf' not in expert_set %s -- V2 baseline "
                     "needs the NSF column", cfg.expert_set)
        raise
    tot_nll, n_batches = 0.0, 0
    neff_acc, weights_acc = 0.0, None
    soft_rec_acc, nsf_rec_acc = 0.0, 0.0
    numeric_ok = True
    for batch in val_dl:
        x_flat, y, ldj_deq = _unpack(batch, device, eval_gen)
        lp_ke = per_expert_logp(experts, x_flat, y)
        log_w = gate.log_weights(y)                        # (B,K)
        tot_nll += float(-mixture_logp(lp_ke, log_w, ldj_deq).mean())
        gm = gate_metrics(log_w, lp_ke, weight_sum_tol=1e-4)
        neff_acc += gm["Neff_mean"]
        w = log_w.exp()                                    # (B,K)
        weights_acc = w.mean(dim=0) if weights_acc is None else weights_acc + w.mean(dim=0)
        # canonical per-expert rec (z-bank proxy), then gate-weight into mixture
        rec_ke = per_expert_rec(experts, y, z_bank,
                                blur_sigma=cfg.blur_sigma, scale=cfg.scale)  # (B,K)
        soft_rec = (w * rec_ke).sum(dim=1)                 # (B,) mixture residual
        soft_rec_acc += float(soft_rec.mean())
        nsf_rec_acc += float(rec_ke[:, nsf_idx].mean())    # NSF-only residual
        n_batches += 1
    nb = max(n_batches, 1)
    return {"mixture_nll": tot_nll / nb,
            "Neff_mean": neff_acc / nb,
            "mean_weight_per_expert": (weights_acc / nb).tolist(),
            "soft_fwd_rel": soft_rec_acc / nb,             # mixture (gate-weighted)
            "nsf_only_fwd_rel_measured": nsf_rec_acc / nb, # self-contained NSF baseline
            "recargmin_tier": "PENDING_BREAKDOWN",
            "numeric_ok": numeric_ok}


def _cfg_dict(cfg: Stage23Cfg) -> dict:
    from dataclasses import asdict
    d = asdict(cfg)
    d["expert_set"] = list(cfg.expert_set)
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dirs", nargs="+", required=True,
                   help="K CB-expert run dirs, ORDER matching --expert-set")
    p.add_argument("--expert-set", nargs="+", required=True)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--noise-sigma", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--baseline-nsf-only-fwd-rel", type=float, required=True,
                   help="frozen Stage-1.3 NSF-only mixture reconstruction rel error")
    p.add_argument("--nll-baseline", type=float, required=True,
                   help="N0 baseline mixture NLL (must not critically regress)")
    p.add_argument("--out-root", default="./CSMF2/experiments/step_2_3/results")
    p.add_argument("--log-every", type=int, default=50,
                   help="log/grad-diagnostic stride; set LOW (e.g. 5) for the "
                        "1-epoch smoke test so grad norms are actually recorded")
    p.add_argument("--lr", type=float, default=5e-4,
                   help="optimizer LR. Joint fine-tune of UNFROZEN experts is "
                        "sensitive; lower (1e-4) if training diverges (NaN h).")
    p.add_argument("--grad-clip", type=float, default=5.0,
                   help="grad-norm clip. Tighten (1.0) alongside lower LR to "
                        "prevent the step that reaches a non-finite region.")
    p.add_argument("--rec-exclude", nargs="*", default=[],
                   help="expert names to EXCLUDE from the rec term (e.g. nsf). "
                        "Fix D: NSF spline-inverse drifts non-invertible under rec "
                        "pressure and needs no rec rescue. Excluded experts stay "
                        "full in NLL+gate+mixture; only the rec gradient skips them.")
    a = p.parse_args()
    cfg = Stage23Cfg(expert_set=tuple(a.expert_set), scale=a.scale,
                     noise_sigma=a.noise_sigma, seed=a.seed, epochs=a.epochs,
                     out_root=a.out_root, phase="A", log_every=a.log_every,
                     lr=a.lr, grad_clip=a.grad_clip,
                     rec_exclude=tuple(a.rec_exclude))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_phase_a(cfg, a.ckpt_dirs, device,
                  baseline_nsf_only_fwd_rel=a.baseline_nsf_only_fwd_rel,
                  nll_baseline=a.nll_baseline)


if __name__ == "__main__":
    main()
