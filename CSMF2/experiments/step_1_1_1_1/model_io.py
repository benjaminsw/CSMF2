# =============================================================================
# STEP-1_1_1_1 v0.2 -- experiments.step_1_1_1_1.model_io
# Purpose: load a trained checkpoint and rebuild expert + conditioner
#          ARCHITECTURE-AGNOSTICALLY from report.json["cfg"]. v0.2: also
#          CB-AWARE -- if the checkpoint was trained with a conditional base
#          (cfg.use_conditional_base), build the ConditionalBase, load its
#          weights, and wrap in CBExpert so every downstream stage (1.2, 1.3,
#          2.3) loads the correct model with no per-stage change.
# CONVENTION: missing files/keys, or cfg/ckpt CB mismatch -> logger.error +
#             raise. No fallback / mock. The model is returned frozen.
# Changelog (v0.1 -> v0.2):
#   * CB-aware: load_cfg now returns (StepCfg, raw_cfg_dict) and tolerates
#     extra CB fields (filters to StepCfg fields). build_from_report builds +
#     loads ConditionalBase and wraps CBExpert when use_conditional_base=True;
#     both-ways guard (CB cfg<->'base' key must agree) + explicit type log.
#     Returns the WRAPPED model (frozen incl. base).
# Changelog (NEW in v0.1):
#   * Introduced. build_from_report() mirrors step_1_1/run.py's build block.
# Update summary:
#   v0.2 makes the single shared loader handle plain and conditional-base
#   checkpoints identically, so the Stage 1.3 RECGATE rerun (and Stage 2.3)
#   load CB experts correctly without touching their code. Verify via the log
#   line "loaded <expert> as CBExpert (base=conditional)".
# =============================================================================
from __future__ import annotations
import json
import logging
from pathlib import Path

import torch
from dataclasses import fields as _dc_fields

from ..step_1_1.config import StepCfg
from ...models.conditioner import Conditioner
from ...models.experts import build_expert
from ..step_1_4a.cond_base import ConditionalBase
from ..step_1_4a.cb_expert import CBExpert

logger = logging.getLogger(__name__)
__version__ = "0.2"
__abbr__ = "STEP-1_1_1_1"


def load_cfg(ckpt_dir: Path):
    """Reconstruct (StepCfg, raw_cfg_dict) from report.json['cfg'].
    Tolerates extra fields (e.g. CB checkpoints store CBCfg fields that
    StepCfg does not have): StepCfg is built from the intersecting fields,
    and the full raw dict is returned for CB-specific reads."""
    report_path = ckpt_dir / "report.json"
    if not report_path.exists():
        logger.error("[model_io] no report.json in %s", ckpt_dir)
        raise FileNotFoundError(f"{report_path} not found")
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("[model_io] cannot read %s: %s", report_path, exc)
        raise
    if "cfg" not in report:
        logger.error("[model_io] report.json missing 'cfg' block in %s",
                     ckpt_dir)
        raise KeyError(f"{report_path}: missing 'cfg'")
    raw = report["cfg"]
    known = {f.name for f in _dc_fields(StepCfg)}
    filtered = {k: v for k, v in raw.items() if k in known}
    try:
        cfg = StepCfg(**filtered)
    except TypeError as exc:
        logger.error("[model_io] report cfg incompatible with StepCfg: %s", exc)
        raise
    return cfg, raw


def build_from_report(ckpt_dir: str, device: torch.device):
    """Rebuild (expert, cond, cfg) from a step_1_1 run dir and load weights.
    Mirrors step_1_1/run.py build block exactly. Returns frozen modules."""
    ckpt_dir = Path(ckpt_dir)
    cfg, raw_cfg = load_cfg(ckpt_dir)

    # ---- conditioner (mirror run.py v0.13) --------------------------------
    _y_in = (28 // cfg.scale) * (28 // cfg.scale)
    cond_kwargs: dict = dict(width=cfg.cond_width, h_dim=cfg.h_dim,
                             use_v2=cfg.use_v2_conditioner)
    if cfg.cond_y_residual_alpha_init > 0.0:
        cond_kwargs["y_residual_alpha_init"] = cfg.cond_y_residual_alpha_init
        cond_kwargs["y_input_size"] = _y_in
    cond = Conditioner(**cond_kwargs).to(device)

    # ---- expert (mirror run.py v0.8+) -------------------------------------
    film_kwargs: dict = {}
    if cfg.expert in ("nice", "realnvp", "glow"):
        film_kwargs = dict(film_hidden=cfg.film_hidden,
                           film_depth=cfg.film_depth,
                           film_use_gelu=cfg.film_use_gelu)
    extra_kwargs: dict = {}
    if cfg.expert == "realnvp":
        extra_kwargs.update(n_layers=cfg.realnvp_n_couplings)
    elif cfg.expert == "glow":
        extra_kwargs.update(
            n_layers=cfg.glow_n_steps, s_max=cfg.glow_s_max,
            image_shape=(cfg.glow_image_c, cfg.glow_image_h, cfg.glow_image_w),
            inv1x1_seed_base=cfg.seed, film_gain_init=cfg.glow_film_gain_init)
    hidden_for_build = (cfg.glow_coupling_hidden
                        if cfg.expert == "glow" else cfg.flow_hidden)
    expert = build_expert(cfg.expert, dim=cfg.dim, h_dim=cfg.h_dim,
                          conditioner=cond, hidden=hidden_for_build,
                          use_film=cfg.use_film,
                          **film_kwargs, **extra_kwargs).to(device)

    # ---- load weights ------------------------------------------------------
    ckpt_path = ckpt_dir / "ckpt.pt"
    if not ckpt_path.exists():
        logger.error("[model_io] no ckpt.pt in %s", ckpt_dir)
        raise FileNotFoundError(f"{ckpt_path} not found")
    state = torch.load(ckpt_path, map_location=device)
    for key, module in (("expert", expert), ("cond", cond)):
        if key not in state:
            logger.error("[model_io] ckpt.pt missing '%s' state_dict", key)
            raise KeyError(f"ckpt.pt missing '{key}'")
        module.load_state_dict(state[key])

    # ---- conditional base (CB-aware; v0.2) --------------------------------
    use_cb = bool(raw_cfg.get("use_conditional_base", False))
    has_base = "base" in state
    # both-ways guard: never silently mismatch cfg and checkpoint
    if use_cb and not has_base:
        logger.error("[model_io] cfg.use_conditional_base=True but ckpt.pt has "
                     "no 'base' weights in %s", ckpt_dir)
        raise KeyError("CB cfg but no 'base' state_dict in ckpt.pt")
    if (not use_cb) and has_base:
        logger.error("[model_io] ckpt.pt has 'base' weights but "
                     "cfg.use_conditional_base is False/absent in %s", ckpt_dir)
        raise KeyError("'base' weights present but cfg says plain expert")

    if use_cb:
        base = ConditionalBase(
            cfg.dim, cfg.h_dim,
            mu_hidden=int(raw_cfg["base_mu_hidden"]),
            logsigma_hidden=int(raw_cfg["base_logsigma_hidden"]),
            logsigma_min=float(raw_cfg["base_logsigma_min"]),
            logsigma_max=float(raw_cfg["base_logsigma_max"]),
            base_init=raw_cfg.get("base_init", "zero_mu_unit_sigma"),
            base_gain=float(raw_cfg.get("base_gain", 1.0))).to(device)
        base.load_state_dict(state["base"])
        model = CBExpert(expert, base)
    else:
        model = expert

    # ---- freeze (the wrapped model, so base params are frozen too) --------
    model.eval(); cond.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in cond.parameters():
        p.requires_grad_(False)

    logger.info("[model_io] loaded %s as %s (base=%s, dim=%d scale=%d "
                "blur=%.2f) from %s -- frozen", cfg.expert,
                type(model).__name__,
                "conditional" if use_cb else "N(0,I)",
                cfg.dim, cfg.scale, cfg.blur_sigma, ckpt_dir)
    return model, cond, cfg
