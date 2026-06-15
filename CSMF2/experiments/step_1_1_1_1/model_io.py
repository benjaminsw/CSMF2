# =============================================================================
# STEP-1_1_1_1 v0.1 -- experiments.step_1_1_1_1.model_io
# Purpose: load a trained step_1_1 checkpoint and rebuild expert + conditioner
#          ARCHITECTURE-AGNOSTICALLY -- every build parameter (expert type,
#          dim, h_dim, film/glow/realnvp packs, scale, blur_sigma, ...) is read
#          from the checkpoint dir's report.json["cfg"], never hard-coded.
#          Returns frozen modules (eval, requires_grad=False) ready for MAP.
# CONVENTION: missing files / keys / non-finite -> logger.error + raise.
#             No fallback / mock / dummy. The flow is never retrained here.
# Changelog (NEW in v0.1):
#   * Introduced. build_from_report() mirrors step_1_1/run.py's build block
#     (cond_kwargs / film_kwargs / extra_kwargs / hidden_for_build) so any
#     NICE / RealNVP / NSF checkpoint loads without code changes.
# Update summary:
#   v0.1 reads report.json["cfg"] -> StepCfg, rebuilds cond + expert exactly as
#   training did, loads ckpt.pt state_dicts, freezes everything. Glow is not
#   part of the active roster; it will still load if a Glow ckpt is supplied,
#   but is out of scope for the MAP ablation.
# =============================================================================
from __future__ import annotations
import json
import logging
from pathlib import Path

import torch

from ..step_1_1.config import StepCfg
from ...models.conditioner import Conditioner
from ...models.experts import build_expert

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_1_1_1"


def load_cfg(ckpt_dir: Path) -> StepCfg:
    """Reconstruct the training StepCfg from report.json['cfg']."""
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
    try:
        cfg = StepCfg(**report["cfg"])
    except TypeError as exc:
        logger.error("[model_io] report cfg fields incompatible with StepCfg: "
                     "%s", exc)
        raise
    return cfg


def build_from_report(ckpt_dir: str, device: torch.device):
    """Rebuild (expert, cond, cfg) from a step_1_1 run dir and load weights.
    Mirrors step_1_1/run.py build block exactly. Returns frozen modules."""
    ckpt_dir = Path(ckpt_dir)
    cfg = load_cfg(ckpt_dir)

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

    # ---- freeze ------------------------------------------------------------
    expert.eval(); cond.eval()
    for p in expert.parameters():
        p.requires_grad_(False)
    for p in cond.parameters():
        p.requires_grad_(False)

    logger.info("[model_io] loaded %s expert (dim=%d, scale=%d, blur=%.2f) "
                "from %s -- frozen", cfg.expert, cfg.dim, cfg.scale,
                cfg.blur_sigma, ckpt_dir)
    return expert, cond, cfg
