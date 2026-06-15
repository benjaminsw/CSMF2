# =============================================================================
# STEP-1_2 v0.1 -- experiments.step_1_2.model_io
# Purpose: load K frozen step_1_1 experts for the mixture skeleton. Reuses the
#          architecture-agnostic loader from step_1_1_1_1 and asserts all
#          experts are mutually compatible (same dim + same degradation A), so
#          their per-expert log_prob values are directly comparable.
# CONVENTION: incompatible experts / missing files -> logger.error + raise.
#             No fallback / mock. Experts are returned frozen (eval, no grad).
# Changelog (NEW in v0.1):
#   * Introduced. load_experts(ckpt_dirs) -> (experts, train_cfgs, ref_cfg).
# Update summary:
#   v0.1 wraps step_1_1_1_1.model_io.build_from_report over a list of run dirs
#   and refuses to mix experts trained on different dim/scale/blur/noise (the
#   forward operator and base measure must match for a valid mixture).
# =============================================================================
from __future__ import annotations
import logging

import torch

from ..step_1_1_1_1.model_io import build_from_report

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_2"

_MUST_MATCH = ("dim", "scale", "blur_sigma", "noise_sigma", "data_root")


def load_experts(ckpt_dirs, device: torch.device):
    """Returns (experts list, train_cfgs list, ref_cfg). All experts frozen.
    Raises if any two experts disagree on dim/scale/blur/noise/data_root."""
    if len(ckpt_dirs) < 2:
        logger.error("[model_io] need >=2 ckpt_dirs, got %d", len(ckpt_dirs))
        raise ValueError("need >=2 ckpt_dirs for a mixture")
    experts, cfgs = [], []
    for d in ckpt_dirs:
        expert, _cond, cfg = build_from_report(d, device)
        experts.append(expert)
        cfgs.append(cfg)
    ref = cfgs[0]
    for d, cfg in zip(ckpt_dirs[1:], cfgs[1:]):
        for key in _MUST_MATCH:
            if getattr(cfg, key) != getattr(ref, key):
                logger.error("[model_io] expert %s disagrees on %s: %s != %s "
                             "(cannot mix incompatible experts)", d, key,
                             getattr(cfg, key), getattr(ref, key))
                raise ValueError(
                    f"expert {d} {key}={getattr(cfg, key)} != "
                    f"ref {key}={getattr(ref, key)}")
    names = [c.expert for c in cfgs]
    logger.info("[model_io] loaded %d frozen experts %s (dim=%d scale=%d "
                "blur=%.2f noise=%.2f)", len(experts), names, ref.dim,
                ref.scale, ref.blur_sigma, ref.noise_sigma)
    return experts, cfgs, ref
