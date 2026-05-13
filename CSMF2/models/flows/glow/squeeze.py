# =============================================================================
# STEP-1_1 v0.1 -- models.flows.glow.squeeze
# Purpose: 2x2 squeeze / unsqueeze. (B, C, H, W) <-> (B, 4C, H/2, W/2).
#          Invertible by construction; logdet = 0.
# CONVENTION: shape errors -> logger.error + raise.
# Changelog (NEW in v0.1):
#   * Introduced as part of glow_improved -> shared merge.
# Update summary:
#   Pure reshape via pixel_unshuffle / pixel_shuffle. Single-level squeeze is
#   the architectural switch that makes Glow's 1x1 conv non-trivial for MNIST
#   (lifts C=1 to C=4).
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_1"

import torch
import torch.nn.functional as F


def squeeze2x2(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        logger.error("[squeeze2x2] expected (B,C,H,W), got %s", tuple(x.shape))
        raise ValueError(f"expected (B,C,H,W), got {tuple(x.shape)}")
    if x.size(2) % 2 or x.size(3) % 2:
        logger.error("[squeeze2x2] H,W must be even, got %dx%d",
                     x.size(2), x.size(3))
        raise ValueError(f"H,W must be even, got {x.size(2)}x{x.size(3)}")
    return F.pixel_unshuffle(x, downscale_factor=2)


def unsqueeze2x2(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        logger.error("[unsqueeze2x2] expected (B,C,H,W), got %s", tuple(x.shape))
        raise ValueError(f"expected (B,C,H,W), got {tuple(x.shape)}")
    if x.size(1) % 4:
        logger.error("[unsqueeze2x2] C must be divisible by 4, got %d", x.size(1))
        raise ValueError(f"C must be divisible by 4, got {x.size(1)}")
    return F.pixel_shuffle(x, upscale_factor=2)
