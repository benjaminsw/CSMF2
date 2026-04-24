# =============================================================================
# EXP-SCAFFOLD v2.2 -- common.seed
# Purpose: deterministic seeding
# CONVENTION: NLL = LOSS (lower = better). Artifacts scoped by (step, seed, cfg_hash).
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "2.2"


def set_seed(*args, **kwargs):
    logger.error("[set_seed] not implemented -- seed torch / numpy / random + set cuDNN deterministic flags")
    raise NotImplementedError("set_seed: seed torch / numpy / random + set cuDNN deterministic flags")
