# =============================================================================
# EXP-SCAFFOLD v2.2 -- common.hashing
# Purpose: config + git-SHA hashing for reproducibility
# CONVENTION: NLL = LOSS (lower = better). Artifacts scoped by (step, seed, cfg_hash).
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "2.2"


def config_hash(*args, **kwargs):
    logger.error("[config_hash] not implemented -- sha256(canonical-json(cfg)) -> first 12 hex chars")
    raise NotImplementedError("config_hash: sha256(canonical-json(cfg)) -> first 12 hex chars")


def git_sha(*args, **kwargs):
    logger.error("[git_sha] not implemented -- git rev-parse HEAD; fallback 'DIRTY-<timestamp>' if unavailable")
    raise NotImplementedError("git_sha: git rev-parse HEAD; fallback 'DIRTY-<timestamp>' if unavailable")
