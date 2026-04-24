# =============================================================================
# EXP-SCAFFOLD v2.2 -- common.logger
# Purpose: unified logger
# CONVENTION: NLL = LOSS (lower = better). Artifacts scoped by (step, seed, cfg_hash).
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "2.2"


def get_logger(name, level=logging.INFO):
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s :: %(message)s"))
        lg.addHandler(h)
        lg.setLevel(level)
    return lg
