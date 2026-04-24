# =============================================================================
# EXP-SCAFFOLD v2.2 -- common.metrics_io
# Purpose: append row to per-step metrics.csv
# CONVENTION: NLL = LOSS (lower = better). Artifacts scoped by (step, seed, cfg_hash).
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "2.2"

import csv
import traceback
from pathlib import Path

FIELDS = ["seed", "epoch", "nll", "residual", "sw2", "es", "neff", "notes"]


def append_row(step_dir, **row):
    missing = [k for k in FIELDS if k not in row]
    if missing:
        logger.error("missing metric fields: %s", missing)
        raise ValueError("missing metric fields: " + str(missing))
    p = Path(step_dir) / "metrics.csv"
    try:
        with p.open("a", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow([row[k] for k in FIELDS])
    except OSError:
        logger.error("metrics append failed at %s\n%s", p, traceback.format_exc())
        raise
