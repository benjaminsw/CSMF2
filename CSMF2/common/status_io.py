# =============================================================================
# EXP-SCAFFOLD v2.2 -- common.status_io
# Purpose: atomic read/write of per-step status.json
# CONVENTION: NLL = LOSS (lower = better). Artifacts scoped by (step, seed, cfg_hash).
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "2.2"

import json
import os
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_status(step_dir):
    p = Path(step_dir) / "status.json"
    if not p.exists():
        logger.error("status.json not found at %s", p)
        raise FileNotFoundError(str(p))
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.error("failed to read %s\n%s", p, traceback.format_exc())
        raise


def write_status(step_dir, status):
    """Atomic write: temp file in same dir + os.replace."""
    step_dir = Path(step_dir)
    p = step_dir / "status.json"
    status["last_updated"] = _now_iso()
    fd, tmp_path = tempfile.mkstemp(dir=str(step_dir), prefix=".status.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(status, indent=2) + "\n")
        os.replace(tmp_path, p)
    except OSError:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logger.error("atomic write failed for %s\n%s", p, traceback.format_exc())
        raise


def update_status(step_dir, seed_done=None, **fields):
    """Read -> mutate -> atomic write. Appends seed without duplicates."""
    st = read_status(step_dir)
    if seed_done is not None and seed_done not in st["seeds_done"]:
        st["seeds_done"].append(seed_done)
    st.update(fields)
    write_status(step_dir, st)
    return st
