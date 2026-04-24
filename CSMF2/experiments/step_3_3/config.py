# =============================================================================
# EXP-SCAFFOLD v2.2 -- experiments.step_3_3.config
# Purpose: Step-local config for step 3.3 -- Stage C light joint fine-tune
# CONVENTION: NLL = LOSS (lower = better). Artifacts scoped by (step, seed, cfg_hash).
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "2.2"


# Fill in step-specific hyperparameters below. Keep this dict canonical
# (sorted keys, no runtime state) so that config_hash() is stable across runs.
CONFIG = {
    "step_id":   "3.3",
    "step_name": "Stage C light joint fine-tune",
    "wp":        "WP2",
    # --- fill in below ---
    # "batch_size":   128,
    # "lr":           1e-3,
    # "epochs":       50,
    # "lambda_cons":  0.0,
    # "lambda_trans": 0.0,
    # "tau":          1.1,
}


def get_config():
    return dict(CONFIG)
