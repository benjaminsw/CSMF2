# =============================================================================
# EXP-SCAFFOLD v2.2 -- experiments.step_5_2.run
# Purpose: Entry point for step 5.2 -- Port to SAR prototype (WP4)
# CONVENTION: NLL = LOSS (lower = better). Artifacts scoped by (step, seed, cfg_hash).
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "2.2"


import argparse
import sys
import traceback
from pathlib import Path

STEP_ID = "5.2"
STEP_NAME = "Port to SAR prototype"
WP = "WP4"
STEP_DIR = Path(__file__).parent


def output_dir(seed, cfg_hash):
    """Per-run output directory -- scoped by (seed, cfg_hash) so reruns never collide."""
    out = STEP_DIR / "results" / ("seed" + str(seed) + "_cfg" + str(cfg_hash))
    out.mkdir(parents=True, exist_ok=True)
    return out


def run(seed, cfg_overrides=None):
    """Training / evaluation for this step. IMPLEMENT ME.

    Expected calls inside:
      * common.seed.set_seed(seed)
      * cfg_hash = common.hashing.config_hash(cfg)
      * out = output_dir(seed, cfg_hash)
      * common.metrics_io.append_row(STEP_DIR, seed=..., epoch=..., nll=..., ...)
      * common.status_io.update_status(STEP_DIR, status="running")
      * common.status_io.update_status(STEP_DIR, seed_done=seed, status="done",
                                       exit_criteria_met=<bool>)
    """
    logger.error("[run] step 5.2: training/eval not implemented")
    raise NotImplementedError(
        "step 5.2: wire training/eval here; call append_row() + update_status() on progress"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="step 5.2 -- Port to SAR prototype")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s :: %(message)s",
    )
    try:
        run(args.seed)
    except NotImplementedError:
        raise
    except Exception:
        logger.error("step 5.2 crashed\n%s", traceback.format_exc())
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
