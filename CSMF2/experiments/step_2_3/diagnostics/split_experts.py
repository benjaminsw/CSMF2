# =============================================================================
# NWS v0.4 -- CSMF2.experiments.step_2_3.diagnostics.split_experts
# Purpose: Step 2.3-NWS Step -1. Split the TRAINED 2.3-A mixture ckpt into three
#          per-expert run dirs (one per expert in expert_set order) using the
#          project's own export_trained_experts. These dirs are what
#          load_experts_trainable / export_experts_rec consume. Run ONCE.
# CONVENTION: No silent fallback. Failure -> logger.error + raise.
# Changelog (NEW in v0.4):
#   * Introduced. Wraps export_trained_experts(mixture_ckpt, mixture_report,
#     warm_dirs[expert_set order], out_root). Asserts the 3 warm dirs are passed
#     in expert_set order (the function enforces length; ORDER is the caller's
#     responsibility -- we log the mapping so a wrong order is visible).
# Update summary:
#   Thin, run-once wrapper. No reconstruction maths here -- it only re-homes the
#   trained per-expert weights so the rest of the diagnostic can load them.
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import os

from CSMF2.experiments.step_2_3.diagnostics.nws_common import (
    EXPERTS, setup_logging, _wire_export_trained_experts,
)

logger = logging.getLogger(__name__)
__version__ = "0.4"
__abbr__ = "NWS"


def split(mixture_dir: str, warm_dirs: list[str], out_root: str, device: str) -> list[str]:
    import torch
    export_trained_experts = _wire_export_trained_experts()

    report_path = os.path.join(mixture_dir, "report.json")
    ckpt_path = os.path.join(mixture_dir, "ckpt.pt")
    for p in (report_path, ckpt_path):
        if not os.path.exists(p):
            logger.error("[split] missing %s", p)
            raise FileNotFoundError(p)

    expert_set = json.loads(open(report_path).read())["expert_set"]
    if list(expert_set) != list(EXPERTS):
        logger.error("[split] report expert_set %s != expected %s", expert_set, EXPERTS)
        raise ValueError("expert_set mismatch")
    if len(warm_dirs) != len(expert_set):
        logger.error("[split] %d warm dirs for %d experts", len(warm_dirs), len(expert_set))
        raise ValueError("warm-dir count != expert_set")

    for name, d in zip(expert_set, warm_dirs):
        logger.info("[split] %-9s <- warm-start %s", name, d)
        if not os.path.exists(os.path.join(d, "report.json")):
            logger.error("[split] warm dir has no report.json: %s", d)
            raise FileNotFoundError(os.path.join(d, "report.json"))

    exported = export_trained_experts(
        ckpt_path, report_path, warm_dirs, out_root,
        device=torch.device(device))
    logger.info("[split] exported %d per-expert dirs:", len(exported))
    for d in exported:
        logger.info("[split]   %s", d)
    return [str(d) for d in exported]


def main() -> None:
    ap = argparse.ArgumentParser(description=f"{__abbr__} v{__version__} split experts")
    ap.add_argument("--ckpt-dir", required=True, help="trained 2.3-A mixture run dir (28eb...)")
    ap.add_argument("--warm-dirs", nargs=3, required=True,
                    help="1.4a CB warm-start dirs in expert_set order: nsf realnvp nice_mix")
    ap.add_argument("--out-root", default=None,
                    help="default: <ckpt-dir>/per_expert")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    setup_logging()
    out_root = args.out_root or os.path.join(args.ckpt_dir, "per_expert")
    split(args.ckpt_dir, args.warm_dirs, out_root, args.device)


if __name__ == "__main__":
    main()
