# =============================================================================
# STEP-1_2 v0.1 -- experiments.step_1_2.config
# Purpose: typed config for one mixture-skeleton run. Frozen dataclass, sha256
#          run_tag. Builds on frozen step_1_1 expert checkpoints; the skeleton
#          combines them under a gate and measures the mechanics (pure NLL).
# CONVENTION: no silent defaults. Every invariant -> logger.error + raise.
#             No fallback / mock / dummy / placeholder.
# Changelog (NEW in v0.1):
#   * Introduced (MIX-SKEL v0.2 spec). Three modes: single / uniform /
#     learned. ckpt_dirs = one trained step_1_1 run dir per expert.
# Update summary:
#   v0.1 ships the skeleton knobs: ckpt_dirs (>=2 frozen experts), mode, tau
#   (gate softmax temperature), gate_hidden / lr_gate / epochs (learned mode
#   only), weight_sum_tol (hard assert). run_tag hashes the field set.
# =============================================================================
from __future__ import annotations
import logging
import hashlib
import json
from dataclasses import dataclass, asdict, field
logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_2"

_VALID_MODE = ("single", "uniform", "learned")


@dataclass(frozen=True)
class MixCfg:
    # frozen experts -- one trained step_1_1 run dir per expert (ckpt.pt+report)
    ckpt_dirs: tuple[str, ...] = ()
    mode: str = "learned"
    # gate (learned mode only)
    tau: float = 1.0
    gate_hidden: int = 128
    lr_gate: float = 1e-3
    epochs: int = 20
    batch_size: int = 128
    # numerical safety
    weight_sum_tol: float = 1e-5
    # bookkeeping
    seed: int = 0
    out_root: str = "./CSMF2/experiments/step_1_2/results"
    log_every: int = 50

    def __post_init__(self):
        if len(self.ckpt_dirs) < 2:
            logger.error("[MixCfg] need >=2 ckpt_dirs for a mixture, got %d",
                         len(self.ckpt_dirs))
            raise ValueError("ckpt_dirs must list >=2 trained step_1_1 run dirs")
        if self.mode not in _VALID_MODE:
            logger.error("[MixCfg] mode must be one of %s, got %r",
                         _VALID_MODE, self.mode)
            raise ValueError(f"mode {self.mode!r} not in {_VALID_MODE}")
        if self.tau <= 0.0:
            logger.error("[MixCfg] tau must be > 0, got %s", self.tau)
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if self.gate_hidden < 1:
            logger.error("[MixCfg] gate_hidden must be >=1, got %s",
                         self.gate_hidden)
            raise ValueError(f"gate_hidden must be >=1, got {self.gate_hidden}")
        if self.mode == "learned":
            if self.lr_gate <= 0.0:
                logger.error("[MixCfg] lr_gate must be > 0, got %s", self.lr_gate)
                raise ValueError(f"lr_gate must be > 0, got {self.lr_gate}")
            if self.epochs < 1:
                logger.error("[MixCfg] epochs must be >=1 for learned, got %s",
                             self.epochs)
                raise ValueError(f"epochs must be >=1 for learned mode")
        if self.weight_sum_tol <= 0.0:
            logger.error("[MixCfg] weight_sum_tol must be > 0, got %s",
                         self.weight_sum_tol)
            raise ValueError(f"weight_sum_tol must be > 0, got {self.weight_sum_tol}")

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def run_tag(self) -> str:
        return f"{self.mode}_K{len(self.ckpt_dirs)}_seed{self.seed}_{self.hash()}"
