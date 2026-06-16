# =============================================================================
# STEP-1_3 v0.2 -- experiments.step_1_3.config
# Purpose: typed config for one reconstruction-aware gate run (RECGATE v0.3).
#          Gate-only, experts frozen. Hybrid score
#          score_k = alpha*NLL_norm_k + beta*rec_norm_k, calibration stats
#          frozen, fixed shared z-bank for the rec proxy.
# CONVENTION: no silent defaults. Every invariant -> logger.error + raise.
#             No fallback / mock / dummy / placeholder.
# Changelog (v0.1 -> v0.2):
#   * NEW: rec_norm (per_expert | global), default 'global'. v0.2 RECGATE v0.3
#     fix -- reconstruction residual is in shared y-space units, so it must
#     NOT be standardized per-expert (that inverted the signal in v0.2 runs:
#     rising beta routed AWAY from the best reconstructor). NLL stays
#     per-expert standardized (raw NLL scales differ ~700 nats across flows).
# Changelog (NEW in v0.1):
#   * Introduced. alpha/beta/tau, rec_z_mode (fixed_shared|zero), z_bank_size,
#     z_bank_seed, calib_batches, train_batches, gate_hidden, lr_gate, epochs,
#     min_sigma, weight_sum_tol. gamma reserved (must be 0.0).
# Update summary:
#   v0.2 adds rec_norm. Use rec_norm='global' (shared z-score over all
#   experts+samples) so lower ABSOLUTE residual = better reconstruction;
#   'per_expert' retained only to reproduce the v0.2 debug finding. rec is a
#   DETERMINISTIC PROXY (fixed shared z-bank), not the true posterior mean.
# =============================================================================
from __future__ import annotations
import logging
import hashlib
import json
from dataclasses import dataclass, asdict
logger = logging.getLogger(__name__)
__version__ = "0.2"
__abbr__ = "STEP-1_3"

_VALID_REC = ("fixed_shared", "zero")
_VALID_REC_NORM = ("per_expert", "global")


@dataclass(frozen=True)
class Stage13Cfg:
    ckpt_dirs: tuple[str, ...] = ()
    # hybrid score weights
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.0                 # reserved; must be 0.0 in v0.2
    tau: float = 1.0
    # reconstruction proxy
    rec_z_mode: str = "fixed_shared"   # fixed_shared (S samples) | zero (debug)
    rec_norm: str = "global"           # global (shared z-score) | per_expert (debug)
    z_bank_size: int = 4               # S; shared across all experts
    z_bank_seed: int = 1234
    # calibration + training scope
    calib_batches: int = 20            # train batches for frozen mu/sigma
    train_batches: int = 200           # train batches cached for gate training
    gate_hidden: int = 128
    lr_gate: float = 1e-3
    epochs: int = 20
    batch_size: int = 128
    # numerical safety
    min_sigma: float = 1e-6            # sigma below this -> raise (not clamp)
    weight_sum_tol: float = 1e-5
    # bookkeeping
    seed: int = 0
    out_root: str = "./CSMF2/experiments/step_1_3/results"
    log_every: int = 50

    def __post_init__(self):
        if len(self.ckpt_dirs) < 2:
            logger.error("[Stage13Cfg] need >=2 ckpt_dirs, got %d",
                         len(self.ckpt_dirs))
            raise ValueError("ckpt_dirs must list >=2 trained step_1_1 run dirs")
        if self.alpha < 0.0 or self.beta < 0.0:
            logger.error("[Stage13Cfg] alpha/beta must be >=0, got %s/%s",
                         self.alpha, self.beta)
            raise ValueError("alpha/beta must be >=0")
        if self.gamma != 0.0:
            logger.error("[Stage13Cfg] gamma must be 0.0 in v0.2 (diversity "
                         "term deferred), got %s", self.gamma)
            raise ValueError("gamma must be 0.0 in v0.2 (deferred)")
        if self.tau <= 0.0:
            logger.error("[Stage13Cfg] tau must be > 0, got %s", self.tau)
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if self.rec_z_mode not in _VALID_REC:
            logger.error("[Stage13Cfg] rec_z_mode must be one of %s, got %r",
                         _VALID_REC, self.rec_z_mode)
            raise ValueError(f"rec_z_mode {self.rec_z_mode!r} not in {_VALID_REC}")
        if self.rec_norm not in _VALID_REC_NORM:
            logger.error("[Stage13Cfg] rec_norm must be one of %s, got %r",
                         _VALID_REC_NORM, self.rec_norm)
            raise ValueError(f"rec_norm {self.rec_norm!r} not in {_VALID_REC_NORM}")
        if self.z_bank_size < 1:
            logger.error("[Stage13Cfg] z_bank_size must be >=1, got %s",
                         self.z_bank_size)
            raise ValueError("z_bank_size must be >=1")
        if self.rec_z_mode == "zero" and self.z_bank_size != 1:
            logger.error("[Stage13Cfg] rec_z_mode=zero requires z_bank_size=1, "
                         "got %s", self.z_bank_size)
            raise ValueError("rec_z_mode=zero requires z_bank_size=1")
        for name, v in (("calib_batches", self.calib_batches),
                        ("train_batches", self.train_batches),
                        ("epochs", self.epochs), ("gate_hidden", self.gate_hidden)):
            if v < 1:
                logger.error("[Stage13Cfg] %s must be >=1, got %s", name, v)
                raise ValueError(f"{name} must be >=1, got {v}")
        if self.lr_gate <= 0.0:
            logger.error("[Stage13Cfg] lr_gate must be > 0, got %s", self.lr_gate)
            raise ValueError("lr_gate must be > 0")
        if self.min_sigma <= 0.0 or self.weight_sum_tol <= 0.0:
            logger.error("[Stage13Cfg] min_sigma/weight_sum_tol must be > 0")
            raise ValueError("min_sigma/weight_sum_tol must be > 0")

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def run_tag(self) -> str:
        return (f"recgate_a{self.alpha}_b{self.beta}_{self.rec_z_mode}"
                f"_{self.rec_norm}_S{self.z_bank_size}_seed{self.seed}_"
                f"{self.hash()}")
