# =============================================================================
# STEP-1_1_1_1 v0.1 -- experiments.step_1_1_1_1.config
# Purpose: typed config for one MAP / IS+MAP latent-refinement run. Frozen
#          dataclass, sha256 hash baked into run_tag for reproducibility.
#          This experiment is INDEPENDENT of step_1_1: it loads a trained
#          checkpoint via ckpt_dir and never retrains the flow.
# CONVENTION: no silent defaults. Every invariant -> logger.error + raise.
#             No fallback / mock / dummy / placeholder.
# Changelog (NEW in v0.1):
#   * Introduced. Core three-arm latent refinement: random_map / is_only /
#     is_map. Architecture-agnostic -- expert type is read from the loaded
#     checkpoint's report.json, never hard-coded here.
# Update summary:
#   v0.1 ships the MAP-ABL core knobs: ckpt_dir (required interface), init arm,
#   K candidates, map_steps S, lr_z, lambda_prior, sigma_y (IS weight scale,
#   informational), n_images, seed. run_tag hashes the full field set so
#   different arms/seeds never share a results folder.
# =============================================================================
from __future__ import annotations
import logging
import hashlib
import json
from dataclasses import dataclass, asdict
logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_1_1_1"

_VALID_INIT = ("random_map", "is_only", "is_map")


@dataclass(frozen=True)
class MAPCfg:
    # interface -- the trained flow to refine (required; no default that works)
    ckpt_dir: str = ""                # dir containing ckpt.pt + report.json
    # arm
    init: str = "is_map"              # one of _VALID_INIT
    K: int = 64                       # IS candidate count (ignored by random_map)
    map_steps: int = 100              # MAP optimisation steps (0 for is_only)
    lr_z: float = 0.05                # Adam step size on the latent z
    lambda_prior: float = 1.0e-3      # prior weight in residual + lambda*||z||^2
    sigma_y: float = 0.1              # IS weight scale (informational only)
    conv_tol: float = 1.0e-4          # rel-change convergence flag threshold
    # data / scope
    n_images: int = 20                # how many y to refine
    seed: int = 0
    # bookkeeping
    out_root: str = "./CSMF2/experiments/step_1_1_1_1/results"
    log_every: int = 10

    def __post_init__(self):
        if not self.ckpt_dir:
            logger.error("[MAPCfg] ckpt_dir is required (no default flow)")
            raise ValueError("ckpt_dir must be a non-empty path to a step_1_1 "
                             "run dir containing ckpt.pt + report.json")
        if self.init not in _VALID_INIT:
            logger.error("[MAPCfg] init must be one of %s, got %r",
                         _VALID_INIT, self.init)
            raise ValueError(f"init {self.init!r} not in {_VALID_INIT}")
        if self.init in ("is_only", "is_map") and self.K < 2:
            logger.error("[MAPCfg] K must be >=2 for IS arms, got %s", self.K)
            raise ValueError(f"K must be >=2 for {self.init}, got {self.K}")
        if self.init == "is_only" and self.map_steps != 0:
            logger.error("[MAPCfg] is_only requires map_steps=0, got %s",
                         self.map_steps)
            raise ValueError("is_only requires map_steps=0 (selection only)")
        if self.init in ("random_map", "is_map") and self.map_steps < 1:
            logger.error("[MAPCfg] %s requires map_steps>=1, got %s",
                         self.init, self.map_steps)
            raise ValueError(f"{self.init} requires map_steps>=1")
        if self.lr_z <= 0.0:
            logger.error("[MAPCfg] lr_z must be > 0, got %s", self.lr_z)
            raise ValueError(f"lr_z must be > 0, got {self.lr_z}")
        if self.lambda_prior < 0.0:
            logger.error("[MAPCfg] lambda_prior must be >=0, got %s",
                         self.lambda_prior)
            raise ValueError(f"lambda_prior must be >=0, got {self.lambda_prior}")
        if self.sigma_y <= 0.0:
            logger.error("[MAPCfg] sigma_y must be > 0, got %s", self.sigma_y)
            raise ValueError(f"sigma_y must be > 0, got {self.sigma_y}")
        if self.conv_tol < 0.0:
            logger.error("[MAPCfg] conv_tol must be >=0, got %s", self.conv_tol)
            raise ValueError(f"conv_tol must be >=0, got {self.conv_tol}")
        if self.n_images < 1:
            logger.error("[MAPCfg] n_images must be >=1, got %s", self.n_images)
            raise ValueError(f"n_images must be >=1, got {self.n_images}")

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def run_tag(self) -> str:
        return (f"{self.init}_K{self.K}_S{self.map_steps}_"
                f"seed{self.seed}_{self.hash()}")
