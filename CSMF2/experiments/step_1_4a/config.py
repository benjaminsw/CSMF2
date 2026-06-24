# =============================================================================
# STEP-1_4A v0.3 -- experiments.step_1_4a.config
# Purpose: typed config for one conditional-base expert training run. Extends
#          the Step-1.1 StepCfg with the conditional-base knobs. Frozen
#          dataclass, sha256 run_tag.
# CONVENTION: no silent defaults. Every invariant -> logger.error + raise.
#             No fallback / mock / dummy / placeholder.
# Changelog (v0.2 -> v0.3):
#   * Accept expert='nice_mix' (NCP-N8 additive NICE + fixed-perm ablation) in
#     the roster check. NEW field nice_n_layers (default 4, invariant >=1) ->
#     NICE/nice_mix coupling depth; absorbed by hash()/run_tag() so depth sweeps
#     land in distinct dirs. Default 4 = unchanged for existing nice runs.
# Changelog (v0.1 -> v0.2):
#   * early_stop_min_delta default 0.005 -> 0.001 (0.5% was too coarse on the
#     NLL scale: ~16 nats at NLL=-3200, so real single-digit-nat gains were
#     ignored by keep-best). 0.001 (~3 nats) counts real gains but still skips
#     sub-nat noise; 0.0 rejected (chases noise, defeats the plateau detector).
#     NOTE: runs trained at 0.005 (the original depth sweep) are NOT perfectly
#     comparable to new 0.001 runs -- keep the old sweep as historical evidence,
#     or retrain the adopted candidate at 0.001.
#   * Depth-sweep support: realnvp_n_couplings exposed + invariant (>=1);
#     run_tag embeds _cN for RealNVP so depths land in distinct dirs.
# Changelog (NEW in v0.1):
#   * Introduced. use_conditional_base, base_mu_hidden, base_logsigma_hidden,
#     base_logsigma_min/max, base_init, base_gain, base_tau (alive threshold).
#     CB applied to ALL experts (declared) to preserve the fair comparison.
# Update summary:
#   v0.2 tunes the early-stop threshold for the NLL scale and adds the RealNVP
#   depth-sweep plumbing. use_conditional_base=False reproduces the plain
#   Step-1.1 expert.
# =============================================================================
from __future__ import annotations
import logging
import hashlib
import json
from dataclasses import dataclass, asdict
logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_4A"

_VALID_INIT = ("zero_mu_unit_sigma", "random")


@dataclass(frozen=True)
class CBCfg:
    # data
    data_root: str = "./mnist_data"
    scale: int = 2
    blur_sigma: float = 1.0
    noise_sigma: float = 0.05
    batch_size: int = 128
    # expert (mirrors StepCfg essentials)
    expert: str = "nice"               # nice | realnvp | nsf
    dim: int = 784
    cond_width: int = 128
    h_dim: int = 256
    flow_hidden: int = 256
    use_film: bool = True
    use_v2_conditioner: bool = True
    film_depth: int = 2
    film_hidden: int = 128
    film_use_gelu: bool = True
    realnvp_n_couplings: int = 6
    nice_n_layers: int = 4                # NICE / nice_mix coupling depth (NCP-N8)
    realnvp_s_max: float = 2.0
    # --- image RealNVP (Stage 1.4b-A) -------------------------------------
    realnvp_type: str = "flat"            # flat | image
    image_realnvp_n_couplings: int = 8    # CNN coupling blocks (image path)
    image_realnvp_hidden: int = 64        # CNN hidden CHANNELS (NOT flow_hidden)
    image_realnvp_mask_type: str = "checkerboard"  # checkerboard only in v0.1
    image_realnvp_s_max: float = 2.0
    # conditional base (NEW)
    use_conditional_base: bool = True
    applied_to: str = "all_experts"    # declared scope
    base_mu_hidden: int = 128
    base_logsigma_hidden: int = 128
    base_logsigma_min: float = -5.0
    base_logsigma_max: float = 2.0
    base_init: str = "zero_mu_unit_sigma"
    base_gain: float = 1.0
    base_tau: float = 1e-3             # base_alive threshold on mu/logsigma std
    # train
    lr: float = 1e-3
    epochs: int = 150                  # max epochs; early-stop usually ends sooner
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    # early stopping (CBASE v0.3): stop on val-NLL plateau/overfit; keep best
    early_stop_patience: int = 20      # epochs of no val improvement before stop
    early_stop_min_delta: float = 0.001  # rel val-NLL improvement to count as progress (0.1%)
    # CCR (Phase-4 conditioning rescue; carried from Step 1.1)
    shuffle_loss_lambda: float = 0.1
    shuffle_loss_margin: float = 0.5
    h_std_penalty_mu: float = 10.0
    h_std_target: float = 0.05
    cond_y_residual_alpha_init: float = 0.3
    # bookkeeping
    seed: int = 0
    out_root: str = "./CSMF2/experiments/step_1_4a/results"
    log_every: int = 50

    def __post_init__(self):
        if self.expert not in ("nice", "realnvp", "nsf", "nice_mix"):
            logger.error("[CBCfg] expert must be nice/realnvp/nsf/nice_mix, got %r",
                         self.expert)
            raise ValueError(f"expert {self.expert!r} not in active roster")
        if self.scale not in (1, 2, 4):
            logger.error("[CBCfg] scale must be 1/2/4, got %s", self.scale)
            raise ValueError("scale must be 1, 2 or 4")
        if self.noise_sigma not in (0.0, 0.05, 0.1):
            logger.error("[CBCfg] noise_sigma must be 0/0.05/0.1, got %s",
                         self.noise_sigma)
            raise ValueError("noise_sigma out of set")
        if self.base_init not in _VALID_INIT:
            logger.error("[CBCfg] base_init must be one of %s, got %r",
                         _VALID_INIT, self.base_init)
            raise ValueError(f"base_init {self.base_init!r} invalid")
        if self.base_logsigma_min >= self.base_logsigma_max:
            logger.error("[CBCfg] base_logsigma_min must be < max, got %s/%s",
                         self.base_logsigma_min, self.base_logsigma_max)
            raise ValueError("base_logsigma_min must be < base_logsigma_max")
        for nm, v in (("base_mu_hidden", self.base_mu_hidden),
                      ("base_logsigma_hidden", self.base_logsigma_hidden),
                      ("epochs", self.epochs)):
            if v < 1:
                logger.error("[CBCfg] %s must be >=1, got %s", nm, v)
                raise ValueError(f"{nm} must be >=1")
        if self.base_gain <= 0.0:
            logger.error("[CBCfg] base_gain must be > 0, got %s", self.base_gain)
            raise ValueError("base_gain must be > 0")
        if self.base_tau < 0.0:
            logger.error("[CBCfg] base_tau must be >=0, got %s", self.base_tau)
            raise ValueError("base_tau must be >=0")
        if self.realnvp_n_couplings < 1:
            logger.error("[CBCfg] realnvp_n_couplings must be >=1, got %s",
                         self.realnvp_n_couplings)
            raise ValueError("realnvp_n_couplings must be >=1")
        if self.nice_n_layers < 1:
            logger.error("[CBCfg] nice_n_layers must be >=1, got %s",
                         self.nice_n_layers)
            raise ValueError("nice_n_layers must be >=1")
        if self.realnvp_type not in ("flat", "image"):
            logger.error("[CBCfg] realnvp_type must be flat|image, got %r",
                         self.realnvp_type)
            raise ValueError("realnvp_type must be flat|image")
        if self.image_realnvp_n_couplings < 1:
            logger.error("[CBCfg] image_realnvp_n_couplings must be >=1, got %s",
                         self.image_realnvp_n_couplings)
            raise ValueError("image_realnvp_n_couplings must be >=1")
        if self.image_realnvp_mask_type != "checkerboard":
            logger.error("[CBCfg] image_realnvp_mask_type: only 'checkerboard' "
                         "in v0.1, got %r", self.image_realnvp_mask_type)
            raise ValueError("image_realnvp_mask_type must be 'checkerboard'")
        if self.early_stop_patience < 1:
            logger.error("[CBCfg] early_stop_patience must be >=1, got %s",
                         self.early_stop_patience)
            raise ValueError("early_stop_patience must be >=1")
        if self.early_stop_min_delta < 0.0:
            logger.error("[CBCfg] early_stop_min_delta must be >=0, got %s",
                         self.early_stop_min_delta)
            raise ValueError("early_stop_min_delta must be >=0")

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def run_tag(self) -> str:
        cb = "cb" if self.use_conditional_base else "nocb"
        if self.expert == "realnvp" and self.realnvp_type == "image":
            depth = f"_img_c{self.image_realnvp_n_couplings}"
        elif self.expert == "realnvp":
            depth = f"_c{self.realnvp_n_couplings}"
        else:
            depth = ""
        return (f"{self.expert}_{cb}{depth}_s{self.scale}_n{self.noise_sigma:.2f}_"
                f"seed{self.seed}_{self.hash()}")
