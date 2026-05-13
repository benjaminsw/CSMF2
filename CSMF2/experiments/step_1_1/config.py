# =============================================================================
# STEP-1_1 v0.8 -- experiments.step_1_1.config
# Purpose: typed config for a single step_1_1 run. Frozen dataclass, sha256
#          hash baked into run_tag for reproducibility.
# CONVENTION: no silent defaults. Every invariant -> logger.error + raise.
# Changelog (v0.7 -> v0.8):
#   * NEW: cond_y_residual_alpha_init (float, default 0.0). Initial value
#     for the learnable alpha that scales the optional y-residual bypass
#     in Conditioner (v0.5+):
#         h = cnn_head(y) + alpha * Linear(y.flatten(1))
#     When 0.0, the bypass is disabled and Conditioner is v0.4-equivalent.
#     Recommended for Glow rescue experiments: 0.3.
#   * Invariant: cond_y_residual_alpha_init >= 0.
# Changelog (v0.6 -> v0.7):
#   * NEW: h_std_penalty_mu, h_std_target.
# Changelog (v0.5 -> v0.6):
#   * NEW: shuffle_loss_lambda, shuffle_loss_margin.
# Changelog (NEW in v0.1):
#   * Introduced.
# Update summary:
#   v0.8 adds the y-residual rescue knob. Default OFF preserves v0.7
#   behaviour. Use --cond-y-residual-alpha-init 0.3 to enable.
# =============================================================================
from __future__ import annotations
import logging
import hashlib
import json
from dataclasses import dataclass, asdict, field
logger = logging.getLogger(__name__)
__version__ = "0.8"
__abbr__ = "STEP-1_1"


@dataclass(frozen=True)
class StepCfg:
    # data
    data_root: str = "./mnist_data"
    scale: int = 2                    # in {2, 4}
    blur_sigma: float = 1.0
    noise_sigma: float = 0.0          # in {0.0, 0.05, 0.1}
    batch_size: int = 128
    # model
    expert: str = "nice"              # v0.3 default flip: realnvp -> nice (v2 needs nice)
    dim: int = 784                    # 28*28
    cond_width: int = 128             # v0.3 default flip: 64 -> 128
    h_dim: int = 256                  # v0.3 default flip: 128 -> 256
    flow_hidden: int = 256
    use_film: bool = True
    cache_h: bool = True
    # model -- v2-conditioner toggles (NICE only consumes film_*)
    use_v2_conditioner: bool = True   # v0.3 default flip: False -> True
    film_depth: int = 2               # v0.3 default flip: 1 -> 2
    film_hidden: int = 128            # v0.3 default flip: 64 -> 128
    film_use_gelu: bool = True        # v0.3 default flip: False -> True
    # model -- RealNVP-specific (v0.4; consumed only when expert='realnvp')
    realnvp_n_couplings: int = 6
    realnvp_s_max: float = 2.0
    # model -- Glow-specific (v0.4; consumed only when expert='glow')
    glow_n_steps: int = 8
    glow_coupling_hidden: int = 256
    glow_s_max: float = 2.0
    glow_squeeze: bool = True         # LOCKED in v0.4 (single-level)
    glow_n_levels: int = 1            # LOCKED in v0.4
    glow_image_c: int = 1
    glow_image_h: int = 28
    glow_image_w: int = 28
    glow_film_gain_init: float = 0.3   # v0.5: learnable FiLM gain init
    # train
    lr: float = 1e-3
    epochs: int = 5
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    # train -- shuffle-gap hinge loss (v0.6; default OFF)
    shuffle_loss_lambda: float = 0.0   # 0.0 = off (preserves v0.5 runtime)
    shuffle_loss_margin: float = 0.5   # target mean gap in nats when lambda > 0
    # train -- direct h.std penalty (v0.7; default OFF)
    h_std_penalty_mu: float = 0.0      # 0.0 = off
    h_std_target: float = 0.05         # active when h.std() < this value
    # model -- conditioner y-residual bypass (v0.8; default OFF)
    cond_y_residual_alpha_init: float = 0.0   # 0.0 = bypass disabled
    # bookkeeping
    seed: int = 0
    out_root: str = "./CSMF2/experiments/step_1_1/results"
    log_every: int = 50
    sanity_every_epoch: bool = True

    def __post_init__(self):
        if self.scale not in (2, 4):
            logger.error("[StepCfg] scale must be 2 or 4, got %s", self.scale)
            raise ValueError(f"scale must be in {{2,4}}, got {self.scale}")
        if self.noise_sigma not in (0.0, 0.05, 0.1):
            logger.error("[StepCfg] noise_sigma must be in {0.0, 0.05, 0.1}, got %s",
                         self.noise_sigma)
            raise ValueError(f"noise_sigma {self.noise_sigma} out of set")
        if self.expert not in ("nice", "realnvp", "nsf", "glow"):
            logger.error("[StepCfg] expert must be nice/realnvp/nsf/glow, got %s",
                         self.expert)
            raise ValueError(f"expert {self.expert!r} not recognised")
        if self.cond_width not in (64, 128):
            logger.error("[StepCfg] cond_width must be 64 or 128, got %s", self.cond_width)
            raise ValueError(f"cond_width {self.cond_width} not in {{64,128}}")
        if self.film_depth < 1:
            logger.error("[StepCfg] film_depth must be >=1, got %s", self.film_depth)
            raise ValueError(f"film_depth {self.film_depth} < 1")
        if self.h_dim < 1 or self.film_hidden < 1:
            logger.error("[StepCfg] h_dim and film_hidden must be positive, "
                         "got h_dim=%s film_hidden=%s", self.h_dim, self.film_hidden)
            raise ValueError("h_dim/film_hidden must be positive")
        if self.use_v2_conditioner:
            req = {"cond_width": 128, "h_dim": 256, "film_depth": 2,
                   "film_hidden": 128, "film_use_gelu": True}
            mismatch = {k: (getattr(self, k), v)
                        for k, v in req.items() if getattr(self, k) != v}
            if mismatch:
                logger.error("[StepCfg] use_v2_conditioner=True requires %s, "
                             "mismatch=%s", req, mismatch)
                raise ValueError(
                    f"use_v2_conditioner=True requires {req}; mismatch={mismatch}")
            # v0.4: v2 supported for nice / realnvp / glow. NSF still excluded.
            if self.expert not in ("nice", "realnvp", "glow"):
                logger.error("[StepCfg] use_v2_conditioner=True supported for "
                             "expert in {nice,realnvp,glow}, got %r",
                             self.expert)
                raise ValueError(
                    f"use_v2_conditioner=True not supported for "
                    f"expert={self.expert!r}; valid: {{nice,realnvp,glow}}")
        # ---- Glow-specific invariants (whether or not expert='glow') -----
        if self.glow_n_steps < 1:
            logger.error("[StepCfg] glow_n_steps must be >=1, got %s",
                         self.glow_n_steps)
            raise ValueError(f"glow_n_steps {self.glow_n_steps} < 1")
        if self.glow_s_max <= 0.0:
            logger.error("[StepCfg] glow_s_max must be > 0, got %s",
                         self.glow_s_max)
            raise ValueError(f"glow_s_max {self.glow_s_max} <= 0")
        if not self.glow_squeeze:
            logger.error("[StepCfg] glow_squeeze=True is LOCKED in v0.4")
            raise ValueError("glow_squeeze=True required in v0.4")
        if self.glow_n_levels != 1:
            logger.error("[StepCfg] glow_n_levels=1 is LOCKED in v0.4, got %d",
                         self.glow_n_levels)
            raise ValueError(f"glow_n_levels=1 required in v0.4, got "
                             f"{self.glow_n_levels}")
        if (self.glow_image_h % 2) or (self.glow_image_w % 2):
            logger.error("[StepCfg] Glow squeeze needs even H,W, got %dx%d",
                         self.glow_image_h, self.glow_image_w)
            raise ValueError("Glow squeeze needs even H,W")
        if (self.glow_image_c * self.glow_image_h * self.glow_image_w
                != self.dim):
            logger.error("[StepCfg] glow_image_c*h*w (%d) != dim (%d)",
                         self.glow_image_c * self.glow_image_h
                         * self.glow_image_w, self.dim)
            raise ValueError("glow_image_{c,h,w} product must equal dim")
        if self.glow_film_gain_init < 0.0:
            logger.error("[StepCfg] glow_film_gain_init must be >=0, got %s",
                         self.glow_film_gain_init)
            raise ValueError(
                f"glow_film_gain_init must be >=0, got {self.glow_film_gain_init}")
        # ---- RealNVP-specific invariants ---------------------------------
        if self.realnvp_n_couplings < 1:
            logger.error("[StepCfg] realnvp_n_couplings must be >=1, got %s",
                         self.realnvp_n_couplings)
            raise ValueError(
                f"realnvp_n_couplings {self.realnvp_n_couplings} < 1")
        if self.realnvp_s_max <= 0.0:
            logger.error("[StepCfg] realnvp_s_max must be > 0, got %s",
                         self.realnvp_s_max)
            raise ValueError(f"realnvp_s_max {self.realnvp_s_max} <= 0")
        # ---- shuffle-loss invariants (v0.6) ------------------------------
        if self.shuffle_loss_lambda < 0.0:
            logger.error("[StepCfg] shuffle_loss_lambda must be >=0, got %s",
                         self.shuffle_loss_lambda)
            raise ValueError(
                f"shuffle_loss_lambda must be >=0, got {self.shuffle_loss_lambda}")
        if self.shuffle_loss_margin < 0.0:
            logger.error("[StepCfg] shuffle_loss_margin must be >=0, got %s",
                         self.shuffle_loss_margin)
            raise ValueError(
                f"shuffle_loss_margin must be >=0, got {self.shuffle_loss_margin}")
        # ---- h.std penalty invariants (v0.7) -----------------------------
        if self.h_std_penalty_mu < 0.0:
            logger.error("[StepCfg] h_std_penalty_mu must be >=0, got %s",
                         self.h_std_penalty_mu)
            raise ValueError(
                f"h_std_penalty_mu must be >=0, got {self.h_std_penalty_mu}")
        if self.h_std_target < 0.0:
            logger.error("[StepCfg] h_std_target must be >=0, got %s",
                         self.h_std_target)
            raise ValueError(
                f"h_std_target must be >=0, got {self.h_std_target}")
        # ---- cond y-residual invariant (v0.8) ----------------------------
        if self.cond_y_residual_alpha_init < 0.0:
            logger.error("[StepCfg] cond_y_residual_alpha_init must be >=0, "
                         "got %s", self.cond_y_residual_alpha_init)
            raise ValueError(
                f"cond_y_residual_alpha_init must be >=0, got "
                f"{self.cond_y_residual_alpha_init}")

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def run_tag(self) -> str:
        return (f"{self.expert}_s{self.scale}_n{self.noise_sigma:.2f}_"
                f"seed{self.seed}_{self.hash()}")
