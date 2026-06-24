# =============================================================================
# STEP-2_3 v0.1 -- experiments.step_2_3.config  (S2.3 hybrid-training config)
# Purpose: typed config for a Stage 2.3 run. Frozen dataclass, sha256 hash baked
#          into run_tag. Encodes the S2.3-PLAN v0.4 arms/knobs:
#            consistency_mode {all_expert | soft_gate | residual}
#            norm_mode {l0 | ema | none}   gradnorm {on|off}   pcgrad {on|off}
#            entropy_lambda / load_balance_lambda (anti-collapse, fixed coef)
#          + the explicit GO/STOP thresholds (neff_min, max_weight_max, ...).
# CONVENTION: no silent defaults. Every invariant -> logger.error + raise.
#          2.3-A DEFAULTS: consistency_mode=all_expert, norm_mode=l0,
#          gradnorm=off, pcgrad=off (per the approved first-run setup).
# Changelog (v0.1 -> v0.2, fix D):
#   * + rec_exclude: tuple = (). Experts named here are dropped from the rec term
#     (validated subset of expert_set; cannot remove all). Used as --rec-exclude
#     nsf to stop NSF's spline-inverse divergence under reconstruction pressure.
# Changelog (NEW in v0.1):
#   * Introduced. Phase/arm/normalizer/anti-collapse knobs + GO/STOP thresholds.
# Update summary:
#   v0.1 config for 2.3-A only; transport/calibration weights present but their
#   enable-flags default OFF (added at v0.3/v0.4 of the loss schedule).
# =============================================================================
from __future__ import annotations
import logging
import hashlib
import json
from dataclasses import dataclass, asdict
logger = logging.getLogger(__name__)
__version__ = "0.2"
__abbr__ = "STEP-2_3"


@dataclass(frozen=True)
class Stage23Cfg:
    # data / cell
    data_root: str = "./mnist_data"
    scale: int = 2
    blur_sigma: float = 1.0
    noise_sigma: float = 0.05
    batch_size: int = 128
    # experts / roster (warm-started from 1.4a CB ckpts; NO glow)
    expert_set: tuple = ("nsf", "realnvp", "nice_mix")   # or ("nsf","realnvp")
    warm_start: str = "1.4a"                # {"1.4a", "scratch"}
    # phase / arm
    phase: str = "A"                        # {"A","B","C"}
    consistency_mode: str = "all_expert"    # {all_expert, soft_gate, residual}
    # rec_exclude: experts whose name is here are SKIPPED in the rec term (their
    # decode/spline-inverse is never run, so no rec gradient touches them). Fix D
    # for 2.3-A: NSF's RQ-spline inverse drifts non-invertible under rec pressure
    # (disc -> -3e-2, worsening as LR drops) and NSF already dominates rec, so it
    # needs no rec rescue. Excluded experts stay full in NLL + gate + mixture.
    rec_exclude: tuple = ()
    # normalization (Level 1 / Level 2)
    norm_mode: str = "l0"                   # {l0, ema, none}
    l0_warmup_batches: int = 5
    gradnorm: bool = False                  # OFF for 2.3-A
    gradnorm_alpha: float = 1.5
    gradnorm_lr: float = 0.025
    pcgrad: bool = False                    # OFF until grad_cosine<0
    # objective term weights (pre-normalization priors; GradNorm overrides when on)
    alpha_nll: float = 1.0
    beta_rec: float = 1.0
    gamma_transport: float = 0.0           # 0 -> term OFF (v0.3+)
    delta_calib: float = 0.0               # 0 -> term OFF (v0.4+)
    # anti-collapse (OUTSIDE normalized objective; small FIXED coef; LIGHT in A)
    entropy_lambda: float = 0.01
    load_balance_lambda: float = 0.01
    # gate
    gate_hidden: int = 128
    gate_tau: float = 1.0
    # reconstruction-residual z-bank (matches step_1_3.scores.per_expert_rec, so
    # the V2 metric is comparable to the frozen Stage-1.3 NSF-only baseline)
    rec_z_bank_size: int = 4
    rec_z_bank_seed: int = 1234
    rec_z_mode: str = "fixed_shared"     # {"fixed_shared","zero"}
    # train
    lr: float = 5e-4                        # low LR (joint fine-tune sensibility)
    epochs: int = 30
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    # GO/STOP thresholds (executable verdict, S2.3-PLAN)
    neff_min: float = 1.5
    max_weight_max: float = 0.70
    nll_regression_tol: float = 0.05
    # bookkeeping
    seed: int = 0
    out_root: str = "./CSMF2/experiments/step_2_3/results"
    log_every: int = 50

    def __post_init__(self):
        if self.scale not in (1, 2, 4):
            logger.error("[Stage23Cfg] scale must be 1/2/4, got %s", self.scale)
            raise ValueError(f"scale {self.scale} invalid")
        if self.noise_sigma not in (0.0, 0.05, 0.1):
            logger.error("[Stage23Cfg] noise_sigma must be 0/0.05/0.1, got %s",
                         self.noise_sigma)
            raise ValueError(f"noise_sigma {self.noise_sigma} invalid")
        if self.phase not in ("A", "B", "C"):
            logger.error("[Stage23Cfg] phase must be A/B/C, got %s", self.phase)
            raise ValueError(f"phase {self.phase} invalid")
        if self.consistency_mode not in ("all_expert", "soft_gate", "residual"):
            logger.error("[Stage23Cfg] bad consistency_mode %s",
                         self.consistency_mode)
            raise ValueError(f"consistency_mode {self.consistency_mode} invalid")
        if self.norm_mode not in ("l0", "ema", "none"):
            logger.error("[Stage23Cfg] bad norm_mode %s", self.norm_mode)
            raise ValueError(f"norm_mode {self.norm_mode} invalid")
        if self.warm_start not in ("1.4a", "scratch"):
            logger.error("[Stage23Cfg] bad warm_start %s", self.warm_start)
            raise ValueError(f"warm_start {self.warm_start} invalid")
        bad = [e for e in self.expert_set
               if e not in ("nice", "realnvp", "nsf", "nice_mix")]
        if bad:
            logger.error("[Stage23Cfg] unknown experts %s (glow excluded)", bad)
            raise ValueError(f"unknown experts {bad}")
        if "glow" in self.expert_set:
            logger.error("[Stage23Cfg] glow is EXCLUDED from the 2.3 roster")
            raise ValueError("glow excluded")
        if len(self.expert_set) < 2:
            logger.error("[Stage23Cfg] need >=2 experts, got %s", self.expert_set)
            raise ValueError("need >=2 experts")
        bad_excl = [e for e in self.rec_exclude if e not in self.expert_set]
        if bad_excl:
            logger.error("[Stage23Cfg] rec_exclude %s not in expert_set %s",
                         bad_excl, self.expert_set)
            raise ValueError(f"rec_exclude {bad_excl} not in expert_set")
        if set(self.rec_exclude) >= set(self.expert_set):
            logger.error("[Stage23Cfg] rec_exclude %s removes ALL experts from "
                         "the rec term; >=1 must remain", self.rec_exclude)
            raise ValueError("rec_exclude cannot remove all experts")
        for name, v in (("entropy_lambda", self.entropy_lambda),
                        ("load_balance_lambda", self.load_balance_lambda),
                        ("alpha_nll", self.alpha_nll), ("beta_rec", self.beta_rec),
                        ("gamma_transport", self.gamma_transport),
                        ("delta_calib", self.delta_calib)):
            if v < 0.0:
                logger.error("[Stage23Cfg] %s must be >=0, got %s", name, v)
                raise ValueError(f"{name} must be >=0")
        if self.gate_tau <= 0.0:
            logger.error("[Stage23Cfg] gate_tau must be >0, got %s", self.gate_tau)
            raise ValueError("gate_tau must be >0")
        if self.rec_z_mode not in ("fixed_shared", "zero"):
            logger.error("[Stage23Cfg] rec_z_mode must be fixed_shared/zero, got %s",
                         self.rec_z_mode)
            raise ValueError(f"rec_z_mode {self.rec_z_mode} invalid")
        if self.rec_z_bank_size < 1:
            logger.error("[Stage23Cfg] rec_z_bank_size must be >=1, got %s",
                         self.rec_z_bank_size)
            raise ValueError("rec_z_bank_size must be >=1")
        if self.l0_warmup_batches < 1:
            logger.error("[Stage23Cfg] l0_warmup_batches must be >=1")
            raise ValueError("l0_warmup_batches must be >=1")
        # 2.3-A guard: Phase A is expert-pressure-first, GradNorm should be OFF
        if self.phase == "A" and self.gradnorm:
            logger.error("[Stage23Cfg] Phase A is expert-pressure-first; enable "
                         "GradNorm at v0.2 (>=3 terms), not in 2.3-A")
            raise ValueError("gradnorm must be OFF in Phase A (2.3-A)")

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def run_tag(self) -> str:
        es = "-".join(self.expert_set)
        return (f"s23{self.phase}_{es}_s{self.scale}_n{self.noise_sigma:.2f}_"
                f"seed{self.seed}_{self.hash()}")
