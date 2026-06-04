# =============================================================================
# STEP-1_1_1_1 v0.2 -- experiments.step_1_1_1_1.config
# Purpose: typed config for a single MAP-refinement run (inference time).
#          Frozen dataclass, sha256 hash baked into run_tag for reproducibility.
# CONVENTION: no silent defaults. Every invariant -> logger.error + raise.
#
# Design intent (independent experiment):
#   step_1_1_1_1 is a GENERIC inference-time experiment. It tests whether
#   MAP refinement improves reconstruction for ANY trained conditional flow
#   checkpoint -- not just those from step_1_1_1. The primary interface is:
#
#       --ckpt-dir PATH
#
#   pointing at any directory containing ckpt.pt + report.json with a cfg
#   block. Works for checkpoints from step_1_1, step_1_1_1, future
#   step_1_2 mixtures, future CIFAR runs, etc.
#
#   --best-params + --expert is an OPTIONAL convenience helper that resolves
#   to a ckpt-dir via a lookup table. It is NOT a dependency. If best_params
#   is None (default), the experiment runs without it.
#
# Changelog (v0.1.1 -> v0.2):
#   * NEW init mode "is_random": importance-sampled random init. Samples
#     n_candidates z0's, picks the one with the lowest initial residual
#     ||A(decode(z0,h)) - y||^2 PER IMAGE, then runs MAP from that winner.
#   * NEW field: n_candidates (int >= 1). Only meaningful when init=is_random.
#     For init=random and init=encoded, n_candidates is forced to 1 in
#     __post_init__ so the run_tag hash stays comparable.
#   * NEW field: track_candidates (bool). When True, full per-image
#     selection log (best_k, all-K residuals) is written to metrics.json.
#   * The default init remains "random" -- v0.2 is backward compatible.
# Changelog (v0.1 -> v0.1.1):
#   * Demoted best_params from "default required" to "optional helper".
#     Default is now None. User must supply ONE of:
#         --ckpt-dir PATH                          (primary)
#         --best-params PATH --best-params-expert  (convenience)
#   * NEW field: best_params_expert (str | None). Separates the "expert to
#     look up in best_params" from the (no-longer-existing) cfg.expert,
#     making the design cleaner: the ckpt itself dictates which expert,
#     not the cfg.
#   * REMOVED: cfg.expert. The experiment doesn't know or care which
#     expert architecture is in the ckpt -- it just loads and refines.
#     expert info comes from the loaded ckpt's report.json -> cfg.expert.
#   * REMOVED hard tie to step_1_1_1 in default paths. best_params and
#     train_results_root are now both None by default.
# Changelog (NEW in v0.1):
#   * Introduced.
# Update summary:
#   v0.2 adds importance-sampled init as a drop-in alternative to random.
#   Same MAP loop downstream; just a smarter starting z. The expected gain
#   is largest for experts with poor latent KS (Glow, NICE); experts with
#   near-perfect latent (NSF KS=0.008) should gain little since random init
#   already lands close to a good region.
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import logging

logger = logging.getLogger(__name__)
__version__ = "0.2"
__abbr__ = "STEP-1_1_1_1"


@dataclass(frozen=True)
class MAPCfg:
    # ---- which trained checkpoint to refine (PRIMARY interface) --------
    ckpt_dir: str | None = None
    # ---- best_params lookup (OPTIONAL convenience helper) --------------
    # If --ckpt-dir is not given, supply --best-params + --best-params-expert
    # and the run resolves a ckpt_dir from the JSON.
    best_params: str | None = None
    best_params_expert: str | None = None      # which expert to look up
    train_results_root: str | None = None      # where the resolved ckpt lives;
                                               # required only when using best_params
    # ---- MAP optimisation ----------------------------------------------
    steps: int = 50
    lr: float = 1e-2
    lambda_prior: float = 1e-3
    init: str = "random"                       # {"random", "is_random", "encoded"}
    # ---- importance-sampled init (v0.2) --------------------------------
    n_candidates: int = 8                      # K. Only used when init="is_random".
                                               # For other inits, coerced to 1.
    track_candidates: bool = False             # save per-image best_k + all-K
                                               # initial residuals to metrics.json
    # ---- evaluation ----------------------------------------------------
    n_test: int = 256
    batch_size: int = 64
    seed: int = 0
    # ---- output --------------------------------------------------------
    out_root: str = "./CSMF2/experiments/step_1_1_1_1/results"

    def __post_init__(self):
        # ---- ckpt resolution: ckpt_dir XOR best_params trio --------------
        using_ckpt_dir = self.ckpt_dir is not None
        using_best     = self.best_params is not None
        if using_ckpt_dir == using_best:
            logger.error("[MAPCfg] supply EXACTLY ONE of:\n"
                         "  --ckpt-dir PATH                          (primary)\n"
                         "  --best-params PATH --best-params-expert  (helper)\n"
                         "got ckpt_dir=%r best_params=%r",
                         self.ckpt_dir, self.best_params)
            raise ValueError(
                "supply exactly one of --ckpt-dir OR --best-params")
        if using_best:
            if self.best_params_expert is None:
                logger.error("[MAPCfg] --best-params requires "
                             "--best-params-expert (expert key to look up)")
                raise ValueError("--best-params requires --best-params-expert")
            if self.best_params_expert not in (
                    "nice", "realnvp", "nsf", "glow"):
                logger.error("[MAPCfg] best_params_expert must be in "
                             "{nice,realnvp,nsf,glow}, got %r",
                             self.best_params_expert)
                raise ValueError(
                    f"best_params_expert must be in "
                    f"{{nice,realnvp,nsf,glow}}, got "
                    f"{self.best_params_expert!r}")
            if self.train_results_root is None:
                logger.error("[MAPCfg] --best-params requires "
                             "--train-results-root (where ckpts live)")
                raise ValueError(
                    "--best-params requires --train-results-root")
        # ---- MAP knobs ----------------------------------------------------
        if self.steps < 1:
            logger.error("[MAPCfg] steps must be >=1, got %d", self.steps)
            raise ValueError(f"steps must be >=1, got {self.steps}")
        if self.lr <= 0.0:
            logger.error("[MAPCfg] lr must be >0, got %s", self.lr)
            raise ValueError(f"lr must be >0, got {self.lr}")
        if self.lambda_prior < 0.0:
            logger.error("[MAPCfg] lambda_prior must be >=0, got %s",
                         self.lambda_prior)
            raise ValueError(
                f"lambda_prior must be >=0, got {self.lambda_prior}")
        if self.init not in ("random", "is_random", "encoded"):
            logger.error("[MAPCfg] init must be in {random,is_random,encoded}, "
                         "got %r", self.init)
            raise ValueError(
                f"init must be in {{random,is_random,encoded}}, "
                f"got {self.init!r}")
        # n_candidates invariants
        if self.n_candidates < 1:
            logger.error("[MAPCfg] n_candidates must be >=1, got %d",
                         self.n_candidates)
            raise ValueError(
                f"n_candidates must be >=1, got {self.n_candidates}")
        # For modes that don't use candidates, force K=1 so the run_tag hash
        # stays comparable to v0.1.1 baselines (frozen dataclass: use
        # object.__setattr__).
        if self.init != "is_random" and self.n_candidates != 1:
            object.__setattr__(self, "n_candidates", 1)
        # Warn (not error) at very large K -- usually wasted compute.
        if self.init == "is_random" and self.n_candidates > 64:
            logger.warning("[MAPCfg] n_candidates=%d is large; diminishing "
                           "returns past ~16-32 in practice",
                           self.n_candidates)
        if self.n_test < 1:
            logger.error("[MAPCfg] n_test must be >=1, got %d", self.n_test)
            raise ValueError(f"n_test must be >=1, got {self.n_test}")
        if self.batch_size < 1:
            logger.error("[MAPCfg] batch_size must be >=1, got %d",
                         self.batch_size)
            raise ValueError(f"batch_size must be >=1, got {self.batch_size}")

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def run_tag(self) -> str:
        # Tag does NOT include expert (independent design -- expert is whatever
        # the ckpt contains). Source-of-ckpt prefix tells you whether this was
        # a direct ckpt-dir run or a best-params lookup.
        if self.ckpt_dir is not None:
            src = "ckpt"
        else:
            src = f"best-{self.best_params_expert}"
        # init=is_random gets a K suffix so the tag distinguishes K values.
        if self.init == "is_random":
            init_str = f"is{self.n_candidates}"
        else:
            init_str = self.init
        return (f"map_{src}_init-{init_str}_steps{self.steps}_"
                f"lr{self.lr:.0e}_lp{self.lambda_prior:.0e}_"
                f"seed{self.seed}_{self.hash()}")
