# =============================================================================
# STEP-1_4A v0.1 -- experiments.step_1_4a.cb_expert
# Purpose: wrap a trained/buildable flow expert with a ConditionalBase so the
#          base over the flow latent w becomes N(mu(h), sigma(h)) instead of
#          N(0,I). Exposes the SAME API as a plain expert (cond / encode /
#          decode / log_prob / dim) so Stage 1.3 RECGATE loads it unchanged.
# CONVENTION: non-finite -> logger.error + raise. No fallback / mock / dummy.
# Semantics (posterior-safe):
#   log_prob(x,y) = base.log_prob(w, h) + ldj_flow,  (w,ldj)=expert.encode(x,h)
#   decode(eps,h) = expert.decode( mu(h)+sigma(h)*eps , h )   # eps ~ N(0,I)
# The decode contract keeps eps standard-normal so RECGATE's shared z-bank
# stays a fair cross-expert comparison; CB maps eps -> proper base sample.
# Changelog (NEW in v0.1):
#   * Introduced. CBExpert wrapper + state_dict round-trip (expert + base).
# Update summary:
#   v0.1 lets a conditional-base expert be a drop-in for the plain expert
#   API. cond() returns the underlying expert's h (base + couplings share h).
# =============================================================================
from __future__ import annotations
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_4A"


class CBExpert(nn.Module):
    """Conditional-base wrapper. Same API surface as the wrapped expert."""
    def __init__(self, expert, base):
        super().__init__()
        for attr in ("cond", "encode", "decode", "log_prob", "dim"):
            if not hasattr(expert, attr):
                logger.error("[CBExpert] wrapped expert missing %r", attr)
                raise AttributeError(f"expert must expose {attr}")
        self.expert = expert
        self.base = base

    @property
    def dim(self):
        return self.expert.dim

    def cond(self, y):
        return self.expert.cond(y)               # base + couplings share h

    def encode(self, x_flat, h):
        return self.expert.encode(x_flat, h)     # (w, ldj_flow)

    def decode(self, eps, h):
        # eps ~ N(0,I) -> base sample w = mu(h) + sigma(h)*eps -> x
        w = self.base.to_w(eps, h)
        return self.expert.decode(w, h)

    def log_prob(self, x_flat, y):
        h = self.expert.cond(y)
        w, ldj_flow = self.expert.encode(x_flat, h)
        lp = self.base.log_prob(w, h) + ldj_flow
        if not torch.isfinite(lp).all():
            logger.error("[CBExpert] non-finite log_prob")
            raise RuntimeError("non-finite CBExpert log_prob")
        return lp
