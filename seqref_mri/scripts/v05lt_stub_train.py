# SEQREF-MRI-V05LTS v0.2 -- scripts.v05lt_stub_train
# LIFETIME: KEEP
# =============================================================================
# FIXTURE-ONLY STUB (declared; never used by production). A small model with
#   the two methods the V05LT trainer and shared metrics call
#   (log_prob_free, decode_scalars), so fixtures can exercise the REAL
#   trainer control flow (order, validation, plateau/selection, retention,
#   resume, failure taxonomy) many epochs at a time in seconds:
#     decode_scalars(z) = z * exp(s) + W h(cond)        (affine in z)
#     log_prob_free(u)  = N(0, I) log-density of the inverse map
#   h(cond) = mean-pooled cond channels; parameters W (13824 x 2), s.
#   Not a substitute for the real-model evidence: bitwise resume
#   equivalence and failure tests must also run on the real trainer path
#   (I1.6(c)). `poison` lets fixtures inject a non-finite loss.
#   v0.2: condition() and flow.encode() expose the same decomposition as
#   the production model (log p = log N(z; 0, I) + ldj) for the shared
#   validation's NLL components (I1.8 b0, G3).
# CONVENTION: logger.error + V05LTError(code); no silent pass.
# Changelog (v0.2, 2026-10-24; I1.8 b0, G3):
#   * condition() and flow.encode() added; log_prob_free unchanged.
# Changelog (NEW in v0.1, 2026-10-21):
#   * Introduced under I1.6(b) (SEQREF-MRI-V05LTI16B v0.1) for retention,
#     resume-state and taxonomy fixtures.
# =============================================================================
from __future__ import annotations

import logging
import math

logger = logging.getLogger("seqref_mri.v05lt_stub_train")

__version__ = "0.2"
__abbr__ = "SEQREF-MRI-V05LTS"

DIM = 13824


def build_stub(seed: int, poison_after: int | None = None):
    import torch
    from torch import nn

    class _Enc:
        """Non-module view so the state_dict is unchanged."""
        def __init__(self, owner):
            self._owner = owner

        def encode(self, u, h):
            m = self._owner
            z = (u.to(torch.float32) - h @ m.W.T) * torch.exp(-m.s)
            ldj = (-DIM * m.s[0]).expand(u.shape[0]).to(torch.float32)
            return z, ldj

    class StubFlow(nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(int(seed))   # local generator
            self.W = nn.Parameter(0.01 * torch.randn(DIM, 2, generator=g))
            self.s = nn.Parameter(torch.zeros(1))
            self.calls = 0
            object.__setattr__(self, "flow", _Enc(self))

        def condition(self, cond_in, mask):
            return self._h(cond_in)

        def _h(self, cond):
            return cond.mean(dim=(-2, -1)).to(torch.float32)   # (B, 2)

        def decode_scalars(self, z, cond_in, mask):
            return z.to(torch.float32) * torch.exp(self.s) + \
                self._h(cond_in) @ self.W.T

        def log_prob_free(self, u, cond_in, mask):
            self.calls += 1
            z = (u.to(torch.float32) - self._h(cond_in) @ self.W.T) \
                * torch.exp(-self.s)
            lp = -0.5 * (z ** 2).sum(1) - DIM * (self.s[0]
                                                 + 0.5 * math.log(2 * math.pi))
            if poison_after is not None and self.training \
                    and self.calls > poison_after:
                lp = lp * float("nan")
            return lp
    torch.manual_seed(int(seed))      # mirror build_model's global seeding
    return StubFlow()
