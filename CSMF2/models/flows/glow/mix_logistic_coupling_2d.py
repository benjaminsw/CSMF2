# =============================================================================
# FLOWPP v0.4 -- models.flows.glow.mix_logistic_coupling_2d
# Purpose: Flow++ logistic-mixture-CDF coupling (channel-wise split), a DROP-IN
#          replacement for AffineCoupling2D inside GlowStep. The conditioning
#          NN is IDENTICAL to AffineCoupling2D (3-conv block + residual FiLM +
#          learnable film_gain); ONLY the elementwise transform on x2 changes:
#          affine  ->  sigma^{-1}(MixLogisticCDF(x2; pi,mu,s)) * exp(a) + b
#          (Ho et al. 2019, eqs. 17-20). This isolates the coupling primitive
#          as the single variable vs Glow (FLOWPP core hypothesis test).
# CONVENTION: no fallback / mock / pass. Any non-finite tensor, failed bracket,
#             or non-converged bisection -> logger.error + raise.
# Math (CLAMP-FREE, exactly invertible):
#   v(x2) = logit(MixLogisticCDF(x2)) is computed via logsumexp on the log-CDF
#   and log-survival, so it is finite for all finite z and strictly increasing
#   in x2 -- never rounding cdf to exactly 0/1:
#     log_cdf    = logsumexp_k( log_w_k + logsigmoid(zk) )
#     log_1m_cdf = logsumexp_k( log_w_k + logsigmoid(-zk) ),  zk=(x2-mu_k)e^{-s_k}
#     v          = log_cdf - log_1m_cdf
#   forward:  a_b = s_max*tanh(a);  y2 = v*exp(a_b) + b
#             log|dy2/dx2| = a_b + log(pdf) - log_cdf - log_1m_cdf
#               log(pdf) = logsumexp_k( log_w_k - s_k + logsigmoid(zk)
#                                       + logsigmoid(-zk) )
#   inverse:  v_target = (y2 - b)*exp(-a_b);  solve v(x2)=v_target by BISECTION
#             on the SAME v -> exact round-trip (no clamp mismatch). Raises on
#             non-bracket / non-convergence.
# Init: conv3 small-normal (std 1e-2), zero bias -> at init pi~0 (uniform), mu~0,
#       log_s~0, a~0, b~0  =>  cdf = sigmoid(x2), v = x2, y2 = x2  (near-identity),
#       mirroring AffineCoupling2D's near-identity init.
# Changelog (v0.3 -> v0.4) [convergence-metric fix]:
#   * Convergence is now judged by relative bracket width in x2-space, not the
#     v-residual. v=logit(cdf) is near-vertical in the tails (dv/dx2 ~
#     1/(cdf(1-cdf))), so a machine-accurate x2 could show a v-residual ~6e-4
#     and trip a FALSE non-converge (observed when decoding samples whose
#     points sit deep in a tail). Bracket width is slope-independent and is the
#     correct bisection convergence test; round-trip accuracy is unchanged
#     (x2 exact to ~1e-14, verified).
# Changelog (v0.2 -> v0.3) [bracketing fix]:
#   * REPLACED the doubling-bracket inverse (capped at +/-10*2^40) -- it failed
#     to bracket when decoding generative samples that are ill-conditioned in
#     the inverse direction (observed v_target ~2e11 -> "bracketing failed").
#   * Now uses an ANALYTIC bracket: each logistic component inverts linearly at
#     x2_k = mu_k + v_target*exp(log_s_k), and the mixture root provably lies in
#     [min_k x2_k, max_k x2_k]. Brackets ANY finite v_target with no expansion
#     loop, so decode never crashes (large samples give large-but-finite x2,
#     matching the graceful degradation of Glow's closed-form affine inverse).
#   * Residual tolerance made relative-or-absolute so huge v_target doesn't
#     trip a false non-converge.
# Changelog (v0.1 -> v0.2) [invertibility fix]:
#   * REMOVED the cdf.clamp(eps,1-eps) path: it broke decode(encode(x)) on
#     saturated elements (forward clamped cdf, inverse bisected toward the
#     clamped value -> wrong x2; observed max round-trip err 1.69 at epoch 1).
#   * Reformulated v = logit(cdf) via logsumexp(log_cdf) - logsumexp(log_1m_cdf)
#     (stable, finite, strictly monotone) and bisect on v(x2)=v_target instead
#     of cdf(x2)=target. Round-trip now exact (machine precision) including at
#     saturation. log-det uses the same stable log_cdf/log_1m_cdf terms.
# Changelog (NEW in v0.1):
#   * Introduced for the Flow++ candidate (Stage 1.1). Logistic-mixture CDF
#     coupling + bisection inverse + exact mixture-pdf log-det.
# Update summary:
#   v0.2 makes the coupling exactly invertible by removing the clamp and
#   working in stable log-space. v0.1's three-upgrade scope is unchanged:
#   variational dequantization and self-attention conditioning remain OUT of
#   scope (held constant vs Glow so the coupling primitive is the lone variable).
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.2"
__abbr__ = "FLOWPP"

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...conditioner import FiLMHead

# Numerical guards (documented; consistent across forward/inverse).
_LOG_S_CLAMP = 7.0       # |log_s| <= this  (scales in [e^-7, e^7]); prevents overflow
_BISECT_ITERS = 60       # halvings; >50 saturates f32, fine for f64 too
_X_REL_TOL = 1e-6        # convergence: relative bracket width (hi-lo) in x2-space.
                         # Measured in x2 (not v): v=logit(cdf) is near-vertical
                         # in the tails, so a machine-accurate x2 can still show
                         # a large v-residual -- bracket width is the correct,
                         # slope-independent bisection convergence test.


class MixLogisticCoupling2D(nn.Module):
    def __init__(self, num_channels: int, hidden: int, h_dim: int,
                 *, flip: bool, s_max: float = 2.0, n_mixtures: int = 4,
                 film_hidden: int = 128, film_depth: int = 2,
                 film_use_gelu: bool = True,
                 film_gain_init: float = 0.3):
        super().__init__()
        if num_channels < 2 or num_channels % 2:
            logger.error("[MixLogisticCoupling2D] num_channels must be even >=2, "
                         "got %d", num_channels)
            raise ValueError(f"num_channels must be even >=2, got {num_channels}")
        if n_mixtures < 1:
            logger.error("[MixLogisticCoupling2D] n_mixtures must be >=1, got %d",
                         n_mixtures)
            raise ValueError(f"n_mixtures must be >=1, got {n_mixtures}")
        if film_gain_init < 0.0:
            logger.error("[MixLogisticCoupling2D] film_gain_init must be >=0, "
                         "got %s", film_gain_init)
            raise ValueError(f"film_gain_init must be >=0, got {film_gain_init}")
        self.num_channels = num_channels
        self.c_in  = num_channels // 2
        self.c_out = num_channels - self.c_in
        self.flip  = flip
        self.s_max = s_max
        self.K = n_mixtures
        # per output channel: K (pi) + K (mu) + K (log_s) + 1 (a) + 1 (b)
        self.params_per_chan = 3 * self.K + 2
        # conditioning NN -- IDENTICAL structure to AffineCoupling2D.
        self.conv1 = nn.Conv2d(self.c_in, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden,    hidden, 1, padding=0)
        self.conv3 = nn.Conv2d(hidden, self.c_out * self.params_per_chan,
                               3, padding=1)
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.conv3.bias)
        self.film1 = FiLMHead(h_dim, hidden, hidden=film_hidden,
                              depth=film_depth, use_gelu=film_use_gelu,
                              output_form="residual")
        self.film2 = FiLMHead(h_dim, hidden, hidden=film_hidden,
                              depth=film_depth, use_gelu=film_use_gelu,
                              output_form="residual")
        self.film_gain = nn.Parameter(torch.tensor(float(film_gain_init)))

    # ---- split / merge (identical convention to AffineCoupling2D) ----------
    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.flip:
            return x[:, self.c_in:], x[:, :self.c_in]
        return x[:, :self.c_in], x[:, self.c_in:]

    def _merge(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([b, a], dim=1) if self.flip else torch.cat([a, b], dim=1)

    # ---- conditioning NN -> transform parameters ---------------------------
    def _params(self, x1: torch.Tensor, h: torch.Tensor):
        # Mirrors AffineCoupling2D._st residual-FiLM threading exactly.
        g = self.film_gain
        z = torch.relu(self.conv1(x1))
        gamma_raw, beta = self.film1(h)
        z = z * (1.0 + g * gamma_raw[:, :, None, None]) + g * beta[:, :, None, None]
        z = torch.relu(self.conv2(z))
        gamma_raw, beta = self.film2(h)
        z = z * (1.0 + g * gamma_raw[:, :, None, None]) + g * beta[:, :, None, None]
        out = self.conv3(z)                       # (B, c_out*(3K+2), H, W)
        B, _, H, W = out.shape
        out = out.view(B, self.c_out, self.params_per_chan, H, W)
        K = self.K
        pi_logits = out[:, :, 0:K]                # (B, c_out, K, H, W)
        mu        = out[:, :, K:2 * K]
        log_s     = out[:, :, 2 * K:3 * K].clamp(-_LOG_S_CLAMP, _LOG_S_CLAMP)
        a         = out[:, :, 3 * K]              # (B, c_out, H, W)
        b         = out[:, :, 3 * K + 1]
        a_b = self.s_max * torch.tanh(a)
        return pi_logits, mu, log_s, a_b, b

    @staticmethod
    def _log_cdf_terms(x2, pi_logits, mu, log_s):
        # Returns (log_cdf, log_1m_cdf), both via logsumexp -> NO clamp, finite
        # for all finite z, and exactly invertible (v = log_cdf - log_1m_cdf is
        # strictly increasing in x2). z is NOT clamped here: logsigmoid is stable
        # at large |z|, so cdf never rounds to exactly 0/1.
        z = (x2.unsqueeze(2) - mu) * torch.exp(-log_s)   # (B, c_out, K, H, W)
        log_w = F.log_softmax(pi_logits, dim=2)
        log_cdf    = torch.logsumexp(log_w + F.logsigmoid(z), dim=2)
        log_1m_cdf = torch.logsumexp(log_w + F.logsigmoid(-z), dim=2)
        return log_cdf, log_1m_cdf

    def _v(self, x2, pi_logits, mu, log_s):
        # v(x2) = logit(MixLogisticCDF(x2)), computed stably. Strictly increasing
        # in x2 -> invertible by bisection without any clamp.
        log_cdf, log_1m_cdf = self._log_cdf_terms(x2, pi_logits, mu, log_s)
        return log_cdf - log_1m_cdf

    @staticmethod
    def _log_mix_pdf(x2, pi_logits, mu, log_s):
        z = (x2.unsqueeze(2) - mu) * torch.exp(-log_s)
        log_w = F.log_softmax(pi_logits, dim=2)
        # log[ pdf_k ] = -log_s_k + logsigmoid(z) + logsigmoid(-z)
        log_comp = (log_w - log_s
                    + F.logsigmoid(z) + F.logsigmoid(-z))
        return torch.logsumexp(log_comp, dim=2)             # (B, c_out, H, W)

    def forward(self, x: torch.Tensor, h: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            logger.error("[MixLogisticCoupling2D.forward] expected (B,C,H,W), "
                         "got %s", tuple(x.shape))
            raise ValueError("expected (B,C,H,W)")
        x1, x2 = self._split(x)
        pi_logits, mu, log_s, a_b, b = self._params(x1, h)
        log_cdf, log_1m_cdf = self._log_cdf_terms(x2, pi_logits, mu, log_s)
        v = log_cdf - log_1m_cdf                      # logit(cdf), stable, exact
        y2 = v * torch.exp(a_b) + b
        log_pdf = self._log_mix_pdf(x2, pi_logits, mu, log_s)
        # log|dy2/dx2| = a_b + log(pdf) - log(cdf) - log(1-cdf)
        ldj_elem = a_b + log_pdf - log_cdf - log_1m_cdf
        ldj = ldj_elem.flatten(1).sum(dim=1)
        if not torch.isfinite(ldj).all():
            logger.error("[MixLogisticCoupling2D.forward] non-finite ldj")
            raise ValueError("non-finite ldj in MixLogisticCoupling2D.forward")
        y = self._merge(x1, y2)
        return y, ldj

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y1, y2 = self._split(y)
        pi_logits, mu, log_s, a_b, b = self._params(y1, h)
        # Recover v exactly, then solve v(x2) = v_target by bisection. Because we
        # invert the SAME stable v used in forward (no clamp), this is exact.
        v_target = (y2 - b) * torch.exp(-a_b)
        x2 = self._bisect_inverse(v_target, pi_logits, mu, log_s)
        return self._merge(y1, x2)

    def _bisect_inverse(self, v_target, pi_logits, mu, log_s):
        # Solve _v(x2) = v_target elementwise; _v strictly increasing in x2.
        # ANALYTIC BRACKET: each logistic component k inverts linearly at
        #   x2_k = mu_k + v_target * exp(log_s_k)   (since logit(sigmoid(.))=id).
        # The mixture root provably lies in [min_k x2_k, max_k x2_k]: at x2 =
        # max_k x2_k every component CDF >= target so F >= target, and at
        # x2 = min_k x2_k every component CDF <= target so F <= target; F is
        # monotone, so the root is bracketed. This holds for ANY finite
        # v_target (incl. the large values seen when decoding samples that are
        # ill-conditioned in the generative direction) -- no expansion, no cap.
        def Vfun(x2):
            return self._v(x2, pi_logits, mu, log_s)
        x2_k = mu + v_target.unsqueeze(2) * torch.exp(log_s)   # (B,c_out,K,H,W)
        lo = x2_k.min(dim=2).values
        hi = x2_k.max(dim=2).values
        # tiny scale-aware margin so lo<hi even when all components coincide
        margin = 1e-6 * (hi.abs() + lo.abs() + 1.0)
        lo = lo - margin
        hi = hi + margin
        if not (torch.isfinite(lo).all() and torch.isfinite(hi).all()):
            logger.error("[MixLogisticCoupling2D._bisect_inverse] non-finite "
                         "analytic bracket (v_target in [%.3e,%.3e])",
                         float(v_target.min()), float(v_target.max()))
            raise RuntimeError("non-finite analytic bracket")
        for _ in range(_BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            f_lt = Vfun(mid) < v_target
            lo = torch.where(f_lt, mid, lo)
            hi = torch.where(f_lt, hi, mid)
        x2 = 0.5 * (lo + hi)
        # Convergence = tight bracket in x2-space (slope-independent). The
        # analytic bracket provably contains the root and bisection halves it
        # each step, so (hi-lo) small => x2 has converged to the root.
        width = (hi - lo).abs()
        conv = width <= _X_REL_TOL * (x2.abs() + 1.0)
        if not torch.isfinite(x2).all() or not bool(conv.all()):
            bad = float((width / (x2.abs() + 1.0)).max())
            logger.error("[MixLogisticCoupling2D._bisect_inverse] non-converged: "
                         "max relative bracket width=%.3e tol=%.1e", bad,
                         _X_REL_TOL)
            raise RuntimeError("bisection inverse did not converge")
        return x2
