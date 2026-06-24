# =============================================================================
# STEP-2_3 v0.1 -- experiments.step_2_3.gradnorm  (S2.3 GradNorm + L0-norm)
# Purpose: loss-balancing primitives for the Stage 2.3 hybrid objective.
#          Two LEVELS, applied in order (see S2.3-PLAN v0.4):
#            Level 1  L0Normalizer  -- value-scale: L_k / stopgrad(L0_k), so every
#                                      term starts ~1.0. FIXED initial references.
#            Level 2  GradNorm      -- gradient-balance: learnable per-term weights
#                                      rebalanced from gradient norms + relative
#                                      training rates (Chen+ 2018).
#          v0.1 ships BOTH but GradNorm is OFF by default for 2.3-A (logged/hooked
#          only); enable once transport/calibration enter (>=3 terms).
# CONVENTION: no fallback/mock. Non-finite -> logger.error + raise. The shared
#          parameter set for GradNorm grad-norms is the LAST shared layer the
#          caller passes in (paper uses W); we take an explicit param list so the
#          caller controls it -- no silent global .parameters() scan.
# Changelog (NEW in v0.1):
#   * L0Normalizer: record_reference() (first-few-batches mean, frozen) +
#     normalize() (divide by stopgrad reference). State is serializable.
#   * GradNorm: learnable log-weights w_k>=0, target from mean grad-norm x
#     (inverse training rate)^alpha, L1 grad-loss; .weights(), .gradnorm_step().
#   * grad_norm_of(loss, params) + grad_cosine(g_a, g_b) instrumentation helpers.
# Update summary:
#   v0.1 is the only new loss-balancing math for 2.3. The trainer calls
#   L0Normalizer every step (Level 1) and, when enabled, GradNorm (Level 2).
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-2_3"

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Level 1 -- L0-normalization (value scale; ON from v0.1)
# --------------------------------------------------------------------------- #
class L0Normalizer:
    """Divide each loss term by a FROZEN initial reference so all terms start
    ~1.0. References are the mean of the first `warmup_batches` raw values per
    term, recorded once then never updated (NOT per-batch -- that is unstable
    and hides progress; see S2.3-PLAN technique verdict, B vs A).

    Usage:
        norm = L0Normalizer(["nll", "rec"], warmup_batches=5)
        # during warmup:
        norm.observe({"nll": l_nll.item(), "rec": l_rec.item()})
        # once norm.ready:
        terms_norm = norm.normalize({"nll": l_nll, "rec": l_rec})
    """
    def __init__(self, names, warmup_batches: int = 5, eps: float = 1e-8):
        if warmup_batches < 1:
            logger.error("[L0Normalizer] warmup_batches must be >=1, got %s",
                         warmup_batches)
            raise ValueError("warmup_batches must be >=1")
        self.names = list(names)
        self.warmup_batches = int(warmup_batches)
        self.eps = float(eps)
        self._sums = {n: 0.0 for n in self.names}
        self._count = 0
        self.refs: dict | None = None          # frozen L0_k once ready

    @property
    def ready(self) -> bool:
        return self.refs is not None

    def observe(self, raw_values: dict) -> None:
        """Accumulate raw loss values during warmup. Freezes refs once
        warmup_batches have been seen."""
        if self.ready:
            return
        for n in self.names:
            if n not in raw_values:
                logger.error("[L0Normalizer] missing term %r in observe()", n)
                raise KeyError(f"missing term {n!r}")
            v = float(raw_values[n])
            if not (v == v):   # NaN guard
                logger.error("[L0Normalizer] NaN observed for term %r", n)
                raise RuntimeError(f"NaN warmup value for {n!r}")
            self._sums[n] += v
        self._count += 1
        if self._count >= self.warmup_batches:
            self.refs = {n: max(abs(self._sums[n] / self._count), self.eps)
                         for n in self.names}
            logger.info("[L0Normalizer] froze references after %d batches: %s",
                        self._count, self.refs)

    def normalize(self, terms: dict) -> dict:
        """L_k / stopgrad(L0_k). refs are python floats (already detached)."""
        if not self.ready:
            logger.error("[L0Normalizer] normalize() before references frozen")
            raise RuntimeError("L0 references not frozen yet (still in warmup)")
        out = {}
        for n, t in terms.items():
            if n not in self.refs:
                logger.error("[L0Normalizer] term %r has no reference", n)
                raise KeyError(f"no reference for {n!r}")
            out[n] = t / self.refs[n]
        return out

    def state_dict(self) -> dict:
        return {"names": self.names, "warmup_batches": self.warmup_batches,
                "eps": self.eps, "refs": self.refs, "count": self._count}


# --------------------------------------------------------------------------- #
# Instrumentation -- per-term gradient norms + pairwise cosine (LOG from v0.1)
# --------------------------------------------------------------------------- #
def grad_norm_of(loss: torch.Tensor, params) -> float:
    """L2 norm of d(loss)/d(params) over the given shared params, as a float.
    retain_graph=True so the caller can call again for other terms / the real
    backward. create_graph=False (we only need magnitudes, not 2nd order)."""
    params = [p for p in params if p.requires_grad]
    if not params:
        logger.error("[grad_norm_of] no grad-enabled params passed")
        raise ValueError("no grad-enabled params")
    grads = torch.autograd.grad(loss, params, retain_graph=True,
                                create_graph=False, allow_unused=True)
    sq = 0.0
    for g in grads:
        if g is not None:
            sq += float(g.pow(2).sum())
    return sq ** 0.5


def grad_vec(loss: torch.Tensor, params) -> torch.Tensor:
    """Flattened gradient vector over shared params (for grad_cosine)."""
    params = [p for p in params if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True,
                                create_graph=False, allow_unused=True)
    flat = []
    for g, p in zip(grads, params):
        flat.append((g if g is not None else torch.zeros_like(p)).flatten())
    return torch.cat(flat)


def grad_cosine(g_a: torch.Tensor, g_b: torch.Tensor) -> float:
    """cosine(g_a, g_b). Negative -> terms fight -> consider PCGrad (S2.3-PLAN)."""
    denom = (g_a.norm() * g_b.norm()).clamp_min(1e-12)
    return float((g_a @ g_b) / denom)


# --------------------------------------------------------------------------- #
# Level 2 -- GradNorm (gradient balance; OFF for 2.3-A, enable >=3 terms)
# --------------------------------------------------------------------------- #
class GradNorm(nn.Module):
    """GradNorm (Chen+ 2018). Learnable per-term weights w_k (via softplus on a
    raw parameter, so w_k >= 0). Each `gradnorm_step` measures per-term grad
    norms on a SHARED parameter set, builds the speed-adjusted target, and takes
    one optimizer step on the GradNorm L1 grad-loss. The returned weights then
    scale the terms in the MAIN objective.

    NOTE: weights are renormalized to sum to K after each update (paper keeps
    sum(w)=K so the overall LR is unchanged). DIRECTION-agnostic -- it balances
    magnitudes only; for direction conflict use PCGrad (monitor grad_cosine).
    """
    def __init__(self, names, alpha: float = 1.5, lr: float = 0.025):
        super().__init__()
        self.names = list(names)
        self.k = len(self.names)
        if self.k < 2:
            logger.error("[GradNorm] needs >=2 terms, got %d", self.k)
            raise ValueError("GradNorm needs >=2 terms")
        self.alpha = float(alpha)
        # raw params -> softplus -> w_k; init so all w_k ~ 1.0
        self._raw = nn.Parameter(torch.full((self.k,), 0.5413))  # softplus(0.5413)~1
        self.opt = torch.optim.Adam([self._raw], lr=lr)
        self._L0: dict | None = None     # initial per-term loss for training-rate

    def weights(self) -> torch.Tensor:
        w = torch.nn.functional.softplus(self._raw)
        return self.k * w / w.sum().clamp_min(1e-12)     # renorm sum -> K

    @torch.no_grad()
    def _record_L0(self, raw_losses: dict):
        self._L0 = {n: max(float(raw_losses[n]), 1e-8) for n in self.names}

    def gradnorm_step(self, raw_losses: dict, shared_params) -> dict:
        """One GradNorm update. raw_losses: {name: scalar tensor} (the UNWEIGHTED,
        already-L0-normalized terms). shared_params: the param set whose grad
        norms are balanced (paper: last shared layer). Returns a log dict."""
        if self._L0 is None:
            self._record_L0(raw_losses)
        # materialize once -- a generator would be exhausted after the first term
        shared_params = [p for p in shared_params if p.requires_grad]
        if not shared_params:
            logger.error("[GradNorm] no grad-enabled shared params")
            raise ValueError("no grad-enabled shared params")
        w = self.weights()                                # (K,) sum=K
        # per-term grad norm of (w_k * L_k) wrt shared params
        gnorms = []
        for i, n in enumerate(self.names):
            gnorms.append(grad_norm_of(w[i] * raw_losses[n], shared_params))
        gnorms_t = torch.tensor(gnorms, device=self._raw.device)
        gbar = gnorms_t.mean()
        # inverse training rate r_k = (L_k/L0_k) / mean_j(L_j/L0_j)
        ratios = torch.tensor(
            [float(raw_losses[n].detach()) / self._L0[n] for n in self.names],
            device=self._raw.device)
        inv_rate = ratios / ratios.mean().clamp_min(1e-12)
        target = (gbar * inv_rate.pow(self.alpha)).detach()
        grad_loss = (gnorms_t - target).abs().sum()
        self.opt.zero_grad()
        # grad_loss depends on w (through gnorms which scale linearly in w_i);
        # gnorms_t came from .grad (no graph to _raw), so recompute the w-path:
        # approximate GradNorm: d grad_loss / d w_i ~ sign(gnorm_i - target_i)*gnorm_i/w_i
        # -- standard implementations backprop through ||w_i * g||; here gnorms
        # were computed as floats, so apply the closed-form surrogate gradient.
        with torch.no_grad():
            sign = torch.sign(gnorms_t - target)
            # gnorm_i is linear in w_i -> d gnorm_i/d w_i = gnorm_i / w_i
            surrogate = sign * (gnorms_t / w.clamp_min(1e-12))
            self._raw.grad = surrogate * torch.sigmoid(self._raw)  # softplus' = sigmoid
        self.opt.step()
        return {"gradnorm_weights": self.weights().detach().tolist(),
                "grad_norms": gnorms,
                "target_norms": target.tolist(),
                "grad_loss": float(grad_loss)}
