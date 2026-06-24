# =============================================================================
# STEP-1_1 v0.6 -- models.experts
# Purpose: four conditional flow experts with a shared Conditioner. Each maps
#          x -> z through a stack of coupling layers and returns (z, logdet).
#          Base density: standard Normal N(0, I).
# CONVENTION: .log_prob(x | y) returns shape (B,). .sample(n, y) returns x.
#             No fallback / mock / pass. Failures raise with logger.error.
# Changelog (v0.5 -> v0.6):
#   * NEW expert 'nice_mix' (CondNICEMix): additive NICE + FixedPermute mixing
#     between couplings (NCP-N8 expressiveness ablation). Separate class so
#     'nice' stays byte-identical; registered in EXPERTS and given its own
#     build_expert branch (before the glow fallthrough) that forwards n_layers.
#     Additive-only (no exp(s)/learned-1x1); permutation log-det 0.
# Changelog (v0.4 -> v0.5):
#   * IMG-RNVP v0.1: build_expert routes expert='realnvp' with
#     realnvp_type='image' -> ImageCondRealNVP (CNN image couplings, Stage
#     1.4b-A). New image_keys allowlist {realnvp_type, image_n_couplings,
#     image_hidden, image_mask_type, image_s_max}, guarded realnvp-only.
#     realnvp_type='flat' (default) is unchanged -> CondRealNVP. image_mask_type
#     is consumed (cfg-validated, checkerboard-only) but not forwarded.
# Changelog (v0.3 -> v0.4):
#   * CondGlow accepts film_gain_init (default 0.3) and forwards to GlowStep
#     -> AffineCoupling2D. Drives the residual-FiLM contribution introduced
#     in models/flows/glow/affine_coupling_2d.py v0.3.
#   * build_expert glow_keys allowlist gains 'film_gain_init'.
# Changelog (v0.2 -> v0.3):
#   * NEW: CondGlow expert. CondRealNVP/CondNICE thread FiLM kwargs.
#     build_expert routes 'glow' and validates kwargs per expert.
# Changelog (v0.1 -> v0.2):
#   * CondNICE accepts film_hidden / film_depth / film_use_gelu kwargs.
# Changelog (NEW in v0.1):
#   * Introduced. CondNICE, CondRealNVP, CondNSF; FiLM by default.
# Update summary:
#   v0.6 adds the nice_mix expert (NCP-N8) as an additive NICE + fixed-perm
#   variant, isolated behind its own class/branch; nice/realnvp/nsf/glow paths
#   untouched. v0.5 added the image-RealNVP routing (Stage 1.4b-A) without
#   touching the flat/NICE/NSF/Glow paths; realnvp_type defaults to 'flat'.
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.4"
__abbr__ = "STEP-1_1"

import math
import torch
import torch.nn as nn

from .conditioner import Conditioner
from .flows.nice_layer import NICECoupling, DiagScale, FixedPermute
from .flows.realnvp_layer import RealNVPCoupling
from .flows.nsf_layer import NSFCoupling
from .flows.glow.squeeze import squeeze2x2, unsqueeze2x2
from .flows.glow.glow_step import GlowStep


def _gaussian_logprob(z: torch.Tensor) -> torch.Tensor:
    # z: (B, D). Returns (B,) log N(z; 0, I).
    return -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1)


class _BaseExpert(nn.Module):
    def __init__(self, *, dim: int, h_dim: int, conditioner: Conditioner):
        super().__init__()
        self.dim = dim
        self.h_dim = h_dim
        self.cond = conditioner
        self.layers: nn.ModuleList = nn.ModuleList()   # filled by subclass

    def encode(self, x: torch.Tensor, h: torch.Tensor):
        ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        z = x
        for layer in self.layers:
            z, d = layer(z, h)
            ldj = ldj + d
        return z, ldj

    def decode(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x, h)
        return x

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            logger.error("[Expert.log_prob] expected x (B, D), got %s", tuple(x.shape))
            raise ValueError(f"expected x (B, D), got {tuple(x.shape)}")
        h = self.cond(y)
        z, ldj = self.encode(x, h)
        lp = _gaussian_logprob(z) + ldj
        if not torch.isfinite(lp).all():
            logger.error("[Expert.log_prob] non-finite log_prob")
            raise ValueError("non-finite log_prob")
        return lp

    @torch.no_grad()
    def sample(self, n: int, y_one: torch.Tensor) -> torch.Tensor:
        # y_one: (1, 1, H, W). Returns (n, D).
        if y_one.dim() != 4 or y_one.size(0) != 1:
            logger.error("[Expert.sample] expected y_one (1,1,H,W), got %s",
                         tuple(y_one.shape))
            raise ValueError("y_one must have batch=1")
        h = self.cond(y_one).expand(n, -1)
        z = torch.randn(n, self.dim, device=y_one.device, dtype=y_one.dtype)
        return self.decode(z, h)


# ---- NICE -------------------------------------------------------------------
class CondNICE(_BaseExpert):
    def __init__(self, *, dim: int, h_dim: int, conditioner: Conditioner,
                 hidden: int = 256, n_layers: int = 4, use_film: bool = True,
                 film_hidden: int = 64, film_depth: int = 1,
                 film_use_gelu: bool = False):
        super().__init__(dim=dim, h_dim=h_dim, conditioner=conditioner)
        for i in range(n_layers):
            self.layers.append(NICECoupling(
                dim=dim, hidden=hidden, h_dim=h_dim,
                flip=bool(i % 2), use_film=use_film,
                film_hidden=film_hidden, film_depth=film_depth,
                film_use_gelu=film_use_gelu))
        self.layers.append(DiagScaleWrapper(dim))


class DiagScaleWrapper(nn.Module):
    # wraps DiagScale to match the (x, h) signature used by _BaseExpert.encode
    def __init__(self, dim: int):
        super().__init__()
        self.scale = DiagScale(dim)

    def forward(self, x, h):
        return self.scale(x)

    def inverse(self, y, h):
        return self.scale.inverse(y)


class CondNICEMix(_BaseExpert):
    """NCP-N8 ablation: additive NICE + fixed permutation mixing between
    couplings. Identical to CondNICE except a FixedPermute is inserted between
    successive NICECouplings (not after the last one, and before the final
    DiagScaleWrapper). Additive-only -- NO exp(s), NO learned 1x1 -- so it stays
    a distinct, NICE-flavoured expert (not RealNVP/Glow). Kept as a SEPARATE
    class so plain 'nice' remains byte-identical and the frozen N0 baseline is
    reproducible. Permutation log-det is 0, so encode/decode and the f64 logdet
    sanity check carry over unchanged."""
    def __init__(self, *, dim: int, h_dim: int, conditioner: Conditioner,
                 hidden: int = 256, n_layers: int = 4, use_film: bool = True,
                 film_hidden: int = 64, film_depth: int = 1,
                 film_use_gelu: bool = False):
        super().__init__(dim=dim, h_dim=h_dim, conditioner=conditioner)
        for i in range(n_layers):
            self.layers.append(NICECoupling(
                dim=dim, hidden=hidden, h_dim=h_dim,
                flip=bool(i % 2), use_film=use_film,
                film_hidden=film_hidden, film_depth=film_depth,
                film_use_gelu=film_use_gelu))
            if i < n_layers - 1:
                # fixed, deterministic per-position permutation (seed varies by i
                # so layers don't share the same shuffle); log-det 0.
                self.layers.append(FixedPermute(dim, seed=1000 + i))
        self.layers.append(DiagScaleWrapper(dim))


# ---- RealNVP ----------------------------------------------------------------
class CondRealNVP(_BaseExpert):
    def __init__(self, *, dim: int, h_dim: int, conditioner: Conditioner,
                 hidden: int = 256, n_layers: int = 6, use_film: bool = True,
                 film_hidden: int = 64, film_depth: int = 1,
                 film_use_gelu: bool = False):
        super().__init__(dim=dim, h_dim=h_dim, conditioner=conditioner)
        for i in range(n_layers):
            self.layers.append(RealNVPCoupling(
                dim=dim, hidden=hidden, h_dim=h_dim,
                flip=bool(i % 2), use_film=use_film,
                film_hidden=film_hidden, film_depth=film_depth,
                film_use_gelu=film_use_gelu))


# ---- NSF --------------------------------------------------------------------
class CondNSF(_BaseExpert):
    def __init__(self, *, dim: int, h_dim: int, conditioner: Conditioner,
                 hidden: int = 256, n_layers: int = 6, K: int = 8, B: float = 3.0,
                 use_film: bool = True):
        super().__init__(dim=dim, h_dim=h_dim, conditioner=conditioner)
        for i in range(n_layers):
            self.layers.append(NSFCoupling(dim=dim, hidden=hidden, h_dim=h_dim,
                                           flip=bool(i % 2), K=K, B=B,
                                           use_film=use_film))


# ---- Glow -------------------------------------------------------------------
class CondGlow(_BaseExpert):
    # External interface: flat (B, dim=C*H*W). Internal: image (B, C', H/2, W/2)
    # after squeeze. Stores K GlowSteps in self.layers; encode/decode wrap with
    # squeeze/unsqueeze. Default image_shape is (1, 28, 28); override for non-MNIST.
    def __init__(self, *, dim: int, h_dim: int, conditioner: Conditioner,
                 hidden: int = 256, n_layers: int = 8, use_film: bool = True,
                 s_max: float = 2.0,
                 film_hidden: int = 128, film_depth: int = 2,
                 film_use_gelu: bool = True,
                 image_shape: tuple[int, int, int] = (1, 28, 28),
                 inv1x1_seed_base: int = 0,
                 film_gain_init: float = 0.3):
        super().__init__(dim=dim, h_dim=h_dim, conditioner=conditioner)
        if not use_film:
            logger.error("[CondGlow] use_film=False unsupported (FiLM locked in v0.1)")
            raise ValueError("CondGlow requires use_film=True")
        if film_gain_init < 0.0:
            logger.error("[CondGlow] film_gain_init must be >=0, got %s",
                         film_gain_init)
            raise ValueError(f"film_gain_init must be >=0, got {film_gain_init}")
        C, H, W = image_shape
        if C * H * W != dim:
            logger.error("[CondGlow] image_shape %s != dim %d", image_shape, dim)
            raise ValueError(f"image_shape {image_shape} != dim {dim}")
        if H % 2 or W % 2:
            logger.error("[CondGlow] H,W must be even for 2x2 squeeze, got %dx%d",
                         H, W)
            raise ValueError("H,W must be even for 2x2 squeeze")
        self.image_shape = image_shape
        self.squeezed_c = C * 4
        for i in range(n_layers):
            self.layers.append(GlowStep(
                num_channels=self.squeezed_c,
                coupling_hidden=hidden, h_dim=h_dim,
                flip=bool(i % 2), s_max=s_max,
                film_hidden=film_hidden, film_depth=film_depth,
                film_use_gelu=film_use_gelu,
                inv1x1_seed=inv1x1_seed_base * 1000 + i,
                film_gain_init=film_gain_init))

    def _to_image(self, x_flat: torch.Tensor) -> torch.Tensor:
        if x_flat.dim() != 2 or x_flat.size(1) != self.dim:
            logger.error("[CondGlow._to_image] expected (B, %d), got %s",
                         self.dim, tuple(x_flat.shape))
            raise ValueError("flat dim mismatch")
        C, H, W = self.image_shape
        return x_flat.view(-1, C, H, W)

    def encode(self, x: torch.Tensor, h: torch.Tensor):
        # x: (B, dim). Wraps the GlowStep stack with squeeze / unsqueeze.
        x_img = self._to_image(x)
        z_sq  = squeeze2x2(x_img)
        ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        z = z_sq
        for layer in self.layers:
            z, d = layer(z, h)
            ldj = ldj + d
        z_img = unsqueeze2x2(z)
        return z_img.flatten(1), ldj

    def decode(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z_img = self._to_image(z)
        z_sq  = squeeze2x2(z_img)
        x_sq = z_sq
        for layer in reversed(self.layers):
            x_sq = layer.inverse(x_sq, h)
        x_img = unsqueeze2x2(x_sq)
        return x_img.flatten(1)

    @torch.no_grad()
    def init_actnorm(self, x_first_batch: torch.Tensor,
                     y_first_batch: torch.Tensor) -> None:
        # Walk one batch through the network, initialising each Actnorm in turn.
        if x_first_batch.dim() != 2:
            logger.error("[CondGlow.init_actnorm] expected x flat (B, D), got %s",
                         tuple(x_first_batch.shape))
            raise ValueError("expected x flat (B, D)")
        was_training = self.training
        self.eval()
        h = self.cond(y_first_batch)
        x_img = self._to_image(x_first_batch)
        z = squeeze2x2(x_img)
        for i, step in enumerate(self.layers):
            step.actnorm.init_from_batch(z)
            z, _ = step.actnorm(z)
            z, _ = step.inv1x1(z)
            z, _ = step.coupling(z, h)
            logger.info("[CondGlow.init_actnorm] step %d done", i)
        if was_training:
            self.train()


EXPERTS = {
    "nice":    CondNICE,
    "nice_mix": CondNICEMix,
    "realnvp": CondRealNVP,
    "nsf":     CondNSF,
    "glow":    CondGlow,
}


def build_expert(name: str, *, dim: int, h_dim: int,
                 conditioner: Conditioner, hidden: int,
                 use_film: bool, **kwargs) -> _BaseExpert:
    # kwargs can include:
    #   FiLM:  film_hidden, film_depth, film_use_gelu  (NICE, RealNVP, Glow)
    #   Glow:  s_max, image_shape, inv1x1_seed_base, n_layers  (Glow only)
    #   RealNVP: n_layers  (also for NSF / Glow)
    # NSF rejects film_kwargs; passing them raises.
    if name not in EXPERTS:
        logger.error("[build_expert] unknown expert '%s' (known: %s)",
                     name, list(EXPERTS.keys()))
        raise ValueError(f"unknown expert {name!r}")

    film_keys = {"film_hidden", "film_depth", "film_use_gelu"}
    glow_keys = {"s_max", "image_shape", "inv1x1_seed_base", "film_gain_init"}
    # IMG-RNVP v0.1: image RealNVP (Stage 1.4b-A) kwargs, realnvp-only.
    image_keys = {"realnvp_type", "image_n_couplings", "image_hidden",
                  "image_mask_type", "image_s_max"}

    unknown = set(kwargs) - (film_keys | glow_keys | image_keys | {"n_layers"})
    if unknown:
        logger.error("[build_expert] unknown kwargs %s for expert=%r",
                     sorted(unknown), name)
        raise ValueError(f"unknown kwargs {sorted(unknown)} for expert={name!r}")

    film_kwargs = {k: kwargs[k] for k in film_keys if k in kwargs}
    if film_kwargs and name == "nsf":
        logger.error("[build_expert] film_kwargs not supported for expert='nsf'"
                     ", got %s", list(film_kwargs))
        raise ValueError(f"film_kwargs not supported for expert='nsf'; "
                         f"got {list(film_kwargs)}")

    glow_kwargs = {k: kwargs[k] for k in glow_keys if k in kwargs}
    if glow_kwargs and name != "glow":
        logger.error("[build_expert] glow_kwargs %s only supported for "
                     "expert='glow', got expert=%r",
                     list(glow_kwargs), name)
        raise ValueError(
            f"glow_kwargs only supported for expert='glow'; got expert={name!r}")

    image_kwargs = {k: kwargs[k] for k in image_keys if k in kwargs}
    if image_kwargs and name != "realnvp":
        logger.error("[build_expert] image_kwargs %s only supported for "
                     "expert='realnvp', got expert=%r",
                     list(image_kwargs), name)
        raise ValueError(
            f"image_kwargs only supported for expert='realnvp'; got expert={name!r}")

    n_layers = kwargs.get("n_layers")

    if name == "nice":
        ctor_kwargs = dict(dim=dim, h_dim=h_dim, conditioner=conditioner,
                           hidden=hidden, use_film=use_film, **film_kwargs)
        if n_layers is not None:
            ctor_kwargs["n_layers"] = n_layers
        return CondNICE(**ctor_kwargs)

    if name == "nice_mix":
        ctor_kwargs = dict(dim=dim, h_dim=h_dim, conditioner=conditioner,
                           hidden=hidden, use_film=use_film, **film_kwargs)
        if n_layers is not None:
            ctor_kwargs["n_layers"] = n_layers
        return CondNICEMix(**ctor_kwargs)

    if name == "realnvp":
        realnvp_type = kwargs.get("realnvp_type", "flat")
        if realnvp_type == "image":
            from .image_cond_realnvp import ImageCondRealNVP
            # image_mask_type is consumed (validated in cfg, checkerboard-only)
            # but not forwarded: the coupling hardcodes checkerboard in v0.1.
            return ImageCondRealNVP(
                dim=dim, h_dim=h_dim, conditioner=conditioner,
                channels=1, img_hw=28,
                image_n_couplings=kwargs.get("image_n_couplings", 8),
                image_hidden=kwargs.get("image_hidden", 64),
                image_s_max=kwargs.get("image_s_max", 2.0),
                use_film=use_film, **film_kwargs)
        ctor_kwargs = dict(dim=dim, h_dim=h_dim, conditioner=conditioner,
                           hidden=hidden, use_film=use_film, **film_kwargs)
        if n_layers is not None:
            ctor_kwargs["n_layers"] = n_layers
        return CondRealNVP(**ctor_kwargs)

    if name == "nsf":
        ctor_kwargs = dict(dim=dim, h_dim=h_dim, conditioner=conditioner,
                           hidden=hidden, use_film=use_film)
        if n_layers is not None:
            ctor_kwargs["n_layers"] = n_layers
        return CondNSF(**ctor_kwargs)

    # glow
    ctor_kwargs = dict(dim=dim, h_dim=h_dim, conditioner=conditioner,
                       hidden=hidden, use_film=use_film,
                       **film_kwargs, **glow_kwargs)
    if n_layers is not None:
        ctor_kwargs["n_layers"] = n_layers
    return CondGlow(**ctor_kwargs)
