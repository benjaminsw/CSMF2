# =============================================================================
# SEQREF-EXPERTS v0.7 -- src.base_experts  (was STEP-1_1 v0.4)
# LIFETIME: KEEP
# Purpose: conditional flow experts with a shared Conditioner. Each maps
#          x -> z through a stack of coupling layers and returns (z, logdet).
#          Base density: standard Normal N(0, I).
# CONVENTION: .log_prob(x | y) returns shape (B,). .sample(n, y) returns x.
#             No fallback / mock / pass. Failures raise with logger.error.
# Changelog (v0.6 -> v0.7, SEQREF-NICER3):
#   * CondNICE gains use_permute (default False) and post_init_std (default
#     None). use_permute=True interleaves FixedPermute (seed 7770+i, fixed,
#     arm/seed-invariant) between additive couplings; DiagScale stays last.
#   * build_expert: post_init_std now legal for {'realnvp','nice'};
#     use_permute legal for 'nice' ONLY. Explicit raise otherwise.
#     Defaults preserve v0.6 models exactly.
# Changelog (v0.5 -> v0.6, SEQREF-SMAX):
#   * CondRealNVP gains s_max (default 2.0) and post_init_std (default 0.0)
#     kwargs, threaded to every RealNVPCoupling. Defaults preserve v0.5.
#   * build_expert: s_max now legal for 'realnvp' and 'glow' only;
#     post_init_std legal for 'realnvp' only. Explicit raise otherwise
#     (incl. 'realnvp_image', closed as on-record negative -- no silent thread).
# Changelog (v0.4 -> v0.5, SEQREF):
#   * R2: CondRealNVPImage expert; registry 'realnvp_image'.
# Changelog (v0.3 -> v0.4):
#   * CondGlow film_gain_init (CondGlow retained as dead code -- do not build).
# Changelog (v0.2 -> v0.3):
#   * CondGlow expert; RealNVP/NICE thread FiLM kwargs.
# Update summary:
#   v0.7 wires SEQREF-NICER3: FixedPermute (R3, the single A/B difference)
#   and the post_init_std carry-over become config-reachable for NICE, with
#   the same hard-gating discipline as SMAX. RealNVP paths untouched.
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.7"
__abbr__ = "SEQREF-EXPERTS"

import math
import torch
import torch.nn as nn

from .conditioner import Conditioner
from .flows.nice_layer import NICECoupling, DiagScale, FixedPermute
from .flows.realnvp_layer import RealNVPCoupling
from .flows.realnvp_image_layer import RealNVPImageCoupling
from .flows.nsf_layer import NSFCoupling


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
    # SEQREF-NICER3: use_permute=True (R3) interleaves FixedPermute between
    # couplings (n_layers-1 permutes; DiagScale stays last). Permute seeds are
    # 7770+i -- fixed constants, invariant across arms and run seeds so paired
    # A/B runs and multi-seed runs share identical permutations.
    _PERMUTE_SEED_BASE = 7770

    def __init__(self, *, dim: int, h_dim: int, conditioner: Conditioner,
                 hidden: int = 256, n_layers: int = 4, use_film: bool = True,
                 use_permute: bool = False, post_init_std: float | None = None,
                 film_hidden: int = 64, film_depth: int = 1,
                 film_use_gelu: bool = False):
        super().__init__(dim=dim, h_dim=h_dim, conditioner=conditioner)
        for i in range(n_layers):
            self.layers.append(NICECoupling(
                dim=dim, hidden=hidden, h_dim=h_dim,
                flip=bool(i % 2), use_film=use_film,
                post_init_std=post_init_std,
                film_hidden=film_hidden, film_depth=film_depth,
                film_use_gelu=film_use_gelu))
            if use_permute and i < n_layers - 1:
                self.layers.append(FixedPermute(dim,
                                                seed=self._PERMUTE_SEED_BASE + i))
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


# ---- RealNVP ----------------------------------------------------------------
class CondRealNVP(_BaseExpert):
    def __init__(self, *, dim: int, h_dim: int, conditioner: Conditioner,
                 hidden: int = 256, n_layers: int = 6, use_film: bool = True,
                 s_max: float = 2.0, post_init_std: float = 0.0,
                 film_hidden: int = 64, film_depth: int = 1,
                 film_use_gelu: bool = False):
        super().__init__(dim=dim, h_dim=h_dim, conditioner=conditioner)
        for i in range(n_layers):
            self.layers.append(RealNVPCoupling(
                dim=dim, hidden=hidden, h_dim=h_dim,
                flip=bool(i % 2), use_film=use_film,
                s_max=s_max, post_init_std=post_init_std,
                film_hidden=film_hidden, film_depth=film_depth,
                film_use_gelu=film_use_gelu))


# ---- RealNVP-Image (R2) -------------------------------------------------------
class CondRealNVPImage(_BaseExpert):
    # R2: image checkerboard CNN coupling stack. Alternating parity via flip.
    # NOTE: R2 is an on-record negative for s4/n0.10 seed0 (SEQREF-RECONRESCUE
    # v0.4). Retained; do not extend without a fresh decision.
    def __init__(self, *, dim: int, h_dim: int, conditioner: Conditioner,
                 hidden: int = 256, n_layers: int = 6, use_film: bool = True,
                 film_hidden: int = 128, film_depth: int = 2,
                 film_use_gelu: bool = True, s_max: float = 2.0,
                 cnn_channels: int = 64, film_gain_init: float = 0.3):
        super().__init__(dim=dim, h_dim=h_dim, conditioner=conditioner)
        for i in range(n_layers):
            self.layers.append(RealNVPImageCoupling(
                dim=dim, hidden=hidden, h_dim=h_dim,
                flip=bool(i % 2), use_film=use_film, s_max=s_max,
                film_hidden=film_hidden, film_depth=film_depth,
                film_use_gelu=film_use_gelu, cnn_channels=cnn_channels,
                film_gain_init=film_gain_init))


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
    "realnvp": CondRealNVP,
    "realnvp_image": CondRealNVPImage,
    "nsf":     CondNSF,
    "glow":    CondGlow,
}


def build_expert(name: str, *, dim: int, h_dim: int,
                 conditioner: Conditioner, hidden: int,
                 use_film: bool, **kwargs) -> _BaseExpert:
    # kwargs can include:
    #   FiLM:    film_hidden, film_depth, film_use_gelu  (NICE, RealNVP, Glow)
    #   RealNVP: s_max, post_init_std  (SEQREF-SMAX; realnvp only)
    #   Glow:    s_max, image_shape, inv1x1_seed_base, film_gain_init (Glow only)
    #   n_layers (NICE / RealNVP / realnvp_image / NSF / Glow)
    # NSF rejects film_kwargs; passing them raises.
    if name not in EXPERTS:
        logger.error("[build_expert] unknown expert '%s' (known: %s)",
                     name, list(EXPERTS.keys()))
        raise ValueError(f"unknown expert {name!r}")

    film_keys = {"film_hidden", "film_depth", "film_use_gelu"}
    realnvp_keys = {"s_max", "post_init_std"}
    nice_keys = {"use_permute", "post_init_std"}
    glow_only_keys = {"image_shape", "inv1x1_seed_base", "film_gain_init"}

    unknown = set(kwargs) - (film_keys | realnvp_keys | nice_keys
                             | glow_only_keys | {"n_layers"})
    if unknown:
        logger.error("[build_expert] unknown kwargs %s for expert=%r",
                     sorted(unknown), name)
        raise ValueError(f"unknown kwargs {sorted(unknown)} for expert={name!r}")

    # SEQREF-SMAX/NICER3 validation: s_max -> realnvp/glow;
    # post_init_std -> realnvp/nice; use_permute -> nice only.
    if "post_init_std" in kwargs and name not in ("realnvp", "nice"):
        logger.error("[build_expert] post_init_std only supported for expert in"
                     " ('realnvp','nice'), got expert=%r", name)
        raise ValueError(
            f"post_init_std only supported for ('realnvp','nice'); got {name!r}")
    if "use_permute" in kwargs and name != "nice":
        logger.error("[build_expert] use_permute only supported for "
                     "expert='nice', got expert=%r", name)
        raise ValueError(
            f"use_permute only supported for expert='nice'; got {name!r}")
    if "s_max" in kwargs and name not in ("realnvp", "glow"):
        logger.error("[build_expert] s_max only supported for expert in "
                     "('realnvp','glow'), got expert=%r "
                     "(realnvp_image is a closed on-record negative)", name)
        raise ValueError(
            f"s_max only supported for expert in ('realnvp','glow'); got {name!r}")

    film_kwargs = {k: kwargs[k] for k in film_keys if k in kwargs}
    if film_kwargs and name == "nsf":
        logger.error("[build_expert] film_kwargs not supported for expert='nsf'"
                     ", got %s", list(film_kwargs))
        raise ValueError(f"film_kwargs not supported for expert='nsf'; "
                         f"got {list(film_kwargs)}")

    glow_only_kwargs = {k: kwargs[k] for k in glow_only_keys if k in kwargs}
    if glow_only_kwargs and name != "glow":
        logger.error("[build_expert] glow-only kwargs %s only supported for "
                     "expert='glow', got expert=%r",
                     list(glow_only_kwargs), name)
        raise ValueError(
            f"glow-only kwargs only supported for expert='glow'; got expert={name!r}")

    realnvp_kwargs = ({k: float(kwargs[k]) for k in realnvp_keys if k in kwargs}
                      if name == "realnvp" else {})
    nice_kwargs = {}
    if name == "nice":
        if "use_permute" in kwargs:
            nice_kwargs["use_permute"] = bool(kwargs["use_permute"])
        if "post_init_std" in kwargs:
            nice_kwargs["post_init_std"] = (
                None if kwargs["post_init_std"] is None
                else float(kwargs["post_init_std"]))
    n_layers = kwargs.get("n_layers")

    if name == "nice":
        ctor_kwargs = dict(dim=dim, h_dim=h_dim, conditioner=conditioner,
                           hidden=hidden, use_film=use_film,
                           **film_kwargs, **nice_kwargs)
        if n_layers is not None:
            ctor_kwargs["n_layers"] = n_layers
        return CondNICE(**ctor_kwargs)

    if name == "realnvp":
        ctor_kwargs = dict(dim=dim, h_dim=h_dim, conditioner=conditioner,
                           hidden=hidden, use_film=use_film,
                           **film_kwargs, **realnvp_kwargs)
        if n_layers is not None:
            ctor_kwargs["n_layers"] = n_layers
        return CondRealNVP(**ctor_kwargs)

    if name == "realnvp_image":
        ctor_kwargs = dict(dim=dim, h_dim=h_dim, conditioner=conditioner,
                           hidden=hidden, use_film=use_film, **film_kwargs)
        if n_layers is not None:
            ctor_kwargs["n_layers"] = n_layers
        return CondRealNVPImage(**ctor_kwargs)

    if name == "nsf":
        ctor_kwargs = dict(dim=dim, h_dim=h_dim, conditioner=conditioner,
                           hidden=hidden, use_film=use_film)
        if n_layers is not None:
            ctor_kwargs["n_layers"] = n_layers
        return CondNSF(**ctor_kwargs)

    # glow (dead code path -- retained for reference; do not build)
    glow_kwargs = dict(glow_only_kwargs)
    if "s_max" in kwargs:
        glow_kwargs["s_max"] = float(kwargs["s_max"])
    ctor_kwargs = dict(dim=dim, h_dim=h_dim, conditioner=conditioner,
                       hidden=hidden, use_film=use_film,
                       **film_kwargs, **glow_kwargs)
    if n_layers is not None:
        ctor_kwargs["n_layers"] = n_layers
    return CondGlow(**ctor_kwargs)
