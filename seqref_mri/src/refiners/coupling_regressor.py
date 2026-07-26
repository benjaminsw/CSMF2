# SEQREF-CPLREG v0.5 -- refiners.coupling_regressor
# LIFETIME: KEEP
# Purpose: Level-1 coupling-as-regressor refiner. dx = stack(x0_flat | h) - x0_flat
#          where the stack is the expert's coupling blocks used as a plain
#          deterministic regressor (no NLL, no invertibility requirement) and
#          h = RefinerConditioner([y_up, x0, Aᵀr0]). Flavor:
#            realnvp -> affine RealNVPCoupling stack (s_max=4, post_init_std=1e-3)
#            nice    -> additive NICECoupling stack + DiagScale (no permute)
#          Near-identity at init (small post init + DiagScale zeros) -> dx ~= 0.
#          Gate lives in gated_update.GatedUpdate (owned by CplRegRefiner).
# v0.5 (bugfix, pre-formal-I2): SpatialGate returns (B,1,H,W) but the
#     locked state is two-channel (B,2,H,W); the v0.4 exact-shape check made
#     every spatial-gate forward raise. Now g must be (B,1,H,W) and
#     broadcasts across the Re/Im channels in x1 = x0 + g * dx. (Latent on
#     the MRI path -- scalar-first is locked -- but introduced by v0.4.)
# v0.4 (SEQREF-I2 paired channel-contract rebuild):
#     * dim REQUIRED (silent 784 MNIST default removed; raise if absent).
#       MRI state is the complex two-channel image (EXEC 3.14): x0 is
#       (B,2,H,W), dim = flattened state numel (2*96*96 = 18432 at the
#       locked cell, subject to the section-6 memory smoke).
#     * in_channels REQUIRED on RefinerConditioner / SpatialGate /
#       CplRegRefiner. Channel semantics REPLACED: the legacy
#       [y_up, x0, Aᵀr0] stack has NO MRI analogue -- the conditioning
#       input is the LOCKED 3.8 stack [|x0|, Re(Aᴴr), Im(Aᴴr)] assembled
#       by refiners.channel_assembly (provenance-gated normalization).
#     * Spatial contract comments updated 28x28 -> cell-supplied H,W.
#       Architecture (stacks, gate, warm-start audit) unchanged: INHERIT.
# v0.3 (SEQREF-SPGATE V1): gate_mode in {scalar, spatial}. spatial adds a
#     SpatialGate head reading the RAW 3-channel input stream (the same raw
#     conditioning inputs as C, NOT the pooled conditioner features h, which
#     cannot produce a pixel map): Conv(3->16)->ReLU->Conv(16->1)->sigmoid
#     * g_max, no hard-coded spatial size. Init: last conv weight zero, bias
#     = logit(g_init/g_max) -> uniform g == g_init at startup. Param names
#     gate_spatial.* (FRESH(new) under warm start -- expected, recorded).
#     Scalar path byte-identical to v0.2.
# CONVENTION: no fallback/mock/pass. Warm-start is PARTIAL + AUDITED: load
#             shape-compatible tensors by name, fresh-init the rest, log every
#             tensor, raise if loaded numel fraction < min_loaded_fraction (0.8).
# Changelog (v0.1 -> v0.2):
#   * Warm-start EXCLUSION POLICY (identity-safety fix): base checkpoints were
#     trained as logit-space full reconstructors; loading their output heads
#     into a pixel-space residual regressor produced |dx|~1e2 at init (smoke).
#     load_warm_start now takes exclude_patterns (fnmatch); default policy
#     DEFAULT_EXCLUDE = layers.*.post.* , layers.*.scale.log_s , gate.* --
#     conditioner/FiLM/pre/mid load, output heads stay fresh (post_init 1e-3)
#     -> dx ~= 0, x1 ~= x0, g ~= g_init restored. min_loaded_fraction now
#     caller-supplied (config; 0.5 with policy). Audit reports
#     excluded_by_policy patterns + excluded tensor names.
# Changelog (v0.1):
#   * RefinerConditioner (3-channel v2 CNN head, mirrors Conditioner v2, no
#     y-residual bypass -- input already carries x0/Aᵀr0); CplRegRefiner
#     (flavor stack + gate); load_warm_start() audit loader.
# Update summary:
#   v0.2 makes the partial warm-start scale-safe: reuse what transfers
#   (conditioning + feature layers), never the wrong-scale output heads.
from __future__ import annotations
import fnmatch
import math
import logging

import torch
import torch.nn as nn

from ..flows.realnvp_layer import RealNVPCoupling
from ..flows.nice_layer import NICECoupling, DiagScale
from .gated_update import GatedUpdate

logger = logging.getLogger("seqref_mri.refiners.coupling_regressor")
__version__ = "0.5"
__abbr__ = "SEQREF-CPLREG"

DEFAULT_EXCLUDE = ("layers.*.post.*", "layers.*.scale.log_s", "gate.*")

_FLAVORS = ("realnvp", "nice")


class RefinerConditioner(nn.Module):
    # LOCKED 3.8 stack [|x0|, Re(Aᴴr), Im(Aᴴr)] (B,in_channels,H,W) -> h
    # (B, h_dim); assembled by refiners.channel_assembly (provenance-gated).
    # Mirrors Conditioner v2 head; no y-residual bypass. Tensor names match
    # Conditioner ("net.N.*") so shape-compatible base-expert conditioner
    # weights (net.2 onward) can warm-start; net.0 fresh-inits by design.
    # v0.4: in_channels REQUIRED (no default).
    def __init__(self, *, in_channels: int, width: int = 128,
                 h_dim: int = 256):
        super().__init__()
        if not isinstance(in_channels, int) or in_channels < 1:
            logger.error("[RefinerConditioner] in_channels must be a positive "
                         "int (REQUIRED), got %r", in_channels)
            raise ValueError(
                f"in_channels must be a positive int, got {in_channels!r}")
        if width not in (64, 128):
            logger.error("[RefinerConditioner] width must be 64/128, got %s", width)
            raise ValueError(f"width must be 64 or 128, got {width}")
        w = width
        self.in_channels = in_channels
        self.h_dim = h_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, w // 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(w // 2, w, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=1),      nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(w, w, 3, padding=1),      nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(w * 4 * 4, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if inp.dim() != 4 or inp.size(1) != self.in_channels:
            logger.error("[RefinerConditioner] expected (B,%d,H,W), got %s",
                         self.in_channels, tuple(inp.shape))
            raise ValueError(
                f"expected (B,{self.in_channels},H,W), got {tuple(inp.shape)}")
        h = self.net(inp)
        if not torch.isfinite(h).all():
            logger.error("[RefinerConditioner] non-finite h")
            raise ValueError("non-finite h in RefinerConditioner")
        return h


class _DiagScaleWrap(nn.Module):
    # (x, h) signature wrapper around DiagScale (matches base_experts pattern,
    # tensor name "scale.log_s" preserved for warm-start).
    def __init__(self, dim: int):
        super().__init__()
        self.scale = DiagScale(dim)

    def forward(self, x, h):
        y, _ = self.scale(x)
        return y


class SpatialGate(nn.Module):
    # Per-pixel soft gate from the raw input stream (B,3,H,W) -> (B,1,H,W).
    # Convolutions preserve whatever spatial size is supplied.
    def __init__(self, *, in_channels: int, g_max: float, g_init: float,
                 width: int = 16):
        super().__init__()
        if not isinstance(in_channels, int) or in_channels < 1:
            logger.error("[SpatialGate] in_channels must be a positive int "
                         "(REQUIRED), got %r", in_channels)
            raise ValueError(
                f"in_channels must be a positive int, got {in_channels!r}")
        if not (0.0 < g_init < g_max):
            logger.error("[SpatialGate] need 0 < g_init < g_max, got "
                         "g_init=%r g_max=%r", g_init, g_max)
            raise ValueError("need 0 < g_init < g_max")
        self.g_max = float(g_max)
        self.conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        with torch.no_grad():
            p = g_init / g_max
            self.conv2.bias.fill_(math.log(p / (1.0 - p)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        g = self.g_max * torch.sigmoid(self.conv2(torch.relu(self.conv1(inp))))
        if not torch.isfinite(g).all():
            logger.error("[SpatialGate] non-finite g")
            raise ValueError("non-finite g")
        return g


class CplRegRefiner(nn.Module):
    # dx = couplings(x0_flat, h) - x0_flat ; x1, g = gate(x0, dx, h).
    def __init__(self, *, flavor: str, dim: int, in_channels: int,
                 h_dim: int = 256,
                 hidden: int = 256, n_layers: int | None = None,
                 cond_width: int = 128,
                 film_hidden: int = 128, film_depth: int = 2,
                 film_use_gelu: bool = True,
                 s_max: float = 4.0, post_init_std: float = 1e-3,
                 g_max: float = 0.5, g_init: float = 0.05,
                 gate_mode: str = "scalar"):
        super().__init__()
        if flavor not in _FLAVORS:
            logger.error("[CplRegRefiner] flavor must be in %s, got %r",
                         _FLAVORS, flavor)
            raise ValueError(f"flavor must be in {_FLAVORS}, got {flavor!r}")
        if not isinstance(dim, int) or dim < 1:
            logger.error("[CplRegRefiner] dim is REQUIRED from the cell "
                         "(positive int; no MNIST default), got %r", dim)
            raise ValueError(f"dim must be a positive int, got {dim!r}")
        self.flavor = flavor
        self.dim = dim
        self.in_channels = in_channels
        self.cond = RefinerConditioner(in_channels=in_channels,
                                       width=cond_width, h_dim=h_dim)
        self.layers = nn.ModuleList()
        if flavor == "realnvp":
            nl = 6 if n_layers is None else int(n_layers)
            for i in range(nl):
                self.layers.append(RealNVPCoupling(
                    dim=dim, hidden=hidden, h_dim=h_dim, flip=bool(i % 2),
                    use_film=True, s_max=s_max, post_init_std=post_init_std,
                    film_hidden=film_hidden, film_depth=film_depth,
                    film_use_gelu=film_use_gelu))
        else:  # nice
            nl = 4 if n_layers is None else int(n_layers)
            for i in range(nl):
                self.layers.append(NICECoupling(
                    dim=dim, hidden=hidden, h_dim=h_dim, flip=bool(i % 2),
                    use_film=True, post_init_std=post_init_std,
                    film_hidden=film_hidden, film_depth=film_depth,
                    film_use_gelu=film_use_gelu))
            self.layers.append(_DiagScaleWrap(dim))
        if gate_mode not in ("scalar", "spatial"):
            logger.error("[CplRegRefiner] gate_mode must be scalar|spatial, "
                         "got %r", gate_mode)
            raise ValueError(f"invalid gate_mode {gate_mode!r}")
        self.gate_mode = gate_mode
        if gate_mode == "scalar":
            self.gate = GatedUpdate(h_dim, g_max=g_max, g_init=g_init)
        else:
            self.gate_spatial = SpatialGate(in_channels=in_channels,
                                            g_max=g_max, g_init=g_init)
        self.g_max = g_max

    def delta(self, x0: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x0: (B,2,H,W) two-channel complex state (EXEC 3.14) -> dx same
        # shape. Regressor pass, no ldj use. numel per sample must equal dim.
        z = x0.flatten(1)
        out = z
        for layer in self.layers:
            res = layer(out, h)
            out = res[0] if isinstance(res, tuple) else res
        dx = (out - z).view_as(x0)
        if not torch.isfinite(dx).all():
            logger.error("[CplRegRefiner] non-finite dx")
            raise ValueError("non-finite dx")
        return dx

    def forward(self, inp: torch.Tensor, x0: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # inp: (B,in_channels,H,W) = LOCKED 3.8 stack [|x0|,Re(Aᴴr),Im(Aᴴr)]
        # (normalized, locked-I7 provenance); returns (x1, dx, g).
        if inp.dim() != 4 or inp.size(1) != self.in_channels:
            logger.error("[CplRegRefiner] inp expected (B,%d,H,W), got %s",
                         self.in_channels, tuple(inp.shape))
            raise ValueError("inp channel count != in_channels")
        if inp.shape[0] != x0.shape[0] or inp.shape[-2:] != x0.shape[-2:]:
            logger.error("[CplRegRefiner] inp/x0 shape mismatch: %s vs %s",
                         tuple(inp.shape), tuple(x0.shape))
            raise ValueError("inp/x0 shape mismatch")
        if x0[0].numel() != self.dim:
            logger.error("[CplRegRefiner] x0 numel per sample %d != dim %d",
                         x0[0].numel(), self.dim)
            raise ValueError("x0 numel per sample != dim")
        h = self.cond(inp)
        dx = self.delta(x0, h)
        if self.gate_mode == "scalar":
            x1, g = self.gate(x0, dx, h)
        else:
            g = self.gate_spatial(inp)
            want = (dx.shape[0], 1, dx.shape[2], dx.shape[3])
            if tuple(g.shape) != want:
                logger.error("[CplRegRefiner] spatial gate shape %s != %s "
                             "(B,1,H,W broadcasting over state channels)",
                             tuple(g.shape), want)
                raise ValueError("spatial gate must be (B,1,H,W)")
            x1 = x0 + g * dx      # broadcasts over the 2 state channels
        return x1, dx, g


def load_warm_start(model: CplRegRefiner, ckpt_path: str, *,
                    min_loaded_fraction: float,
                    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE) -> dict:
    # Partial, audited warm-start from a base-expert checkpoint (train_base
    # format: {"model": state_dict} with keys "cond.*" / "layers.*").
    # POLICY (v0.2): tensors matching exclude_patterns are NEVER loaded even
    # if shapes match -- base output heads speak logit-space full-recon scale,
    # not pixel-space residual scale. Loads name+shape matches outside the
    # policy; fresh-init keeps everything else. Logs every tensor. Raises if
    # loaded numel / model numel < min_loaded_fraction.
    blob = torch.load(ckpt_path, map_location="cpu")
    if "model" not in blob:
        logger.error("[warm_start] checkpoint %s missing 'model' key", ckpt_path)
        raise KeyError("checkpoint missing 'model'")
    src = blob["model"]
    dst = model.state_dict()
    def _excluded(k: str) -> bool:
        return any(fnmatch.fnmatch(k, p) for p in exclude_patterns)
    loaded, skipped_shape, absent, excluded = [], [], [], []
    new_state = {}
    for k, v in dst.items():
        if _excluded(k):
            new_state[k] = v
            excluded.append(k)
        elif k in src and src[k].shape == v.shape:
            new_state[k] = src[k]
            loaded.append(k)
        else:
            new_state[k] = v
            (skipped_shape if k in src else absent).append(k)
    unused = [k for k in src if k not in dst]
    model.load_state_dict(new_state)
    total_numel = sum(v.numel() for v in dst.values())
    loaded_numel = sum(dst[k].numel() for k in loaded)
    frac = loaded_numel / total_numel
    for k in loaded:
        logger.info("[warm_start] LOADED       %s", k)
    for k in excluded:
        logger.info("[warm_start] FRESH(policy) %s", k)
    for k in skipped_shape:
        logger.info("[warm_start] FRESH(shape)  %s", k)
    for k in absent:
        logger.info("[warm_start] FRESH(new)    %s", k)
    for k in unused:
        logger.info("[warm_start] UNUSED(src)   %s", k)
    logger.info("[warm_start] loaded fraction = %.4f (%d loaded / %d excluded "
                "by policy / %d tensors)", frac, len(loaded), len(excluded),
                len(dst))
    if frac < min_loaded_fraction:
        logger.error("[warm_start] loaded fraction %.4f < %.2f -- refusing "
                     "silent mostly-fresh init", frac, min_loaded_fraction)
        raise RuntimeError(f"warm-start loaded fraction {frac:.4f} < "
                           f"{min_loaded_fraction}")
    return {"loaded_fraction": frac, "n_loaded": len(loaded),
            "n_fresh_shape": len(skipped_shape), "n_fresh_new": len(absent),
            "excluded_by_policy": list(exclude_patterns),
            "excluded_tensors": excluded,
            "skipped_tensors": skipped_shape + absent, "unused_src": unused}
