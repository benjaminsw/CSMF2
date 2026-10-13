# =============================================================================
# SEQREF-MRI-V04M v0.1 -- scripts.v04_model
# LIFETIME: KEEP
# Purpose: V04M wiring (V04PLAN v1.0 article 178 §4 step 1) — the conditional
#          RealNVP free-coordinate model. Reuses the registered RealNVPCoupling
#          (src.flows.realnvp_layer) via CondRealNVP with the frozen V04 pin:
#          6 affine coupling layers, hidden 256, alternating half-split masks
#          (flip=layer parity), FiLM conditioning, s_max=2.0 tanh soft clamp,
#          post_init_std=0.0 (exact identity at init). Conditioner, mask branch
#          and free-coordinate contract are the V03-frozen ones (imported from
#          src.free_flow_runtime — never redefined here).
# CONVENTION: no fallback / mock / placeholder / silent pass. Every failure
#             path -> logger.error + raise V04MError with a typed code.
# =============================================================================
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from seqref_mri.src.base_experts import CondRealNVP, _gaussian_logprob
from seqref_mri.src.conditioner import Conditioner
from seqref_mri.src.free_flow_runtime import (
    COND_IN_CHANNELS,
    COND_USE_V2,
    COND_WIDTH,
    FLOW_DIM_REAL,
    H_DIM,
    MASK_BITS,
    MASK_EMBED_DIM,
    MODEL_INIT_SEED,
    Y_RESIDUAL_ALPHA_INIT,
    MaskBranch,
)
from seqref_mri.src.flows.realnvp_layer import RealNVPCoupling

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "SEQREF-MRI-V04M"

# -- V04M pins (178 §4 / SPEC §24 Table B) ------------------------------------
V04M_N_LAYERS = 6
V04M_HIDDEN = 256
V04M_S_MAX = 2.0                    # tanh soft clamp (repo mechanism)
V04M_POST_INIT_STD = 0.0            # exact identity at init
V04M_FILM_HIDDEN = 64
V04M_FILM_DEPTH = 1
V04M_FILM_USE_GELU = False
V04M_D_IN = FLOW_DIM_REAL // 2      # contiguous half-split: 6912


class V04MError(RuntimeError):
    """Typed V04M construction/runtime failure (code, message)."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[SEQREF-MRI-V04M] %s: %s", code, message)
    raise V04MError(code, message)


class V04MFlowModel(nn.Module):
    """Conditional RealNVP over the V03-frozen free-coordinate state.

    flow:        CondRealNVP(dim=FLOW_DIM_REAL, h_dim=H_DIM, hidden=256,
                 n_layers=6, use_film=True, s_max=2.0, post_init_std=0.0,
                 film_hidden=64, film_depth=1, film_use_gelu=False) over a
                 Conditioner(in_channels=2, width=64, h_dim=128, use_v2=False,
                 y_residual_alpha_init=0.0) — identical conditioner contract
                 to the V03 NSF model.
    mask_branch: MaskBranch (added to the conditioner output), V03-frozen.
    """

    def __init__(self) -> None:
        super().__init__()
        cond = Conditioner(in_channels=COND_IN_CHANNELS, width=COND_WIDTH,
                           h_dim=H_DIM, use_v2=COND_USE_V2,
                           y_residual_alpha_init=Y_RESIDUAL_ALPHA_INIT)
        self.flow = CondRealNVP(
            dim=FLOW_DIM_REAL, h_dim=H_DIM, conditioner=cond,
            hidden=V04M_HIDDEN, n_layers=V04M_N_LAYERS, use_film=True,
            s_max=V04M_S_MAX, post_init_std=V04M_POST_INIT_STD,
            film_hidden=V04M_FILM_HIDDEN, film_depth=V04M_FILM_DEPTH,
            film_use_gelu=V04M_FILM_USE_GELU)
        self.mask_branch = MaskBranch()
        self._verify_construction_fixity()

    # -- construction-fixity gates (frozen architecture; drift = ERROR) --
    def _verify_construction_fixity(self) -> None:
        if self.flow.dim != FLOW_DIM_REAL:
            _fail("CONSTRUCTION_DIM_MISMATCH",
                  f"flow.dim={self.flow.dim} != {FLOW_DIM_REAL}")
        if len(self.flow.layers) != V04M_N_LAYERS:
            _fail("CONSTRUCTION_LAYERS_MISMATCH",
                  f"{len(self.flow.layers)} coupling layers != "
                  f"{V04M_N_LAYERS}")
        for i, layer in enumerate(self.flow.layers):
            if not isinstance(layer, RealNVPCoupling):
                _fail("CONSTRUCTION_LAYER_TYPE",
                      f"layer {i} is {type(layer).__name__}, expected "
                      f"RealNVPCoupling")
            if layer.flip != bool(i % 2):
                _fail("CONSTRUCTION_FLIP_PARITY",
                      f"layer {i} flip={layer.flip}, expected {bool(i % 2)}")
            if layer.d_in != V04M_D_IN or layer.d_out != FLOW_DIM_REAL - V04M_D_IN:
                _fail("CONSTRUCTION_SPLIT_MISMATCH",
                      f"layer {i} split d_in={layer.d_in} d_out={layer.d_out}, "
                      f"expected {V04M_D_IN}/{FLOW_DIM_REAL - V04M_D_IN}")
            if float(layer.s_max) != V04M_S_MAX:
                _fail("CONSTRUCTION_S_MAX_MISMATCH",
                      f"layer {i} s_max={float(layer.s_max)!r} != {V04M_S_MAX}")
            if layer.pre.in_features != layer.d_in \
                    or layer.pre.out_features != V04M_HIDDEN \
                    or layer.mid.in_features != V04M_HIDDEN \
                    or layer.mid.out_features != V04M_HIDDEN \
                    or layer.post.in_features != V04M_HIDDEN \
                    or layer.post.out_features != 2 * layer.d_out:
                _fail("CONSTRUCTION_HIDDEN_MISMATCH",
                      f"layer {i} param-net widths diverge from the frozen "
                      f"hidden={V04M_HIDDEN}, post=2*d_out")
            if not bool((layer.post.weight == 0.0).all()) \
                    or not bool((layer.post.bias == 0.0).all()):
                _fail("CONSTRUCTION_INIT_MISMATCH",
                      f"layer {i} post is not exactly zero-init "
                      f"(post_init_std={V04M_POST_INIT_STD} pins identity)")
            film = getattr(layer, "film", None)
            if film is None:
                _fail("CONSTRUCTION_FILM_MISSING",
                      f"layer {i} lacks a FiLM head (use_film is frozen True)")
            mods = list(film.mlp)
            if film.output_form != "affine" or len(mods) != 3 \
                    or not isinstance(mods[0], nn.Linear) \
                    or mods[0].in_features != H_DIM \
                    or mods[0].out_features != V04M_FILM_HIDDEN \
                    or not isinstance(mods[1], nn.ReLU) \
                    or isinstance(mods[1], nn.GELU) \
                    or not isinstance(mods[2], nn.Linear) \
                    or mods[2].out_features != 2 * V04M_HIDDEN:
                _fail("CONSTRUCTION_FILM_MISMATCH",
                      f"layer {i} FiLM architecture diverges from the frozen "
                      f"hidden={V04M_FILM_HIDDEN}, depth={V04M_FILM_DEPTH}, "
                      f"use_gelu=False, output_form='affine'")
        if self.flow.cond.in_channels != COND_IN_CHANNELS \
                or self.flow.cond.width != COND_WIDTH \
                or self.flow.cond.h_dim != H_DIM \
                or self.flow.cond.use_v2 != COND_USE_V2 \
                or self.flow.cond.y_residual_enabled:
            _fail("CONSTRUCTION_CONDITIONER_MISMATCH",
                  "the conditioner diverges from the frozen in_channels=2, "
                  "width=64, h_dim=128, use_v2=False, y-residual DISABLED")
        mb = self.mask_branch.proj
        if mb.in_features != MASK_BITS or mb.out_features != MASK_EMBED_DIM:
            _fail("CONSTRUCTION_MASK_BRANCH_MISMATCH",
                  f"mask branch {mb.in_features}->{mb.out_features} != "
                  f"{MASK_BITS}->{MASK_EMBED_DIM}")
        if not bool((mb.bias == 0.0).all()):
            _fail("CONSTRUCTION_MASK_BIAS",
                  "mask-branch bias must be exactly 0 at init")
        if not torch.isfinite(mb.weight).all():
            _fail("CONSTRUCTION_MASK_WEIGHT_NON_FINITE",
                  "mask-branch weight init produced a non-finite value")

    # -- conditioning ---------------------------------------------------
    def condition(self, cond_in: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        """h = c_eta(cond_in) + mask_branch(mask). Identical to V03."""
        h = self.flow.cond(cond_in) + self.mask_branch(mask)
        if not torch.isfinite(h).all():
            _fail("CONDITIONING_NON_FINITE",
                  "the combined conditioning vector is non-finite")
        return h

    # -- probability / flow directions ----------------------------------
    def encode_free(self, u_scaled: torch.Tensor, cond_in: torch.Tensor,
                    mask: torch.Tensor):
        """Free scalars -> latent. Returns (z, ldj), shapes (B,13824), (B,)."""
        if u_scaled.dim() != 2 or int(u_scaled.shape[1]) != FLOW_DIM_REAL:
            _fail("STATE_LAYOUT_UNEXPECTED",
                  f"u_scaled must be (B, {FLOW_DIM_REAL}), got "
                  f"{tuple(u_scaled.shape)}")
        h = self.condition(cond_in, mask)
        return self.flow.encode(u_scaled.to(torch.float32), h)

    def log_prob_free(self, u_scaled: torch.Tensor,
                      cond_in: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        """log p(u_scaled | cond_in, mask) under N(0, I), shape (B,)."""
        z, ldj = self.encode_free(u_scaled, cond_in, mask)
        lp = _gaussian_logprob(z) + ldj
        if not torch.isfinite(lp).all():
            _fail("LOG_PROB_NON_FINITE",
                  "non-finite log_prob on the free-coordinate state")
        return lp

    def decode_scalars(self, z: torch.Tensor, cond_in: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
        """z -> RealNVP inverse -> standardised free scalars (B, 13824) f32."""
        if z.dim() != 2 or int(z.shape[1]) != FLOW_DIM_REAL:
            _fail("STATE_LAYOUT_UNEXPECTED",
                  f"z must be (B, {FLOW_DIM_REAL}), got {tuple(z.shape)}")
        h = self.condition(cond_in, mask)
        u = self.flow.decode(z.to(torch.float32), h)
        if not torch.isfinite(u).all():
            _fail("DECODE_NON_FINITE",
                  "the RealNVP inverse produced a non-finite scalar state")
        return u

    # -- test-registration parameter group (NOT an optimizer group) -----
    def realnvp_transform_parameters(self) -> dict:
        """Affine parameter nets of every coupling (pre/mid/post)."""
        out = {}
        for name, p in self.named_parameters():
            if name.startswith("flow.layers.") and \
                    any(f".{w}." in name for w in ("pre", "mid", "post")):
                out[name] = p
        if not out:
            _fail("GROUP_EMPTY", "RealNVP-transform parameter group is empty")
        return out


def build_v04_model(*, init_seed: int = MODEL_INIT_SEED) -> V04MFlowModel:
    """Deterministically-seeded construction (V04 registered fresh init,
    seed 20260813 = MODEL_INIT_SEED). The seed governs module init ONLY."""
    torch.manual_seed(int(init_seed))
    model = V04MFlowModel()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("[SEQREF-MRI-V04M] model built: dim=%d layers=%d hidden=%d "
                "s_max=%.17g post_init_std=%.17g params=%d init_seed=%d",
                FLOW_DIM_REAL, V04M_N_LAYERS, V04M_HIDDEN, V04M_S_MAX,
                V04M_POST_INIT_STD, n_params, int(init_seed))
    return model
