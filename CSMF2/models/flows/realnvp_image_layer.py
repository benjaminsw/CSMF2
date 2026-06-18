# =============================================================================
# STEP-1_1 v0.1 -- models.flows.realnvp_image_layer  (IMG-RNVP v0.1)
# Purpose: Image-shaped RealNVP affine coupling. Operates on x as [B,1,28,28]
#          (NO flatten). Checkerboard binary mask m; s,t produced by a small
#          CNN that sees cat([m*x, m]) (mask as an extra channel) and is
#          FiLM-conditioned (channel-wise) on h(y). Affine on the complement:
#              y = m*x + (1-m) * ( x*exp(s) + t ),  s = s_max * tanh(raw_s)
#          ldj = sum over the transformed (1-m) pixels of s. Posterior-safe.
# CONVENTION: s bounded by tanh -> exp(s) finite; non-finite ldj/inverse ->
#             logger.error + raise. No fallback / mock / dummy / pass.
# Scope (1.4b-A): checkerboard mask ONLY; NO squeeze / split / factor-out.
#   Mask alternates parity each layer (analogous to `flip` in the flat layer).
# Key choices (folded corrections):
#   * CNN input is cat([m*x, m], dim=1) -> in_channels=2, so the net can tell
#     "0 because masked" from "0 because background" (MNIST background is 0).
#   * FiLM is conv-shaped AND residual: per-CHANNEL gamma/beta from FiLMHead
#     (output_form='residual', non-zero at init), broadcast as [B,C,1,1] and
#     applied z*(1+gamma_raw)+beta. Residual form keeps the conditioning
#     gradient alive even with the coupling's zero-init post conv (avoids the
#     two-zero-inits-in-series trap; see FiLMHead v0.4).
#   * post-conv zero-init -> identity coupling at init (s=0,t=0), matching the
#     flat layer's zero-init safety.
# Changelog (NEW in v0.1):
#   * Introduced. RealNVPImageCoupling with checkerboard mask + CNN s,t +
#     conv FiLM + exact ldj + inverse. Same call API as RealNVPCoupling
#     (forward(x,h)->(y,ldj); inverse(y,h)->x), but x is image-shaped.
# Update summary:
#   v0.1 is the medium-cost bridge from flat RealNVP to full multi-scale: real
#   image structure (CNN couplings, spatial mask) with no squeeze/split yet.
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-1_1"

import torch
import torch.nn as nn

from ..conditioner import FiLMHead


def checkerboard_mask(h: int, w: int, parity: int, device, dtype) -> torch.Tensor:
    """Binary [1,1,h,w] mask; m[i,j]=1 where (i+j)%2==parity (active/kept)."""
    yy = torch.arange(h, device=device).view(h, 1)
    xx = torch.arange(w, device=device).view(1, w)
    m = ((yy + xx) % 2 == parity).to(dtype)
    return m.view(1, 1, h, w)


class _FiLMConv(nn.Module):
    """One Conv3x3 followed by channel-wise FiLM(h) and an activation.
    Uses RESIDUAL FiLM (output_form='residual'): the head's last layer is
    small-normal-init (non-zero at step 0), so the conditioning gradient flows
    from the start even though the coupling's final `post` conv is zero-init.
    This avoids the "two zero-inits in series" vanishing-gradient trap (see
    FiLMHead v0.4). gamma_raw/beta come as vectors (width=out_ch), reshaped to
    [B,out_ch,1,1] and applied residually: z' = z*(1 + gamma_raw) + beta."""
    def __init__(self, in_ch: int, out_ch: int, h_dim: int, *,
                 film_hidden: int, film_depth: int, film_use_gelu: bool,
                 act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.film = FiLMHead(h_dim, out_ch, hidden=film_hidden,
                             depth=film_depth, use_gelu=film_use_gelu,
                             output_form="residual")
        self.act = nn.GELU() if (act and film_use_gelu) else (
            nn.ReLU() if act else None)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        gamma_raw, beta = self.film(h)             # residual form: raw values
        gamma_raw = gamma_raw.unsqueeze(-1).unsqueeze(-1)  # [B, out_ch, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        z = z * (1.0 + gamma_raw) + beta
        if self.act is not None:
            z = self.act(z)
        return z


class RealNVPImageCoupling(nn.Module):
    """Image affine coupling. x: [B, C, H, W] (C=1 for MNIST). Half the pixels
    (checkerboard) are kept; the complement is affine-transformed by s,t from a
    CNN that sees the kept pixels + the mask channel, conditioned on h(y)."""
    def __init__(self, channels: int, h_dim: int, *, parity: int,
                 hidden_ch: int = 64, img_hw: int = 28, s_max: float = 2.0,
                 film_hidden: int = 64, film_depth: int = 1,
                 film_use_gelu: bool = False):
        super().__init__()
        self.channels = channels
        self.parity = parity
        self.s_max = s_max
        self.img_hw = img_hw
        # CNN: in = channels (m*x) + channels (mask) ; out = 2*channels (s,t)
        in_ch = 2 * channels
        self.block1 = _FiLMConv(in_ch, hidden_ch, h_dim, film_hidden=film_hidden,
                                film_depth=film_depth, film_use_gelu=film_use_gelu)
        self.block2 = _FiLMConv(hidden_ch, hidden_ch, h_dim,
                                film_hidden=film_hidden, film_depth=film_depth,
                                film_use_gelu=film_use_gelu)
        self.post = nn.Conv2d(hidden_ch, 2 * channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.post.weight)           # identity coupling at init
        nn.init.zeros_(self.post.bias)

    def _st(self, x_kept: torch.Tensor, m: torch.Tensor, h: torch.Tensor):
        # mask fed as an extra channel so the CNN distinguishes masked-0 vs bg-0
        mask_ch = m.expand(x_kept.size(0), self.channels, -1, -1)
        inp = torch.cat([x_kept, mask_ch], dim=1)  # [B, 2C, H, W]
        z = self.block1(inp, h)
        z = self.block2(z, h)
        out = self.post(z)                         # [B, 2C, H, W]
        raw_s, t = out.chunk(2, dim=1)
        s = self.s_max * torch.tanh(raw_s)
        return s, t

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        m = checkerboard_mask(x.size(-2), x.size(-1), self.parity,
                              x.device, x.dtype)
        x_kept = m * x
        s, t = self._st(x_kept, m, h)
        # transform only the complement (1-m); kept pixels pass through
        comp = 1.0 - m
        y = x_kept + comp * (x * torch.exp(s) + t)
        ldj = (comp * s).flatten(1).sum(dim=-1)
        if not torch.isfinite(ldj).all():
            logger.error("[RealNVPImageCoupling.fwd] non-finite ldj")
            raise ValueError("non-finite ldj in image coupling forward")
        return y, ldj

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        m = checkerboard_mask(y.size(-2), y.size(-1), self.parity,
                              y.device, y.dtype)
        y_kept = m * y                             # kept pixels identical fwd/inv
        s, t = self._st(y_kept, m, h)
        comp = 1.0 - m
        x = y_kept + comp * (y - t) * torch.exp(-s)
        if not torch.isfinite(x).all():
            logger.error("[RealNVPImageCoupling.inv] non-finite inverse")
            raise ValueError("non-finite inverse in image coupling")
        return x
