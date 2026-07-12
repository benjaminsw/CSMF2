# SEQREF-RNVPIMG v0.1 -- flows.realnvp_image_layer
# LIFETIME: KEEP
# R2 (RECONRESCUE): image-shaped RealNVP coupling. Checkerboard mask on
# (B,1,28,28), CNN produces s,t from [x*mask, mask] (2 input channels),
# residual FiLM conditioning on the mid feature map, zero-init post conv
# (identity at init), s bounded by s_max*tanh, EXACT ldj = sum over unmasked s.
# Flat (B,784) in/out to match _BaseExpert.encode/decode contract.
# No fallback/mock/pass. Non-finite -> logger.error + raise.
# Changelog (v0.1):
#   * Introduced per RECONRESCUE v0.3 R2 spec (checkerboard CNN s,t +
#     residual FiLM + mask channel + zero-init post + exact ldj).
from __future__ import annotations
import logging

import torch
import torch.nn as nn

from ..conditioner import FiLMHead

logger = logging.getLogger(__name__)
__version__ = "0.1"

_HW = 28


def _checkerboard(parity: int, device, dtype) -> torch.Tensor:
    # (1,1,28,28) mask; parity 0 -> (i+j) even = 1, parity 1 -> odd = 1.
    ii = torch.arange(_HW, device=device).view(-1, 1)
    jj = torch.arange(_HW, device=device).view(1, -1)
    m = ((ii + jj) % 2 == parity).to(dtype)
    return m.view(1, 1, _HW, _HW)


class RealNVPImageCoupling(nn.Module):
    # y = m*x + (1-m)*(x*exp(s)+t),  s,t = CNN([x*m, m], FiLM(h)).
    # ldj = sum((1-m)*s). inverse exact. Flat (B,784) interface.
    def __init__(self, dim: int, hidden: int, h_dim: int, *,
                 flip: bool, use_film: bool = True, s_max: float = 2.0,
                 film_hidden: int = 128, film_depth: int = 2,
                 film_use_gelu: bool = True, cnn_channels: int = 64,
                 film_gain_init: float = 0.3):
        super().__init__()
        if dim != _HW * _HW:
            logger.error("[RNVPImg] dim must be %d (28x28), got %d", _HW * _HW, dim)
            raise ValueError(f"dim must be {_HW*_HW}, got {dim}")
        if not use_film:
            logger.error("[RNVPImg] use_film=False unsupported (FiLM is the "
                         "conditioning path)")
            raise ValueError("RealNVPImageCoupling requires use_film=True")
        if s_max <= 0.0:
            logger.error("[RNVPImg] s_max must be > 0, got %s", s_max)
            raise ValueError(f"s_max must be > 0, got {s_max}")
        self.dim = dim
        self.parity = 1 if flip else 0
        self.s_max = float(s_max)
        c = cnn_channels
        self.conv1 = nn.Conv2d(2, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.post = nn.Conv2d(c, 2, 3, padding=1)          # -> (s, t) maps
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)                     # identity at init
        self.act = nn.GELU()
        # residual FiLM on the mid feature map: feat*(1+gain*g) + gain*b per-channel
        self.film = FiLMHead(h_dim, c, hidden=film_hidden, depth=film_depth,
                             use_gelu=film_use_gelu, output_form="residual")
        self.film_gain = nn.Parameter(torch.tensor(float(film_gain_init)))

    def _st(self, x_img: torch.Tensor, m: torch.Tensor, h: torch.Tensor):
        inp = torch.cat([x_img * m, m.expand_as(x_img)], dim=1)   # (B,2,28,28)
        f = self.act(self.conv1(inp))
        g_raw, b = self.film(h)                                    # (B,c) each
        f = f * (1.0 + self.film_gain * g_raw.unsqueeze(-1).unsqueeze(-1)) \
            + self.film_gain * b.unsqueeze(-1).unsqueeze(-1)
        f = self.act(self.conv2(f))
        out = self.post(f)                                         # (B,2,28,28)
        s, t = out[:, :1], out[:, 1:]
        s = self.s_max * torch.tanh(s)
        return s, t

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        if x.dim() != 2 or x.size(1) != self.dim:
            logger.error("[RNVPImg.fwd] expected (B,%d), got %s", self.dim,
                         tuple(x.shape))
            raise ValueError("flat dim mismatch")
        x_img = x.view(-1, 1, _HW, _HW)
        m = _checkerboard(self.parity, x.device, x.dtype)
        s, t = self._st(x_img, m, h)
        um = 1.0 - m
        y_img = m * x_img + um * (x_img * torch.exp(s) + t)
        ldj = (um * s).flatten(1).sum(dim=-1)
        if not torch.isfinite(ldj).all():
            logger.error("[RNVPImg.fwd] non-finite ldj")
            raise ValueError("non-finite ldj in image coupling forward")
        return y_img.flatten(1), ldj

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if y.dim() != 2 or y.size(1) != self.dim:
            logger.error("[RNVPImg.inv] expected (B,%d), got %s", self.dim,
                         tuple(y.shape))
            raise ValueError("flat dim mismatch")
        y_img = y.view(-1, 1, _HW, _HW)
        m = _checkerboard(self.parity, y.device, y.dtype)
        # masked coords pass through unchanged -> s,t computable from y directly
        s, t = self._st(y_img, m, h)
        um = 1.0 - m
        x_img = m * y_img + um * ((y_img - t) * torch.exp(-s))
        return x_img.flatten(1)
