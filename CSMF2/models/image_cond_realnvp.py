# =============================================================================
# STEP-1_1 v0.2 -- models.image_cond_realnvp  (IMG-RNVP v0.2)
# Purpose: RealNVP expert with IMAGE CNN couplings (RealNVPImageCoupling) that
#          subclasses _BaseExpert so it shares the SAME conditioner ownership,
#          log_prob, sample, and checkpoint key layout as the flat experts:
#            * conditioner held as self.cond (built+loaded SEPARATELY by the
#              loader -- NOT stored inside the expert state_dict)
#            * couplings live in self.layers (keys layers.* like the flat path)
#          External latent is flat [B, C*H*W]; couplings act on [B,C,H,W] via a
#          reshape inside encode/decode. NO squeeze/split (that is 3e).
# CONVENTION: shape asserts on the flat<->image boundary -> raise on mismatch.
#             No fallback / mock / dummy / pass.
# Why v0.2 (subclass _BaseExpert): v0.1 stored the conditioner as self.cond_net
#   INSIDE the module, so its weights landed in the expert state_dict as
#   cond_net.* -- which the shared loader (build_from_report) does not expect
#   (it loads the conditioner separately). v0.2 matches the flat convention so
#   the loader needs only realnvp_type routing, no image conditioner path.
#   NOTE: checkpoints trained under v0.1 (couplings.*/cond_net.* keys) are NOT
#   loadable by v0.2 -- retrain the image seeds (cheap).
# Changelog (v0.1 -> v0.2):
#   * Subclass _BaseExpert: self.cond + self.layers (was self.cond_net +
#     self.couplings). encode/decode now reshape flat<->image around the
#     inherited iteration contract. log_prob/sample inherited (Gaussian base).
# Update summary:
#   v0.2 makes the image expert a first-class _BaseExpert so the frozen-expert
#   loader + RECGATE treat it exactly like the flat experts.
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.2"
__abbr__ = "STEP-1_1"

import torch

from .experts import _BaseExpert
from .conditioner import Conditioner
from .flows.realnvp_image_layer import RealNVPImageCoupling


class ImageCondRealNVP(_BaseExpert):
    """Image-coupling RealNVP expert. Inherits _BaseExpert (self.cond,
    self.layers, log_prob, sample, Gaussian base over the flat latent). Only
    encode/decode are overridden to reshape flat[B,C*H*W] <-> image[B,C,H,W]
    around the coupling stack."""
    def __init__(self, *, dim: int, h_dim: int, conditioner: Conditioner,
                 channels: int = 1, img_hw: int = 28,
                 image_n_couplings: int = 8, image_hidden: int = 64,
                 image_s_max: float = 2.0, use_film: bool = True,
                 film_hidden: int = 64, film_depth: int = 1,
                 film_use_gelu: bool = False, **_ignored):
        super().__init__(dim=dim, h_dim=h_dim, conditioner=conditioner)
        if dim != channels * img_hw * img_hw:
            logger.error("[ImageCondRealNVP] dim %d != C*H*W %d", dim,
                         channels * img_hw * img_hw)
            raise ValueError("dim must equal channels*img_hw*img_hw")
        if not use_film:
            logger.error("[ImageCondRealNVP] image path requires FiLM conditioning")
            raise ValueError("use_film must be True for image RealNVP")
        self.channels = channels
        self.img_hw = img_hw
        for i in range(image_n_couplings):
            self.layers.append(RealNVPImageCoupling(
                channels, h_dim, parity=i % 2, hidden_ch=image_hidden,
                img_hw=img_hw, s_max=image_s_max, film_hidden=film_hidden,
                film_depth=film_depth, film_use_gelu=film_use_gelu))

    def _to_img(self, x_flat: torch.Tensor) -> torch.Tensor:
        if x_flat.dim() != 2 or x_flat.size(1) != self.dim:
            logger.error("[ImageCondRealNVP] expected flat (B,%d), got %s",
                         self.dim, tuple(x_flat.shape))
            raise ValueError("flat input has wrong dim")
        return x_flat.view(-1, self.channels, self.img_hw, self.img_hw)

    def _to_flat(self, x_img: torch.Tensor) -> torch.Tensor:
        return x_img.reshape(x_img.size(0), -1)

    # encode/decode override _BaseExpert to add the flat<->image reshape;
    # the coupling iteration contract (layer(z,h)->(z,ldj); layer.inverse)
    # is identical to the flat path.
    def encode(self, x: torch.Tensor, h: torch.Tensor):
        z = self._to_img(x)
        ldj = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        for layer in self.layers:
            z, d = layer(z, h)
            ldj = ldj + d
        return self._to_flat(z), ldj

    def decode(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = self._to_img(z)
        for layer in reversed(self.layers):
            x = layer.inverse(x, h)
        return self._to_flat(x)
