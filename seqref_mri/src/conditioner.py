# =============================================================================
# STEP-1_1 v0.5 -- models.conditioner
# Purpose: shared conditioner c_eta(y) + FiLM head used by every expert.
# CONVENTION: NaN/shape errors -> logger.error + raise. No silent fallback.
# Changelog (v0.4 -> v0.5):
#   * NEW (Glow rescue): optional y-residual bypass in Conditioner.
#     When y_residual_alpha_init > 0:
#         h = cnn_head(y) + alpha * Linear(y.flatten(1))
#     alpha is a learnable nn.Parameter. The linear bypass cannot collapse
#     to a constant because it's a direct linear function of y -- only by
#     driving alpha to zero, which itself receives gradient.
#   * Default y_residual_alpha_init=0.0 -> bypass disabled -> v0.4 byte-identical.
#   * Requires y_input_size kwarg when enabled (size of flattened y, e.g.
#     14*14 for scale=2). Raises if missing.
# Changelog (v0.3 -> v0.4):
#   * FiLMHead last-Linear init depends on output_form.
# Changelog (v0.2 -> v0.3):
#   * FiLMHead gains output_form kwarg.
# Changelog (v0.1 -> v0.2):
#   * Conditioner gains use_v2 kwarg.
# Changelog (NEW in v0.1):
#   * Introduced.
# Update summary:
#   v0.5 provides an architectural escape from the conditioner-collapse
#   attractor: a learnable, linear bypass from y direct to h. The CNN can
#   still be optimized into constant output, but the bypass cannot (it is
#   linear in y by construction). Default OFF -- only activated when
#   y_residual_alpha_init > 0 via cfg.
# =============================================================================
from __future__ import annotations
import logging
import traceback
logger = logging.getLogger(__name__)
__version__ = "0.5"
__abbr__ = "STEP-1_1"

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conditioner(nn.Module):
    # y (B,1,h,w) -> h (B, h_dim). Same architecture for x2 and x4 via adaptive pool.
    # use_v2 (default False) preserves v0.1 head:  Linear(w*16, h_dim).
    # use_v2=True activates v2 head:               Linear(w*16, h_dim) -> GELU -> Linear(h_dim, h_dim).
    # v0.5: optional y-residual bypass. When y_residual_alpha_init > 0:
    #     h = cnn_head(y) + alpha * Linear(y.flatten(1))
    #   provides a DIRECT linear function of y that the CNN cannot
    #   "collapse" via training. Used to rescue Glow's conditioner from
    #   the constant-output attractor (audited v0.11-v0.12).
    def __init__(self, *, width: int = 64, h_dim: int = 128,
                 use_v2: bool = False,
                 y_residual_alpha_init: float = 0.0,
                 y_input_size: int | None = None):
        super().__init__()
        if width not in (64, 128):
            logger.error("[Conditioner] width must be 64 or 128, got %s", width)
            raise ValueError(f"width must be 64 or 128, got {width}")
        if h_dim < 1:
            logger.error("[Conditioner] h_dim must be positive, got %s", h_dim)
            raise ValueError(f"h_dim must be positive, got {h_dim}")
        if y_residual_alpha_init < 0.0:
            logger.error("[Conditioner] y_residual_alpha_init must be >=0, got %s",
                         y_residual_alpha_init)
            raise ValueError(
                f"y_residual_alpha_init must be >=0, got {y_residual_alpha_init}")
        if y_residual_alpha_init > 0.0 and (y_input_size is None
                                            or y_input_size < 1):
            logger.error("[Conditioner] y_residual_alpha_init>0 requires "
                         "y_input_size (positive int), got %s", y_input_size)
            raise ValueError(
                "y_residual_alpha_init>0 requires positive y_input_size")
        self.width = width
        self.h_dim = h_dim
        self.use_v2 = bool(use_v2)
        self.y_residual_enabled = (y_residual_alpha_init > 0.0)
        w = width
        if not self.use_v2:
            # v0.1 head -- byte-identical to legacy
            self.net = nn.Sequential(
                nn.Conv2d(1, w // 2, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(w // 2, w, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(w, w, 3, padding=1),      nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4),
                nn.Conv2d(w, w, 3, padding=1),      nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(w * 4 * 4, h_dim),
            )
        else:
            # v0.2 head -- Linear -> GELU -> Linear
            self.net = nn.Sequential(
                nn.Conv2d(1, w // 2, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(w // 2, w, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(w, w, 3, padding=1),      nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4),
                nn.Conv2d(w, w, 3, padding=1),      nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(w * 4 * 4, h_dim),
                nn.GELU(),
                nn.Linear(h_dim, h_dim),
            )
        # v0.5: y-residual bypass.
        if self.y_residual_enabled:
            self.y_residual_proj = nn.Linear(y_input_size, h_dim)
            # default Kaiming-uniform init is fine for the projection.
            # alpha is a LEARNABLE scalar; bounded below 0 only via grad.
            self.y_residual_alpha = nn.Parameter(
                torch.tensor(float(y_residual_alpha_init)))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 4 or y.size(1) != 1:
            logger.error("[Conditioner] expected (B,1,H,W), got %s", tuple(y.shape))
            raise ValueError(f"expected (B,1,H,W), got {tuple(y.shape)}")
        h = self.net(y)
        if self.y_residual_enabled:
            y_flat = y.flatten(1)            # (B, H*W)
            h = h + self.y_residual_alpha * self.y_residual_proj(y_flat)
        if not torch.isfinite(h).all():
            logger.error("[Conditioner] non-finite h (any NaN=%s, Inf=%s)",
                         bool(torch.isnan(h).any()), bool(torch.isinf(h).any()))
            raise ValueError("non-finite h in Conditioner")
        return h


class FiLMHead(nn.Module):
    # h (B, h_dim) -> (gamma, beta) each (B, feat_width).
    # output_form='affine'   (default; legacy)
    #     gamma = 1 + tanh(raw_gamma)   -- intended for z' = gamma*z + beta.
    #     Initialised so gamma ~= 1, beta ~= 0 (identity at init).
    # output_form='residual' (Glow)
    #     gamma is returned RAW (no 1+tanh).  Caller does
    #         z' = z * (1 + gain * gamma_raw) + gain * beta
    #     This makes the conditioning contribution first-order in the FiLM
    #     weights even when downstream layers are zero-init; the caller owns
    #     the 'gain' parameter to scale the effect.
    def __init__(self, h_dim: int, feat_width: int, hidden: int = 64,
                 *, depth: int = 1, use_gelu: bool = False,
                 output_form: str = "affine"):
        super().__init__()
        if depth < 1:
            logger.error("[FiLMHead] depth must be >=1, got %s", depth)
            raise ValueError(f"depth must be >=1, got {depth}")
        if output_form not in ("affine", "residual"):
            logger.error("[FiLMHead] output_form must be 'affine' or "
                         "'residual', got %r", output_form)
            raise ValueError(
                f"output_form must be 'affine' or 'residual', got {output_form!r}")
        self.output_form = output_form
        layers: list[nn.Module] = []
        in_dim = h_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            if use_gelu:
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU(inplace=True))
            in_dim = hidden
        last = nn.Linear(in_dim, 2 * feat_width)
        # v0.4: init depends on output_form.
        #   'affine'   -> zero-init so FiLM is identity (gamma=1, beta=0). The
        #                 downstream coupling NN provides the necessary
        #                 non-identity behaviour as it trains.
        #   'residual' -> small-normal init so (gamma_raw, beta) are non-zero
        #                 at step 0. Without this, the chain
        #                   cond -> film(h) -> coupling -> NLL
        #                 has two zero-inits in series (film.last AND
        #                 coupling.conv3), so the gradient w.r.t. either
        #                 vanishes -- a second-order vanishing problem the
        #                 residual form was supposed to break, but couldn't,
        #                 because film.last itself was zero-init.
        if output_form == "affine":
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        else:  # "residual"
            nn.init.normal_(last.weight, mean=0.0, std=0.01)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.mlp = nn.Sequential(*layers)
        self.feat_width = feat_width

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(h)
        gamma_raw, beta = out.chunk(2, dim=-1)
        if self.output_form == "affine":
            gamma = 1.0 + torch.tanh(gamma_raw)          # range (0, 2), centred at 1
            return gamma, beta
        # residual form: return raw values; caller handles (1 + gain * gamma_raw)
        return gamma_raw, beta


class ConcatInjector(nn.Module):
    # Alternative to FiLM for the ablation: project h and concat to features.
    # Used when cfg.use_film is False.
    def __init__(self, h_dim: int, feat_width: int):
        super().__init__()
        self.proj = nn.Linear(h_dim, feat_width)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # feat: (B, W), h: (B, h_dim) -> (B, W) via residual add of projected h.
        if feat.shape[0] != h.shape[0]:
            logger.error("[ConcatInjector] batch mismatch feat=%s h=%s",
                         feat.shape, h.shape)
            raise ValueError("batch dim mismatch")
        return feat + self.proj(h)
