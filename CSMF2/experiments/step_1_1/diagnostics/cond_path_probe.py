# =============================================================================
# CPATH v0.1 -- experiments.step_1_1.diagnostics.cond_path_probe
# Purpose: localize WHERE conditioning is lost in a CondGlow. Logs per-layer
#          (1) film_gain, (2) coupling conv3 weight norm, and (3) the std of
#          the coupling output (s,t) across DIFFERENT y at a fixed input.
#          Together these say whether the y-signal reaches (s,t) at all, and
#          which of {film_gain, conv3} is the choke point.
# CONVENTION: no fallback / mock / pass. Bad input / non-finite -> raise.
# Reads (per Glow step):
#   film_gain        = step.coupling.film_gain  (learnable scalar)
#   conv3_w_norm     = ||step.coupling.conv3.weight||_2
#   st_*_y_std       = std over y of (s,t) from layer-0 coupling at fixed x1
# Interpretation:
#   conv3 norm healthy + st_y_std ~ 0  -> choke is film_gain (signal not
#                                         reaching conv3 input)
#   st_y_std rises with a fix          -> conditioning now reaches (s,t)
# Changelog (NEW in v0.1):
#   * Introduced for FZDY follow-up (step 2 of the Glow conditioning debug).
# Update summary:
#   v0.1 adds the upstream-of-FZDY probe so a film_gain floor / conv3 init
#   bump can be verified to actually move the conditioning signal, not just
#   shuffle NLL around.
# =============================================================================
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "CPATH"

import torch

_FIXED_X1_SEED = 7777


@torch.no_grad()
def cond_path_probe(model, y_batch: torch.Tensor, *, n_y: int = 6) -> dict:
    # model: a CondGlow exposing .cond(y), .layers (GlowStep list with
    #        .coupling.{film_gain, conv3, _st, c_in}), .image_shape.
    # y_batch: (B,1,h,w); first n_y rows used to vary the conditioning.
    if not hasattr(model, "layers") or len(model.layers) == 0:
        logger.error("[CPATH] model has no .layers")
        raise AttributeError("model must expose .layers (GlowStep list)")
    if y_batch.dim() != 4 or y_batch.size(1) != 1:
        logger.error("[CPATH] expected y (B,1,H,W), got %s", tuple(y_batch.shape))
        raise ValueError(f"expected y (B,1,H,W), got {tuple(y_batch.shape)}")
    if y_batch.size(0) < n_y or n_y < 2:
        logger.error("[CPATH] need batch>=n_y>=2, got batch=%d n_y=%d",
                     y_batch.size(0), n_y)
        raise ValueError("need batch>=n_y>=2")

    steps = model.layers
    # --- per-layer static stats ---------------------------------------------
    film_gain = []
    conv3_norm = []
    for st in steps:
        cp = getattr(st, "coupling", None)
        if cp is None or not hasattr(cp, "conv3") or not hasattr(cp, "film_gain"):
            logger.error("[CPATH] step has no coupling.conv3/film_gain "
                         "(not a CondGlow?)")
            raise AttributeError("expected GlowStep.coupling.{conv3,film_gain}")
        film_gain.append(float(cp.film_gain.detach()))
        conv3_norm.append(float(cp.conv3.weight.detach().norm()))

    # --- (s,t) sensitivity to y at fixed input (layer 0) --------------------
    cp0 = steps[0].coupling
    c_in = int(cp0.c_in)
    C, H, W = model.image_shape
    Hs, Ws = H // 2, W // 2                       # post-squeeze spatial dims
    device = y_batch.device
    dtype = y_batch.dtype
    g = torch.Generator(device=device).manual_seed(_FIXED_X1_SEED)
    x1 = torch.randn(1, c_in, Hs, Ws, generator=g, device=device, dtype=dtype)

    y_sel = y_batch[:n_y]
    s_list, t_list = [], []
    for i in range(n_y):
        h_i = model.cond(y_sel[i:i + 1])          # (1, h_dim)
        s_i, t_i = cp0._st(x1, h_i)               # each (1, c_out, Hs, Ws)
        s_list.append(s_i)
        t_list.append(t_i)
    S = torch.cat(s_list, dim=0)                  # (n_y, c_out, Hs, Ws)
    T = torch.cat(t_list, dim=0)
    if not (torch.isfinite(S).all() and torch.isfinite(T).all()):
        logger.error("[CPATH] non-finite (s,t)")
        raise ValueError("non-finite (s,t) in cond_path_probe")
    st_s_y_std = float(S.var(dim=0, unbiased=False).mean().sqrt())
    st_t_y_std = float(T.var(dim=0, unbiased=False).mean().sqrt())

    n = len(film_gain)
    out = {
        "cond_path_probe": {
            "film_gain_min": min(film_gain),
            "film_gain_mean": sum(film_gain) / n,
            "film_gain_max": max(film_gain),
            "conv3_norm_min": min(conv3_norm),
            "conv3_norm_mean": sum(conv3_norm) / n,
            "conv3_norm_max": max(conv3_norm),
            "st_s_y_std": st_s_y_std,      # how much s moves across y (layer 0)
            "st_t_y_std": st_t_y_std,      # how much t moves across y (layer 0)
            "per_layer_film_gain": film_gain,
            "per_layer_conv3_norm": conv3_norm,
        }
    }
    logger.info("[CPATH] film_gain mean=%.3f conv3_norm mean=%.3f "
                "st_s_y_std=%.3e st_t_y_std=%.3e",
                out["cond_path_probe"]["film_gain_mean"],
                out["cond_path_probe"]["conv3_norm_mean"],
                st_s_y_std, st_t_y_std)
    return out
