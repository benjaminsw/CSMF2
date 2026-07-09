# SEQREF-ADJCHK v0.1 -- adjoint_check
# LIFETIME: KEEP
# L1 gate: <A x, r> == <x, Aᵀ r> in float64. Precondition for refiners
# (Aᵀr is meaningless if Aᵀ wrong). No fallback/mock/pass. Fail -> log + raise.
# Changelog (v0.1):
#   * Standalone runnable gate + importable adjoint_preflight(...).
from __future__ import annotations
import argparse
import logging
import sys
import torch
from .forward_operator import A_forward, A_adjoint

logger = logging.getLogger("fashion_seqref.adjoint_check")
__version__ = "0.1"
_EPS = 1e-12


def adjoint_preflight(blur_sigma: float, scale: int,
                      image_hw: tuple[int, int], device: str = "cpu",
                      tol: float = 1e-6) -> float:
    H, W = image_hw
    if H % scale or W % scale:
        logger.error("[adjchk] hw %s not divisible by scale %d", image_hw, scale)
        raise ValueError("image_hw must be divisible by scale")
    x = torch.randn(4, 1, H, W, dtype=torch.float64, device=device)
    r = torch.randn(4, 1, H // scale, W // scale, dtype=torch.float64, device=device)
    lhs = (A_forward(x, blur_sigma, scale) * r).sum()
    rhs = (x * A_adjoint(r, blur_sigma, scale, image_hw)).sum()
    rel = float((lhs - rhs).abs() / (lhs.abs() + _EPS))
    if rel > tol:
        logger.error("[adjchk] FAIL <Ax,r>=%.6f <x,Aᵀr>=%.6f rel=%.3e tol=%.1e",
                     float(lhs), float(rhs), rel, tol)
        raise RuntimeError("adjoint dot-product test failed")
    logger.info("[adjchk] PASS blur=%.2f scale=%d hw=%s rel=%.3e",
                blur_sigma, scale, image_hw, rel)
    return rel


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--blur-sigma", type=float, default=1.0)
    p.add_argument("--scale", type=int, default=4, choices=(1, 2, 4))
    p.add_argument("--hw", type=int, default=28)
    p.add_argument("--tol", type=float, default=1e-6)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    a = _args()
    try:
        adjoint_preflight(a.blur_sigma, a.scale, (a.hw, a.hw), tol=a.tol)
    except Exception:
        logger.error("adjoint_check FAILED", exc_info=True)
        sys.exit(1)
