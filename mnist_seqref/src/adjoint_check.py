# =============================================================================
# SEQREF-ADJCHK v0.1 -- mnist_seqref.src.adjoint_check
# Purpose: L1 correctness gate. Verify A^T is the exact adjoint of A via the
#          dot-product test  <A x, r> == <x, A^T r>  in float64. This is a
#          PRECONDITION for any refiner (the refiner input A^T r is physically
#          meaningless if A^T is wrong) -- run before training, abort on fail.
# CONVENTION: no fallback / mock / pass. Fail -> logger.error + raise.
# Changelog (NEW in v0.1):
#   * Introduced for the start-over tree. Standalone runnable check + importable
#     adjoint_preflight(...).
# Update summary:
#   v0.1 ports the verified L1 gate: A is linear, autograd A^T is exact, the
#   test historically passes to ~1e-15. Run as:
#     python -m mnist_seqref.src.adjoint_check
# =============================================================================
from __future__ import annotations
import argparse
import logging
import sys

import torch

from .forward_operator import A_forward, A_adjoint

logger = logging.getLogger("mnist_seqref.adjoint_check")
__version__ = "0.1"
__abbr__ = "SEQREF-ADJCHK"
_EPS = 1e-12


def adjoint_preflight(blur_sigma: float, scale: int,
                      image_hw: tuple[int, int], device: str = "cpu",
                      tol: float = 1e-6) -> float:
    # <A x, r> must equal <x, A^T r>. float64. Returns rel_err; raises on fail.
    H, W = image_hw
    if H % scale or W % scale:
        logger.error("[adjchk] image_hw %s not divisible by scale %d",
                     image_hw, scale)
        raise ValueError("image_hw must be divisible by scale")
    M_h, M_w = H // scale, W // scale
    x = torch.randn(4, 1, H, W, dtype=torch.float64, device=device)
    r = torch.randn(4, 1, M_h, M_w, dtype=torch.float64, device=device)
    lhs = (A_forward(x, blur_sigma, scale) * r).sum()
    rhs = (x * A_adjoint(r, blur_sigma, scale, image_hw)).sum()
    rel = float((lhs - rhs).abs() / (lhs.abs() + _EPS))
    if rel > tol:
        logger.error("[adjchk] ADJOINT TEST FAILED: <Ax,r>=%.6f <x,A^Tr>=%.6f "
                     "rel_err=%.3e tol=%.1e -- A^T is wrong", float(lhs),
                     float(rhs), rel, tol)
        raise RuntimeError("adjoint dot-product test failed")
    logger.info("[adjchk] PASS  blur=%.2f scale=%d hw=%s  rel_err=%.3e (tol %.1e)",
                blur_sigma, scale, image_hw, rel, tol)
    return rel


def _parse_args():
    p = argparse.ArgumentParser(description="L1 adjoint dot-product pre-flight")
    p.add_argument("--blur-sigma", type=float, default=1.0)
    p.add_argument("--scale", type=int, default=4, choices=(1, 2, 4))
    p.add_argument("--hw", type=int, default=28)
    p.add_argument("--tol", type=float, default=1e-6)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    a = _parse_args()
    try:
        # sanity across the cells used in the project
        for sc in (a.scale,):
            adjoint_preflight(a.blur_sigma, sc, (a.hw, a.hw), tol=a.tol)
    except Exception:
        logger.error("adjoint_check FAILED", exc_info=True)
        sys.exit(1)
