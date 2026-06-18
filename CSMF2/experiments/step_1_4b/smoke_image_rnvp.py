# =============================================================================
# STEP-1_4B v0.1 -- experiments.step_1_4b.smoke_image_rnvp  (IMG-RNVP v0.1)
# Purpose: FAIL-FAST invertibility/correctness gate for the image RealNVP
#          coupling + expert, run BEFORE committing GPU time. Checks:
#            1. encode->decode cycle  < 1e-5 (FP64)
#            2. ldj finite
#            3. analytic ldj == autograd logdet (single small layer)
#            4. conditioning active (different y changes s,t)  [needs trained-ish FiLM]
#            5. checkerboard mask: kept pixels fixed, complement transformed; parity alternates
#            6. expert flat API: encode(x_flat)->(w_flat,ldj), decode round-trips, log_prob finite
# CONVENTION: any failed check -> logger.error + raise (exit 1). No silent pass.
# Usage: python -m CSMF2.experiments.step_1_4b.smoke_image_rnvp
# Changelog (NEW in v0.1):
#   * Introduced. Gates Stage 1.4b-A training on coupling correctness.
# Update summary:
#   v0.1 must pass before run_image_rnvp.sh. FP64 throughout for tight bounds.
# =============================================================================
from __future__ import annotations
import logging
import sys
import traceback

import torch

from ...models.flows.realnvp_image_layer import (RealNVPImageCoupling,
                                                 checkerboard_mask)
from ...models.image_cond_realnvp import ImageCondRealNVP
from ...models.conditioner import Conditioner

logger = logging.getLogger("CSMF2.step_1_4b.smoke")
__version__ = "0.1"

CYCLE_TOL = 1e-5
LDJ_TOL = 1e-6


def _perturb(coupling):
    # Make the coupling non-identity WITHOUT assuming FiLMHead internals:
    # nudge every parameter (conv blocks, FiLM heads, post conv) by small noise.
    with torch.no_grad():
        for p in coupling.parameters():
            p.add_(torch.randn_like(p) * 0.05)


def run() -> None:
    torch.manual_seed(0)
    B, C, H, h_dim = 4, 1, 28, 16
    dim = C * H * H
    dt = torch.float64

    # stack of couplings
    blks = [RealNVPImageCoupling(C, h_dim, parity=i % 2, hidden_ch=16, img_hw=H,
                                 s_max=2.0, film_hidden=16, film_depth=1).double()
            for i in range(8)]
    for b in blks:
        _perturb(b)
    x = torch.randn(B, C, H, H, dtype=dt)
    hh = torch.randn(B, h_dim, dtype=dt)

    z = x
    tot = torch.zeros(B, dtype=dt)
    for b in blks:
        z, ldj = b(z, hh)
        tot = tot + ldj
    xr = z
    for b in reversed(blks):
        xr = b.inverse(xr, hh)
    cyc = (x - xr).abs().max().item()
    if not (cyc < CYCLE_TOL):
        logger.error("[smoke] cycle_error %.2e >= %.0e", cyc, CYCLE_TOL)
        raise RuntimeError("image coupling cycle error too large")
    if not torch.isfinite(tot).all():
        logger.error("[smoke] non-finite ldj")
        raise RuntimeError("non-finite ldj")
    logger.info("[smoke] (1) cycle=%.2e OK  (2) ldj finite OK", cyc)

    # analytic ldj vs autograd logdet (small 6x6)
    b0 = RealNVPImageCoupling(C, h_dim, parity=0, hidden_ch=8, img_hw=6,
                              s_max=2.0, film_hidden=8, film_depth=1).double()
    _perturb(b0)
    xs = torch.randn(1, C, 6, 6, dtype=dt)
    hs = torch.randn(1, h_dim, dtype=dt)
    J = torch.autograd.functional.jacobian(
        lambda v: b0(v.view(1, C, 6, 6), hs)[0].reshape(-1), xs.reshape(-1))
    _, logabsdet = torch.linalg.slogdet(J)
    _, ldj0 = b0(xs, hs)
    diff = abs(ldj0.item() - logabsdet.item())
    if not (diff < LDJ_TOL):
        logger.error("[smoke] analytic ldj %.6f vs autograd %.6f diff %.2e",
                     ldj0.item(), logabsdet.item(), diff)
        raise RuntimeError("analytic ldj != autograd logdet")
    logger.info("[smoke] (3) analytic==autograd logdet OK (diff=%.2e)", diff)

    # conditioning active
    h2 = torch.randn(B, h_dim, dtype=dt)
    y1, _ = blks[0](x, hh)
    y2, _ = blks[0](x, h2)
    d = (y1 - y2).abs().max().item()
    if not (d > 1e-6):
        logger.error("[smoke] conditioning inactive: |d|=%.2e", d)
        raise RuntimeError("conditioning inactive (s,t do not change with y)")
    logger.info("[smoke] (4) conditioning active OK (|d|=%.3e)", d)

    # mask behaviour
    m0 = checkerboard_mask(H, H, 0, x.device, x.dtype)
    kept = (m0 * (y1 - x)).abs().max().item()
    comp = ((1 - m0) * (y1 - x)).abs().max().item()
    if not (kept < 1e-9 and comp > 1e-6):
        logger.error("[smoke] mask wrong: kept=%.2e comp=%.2e", kept, comp)
        raise RuntimeError("checkerboard mask not behaving")
    if blks[0].parity == blks[1].parity:
        logger.error("[smoke] parity does not alternate")
        raise RuntimeError("mask parity must alternate")
    logger.info("[smoke] (5) mask kept-fixed/comp-moved + parity alternates OK")

    # expert flat API
    y_hw = 28 // 2                       # scale 2 -> degraded y is 14x14
    cond = Conditioner(width=64, h_dim=h_dim, use_v2=True).double()
    exp = ImageCondRealNVP(dim=dim, h_dim=h_dim, conditioner=cond,
                           channels=C, img_hw=H,
                           image_n_couplings=8, image_hidden=16, image_s_max=2.0,
                           use_film=True, film_hidden=16, film_depth=1).double()
    for c in exp.layers:
        _perturb(c)
    xf = torch.randn(B, dim, dtype=dt)
    yy = torch.randn(B, 1, y_hw, y_hw, dtype=dt)   # (B,1,H,W) as Conditioner expects
    he = exp.cond(yy)
    w, ldj = exp.encode(xf, he)
    xfr = exp.decode(w, he)
    ecyc = (xf - xfr).abs().max().item()
    lp = exp.log_prob(xf, yy)
    if not (w.shape == (B, dim) and ecyc < CYCLE_TOL
            and torch.isfinite(lp).all() and exp.dim == dim):
        logger.error("[smoke] expert API fail: wshape=%s ecyc=%.2e lp_finite=%s dim=%d",
                     tuple(w.shape), ecyc, torch.isfinite(lp).all().item(), exp.dim)
        raise RuntimeError("expert flat API check failed")
    logger.info("[smoke] (6) expert flat API OK (cycle=%.2e, w=%s, dim=%d)",
                ecyc, tuple(w.shape), exp.dim)
    logger.info("[smoke] ALL CHECKS PASSED -- safe to train image RealNVP")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    try:
        run()
        sys.exit(0)
    except Exception:
        logger.error("IMG-RNVP smoke FAILED\n%s", traceback.format_exc())
        sys.exit(1)
