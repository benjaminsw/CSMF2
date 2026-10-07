# SEQREF-V03L v0.2 -- src.flows.v03_objective
# LIFETIME: KEEP
# =============================================================================
# Purpose: V03 hybrid objective (V03SPEC v0.3 SS6, article 163 FROZEN;
#          V03PLAN v0.3 SS4 step 1, article 164 FROZEN).
#            L = L_density + lambda_rec * L_rec
# Locked quantities (quoted, never CLI-tunable):
#   * L_density = train_free_flow.nll_objective -- the V02 production NLL
#     path CONSUMED, never reimplemented
#   * L_rec = normalised-space MSE on the z=0 deterministic readout (R0),
#     mean reduction over batch AND pixels; the R0 image is built by the
#     registered decode chain (decode_scalars -> unpack -> unstandardise
#     -> decode_normalised), mirrored 1:1 in torch so autograd flows
#   * lambda_rec = 0 returns the V02 NLL tensor object UNTOUCHED
#     (bit-exact code path; fixture f02 gates this)
#   * data_range = 1.0 normalised space (evaluation-side PSNR only;
#     L_rec itself is plain MSE per SPEC SS6)
# Differentiable R0 path: identical arithmetic ORDER to
#   ffr.decode_to_image (float32 decode_scalars -> float64 unpack ->
#   float64 unstandardise -> complex128 decode_normalised), minus the
#   registered .detach()/numpy boundary that exists only for evidence
#   evaluation. Forward equivalence vs the production decode is gated by
#   fixture f03 (max |diff| <= 1e-9); the production evaluation path is
#   unchanged.
# CONVENTION: logger.error + typed raise (V03LError). No fallback, no
#   mock, no placeholder, no silent pass.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import logging

import numpy as np
import torch

from seqref_mri.src import free_flow_runtime as ffr
from seqref_mri.src import residual_decoder as dec
from seqref_mri.scripts import train_free_flow as tff

logger = logging.getLogger("seqref_mri.v03_objective")

__version__ = "0.2"
__abbr__ = "SEQREF-V03L"

DECODE_EQ_TOL = 1e-9          # f03 forward-equivalence tolerance (locked)


class V03LError(RuntimeError):
    """Single typed identity for the V03 objective module."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V03LError(code, message)


def r0_reconstruction_differentiable(model, cond_in: torch.Tensor,
                                     mask: torch.Tensor,
                                     y_raw: torch.Tensor,
                                     amax: torch.Tensor,
                                     maps: list) -> torch.Tensor:
    """z=0 readout (R0) with autograd: (B,H,W) complex128 image.

    maps: per-sample (cmap, vecs) aligned with the batch -- the mandatory
    per-realisation re-derived coordinate maps and their registered
    standardisation vectors (float64 numpy, consumed never recomputed).
    Arithmetic order mirrors ffr.decode_to_image exactly.
    """
    b = int(cond_in.shape[0])
    if len(maps) != b:
        _fail("MAP_BATCH_MISMATCH",
              f"{len(maps)} coordinate maps for batch of {b}")
    z0 = torch.zeros(b, ffr.FLOW_DIM_REAL, dtype=cond_in.dtype,
                     device=cond_in.device)
    us = model.decode_scalars(z0, cond_in, mask)          # (B,13824) f32
    if not torch.isfinite(us).all():
        _fail("U_NON_FINITE",
              "decode_scalars produced a non-finite standardised state; "
              "no fallback is permitted")
    us64 = us.to(torch.float64)
    re_s, im_s = us64[..., 0::2], us64[..., 1::2]         # registered unpack
    m_re = torch.as_tensor(np.stack([v["mean_re"] for _, v in maps]),
                           dtype=torch.float64, device=us64.device)
    s_re = torch.as_tensor(np.stack([v["scale_re"] for _, v in maps]),
                           dtype=torch.float64, device=us64.device)
    m_im = torch.as_tensor(np.stack([v["mean_im"] for _, v in maps]),
                           dtype=torch.float64, device=us64.device)
    s_im = torch.as_tensor(np.stack([v["scale_im"] for _, v in maps]),
                           dtype=torch.float64, device=us64.device)
    u_re = re_s * s_re + m_re                             # exact inverse
    u_im = im_s * s_im + m_im
    if not (torch.isfinite(u_re).all() and torch.isfinite(u_im).all()):
        _fail("U_NON_FINITE",
              "the unstandardised state contains a non-finite component; "
              "no fallback is permitted")
    xs = []
    for i in range(b):
        u_i = torch.complex(u_re[i], u_im[i]).unsqueeze(0)   # (1,n_free)
        x_i = dec.decode_normalised(y_raw[i:i + 1], amax[i:i + 1],
                                    u_i, maps[i][0])          # (1,H,W)
        xs.append(x_i[0])
    return torch.stack(xs)                                    # (B,H,W) c128


def mse_normalised(mag_hat: torch.Tensor,
                   mag_true: torch.Tensor) -> torch.Tensor:
    """Normalised-space MSE, mean reduction over batch AND pixels
    (SPEC SS6). Both inputs (B,H,W) real."""
    if mag_hat.shape != mag_true.shape:
        _fail("LOSS_CONTRACT_VIOLATION",
              f"magnitude shapes differ: {tuple(mag_hat.shape)} vs "
              f"{tuple(mag_true.shape)}")
    val = ((mag_hat - mag_true) ** 2).mean()
    if not torch.isfinite(val):
        _fail("LOSS_CONTRACT_VIOLATION",
              "L_rec is non-finite; no fallback is permitted")
    return val


def l_rec(model, cond_in: torch.Tensor, mask: torch.Tensor,
          y_raw: torch.Tensor, amax: torch.Tensor, tgt_norm: torch.Tensor,
          maps: list) -> torch.Tensor:
    """Scalar L_rec: MSE between |R0 image| and the normalised target
    magnitude tgt_norm (B,1,H,W from train_base._prepare)."""
    x = r0_reconstruction_differentiable(model, cond_in, mask, y_raw,
                                         amax, maps)
    mag = x.abs()
    tgt = tgt_norm.view(mag.shape).to(torch.float64)
    return mse_normalised(mag, tgt)


def combined_loss(model, u_scaled: torch.Tensor, cond_in: torch.Tensor,
                  mask: torch.Tensor, *, lam: float,
                  y_raw: torch.Tensor, amax: torch.Tensor,
                  tgt_norm: torch.Tensor, maps: list) -> torch.Tensor:
    """L = L_density + lam * L_rec. lam == 0.0 returns the V02 NLL tensor
    UNTOUCHED (fixture f02: object identity with nll_objective output)."""
    nll = tff.nll_objective(model, u_scaled, cond_in, mask)
    if float(lam) == 0.0:
        return nll
    rec = l_rec(model, cond_in, mask, y_raw, amax, tgt_norm, maps)
    total = nll + float(lam) * rec
    if not torch.isfinite(total):
        _fail("LOSS_CONTRACT_VIOLATION",
              f"combined objective is non-finite (nll={float(nll)!r}, "
              f"l_rec={float(rec)!r}, lam={lam!r})")
    return total


# ---------------------------------------------------------------------------
# Selftest: fixtures f02 (lambda=0 equivalence) and f03 (L_rec contract +
# differentiable-decode forward equivalence). CPU-scale synthetic data;
# production code paths only, no holdout contact. PASS=0 / ERROR=2.
# ---------------------------------------------------------------------------

def _selftest() -> int:
    from seqref_mri.src.fastmri_data import make_cartesian_mask

    torch.manual_seed(0)
    model = ffr.build_model()                      # registered init seed
    b, h, w = 2, ffr.GRID_H, ffr.GRID_W
    maps = []
    masks = []
    for i in range(b):
        m = make_cartesian_mask(w, seed=1000 + i)  # 4x-style synthetic mask
        cols = [int(c) for c in np.flatnonzero(m)]
        cmap = dec.build_coordinate_map(cols, h, w)
        n = cmap.n_free_complex
        vecs = {"mean_re": np.zeros(n), "scale_re": np.ones(n),
                "mean_im": np.zeros(n), "scale_im": np.ones(n),
                "n_free_complex": n}               # fixture-synthetic
        maps.append((cmap, vecs))
        masks.append(torch.as_tensor(m, dtype=torch.bool))
    mask = torch.stack(masks)                       # (B,96) bool
    cond_in = torch.randn(b, 2, h, w)
    y_raw = torch.randn(b, h, w, dtype=torch.complex64)
    amax = torch.rand(b) + 0.5
    tgt_norm = torch.rand(b, 1, h, w)
    u_scaled = torch.randn(b, ffr.FLOW_DIM_REAL)

    # f02: lam=0 must be the untouched V02 NLL path -- (a) bitwise value
    # equality with a direct nll_objective call on the same inputs, and
    # (b) behavioural proof the R0 branch is never entered: maps=[] would
    # abort MAP_BATCH_MISMATCH if L_rec were computed.
    nll_ref = tff.nll_objective(model, u_scaled, cond_in, mask)
    nll_l0 = combined_loss(model, u_scaled, cond_in, mask, lam=0.0,
                           y_raw=y_raw, amax=amax, tgt_norm=tgt_norm,
                           maps=maps)
    if float(nll_l0) != float(nll_ref):
        _fail("LAMBDA0_PATH_MISMATCH",
              "lam=0 value differs from the V02 NLL path")
    nll_l0_bare = combined_loss(model, u_scaled, cond_in, mask, lam=0.0,
                                y_raw=None, amax=None, tgt_norm=None,
                                maps=[])
    if float(nll_l0_bare) != float(nll_ref):
        _fail("LAMBDA0_PATH_MISMATCH",
              "lam=0 with empty maps differs; the R0 branch was entered")
    logger.info("[%s] f02 PASS: lam=0 bit-exact V02 path (R0 branch cold)",
                __abbr__)

    # f02b: lam=0 GRADIENT equivalence -- bitwise-identical parameter
    # gradients to a direct V02 nll_objective backward on the same inputs.
    # Compared via an incremental SHA-256 over (name, presence, raw
    # bytes) in fixed parameter order -- no gradient clones, no dtype
    # copies (OOM-safe); raw f32 bytes ARE the bitwise truth.
    def _grad_digest() -> str:
        h = hashlib.sha256()
        for n, p in model.named_parameters():
            g = p.grad
            h.update(n.encode())
            h.update(b"\x01" if g is not None else b"\x00")
            if g is not None:
                h.update(g.detach().numpy().tobytes())
        return h.hexdigest()

    model.zero_grad(set_to_none=True)
    tff.nll_objective(model, u_scaled, cond_in, mask).backward()
    ref_digest = _grad_digest()
    model.zero_grad(set_to_none=True)
    combined_loss(model, u_scaled, cond_in, mask, lam=0.0,
                  y_raw=None, amax=None, tgt_norm=None, maps=[]).backward()
    if _grad_digest() != ref_digest:
        _fail("LAMBDA0_PATH_MISMATCH",
              "lam=0 parameter gradients differ from the V02 path")
    logger.info("[%s] f02b PASS: lam=0 gradients bitwise-identical",
                __abbr__)
    model.zero_grad(set_to_none=True)

    # f03a: MSE contract -- zero iff identical, positive otherwise.
    mag = torch.rand(2, h, w, dtype=torch.float64)
    if float(mse_normalised(mag, mag.clone())) != 0.0:
        _fail("LOSS_CONTRACT_VIOLATION", "MSE(x,x) != 0")
    if float(mse_normalised(mag, mag + 0.1)) <= 0.0:
        _fail("LOSS_CONTRACT_VIOLATION", "MSE(x,y) <= 0 for x != y")
    # f03b: differentiable decode forward-equivalence vs production path.
    z0 = torch.zeros(1, ffr.FLOW_DIM_REAL)
    x_diff = r0_reconstruction_differentiable(
        model, cond_in[:1], mask[:1], y_raw[:1], amax[:1], [maps[0]])
    x_prod = ffr.decode_to_image(model, z0, cond_in[:1], mask[:1],
                                 y_raw[:1], amax[:1], maps[0][0],
                                 maps[0][1])
    d = float((x_diff - x_prod.to(torch.complex128)).abs().max())
    if not np.isfinite(d) or d > DECODE_EQ_TOL:
        _fail("LOSS_CONTRACT_VIOLATION",
              f"differentiable R0 decode diverges from production decode: "
              f"max|diff|={d!r} > {DECODE_EQ_TOL}")
    logger.info("[%s] f03ab PASS: MSE contract ok; decode max|diff|=%.3e",
                __abbr__, d)
    # f03c: gradient flows from L_rec into the model.
    model.zero_grad(set_to_none=True)
    rec = l_rec(model, cond_in, mask, y_raw, amax, tgt_norm, maps)
    rec.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        _fail("LOSS_CONTRACT_VIOLATION", "L_rec produced no gradients")
    gnorm = float(torch.sqrt(sum((g ** 2).sum() for g in grads)))
    if not np.isfinite(gnorm) or gnorm == 0.0:
        _fail("LOSS_CONTRACT_VIOLATION",
              f"L_rec gradient norm is {gnorm!r}; must be finite and > 0")
    # combined path smoke: lam=1 finite.
    total = combined_loss(model, u_scaled, cond_in, mask, lam=1.0,
                          y_raw=y_raw, amax=amax, tgt_norm=tgt_norm,
                          maps=maps)
    if not torch.isfinite(total):
        _fail("LOSS_CONTRACT_VIOLATION", "combined lam=1 non-finite")
    logger.info("[%s] f03c PASS: grad norm=%.6e; combined(1.0)=%.6f",
                __abbr__, gnorm, float(total))
    print(f"PASS {__abbr__} v{__version__} selftest (f02+f02b, f03)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="v03_objective")
    ap.add_argument("--mode", choices=("selftest",), required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        if args.mode == "selftest":
            return _selftest()
    except V03LError as exc:
        logger.error("[%s] selftest ERROR: %s", __abbr__, exc)
        return 2
    except Exception:
        logger.exception("[%s] unexpected exception in selftest", __abbr__)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# Changelog
# v0.1 (2026-10-06): initial implementation under V03SPEC v0.3 (163,
#   FROZEN) / V03PLAN v0.3 (164, FROZEN) SS4 step 1. combined_loss with
#   bit-exact lam=0 path; differentiable R0 readout mirroring
#   ffr.decode_to_image arithmetic order; normalised-space MSE (mean over
#   batch+pixels); fixtures f02/f03 in --mode selftest; typed V03LError,
#   logger.error on every failure path.
# v0.2 (2026-10-06): review strengthening (non-blocking reviewer item):
#   f02b added -- lam=0 parameter gradients must be bitwise-identical to
#   a direct V02 nll_objective backward (incremental SHA-256 over raw
#   grad bytes; no clones). Scalar equality retained as f02. No
#   production-path change.
# ---------------------------------------------------------------------------
