# =============================================================================
# SEQREF-I1 v0.1 -- src.adjoint_check
# LIFETIME: KEEP
# Purpose: adjoint preflight for the masked-Fourier operator (EXEC 3.12 +
#   protocol in EXEC section 5 I1). Conjugate-aware complex inner products,
#   FULL complex comparison:
#       rel = |<Ax,r> - <x,Aᴴr>| / max(|<Ax,r>|, |<x,Aᴴr>|, eps)
#   >=5 CPU trials + >=5 GPU trials, complex64 (complex128 optional
#   reference). ENGINEERING tolerances (not scientific success criteria),
#   SEPARATED BY DTYPE: complex64 rel <= 1e-5 · complex128 rel <= 1e-10.
#   GPU: runs when CUDA is available; when unavailable the record states
#   "cuda_not_available" EXPLICITLY (logged as a warning) -- never a silent
#   skip. A CUDA-available machine that fails a GPU trial FAILS the preflight.
# CONVENTION: logger.error + raise on failure. No fallback.
# REPLACES the MNIST real-inner-product check (S1 ledger: INHERIT pattern /
#   REBUILD arithmetic).
# Changelog (NEW in v0.1):
#   * Introduced complex preflight: per-trial records (Re/Im of both inner
#     products, abs diff, rel error, dtype, device, mask seed, shapes),
#     dtype-separated tolerances, CLI entry point.
# Update summary: the preflight is arithmetic-correct for complex operators;
#   records everything EXEC section 5 I1 requires.
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging

import torch

from .fastmri_data import make_cartesian_mask, CELL_HW
from .forward_operator import MaskedFourierOperator

logger = logging.getLogger("seqref_mri.adjoint_check")

__version__ = "0.1"
__abbr__ = "SEQREF-I1"

_EPS = 1e-30
TOL = {torch.complex64: 1e-5, torch.complex128: 1e-10}   # EXEC section 5 I1


def _cdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # conjugate-aware complex inner product <a, b> = sum a * conj(b)
    return (a * b.conj()).sum()


def adjoint_trial(op: MaskedFourierOperator, *, dtype: torch.dtype,
                  device: str, gen: torch.Generator) -> dict:
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    hw = op.image_hw

    def crand() -> torch.Tensor:
        re = torch.randn(hw, hw, generator=gen, dtype=real_dtype)
        im = torch.randn(hw, hw, generator=gen, dtype=real_dtype)
        return torch.complex(re, im).to(device)

    x, r = crand(), crand()
    lhs = _cdot(op.A(x), r)                 # <Ax, r>
    rhs = _cdot(x, op.A_adjoint(r))         # <x, Aᴴr>
    diff = (lhs - rhs).abs().item()
    rel = diff / max(lhs.abs().item(), rhs.abs().item(), _EPS)
    return {"dtype": str(dtype), "device": device,
            "lhs_re": lhs.real.item(), "lhs_im": lhs.imag.item(),
            "rhs_re": rhs.real.item(), "rhs_im": rhs.imag.item(),
            "abs_diff": diff, "rel_error": rel,
            "shape": [hw, hw]}


def run_preflight(*, n_trials: int = 5, seed: int = 20260901,
                  include_complex128: bool = True) -> dict:
    if n_trials < 5:
        logger.error("[adjchk] protocol requires >=5 trials, got %d", n_trials)
        raise ValueError(f"protocol requires >=5 trials, got {n_trials}")
    devices = ["cpu"]
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        devices.append("cuda")
    else:
        logger.warning("[adjchk] CUDA NOT AVAILABLE -- GPU trials not run; "
                       "recorded explicitly, not silently skipped")
    dtypes = [torch.complex64] + ([torch.complex128] if include_complex128
                                  else [])
    gen = torch.Generator().manual_seed(seed)
    trials, max_rel = [], {}
    for device in devices:
        for dtype in dtypes:
            for t in range(n_trials):
                mask_seed = seed + 1000 * t + (0 if device == "cpu" else 500)
                mask = torch.from_numpy(
                    make_cartesian_mask(CELL_HW, mask_seed))
                op = MaskedFourierOperator(mask.to(device))
                rec = adjoint_trial(op, dtype=dtype, device=device, gen=gen)
                rec["mask_seed"] = mask_seed
                tol = TOL[dtype]
                rec["tolerance"] = tol
                rec["pass"] = rec["rel_error"] <= tol
                trials.append(rec)
                key = f"{device}/{dtype}"
                max_rel[key] = max(max_rel.get(key, 0.0), rec["rel_error"])
                if not rec["pass"]:
                    logger.error("[adjchk] FAIL %s trial %d: rel=%.3e > "
                                 "tol=%.0e", key, t, rec["rel_error"], tol)
                    raise RuntimeError(
                        f"adjoint preflight FAIL {key}: rel {rec['rel_error']:.3e}"
                        f" > tol {tol:.0e}")
    result = {"n_trials_per_cell": n_trials,
              "cuda_available": cuda_available,
              "gpu_status": "ran" if cuda_available else "cuda_not_available",
              "tolerances": {str(k): v for k, v in TOL.items()},
              "max_rel_error": max_rel, "trials": trials, "verdict": "PASS"}
    for key, v in max_rel.items():
        logger.info("[adjchk] PASS %s: max rel %.3e", key, v)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--no-complex128", action="store_true")
    p.add_argument("--out", type=str, default=None,
                   help="optional JSON output path")
    a = p.parse_args()
    result = run_preflight(n_trials=a.trials, seed=a.seed,
                           include_complex128=not a.no_complex128)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("[adjchk] wrote %s", a.out)


if __name__ == "__main__":
    main()
