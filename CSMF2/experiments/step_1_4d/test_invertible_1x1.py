# =============================================================================
# STEP-1_1 v0.1 -- experiments.step_1_4d.test_invertible_1x1  (GLOW-LADDER step 2)
# Purpose: acceptance gate for the EXISTING glow.invertible_1x1.InvertibleConv1x1
#   (1) init: W is ~rotation (|det W| ~ 1, ldj ~ 0) at construction
#   (2) det(W) != 0 and finite (invertible by LU construction)
#   (3) invertibility: inverse(forward(x)) == x
#   (4) logdet sanity: analytic ldj == numerical logdet of the Jacobian (f64)
#   (5) ldj == H*W*sum(log_s) == H*W*log|det W|, constant per sample
# Run (from /home/benjamin/CSMFII, venv active):
#   python -m CSMF2.experiments.step_1_4d.test_invertible_1x1
# Exit 0 = all pass; any failure -> raise (no silent pass).
# =============================================================================
from __future__ import annotations
import torch

from CSMF2.models.flows.glow.invertible_1x1 import InvertibleConv1x1


def _check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"[FAIL] {name} {detail}")
    print(f"[pass] {name} {detail}")


def main() -> None:
    torch.manual_seed(0)
    dtype = torch.float64
    device = "cpu"

    C = 4                              # post-squeeze MNIST channel count
    conv = InvertibleConv1x1(C, seed=0).to(device).to(dtype)

    # ---- (1) init is ~rotation: |det W| ~ 1, ldj ~ 0 -------------------------
    W0 = conv._W().detach()
    det0 = torch.linalg.det(W0)
    _check("init |det W| ~ 1", torch.allclose(det0.abs(),
           torch.tensor(1.0, dtype=dtype), atol=1e-6),
           f"|det|={float(det0.abs()):.6f}")
    ldj0 = conv.log_s.sum().detach()
    _check("init ldj-per-pixel ~ 0", torch.allclose(ldj0,
           torch.zeros_like(ldj0), atol=1e-6), f"sum(log_s)={float(ldj0):.2e}")

    # ---- (2) det finite & nonzero (perturb params off init) ------------------
    with torch.no_grad():
        conv.log_s += 0.4 * torch.randn_like(conv.log_s)
        conv.L += 0.2 * torch.randn_like(conv.L)
        conv.U += 0.2 * torch.randn_like(conv.U)
    W = conv._W().detach()
    det = torch.linalg.det(W)
    _check("det(W) finite & nonzero", bool(torch.isfinite(det)) and det.abs() > 1e-8,
           f"det={float(det):.4e}")

    # ---- (3) invertibility ---------------------------------------------------
    B, H, Wd = 8, 7, 7
    x = torch.randn(B, C, H, Wd, dtype=dtype, device=device)
    y, ldj = conv(x)
    x_rt = conv.inverse(y)
    err = (x_rt - x).abs().max()
    _check("invertibility inverse(forward(x))==x", err < 1e-9,
           f"max abs err={err:.2e}")

    # ---- (4) logdet sanity: analytic vs numerical Jacobian (tiny instance) ---
    Bs, Cs, Hs, Ws = 1, 3, 2, 2
    cs = InvertibleConv1x1(Cs, seed=1).to(device).to(dtype)
    with torch.no_grad():
        cs.log_s += 0.3 * torch.randn_like(cs.log_s)
        cs.L += 0.2 * torch.randn_like(cs.L)
        cs.U += 0.2 * torch.randn_like(cs.U)
    xs = torch.randn(Bs, Cs, Hs, Ws, dtype=dtype, device=device)

    def f(v):
        return cs(v.view(Bs, Cs, Hs, Ws))[0].reshape(-1)

    J = torch.autograd.functional.jacobian(f, xs.reshape(-1))
    sign, num_logdet = torch.linalg.slogdet(J)
    analytic = cs(xs)[1][0].detach()
    _check("analytic ldj == numerical logdet",
           torch.allclose(analytic, num_logdet, atol=1e-8),
           f"analytic={float(analytic):.6f} numerical={float(num_logdet):.6f}")

    # ---- (5) ldj == H*W*log|det W|  and  == H*W*sum(log_s) -------------------
    Wcs = cs._W().detach()
    logabsdet = torch.log(torch.linalg.det(Wcs).abs())
    _check("ldj == H*W*log|det W|",
           torch.allclose(analytic, (Hs * Ws) * logabsdet, atol=1e-8),
           f"diff={float((analytic-(Hs*Ws)*logabsdet).abs()):.2e}")
    closed = (Hs * Ws) * cs.log_s.sum().detach()
    _check("ldj == H*W*sum(log_s)", torch.allclose(analytic, closed, atol=1e-10),
           f"diff={float((analytic-closed).abs()):.2e}")
    ldj_full = cs(torch.randn(5, Cs, Hs, Ws, dtype=dtype))[1]
    _check("ldj constant across batch",
           torch.allclose(ldj_full, ldj_full[:1].expand_as(ldj_full), atol=0),
           f"shape={tuple(ldj_full.shape)}")

    print("\nALL INVERTIBLE-1x1 CHECKS PASSED")


if __name__ == "__main__":
    main()
