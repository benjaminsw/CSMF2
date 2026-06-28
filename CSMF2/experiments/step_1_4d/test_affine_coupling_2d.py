# =============================================================================
# STEP-1_1 v0.1 -- experiments.step_1_4d.test_affine_coupling_2d  (GLOW-LADDER step 4)
# Purpose: acceptance gate for glow.affine_coupling_2d.AffineCoupling2D (v0.3).
#   (1) invertibility: inverse(forward(x,h),h) == x
#   (2) logdet sanity: analytic ldj (=sum s) == numerical logdet of d y/d x (f64)
#   (3) near-identity at init: conv3 small-normal -> ||y-x|| small
#   (4) CONDITIONING ALIVE FROM STEP 0 (the v0.3 fix / FZDY):
#         hold x fixed, vary h -> ||y(h_a) - y(h_b)|| MEANINGFULLY > 0 AT INIT,
#         and the sensitivity scales with film_gain (set gain=0 -> sensitivity
#         collapses to ~0, confirming the residual-FiLM path is the carrier).
#       This is exactly what v0.2 failed (h.std ~ 0.022) and v0.3 should pass.
#   (5) film_gain carries gradient (trainable conditioning strength).
# NOTE on (3)+(4): conv3 near-zero makes the coupling near-identity, so a naive
#   h-sensitivity probe could look dead. v0.3 routes conditioning through the
#   residual-FiLM path (gain * gamma_raw/beta), which is alive even with conv3~0.
#   So a LIVE v0.3 shows h-sensitivity AT INIT; a regressed one shows ~0.
# Run: python -m CSMF2.experiments.step_1_4d.test_affine_coupling_2d
# Exit 0 = all pass; any failure -> raise.
# =============================================================================
from __future__ import annotations
import torch

from CSMF2.models.flows.glow.affine_coupling_2d import AffineCoupling2D


def _check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"[FAIL] {name} {detail}")
    print(f"[pass] {name} {detail}")


def main() -> None:
    torch.manual_seed(0)
    dtype = torch.float64
    device = "cpu"
    C, hidden, h_dim = 4, 32, 16        # post-squeeze MNIST = 4ch

    cp = AffineCoupling2D(C, hidden, h_dim, flip=False).to(device).to(dtype)

    B, H, W = 8, 7, 7
    x = torch.randn(B, C, H, W, dtype=dtype, device=device)
    h = torch.randn(B, h_dim, dtype=dtype, device=device)

    # ---- (1) invertibility ---------------------------------------------------
    y, ldj = cp(x, h)
    x_rt = cp.inverse(y, h)
    err = (x_rt - x).abs().max()
    _check("invertibility inverse(forward(x,h),h)==x", err < 1e-9,
           f"max abs err={err:.2e}")

    # ---- (2) logdet sanity: analytic vs numerical (tiny instance) ------------
    Bs, Cs, Hs, Ws = 1, 4, 2, 2
    cs = AffineCoupling2D(Cs, 16, h_dim, flip=False).to(device).to(dtype)
    # perturb conv3 off ~0 so s is non-trivial and the Jacobian is informative
    with torch.no_grad():
        cs.conv3.weight += 0.5 * torch.randn_like(cs.conv3.weight)
        cs.conv3.bias += 0.2 * torch.randn_like(cs.conv3.bias)
    xs = torch.randn(Bs, Cs, Hs, Ws, dtype=dtype, device=device)
    hs = torch.randn(Bs, h_dim, dtype=dtype, device=device)

    def f(v):
        return cs(v.view(Bs, Cs, Hs, Ws), hs)[0].reshape(-1)

    J = torch.autograd.functional.jacobian(f, xs.reshape(-1))
    sign, num_logdet = torch.linalg.slogdet(J)
    analytic = cs(xs, hs)[1][0].detach()
    _check("analytic ldj == numerical logdet",
           torch.allclose(analytic, num_logdet, atol=1e-7),
           f"analytic={float(analytic):.6f} numerical={float(num_logdet):.6f}")

    # ---- (3) bounded/finite at init (conv3 small-normal -> no blow-up) -------
    # NOTE: exact closeness depends on the real FiLMHead residual init; this is
    # a BLOW-UP guard (output stays same order as input), not a strict identity.
    cp0 = AffineCoupling2D(C, hidden, h_dim, flip=False).to(device).to(dtype)
    y0, _ = cp0(x, h)
    rel = (y0 - x).abs().max() / x.abs().max()
    _check("bounded at init (no blow-up)", torch.isfinite(y0).all() and rel < 0.5,
           f"rel||y-x||={rel:.3e}")

    # ---- (4) CONDITIONING ALIVE AT INIT (v0.3 / FZDY) ------------------------
    # same x, two different h: output MUST move at init
    ha = torch.randn(B, h_dim, dtype=dtype, device=device)
    hb = torch.randn(B, h_dim, dtype=dtype, device=device)
    ya, _ = cp0(x, ha)
    yb, _ = cp0(x, hb)
    fzdy = (ya - yb).abs().max()
    _check("conditioning ALIVE at init (FZDY > 0)", fzdy > 1e-4,
           f"max|y(h_a)-y(h_b)|={fzdy:.3e}")

    # gain=0 must collapse the sensitivity -> proves residual-FiLM is the carrier
    with torch.no_grad():
        cp0.film_gain.zero_()
    ya0, _ = cp0(x, ha)
    yb0, _ = cp0(x, hb)
    fzdy0 = (ya0 - yb0).abs().max()
    _check("gain=0 collapses h-sensitivity (carrier = residual FiLM)",
           fzdy0 < 1e-9, f"max|Δ| at gain0={fzdy0:.3e}")

    # ---- (5) film_gain trainable (carries gradient) --------------------------
    cp1 = AffineCoupling2D(C, hidden, h_dim, flip=False).to(device).to(dtype)
    yg, ldjg = cp1(x, h)
    loss = yg.pow(2).mean() + ldjg.mean()
    loss.backward()
    g = cp1.film_gain.grad
    _check("film_gain receives gradient",
           g is not None and torch.isfinite(g).all() and g.abs() > 0,
           f"grad={float(g):.3e}")

    print("\nALL AFFINE-COUPLING-2D CHECKS PASSED")


if __name__ == "__main__":
    main()
