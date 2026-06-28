# =============================================================================
# STEP-1_1 v0.1 -- experiments.step_1_4d.test_glow_step  (GLOW-LADDER step 5)
# Purpose: acceptance gate for the ASSEMBLED glow.glow_step.GlowStep
#          (Actnorm -> InvertibleConv1x1 -> AffineCoupling2D).
#   (1) DDI first: actnorm.init_from_batch on the raw input (forward order)
#   (2) end-to-end invertibility: inverse(forward(x,h),h) == x
#   (3) ldj_total == sum of the three component ldjs
#   (4) logdet sanity: analytic ldj_total == numerical logdet of the WHOLE step
#   (5) inverse composes in REVERSED order (coupling^-1 -> inv1x1^-1 -> actnorm^-1):
#       a same-order inverse would fail round-trip -> (2) is the guard
#   (6) conditioning survives composition: vary h -> output moves (FZDY > 0)
# Run: python -m CSMF2.experiments.step_1_4d.test_glow_step
# Exit 0 = all pass; any failure -> raise.
# =============================================================================
from __future__ import annotations
import torch

from CSMF2.models.flows.glow.glow_step import GlowStep


def _check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"[FAIL] {name} {detail}")
    print(f"[pass] {name} {detail}")


def _mk(C, hidden, h_dim, seed, dtype):
    step = GlowStep(C, hidden, h_dim, flip=False, s_max=2.0,
                    film_hidden=64, film_depth=2, film_use_gelu=True,
                    inv1x1_seed=seed, film_gain_init=0.3).to(dtype)
    return step


def main() -> None:
    torch.manual_seed(0)
    dtype = torch.float64
    C, hidden, h_dim = 4, 32, 16

    step = _mk(C, hidden, h_dim, 0, dtype)
    B, H, W = 8, 7, 7
    x = torch.randn(B, C, H, W, dtype=dtype)
    h = torch.randn(B, h_dim, dtype=dtype)

    # ---- (1) DDI first (forward order: actnorm sees raw input) ---------------
    step.actnorm.init_from_batch(x)
    _check("actnorm DDI fired", bool(step.actnorm.initialised))

    # perturb coupling conv3 + inv1x1 off init so the step is non-trivial
    with torch.no_grad():
        step.coupling.conv3.weight += 0.3 * torch.randn_like(step.coupling.conv3.weight)
        step.inv1x1.log_s += 0.2 * torch.randn_like(step.inv1x1.log_s)

    # ---- (2) end-to-end invertibility ---------------------------------------
    y, ldj = step(x, h)
    x_rt = step.inverse(y, h)
    err = (x_rt - x).abs().max()
    _check("end-to-end invertibility", err < 1e-9, f"max abs err={err:.2e}")

    # ---- (3) ldj_total == sum of component ldjs ------------------------------
    z1, d1 = step.actnorm(x)
    z2, d2 = step.inv1x1(z1)
    z3, d3 = step.coupling(z2, h)
    _check("ldj_total == d_actnorm + d_inv1x1 + d_coupling",
           torch.allclose(ldj, d1 + d2 + d3, atol=1e-10),
           f"max diff={float((ldj-(d1+d2+d3)).abs().max()):.2e}")
    _check("forward output matches manual composition",
           torch.allclose(y, z3, atol=1e-12))

    # ---- (4) logdet sanity on the WHOLE step (tiny instance) -----------------
    Bs, Cs, Hs, Ws = 1, 4, 2, 2
    st = _mk(Cs, 16, h_dim, 1, dtype)
    xs = torch.randn(Bs, Cs, Hs, Ws, dtype=dtype)
    hs = torch.randn(Bs, h_dim, dtype=dtype)
    st.actnorm.init_from_batch(xs)
    with torch.no_grad():
        st.coupling.conv3.weight += 0.5 * torch.randn_like(st.coupling.conv3.weight)
        st.coupling.conv3.bias += 0.2 * torch.randn_like(st.coupling.conv3.bias)
        st.inv1x1.log_s += 0.3 * torch.randn_like(st.inv1x1.log_s)

    def f(v):
        return st(v.view(Bs, Cs, Hs, Ws), hs)[0].reshape(-1)

    J = torch.autograd.functional.jacobian(f, xs.reshape(-1))
    sign, num_logdet = torch.linalg.slogdet(J)
    analytic = st(xs, hs)[1][0].detach()
    _check("analytic ldj_total == numerical logdet(whole step)",
           torch.allclose(analytic, num_logdet, atol=1e-7),
           f"analytic={float(analytic):.6f} numerical={float(num_logdet):.6f}")

    # ---- (5) reversed-order inverse is the correct one -----------------------
    # (already implied by (2); make the failure mode explicit: a same-order
    #  inverse would NOT round-trip. We re-confirm round-trip holds tightly.)
    _check("round-trip confirms reversed-order inverse", err < 1e-9,
           f"err={err:.2e}")

    # ---- (6) conditioning survives composition (FZDY > 0) --------------------
    s2 = _mk(C, hidden, h_dim, 2, dtype)
    s2.actnorm.init_from_batch(x)
    ha = torch.randn(B, h_dim, dtype=dtype)
    hb = torch.randn(B, h_dim, dtype=dtype)
    ya, _ = s2(x, ha)
    yb, _ = s2(x, hb)
    fzdy = (ya - yb).abs().max()
    _check("conditioning survives composition (FZDY > 0)", fzdy > 1e-5,
           f"max|y(h_a)-y(h_b)|={fzdy:.3e}")

    print("\nALL GLOW-STEP CHECKS PASSED")


if __name__ == "__main__":
    main()
