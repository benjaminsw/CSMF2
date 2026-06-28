# =============================================================================
# STEP-1_1 v0.2 -- experiments.step_1_4d.test_actnorm  (GLOW-LADDER step 1 gate)
# Purpose: acceptance gate for the EXISTING glow.actnorm.Actnorm before stacking.
#   (1) DDI: post-actnorm per-channel mean~0, var~1 on the init batch
#   (2) invertibility: inverse(forward(x)) == x
#   (3) logdet sanity: analytic ldj == numerical logdet of the Jacobian (f64)
#   (4) ldj == H*W*sum_c log_s_c, constant per sample
# Run (module-style, from /home/benjamin/CSMFII, venv active):
#   python -m CSMF2.experiments.step_1_4d.test_actnorm
# Exit 0 = all pass; any failure -> raise (no silent pass).
# Changelog (v0.1 -> v0.2):
#   * Retargeted at the real class: glow.actnorm.Actnorm (was actnorm_layer.ActNorm).
#   * Explicit DDI via init_from_batch(x); forward/inverse take NO h arg.
#   * Variance check uses the unbiased estimator (matches torch.std DDI).
# =============================================================================
from __future__ import annotations
import torch

from CSMF2.models.flows.glow.actnorm import Actnorm


def _check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"[FAIL] {name} {detail}")
    print(f"[pass] {name} {detail}")


def main() -> None:
    torch.manual_seed(0)
    dtype = torch.float64
    device = "cpu"

    # ---- (1) DDI statistics --------------------------------------------------
    B, C, H, W = 16, 4, 7, 7           # 4ch mimics post-squeeze MNIST
    x = (torch.randn(B, C, H, W, dtype=dtype, device=device) * 3.0
         + torch.tensor([0.5, -2.0, 10.0, 0.0], dtype=dtype).view(1, C, 1, 1))
    an = Actnorm(C).to(device).to(dtype)
    an.init_from_batch(x)              # explicit DDI (this class does NOT auto-init)
    _check("DDI flag set", bool(an.initialised))

    y, ldj = an(x)                     # forward(x) -> (y, ldj); no h
    ch_mean = y.mean(dim=(0, 2, 3))
    ch_var = y.var(dim=(0, 2, 3), unbiased=True)   # match torch.std (ddof=1)
    _check("DDI per-channel mean ~ 0", torch.allclose(ch_mean,
           torch.zeros_like(ch_mean), atol=1e-6),
           f"max|mean|={ch_mean.abs().max():.2e}")
    _check("DDI per-channel var ~ 1 (unbiased)", torch.allclose(ch_var,
           torch.ones_like(ch_var), atol=1e-6),
           f"max|var-1|={(ch_var-1).abs().max():.2e}")

    # ---- (2) invertibility (perturb params off init so it isn't trivial) -----
    with torch.no_grad():
        an.log_s += 0.3 * torch.randn_like(an.log_s)
        an.b += 0.3 * torch.randn_like(an.b)
    y, ldj = an(x)
    x_rt = an.inverse(y)
    err = (x_rt - x).abs().max()
    _check("invertibility inverse(forward(x))==x", err < 1e-10,
           f"max abs err={err:.2e}")

    # ---- (3) logdet sanity: analytic vs numerical Jacobian (tiny instance) ----
    Bs, Cs, Hs, Ws = 1, 3, 2, 2
    xs = torch.randn(Bs, Cs, Hs, Ws, dtype=dtype, device=device)
    ans = Actnorm(Cs).to(device).to(dtype)
    ans.init_from_batch(xs)
    with torch.no_grad():
        ans.log_s += 0.2 * torch.randn_like(ans.log_s)

    def f(v):
        return ans(v.view(Bs, Cs, Hs, Ws))[0].reshape(-1)

    J = torch.autograd.functional.jacobian(f, xs.reshape(-1))
    sign, num_logdet = torch.linalg.slogdet(J)
    analytic = ans(xs)[1][0].detach()
    _check("logdet sign positive", float(sign) > 0, f"sign={float(sign)}")
    _check("analytic ldj == numerical logdet",
           torch.allclose(analytic, num_logdet, atol=1e-8),
           f"analytic={float(analytic):.6f} numerical={float(num_logdet):.6f}")

    # ---- (4) ldj closed form + per-sample constant ---------------------------
    closed = (Hs * Ws) * ans.log_s.sum()
    _check("ldj == H*W*sum(log_s)", torch.allclose(analytic, closed.detach(),
           atol=1e-10), f"diff={float((analytic-closed).abs()):.2e}")
    ldj_full = ans(torch.randn(5, Cs, Hs, Ws, dtype=dtype))[1]
    _check("ldj constant across batch",
           torch.allclose(ldj_full, ldj_full[:1].expand_as(ldj_full), atol=0),
           f"shape={tuple(ldj_full.shape)}")

    print("\nALL ACTNORM CHECKS PASSED")


if __name__ == "__main__":
    main()
