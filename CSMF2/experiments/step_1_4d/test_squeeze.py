# =============================================================================
# STEP-1_1 v0.1 -- experiments.step_1_4d.test_squeeze  (GLOW-LADDER step 3)
# Purpose: shape / round-trip gate for glow.squeeze (pure reshape, logdet 0).
#   (1) shape contract: squeeze [B,C,H,W] -> [B,4C,H/2,W/2]; unsqueeze inverts
#   (2) round-trip: unsqueeze(squeeze(x)) == x exactly (and squeeze(unsqueeze)==x)
#   (3) content: round-trip is bit-exact (no scramble) for a ramp tensor
#   (4) MNIST size: 28x28 -> 14x14 works (even); C=1 -> 4
#   (5) guards: odd H/W and C%4!=0 raise
# NOTE: no ldj here -- squeeze is a pure permutation/reshape, logdet = 0.
# Run (from /home/benjamin/CSMFII, venv active):
#   python -m CSMF2.experiments.step_1_4d.test_squeeze
# Exit 0 = all pass; any failure -> raise.
# =============================================================================
from __future__ import annotations
import torch

from CSMF2.models.flows.glow.squeeze import squeeze2x2, unsqueeze2x2


def _check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"[FAIL] {name} {detail}")
    print(f"[pass] {name} {detail}")


def main() -> None:
    torch.manual_seed(0)
    dtype = torch.float64

    # ---- (1) shape contract --------------------------------------------------
    B, C, H, W = 8, 3, 8, 8
    x = torch.randn(B, C, H, W, dtype=dtype)
    s = squeeze2x2(x)
    _check("squeeze shape [B,C,H,W]->[B,4C,H/2,W/2]",
           tuple(s.shape) == (B, 4 * C, H // 2, W // 2), f"got {tuple(s.shape)}")
    u = unsqueeze2x2(s)
    _check("unsqueeze restores shape", tuple(u.shape) == (B, C, H, W),
           f"got {tuple(u.shape)}")

    # ---- (2) round-trip both directions, bit-exact ---------------------------
    _check("unsqueeze(squeeze(x)) == x (exact)",
           torch.equal(unsqueeze2x2(squeeze2x2(x)), x))
    xs = torch.randn(B, 4 * C, H // 2, W // 2, dtype=dtype)
    _check("squeeze(unsqueeze(x)) == x (exact)",
           torch.equal(squeeze2x2(unsqueeze2x2(xs)), xs))

    # ---- (3) content preserved (ramp makes a scramble visible) ---------------
    ramp = torch.arange(B * C * H * W, dtype=dtype).reshape(B, C, H, W)
    _check("round-trip preserves content (no scramble)",
           torch.equal(unsqueeze2x2(squeeze2x2(ramp)), ramp))

    # ---- (4) MNIST size 28->14, C=1->4 ---------------------------------------
    xm = torch.randn(5, 1, 28, 28, dtype=dtype)
    sm = squeeze2x2(xm)
    _check("MNIST 28x28 C1 -> 14x14 C4", tuple(sm.shape) == (5, 4, 14, 14),
           f"got {tuple(sm.shape)}")
    _check("MNIST round-trip exact", torch.equal(unsqueeze2x2(sm), xm))

    # ---- (5) guards ----------------------------------------------------------
    raised = False
    try:
        squeeze2x2(torch.randn(1, 1, 7, 8))      # odd H
    except ValueError:
        raised = True
    _check("odd H raises", raised)

    raised = False
    try:
        unsqueeze2x2(torch.randn(1, 3, 4, 4))    # C % 4 != 0
    except ValueError:
        raised = True
    _check("C%4!=0 raises", raised)

    print("\nALL SQUEEZE CHECKS PASSED")


if __name__ == "__main__":
    main()
