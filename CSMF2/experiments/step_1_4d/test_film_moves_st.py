# =============================================================================
# CPATH v0.1 -- experiments.step_1_4d.test_film_moves_st  (GLOW-FIX Phase A)
# Purpose: Phase-A wiring fork for the Glow conditioner-death fix. Runs the
#          existing cond_path_probe on a FRESH (untrained) CondGlow and decides
#          whether y reaches (s,t) MEANINGFULLY at init -- not merely > 0.
#
#   The plan's gate "st_s_y_std > 0" is too lenient: the real init signal is
#   ~2.6e-6 (numerically nonzero but ~5 orders below a working conditioner),
#   because conditioning is throttled by the PRODUCT film_gain * conv3_init,
#   both small (the FiLMHead-v0.4 "two small-inits in series" trap, only
#   partially escaped). So this gate uses a MEANINGFUL floor and a lever sweep.
#
# Verdict:
#   DEAD       st_s_y_std ~ 0 AND does not rise with conv3/gain/y-residual
#                -> conditioner->FiLM WIRING is broken; STOP, debug wiring.
#   THROTTLED  st_s_y_std tiny at baseline but RISES with conv3 / film_gain /
#              y_residual levers (multiplicative)  -> wiring present but
#              strangled at init; Phase B levers (esp. cond_y_residual_alpha)
#              are the fix; conv3/film_gain floor may also be needed.
#   WIRED-OK   baseline st_s_y_std >= THRESH  -> conditioning meaningfully
#              alive at init; the death is purely OBJECTIVE -> Phase B.
#
# Run (from /home/benjamin/CSMFII, venv active):
#   python -m CSMF2.experiments.step_1_4d.test_film_moves_st
# Exit 0 always (this is a DIAGNOSTIC fork, not a hard pass/fail); it PRINTS
# the verdict and the lever table so Phase B/C can be chosen on evidence.
# =============================================================================
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "CPATH"

import torch

from CSMF2.models.conditioner import Conditioner
from CSMF2.models.flows.glow.glow_step import GlowStep
from CSMF2.experiments.step_1_1.diagnostics.cond_path_probe import cond_path_probe

# meaningful floor: below this, (s,t) movement across y is negligible vs a
# working conditioner. Tunable; 1e-4 is ~50x the observed dead baseline (2.6e-6)
# and ~the level the y-residual bypass reaches at alpha=0.3.
THRESH = 1e-4


def _build(*, n_layers: int = 8, h_dim: int = 128, image_shape=(1, 28, 28),
           film_gain_init: float = 0.3, y_residual_alpha_init: float = 0.0):
    """Minimal CondGlow surface (.cond/.layers/.image_shape) from REAL parts."""
    C, H, W = image_shape

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_shape = image_shape
            self.dim = C * H * W
            if y_residual_alpha_init > 0:
                self.cond = Conditioner(
                    h_dim=h_dim, y_residual_alpha_init=y_residual_alpha_init,
                    y_input_size=(H // 2) * (W // 2))
            else:
                self.cond = Conditioner(h_dim=h_dim)
            self.layers = torch.nn.ModuleList([
                GlowStep(num_channels=C * 4, coupling_hidden=64, h_dim=h_dim,
                         flip=bool(i % 2), s_max=2.0, film_hidden=128,
                         film_depth=2, film_use_gelu=True, inv1x1_seed=i,
                         film_gain_init=film_gain_init)
                for i in range(n_layers)])
    return _M()


def _probe(m, seed=0):
    torch.manual_seed(seed)
    y = torch.randn(6, 1, 14, 14)
    return cond_path_probe(m, y, n_y=6)["cond_path_probe"]


def main() -> None:
    print(f"[Phase A] meaningful threshold THRESH = {THRESH:.1e}\n")

    rows = []
    # baseline (real init)
    torch.manual_seed(0)
    rows.append(("baseline (conv3 std0.01, gain0.3)", _probe(_build())))
    # lever: conv3 x10
    torch.manual_seed(0)
    m = _build()
    with torch.no_grad():
        for st in m.layers:
            st.coupling.conv3.weight.normal_(0, 0.1)
    rows.append(("conv3 std0.1 (10x)", _probe(m)))
    # lever: film_gain 1.0
    torch.manual_seed(0)
    rows.append(("film_gain=1.0", _probe(_build(film_gain_init=1.0))))
    # lever: y-residual bypass (the conditioner's escape hatch)
    torch.manual_seed(0)
    rows.append(("y_residual_alpha=0.3",
                 _probe(_build(y_residual_alpha_init=0.3))))

    print(f"{'config':36s} {'st_s_y_std':>12s} {'st_t_y_std':>12s}")
    for name, o in rows:
        print(f"{name:36s} {o['st_s_y_std']:12.3e} {o['st_t_y_std']:12.3e}")

    base = rows[0][1]["st_s_y_std"]
    yres = rows[-1][1]["st_s_y_std"]
    rises = yres > 10 * base                      # signal scales with a lever

    print()
    if base >= THRESH:
        verdict = "WIRED-OK"
        msg = ("conditioning meaningfully alive at init -> death is OBJECTIVE "
               "-> proceed to Phase B")
    elif rises:
        verdict = "THROTTLED"
        msg = (f"baseline {base:.2e} below {THRESH:.0e} but RISES with levers "
               f"(y_residual gives {yres:.2e}, {yres/base:.0f}x). Wiring present "
               f"but strangled by film_gain*conv3_init. FIX: Phase B levers "
               f"(esp. cond_y_residual_alpha_init=0.3); consider conv3/film_gain "
               f"floor. Phase B alone may be uphill -- open the throttle too.")
    else:
        verdict = "DEAD"
        msg = ("signal ~0 and does NOT rise with any lever -> conditioner->FiLM "
               "WIRING is broken. STOP. Debug cond->FiLMHead->_st before any "
               "training.")

    print(f"[Phase A VERDICT] {verdict}")
    print(f"  {msg}")


if __name__ == "__main__":
    main()
