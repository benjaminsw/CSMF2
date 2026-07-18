# SEQREF-SMK-V09 v0.1 -- smoke_v09_gate
# LIFETIME: EPHEMERAL
# One-batch smoke for trainer v0.9 / model v0.3 BEFORE the V1 run. Run from
# repo root:  python -m seqref_mri.scripts.smoke_v09_gate
# Checks (review-mandated): scalar fwd+bwd; spatial fwd+bwd;
# gate_spatial.conv2.weight.grad nonzero; correction-layer grad nonzero;
# g.shape == dx.shape (spatial); initial spatial g_mean ~= 0.05.
# No fallback/mock: any failed check raises.
import logging
import torch

from seqref_mri.src.refiners.coupling_regressor import CplRegRefiner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seqref_mri.smoke_v09")

B, H, W = 8, 28, 28
G_INIT = 0.05


def _one(mode: str) -> None:
    torch.manual_seed(0)
    m = CplRegRefiner(flavor="realnvp", gate_mode=mode)
    inp = torch.randn(B, 3, H, W)
    x0 = torch.rand(B, 1, H, W)
    x_true = torch.rand(B, 1, H, W)
    x1, dx, g = m(inp, x0)
    if mode == "spatial":
        if g.shape != dx.shape:
            raise AssertionError(f"g {tuple(g.shape)} != dx {tuple(dx.shape)}")
        gm = float(g.mean())
        if not (0.8 * G_INIT <= gm <= 1.2 * G_INIT):
            raise AssertionError(f"initial spatial g_mean {gm:.5f} off 0.05")
        logger.info("[smoke:%s] init g_mean=%.5f shape ok", mode, gm)
    loss = (x1 - x_true).pow(2).mean()
    loss.backward()
    cpl = [p.grad.abs().sum() for n, p in m.named_parameters()
           if n.startswith("layers.") and p.grad is not None]
    if not cpl or not torch.stack(cpl).sum() > 0:
        raise AssertionError(f"[{mode}] correction-layer grads all zero/None")
    if mode == "spatial":
        wg = m.gate_spatial.conv2.weight.grad
        if wg is None or not wg.abs().sum() > 0:
            raise AssertionError("gate_spatial.conv2.weight.grad zero/None")
        logger.info("[smoke:spatial] gate + correction grads nonzero")
    else:
        logger.info("[smoke:scalar] fwd+bwd + correction grads nonzero")


for mode in ("scalar", "spatial"):
    _one(mode)
print("SMOKE_V09: ALL CHECKS PASSED")
