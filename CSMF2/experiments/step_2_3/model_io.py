# =============================================================================
# STEP-2_3 v0.1 -- experiments.step_2_3.model_io  (S2.3 multi-expert + gate load)
# Purpose: load the K CB experts for Stage 2.3 and build the gate. Unlike
#          step_1_2.load_experts (which discards conds and returns FROZEN experts
#          for the frozen-gate stages), 2.3 needs:
#            (a) the experts UNFROZEN (experts train jointly with the gate), and
#            (b) each expert's CONDITIONER kept (it lives INSIDE CBExpert as
#                .cond(y); the consistency term needs h_k = expert_k.cond(y)).
#          So we loop step_1_1_1_1.build_from_report directly (keeps conds), reuse
#          step_1_2's compatibility contract (same dim/scale/blur/noise), then
#          set requires_grad_(True) on the experts.
# CONVENTION: incompatible experts / missing ckpts -> logger.error + raise. No
#          fallback/mock. NOTE: build_from_report returns frozen modules; we
#          EXPLICITLY unfreeze here -- that is the whole point of 2.3 vs 1.3.
# Changelog (NEW in v0.1):
#   * load_experts_trainable(ckpt_dirs, device, train=True) -> (experts, ref_cfg).
#     Reuses build_from_report per dir (keeps cond inside CBExpert), asserts the
#     step_1_2 _MUST_MATCH compatibility set, unfreezes when train=True.
#   * build_gate(y_in, k, hidden, tau) -> LearnedGlobalGate (soft, global).
# Update summary:
#   v0.1 is the loader for 2.3-A: K trainable CB experts (conds inside) + a soft
#   global gate. Glow is never in the 2.3 roster (enforced by Stage23Cfg).
# =============================================================================
from __future__ import annotations
import logging

import torch

from ..step_1_1_1_1.model_io import build_from_report
from ..step_1_2.model_io import _MUST_MATCH
from ..step_1_2.mixture import LearnedGlobalGate

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-2_3"


def load_experts_trainable(ckpt_dirs, device: torch.device, *, train: bool = True):
    """Load K CB experts for joint training. Returns (experts, ref_cfg).

    Each expert is a CBExpert with its conditioner INSIDE (expert.cond(y)); we
    keep the whole module so the consistency term can compute h_k = expert.cond(y).
    build_from_report returns FROZEN modules -- we unfreeze when train=True (the
    defining difference of Stage 2.3: experts update, not just the gate).
    Compatibility (same dim/scale/blur/noise/data_root) is enforced exactly as
    step_1_2.load_experts does, so per-expert log_probs stay comparable."""
    if len(ckpt_dirs) < 2:
        logger.error("[s2_3.model_io] need >=2 ckpt_dirs, got %d", len(ckpt_dirs))
        raise ValueError("need >=2 ckpt_dirs for a mixture")
    experts, cfgs = [], []
    for d in ckpt_dirs:
        expert, _cond, cfg = build_from_report(d, device)   # _cond is INSIDE expert
        experts.append(expert)
        cfgs.append(cfg)
    ref = cfgs[0]
    for d, cfg in zip(ckpt_dirs[1:], cfgs[1:]):
        for key in _MUST_MATCH:
            if getattr(cfg, key) != getattr(ref, key):
                logger.error("[s2_3.model_io] expert %s disagrees on %s: %s != %s",
                             d, key, getattr(cfg, key), getattr(ref, key))
                raise ValueError(
                    f"expert {d} {key}={getattr(cfg, key)} != ref={getattr(ref, key)}")
    if train:
        for m in experts:
            m.train()
            for p in m.parameters():
                p.requires_grad_(True)
        n_train = sum(p.numel() for m in experts for p in m.parameters()
                      if p.requires_grad)
        logger.info("[s2_3.model_io] UNFROZE %d experts (%s) -- %d trainable params",
                    len(experts), [c.expert for c in cfgs], n_train)
    else:
        for m in experts:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
    return experts, ref


def build_gate(y_in: int, k: int, hidden: int, tau: float,
               device: torch.device) -> LearnedGlobalGate:
    """Soft global gate: one K-vector per image from y only (log_softmax(MLP(y)/tau)).
    Uniform-init-ish at tau>=1; argmax is DIAGNOSTIC only -- training uses the
    soft weights, never a hard winner (see ROUTING sub-plan)."""
    gate = LearnedGlobalGate(y_in=y_in, k=k, hidden=hidden, tau=tau).to(device)
    gate.train()
    return gate
