# =============================================================================
# STEP-1_1_1_1 v0.2 -- experiments.step_1_1_1_1.map_refine
# Purpose: core MAP refinement function. Frozen flow weights; optimise z only.
# CONVENTION: every failure -> logger.error + raise. No fallback / placeholder.
# Changelog (v0.1.1 -> v0.2):
#   * NEW init mode "is_random": sample n_candidates z0's, score each by
#     ||A(decode(z0,h)) - y||^2, pick the best PER IMAGE, refine from there.
#   * refine() now returns log["selection"] dict containing:
#       - best_k          : (B,) int  -- which candidate idx won per image
#       - res_init_best   : (B,) float -- the winning candidate's residual
#       - res_init_worst  : (B,) float -- worst candidate's residual
#       - res_init_mean   : (B,) float -- mean over all K candidates
#     For init in {random, encoded}, selection is None.
# Changelog (NEW in v0.1):
#   * refine(): Adam on z. Forward operator A_fn applied to decode(z, h);
#     loss = ||A(x_hat) - y||^2  +  lambda_prior * ||z||^2 / D.
#   * Pixel reconstruction goes through inverse_logit (NO clamp inside loop)
#     so gradients stay alive at pixel saturation.
#   * Two init modes:
#         "random"  : z ~ N(0,1)            (realistic inference)
#         "encoded" : z = encode(x_true, h) (eval-only ceiling check)
#   * Snapshots x_hat at 4 timesteps for the film-strip plot.
#   * Logs residual, prior, total loss, ||z|| trajectory.
# Update summary:
#   v0.2 adds importance-sampled init as a third option. Same MAP loop
#   downstream; only the starting z changes.
# =============================================================================
from __future__ import annotations
import logging
import torch

logger = logging.getLogger(__name__)
__version__ = "0.2"
__abbr__ = "STEP-1_1_1_1"


def refine(*, expert, cond, y, A_fn, inverse_logit, dequantize_logit,
           steps: int, lr: float, lambda_prior: float, init: str,
           n_candidates: int = 1,
           x_true: torch.Tensor | None = None,
           device: torch.device,
           gen: torch.Generator):
    """MAP refinement of latent z.

    Returns:
        x_star      : (B, 1, 28, 28)  refined reconstruction in pixel space
        z_star      : (B, D)          refined latent (detached)
        log         : dict with trajectories + snapshots + selection info
    """
    if y.dim() != 4 or y.size(1) != 1:
        logger.error("[refine] y must be (B,1,H,W), got %s", tuple(y.shape))
        raise ValueError(f"y must be (B,1,H,W), got {tuple(y.shape)}")
    if steps < 1:
        logger.error("[refine] steps must be >=1, got %d", steps)
        raise ValueError(f"steps must be >=1, got {steps}")
    if n_candidates < 1:
        logger.error("[refine] n_candidates must be >=1, got %d", n_candidates)
        raise ValueError(f"n_candidates must be >=1, got {n_candidates}")
    # Paranoia: ensure frozen. expert/cond should already be eval() but
    # confirm parameters do not have grad.
    for p in expert.parameters():
        p.requires_grad_(False)
    for p in cond.parameters():
        p.requires_grad_(False)

    B = y.size(0)
    with torch.no_grad():
        h = cond(y)            # (B, h_dim)

    selection = None    # populated only for init=is_random

    # Init z
    if init == "random":
        if n_candidates != 1:
            logger.error("[refine] init='random' requires n_candidates=1, "
                         "got %d", n_candidates)
            raise ValueError(
                f"init='random' requires n_candidates=1, got {n_candidates}")
        z = torch.randn(B, expert.dim, device=device, generator=gen,
                        dtype=h.dtype).requires_grad_(True)
    elif init == "is_random":
        # Sample K candidates, score each, pick the best per image.
        z_all = []                                   # list of (B, D)
        res_all = []                                 # list of (B,)
        for k in range(n_candidates):
            z_cand = torch.randn(B, expert.dim, device=device,
                                 generator=gen, dtype=h.dtype)
            with torch.no_grad():
                x_logit_c = expert.decode(z_cand, h)
                x_img_c   = inverse_logit(x_logit_c).view(B, 1, 28, 28)
                Ax_c      = A_fn(x_img_c)
                if Ax_c.shape != y.shape:
                    logger.error("[refine/is] A(x_hat) shape %s != y shape "
                                 "%s on candidate k=%d",
                                 tuple(Ax_c.shape), tuple(y.shape), k)
                    raise ValueError(
                        f"forward operator shape mismatch on candidate {k}")
                r_c = (Ax_c - y).flatten(1).pow(2).mean(dim=1)   # (B,)
                if not torch.isfinite(r_c).all():
                    logger.error("[refine/is] non-finite residual on "
                                 "candidate k=%d", k)
                    raise RuntimeError(
                        f"non-finite candidate residual at k={k}")
            z_all.append(z_cand)
            res_all.append(r_c)
        z_stack    = torch.stack(z_all,   dim=0)        # (K, B, D)
        res_stack  = torch.stack(res_all, dim=0)        # (K, B)
        best_k     = res_stack.argmin(dim=0)            # (B,)
        # Per-image gather: pick row best_k[b] from z_stack[:, b, :]
        z = z_stack[best_k, torch.arange(B, device=device), :].clone(
            ).requires_grad_(True)
        # Selection log (everything detached, CPU-friendly)
        res_init_best  = res_stack.gather(
            0, best_k.unsqueeze(0)).squeeze(0)          # (B,)
        res_init_worst = res_stack.max(dim=0).values    # (B,)
        res_init_mean  = res_stack.mean(dim=0)          # (B,)
        selection = {
            "n_candidates":  int(n_candidates),
            "best_k":        best_k.detach().cpu().tolist(),
            "res_init_best": res_init_best.detach().cpu().tolist(),
            "res_init_worst": res_init_worst.detach().cpu().tolist(),
            "res_init_mean": res_init_mean.detach().cpu().tolist(),
            "res_all":       res_stack.detach().cpu().tolist(),  # (K, B)
        }
    elif init == "encoded":
        if n_candidates != 1:
            logger.error("[refine] init='encoded' requires n_candidates=1, "
                         "got %d", n_candidates)
            raise ValueError(
                f"init='encoded' requires n_candidates=1, got {n_candidates}")
        if x_true is None:
            logger.error("[refine] encoded init requires x_true; got None")
            raise ValueError("encoded init requires x_true")
        if x_true.shape[-2:] != (28, 28):
            logger.error("[refine] x_true expected (B,1,28,28), got %s",
                         tuple(x_true.shape))
            raise ValueError(
                f"x_true expected (B,1,28,28), got {tuple(x_true.shape)}")
        with torch.no_grad():
            x_logit, _ = dequantize_logit(x_true, generator=gen)
            z_enc, _   = expert.encode(x_logit.flatten(1), h)
        z = z_enc.detach().clone().requires_grad_(True)
    else:
        logger.error("[refine] init must be in {random,is_random,encoded}, "
                     "got %r", init)
        raise ValueError(
            f"init must be in {{random,is_random,encoded}}, got {init!r}")

    opt = torch.optim.Adam([z], lr=lr)

    # snapshot timesteps for film strip (always evenly spaced incl. ends)
    if steps >= 4:
        snap = {0, max(1, steps // 4), steps // 2, steps - 1}
    else:
        snap = set(range(steps))
    log = {"residual": [], "prior": [], "loss": [],
           "z_norm": [], "x_snapshots": [],
           "selection": selection}     # None for non-IS modes

    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        x_logit = expert.decode(z, h)                      # logit space
        x_img   = inverse_logit(x_logit).view(B, 1, 28, 28)  # pixel space
        # NO clamp here: gradients must remain alive.
        Ax = A_fn(x_img)
        if Ax.shape != y.shape:
            logger.error("[refine] A(x_hat) shape %s != y shape %s",
                         tuple(Ax.shape), tuple(y.shape))
            raise ValueError(
                f"forward operator output shape {tuple(Ax.shape)} != "
                f"y shape {tuple(y.shape)}")
        residual = ((Ax - y) ** 2).mean()
        prior    = (z ** 2).mean()      # ||z||^2 / D
        loss     = residual + lambda_prior * prior
        if not torch.isfinite(loss):
            logger.error("[refine] non-finite at step=%d residual=%s "
                         "prior=%s loss=%s",
                         t, residual.item(), prior.item(), loss.item())
            raise RuntimeError("non-finite MAP loss")
        loss.backward()
        opt.step()
        log["residual"].append(float(residual.item()))
        log["prior"].append(float(prior.item()))
        log["loss"].append(float(loss.item()))
        log["z_norm"].append(
            float(z.detach().norm(dim=-1).mean().item()))
        if t in snap:
            log["x_snapshots"].append({
                "step": t,
                "img":  x_img.detach().clamp(0.0, 1.0).cpu(),  # display only
            })

    # Final reconstruction (display-clamped)
    with torch.no_grad():
        x_logit_final = expert.decode(z, h)
        x_star = inverse_logit(x_logit_final).view(B, 1, 28, 28).clamp(
            0.0, 1.0)
    z_star = z.detach()
    return x_star, z_star, log


def degrade_diff(x_img: torch.Tensor, *, sigma: float, scale: int,
                 blur_fn, downsample_fn) -> torch.Tensor:
    """Differentiable, deterministic forward operator A(x_img).

    Same as data.degrade.degrade but WITHOUT the trailing clamp(0,1) and
    WITHOUT noise injection -- both kill / inject randomness into the MAP
    gradient. Use this inside the inner optimisation loop.

    x_img: (B, 1, 28, 28). Returns y_hat: (B, 1, 28/scale, 28/scale).
    """
    if x_img.dim() != 4 or x_img.size(1) != 1:
        logger.error("[degrade_diff] x_img must be (B,1,H,W), got %s",
                     tuple(x_img.shape))
        raise ValueError(
            f"x_img must be (B,1,H,W), got {tuple(x_img.shape)}")
    return downsample_fn(blur_fn(x_img, sigma), scale)
