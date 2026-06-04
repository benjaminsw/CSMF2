# =============================================================================
# STEP-1_1_1_1 v0.3 -- experiments.step_1_1_1_1.map_refine
# Purpose: core MAP refinement function. Frozen flow weights; optimise z only.
# CONVENTION: every failure -> logger.error + raise. No fallback / placeholder.
# Changelog (v0.2 -> v0.3):
#   * Refactored into 3 functions:
#       _select_topS_z()    -- sample K cands, return top-S per image
#       _run_map_inner()    -- the existing Adam loop on a single start
#       refine()            -- orchestrator; dispatches to inits + multi-start
#   * NEW multi-start path: when init="is_random" and n_starts>1, run S
#     independent MAP loops from top-S candidates (sequential, memory-safe),
#     pick per-image winner by final_objective = residual + lambda_prior*prior
#     (NOT residual alone -- avoids z drift past the prior).
#   * refine() return signature unchanged externally: (x_star, z_star, log).
#     log["selection"] gains per_start_obj_final, best_s, winner_was_top1,
#     per_start_z_norm, wall_clock_per_start_s when S>1.
# Changelog (v0.1.1 -> v0.2):
#   * NEW init mode "is_random": sample n_candidates z0's, score each by
#     ||A(decode(z0,h)) - y||^2, pick the best PER IMAGE, refine from there.
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
#   v0.3 generalizes IS to multi-start. The same K-sample procedure produces
#   S starting points (instead of 1), each refined independently. Same MAP
#   inner loop reused S times sequentially.
# =============================================================================
from __future__ import annotations
import logging
import time
import torch

logger = logging.getLogger(__name__)
__version__ = "0.3"
__abbr__ = "STEP-1_1_1_1"


def _select_topS_z(*, expert, h, y, A_fn, inverse_logit,
                   n_candidates: int, n_starts: int,
                   device, gen):
    """Sample K candidates from N(0,1), score each by initial residual,
    return top-S (smallest residual) per image.

    Returns:
        top_z        : (S, B, D) starting z's, one per (start, image)
        selection    : dict with full bookkeeping (K residuals, top-S idx,
                       res statistics)
    """
    if n_starts > n_candidates:
        logger.error("[_select_topS_z] n_starts=%d > n_candidates=%d",
                     n_starts, n_candidates)
        raise ValueError(
            f"n_starts ({n_starts}) > n_candidates ({n_candidates})")
    B = y.size(0)
    z_all, res_all = [], []
    for k in range(n_candidates):
        z_cand = torch.randn(B, expert.dim, device=device,
                             generator=gen, dtype=h.dtype)
        with torch.no_grad():
            x_logit_c = expert.decode(z_cand, h)
            x_img_c   = inverse_logit(x_logit_c).view(B, 1, 28, 28)
            Ax_c      = A_fn(x_img_c)
            if Ax_c.shape != y.shape:
                logger.error("[_select_topS_z] A shape %s != y shape %s "
                             "on k=%d",
                             tuple(Ax_c.shape), tuple(y.shape), k)
                raise ValueError(
                    f"forward operator shape mismatch on candidate {k}")
            r_c = (Ax_c - y).flatten(1).pow(2).mean(dim=1)   # (B,)
            if not torch.isfinite(r_c).all():
                logger.error("[_select_topS_z] non-finite residual on k=%d",
                             k)
                raise RuntimeError(
                    f"non-finite candidate residual at k={k}")
        z_all.append(z_cand)
        res_all.append(r_c)
    z_stack   = torch.stack(z_all,   dim=0)        # (K, B, D)
    res_stack = torch.stack(res_all, dim=0)        # (K, B)
    # Per-image top-S smallest residuals
    topS_res, topS_idx = torch.topk(
        res_stack, k=n_starts, dim=0, largest=False)   # both (S, B)
    # Gather corresponding z's: top_z[s, b, :] = z_stack[topS_idx[s, b], b, :]
    batch_idx = torch.arange(B, device=device).unsqueeze(0).expand(
        n_starts, -1)                                  # (S, B)
    top_z = z_stack[topS_idx, batch_idx, :]            # (S, B, D)
    # Bookkeeping
    selection = {
        "n_candidates":  int(n_candidates),
        "n_starts":      int(n_starts),
        "topS_residuals": topS_res.detach().cpu().tolist(),    # (S, B)
        "topS_idx_in_K":  topS_idx.detach().cpu().tolist(),    # (S, B)
        # v0.2 backward-compat fields (S=1 case)
        "best_k":        topS_idx[0].detach().cpu().tolist(),  # (B,) -- top-1 idx
        "res_init_best": topS_res[0].detach().cpu().tolist(),  # (B,) -- best residual
        "res_init_worst": res_stack.max(dim=0).values
                                  .detach().cpu().tolist(),    # (B,)
        "res_init_mean": res_stack.mean(dim=0)
                                  .detach().cpu().tolist(),    # (B,)
        "res_all":       res_stack.detach().cpu().tolist(),    # (K, B)
    }
    return top_z, selection


def _run_map_inner(*, expert, h, y, z0, A_fn, inverse_logit,
                   steps: int, lr: float, lambda_prior: float,
                   collect_snapshots: bool,
                   device):
    """Run the Adam MAP loop on a single starting z0.

    Returns:
        x_star          : (B, 1, 28, 28)
        z_star          : (B, D) detached
        per_step_log    : dict with residual/prior/loss/z_norm trajectories
                          and optionally x_snapshots.
        per_image_final : dict with per-image final residual + prior + obj
                          + z_norm (all shape (B,) torch tensors, on device).
    """
    B = y.size(0)
    z = z0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)

    # Snapshot timesteps for film strip
    if collect_snapshots and steps >= 4:
        snap = {0, max(1, steps // 4), steps // 2, steps - 1}
    elif collect_snapshots:
        snap = set(range(steps))
    else:
        snap = set()

    per_step_log = {
        "residual": [], "prior": [], "loss": [],
        "z_norm":   [], "x_snapshots": []}

    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        x_logit = expert.decode(z, h)
        x_img   = inverse_logit(x_logit).view(B, 1, 28, 28)
        Ax = A_fn(x_img)
        if Ax.shape != y.shape:
            logger.error("[_run_map_inner] A(x_hat) shape %s != y shape %s",
                         tuple(Ax.shape), tuple(y.shape))
            raise ValueError(
                f"forward operator output shape {tuple(Ax.shape)} != "
                f"y shape {tuple(y.shape)}")
        residual = ((Ax - y) ** 2).mean()
        prior    = (z ** 2).mean()
        loss     = residual + lambda_prior * prior
        if not torch.isfinite(loss):
            logger.error("[_run_map_inner] non-finite at step=%d residual=%s "
                         "prior=%s loss=%s",
                         t, residual.item(), prior.item(), loss.item())
            raise RuntimeError("non-finite MAP loss")
        loss.backward()
        opt.step()
        per_step_log["residual"].append(float(residual.item()))
        per_step_log["prior"].append(float(prior.item()))
        per_step_log["loss"].append(float(loss.item()))
        per_step_log["z_norm"].append(
            float(z.detach().norm(dim=-1).mean().item()))
        if t in snap:
            per_step_log["x_snapshots"].append({
                "step": t,
                "img":  x_img.detach().clamp(0.0, 1.0).cpu(),
            })

    with torch.no_grad():
        x_logit_final = expert.decode(z, h)
        x_star_pixel  = inverse_logit(x_logit_final).view(B, 1, 28, 28)
        x_star        = x_star_pixel.clamp(0.0, 1.0)
        # Per-image final residual / prior / objective / z_norm
        Ax_final          = A_fn(x_star_pixel)
        r_per_img         = (Ax_final - y).flatten(1).pow(2).mean(dim=1)
        p_per_img         = z.pow(2).mean(dim=1)
        obj_per_img       = r_per_img + lambda_prior * p_per_img
        z_norm_per_img    = z.norm(dim=-1)
    z_star = z.detach()
    per_image_final = {
        "residual": r_per_img.detach(),
        "prior":    p_per_img.detach(),
        "objective": obj_per_img.detach(),
        "z_norm":   z_norm_per_img.detach(),
    }
    return x_star, z_star, per_step_log, per_image_final


def refine(*, expert, cond, y, A_fn, inverse_logit, dequantize_logit,
           steps: int, lr: float, lambda_prior: float, init: str,
           n_candidates: int = 1,
           n_starts: int = 1,
           x_true: torch.Tensor | None = None,
           device: torch.device,
           gen: torch.Generator):
    """MAP refinement of latent z.

    Returns:
        x_star      : (B, 1, 28, 28)  refined reconstruction in pixel space
        z_star      : (B, D)          refined latent (detached)
        log         : dict with trajectories + snapshots + selection info
    """
    # ---- arg validation ------------------------------------------------
    if y.dim() != 4 or y.size(1) != 1:
        logger.error("[refine] y must be (B,1,H,W), got %s", tuple(y.shape))
        raise ValueError(f"y must be (B,1,H,W), got {tuple(y.shape)}")
    if steps < 1:
        logger.error("[refine] steps must be >=1, got %d", steps)
        raise ValueError(f"steps must be >=1, got {steps}")
    if n_candidates < 1:
        logger.error("[refine] n_candidates must be >=1, got %d", n_candidates)
        raise ValueError(f"n_candidates must be >=1, got {n_candidates}")
    if n_starts < 1:
        logger.error("[refine] n_starts must be >=1, got %d", n_starts)
        raise ValueError(f"n_starts must be >=1, got {n_starts}")
    if init != "is_random" and n_starts != 1:
        logger.error("[refine] n_starts=%d requires init='is_random', got %r",
                     n_starts, init)
        raise ValueError(
            f"n_starts={n_starts} requires init='is_random', got {init!r}")

    # Paranoia: ensure flow weights frozen
    for p in expert.parameters():
        p.requires_grad_(False)
    for p in cond.parameters():
        p.requires_grad_(False)

    B = y.size(0)
    with torch.no_grad():
        h = cond(y)            # (B, h_dim)

    selection = None

    # ---- Single-start paths (v0.1.1, v0.2) -----------------------------
    if init == "random":
        if n_candidates != 1:
            logger.error("[refine] init='random' requires n_candidates=1, "
                         "got %d", n_candidates)
            raise ValueError(
                f"init='random' requires n_candidates=1, got {n_candidates}")
        z0 = torch.randn(B, expert.dim, device=device, generator=gen,
                         dtype=h.dtype)
        x_star, z_star, per_step_log, _ = _run_map_inner(
            expert=expert, h=h, y=y, z0=z0, A_fn=A_fn,
            inverse_logit=inverse_logit, steps=steps, lr=lr,
            lambda_prior=lambda_prior, collect_snapshots=True,
            device=device)
        per_step_log["selection"] = None
        return x_star, z_star, per_step_log

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
        z0 = z_enc.detach()
        x_star, z_star, per_step_log, _ = _run_map_inner(
            expert=expert, h=h, y=y, z0=z0, A_fn=A_fn,
            inverse_logit=inverse_logit, steps=steps, lr=lr,
            lambda_prior=lambda_prior, collect_snapshots=True,
            device=device)
        per_step_log["selection"] = None
        return x_star, z_star, per_step_log

    elif init == "is_random":
        # ---- IS init: get top-S candidates ---------------------------
        top_z, selection = _select_topS_z(
            expert=expert, h=h, y=y, A_fn=A_fn,
            inverse_logit=inverse_logit,
            n_candidates=n_candidates, n_starts=n_starts,
            device=device, gen=gen)
        # top_z shape: (S, B, D)

        # ---- v0.2 path (S=1): single start ----------------------------
        if n_starts == 1:
            z0 = top_z[0]                                   # (B, D)
            x_star, z_star, per_step_log, _ = _run_map_inner(
                expert=expert, h=h, y=y, z0=z0, A_fn=A_fn,
                inverse_logit=inverse_logit, steps=steps, lr=lr,
                lambda_prior=lambda_prior, collect_snapshots=True,
                device=device)
            per_step_log["selection"] = selection
            return x_star, z_star, per_step_log

        # ---- v0.3 path (S>1): multi-start MAP -------------------------
        # Sequential loop over S to bound memory. Each iteration runs the
        # full MAP loop on one start, collects final per-image objective.
        x_stars        = []     # list of (B,1,28,28)
        z_stars        = []     # list of (B,D)
        per_step_logs  = []     # list of dicts (one per start)
        per_image_objs = []     # list of (B,) tensors (final objective per start)
        per_image_zn   = []     # list of (B,) tensors (final z_norm per start)
        wall_per_start = []
        for s in range(n_starts):
            t0 = time.time()
            z0_s = top_z[s]                                # (B, D)
            # Only collect snapshots from start s=0 (representative; saves memory)
            x_star_s, z_star_s, per_step_log_s, per_img_s = _run_map_inner(
                expert=expert, h=h, y=y, z0=z0_s, A_fn=A_fn,
                inverse_logit=inverse_logit, steps=steps, lr=lr,
                lambda_prior=lambda_prior, collect_snapshots=(s == 0),
                device=device)
            wall_per_start.append(time.time() - t0)
            x_stars.append(x_star_s)
            z_stars.append(z_star_s)
            per_step_logs.append(per_step_log_s)
            per_image_objs.append(per_img_s["objective"])
            per_image_zn.append(per_img_s["z_norm"])

        # Per-image winner selection by FINAL OBJECTIVE (NOT residual alone)
        obj_stack = torch.stack(per_image_objs, dim=0)     # (S, B)
        zn_stack  = torch.stack(per_image_zn,  dim=0)      # (S, B)
        best_s    = obj_stack.argmin(dim=0)                # (B,)
        # Gather winner x_star and z_star per image
        x_stack = torch.stack(x_stars, dim=0)              # (S, B, 1, 28, 28)
        z_stack = torch.stack(z_stars, dim=0)              # (S, B, D)
        batch_idx = torch.arange(B, device=device)
        x_star_final = x_stack[best_s, batch_idx]          # (B, 1, 28, 28)
        z_star_final = z_stack[best_s, batch_idx]          # (B, D)

        # Top-1-initial was start 0 (smallest initial residual); compare
        winner_was_top1 = (best_s == 0).detach().cpu().tolist()  # (B,)

        # Use start-0's per-step log as the "headline" trajectory for plotting
        # (we don't have a winner-per-image trajectory; representative is fine)
        per_step_log_out = per_step_logs[0]

        # Extend selection with multi-start info
        selection.update({
            "per_start_obj_final":  obj_stack.detach().cpu().tolist(),   # (S, B)
            "per_start_z_norm":     zn_stack.detach().cpu().tolist(),    # (S, B)
            "best_s":               best_s.detach().cpu().tolist(),      # (B,)
            "winner_was_top1_initial": winner_was_top1,                  # (B,)
            "wall_clock_per_start_s": wall_per_start,                    # (S,)
            # x_top1_starts: the reconstruction from start s=0 (top-1 initial
            # residual). Used by run.py to compute psnr_gain per image.
            # Detached CPU tensor; memory is small.
            "_x_top1_starts": x_stars[0].detach().cpu(),                 # (B,1,28,28)
        })
        per_step_log_out["selection"] = selection
        return x_star_final, z_star_final, per_step_log_out

    else:
        logger.error("[refine] init must be in {random,is_random,encoded}, "
                     "got %r", init)
        raise ValueError(
            f"init must be in {{random,is_random,encoded}}, got {init!r}")


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
