# SEQREF-BASEIO v0.2 -- refiners.base_io
# LIFETIME: KEEP
# Purpose: load + FREEZE the chosen base expert (NSF) from a train_base run
#          dir, produce x0 (posterior pixel-mean, f64 decode), r0 = y - A(x0),
#          and Aᵀr0, and build [y_up, x0, Aᵀr0] refiner inputs. Provides a
#          deterministic, sha-validated disk cache so the expensive f64 NSF
#          decode runs ONCE per (base, split, n_post) not once per epoch.
# CONVENTION: base weights requires_grad=False + eval(); decode in float64
#             (stacked quadratic-root inversions accumulate float32 error);
#             outputs returned in float32. No fallback/mock/pass -- all
#             failure paths logger.error + raise. Cache is validated against
#             base checkpoint sha256 + n_post + rng seed; mismatch -> raise
#             (never silently recompute over a stale run).
# Changelog (v0.1 -> v0.2, SEQREF-REFINE2):
#   * NEW FrozenStage1: rebuild + FREEZE a trained CplRegRefiner from its
#     train_refiner run dir (config.yaml refiner block + checkpoint.pt);
#     serves x1 = clamp(x0 + g*dx, 0, 1) -- the clamp matches stage-1's
#     reported metrics. NEW precompute_stage2(): transforms the already-
#     cached base split into stage-2 tensors (x1, [y_up, x1, Aᵀr1]) without
#     re-running the f64 base decode; cache keyed on BASE sha + STAGE-1 sha,
#     meta mismatch raises.
#   * v0.1.1 bug-fix folded in: A_adjoint call passes image_hw=(28,28).
# Changelog (v0.1):
#   * FrozenBase (load/rebuild from run_dir config.yaml + checkpoint.pt,
#     freeze, f64 posterior_pixel_mean), refiner_inputs(), precompute_split()
#     with sha-keyed .pt cache under results/refiners/_cache/.
from __future__ import annotations
import logging
import os

import torch
import torch.nn.functional as F
import yaml

from ..conditioner import Conditioner
from ..base_experts import build_expert
from ..degrade import inverse_logit
from ..forward_operator import A_forward, A_adjoint
from ..train_utils import sha256_file
from .coupling_regressor import CplRegRefiner

logger = logging.getLogger("fashion_seqref.refiners.base_io")
__version__ = "0.2"
__abbr__ = "SEQREF-BASEIO"

_FILM_KEYS = ("film_hidden", "film_depth", "film_use_gelu")


class FrozenBase:
    # Rebuilds the base expert from its run dir, loads the best checkpoint,
    # freezes it, and serves f64 posterior-pixel-mean reconstructions.
    def __init__(self, run_dir: str, device: str):
        cfg_path = os.path.join(run_dir, "config.yaml")
        ckpt_path = os.path.join(run_dir, "checkpoint.pt")
        for p in (cfg_path, ckpt_path):
            if not os.path.isfile(p):
                logger.error("[base_io] missing %s", p)
                raise FileNotFoundError(p)
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        self.expert = self.cfg["expert"]
        self.device = device
        self.checkpoint_path = ckpt_path
        self.checkpoint_sha256 = sha256_file(ckpt_path)
        hash_file = os.path.join(run_dir, "config_hash.txt")
        if not os.path.isfile(hash_file):
            logger.error("[base_io] missing %s", hash_file)
            raise FileNotFoundError(hash_file)
        with open(hash_file) as f:
            self.cfg_hash = f.read().strip()

        m = self.cfg["model"]
        cell = self.cfg["cell"]
        self.scale = int(cell["scale"])
        self.blur_sigma = float(cell["blur_sigma"])
        ccr = self.cfg.get("ccr", {})
        cond_kwargs = dict(width=int(m.get("cond_width", 64)),
                           h_dim=int(m.get("h_dim", 128)),
                           use_v2=bool(m.get("cond_use_v2", False)))
        alpha0 = float(ccr.get("cond_y_residual_alpha_init", 0.0))
        if alpha0 > 0.0:
            cond_kwargs["y_residual_alpha_init"] = alpha0
            cond_kwargs["y_input_size"] = (28 // self.scale) * (28 // self.scale)
        cond = Conditioner(**cond_kwargs)
        kw = {}
        if "n_layers" in m:
            kw["n_layers"] = int(m["n_layers"])
        if self.expert != "nsf":
            for k in _FILM_KEYS:
                if k in m:
                    kw[k] = m[k]
        self.model = build_expert(self.expert, dim=int(m["dim"]),
                                  h_dim=int(m.get("h_dim", 128)),
                                  conditioner=cond,
                                  hidden=int(m.get("hidden", 256)),
                                  use_film=bool(m.get("use_film", True)), **kw)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        # FREEZE + f64 decode path.
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.double()
        self.model.to(device)
        n_frozen = sum(1 for p in self.model.parameters() if not p.requires_grad)
        n_total = sum(1 for _ in self.model.parameters())
        if n_frozen != n_total:
            logger.error("[base_io] freeze failed: %d/%d frozen", n_frozen, n_total)
            raise RuntimeError("base freeze failed")
        logger.info("[base_io] frozen %s base loaded: %s (sha %s...)",
                    self.expert, run_dir, self.checkpoint_sha256[:12])

    def grad_max_abs(self) -> float:
        vals = [p.grad.abs().max().item() for p in self.model.parameters()
                if p.grad is not None]
        return max(vals) if vals else 0.0

    @torch.no_grad()
    def x0(self, y: torch.Tensor, n_post: int, gen: torch.Generator) -> torch.Tensor:
        # y: (B,1,h,w) float32. f64 posterior pixel-mean -> float32 (B,1,28,28).
        y64 = y.to(self.device, torch.float64)
        B = y64.size(0)
        h = self.model.cond(y64)
        acc = torch.zeros(B, 1, 28, 28, device=self.device, dtype=torch.float64)
        for _ in range(n_post):
            z = torch.randn(B, self.model.dim, generator=gen,
                            device=self.device, dtype=torch.float64)
            x_logit = self.model.decode(z, h)      # raises on failure (no skip)
            acc += inverse_logit(x_logit).view(B, 1, 28, 28).clamp(0, 1)
        x0 = (acc / n_post).float()
        if not torch.isfinite(x0).all():
            logger.error("[base_io] non-finite x0")
            raise ValueError("non-finite x0")
        return x0


def refiner_inputs(y: torch.Tensor, x0: torch.Tensor, blur_sigma: float,
                   scale: int) -> torch.Tensor:
    # Returns (B,3,28,28) = [y_up, x0, Aᵀr0], float32. r0 = y - A(x0).
    if y.dim() != 4 or x0.dim() != 4 or x0.shape[-2:] != (28, 28):
        logger.error("[base_io.refiner_inputs] bad shapes y=%s x0=%s",
                     tuple(y.shape), tuple(x0.shape))
        raise ValueError("refiner_inputs bad shapes")
    r0 = y - A_forward(x0, blur_sigma, scale)
    atr0 = A_adjoint(r0, blur_sigma, scale, (28, 28))
    if atr0.shape != x0.shape:
        logger.error("[base_io.refiner_inputs] Aᵀr0 %s != x0 %s",
                     tuple(atr0.shape), tuple(x0.shape))
        raise ValueError("adjoint shape mismatch")
    y_up = F.interpolate(y, size=(28, 28), mode="nearest")
    return torch.cat([y_up, x0, atr0], dim=1)


@torch.no_grad()
def precompute_split(base: FrozenBase, loader, *, n_post: int, rng_seed: int,
                     cache_dir: str, split_name: str, device: str):
    # One pass: (x_true, y, x0, inputs) tensors for the whole split, cached to
    # disk keyed by base sha + n_post + seed. Cache meta mismatch -> raise.
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"{split_name}_np{n_post}_s{rng_seed}_{base.checkpoint_sha256[:12]}"
    path = os.path.join(cache_dir, f"x0cache_{tag}.pt")
    meta = {"base_sha256": base.checkpoint_sha256, "n_post": n_post,
            "rng_seed": rng_seed, "split": split_name}
    if os.path.isfile(path):
        blob = torch.load(path, map_location="cpu")
        if blob.get("meta") != meta:
            logger.error("[base_io.cache] meta mismatch at %s: %s != %s",
                         path, blob.get("meta"), meta)
            raise RuntimeError("x0 cache meta mismatch -- delete stale cache")
        logger.info("[base_io.cache] loaded %s (%d samples)", path,
                    blob["x_true"].size(0))
        return blob["x_true"], blob["y"], blob["x0"], blob["inputs"]
    gen = torch.Generator(device=device).manual_seed(rng_seed)
    xs, ys, x0s, ins = [], [], [], []
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x0 = base.x0(y, n_post, gen)
        inp = refiner_inputs(y, x0, base.blur_sigma, base.scale)
        xs.append(x.cpu()); ys.append(y.cpu()); x0s.append(x0.cpu())
        ins.append(inp.cpu())
        if i % 50 == 0:
            logger.info("[base_io.cache] %s batch %d", split_name, i)
    x_true = torch.cat(xs); y_all = torch.cat(ys)
    x0_all = torch.cat(x0s); in_all = torch.cat(ins)
    torch.save({"meta": meta, "x_true": x_true, "y": y_all,
                "x0": x0_all, "inputs": in_all}, path)
    logger.info("[base_io.cache] wrote %s (%d samples)", path, x_true.size(0))
    return x_true, y_all, x0_all, in_all


class FrozenStage1:
    # Rebuilds a trained candidate refiner from its train_refiner run dir,
    # loads its checkpoint, freezes it. Serves x1 (clamped) for stage-2.
    def __init__(self, run_dir: str, device: str):
        cfg_path = os.path.join(run_dir, "config.yaml")
        ckpt_path = os.path.join(run_dir, "checkpoint.pt")
        for p in (cfg_path, ckpt_path):
            if not os.path.isfile(p):
                logger.error("[base_io.stage1] missing %s", p)
                raise FileNotFoundError(p)
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        r = cfg["refiner"]
        self.flavor = r["flavor"]
        self.device = device
        self.checkpoint_path = ckpt_path
        self.checkpoint_sha256 = sha256_file(ckpt_path)
        self.model = CplRegRefiner(
            flavor=r["flavor"], dim=int(r.get("dim", 784)),
            h_dim=int(r.get("h_dim", 256)), hidden=int(r.get("hidden", 256)),
            n_layers=r.get("n_layers"), cond_width=int(r.get("cond_width", 128)),
            film_hidden=int(r.get("film_hidden", 128)),
            film_depth=int(r.get("film_depth", 2)),
            film_use_gelu=bool(r.get("film_use_gelu", True)),
            s_max=float(r.get("s_max", 4.0)),
            post_init_std=float(r.get("post_init_std", 1e-3)),
            g_max=float(r.get("g_max", 0.5)), g_init=float(r.get("g_init", 0.05)))
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location="cpu")["model"])
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(device)
        if any(p.requires_grad for p in self.model.parameters()):
            logger.error("[base_io.stage1] freeze failed")
            raise RuntimeError("stage1 freeze failed")
        logger.info("[base_io.stage1] frozen %s stage-1 loaded: %s (sha %s...)",
                    self.flavor, run_dir, self.checkpoint_sha256[:12])

    def grad_max_abs(self) -> float:
        vals = [p.grad.abs().max().item() for p in self.model.parameters()
                if p.grad is not None]
        return max(vals) if vals else 0.0

    @torch.no_grad()
    def x1(self, inputs0: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        out, _, _ = self.model(inputs0.to(self.device), x0.to(self.device))
        return out.clamp(0.0, 1.0)


@torch.no_grad()
def precompute_stage2(stage1: FrozenStage1, base: FrozenBase, x_true, y, x0,
                      inputs0, *, batch_size: int, cache_dir: str,
                      split_name: str):
    # Stage-2 tensors from a cached base split: x1 = stage1(inputs0, x0),
    # inputs1 = [y_up, x1, Aᵀr1]. Cache keyed on base sha + stage-1 sha.
    os.makedirs(cache_dir, exist_ok=True)
    tag = (f"{split_name}_b{base.checkpoint_sha256[:12]}"
           f"_s1{stage1.checkpoint_sha256[:12]}")
    path = os.path.join(cache_dir, f"x1cache_{tag}.pt")
    meta = {"base_sha256": base.checkpoint_sha256,
            "stage1_sha256": stage1.checkpoint_sha256, "split": split_name}
    if os.path.isfile(path):
        blob = torch.load(path, map_location="cpu")
        if blob.get("meta") != meta:
            logger.error("[base_io.stage2cache] meta mismatch at %s", path)
            raise RuntimeError("x1 cache meta mismatch -- delete stale cache")
        logger.info("[base_io.stage2cache] loaded %s", path)
        return blob["x1"], blob["inputs1"]
    x1s, ins = [], []
    for i in range(0, x0.size(0), batch_size):
        sl = slice(i, i + batch_size)
        x1b = stage1.x1(inputs0[sl], x0[sl])
        inp1 = refiner_inputs(y[sl].to(stage1.device), x1b,
                              base.blur_sigma, base.scale)
        x1s.append(x1b.cpu()); ins.append(inp1.cpu())
    x1_all = torch.cat(x1s); in1_all = torch.cat(ins)
    torch.save({"meta": meta, "x1": x1_all, "inputs1": in1_all}, path)
    logger.info("[base_io.stage2cache] wrote %s (%d samples)", path,
                x1_all.size(0))
    return x1_all, in1_all
