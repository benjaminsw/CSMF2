# =============================================================================
# STEP-1_1 v0.4 -- data.degrade
# Purpose: deterministic MNIST degradation pipeline y = Ax + n, with
#          A = Downsample_s o Gauss_blur(sigma). Returns (x_clean, y_degraded).
# CONVENTION: NLL = LOSS (lower = better). No fallback / mock / pass.
#             Any failure -> logger.error(...) + raise.
# Changelog (v0.3 -> v0.4, SEQREF-FSEQ W1+W2):
#   * Dataset generalisation: shared _VisionDegraded base (all v0.3 logic,
#     parameterised by TV_DATASET); MNISTDegraded now subclasses it
#     (byte-identical behaviour); NEW FashionMNISTDegraded
#     (torchvision.datasets.FashionMNIST; same 60k/10k splits, same
#     SPLIT_SEED partition, identical interface).
#   * NEW make_degraded(dataset, root, ...) factory: dataset REQUIRED, one of
#     {"mnist","fashion_mnist"}; absent/unknown -> logger.error + raise (no
#     silent default). Training scripts construct datasets ONLY through this.
# Changelog (v0.2 -> v0.3):
#   * NEW (Test-0 / identity task): blur(sigma=0.0) is the IDENTITY (a
#     Gaussian with sigma->0 is a delta kernel). This is an explicit,
#     mathematically-correct path, NOT a silent fallback. sigma<0 still
#     raises. _gauss_kernel_2d stays strict (sigma<=0 raises) since it is a
#     pure kernel builder and must never be called at 0.
#   * NEW: downsample(scale=1) is the IDENTITY (avg_pool k=1,s=1). scale now
#     in {1,2,4}; scale=1 keeps y at native 28x28.
#   * Net effect: degrade(sigma=0, scale=1, noise_sigma=0) yields y == x,
#     the "y=x" identity task used to verify an expert can learn an EASY
#     conditional map before blaming the inverse problem.
# Changelog (v0.1 -> v0.2):
#   * NEW: train/val/test split. The 60k MNIST training set is partitioned
#     into 55k train + 5k val with a fixed seed (SPLIT_SEED=12345). The 10k
#     MNIST test set is untouched and used only for final reporting.
#   * MNISTDegraded REQUIRES `split: {"train","val","test"}`. The old
#     `train: bool` kwarg is REMOVED -- callers must pass split=... explicitly.
#     This is intentional: v0.1 callers that still passed train=True would
#     have silently switched from 60k to 55k. A hard break surfaces every
#     such site at import time.
#   * NEW split="train_legacy_60k": full 60k for strict v0.1 reproducibility
#     (escape hatch only; mainline runs should use split="train").
#   * Split is deterministic w.r.t. SPLIT_SEED so val/test sets do not leak
#     into training across runs / seeds.
# Changelog (NEW in v0.1):
#   * Introduced. Provides MNISTDegraded (torch Dataset), gaussian blur (fixed
#     kernel), area-average downsample, AWGN.
#   * Supports scale in {2, 4} and noise_sigma in {0.0, 0.05, 0.1}.
#   * Logit-dequantization helpers (for flows trained in logit space).
# Update summary:
#   v0.4 generalises the degraded dataset across torchvision sources: one
#   shared base class, MNIST unchanged, FashionMNIST added, and a mandatory
#   dataset-key factory so no script can silently train on the wrong data.
#   v0.3 adds the identity task (y=x) via sigma=0 (delta-blur) and scale=1
#   (no downsample). This is the Test-0 sanity check: if an expert cannot
#   learn y=x, the fault is the implementation, not the inverse problem.
#   v0.2 behaviour is byte-identical for sigma>0, scale in {2,4}.
# =============================================================================
from __future__ import annotations
import logging
import traceback
logger = logging.getLogger(__name__)
__version__ = "0.4"
__abbr__ = "STEP-1_1"

import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def _gauss_kernel_2d(sigma: float, ksize: int = 5) -> torch.Tensor:
    if sigma <= 0.0:
        logger.error("[degrade] sigma must be > 0, got %.4f", sigma)
        raise ValueError(f"sigma must be > 0, got {sigma}")
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    g1 = torch.exp(-(ax ** 2) / (2.0 * sigma * sigma))
    g1 = g1 / g1.sum()
    k2 = torch.outer(g1, g1)
    return k2.view(1, 1, ksize, ksize)


def blur(x: torch.Tensor, sigma: float, ksize: int = 5) -> torch.Tensor:
    # x: (B,1,H,W) in [0,1]. Returns (B,1,H,W) blurred.
    # sigma == 0.0 -> IDENTITY (delta kernel; explicit Test-0 path). sigma < 0
    # -> raise. sigma > 0 -> Gaussian blur.
    if x.dim() != 4 or x.size(1) != 1:
        logger.error("[degrade.blur] expected (B,1,H,W), got %s", tuple(x.shape))
        raise ValueError(f"expected (B,1,H,W), got {tuple(x.shape)}")
    if sigma < 0.0:
        logger.error("[degrade.blur] sigma must be >= 0, got %.4f", sigma)
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if sigma == 0.0:
        return x                                  # identity blur (delta)
    k = _gauss_kernel_2d(sigma, ksize).to(x.device, x.dtype)
    pad = ksize // 2
    return F.conv2d(F.pad(x, [pad] * 4, mode="reflect"), k)


def downsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    if scale not in (1, 2, 4):
        logger.error("[degrade.downsample] scale must be 1, 2 or 4, got %s",
                     scale)
        raise ValueError(f"scale must be in {{1, 2, 4}}, got {scale}")
    if scale == 1:
        return x                                  # identity (no downsample)
    return F.avg_pool2d(x, kernel_size=scale, stride=scale)


def degrade(x: torch.Tensor, *, sigma: float, scale: int, noise_sigma: float,
            generator: torch.Generator | None = None) -> torch.Tensor:
    # x: (B,1,28,28). Returns y: (B,1,28/scale,28/scale).
    y = downsample(blur(x, sigma), scale)
    if noise_sigma < 0.0:
        logger.error("[degrade] noise_sigma must be >= 0, got %.4f", noise_sigma)
        raise ValueError(f"noise_sigma must be >= 0, got {noise_sigma}")
    if noise_sigma > 0.0:
        eps = torch.randn(y.shape, generator=generator,
                          device=y.device, dtype=y.dtype) * noise_sigma
        y = y + eps
    return y.clamp(0.0, 1.0)


# ---------- dequantization helpers (train in logit space) --------------------
LOGIT_ALPHA = 0.05


def dequantize_logit(x: torch.Tensor,
                     generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    # x in [0,1]. Returns (x_logit, log|det d(x_logit)/dx|). Standard for flows.
    if (x.min() < 0.0) or (x.max() > 1.0):
        logger.error("[dequantize_logit] x out of [0,1]: [%.3f, %.3f]",
                     x.min().item(), x.max().item())
        raise ValueError("dequantize_logit expects x in [0,1]")
    u = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype) / 256.0
    xq = (x * 255.0 + u * 256.0) / 256.0          # dequantize
    xq = xq.clamp(1e-5, 1.0 - 1e-5)
    z = LOGIT_ALPHA + (1.0 - 2.0 * LOGIT_ALPHA) * xq
    logit = torch.log(z) - torch.log1p(-z)
    ldj = (torch.log(torch.tensor(1.0 - 2.0 * LOGIT_ALPHA, device=x.device)) -
           torch.log(z) - torch.log1p(-z)).flatten(1).sum(-1)
    return logit, ldj


def inverse_logit(logit: torch.Tensor) -> torch.Tensor:
    z = torch.sigmoid(logit)
    x = (z - LOGIT_ALPHA) / (1.0 - 2.0 * LOGIT_ALPHA)
    return x.clamp(0.0, 1.0)


# ---------- Dataset ----------------------------------------------------------
SPLIT_SEED: int = 12345         # frozen; do NOT change once runs exist
N_TRAIN: int = 55_000           # of 60k torchvision train split
N_VAL:   int = 5_000            # 60k - 55k = 5k held out
VALID_SPLITS = ("train", "val", "test", "train_legacy_60k")


def _build_split_indices(split: str) -> tuple[bool, list[int] | None]:
    # Returns (use_torchvision_train_set, index_subset_or_None).
    # train     -> (True,  first 55k of a fixed permutation of 60k)
    # val       -> (True,  remaining 5k)
    # test      -> (False, None)            torchvision test (10k, untouched)
    # train_legacy_60k -> (True, None)      v0.1-equivalent full 60k
    if split == "train_legacy_60k":
        return True, None
    if split == "test":
        return False, None
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(N_TRAIN + N_VAL, generator=g).tolist()
    if split == "train":
        return True, perm[:N_TRAIN]
    if split == "val":
        return True, perm[N_TRAIN:N_TRAIN + N_VAL]
    logger.error("[MNISTDegraded] unknown split %r", split)
    raise ValueError(f"unknown split {split!r}; valid: {VALID_SPLITS}")


class _VisionDegraded(Dataset):
    # Shared degraded-dataset logic; subclasses set TV_DATASET. The 55k/5k
    # SPLIT_SEED partition applies identically (both MNIST and FashionMNIST
    # ship 60k train / 10k test).
    TV_DATASET = None

    def __init__(self, root: str, *, split: str, sigma: float, scale: int,
                 noise_sigma: float, download: bool = True):
        cls = type(self).__name__
        if type(self).TV_DATASET is None:
            logger.error("[%s] TV_DATASET unset -- use a concrete subclass",
                         cls)
            raise TypeError("_VisionDegraded is abstract; use a subclass")
        if split not in VALID_SPLITS:
            logger.error("[%s] invalid split %r (valid: %s)", cls, split,
                         VALID_SPLITS)
            raise ValueError(f"invalid split {split!r}; valid: {VALID_SPLITS}")

        use_train_set, subset_idx = _build_split_indices(split)
        try:
            self.base = type(self).TV_DATASET(root=root, train=use_train_set,
                                              download=download,
                                              transform=transforms.ToTensor())
        except Exception:
            logger.error("[%s] failed to load dataset from %s\n%s", cls,
                         root, traceback.format_exc())
            raise
        self.split = split
        self.subset_idx = subset_idx
        self.sigma = sigma
        self.scale = scale
        self.noise_sigma = noise_sigma

    def __len__(self) -> int:
        return len(self.base) if self.subset_idx is None else len(self.subset_idx)

    def __getitem__(self, idx: int):
        real_idx = idx if self.subset_idx is None else self.subset_idx[idx]
        x, _ = self.base[real_idx]               # (1,28,28) in [0,1]
        x = x.unsqueeze(0)                       # (1,1,28,28)
        y = degrade(x, sigma=self.sigma, scale=self.scale,
                    noise_sigma=self.noise_sigma)
        return x.squeeze(0), y.squeeze(0)


class MNISTDegraded(_VisionDegraded):
    TV_DATASET = datasets.MNIST


class FashionMNISTDegraded(_VisionDegraded):
    TV_DATASET = datasets.FashionMNIST


DATASETS = {"mnist": MNISTDegraded, "fashion_mnist": FashionMNISTDegraded}


def make_degraded(dataset: str | None, root: str, *, split: str, sigma: float,
                  scale: int, noise_sigma: float,
                  download: bool = True) -> _VisionDegraded:
    # W2: dataset key REQUIRED; absent/unknown -> raise (no silent default).
    if dataset not in DATASETS:
        logger.error("[make_degraded] cell.dataset required, one of %s, "
                     "got %r", sorted(DATASETS), dataset)
        raise ValueError(f"cell.dataset required, one of "
                         f"{sorted(DATASETS)}, got {dataset!r}")
    return DATASETS[dataset](root, split=split, sigma=sigma, scale=scale,
                             noise_sigma=noise_sigma, download=download)
