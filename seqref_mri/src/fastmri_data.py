# =============================================================================
# SEQREF-I1 v0.2 -- src.fastmri_data
# LIFETIME: KEEP
# Purpose: fastMRI single-coil knee slice dataset for the seqref_mri campaign.
#   Implements the S3-locked conventions ONLY (EXEC 3.2/3.3/3.7/3.10/3.14):
#   construction A, centred orthonormal FFT pair, exact-count Cartesian mask
#   on the LAST axis (phase-encode columns), SHA-256 seed policy, complex
#   two-channel reconstruction state, ESC-enforced targets.
# CONVENTION: every failure path -> logger.error + raise. No fallback, no
#   mock, no silent pass.
# Item contract (one sample):
#   x_true     float32 (2, 96, 96)  Re/Im of the complex ground-truth state
#   y          complex64 (96, 96)   masked k-space observation M(F x_true)
#   mask       bool (96,)           sampled COLUMNS (last axis)
#   target_mag float32 (96, 96)     |x_true| -- magnitude target for metrics
#   meta       dict: file (relpath), slice_index, split, mode, mask_seed
#   NOTE: ESC is a MAGNITUDE target, never the complex state. The 96x96
#   magnitude target equals centre-crop of the dataset's own ESC by the
#   S3-proven convention; the I1 report verifies this agreement.
# Construction A (exact): full k-space -> centred orthonormal IFFT ->
#   image-centre-crop to 96x96 (this complex image IS x_true) -> centred
#   orthonormal FFT of x_true is the 96x96 k-space; y = M * F(x_true).
# Seed serialization (locked): tuple string "{seed}|{epoch}|{relpath}|{slice}"
#   (train) or "{seed}|{relpath}|{slice}" (eval); relpath = POSIX path
#   relative to data_root; epoch/slice as decimal integers; separator '|';
#   UTF-8 -> SHA-256 -> first 8 bytes big-endian -> 64-bit unsigned seed.
#   Epoch reaches DataLoader workers via set_epoch(): call it BEFORE creating
#   the epoch's iterator; with persistent_workers=False the dataset object is
#   re-pickled to workers each epoch, so the value propagates. Using
#   persistent_workers=True without a worker-side epoch update is an ERROR
#   and is the caller's responsibility to avoid.
# Changelog (v0.1 -> v0.2, pre-deployment review fixes):
#   * Split-dir resolver: accepts BOTH <root>/singlecoil_<split> and the
#     established extraction layout <root>/knee_singlecoil_<split>/
#     singlecoil_<split>; error lists every attempted path.
#   * NEW make_train_loader(): ENFORCED loader construction rule --
#     persistent_workers is forbidden for train-mode datasets (raises),
#     because set_epoch() cannot reach persistent workers; fresh train
#     masks per epoch are locked (EXEC 3.7), so this is a hard rule, not
#     documentation.
#   * set_epoch() ENFORCED: train-mode epoch initialises to None and any
#     train sampling before set_epoch() raises -- the fresh-mask policy is
#     enforceable, not advisory (a caller can no longer silently reuse
#     epoch-zero masks by forgetting the call).
# Changelog (NEW in v0.1):
#   * Introduced: canonical_mask_seed, make_cartesian_mask, fft2c/ifft2c,
#     center_crop, FastMRISliceDataset (train/eval modes, set_epoch).
# Update summary (v0.2): the two review-blocking dataset risks are closed
#   structurally -- the resolver matches the real on-disk layout, and the
#   persistent-workers epoch hazard is an enforced error instead of a
#   caller obligation.
# =============================================================================
from __future__ import annotations

import hashlib
import logging
from pathlib import Path, PurePosixPath

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("seqref_mri.fastmri_data")

__version__ = "0.2"
__abbr__ = "SEQREF-I1"

CELL_HW = 96                      # EXEC 3.1: 96x96 self-consistent problem
CENTER_FRACTION = 0.08            # EXEC 3.7: campaign-locked
ACCELERATION = 4                  # EXEC 3.4: one acceleration
TRAIN_BASE_SEED = 20261000        # EXEC 3.7
EVAL_BASE_SEED = 20261001         # EXEC 3.7


# ---- seed policy (EXEC 3.7) --------------------------------------------------
def canonical_mask_seed(base_seed: int, relpath: str, slice_index: int,
                        epoch: int | None = None) -> int:
    # Canonical tuple string; epoch present ONLY for train seeds.
    if not isinstance(base_seed, int) or not isinstance(slice_index, int):
        logger.error("[seed] base_seed/slice_index must be int, got %r/%r",
                     base_seed, slice_index)
        raise TypeError("base_seed and slice_index must be int")
    rel = str(PurePosixPath(relpath))
    if rel.startswith("/") or rel.startswith(".."):
        logger.error("[seed] relpath must be relative to data_root, got %r", rel)
        raise ValueError(f"relpath must be relative, got {rel!r}")
    if epoch is None:
        s = f"{base_seed}|{rel}|{slice_index}"
    else:
        if not isinstance(epoch, int) or epoch < 0:
            logger.error("[seed] epoch must be a non-negative int, got %r", epoch)
            raise ValueError(f"epoch must be a non-negative int, got {epoch!r}")
        s = f"{base_seed}|{epoch}|{rel}|{slice_index}"
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")   # 64-bit truncation, big-endian


# ---- mask (EXEC 3.7, exact counts, columns axis) -----------------------------
def mask_counts(width: int) -> tuple[int, int]:
    if width < 2:
        logger.error("[mask] width must be >= 2, got %d", width)
        raise ValueError(f"width must be >= 2, got {width}")
    n_center = max(1, round(CENTER_FRACTION * width))
    n_total = max(n_center, round(width / ACCELERATION))
    return n_center, n_total


def make_cartesian_mask(width: int, seed: int) -> np.ndarray:
    # Returns bool (width,) over COLUMNS (last numpy axis).
    n_center, n_total = mask_counts(width)
    mask = np.zeros(width, dtype=bool)
    start = (width - n_center) // 2
    mask[start:start + n_center] = True       # centred fully-sampled block
    n_rand = n_total - n_center
    if n_rand > 0:
        pool = np.flatnonzero(~mask)
        rng = np.random.Generator(np.random.PCG64(seed))
        chosen = rng.choice(pool, size=n_rand, replace=False)
        mask[chosen] = True
    got = int(mask.sum())
    if got != n_total:
        logger.error("[mask] count invariant violated: got %d expected %d "
                     "(width=%d)", got, n_total, width)
        raise RuntimeError(f"mask count {got} != n_total {n_total}")
    return mask


# ---- centred orthonormal FFT pair (S3-validated convention) ------------------
def fft2c(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(x):
        logger.error("[fft2c] expected complex input, got %s", x.dtype)
        raise TypeError(f"fft2c expects complex input, got {x.dtype}")
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1))


def ifft2c(k: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(k):
        logger.error("[ifft2c] expected complex input, got %s", k.dtype)
        raise TypeError(f"ifft2c expects complex input, got {k.dtype}")
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1))


def center_crop(x: torch.Tensor, hw: int) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    if h < hw or w < hw:
        logger.error("[crop] input %dx%d smaller than crop %d", h, w, hw)
        raise ValueError(f"cannot centre-crop {h}x{w} to {hw}")
    top = (h - hw) // 2
    left = (w - hw) // 2
    return x[..., top:top + hw, left:left + hw]


# ---- dataset -----------------------------------------------------------------
class FastMRISliceDataset(Dataset):
    # split: 'train' | 'val' (official fastMRI singlecoil dirs).
    # mode:  'train' -> fresh mask per slice per epoch (set_epoch REQUIRED
    #        each epoch); 'eval' -> deterministic mask per (file, slice).
    def __init__(self, data_root: str, *, split: str, mode: str):
        if split not in ("train", "val"):
            logger.error("[data] split must be 'train'|'val', got %r", split)
            raise ValueError(f"split must be 'train'|'val', got {split!r}")
        if mode not in ("train", "eval"):
            logger.error("[data] mode must be 'train'|'eval', got %r", mode)
            raise ValueError(f"mode must be 'train'|'eval', got {mode!r}")
        self.data_root = Path(data_root)
        self.split = split
        self.mode = mode
        self.epoch: int | None = None   # train mode: set_epoch() REQUIRED
                                        # before any sampling (enforced)
        candidates = [
            self.data_root / f"singlecoil_{split}",
            self.data_root / f"knee_singlecoil_{split}" / f"singlecoil_{split}",
        ]
        split_dir = next((c for c in candidates if c.is_dir()), None)
        if split_dir is None:
            logger.error("[data] split dir missing; tried: %s",
                         "; ".join(str(c) for c in candidates))
            raise FileNotFoundError(
                "split dir missing; tried: "
                + "; ".join(str(c) for c in candidates))
        files = sorted(split_dir.glob("*.h5"))
        if not files:
            logger.error("[data] no .h5 files under %s", split_dir)
            raise FileNotFoundError(f"no .h5 files under {split_dir}")
        self.index: list[tuple[Path, int]] = []
        for f in files:
            with h5py.File(f, "r") as h:
                if "kspace" not in h:
                    logger.error("[data] %s lacks 'kspace'", f.name)
                    raise KeyError(f"{f.name} lacks 'kspace'")
                if "reconstruction_esc" not in h:
                    # ESC required and enforced (EXEC 3.10; multi-coil guard)
                    logger.error("[data] %s lacks 'reconstruction_esc' -- "
                                 "wrong extraction or multi-coil data", f.name)
                    raise KeyError(f"{f.name} lacks 'reconstruction_esc'")
                n_slices = h["kspace"].shape[0]
            for s in range(n_slices):
                self.index.append((f, s))
        logger.info("[data] %s/%s: %d files, %d slices",
                    split, mode, len(files), len(self.index))

    def set_epoch(self, epoch: int) -> None:
        if self.mode != "train":
            logger.error("[data] set_epoch called on eval-mode dataset")
            raise RuntimeError("set_epoch is train-mode only")
        if not isinstance(epoch, int) or epoch < 0:
            logger.error("[data] set_epoch: bad epoch %r", epoch)
            raise ValueError(f"epoch must be a non-negative int, got {epoch!r}")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.index)

    def _mask_seed(self, path: Path, slice_index: int) -> int:
        rel = path.relative_to(self.data_root).as_posix()
        if self.mode == "train":
            if self.epoch is None:
                logger.error("[data] set_epoch() must be called before "
                             "train sampling (fresh-mask policy, EXEC 3.7)")
                raise RuntimeError("set_epoch() required before train sampling")
            return canonical_mask_seed(TRAIN_BASE_SEED, rel, slice_index,
                                       epoch=self.epoch)
        return canonical_mask_seed(EVAL_BASE_SEED, rel, slice_index)

    def __getitem__(self, i: int) -> dict:
        path, slice_index = self.index[i]
        with h5py.File(path, "r") as h:
            k_full = np.asarray(h["kspace"][slice_index])
        if k_full.dtype != np.complex64:
            logger.error("[data] %s slice %d kspace dtype %s != complex64",
                         path.name, slice_index, k_full.dtype)
            raise TypeError(f"kspace dtype {k_full.dtype} != complex64")
        k_t = torch.from_numpy(k_full)                    # (640, W) complex64
        # Construction A: full k-space -> centred orthonormal IFFT ->
        # image-centre-crop to 96x96 == the complex ground-truth state.
        x_full = ifft2c(k_t)
        x_true_c = center_crop(x_full, CELL_HW).contiguous()   # (96,96) c64
        k96 = fft2c(x_true_c)                                  # (96,96) c64
        seed = self._mask_seed(path, slice_index)
        mask_np = make_cartesian_mask(CELL_HW, seed)
        mask = torch.from_numpy(mask_np)                       # (96,) bool
        y = k96 * mask.to(k96.dtype).unsqueeze(0)              # zero unsampled
        x_true = torch.stack([x_true_c.real, x_true_c.imag])   # (2,96,96) f32
        target_mag = torch.abs(x_true_c)                       # (96,96) f32
        for name, t in (("x_true", x_true), ("target_mag", target_mag)):
            if not torch.isfinite(t).all():
                logger.error("[data] non-finite %s in %s slice %d",
                             name, path.name, slice_index)
                raise ValueError(f"non-finite {name}")
        return {
            "x_true": x_true,
            "y": y,
            "mask": mask,
            "target_mag": target_mag,
            "meta": {"file": path.relative_to(self.data_root).as_posix(),
                     "slice_index": slice_index, "split": self.split,
                     "mode": self.mode, "mask_seed": seed},
        }

    # ESC crop for competence checks (I1 report): centre-crop of the
    # dataset's OWN magnitude target; must agree with |x_true| by the
    # S3-proven convention. Not part of the training item (avoids a second
    # dataset read per sample).
    def esc_crop(self, i: int) -> torch.Tensor:
        path, slice_index = self.index[i]
        with h5py.File(path, "r") as h:
            esc = np.asarray(h["reconstruction_esc"][slice_index])
        if esc.dtype != np.float32:
            logger.error("[data] %s slice %d ESC dtype %s != float32",
                         path.name, slice_index, esc.dtype)
            raise TypeError(f"ESC dtype {esc.dtype} != float32")
        return center_crop(torch.from_numpy(esc), CELL_HW)


# ---- ENFORCED loader construction rule (v0.2) --------------------------------
def make_train_loader(dataset: FastMRISliceDataset, **loader_kwargs):
    # The ONLY sanctioned way to build a train-mode DataLoader.
    # persistent_workers=True is FORBIDDEN for train mode: persistent workers
    # hold a pickled copy of the dataset, so set_epoch() on the main-process
    # object never reaches them and epoch-0 masks would silently repeat --
    # violating the locked fresh-mask-per-epoch policy (EXEC 3.7).
    from torch.utils.data import DataLoader
    if dataset.mode != "train":
        logger.error("[loader] make_train_loader requires a train-mode "
                     "dataset, got mode=%r", dataset.mode)
        raise ValueError(f"train-mode dataset required, got {dataset.mode!r}")
    if loader_kwargs.get("persistent_workers", False):
        logger.error("[loader] persistent_workers=True is FORBIDDEN for "
                     "train mode (set_epoch cannot reach persistent workers)")
        raise ValueError("persistent_workers=True forbidden for train mode")
    loader_kwargs["persistent_workers"] = False
    return DataLoader(dataset, **loader_kwargs)
