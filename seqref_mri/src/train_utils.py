# SEQREF-TRNUTIL v0.2 -- train_utils
# LIFETIME: KEEP
# Shared helpers: seed map (0/1/2->2026/7/8), cfg hash, run-dir, json, sha256,
# logger. No fallback/mock/pass. Failures logger.error + raise.
# Changelog (v0.1 -> v0.2, SEQREF-I3): make_run_dir renamed to the MRI
#   cell -- {expert}_a{accel}x_cf{center_fraction}_seed{idx}_{hash};
#   test0 -> test0_fullmask_{hash}. MNIST scale/noise params removed.
# Changelog (v0.1):
#   * seed_from_index, cfg_hash, make_run_dir, write_json, sha256_file,
#     setup_logger.
from __future__ import annotations
import hashlib
import json
import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger("seqref_mri.train_utils")
__version__ = "0.2"

_SEED_BASE = 2026
_VALID_IDX = (0, 1, 2)


def setup_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s :: %(message)s"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg


def seed_from_index(seed_index: int) -> int:
    # 0/1/2 -> 2026/2027/2028. Sets torch/numpy/random. Returns rng_seed.
    if seed_index not in _VALID_IDX:
        logger.error("[seed] seed_index must be in %s, got %s", _VALID_IDX, seed_index)
        raise ValueError(f"seed_index must be in {_VALID_IDX}, got {seed_index}")
    rng_seed = _SEED_BASE + seed_index
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    return rng_seed


def cfg_hash(cfg: dict) -> str:
    # 12-hex SHA256 of the canonical-JSON config.
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def make_run_dir(root: str, *, expert: str, accel: int,
                 center_fraction: float, seed_index: int,
                 cfg_hash_hex: str, test0: bool) -> str:
    # MRI cell naming (v0.2): acceleration + centre fraction, no MNIST params.
    if test0:
        name = f"test0_fullmask_{cfg_hash_hex}"
    else:
        name = (f"{expert}_a{accel}x_cf{center_fraction:.2f}"
                f"_seed{seed_index}_{cfg_hash_hex}")
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def sha256_file(path: str) -> str:
    if not os.path.isfile(path):
        logger.error("[sha256] file not found: %s", path)
        raise FileNotFoundError(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
