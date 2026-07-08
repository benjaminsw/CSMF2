# SEQREF-TRNUTIL v0.1 -- train_utils
# LIFETIME: KEEP
# Shared helpers: seed map (0/1/2->2026/7/8), cfg hash, run-dir, json, sha256,
# logger. No fallback/mock/pass. Failures logger.error + raise.
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

logger = logging.getLogger("mnist_seqref.train_utils")
__version__ = "0.1"

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


def make_run_dir(root: str, *, expert: str, scale: int, noise_sigma: float,
                 seed_index: int, cfg_hash_hex: str, test0: bool) -> str:
    if test0:
        name = f"test0_identity_{cfg_hash_hex}"
    else:
        name = f"{expert}_s{scale}_n{noise_sigma:.2f}_seed{seed_index}_{cfg_hash_hex}"
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
