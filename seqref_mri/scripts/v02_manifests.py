# SEQREF-V02M v0.2 -- scripts.v02_manifests
# LIFETIME: KEEP
# =============================================================================
# Purpose: candidate v0.2 data/manifest construction (V02SPEC v0.1 §3,
#          V02PLAN v0.2 §3). Owns every data-enumeration draw for the
#          scientific run; all manifests are materialized and SHA-256
#          hashed BEFORE the preflight, and both preflight and the
#          scientific run consume the frozen files -- nothing re-draws.
# Locked quantities (V02SPEC §3, quoted not redefined):
#   * train pool: 973 official train volumes / 34,742 eligible slices
#   * holdout: official single-coil val split, 199 volumes; one slice
#     per volume via the D2c midpoint rule (n even -> n/2 - 1, the lower
#     of the two middle slices; n odd -> (n-1)/2, the unique middle)
#   * 3 epochs, sampling WITHOUT replacement within each epoch; ONE
#     deterministic PCG64(0) stream, permutations drawn epoch 0,1,2 in
#     sequence; batch 32; final partial batch (22 slices) kept, never
#     dropped; repetition cap 3 (satisfied exactly by construction)
#   * D3 monitor subset: np.random.Generator(np.random.PCG64(2))
#     .choice(199, 32, replace=False) over the canonical ordered
#     199-volume holdout list, draw order preserved (V02SPEC §8)
# Dependency note: the dataset binding (FastMRISliceDataset) is imported
#   lazily inside build_all so the pure construction logic above stays
#   importable without the data environment. The lazy import is TOTAL:
#   ImportError propagates as ERROR; there is no fallback path.
# CONVENTION: logger.error + typed raise (V02Error). No fallback, no
#   mock, no placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * Introduced under V02PLAN v0.2 (LOCKED 2026-08-21).
# v0.2 (bug fix, reviewer blocker, 2026-08-22): main() had NO
#   exception boundary -- even a typed V02Error escaped as exit 1.
#   Now returns 0/2 with the registered unexpected-exception boundary
#   (logger.exception + exit 2), matching V02T/V02P/V02E/V02S.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("seqref_mri.v02_manifests")

__version__ = "0.2"
__abbr__ = "SEQREF-V02M"

SCHEMA = "seqref-v02-manifest/1"

# Locked population counts (V02SPEC §3). A mismatch is a data-premises
# ERROR, never a reason to proceed with a different population.
N_TRAIN_VOLUMES = 973
N_TRAIN_SLICES = 34742
N_HOLDOUT_VOLUMES = 199
N_EPOCHS = 3
BATCH_SIZE = 32
EXPECTED_BATCHES_PER_EPOCH = 1086        # ceil(34742 / 32)
EXPECTED_FINAL_BATCH = 22                # 34742 - 1085*32
EPOCH_SHUFFLE_SEED = 0                   # PCG64(0), one stream, in order
D3_SEED = 2                              # PCG64(2), V02SPEC §8
D3_SUBSET_N = 32
REPETITION_CAP = 3                       # == N_EPOCHS


class V02Error(RuntimeError):
    """Typed candidate-v0.2 failure. Shared single identity: every other
    v02 module imports THIS class, never redefines it."""


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V02Error(f"{code}: {message}")


# ---------------------------------------------------------------------------
# Pure construction logic (no dataset, no I/O; selftest exercises these)
# ---------------------------------------------------------------------------

def epoch_orders(n_slices: int, n_epochs: int = N_EPOCHS,
                 seed: int = EPOCH_SHUFFLE_SEED) -> list[np.ndarray]:
    """Per-epoch global slice permutations from ONE PCG64(seed) stream,
    consumed epoch 0,1,2 in sequence (V02SPEC §3). Without replacement
    within each epoch by construction (a permutation)."""
    if n_slices <= 0 or n_epochs <= 0:
        _fail("POPULATION_INVALID",
              f"n_slices={n_slices}, n_epochs={n_epochs}; both must be "
              f"positive")
    rng = np.random.Generator(np.random.PCG64(seed))
    return [rng.permutation(n_slices) for _ in range(n_epochs)]


def batch_partition(n_slices: int,
                    batch: int = BATCH_SIZE) -> list[tuple[int, int]]:
    """[start, end) batch windows over one epoch order; the final partial
    batch is kept, never dropped (V02SPEC §3)."""
    if n_slices <= 0 or batch <= 0:
        _fail("PARTITION_INVALID",
              f"n_slices={n_slices}, batch={batch}; both must be positive")
    return [(s, min(s + batch, n_slices)) for s in range(0, n_slices, batch)]


def midpoint_slice(n_slices_in_volume: int) -> int:
    """D2c midpoint rule: n even -> n/2 - 1 (lower of the two middle
    slices); n odd -> (n-1)/2 (the unique middle)."""
    n = n_slices_in_volume
    if not isinstance(n, (int, np.integer)) or n <= 0:
        _fail("MIDPOINT_INVALID", f"volume slice count {n!r} is not a "
              f"positive integer")
    return n // 2 - 1 if n % 2 == 0 else (n - 1) // 2


def d3_monitor_positions(n_holdout: int = N_HOLDOUT_VOLUMES,
                         k: int = D3_SUBSET_N,
                         seed: int = D3_SEED) -> np.ndarray:
    """The frozen D3 monitor draw (V02SPEC §8): PCG64(2).choice over the
    canonical ordered holdout-volume list, replace=False, draw order
    preserved (no sorting)."""
    if n_holdout <= 0 or k <= 0 or k > n_holdout:
        _fail("D3_DRAW_INVALID",
              f"n_holdout={n_holdout}, k={k}; need 0 < k <= n_holdout")
    return np.random.Generator(np.random.PCG64(seed)).choice(
        n_holdout, k, replace=False)


def exposure_counts(epoch_entries: list[list[int]],
                    n_slices: int) -> np.ndarray:
    """Per-slice exposure over the epoch manifests. Full pool each epoch
    makes the count exactly N_EPOCHS for every slice; anything else is a
    construction defect and raises."""
    counts = np.zeros(n_slices, dtype=np.int64)
    for ep, entries in enumerate(epoch_entries):
        arr = np.asarray(entries, dtype=np.int64)
        if arr.ndim != 1 or arr.size != n_slices:
            _fail("MANIFEST_SHAPE_INVALID",
                  f"epoch {ep} has {arr.size} entries; expected exactly "
                  f"{n_slices}")
        if np.unique(arr).size != n_slices:
            _fail("MANIFEST_NOT_A_PERMUTATION",
                  f"epoch {ep} repeats or omits slices within the epoch; "
                  f"sampling is without replacement")
        counts += np.bincount(arr, minlength=n_slices)
    return counts


def canonical_json(manifest: dict) -> bytes:
    """Deterministic serialisation: sorted keys, fixed separators, UTF-8.
    The hash below is over THESE bytes, so the hash is a property of the
    manifest content, not of dict ordering accidents."""
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def manifest_sha256(manifest: dict) -> str:
    return hashlib.sha256(canonical_json(manifest)).hexdigest()


# ---------------------------------------------------------------------------
# Dataset-bound construction (user-side data environment)
# ---------------------------------------------------------------------------

def _dataset(data_root: str, split: str, mode: str):
    """Total lazy import of the environment-bound dataset class."""
    try:
        from seqref_mri.src.fastmri_data import FastMRISliceDataset
    except ImportError as exc:
        _fail("DATASET_IMPORT_FAILED",
              f"FastMRISliceDataset is not importable in this environment: "
              f"{exc}")
    return FastMRISliceDataset(data_root, split=split, mode=mode)


def _index_entries(ds) -> list[dict]:
    """Canonical traversal entries [{file, slice_index, dataset_index}].
    Sample identity binds to the dataset's own traversal index, never to
    a loop counter (SEQREF-IMPLT precedent)."""
    entries = []
    for k, (path, sl) in enumerate(ds.index):
        entries.append({
            "file": path.relative_to(ds.data_root).as_posix(),
            "slice_index": int(sl),
            "dataset_index": int(k)})
    return entries


def _group_by_volume(entries: list[dict]) -> dict[str, list[dict]]:
    """Canonical ordered volume list: first-appearance order in the
    traversal; slices within a volume sorted by slice_index."""
    volumes: dict[str, list[dict]] = {}
    for e in entries:
        volumes.setdefault(e["file"], []).append(e)
    for f in volumes:
        volumes[f].sort(key=lambda e: e["slice_index"])
    return volumes


def build_all(data_root: str, out_dir: str) -> dict:
    """Build, write and hash all v0.2 manifests. Returns their sha256
    map. ERROR on any population mismatch -- the locked counts are data
    premises (LOCK 2)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds_train = _dataset(data_root, "train", "train")
    train_entries = _index_entries(ds_train)
    train_volumes = _group_by_volume(train_entries)
    if len(train_volumes) != N_TRAIN_VOLUMES:
        _fail("POPULATION_MISMATCH",
              f"train volumes {len(train_volumes)} != locked "
              f"{N_TRAIN_VOLUMES}")
    if len(train_entries) != N_TRAIN_SLICES:
        _fail("POPULATION_MISMATCH",
              f"train slices {len(train_entries)} != locked "
              f"{N_TRAIN_SLICES}")

    orders = epoch_orders(N_TRAIN_SLICES)
    counts = exposure_counts([o.tolist() for o in orders], N_TRAIN_SLICES)
    if not np.all(counts == REPETITION_CAP):
        bad = int(np.sum(counts != REPETITION_CAP))
        _fail("EXPOSURE_CAP_VIOLATION",
              f"{bad} slices deviate from exactly {REPETITION_CAP} "
              f"exposures; construction defect")

    shas: dict[str, str] = {}
    for ep, order in enumerate(orders):
        entries = [train_entries[int(i)] for i in order]
        batches = batch_partition(N_TRAIN_SLICES)
        if len(batches) != EXPECTED_BATCHES_PER_EPOCH:
            _fail("PARTITION_MISMATCH",
                  f"epoch {ep}: {len(batches)} batches != locked "
                  f"{EXPECTED_BATCHES_PER_EPOCH}")
        if batches[-1][1] - batches[-1][0] != EXPECTED_FINAL_BATCH:
            _fail("PARTITION_MISMATCH",
                  f"epoch {ep}: final batch size "
                  f"{batches[-1][1] - batches[-1][0]} != locked "
                  f"{EXPECTED_FINAL_BATCH}")
        manifest = {
            "schema": SCHEMA, "kind": "train_epoch", "epoch": ep,
            "generator": {"rule": "one PCG64(0) stream; permutation per "
                                  "epoch, drawn in epoch order",
                          "generator": "PCG64", "seed": EPOCH_SHUFFLE_SEED},
            "n_slices": N_TRAIN_SLICES, "batch": BATCH_SIZE,
            "batches": [[s, e] for s, e in batches],
            "entries": entries}
        sha = manifest_sha256(manifest)
        manifest["manifest_sha256"] = sha
        name = f"v02_epoch{ep}_manifest.json"
        (out / name).write_bytes(canonical_json(manifest))
        (out / (name + ".sha256")).write_text(sha + "\n")
        shas[f"epoch{ep}"] = sha
        logger.info("[%s] %s written, sha256 %s", __abbr__, name, sha[:12])

    # Holdout: canonical ordered 199-volume list + midpoint slice each.
    ds_val = _dataset(data_root, "val", "eval")
    val_entries = _index_entries(ds_val)
    val_volumes = _group_by_volume(val_entries)
    if len(val_volumes) != N_HOLDOUT_VOLUMES:
        _fail("POPULATION_MISMATCH",
              f"holdout volumes {len(val_volumes)} != locked "
              f"{N_HOLDOUT_VOLUMES}")
    holdout_entries = []
    for ordinal, (fname, slices) in enumerate(val_volumes.items()):
        pick = midpoint_slice(len(slices))
        e = dict(slices[pick])
        e["volume_ordinal"] = ordinal
        holdout_entries.append(e)
    holdout_manifest = {
        "schema": SCHEMA, "kind": "holdout",
        "selection": {"rule": "D2c midpoint per volume (n even -> "
                              "n/2 - 1; n odd -> (n-1)/2); canonical "
                              "ordered volume list = traversal "
                              "first-appearance order",
                      "split": "val", "mode": "eval"},
        "n_volumes": N_HOLDOUT_VOLUMES, "entries": holdout_entries}
    sha = manifest_sha256(holdout_manifest)
    holdout_manifest["manifest_sha256"] = sha
    name = "v02_holdout_manifest.json"
    (out / name).write_bytes(canonical_json(holdout_manifest))
    (out / (name + ".sha256")).write_text(sha + "\n")
    shas["holdout"] = sha
    logger.info("[%s] %s written, sha256 %s", __abbr__, name, sha[:12])

    # D3 monitor subset: frozen draw, order preserved.
    positions = d3_monitor_positions()
    d3_entries = []
    for rank, pos in enumerate(positions):
        e = dict(holdout_entries[int(pos)])
        e["draw_rank"] = rank
        e["holdout_position"] = int(pos)
        d3_entries.append(e)
    d3_manifest = {
        "schema": SCHEMA, "kind": "d3_monitor",
        "selection": {"rule": "np.random.Generator(np.random.PCG64(2))"
                              ".choice(199, 32, replace=False), draw "
                              "order preserved",
                      "generator": "PCG64", "seed": D3_SEED},
        "n": D3_SUBSET_N, "entries": d3_entries}
    sha = manifest_sha256(d3_manifest)
    d3_manifest["manifest_sha256"] = sha
    name = "v02_d3_monitor_manifest.json"
    (out / name).write_bytes(canonical_json(d3_manifest))
    (out / (name + ".sha256")).write_text(sha + "\n")
    shas["d3_monitor"] = sha
    logger.info("[%s] %s written, sha256 %s", __abbr__, name, sha[:12])
    return shas


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"{__abbr__} v{__version__} -- candidate v0.2 "
                    f"manifest builder (V02SPEC §3 / V02PLAN §3)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s "
                               "%(message)s")
    try:
        shas = build_all(args.data_root, args.out_dir)
    except V02Error:
        return 2
    except Exception:  # noqa: BLE001 -- the registered boundary: no
        logger.exception("[%s] unexpected runtime failure", __abbr__)
        return 2                # exception may escape as exit 1
    print(json.dumps(shas, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
