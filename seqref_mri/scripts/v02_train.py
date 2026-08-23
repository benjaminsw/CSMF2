# SEQREF-V02T v0.3 -- scripts.v02_train
# LIFETIME: KEEP
# =============================================================================
# Purpose: candidate v0.2 scientific training driver (V02SPEC v0.1 §3/§12,
#          V02PLAN v0.2 §4/§9). ONE clean run from a pinned commit; no
#          early stopping, no checkpoint selection, no mid-run restart.
# Locked quantities (quoted, never CLI-tunable):
#   * 3 epochs x 1,086 batches = 3,258 steps; batch 32; final partial
#     batch of 22 processed, never dropped
#   * Adam lr = 1e-4 on the FULL production parameter set
#   * checkpoints at steps 0 / 1086 / 2172 / 3258 (step-0 = initial
#     state BEFORE any optimizer step)
#   * abort = ERROR (exit 2): non-finite NLL or gradients, batch-manifest
#     drift, checkpoint state-hash mismatch on reload, wall-clock beyond
#     the 48 h ceiling mid-run
# Data contract: frozen epoch manifests from SEQREF-V02M (hash-verified
#   before use; the manifest file hash must equal its .sha256 sidecar).
#   Dataset = FastMRISliceDataset(split="train", mode="train") -- the
#   full-train production path (P4 precedent); train-mode masks belong
#   to this stage (SEQREF-IMPLT stage boundary). Per-realisation
#   coordinate maps are RE-DERIVED from the applied batch mask via
#   dec.build_coordinate_map (the same mandatory re-derivation rule as
#   ffr.verify_binding_identity); recorded P3 bindings exist only for
#   the frozen 256 corpus and are NOT required here -- the structural
#   invariants (24 acquired columns, centre 44..51 acquired, n_free
#   6912 / dim 13824) gate instead.
# RNG: model init via ffr.MODEL_INIT_SEED (registered); DataLoader
#   shuffle=False, num_workers=0; the driver draws nothing randomly.
# CONVENTION: logger.error + typed raise (V02Error, single identity from
#   SEQREF-V02M). No fallback, no mock, no placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * Introduced under V02PLAN v0.2 (LOCKED 2026-08-21).
# v0.2 (bug fix, reviewer blocker 1, 2026-08-22): main() gained the
#   registered unexpected-exception boundary (logger.exception + exit 2);
#   no exception may escape Python as exit 1 (frozen 0/2 taxonomy).
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from seqref_mri.src import free_flow_runtime as ffr
from seqref_mri.src import residual_decoder as dec
from seqref_mri.src.fastmri_data import FastMRISliceDataset
from seqref_mri.scripts.train_base import _collate, _prepare
from seqref_mri.scripts.train_free_flow import (targets_from_prepared,
                                                train_step)
from seqref_mri.scripts.v02_manifests import (BATCH_SIZE, N_EPOCHS,
                                              N_TRAIN_SLICES, V02Error,
                                              canonical_json)

logger = logging.getLogger("seqref_mri.v02_train")

__version__ = "0.3"
__abbr__ = "SEQREF-V02T"

LEARNING_RATE = 1e-4                     # locked (V02SPEC §3)
TOTAL_STEPS = 3258                       # 3 x ceil(34742/32), locked
CHECKPOINT_STEPS = (0, 1086, 2172, 3258)  # locked
WALL_CLOCK_CEILING_S = 48 * 3600         # 48 h, training only (§7)
REQUIRED_PREPARE_KEYS = ("y", "x_norm", "cond_in", "tgt_norm", "amax",
                         "ops")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V02Error(f"{code}: {message}")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_epoch_manifest(manifest_dir: str, epoch: int) -> dict:
    """Load one frozen epoch manifest; the file's SHA-256 must equal its
    .sha256 sidecar (manifest-drift = ERROR, V02SPEC §12)."""
    path = Path(manifest_dir) / f"v02_epoch{epoch}_manifest.json"
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not sidecar.exists():
        _fail("MANIFEST_MISSING", f"{path} or its sidecar does not exist; "
              f"manifests must be materialized before the run")
    recorded = sidecar.read_text().strip()
    actual = _file_sha256(path)
    if actual != recorded:
        _fail("MANIFEST_HASH_MISMATCH",
              f"{path.name}: file sha256 {actual[:12]}… != sidecar "
              f"{recorded[:12]}…; batch-manifest drift is an abort")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "seqref-v02-manifest/1" \
            or manifest.get("kind") != "train_epoch":
        _fail("MANIFEST_SCHEMA_INVALID",
              f"{path.name}: unexpected schema/kind "
              f"{manifest.get('schema')!r}/{manifest.get('kind')!r}")
    return manifest


def state_sha256(model: torch.nn.Module) -> str:
    """Canonical state hash: sha256 over sorted-name parameter bytes
    (float32, C-order). Insertion-order independent (TDIAG T6 rule)."""
    h = hashlib.sha256()
    for name in sorted(model.state_dict()):
        t = model.state_dict()[name].detach().cpu()
        h.update(name.encode("utf-8"))
        h.update(np.ascontiguousarray(t.numpy(), dtype="<f4").tobytes())
    return h.hexdigest()


def save_checkpoint(model: torch.nn.Module, step: int, out_root: Path,
                    log: list) -> dict:
    ckpt = out_root / f"v02_ckpt_step{step}.pt"
    torch.save({"model": model.state_dict(), "step": int(step),
                "abbr": __abbr__, "version": __version__}, ckpt)
    rec = {"step": int(step), "file": ckpt.name,
           "state_sha256": state_sha256(model),
           "file_sha256": _file_sha256(ckpt)}
    log.append(rec)
    logger.info("[%s] checkpoint step=%d state=%s", __abbr__, step,
                rec["state_sha256"][:12])
    return rec


def check_gradients_finite(model: torch.nn.Module, step: int) -> None:
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            _fail("GRADIENT_NON_FINITE",
                  f"step {step}: gradient of {name} is non-finite; abort "
                  f"(V02SPEC §12)")


def derive_cmap_from_mask(mask: torch.Tensor):
    """Mandatory re-derivation of the per-realisation coordinate map from
    the applied batch mask, with the registered structural invariants as
    ERROR gates (no recorded binding exists for full-train slices)."""
    cols = [int(c) for c in np.flatnonzero(
        mask.to(torch.bool).cpu().numpy()).tolist()]
    if len(cols) != ffr.EXPECTED_ACQUIRED_COLUMNS:
        _fail("MASK_ACQUIRED_COUNT_UNEXPECTED",
              f"applied mask has {len(cols)} acquired columns; the count "
              f"is fixed at {ffr.EXPECTED_ACQUIRED_COLUMNS}")
    if not ffr.CENTRE_COLUMNS.issubset(cols):
        _fail("MASK_CENTRE_NOT_ACQUIRED",
              "applied mask lacks centre columns 44..51")
    cmap = dec.build_coordinate_map(cols, ffr.GRID_H, ffr.GRID_W)
    if cmap.n_free_complex != ffr.N_FREE_COMPLEX \
            or cmap.flow_dim_real != ffr.FLOW_DIM_REAL:
        _fail("MAP_DIMENSION_MISMATCH",
              f"re-derived map dims {cmap.n_free_complex}/"
              f"{cmap.flow_dim_real} != registered "
              f"{ffr.N_FREE_COMPLEX}/{ffr.FLOW_DIM_REAL}")
    return cmap


def run(cfg: dict) -> dict:
    """The scientific run. cfg keys: data_root, manifest_dir, p4_stats2,
    implb_facts, out_root. Architecture/training constants are locked in
    this module and SEQREF-IMPLR, never in cfg."""
    t0 = time.time()
    out_root = Path(cfg["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    manifests = [load_epoch_manifest(cfg["manifest_dir"], ep)
                 for ep in range(N_EPOCHS)]
    for ep, man in enumerate(manifests):
        if int(man["epoch"]) != ep or int(man["n_slices"]) != N_TRAIN_SLICES:
            _fail("MANIFEST_CONTENT_MISMATCH",
                  f"epoch {ep}: manifest declares epoch="
                  f"{man.get('epoch')}, n_slices={man.get('n_slices')}")

    p4 = ffr.load_p4s2_parent(cfg["p4_stats2"])
    implb = ffr.load_implb_parent(cfg["implb_facts"])
    ffr.require_spline_b(implb["spline_b"])
    logger.info("[%s] parents pinned: P4/2 %s | IMPL-B %s (B=%.17g)",
                __abbr__, p4["file_sha256"][:12],
                implb["file_sha256"][:12], implb["spline_b"])

    model = ffr.build_model(spline_b=implb["spline_b"])
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ds = FastMRISliceDataset(cfg["data_root"], split="train", mode="train")
    index_of = {}
    for k, (path, sl) in enumerate(ds.index):
        index_of[(path.relative_to(ds.data_root).as_posix(), int(sl))] = k

    checkpoints: list[dict] = []
    step = 0
    save_checkpoint(model, step, out_root, checkpoints)   # step 0 = init

    vec_cache: dict[str, tuple] = {}
    telemetry = []
    for ep, man in enumerate(manifests):
        # Fresh-mask policy (EXEC SS3.7): declare the manifest epoch
        # before this epoch's DataLoader samples; regression proof
        # SEQREF-V02S f11.
        ds.set_epoch(ep)
        order = []
        for e in man["entries"]:
            key = (e["file"], int(e["slice_index"]))
            if key not in index_of:
                _fail("MANIFEST_ENTRY_UNKNOWN",
                      f"manifest entry {key} is not in the dataset "
                      f"traversal index; batch-manifest drift")
            order.append(int(index_of[key]))
        loader = DataLoader(Subset(ds, order), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0,
                            collate_fn=_collate)
        model.train()
        ep_nll = []
        for bi, batch in enumerate(loader):
            got = len(batch["meta"])
            lo, hi = man["batches"][bi]
            if got != hi - lo:
                _fail("BATCH_MANIFEST_DRIFT",
                      f"epoch {ep} batch {bi}: loader yielded {got} "
                      f"slices, manifest window declares {hi - lo}")
            prep = _prepare(batch, "cpu", test0=False)
            missing = [k for k in REQUIRED_PREPARE_KEYS if k not in prep]
            if missing:
                _fail("PREPARE_KEYS_MISSING",
                      f"_prepare result lacks {missing}")
            targets, cond_ins, masks = [], [], []
            for j, meta in enumerate(batch["meta"]):
                manifest_entry = man["entries"][lo + j]
                if Path(str(meta["file"])).name != Path(
                        manifest_entry["file"]).name \
                        or int(meta["slice_index"]) != int(
                            manifest_entry["slice_index"]):
                    _fail("BATCH_MANIFEST_DRIFT",
                          f"epoch {ep} batch {bi} position {j}: live "
                          f"sample ({meta['file']!r}, "
                          f"{meta['slice_index']!r}) != manifest entry "
                          f"({manifest_entry['file']!r}, "
                          f"{manifest_entry['slice_index']!r})")
                cmap = derive_cmap_from_mask(batch["mask"][j])
                map_key = cmap.payload()["map_payload_sha256"]
                if map_key not in vec_cache:
                    vec_cache[map_key] = (
                        cmap, ffr.standardisation_vectors(
                            cmap, p4["location_index"]))
                cmap, vecs = vec_cache[map_key]
                one = {"x_norm": prep["x_norm"][j:j + 1],
                       "cond_in": prep["cond_in"][j:j + 1]}
                targets.append(targets_from_prepared(one, cmap, vecs))
                cond_ins.append(prep["cond_in"][j:j + 1])
                masks.append(batch["mask"][j:j + 1])
            step += 1
            ep_nll.append(train_step(model, opt, torch.cat(targets, 0),
                                     torch.cat(cond_ins, 0),
                                     torch.cat(masks, 0)))
            check_gradients_finite(model, step)
            if step in CHECKPOINT_STEPS:
                save_checkpoint(model, step, out_root, checkpoints)
            if time.time() - t0 > WALL_CLOCK_CEILING_S:
                _fail("WALL_CLOCK_EXCEEDED",
                      f"step {step}: training exceeded the 48 h ceiling "
                      f"mid-run; abort (V02SPEC §12)")
        telemetry.append({"epoch": ep,
                          "train_nll_mean": float(np.mean(ep_nll)),
                          "train_nll_last": float(ep_nll[-1])})
        logger.info("[%s] epoch %d done, nll_mean=%.4f", __abbr__, ep,
                    telemetry[-1]["train_nll_mean"])

    if step != TOTAL_STEPS:
        _fail("STEP_COUNT_MISMATCH",
              f"run completed {step} steps; the locked budget is "
              f"{TOTAL_STEPS}")
    if [c["step"] for c in checkpoints] != list(CHECKPOINT_STEPS):
        _fail("CHECKPOINT_SCHEDULE_MISMATCH",
              f"checkpoints at {[c['step'] for c in checkpoints]}; the "
              f"locked schedule is {list(CHECKPOINT_STEPS)}")

    record = {"script": f"{__abbr__} v{__version__}",
              "schema": "seqref-v02-train-record/1",
              "steps": step, "epochs": N_EPOCHS, "batch": BATCH_SIZE,
              "lr": LEARNING_RATE,
              "manifest_sha256": [m["manifest_sha256"]
                                  for m in manifests],
              "checkpoints": checkpoints, "telemetry": telemetry,
              "wall_clock_s": time.time() - t0}
    rec_path = out_root / "v02_train_record.json"
    rec_path.write_bytes(canonical_json(record))
    sha = _file_sha256(rec_path)
    rec_path.with_suffix(".json.sha256").write_text(sha + "\n")
    logger.info("[%s] run complete: %d steps, record %s", __abbr__, step,
                sha[:12])
    return record


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"{__abbr__} v{__version__} -- candidate v0.2 "
                    f"scientific training driver (V02SPEC §3/§12)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--p4-stats2", required=True)
    ap.add_argument("--implb-facts", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()
    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode="w"))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s "
                               "%(message)s")
    try:
        run(vars(args))
    except V02Error:
        return 2
    except Exception:  # noqa: BLE001 -- the registered boundary: no
        logger.exception("[%s] unexpected runtime failure", __abbr__)
        return 2                # exception may escape as exit 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
