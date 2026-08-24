# SEQREF-V02P v0.5 -- scripts.v02_preflight
# LIFETIME: KEEP
# =============================================================================
# Purpose: candidate v0.2 deterministic throughput preflight (V02SPEC v0.1
#          §7, V02PLAN v0.2 §8). Runs on the ACTUAL production model and
#          data path, AFTER manifests are materialized+hashed, BEFORE the
#          scientific run. Preflight state is discarded and can never
#          seed the scientific run.
# Frozen protocol (V02SPEC §7, quoted not redefined):
#   * warm-up: 20 steps on batches 1-20 of the epoch-0 training manifest
#   * timed window: batches 21-120 of the same manifest (exactly 100
#     deterministic production batches -- nothing else counts)
#   * I/O + batch construction time and full optimizer-step time measured
#     SEPARATELY; peak device memory recorded
#   * projected 3,258-step wall-clock vs the 48 h ceiling (training
#     only); endpoint-evaluation cost projected separately, same block
#   * projection > ceiling or batch-32 infeasible => ERROR (exit 2)
#     BEFORE the scientific run; no silent downscale of epochs, batch,
#     or pool -- rescaling is a named amendment
# RNG/model isolation (V02PLAN §8 contract): the preflight constructs
#   its OWN model instance from the registered init seed and discards
#   it; it draws nothing from any scientific-run stream (the driver
#   holds no other stochastic state). The selftest proves that model
#   initialisation after a preflight-equivalent sequence is identical to
#   initialisation without it.
# LIFETIME rationale: the SOURCE MODULE is KEEP like all six V02PLAN §2
#   modules. The preflight STATE and the v02_preflight.json REPORT are
#   EPHEMERAL (V02SPEC §7; the report declares "lifetime": "EPHEMERAL"
#   in its schema block): projections are inherited into the
#   v02_facts.json preflight block (V02SPEC §9), which is the durable
#   record. (V02PLAN v0.2 §13 lists the report file as KEEP; V02SPEC
#   governs on conflict -- flagged for a V02PLAN housekeeping bump.)
# CONVENTION: logger.error + typed raise (V02Error). No fallback, no
#   mock, no placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * Introduced under V02PLAN v0.2 (LOCKED 2026-08-21).
# v0.2 (bug fix, reviewer blocker 2, 2026-08-22): main() gained the
#   registered unexpected-exception boundary (logger.exception + exit 2);
#   no exception may escape Python as exit 1 (frozen 0/2 taxonomy).
# v0.3 (bug fix, reviewer correction, 2026-08-22): script lifetime
#   corrected EPHEMERAL -> KEEP (all six source modules are KEEP);
#   EPHEMERAL applies to the preflight state and report only.
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from seqref_mri.src import free_flow_runtime as ffr
from seqref_mri.src.fastmri_data import FastMRISliceDataset
from seqref_mri.scripts.train_base import _collate, _prepare
from seqref_mri.scripts.train_free_flow import (targets_from_prepared,
                                                train_step)
from seqref_mri.scripts.v02_manifests import V02Error, canonical_json
from seqref_mri.scripts.v02_train import (BATCH_SIZE, CHECKPOINT_STEPS,
                                          LEARNING_RATE, TOTAL_STEPS,
                                          WALL_CLOCK_CEILING_S,
                                          derive_cmap_from_mask,
                                          load_epoch_manifest)

logger = logging.getLogger("seqref_mri.v02_preflight")

__version__ = "0.5"
__abbr__ = "SEQREF-V02P"

WARMUP_BATCHES = 20          # batches 1-20 of the epoch-0 manifest
TIMED_BATCHES = 100          # batches 21-120 of the epoch-0 manifest
EVAL_PROBE_BATCHES = 4       # eval-mode projection probe (batches 1-4)
N_TRAIN_SLICES = 34742
N_HOLDOUT_SLICES = 199
D3_SUBSET_N = 32
PM_BANK_N = 128


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V02Error(f"{code}: {message}")


def _peak_memory_bytes() -> int:
    if not torch.cuda.is_available():
        _fail("REGISTERED_HOST_MISSING",
              "CUDA is unavailable; the 48 h ceiling is defined on the "
              "registered CUDA training host, so a CPU preflight proves "
              "nothing and is ERROR, never a downscale")
    return int(torch.cuda.max_memory_allocated())


def _batch_work(model, opt, batch, p4, *, train_mode: bool,
                device: str) -> tuple[float, float]:
    """One production batch. Returns (io_construct_s, step_s): data
    preparation + target construction vs model forward/backward/step
    (train) or eval-mode forward (eval)."""
    t_io = time.perf_counter()
    prep = _prepare(batch, device, test0=False)
    targets, cond_ins, masks = [], [], []
    for j in range(len(batch["meta"])):
        # Fresh masks never repeat (EXEC SS3.7): no cache -- derive
        # and compute per slice, release with the batch; preflight
        # times the exact production path (OOM fix 2026-08-24).
        cmap = derive_cmap_from_mask(batch["mask"][j])
        vecs = ffr.standardisation_vectors(cmap, p4["location_index"])
        one = {"x_norm": prep["x_norm"][j:j + 1],
               "cond_in": prep["cond_in"][j:j + 1]}
        targets.append(targets_from_prepared(one, cmap, vecs))
        cond_ins.append(prep["cond_in"][j:j + 1])
        masks.append(batch["mask"][j:j + 1])
    u = torch.cat(targets, 0).to(device)
    c = torch.cat(cond_ins, 0).to(device)
    m = torch.cat(masks, 0).to(device)
    io_s = time.perf_counter() - t_io
    t_step = time.perf_counter()
    if train_mode:
        train_step(model, opt, u, c, m)
    else:
        model.eval()
        with torch.no_grad():
            lp = model.log_prob_free(u, c, m)
            if not torch.isfinite(lp).all():
                _fail("EVAL_PROBE_NON_FINITE",
                      "non-finite log-prob in the eval-mode probe")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return io_s, time.perf_counter() - t_step


def run(cfg: dict) -> dict:
    """cfg keys: data_root, manifest_dir, p4_stats2, implb_facts,
    out_dir. Nothing else is configurable; the protocol is frozen."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        _fail("REGISTERED_HOST_MISSING",
              "preflight must run on the registered CUDA host; CPU "
              "throughput is not representative and never substitutes")

    manifest = load_epoch_manifest(cfg["manifest_dir"], 0)
    p4 = ffr.load_p4s2_parent(cfg["p4_stats2"])
    implb = ffr.load_implb_parent(cfg["implb_facts"])
    ffr.require_spline_b(implb["spline_b"])

    # Own model instance, discarded at return: preflight state can never
    # seed the scientific run (V02SPEC §7).
    model = ffr.build_model(spline_b=implb["spline_b"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ds = FastMRISliceDataset(cfg["data_root"], split="train", mode="train")
    # Fresh-mask policy (EXEC SS3.7): the frozen protocol times the
    # epoch-0 manifest batches, so epoch 0 is declared before the
    # DataLoader samples; regression proof SEQREF-V02S f11.
    ds.set_epoch(0)
    index_of = {}
    for k, (path, sl) in enumerate(ds.index):
        index_of[(path.relative_to(ds.data_root).as_posix(), int(sl))] = k
    order = []
    for e in manifest["entries"]:
        key = (e["file"], int(e["slice_index"]))
        if key not in index_of:
            _fail("MANIFEST_ENTRY_UNKNOWN",
                  f"epoch-0 manifest entry {key} is not in the dataset "
                  f"traversal index")
        order.append(int(index_of[key]))

    windows = manifest["batches"][: WARMUP_BATCHES + TIMED_BATCHES]
    if len(windows) < WARMUP_BATCHES + TIMED_BATCHES:
        _fail("MANIFEST_TOO_SHORT",
              f"epoch-0 manifest has {len(manifest['batches'])} batches; "
              f"the frozen window needs {WARMUP_BATCHES + TIMED_BATCHES}")
    flat = [i for (lo, hi) in windows for i in range(lo, hi)]
    loader = DataLoader(Subset(ds, [order[i] for i in flat]),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, collate_fn=_collate)

    io_times, step_times = [], []
    torch.cuda.reset_peak_memory_stats()
    for bi, batch in enumerate(loader):
        model.train()
        io_s, step_s = _batch_work(model, opt, batch, p4,
                                   train_mode=True, device=device)
        if bi >= WARMUP_BATCHES:
            io_times.append(io_s)
            step_times.append(step_s)
    if len(step_times) != TIMED_BATCHES:
        _fail("TIMED_WINDOW_SHORT",
              f"timed window collected {len(step_times)} batches; the "
              f"frozen count is {TIMED_BATCHES}")
    peak_mem = _peak_memory_bytes()

    # Eval-mode projection probe (batches 1-4 re-read in eval mode).
    eval_step_times = []
    model.eval()
    for bi, batch in enumerate(loader):
        if bi >= EVAL_PROBE_BATCHES:
            break
        _, s = _batch_work(model, opt, batch, p4,
                           train_mode=False, device=device)
        eval_step_times.append(s)
    if len(eval_step_times) != EVAL_PROBE_BATCHES:
        _fail("EVAL_PROBE_SHORT",
              "the eval-mode projection probe could not collect its "
              "frozen batches")

    mean_io = float(np.mean(io_times))
    mean_step = float(np.mean(step_times))
    per_batch = mean_io + mean_step
    projected_training_s = per_batch * TOTAL_STEPS
    mean_eval_batch = float(np.mean(eval_step_times))
    eval_slices = (2 * N_TRAIN_SLICES            # train NLL at 0/final
                   + len(CHECKPOINT_STEPS) * N_HOLDOUT_SLICES)  # holdout
    projected_eval_s = (mean_eval_batch / BATCH_SIZE) * eval_slices
    # PM bank decodes (2 endpoints x 32 slices x 128 bank) are charged at
    # the eval per-slice rate with an explicit per-decode note; bank
    # decode cost dominates this term and is recorded, not hidden.
    projected_pm_s = (mean_eval_batch / BATCH_SIZE) * (
        2 * D3_SUBSET_N * PM_BANK_N)

    report = {
        "script": f"{__abbr__} v{__version__}",
        "schema": "seqref-v02-preflight/1",
        "lifetime": "EPHEMERAL",
        "manifest_sha256": manifest["manifest_sha256"],
        "warmup_batches": WARMUP_BATCHES,
        "timed_batches": TIMED_BATCHES,
        "io_per_batch_s": mean_io,
        "step_per_batch_s": mean_step,
        "peak_memory_bytes": peak_mem,
        "projected_training_s": projected_training_s,
        "projected_training_h": projected_training_s / 3600.0,
        "ceiling_s": WALL_CLOCK_CEILING_S,
        "projected_endpoint_eval_s": projected_eval_s,
        "projected_pm_bank_s": projected_pm_s,
        "endpoint_eval_note": "projected separately; NOT charged "
                              "against the 48 h training ceiling "
                              "(V02SPEC §7)"}
    if projected_training_s > WALL_CLOCK_CEILING_S:
        _fail("CEILING_PROJECTION_EXCEEDED",
              f"projected training {projected_training_s / 3600.0:.2f} h "
              f"> 48 h ceiling; ERROR before the scientific run -- no "
              f"silent downscale (V02SPEC §7)")

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "v02_preflight.json"
    path.write_bytes(canonical_json(report))
    logger.info("[%s] projection %.2f h vs 48 h ceiling; eval %.2f h "
                "separate; peak mem %.2f GiB", __abbr__,
                report["projected_training_h"],
                projected_eval_s / 3600.0, peak_mem / 2**30)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"{__abbr__} v{__version__} -- candidate v0.2 "
                    f"throughput preflight (V02SPEC §7)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--p4-stats2", required=True)
    ap.add_argument("--implb-facts", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
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
