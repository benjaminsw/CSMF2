# SEQREF-IMPLT v0.1 -- scripts.train_free_flow
# LIFETIME: KEEP
# =============================================================================
# Purpose: training entry point for the registered free-coordinate
#          conditional NSF (SEQREF-MRI-IMPLSPEC v0.1; FLOW_FAMILY=NSF;
#          B = SPLINE_B consumed from the pinned IMPL-B artefact).
#          Owns the PRODUCTION objective and optimizer step that the
#          Class-A A6 micro-training check calls -- A6 exercises THIS
#          code, never a reimplemented loss.
# Stage boundary (binding):
#   * run_training is the future TINY-stage entry. It is REAL, complete
#     code, but the SEQREF-IMPL v0.1 stage never executes it: the IMPL
#     stage ends when Class-A A1-A10 passes. There is no smoke mode, no
#     tiny alias, and no unregistered execution path.
#   * Training consumes the frozen P0S 256-slice corpus in EVAL mode
#     (deterministic masks identical to the recorded P3 bindings --
#     the same realisations IMPL-B calibrated). Train-mode fresh masks
#     belong to later stages and are never silently substituted here.
#   * Binding verification (ffr.verify_binding_identity) runs BEFORE any
#     target construction, decode or optimizer work on each sample.
#   * The optimizer ALWAYS receives the full production parameter set.
#     The A6 named groups (NSF-transform / conditioning) are
#     test-registration groups and never reach an optimizer.
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass. Architecture constants are frozen in
#   SEQREF-IMPLR and are NEVER CLI-tunable here; only the training
#   protocol (epochs/batch/lr/slices) is cfg-driven, and it is owned by
#   the TINY registration, not by this stage.
# Changelog (NEW in v0.1):
#   * Introduced for SEQREF-IMPL v0.1 (Class-A contract stage).
# Update summary:
#   v0.1 lands the production NLL objective, optimizer step, per-sample
#   bound target construction and the (stage-inert) eval-corpus training
#   loop. A6 calls nll_objective/train_step directly on a synthetic
#   fixture; everything else awaits the Class-A gate.
# =============================================================================
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from seqref_mri.src import free_flow_runtime as ffr
from seqref_mri.src.fastmri_data import FastMRISliceDataset
from seqref_mri.scripts.train_base import _collate, _prepare

logger = logging.getLogger("seqref_mri.train_free_flow")

__version__ = "0.1"
__abbr__ = "SEQREF-IMPLT"

REQUIRED_PREPARE_KEYS = ("y", "x_norm", "cond_in", "tgt_norm", "amax", "ops")


def _fail(code: str, message: str) -> None:
    logger.error("[SEQREF-IMPLT] %s: %s", code, message)
    raise ffr.FreeFlowError(code, message)


# ---------------------------------------------------------------------------
# Production objective + optimizer step (called by Class-A A6 verbatim)
# ---------------------------------------------------------------------------

def nll_objective(model: ffr.FreeFlowModel, u_scaled: torch.Tensor,
                  cond_in: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean batch NLL = -log p(u_scaled | cond_in, mask). Scalar tensor."""
    lp = model.log_prob_free(u_scaled, cond_in, mask)
    nll = -lp.mean()
    if not torch.isfinite(nll):
        _fail("NLL_NON_FINITE", "the production NLL objective is "
              "non-finite; no fallback is permitted")
    return nll


def train_step(model: ffr.FreeFlowModel, optimizer: torch.optim.Optimizer,
               u_scaled: torch.Tensor, cond_in: torch.Tensor,
               mask: torch.Tensor) -> float:
    """One production optimizer step on a fixed batch. Returns the
    PRE-step scalar NLL (the loss that produced these gradients)."""
    nll = nll_objective(model, u_scaled, cond_in, mask)
    value = float(nll.detach())
    optimizer.zero_grad(set_to_none=True)
    nll.backward()
    optimizer.step()
    return value


# ---------------------------------------------------------------------------
# Bound target construction for prepared batches
# ---------------------------------------------------------------------------

def targets_from_prepared(prep: dict, cmap, vecs: dict) -> torch.Tensor:
    """(B, FLOW_DIM_REAL) float32 training targets from a _prepare()
    result sharing ONE coordinate map. Float64 registered arithmetic
    inside; the float32 cast happens here, at the model boundary."""
    missing = [k for k in ("x_norm", "cond_in") if k not in prep]
    if missing:
        _fail("PREPARE_KEYS_MISSING",
              f"the prepared batch lacks {missing}; _prepare is "
              f"consumed, never reimplemented")
    scalars = ffr.encode_target(prep["x_norm"], prep["cond_in"], cmap, vecs)
    return torch.from_numpy(np.ascontiguousarray(scalars)).to(torch.float32)


def live_row(binding: dict, meta: dict, mask: torch.Tensor) -> dict:
    """The live-realisation row for binding verification. Identity fields
    come from the RECORDED P3 binding (corpus-rooted paths); the live
    columns come from the APPLIED batch mask. The dataset meta carries a
    data_root-relative path, so the sample is cross-checked on basename
    + slice index BEFORE verification -- the sample must be the recorded
    one (the same path-normalisation distinction as the Class-A
    fixture loader; verify_binding_identity itself is not weakened)."""
    if int(meta["slice_index"]) != int(binding["slice_index"]) \
            or Path(str(meta["file"])).name != Path(binding["file"]).name:
        _fail("BINDING_IDENTITY_MISMATCH",
              f"the sample meta file/slice ({meta['file']!r}, "
              f"{meta['slice_index']!r}) disagrees with the recorded P3 "
              f"binding ({binding['file']!r}, {binding['slice_index']!r}) "
              f"-- the corpus traversal is not the frozen one")
    cols = [int(c) for c in np.flatnonzero(
        mask.to(torch.bool).cpu().numpy()).tolist()]
    return {"dataset_index": int(binding["dataset_index"]),
            "file": binding["file"],
            "slice_index": int(binding["slice_index"]),
            "split": binding["split"],
            "mask_seed": int(binding["mask_seed"]),
            "live_columns": cols}


def dataset_index_map(ds: FastMRISliceDataset) -> dict:
    """(file_relpath, slice_index) -> dataset_index, built ONCE from the
    dataset's own traversal index. Sample identity binds to recorded
    data, never to a loop counter."""
    out = {}
    for k, (path, sl) in enumerate(ds.index):
        key = (path.relative_to(ds.data_root).as_posix(), int(sl))
        out[key] = k
    return out


def binding_for_dataset_index(bindings: list, dataset_index: int) -> dict:
    """The recorded P3 binding for one dataset index. A missing binding is
    ERROR: IMPL trains only on the frozen, recorded corpus."""
    for b in bindings:
        if int(b["dataset_index"]) == int(dataset_index):
            return b
    _fail("BINDING_NOT_RECORDED",
          f"dataset_index {dataset_index} has no recorded P3 binding; "
          f"IMPL consumes the frozen 256-slice corpus only")


# ---------------------------------------------------------------------------
# Training loop (TINY-stage entry; built now, executed NEVER in this stage)
# ---------------------------------------------------------------------------

def run_training(cfg: dict) -> dict:
    """Train the registered model on the frozen eval-mode corpus.

    cfg keys: repo_dir, data_root, p3_facts, p4_stats2, implb_facts,
    out_root, epochs (int>0), batch (int>0), lr (float>0),
    seed_index (int>=0), max_slices (int|None).
    Architecture constants come from SEQREF-IMPLR, never from cfg.
    Returns an in-memory facts dict (the TINY stage owns persistence)."""
    for key, pred in (("epochs", lambda v: type(v) is int and v > 0),
                      ("batch", lambda v: type(v) is int and v > 0),
                      ("lr", lambda v: isinstance(v, (int, float)) and v > 0),
                      ("seed_index", lambda v: type(v) is int and v >= 0)):
        if not pred(cfg.get(key)):
            _fail("CFG_INVALID", f"cfg[{key!r}]={cfg.get(key)!r} fails its "
                  f"registered predicate")
    t0 = time.time()
    p3 = ffr.load_p3_parent(cfg["p3_facts"])
    p4 = ffr.load_p4s2_parent(cfg["p4_stats2"])
    implb = ffr.load_implb_parent(cfg["implb_facts"])
    logger.info("[SEQREF-IMPLT] parents pinned: P3 %s | P4/2 %s | IMPL-B "
                "%s (B=%.17g)", p3["file_sha256"][:12],
                p4["file_sha256"][:12], implb["file_sha256"][:12],
                implb["spline_b"])

    model = ffr.build_model(spline_b=implb["spline_b"])
    n_params = sum(p.numel() for p in model.parameters())
    # FULL production parameter set; A6 named groups never reach here.
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    ds = FastMRISliceDataset(cfg["data_root"], split="train", mode="eval")
    index_of = dataset_index_map(ds)
    order = [int(b["dataset_index"]) for b in p3["bindings"]]
    if cfg.get("max_slices") is not None:
        order = order[: int(cfg["max_slices"])]
    bindings_by_index = {int(b["dataset_index"]): b
                         for b in p3["bindings"]}
    loader = DataLoader(Subset(ds, order), batch_size=int(cfg["batch"]),
                        shuffle=False, collate_fn=_collate)

    vec_cache: dict[str, tuple] = {}
    history = []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        ep_nll = []
        for batch in loader:
            prep = _prepare(batch, "cpu", test0=False)
            missing = [k for k in REQUIRED_PREPARE_KEYS if k not in prep]
            if missing:
                _fail("PREPARE_KEYS_MISSING",
                      f"_prepare result lacks {missing}")
            targets, cond_ins, masks = [], [], []
            # Per-sample binding verification runs BEFORE any target
            # construction, decode or optimizer work on the sample.
            for j, meta in enumerate(batch["meta"]):
                key = (meta["file"], int(meta["slice_index"]))
                if key not in index_of:
                    _fail("SAMPLE_NOT_IN_CORPUS",
                          f"live sample {key} is not in the dataset "
                          f"traversal index")
                ds_index = int(index_of[key])
                if ds_index not in bindings_by_index:
                    _fail("BINDING_NOT_RECORDED",
                          f"dataset_index {ds_index} has no recorded P3 "
                          f"binding; IMPL consumes the frozen 256-slice "
                          f"corpus only")
                binding = bindings_by_index[ds_index]
                row = live_row(binding, meta, batch["mask"][j])
                cmap = ffr.verify_binding_identity(row, binding)
                map_key = binding["map_sha256"]
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
            u_scaled = torch.cat(targets, dim=0)
            cond_in = torch.cat(cond_ins, dim=0)
            mask_b = torch.cat(masks, dim=0)
            ep_nll.append(train_step(model, opt, u_scaled, cond_in, mask_b))
        rec = {"epoch": epoch, "train_nll_mean": float(np.mean(ep_nll)),
               "train_nll_last": float(ep_nll[-1])}
        history.append(rec)
        logger.info("[SEQREF-IMPLT] ep%d nll_mean=%.4f", epoch,
                    rec["train_nll_mean"])
    out_root = Path(cfg["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt = out_root / "free_flow_final.pt"
    torch.save({"model": model.state_dict(),
                "epoch": int(cfg["epochs"]) - 1}, ckpt)
    return {"script": f"{__abbr__} v{__version__}",
            "n_params": n_params, "history": history,
            "checkpoint_name": ckpt.name,
            "elapsed_s": time.time() - t0}


def main() -> None:
    """TINY-stage entry point. Exists and is real; NOT executed under the
    SEQREF-IMPL v0.1 stage (Class-A gate first)."""
    ap = argparse.ArgumentParser(
        description=f"{__abbr__} v{__version__} -- free-coordinate "
                    f"conditional-NSF trainer (TINY-stage entry; inert "
                    f"during the Class-A stage)")
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--p3-facts", required=True)
    ap.add_argument("--p4-stats2", required=True)
    ap.add_argument("--implb-facts", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed-index", type=int, default=0)
    ap.add_argument("--max-slices", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s "
                               "%(message)s")
    run_training(vars(args))


if __name__ == "__main__":
    main()
