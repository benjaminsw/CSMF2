# SEQREF-TDIAG v0.1 -- tdiag.replay
# LIFETIME: KEEP
# =============================================================================
# Purpose: R0 -- replay validity (EXEC SS10.6, locked 2026-08-15). TDIAG
#          deterministically reconstructs the step-0 and step-500 model
#          states from the registered TINY configuration and requires
#          EXACT equality of the registered serialized values against the
#          dual-pinned authoritative tiny_facts.json:
#            * endpoint batch-mean NLL (step 0 / step 500),
#            * endpoint mean z=0 PSNR, endpoint mean z=0 NMSE_u,
#            * the NLL trace at steps {0, 50, ..., 500} (11 points),
#            * the selection manifest and draw order,
#            * the IMPL/TINY parent hashes.
#          NO tolerance is introduced. Any deviation => typed StageError
#          (R0_REPLAY_MISMATCH / R0_TRACE_INCOMPLETE / parent-pin codes);
#          no diagnosis is emitted on an invalid replay.
# Construction doctrine: the replay REUSES the production TINY
#   construction primitives from scripts.tiny_gate (selection, slice
#   states, endpoint metric engine, train loop) verbatim -- a replay that
#   reimplemented training would test the reimplementation, not the
#   registered configuration. TINY's PASS/BLOCK taxonomy is NOT imported;
#   only construction primitives are reused.
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * Introduced with the R0 slice after the 2026-08-15 EXEC SS10.6 lock.
#   * Review-repair round (2026-08-16, pre-execution; NO contract
#     change): compare_registered now receives the FRESHLY VERIFIED live
#     parent identities (IMPL file+semantic, TINY file) from the driver
#     and compares the registered records against them; previously the
#     parent_impl_semantic_sha256 and parent_tiny_file_sha256 rows were
#     tautological (artefact field vs itself / constant vs itself).
#   * D1 slice (2026-08-18, under the same SS10.6 lock; NO contract
#     change): ReplayContext -- the frozen step-500 runtime (model,
#     slice states, selection, spline_b, s_ref) is now handed from R0 to
#     D1 via run_r0_with_context; the context holds live torch objects
#     and is NEVER serialized into facts. R0's scientific calculation is
#     unchanged; run_r0 keeps its original signature.
#   * D2a slice (2026-08-19, under the same SS10.6 lock; NO contract
#     change): ReplayContext now ALSO carries the captured step-0
#     state_dict (state0) so D2a can swap the verified step-0 state into
#     the SAME model object under state-hash verification -- no second
#     model is built. R0's scientific calculation is unchanged; run_r0
#     keeps its original signature.
# Update summary:
#   v0.1 lands the TINY dual-pin loader, the registered-selection
#   re-derivation check, the deterministic state-capture/hash helpers,
#   the exact serialized-value comparison engine (per-quantity equality
#   booleans, never one silent overall flag) and the R0->D1-D3 frozen
#   runtime handover (ReplayContext, facts-free; now including the
#   captured step-0 state_dict for the D2a state-swap invariant).
# =============================================================================
from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from seqref_mri.tdiag import _bootstrap  # noqa: F401

from preflight_io import canonical_hash, file_sha256, verify_sidecar
from preflight_parents import StageError
from seqref_mri.scripts import tiny_gate as tg
from seqref_mri.scripts.train_base import _collate

from seqref_mri.tdiag.invariants import (R0_MODEL_INIT_SEED,
                                         R0_TRACE_CHECKPOINTS,
                                         TINY_FACTS_FILE_SHA256,
                                         TINY_FACTS_SCHEMA,
                                         TINY_FACTS_SEMANTIC_SHA256,
                                         TINY_REQUIRED_VERDICT)

logger = logging.getLogger("SEQREF-TDIAG")


def _fail(code: str, message: str, **kwargs) -> StageError:
    logger.error("[SEQREF-TDIAG] %s: %s", code, message)
    return StageError(code, message, **kwargs)


# ---------------------------------------------------------------------------
# Authoritative TINY parent: dual-pin load (file sha + sidecar + schema +
# semantic + required BLOCK verdict), mirroring tiny_gate._load_impl_parent
# doctrine. The registered comparison values are read from the VERIFIED
# artefact -- never hardcoded downstream of the pin.
# ---------------------------------------------------------------------------

def load_tiny_parent(path: str) -> tuple[dict, str]:
    """Dual-pin the authoritative TINY artefact. Returns (parsed facts,
    VERIFIED file sha256)."""
    sha = file_sha256(path)
    if sha != TINY_FACTS_FILE_SHA256:
        raise _fail("PARENT_FILE_MISMATCH",
                    f"TINY facts file sha256 {sha} != registered pin "
                    f"{TINY_FACTS_FILE_SHA256}")
    verify_sidecar(path)
    with open(path, "r", encoding="utf-8") as fh:
        art = json.load(fh)
    if art.get("schema") != TINY_FACTS_SCHEMA:
        raise _fail("PARENT_SCHEMA_MISMATCH",
                    f"TINY facts schema {art.get('schema')!r} != "
                    f"{TINY_FACTS_SCHEMA!r}")
    if art.get("semantic_sha256") != TINY_FACTS_SEMANTIC_SHA256:
        raise _fail("PARENT_SEMANTIC_MISMATCH",
                    f"TINY facts semantic sha256 "
                    f"{art.get('semantic_sha256')} != registered pin "
                    f"{TINY_FACTS_SEMANTIC_SHA256}")
    if art.get("verdict") != TINY_REQUIRED_VERDICT:
        raise _fail("PARENT_VERDICT_MISMATCH",
                    f"TINY facts verdict {art.get('verdict')!r} != "
                    f"{TINY_REQUIRED_VERDICT!r}; R0 replays the "
                    f"authoritative CLOSED BLOCK run only")
    if art.get("authoritative") is not True:
        raise _fail("PARENT_NOT_AUTHORITATIVE",
                    "TINY facts authoritative flag is not true; R0 "
                    "requires the authoritative artefact")
    logger.info("[SEQREF-TDIAG] TINY parent pinned: file %s | semantic %s",
                sha[:12], art["semantic_sha256"][:12])
    return art, sha


# ---------------------------------------------------------------------------
# Deterministic state capture + canonical hashes.
# ---------------------------------------------------------------------------

def capture_state(model) -> dict:
    """Deep CPU copy of the model state_dict. Called exactly twice by the
    replay: on the untouched step-0 model and after exactly 500 steps."""
    return {k: v.detach().cpu().clone() for k, v in
            model.state_dict().items()}


def state_hash(state: dict) -> str:
    """Canonical sha256 over a state_dict: sorted keys; per entry the key,
    dtype, shape and the raw C-order bytes. Insertion order and device do
    not affect the hash; the float32 payload is preserved exactly."""
    h = hashlib.sha256()
    for key in sorted(state):
        arr = np.asarray(state[key].detach().cpu().contiguous().numpy())
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(str(arr.dtype).encode("ascii"))
        h.update(repr(arr.shape).encode("ascii"))
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def replay_config_hash(selection: dict, spline_b: float) -> str:
    """Canonical hash of the full registered replay configuration."""
    return canonical_hash({
        "stage": "TDIAG-R0",
        "selection_manifest_sha256": selection["manifest_sha256"],
        "steps": 500,
        "trace_checkpoints": list(R0_TRACE_CHECKPOINTS),
        "model_init_seed": R0_MODEL_INIT_SEED,
        "optimizer": "Adam(lr=0.0001, betas=(0.9, 0.999), eps=1e-08, "
                     "weight_decay=0.0); no schedule",
        "spline_b": float(spline_b),
        "flow_dim_real": int(tg.ffr.FLOW_DIM_REAL),
        "precision": "float32 production forward/training; float64 "
                     "metrics",
        "threads": 1})


# ---------------------------------------------------------------------------
# Exact serialized-value comparison engine (EXEC SS10.6 R0): per-quantity
# equality booleans, exact (==) semantics -- no epsilon, and NaN never
# equals NaN so a non-finite replay value is a mismatch, not a pass.
# ---------------------------------------------------------------------------

def _trace_points(facts_trace: dict, replayed_trace: dict,
                  comparisons: list) -> None:
    keys = [str(k) for k in R0_TRACE_CHECKPOINTS]
    missing_reg = [k for k in keys if k not in facts_trace]
    missing_rep = [k for k in keys if k not in replayed_trace]
    if missing_reg or missing_rep:
        raise _fail("R0_TRACE_INCOMPLETE",
                    f"trace checkpoint(s) missing -- registered "
                    f"{missing_reg}, replayed {missing_rep}; the "
                    f"11-point grid {keys} is registered")
    for k in keys:
        reg, rep = facts_trace[k], replayed_trace[k]
        comparisons.append({"quantity": f"nll_trace.step{k}",
                            "registered": reg, "replayed": rep,
                            "equal": bool(reg == rep)})


def compare_registered(tiny_facts: dict, selection: dict,
                       trace: dict, m0: dict, m500: dict,
                       impl_file_sha: str, impl_semantic_sha: str,
                       tiny_file_sha: str) -> dict:
    """Build the R0 comparison record. Returns {"comparisons": [...],
    "valid": bool}. valid requires EVERY registered quantity equal.
    The three live identity arguments are the FRESHLY VERIFIED values
    (loader output on this execution), never re-read from the artefact
    being checked against -- the parent-hash rows compare registered
    records against live verification, not against themselves."""
    ep0 = tiny_facts["endpoints"]["initial"]
    ep500 = tiny_facts["endpoints"]["final"]
    sel_reg = tiny_facts["selection"]
    impl_reg = tiny_facts["parents"]["impl_class_a"]
    tiny_semantic_live = tiny_facts["semantic_sha256"]  # pin-verified load
    pairs = [
        ("step0_nll", ep0["nll_batch_mean"], m0["nll_batch_mean"]),
        ("step500_nll", ep500["nll_batch_mean"], m500["nll_batch_mean"]),
        ("step0_z0_mean_psnr", ep0["mean_psnr_z0"], m0["mean_psnr_z0"]),
        ("step500_z0_mean_psnr", ep500["mean_psnr_z0"],
         m500["mean_psnr_z0"]),
        ("step0_z0_mean_nmse_u", ep0["mean_nmse_u_z0"],
         m0["mean_nmse_u_z0"]),
        ("step500_z0_mean_nmse_u", ep500["mean_nmse_u_z0"],
         m500["mean_nmse_u_z0"]),
        ("selection_manifest_sha256", sel_reg["manifest_sha256"],
         selection["manifest_sha256"]),
        ("selection_draw_order", list(sel_reg["draw_order_indices"]),
         list(selection["draw_order_indices"])),
        ("parent_impl_file_sha256", impl_reg["file_sha256"],
         impl_file_sha),
        ("parent_impl_semantic_sha256", impl_reg["semantic_sha256"],
         impl_semantic_sha),
        ("parent_tiny_file_sha256", TINY_FACTS_FILE_SHA256,
         tiny_file_sha),
        ("parent_tiny_semantic_sha256", TINY_FACTS_SEMANTIC_SHA256,
         tiny_semantic_live),
    ]
    comparisons = [{"quantity": name, "registered": reg, "replayed": rep,
                    "equal": bool(reg == rep)}
                   for name, reg, rep in pairs]
    _trace_points(tiny_facts["nll_trace"], trace, comparisons)
    valid = all(c["equal"] for c in comparisons)
    return {"comparisons": comparisons, "valid": bool(valid),
            "rule": "exact equality of the registered serialized values; "
                    "no tolerance (EXEC SS10.6 R0)"}


# ---------------------------------------------------------------------------
# R0 orchestration: the registered TINY construction, replayed.
# ---------------------------------------------------------------------------

@dataclass
class ReplayContext:
    """The frozen step-500 runtime handed from R0 to D1-D3: the trained
    model object, the per-slice states (conditioner tensors and masks
    live inside them), the registered selection, spline_b and s_ref,
    plus the CAPTURED step-0 state_dict (state0) for the D2a state-swap
    identity invariant. INTERNAL ONLY -- it holds live torch objects and
    is NEVER serialized into the facts document. state0 defaults to
    None for R0-only/D1 callers; run_d2a refuses a missing state0
    (D2A_STATE0_MISSING) and discards it (sets None) when done."""
    model: object
    states: list
    selection: dict
    spline_b: float
    s_ref: float
    state0: dict | None = None


def _run_r0_impl(data_root: str, tiny_facts: dict, impl_file_sha: str,
                 impl_semantic_sha: str, tiny_file_sha: str,
                 spline_b: float, p4: dict,
                 s_ref: float) -> tuple[dict, ReplayContext]:
    """Replay the registered TINY configuration and compare against the
    verified authoritative artefact. On any deviation: typed StageError
    (no partial diagnosis is returned). spline_b comes from the verified
    IMPL-B parent; impl_file_sha/impl_semantic_sha/tiny_file_sha are the
    FRESHLY VERIFIED live parent identities (driver-provided). P0S
    overlap is NOT re-derived here: it is a TINY-gate observation
    already recorded inside the authoritative artefact; R0 consumes only
    the selection manifest/draw. Returns (result, ReplayContext) -- the
    context carries the FROZEN step-500 runtime for D1-D3; the result
    dict is unchanged from the R0-only contract.
    """
    ds = tg.FastMRISliceDataset(data_root, split="train", mode="eval")
    selection = tg._select_batch(ds)
    loader = DataLoader(Subset(ds, selection["draw_order_indices"]),
                        batch_size=tg.TINY_BATCH, shuffle=False,
                        collate_fn=_collate)
    batch = next(iter(loader))
    states = tg._build_slice_states(batch, selection, p4, s_ref)

    model = tg.ffr.build_model(spline_b=spline_b,
                               init_seed=R0_MODEL_INIT_SEED)
    latents = torch.randn(
        tg.LATENT_BANK_N, tg.ffr.FLOW_DIM_REAL,
        generator=torch.Generator().manual_seed(tg.LATENT_BANK_SEED))
    targets = torch.from_numpy(np.concatenate(
        [st["target"] for st in states], axis=0).astype(np.float32))
    cond = torch.cat([st["cond"] for st in states], dim=0)
    mask = torch.cat([st["mask"] for st in states], dim=0)

    # Step 0: the UNTOUCHED model (same endpoint engine as TINY).
    state0 = capture_state(model)
    m0 = tg._endpoint_metrics(model, states, s_ref, latents)
    trace = {"0": m0["nll_batch_mean"]}

    optimizer = torch.optim.Adam(model.parameters(), lr=tg.TINY_LR,
                                 betas=tg.TINY_BETAS, eps=tg.TINY_EPS,
                                 weight_decay=tg.TINY_WEIGHT_DECAY)
    model.train()
    for step in range(1, 501):
        try:
            tg.tff.train_step(model, optimizer, targets, cond, mask)
        except tg.ffr.FreeFlowError as exc:
            raise _fail("R0_REPLAY_NON_FINITE",
                        f"step {step}: {type(exc).__name__}: {exc}")
        except ValueError as exc:
            if "non-finite" in str(exc).lower():
                raise _fail("R0_REPLAY_NON_FINITE",
                            f"step {step}: {type(exc).__name__}: {exc}")
            raise
        if step in R0_TRACE_CHECKPOINTS:
            trace[str(step)] = tg._nll(model, targets, cond, mask)

    # Step 500: exactly 500 steps, no checkpoint selection anywhere.
    state500 = capture_state(model)
    m500 = tg._endpoint_metrics(model, states, s_ref, latents)

    comparison = compare_registered(tiny_facts, selection, trace, m0,
                                    m500, impl_file_sha,
                                    impl_semantic_sha, tiny_file_sha)
    result = {
        "selection": selection,
        "trace": trace,
        "endpoints": {"initial": m0, "final": m500},
        "step0_state_hash": state_hash(state0),
        "step500_state_hash": state_hash(state500),
        "replay_config_hash": replay_config_hash(selection, spline_b),
        **comparison}
    if not result["valid"]:
        mismatches = [c["quantity"] for c in result["comparisons"]
                      if not c["equal"]]
        raise _fail("R0_REPLAY_MISMATCH",
                    f"the deterministic replay does not reproduce the "
                    f"registered serialized values exactly; mismatched "
                    f"quantities: {mismatches}; NO diagnosis is emitted "
                    f"(EXEC SS10.6 R0)",
                    detail={"mismatches": mismatches,
                            "comparisons": result["comparisons"]})
    logger.info("[SEQREF-TDIAG] R0 replay VALID: %d/%d registered "
                "quantities exactly reproduced",
                len(result["comparisons"]), len(result["comparisons"]))
    ctx = ReplayContext(model=model, states=states,
                        selection=selection, spline_b=float(spline_b),
                        s_ref=float(s_ref), state0=state0)
    return result, ctx


def run_r0(data_root: str, tiny_facts: dict, impl_file_sha: str,
           impl_semantic_sha: str, tiny_file_sha: str,
           spline_b: float, p4: dict, s_ref: float) -> dict:
    """R0 replay, result only (original R0-slice signature, kept for
    callers that do not continue into D1-D3)."""
    return _run_r0_impl(data_root, tiny_facts, impl_file_sha,
                        impl_semantic_sha, tiny_file_sha, spline_b, p4,
                        s_ref)[0]


def run_r0_with_context(data_root: str, tiny_facts: dict,
                        impl_file_sha: str, impl_semantic_sha: str,
                        tiny_file_sha: str, spline_b: float, p4: dict,
                        s_ref: float) -> tuple[dict, ReplayContext]:
    """R0 replay plus the frozen step-500 runtime context for D1-D3.
    The model is handed over at exactly step 500 -- it is never rebuilt
    or retrained downstream."""
    return _run_r0_impl(data_root, tiny_facts, impl_file_sha,
                        impl_semantic_sha, tiny_file_sha, spline_b, p4,
                        s_ref)
