# SEQREF-TDIAG v0.1 -- tdiag.d2c
# LIFETIME: KEEP
# =============================================================================
# Purpose: D2c -- VOLUME-level held-out generalization with locked
#          selection (EXEC SS10.6, locked 2026-08-15). The trained
#          trajectory is FIXED: the R0-registered step-0 and step-500
#          model states are swapped into the same model and evaluated
#          on 32 slices from volumes NEVER represented among the 8
#          TINY training slices. NO retraining on the holdout set
#          (review 2026-08-20): this is a memorization-vs-transfer
#          diagnostic of the TRAINED model, not a new training run.
#          Observable:
#            G       = delta_NLL / dim,  delta = step0 - step500
#                      (POSITIVE = likelihood improvement),
#            G_hold  = mean over the 32 per-slice delta_NLL / dim
#                      (AGGREGATE MEAN of per-slice deltas is the
#                      registered observable; the two-endpoint batch
#                      form is recorded alongside as a cross-check),
#            G_train = (18883.5859375 - (-35316.66015625)) / 13824 from
#                      the REGISTERED R0/TINY endpoints (single
#                      authoritative source; never recomputed from the
#                      D2b decomposition),
#            R       = G_hold / G_train.
#          Frozen bands: R <= 0.25 strong memorization-CONSISTENT
#          evidence (NOT proof of memorization: train/holdout
#          difficulty differences or finite-sample effects can also
#          weaken holdout gain); R >= 0.75 strong transfer evidence for
#          LIKELIHOOD GAIN (not necessarily reconstruction transfer);
#          between = mixed. R feeds the decision matrix (R <= 0.25 =>
#          data/budget redesign => candidate v0.2); the per-slice
#          records, PSNR/NMSE_u context, improved fraction and the
#          bootstrap CI are DESCRIPTIVE ONLY and never routed.
# Gates and invariants:
#   * hard selection invariants: exactly 32 selected files, all unique,
#     disjoint from every TINY source file (else D2C_SELECTION_*);
#   * state-swap identity: the same 4-boundary state_hash verification
#     as D2a/D2b against the R0-registered step0/step500 hashes (the
#     step-0 state_dict lifetime is DRIVER-OWNED -- run_d2c never
#     discards it);
#   * identity order: the measured per-slice identities at step 0 and
#     step 500 must equal the state list order EXACTLY (the same 32
#     identities at both steps), else D2C_IDENTITY_ORDER_MISMATCH;
#   * holdout_reconciliation_error = G_hold_batch - G_hold_per_slice is
#     RECORDED, never gated (no tolerance is frozen before the real
#     code path's numerical bound is characterised -- same doctrine as
#     the D2b reconciliation, review 2026-08-19/2026-08-20);
#   * the bootstrap CI (PCG64(3), B=10000) is DESCRIPTIVE ONLY and does
#     not alter the locked R classification (review 2026-08-20).
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * D2c slice (2026-08-20, under the same SS10.6 lock; NO contract
#     change): module introduced with the evidence-grade locked
#     selection (PCG64(1) over the canonical eligible-file list, frozen
#     midpoint rule), the two-state holdout measurement, G/R with the
#     registered-endpoint G_train, per-slice + aggregate statistics,
#     the descriptive z=0 PSNR/NMSE_u context and the descriptive-only
#     bootstrap CI.
# Update summary:
#   v0.1 D2c lands the volume-level holdout generalization measurement
#   on the registered step-0/step-500 states: locked 32-volume
#   selection with hard uniqueness/disjointness invariants, exact
#   identity-order and state-identity gates, G_hold as the per-slice
#   aggregate mean with a recorded batch cross-check, locked-band
#   classification, and descriptive-only reconstruction context and
#   bootstrap uncertainty.
# =============================================================================
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from seqref_mri.tdiag import _bootstrap  # noqa: F401

from preflight_io import canonical_hash
from preflight_parents import StageError
from seqref_mri.scripts import tiny_gate as tg
from seqref_mri.scripts.train_base import _collate
from seqref_mri.tdiag import d2b, replay
from seqref_mri.tdiag import invariants as tinv

logger = logging.getLogger("SEQREF-TDIAG")


def _fail(code: str, message: str, **kwargs) -> StageError:
    logger.error("[SEQREF-TDIAG] %s: %s", code, message)
    return StageError(code, message, **kwargs)


def _verify_state(model, expected: str, label: str) -> str:
    h = replay.state_hash(replay.capture_state(model))
    if h != expected:
        raise _fail("D2C_STATE_MISMATCH",
                    f"state-hash mismatch at boundary '{label}': model "
                    f"state {h} != registered {expected}")
    return h


# ---------------------------------------------------------------------------
# Locked selection (EXEC SS10.6 D2c). Pure/dataset-boundary separated so
# fixtures exercise the REAL selection logic against a stub dataset.
# ---------------------------------------------------------------------------

def _slice_position(n: int) -> int:
    """Frozen midpoint rule (EXEC SS10.6 D2c): odd n -> floor(n/2);
    even n -> n/2 - 1, i.e. the LOWER of the two middle slices. Frozen
    in code and pinned by selftest so the convention can never silently
    change (review 2026-08-20)."""
    if n < 1:
        raise _fail("D2C_EMPTY_FILE",
                    f"file contributes {n} eval slices; the midpoint "
                    "rule is undefined for an empty file")
    return n // 2 if n % 2 == 1 else n // 2 - 1


def _file_inventory(ds) -> tuple:
    """(canonical sorted unique relative file list, per-file ascending
    (slice_index, dataset_index) entries). Production identity
    convention (tiny_gate._select_batch): path RELATIVE to data_root,
    POSIX spelling."""
    per_file: dict[str, list] = {}
    for i, (path, slice_index) in enumerate(ds.index):
        rel = Path(path).relative_to(ds.data_root).as_posix()
        per_file.setdefault(rel, []).append((int(slice_index), i))
    files = sorted(per_file)
    slices_by_file = {f: sorted(entries) for f, entries in
                      per_file.items()}
    return files, slices_by_file


def select_holdout(ds, tiny_selection: dict) -> dict:
    """Locked holdout selection: exclude every source .h5 file
    represented among the 8 TINY slices; from the canonical sorted
    eligible list, Generator(PCG64(1)).choice(n_eligible, 32,
    replace=False); one slice per selected file by the frozen midpoint
    rule. Selection is EVIDENCE (the routing decision depends on it):
    full provenance + hard uniqueness/disjointness invariants."""
    files, slices_by_file = _file_inventory(ds)
    population = len(files)
    tiny_idents = tiny_selection.get("ordered_identities")
    if not isinstance(tiny_idents, list) or not tiny_idents:
        raise _fail("D2C_TINY_SELECTION_MISSING",
                    "the verified TINY facts carry no usable "
                    "selection.ordered_identities; the exclusion set "
                    "cannot be derived")
    excluded = sorted({str(r["file"]) for r in tiny_idents})
    excluded_set = set(excluded)
    missing = sorted(excluded_set - set(files))
    if missing:
        raise _fail("D2C_TINY_FILE_NOT_IN_POPULATION",
                    f"TINY source files absent from the eval "
                    f"population: {missing}; the dataset root or split "
                    "does not match the registered TINY selection")
    eligible = [f for f in files if f not in excluded_set]
    n_eligible = len(eligible)
    if n_eligible < tinv.D2C_HOLDOUT_N:
        raise _fail("D2C_POPULATION_TOO_SMALL",
                    f"eligible files {n_eligible} < D2C_HOLDOUT_N "
                    f"{tinv.D2C_HOLDOUT_N} after excluding "
                    f"{len(excluded)} TINY source files from "
                    f"{population}")
    rng = np.random.Generator(np.random.PCG64(tinv.D2C_SELECTION_SEED))
    draw = rng.choice(n_eligible, tinv.D2C_HOLDOUT_N, replace=False)
    draw_idx = [int(i) for i in draw]
    if (len(set(draw_idx)) != tinv.D2C_HOLDOUT_N
            or min(draw_idx) < 0 or max(draw_idx) >= n_eligible):
        raise _fail("D2C_SELECTION_NOT_UNIQUE",
                    f"the locked draw produced invalid indices "
                    f"{draw_idx} over {n_eligible} eligible files")
    selected = []
    for order, ei in enumerate(draw_idx):
        f = eligible[ei]
        entries = slices_by_file[f]            # ascending slice_index
        pos = _slice_position(len(entries))
        slice_index, dataset_index = entries[pos]
        selected.append({
            "draw_order": order,
            "eligible_file_index": ei,
            "canonical_file_index": files.index(f),
            "file": f,
            "n_slices": len(entries),
            "selected_slice_position": pos,
            "slice_index": slice_index,
            "dataset_index": dataset_index,
            "identity": {"split": "train", "file": f,
                         "slice_index": slice_index,
                         "dataset_index": dataset_index}})
    # Hard invariants (review 2026-08-20): exactly 32 unique files,
    # disjoint from every TINY source file. Guaranteed by construction;
    # verified loudly so a future edit cannot silently break them.
    sel_files = [r["file"] for r in selected]
    if len(set(sel_files)) != tinv.D2C_HOLDOUT_N:
        raise _fail("D2C_SELECTION_NOT_UNIQUE",
                    "selected files are not 32 unique volumes")
    contaminated = sorted(excluded_set & set(sel_files))
    if contaminated:
        raise _fail("D2C_SELECTION_CONTAMINATION",
                    f"selected files overlap the TINY training sources: "
                    f"{contaminated}")
    canonical_sorted_identities = sorted(
        ({k: v for k, v in r["identity"].items() if k != "dataset_index"}
         for r in selected),
        key=lambda r: (r["file"], r["slice_index"]))
    eligible_manifest = canonical_hash({
        "rule": "canonical sorted unique eval files minus every TINY "
                "source file",
        "population_file_count": population,
        "excluded_tiny_files": excluded,
        "eligible_files": eligible})
    selection_manifest = canonical_hash({
        "rule": "PCG64(1).choice(n_eligible_files, 32, replace=False); "
                "draw order fixed; midpoint: n odd -> floor(n/2), "
                "n even -> n/2 - 1",
        "eligible_file_count": n_eligible,
        "draw_file_indices": draw_idx,
        "ordered_identities": [r["identity"] for r in selected]})
    identity_manifest = canonical_hash({
        "rule": "canonical sorted identities of the 32 selected "
                "holdout slices",
        "canonical_sorted_identities": canonical_sorted_identities})
    logger.info("[%s] D2c selection: population=%d excluded=%d "
                "eligible=%d draw manifest=%s", "SEQREF-TDIAG",
                population, len(excluded), n_eligible,
                selection_manifest[:12])
    return {
        "rule": "PCG64(1).choice(n_eligible_files, 32, replace=False); "
                "draw order fixed",
        "midpoint_rule": "n odd -> floor(n/2); n even -> n/2 - 1 "
                         "(the LOWER of the two middle slices)",
        "population_file_count": population,
        "excluded_tiny_slice_count": len(tiny_idents),
        "excluded_tiny_file_count": len(excluded),
        "excluded_tiny_files": excluded,
        "eligible_file_count": n_eligible,
        "rng": {"generator": "PCG64", "seed": tinv.D2C_SELECTION_SEED},
        "draw_file_indices": draw_idx,
        "selected": selected,
        "invariants": {"selected_files_unique": True,
                       "selected_disjoint_from_tiny": True},
        "eligible_manifest_sha256": eligible_manifest,
        "selection_manifest_sha256": selection_manifest,
        "selected_identity_manifest_sha256": identity_manifest}


def build_holdout_states(ds, selection: dict, p4: dict,
                         s_ref: float) -> list:
    """Slice states for the 32 selected holdout slices through the SAME
    production construction path as the TINY states (loader order =
    draw order; identity cross-check inside _build_slice_states)."""
    dataset_indices = [r["dataset_index"] for r in selection["selected"]]
    loader = DataLoader(Subset(ds, dataset_indices),
                        batch_size=len(dataset_indices), shuffle=False,
                        collate_fn=_collate)
    batch = next(iter(loader))
    sel = {"ordered_identities": [r["identity"]
                                  for r in selection["selected"]]}
    states = tg._build_slice_states(batch, sel, p4, s_ref)
    if len(states) != tinv.D2C_HOLDOUT_N:
        raise _fail("D2C_STATE_COUNT_MISMATCH",
                    f"built {len(states)} holdout states, expected "
                    f"{tinv.D2C_HOLDOUT_N}")
    return states


# ---------------------------------------------------------------------------
# Measurement: production NLL (batch) + per-slice decomposition terms +
# descriptive z=0 reconstruction context, at one model state.
# ---------------------------------------------------------------------------

def _z0_metrics(model, st: dict) -> tuple:
    """(z=0 PSNR, z=0 NMSE_u or None when excluded) through the
    production TINY decode/metric path. DESCRIPTIVE reconstruction
    context only -- never routed (review 2026-08-20)."""
    z0 = torch.zeros(1, tg.ffr.FLOW_DIM_REAL)
    x0_c, u0 = tg._decode_z(model, z0, st)
    psnr = tg._psnr(x0_c.abs(), st["x_true_mag"])
    nmse = None if st["excluded"] else tg._nmse(u0, st["u_true"])
    return psnr, nmse


def _measure_at_state(model, states: list, step_label: str) -> dict:
    model.eval()
    nll_batch = tg._nll(model, *d2b._batch_tensors(states))
    if not math.isfinite(nll_batch):
        raise _fail("D2C_TERM_NON_FINITE",
                    f"step {step_label}: the batch production NLL is "
                    f"non-finite ({nll_batch!r})")
    per_slice = []
    with torch.no_grad():
        for st in states:
            try:
                _z, ldj, log_pz = d2b._encode_slice(model, st)
            except StageError as exc:
                raise _fail("D2C_ENCODE_FAILURE",
                            f"step {step_label}, slice "
                            f"{st['identity']!r}: encode terms failed "
                            f"({exc.error_code}: {exc.reason})")
            nll = -(log_pz + ldj)
            if not math.isfinite(nll):
                raise _fail("D2C_TERM_NON_FINITE",
                            f"step {step_label}, slice "
                            f"{st['identity']!r}: per-slice NLL is "
                            "non-finite")
            psnr, nmse = _z0_metrics(model, st)
            per_slice.append({"identity": st["identity"],
                              "nll": nll,
                              "nll_per_dim": nll / tg.ffr.FLOW_DIM_REAL,
                              "z0_psnr": psnr,
                              "z0_nmse_u": nmse})
    return {"step": step_label, "nll_batch": nll_batch,
            "per_slice": per_slice}


# ---------------------------------------------------------------------------
# Aggregation: G, R, locked-band classification, descriptive statistics.
# ---------------------------------------------------------------------------

def _classify(R: float) -> dict:
    """Locked bands (EXEC SS10.6 D2c): R <= 0.25 strong
    memorization-CONSISTENT evidence; R >= 0.75 strong transfer
    evidence for LIKELIHOOD GAIN; between = mixed. The phrasing is
    deliberate (review 2026-08-20): neither band is PROOF."""
    if R <= tinv.D2C_BAND_MEMORIZATION:
        return {"label": "strong_memorization_consistent",
                "note": "strong memorization-CONSISTENT evidence under "
                        "the registered D2c diagnostic -- NOT proof of "
                        "memorization: weak holdout gain can also "
                        "arise from train/holdout difficulty "
                        "differences or finite-sample effects"}
    if R >= tinv.D2C_BAND_TRANSFER:
        return {"label": "strong_transfer_likelihood_gain",
                "note": "strong transfer evidence for LIKELIHOOD GAIN "
                        "only -- not necessarily reconstruction "
                        "transfer"}
    return {"label": "mixed",
            "note": "between the locked 0.25/0.75 bands: mixed "
                    "evidence"}


def _quantile_stats(vals: np.ndarray) -> dict:
    """Frozen descriptive stat set for the per-slice delta_NLL/dim
    distribution: population std (ddof=0) and np.percentile default
    (linear) interpolation -- conventions frozen here so a future edit
    cannot silently change them."""
    return {"mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "q05": float(np.percentile(vals, 5)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "q95": float(np.percentile(vals, 95)),
            "max": float(vals.max())}


def _bootstrap_r(per_dim_deltas: np.ndarray, g_train: float) -> dict:
    """DESCRIPTIVE-ONLY bootstrap over the per-volume delta_NLL/dim
    values (resample volumes with replacement, B=10000, PCG64(3) --
    distinct from selection seed 1, D1 bank seed 0, JVP seed 2). Never
    alters the locked R classification (review 2026-08-20)."""
    rng = np.random.Generator(np.random.PCG64(tinv.D2C_BOOTSTRAP_SEED))
    n = int(per_dim_deltas.shape[0])
    idx = rng.integers(0, n, size=(tinv.D2C_BOOTSTRAP_N, n))
    r_b = per_dim_deltas[idx].mean(axis=1) / g_train
    return {"n_resamples": tinv.D2C_BOOTSTRAP_N,
            "rng": {"generator": "PCG64", "seed": tinv.D2C_BOOTSTRAP_SEED},
            "R_median": float(np.median(r_b)),
            "R_ci_2p5": float(np.percentile(r_b, 2.5)),
            "R_ci_97p5": float(np.percentile(r_b, 97.5)),
            "note": "DESCRIPTIVE ONLY -- does not alter the locked R "
                    "classification"}


def run_d2c_core(ctx, r0: dict, states: list) -> dict:
    """Two-state holdout measurement + aggregation, given the selected
    slice states. Dataset-independent: fixtures drive this with stub
    states; the production driver reaches it through run_d2c."""
    t0 = time.perf_counter()
    if ctx.state0 is None:
        raise _fail("D2C_STATE0_MISSING",
                    "the driver-owned step-0 state_dict is absent; D2c "
                    "needs the registered step-0 state for the swap")
    endpoints = r0.get("endpoints")
    try:
        nll0_reg = float(endpoints["initial"]["nll_batch_mean"])
        nll500_reg = float(endpoints["final"]["nll_batch_mean"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise _fail("D2C_ENDPOINTS_MISSING",
                    f"the R0 result lacks registered endpoint "
                    f"nll_batch_mean values: {exc}")
    dim = tg.ffr.FLOW_DIM_REAL
    delta_train = nll0_reg - nll500_reg
    if delta_train == 0.0:
        raise _fail("D2C_ZERO_TRAIN_GAIN",
                    "the registered train delta_NLL is exactly 0; R is "
                    "undefined")
    g_train = delta_train / dim

    state_identity = {}
    state_identity["pre_swap_step500"] = {
        "equal": True,
        "hash": _verify_state(ctx.model, r0["step500_state_hash"],
                              "pre-swap step500")}
    state500 = replay.capture_state(ctx.model)
    try:
        ctx.model.load_state_dict(ctx.state0)
    except Exception as exc:  # noqa: BLE001 -- typed boundary
        raise _fail("D2C_STATE_LOAD_FAILED",
                    f"loading the driver-owned step-0 state_dict "
                    f"failed: {type(exc).__name__}: {exc}")
    state_identity["step0_loaded"] = {
        "equal": True,
        "hash": _verify_state(ctx.model, r0["step0_state_hash"],
                              "step0 loaded")}
    meas0 = _measure_at_state(ctx.model, states, "step0")
    ctx.model.load_state_dict(state500)
    state_identity["step500_restored"] = {
        "equal": True,
        "hash": _verify_state(ctx.model, r0["step500_state_hash"],
                              "step500 restored")}
    meas500 = _measure_at_state(ctx.model, states, "step500")
    state_identity["post_measurement_step500"] = {
        "equal": True,
        "hash": _verify_state(ctx.model, r0["step500_state_hash"],
                              "post-measurement step500")}

    state_idents = [st["identity"] for st in states]
    for meas, lbl in ((meas0, "step0"), (meas500, "step500")):
        if [r["identity"] for r in meas["per_slice"]] != state_idents:
            raise _fail("D2C_IDENTITY_ORDER_MISMATCH",
                        f"the measured per-slice identities at {lbl} "
                        "do not equal the state list order exactly")

    per_slice = []
    for r0m, r5m in zip(meas0["per_slice"], meas500["per_slice"]):
        dnll = r0m["nll"] - r5m["nll"]
        per_slice.append({
            "identity": r0m["identity"],
            "step0": {k: v for k, v in r0m.items() if k != "identity"},
            "step500": {k: v for k, v in r5m.items() if k != "identity"},
            "delta": {
                "delta_nll": dnll,
                "delta_nll_per_dim": dnll / dim,
                "delta_psnr": r5m["z0_psnr"] - r0m["z0_psnr"],
                "delta_nmse_u": (None if r0m["z0_nmse_u"] is None
                                 else r0m["z0_nmse_u"]
                                 - r5m["z0_nmse_u"])}})
    deltas = np.array([r["delta"]["delta_nll"] for r in per_slice],
                      dtype=np.float64)
    per_dim = deltas / dim
    g_hold_per_slice = float(per_dim.mean())
    g_hold_batch = (meas0["nll_batch"] - meas500["nll_batch"]) / dim
    R = g_hold_per_slice / g_train
    classification = _classify(R)
    n_pos = int(np.count_nonzero(deltas > 0.0))
    n_zero = int(np.count_nonzero(deltas == 0.0))
    n_neg = int(np.count_nonzero(deltas < 0.0))
    psnr0 = np.array([r["step0"]["z0_psnr"] for r in per_slice])
    psnr500 = np.array([r["step500"]["z0_psnr"] for r in per_slice])
    dpsnr = np.array([r["delta"]["delta_psnr"] for r in per_slice])
    nmse_rows = [(r["step0"]["z0_nmse_u"], r["step500"]["z0_nmse_u"],
                  r["delta"]["delta_nmse_u"]) for r in per_slice
                 if r["step0"]["z0_nmse_u"] is not None]
    nmse_block = None
    if nmse_rows:
        a0 = np.array([r[0] for r in nmse_rows])
        a5 = np.array([r[1] for r in nmse_rows])
        ad = np.array([r[2] for r in nmse_rows])
        nmse_block = {"mean_step0": float(a0.mean()),
                      "mean_step500": float(a5.mean()),
                      "mean_delta": float(ad.mean()),
                      "median_delta": float(np.median(ad)),
                      "n_included": len(nmse_rows)}
    aggregate = {
        "n": len(per_slice),
        "NLL_step0_mean": meas0["nll_batch"],
        "NLL_step500_mean": meas500["nll_batch"],
        "delta_NLL_mean": float(deltas.mean()),
        "G_hold": g_hold_per_slice,
        "G_hold_batch": g_hold_batch,
        "holdout_reconciliation_error": g_hold_batch - g_hold_per_slice,
        "G_train": g_train,
        "R": R,
        "classification": classification,
        "n_positive_delta_nll": n_pos,
        "n_zero_delta_nll": n_zero,
        "n_negative_delta_nll": n_neg,
        "holdout_improved_fraction": n_pos / len(per_slice),
        "delta_nll_per_dim": _quantile_stats(per_dim),
        "psnr": {"mean_step0": float(psnr0.mean()),
                 "mean_step500": float(psnr500.mean()),
                 "mean_delta": float(dpsnr.mean()),
                 "median_delta": float(np.median(dpsnr))},
        "nmse_u": nmse_block,
        "bootstrap": _bootstrap_r(per_dim, g_train)}
    runtime = {"seconds": time.perf_counter() - t0,
               "n_slices": len(per_slice),
               "measure_calls": "2 states x (1 batch NLL + n encodes + "
                                "n z=0 decodes)",
               "note": "descriptive provenance, never scientific "
                       "routing evidence"}
    logger.info("[%s] D2c complete: G_hold=%.6f G_train=%.6f R=%.4f "
                "(%s), improved %d/%d, %.1f s", "SEQREF-TDIAG",
                g_hold_per_slice, g_train, R, classification["label"],
                n_pos, len(per_slice), runtime["seconds"])
    return {
        "spec": "EXEC SS10.6 D2c (locked 2026-08-15): volume-level "
                "held-out generalization; R = G_hold / G_train with "
                "G = delta_NLL/dim step-0 -> step-500",
        "purpose": "memorization-vs-transfer diagnostic of the TRAINED "
                   "model on unseen volumes; NO retraining -- the "
                   "registered step-0/step-500 states are evaluated on "
                   "the holdout slices",
        "routing": "R <= 0.25 => strong memorization-consistent "
                   "evidence => decision matrix: data/budget redesign "
                   "=> candidate v0.2; R >= 0.75 => strong transfer "
                   "evidence for likelihood gain; between = mixed. "
                   "ONLY the point-estimate R routes; PSNR/NMSE_u, "
                   "improved fraction and the bootstrap CI are "
                   "descriptive, never routed",
        "sign_convention": {
            "delta_nll": "step0 - step500 (positive = likelihood "
                         "improvement)",
            "delta_psnr": "step500 - step0 (positive = improvement)",
            "delta_nmse_u": "step0 - step500 (positive = improvement)",
            "dim": dim},
        "z_true_rule": "z_true = f(u_true | c) per slice via the "
                       "production encode (d2b._encode_slice verbatim)",
        "state_identity": {**state_identity,
                           "rule": "state_hash equality against the "
                                   "R0-registered step0/step500 hashes "
                                   "at every swap boundary"},
        "g_train_source": {
            "delta_nll": delta_train,
            "dim": dim,
            "value": g_train,
            "source": "registered R0/TINY endpoints (initial - final "
                      "nll_batch_mean); single authoritative source, "
                      "never recomputed from the D2b decomposition "
                      "(review 2026-08-20)"},
        "per_slice": per_slice,
        "aggregate": aggregate,
        "runtime": runtime}


def run_d2c(ctx, r0: dict, tiny_facts: dict, data_root: str,
            p4: dict) -> dict:
    """D2c driver entry: dataset construction + locked selection +
    state building + the two-state measurement. tiny_facts is the
    VERIFIED TINY artefact (its selection manifest is the exclusion
    source)."""
    tiny_selection = tiny_facts.get("selection")
    if not isinstance(tiny_selection, dict):
        raise _fail("D2C_TINY_SELECTION_MISSING",
                    "the verified TINY facts carry no selection block; "
                    "the exclusion set cannot be derived")
    ds = tg.FastMRISliceDataset(data_root, split="train", mode="eval")
    selection = select_holdout(ds, tiny_selection)
    states = build_holdout_states(ds, selection, p4, ctx.s_ref)
    block = run_d2c_core(ctx, r0, states)
    if ([r["identity"] for r in block["per_slice"]]
            != [r["identity"] for r in selection["selected"]]):
        raise _fail("D2C_IDENTITY_ORDER_MISMATCH",
                    "the measured per-slice identities do not equal "
                    "the locked selection draw order exactly")
    block["selection"] = selection
    return block
