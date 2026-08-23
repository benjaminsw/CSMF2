# SEQREF-V02E v0.2 -- scripts.v02_eval
# LIFETIME: KEEP
# =============================================================================
# Purpose: candidate v0.2 endpoint evaluator and facts assembler (V02SPEC
#          v0.1 SS4/SS5/SS8/SS9, V02PLAN v0.2 SS5/SS6). Consumes ONLY frozen
#          inputs: hash-verified manifests (SEQREF-V02M), the hash-verified
#          train record + four checkpoints (SEQREF-V02T), the EPHEMERAL
#          preflight report (SEQREF-V02P) and pinned parents (P0S s_ref,
#          P4/2 location_index, IMPL-B spline_b). Emits v02_facts.json
#          (schema seqref-v02-facts/1) -- evidence only, NO verdict field.
# Locked quantities (quoted from V02SPEC, never CLI-tunable):
#   * checkpoints at steps 0 / 1086 / 2172 / 3258; endpoints = step 0
#     (initial state, before any optimizer step) and step 3258 (final)
#   * G_train = mean per-slice (NLL_0 - NLL_final) / 13824 over the full
#     34,742-slice train population; G_hold over the 199 holdout slices;
#     R = G_hold / G_train
#   * V1: G_train >= 0.10; V2 (evaluated only if V1 passes): R >= 0.75
#     (bands 0.25/0.75); G_train == 0 => R is null, never coerced
#   * V3: holdout mean z=0 PSNR gain >= +1.0 dB AND
#     mean NMSE_u(final) <= 0.75 x mean NMSE_u(step0)
#   * bootstrap: ONE PCG64(3) stream, B = 10,000, over VOLUMES, ratio of
#     resampled arithmetic means; locked stream consumption order:
#     (1) holdout-gain index matrix, (2) train-gain index matrix,
#     (3) holdout-NMSE index matrix (each (B, n_volumes) via rng.integers)
#   * monitoring freeze (SS8): full-train NLL at steps 0/final ONLY;
#     holdout NLL + z=0 at all four checkpoints; PM at 0/final on the
#     frozen 32-slice D3 subset with the registered 128-latent bank;
#     D3 conditions C0-C3 at the FINAL state only; no full-train PM ever
#   * candidate classification (computed INTO v02_facts.json per SS9;
#     exists only after every validity gate passes, SS6):
#       V1 fail              -> LIKELIHOOD_LEARNING_NOT_ESTABLISHED
#       V1 pass, V2 fail     -> TRANSFER_NOT_SUPPORTED
#       V1+V2+V3 pass        -> PROMISING_DATA_BUDGET_REDESIGN
#       V1+V2 pass, V3 fail  -> LIKELIHOOD_TRANSFER_WITHOUT_RECONSTRUCTION_SUPPORT
# Taxonomy: exit 0 = evidence artefact complete; exit 2 = ERROR. There is
#   NO scientific BLOCK and NO exit 1 in this stage (V02SPEC SS6); every
#   failure path is logger.error + typed raise (V02Error, the single
#   identity from SEQREF-V02M); unexpected exceptions are remapped to
#   ERROR at the registered main() boundary. No fallback, no mock, no
#   placeholder, no silent pass.
# Reuse seams (verbatim, zero reimplementation):
#   * d2b._encode_slice            -- per-slice (z_true f32, ldj, log_pz)
#   * tg._build_slice_states       -- production slice-state construction
#     (identity cross-check against the manifest entries runs INSIDE it)
#   * tg._decode_z / tg._psnr / tg._nmse -- production z=0 metric engine
#   * d3.derangement / d3._measure_condition / d3._sign_counts -- the D3
#     machinery (run_d3 itself is pinned to the TINY n=8 context and is
#     NOT reused; the v0.2 driver loop here is new)
#   * estimators.z_diag_bank       -- the registered PCG64(0) (128, 13824)
#     float64->float32 posterior-mean bank
#   * v02_train.state_sha256       -- checkpoint state-hash verification
#   * preflight_parents.attach_semantic_hash / publish_stage -- publication
# Aggregation rule mirrored from the TDIAG D3 lock (documented, not
#   imported: d3._aggregate_condition would drag TINY-registered score
#   references into a v0.2 artefact): per-slice signed deltas vs C0 first,
#   then the arithmetic mean; delta = Ck - C0 (the signed perturbation
#   effect). v0.2 registers NO holdout exclusion: the TINY exclusion flag
#   is recorded for transparency, and a zero-energy NMSE denominator is a
#   validity ERROR, never a dropped slice.
# Environment-bound imports (torch, the dataset, TDIAG/parent modules)
#   are LAZY and TOTAL via _env(): ImportError propagates as ERROR; there
#   is no fallback path. Pure logic stays importable without the data
#   environment so the selftest can exercise it in isolation.
# Changelog (NEW in v0.1):
#   * Introduced under V02PLAN v0.2 (LOCKED 2026-08-21).
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from seqref_mri.scripts.v02_manifests import (D3_SUBSET_N, N_EPOCHS,
                                              N_HOLDOUT_VOLUMES,
                                              N_TRAIN_SLICES, V02Error,
                                              d3_monitor_positions,
                                              exposure_counts,
                                              manifest_sha256)

logger = logging.getLogger("seqref_mri.v02_eval")

__version__ = "0.2"
__abbr__ = "SEQREF-V02E"

FACTS_SCHEMA = "seqref-v02-facts/1"
FACTS_PREFIX = "v02_facts"
STAGE = "SEQREF-V02E"
EXIT_COMPLETE = 0
EXIT_ERROR = 2

CHECKPOINT_STEPS = (0, 1086, 2172, 3258)   # quoted from V02SPEC SS3
STEP0, FINAL_STEP = 0, 3258
FLOW_DIM = 13824                           # quoted; == ffr.FLOW_DIM_REAL
V1_FLOOR = 0.10                            # quoted from V02SPEC SS5
V2_BANDS = (0.25, 0.75)                    # quoted from V02SPEC SS5
V2_R_MIN = 0.75                            # quoted from V02SPEC SS5
V3_PSNR_MIN_DB = 1.0                       # quoted from V02SPEC SS5
V3_NMSE_RATIO_MAX = 0.75                   # quoted from V02SPEC SS5
BOOTSTRAP_SEED = 3                         # PCG64(3), quoted from SS5
BOOTSTRAP_B = 10000                        # quoted from SS5
TRAIN_CHUNK = 512                          # I/O memory bound only;
                                           # scientifically inert (order
                                           # preserved, identity-checked)

LABEL_LEARNING_NOT_ESTABLISHED = "LIKELIHOOD_LEARNING_NOT_ESTABLISHED"
LABEL_TRANSFER_NOT_SUPPORTED = "TRANSFER_NOT_SUPPORTED"
LABEL_PROMISING = "PROMISING_DATA_BUDGET_REDESIGN"
LABEL_WITHOUT_RECON = "LIKELIHOOD_TRANSFER_WITHOUT_RECONSTRUCTION_SUPPORT"


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V02Error(f"{code}: {message}")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _env() -> SimpleNamespace:
    """Total lazy import of every environment-bound dependency (the
    registered scientific host). ImportError propagates as ERROR; there
    is no fallback path."""
    try:
        from seqref_mri.tdiag import _bootstrap  # noqa: F401 --
        # registered path bootstrap (no PYTHONPATH): the campaign
        # preflight modules live in seqref_mri/src/ and are imported
        # under the LEGACY top-level names below; the bootstrap makes
        # them resolve to the SAME module objects TINY/TDIAG use (one
        # preflight_parents, one StageError identity, TDIAGT T10).
        import torch
        from torch.utils.data import DataLoader, Subset
        from preflight_io import utc_stamp
        from preflight_parents import (StageError, attach_semantic_hash,
                                       publish_stage)
        from seqref_mri.src import free_flow_runtime as ffr
        from seqref_mri.src.fastmri_data import FastMRISliceDataset
        from seqref_mri.scripts import tiny_gate as tg
        from seqref_mri.scripts.train_base import _collate
        from seqref_mri.scripts.v02_train import state_sha256
        from seqref_mri.tdiag import d2b, d3, estimators
    except ImportError as exc:
        _fail("ENV_IMPORT_FAILED",
              f"an environment-bound dependency is not importable in "
              f"this environment: {exc}")
    return SimpleNamespace(
        torch=torch, DataLoader=DataLoader, Subset=Subset,
        utc_stamp=utc_stamp, StageError=StageError,
        attach_semantic_hash=attach_semantic_hash,
        publish_stage=publish_stage, ffr=ffr,
        FastMRISliceDataset=FastMRISliceDataset, tg=tg, collate=_collate,
        state_sha256=state_sha256, d2b=d2b, d3=d3, estimators=estimators)


# ---------------------------------------------------------------------------
# Frozen-input loading and verification (sidecar = sha256 hex + newline,
# the SEQREF-V02M/V02T writer convention).
# ---------------------------------------------------------------------------

def _load_sidecar_json(path: Path, schema: str, code: str) -> dict:
    sidecar = Path(str(path) + ".sha256")
    if not path.exists() or not sidecar.exists():
        _fail(code, f"{path} or its sidecar does not exist; frozen "
              f"inputs must be materialized before evaluation")
    recorded = sidecar.read_text().split()[0].strip()
    actual = _file_sha256(path)
    if actual != recorded:
        _fail(code, f"{path.name}: file sha256 {actual[:12]}... != "
              f"sidecar {recorded[:12]}...; frozen-input drift is an "
              f"abort (V02SPEC SS12)")
    doc = json.loads(path.read_text())
    if doc.get("schema") != schema:
        _fail(code, f"{path.name}: schema {doc.get('schema')!r} != "
              f"{schema!r}")
    return doc


def _load_manifest(manifest_dir: Path, name: str, kind: str) -> dict:
    man = _load_sidecar_json(manifest_dir / name, "seqref-v02-manifest/1",
                             "MANIFEST_HASH_MISMATCH")
    if man.get("kind") != kind:
        _fail("MANIFEST_SCHEMA_INVALID",
              f"{name}: kind {man.get('kind')!r} != {kind!r}")
    body = {k: v for k, v in man.items() if k != "manifest_sha256"}
    if manifest_sha256(body) != man.get("manifest_sha256"):
        _fail("MANIFEST_HASH_MISMATCH",
              f"{name}: internal manifest_sha256 field does not match "
              f"the recomputed content hash")
    return man


def _load_train_record(path: str, manifests: list) -> dict:
    rec = _load_sidecar_json(Path(path), "seqref-v02-train-record/1",
                             "TRAIN_RECORD_MISMATCH")
    if int(rec.get("steps", -1)) != FINAL_STEP:
        _fail("TRAIN_RECORD_MISMATCH",
              f"train record declares {rec.get('steps')} steps; the "
              f"locked budget is {FINAL_STEP}")
    ckpts = rec.get("checkpoints")
    if not isinstance(ckpts, list) \
            or [int(c["step"]) for c in ckpts] != list(CHECKPOINT_STEPS):
        _fail("CHECKPOINT_SCHEDULE_MISMATCH",
              "train record checkpoint schedule diverges from the locked "
              f"{list(CHECKPOINT_STEPS)}")
    if list(rec.get("manifest_sha256", [])) != [m["manifest_sha256"]
                                                for m in manifests]:
        _fail("TRAIN_RECORD_MISMATCH",
              "train record manifest hashes != the verified epoch "
              "manifest hashes; training and evaluation would not refer "
              "to the same frozen data order")
    return rec


def _load_preflight(path: str) -> dict:
    """The EPHEMERAL preflight report (SEQREF-V02P writes no sidecar --
    its lifetime is EPHEMERAL per V02SPEC SS7, which governs over the
    V02PLAN v0.2 SS13 listing). Consumed as projection evidence only."""
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != "seqref-v02-preflight/1":
        _fail("PREFLIGHT_SCHEMA_INVALID",
              f"preflight schema {doc.get('schema')!r} != "
              f"'seqref-v02-preflight/1'")
    out = {}
    for key in ("projected_training_s", "projected_training_h",
                "ceiling_s", "projected_endpoint_eval_s",
                "projected_pm_bank_s", "peak_memory_bytes"):
        val = doc.get(key)
        if not isinstance(val, (int, float)) or isinstance(val, bool) \
                or not math.isfinite(val):
            _fail("PREFLIGHT_SCHEMA_INVALID",
                  f"preflight field {key!r} is missing or non-finite")
        out[key] = float(val)
    if out["projected_training_s"] > out["ceiling_s"]:
        _fail("PREFLIGHT_PROJECTION_INVALID",
              "the consumed preflight projects training beyond the 48 h "
              "ceiling; such a preflight may not seed a scientific run "
              "(V02SPEC SS7)")
    out["lifetime"] = "EPHEMERAL"
    out["note"] = ("projection evidence copied into this KEEP artefact; "
                   "the preflight artefact itself is EPHEMERAL and "
                   "carries no sidecar")
    return out


def _load_checkpoint(env: SimpleNamespace, run_root: Path, rec: dict,
                     spline_b: float):
    step = int(rec["step"])
    path = run_root / rec["file"]
    if not path.exists():
        _fail("CHECKPOINT_MISSING", f"{path} does not exist")
    if _file_sha256(path) != rec["file_sha256"]:
        _fail("CHECKPOINT_FILE_MISMATCH",
              f"{path.name}: file sha256 != the train-record pin")
    try:
        payload = env.torch.load(path, map_location="cpu",
                                 weights_only=True)
    except Exception as exc:  # torch.load raises several runtime types
        _fail("CHECKPOINT_UNREADABLE",
              f"{path.name}: {type(exc).__name__}: {exc}")
    if int(payload.get("step", -1)) != step \
            or payload.get("abbr") != "SEQREF-V02T":
        _fail("CHECKPOINT_PROVENANCE_MISMATCH",
              f"{path.name}: payload step/abbr "
              f"{payload.get('step')!r}/{payload.get('abbr')!r}")
    model = env.ffr.build_model(spline_b=spline_b)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, KeyError) as exc:
        _fail("CHECKPOINT_STATE_LOAD_FAILURE",
              f"{path.name}: {type(exc).__name__}: {exc}")
    actual = env.state_sha256(model)
    if actual != rec["state_sha256"]:
        _fail("STATE_HASH_MISMATCH",
              f"step {step}: reloaded state hash {actual[:12]}... != "
              f"the train-record pin {rec['state_sha256'][:12]}...; "
              f"state-hash mismatch is an abort (V02SPEC SS12)")
    model.eval()
    logger.info("[%s] checkpoint step=%d verified state=%s", __abbr__,
                step, actual[:12])
    return model


# ---------------------------------------------------------------------------
# Slice-state construction over manifest entries (identity-driven; the
# loader order == the manifest order and the identity cross-check runs
# inside tg._build_slice_states).
# ---------------------------------------------------------------------------

def _iter_state_batches(env: SimpleNamespace, ds, entries: list,
                        p4: dict, s_ref: float, chunk: int):
    index_of = {}
    for k, (path, sl) in enumerate(ds.index):
        index_of[(path.relative_to(ds.data_root).as_posix(), int(sl))] = k
    for lo in range(0, len(entries), chunk):
        part = entries[lo:lo + chunk]
        idx = []
        for e in part:
            key = (e["file"], int(e["slice_index"]))
            if key not in index_of:
                _fail("MANIFEST_ENTRY_UNKNOWN",
                      f"manifest entry {key} is not in the eval-mode "
                      f"dataset traversal index; frozen-input drift")
            idx.append(index_of[key])
        loader = env.DataLoader(env.Subset(ds, idx), batch_size=len(idx),
                                shuffle=False, num_workers=0,
                                collate_fn=env.collate)
        batch = next(iter(loader))
        sel = {"ordered_identities": [
            {"file": e["file"], "slice_index": int(e["slice_index"])}
            for e in part]}
        try:
            states = env.tg._build_slice_states(batch, sel, p4, s_ref)
        except env.StageError as exc:
            _fail("EVAL_STATE_BUILD_FAILURE",
                  f"entries[{lo}:{lo + len(part)}]: {exc.error_code}: "
                  f"{exc.reason}")
        yield states


def _collect_states(env, ds, entries, p4, s_ref, chunk) -> list:
    states = []
    for part in _iter_state_batches(env, ds, entries, p4, s_ref, chunk):
        states.extend(part)
    if len(states) != len(entries):
        _fail("STATE_COUNT_MISMATCH",
              f"built {len(states)} states for {len(entries)} manifest "
              f"entries")
    return states


def _encode(env: SimpleNamespace, model, st: dict, where: str) -> tuple:
    """d2b._encode_slice verbatim; StageError remapped to the v0.2
    single error identity."""
    try:
        return env.d2b._encode_slice(model, st)
    except env.StageError as exc:
        _fail("EVAL_ENCODE_FAILURE",
              f"{where}: {exc.error_code}: {exc.reason}")


def _z0_metrics(env: SimpleNamespace, model, st: dict,
                where: str) -> tuple:
    """(z=0 PSNR, z=0 NMSE_u) through the production decode/metric
    engine. v0.2 registers NO exclusion: a zero-energy NMSE denominator
    is a validity ERROR, never a dropped slice (V02SPEC SS6)."""
    if float(st["u_true_energy"]) == 0.0:
        _fail("VALIDITY_NMSE_UNDEFINED",
              f"{where}: slice {st['identity']!r} has zero free-space "
              f"energy; NMSE_u is undefined and never coerced")
    z0 = env.torch.zeros(1, env.ffr.FLOW_DIM_REAL)
    try:
        x0_c, u0 = env.tg._decode_z(model, z0, st)
        psnr = env.tg._psnr(x0_c.abs(), st["x_true_mag"])
        nmse = env.tg._nmse(u0, st["u_true"])
    except env.StageError as exc:
        _fail("EVAL_Z0_FAILURE",
              f"{where}: {exc.error_code}: {exc.reason}")
    return psnr, nmse


# ---------------------------------------------------------------------------
# Population measurements.
# ---------------------------------------------------------------------------

def _measure_train_endpoints(env, model0, model1, ds, entries, p4,
                             s_ref) -> dict:
    """Per-slice NLL/L_base/L_logdet at steps 0 and final over the FULL
    train population (monitoring freeze: no intermediate-train states).
    Streams in chunks; only float64 scalars survive."""
    n = len(entries)
    arr = {k: np.empty(n, dtype=np.float64)
           for k in ("nll0", "nll1", "base0", "base1", "ldj0", "ldj1")}
    off = 0
    for states in _iter_state_batches(env, ds, entries, p4, s_ref,
                                      TRAIN_CHUNK):
        for model, sfx in ((model0, "0"), (model1, "1")):
            with env.torch.no_grad():
                for j, st in enumerate(states):
                    _z, ldj, log_pz = _encode(
                        env, model, st, f"train entry {off + j} step {sfx}")
                    arr["nll" + sfx][off + j] = -log_pz - ldj
                    arr["base" + sfx][off + j] = -log_pz
                    arr["ldj" + sfx][off + j] = -ldj
        off += len(states)
        if off % (TRAIN_CHUNK * 8) == 0 or off == n:
            logger.info("[%s] train endpoints: %d/%d slices", __abbr__,
                        off, n)
    if off != n:
        _fail("STATE_COUNT_MISMATCH",
              f"measured {off} train slices, manifest declares {n}")
    return arr


def _measure_holdout(env, models: dict, states: list) -> dict:
    """Holdout NLL decomposition + z=0 metrics at ALL four checkpoints
    (monitoring freeze)."""
    n = len(states)
    nll = {s: np.empty(n, dtype=np.float64) for s in CHECKPOINT_STEPS}
    base = {s: np.empty(n, dtype=np.float64) for s in CHECKPOINT_STEPS}
    ldj = {s: np.empty(n, dtype=np.float64) for s in CHECKPOINT_STEPS}
    z0_psnr = {s: np.empty(n, dtype=np.float64) for s in CHECKPOINT_STEPS}
    z0_nmse = {s: np.empty(n, dtype=np.float64) for s in CHECKPOINT_STEPS}
    for step in CHECKPOINT_STEPS:
        model = models[step]
        with env.torch.no_grad():
            for j, st in enumerate(states):
                where = f"holdout slice {j} step {step}"
                _z, l, lpz = _encode(env, model, st, where)
                nll[step][j] = -lpz - l
                base[step][j] = -lpz
                ldj[step][j] = -l
                psnr, nmse = _z0_metrics(env, model, st, where)
                z0_psnr[step][j] = psnr
                z0_nmse[step][j] = nmse
        logger.info("[%s] holdout step=%d: nll_mean=%.4f "
                    "z0_psnr_mean=%.3f z0_nmse_mean=%.5f", __abbr__, step,
                    float(nll[step].mean()), float(z0_psnr[step].mean()),
                    float(z0_nmse[step].mean()))
    return {"nll": nll, "base": base, "ldj": ldj,
            "z0_psnr": z0_psnr, "z0_nmse": z0_nmse}


def _measure_condition(env, model, states, p, k, bank, where) -> dict:
    """d3._measure_condition verbatim (production batch NLL, per-slice
    conditioned encode, z=0 and PM metrics for one condition)."""
    try:
        return env.d3._measure_condition(model, states, p, k, bank)
    except env.StageError as exc:
        _fail("D3_MEASURE_FAILURE",
              f"{where}: {exc.error_code}: {exc.reason}")


def _d3_condition_summary(env, meas: dict, c0: dict) -> dict:
    """Locked aggregation (mirrored TDIAG D3 rule): per-slice signed
    deltas vs C0 first, then the arithmetic mean; delta = Ck - C0."""
    n = len(meas["per_slice"])
    if len(c0["per_slice"]) != n:
        _fail("D3_POPULATION_MISMATCH",
              f"condition {meas['condition']} has {n} slices, C0 has "
              f"{len(c0['per_slice'])}")
    dnll = np.array([meas["per_slice"][i]["nll"]
                     - c0["per_slice"][i]["nll"] for i in range(n)],
                    dtype=np.float64)
    dz0 = np.array([meas["per_slice"][i]["z0_psnr"]
                    - c0["per_slice"][i]["z0_psnr"] for i in range(n)],
                   dtype=np.float64)
    dpm = np.array([meas["per_slice"][i]["pm_psnr"]
                    - c0["per_slice"][i]["pm_psnr"] for i in range(n)],
                   dtype=np.float64)
    return {"condition": str(meas["condition"]), "n_slices": n,
            "nll_batch": float(meas["nll_batch"]),
            "delta_nll_batch_vs_c0": float(meas["nll_batch"]
                                           - c0["nll_batch"]),
            "mean_delta_nll": float(dnll.mean()),
            "mean_delta_psnr": float(dz0.mean()),
            "mean_delta_pm_psnr": float(dpm.mean()),
            "sign_counts": {"delta_nll": env.d3._sign_counts(dnll),
                            "delta_z0_psnr": env.d3._sign_counts(dz0),
                            "delta_pm_psnr": env.d3._sign_counts(dpm)}}


# ---------------------------------------------------------------------------
# Pure endpoint statistics (selftest exercises these in isolation).
# ---------------------------------------------------------------------------

def _volume_groups(entries: list, values: np.ndarray) -> list:
    """Per-volume gain arrays in first-appearance order over the entry
    list (the bootstrap resamples VOLUMES, V02SPEC SS5)."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(entries),):
        _fail("POPULATION_MISMATCH",
              f"{values.shape[0]} values for {len(entries)} entries")
    groups: dict[str, list] = {}
    order: list[str] = []
    for e, v in zip(entries, values):
        f = e["file"]
        if f not in groups:
            groups[f] = []
            order.append(f)
        groups[f].append(float(v))
    return [np.asarray(groups[f], dtype=np.float64) for f in order]


def _summ(samples: np.ndarray) -> dict:
    return {"mean": float(np.mean(samples)),
            "ci95": [float(v) for v in
                     np.percentile(samples, [2.5, 97.5])]}


def _bootstrap(g_train_by_vol: list, g_hold: np.ndarray,
               nmse0: np.ndarray, nmse1: np.ndarray,
               *, seed: int = BOOTSTRAP_SEED, B: int = BOOTSTRAP_B) -> dict:
    """Locked bootstrap (V02SPEC SS5): ONE PCG64(seed) stream, B resamples
    over volumes, ratio of resampled arithmetic means. Stream consumption
    order (locked, pinned by selftest): (1) holdout-gain index matrix,
    (2) train-gain index matrix, (3) holdout-NMSE index matrix."""
    g_hold = np.asarray(g_hold, dtype=np.float64)
    nmse0 = np.asarray(nmse0, dtype=np.float64)
    nmse1 = np.asarray(nmse1, dtype=np.float64)
    if g_hold.ndim != 1 or g_hold.size == 0 \
            or nmse0.shape != g_hold.shape or nmse1.shape != g_hold.shape:
        _fail("BOOTSTRAP_INPUT_INVALID",
              "holdout gain/NMSE vectors must be matching non-empty 1-D "
              "arrays (one slice per volume)")
    for name, a in (("g_hold", g_hold), ("nmse0", nmse0),
                    ("nmse1", nmse1)):
        if not np.isfinite(a).all():
            _fail("BOOTSTRAP_INPUT_NON_FINITE",
                  f"{name} contains non-finite values")
    if not g_train_by_vol:
        _fail("BOOTSTRAP_INPUT_INVALID", "no train volume groups")
    vol_sums = np.array([float(np.sum(v)) for v in g_train_by_vol],
                        dtype=np.float64)
    vol_counts = np.array([float(v.size) for v in g_train_by_vol],
                          dtype=np.float64)
    if not np.isfinite(vol_sums).all() or np.any(vol_counts <= 0):
        _fail("BOOTSTRAP_INPUT_INVALID",
              "train volume groups are empty or non-finite")
    n_h, n_t = g_hold.size, vol_sums.size
    rng = np.random.Generator(np.random.PCG64(seed))
    hold_idx = rng.integers(0, n_h, size=(B, n_h))
    train_idx = rng.integers(0, n_t, size=(B, n_t))
    nmse_idx = rng.integers(0, n_h, size=(B, n_h))
    g_hold_b = g_hold[hold_idx].mean(axis=1)
    g_train_b = (vol_sums[train_idx].sum(axis=1)
                 / vol_counts[train_idx].sum(axis=1))
    if np.any(g_train_b == 0.0):
        _fail("BOOTSTRAP_DENOMINATOR_ZERO",
              "a resampled G_train mean is exactly 0.0; R is undefined "
              "for that resample and no coercion is permitted")
    r_b = g_hold_b / g_train_b
    nmse0_b = nmse0[nmse_idx].mean(axis=1)
    if np.any(nmse0_b == 0.0):
        _fail("BOOTSTRAP_DENOMINATOR_ZERO",
              "a resampled step-0 NMSE_u mean is exactly 0.0; the V3 "
              "ratio is undefined for that resample and no coercion is "
              "permitted")
    ratio_b = nmse1[nmse_idx].mean(axis=1) / nmse0_b
    return {"generator": "PCG64", "seed": int(seed), "B": int(B),
            "unit": "volume",
            "estimator": "ratio of resampled arithmetic means",
            "stream_order": ["holdout_gain", "train_gain",
                             "holdout_nmse"],
            "g_train": _summ(g_train_b), "g_hold": _summ(g_hold_b),
            "r": _summ(r_b), "nmse_ratio": _summ(ratio_b)}


def _transfer_block(g_train_slices: np.ndarray, g_hold_slices: np.ndarray,
                    boot: dict) -> dict:
    """V1/V2 with the registered guards (V02SPEC SS5): V2 is evaluated
    only if V1 passes; G_train == 0 => R is null, never coerced."""
    g_train = float(np.mean(g_train_slices))
    g_hold = float(np.mean(g_hold_slices))
    if not (math.isfinite(g_train) and math.isfinite(g_hold)):
        _fail("ENDPOINT_NON_FINITE",
              f"G_train={g_train!r} G_hold={g_hold!r}")
    r = None if g_train == 0.0 else g_hold / g_train
    v1_pass = bool(g_train >= V1_FLOOR)
    if not v1_pass:
        v2 = {"r_min": V2_R_MIN, "bands": list(V2_BANDS), "pass": None,
              "guard": "V2 is evaluated only if V1 passes (V02SPEC SS5); "
                       "V1 did not pass"}
    elif r is None:
        v2 = {"r_min": V2_R_MIN, "bands": list(V2_BANDS), "pass": None,
              "guard": "G_train == 0 => R is null, never coerced "
                       "(V02SPEC SS5)"}
    else:
        v2 = {"r_min": V2_R_MIN, "bands": list(V2_BANDS),
              "pass": bool(r >= V2_R_MIN), "guard": None}
    r_boot = boot["r"] if r is not None else {
        "null": True,
        "reason": "G_train == 0.0; R and its bootstrap are null, never "
                  "coerced (V02SPEC SS5)"}
    return {"g_train": g_train, "g_hold": g_hold, "r": r,
            "v1": {"floor": V1_FLOOR, "pass": v1_pass},
            "v2": v2,
            "bootstrap": {"generator": boot["generator"],
                          "seed": boot["seed"], "B": boot["B"],
                          "unit": boot["unit"],
                          "estimator": boot["estimator"],
                          "stream_order": boot["stream_order"],
                          "g_train": boot["g_train"],
                          "g_hold": boot["g_hold"], "r": r_boot}}


def _v3_block(z0_psnr: dict, z0_nmse: dict, boot: dict) -> dict:
    """V3 (V02SPEC SS5): holdout mean z=0 PSNR gain >= +1.0 dB AND
    mean NMSE_u(final) <= 0.75 x mean NMSE_u(step0)."""
    psnr_gain = float(np.mean(z0_psnr[FINAL_STEP])
                      - np.mean(z0_psnr[STEP0]))
    nmse0_mean = float(np.mean(z0_nmse[STEP0]))
    nmse1_mean = float(np.mean(z0_nmse[FINAL_STEP]))
    if nmse0_mean == 0.0:
        _fail("VALIDITY_NMSE_UNDEFINED",
              "holdout mean step-0 NMSE_u is exactly 0.0; the V3 ratio "
              "is undefined and never coerced")
    nmse_ratio = nmse1_mean / nmse0_mean
    for name, v in (("psnr_gain", psnr_gain), ("nmse_ratio", nmse_ratio)):
        if not math.isfinite(v):
            _fail("ENDPOINT_NON_FINITE", f"V3 {name} is {v!r}")
    return {"psnr_gain_db": psnr_gain, "psnr_min_db": V3_PSNR_MIN_DB,
            "nmse_ratio": nmse_ratio, "nmse_ratio_max": V3_NMSE_RATIO_MAX,
            "nmse_ratio_bootstrap_mean": boot["nmse_ratio"]["mean"],
            "nmse_ratio_bootstrap_ci95": boot["nmse_ratio"]["ci95"],
            "pass": bool(psnr_gain >= V3_PSNR_MIN_DB
                         and nmse_ratio <= V3_NMSE_RATIO_MAX)}


def _classify(v1_pass: bool, v2_pass, v3_pass: bool) -> str:
    """The locked candidate-classification tree (V02SPEC SS5); computed
    only after every validity gate has passed (SS6)."""
    if not v1_pass:
        return LABEL_LEARNING_NOT_ESTABLISHED
    if v2_pass is not True:
        return LABEL_TRANSFER_NOT_SUPPORTED
    return LABEL_PROMISING if v3_pass else LABEL_WITHOUT_RECON


def _decomposition_shares(base0: np.ndarray, base1: np.ndarray,
                          ldj0: np.ndarray, ldj1: np.ndarray) -> dict:
    """D2b endpoint decomposition shares over one full population
    (same-encode byproduct, V02SPEC SS4/SS8). A zero endpoint-change
    denominator yields a DEFINED null with reason -- never a coerced
    share, never a dropped population."""
    d_base = float(np.mean(base0 - base1))
    d_ldj = float(np.mean(ldj0 - ldj1))
    d_nll = d_base + d_ldj
    out = {"mean_delta_l_base": d_base, "mean_delta_l_logdet": d_ldj,
           "mean_delta_nll": d_nll,
           "rule": "share = mean endpoint change of the term / mean "
                   "endpoint NLL change (step0 - final; positive = "
                   "improvement)"}
    if d_nll == 0.0:
        out["base_share_pct"] = None
        out["logdet_share_pct"] = None
        out["null_reason"] = ("mean endpoint NLL change is exactly 0.0; "
                              "shares are undefined (defined null, never "
                              "coerced)")
        return out
    out["base_share_pct"] = 100.0 * d_base / d_nll
    out["logdet_share_pct"] = 100.0 * d_ldj / d_nll
    out["null_reason"] = None
    return out


def _no_verdict_scan(node, path: str) -> None:
    """Evidence-only invariant: no 'verdict' key anywhere in the facts
    (V02SPEC SS9 forbids a campaign verdict field)."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "verdict":
                _fail("FACTS_SCHEMA_VIOLATION",
                      f"a verdict key entered the v0.2 facts at "
                      f"{path}.verdict; the stage is evidence-only by "
                      f"preregistration")
            _no_verdict_scan(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _no_verdict_scan(v, f"{path}[{i}]")


# ---------------------------------------------------------------------------
# Top-level run.
# ---------------------------------------------------------------------------

def run(cfg: dict) -> dict:
    """The evaluation run. cfg keys: data_root, manifest_dir,
    train_record, run_root, preflight, p0s_facts, p4_stats2, implb_facts,
    out_dir. Every scientific constant is locked in this module and its
    frozen inputs, never in cfg."""
    env = _env()
    env.torch.set_num_threads(1)
    t0 = time.time()
    man_dir = Path(cfg["manifest_dir"])

    manifests = [_load_manifest(man_dir, f"v02_epoch{ep}_manifest.json",
                                "train_epoch") for ep in range(N_EPOCHS)]
    for ep, man in enumerate(manifests):
        if int(man["epoch"]) != ep \
                or int(man["n_slices"]) != N_TRAIN_SLICES:
            _fail("MANIFEST_CONTENT_MISMATCH",
                  f"epoch {ep}: manifest declares epoch="
                  f"{man.get('epoch')}, n_slices={man.get('n_slices')}")
    holdman = _load_manifest(man_dir, "v02_holdout_manifest.json",
                             "holdout")
    if int(holdman["n_volumes"]) != N_HOLDOUT_VOLUMES \
            or len(holdman["entries"]) != N_HOLDOUT_VOLUMES:
        _fail("MANIFEST_CONTENT_MISMATCH",
              f"holdout manifest declares "
              f"{holdman.get('n_volumes')} volumes / "
              f"{len(holdman['entries'])} entries; locked "
              f"{N_HOLDOUT_VOLUMES}")
    d3man = _load_manifest(man_dir, "v02_d3_monitor_manifest.json",
                           "d3_monitor")
    if int(d3man["n"]) != D3_SUBSET_N \
            or len(d3man["entries"]) != D3_SUBSET_N:
        _fail("MANIFEST_CONTENT_MISMATCH",
              "D3 monitor manifest size diverges from the locked "
              f"{D3_SUBSET_N}")

    # Validity gate: re-derive the frozen D3 draw and re-bind it to the
    # verified holdout manifest (draw drift = ERROR, never a redraw).
    positions = [int(p) for p in d3_monitor_positions()]
    drawn = [int(e["holdout_position"]) for e in d3man["entries"]]
    if drawn != positions or [int(e["draw_rank"])
                              for e in d3man["entries"]] != list(
                                  range(D3_SUBSET_N)):
        _fail("D3_DRAW_MISMATCH",
              "the D3 monitor manifest does not reproduce the frozen "
              "PCG64(2).choice(199, 32, replace=False) draw in order")
    for e in d3man["entries"]:
        h = holdman["entries"][int(e["holdout_position"])]
        if e["file"] != h["file"] \
                or int(e["slice_index"]) != int(h["slice_index"]):
            _fail("D3_SUBSET_NOT_IN_HOLDOUT",
                  f"D3 entry {e['file']!r}/{e['slice_index']} is not the "
                  f"holdout entry at position {e['holdout_position']}")

    # Validity gate: exposure recount from the verified manifests.
    counts = exposure_counts(
        [[int(e["dataset_index"]) for e in m["entries"]]
         for m in manifests], N_TRAIN_SLICES)
    if not np.all(counts == N_EPOCHS):
        _fail("EXPOSURE_CAP_VIOLATION",
              f"{int(np.sum(counts != N_EPOCHS))} slices deviate from "
              f"exactly {N_EPOCHS} exposures")

    record = _load_train_record(cfg["train_record"], manifests)
    preflight = _load_preflight(cfg["preflight"])

    try:
        p4 = env.ffr.load_p4s2_parent(cfg["p4_stats2"])
        implb = env.ffr.load_implb_parent(cfg["implb_facts"])
        env.ffr.require_spline_b(implb["spline_b"])
        s_ref = env.tg._s_ref_from_p0s(cfg["p0s_facts"])
    except (env.ffr.FreeFlowError, env.StageError) as exc:
        _fail("PARENT_VERIFICATION_FAILED", f"{type(exc).__name__}: {exc}")
    logger.info("[%s] parents pinned: P4/2 %s | IMPL-B %s | s_ref=%.6g",
                __abbr__, p4["file_sha256"][:12],
                implb["file_sha256"][:12], s_ref)

    try:
        ds_train = env.FastMRISliceDataset(cfg["data_root"], split="train",
                                           mode="eval")
        ds_val = env.FastMRISliceDataset(cfg["data_root"], split="val",
                                         mode="eval")
    except Exception as exc:  # the dataset ctor is a typed, logging
        _fail("DATA_PREMISE_FAILURE", f"{type(exc).__name__}: {exc}")

    ckpt_recs = {int(c["step"]): c for c in record["checkpoints"]}
    models = {step: _load_checkpoint(env, Path(cfg["run_root"]),
                                     ckpt_recs[step], implb["spline_b"])
              for step in CHECKPOINT_STEPS}

    # Endpoint measurements (monitoring freeze, V02SPEC SS8).
    train_entries = manifests[0]["entries"]
    tr = _measure_train_endpoints(env, models[STEP0], models[FINAL_STEP],
                                  ds_train, train_entries, p4, s_ref)
    hold_states = _collect_states(env, ds_val, holdman["entries"], p4,
                                  s_ref, N_HOLDOUT_VOLUMES)
    hold = _measure_holdout(env, models, hold_states)

    # D3 monitor + PM on the frozen 32-slice subset (bank registered).
    d3_states = _collect_states(env, ds_val, d3man["entries"], p4, s_ref,
                                D3_SUBSET_N)
    bank_rec = env.estimators.z_diag_bank()
    bank = bank_rec["bank"]
    p = env.d3.derangement(D3_SUBSET_N)
    pm0 = _measure_condition(env, models[STEP0], d3_states, p, 0, bank,
                             "step0-C0")
    conds = [_measure_condition(env, models[FINAL_STEP], d3_states, p, k,
                                bank, f"final-C{k}") for k in range(4)]

    # Endpoint statistics.
    g_train_slices = (tr["nll0"] - tr["nll1"]) / float(FLOW_DIM)
    g_hold_slices = (hold["nll"][STEP0] - hold["nll"][FINAL_STEP]) \
        / float(FLOW_DIM)
    vol_gains = _volume_groups(train_entries, g_train_slices)
    if len(vol_gains) == 0:
        _fail("POPULATION_MISMATCH", "no train volume groups formed")
    boot = _bootstrap(vol_gains, g_hold_slices, hold["z0_nmse"][STEP0],
                      hold["z0_nmse"][FINAL_STEP])
    transfer = _transfer_block(g_train_slices, g_hold_slices, boot)
    transfer["v3"] = _v3_block(hold["z0_psnr"], hold["z0_nmse"], boot)
    label = _classify(transfer["v1"]["pass"], transfer["v2"]["pass"],
                      transfer["v3"]["pass"])

    d3_block = {
        "at": "final state only (V02SPEC SS8); PM additionally at step 0 "
              "under C0 (own/own), the production conditioning",
        "subset_manifest_sha256": d3man["manifest_sha256"],
        "bank": {"rule": "np.random.Generator(np.random.PCG64(0))."
                         "standard_normal(size=(128, 13824), dtype="
                         "float64).astype(float32) -- the registered "
                         "TDIAG bank, reused verbatim",
                 "bank_sha256": bank_rec["bank_sha256"],
                 "manifest_sha256": bank_rec["manifest_sha256"]},
        "derangement": "p(i) = (i+1) mod 32 (d3.derangement verbatim)",
        "aggregation_rule": "per-slice signed deltas vs C0 first, then "
                            "the arithmetic mean; delta = Ck - C0",
        "conditions": [_d3_condition_summary(env, conds[k], conds[0])
                       for k in range(4)],
        "pm": {"step0": {"pm_psnr_mean": float(np.mean(
                    [s["pm_psnr"] for s in pm0["per_slice"]])),
                "pm_nmse_u_mean": float(np.mean(
                    [s["pm_nmse_u"] for s in pm0["per_slice"]]))},
               "final": {"pm_psnr_mean": float(np.mean(
                    [s["pm_psnr"] for s in conds[0]["per_slice"]])),
                "pm_nmse_u_mean": float(np.mean(
                    [s["pm_nmse_u"] for s in conds[0]["per_slice"]]))}}}

    per_ckpt = [{"step": step,
                 "nll_mean": float(np.mean(hold["nll"][step])),
                 "z0_psnr_mean": float(np.mean(hold["z0_psnr"][step])),
                 "z0_nmse_u_mean": float(np.mean(hold["z0_nmse"][step]))}
                for step in CHECKPOINT_STEPS]

    facts = {
        "schema": FACTS_SCHEMA,
        "script": f"{__abbr__} v{__version__}",
        "spec": "SEQREF-MRI-V02SPEC v0.1 (preregistered 2026-08-21)",
        "plan": "SEQREF-MRI-V02PLAN v0.2 (LOCKED 2026-08-21)",
        "preflight": preflight,
        "train_manifest": {
            "epoch_manifest_sha256": [m["manifest_sha256"]
                                      for m in manifests],
            "n_epochs": N_EPOCHS, "n_slices": N_TRAIN_SLICES,
            "eval_order": "epoch-0 manifest entry order (frozen); order "
                          "is scientifically inert for the full-"
                          "population means and is identity-checked"},
        "holdout_manifest": {
            "manifest_sha256": holdman["manifest_sha256"],
            "n_volumes": N_HOLDOUT_VOLUMES,
            "selection": holdman["selection"]},
        "exposure_accounting": {
            "repetition_cap": N_EPOCHS,
            "per_slice_exposures": {"min": int(counts.min()),
                                    "max": int(counts.max())},
            "recount_source": "verified epoch manifests (recomputed at "
                              "evaluation time)",
            "violated": False},
        "endpoint_measurements": {
            "train": {"n_slices": N_TRAIN_SLICES,
                      "nll_step0": tr["nll0"].tolist(),
                      "nll_final": tr["nll1"].tolist(),
                      "l_base_step0": tr["base0"].tolist(),
                      "l_base_final": tr["base1"].tolist(),
                      "l_logdet_step0": tr["ldj0"].tolist(),
                      "l_logdet_final": tr["ldj1"].tolist(),
                      "mean_gain_per_dim": transfer["g_train"]},
            "holdout": {"n_volumes": N_HOLDOUT_VOLUMES,
                        "nll_step0": hold["nll"][STEP0].tolist(),
                        "nll_final": hold["nll"][FINAL_STEP].tolist(),
                        "l_base_step0": hold["base"][STEP0].tolist(),
                        "l_base_final": hold["base"][FINAL_STEP].tolist(),
                        "l_logdet_step0": hold["ldj"][STEP0].tolist(),
                        "l_logdet_final": hold["ldj"][FINAL_STEP].tolist(),
                        "mean_gain_per_dim": transfer["g_hold"],
                        "per_checkpoint": per_ckpt}},
        "v1_v2_v3": transfer,
        "secondary_monitoring": {
            "d2b_decomposition": {
                "train": _decomposition_shares(tr["base0"], tr["base1"],
                                               tr["ldj0"], tr["ldj1"]),
                "holdout": _decomposition_shares(
                    hold["base"][STEP0], hold["base"][FINAL_STEP],
                    hold["ldj"][STEP0], hold["ldj"][FINAL_STEP])},
            "d3_monitor": d3_block,
            "monitoring_freeze": "full-train NLL at steps 0/final only; "
                                 "holdout NLL + z=0 at all four "
                                 "checkpoints; PM at 0/final on the "
                                 "frozen 32-slice subset; no full-train "
                                 "PM ever (V02SPEC SS8)"},
        "validity": {
            "taxonomy": "exit 0 = evidence artefact complete; exit 2 = "
                        "ERROR; no scientific BLOCK and no exit 1 exist "
                        "in this stage (V02SPEC SS6)",
            "manifest_sidecars_verified": [
                f"v02_epoch{ep}_manifest.json" for ep in range(N_EPOCHS)]
                + ["v02_holdout_manifest.json",
                   "v02_d3_monitor_manifest.json"],
            "train_record_sha256_verified": True,
            "checkpoint_state_sha256_verified": {
                str(step): ckpt_recs[step]["state_sha256"]
                for step in CHECKPOINT_STEPS},
            "d3_draw_rederived_match": True,
            "exposure_recount_all_exactly_3": True,
            "non_finite_events": 0,
            "holdout_excluded_flag_count": int(sum(
                bool(st["excluded"]) for st in hold_states)),
            "exclusion_note": "the TINY-registered exclusion flag is "
                              "recorded for transparency; v0.2 registers "
                              "NO exclusion -- a zero-energy NMSE "
                              "denominator is a validity ERROR, never a "
                              "dropped slice"},
        "candidate_classification": {
            "label": label,
            "rule": "V1 fail -> LIKELIHOOD_LEARNING_NOT_ESTABLISHED; "
                    "V1 pass + V2 fail -> TRANSFER_NOT_SUPPORTED; "
                    "V1+V2+V3 pass -> PROMISING_DATA_BUDGET_REDESIGN; "
                    "V1+V2 pass + V3 fail -> "
                    "LIKELIHOOD_TRANSFER_WITHOUT_RECONSTRUCTION_SUPPORT "
                    "(V02SPEC SS5); computed only after every validity "
                    "gate passed (SS6)",
            "inputs": {"v1_pass": transfer["v1"]["pass"],
                       "v2_pass": transfer["v2"]["pass"],
                       "v3_pass": transfer["v3"]["pass"]}},
        "run": {"utc": env.utc_stamp(), "argv": sys.argv,
                "wall_clock_s": time.time() - t0},
    }
    _no_verdict_scan(facts, "facts")
    semantic = {k: v for k, v in facts.items() if k != "run"}
    env.attach_semantic_hash(facts, semantic)

    out_dir = str(Path(cfg["out_dir"]))
    auth = Path(out_dir) / f"{FACTS_PREFIX}.json"
    if auth.exists():
        sidecar = Path(str(auth) + ".sha256")
        if not sidecar.exists() or _file_sha256(auth) != \
                sidecar.read_text().split()[0].strip():
            _fail("RERUN_PRIOR_ARTEFACT_MISMATCH",
                  "an authoritative v02_facts.json exists but fails its "
                  "own sidecar verification; no sibling is published "
                  "next to an unverifiable artefact")
        prior = json.loads(auth.read_text())
        if prior.get("semantic_sha256") != facts["semantic_sha256"]:
            _fail("SEMANTIC_RERUN_MISMATCH",
                  "the existing authoritative v02_facts.json has "
                  f"semantic {str(prior.get('semantic_sha256'))[:12]}... "
                  f"!= this run's {facts['semantic_sha256'][:12]}...; a "
                  f"scientifically different rerun may NOT publish "
                  f"silently alongside")
    try:
        path, sha = env.publish_stage(facts, out_dir, FACTS_PREFIX, STAGE)
    except env.StageError as exc:
        _fail("PUBLICATION_FAILURE", f"{exc.error_code}: {exc.reason}")
    logger.info("[%s] v0.2 facts published %s sha256=%s; classification "
                "%s (V1=%s V2=%s V3=%s)", __abbr__, path, sha[:12], label,
                transfer["v1"]["pass"], transfer["v2"]["pass"],
                transfer["v3"]["pass"])
    return facts


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"{__abbr__} v{__version__} -- candidate v0.2 "
                    f"endpoint evaluator / facts assembler (V02SPEC "
                    f"SS4/SS5/SS8/SS9)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--train-record", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--preflight", required=True)
    ap.add_argument("--p0s-facts", required=True)
    ap.add_argument("--p4-stats2", required=True)
    ap.add_argument("--implb-facts", required=True)
    ap.add_argument("--out-dir", required=True)
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
        return EXIT_ERROR
    except Exception:  # noqa: BLE001 -- the registered boundary: no
        logger.exception("[%s] unexpected runtime failure", __abbr__)
        return EXIT_ERROR                    # exception may exit 1
    return EXIT_COMPLETE


if __name__ == "__main__":
    raise SystemExit(main())
