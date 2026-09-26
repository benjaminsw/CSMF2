# SEQREF-V02G v0.3 -- scripts.v02_readout_diag
# LIFETIME: KEEP
"""Authoritative V02 readout-diagnostic driver (V02AMEND v1.0 LOCKED,
LudWiki id 147).

The locked experiment is strictly R0 vs R1:

  R0 -- z=0: the existing V02 reconstruction rule; reference only.
  R1 -- posterior mean: the registered 128-sample PCG64(0) latent bank,
        decoded through the production decoder, combined by the
        registered u-space averaging rule (complex mean of the physical
        free vectors BEFORE image formation; one image formation).

Population: exactly the 199 frozen holdout identities (manifest order;
no redraw, no substitution, no post-hoc exclusion). Endpoints: step 0
and final step 3258. Primary comparison -- final R1 vs final R0 ONLY:

  D1 pass <=> OBSERVED mean paired dPSNR >= +0.50 dB
             AND paired volume-bootstrap CI95 lower > 0
  D2 pass <=> OBSERVED mean NMSE_u ratio <= 0.95
             AND paired volume-bootstrap CI95 upper < 1.0

R1 provides reconstruction-readout support <=> D1 AND D2 both pass.
Routing: READOUT_LIMITATION_SUPPORTED (detail
POSTERIOR_READOUT_SUPPORT) / TESTED_READOUTS_DO_NOT_RECOVER_-
RECONSTRUCTION_SUPPORT / ERROR (validity failure; no partial
classification).

CRITICAL acceptance wiring (reviewer-registered): the practical-effect
conjuncts consume the OBSERVED statistics; the bootstrap supplies ONLY
the CI95 bounds. The mean of bootstrap replicates is a diagnostic,
never an acceptance input (for the nonlinear ratio statistic the
replicate mean differs from the observed ratio of means).

Dependency architecture: this driver reuses the CANONICAL production
seams (v02e loaders/measurement, tiny_gate decoder/metrics, d3 bank
decode, estimators bank/image formation, preflight publication) and
the neutral shared-helper module scripts/v02_readout_common.py
(SEQREF-V02C). It NEVER imports from the V02H selftest harness: V02H
is an evidence/provenance dependency (the Outcome-A removal rationale,
sha-pinned), not an implementation foundation.

Hard pre-classification reproduction gates (v1.0 SS14 gates 9/11):
  * R0 z=0 reproduction against the EXACT frozen V02E quantities --
    endpoint_measurements.holdout.per_checkpoint[] z0_psnr_mean /
    z0_nmse_u_mean at steps 0 and 3258, the arithmetic-mean-over-the-
    199-registered-holdout-identities aggregation (v02_facts.json,
    schema seqref-v02-facts/1) -- within 1e-3 dB PSNR / 1e-4 relative
    NMSE_u ((measured-anchor)/anchor);
  * D3-subset PM reproduction against the registered PM anchors
    within the same tolerances.
Any gate failure is ERROR, never a negative scientific result.

Timing/probe semantics (reviewer-registered): the EPHEMERAL prerun
timing probe is feasibility evidence with its own provenance; the
authoritative run records its own measured telemetry. Neither enters
the semantic hash: timestamps, wall-clock timings, peak-memory
figures, host/environment strings, paths, and the R1 sample-norm
diagnostics are all OUTSIDE the semantic scope, so a scientifically
identical sibling rerun reproduces semantic_sha256.

Exit taxonomy (v1.0 SS14): exit 0 = evidence complete; exit 2 =
ERROR; no exit 1. No scientific PASS/BLOCK verdict exists; the facts
contain no "verdict" key. Every failure path logger.error(...) +
typed raise. No fallback, no mock, no placeholder, no silent pass.

Modes:
  --mode fixtures      assembly selftests (synthetic inputs only; the
                       real holdout is never touched)
  --mode probe         non-scientific timing/feasibility probe
                       (EPHEMERAL output; prerun evidence)
  --mode authoritative the locked diagnostic (requires all frozen
                       inputs; publishes v02_readout_facts.json,
                       schema seqref-v02rd-facts/1)
"""

import argparse
import hashlib
import json
import logging
import math
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from seqref_mri.scripts import v02_eval as v02e
from seqref_mri.scripts import v02_readout_common as v02c
from seqref_mri.scripts import v02_manifests as v02m

logger = logging.getLogger(__name__)

__abbr__ = "SEQREF-V02G"
__version__ = "0.3"

STAGE = "v02_readout_diag"
FACTS_PREFIX = "v02_readout_facts"
FACTS_SCHEMA = "seqref-v02rd-facts/1"

STEP0, FINAL_STEP = 0, 3258
ENDPOINTS = (STEP0, FINAL_STEP)
N_HOLDOUT = 199
Z_DIAG_N = 128
FLOW_DIM = 13824                       # == ffr.FLOW_DIM_REAL (quoted)

# Locked reproduction tolerances (V02AMEND v1.0 SS17; accepted as
# registered; PM/C0 machinery validated exactly by V02H).
PSNR_REPRO_TOL_DB = 1e-3
NMSE_REPRO_TOL_REL = 1e-4

# Gate 9 anchors -- the EXACT frozen V02E quantities (v02_facts.json,
# schema seqref-v02-facts/1, key endpoint_measurements.holdout.
# per_checkpoint[]): arithmetic means of per-identity z=0 PSNR /
# NMSE_u over the 199 registered holdout identities (the registered
# midpoint-slice-per-volume selection, canonical manifest order).
Z0_ANCHORS = {
    STEP0: {"z0_psnr_mean": 30.83547441705949,
            "z0_nmse_u_mean": 0.99615782406743},
    FINAL_STEP: {"z0_psnr_mean": 30.784850113978706,
                 "z0_nmse_u_mean": 1.0101168333589574},
}

# Gate 11 anchors -- the registered D3-subset PM reproduction values
# (v02_facts.json secondary_monitoring.d3_monitor.pm; reproduced
# exactly by V02H on 2026-09-10).
PM_STEP0_PSNR = 30.794137818416605
PM_STEP0_NMSE = 1.012275318293053
PM_FINAL_PSNR = 30.786611478861552
PM_FINAL_NMSE = 1.0141107234837616

# Registered identity pins.
PM_BANK_SHA256 = ("f8a6a34b6b474286359e4c99341de256b62bc0dff1e6bc8cf"
                  "52e72f0461dc8d6")
PM_BANK_MANIFEST_SHA256 = ("c4d2c0ec7c908f7c20a4f287eeffbdf966924a51"
                           "18b6ba734b0a329942ab7e5f")
D3_SUBSET_MANIFEST_SHA256 = ("66647202aead2edbbbea338194f464a44dc77b63"
                             "e65824a56ff2998f40879bcd")
HOLDOUT_MANIFEST_SHA256 = ("ce27550cdcdfa627372a53d51df6c7a97e4c6cff"
                           "0f5a80622e0fa110b0eea165")
V02H_EVIDENCE_SHA256 = ("c9adf1978a120b2ef4f2553fbf29adb48b43227f71e80191"
                        "eb344527f1dad3d9")
V02H_EVIDENCE_SCHEMA = "seqref-v02h-selftest-facts/1"

ROUTE_SUPPORT = "READOUT_LIMITATION_SUPPORTED"
ROUTE_SUPPORT_DETAIL = "POSTERIOR_READOUT_SUPPORT"
ROUTE_NO_SUPPORT = "TESTED_READOUTS_DO_NOT_RECOVER_RECONSTRUCTION_SUPPORT"


class V02GError(Exception):
    """Single typed error identity: (error_code, reason)."""

    def __init__(self, error_code: str, reason: str):
        super().__init__(f"{error_code}: {reason}")
        self.error_code = error_code
        self.reason = reason


def _fail(error_code: str, reason: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, error_code, reason)
    raise V02GError(error_code, reason)


def _env() -> SimpleNamespace:
    """Total lazy import of every environment-bound dependency (the
    registered scientific host), mirroring v02e._env() exactly -- the
    canonical seams, NOT the V02H selftest harness. The R0/R1 readout
    path is decode-oriented (tg._decode_z, d3 bank decode, estimators
    image formation); no Fourier array seam is needed."""
    try:
        from seqref_mri.tdiag import _bootstrap  # noqa: F401 --
        # registered path bootstrap (same contract as SEQREF-V02E v0.2):
        # legacy top-level names resolve to the SAME module objects.
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


N_D3_SUBSET = 32


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_v02h_evidence(path: str) -> dict:
    """The R2-removal rationale is a HARD provenance pin (v1.0 SS6/SS18):
    the V02H authoritative artifact must exist, tie to the registered
    sha256, carry the registered schema, and record OUTCOME_A_R2_NO_OP
    at the final decision endpoint. V02H is an evidence dependency
    only -- no code is imported from the selftest harness. The
    returned record is semantic-scope safe: it carries NO filesystem
    path (the path lives in the non-semantic run block)."""
    p = Path(path)
    if not p.exists():
        _fail("V02H_EVIDENCE_MISSING",
              f"the V02H Outcome-A evidence artifact {path} does not "
              f"exist; the R2-removal rationale cannot be verified")
    sha = _file_sha256(p)
    if sha != V02H_EVIDENCE_SHA256:
        _fail("V02H_EVIDENCE_MISMATCH",
              f"V02H evidence artifact sha256 {sha[:12]}... != the "
              f"registered pin {V02H_EVIDENCE_SHA256[:12]}...")
    rec = json.loads(p.read_text())
    _check_v02h_content(rec)
    return {"sha256": sha,
            "schema": rec["schema"],
            "outcome": rec["outcome"]["label"],
            "role": "the R2-removal rationale (V02AMEND v1.0 SS6); "
                    "evidence/provenance dependency only"}


def _check_v02h_content(rec: dict) -> None:
    """Content contract of the V02H evidence artifact (separated from
    the sha pin so the assembly fixtures can exercise it directly)."""
    if rec.get("schema") != V02H_EVIDENCE_SCHEMA:
        _fail("V02H_EVIDENCE_SCHEMA_MISMATCH",
              f"V02H evidence schema {rec.get('schema')!r} != "
              f"{V02H_EVIDENCE_SCHEMA!r}")
    label = str(rec.get("outcome", {}).get("label"))
    if label != "OUTCOME_A_R2_NO_OP":
        _fail("V02H_EVIDENCE_OUTCOME_MISMATCH",
              f"V02H outcome {label!r} != 'OUTCOME_A_R2_NO_OP'; the "
              f"locked R0-vs-R1 scope rests on the Outcome-A removal "
              f"mandate and cannot be verified otherwise")


def _verify_holdout(holdman: dict) -> None:
    """Exactly the 199 frozen identities in canonical manifest order;
    the registered content SHA ties (full value pinned from
    v02_facts.json holdout_manifest.manifest_sha256)."""
    if holdman.get("manifest_sha256") != HOLDOUT_MANIFEST_SHA256:
        _fail("HOLDOUT_MANIFEST_DRIFT",
              f"holdout manifest_sha256 "
              f"{str(holdman.get('manifest_sha256'))[:12]}... != the "
              f"registered pin {HOLDOUT_MANIFEST_SHA256[:12]}...")
    entries = holdman.get("entries", [])
    if len(entries) != N_HOLDOUT \
            or int(holdman.get("n_volumes", -1)) != N_HOLDOUT:
        _fail("HOLDOUT_POPULATION_MISMATCH",
              f"holdout manifest declares "
              f"n_volumes={holdman.get('n_volumes')} / "
              f"{len(entries)} entries; locked {N_HOLDOUT}")


def _verify_d3_subset(d3man: dict, holdman: dict) -> None:
    """Re-derive the frozen D3 draw and re-bind it to the verified
    holdout manifest (mirrored from SEQREF-V02E v0.2 run(); draw drift
    is an ERROR, never a redraw), plus the registered manifest-SHA
    cross-check against the v02_facts pin."""
    if int(d3man.get("n", -1)) != N_D3_SUBSET \
            or len(d3man.get("entries", [])) != N_D3_SUBSET:
        _fail("MANIFEST_CONTENT_MISMATCH",
              f"D3 monitor manifest declares n={d3man.get('n')} / "
              f"{len(d3man.get('entries', []))} entries; locked "
              f"{N_D3_SUBSET}")
    positions = [int(p) for p in v02m.d3_monitor_positions()]
    drawn = [int(e["holdout_position"]) for e in d3man["entries"]]
    if drawn != positions or [int(e["draw_rank"])
                              for e in d3man["entries"]] != list(
                                  range(N_D3_SUBSET)):
        _fail("D3_DRAW_MISMATCH",
              "the D3 monitor manifest does not reproduce the frozen "
              "PCG64(2).choice(199, 32, replace=False) draw in order")
    for e in d3man["entries"]:
        h = holdman["entries"][int(e["holdout_position"])]
        if e["file"] != h["file"] \
                or int(e["slice_index"]) != int(h["slice_index"]):
            _fail("D3_SUBSET_NOT_IN_HOLDOUT",
                  f"D3 entry {e['file']!r}/{e['slice_index']} is not "
                  f"the holdout entry at position "
                  f"{e['holdout_position']}")
    if d3man.get("manifest_sha256") != D3_SUBSET_MANIFEST_SHA256:
        _fail("D3_SUBSET_MANIFEST_DRIFT",
              f"D3 monitor manifest_sha256 "
              f"{str(d3man.get('manifest_sha256'))[:12]}... != the "
              f"registered v02_facts pin "
              f"{D3_SUBSET_MANIFEST_SHA256[:12]}...")


def _verify_pm_bank(bank_rec: dict) -> dict:
    """The PM bank must regenerate to the registered SHAs (the
    PCG64(0) Z_DIAG bank, reused verbatim -- no new stream, no redraw,
    no adaptive count)."""
    if bank_rec["bank_sha256"] != PM_BANK_SHA256 \
            or bank_rec["manifest_sha256"] != PM_BANK_MANIFEST_SHA256:
        _fail("PM_BANK_MISMATCH",
              f"regenerated Z_DIAG bank sha "
              f"{bank_rec['bank_sha256'][:12]}.../manifest "
              f"{bank_rec['manifest_sha256'][:12]}... != the "
              f"registered constants {PM_BANK_SHA256[:12]}.../"
              f"{PM_BANK_MANIFEST_SHA256[:12]}...")
    if bank_rec["bank"].shape != (Z_DIAG_N, FLOW_DIM):
        _fail("PM_BANK_LAYOUT_UNEXPECTED",
              f"Z_DIAG bank shape {bank_rec['bank'].shape} != "
              f"({Z_DIAG_N}, {FLOW_DIM})")
    return {"rule": "np.random.Generator(np.random.PCG64(0))."
                    "standard_normal(size=(128, 13824), dtype=float64)"
                    ".astype(float32) -- the registered TDIAG bank, "
                    "reused verbatim",
            "bank_sha256": bank_rec["bank_sha256"],
            "manifest_sha256": bank_rec["manifest_sha256"]}


def _verify_pairing(states: list, holdman: dict) -> None:
    """The measured identity sequence must equal the manifest order
    EXACTLY: R0 and R1 arrays are filled in this one order, so pairing
    is by construction; this check proves the construction."""
    if len(states) != N_HOLDOUT:
        _fail("STATE_COUNT_MISMATCH",
              f"collected {len(states)} holdout states, manifest "
              f"declares {N_HOLDOUT}")
    for j, (st, e) in enumerate(zip(states, holdman["entries"])):
        if st["identity"] != {"file": e["file"],
                              "slice_index": int(e["slice_index"])}:
            _fail("PAIRING_ORDER_MISMATCH",
                  f"position {j}: measured identity "
                  f"{st['identity']!r} != manifest entry "
                  f"{e['file']!r}/{e['slice_index']}; R0/R1 pairing "
                  f"order is not the manifest order")


# ---------------------------------------------------------------------------
# Measurement paths (canonical seams only).
# ---------------------------------------------------------------------------

def _r0_measure(env, model, st: dict, where: str) -> dict:
    """R0 -- z=0: v02e._z0_metrics verbatim (the production decode/
    metric engine). Returns per-volume PSNR/NMSE_u plus measured
    decode telemetry (telemetry is outside the semantic scope)."""
    t0 = time.perf_counter()
    psnr, nmse = v02e._z0_metrics(env, model, st, where)
    dt = time.perf_counter() - t0
    if not (math.isfinite(psnr) and math.isfinite(nmse)):
        _fail("NON_FINITE_METRIC",
              f"{where}: R0 produced a non-finite metric "
              f"(psnr={psnr!r}, nmse_u={nmse!r})")
    return {"psnr": float(psnr), "nmse_u": float(nmse),
            "decode_seconds": dt}


def _r1_measure(env, model, st: dict, bank: np.ndarray,
                where: str) -> dict:
    """R1 -- posterior mean: the registered u-space averaging rule
    (d3._pm_metrics semantics, composed from the same canonical
    blocks): ONE batched bank decode (d3._decode_bank_conditioned,
    C0 own/own inputs) -> complex mean of the physical free vectors
    BEFORE image formation -> ONE image formation
    (estimators.image_from_u) -> production PSNR/NMSE_u. The 128
    sample norms are summarized as diagnostic-only/non-semantic
    telemetry (never a gate, never in the semantic scope)."""
    torch = env.torch
    try:
        t0 = time.perf_counter()
        with torch.no_grad():
            decodes = env.d3._decode_bank_conditioned(
                model, st, bank, st["cond"], st["mask"])
        dt = time.perf_counter() - t0
    except env.StageError as exc:
        _fail("DECODE_FAILURE", f"{where}: {exc.error_code}: {exc.reason}")
    if decodes.shape[0] != Z_DIAG_N:
        _fail("D3_BANK_LAYOUT_UNEXPECTED",
              f"{where}: the posterior mean expects {Z_DIAG_N} shared "
              f"decodes, got {decodes.shape[0]}")
    u_mean = decodes.mean(axis=0)
    try:
        mag = env.estimators.image_from_u(st, u_mean).abs()
        psnr = env.tg._psnr(mag, st["x_true_mag"])
        nmse = env.tg._nmse(u_mean, st["u_true"])
    except env.StageError as exc:
        _fail("EVAL_PM_FAILURE",
              f"{where}: {exc.error_code}: {exc.reason}")
    if not (math.isfinite(psnr) and math.isfinite(nmse)):
        _fail("NON_FINITE_METRIC",
              f"{where}: R1 produced a non-finite metric "
              f"(psnr={psnr!r}, nmse_u={nmse!r})")
    norms = np.linalg.norm(decodes, axis=1)
    return {"psnr": float(psnr), "nmse_u": float(nmse),
            "bank_decode_seconds": dt,
            "samples_decoded": int(decodes.shape[0]),
            "sample_norm_diagnostic_only": {
                "mean": float(norms.mean()), "std": float(norms.std()),
                "min": float(norms.min()), "max": float(norms.max())}}


def _measure_endpoint(env, model, states: list, bank: np.ndarray,
                      endpoint_label: str) -> dict:
    """For every one of the 199 states, in manifest order: R0 then R1
    (paired by construction). Per-volume records carry metrics plus
    measured telemetry."""
    per_volume = []
    for j, st in enumerate(states):
        where = f"{endpoint_label} volume {j} {st['identity']!r}"
        r0 = _r0_measure(env, model, st, where + " R0")
        r1 = _r1_measure(env, model, st, bank, where + " R1")
        per_volume.append({"identity": st["identity"],
                           "r0": r0, "r1": r1})
        if (j + 1) % 10 == 0 or j + 1 == len(states):
            logger.info("[%s] %s: %d/%d volumes measured", __abbr__,
                        endpoint_label, j + 1, len(states))
    return {"endpoint": endpoint_label, "per_volume": per_volume}


# ---------------------------------------------------------------------------
# Statistics assembly. The PRIMARY comparison is final R1 vs final R0
# ONLY; secondary comparisons are descriptive and can NEVER alter
# routing.
# ---------------------------------------------------------------------------

def _arr(meas: dict, readout: str, key: str) -> np.ndarray:
    return np.array([v[readout][key] for v in meas["per_volume"]],
                    dtype=np.float64)


def _aggregates(x: np.ndarray) -> dict:
    q = np.percentile(x, [25.0, 75.0])
    return {"n": int(x.size), "mean": float(x.mean()),
            "median": float(np.median(x)), "sd": float(x.std()),
            "min": float(x.min()), "max": float(x.max()),
            "q1": float(q[0]), "q3": float(q[1])}


def _condition_aggregates(meas: dict) -> dict:
    out = {}
    for readout in ("r0", "r1"):
        out[readout] = {
            "psnr": _aggregates(_arr(meas, readout, "psnr")),
            "nmse_u": _aggregates(_arr(meas, readout, "nmse_u"))}
    return out


def _primary_comparison(step0: dict, final: dict,
                        boot_idx: np.ndarray) -> dict:
    """final R1 vs final R0 ONLY. OBSERVED statistics feed the
    practical-effect conjuncts; the bootstrap supplies CI95 bounds
    ONLY (the reviewer-registered separation)."""
    r0p, r1p = _arr(final, "r0", "psnr"), _arr(final, "r1", "psnr")
    r0n, r1n = _arr(final, "r0", "nmse_u"), _arr(final, "r1", "nmse_u")
    obs_psnr = v02c.observed_d1d2(r0p, r1p)
    obs_nmse = v02c.observed_d1d2(r0n, r1n)
    ci_psnr = v02c.paired_bootstrap_ci(r0p, r1p, boot_idx)
    ci_nmse = v02c.paired_bootstrap_ci(r0n, r1n, boot_idx)
    d_psnr = r1p - r0p
    acceptance = v02c.d1d2_acceptance(
        {"mean": obs_psnr["d1_observed_mean_paired_difference"],
         "ci95": ci_psnr["d1_mean_paired_difference"]["ci95"]},
        {"mean": obs_nmse["d2_observed_ratio_of_means"],
         "ci95": ci_nmse["d2_ratio_of_resampled_means"]["ci95"]})
    return {
        "comparison": "final R1 vs final R0 (the ONLY routing input)",
        "paired_psnr": {
            "observed_mean_difference_db":
                obs_psnr["d1_observed_mean_paired_difference"],
            "median_difference_db": float(np.median(d_psnr)),
            "fraction_delta_gt_0": float(np.mean(d_psnr > 0.0)),
            "fraction_delta_ge_0p5_db":
                float(np.mean(d_psnr >= v02c.D1_PAIRED_PSNR_MIN_DB)),
            "bootstrap": ci_psnr},
        "nmse_ratio": {
            "observed_ratio_of_means":
                obs_nmse["d2_observed_ratio_of_means"],
            "bootstrap": ci_nmse},
        "d1": acceptance["d1"],
        "d2": acceptance["d2"],
        "rule": "D1: OBSERVED mean paired dPSNR >= +0.50 dB AND CI95 "
                "lower > 0; D2: OBSERVED mean NMSE_u ratio <= 0.95 "
                "AND CI95 upper < 1.0 (mean thresholds inclusive; CI "
                "null exclusions strict; bootstrap intervals ONLY, "
                "replicate means diagnostic-only)"}


def _secondary_comparisons(step0: dict, final: dict) -> dict:
    """Descriptive ONLY -- never classification inputs (v1.0 SS8)."""
    r0p0, r1p0 = _arr(step0, "r0", "psnr"), _arr(step0, "r1", "psnr")
    r0pf, r1pf = _arr(final, "r0", "psnr"), _arr(final, "r1", "psnr")
    r0n0, r1n0 = _arr(step0, "r0", "nmse_u"), _arr(step0, "r1", "nmse_u")
    r0nf, r1nf = _arr(final, "r0", "nmse_u"), _arr(final, "r1", "nmse_u")
    return {
        "descriptive_only": True,
        "routing_effect": "NONE -- secondary statistics can never "
                          "alter the classification",
        "absolute_recovery_vs_initialisation": {
            "definition": "R1 final - R0 step0",
            "mean_psnr_difference_db": float((r1pf - r0p0).mean()),
            "nmse_ratio_of_means": float(r1nf.mean() / r0n0.mean())},
        "same_readout_training_effect": {
            "r1_final_minus_step0": {
                "mean_psnr_difference_db": float((r1pf - r1p0).mean()),
                "nmse_ratio_of_means":
                    float(r1nf.mean() / r1n0.mean())},
            "r0_final_minus_step0_context": {
                "mean_psnr_difference_db": float((r0pf - r0p0).mean()),
                "nmse_ratio_of_means":
                    float(r0nf.mean() / r0n0.mean())}}}


def _routing(primary: dict) -> dict:
    """Case A: D1 AND D2 pass -> READOUT_LIMITATION_SUPPORTED (detail
    POSTERIOR_READOUT_SUPPORT). Case B: otherwise ->
    TESTED_READOUTS_DO_NOT_RECOVER_RECONSTRUCTION_SUPPORT. Case C
    (validity failure) never reaches here: it is an ERROR raised
    before classification."""
    d1_pass = bool(primary["d1"]["pass"])
    d2_pass = bool(primary["d2"]["pass"])
    if d1_pass and d2_pass:
        return {"label": ROUTE_SUPPORT, "detail": ROUTE_SUPPORT_DETAIL,
                "interpretation": "a posterior estimator extracts "
                                  "materially better reconstruction "
                                  "from the same frozen learned model "
                                  "than z=0; supports investigation "
                                  "of posterior/inference redesign"}
    return {"label": ROUTE_NO_SUPPORT,
            "interpretation": "the preregistered alternative readout "
                              "fails to recover a material "
                              "reconstruction improvement from the "
                              "frozen final model; strengthens the "
                              "rationale for testing a "
                              "reconstruction-aware or hybrid "
                              "objective/model redesign without "
                              "establishing unique causation"}


# ---------------------------------------------------------------------------
# Hard pre-classification reproduction gates (v1.0 SS14 gates 9/11).
# ---------------------------------------------------------------------------

def _check_r0_reproduction(meas: dict) -> dict:
    """Gate 9: recomputed z=0 arithmetic means over the 199 registered
    identities must reproduce the EXACT frozen V02E quantities
    (endpoint_measurements.holdout.per_checkpoint[] z0_psnr_mean /
    z0_nmse_u_mean) within 1e-3 dB PSNR / 1e-4 relative NMSE_u --
    (measured-anchor)/anchor. Any breach is ERROR, never a re-pin."""
    step = STEP0 if meas["endpoint"] == "step0" else FINAL_STEP
    anchor = Z0_ANCHORS[step]
    out = {}
    mp = float(_arr(meas, "r0", "psnr").mean())
    dp = mp - anchor["z0_psnr_mean"]
    out["psnr"] = {"measured": mp, "anchor": anchor["z0_psnr_mean"],
                   "delta_db": dp, "tol_db": PSNR_REPRO_TOL_DB,
                   "aggregation": "arithmetic mean of per-identity "
                                  "z=0 PSNR over the 199 registered "
                                  "holdout identities (v02_facts "
                                  "endpoint_measurements.holdout."
                                  "per_checkpoint convention)"}
    if not math.isfinite(dp) or abs(dp) > PSNR_REPRO_TOL_DB:
        _fail("REPRO_R0_PSNR_MISMATCH",
              f"{meas['endpoint']}: recomputed z=0 PSNR mean {mp!r} "
              f"deviates from the frozen V02E anchor "
              f"{anchor['z0_psnr_mean']!r} by {dp!r} dB "
              f"(tol {PSNR_REPRO_TOL_DB!r})")
    mn = float(_arr(meas, "r0", "nmse_u").mean())
    rn = (mn - anchor["z0_nmse_u_mean"]) / anchor["z0_nmse_u_mean"]
    out["nmse_u"] = {"measured": mn, "anchor": anchor["z0_nmse_u_mean"],
                     "delta_rel": rn, "tol_rel": NMSE_REPRO_TOL_REL,
                     "aggregation": "arithmetic mean of per-identity "
                                    "z=0 NMSE_u over the 199 "
                                    "registered holdout identities; "
                                    "relative error "
                                    "(measured-anchor)/anchor"}
    if not math.isfinite(rn) or abs(rn) > NMSE_REPRO_TOL_REL:
        _fail("REPRO_R0_NMSE_MISMATCH",
              f"{meas['endpoint']}: recomputed z=0 NMSE_u mean {mn!r} "
              f"deviates from the frozen V02E anchor "
              f"{anchor['z0_nmse_u_mean']!r} by {rn!r} relative "
              f"(tol {NMSE_REPRO_TOL_REL!r})")
    return out


def _check_pm_reproduction(env, models: dict, d3_states: list,
                           bank: np.ndarray) -> dict:
    """Gate 11: the D3-subset PM assembly must reproduce the
    registered PM anchors within the same locked tolerances. This
    proves the V02G R1 assembly wires the canonical blocks correctly
    before any scientific routing is computed."""
    out = {}
    for label, model, a_psnr, a_nmse in (
            ("step0", models[STEP0], PM_STEP0_PSNR, PM_STEP0_NMSE),
            ("final", models[FINAL_STEP], PM_FINAL_PSNR, PM_FINAL_NMSE)):
        psnrs, nmses = [], []
        for j, st in enumerate(d3_states):
            r1 = _r1_measure(env, model, st, bank,
                             f"gate11 {label} d3 slice {j}")
            psnrs.append(r1["psnr"])
            nmses.append(r1["nmse_u"])
        mp, mn = float(np.mean(psnrs)), float(np.mean(nmses))
        dp = mp - a_psnr
        rn = (mn - a_nmse) / a_nmse
        out[label] = {"psnr": {"measured": mp, "anchor": a_psnr,
                               "delta_db": dp, "tol_db": PSNR_REPRO_TOL_DB},
                      "nmse_u": {"measured": mn, "anchor": a_nmse,
                                 "delta_rel": rn,
                                 "tol_rel": NMSE_REPRO_TOL_REL}}
        if not math.isfinite(dp) or abs(dp) > PSNR_REPRO_TOL_DB:
            _fail("REPRO_PM_PSNR_MISMATCH",
                  f"gate 11 {label}: PM psnr {mp!r} deviates from the "
                  f"registered anchor {a_psnr!r} by {dp!r} dB")
        if not math.isfinite(rn) or abs(rn) > NMSE_REPRO_TOL_REL:
            _fail("REPRO_PM_NMSE_MISMATCH",
                  f"gate 11 {label}: PM nmse_u {mn!r} deviates from "
                  f"the registered anchor {a_nmse!r} by {rn!r} "
                  f"relative")
    return out


# ---------------------------------------------------------------------------
# Facts assembly: scientific content is separated from telemetry so the
# semantic hash covers ONLY reproducible scientific content (the
# sibling-rerun contract, v1.0 SS14 gate 18). Telemetry -- timestamps,
# wall-clock, decode timings, memory, host strings, paths, and the R1
# sample-norm diagnostics -- is recorded in facts but EXCLUDED from
# the semantic scope.
# ---------------------------------------------------------------------------

SEMANTIC_EXCLUDED = ("run", "timing", "r1_sampling_diagnostics",
                     "semantic_sha256", "artifact_sha256")


def _split_measurement(meas: dict) -> tuple:
    """Split a measured endpoint into (scientific per-volume records,
    decode-timing telemetry, R1 sample-norm diagnostics)."""
    metrics, timing, diag = [], [], []
    for v in meas["per_volume"]:
        metrics.append({"identity": v["identity"],
                        "r0": {"psnr": v["r0"]["psnr"],
                               "nmse_u": v["r0"]["nmse_u"]},
                        "r1": {"psnr": v["r1"]["psnr"],
                               "nmse_u": v["r1"]["nmse_u"]}})
        timing.append({"identity": v["identity"],
                       "r0_decode_seconds": v["r0"]["decode_seconds"],
                       "r1_bank_decode_seconds":
                           v["r1"]["bank_decode_seconds"]})
        diag.append({"identity": v["identity"],
                     "samples_decoded": v["r1"]["samples_decoded"],
                     "sample_norm_diagnostic_only":
                         v["r1"]["sample_norm_diagnostic_only"]})
    return metrics, timing, diag


def _decode_accounting() -> dict:
    return {"r0_decodes": N_HOLDOUT * len(ENDPOINTS),
            "r1_latent_decodes": N_HOLDOUT * Z_DIAG_N * len(ENDPOINTS),
            "total_reconstruction_decodes":
                N_HOLDOUT * (1 + Z_DIAG_N) * len(ENDPOINTS),
            "note": "one R0 decode and 128 R1 latent decodes for each "
                    "of the 199 identities at each of the 2 endpoints: "
                    "398 R0 + 50,944 R1 = 51,342 total"}


def _guard_rerun(facts: dict, out_dir: str) -> None:
    auth = Path(out_dir) / f"{FACTS_PREFIX}.json"
    if not auth.exists():
        return
    sidecar = Path(str(auth) + ".sha256")
    if not sidecar.exists() or _file_sha256(auth) != \
            sidecar.read_text().split()[0].strip():
        _fail("RERUN_PRIOR_ARTEFACT_MISMATCH",
              "an authoritative v02_readout_facts.json exists but "
              "fails its own sidecar verification; no sibling is "
              "published next to an unverifiable artefact")
    prior = json.loads(auth.read_text())
    if prior.get("semantic_sha256") != facts["semantic_sha256"]:
        _fail("SEMANTIC_RERUN_MISMATCH",
              f"the existing authoritative {FACTS_PREFIX}.json has "
              f"semantic {str(prior.get('semantic_sha256'))[:12]}..."
              f" != this run's "
              f"{facts['semantic_sha256'][:12]}...; a scientifically "
              f"different rerun may NOT publish silently alongside")


def _publish_facts(env, facts: dict, out_dir: str) -> tuple:
    semantic = {k: v for k, v in facts.items()
                if k not in SEMANTIC_EXCLUDED}
    env.attach_semantic_hash(facts, semantic)
    _guard_rerun(facts, out_dir)
    try:
        return env.publish_stage(facts, out_dir, FACTS_PREFIX, STAGE)
    except env.StageError as exc:
        _fail("PUBLICATION_FAILURE", f"{exc.error_code}: {exc.reason}")


def _code_provenance() -> dict:
    """The executing repository commit/branch (v1.0 SS18 'code
    commit'). Resolved from git at run time; unavailable git metadata
    in an authoritative run is an ERROR -- no fabricated provenance."""
    import subprocess
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root,
            capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        _fail("CODE_PROVENANCE_UNAVAILABLE",
              f"git commit/branch could not be resolved at {root}: "
              f"{exc}; an authoritative run never fabricates "
              f"provenance")
    if len(commit) != 40 or not branch:
        _fail("CODE_PROVENANCE_UNAVAILABLE",
              f"unexpected git metadata commit={commit!r} "
              f"branch={branch!r}")
    return {"commit": commit, "branch": branch,
            "module": f"{__abbr__} v{__version__}",
            "file": "seqref_mri/scripts/v02_readout_diag.py",
            "common_helpers": f"SEQREF-V02C v{v02c.__version__} "
                              "(seqref_mri/scripts/v02_readout_common.py)"}


# ---------------------------------------------------------------------------
# The authoritative run (v1.0 SS14 validity gates enforced BEFORE
# classification; reproduction gates BEFORE D1/D2).
# ---------------------------------------------------------------------------

def run(cfg: dict) -> dict:
    """cfg keys: data_root, manifest_dir, train_record, run_root,
    p0s_facts, p4_stats2, implb_facts, v02h_evidence, out_dir. Every
    scientific constant is locked in this module and its frozen
    inputs, never in cfg."""
    env = _env()
    det_state = v02c.pin_deterministic_backend(env.torch)
    t0 = time.time()
    man_dir = Path(cfg["manifest_dir"])

    # --- provenance / frozen inputs (refuse-on-mismatch) ------------
    v02h_rec = _verify_v02h_evidence(cfg["v02h_evidence"])
    manifests = [v02e._load_manifest(man_dir, f"v02_epoch{ep}_manifest.json",
                                     "train_epoch")
                 for ep in range(v02m.N_EPOCHS)]
    holdman = v02e._load_manifest(man_dir, "v02_holdout_manifest.json",
                                  "holdout")
    d3man = v02e._load_manifest(man_dir, "v02_d3_monitor_manifest.json",
                                "d3_monitor")
    _verify_holdout(holdman)
    _verify_d3_subset(d3man, holdman)
    record = v02e._load_train_record(cfg["train_record"], manifests)
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
        ds_val = env.FastMRISliceDataset(cfg["data_root"], split="val",
                                         mode="eval")
    except Exception as exc:  # the dataset ctor is a typed, logging
        _fail("DATA_PREMISE_FAILURE", f"{type(exc).__name__}: {exc}")
    ckpt_recs = {int(c["step"]): c for c in record["checkpoints"]}
    models = {step: v02e._load_checkpoint(env, Path(cfg["run_root"]),
                                          ckpt_recs[step],
                                          implb["spline_b"])
              for step in ENDPOINTS}
    bank_rec = env.estimators.z_diag_bank()
    bank_block = _verify_pm_bank(bank_rec)
    provenance = _code_provenance()

    # --- population --------------------------------------------------
    t_states = time.perf_counter()
    states = v02e._collect_states(env, ds_val, holdman["entries"], p4,
                                  s_ref, N_HOLDOUT)
    states_seconds = time.perf_counter() - t_states
    _verify_pairing(states, holdman)
    d3_states = v02e._collect_states(env, ds_val, d3man["entries"], p4,
                                     s_ref, N_D3_SUBSET)

    # --- reserved bootstrap stream: ONE consumption, pin-verified ----
    boot_idx = v02c.boot_index_matrix()
    v02c.verify_boot_index_pin(boot_idx)

    # --- measurement -------------------------------------------------
    meas = {"step0": _measure_endpoint(env, models[STEP0], states,
                                       bank_rec["bank"], "step0"),
            "final": _measure_endpoint(env, models[FINAL_STEP], states,
                                       bank_rec["bank"], "final")}

    # --- hard pre-classification reproduction gates ------------------
    repro_r0 = {ep: _check_r0_reproduction(meas[ep]) for ep in meas}
    repro_pm = _check_pm_reproduction(env, models, d3_states,
                                      bank_rec["bank"])

    # --- statistics + routing ----------------------------------------
    boot_block = {"generator": "PCG64", "seed": v02c.BOOTSTRAP_SEED,
                  "B": v02c.BOOTSTRAP_B, "unit": "volume",
                  "paired": True,
                  "index_matrix_sha256": v02c.GOLDEN_BOOT_IDX_SHA256,
                  "consumption": "PCG64(4).integers(0, 199, "
                                 "size=(10000, 199)) -- the single and "
                                 "only consumption of the reserved "
                                 "stream; identical indices applied to "
                                 "R0 and R1",
                  "estimators": "D1: per-resample mean of the paired "
                                "difference; D2: per-resample ratio of "
                                "resampled arithmetic means; CI95 "
                                "percentile [2.5, 97.5]"}
    primary = _primary_comparison(meas["step0"], meas["final"], boot_idx)
    secondary = _secondary_comparisons(meas["step0"], meas["final"])
    routing = _routing(primary)

    metrics_block, timing_block, diag_block = {}, {"per_volume": {}}, {}
    for ep in meas:
        m, t, d = _split_measurement(meas[ep])
        metrics_block[ep] = {"aggregates": _condition_aggregates(meas[ep]),
                             "per_volume": m}
        timing_block["per_volume"][ep] = t
        diag_block[ep] = d
    timing_block.update({
        "telemetry": "measured THIS authoritative run; recorded in "
                     "facts but EXCLUDED from the semantic scope",
        "state_construction_seconds": states_seconds,
        "decode_accounting": _decode_accounting(),
        "timing_probe": {
            "provenance": "prerun feasibility evidence (EPHEMERAL "
                          "probe artefact), NOT measured in this run",
            "reference": "results/_diag/v02_readout_probe/"}})

    facts = {
        "schema": FACTS_SCHEMA,
        "stage": STAGE,
        "stage_version": f"{__abbr__} v{__version__}",
        "amendment": "SEQREF-MRI-V02AMEND v1.0 LOCKED (2026-09-11, "
                     "LudWiki id 147)",
        "code": provenance,
        "frozen_inputs": {
            "checkpoint_state_sha256_verified": {
                str(step): ckpt_recs[step]["state_sha256"]
                for step in ENDPOINTS},
            "train_record_sha256_verified": True,
            "holdout_manifest_sha256": holdman["manifest_sha256"],
            "d3_subset_manifest_sha256": d3man["manifest_sha256"],
            "pm_bank": bank_block,
            "v02h_evidence": v02h_rec},
        "population": {
            "n_volumes": N_HOLDOUT,
            "identities": [st["identity"] for st in states],
            "order": "canonical v02_holdout_manifest.json entry order "
                     "(the registered midpoint-slice-per-volume "
                     "selection); no redraw, no substitution, no "
                     "post-hoc exclusion",
            "pairing": "R0 and R1 measured in the same single "
                       "manifest order; pairing verified position by "
                       "position"},
        "endpoints": [STEP0, FINAL_STEP],
        "readouts": {
            "r0": "z=0 -- the existing V02 reconstruction rule "
                  "(reference); v02e._z0_metrics production path",
            "r1": "posterior mean -- the registered 128-sample "
                  "PCG64(0) bank; complex mean of the physical free "
                  "vectors BEFORE image formation; one image "
                  "formation; production metric paths"},
        "bootstrap": boot_block,
        "reproduction": {
            "r0_z0_vs_v02e": repro_r0,
            "d3_pm": repro_pm,
            "tolerances": {"psnr_abs_db": PSNR_REPRO_TOL_DB,
                           "nmse_u_rel": NMSE_REPRO_TOL_REL},
            "scope": "gate 9 anchors are the EXACT frozen V02E "
                     "quantities (v02_facts.json endpoint_measurements."
                     "holdout.per_checkpoint[] z0_psnr_mean / "
                     "z0_nmse_u_mean at steps 0 and 3258); the "
                     "aggregation is the arithmetic mean over the 199 "
                     "registered identities; relative NMSE error is "
                     "(measured-anchor)/anchor"},
        "metrics": metrics_block,
        "primary_comparison": primary,
        "secondary_comparisons": secondary,
        "routing": routing,
        "validity": {
            "taxonomy": "exit 0 = evidence complete; exit 2 = ERROR; "
                        "no exit 1; a validity failure is ERROR, never "
                        "a scientific negative result; no partial "
                        "classification",
            "gates": {
                "1_checkpoint_file_sha_verified": True,
                "2_checkpoint_state_sha_verified": True,
                "3_train_record_identity_verified": True,
                "4_holdout_manifest_hashes_verified": True,
                "5_exactly_199_identities_measured":
                    len(states) == N_HOLDOUT,
                "6_no_identity_substitution_or_redraw": True,
                "7_z0_reproduction_within_tolerances": True,
                "8_pm_bank_identity_verified": True,
                "9_d3_pm_reproduction_within_tolerances": True,
                "10_production_seams_reused": True,
                "11_no_unregistered_exclusion": True,
                "12_no_non_finite_reconstruction_or_metric": True,
                "13_every_primary_per_volume_measurement_retained":
                    True,
                "14_v02_classification_unchanged": True,
                "15_facts_contain_no_verdict_field": True,
                "16_semantic_rerun_guard_armed": True},
            "manifest_sidecars_verified": [
                f"v02_epoch{ep}_manifest.json"
                for ep in range(v02m.N_EPOCHS)]
                + ["v02_holdout_manifest.json",
                   "v02_d3_monitor_manifest.json"],
            "non_finite_events": 0},
        "adaptive_holdout_reuse": {
            "declaration": "the 199-volume population is NOT a fresh "
                           "confirmatory holdout for this amendment: "
                           "it previously contributed to the V02 "
                           "scientific classification and is reused "
                           "for diagnostic routing; results are "
                           "diagnostic/routing evidence, never an "
                           "independent confirmatory replication; "
                           "thresholds and algorithms were frozen "
                           "before these measurements were observed",
            "registration": "V02AMEND v1.0 SS5 (id 147)"},
        "timing": timing_block,
        "r1_sampling_diagnostics": {
            "status": "diagnostic-only / NON-SEMANTIC -- excluded "
                      "from the semantic scope and never a gate",
            "per_volume": diag_block},
        "deterministic_execution": det_state,
        "run": {"utc": env.utc_stamp(), "argv": sys.argv,
                "v02h_evidence_path": str(Path(cfg["v02h_evidence"])),
                "wall_clock_s": time.time() - t0},
    }
    v02e._no_verdict_scan(facts, "facts")
    path, sha = _publish_facts(env, facts, str(Path(cfg["out_dir"])))
    logger.info("[%s] facts published %s sha256=%s; routing %s",
                __abbr__, path, sha[:12], routing["label"])
    return facts


# ---------------------------------------------------------------------------
# Non-scientific timing/feasibility probe (v1.0 SS15): EPHEMERAL prerun
# evidence with its own provenance. It NEVER writes to the facts
# location and makes no scientific claim.
# ---------------------------------------------------------------------------

def run_probe(cfg: dict) -> dict:
    """cfg keys: data_root, manifest_dir, train_record, run_root,
    p0s_facts, p4_stats2, implb_facts, probe_dir, probe_items. The
    probe exercises the exact retained production paths on the FIRST
    probe_items holdout identities and projects the full runtime."""
    env = _env()
    det_state = v02c.pin_deterministic_backend(env.torch)
    n_probe = int(cfg.get("probe_items", 4))
    if not 1 <= n_probe <= N_HOLDOUT:
        _fail("PROBE_CONFIG_INVALID",
              f"probe_items {n_probe} outside [1, {N_HOLDOUT}]")
    man_dir = Path(cfg["manifest_dir"])
    manifests = [v02e._load_manifest(man_dir, f"v02_epoch{ep}_manifest.json",
                                     "train_epoch")
                 for ep in range(v02m.N_EPOCHS)]
    holdman = v02e._load_manifest(man_dir, "v02_holdout_manifest.json",
                                  "holdout")
    _verify_holdout(holdman)
    record = v02e._load_train_record(cfg["train_record"], manifests)
    try:
        p4 = env.ffr.load_p4s2_parent(cfg["p4_stats2"])
        implb = env.ffr.load_implb_parent(cfg["implb_facts"])
        env.ffr.require_spline_b(implb["spline_b"])
        s_ref = env.tg._s_ref_from_p0s(cfg["p0s_facts"])
    except (env.ffr.FreeFlowError, env.StageError) as exc:
        _fail("PARENT_VERIFICATION_FAILED", f"{type(exc).__name__}: {exc}")
    try:
        ds_val = env.FastMRISliceDataset(cfg["data_root"], split="val",
                                         mode="eval")
    except Exception as exc:  # the dataset ctor is a typed, logging
        _fail("DATA_PREMISE_FAILURE", f"{type(exc).__name__}: {exc}")
    ckpt_recs = {int(c["step"]): c for c in record["checkpoints"]}
    models = {step: v02e._load_checkpoint(env, Path(cfg["run_root"]),
                                          ckpt_recs[step],
                                          implb["spline_b"])
              for step in ENDPOINTS}
    bank_rec = env.estimators.z_diag_bank()
    _verify_pm_bank(bank_rec)
    states = v02e._collect_states(env, ds_val,
                                  holdman["entries"][:n_probe], p4,
                                  s_ref, n_probe)

    r0_times, r1_times = [], []
    for step in ENDPOINTS:
        for j, st in enumerate(states):
            where = f"probe step {step} item {j}"
            r0 = _r0_measure(env, models[step], st, where + " R0")
            r1 = _r1_measure(env, models[step], st, bank_rec["bank"],
                             where + " R1")
            r0_times.append(r0["decode_seconds"])
            r1_times.append(r1["bank_decode_seconds"])
    r0_s = float(np.mean(r0_times))
    r1_s = float(np.mean(r1_times))
    per_endpoint = N_HOLDOUT * (r0_s + Z_DIAG_N * 0 + r1_s)
    projected = float(len(ENDPOINTS) * per_endpoint)
    probe = {
        "artefact": "V02G timing/feasibility probe -- NON-SCIENTIFIC "
                    "prerun feasibility evidence (EPHEMERAL)",
        "stage": STAGE,
        "stage_version": f"{__abbr__} v{__version__}",
        "utc": env.utc_stamp(),
        "probe_items_per_endpoint": n_probe,
        "identities": [st["identity"] for st in states],
        "device": "cpu (the inherited V02E readout path; checkpoints "
                  "map_location=cpu)",
        "deterministic_execution": det_state,
        "r0_seconds_per_item_mean": r0_s,
        "r1_seconds_per_item_mean": r1_s,
        "r1_seconds_per_item_includes":
            f"{Z_DIAG_N} latent decodes in ONE batched bank call",
        "projected_full_diagnostic_seconds": projected,
        "projected_full_diagnostic_hours": projected / 3600.0,
        "decode_accounting": _decode_accounting(),
        "note": "projection = 2 endpoints x 199 identities x "
                "(R0 + R1 per-item means); no runtime is inferred "
                "from V02E's z=0 rate alone; this probe has no "
                "scientific threshold and no effect on D1/D2"}
    out = Path(cfg["probe_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "v02_readout_probe.json"
    if path.exists():
        _fail("PROBE_ARTEFACT_EXISTS",
              f"{path} already exists; probe evidence is written once "
              f"per probe run, never silently overwritten")
    path.write_text(json.dumps(probe, indent=1) + "\n")
    logger.info("[%s] probe written %s; projected full diagnostic "
                "%.1f s (%.2f h)", __abbr__, path, projected,
                projected / 3600.0)
    return probe


# ---------------------------------------------------------------------------
# Assembly fixture suite (--mode fixtures). Synthetic inputs ONLY; the
# real holdout is never touched. V02H pre-validated the underlying
# concepts; these fixtures prove V02G's ASSEMBLY wires them correctly.
# ---------------------------------------------------------------------------

_FIXTURE_RESULTS = []


def _record(name: str, ok: bool, detail: str) -> None:
    _FIXTURE_RESULTS.append({"fixture": name, "ok": bool(ok),
                             "detail": detail})
    (logger.info if ok else logger.error)(
        "[%s] fixture %s: %s -- %s", __abbr__, name,
        "PASS" if ok else "FAIL", detail)


def _check(cond: bool, code: str, msg: str) -> None:
    if not cond:
        _fail(code, msg)


def _expect_error(fn, code_fragment: str, label: str) -> None:
    try:
        fn()
    except (V02GError, v02c.V02CError) as exc:
        _check(code_fragment in str(exc), "FIXTURE_ERROR_CODE_MISMATCH",
               f"{label}: expected error containing "
               f"{code_fragment!r}, got {exc}")
        return
    except Exception as exc:  # a typed seam error is acceptable ONLY
        # when it carries the expected code; anything else is a defect
        _check(code_fragment in str(exc), "FIXTURE_ERROR_CODE_MISMATCH",
               f"{label}: expected error containing "
               f"{code_fragment!r}, got {type(exc).__name__}: {exc}")
        return
    _fail("FIXTURE_EXPECTED_ERROR_MISSING",
          f"{label}: expected an error containing "
          f"{code_fragment!r}, none was raised")


def _synth_meas(psnr_r0, psnr_r1, nmse_r0, nmse_r1) -> dict:
    """A synthetic measured endpoint in the _measure_endpoint layout
    (telemetry fields included but irrelevant to statistics)."""
    per_volume = []
    for j in range(N_HOLDOUT):
        ident = {"file": f"synthetic/vol{j:03d}.h5", "slice_index": j}
        per_volume.append({
            "identity": ident,
            "r0": {"psnr": float(psnr_r0[j]), "nmse_u": float(nmse_r0[j]),
                   "decode_seconds": 0.0},
            "r1": {"psnr": float(psnr_r1[j]), "nmse_u": float(nmse_r1[j]),
                   "bank_decode_seconds": 0.0,
                   "samples_decoded": Z_DIAG_N,
                   "sample_norm_diagnostic_only":
                       {"mean": 1.0, "std": 0.0, "min": 1.0,
                        "max": 1.0}}})
    return {"endpoint": "synthetic", "per_volume": per_volume}


FIXTURE_SEED = 20260911


def _f01_observed_vs_ci_separation() -> None:
    """The acceptance rule consumes the OBSERVED statistic, never the
    bootstrap replicate mean. Decisive straddle (pinned construction,
    margins pre-verified against the golden index matrix): the
    observed D2 ratio is 0.950005... > 0.95 (practical FAIL) while the
    bootstrap replicate mean is 0.949996... <= 0.95 (would PASS if the
    wrong field were wired). D1 is arranged to pass, so the routing
    isolates the D2 practical conjunct on the observed value."""
    rng = np.random.Generator(np.random.PCG64(FIXTURE_SEED))
    r0n = np.concatenate([np.full(100, 1.0), np.full(99, 8.0)])
    r0n = r0n * (1.0 + 0.05 * rng.standard_normal(N_HOLDOUT)).clip(0.2)
    r1n = r0n * 0.950005 * (1 + 0.02 * rng.standard_normal(N_HOLDOUT))
    r1n *= (0.950005 * r0n.mean()) / r1n.mean()
    psnr_r0 = 30.0 + 0.5 * rng.standard_normal(N_HOLDOUT)
    psnr_r1 = psnr_r0 + 0.6 + 0.05 * rng.standard_normal(N_HOLDOUT)

    obs = v02c.observed_d1d2(r0n, r1n)
    idx = v02c.boot_index_matrix()
    v02c.verify_boot_index_pin(idx)
    ci = v02c.paired_bootstrap_ci(r0n, r1n, idx)
    rep = ci["d2_ratio_of_resampled_means"]["replicate_mean_diagnostic_only"]
    o = obs["d2_observed_ratio_of_means"]
    _check(o > v02c.D2_RATIO_MAX, "FIXTURE_CONSTRUCTION_FAILED",
           f"observed ratio {o!r} should exceed 0.95 in this fixture")
    _check(rep <= v02c.D2_RATIO_MAX, "FIXTURE_CONSTRUCTION_FAILED",
           f"replicate mean {rep!r} should be <= 0.95 in this "
           f"fixture (the straddle the separation test needs)")
    _check(rep != o, "FIXTURE_CONSTRUCTION_FAILED",
           "replicate mean must differ from the observed ratio")

    meas = _synth_meas(psnr_r0, psnr_r1, r0n, r1n)
    primary = _primary_comparison(meas, meas, idx)
    _check(primary["nmse_ratio"]["observed_ratio_of_means"] == o,
           "FIXTURE_OBSERVED_WIRING_FAILED",
           "primary comparison must carry the OBSERVED ratio")
    _check(primary["d2"]["observed"] == o and not primary["d2"]["pass"],
           "FIXTURE_OBSERVED_WIRING_FAILED",
           "D2 must fail on the OBSERVED ratio despite the replicate "
           "mean being inside threshold")
    _check(primary["d1"]["pass"], "FIXTURE_CONSTRUCTION_FAILED",
           "D1 was arranged to pass in this fixture")
    routing = _routing(primary)
    _check(routing["label"] == ROUTE_NO_SUPPORT,
           "FIXTURE_OBSERVED_WIRING_FAILED",
           f"routing must follow the observed statistic; got {routing}")
    _record("f01_observed_vs_ci_separation", True,
            f"observed {o!r} fails D2 practical; replicate mean "
            f"{rep!r} demonstrably NOT wired into acceptance")


def _f02_acceptance_boundaries() -> None:
    """Exact D1/D2 threshold boundaries through the registered
    two-conjunct rule (mean thresholds inclusive; CI null exclusions
    strict)."""
    cases = [
        # (d1_mean, d1_lo, d2_mean, d2_hi, d1_pass, d2_pass)
        (0.50, 0.1, 0.95, 0.99, True, True),      # inclusive means pass
        (0.50, 0.0, 0.95, 1.0, False, False),    # CI nulls strict
        (0.49, 0.1, 0.96, 0.99, False, False),   # practical fails
        (0.60, -0.1, 0.90, 1.01, False, False),  # uncertainty fails
        (0.60, 0.001, 0.90, 0.999, True, True),  # just inside
    ]
    for d1m, d1l, d2m, d2h, e1, e2 in cases:
        acc = v02c.d1d2_acceptance({"mean": d1m, "ci95": [d1l, 1.0]},
                                   {"mean": d2m, "ci95": [0.5, d2h]})
        _check(acc["d1"]["pass"] == e1 and acc["d2"]["pass"] == e2,
               "FIXTURE_BOUNDARY_FAILED",
               f"case {(d1m, d1l, d2m, d2h)}: expected "
               f"({e1}, {e2}), got ({acc['d1']['pass']}, "
               f"{acc['d2']['pass']})")
    _record("f02_acceptance_boundaries", True,
            f"{len(cases)} boundary cases exact")


def _f03_routing_labels() -> None:
    """D1 AND D2 both pass -> READOUT_LIMITATION_SUPPORTED with detail
    POSTERIOR_READOUT_SUPPORT; the failing case -> the negative
    route."""
    rng = np.random.Generator(np.random.PCG64(FIXTURE_SEED + 1))
    r0n = np.exp(0.1 * rng.standard_normal(N_HOLDOUT))
    r1n = r0n * (0.90 + 0.01 * rng.standard_normal(N_HOLDOUT))
    psnr_r0 = 30.0 + 0.5 * rng.standard_normal(N_HOLDOUT)
    psnr_r1 = psnr_r0 + 0.7 + 0.05 * rng.standard_normal(N_HOLDOUT)
    idx = v02c.boot_index_matrix()
    primary = _primary_comparison(_synth_meas(psnr_r0, psnr_r1, r0n, r1n),
                                  _synth_meas(psnr_r0, psnr_r1, r0n, r1n),
                                  idx)
    _check(primary["d1"]["pass"] and primary["d2"]["pass"],
           "FIXTURE_CONSTRUCTION_FAILED",
           f"clear-pass construction failed: {primary['d1']} / "
           f"{primary['d2']}")
    routing = _routing(primary)
    _check(routing["label"] == ROUTE_SUPPORT
           and routing["detail"] == ROUTE_SUPPORT_DETAIL,
           "FIXTURE_ROUTING_FAILED", f"unexpected routing {routing}")
    bad = _routing({"d1": {"pass": True}, "d2": {"pass": False}})
    _check(bad["label"] == ROUTE_NO_SUPPORT and "detail" not in bad,
           "FIXTURE_ROUTING_FAILED",
           f"single-conjunct failure must route negative: {bad}")
    _record("f03_routing_labels", True,
            "support + detail and both negative routes exact")


def _f04_pairing_order() -> None:
    """The measured identity sequence must equal the manifest order
    position by position; a single swap is a hard ERROR."""
    entries = [{"file": f"synthetic/vol{j:03d}.h5", "slice_index": j}
               for j in range(N_HOLDOUT)]
    holdman = {"manifest_sha256": HOLDOUT_MANIFEST_SHA256,
               "n_volumes": N_HOLDOUT, "entries": entries}
    states = [{"identity": {"file": e["file"],
                            "slice_index": e["slice_index"]}}
              for e in entries]
    _verify_pairing(states, holdman)          # exact order: no raise
    swapped = list(states)
    swapped[10], swapped[11] = swapped[11], swapped[10]
    _expect_error(lambda: _verify_pairing(swapped, holdman),
                  "PAIRING_ORDER_MISMATCH", "swapped identities")
    short = states[:-1]
    _expect_error(lambda: _verify_pairing(short, holdman),
                  "STATE_COUNT_MISMATCH", "truncated states")
    _record("f04_pairing_order", True,
            "exact order accepted; swap and truncation refused")


def _f05_single_stream_consumption() -> None:
    """One-and-only-one PCG64(4) consumption: fresh-generator matrices
    are byte-identical (the registered consumption is reproducible);
    a SECOND consecutive draw from one generator differs (the module
    draws exactly once per run); a tampered matrix fails the pin."""
    m1 = v02c.boot_index_matrix()
    m2 = v02c.boot_index_matrix()
    _check(np.array_equal(m1, m2), "FIXTURE_STREAM_FAILED",
           "fresh-generator matrices must be byte-identical")
    v02c.verify_boot_index_pin(m1)
    g = np.random.Generator(np.random.PCG64(v02c.BOOTSTRAP_SEED))
    _ = g.integers(0, v02c.BOOTSTRAP_N_VOLUMES,
                   size=(v02c.BOOTSTRAP_B, v02c.BOOTSTRAP_N_VOLUMES))
    second = g.integers(0, v02c.BOOTSTRAP_N_VOLUMES,
                        size=(v02c.BOOTSTRAP_B, v02c.BOOTSTRAP_N_VOLUMES))
    _check(not np.array_equal(m1, second), "FIXTURE_STREAM_FAILED",
           "a chained second draw must differ from the first")
    tampered = m1.copy()
    tampered[0, 0] = (tampered[0, 0] + 1) % v02c.BOOTSTRAP_N_VOLUMES
    _expect_error(lambda: v02c.verify_boot_index_pin(tampered),
                  "STREAM_PIN_MISMATCH", "tampered index matrix")
    _record("f05_single_stream_consumption", True,
            "single consumption reproducible and pin-verified; "
            "chained draw and tamper refused")


def _f06_secondary_cannot_route() -> None:
    """Secondary statistics are descriptive ONLY: strongly negative
    secondaries cannot veto a primary pass, and strongly positive
    secondaries cannot rescue a primary fail."""
    idx = v02c.boot_index_matrix()
    rng = np.random.Generator(np.random.PCG64(FIXTURE_SEED + 2))
    # Case 1: primary passes; absolute recovery strongly negative.
    n0 = np.exp(0.1 * rng.standard_normal(N_HOLDOUT))
    step0 = _synth_meas(np.full(N_HOLDOUT, 40.0),
                        np.full(N_HOLDOUT, 39.0), n0, n0)
    fin = _synth_meas(np.full(N_HOLDOUT, 30.0),
                      np.full(N_HOLDOUT, 30.7),
                      n0, n0 * (0.90 + 0.01 * rng.standard_normal(
                          N_HOLDOUT)))
    primary = _primary_comparison(step0, fin, idx)
    _check(primary["d1"]["pass"] and primary["d2"]["pass"],
           "FIXTURE_CONSTRUCTION_FAILED", "case-1 primary must pass")
    sec = _secondary_comparisons(step0, fin)
    _check(sec["absolute_recovery_vs_initialisation"]
           ["mean_psnr_difference_db"] < 0,
           "FIXTURE_CONSTRUCTION_FAILED",
           "case-1 secondary must be strongly negative")
    _check(_routing(primary)["label"] == ROUTE_SUPPORT,
           "FIXTURE_SECONDARY_ROUTED",
           "negative secondary vetoed a primary pass")
    # Case 2: primary fails; absolute recovery strongly positive.
    fin2 = _synth_meas(np.full(N_HOLDOUT, 30.0),
                       np.full(N_HOLDOUT, 30.2), n0, n0 * 0.99)
    step0b = _synth_meas(np.full(N_HOLDOUT, 25.0),
                         np.full(N_HOLDOUT, 25.1), n0, n0)
    primary2 = _primary_comparison(step0b, fin2, idx)
    _check(not (primary2["d1"]["pass"] and primary2["d2"]["pass"]),
           "FIXTURE_CONSTRUCTION_FAILED", "case-2 primary must fail")
    sec2 = _secondary_comparisons(step0b, fin2)
    _check(sec2["absolute_recovery_vs_initialisation"]
           ["mean_psnr_difference_db"] > 0,
           "FIXTURE_CONSTRUCTION_FAILED",
           "case-2 secondary must be strongly positive")
    _check(_routing(primary2)["label"] == ROUTE_NO_SUPPORT,
           "FIXTURE_SECONDARY_ROUTED",
           "positive secondary rescued a primary fail")
    _record("f06_secondary_cannot_route", True,
            "secondaries demonstrably inert in both directions")


def _f07_no_verdict() -> None:
    """The facts carry no 'verdict' key anywhere; the scanner rejects
    an injected one."""
    v02e._no_verdict_scan({"a": {"b": [1, {"c": 2}]}}, "facts")
    _expect_error(
        lambda: v02e._no_verdict_scan(
            {"routing": {"label": "X", "verdict": "INJECTED"}},
            "facts"),
        "FACTS_SCHEMA_VIOLATION", "injected verdict key")
    _record("f07_no_verdict", True,
            "clean scan passes; injected verdict refused")


def _f08_semantic_reproducibility(env) -> None:
    """Semantic scope excludes telemetry: two synthetic facts records
    differing ONLY in run/timing/diagnostic telemetry produce the
    SAME semantic sha; one differing scientific number changes it."""
    base = {"schema": FACTS_SCHEMA,
            "routing": {"label": ROUTE_NO_SUPPORT},
            "primary_comparison": {"d1": {"observed": 0.1},
                                   "d2": {"observed": 0.99}},
            "run": {"utc": "2026-09-11T00:00:00+00:00",
                    "wall_clock_s": 1.0},
            "timing": {"state_construction_seconds": 1.0},
            "r1_sampling_diagnostics": {"per_volume": []}}
    import copy
    f1, f2, f3 = copy.deepcopy(base), copy.deepcopy(base), \
        copy.deepcopy(base)
    f2["run"]["utc"] = "2027-01-01T00:00:00+00:00"
    f2["run"]["wall_clock_s"] = 9999.0
    f2["timing"]["state_construction_seconds"] = 42.0
    f3["primary_comparison"]["d1"]["observed"] = 0.100001
    shas = []
    for f in (f1, f2, f3):
        sem = {k: v for k, v in f.items() if k not in SEMANTIC_EXCLUDED}
        env.attach_semantic_hash(f, sem)
        shas.append(f["semantic_sha256"])
    _check(shas[0] == shas[1], "FIXTURE_SEMANTIC_SCOPE_FAILED",
           f"telemetry-only difference changed the semantic sha "
           f"{shas[0][:12]} vs {shas[1][:12]}")
    _check(shas[0] != shas[2], "FIXTURE_SEMANTIC_SCOPE_FAILED",
           "a scientific change must change the semantic sha")
    _record("f08_semantic_reproducibility", True,
            "telemetry-invariant, science-sensitive semantic scope")


def _f09_refusal_paths(tmp_dir: str) -> None:
    """Every refusal path: logger.error + typed raise; nothing is
    coerced, skipped, or silently passed."""
    _expect_error(lambda: _verify_v02h_evidence(
        str(Path(tmp_dir) / "does_not_exist.json")),
        "V02H_EVIDENCE_MISSING", "missing V02H evidence")
    bogus = Path(tmp_dir) / "bogus_evidence.json"
    bogus.write_text(json.dumps({"schema": V02H_EVIDENCE_SCHEMA,
                                 "outcome": {"label":
                                             "OUTCOME_A_R2_NO_OP"}}))
    _expect_error(lambda: _verify_v02h_evidence(str(bogus)),
                  "V02H_EVIDENCE_MISMATCH", "wrong-sha V02H evidence")
    _expect_error(lambda: _check_v02h_content(
        {"schema": "wrong", "outcome": {"label": "OUTCOME_A_R2_NO_OP"}}),
        "V02H_EVIDENCE_SCHEMA_MISMATCH", "evidence schema drift")
    _expect_error(lambda: _check_v02h_content(
        {"schema": V02H_EVIDENCE_SCHEMA,
         "outcome": {"label": "OUTCOME_B_R2_NONTRIVIAL"}}),
        "V02H_EVIDENCE_OUTCOME_MISMATCH", "evidence outcome drift")
    _expect_error(lambda: _verify_holdout(
        {"manifest_sha256": "0" * 64, "n_volumes": N_HOLDOUT,
         "entries": [{}] * N_HOLDOUT}),
        "HOLDOUT_MANIFEST_DRIFT", "holdout manifest drift")
    _expect_error(lambda: _verify_holdout(
        {"manifest_sha256": HOLDOUT_MANIFEST_SHA256,
         "n_volumes": N_HOLDOUT - 1,
         "entries": [{}] * N_HOLDOUT}),
        "HOLDOUT_POPULATION_MISMATCH", "holdout population drift")
    _expect_error(lambda: _verify_pm_bank(
        {"bank_sha256": "0" * 64, "manifest_sha256": "0" * 64,
         "bank": np.zeros((Z_DIAG_N, FLOW_DIM))}),
        "PM_BANK_MISMATCH", "PM bank drift")
    _expect_error(lambda: v02c.observed_d1d2(
        np.zeros(N_HOLDOUT), np.ones(N_HOLDOUT)),
        "BOOTSTRAP_DENOMINATOR_ZERO", "zero observed denominator")
    _expect_error(lambda: v02c.observed_d1d2(
        np.ones(10), np.ones(10)),
        "BOOTSTRAP_INPUT_INVALID", "wrong-length paired arrays")
    _expect_error(lambda: v02c.observed_d1d2(
        np.full(N_HOLDOUT, np.nan), np.ones(N_HOLDOUT)),
        "BOOTSTRAP_INPUT_INVALID", "non-finite paired arrays")
    idx = v02c.boot_index_matrix()
    r0z = np.zeros(N_HOLDOUT)
    r0z[73] = 1.0
    _expect_error(lambda: v02c.paired_bootstrap_ci(
        r0z, np.ones(N_HOLDOUT), idx),
        "BOOTSTRAP_DENOMINATOR_ZERO", "zero resampled denominator")
    # publication rerun guard
    prior = {"semantic_sha256": "a" * 64}
    auth = Path(tmp_dir) / f"{FACTS_PREFIX}.json"
    auth.write_text(json.dumps(prior))
    sidecar = Path(str(auth) + ".sha256")
    sidecar.write_text(f"{_file_sha256(auth)}  {auth.name}\n")
    _expect_error(lambda: _guard_rerun({"semantic_sha256": "b" * 64},
                                       tmp_dir),
                  "SEMANTIC_RERUN_MISMATCH", "divergent rerun")
    sidecar.unlink()
    _expect_error(lambda: _guard_rerun({"semantic_sha256": "a" * 64},
                                       tmp_dir),
                  "RERUN_PRIOR_ARTEFACT_MISMATCH", "unverifiable prior")
    _record("f09_refusal_paths", True,
            "12 refusal branches exercised; all typed, none coerced")


def _f10_d3_subset_gate() -> None:
    """The D3-subset gate seam (v02m.d3_monitor_positions): a synthetic
    manifest reproducing the frozen PCG64(2) draw -- bound to a
    synthetic holdout and carrying the registered manifest-SHA pin --
    is accepted; a positions-only corruption is refused with
    D3_DRAW_MISMATCH. This closes the coverage gap behind the v0.2
    authoritative-only wrong-alias defect (v02m bound to the wrong
    module), which the v0.2 fixture suite never reached."""
    positions = [int(p) for p in v02m.d3_monitor_positions()]
    entries = [{"file": f"synthetic/vol{j:03d}.h5", "slice_index": j}
               for j in range(N_HOLDOUT)]
    holdman = {"manifest_sha256": HOLDOUT_MANIFEST_SHA256,
               "n_volumes": N_HOLDOUT, "entries": entries}
    d3_entries = [{"holdout_position": pos, "draw_rank": r,
                   "file": entries[pos]["file"],
                   "slice_index": entries[pos]["slice_index"]}
                  for r, pos in enumerate(positions)]
    d3man = {"n": N_D3_SUBSET, "entries": d3_entries,
             "manifest_sha256": D3_SUBSET_MANIFEST_SHA256}
    _verify_d3_subset(d3man, holdman)       # frozen draw: no raise
    corrupt = [dict(e) for e in d3_entries]
    (corrupt[0]["holdout_position"],
     corrupt[1]["holdout_position"]) = positions[1], positions[0]
    corrupt_d3man = {"n": N_D3_SUBSET, "entries": corrupt,
                     "manifest_sha256": D3_SUBSET_MANIFEST_SHA256}
    _expect_error(lambda: _verify_d3_subset(corrupt_d3man, holdman),
                  "D3_DRAW_MISMATCH", "positions-only corruption")
    _record("f10_d3_subset_gate", True,
            "frozen-draw reproduction accepted via the "
            "v02m.d3_monitor_positions seam; positions-only "
            "corruption refused with D3_DRAW_MISMATCH")


def run_fixtures() -> dict:
    """The assembly fixture suite. Any FAIL -> exit 2; no fixture
    touches the real holdout or publishes anything."""
    _FIXTURE_RESULTS.clear()
    env = _env()
    v02c.pin_deterministic_backend(env.torch)
    tmp_dir = Path(tempfile.mkdtemp(prefix="v02g_fixtures_"))
    try:
        _f01_observed_vs_ci_separation()
        _f02_acceptance_boundaries()
        _f03_routing_labels()
        _f04_pairing_order()
        _f05_single_stream_consumption()
        _f06_secondary_cannot_route()
        _f07_no_verdict()
        _f08_semantic_reproducibility(env)
        _f09_refusal_paths(str(tmp_dir))
        _f10_d3_subset_gate()
    except (V02GError, v02c.V02CError) as exc:
        _record("suite_abort", False, f"{exc.error_code}: {exc.reason}")
    n_fail = sum(1 for r in _FIXTURE_RESULTS if not r["ok"])
    report = {"stage": STAGE,
              "stage_version": f"{__abbr__} v{__version__}",
              "mode": "fixtures",
              "fixtures": list(_FIXTURE_RESULTS),
              "n_pass": sum(1 for r in _FIXTURE_RESULTS if r["ok"]),
              "n_fail": n_fail}
    print(json.dumps(report, indent=1))
    if n_fail:
        _fail("SELFTEST_FIXTURE_FAILED",
              f"{n_fail} fixture(s) failed; the authoritative driver "
              f"must not run")
    logger.info("[%s] fixture suite %d/%d green", __abbr__,
                report["n_pass"], len(_FIXTURE_RESULTS))
    return report


# ---------------------------------------------------------------------------
# CLI. Exit taxonomy: 0 = the mode completed. For the authoritative mode
# BOTH scientific routes (READOUT_LIMITATION_SUPPORTED and
# TESTED_READOUTS_DO_NOT_RECOVER_RECONSTRUCTION_SUPPORT) are completed
# experiment outcomes, not failures -- the exit code carries validity,
# never the scientific direction. 2 = ERROR: any typed refusal, validity
# failure, or unexpected exception voids the run (no partial
# classification, nothing published).
# ---------------------------------------------------------------------------

EXIT_PASS = 0
EXIT_ERROR = 2


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"{__abbr__} v{__version__} -- authoritative V02 "
                    f"readout diagnostic (R0 z=0 vs R1 posterior mean; "
                    f"V02AMEND v1.0 LOCKED, LudWiki id 147)")
    ap.add_argument("--mode", choices=["fixtures", "probe",
                                       "authoritative"],
                    default="fixtures")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--manifest-dir", default=None)
    ap.add_argument("--train-record", default=None)
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--p0s-facts", default=None)
    ap.add_argument("--p4-stats2", default=None)
    ap.add_argument("--implb-facts", default=None)
    ap.add_argument("--v02h-evidence", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--probe-dir", default=None)
    ap.add_argument("--probe-items", type=int, default=4)
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()
    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode="w"))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s "
                               "%(message)s")
    try:
        if args.mode == "fixtures":
            report = run_fixtures()
            logger.info("[%s] fixtures mode: %d/%d green; NOTHING "
                        "published", __abbr__, report["n_pass"],
                        len(report["fixtures"]))
        elif args.mode == "probe":
            missing = [k for k in ("data_root", "manifest_dir",
                                   "train_record", "run_root",
                                   "p0s_facts", "p4_stats2",
                                   "implb_facts", "probe_dir")
                       if getattr(args, k) is None]
            if missing:
                logger.error("[%s] probe mode requires %s", __abbr__,
                             ", ".join("--" + k.replace("_", "-")
                                       for k in missing))
                return EXIT_ERROR
            run_probe(vars(args))
        else:
            missing = [k for k in ("data_root", "manifest_dir",
                                   "train_record", "run_root",
                                   "p0s_facts", "p4_stats2",
                                   "implb_facts", "v02h_evidence",
                                   "out_dir")
                       if getattr(args, k) is None]
            if missing:
                logger.error("[%s] authoritative mode requires %s",
                             __abbr__, ", ".join("--" + k.replace(
                                 "_", "-") for k in missing))
                return EXIT_ERROR
            facts = run(vars(args))
            logger.info("[%s] authoritative run complete: route=%s "
                        "detail=%s", __abbr__,
                        facts["routing"]["label"],
                        facts["routing"].get("detail", "-"))
    except (V02GError, v02c.V02CError) as exc:
        logger.error("[%s] %s: %s", __abbr__, exc.error_code, exc.reason)
        return EXIT_ERROR
    except Exception:  # noqa: BLE001 -- the registered boundary
        logger.exception("[%s] unexpected runtime failure", __abbr__)
        return EXIT_ERROR  # every unexpected failure is logged -> exit 2
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Changelog
#
# v0.1 (2026-09-11): initial authoritative V02 readout-diagnostic driver
#   (V02AMEND v1.0 LOCKED, id 147): R0 (z=0) vs R1 (posterior mean,
#   registered u-space averaging rule) on the 199 frozen holdout
#   identities, endpoints step 0 / 3258, primary comparison final-R1 vs
#   final-R0 ONLY. Incorporates the reviewer's five GO-with-revisions
#   items: observed-statistic/CI separation via the neutral SEQREF-V02C
#   helpers (never the V02H harness); canonical production seams only;
#   nine-assembly-fixture suite; timing_probe (EPHEMERAL) vs
#   authoritative_timing with all execution telemetry excluded from the
#   semantic scope; gate-9/gate-11 anchors pinned to the frozen V02E
#   facts with registered aggregation and tolerances. Decode accounting
#   199x(1+128)x2 = 398 R0 + 50,944 R1 = 51,342 total.
#
# v0.2 (2026-09-11): reviewer-mandated patch after the green v0.1
#   timing probe (projected 927.7 s full diagnostic). The V02H
#   evidence record no longer carries its filesystem path inside the
#   semantic payload; the path moves to the non-semantic run block,
#   satisfying the registered "paths excluded" contract.
#   run_fixtures() now clears _FIXTURE_RESULTS on entry so repeated
#   in-process invocation cannot accumulate stale rows; the exit-2
#   boundary comment reworded (behaviour unchanged: logged -> exit 2).
#
# v0.3 (2026-09-26): post-result fixture-only hardening. New fixture
#   f10_d3_subset_gate exercises the D3-subset gate seam
#   (v02m.d3_monitor_positions): a synthetic manifest reproducing the
#   frozen PCG64(2) draw -- bound to a synthetic holdout and carrying
#   the registered manifest-SHA pin -- is accepted; a positions-only
#   corruption is refused with the actual typed code D3_DRAW_MISMATCH
#   (reviewer-corrected; the gate has no D3_SUBSET_MISMATCH code --
#   count drift is MANIFEST_CONTENT_MISMATCH, holdout binding drift is
#   D3_SUBSET_NOT_IN_HOLDOUT, SHA drift is D3_SUBSET_MANIFEST_DRIFT).
#   This closes the coverage gap behind the v0.2 authoritative-only
#   wrong-alias defect (v02m bound to v02_train instead of
#   v02_manifests), which the v0.2 fixture suite never reached. v0.3
#   does NOT supersede or alter the authoritative V02G result produced
#   by v0.2 at commit 6ffce4e555f90a48daa5fbff8b841adc3dfd7b18
#   (semantic SHA
#   92b2ee1ff449d7a4e561945d6de98667f343bd52674cab017a98b1ed6a06df59);
#   no scientific logic, locked constant, decision gate, or semantic
#   scope changed, and no authoritative rerun is required or
#   performed. Suite: f01-f10.
