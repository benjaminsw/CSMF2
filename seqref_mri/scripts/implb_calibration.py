# SEQREF-IMPLB v0.3 -- scripts.implb_calibration
# LIFETIME: KEEP
# Purpose: IMPL-B spline-bound calibration for the registered free-coefficient
#   conditional flow (IMPLSPEC v0.1 + 2026-08-11 percentile amendment,
#   incorporated by reference from EXEC §13). Measures q = p99.9 of
#   |u_scaled| over the FROZEN P0S 256-slice corpus and publishes
#   B = SPLINE_MARGIN * q, frozen at PASS for IMPL -> FORMAL.
# Registered rule (IMPLSPEC, verbatim binding):
#   * corpus: the frozen P0S 256-slice subset, eval mode, dataset-index
#     (manifest) order. NEVER the §10.4 TINY corpus (contamination).
#   * per sample: eval mask -> RE-DERIVE the coordinate map from the
#     RECORDED acquired_columns (P3 consumer contract: "re-derivation is
#     mandatory, not optional"; a derived map_sha256 != recorded is ERROR)
#     -> x_norm / x0 = cond_in via the REGISTERED live `_prepare` (never
#     reimplemented) -> delta_k = fft2c(x_norm - x0) -> gather_unmeasured.
#   * scaling: the P4 /2 APPLIED affine pair at the same physical (r, c),
#     gathered through the same map; real and imaginary components
#     standardised SEPARATELY. Means/scales/floors/branch are NEVER
#     recomputed here.
#   * statistic: q = np.percentile(np.abs(u_scaled_scalar), 99.9,
#     method="linear") over the COMPLETE scalar corpus of separately
#     standardised real and imaginary components -- exactly
#     256 * 13,824 = 3,538,944 float64 values. Scalar absolute value, NOT
#     complex-coordinate magnitude. No subsampling, streaming, t-digest,
#     histogram or GPU path.
#   * B = 1.1 * q (SPLINE_PERCENTILE=p99.9, SPLINE_MARGIN=1.1; rule recovered
#     from the registered I3/D3 pilot, EXEC revision 20; the pilot VALUE is
#     superseded and never reused).
#   * NSF linear identity tails outside [-B, B] remain operative: outside
#     values are NOT errors and are NEVER clipped.
#   * P4 CONSUMPTION VALIDITY: applied_scale finite AND strictly > 0 at
#     every gathered location; applied_mean finite; branch == PER-LOCATION.
#   * ARITHMETIC PATH: CPU only, torch.set_num_threads(1), NumPy float64.
#   * verdict: PASS | ERROR only. There is NO BLOCK path in this stage.
# Parent-pair doctrine: BOTH registered pins (file SHA-256 + embedded
#   semantic_sha256), the authoritative sidecar (MANDATORY per the P3
#   consumer contract and the §9 file+sidecar pairing doctrine), schema,
#   stage and authoritative-PASS status.
# RUN MODE: exactly ONE scientific run mode -- the authoritative frozen-256
#   calibration. The approved scope is core + A1 + A3; A2 (operator
#   rehearsal) was explicitly skipped, so there is NO smoke mode. The
#   determinism sibling comes from RERUNNING the authoritative stage.
# CONVENTION: logger.error + raise on every failure path. No fallback, no
#   mock, no placeholder, no silent pass.
# Changelog
#   v0.1 (2026-08-12) Created against the registered IMPL-B contract
#     (IMPLSPEC v0.1 + 2026-08-11 percentile amendment). Never executed.
#   v0.2 (2026-08-12) Pre-execution review remediation (reviewer HOLD of
#     v0.1, three findings): smoke mode REMOVED (prefixes, --smoke, corpus
#     slicing, run-mode branching) -- single authoritative run mode per the
#     approved core + A1 + A3 scope; P3 and P4 /2 parent sidecars now
#     MANDATORY (PARENT_SIDECAR_MISSING / PARENT_SIDECAR_MISMATCH);
#     seqref_mri/src/preflight_parents_p3.py added to the IMPL-B-local
#     hashed closure. No calibration observation was made with v0.1.
#   v0.3 (2026-08-12) Pre-execution review remediation (reviewer HOLD of
#     v0.2): registered failure boundary added to main() -- a final
#     `except Exception` wraps ordinary runtime faults as
#     UNEXPECTED_RUNTIME_ERROR with a typed ERROR record (P1/P2/P3/P4
#     doctrine, verbatim pattern); KeyboardInterrupt/SystemExit
#     deliberately escape. Error records now carry code/run context on
#     both handlers. No calibration observation was made with v0.2.
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

from preflight_io import (canonical_hash, file_sha256,  # noqa: E402
                          verify_sidecar)
from preflight_parents import (EXIT_ERROR, EXIT_PASS,  # noqa: E402
                               REQUIRED_PREPARE_KEYS, StageError,
                               attach_semantic_hash, environment_record,
                               hash_project_code, publish_error,
                               publish_stage, require_finite, verify_parents)
from preflight_parents_p3 import (bind_mask_seed_provenance,  # noqa: E402
                                  dataset_provenance)
import residual_decoder as dec  # noqa: E402
from seqref_mri.src.fastmri_data import FastMRISliceDataset  # noqa: E402
from seqref_mri.scripts.train_base import _collate, _prepare  # noqa: E402

SCRIPT_ID = "SEQREF-IMPLB"
SCRIPT_VERSION = "v0.3"
FACTS_SCHEMA = "seqref-implb-facts/1"
FACTS_PREFIX = "implb_facts"
ERROR_PREFIX = "implb_error"
STAGE = "IMPL-B"

logger = logging.getLogger(SCRIPT_ID)

# ---- registered calibration constants (IMPLSPEC v0.1 + 2026-08-11 amendment)
GRID_H = 96
GRID_W = 96
EXPECTED_CORPUS_SLICES = 256          # the frozen P0S subset, never TINY
EXPECTED_ACQUIRED_COLUMNS = 24        # fixed by construction (mask_counts)
CENTRE_COLUMNS = frozenset(range(44, 52))   # 44..51 acquired, P2 convention
EXPECTED_N_FREE_COMPLEX = 6912        # 96 * (96 - 24)
EXPECTED_FLOW_DIM_REAL = 13824        # 2 * 6912, interleaved re/im
SPLINE_PERCENTILE = 99.9              # recovered pilot RULE (EXEC rev. 20)
PERCENTILE_METHOD = "linear"          # preregistered interpolation rule
SPLINE_MARGIN = 1.1                   # recovered pilot RULE (EXEC rev. 20)
EXPECTED_OBSERVATIONS = (EXPECTED_CORPUS_SLICES * EXPECTED_FLOW_DIM_REAL)
REGISTERED_P4_LOCATIONS = 8448        # 96 rows x 88 eligible columns

# ---- parent pins (EXEC §9.1/§13: BOTH hashes, registered pre-measurement).
# An unpinned load would accept any PASSing artefact; only these prove the
# parent is the artefact whose verdict the campaign closed.
P3_FACTS_SCHEMA = "seqref-p3-facts/2"
P3_FILE_SHA256 = ("27fe84dab5fb2edb5c0c6230d5c4b1dee37db33cc8a76ecd92"
                  "bb60de783b0bcf")
P3_SEMANTIC_SHA256 = ("eaab6776f6ec3a8af39ef5630a6160c9cb97d947758ad9"
                      "83d32eafa1f1cebbb3")
P4S2_FACTS_SCHEMA = "seqref-p4-stats/2"
P4S2_FILE_SHA256 = ("db0bf2c2e0cd3bd5e1ed86bf208198199f16df5b2fbbe0cc"
                    "72903c8791f85684")
P4S2_SEMANTIC_SHA256 = ("27bd5b693548d80b2bc837853469b5e62f3da6f0cb218"
                        "4e5db85fde5c9c91008")
# The mask generator is HASH-BOUND (same binding as P3/P4): the executing
# generator's source hash must EQUAL this value or the run is ERROR.
GENERATOR_SOURCE_SHA256 = ("610cc1d1d7968deebc88f645270e1baefb6589cb56841b"
                           "dd327450ca1069cb44")


# ---------------------------------------------------------------------------
# Generic dual-pin parent verification helpers
# ---------------------------------------------------------------------------

def _parent_file_sidecar(path: str, expected_file_sha: str,
                         label: str) -> tuple[dict, str]:
    """Existence, file-hash pin, MANDATORY sidecar verification, JSON
    parse. Every mismatch is ERROR: the parent must be THE artefact IMPL-B
    was registered against, never an unpinned lookalike. The sidecar is
    not optional: the P3 consumer contract requires verifying the file
    against its sidecar BEFORE use, and the §9 doctrine pairs every
    authoritative artefact with its sidecar."""
    if not os.path.isfile(path):
        logger.error("[%s] %s parent not found: %s", SCRIPT_ID, label, path)
        raise StageError("PARENT_NOT_FOUND",
                         f"the {label} parent artefact does not exist at "
                         f"{path}")
    file_sha = file_sha256(path)
    if file_sha != expected_file_sha:
        logger.error("[%s] %s parent file hash %s != pin %s", SCRIPT_ID,
                     label, file_sha, expected_file_sha)
        raise StageError("PARENT_FILE_HASH_MISMATCH",
                         f"the {label} parent hashes to {file_sha}, but the "
                         f"registered pin is {expected_file_sha}; the parent "
                         f"is not the artefact IMPL-B was registered "
                         f"against")
    if not os.path.isfile(path + ".sha256"):
        logger.error("[%s] %s sidecar missing beside %s", SCRIPT_ID, label,
                     path)
        raise StageError("PARENT_SIDECAR_MISSING",
                         f"the {label} parent at {path} has no sidecar; "
                         f"the consumer contract requires verifying the "
                         f"file against its sidecar before use, so a "
                         f"missing sidecar is ERROR, never a skipped "
                         f"check")
    try:
        verify_sidecar(path)
    except (OSError, RuntimeError) as exc:
        logger.error("[%s] %s sidecar verification failed: %s", SCRIPT_ID,
                     label, exc)
        raise StageError("PARENT_SIDECAR_MISMATCH",
                         f"the {label} sidecar does not verify against "
                         f"the pinned artefact: {exc}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            art = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.error("[%s] %s parent unparsable: %s", SCRIPT_ID, label, exc)
        raise StageError("PARENT_STRUCTURE_INVALID",
                         f"the {label} parent at {path} is not valid JSON: "
                         f"{exc}")
    return art, file_sha


def _parent_status_semantic(art: dict, *, schema: str, stage: str,
                            expected_semantic_sha: str, label: str) -> str:
    """Schema, stage, authoritative-PASS status, embedded semantic pin."""
    if art.get("schema") != schema:
        logger.error("[%s] %s schema %r != %r", SCRIPT_ID, label,
                     art.get("schema"), schema)
        raise StageError("PARENT_SCHEMA_MISMATCH",
                         f"expected schema {schema}, got "
                         f"{art.get('schema')!r}; IMPL-B rejects any other "
                         f"schema rather than reinterpret it")
    if art.get("stage") != stage:
        logger.error("[%s] %s stage %r != %r", SCRIPT_ID, label,
                     art.get("stage"), stage)
        raise StageError("PARENT_STRUCTURE_INVALID",
                         f"the {label} artefact records stage "
                         f"{art.get('stage')!r}, expected {stage!r}")
    if not (art.get("authoritative") and art.get("run_mode") == "authoritative"
            and art.get("verdict") == "PASS"):
        logger.error("[%s] %s is not an authoritative PASS: authoritative=%r "
                     "run_mode=%r verdict=%r", SCRIPT_ID, label,
                     art.get("authoritative"), art.get("run_mode"),
                     art.get("verdict"))
        raise StageError("PARENT_NOT_AUTHORITATIVE_PASS",
                         f"IMPL-B inherits only from an authoritative PASS "
                         f"{label} artefact")
    sem = art.get("semantic_sha256")
    if sem != expected_semantic_sha:
        logger.error("[%s] %s semantic hash %s != pin %s", SCRIPT_ID, label,
                     sem, expected_semantic_sha)
        raise StageError("PARENT_SEMANTIC_HASH_MISMATCH",
                         f"the {label} parent's embedded semantic_sha256 is "
                         f"{sem}, but the registered pin is "
                         f"{expected_semantic_sha}")
    return sem


# ---------------------------------------------------------------------------
# P3 parent: per-slice bindings under the mandatory re-derivation contract
# ---------------------------------------------------------------------------

def load_p3_parent(path: str, *, expected_file_sha: str = P3_FILE_SHA256,
                   expected_semantic_sha: str = P3_SEMANTIC_SHA256) -> dict:
    """Verify the authoritative P3 coordinate-map artefact against BOTH
    registered pins and extract the 256 per-slice bindings. The keyword
    pins default to the registered values; the self-test overrides them to
    reach each gate without touching the production defaults."""
    art, file_sha = _parent_file_sidecar(path, expected_file_sha, "P3")
    sem = _parent_status_semantic(art, schema=P3_FACTS_SCHEMA, stage="P3",
                                  expected_semantic_sha=expected_semantic_sha,
                                  label="P3")
    cmap_block = art.get("coordinate_map") or {}
    if (cmap_block.get("enumeration_rule") != dec.P3_FLATTEN_ORDER
            or cmap_block.get("complex_packing_order")
            != dec.P3_COMPLEX_PACKING_ORDER):
        logger.error("[%s] P3 enumeration/packing rule drift: %r / %r",
                     SCRIPT_ID, cmap_block.get("enumeration_rule"),
                     cmap_block.get("complex_packing_order"))
        raise StageError("PARENT_STRUCTURE_INVALID",
                         "the P3 artefact's enumeration or packing rule "
                         "differs from the live decoder constants; IMPL-B "
                         "must not reinterpret the map order")
    bindings = art.get("per_slice_bindings")
    if not isinstance(bindings, list) \
            or len(bindings) != EXPECTED_CORPUS_SLICES:
        logger.error("[%s] P3 per_slice_bindings is %s, expected a list of "
                     "%d", SCRIPT_ID,
                     ("missing" if bindings is None else
                      f"a list of {len(bindings)}"), EXPECTED_CORPUS_SLICES)
        raise StageError("PARENT_STRUCTURE_INVALID",
                         f"the P3 artefact must carry exactly "
                         f"{EXPECTED_CORPUS_SLICES} per-slice bindings (the "
                         f"frozen corpus); IMPL-B calibrates on no other "
                         f"population")
    required = ("dataset_index", "file", "slice_index", "split", "mask_seed",
                "acquired_columns", "mask_sha256", "map_sha256",
                "n_free_complex", "flow_dim_real")
    for k, b in enumerate(bindings):
        missing = [f for f in required if f not in b]
        if missing:
            logger.error("[%s] P3 binding %d lacks %s", SCRIPT_ID, k,
                         missing)
            raise StageError("PARENT_STRUCTURE_INVALID",
                             f"P3 per-slice binding {k} lacks {missing}; "
                             f"the consumer contract is unbindable")
    return {"path": path, "file_sha256": file_sha, "semantic_sha256": sem,
            "sidecar_verified": True, "bindings": bindings,
            "enumeration_rule": cmap_block.get("enumeration_rule"),
            "packing_order": cmap_block.get("complex_packing_order"),
            "grid_shape": cmap_block.get("grid_shape"),
            "consumer_contract": cmap_block.get("consumer_contract"),
            "parents_record": art.get("parents")}


# ---------------------------------------------------------------------------
# P4 /2 parent: PER-LOCATION applied affine pairs, indexed by physical (r,c)
# ---------------------------------------------------------------------------

def build_location_index(locations: list) -> dict:
    """(row, column) -> applied-pair record. Keyed by PHYSICAL grid
    location, never by packed free-coordinate index: under the
    per-realisation map the same packed index denotes different Fourier
    locations on different samples. Duplicate or malformed records are
    ERROR."""
    index: dict[tuple[int, int], dict] = {}
    required = ("row", "column", "applied_mean_re", "applied_mean_im",
                "applied_scale_re", "applied_scale_im")
    for rec in locations:
        missing = [f for f in required if f not in rec]
        if missing:
            logger.error("[%s] P4 /2 location record lacks %s: %s",
                         SCRIPT_ID, missing, rec)
            raise StageError("PARENT_STRUCTURE_INVALID",
                             f"a P4 /2 location record lacks {missing}")
        key = (int(rec["row"]), int(rec["column"]))
        if key in index:
            logger.error("[%s] duplicate P4 /2 location record at %s",
                         SCRIPT_ID, key)
            raise StageError("PARENT_STRUCTURE_INVALID",
                             f"duplicate P4 /2 location record at {key}")
        index[key] = rec
    return index


def applied_pair(loc_index: dict, r: int, c: int) -> tuple:
    """The registered P4 CONSUMPTION VALIDITY gate, applied at gather time
    to EVERY gathered location: the location must exist, applied_mean must
    be finite, applied_scale must be finite AND strictly > 0. Returns
    float64 (mean_re, scale_re, mean_im, scale_im). No fallback value is
    ever invented."""
    rec = loc_index.get((int(r), int(c)))
    if rec is None:
        logger.error("[%s] no P4 /2 applied pair at physical location "
                     "(%d, %d)", SCRIPT_ID, r, c)
        raise StageError("P4_LOCATION_MISSING",
                         f"the gathered free coordinate ({r}, {c}) has no "
                         f"P4 /2 applied pair; the map and the scaling "
                         f"artefact disagree")
    out = []
    for ch in ("re", "im"):
        mean = float(rec[f"applied_mean_{ch}"])
        scale = float(rec[f"applied_scale_{ch}"])
        if not np.isfinite(mean):
            logger.error("[%s] non-finite applied_mean_%s at (%d, %d): %r",
                         SCRIPT_ID, ch, r, c, mean)
            raise StageError("P4_APPLIED_PAIR_INVALID",
                             f"applied_mean_{ch} at ({r}, {c}) is {mean!r}; "
                             f"it must be finite")
        if not np.isfinite(scale) or scale <= 0.0:
            logger.error("[%s] invalid applied_scale_%s at (%d, %d): %r",
                         SCRIPT_ID, ch, r, c, scale)
            raise StageError("P4_APPLIED_PAIR_INVALID",
                             f"applied_scale_{ch} at ({r}, {c}) is "
                             f"{scale!r}; it must be finite AND strictly "
                             f"> 0 (no fallback)")
        out.append(np.float64(mean))
        out.append(np.float64(scale))
    return out[0], out[1], out[2], out[3]


def load_p4s2_parent(path: str, *,
                     expected_file_sha: str = P4S2_FILE_SHA256,
                     expected_semantic_sha: str = P4S2_SEMANTIC_SHA256
                     ) -> dict:
    """Verify the authoritative P4 /2 scaling-statistics artefact against
    BOTH registered pins; enforce the PER-LOCATION branch; index the
    applied pairs by physical (r, c). Means, scales, floors and the branch
    are NEVER recomputed here -- consumption only."""
    art, file_sha = _parent_file_sidecar(path, expected_file_sha, "P4 /2")
    sem = _parent_status_semantic(art, schema=P4S2_FACTS_SCHEMA, stage="P4",
                                  expected_semantic_sha=expected_semantic_sha,
                                  label="P4 /2")
    branch = (art.get("branch") or {}).get("selected")
    if branch != "PER-LOCATION":
        logger.error("[%s] P4 /2 branch.selected is %r, expected "
                     "PER-LOCATION", SCRIPT_ID, branch)
        raise StageError("P4_BRANCH_UNEXPECTED",
                         f"IMPL-B consumes only the PER-LOCATION applied "
                         f"affine pair; the pinned P4 /2 artefact records "
                         f"branch {branch!r}")
    locations = art.get("locations")
    if not isinstance(locations, list) \
            or len(locations) != REGISTERED_P4_LOCATIONS:
        logger.error("[%s] P4 /2 locations is %s, expected a list of %d",
                     SCRIPT_ID,
                     ("missing" if locations is None else
                      f"a list of {len(locations)}"),
                     REGISTERED_P4_LOCATIONS)
        raise StageError("PARENT_STRUCTURE_INVALID",
                         f"the authoritative P4 /2 artefact must carry all "
                         f"{REGISTERED_P4_LOCATIONS} eligible locations "
                         f"(coverage was gated at the parent stage)")
    loc_index = build_location_index(locations)
    # Load-time validation of the FULL table: a superset of the registered
    # per-gather rule (every gatherable location is checked here, and
    # applied_pair re-checks at gather time).
    for (r, c) in loc_index:
        applied_pair(loc_index, r, c)
    return {"path": path, "file_sha256": file_sha, "semantic_sha256": sem,
            "sidecar_verified": True, "branch": branch,
            "locations_order": art.get("locations_order"),
            "location_index": loc_index,
            "parents_record": art.get("parents")}


# ---------------------------------------------------------------------------
# Per-sample binding identity + mandatory map re-derivation (P3 consumer
# contract: "IMPL must verify ... then RE-DERIVE each realisation's map from
# the recorded acquired_columns under the published enumeration rule and
# REQUIRE the derived map_sha256 to equal the recorded value; a mismatch is
# ERROR. Re-derivation is mandatory, not optional")
# ---------------------------------------------------------------------------

def verify_binding_identity(row: dict, binding: dict, height: int,
                            width: int):
    """Bind sample <-> mask <-> map for ONE corpus slice and return the
    re-derived CoordinateMap. `row` carries the LIVE realisation
    (dataset_index, file, slice_index, split, mask_seed, live_columns from
    the applied batch mask); `binding` is the RECORDED P3 entry. Any
    disagreement is ERROR: it is code, data or provenance drift, not a
    data verdict."""
    order = row.get("corpus_order")
    for field in ("dataset_index", "file", "slice_index", "split",
                  "mask_seed"):
        live, recorded = row.get(field), binding.get(field)
        if field in ("dataset_index", "slice_index", "mask_seed"):
            live, recorded = int(live), int(recorded)
        if live != recorded:
            logger.error("[%s] corpus position %s: live %s=%r != recorded "
                         "%r", SCRIPT_ID, order, field, live, recorded)
            raise StageError("BINDING_IDENTITY_MISMATCH",
                             f"the live realisation at corpus position "
                             f"{order} disagrees with the P3 binding on "
                             f"{field} ({live!r} != {recorded!r}); the "
                             f"frozen corpus is not being traversed as "
                             f"recorded")
    live_cols = tuple(int(c) for c in row["live_columns"])
    recorded_cols = tuple(int(c) for c in binding["acquired_columns"])
    if live_cols != recorded_cols:
        logger.error("[%s] corpus position %s: live acquired columns %s != "
                     "recorded %s", SCRIPT_ID, order, live_cols,
                     recorded_cols)
        raise StageError("MASK_LIVE_BINDING_MISMATCH",
                         f"the live eval-mode mask at corpus position "
                         f"{order} does not reproduce the recorded P3 "
                         f"acquired columns; this is generator or "
                         f"provenance drift, not a verdict")
    if len(recorded_cols) != EXPECTED_ACQUIRED_COLUMNS:
        logger.error("[%s] corpus position %s: %d acquired columns, "
                     "expected %d", SCRIPT_ID, order, len(recorded_cols),
                     EXPECTED_ACQUIRED_COLUMNS)
        raise StageError("BINDING_ACQUIRED_COUNT_UNEXPECTED",
                         f"the recorded mask at corpus position {order} "
                         f"has {len(recorded_cols)} acquired columns; the "
                         f"count is fixed at {EXPECTED_ACQUIRED_COLUMNS} "
                         f"by construction, so this is a broken generator "
                         f"contract")
    if width == GRID_W and not CENTRE_COLUMNS.issubset(recorded_cols):
        logger.error("[%s] corpus position %s: centre columns 44..51 not "
                     "acquired in %s", SCRIPT_ID, order, recorded_cols)
        raise StageError("BINDING_CENTRE_NOT_ACQUIRED",
                         f"the recorded mask at corpus position {order} "
                         f"lacks centre columns 44..51; the registered "
                         f"mask validity contract is broken")
    mask_sha = canonical_hash({"width": width,
                               "selected_columns": list(recorded_cols)})
    if mask_sha != binding["mask_sha256"]:
        logger.error("[%s] corpus position %s: recomputed mask_sha256 %s "
                     "!= recorded %s", SCRIPT_ID, order, mask_sha,
                     binding["mask_sha256"])
        raise StageError("MASK_HASH_MISMATCH",
                         f"the mask hash recomputed from the recorded "
                         f"acquired columns at corpus position {order} "
                         f"differs from the recorded value")
    cmap = dec.build_coordinate_map(list(recorded_cols), height, width)
    map_sha = cmap.payload()["map_payload_sha256"]
    if map_sha != binding["map_sha256"]:
        logger.error("[%s] corpus position %s: RE-DERIVED map_sha256 %s "
                     "!= recorded %s", SCRIPT_ID, order, map_sha,
                     binding["map_sha256"])
        raise StageError("MAP_HASH_MISMATCH",
                         f"the map re-derived from the recorded acquired "
                         f"columns at corpus position {order} hashes to "
                         f"{map_sha}, but P3 recorded "
                         f"{binding['map_sha256']}; re-derivation is "
                         f"mandatory and a mismatch is ERROR")
    if cmap.n_free_complex != EXPECTED_N_FREE_COMPLEX \
            or cmap.flow_dim_real != EXPECTED_FLOW_DIM_REAL \
            or int(binding["n_free_complex"]) != EXPECTED_N_FREE_COMPLEX \
            or int(binding["flow_dim_real"]) != EXPECTED_FLOW_DIM_REAL:
        logger.error("[%s] corpus position %s: n_free=%d flow_dim=%d "
                     "(binding %d/%d), expected %d/%d", SCRIPT_ID, order,
                     cmap.n_free_complex, cmap.flow_dim_real,
                     binding["n_free_complex"], binding["flow_dim_real"],
                     EXPECTED_N_FREE_COMPLEX, EXPECTED_FLOW_DIM_REAL)
        raise StageError("BINDING_DIMENSION_MISMATCH",
                         f"the free-coefficient dimensions at corpus "
                         f"position {order} differ from the registered "
                         f"{EXPECTED_N_FREE_COMPLEX}/"
                         f"{EXPECTED_FLOW_DIM_REAL}")
    return cmap


# ---------------------------------------------------------------------------
# Standardisation + scalar packing (registered arithmetic path)
# ---------------------------------------------------------------------------

def standardise_free(u: torch.Tensor, cmap, loc_index: dict) -> tuple:
    """Standardise the gathered free coefficients with the P4 /2 APPLIED
    affine pair at the SAME physical (r, c), real and imaginary components
    SEPARATELY, in float64. Returns (re_scaled, im_scaled), each (n_free,)
    float64 in the map's canonical flatten order."""
    if not torch.is_complex(u) or u.dim() != 1 \
            or int(u.shape[0]) != cmap.n_free_complex:
        logger.error("[%s] free-coefficient vector has layout %s %s, "
                     "expected a complex 1-D vector of length %d",
                     SCRIPT_ID, tuple(u.shape), u.dtype,
                     cmap.n_free_complex)
        raise StageError("STATE_LAYOUT_UNEXPECTED",
                         f"gathered free vector {tuple(u.shape)} "
                         f"{u.dtype} does not match the map's n_free="
                         f"{cmap.n_free_complex}")
    u_np = np.asarray(u.detach().to(torch.complex128).cpu().numpy())
    if not (np.isfinite(u_np.real).all() and np.isfinite(u_np.imag).all()):
        logger.error("[%s] non-finite free coefficient in gathered vector",
                     SCRIPT_ID)
        raise StageError("U_NON_FINITE",
                         "the gathered free-coefficient vector contains a "
                         "non-finite component; no fallback is permitted")
    n = cmap.n_free_complex
    re_scaled = np.empty(n, dtype=np.float64)
    im_scaled = np.empty(n, dtype=np.float64)
    re = u_np.real.astype(np.float64)
    im = u_np.imag.astype(np.float64)
    for k in range(n):
        m_re, s_re, m_im, s_im = applied_pair(
            loc_index, int(cmap.free_rows[k]), int(cmap.free_cols[k]))
        re_scaled[k] = (re[k] - m_re) / s_re
        im_scaled[k] = (im[k] - m_im) / s_im
    return re_scaled, im_scaled


def pack_scalar_corpus(re_scaled: np.ndarray,
                       im_scaled: np.ndarray) -> np.ndarray:
    """Interleaved re/im packing per complex coordinate, P3 canonical
    order: scalar[2k] = re_scaled[k], scalar[2k+1] = im_scaled[k]. One
    sample contributes exactly flow_dim_real = 2 * n_free scalars."""
    if re_scaled.shape != im_scaled.shape or re_scaled.ndim != 1:
        logger.error("[%s] packing inputs %s / %s are not equal 1-D shapes",
                     SCRIPT_ID, re_scaled.shape, im_scaled.shape)
        raise StageError("STATE_LAYOUT_UNEXPECTED",
                         "standardised components must be equal-shape 1-D "
                         "arrays before packing")
    out = np.empty(2 * re_scaled.shape[0], dtype=np.float64)
    out[0::2] = re_scaled
    out[1::2] = im_scaled
    return out


def compute_q_b(corpus: np.ndarray, *, expected_count: int) -> dict:
    """The registered statistic. EXACTLY expected_count float64 scalars
    (authoritative: 256 * 13,824 = 3,538,944); scalar absolute value, NOT
    complex-coordinate magnitude; np.percentile method="linear"; no
    subsampling, streaming, t-digest, histogram or GPU path. B =
    SPLINE_MARGIN * q. Outside values are diagnostics, NEVER errors and
    NEVER clipped (the NSF linear identity tails remain operative)."""
    if corpus.ndim != 1 or corpus.dtype != np.float64:
        logger.error("[%s] corpus has shape %s dtype %s, expected 1-D "
                     "float64", SCRIPT_ID, corpus.shape, corpus.dtype)
        raise StageError("CORPUS_LAYOUT_UNEXPECTED",
                         "the scalar corpus must be a 1-D float64 array")
    if int(corpus.shape[0]) != expected_count:
        logger.error("[%s] corpus holds %d scalars, expected exactly %d",
                     SCRIPT_ID, corpus.shape[0], expected_count)
        raise StageError("CORPUS_SIZE_MISMATCH",
                         f"the scalar corpus holds {corpus.shape[0]} "
                         f"values, expected exactly {expected_count}; the "
                         f"registered observation count is not negotiable")
    if not np.isfinite(corpus).all():
        logger.error("[%s] non-finite value in the scalar corpus",
                     SCRIPT_ID)
        raise StageError("CORPUS_NON_FINITE",
                         "the scalar corpus contains a non-finite value; "
                         "the percentile is undefined and no fallback is "
                         "permitted")
    abs_c = np.abs(corpus)
    q = float(np.percentile(abs_c, SPLINE_PERCENTILE,
                            method=PERCENTILE_METHOD))
    if not np.isfinite(q) or q <= 0.0:
        logger.error("[%s] p%.1f of |u_scaled| is %r", SCRIPT_ID,
                     SPLINE_PERCENTILE, q)
        raise StageError("Q_INVALID",
                         f"the calibrated percentile q is {q!r}; it must "
                         f"be finite and strictly positive for B to be "
                         f"defined")
    B = float(np.float64(SPLINE_MARGIN) * np.float64(q))
    require_finite({"B": B}, "IMPL-B spline bound")
    n_beyond = int((abs_c > B).sum())
    max_abs = float(abs_c.max())
    return {"percentile": SPLINE_PERCENTILE,
            "method": PERCENTILE_METHOD,
            "method_literal": "np.percentile(np.abs(u_scaled_scalar), "
                              "99.9, method=\"linear\")",
            "numpy_version": np.__version__,
            "q": q,
            "margin": SPLINE_MARGIN,
            "B": B,
            "observation_count": int(corpus.shape[0]),
            "diagnostics": {
                "max_abs_u_scaled": max_abs,
                "count_abs_u_scaled_beyond_B": n_beyond,
                "fraction_abs_u_scaled_beyond_B":
                    n_beyond / int(corpus.shape[0]),
                "tail_note": "values outside [-B, B] are EXPECTED under "
                             "the NSF linear identity tails; they are "
                             "diagnostics, never errors, and are never "
                             "clipped"}}


# ---------------------------------------------------------------------------
# Generator pin (same binding as P3/P4: a GATE, not a record)
# ---------------------------------------------------------------------------

def enforce_generator_pin(seed_prov: dict) -> None:
    got = (seed_prov or {}).get("mask_seed_source_sha256")
    if not (seed_prov or {}).get("resolved") \
            or got != GENERATOR_SOURCE_SHA256:
        logger.error("[%s] generator hash %s != registered pin %s",
                     SCRIPT_ID, got, GENERATOR_SOURCE_SHA256)
        raise StageError("GENERATOR_HASH_MISMATCH",
                         f"the executing mask generator hashes to {got}, "
                         f"but the registered frame is hash-bound to "
                         f"{GENERATOR_SOURCE_SHA256}")


def _local_sha(repo_dir: str, relpath: str) -> str | None:
    path = os.path.join(repo_dir, relpath)
    return file_sha256(path) if os.path.isfile(path) else None


# ---------------------------------------------------------------------------
# Frozen-corpus traversal
# ---------------------------------------------------------------------------

def _collect(parents: dict, bindings: list, loc_index: dict,
             data_root: str, batch: int) -> tuple:
    """Traverse the FROZEN P0S 256-slice corpus in dataset-index (manifest)
    order, eval mode -- the complete corpus, never a prefix. Per sample:
    live mask -> binding identity -> MANDATORY map re-derivation ->
    registered `_prepare` (never reimplemented) -> delta_k = fft2c(x_norm
    - cond_in) -> gather_unmeasured -> P4 /2 applied-pair standardisation
    -> interleaved scalar pack."""
    indices = parents["subset_indices"]
    recorded_order = [int(b["dataset_index"]) for b in bindings]
    if recorded_order != list(indices):
        logger.error("[%s] P3 binding order %s... != frozen subset order "
                     "%s...", SCRIPT_ID, recorded_order[:4],
                     list(indices)[:4])
        raise StageError("CORPUS_ORDER_MISMATCH",
                         "the P3 per-slice binding order does not equal "
                         "the P0S canonical subset order; the ordered "
                         "corpus manifest cannot be established")
    ds = FastMRISliceDataset(data_root, split="train", mode="eval")
    dataset_prov = dataset_provenance(FastMRISliceDataset, ds)
    if len(ds) != parents["p0s"]["population_size"]:
        raise StageError("POPULATION_CHANGED",
                         f"dataset now holds {len(ds)} slices but P0S "
                         f"froze its subset against "
                         f"{parents['p0s']['population_size']}")
    if not indices:
        raise StageError("EMPTY_SUBSET", "the frozen subset selection is "
                                         "empty")
    torch.set_num_threads(1)
    loader = DataLoader(Subset(ds, indices), batch_size=batch,
                        shuffle=False, num_workers=0, collate_fn=_collate)

    n_sel = len(indices)
    corpus = np.empty((n_sel, EXPECTED_FLOW_DIM_REAL), dtype=np.float64)
    rows: list[dict] = []
    k = 0
    for b in loader:
        p = _prepare(b, "cpu", test0=False)
        missing = [key for key in REQUIRED_PREPARE_KEYS if key not in p]
        if missing:
            raise StageError("PREPARE_CONTRACT_CHANGED",
                             f"_prepare() returned no {missing}")
        x_norm, cond_in = p["x_norm"], p["cond_in"]
        if cond_in.shape != x_norm.shape or x_norm.shape[1] != 2 \
                or tuple(x_norm.shape[-2:]) != (GRID_H, GRID_W):
            raise StageError("STATE_SHAPE_UNEXPECTED",
                             f"x_norm {tuple(x_norm.shape)} / cond_in "
                             f"{tuple(cond_in.shape)} are not the "
                             f"expected (B, 2, {GRID_H}, {GRID_W}) "
                             f"states")
        masks = b["mask"]
        for j, meta in enumerate(b["meta"]):
            m = masks[j]
            if m.dtype != torch.bool or m.dim() != 1 \
                    or int(m.shape[-1]) != GRID_W:
                raise StageError("MASK_SHAPE",
                                 f"batch mask must be 1-D bool of width "
                                 f"{GRID_W}, got {tuple(m.shape)} "
                                 f"{m.dtype}")
            row = {"corpus_order": k,
                   "dataset_index": int(indices[k]),
                   "file": meta["file"],
                   "slice_index": int(meta["slice_index"]),
                   "split": meta["split"], "mode": meta["mode"],
                   "mask_seed": int(meta["mask_seed"]),
                   "live_columns": tuple(int(c) for c in
                                         torch.nonzero(m).flatten()
                                         .tolist())}
            cmap = verify_binding_identity(row, bindings[k], GRID_H,
                                           GRID_W)
            x_c = torch.complex(x_norm[j][0], x_norm[j][1])
            c_c = torch.complex(cond_in[j][0], cond_in[j][1])
            k_dx = dec.fft2c(x_c - c_c)
            u = dec.gather_unmeasured(k_dx, cmap)
            re_s, im_s = standardise_free(u, cmap, loc_index)
            corpus[k] = pack_scalar_corpus(re_s, im_s)
            row.pop("live_columns")
            row.update({
                "mask_sha256": bindings[k]["mask_sha256"],
                "map_sha256": bindings[k]["map_sha256"],
                "max_abs_u_scaled": float(max(
                    np.abs(re_s).max(), np.abs(im_s).max()))})
            rows.append(row)
            k += 1
    if k != n_sel:
        raise StageError("SUBSET_SIZE_MISMATCH",
                         f"collected {k} slices, expected {n_sel}")
    map_stats = {
        "n_bindings_verified": k,
        "n_distinct_maps": len({r["map_sha256"] for r in rows}),
        "n_distinct_masks": len({r["mask_sha256"] for r in rows}),
        "rederivation_rule": "each map RE-DERIVED from the RECORDED "
                             "acquired_columns under the published "
                             "enumeration rule; derived map_sha256 "
                             "required to equal the recorded value "
                             "(P3 consumer contract -- mandatory, not "
                             "optional)",
        "enumeration_rule": dec.P3_FLATTEN_ORDER,
        "packing_order": dec.P3_COMPLEX_PACKING_ORDER}
    return rows, corpus, map_stats, dataset_prov


# ---------------------------------------------------------------------------
# Facts + publication
# ---------------------------------------------------------------------------

def _build_facts(parents, p3, p4, rows, qb, map_stats, verdict, reason,
                 repo_dir, script, argv, t0, seed_prov,
                 dataset_prov) -> dict:
    thresholds = {
        "SPLINE_PERCENTILE": SPLINE_PERCENTILE,
        "PERCENTILE_METHOD": PERCENTILE_METHOD,
        "SPLINE_MARGIN": SPLINE_MARGIN,
        "calibration_rule": "q = np.percentile(np.abs(u_scaled_scalar), "
                            "99.9, method=\"linear\") over the complete "
                            "scalar corpus of separately standardised "
                            "real and imaginary components; B = 1.1 * q "
                            "(rule recovered from the registered I3/D3 "
                            "pilot, EXEC revision 20; the pilot VALUE is "
                            "superseded and never reused)",
        "scalar_not_magnitude": "the corpus is the interleaved scalar "
                                "real/imag components; |.| is the SCALAR "
                                "absolute value, NOT the complex-"
                                "coordinate magnitude",
        "EXPECTED_CORPUS_SLICES": EXPECTED_CORPUS_SLICES,
        "EXPECTED_OBSERVATIONS": EXPECTED_OBSERVATIONS,
        "EXPECTED_ACQUIRED_COLUMNS": EXPECTED_ACQUIRED_COLUMNS,
        "EXPECTED_N_FREE_COMPLEX": EXPECTED_N_FREE_COMPLEX,
        "EXPECTED_FLOW_DIM_REAL": EXPECTED_FLOW_DIM_REAL,
        "exact_count_rule": "no subsampling, streaming, t-digest, "
                            "histogram or GPU path; the observation count "
                            "is gated EXACTLY",
        "tail_rule": "NSF linear identity tails outside [-B, B] remain "
                     "operative; outside values are NOT errors and are "
                     "NEVER clipped",
        "p4_consumption_rule": "applied_scale finite AND strictly > 0 at "
                               "every gathered location; applied_mean "
                               "finite; branch == PER-LOCATION; means, "
                               "scales, floors and the branch are NEVER "
                               "recomputed",
        "corpus_rule": "the FROZEN P0S 256-slice subset in dataset-index "
                       "order, eval mode; NEVER the TINY corpus "
                       "(contamination)"}
    corpus_manifest = [
        {"corpus_order": r["corpus_order"],
         "dataset_index": r["dataset_index"], "file": r["file"],
         "slice_index": r["slice_index"],
         "mask_sha256": r["mask_sha256"],
         "map_sha256": r["map_sha256"]} for r in rows]
    arithmetic_path = {
        "device": "cpu", "torch_threads": 1, "dtype": "float64",
        "traversal_order": "dataset index (manifest) order",
        "preparation": "registered live train_base._prepare (batch -> "
                       "x_norm = x_true / file_attr_max, cond_in = "
                       "two-channel A^H y / file_attr_max); the division "
                       "is NEVER reimplemented here",
        "target": "delta_k = fft2c(x_norm - cond_in); free coefficients "
                  "gathered through the re-derived per-sample map",
        "standardisation": "P4 /2 APPLIED affine pair at the same "
                           "physical (r, c), real and imaginary "
                           "components separately, NumPy float64",
        "packing": "interleaved re/im per complex coordinate, P3 "
                   "canonical order",
        "statistic": "np.percentile(np.abs(corpus), 99.9, "
                     "method=\"linear\") on CPU in float64"}
    p4_consumption = {
        "branch": p4["branch"],
        "locations_order": p4["locations_order"],
        "n_locations_indexed": len(p4["location_index"]),
        "validity": "applied_scale finite AND strictly > 0 and "
                    "applied_mean finite at ALL indexed locations "
                    "(load-time superset of the registered per-gather "
                    "rule; applied_pair re-checks every gathered "
                    "location)",
        "recompute": "NONE -- means, scales, floors and the branch are "
                     "consumed, never recomputed"}
    # Path-free parent identifiers: the semantic payload must be stable
    # across machines, so absolute paths never enter it (campaign
    # precedent: P4S2 records hashes only).
    p0_rec = {k: v for k, v in parents["p0"].items() if k != "path"}
    p0s_rec = {k: v for k, v in parents["p0s"].items() if k != "path"}
    parents_rec = {
        "p0": p0_rec, "p0s": p0s_rec,
        "p3": {"schema": P3_FACTS_SCHEMA,
               "file_sha256": p3["file_sha256"],
               "semantic_sha256": p3["semantic_sha256"],
               "verdict": "PASS",
               "sidecar_verified": p3["sidecar_verified"]},
        "p4_s2": {"schema": P4S2_FACTS_SCHEMA,
                  "file_sha256": p4["file_sha256"],
                  "semantic_sha256": p4["semantic_sha256"],
                  "verdict": "PASS",
                  "sidecar_verified": p4["sidecar_verified"]},
        "s_ref": parents["s_ref"],
        "subset_size": parents["subset_size"],
        "subset_indices_sha256": canonical_hash(
            [int(i) for i in parents["subset_indices"]])}
    code = hash_project_code(repo_dir, script)
    # The frozen CODE_HASH_FILES cannot name P3-era modules -- editing that
    # list would invalidate the passed stages (P3 precedent:
    # hash_p3_local_code). IMPL-B hashes them here and both blocks are
    # recorded, so no code that can affect the calibration is unhashed.
    # fastmri_data / train_base are ALSO in the frozen list; they are
    # repeated deliberately (P4S2 precedent) because the mask-generator
    # hash-binding and the live _prepare owner are load-bearing here.
    implb_local_files = [
        "seqref_mri/src/fastmri_data.py",
        "seqref_mri/src/residual_decoder.py",
        "seqref_mri/src/preflight_parents_p3.py",
        "seqref_mri/scripts/train_base.py"]
    hashed = []
    for rel in implb_local_files:
        sha = _local_sha(repo_dir, rel)
        if sha is None:
            logger.error("[%s] IMPL-B-local code-hash file missing: %s",
                         SCRIPT_ID, os.path.join(repo_dir, rel))
            raise StageError("CODE_HASH_FILE_MISSING",
                             f"project-local file required for the IMPL-B "
                             f"code hash is missing: {rel}")
        hashed.append({"relpath": rel, "sha256": sha})
    implb_local = {"implb_local": hashed,
        "implb_local_note": "P3-era modules the frozen CODE_HASH_FILES "
                            "cannot name, plus the deliberately repeated "
                            "load-bearing pair; the frozen block already "
                            "covers preflight_io, preflight_parents, "
                            "contract_hash, normalisation_profile, "
                            "parent_expectations, forward_operator, "
                            "fastmri_data and train_base"}
    summary = {"verdict": verdict, "verdict_reason": reason,
               "n_slices": len(rows),
               "observation_count": qb["observation_count"],
               "q": qb["q"], "B": qb["B"],
               "fraction_abs_u_scaled_beyond_B":
                   qb["diagnostics"]["fraction_abs_u_scaled_beyond_B"],
               "max_abs_u_scaled":
                   qb["diagnostics"]["max_abs_u_scaled"]}
    per_slice_diag = [
        {"corpus_order": r["corpus_order"],
         "dataset_index": r["dataset_index"],
         "max_abs_u_scaled": r["max_abs_u_scaled"]} for r in rows]
    semantic = {"schema": FACTS_SCHEMA, "stage": STAGE,
                "thresholds": thresholds, "verdict": verdict,
                "calibration": qb, "corpus_manifest": corpus_manifest,
                "per_slice_diagnostics": per_slice_diag,
                "map_binding": map_stats,
                "p4_consumption": p4_consumption,
                "arithmetic_path": arithmetic_path,
                "summary": summary, "parents": parents_rec,
                "code": code["project_local"]
                + implb_local["implb_local"]}
    facts = {
        "schema": FACTS_SCHEMA,
        "script": {"id": SCRIPT_ID, "version": SCRIPT_VERSION,
                   "lifetime": "KEEP"},
        "stage": STAGE,
        "artefact_type": "stage_facts",
        "run_mode": "authoritative",
        "authoritative": True,
        "run_mode_note": "exactly one scientific run mode exists: the "
                         "authoritative frozen-256 calibration (approved "
                         "scope core + A1 + A3; A2 rehearsal skipped, so "
                         "there is no smoke mode). The determinism "
                         "sibling comes from rerunning THIS stage.",
        "stage_description": "IMPL-B spline-bound calibration: p99.9 of "
                             "|u_scaled| over the frozen 256-slice "
                             "corpus; B = 1.1 * q, frozen at PASS for "
                             "IMPL -> FORMAL",
        "schema_scope": {
            "covers": ["spline_bound_calibration"],
            "note": "seqref-implb-facts/1 is registered in EXEC §9.1 "
                    "pre-measurement; the frozen B feeds the IMPL NSF "
                    "constants"},
        "thresholds": thresholds,
        "verdict": verdict,
        "verdict_reason": reason,
        "calibration": qb,
        "corpus_manifest": corpus_manifest,
        "per_slice_diagnostics": per_slice_diag,
        "map_binding": map_stats,
        "p4_consumption": p4_consumption,
        "arithmetic_path": arithmetic_path,
        "summary": summary,
        "parents": parents_rec,
        "p3_parents_record": p3["parents_record"],
        "p4_s2_parents_record": p4["parents_record"],
        "mask_seed_provenance": seed_prov or {"resolved": False},
        "dataset_provenance": dataset_prov or {},
        "code": {**code, **implb_local},
        "run": {**environment_record(repo_dir, argv),
                "runtime_seconds": time.time() - t0},
        "hash_note": "the authoritative artefact SHA is the SHA-256 of "
                     "THIS FILE'S bytes, in the sidecar; semantic_sha256 "
                     "covers scientific content only; a determinism "
                     "sibling MUST agree on semantic_sha256 and need not "
                     "agree on the file SHA",
    }
    return attach_semantic_hash(facts, semantic)


def publish_implb(facts: dict, out_dir: str, prefix: str) -> tuple:
    """The stage's publication path, isolated so the self-test can drive
    it directly (no-overwrite, claim, pairing machinery)."""
    return publish_stage(facts, out_dir, prefix, STAGE)


def _error_parent_ids(parents, p3, p4) -> dict | None:
    """Parent identifiers for an ERROR record; None until the P0/P0S
    context exists (publish_error then decides record-worthiness via the
    StageError's write_record flag)."""
    if parents is None:
        return None
    return {"p0": parents["p0"], "p0s": parents["p0s"],
            "p3": (p3 and {"file_sha256": p3["file_sha256"],
                           "semantic_sha256": p3["semantic_sha256"]}),
            "p4_s2": (p4 and {"file_sha256": p4["file_sha256"],
                              "semantic_sha256": p4["semantic_sha256"]})}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=f"{SCRIPT_ID} {SCRIPT_VERSION} -- IMPL-B spline-bound "
                    f"calibration (schema {FACTS_SCHEMA})")
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--p0-facts", required=True)
    ap.add_argument("--p0s-facts", required=True)
    ap.add_argument("--p0s-script", required=True)
    ap.add_argument("--p3-facts", required=True,
                    help="authoritative P3 coordinate-map artefact; "
                         "verified against BOTH registered pins")
    ap.add_argument("--p4-stats2", required=True,
                    help="authoritative P4 /2 scaling-statistics "
                         "artefact; verified against BOTH registered "
                         "pins")
    ap.add_argument("--out-dir", required=True,
                    help="IMPL-B output directory")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s "
                               "%(message)s")
    t0 = time.time()
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    script = os.path.abspath(__file__)
    parents = p3 = p4 = None

    try:
        logger.info("%s %s run_mode=authoritative out_dir=%s", SCRIPT_ID,
                    SCRIPT_VERSION, args.out_dir)
        # Registered arithmetic path: CPU, single torch thread, float64.
        torch.set_num_threads(1)
        parents = verify_parents(args.repo_dir, args.p0_facts,
                                 args.p0s_facts, args.p0s_script)
        p3 = load_p3_parent(args.p3_facts)
        p4 = load_p4s2_parent(args.p4_stats2)
        logger.info("%s parents pinned: P3 file=%s semantic=%s | P4/2 "
                    "file=%s semantic=%s", SCRIPT_ID,
                    p3["file_sha256"][:12], p3["semantic_sha256"][:12],
                    p4["file_sha256"][:12], p4["semantic_sha256"][:12])
        seed_prov = bind_mask_seed_provenance(args.repo_dir)
        enforce_generator_pin(seed_prov)

        rows, corpus, map_stats, dataset_prov = _collect(
            parents, p3["bindings"], p4["location_index"],
            args.data_root, args.batch)
        expected = len(rows) * EXPECTED_FLOW_DIM_REAL
        qb = compute_q_b(corpus.ravel(), expected_count=expected)

        frac_beyond = qb["diagnostics"]["fraction_abs_u_scaled_beyond_B"]
        max_abs = qb["diagnostics"]["max_abs_u_scaled"]
        reason = (f"IMPL-B calibrated on the frozen {len(rows)}-slice "
                  f"corpus: q = p99.9(|u_scaled|) = {qb['q']!r} over "
                  f"exactly {qb['observation_count']} float64 scalars "
                  f"(np.percentile method=\"linear\"), B = 1.1 * q = "
                  f"{qb['B']!r}; fraction beyond B {frac_beyond:.3e}, "
                  f"max |u_scaled| {max_abs:.6g}")
        facts = _build_facts(parents, p3, p4, rows, qb, map_stats,
                             "PASS", reason, args.repo_dir, script,
                             raw_argv, t0, seed_prov, dataset_prov)
        path, sha = publish_implb(facts, args.out_dir, FACTS_PREFIX)
        logger.info("%s PASS n=%d q=%.12g B=%.12g facts=%s "
                    "file_sha256=%s semantic_sha256=%s", STAGE,
                    len(rows), qb["q"], qb["B"], path, sha,
                    facts["semantic_sha256"])
        return EXIT_PASS
    except StageError as exc:
        logger.error("[%s] %s: %s", exc.error_code, STAGE, exc.reason)
        publish_error(exc, args.out_dir, ERROR_PREFIX, STAGE,
                      parents=_error_parent_ids(parents, p3, p4),
                      code={"script": script}, run={"argv": raw_argv})
        return EXIT_ERROR
    except Exception as exc:
        # Failure boundary (P1/P2/P3/P4 doctrine, verbatim pattern).
        # KeyboardInterrupt/SystemExit are deliberately NOT caught (they
        # are BaseException): an ordinary exception must never surface as
        # a bare traceback -- no exit-code contract, no artefact.
        logger.exception("%s UNEXPECTED ERROR", SCRIPT_ID)
        wrapped = StageError(
            "UNEXPECTED_RUNTIME_ERROR", f"{type(exc).__name__}: {exc}",
            detail={"exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "raised_after_parent_verification":
                        parents is not None},
            write_record=parents is not None)
        publish_error(wrapped, args.out_dir, ERROR_PREFIX, STAGE,
                      parents=_error_parent_ids(parents, p3, p4),
                      code={"script": script}, run={"argv": raw_argv})
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
