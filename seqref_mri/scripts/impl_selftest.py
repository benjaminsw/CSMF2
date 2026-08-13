# SEQREF-IMPLST v0.1 -- scripts.impl_selftest
# LIFETIME: KEEP
# Purpose: Class-A implementation-validity stage (A1-A10) for SEQREF-IMPL
#   v0.1 -- the free-coordinate conditional NSF (IMPLSPEC v0.1, EXEC
#   §13). TWO modes, one production path:
#     --mode fixtures      (default): A1-A10 production checks PLUS
#                           tamper fixtures proving every gate fires, plus
#                           parent/sidecar/pin/publication/failure-
#                           boundary infrastructure fixtures. Exit 0 iff
#                           all checks pass AND coverage matches the
#                           registry. Publishes NOTHING.
#     --mode authoritative : runs A1-A10 cleanly and publishes
#                           impl/implementation_facts.json
#                           (seqref-impl-facts/1) via the registered
#                           claim/publish machinery. PASS(0)|ERROR(2)
#                           only -- no BLOCK (no data premises; LOCK 2).
# Registered bindings (frozen plan 2026-08-12):
#   * A8-A10 (and A3/A5) exercise the SAME production functions used by
#     training/evaluation (SEQREF-IMPLR/IMPLT), never test-only replicas.
#   * A6 calls the production nll_objective/train_step from SEQREF-IMPLT
#     on the pinned synthetic micro-fixture (seed 20260812, batch 4,
#     Adam lr 3e-4, 8 steps, CPU/1 thread); nll_trace[9] = nll_before +
#     post-step[1..8]; the named parameter groups are TEST-REGISTRATION
#     groups only, never optimizer groups.
#   * A4's binding criterion is the float64 P4 scaling/inverse round-trip
#     <= 1e-12; the float32 NSF fwd/inv error is AUXILIARY evidence only
#     (recorded at <= 1e-5, never gating, no A4b).
#   * Fixture bindings F-A/F-B/F-C are PINNED (corpus orders 0/127/255
#     of the IMPL-B authoritative manifest); the A8 negatives are pinned
#     (F-A slice_index 20->21; F-B map payload hash mutation) and must
#     fail binding verification BEFORE any decode/training work.
# REGISTRY DISCIPLINE: EXPECTED_COUNTS is a STATIC count of this source,
#   re-derived per rewrite, never carried forward.
# CONVENTION: logger.error + typed raise; fixture failures are reported,
#   never hidden. No fallback, no mock, no placeholder, no silent pass.
# Changelog
#   v0.1 (2026-08-12) Created against the frozen SEQREF-IMPL v0.1
#     Class-A contract (plan v0.1 + reviewer rulings: A4 float64 reading,
#     A6 9-trace + dual-group update criterion, pinned F-A/F-B/F-C, no
#     plots, core + determinism-sibling scope). Never executed.
# =============================================================================
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import tempfile

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

from preflight_io import (canonical_hash, file_sha256,  # noqa: E402
                          verify_sidecar)
from preflight_parents import (StageError, EXIT_PASS, EXIT_ERROR,  # noqa: E402
                               verify_parents, hash_project_code,
                               environment_record,
                               attach_semantic_hash, publish_stage,
                               publish_error)
from preflight_parents_p3 import bind_mask_seed_provenance  # noqa: E402
import residual_decoder as dec  # noqa: E402
from seqref_mri.src import fastmri_data as fdm  # noqa: E402
from seqref_mri.src import free_flow_runtime as ffr  # noqa: E402
from seqref_mri.scripts import train_free_flow as tff  # noqa: E402
from seqref_mri.scripts.train_base import _collate, _prepare  # noqa: E402

SCRIPT_ID = "SEQREF-IMPLST"
SCRIPT_VERSION = "v0.1"
STAGE = "IMPL"
FACTS_SCHEMA = "seqref-impl-facts/1"
FACTS_PREFIX = "implementation_facts"
ERROR_PREFIX = "impl_error"
logger = logging.getLogger(SCRIPT_ID)

_RESULTS: list[tuple[str, str, bool, str]] = []
_CURRENT = ["<none>"]

# Parent artefacts (installation paths; --repo-dir must resolve here).
P3_ART = os.path.join(_REPO, "seqref_mri", "results", "_diag", "p3",
                      "coordinate_map.json")
P4S2_ART = os.path.join(_REPO, "seqref_mri", "results", "_diag", "p4",
                        "scaling_statistics.json")
IMPLB_ART = os.path.join(_REPO, "seqref_mri", "results", "_diag", "impl",
                         "implb_facts.json")

# PINNED fixture bindings: first/middle/last of the IMPL-B authoritative
# corpus manifest (frozen 2026-08-12; verified against the artefact).
PINNED_FIXTURES = (
    {"label": "F-A", "corpus_order": 0, "dataset_index": 94,
     "file": "knee_singlecoil_train/singlecoil_train/file1000003.h5",
     "slice_index": 20,
     "mask_sha256": "6573d0eed6b936d9c78c7d06317ffb6c9c0bbadd1694d44087f3b7eed3285b55",
     "map_sha256": "9df96419e3c24e874034ff1c3a2886ba5395acf505c1ad5137ed7e2f366d138f"},
    {"label": "F-B", "corpus_order": 127, "dataset_index": 18154,
     "file": "knee_singlecoil_train/singlecoil_train/file1001388.h5",
     "slice_index": 4,
     "mask_sha256": "8a0f8de51d708f80ccbad05f3a3f0b2f726be043c57187ec1180cca93ed98b1d",
     "map_sha256": "c4c1f6bc93b4670f66b5bcb571703e91ada29a19c7a568cad014a71e7e06a401"},
    {"label": "F-C", "corpus_order": 255, "dataset_index": 34492,
     "file": "knee_singlecoil_train/singlecoil_train/file1002554.h5",
     "slice_index": 9,
     "mask_sha256": "f8a036e17b52e1d87b46f2eede72bd22b7d85f6a30c71f0aaa84ee9ef445eb65",
     "map_sha256": "64533f8344ffc3d8c8afe04ff2e69101184d5bce73a444654ae4477d766af02c"},
)

FIXTURE_SEED = 20260814           # synthetic tensors (A1/A3/A4/A10)
A6_SEED = 20260812                # pinned micro-training protocol
A6_BATCH = 4
A6_LR = 3e-4
A6_STEPS = 8
# A7 clarification, pre-authoritative:
# Because the preregistered zero-init post->FiLM chain yields exactly zero
# mask-branch gradient at initialization, gradient reachability is tested
# after exactly 2 fixed production Adam steps at lr=3e-4 on a separate
# seeded build. This is a reachability probe, not a learning claim.
A7_GRAD_PROBE_STEPS = 2
A7_GRAD_PROBE_LR = 3e-4           # pinned under the A7 namespace: the
                                  # clarified probe must not silently
                                  # inherit the unrelated A6_LR


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((_CURRENT[0], name, bool(cond), detail))


def expect_error(name: str, code: str, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except StageError as exc:
        check(name, exc.error_code == code,
              f"got {exc.error_code}, want {code}")
        return
    except Exception as exc:  # noqa: BLE001 -- reported, not hidden
        check(name, False, f"wrong exception {type(exc).__name__}: {exc}")
        return
    check(name, False, f"no error raised, want {code}")


def _a_fail(code: str, exc: Exception) -> StageError:
    """Map a runtime FreeFlowError onto the Class-A taxonomy, preserving
    the runtime code in the reason (audit trail)."""
    logger.error("[%s] %s: %s: %s", SCRIPT_ID, code,
                 type(exc).__name__, exc)
    return StageError(code, f"{type(exc).__name__}: {exc}")


def _fail_a(code: str, message: str, **kwargs) -> StageError:
    """logger.error + typed raise, at the point of detection (the
    runtime _fail pattern; no silent failure path)."""
    logger.error("[%s] %s: %s", SCRIPT_ID, code, message)
    return StageError(code, message, **kwargs)


# ---------------------------------------------------------------------------
# Stage context: parents, dataset fixtures, model. ONE builder shared by
# both modes -- fixtures and the authoritative run see identical inputs.
# ---------------------------------------------------------------------------

def _load_fixture_sample(ds, pin: dict, p3_bindings: list,
                         loc_index: dict) -> dict:
    """Load one PINNED fixture sample through the production dataset path
    and bind it (verification BEFORE any decode/training work)."""
    binding = p3_bindings[pin["corpus_order"]]
    for field in ("dataset_index", "file", "slice_index", "mask_sha256",
                  "map_sha256"):
        recorded = binding.get(field)
        if field in ("dataset_index", "slice_index"):
            recorded = int(recorded)
        if recorded != pin[field]:
            raise _fail_a(
                "A8_BINDING_MISMATCH",
                f"the P3 binding at corpus position {pin['corpus_order']} "
                f"disagrees with the PINNED fixture {pin['label']} on "
                f"{field} ({recorded!r} != {pin[field]!r}); the frozen "
                f"manifest order is not being honoured")
    item = ds[int(pin["dataset_index"])]
    batch = _collate([item])
    prep = _prepare(batch, "cpu", test0=False)
    meta = item["meta"]
    # Dataset-side identity evidence: the sample loaded BY the pinned
    # dataset_index must agree with the recorded binding identity. (The
    # binding file is corpus-rooted; the dataset meta is data_root-
    # relative -- the basename is the common identity anchor.)
    if int(meta["slice_index"]) != int(binding["slice_index"]) \
            or os.path.basename(meta["file"]) != os.path.basename(
                binding["file"]):
        raise _fail_a(
            "A8_BINDING_MISMATCH",
            f"the sample loaded at dataset_index {pin['dataset_index']} "
            f"carries meta file/slice ({meta['file']!r}, "
            f"{meta['slice_index']!r}) that disagrees with the recorded "
            f"P3 binding ({binding['file']!r}, "
            f"{binding['slice_index']!r}) -- the corpus traversal is not "
            f"the frozen one")
    # The identity row asserts "the sample I loaded IS this recorded
    # binding" (IMPLBT _row_for pattern): identity fields come from the
    # BINDING; the live columns come from the SAMPLE, and the verification
    # chain (live columns -> recorded mask hash -> recorded map hash) is
    # what proves the sample actually matches.
    row = {"corpus_order": pin["corpus_order"],
           "dataset_index": binding["dataset_index"],
           "file": binding["file"],
           "slice_index": binding["slice_index"],
           "split": binding["split"],
           "mask_seed": binding["mask_seed"],
           "live_columns": [int(c) for c in np.flatnonzero(
               item["mask"].cpu().numpy()).tolist()]}
    try:
        cmap = ffr.verify_binding_identity(row, binding)
    except ffr.FreeFlowError as exc:
        raise _a_fail("A8_BINDING_MISMATCH", exc)
    vecs = ffr.standardisation_vectors(cmap, loc_index)
    return {"pin": pin, "binding": binding, "row": row, "cmap": cmap,
            "vecs": vecs, "prep": prep, "item": item,
            "y_raw": prep["y"], "amax": prep["amax"],
            "cond_in": prep["cond_in"], "mask": batch["mask"]}


def build_stage_context(data_root: str) -> dict:
    """Parents (dual-pin + mandatory sidecars), generator pin, dataset,
    pinned fixture samples, seeded production model."""
    p3 = ffr.load_p3_parent(P3_ART)
    p4 = ffr.load_p4s2_parent(P4S2_ART)
    implb = ffr.load_implb_parent(IMPLB_ART)
    seed_prov = bind_mask_seed_provenance(_REPO)
    ffr.enforce_generator_pin(seed_prov)
    ds = fdm.FastMRISliceDataset(data_root, split="train", mode="eval")
    fixtures = [_load_fixture_sample(ds, pin, p3["bindings"],
                                     p4["location_index"])
                for pin in PINNED_FIXTURES]
    model = ffr.build_model(spline_b=implb["spline_b"])
    model.eval()
    return {"p3": p3, "p4": p4, "implb": implb, "seed_prov": seed_prov,
            "dataset": ds, "fixtures": fixtures, "model": model,
            "loc_index": p4["location_index"]}


# ---------------------------------------------------------------------------
# A1 -- finiteness (production construction + forward/inverse/log_prob)
# ---------------------------------------------------------------------------

def _a1_run(model, u, cond_in, mask) -> dict:
    try:
        h = model.condition(cond_in, mask)
        z, ldj = model.flow.encode(u, h)
        lp = model.log_prob_free(u, cond_in, mask)
        u_dec = model.decode_scalars(z, cond_in, mask)
    except ffr.FreeFlowError as exc:
        raise _a_fail("A1_NON_FINITE", exc)
    except ValueError as exc:
        # The NSF backend raises a bare ValueError("non-finite output in RQ
        # spline") BEFORE the runtime's own FreeFlowError gates can fire
        # (nsf_layer.py _rq_spline). Map that specific signal into the
        # Class-A taxonomy; re-raise any other ValueError untouched.
        if "non-finite" in str(exc).lower():
            raise _a_fail("A1_NON_FINITE", exc)
        raise
    tensors = {"z": z, "logdet": ldj, "log_prob": lp, "u_dec": u_dec}
    for name, t in tensors.items():
        if not torch.isfinite(t).all():
            raise _fail_a("A1_NON_FINITE",
                             f"{name} contains a non-finite value on a "
                             f"finite synthetic input batch")
    return {"finite": True,
            "max_abs_z": float(z.abs().max()),
            "max_abs_logdet": float(ldj.abs().max()),
            "max_abs_log_prob": float(lp.abs().max()),
            "max_abs_u_dec": float(u_dec.abs().max())}


def a1_finiteness(ctx: dict) -> dict:
    g = torch.Generator().manual_seed(FIXTURE_SEED)
    u = torch.randn(A6_BATCH, ffr.FLOW_DIM_REAL, generator=g)
    cond_in = torch.randn(A6_BATCH, 2, ffr.GRID_H, ffr.GRID_W, generator=g)
    mask = torch.stack([torch.from_numpy(
        fdm.make_cartesian_mask(ffr.GRID_W, FIXTURE_SEED + 100 + i))
        for i in range(A6_BATCH)])
    ev = _a1_run(ctx["model"], u, cond_in, mask)
    ev["batch"] = A6_BATCH
    return ev


def test_a1_finiteness() -> None:
    _CURRENT[0] = "test_a1_finiteness"
    ctx = _CTX[0]
    ev = a1_finiteness(ctx)
    check("a1 production forward/inverse/log_prob finite",
          ev["finite"] is True, json.dumps(ev))
    check("a1 evidence carries all maxima",
          all(k in ev for k in ("max_abs_z", "max_abs_logdet",
                                "max_abs_log_prob", "max_abs_u_dec")),
          str(sorted(ev.keys())))
    g = torch.Generator().manual_seed(FIXTURE_SEED + 1)
    u_bad = torch.randn(2, ffr.FLOW_DIM_REAL, generator=g)
    u_bad[0, 0] = float("nan")
    cond_in = torch.randn(2, 2, ffr.GRID_H, ffr.GRID_W, generator=g)
    mask = torch.stack([torch.from_numpy(
        fdm.make_cartesian_mask(ffr.GRID_W, FIXTURE_SEED + 101 + i))
        for i in range(2)])
    expect_error("a1 rejects a non-finite input state", "A1_NON_FINITE",
                 _a1_run, ctx["model"], u_bad, cond_in, mask)


# ---------------------------------------------------------------------------
# A2 -- dim = 13,824 counted free DoF
# ---------------------------------------------------------------------------

def _a2_gate(model_dim: int, counted: int) -> None:
    if model_dim != ffr.FLOW_DIM_REAL or counted != ffr.FLOW_DIM_REAL \
            or counted != 2 * ffr.N_FREE_COMPLEX:
        raise _fail_a(
            "A2_DIM_MISMATCH",
            f"model dim {model_dim} / counted free DoF {counted} diverge "
            f"from the registered {ffr.FLOW_DIM_REAL} = "
            f"2 * {ffr.N_FREE_COMPLEX}")


def a2_dimensions(ctx: dict) -> dict:
    per = []
    for fx in ctx["fixtures"]:
        counted = 2 * fx["cmap"].n_free_complex
        _a2_gate(ctx["model"].flow.dim, counted)
        if int(fx["binding"]["flow_dim_real"]) != ffr.FLOW_DIM_REAL \
                or int(fx["binding"]["n_free_complex"]) \
                != ffr.N_FREE_COMPLEX:
            raise _fail_a("A2_DIM_MISMATCH",
                             f"{fx['pin']['label']}: the recorded P3 "
                             f"binding dims diverge from the registered "
                             f"constants")
        per.append({"label": fx["pin"]["label"],
                    "counted_flow_dim_real": counted})
    return {"flow_dim_real": ffr.FLOW_DIM_REAL,
            "n_free_complex": ffr.N_FREE_COMPLEX, "per_fixture": per}


def test_a2_dimensions() -> None:
    _CURRENT[0] = "test_a2_dimensions"
    ev = a2_dimensions(_CTX[0])
    check("a2 model dim equals counted free DoF on pinned fixtures",
          ev["flow_dim_real"] == 13824 and
          all(p["counted_flow_dim_real"] == 13824
              for p in ev["per_fixture"]), json.dumps(ev))
    check("a2 covers all three pinned fixtures",
          len(ev["per_fixture"]) == 3, str(len(ev["per_fixture"])))
    check("a2 registered constants consistent",
          ffr.FLOW_DIM_REAL == 2 * ffr.N_FREE_COMPLEX == 13824,
          f"{ffr.N_FREE_COMPLEX}/{ffr.FLOW_DIM_REAL}")
    expect_error("a2 rejects a dim drift", "A2_DIM_MISMATCH",
                 _a2_gate, 13823, 13824)


# ---------------------------------------------------------------------------
# A3 -- acquired fixity <= 1e-5 on the production decode path (real
# measured-k from PINNED F-A; fixed synthetic z)
# ---------------------------------------------------------------------------

def _a3_gate(abs_err: float, rel_err: float) -> None:
    if not (np.isfinite(abs_err) and abs_err <= ffr.A3_ACQUIRED_FIXITY_MAX):
        raise _fail_a(
            "A3_ACQUIRED_FIXITY",
            f"the production decode path moved ACQUIRED k-space: abs_err "
            f"{abs_err!r} exceeds the registered "
            f"{ffr.A3_ACQUIRED_FIXITY_MAX} (rel {rel_err!r})")


def a3_acquired_fixity(ctx: dict) -> dict:
    fx = ctx["fixtures"][0]                      # PINNED F-A
    g = torch.Generator().manual_seed(FIXTURE_SEED + 200)
    z = torch.randn(1, ffr.FLOW_DIM_REAL, generator=g)
    x_cand = ffr.decode_to_image(ctx["model"], z, fx["cond_in"],
                                 fx["mask"], fx["y_raw"], fx["amax"],
                                 fx["cmap"], fx["vecs"])
    abs_err, rel_err = dec.measured_fixity(x_cand, fx["y_raw"],
                                           fx["amax"], fx["cmap"])
    _a3_gate(abs_err, rel_err)
    return {"fixture": fx["pin"]["label"], "abs_err": abs_err,
            "rel_err": rel_err, "tolerance": ffr.A3_ACQUIRED_FIXITY_MAX,
            "measured_source": "real eval-mode measured k-space from the "
                               "pinned F-A dataset sample; z fixed "
                               "synthetic (seeded)"}


def test_a3_acquired_fixity() -> None:
    _CURRENT[0] = "test_a3_acquired_fixity"
    ev = a3_acquired_fixity(_CTX[0])
    check("a3 acquired fixity within tolerance on real measured-k",
          ev["abs_err"] <= ffr.A3_ACQUIRED_FIXITY_MAX, json.dumps(ev))
    check("a3 evidence records abs and rel error",
          "abs_err" in ev and "rel_err" in ev, str(sorted(ev.keys())))
    expect_error("a3 rejects a moved acquired coefficient",
                 "A3_ACQUIRED_FIXITY", _a3_gate, 1e-3, 1e-3)


# ---------------------------------------------------------------------------
# A4 -- P4 scaling/inverse-scaling round-trip <= 1e-12 (float64, BINDING);
# float32 NSF fwd/inv recorded as AUXILIARY evidence (<= 1e-5, non-gating)
# ---------------------------------------------------------------------------

def _a4_gate(max_err: float) -> None:
    if not (np.isfinite(max_err)
            and max_err <= ffr.A4_SCALING_ROUNDTRIP_MAX):
        raise _fail_a(
            "A4_SCALING_ROUNDTRIP",
            f"the float64 scaling/inverse-scaling round-trip error "
            f"{max_err!r} exceeds the registered "
            f"{ffr.A4_SCALING_ROUNDTRIP_MAX}")


def a4_scaling_roundtrip(ctx: dict) -> dict:
    fx = ctx["fixtures"][0]
    rng = np.random.default_rng(FIXTURE_SEED + 300)
    u = (rng.standard_normal(ffr.N_FREE_COMPLEX)
         + 1j * rng.standard_normal(ffr.N_FREE_COMPLEX)).astype(
             np.complex128)
    re_s, im_s = ffr.standardise_free(u, fx["cmap"], fx["vecs"])
    u2 = ffr.unstandardise_free(re_s, im_s, fx["cmap"], fx["vecs"])
    max_err = float(np.max(np.abs(u2 - u)))
    _a4_gate(max_err)
    # Auxiliary (non-gating): float32 NSF forward after inverse.
    g = torch.Generator().manual_seed(FIXTURE_SEED + 301)
    z0 = torch.randn(2, ffr.FLOW_DIM_REAL, generator=g)
    cond_in = torch.randn(2, 2, ffr.GRID_H, ffr.GRID_W, generator=g)
    mask = torch.stack([torch.from_numpy(
        fdm.make_cartesian_mask(ffr.GRID_W, FIXTURE_SEED + 302 + i))
        for i in range(2)])
    model = ctx["model"]
    u_m = model.decode_scalars(z0, cond_in, mask)
    h = model.condition(cond_in, mask)
    z1, _ = model.flow.encode(u_m, h)
    aux_err = float((z1 - z0).abs().max())
    return {"max_roundtrip_err": max_err,
            "tolerance": ffr.A4_SCALING_ROUNDTRIP_MAX,
            "auxiliary": {"nsf_fwd_inv_max_err": aux_err,
                          "aux_tolerance": ffr.A4_AUX_NSF_ROUNDTRIP_MAX,
                          "within_aux_tolerance":
                              bool(aux_err <= ffr.A4_AUX_NSF_ROUNDTRIP_MAX),
                          "gating": False,
                          "note": "float32 NSF fwd/inv consistency is "
                                  "AUXILIARY evidence only; it never "
                                  "gates A4 (no A4b)"}}


def test_a4_scaling_roundtrip() -> None:
    _CURRENT[0] = "test_a4_scaling_roundtrip"
    ev = a4_scaling_roundtrip(_CTX[0])
    check("a4 float64 scaling round-trip within 1e-12",
          ev["max_roundtrip_err"] <= ffr.A4_SCALING_ROUNDTRIP_MAX,
          json.dumps(ev))
    check("a4 auxiliary NSF evidence recorded and non-gating",
          ev["auxiliary"]["gating"] is False and
          "nsf_fwd_inv_max_err" in ev["auxiliary"],
          str(ev["auxiliary"]))
    expect_error("a4 rejects a scaling round-trip drift",
                 "A4_SCALING_ROUNDTRIP", _a4_gate, 1e-9)


# ---------------------------------------------------------------------------
# A5 -- coordinate handling vs the registered P3 maps
# ---------------------------------------------------------------------------

def _a5_verify(row: dict, binding: dict):
    try:
        return ffr.verify_binding_identity(row, binding)
    except ffr.FreeFlowError as exc:
        raise _a_fail("A5_MAP_MISMATCH", exc)


def a5_coordinate_maps(ctx: dict) -> dict:
    per = []
    for i, fx in enumerate(ctx["fixtures"]):
        payload_hash = fx["cmap"].payload()["map_payload_sha256"]
        if payload_hash != fx["binding"]["map_sha256"]:
            raise _fail_a(
                "A5_MAP_MISMATCH",
                f"{fx['pin']['label']}: the production re-derived "
                f"coordinate map payload hash diverges from the recorded "
                f"P3 binding")
        rng = np.random.default_rng(FIXTURE_SEED + 500 + i)
        k = torch.from_numpy(
            rng.standard_normal((ffr.GRID_H, ffr.GRID_W))
            + 1j * rng.standard_normal((ffr.GRID_H, ffr.GRID_W)))
        u = dec.gather_unmeasured(k, fx["cmap"])
        k2 = dec.scatter_unmeasured(u, fx["cmap"])
        if not torch.equal(dec.gather_unmeasured(k2, fx["cmap"]), u):
            raise _fail_a(
                "A5_MAP_MISMATCH",
                f"{fx['pin']['label']}: the scatter/gather round-trip on "
                f"the re-derived production map is not exact")
        acq_idx = torch.as_tensor(
            [int(c) for c in fx["cmap"].mask_columns], dtype=torch.long)
        if float(k2[:, acq_idx].abs().max()) != 0.0:
            raise _fail_a(
                "A5_MAP_MISMATCH",
                f"{fx['pin']['label']}: scatter_unmeasured wrote into an "
                f"ACQUIRED column")
        per.append({"label": fx["pin"]["label"],
                    "map_payload_hash_equal": True,
                    "scatter_gather_roundtrip_exact": True,
                    "acquired_columns_untouched": True})
    return {"per_fixture": per}


def test_a5_coordinate_maps() -> None:
    _CURRENT[0] = "test_a5_coordinate_maps"
    ctx = _CTX[0]
    ev = a5_coordinate_maps(ctx)
    for per in ev["per_fixture"]:
        check(f"a5 map payload hash matches recorded binding "
              f"({per['label']})", per["map_payload_hash_equal"] is True,
              per["label"])
        check(f"a5 scatter/gather round-trip exact ({per['label']})",
              per["scatter_gather_roundtrip_exact"] is True, per["label"])
        check(f"a5 acquired columns untouched ({per['label']})",
              per["acquired_columns_untouched"] is True, per["label"])
    # Tamper: swap one acquired column for one free column in the LIVE
    # columns -- the recorded P3 mask hash must reject it.
    fx = ctx["fixtures"][0]
    row = dict(fx["row"])
    live = list(row["live_columns"])
    free = sorted(set(range(ffr.GRID_W)) - set(live))
    live[0] = free[0]
    row["live_columns"] = sorted(live)
    expect_error("a5 rejects a live-column set that disagrees with the "
                 "recorded P3 mask", "A5_MAP_MISMATCH",
                 _a5_verify, row, fx["binding"])


# ---------------------------------------------------------------------------
# A6 -- micro-training: NLL improves and parameters update
# Pinned protocol: seed 20260812, batch 4, Adam lr 3e-4, 8 steps, CPU,
# threads=1; nll_trace[9] = nll_before + post-step[1..8]; >= 1 NSF-
# transform parameter AND >= 1 conditioning-path parameter must update.
# The named groups are TEST-REGISTRATION groups ONLY (never optimizer
# groups): the optimizer always receives model.parameters().
# ---------------------------------------------------------------------------

def _a6_protocol(seed_offset: int = 0):
    g = torch.Generator().manual_seed(A6_SEED + seed_offset)
    cond_in = torch.randn(A6_BATCH, 2, ffr.GRID_H, ffr.GRID_W,
                          generator=g)
    masks = torch.stack([torch.from_numpy(
        fdm.make_cartesian_mask(ffr.GRID_W, A6_SEED + 1 + i))
        for i in range(A6_BATCH)])
    targets = torch.randn(A6_BATCH, ffr.FLOW_DIM_REAL, generator=g)
    return cond_in, masks, targets


def _a6_gate(trace: list, nsf_max: float, cond_max: float) -> None:
    """The registered A6 gate: improvement AND dual-group update."""
    if trace[-1] >= trace[0]:
        raise _fail_a(
            "A6_NLL_NOT_IMPROVED",
            f"nll_trace[{len(trace) - 1}] = {trace[-1]!r} did not "
            f"improve over nll_before = {trace[0]!r}")
    if nsf_max <= 0.0 or cond_max <= 0.0:
        raise _fail_a(
            "A6_PARAM_STATIC",
            f"required update groups static: NSF-transform max update "
            f"{nsf_max!r}, conditioning-path max update {cond_max!r}")


def _a6_nll(model, targets, cond_in, masks) -> float:
    """Production NLL through the Class-A taxonomy: a non-finite NLL
    arrives either as a runtime FreeFlowError (LOG_PROB_NON_FINITE /
    NLL_NON_FINITE) or as the NSF backend's bare ValueError("non-finite
    output in RQ spline"), which fires before the runtime gates; both are
    mapped to A6_NON_FINITE. Any other ValueError propagates untouched."""
    try:
        return float(tff.nll_objective(model, targets, cond_in, masks))
    except ffr.FreeFlowError as exc:
        raise _a_fail("A6_NON_FINITE", exc)
    except ValueError as exc:
        if "non-finite" in str(exc).lower():
            raise _a_fail("A6_NON_FINITE", exc)
        raise


def _a6_run(lr: float, inject_nan: bool = False) -> dict:
    """Production micro-training probe on a SEPARATE seeded model build;
    raises StageError A6_* on failure. Returns trace + update evidence."""
    torch.set_num_threads(1)
    model = None
    optimizer = None
    snap = None
    try:
        model = ffr.build_model()       # same pinned init seed -> same init
        model.train()
        cond_in, masks, targets = _a6_protocol()
        if inject_nan:
            targets = targets.clone()
            targets[0, 0] = float("nan")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     betas=(0.9, 0.999), eps=1e-8)
        snap = {name: p.detach().clone()
                for name, p in model.named_parameters()}
        trace = [_a6_nll(model, targets, cond_in, masks)]
        if not np.isfinite(trace[0]):
            raise _fail_a("A6_NON_FINITE",
                          f"pre-step NLL is non-finite ({trace[0]!r})")
        for step in range(A6_STEPS):
            tff.train_step(model, optimizer, targets, cond_in, masks)
            nll = _a6_nll(model, targets, cond_in, masks)
            if not np.isfinite(nll):
                raise _fail_a("A6_NON_FINITE",
                              f"NLL after step {step + 1} is "
                              f"non-finite ({nll!r})")
            trace.append(nll)
        final = dict(model.named_parameters())
        def _max_update(names: list) -> tuple:
            best, best_name = 0.0, ""
            for name in names:
                delta = float((final[name].detach()
                               - snap[name]).abs().max())
                if delta > best:
                    best, best_name = delta, name
            return best, best_name
        nsf_max, nsf_name = _max_update(model.nsf_transform_parameters())
        cond_max, cond_name = _max_update(model.conditioning_parameters())
        mask_max, mask_name = _max_update(model.mask_branch_parameters())
        _a6_gate(trace, nsf_max, cond_max)
        return {"nll_trace": trace, "nsf_max_update": nsf_max,
                "nsf_updated_param": nsf_name,
                "conditioning_max_update": cond_max,
                "conditioning_updated_param": cond_name,
                "mask_branch_max_update": mask_max,
                "mask_branch_updated_param": mask_name}
    finally:
        del optimizer, model, snap
        gc.collect()


def a6_micro_training(ctx: dict) -> dict:
    ev = _a6_run(A6_LR)
    ev["protocol"] = {"seed": A6_SEED, "batch": A6_BATCH, "lr": A6_LR,
                      "steps": A6_STEPS, "optimizer": "Adam(0.9, 0.999, "
                      "eps=1e-8)", "device": "cpu", "threads": 1,
                      "nll_trace_length": A6_STEPS + 1,
                      "targets": "synthetic Normal(0, 1) -- proves "
                                 "plumbing, not learning",
                      "param_groups": "TEST-REGISTRATION groups only; "
                                      "optimizer always receives "
                                      "model.parameters()"}
    return ev


def test_a6_micro_training() -> None:
    _CURRENT[0] = "test_a6_micro_training"
    ev = a6_micro_training(_CTX[0])
    check("a6 nll_trace has 9 entries (before + 8 post-step)",
          len(ev["nll_trace"]) == 9, str(len(ev["nll_trace"])))
    check("a6 nll_trace finite throughout",
          all(np.isfinite(v) for v in ev["nll_trace"]),
          json.dumps(ev["nll_trace"]))
    check("a6 nll improved (trace[8] < trace[0])",
          ev["nll_trace"][8] < ev["nll_trace"][0],
          f"{ev['nll_trace'][8]} < {ev['nll_trace'][0]}")
    check("a6 both registered parameter groups updated",
          ev["nsf_max_update"] > 0.0 and
          ev["conditioning_max_update"] > 0.0,
          f"nsf={ev['nsf_max_update']} ({ev['nsf_updated_param']}); "
          f"cond={ev['conditioning_max_update']} "
          f"({ev['conditioning_updated_param']})")
    expect_error("a6 rejects a zero-lr control (no improvement)",
                 "A6_NLL_NOT_IMPROVED", _a6_run, 0.0)
    expect_error("a6 rejects a non-finite training trajectory",
                 "A6_NON_FINITE", _a6_run, A6_LR, True)
    expect_error("a6 rejects static registered parameter groups",
                 "A6_PARAM_STATIC", _a6_gate, [9.0, 8.0], 0.0, 0.5)


# ---------------------------------------------------------------------------
# A7 -- mask-path reachability (mask embedding changes h; mask branch
# receives gradient)
# ---------------------------------------------------------------------------

def _a7_effect(model) -> tuple[float, float]:
    g = torch.Generator().manual_seed(FIXTURE_SEED + 400)
    # Isolation (frozen contract): the pair differs ONLY in the mask, so
    # the measured conditioning difference is attributable to the mask
    # path alone; with a dead mask branch h[0] == h[1] and rel -> 0.
    cond_one = torch.randn(1, 2, ffr.GRID_H, ffr.GRID_W, generator=g)
    cond_in = cond_one.repeat(2, 1, 1, 1)
    m_a = torch.from_numpy(
        fdm.make_cartesian_mask(ffr.GRID_W, FIXTURE_SEED + 401))
    m_b = torch.from_numpy(
        fdm.make_cartesian_mask(ffr.GRID_W, FIXTURE_SEED + 402))
    mask = torch.stack([m_a, m_b])
    h = model.condition(cond_in, mask)
    rel = float(((h[0] - h[1]).norm() / h[0].norm()).detach())
    if not (np.isfinite(rel) and rel >= ffr.MASK_EFFECT_REL_MIN):
        raise _fail_a(
            "A7_MASK_PATH_DEAD",
            f"the mask path is unreachable: relative conditioning effect "
            f"{rel!r} < {ffr.MASK_EFFECT_REL_MIN}")
    # Gradient-reachability probe. The zero-init chain (NSF post layers
    # AND the FiLM affine output head) makes the transform an EXACT
    # identity w.r.t. h at init: mask-branch gradient is exactly zero at
    # init and after ONE step BY CONSTRUCTION, and becomes nonzero only
    # after the chain has been stepped open (post, then the FiLM head).
    # The probe therefore runs A7_GRAD_PROBE_STEPS production train
    # steps on a SEPARATE seeded build (the shared ctx model is never
    # mutated), then reads the mask-branch gradient on a fresh backward.
    u = torch.randn(2, ffr.FLOW_DIM_REAL, generator=g)
    probe = None
    optimizer = None
    try:
        probe = ffr.build_model()     # same pinned init seed -> same init
        probe.train()
        optimizer = torch.optim.Adam(probe.parameters(),
                                     lr=A7_GRAD_PROBE_LR,
                                     betas=(0.9, 0.999), eps=1e-8)
        for _ in range(A7_GRAD_PROBE_STEPS):
            tff.train_step(probe, optimizer, u, cond_in, mask)
        probe.zero_grad(set_to_none=True)
        nll = tff.nll_objective(probe, u, cond_in, mask)
        nll.backward()
        grad = probe.mask_branch.proj.weight.grad
        gnorm = float(grad.norm()) if grad is not None else 0.0
    finally:
        del optimizer, probe
        gc.collect()
    if not (np.isfinite(gnorm) and gnorm > 0.0):
        raise _fail_a("A7_MASK_PATH_DEAD",
                      f"the mask branch received no gradient after "
                      f"{A7_GRAD_PROBE_STEPS} production train steps "
                      f"(norm {gnorm!r})")
    return rel, gnorm


def a7_mask_reachability(ctx: dict) -> dict:
    rel, gnorm = _a7_effect(ctx["model"])
    return {"relative_conditioning_effect": rel,
            "floor": ffr.MASK_EFFECT_REL_MIN, "mask_grad_norm": gnorm,
            "grad_probe": f"{A7_GRAD_PROBE_STEPS} production train_steps "
                          f"at lr={A7_GRAD_PROBE_LR} on a separate seeded "
                          "build (the zero-init post->FiLM chain makes "
                          "the init gradient exactly zero by "
                          "construction; pre-authoritative A7 "
                          "clarification, reachability not learning)",
            "masks_differ": True}


def test_a7_mask_reachability() -> None:
    _CURRENT[0] = "test_a7_mask_reachability"
    ctx = _CTX[0]
    ev = a7_mask_reachability(ctx)
    check("a7 mask changes the conditioning vector", 
          ev["relative_conditioning_effect"] >= ffr.MASK_EFFECT_REL_MIN,
          json.dumps(ev))
    check("a7 effect floor is the registered 1e-5",
          ev["floor"] == ffr.MASK_EFFECT_REL_MIN, str(ev["floor"]))
    check("a7 mask branch receives gradient",
          ev["mask_grad_norm"] > 0.0, str(ev["mask_grad_norm"]))
    weight = ctx["model"].mask_branch.proj.weight
    saved = weight.detach().clone()
    try:
        with torch.no_grad():
            weight.zero_()
        expect_error("a7 rejects a zeroed mask branch",
                     "A7_MASK_PATH_DEAD", _a7_effect, ctx["model"])
    finally:
        with torch.no_grad():
            weight.copy_(saved)


# ---------------------------------------------------------------------------
# A8 -- sample <-> mask <-> P3-map binding (positives + PINNED negatives;
# failures must occur BEFORE any decode/training work)
# ---------------------------------------------------------------------------

def _a8_verify(row: dict, binding: dict):
    try:
        return ffr.verify_binding_identity(row, binding)
    except ffr.FreeFlowError as exc:
        raise _a_fail("A8_BINDING_MISMATCH", exc)


def a8_binding(ctx: dict) -> dict:
    per = []
    for fx in ctx["fixtures"]:
        cmap = _a8_verify(fx["row"], fx["binding"])
        per.append({"label": fx["pin"]["label"],
                    "dataset_index": fx["pin"]["dataset_index"],
                    "verified": True,
                    "verified_before_decode": True,
                    "n_free_complex": cmap.n_free_complex})
    return {"per_fixture": per}


def test_a8_binding() -> None:
    _CURRENT[0] = "test_a8_binding"
    ctx = _CTX[0]
    ev = a8_binding(ctx)
    for per in ev["per_fixture"]:
        check(f"a8 binding verifies for {per['label']}",
              per["verified"] is True, json.dumps(per))
        check(f"a8 binding order pin for {per['label']}",
              per["dataset_index"] ==
              dict((p["label"], p["dataset_index"])
                   for p in PINNED_FIXTURES)[per["label"]],
              str(per["dataset_index"]))
    # PINNED negative (i): F-A slice_index 20 -> 21 attacks the
    # canonical-seed/binding chain.
    row = dict(ctx["fixtures"][0]["row"])
    row["slice_index"] = 21
    expect_error("a8 PINNED negative: wrong slice index fails binding "
                 "verification BEFORE decode/training",
                 "A8_BINDING_MISMATCH", _a8_verify, row,
                 ctx["fixtures"][0]["binding"])
    # PINNED negative (ii): F-B map payload hash mutation proves the
    # payload-hash check is alive.
    binding = dict(ctx["fixtures"][1]["binding"])
    binding["map_sha256"] = ("0" * 63 +
                             ("1" if binding["map_sha256"][-1] != "1"
                              else "2"))
    expect_error("a8 PINNED negative: mutated map payload hash fails "
                 "binding verification", "A8_BINDING_MISMATCH",
                 _a8_verify, ctx["fixtures"][1]["row"], binding)


# ---------------------------------------------------------------------------
# A9 -- physical-grid P4 stats gather (location-keyed, order-independent)
# ---------------------------------------------------------------------------

def a9_p4_gather(ctx: dict) -> dict:
    fx = ctx["fixtures"][0]
    vecs_normal = ffr.standardisation_vectors(
        fx["cmap"], ctx["loc_index"])
    shuffled = {k: ctx["loc_index"][k]
                for k in sorted(ctx["loc_index"].keys(), reverse=True)}
    vecs_shuffled = ffr.standardisation_vectors(fx["cmap"], shuffled)
    for name in ("mean_re", "scale_re", "mean_im", "scale_im"):
        a = vecs_normal[name]
        b = vecs_shuffled[name]
        if a.tobytes() != b.tobytes():
            raise _fail_a(
                "A9_P4_GATHER",
                f"the P4 statistics gather is not keyed by physical "
                f"location: {name} differs under table reordering")
    return {"keyed_equal_under_reordering": True,
            "n_locations": len(ctx["loc_index"]),
            "fixture": fx["pin"]["label"]}


def test_a9_p4_gather() -> None:
    _CURRENT[0] = "test_a9_p4_gather"
    ctx = _CTX[0]
    ev = a9_p4_gather(ctx)
    check("a9 gather is order-independent (physical-keyed)",
          ev["keyed_equal_under_reordering"] is True, json.dumps(ev))
    # Tamper: remove one required physical location -> runtime gate.
    loc_index = dict(ctx["loc_index"])
    r0, c0 = (ctx["fixtures"][0]["cmap"].free_rows[0],
              ctx["fixtures"][0]["cmap"].free_cols[0])
    del loc_index[(r0, c0)]
    expect_error("a9 rejects a missing physical location",
                 "A9_P4_GATHER",
                 lambda: _a9_wrap(ctx["fixtures"][0]["cmap"], loc_index))
    # Tamper: corrupt one applied pair (scale = 0) -> runtime gate.
    loc_bad = dict(ctx["loc_index"])
    rec = dict(loc_bad[(r0, c0)])
    rec["applied_scale_re"] = 0.0
    loc_bad[(r0, c0)] = rec
    expect_error("a9 rejects a zero scale in an applied pair",
                 "A9_P4_GATHER",
                 lambda: _a9_wrap(ctx["fixtures"][0]["cmap"], loc_bad))


def _a9_wrap(cmap, loc_index):
    try:
        ffr.standardisation_vectors(cmap, loc_index)
    except ffr.FreeFlowError as exc:
        raise _a_fail("A9_P4_GATHER", exc)


# ---------------------------------------------------------------------------
# A10 -- interleaved re/im packing (registered P3_COMPLEX_PACKING_ORDER)
# ---------------------------------------------------------------------------

def _a10_unpack(vec: np.ndarray):
    try:
        return ffr.unpack_scalars(vec)
    except ffr.FreeFlowError as exc:
        raise _a_fail("A10_PACKING_ORDER", exc)


def a10_packing(ctx: dict) -> dict:
    rng = np.random.default_rng(FIXTURE_SEED + 600)
    re = rng.standard_normal(ffr.N_FREE_COMPLEX).astype(np.float64)
    im = rng.standard_normal(ffr.N_FREE_COMPLEX).astype(np.float64)
    vec = ffr.pack_scalars(re, im)
    if vec.shape != (ffr.FLOW_DIM_REAL,) or vec.dtype != np.float64:
        raise _fail_a("A10_PACKING_ORDER",
                         f"packed vector shape/dtype {vec.shape}/"
                         f"{vec.dtype} unexpected")
    if not (vec[0::2].tobytes() == re.tobytes()
            and vec[1::2].tobytes() == im.tobytes()):
        raise _fail_a("A10_PACKING_ORDER",
                         "the packed layout is not interleaved "
                         "re/im per complex coordinate")
    re2, im2 = _a10_unpack(vec)
    if not (re2.tobytes() == re.tobytes() and im2.tobytes() == im.tobytes()):
        raise _fail_a("A10_PACKING_ORDER",
                         "the unpack round-trip is not bitwise-exact")
    if dec.P3_COMPLEX_PACKING_ORDER != \
            "interleaved_real_imag_per_complex_coordinate":
        raise _fail_a("A10_PACKING_ORDER",
                         f"registered packing order literal drifted: "
                         f"{dec.P3_COMPLEX_PACKING_ORDER!r}")
    return {"order": dec.P3_COMPLEX_PACKING_ORDER,
            "n_scalars": int(vec.shape[0]),
            "interleave_bitwise": True, "roundtrip_bitwise": True}


def test_a10_packing() -> None:
    _CURRENT[0] = "test_a10_packing"
    ev = a10_packing(_CTX[0])
    check("a10 interleave is bitwise (re at even, im at odd)",
          ev["interleave_bitwise"] is True, json.dumps(ev))
    check("a10 unpack round-trip bitwise", ev["roundtrip_bitwise"] is True)
    check("a10 registered order literal intact",
          ev["order"] ==
          "interleaved_real_imag_per_complex_coordinate", ev["order"])
    check("a10 packed length is the registered 13,824",
          ev["n_scalars"] == 13824, str(ev["n_scalars"]))
    rng = np.random.default_rng(FIXTURE_SEED + 601)
    blocked = np.concatenate([rng.standard_normal(4),
                              rng.standard_normal(4)]).astype(np.float64)
    re_b, im_b = _a10_unpack(blocked)
    check("a10 de-interleave of a blocked layout provably differs",
          not (re_b.tobytes() == blocked[:4].tobytes()
               and im_b.tobytes() == blocked[4:].tobytes()),
          "blocked-order input does NOT survive the interleaved unpack "
          "as identity -- layout confusion is detectable")
    expect_error("a10 rejects an odd-length scalar vector",
                 "A10_PACKING_ORDER", _a10_unpack, np.zeros(7))


# ---------------------------------------------------------------------------
# Infrastructure fixtures (construction fixity, parents, pins, boundary,
# taxonomy, publication)
# ---------------------------------------------------------------------------

def expect_ff_error(name: str, code: str, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except ffr.FreeFlowError as exc:
        check(name, exc.code == code, f"got {exc.code}, want {code}")
        return
    except Exception as exc:  # noqa: BLE001 -- reported, not hidden
        check(name, False, f"wrong exception {type(exc).__name__}: {exc}")
        return
    check(name, False, f"no error raised, want {code}")


def expect_runtime_error(name: str, substr: str, fn, *args,
                         **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except RuntimeError as exc:
        check(name, substr in str(exc),
              f"got RuntimeError({exc}), want substring {substr!r}")
        return
    except Exception as exc:  # noqa: BLE001 -- reported, not hidden
        check(name, False, f"wrong exception {type(exc).__name__}: {exc}")
        return
    check(name, False, f"no error raised, want RuntimeError({substr!r})")


def test_construction_fixity() -> None:
    _CURRENT[0] = "test_construction_fixity"
    ctx = _CTX[0]
    model = ctx["model"]
    check("construction: flow dim is the registered 13,824",
          model.flow.dim == ffr.FLOW_DIM_REAL, str(model.flow.dim))
    check("construction: exactly 6 coupling layers",
          len(model.flow.layers) == ffr.NSF_N_LAYERS,
          str(len(model.flow.layers)))
    check("construction: every layer is NSF with K=8 and the calibrated B",
          all(getattr(l, "K", None) == ffr.NSF_K and
              getattr(l, "B", None) == ffr.SPLINE_B
              for l in model.flow.layers),
          str([(getattr(l, "K", None), getattr(l, "B", None))
               for l in model.flow.layers]))
    def _film_fixed(layer) -> bool:
        mlp = getattr(getattr(layer, "film", None), "mlp", None)
        if mlp is None or len(mlp) != 3:
            return False
        return (isinstance(mlp[0], torch.nn.Linear)
                and mlp[0].in_features == ffr.H_DIM
                and mlp[0].out_features == ffr.FILM_HIDDEN
                and isinstance(mlp[1], torch.nn.ReLU)
                and isinstance(mlp[2], torch.nn.Linear)
                and mlp[2].in_features == ffr.FILM_HIDDEN
                and mlp[2].out_features == 2 * ffr.NSF_HIDDEN)
    check("construction: FiLM head fixed at h_dim->64 ReLU 64->2*hidden "
          "on every layer", all(_film_fixed(l) for l in model.flow.layers))
    cond = model.flow.cond
    check("construction: conditioner identity (2ch/64/128/v1/no residual)",
          cond.in_channels == 2 and cond.width == ffr.COND_WIDTH
          and cond.h_dim == ffr.H_DIM and cond.use_v2 is False
          and getattr(cond, "y_residual_enabled", True) is False,
          str((cond.in_channels, cond.width, cond.h_dim, cond.use_v2)))
    proj = model.mask_branch.proj
    check("construction: mask branch 96->128, zero bias, finite weights",
          tuple(proj.weight.shape) == (ffr.MASK_EMBED_DIM, ffr.MASK_BITS)
          and bool((proj.bias == 0).all())
          and bool(torch.isfinite(proj.weight).all()),
          str(tuple(proj.weight.shape)))
    expect_ff_error("construction: a drifted B is rejected at build time",
                    "SPLINE_B_MISMATCH", ffr.FreeFlowModel, spline_b=3.0)


def test_parent_loaders() -> None:
    _CURRENT[0] = "test_parent_loaders"
    ctx = _CTX[0]
    check("parents: P3 dual pins match the registered constants",
          ctx["p3"]["file_sha256"] == ffr.P3_FILE_SHA256
          and ctx["p3"]["semantic_sha256"] == ffr.P3_SEMANTIC_SHA256
          and ctx["p3"]["sidecar_verified"] is True)
    check("parents: P4 /2 dual pins match the registered constants",
          ctx["p4"]["file_sha256"] == ffr.P4S2_FILE_SHA256
          and ctx["p4"]["semantic_sha256"] == ffr.P4S2_SEMANTIC_SHA256
          and ctx["p4"]["sidecar_verified"] is True
          and ctx["p4"]["branch"] == "PER-LOCATION")
    check("parents: IMPL-B dual pins match the registered constants",
          ctx["implb"]["file_sha256"] == ffr.IMPLB_FILE_SHA256
          and ctx["implb"]["semantic_sha256"] == ffr.IMPLB_SEMANTIC_SHA256
          and ctx["implb"]["sidecar_verified"] is True)
    check("parents: consumed SPLINE_B is the frozen literal",
          ctx["implb"]["spline_b"] == ffr.SPLINE_B,
          repr(ctx["implb"]["spline_b"]))
    expect_ff_error("parents: P3 file-pin drift is rejected",
                    "PARENT_FILE_HASH_MISMATCH", ffr.load_p3_parent,
                    P3_ART, expected_file_sha="0" * 64)
    expect_ff_error("parents: P3 semantic-pin drift is rejected",
                    "PARENT_SEMANTIC_HASH_MISMATCH", ffr.load_p3_parent,
                    P3_ART, expected_semantic_sha="0" * 64)
    expect_ff_error("parents: P4 file-pin drift is rejected",
                    "PARENT_FILE_HASH_MISMATCH", ffr.load_p4s2_parent,
                    P4S2_ART, expected_file_sha="0" * 64)
    expect_ff_error("parents: P4 semantic-pin drift is rejected",
                    "PARENT_SEMANTIC_HASH_MISMATCH", ffr.load_p4s2_parent,
                    P4S2_ART, expected_semantic_sha="0" * 64)
    expect_ff_error("parents: IMPL-B file-pin drift is rejected",
                    "PARENT_FILE_HASH_MISMATCH", ffr.load_implb_parent,
                    IMPLB_ART, expected_file_sha="0" * 64)
    expect_ff_error("parents: IMPL-B semantic-pin drift is rejected",
                    "PARENT_SEMANTIC_HASH_MISMATCH", ffr.load_implb_parent,
                    IMPLB_ART, expected_semantic_sha="0" * 64)


def test_parent_sidecars() -> None:
    _CURRENT[0] = "test_parent_sidecars"
    for art, loader, label in (
            (P3_ART, ffr.load_p3_parent, "P3"),
            (P4S2_ART, ffr.load_p4s2_parent, "P4"),
            (IMPLB_ART, ffr.load_implb_parent, "IMPL-B")):
        with tempfile.TemporaryDirectory() as td:
            bare = os.path.join(td, os.path.basename(art))
            with open(art, "rb") as src, open(bare, "wb") as dst:
                dst.write(src.read())
            expect_ff_error(f"sidecars: {label} without sidecar refused",
                            "PARENT_SIDECAR_MISSING", loader, bare)
        with tempfile.TemporaryDirectory() as td:
            bare = os.path.join(td, os.path.basename(art))
            with open(art, "rb") as src, open(bare, "wb") as dst:
                dst.write(src.read())
            with open(bare + ".sha256", "w", encoding="utf-8") as fh:
                fh.write(f"{'0' * 64}  {os.path.basename(art)}\n")
            expect_ff_error(f"sidecars: {label} with a wrong sidecar "
                            f"refused", "PARENT_SIDECAR_MISMATCH",
                            loader, bare)


def test_spline_b_consumption() -> None:
    _CURRENT[0] = "test_spline_b_consumption"
    ctx = _CTX[0]
    got = ffr.require_spline_b(ffr.SPLINE_B)
    check("spline_b: exact float64 equality returns the value",
          got == ffr.SPLINE_B and isinstance(got, float), repr(got))
    expect_ff_error("spline_b: a mismatched B is rejected",
                    "SPLINE_B_MISMATCH", ffr.require_spline_b, 3.0)
    expect_ff_error("spline_b: a non-finite B is rejected",
                    "SPLINE_B_INVALID", ffr.require_spline_b,
                    float("nan"))
    cal = ctx["implb"]["calibration"]
    check("spline_b: calibration provenance (q/p99.9/linear/1.1)",
          cal["q"] == 4.690530441415376 and cal["percentile"] == 99.9
          and cal["method"] == "linear" and cal["margin"] == 1.1,
          json.dumps(cal))


def test_generator_pin() -> None:
    _CURRENT[0] = "test_generator_pin"
    ctx = _CTX[0]
    try:
        ffr.enforce_generator_pin(ctx["seed_prov"])
        ok = True
    except ffr.FreeFlowError as exc:  # noqa: BLE001 -- reported
        ok = False
        logger.error("generator pin rejected the live source: %s", exc)
    check("generator: live mask generator satisfies the registered pin",
          ok, str(ctx["seed_prov"].get("mask_seed_source_sha256")))
    forged = dict(ctx["seed_prov"])
    forged["mask_seed_source_sha256"] = "0" * 64
    expect_ff_error("generator: a forged generator hash is rejected",
                    "GENERATOR_HASH_MISMATCH", ffr.enforce_generator_pin,
                    forged)
    expect_ff_error("generator: an unresolved provenance is rejected",
                    "GENERATOR_HASH_MISMATCH", ffr.enforce_generator_pin,
                    {"resolved": False})


def test_failure_boundary() -> None:
    """Drive main() with parents verified and a boom AFTER parent
    verification: exit 2, typed UNEXPECTED_RUNTIME_ERROR record, sidecar
    verifies, no facts artefact, no claim residue. Module attributes are
    stubbed and restored (IMPLBT pattern); nothing is written outside the
    temp dir."""
    _CURRENT[0] = "test_failure_boundary"
    # sys.modules[__name__] is the RUNNING module object both when the
    # script executes as __main__ and when it is imported as a package
    # module -- a plain `import ... as self_mod` would resolve a SECOND
    # module instance under the package name when running as __main__,
    # and the stubs would land on the wrong object.
    self_mod = sys.modules[__name__]
    with tempfile.TemporaryDirectory() as td:
        orig_vp = self_mod.verify_parents
        orig_ctx = self_mod.build_stage_context
        orig_run = self_mod._run_class_a
        self_mod.verify_parents = lambda *a, **k: {
            "p0": {"facts_sha256": "stub"}, "p0s": {"facts_sha256": "stub"}}
        self_mod.build_stage_context = lambda data_root: {"stub": True}
        def _boom(ctx):
            raise RuntimeError("synthetic boundary probe boom")
        self_mod._run_class_a = _boom
        try:
            rc = main(["--repo-dir", _REPO, "--data-root", "/unused",
                       "--out-dir", td, "--mode", "authoritative",
                       "--p0-facts", "x", "--p0s-facts", "x",
                       "--p0s-script", "x"])
        finally:
            self_mod.verify_parents = orig_vp
            self_mod.build_stage_context = orig_ctx
            self_mod._run_class_a = orig_run
        check("boundary: a post-parent-verification failure exits 2",
              rc == EXIT_ERROR, str(rc))
        err_path = os.path.join(td, f"{ERROR_PREFIX}.json")
        check("boundary: the error record is written under the error "
              "prefix", os.path.isfile(err_path), err_path)
        record = {}
        if os.path.isfile(err_path):
            with open(err_path, "r", encoding="utf-8") as fh:
                record = json.load(fh)
        check("boundary: error record is typed UNEXPECTED_RUNTIME_ERROR "
              "and artefact_type=error",
              record.get("error_code") == "UNEXPECTED_RUNTIME_ERROR"
              and record.get("artefact_type") == "error"
              and record.get("schema") == "seqref-impl-error/1",
              str(record.get("error_code")))
        check("boundary: no facts artefact was published",
              not os.path.exists(os.path.join(td, f"{FACTS_PREFIX}.json")))
        side_ok = False
        if os.path.isfile(err_path):
            try:
                side_ok = verify_sidecar(err_path) == file_sha256(err_path)
            except Exception as exc:  # noqa: BLE001 -- reported
                logger.error("boundary sidecar verification raised: %s",
                             exc)
        check("boundary: the error record's sidecar verifies", side_ok)
        residue = [n for n in os.listdir(td) if n.endswith(".claim")]
        check("boundary: no claim residue remains", residue == [],
              str(residue))
        detail = record.get("detail") or {}
        check("boundary: failure is recorded as AFTER parent verification",
              detail.get("raised_after_parent_verification") is True,
              str(detail))


def test_no_block_taxonomy() -> None:
    _CURRENT[0] = "test_no_block_taxonomy"
    block_exit = "EXIT_" + "BLOCK"
    block_type = "Stage" + "Block"
    sources = []
    for path in (os.path.join(_REPO, "seqref_mri", "src",
                              "free_flow_runtime.py"),
                 os.path.join(_REPO, "seqref_mri", "scripts",
                              "train_free_flow.py"),
                 os.path.abspath(__file__)):
        with open(path, "r", encoding="utf-8") as fh:
            sources.append(fh.read())
    check("taxonomy: no BLOCK exit anywhere in the IMPL stage sources",
          all(block_exit not in s for s in sources))
    check("taxonomy: no BLOCK verdict type anywhere in the IMPL stage "
          "sources", all(block_type not in s for s in sources))


def test_publication() -> None:
    _CURRENT[0] = "test_publication"
    facts = {"schema": FACTS_SCHEMA, "artefact_type": "publication_probe",
             "stage": STAGE, "run": {}}
    with tempfile.TemporaryDirectory() as td:
        path, sha = publish_stage(dict(facts), td, FACTS_PREFIX, STAGE)
        check("publication: first publish lands at the authoritative name",
              os.path.basename(path) == f"{FACTS_PREFIX}.json",
              os.path.basename(path))
        check("publication: published sidecar verifies against the file",
              verify_sidecar(path) == file_sha256(path))
        path2, _ = publish_stage(dict(facts), td, FACTS_PREFIX, STAGE)
        check("publication: rerun writes a stamped sibling, never "
              "overwrites",
              os.path.basename(path2).startswith(f"{FACTS_PREFIX}.")
              and os.path.basename(path2) != f"{FACTS_PREFIX}.json",
              os.path.basename(path2))
        check("publication: the authoritative file is untouched by the "
              "rerun", file_sha256(path) == sha)
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, f".{FACTS_PREFIX}.claim"), "w",
                  encoding="utf-8") as fh:
            fh.write("stage=IMPL pid=0 token=stale utc=probe\n")
        expect_error("publication: a held claim refuses publication",
                     "PUBLICATION_CLAIM_HELD", publish_stage,
                     dict(facts), td, FACTS_PREFIX, STAGE)
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, f"{FACTS_PREFIX}.json.sha256"), "w",
                  encoding="utf-8") as fh:
            fh.write(f"{'0' * 64}  {FACTS_PREFIX}.json\n")
        expect_runtime_error("publication: an orphan sidecar halts "
                             "publication", "unpaired", publish_stage,
                             dict(facts), td, FACTS_PREFIX, STAGE)
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, f"{FACTS_PREFIX}.tmp0"), "w",
                  encoding="utf-8") as fh:
            fh.write("residue")
        expect_error("publication: stale temporary residue halts "
                     "publication", "STALE_TEMPORARY_FOUND", publish_stage,
                     dict(facts), td, FACTS_PREFIX, STAGE)


# ---------------------------------------------------------------------------
# Fixture registry + STATIC expected counts (re-derived per rewrite, never
# carried forward)
# ---------------------------------------------------------------------------

FIXTURES = [
    ("test_a1_finiteness", test_a1_finiteness),
    ("test_a2_dimensions", test_a2_dimensions),
    ("test_a3_acquired_fixity", test_a3_acquired_fixity),
    ("test_a4_scaling_roundtrip", test_a4_scaling_roundtrip),
    ("test_a5_coordinate_maps", test_a5_coordinate_maps),
    ("test_a6_micro_training", test_a6_micro_training),
    ("test_a7_mask_reachability", test_a7_mask_reachability),
    ("test_a8_binding", test_a8_binding),
    ("test_a9_p4_gather", test_a9_p4_gather),
    ("test_a10_packing", test_a10_packing),
    ("test_construction_fixity", test_construction_fixity),
    ("test_parent_loaders", test_parent_loaders),
    ("test_parent_sidecars", test_parent_sidecars),
    ("test_spline_b_consumption", test_spline_b_consumption),
    ("test_generator_pin", test_generator_pin),
    ("test_failure_boundary", test_failure_boundary),
    ("test_no_block_taxonomy", test_no_block_taxonomy),
    ("test_publication", test_publication),
]

EXPECTED_COUNTS = {
    "test_a1_finiteness": 3,
    "test_a2_dimensions": 4,
    "test_a3_acquired_fixity": 3,
    "test_a4_scaling_roundtrip": 3,
    "test_a5_coordinate_maps": 10,
    "test_a6_micro_training": 7,
    "test_a7_mask_reachability": 4,
    "test_a8_binding": 8,
    "test_a9_p4_gather": 3,
    "test_a10_packing": 6,
    "test_construction_fixity": 7,
    "test_parent_loaders": 10,
    "test_parent_sidecars": 6,
    "test_spline_b_consumption": 4,
    "test_generator_pin": 3,
    "test_failure_boundary": 7,
    "test_no_block_taxonomy": 2,
    "test_publication": 7,
}

_CTX: list = [None]


# ---------------------------------------------------------------------------
# Class-A runner (authoritative mode) -- the SAME production check bodies
# the fixtures exercise; any failure propagates as StageError to the
# failure boundary.
# ---------------------------------------------------------------------------

A_CHECKS = [
    ("a1_finiteness", a1_finiteness),
    ("a2_dimensions", a2_dimensions),
    ("a3_acquired_fixity", a3_acquired_fixity),
    ("a4_scaling_roundtrip", a4_scaling_roundtrip),
    ("a5_coordinate_maps", a5_coordinate_maps),
    ("a6_micro_training", a6_micro_training),
    ("a7_mask_reachability", a7_mask_reachability),
    ("a8_binding", a8_binding),
    ("a9_p4_gather", a9_p4_gather),
    ("a10_packing", a10_packing),
]


def _run_class_a(ctx: dict) -> dict:
    evidence = {}
    for name, fn in A_CHECKS:
        logger.info("[%s] Class-A check %s", SCRIPT_ID, name)
        evidence[name] = fn(ctx)
        logger.info("[%s] %s PASS: %s", SCRIPT_ID, name,
                    json.dumps(evidence[name], default=str)[:400])
    return evidence


def _run_fixtures(ctx: dict) -> bool:
    _CTX[0] = ctx
    for name, fn in FIXTURES:
        _CURRENT[0] = name
        logger.info("[%s] fixture %s", SCRIPT_ID, name)
        try:
            fn()
        except StageError as exc:
            check("fixture body raised an unexpected StageError", False,
                  f"{exc.error_code}: {exc.reason}")
        except Exception as exc:  # noqa: BLE001 -- reported, not hidden
            check("fixture body raised an unexpected exception", False,
                  f"{type(exc).__name__}: {exc}")
    for fixture, name, cond, detail in _RESULTS:
        status = "PASS" if cond else "FAIL"
        line = f"[{status}] {fixture} :: {name}"
        if detail:
            line += f" -- {detail}"
        (logger.info if cond else logger.error)(line)
    per_fixture = {}
    for fixture, _, cond, _ in _RESULTS:
        per_fixture.setdefault(fixture, []).append(cond)
    coverage_ok = True
    for name, expected in EXPECTED_COUNTS.items():
        got = len(per_fixture.get(name, []))
        if got != expected:
            logger.error("[%s] coverage mismatch for %s: ran %d checks, "
                         "registry expects %d -- the registry is STALE; "
                         "re-derive it", SCRIPT_ID, name, got, expected)
            coverage_ok = False
    ran = set(per_fixture)
    if ran != set(EXPECTED_COUNTS):
        logger.error("[%s] fixture registry drifted: ran %s, expected %s",
                     SCRIPT_ID, sorted(ran), sorted(EXPECTED_COUNTS))
        coverage_ok = False
    all_pass = all(cond for _, _, cond, _ in _RESULTS)
    n_fail = sum(1 for _, _, cond, _ in _RESULTS if not cond)
    logger.info("[%s] fixtures: %d checks, %d failed, coverage_ok=%s",
                SCRIPT_ID, len(_RESULTS), n_fail, coverage_ok)
    return bool(all_pass and coverage_ok)


# ---------------------------------------------------------------------------
# Facts (authoritative mode): schema seqref-impl-facts/1
# ---------------------------------------------------------------------------

IMPL_LOCAL_FILES = [
    "seqref_mri/src/free_flow_runtime.py",
    "seqref_mri/scripts/train_free_flow.py",
    "seqref_mri/scripts/impl_selftest.py",
    "seqref_mri/src/base_experts.py",
    "seqref_mri/src/conditioner.py",
    "seqref_mri/src/flows/nsf_layer.py",
    "seqref_mri/src/fastmri_data.py",
    "seqref_mri/src/residual_decoder.py",
    "seqref_mri/src/preflight_parents_p3.py",
    "seqref_mri/scripts/train_base.py",
]


def _path_free(rec):
    if isinstance(rec, dict):
        return {k: _path_free(v) for k, v in rec.items() if k != "path"}
    if isinstance(rec, list):
        return [_path_free(v) for v in rec]
    return rec


def _code_record() -> dict:
    code = dict(hash_project_code(_REPO, os.path.abspath(__file__)))
    hashed = []
    for rel in IMPL_LOCAL_FILES:
        path = os.path.join(_REPO, rel)
        if not os.path.isfile(path):
            logger.error("[%s] IMPL-local code-hash file missing: %s",
                         SCRIPT_ID, path)
            raise _fail_a("CODE_HASH_FILE_MISSING",
                             f"project-local file required for the IMPL "
                             f"code hash is missing: {rel}")
        hashed.append({"relpath": rel, "sha256": file_sha256(path)})
    code["impl_local"] = hashed
    code["impl_local_note"] = (
        "the IMPL stage's own modules plus every P3-era module the "
        "production path executes; the frozen project hash block covers "
        "the preflight core")
    return code


def _build_facts(ctx: dict, parents: dict, evidence: dict,
                 args) -> dict:
    frozen_constants = {
        "FLOW_FAMILY": ffr.FLOW_FAMILY,
        "FLOW_DIM_REAL": ffr.FLOW_DIM_REAL,
        "N_FREE_COMPLEX": ffr.N_FREE_COMPLEX,
        "GRID": [ffr.GRID_H, ffr.GRID_W],
        "SPLINE_B": ffr.SPLINE_B,
        "SPLINE_PERCENTILE": ffr.SPLINE_PERCENTILE,
        "SPLINE_PERCENTILE_METHOD": ffr.SPLINE_PERCENTILE_METHOD,
        "SPLINE_MARGIN": ffr.SPLINE_MARGIN,
        "SPLINE_B_CONSUMPTION": "hash-pinned empirical constant from the "
                                "authoritative IMPL-B artefact; exact "
                                "float64 equality asserted",
        "NSF_K": ffr.NSF_K, "NSF_HIDDEN": ffr.NSF_HIDDEN,
        "NSF_N_LAYERS": ffr.NSF_N_LAYERS,
        "H_DIM": ffr.H_DIM, "COND_WIDTH": ffr.COND_WIDTH,
        "COND_IN_CHANNELS": ffr.COND_IN_CHANNELS,
        "USE_FILM": ffr.USE_FILM, "COND_USE_V2": ffr.COND_USE_V2,
        "FILM_HIDDEN": ffr.FILM_HIDDEN, "FILM_DEPTH": ffr.FILM_DEPTH,
        "FILM_USE_GELU": ffr.FILM_USE_GELU,
        "FILM_AFFINE": ffr.FILM_AFFINE,
        "Y_RESIDUAL_ALPHA_INIT": ffr.Y_RESIDUAL_ALPHA_INIT,
        "MASK_BITS": ffr.MASK_BITS,
        "MASK_EMBED_DIM": ffr.MASK_EMBED_DIM,
        "MASK_WEIGHT_INIT_STD": ffr.MASK_WEIGHT_INIT_STD,
        "MASK_BIAS_INIT": ffr.MASK_BIAS_INIT,
        "MASK_EFFECT_REL_MIN": ffr.MASK_EFFECT_REL_MIN,
        "MODEL_INIT_SEED": ffr.MODEL_INIT_SEED,
        "A3_ACQUIRED_FIXITY_MAX": ffr.A3_ACQUIRED_FIXITY_MAX,
        "A4_SCALING_ROUNDTRIP_MAX": ffr.A4_SCALING_ROUNDTRIP_MAX,
        "A4_AUX_NSF_ROUNDTRIP_MAX": ffr.A4_AUX_NSF_ROUNDTRIP_MAX,
        "A6_SEED": A6_SEED, "FIXTURE_SEED": FIXTURE_SEED,
    }
    parents_rec = {
        "parents_id": parents.get("parents_id"),
        "p0": _path_free(parents.get("p0")),
        "p0s": _path_free(parents.get("p0s")),
        "p3_coordinate_map": _path_free({k: v for k, v in ctx["p3"].items()
                                         if k != "bindings"}),
        "p3_bindings_count": len(ctx["p3"]["bindings"]),
        "p4_scaling_statistics": _path_free({k: v
                                             for k, v in ctx["p4"].items()
                                             if k != "location_index"}),
        "implb_calibration": _path_free(ctx["implb"]),
        "mask_seed_generator": {
            "source_sha256": ffr.GENERATOR_SOURCE_SHA256,
            "binding": "executing generator hash == registered pin"},
    }
    facts = {
        "schema": FACTS_SCHEMA,
        "script": {"id": SCRIPT_ID, "version": SCRIPT_VERSION,
                   "lifetime": "KEEP"},
        "stage": STAGE,
        "artefact_type": "stage_facts",
        "run_mode": "authoritative",
        "authoritative": True,
        "frozen_constants": frozen_constants,
        "class_a": evidence,
        "a6_protocol": evidence["a6_micro_training"]["protocol"],
        "verdict": "PASS",
        "verdict_reason": "A1-A10 all PASS on the production path with "
                          "the pinned fixtures and pinned micro-training "
                          "protocol; implementation validity established; "
                          "TINY remains a separate stage",
        "summary": {
            "class_a_checks": [name for name, _ in A_CHECKS],
            "checks_passed": len(A_CHECKS),
            "checks_failed": 0,
            "fixtures": [p["label"] for p in PINNED_FIXTURES],
            "exit_rule": "PASS only if A1-A10 all PASS; any failed check "
                         "=> overall verdict ERROR, exit 2, typed error "
                         "record, process HOLD; TINY blocked until "
                         "Class-A PASS",
            "plots": "none (machine-readable evidence only)"},
        "parents": parents_rec,
        "mask_seed_provenance": _path_free(ctx["seed_prov"]),
        "dataset_provenance": {
            "split": "train", "mode": "eval",
            "corpus_slices": ffr.EXPECTED_CORPUS_SLICES,
            "pinned_fixtures": PINNED_FIXTURES,
            "binding_rule": "binding verification BEFORE any "
                            "decode/training work"},
        "code": _code_record(),
        "run": {**environment_record(_REPO, sys.argv),
                "hash_note": "file sha256 + sidecar; semantic sha256 "
                             "over the path-free semantic payload (run/ "
                             "excluded as volatile)"},
    }
    semantic = {k: v for k, v in facts.items() if k != "run"}
    attach_semantic_hash(facts, semantic)
    return facts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SEQREF-IMPL v0.1 Class-A stage (A1-A10); fixtures "
                    "mode publishes nothing, authoritative mode publishes "
                    "impl/implementation_facts.json (seqref-impl-facts/1)")
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", default=os.path.join(
        _REPO, "seqref_mri", "results", "_diag", "impl"))
    p.add_argument("--mode", choices=("fixtures", "authoritative"),
                   default="fixtures")
    p.add_argument("--p0-facts", default=None)
    p.add_argument("--p0s-facts", default=None)
    p.add_argument("--p0s-script", default=None)
    p.add_argument("--log-file", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode="w",
                                            encoding="utf-8"))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s",
                        handlers=handlers, force=True)
    if os.path.realpath(args.repo_dir) != os.path.realpath(_REPO):
        print(f"--repo-dir {args.repo_dir!r} does not resolve to the "
              f"installation {_REPO!r}; refusing to run against a copy",
              file=sys.stderr)
        return EXIT_ERROR
    torch.set_num_threads(1)
    parents = None
    try:
        if args.mode == "authoritative":
            if not (args.p0_facts and args.p0s_facts and args.p0s_script):
                raise _fail_a(
                    "PARENT_INPUT_MISSING",
                    "authoritative mode requires --p0-facts, --p0s-facts "
                    "and --p0s-script so the parent chain is verified, "
                    "not assumed",
                    detail={}, write_record=False)
            parents = verify_parents(_REPO, args.p0_facts, args.p0s_facts,
                                     args.p0s_script)
        ctx = build_stage_context(args.data_root)
        if args.mode == "fixtures":
            ok = _run_fixtures(ctx)
            if not ok:
                logger.error("[%s] fixtures FAILED -- implementation is "
                             "NOT validated; no artefact published",
                             SCRIPT_ID)
                return EXIT_ERROR
            logger.info("[%s] all fixtures PASS -- the stage may be run "
                        "in authoritative mode", SCRIPT_ID)
            return EXIT_PASS
        evidence = _run_class_a(ctx)
        facts = _build_facts(ctx, parents, evidence, args)
        path, sha = publish_stage(facts, args.out_dir, FACTS_PREFIX,
                                  STAGE)
        logger.info("[%s] verdict PASS; published %s sha256=%s",
                    SCRIPT_ID, path, sha)
        return EXIT_PASS
    except StageError as exc:
        logger.error("[%s] %s: %s", SCRIPT_ID, exc.error_code, exc.reason)
        publish_error(exc, args.out_dir, ERROR_PREFIX, STAGE,
                      parents=parents)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 -- the registered boundary
        logger.exception("[%s] unexpected runtime failure", SCRIPT_ID)
        wrapped = StageError(
            "UNEXPECTED_RUNTIME_ERROR",
            f"{type(exc).__name__}: {exc}",
            detail={"exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "raised_after_parent_verification":
                        parents is not None},
            write_record=parents is not None)
        publish_error(wrapped, args.out_dir, ERROR_PREFIX, STAGE,
                      parents=parents)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
