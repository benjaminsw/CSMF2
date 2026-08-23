# SEQREF-V02S v0.5 -- scripts.v02_selftest
# LIFETIME: KEEP
# =============================================================================
# Purpose: candidate v0.2 selftest (V02PLAN v0.2 SS7 ten-row matrix, SS12
#          no-fallback register). Every row is assert-verified on synthetic
#          fixtures; every exercised ERROR path must logger.error + typed
#          raise (V02Error, the single identity from SEQREF-V02M). Golden
#          arrays/values were pinned 2026-08-22 against NumPy PCG64 in the
#          campaign sandbox and are embedded as constants -- a generator or
#          stream-order change FAILS the pin, never silently re-pins.
# Invocation modes (impl_selftest precedent):
#   * --mode fixtures (default): the full suite, publishes NOTHING
#   * --mode authoritative: the same suite + publication of the selftest
#     report (schema seqref-v02-selftest/1) via the claim-guarded
#     publication machinery
# Taxonomy: exit 0 = all fixtures PASS and coverage_ok; exit 2 = ERROR.
#   There is no exit 1 and no skip: an unexercised registered fixture or
#   an ERROR path that fails to fire is itself an ERROR (SS7 row 10).
# Environment-bound fixtures (model/dataset machinery) import their
#   dependencies lazily and totally; they run on the registered host.
# Changelog (NEW in v0.1):
#   * Introduced under V02PLAN v0.2 (LOCKED 2026-08-21).
# v0.2 (2026-08-22, reviewer NO-GO follow-up): f09 gained proofs for
#   the V02T/V02P unexpected-exception boundary catches (exit 2, never
#   1) and the V02F defined-null decomposition-share render; register
#   gained FACTS_SHAPE_MISMATCH. Bug-fix verification extension only.
# v0.3 (2026-08-22, reviewer NO-GO follow-up): f09 gained boundary-
#   catch proofs for V02M and V02F (all four executable modules now
#   proven exit-2 under unexpected exceptions). Bug-fix extension.
# v0.4 (2026-08-22, reviewer NO-GO follow-up, coverage semantics): the
#   SS7 row-9 coverage contract is now MECHANICALLY EXHAUSTIVE. The
#   coverage universe is extracted from the five sibling module sources
#   at runtime (every _fail("CODE") site plus the code literals threaded
#   through v02_eval's sidecar loader), never curated by hand; f10 proves
#   per-module totality (extracted == observed + deferred, every deferral
#   justified in writing, no stale deferrals, no unknown observations).
#   f09 gained fixtures for every fixture-reachable branch: V02M
#   DATASET_IMPORT_FAILED; V02T run()-level manifest gates + both mask
#   gates; V02E import/preflight/manifest/checkpoint loader gates, the
#   run()-level content/validity gates over a fabricated frozen-input
#   set, and seam-tamper proofs of the three StageError-conversion
#   wrappers; V02F value/consistency/checkpoint gates. _expect_error now
#   records (module, code) pairs -- a shared code string must fire in
#   EACH module's own branch. Also fixed in v0.4 (defect found during
#   sandbox verification): f04's same-encode wiring tolerance is now
#   RELATIVE (1e-6) with per-slice comparisons -- the eval-mode
#   per-slice path accumulates the Gaussian base term in float64 while
#   the production batch objective accumulates in float32, and at
#   FLOW_DIM_REAL = 13,824 the f32 summation floor (~1 ulp of the NLL
#   magnitude, up to 3.2e-3 observed) exceeds any absolute tolerance
#   below ~4e-3; the pinned 1e-3 absolute was environment-fragile. A
#   real wiring break still fails by 4+ orders of magnitude.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from seqref_mri.scripts import v02_eval as v02e
from seqref_mri.scripts import v02_manifests as v02m

logger = logging.getLogger("seqref_mri.v02_selftest")

__version__ = "0.5"
__abbr__ = "SEQREF-V02S"

EXIT_PASS = 0
EXIT_ERROR = 2
REPORT_SCHEMA = "seqref-v02-selftest/1"
REPORT_PREFIX = "v02_selftest"

SAME_ENCODE_RTOL = 1e-6    # RELATIVE wiring-invariant tolerance
                            # (selftest only; NOT a scientific gate). The
                            # eval-mode per-slice path accumulates the
                            # Gaussian base term in float64 while the
                            # production batch objective accumulates in
                            # float32; at FLOW_DIM_REAL = 13,824 the f32
                            # summation noise floor is ~1 ulp of the NLL
                            # magnitude (~2e-3 on NLL ~ 2e4), so an
                            # ABSOLUTE tolerance is environment-fragile.
                            # A real wiring break (wrong mask, wrong
                            # coordinate map, dropped logdet) shifts the
                            # NLL by O(10^2)-O(10^3) -- 4+ orders of
                            # magnitude above this tolerance.
BOOT_FIXTURE_B = 200        # fixture bootstrap size (the scientific
                            # B = 10,000 stays locked in SEQREF-V02E)

# Golden pins (2026-08-22, NumPy PCG64; computed from the registered
# generator calls, embedded verbatim).
GOLDEN_D3_POSITIONS = [83, 49, 160, 51, 58, 130, 140, 106, 78, 57, 172,
                       18, 37, 188, 143, 50, 185, 10, 102, 34, 43, 178,
                       28, 16, 71, 44, 194, 144, 122, 131, 126, 196]
GOLDEN_EPOCH_ORDERS_16 = [
    [2, 11, 3, 10, 0, 4, 7, 5, 14, 12, 6, 9, 13, 8, 1, 15],
    [2, 11, 14, 5, 12, 4, 1, 0, 8, 10, 13, 6, 7, 9, 3, 15],
    [13, 2, 15, 8, 12, 4, 7, 9, 14, 6, 3, 5, 0, 1, 10, 11]]
GOLDEN_MIDPOINTS = {36: 17, 35: 17, 2: 0, 1: 0}
GOLDEN_BOOT = {
    "g_train": (0.1063757142857143, (0.09999999999999999, 0.13)),
    "g_hold": (0.0956, (0.08499999999999999, 0.10506250000000002)),
    "r": (0.9021023424687584, (0.7584336180124224, 1.05)),
    "nmse_ratio": (0.6398901381145613, (0.6, 0.6891891891891891))}

# ---------------------------------------------------------------------------
# Coverage contract (V02PLAN SS7 row 9: "every ERROR path is exercised
# deliberately"). The coverage universe is EXTRACTED MECHANICALLY from
# the five sibling module sources at runtime -- every `_fail("CODE")`
# site plus the code literals threaded through v02_eval's sidecar-loader
# wrapper -- never curated by hand. Each extracted code is then either
# OBSERVED (a fixture fired that exact branch, recorded per module) or
# DEFERRED with a written justification below; f10 proves the per-module
# totality extracted == observed + deferred. Structural source-token
# checks are supplementary and never substitute for an exercised branch.
# SEQREF-V02S's own three harness codes (SELFTEST_FIXTURE_FAILED,
# ENV_IMPORT_FAILED, PUBLICATION_FAILURE) are out of scope: they are the
# suite's own failure surface, not a module-under-test's contract.
# ---------------------------------------------------------------------------
DEFERRED_JUSTIFICATIONS: dict = {
    "SEQREF-V02M": {
        "POPULATION_MISMATCH":
            "dataset-bound: fires inside build_all against the live "
            "fastMRI train/val traversal; a fabricated traversal would "
            "exercise the fixture, not the gate.",
        "PARTITION_MISMATCH":
            "dataset-bound: same build_all path, guarding the written "
            "epoch manifests against the verified dataset population.",
        "EXPOSURE_CAP_VIOLATION":
            "unreachable by construction: three valid permutations of "
            "one population yield exactly 3 exposures per slice "
            "(proven positively in f08); a manifest that would trip "
            "this gate is rejected earlier as MANIFEST_NOT_A_PERMUTATION."},
    "SEQREF-V02T": {
        "MANIFEST_ENTRY_UNKNOWN":
            "in-loop, dataset-bound: a manifest entry absent from the "
            "live traversal mid-run.",
        "BATCH_MANIFEST_DRIFT":
            "in-loop, dataset-bound: the per-batch identity re-check "
            "against the live pipeline.",
        "PREPARE_KEYS_MISSING":
            "in-loop, dataset-bound: batch tensors from the live "
            "pipeline.",
        "MAP_DIMENSION_MISMATCH":
            "requires the pinned production coordinate-map builder "
            "(dec.build_coordinate_map) to disagree with the registered "
            "constants; unreachable with the pinned IMPLR chain, and "
            "fabricating the disagreement would exercise the fixture.",
        "STEP_COUNT_MISMATCH":
            "post-run assert; requires executing the full locked "
            "3,258-step budget.",
        "CHECKPOINT_SCHEDULE_MISMATCH":
            "post-run assert; requires the full training run.",
        "WALL_CLOCK_EXCEEDED":
            "48 h wall-clock guard; fires only mid-run on the "
            "registered host."},
    "SEQREF-V02P": {
        "MANIFEST_ENTRY_UNKNOWN":
            "dataset-bound manifest cross-check against the live "
            "traversal on the registered CUDA host.",
        "MANIFEST_TOO_SHORT":
            "dataset-bound: the live manifest population must support "
            "the frozen timing windows.",
        "TIMED_WINDOW_SHORT":
            "measurement-bound: the training-timing window on the "
            "registered CUDA host.",
        "EVAL_PROBE_SHORT":
            "measurement-bound: the eval-mode projection probe on the "
            "registered host.",
        "EVAL_PROBE_NON_FINITE":
            "measurement-bound: a non-finite probe measurement from "
            "the live model/dataset.",
        "CEILING_PROJECTION_EXCEEDED":
            "projection gate over real timing measurements; a fixture "
            "projection would exercise the fixture, not the measured "
            "ceiling."},
    "SEQREF-V02E": {
        "MANIFEST_ENTRY_UNKNOWN":
            "measurement-loop gate over the live dataset traversal.",
        "EVAL_STATE_BUILD_FAILURE":
            "measurement-loop gate: state construction requires the "
            "live DataLoader/collate pipeline.",
        "STATE_COUNT_MISMATCH":
            "measurement-loop gate: live state count vs manifest.",
        "DATA_PREMISE_FAILURE":
            "dataset construction past the pinned parent chain; "
            "fixtures are synthetic-only by design.",
        "EXPOSURE_CAP_VIOLATION":
            "unreachable by construction (see SEQREF-V02M): verified "
            "manifests that pass exposure_counts always yield exactly "
            "3 exposures.",
        "SEMANTIC_RERUN_MISMATCH":
            "end-of-run publication path (stamped-sibling rerun "
            "semantics against a prior published artefact).",
        "RERUN_PRIOR_ARTEFACT_MISMATCH":
            "end-of-run publication path.",
        "PUBLICATION_FAILURE":
            "end-of-run publication path (the claim-guarded "
            "publisher)."},
    "SEQREF-V02F": {}}


def _lazy_modules() -> dict:
    """abbr -> module object; V02T/V02P/V02F stay lazy (torch- and
    matplotlib-bound at module import)."""
    from seqref_mri.scripts import v02_plots as v02f
    from seqref_mri.scripts import v02_preflight as v02p
    from seqref_mri.scripts import v02_train as v02t
    return {"SEQREF-V02M": v02m, "SEQREF-V02T": v02t,
            "SEQREF-V02P": v02p, "SEQREF-V02E": v02e,
            "SEQREF-V02F": v02f}


def _extract_codes() -> dict:
    """The mechanical coverage universe: every typed ERROR code in each
    module's source -- `_fail("CODE")` sites plus the code literals
    handed to v02_eval's sidecar-loader wrapper (which re-raises them
    through the same logger.error + typed-raise funnel)."""
    out = {}
    for abbr, mod in _lazy_modules().items():
        src = Path(mod.__file__).read_text()
        codes = set(re.findall(r'_fail\(\s*"([A-Z0-9_]+)"', src))
        for call in re.findall(r"_load_sidecar_json\((.*?)\)", src,
                               re.DOTALL):
            codes.update(re.findall(r'"([A-Z][A-Z0-9_]{4,})"', call))
        _check(codes,
               f"{abbr}: mechanical extraction found no ERROR codes; "
               f"the module source or the _fail convention changed")
        out[abbr] = sorted(codes)
    return out

# Observations are (module abbr, code) pairs: two modules may share a
# code string, and each branch must be exercised in its OWN module.
_OBSERVED: set = set()


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise v02m.V02Error(f"{code}: {message}")


def _expect_error(fn, code: str, abbr: str) -> None:
    """The ERROR path must fire with logger.error + V02Error carrying the
    expected code FROM THE NAMED MODULE'S branch; anything else --
    including silent success -- is an ERROR of the selftest itself
    (SS7 row 10). Observations are recorded per (module, code): two
    modules may share a code string, and each branch must be exercised
    in its own module."""
    try:
        fn()
    except v02m.V02Error as exc:
        got = str(exc).split(":", 1)[0]
        if got != code:
            _fail("SELFTEST_FIXTURE_FAILED",
                  f"{abbr}: expected {code}, got {got} ({exc})")
        _OBSERVED.add((abbr, code))
        return
    _fail("SELFTEST_FIXTURE_FAILED",
          f"{abbr}: {code} did not fire; a silent pass in the selftest "
          f"is an ERROR (V02PLAN SS7 row 10)")


def _check(cond: bool, what: str) -> None:
    if not cond:
        _fail("SELFTEST_FIXTURE_FAILED", what)


# ---------------------------------------------------------------------------
# Row 1 -- manifest determinism (SEQREF-V02M).
# ---------------------------------------------------------------------------

def f01_manifest_determinism() -> dict:
    a = v02m.epoch_orders(64, 3, 0)
    b = v02m.epoch_orders(64, 3, 0)
    _check(all(np.array_equal(x, y) for x, y in zip(a, b)),
           "epoch_orders is not bitwise deterministic")
    d1 = {"b": 1, "a": [1, 2], "c": {"y": 2, "x": 1}}
    d2 = {"c": {"x": 1, "y": 2}, "a": [1, 2], "b": 1}
    _check(v02m.canonical_json(d1) == v02m.canonical_json(d2),
           "canonical_json depends on key insertion order")
    _check(v02m.manifest_sha256(d1) == v02m.manifest_sha256(d2),
           "manifest_sha256 depends on key insertion order")
    golden = v02m.epoch_orders(16, 3, 0)
    _check([o.tolist() for o in golden] == GOLDEN_EPOCH_ORDERS_16,
           "epoch_orders(16, 3, 0) diverges from the pinned golden "
           "stream; the PCG64(0) one-stream contract changed")
    _expect_error(lambda: v02m.epoch_orders(0), "POPULATION_INVALID",
                  "SEQREF-V02M")
    _expect_error(lambda: v02m.batch_partition(5, 0), "PARTITION_INVALID",
                  "SEQREF-V02M")
    return {"golden_pin": "epoch_orders(16,3,0)", "streams_checked": 2}


# ---------------------------------------------------------------------------
# Row 2 -- partition exactness (SEQREF-V02M).
# ---------------------------------------------------------------------------

def f02_partition_exactness() -> dict:
    parts = v02m.batch_partition(v02m.N_TRAIN_SLICES, v02m.BATCH_SIZE)
    _check(len(parts) == v02m.EXPECTED_BATCHES_PER_EPOCH,
           f"{len(parts)} batches != locked "
           f"{v02m.EXPECTED_BATCHES_PER_EPOCH}")
    sizes = [e - s for s, e in parts]
    _check(sizes[:-1] == [v02m.BATCH_SIZE] * (len(parts) - 1)
           and sizes[-1] == v02m.EXPECTED_FINAL_BATCH,
           "partition is not 1085 x 32 + 1 x 22")
    _check(parts[0][0] == 0 and parts[-1][1] == v02m.N_TRAIN_SLICES
           and all(parts[i][1] == parts[i + 1][0]
                   for i in range(len(parts) - 1)),
           "batch windows are not contiguous over [0, 34742)")
    orders = v02m.epoch_orders(v02m.N_TRAIN_SLICES)
    for ep, o in enumerate(orders):
        _check(int(np.unique(o).size) == v02m.N_TRAIN_SLICES,
               f"epoch {ep} repeats or omits slices within the epoch")
    return {"batches": len(parts), "final_batch": sizes[-1]}


# ---------------------------------------------------------------------------
# Row 3 -- checkpoint schedule (SEQREF-V02T; lazy torch-bound import).
# ---------------------------------------------------------------------------

def f03_checkpoint_schedule() -> dict:
    try:
        from seqref_mri.scripts import v02_train as v02t
    except ImportError as exc:
        _fail("ENV_IMPORT_FAILED", f"v02_train not importable: {exc}")
    _check(tuple(v02t.CHECKPOINT_STEPS) == (0, 1086, 2172, 3258),
           f"v02_train.CHECKPOINT_STEPS == {v02t.CHECKPOINT_STEPS}")
    _check(v02t.TOTAL_STEPS == 3258 and v02t.LEARNING_RATE == 1e-4,
           "v02_train locked budget/lr diverge from V02SPEC SS3")
    events = {0} | {s for s in range(1, v02t.TOTAL_STEPS + 1)
                    if s in v02t.CHECKPOINT_STEPS}
    _check(events == {0, 1086, 2172, 3258},
           f"simulated save events {sorted(events)} != the locked "
           f"schedule; saves must occur there and nowhere else")
    src = Path(v02t.__file__).read_text()
    for token in ("CHECKPOINT_SCHEDULE_MISMATCH", "STEP_COUNT_MISMATCH",
                  "save_checkpoint(model, step, out_root, checkpoints)",
                  "WALL_CLOCK_EXCEEDED", "logger.error"):
        _check(token in src,
               f"v02_train source lost the {token!r} guard token")
    return {"events": sorted(events)}


# ---------------------------------------------------------------------------
# Row 4 -- same-encode invariant (eval-mode endpoint encode vs production
# batch NLL; real model, synthetic states; registered host).
# ---------------------------------------------------------------------------

def f04_same_encode_invariant() -> dict:
    env = v02e._env()
    torch = env.torch
    model = env.ffr.build_model()      # registered default spline_b/seed
    model.eval()
    rng = np.random.Generator(np.random.PCG64(99))
    cols = (list(range(0, 44, 4)) + list(range(52, 96, 4)))[:16] \
        + list(range(44, 52))
    mask = torch.zeros(96, dtype=torch.bool)
    mask[torch.tensor(cols)] = True
    states = []
    for i in range(4):
        states.append({
            "identity": {"file": f"fixture_{i}.h5", "slice_index": i},
            "target": rng.standard_normal(
                (1, env.ffr.FLOW_DIM_REAL)),
            "cond": torch.from_numpy(rng.standard_normal(
                (1, 2, 96, 96)).astype(np.float32)),
            "mask": mask.unsqueeze(0)})
    per_slice = []
    for st in states:
        _z, ldj, log_pz = env.d2b._encode_slice(model, st)
        per_slice.append(-log_pz - ldj)
    targets, cond, msk = env.d2b._batch_tensors(states)
    with torch.no_grad():
        lp = model.log_prob_free(targets, cond, msk)  # production core
    nll_per = (-lp.detach().cpu().numpy()).astype(np.float64)
    nll_batch = float(env.tg._nll(model, targets, cond, msk))
    _check(np.isfinite(per_slice).all() and np.isfinite(nll_per).all()
           and np.isfinite(nll_batch),
           "non-finite NLL in the same-encode fixture")
    for i, (ps, bp) in enumerate(zip(per_slice, nll_per)):
        _check(abs(ps - float(bp))
               <= SAME_ENCODE_RTOL * max(1.0, abs(ps)),
               f"same-encode invariant violated at slice {i}: "
               f"|eval-mode per-slice NLL - production batch NLL| = "
               f"{abs(ps - float(bp)):.3e} exceeds the relative wiring "
               f"tolerance {SAME_ENCODE_RTOL:.0e} (eval mode, synthetic "
               f"states)")
    diff = abs(float(np.mean(per_slice)) - float(np.mean(nll_per)))
    _check(diff <= SAME_ENCODE_RTOL * max(1.0, abs(nll_batch)),
           f"same-encode invariant violated at the mean: {diff:.3e}")
    _check(abs(float(nll_per.mean()) - nll_batch)
           <= SAME_ENCODE_RTOL * max(1.0, abs(nll_batch)),
           "the production entry point tg._nll diverges from the "
           "per-sample log_prob_free mean beyond the wiring tolerance")
    return {"max_rel_diff": max(
                abs(ps - float(bp)) / max(1.0, abs(ps))
                for ps, bp in zip(per_slice, nll_per)),
            "rtol": SAME_ENCODE_RTOL, "n": 4}


# ---------------------------------------------------------------------------
# Row 5 -- R guard (SEQREF-V02E, pure).
# ---------------------------------------------------------------------------

def f05_r_guard() -> dict:
    boot = v02e._bootstrap([np.array([0.2, 0.2])], np.array([0.1, 0.1]),
                           np.array([1.0, 1.0]), np.array([0.5, 0.5]),
                           B=8)
    zero = v02e._transfer_block(np.zeros(8), np.zeros(4), boot)
    _check(zero["r"] is None and zero["v1"]["pass"] is False
           and zero["v2"]["pass"] is None,
           "G_train == 0 must yield R = null and V2 unevaluated; no "
           "coercion is permitted")
    _check(zero["bootstrap"]["r"].get("null") is True,
           "bootstrap R block must be a defined null when G_train == 0")
    neg = v02e._transfer_block(np.full(8, 0.20), np.full(4, -0.05), boot)
    _check(neg["r"] == -0.25 and neg["v1"]["pass"] is True
           and neg["v2"]["pass"] is False,
           "negative R must be recorded as-is and fail V2; no coercion")
    _check(v02e._classify(False, None, False)
           == "LIKELIHOOD_LEARNING_NOT_ESTABLISHED"
           and v02e._classify(True, False, True) == "TRANSFER_NOT_SUPPORTED"
           and v02e._classify(True, True, True)
           == "PROMISING_DATA_BUDGET_REDESIGN"
           and v02e._classify(True, True, False)
           == "LIKELIHOOD_TRANSFER_WITHOUT_RECONSTRUCTION_SUPPORT",
           "the locked four-label classification tree diverges")
    _expect_error(lambda: v02e._transfer_block(
        np.array([np.inf, 1.0]), np.array([0.1]), boot),
        "ENDPOINT_NON_FINITE", "SEQREF-V02E")
    return {"r_negative_recorded": -0.25, "r_null_on_zero_g_train": True}


# ---------------------------------------------------------------------------
# Row 6 -- bootstrap reproducibility (SEQREF-V02E, pure, golden-pinned).
# ---------------------------------------------------------------------------

def f06_bootstrap_reproducibility() -> dict:
    vol = [np.array([0.11, 0.09]), np.array([0.13]),
           np.array([0.10, 0.12, 0.08])]
    g_hold = np.array([0.09, 0.10, 0.11, 0.08])
    nmse0 = np.array([1.0, 1.1, 0.9, 1.0])
    nmse1 = np.array([0.6, 0.7, 0.65, 0.6])
    b1 = v02e._bootstrap(vol, g_hold, nmse0, nmse1, B=BOOT_FIXTURE_B)
    b2 = v02e._bootstrap(vol, g_hold, nmse0, nmse1, B=BOOT_FIXTURE_B)
    _check(b1 == b2, "bootstrap is not bitwise reproducible")
    _check(b1["seed"] == 3 and b1["generator"] == "PCG64"
           and b1["stream_order"] == ["holdout_gain", "train_gain",
                                      "holdout_nmse"],
           "bootstrap stream contract fields diverge")
    for key, (mean, ci) in GOLDEN_BOOT.items():
        _check(b1[key]["mean"] == mean
               and tuple(b1[key]["ci95"]) == tuple(ci),
               f"bootstrap golden pin {key} failed: "
               f"{b1[key]['mean']!r} != {mean!r}; the PCG64(3) stream "
               f"or ratio-of-means reduction changed")
    _expect_error(lambda: v02e._bootstrap(
        [np.array([0.0])], np.array([0.0]), np.array([1.0]),
        np.array([1.0]), B=4), "BOOTSTRAP_DENOMINATOR_ZERO", "SEQREF-V02E")
    _expect_error(lambda: v02e._bootstrap(
        vol, np.array([np.nan, 1.0, 1.0, 1.0]), nmse0, nmse1, B=4),
        "BOOTSTRAP_INPUT_NON_FINITE", "SEQREF-V02E")
    _expect_error(lambda: v02e._bootstrap([], g_hold, nmse0, nmse1, B=4),
                  "BOOTSTRAP_INPUT_INVALID", "SEQREF-V02E")
    return {"golden_pin": "PCG64(3) B=200 toy fixture"}


# ---------------------------------------------------------------------------
# Row 7 -- D3 draw reproducibility (SEQREF-V02M, pure, golden-pinned).
# ---------------------------------------------------------------------------

def f07_d3_draw_reproducibility() -> dict:
    pos = v02m.d3_monitor_positions()
    registered = np.random.Generator(np.random.PCG64(2)).choice(
        199, 32, replace=False)
    _check(np.array_equal(pos, registered),
           "d3_monitor_positions diverges from the registered PCG64(2) "
           "call")
    _check(pos.tolist() == GOLDEN_D3_POSITIONS,
           "the frozen 32-of-199 draw diverges from the golden pin")
    _check(len(set(pos.tolist())) == 32, "the D3 draw repeats positions")
    for n, expected in GOLDEN_MIDPOINTS.items():
        _check(v02m.midpoint_slice(n) == expected,
               f"midpoint_slice({n}) != {expected}")
    _expect_error(lambda: v02m.d3_monitor_positions(199, 200),
                  "D3_DRAW_INVALID", "SEQREF-V02M")
    _expect_error(lambda: v02m.midpoint_slice(0), "MIDPOINT_INVALID",
                  "SEQREF-V02M")
    return {"draw": pos.tolist()[:4], "golden_pin": True}


# ---------------------------------------------------------------------------
# Row 8 -- exposure accounting (SEQREF-V02M, pure).
# ---------------------------------------------------------------------------

def f08_exposure_accounting() -> dict:
    perms = [np.random.Generator(np.random.PCG64(10 + ep))
             .permutation(100) for ep in range(3)]
    counts = v02m.exposure_counts([p.tolist() for p in perms], 100)
    _check(np.all(counts == 3),
           "per-slice exposure over three permutations is not exactly 3 "
           "(the cap-3 contract is satisfied exactly by construction)")
    bad = perms[1].copy()
    bad[0] = bad[1]                     # repeat a slice, drop another
    _expect_error(lambda: v02m.exposure_counts(
        [perms[0].tolist(), bad.tolist(), perms[2].tolist()], 100),
        "MANIFEST_NOT_A_PERMUTATION", "SEQREF-V02M")
    _expect_error(lambda: v02m.exposure_counts([[0, 1]], 100),
                  "MANIFEST_SHAPE_INVALID", "SEQREF-V02M")
    return {"cap": 3, "observed_exact": True}


# ---------------------------------------------------------------------------
# Row 9 -- ERROR paths (every one must logger.error + typed raise).
# ---------------------------------------------------------------------------

def _write_with_sidecar(path: Path, doc: dict, tamper: bool) -> None:
    payload = json.dumps(doc, sort_keys=True).encode("utf-8")
    path.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    if tamper:
        sha = "0" * 64
    Path(str(path) + ".sha256").write_text(sha + "\n")


def _fab_manifest_set(root: Path) -> dict:
    """Fabricate a complete, hash-consistent frozen-input set (three
    identity-permutation epoch manifests over a synthetic 34,742-slice
    pool, a 199-volume holdout manifest, the registered D3 monitor
    draw, a matching train record, a within-ceiling preflight report).
    Every v02_eval.run() gate up to parent verification reads only
    these artefacts; identities are synthetic and no dataset,
    checkpoint file, or parent artefact is ever touched. Rewriting the
    set is idempotent (content hashing), so one directory serves every
    run()-level fixture in sequence."""
    n_train = v02m.N_TRAIN_SLICES
    n_hold = v02m.N_HOLDOUT_VOLUMES
    hashes = []
    for ep in range(v02m.N_EPOCHS):
        man = {"schema": "seqref-v02-manifest/1", "kind": "train_epoch",
               "epoch": ep, "n_slices": n_train,
               "entries": [{"dataset_index": i,
                            "file": f"fab_train_{i // 36}.h5",
                            "slice_index": i % 36}
                           for i in range(n_train)]}
        man["manifest_sha256"] = v02m.manifest_sha256(man)
        hashes.append(man["manifest_sha256"])
        _write_with_sidecar(root / f"v02_epoch{ep}_manifest.json", man,
                            tamper=False)
    hold = {"schema": "seqref-v02-manifest/1", "kind": "holdout",
            "n_volumes": n_hold,
            "entries": [{"file": f"fab_hold_{i}.h5",
                         "slice_index": i % 36}
                        for i in range(n_hold)]}
    hold["manifest_sha256"] = v02m.manifest_sha256(hold)
    _write_with_sidecar(root / "v02_holdout_manifest.json", hold,
                        tamper=False)
    positions = [int(p) for p in v02m.d3_monitor_positions()]
    d3m = {"schema": "seqref-v02-manifest/1", "kind": "d3_monitor",
           "n": len(positions),
           "entries": [{"draw_rank": r, "holdout_position": p,
                        "file": f"fab_hold_{p}.h5",
                        "slice_index": p % 36}
                       for r, p in enumerate(positions)]}
    d3m["manifest_sha256"] = v02m.manifest_sha256(d3m)
    _write_with_sidecar(root / "v02_d3_monitor_manifest.json", d3m,
                        tamper=False)
    rec = {"schema": "seqref-v02-train-record/1", "steps": 3258,
           "checkpoints": [{"step": s, "file": f"v02_ckpt_step{s}.pt",
                            "state_sha256": "a", "file_sha256": "b"}
                           for s in (0, 1086, 2172, 3258)],
           "manifest_sha256": hashes}
    _write_with_sidecar(root / "v02_train_record.json", rec,
                        tamper=False)
    (root / "v02_preflight.json").write_text(json.dumps({
        "schema": "seqref-v02-preflight/1",
        "projected_training_s": 100.0,
        "projected_training_h": 100.0 / 3600.0,
        "ceiling_s": 172800.0,
        "projected_endpoint_eval_s": 10.0,
        "projected_pm_bank_s": 10.0,
        "peak_memory_bytes": 1024}))
    return {"data_root": "x", "manifest_dir": str(root),
            "train_record": str(root / "v02_train_record.json"),
            "run_root": "x",
            "preflight": str(root / "v02_preflight.json"),
            "p0s_facts": "x", "p4_stats2": "x", "implb_facts": "x",
            "out_dir": "x"}


def f09_error_paths() -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="v02s_"))
    # manifest hash mismatch (V02E loader)
    man = {"schema": "seqref-v02-manifest/1", "kind": "holdout",
           "entries": [], "manifest_sha256": "x"}
    _write_with_sidecar(tmp / "v02_holdout_manifest.json", man,
                        tamper=True)
    _expect_error(lambda: v02e._load_manifest(
        tmp, "v02_holdout_manifest.json", "holdout"),
        "MANIFEST_HASH_MISMATCH", "SEQREF-V02E")
    # manifest hash mismatch (V02T loader, lazy torch-bound import)
    try:
        from seqref_mri.scripts import v02_train as v02t
    except ImportError as exc:
        _fail("ENV_IMPORT_FAILED", f"v02_train not importable: {exc}")
    eph = {"schema": "seqref-v02-manifest/1", "kind": "train_epoch",
           "epoch": 0, "n_slices": 0, "entries": [], "batches": []}
    _write_with_sidecar(tmp / "v02_epoch0_manifest.json", eph,
                        tamper=True)
    _expect_error(lambda: v02t.load_epoch_manifest(str(tmp), 0),
                  "MANIFEST_HASH_MISMATCH", "SEQREF-V02T")
    # partial-run detection (V02E): wrong step count, then a wrong
    # checkpoint schedule, both with valid sidecars
    rec = {"schema": "seqref-v02-train-record/1", "steps": 1000,
           "checkpoints": [{"step": s, "file": f"v02_ckpt_step{s}.pt",
                            "state_sha256": "a", "file_sha256": "b"}
                           for s in (0, 1086, 2172, 3258)],
           "manifest_sha256": ["m0", "m1", "m2"]}
    _write_with_sidecar(tmp / "v02_train_record.json", rec, tamper=False)
    mans = [{"manifest_sha256": f"m{i}"} for i in range(3)]
    _expect_error(lambda: v02e._load_train_record(
        str(tmp / "v02_train_record.json"), mans),
        "TRAIN_RECORD_MISMATCH", "SEQREF-V02E")
    rec2 = dict(rec, steps=3258,
                checkpoints=[{"step": s, "file": "f", "state_sha256": "a",
                              "file_sha256": "b"} for s in (0, 1086, 3258)])
    _write_with_sidecar(tmp / "v02_train_record2.json", rec2,
                        tamper=False)
    _expect_error(lambda: v02e._load_train_record(
        str(tmp / "v02_train_record2.json"), mans),
        "CHECKPOINT_SCHEDULE_MISMATCH", "SEQREF-V02E")
    # missing checkpoint (the existence gate fires before any env use;
    # env=None is a fixture, never reached)
    _expect_error(lambda: v02e._load_checkpoint(
        None, tmp, {"step": 0, "file": "nope.pt", "state_sha256": "a",
                    "file_sha256": "b"}, 1.0), "CHECKPOINT_MISSING", "SEQREF-V02E")
    # non-finite gradients (V02T gate, fabricated parameter)
    torch = v02e._env().torch
    model = torch.nn.Linear(2, 2)
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    _expect_error(lambda: v02t.check_gradients_finite(model, 7),
                  "GRADIENT_NON_FINITE", "SEQREF-V02T")
    # V02P host gate (tamper fixture: CUDA reported unavailable; run()
    # checks the host before touching cfg, so an empty cfg suffices)
    try:
        from seqref_mri.scripts import v02_preflight as v02p
    except ImportError as exc:
        _fail("ENV_IMPORT_FAILED", f"v02_preflight not importable: {exc}")
    original = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        _expect_error(lambda: v02p.run({}),
                      "REGISTERED_HOST_MISSING", "SEQREF-V02P")
    finally:
        torch.cuda.is_available = original
    # invalid facts schema / verdict key / missing key (V02F)
    try:
        from seqref_mri.scripts import v02_plots as v02f
    except ImportError as exc:
        _fail("ENV_IMPORT_FAILED", f"v02_plots not importable: {exc}")
    bad = tmp / "bad_schema.json"
    bad.write_text(json.dumps({"schema": "wrong/9"}))
    _expect_error(lambda: v02f.load_facts(str(bad)),
                  "FACTS_SCHEMA_MISMATCH", "SEQREF-V02F")
    ver = tmp / "verdict.json"
    ver.write_text(json.dumps({"schema": "seqref-v02-facts/1",
                               "verdict": "PASS"}))
    _expect_error(lambda: v02f.load_facts(str(ver)),
                  "FACTS_VERDICT_PRESENT", "SEQREF-V02F")
    _expect_error(lambda: v02f.plot_gain_summary({}, tmp),
                  "FACTS_KEY_MISSING", "SEQREF-V02F")
    # V02E remaining registered gates
    _expect_error(lambda: v02e._no_verdict_scan({"x": [{"verdict": 1}]},
                                                "facts"),
                  "FACTS_SCHEMA_VIOLATION", "SEQREF-V02E")
    _expect_error(lambda: v02e._v3_block(
        {0: np.array([20.0]), 3258: np.array([21.0])},
        {0: np.array([0.0]), 3258: np.array([0.5])},
        {"nmse_ratio": {"mean": 0.5, "ci95": [0.4, 0.6]}}),
        "VALIDITY_NMSE_UNDEFINED", "SEQREF-V02E")
    # Boundary-catch proofs (V02T/V02P v0.2, tamper fixtures: run is
    # patched to raise an unexpected exception): main() must log it and
    # return exit 2; no exception may escape Python as exit 1.
    argv_keep = sys.argv
    def _boom(cfg):
        raise RuntimeError("fixture-induced unexpected failure")
    orig_t, orig_p = v02t.run, v02p.run
    try:
        v02t.run = _boom
        sys.argv = ["v02_train.py", "--data-root", "x",
                    "--manifest-dir", "x", "--p4-stats2", "x",
                    "--implb-facts", "x", "--out-root", "x"]
        rc_t = v02t.main()
        v02p.run = _boom
        sys.argv = ["v02_preflight.py", "--data-root", "x",
                    "--manifest-dir", "x", "--p4-stats2", "x",
                    "--implb-facts", "x", "--out-dir", "x"]
        rc_p = v02p.main()
    finally:
        v02t.run, v02p.run = orig_t, orig_p
        sys.argv = argv_keep
    _check(rc_t == 2 and rc_p == 2,
           f"boundary catches must convert an unexpected exception to "
           f"exit 2 (got train={rc_t}, preflight={rc_p}); exit 1 is a "
           f"taxonomy violation")
    # V02M v0.2 boundary proof (tamper fixture: build_all patched to
    # raise); v02m is already module-level imported.
    orig_m = v02m.build_all
    try:
        v02m.build_all = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("fixture-induced unexpected failure"))
        sys.argv = ["v02_manifests.py", "--data-root", "x",
                    "--out-dir", "x"]
        rc_m = v02m.main()
    finally:
        v02m.build_all = orig_m
        sys.argv = argv_keep
    _check(rc_m == 2,
           f"v02_manifests.main() must convert an unexpected exception "
           f"to exit 2 (got {rc_m}); v0.1 had NO boundary at all")
    # V02F v0.2: defined-null decomposition shares render an explicit
    # "undefined" marker -- no bare TypeError, no skipped population.
    nullfacts = {"schema": "seqref-v02-facts/1",
                 "secondary_monitoring": {"d2b_decomposition": {
                     "train": {"base_share_pct": None,
                               "logdet_share_pct": None,
                               "null_reason": "mean endpoint NLL change "
                                              "is exactly 0.0"},
                     "holdout": {"base_share_pct": 75.0,
                                 "logdet_share_pct": 25.0,
                                 "null_reason": None}}}}
    v02f.plot_decomposition(nullfacts, tmp)
    _check((tmp / "v02_decomposition.png").exists(),
           "defined-null decomposition shares must still render "
           "v02_decomposition.png")
    mixed = {"schema": "seqref-v02-facts/1",
             "secondary_monitoring": {"d2b_decomposition": {
                 "train": {"base_share_pct": None,
                           "logdet_share_pct": 25.0},
                 "holdout": {"base_share_pct": 75.0,
                             "logdet_share_pct": 25.0}}}}
    _expect_error(lambda: v02f.plot_decomposition(mixed, tmp),
                  "FACTS_SHAPE_MISMATCH", "SEQREF-V02F")
    # V02F v0.3 boundary proof (tamper fixture: load_facts patched to
    # raise); main() must log and return exit 2.
    orig_lf = v02f.load_facts
    try:
        v02f.load_facts = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("fixture-induced unexpected failure"))
        sys.argv = ["v02_plots.py", "--facts", "x", "--out-dir", "x"]
        rc_f = v02f.main()
    finally:
        v02f.load_facts = orig_lf
        sys.argv = argv_keep
    _check(rc_f == 2,
           f"v02_plots.main() must convert an unexpected exception to "
           f"exit 2 (got {rc_f})")
    # ------------------------------------------------------------------
    # v0.4: every remaining fixture-reachable branch (the mechanical
    # coverage universe is extracted in f10; the branches exercised
    # below are the reachable complement of the justified-deferred set).
    # ------------------------------------------------------------------
    env = v02e._env()
    # V02M dataset-import gate (tamper: the deployed module is masked
    # out of sys.modules, so the lazy import must fail)
    key_fdm = "seqref_mri.src.fastmri_data"
    kept_fdm = sys.modules.get(key_fdm)
    sys.modules[key_fdm] = None
    try:
        _expect_error(lambda: v02m._dataset("x", "train", "eval"),
                      "DATASET_IMPORT_FAILED", "SEQREF-V02M")
    finally:
        if kept_fdm is None:
            sys.modules.pop(key_fdm, None)
        else:
            sys.modules[key_fdm] = kept_fdm
    # V02E total-import gate (same tamper on torch)
    kept_torch = sys.modules.get("torch")
    sys.modules["torch"] = None
    try:
        _expect_error(v02e._env, "ENV_IMPORT_FAILED", "SEQREF-V02E")
    finally:
        sys.modules["torch"] = kept_torch
    # V02E preflight consumption gates
    pf_bad = tmp / "pf_bad_schema.json"
    pf_bad.write_text(json.dumps({"schema": "wrong/9"}))
    _expect_error(lambda: v02e._load_preflight(str(pf_bad)),
                  "PREFLIGHT_SCHEMA_INVALID", "SEQREF-V02E")
    pf_over = tmp / "pf_over_ceiling.json"
    pf_over.write_text(json.dumps({
        "schema": "seqref-v02-preflight/1",
        "projected_training_s": 999999.0,
        "projected_training_h": 999999.0 / 3600.0,
        "ceiling_s": 172800.0, "projected_endpoint_eval_s": 1.0,
        "projected_pm_bank_s": 1.0, "peak_memory_bytes": 1024}))
    _expect_error(lambda: v02e._load_preflight(str(pf_over)),
                  "PREFLIGHT_PROJECTION_INVALID", "SEQREF-V02E")
    # V02E manifest kind gate (valid sidecar, consistent internal hash,
    # wrong kind)
    badkind = {"schema": "seqref-v02-manifest/1", "kind": "holdout",
               "entries": []}
    badkind["manifest_sha256"] = v02m.manifest_sha256(badkind)
    _write_with_sidecar(tmp / "v02_epochK_manifest.json", badkind,
                        tamper=False)
    _expect_error(lambda: v02e._load_manifest(
        tmp, "v02_epochK_manifest.json", "train_epoch"),
        "MANIFEST_SCHEMA_INVALID", "SEQREF-V02E")
    # V02E checkpoint loader chain (the existence gate is proven above)
    ck_a = tmp / "ck_a.pt"
    ck_a.write_bytes(b"fixture-bytes")
    _expect_error(lambda: v02e._load_checkpoint(
        None, tmp, {"step": 0, "file": "ck_a.pt", "state_sha256": "a",
                    "file_sha256": "0" * 64}, 1.0),
        "CHECKPOINT_FILE_MISMATCH", "SEQREF-V02E")
    ck_b = tmp / "ck_b.pt"
    ck_b.write_bytes(b"not a torch checkpoint")
    _expect_error(lambda: v02e._load_checkpoint(
        env, tmp, {"step": 0, "file": "ck_b.pt", "state_sha256": "a",
                   "file_sha256": v02e._file_sha256(ck_b)}, 1.0),
        "CHECKPOINT_UNREADABLE", "SEQREF-V02E")
    ck_c = tmp / "ck_c.pt"
    env.torch.save({"model": {}, "step": 999, "abbr": "SEQREF-V02T",
                    "version": "fixture"}, ck_c)
    _expect_error(lambda: v02e._load_checkpoint(
        env, tmp, {"step": 0, "file": "ck_c.pt", "state_sha256": "a",
                   "file_sha256": v02e._file_sha256(ck_c)}, 1.0),
        "CHECKPOINT_PROVENANCE_MISMATCH", "SEQREF-V02E")
    ck_d = tmp / "ck_d.pt"
    env.torch.save({"model": {}, "step": 0, "abbr": "SEQREF-V02T",
                    "version": "fixture"}, ck_d)
    _expect_error(lambda: v02e._load_checkpoint(
        env, tmp, {"step": 0, "file": "ck_d.pt", "state_sha256": "a",
                   "file_sha256": v02e._file_sha256(ck_d)},
        env.ffr.SPLINE_B),
        "CHECKPOINT_STATE_LOAD_FAILURE", "SEQREF-V02E")
    ck_e = tmp / "ck_e.pt"
    env.torch.save({"model": env.ffr.build_model().state_dict(),
                    "step": 0, "abbr": "SEQREF-V02T",
                    "version": "fixture"}, ck_e)
    _expect_error(lambda: v02e._load_checkpoint(
        env, tmp, {"step": 0, "file": "ck_e.pt",
                   "state_sha256": "0" * 64,
                   "file_sha256": v02e._file_sha256(ck_e)},
        env.ffr.SPLINE_B), "STATE_HASH_MISMATCH", "SEQREF-V02E")
    # V02E pure population gates
    _expect_error(lambda: v02e._d3_condition_summary(
        None, {"condition": "C1", "nll_batch": 0.0,
               "per_slice": [{"nll": 1.0}]},
        {"per_slice": [{"nll": 1.0}, {"nll": 2.0}]}),
        "D3_POPULATION_MISMATCH", "SEQREF-V02E")
    _expect_error(lambda: v02e._volume_groups(
        [{"file": "a.h5", "slice_index": 0}], np.array([1.0, 2.0])),
        "POPULATION_MISMATCH", "SEQREF-V02E")
    # V02E StageError-conversion wrappers (seam tamper: the wrapped
    # production function raises StageError; the wrapper must convert
    # it to the module's typed ERROR, never let it escape)
    def _stage_boom(*a, **k):
        raise env.StageError("FIXTURE_TAMPER", "deliberate seam tamper")
    orig_enc = env.d2b._encode_slice
    try:
        env.d2b._encode_slice = _stage_boom
        _expect_error(lambda: v02e._encode(env, None, {}, "fixture"),
                      "EVAL_ENCODE_FAILURE", "SEQREF-V02E")
    finally:
        env.d2b._encode_slice = orig_enc
    orig_dec = env.tg._decode_z
    try:
        env.tg._decode_z = _stage_boom
        _expect_error(lambda: v02e._z0_metrics(
            env, None,
            {"u_true_energy": 1.0,
             "identity": {"file": "f.h5", "slice_index": 0}},
            "fixture"), "EVAL_Z0_FAILURE", "SEQREF-V02E")
    finally:
        env.tg._decode_z = orig_dec
    orig_mc = env.d3._measure_condition
    try:
        env.d3._measure_condition = _stage_boom
        _expect_error(lambda: v02e._measure_condition(
            env, None, [], None, None, None, "fixture"),
            "D3_MEASURE_FAILURE", "SEQREF-V02E")
    finally:
        env.d3._measure_condition = orig_mc
    # V02T run()-level manifest gates (they fire before any parent or
    # dataset access; the cfg beyond manifest_dir/out_root is unread)
    _expect_error(lambda: v02t.run({
        "data_root": "x", "manifest_dir": str(tmp / "no_such_dir"),
        "p4_stats2": "x", "implb_facts": "x",
        "out_root": str(tmp / "run_t1")}),
        "MANIFEST_MISSING", "SEQREF-V02T")
    tdir1 = tmp / "tman1"
    tdir1.mkdir()
    _write_with_sidecar(tdir1 / "v02_epoch0_manifest.json",
                        {"schema": "wrong/9", "kind": "train_epoch"},
                        tamper=False)
    _expect_error(lambda: v02t.run({
        "data_root": "x", "manifest_dir": str(tdir1),
        "p4_stats2": "x", "implb_facts": "x",
        "out_root": str(tmp / "run_t2")}),
        "MANIFEST_SCHEMA_INVALID", "SEQREF-V02T")
    tdir2 = tmp / "tman2"
    tdir2.mkdir()
    for ep in range(3):
        _write_with_sidecar(
            tdir2 / f"v02_epoch{ep}_manifest.json",
            {"schema": "seqref-v02-manifest/1", "kind": "train_epoch",
             "epoch": ep, "n_slices": 1, "entries": [], "batches": []},
            tamper=False)
    _expect_error(lambda: v02t.run({
        "data_root": "x", "manifest_dir": str(tdir2),
        "p4_stats2": "x", "implb_facts": "x",
        "out_root": str(tmp / "run_t3")}),
        "MANIFEST_CONTENT_MISMATCH", "SEQREF-V02T")
    # V02T mask structural gates
    zeros_mask = torch.zeros(96, dtype=torch.bool)
    _expect_error(lambda: v02t.derive_cmap_from_mask(zeros_mask),
                  "MASK_ACQUIRED_COUNT_UNEXPECTED", "SEQREF-V02T")
    nocentre = torch.zeros(96, dtype=torch.bool)
    nocentre[torch.tensor(list(range(0, 96, 4)))] = True
    _expect_error(lambda: v02t.derive_cmap_from_mask(nocentre),
                  "MASK_CENTRE_NOT_ACQUIRED", "SEQREF-V02T")
    # V02F value/consistency/checkpoint gates
    _expect_error(lambda: v02f.plot_per_slice_delta(
        {"endpoint_measurements": {"train": {
            "nll_step0": [float("nan")], "nll_final": [1.0],
            "mean_gain_per_dim": 0.0}}}, tmp),
        "FACTS_VALUE_INVALID", "SEQREF-V02F")
    _expect_error(lambda: v02f.plot_per_slice_delta(
        {"endpoint_measurements": {"train": {
            "nll_step0": [2.0], "nll_final": [1.0],
            "mean_gain_per_dim": 999.0}}}, tmp),
        "FACTS_CONSISTENCY_FAILED", "SEQREF-V02F")
    _expect_error(lambda: v02f.plot_holdout_trajectory(
        {"endpoint_measurements": {"holdout": {
            "per_checkpoint": [{"step": 0, "z0_psnr_mean": 1.0,
                                "z0_nmse_u_mean": 1.0}]}}}, tmp),
        "FACTS_CHECKPOINT_MISMATCH", "SEQREF-V02F")
    # V02E run()-level content/validity gates over a fabricated frozen-
    # input set (every gate fires BEFORE any dataset or parent access;
    # identities are synthetic and no dataset is ever touched)
    fab = Path(tempfile.mkdtemp(prefix="v02s_fab_"))
    cfg = _fab_manifest_set(fab)
    t0doc = json.loads((fab / "v02_epoch0_manifest.json").read_text())
    t0doc["n_slices"] = 1
    t0doc["manifest_sha256"] = v02m.manifest_sha256(
        {k: v for k, v in t0doc.items() if k != "manifest_sha256"})
    _write_with_sidecar(fab / "v02_epoch0_manifest.json", t0doc,
                        tamper=False)
    _expect_error(lambda: v02e.run(dict(cfg)),
                  "MANIFEST_CONTENT_MISMATCH", "SEQREF-V02E")
    cfg = _fab_manifest_set(fab)
    d3doc = json.loads(
        (fab / "v02_d3_monitor_manifest.json").read_text())
    (d3doc["entries"][0]["holdout_position"],
     d3doc["entries"][1]["holdout_position"]) = (
        d3doc["entries"][1]["holdout_position"],
        d3doc["entries"][0]["holdout_position"])
    d3doc["manifest_sha256"] = v02m.manifest_sha256(
        {k: v for k, v in d3doc.items() if k != "manifest_sha256"})
    _write_with_sidecar(fab / "v02_d3_monitor_manifest.json", d3doc,
                        tamper=False)
    _expect_error(lambda: v02e.run(dict(cfg)),
                  "D3_DRAW_MISMATCH", "SEQREF-V02E")
    cfg = _fab_manifest_set(fab)
    d3doc = json.loads(
        (fab / "v02_d3_monitor_manifest.json").read_text())
    d3doc["entries"][0]["file"] = "fab_WRONG.h5"
    d3doc["manifest_sha256"] = v02m.manifest_sha256(
        {k: v for k, v in d3doc.items() if k != "manifest_sha256"})
    _write_with_sidecar(fab / "v02_d3_monitor_manifest.json", d3doc,
                        tamper=False)
    _expect_error(lambda: v02e.run(dict(cfg)),
                  "D3_SUBSET_NOT_IN_HOLDOUT", "SEQREF-V02E")
    cfg = _fab_manifest_set(fab)
    cfg["p4_stats2"] = str(fab / "no_such_parent.json")
    _expect_error(lambda: v02e.run(dict(cfg)),
                  "PARENT_VERIFICATION_FAILED", "SEQREF-V02E")
    return {"error_paths_exercised": sorted(f"{a}:{c}"
                                            for a, c in _OBSERVED),
            "boundary_catches_proven": ["SEQREF-V02M", "SEQREF-V02T",
                                        "SEQREF-V02P", "SEQREF-V02F"],
            "null_share_render_proven": True,
            "run_level_gates_proven": True,
            "seam_tamper_wrapper_proofs": ["EVAL_ENCODE_FAILURE",
                                           "EVAL_Z0_FAILURE",
                                           "D3_MEASURE_FAILURE"]}


# ---------------------------------------------------------------------------
# Row 10 -- no silent pass: registry/coverage totality (harness meta-row).
# ---------------------------------------------------------------------------

def f10_no_silent_pass(results: list) -> dict:
    _check(len(results) == 9,
           f"{len(results)} fixtures returned evidence; the registry "
           f"declares 9 -- an unexecuted fixture is an ERROR, not a skip")
    for res in results:
        _check(isinstance(res.get("evidence"), dict)
               and len(res["evidence"]) > 0,
               f"fixture {res.get('name')} returned no evidence; a "
               f"fixture without assert-produced evidence is an ERROR")
    extracted = _extract_codes()
    coverage = {}
    for abbr, codes in extracted.items():
        observed = {c for a, c in _OBSERVED if a == abbr}
        deferred = DEFERRED_JUSTIFICATIONS.get(abbr, {})
        unknown = sorted(observed - set(codes))
        _check(not unknown,
               f"{abbr}: fixtures observed codes that mechanical "
               f"extraction does not find in the module source: "
               f"{unknown}; a stale or mistyped fixture claim is an "
               f"ERROR, not coverage")
        stale = sorted(set(deferred) - set(codes))
        _check(not stale,
               f"{abbr}: deferred codes no longer present in the "
               f"module source: {stale}; a deferral is tied to a live "
               f"guard, never to a removed one")
        empty = sorted(c for c, j in deferred.items()
                       if not str(j).strip())
        _check(not empty,
               f"{abbr}: deferred codes without a written "
               f"justification: {empty}; an unjustified deferral is a "
               f"silent pass (SS7 row 10)")
        missing = sorted(set(codes) - observed - set(deferred))
        _check(not missing,
               f"{abbr}: ERROR paths neither exercised nor deferred: "
               f"{missing}; per SS7 row 9 every ERROR path is "
               f"exercised deliberately or its deferral justified -- "
               f"coverage is total or the suite is an ERROR")
        coverage[abbr] = {
            "extracted": codes,
            "observed": sorted(observed & set(codes)),
            "deferred": {c: deferred[c]
                         for c in sorted(set(deferred) & set(codes))}}
    stray = sorted({a for a, _c in _OBSERVED} - set(extracted))
    _check(not stray,
           f"fixtures recorded observations for modules outside the "
           f"coverage universe: {stray}")
    # Supplementary structural checks (never a substitute for the
    # exercised branches proven above).
    for abbr, mod in (("SEQREF-V02M", v02m), ("SEQREF-V02E", v02e)):
        src = Path(mod.__file__).read_text()
        _check("logger.error" in src and "_fail(" in src,
               f"{abbr} source lost the logger.error + typed-raise "
               f"convention")
    return {"coverage_ok": True,
            "coverage_mode": "mechanical extraction over module "
                             "sources; per-module totality "
                             "extracted == observed + deferred, every "
                             "deferral justified in writing",
            "coverage": coverage,
            "observed": sorted(f"{a}:{c}" for a, c in _OBSERVED),
            "harness_codes_out_of_scope": {
                "SEQREF-V02S": ["SELFTEST_FIXTURE_FAILED",
                                "ENV_IMPORT_FAILED",
                                "PUBLICATION_FAILURE"]}}


FIXTURES = [
    ("f01_manifest_determinism", 1, f01_manifest_determinism),
    ("f02_partition_exactness", 2, f02_partition_exactness),
    ("f03_checkpoint_schedule", 3, f03_checkpoint_schedule),
    ("f04_same_encode_invariant", 4, f04_same_encode_invariant),
    ("f05_r_guard", 5, f05_r_guard),
    ("f06_bootstrap_reproducibility", 6, f06_bootstrap_reproducibility),
    ("f07_d3_draw_reproducibility", 7, f07_d3_draw_reproducibility),
    ("f08_exposure_accounting", 8, f08_exposure_accounting),
    ("f09_error_paths", 9, f09_error_paths)]


def run_suite() -> dict:
    _OBSERVED.clear()
    results = []
    for name, row, fn in FIXTURES:
        t0 = time.time()
        evidence = fn()          # any V02Error propagates as suite ERROR
        results.append({"name": name, "matrix_row": int(row),
                        "result": "PASS",
                        "elapsed_s": time.time() - t0,
                        "evidence": evidence})
        logger.info("[%s] %s (row %d): PASS", __abbr__, name, row)
    evidence10 = f10_no_silent_pass(results)
    results.append({"name": "f10_no_silent_pass", "matrix_row": 10,
                    "result": "PASS", "elapsed_s": 0.0,
                    "evidence": evidence10})
    logger.info("[%s] %s (row %d): PASS", __abbr__, "f10_no_silent_pass",
                10)
    return {"schema": REPORT_SCHEMA,
            "script": f"{__abbr__} v{__version__}",
            "matrix": "V02PLAN v0.2 SS7 (ten rows)",
            "results": results,
            "coverage": evidence10,
            "golden_pins": "pinned 2026-08-22 against NumPy PCG64; a "
                           "generator or stream-order change fails the "
                           "pin, never silently re-pins",
            "taxonomy": "exit 0 = all fixtures PASS + coverage_ok; "
                        "exit 2 = ERROR; no exit 1, no skip (SS7 row 10)"}


def _publish(report: dict, out_dir: str) -> tuple:
    try:
        from seqref_mri.tdiag import _bootstrap  # noqa: F401 --
        # registered path bootstrap (same contract as SEQREF-V02E v0.2)
        from preflight_parents import (StageError, attach_semantic_hash,
                                       publish_stage)
    except ImportError as exc:
        _fail("ENV_IMPORT_FAILED",
              f"publication machinery not importable: {exc}")
    semantic = {k: v for k, v in report.items() if k != "run"}
    attach_semantic_hash(report, semantic)
    try:
        return publish_stage(report, out_dir, REPORT_PREFIX, __abbr__)
    except StageError as exc:
        _fail("PUBLICATION_FAILURE", f"{exc.error_code}: {exc.reason}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"{__abbr__} v{__version__} -- candidate v0.2 "
                    f"selftest (V02PLAN SS7 matrix, both invocation "
                    f"modes)")
    ap.add_argument("--mode", choices=["fixtures", "authoritative"],
                    default="fixtures")
    ap.add_argument("--out-dir", default=None,
                    help="required in authoritative mode")
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()
    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode="w"))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s "
                               "%(message)s")
    if args.mode == "authoritative" and not args.out_dir:
        logger.error("[%s] authoritative mode requires --out-dir",
                     __abbr__)
        return EXIT_ERROR
    try:
        report = run_suite()
        report["mode"] = args.mode
        report["run"] = {"utc": time.strftime("%Y%m%dT%H%M%S+0000",
                                              time.gmtime())}
        if args.mode == "authoritative":
            path, sha = _publish(report, args.out_dir)
            logger.info("[%s] authoritative selftest report published "
                        "%s sha256=%s", __abbr__, path, sha[:12])
        else:
            logger.info("[%s] fixtures mode: %d rows PASS, coverage_ok "
                        "= True; NOTHING published", __abbr__,
                        len(report["results"]))
    except v02m.V02Error:
        return EXIT_ERROR
    except Exception:  # noqa: BLE001 -- the registered boundary: no
        logger.exception("[%s] unexpected runtime failure", __abbr__)
        return EXIT_ERROR            # exception may pass silently / exit 1
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
