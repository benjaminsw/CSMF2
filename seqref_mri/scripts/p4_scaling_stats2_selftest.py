# =============================================================================
# SEQREF-P4S2T v0.1 -- scripts.p4_scaling_stats2_selftest
# LIFETIME: KEEP
# Purpose: self-test for SEQREF-P4S2 (P4 statistics layer, schema
#   seqref-p4-stats/2). Pure-core fixtures: no dataset, no h5, no kspace.
#   The suite pins the seed SERIALIZATION strings, the /1 dual-hash parent
#   pin, the Welford/ddof=0 estimation semantics, the pooled same-
#   population globals, the STRICT-< floor boundary (equality is NOT a
#   hit), the per-location candidate scale, the integer branch boundary
#   (422/423 at 8,448), the branch-selected applied affine pair, the
#   pre-vote validity gate, the C7 round-trip, the authoritative-only
#   parity helpers, the generator pin, the no-BLOCK taxonomy, and the
#   publication overwrite machinery.
# REGISTRY DISCIPLINE: EXPECTED_COUNTS is a STATIC count of this source,
#   re-derived per rewrite, never carried forward. The coverage audit fails
#   loudly on any mismatch: a green suite with a stale registry is
#   impossible by construction.
# CONVENTION: fixture failures are reported, never hidden; the suite exits
#   nonzero unless every check passes AND coverage matches the registry.
# Changelog
#   v0.1 (2026-08-09) Created with SEQREF-P4S2 v0.1 against the frozen /2
#     registration (EXEC §13 P4 /2 block, Concept D4 statistics-semantics
#     clarification, both 2026-08-09, before any /2 observation). Includes
#     one integration regression (test_main_integration_smoke) that drives
#     main() to a SMOKE PASS with traversal/provenance stubbed dataset-free,
#     pinning the main() wiring (dict-typed `consistency`, artefact shape,
#     branch record) that pure fixtures cannot reach.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

from preflight_io import file_sha256  # noqa: E402
from preflight_parents import StageError, publish_stage  # noqa: E402
from seqref_mri.scripts import p4_scaling_stats2 as p4s2  # noqa: E402
from seqref_mri.src import fastmri_data as fdm  # noqa: E402

SCRIPT_ID = "SEQREF-P4S2T"
SCRIPT_VERSION = "v0.1"
logger = logging.getLogger(SCRIPT_ID)

_RESULTS: list[tuple[str, str, bool, str]] = []
_CURRENT = ["<none>"]

CENTRE = p4s2.centre_columns(96)
ELIGIBLE = [c for c in range(96) if c not in CENTRE]


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def test_mask_seed_conventions() -> None:
    _CURRENT[0] = "test_mask_seed_conventions"
    check("mask_counts(96)==(8,24)", fdm.mask_counts(96) == (8, 24))
    check("centre==44..51", CENTRE == frozenset(range(44, 52)))
    m1 = fdm.make_cartesian_mask(96, 12345)
    check("mask deterministic + exact count",
          bool((m1 == fdm.make_cartesian_mask(96, 12345)).all())
          and int(m1.sum()) == 24)
    train = fdm.canonical_mask_seed(fdm.TRAIN_BASE_SEED, "f.h5", 3, epoch=0)
    want = int.from_bytes(
        hashlib.sha256(b"20261000|0|f.h5|3").digest()[:8], "big")
    check("train serialization pinned base|epoch|relpath|slice",
          train == want)
    ev = fdm.canonical_mask_seed(fdm.EVAL_BASE_SEED, "f.h5", 3)
    want_ev = int.from_bytes(
        hashlib.sha256(b"20261001|f.h5|3").digest()[:8], "big")
    check("eval serialization pinned base|relpath|slice", ev == want_ev)


def test_parent_loader() -> None:
    _CURRENT[0] = "test_parent_loader"
    art = os.path.join(_REPO, "seqref_mri", "results", "_diag", "p4",
                       "scaling_stats.json")
    if not os.path.isfile(art):
        check("authoritative /1 artefact present in repo", False,
              f"missing {art}")
        return
    parent = p4s2.load_p4s1_parent(art)
    check("dual-hash pin passes on the real /1 artefact", True)
    check("eligible set inherited: 88 non-centre columns",
          parent["eligible_columns"] == ELIGIBLE)
    check("parent tables complete",
          len(parent["files"]) == 973
          and parent["parent_n_free_raw"][0] == 28545
          and len(parent["parent_w"][0]) == 973)
    expect_error("missing parent -> PARENT_NOT_FOUND", "PARENT_NOT_FOUND",
                 p4s2.load_p4s1_parent, art + ".absent")
    with tempfile.TemporaryDirectory() as tmp:
        forged = os.path.join(tmp, "scaling_stats.json")
        with open(art, encoding="utf-8") as fh:
            payload = json.load(fh)
        payload["columns"][0]["n_free_raw"] += 1
        with open(forged, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        expect_error("tampered parent -> PARENT_FILE_HASH_MISMATCH",
                     "PARENT_FILE_HASH_MISMATCH",
                     p4s2.load_p4s1_parent, forged)


def _acc_with(values_by_col: dict, ch_both: bool = True) -> dict:
    """Craft an accumulator: values_by_col[col] = (re_values, im_values)
    applied to EVERY row of the column (as the 1-D column mask implies)."""
    acc = p4s2.new_accumulator(96)
    for col, (re_v, im_v) in values_by_col.items():
        v = np.asarray(re_v) + 1j * np.asarray(im_v)
        for k in range(v.shape[0]):
            block = np.tile(v[k], (96, 1)).astype(np.complex128)
            p4s2.accumulate_observations(acc, (col,), block)
    return acc


def test_welford_accumulation() -> None:
    _CURRENT[0] = "test_welford_accumulation"
    acc = _acc_with({0: ([1.0, 2.0, 3.0], [10.0, 10.0, 10.0])})
    check("count per location/channel", acc["count"][5, 0, 0] == 3
          and acc["count"][95, 0, 1] == 3)
    check("mean exact", acc["mean"][5, 0, 0] == 2.0
          and acc["mean"][5, 0, 1] == 10.0)
    check("M2 exact (ddof=0 realised later)", acc["M2"][5, 0, 0] == 2.0
          and acc["M2"][5, 0, 1] == 0.0)
    check("untouched locations stay zero",
          acc["count"][:, 1, :].sum() == 0
          and acc["count"][0, 44, 0] == 0)
    rng = np.random.default_rng(7)
    re = rng.normal(0.3, 0.7, (96, 40))
    im = rng.normal(-0.2, 1.3, (96, 40))
    acc2 = p4s2.new_accumulator(96)
    for k in range(40):
        p4s2.accumulate_observations(
            acc2, (3,), (re[:, k] + 1j * im[:, k]).reshape(96, 1)
            .astype(np.complex128))
    check("random-data mean matches numpy",
          abs(acc2["mean"][10, 3, 0] - re[10].mean()) < 1e-15
          and abs(acc2["mean"][10, 3, 1] - im[10].mean()) < 1e-15)
    check("random-data population std matches numpy ddof=0",
          abs(np.sqrt(acc2["M2"][10, 3, 0] / 40) - re[10].std()) < 1e-12
          and abs(np.sqrt(acc2["M2"][10, 3, 1] / 40) - im[10].std()) < 1e-12)
    expect_error("shape mismatch", "OBSERVATION_BLOCK_SHAPE",
                 p4s2.accumulate_observations, p4s2.new_accumulator(96),
                 (0,), np.ones((95, 1), dtype=np.complex128))
    expect_error("non-finite observation", "NON_FINITE_OBSERVATION",
                 p4s2.accumulate_observations, p4s2.new_accumulator(96),
                 (0,), np.full((96, 1), np.nan + 0j))


def test_pooled_moments() -> None:
    _CURRENT[0] = "test_pooled_moments"
    rng = np.random.default_rng(11)
    cols = (0, 5, 27)
    re = rng.normal(0.5, 0.4, (96, 3, 30))
    im = rng.normal(-0.5, 2.0, (96, 3, 30))
    acc = p4s2.new_accumulator(96)
    for k in range(30):
        p4s2.accumulate_observations(
            acc, cols, (re[:, :, k] + 1j * im[:, :, k])
            .astype(np.complex128))
    mg, sg, N = p4s2.pooled_channel_moments(acc["count"], acc["mean"],
                                            acc["M2"], list(cols), 0)
    flat = re.reshape(-1)
    check("pooled mean == brute force", abs(mg - flat.mean()) < 1e-13)
    check("pooled sigma == brute force ddof=0",
          abs(sg - flat.std()) < 1e-13)
    check("one population for mean and std", N == flat.size)
    expect_error("empty channel population", "EMPTY_CHANNEL_POPULATION",
                 p4s2.pooled_channel_moments, acc["count"], acc["mean"],
                 acc["M2"], [1, 2], 0)


def test_floor_boundary() -> None:
    _CURRENT[0] = "test_floor_boundary"
    floor = np.array([0.005, 0.02])
    raw = np.zeros((96, 2, 2))
    raw[:, :, 0] = 0.005          # EXACTLY floor_re
    raw[:, :, 1] = 2.0            # far above floor_im
    check("raw_std == floor is NOT a hit (strict <)",
          not p4s2.floor_hit_mask(raw, floor).any())
    raw_below = raw.copy()
    raw_below[0, 0, 0] = np.nextafter(0.005, 0.0)
    hits = p4s2.floor_hit_mask(raw_below, floor)
    check("raw_std one ulp below floor IS a hit",
          bool(hits[0, 0]) and int(hits.sum()) == 1)
    raw_above = raw.copy()
    raw_above[0, 0, 0] = np.nextafter(0.005, 1.0)
    check("raw_std one ulp above floor is NOT a hit",
          not p4s2.floor_hit_mask(raw_above, floor).any())
    check("EITHER channel triggers the location hit",
          bool(p4s2.floor_hit_mask(raw_below, floor)[0, 0]))
    pls = p4s2.per_location_scales(raw_below, floor)
    check("per_location_scale == floor when raw < floor",
          pls[0, 0, 0] == floor[0])
    check("per_location_scale == raw_std when raw >= floor",
          pls[0, 0, 1] == 2.0 and pls[1, 0, 0] == 0.005)


def test_branch_decision() -> None:
    _CURRENT[0] = "test_branch_decision"
    d0 = p4s2.branch_decision(0, 8448)
    check("0 hits -> PER-LOCATION", d0["selected"] == "PER-LOCATION"
          and d0["lhs_20x_hits"] == 0)
    d422 = p4s2.branch_decision(422, 8448)
    check("422 hits at 8448 -> PER-LOCATION (8440 <= 8448)",
          d422["selected"] == "PER-LOCATION"
          and d422["lhs_20x_hits"] == 8440)
    d423 = p4s2.branch_decision(423, 8448)
    check("423 hits at 8448 -> GLOBAL PER-CHANNEL (8460 > 8448)",
          d423["selected"] == "GLOBAL PER-CHANNEL"
          and d423["lhs_20x_hits"] == 8460)
    d_all = p4s2.branch_decision(8448, 8448)
    check("all hits -> GLOBAL PER-CHANNEL",
          d_all["selected"] == "GLOBAL PER-CHANNEL")
    check("integer form recorded, no float division",
          d422["rhs_n_eligible"] == 8448
          and isinstance(d422["lhs_20x_hits"], int))
    expect_error("negative hits", "BRANCH_OPERANDS_INVALID",
                 p4s2.branch_decision, -1, 8448)
    expect_error("zero eligible", "BRANCH_OPERANDS_INVALID",
                 p4s2.branch_decision, 0, 0)


def test_c7_roundtrip() -> None:
    _CURRENT[0] = "test_c7_roundtrip"
    rng = np.random.default_rng(13)
    m = rng.normal(0.0, 3.0, (500, 2))
    s = np.abs(rng.normal(1.0, 0.5, (500, 2))) + 1e-3
    rec = p4s2.c7_roundtrip(m, s)
    check("sane pair passes", rec["ok"] and rec["max_rel_err"] <= 1e-12)
    check("metric + tolerance pinned",
          rec["metric"] == "max|x_rt - x| / max(1, max|x|)"
          and rec["tolerance"] == p4s2.C7_RTOL == 1e-12)
    check("probes are published-parameter only",
          rec["probes"] == ["applied_mean", "applied_mean + applied_scale",
                            "applied_mean - 2*applied_scale"])
    bad_scale = s.copy()
    bad_scale[3, 1] = 0.0
    expect_error("non-positive scale", "C7_OPERANDS_INVALID",
                 p4s2.c7_roundtrip, m, bad_scale)
    bad_mean = m.copy()
    bad_mean[0, 0] = np.nan
    expect_error("non-finite mean", "C7_OPERANDS_INVALID",
                 p4s2.c7_roundtrip, bad_mean, s)
    expect_error("shape mismatch", "C7_OPERANDS_INVALID",
                 p4s2.c7_roundtrip, m[:10], s)


def _finalize_fixture(hit: bool) -> dict:
    """Two eligible columns, fully observed. hit=True: column 0 constant
    (raw_std 0 -> floor hit everywhere), column 1 varying."""
    rng = np.random.default_rng(17)
    acc = p4s2.new_accumulator(96)
    varying_re = rng.normal(0.4, 0.6, 50)
    varying_im = rng.normal(-0.4, 0.8, 50)
    for k in range(50):
        block1 = np.full((96, 1), varying_re[k] + 1j * varying_im[k],
                         dtype=np.complex128)
        p4s2.accumulate_observations(acc, (1,), block1)
        if hit:
            block0 = np.full((96, 1), 0.1 + 0.2j, dtype=np.complex128)
        else:
            block0 = np.full((96, 1),
                             varying_re[k] - 0.1 + 1j * (varying_im[k] + 0.1),
                             dtype=np.complex128)
        p4s2.accumulate_observations(acc, (0,), block0)
    return p4s2.finalize_statistics(acc, [0, 1], True)


def test_finalize_per_location_branch() -> None:
    _CURRENT[0] = "test_finalize_per_location_branch"
    st = _finalize_fixture(hit=False)
    check("full coverage gated and met",
          st["full_coverage"] and st["n_locations_observed"] == 192)
    check("0 floor hits -> PER-LOCATION", st["n_floor_hits"] == 0
          and st["decision"]["selected"] == "PER-LOCATION")
    check("branch denominator is the inherited eligible location count",
          st["decision"]["n_eligible"] == 192 == 96 * 2)
    sel = st["observed"]
    check("applied pair == per-location pair under PER-LOCATION",
          np.allclose(st["applied_mean"][sel], st["loc_mean"][sel])
          and np.allclose(st["applied_scale"][sel],
                          st["per_location_scale"][sel]))
    check("channel globals recorded regardless of branch",
          st["channels"]["re"]["sigma_global"] > 0
          and st["channels"]["im"]["floor"]
          == 0.01 * st["channels"]["im"]["sigma_global"])


def test_finalize_global_branch() -> None:
    _CURRENT[0] = "test_finalize_global_branch"
    st = _finalize_fixture(hit=True)
    check("constant column hits everywhere",
          st["n_floor_hits"] == 96)
    check("96 hits at 192 -> GLOBAL PER-CHANNEL (1920 > 192)",
          st["decision"]["selected"] == "GLOBAL PER-CHANNEL")
    sel = st["observed"]
    am, as_ = st["applied_mean"][sel], st["applied_scale"][sel]
    check("applied pair broadcast from channel globals",
          np.allclose(am[:, 0], st["channels"]["re"]["mean_global"])
          and np.allclose(as_[:, 1], st["channels"]["im"]["sigma_global"]))
    check("unselected per-location scales remain recorded, never applied",
          np.isfinite(st["per_location_scale"][sel]).all()
          and not np.allclose(as_[:, 0],
                              st["per_location_scale"][sel][:, 0]))


def test_pre_vote_validity() -> None:
    _CURRENT[0] = "test_pre_vote_validity"
    acc = p4s2.new_accumulator(96)
    for k in range(10):
        p4s2.accumulate_observations(
            acc, (0,), np.full((96, 1), 0.3 + 0.7j, dtype=np.complex128))
    expect_error("zero sigma_global channel -> PRE_VOTE_VALIDITY_FAILURE",
                 "PRE_VOTE_VALIDITY_FAILURE",
                 p4s2.finalize_statistics, acc, [0], False)


def test_finalize_coverage_gate() -> None:
    _CURRENT[0] = "test_finalize_coverage_gate"
    acc = p4s2.new_accumulator(96)
    rng = np.random.default_rng(19)
    for k in range(10):
        p4s2.accumulate_observations(
            acc, (0,), (rng.normal(0, 1, (96, 1))
                        + 1j * rng.normal(0, 1, (96, 1)))
            .astype(np.complex128))
    expect_error("authoritative partial coverage",
                 "ELIGIBLE_LOCATION_UNOBSERVED",
                 p4s2.finalize_statistics, acc, [0, 1], True)
    st = p4s2.finalize_statistics(acc, [0, 1], False)
    check("smoke partial coverage recorded, not gated",
          not st["full_coverage"] and st["n_locations_observed"] == 96
          and st["decision"]["n_eligible"] == 96)


def test_parity_helpers() -> None:
    _CURRENT[0] = "test_parity_helpers"
    parent = {"parent_w": {0: {"a": 2, "b": 1}, 1: {"a": 3}},
              "parent_n_free_raw": {0: 3, 1: 3}}
    own = {0: {"a": 2, "b": 1}, 1: {"a": 3}}
    rec = p4s2.compare_weights_vs_parent(own, parent)
    check("identical sparse tables pass", rec["identical"]
          and rec["columns_compared"] == 2)
    expect_error("weight mismatch", "PARENT_WEIGHTS_MISMATCH",
                 p4s2.compare_weights_vs_parent, {0: {"a": 2, "b": 2},
                                                  1: {"a": 3}}, parent)
    expect_error("missing column", "PARENT_WEIGHTS_MISMATCH",
                 p4s2.compare_weights_vs_parent, {0: {"a": 2, "b": 1}},
                 parent)
    acc = p4s2.new_accumulator(96)
    for k in range(3):
        p4s2.accumulate_observations(
            acc, (0,), np.full((96, 1), 1.0 + 0.5j, dtype=np.complex128))
    rec2 = p4s2.transpose_invariant(acc, parent, [0])
    check("transpose invariant holds", rec2["holds"])
    expect_error("transpose violated", "TRANSPOSE_INVARIANT_VIOLATED",
                 p4s2.transpose_invariant, acc, parent, [1])
    rec3 = p4s2.own_count_weight_consistency(acc, {0: {"a": 3}}, [0])
    check("own count/weight consistency holds", rec3["holds"])
    expect_error("own count/weight mismatch", "OWN_COUNT_WEIGHT_MISMATCH",
                 p4s2.own_count_weight_consistency, acc, {0: {"a": 2}}, [0])


def test_generator_pin() -> None:
    _CURRENT[0] = "test_generator_pin"
    good = {"resolved": True,
            "mask_seed_source_sha256": p4s2.GENERATOR_SOURCE_SHA256}
    p4s2.enforce_generator_pin(good)
    check("matching pin passes", True)
    expect_error("wrong generator hash -> ERROR", "GENERATOR_HASH_MISMATCH",
                 p4s2.enforce_generator_pin,
                 {"resolved": True, "mask_seed_source_sha256": "0" * 64})
    expect_error("unbound provenance -> ERROR", "GENERATOR_HASH_MISMATCH",
                 p4s2.enforce_generator_pin, {"resolved": False})


def test_no_block_path() -> None:
    _CURRENT[0] = "test_no_block_path"
    with open(p4s2.__file__, encoding="utf-8") as fh:
        src = fh.read()
    code_only = "\n".join(l for l in src.splitlines()
                          if not l.lstrip().startswith("#"))
    check("no StageBlock anywhere in stage code",
          "StageBlock" not in code_only)
    check("no BLOCK handler and EXIT_BLOCK never imported",
          "except StageBlock" not in code_only
          and "EXIT_BLOCK" not in code_only)


# ---------------------------------------------------------------------------
# Facts + publication fixtures (stub parent aligned with the 2-column
# statistics fixture; the REAL parent loader is pinned separately above)
# ---------------------------------------------------------------------------

def _stub_parent() -> dict:
    return {"path": "/x/scaling_stats.json",
            "file_sha256": p4s2.P4S1_FILE_SHA256,
            "semantic_sha256": p4s2.P4S1_SEMANTIC_SHA256,
            "sidecar_present": True,
            "files": ["a.h5"], "eligible_columns": [0, 1],
            "parent_w": {}, "parent_n_free_raw": {},
            "grandparents": {"p0_p0s": None, "p1_p2": None}}


def _build(smoke=10, stats=None):
    stats = stats if stats is not None else _finalize_fixture(hit=False)
    sel = stats["observed"]
    c7 = p4s2.c7_roundtrip(stats["applied_mean"][sel],
                           stats["applied_scale"][sel])
    trav = {"per_slice": {"rows": smoke or 34742, "seed_agreement": True,
                          "mask_identity": True, "acquired_count": True},
            "files": ["a.h5"]}
    population = p4s2.check_population(trav, False)
    consistency = {"per_slice": trav["per_slice"],
                   "parity_vs_p4_s1": {"evaluated": False,
                                       "reason": "fixture smoke"}}
    return p4s2._build_facts(_stub_parent(), trav, stats, c7, population,
                             consistency, "PASS", "fixture", _REPO,
                             os.path.abspath(p4s2.__file__), [], 0.0, smoke,
                             {"resolved": True}, {"dataset_class": "fixture"})


def test_facts_builder() -> None:
    _CURRENT[0] = "test_facts_builder"
    facts = _build()
    check("schema /2", facts["schema"] == "seqref-p4-stats/2")
    scope = facts["schema_scope"]
    check("scope: statistics layer, /1 frozen parent",
          scope["parent_schema"] == "seqref-p4-stats/1"
          and "per_location_scaling_statistics" in scope["covers"]
          and "branch_vote" in scope["covers"])
    inh = facts["inheritance"]
    check("inheritance pins == registered /1 hashes",
          inh["parent_file_sha256"] == p4s2.P4S1_FILE_SHA256
          and inh["parent_semantic_sha256"] == p4s2.P4S1_SEMANTIC_SHA256
          and inh["n_eligible_columns"] == 2)
    th = facts["thresholds"]
    check("registered constants carried",
          th["FLOOR_FACTOR"] == 1e-2 and th["BRANCH_DENOM"] == 20
          and th["C7_RTOL"] == 1e-12 and th["DDOF"] == 0)
    check("branch smoke-scale flagged, integer form recorded",
          facts["branch"]["smoke_scale"] is True
          and facts["branch"]["lhs_20x_hits"] == 0
          and facts["branch"]["n_eligible"] == 192)
    locs = facts["locations"]
    check("locations: observed only, row-major, complete records",
          len(locs) == 192 and (locs[0]["row"], locs[0]["column"]) == (0, 0)
          and (locs[1]["row"], locs[1]["column"]) == (0, 1)
          and len(locs[0]) == 14)
    check("semantic hash present",
          isinstance(facts.get("semantic_sha256"), str)
          and len(facts["semantic_sha256"]) == 64)
    stats2 = _finalize_fixture(hit=False)
    stats2["n_floor_hits"] += 1
    stats2["decision"] = p4s2.branch_decision(1, 192)
    facts2 = _build(stats=stats2)
    check("semantic sensitivity: one floor-hit changes hash",
          facts["semantic_sha256"] != facts2["semantic_sha256"])


def test_publication_e2e() -> None:
    _CURRENT[0] = "test_publication_e2e"
    with tempfile.TemporaryDirectory() as tmp:
        facts = _build()
        path, sha = publish_stage(facts, tmp, "scaling_statistics", "P4")
        check("artefact written", os.path.isfile(path))
        check("sidecar verifies", file_sha256(path) == sha
              and os.path.isfile(path + ".sha256"))
        facts2 = _build()
        facts2["summary"]["n_rows"] += 1
        path2, sha2 = publish_stage(facts2, tmp, "scaling_statistics", "P4")
        check("authoritative untouched by rerun",
              file_sha256(os.path.join(tmp, "scaling_statistics.json"))
              == sha)
        siblings = [n for n in os.listdir(tmp)
                    if n.startswith("scaling_statistics.")
                    and n.endswith(".json") and n != "scaling_statistics.json"]
        check("timestamped sibling written",
              len(siblings) == 1 and path2.endswith(siblings[0]))
        claim = os.path.join(tmp, ".scaling_statistics.claim")
        with open(claim, "w") as fh:
            fh.write("stage=P4 pid=0 token=fixture utc=now\n")
        expect_error("concurrent claim refused", "PUBLICATION_CLAIM_HELD",
                     publish_stage, _build(), tmp, "scaling_statistics",
                     "P4")
    with tempfile.TemporaryDirectory() as tmp2:
        with open(os.path.join(tmp2, "scaling_statistics.json.sha256"),
                  "w") as fh:
            fh.write("0" * 64 + "  scaling_statistics.json\n")
        try:
            publish_stage(_build(), tmp2, "scaling_statistics", "P4")
            check("orphan sidecar -> ERROR", False, "no error raised")
        except StageError as exc:
            check("orphan sidecar -> ERROR", True, exc.error_code)
        except RuntimeError as exc:
            check("orphan sidecar -> ERROR", "unpaired" in str(exc),
                  str(exc)[:80])
    with tempfile.TemporaryDirectory() as tmp3:
        with open(os.path.join(tmp3, "scaling_statistics.tmp999"), "w") as fh:
            fh.write("residue")
        expect_error("stale temporary -> STALE_TEMPORARY_FOUND",
                     "STALE_TEMPORARY_FOUND",
                     publish_stage, _build(), tmp3, "scaling_statistics",
                     "P4")


def test_main_integration_smoke() -> None:
    """Integration regression (2026-08-09): drive main() to a SMOKE PASS
    with traversal/parent/provenance stubbed dataset-free. Pins the main()
    wiring that pure fixtures cannot reach -- a 1-tuple `consistency`
    (trailing-comma defect) or any future main() wiring break crashes this
    fixture rather than reaching the real-data path."""
    _CURRENT[0] = "test_main_integration_smoke"
    rng = np.random.default_rng(23)
    acc = p4s2.new_accumulator(96)
    for k in range(50):
        block = (rng.normal(0.0, 1.0, (96, 2))
                 + 1j * rng.normal(0.0, 1.0, (96, 2)))
        p4s2.accumulate_observations(acc, (0, 1),
                                     block.astype(np.complex128))
    own_w = {0: {"a.h5": 50}, 1: {"a.h5": 50}}
    trav = {"acc": acc, "weights": own_w, "files": ["a.h5"],
            "per_slice": {"rows": 2, "seed_agreement": True,
                          "mask_identity": True, "acquired_count": True},
            "dataset": None}
    saved = {}
    for name in ("load_p4s1_parent", "bind_mask_seed_provenance",
                 "traverse_frame", "dataset_provenance"):
        saved[name] = getattr(p4s2, name)
    try:
        p4s2.load_p4s1_parent = lambda path: _stub_parent()
        p4s2.bind_mask_seed_provenance = lambda repo: {
            "resolved": True, "mask_seed_source_sha256":
            p4s2.GENERATOR_SOURCE_SHA256}
        p4s2.traverse_frame = lambda root, smoke, eligible: trav
        p4s2.dataset_provenance = lambda cls, obj=None: {
            "dataset_class": "stub"}
        with tempfile.TemporaryDirectory() as tmp:
            rc = p4s2.main(["--repo-dir", _REPO, "--data-root", "/unused",
                            "--p4-stats1", "/unused", "--out-dir", tmp,
                            "--smoke", "2"])
            facts = [n for n in os.listdir(tmp)
                     if n.startswith("smoke_scaling_statistics")
                     and n.endswith(".json")]
            check("main() smoke integration -> EXIT_PASS",
                  rc == p4s2.EXIT_PASS)
            check("smoke artefact + sidecar written",
                  len(facts) == 1
                  and os.path.isfile(os.path.join(tmp, facts[0])
                                     + ".sha256"))
            with open(os.path.join(tmp, facts[0])) as fh:
                payload = json.load(fh)
            cons = payload.get("consistency")
            check("consistency record is a dict (tuple-comma regression)",
                  isinstance(cons, dict)
                  and cons["per_slice"]["rows"] == 2)
            check("internal invariants evaluated in main()",
                  cons["internal_invariants"]["own_count_weight"]["holds"]
                  is True)
            check("branch smoke-scale flagged in published record",
                  payload["branch"]["smoke_scale"] is True
                  and payload["branch"]["selected"] == "PER-LOCATION")
    finally:
        for name, fn in saved.items():
            setattr(p4s2, name, fn)


def test_failure_boundary() -> None:
    _CURRENT[0] = "test_failure_boundary"
    with tempfile.TemporaryDirectory() as tmp:
        rc = p4s2.main(["--repo-dir", _REPO, "--data-root", tmp,
                        "--p4-stats1", "x", "--out-dir", tmp,
                        "--smoke", "0"])
        recs = [n for n in os.listdir(tmp)
                if n.startswith("smoke_p4s2_error") and n.endswith(".json")]
        payload = {}
        if recs:
            with open(os.path.join(tmp, recs[0])) as fh:
                payload = json.load(fh)
        check("BAD_SMOKE_SIZE -> exit 2 + error record",
              rc == p4s2.EXIT_ERROR
              and payload.get("error_code") == "BAD_SMOKE_SIZE")
    expect_error("POPULATION code", "FRAME_POPULATION_MISMATCH",
                 p4s2.check_population,
                 {"per_slice": {"rows": 1}, "files": []}, True)
    expect_error("BRANCH code", "BRANCH_OPERANDS_INVALID",
                 p4s2.branch_decision, 1, 0)
    expect_error("EMPTY_CHANNEL code", "EMPTY_CHANNEL_POPULATION",
                 p4s2.pooled_channel_moments,
                 p4s2.new_accumulator(96)["count"],
                 p4s2.new_accumulator(96)["mean"],
                 p4s2.new_accumulator(96)["M2"], [0], 0)
    expect_error("C7 code", "C7_OPERANDS_INVALID",
                 p4s2.c7_roundtrip, np.zeros((2, 2)), np.zeros((2, 2)))
    expect_error("PARENT_NOT_FOUND code", "PARENT_NOT_FOUND",
                 p4s2.load_p4s1_parent, "/nonexistent/scaling_stats.json")


# ---------------------------------------------------------------------------
# Registry + runner
# ---------------------------------------------------------------------------

EXPECTED_COUNTS = {
    "test_mask_seed_conventions": 5,
    "test_parent_loader": 5,
    "test_welford_accumulation": 8,
    "test_pooled_moments": 4,
    "test_floor_boundary": 6,
    "test_branch_decision": 7,
    "test_c7_roundtrip": 6,
    "test_finalize_per_location_branch": 5,
    "test_finalize_global_branch": 4,
    "test_pre_vote_validity": 1,
    "test_finalize_coverage_gate": 2,
    "test_parity_helpers": 7,
    "test_generator_pin": 3,
    "test_no_block_path": 2,
    "test_facts_builder": 8,
    "test_publication_e2e": 7,
    "test_main_integration_smoke": 5,
    "test_failure_boundary": 6,
}

FIXTURES = [test_mask_seed_conventions, test_parent_loader,
            test_welford_accumulation, test_pooled_moments,
            test_floor_boundary, test_branch_decision, test_c7_roundtrip,
            test_finalize_per_location_branch, test_finalize_global_branch,
            test_pre_vote_validity, test_finalize_coverage_gate,
            test_parity_helpers, test_generator_pin, test_no_block_path,
            test_facts_builder, test_publication_e2e,
            test_main_integration_smoke, test_failure_boundary]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=f"{SCRIPT_ID} {SCRIPT_VERSION}")
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args(argv)
    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode="w"))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    for fx in FIXTURES:
        try:
            fx()
        except Exception as exc:  # noqa: BLE001 -- reported, not hidden
            logger.exception("fixture %s crashed", fx.__name__)
            _RESULTS.append((_CURRENT[0], "fixture completed without crash",
                             False, f"{type(exc).__name__}: {exc}"))
    per_fixture: dict[str, int] = {}
    failed = []
    for fixture, name, ok, detail in _RESULTS:
        per_fixture[fixture] = per_fixture.get(fixture, 0) + 1
        if not ok:
            failed.append({"fixture": fixture, "check": name,
                           "detail": detail})
    passed = sum(1 for r in _RESULTS if r[2])
    total = len(_RESULTS)
    coverage_ok = per_fixture == EXPECTED_COUNTS
    for fixture, count in sorted(per_fixture.items()):
        logger.info("coverage %-36s %d/%d", fixture, count,
                    EXPECTED_COUNTS.get(fixture, -1))
    if not coverage_ok:
        logger.error("COVERAGE MISMATCH: registry %s != actual %s -- the "
                     "registry is stale; re-derive it from THIS source",
                     EXPECTED_COUNTS, per_fixture)
    for f in failed:
        logger.error("FAIL %s::%s -- %s", f["fixture"], f["check"],
                     f["detail"])
    logger.info("%s %s: %d/%d checks passed (coverage_ok=%s)",
                SCRIPT_ID, SCRIPT_VERSION, passed, total, coverage_ok)
    print(json.dumps({"script": SCRIPT_ID, "version": SCRIPT_VERSION,
                      "passed": passed, "total": total,
                      "coverage_ok": coverage_ok,
                      "per_fixture": per_fixture,
                      "failed": failed}, indent=2))
    return 0 if (passed == total and coverage_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
