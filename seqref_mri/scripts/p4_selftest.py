# =============================================================================
# SEQREF-P4ST v0.2 -- scripts.p4_selftest
# LIFETIME: KEEP
# Purpose: self-test for SEQREF-P4CS (P4 census/support layer, A5 Route C).
#   Pure-core fixtures: no dataset, no h5, no kspace. The suite pins the
#   seed SERIALIZATION strings, the exact-integer gate form, every ERROR
#   path of the safeguard block, the schema /1 scope (statistics and branch
#   vote ABSENT, never nulled), and the publication overwrite machinery.
# REGISTRY DISCIPLINE: EXPECTED_COUNTS is a STATIC count of this source,
#   re-derived per rewrite, never carried forward. The coverage audit fails
#   loudly on any mismatch: a green suite with a stale registry is
#   impossible by construction.
# CONVENTION: fixture failures are reported, never hidden; the suite exits
#   nonzero unless every check passes AND coverage matches the registry.
# Changelog
#   v0.2 (2026-08-08) Gap analysis against the reviewer's minimum fixture
#     list: added test_parent_pinning (parent SHA mismatch / unverifiable /
#     missing -> ERROR), test_generator_pin (the registered generator hash
#     is a GATE -- mismatch and unbound provenance -> ERROR), and
#     test_no_block_path (no StageBlock import and no BLOCK handler anywhere
#     in the stage source); extended test_publication_e2e with orphan-sidecar
#     and stale-temporary publication states. Registry re-derived statically:
#     66 -> 76 checks.
#   v0.2 addendum (2026-08-08) Reviewer HOLD closure: version-identity
#     aligned (header, SCRIPT_VERSION constant, CLI description, final
#     summary log) so a green log proves the version that ran;
#     ZERO/DENOMINATOR coverage completed with explicit negative, NaN and
#     +Inf weight fixtures (the stated minimum was
#     negative/non-integer/non-finite; only 1.5 and 0 were pinned).
#     Registry re-derived statically: 76 -> 81 checks.
#   v0.1 (2026-08-08) Created with SEQREF-P4CS v0.1 under A5.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

from preflight_io import file_sha256  # noqa: E402
from preflight_parents import StageError, publish_stage  # noqa: E402
from seqref_mri.scripts import p4_scaling_stats as p4  # noqa: E402
from seqref_mri.src import fastmri_data as fdm  # noqa: E402

SCRIPT_ID = "SEQREF-P4ST"
SCRIPT_VERSION = "v0.2"
logger = logging.getLogger(SCRIPT_ID)

_RESULTS: list[tuple[str, str, bool, str]] = []
_CURRENT = ["<none>"]


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
    check("centre==44..51", p4.centre_columns(96) == frozenset(range(44, 52)))
    m1 = fdm.make_cartesian_mask(96, 12345)
    m2 = fdm.make_cartesian_mask(96, 12345)
    check("mask deterministic", bool((m1 == m2).all()))
    check("acquired count exact", int(m1.sum()) == 24)
    check("centre block acquired", bool(m1[44:52].all()))
    train = fdm.canonical_mask_seed(fdm.TRAIN_BASE_SEED, "f.h5", 3, epoch=0)
    want = int.from_bytes(
        hashlib.sha256(b"20261000|0|f.h5|3").digest()[:8], "big")
    check("train serialization pinned base|epoch|relpath|slice",
          train == want)
    ev = fdm.canonical_mask_seed(fdm.EVAL_BASE_SEED, "f.h5", 3)
    want_ev = int.from_bytes(
        hashlib.sha256(b"20261001|f.h5|3").digest()[:8], "big")
    check("eval serialization pinned base|relpath|slice", ev == want_ev)


def test_accumulation() -> None:
    _CURRENT[0] = "test_accumulation"
    rows = [("a.h5", (1, 2)), ("b.h5", (1,)), ("a.h5", (2, 3))]
    w, n = p4.accumulate_weights(rows)
    check("col1 weights", w[1] == {"a.h5": 1, "b.h5": 1})
    check("col2 accumulates", w[2] == {"a.h5": 2})
    check("col3 single", w[3] == {"a.h5": 1})
    check("n_rows", n == 3)
    check("sparse: no zero entries",
          all(v > 0 for per in w.values() for v in per.values()))


def test_column_statistics() -> None:
    _CURRENT[0] = "test_column_statistics"
    cols = p4.column_statistics({0: {"a": 2, "b": 1}}, 4)
    check("all columns present", [c["column"] for c in cols] == [0, 1, 2, 3])
    c0 = cols[0]
    check("exact integers",
          (c0["n_free_raw"], c0["n_free_files"], c0["sum_w2"]) == (3, 2, 5))
    check("kish value", abs(c0["n_eff_kish"] - 9 / 5) < 1e-15)
    c1 = cols[1]
    check("never-free kish None with reason",
          c1["n_eff_kish"] is None and bool(c1["kish_note"]))
    check("weights preserved", c0["weights"] == {"a": 2, "b": 1})
    expect_error("non-integer weight", "WEIGHT_NOT_INTEGER",
                 p4.column_statistics, {0: {"a": 1.5}}, 2)
    expect_error("non-positive weight", "WEIGHT_NON_FINITE_OR_NONPOSITIVE",
                 p4.column_statistics, {0: {"a": 0}}, 2)
    expect_error("negative weight", "WEIGHT_NON_FINITE_OR_NONPOSITIVE",
                 p4.column_statistics, {0: {"a": -2}}, 2)
    expect_error("non-finite weight", "WEIGHT_NOT_INTEGER",
                 p4.column_statistics, {0: {"a": float("nan")}}, 2)


def test_classification() -> None:
    _CURRENT[0] = "test_classification"
    check("never-free", p4.classify_column(0, 0) == "never-free")
    check("boundary eligible (raw^2 == N*w2)",
          p4.classify_column(900, 900) == "eligible")
    check("one below boundary under-supported",
          p4.classify_column(899, 899) == "under-supported")
    import random
    rng = random.Random(20260808)
    agree = True
    for _ in range(200):
        files = rng.randint(1, 973)
        ws = [rng.randint(1, 50) for _ in range(files)]
        raw = sum(ws)
        w2 = sum(x * x for x in ws)
        intform = p4.classify_column(raw, w2)
        floatform = ("under-supported"
                     if raw * raw / w2 < p4.N_EFF_MIN else "eligible")
        if intform != floatform:
            agree = False
            break
    check("integer gate == float gate (200 cases)", agree)
    check("uniform weights: kish == n_files",
          p4.classify_column(1000, 1000) == "eligible")


def _guard_columns(free_map: dict, width: int = 96) -> list:
    return p4.column_statistics(free_map, width)


def test_structural_guard() -> None:
    _CURRENT[0] = "test_structural_guard"
    centre = p4.centre_columns(96)
    clean = {c: {"f%03d.h5" % i: 1 for i in range(3)}
             for c in range(96) if c not in centre}
    cols = _guard_columns(clean)
    g = p4.structural_guard(cols, centre, True)
    check("clean passes", g["centre_never_free_ok"] and
          g["noncentre_zero_columns"] == [])
    bad_centre = dict(clean)
    bad_centre[44] = {"x.h5": 1}
    expect_error("centre free (authoritative)", "CENTRE_COLUMN_OBSERVED_FREE",
                 p4.structural_guard, _guard_columns(bad_centre), centre,
                 True)
    expect_error("centre free (smoke still gates)",
                 "CENTRE_COLUMN_OBSERVED_FREE",
                 p4.structural_guard, _guard_columns(bad_centre), centre,
                 False)
    missing = {c: v for c, v in clean.items() if c != 0}
    expect_error("noncentre zero (authoritative)",
                 "NONCENTRE_COLUMN_NEVER_FREE",
                 p4.structural_guard, _guard_columns(missing), centre, True)
    g2 = p4.structural_guard(_guard_columns(missing), centre, False)
    check("noncentre zero (smoke recorded, not gated)",
          g2["noncentre_zero_columns"] == [0] and
          g2["noncentre_positive_gated"] is False)


def test_counting_invariants() -> None:
    _CURRENT[0] = "test_counting_invariants"
    ok = p4.counting_invariants(
        _guard_columns({0: {"a": 2, "b": 1}}, 4))
    check("clean pass", ok["all_ok"] and ok["non_negative_integers"])
    forged = [{"column": 0, "n_free_raw": 1, "n_free_files": 2,
               "sum_w2": 1, "n_eff_kish": 1.0, "kish_note": None,
               "weights": {}}]
    expect_error("files > raw", "COUNTING_INVARIANT_VIOLATED",
                 p4.counting_invariants, forged)
    forged_neg = [{"column": 0, "n_free_raw": -1, "n_free_files": 0,
                   "sum_w2": 0, "n_eff_kish": None, "kish_note": "x",
                   "weights": {}}]
    expect_error("negative count", "COUNTING_INVARIANT_VIOLATED",
                 p4.counting_invariants, forged_neg)
    forged_files = [{"column": 0, "n_free_raw": 2000, "n_free_files": 1000,
                     "sum_w2": 2000, "n_eff_kish": 2000.0,
                     "kish_note": None, "weights": {}}]
    expect_error("files bound / cauchy", "COUNTING_INVARIANT_VIOLATED",
                 p4.counting_invariants, forged_files)


def _pred_columns(kish_target: float, n_col: int = 88) -> list:
    # uniform weights: kish == n_files exactly
    files = int(round(kish_target))
    centre = p4.centre_columns(96)
    free_map = {}
    for c in range(96):
        if c in centre:
            continue
        free_map[c] = {"f%03d.h5" % i: 1 for i in range(files)}
    return p4.column_statistics(free_map, 96)


def test_frozen_prediction() -> None:
    _CURRENT[0] = "test_frozen_prediction"
    centre = p4.centre_columns(96)
    none_pos = p4.evaluate_frozen_prediction(
        p4.column_statistics({}, 96), centre)
    check("no positive column -> not evaluated",
          none_pos["evaluated"] is False and none_pos["gating"] is False)
    good = p4.evaluate_frozen_prediction(_pred_columns(958), centre)
    check("in-range -> consistent",
          good["finding"] == "consistent_with_prediction" and
          good["min_within_anticipated_range"])
    falsified = p4.evaluate_frozen_prediction(_pred_columns(940), centre)
    check("below floor -> falsified_model_recorded, NON-GATING",
          falsified["finding"] == "falsified_model_recorded" and
          falsified["gating"] is False)
    check("falsification does not raise", True)
    under = p4.evaluate_frozen_prediction(_pred_columns(899), centre)
    check("all_columns_pass_gate False when a column fails",
          under["all_columns_pass_gate"] is False)


def test_census_core() -> None:
    _CURRENT[0] = "test_census_core"
    centre = p4.centre_columns(96)
    noncentre = tuple(c for c in range(96) if c not in centre)
    rows = [("f%03d.h5" % i, noncentre) for i in range(100) for _ in range(3)]
    res = p4.census_core(rows, 96, authoritative=False)
    check("classes", res["classes"] == {"never-free": 8,
                                        "under-supported": 88,
                                        "eligible": 0})
    check("under-supported list complete",
          len(res["under_supported_columns"]) == 88)
    check("files table sorted", res["files"] == sorted(res["files"]) and
          len(res["files"]) == 100)
    check("n_rows", res["n_rows"] == 300)
    check("centre all never-free",
          all(res["columns"][c]["class"] == "never-free" for c in centre))
    expect_error("authoritative under-supported raises",
                 "UNDER_SUPPORTED_COLUMN",
                 p4.census_core, rows, 96, True)


def test_population_gate() -> None:
    _CURRENT[0] = "test_population_gate"
    expect_error("authoritative mismatch", "FRAME_POPULATION_MISMATCH",
                 p4.check_population, [("a", ())] * 10, ["a"], True)
    rec = p4.check_population([("a", ())] * 10, ["a"], False)
    check("smoke recorded only", rec["gated"] is False and
          rec["n_slices"] == 10)
    ok = p4.check_population([("f%04d" % (i % 973), ())
                              for i in range(34742)],
                             ["f%04d" % i for i in range(973)], True)
    check("authoritative match", ok["n_slices"] == 34742 and
          ok["n_files"] == 973)


def _mini_census():
    centre = p4.centre_columns(96)
    noncentre = tuple(c for c in range(96) if c not in centre)
    rows = [("f%03d.h5" % i, noncentre) for i in range(100)]
    return p4.census_core(rows, 96, authoritative=False)


_STUB_PARENTS = {"p0": {"facts_sha256": "p0", "contract_hash": "c"},
                 "p0s": {"facts_sha256": "p0s", "subset_manifest_sha256": "m"}}
_STUB_P1P2 = {"p1": {"facts_sha256": "p1", "semantic_sha256": "s1",
                     "ruling": "PASS"},
              "p2": {"facts_sha256": "p2", "semantic_sha256": "s2"}}


def _build(smoke=4, census=None):
    census = census if census is not None else _mini_census()
    pop = p4.check_population([("a", ())] * 10, ["a"], False)
    return p4._build_facts(_STUB_PARENTS, _STUB_P1P2, census, pop, 96,
                           "PASS", "fixture", _REPO,
                           os.path.abspath(p4.__file__), [], 0.0, smoke,
                           {"resolved": True}, {"dataset_class": "fixture"})


def test_facts_builder() -> None:
    _CURRENT[0] = "test_facts_builder"
    facts = _build()
    check("schema /1", facts["schema"] == "seqref-p4-stats/1")
    scope = facts["schema_scope"]
    check("absent_by_schema declares statistics+branch",
          set(scope["absent_by_schema"]) ==
          {"per_location_scaling_statistics", "branch_vote"})
    check("no statistics/branch keys in facts",
          not any(k in facts for k in
                  ("scaling_statistics", "branch_vote", "branch",
                   "location_statistics")))
    check("run_mode smoke, not authoritative",
          facts["run_mode"] == "smoke" and facts["authoritative"] is False)
    check("semantic hash present",
          isinstance(facts.get("semantic_sha256"), str) and
          len(facts["semantic_sha256"]) == 64)
    alt = _mini_census()
    alt["columns"][0]["weights"] = {"f000.h5": 2}
    alt["columns"][0]["n_free_raw"] += 1
    facts2 = _build(census=alt)
    check("semantic sensitivity: one weight flip changes hash",
          facts["semantic_sha256"] != facts2["semantic_sha256"])
    c0 = facts["columns"][0]
    check("w_i_sparse keyed by file-index strings",
          all(isinstance(k, str) and k.isdigit()
              for k in c0["w_i_sparse"]) and len(c0["w_i_sparse"]) == 100)


def test_publication_e2e() -> None:
    _CURRENT[0] = "test_publication_e2e"
    with tempfile.TemporaryDirectory() as tmp:
        facts = _build()
        path, sha = publish_stage(facts, tmp, "scaling_stats", "P4")
        check("artefact written", os.path.isfile(path))
        check("sidecar verifies", file_sha256(path) == sha and
              os.path.isfile(path + ".sha256"))
        facts2 = _build()
        facts2["summary"]["n_rows"] += 1
        path2, sha2 = publish_stage(facts2, tmp, "scaling_stats", "P4")
        check("authoritative untouched by rerun",
              file_sha256(os.path.join(tmp, "scaling_stats.json")) == sha)
        siblings = [n for n in os.listdir(tmp)
                    if n.startswith("scaling_stats.") and
                    n.endswith(".json") and n != "scaling_stats.json"]
        check("timestamped sibling written",
              len(siblings) == 1 and path2.endswith(siblings[0]))
        claim = os.path.join(tmp, ".scaling_stats.claim")
        with open(claim, "w") as fh:
            fh.write("stage=P4 pid=0 token=fixture utc=now\n")
        expect_error("concurrent claim refused", "PUBLICATION_CLAIM_HELD",
                     publish_stage, _build(), tmp, "scaling_stats", "P4")
    with tempfile.TemporaryDirectory() as tmp2:
        # Orphan sidecar: a sidecar with no facts file is residue of an
        # interrupted publication; check_pairing must reject the directory.
        with open(os.path.join(tmp2, "scaling_stats.json.sha256"),
                  "w") as fh:
            fh.write("0" * 64 + "  scaling_stats.json\n")
        try:
            publish_stage(_build(), tmp2, "scaling_stats", "P4")
            check("orphan sidecar -> ERROR", False, "no error raised")
        except StageError as exc:
            check("orphan sidecar -> ERROR", True, exc.error_code)
        except RuntimeError as exc:
            check("orphan sidecar -> ERROR", "unpaired" in str(exc),
                  str(exc)[:80])
    with tempfile.TemporaryDirectory() as tmp3:
        # Stale temporary: residue of an interrupted write must ERROR, never
        # be silently cleaned.
        with open(os.path.join(tmp3, "scaling_stats.tmp999"), "w") as fh:
            fh.write("residue")
        expect_error("stale temporary -> STALE_TEMPORARY_FOUND",
                     "STALE_TEMPORARY_FOUND",
                     publish_stage, _build(), tmp3, "scaling_stats", "P4")


def test_parent_pinning() -> None:
    _CURRENT[0] = "test_parent_pinning"
    from preflight_parents_p3 import verify_p1_p2
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for stage, schema in (("P1", "seqref-p1-facts/1"),
                              ("P2", "seqref-p2-facts/1")):
            p = os.path.join(tmp, f"{stage.lower()}_facts.json")
            with open(p, "w") as fh:
                json.dump({"schema": schema, "stage": stage,
                           "artefact_type": "stage_facts", "verdict": "PASS",
                           "semantic_sha256": "0" * 64}, fh)
            with open(p + ".sha256", "w") as fh:
                fh.write(f"{file_sha256(p)}  {os.path.basename(p)}\n")
            paths.append(p)
        expect_error("parent SHA mismatch -> ERROR", "P3_PARENT_SHA_MISMATCH",
                     verify_p1_p2, paths[0], paths[1],
                     expected_p1_sha=p4.P1_FACTS_SHA256,
                     expected_p2_sha=p4.P2_FACTS_SHA256,
                     expected_p1_semantic_sha=p4.P1_SEMANTIC_SHA256,
                     expected_p2_semantic_sha=p4.P2_SEMANTIC_SHA256)
        with open(paths[0], "a") as fh:
            fh.write(" ")   # tamper: sidecar now stale
        expect_error("tampered parent -> UNVERIFIABLE",
                     "P3_PARENT_UNVERIFIABLE",
                     verify_p1_p2, paths[0], paths[1],
                     expected_p1_sha=None, expected_p2_sha=None)
        expect_error("missing parent -> P3_PARENT_MISSING",
                     "P3_PARENT_MISSING",
                     verify_p1_p2, os.path.join(tmp, "absent.json"),
                     paths[1],
                     expected_p1_sha=None, expected_p2_sha=None)


def test_generator_pin() -> None:
    _CURRENT[0] = "test_generator_pin"
    good = {"resolved": True,
            "mask_seed_source_sha256": p4.GENERATOR_SOURCE_SHA256}
    p4.enforce_generator_pin(good)
    check("matching pin passes", True)
    expect_error("wrong generator hash -> ERROR", "GENERATOR_HASH_MISMATCH",
                 p4.enforce_generator_pin,
                 {"resolved": True, "mask_seed_source_sha256": "0" * 64})
    expect_error("unbound provenance -> ERROR", "GENERATOR_HASH_MISMATCH",
                 p4.enforce_generator_pin, {"resolved": False})


def test_no_block_path() -> None:
    _CURRENT[0] = "test_no_block_path"
    with open(p4.__file__, encoding="utf-8") as fh:
        src = fh.read()
    code_only = "\n".join(l for l in src.splitlines()
                          if not l.lstrip().startswith("#"))
    check("no StageBlock anywhere in stage code", "StageBlock" not in code_only)
    check("no BLOCK handler and EXIT_BLOCK never imported",
          "except StageBlock" not in code_only and
          "EXIT_BLOCK" not in code_only)


def test_failure_boundary() -> None:
    _CURRENT[0] = "test_failure_boundary"
    with tempfile.TemporaryDirectory() as tmp:
        rc = p4.main(["--repo-dir", _REPO, "--data-root", tmp,
                      "--p0-facts", "x", "--p0s-facts", "x",
                      "--p0s-script", "x", "--p1-facts", "x",
                      "--p2-facts", "x", "--out-dir", tmp, "--smoke", "0"])
        recs = [n for n in os.listdir(tmp)
                if n.startswith("smoke_p4_error") and n.endswith(".json")]
        payload = {}
        if recs:
            with open(os.path.join(tmp, recs[0])) as fh:
                payload = json.load(fh)
        check("BAD_SMOKE_SIZE -> exit 2 + error record",
              rc == p4.EXIT_ERROR and
              payload.get("error_code") == "BAD_SMOKE_SIZE")
    expect_error("WEIGHT_NOT_INTEGER code", "WEIGHT_NOT_INTEGER",
                 p4.validate_weight, 0, "a", 1.5)
    expect_error("WEIGHT negative code", "WEIGHT_NON_FINITE_OR_NONPOSITIVE",
                 p4.validate_weight, 0, "a", -2)
    expect_error("WEIGHT NaN code", "WEIGHT_NOT_INTEGER",
                 p4.validate_weight, 0, "a", float("nan"))
    expect_error("WEIGHT +Inf code", "WEIGHT_NOT_INTEGER",
                 p4.validate_weight, 0, "a", float("inf"))
    centre = p4.centre_columns(96)
    expect_error("CENTRE code", "CENTRE_COLUMN_OBSERVED_FREE",
                 p4.structural_guard,
                 _guard_columns({44: {"a": 1}}), centre, True)
    expect_error("NONCENTRE code", "NONCENTRE_COLUMN_NEVER_FREE",
                 p4.structural_guard, _guard_columns({}), centre, True)
    noncentre = tuple(c for c in range(96) if c not in centre)
    rows = [("f.h5", noncentre)]
    expect_error("UNDER_SUPPORTED code", "UNDER_SUPPORTED_COLUMN",
                 p4.census_core, rows, 96, True)
    expect_error("POPULATION code", "FRAME_POPULATION_MISMATCH",
                 p4.check_population, [], [], True)
    expect_error("INVARIANT code", "COUNTING_INVARIANT_VIOLATED",
                 p4.counting_invariants,
                 [{"column": 0, "n_free_raw": 1, "n_free_files": 2,
                   "sum_w2": 1, "n_eff_kish": 1.0, "kish_note": None,
                   "weights": {}}])


# ---------------------------------------------------------------------------
# Registry + runner
# ---------------------------------------------------------------------------

EXPECTED_COUNTS = {
    "test_mask_seed_conventions": 7,
    "test_accumulation": 5,
    "test_column_statistics": 9,
    "test_classification": 5,
    "test_structural_guard": 5,
    "test_counting_invariants": 4,
    "test_frozen_prediction": 5,
    "test_census_core": 6,
    "test_population_gate": 3,
    "test_facts_builder": 7,
    "test_publication_e2e": 7,
    "test_parent_pinning": 3,
    "test_generator_pin": 3,
    "test_no_block_path": 2,
    "test_failure_boundary": 10,
}

FIXTURES = [test_mask_seed_conventions, test_accumulation,
            test_column_statistics, test_classification,
            test_structural_guard, test_counting_invariants,
            test_frozen_prediction, test_census_core,
            test_population_gate, test_facts_builder,
            test_publication_e2e, test_parent_pinning,
            test_generator_pin, test_no_block_path,
            test_failure_boundary]


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
