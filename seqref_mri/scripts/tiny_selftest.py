# SEQREF-TINYT v0.1 -- scripts.tiny_selftest
# LIFETIME: KEEP
# =============================================================================
# Purpose: fixtures harness for SEQREF-TINY (scripts.tiny_gate). Built
#          AFTER the 2026-08-13 review-repair round as the regression net
#          for its five fixes, so the repaired defects can never re-enter
#          silently. Pure-synthetic fixtures: no dataset, no model, no
#          parent artefacts -- each fixture isolates one repaired behaviour
#          against the REAL driver code (imported, never reimplemented).
# Coverage:
#   F1  IMPL parent byte-hash propagation: facts assembly must succeed
#       when the IMPL artefact JSON has no top-level file_sha256 (its byte
#       hash is external/sidecar provenance), embedding the VERIFIED sha
#       returned by the loader. Regression: KeyError after 500 steps.
#   F2  secondary posterior mean = mean of COMPLEX states, magnitude ONCE
#       (production train_base._posterior_mean convention); phase-opposed
#       bank collapses to zero, where mean-of-magnitudes would not.
#   F3  selection identity portability: identical relative structure under
#       two different data-root spellings yields the identical manifest
#       hash; no machine-specific root string anywhere in the selection.
#   F4  B3 gates the registered inequality final <= 0.5*initial DIRECTLY:
#       initial=0/final=0 -> pass, initial=0/final>0 -> fail, exact
#       boundary final == 0.5*initial -> pass, above -> fail, non-finite
#       -> fail; ratio report-only and null at initial == 0.
#   F5  endpoint stability invariants are EXACT: equal-count/different-set
#       exclusions raise EXCLUSION_DRIFT; equal-mean/different-vector
#       anchor PSNRs raise ANCHOR_DRIFT; identical endpoints pass.
#   F6  B1 exact 0.10 boundary and the double-division trap (a driver
#       dividing by FLOW_DIM_REAL twice reports ~7.2e-6 at the boundary).
#   F7  B2's two clauses gate INDEPENDENTLY (delta-only pass, anchor-only
#       pass, exact >= boundaries).
#   F8  P0S overlap is OBSERVED AND RECORDED (count/indices/identities
#       from the verified artefact's sampling.canonical_sorted_indices;
#       draw never altered; malformed sampling record refuses).
#   F9  the IDENTICAL fixed latent bank is decoded at both endpoints, with
#       the z=0 primary first (patched-decode recording fixture).
#   F10 endpoint-only structure: exactly two endpoint evaluations, no
#       metric call inside the training loop, no best-checkpoint
#       machinery; trace checkpoints record NLL only (structural guard).
#   F11 parent/sidecar refusal: unpinned IMPL file -> PARENT_FILE_MISMATCH;
#       missing/mismatched sidecar -> the real verifier raises.
#   F12 publication pairing: BLOCK verdict still publishes; published file
#       verifies against its sidecar; a rerun writes a stamped sibling and
#       never overwrites; exit constants are 0/1/2.
#   F13 ERROR boundary: missing parent arguments -> exit 2 with NO facts
#       and NO error record (untrusted-context rule), nothing masquerades
#       as scientific evidence.
#   F14 trusted-context ERROR boundary: an injected ordinary runtime
#       failure AFTER parents are trusted -> UNEXPECTED_RUNTIME_ERROR, a
#       DISTINCT tiny_error record WITH sidecar, NO tiny_facts, exit 2.
# Coverage registry: EXPECTED_COUNTS pins the check count of every
#   fixture plus the suite total; coverage_ok requires zero failures AND
#   exact count matches, so a green suite cannot silently shrink.
# Invocation: both `python seqref_mri/scripts/tiny_selftest.py` and
#   `python -m seqref_mri.scripts.tiny_selftest` are supported.
# Taxonomy: all fixtures PASS -> exit 0; any failure -> exit 2 (a failing
#   fixture is a construction/contract defect, ERROR class under LOCK 2;
#   never a scientific result). No fallback, no mock, no placeholder, no
#   silent pass: every failure path is logger.error + typed outcome.
# Changelog (NEW in v0.1):
#   * Introduced after the 2026-08-13 review-repair round on tiny_gate.
#   * Coverage extension (2026-08-13, pre-execution): F6-F13 close the
#     remaining agreed fixture scope -- B1 boundary/single-division, B2
#     clause independence, P0S overlap-allowed recording (paired with the
#     tiny_gate overlap repair), identical-bank reuse, endpoint-only
#     structure, parent/sidecar refusal, publication pairing with sibling
#     semantics, and the ERROR/exit-code boundary.
#   * Closure hardening (2026-08-13, pre-execution): F14 adds the
#     trusted-context ERROR integration case; EXPECTED_COUNTS static
#     registry pins per-fixture and total check counts against silent
#     shrinkage; import made robust for both direct and `-m` invocation.
# Update summary:
#   v0.1 pins the repairs as executable regressions: SHA-less parent
#   assembly, complex-mean secondary estimator, portable manifest
#   identity, direct B3 inequality with zero-initial edges, exact
#   endpoint-stability invariants, plus the wider pre-authoritative
#   contract guards (B1/B2 boundaries, overlap recording, bank identity,
#   endpoint-only, publication pairing, both ERROR-context boundaries)
#   under a static expected-count coverage registry.
# =============================================================================
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

if __package__:  # `python -m seqref_mri.scripts.tiny_selftest`
    from seqref_mri.scripts import tiny_gate as tg
else:  # direct script run: scripts/ is on sys.path
    import tiny_gate as tg
from preflight_io import file_sha256, verify_sidecar  # noqa: E402
from preflight_parents import (StageError, EXIT_PASS, EXIT_BLOCK,  # noqa: E402
                               EXIT_ERROR, publish_stage)

SCRIPT_ID = "SEQREF-TINYT"
SCRIPT_VERSION = "v0.1"
logger = logging.getLogger(SCRIPT_ID)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    if not ok:
        logger.error("[%s] fixture FAILED: %s -- %s", SCRIPT_ID, name,
                     detail)


def expect_stage_error(name: str, fn, code: str) -> None:
    try:
        fn()
    except StageError as exc:
        check(name, exc.error_code == code,
              f"StageError code {exc.error_code!r} (expected {code!r})")
        return
    except Exception as exc:  # wrong failure class: still a failure
        check(name, False,
              f"raised {type(exc).__name__} instead of StageError "
              f"{code}: {exc}")
        return
    check(name, False, f"no error raised; expected StageError {code}")


# ---------------------------------------------------------------------------
# F1 -- IMPL parent byte-hash propagation (review defect 1)
# ---------------------------------------------------------------------------

def f1_impl_sha_propagation() -> None:
    impl_no_file_sha = {  # mirrors the REAL artefact: no top-level hash
        "schema": "seqref-impl-facts/1",
        "semantic_sha256": "b" * 64,
        "verdict": "PASS",
    }
    verified_sha = "a" * 64  # the loader-verified byte hash, passed in
    m0 = {"nll_batch_mean": 100.0, "per_slice": [], "mean_psnr_z0": 10.0,
          "mean_psnr_anchor": 12.0, "mean_psnr_posterior_mean": 9.0,
          "excluded_count": 0, "excluded_identities": [],
          "mean_nmse_u_z0": 0.4, "mean_nmse_u_posterior_mean": 0.5,
          "aggregate_nmse_u_z0": 0.4, "aggregate_nmse_u_posterior_mean": 0.5}
    m500 = dict(m0)
    gates = {"verdict": "PASS", "failed_gates": []}
    selection = {"population": 34742}
    saved_code, saved_env = tg._code_record, tg.environment_record
    tg._code_record = lambda: {"fixture": "environment isolated"}  # noqa: E731
    tg.environment_record = lambda *a, **k: {"fixture": True}  # noqa: E731
    try:
        facts = tg._build_facts(selection, m0, m500, {"0": 100.0}, gates,
                                {"parents_id": "fixture", "p0": {}, "p0s": {}},
                                {}, {}, {}, impl_no_file_sha, verified_sha,
                                15.62704)
    except KeyError as exc:
        check("F1 assembly without parent file_sha256 field", False,
              f"KeyError regression: {exc}")
        return
    except Exception as exc:
        check("F1 assembly without parent file_sha256 field", False,
              f"unexpected {type(exc).__name__}: {exc}")
        return
    finally:
        tg._code_record, tg.environment_record = saved_code, saved_env
    rec = facts["parents"]["impl_class_a"]
    check("F1 assembly without parent file_sha256 field", True,
          "no KeyError")
    check("F1 embedded sha is the VERIFIED loader value",
          rec["file_sha256"] == verified_sha,
          f"got {rec['file_sha256']!r}")
    check("F1 parent schema/semantic/verdict preserved",
          rec["schema"] == "seqref-impl-facts/1"
          and rec["semantic_sha256"] == "b" * 64
          and rec["verdict"] == "PASS")


# ---------------------------------------------------------------------------
# F2 -- secondary posterior mean convention (review defect 2)
# ---------------------------------------------------------------------------

def f2_posterior_mean_convention() -> None:
    a = torch.tensor([[1.0 + 2.0j, -3.0 + 0.5j]])
    opposed = [a, -a]  # cancels in complex mean; magnitudes do NOT cancel
    mag = tg._bank_mean_mag(opposed)
    check("F2 phase-opposed bank collapses under complex mean",
          bool((mag == 0).all()), f"got {mag}")
    mean_of_mags = (opposed[0].abs() + opposed[1].abs()) / 2
    check("F2 regression witness: mean-of-magnitudes would differ",
          bool((mean_of_mags != mag).any()),
          "the two estimators coincide on this bank -- fixture broken")
    same = tg._bank_mean_mag([a, a, a])
    check("F2 identical bank reproduces the common magnitude",
          bool(torch.allclose(same, a.abs())))
    expect_stage_error("F2 empty bank raises TINY_BANK_EMPTY",
                       lambda: tg._bank_mean_mag([]), "TINY_BANK_EMPTY")


# ---------------------------------------------------------------------------
# F3 -- portable selection identity (review fix 3)
# ---------------------------------------------------------------------------

class _StubDS:
    """Synthetic dataset shell carrying ONLY the attributes _select_batch
    consumes (len, index, data_root); two roots, one relative structure."""

    def __init__(self, root: str, rels: list[tuple[str, int]]):
        self.data_root = Path(root)
        self.index = [(self.data_root / rel, s) for rel, s in rels]

    def __len__(self) -> int:
        return len(self.index)


def f3_portable_manifest() -> None:
    rels = [(f"singlecoil_train/file{k:02d}.h5", k % 3) for k in range(12)]
    sel_a = tg._select_batch(_StubDS("/machineA/data", rels))
    sel_b = tg._select_batch(_StubDS("/machineB/other_root", rels))
    check("F3 manifest stable across data-root spellings",
          sel_a["manifest_sha256"] == sel_b["manifest_sha256"],
          f"{sel_a['manifest_sha256'][:12]} vs "
          f"{sel_b['manifest_sha256'][:12]}")
    check("F3 ordered identities stable across roots",
          sel_a["ordered_identities"] == sel_b["ordered_identities"])
    blob = json.dumps(sel_a)
    check("F3 no machine-specific root spelling in selection",
          "machineA" not in blob and "/data/" not in blob)
    ident = sel_a["ordered_identities"][0]
    check("F3 identity shape: split+relative file+slice+dataset_index",
          set(ident) == {"split", "file", "slice_index", "dataset_index"}
          and ident["split"] == "train"
          and not ident["file"].startswith("/"))


# ---------------------------------------------------------------------------
# F4 -- B3 direct inequality, zero-initial edges (review fix 4)
# ---------------------------------------------------------------------------

def _metrics(nll: float, psnr: float, anchor: float, nmse: float) -> dict:
    return {"nll_batch_mean": nll, "per_slice": [],
            "mean_psnr_z0": psnr, "mean_psnr_anchor": anchor,
            "mean_psnr_posterior_mean": psnr, "excluded_count": 0,
            "excluded_identities": [], "mean_nmse_u_z0": nmse,
            "mean_nmse_u_posterior_mean": nmse,
            "aggregate_nmse_u_z0": nmse,
            "aggregate_nmse_u_posterior_mean": nmse}


def _b3_only(initial: float, final: float) -> dict:
    # B1/B2 wired to pass so the verdict isolates B3.
    m0 = _metrics(100.0, 10.0, 12.0, initial)
    m500 = _metrics(100.0 - 0.2 * 13824.0, 13.0, 12.0, final)
    return tg._evaluate_gates(m0, m500)


def f4_b3_direct_inequality() -> None:
    g = _b3_only(0.0, 0.0)
    check("F4 initial=0, final=0 passes (inequality defined at zero)",
          g["b3_pass"] and g["verdict"] == "PASS")
    check("F4 ratio null at initial=0", g["b3_nmse_ratio"] is None)
    g = _b3_only(0.0, 0.1)
    check("F4 initial=0, final>0 fails", not g["b3_pass"]
          and g["failed_gates"] == ["B3"] and g["verdict"] == "BLOCK")
    g = _b3_only(0.4, 0.2)  # final == 0.5*initial EXACTLY
    check("F4 exact boundary final == 0.5*initial passes (<=)",
          g["b3_pass"] and abs(g["b3_nmse_ratio"] - 0.5) < 1e-15)
    g = _b3_only(0.4, 0.2 + 1e-12)
    check("F4 epsilon above boundary fails", not g["b3_pass"])
    g = _b3_only(float("nan"), 0.1)
    check("F4 non-finite initial fails closed", not g["b3_pass"])
    g = _b3_only(0.4, float("inf"))
    check("F4 non-finite final fails closed", not g["b3_pass"])


# ---------------------------------------------------------------------------
# F5 -- exact endpoint-stability invariants (review fix 5)
# ---------------------------------------------------------------------------

def _endpoint(anchors: list[float], excluded: list[dict]) -> dict:
    return {"per_slice": [{"psnr_anchor": a} for a in anchors],
            "excluded_count": len(excluded),
            "excluded_identities": excluded}


def f5_exact_drift_invariants() -> None:
    ident = {"split": "train", "file": "f.h5", "slice_index": 3,
             "dataset_index": 7}
    stable = _endpoint([20.0, 21.0], [ident])
    tg._assert_endpoint_stability(stable, dict(stable))
    check("F5 identical endpoints pass", True)
    # same COUNT (1), different identity -> must raise
    other = dict(ident, slice_index=4)
    expect_stage_error(
        "F5 equal-count/different-set exclusions raise EXCLUSION_DRIFT",
        lambda: tg._assert_endpoint_stability(
            stable, _endpoint([20.0, 21.0], [other])),
        "EXCLUSION_DRIFT")
    # same MEAN (20.5), different per-slice vector -> must raise
    expect_stage_error(
        "F5 equal-mean/different-vector anchors raise ANCHOR_DRIFT",
        lambda: tg._assert_endpoint_stability(
            stable, _endpoint([21.0, 20.0], [ident])),
        "ANCHOR_DRIFT")
    expect_stage_error(
        "F5 missing exclusion entirely raises EXCLUSION_DRIFT",
        lambda: tg._assert_endpoint_stability(
            stable, _endpoint([20.0, 21.0], [])),
        "EXCLUSION_DRIFT")


# ---------------------------------------------------------------------------
# F6 -- B1 exact boundary + single division (double-division trap)
# ---------------------------------------------------------------------------

def f6_b1_boundary_and_single_division() -> None:
    d = float(tg.ffr.FLOW_DIM_REAL)
    thr = tg.B1_NLL_DROP_PER_DIM_MIN
    # exact-boundary drop, reconstructed by exact negation arithmetic
    m0 = _metrics(0.0, 10.0, 12.0, 0.4)
    m500 = _metrics(-(thr * d), 13.0, 12.0, 0.1)
    g = tg._evaluate_gates(m0, m500)
    check("F6 drop of exactly the threshold passes (>=)",
          g["b1_pass"], f"b1={g['b1_nll_drop_per_dim']!r}")
    check("F6 single division: b1 ~ 0.10, not 0.10/FLOW_DIM_REAL",
          abs(g["b1_nll_drop_per_dim"] - thr) <= 1e-12,
          f"b1={g['b1_nll_drop_per_dim']!r} (twice-divided would be "
          f"~{thr / d:.3e})")
    m500b = _metrics(-((thr - 1e-9) * d), 13.0, 12.0, 0.1)
    g2 = tg._evaluate_gates(m0, m500b)
    check("F6 epsilon below threshold fails",
          not g2["b1_pass"] and g2["failed_gates"] == ["B1"])


# ---------------------------------------------------------------------------
# F7 -- B2 clauses gate independently
# ---------------------------------------------------------------------------

def f7_b2_clauses_independent() -> None:
    nll0, nll1 = 0.0, -(0.2 * float(tg.ffr.FLOW_DIM_REAL))  # B1 passes
    base = dict(nmse0=0.4, nmse1=0.1)                      # B3 passes
    g = tg._evaluate_gates(_metrics(nll0, 10.0, 12.0, base["nmse0"]),
                           _metrics(nll1, 12.0, 12.5, base["nmse1"]))
    check("F7 exact boundaries pass: delta == 2.0 AND final == anchor-0.5",
          g["b2_pass"] and g["verdict"] == "PASS")
    g = tg._evaluate_gates(_metrics(nll0, 10.0, 12.0, base["nmse0"]),
                           _metrics(nll1, 12.5, 13.5, base["nmse1"]))
    check("F7 anchor clause fails independently of delta",
          not g["b2_pass"] and g["b2_clause_delta_pass"]
          and not g["b2_clause_anchor_pass"]
          and g["failed_gates"] == ["B2"])
    g = tg._evaluate_gates(_metrics(nll0, 10.0, 12.0, base["nmse0"]),
                           _metrics(nll1, 11.0, 11.4, base["nmse1"]))
    check("F7 delta clause fails independently of anchor",
          not g["b2_pass"] and not g["b2_clause_delta_pass"]
          and g["b2_clause_anchor_pass"]
          and g["failed_gates"] == ["B2"])
    g = tg._evaluate_gates(_metrics(nll0, 10.0, 12.0, base["nmse0"]),
                           _metrics(nll1, 11.9, 13.5, base["nmse1"]))
    check("F7 both clauses failing still reports a single B2 failure",
          not g["b2_pass"] and g["failed_gates"] == ["B2"])


# ---------------------------------------------------------------------------
# F8 -- P0S overlap observed and recorded (paired driver repair)
# ---------------------------------------------------------------------------

def _selection_stub(indices: list[int]) -> dict:
    return {"draw_order_indices": list(indices),
            "ordered_identities": [
                {"split": "train", "file": f"f{i}.h5", "slice_index": k,
                 "dataset_index": i}
                for k, i in enumerate(indices)],
            "p0s_overlap_rule": "no exclusion; permitted and recorded"}


def f8_p0s_overlap_recording() -> None:
    sel = _selection_stub([5, 9, 100, 42])
    tg._record_p0s_overlap(sel, {9, 42, 7})
    ov = sel["p0s_overlap"]
    check("F8 overlap count and sorted indices recorded",
          ov["count"] == 2 and ov["dataset_indices"] == [9, 42])
    check("F8 overlap identities recorded",
          [i["dataset_index"] for i in ov["identities"]] == [9, 42])
    check("F8 recording never alters the draw",
          sel["draw_order_indices"] == [5, 9, 100, 42])
    sel2 = _selection_stub([5, 9])
    tg._record_p0s_overlap(sel2, {1, 2})
    check("F8 empty overlap recorded as zero, not omitted",
          sel2["p0s_overlap"]["count"] == 0
          and sel2["p0s_overlap"]["dataset_indices"] == []
          and sel2["p0s_overlap"]["identities"] == [])
    with tempfile.TemporaryDirectory() as td:
        good = os.path.join(td, "p0s.json")
        with open(good, "w", encoding="utf-8") as fh:
            json.dump({"sampling": {"canonical_sorted_indices": [3, 1, 2]}},
                      fh)
        check("F8 P0S index set extracted from verified artefact",
              tg._p0s_indices_from_artefact(good) == {1, 2, 3})
        bad = os.path.join(td, "bad.json")
        with open(bad, "w", encoding="utf-8") as fh:
            json.dump({"sampling": {}}, fh)
        expect_stage_error(
            "F8 malformed sampling record raises PARENT_FIELD_MISSING",
            lambda: tg._p0s_indices_from_artefact(bad),
            "PARENT_FIELD_MISSING")


# ---------------------------------------------------------------------------
# F9 -- identical fixed latent bank at both endpoints
# ---------------------------------------------------------------------------

def f9_identical_bank_across_endpoints() -> None:
    seen: list[torch.Tensor] = []
    real_decode, real_nll = tg._decode_z, tg._nll

    def fake_decode(model, z, st):
        seen.append(z.clone())
        return torch.ones(2, 2, dtype=torch.complex64), np.zeros(4)

    st = {"identity": {"dataset_index": 0}, "target": np.zeros((1, 4)),
          "cond": torch.zeros(1, 2), "mask": torch.zeros(1, 2),
          "x_true_mag": torch.ones(2, 2), "anchor_mag": torch.ones(2, 2),
          "u_true": np.ones(4), "u_true_ratio": 1.0, "excluded": False}
    class _EvalStub:
        def eval(self):
            return None

    tg._decode_z, tg._nll = fake_decode, lambda *a, **k: 1.0
    try:
        latents = torch.randn(
            tg.LATENT_BANK_N, 4,
            generator=torch.Generator().manual_seed(tg.LATENT_BANK_SEED))
        m_a = tg._endpoint_metrics(_EvalStub(), [st], 15.62704, latents)
        n_first = len(seen)
        m_b = tg._endpoint_metrics(_EvalStub(), [st], 15.62704, latents)
    finally:
        tg._decode_z, tg._nll = real_decode, real_nll
    seq_a, seq_b = seen[:n_first], seen[n_first:]
    check("F9 identical bank decoded at both endpoints",
          len(seq_a) == len(seq_b) == tg.LATENT_BANK_N + 1
          and all(torch.equal(x, y) for x, y in zip(seq_a, seq_b)))
    check("F9 z=0 primary decode comes first",
          bool((seq_a[0] == 0).all()))
    check("F9 identical bank reproduces endpoint metrics",
          m_a["mean_psnr_posterior_mean"]
          == m_b["mean_psnr_posterior_mean"])


# ---------------------------------------------------------------------------
# F10 -- endpoint-only / no-checkpoint-selection structural guard
# ---------------------------------------------------------------------------

def f10_endpoint_only_structure() -> None:
    with open(tg.__file__, "r", encoding="utf-8") as fh:
        src = fh.read()
    check("F10 exactly two endpoint metric evaluations in main",
          src.count("_endpoint_metrics(model, states, s_ref, latents)") == 2)
    loop_start = src.index("for step in range(1, TINY_STEPS + 1):")
    loop_end = src.index("# Endpoint 2")
    loop_body = src[loop_start:loop_end]
    check("F10 no metric evaluation inside the training loop",
          "_endpoint_metrics" not in loop_body
          and "_psnr(" not in loop_body and "_nmse(" not in loop_body)
    check("F10 trace checkpoints record NLL only",
          loop_body.count("_nll(") == 1)
    check("F10 no best-checkpoint selection machinery",
          "argmin" not in src and "best_step" not in src)


# ---------------------------------------------------------------------------
# F11 -- parent / sidecar refusal
# ---------------------------------------------------------------------------

def f11_parent_and_sidecar_refusal() -> None:
    with tempfile.TemporaryDirectory() as td:
        fake = os.path.join(td, "implementation_facts.json")
        with open(fake, "w", encoding="utf-8") as fh:
            json.dump({"schema": "seqref-impl-facts/1",
                       "semantic_sha256": "b" * 64, "verdict": "PASS"}, fh)
        expect_stage_error(
            "F11 unpinned IMPL file refused (PARENT_FILE_MISMATCH)",
            lambda: tg._load_impl_parent(fake), "PARENT_FILE_MISMATCH")
        artefact = os.path.join(td, "art.json")
        with open(artefact, "w", encoding="utf-8") as fh:
            json.dump({"x": 1}, fh)
        try:
            verify_sidecar(artefact)
            missing_raised = False
        except FileNotFoundError:
            missing_raised = True
        check("F11 missing sidecar refused by the real verifier",
              missing_raised)
        with open(artefact + ".sha256", "w", encoding="utf-8") as fh:
            fh.write("0" * 64 + "  art.json")
        try:
            verify_sidecar(artefact)
            mismatch_raised = False
        except RuntimeError:
            mismatch_raised = True
        check("F11 mismatched sidecar refused by the real verifier",
              mismatch_raised)


# ---------------------------------------------------------------------------
# F12 -- publication pairing, sibling semantics, exit constants
# ---------------------------------------------------------------------------

def f12_publication_pairing_and_taxonomy() -> None:
    with tempfile.TemporaryDirectory() as td:
        facts = {"schema": "seqref-tiny-facts/1", "stage": "TINY",
                 "verdict": "BLOCK", "failed_gates": ["B2"], "run": {}}
        p1, s1 = publish_stage(facts, td, "tiny_facts", "TINY")
        check("F12 BLOCK verdict still publishes (taxonomy)",
              os.path.isfile(p1) and os.path.isfile(p1 + ".sha256"))
        check("F12 published artefact verifies against its sidecar",
              verify_sidecar(p1) == s1)
        first_bytes = open(p1, "rb").read()
        p2, _ = publish_stage({"schema": "seqref-tiny-facts/1",
                               "stage": "TINY", "verdict": "PASS",
                               "run": {}}, td, "tiny_facts", "TINY")
        check("F12 rerun writes a stamped sibling, original untouched",
              p2 != p1 and open(p1, "rb").read() == first_bytes
              and file_sha256(p1) == s1)
    check("F12 exit constants PASS/BLOCK/ERROR == 0/1/2",
          EXIT_PASS == 0 and EXIT_BLOCK == 1 and EXIT_ERROR == 2)


# ---------------------------------------------------------------------------
# F13 -- ERROR boundary: no facts after untrusted-context ERROR
# ---------------------------------------------------------------------------

def f13_error_boundary_and_exit_taxonomy() -> None:
    with tempfile.TemporaryDirectory() as td:
        rc = tg.main(["--repo-dir", os.path.realpath(tg._REPO),
                      "--data-root", td, "--out-dir", td])
        check("F13 missing parent arguments -> exit 2 (ERROR)", rc == 2)
        leftovers = [p for p in os.listdir(td)
                     if p.startswith("tiny_facts")
                     or p.startswith("tiny_error")]
        check("F13 no facts/error artefact after PARENT_INPUT_MISSING",
              leftovers == [], f"found {leftovers}")


# ---------------------------------------------------------------------------
# F14 -- trusted-context ERROR boundary (integration)
# ---------------------------------------------------------------------------

def f14_trusted_context_error() -> None:
    saved: dict = {}

    def _patch(name, value, owner=tg):
        saved[(owner, name)] = getattr(owner, name)
        setattr(owner, name, value)

    def _boom(*a, **k):
        raise RuntimeError("injected trusted-context failure")

    try:
        _patch("verify_parents",
               lambda *a, **k: {"parents_id": "fixture-trusted"})
        _patch("_load_impl_parent",
               lambda p: ({"schema": "x", "semantic_sha256": "s",
                           "verdict": "PASS"}, "f" * 64))
        _patch("_s_ref_from_p0s", lambda p: 15.62704)
        for name in ("load_p3_parent", "load_p4s2_parent",
                     "load_implb_parent"):
            _patch(name, lambda p: {}, owner=tg.ffr)
        _patch("FastMRISliceDataset", _boom)
        with tempfile.TemporaryDirectory() as td:
            rc = tg.main(["--repo-dir", os.path.realpath(tg._REPO),
                          "--data-root", td, "--out-dir", td,
                          "--p0-facts", "x", "--p0s-facts", "x",
                          "--p0s-script", "x", "--p3-facts", "x",
                          "--p4-stats2", "x", "--implb-facts", "x",
                          "--impl-facts", "x"])
            check("F14 trusted-context runtime failure -> exit 2",
                  rc == 2)
            errs = [p for p in os.listdir(td)
                    if p.startswith("tiny_error") and p.endswith(".json")]
            facts = [p for p in os.listdir(td)
                     if p.startswith("tiny_facts")]
            check("F14 distinct error record written", len(errs) == 1,
                  f"{errs}")
            check("F14 NO stage facts published after ERROR",
                  facts == [], f"{facts}")
            with open(os.path.join(td, errs[0]),
                      encoding="utf-8") as fh:
                rec = json.load(fh)
            check("F14 record typed UNEXPECTED_RUNTIME_ERROR, not facts",
                  rec.get("error_code") == "UNEXPECTED_RUNTIME_ERROR"
                  and rec.get("artefact_type") == "error"
                  and rec.get("schema") == "seqref-tiny-error/1"
                  and "injected trusted-context failure"
                  in rec.get("error_reason", ""))
            sidecars = [p for p in os.listdir(td)
                        if p.startswith("tiny_error")
                        and p.endswith(".sha256")]
            check("F14 error record carries a sidecar",
                  len(sidecars) == 1, f"{sidecars}")
    finally:
        for (owner, name), value in saved.items():
            setattr(owner, name, value)


# ---------------------------------------------------------------------------

EXPECTED_COUNTS = {  # static registry: a green suite cannot shrink
    "f1_impl_sha_propagation": 3,
    "f2_posterior_mean_convention": 4,
    "f3_portable_manifest": 4,
    "f4_b3_direct_inequality": 7,
    "f5_exact_drift_invariants": 4,
    "f6_b1_boundary_and_single_division": 3,
    "f7_b2_clauses_independent": 4,
    "f8_p0s_overlap_recording": 6,
    "f9_identical_bank_across_endpoints": 3,
    "f10_endpoint_only_structure": 4,
    "f11_parent_and_sidecar_refusal": 3,
    "f12_publication_pairing_and_taxonomy": 4,
    "f13_error_boundary_and_exit_taxonomy": 2,
    "f14_trusted_context_error": 5,
}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    fixtures = [f1_impl_sha_propagation, f2_posterior_mean_convention,
                f3_portable_manifest, f4_b3_direct_inequality,
                f5_exact_drift_invariants,
                f6_b1_boundary_and_single_division,
                f7_b2_clauses_independent,
                f8_p0s_overlap_recording,
                f9_identical_bank_across_endpoints,
                f10_endpoint_only_structure,
                f11_parent_and_sidecar_refusal,
                f12_publication_pairing_and_taxonomy,
                f13_error_boundary_and_exit_taxonomy,
                f14_trusted_context_error]
    counts_ok = True
    for fn in fixtures:
        before = len(RESULTS)
        fn()
        want = EXPECTED_COUNTS[fn.__name__]
        got = len(RESULTS) - before
        if got != want:
            counts_ok = False
            logger.error("[%s] coverage shrinkage: %s emitted %d checks, "
                         "registry pins %d", SCRIPT_ID, fn.__name__, got,
                         want)
    total = len(RESULTS)
    if total != EXPECTED_TOTAL:
        counts_ok = False
        logger.error("[%s] coverage shrinkage: suite emitted %d checks, "
                     "registry pins %d", SCRIPT_ID, total, EXPECTED_TOTAL)
    failed = [r for r in RESULTS if not r[1]]
    coverage_ok = not failed and counts_ok
    for name, ok, detail in RESULTS:
        logger.info("[%s] %s %s%s", SCRIPT_ID, "PASS" if ok else "FAIL",
                    name, f" -- {detail}" if (detail and not ok) else "")
    logger.info("[%s] fixtures: %d/%d PASS, coverage_ok=%s", SCRIPT_ID,
                total - len(failed), total, str(coverage_ok).lower())
    if failed or not counts_ok:
        if failed:
            logger.error("[%s] %d fixture(s) failed -- driver repair "
                         "required before any TINY execution", SCRIPT_ID,
                         len(failed))
        if not counts_ok:
            logger.error("[%s] coverage registry mismatch -- refusing "
                         "green exit", SCRIPT_ID)
        return 2
    logger.info("[%s] all fixtures green; tiny_gate v0.1 repairs hold",
                SCRIPT_ID)
    return 0


if __name__ == "__main__":
    sys.exit(main())
