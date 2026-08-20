# SEQREF-TDIAG v0.1 -- scripts.tdiag
# LIFETIME: KEEP
# =============================================================================
# Purpose: TDIAG diagnostic stage driver (EXEC SS10.6, locked 2026-08-15
#          pre-implementation) -- diagnosis of the TINY likelihood-
#          reconstruction mismatch. This slice implements R0 (replay
#          validity), D1 (estimator slate) and D2a (true-latent
#          geometry); D2b/D2c/D3 land in later slices under the same
#          driver; D4/D5/D6 are amendment-gated and have NO execution
#          path (tdiag.invariants.refuse_deferred_probe). The D1 slice
#          (2026-08-18) adds the estimator slate on the frozen R0
#          runtime: E0-E4 + JVP, the E0/R0 exact-equivalence gate,
#          frozen-band materiality and four diagnostic figures. The D2a
#          slice (2026-08-19) adds the true-latent geometry measurements
#          under the state-swap identity invariant plus three figures.
#          The D2b slice (2026-08-19) adds the signed NLL decomposition
#          (L_base / L_logdet, registered-endpoint exact gate, D2a
#          z_true cross-tie) plus two figures. The D2c slice
#          (2026-08-20) adds the volume-level holdout generalization
#          (locked PCG64(1) 32-volume selection, R = G_hold/G_train,
#          locked-band classification) plus three figures.
# TAXONOMY (locked for this stage): the driver owns a standalone 0/2
#   contract -- 0 = a valid diagnostic evidence report was produced and
#   published; 2 = typed ERROR (invariant/replay/parent failure), error
#   record published where the context is trusted. The scientific
#   PASS/BLOCK tokens DO NOT EXIST here: TDIAG is evidence-only, can
#   never unblock PILOT/SCREEN/FORMAL and never converts the TINY BLOCK
#   into PASS. TINY's gate-verdict exit constants are deliberately NOT
#   imported -- BLOCK cannot leak in through copied infrastructure.
# Publication: seqref_mri/results/_diag/diag/tdiag_facts.json
#   (schema seqref-tdiag-facts/1); reruns write a stamped sibling, never
#   overwrite. ERROR writes a typed tdiag_error record instead.
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * Introduced with the R0 slice after the 2026-08-15 EXEC SS10.6 lock.
#   * Review-repair round (2026-08-16, pre-execution; NO contract
#     change): the IMPL cross-pin now checks the semantic hash alongside
#     the file hash, and run_r0 receives the freshly verified live
#     parent identities (IMPL file+semantic, TINY file) so the R0
#     evidence record compares registered records against live
#     verification, never against themselves.
#   * Review-repair round (2026-08-18, pre-execution; NO contract
#     change): ALL-OR-NOTHING publication -- the four D1 figures now
#     render BEFORE facts assembly/publication, so a D1_PLOT_FAILURE
#     aborts the execution with a typed ERROR and NO facts artefact
#     (never a valid report + ERROR exit from one run).
#   * D1 slice (2026-08-18, under the same SS10.6 lock; NO contract
#     change): the driver now continues from a VALID R0 into D1 -- the
#     frozen step-500 runtime is handed over via ReplayContext (never
#     rebuilt, never retrained), the estimator slate executes under the
#     E0/R0 equivalence gate, the evidence report gains the D1 block
#     (completeness D1 complete; run_mode validation-r0-d1) and the
#     four descriptive figures render BEFORE facts assembly/
#     publication (all-or-nothing, 2026-08-18 repair).
#   * D2a slice (2026-08-19, under the same SS10.6 lock; NO contract
#     change): the driver continues into D2a -- the verified step-0
#     state_dict is swapped into the SAME model under state-hash
#     verification for the step-0 measurements, the registered step-500
#     state is restored and re-verified, the three D2a figures render
#     BEFORE facts assembly (all-or-nothing extended), and the report
#     gains the nested d2 block (completeness D2 partial; run_mode
#     validation-r0-d1-d2a).
#   * D2b slice (2026-08-19, under the same SS10.6 lock; NO contract
#     change): the driver continues into D2b (signed NLL decomposition
#     with the registered-endpoint exact gate and the D2a z_true
#     cross-tie); the DRIVER now owns the step-0 state_dict lifetime --
#     it is cleared after the last D2-family consumer (D2c will need
#     it; move the clear when D2c lands); nine figures render BEFORE
#     facts assembly; the report gains d2.d2b (run_mode
#     validation-r0-d1-d2a-d2b).
#   * D2c slice (2026-08-20, under the same SS10.6 lock; NO contract
#     change): the driver continues into D2c (locked 32-volume holdout
#     selection, two-state measurement, G/R with the
#     registered-endpoint G_train, locked-band classification); the
#     step-0 state_dict is cleared after run_d2c (its last consumer);
#     twelve figures render BEFORE facts assembly; the report gains
#     d2.d2c, D2 flips complete (run_mode validation-r0-d1-d2a-d2b-
#     d2c).
# Update summary:
#   v0.1 lands the R0+D1+D2a+D2b+D2c driver: full parent chain (campaign
#   verifier + P3/P4/IMPL-B runtime loaders + IMPL dual-pin + TINY
#   dual-pin), registered-selection re-derivation, deterministic replay
#   of the 500 registered Adam steps through the production train_step,
#   exact serialized-value comparison, the D1 estimator slate plus the
#   D2a true-latent geometry and D2b likelihood decomposition on the
#   frozen replay runtime, evidence publication, descriptive
#   D1+D2a+D2b figures and the standalone 0/2 exit contract with the
#   startup-infrastructure guard.
# =============================================================================
from __future__ import annotations

import argparse
import logging
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

from preflight_parents import (StageError, verify_parents,  # noqa: E402
                               publish_stage, publish_error)
from seqref_mri.src import free_flow_runtime as ffr  # noqa: E402
from seqref_mri.scripts import tiny_gate as tg  # noqa: E402
from seqref_mri.tdiag import d1_plots  # noqa: E402
from seqref_mri.tdiag import d2a  # noqa: E402
from seqref_mri.tdiag import d2a_plots  # noqa: E402
from seqref_mri.tdiag import d2b  # noqa: E402
from seqref_mri.tdiag import d2b_plots  # noqa: E402
from seqref_mri.tdiag import d2c  # noqa: E402
from seqref_mri.tdiag import d2c_plots  # noqa: E402
from seqref_mri.tdiag import estimators  # noqa: E402
from seqref_mri.tdiag import facts as tfacts  # noqa: E402
from seqref_mri.tdiag import replay  # noqa: E402

SCRIPT_ID = "SEQREF-TDIAG"
SCRIPT_VERSION = "v0.1"
logger = logging.getLogger(SCRIPT_ID)

# Standalone 0/2 contract (see header). No PASS/BLOCK exit codes exist.
EXIT_REPORT = 0
EXIT_ERROR = 2

# Sanity: the campaign preflight ERROR code and this stage's must agree.
from preflight_parents import EXIT_ERROR as _PREFLIGHT_EXIT_ERROR  # noqa: E402
if _PREFLIGHT_EXIT_ERROR != EXIT_ERROR:  # pragma: no cover - import guard
    raise RuntimeError("preflight EXIT_ERROR drifted from the TDIAG 0/2 "
                       "contract")


def _fail(code: str, message: str, **kwargs) -> StageError:
    logger.error("[%s] %s: %s", SCRIPT_ID, code, message)
    return StageError(code, message, **kwargs)


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SEQREF-TDIAG v0.1 -- TINY mismatch diagnosis "
                    "(evidence-only; R0 slice). Publishes "
                    "diag/tdiag_facts.json (seqref-tdiag-facts/1); ERROR "
                    "writes a typed tdiag_error record instead. Exit "
                    "contract: 0 report | 2 ERROR (no PASS/BLOCK exist).")
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", default=os.path.join(
        _REPO, "seqref_mri", "results", "_diag", "diag"))
    p.add_argument("--p0-facts", default=None)
    p.add_argument("--p0s-facts", default=None)
    p.add_argument("--p0s-script", default=None)
    p.add_argument("--p3-facts", default=None)
    p.add_argument("--p4-stats2", default=None)
    p.add_argument("--implb-facts", default=None)
    p.add_argument("--impl-facts", default=None)
    p.add_argument("--tiny-facts", default=None)
    p.add_argument("--log-file", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    # Startup infrastructure must NEVER escape as a raw nonzero code other
    # than the registered ERROR code: directory preparation and the log
    # FileHandler run under a guard returning EXIT_ERROR (same pattern as
    # the realpath guard below).
    try:
        os.makedirs(args.out_dir, exist_ok=True)
        if args.log_file:
            os.makedirs(os.path.dirname(os.path.abspath(args.log_file)),
                        exist_ok=True)
        handlers = [logging.StreamHandler(sys.stdout)]
        if args.log_file:
            handlers.append(logging.FileHandler(args.log_file, mode="w",
                                                encoding="utf-8"))
    except OSError as exc:
        print(f"startup infrastructure failure: could not prepare the "
              f"output/log targets: {exc}", file=sys.stderr)
        return EXIT_ERROR
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
        required = (args.p0_facts, args.p0s_facts, args.p0s_script,
                    args.p3_facts, args.p4_stats2, args.implb_facts,
                    args.impl_facts, args.tiny_facts)
        if not all(required):
            raise _fail(
                "PARENT_INPUT_MISSING",
                "TDIAG requires --p0-facts, --p0s-facts, --p0s-script, "
                "--p3-facts, --p4-stats2, --implb-facts, --impl-facts "
                "and --tiny-facts so the complete parent chain plus the "
                "authoritative TINY artefact are verified, not assumed",
                detail={}, write_record=False)
        parents = verify_parents(_REPO, args.p0_facts, args.p0s_facts,
                                 args.p0s_script)
        p3 = ffr.load_p3_parent(args.p3_facts)
        p4 = ffr.load_p4s2_parent(args.p4_stats2)
        implb = ffr.load_implb_parent(args.implb_facts)
        impl, impl_file_sha = tg._load_impl_parent(args.impl_facts)
        s_ref = tg._s_ref_from_p0s(args.p0s_facts)
        tiny_facts, tiny_file_sha = replay.load_tiny_parent(
            args.tiny_facts)
        # Cross-pin: the TINY artefact's own IMPL parent record must match
        # the freshly verified IMPL artefact.
        tiny_impl = tiny_facts.get("parents", {}).get("impl_class_a", {})
        if tiny_impl.get("file_sha256") != impl_file_sha:
            raise _fail("PARENT_CROSS_PIN_MISMATCH",
                        f"TINY facts pin IMPL file "
                        f"{tiny_impl.get('file_sha256')} but the verified "
                        f"IMPL artefact is {impl_file_sha}")
        if tiny_impl.get("semantic_sha256") != impl["semantic_sha256"]:
            raise _fail("PARENT_CROSS_PIN_MISMATCH",
                        f"TINY facts pin IMPL semantic "
                        f"{tiny_impl.get('semantic_sha256')} but the "
                        f"verified IMPL artefact carries "
                        f"{impl['semantic_sha256']}")

        r0, ctx = replay.run_r0_with_context(
            args.data_root, tiny_facts, impl_file_sha,
            impl["semantic_sha256"], tiny_file_sha,
            float(implb["spline_b"]), p4, s_ref)
        d1 = estimators.run_d1(ctx, r0)
        d2a_block = d2a.run_d2a(ctx, r0, d1)
        d2b_block = d2b.run_d2b(ctx, r0, d2a_block)
        d2c_block = d2c.run_d2c(ctx, r0, tiny_facts, args.data_root, p4)
        # The DRIVER owns the step-0 state_dict lifetime (review
        # 2026-08-19): D2b/D2c swap the same verified state into the
        # same model, so no D2-family module may discard it. Cleared
        # after the last D2-family consumer (D2c, 2026-08-20).
        ctx.state0 = None
        # All-or-nothing publication (2026-08-18 repair, extended to D2a
        # on 2026-08-19, D2b on 2026-08-19 and D2c on 2026-08-20): ALL
        # descriptive figures render BEFORE the facts are
        # assembled/published. A D1_PLOT_FAILURE / D2A_PLOT_FAILURE /
        # D2B_PLOT_FAILURE / D2C_PLOT_FAILURE therefore aborts with a
        # typed ERROR and NO evidence artefact -- one execution can
        # never leave a valid report alongside an ERROR exit.
        figures = d1_plots.render_d1_figures(d1, args.out_dir)
        figures += d2a_plots.render_d2a_figures(d2a_block, args.out_dir)
        figures += d2b_plots.render_d2b_figures(d2b_block, args.out_dir)
        figures += d2c_plots.render_d2c_figures(d2c_block, args.out_dir)
        facts = tfacts.build_d2c_facts(
            r0, d1, d2a_block, d2b_block, d2c_block, tiny_facts,
            tiny_file_sha, impl, impl_file_sha, parents, p3, p4, implb,
            s_ref, _REPO, sys.argv)
        path, sha = publish_stage(facts, args.out_dir,
                                  tfacts.FACTS_PREFIX, tfacts.STAGE)
        logger.info("[%s] R0 replay VALID + D1 slate + D2a + D2b + D2c "
                    "complete; evidence report published %s sha256=%s "
                    "(partial: D3 pending; no verdict exists in "
                    "this stage); %d descriptive figures written "
                    "(non-evidence)",
                    SCRIPT_ID, path, sha, len(figures))
        return EXIT_REPORT
    except StageError as exc:
        logger.error("[%s] %s: %s", SCRIPT_ID, exc.error_code, exc.reason)
        publish_error(exc, args.out_dir, tfacts.ERROR_PREFIX,
                      tfacts.STAGE, parents=parents)
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
        logger.error("[%s] %s: %s", SCRIPT_ID, wrapped.error_code,
                     wrapped.reason)
        publish_error(wrapped, args.out_dir, tfacts.ERROR_PREFIX,
                      tfacts.STAGE, parents=parents)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
