# SEQREF-TDIAG v0.1 -- tdiag (package)
# LIFETIME: KEEP
# =============================================================================
# Purpose: TDIAG diagnostic stage package (EXEC SS10.6, locked 2026-08-15,
#          pre-implementation). TDIAG diagnoses the TINY likelihood-
#          reconstruction mismatch. It is EVIDENCE-ONLY: it emits a
#          diagnostic evidence report (schema seqref-tdiag-facts/1), never
#          a PASS/BLOCK verdict, cannot unblock PILOT/SCREEN/FORMAL and
#          never converts the TINY BLOCK into PASS.
# Modules:
#   _bootstrap.py  -- explicit import-path bootstrap (no
#                     __init__ side effects, no PYTHONPATH)
#   invariants.py  -- locked constants, registered pins, deferred-probe
#                     guard (D4/D5/D6 are amendment-gated)
#   replay.py      -- R0 replay validity (exact serialized-value equality
#                     against the dual-pinned authoritative TINY artefact)
#                     + the frozen step-500 runtime handover
#                     (ReplayContext) for D1-D3
#   estimators.py  -- D1 estimator slate (Z_DIAG bank, E0-E4, JVP,
#                     E0/R0 equivalence gate, frozen-band materiality)
#   d1_plots.py    -- D1 descriptive figures (never registered evidence)
#   d2a.py         -- D2a true-latent geometry (state-swap identity
#                     invariant, Z_DIAG density reference, Gaussian
#                     identity check, step deltas, top-K drift)
#   d2a_plots.py   -- D2a descriptive figures (never registered evidence)
#   d2b.py         -- D2b signed NLL decomposition (endpoint exact gate,
#                     D2a z_true cross-tie, per-slice/aggregate deltas)
#   d2b_plots.py   -- D2b descriptive figures (never registered evidence)
#   d2c.py         -- D2c volume-level holdout generalization (locked
#                     PCG64(1) selection, two-state measurement, G/R,
#                     locked-band classification)
#   d2c_plots.py   -- D2c descriptive figures (never registered evidence)
#   facts.py       -- seqref-tdiag-facts/1 assembly + code record
# Driver: seqref_mri/scripts/tdiag.py (taxonomy 0 = report, 2 = ERROR;
#   the scientific PASS/BLOCK tokens do not exist in this stage).
# Selftest: seqref_mri/scripts/tdiag_selftest.py (SEQREF-TDIAGT v0.1).
# Changelog:
#   * v0.1 R0 slice (2026-08-15/17): package introduced (bootstrap-only
#     __init__, no path side effects).
#   * v0.1 D1 slice (2026-08-18, same SS10.6 lock; NO contract change):
#     module list extended with estimators.py and d1_plots.py; replay.py
#     now also hands the frozen step-500 runtime to D1-D3.
#   * v0.1 D2a slice (2026-08-19, same SS10.6 lock; NO contract change):
#     module list extended with d2a.py and d2a_plots.py; ReplayContext
#     now also carries the captured step-0 state_dict for D2a.
#   * v0.1 D2b slice (2026-08-19, same SS10.6 lock; NO contract change):
#     module list extended with d2b.py and d2b_plots.py; the step-0
#     state_dict lifetime is now driver-owned (D2b/D2c reuse it).
#   * v0.1 D2c slice (2026-08-20, same SS10.6 lock; NO contract change):
#     module list extended with d2c.py and d2c_plots.py; the driver
#     clears the step-0 state_dict after run_d2c (its last consumer).
# =============================================================================
# =============================================================================
