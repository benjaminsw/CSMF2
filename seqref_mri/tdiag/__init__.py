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
#   facts.py       -- seqref-tdiag-facts/1 assembly + code record
# Driver: seqref_mri/scripts/tdiag.py (taxonomy 0 = report, 2 = ERROR;
#   the scientific PASS/BLOCK tokens do not exist in this stage).
# Selftest: seqref_mri/scripts/tdiag_selftest.py (SEQREF-TDIAGT v0.1).
# =============================================================================
# =============================================================================
