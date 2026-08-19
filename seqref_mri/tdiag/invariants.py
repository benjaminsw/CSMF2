# SEQREF-TDIAG v0.1 -- tdiag.invariants
# LIFETIME: KEEP
# =============================================================================
# Purpose: locked constants and refusal guards for the TDIAG diagnostic
#          stage (EXEC SS10.6, locked 2026-08-15 pre-implementation).
#          This module is the single in-code home of:
#            * the authoritative TINY artefact dual-pin (file + semantic),
#            * the R0 registered-observable checkpoint grid,
#            * the deferred-probe guard: D4/D5/D6 are AMENDMENT-GATED and
#              have NO execution path under SEQREF-TDIAG v0.1 -- any
#              request for them is logger.error + typed StageError.
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * Introduced with the R0 slice after the 2026-08-15 EXEC SS10.6 lock.
#   * D1 slice (2026-08-18, under the same SS10.6 lock; NO contract
#     change): added the locked D1 estimator-slate constants -- Z_DIAG
#     bank, JVP probes, the MAP multi-start protocol, trajectory
#     checkpoints and the frozen materiality bands.
#   * D2a slice (2026-08-19, under the same SS10.6 lock; NO contract
#     change): added the locked D2a recording conventions -- the top-K
#     coordinate-drift count, the bank-percentile tie rule and the
#     analytic Gaussian-identity tolerance. D2a is
#     descriptive-mechanistic: NO band, NO routing constants exist.
# Update summary:
#   v0.1 pins the TINY dual-pin, the R0 checkpoint grid, the locked D1
#   estimator-slate constants, the D2a recording conventions and the
#   amendment-gated D4/D5/D6 refusal guard.
# =============================================================================
from __future__ import annotations

import logging

from seqref_mri.tdiag import _bootstrap  # noqa: F401

from preflight_parents import StageError

logger = logging.getLogger("SEQREF-TDIAG")

# --- Authoritative TINY artefact dual-pin (RECORD SS14; EXEC SS9.1) ------
# TDIAG's R0 replays the configuration that produced THIS artefact; the
# registered values are read from the verified artefact itself, and the
# pins below prove the artefact is the registered one.
TINY_FACTS_SCHEMA = "seqref-tiny-facts/1"
TINY_FACTS_FILE_SHA256 = ("9201b491622f3d03aac4657849ee6a82c9e65a3eccbd8fdd"
                          "0a6ecf82fafb8d4b")
TINY_FACTS_SEMANTIC_SHA256 = ("fdffd6597084222ace038e4f8e7eca739ef5cbb368f77"
                              "460f2d65834a7f31cd7")
TINY_REQUIRED_VERDICT = "BLOCK"  # the authoritative TINY closure (SS14)

# --- R0 checkpoint grid (EXEC SS10.6: NLL trace at {0, 50, ..., 500}) ----
R0_TRACE_CHECKPOINTS = tuple(range(0, 501, 50))          # 11 points
R0_STEPS = 500                                           # registered TINY
R0_MODEL_INIT_SEED = 0                                   # registered TINY
R0_SELECTION_SEED = 0                                    # registered TINY

# --- D1 estimator-slate locks (EXEC SS10.6 D1, locked 2026-08-15) ------
Z_DIAG_N = 128                      # shared latent bank size
Z_DIAG_SEED = 0                     # PCG64(0)
Z_DIAG_GENERATOR = "PCG64"
JVP_N_PROBES = 16                   # Rademacher probes at z=0
JVP_SEED = 2                        # PCG64(2)
MAP_N_STARTS = 8                    # z=0 + Z_DIAG[0:7], exactly 8
MAP_STEPS = 200                     # Adam steps per start (E3/E4)
MAP_LR = 1e-3                       # Adam lr (E3/E4)
MAP_TRAJ_CHECKPOINTS = (0, 25, 50, 75, 100, 125, 150, 175, 200)
MATERIAL_PSNR_DELTA_DB = 2.0        # material: mean_PSNR >= E0 + 2.0
MATERIAL_NMSE_RATIO = 0.5           # material: mean_NMSE_u <= 0.5 * E0

# --- Deferred probes (EXEC SS10.6 deferred block): DESIGN ONLY -----------
# D4 extended-budget continuation, D5 gradient/objective alignment,
# D6 layerwise. They are unlocked ONLY by a future amendment; under v0.1
# there is intentionally no functional hook for them.
DEFERRED_PROBES = ("D4", "D5", "D6")


def refuse_deferred_probe(probe: str) -> None:
    """Amendment-gated refusal guard. ANY attempt to execute a deferred
    probe under SEQREF-TDIAG v0.1 is a typed ERROR -- never a silent skip,
    never a stub result."""
    logger.error("[%s] DEFERRED_PROBE_AMENDMENT_GATED: probe %r is "
                 "deferred under EXEC SS10.6 and has no execution path in "
                 "SEQREF-TDIAG v0.1; it requires a separate amendment",
                 "SEQREF-TDIAG", probe)
    raise StageError(
        "DEFERRED_PROBE_AMENDMENT_GATED",
        f"probe {probe!r} is amendment-gated (EXEC SS10.6 deferred "
        f"block); SEQREF-TDIAG v0.1 implements R0/D1/D2/D3 only",
        detail={"probe": str(probe),
                "deferred_probes": list(DEFERRED_PROBES)})


# --- D2a locks (EXEC SS10.6 D2a, locked 2026-08-15) ----------------------
# D2a is DESCRIPTIVE-MECHANISTIC: the decision matrix does NOT consume
# it, so no band/threshold constants exist here -- only recording
# conventions are frozen.
D2A_TOP_K = 20                    # top-|z500 - z0| coordinates per slice
D2A_PERCENTILE_TIE_RULE = "<= observed value"
# Analytic base-density identity: log p_Z(z) = -0.5*||z||^2 - d/2*log(2pi)
# in float64; the production _gaussian_logprob must agree within this
# frozen ABSOLUTE tolerance (float64 summation-order noise only, ~1e-12).
GAUSS_LOGPROB_CHECK_TOL = 1e-9
