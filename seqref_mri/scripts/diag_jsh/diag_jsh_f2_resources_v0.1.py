# SEQREF-MRI-DIAGJSHF2RES v0.1 -- DIAG-JSH F2 resource record (main-run limits), bound to the F1 manifest
# LIFETIME: KEEP
# =============================================================================
# Separate F2 resource record (reviewer edit 1, 217 J17): the F1 manifest
# d3e8176e... stays unchanged; this record is bound to its sha256 and is
# itself bound by code hash in the G command. Frozen rule (§8):
#   T = 1.5 x (t_probe x N x 6 + t_load x 6 + t_boot + t_plot)
# with t_probe, N, t_load, t_boot copied from the manifest's main_run_limits
# and t_plot the registered plotting bound (below). T is the Python stop
# limit; it is never extended. Command cap MAX_WALL_S = ceil(T) + 120 s.
# Write budget and log allowance (ENFORCED by WriteBudget; values are bounds,
# derived below from the sandbox probe and F1's measured log).
# Changelog (v0.1, 2026-11-11): first record.
# =============================================================================
import math

MANIFEST_SHA256 = ("d3e8176ef91dc1eb319b42f8510170eb6fe658a1ad1d99c561523fe"
                   "cd121d765")
# Copied verbatim from the manifest's main_run_limits (F1 run 3):
T_PROBE_PER_SLICE_S = 0.5108313020318747
N_SLICES = 3461
T_MODEL_LOAD_S = 3.541905004531145
T_BOOTSTRAP_S = 1.4135081395506859
# Registered plotting bound (see PLOT_BOUND_JUSTIFICATION):
T_PLOT_S = 120.0
PLOT_BOUND_JUSTIFICATION = (
    "sandbox probe SEQREF-MRI-DIAGJSHF2PROBE v0.1 at the real shape (3 x 97 "
    "volumes, 4 retained slices; synthetic values): figures 1.37 s, analysis "
    "3.04 s; bound 120 s (about 85 x the measured plotting time); the G run "
    "records the host time")
MARGIN = 1.5
T_S = MARGIN * (T_PROBE_PER_SLICE_S * N_SLICES * 6 + T_MODEL_LOAD_S * 6 +
                T_BOOTSTRAP_S + T_PLOT_S)
MAX_WALL_S = math.ceil(T_S) + 120
G_BUDGET_BYTES = 32 * 2 ** 20
LOG_ALLOWANCE_BYTES = 4 * 2 ** 20
BUDGET_JUSTIFICATION = (
    "probe outputs incl. sidecars and the matplotlib font cache 2,561,551 B "
    "(random images: S2 PNG 1.66 MB, an upper-range case); F1 run-3 log "
    "2,142,472 B (per-slice decoder lines, same decode in G); projected about "
    "4.8 MB, budget 32 MiB (about 6.5 x); log allowance 4 MiB (about 1.9 x "
    "the F1 log)")

RESOURCES = {
    "schema": "SEQREF-MRI-DIAG-JSH-F2RES/0.1",
    "manifest_sha256": MANIFEST_SHA256,
    "rule": "T = 1.5 x (t_probe x N x 6 + t_load x 6 + t_boot + t_plot) (§8)",
    "t_probe_per_slice_s": T_PROBE_PER_SLICE_S, "n_slices": N_SLICES,
    "t_model_load_s": T_MODEL_LOAD_S, "t_bootstrap_s": T_BOOTSTRAP_S,
    "t_plot_s": T_PLOT_S, "plot_bound": PLOT_BOUND_JUSTIFICATION,
    "T_s": T_S, "max_wall_s": float(MAX_WALL_S),
    "g_budget_bytes": G_BUDGET_BYTES,
    "log_allowance_bytes": LOG_ALLOWANCE_BYTES,
    "budget_bound": BUDGET_JUSTIFICATION,
}
