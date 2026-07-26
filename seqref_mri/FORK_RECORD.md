# seqref_mri fork record
Forked from seqref_warm (tag seqref-warm-final) per FORK-RECIPE v0.1.

RECORDED EXCEPTION to FORK-RECIPE v0.1 SS4 (third-tree refactor guidance):
a separate seqref_mri tree is created because MRI introduces complex
k-space, HDF5 input, Fourier operators, and substantially different
normalization and memory requirements. Shared components will be
reconsidered only after the MRI competence pilot establishes which
interfaces are genuinely stable.

Environment: seqref_mri/.venv_mri (separate; never committed).
Legacy-assumption scan at fork time: /tmp/mri_legacy_scan.txt -- MUST be
reviewed line by line before any MRI logic is added (MNIST loaders,
28x28/784 shapes, and degradation ops treated as suspect, not inherited).
This record is updated with the review outcome afterwards.

## S1 review outcome (2026-07-24)

Original fork scan recovered intact: 189 lines, preserved at
`results/_diag/mri_legacy_scan.txt` (copied from /tmp before loss).
All 189 scan lines were reviewed and mapped to one or more explicit
assumption verdicts. Detailed classifications are recorded in
SEQREF-MRI-EXEC section 4 (S1 ledger).

Verdict counts, BY SCAN LINE (each line counted once under its primary
verdict; one line carries two assumption verdicts, noted below):

    REBUILD                127
    REMOVE (from MRI path)  38   (deletion deferred until replacement +
                                  audit obligations complete)
    INHERIT                 19
    REVIEW                   5   (grad_diag.py -- keep-or-drop undecided)
    TOTAL                  189

BY CLASSIFIED ASSUMPTION: 190 verdicts. The single dual-verdict line is
`src/refiners/coupling_regressor.py:136` (regressor architecture INHERIT;
`dim=784` default REBUILD to required-from-cell).

Supplementary coverage beyond the scan baseline (source-context reads and
broader greps during review) produced additional recorded verdicts not
counted above, including: metrics.py SSIM internals (INHERIT) and its
implicit data range (REBUILD, locked at the pilot metric-sanity check);
base_experts.py CondGlow / CondRealNVPImage internals (REMOVE from MRI
path -- roster exclusion / on-record negative); the conditioner's
single-channel input layer and validation (REBUILD at I2 to the locked
3-channel stack); flow_matching_refiner.py (REVIEW). These are likewise
recorded in EXEC section 4.

S1 is CLOSED on this basis. The campaign advances to implementation-ready
(I1 next) per the EXEC tracker.