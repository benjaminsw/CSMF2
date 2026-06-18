# NICE-CB clamp-2p0 frozen baseline

`NCP-N0 v0.6` · roadmap slot 1.3b/N0 · created 2026-06-18T12:24:04.274300+00:00

## Identity
- expert: `nice`  (conditional base: True)
- base_logsigma_max: **2.0**
- seed: 0  (rng_seed: not_available)
- cfg_hash: `6e5678b743b4`
- degradation: s2_n0.05 (scale 2, noise 0.05)
- checkpoint: `/home/benjamin/CSMFII/CSMF2/experiments/step_1_4a/results/nice_cb_s2_n0.05_seed0_6e5678b743b4/ckpt.pt`
- checkpoint_sha256: `92cea6e9136729bbc99ca6e53dc9e0032dc4e5b87adc5d87540ea3ecb56e7377`
- git_commit: `11b175d1b11fea62e69cb4bf2373a6b6d4f1de60`
- freeze_hash: `e6da4944017610e6915987fd0a31a38896b5123d878d505bf064e8b18554f491`

## Metrics (contract)
- rec_argmin (NICE): **0**  ·  tier: **FLAT**
- RECGATE-global: NO_SUCCESS (soft_fwd_rel=0.1623 vs nsf_only=0.1587; neff>1.5=False, max_w<0.70=True)
- NLL (train/val/test): not_available / -3053.802667578125 / not_available
- invertibility max |·|inf: 2.245e+02  (pass < 1e-05: **False**)
- logdet sanity (f64): **pass**  (|err| mean 5.491e-06, max 5.976e-06)
- conditioning evidence: **proxy_recorded** (conditional_base_shuffle_sensitivity)

## Provenance (links only)
- recon grid: `/home/benjamin/CSMFII/CSMF2/experiments/step_1_4a/results/nice_cb_s2_n0.05_seed0_6e5678b743b4/plots/p4_recon_panel.png`
- NLL curve: `/home/benjamin/CSMFII/CSMF2/experiments/step_1_4a/results/nice_cb_s2_n0.05_seed0_6e5678b743b4/plots/p_nll_curve.png`
- RECGATE beta plot: `not_available`

## Rule
No NICE change is trusted unless it **beats** this baseline (rec_argmin PROVISIONAL+, RECGATE improves, NLL not critically regressed, invertibility < 1e-05, logdet f64 pass, conditioning active). WEAK tier is inconclusive, not a rescue.
