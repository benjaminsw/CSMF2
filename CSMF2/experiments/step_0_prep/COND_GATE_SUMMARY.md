# STEP-0 / COND-GATE v0.3 — Summary

**Scaffolder:** EXP-SCAFFOLD v2.3 &nbsp;·&nbsp; **Diagnostics:** COND-GATE v0.3
**Location:** `CSMF2/common/`

---

## The three files

| File | What it does |
|---|---|
| `cond_diagnostics.py` | Checks 1–8 on the **conditioner** (`h`) and **FiLM heads** (`γ, β`): validity, diversity, gradient flow, cache correctness, seed-determinism, per-layer health. Provides `run_global_gate(...)` bundler and `check_move_forward(log)` exit-criteria gate. |
| `gate_diagnostics.py` | Check 9 — the **gate collapse probe**. Reports `Neff`, entropy, `max_w`, per-input weight variance, argmax-expert histogram. Raises if gate is dead / one expert dominates / weights ignore inputs. |
| `cond_viz.py` | 8 diagnostic plots (h/FiLM histograms, diversity heatmap, grad trajectory, logp-shuffle bar, NaN-Inf trajectory, determinism trajectory, per-layer FiLM bars, gate collapse panel). Saved to `step_X/plots/cond_gate/`. |

**All bodies raise `ValueError` on fail + log via `logger.error` — no silent pass / mock / dummy.**

---

## STEP-0 gate: 9 checks, all required every epoch

| # | Check | Fails if |
|---|---|---|
| 1 | `h_stats` | NaN / Inf / constant `h` |
| 2 | `h_diversity` | `h` collapses across batch |
| 3 | `film_stats` | γ or β NaN / Inf / zero-std (aggregate) |
| 4 | `st_sensitivity` | shuffle(`h`) doesn't change `s,t` or `logp` |
| 5 | `grad_norms` | conditioner OR any FiLM head has zero grad |
| 6 | `cache_check` | cached `h` ≠ fresh `h` (tol 1e-5) |
| 7 | `determinism_check` | `(y, seed)` → different `h` (tol 1e-6) |
| 8 | `film_stats_per_layer` | **any single** FiLM layer dead |
| 9 | `gate_collapse_probe` | `Neff < 1.5` / `max_w > 0.95` / `w` constant across inputs |

**Move-forward rule (`check_move_forward`):** all 9 pass · NLL not regressed · ≥3 seeds · main metric improved.

---

## Recommended add-ons (not in core — add when triggered)

| Add-on | Purpose | When to use |
|---|---|---|
| **Fuzz** (adversarial `y` inputs) | Catches rare-input NaN in conditioner / FiLM before they contaminate gate training | Before **Stage B** and any deployment / **WP1+** port (SR, SAR) |
| **W&B** (Weights & Biases logging) | Debug gate collapse; compare seeds, β, τ sweeps; visualise trends; cross-run per-layer FiLM stats when one expert behaves oddly | When doing **many runs** — β/τ sweeps, seed panels, WP3 ablation matrix |
| **Per-layer FiLM deep dive** (already in check #8 but expanded with per-epoch dumps) | Isolate one bad FiLM head when an expert degrades | Triggered on-demand when an expert looks odd |

---

## When to run each probe

| Probe | Step range | Why |
|---|---|---|
| **Collapse probe** (check #9) | **Step 1.2 → 2.x** | Gate exists from 1.2 onward; collapse must be caught *before* Stage B freezes experts. |
| **Fuzz** | **WP1+** (Step 2.2 onward, all of WP4 / WP5) | Measurement-consistency layer + real imagery introduce degenerate inputs. |
| **W&B** | Whenever **sweeping** (WP3 ablations, WP5 final sweeps over λ_cons, T, τ) | Comparing ≥5 runs becomes unmanageable without a dashboard. |
| **Checks 1–8** | **Every step, every epoch** | Conditioner is used everywhere; these are the always-on gate. |

---

## Quick invocation

```python
from CSMF2.common import cond_diagnostics as cd
from CSMF2.common.cond_viz import plot_h_hist, plot_gate_collapse

result = cd.run_global_gate(
    h=h, h_batch=h_batch,
    gamma_aggregate=g, beta_aggregate=b,
    film_per_layer_outputs=film_outs,
    flow_fn=flow, x=x, h_shuffled=h_shuf,
    cond_net=cnet, film_heads=fheads,
    h_cached=hc, h_fresh=hf,
    y=y, seed=0, set_seed_fn=set_seed,
    gate_fn=gate, y_batch=y_batch,
)
if not result["passed"]:
    logger.error("COND-GATE failed: %s", result["reasons"])
```
