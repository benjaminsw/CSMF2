# =============================================================================
# SEQREF-MRI-V04P v0.3 -- scripts.v04_probe
# LIFETIME: KEEP
# Purpose: V04 lambda probe (V04SPEC v1.0 article 176 FROZEN SS9;
#          V04PLAN v1.0 article 178 FROZEN SS6). Gradient-magnitude balance
#          estimator for the hybrid objective at the registered fresh V04M
#          init (seed 20260813).
# Locked quantities (quoted, never CLI-tunable):
#   * probe batch list: the first 100 batches of
#     FastMRISliceDataset(split="train", mode="eval"), batch 32, sequential
#     -- identical list to V03P (A2 convention; the official val split IS
#     the 199-volume holdout and is never touched)
#   * forward+backward ONLY -- no optimizer exists in this module
#   * estimator: mean per-batch global L2 gradient norms g_density, g_rec
#     over the full V04M parameter set, computed in float64; the two terms
#     measured in SEPARATE forward/backward passes with
#     model.zero_grad(set_to_none=True) between them; g_rec on the
#     SCALE-INCLUDED rec term (s = 1e7 applied, lambda-free; A6)
#   * conversion: lambda_2' = 0.5 * (g_density_mean / g_rec_mean)
#   * validity chain (178 SS6.5/6.6; SPEC SS18): norms zero / non-finite /
#     < 1e-12, or lambda non-finite / <= 0 / outside [1e-6, 1e6]
#     => PROBE_VALIDITY_FAILURE (pre-run validity abort; amendment required
#     before re-probing; NO silent clipping)
#   * probe weight state: registered fresh V04M init (build_v04_model,
#     MODEL_INIT_SEED) -- recorded in the probe record
# Emits v04_probe_record.json (schema seqref-v04-probe-record/1) + SHA-256
#   sidecar. Evidence only -- no verdict fields.
# GATE: the authoritative run (--mode run) requires the pinned
#   implementation commit with fixtures f01-f17 green (178 SS6.2). It is
#   NOT executed at v0.1 -- this stage delivers the module + fixture f04
#   (synthetic failure branches + no-parameter-update proof) only.
# CONVENTION: logger.error + typed raise (V04PError). PASS(0)/ERROR(2).
#   No fallback, no mock, no placeholder, no silent pass.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import logging
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from seqref_mri.src import free_flow_runtime as ffr
from seqref_mri.src.fastmri_data import FastMRISliceDataset
from seqref_mri.scripts import v04_objective as v04o
from seqref_mri.scripts.train_base import _collate, _prepare
from seqref_mri.scripts.train_free_flow import targets_from_prepared
from seqref_mri.scripts.v02_manifests import BATCH_SIZE, canonical_json
from seqref_mri.scripts.v02_train import derive_cmap_from_mask
from seqref_mri.scripts.v04_model import build_v04_model

logger = logging.getLogger("seqref_mri.v04_probe")

__version__ = "0.3"
__abbr__ = "SEQREF-MRI-V04P"

N_PROBE_BATCHES = 100                       # locked (178 SS6.3)
LAMBDA_PRIME_FACTOR = 0.5                   # locked (178 SS6.6)
NORM_FLOOR = 1e-12                          # locked (178 SS6.5)
LAMBDA_BAND = (1e-6, 1e6)                   # locked (178 SS6.6)
PROBE_WEIGHT_STATE = "init"                 # locked: registered fresh init
PROBE_INIT_SEED = 20260813                  # locked (178 SS6.4)
SCHEMA = "seqref-v04-probe-record/1"
PROBE_BATCH_KEYS = ("u_scaled", "cond_in", "mask", "y_raw", "amax",
                    "tgt_norm", "maps")


class V04PError(RuntimeError):
    """Single typed identity for the V04 probe module."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V04PError(code, message)


# ---------------------------------------------------------------------------
# Pure estimator core (fixture f04 domain -- no data, no model)
# ---------------------------------------------------------------------------

def validate_norms(g_density: float, g_rec: float) -> None:
    """178 SS6.5 norm gates, per term: zero / non-finite / < 1e-12 abort."""
    for name, g in (("g_density", g_density), ("g_rec", g_rec)):
        if not np.isfinite(g):
            _fail("PROBE_VALIDITY_FAILURE",
                  f"{name} is non-finite ({g!r}); pre-run validity abort")
        if g == 0.0:
            _fail("PROBE_VALIDITY_FAILURE",
                  f"{name} is exactly zero; pre-run validity abort")
        if abs(g) < NORM_FLOOR:
            _fail("PROBE_VALIDITY_FAILURE",
                  f"{name}={g!r} is below the {NORM_FLOOR} floor; "
                  f"pre-run validity abort")


def validate_lambda(lam: float) -> None:
    """178 SS6.6 lambda gates: non-finite / <= 0 / outside the band."""
    lo, hi = LAMBDA_BAND
    if not np.isfinite(lam):
        _fail("PROBE_VALIDITY_FAILURE",
              f"lambda_2' is non-finite ({lam!r}); pre-run validity abort")
    if lam <= 0.0:
        _fail("PROBE_VALIDITY_FAILURE",
              f"lambda_2' is non-positive ({lam!r}); pre-run validity abort")
    if not (lo <= lam <= hi):
        _fail("PROBE_VALIDITY_FAILURE",
              f"lambda_2' is {lam!r}, outside the locked band [{lo}, {hi}]; "
              f"pre-run validity abort")


def compute_lambda_prime(g_density: float, g_rec: float) -> float:
    """lambda_2' = 0.5 * (g_density / g_rec) with the full 178 SS6
    validity chain. Norm gates fire before the lambda gates. No clipping."""
    validate_norms(g_density, g_rec)
    lam = LAMBDA_PRIME_FACTOR * (g_density / g_rec)
    validate_lambda(lam)
    return float(lam)


def global_grad_norm(model) -> float:
    """Global L2 norm over every parameter gradient (float64)."""
    sq = 0.0
    n = 0
    for p in model.parameters():
        if p.grad is not None:
            sq += float((p.grad.detach().to(torch.float64) ** 2).sum())
            n += 1
    if n == 0:
        _fail("PROBE_VALIDITY_FAILURE",
              "no parameter carries a gradient; treated as zero norm")
    return float(np.sqrt(sq))


def grad_coverage(model) -> tuple:
    """(n_trainable, n_with_grad) -- telemetry for lambda interpretation."""
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    return n_train, n_grad


def parameter_digest(model) -> str:
    """SHA-256 over (name, raw parameter bytes) in fixed parameter order.
    Bitwise truth: the probe must leave every parameter unchanged."""
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        h.update(n.encode())
        h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Measurement (gradient contract pinned to v03_probe.py v0.5, 178 SS6.5)
# ---------------------------------------------------------------------------

def _check_batch_schema(batch: dict, bi: int) -> None:
    missing = [k for k in PROBE_BATCH_KEYS if k not in batch]
    if missing:
        _fail("PROBE_BATCH_SCHEMA",
              f"probe batch {bi} is missing keys {missing}; required "
              f"{list(PROBE_BATCH_KEYS)}")


def measure_probe_gradients(model, batches) -> dict:
    """Mean per-batch global L2 gradient norms over the given prepared
    probe batches. Density and reconstruction terms are measured in
    SEPARATE forward/backward passes with zero_grad(set_to_none=True)
    between them. NO optimizer exists in this module and no optimizer step
    occurs; parameters are digest-verified bitwise-identical before and
    after, and gradients are left cleared."""
    digest_before = parameter_digest(model)
    g_d, g_r = [], []
    cov_d0 = cov_r0 = None
    for bi, batch in enumerate(batches):
        _check_batch_schema(batch, bi)
        model.zero_grad(set_to_none=True)
        nll = v04o.density_only(model, batch["u_scaled"], batch["cond_in"],
                                batch["mask"])
        nll.backward()
        g_d.append(global_grad_norm(model))
        cov_d = grad_coverage(model)

        model.zero_grad(set_to_none=True)
        rec = v04o.rec_term(model, batch["cond_in"], batch["mask"],
                            batch["y_raw"], batch["amax"], batch["tgt_norm"],
                            batch["maps"])
        rec.backward()
        g_r.append(global_grad_norm(model))
        cov_r = grad_coverage(model)
        if bi == 0:
            cov_d0, cov_r0 = cov_d, cov_r
        elif cov_d != cov_d0 or cov_r != cov_r0:
            _fail("GRAD_COVERAGE_DRIFT",
                  f"batch {bi}: gradient coverage drifted from batch 0 "
                  f"({cov_d}/{cov_r} vs {cov_d0}/{cov_r0})")
    model.zero_grad(set_to_none=True)
    if parameter_digest(model) != digest_before:
        _fail("PROBE_STATE_MUTATION",
              "model parameters changed during probe measurement; the probe "
              "contract forbids any update")
    g_d_arr = np.asarray(g_d, dtype=np.float64)
    g_r_arr = np.asarray(g_r, dtype=np.float64)
    return {
        "g_density_per_batch": g_d_arr.tolist(),
        "g_rec_per_batch": g_r_arr.tolist(),
        "g_density_mean": float(g_d_arr.mean()),
        "g_rec_mean": float(g_r_arr.mean()),
        "parameters": {"trainable": cov_d0[0],
                       "with_grad_density": cov_d0[1],
                       "with_grad_rec": cov_r0[1]},
        "parameter_sha256": digest_before,
    }


# ---------------------------------------------------------------------------
# Probe record writer (evidence only -- no verdict fields)
# ---------------------------------------------------------------------------

def write_probe_record(record: dict, out_dir: str) -> tuple:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "v04_probe_record.json"
    path.write_bytes(canonical_json(record))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "v04_probe_record.json.sha256").write_text(
        f"{digest}  {path.name}\n")
    logger.info("[%s] probe record written: %s (sha256 %s...)", __abbr__,
                path, digest[:12])
    return str(path), digest


# ---------------------------------------------------------------------------
# Authoritative run (GATED: pinned commit + f01-f17 green; not run at v0.2)
# ---------------------------------------------------------------------------

def prepare_probe_batches(data_root: str, p4_stats2: str):
    """The locked probe batch list (178 SS6.3): first 100 batches of
    FastMRISliceDataset(split="train", mode="eval"), batch 32, sequential
    -- identical to V03P. Returns (batches, batch_list_sha256, p4).
    Single source: V04P's authoritative run and V04G1's gates both
    consume this function."""
    p4 = ffr.load_p4s2_parent(p4_stats2)
    logger.info("[%s] P4 /2 parent verified (file %s...)", __abbr__,
                p4["file_sha256"][:12])
    # Amendment A2: probe validation = TRAIN split in eval mode
    # (deterministic masks, training distribution). The official val
    # split IS the 199-volume holdout; using it here would be
    # pre-evaluation holdout contact.
    ds = FastMRISliceDataset(data_root, split="train", mode="eval")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, collate_fn=_collate)
    if len(loader) < N_PROBE_BATCHES:
        _fail("PROBE_DATA_SHORT",
              f"validation split yields {len(loader)} batches; the locked "
              f"probe requires {N_PROBE_BATCHES}")
    batches = []
    batch_list_hash = hashlib.sha256()
    for bi, batch in enumerate(loader):
        if bi >= N_PROBE_BATCHES:
            break
        batch_list_hash.update(canonical_json(
            {"batch": bi,
             "meta": [{"file": m["file"], "slice_index": m["slice_index"]}
                      for m in batch["meta"]]}))
        prep = _prepare(batch, "cpu", test0=False)
        targets, cond_ins, masks, maps = [], [], [], []
        for j in range(len(batch["meta"])):
            cmap = derive_cmap_from_mask(batch["mask"][j])
            vecs = ffr.standardisation_vectors(cmap, p4["location_index"])
            one = {"x_norm": prep["x_norm"][j:j + 1],
                   "cond_in": prep["cond_in"][j:j + 1]}
            targets.append(targets_from_prepared(one, cmap, vecs))
            cond_ins.append(prep["cond_in"][j:j + 1])
            masks.append(batch["mask"][j:j + 1])
            maps.append((cmap, vecs))
        batches.append({"u_scaled": torch.cat(targets, 0),
                        "cond_in": torch.cat(cond_ins, 0),
                        "mask": torch.cat(masks, 0),
                        "y_raw": prep["y"], "amax": prep["amax"],
                        "tgt_norm": prep["tgt_norm"], "maps": maps})
        if (bi + 1) % 10 == 0:
            logger.info("[%s] prepared batch %d/%d", __abbr__, bi + 1,
                        N_PROBE_BATCHES)
    return batches, batch_list_hash.hexdigest(), p4


def _run(data_root: str, p4_stats2: str, out_dir: str) -> int:
    if ffr.MODEL_INIT_SEED != PROBE_INIT_SEED:
        _fail("PROBE_CONFIG_MISMATCH",
              f"MODEL_INIT_SEED={ffr.MODEL_INIT_SEED} != locked "
              f"{PROBE_INIT_SEED} (178 SS6.4)")
    batches, batch_list_hash, p4 = prepare_probe_batches(data_root,
                                                         p4_stats2)
    model = build_v04_model()           # registered init; PROBE_WEIGHT_STATE
    model.train(False)

    res = measure_probe_gradients(model, batches)
    lam = compute_lambda_prime(res["g_density_mean"], res["g_rec_mean"])
    logger.info("[%s] means: g_density=%.6e g_rec=%.6e lambda_2'=%.6e",
                __abbr__, res["g_density_mean"], res["g_rec_mean"], lam)

    record = {
        "schema": SCHEMA,
        "script": f"{__abbr__} v{__version__}",
        "spec": {"article": 176, "status": "FROZEN"},
        "plan": {"article": 178, "status": "FROZEN"},
        "weight_state": {"kind": PROBE_WEIGHT_STATE,
                         "model_init_seed": PROBE_INIT_SEED},
        "n_batches": N_PROBE_BATCHES,
        "batch_size": BATCH_SIZE,
        "split": "train", "mode": "eval", "role": "validation",
        "validation_source_amendment": "A2",
        "batch_list_sha256": batch_list_hash,
        "g_density_per_batch": res["g_density_per_batch"],
        "g_rec_per_batch": res["g_rec_per_batch"],
        "g_density_mean": res["g_density_mean"],
        "g_rec_mean": res["g_rec_mean"],
        "parameters": res["parameters"],
        "parameter_sha256": res["parameter_sha256"],
        "lambda_prime_factor": LAMBDA_PRIME_FACTOR,
        "lambda_prime": lam,
        "lambda_band": list(LAMBDA_BAND),
        "norm_floor": NORM_FLOOR,
        "l_rec_scale": v04o.L_REC_SCALE,
        "parents": {"p4_stats2": {"path": p4["path"],
                                  "file_sha256": p4["file_sha256"],
                                  "semantic_sha256": p4["semantic_sha256"]}},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "platform": platform.platform()},
    }
    path, digest = write_probe_record(record, out_dir)
    print(f"PASS {__abbr__} v{__version__} probe: lambda_2'={lam:.6e} "
          f"record={path} sha256={digest[:12]}...")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="v04_probe")
    ap.add_argument("--mode", choices=("run",), required=True,
                    help="authoritative probe only; fixture f04 lives in "
                         "scripts.v04_fixtures")
    ap.add_argument("--data-root")
    ap.add_argument("--p4-stats2")
    ap.add_argument("--out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        missing = [k for k, v in (("data-root", args.data_root),
                                  ("p4-stats2", args.p4_stats2),
                                  ("out", args.out)) if not v]
        if missing:
            _fail("PROBE_ARGS_MISSING", f"--mode run requires {missing}")
        return _run(args.data_root, args.p4_stats2, args.out)
    except V04PError as exc:
        logger.error("[%s] ERROR: %s", __abbr__, exc)
        return 2
    except v04o.V04LError as exc:
        logger.error("[%s] objective ERROR: %s", __abbr__, exc)
        return 2
    except Exception:
        logger.exception("[%s] unexpected exception", __abbr__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# Changelog
# v0.2 (2026-10-12): refactor for V04G1 reuse -- probe-batch preparation
#   extracted into prepare_probe_batches() (single source for the probe
#   and the reliability gate; identical locked batch list + hash).
#   Estimator, validity chain, measurement contract, record schema and
#   gating unchanged.
# v0.1 (2026-10-12): initial implementation under V04SPEC v1.0 (176,
#   FROZEN) SS9 / V04PLAN v1.0 (178, FROZEN) SS6. Estimator + validity
#   chain mirrored from the pinned v03_probe.py v0.5 procedure (all-
#   parameter float64 global L2 norms, separate forward/backward passes
#   with zero_grad between terms, no optimizer step, scale-included g_rec
#   at s = 1e7); V04-specific conversion lambda_2' = 0.5 * (g_d/g_r) with
#   the [1e-6, 1e6] band; no-parameter-update proof via SHA-256 parameter
#   digest before/after measurement (advisor-endorsed addition); probe
#   record seqref-v04-probe-record/1 gains batch_list_sha256 +
#   parameter_sha256 + lambda_prime_factor (178 SS7(c) locked fields).
#   Authoritative --mode run implemented but gated: not executed until
#   fixtures f01-f17 are green and the implementation commit is pinned.
# ---------------------------------------------------------------------------
