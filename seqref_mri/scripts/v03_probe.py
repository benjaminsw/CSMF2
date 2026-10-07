# SEQREF-V03P v0.5 -- scripts.v03_probe
# LIFETIME: KEEP
# =============================================================================
# Purpose: V03 lambda probe (V03SPEC v0.3 SS9, article 163 FROZEN;
#          V03PLAN v0.3 SS6, article 164 FROZEN). Validation-only
#          gradient-magnitude balance estimator for the hybrid objective.
# Locked quantities (quoted, never CLI-tunable):
#   * 100 validation batches (Amendment A2: split="train", mode="eval";
#     deterministic per-slice masks; the official val split IS the
#     199-volume holdout and is never touched here), batch 32 inherited
#     from the locked V02 parent config; sequential order, first 100
#     batches
#   * forward+backward ONLY -- no optimizer exists in this module
#   * estimator: mean per-batch global L2 gradient norms g_density,
#     g_rec over the full production parameter set
#   * conversion: lambda_k = (alpha_k/(1-alpha_k)) * (g_density/g_rec),
#     alpha in {0.1, 0.5, 0.9}; lambda_2 (alpha=0.5) is the primary arm
#   * probe-failure rule (SPEC SS9.5): norms zero / non-finite / < 1e-12,
#     or lambda non-finite / non-positive / outside [1e-6, 1e6]
#     => PROBE_VALIDITY_FAILURE (pre-run validity abort, amendment
#     required; NO silent clipping)
#   * probe weight state: registered fresh init (ffr.build_model,
#     MODEL_INIT_SEED) -- recorded in the probe record
# Emits v03_probe_record.json (schema seqref-v03-probe-record/1) + SHA-256
#   sidecar. Evidence only -- no verdict fields.
# CONVENTION: logger.error + typed raise (V03PError). PASS(0)/ERROR(2).
#   No fallback, no mock, no placeholder, no silent pass.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from seqref_mri.src import free_flow_runtime as ffr
from seqref_mri.src import residual_decoder as dec  # noqa: F401  (map type)
from seqref_mri.src.fastmri_data import FastMRISliceDataset
from seqref_mri.scripts import train_free_flow as tff
from seqref_mri.scripts.train_base import _collate, _prepare
from seqref_mri.scripts.train_free_flow import targets_from_prepared
from seqref_mri.scripts.v02_manifests import BATCH_SIZE, canonical_json
from seqref_mri.scripts.v02_train import derive_cmap_from_mask
from seqref_mri.src.flows import v03_objective as v03l

logger = logging.getLogger("seqref_mri.v03_probe")

__version__ = "0.5"
__abbr__ = "SEQREF-V03P"

N_PROBE_BATCHES = 100                       # locked (SPEC SS9)
ALPHAS = (0.1, 0.5, 0.9)                    # locked (SPEC SS9)
NORM_FLOOR = 1e-12                          # locked (SPEC SS9.5)
LAMBDA_BAND = (1e-6, 1e6)                   # locked (SPEC SS9.5)
PROBE_WEIGHT_STATE = "init"                 # locked: registered fresh init
SCHEMA = "seqref-v03-probe-record/1"


class V03PError(RuntimeError):
    """Single typed identity for the V03 probe module."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V03PError(code, message)


# ---------------------------------------------------------------------------
# Pure estimator core (fixture f04 domain -- no data, no model)
# ---------------------------------------------------------------------------

def validate_norms(g_density: float, g_rec: float) -> None:
    """SPEC SS9.5 norm gates: zero / non-finite / < 1e-12 abort."""
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


def validate_lambda(lam: float, alpha: float) -> None:
    """SPEC SS9.5 lambda gates: non-finite / non-positive / out of band."""
    lo, hi = LAMBDA_BAND
    if not np.isfinite(lam):
        _fail("PROBE_VALIDITY_FAILURE",
              f"lambda for alpha={alpha} is non-finite ({lam!r}); "
              f"pre-run validity abort")
    if lam <= 0.0:
        _fail("PROBE_VALIDITY_FAILURE",
              f"lambda for alpha={alpha} is non-positive ({lam!r}); "
              f"pre-run validity abort")
    if not (lo <= lam <= hi):
        _fail("PROBE_VALIDITY_FAILURE",
              f"lambda for alpha={alpha} is {lam!r}, outside the locked "
              f"band [{lo}, {hi}]; pre-run validity abort")


def compute_lambdas(g_density: float, g_rec: float) -> dict:
    """lambda_k = (alpha_k/(1-alpha_k)) * (g_density/g_rec) per locked
    alpha grid, with the full SPEC SS9.5 validity chain. No clipping."""
    validate_norms(g_density, g_rec)
    ratio = g_density / g_rec
    out = {}
    for a in ALPHAS:
        lam = (a / (1.0 - a)) * ratio
        validate_lambda(lam, a)
        out[f"{a}"] = float(lam)
    return out


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
    """(n_trainable, n_with_grad) over model parameters -- telemetry for
    later lambda interpretation (which parameters each term reaches)."""
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    return n_train, n_grad


# ---------------------------------------------------------------------------
# Authoritative run
# ---------------------------------------------------------------------------

def _run(data_root: str, p4_stats2: str, out_dir: str) -> int:
    p4 = ffr.load_p4s2_parent(p4_stats2)
    logger.info("[%s] P4 /2 parent verified (file %s…)", __abbr__,
                p4["file_sha256"][:12])
    model = ffr.build_model()           # registered init; PROBE_WEIGHT_STATE
    model.train(False)
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

    g_d = np.empty(N_PROBE_BATCHES, dtype=np.float64)
    g_r = np.empty(N_PROBE_BATCHES, dtype=np.float64)
    for bi, batch in enumerate(loader):
        if bi >= N_PROBE_BATCHES:
            break
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
        tgt = torch.cat(targets, 0)
        cond = torch.cat(cond_ins, 0)
        msk = torch.cat(masks, 0)

        model.zero_grad(set_to_none=True)
        nll = tff.nll_objective(model, tgt, cond, msk)
        nll.backward()
        g_d[bi] = global_grad_norm(model)
        cov_d = grad_coverage(model)

        model.zero_grad(set_to_none=True)
        rec = v03l.l_rec(model, cond, msk, prep["y"], prep["amax"],
                         prep["tgt_norm"], maps)
        rec.backward()
        g_r[bi] = global_grad_norm(model)
        cov_r = grad_coverage(model)
        if bi == 0:
            cov_d0, cov_r0 = cov_d, cov_r
        elif cov_d != cov_d0 or cov_r != cov_r0:
            _fail("GRAD_COVERAGE_DRIFT",
                  f"batch {bi}: gradient coverage drifted from batch 0 "
                  f"({cov_d}/{cov_r} vs {cov_d0}/{cov_r0})")
        # NO optimizer step: the probe never updates parameters.
        if (bi + 1) % 10 == 0:
            logger.info("[%s] batch %d/%d: g_density=%.6e g_rec=%.6e",
                        __abbr__, bi + 1, N_PROBE_BATCHES, g_d[bi],
                        g_r[bi])

    mean_d = float(g_d.mean())
    mean_r = float(g_r.mean())
    lambdas = compute_lambdas(mean_d, mean_r)
    logger.info("[%s] means: g_density=%.6e g_rec=%.6e lambdas=%s",
                __abbr__, mean_d, mean_r, lambdas)

    record = {
        "schema": SCHEMA,
        "script": f"{__abbr__} v{__version__}",
        "spec": {"article": 163, "status": "FROZEN"},
        "plan": {"article": 164, "status": "FROZEN"},
        "weight_state": {"kind": PROBE_WEIGHT_STATE,
                         "model_init_seed": ffr.MODEL_INIT_SEED},
        "n_batches": N_PROBE_BATCHES,
        "batch_size": BATCH_SIZE,
        "split": "train", "mode": "eval", "role": "validation",
        "validation_source_amendment": "A2",
        "alphas": list(ALPHAS),
        "g_density_per_batch": g_d.tolist(),
        "g_rec_per_batch": g_r.tolist(),
        "g_density_mean": mean_d,
        "g_rec_mean": mean_r,
        "parameters": {"trainable": cov_d0[0],
                       "with_grad_density": cov_d0[1],
                       "with_grad_rec": cov_r0[1]},
        "lambdas": lambdas,
        "lambda_band": list(LAMBDA_BAND),
        "norm_floor": NORM_FLOOR,
        "l_rec_scale": v03l.L_REC_SCALE,
        "parents": {"p4_stats2": {"path": p4["path"],
                                  "file_sha256": p4["file_sha256"],
                                  "semantic_sha256": p4["semantic_sha256"]}},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "platform": platform.platform()},
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "v03_probe_record.json"
    path.write_bytes(canonical_json(record))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "v03_probe_record.json.sha256").write_text(
        f"{digest}  {path.name}\n")
    logger.info("[%s] probe record written: %s (sha256 %s…)", __abbr__,
                path, digest[:12])
    print(f"PASS {__abbr__} v{__version__} probe: lambdas={lambdas}")
    return 0


# ---------------------------------------------------------------------------
# Selftest: fixture f04 -- the six SPEC SS9.5 abort branches + happy path.
# Pure synthetic values; no data, no model, no holdout contact.
# ---------------------------------------------------------------------------

def _selftest() -> int:
    cases = [
        ("zero norm", lambda: compute_lambdas(0.0, 1.0)),
        ("NaN norm", lambda: compute_lambdas(1.0, float("nan"))),
        ("norm below 1e-12 floor", lambda: compute_lambdas(1.0, 5e-13)),
        ("lambda non-finite", lambda: compute_lambdas(1e300, 1e-11)),
        ("lambda below band", lambda: compute_lambdas(1.0, 1e12)),
        ("lambda above band", lambda: compute_lambdas(1e12, 1.0)),
        ("lambda non-positive", lambda: validate_lambda(-1.0, 0.5)),
    ]
    for name, fn in cases:
        try:
            fn()
        except V03PError as exc:
            if exc.code != "PROBE_VALIDITY_FAILURE":
                _fail("PROBE_SELFTEST_WRONG_CODE",
                      f"case {name!r} raised {exc.code}, expected "
                      f"PROBE_VALIDITY_FAILURE")
            logger.info("[%s] f04 branch PASS: %s", __abbr__, name)
        else:
            _fail("PROBE_SELFTEST_NO_ABORT",
                  f"case {name!r} did not abort; the SS9.5 gate is dead")
    got = compute_lambdas(2.0, 1.0)
    expect = {"0.1": (0.1 / 0.9) * 2.0, "0.5": 2.0, "0.9": 9.0 * 2.0}
    for k, v in expect.items():
        if abs(got[k] - v) > 1e-15 * max(1.0, abs(v)):
            _fail("PROBE_SELFTEST_WRONG_VALUE",
                  f"lambda[{k}]={got[k]!r} != expected {v!r}")
    logger.info("[%s] f04 happy path PASS: %s", __abbr__, got)

    # f04 telemetry: grad_coverage counts on a tiny synthetic module, and
    # the no-gradient guard.
    tiny = torch.nn.Linear(3, 2)          # 2 parameter tensors (W, b)
    if grad_coverage(tiny) != (2, 0):
        _fail("PROBE_SELFTEST_WRONG_VALUE",
              f"grad_coverage on fresh Linear(3,2) = {grad_coverage(tiny)}"
              f", expected (2, 0)")
    tiny(torch.randn(5, 3)).sum().backward()
    if grad_coverage(tiny) != (2, 2):
        _fail("PROBE_SELFTEST_WRONG_VALUE",
              "grad_coverage after backward != (2, 2)")
    try:
        global_grad_norm(torch.nn.Linear(3, 2))
    except V03PError as exc:
        if exc.code != "PROBE_VALIDITY_FAILURE":
            _fail("PROBE_SELFTEST_WRONG_CODE",
                  f"no-grad guard raised {exc.code}")
    else:
        _fail("PROBE_SELFTEST_NO_ABORT",
              "global_grad_norm accepted a gradient-free model")
    logger.info("[%s] f04 telemetry PASS: coverage + no-grad guard",
                __abbr__)
    print(f"PASS {__abbr__} v{__version__} selftest (f04: 7 abort branches"
          f" + happy path + telemetry)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="v03_probe")
    ap.add_argument("--mode", choices=("selftest", "run"), required=True)
    ap.add_argument("--data-root")
    ap.add_argument("--p4-stats2")
    ap.add_argument("--out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        if args.mode == "selftest":
            return _selftest()
        if args.mode == "run":
            missing = [k for k, v in (("data-root", args.data_root),
                                      ("p4-stats2", args.p4_stats2),
                                      ("out", args.out)) if not v]
            if missing:
                _fail("PROBE_ARGS_MISSING",
                      f"--mode run requires {missing}")
            return _run(args.data_root, args.p4_stats2, args.out)
    except V03PError as exc:
        logger.error("[%s] ERROR: %s", __abbr__, exc)
        return 2
    except v03l.V03LError as exc:
        logger.error("[%s] objective ERROR: %s", __abbr__, exc)
        return 2
    except Exception:
        logger.exception("[%s] unexpected exception", __abbr__)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# Changelog
# v0.1 (2026-10-06): initial implementation under V03SPEC v0.3 (163,
#   FROZEN) SS9 / V03PLAN v0.3 (164, FROZEN) SS6. 100-batch validation-only
#   probe (forward+backward, no updates); mean per-batch global L2 grad
#   norms; lambda conversion on the locked alpha grid with the full SS9.5
#   validity chain (typed PROBE_VALIDITY_FAILURE, no clipping); probe
#   record seqref-v03-probe-record/1 + SHA-256 sidecar; fixture f04 in
#   --mode selftest. Probe weight state locked to registered fresh init
#   (MODEL_INIT_SEED) and recorded in the record.
# v0.2 (2026-10-06): review strengthening (non-blocking reviewer item):
#   grad-coverage telemetry -- the record now carries trainable parameter
#   count and per-term gradient coverage (with_grad_density/with_grad_rec),
#   with a GRAD_COVERAGE_DRIFT abort if coverage changes across batches;
#   selftest gained coverage + no-grad-guard cases. No estimator change.
# v0.3 (2026-10-06): Amendment A2 alignment -- probe validation source
#   changed from split="val" to split="train" mode="eval": the official
#   val split IS the locked 199-volume holdout (V02E anchors, SPEC
#   SS14), so val-split probing would have been pre-evaluation holdout
#   contact. Record fields updated (split/role/amendment). Estimator,
#   gates and fixtures unchanged.
# v0.5 (2026-10-07): Amendment A6 alignment -- probe record gains
#   l_rec_scale (the pinned V03L constant) so the training-stage
#   contract can verify probe/train scale agreement. Estimator, gates,
#   and fixtures unchanged. (Note: this changelog has no v0.4 entry --
#   pre-existing documentation gap, left as found.)
# ---------------------------------------------------------------------------
