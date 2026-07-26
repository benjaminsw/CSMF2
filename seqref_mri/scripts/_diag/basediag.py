# =============================================================================
# SEQREF-BDIAG v0.5 -- scripts._diag.basediag
# LIFETIME: DIAGNOSTIC
# Purpose: shared no-retraining diagnostic over the THREE saved best-PSNR
#   base checkpoints (I4 NSF / I5 RealNVP / I6 NICE), per the post-I7-block
#   ruling. FINDINGS ONLY: no scientific acceptance gates; the script does
#   NOT choose a redesign. Hard failures are EXECUTION-VALIDITY gates only:
#     checkpoint/config mismatch (strict state-dict load) · wrong expert
#     architecture · missing/non-finite tensors · wrong subset size ·
#     checkpoint SHA mismatch at execution time · failure to reuse
#     identical z across controlled comparisons (z tensors hashed) ·
#     full-mask x0 ~= x_true invariant failure · helper-vs-independent
#     posterior-mean disagreement.
# Questions kept DISTINCT (EXEC section 5 BASEDIAG):
#   (1) estimator noise      n_post in {4,16,32,64} x repeated sampling
#                            seeds -> PSNR/SSIM mean AND spread
#   (2) conditioning use     decode(z, h_true) vs h_shuffled (fixed
#                            derangement per seed, no fixed points,
#                            permutation recorded) vs TWO zero variants:
#                            zero-h (decoder dependence on h) and
#                            zero-input (whole conditioning path) --
#                            IDENTICAL latent draws across all conditions
#   (3) latent vs cond влия  same h different z; same z different h
#   (4) invertibility        x->encode->decode AND z->decode->encode
#                            relative errors (max + median), dtype/device
#   (5) latent statistics    encode(x_norm) moments + ldj-path consistency
#                            (log_prob vs prior(z)+ldj recomputed) --
#                            NOT labelled "direction confirmed" from a
#                            single small round-trip
#   (6) full-mask Test-0     PRECISE construction: mask=ones; y REGENERATED
#                            = F(x_true) (never the undersampled y with a
#                            swapped mask); x0 = A^H y; same per-file
#                            normalization; VERIFY x0 ~= x_true (validity
#                            gate) BEFORE evaluating the model
#   (7) utility              posterior mean vs z=0 decode vs single sample
#                            vs zero-filled x0 (PSNR/SSIM/consistency)
# Distance statistics are SCALE-AWARE: per-sample relative L2
#   (||a-b||/max(||a||,eps)) AND absolute RMS, aggregated per sample first
#   -> mean/median/p5/p95 (a global norm can be dominated by high-energy
#   slices).
# Independent posterior mean (amendment 1): the primary path uses
#   train_base._posterior_mean; an in-file implementation re-seeds the SAME
#   generator state and reproduces the estimate with explicit
#   h=cond(...) / z=randn / decode(z,h); max rel disagreement must be
#   <= 1e-5 (validity gate) -- a shared-helper bug cannot pass unnoticed.
# Visualisation selection FIXED IN ADVANCE: first / middle / last frozen
#   subset positions (never chosen after seeing results).
# Recorded provenance: checkpoint epoch + cfg + sha256 per model, subset
#   indices, sampling seeds, permutation seeds, latent-tensor sha256s.
# Invocation: python -m seqref_mri.scripts._diag.basediag \
#     --data-root seqref_mri/data/fastmri \
#     --ckpt-nice <path> --ckpt-realnvp <path> --ckpt-nsf <path> [--smoke]
# Changelog (v0.4 -> v0.5, hook patch):
#   * Conditioner FORWARD HOOK registered per expert immediately after
#     load and removed before model deletion: gates conditioner inputs
#     AND outputs on EVERY invocation -- including the hidden calls
#     inside _posterior_mean() and model.log_prob() that the v0.4
#     "_condition_checked at every call site" claim missed, and any
#     future call added by refactoring. Coverage recorded in facts
#     ("conditioner_finite_hook_active": true).
#   * Tie rule genuinely ENFORCED, order-independent: best PSNR located
#     first, then the MINIMUM epoch among exact-tie rows; epochs
#     validated as unique integers. (v0.4 used max(), which returns the
#     first row in LIST order -- equivalent only if history order was
#     assumed.)
#   * h_zero finite-gated; visualisation vmax checked finite/positive
#     with display-only fallback 1.0 for an all-zero target.
#   * (post-review robustness) hook cleanup via try/finally; hook
#     INVOCATION COUNT recorded and asserted positive (registration
#     proven AND execution proven); vis outputs independently gated.
# Changelog (v0.3 -> v0.4, conditioner-gating patch):
#   * Conditioner activations finite-gated DIRECTLY via
#     _condition_checked() at EVERY model.cond() call site -- a non-finite
#     h feeding a decoder that ignores h would otherwise yield finite
#     outputs and a falsely clean "conditioning ignored" finding.
#   * Prepared batches validated once after _prepare() (cond_in, x_norm,
#     tgt_norm, amax, every y) for both the masked batch and Test-0.
#   * Best-selection hardening: every history row validated (epoch/val/
#     psnr present, PSNR finite) BEFORE argmax; tie rule recorded
#     explicitly (earliest epoch wins exact PSNR ties -- Python max()).
#   * Shuffled decodes computed ONCE and reused for gaps, metrics, and
#     paired differences (NSF runtime cut, results unchanged).
#   * Unused MaskedFourierOperator import removed.
# Changelog (v0.2 -> v0.3, final pre-smoke fixes):
#   * Subset sizes are CONSTANTS (formal 64 / smoke 8); --subset-size
#     removed -- a formal run cannot silently use another size.
#   * PSNR/SSIM finiteness-gated BEFORE serialisation (incl. aggregated
#     estimator-noise statistics).
#   * test0 check is strict: cfg["test0"] must be EXPLICITLY False.
#   * Test-0 interpretation SOFTENED (full-mask conditioning is outside
#     the training distribution): failure indicates lack of full-mask
#     identity behaviour under the current model/evaluation path; it does
#     NOT by itself distinguish conditioning failure from full-mask
#     distribution shift.
#   * Run facts.json SHA recorded and re-checked at execution time; full
#     64-char SHA-256 stored for checkpoints and evidence files (module
#     hashes stay 16-char prefixes, labelled as prefixes).
#   * Paired per-sample PSNR differences computed for ALL five
#     derangements, then summarised across permutations.
# Changelog (v0.1 -> v0.2, pre-run review fixes):
#   * NON-FINITE GATING made real: _require_finite() applied to loaded
#     parameters/buffers, posterior means, decoded outputs, latents, ldjs,
#     distance vectors, metric outputs, the full-mask invariant, and the
#     helper-agreement error. (v0.1's `rel > tol` comparisons were FALSE
#     for NaN -- the gates could not catch the pathology they existed for.)
#   * BEST-CHECKPOINT VERIFICATION: the sibling run facts.json is located
#     (or given via --facts-*), argmax(history[*].val.psnr) is recomputed,
#     and the checkpoint epoch + cfg must MATCH (recording alone is not
#     the predeclared control).
#   * LOCKED-3.15 CONFIG VALIDATION: every locked field (epochs 30,
#     8000/1000 slices, batch 8, lr 1e-4, seed 0, subset_seed 20260904,
#     widths, per-expert n_layers, NSF B, test0 false) is checked --
#     strict state-dict load proves shapes, not protocol compliance.
#   * Latent-reuse ASSERTED: every controlled branch re-hashes the z bank
#     against the canonical hash list (refactor protection).
#   * Independent posterior-mean cross-check at n_post = 4 AND max(grid).
#   * Sampling seeds are CAMPAIGN CONSTANTS (documented; no CLI flag) --
#     plan interface amended from the earlier --n-post-seeds proposal.
#   * Visual panels share one intensity range per sample
#     (vmin=0, vmax=p99.5 of the target) -- no per-panel autoscale.
#   * Sec-2 adds PAIRED per-sample PSNR differences (true−shuffled,
#     true−zero_h, true−zero_input): mean/median/p5/p95.
# Changelog (NEW in v0.1): Introduced (all seven review amendments folded
#   in pre-build).
# Update summary: one script, three checkpoints, findings only.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from seqref_mri.src.fastmri_data import FastMRISliceDataset
from seqref_mri.src.forward_operator import two_channel_to_complex
from seqref_mri.src.metrics import psnr_per_sample, ssim_per_sample
from seqref_mri.src.conditioner import Conditioner
from seqref_mri.src.base_experts import build_expert, CondNSF, _gaussian_logprob
from seqref_mri.scripts.train_base import (_collate, _prepare,
                                           _posterior_mean, DIM, IN_CHANNELS,
                                           NORMALIZED_DATA_RANGE, CELL_HW)

logger = logging.getLogger("seqref_mri.basediag")

__version__ = "0.5"
__abbr__ = "SEQREF-BDIAG"

SUBSET_SEED = 20260906
FORMAL_SUBSET_SIZE = 64
SMOKE_SUBSET_SIZE = 8
NPOST_GRID = (4, 16, 32, 64)
SAMPLING_SEEDS = (20260910, 20260911, 20260912, 20260913, 20260914)
PERM_SEEDS = (20260920, 20260921, 20260922, 20260923, 20260924)
ZBANK_SEED = 20260930
UTILITY_NPOST = 32
FULLMASK_NPOST = 16
EPS = 1e-12
EXPERTS = ("nice", "realnvp", "nsf")


def _fail(msg: str) -> None:
    logger.error("[bdiag] %s", msg)
    raise RuntimeError(msg)


def _require_finite(name: str, x) -> None:
    t = x if torch.is_tensor(x) else torch.as_tensor(x)
    if not torch.isfinite(t).all():
        _fail(f"{name}: non-finite value detected (validity gate)")


def _condition_checked(model, cond_in: torch.Tensor, label: str
                       ) -> torch.Tensor:
    # EVERY conditioner call goes through here: the conditioner is the
    # object under investigation, so its own activations are gated.
    _require_finite(f"{label} conditioner input", cond_in)
    h = model.cond(cond_in)
    _require_finite(f"{label} conditioner output h", h)
    return h


class _CondFiniteHook:
    # Counts invocations so facts can prove the hook RAN, not merely that
    # it was registered.
    def __init__(self):
        self.calls = 0

    def __call__(self, module, inputs, output):
        self.calls += 1
        for i, inp in enumerate(inputs):
            _require_finite(f"conditioner hook input[{i}]", inp)
        _require_finite("conditioner hook output", output)


def _validate_prepared(p: dict, label: str) -> None:
    for key in ("cond_in", "x_norm", "tgt_norm", "amax"):
        _require_finite(f"{label} prepared {key}", p[key])
    for i, y in enumerate(p["y"]):
        _require_finite(f"{label} prepared y[{i}]", y)


def _sha(path: Path) -> str:
    # 16-char PREFIX (module provenance display)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _sha_full(path: Path) -> str:
    # full 64-char SHA-256 (checkpoints + evidence files)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def provenance(argv, *, allow_dirty: bool) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"],
                                capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True,
                                    check=True).stdout.strip())
    except Exception as e:
        logger.error("[bdiag] git provenance unobtainable: %r", e)
        raise RuntimeError(f"git provenance unobtainable: {e!r}") from e
    if dirty and not allow_dirty:
        _fail("working tree DIRTY -- commit before the formal BASEDIAG run")
    if dirty:
        logger.warning("[bdiag] DIRTY TREE PERMITTED (smoke): PROVISIONAL "
                       "-- NOT FORMAL EVIDENCE")
    import seqref_mri.src.base_experts as be
    import seqref_mri.src.conditioner as co
    import seqref_mri.src.metrics as me
    import seqref_mri.src.fastmri_data as fd
    import seqref_mri.src.forward_operator as fo
    import seqref_mri.scripts.train_base as tb
    hashes = {Path(m.__file__).name: _sha(Path(m.__file__))
              for m in (be, co, me, fd, fo, tb, sys.modules[__name__])}
    return {"git_commit": commit, "git_dirty": dirty, "argv": argv,
            "script_sha256": hashes, "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available()}


LOCKED_315 = {"epochs": 30, "train_slices": 8000, "val_slices": 1000,
              "batch": 8, "lr": 1e-4, "seed_index": 0,
              "subset_seed": 20260904, "cond_width": 64, "h_dim": 128,
              "hidden": 256, "n_post": 4}
LOCKED_LAYERS = {"nice": 4, "realnvp": 6, "nsf": 6}
LOCKED_NSF_B = 0.5790575438201858


def _verify_locked_cfg(expert: str, cfg: dict) -> None:
    for k, v in LOCKED_315.items():
        if cfg.get(k) != v:
            _fail(f"{expert}: cfg[{k!r}]={cfg.get(k)!r} != locked 3.15 "
                  f"value {v!r} (validity gate)")
    if cfg.get("n_layers") != LOCKED_LAYERS[expert]:
        _fail(f"{expert}: n_layers {cfg.get('n_layers')!r} != locked "
              f"{LOCKED_LAYERS[expert]}")
    if cfg.get("test0", None) is not False:
        _fail(f"{expert}: cfg test0 must be EXPLICITLY False (locked "
              "masked-run field), got {!r}".format(cfg.get("test0", None)))
    if expert == "nsf" and cfg.get("nsf_B") != LOCKED_NSF_B:
        _fail(f"nsf: B {cfg.get('nsf_B')!r} != locked {LOCKED_NSF_B!r}")


def _verify_best_selection(expert: str, ckpt_path: Path, ckpt_cfg: dict,
                           ckpt_epoch: int, facts_path: str | None) -> dict:
    fp = Path(facts_path) if facts_path else ckpt_path.parent / "facts.json"
    if not fp.is_file():
        _fail(f"{expert}: run facts.json not found at {fp} -- needed to "
              "VERIFY best-PSNR checkpoint selection (--facts-{expert} to "
              "override)")
    facts_sha = _sha_full(fp)
    run_facts = json.loads(fp.read_text())
    hist = run_facts.get("history")
    if not hist:
        _fail(f"{expert}: run facts lack history")
    for i, row in enumerate(hist):
        if ("epoch" not in row or "val" not in row
                or "psnr" not in row["val"]):
            _fail(f"{expert}: malformed history row {i}")
        _require_finite(f"{expert} history[{i}] val PSNR",
                        row["val"]["psnr"])
    epochs = [row["epoch"] for row in hist]
    if (len(set(epochs)) != len(epochs)
            or not all(isinstance(e, int) for e in epochs)):
        _fail(f"{expert}: history epochs not unique integers")
    # tie rule, ENFORCED order-independently: EARLIEST epoch wins exact
    # PSNR ties (matches the trainer, which replaces best.pt only on
    # strict improvement).
    best_psnr = max(row["val"]["psnr"] for row in hist)
    best_ep = min(row["epoch"] for row in hist
                  if row["val"]["psnr"] == best_psnr)
    if ckpt_epoch != best_ep:
        _fail(f"{expert}: checkpoint epoch {ckpt_epoch} != recomputed "
              f"best-PSNR epoch {best_ep} (selection-rule violation)")
    if run_facts.get("cfg") != ckpt_cfg:
        _fail(f"{expert}: checkpoint cfg differs from the run facts cfg")
    if run_facts["cfg"].get("expert") != expert:
        _fail(f"{expert}: run facts belong to {run_facts['cfg'].get('expert')!r}")
    if _sha_full(fp) != facts_sha:
        _fail(f"{expert}: run facts.json changed during execution")
    return {"facts_path": str(fp), "facts_sha256": facts_sha,
            "verified_best_epoch": best_ep,
            "best_val_psnr": max(e["val"]["psnr"] for e in hist)}


def load_checkpoint(expert: str, path: str, device: str,
                    facts_path: str | None) -> tuple:
    p = Path(path)
    if not p.is_file():
        _fail(f"{expert}: checkpoint missing at {path}")
    sha = _sha_full(p)
    blob = torch.load(p, map_location="cpu")
    for key in ("model", "cfg", "epoch"):
        if key not in blob:
            _fail(f"{expert}: checkpoint lacks '{key}'")
    cfg = blob["cfg"]
    if cfg.get("expert") != expert:
        _fail(f"{expert}: checkpoint cfg says expert={cfg.get('expert')!r} "
              "-- wrong architecture (validity gate)")
    _verify_locked_cfg(expert, cfg)
    sel = _verify_best_selection(expert, p, cfg, blob["epoch"], facts_path)
    cond = Conditioner(in_channels=IN_CHANNELS, width=cfg["cond_width"],
                       h_dim=cfg["h_dim"])
    if expert == "nsf":
        model = CondNSF(dim=DIM, h_dim=cfg["h_dim"], conditioner=cond,
                        hidden=cfg["hidden"], n_layers=cfg["n_layers"],
                        K=8, B=float(cfg["nsf_B"]), use_film=True)
    else:
        model = build_expert(expert, dim=DIM, h_dim=cfg["h_dim"],
                             conditioner=cond, hidden=cfg["hidden"],
                             use_film=True, n_layers=cfg["n_layers"])
    try:
        model.load_state_dict(blob["model"], strict=True)
    except Exception as e:
        _fail(f"{expert}: strict state-dict load FAILED (checkpoint/config "
              f"mismatch): {e!r}")
    model = model.to(device).eval()
    for pname, t in list(model.named_parameters()) + list(model.named_buffers()):
        _require_finite(f"{expert} param/buffer {pname}", t)
    # SHA re-check at execution time (validity gate)
    if _sha_full(p) != sha:
        _fail(f"{expert}: checkpoint SHA changed during execution")
    return model, {"path": str(p), "sha256": sha,
                   "selected_epoch": blob["epoch"], "cfg": cfg,
                   "best_selection_verified": sel}


def _dist_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    # Per-sample first (amendment 5): rel L2 + abs RMS, then aggregate.
    _require_finite("dist input a", a)
    _require_finite("dist input b", b)
    fa, fb = a.flatten(1), b.flatten(1)
    rel = (torch.linalg.vector_norm(fa - fb, dim=1)
           / torch.clamp(torch.linalg.vector_norm(fa, dim=1), min=EPS))
    rms = ((fa - fb) ** 2).mean(dim=1).sqrt()
    def agg(t):
        v = t.detach().cpu().numpy()
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "p5": float(np.percentile(v, 5)),
                "p95": float(np.percentile(v, 95))}
    _require_finite("dist rel_l2", rel)
    _require_finite("dist abs_rms", rms)
    return {"rel_l2": agg(rel), "abs_rms": agg(rms)}


def _metrics_of_mean(xm_flat: torch.Tensor, p: dict) -> dict:
    _require_finite("reconstruction", xm_flat)
    xm_c = two_channel_to_complex(xm_flat.view(-1, 2, CELL_HW, CELL_HW))
    mag = xm_c.abs().unsqueeze(1).cpu()
    tgt = p["tgt_norm"].cpu()
    cons = [float(p["ops"][i].consistency(
        (xm_c[i] * p["amax"][i]), p["y"][i])) for i in range(len(p["ops"]))]
    _require_finite("consistency", torch.tensor(cons))
    psnr = psnr_per_sample(mag, tgt, data_range=NORMALIZED_DATA_RANGE)
    ssim = ssim_per_sample(mag, tgt, data_range=NORMALIZED_DATA_RANGE)
    _require_finite("PSNR", psnr)
    _require_finite("SSIM", ssim)
    return {"psnr": float(psnr.mean()), "ssim": float(ssim.mean()),
            "consistency_mean": float(np.mean(cons))}


def _psnr_per_sample_of(xm_flat: torch.Tensor, p: dict) -> torch.Tensor:
    _require_finite("reconstruction", xm_flat)
    xm_c = two_channel_to_complex(xm_flat.view(-1, 2, CELL_HW, CELL_HW))
    vals = psnr_per_sample(xm_c.abs().unsqueeze(1).cpu(), p["tgt_norm"].cpu(),
                           data_range=NORMALIZED_DATA_RANGE)
    _require_finite("psnr per-sample", vals)
    return vals


def _seeded_pm_helper(model, cond_in, n_post: int, seed: int):
    torch.manual_seed(seed)
    return _posterior_mean(model, cond_in, n_post)


def _seeded_pm_independent(model, cond_in, n_post: int, seed: int):
    # Amendment 1: explicit local implementation with the SAME generator
    # state and draw order as the helper -- identical z sequence.
    torch.manual_seed(seed)
    h = _condition_checked(model, cond_in, "independent-pm")
    acc, z_hashes = None, []
    for _ in range(n_post):
        z = torch.randn(cond_in.shape[0], model.dim,
                        device=cond_in.device, dtype=cond_in.dtype)
        z_hashes.append(_tensor_sha(z))
        x = model.decode(z, h)
        acc = x if acc is None else acc + x
    return acc / n_post, z_hashes


def sec1_estimator_noise(model, p: dict) -> dict:
    out = {"per_n_post": {}}
    for n_post in NPOST_GRID:
        psnrs, ssims = [], []
        for seed in SAMPLING_SEEDS:
            xm = _seeded_pm_helper(model, p["cond_in"], n_post, seed)
            m = _metrics_of_mean(xm, p)
            psnrs.append(m["psnr"]); ssims.append(m["ssim"])
        _require_finite("estimator-noise psnr stats", torch.tensor(psnrs))
        _require_finite("estimator-noise ssim stats", torch.tensor(ssims))
        out["per_n_post"][n_post] = {
            "psnr": {"per_seed": psnrs, "mean": float(np.mean(psnrs)),
                     "std": float(np.std(psnrs)),
                     "min": float(np.min(psnrs)), "max": float(np.max(psnrs))},
            "ssim": {"per_seed": ssims, "mean": float(np.mean(ssims)),
                     "std": float(np.std(ssims))}}
    # validity gate: helper vs independent agreement at n_post=4 AND the
    # grid maximum (loop-count / accumulation errors caught at the top end)
    out["helper_vs_independent"] = {}
    for n_post in (NPOST_GRID[0], NPOST_GRID[-1]):
        xm_h = _seeded_pm_helper(model, p["cond_in"], n_post,
                                 SAMPLING_SEEDS[0])
        xm_i, z_hashes = _seeded_pm_independent(model, p["cond_in"], n_post,
                                                SAMPLING_SEEDS[0])
        _require_finite("helper posterior mean", xm_h)
        _require_finite("independent posterior mean", xm_i)
        rel = (torch.linalg.vector_norm(xm_h - xm_i)
               / torch.clamp(torch.linalg.vector_norm(xm_h), min=EPS))
        _require_finite("helper-agreement rel", rel)
        if rel.item() > 1e-5:
            _fail(f"helper vs independent posterior-mean disagreement "
                  f"rel={rel.item():.3e} at n_post={n_post}")
        out["helper_vs_independent"][n_post] = {
            "rel": rel.item(), "independent_z_sha256": z_hashes}
    return out


def _derangement(n: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def sec2_conditioning_use(model, p: dict) -> dict:
    B = p["cond_in"].shape[0]
    torch.manual_seed(ZBANK_SEED)
    z_bank = [torch.randn(B, model.dim, device=p["cond_in"].device)
              for _ in range(FULLMASK_NPOST)]
    z_bank_sha = [_tensor_sha(z) for z in z_bank]
    h_true = _condition_checked(model, p["cond_in"], "sec2-true")
    h_zero = torch.zeros_like(h_true)                       # zero-h
    _require_finite("sec2 h_zero", h_zero)
    h_zin = _condition_checked(model, torch.zeros_like(p["cond_in"]),
                               "sec2-zero-input")           # zero-input
    out = {"z_bank_sha256": z_bank_sha, "perm_seeds": list(PERM_SEEDS),
           "gaps": {}, "metrics": {}}

    def pm(h):
        acc = None
        for zi, z in enumerate(z_bank):                     # IDENTICAL z
            if _tensor_sha(z) != z_bank_sha[zi]:            # ASSERTED reuse
                _fail("z-bank hash mismatch -- latent reuse violated "
                      "(validity gate)")
            x = model.decode(z, h)
            _require_finite("decoded output", x)
            acc = x if acc is None else acc + x
        return acc / len(z_bank)

    xm_true = pm(h_true)
    out["metrics"]["true"] = _metrics_of_mean(xm_true, p)
    # shuffled: one fixed derangement per perm seed; each decoded ONCE
    # and reused for gaps, metrics, and paired differences
    shuf_gaps, perms, xm_shuf_all = [], [], []
    for ps in PERM_SEEDS:
        perm = _derangement(B, ps)
        perms.append(perm.tolist())
        xm_shuf = pm(h_true[torch.from_numpy(perm).to(h_true.device)])
        xm_shuf_all.append(xm_shuf)
        shuf_gaps.append(_dist_stats(xm_true, xm_shuf))
    out["permutations"] = perms
    out["gaps"]["true_vs_shuffled_per_perm"] = shuf_gaps
    out["metrics"]["shuffled_perm0"] = _metrics_of_mean(xm_shuf_all[0], p)
    xm_variants: dict = {}
    for tag, h in (("zero_h", h_zero), ("zero_input", h_zin)):
        xm = pm(h)
        out["gaps"][f"true_vs_{tag}"] = _dist_stats(xm_true, xm)
        out["metrics"][tag] = _metrics_of_mean(xm, p)
        xm_variants[tag] = xm
    # PAIRED per-sample PSNR differences (more diagnostic than aggregate
    # deltas): true − variant, per sample, then mean/median/p5/p95.
    # Shuffled: computed for ALL five derangements, then summarised across
    # permutations so no single arbitrary derangement dominates.
    def _pair(d: np.ndarray) -> dict:
        return {"mean": float(d.mean()), "median": float(np.median(d)),
                "p5": float(np.percentile(d, 5)),
                "p95": float(np.percentile(d, 95))}
    ps_true = _psnr_per_sample_of(xm_true, p)
    out["paired_psnr_diff"] = {}
    per_perm_means = []
    per_perm = {}
    for k, xm in enumerate(xm_shuf_all):
        d = (ps_true - _psnr_per_sample_of(xm, p)).numpy()
        per_perm[f"perm{k}"] = _pair(d)
        per_perm_means.append(float(d.mean()))
    out["paired_psnr_diff"]["true_minus_shuffled"] = {
        "per_perm": per_perm,
        "across_perms_mean": float(np.mean(per_perm_means)),
        "across_perms_std": float(np.std(per_perm_means))}
    for tag, xm in xm_variants.items():
        d = (ps_true - _psnr_per_sample_of(xm, p)).numpy()
        out["paired_psnr_diff"][f"true_minus_{tag}"] = _pair(d)
    return out


def sec3_crossings(model, p: dict) -> dict:
    B = p["cond_in"].shape[0]
    h = _condition_checked(model, p["cond_in"], "sec3")
    torch.manual_seed(ZBANK_SEED + 1)
    z1 = torch.randn(B, model.dim, device=h.device)
    z2 = torch.randn(B, model.dim, device=h.device)
    perm = torch.from_numpy(_derangement(B, PERM_SEEDS[0])).to(h.device)
    return {"same_h_diff_z": _dist_stats(model.decode(z1, h),
                                         model.decode(z2, h)),
            "same_z_diff_h": _dist_stats(model.decode(z1, h),
                                         model.decode(z1, h[perm])),
            "z_sha256": [_tensor_sha(z1), _tensor_sha(z2)]}


def sec45_invertibility_latent(model, p: dict, device: str) -> dict:
    x = p["x_norm"].flatten(1)
    h = _condition_checked(model, p["cond_in"], "sec45")
    z, ldj = model.encode(x, h)
    x_rt = model.decode(z, h)
    rel_x = (torch.linalg.vector_norm(x_rt - x, dim=1)
             / torch.clamp(torch.linalg.vector_norm(x, dim=1), min=EPS))
    torch.manual_seed(ZBANK_SEED + 2)
    z0 = torch.randn_like(z)
    x_dec = model.decode(z0, h)
    z_rt, _ = model.encode(x_dec, h)
    rel_z = (torch.linalg.vector_norm(z_rt - z0, dim=1)
             / torch.clamp(torch.linalg.vector_norm(z0, dim=1), min=EPS))
    # ldj-path consistency: model.log_prob vs prior(z)+ldj recomputed here
    lp_model = model.log_prob(x, p["cond_in"])
    lp_recon = _gaussian_logprob(z) + ldj
    lp_diff = (lp_model - lp_recon).abs()
    for name, t in (("x_encode_decode rel", rel_x),
                    ("z_decode_encode rel", rel_z),
                    ("logprob diff", lp_diff)):
        _require_finite(name, t)
    zf = z.flatten()
    for name, t in (("x", x), ("z", z), ("ldj", ldj)):
        if not torch.isfinite(t).all():
            _fail(f"non-finite {name} in invertibility section")
    return {"x_encode_decode_rel": {"max": float(rel_x.max()),
                                    "median": float(rel_x.median())},
            "z_decode_encode_rel": {"max": float(rel_z.max()),
                                    "median": float(rel_z.median())},
            "logprob_vs_prior_plus_ldj_abs": {"max": float(lp_diff.max()),
                                              "median": float(lp_diff.median())},
            "latent_stats_encode_x": {
                "mean": float(zf.mean()), "std": float(zf.std()),
                "per_dim_absmean_max": float(z.mean(0).abs().max()),
                "per_dim_std_min": float(z.std(0).min()),
                "per_dim_std_max": float(z.std(0).max()),
                "frac_abs_gt4": float((zf.abs() > 4).float().mean())},
            "dtype": str(x.dtype), "device": device,
            "note": "invertibility measured BOTH directions + ldj-path "
                    "consistency; not labelled 'direction confirmed' from "
                    "a single round-trip"}


def sec6_fullmask(model, batch: dict, device: str) -> dict:
    # Amendment 2 precise construction via _prepare(test0=True):
    # mask=ones, y REGENERATED = F(x_true), x0 = A^H y, same normalization.
    p0 = _prepare(batch, device, test0=True)
    _validate_prepared(p0, "test0")
    x0_state = p0["cond_in"]                    # = normalized [Re,Im](x0)
    inv = (torch.linalg.vector_norm(
              (x0_state - p0["x_norm"]).flatten(1), dim=1)
           / torch.clamp(torch.linalg.vector_norm(
              p0["x_norm"].flatten(1), dim=1), min=EPS))
    _require_finite("full-mask invariant", inv)
    if float(inv.max()) > 1e-4:
        _fail(f"full-mask invariant x0 ~= x_true FAILED: max rel "
              f"{float(inv.max()):.3e} (validity gate)")
    xm = _seeded_pm_helper(model, p0["cond_in"], FULLMASK_NPOST,
                           SAMPLING_SEEDS[0])
    return {"invariant_x0_eq_xtrue_max_rel": float(inv.max()),
            "posterior_mean_vs_exact_answer": _metrics_of_mean(xm, p0),
            "note": "Failure indicates lack of full-mask identity "
                    "behaviour under the current model and evaluation "
                    "path; it does NOT by itself distinguish conditioning "
                    "failure from full-mask distribution shift (full-mask "
                    "conditioning is outside the training distribution)."}


def sec7_utility(model, p: dict) -> dict:
    xm = _seeded_pm_helper(model, p["cond_in"], UTILITY_NPOST,
                           SAMPLING_SEEDS[0])
    h = _condition_checked(model, p["cond_in"], "sec7")
    x_z0 = model.decode(torch.zeros(p["cond_in"].shape[0], model.dim,
                                    device=p["cond_in"].device), h)
    torch.manual_seed(ZBANK_SEED + 3)
    x_one = model.decode(torch.randn(p["cond_in"].shape[0], model.dim,
                                     device=p["cond_in"].device), h)
    x0_flat = p["cond_in"].flatten(1)           # zero-filled state (norm)
    return {"posterior_mean_npost32": _metrics_of_mean(xm, p),
            "z0_decode": _metrics_of_mean(x_z0, p),
            "single_sample": _metrics_of_mean(x_one, p),
            "zero_filled_x0": _metrics_of_mean(x0_flat, p)}


def vis_panels(model, expert: str, batch: dict, p: dict, subset_idx,
               plots_dir: Path, device: str) -> list[str]:
    # FIXED positions: first / middle / last of the frozen subset.
    positions = [0, len(subset_idx) // 2, len(subset_idx) - 1]
    xm = _seeded_pm_helper(model, p["cond_in"], UTILITY_NPOST,
                           SAMPLING_SEEDS[0])
    h = _condition_checked(model, p["cond_in"], "vis")
    xz0 = model.decode(torch.zeros(p["cond_in"].shape[0], model.dim,
                                   device=device), h)
    _require_finite("visual posterior mean", xm)
    _require_finite("visual z0 decode", xz0)
    written = []
    for pos in positions:
        panels = [
            (p["tgt_norm"][pos, 0].cpu(), "target"),
            (two_channel_to_complex(
                p["cond_in"][pos:pos+1]).abs()[0].cpu(), "|x0| zero-filled"),
            (two_channel_to_complex(
                xm.view(-1, 2, CELL_HW, CELL_HW)[pos:pos+1]
             ).abs()[0].cpu(), "posterior mean"),
            (two_channel_to_complex(
                xz0.view(-1, 2, CELL_HW, CELL_HW)[pos:pos+1]
             ).abs()[0].cpu(), "z=0 decode"),
        ]
        tgt_img = panels[0][0]
        vmax = float(torch.quantile(tgt_img.flatten(), 0.995))
        if not np.isfinite(vmax) or vmax <= 0.0:
            logger.warning("[bdiag] vis vmax %r invalid -- display-only "
                           "fallback 1.0", vmax)
            vmax = 1.0
        vmin = 0.0
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
        for ax, (img, title) in zip(axes, panels):
            ax.imshow(img.numpy(), cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"{expert}: {title}", fontsize=8)
            ax.axis("off")
        f = plots_dir / f"{expert}_pos{pos:02d}.png"
        fig.savefig(f, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(f.name)
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt-nice", required=True)
    ap.add_argument("--ckpt-realnvp", required=True)
    ap.add_argument("--ckpt-nsf", required=True)
    ap.add_argument("--facts-nice", default=None)
    ap.add_argument("--facts-realnvp", default=None)
    ap.add_argument("--facts-nsf", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(a.out) if a.out else (
        Path(a.data_root).parent.parent / "results" / "_diag" / "basediag")
    tmp = out_dir.parent / (out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "plots").mkdir(parents=True)

    global NPOST_GRID, SAMPLING_SEEDS
    # Subset sizes are CONSTANTS: formal runs cannot use another size.
    subset_size = SMOKE_SUBSET_SIZE if a.smoke else FORMAL_SUBSET_SIZE
    if a.smoke:
        NPOST_GRID = (4, 16)
        SAMPLING_SEEDS = SAMPLING_SEEDS[:2]

    facts: dict = {"script": f"{__abbr__} v{__version__} basediag",
                   "smoke": a.smoke,
                   "constants": {"subset_seed": SUBSET_SEED,
                                 "subset_size": subset_size,
                                 "n_post_grid": list(NPOST_GRID),
                                 "sampling_seeds": list(SAMPLING_SEEDS),
                                 "perm_seeds": list(PERM_SEEDS),
                                 "z_bank_seed": ZBANK_SEED}}
    facts["provenance"] = provenance(sys.argv, allow_dirty=a.smoke)
    if facts["provenance"]["git_dirty"]:
        facts["EVIDENCE_STATUS"] = ("PROVISIONAL -- dirty tree permitted "
                                    "for smoke; NOT formal evidence")

    ds = FastMRISliceDataset(a.data_root, split="val", mode="eval")
    rng = np.random.Generator(np.random.PCG64(SUBSET_SEED))
    subset_idx = sorted(int(i) for i in
                        rng.choice(len(ds), size=subset_size, replace=False))
    facts["subset_indices"] = subset_idx
    items = [ds[i] for i in subset_idx]
    if len(items) != subset_size:
        _fail("subset size mismatch (validity gate)")
    batch = _collate(items)
    p = _prepare(batch, device, test0=False)
    _validate_prepared(p, "masked")

    ckpts = {"nice": a.ckpt_nice, "realnvp": a.ckpt_realnvp,
             "nsf": a.ckpt_nsf}
    facts_paths = {"nice": a.facts_nice, "realnvp": a.facts_realnvp,
                   "nsf": a.facts_nsf}
    facts["experts"] = {}
    with torch.no_grad():
        for name in EXPERTS:
            logger.info("[bdiag] ===== %s =====", name)
            model, ck = load_checkpoint(name, ckpts[name], device,
                                        facts_paths[name])
            hook_fn = _CondFiniteHook()
            hook = model.cond.register_forward_hook(hook_fn)
            rec = {"checkpoint": ck,
                   "conditioner_finite_hook_active": True}
            try:
                rec["s1_estimator_noise"] = sec1_estimator_noise(model, p)
                rec["s2_conditioning_use"] = sec2_conditioning_use(model, p)
                rec["s3_crossings"] = sec3_crossings(model, p)
                rec["s45_invertibility_latent"] = \
                    sec45_invertibility_latent(model, p, device)
                rec["s6_fullmask_test0"] = sec6_fullmask(model, batch,
                                                         device)
                rec["s7_utility"] = sec7_utility(model, p)
                rec["plots"] = vis_panels(model, name, batch, p,
                                          subset_idx, tmp / "plots",
                                          device)
            finally:
                hook.remove()
            if hook_fn.calls <= 0:
                _fail(f"{name}: conditioner hook never invoked -- "
                      "coverage claim unsupported")
            rec["conditioner_finite_hook_calls"] = hook_fn.calls
            facts["experts"][name] = rec
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    facts["verdict"] = ("FINDINGS RECORDED (smoke, provisional)" if a.smoke
                        else "FINDINGS RECORDED")
    with open(tmp / "facts.json", "w") as f:
        json.dump(facts, f, indent=2)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    logger.info("[bdiag] %s -- report at %s", facts["verdict"], out_dir)


if __name__ == "__main__":
    main()
