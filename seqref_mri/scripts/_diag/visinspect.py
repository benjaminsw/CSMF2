# =============================================================================
# SEQREF-VINS v0.4 -- scripts._diag.visinspect
# LIFETIME: DIAGNOSTIC
# Purpose: DESCRIPTIVE fixed-panel reconstruction inspection (NO gate) to
#   shape the residual-base design. Produces evidence only; the taxonomy
#   classification (blur / noise-averaging / intensity error /
#   hallucination / lost measured detail / phase artefacts /
#   slice-concentrated failure) is done by the humans on the panels.
# Reconstructions compared (FIVE; target is a reference, not a recon):
#   x0 (zero-filled) · NICE-ep29 posterior mean · NICEXT-ep57 posterior
#   mean · NICE-ep29 z=0 · NICEXT-ep57 z=0.
# Slice selection (deterministic, PRE-REGISTERED; no visual choice):
#   priority order = first / middle / last frozen positions, then
#   argmax x0 PSNR, then max and min of the SIGNED paired
#   (ep29-PM PSNR − x0 PSNR). DEDUP RULE: walk in priority order; on a
#   duplicate, take the next-ranked unused slice under the SAME
#   criterion until six unique slices exist. Original candidates AND
#   final positions both recorded. The max signed diff is labelled
#   "least-degraded NICE case" unless the value is actually positive.
# Latents: ONE z bank (seed 20260910, n_post 32) generated once, reused
#   for BOTH checkpoints (hashes recorded) -- visual differences are
#   attributable to weights, not sampling noise. z=0 identical trivially.
# Subset: the EXACT formal BASEDIAG indices, loaded from its facts.json
#   (--basediag-facts) and verified: smoke==false, git_dirty==false,
#   subset_size==64, subset_seed==20260906. A seed is a recipe, not a
#   record -- the recorded indices are the authoritative BASEDIAG positions.
# Checkpoints: SHA-declared and verified (--sha-nice / --sha-nicext).
#   NICE ep29: locked-3.15 cfg, epoch 29. NICEXT ep57: epoch 57,
#   init.init_from_epoch==29, init.init_from_sha256==sha-nice,
#   declared_sha_verified==true, locked fields equal except the declared
#   continuation differences (epochs).
# Error maps: magnitude-domain | |x_recon| − |x_target| | (matches the
#   PSNR/SSIM display domain); per slice, ONE shared error scale
#   (vmin 0, vmax p99.5 across the FOUR error maps), finite/positive
#   fallback. Image panels share vmin 0 / vmax p99.5 of the target.
# Layout per slice (2×5): target | x0 | ep29 PM | ep57 PM | ep29 z0
#                         ep57 z0 | x0 err | ep29 PM err | ep57 PM err
#                         | ep57 z0 err
# facts.json: per-slice metrics for the five reconstructions on the
#   selected slices, per-slice x0/ep29-PM PSNR over all 64 (criteria
#   inputs), ep57−ep29 deltas (PM and z0) on selected slices, selection
#   record, z-bank hashes, checkpoint SHAs, provenance.
# CONVENTION: logger.error + raise; smoke permits dirty tree
#   (PROVISIONAL); no fallback.
# Changelog (v0.3 -> v0.4, final evidence-integrity fixes):
#   * --out hardened: smoke may not target the formal directory; an
#     explicit --out refuses to overwrite ANY existing directory; only
#     the default smoke directory is auto-replaceable. "Smoke cannot
#     erase formal evidence" now holds unconditionally.
#   * NICEXT competence RECOMPUTED from history (floors 31.53/0.691,
#     per-row finite PSNR AND SSIM, qualifying epochs re-derived) and
#     required to equal the recorded competence block exactly --
#     verification of the declaration, not trust in its summary. (The
#     v0.3 check also silently permitted a missing qualifying_epochs
#     field via None-falsiness.)
#   * Source-facts TOCTOU closed: bytes read ONCE, hashed and parsed
#     from the same buffer (both BASEDIAG and NICEXT facts).
#   * load_basediag annotation fixed (3-tuple); manifest wording made
#     precise: an ORDERED SUBSET manifest hash (portable relative-path
#     identifiers), not a whole-dataset fingerprint.
#   * (post-review) Manifest record is a STRUCTURED block (n_slices as
#     data, not key-name semantics) -- the hardcoded "64slice" key would
#     have lied during smoke (8 slices), the same record/reality bug
#     class fixed earlier for n_post. constants record both
#     basediag_subset_size (64) and active_subset_size.
#     slice_identifiers disambiguated: subset_position vs dataset_index.
#   * (pre-formal fixes per consolidated review) All three selection
#     argsorts use kind="stable" -- tie order was NumPy-version-
#     dependent, contradicting the pre-registered determinism claim.
#     Git provenance pinned to the repository containing THIS script
#     (git -C repo_root, derived from __file__ with a loud sanity
#     check); output base likewise derived from __file__, independent
#     of data-root layout. 2×5 four-error-map layout CONSCIOUSLY
#     CONFIRMED (ep29 z=0 metrics live in facts; supplementary panel
#     only if the first inspection makes the z=0 transition central).
#     torch.load FutureWarning accepted and recorded for this run.
# Changelog (v0.2 -> v0.3, evidence-integrity fixes):
#   * Output separation: smoke writes to visinspect_smoke/, formal to
#     visinspect/; a formal run REFUSES to overwrite existing formal
#     output (delete manually to rerun) -- smoke can no longer erase
#     formal evidence.
#   * Dataset-identity honesty: indices are POSITIONS, not identities.
#     The script records each loaded slice's stable (relative file,
#     slice_index) identifier plus a 64-slice ORDERED SUBSET manifest
#     sha256 (portable across data-root relocation; NOT a whole-dataset
#     fingerprint). LIMITATION
#     stated in facts: BASEDIAG recorded only indices, so exact index
#     reuse is verified, but dataset-order identity at BASEDIAG time
#     cannot be independently proven from the available record. (The
#     v0.1 reordering-protection claim is WITHDRAWN.)
#   * NICEXT facts BOUND to the checkpoint: facts must be facts.json in
#     the SAME run directory as the checkpoint; that directory's
#     best.pt is independently hashed against --sha-nicext.
#   * Full 64-char latent hashes (field name now truthful); NICEXT
#     trajectory fully verified (epochs exactly 30..59, length 30,
#     best_val_psnr == recomputed max, complete epoch-57 val record
#     equality, facts init == checkpoint init); BASEDIAG facts sha256
#     recorded symmetrically.
# Changelog (v0.1 -> v0.2, pre-smoke review fixes):
#   * Smoke facts record the ACTUAL n_post used (4) plus formal_n_post
#     (32) -- v0.1 would have claimed 32 while running 4.
#   * --nicext-facts REQUIRED: ep57 is verified as the formal NICEXT
#     best through its run facts (script version, best_epoch 57 as the
#     earliest max-PSNR epoch, best_val consistency, competence.passed
#     false with empty qualifying_epochs, init chain, cfg equality,
#     facts SHA recorded) -- a file named best.pt is not proof.
#   * ep29 cross-anchored against the already-loaded BASEDIAG facts
#     (sha256, selected_epoch 29, verified_best_epoch 29).
#   * Hardening: subset indices validated (ints, unique, in-range);
#     checkpoint key checks + TOCTOU rehash around both torch.load
#     calls; both declared SHAs validated as 64-hex before use.
# Changelog (NEW in v0.1): Introduced (all seven pre-build review fixes
#   folded in: dedup rule, least-degraded labelling, five-recon count,
#   precise error maps, shared latent bank, exact-index subset reuse,
#   NICEXT provenance-chain verification; 2×5 layout with four error
#   maps).
# Update summary: evidence generator for the residual design; no gate.
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
from seqref_mri.src.base_experts import build_expert
from seqref_mri.scripts.train_base import (_collate, _prepare, DIM,
                                           IN_CHANNELS,
                                           NORMALIZED_DATA_RANGE, CELL_HW)

logger = logging.getLogger("seqref_mri.visinspect")

__version__ = "0.4"
__abbr__ = "SEQREF-VINS"

Z_SEED = 20260910
N_POST = 32
N_SELECT = 6
LOCKED_315 = {"train_slices": 8000, "val_slices": 1000, "batch": 8,
              "lr": 1e-4, "seed_index": 0, "subset_seed": 20260904,
              "cond_width": 64, "h_dim": 128, "hidden": 256, "n_post": 4,
              "n_layers": 4, "expert": "nice", "test0": False}


def _fail(msg: str) -> None:
    logger.error("[vins] %s", msg)
    raise RuntimeError(msg)


def _require_finite(name: str, x) -> None:
    t = x if torch.is_tensor(x) else torch.as_tensor(x)
    if not torch.isfinite(t).all():
        _fail(f"{name}: non-finite value detected (validity gate)")


def _sha_full(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _valid_sha_arg(name: str, s: str) -> str:
    import re
    if not re.fullmatch(r"[0-9a-fA-F]{64}", s):
        _fail(f"{name} must be exactly 64 hexadecimal characters")
    return s.lower()


def _tensor_sha(t: torch.Tensor) -> str:
    # FULL 64-char digest (field names say sha256; truncation would lie)
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def provenance(argv, *, allow_dirty: bool, repo_root: Path) -> dict:
    # Pinned to the repository containing THIS script -- cwd-dependent
    # git could silently record a parent repo's commit/dirty state.
    if not (repo_root / "seqref_mri").is_dir():
        _fail(f"derived repo root {repo_root} does not contain "
              "seqref_mri/ -- deployment layout changed?")
    try:
        commit = subprocess.run(["git", "-C", str(repo_root), "rev-parse",
                                 "--short=12", "HEAD"],
                                capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo_root), "status",
                                     "--porcelain"],
                                    capture_output=True, text=True,
                                    check=True).stdout.strip())
    except Exception as e:
        logger.error("[vins] git provenance unobtainable: %r", e)
        raise RuntimeError(f"git provenance unobtainable: {e!r}") from e
    if dirty and not allow_dirty:
        _fail("working tree DIRTY -- commit before the formal VISINSPECT run")
    if dirty:
        logger.warning("[vins] DIRTY TREE PERMITTED (smoke): PROVISIONAL")
    return {"git_commit": commit, "git_dirty": dirty, "argv": argv,
            "python": sys.version.split()[0], "torch": torch.__version__}


def load_basediag(path: str, sha_nice: str) -> tuple[list[int], dict, str]:
    p = Path(path)
    if not p.is_file():
        _fail(f"--basediag-facts missing at {path}")
    raw = p.read_bytes()          # read ONCE: hash and parse same bytes
    bd_sha = hashlib.sha256(raw).hexdigest()
    bd = json.loads(raw)
    if bd.get("smoke") is not False:
        _fail("basediag facts: smoke must be false (formal evidence only)")
    if bd.get("provenance", {}).get("git_dirty") is not False:
        _fail("basediag facts: git_dirty must be false")
    c = bd.get("constants", {})
    if c.get("subset_size") != 64 or c.get("subset_seed") != 20260906:
        _fail(f"basediag facts: subset {c.get('subset_size')}/"
              f"{c.get('subset_seed')} != 64/20260906")
    idx = bd.get("subset_indices")
    if not isinstance(idx, list) or len(idx) != 64:
        _fail("basediag facts: subset_indices missing or wrong length")
    if not all(type(i) is int for i in idx):
        _fail("basediag subset indices must be integers")
    if len(set(idx)) != 64:
        _fail("basediag subset indices are not unique")
    # ep29 cross-anchor: BASEDIAG already verified the NICE checkpoint
    nice_ck = bd.get("experts", {}).get("nice", {}).get("checkpoint", {})
    if nice_ck.get("sha256") != sha_nice:
        _fail("basediag facts: nice checkpoint sha != --sha-nice -- "
              "ep29 anchor broken")
    if nice_ck.get("selected_epoch") != 29:
        _fail(f"basediag facts: nice selected_epoch "
              f"{nice_ck.get('selected_epoch')} != 29")
    if nice_ck.get("best_selection_verified", {}).get(
            "verified_best_epoch") != 29:
        _fail("basediag facts: nice verified_best_epoch != 29")
    return idx, bd, bd_sha


def verify_nicext_facts(path: str, sha_nicext: str, sha_nice: str,
                        ckpt_cfg: dict, ckpt_path: str,
                        ckpt_init: dict) -> dict:
    p = Path(path).resolve()
    if not p.is_file():
        _fail(f"--nicext-facts missing at {path}")
    if p.name != "facts.json":
        _fail("--nicext-facts must name facts.json")
    # BINDING: facts and checkpoint must cohabit one run directory, and
    # that directory's best.pt must hash to the declared SHA.
    expected_ckpt = p.parent / "best.pt"
    if expected_ckpt.resolve() != Path(ckpt_path).resolve():
        _fail("NICEXT facts and checkpoint are not from the same run "
              "directory")
    if _sha_full(expected_ckpt) != sha_nicext:
        _fail("NICEXT facts-directory best.pt SHA != --sha-nicext")
    raw = p.read_bytes()          # read ONCE: hash and parse same bytes
    fsha = hashlib.sha256(raw).hexdigest()
    nf = json.loads(raw)
    if nf.get("script") != "SEQREF-TB-MRI v0.5":
        _fail(f"nicext facts: script {nf.get('script')!r} != v0.5")
    hist = nf.get("history") or []
    if not hist:
        _fail("nicext facts: history missing")
    for i, row in enumerate(hist):
        val = row.get("val", {})
        if "epoch" not in row or "psnr" not in val or "ssim" not in val:
            _fail(f"nicext facts: malformed history row {i}")
        _require_finite(f"nicext history[{i}] psnr", val["psnr"])
        _require_finite(f"nicext history[{i}] ssim", val["ssim"])
    epochs = [r["epoch"] for r in hist]
    if epochs != list(range(30, 60)):
        _fail(f"nicext facts: history epochs {epochs[:3]}..{epochs[-1:]} "
              "!= exactly 30..59")
    best_psnr = max(r["val"]["psnr"] for r in hist)
    best_ep = min(r["epoch"] for r in hist
                  if r["val"]["psnr"] == best_psnr)
    if nf.get("best_epoch") != 57 or best_ep != 57:
        _fail(f"nicext facts: best epoch recorded {nf.get('best_epoch')}, "
              f"recomputed {best_ep} -- != 57")
    if abs(nf.get("best_val_psnr", -1) - best_psnr) > 1e-9:
        _fail("nicext facts: best_val_psnr != recomputed maximum")
    row57 = next(r for r in hist if r["epoch"] == 57)
    if nf.get("best_val") != row57["val"]:
        _fail("nicext facts: best_val != complete epoch-57 val record")
    if nf.get("init") != ckpt_init:
        _fail("nicext facts: init != checkpoint init")
    # RECOMPUTE competence from history; the recorded block must match
    # exactly (verification of the declaration, not trust in it)
    PSNR_FLOOR, SSIM_FLOOR = 31.53, 0.691
    qualifying = [row["epoch"] for row in hist
                  if row["val"]["psnr"] >= PSNR_FLOOR
                  and row["val"]["ssim"] >= SSIM_FLOOR]
    expected_comp = {"psnr_floor": PSNR_FLOOR, "ssim_floor": SSIM_FLOOR,
                     "passed": bool(qualifying),
                     "qualifying_epochs": qualifying}
    if nf.get("competence") != expected_comp:
        _fail("nicext facts: competence block does not match the "
              "recomputed history result")
    if qualifying:
        _fail("nicext facts: qualifying epochs non-empty -- this is not "
              "the falsified NICEXT run the ruling recorded")
    init = nf.get("init", {})
    if (init.get("init_from_epoch") != 29
            or init.get("init_from_sha256") != sha_nice
            or init.get("declared_sha_verified") is not True):
        _fail("nicext facts: init chain invalid")
    if nf.get("cfg") != ckpt_cfg:
        _fail("nicext facts: cfg != checkpoint cfg")
    return {"facts_path": str(p), "facts_sha256": fsha,
            "verified_best_epoch": 57}


def _build_nice(cfg: dict, device: str):
    cond = Conditioner(in_channels=IN_CHANNELS, width=cfg["cond_width"],
                       h_dim=cfg["h_dim"])
    m = build_expert("nice", dim=DIM, h_dim=cfg["h_dim"], conditioner=cond,
                     hidden=cfg["hidden"], use_film=True,
                     n_layers=cfg["n_layers"])
    return m.to(device).eval()


def load_ckpt_nice29(path: str, declared_sha: str, device: str):
    p = Path(path)
    if not p.is_file():
        _fail(f"nice ep29 checkpoint missing: {path}")
    sha = _sha_full(p)
    if sha != declared_sha:
        _fail(f"nice ep29 SHA mismatch: computed {sha}")
    blob = torch.load(p, map_location="cpu")
    for key in ("model", "cfg", "epoch"):
        if key not in blob:
            _fail(f"nice ep29 checkpoint lacks {key!r}")
    if _sha_full(p) != sha:
        _fail("nice ep29 checkpoint changed during loading")
    cfg = blob["cfg"]
    for k, v in LOCKED_315.items():
        if cfg.get(k) != v:
            _fail(f"nice ep29 cfg[{k!r}]={cfg.get(k)!r} != locked {v!r}")
    if cfg.get("epochs") != 30 or blob["epoch"] != 29:
        _fail(f"nice ep29: epochs={cfg.get('epochs')}, epoch={blob['epoch']}"
              " -- not the I6 best checkpoint")
    m = _build_nice(cfg, device)
    m.load_state_dict(blob["model"], strict=True)
    for n_, t in list(m.named_parameters()) + list(m.named_buffers()):
        _require_finite(f"nice29 {n_}", t)
    return m, {"sha256": sha, "epoch": 29}


def load_ckpt_nicext57(path: str, declared_sha: str, sha_nice: str,
                       device: str):
    p = Path(path)
    if not p.is_file():
        _fail(f"nicext ep57 checkpoint missing: {path}")
    sha = _sha_full(p)
    if sha != declared_sha:
        _fail(f"nicext ep57 SHA mismatch: computed {sha}")
    blob = torch.load(p, map_location="cpu")
    for key in ("model", "cfg", "epoch"):
        if key not in blob:
            _fail(f"nicext ep57 checkpoint lacks {key!r}")
    if _sha_full(p) != sha:
        _fail("nicext ep57 checkpoint changed during loading")
    cfg, init = blob["cfg"], blob.get("init")
    if init is None:
        _fail("nicext ep57: checkpoint lacks init metadata -- not a "
              "continuation checkpoint")
    if blob["epoch"] != 57:
        _fail(f"nicext ep57: epoch={blob['epoch']} != 57")
    if init.get("init_from_epoch") != 29:
        _fail(f"nicext ep57: init_from_epoch={init.get('init_from_epoch')}")
    if init.get("init_from_sha256") != sha_nice:
        _fail("nicext ep57: stored source SHA != the NICE ep29 SHA -- "
              "provenance chain broken")
    if init.get("declared_sha_verified") is not True:
        _fail("nicext ep57: declared_sha_verified is not true")
    for k, v in LOCKED_315.items():
        if cfg.get(k) != v:
            _fail(f"nicext cfg[{k!r}]={cfg.get(k)!r} != locked {v!r}")
    if cfg.get("epochs") != 30:
        _fail(f"nicext: added epochs {cfg.get('epochs')!r} != declared 30")
    m = _build_nice(cfg, device)
    m.load_state_dict(blob["model"], strict=True)
    for n_, t in list(m.named_parameters()) + list(m.named_buffers()):
        _require_finite(f"nicext57 {n_}", t)
    return m, {"sha256": sha, "epoch": 57,
               "init_chain_verified": True}, cfg, init


@torch.no_grad()
def recon_set(model, p: dict, z_bank: list[torch.Tensor]) -> dict:
    h = model.cond(p["cond_in"])
    _require_finite("h", h)
    acc = None
    for z in z_bank:
        x = model.decode(z, h)
        _require_finite("decode", x)
        acc = x if acc is None else acc + x
    pm = acc / len(z_bank)
    z0 = model.decode(torch.zeros(p["cond_in"].shape[0], model.dim,
                                  device=p["cond_in"].device), h)
    _require_finite("z0 decode", z0)
    return {"pm": pm, "z0": z0}


def _mag(flat: torch.Tensor) -> torch.Tensor:
    return two_channel_to_complex(
        flat.view(-1, 2, CELL_HW, CELL_HW)).abs().unsqueeze(1)


def _metrics(mag: torch.Tensor, p: dict, flat: torch.Tensor) -> dict:
    psnr = psnr_per_sample(mag.cpu(), p["tgt_norm"].cpu(),
                           data_range=NORMALIZED_DATA_RANGE)
    ssim = ssim_per_sample(mag.cpu(), p["tgt_norm"].cpu(),
                           data_range=NORMALIZED_DATA_RANGE)
    _require_finite("psnr", psnr)
    _require_finite("ssim", ssim)
    xm_c = two_channel_to_complex(flat.view(-1, 2, CELL_HW, CELL_HW))
    cons = [float(p["ops"][i].consistency(xm_c[i] * p["amax"][i],
                                          p["y"][i]))
            for i in range(len(p["ops"]))]
    return {"psnr": psnr, "ssim": ssim, "cons": cons}


def select_slices(x0_psnr: np.ndarray, diff: np.ndarray) -> dict:
    n = len(x0_psnr)
    first, middle, last = 0, n // 2, n - 1
    # ranked lists per criterion (descending preference)
    rank_x0 = list(np.argsort(-x0_psnr, kind="stable"))
    rank_diff_max = list(np.argsort(-diff, kind="stable"))
    rank_diff_min = list(np.argsort(diff, kind="stable"))
    candidates = [("first", [first]), ("middle", [middle]),
                  ("last", [last]),
                  ("argmax_x0_psnr", rank_x0),
                  ("max_signed_ep29pm_minus_x0", rank_diff_max),
                  ("min_signed_ep29pm_minus_x0", rank_diff_min)]
    chosen, record = [], []
    for crit, ranked in candidates:
        pick = None
        for pos in ranked:
            if int(pos) not in chosen:
                pick = int(pos)
                break
        if pick is None:
            _fail(f"selection: no unused slice left for {crit}")
        chosen.append(pick)
        record.append({"criterion": crit, "original_candidate":
                       int(ranked[0]), "final_position": pick,
                       "deduplicated": pick != int(ranked[0])})
    return {"positions": chosen, "record": record}


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt-nice", required=True)
    ap.add_argument("--sha-nice", required=True)
    ap.add_argument("--ckpt-nicext", required=True)
    ap.add_argument("--sha-nicext", required=True)
    ap.add_argument("--basediag-facts", required=True)
    ap.add_argument("--nicext-facts", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_path = Path(__file__).resolve()
    # seqref_mri/scripts/_diag/visinspect.py -> parents[2]=seqref_mri,
    # parents[3]=repo root
    seqref_root = script_path.parents[2]
    repo_root = script_path.parents[3]
    if seqref_root.name != "seqref_mri":
        _fail(f"script not at the expected seqref_mri/scripts/_diag "
              f"location (got {seqref_root}) -- refusing to guess paths")
    base = seqref_root / "results" / "_diag"
    default_formal = (base / "visinspect").resolve()
    default_smoke = (base / "visinspect_smoke").resolve()
    out_dir = (Path(a.out).resolve() if a.out
               else (default_smoke if a.smoke else default_formal))
    if a.smoke and out_dir == default_formal:
        _fail("smoke output may not target the formal VISINSPECT "
              "directory")
    if a.out and out_dir.exists():
        _fail(f"explicit --out target already exists: {out_dir} -- "
              "refusing to overwrite any existing directory")
    if not a.smoke and out_dir.exists():
        _fail(f"formal output already exists: {out_dir} -- delete "
              "manually to rerun")
    tmp = out_dir.parent / (out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "panels").mkdir(parents=True)

    actual_n_post = 4 if a.smoke else N_POST
    facts: dict = {"script": f"{__abbr__} v{__version__} visinspect",
                   "smoke": a.smoke,
                   "constants": {"z_seed": Z_SEED,
                                 "n_post": actual_n_post,
                                 "formal_n_post": N_POST,
                                 "n_select": N_SELECT,
                                 "basediag_subset_size": 64}}
    facts["provenance"] = provenance(sys.argv, allow_dirty=a.smoke,
                                     repo_root=repo_root)
    facts["provenance"]["torch_load_futurewarning"] = (
        "accepted for this diagnostic; checkpoints SHA-verified with "
        "TOCTOU rehash")
    if facts["provenance"]["git_dirty"]:
        facts["EVIDENCE_STATUS"] = "PROVISIONAL (smoke, dirty tree)"

    sha_nice = _valid_sha_arg("--sha-nice", a.sha_nice)
    sha_nicext = _valid_sha_arg("--sha-nicext", a.sha_nicext)
    subset_idx, bd_facts, bd_sha = load_basediag(a.basediag_facts,
                                                 sha_nice)
    facts["basediag_facts_sha256"] = bd_sha
    if a.smoke:
        subset_idx = subset_idx[:8]
    facts["subset_indices"] = subset_idx
    facts["constants"]["active_subset_size"] = len(subset_idx)
    facts["ep29_anchor"] = "verified against formal BASEDIAG facts"

    ds = FastMRISliceDataset(a.data_root, split="val", mode="eval")
    if min(subset_idx) < 0 or max(subset_idx) >= len(ds):
        _fail("basediag subset index outside current dataset")
    items = [ds[i] for i in subset_idx]
    # stable identities + dataset fingerprint (indices are POSITIONS)
    facts["slice_identifiers"] = [
        {"subset_position": pos, "dataset_index": int(ds_idx),
         "file": it["meta"]["file"],
         "slice_index": it["meta"]["slice_index"]}
        for pos, (ds_idx, it) in enumerate(zip(subset_idx, items))]
    manifest = "\n".join(f'{it["meta"]["file"]}|{it["meta"]["slice_index"]}'
                          for it in items)
    facts["subset_manifest"] = {
        "n_slices": len(subset_idx), "ordered": True,
        "sha256": hashlib.sha256(manifest.encode()).hexdigest()}
    facts["dataset_identity_limitation"] = (
        "Exact index reuse is verified, but dataset-order identity at "
        "BASEDIAG time cannot be independently proven from the available "
        "BASEDIAG record (it stored indices only). The identifiers above "
        "bind THIS run's data; future runs can verify against them.")
    batch = _collate(items)
    p = _prepare(batch, device, test0=False)
    for key in ("cond_in", "x_norm", "tgt_norm", "amax"):
        _require_finite(f"prepared {key}", p[key])

    with torch.no_grad():
        nice29, ck29 = load_ckpt_nice29(a.ckpt_nice, sha_nice, device)
        next57, ck57, next_cfg, next_init = load_ckpt_nicext57(
            a.ckpt_nicext, sha_nicext, sha_nice, device)
        ck57["best_verified"] = verify_nicext_facts(
            a.nicext_facts, sha_nicext, sha_nice, next_cfg,
            a.ckpt_nicext, next_init)
        facts["checkpoints"] = {"nice_ep29": ck29, "nicext_ep57": ck57}

        # ONE z bank, reused for both checkpoints
        torch.manual_seed(Z_SEED)
        z_bank = [torch.randn(len(subset_idx), DIM, device=device)
                  for _ in range(actual_n_post)]
        facts["z_bank_sha256"] = [_tensor_sha(z) for z in z_bank]

        r29 = recon_set(nice29, p, z_bank)
        r57 = recon_set(next57, p, z_bank)

    x0_flat = p["cond_in"].flatten(1)
    recons = {"x0": x0_flat, "ep29_pm": r29["pm"], "ep57_pm": r57["pm"],
              "ep29_z0": r29["z0"], "ep57_z0": r57["z0"]}
    mags = {k: _mag(v) for k, v in recons.items()}
    mets = {k: _metrics(mags[k], p, v) for k, v in recons.items()}

    x0_psnr = mets["x0"]["psnr"].numpy()
    diff = mets["ep29_pm"]["psnr"].numpy() - x0_psnr
    facts["criteria_inputs"] = {
        "x0_psnr_per_slice": [float(v) for v in x0_psnr],
        "ep29pm_minus_x0_signed": [float(v) for v in diff]}
    sel = select_slices(x0_psnr, diff)
    # honest labelling: only an "improvement" if actually positive
    max_diff = float(diff[sel["positions"][4]])
    sel["record"][4]["label"] = ("NICE improvement case" if max_diff > 0
                                 else "least-degraded NICE case (max "
                                      "signed diff still negative)")
    facts["selection"] = sel

    # per-slice records for the five reconstructions on selected slices
    facts["selected_metrics"] = {}
    facts["ep57_deltas"] = {}
    for pos in sel["positions"]:
        facts["selected_metrics"][pos] = {
            k: {"psnr": float(mets[k]["psnr"][pos]),
                "ssim": float(mets[k]["ssim"][pos]),
                "consistency": mets[k]["cons"][pos]}
            for k in recons}
        facts["ep57_deltas"][pos] = {
            "pm_psnr": float(mets["ep57_pm"]["psnr"][pos]
                             - mets["ep29_pm"]["psnr"][pos]),
            "z0_psnr": float(mets["ep57_z0"]["psnr"][pos]
                             - mets["ep29_z0"]["psnr"][pos])}

    # panels: 2x5, shared scales
    tgt = p["tgt_norm"].cpu()
    for pos in sel["positions"]:
        t_img = tgt[pos, 0]
        vmax = float(torch.quantile(t_img.flatten(), 0.995))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        imgs_top = [(t_img, "target"),
                    (mags["x0"][pos, 0].cpu(), "x0 zero-filled"),
                    (mags["ep29_pm"][pos, 0].cpu(), "ep29 PM"),
                    (mags["ep57_pm"][pos, 0].cpu(), "ep57 PM"),
                    (mags["ep29_z0"][pos, 0].cpu(), "ep29 z=0")]
        errs = {k: (mags[k][pos, 0].cpu() - t_img).abs()
                for k in ("x0", "ep29_pm", "ep57_pm", "ep57_z0")}
        evmax = float(torch.quantile(
            torch.cat([e.flatten() for e in errs.values()]), 0.995))
        if not np.isfinite(evmax) or evmax <= 0:
            evmax = 1.0
        imgs_bot = [(mags["ep57_z0"][pos, 0].cpu(), "ep57 z=0", "img"),
                    (errs["x0"], "x0 |err|", "err"),
                    (errs["ep29_pm"], "ep29 PM |err|", "err"),
                    (errs["ep57_pm"], "ep57 PM |err|", "err"),
                    (errs["ep57_z0"], "ep57 z0 |err|", "err")]
        fig, axes = plt.subplots(2, 5, figsize=(16, 6.6))
        for ax, (img, title) in zip(axes[0], imgs_top):
            ax.imshow(img.numpy(), cmap="gray", vmin=0.0, vmax=vmax)
            ax.set_title(title, fontsize=8); ax.axis("off")
        for ax, (img, title, kind) in zip(axes[1], imgs_bot):
            ax.imshow(img.numpy(), cmap="gray", vmin=0.0,
                      vmax=(vmax if kind == "img" else evmax))
            ax.set_title(title, fontsize=8); ax.axis("off")
        f = tmp / "panels" / f"slice_pos{pos:02d}.png"
        fig.suptitle(f"subset position {pos} (ds index "
                     f"{subset_idx[pos]})", fontsize=9)
        fig.savefig(f, dpi=110, bbox_inches="tight")
        plt.close(fig)

    facts["verdict"] = ("EVIDENCE RECORDED (smoke, provisional)"
                        if a.smoke else "EVIDENCE RECORDED (no gate)")
    with open(tmp / "facts.json", "w") as f:
        json.dump(facts, f, indent=2)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    logger.info("[vins] %s -- report at %s", facts["verdict"], out_dir)


if __name__ == "__main__":
    main()
