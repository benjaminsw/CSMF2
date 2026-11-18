# SEQREF-DUNC-ADP v0.1 -- diag_unc_adapter.py: D-UNC adapter to the pinned V05LT production seams (plan SEQREF-DUNC-IMPL v0.10.1 §3)
# LIFETIME: KEEP
"""Thin adapter between the D-UNC extractor (223 v1.0 + AM1) and the pinned V05LT production code.

Every scientific operation is a call into a pinned production function; nothing is copied or
re-implemented here. The only arithmetic in this file is the reversible re-ordering of the
standardised scalar vector between the production layout and the step-B layout:

  production (free_flow_runtime.pack_scalars) : interleaved  [Re0, Im0, Re1, Im1, ...]
  step B (diag_unc_core.standardise)          : block        [Re0, Re1, ... | Im0, Im1, ...]

Seams used (module -> function):
  v05lt_common      load_split_manifest, holdout_ids, verify_p0s, build_split_views, Paths
  v05lt_eval        build_states, load_trained_checkpoint, decode_bank, r0_image, REG_F10["p4_rel"]
  free_flow_runtime load_p4s2_parent, standardise_free, pack_scalars
  estimators        z_diag_bank, image_from_u
The validation view is build_split_views(...)[1] (split 'train', mode 'eval'). The production
split 'val' is the V02 holdout and is never requested here.

The repository root must already be on sys.path. Importing this module imports no project code;
seams() does. Every failure is logged and raised as AdapterError (or as the production error).
No fallback, placeholder, mock or dummy return.

Changelog:
  v0.1 (2026-11-18) first version, for fixture 11b.
"""
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

__version__ = "0.1"
LOG = logging.getLogger("diag_unc_adapter")
_SEAMS = None


class AdapterError(Exception):
    """Adapter failure. Always logged; never replaced by a substitute value."""

    def __init__(self, message):
        super().__init__(message)
        LOG.error("ADAPTER_ERROR: %s", message)


def _require(cond, message):
    if not cond:
        raise AdapterError(message)


def _call(where, fn, *args, **kwargs):
    """Run a production seam; log any failure with its location, then re-raise it unchanged."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        LOG.error("production seam failed in %s: %s: %s", where, type(exc).__name__, exc)
        raise


def seams():
    """Import the production modules once and return them."""
    global _SEAMS
    if _SEAMS is None:
        try:
            from seqref_mri.tdiag import _bootstrap  # noqa: F401 -- registered path setup, first as in v05lt_eval
            import torch
            from seqref_mri.scripts import v05lt_common as C
            from seqref_mri.scripts import v05lt_eval as EV
            from seqref_mri.scripts import tiny_gate as tg
            from seqref_mri.tdiag import d3, estimators as est
            from seqref_mri.src import free_flow_runtime as ffr
            from seqref_mri.src import fastmri_data as fdm
        except Exception as exc:
            raise AdapterError("production import failed: %s: %s" % (type(exc).__name__, exc))
        _SEAMS = SimpleNamespace(torch=torch, C=C, EV=EV, tg=tg, d3=d3, est=est, ffr=ffr, fdm=fdm, dec=tg.dec)
    return _SEAMS


# ---------------------------------------------------------------- inputs and data
def load_inputs(repo):
    """Registered split manifest, holdout ID set (JSON manifest only), S_ref and the P4 parent."""
    s = seams()
    paths = s.C.Paths(repo=Path(repo))
    sp = paths.inputs / "v05lt_split_manifest.json"
    split_doc = _call("load_split_manifest", s.C.load_split_manifest, sp, s.C.REGISTERED_INPUTS[sp.name])
    holdout = _call("holdout_ids", s.C.holdout_ids, paths.v02_manifests)
    s_ref = _call("verify_p0s", s.C.verify_p0s, paths.p0s)
    p4_path = Path(repo) / s.EV.REG_F10["p4_rel"]
    p4 = _call("load_p4s2_parent", s.ffr.load_p4s2_parent, str(p4_path))
    return {"paths": paths, "split_doc": split_doc, "holdout": holdout, "s_ref": s_ref, "p4": p4, "p4_path": p4_path}


def validation_view(inputs):
    """Production validation view. Builds the two production indexes (metadata opens only)."""
    s = seams()
    _train, val = _call("build_split_views", s.C.build_split_views, inputs["paths"].data_root,
                        inputs["split_doc"], inputs["holdout"])
    _require(val.role == "validation" and val.base.split == "train" and val.base.mode == "eval",
             "validation view is not (role validation, split train, mode eval)")
    return val


def positions_for(view, identities):
    """View positions of (file, slice_index) identities, from the index only (no item is read)."""
    want = [(str(f), int(i)) for f, i in identities]
    _require(len(set(want)) == len(want), "duplicate identity requested")
    found = {}
    for pos in range(len(view)):
        ent = view.entry(pos)
        if ent in want:
            _require(ent not in found, "identity %s|%d occurs twice in the view" % ent)
            found[ent] = pos
    missing = [w for w in want if w not in found]
    _require(not missing, "identities not in the validation view: %s" % missing)
    return [found[w] for w in want]


def build_states(view, positions, inputs):
    """Production slice states (v05lt_eval.build_states) for the given view positions."""
    s = seams()
    return _call("build_states", s.EV.build_states, view, [int(p) for p in positions], inputs["p4"], inputs["s_ref"])


# ---------------------------------------------------------------- checkpoint, bank, decode
def load_checkpoint(path, file_sha256, model_state_sha256, *, seed, arm, epoch):
    """Verified checkpoint -> (model in eval mode, VerifiedCheckpoint). Production factory only."""
    s = seams()
    return _call("load_trained_checkpoint", s.EV.load_trained_checkpoint, Path(path), file_sha256,
                 model_state_sha256, seed=seed, arm=arm, epoch=epoch)


def z_bank():
    """Registered Z_DIAG bank record: {'bank', 'bank_sha256', 'manifest_sha256'}."""
    s = seams()
    return _call("z_diag_bank", s.est.z_diag_bank)


def decode_bank(model, st, bank):
    """All bank rows for one slice -> (128, 6912) complex128 physical free coefficients."""
    s = seams()
    return _call("decode_bank", s.EV.decode_bank, model, st, bank)


def decode_r0(model, st):
    """z = 0 decode -> (complex image (96, 96), free coefficients (6912,) complex128)."""
    s = seams()
    return _call("r0_image", s.EV.r0_image, model, st)


def image_from_u(st, u):
    """Physical free coefficients (6912,) -> complex image (96, 96), measured k-space retained."""
    s = seams()
    return _call("image_from_u", s.est.image_from_u, st, u)


# ---------------------------------------------------------------- coefficients and layouts
def observed_u(st):
    """Reference free coefficients of the slice: the production state value (6912,) complex128."""
    u = st["u_true"]
    _require(isinstance(u, np.ndarray) and u.dtype == np.complex128 and u.shape == (st["cmap"].n_free_complex,),
             "state u_true is not (n_free,) complex128")
    return u


def free_coordinates(st):
    """(rows, cols) int64 arrays of the unacquired coefficients, in the registered flatten order."""
    cmap = st["cmap"]
    return np.asarray(cmap.free_rows, dtype=np.int64), np.asarray(cmap.free_cols, dtype=np.int64)


def standardise_interleaved(u, st):
    """Complex (..., C) -> production standardised scalars (..., 2C) float64, interleaved."""
    s = seams()
    re, im = _call("standardise_free", s.ffr.standardise_free, u, st["cmap"], st["vecs"])
    return _call("pack_scalars", s.ffr.pack_scalars, re, im)


def loc_scale_block(st):
    """P4 applied pairs of the slice as step-B (loc, scale), each (2C,) float64 in block layout."""
    v = st["vecs"]
    loc = np.concatenate([v["mean_re"], v["mean_im"]]).astype(np.float64, copy=False)
    scale = np.concatenate([v["scale_re"], v["scale_im"]]).astype(np.float64, copy=False)
    _require(bool(np.all(np.isfinite(loc)) and np.all(np.isfinite(scale)) and np.all(scale > 0)),
             "P4 applied pair is non-finite or has a non-positive scale")
    return loc, scale


def _even_last_axis(v):
    v = np.asarray(v)
    _require(v.ndim >= 1 and v.shape[-1] % 2 == 0, "scalar vector must have an even last axis, got shape %s" % (v.shape,))
    return v


def interleaved_to_block(v):
    """[Re0, Im0, Re1, Im1, ...] -> [Re... | Im...] along the last axis (exact re-ordering)."""
    v = _even_last_axis(v)
    return np.concatenate([v[..., 0::2], v[..., 1::2]], axis=-1)


def block_to_interleaved(v):
    """[Re... | Im...] -> [Re0, Im0, Re1, Im1, ...] along the last axis (exact inverse)."""
    v = _even_last_axis(v)
    n = v.shape[-1] // 2
    out = np.empty_like(v)
    out[..., 0::2] = v[..., :n]
    out[..., 1::2] = v[..., n:]
    return out
