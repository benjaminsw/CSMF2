# SEQREF-IMPLR v0.1 -- src.free_flow_runtime
# LIFETIME: KEEP
# =============================================================================
# Purpose: free-coordinate conditional-NSF runtime for the SEQREF-MRI IMPL
#          stage (FLOW_FAMILY=NSF, preregistered in SEQREF-MRI-IMPLSPEC v0.1,
#          incorporated by EXEC §13). This module owns the PRODUCTION
#          machinery that the trainer and the Class-A self-test both call:
#            * model construction (CondNSF DIRECT, K=8, hidden=256,
#              n_layers=6, B=SPLINE_B frozen by IMPL-B) + mask branch
#            * conditioning h = c_eta(cond_in) + mask_branch(mask)
#            * P3 map binding + mandatory re-derivation (consumer contract)
#            * P4 /2 PER-LOCATION standardisation and its exact inverse
#              (float64 arithmetic path)
#            * registered interleaved re/im scalar packing + unpacking
#            * encode_target  : x_norm - cond_in -> fft2c -> gather ->
#                               standardise -> pack           (B, 13824)
#            * decode_to_image: z -> NSF inverse -> unstandardise ->
#                               de-interleave -> scatter through the
#                               re-derived map -> measured k retained
#                               EXACTLY -> ifft2c
# Registered rules (binding):
#   * SPLINE_B is CONSUMED from the pinned IMPL-B facts artefact and asserted
#     equal to the frozen literal; it is never re-derived and never tuned.
#   * _BaseExpert.sample() is NOT used anywhere: its (1,1,H,W)
#     single-channel check is incompatible with 2-channel conditioning and
#     it bypasses the P3/P4 binding chain (train_base header, I4 flag).
#   * The decode path retains measured k-space EXACTLY (DEC
#     decode_normalised); acquired-coefficient fixity is gated by Class-A
#     A3 at <= 1e-5.
#   * Standardisation / inverse standardisation is the float64 arithmetic
#     path (IMPL-B precedent); the NSF itself runs float32.
#   * Every failure path: logger.error + typed raise (FreeFlowError).
#     No fallback, no mock, no placeholder, no silent pass.
#   * NSF_TRANSFORM_PARAMETERS / CONDITIONING_PARAMETERS are
#     TEST-REGISTRATION GROUPS ONLY (Class-A A6). They are never passed to
#     an optimizer and never alter training behaviour; the optimizer always
#     receives the full production parameter set.
# Changelog (NEW in v0.1):
#   * Introduced for SEQREF-IMPL v0.1 (Class-A contract stage; TINY is a
#     separate stage and consumes this module only after A1-A10 PASS).
# Update summary:
#   v0.1 lands the complete production runtime against the closed P3 /2,
#   P4 /2 and IMPL-B parents: construction-fixity gates on the frozen
#   architecture, binding identity with mandatory map re-derivation,
#   float64 scaling round-trip, interleaved packing, and the registered
#   encode/decode paths. No training loop lives here (scripts own that).
# =============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from seqref_mri.src import residual_decoder as dec
from seqref_mri.src.base_experts import CondNSF, _gaussian_logprob
from seqref_mri.src.conditioner import Conditioner
from seqref_mri.src.fastmri_data import fft2c  # inherited frozen convention
from seqref_mri.src.flows.nsf_layer import NSFCoupling

logger = logging.getLogger("seqref_mri.free_flow_runtime")

__version__ = "0.1"
__abbr__ = "SEQREF-IMPLR"

# ---------------------------------------------------------------------------
# Frozen constants (SEQREF-IMPL v0.1 plan, frozen 2026-08-12; never
# CLI-tunable). Provenance: IMPLSPEC v0.1 + IMPL-B facts (pinned).
# ---------------------------------------------------------------------------
FLOW_FAMILY = "NSF"
GRID_H = 96
GRID_W = 96
N_FREE_COMPLEX = 6912
FLOW_DIM_REAL = 13824                 # 2 * N_FREE_COMPLEX, interleaved re/im

SPLINE_B = 5.159583485556914          # IMPL-B CLOSED PASS 2026-08-12
SPLINE_PERCENTILE = 99.9              # provenance of SPLINE_B (not re-used)
SPLINE_PERCENTILE_METHOD = "linear"   # provenance of SPLINE_B (not re-used)
SPLINE_MARGIN = 1.1                   # provenance of SPLINE_B (not re-used)

NSF_K = 8
NSF_HIDDEN = 256
NSF_N_LAYERS = 6

H_DIM = 128
COND_WIDTH = 64
COND_IN_CHANNELS = 2
USE_FILM = True
COND_USE_V2 = False
FILM_HIDDEN = 64
FILM_DEPTH = 1
FILM_USE_GELU = False
FILM_AFFINE = True                    # FiLMHead output_form == "affine"

Y_RESIDUAL_ALPHA_INIT = 0.0           # bypass DISABLED (registered)

MASK_BITS = 96
MASK_EMBED_DIM = 128
MASK_WEIGHT_INIT_STD = 0.01
MASK_BIAS_INIT = 0.0
MASK_EFFECT_REL_MIN = 1e-5            # A7 floor (EXEC §13 registration)

EXPECTED_ACQUIRED_COLUMNS = 24        # mask_counts(96) = (8, 24)
CENTRE_COLUMNS = frozenset(range(44, 52))

# Class-A tolerances that gate runtime behaviour (registered):
A3_ACQUIRED_FIXITY_MAX = 1e-5         # acquired fixity, decode path
A4_SCALING_ROUNDTRIP_MAX = 1e-12      # float64 scaling round-trip (binding)
A4_AUX_NSF_ROUNDTRIP_MAX = 1e-5       # float32 NSF fwd/inv (auxiliary only)

# MODEL_INIT_SEED makes the (RNG-dependent) module initialisation itself
# deterministic, so the Class-A evidence and the determinism sibling are
# bit-reproducible. Pinned as part of the frozen plan's A6 micro-protocol.
MODEL_INIT_SEED = 20260813

PACKING_ORDER_LITERAL = dec.P3_COMPLEX_PACKING_ORDER
FLATTEN_ORDER_LITERAL = dec.P3_FLATTEN_ORDER


class FreeFlowError(RuntimeError):
    """Typed runtime failure. `code` is a stable machine-readable tag;
    the stage layer maps these onto its own taxonomy. Always preceded by
    logger.error at the raise site."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[SEQREF-IMPLR] %s: %s", code, message)
    raise FreeFlowError(code, message)


def require_spline_b(value) -> float:
    """IMPL-B consumption gate: the value read from the pinned IMPL-B facts
    artefact must equal the frozen literal EXACTLY (float64). B is
    consumed, never re-derived, never tuned."""
    b = float(value)
    if not np.isfinite(b) or b <= 0.0:
        _fail("SPLINE_B_INVALID",
              f"the IMPL-B artefact records B={value!r}; it must be finite "
              f"and strictly positive")
    if b != SPLINE_B:
        _fail("SPLINE_B_MISMATCH",
              f"the IMPL-B artefact records B={b!r}, but the frozen "
              f"literal is {SPLINE_B!r}; IMPL consumes exactly the "
              f"calibrated bound and rejects any drift")
    return b


# ---------------------------------------------------------------------------
# Mask branch (NEW for IMPL; registered: Linear(96 -> 128),
# weight ~ Normal(0, 0.01), bias = 0)
# ---------------------------------------------------------------------------
class MaskBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(MASK_BITS, MASK_EMBED_DIM)
        nn.init.normal_(self.proj.weight, mean=0.0,
                        std=MASK_WEIGHT_INIT_STD)
        nn.init.constant_(self.proj.bias, MASK_BIAS_INIT)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() != 2 or int(mask.shape[1]) != MASK_BITS:
            _fail("MASK_LAYOUT_UNEXPECTED",
                  f"mask must be (B, {MASK_BITS}), got {tuple(mask.shape)}")
        out = self.proj(mask.to(torch.float32))
        if not torch.isfinite(out).all():
            _fail("MASK_BRANCH_NON_FINITE",
                  "the mask branch produced a non-finite embedding")
        return out


# ---------------------------------------------------------------------------
# Model assembly (CondNSF DIRECT construction -- build_expert has no K/B
# passthrough; train_base precedent). Conditioning:
#   h = c_eta(cond_in) + mask_branch(mask)
# ---------------------------------------------------------------------------
class FreeFlowModel(nn.Module):
    """The registered free-coordinate conditional NSF plus the mask branch.

    flow:        CondNSF(dim=FLOW_DIM_REAL, h_dim=H_DIM, K=NSF_K,
                 B=SPLINE_B, hidden=NSF_HIDDEN, n_layers=NSF_N_LAYERS,
                 use_film=True) over a Conditioner(in_channels=2,
                 width=64, h_dim=128, use_v2=False,
                 y_residual_alpha_init=0.0).
    mask_branch: MaskBranch (added to the conditioner output).
    """

    def __init__(self, *, spline_b: float = SPLINE_B):
        super().__init__()
        b = require_spline_b(spline_b)
        cond = Conditioner(in_channels=COND_IN_CHANNELS, width=COND_WIDTH,
                           h_dim=H_DIM, use_v2=COND_USE_V2,
                           y_residual_alpha_init=Y_RESIDUAL_ALPHA_INIT)
        self.flow = CondNSF(dim=FLOW_DIM_REAL, h_dim=H_DIM, conditioner=cond,
                            hidden=NSF_HIDDEN, n_layers=NSF_N_LAYERS,
                            K=NSF_K, B=b, use_film=USE_FILM)
        self.mask_branch = MaskBranch()
        self._verify_construction_fixity()

    # -- construction-fixity gates (frozen architecture; drift = ERROR) --
    def _verify_construction_fixity(self) -> None:
        if self.flow.dim != FLOW_DIM_REAL:
            _fail("CONSTRUCTION_DIM_MISMATCH",
                  f"flow.dim={self.flow.dim} != {FLOW_DIM_REAL}")
        if len(self.flow.layers) != NSF_N_LAYERS:
            _fail("CONSTRUCTION_LAYERS_MISMATCH",
                  f"{len(self.flow.layers)} coupling layers != "
                  f"{NSF_N_LAYERS}")
        for i, layer in enumerate(self.flow.layers):
            if not isinstance(layer, NSFCoupling):
                _fail("CONSTRUCTION_LAYER_TYPE",
                      f"layer {i} is {type(layer).__name__}, expected "
                      f"NSFCoupling")
            if layer.K != NSF_K or float(layer.B) != SPLINE_B:
                _fail("CONSTRUCTION_SPLINE_MISMATCH",
                      f"layer {i}: K={layer.K} B={float(layer.B)!r}, "
                      f"expected K={NSF_K} B={SPLINE_B!r}")
            if layer.pre.in_features != layer.d_in \
                    or layer.pre.out_features != NSF_HIDDEN \
                    or layer.mid.out_features != NSF_HIDDEN \
                    or layer.post.out_features != layer.d_out * (3 * NSF_K - 1):
                _fail("CONSTRUCTION_HIDDEN_MISMATCH",
                      f"layer {i} param-net widths diverge from the frozen "
                      f"hidden={NSF_HIDDEN}, 3K-1={3 * NSF_K - 1}")
            film = getattr(layer, "film", None)
            if film is None:
                _fail("CONSTRUCTION_FILM_MISSING",
                      f"layer {i} lacks a FiLM head (USE_FILM is frozen "
                      f"True)")
            # Frozen FiLM: Linear(h_dim -> FILM_HIDDEN) -> ReLU (depth 1,
            # no GELU) -> Linear(FILM_HIDDEN -> 2*hidden), output_form
            # "affine" (identity at init).
            mods = list(film.mlp)
            if film.output_form != "affine" or len(mods) != 3 \
                    or not isinstance(mods[0], nn.Linear) \
                    or mods[0].in_features != H_DIM \
                    or mods[0].out_features != FILM_HIDDEN \
                    or not isinstance(mods[1], nn.ReLU) \
                    or isinstance(mods[1], nn.GELU) \
                    or not isinstance(mods[2], nn.Linear) \
                    or mods[2].out_features != 2 * NSF_HIDDEN:
                _fail("CONSTRUCTION_FILM_MISMATCH",
                      f"layer {i} FiLM architecture diverges from the "
                      f"frozen hidden={FILM_HIDDEN}, depth={FILM_DEPTH}, "
                      f"use_gelu=False, output_form='affine'")
        if self.flow.cond.in_channels != COND_IN_CHANNELS \
                or self.flow.cond.width != COND_WIDTH \
                or self.flow.cond.h_dim != H_DIM \
                or self.flow.cond.use_v2 != COND_USE_V2 \
                or self.flow.cond.y_residual_enabled:
            _fail("CONSTRUCTION_CONDITIONER_MISMATCH",
                  "the conditioner diverges from the frozen in_channels=2, "
                  "width=64, h_dim=128, use_v2=False, y-residual DISABLED")
        mb = self.mask_branch.proj
        if mb.in_features != MASK_BITS or mb.out_features != MASK_EMBED_DIM:
            _fail("CONSTRUCTION_MASK_BRANCH_MISMATCH",
                  f"mask branch {mb.in_features}->{mb.out_features} != "
                  f"{MASK_BITS}->{MASK_EMBED_DIM}")
        if not bool((mb.bias == MASK_BIAS_INIT).all()):
            _fail("CONSTRUCTION_MASK_BIAS",
                  "mask-branch bias must be exactly 0 at init")
        if not torch.isfinite(mb.weight).all():
            _fail("CONSTRUCTION_MASK_WEIGHT_NON_FINITE",
                  "mask-branch weight init produced a non-finite value")

    # -- conditioning ---------------------------------------------------
    def condition(self, cond_in: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        """h = c_eta(cond_in) + mask_branch(mask). Both branches float32."""
        h = self.flow.cond(cond_in) + self.mask_branch(mask)
        if not torch.isfinite(h).all():
            _fail("CONDITIONING_NON_FINITE",
                  "the combined conditioning vector is non-finite")
        return h

    # -- probability / flow directions ----------------------------------
    def log_prob_free(self, u_scaled: torch.Tensor,
                      cond_in: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        """log p(u_scaled | cond_in, mask) under N(0, I), shape (B,)."""
        if u_scaled.dim() != 2 or int(u_scaled.shape[1]) != FLOW_DIM_REAL:
            _fail("STATE_LAYOUT_UNEXPECTED",
                  f"u_scaled must be (B, {FLOW_DIM_REAL}), got "
                  f"{tuple(u_scaled.shape)}")
        h = self.condition(cond_in, mask)
        z, ldj = self.flow.encode(u_scaled.to(torch.float32), h)
        lp = _gaussian_logprob(z) + ldj
        if not torch.isfinite(lp).all():
            _fail("LOG_PROB_NON_FINITE",
                  "non-finite log_prob on the free-coordinate state")
        return lp

    def decode_scalars(self, z: torch.Tensor, cond_in: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
        """z -> NSF inverse -> standardised free scalars (B, 13824) f32."""
        if z.dim() != 2 or int(z.shape[1]) != FLOW_DIM_REAL:
            _fail("STATE_LAYOUT_UNEXPECTED",
                  f"z must be (B, {FLOW_DIM_REAL}), got {tuple(z.shape)}")
        h = self.condition(cond_in, mask)
        u = self.flow.decode(z.to(torch.float32), h)
        if not torch.isfinite(u).all():
            _fail("DECODE_NON_FINITE",
                  "the NSF inverse produced a non-finite scalar state")
        return u

    # -- test-registration parameter groups (A6; NOT optimizer groups) --
    def nsf_transform_parameters(self) -> dict:
        """Spline parameter nets of every coupling (pre/mid/post)."""
        out = {}
        for name, p in self.named_parameters():
            if name.startswith("flow.layers.") and \
                    any(f".{w}." in name for w in ("pre", "mid", "post")):
                out[name] = p
        if not out:
            _fail("GROUP_EMPTY", "NSF-transform parameter group is empty")
        return out

    def conditioning_parameters(self) -> dict:
        """Conditioner trunk + FiLM heads. The mask branch is deliberately
        EXCLUDED: A7 owns mask-path reachability, so A6 does not rely on
        it (it is recorded separately as evidence)."""
        out = {}
        for name, p in self.named_parameters():
            if name.startswith("flow.cond.") or ".film." in name:
                out[name] = p
        if not out:
            _fail("GROUP_EMPTY", "conditioning parameter group is empty")
        return out

    def mask_branch_parameters(self) -> dict:
        out = {n: p for n, p in self.named_parameters()
               if n.startswith("mask_branch.")}
        if not out:
            _fail("GROUP_EMPTY", "mask-branch parameter group is empty")
        return out


def build_model(*, spline_b: float = SPLINE_B,
                init_seed: int = MODEL_INIT_SEED) -> FreeFlowModel:
    """Deterministically-seeded construction (Class-A evidence and the
    determinism sibling must be bit-reproducible). The seed governs module
    init ONLY; it never touches data, masks or bindings."""
    torch.manual_seed(int(init_seed))
    model = FreeFlowModel(spline_b=spline_b)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("[SEQREF-IMPLR] model built: dim=%d K=%d B=%.17g layers=%d "
                "params=%d init_seed=%d", FLOW_DIM_REAL, NSF_K, SPLINE_B,
                NSF_N_LAYERS, n_params, init_seed)
    return model


# ---------------------------------------------------------------------------
# P4 /2 applied affine pairs, indexed by PHYSICAL (r, c) — consumption only;
# means/scales/floors are never recomputed here (IMPL-B doctrine).
# ---------------------------------------------------------------------------

def build_location_index(locations: list) -> dict:
    """(row, column) -> applied-pair record, keyed by PHYSICAL grid
    location, never by packed free-coordinate index: under the
    per-realisation map the same packed index denotes different Fourier
    locations on different samples. Duplicate or malformed records are
    ERROR."""
    index: dict[tuple[int, int], dict] = {}
    required = ("row", "column", "applied_mean_re", "applied_mean_im",
                "applied_scale_re", "applied_scale_im")
    for rec in locations:
        missing = [f for f in required if f not in rec]
        if missing:
            _fail("P4_STRUCTURE_INVALID",
                  f"a P4 /2 location record lacks {missing}: {rec}")
        key = (int(rec["row"]), int(rec["column"]))
        if key in index:
            _fail("P4_STRUCTURE_INVALID",
                  f"duplicate P4 /2 location record at {key}")
        index[key] = rec
    return index


def applied_pair(loc_index: dict, r: int, c: int) -> tuple:
    """The registered P4 CONSUMPTION VALIDITY gate, applied at gather time
    to EVERY gathered location: the location must exist, applied_mean must
    be finite, applied_scale must be finite AND strictly > 0. Returns
    float64 (mean_re, scale_re, mean_im, scale_im). No fallback value is
    ever invented."""
    rec = loc_index.get((int(r), int(c)))
    if rec is None:
        _fail("P4_LOCATION_MISSING",
              f"the gathered free coordinate ({r}, {c}) has no P4 /2 "
              f"applied pair; the map and the scaling artefact disagree")
    out = []
    for ch in ("re", "im"):
        mean = float(rec[f"applied_mean_{ch}"])
        scale = float(rec[f"applied_scale_{ch}"])
        if not np.isfinite(mean):
            _fail("P4_APPLIED_PAIR_INVALID",
                  f"applied_mean_{ch} at ({r}, {c}) is {mean!r}; it must "
                  f"be finite")
        if not np.isfinite(scale) or scale <= 0.0:
            _fail("P4_APPLIED_PAIR_INVALID",
                  f"applied_scale_{ch} at ({r}, {c}) is {scale!r}; it "
                  f"must be finite AND strictly > 0 (no fallback)")
        out.append(np.float64(mean))
        out.append(np.float64(scale))
    return out[0], out[1], out[2], out[3]


def standardisation_vectors(cmap: dec.CoordinateMap,
                            loc_index: dict) -> dict:
    """The four per-coordinate float64 affine vectors (mean_re, scale_re,
    mean_im, scale_im) in the map's canonical flatten order, with the
    consumption-validity gate applied to EVERY coordinate of THIS map.
    Computed once per map; the same vectors drive standardise and its
    exact inverse."""
    n = cmap.n_free_complex
    m_re = np.empty(n, dtype=np.float64)
    s_re = np.empty(n, dtype=np.float64)
    m_im = np.empty(n, dtype=np.float64)
    s_im = np.empty(n, dtype=np.float64)
    for k in range(n):
        a = applied_pair(loc_index, int(cmap.free_rows[k]),
                         int(cmap.free_cols[k]))
        m_re[k], s_re[k], m_im[k], s_im[k] = a
    return {"mean_re": m_re, "scale_re": s_re,
            "mean_im": m_im, "scale_im": s_im,
            "n_free_complex": n}


def standardise_free(u, cmap: dec.CoordinateMap, vecs: dict) -> tuple:
    """(u - mean) / scale per component, float64, vectorised over the
    canonical flatten order. u: complex array-like (..., n_free). Returns
    (re_scaled, im_scaled) float64 with the same leading shape."""
    u_np = np.asarray(u)
    if not np.iscomplexobj(u_np) or u_np.shape[-1] != cmap.n_free_complex:
        _fail("STATE_LAYOUT_UNEXPECTED",
              f"free vector shape {u_np.shape} dtype {u_np.dtype} does "
              f"not match n_free={cmap.n_free_complex}")
    re = u_np.real.astype(np.float64)
    im = u_np.imag.astype(np.float64)
    if not (np.isfinite(re).all() and np.isfinite(im).all()):
        _fail("U_NON_FINITE",
              "the free-coefficient vector contains a non-finite "
              "component; no fallback is permitted")
    return (re - vecs["mean_re"]) / vecs["scale_re"], \
           (im - vecs["mean_im"]) / vecs["scale_im"]


def unstandardise_free(re_scaled, im_scaled, cmap: dec.CoordinateMap,
                       vecs: dict) -> np.ndarray:
    """The EXACT inverse of standardise_free: u = scaled * scale + mean,
    float64, returning complex128 (..., n_free). Registered A4 round-trip
    tolerance <= 1e-12 holds by construction of this inverse."""
    re = np.asarray(re_scaled, dtype=np.float64)
    im = np.asarray(im_scaled, dtype=np.float64)
    if re.shape != im.shape or re.shape[-1] != cmap.n_free_complex:
        _fail("STATE_LAYOUT_UNEXPECTED",
              f"standardised components {re.shape}/{im.shape} do not "
              f"match n_free={cmap.n_free_complex}")
    if not (np.isfinite(re).all() and np.isfinite(im).all()):
        _fail("U_NON_FINITE",
              "the standardised state contains a non-finite component; "
              "no fallback is permitted")
    re_u = re * vecs["scale_re"] + vecs["mean_re"]
    im_u = im * vecs["scale_im"] + vecs["mean_im"]
    return (re_u + 1j * im_u).astype(np.complex128)


# ---------------------------------------------------------------------------
# Registered interleaved re/im scalar packing (P3 canonical order):
# scalar[2k] = re[k], scalar[2k+1] = im[k].
# ---------------------------------------------------------------------------

def pack_scalars(re_scaled, im_scaled) -> np.ndarray:
    """Interleave per complex coordinate -> (..., flow_dim_real) float64."""
    re = np.asarray(re_scaled, dtype=np.float64)
    im = np.asarray(im_scaled, dtype=np.float64)
    if re.shape != im.shape:
        _fail("STATE_LAYOUT_UNEXPECTED",
              f"packing inputs {re.shape} / {im.shape} differ")
    n = re.shape[-1]
    out = np.empty(re.shape[:-1] + (2 * n,), dtype=np.float64)
    out[..., 0::2] = re
    out[..., 1::2] = im
    return out


def unpack_scalars(vec) -> tuple:
    """The exact inverse of pack_scalars; bitwise-exact in float64.
    The interleave invariant (registered order) is structural: positions
    0::2 are real, 1::2 imaginary per complex coordinate."""
    v = np.asarray(vec, dtype=np.float64)
    if v.shape[-1] % 2 != 0:
        _fail("PACKING_LAYOUT_UNEXPECTED",
              f"packed scalar vector length {v.shape[-1]} is odd; the "
              f"registered interleaved packing requires an even length")
    if not np.isfinite(v).all():
        _fail("U_NON_FINITE",
              "the packed scalar vector contains a non-finite component")
    return v[..., 0::2].copy(), v[..., 1::2].copy()


# ---------------------------------------------------------------------------
# Binding identity + MANDATORY map re-derivation (P3 consumer contract:
# "IMPL must ... RE-DERIVE each realisation's map from the recorded
# acquired_columns under the published enumeration rule and REQUIRE the
# derived map_sha256 to equal the recorded value; a mismatch is ERROR").
# This gate runs BEFORE any decode/training work on the sample.
# ---------------------------------------------------------------------------

def verify_binding_identity(row: dict, binding: dict, height: int = GRID_H,
                            width: int = GRID_W) -> dec.CoordinateMap:
    """Bind sample <-> mask <-> map for ONE slice and return the re-derived
    CoordinateMap. `row` carries the LIVE realisation (dataset_index,
    file, slice_index, split, mask_seed, live_columns from the applied
    batch mask); `binding` is the RECORDED P3 entry. Any disagreement is
    ERROR: it is code, data or provenance drift, not a data verdict."""
    order = row.get("corpus_order")
    for field in ("dataset_index", "file", "slice_index", "split",
                  "mask_seed"):
        live, recorded = row.get(field), binding.get(field)
        if field in ("dataset_index", "slice_index", "mask_seed"):
            live, recorded = int(live), int(recorded)
        if live != recorded:
            _fail("BINDING_IDENTITY_MISMATCH",
                  f"the live realisation at corpus position {order} "
                  f"disagrees with the P3 binding on {field} "
                  f"({live!r} != {recorded!r}); the frozen corpus is not "
                  f"being traversed as recorded")
    live_cols = tuple(int(c) for c in row["live_columns"])
    recorded_cols = tuple(int(c) for c in binding["acquired_columns"])
    if live_cols != recorded_cols:
        _fail("MASK_LIVE_BINDING_MISMATCH",
              f"the live eval-mode mask at corpus position {order} does "
              f"not reproduce the recorded P3 acquired columns; this is "
              f"generator or provenance drift, not a verdict")
    if len(recorded_cols) != EXPECTED_ACQUIRED_COLUMNS:
        _fail("BINDING_ACQUIRED_COUNT_UNEXPECTED",
              f"the recorded mask at corpus position {order} has "
              f"{len(recorded_cols)} acquired columns; the count is "
              f"fixed at {EXPECTED_ACQUIRED_COLUMNS} by construction")
    if width == GRID_W and not CENTRE_COLUMNS.issubset(recorded_cols):
        _fail("BINDING_CENTRE_NOT_ACQUIRED",
              f"the recorded mask at corpus position {order} lacks "
              f"centre columns 44..51; the registered mask validity "
              f"contract is broken")
    from seqref_mri.src.preflight_io import canonical_hash
    mask_sha = canonical_hash({"width": width,
                               "selected_columns": list(recorded_cols)})
    if mask_sha != binding["mask_sha256"]:
        _fail("MASK_HASH_MISMATCH",
              f"the mask hash recomputed from the recorded acquired "
              f"columns at corpus position {order} differs from the "
              f"recorded value")
    cmap = dec.build_coordinate_map(list(recorded_cols), height, width)
    map_sha = cmap.payload()["map_payload_sha256"]
    if map_sha != binding["map_sha256"]:
        _fail("MAP_HASH_MISMATCH",
              f"the map re-derived from the recorded acquired columns at "
              f"corpus position {order} hashes to {map_sha}, but P3 "
              f"recorded {binding['map_sha256']}; re-derivation is "
              f"mandatory and a mismatch is ERROR")
    if cmap.n_free_complex != N_FREE_COMPLEX \
            or cmap.flow_dim_real != FLOW_DIM_REAL \
            or int(binding["n_free_complex"]) != N_FREE_COMPLEX \
            or int(binding["flow_dim_real"]) != FLOW_DIM_REAL:
        _fail("BINDING_DIMENSION_MISMATCH",
              f"the free-coefficient dimensions at corpus position "
              f"{order} differ from the registered {N_FREE_COMPLEX}/"
              f"{FLOW_DIM_REAL}")
    return cmap


# ---------------------------------------------------------------------------
# Registered encode / decode paths
# ---------------------------------------------------------------------------

def encode_target(x_norm: torch.Tensor, cond_in: torch.Tensor,
                  cmap: dec.CoordinateMap, vecs: dict) -> np.ndarray:
    """The production training target for ONE map-shared batch:
        dx      = x_norm - cond_in            (complex, per sample)
        k_dx    = fft2c(dx)                   (centred orthonormal)
        u       = gather_unmeasured(k_dx)     (canonical flatten order)
        u_s     = standardise(u)              (float64, P4 /2 applied)
        scalars = interleave(re, im)          (B, FLOW_DIM_REAL) float64
    Returns a float64 ndarray; the float32 cast happens at the model
    boundary only."""
    if x_norm.shape != cond_in.shape or x_norm.dim() != 4 \
            or int(x_norm.shape[1]) != 2 \
            or tuple(x_norm.shape[-2:]) != (GRID_H, GRID_W):
        _fail("STATE_LAYOUT_UNEXPECTED",
              f"x_norm/cond_in must be (B, 2, {GRID_H}, {GRID_W}), got "
              f"{tuple(x_norm.shape)} / {tuple(cond_in.shape)}")
    dx = torch.complex(x_norm[:, 0], x_norm[:, 1]) \
        - torch.complex(cond_in[:, 0], cond_in[:, 1])
    k_dx = fft2c(dx)
    u = dec.gather_unmeasured(k_dx, cmap)          # (B, n_free) c64
    u_np = np.asarray(u.detach().to(torch.complex128).cpu().numpy())
    re_s, im_s = standardise_free(u_np, cmap, vecs)
    return pack_scalars(re_s, im_s)


def decode_to_image(model: FreeFlowModel, z: torch.Tensor,
                    cond_in: torch.Tensor, mask: torch.Tensor,
                    y_raw: torch.Tensor, amax: torch.Tensor,
                    cmap: dec.CoordinateMap, vecs: dict) -> torch.Tensor:
    """The registered production sampling/decode path (NO
    _BaseExpert.sample()):
        z -> NSF inverse -> standardised scalars (float32)
          -> unpack + unstandardise (float64) -> u (complex128)
          -> scatter through THIS sample's re-derived map, measured
             k-space retained EXACTLY (DEC decode_normalised)
          -> ifft2c -> x_cand_norm (complex, B, H, W)
    Acquired-coefficient fixity is a Class-A A3 gate (<= 1e-5), checked by
    the caller with dec.measured_fixity."""
    us = model.decode_scalars(z, cond_in, mask)      # (B, 13824) f32
    us_np = np.asarray(us.detach().to(torch.float64).cpu().numpy())
    re_s, im_s = unpack_scalars(us_np)
    u_np = unstandardise_free(re_s, im_s, cmap, vecs)  # (B, n_free) c128
    u = torch.from_numpy(np.ascontiguousarray(u_np))
    if y_raw.dim() != 3 or tuple(y_raw.shape[-2:]) != (GRID_H, GRID_W):
        _fail("STATE_LAYOUT_UNEXPECTED",
              f"y_raw must be (B, {GRID_H}, {GRID_W}) complex, got "
              f"{tuple(y_raw.shape)}")
    return dec.decode_normalised(y_raw, amax, u, cmap)


# ---------------------------------------------------------------------------
# Parent artefacts: dual-pin verification + MANDATORY sidecars (IMPL-B
# doctrine; the P3 consumer contract requires verifying the file against
# its sidecar BEFORE use). ONE production loader set, shared by the
# trainer and the Class-A stage -- no test-only replicas. Failures are
# FreeFlowError with the registered PARENT_* codes; the stage layer maps
# them onto its own taxonomy unchanged.
# ---------------------------------------------------------------------------
import json
import os

from seqref_mri.src.preflight_io import (canonical_hash as _canonical_hash,
                                         file_sha256 as _file_sha256,
                                         verify_sidecar as _verify_sidecar)

P3_FACTS_SCHEMA = "seqref-p3-facts/2"
P3_FILE_SHA256 = "27fe84dab5fb2edb5c0c6230d5c4b1dee37db33cc8a76ecd92bb60de783b0bcf"
P3_SEMANTIC_SHA256 = "eaab6776f6ec3a8af39ef5630a6160c9cb97d947758ad983d32eafa1f1cebbb3"

P4S2_FACTS_SCHEMA = "seqref-p4-stats/2"
P4S2_FILE_SHA256 = "db0bf2c2e0cd3bd5e1ed86bf208198199f16df5b2fbbe0cc72903c8791f85684"
P4S2_SEMANTIC_SHA256 = "27bd5b693548d80b2bc837853469b5e62f3da6f0cb2184e5db85fde5c9c91008"

IMPLB_FACTS_SCHEMA = "seqref-implb-facts/1"
IMPLB_FILE_SHA256 = "fb7a5de70cb4b1e747728d4833a02dc135bdd4e0bd8ad9509e3ac5492e4e1b2c"
IMPLB_SEMANTIC_SHA256 = "ae715fdd5aa6031a50f891c7b7ab1d4010330e3d3d6ca0bec095c3b18b7cc034"

GENERATOR_SOURCE_SHA256 = "610cc1d1d7968deebc88f645270e1baefb6589cb56841bdd327450ca1069cb44"

EXPECTED_CORPUS_SLICES = 256
REGISTERED_P4_LOCATIONS = 8448


def _parent_file_sidecar(path: str, expected_file_sha: str,
                         label: str) -> tuple:
    """Existence, file-hash pin, MANDATORY sidecar verification, JSON
    parse. Every mismatch is ERROR: the parent must be THE artefact IMPL
    was registered against, never an unpinned lookalike."""
    if not os.path.isfile(path):
        _fail("PARENT_NOT_FOUND",
              f"the {label} parent artefact does not exist at {path}")
    file_sha = _file_sha256(path)
    if file_sha != expected_file_sha:
        _fail("PARENT_FILE_HASH_MISMATCH",
              f"the {label} parent hashes to {file_sha}, but the "
              f"registered pin is {expected_file_sha}")
    if not os.path.isfile(path + ".sha256"):
        _fail("PARENT_SIDECAR_MISSING",
              f"the {label} parent at {path} has no sidecar; the "
              f"consumer contract requires verifying the file against "
              f"its sidecar before use, so a missing sidecar is ERROR, "
              f"never a skipped check")
    try:
        _verify_sidecar(path)
    except (OSError, RuntimeError) as exc:
        _fail("PARENT_SIDECAR_MISMATCH",
              f"the {label} sidecar does not verify against the pinned "
              f"artefact: {exc}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            art = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _fail("PARENT_STRUCTURE_INVALID",
              f"the {label} parent at {path} is not valid JSON: {exc}")
    return art, file_sha


def _parent_status_semantic(art: dict, *, schema: str, stage: str,
                            expected_semantic_sha: str, label: str) -> str:
    """Schema, stage, authoritative-PASS status, embedded semantic pin."""
    if art.get("schema") != schema:
        _fail("PARENT_SCHEMA_MISMATCH",
              f"expected schema {schema}, got {art.get('schema')!r}; "
              f"IMPL rejects any other schema rather than reinterpret it")
    if art.get("stage") != stage:
        _fail("PARENT_STRUCTURE_INVALID",
              f"the {label} artefact records stage {art.get('stage')!r}, "
              f"expected {stage!r}")
    if not (art.get("authoritative") and art.get("run_mode") == "authoritative"
            and art.get("verdict") == "PASS"):
        _fail("PARENT_NOT_AUTHORITATIVE_PASS",
              f"IMPL inherits only from an authoritative PASS {label} "
              f"artefact (authoritative={art.get('authoritative')!r} "
              f"run_mode={art.get('run_mode')!r} "
              f"verdict={art.get('verdict')!r})")
    sem = art.get("semantic_sha256")
    if sem != expected_semantic_sha:
        _fail("PARENT_SEMANTIC_HASH_MISMATCH",
              f"the {label} parent's embedded semantic_sha256 is {sem}, "
              f"but the registered pin is {expected_semantic_sha}")
    return sem


def load_p3_parent(path: str, *, expected_file_sha: str = P3_FILE_SHA256,
                   expected_semantic_sha: str = P3_SEMANTIC_SHA256) -> dict:
    """Verify the authoritative P3 coordinate-map artefact against BOTH
    registered pins and extract the 256 per-slice bindings. The keyword
    pins default to the registered values; the self-test overrides them
    to reach each gate without touching the production defaults."""
    art, file_sha = _parent_file_sidecar(path, expected_file_sha, "P3")
    sem = _parent_status_semantic(art, schema=P3_FACTS_SCHEMA, stage="P3",
                                  expected_semantic_sha=expected_semantic_sha,
                                  label="P3")
    cmap_block = art.get("coordinate_map") or {}
    if (cmap_block.get("enumeration_rule") != FLATTEN_ORDER_LITERAL
            or cmap_block.get("complex_packing_order")
            != PACKING_ORDER_LITERAL):
        _fail("PARENT_STRUCTURE_INVALID",
              "the P3 artefact's enumeration or packing rule differs "
              "from the live decoder constants; IMPL must not "
              "reinterpret the map order")
    bindings = art.get("per_slice_bindings")
    if not isinstance(bindings, list) \
            or len(bindings) != EXPECTED_CORPUS_SLICES:
        _fail("PARENT_STRUCTURE_INVALID",
              f"the P3 artefact must carry exactly "
              f"{EXPECTED_CORPUS_SLICES} per-slice bindings (the frozen "
              f"corpus); IMPL binds against no other population")
    required = ("dataset_index", "file", "slice_index", "split", "mask_seed",
                "acquired_columns", "mask_sha256", "map_sha256",
                "n_free_complex", "flow_dim_real")
    for k, b in enumerate(bindings):
        missing = [f for f in required if f not in b]
        if missing:
            _fail("PARENT_STRUCTURE_INVALID",
                  f"P3 per-slice binding {k} lacks {missing}; the "
                  f"consumer contract is unbindable")
    return {"path": path, "file_sha256": file_sha, "semantic_sha256": sem,
            "sidecar_verified": True, "bindings": bindings,
            "enumeration_rule": cmap_block.get("enumeration_rule"),
            "packing_order": cmap_block.get("complex_packing_order"),
            "grid_shape": cmap_block.get("grid_shape"),
            "parents_record": art.get("parents")}


def load_p4s2_parent(path: str, *,
                     expected_file_sha: str = P4S2_FILE_SHA256,
                     expected_semantic_sha: str = P4S2_SEMANTIC_SHA256
                     ) -> dict:
    """Verify the authoritative P4 /2 scaling-statistics artefact against
    BOTH registered pins; enforce the PER-LOCATION branch; index the
    applied pairs by physical (r, c). Consumption only."""
    art, file_sha = _parent_file_sidecar(path, expected_file_sha, "P4 /2")
    sem = _parent_status_semantic(art, schema=P4S2_FACTS_SCHEMA,
                                  stage="P4",
                                  expected_semantic_sha=expected_semantic_sha,
                                  label="P4 /2")
    branch = (art.get("branch") or {}).get("selected")
    if branch != "PER-LOCATION":
        _fail("P4_BRANCH_UNEXPECTED",
              f"IMPL consumes only the PER-LOCATION applied affine "
              f"pair; the pinned P4 /2 artefact records branch "
              f"{branch!r}")
    locations = art.get("locations")
    if not isinstance(locations, list) \
            or len(locations) != REGISTERED_P4_LOCATIONS:
        _fail("PARENT_STRUCTURE_INVALID",
              f"the authoritative P4 /2 artefact must carry all "
              f"{REGISTERED_P4_LOCATIONS} eligible locations")
    loc_index = build_location_index(locations)
    for (r, c) in loc_index:      # load-time full-table validity gate
        applied_pair(loc_index, r, c)
    return {"path": path, "file_sha256": file_sha, "semantic_sha256": sem,
            "sidecar_verified": True, "branch": branch,
            "location_index": loc_index,
            "parents_record": art.get("parents")}


def load_implb_parent(path: str, *,
                      expected_file_sha: str = IMPLB_FILE_SHA256,
                      expected_semantic_sha: str = IMPLB_SEMANTIC_SHA256
                      ) -> dict:
    """Verify the authoritative IMPL-B facts artefact against BOTH
    registered pins and extract the calibrated bound. SPLINE_B passes the
    consumption gate (exact float64 equality with the frozen literal)."""
    art, file_sha = _parent_file_sidecar(path, expected_file_sha, "IMPL-B")
    sem = _parent_status_semantic(art, schema=IMPLB_FACTS_SCHEMA,
                                  stage="IMPL-B",
                                  expected_semantic_sha=expected_semantic_sha,
                                  label="IMPL-B")
    cal = art.get("calibration") or {}
    b = require_spline_b(cal.get("B"))
    return {"path": path, "file_sha256": file_sha, "semantic_sha256": sem,
            "sidecar_verified": True, "spline_b": b,
            "calibration": {"q": cal.get("q"),
                            "percentile": cal.get("percentile"),
                            "method": cal.get("method"),
                            "margin": cal.get("margin"),
                            "observation_count":
                                cal.get("observation_count")},
            "parents_record": art.get("parents")}


def enforce_generator_pin(seed_prov: dict) -> None:
    """Same binding as P3/P4/IMPL-B: the executing mask generator is a
    GATE, not a record."""
    got = (seed_prov or {}).get("mask_seed_source_sha256")
    if not (seed_prov or {}).get("resolved") \
            or got != GENERATOR_SOURCE_SHA256:
        _fail("GENERATOR_HASH_MISMATCH",
              f"the executing mask generator hashes to {got}, but the "
              f"registered frame is hash-bound to "
              f"{GENERATOR_SOURCE_SHA256}")
