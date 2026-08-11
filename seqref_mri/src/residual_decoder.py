# SEQREF-DEC v0.3 -- exact-DC residual decoder and free-coordinate map (COMPLEX)
# LIFETIME: KEEP
#
# CHANGELOG
# - v0.3 (2026-07-30): the Fourier pair is now IMPORTED from the frozen
#   fastmri_data module directly, exactly as P2 does, rather than injected from
#   a stage-registered constant. v0.1 registered its own convention with a
#   parser defect (`endswith("shift")` matched "ortho_no_shift"); v0.2 injected
#   the frozen pair but kept the injection plumbing. The convention is now
#   inherited by import and cannot drift.
# - The explicit name->(norm, shifted) table survives for SELF-TEST use only,
#   so the four names can be proven distinguishable. It never governs a run.
# - Adds the vectorised unmeasured conjugate-pair diagnostic: the v0.2 form was
#   a Python loop with a .item() per coordinate, ~1.8M syncs over 256 slices,
#   which would make a NON-VERDICT diagnostic dominate the stage runtime.
# - Docstring corrected: free coefficients are free of registered exact
#   algebraic completion constraints. No statistical independence is claimed --
#   modelling their dependence is the flow's purpose.

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from seqref_mri.src.fastmri_data import fft2c, ifft2c

logger = logging.getLogger("SEQREF-DEC")

__version__ = "0.3"
__abbr__ = "SEQREF-DEC"

# Registered flatten / packing rules. Packed adjacency is NOT physical k-space
# adjacency; that constrains a convolutional scale-up, not this pilot.
P3_FLATTEN_ORDER = "row_major_row_then_ascending_unmeasured_column"
P3_COMPLEX_PACKING_ORDER = "interleaved_real_imag_per_complex_coordinate"

# SELF-TEST ONLY. Explicit mapping, no endswith. The executing transform is the
# frozen imported pair; this table exists so the self-test can demonstrate that
# the four names are genuinely distinct after the v0.1 parser defect.
FFT_CONVENTIONS: dict[str, tuple[str, bool]] = {
    "ortho_no_shift": ("ortho", False),
    "ortho_shift": ("ortho", True),
    "backward_no_shift": ("backward", False),
    "backward_shift": ("backward", True),
}


def reference_fft2(x: torch.Tensor, convention: str) -> torch.Tensor:
    """Reference transform for the SELF-TEST only."""
    if convention not in FFT_CONVENTIONS:
        raise ValueError(f"unregistered convention {convention!r}")
    norm, shifted = FFT_CONVENTIONS[convention]
    if shifted:
        return torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1),
                           norm=norm), dim=(-2, -1))
    return torch.fft.fft2(x, dim=(-2, -1), norm=norm)


def fourier_provenance() -> dict:
    import seqref_mri.src.fastmri_data as _fd
    return {"fft_module": "seqref_mri.src.fastmri_data",
            "fft_symbol": "fft2c", "ifft_symbol": "ifft2c",
            "fft_convention_inherited": True,
            "fft_module_file": getattr(_fd, "__file__", None),
            "note": "the transform pair is imported from the frozen, "
                    "contract-hashed pipeline module; P3 registers no "
                    "convention of its own and none can drift"}


# ---------------------------------------------------------------------------
# Coordinate map
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoordinateMap:
    """Classification of every grid location over the complete fixed mask.

    COMPLEX branch: unmeasured coefficients are FREE OF REGISTERED EXACT
    ALGEBRAIC COMPLETION CONSTRAINTS. The three completion classes exist as
    explicit empties and their emptiness is MEASURED, never assumed.
    """

    height: int
    width: int
    mask_columns: tuple[int, ...]
    free_rows: np.ndarray
    free_cols: np.ndarray
    flatten_order: str = P3_FLATTEN_ORDER
    packing_order: str = P3_COMPLEX_PACKING_ORDER
    conjugate_filled: tuple = ()
    determined_from_partner: tuple = ()
    self_conjugate_real: tuple = ()

    @property
    def n_free_complex(self) -> int:
        return int(self.free_rows.shape[0])

    @property
    def n_acquired(self) -> int:
        return int(self.height * len(self.mask_columns))

    @property
    def flow_dim_real(self) -> int:
        return 2 * self.n_free_complex

    def ordered_coordinates(self) -> list[list[int]]:
        return [[int(r), int(c)] for r, c in zip(self.free_rows, self.free_cols)]

    def class_counts(self) -> dict:
        return {"acquired": self.n_acquired, "free": self.n_free_complex,
                "conjugate_filled_from_free": len(self.conjugate_filled),
                "determined_from_acquired_partner":
                    len(self.determined_from_partner),
                "self_conjugate_real": len(self.self_conjugate_real)}

    def payload(self) -> dict:
        """M3: the artefact CARRIES the map; IMPL consumes it, never rebuilds."""
        coords = self.ordered_coordinates()
        raw = np.asarray(coords, dtype=np.int32)
        return {
            "map_serialized": True,
            "free_coordinates": coords,
            "map_payload_shape": list(raw.shape),
            "map_payload_dtype": "int32",
            "map_payload_order": "C",
            "map_payload_sha256":
                hashlib.sha256(raw.tobytes(order="C")).hexdigest(),
            "map_payload_hash_rule":
                "SHA-256 of the raw int32 C-order bytes; dtype, shape and "
                "order are recorded, so this is a BINARY payload hash and is "
                "deliberately distinct from the canonical structured hash",
            "mask_columns": list(self.mask_columns),
            "grid_shape": [self.height, self.width],
            "flatten_order": self.flatten_order,
            "packing_order": self.packing_order,
            "completion_classes": {
                "conjugate_filled_from_free": list(self.conjugate_filled),
                "determined_from_acquired_partner":
                    list(self.determined_from_partner),
                "self_conjugate_real": list(self.self_conjugate_real)},
        }

    @staticmethod
    def from_payload(payload: dict) -> "CoordinateMap":
        """IMPL entry point: load the verified map instead of rebuilding it."""
        raw = np.asarray(payload["free_coordinates"], dtype=np.int32)
        actual = hashlib.sha256(raw.tobytes(order="C")).hexdigest()
        if actual != payload["map_payload_sha256"]:
            logger.error("map payload SHA mismatch: %s != %s", actual,
                         payload["map_payload_sha256"])
            raise ValueError("coordinate map payload does not match its SHA")
        h, w = payload["grid_shape"]
        return CoordinateMap(
            height=int(h), width=int(w),
            mask_columns=tuple(int(c) for c in payload["mask_columns"]),
            free_rows=np.ascontiguousarray(raw[:, 0].astype(np.int64)),
            free_cols=np.ascontiguousarray(raw[:, 1].astype(np.int64)),
            flatten_order=payload["flatten_order"],
            packing_order=payload["packing_order"])


def build_coordinate_map(mask_columns: Sequence[int], height: int,
                         width: int) -> CoordinateMap:
    """PRODUCTION enumeration: vectorised argwhere over a broadcast 2-D mask.

    The audit's independent enumeration is a separate pure-Python
    implementation in the stage script, so the two cannot share a defect.
    """
    cols = sorted(int(c) for c in mask_columns)
    if len(set(cols)) != len(cols):
        logger.error("duplicate acquired columns: %s", cols)
        raise ValueError("mask columns contain duplicates")
    if cols and (cols[0] < 0 or cols[-1] >= width):
        logger.error("acquired column out of range for width %d: %s", width,
                     cols)
        raise ValueError("mask column index out of range")
    col_mask = np.zeros(width, dtype=bool)
    if cols:
        col_mask[np.asarray(cols, dtype=np.int64)] = True
    free_rc = np.argwhere(~np.broadcast_to(col_mask[None, :], (height, width)))
    logger.info("map: %d acquired columns, n_free_complex=%d, flow_dim=%d",
                len(cols), free_rc.shape[0], 2 * free_rc.shape[0])
    return CoordinateMap(height=height, width=width, mask_columns=tuple(cols),
                         free_rows=np.ascontiguousarray(free_rc[:, 0]),
                         free_cols=np.ascontiguousarray(free_rc[:, 1]))


# ---------------------------------------------------------------------------
# gather / scatter
# ---------------------------------------------------------------------------

def gather_unmeasured(k: torch.Tensor, cmap: CoordinateMap) -> torch.Tensor:
    if tuple(k.shape[-2:]) != (cmap.height, cmap.width):
        raise ValueError(f"grid {tuple(k.shape[-2:])} != map "
                         f"({cmap.height},{cmap.width})")
    if not torch.is_complex(k):
        raise ValueError("gather_unmeasured requires a complex tensor")
    rows = torch.as_tensor(cmap.free_rows, dtype=torch.long, device=k.device)
    cols = torch.as_tensor(cmap.free_cols, dtype=torch.long, device=k.device)
    return k[..., rows, cols]


def scatter_unmeasured(u: torch.Tensor, cmap: CoordinateMap) -> torch.Tensor:
    """Acquired locations and every completion class remain EXACTLY zero."""
    if not torch.is_complex(u):
        raise ValueError("scatter_unmeasured requires a complex tensor")
    if u.shape[-1] != cmap.n_free_complex:
        raise ValueError(f"u length {u.shape[-1]} != n_free_complex "
                         f"{cmap.n_free_complex}")
    grid = torch.zeros((*u.shape[:-1], cmap.height, cmap.width),
                       dtype=u.dtype, device=u.device)
    rows = torch.as_tensor(cmap.free_rows, dtype=torch.long, device=u.device)
    cols = torch.as_tensor(cmap.free_cols, dtype=torch.long, device=u.device)
    grid[..., rows, cols] = u
    return grid


def column_mask_tensor(cmap: CoordinateMap, device, dtype) -> torch.Tensor:
    """1-D (W,) column mask broadcasting over rows -- the locked P2 convention,
    matching MaskedFourierOperator._m: cast to the k-space dtype and multiply."""
    m = torch.zeros(cmap.width, dtype=torch.bool, device=device)
    if cmap.mask_columns:
        m[torch.as_tensor(cmap.mask_columns, dtype=torch.long, device=device)] = True
    return m.to(dtype)


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

def decode_normalised(y_raw, amax, u_norm, cmap) -> torch.Tensor:
    """PRIMARY registered decoder -- normalised k-space assembly (concept §2b).

        y_norm      = y_raw / amax
        k_norm      = M . y_norm + scatter(u_norm)
        x_cand_norm = F^H(k_norm)

    u_norm is untouched by any scaling round trip, so no roundoff is injected
    into the coordinates the flow learns.
    """
    if not torch.is_complex(y_raw) or not torch.is_complex(u_norm):
        raise ValueError("decode_normalised requires complex y_raw and u_norm")
    m = column_mask_tensor(cmap, y_raw.device, y_raw.dtype)
    return ifft2c((y_raw / amax) * m + scatter_unmeasured(u_norm, cmap))


def decode_raw_path(y_raw, amax, u_norm, cmap) -> torch.Tensor:
    """NON-BLOCKING operation-order sensitivity and lineage path.

        k_raw       = M . y_raw + amax * scatter(u_norm)
        x_cand_norm = F^H(k_raw) / amax

    The A3-registered x0 order. At u = 0 it is EXPECTED to reproduce cond_in
    bitwise, since P2 measured x0_rel_error against exactly this construction.
    Recorded as a diagnostic; it NEVER gates.
    """
    if not torch.is_complex(y_raw) or not torch.is_complex(u_norm):
        raise ValueError("decode_raw_path requires complex y_raw and u_norm")
    m = column_mask_tensor(cmap, y_raw.device, y_raw.dtype)
    return ifft2c(y_raw * m + amax * scatter_unmeasured(u_norm, cmap)) / amax


def measured_fixity(x_cand_norm, y_raw, amax, cmap) -> tuple[float, float]:
    """C3c primary guarantee: M F x_cand_norm == y_raw / amax.

    Complex magnitude per Fourier pixel; denominator max|y_raw/amax| over
    MEASURED locations. The mask is applied explicitly rather than relying on
    A3's finding that the stored y is already masked -- stated, not assumed.
    """
    m = column_mask_tensor(cmap, y_raw.device, y_raw.dtype)
    lhs = fft2c(x_cand_norm) * m
    rhs = (y_raw / amax) * m
    abs_err = float(torch.max(torch.abs(lhs - rhs)).item())
    denom = float(torch.max(torch.abs(rhs)).item())
    if not np.isfinite(denom) or denom <= 0.0:
        logger.error("fixity denominator invalid: %r", denom)
        raise ValueError("fixity denominator is not finite and positive")
    return abs_err, abs_err / denom


def raw_fixity(x_cand_norm, y_raw, amax, cmap) -> tuple[float, float]:
    """NON-BLOCKING fp32 scaling round-trip probe: M F (amax * x) ~= y_raw.
    Follows from the primary guarantee by linearity in exact arithmetic."""
    m = column_mask_tensor(cmap, y_raw.device, y_raw.dtype)
    lhs = fft2c(amax * x_cand_norm) * m
    rhs = y_raw * m
    abs_err = float(torch.max(torch.abs(lhs - rhs)).item())
    denom = float(torch.max(torch.abs(rhs)).item())
    if not np.isfinite(denom) or denom <= 0.0:
        logger.error("raw fixity denominator invalid: %r", denom)
        raise ValueError("raw fixity denominator is not finite and positive")
    return abs_err, abs_err / denom


# ---------------------------------------------------------------------------
# Vectorised unmeasured conjugate-pair diagnostic (NON-VERDICT)
# ---------------------------------------------------------------------------

def conjugate_pair_index(cmap: CoordinateMap) -> dict:
    """Precompute the unmeasured conjugate-pair index ONCE from the map.

    Partner of (r, c) on the unshifted DFT is ((-r) mod H, (-c) mod W), the
    same pairing P1 recorded. Only pairs whose PARTNER is also unmeasured are
    tested; each pair is kept once, lexicographically ordered.
    """
    H, W = cmap.height, cmap.width
    r = cmap.free_rows.astype(np.int64)
    c = cmap.free_cols.astype(np.int64)
    pr, pc = (-r) % H, (-c) % W
    acquired = np.zeros(W, dtype=bool)
    if cmap.mask_columns:
        acquired[np.asarray(cmap.mask_columns, dtype=np.int64)] = True
    partner_free = ~acquired[pc]
    keep = partner_free & ((r * W + c) <= (pr * W + pc))
    return {"rows": r[keep], "cols": c[keep], "prows": pr[keep],
            "pcols": pc[keep], "n_pairs": int(keep.sum()),
            "n_self_paired": int(((r == pr) & (c == pc) & keep).sum()),
            "n_free_without_free_partner": int((~partner_free).sum()),
            "pairing": "((-u) mod H, (-v) mod W) on the unshifted DFT"}


def conjugate_pair_violation(k: torch.Tensor, idx: dict) -> np.ndarray:
    """One vectorised reduction per slice, relative to max|K| on that slice."""
    den = float(torch.max(torch.abs(k)).item())
    if not np.isfinite(den) or den <= 0.0:
        raise ValueError("conjugate-pair denominator max|K| invalid")
    dev = k.device
    a = k[torch.as_tensor(idx["rows"], dtype=torch.long, device=dev),
          torch.as_tensor(idx["cols"], dtype=torch.long, device=dev)]
    b = k[torch.as_tensor(idx["prows"], dtype=torch.long, device=dev),
          torch.as_tensor(idx["pcols"], dtype=torch.long, device=dev)]
    return (torch.abs(a - torch.conj(b)) / den).detach().cpu().numpy()


def to_precision(t: torch.Tensor, double: bool) -> torch.Tensor:
    """E2: genuine precision variants, distinct from operation-order change."""
    if not torch.is_complex(t):
        return t.to(torch.float64 if double else torch.float32)
    return t.to(torch.complex128 if double else torch.complex64)
