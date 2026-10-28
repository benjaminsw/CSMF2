# SEQREF-MRI-V05LTR v0.2 -- scripts.v05lt_rules
# LIFETIME: KEEP
# =============================================================================
# Purpose: stopping and selection rules of SEQREF-MRI-V05LT, kept SEPARATE
#   (195 §4, §5; 194 §6.0, §6.1, §7):
#   * Plateau (stopping only): metrics m in {R0 PSNR (higher better), NLL
#     per free real dimension (lower better)}. After epoch 1: ref_m = value,
#     c_m = 0. Epoch e >= 2: improvement d = new - ref (PSNR) or ref - new
#     (NLL), computed in float64; if d > tol_m -> ref_m = new, c_m = 0, else
#     c_m += 1. Stop at the end of epoch e iff e >= 30 and c_PSNR >= 10 and
#     c_NLL >= 10. Reaching epoch 100 without stopping -> "budget exhausted;
#     plateau not established". Epoch 0 is never a check. Epochs must be
#     consecutive.
#   * Selection (never stops anything): best validation R0 PSNR, replaced
#     only on strict improvement (ties keep the earliest epoch); best NLL
#     (secondary, strict decrease). Epoch 0 is never a candidate.
#   * Milestone epochs 3, 10, 20, 30, 50, 75, 100.
#   * Exact JSON-serialisable state for resume (to_dict / from_dict).
# CONVENTION: logger.error + V05LTError(code); no fallback, no placeholder.
# Changelog (v0.2, 2026-10-21; I1.6(b), open fix 11):
#   * from_dict() validates restored state before accepting it: epoch range,
#     integer counters 0..last_epoch-1, finite references, reset records
#     inside the epoch range, best records with integer epoch <= last epoch
#     and finite value (RULE_STATE_INVALID).
# Changelog (NEW in v0.1, 2026-10-21):
#   * Introduced under I1.6(a) (SEQREF-MRI-V05LTI16A v0.1): plateau and
#     selection as independent state machines with resume serialisation.
# =============================================================================
from __future__ import annotations

import logging
import math

from seqref_mri.scripts import v05lt_common as C

logger = logging.getLogger("seqref_mri.v05lt_rules")

__version__ = "0.2"
__abbr__ = "SEQREF-MRI-V05LTR"

PSNR_TOL = 0.01            # dB (194 §6)
NLL_TOL = 0.001            # nats per free real dimension (194 §6)
PATIENCE = 10
MIN_EPOCHS = 30
MAX_EPOCHS = 100
MILESTONES = (3, 10, 20, 30, 50, 75, 100)

STOPPED = "stopped_plateau"
EXHAUSTED = "budget exhausted; plateau not established"
CONTINUE = "continue"


def _finite(x, what: str) -> float:
    if not (isinstance(x, float) and math.isfinite(x)):
        C.fail("RULE_VALUE_INVALID", f"{what}: {x!r}")
    return x


class Plateau:
    """Stopping rule only (194 §6.0)."""

    def __init__(self):
        self.last_epoch = 0
        self.ref = {"psnr": None, "nll": None}
        self.count = {"psnr": 0, "nll": 0}
        self.resets = []            # (epoch, metric) for reporting

    def update(self, epoch: int, psnr: float, nll: float) -> str:
        C.require(isinstance(epoch, int) and epoch >= 1, "PLATEAU_EPOCH_INVALID",
                  f"epoch {epoch!r} (epoch 0 is never a check)")
        C.require(epoch == self.last_epoch + 1, "PLATEAU_EPOCH_ORDER",
                  f"epoch {epoch} after {self.last_epoch}")
        C.require(epoch <= MAX_EPOCHS, "PLATEAU_BEYOND_BUDGET", str(epoch))
        v = {"psnr": _finite(psnr, "psnr"), "nll": _finite(nll, "nll")}
        if epoch == 1:
            self.ref = dict(v)
            self.count = {"psnr": 0, "nll": 0}
        else:
            for m, tol in (("psnr", PSNR_TOL), ("nll", NLL_TOL)):
                d = (v[m] - self.ref[m]) if m == "psnr" else (self.ref[m] - v[m])
                if d > tol:
                    self.ref[m] = v[m]
                    self.count[m] = 0
                    self.resets.append([epoch, m])
                else:
                    self.count[m] += 1
        self.last_epoch = epoch
        if epoch >= MIN_EPOCHS and self.count["psnr"] >= PATIENCE \
                and self.count["nll"] >= PATIENCE:
            return STOPPED
        if epoch == MAX_EPOCHS:
            return EXHAUSTED
        return CONTINUE

    def to_dict(self) -> dict:
        return {"last_epoch": self.last_epoch, "ref": dict(self.ref),
                "count": dict(self.count),
                "resets": [list(r) for r in self.resets]}

    @classmethod
    def from_dict(cls, d: dict) -> "Plateau":
        _check_plateau_state(d)
        p = cls()
        p.last_epoch = int(d["last_epoch"])
        p.ref = {"psnr": d["ref"]["psnr"], "nll": d["ref"]["nll"]}
        p.count = {"psnr": int(d["count"]["psnr"]),
                   "nll": int(d["count"]["nll"])}
        p.resets = [list(r) for r in d["resets"]]
        return p


class Selection:
    """Checkpoint selection only (194 §7); independent of stopping."""

    def __init__(self):
        self.best_psnr = None       # {"epoch", "value"}
        self.best_nll = None

    def update(self, epoch: int, psnr: float, nll: float) -> dict:
        """-> {"psnr": bool, "nll": bool}: which best record changed."""
        C.require(isinstance(epoch, int) and epoch >= 1, "SELECTION_EPOCH_INVALID",
                  f"epoch {epoch!r} (epoch 0 is never a candidate)")
        psnr, nll = _finite(psnr, "psnr"), _finite(nll, "nll")
        ch = {"psnr": False, "nll": False}
        if self.best_psnr is None or psnr > self.best_psnr["value"]:
            self.best_psnr = {"epoch": epoch, "value": psnr}
            ch["psnr"] = True
        if self.best_nll is None or nll < self.best_nll["value"]:
            self.best_nll = {"epoch": epoch, "value": nll}
            ch["nll"] = True
        return ch

    def to_dict(self) -> dict:
        return {"best_psnr": self.best_psnr, "best_nll": self.best_nll}

    @classmethod
    def from_dict(cls, d: dict, last_epoch: int | None = None) -> "Selection":
        _check_selection_state(d, last_epoch)
        s = cls()
        s.best_psnr = d["best_psnr"]
        s.best_nll = d["best_nll"]
        return s


def _bad(msg: str):
    C.fail("RULE_STATE_INVALID", msg)


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _check_plateau_state(d) -> None:
    if not isinstance(d, dict) or set(d) != {"last_epoch", "ref", "count",
                                             "resets"}:
        _bad(f"plateau keys {sorted(d) if isinstance(d, dict) else d!r}")
    e = d["last_epoch"]
    if not (_is_int(e) and 0 <= e <= MAX_EPOCHS):
        _bad(f"last_epoch {e!r}")
    for m in ("psnr", "nll"):
        r, c = d["ref"].get(m, "missing"), d["count"].get(m, "missing")
        if e == 0:
            if r is not None or c != 0:
                _bad(f"{m}: epoch 0 state must be empty ({r!r}, {c!r})")
            continue
        if not (isinstance(r, float) and math.isfinite(r)):
            _bad(f"{m} reference {r!r}")
        if not (_is_int(c) and 0 <= c <= e - 1):
            _bad(f"{m} counter {c!r} at epoch {e}")
    for rs in d["resets"]:
        if not (isinstance(rs, list) and len(rs) == 2 and _is_int(rs[0])
                and 2 <= rs[0] <= e and rs[1] in ("psnr", "nll")):
            _bad(f"reset record {rs!r}")


def _check_selection_state(d, last_epoch) -> None:
    if not isinstance(d, dict) or set(d) != {"best_psnr", "best_nll"}:
        _bad(f"selection keys {sorted(d) if isinstance(d, dict) else d!r}")
    for k in ("best_psnr", "best_nll"):
        b = d[k]
        if b is None:
            if last_epoch not in (None, 0):
                _bad(f"{k} missing after epoch {last_epoch}")
            continue
        if not (isinstance(b, dict) and set(b) == {"epoch", "value"}
                and _is_int(b["epoch"]) and 1 <= b["epoch"] <= MAX_EPOCHS
                and isinstance(b["value"], float) and math.isfinite(b["value"])):
            _bad(f"{k} record {b!r}")
        if last_epoch is not None and b["epoch"] > last_epoch:
            _bad(f"{k} epoch {b['epoch']} after last epoch {last_epoch}")


def is_milestone(epoch: int) -> bool:
    return epoch in MILESTONES
