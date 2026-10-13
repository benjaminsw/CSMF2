# =============================================================================
# SEQREF-MRI-V04G1 v0.1 -- scripts.v04_gate
# LIFETIME: KEEP
# Purpose: V04 reliability gate (V04SPEC v1.0 article 176 FROZEN SS17;
#          V04PLAN v1.0 article 178 FROZEN SS7). Six gates, ALL must PASS
#          before authoritative training:
#            (a) pipeline bit-exactness -- frozen V02/V03 manifests +
#                tensor SHAs vs the pinned reference table (f01; exact
#                match, else MANIFEST_CONTENT_MISMATCH)
#            (b) lambda=0 step parity -- combined loss at lam=0 vs the
#                density-only path: relative difference EXACTLY 0.0
#                (value + gradient digest)
#            (c) probe-record hard gates -- hash-verified record; schema,
#                l_rec_scale = 1e7, weight-state seed 20260813, batch-list
#                hash, lambda_2' in band, lambda factor 0.5 (via the V04T
#                contract loader) + the recorded parameter sha256 must
#                equal the fresh-init V04M parameter digest
#            (d) invertibility round-trip -- full model, max abs err
#                <= 1e-4 (f12 protocol)
#            (e) log-det agreement -- f13 protocol (8-dim FD slogdet,
#                normalised dev <= 1e-3) + f16 fwd/inv cancellation
#                <= 1e-5
#            (f) conditioning -- f14 numerical (raw pre-clamp outputs
#                finite, saturation exactly 0.0) + f15 conditioning-path
#                connectivity, AMENDED prong set per frozen article 180
# Gate semantics: a nonconforming gate is recorded FAIL with its typed
#          code; the record + SHA-256 sidecar are ALWAYS written; any
#          FAIL -> typed GATE_FAILURE abort AFTER the record is written
#          (hard abort, no registered claim). Missing/tampered INPUTS
#          raise typed errors immediately (GATE_INPUT_MISSING /
#          GATE_INPUT_DRIFT). No fallback, no mock, no placeholder, no
#          silent pass.
# Measurement single-source: gates (d)/(e)/(f) call the fixture protocol
#   functions in scripts.v04_fixtures via lazy import (fixtures never
#   import this module at top level).
# FORM field: with synthetic batches the record carries
#   form="preliminary-synthetic" (f14's authoritative form requires the
#   100 registered probe batches); authoritative runs carry
#   form="authoritative". --mode run REFUSES without real inputs
#   (fail-closed): manifests dir + reference table + probe record +
#   data-root + commit pin are all required.
# CONVENTION: logger.error + typed raise (V04G1Error). PASS(0)/ERROR(2).
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

from seqref_mri.scripts import v04_objective as v04o
from seqref_mri.scripts import v04_probe as v04p
from seqref_mri.scripts import v04_train as v04t
from seqref_mri.scripts.v04_model import build_v04_model
from seqref_mri.scripts.v02_manifests import canonical_json
from seqref_mri.src import free_flow_runtime as ffr

logger = logging.getLogger("seqref_mri.v04_gate")

__version__ = "0.1"
__abbr__ = "SEQREF-MRI-V04G1"

GATE_SCHEMA = "seqref-v04-gate-record/1"
REF_SCHEMA = "seqref-v04-pipeline-reference/1"
GATE_NAMES = ("a", "b", "c", "d", "e", "f")      # 178 SS7, locked order


class V04G1Error(RuntimeError):
    """Single typed identity for the V04 reliability gate."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V04G1Error(code, message)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exception_code(exc: Exception, prefix: str) -> str:
    """Typed code for a recorded gate failure: preserve the originator's
    typed code when present; otherwise namespace the untyped exception
    under the gate's own typed prefix -- a bare Python type name is never
    a recorded code."""
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    return f"{prefix}:{type(exc).__name__}"


def _fixtures():
    """Lazy single-source import of the measurement protocols."""
    from seqref_mri.scripts import v04_fixtures as fx
    return fx


# ---------------------------------------------------------------------------
# Gate (a): pipeline bit-exactness (f01)
# ---------------------------------------------------------------------------

def load_reference_table(path) -> dict:
    """Hash-verified pinned reference table: {schema, artifacts:
    {relpath: sha256}}."""
    p = Path(path)
    side = p.with_suffix(p.suffix + ".sha256")
    if not p.is_file() or not side.is_file():
        _fail("GATE_INPUT_MISSING",
              f"pipeline reference table or sidecar missing at {p}")
    recorded = side.read_text().split()[0]
    if _file_sha256(p) != recorded:
        _fail("GATE_INPUT_DRIFT",
              "reference table content diverges from its sidecar")
    ref = json.loads(p.read_text())
    if ref.get("schema") != REF_SCHEMA \
            or not isinstance(ref.get("artifacts"), dict) or not ref["artifacts"]:
        _fail("GATE_INPUT_SCHEMA_MISMATCH",
              f"reference table schema {ref.get('schema')!r} != "
              f"{REF_SCHEMA!r} or empty artifacts")
    return ref


def gate_a_pipeline_bitexact(reference_path, target_dir) -> dict:
    """f01: every pinned artifact must exist with the exact pinned sha256;
    unexpected extra files count as drift. Any drift -> FAIL with
    MANIFEST_CONTENT_MISMATCH."""
    ref = load_reference_table(reference_path)
    target = Path(target_dir)
    drift = []
    for rel, sha in sorted(ref["artifacts"].items()):
        f = target / rel
        if not f.is_file():
            drift.append({"file": rel, "reason": "missing"})
        elif _file_sha256(f) != sha:
            drift.append({"file": rel, "reason": "sha256 drift"})
    pinned = set(ref["artifacts"])
    for f in sorted(target.rglob("*")):
        if f.is_file() and f.suffix != ".sha256":
            rel = f.relative_to(target).as_posix()
            if rel not in pinned:
                drift.append({"file": rel, "reason": "unexpected file"})
    if drift:
        return {"status": "FAIL", "code": "MANIFEST_CONTENT_MISMATCH",
                "drift": drift}
    return {"status": "PASS", "artifacts_verified": len(pinned)}


# ---------------------------------------------------------------------------
# Gate (b): lambda=0 step parity
# ---------------------------------------------------------------------------

def gate_b_lambda0_parity(model, batch) -> dict:
    """Step-1 loss under lam=0 vs the density-only path: relative
    difference EXACTLY 0.0 (value bit-equality + gradient digest).
    Total: any measurement-level failure is recorded FAIL with its
    code, never raised -- the aggregate record must always be written."""
    try:
        u, cond_in, mask = (batch["u_scaled"], batch["cond_in"],
                            batch["mask"])
        ref = v04o.density_only(model, u, cond_in, mask)
        l0 = v04o.combined_loss(model, u, cond_in, mask, lam=0.0,
                                y_raw=None, amax=None, tgt_norm=None,
                                maps=[])
        if float(l0) != float(ref):
            return {"status": "FAIL", "code": "LAMBDA0_PATH_MISMATCH",
                    "detail": "lam=0 value differs from the density-only "
                              "path"}
        model.zero_grad(set_to_none=True)
        v04o.density_only(model, u, cond_in, mask).backward()
        d_ref = _grad_digest(model)
        model.zero_grad(set_to_none=True)
        v04o.combined_loss(model, u, cond_in, mask, lam=0.0, y_raw=None,
                           amax=None, tgt_norm=None, maps=[]).backward()
        d_l0 = _grad_digest(model)
        model.zero_grad(set_to_none=True)
    except Exception as exc:
        return {"status": "FAIL",
                "code": _exception_code(exc, "LAMBDA0_STEP_EXCEPTION"),
                "detail": str(exc)}
    if d_l0 != d_ref:
        return {"status": "FAIL", "code": "LAMBDA0_PATH_MISMATCH",
                "detail": "lam=0 gradients differ from the density-only "
                          "path"}
    return {"status": "PASS", "relative_difference": 0.0,
            "gradient_digest": d_ref}


def _grad_digest(model) -> str:
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        g = p.grad
        h.update(n.encode())
        h.update(b"\x01" if g is not None else b"\x00")
        if g is not None:
            h.update(g.detach().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Gate (c): probe-record hard gates (schema + locked fields + hashes)
# ---------------------------------------------------------------------------

def gate_c_probe_record(probe_record_path,
                        expected_batch_list_sha256=None) -> dict:
    """Hash-verified probe record, all locked fields conform (via the V04T
    contract loader), the recorded parameter sha256 must equal the
    fresh-init V04M parameter digest, and -- at the authoritative run --
    the recorded batch-list sha256 must equal the list recomputed from
    the data."""
    try:
        rec = v04t.load_probe_record(probe_record_path)
        fresh = build_v04_model()            # registered init seed 20260813
        digest = v04p.parameter_digest(fresh)
    except Exception as exc:
        return {"status": "FAIL",
                "code": _exception_code(exc, "PROBE_RECORD_EXCEPTION"),
                "detail": str(exc)}
    if rec.get("parameter_sha256") != digest:
        return {"status": "FAIL", "code": "PROBE_RECORD_CONFIG_MISMATCH",
                "detail": "recorded parameter_sha256 != the fresh-init "
                          "V04M digest"}
    if expected_batch_list_sha256 is not None \
            and rec.get("batch_list_sha256") != expected_batch_list_sha256:
        return {"status": "FAIL", "code": "PROBE_RECORD_CONFIG_MISMATCH",
                "detail": "recorded batch_list_sha256 != the recomputed "
                          "probe batch list"}
    return {"status": "PASS", "lambda_prime": rec["lambda_prime"],
            "batch_list_sha256": rec["batch_list_sha256"],
            "parameter_sha256": digest,
            "record_sha256": _file_sha256(Path(probe_record_path))}


# ---------------------------------------------------------------------------
# Gates (d)/(e)/(f): measurement protocols single-sourced from the fixtures
# ---------------------------------------------------------------------------

def _protocol_gate(name: str, fn, extra=None,
                   exc_prefix: str = "MEASUREMENT_EXCEPTION") -> dict:
    try:
        fn()
    except Exception as exc:
        code = _exception_code(exc, exc_prefix)
        return {"status": "FAIL", "code": code, "detail": str(exc)}
    out = {"status": "PASS"}
    if extra:
        out.update(extra)
    return out


def _f12_over_batches(model, batches) -> float:
    """Round-trip over EXPLICIT batches; returns max abs error."""
    fx = _fixtures()
    worst = 0.0
    for u, cond_in, mask in batches:
        z, _ = model.encode_free(u, cond_in, mask)
        err = (model.decode_scalars(z, cond_in, mask) - u).abs().max()
        if not torch.isfinite(err):
            _fail("INVERTIBILITY_FAILURE",
                  "round-trip produced a non-finite error")
        worst = max(worst, float(err))
    if worst > fx.F12_ROUNDTRIP_TOL:
        _fail("INVERTIBILITY_FAILURE",
              f"round-trip max abs err {worst:.3e} > "
              f"{fx.F12_ROUNDTRIP_TOL:.1e}")
    return worst


def _f16_over_batches(model, batches) -> float:
    """fwd/inv log-det cancellation over EXPLICIT batches; returns the
    worst residual (f16 protocol)."""
    fx = _fixtures()
    worst = 0.0
    for u, cond_in, mask in batches:
        h = model.condition(cond_in, mask)
        s_fwd, s_inv = [], []
        handles = []
        for layer in model.flow.layers:
            def make(store, s_max=layer.s_max):
                def hook(mod, inp, out):
                    s = s_max * torch.tanh(out.chunk(2, dim=-1)[0])
                    store.append(s.detach().sum(dim=-1))
                return hook
            handles.append(layer.post.register_forward_hook(make(s_fwd)))
        try:
            _, ldj_fwd = model.encode_free(u, cond_in, mask)
        finally:
            for hd in handles:
                hd.remove()
        z, _ = model.encode_free(u, cond_in, mask)
        handles = []
        for layer in model.flow.layers:
            def make2(store, s_max=layer.s_max):
                def hook(mod, inp, out):
                    s = s_max * torch.tanh(out.chunk(2, dim=-1)[0])
                    store.append(s.detach().sum(dim=-1))
                return hook
            handles.append(layer.post.register_forward_hook(make2(s_inv)))
        try:
            model.decode_scalars(z, cond_in, mask)
        finally:
            for hd in handles:
                hd.remove()
        ldj_inv = -torch.stack(list(reversed(s_inv))).sum(dim=0)
        resid = (ldj_fwd + ldj_inv).abs().max().item()
        if resid > worst:
            worst = resid
    if worst > fx.F16_CANCEL_TOL:
        _fail("LOGDET_MISMATCH",
              f"fwd/inv cancellation residual {worst:.3e} > "
              f"{fx.F16_CANCEL_TOL:.0e}")
    return worst


def gate_d_invertibility(model, gen, batches=None) -> dict:
    fx = _fixtures()
    if batches is None:
        return _protocol_gate(
            "d", lambda: fx.f12_invertibility(model, gen),
            {"tolerance": fx.F12_ROUNDTRIP_TOL,
             "form": "preliminary-synthetic"},
            exc_prefix="INVERTIBILITY_EXCEPTION")
    return _protocol_gate(
        "d", lambda: _f12_over_batches(model, batches),
        {"tolerance": fx.F12_ROUNDTRIP_TOL, "form": "authoritative"},
        exc_prefix="INVERTIBILITY_EXCEPTION")


def gate_e_logdet(model, gen, batches=None) -> dict:
    fx = _fixtures()
    r13 = _protocol_gate("e-f13", fx.f13_jacobian_logdet,
                         {"rel_tol": fx.F13_REL_TOL},
                         exc_prefix="LOGDET_EXCEPTION")
    if r13["status"] != "PASS":
        return {"status": "FAIL", "code": r13["code"],
                "detail": "f13: " + r13.get("detail", "")}
    if batches is None:
        r16 = _protocol_gate(
            "e-f16", lambda: fx.f16_logdet_cancellation(model, gen),
            {"cancel_tol": fx.F16_CANCEL_TOL,
             "form": "preliminary-synthetic"},
            exc_prefix="LOGDET_EXCEPTION")
    else:
        r16 = _protocol_gate(
            "e-f16", lambda: _f16_over_batches(model, batches),
            {"cancel_tol": fx.F16_CANCEL_TOL, "form": "authoritative"},
            exc_prefix="LOGDET_EXCEPTION")
    if r16["status"] != "PASS":
        return {"status": "FAIL", "code": r16["code"],
                "detail": "f16: " + r16.get("detail", "")}
    return {"status": "PASS", "f13_rel_tol": fx.F13_REL_TOL,
            "f16_cancel_tol": fx.F16_CANCEL_TOL}


def _f14_over_batches(model, batches) -> None:
    """f14 protocol over EXPLICIT batches (authoritative form: the 100
    registered probe batches). Mirrors f14_numerical_conditioning but
    consumes the given batch list instead of synthetic draws."""
    fx = _fixtures()
    for b, (u, cond_in, mask) in enumerate(batches):
        h = model.condition(cond_in, mask)
        x = u.to(torch.float32)
        for i, layer in enumerate(model.flow.layers):
            x1, _ = layer._split(x)
            raw, s, t = fx._post_outputs(layer, x1, h)
            if not torch.isfinite(raw).all() or not torch.isfinite(s).all() \
                    or not torch.isfinite(t).all():
                _fail("CONDITIONING_FAILURE",
                      f"layer {i} batch {b}: non-finite raw/s/t outputs")
            if bool((s.abs() > layer.s_max).any()):
                _fail("CONDITIONING_FAILURE",
                      f"layer {i} batch {b}: |s| exceeds s_max")
            if layer.last_s_clamp_frac != 0.0:
                _fail("CONDITIONING_FAILURE",
                      f"layer {i} batch {b}: last_s_clamp_frac="
                      f"{layer.last_s_clamp_frac} != 0.0 exactly at init")
            x, _ = layer(x, h)


def gate_f_conditioning(model, gen, authoritative_batches=None) -> dict:
    """f14 numerical + f15 amended (article 180). With explicit batches
    f14 runs its authoritative 100-probe-batch form; without them it runs
    the preliminary synthetic form -- the record's form field says which."""
    fx = _fixtures()
    if authoritative_batches is None:
        form = "preliminary-synthetic"
        r14 = _protocol_gate("f-f14", lambda: fx.f14_numerical_conditioning(
            model, gen), exc_prefix="CONDITIONING_EXCEPTION")
    else:
        form = "authoritative"
        r14 = _protocol_gate("f-f14", lambda: _f14_over_batches(
            model, authoritative_batches),
            exc_prefix="CONDITIONING_EXCEPTION")
    if r14["status"] != "PASS":
        return {"status": "FAIL", "code": r14["code"], "form": form,
                "detail": "f14: " + r14.get("detail", "")}
    r15 = _protocol_gate("f-f15",
                         lambda: fx.f15_conditioning_connectivity(model),
                         exc_prefix="CONDITIONING_EXCEPTION")
    if r15["status"] != "PASS":
        return {"status": "FAIL", "code": r15["code"], "form": form,
                "detail": "f15: " + r15.get("detail", "")}
    return {"status": "PASS", "form": form,
            "f15_prongs": ["i'", "ii'", "iii", "iv'", "v'", "vi'"],
            "f15_amendment": {"article": 180, "status": "FROZEN"}}


# ---------------------------------------------------------------------------
# Aggregate runner: record ALWAYS written; any FAIL -> GATE_FAILURE abort
# ---------------------------------------------------------------------------

def run_gates(gates: dict, out_dir, form: str) -> dict:
    """gates: {"a": {...}, ...} already evaluated. Writes the record +
    sidecar; raises GATE_FAILURE if any gate is not PASS."""
    if set(gates) != set(GATE_NAMES):
        _fail("GATE_SCHEMA_MISMATCH",
              f"gate results cover {sorted(gates)}, expected "
              f"{list(GATE_NAMES)}")
    record = {
        "schema": GATE_SCHEMA,
        "script": f"{__abbr__} v{__version__}",
        "spec": {"article": 176, "status": "FROZEN"},
        "plan": {"article": 178, "status": "FROZEN"},
        "form": form,
        "gates": gates,
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "platform": platform.platform()},
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "v04_gate_record.json"
    path.write_bytes(canonical_json(record))
    digest = _file_sha256(path)
    (out / "v04_gate_record.json.sha256").write_text(
        f"{digest}  {path.name}\n")
    logger.info("[%s] gate record written: %s (sha256 %s...)", __abbr__,
                path, digest[:12])
    failed = {k: v.get("code") for k, v in gates.items()
              if v.get("status") != "PASS"}
    if failed:
        logger.error("[%s] gates FAILED: %s -- hard abort, no registered "
                     "claim", __abbr__, failed)
        _fail("GATE_FAILURE",
              f"reliability gates failed: {failed}; record written to "
              f"{path}")
    return record


# ---------------------------------------------------------------------------
# Authoritative run (GATED -- fail-closed; not executed at v0.1)
# ---------------------------------------------------------------------------

def _run(data_root: str, p4_stats2: str, manifest_dir: str,
         reference: str, probe_record: str, commit_pin: str,
         out_dir: str) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    v04t.verify_commit_pin(commit_pin, repo_root)
    batches, batch_list_sha, _p4 = v04p.prepare_probe_batches(
        data_root, p4_stats2)
    model = build_v04_model()
    gen = torch.Generator().manual_seed(ffr.MODEL_INIT_SEED)
    tuple_batches = [(b["u_scaled"], b["cond_in"], b["mask"])
                     for b in batches]
    gates = {
        "a": gate_a_pipeline_bitexact(reference, manifest_dir),
        "b": gate_b_lambda0_parity(model, batches[0]),
        "c": gate_c_probe_record(probe_record,
                                 expected_batch_list_sha256=batch_list_sha),
        "d": gate_d_invertibility(model, gen, batches=tuple_batches),
        "e": gate_e_logdet(model, gen, batches=tuple_batches),
        "f": gate_f_conditioning(model, gen,
                                 authoritative_batches=tuple_batches),
    }
    run_gates(gates, out_dir, form="authoritative")
    print(f"PASS {__abbr__} v{__version__}: all six gates PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="v04_gate")
    ap.add_argument("--mode", choices=("run",), required=True,
                    help="authoritative gate only; refusal fixtures live "
                         "in scripts.v04_fixtures")
    ap.add_argument("--data-root")
    ap.add_argument("--p4-stats2")
    ap.add_argument("--manifest-dir")
    ap.add_argument("--reference")
    ap.add_argument("--probe-record")
    ap.add_argument("--commit-pin")
    ap.add_argument("--out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        missing = [k for k, v in (
            ("data-root", args.data_root),
            ("p4-stats2", args.p4_stats2),
            ("manifest-dir", args.manifest_dir),
            ("reference", args.reference),
            ("probe-record", args.probe_record),
            ("commit-pin", args.commit_pin),
            ("out", args.out)) if not v]
        if missing:
            _fail("GATE_ARGS_MISSING", f"--mode run requires {missing}")
        return _run(args.data_root, args.p4_stats2, args.manifest_dir,
                    args.reference, args.probe_record, args.commit_pin,
                    args.out)
    except V04G1Error as exc:
        logger.error("[%s] ERROR: %s", __abbr__, exc)
        return 2
    except Exception:
        logger.exception("[%s] unexpected exception", __abbr__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# Changelog
# v0.1 (2026-10-12): initial implementation under V04SPEC v1.0 (176,
#   FROZEN) SS17 / V04PLAN v1.0 (178, FROZEN) SS7 (advisor GO: core +
#   synthetic failure-injection tests). Six gates (a)-(f) with the frozen
#   criteria: (a) f01 pipeline bit-exactness against a hash-verified
#   pinned reference table (missing/drift/unexpected -> FAIL
#   MANIFEST_CONTENT_MISMATCH); (b) lam=0 exact parity (value + gradient
#   digest); (c) probe-record hard gates via the V04T contract loader +
#   fresh-init parameter-digest agreement; (d) f12 round-trip <= 1e-4;
#   (e) f13 + f16; (f) f14 + f15 amended (article 180), form field
#   distinguishing authoritative vs preliminary-synthetic. Measurements
#   single-sourced from scripts.v04_fixtures via lazy import. Record
#   seqref-v04-gate-record/1 + SHA-256 sidecar ALWAYS written; any FAIL
#   -> GATE_FAILURE after the write (hard abort, no registered claim);
#   record consumable by the V04T fail-closed prerequisite loader.
#   Authoritative --mode run implemented (fail-closed: commit pin,
#   manifest dir, reference table, probe record all required) but NOT
#   executed -- frozen V02/V03 manifest artifacts and the probe record do
#   not exist in this environment yet.
# ---------------------------------------------------------------------------
