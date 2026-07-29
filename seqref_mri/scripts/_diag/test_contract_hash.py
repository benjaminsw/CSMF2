# SEQREF-CHASH-TEST v0.1 -- tests for contract_hash + prepare-binding assertions
# LIFETIME: DIAGNOSTIC
#
# Run:  python seqref_mri/scripts/_diag/test_contract_hash.py \
#           --repo-dir .
# Exits 0 if every case behaves as specified, 1 otherwise. Prints a table.
#
# These tests exist because a claim that a procedure was tested is not evidence
# that it was: the test must be runnable by the reviewer.
#
# CONVENTION: every failure path -> logger.error + raise. No fallback, no mock.
#
# Changelog
#   v0.1 (2026-07-29) Created under Amendment A2.

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from contract_hash import (contract_hash, check_prepare_binding,  # noqa: E402
                           PROCEDURE_ID, ASSERT_PROCEDURE_ID)

FASTMRI = "seqref_mri/src/fastmri_data.py"
TRAINBASE = "seqref_mri/scripts/train_base.py"

ENTITIES = {
    FASTMRI: [("assign", "__version__"), ("assign", "__abbr__"),
              ("method", "FastMRISliceDataset.__getitem__")],
    TRAINBASE: [("assign", "__version__"), ("assign", "__abbr__"),
                ("assign", "NORMALIZED_DATA_RANGE"),
                ("function", "_collate"), ("function", "_prepare"),
                ("function", "_validate")],
}

results: list[tuple[str, str, bool]] = []


def record(name: str, expect: str, ok: bool) -> None:
    results.append((name, expect, ok))


def hash_with(src_a: bytes, src_b: bytes) -> str:
    return contract_hash([
        {"relpath": FASTMRI, "source_bytes": src_a,
         "entities": ENTITIES[FASTMRI]},
        {"relpath": TRAINBASE, "source_bytes": src_b,
         "entities": ENTITIES[TRAINBASE]},
    ])["contract_hash"]


def expect_raises(fn, exc_types) -> bool:
    try:
        fn()
    except exc_types:
        return True
    except Exception:
        return False
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    args = ap.parse_args(argv)

    a = open(os.path.join(args.repo_dir, FASTMRI), "rb").read()
    b = open(os.path.join(args.repo_dir, TRAINBASE), "rb").read()
    base = hash_with(a, b)

    # ---- hash: insensitive to edits OUTSIDE the declared entities ----------
    record("unrelated comment at EOF", "no change",
           hash_with(a, b + b"\n# unrelated note\n") == base)
    record("unrelated header edit", "no change",
           hash_with(a, b.replace(b"# Purpose:", b"# PURPOSE (edited):", 1))
           == base)

    lines = b.decode().split("\n")
    # Fixture guards: these tests pin line positions inside _prepare. If the
    # source moves, FAIL LOUDLY AS A FIXTURE PROBLEM rather than silently
    # testing the wrong lines and reporting a contract defect.
    if 'y = batch["y"]' not in lines[142]:
        raise AssertionError(
            "FIXTURE OUT OF DATE: line 143 of train_base.py is no longer the "
            f"expected _prepare line (found: {lines[142]!r}). Update the test "
            "fixture; this is NOT a contract defect.")
    if not lines[140].startswith("def _prepare"):
        raise AssertionError(
            "FIXTURE OUT OF DATE: _prepare no longer starts at line 141 "
            f"(found: {lines[140]!r}). Update the test fixture; this is NOT a "
            "contract defect.")
    ws = list(lines)
    ws[142] = ws[142] + "    "
    record("trailing whitespace in _prepare", "no change",
           hash_with(a, "\n".join(ws).encode()) == base)

    crlf = list(lines)
    for i in range(140, 163):
        crlf[i] = crlf[i] + "\r"
    record("CRLF line endings", "no change",
           hash_with(a, "\n".join(crlf).encode()) == base)

    # ---- hash: sensitive to edits INSIDE the declared entities -------------
    record("edit inside _prepare", "CHANGE",
           hash_with(a, b.replace(b"x_norm = x_true / a",
                                  b"x_norm = x_true / (a + 1e-12)", 1))
           != base)
    record("NORMALIZED_DATA_RANGE changed", "CHANGE",
           hash_with(a, b.replace(b"NORMALIZED_DATA_RANGE = 1.0",
                                  b"NORMALIZED_DATA_RANGE = 2.0", 1)) != base)
    record("@torch.no_grad() removed", "CHANGE",
           hash_with(a, b.replace(b"@torch.no_grad()\ndef _validate",
                                  b"def _validate", 1)) != base)
    record("comment inside _prepare", "CHANGE",
           hash_with(a, b.replace(b"# Returns normalized tensors",
                                  b"# returns normalised tensors", 1)) != base)
    record("determinism (repeat)", "identical", hash_with(a, b) == base)
    record("file order permuted", "CHANGE",
           contract_hash([
               {"relpath": TRAINBASE, "source_bytes": b,
                "entities": ENTITIES[TRAINBASE]},
               {"relpath": FASTMRI, "source_bytes": a,
                "entities": ENTITIES[FASTMRI]},
           ])["contract_hash"] != base)

    # ---- hash: hard errors -------------------------------------------------
    record("declared entity missing", "RAISE", expect_raises(
        lambda: contract_hash([{"relpath": TRAINBASE, "source_bytes": b,
                                "entities": [("function", "_nope")]}]),
        LookupError))
    record("parse failure", "RAISE", expect_raises(
        lambda: contract_hash([{"relpath": TRAINBASE,
                                "source_bytes": b + b"\ndef broken(:\n",
                                "entities": ENTITIES[TRAINBASE]}]),
        SyntaxError))

    # ---- binding assertions: real code must PASS ---------------------------
    for fn in ("_validate", "run_training"):
        try:
            r = check_prepare_binding(b, TRAINBASE, fn, "_prepare")
            record(f"{fn} binding assertions", "PASS", r["result"] is True)
        except Exception:
            record(f"{fn} binding assertions", "PASS", False)

    # ---- binding assertions: adversarial mutations must RAISE --------------
    def mutated(old: bytes, new: bytes, fn: str = "run_training"):
        src = b.replace(old, new, 1)
        if src == b:
            raise AssertionError(
                "FIXTURE OUT OF DATE: the mutation target string was not found "
                f"in train_base.py ({old[:60]!r}...). Update the test fixture; "
                "this is NOT a contract defect.")
        return lambda: check_prepare_binding(src, TRAINBASE, fn, "_prepare")

    record("A0 shadowing genexp not flagged", "PASS (no raise)",
           not expect_raises(
               mutated(b'            p = _prepare(batch, device, test0=test0)',
                       b'            _ = [p for p in range(3)]\n'
                       b'            p = _prepare(batch, device, test0=test0)'),
               Exception))
    record("A2 _prepare bypassed", "RAISE", expect_raises(
        mutated(b'            p = _prepare(batch, device, test0=test0)',
                b'            p = _other_prepare(batch, device)'),
        LookupError))
    record("A2 second _prepare call", "RAISE", expect_raises(
        mutated(b'            p = _prepare(batch, device, test0=test0)',
                b'            p = _prepare(batch, device, test0=test0)\n'
                b'            q = _prepare(batch, device, test0=False)'),
        LookupError))
    record("A3 result not bound to a Name", "RAISE", expect_raises(
        mutated(b'            p = _prepare(batch, device, test0=test0)',
                b'            d["p"] = _prepare(batch, device, test0=test0)'),
        ValueError))
    record("A4 rebinding in same scope", "RAISE", expect_raises(
        mutated(b'            nll = -model.log_prob(p["x_norm"].flatten(1), '
                b'p["cond_in"]).mean()',
                b'            p = dict(p)\n'
                b'            nll = -model.log_prob(p["x_norm"].flatten(1), '
                b'p["cond_in"]).mean()'),
        ValueError))
    record("A5 in-place mutation p[k] = ...", "RAISE", expect_raises(
        mutated(b'            nll = -model.log_prob(p["x_norm"].flatten(1), '
                b'p["cond_in"]).mean()',
                b'            p["x_norm"] = p["x_norm"] * 2.0\n'
                b'            nll = -model.log_prob(p["x_norm"].flatten(1), '
                b'p["cond_in"]).mean()'),
        ValueError))
    record("A5 augmented p[k] *= ...", "RAISE", expect_raises(
        mutated(b'            nll = -model.log_prob(p["x_norm"].flatten(1), '
                b'p["cond_in"]).mean()',
                b'            p["x_norm"] *= 2.0\n'
                b'            nll = -model.log_prob(p["x_norm"].flatten(1), '
                b'p["cond_in"]).mean()'),
        ValueError))
    record("A6 bare alias q = p", "RAISE", expect_raises(
        mutated(b'            nll = -model.log_prob(p["x_norm"].flatten(1), '
                b'p["cond_in"]).mean()',
                b'            q = p\n'
                b'            q["x_norm"] = 0.0\n'
                b'            nll = -model.log_prob(p["x_norm"].flatten(1), '
                b'p["cond_in"]).mean()'),
        ValueError))
    record("A7 binding passed bare to a call", "RAISE", expect_raises(
        mutated(b'            nll = -model.log_prob(p["x_norm"].flatten(1), '
                b'p["cond_in"]).mean()',
                b'            _rescale(p)\n'
                b'            nll = -model.log_prob(p["x_norm"].flatten(1), '
                b'p["cond_in"]).mean()'),
        ValueError))
    record("A7 read-through arg still allowed", "PASS (no raise)",
           not expect_raises(
               mutated(b'            nll = -model.log_prob(p["x_norm"].'
                       b'flatten(1), p["cond_in"]).mean()',
                       b'            _log(p["x_norm"].flatten(1))\n'
                       b'            nll = -model.log_prob(p["x_norm"].'
                       b'flatten(1), p["cond_in"]).mean()'),
               Exception))

    width = max(len(n) for n, _, _ in results)
    print(f"\nprocedures: {PROCEDURE_ID} · {ASSERT_PROCEDURE_ID}")
    print(f"contract_hash: {base}\n")
    failed = 0
    for name, expect, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  expect: {expect}")
        failed += (not ok)
    print(f"\n{len(results) - failed}/{len(results)} cases behaved as specified")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
