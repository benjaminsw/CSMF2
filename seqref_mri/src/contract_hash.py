# SEQREF-CHASH v0.2 -- normalisation-contract hashing + binding assertions
# LIFETIME: KEEP
#
# Why this exists
#   The normalisation declaration must guard the NORMALISATION CONTRACT, not
#   every line of an actively developed training script.  A whole-file blocking
#   SHA on train_base.py would raise after a comment or logging edit, which
#   trains people to refresh the declaration without checking behaviour.  This
#   module hashes only a DECLARED list of named code entities.  Whole-file SHAs
#   remain recorded as provenance, but are NOT blocking.
#
# LOCKED PROCEDURE (v1) -- any change here is a spec amendment, because two
# implementations must produce identical hashes from identical code:
#   1. Source is read as bytes and decoded UTF-8 (strict).  Parse with
#      ast.parse().  A parse failure is an ERROR, never a mismatch.
#   2. Entities are declared as ordered (kind, name) pairs.  Hashing follows
#      the DECLARED order, not source order, so the declaration fixes it.
#      kind = "assign"   -> module-level assignment to a Name target
#      kind = "function" -> module-level def
#      kind = "method"   -> "ClassName.method_name", def inside a module-level
#                           class
#   3. Exactly one node must match each entity.  Zero or several -> ERROR.
#   4. Segment extraction is LINE-BASED and INCLUDES DECORATORS: start line is
#      min(node.lineno, decorator linenos); end line is node.end_lineno.
#      ast.get_source_segment() is deliberately NOT used, because it omits
#      decorators -- @torch.no_grad() on _validate would then be invisible to
#      the hash.
#   5. Segment normalisation, in this order: CRLF/CR -> LF; trailing whitespace
#      stripped per line; trailing blank lines removed; common leading indent
#      removed (methods hash identically wherever they sit).  Comments and
#      docstrings are KEPT: inside a contract entity they are part of the
#      declared contract, and the scope is small enough that this is not the
#      noisy-failure case the split was designed to avoid.
#   6. Framing is LENGTH-PREFIXED so no segment content can imitate a
#      delimiter:  f"{relpath}|{kind}|{name}|{len(segment_bytes)}\n" + segment
#      + "\n"  per entity, concatenated in declared order.
#   7. contract_hash = SHA-256 over the UTF-8 bytes of that concatenation.
#      Per-entity hashes are also returned so a mismatch names the entity that
#      changed rather than only the file.
#
# CONVENTION: every failure path -> logger.error + raise. No fallback, no mock,
#   no silent pass.
#
# LOCKED ASSERTION PROCEDURE "seqref-prepare-binding/1"
#   Entity hashing cannot protect a claim about a CALL SITE in a function that
#   is deliberately not hashed (run_training is large; hashing it would
#   recreate the whole-file noise problem).  These assertions close that gap.
#   They FAIL CLOSED: any ambiguity raises rather than passing.
#
#   For a declared (function_name, callee_name) pair, within the module-level
#   function `function_name`:
#     A1 exactly one module-level def named function_name  (else ERROR)
#     A2 exactly one Call to Name(callee_name) anywhere in its subtree,
#        INCLUDING nested defs/lambdas (zero = missing, >1 = ambiguous;
#        both ERROR)
#     A3 that Call is the direct value of an Assign with exactly ONE target,
#        and that target is a bare Name B.  Tuple unpack, Subscript/Attribute
#        targets, AnnAssign, AugAssign, walrus, or a bare expression statement
#        are all ERROR -- the binding must be unambiguous.
#     A0 SCOPE. Checks are applied only where B refers to OUR binding. A
#        nested scope (comprehension, lambda, def, class) that binds the same
#        name locally SHADOWS it, and its subtree is skipped entirely -- e.g.
#        `sum(p.numel() for p in model.parameters())` does not rebind an outer
#        `p`. A nested scope that does NOT bind the name is descended into,
#        because there B is free and refers to ours. Any `nonlocal B` in a
#        nested scope is ERROR: it could write through to our binding.
#     A4 B is bound EXACTLY ONCE syntactically in the function.  Any further
#        Assign/AnnAssign/AugAssign/for-target/with-as/except-as/comprehension
#        target/import-as binding B, or `del B`, `global B`, `nonlocal B`,
#        is ERROR.  (Re-execution in a loop is one syntactic binding.)
#     A5 no Assign/AnnAssign/AugAssign target is a Subscript or Attribute
#        rooted at B  -- blocks p["x_norm"] = ... and p["x_norm"] *= ...
#     A6 no Assign value is a BARE Name B  -- blocks the alias q = p, which
#        would otherwise route mutation around A5
#     A7 B is never passed as a BARE argument (positional, keyword, *args or
#        **kwargs) to any call -- a callee could mutate it in place.
#        Reading THROUGH B is allowed: p["x_norm"] and p["x_norm"].flatten(1)
#        are Subscript/Attribute expressions, not B itself, and passing those
#        is the normal consumption pattern.
#
#   What these assertions do NOT establish, and must not be reported as
#   establishing: that the loss consumes ONLY prepared tensors.  A path could
#   build additional tensors directly from `batch` alongside B.  The
#   declaration states that residual scope explicitly.
#
# Changelog
#   v0.2 (2026-07-29) Added the locked prepare-binding assertion procedure
#     with SCOPE-AWARE traversal (A0): nested comprehension/lambda/def scopes
#     that shadow the binding name are skipped, so an unrelated throwaway
#     variable of the same name is not misreported as a rebinding.
#     (A1-A7) so the training call site is covered without hashing the whole of
#     run_training; call presence alone does not prove the returned tensors are
#     unmutated. Assertions fail closed on aliasing, rebinding and ambiguity.
#   v0.1 (2026-07-29) Created under Amendment A2. Locked entity-level hashing
#     to replace whole-file blocking SHAs; decorator-inclusive line extraction;
#     length-prefixed framing; declared-order canonicalisation.
#
# Update summary (v0.2): the hash proves that the declared normalisation code
#   has not changed, but it could say nothing about a function excluded from
#   the declared scope, so an assertion layer was added for exactly the claim
#   the hash cannot carry -- that both the training and validation paths obtain
#   their tensors from the same preparation function and do not visibly mutate
#   the result. The assertions reject aliasing and rebinding rather than
#   attempting to reason about them, and the declaration now records what they
#   leave unverified instead of asserting flat equivalence.

from __future__ import annotations

import ast
import hashlib
import logging
from typing import Iterable, Sequence

logger = logging.getLogger("seqref_mri.contract_hash")

__version__ = "0.2"
__abbr__ = "SEQREF-CHASH"

PROCEDURE_ID = "seqref-contract-hash/1"
ASSERT_PROCEDURE_ID = "seqref-prepare-binding/1"

VALID_KINDS = ("assign", "function", "method")


def _normalise_segment(lines: Sequence[str]) -> str:
    out = [ln.replace("\r\n", "\n").replace("\r", "\n").rstrip() for ln in lines]
    while out and not out[-1]:
        out.pop()
    if not out:
        return ""
    indents = [len(ln) - len(ln.lstrip()) for ln in out if ln]
    strip_n = min(indents) if indents else 0
    return "\n".join(ln[strip_n:] if ln else "" for ln in out)


def _node_lines(node: ast.AST) -> tuple[int, int]:
    start = node.lineno
    for dec in getattr(node, "decorator_list", []) or []:
        start = min(start, dec.lineno)
    return start, node.end_lineno


def _find(tree: ast.Module, kind: str, name: str) -> list[ast.AST]:
    hits: list[ast.AST] = []
    if kind == "assign":
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (node.targets if isinstance(node, ast.Assign)
                           else [node.target])
                for t in targets:
                    if isinstance(t, ast.Name) and t.id == name:
                        hits.append(node)
    elif kind == "function":
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == name:
                hits.append(node)
    elif kind == "method":
        if "." not in name:
            logger.error("method entity %r must be 'ClassName.method'", name)
            raise ValueError(f"malformed method entity: {name}")
        cls_name, meth_name = name.split(".", 1)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                            and sub.name == meth_name:
                        hits.append(sub)
    else:
        logger.error("unknown entity kind %r (valid: %s)", kind, VALID_KINDS)
        raise ValueError(f"unknown entity kind: {kind}")
    return hits


def hash_entities(source_bytes: bytes, relpath: str,
                  entities: Iterable[tuple[str, str]]) -> dict:
    """Hash a declared, ordered list of (kind, name) entities from one file.

    Returns {"relpath", "entities": [...], "frame": <str>} where each entity
    record carries kind, name, start/end lines and its own sha256.
    """
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        logger.error("%s is not valid UTF-8: %s", relpath, exc)
        raise
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.error("%s failed to parse: %s", relpath, exc)
        raise
    lines = source.split("\n")

    records: list[dict] = []
    frame_parts: list[str] = []
    for kind, name in entities:
        if kind not in VALID_KINDS:
            logger.error("%s: unknown entity kind %r for %r", relpath, kind,
                         name)
            raise ValueError(f"unknown entity kind: {kind}")
        hits = _find(tree, kind, name)
        if len(hits) == 0:
            logger.error("%s: declared contract entity %s %r NOT FOUND -- the "
                         "declaration names code that no longer exists",
                         relpath, kind, name)
            raise LookupError(f"{relpath}: missing entity {kind} {name}")
        if len(hits) > 1:
            logger.error("%s: declared contract entity %s %r is AMBIGUOUS "
                         "(%d matches at lines %s)", relpath, kind, name,
                         len(hits), [_node_lines(h)[0] for h in hits])
            raise LookupError(f"{relpath}: ambiguous entity {kind} {name}")
        start, end = _node_lines(hits[0])
        segment = _normalise_segment(lines[start - 1:end])
        seg_bytes = segment.encode("utf-8")
        frame_parts.append(
            f"{relpath}|{kind}|{name}|{len(seg_bytes)}\n{segment}\n")
        records.append({
            "kind": kind, "name": name,
            "start_line": start, "end_line": end,
            "segment_bytes": len(seg_bytes),
            "sha256": hashlib.sha256(seg_bytes).hexdigest(),
        })
    return {"relpath": relpath, "entities": records,
            "frame": "".join(frame_parts)}


def contract_hash(files: Sequence[dict]) -> dict:
    """files: ordered [{"relpath","source_bytes","entities":[(kind,name),...]}]

    Returns {"procedure", "contract_hash", "files": [per-file records]}.
    File order is the declared order and is part of the hash.
    """
    frames: list[str] = []
    per_file: list[dict] = []
    for spec in files:
        res = hash_entities(spec["source_bytes"], spec["relpath"],
                            spec["entities"])
        frames.append(res["frame"])
        per_file.append({"relpath": res["relpath"],
                         "entities": res["entities"]})
    digest = hashlib.sha256("".join(frames).encode("utf-8")).hexdigest()
    return {"procedure": PROCEDURE_ID, "contract_hash": digest,
            "files": per_file}


# ---------------------------------------------------------------------------
# LOCKED ASSERTION PROCEDURE "seqref-prepare-binding/1"  (see header A1-A7)
# ---------------------------------------------------------------------------

def _root_name(node: ast.AST) -> str | None:
    """Ultimate base Name of a Subscript/Attribute chain, else None."""
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _bound_names(node: ast.AST) -> list[str]:
    """Names syntactically BOUND by this statement/target node."""
    out: list[str] = []
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store,
                                                            ast.Del)):
        out.append(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            out.extend(_bound_names(elt))
    elif isinstance(node, ast.Starred):
        out.extend(_bound_names(node.value))
    return out


_SCOPE_NODES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef,
                ast.ClassDef)


def _scope_binds(node: ast.AST, name: str) -> bool:
    """Does this nested scope bind `name` locally (thus shadowing an outer)?"""
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                         ast.GeneratorExp)):
        return any(name in _bound_names(g.target) for g in node.generators)
    if isinstance(node, ast.Lambda):
        a = node.args
        names = [x.arg for x in (a.posonlyargs + a.args + a.kwonlyargs)]
        for extra in (a.vararg, a.kwarg):
            if extra is not None:
                names.append(extra.arg)
        return name in names
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = node.args
        names = [x.arg for x in (a.posonlyargs + a.args + a.kwonlyargs)]
        for extra in (a.vararg, a.kwarg):
            if extra is not None:
                names.append(extra.arg)
        if name in names:
            return True
        for sub in _walk_scope(node, name, _descend_only=True):
            if isinstance(sub, (ast.Global, ast.Nonlocal)) and \
                    name in sub.names:
                return False          # explicitly reaches outward, not a shadow
            if isinstance(sub, ast.Assign):
                if any(name in _bound_names(t) for t in sub.targets):
                    return True
            if isinstance(sub, (ast.AnnAssign, ast.AugAssign, ast.For,
                                ast.AsyncFor)) and \
                    name in _bound_names(sub.target):
                return True
        return False
    if isinstance(node, ast.ClassDef):
        for sub in node.body:
            if isinstance(sub, ast.Assign) and \
                    any(name in _bound_names(t) for t in sub.targets):
                return True
        return False
    return False


def _walk_scope(root: ast.AST, name: str, _descend_only: bool = False):
    """Yield nodes under `root` WITHOUT crossing a scope that shadows `name`."""
    stack = list(ast.iter_child_nodes(root))
    while stack:
        node = stack.pop()
        if isinstance(node, _SCOPE_NODES) and not _descend_only:
            if _scope_binds(node, name):
                continue              # shadowed: skip this subtree entirely
        yield node
        stack.extend(ast.iter_child_nodes(node))


def check_prepare_binding(source_bytes: bytes, relpath: str,
                          function_name: str, callee_name: str) -> dict:
    """A1-A7. Returns a record on success; raises on ANY failure (fail closed)."""
    source = source_bytes.decode("utf-8")
    tree = ast.parse(source)

    # A1
    fns = [n for n in tree.body
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
           and n.name == function_name]
    if len(fns) != 1:
        logger.error("%s: expected exactly one module-level def %r, found %d",
                     relpath, function_name, len(fns))
        raise LookupError(f"{relpath}: def {function_name} x{len(fns)}")
    fn = fns[0]

    # A2
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == callee_name]
    if len(calls) != 1:
        logger.error("%s:%s expected exactly ONE call to %s(), found %d at "
                     "lines %s -- zero is a bypass, several are ambiguous",
                     relpath, function_name, callee_name, len(calls),
                     [c.lineno for c in calls])
        raise LookupError(
            f"{relpath}:{function_name} {callee_name}() x{len(calls)}")
    call = calls[0]

    # A3
    binding = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and node.value is call:
            if len(node.targets) != 1 or not isinstance(node.targets[0],
                                                        ast.Name):
                logger.error("%s:%s the %s() result must be assigned to ONE "
                             "bare Name (line %d)", relpath, function_name,
                             callee_name, node.lineno)
                raise ValueError(f"{relpath}:{function_name} bad binding form")
            binding = node.targets[0].id
            binding_line = node.lineno
            break
    if binding is None:
        logger.error("%s:%s the %s() call at line %d is not the direct value "
                     "of a simple assignment (AnnAssign/AugAssign/walrus/bare "
                     "expression are all rejected)", relpath, function_name,
                     callee_name, call.lineno)
        raise ValueError(f"{relpath}:{function_name} unbound {callee_name}()")

    binds, mutations, aliases, bare_args = [], [], [], []
    for node in _walk_scope(fn, binding):
        if isinstance(node, ast.Nonlocal) and binding in node.names:
            logger.error("%s:%s nested `nonlocal %s` can write through to the "
                         "outer binding (line %d)", relpath, function_name,
                         binding, node.lineno)
            raise ValueError(f"{relpath}:{function_name} nonlocal {binding}")
        # A4 -- binding sites
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if binding in _bound_names(t):
                    binds.append(node.lineno)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if binding in _bound_names(node.target):
                binds.append(node.lineno)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            if binding in _bound_names(node.target):
                binds.append(node.lineno)
        elif isinstance(node, ast.comprehension):
            if binding in _bound_names(node.target):
                binds.append(getattr(node.target, "lineno", fn.lineno))
        elif isinstance(node, ast.withitem):
            if node.optional_vars is not None and \
                    binding in _bound_names(node.optional_vars):
                binds.append(getattr(node.optional_vars, "lineno", fn.lineno))
        elif isinstance(node, ast.ExceptHandler):
            if node.name == binding:
                binds.append(node.lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if (a.asname or a.name.split(".")[0]) == binding:
                    binds.append(node.lineno)
        elif isinstance(node, ast.NamedExpr):
            if isinstance(node.target, ast.Name) and node.target.id == binding:
                binds.append(node.lineno)
        elif isinstance(node, ast.Delete):
            for t in node.targets:
                if binding in _bound_names(t):
                    logger.error("%s:%s `del %s` at line %d", relpath,
                                 function_name, binding, node.lineno)
                    raise ValueError(f"{relpath}:{function_name} del {binding}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            if binding in node.names:
                logger.error("%s:%s global/nonlocal %s at line %d", relpath,
                             function_name, binding, node.lineno)
                raise ValueError(
                    f"{relpath}:{function_name} global/nonlocal {binding}")

        # A5 -- in-place mutation through the binding
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            for t in targets:
                if isinstance(t, (ast.Subscript, ast.Attribute)) and \
                        _root_name(t) == binding:
                    mutations.append(node.lineno)

        # A6 -- bare alias
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) \
                and node.value.id == binding:
            aliases.append(node.lineno)

        # A7 -- binding passed bare into a call
        if isinstance(node, ast.Call):
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id == binding:
                    bare_args.append(node.lineno)
                if isinstance(arg, ast.Starred) and \
                        isinstance(arg.value, ast.Name) and \
                        arg.value.id == binding:
                    bare_args.append(node.lineno)
            for kw in node.keywords:
                if isinstance(kw.value, ast.Name) and kw.value.id == binding:
                    bare_args.append(node.lineno)

    if len(binds) != 1 or binds[0] != binding_line:
        logger.error("%s:%s %r must be bound exactly once (found lines %s)",
                     relpath, function_name, binding, sorted(set(binds)))
        raise ValueError(f"{relpath}:{function_name} rebinding of {binding}")
    if mutations:
        logger.error("%s:%s in-place mutation through %r at lines %s",
                     relpath, function_name, binding, sorted(set(mutations)))
        raise ValueError(f"{relpath}:{function_name} mutation of {binding}")
    if aliases:
        logger.error("%s:%s bare alias of %r at lines %s -- rejected because "
                     "an alias routes mutation around the target check",
                     relpath, function_name, binding, sorted(set(aliases)))
        raise ValueError(f"{relpath}:{function_name} alias of {binding}")
    if bare_args:
        logger.error("%s:%s %r passed as a bare argument at lines %s -- a "
                     "callee could mutate it in place; read through it "
                     "instead", relpath, function_name, binding,
                     sorted(set(bare_args)))
        raise ValueError(f"{relpath}:{function_name} {binding} passed bare")

    return {"procedure": ASSERT_PROCEDURE_ID, "relpath": relpath,
            "function": function_name, "callee": callee_name,
            "binding": binding, "binding_line": binding_line,
            "call_line": call.lineno, "result": True}
