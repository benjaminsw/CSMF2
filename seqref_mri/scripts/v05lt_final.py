# SEQREF-MRI-V05LTFW v0.2 -- scripts.v05lt_final
# LIFETIME: KEEP
# =============================================================================
# Purpose: final-weights retention for SEQREF-MRI-V05LT (195 PA5-B, option
#   (a); I1.9 c2, plan SEQREF-MRI-V05LTI19 v0.1; V05LTEXEC item 26). The
#   trainer retains no final-epoch weights (195 §5); the final model state
#   exists only inside the last committed resume file. This module extracts
#   it after the run ends and publishes it write-once. The trainer and its
#   retention rules are unchanged.
#   * Source (all from the run ledger, read-only): run_start first; exactly
#     one run_end with status `stopped_plateau` or `budget exhausted;
#     plateau not established` (fixture runs also `fixture_budget_end`);
#     the last commit precedes run_end and carries the same status and
#     epoch; no commit after run_end.
#   * Extraction: the committed resume file is hash-verified (sha256 = the
#     commit; sidecar); its resume-state hash = the commit's; next epoch =
#     epoch + 1 and step = the commit; the model-state hash of its weights
#     = the commit's. Weights are written in the trainer's weights schema
#     (seqref-v05lt-weights/1) with meta {seed, arm, lambda, epoch = last
#     epoch, step, model_state_sha256, roles ["final"]}, so the accepted
#     checkpoint loader (v05lt_eval.load_trained_checkpoint) reads them.
#   * Publication: <run>/.final.staging/ holds weights_final.pt,
#     final_weights.json (provenance record bound to the commit and run_end
#     ledger lines) and their sha256 sidecars, all fsynced; it is renamed to
#     <run>/final/ with renameat2(RENAME_NOREPLACE). Interrupted before the
#     rename -> no final directory; a later call moves the stale staging
#     directory to <run>/_incomplete/ (never deleted) and extracts again.
#     An existing final directory is only verified: a mismatched or
#     incomplete one is refused (FINAL_WEIGHTS_MISMATCH / FINAL_INCOMPLETE),
#     never overwritten.
#   * verify_final: exact file set, sidecars, record = the values recomputed
#     from the ledger, weights file hash = record, weights schema and meta,
#     model-state hash = the commit's.
# Interpretation (declared for review): PA5-B.2 "recorded in the run
#   manifest" = the final_weights.json record in <run>/final/ plus the
#   launcher's campaign-ledger `advance` record; the trainer's run ledger is
#   not appended after run_end.
# `hook` exists only for fault-injection fixtures; production passes None.
# CONVENTION: logger.error + V05LTError(code); no fallback, no placeholder.
# Changelog (v0.2, 2026-10-25; c2 revision-1 review fix):
#   * Every operational failure is typed, so a launcher halt is always
#     recorded: sidecars read with C.read_sidecar (regular file, strict
#     ASCII, '<sha256>  <name>') -> FINAL_SIDECAR_INVALID (published files)
#     or FINAL_RESUME_SIDECAR (resume file); ledger, directory and record
#     reads -> FINAL_READ_FAILED; quarantine, staging and write failures ->
#     FINAL_WRITE_FAILED (staging left for the next attempt). Process kills
#     (BaseException) still pass through untouched.
#   * Fixture hook point final:staging_open added (before the first write).
# Changelog (NEW in v0.1, 2026-10-25; I1.9 c2, PA5-B):
#   * Introduced: extract_final / verify_final; recoverable publication;
#     stale staging quarantined; mismatched output refused.
# =============================================================================
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from seqref_mri.scripts import v05lt_common as C

logger = logging.getLogger("seqref_mri.v05lt_final")

__version__ = "0.2"
__abbr__ = "SEQREF-MRI-V05LTFW"

RECORD_SCHEMA = "seqref-v05lt-final-weights/1"
FINAL_DIR = "final"
STAGING = ".final.staging"
WEIGHTS = "weights_final.pt"
RECORD = "final_weights.json"
FILES = frozenset({WEIGHTS, WEIGHTS + ".sha256", RECORD, RECORD + ".sha256"})
STOPPED = "stopped_plateau"
EXHAUSTED = "budget exhausted; plateau not established"
FIXTURE_END = "fixture_budget_end"
_RECORD_KEYS = {"schema", "abbr", "version", "run", "status", "epoch", "step",
                "model_state_sha256", "weights_file", "weights_sha256",
                "source"}


def _hook(hook, stage):
    if hook is not None:
        hook(stage)


def _io(code: str, what: str, fn, *a, **k):
    """Run an I/O step; typed V05LT errors pass through, any other
    exception (OSError, decode, serialisation) -> V05LTError(code)."""
    try:
        return fn(*a, **k)
    except C.V05LTError:
        raise
    except Exception as exc:  # noqa: BLE001 -- converted to a typed failure
        C.fail(code, f"{what}: {type(exc).__name__}: {exc}")


def _line_sha(rec: dict) -> str:
    return C.sha256_bytes(C.canonical_json(rec))


def source(run_dir: Path, *, fixture: bool = False) -> dict:
    """Completed run -> its last commit, run_end and identities (read-only).
    Refuses anything but a single completed run_end bound to the last
    commit."""
    from seqref_mri.scripts import v05lt_train as T
    d = Path(run_dir)
    led = C.Ledger(d / "ledger.jsonl")
    C.require(led.path.is_file(), "FINAL_SOURCE_INVALID", f"{d}: no ledger")
    recs = _io("FINAL_READ_FAILED", f"{d}: run ledger", led.records)
    C.require(bool(recs) and recs[0]["kind"] == "run_start",
              "FINAL_SOURCE_INVALID", f"{d}: no run_start")
    ends = [r for r in recs if r["kind"] == "run_end"]
    C.require(len(ends) == 1, "FINAL_RUN_NOT_COMPLETED",
              f"{d}: {len(ends)} run_end records")
    end = ends[0]
    allowed = (STOPPED, EXHAUSTED) + ((FIXTURE_END,) if fixture else ())
    C.require(end["body"].get("status") in allowed, "FINAL_RUN_NOT_COMPLETED",
              f"{d}: run ended {end['body'].get('status')!r}")
    commits = [r for r in recs if r["kind"] == C.COMMIT_KIND]
    C.require(bool(commits), "FINAL_SOURCE_INVALID", f"{d}: no commit")
    last = commits[-1]
    b = last["body"]
    C.require(last["seq"] < end["seq"] and b.get("status") ==
              end["body"]["status"] and b.get("epoch") ==
              end["body"].get("epoch"), "FINAL_SOURCE_INVALID",
              f"{d}: last commit (seq {last['seq']}, epoch {b.get('epoch')}, "
              f"status {b.get('status')!r}) not bound to run_end "
              f"(seq {end['seq']}, epoch {end['body'].get('epoch')})")
    res = b.get("retained", {}).get("resume")
    C.require(isinstance(res, dict) and {"name", "sha256"} <= set(res),
              "FINAL_SOURCE_INVALID", f"{d}: last commit has no resume file")
    cfg = recs[0]["body"]["config"]
    seed, arm = cfg.get("seed"), cfg.get("arm")
    C.require(isinstance(seed, int) and arm in T.ARMS, "FINAL_SOURCE_INVALID",
              f"{d}: run_start seed/arm {seed!r}/{arm!r}")
    meta = {"seed": seed, "arm": arm, "lambda": T.ARMS[arm],
            "epoch": b["epoch"], "step": b["step"],
            "model_state_sha256": b["model_state_sha256"], "roles": ["final"]}
    record = {"schema": RECORD_SCHEMA,
              "run": {"seed": seed, "arm": arm, "config_sha256":
                      C.sha256_bytes(C.canonical_json(cfg))},
              "status": end["body"]["status"], "epoch": b["epoch"],
              "step": b["step"], "model_state_sha256": b["model_state_sha256"],
              "weights_file": WEIGHTS,
              "source": {"resume_file": res["name"],
                         "resume_sha256": res["sha256"],
                         "resume_state_sha256": b.get("resume_state_sha256"),
                         "commit_seq": last["seq"],
                         "commit_line_sha256": _line_sha(last),
                         "run_end_seq": end["seq"],
                         "run_end_line_sha256": _line_sha(end)}}
    return {"dir": d, "commit": b, "resume": res, "meta": meta,
            "record": record}


def _write_fsync(path: Path, writer) -> str:
    with open(path, "xb") as fh:
        writer(fh)
        fh.flush()
        os.fsync(fh.fileno())
    return C.sha256_file(path)


def _sidecar(path: Path, sha: str) -> None:
    C.write_durable_new(Path(str(path) + ".sha256"),
                        f"{sha}  {path.name}\n".encode())


def extract_final(run_dir: Path, *, fixture: bool = False,
                  hook=None) -> dict:
    """Publish <run>/final once (PA5-B); idempotent when already published
    and valid. Returns verify_final()."""
    import torch
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_train as T
    src = source(run_dir, fixture=fixture)
    d = src["dir"]
    dest, stg = d / FINAL_DIR, d / STAGING
    if os.path.lexists(dest):
        return verify_final(d, fixture=fixture)
    if os.path.lexists(stg):
        q = d / "_incomplete" / f"final_staging.{C.utc_now()}"

        def quarantine():
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stg), str(q))
            C.fsync_dir(d)
        _io("FINAL_WRITE_FAILED", f"{stg}: quarantine", quarantine)
        logger.warning("[%s] FINAL_STAGING_QUARANTINED: interrupted "
                       "publication moved to %s; extracting again", __abbr__, q)
    b, res = src["commit"], src["resume"]
    rpath = d / "ckpt" / res["name"]
    got = C.sha256_file(rpath)
    C.require(got == res["sha256"], "FINAL_RESUME_HASH",
              f"{rpath.name}: {got} != committed {res['sha256']}")
    side = Path(str(rpath) + ".sha256")
    C.require(C.read_sidecar(side, "FINAL_RESUME_SIDECAR") == got,
              "FINAL_RESUME_SIDECAR", f"{side.name}: sidecar != file")
    obj = T.load_resume(rpath)
    st = obj["state"]
    C.require(T.state_sha256(st) == b.get("resume_state_sha256"),
              "FINAL_RESUME_STATE", f"{rpath.name}: resume-state hash != commit")
    C.require(st.get("next_epoch") == b["epoch"] + 1 and
              st.get("step") == b["step"], "FINAL_RESUME_STATE",
              f"{rpath.name}: epoch/step != commit")
    model = obj.pop("model")
    del obj
    ms = C.model_state_sha256(model)
    C.require(ms == b["model_state_sha256"], "FINAL_MODEL_STATE",
              f"{rpath.name}: model state {ms} != commit "
              f"{b['model_state_sha256']}")
    wpath, rpath2 = stg / WEIGHTS, stg / RECORD

    def stage() -> str:
        stg.mkdir()
        _hook(hook, "final:staging_open")
        wsha = _write_fsync(wpath, lambda fh: torch.save(
            {"schema": T.WEIGHTS_SCHEMA, "model": model, "meta": src["meta"]},
            fh))
        _hook(hook, "final:weights_written")
        _sidecar(wpath, wsha)
        rec = dict(src["record"], abbr=__abbr__, version=__version__,
                   weights_sha256=wsha)
        rsha = C.write_durable_new(rpath2, C.canonical_json(rec))
        _sidecar(rpath2, rsha)
        C.fsync_dir(stg)
        return wsha
    wsha = _io("FINAL_WRITE_FAILED", f"{stg}: staging", stage)
    del model
    _hook(hook, "final:staged")
    EV.publish_noreplace(stg, dest)
    _hook(hook, "final:published")
    logger.info("[%s] final weights published: %s (sha256 %s, epoch %d)",
                __abbr__, dest / WEIGHTS, wsha, b["epoch"])
    return verify_final(d, fixture=fixture)


def verify_final(run_dir: Path, *, fixture: bool = False) -> dict:
    """Published <run>/final -> verified identities; anything else refused."""
    import torch
    from seqref_mri.scripts import v05lt_train as T
    src = source(run_dir, fixture=fixture)
    dest = src["dir"] / FINAL_DIR
    C.require(dest.is_dir() and not dest.is_symlink(), "FINAL_INCOMPLETE",
              f"{dest}: not a directory")
    names = _io("FINAL_READ_FAILED", f"{dest}: listing",
                lambda: {p.name for p in dest.iterdir()})
    C.require(names == FILES, "FINAL_INCOMPLETE",
              f"{dest}: files {sorted(names)} != {sorted(FILES)}")
    sums = {}
    for n in (WEIGHTS, RECORD):
        p = dest / n
        C.require(p.is_file() and not p.is_symlink(), "FINAL_INCOMPLETE",
                  str(p))
        sums[n] = C.sha256_file(p)
        side = C.read_sidecar(dest / (n + ".sha256"), "FINAL_SIDECAR_INVALID")
        C.require(side == sums[n], "FINAL_WEIGHTS_MISMATCH",
                  f"{n}: sidecar != file")
    raw = _io("FINAL_READ_FAILED", f"{RECORD}: read",
              (dest / RECORD).read_bytes)
    try:
        rec = json.loads(raw)
    except ValueError as exc:            # includes undecodable bytes
        C.fail("FINAL_WEIGHTS_MISMATCH", f"{RECORD}: {exc}")
    C.require(isinstance(rec, dict) and set(rec) == _RECORD_KEYS and
              isinstance(rec.get("abbr"), str) and
              isinstance(rec.get("version"), str), "FINAL_WEIGHTS_MISMATCH",
              f"{RECORD}: fields")
    want = dict(src["record"], weights_sha256=sums[WEIGHTS])
    diff = sorted(k for k in want if rec.get(k) != want[k])
    C.require(not diff, "FINAL_WEIGHTS_MISMATCH",
              f"{RECORD}: fields {diff} differ from the run ledger / file")
    try:
        obj = torch.load(dest / WEIGHTS, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 -- converted to a typed failure
        C.fail("FINAL_WEIGHTS_MISMATCH", f"{WEIGHTS}: {type(exc).__name__}: "
               f"{exc}")
    C.require(isinstance(obj, dict) and set(obj) == {"schema", "model", "meta"}
              and obj["schema"] == T.WEIGHTS_SCHEMA and
              obj["meta"] == src["meta"], "FINAL_WEIGHTS_MISMATCH",
              f"{WEIGHTS}: schema / meta")
    ms = C.model_state_sha256(obj["model"])
    C.require(ms == src["commit"]["model_state_sha256"],
              "FINAL_WEIGHTS_MISMATCH", f"{WEIGHTS}: model state {ms}")
    return {"path": dest / WEIGHTS, "sha256": sums[WEIGHTS],
            "record_sha256": sums[RECORD], "epoch": src["meta"]["epoch"],
            "step": src["meta"]["step"], "model_state_sha256": ms,
            "seed": src["meta"]["seed"], "arm": src["meta"]["arm"]}
