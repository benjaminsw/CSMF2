# SEQREF-MRI-V05LTE0 v0.3 -- scripts.v05lt_epoch0
# LIFETIME: KEEP
# =============================================================================
# Purpose: epoch-0 checkpoint per seed (194 AM3-A/B, AM3-G1; 195 §4, §5):
#   fresh untrained weights from the production ffr.build_model(init_seed=
#   seed), saved once per seed under results/v05lt/epoch0/seed_<s>/ through
#   the crash-safe store + ledger of v05lt_common, with:
#     * model-state hash (AM3-G1) -- the arm-identity value;
#     * file sha256 -- integrity only;
#     * environment record (addition A): Python, torch, NumPy, CPU model,
#       thread settings, deterministic settings, relevant env variables;
#     * implementation identity: script/common sha256, repo HEAD, branch.
#   load_epoch0() reloads a committed checkpoint in eval mode and verifies
#   its model-state hash; it consumes no global RNG.
# Directory rules: an existing seed directory blocks a fresh start
#   (EPOCH0_DIR_EXISTS). An incomplete save is handled ONLY by the explicit
#   `--recover` procedure: store.recover() (uncommitted files quarantined,
#   committed files verified); committed+verified -> already complete, no
#   change; committed, not verified -> verification only; nothing committed
#   -> logged rebuild (EPOCH0_RECOVERY_REBUILD) from the same seed. Nothing
#   is overwritten or rebuilt silently.
# The production epoch-0 stage runs only after the full 195 §8.3 gate;
#   fixtures call these functions in temporary directories.
# CONVENTION: logger.error + V05LTError(code); exit 0 PASS / 2 failure.
#   No fallback, no mock, no placeholder, no silent pass.
# Changelog (v0.3, 2026-10-26; I1.9 c2 production wiring, V05LTI19C2P v0.1):
#   * create / recover / load_epoch0 accept a FIXTURE-ONLY `model_factory`
#     (stub architecture for synthetic production-stage tests), as the
#     trainer and load_trained_checkpoint already do. Production passes
#     none: ffr.build_model(init_seed=seed) unchanged; the checkpoint,
#     ledger and verification rules are unchanged.
# Changelog (v0.2, 2026-10-21; I1.6(b), open items 7 and 8):
#   * RNG wording made explicit: load_epoch0 saves/restores the torch CPU
#     RNG only (registered setup is CPU-only; no CUDA RNG exists in use).
#   * Rebuild compatibility: before EPOCH0_RECOVERY_REBUILD, the seed, code
#     identity (script/common sha256, repo HEAD, branch) and environment
#     (Python, torch, NumPy, CPU model, thread settings, deterministic
#     settings, env variables) must equal the seed directory's `start`
#     record; otherwise EPOCH0_RECOVERY_INCOMPATIBLE and nothing is built.
# Changelog (NEW in v0.1, 2026-10-21):
#   * Introduced under I1.5 (SEQREF-MRI-V05LTI15 v0.2, core + A): save,
#     verify, load, environment + implementation identity, explicit
#     recovery of an interrupted save.
# =============================================================================
from __future__ import annotations

import argparse
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

from seqref_mri.scripts import v05lt_common as C

logger = logging.getLogger("seqref_mri.v05lt_epoch0")

__version__ = "0.3"
__abbr__ = "SEQREF-MRI-V05LTE0"

SEEDS = (2026, 2027, 2028)
CKPT_SCHEMA = "seqref-v05lt-epoch0-checkpoint/1"
ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG")


def ckpt_name(seed: int) -> str:
    return f"epoch0_seed{seed}.pt"


def seed_dir(root: Path, seed: int) -> Path:
    return Path(root) / f"seed_{seed}"


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError as exc:
        C.fail("ENV_CPUINFO_UNREADABLE", str(exc))
    C.fail("ENV_CPU_MODEL_MISSING", "/proc/cpuinfo has no 'model name'")


def environment() -> dict:
    import numpy as np
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
            "cpu_model": _cpu_model(), "os_cpu_count": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms":
                torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only":
                torch.is_deterministic_algorithms_warn_only_enabled(),
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
            "env": {k: os.environ.get(k) for k in ENV_VARS}}


def _git(repo: Path, *args) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                       text=True)
    C.require(r.returncode == 0, "GIT_FAILED", r.stderr.strip())
    return r.stdout.strip()


def implementation(repo: Path) -> dict:
    return {"abbr": __abbr__, "version": __version__,
            "script_sha256": C.sha256_file(Path(__file__)),
            "common_sha256": C.sha256_file(Path(C.__file__)),
            "repo_head": _git(repo, "rev-parse", "HEAD"),
            "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD")}


def _store(d: Path) -> C.ArtifactStore:
    return C.ArtifactStore(d / "ckpt", C.Ledger(d / "ledger.jsonl"),
                           d / "_incomplete")


def _build(seed: int, model_factory=None):
    """Production: ffr.build_model(init_seed=seed). model_factory is
    fixture-only (stub architecture)."""
    if model_factory is not None:
        return model_factory(seed)
    from seqref_mri.src import free_flow_runtime as ffr
    return ffr.build_model(init_seed=seed)


def _build_and_commit(store: C.ArtifactStore, seed: int, env: dict,
                      impl: dict, hook=None, model_factory=None) -> str:
    import torch
    model = _build(seed, model_factory)
    msh = C.model_state_sha256(model)
    meta = {"schema": CKPT_SCHEMA, "seed": seed, "init_seed": seed,
            "epoch": 0, "model_state_sha256": msh}
    sd = model.state_dict()
    store.stage(ckpt_name(seed),
                lambda fh: torch.save({"model": sd, "meta": meta}, fh), hook)
    store.commit({"epoch0": ckpt_name(seed)},
                 {"seed": seed, "model_state_sha256": msh, "env": env,
                  "implementation": impl}, hook)
    return msh


def _verify(store: C.ArtifactStore, seed: int, model_factory=None) -> dict:
    body = store.recover()
    C.require(bool(body), "EPOCH0_NOT_COMMITTED", str(store.dir))
    rec = body["retained"]["epoch0"]
    model = load_epoch0(store.dir / rec["name"], body["model_state_sha256"],
                        seed, model_factory=model_factory)
    del model
    out = {"seed": seed, "file": rec["name"], "file_sha256": rec["sha256"],
           "model_state_sha256": body["model_state_sha256"]}
    store.ledger.append("verified", out)
    return out


def load_epoch0(path: Path, expected_model_state: str, seed: int,
                model_factory=None):
    """Committed epoch-0 file -> model in eval mode with verified state.
    Builds the architecture via build_model (which seeds torch), so the
    global torch **CPU** RNG state is saved before and restored after. The
    registered setup is CPU-only: no CUDA RNG is used, so none is saved.
    NumPy and Python RNGs are not consumed here."""
    import torch
    rng = torch.get_rng_state()
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        C.require(isinstance(obj, dict) and obj.get("meta", {}).get("schema")
                  == CKPT_SCHEMA and obj["meta"].get("seed") == seed,
                  "EPOCH0_FILE_INVALID", str(path))
        got = C.model_state_sha256(obj["model"])
        C.require(got == expected_model_state == obj["meta"][
            "model_state_sha256"], "EPOCH0_STATE_MISMATCH",
            f"{path.name}: {got} != {expected_model_state}")
        model = _build(seed, model_factory)
        model.load_state_dict(obj["model"], strict=True)
        model.eval()
        C.require(C.model_state_sha256(model) == got, "EPOCH0_LOAD_MISMATCH",
                  path.name)
    finally:
        torch.set_rng_state(rng)
    return model


def assert_matches_epoch0(model, expected_model_state: str) -> None:
    """195 §4 pairing: an arm's freshly built initial weights must equal
    the seed's epoch-0 model-state hash before the first update."""
    got = C.model_state_sha256(model)
    if got != expected_model_state:
        C.fail("EPOCH0_ARM_MISMATCH", f"initial model-state {got} != "
               f"epoch-0 {expected_model_state}")


def create(root: Path, seed: int, repo: Path, hook=None,
           model_factory=None) -> dict:
    d = seed_dir(root, seed)
    if d.exists():
        C.fail("EPOCH0_DIR_EXISTS", f"{d} exists; a fresh start is refused "
               "-- use the explicit --recover procedure")
    d.mkdir(parents=True)
    store = _store(d)
    env, impl = environment(), implementation(repo)
    store.ledger.append("start", {"seed": seed, "env": env,
                                  "implementation": impl}, hook)
    _build_and_commit(store, seed, env, impl, hook, model_factory)
    return _verify(store, seed, model_factory)


def recover(root: Path, seed: int, repo: Path, env_override=None,
            model_factory=None) -> dict:
    """`env_override` and `model_factory` are fixture-only."""
    d = seed_dir(root, seed)
    C.require(d.is_dir(), "EPOCH0_RECOVER_NO_DIR", str(d))
    store = _store(d)
    body = store.recover()
    recs = store.ledger.records()
    if body and recs[-1]["kind"] == "verified":
        logger.info("[%s] seed %d already complete; no change", __abbr__, seed)
        return {**recs[-1]["body"], "recovery": "already_complete"}
    env = environment() if env_override is None else env_override
    impl = implementation(repo)
    if body:
        store.ledger.append("recover", {"action": "verify_only"})
        return {**_verify(store, seed, model_factory),
                "recovery": "verified_committed"}
    starts = [r for r in recs if r["kind"] == "start"]
    C.require(len(starts) == 1, "EPOCH0_RECOVERY_INCOMPATIBLE",
              f"{len(starts)} start records; cannot establish compatibility")
    st = starts[0]["body"]
    diffs = [k for k, a, b in (("seed", st["seed"], seed),
                               ("implementation", st["implementation"], impl),
                               ("environment", st["env"], env)) if a != b]
    if diffs:
        C.fail("EPOCH0_RECOVERY_INCOMPATIBLE", f"seed {seed}: {diffs} differ "
               "from the start record; no rebuild")
    logger.warning("[%s] EPOCH0_RECOVERY_REBUILD: seed %d had no committed "
                   "checkpoint; rebuilding from the same seed", __abbr__, seed)
    store.ledger.append("recover", {"action": "rebuild_uncommitted",
                                    "env": env, "implementation": impl})
    _build_and_commit(store, seed, env, impl, model_factory=model_factory)
    return {**_verify(store, seed, model_factory),
            "recovery": "rebuilt_uncommitted"}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="V05LT epoch-0 checkpoint")
    ap.add_argument("--seed", type=int, required=True, choices=SEEDS)
    ap.add_argument("--repo", default=str(C.Paths().repo))
    ap.add_argument("--recover", action="store_true")
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    root = C.Paths(repo=repo).v05lt / "epoch0"
    try:
        out = (recover if a.recover else create)(root, a.seed, repo)
    except C.V05LTError as exc:
        print(f"RESULT FAIL {exc.code}")
        return 2
    except Exception as exc:  # noqa: BLE001 -- logged, never swallowed
        logger.exception("UNEXPECTED: %s", exc)
        print(f"RESULT FAIL UNEXPECTED {type(exc).__name__}")
        return 2
    print(f"RESULT PASS seed={a.seed} file_sha256={out['file_sha256']} "
          f"model_state_sha256={out['model_state_sha256']} "
          f"recovery={out.get('recovery', 'none')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
