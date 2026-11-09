# SEQREF-MRI-DIAGJSHF1 v0.4 -- DIAG-JSH preparation step F1 (frozen mini-spec v1.0 §0 F1, §1.3, §5, §6, §8)
# LIFETIME: KEEP
# =============================================================================
# Run ONLY through diag_jsh_f1_cmd.sh (reviewed), from the V05LT tree, with
# the V05LT interpreter -B -E -s. Reads real data; computes NO arm
# difference. Order:
#   0. the command has created the empty run folder OUT; Python checks it
#      (inside DIAG_JSH/out, no symlinked ancestor, empty) and installs the
#      write guard on OUT ONLY (code and every other path protected) BEFORE
#      any write (log) and before any pinned import;
#   1. disk check (free >= reserve + F1 budget) and the hard wall cap;
#      diagnostic code = reviewed hashes;
#   2. pinned binding (bind_production): launch chain vs approved sha256 ->
#      launch-inputs record (pin, inputs, HEAD = pin); completion re-verified
#      (read-only); registered backend (12/12, deterministic off) checked;
#   3. inputs: data, eligibility, P4, s_ref, ZF; six runs by pinned
#      load_run; selected checkpoints; population check (§1.3 item 1);
#   4. early check + probe: 2 slices of the first checkpoint reproduce the
#      stored rows; the preparation limit is then derived and enforced
#      (never above the hard cap);
#   5. FULL reproduction of every slice of all six selected checkpoints
#      (§1.3 item 2); disk checked after each checkpoint;
#   6. loaded modules, H-D/H-F thirds, retained slices, measured timing
#      components for the main-run limit;
#   7. manifest SEQREF-MRI-DIAG-JSH-MAN (disk check first; write-once).
# The main-run limits (T with plot time; disk projection of DIAG-I3 outputs)
#   are NOT final here: they are completed and bound at F2 (flagged in the
#   manifest as "pending F2"). S-M parity: recorded against a PROPOSED
#   criterion that needs registration before DIAG-I3 may use it.
# Prints only STOP/PASS, stages, deviation maxima, counts, hashes.
# Exit: 0 PASS; 1 typed STOP (DiagError); 2 usage / unexpected error;
#   77 write-guard halt (a write outside OUT was attempted).
# Supported write paths (the only ones; everything else halts, exit 77):
#   OUT/f1.log (BudgetLogHandler), OUT/diag_jsh_manifest.json and its
#   .sha256 sidecar (WriteBudget.write_new). No tmp folder: any temporary
#   file attempt halts (fail closed). Native (C-level) writes bypass the
#   audit hook and are NOT covered (residual, untested).
# Time: Python overall deadline = start + MAX_WALL_S - 120 s (bootstrap and
#   summaries included); replay deadline = min(overall, probe end +
#   1.5 x (t_probe x N x 6 + t_load x 6)). The command terminates at
#   MAX_WALL_S - 90 s (TERM) and kills at MAX_WALL_S - 30 s: a hard cap.
# Changelog (v0.4, 2026-11-09): bind_production split into a thin
#   production constructor (pinned Paths, ProductionStages, Campaign with
#   fixture=False -- unchanged) and bind_from(), which holds every binding
#   check and is exercised under the strict guard on a pinned fixture
#   campaign (test_diag_jsh_bind.py). No behaviour change in production.
# Changelog (v0.3, 2026-11-09; F1 review 2):
#   * every write reserved in an enforced WriteBudget before it happens;
#     strict guard (budget-registered files only); log via BudgetLogHandler
#     with its own allowance, no console copy; tmp folder removed;
#   * replay deadline separated from the overall Python deadline;
#   * S-M: fallback label only, zero-MSE slices counted, criterion not
#     adopted.
# Changelog (v0.2, 2026-11-09; F1 review):
#   * guard installed on OUT before any write; OUT validated (no symlinks);
#   * disk checked before preparation, after each checkpoint and before the
#     manifest; F1 write budget registered; fixed main-run allowance removed;
#   * hard wall cap and derived preparation limit enforced;
#   * required manifest fields, resolvable row paths, selection evidence;
#   * bootstrap probe times the actual estimator calls (ratio as ratio);
#   * bind_production / prepare split; synthetic integration test.
# Changelog (v0.1, 2026-11-09): first F1 command.
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

__abbr__ = "SEQREF-MRI-DIAGJSHF1"
__version__ = "0.3"
DIAG_ROOT = Path("/home/benjamin/CSMFII/v06lt_evidence/DIAG_JSH")
V05LT = Path("/home/benjamin/CSMFII")
PIN = "7455da8027f33a6cca4dbf72f04e72420e938f45"
CODE_FILES = ("diag_jsh_core.py", "diag_jsh_adapter.py",
              "diag_jsh_f1_prepare.py")
MARGIN = 1.5                    # §8
# F1 write budget (ENFORCED by WriteBudget; the value is an estimate):
# log + manifest + sidecar. Justification: the
# synthetic integration run (6 checkpoints, 8 slices, 4 volumes) wrote
# 17,142 bytes in total (test_diag_jsh_f1_integration P7). The manifest grows
# with volumes (subgroup lists) and loaded modules, the log with stages, not
# with slices: scaled to 97 volumes about 0.42 MB, so 16 MiB is about 40 x.
# The command's stdout copy (beside OUT) is bounded by the same log.
# Registered here; reviewed with this file.
F1_BUDGET = 16 * 2 ** 20
LOG_ALLOWANCE = 4 * 2 ** 20     # f1.log share of F1_BUDGET
WALL_MARGIN_S = 120             # Python overall deadline before MAX_WALL_S
MIN_WALL_S = 300
# Counted interval computations of §2-§5 (timed with the actual calls):
N_DELTA = 22        # PSNR, NLL, B, J + 6 S-C cells + 6 S-C contrib + 6 S-M
N_RATIO = 1         # NLL ratio (density label)
N_SPEARMAN = 2      # rho(dJ, dPSNR), rho(dNLL, dPSNR)
N_SUBGROUP = 20     # 2 stratifiers x (PSNR, NLL, B, J + 6 S-C cells)
logger = logging.getLogger("seqref_diag.jsh_f1")


def _args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--chain", required=True)
    p.add_argument("--chain-sha256", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reserve-bytes", type=int, required=True)
    p.add_argument("--max-wall-s", type=float, required=True)
    for f in CODE_FILES:
        p.add_argument(f"--sha-{f.split('.')[0]}", required=True)
    return p.parse_args(argv)


def check_out(out: Path, parent: Path) -> None:
    """OUT: existing, empty, directly inside parent, no symlinked part."""
    ap = Path(os.path.abspath(out))
    ok = (ap.is_dir() and not any(ap.iterdir()) and ap.parent == parent and
          os.path.realpath(ap) == str(ap))
    if not ok:
        raise SystemExit(f"STOP usage: {out} must be an existing empty "
                         f"folder directly in {parent}, reached without "
                         "symlinks")


def start(out: Path, A, reserve: int):
    """Guard first, then the enforced budget, then the only log."""
    A.install_write_guard(out, git_repo=V05LT)
    budget = A.WriteBudget(out, reserve, F1_BUDGET)
    os.environ["TMPDIR"] = str(out)          # unregistered -> halts
    os.environ["MPLCONFIGDIR"] = str(out)
    h = A.BudgetLogHandler(budget, out / "f1.log", LOG_ALLOWANCE)
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[h], force=True)
    return budget


def main(argv=None) -> int:
    a = _args(argv)
    code_dir = Path(__file__).resolve().parent
    if code_dir != DIAG_ROOT / "code":
        print("STOP usage: run from DIAG_JSH/code", file=sys.stderr)
        return 2
    out = Path(a.out)
    try:
        check_out(out, DIAG_ROOT / "out")
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    sys.path.insert(0, str(code_dir))
    import diag_jsh_adapter as A                 # numpy + core only
    import diag_jsh_core as K
    t0 = time.monotonic()
    if a.max_wall_s < MIN_WALL_S or a.reserve_bytes < 0:
        print(f"STOP usage: MAX_WALL_S >= {MIN_WALL_S}, reserve >= 0",
              file=sys.stderr)
        return 2
    try:
        budget = start(out, A, a.reserve_bytes)
        lim = {"reserve": a.reserve_bytes, "budget": F1_BUDGET,
               "max_wall_s": a.max_wall_s,
               "deadline": t0 + a.max_wall_s - WALL_MARGIN_S}
        want = {f: getattr(a, f"sha_{f.split('.')[0]}") for f in CODE_FILES}
        code = A.check_code(code_dir, want)
        ctx = bind_production(a, A, K)
        sha = prepare(ctx, out, lim, code, A, K, budget)
    except K.DiagError as exc:
        print(f"RESULT DIAG_F1 STOP {exc.code}")
        return 1
    except Exception as exc:  # noqa: BLE001 -- logged, never swallowed
        logger.error("UNEXPECTED %s:\n%s", type(exc).__name__,
                     traceback.format_exc())
        print(f"RESULT DIAG_F1 FAILED UNEXPECTED {type(exc).__name__}")
        return 2
    print(f"RESULT DIAG_F1 PASS manifest_sha256={sha}")
    return 0


def bind_production(a, A, K) -> dict:
    """Production context through the pinned V05LT code only."""
    sys.path.insert(0, str(V05LT))
    import torch
    torch.set_num_threads(12)
    torch.set_num_interop_threads(12)
    from seqref_mri.scripts import v05lt_common as C
    from seqref_mri.scripts import v05lt_launch as L
    from seqref_mri.scripts import v05lt_prodstages as PS
    paths = C.Paths()
    stages = PS.ProductionStages(paths, paths.data_root)
    camp = L.Campaign(paths.v05lt, stages, chain=(Path(a.chain),
                                                  a.chain_sha256, paths.repo),
                      fixture=False)
    return bind_from(camp, stages, paths.repo, a.chain, a.chain_sha256, A, K,
                     pin=PIN)


def bind_from(camp, stages, repo, chain, chain_sha256, A, K, *,
              pin: str) -> dict:
    """Every binding check, on an already constructed pinned Campaign and
    stages (production: bind_production; test: a pinned fixture campaign).
    Read-only; runs under the strict write guard."""
    from seqref_mri.scripts import v05lt_prodstages as PS
    A.pinned("DIAG_BINDING", "launch chain", camp._verify_chain)
    _ev, ev_sha = A.pinned("DIAG_NOT_COMPLETE", "completion",
                           camp.verify_completion)
    backend = A.pinned("DIAG_BACKEND", "backend", PS.check_backend,
                       PS.backend_now())
    ins = A.pinned("DIAG_BINDING", "bound inputs", stages._bound)
    head = A.pinned("DIAG_BINDING", "HEAD", PS._git, repo, "rev-parse",
                    "HEAD")
    K.require(head == pin, "DIAG_PIN_MISMATCH", f"HEAD {head} != pin")
    for k in ("record_sha256", "pin_code", "split_manifest", "eligibility",
              "zf", "data_content"):
        K.require(k in ins and ins[k], "DIAG_BINDING", f"bound input {k}")
    logger.info("STAGE 2 binding, completion and backend OK")
    data = A.pinned("DIAG_INPUT_INVALID", "data", stages.data)
    zf = A.load_zf(stages)
    return {"binding": {"pin": head, "launch_chain": {"path": str(chain),
                                                      "sha256": chain_sha256},
                        "launch_inputs_sha256": ins["record_sha256"],
                        "completion_evidence_sha256": ev_sha,
                        "backend": backend,
                        "inputs": {k: ins[k] for k in (
                            "split_manifest", "eligibility", "zf",
                            "data_content")}},
            "data": data, "zf_rows": zf["rows"],
            "zf_record_sha256": zf["record_sha256"],
            "load_selected": lambda k: A.load_selected(stages, *k),
            "loaded_modules": lambda: A.check_loaded(stages, ins["pin_code"])}


def bootstrap_probe(K, vols: list, strat: dict) -> float:
    """Time of the counted §2-§5 interval calls, each with its own core
    function, on random matrices of the real shape (timing only)."""
    import numpy as np
    rng = np.random.default_rng(0)
    V = len(vols)
    pos = {v: i for i, v in enumerate(vols)}
    dr = K.make_draws(V, [[len(g) for g in strat[s]["thirds"]]
                          for s in ("H-D", "H-F")])
    X, Y = rng.standard_normal((3, V)) + 5, rng.standard_normal((3, V)) + 5
    t0 = time.monotonic()
    for _ in range(N_DELTA):
        K.delta_estimate(X, Y, dr["v"])
    for _ in range(N_RATIO):
        K.ratio_estimate(X, Y, dr["v"])
    for _ in range(N_SPEARMAN):
        K.spearman_estimate(X - Y, (Y - X) ** 3, dr["v"])
    for i in range(N_SUBGROUP):
        s = ("H-D", "H-F")[i % 2]
        groups = [[pos[v] for v in g] for g in strat[s]["thirds"]]
        K.subgroup_estimates(X, Y, groups, dr["strata"][i % 2])
    return time.monotonic() - t0


def prepare(ctx: dict, out: Path, lim: dict, code: dict, A, K,
            budget) -> str:
    t_all = time.monotonic()
    hard = lim["deadline"]                   # overall Python deadline
    budget.check(0, "before preparation")
    data = ctx["data"]
    vv, elig, p4, s_ref = (data["vv"], data["eligibility"], data["p4"],
                           data["s_ref"])
    sels = {k: ctx["load_selected"](k) for k in A.SEED_ARMS}
    docs = {k: s["rows"] for k, s in sels.items()}
    entries = A.check_population(docs, ctx["zf_rows"])
    n = len(entries)
    K.require(n == len(vv), "DIAG_POPULATION_MISMATCH",
              f"{n} rows vs {len(vv)} view slices")
    A.check_deadline(hard, "inputs")
    logger.info("STAGE 3 inputs bound; population %d slices OK", n)

    k0 = A.SEED_ARMS[0]
    t0 = time.monotonic()
    m0, _ = A.load_model(sels[k0])
    t_load = time.monotonic() - t0
    t0 = time.monotonic()
    pr = A.replay(m0, vv, elig, p4, s_ref, positions=[0, 1], deadline=hard)
    t_probe = (time.monotonic() - t0) / 2
    A.reproduce_prefix(docs[k0], pr)
    del m0
    t_prep = MARGIN * (t_probe * n * 6 + t_load * 6)
    deadline = min(hard, time.monotonic() + t_prep)     # replay deadline
    logger.info("STAGE 4 early 2-slice check OK; replay limit %.0f s "
                "(overall Python limit at MAX_WALL_S - %d s)", t_prep,
                WALL_MARGIN_S)

    repro, hfs, smdev, smzero, t_loads, t_reps, disk = ({}, [], 0.0, 0, [],
                                                        [], [])
    for k in A.SEED_ARMS:
        rid = f"{k[0]}_{k[1]}"
        t0 = time.monotonic()
        model, _vc = A.load_model(sels[k])
        t_loads.append(time.monotonic() - t0)
        t0 = time.monotonic()
        rep = A.replay(model, vv, elig, p4, s_ref, deadline=deadline)
        t_reps.append(time.monotonic() - t0)
        repro[rid] = A.reproduce(docs[k], rep, n)
        hfs.append(rep["hf"])
        smdev = max(smdev, rep["sm_rel_dev"])
        smzero += rep["sm_zero_mse_slices"]
        del model
        disk.append(budget.check(0, f"after {rid}"))
        logger.info("STAGE 5 %s reproduced: %s", rid, repro[rid])
    hf = A.check_reference_equal(hfs)
    mask_sha = A.canonical_sha(sorted(rep["mask_regen"].items()))

    mods = ctx["loaded_modules"]()
    strat = A.stratifiers(entries, ctx["zf_rows"], hf)
    retained = A.retained_slices(entries, elig)
    vols = sorted({e[0] for e in entries})
    t_boot = bootstrap_probe(K, vols, strat)
    A.check_deadline(hard, "summaries")
    man = {
        "schema": "SEQREF-MRI-DIAG-JSH-MAN/0.2", "spec": "SEQREF-MRI-DIAG-JSH "
        "v1.0 8665d790f4ab96d1cb526c442826da06b81a672c387a170a3c08adc2f4018336",
        "contrast": "lambda2 - lambda0 objective comparison (not a "
                    "before/after refinement experiment)",
        "f1": {"abbr": __abbr__, "version": __version__,
               "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime())},
        "code_sha256": code, "loaded_pinned_modules": mods,
        **ctx["binding"], "zf_record_sha256": ctx["zf_record_sha256"],
        "checkpoints": {f"{k[0]}_{k[1]}": {
            x: s[x] for x in ("epoch", "model_state_sha256", "weights",
                              "weights_sha256", "rows_file", "rows_sha256",
                              "selection_evidence")}
            for k, s in sels.items()},
        "population": {"slices": n, "volumes": len(vols),
                       "slice_ids_sha256": docs[k0]["slice_ids_sha256"],
                       "case_order_sha256": A.canonical_sha(entries),
                       "masks_regenerated_sha256": mask_sha},
        "reproduction": {"limits": {"psnr_db": K.PSNR_REPLAY_TOL,
                                    "nll_nats_per_dim": K.NLL_REPLAY_TOL},
                         "per_checkpoint": repro},
        "sm_parity": {"max_rel_dev_vs_registered_mse": smdev,
                      "zero_mse_slices_not_compared": smzero,
                      "criterion": "NOT ADOPTED (needs justification and "
                                   "zero/near-zero MSE handling)",
                      "label": "magnitude-spectrum error (frozen fallback)"},
        "subgroups": {s: {"thirds": strat[s]["thirds"],
                          "boundaries": [[strat[s]["values"][g[0]],
                                          strat[s]["values"][g[-1]]]
                                         for g in strat[s]["thirds"]],
                          "sha256": strat[s]["sha256"]} for s in strat},
        "retained_slices": retained,
        "preparation_limits": {
            "max_wall_s": lim["max_wall_s"],
            "python_deadline_rule": f"start + max_wall_s - {WALL_MARGIN_S} "
                                    "s (all stages)",
            "command_cap": "TERM at max_wall_s - 90 s, KILL at - 30 s",
            "replay_limit_s": t_prep,
            "replay_rule": "1.5 x (t_probe x N x 6 + t_load x 6), capped by "
                           "the Python deadline",
            "f1_wall_s": time.monotonic() - t_all,
            "log_allowance_bytes": LOG_ALLOWANCE,
            "reserve_bytes": lim["reserve"], "f1_budget_bytes": lim["budget"],
            "disk_checks": disk},
        "main_run_limits": {
            "status": "PENDING F2: t_plot and the DIAG-I3 output projection "
                      "are added and bound at F2",
            "rule": "T = 1.5 x (t_probe x N x 6 + t_load x 6 + t_boot + "
                    "t_plot)  (§8)",
            "t_probe_per_slice_s": t_probe, "t_model_load_s": t_load,
            "t_bootstrap_s": t_boot,
            "bootstrap_counts": {"delta": N_DELTA, "ratio": N_RATIO,
                                 "spearman": N_SPEARMAN,
                                 "subgroup": N_SUBGROUP},
            "measured_full_replay_s": t_reps, "measured_load_s": t_loads}}
    body = (json.dumps(man, sort_keys=True, indent=1, allow_nan=False)
            + "\n").encode()
    sha = A.sha256_bytes(body)
    side = f"{sha}  diag_jsh_manifest.json\n".encode()
    budget.check(len(body) + len(side), "before manifest")
    budget.write_new(out / "diag_jsh_manifest.json", body, "manifest")
    budget.write_new(out / "diag_jsh_manifest.json.sha256", side, "sidecar")
    logger.info("STAGE 7 manifest written (%d bytes); F1 bytes %d "
                "(budgeted %d)", len(body), A.tree_bytes(out), budget.used)
    return sha


if __name__ == "__main__":
    sys.exit(main())
