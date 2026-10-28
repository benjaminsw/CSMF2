# SEQREF-MRI-V05LTF v0.45 -- scripts.v05lt_fixtures
# LIFETIME: KEEP
# =============================================================================
# Purpose: fixtures of SEQREF-MRI-V05LT (195 §7 + PA1). This version holds
#   LOAD, ANA (I1.8(b) b1: verified loader, analysis stage),
#   SMK (I1.10: F17 smoke runner, local pre-check),
#   VRF, NLD, PEV (I1.8 b0: G1 validation-row files, G3 NLL components,
#   G2 post-run evaluation),
#   F06, F15, BSA, BSB, CFG and the F04 statistics-stream part (I1.8a),
#   CLS (item 18, I1.7c2),
#   PUB, E0V, F16, F22E (epoch-0 part of F22), F21E (I1.7c1),
#   F12, F22Z (ZF part of F22), F23, CPB DC/ZF timing (I1.7b),
#   F10V, F10, F11, F22R, CPB, STB (I1.7a),
#   TFX, REP, F09, F14, F13C, F21C, F08R, KIL, RNGC (I1.6c),
#   RET, RES, TRN, GRD, CHL (I1.6b), F07, F08 (part), MET, VAL (I1.6a), F03, F13/F21 (parts), E0S (I1.5),
#   F19 (I1.4) and the I1.3 groups (SEQREF-MRI-V05LTI13 v0.2): F01, F02, F04 (part),
#   F05 (named modules), F09CS (commit/ledger crash safety -- the early part
#   of F09, addition B), F18, F20, U13 (model-state hash utility; F13 itself
#   is I1.5) and SCAN (targeted static check, addition C).
# Two parts per group:
#   * synthetic -- always runs; temp directories, synthetic h5 volumes and
#     synthetic manifests/registries; never the real data or inputs;
#   * host      -- only with --host; real repo, inputs, V02 sources, P0S,
#     training-split data. Never constructs split="val" (the V02 holdout)
#     EXCEPT the registered F10 exception (195 §6): group F10 reproduces
#     the historical D3-subset anchors on the V02 checkpoints only, in a
#     separate process; no V05LT checkpoint touches holdout data. Never
#     writes into the repository or results (temp dirs only).
#     The c3 groups (HRN, HNC, HRD, HRB) build split="val" ONLY over
#     synthetic holdout .h5 files in a temporary data root, never the real
#     data root; HRB's host part adds only the registered P4 and S_ref.
#     The c4 groups (HLW, HLB) likewise use synthetic holdout .h5 files in
#     temporary folders only; HLB's host part trains six fixture runs with
#     the real architecture on synthetic training data (temporary folders).
# A group is PASS only if its host part ran; without --host the result is
#   PASS-SYNTHETIC (local pre-check, never recorded as a host PASS).
# Synthetic stand-ins (disclosed, fixture-confined): h5 volumes, split
#   manifests, registries and git repositories built in temp dirs; crash
#   simulation by os._exit in a forked child at hook points (process kill,
#   NOT power loss / lost page cache); torn ledger tail written directly.
# CONVENTION: exit 0 = all selected groups PASS (or PASS-SYNTHETIC without
#   --host), 2 = any failure; every failure logged with a typed code.
# Changelog (v0.45, 2026-10-28; I1.10 revision-1 review):
#   * SMK: complete optimiser-state comparison (step counters, param-group
#     settings, moments; synthetic and real resume state); stability
#     documents with missing / non-finite / unexpected values, slices or
#     keys refused, null NMSE_u only where ineligible, differing backends
#     refused; backend enforcement (wrong expected backend refused in the
#     parent and directly in a child before any stage; every child's backend
#     and run code recorded); run-code approval; freeze-code
#     provenance (unknown refused); host part: the exact frozen configuration
#     on the host (9bffdcd6...) loads unmodified, inputs re-verified.
# Changelog (v0.44, 2026-10-28; I1.10, SEQREF-MRI-V05LTI110 v0.1):
#   * SMK (synthetic by design): v05lt_smoke freeze -> run -> record on
#     synthetic sources with the stub model through a fixture-only context
#     (_smk_ctx); output-root, configuration and input refusals; real child
#     SIGKILL at the declared boundary (and its check) and comparison;
#     stability, item-24, report and record checks; comparison negatives;
#     CLI refusal.
# Changelog (v0.43, 2026-10-27; c4 revision-2 review):
#   * HLW: re-entry read errors (PermissionError while hashing a record,
#     then the index; fixture patch of C.sha256_file) -> recorded c3 halt
#     HOLDOUT_STAGE_ERROR (stage, task, type, traceback), mirrored; clearing
#     the fault alone blocked; consumed decision stale; bound retry
#     completes and retains the failed attempt.
# Changelog (v0.42, 2026-10-27; c4 revision-1 review):
#   * HLW: an altered record or index at re-entry is a recorded c3 halt
#     (HOLDOUT_OUTPUT_MISMATCH, stage reentry, task) mirrored by
#     reference; restoring the files alone stays blocked; unrelated,
#     unapproved and stale (consumed) decisions refused; bound retries
#     complete and retain the failed attempt (task directory, index).
# Changelog (v0.41, 2026-10-27; I1.9 c4, SEQREF-MRI-V05LTI19C4 v0.1):
#   * HLW (synthetic by design): ONE fixture campaign through the launcher
#     (ProductionStages, stub model) whose analysis chain, selection and
#     frozen holdout configuration are built from that campaign; the
#     launcher's holdout command: new/resume never reach it; refusals
#     before the gate (no campaign, unapproved / wrong-kind approval,
#     launch-chain mismatch); verified completion on a campaign copy
#     (missing / extra / repeated run identities, incomplete launch, open
#     campaign halt, altered final weights, missing post-run record or
#     epoch-0 stage); selection binding (foreign campaign, 7 unit edits);
#     opens only after the gate; mirror records; re-entry without
#     re-evaluation; mirror-write failures and reconciliation; holdout
#     halt + bound retry (no second retry mechanism); concurrent
#     invocations; read-only status.
#   * HLB (addition B): the same path in a fresh process; host: real
#     architecture (six runs, 3 epochs, synthetic data); local: stub.
#   * _prd_layout(factory="real") omits the fixture model factory
#     (fixture-only).
# Changelog (v0.40, 2026-10-27; c3 revision-1 review):
#   * HRF: post-gate faults (data loading, width check, holdout view,
#     parent / state preparation, existing-output verification, index;
#     typed and unexpected; hooks and fixture patches of module functions)
#     -> recorded halts; cleared fault without a decision refused; bound
#     retry completes. Output-root guard: symlinked root, alias through a
#     symlinked parent into results/v05lt, ledger symlink, production root
#     redirected (root or parent symlink), other root, unregistered data
#     root -- all refused before anything is created.
#   * The c3 groups also cover runner v0.2 (stage boundary, check_out_dir).
# Changelog (v0.39, 2026-10-27; I1.9 c3, SEQREF-MRI-V05LTI19C3 v0.1):
#   * HRN (synthetic by design): holdout runner on synthetic holdout .h5
#     files -- approval / chain / binding refusals, scope, coincident
#     checkpoints evaluated once (end to end with _stub_w1000), width and
#     mask checks (fixture stand-in view), write-once records, reuse,
#     production-path refusals, round trip through the b3 loader and the
#     holdout analysis.
#   * HNC (addition C): h5py opens tracked -- refused gates open nothing;
#     a valid run opens holdout files only after the recorded gate.
#   * HRD (addition D): kill after staging, failed tasks, altered outputs
#     -> recorded halts; bound retry retains the failed attempt; owner halt.
#   * HRB (addition B): real architecture through the runner's evaluation,
#     publication and verification functions on synthetic holdout files.
#   * _syn_campaign / _b2_analysis / _b3_setup: optional fixture-only
#     factory (default unchanged).
# Changelog (v0.38, 2026-10-26; failed host run of revision 3 / review):
#   * PRD: excluded venv folder inside the package (as on the host):
#     third-party module accepted; seqref_mri module there refused; source
#     tree symlink resolving into it refused (identity and loaded check);
#     external seqref_mri module refused.
#   * PRB: child result read by _child_payload -- the child's error code and
#     message are preserved; a missing payload is refused explicitly
#     (PRB_RESULT_MISSING); tested in PRD.
# Changelog (v0.37, 2026-10-26; c2-wiring revision-2 review):
#   * PRD: directory / file symlinks under seqref_mri/src (copied tree and
#     committed copy after the pin) refused; a seqref_mri module executing
#     from an external location refused by the loaded-module check;
#     third-party modules outside the package not pin-checked.
# Changelog (v0.36, 2026-10-26; c2-wiring revision-1 review):
#   * PRD: changed scientific dependency / added / config file refused while
#     the launcher scripts are unchanged; production HEAD-blob pin check on
#     a committed copy (changed, untracked, HEAD moved); loaded-module check.
#   * PRD: reused records with a wrong mask hash, code or backend refused,
#     incl. a wrong mask hash shared by all records of a run; reuse of a
#     complete campaign makes no evaluation and no decode (call counters);
#     epoch-0 context with other code / backend refused.
#   * PRD: direct launcher-to-input binding assertions (LAUNCH_INPUTS_UNBOUND).
# Changelog (v0.35, 2026-10-26; I1.9 c2 production wiring, V05LTI19C2P v0.1):
#   * PRD (synthetic by design): v05lt_prodstages.ProductionStages on fixture
#     layouts through the launcher (6-run campaigns; accepted results loader
#     reads them); refinements 1-6: launch-inputs record and chain role,
#     fixture overrides refused in production, absent / complete /
#     incomplete outputs with owner-decision recovery, planner and per-task
#     record checks (coincident and distinct checkpoints), stage-boundary
#     recording (addition C), isolation through the wiring.
#   * PRB (addition B): real architecture through ProductionStages in a
#     fresh process (host: real-data fixture layout, one seed, lam0 + lam2,
#     3 epochs; local: synthetic data, lam2); timing provisional.
#   * LCH: launcher v0.4 (chain role launch_inputs; fixture stage set gains
#     bind_chain; production refusal codes).
# Changelog (v0.34, 2026-10-25; c2 revision-1 review fixes):
#   * FWT / DHS: sidecar as directory, dangling symlink, invalid UTF-8,
#     wrong name; record / manifest with invalid UTF-8; staging, write and
#     quarantine failures -> typed codes (raw exceptions fail the check).
#   * LCH: the same faults through the launcher -> recorded campaign halt
#     with the typed cause; repair alone -> LAUNCH_HALT_UNRESOLVED; a bound
#     retry completes (final sidecars, data-manifest sidecar before epoch 0
#     and before training, staging write failure).
# Changelog (v0.33, 2026-10-25; I1.9 c2, PA5-A/B):
#   * DHS: data-content manifest generate/verify (refusals; host: temporary
#     manifest of the real 973 files + one full verification, timed).
#   * FWT (synthetic) / FWR (real architecture): final-weights extraction,
#     recovery, refusals; accepted loader and post-run evaluation.
#   * Regressions: launcher fixture stages use the REAL datahash and final
#     weights; synthetic campaign post-evaluates final from the published
#     file; F18A selection final = final weights, resume refused.
# Changelog (v0.32, 2026-10-25; c1 revision-2 review fix):
#   * LCH: epoch-0 verification errors before execution (resume) and after
#     the stage returns are recorded halts (seed, original code);
#     restoration alone stays blocked; a bound retry succeeds.
# Changelog (v0.31, 2026-10-25; c1 revision-1 review fixes):
#   * LCH: clearing a data / epoch-0 / post-run fault alone does not permit
#     continuation; a campaign decision bound to the halt permits retry;
#     stale, unrelated and unapproved decisions refused; `halt` final;
#     malformed launch-chain entries -> typed refusals (approved hashes).
# Changelog (v0.30, 2026-10-25; I1.9 c1, SEQREF-MRI-V05LTI19 v0.1):
#   * LCH: launcher logic on the real trainer path (stub model, synthetic
#     data): launch chain, modes, locks, order, interruption and session
#     limit, owner decisions (binding, staleness, limits), terminal runs,
#     advancement on verified post-run outputs, data verification calls.
#   * LCR (addition B): two launchers started concurrently; exactly one runs.
#   * Epoch-0 and post-run stages are fixture stand-ins until c2 (declared).
# Changelog (v0.29, 2026-10-24; b3 revision-1 review fixes):
#   * HOL: mask identity -- invalid sha256 forms, a shared but incorrect
#     masks sha256 across all records, seed identity / per-row seed / width
#     refusals, independent seed recomputation; conditional statements.
#   * F18A: source identity alteration and deletion under the original
#     approved manifest; incomplete selection manifests with valid hashes;
#     selection vs learning-curve epochs.
# Changelog (v0.28, 2026-10-24; I1.8 b3, SEQREF-MRI-V05LTI18B3 v0.1):
#   * HOA: 21 holdout comparisons, appended population, validation
#     preservation, label scope, AM4-B, refusals. HOL: holdout records and
#     loader (scope, identity, masks, bank, bindings), holdout analysis and
#     report. F18A: selection manifest and analysis-chain evidence manifest
#     (approved hash, joint alteration, bindings). Synthetic holdout only.
# Changelog (v0.27, 2026-10-24; b2 revision-1 review fixes):
#   * PLT: plot 4 reference from |N_ref| incl. N_ref = -2 -> -1.98; CI not
#     containing the estimate renders. RPT: per-seed CIs and untruncated
#     appendices; verifier refuses self-consistent checksum lists that omit
#     required files, and incomplete plot / Markdown bindings.
# Changelog (v0.26, 2026-10-24; I1.8 b2, SEQREF-MRI-V05LTI18B v0.2):
#   * RPT: report bound to analysis/config/code; content copied from the
#     analysis; PA3/unavailable listing; write-once, staging, symlink,
#     registered-tree, hash, sidecar and malformed-document refusals.
#   * PLT: plots 1-7 values = analysis values; deterministic PNG without
#     metadata; backend refusal. E2E (addition A): 11-epoch stub chain.
# Changelog (v0.25, 2026-10-24; b1 revision-3 review fix):
#   * ANA: new run_analysis() test removing an entire epoch-0 midpoint row
#     (dependent comparisons unavailable; both DC blocks partial with the
#     loader "slice absent" entry listed, n_expected vs n_observed); d1b
#     uses the renamed keys.
# Changelog (v0.24, 2026-10-24; b1 revision-2 review fixes):
#   * ANA: missing-reference cases through run_analysis (epoch-0 R1+DC
#     record, R1 PSNR, all-slice NLL, midpoint NMSE_u / zero_error, ZF
#     SSIM): analysis completes, affected comparisons and descriptive
#     values unavailable with identities, D6 checks not performed listed,
#     references recomputed (stored summary not used); malformed bank,
#     prefix and DC structures -> LOAD_RECORD_INVALID.
# Changelog (v0.23, 2026-10-24; b1 review fixes):
#   * LOAD: synthetic case manifest bound in the layout; negatives for R1
#     rows missing declared prefixes / banks, epoch-0 and ZF missing values
#     (preserved; engine reports the comparisons unavailable with
#     identities), wrong types, non-finite values, trained missing values,
#     and panel cases differing from the case manifest.
# Changelog (v0.22, 2026-10-24; I1.8(b) b1, SEQREF-MRI-V05LTI18B v0.2):
#   * LOAD: synthetic campaign built with the real writers (6 stub runs,
#     post-run, epoch-0 and ZF records); loader positives (sources,
#     run_info, D6) and typed refusals (tamper, D6, D7, D8, provenance).
#   * ANA: PA4 difficult cases (tie rule), configuration write, gated
#     analysis, descriptive block, write-once, determinism, refusals.
# Changelog (v0.21, 2026-10-24; b0 review fixes):
#   * PEV: weights schema/meta refusals (schema, extra key, missing meta or
#     field, wrong type, unreadable file; committed seed/arm/epoch/model-
#     state != saved), VerifiedCheckpoint required, epoch and run identity
#     from the verified file; panels = declared cases on write; loader
#     refusals for missing/extra panel files, malformed record and sums.
#   * VRF: malformed row documents -> VAL_ROWS_INVALID (no raw exceptions);
#     trainer-written milestone weights load through the verified loader.
#     _raises_typed records a raw exception as a failed check.
# Changelog (v0.20, 2026-10-24; I1.8 b0, SEQREF-MRI-V05LTI18B v0.1):
#   * VRF (G1): row files committed per epoch, bound to the commit, summary
#     = committed validation summary, retained; crash before commit ->
#     quarantine + identical retraining; tamper / write failure refused.
#   * NLD (G3): sign convention, defined consistency check, production NLL
#     unchanged; host part on the real model.
#   * PEV (G2): verified checkpoint loading, deduplicated task, provenance,
#     refusals, no-overwrite publication; host part on the real model.
# Changelog (v0.18, 2026-10-23; I1.8(a) review fixes):
#   * F06: slice completeness -- deleted trained slice -> validity failure;
#     deleted epoch-0 slice -> affected comparisons unavailable with
#     identities; unexpected slice, wrong volume, eligibility binding.
#   * F15: readout family (R1-R0 etc.), DC identity results, PA3 reporting
#     (seeds reached / absent and why); BSA covers both.
# Changelog (v0.17, 2026-10-23; I1.8(a), SEQREF-MRI-V05LTI18 v0.1):
#   * F06 (aggregation through the statistics path), F15 (crossed and
#     per-seed bootstrap, r, labels at boundaries, outcomes, AM3, D1-D4).
#   * F04: statistics stream (one registered generator; global RNGs
#     untouched; array record). BSA: independent brute-force bootstrap,
#     exact equality (addition A). BSB: bitwise across two fresh processes
#     (addition B). CFG: analysis configuration and hash (addition C).
# Changelog (v0.16, 2026-10-22; I1.7(c2) review: CLS completed):
#   * CLS adds prepare, mask epoch, backward (finite / non-finite),
#     optimiser, validation summary, weights staging and commit injections;
#     records the terminal-I/O consequence.
# Changelog (v0.15, 2026-10-22; I1.7(c2), item 18):
#   * CLS: injected failures at EVERY trainer / shared-validation seam
#     (load, prepare, mask epoch, targets, objective finite/non-finite,
#     backward finite/non-finite, optimiser, validation load/NLL/decode/
#     summary, weights/resume staging, commit) ->
#     typed code, registered taxonomy, terminal run_end, no session_error,
#     never resumed; validate() unit cases; error outside the seams still
#     session_error.
# Changelog (v0.14, 2026-10-22; I1.7(c1) review fixes):
#   * PUB: dangling and directory symlinks at destination and staging,
#     through write_zf and write_epoch0_eval; gate judged on the
#     unfollowed name.
#   * F16: PNG text rows, font bitmaps, label refusals, duplicate cases.
# Changelog (v0.13, 2026-10-22; I1.7(c1), SEQREF-MRI-V05LTI17C v0.1):
#   * PUB: no-overwrite publication (empty dir, non-empty dir, file,
#     symlink refused; missing syscall -> stop; ZF race closed).
#   * E0V: epoch-0 evaluation + writer (synthetic; host child on a
#     temporary seed-2026 epoch-0 file, registered cases 0-1); timing.
#   * F16 panels; F22E epoch-0 non-finite; F21E real-trainer isolation
#     around the actual reconstruction evaluator.
# Changelog (v0.12, 2026-10-22; bug fix):
#   * F22Z interruption child now uses the fork-context Process and
#     join_bounded (timeout, TERM/KILL escalation, exit code checked).
# Changelog (v0.11, 2026-10-22; I1.7(b) review corrections):
#   * F12: EVAL_STATE_MISMATCH (GT and conditioning) and
#     DC_DETECTION_FAILED negatives; invalid tolerances (0, < 0, NaN, Inf,
#     non-float) refused in identity, AM1 and evaluate_states.
#   * F22Z: volume-weighted summary (0, 0 | 30 -> 15 dB), NMSE null
#     volumes, slice identities, strict JSON, atomic pair, kill after
#     staging (forked child), staging/incomplete refusals.
#   * CPB: probe overhead (extra R0 decode + R1 image) timed separately.
# Changelog (v0.10, 2026-10-22; I1.7(b), SEQREF-MRI-V05LTI17 v0.2):
#   * F12: DC calibration/tolerance/provenance, zero-denominator,
#     non-finite and invalid-floor stops, AM1 tests 1 and 2 (+ negative),
#     four readouts on identical states with no extra decodes; host part
#     in a fresh child on 6 real validation slices (temp outputs only).
#   * F22Z: ZF NMSE_u = 1, no NLL, EPOCH0_NONFINITE, summary, write-once
#     and gate-2a guard (temporary directories only); host child on 6
#     real slices. F23: holdout NMSE_u rule on a synthetic 199 population.
#   * CPB: DC calibration, R0+DC/R1+DC and ZF timing added.
# Changelog (v0.9, 2026-10-22; T7 verifier fix):
#   * Synthetic F10 facts now reproduce the real two-list situation
#     (payload hashed without V02G's SEMANTIC_EXCLUDED incl. timing and
#     r1_sampling_diagnostics; semantic_scope.excluded = generic list).
#   * New F10V regressions: altered included-key list, broadened exclusion
#     (re-hashed consistently) refused; historical procedure check on a
#     temporary git repository (identical PASS; drift, registry mismatch,
#     missing commit refused). Host: procedure taken from 6ffce4e5.
# Changelog (v0.8, 2026-10-22; I1.7(a) reviewer corrections):
#   * Header: the registered F10 exception to "never constructs
#     split='val'" stated explicitly.
#   * F11: refusal tests for unequal states/eligibility, epoch-0 pairing
#     across arms, data-loading and metric non-finite classification;
#     model mode restored; prefixes on the primary bank only.
#   * STB: each child records its actual backend; both must equal each
#     other and the registered thread/deterministic settings.
# Changelog (v0.7, 2026-10-21; I1.7(a), SEQREF-MRI-V05LTI17 v0.2):
#   * F10V: strict F10 input verification on a synthetic evidence set and
#     on the real inputs, plus 22 negative cases (each evidence category
#     altered in a copy -> typed refusal); input-verification PASS is kept
#     separate from F10 reproduction PASS.
#   * F10: synthetic -- V05LT R1 = V02G v0.2 _r1_measure bit-identical;
#     host -- D3-subset PM reproduction on the V02 checkpoints only, in a
#     fresh process under the V02G deterministic backend.
#   * F11 (schedule/reuse/readouts), F22R (readout cap flags), CPB (cost
#     probe, addition B), STB (stability pre-check, addition C; host only).
# Changelog (v0.6, 2026-10-21; I1.6(c), SEQREF-MRI-V05LTI16C v0.2):
#   * TFX (stub): trainer fixes 13-17 -- ten resume-state tamper cases,
#     file integrity, pre-first-commit restart, final commit without
#     run_end, code/env/data compatibility refusals, terminal recording.
#   * Real trainer in fresh spawned processes (registered tiny configs):
#     REP (addition B, run first), F09, F14 (actual NaN parameter / NaN
#     gradient; Adam tamper at resume), F13C, F21C (with vs without epoch-0
#     evaluation), F08R, KIL (addition A: SIGKILL at 3 boundaries), RNGC
#     (addition C). Predeclared permitted differences: PERMITTED_DIFFS.
#   * _commits drops resume_state_sha256 (embeds the ledger position).
# Changelog (v0.5, 2026-10-21; I1.6(b), SEQREF-MRI-V05LTI16B v0.1):
#   * Fix 9: energy comparison handles a zero reference explicitly.
#   * Fix 10: child cleanup escalates terminate -> kill and checks the exit
#     status before any result file is read; CHL tests the escalation.
#   * Fix 11: F07/F08 test rejection of invalid restored rule state.
#   * E0S: incompatible-environment rebuild refused (item 8).
#   * New: RET (F08 retention part), RES (stub resume / order / session
#     limit / taxonomy on the real trainer control flow), TRN (registered
#     tiny real-model smoke), GRD (trainer guards).
# Changelog (v0.4, 2026-10-21; I1.6(a), SEQREF-MRI-V05LTI16A v0.1):
#   * F07 (plateau) and F08 (selection part); MET (PSNR-cap flags = F22
#     part, aggregation = F06 part); VAL (shared validation metrics on the
#     real model; host cross-checks vs production NLL objective and numpy).
#   * F19 extended (I1.4 follow-ups): zero-energy slices; host temporary
#     `generate` run twice -> byte-identical, inputs/ untouched.
#   * Bounded child-process waits (CHILD_TIMEOUT) in F09CS, F21, E0S.
# Changelog (v0.3, 2026-10-21; I1.5, SEQREF-MRI-V05LTI15 v0.2):
#   * F03: mask-derived d_free vs the production encoded target length
#     (train_free_flow.targets_from_prepared) for every validation slice.
#   * F13 (part) and F21 (part) on the real model; E0S: epoch-0 save
#     crash points + explicit recovery. Data-order pairing, first-epoch
#     equivalence and full F13/F21 acceptance remain for I1.6.
#   * I1.3/I1.4 groups unchanged.
# Changelog (v0.2, 2026-10-21; I1.4, SEQREF-MRI-V05LTI14 v0.2):
#   * F19 added: eligibility rule boundary, energy vs an independent numpy
#     computation, per-slice records, null volumes, NMSE-only scope,
#     reproducibility, no-overwrite; host part cross-checks TINY's actual
#     _build_slice_states (energy, ratio, S_ref, threshold) on 32 slices.
#   * I1.3 groups unchanged.
# Changelog (NEW in v0.1, 2026-10-21):
#   * Introduced under I1.3 (V05LTI13 v0.2) with addition B (F09CS fault
#     injection) and revised addition C (SCAN inside the runner).
# =============================================================================
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import io
import json
import logging
import math
import multiprocessing
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from seqref_mri.scripts import v05lt_common as C
from seqref_mri.scripts import v05lt_evidence_check as EV

logger = logging.getLogger("seqref_mri.v05lt_fixtures")

__version__ = "0.45"
__abbr__ = "SEQREF-MRI-V05LTF"

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent.parent


CHILD_TIMEOUT_S = {"fork": 300, "spawn": 3600}


def join_bounded(p, kind: str, timeout: float | None = None,
                 grace: float = 30.0) -> int:
    """Wait for a child at most `timeout` (default CHILD_TIMEOUT_S[kind])
    seconds. A hung child is sent SIGTERM; if still alive after `grace`
    seconds, SIGKILL; if still alive after that, CHILD_UNKILLABLE. Any
    timeout fails the fixture (CHILD_TIMEOUT) -- never blocks forever.
    Callers must check the returned exit code before reading results."""
    limit = CHILD_TIMEOUT_S[kind] if timeout is None else timeout
    p.join(limit)
    if p.is_alive():
        p.terminate()
        p.join(grace)
        how = "terminated"
        if p.is_alive():
            p.kill()
            p.join(grace)
            how = "killed"
            if p.is_alive():
                C.fail("CHILD_UNKILLABLE", f"child pid {p.pid} survived kill")
        C.fail("CHILD_TIMEOUT", f"child pid {p.pid} exceeded {limit} s and "
               f"was {how} (exit {p.exitcode})")
    return p.exitcode


class FixtureFailure(Exception):
    def __init__(self, code: str, label: str):
        super().__init__(f"{code}: {label}")
        self.code = code


class Ctx:
    def __init__(self, host: bool, paths: C.Paths, data_root: Path):
        self.host = host
        self.paths = paths
        self.data_root = data_root
        self.phase = "synthetic"
        self.n = {"synthetic": 0, "host": 0}
        self.records: list = []      # (phase, label, detail) for EXEC

    def ok(self, cond: bool, label: str, code: str = "CHECK_FAILED",
           detail: str = "") -> None:
        if not cond:
            logger.error("CHECK FAIL [%s] %s %s", self.phase, label, detail)
            raise FixtureFailure(code, label)
        self.n[self.phase] += 1
        logger.info("ok [%s] %s %s", self.phase, label, detail)

    def raises(self, code: str, fn, label: str) -> None:
        try:
            fn()
        except C.V05LTError as exc:
            self.ok(exc.code == code, label, "WRONG_ERROR_CODE",
                    f"(expected {code}, got {exc.code})")
            return
        self.ok(False, label, "NO_ERROR_RAISED", f"(expected {code})")

    def record(self, label: str, value) -> None:
        self.records.append((self.phase, label, value))
        logger.info("RECORD [%s] %s = %s", self.phase, label, value)


def _raises_typed(cx: Ctx, code: str, fn, label: str) -> None:
    """cx.raises, but a raw (non-V05LTError) exception is recorded as a
    failed check (RAW_EXCEPTION) instead of ending the group."""
    def wrapped():
        try:
            fn()
        except C.V05LTError:
            raise
        except Exception as exc:  # noqa: BLE001 -- the defect under test
            logger.error("raw exception (%s): %s: %s", label,
                         type(exc).__name__, exc)
            raise C.V05LTError("RAW_EXCEPTION",
                               f"{type(exc).__name__}: {exc}") from exc
    cx.raises(code, wrapped, label)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------
def write_volume(path: Path, n_slices: int, seed: int) -> None:
    import h5py
    rng = np.random.default_rng(seed)  # fixture data only
    k = (rng.standard_normal((n_slices, 100, 100))
         + 1j * rng.standard_normal((n_slices, 100, 100))).astype(np.complex64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.create_dataset("kspace", data=k)
        h.create_dataset("reconstruction_esc",
                         data=np.ones((n_slices, 96, 96), np.float32))
        h.attrs["max"] = 1.0


def synth_split(tmp: Path, vols: dict, train: list, val: list) -> tuple:
    """-> (data_root, manifest_path, manifest_sha, expect)."""
    root = tmp / "data"
    for i, (vid, n) in enumerate(sorted(vols.items())):
        write_volume(root / "singlecoil_train" / vid, n, 1000 + i)
    tr, va = sorted(train), sorted(val)

    def rows(ids):
        return [{"volume_id": v, "file": f"singlecoil_train/{v}",
                 "n_slices": vols[v]} for v in ids]
    doc = {"schema": C.SPLIT_SCHEMA, "train": rows(tr), "validation": rows(va),
           "train_ids_sha256": C._ids_sha(tr),
           "validation_ids_sha256": C._ids_sha(va), "holdout_overlap": 0}
    mp = tmp / "split.json"
    mp.write_bytes(C.canonical_json(doc))
    expect = C.SplitExpect(len(tr), len(va), 1,
                           sum(vols[v] for v in tr), sum(vols[v] for v in va),
                           C._ids_sha(tr), C._ids_sha(va))
    return root, mp, C.sha256_file(mp), expect


SYN_VOLS = {"file9001.h5": 3, "file9002.h5": 4, "file9003.h5": 2,
            "file9004.h5": 3, "file9005.h5": 5}
SYN_TRAIN = ["file9001.h5", "file9002.h5", "file9004.h5"]
SYN_VAL = ["file9003.h5", "file9005.h5"]
SYN_HOLDOUT = frozenset({"file9999.h5"})


# ---------------------------------------------------------------------------
# F01 -- manifest identity, counts, holdout refusal
# ---------------------------------------------------------------------------
def f01(cx: Ctx) -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        root, mp, msha, ex = synth_split(tmp, SYN_VOLS, SYN_TRAIN, SYN_VAL)
        doc = C.load_split_manifest(mp, msha, ex)
        cx.ok(len(doc["train"]) == 3, "synthetic manifest loads")
        cx.raises("SPLIT_HASH_MISMATCH",
                  lambda: C.load_split_manifest(mp, "0" * 64, ex),
                  "wrong registered sha256 refused")
        bad = json.loads(mp.read_bytes())
        bad["validation"].append(bad["train"].pop())
        bad["validation"].sort(key=lambda r: r["volume_id"])
        bp = tmp / "bad.json"
        bp.write_bytes(C.canonical_json(bad))
        cx.raises("SPLIT_COUNTS",
                  lambda: C.load_split_manifest(bp, C.sha256_file(bp), ex),
                  "count change refused")
        tv, vv = C.build_split_views(root, doc, SYN_HOLDOUT, ex)
        cx.ok(len(tv) == ex.train_slices and len(vv) == ex.validation_slices,
              "view slice counts", detail=f"{len(tv)}/{len(vv)}")
        ent = [tv.entry(i) for i in range(len(tv))]
        cx.ok(ent == sorted(ent) and {Path(r).name for r, _ in ent}
              == set(SYN_TRAIN), "train view = dataset order, train IDs only")
        cx.ok({Path(vv.entry(i)[0]).name for i in range(len(vv))}
              == set(SYN_VAL), "validation view = validation IDs only")
        cx.raises("HOLDOUT_CONTACT",
                  lambda: C.make_dataset(root, "val", "eval"),
                  "split='val' refused before any dataset is built")
        cx.raises("HOLDOUT_CONTACT",
                  lambda: C.build_split_views(root, doc,
                                              frozenset({"file9003.h5"}), ex),
                  "manifest volume that is a holdout ID refused")
        write_volume(root / "singlecoil_train" / "file9999.h5", 2, 7)
        cx.raises("HOLDOUT_CONTACT",
                  lambda: C.build_split_views(root, doc, SYN_HOLDOUT, ex),
                  "holdout volume present in train directory refused")
        (root / "singlecoil_train" / "file9999.h5").unlink()
        rp = json.loads(mp.read_bytes())
        rp["train"][0]["file"] = "other/" + rp["train"][0]["volume_id"]
        cx.raises("SPLIT_RELPATH_MISMATCH",
                  lambda: C.build_split_views(root, rp, SYN_HOLDOUT, ex),
                  "manifest relpath must equal dataset relpath (mask seeds)")
        foreign = C.SplitView(tv.base, [vv.positions[0]], tv.allowed, "train")
        foreign.set_training_epoch(1)
        cx.raises("SPLIT_VIEW_FOREIGN_ITEM", lambda: foreign[0],
                  "view refuses an item outside its volume set")
        cx.ok(tv.order_lines_sha256() == tv.order_lines_sha256(),
              "base-order hash reproducible")
    if not cx.host:
        return
    cx.phase = "host"
    p = cx.paths
    C.verify_registered_inputs(p.inputs, {
        k: C.REGISTERED_INPUTS[k] for k in ("v05lt_split_manifest.json",
                                            "v05lt_case_manifest.json")})
    cx.ok(True, "split/case manifests in inputs/ = registered sha256")
    doc = C.load_split_manifest(p.inputs / "v05lt_split_manifest.json",
                                C.REGISTERED_INPUTS["v05lt_split_manifest.json"])
    cx.ok(True, "registered counts 876/97 vols, 31,281/3,461 slices, list hashes")
    from seqref_mri.scripts import v05lt_make_manifests as MM
    sb, cb = MM.build_payloads(p.repo)
    cx.ok(sb == (p.inputs / "v05lt_split_manifest.json").read_bytes()
          and cb == (p.inputs / "v05lt_case_manifest.json").read_bytes(),
          "manifests regenerated from V02 sources byte-identical to inputs/")
    ho = C.holdout_ids(p.v02_manifests)
    cx.ok(len(ho) == 199, "holdout IDs loaded (identity only, no data)")
    tv, vv = C.build_split_views(cx.data_root, doc, ho)
    cx.ok(len(tv) == 31281 and len(vv) == 3461, "real split views",
          detail=f"{len(tv)}/{len(vv)}")
    cx.record("AM2-A base order L sha256", tv.order_lines_sha256())
    cx.record("validation order sha256", vv.order_lines_sha256())
    one = sorted(ho)[0]
    cx.raises("HOLDOUT_CONTACT",
              lambda: C.assert_no_holdout([one], ho, "fixture"),
              "deliberate holdout ID refused")
    cx.raises("HOLDOUT_CONTACT",
              lambda: C.make_dataset(cx.data_root, "val", "eval"),
              "split='val' refused (no dataset built)")
    foreign = C.SplitView(tv.base, [vv.positions[0]], tv.allowed, "train")
    foreign.set_training_epoch(1)
    cx.raises("SPLIT_VIEW_FOREIGN_ITEM", lambda: foreign[0],
              "train view refuses a validation-volume item")


# ---------------------------------------------------------------------------
# F02 -- mask conventions by delegation to the production dataset
# ---------------------------------------------------------------------------
def _mask_checks(cx: Ctx, tv, vv, val_sample: list, tr_sample: list) -> None:
    from seqref_mri.src.fastmri_data import (CELL_HW, EVAL_BASE_SEED,
                                             TRAIN_BASE_SEED,
                                             canonical_mask_seed,
                                             make_cartesian_mask)
    allv = 0
    for i in range(len(vv)):
        rel, s = vv.entry(i)
        path = vv.base.index[vv.positions[i]][0]
        allv += vv.base._mask_seed(path, s) == canonical_mask_seed(
            EVAL_BASE_SEED, rel, s)
    cx.ok(allv == len(vv), "every validation seed = EVAL convention, no epoch",
          detail=f"{allv}/{len(vv)}")
    first = {}
    for i in val_sample:
        it = vv[i]
        rel, s = vv.entry(i)
        sd = canonical_mask_seed(EVAL_BASE_SEED, rel, s)
        first[i] = it["mask"].numpy().copy()
        cx.ok(it["meta"]["mask_seed"] == sd and bool(
            (first[i] == make_cartesian_mask(CELL_HW, sd)).all()),
            f"validation mask = eval-convention mask ({rel}|{s})")
    cx.raises("SET_EPOCH_ROLE", lambda: vv.set_training_epoch(1),
              "validation view refuses an epoch")
    cx.ok(tv.base.epoch is None, "fresh training view has no epoch set")
    try:
        tv[tr_sample[0]]
    except RuntimeError:
        cx.ok(True, "train sampling before set_epoch raises (production)")
    else:
        cx.ok(False, "train sampling before set_epoch must raise",
              "NO_ERROR_RAISED")
    masks = {}
    for e in (1, 2, 3):
        tv.set_training_epoch(e)
        cx.ok(tv.base.epoch == e - 1, f"epoch {e} -> production set_epoch({e - 1})")
        for i in tr_sample:
            it = tv[i]
            rel, s = tv.entry(i)
            sd = canonical_mask_seed(TRAIN_BASE_SEED, rel, s, epoch=e - 1)
            m = it["mask"].numpy()
            cx.ok(it["meta"]["mask_seed"] == sd and bool(
                (m == make_cartesian_mask(CELL_HW, sd)).all()),
                f"train mask e={e} = TRAIN convention epoch {e - 1} ({rel}|{s})")
            masks[(e, i)] = m.copy()
        for i in val_sample:
            cx.ok(bool((vv[i]["mask"].numpy() == first[i]).all()),
                  f"validation mask unchanged after train epoch {e}")
    diff = sum(not (masks[(1, i)] == masks[(2, i)]).all() for i in tr_sample)
    cx.record("train masks differing epoch 1 vs 2 (informational)",
              f"{diff}/{len(tr_sample)}")
    cx.raises("SET_EPOCH_INVALID", lambda: tv.set_training_epoch(0),
              "epoch 0 is not a training epoch")


def f02(cx: Ctx) -> None:
    with tempfile.TemporaryDirectory() as t:
        root, mp, msha, ex = synth_split(Path(t), SYN_VOLS, SYN_TRAIN, SYN_VAL)
        doc = C.load_split_manifest(mp, msha, ex)
        tv, vv = C.build_split_views(root, doc, SYN_HOLDOUT, ex)
        _mask_checks(cx, tv, vv, list(range(len(vv))), list(range(len(tv))))
    if not cx.host:
        return
    cx.phase = "host"
    p = cx.paths
    doc = C.load_split_manifest(p.inputs / "v05lt_split_manifest.json",
                                C.REGISTERED_INPUTS["v05lt_split_manifest.json"])
    tv, vv = C.build_split_views(cx.data_root, doc, C.holdout_ids(p.v02_manifests))
    case = json.loads((p.inputs / "v05lt_case_manifest.json").read_bytes())
    want = {(c["file"], c["slice_index"]) for c in case["cases"]}
    vs = [i for i in range(len(vv)) if vv.entry(i) in want]
    cx.ok(len(vs) == 6, "6 case slices located in validation view")
    vs += list(range(0, len(vv), 300))
    ts = list(range(0, len(tv), len(tv) // 20))[:20]
    _mask_checks(cx, tv, vv, vs, ts)


# ---------------------------------------------------------------------------
# F04 (I1.3 part) -- bank identity checks + static random-stream scan
# ---------------------------------------------------------------------------
RNG_PAT = re.compile(r"default_rng\(|PCG64\(|manual_seed\(|np\.random\.seed\("
                     r"|random\.seed\(|torch\.Generator\(")
# Production files allowed to construct generators, with the registered use.
RNG_ALLOWED = {
    "v05lt_make_banks.py": "PCG64(0/20261023/20261024) bank recipe (194 §3.1)",
    "v05lt_make_manifests.py": "default_rng(20261020/20261021) (194 §3.1)",
    "v05lt_fixtures.py": "fixture-only synthetic data",
    "v05lt_stub_train.py": "fixture-only stub model initialisation",
}
# Production files with one registered generator use (line-level allow).
RNG_ALLOWED_LINES = {
    "v05lt_train.py": re.compile(r"default_rng\(cfg\.seed\)|"
                                 r"numpy\.default_rng\(seed\)|"   # AM2-A
                                 r"^\s*random\.seed\(int\(seed\)\)$|"
                                 r"^\s*np\.random\.seed\(int\(seed\)\)$"),
    # v0.2 seed_global_rngs: run-seeded Python/NumPy globals (finding 14)
    "v05lt_stats.py": re.compile(r"rng = np\.random\.default_rng\(STATS_SEED\)"),
    # I1.8: the bootstrap stream default_rng(20261022) (194 §3.1)
}


def _f04_stats(cx: Ctx) -> None:
    """I1.8 statistics-stream part of F04 (synthetic)."""
    S = _S()
    src = (SCRIPTS_DIR / "v05lt_stats.py").read_text().splitlines()
    gen = [ln for ln in src if RNG_PAT.search(ln)]
    cx.ok(len(gen) == 1 and "default_rng(STATS_SEED)" in gen[0]
          and S.STATS_SEED == 20261022, "stats: exactly one generator, "
          "default_rng(20261022) (194 §3.1 bootstrap stream)", detail=str(gen))
    import random as _random
    _random.seed(11)
    np.random.seed(12)        # fixture: known global states to compare
    before = (_random.getstate(), np.random.get_state()[1].copy())
    res = _st_run(60)
    after = (_random.getstate(), np.random.get_state()[1])
    cx.ok(before[0] == after[0] and np.array_equal(before[1], after[1]),
          "analysis leaves the global Python / NumPy RNG states unchanged")
    cx.ok(len(res["stream"]) == 3 and all(
        len(r.get("volume_only_sha256", {})) == 3 for r in res["stream"]),
        "stream record: array sha256 per population (crossed seed, crossed "
        "volume, three volume-only)")
    orig = S.draw_indices

    def consuming(cfg):
        np.random.random()                       # fixture patch
        return orig(cfg)
    with _patched(S, "draw_indices", consuming):
        cx.raises("STATS_GLOBAL_RNG_CONSUMED", lambda: _st_run(60),
                  "global RNG consumption during the analysis refused")


def production_files() -> list:
    return sorted(p for p in SCRIPTS_DIR.glob("v05lt_*")
                  if p.suffix in (".py", ".sh") and p.is_file())


def f04(cx: Ctx) -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        bp = tmp / "b.npy"
        np.save(bp, arr)
        fs, asha = C.sha256_file(bp), C.sha256_bytes(arr.tobytes(order="C"))
        C.verify_bank_npy(bp, fs, asha, (3, 4))
        cx.ok(True, "bank npy: file + array hash + layout + finite")
        bad = arr.copy()
        bad[0, 0] = 99.0
        np.save(tmp / "x.npy", bad)
        cx.raises("BANK_HASH_MISMATCH",
                  lambda: C.verify_bank_npy(tmp / "x.npy", fs, asha, (3, 4)),
                  "altered bank refused (file hash)")
        cx.raises("BANK_ARRAY_HASH_MISMATCH",
                  lambda: C.verify_bank_npy(tmp / "x.npy",
                                            C.sha256_file(tmp / "x.npy"),
                                            asha, (3, 4)),
                  "altered bank refused (array hash vs registered)")
        nan = arr.copy()
        nan[1, 1] = np.nan
        np.save(tmp / "n.npy", nan)
        cx.raises("BANK_NON_FINITE",
                  lambda: C.verify_bank_npy(tmp / "n.npy",
                                            C.sha256_file(tmp / "n.npy"),
                                            asha, (3, 4)), "non-finite bank")
        np.save(tmp / "d.npy", arr.astype(np.float64))
        cx.raises("BANK_LAYOUT",
                  lambda: C.verify_bank_npy(tmp / "d.npy",
                                            C.sha256_file(tmp / "d.npy"),
                                            asha, (3, 4)), "float64 bank")
    hits = []
    for f in production_files():
        for n, line in enumerate(f.read_text().splitlines(), 1):
            allow = RNG_ALLOWED_LINES.get(f.name)
            if RNG_PAT.search(line) and f.name not in RNG_ALLOWED and not (
                    allow is not None and allow.search(line)):
                hits.append(f"{f.name}:{n}: {line.strip()}")
    cx.ok(not hits, "no unregistered generator construction in production "
          "v05lt files", "RNG_UNREGISTERED", "; ".join(hits))
    _f04_stats(cx)
    if not cx.host:
        return
    cx.phase = "host"
    shapes = C.verify_extra_banks(cx.paths.inputs)
    cx.ok(True, "banks B/C: file, array sha256, (128, 13824) float32, finite",
          detail=str(shapes))
    z = C.verify_zdiag()
    cx.ok(True, "Z_DIAG regenerated = registered bank + manifest sha256")
    cx.record("Z_DIAG", z)


# ---------------------------------------------------------------------------
# F05 -- source identity of NAMED reused modules (not a numerical test;
# R1 reproduction stays F10)
# ---------------------------------------------------------------------------
REUSED_MODULES = (
    ("seqref_mri/src/fastmri_data.py", "seqref_mri.src.fastmri_data", "3393fbee"),
    ("seqref_mri/scripts/v02_manifests.py", "seqref_mri.scripts.v02_manifests", "3393fbee"),
    ("seqref_mri/src/free_flow_runtime.py", "seqref_mri.src.free_flow_runtime", "3393fbee"),
    ("seqref_mri/scripts/train_free_flow.py", "seqref_mri.scripts.train_free_flow", "3393fbee"),
    ("seqref_mri/src/flows/v03_objective.py", "seqref_mri.src.flows.v03_objective", "3393fbee"),
    ("seqref_mri/src/metrics.py", "seqref_mri.src.metrics", "6ffce4e5"),
    ("seqref_mri/scripts/tiny_gate.py", "seqref_mri.scripts.tiny_gate", "6ffce4e5"),
    ("seqref_mri/tdiag/d3.py", "seqref_mri.tdiag.d3", "6ffce4e5"),
    ("seqref_mri/tdiag/estimators.py", "seqref_mri.tdiag.estimators", "6ffce4e5"),
)
REGISTERED_SCRIPTS = {  # 194 §3.1, 195 §2 (provenance scripts)
    "seqref_mri/scripts/v05lt_make_manifests.py":
        "4a898b2c5a35ddda2fa68b3d3852db7b9ed1be4bad9f823b04eee75bdaa12e6e",
    "seqref_mri/scripts/v05lt_make_banks.py":
        "984c41135bff821ec29363d16102120bd660c3fd2b08436e1bc38ca99c509782",
    "seqref_mri/scripts/v05lt_relocate_inputs.sh":
        "b965ed93d0e79021cb29b1cf1a1fc46e2a2df52f969e60eff285f744cab7c550",
}


def _git(repo: Path, *args) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                       text=True)
    if r.returncode != 0:
        C.fail("GIT_FAILED", f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def source_identity(repo: Path, rel: str, ref: str) -> dict:
    full = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    at_ref = _git(repo, "rev-parse", f"{full}:{rel}")
    at_head = _git(repo, "rev-parse", f"HEAD:{rel}")
    wt = _git(repo, "hash-object", rel)
    C.require(wt == at_ref, "SOURCE_IDENTITY_MISMATCH",
              f"{rel}: working tree {wt} != {ref} {at_ref}")
    C.require(at_head == at_ref, "SOURCE_HEAD_MISMATCH",
              f"{rel}: HEAD {at_head} != {ref} {at_ref}")
    return {"path": rel, "ref": full, "blob": at_ref}


def f05(cx: Ctx) -> None:
    with tempfile.TemporaryDirectory() as t:
        r = Path(t)
        _git(r, "init", "-q")
        _git(r, "config", "user.email", "fixture@local")
        _git(r, "config", "user.name", "fixture")
        (r / "m.py").write_text("X = 1\n")
        _git(r, "add", "m.py")
        _git(r, "commit", "-q", "-m", "ref")
        ref = _git(r, "rev-parse", "HEAD")[:8]
        cx.ok(source_identity(r, "m.py", ref)["blob"] != "",
              "identity check passes on an unchanged file")
        (r / "m.py").write_text("X = 2\n")
        cx.raises("SOURCE_IDENTITY_MISMATCH",
                  lambda: source_identity(r, "m.py", ref),
                  "modified working-tree file refused")
        _git(r, "commit", "-q", "-am", "edit")
        cx.raises("SOURCE_IDENTITY_MISMATCH",
                  lambda: source_identity(r, "m.py", ref),
                  "committed change refused")
    if not cx.host:
        return
    cx.phase = "host"
    repo = cx.paths.repo
    cx.record("HEAD", _git(repo, "rev-parse", "HEAD"))
    cx.record("branch", _git(repo, "rev-parse", "--abbrev-ref", "HEAD"))
    for rel, mod, ref in REUSED_MODULES:
        rec = source_identity(repo, rel, ref)
        m = importlib.import_module(mod)
        cx.ok(Path(m.__file__).resolve() == (repo / rel).resolve(),
              f"{rel} identical to {ref} and imported from the repo",
              "IMPORT_ORIGIN", detail=f"blob {rec['blob'][:12]} @ {rec['ref'][:12]}")
    for rel, sha in sorted(REGISTERED_SCRIPTS.items()):
        got = C.sha256_file(repo / rel)
        cx.ok(got == sha, f"{rel} = registered sha256", "SCRIPT_HASH_MISMATCH",
              detail=got[:12])


# ---------------------------------------------------------------------------
# F09CS -- crash safety of store + ledger (addition B; early part of F09)
# ---------------------------------------------------------------------------
def _store(run: Path) -> C.ArtifactStore:
    return C.ArtifactStore(run / "ckpt", C.Ledger(run / "ledger.jsonl"),
                           run / "_incomplete")


def _payload(tag: str) -> bytes:
    return hashlib.sha256(tag.encode()).digest() * 4096   # 128 KiB


def _commit1(run: Path) -> None:
    st = _store(run)
    st.stage("w_e0001.pt", lambda fh: fh.write(_payload("w1")))
    st.stage("resume_e0001.pt", lambda fh: fh.write(_payload("r1")))
    st.commit({"milestone_e1": "w_e0001.pt", "resume": "resume_e0001.pt"},
              {"epoch": 1, "best_psnr": {"epoch": 1, "value": 20.0}})


def _commit2(run: Path, hook=None) -> None:
    st = _store(run)
    st.recover()
    st.stage("w_e0002.pt", lambda fh: fh.write(_payload("w2")), hook)
    st.stage("resume_e0002.pt", lambda fh: fh.write(_payload("r2")), hook)
    st.commit({"milestone_e1": "w_e0001.pt", "best_psnr": "w_e0002.pt",
               "resume": "resume_e0002.pt"},
              {"epoch": 2, "best_psnr": {"epoch": 2, "value": 21.0}}, hook)


def _child(run: str, crash_at: int, trace: str) -> None:
    calls = []

    def hook(stage):
        calls.append(stage)
        if trace:
            Path(trace).write_text("\n".join(calls))
        if len(calls) == crash_at:
            os._exit(91)                      # simulated process kill
    _commit2(Path(run), hook)
    os._exit(0)


def _run_child(run: Path, crash_at: int, trace: str = "") -> int:
    ctx = multiprocessing.get_context("fork")
    p = ctx.Process(target=_child, args=(str(run), crash_at, trace))
    p.start()
    return join_bounded(p, "fork")


def _state_ok(cx: Ctx, run: Path, label: str) -> int:
    st = _store(run)
    body = st.recover()
    names = {v["name"] for v in body["retained"].values()}
    listing = {p.name for p in (run / "ckpt").iterdir()}
    cx.ok(listing == names | {n + ".sha256" for n in names},
          f"{label}: directory == committed set", detail=str(sorted(listing)))
    for v in body["retained"].values():
        tag = {"w_e0001.pt": "w1", "resume_e0001.pt": "r1",
               "w_e0002.pt": "w2", "resume_e0002.pt": "r2"}[v["name"]]
        cx.ok((run / "ckpt" / v["name"]).read_bytes() == _payload(tag),
              f"{label}: {v['name']} bytes intact")
    C.Ledger(run / "ledger.jsonl").records()
    return body["epoch"]


def f09cs(cx: Ctx) -> None:
    with tempfile.TemporaryDirectory() as t:
        base = Path(t)
        ref = base / "ref"
        ref.mkdir()
        _commit1(ref)
        trace = base / "trace.txt"
        code = _run_child(ref, 0, str(trace))
        stages = trace.read_text().splitlines()
        cx.ok(code == 0 and _state_ok(cx, ref, "clean run") == 2,
              "clean epoch-2 commit", detail=f"{len(stages)} hook points")
        commit_at = stages.index("ledger:after_fsync") + 1
        for k in range(1, len(stages) + 1):
            run = base / f"crash{k:02d}"
            run.mkdir()
            _commit1(run)
            ec = _run_child(run, k)
            cx.ok(ec == 91, f"crash at #{k} {stages[k - 1]} simulated")
            ep = _state_ok(cx, run, f"crash #{k}")
            cx.ok(ep == (2 if k >= commit_at else 1),
                  f"crash #{k}: recovered to epoch {ep} (commit point #{commit_at})")
            if ep == 1:
                _commit2(run)
                cx.ok(_state_ok(cx, run, f"crash #{k} redo") == 2,
                      f"crash #{k}: epoch-2 commit succeeds after recovery")
        # torn ledger tail
        run = base / "torn"
        run.mkdir()
        _commit1(run)
        line = C.canonical_json({"schema": C.LEDGER_SCHEMA, "seq": 1})
        with open(run / "ledger.jsonl", "ab") as fh:
            fh.write(line[: len(line) // 2])
        led = C.Ledger(run / "ledger.jsonl")
        cx.raises("LEDGER_TORN_TAIL_PENDING", led.records,
                  "torn tail detected before recovery")
        cx.raises("LEDGER_TORN_TAIL_PENDING",
                  lambda: led.append("x", {}), "append refused on torn tail")
        cx.ok(_state_ok(cx, run, "torn tail") == 1
              and any((run / "_incomplete").iterdir()),
              "torn tail quarantined; last committed epoch kept")
        # corrupt complete line: never auto-repaired
        run = base / "corrupt"
        run.mkdir()
        _commit1(run)
        good = (run / "ledger.jsonl").read_bytes()
        for pos in (12, len(good) // 2, len(good) - 3):
            data = bytearray(good)
            data[pos] ^= 0x01
            (run / "ledger.jsonl").write_bytes(bytes(data))
            cx.raises("LEDGER_CORRUPT", lambda: _store(run).recover(),
                      f"byte {pos} of committed line altered -> LEDGER_CORRUPT")
            cx.ok((run / "ledger.jsonl").read_bytes() == bytes(data),
                  "corrupt ledger not modified by recovery (no auto-repair)")
        # committed artefact problems
        for case, act, code in (
                ("tamper", lambda d: (d / "w_e0001.pt").write_bytes(b"x"),
                 "STORE_HASH_MISMATCH"),
                ("missing", lambda d: (d / "w_e0001.pt").unlink(),
                 "STORE_COMMITTED_MISSING"),
                ("sidecar", lambda d: (d / "w_e0001.pt.sha256").write_text(
                    "0" * 64 + "  w_e0001.pt\n"), "STORE_SIDECAR_MISMATCH")):
            run = base / case
            run.mkdir()
            _commit1(run)
            act(run / "ckpt")
            cx.raises(code, lambda: _store(run).recover(),
                      f"committed artefact {case} -> {code}")
        # protocol refusals
        run = base / "proto"
        run.mkdir()
        _commit1(run)
        st = _store(run)
        cx.raises("ARTIFACT_EXISTS",
                  lambda: st.stage("w_e0001.pt", lambda fh: fh.write(b"y")),
                  "no overwrite of an existing artefact")
        st.stage("extra.pt", lambda fh: fh.write(b"e"))
        cx.raises("STORE_STAGED_UNUSED",
                  lambda: st.commit({"r": "resume_e0001.pt"}, {"epoch": 2}),
                  "staged artefact must be retained or nothing committed")
        cx.raises("STORE_UNKNOWN_ARTIFACT",
                  lambda: st.commit({"r": "nope.pt", "e": "extra.pt"},
                                    {"epoch": 2}), "unknown artefact refused")
        (run / "ckpt" / "stray.bin").write_bytes(b"s")
        cx.raises("STORE_DIR_UNEXPECTED",
                  lambda: _store(run).commit({}, {"epoch": 2}),
                  "unexpected file in store blocks commit")
        _store(run).recover()
        cx.ok(not (run / "ckpt" / "stray.bin").exists()
              and not (run / "ckpt" / "extra.pt").exists(),
              "recovery quarantines uncommitted files")
    cx.record("F09CS limit", "process-kill simulation only; power loss and "
              "page-cache loss are not simulated")


# ---------------------------------------------------------------------------
# U13 -- model-state hash utility (194 AM3-G1); F13 itself is I1.5
# ---------------------------------------------------------------------------
def u13(cx: Ctx) -> None:
    import torch
    torch.manual_seed(0)  # fixture-only synthetic weights
    m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.BatchNorm1d(3))
    h0 = C.model_state_sha256(m)
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    b1 = buf.getvalue()
    buf.seek(0)
    sd = torch.load(buf, weights_only=True)
    cx.ok(C.model_state_sha256(sd) == h0, "re-serialised state: same hash")
    buf2 = io.BytesIO()
    torch.save(dict(reversed(list(sd.items()))), buf2)
    cx.ok(buf2.getvalue() != b1, "different serialisation bytes (file hash differs)")
    cx.ok(C.model_state_sha256(dict(reversed(list(sd.items())))) == h0,
          "key insertion order does not change the hash")
    sd2 = {k: v.clone() for k, v in sd.items()}
    sd2["0.weight"][0, 0] += 1e-7
    cx.ok(C.model_state_sha256(sd2) != h0, "one-element change detected")
    sd3 = dict(sd)
    sd3["0.weight"] = sd["0.weight"].double()
    cx.ok(C.model_state_sha256(sd3) != h0, "dtype change detected")
    sd4 = dict(sd)
    sd4["0.bias"] = sd["0.bias"].reshape(3, 1)
    cx.ok(C.model_state_sha256(sd4) != h0, "shape change detected")


# ---------------------------------------------------------------------------
# F18 -- evidence checker (anchored manifests)
# ---------------------------------------------------------------------------
def f18(cx: Ctx) -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        (root / "a.bin").write_bytes(b"alpha")
        (root / "b.bin").write_bytes(b"beta")
        anchors = {"a.bin": C.sha256_bytes(b"alpha")}
        man = root / "m.json"
        msha = EV.make_manifest(man, root, [("a.bin", "registered:a.bin"),
                                            ("b.bin", "approved_manifest")],
                                anchors)
        cx.ok(len(EV.check_manifest(man, msha, root, anchors)) == 2,
              "approved, anchored manifest passes")
        cx.raises("EVIDENCE_MANIFEST_NOT_APPROVED",
                  lambda: EV.check_manifest(man, "1" * 64, root, anchors),
                  "manifest hash must equal the approved value")
        (root / "b.bin").write_bytes(b"BETA")
        cx.raises("EVIDENCE_HASH_MISMATCH",
                  lambda: EV.check_manifest(man, msha, root, anchors),
                  "changed file refused")
        (root / "b.bin").unlink()
        cx.raises("EVIDENCE_MISSING",
                  lambda: EV.check_manifest(man, msha, root, anchors),
                  "missing file refused")
        (root / "a.bin").write_bytes(b"wrong")
        cx.raises("EVIDENCE_ANCHOR_MISMATCH",
                  lambda: EV.make_manifest(root / "m2.json", root,
                                           [("a.bin", "registered:a.bin")],
                                           anchors),
                  "manifest of a wrong registered input cannot be built")
        forged = {"schema": EV.MANIFEST_SCHEMA, "entries": [
            {"path": "a.bin", "sha256": C.sha256_bytes(b"wrong"),
             "anchor": "registered:a.bin"}]}
        fp = root / "forged.json"
        fp.write_bytes(C.canonical_json(forged))
        cx.raises("EVIDENCE_ANCHOR_MISMATCH",
                  lambda: EV.check_manifest(fp, C.sha256_file(fp), root, anchors),
                  "file and manifest agree but differ from registered -> refused")
        for bad, code in (({"path": "/etc/x", "sha256": "0", "anchor":
                            "approved_manifest"}, "EVIDENCE_PATH_INVALID"),
                          ({"path": "../x", "sha256": "0", "anchor":
                            "approved_manifest"}, "EVIDENCE_PATH_INVALID"),
                          ({"path": "a.bin", "sha256": "0", "anchor": ""},
                           "EVIDENCE_ANCHOR_INVALID")):
            bp = root / "bad.json"
            bp.write_bytes(C.canonical_json({"schema": EV.MANIFEST_SCHEMA,
                                             "entries": [bad]}))
            cx.raises(code, lambda: EV.check_manifest(
                bp, C.sha256_file(bp), root, anchors), f"refused: {bad}")
            bp.unlink()
        dup = {"schema": EV.MANIFEST_SCHEMA, "entries": [
            {"path": "a.bin", "sha256": C.sha256_bytes(b"wrong"),
             "anchor": "approved_manifest"}] * 2}
        dp = root / "dup.json"
        dp.write_bytes(C.canonical_json(dup))
        cx.raises("EVIDENCE_DUPLICATE", lambda: EV.check_manifest(
            dp, C.sha256_file(dp), root, anchors), "duplicate entry refused")
    if not cx.host:
        return
    cx.phase = "host"
    rows = EV.check_inputs(cx.paths, zdiag=True)
    cx.ok(len(rows) == 6 + 2 + 2 + 1 + 1, "evidence checker `inputs` on real "
          "inputs: all registered anchors", detail=f"{len(rows)} checks")


# ---------------------------------------------------------------------------
# F20 -- input relocation (AM2-D)
# ---------------------------------------------------------------------------
RELOC_SCRIPT_SHA = REGISTERED_SCRIPTS["seqref_mri/scripts/v05lt_relocate_inputs.sh"]


def check_relocation(paths: C.Paths, registry: dict, reloc_sha: str) -> int:
    inp = paths.inputs
    origin = {"manifests": paths.originals_manifests,
              "banks": paths.originals_banks}
    names = set(registry) | set(C.UNREGISTERED_INPUTS)
    present = {p.name for p in inp.iterdir()}
    C.require(names <= present, "RELOC_MISSING", str(sorted(names - present)))
    extra = present - names - set(C.INPUTS_LATER)
    C.require(not extra, "RELOC_UNEXPECTED_FILE", str(sorted(extra)))
    for n in sorted(names):
        cp, og = inp / n, origin[C.INPUT_ORIGIN[n]] / n
        C.require(og.is_file(), "RELOC_ORIGINAL_GONE", str(og))
        a, b = C.sha256_file(cp), C.sha256_file(og)
        C.require(a == b, "RELOC_COPY_MISMATCH", n)
        if n in registry:
            C.require(a == registry[n], "RELOC_REGISTERED_MISMATCH", n)
        C.require(cp.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP
                                       | stat.S_IWOTH) == 0,
                  "RELOC_WRITABLE", n)
    log = paths.reloc_log.read_text()
    C.require(f"RESULT PASS files={len(names)}" in log, "RELOC_LOG_NOT_PASS",
              str(paths.reloc_log))
    C.require(f"script_sha256 {reloc_sha}" in log, "RELOC_SCRIPT_HASH",
              "relocation log does not carry the registered script sha256")
    return len(names)


def f20(cx: Ctx) -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        p = C.Paths(repo=tmp / "repo", originals_manifests=tmp / "om",
                    originals_banks=tmp / "ob")
        reg = {}
        for n in C.INPUT_ORIGIN:
            d = p.originals_manifests if C.INPUT_ORIGIN[n] == "manifests" \
                else p.originals_banks
            d.mkdir(parents=True, exist_ok=True)
            (d / n).write_bytes(n.encode())
            if n in C.REGISTERED_INPUTS:
                reg[n] = C.sha256_bytes(n.encode())
        p.inputs.mkdir(parents=True)
        for n in C.INPUT_ORIGIN:
            d = p.originals_manifests if C.INPUT_ORIGIN[n] == "manifests" \
                else p.originals_banks
            (p.inputs / n).write_bytes((d / n).read_bytes())
            os.chmod(p.inputs / n, 0o444)
        p.reloc_log.parent.mkdir(parents=True)
        p.reloc_log.write_text(f"script_sha256 {'a' * 64} x\nRESULT PASS files=9\n")
        cx.ok(check_relocation(p, reg, "a" * 64) == 9, "synthetic relocation passes")
        os.chmod(p.inputs / "v05lt_case_manifest.json", 0o644)
        cx.raises("RELOC_WRITABLE", lambda: check_relocation(p, reg, "a" * 64),
                  "writable input refused")
        (p.inputs / "v05lt_case_manifest.json").write_bytes(b"changed")
        cx.raises("RELOC_COPY_MISMATCH", lambda: check_relocation(p, reg, "a" * 64),
                  "copy differing from original refused")
        (p.inputs / "surprise.json").write_bytes(b"s")
        cx.raises("RELOC_UNEXPECTED_FILE",
                  lambda: check_relocation(p, reg, "a" * 64), "unexpected file")
    if not cx.host:
        return
    cx.phase = "host"
    n = check_relocation(cx.paths, C.REGISTERED_INPUTS, RELOC_SCRIPT_SHA)
    cx.ok(n == 9, "inputs/: 9 files = registered = originals; read-only; "
          "log PASS with registered script sha256")


# ---------------------------------------------------------------------------
# SCAN -- targeted static check (addition C, revised): blocking rules +
# advisory list for manual review. Patterns assembled so this file's own
# source does not self-trigger.
# ---------------------------------------------------------------------------
_T = ["TO" + "DO", "FIX" + "ME", "X" + "XX", "HA" + "CK"]
BLOCKING = (
    ("SWALLOWED_EXCEPT", re.compile(r"except[^\n:]*:\s*(pass|\.\.\.)\s*$", re.M)),
    ("BARE_EXCEPT", re.compile(r"^\s*except\s*:", re.M)),
    ("WORK_MARKER", re.compile(r"\b(" + "|".join(_T) + r")\b")),
    ("NOT_IMPLEMENTED", re.compile(r"raise\s+Not" + "Implemented" + "Error")),
)
ADVISORY = re.compile(r"(?i)\b(mo" + "ck|dum" + "my|place" + "holder|fall"
                      + "back|st" + "ub)\b")
ADVISORY_SH = re.compile(r"\|\|\s*true|2>/dev/null")
HANDLED = re.compile(r"logger\.|log\.|\bfail\(|\braise\b|require\(|print\(|"
                     r"\.ok\(|\.raises\(|err |exit ")


def scan_file(path: Path) -> tuple:
    src = path.read_text()
    lines = src.splitlines()
    block, adv = [], []
    for code, pat in BLOCKING:
        for m in pat.finditer(src):
            block.append(f"{code} {path.name}:{src.count(chr(10), 0, m.start()) + 1}")
    if path.suffix == ".py":
        for i, line in enumerate(lines):
            if re.match(r"\s*except\b", line):
                window = "\n".join(lines[i: i + 5])
                if not HANDLED.search(window):
                    block.append(f"SILENT_EXCEPT {path.name}:{i + 1}")
    exempt = path.name in ("v05lt_fixtures.py", "v05lt_stub_train.py")
    for i, line in enumerate(lines, 1):
        if not exempt and ADVISORY.search(line):
            adv.append(f"{path.name}:{i}: {line.strip()[:100]}")
        if path.suffix == ".sh" and ADVISORY_SH.search(line):
            adv.append(f"{path.name}:{i}: {line.strip()[:100]}")
    return block, adv


def scan(cx: Ctx) -> None:
    with tempfile.TemporaryDirectory() as t:
        bad = Path(t) / "v05lt_bad.py"
        bad.write_text("try:\n    x = 1\nexcept Exception:\n    pass\n"
                       f"# {_T[0]} later\nexcept:\n    y = 2\n")
        b, _ = scan_file(bad)
        kinds = {x.split()[0] for x in b}
        cx.ok({"SWALLOWED_EXCEPT", "WORK_MARKER", "BARE_EXCEPT",
               "SILENT_EXCEPT"} <= kinds, "scanner detects seeded patterns",
              detail=str(sorted(kinds)))
    files = production_files()
    block, adv = [], []
    for f in files:
        b, a = scan_file(f)
        block += b
        adv += a
    for a in adv:
        logger.info("SCAN advisory (manual review): %s", a)
    cx.record("SCAN files", [f.name for f in files])
    cx.record("SCAN advisory count", len(adv))
    cx.ok(not block, f"no blocking pattern in {len(files)} v05lt files",
          "SCAN_BLOCKING", "; ".join(block))


# ---------------------------------------------------------------------------
# F19 -- NMSE eligibility (194 AM2-C; addition A = TINY cross-check)
# ---------------------------------------------------------------------------
P4_REL = "seqref_mri/results/_diag/p4/scaling_statistics.json"


def energy_agrees(prod: float, ref: float, rel: float = 1e-5) -> bool:
    """Fix 9: a zero reference requires an exactly zero production energy;
    otherwise a relative-error check."""
    if ref == 0.0:
        return prod == 0.0
    return abs(prod - ref) / abs(ref) < rel


def _numpy_energy(item) -> float:
    """Independent float64 numpy energy of the unmeasured coefficients of
    fft2c(x_true / amax) (the zero-filled input has none there)."""
    a = float(item["meta"]["file_attr_max"])
    x = item["x_true"].numpy().astype(np.float64)
    y = item["y"].numpy().astype(np.complex128)
    m = item["mask"].numpy().astype(bool)

    def f2(z):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(z), norm="ortho"))

    def if2(k):
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k), norm="ortho"))
    dx = (x[0] + 1j * x[1]) / a - if2(y) / a
    return float(np.sum(np.abs(f2(dx)[:, ~m]) ** 2))


def _syn_rows():
    """volume A: 2 eligible; B: none eligible (one exactly at threshold);
    C: 1 of 2 eligible."""
    t = 1e-10
    spec = [("A.h5", 0, 5e-10), ("A.h5", 1, 2e-10), ("B.h5", 0, t),
            ("B.h5", 1, 0.0), ("C.h5", 0, 3e-10), ("C.h5", 1, 1e-11)]
    from seqref_mri.scripts import v05lt_eligibility as E
    return [{"volume_id": v, "file": f"singlecoil_train/{v}", "slice_index": s,
             "energy": r, "ratio": r, "eligible": E.is_eligible(r)}
            for v, s, r in spec]


def f19(cx: Ctx) -> None:
    from seqref_mri.scripts import v05lt_eligibility as E
    t = E.R_FREE_MIN
    cx.ok(not E.is_eligible(t) and E.is_eligible(float(np.nextafter(t, 1.0)))
          and not E.is_eligible(float(np.nextafter(t, 0.0)))
          and not E.is_eligible(0.0),
          "boundary: below/equal ineligible, next float above eligible")
    for bad in (float("nan"), float("inf"), -1.0):
        cx.raises("ELIG_RATIO_INVALID", lambda b=bad: E.is_eligible(b),
                  f"invalid ratio {bad} refused")
    rows = _syn_rows()
    split = {"validation": [{"volume_id": v, "n_slices": 2}
                            for v in ("A.h5", "B.h5", "C.h5")]}
    case = {"cases": [{"file": "singlecoil_train/C.h5", "slice_index": 1,
                       "case": 0}]}
    doc = E.build_manifest(rows, split, case, {"p": 1})
    sa = doc["summary"]["all_slices"]
    cx.ok(sa["excluded"] == 3 and sa["volumes_without_eligible_slice"]
          == ["B.h5"] and sa["nmse_population_volumes"] == 2,
          "summary: excluded slices, null volume, reduced NMSE population")
    cx.ok(all({"volume_id", "file", "slice_index", "energy", "ratio",
               "eligible", "midpoint", "case"} <= set(r) for r in doc["slices"]),
          "every slice recorded explicitly")
    cx.ok([r["midpoint"] for r in doc["slices"]] == [True, False] * 3
          and doc["summary"]["cases"]["excluded"] == 1,
          "midpoint ((n-1)//2) and case flags")
    ent = [(r["volume_id"], r["file"], r["slice_index"]) for r in rows]
    el = {(r["file"], r["slice_index"]): r["eligible"] for r in rows}
    nm = {(r["file"], r["slice_index"]): 0.1 * (i + 1)
          for i, r in enumerate(rows)}
    vm = E.volume_means(nm, ent, el)
    cx.ok(vm["B.h5"] is None and abs(vm["C.h5"] - 0.5) < 1e-12
          and abs(vm["A.h5"] - 0.15) < 1e-12, "NMSE: eligible-only means, "
          "null (not 0) for the volume without eligible slices")
    cx.ok(E.nmse_population(vm) == ["A.h5", "C.h5"], "NMSE population fixed")
    ps = E.volume_means(nm, ent)
    cx.ok(ps["B.h5"] is not None and abs(ps["C.h5"] - 0.55) < 1e-12,
          "PSNR/SSIM/NLL path keeps every slice (TINY policy not carried over)")
    b1 = C.canonical_json(E.build_manifest(rows, split, case, {"p": 1}))
    cx.ok(b1 == C.canonical_json(doc), "manifest bytes reproducible")
    with tempfile.TemporaryDirectory() as tdir:
        tmp = Path(tdir)
        out = tmp / "elig.json"
        C.write_durable_new(out, b1)
        cx.raises("ARTIFACT_EXISTS", lambda: C.write_durable_new(out, b1),
                  "existing manifest never overwritten")
        p0 = {"schema": C.P0S_SCHEMA, "verdict": "PASS",
              "s_ref": {"value": 2.5, "valid_for_downstream": True}}
        pp = tmp / "p0s.json"
        pp.write_bytes(C.canonical_json(p0))
        ps_ = C.sha256_file(pp)
        cx.ok(C.verify_p0s(pp, ps_, 2.5) == 2.5, "S_ref read from pinned P0S")
        cx.raises("P0S_VALUE_MISMATCH", lambda: C.verify_p0s(pp, ps_, 2.6),
                  "S_ref value must equal the registered value")
        root, mp, msha, ex = synth_split(tmp, SYN_VOLS, SYN_TRAIN, SYN_VAL)
        sdoc = C.load_split_manifest(mp, msha, ex)
        _, vv = C.build_split_views(root, sdoc, SYN_HOLDOUT, ex)
        recs = E.compute_records(vv, 1.0)
        agree = [energy_agrees(r["energy"], _numpy_energy(vv[i]))
                 for i, r in enumerate(recs)]
        cx.ok(len(recs) == len(vv) and all(agree),
              "production energy = independent numpy energy (synthetic)")
        cx.ok(energy_agrees(0.0, 0.0) and not energy_agrees(1e-30, 0.0)
              and energy_agrees(1.0, 1.0 + 1e-7) and not energy_agrees(1.0, 2.0),
              "comparison: zero reference needs exact zero; else relative")
        cx.ok([(r["file"], r["slice_index"]) for r in recs]
              == [vv.entry(i) for i in range(len(vv))], "rows in view order")
    with tempfile.TemporaryDirectory() as tdir:     # zero-energy volume
        tmp = Path(tdir)
        vols = dict(SYN_VOLS, **{"file9006.h5": 2})
        root, mp, msha, ex = synth_split(tmp, vols, SYN_TRAIN,
                                         SYN_VAL + ["file9006.h5"])
        import h5py
        with h5py.File(root / "singlecoil_train" / "file9006.h5", "r+") as h:
            h["kspace"][...] = 0
        sdoc = C.load_split_manifest(mp, msha, ex)
        _, vv = C.build_split_views(root, sdoc, SYN_HOLDOUT, ex)
        recs = E.compute_records(vv, 1.0)
        z = [r for r in recs if r["volume_id"] == "file9006.h5"]
        zi = [i for i in range(len(vv)) if vv.entry(i)[0].endswith("file9006.h5")]
        cx.ok(all(energy_agrees(recs[i]["energy"], _numpy_energy(vv[i]))
                  for i in zi) and all(_numpy_energy(vv[i]) == 0.0 for i in zi),
              "zero-energy slices: numpy reference exactly 0 and production 0")
        cx.ok(len(z) == 2 and all(r["energy"] == 0.0 and r["zero_energy"]
                                  and not r["eligible"] for r in z)
              and not any(r["zero_energy"] for r in recs if r not in z),
              "zero energy: valid slice, flagged, ineligible, no error")
        split = {"validation": sdoc["validation"]}
        doc = E.build_manifest(recs, split, {"cases": []}, {"p": 1})
        sa = doc["summary"]["all_slices"]
        cx.ok(sa["zero_energy"] == 2 and "file9006.h5"
              in sa["volumes_without_eligible_slice"],
              "zero-energy slices counted; volume NMSE null")
    if not cx.host:
        return
    cx.phase = "host"
    from seqref_mri.scripts import tiny_gate as tg
    from seqref_mri.scripts.train_base import _collate
    from seqref_mri.src import free_flow_runtime as ffr
    p = cx.paths
    s_ref = C.verify_p0s(p.p0s)
    cx.ok(tg._s_ref_from_p0s(str(p.p0s)) == s_ref and tg.R_FREE_MIN == t
          and s_ref ** 2 == 244.20448939950668,
          "S_ref, S_ref^2 and threshold identical to TINY's")
    sdoc = C.load_split_manifest(p.inputs / "v05lt_split_manifest.json",
                                 C.REGISTERED_INPUTS["v05lt_split_manifest.json"])
    _, vv = C.build_split_views(cx.data_root, sdoc,
                                C.holdout_ids(p.v02_manifests))
    case = json.loads((p.inputs / "v05lt_case_manifest.json").read_bytes())
    want = [(c["file"], c["slice_index"]) for c in case["cases"]]
    pos = {vv.entry(i): i for i in range(len(vv))}
    idx = [pos[w] for w in want]
    idx += [i for i in range(0, len(vv), len(vv) // 26) if i not in idx][:26]
    recs = E.compute_records(vv, s_ref, idx)
    items = [vv[i] for i in idx]
    batch = _collate(items)
    sel = {"ordered_identities": [{"file": r["file"],
                                   "slice_index": r["slice_index"]}
                                  for r in recs]}
    p4 = ffr.load_p4s2_parent(str(p.repo / P4_REL))
    states = tg._build_slice_states(batch, sel, p4, s_ref)
    same_e = sum(a["energy"] == b["u_true_energy"] for a, b in zip(recs, states))
    same_r = sum(a["ratio"] == b["u_true_ratio"] for a, b in zip(recs, states))
    same_f = sum(a["eligible"] == (not b["excluded"])
                 for a, b in zip(recs, states))
    cx.ok(len(states) == len(recs) == 32 and same_e == same_r == same_f == 32,
          "TINY _build_slice_states: energy, ratio and flag bit-identical",
          detail=f"{same_e}/{same_r}/{same_f} of 32")
    cx.ok(all(energy_agrees(r["energy"], _numpy_energy(it))
              for r, it in zip(recs, items)),
          "independent numpy energy agrees on real slices (zero-safe)")
    before = {q.name: C.sha256_file(q) for q in sorted(p.inputs.iterdir())}
    with tempfile.TemporaryDirectory() as tdir:     # I1.4 follow-up 1
        o1, o2 = Path(tdir) / "a" / E.OUT_NAME, Path(tdir) / "b" / E.OUT_NAME
        o1.parent.mkdir()
        o2.parent.mkdir()
        h1 = E.generate(p, cx.data_root, o1)
        h2 = E.generate(p, cx.data_root, o2)
        gd = json.loads(o1.read_bytes())
        sa = gd["summary"]["all_slices"]
        pv = gd["provenance"]
        cx.ok(h1 == h2 and o1.read_bytes() == o2.read_bytes(),
              "temporary generate x2: byte-identical", detail=h1)
        cx.ok(gd["schema"] == E.SCHEMA and len(gd["slices"]) == 3461
              and sa["slices"] == 3461 and gd["summary"]["midpoint"]["slices"]
              == 97 and gd["summary"]["cases"]["slices"] == 6,
              "3,461 explicit rows; 97 midpoints; 6 cases")
        cx.ok(pv["split_manifest_sha256"] == C.REGISTERED_INPUTS[
            "v05lt_split_manifest.json"] and pv["p0s_sha256"] == C.P0S_SHA256
              and pv["s_ref"] == C.S_REF, "provenance pins = registered")
        cx.record("F19 temporary generate (not registered)", {
            "sha256": h1, "eligible": sa["eligible"], "excluded": sa["excluded"],
            "zero_energy": sa["zero_energy"],
            "null_volumes": len(sa["volumes_without_eligible_slice"]),
            "midpoint_excluded": gd["summary"]["midpoint"]["excluded"],
            "cases_excluded": gd["summary"]["cases"]["excluded"]})
    after = {q.name: C.sha256_file(q) for q in sorted(p.inputs.iterdir())}
    cx.ok(before == after, "inputs/ unchanged by the temporary generate runs")
    cx.record("F19 sample (6 cases + 26)", {
        "eligible": sum(r["eligible"] for r in recs),
        "min_ratio": min(r["ratio"] for r in recs),
        "max_ratio": max(r["ratio"] for r in recs)})


# ---------------------------------------------------------------------------
# F03 -- d_free: mask-derived dimension vs the production encoded vector
# ---------------------------------------------------------------------------
def _f03_slices(cx: Ctx, view, vecs_fn, label: str) -> int:
    from torch.utils.data import DataLoader
    from seqref_mri.scripts.train_base import _collate, _prepare
    from seqref_mri.scripts.train_free_flow import targets_from_prepared
    from seqref_mri.scripts.v02_train import derive_cmap_from_mask
    from seqref_mri.src import free_flow_runtime as ffr
    n = bad = 0
    for batch in DataLoader(view, batch_size=32, shuffle=False, num_workers=0,
                            collate_fn=_collate):
        prep = _prepare(batch, "cpu", test0=False)
        for j in range(len(batch["meta"])):
            m = batch["mask"][j].numpy().astype(bool)
            d_mask = 2 * int((~m).sum()) * int(m.shape[0])
            cmap = derive_cmap_from_mask(batch["mask"][j])
            one = {"x_norm": prep["x_norm"][j:j + 1],
                   "cond_in": prep["cond_in"][j:j + 1]}
            tgt = targets_from_prepared(one, cmap, vecs_fn(cmap))
            ok = (tuple(tgt.shape) == (1, d_mask) and d_mask
                  == ffr.FLOW_DIM_REAL and bool(np.isfinite(tgt.numpy()).all()))
            bad += not ok
            n += 1
    cx.ok(bad == 0 and n == len(view), f"{label}: d_free (mask) = encoded "
          f"target length = FLOW_DIM_REAL for every slice", "F03_DIM",
          detail=f"{n - bad}/{n}")
    return n


def f03(cx: Ctx) -> None:
    from seqref_mri.scripts.v02_train import derive_cmap_from_mask
    import torch

    def unit_vecs(cmap):   # fixture-only stand-in for P4 standardisation
        k = cmap.n_free_complex
        return {"mean_re": np.zeros(k), "scale_re": np.ones(k),
                "mean_im": np.zeros(k), "scale_im": np.ones(k),
                "n_free_complex": k}
    with tempfile.TemporaryDirectory() as t:
        root, mp, msha, ex = synth_split(Path(t), SYN_VOLS, SYN_TRAIN, SYN_VAL)
        _, vv = C.build_split_views(root, C.load_split_manifest(mp, msha, ex),
                                    SYN_HOLDOUT, ex)
        _f03_slices(cx, vv, unit_vecs, "synthetic (unit standardisation)")
    bad = torch.zeros(96, dtype=torch.bool)
    bad[:25] = True
    try:
        derive_cmap_from_mask(bad)
    except Exception as exc:  # production V02Error; logged by production
        cx.ok("MASK_ACQUIRED_COUNT_UNEXPECTED" in str(exc),
              "wrong measured-column count refused by production map")
    else:
        cx.ok(False, "wrong measured-column count must be refused",
              "NO_ERROR_RAISED")
    if not cx.host:
        return
    cx.phase = "host"
    from seqref_mri.src import free_flow_runtime as ffr
    p = cx.paths
    p4 = ffr.load_p4s2_parent(str(p.repo / P4_REL))
    doc = C.load_split_manifest(p.inputs / "v05lt_split_manifest.json",
                                C.REGISTERED_INPUTS["v05lt_split_manifest.json"])
    _, vv = C.build_split_views(cx.data_root, doc, C.holdout_ids(p.v02_manifests))
    n = _f03_slices(cx, vv, lambda cm: ffr.standardisation_vectors(
        cm, p4["location_index"]), "all validation slices (pinned P4)")
    cx.record("F03 slices checked", n)


# ---------------------------------------------------------------------------
# F13 (part) / F21 (part) / E0S -- epoch-0 identity, isolation, save safety
# ---------------------------------------------------------------------------
def _rng_snapshot():
    import random
    import torch
    return (torch.get_rng_state().clone(), np.random.get_state(),
            random.getstate())


def _rng_equal(a, b) -> bool:
    import torch
    return (torch.equal(a[0], b[0]) and a[2] == b[2] and a[1][0] == b[1][0]
            and np.array_equal(a[1][1], b[1][1]) and a[1][2:] == b[1][2:])


def f13(cx: Ctx) -> None:
    if cx.host:            # same real-model checks, host environment
        cx.phase = "host"
    import gc
    import torch
    from seqref_mri.scripts import v05lt_epoch0 as E0
    from seqref_mri.src import free_flow_runtime as ffr

    def built(seed):       # model-state hash of a fresh build; model freed
        m = ffr.build_model(init_seed=seed)
        h_ = C.model_state_sha256(m)
        del m
        gc.collect()
        return h_
    h = {s_: built(s_) for s_ in (2026, 2027, 2028)}
    cx.ok(built(2026) == h[2026],
          "same seed -> identical model-state hash (paired init)")
    cx.ok(len(set(h.values())) == 3, "seeds 2026/2027/2028 differ")
    sd = ffr.build_model(init_seed=2026).state_dict()
    for variant in (sd, dict(reversed(list(sd.items())))):
        b = io.BytesIO()
        torch.save(variant, b)
        b.seek(0)
        cx.ok(C.model_state_sha256(torch.load(b, weights_only=True)) == h[2026],
              "re-serialised state keeps the model-state hash (file hash "
              "not required to differ)")
        del b
        gc.collect()
    del sd, variant
    gc.collect()
    with tempfile.TemporaryDirectory() as t:
        out = E0.create(Path(t), 2026, REPO_ROOT)
        cx.ok(out["model_state_sha256"] == h[2026],
              "epoch-0 checkpoint carries the seed's model-state hash")
        f = Path(t) / "seed_2026" / "ckpt" / out["file"]
        lm = E0.load_epoch0(f, h[2026], 2026)
        cx.ok(not lm.training and C.model_state_sha256(lm) == h[2026],
              "load_epoch0: eval mode, verified state")
        del lm
        gc.collect()
        arm = ffr.build_model(init_seed=2026)
        E0.assert_matches_epoch0(arm, h[2026])
        cx.ok(True, "fresh arm build matches epoch-0 hash")
        with torch.no_grad():
            next(arm.parameters()).view(-1)[0] += 1e-6
        cx.raises("EPOCH0_ARM_MISMATCH",
                  lambda: E0.assert_matches_epoch0(arm, h[2026]),
                  "altered arm weights refused")
        del arm
        gc.collect()
        other = ffr.build_model(init_seed=2027)
        cx.raises("EPOCH0_ARM_MISMATCH",
                  lambda: E0.assert_matches_epoch0(other, h[2026]),
                  "other seed refused")
        del other
        gc.collect()
        cx.raises("EPOCH0_DIR_EXISTS",
                  lambda: E0.create(Path(t), 2026, REPO_ROOT),
                  "existing seed directory blocks a fresh start")
        cx.record("epoch-0 environment", json.loads(
            (Path(t) / "seed_2026" / "ledger.jsonl").read_text()
            .splitlines()[0])["body"]["env"])
    cx.record("F13 deferred to I1.6", "data-order pairing across arms; full "
              "F13 acceptance")


def _f21_child(path: str, msh: str, out: str) -> None:
    import torch
    from seqref_mri.scripts import v05lt_epoch0 as E0
    from seqref_mri.src.fastmri_data import make_cartesian_mask
    model = E0.load_epoch0(Path(path), msh, 2026)
    g = torch.Generator().manual_seed(7)      # local generator, fixture-only
    u = torch.randn(1, 13824, generator=g)
    cond = torch.randn(1, 2, 96, 96, generator=g)
    mask = torch.from_numpy(make_cartesian_mask(96, 11)).unsqueeze(0)
    with torch.no_grad():
        lp = model.log_prob_free(u, cond, mask)
    Path(out).write_text(json.dumps({
        "after": C.model_state_sha256(model), "finite": bool(
            torch.isfinite(lp).all()), "training": model.training}))
    os._exit(0)


def f21(cx: Ctx) -> None:
    if cx.host:            # same real-model checks, host environment
        cx.phase = "host"
    from seqref_mri.scripts import v05lt_epoch0 as E0
    from seqref_mri.src import free_flow_runtime as ffr
    with tempfile.TemporaryDirectory() as t:
        out = E0.create(Path(t), 2026, REPO_ROOT)
        f = Path(t) / "seed_2026" / "ckpt" / out["file"]
        msh = out["model_state_sha256"]
        m = ffr.build_model(init_seed=2026)       # construction seeds torch
        s0, h0 = _rng_snapshot(), C.model_state_sha256(m)
        res = Path(t) / "child.json"
        # spawn, not fork: a forked child of a process that already ran torch
        # intra-op threads can deadlock; spawn is a clean separate process.
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(target=_f21_child, args=(str(f), msh, str(res)))
        p.start()
        ec = join_bounded(p, "spawn")
        cx.ok(ec == 0 and res.is_file(), "evaluation child exited 0 before "
              "its result is read", "CHILD_FAILED", detail=f"exit {ec}")
        r = json.loads(res.read_text())
        cx.ok(r["after"] == msh and r["finite"]
              and not r["training"], "separate-process evaluation: eval mode, "
              "parameters and buffers unchanged by a forward pass")
        cx.ok(_rng_equal(s0, _rng_snapshot()) and C.model_state_sha256(m) == h0,
              "trainer-side RNG states and initial weights unchanged")
        del m
        import gc
        gc.collect()
        s1 = _rng_snapshot()
        E0.load_epoch0(f, msh, 2026)
        cx.ok(_rng_equal(s1, _rng_snapshot()),
              "in-process load_epoch0 restores torch RNG; numpy/python untouched")
    cx.record("F21 deferred to I1.6", "first-epoch results with vs without a "
              "preceding epoch-0 evaluation; full F21 acceptance")


def _e0_child(root: str, target: str) -> None:
    from seqref_mri.scripts import v05lt_epoch0 as E0

    def hook(stage):
        if stage == target:
            os._exit(91)
    E0.create(Path(root), 2026, REPO_ROOT, hook)
    os._exit(0)


def e0s(cx: Ctx) -> None:
    if cx.host:            # same real-model checks, host environment
        cx.phase = "host"
    from seqref_mri.scripts import v05lt_epoch0 as E0
    from seqref_mri.src import free_flow_runtime as ffr
    ref = C.model_state_sha256(ffr.build_model(init_seed=2026))
    import gc
    gc.collect()
    expect = {"ledger:after_fsync": "rebuilt_uncommitted",
              "stage:tmp_fsynced": "rebuilt_uncommitted",
              "stage:linked": "rebuilt_uncommitted",
              "commit:after_ledger": "verified_committed"}
    for target, how in expect.items():
        with tempfile.TemporaryDirectory() as t:
            ctx = multiprocessing.get_context("spawn")   # see f21
            p = ctx.Process(target=_e0_child, args=(t, target))
            p.start()
            join_bounded(p, "spawn")
            cx.ok(p.exitcode == 91, f"crash at {target} simulated")
            cx.raises("EPOCH0_DIR_EXISTS",
                      lambda: E0.create(Path(t), 2026, REPO_ROOT),
                      f"after crash at {target}: fresh start refused")
            out = E0.recover(Path(t), 2026, REPO_ROOT)
            cx.ok(out["recovery"] == how and out["model_state_sha256"] == ref,
                  f"after crash at {target}: explicit recovery -> {how}, "
                  "same model-state hash")
            led = (Path(t) / "seed_2026" / "ledger.jsonl").read_bytes()
            again = E0.recover(Path(t), 2026, REPO_ROOT)
            cx.ok(again["recovery"] == "already_complete" and led == (
                Path(t) / "seed_2026" / "ledger.jsonl").read_bytes(),
                "recovery of a complete save changes nothing")
    with tempfile.TemporaryDirectory() as t:       # item 8
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(target=_e0_child, args=(t, "stage:tmp_fsynced"))
        p.start()
        cx.ok(join_bounded(p, "spawn") == 91, "crash before commit simulated")
        bad_env = dict(E0.environment(), torch="0.0.0-other")
        led = (Path(t) / "seed_2026" / "ledger.jsonl").read_bytes()
        cx.raises("EPOCH0_RECOVERY_INCOMPATIBLE",
                  lambda: E0.recover(Path(t), 2026, REPO_ROOT, bad_env),
                  "rebuild refused when the environment differs from start")
        ck = [q.name for q in (Path(t) / "seed_2026" / "ckpt").iterdir()]
        cx.ok(not [n for n in ck if n.endswith(".pt")],
              "nothing built after the refused rebuild", detail=str(ck))
        out = E0.recover(Path(t), 2026, REPO_ROOT)
        cx.ok(out["recovery"] == "rebuilt_uncommitted" and led != b"",
              "compatible environment: explicit rebuild proceeds")
    with tempfile.TemporaryDirectory() as t:
        out = E0.create(Path(t), 2026, REPO_ROOT)
        f = Path(t) / "seed_2026" / "ckpt" / out["file"]
        os.chmod(f, 0o644)
        with open(f, "r+b") as fh:
            fh.seek(-10, 2)
            b = fh.read(1)
            fh.seek(-10, 2)
            fh.write(bytes([b[0] ^ 1]))
        cx.raises("STORE_HASH_MISMATCH",
                  lambda: E0.recover(Path(t), 2026, REPO_ROOT),
                  "tampered committed epoch-0 file refused, never rebuilt")
    cx.record("E0S limit", "process-kill simulation only")


# ---------------------------------------------------------------------------
# F07 -- plateau (stopping only); F08 (part) -- selection only
# ---------------------------------------------------------------------------
def _run_plateau(seq):
    from seqref_mri.scripts import v05lt_rules as R
    pl, out = R.Plateau(), None
    for e, (ps, nl) in enumerate(seq, 1):
        out = pl.update(e, ps, nl)
        if out != R.CONTINUE:
            return pl, e, out
    return pl, len(seq), out


def f07(cx: Ctx) -> None:
    from seqref_mri.scripts import v05lt_rules as R
    cx.raises("PLATEAU_EPOCH_INVALID", lambda: R.Plateau().update(0, 1.0, 1.0),
              "epoch 0 is never a check")
    pl = R.Plateau()
    pl.update(1, 20.0, 1.0)
    cx.ok(pl.ref == {"psnr": 20.0, "nll": 1.0} and pl.count ==
          {"psnr": 0, "nll": 0}, "epoch-1 initialisation")
    cx.raises("PLATEAU_EPOCH_ORDER", lambda: pl.update(3, 20.0, 1.0),
              "non-consecutive epoch refused")
    p2 = R.Plateau()
    p2.update(1, 0.0, 0.0)
    p2.update(2, 0.01, -0.001)       # exactly equal to tolerance (float64)
    cx.ok(0.01 - 0.0 == R.PSNR_TOL and 0.0 - (-0.001) == R.NLL_TOL
          and p2.count == {"psnr": 1, "nll": 1} and p2.ref["psnr"] == 0.0,
          "improvement equal to tolerance does not reset")
    p3 = R.Plateau()
    vals = [20.0, 20.004, 20.008, 20.0105]
    for e, v in enumerate(vals, 1):
        p3.update(e, v, 1.0)
    cx.ok(p3.ref["psnr"] == 20.0105 and p3.count["psnr"] == 0
          and p3.resets == [[4, "psnr"]],
          "sub-tolerance gains accumulate against the last meaningful best")
    seq = [(20.0 + 0.1 * e, 1.0 - 0.01 * e) for e in range(1, 21)] + \
        [(22.0, 0.80)] * 80
    _, e, out = _run_plateau(seq)
    cx.ok(out == R.STOPPED and e == 30, "both counters reach 10 -> stop "
          "exactly at epoch 30", detail=f"epoch {e}")
    seq = [(20.0 + 0.1 * e, 1.0) for e in range(1, 36)] + [(23.5, 1.0)] * 65
    _, e, out = _run_plateau(seq)
    cx.ok(out == R.STOPPED and e == 45, "one counter < 10 keeps training",
          detail=f"epoch {e}")
    seq = [(20.0 + 0.02 * e, 1.0 - 0.002 * e) for e in range(1, 101)]
    pl4, e, out = _run_plateau(seq)
    cx.ok(out == R.EXHAUSTED and e == 100, "still improving at 100 -> "
          "budget exhausted; plateau not established")
    cx.raises("PLATEAU_BEYOND_BUDGET", lambda: pl4.update(101, 30.0, 0.0),
              "no epoch beyond 100")
    a = R.Plateau()
    for e, v in enumerate(vals, 1):
        a.update(e, v, 1.0)
    b = R.Plateau.from_dict(json.loads(json.dumps(a.to_dict())))
    cx.ok(b.to_dict() == a.to_dict() and b.update(5, 20.5, 0.9)
          == a.update(5, 20.5, 0.9) and b.to_dict() == a.to_dict(),
          "plateau state round-trips exactly through JSON (resume)")
    cx.raises("RULE_VALUE_INVALID",
              lambda: R.Plateau().update(1, float("nan"), 1.0), "NaN refused")
    good = a.to_dict()
    for name, mut in (
            ("counter > epochs", lambda d: d["count"].__setitem__("psnr", 99)),
            ("negative counter", lambda d: d["count"].__setitem__("nll", -1)),
            ("non-int counter", lambda d: d["count"].__setitem__("nll", 1.0)),
            ("epoch > 100", lambda d: d.__setitem__("last_epoch", 101)),
            ("NaN reference", lambda d: d["ref"].__setitem__("nll", float("nan"))),
            ("reset after last epoch", lambda d: d["resets"].append([9, "psnr"])),
            ("extra key", lambda d: d.__setitem__("x", 1))):
        bad = json.loads(json.dumps(good))
        mut(bad)
        cx.raises("RULE_STATE_INVALID", lambda b=bad: R.Plateau.from_dict(b),
                  f"restored plateau refused: {name}")


def f08(cx: Ctx) -> None:
    from seqref_mri.scripts import v05lt_rules as R
    s = R.Selection()
    cx.raises("SELECTION_EPOCH_INVALID", lambda: s.update(0, 30.0, 0.5),
              "epoch 0 is never a selection candidate")
    s.update(1, 20.0, 1.0)
    s.update(2, 21.0, 0.9)
    ch = s.update(3, 21.0, 0.95)
    cx.ok(s.best_psnr == {"epoch": 2, "value": 21.0} and not ch["psnr"],
          "PSNR tie keeps the earliest epoch")
    s.update(4, 20.5, 0.8)
    cx.ok(s.best_nll == {"epoch": 4, "value": 0.8} and s.best_psnr["epoch"]
          == 2, "best NLL tracked independently of best PSNR")
    pl, sel = R.Plateau(), R.Selection()
    seq = [(20.0 + 0.1 * e, 1.0 - 0.01 * e) for e in range(1, 21)] + \
        [(22.0 + 0.005, 0.80)] * 80
    for e, (ps, nl) in enumerate(seq, 1):
        sel.update(e, ps, nl)
        if pl.update(e, ps, nl) != R.CONTINUE:
            break
    cx.ok(e == 30 and sel.best_psnr["epoch"] == 21,
          "stopping and selection independent: a sub-tolerance gain stops "
          "nothing but is selected", detail=f"stop {e}, best {sel.best_psnr}")
    t = R.Selection.from_dict(json.loads(json.dumps(sel.to_dict())), e)
    cx.ok(t.to_dict() == sel.to_dict(), "selection state round-trips (resume)")
    for name, d_, le in (
            ("best after last epoch", {"best_psnr": {"epoch": 31, "value": 1.0},
                                       "best_nll": {"epoch": 2, "value": 1.0}}, 30),
            ("epoch 0 record", {"best_psnr": {"epoch": 0, "value": 1.0},
                                "best_nll": {"epoch": 2, "value": 1.0}}, 30),
            ("NaN value", {"best_psnr": {"epoch": 2, "value": float("nan")},
                           "best_nll": {"epoch": 2, "value": 1.0}}, 30),
            ("missing record", {"best_psnr": None, "best_nll": None}, 5)):
        cx.raises("RULE_STATE_INVALID",
                  lambda d=d_, l=le: R.Selection.from_dict(d, l),
                  f"restored selection refused: {name}")
    cx.ok([e for e in range(0, 101) if R.is_milestone(e)]
          == [3, 10, 20, 30, 50, 75, 100], "milestones 3/10/20/30/50/75/100")
    cx.record("F08 deferred to I1.6(b)/(c)", "retained set incl. epoch 0, "
              "replace-then-delete order, peak file count")


# ---------------------------------------------------------------------------
# MET -- shared metric primitives: PSNR-cap flags (F22 part), aggregation
# (F06 part)
# ---------------------------------------------------------------------------
def met(cx: Ctx) -> None:
    import torch
    from seqref_mri.scripts import v05lt_metrics as M
    eps32 = float(torch.tensor(1e-12, dtype=torch.float32))
    for v, cap, zero in ((0.0, True, True), (eps32 / 4, True, False),
                         (eps32, True, False),
                         (float(np.nextafter(np.float32(eps32), np.float32(1))),
                          False, False), (1e-6, False, False)):
        f = M.cap_flags(torch.tensor(v, dtype=torch.float32))
        cx.ok(f["psnr_capped"] == cap and f["zero_error"] == zero,
              f"cap flags at MSE {v:.3e} (float32): capped={cap} zero={zero}")
    t = torch.full((96, 96), 0.5)
    r0 = M.psnr_record(t, t.clone())
    cx.ok(r0["psnr"] == 120.0 and r0["psnr_capped"] and r0["zero_error"],
          "MSE 0 -> production PSNR 120 dB, capped, zero_error (retained)")
    r1 = M.psnr_record(t + 3e-7, t)
    cx.ok(r1["psnr"] == 120.0 and r1["psnr_capped"] and not r1["zero_error"],
          "0 < MSE < floor -> 120 dB, capped, not zero", detail=f"{r1['mse']:.2e}")
    r2 = M.psnr_record(t + 1e-3, t)
    cx.ok(abs(r2["psnr"] - 60.0) < 1e-3 and not r2["psnr_capped"],
          "MSE 1e-6 -> 60 dB, not capped", detail=f"{r2['psnr']:.6f}")
    rows = []
    spec = [("A", 0, 30.0, 0.1, True, True), ("A", 1, 32.0, 0.3, True, False),
            ("B", 0, 120.0, None, False, True), ("C", 0, 24.0, 0.5, True, True),
            ("C", 1, 26.0, 0.2, True, False), ("C", 2, 28.0, 0.9, True, False)]
    for v, sl, ps, nm, el, mid in spec:
        rows.append({"volume_id": v, "file": f"x/{v}", "slice_index": sl,
                     "midpoint": mid, "eligible": el, "psnr": ps, "ssim": 0.5,
                     "nll_per_dim": 1.0, "nmse_u": nm, "mse": 0.0 if ps == 120
                     else 1e-3, "psnr_capped": ps == 120.0,
                     "zero_error": ps == 120.0})
    sm = M.summarise(rows)["all_slices"]
    cx.ok(abs(sm["psnr"] - (31.0 + 120.0 + 26.0) / 3) < 1e-12,
          "PSNR: slice -> volume mean -> equal-weight mean; capped slice "
          "retained")
    cx.ok(abs(sm["nmse_u"] - (0.2 + (0.5 + 0.2 + 0.9) / 3) / 2) < 1e-12
          and sm["nmse_null_volumes"] == ["B"] and sm[
              "nmse_population_volumes"] == 2,
          "NMSE: eligible-only volume means; null volume outside population")
    cx.ok(sm["psnr_capped"]["count"] == 1 and sm["zero_error"]["identities"]
          == ["x/B|0"], "cap / zero-error counts with identities")
    cx.ok(abs(sm["nmse_u"] - (0.1 + 0.3 + 0.5 + 0.2 + 0.9) / 5) > 1e-6,
          "mean of per-slice ratios differs from pooled/slice-weighted")
    cx.record("MET scope", "F22 part (cap flags) and F06 part (aggregation); "
              "bootstrap at I1.8, F22 remainder at I1.7")


# ---------------------------------------------------------------------------
# VAL -- shared validation on the real model
# ---------------------------------------------------------------------------
def _unit_p4():
    from seqref_mri.src import free_flow_runtime as ffr
    return {"location_index": ffr.build_location_index(
        [{"row": r, "column": c, "applied_mean_re": 0.0, "applied_mean_im": 0.0,
          "applied_scale_re": 1.0, "applied_scale_im": 1.0}
         for r in range(96) for c in range(96)])}


def _val_checks(cx: Ctx, vv, elig, p4, s_ref, idx, label):
    import torch
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.scripts import train_free_flow as tff
    from seqref_mri.scripts.train_base import _collate
    from seqref_mri.src import free_flow_runtime as ffr
    from seqref_mri.scripts import tiny_gate as tg
    model = ffr.build_model(init_seed=2026)
    model.train()
    h0 = C.model_state_sha256(model)
    rng0 = _rng_snapshot()
    rows = M.validate(model, vv, elig, p4, s_ref, idx)
    cx.ok(model.training and C.model_state_sha256(model) == h0
          and _rng_equal(rng0, _rng_snapshot()),
          f"{label}: validation leaves weights, mode and RNG unchanged")
    rows2 = M.validate(model, vv, elig, p4, s_ref, idx)
    cx.ok(rows == rows2, f"{label}: repeated validation bit-identical")
    with torch.no_grad():
        for k, i in enumerate(idx[:3]):
            item = vv[i]
            st = tg._build_slice_states(_collate([item]), {"ordered_identities": [
                {"file": rows[k]["file"], "slice_index": rows[k]["slice_index"]}]},
                p4, s_ref)[0]
            tgt = torch.from_numpy(st["target"].astype(np.float32))
            ref = float(tff.nll_objective(model.eval(), tgt, st["cond"],
                                          st["mask"])) / ffr.FLOW_DIM_REAL
            cx.ok(abs(rows[k]["nll_per_dim"] - ref) <= 1e-5 * abs(ref),
                  f"{label}: NLL/dim = production nll_objective / d_free "
                  "(single-slice batch, rel 1e-5)",
                  detail=f"{rows[k]['nll_per_dim']:.8f} vs {ref:.8f}")
            if rows[k]["eligible"]:
                x0, u0 = tg._decode_z(model, torch.zeros(1, ffr.FLOW_DIM_REAL), st)
                nm = float(np.sum(np.abs(u0 - st["u_true"]) ** 2)
                           / np.sum(np.abs(st["u_true"]) ** 2))
                cx.ok(abs(rows[k]["nmse_u"] - nm) <= 1e-12 * max(1.0, nm),
                      f"{label}: NMSE_u = independent numpy ratio")
    model.train()
    sm = M.summarise(rows)
    cx.ok(sm["all_slices"]["slices"] == len(idx), f"{label}: summary built")
    del model
    import gc
    gc.collect()
    return rows, sm


def val(cx: Ctx) -> None:
    from seqref_mri.scripts import v05lt_eligibility as E
    with tempfile.TemporaryDirectory() as t:
        root, mp, msha, ex = synth_split(Path(t), SYN_VOLS, SYN_TRAIN, SYN_VAL)
        sdoc = C.load_split_manifest(mp, msha, ex)
        _, vv = C.build_split_views(root, sdoc, SYN_HOLDOUT, ex)
        recs = E.compute_records(vv, 1.0)
        doc = E.build_manifest(recs, {"validation": sdoc["validation"]},
                               {"cases": []}, {"p": 1})
        from seqref_mri.scripts import v05lt_metrics as M
        _val_checks(cx, vv, M.eligibility_index(doc), _unit_p4(), 1.0,
                    list(range(len(vv))), "synthetic (unit P4 stand-in)")
    if not cx.host:
        return
    cx.phase = "host"
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.src import free_flow_runtime as ffr
    p = cx.paths
    s_ref = C.verify_p0s(p.p0s)
    p4 = ffr.load_p4s2_parent(str(p.repo / P4_REL))
    sdoc = C.load_split_manifest(p.inputs / "v05lt_split_manifest.json",
                                 C.REGISTERED_INPUTS["v05lt_split_manifest.json"])
    _, vv = C.build_split_views(cx.data_root, sdoc,
                                C.holdout_ids(p.v02_manifests))
    case = json.loads((p.inputs / "v05lt_case_manifest.json").read_bytes())
    want = [(c["file"], c["slice_index"]) for c in case["cases"]]
    pos = {vv.entry(i): i for i in range(len(vv))}
    idx = [pos[w] for w in want] + [0, 1]
    recs = E.compute_records(vv, s_ref, idx)
    doc = E.build_manifest(recs, {"validation": sdoc["validation"]}, case,
                           {"p": "fixture"})
    rows, sm = _val_checks(cx, vv, M.eligibility_index(doc), p4, s_ref, idx,
                           "real slices (pinned P4)")
    cx.record("VAL sample summary (untrained seed-2026 build; fixture only)",
              {k: sm["all_slices"][k] for k in ("psnr", "ssim", "nll_per_dim",
                                                "nmse_u")})


# ---------------------------------------------------------------------------
# I1.6(b): trainer fixtures (stub model on synthetic data unless stated)
# ---------------------------------------------------------------------------
class _SimulatedKill(BaseException):
    """In-process interruption (BaseException: bypasses every handler, as a
    kill would). Process-kill durability itself is covered by F09CS/E0S."""


def _syn_train_data(tmp: Path):
    from seqref_mri.scripts import v05lt_eligibility as E
    from seqref_mri.scripts import v05lt_metrics as M
    root, mp, msha, ex = synth_split(tmp, SYN_VOLS, SYN_TRAIN, SYN_VAL)
    sdoc = C.load_split_manifest(mp, msha, ex)
    tv, vv = C.build_split_views(root, sdoc, SYN_HOLDOUT, ex)
    recs = E.compute_records(vv, 1.0)
    doc = E.build_manifest(recs, {"validation": sdoc["validation"]},
                           {"cases": []}, {"p": "fixture"})
    return {"tv": tv, "vv": vv, "p4": _unit_p4(), "s_ref": 1.0,
            "eligibility": M.eligibility_index(doc)}


def _stub_cfg(run_dir, seed=2026, arm="lam0", max_epochs=12):
    from seqref_mri.scripts import v05lt_train as T
    from seqref_mri.scripts.v05lt_stub_train import build_stub
    return T.TrainConfig(seed=seed, arm=arm, run_dir=str(run_dir),
                         epoch0_model_state=C.model_state_sha256(
                             build_stub(seed)),
                         max_epochs=max_epochs, fixture=True,
                         note="stub fixture")


def _stub_run(cfg, data, hook=None, poison_after=None):
    from seqref_mri.scripts import v05lt_train as T
    from seqref_mri.scripts.v05lt_stub_train import build_stub
    return T.Run(cfg, C.Paths(), data,
                 model_factory=lambda s: build_stub(s, poison_after), hook=hook)


def _killer(stage_name: str):
    def hook(stage):
        if stage == stage_name:
            raise _SimulatedKill(stage)
    return hook


def _commits(run_dir: Path) -> dict:
    out = {}
    for r in C.Ledger(run_dir / "ledger.jsonl").records():
        if r["kind"] == "commit":
            b = json.loads(json.dumps({k: v for k, v in r["body"].items()
                                       if k not in PERMITTED_COMMIT_KEYS}))
            # the resume file embeds the ledger position, which is
            # session-dependent by design; its hash is compared separately
            b["retained"]["resume"].pop("sha256", None)
            out[b["epoch"]] = b
    return out


def _final_state(run_dir: Path):
    import torch
    from seqref_mri.scripts import v05lt_train as T
    st = C.ArtifactStore(run_dir / "ckpt", C.Ledger(run_dir / "ledger.jsonl"),
                         run_dir / "_incomplete").committed()
    obj = T.load_resume(run_dir / "ckpt" / st["resume"]["name"])
    adam = obj["adam"]["state"]
    flat = [t for k in sorted(adam) for t in (adam[k]["exp_avg"],
                                              adam[k]["exp_avg_sq"])]
    return (C.model_state_sha256(obj["model"]),
            [torch.equal(a, a) for a in flat], obj["state"], flat)


def ret(cx: Ctx) -> None:
    """F08 retention part: roles, overlap, coexistence, delete-after-commit,
    peak storage."""
    sizes, events = [], []

    def ckpt_bytes(d):
        return sum(q.stat().st_size for q in d.iterdir() if q.is_file())
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        data = _syn_train_data(tmp)
        run_dir = tmp / "run"
        names_at = {}

        def hook(stage):
            d = run_dir / "ckpt"
            if stage in ("commit:before_ledger", "commit:after_ledger"):
                names = sorted(q.name for q in d.iterdir())
                events.append((stage, names))
                sizes.append(ckpt_bytes(d))
        out = _stub_run(_stub_cfg(run_dir), data, hook).start()
        cx.ok(out["status"] == "fixture_budget_end" and out["epoch"] == 12,
              "stub run reached its fixture budget", detail=str(out["status"]))
        cm = _commits(run_dir)
        st = C.ArtifactStore(run_dir / "ckpt", C.Ledger(run_dir / "ledger.jsonl"),
                             run_dir / "_incomplete")
        final = st.committed()
        cx.ok({"milestone_003", "milestone_010"} <= set(final)
              and final["resume"]["name"] == "resume_e012.pt",
              "milestones 3 and 10 retained; resume = last epoch")
        sel = out["selection"]
        cx.ok(final["best_psnr"]["name"] == f"weights_e{sel['best_psnr']['epoch']:03d}.pt"
              and final["best_nll"]["name"] == f"weights_e{sel['best_nll']['epoch']:03d}.pt",
              "best PSNR / best NLL files match the selection records")
        listing = {q.name for q in (run_dir / "ckpt").iterdir()}
        keep = {v["name"] for v in final.values()}
        cx.ok(listing == keep | {n + ".sha256" for n in keep},
              "directory == retained set (superseded files deleted)")
        shared = [e for e, b in cm.items() if len(b["roles_this_epoch"]) > 1]
        cx.ok(bool(shared) and all(len({b["retained"][r]["name"] for r in
                                         b["roles_this_epoch"]}) == 1
                                    for e, b in cm.items() if e in shared),
              "overlapping roles share one weights file",
              detail=f"epochs {shared}")
        coexist = [n for s_, n in events if s_ == "commit:before_ledger"
                   and sum(x.startswith("resume_") and x.endswith(".pt")
                           for x in n) == 2]
        cx.ok(len(coexist) == 11, "old and new resume coexist at every "
              "commit after epoch 1", detail=str(len(coexist)))
        after = [n for s_, n in events if s_ == "commit:after_ledger"]
        cx.ok(all(sum(x.startswith("resume_") and x.endswith(".pt") for x in n)
                  == (1 if i == 0 else 2) for i, n in enumerate(after)),
              "superseded resume still present right after the durable commit "
              "(deleted only afterwards)")
        peak = max(sizes)
        cx.record("RET peak ckpt bytes (stub)", {"peak": peak,
                                                 "final": ckpt_bytes(run_dir / "ckpt")})
        cx.ok(peak >= ckpt_bytes(run_dir / "ckpt"), "peak >= final retained size")


def res(cx: Ctx) -> None:
    """Stub resume equivalence, fixed order, session limit, taxonomy."""
    from seqref_mri.scripts import v05lt_train as T
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        data = _syn_train_data(tmp)
        n_steps = (len(data["tv"]) + T.BATCH - 1) // T.BATCH
        ua = tmp / "u_lam2"
        _stub_run(_stub_cfg(ua, arm="lam2", max_epochs=8), data).start()
        cu = _commits(ua)
        for label, kill in (("mid-epoch 4 (training step)",
                             f"train:step:{3 * n_steps + 1}"),
                            ("epoch-5 commit before ledger",
                             "commit:before_ledger")):
            ia = tmp / f"i_{len(label)}"
            cfg = _stub_cfg(ia, arm="lam2", max_epochs=8)
            hook = _killer(kill)
            if kill == "commit:before_ledger":
                cnt = {"n": 0}

                def hook(stage, cnt=cnt):
                    if stage == "commit:before_ledger":
                        cnt["n"] += 1
                        if cnt["n"] == 5:
                            raise _SimulatedKill(stage)
            try:
                _stub_run(cfg, data, hook).start()
                cx.ok(False, f"{label}: interruption expected", "NO_KILL")
            except _SimulatedKill:
                cx.ok(True, f"{label}: run interrupted")
            out = _stub_run(cfg, data).resume()
            ci = _commits(ia)
            cx.ok(out["epoch"] == 8 and ci == cu,
                  f"{label}: every epoch record identical to the uninterrupted "
                  "run (order hashes, losses, validation, states)")
            a, b = _final_state(ua), _final_state(ia)
            import torch
            sa = {k: v for k, v in a[2].items() if k != "ledger_seq_before_commit"}
            sb = {k: v for k, v in b[2].items() if k != "ledger_seq_before_commit"}
            cx.ok(a[2]["ledger_seq_before_commit"] != b[2][
                "ledger_seq_before_commit"], f"{label}: ledger position differs "
                "(extra session records) -- the only expected difference")
            cx.ok(a[0] == b[0] and sa == sb and all(
                torch.equal(x, y) for x, y in zip(a[3], b[3])),
                f"{label}: final model, Adam, RNG and rule state bitwise equal")
        o = [cu[e]["order_sha256"] for e in sorted(cu)]
        cx.ok(len(set(o)) == len(o), "a new fixed permutation every epoch")
        ub = tmp / "u_lam0"
        _stub_run(_stub_cfg(ub, arm="lam0", max_epochs=3), data).start()
        cs = tmp / "u_2027"
        _stub_run(_stub_cfg(cs, seed=2027, arm="lam0", max_epochs=3), data).start()
        o0 = [_commits(ub)[e]["order_sha256"] for e in (1, 2, 3)]
        o7 = [_commits(cs)[e]["order_sha256"] for e in (1, 2, 3)]
        cx.ok(o0 == o[:3] and o7 != o0, "same seed -> same order in both arms; "
              "other seed differs (F13 data-order part, stub)")
        sl = tmp / "sess"
        cfg = _stub_cfg(sl, max_epochs=8)
        for k, call in ((1, "start"), (2, "resume"), (3, "resume")):
            r = _stub_run(cfg, data, _killer(f"train:step:{k * n_steps + 1}"))
            try:
                getattr(r, call)()
            except _SimulatedKill as kill:
                logger.info("session %d interrupted as intended (%s)", k, kill)
        cx.raises("SESSION_LIMIT", lambda: _stub_run(cfg, data).resume(),
                  "4th session refused")
        ends = [x for x in C.Ledger(sl / "ledger.jsonl").records()
                if x["kind"] == "run_end"]
        cx.ok(ends and ends[-1]["body"]["status"] == T.INTERRUPTED_STATUS,
              "status 'interrupted — awaiting owner decision' recorded")
        nf = tmp / "nonfinite"
        cfgn = _stub_cfg(nf, max_epochs=4)
        cx.raises("NLL_NON_FINITE",
                  lambda: _stub_run(cfgn, data, poison_after=3).start(),
                  "non-finite loss stopped by the production NLL gate (stub)")
        ends = [x for x in C.Ledger(nf / "ledger.jsonl").records()
                if x["kind"] == "run_end"]
        cx.ok(ends and ends[-1]["body"]["status"] == "numerical_failure",
              "numerical_failure recorded as terminal")
        cx.raises("RESUME_RUN_ENDED", lambda: _stub_run(cfgn, data).resume(),
                  "terminal run never resumed")
    cx.record("RES scope", "stub model on the real trainer control flow; "
              "real-trainer F09, F14 and remaining F13/F21 in I1.6(c)")


def grd(cx: Ctx) -> None:
    from seqref_mri.scripts import v05lt_train as T
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        data = _syn_train_data(tmp)
        cfg = _stub_cfg(C.Paths().v05lt / "fixture_must_not_run")
        cx.raises("FIXTURE_TARGETS_PRODUCTION", lambda: _stub_run(cfg, data),
                  "fixture config pointing into results/v05lt refused")
        cfg = _stub_cfg(tmp / "p", max_epochs=5)
        cfg.fixture = False
        cx.raises("PRODUCTION_CONFIG_ALTERED", lambda: _stub_run(cfg, data),
                  "production config with altered budget refused")
        cx.ok(T.main([]) == 2, "trainer CLI refuses production "
              "(TRAIN_PRODUCTION_DISABLED until I1.9)")
        cfg = _stub_cfg(tmp / "r", max_epochs=2)
        _stub_run(cfg, data).start()
        cx.raises("ARTIFACT_EXISTS", lambda: _stub_run(cfg, data).start(),
                  "existing run directory blocks a fresh start")
        cfg2 = _stub_cfg(tmp / "r", max_epochs=3)
        cx.raises("RESUME_CONFIG_MISMATCH", lambda: _stub_run(cfg2, data).resume(),
                  "resume with a different configuration refused")
        cfg3 = _stub_cfg(tmp / "x", max_epochs=2)
        cfg3.epoch0_model_state = "0" * 64
        cx.raises("EPOCH0_ARM_MISMATCH", lambda: _stub_run(cfg3, data).start(),
                  "initial weights must equal the epoch-0 hash")


def _sleeper(_):
    import signal
    import time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(600)


def chl(cx: Ctx) -> None:
    """Fix 10: escalation terminate -> kill, exit status before results."""
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_sleeper, args=(0,))
    p.start()
    try:
        join_bounded(p, "spawn", timeout=2, grace=2)
        cx.ok(False, "hung child must fail", "NO_ERROR_RAISED")
    except C.V05LTError as exc:
        cx.ok(exc.code == "CHILD_TIMEOUT" and "killed" in str(exc)
              and not p.is_alive(), "SIGTERM-ignoring child escalated to "
              "SIGKILL and reported", detail=str(exc)[:80])


def trn(cx: Ctx) -> None:
    """Registered tiny real-model smoke (2 epochs) in temporary folders."""
    import gc
    from seqref_mri.scripts import v05lt_epoch0 as E0
    from seqref_mri.scripts import v05lt_eligibility as E
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.scripts import v05lt_train as T
    from seqref_mri.src import free_flow_runtime as ffr
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        if cx.host:
            cx.phase = "host"
            p = cx.paths
            sdoc = C.load_split_manifest(
                p.inputs / "v05lt_split_manifest.json",
                C.REGISTERED_INPUTS["v05lt_split_manifest.json"])
            tv, vv = C.build_split_views(cx.data_root, sdoc,
                                         C.holdout_ids(p.v02_manifests))
            s_ref = C.verify_p0s(p.p0s)
            p4 = ffr.load_p4s2_parent(str(p.repo / P4_REL))
            n_train, val_idx = 64, list(range(8))
            recs = E.compute_records(vv, s_ref, val_idx)
            split = {"validation": sdoc["validation"]}
        else:
            data = _syn_train_data(tmp / "syn")
            tv, vv, p4, s_ref = data["tv"], data["vv"], data["p4"], 1.0
            n_train, val_idx = len(tv), list(range(len(vv)))
            recs = E.compute_records(vv, s_ref, val_idx)
            split = {"validation": [{"volume_id": v, "n_slices": n} for v, n in
                                    SYN_VOLS.items() if v in SYN_VAL]}
        elig = M.eligibility_index(E.build_manifest(recs, split, {"cases": []},
                                                    {"p": "fixture"}))
        e0 = E0.create(tmp / "epoch0", 2026, REPO_ROOT)
        conf = {"seed": 2026, "arm": "lam2", "max_epochs": 2,
                "train_indices": list(range(n_train)), "val_indices": val_idx,
                "epoch0_model_state": e0["model_state_sha256"],
                "purpose": "I1.6(b) TRN tiny real-model smoke (fixture only)"}
        cpath = tmp / "trn_config.json"
        csha = C.write_durable_new(cpath, C.canonical_json(conf))
        cx.record("TRN registered config sha256 (written before the run)", csha)
        cfg = T.TrainConfig(seed=2026, arm="lam2", run_dir=str(tmp / "run"),
                            epoch0_model_state=conf["epoch0_model_state"],
                            max_epochs=2, fixture=True,
                            train_indices=conf["train_indices"],
                            val_indices=val_idx, note=csha)
        out = T.Run(cfg, C.Paths(), {"tv": tv, "vv": vv, "p4": p4,
                                     "s_ref": s_ref, "eligibility": elig}).start()
        cm = _commits(tmp / "run")
        cx.ok(out["status"] == "fixture_budget_end" and sorted(cm) == [1, 2],
              "real model: 2 epochs trained, validated and committed")
        cx.ok(all(np.isfinite(cm[e]["losses"]["L_total"])
                  and cm[e]["losses"]["L_rec"] > 0 for e in (1, 2)),
              "lam2 objective finite with a positive L_rec term")
        cx.ok(cm[1]["order_sha256"] != cm[2]["order_sha256"],
              "distinct fixed permutations per epoch")
        st = _final_state(tmp / "run")[2]
        cx.ok(st["next_epoch"] == 3 and st["mask_epoch_next"] == 2
              and set(st) >= {"rng", "order_rng", "plateau", "selection",
                              "ledger_seq_before_commit", "step"},
              "resume file carries the full state")
        cx.record("TRN epoch records", {e: {"L_total": cm[e]["losses"]["L_total"],
                                            "psnr": cm[e]["validation"][
                                                "all_slices"]["psnr"]}
                                        for e in cm})
        gc.collect()


# ---------------------------------------------------------------------------
# I1.6(c) (SEQREF-MRI-V05LTI16C v0.2): trainer fixes 13-17 (TFX, stub) and
# real-trainer groups REP (B), F09, F14, F13C, F21C, F08R, KIL (A), RNGC (C)
# ---------------------------------------------------------------------------
# Predeclared permitted differences between runs that must otherwise be
# bitwise identical (F09, REP, F21C, KIL). Everything else is compared.
PERMITTED_COMMIT_KEYS = ("seconds", "session", "store", "resume_state_sha256")
PERMITTED_DIFFS = (
    "commit.seconds (wall-clock timing)",
    "commit.session (session number)",
    "commit.store (store path)",
    "commit.resume_state_sha256 and the resume file's sha256 (both embed the "
    "ledger position)",
    "resume state ledger_seq_before_commit (ledger position)",
    "session_start / restart_uncommitted / resume_refused / run_end-recovered "
    "ledger records and therefore ledger positions",
)


def _kinds(run_dir: Path) -> list:
    return [r["kind"] for r in C.Ledger(run_dir / "ledger.jsonl").records()]


def _ends(run_dir: Path) -> list:
    return [r["body"] for r in C.Ledger(run_dir / "ledger.jsonl").records()
            if r["kind"] == "run_end"]


def _first_diff(a: dict, b: dict) -> str:
    for e in sorted(set(a) | set(b)):
        if a.get(e) != b.get(e):
            if e not in a or e not in b:
                return f"epoch {e} missing on one side"
            ks = sorted(k for k in set(a[e]) | set(b[e])
                        if a[e].get(k) != b[e].get(k))
            return f"epoch {e}: keys {ks[:6]}"
    return "none"


def _same_runs(cx: Ctx, ref: Path, other: Path, label: str) -> None:
    """Scientific epoch records + final model / Adam / RNG / rule state."""
    import torch
    ca, cb = _commits(ref), _commits(other)
    cx.ok(ca == cb, f"{label}: every epoch record identical (order, losses, "
          "validation, model/Adam/scientific resume-state hashes, rules, "
          "retention)", "RUNS_DIFFER", _first_diff(ca, cb))
    a, b = _final_state(ref), _final_state(other)
    sa = {k: v for k, v in a[2].items() if k != "ledger_seq_before_commit"}
    sb = {k: v for k, v in b[2].items() if k != "ledger_seq_before_commit"}
    cx.ok(a[0] == b[0] and sa == sb and len(a[3]) == len(b[3]) and all(
        torch.equal(x, y) for x, y in zip(a[3], b[3])),
        f"{label}: final model, Adam moments, Python/NumPy/torch-CPU RNG, "
        "shuffle generator and rule state bitwise equal", "RUNS_DIFFER")


def _expect(cx: Ctx, code: str, fn, label: str, phrase: str = "") -> None:
    try:
        fn()
    except C.V05LTError as exc:
        cx.ok(exc.code == code and phrase in str(exc), label,
              "WRONG_ERROR", f"(expected {code} '{phrase}', got {exc})")
        return
    cx.ok(False, label, "NO_ERROR_RAISED", f"(expected {code})")


def _recommit(run_dir: Path, mutate, rehash: bool = False) -> None:
    """Fixture-only tamper: write a modified resume file and commit it
    through the real store with the last commit's body. rehash=True also
    makes the commit's hashes (and the ledger position) consistent with the
    modified file, so deeper checks are reached."""
    import torch
    from seqref_mri.scripts import v05lt_train as T
    led = C.Ledger(run_dir / "ledger.jsonl")
    st = C.ArtifactStore(run_dir / "ckpt", led, run_dir / "_incomplete")
    st.recover()
    last = st.last_commit()["body"]
    obj = T.load_resume(run_dir / "ckpt" / last["retained"]["resume"]["name"])
    if rehash:
        obj["state"]["ledger_seq_before_commit"] = len(led.records())
    mutate(obj)
    name = f"resume_tampered{len(led.records())}.pt"
    st.stage(name, lambda fh: torch.save(obj, fh))
    retained = {k: v["name"] for k, v in last["retained"].items()}
    retained["resume"] = name
    body = {k: v for k, v in last.items()
            if k not in ("retained", "superseded", "store")}
    if rehash:
        body["resume_state_sha256"] = T.state_sha256(obj["state"])
        body["adam_state_sha256"] = T.optimizer_state_sha256(obj["adam"])
    st.commit(retained, body)


def _interrupted_stub(tmp: Path, data, name: str, epochs: int = 4,
                      after_commits: int = 2) -> object:
    """Stub run interrupted inside epoch after_commits + 1 (n_steps = 1)."""
    from seqref_mri.scripts import v05lt_train as T
    cfg = _stub_cfg(tmp / name, max_epochs=epochs)
    n_steps = (len(data["tv"]) + T.BATCH - 1) // T.BATCH
    try:
        _stub_run(cfg, data, _killer(
            f"train:step:{after_commits * n_steps + 1}")).start()
    except _SimulatedKill as kill:
        logger.info("stub run %s interrupted as intended (%s)", name, kill)
    return cfg


def _clone_cfg(cfg, run_dir: Path):
    import dataclasses
    shutil.copytree(cfg.run_dir, run_dir)
    return dataclasses.replace(cfg, run_dir=str(run_dir))


def tfx(cx: Ctx) -> None:
    """Trainer fixes 13-17 on the real trainer control flow (stub model)."""
    import torch
    from seqref_mri.scripts import v05lt_train as T
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        data = _syn_train_data(tmp / "syn")
        ref = _stub_cfg(tmp / "ref", max_epochs=4)
        _stub_run(ref, data).start()
        # -- 13: complete resume-state validation --------------------------
        base = _interrupted_stub(tmp, data, "base")
        cx.ok(len(_commits(Path(base.run_dir))) == 2, "13: base run "
              "interrupted after 2 commits")

        def first_adam(o):
            k = sorted(o["adam"]["state"])[0]
            o["adam"]["state"][k]["exp_avg"] = o["adam"]["state"][k][
                "exp_avg"] + 1.0

        def first_model(o):
            k = sorted(o["model"])[0]
            o["model"][k] = o["model"][k] + 1.0

        def bump(key):
            def f(o):
                o["state"][key] += 1
            return f

        def setf(path, value):
            def f(o):
                d = o
                for p in path[:-1]:
                    d = d[p]
                d[path[-1]] = value
            return f
        cases = [
            ("state changed, commit hash kept", bump("step"), False,
             "resume-state hash"),
            ("Adam moment changed", first_adam, False, "Adam-state hash"),
            ("model tensor changed", first_model, False, "model-state hash"),
            ("step inconsistent with commit", bump("step"), True, "step"),
            ("mask epoch inconsistent", bump("mask_epoch_next"), True,
             "next epoch / mask epoch"),
            ("ledger position wrong", setf(["state",
                                            "ledger_seq_before_commit"], 0),
             True, "ledger position"),
            ("Python RNG state unrestorable", setf(["state", "rng", "python"],
                                                   [3, [1, 2], None]),
             True, "RNG state not restorable"),
            ("shuffle generator type wrong", setf(["state", "order_rng",
                                                   "bit_generator"], "MT19937"),
             True, "shuffle generator"),
            ("Adam learning rate altered", setf(["adam", "param_groups", 0,
                                                 "lr"], 1e-3), True,
             "Adam param groups"),
            ("plateau state differs from commit", _toggle_counter, True,
             "rule state"),
        ]
        for i, (label, mut, rehash, phrase) in enumerate(cases):
            cfg = _clone_cfg(base, tmp / f"t13_{i}")
            _recommit(Path(cfg.run_dir), mut, rehash)
            _expect(cx, "RESUME_STATE_MISMATCH",
                    lambda c=cfg: _stub_run(c, data).resume(),
                    f"13: {label} -> refused", phrase)
            ends = _ends(Path(cfg.run_dir))
            cx.ok(ends and ends[-1]["status"] == "failed_validity"
                  and ends[-1]["stage"] == "resume_validation",
                  f"13: {label} -> terminal failed_validity recorded "
                  "(stage resume_validation)")
            _expect(cx, "RESUME_RUN_ENDED",
                    lambda c=cfg: _stub_run(c, data).resume(),
                    f"13: {label} -> never resumed afterwards")
        cfg = _clone_cfg(base, tmp / "t13_file")
        st = C.ArtifactStore(Path(cfg.run_dir) / "ckpt",
                             C.Ledger(Path(cfg.run_dir) / "ledger.jsonl"),
                             Path(cfg.run_dir) / "_incomplete")
        rf = Path(cfg.run_dir) / "ckpt" / st.committed()["resume"]["name"]
        rf.chmod(0o644)
        with open(rf, "ab") as fh:
            fh.write(b"\0")
        _expect(cx, "STORE_HASH_MISMATCH", lambda: _stub_run(cfg, data).resume(),
                "13/17: altered resume file bytes -> refused")
        cx.ok(_ends(Path(cfg.run_dir))[-1]["status"] == "failed_validity",
              "13/17: file-integrity failure at resume recorded as terminal")
        # untouched clone still resumes and matches the reference
        cfg = _clone_cfg(base, tmp / "t13_ok")
        _stub_run(cfg, data).resume()
        _same_runs(cx, Path(ref.run_dir), Path(cfg.run_dir),
                   "13: unmodified interrupted run")
        # -- 14: interruption before the first epoch commit -----------------
        for label, hook in (("training step 1", _killer("train:step:1")),
                            ("epoch-1 commit before ledger",
                             _killer("commit:before_ledger"))):
            cfg = _stub_cfg(tmp / f"t14_{len(label)}", max_epochs=4)
            try:
                _stub_run(cfg, data, hook).start()
                cx.ok(False, f"14: {label}: interruption expected", "NO_KILL")
            except _SimulatedKill:
                cx.ok(not _commits(Path(cfg.run_dir)),
                      f"14: {label}: interrupted with no commit")
            _stub_run(cfg, data).resume()
            recs = C.Ledger(Path(cfg.run_dir) / "ledger.jsonl").records()
            ss = [r["body"] for r in recs if r["kind"] == "session_start"]
            cx.ok("restart_uncommitted" in [r["kind"] for r in recs]
                  and [s["mode"] for s in ss] == ["fresh", "fresh"]
                  and ss[0]["initial_rng_sha256"] == ss[1][
                      "initial_rng_sha256"],
                  f"14: {label}: explicit restart recorded; identical "
                  "seed-defined initial state")
            _same_runs(cx, Path(ref.run_dir), Path(cfg.run_dir),
                       f"14: {label}: restarted run")
        q = list((tmp / f"t14_{len('epoch-1 commit before ledger')}"
                  / "_incomplete").rglob("*.pt"))
        cx.ok(bool(q), "14: staged but uncommitted epoch-1 files quarantined",
              detail=str(len(q)))
        # -- 15: final commit durable, run_end missing ----------------------
        cfg = _stub_cfg(tmp / "t15", max_epochs=3)
        try:
            _stub_run(cfg, data, _killer("run:before_run_end")).start()
        except _SimulatedKill:
            cx.ok(not _ends(Path(cfg.run_dir)) and max(_commits(Path(
                cfg.run_dir))) == 3, "15: final commit durable, run_end "
                "never written")
        _expect(cx, "RESUME_RUN_ENDED", lambda: _stub_run(cfg, data).resume(),
                "15: resume recognises the final commit and refuses",
                "without run_end")
        e = _ends(Path(cfg.run_dir))
        cx.ok(len(e) == 1 and e[0]["recovered"] is True
              and e[0]["status"] == "fixture_budget_end",
              "15: run_end recorded on recovery (status from the commit)")
        _expect(cx, "RESUME_RUN_ENDED", lambda: _stub_run(cfg, data).resume(),
                "15: later resumes refused", "already ended")
        # -- 16: resume compatibility (code / environment / data) -----------
        cfg = _clone_cfg(base, tmp / "t16")
        from seqref_mri.scripts import v05lt_epoch0 as E0
        env = E0.environment()
        env["torch_num_threads"] = env["torch_num_threads"] + 1
        n0 = _kinds(Path(cfg.run_dir)).count("session_start")
        from seqref_mri.scripts.v05lt_stub_train import build_stub
        def env_run():
            r = _stub_run(cfg, data)
            r.env_override = env
            return r.resume()
        variants = (
            ("environment (thread setting)", env_run, "['env']"),
            ("data (S_ref)", lambda: _stub_run(
                cfg, {**data, "s_ref": 2.0}).resume(), "['data']"),
            ("code (model factory)", lambda: T.Run(
                cfg, C.Paths(), data, model_factory=_other_factory).resume(),
             "['implementation']"),
        )
        for what, fn, phrase in variants:
            _expect(cx, "RESUME_INCOMPATIBLE", fn,
                    f"16: only the {what} differs -> refused", phrase)
        k = _kinds(Path(cfg.run_dir))
        cx.ok(k.count("resume_refused") == 3 and k.count("session_start") == n0
              and "run_end" not in k, "16: refusals recorded; no session "
              "started; run not ended")
        _stub_run(cfg, data).resume()
        _same_runs(cx, Path(ref.run_dir), Path(cfg.run_dir),
                   "16: compatible resume after refusals")
        # -- 17: consistent terminal recording ------------------------------
        cfg = _stub_cfg(tmp / "t17_init", max_epochs=2)
        cfg.epoch0_model_state = "0" * 64
        _expect(cx, "EPOCH0_ARM_MISMATCH", lambda: _stub_run(cfg, data).start(),
                "17: epoch-0 mismatch at initialisation")
        e = _ends(Path(cfg.run_dir))
        cx.ok(len(e) == 1 and e[0]["status"] == "failed_validity"
              and e[0]["stage"] == "initialisation"
              and e[0]["code"] == "EPOCH0_ARM_MISMATCH",
              "17: initialisation failure recorded as terminal")
        _expect(cx, "RESUME_RUN_ENDED", lambda: _stub_run(cfg, data).resume(),
                "17: failed initialisation never resumed")
        cfg = _stub_cfg(tmp / "t17_err", max_epochs=4)

        def boom(stage):
            if stage == "train:step:2":
                raise RuntimeError("fixture-injected unexpected error")
        try:
            _stub_run(cfg, data, boom).start()
            cx.ok(False, "17: unexpected error expected", "NO_ERROR_RAISED")
        except RuntimeError as exc:
            k = _kinds(Path(cfg.run_dir))
            cx.ok("session_error" in k and "run_end" not in k,
                  "17: unexpected (non-V05LT) error -> session_error record, "
                  "not terminal", detail=str(exc))
        _stub_run(cfg, data).resume()
        _same_runs(cx, Path(ref.run_dir), Path(cfg.run_dir),
                   "17: run resumable after an unexpected error")
        cfg2 = _stub_cfg(tmp / "t17_err", max_epochs=5)
        _expect(cx, "RESUME_CONFIG_MISMATCH",
                lambda: _stub_run(cfg2, data).resume(),
                "17: configuration refusal")
        cx.ok(_kinds(Path(cfg.run_dir))[-1] == "resume_refused"
              and len(_ends(Path(cfg.run_dir))) == 1,
              "17: refusal recorded, not terminal")
        del torch


def _toggle_counter(o):
    """Fixture tamper: valid-looking plateau counter (0 <-> 1 at epoch 2)."""
    c = o["state"]["plateau"]["count"]
    c["psnr"] = 1 - c["psnr"]


def _other_factory(seed):
    """Fixture-only: a different model factory (code-identity test)."""
    from seqref_mri.scripts.v05lt_stub_train import build_stub
    return build_stub(seed)


# -- real-trainer infrastructure -------------------------------------------
RT_EPOCHS = 3
RT_MIN_FREE = 16e9          # peak: reference run + one run + epoch-0 files
_RT: dict = {}


def _rt_indices(host: bool) -> tuple:
    """Registered fixture subsets. Host: 40 training slices = 2 optimiser
    steps per epoch (32 + 8). Synthetic (local pre-check): the 10 synthetic
    slices = 1 step per epoch, because a production-size batch of 32 on the
    real lam2 model exceeds the sandbox memory; mid-epoch boundaries with 2
    steps are exercised on the host only (declared)."""
    if host:
        return list(range(40)), list(range(6))
    n = sum(v for k, v in SYN_VOLS.items() if k in SYN_TRAIN)
    return list(range(n)), [0, 1, 2]


def _rt_steps(host: bool) -> int:
    return (len(_rt_indices(host)[0]) + 31) // 32


def _rt_data(host: bool, repo: Path, data_root, tmp: Path) -> dict:
    from seqref_mri.scripts import v05lt_eligibility as E
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.src import free_flow_runtime as ffr
    if not host:
        return _syn_train_data(tmp / "syn")
    p = C.Paths(repo=repo)
    sdoc = C.load_split_manifest(
        p.inputs / "v05lt_split_manifest.json",
        C.REGISTERED_INPUTS["v05lt_split_manifest.json"])
    tv, vv = C.build_split_views(Path(data_root), sdoc,
                                 C.holdout_ids(p.v02_manifests))
    s_ref = C.verify_p0s(p.p0s)
    p4 = ffr.load_p4s2_parent(str(p.repo / P4_REL))
    val_idx = _rt_indices(True)[1]
    recs = E.compute_records(vv, s_ref, val_idx)
    elig = M.eligibility_index(E.build_manifest(
        recs, {"validation": sdoc["validation"]}, {"cases": []},
        {"p": "fixture"}))
    return {"tv": tv, "vv": vv, "p4": p4, "s_ref": s_ref, "eligibility": elig}


def _rt_child(spec_path: str) -> None:
    """Spawned real-trainer process (fresh interpreter). Writes a JSON
    result unless SIGKILLed on purpose."""
    import signal
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s[child]: %(message)s")
    import torch
    from seqref_mri.scripts import v05lt_train as T
    spec = json.loads(Path(spec_path).read_text())
    out = {"pid": os.getpid(), "torch_threads": torch.get_num_threads()}
    try:
        with tempfile.TemporaryDirectory() as t:
            data = _rt_data(spec["host"], Path(spec["repo"]),
                            spec["data_root"], Path(t))
            cfg = T.TrainConfig(**spec["cfg"])
            holder, counts, events, draws = {}, {}, [], []
            factory = None
            if spec.get("poison"):
                def factory(seed):
                    from seqref_mri.src import free_flow_runtime as ffr
                    holder["m"] = ffr.build_model(init_seed=seed)
                    return holder["m"]

            def hook(stage):
                k = spec.get("kill")
                if k and stage == k["stage"]:
                    counts[stage] = counts.get(stage, 0) + 1
                    if counts[stage] == k["occurrence"]:
                        logger.warning("fixture %s at %s (occurrence %d)",
                                       k["mode"], stage, k["occurrence"])
                        if k["mode"] == "sigkill":
                            os.kill(os.getpid(), signal.SIGKILL)
                        raise _SimulatedKill(stage)
                if spec.get("record_events") and stage in (
                        "commit:before_ledger", "commit:after_ledger"):
                    events.append([stage, sorted(
                        q.name for q in (Path(cfg.run_dir) / "ckpt").iterdir())])
                pz = spec.get("poison")
                if pz and stage == pz["stage"]:
                    par = next(holder["m"].parameters())
                    if pz["kind"] == "nan_param":
                        with torch.no_grad():
                            par.fill_(float("nan"))
                    else:
                        par.register_hook(lambda g: g * float("nan"))
                    logger.warning("fixture poison %s at %s", pz["kind"], stage)
            if spec.get("count_draws"):
                orig = T.draw_epoch_order

                def counted(order_rng, n):
                    before = order_rng.bit_generator.state
                    r = orig(order_rng, n)
                    draws.append({"n": n, "before": before,
                                  "after": order_rng.bit_generator.state})
                    return r
                T.draw_epoch_order = counted
            if spec.get("pre_eval"):
                from seqref_mri.scripts import v05lt_epoch0 as E0
                from seqref_mri.scripts import v05lt_metrics as M
                m0 = E0.load_epoch0(Path(spec["pre_eval"]),
                                    cfg.epoch0_model_state, cfg.seed)
                rows = M.validate(m0, data["vv"], data["eligibility"],
                                  data["p4"], data["s_ref"], cfg.val_indices)
                out["pre_eval_psnr"] = M.summarise(rows)["all_slices"]["psnr"]
                if spec.get("pre_eval_full"):
                    from seqref_mri.scripts import v05lt_eval as EV
                    vi = list(cfg.val_indices)
                    mid = vi[:2]          # fixture subset (not midpoints)
                    cases = [{"case": j, "file": data["vv"].entry(i)[0],
                              "slice_index": data["vv"].entry(i)[1]}
                             for j, i in enumerate(mid)]
                    sts = EV.build_states(data["vv"], mid, data["p4"],
                                          data["s_ref"])
                    tol = EV.calibrate_dc(sts)["tolerances"]
                    bank = (__import__("seqref_mri.tdiag.estimators",
                                       fromlist=["z_diag_bank"])
                            .z_diag_bank()["bank"] if spec["host"] else
                            np.random.Generator(np.random.PCG64(0))
                            .standard_normal((128, 13824)).astype(np.float32))
                    cnt = {}
                    rec, _ = EV.epoch0_evaluate(
                        m0, data, bank, tol, cases, all_indices=vi,
                        mid_indices=mid, decode_counter=cnt)
                    out["pre_eval_full"] = {"decodes": cnt.get("primary", 0),
                                            "rng": rec["rng"],
                                            "timing": rec["timing"]}
                del m0
            run = T.Run(cfg, C.Paths(repo=Path(spec["repo"])), data,
                        model_factory=factory, hook=hook)
            try:
                res = getattr(run, spec["action"])()
                out["result"] = {"status": res["status"], "epoch": res["epoch"]}
            except _SimulatedKill as kill:
                logger.warning("child interrupted as intended: %s", kill)
                out["interrupted"] = str(kill)
            except C.V05LTError as exc:
                logger.warning("child run raised %s (returned to parent)",
                               exc.code)
                out["error"] = {"code": exc.code, "taxonomy": exc.taxonomy,
                                "message": str(exc)[:400]}
            out["draws"], out["events"] = draws, events
    except Exception as exc:  # noqa: BLE001 -- reported to the parent
        logger.exception("real-trainer child failed")
        out["crash"] = f"{type(exc).__name__}: {exc}"
    Path(spec["out"]).write_text(json.dumps(out))


def _rt_setup(cx: Ctx) -> dict:
    if "root" in _RT:
        return _RT
    from seqref_mri.scripts import v05lt_epoch0 as E0
    td = tempfile.TemporaryDirectory(prefix="v05lt_i16c_")
    root = Path(td.name)
    free = shutil.disk_usage(root).free
    cx.record("I1.6(c) temp root and free space (GB)",
              f"{root} {free / 1e9:.1f}")
    if free < RT_MIN_FREE:
        td.cleanup()
        C.fail("FIXTURE_DISK_SPACE", f"{free / 1e9:.1f} GB free in "
               f"{root.parent}; need >= {RT_MIN_FREE / 1e9:.0f} GB (real-"
               "model resume files ~2.7 GB each); set TMPDIR to a larger "
               "temporary folder")
    _RT.update({"td": td, "root": root, "host": cx.host, "n": 0})
    e0 = {s: E0.create(root / "epoch0", s, REPO_ROOT) for s in (2026, 2027)}
    _RT["e0"] = {s: v["model_state_sha256"] for s, v in e0.items()}
    _RT["e0_path"] = {s: str(root / "epoch0" / f"seed_{s}" / "ckpt"
                             / E0.ckpt_name(s)) for s in e0}
    tr, va = _rt_indices(cx.host)
    confs = {}
    for name, seed, arm, ep in (("lam2_s2026", 2026, "lam2", RT_EPOCHS),
                                ("lam0_s2026", 2026, "lam0", RT_EPOCHS),
                                ("lam0_s2027", 2027, "lam0", 1)):
        conf = {"seed": seed, "arm": arm, "max_epochs": ep,
                "train_indices": tr, "val_indices": va,
                "epoch0_model_state": _RT["e0"][seed], "host": cx.host,
                "purpose": "I1.6(c) real-trainer fixture (V05LTI16C v0.2)"}
        sha = C.write_durable_new(root / f"config_{name}.json",
                                  C.canonical_json(conf))
        confs[name] = (conf, sha)
    _RT["confs"] = confs
    cx.record("I1.6(c) registered fixture configs (sha256, written before "
              "any run)", {k: v[1] for k, v in confs.items()})
    cx.record("I1.6(c) permitted differences (predeclared)", PERMITTED_DIFFS)
    return _RT


def _rt_run(cx: Ctx, conf: str, run: str, action: str = "start",
            expect_kill: bool = False, **opts) -> dict:
    rt = _rt_setup(cx)
    c, sha = rt["confs"][conf]
    rt["n"] += 1
    spec = {"host": cx.host, "repo": str(REPO_ROOT),
            "data_root": str(cx.data_root) if cx.host else None,
            "cfg": {"seed": c["seed"], "arm": c["arm"],
                    "run_dir": str(rt["root"] / run),
                    "epoch0_model_state": c["epoch0_model_state"],
                    "max_epochs": c["max_epochs"], "fixture": True,
                    "train_indices": c["train_indices"],
                    "val_indices": c["val_indices"], "note": sha},
            "action": action, "out": str(rt["root"] / f"out_{rt['n']}.json"),
            **opts}
    sp = rt["root"] / f"spec_{rt['n']}.json"
    sp.write_text(json.dumps(spec))
    logger.info("real-trainer child %d: %s %s %s", rt["n"], run, action,
                {k: v for k, v in opts.items() if k != "pre_eval"})
    p = multiprocessing.get_context("spawn").Process(target=_rt_child,
                                                     args=(str(sp),))
    p.start()
    code = join_bounded(p, "spawn")
    if expect_kill:
        cx.ok(code == -9 and not Path(spec["out"]).exists(),
              f"{run}: child SIGKILLed at the target boundary",
              "CHILD_NOT_KILLED", f"exit {code}")
        return {}
    cx.ok(code == 0, f"{run} {action}: child exit 0", "CHILD_FAILED",
          f"exit {code}")
    out = json.loads(Path(spec["out"]).read_text())
    cx.ok("crash" not in out, f"{run} {action}: no unexpected error",
          "CHILD_CRASHED", out.get("crash", ""))
    return out


def _rt_dir(name: str) -> Path:
    return _RT["root"] / name


def _rt_drop(*names: str) -> None:
    """Delete finished fixture runs (each real-model resume file is
    ~2.7 GB); only the reference run is kept until the end."""
    for n in names:
        if _rt_dir(n).exists():
            shutil.rmtree(_rt_dir(n))
            logger.info("fixture run %s deleted after its checks", n)


def _rt_ref(cx: Ctx) -> Path:
    cx.ok(not _RT.get("ref_failed"), "reference run available", "REF_FAILED")
    if "ref" not in _RT:
        _RT["ref_failed"] = True
        out = _rt_run(cx, "lam2_s2026", "ref", record_events=True,
                      count_draws=True)
        cx.ok(out.get("result", {}).get("status") == "fixture_budget_end",
              "reference run (lam2, seed 2026) completed", "REF_FAILED",
              str(out.get("error")))
        _RT["ref"], _RT["ref_out"] = _rt_dir("ref"), out
        _RT["ref_failed"] = False
    return _RT["ref"]


def _rt_phase(cx: Ctx) -> None:
    cx.phase = "host" if cx.host else "synthetic"


def rep(cx: Ctx) -> None:
    """Addition B: independent repeatability in fresh processes (first)."""
    _rt_phase(cx)
    ref = _rt_ref(cx)
    out = _rt_run(cx, "lam2_s2026", "rep2")
    cx.ok(out.get("result", {}).get("status") == "fixture_budget_end",
          "second independent run completed")
    sa = C.Ledger(ref / "ledger.jsonl").records()[0]["body"]
    sb = C.Ledger(_rt_dir("rep2") / "ledger.jsonl").records()[0]["body"]
    cx.ok(sa["env"] == sb["env"] and sa["implementation"] == sb[
        "implementation"] and sa["data"] == sb["data"],
        "same registered CPU settings, code and data in both processes")
    cx.record("REP environment (both runs)", sa["env"])
    _same_runs(cx, ref, _rt_dir("rep2"), "REP: two fresh processes")
    _rt_drop("rep2")
    cm = _commits(ref)
    cx.record("REP epoch records", {e: {
        "L_total": cm[e]["losses"]["L_total"],
        "psnr": cm[e]["validation"]["all_slices"]["psnr"],
        "model_state": cm[e]["model_state_sha256"][:16]} for e in cm})


def f09(cx: Ctx) -> None:
    """F09 on the real trainer: interrupted + resumed == uninterrupted."""
    _rt_phase(cx)
    ref = _rt_ref(cx)
    steps = _rt_steps(cx.host)
    for run, stage, occ, label in (
            ("f09_mid2", f"train:step:{steps + 1}", 1, "mid-epoch 2"),
            ("f09_commit2", "commit:before_ledger", 2,
             "epoch-2 commit, before the ledger record")):
        out = _rt_run(cx, "lam2_s2026", run, kill={"stage": stage,
                                                   "occurrence": occ,
                                                   "mode": "simulate"})
        cx.ok(out.get("interrupted") == stage, f"F09 {label}: interrupted")
        out = _rt_run(cx, "lam2_s2026", run, "resume", count_draws=True)
        cx.ok(out.get("result", {}).get("status") == "fixture_budget_end",
              f"F09 {label}: resumed to the end")
        _RT[f"{run}_out"] = out
        k = _kinds(_rt_dir(run))
        cx.ok(k.count("session_start") == 2 and k.count("run_end") == 1,
              f"F09 {label}: two sessions, one run_end")
        _same_runs(cx, ref, _rt_dir(run), f"F09 {label}")
        if run == "f09_commit2":
            q = list((_rt_dir(run) / "_incomplete").rglob("*.pt"))
            cx.ok(bool(q), "F09: staged epoch-2 files of the interrupted "
                  "commit quarantined, epoch 2 retrained", detail=str(len(q)))
        _rt_drop(run)


def f14(cx: Ctx) -> None:
    """F14 on the real trainer: actual non-finite values; resume refused."""
    import torch
    from seqref_mri.scripts import v05lt_train as T
    _rt_phase(cx)
    ep = 1 if _rt_steps(cx.host) >= 2 else 2     # epoch holding step 2
    for run, kind, code, phrase in (
            ("f14_nan_param", "nan_param", None, ""),
            ("f14_nan_grad", "nan_grad", "NUMERICAL_FAILURE",
             "gradient non-finite")):
        out = _rt_run(cx, "lam2_s2026", run, poison={
            "stage": "train:before_step:2", "kind": kind})
        err = out.get("error") or {}
        cx.ok(err.get("taxonomy") == "numerical_failure"
              and (code is None or err.get("code") == code)
              and phrase in err.get("message", ""),
              f"F14 {kind}: classified numerical_failure", "WRONG_TAXONOMY",
              str(err))
        e = _ends(_rt_dir(run))
        cx.ok(len(e) == 1 and e[0]["status"] == "numerical_failure"
              and e[0]["stage"] == "training" and e[0]["epoch"] == ep
              and len(_commits(_rt_dir(run))) == ep - 1,
              f"F14 {kind}: terminal record (training, epoch {ep}), failing "
              "epoch not committed")
        cx.record(f"F14 {kind} failure code", err.get("code"))
        out = _rt_run(cx, "lam2_s2026", run, "resume")
        cx.ok((out.get("error") or {}).get("code") == "RESUME_RUN_ENDED",
              f"F14 {kind}: resume refused (RESUME_RUN_ENDED)")
        _rt_drop(run)
    # failed_validity at resume validation (real trainer, Adam tampered)
    run = "f14_tamper"
    out = _rt_run(cx, "lam2_s2026", run, kill={
        "stage": f"train:step:{_rt_steps(cx.host) + 1}", "occurrence": 1,
        "mode": "simulate"})
    cx.ok("interrupted" in out and len(_commits(_rt_dir(run))) == 1,
          "F14: run interrupted in epoch 2 (one commit) for the tamper case")

    def adam(o):
        k = sorted(o["adam"]["state"])[0]
        o["adam"]["state"][k]["exp_avg"] = o["adam"]["state"][k][
            "exp_avg"] + 1.0
    _recommit(_rt_dir(run), adam)
    out = _rt_run(cx, "lam2_s2026", run, "resume")
    err = out.get("error") or {}
    cx.ok(err.get("code") == "RESUME_STATE_MISMATCH"
          and err.get("taxonomy") == "failed_validity"
          and "Adam-state hash" in err.get("message", ""),
          "F14: altered Adam state at resume -> failed_validity", detail=str(err))
    e = _ends(_rt_dir(run))
    cx.ok(len(e) == 1 and e[0]["stage"] == "resume_validation",
          "F14: resume-validation failure recorded as terminal")
    out = _rt_run(cx, "lam2_s2026", run, "resume")
    cx.ok((out.get("error") or {}).get("code") == "RESUME_RUN_ENDED",
          "F14: failed run never resumed")
    _rt_drop(run)
    del torch, T


def f13c(cx: Ctx) -> None:
    """F13 completion: data-order pairing across arms; other seed differs."""
    _rt_phase(cx)
    ref = _rt_ref(cx)
    out = _rt_run(cx, "lam0_s2026", "f13_lam0")
    cx.ok(out.get("result", {}).get("status") == "fixture_budget_end",
          "lam0 seed 2026 run completed")
    a, b = _commits(ref), _commits(_rt_dir("f13_lam0"))
    cx.ok([a[e]["order_sha256"] for e in sorted(a)]
          == [b[e]["order_sha256"] for e in sorted(b)] and len(a) == RT_EPOCHS,
          "F13: identical per-epoch order hashes in both arms (same seed)")
    sa = [r["body"] for r in C.Ledger(ref / "ledger.jsonl").records()
          if r["kind"] == "session_start"][0]
    sb = [r["body"] for r in C.Ledger(_rt_dir("f13_lam0") / "ledger.jsonl")
          .records() if r["kind"] == "session_start"][0]
    ra = C.Ledger(ref / "ledger.jsonl").records()[0]["body"]["config"]
    rb = C.Ledger(_rt_dir("f13_lam0") / "ledger.jsonl").records()[0][
        "body"]["config"]
    cx.ok(ra["epoch0_model_state"] == rb["epoch0_model_state"]
          == _RT["e0"][2026] and sa["initial_rng_sha256"]
          == sb["initial_rng_sha256"],
          "F13: both arms start from the seed's epoch-0 state (verified "
          "before the first update) with identical initial RNG state")
    cx.ok(a[1]["losses"]["L_rec"] > 0 and b[1]["losses"]["L_rec"] == 0
          and a[1]["model_state_sha256"] != b[1]["model_state_sha256"],
          "F13: arms differ only by the objective (lam2 L_rec > 0, lam0 none)")
    out = _rt_run(cx, "lam0_s2027", "f13_s2027")
    c = _commits(_rt_dir("f13_s2027"))
    cx.ok(c[1]["order_sha256"] != a[1]["order_sha256"]
          and _RT["e0"][2027] != _RT["e0"][2026],
          "F13: seed 2027 has a different order and epoch-0 state")
    _rt_drop("f13_lam0", "f13_s2027")


def f21c(cx: Ctx) -> None:
    """F21 completion: training with vs without a preceding epoch-0
    evaluation (fresh processes) is bitwise identical."""
    _rt_phase(cx)
    ref = _rt_ref(cx)
    out = _rt_run(cx, "lam2_s2026", "f21_preeval",
                  pre_eval=_RT["e0_path"][2026])
    cx.ok("pre_eval_psnr" in out and out.get("result", {}).get(
        "status") == "fixture_budget_end",
        "F21: epoch-0 loaded and evaluated (shared validation) before "
        "training", detail=str(out.get("pre_eval_psnr")))
    _same_runs(cx, ref, _rt_dir("f21_preeval"),
               "F21: with vs without preceding epoch-0 evaluation")
    _rt_drop("f21_preeval")
    cx.record("F21 scope", "reconstruction-evaluator isolation pending I1.7")


def f08r(cx: Ctx) -> None:
    """F08 retention on the real trainer (reference run)."""
    _rt_phase(cx)
    ref = _rt_ref(cx)
    st = C.ArtifactStore(ref / "ckpt", C.Ledger(ref / "ledger.jsonl"),
                         ref / "_incomplete")
    final = st.committed()
    cm = _commits(ref)
    sel = cm[RT_EPOCHS]["selection"]
    cx.ok(final["milestone_003"]["name"] == "weights_e003.pt"
          and final["resume"]["name"] == f"resume_e{RT_EPOCHS:03d}.pt"
          and final["best_psnr"]["name"] == f"weights_e{sel['best_psnr']['epoch']:03d}.pt"
          and final["best_nll"]["name"] == f"weights_e{sel['best_nll']['epoch']:03d}.pt",
          "F08: milestone 3, last resume, best-PSNR and best-NLL files = "
          "selection records")
    keep = {v["name"] for v in final.values()}
    listing = {q.name for q in (ref / "ckpt").iterdir()}
    cx.ok(listing == keep | {n + ".sha256" for n in keep},
          "F08: directory == retained set")
    cx.ok(all(len({b["retained"][r]["name"] for r in b["roles_this_epoch"]})
              <= 1 for b in cm.values()),
          "F08: roles of one epoch share one weights file")
    ev = _RT["ref_out"]["events"]
    before = [n for s_, n in ev if s_ == "commit:before_ledger"]
    after = [n for s_, n in ev if s_ == "commit:after_ledger"]
    nres = [sum(x.startswith("resume_") and x.endswith(".pt") for x in n)
            for n in before]
    nres_a = [sum(x.startswith("resume_") and x.endswith(".pt") for x in n)
              for n in after]
    cx.ok(nres == [1] + [2] * (RT_EPOCHS - 1) and nres_a == nres,
          "F08: new resume staged beside the old one; old deleted only "
          "after the durable commit", detail=f"{nres} {nres_a}")


def kil(cx: Ctx) -> None:
    """Addition A (focused): SIGKILL at key recovery boundaries."""
    _rt_phase(cx)
    ref = _rt_ref(cx)
    for run, stage, occ, label in (
            ("kil_first", "train:step:1", 1, "before the first commit"),
            ("kil_commit2", "commit:after_ledger", 2,
             "epoch-2 commit durable, superseded files not yet deleted")):
        _rt_run(cx, "lam2_s2026", run, expect_kill=True,
                kill={"stage": stage, "occurrence": occ, "mode": "sigkill"})
        out = _rt_run(cx, "lam2_s2026", run, "resume")
        cx.ok(out.get("result", {}).get("status") == "fixture_budget_end",
              f"KIL {label}: resumed to the end")
        if run == "kil_first":
            cx.ok("restart_uncommitted" in _kinds(_rt_dir(run)),
                  "KIL before the first commit: explicit restart recorded")
        _same_runs(cx, ref, _rt_dir(run), f"KIL {label}")
        _rt_drop(run)
    run = "kil_final"
    _rt_run(cx, "lam2_s2026", run, expect_kill=True,
            kill={"stage": "run:before_run_end", "occurrence": 1,
                  "mode": "sigkill"})
    cx.ok(not _ends(_rt_dir(run)) and max(_commits(_rt_dir(run))) == RT_EPOCHS,
          "KIL after the final commit: commit durable, no run_end")
    out = _rt_run(cx, "lam2_s2026", run, "resume")
    cx.ok((out.get("error") or {}).get("code") == "RESUME_RUN_ENDED",
          "KIL after the final commit: resume refuses (run ended)")
    e = _ends(_rt_dir(run))
    cx.ok(len(e) == 1 and e[0]["recovered"] is True
          and e[0]["status"] == "fixture_budget_end",
          "KIL after the final commit: run_end recorded on recovery")
    _same_runs(cx, ref, _rt_dir(run), "KIL after the final commit "
               "(no retraining)")
    _rt_drop(run)


def rngc(cx: Ctx) -> None:
    """Addition C: training RNG consumption (training portion of F04)."""
    import random as _random
    _rt_phase(cx)
    ref = _rt_ref(cx)
    seed, n = 2026, len(_rt_indices(cx.host)[0])
    exp, g = [], np.random.default_rng(seed)
    exp.append(g.bit_generator.state)
    for _ in range(RT_EPOCHS):
        g.permutation(n)
        exp.append(json.loads(json.dumps(g.bit_generator.state)))
    d = _RT["ref_out"]["draws"]
    cx.ok(len(d) == RT_EPOCHS and all(x["n"] == n for x in d),
          "C: exactly one permutation draw per epoch", detail=str(len(d)))
    cx.ok(all(d[e]["before"] == json.loads(json.dumps(exp[e]))
              and d[e]["after"] == exp[e + 1] for e in range(RT_EPOCHS)),
          "C: shuffle-generator state after epoch e = fresh generator after "
          "e draws; nothing else consumes it")
    rd = (_RT.get("f09_mid2_out") or {}).get("draws")
    cx.ok(rd is not None and rd[0]["before"] == exp[1]
          and len(rd) == RT_EPOCHS - 1,
          "C: resumed session continues from the restored generator state "
          "(F09 mid-epoch-2 resume; run F09 before C)", "ORDER_OR_STATE")
    st = _final_state(ref)[2]["rng"]
    py = _random.Random(seed).getstate()
    npst = np.random.RandomState(seed).get_state()
    cx.ok(st["python"] == [py[0], list(py[1]), py[2]]
          and st["numpy"] == [npst[0], npst[1].tolist(), int(npst[2]),
                              int(npst[3]), float(npst[4])],
          "C: Python and NumPy global RNGs not consumed by training or "
          "validation (registered mask generators are separate)")
    cx.record("C torch-CPU RNG", "consumed state carried in the resume "
              "state (compared bitwise by F09/REP)")


# ---------------------------------------------------------------------------
# I1.7(a) (SEQREF-MRI-V05LTI17 v0.2): F10V (strict input verification +
# negative tests), F10 (reproduction, V02 checkpoints only, V02G backend,
# separate process), F11 (schedule/reuse/readouts), F22R (readout cap
# flags), CPB (addition B cost probe), STB (addition C stability)
# ---------------------------------------------------------------------------
def _write_json_with_sidecar(p: Path, obj) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True))
    sha = C.sha256_file(p)
    Path(str(p) + ".sha256").write_text(f"{sha}  {p.name}\n")
    return sha


def _residecar(p: Path) -> None:
    Path(str(p) + ".sha256").write_text(f"{C.sha256_file(p)}  {p.name}\n")


def _syn_f10_set(root: Path) -> tuple:
    """Synthetic, self-consistent F10 evidence set + registry (fixture-only;
    real V02G constants cannot be carried by synthetic content)."""
    from seqref_mri.scripts import v02_manifests as v02m
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.src.preflight_io import canonical_hash
    reg = json.loads(json.dumps(EV.REG_F10))
    anchors = {"step0": {"psnr": 30.5, "nmse_u": 1.01},
               "final": {"psnr": 30.25, "nmse_u": 1.02}}
    reg["anchors"] = anchors
    md = root / reg["manifest_dir_rel"]
    hold_entries = [{"file": f"syn/vol{j:03d}.h5", "slice_index": j % 7}
                    for j in range(199)]
    hold = {"schema": "seqref-v02-manifest/1", "kind": "holdout",
            "n_volumes": 199, "entries": hold_entries}
    hold["manifest_sha256"] = v02m.manifest_sha256(hold)
    pos = [int(x) for x in v02m.d3_monitor_positions()]
    d3e = [{"holdout_position": q, "draw_rank": r, **hold_entries[q]}
           for r, q in enumerate(pos)]
    d3 = {"schema": "seqref-v02-manifest/1", "kind": "d3_monitor", "n": 32,
          "entries": d3e}
    d3["manifest_sha256"] = v02m.manifest_sha256(d3)
    reg["holdout_manifest_file_sha256"] = _write_json_with_sidecar(
        md / "v02_holdout_manifest.json", hold)
    _write_json_with_sidecar(md / "v02_d3_monitor_manifest.json", d3)
    reg["holdout_manifest_sha256"] = hold["manifest_sha256"]
    reg["d3_manifest_sha256"] = d3["manifest_sha256"]
    parents = {}
    for name, body in (("p0s", {"x": 1}), ("p4", {"x": 2}), ("implb", {"x": 3}),
                       ("v02h", {"schema": "seqref-v02h-selftest-facts/1",
                                 "outcome": {"label": "OUTCOME_A_R2_NO_OP"}})):
        p = root / reg[f"{name}_rel"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(body))
        parents[name] = reg[f"{name}_sha256"] = C.sha256_file(p)
    rr = root / reg["run_root_rel"]
    rr.mkdir(parents=True, exist_ok=True)
    cks = []
    for step in (0, 1086, 2172, 3258):
        f = rr / f"v02_ckpt_step{step}.pt"
        f.write_bytes(f"synthetic checkpoint {step}".encode())
        cks.append({"step": step, "file": f.name,
                    "file_sha256": C.sha256_file(f),
                    "state_sha256": hashlib.sha256(
                        f"state{step}".encode()).hexdigest()})
    _write_json_with_sidecar(root / reg["train_record_rel"], {
        "schema": "seqref-v02-train-record/1", "steps": 3258,
        "checkpoints": cks})
    reg["bank_sha256"] = "1" * 64
    reg["bank_manifest_sha256"] = "2" * 64
    facts = {"schema": reg["facts_schema"], "code": {"commit": reg[
        "facts_commit"]},
        "reproduction": {"d3_pm": {s: {"psnr": {"anchor": anchors[s]["psnr"]},
                                       "nmse_u": {"anchor": anchors[s][
                                           "nmse_u"]}}
                                   for s in ("step0", "final")}},
        "frozen_inputs": {
            "holdout_manifest_sha256": hold["manifest_sha256"],
            "d3_subset_manifest_sha256": d3["manifest_sha256"],
            "pm_bank": {"bank_sha256": reg["bank_sha256"],
                        "manifest_sha256": reg["bank_manifest_sha256"]},
            "v02h_evidence": {"sha256": parents["v02h"]},
            "checkpoint_state_sha256_verified": {
                str(c["step"]): c["state_sha256"] for c in cks
                if c["step"] in (0, 3258)}},
        "run": {"argv": ["synthetic"]},
        "timing": {"wall_clock_s": 1.0},
        "r1_sampling_diagnostics": {"status": "diagnostic-only"}}
    # historical procedure: hash without V02G's list; the scope field then
    # records the GENERIC preflight list (the real two-list inconsistency)
    excl = list(EV.REG_F10["semantic_excluded"])
    inc = sorted(k for k in facts if k not in excl)
    facts["semantic_sha256"] = canonical_hash({k: facts[k] for k in inc})
    facts["semantic_scope"] = {"included_keys": inc, "excluded": [
        "run", "code.third_party", "runtime_seconds", "peak_memory_bytes",
        "paths", "timestamps", "artifact_sha256", "semantic_sha256"]}
    reg["facts_semantic_sha256"] = facts["semantic_sha256"]
    reg["facts_sha256"] = _write_json_with_sidecar(root / reg["facts_rel"],
                                                   facts)
    reg["v02_facts_sha256"] = _write_json_with_sidecar(
        root / reg["v02_facts_rel"], {"secondary_monitoring": {"d3_monitor": {
            "pm": {s: {"pm_psnr_mean": anchors[s]["psnr"],
                       "pm_nmse_u_mean": anchors[s]["nmse_u"]}
                   for s in ("step0", "final")}}}})
    bank = {"bank": np.zeros((128, 4), np.float32), "bank_sha256":
            reg["bank_sha256"], "manifest_sha256": reg["bank_manifest_sha256"]}
    return reg, bank


def _edit_json(p: Path, fn, residecar: bool = True) -> None:
    obj = json.loads(p.read_text())
    fn(obj)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True))
    if residecar:
        _residecar(p)


def _f10_negatives(cx: Ctx, root: Path, reg: dict, kw: dict, label: str,
                   synthetic: bool) -> int:
    """Each evidence category altered in a fresh copy -> refused."""
    from seqref_mri.scripts import v05lt_eval as EV
    n = 0

    def run(name, code, mutate, reg_mut=None, kw_mut=None):
        nonlocal n
        with tempfile.TemporaryDirectory() as t:
            r2 = Path(t) / "r"
            shutil.copytree(root, r2, symlinks=True)
            rg = json.loads(json.dumps(reg))
            k2 = dict(kw)
            if mutate:
                mutate(r2, rg)
            if reg_mut:
                reg_mut(rg)
            if kw_mut:
                kw_mut(k2)
            cx.raises(code, lambda: EV.verify_f10_inputs(
                EV.f10_paths(r2, rg), rg, **k2),
                f"{label}: {name} refused")
            n += 1

    def p(r, k):
        return r / reg[k]
    f, vf = "facts_rel", "v02_facts_rel"
    run("facts bytes vs sidecar", "F10_FACTS_SIDECAR",
        lambda r, g: p(r, f).write_text(p(r, f).read_text() + " "))
    run("facts hash vs registered", "F10_FACTS_MISMATCH", None,
        lambda g: g.__setitem__("facts_sha256", "0" * 64))
    run("facts semantic content", "F10_FACTS_SEMANTIC",
        lambda r, g: (_edit_json(p(r, f), lambda o: o["frozen_inputs"]
                                 .__setitem__("extra", 1)),
                      g.__setitem__("facts_sha256", C.sha256_file(p(r, f)))))
    def rehash(r, g, edit):
        from seqref_mri.src.preflight_io import canonical_hash

        def e(o):
            edit(o)
            inc = o["semantic_scope"]["included_keys"]
            o["semantic_sha256"] = canonical_hash({k: o[k] for k in inc})
            g["facts_semantic_sha256"] = o["semantic_sha256"]
        _edit_json(p(r, f), e)
        g["facts_sha256"] = C.sha256_file(p(r, f))
    run("included-key list altered (timing added, re-hashed)",
        "F10_FACTS_SEMANTIC", lambda r, g: rehash(r, g, lambda o: o[
            "semantic_scope"].__setitem__("included_keys", sorted(
                o["semantic_scope"]["included_keys"] + ["timing"]))))
    run("broadened exclusion (code dropped, re-hashed consistently)",
        "F10_FACTS_SEMANTIC", lambda r, g: rehash(r, g, lambda o: o[
            "semantic_scope"].__setitem__("included_keys", [
                k for k in o["semantic_scope"]["included_keys"]
                if k != "code"])))
    run("facts provenance commit", "F10_FACTS_PROVENANCE", None,
        lambda g: g.__setitem__("facts_commit", "f" * 40))
    run("anchor value", "F10_ANCHOR_MISMATCH", None,
        lambda g: g["anchors"]["final"].__setitem__("psnr", 30.0))
    run("v02_facts hash", "F10_V02FACTS_MISMATCH", None,
        lambda g: g.__setitem__("v02_facts_sha256", "0" * 64))
    run("v02_facts bytes vs sidecar", "F10_V02FACTS_SIDECAR",
        lambda r, g: (p(r, vf).write_text(p(r, vf).read_text() + " "),
                      g.__setitem__("v02_facts_sha256",
                                    C.sha256_file(p(r, vf)))))
    md = "manifest_dir_rel"
    hp = lambda r: r / reg[md] / "v02_holdout_manifest.json"  # noqa: E731
    dp = lambda r: r / reg[md] / "v02_d3_monitor_manifest.json"  # noqa: E731
    run("holdout manifest bytes vs sidecar", "F10_MANIFEST_SIDECAR",
        lambda r, g: hp(r).write_text(hp(r).read_text() + " "))
    run("holdout manifest file vs registered", "F10_MANIFEST_FILE_MISMATCH",
        None, lambda g: g.__setitem__("holdout_manifest_file_sha256", "0" * 64))
    run("D3 manifest content vs stored field", "F10_MANIFEST_CONTENT",
        lambda r, g: _edit_json(dp(r), lambda o: o["entries"][0].__setitem__(
            "slice_index", 99)))
    def consistent_d3_change(r, g):
        from seqref_mri.scripts import v02_manifests as v02m

        def f(o):
            o["entries"][0]["slice_index"] = 98
            o.pop("manifest_sha256")
            o["manifest_sha256"] = v02m.manifest_sha256(o)
        _edit_json(dp(r), f)
    run("D3 stored field (self-consistent content) vs registered",
        "F10_MANIFEST_MISMATCH", consistent_d3_change)
    tp = "train_record_rel"
    run("train record bytes vs sidecar", "F10_TRAIN_RECORD_SIDECAR",
        lambda r, g: p(r, tp).write_text(p(r, tp).read_text() + " "))
    run("train record state hash vs pinned facts", "F10_TRAIN_RECORD_MISMATCH",
        lambda r, g: _edit_json(p(r, tp), lambda o: [
            c.__setitem__("state_sha256", "0" * 64) for c in o["checkpoints"]
            if c["step"] == 3258]))
    for name in ("p0s", "p4", "implb", "v02h"):
        run(f"parent {name} file bytes vs approved hash", "F10_PARENT_MISMATCH",
            lambda r, g, name=name: p(r, f"{name}_rel").write_bytes(
                p(r, f"{name}_rel").read_bytes() + b" "))

    def v02h_content(r, g):
        from seqref_mri.src.preflight_io import canonical_hash
        p(r, "v02h_rel").write_text(json.dumps(
            {"schema": "seqref-v02h-selftest-facts/1",
             "outcome": {"label": "OTHER"}}))
        new = C.sha256_file(p(r, "v02h_rel"))
        g["v02h_sha256"] = new

        def edit(o):
            o["frozen_inputs"]["v02h_evidence"]["sha256"] = new
            inc = o["semantic_scope"]["included_keys"]
            o["semantic_sha256"] = canonical_hash({k: o[k] for k in inc})
            g["facts_semantic_sha256"] = o["semantic_sha256"]
        _edit_json(p(r, f), edit)
        g["facts_sha256"] = C.sha256_file(p(r, f))
    run("V02H content (all hashes consistent)", "F10_PARENT_CONTENT",
        v02h_content, None, lambda k: k.__setitem__("cross_check_code", False))
    run("bank vs registered", "F10_BANK_MISMATCH", None, None,
        lambda k: k.__setitem__("bank_override", {
            "bank": np.zeros((128, 4), np.float32), "bank_sha256": "9" * 64,
            "manifest_sha256": "9" * 64}))
    run("registry key missing", "F10_REGISTRY_INCOMPLETE", None,
        lambda g: g.pop("p4_sha256"))
    run("registry anchor missing", "F10_REGISTRY_INCOMPLETE", None,
        lambda g: g["anchors"]["step0"].pop("nmse_u"))
    if synthetic:
        run("checkpoint bytes vs train record", "F10_CHECKPOINT_MISMATCH",
            lambda r, g: (r / reg["run_root_rel"] / "v02_ckpt_step3258.pt")
            .write_bytes(b"altered"), None,
            lambda k: k.__setitem__("hash_checkpoints", True))
    return n


def f10v(cx: Ctx) -> None:
    """F10 strict input verification (acceptance requirements 1, 3, 4, 5).
    Input-verification PASS is NOT F10 reproduction PASS."""
    from seqref_mri.scripts import v05lt_eval as EV
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "syn"
        reg, bank = _syn_f10_set(root)
        kw = {"cross_check_code": False, "hash_checkpoints": True,
              "v02g_contracts": False, "bank_override": bank}
        out = EV.verify_f10_inputs(EV.f10_paths(root, reg), reg, **kw)
        cx.ok(out["result"] == "INPUT_VERIFICATION_PASS"
              and out["facts_semantic"]["known_inconsistency"],
              "synthetic evidence with the real two-list scope verifies "
              "under the historical procedure; inconsistency recorded",
              detail=str(out["facts_semantic"]["known_inconsistency"]))
        n = _f10_negatives(cx, root, reg, kw, "synthetic", True)
        cx.ok(n >= 24, "every evidence category altered and refused",
              detail=str(n))
        _procedure_tests(cx, Path(t))
    cx.ok(EV.check_registry_against_code(EV.REG_F10) is not None,
          "registry equals the V02G v0.2 constants and loader pins")
    bad = json.loads(json.dumps(EV.REG_F10))
    bad["anchors"]["step0"]["psnr"] = 30.0
    cx.raises("F10_REGISTRY_CONFLICT", lambda: EV.check_registry_against_code(
        bad), "registry differing from the code constants refused")
    if not cx.host:
        return
    cx.phase = "host"
    paths = EV.f10_paths(REPO_ROOT)
    out = EV.verify_f10_inputs(paths, procedure_repo=REPO_ROOT)
    cx.ok(out["result"] == "INPUT_VERIFICATION_PASS",
          "real F10 inputs verified (facts, manifests, train record, both "
          "checkpoints hashed, parents, bank)")
    cx.record("F10 input verification", {k: out[k] for k in (
        "facts", "facts_semantic", "procedure", "manifests", "train_record",
        "parents", "bank")})
    with tempfile.TemporaryDirectory() as t:
        mirror = Path(t) / "real"
        for k in ("facts_rel", "v02_facts_rel", "train_record_rel", "p0s_rel",
                  "p4_rel", "implb_rel", "v02h_rel"):
            for suffix in ("", ".sha256"):
                src = REPO_ROOT / (EV.REG_F10[k] + suffix)
                if src.is_file():
                    dst = mirror / (EV.REG_F10[k] + suffix)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
        mdst = mirror / EV.REG_F10["manifest_dir_rel"]
        mdst.mkdir(parents=True, exist_ok=True)
        for name in ("v02_holdout_manifest.json", "v02_d3_monitor_manifest.json"):
            for suffix in ("", ".sha256"):
                shutil.copy2(REPO_ROOT / EV.REG_F10["manifest_dir_rel"] /
                             (name + suffix), mdst / (name + suffix))
        rr = mirror / EV.REG_F10["run_root_rel"]
        for f in (REPO_ROOT / EV.REG_F10["run_root_rel"]).glob("v02_ckpt_step*.pt"):
            (rr / f.name).symlink_to(f)
        kw = {"cross_check_code": True, "hash_checkpoints": False,
              "v02g_contracts": True}
        out = EV.verify_f10_inputs(EV.f10_paths(mirror), EV.REG_F10, **kw)
        cx.ok(out["result"] == "INPUT_VERIFICATION_PASS",
              "mirror of the real evidence verifies before alteration")
        # registry-vs-code is tested above; off here so each alteration
        # reaches its own check
        kw["cross_check_code"] = False
        n = _f10_negatives(cx, mirror, EV.REG_F10, kw, "real copy", False)
        cx.ok(n >= 23, "real evidence copies: each category altered and "
              "refused", detail=str(n))


def _procedure_tests(cx: Ctx, tmp: Path) -> None:
    """historical_semantic_procedure on a temporary git repository."""
    from seqref_mri.scripts import v05lt_eval as EV
    repo = tmp / "procrepo"
    for rel, _ in EV.PROCEDURE_SOURCES:
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dst)
    for args in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.name=f", "-c", "user.email=f@x", "commit", "-q",
                  "-m", "procedure"]):
        _git(repo, *args)
    head = _git(repo, "rev-parse", "HEAD")
    r = EV.historical_semantic_procedure(repo, head)
    cx.ok(r["semantic_excluded"] == EV.REG_F10["semantic_excluded"]
          and len(r["segments_identical_to_working_tree"]) == 5,
          "procedure from the commit: exclusion list = registry; 5 source "
          "segments identical to the working tree")
    bad = json.loads(json.dumps(EV.REG_F10))
    bad["semantic_excluded"] = bad["semantic_excluded"] + ["metrics"]
    cx.raises("F10_PROCEDURE_MISMATCH", lambda: EV.historical_semantic_procedure(
        repo, head, bad), "registry exclusion list != commit list refused")
    cx.raises("F10_PROCEDURE_MISSING", lambda: EV.historical_semantic_procedure(
        repo, "0" * 40), "unknown commit refused")
    src = repo / EV.PROCEDURE_SOURCES[0][0]
    src.write_text(src.read_text().replace('"r1_sampling_diagnostics",',
                                           '"r1_sampling_diagnostics", "x",', 1))
    cx.raises("F10_PROCEDURE_DRIFT", lambda: EV.historical_semantic_procedure(
        repo, head), "working-tree exclusion list drifted from the commit "
        "refused")
    io = repo / EV.PROCEDURE_SOURCES[2][0]
    _git(repo, "checkout", "-q", "--", EV.PROCEDURE_SOURCES[0][0])
    io.write_text(io.read_text().replace("sort_keys=True", "sort_keys=False",
                                         1))
    cx.raises("F10_PROCEDURE_DRIFT", lambda: EV.historical_semantic_procedure(
        repo, head), "hashing function drifted from the commit refused")


def _f12_host(EV, states, model, bank) -> dict:
    """F12 checks inside the evaluator child on real validation states."""
    class _Cx:
        def __init__(self):
            self.failed, self.n, self.rec = [], 0, {}

        def ok(self, cond, label, code="CHECK_FAILED", detail=""):
            self.n += 1
            if not cond:
                logger.error("[F12 host] %s: %s %s", code, label, detail)
                self.failed.append(label)

        def record(self, label, value):
            self.rec[label] = value
    cx = _Cx()
    out = _f12_checks(cx, EV, states, model, bank, "host")
    cal = out["cal"]
    return {"checks": cx.n, "checks_failed": len(cx.failed),
            "failed": cx.failed,
            "record": {"floors": cal["floors"],
                       "tolerances": cal["tolerances"],
                       "argmax_identity": cal["argmax_identity"],
                       "provenance": cal["provenance"],
                       "identity_summary": out["summary"],
                       "n_slices": len(states)}}


def _e0v_host(EV, data_root: Path) -> dict:
    """Host E0V: temporary epoch-0 checkpoint (seed 2026), evaluated from
    the saved file on a registered fixture subset: case 0 and case 1
    midpoints (registered case manifest) + validation positions 0-3."""
    import torch
    from seqref_mri.scripts import v05lt_eligibility as E
    from seqref_mri.scripts import v05lt_epoch0 as E0
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.src import free_flow_runtime as ffr
    from seqref_mri.tdiag import estimators as est
    p = C.Paths(repo=REPO_ROOT)
    sp, cp = p.inputs / "v05lt_split_manifest.json", \
        p.inputs / "v05lt_case_manifest.json"
    sdoc = C.load_split_manifest(sp, C.REGISTERED_INPUTS[sp.name])
    C.verify_file(cp, C.REGISTERED_INPUTS[cp.name], "CASE")
    case_doc = json.loads(cp.read_bytes())
    _, vv = C.build_split_views(data_root, sdoc,
                                C.holdout_ids(p.v02_manifests))
    s_ref = C.verify_p0s(p.p0s)
    p4 = ffr.load_p4s2_parent(str(p.repo / P4_REL))
    cases = case_doc["cases"][:2]
    keys = {(c["file"], c["slice_index"]) for c in cases}
    pos = {vv.entry(i): i for i in range(len(vv))}
    mid = [pos[(c["file"], c["slice_index"])] for c in cases]
    all_idx = sorted(set(mid) | {0, 1, 2, 3})
    recs = E.compute_records(vv, s_ref, all_idx)
    elig = M.eligibility_index(E.build_manifest(
        recs, {"validation": sdoc["validation"]}, case_doc, {"p": "fixture"}))
    C.require(all(elig[k]["midpoint"] for k in keys), "E0V_CASES",
              "registered cases must be midpoints")
    data = {"vv": vv, "p4": p4, "s_ref": s_ref, "eligibility": elig}
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        e0 = E0.create(tmp / "epoch0", 2026, REPO_ROOT)
        ck = tmp / "epoch0" / "seed_2026" / "ckpt" / E0.ckpt_name(2026)
        model = E0.load_epoch0(ck, e0["model_state_sha256"], 2026)
        sd0 = C.model_state_sha256(model.state_dict())
        sts = EV.build_states(vv, mid, p4, s_ref)
        tol = EV.calibrate_dc(sts)["tolerances"]
        cnt = {}
        rec, pan = EV.epoch0_evaluate(model, data, est.z_diag_bank()["bank"],
                                      tol, cases, all_indices=all_idx,
                                      mid_indices=mid, decode_counter=cnt)
        same = C.model_state_sha256(model.state_dict()) == sd0
        out = tmp / "out" / "eval"
        shas = EV.write_epoch0_eval(out, rec, pan, {"fixture": True})
        try:
            EV.write_epoch0_eval(tmp / "results" / "v05lt" / "epoch0" /
                                 "seed_2026" / "eval", rec, pan, {})
            refused = False
        except C.V05LTError as exc:
            logger.warning("expected refusal: %s", exc.code)
            refused = exc.code == "EPOCH0_EVAL_GATE_REQUIRED"
    return {"decodes": cnt.get("primary", 0), "n_mid": len(mid),
            "n_all": len(all_idx), "cases": [c["case"] for c in cases],
            "state_unchanged": same, "rng_restored": rec["rng"]["restored"],
            "written": len(shas) == 1 + 2 * len(cases),
            "registered_refused": refused, "timing": rec["timing"],
            "torch_threads": torch.get_num_threads()}


def _eval_child(spec_path: str) -> None:
    """Spawned evaluator process (fresh interpreter): kind = f10 | cpb | stb |
    f12 | zf."""
    import resource
    import time as _time
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s[child]: %(message)s")
    import torch
    from seqref_mri.scripts import v05lt_eval as EV
    spec = json.loads(Path(spec_path).read_text())
    from seqref_mri.scripts import v05lt_epoch0 as _E0
    out = {"kind": spec["kind"], "torch_threads": torch.get_num_threads(),
           "backend": _E0.environment()}
    try:
        if spec["kind"] == "f10":
            out["f10"] = EV.f10_reproduce(REPO_ROOT, Path(spec["data_root"]))
        elif spec["kind"] == "e0v":
            out["e0v"] = _e0v_host(EV, Path(spec["data_root"]))
        elif spec["kind"] == "pev":
            out["pev"] = _pev_host(EV, Path(spec["data_root"]))
        elif spec["kind"] == "nld":
            out["nld"] = _nld_host(Path(spec["data_root"]))
        elif spec["kind"] == "prb":
            out["prb"] = _prb_child(spec)
        elif spec["kind"] == "hlb":
            out["hlb"] = _hlb_child(spec)
        elif spec["kind"] == "fwr":
            out["fwr"] = _fwr_child(EV, Path(spec["run"]), spec["host"],
                                    spec["data_root"])
        else:
            with tempfile.TemporaryDirectory() as t:
                data = _rt_data(spec["host"], REPO_ROOT, spec["data_root"],
                                Path(t))
                idx = spec["indices"]
                states = EV.build_states(data["vv"], idx, data["p4"],
                                         data["s_ref"])
                el = [data["eligibility"][data["vv"].entry(i)]["eligible"]
                      for i in idx]
                if spec.get("stub"):
                    from seqref_mri.scripts.v05lt_stub_train import build_stub
                    model = build_stub(2026)
                elif spec.get("epoch0"):
                    from seqref_mri.scripts import v05lt_epoch0 as E0
                    model = E0.load_epoch0(Path(spec["epoch0"]),
                                           spec["epoch0_state"], 2026)
                else:
                    from seqref_mri.src import free_flow_runtime as ffr
                    model = ffr.build_model(init_seed=2026)
                model.eval()
                bank = (np.random.Generator(np.random.PCG64(0))
                        .standard_normal((128, 13824)).astype(np.float32)
                        if spec.get("stub") else
                        __import__("seqref_mri.tdiag.estimators", fromlist=[
                            "z_diag_bank"]).z_diag_bank()["bank"])
                if spec["kind"] == "f12":
                    out["f12"] = _f12_host(EV, states, model, bank)
                elif spec["kind"] == "zf":
                    infos = [data["eligibility"][data["vv"].entry(i)]
                             for i in idx]
                    zr = [EV.zf_row(st, inf) for st, inf in zip(states,
                                                                infos)]
                    out["zf"] = {"n_rows": len(zr),
                                 "summary": EV.zf_summary(zr),
                                 "rows_have_identity": all(
                                     {"volume_id", "file", "slice_index"}
                                     <= set(r) for r in zr),
                                 "nmse_eligible": [
                        r["nmse_u"] for r in zr if r["nmse_u"] is not None],
                        "psnr": [r["psnr"] for r in zr],
                        "ssim": [r["ssim"] for r in zr]}
                elif spec["kind"] == "cpb":
                    t_c = _time.perf_counter()
                    cal = EV.calibrate_dc(states)
                    cal_s = _time.perf_counter() - t_c
                    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    per = []
                    infos = [data["eligibility"][data["vv"].entry(i)]
                             for i in idx]
                    for st, e, inf in zip(states, el, infos):
                        t0 = _time.perf_counter()
                        dec = EV.decode_bank(model, st, bank)
                        t1 = _time.perf_counter()
                        for k in EV.PREFIXES:
                            EV.r1_from_decodes(st, dec, k, e)
                        EV.r0_readout(model, st, e)
                        t2 = _time.perf_counter()
                        # probe overhead only: in evaluate_states these two
                        # images are formed once and shared with R0/R1
                        x0, _ = EV.r0_image(model, st)
                        x1, _ = EV.r1_image(st, dec)
                        t2b = _time.perf_counter()
                        EV.dc_readout(st, x0, cal["tolerances"], e,
                                      "validation_eligible")
                        EV.dc_readout(st, x1, cal["tolerances"], e,
                                      "validation_eligible")
                        t3 = _time.perf_counter()
                        EV.zf_row(st, inf)
                        t4 = _time.perf_counter()
                        per.append({"decode_128_s": t1 - t0,
                                    "recon_metrics_s": t2 - t1,
                                    "probe_r0_redecode_r1_image_s": t2b - t2,
                                    "dc_two_readouts_s": t3 - t2b,
                                    "zf_s": t4 - t3})
                    out["cpb"] = {"per_slice": per, "rss_before_kb": rss0,
                                  "calibration_s": cal_s,
                                  "calibration_slices": len(states),
                                  "rss_peak_kb": resource.getrusage(
                                      resource.RUSAGE_SELF).ru_maxrss}
                else:   # stb
                    rows = M_validate(model, data, idx)
                    r1 = [EV.r1_from_decodes(st, EV.decode_bank(model, st,
                                                                bank), 128, e)
                          for st, e in zip(states, el)]
                    out["stb"] = {"r0_psnr": [r["psnr"] for r in rows],
                                  "nll": [r["nll_per_dim"] for r in rows],
                                  "r1_psnr": [r["psnr"] for r in r1]}
    except C.V05LTError as exc:
        logger.warning("evaluator child raised %s (returned to parent)",
                       exc.code)
        out["error"] = {"code": exc.code, "message": str(exc)[:400]}
    except Exception as exc:  # noqa: BLE001 -- reported to the parent
        logger.exception("evaluator child failed")
        out["crash"] = f"{type(exc).__name__}: {exc}"
    Path(spec["out"]).write_text(json.dumps(out))


def M_validate(model, data, idx):
    from seqref_mri.scripts import v05lt_metrics as M
    return M.validate(model, data["vv"], data["eligibility"], data["p4"],
                      data["s_ref"], idx)


def _run_eval_child(cx: Ctx, tmp: Path, spec: dict, label: str) -> dict:
    spec = {"host": cx.host, "data_root": str(cx.data_root) if cx.host
            else None, **spec, "out": str(tmp / f"{label}.json")}
    sp = tmp / f"{label}.spec.json"
    sp.write_text(json.dumps(spec))
    p = multiprocessing.get_context("spawn").Process(target=_eval_child,
                                                     args=(str(sp),))
    p.start()
    code = join_bounded(p, "spawn")
    cx.ok(code == 0, f"{label}: child exit 0", "CHILD_FAILED", f"exit {code}")
    out = json.loads(Path(spec["out"]).read_text())
    cx.ok("crash" not in out, f"{label}: no unexpected error", "CHILD_CRASHED",
          out.get("crash", ""))
    return out


def f10(cx: Ctx) -> None:
    """F10 numerical reproduction (separate from input verification).
    Synthetic: the V05LT R1 assembly equals V02G v0.2 _r1_measure exactly
    (stub model, synthetic states). Host: D3-subset PM anchors on the V02
    checkpoints ONLY, V02G deterministic backend, fresh process."""
    from seqref_mri.scripts import v02_readout_diag as g
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts.v05lt_stub_train import build_stub
    with tempfile.TemporaryDirectory() as t:
        data = _syn_train_data(Path(t))
        idx = [0, 1, 2]
        states = EV.build_states(data["vv"], idx, data["p4"], data["s_ref"])
        model = build_stub(2026)
        model.eval()
        bank = np.random.Generator(np.random.PCG64(0)).standard_normal(
            (128, 13824)).astype(np.float32)
        env = g._env()
        same = 0
        for st in states:
            ours = EV.r1_from_decodes(st, EV.decode_bank(model, st, bank), 128,
                                      True, "v02_holdout_all")
            ref = g._r1_measure(env, model, st, bank, "fixture")
            same += ours["psnr"] == ref["psnr"] and ours["nmse_u"] == ref[
                "nmse_u"]
        cx.ok(same == len(states), "V05LT R1 = V02G v0.2 _r1_measure "
              "bit-identical (PSNR, NMSE_u) on synthetic states",
              detail=f"{same}/{len(states)}")
    if not cx.host:
        return
    cx.phase = "host"
    with tempfile.TemporaryDirectory() as t:
        out = _run_eval_child(cx, Path(t), {"kind": "f10"}, "f10")
    cx.ok("error" not in out, "F10 child completed without a typed refusal",
          "F10_FAILED", str(out.get("error")))
    r = out["f10"]
    cx.ok(r["result"] == "F10_REPRODUCTION_PASS" and r["backend"][
        "num_threads"] == 1 and r["backend"]["deterministic_algorithms"],
        "D3-subset PM anchors reproduced within 1e-3 dB / 1e-4 rel under "
        "the V02G deterministic backend (V02 checkpoints only)")
    cx.record("F10 reproduction", {"endpoints": r["endpoints"],
                                   "backend": r["backend"],
                                   "n_slices": r["n_slices"]})


def _stub_states(n=3):
    from seqref_mri.scripts import v05lt_eval as EV
    t = tempfile.TemporaryDirectory()
    data = _syn_train_data(Path(t.name))
    states = EV.build_states(data["vv"], list(range(n)), data["p4"],
                             data["s_ref"])
    return t, data, states


def f11(cx: Ctx) -> None:
    """Schedule, reuse and readout mechanics."""
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts.v05lt_stub_train import build_stub

    def ms(*a):
        return hashlib.sha256(repr(a).encode()).hexdigest()

    def run(seed, arm, sel=20, fin=40, sel_state=None, fin_state=None):
        return {"seed": seed, "arm": arm, "roles": {
            "epoch0": {"epoch": 0, "model_state": ms("e0", seed)},
            "epoch3": {"epoch": 3, "model_state": ms("e3", seed, arm)},
            "selected": {"epoch": sel, "model_state":
                         sel_state or ms("sel", seed, arm)},
            "final": {"epoch": fin, "model_state":
                      fin_state or ms("fin", seed, arm)}}}
    full = [run(s, a) for s in (2026, 2027, 2028) for a in ("lam0", "lam2")]
    p = EV.plan_campaign(full)
    cx.ok(p["n_tasks"] == 21 and p["decodes"] == 409728
          and p["prefix_extra_decodes"] == 0,
          "registered campaign without checkpoint coincidences: 21 "
          "checkpoints, 409,728 decodes (epoch 0 once per seed; B/C only at "
          "selected); coincidences reduce these totals",
          detail=f"{p['n_tasks']} {p['decodes']}")
    e0 = [t for t in p["tasks"] if t["epoch"] == 0]
    cx.ok(len(e0) == 3 and all(len(t["uses"]) == 2 and t["banks"] ==
                               ["primary"] for t in e0),
          "epoch 0 evaluated once per seed, shared by both arms")
    sel = [t for t in p["tasks"] if any(u["role"] == "selected"
                                        for u in t["uses"])]
    cx.ok(len(sel) == 6 and all(t["banks"] == ["primary", "B", "C"] and
                                t["prefixes"] == [16, 32, 64, 128]
                                for t in sel),
          "banks B/C and prefix stability only at selected checkpoints")
    same = run(2026, "lam0", sel=40, fin=40, sel_state=ms("x"),
               fin_state=ms("x"))
    q = EV.plan_campaign([same])
    t40 = [t for t in q["tasks"] if t["epoch"] == 40]
    cx.ok(len(t40) == 1 and {u["role"] for u in t40[0]["uses"]} ==
          {"selected", "final"} and q["decodes"] == (1 + 1 + 3) * 128 * 97,
          "selected == final: evaluated once, both roles share the result")
    s3 = run(2026, "lam0", sel=3, sel_state=ms("e3", 2026, "lam0"))
    q = EV.plan_campaign([s3])
    t3 = [t for t in q["tasks"] if t["epoch"] == 3]
    cx.ok(len(t3) == 1 and t3[0]["banks"] == ["primary", "B", "C"],
          "selected == epoch 3: one task carrying the selected banks")
    bad = run(2026, "lam0", sel=7, sel_state=ms("e3", 2026, "lam0"))
    cx.raises("PLAN_STATE_EPOCH_CONFLICT", lambda: EV.plan_campaign([bad]),
              "same model state at two epochs refused")
    miss = run(2026, "lam0")
    miss["roles"].pop("epoch3")
    cx.raises("PLAN_ROLES", lambda: EV.plan_campaign([miss]),
              "missing role refused")
    unpaired = [run(2026, "lam0"), run(2026, "lam2")]
    unpaired[1]["roles"]["epoch0"]["model_state"] = ms("other-e0")
    cx.raises("PLAN_EPOCH0_PAIRING", lambda: EV.plan_campaign(unpaired),
              "arms of one seed with different epoch-0 states refused")
    t, data, states = _stub_states()
    try:
        model = build_stub(2026)
        bank = np.random.Generator(np.random.PCG64(0)).standard_normal(
            (128, 13824)).astype(np.float32)
        cnt = {}
        rows = EV.evaluate_states(model, states, [True] * len(states),
                                  {"primary": bank}, EV.PREFIXES,
                                  decode_counter=cnt)
        cx.ok(cnt == {"primary": 128 * len(states)},
              "one 128-row decode per slice; prefixes add no decodes",
              detail=str(cnt))
        cx.ok(all(set(r["r1"]["primary"]) == {"16", "32", "64", "128"}
                  for r in rows), "prefix records 16/32/64/128 produced")
        st = states[0]
        dec = EV.decode_bank(model, st, bank)
        a = EV.r1_from_decodes(st, dec, 128)
        b = EV.r1_from_decodes(st, dec[:128].copy(), 128)
        cx.ok(a == b and rows[0]["r1"]["primary"]["128"] == a,
              "k = 128 equals the mean of all decodes (V02G order)")
        cx.ok(all(r["identity"] == s["identity"] for r, s in
                  zip(rows, states)),
              "R0 and R1 measured on the identical slice states and masks")
        cx.raises("D3_BANK_LAYOUT_UNEXPECTED",
                  lambda: EV.decode_bank(model, st, bank[:127]),
                  "bank != 128 rows refused")
        cx.raises("EVAL_PREFIX_INVALID",
                  lambda: EV.r1_from_decodes(st, dec, 48), "unregistered "
                  "prefix refused")
        el = EV.evaluate_states(model, states[:1], [False],
                                {"primary": bank}, (128,))
        cx.ok(el[0]["r0"]["nmse_u"] is None and el[0]["r1"]["primary"][
            "128"]["nmse_u"] is None and el[0]["r0"]["psnr"] is not None,
            "validation rule: NMSE_u only on eligible slices; PSNR/SSIM kept")
        poisoned = build_stub(2026, poison_after=None)
        with __import__("torch").no_grad():
            poisoned.W.fill_(float("nan"))
        cx.raises("EVAL_NON_FINITE", lambda: EV.decode_bank(poisoned, st,
                                                            bank),
                  "non-finite decode -> typed EVAL_NON_FINITE (item 18)")
        cx.raises("EVAL_LENGTH_MISMATCH", lambda: EV.evaluate_states(
            model, states[:2], [True], {"primary": bank}),
            "states/eligibility length mismatch refused (no silent loss)")
        model.train()
        EV.evaluate_states(model, states[:1], [True], {"primary": bank})
        cx.ok(model.training, "evaluate_states restores train mode")
        model.eval()
        EV.evaluate_states(model, states[:1], [True], {"primary": bank})
        cx.ok(not model.training, "evaluate_states restores eval mode")
        two = EV.evaluate_states(model, states[:1], [True],
                                 {"primary": bank, "B": bank}, EV.PREFIXES)
        cx.ok(set(two[0]["r1"]["primary"]) == {"16", "32", "64", "128"}
              and set(two[0]["r1"]["B"]) == {"128"},
              "prefix stability on the primary bank only; B/C report k = 128")
        bad_st = dict(st)
        bad_st["x_true_mag"] = st["x_true_mag"].clone()
        bad_st["x_true_mag"][0, 0] = float("nan")
        cx.raises("EVAL_NON_FINITE", lambda: EV.r1_from_decodes(bad_st, dec,
                                                                128),
                  "non-finite metric -> EVAL_NON_FINITE (metric call mapped)")
        cx.raises("EVAL_DATA_LOAD", lambda: EV.build_states(
            data["vv"], [len(data["vv"]) + 5], data["p4"], data["s_ref"]),
            "data loading/collation failure -> EVAL_DATA_LOAD")
    finally:
        t.cleanup()
    if not cx.host:
        return
    cx.phase = "host"
    with tempfile.TemporaryDirectory() as tt:
        out = _run_eval_child(cx, Path(tt), {"kind": "cpb", "indices": [0, 1]},
                              "f11_real")
    cx.ok("error" not in out and len(out["cpb"]["per_slice"]) == 2,
          "real model: 128-row decode, prefixes and R0 on 2 validation "
          "slices completed")


def f22r(cx: Ctx) -> None:
    """F22 readout part: R1/R0 records carry raw MSE and cap flags; a
    zero-error readout is capped at 120 dB and retained."""
    import torch
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts.v05lt_stub_train import build_stub
    t, data, states = _stub_states(1)
    try:
        model = build_stub(2026)
        bank = np.random.Generator(np.random.PCG64(0)).standard_normal(
            (128, 13824)).astype(np.float32)
        st = dict(states[0])
        dec = EV.decode_bank(model, st, bank)
        rec = EV.r1_from_decodes(st, dec, 128)
        cx.ok({"mse", "psnr_capped", "zero_error", "psnr", "ssim",
               "nmse_u"} <= set(rec), "R1 record: raw MSE, cap and zero-error "
              "flags, PSNR, SSIM, NMSE_u", detail=str(sorted(rec)))
        from seqref_mri.tdiag import estimators as est
        st["x_true_mag"] = est.image_from_u(st, dec.mean(axis=0)).abs()
        z = EV.r1_from_decodes(st, dec, 128)
        cx.ok(z["zero_error"] and z["psnr_capped"] and z["psnr"] == 120.0,
              "zero-error readout: capped 120 dB, flags set, score retained",
              detail=str((z["mse"], z["psnr"])))
        del torch
    finally:
        t.cleanup()


# ---------------------------------------------------------------------------
# I1.7(b): F12 (DC), F22Z (ZF part of F22), F23 (holdout NMSE_u rule)
# ---------------------------------------------------------------------------
def _patched(mod, name, fn):
    """Context manager: temporarily replace mod.name (fixture-only)."""
    import contextlib

    @contextlib.contextmanager
    def cm():
        old = getattr(mod, name)
        setattr(mod, name, fn)
        try:
            yield
        finally:
            setattr(mod, name, old)
    return cm()


def _f12_checks(cx: Ctx, EV, states, model, bank, label: str) -> dict:
    import torch
    cal = EV.calibrate_dc(states)
    tol = cal["tolerances"]
    cx.ok(all(0 < cal["floors"][k] < 1e-10 and
              tol[k] == EV.DC_TOL_FACTOR * cal["floors"][k]
              for k in EV.DC_MEASURES),
          f"{label}: floors finite, > 0, roundoff-level; tolerance = 10 x "
          "max floor", detail=str(cal["floors"]))
    pv = cal["provenance"]
    cx.ok(pv["n_slices"] == len(states) and pv["dtype"] == "complex128" and
          set(cal["argmax_identity"]) == set(EV.DC_MEASURES) and
          all(v > 0 for v in pv["min_denominators"].values()),
          f"{label}: provenance (dtype, FFT, mask, scaling, backend, "
          "denominators, arg-max identities) recorded")
    cx.ok(all(torch.equal(st["x_true_c"].abs(), st["x_true_mag"]) and
              torch.equal(torch.stack([st["cond_c"].real, st["cond_c"].imag]),
                          st["cond"][0]) for st in states),
          f"{label}: complex GT magnitude == TINY x_true_mag and complex "
          "conditioning == TINY cond (bitwise)")
    am1 = [EV.dc_corruption_tests(st, tol) for st in states]
    cx.ok(all(r["pre_dc_repair"]["pass"] and r["post_dc_detection"]["pass"]
              for r in am1), f"{label}: AM1 pre-DC repair and post-DC "
          "detection pass on every state",
          detail=str(am1[0]))
    cnt = {}
    rows = EV.evaluate_states(model, states, [True] * len(states),
                              {"primary": bank}, (128,),
                              decode_counter=cnt, dc_tol=tol)
    cx.ok(cnt == {"primary": 128 * len(states)} and
          all({"r0", "r1", "r0_dc", "r1_dc"} <= set(r) for r in rows) and
          all(r["identity"] == s["identity"] for r, s in zip(rows, states)),
          f"{label}: four readouts R0, R1, R0+DC, R1+DC on identical states; "
          "R1+DC reuses the R1 decodes (no extra decodes)", detail=str(cnt))
    summ = EV.dc_identity_summary(rows)
    cx.record(f"{label}: DC identity summary", summ)
    return {"cal": cal, "rows": rows, "summary": summ}


def f12(cx: Ctx) -> None:
    """DC calibration, tolerances, stops, AM1 tests 1 and 2, four readouts."""
    import torch
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts.v05lt_stub_train import build_stub
    t, data, states = _stub_states()
    try:
        model = build_stub(2026)
        bank = np.random.Generator(np.random.PCG64(0)).standard_normal(
            (128, 13824)).astype(np.float32)
        out = _f12_checks(cx, EV, states, model, bank, "synthetic")
        tol = out["cal"]["tolerances"]
        streamed = EV.calibrate_dc(EV.iter_states(
            data["vv"], [0, 1, 2], data["p4"], data["s_ref"], chunk=2))
        cx.ok(streamed["floors"] == out["cal"]["floors"] and
              streamed["argmax_identity"] == out["cal"]["argmax_identity"],
              "streamed calibration (iter_states, bounded memory) == "
              "in-memory calibration")
        cx.raises("DC_CALIBRATION_EMPTY", lambda: EV.calibrate_dc(iter(())),
                  "empty calibration set refused")
        r1 = out["rows"][0]
        dec = EV.decode_bank(model, states[0], bank)
        cx.ok(r1["r1"]["primary"]["128"] == EV.r1_from_decodes(
            states[0], dec, 128), "R1 unchanged when DC readouts are added")
        x1, _ = EV.r1_image(states[0], dec)
        m, y = EV._dc_mask(states[0]), EV.y_norm(states[0])
        xd = EV.hard_dc(x1, y, m)
        from seqref_mri.src import fastmri_data as fdm
        k_d, k_1 = fdm.fft2c(xd), fdm.fft2c(x1)
        cx.ok(float((m * (k_d - y)).abs().max()) <= tol["a"] *
              float((m * y).abs().max()) and
              torch.allclose((1 - m) * k_d, (1 - m) * k_1, rtol=0,
                             atol=1e-12),
              "hard DC keeps unmeasured coefficients and replaces measured "
              "ones with y / amax (complex image; magnitude afterwards)")
        cx.raises("DC_DTYPE", lambda: EV.hard_dc(x1.to(torch.complex64), y,
                                                 m),
                  "non-evaluation dtype refused (no silent cast)")
        z = dict(states[0])
        z["x_true_c"] = torch.zeros_like(states[0]["x_true_c"])
        cx.raises("DC_ZERO_DENOMINATOR", lambda: EV.calibrate_dc([z]),
                  "zero denominator -> explicit stop (no epsilon)")
        n = dict(states[0])
        n["x_true_c"] = states[0]["x_true_c"].clone()
        n["x_true_c"][3, 3] = complex(float("nan"), 0.0)
        cx.raises("DC_NON_FINITE", lambda: EV.calibrate_dc([n]),
                  "non-finite measure -> explicit stop")
        zero = {k: 0.0 for k in EV.DC_MEASURES}
        with _patched(EV, "dc_measures", lambda *a, **k: dict(zero)):
            cx.raises("DC_FLOOR_INVALID", lambda: EV.calibrate_dc(states),
                      "zero floor -> DC_FLOOR_INVALID (fixture patch of "
                      "dc_measures; disclosed)")
        inf = {k: math.inf for k in EV.DC_MEASURES}
        with _patched(EV, "dc_measures", lambda *a, **k: dict(inf)):
            cx.raises("DC_FLOOR_INVALID", lambda: EV.calibrate_dc(states),
                      "non-finite floor -> DC_FLOOR_INVALID (fixture patch)")
        with _patched(EV, "hard_dc", lambda x, y, m: x):
            cx.raises("DC_REPAIR_FAILED", lambda: EV.dc_corruption_tests(
                states[0], tol), "AM1 test 1 fails on a DC that does not "
                "repair (fixture patch: DC = identity)")
        big = dict(tol, a=1.0)
        cx.raises("DC_REPAIR_FAILED", lambda: EV.dc_corruption_tests(
            states[0], big), "AM1 corruption below tol_a is not accepted as "
            "a repair test")
        cx.ok(EV.dc_identity(dict(tol), tol) and not EV.dc_identity(
            dict(tol, b=tol["b"] * 1.01), tol),
              "identity iff (a), (b), (c-rms), (c-max) all <= tolerance")
        cx.raises("DC_TOLERANCE_MISSING", lambda: EV.dc_identity(
            tol, {"a": 1.0}), "incomplete tolerance set refused")
        for bad_v in (0.0, -1e-15, float("nan"), float("inf"), 1):
            cx.raises("DC_TOLERANCE_INVALID", lambda b=bad_v: EV.dc_identity(
                tol, dict(tol, b=b)), f"tolerance {bad_v!r} refused before "
                "identity classification")
        cx.raises("DC_TOLERANCE_INVALID", lambda: EV.evaluate_states(
            model, states[:1], [True], {"primary": bank}, (128,),
            dc_tol=dict(tol, a=0.0)), "evaluate_states refuses invalid "
            "tolerances before any decode")
        cx.raises("DC_TOLERANCE_INVALID", lambda: EV.dc_corruption_tests(
            states[0], dict(tol, c_max=float("nan"))),
            "AM1 tests refuse invalid tolerances")
        real_meas = EV.dc_measures

        def blind_post(x, x_dc, y, m, where=""):
            r = real_meas(x, x_dc, y, m, where)
            return dict(r, a=0.0) if where == "AM1 post" else r
        with _patched(EV, "dc_measures", blind_post):
            cx.raises("DC_DETECTION_FAILED", lambda: EV.dc_corruption_tests(
                states[0], tol), "AM1 test 2: a detector that misses the "
                "corrupted DC output -> DC_DETECTION_FAILED (fixture patch: "
                "measure (a) blinded on the post-DC test only)")
        from seqref_mri.scripts import train_base as TB
        real_prep = TB._prepare

        def prep_x(*a, **k):
            r = real_prep(*a, **k)
            r["x_norm"] = r["x_norm"].clone()
            r["x_norm"][0, 0, 5, 5] += 1e-3
            return r

        def prep_c(*a, **k):
            r = real_prep(*a, **k)
            r["cond_in"] = r["cond_in"].clone()
            r["cond_in"][0, 1, 5, 5] += 1e-3
            return r
        for fn, what in ((prep_x, "ground truth"), (prep_c, "conditioning")):
            with _patched(TB, "_prepare", fn):
                cx.raises("EVAL_STATE_MISMATCH", lambda: EV.build_states(
                    data["vv"], [0], data["p4"], data["s_ref"]),
                    f"complex {what} differing from TINY's state -> "
                    "EVAL_STATE_MISMATCH (fixture patch of the second "
                    "_prepare call)")
        cx.record("F12 finding I1.7b-1 (synthetic stub)",
                  "R0+DC identity flags: "
                  f"{out['summary']['r0_dc']} -- stub R0 has ||u|| ~ 2e-4 x "
                  "||u_true||, so measure (b) roundoff exceeds the GT-scale "
                  "tolerance; ZF + DC not measured (by construction, AM3-D)")
    finally:
        t.cleanup()
    if not cx.host:
        return
    cx.phase = "host"
    with tempfile.TemporaryDirectory() as td:
        res = _run_eval_child(cx, Path(td), {"kind": "f12", "indices":
                                             list(range(6))}, "f12")
    cx.ok("error" not in res, "host F12 child completed", "F12_HOST_FAILED",
          str(res.get("error")))
    cx.ok(res["f12"]["checks_failed"] == 0, "host F12 checks (real "
          "validation states, fresh seed-2026 build, Z_DIAG bank)",
          "F12_HOST_FAILED", str(res["f12"]["failed"]))
    cx.record("F12 host (fixture pre-check; NOT the registered P5 values)",
              res["f12"]["record"])


def _zf_syn_row(vid, sl, psnr, ssim=0.5, nmse=1.0, eligible=True,
                midpoint=False):
    return {"volume_id": vid, "file": f"singlecoil_train/{vid}",
            "slice_index": sl, "psnr": float(psnr), "ssim": float(ssim),
            "nmse_u": nmse if eligible else None, "eligible": eligible,
            "midpoint": midpoint, "mse": 1.0, "psnr_capped": False,
            "zero_error": False}


def f22z(cx: Ctx) -> None:
    """F22 ZF part (AM3-D, AM3-E3): NMSE_u = 1 on eligible slices; no NLL;
    non-finite -> EPOCH0_NONFINITE; volume-weighted summary; strict JSON;
    atomic write-once pair with refusal/recovery; gate-2a guard."""
    import torch
    from seqref_mri.scripts import v05lt_eval as EV
    t, data, states = _stub_states()
    try:
        infos = [{"eligible": e, "midpoint": m, "volume_id": st["identity"][
            "file"].split("/")[-1]} for st, e, m in zip(
                states, (True, True, False), (True, False, False))]
        rows = [EV.zf_row(st, inf) for st, inf in zip(states, infos)]
        cx.ok(rows[0]["nmse_u"] == 1.0 and rows[1]["nmse_u"] == 1.0 and
              rows[2]["nmse_u"] is None,
              "ZF NMSE_u == 1 exactly on eligible slices; ineligible -> None")
        cx.ok(all("nll" not in r and "nll_per_dim" not in r for r in rows)
              and all({"psnr", "ssim", "mse", "psnr_capped", "zero_error",
                       "volume_id", "file", "slice_index", "eligible",
                       "midpoint"} <= set(r) for r in rows),
              "ZF row: PSNR, SSIM, raw MSE, cap flags and slice identity "
              "(volume, file, slice, eligible, midpoint); no NLL")
        cx.raises("ZF_ROW_INFO", lambda: EV.zf_row(states[0], True),
                  "ZF row without eligibility entry refused")
        from seqref_mri.tdiag import estimators as est
        img = est.image_from_u(states[0], np.zeros_like(states[0]["u_true"]))
        cx.ok(rows[0]["psnr"] == EV.M.psnr_record(img.abs(), states[0][
            "x_true_mag"])["psnr"],
              "ZF formed through estimators.image_from_u(u = 0) (production "
              "decode_normalised path)")
        bad = dict(states[0])
        bad["y"] = states[0]["y"].clone()
        bad["y"][0, 0, 0] = complex(float("nan"), 0.0)
        cx.raises("EPOCH0_NONFINITE", lambda: EV.zf_row(bad, infos[0]),
                  "NaN in coefficients -> EPOCH0_NONFINITE")
        bad2 = dict(states[0])
        bad2["x_true_mag"] = states[0]["x_true_mag"].clone()
        bad2["x_true_mag"][1, 1] = float("inf")
        cx.raises("EPOCH0_NONFINITE", lambda: EV.zf_row(bad2, infos[0]),
                  "non-finite metric input -> EPOCH0_NONFINITE")
        # volume weighting (reviewer example: 0, 0 | 30 -> 15, not 10)
        ex = [_zf_syn_row("A.h5", 0, 0.0), _zf_syn_row("A.h5", 1, 0.0),
              _zf_syn_row("B.h5", 0, 30.0, midpoint=True)]
        sm = EV.zf_summary(ex)
        cx.ok(sm["all_slices"]["psnr"] == 15.0 and
              sm["all_slices"]["volumes"] == 2 and
              sm["midpoint"]["psnr"] == 30.0,
              "ZF summary: slice -> volume mean -> equal-weight mean "
              "(two-slice volume 0 dB + one-slice volume 30 dB = 15 dB)",
              detail=str(sm["all_slices"]["psnr"]))
        nm = [_zf_syn_row("A.h5", 0, 1.0, nmse=1.0),
              _zf_syn_row("A.h5", 1, 1.0, eligible=False),
              _zf_syn_row("B.h5", 0, 1.0, eligible=False)]
        sn = EV.zf_summary(nm)["all_slices"]
        cx.ok(sn["nmse_u"] == 1.0 and sn["nmse_population_volumes"] == 1 and
              sn["nmse_null_volumes"] == ["B.h5"] and sn["psnr"] == 1.0,
              "NMSE_u over eligible slices; volume without one -> null, "
              "outside the population; PSNR keeps all slices")
        cx.raises("AGG_DUPLICATE_SLICE", lambda: EV.zf_summary(ex + ex[:1]),
                  "duplicate slice refused")
        summ = EV.zf_summary(rows)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            reg = td / "results" / "v05lt" / "zf"
            cx.raises("ZF_GATE_REQUIRED", lambda: EV.write_zf(
                reg, rows, summ, {"fixture": True}),
                "registered results/v05lt/zf refused without gate 2a")
            cx.raises("ZF_GATE_REQUIRED", lambda: EV.write_zf(
                reg / "sub", rows, summ, {"fixture": True}),
                "subdirectory of results/v05lt/zf also refused")
            nanrows = [dict(rows[0], psnr=float("nan"))]
            cx.raises("ZF_JSON_INVALID", lambda: EV.write_zf(
                td / "zf_nan", nanrows, summ, {"fixture": True}),
                "strict JSON: NaN refused (allow_nan=False)")
            cx.ok(not (td / "zf_nan").exists() and
                  not (td / ".zf_nan.staging").exists(),
                  "refused JSON leaves nothing behind")
            other = td / "zf_tmp"
            sha = EV.write_zf(other, rows, summ, {"fixture": True})
            cx.ok(EV.verify_zf(other) == sha and sorted(
                p.name for p in other.iterdir()) == [
                    "zf_reference.json", "zf_reference.json.sha256"] and
                  not (td / ".zf_tmp.staging").exists(),
                  "ZF pair written atomically (data + sidecar, verified)")
            doc = json.loads((other / "zf_reference.json").read_text())
            cx.ok(doc["summary"] == json.loads(json.dumps(summ)) and
                  len(doc["rows"]) == 3, "ZF record round-trips")
            cx.raises("ZF_EXISTS", lambda: EV.write_zf(
                other, rows, summ, {"fixture": True}),
                "ZF write-once: second write refused")
            # interruption after the pair is staged, before the rename
            crash = td / "zf_crash"
            def _crash_child():
                try:
                    EV.write_zf(crash, rows, summ, {"fixture": True},
                                hook=lambda stage: os._exit(9))
                finally:
                    os._exit(3)
            p = multiprocessing.get_context("fork").Process(
                target=_crash_child)
            p.start()
            code = join_bounded(p, "fork")      # timeout + TERM/KILL
            stg = td / ".zf_crash.staging"
            cx.ok(code == 9, "crash child exited at the staged hook (exit 9; "
                  "bounded wait)", "CHILD_FAILED", f"exit {code}")
            cx.ok(not crash.exists() and stg.is_dir(), "killed after "
                  "staging: final directory absent (no partial pair "
                  "visible); staging kept -- process kill only, NOT "
                  "power-loss durability")
            cx.raises("ZF_STAGING_PRESENT", lambda: EV.write_zf(
                crash, rows, summ, {"fixture": True}),
                "rerun refused while an interrupted staging directory "
                "exists (owner inspects; nothing deleted)")
            cx.ok(stg.is_dir(), "staging directory not deleted by refusal")
            half = td / "zf_half"
            half.mkdir()
            (half / "zf_reference.json").write_text("{}\n")
            cx.raises("ZF_INCOMPLETE", lambda: EV.verify_zf(half),
                      "data without sidecar -> ZF_INCOMPLETE")
            cx.raises("ZF_EXISTS", lambda: EV.write_zf(
                half, rows, summ, {"fixture": True}),
                "incomplete directory never overwritten")
            (half / "zf_reference.json.sha256").write_text(
                "0" * 64 + "  zf_reference.json\n")
            cx.raises("ZF_INCOMPLETE", lambda: EV.verify_zf(half),
                      "sidecar hash mismatch -> ZF_INCOMPLETE")
            sha2 = EV.write_zf(reg, rows, summ, {"fixture": True},
                               gate_2a=True)
            cx.ok(EV.verify_zf(reg) == sha2, "gate_2a flag lets the writer "
                  "proceed (temporary directory only; no production write)")
        del torch
    finally:
        t.cleanup()
    if not cx.host:
        return
    cx.phase = "host"
    with tempfile.TemporaryDirectory() as td:
        res = _run_eval_child(cx, Path(td), {"kind": "zf", "indices":
                                             list(range(6))}, "zf")
    cx.ok("error" not in res, "host ZF child completed", "F22Z_HOST_FAILED",
          str(res.get("error")))
    zr = res["zf"]
    cx.ok(all(v == 1.0 for v in zr["nmse_eligible"]) and
          zr["n_rows"] == 6 and zr["rows_have_identity"],
          "host: ZF NMSE_u == 1 on eligible real validation slices; rows "
          "carry identities; nothing written to results", "F22Z_HOST_FAILED",
          str({k: v for k, v in zr.items() if k != "summary"}))
    cx.record("F22Z host (fixture; NOT the gate-2a ZF pass)", zr)


def f23(cx: Ctx) -> None:
    """Holdout NMSE_u rule (AM4-B) on a synthetic 199-slice population;
    validation path applies AM2-C; no holdout data read."""
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_eligibility as EL
    g = np.random.Generator(np.random.PCG64(23))
    s_ref = C.S_REF
    sts, uh = [], []
    for i in range(199):
        u = g.standard_normal(64) + 1j * g.standard_normal(64)
        if i < 5:                                   # 0 < ratio <= 1e-10
            u = u / np.sqrt(np.sum(np.abs(u) ** 2)) * s_ref * 1e-6
        sts.append({"identity": {"file": "syn", "slice_index": i},
                    "u_true": u.astype(np.complex128)})
        uh.append((u + 0.1 * g.standard_normal(64)).astype(np.complex128))
    ratios = [float(np.sum(np.abs(s["u_true"]) ** 2)) / s_ref ** 2
              for s in sts]
    cx.ok(sum(0 < r <= EL.R_FREE_MIN for r in ratios) == 5,
          "synthetic population has 5 slices with 0 < ratio <= 1e-10")
    hold = [EV._nmse_or_none(u, s, True, "v02_holdout_all")
            for u, s in zip(uh, sts)]
    hm = EV.holdout_nmse_mean(hold)
    plain = float(np.mean([np.sum(np.abs(u - s["u_true"]) ** 2) /
                           np.sum(np.abs(s["u_true"]) ** 2)
                           for u, s in zip(uh, sts)]))
    cx.ok(hm["n"] == 199 and math.isclose(hm["mean"], plain, rel_tol=1e-12),
          "holdout: all 199 slices included (no threshold); plain mean",
          detail=f"{hm['mean']} vs {plain}")
    val = [EV._nmse_or_none(u, s, EL.is_eligible(r), "validation_eligible")
           for u, s, r in zip(uh, sts, ratios)]
    cx.ok(sum(v is None for v in val) == 5 and
          sum(v is not None for v in val) == 194,
          "validation path on the same data applies AM2-C (5 excluded): "
          "paths differ as declared")
    z = dict(sts[7])
    z["u_true"] = np.zeros(64, np.complex128)
    cx.raises("VALIDITY_NMSE_UNDEFINED", lambda: EV._nmse_or_none(
        uh[7], z, True, "v02_holdout_all"),
        "exactly zero free-space energy -> VALIDITY_NMSE_UNDEFINED")
    cx.raises("VALIDITY_NMSE_UNDEFINED", lambda: EV.holdout_nmse_mean(
        hold[:10] + [None]), "a missing holdout value is never dropped")
    cx.raises("EVAL_NMSE_RULE", lambda: EV._nmse_or_none(
        uh[0], sts[0], True, "threshold"), "unregistered NMSE rule refused")
    cx.record("F23", "synthetic only by design (no holdout data before §9.3)")


# ---------------------------------------------------------------------------
# I1.7(c1): PUB (item 22), E0V (epoch-0 evaluation), F16 (panels),
# F22E (epoch-0 non-finite), F21E (evaluator isolation, real trainer)
# ---------------------------------------------------------------------------
def _png_shape(png: bytes) -> tuple:
    import struct
    C.require(png[:8] == b"\x89PNG\r\n\x1a\n", "FIXTURE_PNG", "signature")
    w, h = struct.unpack(">II", png[16:24])
    return h, w


def _syn_e0_inputs(tmp: Path):
    from seqref_mri.scripts import v05lt_eval as EV
    data = _syn_train_data(tmp)
    vv, el = data["vv"], data["eligibility"]
    mids = [i for i in range(len(vv)) if el[vv.entry(i)]["midpoint"]]
    cases = [{"case": j, "file": vv.entry(i)[0], "slice_index": vv.entry(i)[1]}
             for j, i in enumerate(mids)]
    states = EV.build_states(vv, mids, data["p4"], data["s_ref"])
    tol = EV.calibrate_dc(states)["tolerances"]
    bank = np.random.Generator(np.random.PCG64(0)).standard_normal(
        (128, 13824)).astype(np.float32)
    return data, mids, cases, tol, bank


def pub(cx: Ctx) -> None:
    """Item 22: atomic no-overwrite publication (renameat2 NOREPLACE)."""
    from seqref_mri.scripts import v05lt_eval as EV
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src"
        src.mkdir()
        (src / "f").write_text("x")
        EV.publish_noreplace(src, td / "dst")
        cx.ok((td / "dst" / "f").read_text() == "x" and not src.exists(),
              "publish to a free destination")
        for kind in ("empty_dir", "nonempty_dir", "file", "symlink"):
            s2 = td / f"s_{kind}"
            s2.mkdir()
            (s2 / "new").write_text("new")
            d2 = td / f"d_{kind}"
            if kind == "empty_dir":
                d2.mkdir()
            elif kind == "nonempty_dir":
                d2.mkdir()
                (d2 / "old").write_text("old")
            elif kind == "file":
                d2.write_text("old")
            else:
                d2.symlink_to(td / "nowhere")
            cx.raises("ARTIFACT_EXISTS", lambda a=s2, b=d2:
                      EV.publish_noreplace(a, b),
                      f"existing destination ({kind}) never replaced")
            cx.ok(s2.is_dir() and (s2 / "new").exists() and
                  (d2.is_symlink() or d2.exists()) and
                  (kind != "empty_dir" or not any(d2.iterdir())),
                  f"{kind}: source and destination both left unchanged")

        def no_syscall():
            C.fail("PUBLISH_NOREPLACE_UNAVAILABLE", "fixture: simulated "
                   "missing renameat2")
        with _patched(EV, "_renameat2", no_syscall):
            cx.raises("PUBLISH_NOREPLACE_UNAVAILABLE", lambda:
                      EV.publish_noreplace(td / "s_file", td / "free"),
                      "missing renameat2 -> stop, no replacing fallback "
                      "(fixture patch)")
        # ZF writer: destination created during the write (the race)
        t, data, states = _stub_states(1)
        try:
            info = {"eligible": True, "midpoint": True, "volume_id": "v"}
            rows = [EV.zf_row(states[0], info)]
            summ = EV.zf_summary(rows)
            race = td / "zf_race"
            cx.raises("ZF_EXISTS", lambda: EV.write_zf(
                race, rows, summ, {"fixture": True},
                hook=lambda stage: race.mkdir()),
                "ZF: empty directory appearing before publication is not "
                "replaced (race closed)")
            cx.ok(race.is_dir() and not any(race.iterdir()) and
                  (td / ".zf_race.staging").is_dir(),
                  "race: destination untouched, complete pair kept in staging")
        finally:
            t.cleanup()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        t, data, states = _stub_states(1)
        try:
            info = {"eligible": True, "midpoint": True, "volume_id": "v"}
            rows = [EV.zf_row(states[0], info)]
            summ = EV.zf_summary(rows)
            for kind in ("dangling", "to_dir"):
                tgt = td / f"target_{kind}"
                if kind == "to_dir":
                    tgt.mkdir()
                link = td / f"zf_link_{kind}"
                link.symlink_to(tgt)
                cx.raises("ZF_EXISTS", lambda a=link: EV.write_zf(
                    a, rows, summ, {"fixture": True}),
                    f"write_zf: {kind} symlink at destination refused, not "
                    "followed")
                cx.ok(link.is_symlink() and (kind == "to_dir" or
                                             not tgt.exists()) and
                      (kind == "dangling" or not any(tgt.iterdir())),
                      f"write_zf {kind}: link kept, nothing created at target")
            sl = td / ".zf_s.staging"
            sl.symlink_to(td / "nowhere_s")
            cx.raises("ZF_STAGING_PRESENT", lambda: EV.write_zf(
                td / "zf_s", rows, summ, {"fixture": True}),
                "write_zf: dangling symlink at the staging path refused")
            cx.ok(not (td / "nowhere_s").exists() and
                  not (td / "zf_s").exists(), "staging link not followed")
        finally:
            t.cleanup()
        data, mids, cases, tol, bank = _syn_e0_inputs(td / "d")
        from seqref_mri.scripts.v05lt_stub_train import build_stub
        rec, pan = EV.epoch0_evaluate(build_stub(2026), data, bank, tol,
                                      cases)
        for kind in ("dangling", "to_dir"):
            tgt = td / f"e0_target_{kind}"
            if kind == "to_dir":
                tgt.mkdir()
            link = td / f"e0_link_{kind}"
            link.symlink_to(tgt)
            cx.raises("ARTIFACT_EXISTS", lambda a=link: EV.write_epoch0_eval(
                a, rec, pan, {"fixture": True}),
                f"write_epoch0_eval: {kind} symlink at destination refused")
            cx.ok(link.is_symlink() and (kind == "to_dir" or
                                         not tgt.exists()) and
                  (kind == "dangling" or not any(tgt.iterdir())),
                  f"write_epoch0_eval {kind}: nothing created at target")
        sl = td / ".e0s.staging"
        sl.symlink_to(td / "nowhere_e")
        cx.raises("STAGING_PRESENT", lambda: EV.write_epoch0_eval(
            td / "e0s", rec, pan, {"fixture": True}),
            "write_epoch0_eval: dangling symlink at the staging path refused")
        cx.ok(not (td / "nowhere_e").exists(), "staging link not followed")
        reg = td / "results" / "v05lt" / "epoch0" / "seed_2026"
        reg.mkdir(parents=True)
        (td / "elsewhere").mkdir()
        (reg / "eval").symlink_to(td / "elsewhere" / "x")
        cx.raises("EPOCH0_EVAL_GATE_REQUIRED", lambda: EV.write_epoch0_eval(
            reg / "eval", rec, pan, {"fixture": True}),
            "registered path judged on the unfollowed name (a link there "
            "does not redirect the gate check)")
    cx.record("PUB", "renameat2(RENAME_NOREPLACE); process-kill and race "
              "tests only -- power-loss durability untested (declared)")


def e0v(cx: Ctx) -> None:
    """Epoch-0 evaluation (AM3-C) + writer; timing recorded (addition C)."""
    import torch
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts.v05lt_stub_train import build_stub
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data, mids, cases, tol, bank = _syn_e0_inputs(td / "d")
        m = build_stub(2026)
        m.train()
        sd0 = {k: v.clone() for k, v in m.state_dict().items()}
        torch.manual_seed(5)
        np.random.seed(5)
        r0 = (torch.get_rng_state().clone(), np.random.get_state()[1].copy())
        cnt = {}
        rec, pan = EV.epoch0_evaluate(m, data, bank, tol, cases,
                                      decode_counter=cnt)
        cx.ok(m.training and all(torch.equal(sd0[k], v) for k, v in
                                 m.state_dict().items()),
              "parameters/buffers unchanged; previous mode restored")
        cx.ok(torch.equal(r0[0], torch.get_rng_state()) and
              np.array_equal(r0[1], np.random.get_state()[1]) and
              rec["rng"]["restored"], "torch/NumPy RNG states identical "
              "after evaluation (AM3-B in-process rule)",
              detail=str(rec["rng"]))
        n_all = len(data["vv"])
        cx.ok(rec["all_slice"]["summary"]["all_slices"]["slices"] == n_all
              and rec["midpoint"]["n"] == len(mids) and
              all({"r0", "r1", "r0_dc", "r1_dc"} <= set(r) for r in
                  rec["midpoint"]["rows"]),
              "all-slice R0/SSIM/NMSE_u/NLL on every validation slice; "
              "R0, R1, R0+DC, R1+DC on the midpoint slices")
        cx.ok(cnt == {"primary": 128 * len(mids)} and
              rec["midpoint"]["decodes"] == 128 * len(mids),
              "exactly 128 primary-bank decodes per midpoint slice; panels "
              "add none", detail=str(cnt))
        cx.ok(set(rec["timing"]) == {"all_slice_s", "midpoint_readouts_s",
                                     "panels_s"} and
              all(v > 0 for v in rec["timing"].values()),
              "timing recorded per stage (addition C)",
              detail=str(rec["timing"]))
        out = td / "e0" / "eval"
        shas = EV.write_epoch0_eval(out, rec, pan, {"fixture": True})
        sums = (out / "SHA256SUMS").read_text().split("\n")
        cx.ok(all(C.sha256_file(out / n) == h for n, h in shas.items()) and
              len([x for x in sums if x]) == len(shas) and
              not (td / "e0" / ".eval.staging").exists(),
              "write-once directory: record, PNG + .npz per case, SHA256SUMS; "
              "published atomically")
        doc = json.loads((out / EV.E0_EVAL_FILE).read_text())
        cx.ok(doc["schema"] == EV.E0_EVAL_SCHEMA and doc["cases"] ==
              [c["case"] for c in cases] and set(doc["panel_sha256"]) ==
              {n for n in shas if n.startswith("panel_")},
              "record lists cases in manifest order and every panel hash")
        cx.raises("ARTIFACT_EXISTS", lambda: EV.write_epoch0_eval(
            out, rec, pan, {"fixture": True}), "second write refused")
        reg = td / "results" / "v05lt" / "epoch0" / "seed_2026" / "eval"
        cx.raises("EPOCH0_EVAL_GATE_REQUIRED", lambda: EV.write_epoch0_eval(
            reg, rec, pan, {"fixture": True}), "registered epoch-0 eval path "
            "refused without the gated stage")
        bad = dict(rec, timing=dict(rec["timing"], all_slice_s=float("nan")))
        cx.raises("EPOCH0_EVAL_JSON_INVALID", lambda: EV.write_epoch0_eval(
            td / "nan" / "eval", bad, pan, {"fixture": True}),
            "strict JSON (NaN refused)")
        cx.raises("PANEL_CASE_MISSING", lambda: EV.epoch0_evaluate(
            m, data, bank, tol, cases, mid_indices=mids[:1]),
            "panel case outside the evaluated midpoints -> error, never "
            "skipped")
        cx.raises("DC_TOLERANCE_INVALID", lambda: EV.epoch0_evaluate(
            m, data, bank, dict(tol, a=0.0), cases), "invalid tolerance "
            "refused before evaluation")
    if not cx.host:
        return
    cx.phase = "host"
    with tempfile.TemporaryDirectory() as t:
        res = _run_eval_child(cx, Path(t), {"kind": "e0v"}, "e0v")
    cx.ok("error" not in res, "host epoch-0 evaluation child completed",
          "E0V_HOST_FAILED", str(res.get("error")))
    e = res["e0v"]
    cx.ok(e["decodes"] == 128 * e["n_mid"] and e["state_unchanged"] and
          e["rng_restored"] and e["written"] and e["registered_refused"],
          "host: temporary epoch-0 checkpoint (seed 2026) evaluated from the "
          "saved file; 128 decodes per midpoint; state unchanged; outputs "
          "published to a temporary directory only", "E0V_HOST_FAILED",
          str(e))
    cx.record("E0V host timing (fixture subset; addition C, P4 input only)",
              {k: e[k] for k in ("timing", "n_all", "n_mid", "cases",
                                 "torch_threads")})


def f16(cx: Ctx) -> None:
    """Panels (194 §10.1): fixed model-independent scales, identical cases
    and order across checkpoints, DC columns only when not identical,
    PNG/.npz content, missing/invalid -> typed error."""
    import io
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts.v05lt_stub_train import build_stub
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data, mids, cases, tol, bank = _syn_e0_inputs(td / "d")
        ra, pa = EV.epoch0_evaluate(build_stub(2026), data, bank, tol, cases)
        rb, pb = EV.epoch0_evaluate(build_stub(2027), data, bank, tol, cases)
        cx.ok([p["case"] for p in pa] == [p["case"] for p in pb] == cases,
              "identical cases, in manifest order, for both checkpoints")
        cx.ok(all(p["scales"] == q["scales"] for p, q in zip(pa, pb)),
              "fixed scales identical across checkpoints (from |GT| and "
              "|Observed - GT| only)")
        st = EV.build_states(data["vv"], mids, data["p4"], data["s_ref"])
        gt = st[0]["x_true_mag"].double().numpy()
        cx.ok(pa[0]["scales"]["image"][1] == float(np.percentile(gt, 99.5)),
              "image scale = p99.5(|GT|)")
        rows = {EV._ident_key(r["identity"]): r for r in
                ra["midpoint"]["rows"]}
        ok = True
        for p in pa:
            r = rows[(p["case"]["file"], p["case"]["slice_index"])]
            for key in ("r0_dc", "r1_dc"):
                has = key in p["columns"]
                ok &= (has == (not r[key]["dc_identity"]))
                ok &= ((not has) == any(n.startswith(key) for n in
                                        p["notes"]))
        cx.ok(ok, "DC columns present iff DC not identical within tolerance; "
              "otherwise the panel notes 'DC identical within tolerance'")
        f = EV.panel_files(pa[0], "panel_case0")
        h, w = _png_shape(f["panel_case0.png"])
        ncol = len(pa[0]["columns"]) + len(pa[0]["errors"])
        cx.ok(h == 19 + 9 + 96 and
              w == ncol * 96 + 2 * EV.BAR_W + (ncol + 1) * EV.PANEL_SEP,
              "PNG: header (case identity, numeric scale endpoints), a label "
              "above every column, all columns + two labelled colour bars",
              detail=f"{h}x{w}")
        import zlib
        px = np.frombuffer(zlib.decompress(f["panel_case0.png"][41:-16]),
                           np.uint8)
        rows_ = px.reshape(h, w + 1)[:, 1:]
        cx.ok(rows_[:7].any() and rows_[9:16].any() and rows_[19:26].any(),
              "PNG text rows rendered (header lines and column labels)")
        z0 = EV.render_text("R0", 12)
        cx.ok(z0.shape == (7, 12) and z0.sum() == sum(
            ch == "1" for g in ("R", "0") for ch in EV.FONT5X7[g]),
            "label glyphs = registered 5x7 font bitmaps")
        cx.raises("PANEL_LABEL_CHARS", lambda: EV.render_text("A#B", 30),
                  "unsupported label character refused (never dropped)")
        cx.raises("PANEL_LABEL_WIDTH", lambda: EV.render_text("X" * 20, 96),
                  "label wider than its column refused (never truncated)")
        dup = [dict(cases[0]), dict(cases[1], case=cases[0]["case"])]
        cx.raises("PANEL_CASE_DUPLICATE", lambda: EV.epoch0_evaluate(
            build_stub(2026), data, bank, tol, dup),
            "duplicate case numbers refused before evaluation")
        cx.raises("PANEL_CASE_DUPLICATE", lambda: EV.write_epoch0_eval(
            td / "dup" / "eval", ra, [pa[0], dict(pa[1], case=dict(
                pa[1]["case"], case=pa[0]["case"]["case"]))],
            {"fixture": True}), "writer refuses duplicate case numbers (no "
            "silent panel replacement)")
        z = np.load(io.BytesIO(f["panel_case0.npz"]))
        cx.ok(all(np.array_equal(z[k], v) for k, v in
                  {**pa[0]["columns"], **pa[0]["errors"]}.items()) and
              list(z["scale_image"]) == pa[0]["scales"]["image"],
              ".npz holds the exact arrays and scales")
        flat = dict(st[0])
        flat["x_true_mag"] = st[0]["x_true_mag"] * 0
        cx.raises("PANEL_SCALE_INVALID", lambda: EV.build_panel(
            cases[0], flat, {"r0": st[0]["x_true_c"].to(
                __import__("torch").complex128),
                "r1": st[0]["x_true_c"].to(__import__("torch").complex128)},
            rows[(cases[0]["file"], cases[0]["slice_index"])]),
            "zero display scale -> PANEL_SCALE_INVALID (never autoscaled)")
        cx.raises("PANEL_IMAGES_MISSING", lambda: EV.build_panel(
            cases[0], st[0], {"r0": None},
            rows[(cases[0]["file"], cases[0]["slice_index"])]),
            "missing readout image -> typed error, panel never skipped")
        cx.record("F16", "PNG labels: case identity, numeric scale "
                  "endpoints, column names, bar names (5x7 bitmap font)")


def f22e(cx: Ctx) -> None:
    """F22 epoch-0 part (AM3-E3): NaN/Inf -> EPOCH0_NONFINITE."""
    import torch
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts.v05lt_stub_train import build_stub
    with tempfile.TemporaryDirectory() as td:
        data, mids, cases, tol, bank = _syn_e0_inputs(Path(td) / "d")
        m = build_stub(2026)
        with torch.no_grad():
            m.W.fill_(float("nan"))
        cx.raises("EPOCH0_NONFINITE", lambda: EV.epoch0_evaluate(
            m, data, bank, tol, cases), "NaN model output -> EPOCH0_NONFINITE "
            "(all-slice stage)")
        m2 = build_stub(2026)
        cx.raises("EPOCH0_NONFINITE", lambda: EV.epoch0_evaluate(
            m2, data, bank * float("nan"), tol, cases),
            "NaN decode bank -> EPOCH0_NONFINITE (midpoint stage)")


def f21e(cx: Ctx) -> None:
    """F21 completion: training with vs without a preceding run of the
    ACTUAL reconstruction evaluator (epoch0_evaluate, in-process with RNG
    save/restore) -- bitwise identical (fresh processes)."""
    _rt_phase(cx)
    ref = _rt_ref(cx)
    out = _rt_run(cx, "lam2_s2026", "f21_e0eval",
                  pre_eval=_RT["e0_path"][2026], pre_eval_full=True)
    cx.ok("pre_eval_full" in out and out["pre_eval_full"]["decodes"] > 0,
          "F21: epoch0_evaluate (all-slice + midpoint R0/R1/DC + panels) ran "
          "on the epoch-0 checkpoint before training",
          detail=str(out.get("pre_eval_full")))
    _same_runs(cx, ref, _rt_dir("f21_e0eval"),
               "F21: with vs without preceding reconstruction evaluation")
    _rt_drop("f21_e0eval")


# ---------------------------------------------------------------------------
# I1.7(c2): CLS -- item 18 failure classification (trainer, shared validation)
# ---------------------------------------------------------------------------
def cls(cx: Ctx) -> None:
    """Known production errors at the trainer / shared-validation seams map
    to the registered taxonomy (terminal run_end), never to session_error;
    genuinely unexpected errors (outside the seams) stay session_error."""
    import torch
    from seqref_mri.scripts import train_base as TB
    from seqref_mri.scripts import train_free_flow as TFF
    from seqref_mri.scripts import tiny_gate as TG
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.scripts import v05lt_train as T
    from seqref_mri.scripts.v05lt_stub_train import build_stub

    def once(fn_real, exc, when=lambda: True):
        state = {"done": False}

        def f(*a, **k):
            if not state["done"] and when():
                state["done"] = True
                raise exc
            return fn_real(*a, **k)
        return f

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        data = _syn_train_data(tmp / "syn")

        def check(name, patch, code, tax, factory=None, label=""):
            cfg = _stub_cfg(tmp / name, max_epochs=3)
            run = (T.Run(cfg, C.Paths(), data, model_factory=factory)
                   if factory else _stub_run(cfg, data))
            try:
                if patch is None:
                    run.start()
                else:
                    with _patched(*patch):
                        run.start()
                cx.ok(False, f"{label}: failure expected", "NO_ERROR_RAISED")
                return
            except C.V05LTError as exc:
                got = (exc.code, exc.taxonomy)
            k = _kinds(Path(cfg.run_dir))
            e = _ends(Path(cfg.run_dir))
            cx.ok(got == (code, tax) and "session_error" not in k and
                  len(e) == 1 and e[0]["status"] == tax and
                  e[0]["code"] == code,
                  f"{label} -> {code} ({tax}), terminal run_end, no "
                  "session_error", detail=f"{got} {k[-3:]} {e}")
            _expect(cx, "RESUME_RUN_ENDED", lambda: _stub_run(cfg,
                                                               data).resume(),
                    f"{label}: never resumed")

        check("dl", (TB, "_collate", once(TB._collate, OSError(
            "fixture: read error"))), "TRAIN_DATA_LOAD", "failed_validity",
            label="training data loading / collation")
        check("tg", (TFF, "targets_from_prepared", once(
            TFF.targets_from_prepared, ValueError("fixture: bad map"))),
            "TRAIN_TARGETS", "failed_validity",
            label="mask / standardisation / target construction")
        check("obj", (TFF, "nll_objective", once(TFF.nll_objective,
                                                 RuntimeError("fixture: "
                                                              "backend"))),
              "OBJECTIVE_FAILED", "failed_validity",
              label="objective backend error (finite)")
        check("objnf", (TFF, "nll_objective", once(
            TFF.nll_objective, RuntimeError("fixture: non-finite value"))),
            "NUMERICAL_FAILURE", "numerical_failure",
            label="objective backend non-finite")
        check("vdec", (TG, "_decode_z", once(TG._decode_z, RuntimeError(
            "U_NON_FINITE: fixture"))), "VAL_NON_FINITE", "failed_validity",
            label="validation decode non-finite (194 §8.2 validity)")
        check("vdec2", (TG, "_decode_z", once(TG._decode_z, KeyError(
            "fixture"))), "VAL_DECODE_FAILED", "failed_validity",
            label="validation decode error")

        def lp_factory(seed):
            m = build_stub(seed)
            real = m.log_prob_free

            def lp(*a, **k):
                if not m.training:
                    raise RuntimeError("fixture: validation backend error")
                return real(*a, **k)
            m.log_prob_free = lp
            return m
        check("vnll", None, "VAL_NLL_FAILED", "failed_validity",
              factory=lp_factory, label="validation NLL backend error")

        def bad_writer(model, od, state):
            def w(fh):
                raise OSError(28, "fixture: No space left on device")
            return w
        check("ckpt", (T, "_resume_writer", bad_writer),
              "CHECKPOINT_WRITE_FAILED", "failed_validity",
              label="checkpoint write (resume file)")

        def bad_wwriter(model, meta):
            def w(fh):
                raise OSError(5, "fixture: I/O error")
            return w
        check("wts", (T, "_weights_writer", bad_wwriter),
              "CHECKPOINT_WRITE_FAILED", "failed_validity",
              label="checkpoint write (weights file staging)")
        check("cmt", (C.ArtifactStore, "commit", once(
            C.ArtifactStore.commit, OSError(5, "fixture: commit I/O"))),
            "CHECKPOINT_WRITE_FAILED", "failed_validity",
            label="checkpoint commit")
        check("prep", (TB, "_prepare", once(TB._prepare, RuntimeError(
            "fixture: prepare"))), "TRAIN_PREPARE", "failed_validity",
            label="training batch preparation")
        check("mep", (type(data["tv"]), "set_training_epoch", once(
            type(data["tv"]).set_training_epoch, RuntimeError(
                "fixture: mask epoch"))), "TRAIN_DATA_LOAD", "failed_validity",
            label="mask-epoch change")
        check("bwd", (torch.Tensor, "backward", once(
            torch.Tensor.backward, RuntimeError("fixture: autograd"))),
            "TRAIN_STEP_FAILED", "failed_validity",
            label="backward pass (finite error)")
        check("bwdnf", (torch.Tensor, "backward", once(
            torch.Tensor.backward, RuntimeError("fixture: non-finite grad"))),
            "TRAIN_STEP_FAILED", "numerical_failure",
            label="backward pass (non-finite)")
        check("opt", (torch.optim.Adam, "step", once(
            torch.optim.Adam.step, RuntimeError("fixture: optimiser"))),
            "TRAIN_STEP_FAILED", "failed_validity",
            label="optimiser step")
        check("vsum", (M, "summarise", once(M.summarise, KeyError(
            "fixture"))), "VAL_SUMMARY_FAILED", "failed_validity",
            label="validation summary")
        # unit: shared validate() on its own
        st = data["vv"]
        with _patched(TG, "_decode_z", once(TG._decode_z, RuntimeError(
                "U_NON_FINITE: fixture"))):
            cx.raises("VAL_NON_FINITE", lambda: M.validate(
                build_stub(2026), st, data["eligibility"], data["p4"],
                data["s_ref"], [0]), "validate(): non-finite decode typed")
        with _patched(TB, "_collate", once(TB._collate, OSError("fx"))):
            cx.raises("VAL_DATA_LOAD", lambda: M.validate(
                build_stub(2026), st, data["eligibility"], data["p4"],
                data["s_ref"], [0]), "validate(): data loading typed")
        # unexpected errors outside the seams remain session_error (TFX 17)
        cfg = _stub_cfg(tmp / "unexp", max_epochs=3)

        def boom(stage):
            if stage == "train:step:1":
                raise RuntimeError("fixture: outside any seam")
        try:
            _stub_run(cfg, data, boom).start()
            cx.ok(False, "unexpected error expected", "NO_ERROR_RAISED")
        except RuntimeError:
            k = _kinds(Path(cfg.run_dir))
            cx.ok("session_error" in k and "run_end" not in k,
                  "error outside the seams still -> session_error (owner "
                  "decision required, item 19)")
        del torch
    cx.record("CLS consequence", "data-loading, checkpoint-write and "
              "commit failures are terminal (failed_validity, 195 §1/§5): a "
              "temporary I/O fault ends that run; resuming it requires an "
              "owner decision outside the trainer")
    cx.record("CLS scope", "trainer: data loading, prepare, mask/"
              "standardisation/targets, objective, backward/optimiser, "
              "mask epoch, validation summary, checkpoint stage/commit; "
              "shared validate(): data loading, NLL, R0 decode")


def cpb(cx: Ctx) -> None:
    """Addition B cost probe: elapsed time and peak memory of the actual
    128-row batched decode + reconstruction + metrics (fresh process)."""
    with tempfile.TemporaryDirectory() as t:
        out = _run_eval_child(cx, Path(t), {"kind": "cpb", "indices": [0, 1, 2],
                                            "stub": not cx.host}, "cpb")
    cx.ok("error" not in out, "cost probe completed", "CPB_FAILED",
          str(out.get("error")))
    per = out["cpb"]["per_slice"]
    rec = {"slices": len(per), "torch_threads": out["torch_threads"],
           "decode_128_s": [round(x["decode_128_s"], 4) for x in per],
           "recon_metrics_s (4 prefixes + R0)": [round(x["recon_metrics_s"], 4)
                                                 for x in per],
           "probe overhead (extra R0 decode + R1 image formation; not "
           "incurred by evaluate_states)": [
               round(x["probe_r0_redecode_r1_image_s"], 4) for x in per],
           "dc_s (DC + measures + metrics, R0+DC and R1+DC)": [
               round(x["dc_two_readouts_s"], 4) for x in per],
           "zf_s": [round(x["zf_s"], 4) for x in per],
           "dc_calibration_s": round(out["cpb"]["calibration_s"], 4),
           "dc_calibration_slices": out["cpb"]["calibration_slices"],
           "rss_before_kb": out["cpb"]["rss_before_kb"],
           "rss_peak_kb": out["cpb"]["rss_peak_kb"],
           "model": "stub (synthetic)" if not cx.host else
           "fresh seed-2026 build (not a production checkpoint)"}
    if cx.host:
        cx.phase = "host"
    cx.ok(all(x["decode_128_s"] > 0 for x in per), "timings recorded")
    cx.record("CPB cost probe (P4 input; not a campaign projection)", rec)


def stb(cx: Ctx) -> None:
    """Addition C stability pre-check (registered config; fresh processes;
    P2 tolerances). Early check only -- formal P2 remains required."""
    import torch
    from seqref_mri.scripts import v05lt_epoch0 as E0
    if not cx.host:
        cx.record("STB", "host only (real model; registered CPU setup)")
        cx.ok(True, "STB synthetic part: none (host-only by design)")
        return
    cx.phase = "host"
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        e0 = E0.create(tmp / "epoch0", 2026, REPO_ROOT)
        ck = tmp / "epoch0" / "seed_2026" / "ckpt" / E0.ckpt_name(2026)
        idx = list(range(6))
        conf = {"checkpoint": "temporary epoch-0 seed 2026 (fixture)",
                "model_state_sha256": e0["model_state_sha256"],
                "slices": "validation view positions " + str(idx),
                "masks": "evaluation convention (no epoch)",
                "precision": "model float32; metrics float64",
                "torch_threads": torch.get_num_threads(),
                "deterministic_algorithms": False,
                "tolerances": {"psnr_db": 0.001, "nll_per_dim": 0.0001},
                "readouts": "R0 PSNR, NLL/dim, R1 primary 128 PSNR"}
        csha = C.write_durable_new(tmp / "stb_config.json",
                                   C.canonical_json(conf))
        cx.record("STB registered config sha256 (written before the test)",
                  {"sha256": csha, **conf})
        outs = [_run_eval_child(cx, tmp, {"kind": "stb", "indices": idx,
                                          "epoch0": str(ck), "epoch0_state":
                                          e0["model_state_sha256"]}, f"stb{i}")
                for i in (1, 2)]
    ba, bb = outs[0]["backend"], outs[1]["backend"]
    cx.ok(ba == bb and ba["torch_num_threads"] == conf["torch_threads"]
          and ba["deterministic_algorithms"] == conf["deterministic_algorithms"],
          "both children ran the registered backend (threads, deterministic "
          "settings, env) and identical backends", "STB_BACKEND_MISMATCH",
          f"{ba.get('torch_num_threads')}/{bb.get('torch_num_threads')}")
    cx.record("STB child backend (both)", ba)
    a, b = outs[0]["stb"], outs[1]["stb"]
    d = {k: max(abs(x - y) for x, y in zip(a[k], b[k]))
         for k in ("r0_psnr", "nll", "r1_psnr")}
    cx.ok(d["r0_psnr"] <= 0.001 and d["r1_psnr"] <= 0.001 and d["nll"]
          <= 0.0001, "two fresh processes within the P2 tolerances "
          "(0.001 dB, 0.0001 nats/dim)", detail=str(d))
    cx.record("STB max differences", {**d, "bitwise_identical":
                                      all(a[k] == b[k] for k in a)})


# ---------------------------------------------------------------------------
# I1.8(a) -- statistics: F06 (aggregation, slice completeness), F15
# (bootstrap, labels, AM3, readouts, DC identity, D1-D3/PA3), F04
# statistics-stream part (in f04), BSA (addition A: independent brute-force
# re-implementation), BSB (addition B: bitwise across two fresh processes),
# CFG (addition C: analysis configuration and its hash).
# All synthetic by design: no run result exists before training.
# ---------------------------------------------------------------------------
ST_VOLS = [f"v{i}" for i in range(6)]
ST_NSL, ST_MID = 3, 1
ST_REACHED_FULL = (3, 10, 20, 30, 50, 75)
ST_POINTS = (("shared", "epoch0"),) + tuple(
    (a, c) for a in ("lam0", "lam2") for c in ("epoch3", "selected", "final"))


def _S():
    from seqref_mri.scripts import v05lt_stats as S
    return S


def _st_file(v: str) -> str:
    return f"singlecoil_train/{v}.h5"


def _st_elig(all_mid_ineligible: bool = False) -> dict:
    """v5: no eligible slice (NMSE null volume); v4: midpoint ineligible;
    v1: slice 0 ineligible (not a midpoint)."""
    el = {}
    for v in ST_VOLS:
        for s in range(ST_NSL):
            e = not (v == "v5" or (v == "v4" and s == ST_MID)
                     or (v == "v1" and s == 0))
            if all_mid_ineligible and s == ST_MID:
                e = False
            el[(_st_file(v), s)] = e
    return el


def _st_slice_rows(el: dict = None) -> list:
    """Synthetic eligibility-manifest rows (fixture stand-in)."""
    el = _st_elig() if el is None else el
    return [{"volume_id": v, "file": _st_file(v), "slice_index": s,
             "eligible": el[(_st_file(v), s)], "midpoint": s == ST_MID}
            for v in ST_VOLS for s in range(ST_NSL)]


def _st_run_info() -> dict:
    """milestone 100 reached nowhere; 75 missing for lam0 seed 2027 (D3)."""
    S = _S()
    r = {(a, sd): {"milestones": list(ST_REACHED_FULL), "last_epoch": 80,
                   "status": "stopped"} for a in S.ARMS for sd in S.SEEDS}
    r[("lam0", 2027)] = {"milestones": [3, 10, 20, 30, 50],
                         "last_epoch": 60, "status": "stopped"}
    return r


def _st_sources(seed: int = 7, el: dict = None, run_info: dict = None,
                r0dc_all_identical: bool = False) -> dict:
    """Synthetic per-slice rows for every source the registered list needs
    (fixture data only; numbers carry no meaning). DC rows carry
    dc_identical: R1+DC identical everywhere; R0+DC not identical on one
    slice unless r0dc_all_identical."""
    S = _S()
    g = np.random.default_rng(seed)
    el = _st_elig() if el is None else el
    run_info = _st_run_info() if run_info is None else run_info
    base = {"lam0": 30.0, "lam2": 30.8, "shared": 20.0, "zf": 25.0}
    ck_off = {"epoch0": 0.0, "epoch3": -0.5, "selected": 0.0, "final": -0.1,
              "best_nll": -0.3}
    ro_off = {"r0": 0.0, "r1": 0.4, "r0_dc": 0.01, "r1_dc": 0.41, "zf": 0.0}

    def rows(arm, ck, ro, scope, nll):
        out = []
        for v in ST_VOLS:
            for s in (range(ST_NSL) if scope == "all" else [ST_MID]):
                f = _st_file(v)
                r = {"volume_id": v, "file": f, "slice_index": s,
                     "psnr": float(base[arm] + ck_off.get(ck, 0.2)
                                   + ro_off[ro] + g.normal(0, 0.3)),
                     "ssim": float(0.6 + g.normal(0, 0.01)),
                     "nmse_u": (float(0.9 + g.normal(0, 0.01))
                                if el[(f, s)] else None),
                     "psnr_capped": False, "zero_error": False}
                if ro in ("r0_dc", "r1_dc"):
                    r["dc_identical"] = not (
                        ro == "r0_dc" and not r0dc_all_identical
                        and arm == "lam2" and ck == "selected" and v == "v2")
                if nll:
                    r["nll_per_dim"] = float(1.5 + (0.004 if arm == "lam2"
                                                    else 0.0)
                                             + g.normal(0, 0.002))
                out.append(r)
        return out
    src = {}
    for sd in S.SEEDS:
        for arm in S.ARMS:
            cks = ["epoch3", "selected", "final", "best_nll"] + [
                f"epoch{m}" for m in sorted(run_info[(arm, sd)]["milestones"])
                if m != 3]
            for ck in cks:
                src[S.source_key(arm, sd, ck, "r0", "all")] = rows(
                    arm, ck, "r0", "all", True)
        for arm, ck in ST_POINTS:
            for ro in ("r0", "r1", "r0_dc", "r1_dc"):
                src[S.source_key(arm, sd, ck, ro, "mid")] = rows(
                    arm, ck, ro, "mid", False)
        src[S.source_key("shared", sd, "epoch0", "r0", "all")] = rows(
            "shared", "epoch0", "r0", "all", True)
    for sc in ("all", "mid"):
        src[S.source_key("zf", None, "zf", "zf", sc)] = rows(
            "zf", "zf", "zf", sc, False)
    return src


def _st_config(B: int, el: dict = None) -> dict:
    S = _S()
    return S.build_config(ST_VOLS, _st_slice_rows(el),
                          eligibility_sha256="synthetic",
                          split_sha256="synthetic", B=B,
                          fixture_override=True)


def _st_run(B: int, sources=None, el=None, run_info=None) -> dict:
    S = _S()
    cfg = _st_config(B, el)
    return S.analyse(cfg, S.config_sha256(cfg),
                     _st_sources(el=el) if sources is None else sources,
                     _st_slice_rows(el),
                     _st_run_info() if run_info is None else run_info)


def _st_by_id(res: dict) -> dict:
    return {c["id"]: c for c in res["comparisons"]}


def _st_drop(src: dict, key: str, file: str, slice_index: int) -> dict:
    """Copy of the sources with ONE row deleted (fixture deletion test)."""
    out = {k: list(v) for k, v in src.items()}
    out[key] = [r for r in out[key] if not (r["file"] == file and
                                             r["slice_index"] == slice_index)]
    C.require(len(out[key]) == len(src[key]) - 1, "FIXTURE_SETUP",
              "deletion did not remove exactly one row")
    return out


def f06(cx: Ctx) -> None:
    """Aggregation and slice completeness through the statistics path
    (194 §8.2, AM2-C, AM3-E4, AM4-A)."""
    S = _S()

    def row(v, s, **k):
        return {"volume_id": v, "file": _st_file(v), "slice_index": s, **k}
    exp = [("a", _st_file("a"), 0), ("a", _st_file("a"), 1),
           ("b", _st_file("b"), 0)]
    rows = [row("a", 0, psnr=10.0), row("a", 1, psnr=20.0),
            row("b", 0, psnr=40.0)]
    vals, miss, _ = S.volume_values(rows, "psnr", exp)
    eq = S.fmean(vals.values())
    cx.ok(vals == {"a": 15.0, "b": 40.0} and eq == 27.5 and not miss,
          "slice -> volume mean -> equal-weight mean (27.5, not the "
          "slice-pooled 23.33)", detail=str(vals))
    el = {(_st_file("a"), 0): True, (_st_file("a"), 1): True,
          (_st_file("b"), 0): False}
    nm = [row("a", 0, nmse_u=1.0), row("a", 1, nmse_u=1.0 / 9.0),
          row("b", 0, nmse_u=None)]
    vals, _, ev = S.volume_values(nm, "nmse_u", exp, eligible=el)
    cx.ok(vals["a"] == S.fmean([1.0, 1.0 / 9.0]) and vals["a"] != 0.2,
          "NMSE_u = mean of per-slice ratios (0.556), not pooled (0.2)",
          detail=str(vals["a"]))
    cx.ok(vals["b"] is None and ev == ["a"],
          "volume without an eligible slice -> None, outside the NMSE "
          "population")
    cx.raises("STATS_NMSE_ELIGIBILITY_MISMATCH", lambda: S.volume_values(
        [row("a", 0, nmse_u=1.0), row("a", 1, nmse_u=1.0),
         row("b", 0, nmse_u=0.5)], "nmse_u", exp, eligible=el),
        "ineligible slice carrying NMSE_u refused")
    cx.raises("STATS_ELIGIBILITY_MISSING", lambda: S.volume_values(
        nm, "nmse_u", exp), "NMSE_u without the eligibility index")
    # slice completeness (review finding): deletion, not only None values
    cx.raises("STATS_SLICE_MISSING", lambda: S.volume_values(
        rows[1:], "psnr", exp), "trained checkpoint: DELETED slice row is a "
        "validity failure (volume still has other rows)")
    vals, miss, _ = S.volume_values(rows[1:], "psnr", exp, reference=True)
    cx.ok(vals == {"a": None, "b": 40.0}
          and miss == [f"{_st_file('a')}|0 (row absent)"],
          "reference: DELETED slice row -> volume None with identity")
    cx.raises("STATS_VALUE_MISSING", lambda: S.volume_values(
        [row("a", 0, psnr=None), row("a", 1, psnr=1.0),
         row("b", 0, psnr=1.0)], "psnr", exp),
        "trained checkpoint: None value is a validity failure")
    vals, miss, _ = S.volume_values([row("a", 0, psnr=None),
                                     row("a", 1, psnr=3.0),
                                     row("b", 0, psnr=1.0)], "psnr", exp,
                                    reference=True)
    cx.ok(vals == {"a": None, "b": 1.0} and miss == [f"{_st_file('a')}|0"],
          "reference: None value -> volume None with identity (AM3-E4)")
    cx.raises("STATS_SLICE_UNEXPECTED", lambda: S.volume_values(
        rows + [row("a", 2, psnr=1.0)], "psnr", exp),
        "unexpected slice refused")
    cx.raises("STATS_SLICE_VOLUME_MISMATCH", lambda: S.volume_values(
        [row("a", 0, psnr=1.0), row("a", 1, psnr=1.0),
         {**row("b", 0, psnr=1.0), "volume_id": "a"}], "psnr", exp),
        "slice assigned to the wrong volume refused")
    cx.raises("STATS_DUPLICATE_SLICE", lambda: S.volume_values(
        rows + [row("a", 0, psnr=1.0)], "psnr", exp),
        "duplicate slice refused")
    cx.raises("STATS_NON_FINITE", lambda: S.volume_values(
        [row("a", 0, psnr=float("nan"))] + rows[1:], "psnr", exp),
        "non-finite value refused")
    cx.raises("STATS_METRIC_UNKNOWN", lambda: S.volume_values(
        rows, "mse", exp), "unregistered metric refused")
    capped = [row("a", 0, psnr=120.0, psnr_capped=True, zero_error=True),
              row("a", 1, psnr=60.0, psnr_capped=False, zero_error=False),
              row("b", 0, psnr=120.0, psnr_capped=True, zero_error=False)]
    vals, miss, _ = S.volume_values(capped, "psnr", exp)
    cc = S.cap_counts(capped)
    cx.ok(vals == {"a": 90.0, "b": 120.0} and not miss
          and cc["psnr_capped"]["count"] == 2
          and cc["zero_error"]["identities"] == [f"{_st_file('a')}|0"],
          "capped 120 dB retained in aggregates; cap / zero-error counts "
          "with identities (AM4-A)", detail=str(cc))
    # analysis level: the reviewer's deletion scenario
    src = _st_sources()
    kt = S.source_key("lam2", 2026, "selected", "r0", "all")
    cx.raises("STATS_SLICE_MISSING", lambda: _st_run(
        50, sources=_st_drop(src, kt, _st_file("v3"), 2)),
        "analysis: one deleted trained slice -> validity failure (no "
        "silent averaging)")
    k0 = S.source_key("shared", 2026, "epoch0", "r0", "all")
    res = _st_run(50, sources=_st_drop(src, k0, _st_file("v3"), 2))
    hit = sorted(c["id"] for c in res["comparisons"]
                 if c["status"] == "unavailable")
    want = sorted([f"am3_epoch0|{a}|{ck}|r0|{m}" for a in S.ARMS
                   for ck in ("epoch3", "selected", "final")
                   for m in ("psnr", "ssim", "nmse_u", "nll_per_dim")
                   if m != "nll_per_dim"]
                  + [f"am3_epoch0|{a}|{ck}|nll|delta" for a in S.ARMS
                     for ck in ("epoch3", "selected", "final")]
                  + [f"am3_zf|shared|epoch0|r0|{m}"
                     for m in ("psnr", "ssim", "nmse_u")])
    ident = f"{k0}: {_st_file('v3')}|2 (row absent)"
    cx.ok(hit == want and all(ident in c["identities"] for c in
                              res["comparisons"] if c["id"] in want),
          "analysis: one deleted epoch-0 slice -> every comparison using "
          "that source unavailable, with the identity", detail=str(len(hit)))
    cx.raises("STATS_SLICE_UNEXPECTED", lambda: _st_run(50, sources={
        **src, kt: src[kt] + [{**src[kt][0], "slice_index": 9}]}),
        "analysis: unexpected extra slice refused")
    base = _st_run(50)
    c = _st_by_id(base)["between|selected|r0|psnr"]
    exp_all = S.expected_slices(ST_VOLS, _st_slice_rows())["all"]
    d = []
    for sd in S.SEEDS:
        a, _, _ = S.volume_values(src[S.source_key(
            "lam2", sd, "selected", "r0", "all")], "psnr", exp_all)
        r, _, _ = S.volume_values(src[S.source_key(
            "lam0", sd, "selected", "r0", "all")], "psnr", exp_all)
        d += [a[v] - r[v] for v in ST_VOLS]
    cx.ok(c["estimate"] == S.fmean(d), "observed dPSNR = mean over seeds "
          "and volumes of volume-level differences", detail=str(c["estimate"]))
    by = _st_by_id(base)
    cx.ok(by["between|selected|r0|nmse_u"]["n_volumes"] == 5
          and by["between|selected|r1|nmse_u"]["n_volumes"] == 4,
          "NMSE populations from the manifest: all-slice 5, midpoint 4 "
          "volumes (PSNR/SSIM/NLL keep all 6)")
    cfg = _st_config(50)
    other = _st_slice_rows()
    other[0] = {**other[0], "eligible": not other[0]["eligible"]}
    cx.raises("STATS_SLICES_MISMATCH", lambda: S.analyse(
        cfg, S.config_sha256(cfg), src, other, _st_run_info()),
        "one slice's eligibility differing from the configuration's "
        "binding refused")
    moved = _st_slice_rows()
    moved[-1] = {**moved[-1], "slice_index": 7}
    cx.raises("STATS_SLICES_MISMATCH", lambda: S.analyse(
        cfg, S.config_sha256(cfg), src, moved, _st_run_info()),
        "different expected slice identities refused")
    bad = json.loads(json.dumps(cfg))
    bad["populations"][1]["volumes"] = ST_VOLS[:4]
    cx.raises("STATS_NMSE_POPULATION_MISMATCH", lambda: S.analyse(
        bad, S.config_sha256(bad), src, _st_slice_rows(), _st_run_info()),
        "configured NMSE population differing from the manifest refused (D2)")
    cx.record("F06 synthetic estimate (between|selected|r0|psnr)",
              c["estimate"])


def f15(cx: Ctx) -> None:
    """Bootstrap, labels, AM3, readouts, DC identity, D1-D4 / PA3 (194 §9)."""
    import time
    S = _S()
    cfg = _st_config(2000)
    draws, rec = S.draw_indices(cfg)
    d = draws["val"]
    cx.ok(d["seed"].shape == (2000, 3) and d["vol"].shape == (2000, 6)
          and d["seed"].min() >= 0 and d["seed"].max() <= 2
          and d["vol"].min() >= 0 and d["vol"].max() <= 5
          and all(d["vo"][sd].shape == (2000, 6) for sd in S.SEEDS),
          "index arrays: B x 3 seeds, B x V volumes, per-seed B x V")
    A = np.array([[1000.0 * k + j for j in range(6)] for k in range(3)])
    R = A - np.array([[0.1 * j for j in range(6)]] * 3)
    out = S.bootstrap_comparison(A, R, "delta", d, keep=True)
    exact = all(out["resamples"][b] == S.fmean(
        [A[d["seed"][b][i], d["vol"][b][j]]
         - R[d["seed"][b][i], d["vol"][b][j]]
         for i in range(3) for j in range(6)]) for b in range(2000))
    cx.ok(exact, "every crossed resample = mean over (sampled seed x the "
          "ONE sampled volume list) cells of side - reference")
    cx.ok(max(out["resamples"]) < 1.0,
          "pairing preserved: seed offsets (1000 x k) cancel between sides")
    cx.ok([p["seed"] for p in out["per_seed"]] == list(S.SEEDS)
          and all(len(p["ci"]) == 2 for p in out["per_seed"]),
          "per-seed volume-only results for each seed")
    A2, R2 = np.array([[2.0, 4.0]] * 3), np.array([[1.0, 3.0]] * 3)
    cx.ok(S._stat(A2, R2, "ratio") == 0.5,
          "r = ratio of means (0.5), not mean of ratios (0.667)")
    cx.ok(S._stat(np.full((3, 2), -1.0), np.full((3, 2), -2.0), "ratio")
          == 0.5 and S._stat(np.full((3, 2), -3.0), np.full((3, 2), -2.0),
                             "ratio") == -0.5,
          "negative N_ref: positive r = NLL increased (worse)")
    cx.raises("STATS_R_ZERO_REFERENCE", lambda: S._stat(
        np.ones((3, 2)), np.zeros((3, 2)), "ratio"), "zero reference mean "
        "refused (no substitute)")
    eps = 1e-12
    cx.ok(S.support_label(0.0, 1.0) == "not established"
          and S.support_label(eps, 1.0) == "supported improvement"
          and S.support_label(-1.0, 0.0) == "not established"
          and S.support_label(-1.0, -eps) == "supported deterioration",
          "support: CI touching 0 -> not established")
    cx.ok(S.practical_label(0.5, 1.0) == "straddles reference"
          and S.practical_label(0.5 + eps, 1.0) == "above reference"
          and S.practical_label(0.0, 0.5) == "straddles reference"
          and S.practical_label(0.0, 0.5 - eps) == "below reference",
          "practical: CI touching +0.5 dB straddles")
    cx.ok(S.density_label(0.0, 0.01) == "unclear"
          and S.density_label(0.0, 0.01 - eps) == "preserved"
          and S.density_label(0.01, 0.02) == "unclear"
          and S.density_label(0.01 + eps, 0.02) == "worse",
          "density: CI touching +1% -> unclear")
    cx.ok([S.outcome("supported improvement", 0.1, 1, x) for x in
           ("preserved", "worse", "unclear")] == [1, 2, 3]
          and S.outcome("not established", -0.4, 0.4, "worse") == 4
          and S.outcome("not established", -0.5, 0.4, "preserved") == 5
          and S.outcome("not established", -0.4, 0.5, "preserved") == 5
          and S.outcome("not established", -0.6, 0.7, "preserved") == 5
          and S.outcome("supported deterioration", -1, -0.1, "preserved")
          == 6, "outcomes 1-6; CI touching -0.5 or +0.5 dB -> 5")
    x = [float(i) for i in range(10_000)]
    p = [S.percentile_linear(x, q) for q in S.LEVELS]
    cx.ok(p == [249.975, 9749.025], "D4: linear percentiles at exact "
          "positions (B = 10,000: h = 249.975, 9749.025)", detail=str(p))
    y = sorted(np.random.default_rng(3).normal(size=10_000).tolist())
    q = [S.percentile_linear(y, q) for q in S.LEVELS]
    ref = np.percentile(np.array(y), [2.5, 97.5], method="linear").tolist()
    cx.ok(all(abs(a - b) <= 1e-12 for a, b in zip(q, ref)),
          "D4 formula agrees with numpy linear percentile (informational, "
          "|diff| <= 1e-12)", detail=str([a - b for a, b in zip(q, ref)]))
    cx.ok(S.percentile_linear([float(i) for i in range(41)], S.LEVELS[0])
          == 1.0, "integer position (t = 0) returns the order statistic")
    # whole analysis (B = 500 fixture configuration)
    res = _st_run(500)
    by = _st_by_id(res)
    am3 = [c for c in res["comparisons"] if c["family"].startswith("am3")]
    cx.ok(am3 and all("support" not in c and "density" not in c
                      and "practical" not in c for c in am3),
          "AM3 comparisons carry no labels")
    cx.ok(all(c["stat"] == "delta" for c in am3
              if c["metric"] == "nll_per_dim")
          and all(c["id"].split("|")[1] in S.ARMS for c in am3
                  if c["metric"] == "nll_per_dim"),
          "raw NLL differences only, and only trained vs epoch 0")
    cx.ok(not [c for c in res["comparisons"] if c["family"] == "am3_zf"
               and c["metric"] == "nll_per_dim"],
          "ZF comparisons: PSNR, SSIM, NMSE_u only (no NLL)")
    lab = [c for c in res["comparisons"] if c["family"] in
           ("between", "within") and c["status"] == "ok"
           and not c.get("dc_identity")]
    cx.ok(all(("support" in c) == (c["metric"] == "psnr") and
              ("density" in c) == (c["metric"] == "nll_per_dim")
              for c in lab), "between / within: support + practical on "
          "dPSNR, density on r; SSIM / NMSE_u unlabelled")
    cx.ok(all(c["status"] == "ok" for c in res["comparisons"]
              if c["metric"] == "nmse_u" and c["milestone"] is None),
          "NMSE ineligibility never makes a comparison unavailable (AM3-G2)")
    # readout comparisons (194 §8, plot 3)
    ro = [c for c in res["comparisons"] if c["family"] == "readout"]
    cx.ok(len(ro) == 63 and all(c["scope"] == "mid" for c in ro),
          "readout family: 7 points x (R1-R0, R0+DC-R0, R1+DC-R1) x 3 "
          "metrics on the midpoint slices")
    r10 = by["readout|lam2|selected|r1-r0|psnr"]
    srcs = _st_sources()
    exp_mid = S.expected_slices(ST_VOLS, _st_slice_rows())["mid"]
    dd = []
    for sd in S.SEEDS:
        a, _, _ = S.volume_values(srcs[S.source_key(
            "lam2", sd, "selected", "r1", "mid")], "psnr", exp_mid)
        b, _, _ = S.volume_values(srcs[S.source_key(
            "lam2", sd, "selected", "r0", "mid")], "psnr", exp_mid)
        dd += [a[v] - b[v] for v in ST_VOLS]
    cx.ok(r10["estimate"] == S.fmean(dd) and "support" in r10
          and "practical" not in r10,
          "R1 - R0 on identical midpoint slices through the engine; support "
          "label only (no practical / outcome)", detail=str(r10["estimate"]))
    cx.ok("support" not in by["readout|shared|epoch0|r1-r0|psnr"],
          "epoch-0 readout comparison descriptive (AM3)")
    # DC identity (194 §8): decided per comparison from its own sources
    cx.ok(res["dc_identity"]["r1_dc"]["identical_all"] and not
          res["dc_identity"]["r0_dc"]["identical_all"]
          and res["dc_identity"]["r0_dc"]["non_identical_count"] == 3,
          "campaign-wide DC identity summary reported separately",
          detail=str({k: v["non_identical_count"] for k, v in
                      res["dc_identity"].items()}))
    w0, w2 = by["within|lam0|r0_dc|psnr"], by["within|lam2|r0_dc|psnr"]
    cx.ok(w0["dc_identity"] and "support" not in w0
          and w2["dc_identity"] is False and "support" in w2,
          "per-comparison identity: lam0 R0+DC (own sources identical) -> "
          "identity result; lam2 R0+DC (non-identical slice at selected) -> "
          "labelled, although the campaign-wide R0+DC is not identical")
    # regression (review finding): an unrelated checkpoint cannot change
    # another comparison's classification
    src_u = _st_sources()
    ku = S.source_key("shared", 2026, "epoch0", "r1_dc", "mid")
    src_u[ku] = [{**r, "dc_identical": False} if i == 0 else r
                 for i, r in enumerate(src_u[ku])]
    res_u = _st_run(500, sources=src_u)
    by_u = _st_by_id(res_u)
    oc_u = {o["id"]: o for o in res_u["outcomes"]}
    sel = [i for i in by if "|selected|" in i and "r1_dc" in i
           and not i.startswith("am3_epoch0")]   # those use epoch 0 itself
    hit = sorted(i for i in by if by_u[i] != by[i])
    cx.ok(all(i.startswith("am3_epoch0") or "|shared|epoch0|" in i
              for i in hit) and hit,
          "only comparisons that use the changed epoch-0 source differ",
          detail=str(len(hit)))
    cx.ok(sel and all(by_u[i] == by[i] for i in sel)
          and oc_u["between|selected|r1_dc"] == {o["id"]: o for o in
                                                 res["outcomes"]}[
              "between|selected|r1_dc"]
          and by_u["between|selected|r1_dc|psnr"]["dc_identity"],
          "changing one epoch-0 R1+DC identity flag leaves every selected-"
          "checkpoint R1+DC comparison and outcome unchanged (identity "
          "result)", detail=str(len(sel)))
    e0 = by_u["readout|shared|epoch0|r1_dc-r1|psnr"]
    cx.ok(e0["dc_identity"] is False and
          by["readout|shared|epoch0|r1_dc-r1|psnr"]["dc_identity"],
          "the changed flag affects only comparisons using that source")
    i1 = by["between|selected|r1_dc|psnr"]
    oc = {o["id"]: o for o in res["outcomes"]}
    cx.ok(i1["dc_identity"] and "support" not in i1 and "identity_result"
          in i1 and oc["between|selected|r1_dc"]["outcome"] is None
          and oc["between|selected|r1_dc"]["reason"] == S.IDENTITY_TEXT,
          "R1+DC identical everywhere -> identity result, no labels, no "
          "outcome")
    i0 = by["between|selected|r0_dc|psnr"]
    cx.ok(i0["dc_identity"] is False and "support" in i0 and
          oc["between|selected|r0_dc"]["outcome"] in range(1, 7),
          "R0+DC not identical on all slices -> labelled treatment")
    src_bad = _st_sources()
    kd = S.source_key("lam0", 2026, "final", "r0_dc", "mid")
    src_bad[kd] = [{k: v for k, v in r.items() if k != "dc_identical"}
                   for r in src_bad[kd]]
    cx.raises("STATS_DC_IDENTITY_MISSING", lambda: _st_run(
        50, sources=src_bad), "DC row without the identity flag refused")
    # D3 / PA3
    m10, m75 = by["between|epoch10|r0|psnr"], by["between|epoch75|r0|psnr"]
    m100 = by["between|epoch100|r0|psnr"]
    cx.ok(m10["status"] == "ok" and "support" in m10 and "seeds_absent"
          not in m10, "D3: milestone reached by both arms in all seeds -> "
          "labelled")
    cx.ok(m75["status"] == "descriptive_d3" and "ci" not in m75
          and "support" not in m75
          and [p["seed"] for p in m75["per_seed"]] == [2026, 2028]
          and m75["seeds_reached"] == [2026, 2028]
          and m75["seeds_absent"] == [{"seed": 2027,
                                       "arms_without_milestone": ["lam0"],
                                       "runs": {"lam0": {"last_epoch": 60,
                                                         "status": "stopped"}}}],
          "PA3: milestone 75 -> per-seed descriptive for 2026, 2028; lists "
          "seeds reached, the absent seed, the arm and how its run ended")
    cx.ok(m100["status"] == "descriptive_d3" and m100["per_seed"] == []
          and m100["seeds_reached"] == [] and len(m100["seeds_absent"]) == 3,
          "PA3: milestone reached in no seed -> descriptive, reasons listed")
    cx.ok(oc["between|epoch75|r0"]["outcome"] is None
          and oc["between|epoch10|r0"]["outcome"] in range(1, 7)
          and oc["between|selected|r0"]["outcome"] in range(1, 7),
          "outcome table: labelled groups only (no fewer-seed labels)")
    cx.ok(all(len(c["per_seed"]) == 3 for c in res["comparisons"]
              if c["status"] == "ok"), "every complete comparison reports "
          "all three per-seed results")
    ri = _st_run_info()
    del ri[("lam2", 2028)]
    cx.raises("STATS_RUN_INFO_MISSING", lambda: _st_run(50, run_info=ri),
              "missing run ledger information refused")
    # D1
    shas = [r.get("crossed_volume_sha256") for r in res["stream"]]
    cx.ok(len(res["stream"]) == 3 and len(set(shas)) == 3,
          "D1: three distinct populations -> three distinct array sets")
    eq = _st_config(500)
    eq["populations"][1]["volumes"] = list(ST_VOLS)
    _, rec2 = S.draw_indices(eq)
    cx.ok(rec2[1] == {"population": "nmse_all", "shares": "val"},
          "D1: identical ordered population shares the arrays")
    # AM3-E4 with a None value
    src = _st_sources()
    k0 = S.source_key("shared", 2027, "epoch0", "r0", "all")
    src[k0][4] = {**src[k0][4], "psnr": None}
    res2 = _st_run(500, sources=src)
    by2 = _st_by_id(res2)
    hit = [c for c in res2["comparisons"] if c["status"] == "unavailable"]
    exp_ids = sorted([f"am3_epoch0|{a}|{ck}|r0|psnr" for a in S.ARMS
                      for ck in ("epoch3", "selected", "final")]
                     + ["am3_zf|shared|epoch0|r0|psnr"])
    cx.ok(sorted(c["id"] for c in hit) == exp_ids and all(
        c["identities"] == [f"{k0}: {_st_file('v1')}|1"] for c in hit),
        "AM3-E4: missing epoch-0 value -> exactly the affected comparisons "
        "unavailable, with identities", detail=str(len(hit)))
    same = all(by2[i] == by[i] for i in by if i not in exp_ids)
    cx.ok(same and res2["stream"] == res["stream"],
          "other comparisons unchanged; no re-draw, no re-bootstrap of the "
          "remaining volumes")
    # D2
    el = _st_elig(all_mid_ineligible=True)
    res3 = _st_run(200, sources=_st_sources(el=el), el=el)
    mids = [c for c in res3["comparisons"] if c["metric"] == "nmse_u"
            and c["scope"] == "mid"]
    cx.ok(mids and all(c["status"] == "unavailable_no_eligible_volume"
                       and "estimate" not in c for c in mids),
          "D2: empty NMSE population -> unavailable (no estimate)")
    # refusals
    cfg5 = _st_config(50)
    cx.raises("STATS_CONFIG_UNAPPROVED", lambda: S.analyse(
        cfg5, "0" * 64, _st_sources(), _st_slice_rows(), _st_run_info()),
        "configuration hash differs from the approved one")
    cx.raises("STATS_FIXTURE_CONFIG", lambda: S.analyse(
        cfg5, S.config_sha256(cfg5), _st_sources(), _st_slice_rows(),
        _st_run_info(), production=True), "fixture configuration refused in "
        "production")
    src = _st_sources()
    del src[S.source_key("lam2", 2026, "selected", "r1", "mid")]
    cx.raises("STATS_SOURCE_MISSING", lambda: S.analyse(
        cfg5, S.config_sha256(cfg5), src, _st_slice_rows(), _st_run_info()),
        "missing trained source refused")
    reg = S.build_config(ST_VOLS, _st_slice_rows(),
                         eligibility_sha256="synthetic",
                         split_sha256="synthetic")
    t0 = time.time()
    dr, _ = S.draw_indices(reg)
    full = S.bootstrap_comparison(A, R, "delta", dr["val"])
    cx.ok(reg["bootstrap"]["B"] == 10_000 and len(full["ci"]) == 2
          and full["ci"][0] <= full["estimate"] <= full["ci"][1],
          "registered B = 10,000 runs (crossed + per-seed)",
          detail=f"{time.time() - t0:.2f} s")


def _bsa_draws(cfg: dict) -> dict:
    """Independent replay of the registered draw order (DRAW_ORDER)."""
    B = cfg["bootstrap"]["B"]
    g = np.random.default_rng(20261022)   # fixture replay of 194 §3.1
    out, seen = {}, {}
    for p in cfg["populations"]:
        key = tuple(p["volumes"])
        if key in seen:
            out[p["id"]] = out[seen[key]]
            continue
        seen[key] = p["id"]
        if not key:
            out[p["id"]] = None
            continue
        s = g.integers(0, 3, size=(B, 3), dtype=np.int64).tolist()
        v = g.integers(0, len(key), size=(B, len(key)),
                       dtype=np.int64).tolist()
        vo = {sd: g.integers(0, len(key), size=(B, len(key)),
                             dtype=np.int64).tolist()
              for sd in (2026, 2027, 2028)}
        out[p["id"]] = (s, v, vo)
    return out


def _bsa_vol(rows, metric, vol, el):
    vals = [r[metric] for r in rows if r["volume_id"] == vol and not (
        metric == "nmse_u" and not el[(r["file"], r["slice_index"])])]
    return math.fsum(vals) / len(vals)


def _bsa_stat(a: list, r: list, stat: str) -> float:
    if stat == "delta":
        return math.fsum(x - y for x, y in zip(a, r)) / len(a)
    ma, mr = math.fsum(a) / len(a), math.fsum(r) / len(r)
    return (ma - mr) / abs(mr)


def _bsa_pct(vals: list) -> list:
    x, n = sorted(vals), len(vals)
    out = []
    for num in (25, 975):
        hn = (n - 1) * num
        lo, rem = hn // 1000, hn % 1000
        out.append(x[lo] if rem == 0 else x[lo] + (rem / 1000) *
                   (x[lo + 1] - x[lo]))
    return out


def bsa(cx: Ctx) -> None:
    """Addition A: independent brute-force bootstrap; exact equality."""
    S = _S()
    B, el = 500, _st_elig()
    cfg, src = _st_config(B, el), _st_sources(el=el)
    res = S.analyse(cfg, S.config_sha256(cfg), src, _st_slice_rows(el),
                    _st_run_info())
    eng = _st_by_id(res)
    dr = _bsa_draws(cfg)
    pops = {p["id"]: p["volumes"] for p in cfg["populations"]}
    ident = {k: all(r["dc_identical"] for r in rows)
             for k, rows in src.items()
             if k.split("|")[3] in ("r0_dc", "r1_dc")}
    seeds = (2026, 2027, 2028)
    n, bad = 0, []
    for c in cfg["comparisons"]:
        e = eng[c["id"]]
        if e["status"] != "ok":
            continue
        vols = pops[c["population"]]

        def mat(side):
            out = {}
            for sd in seeds:
                k = (f"{side['arm']}|{'-' if side['arm'] == 'zf' else sd}|"
                     f"{side['ckpt']}|{side['readout']}|{c['scope']}")
                out[sd] = {v: _bsa_vol(src[k], c["metric"], v, el)
                           for v in vols}
            return out
        a, r = mat(c["side"]), mat(c["ref"])
        s, v, vo = dr[c["population"]]
        cr = []
        for b in range(B):
            ca = [a[seeds[i]][vols[j]] for i in s[b] for j in v[b]]
            cb = [r[seeds[i]][vols[j]] for i in s[b] for j in v[b]]
            cr.append(_bsa_stat(ca, cb, c["stat"]))
        est = _bsa_stat([a[sd][x] for sd in seeds for x in vols],
                        [r[sd][x] for sd in seeds for x in vols], c["stat"])
        ps = []
        for sd in seeds:
            rs = [_bsa_stat([a[sd][vols[j]] for j in vo[sd][b]],
                            [r[sd][vols[j]] for j in vo[sd][b]], c["stat"])
                  for b in range(B)]
            ps.append({"seed": sd, "estimate": _bsa_stat(
                [a[sd][x] for x in vols], [r[sd][x] for x in vols],
                c["stat"]), "ci": _bsa_pct(rs)})
        mine = {"estimate": est, "ci": _bsa_pct(cr), "per_seed": ps}
        dkeys = [f"{sd_['arm']}|{s_}|{sd_['ckpt']}|{sd_['readout']}|"
                 f"{c['scope']}" for sd_ in (c["side"], c["ref"])
                 if sd_["readout"] in ("r0_dc", "r1_dc") for s_ in seeds]
        identity = bool(dkeys) and all(ident[k] for k in dkeys)
        lo, hi = mine["ci"]
        if c["labels"] in ("psnr", "support") and not identity:
            mine["support"] = ("supported improvement" if lo > 0 else
                               "supported deterioration" if hi < 0 else
                               "not established")
        if c["labels"] == "psnr" and not identity:
            mine["practical"] = ("above reference" if lo > 0.5 else
                                 "below reference" if hi < 0.5 else
                                 "straddles reference")
        if c["labels"] == "density":
            mine["density"] = ("preserved" if hi < 0.01 else "worse" if
                               lo > 0.01 else "unclear")
        lab_keys = {"support", "practical", "density"}
        if any(e.get(k) != mine[k] for k in mine) or (
                lab_keys & set(e)) != (lab_keys & set(mine)):
            bad.append(c["id"])
        n += 1
    cx.ok(n > 250 and not bad, f"engine == independent brute force exactly "
          f"for {n} comparisons (estimates, CIs, per-seed, labels incl. DC "
          f"identity)", "BSA_MISMATCH", str(bad[:5]))
    shas = {pid: hashlib.sha256(np.array(t[1], dtype=np.int64).tobytes())
            .hexdigest() for pid, t in dr.items() if t is not None}
    cx.ok(all(rec.get("crossed_volume_sha256") == shas[rec["population"]]
              for rec in res["stream"] if "crossed_volume_sha256" in rec),
          "independent replay of the stream = recorded array sha256")
    cx.record("BSA comparisons checked", n)


def _bsb_child(out: str) -> None:
    logging.basicConfig(level=logging.WARNING)
    res = _st_run(300)
    Path(out).write_text(C.sha256_bytes(C.canonical_json(res)))


def bsb(cx: Ctx) -> None:
    """Addition B: bitwise-identical engine output in two fresh processes
    (tested environments only)."""
    ctx = multiprocessing.get_context("spawn")
    shas = []
    with tempfile.TemporaryDirectory() as t:
        for i in range(2):
            o = Path(t) / f"bsb{i}.txt"
            p = ctx.Process(target=_bsb_child, args=(str(o),))
            p.start()
            code = join_bounded(p, "spawn", timeout=1800)
            cx.ok(code == 0, f"child {i} exit 0", "CHILD_FAILED", str(code))
            shas.append(o.read_text())
    here = C.sha256_bytes(C.canonical_json(_st_run(300)))
    cx.ok(shas[0] == shas[1] == here, "analysis bytes identical in two fresh "
          "processes and in-process", detail=here)
    cx.record("BSB analysis sha256 (synthetic, B = 300)", here)


def cfg_group(cx: Ctx) -> None:
    """Addition C: analysis configuration, hash, write-once, refusals."""
    S = _S()
    rows = _st_slice_rows()
    mk = lambda r=rows, **k: S.build_config(  # noqa: E731
        ST_VOLS, r, eligibility_sha256="synthetic", split_sha256="synthetic",
        **k)
    a, b = mk(), mk()
    cx.ok(S.config_bytes(a) == S.config_bytes(b),
          "configuration deterministic (canonical bytes identical)")
    cx.ok(a["bootstrap"]["B"] == 10_000 and a["bootstrap"]["seed"] ==
          20261022 and a["percentile"]["levels"] == ["1/40", "39/40"]
          and "linear interpolation" in a["percentile"]["method"]
          and set(a["decisions"]) == {"D1", "D2", "D3", "D4"}
          and "draw_order" in a["bootstrap"] and not a["fixture_override"]
          and len(a["comparisons"]) == 296 and len(a["outcome_groups"]) == 19
          and a["slices"]["all"]["n"] == 18 and a["slices"]["mid"]["n"] == 6
          and "D-UNC" in a["out_of_scope"] and a["dc_identity_rule"],
          "configuration records B, stream, draw order, percentile method, "
          "D1-D4, slice bindings, comparison list (296), outcome groups (19)")
    cx.ok(a["populations"][1]["volumes"] == ST_VOLS[:5]
          and a["populations"][2]["volumes"] == ST_VOLS[:4],
          "NMSE populations derived from the manifest rows (D2)")
    cx.raises("STATS_CONFIG_INVALID", lambda: mk(B=500),
              "B != 10,000 refused without the fixture override")
    cx.ok(mk(B=500, fixture_override=True)["fixture_override"] is True,
          "fixture override is recorded in the configuration")
    cx.raises("STATS_CONFIG_INVALID", lambda: mk(rows + [rows[0]]),
              "duplicate slice row refused")
    cx.raises("STATS_CONFIG_INVALID", lambda: mk(
        [{**r, "midpoint": True} if r["volume_id"] == "v0" else r
         for r in rows]), "volume with more than one midpoint refused")
    cx.raises("STATS_CONFIG_INVALID", lambda: mk(
        [r for r in rows if r["volume_id"] != "v3"]),
        "registered volume without slice rows refused")
    cx.raises("STATS_CONFIG_INVALID", lambda: mk(
        rows + [{**rows[0], "volume_id": "vx", "file": "x.h5"}]),
        "slice row of an unregistered volume refused")
    sha = S.config_sha256(a)
    alt = json.loads(json.dumps(a))
    alt["thresholds"]["practical_db"] = 0.4
    cx.ok(S.config_sha256(alt) != sha, "any edit changes the hash")
    cx.raises("STATS_CONFIG_UNAPPROVED", lambda: S.analyse(
        alt, sha, _st_sources(), rows, _st_run_info()),
        "edited configuration refused against the approved hash")
    alt2 = _st_config(50)
    alt2["comparisons"] = alt2["comparisons"][:-1]
    cx.raises("STATS_CONFIG_INVALID", lambda: S.analyse(
        alt2, S.config_sha256(alt2), _st_sources(), rows, _st_run_info()),
        "comparison list other than the registered one refused even with a "
        "matching hash")
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "v05lt_analysis_config.json"
        got = S.write_json_once(p, a)
        side = (Path(t) / "v05lt_analysis_config.json.sha256").read_text()
        cx.ok(got == sha == C.sha256_file(p) and side.startswith(sha),
              "written bytes and sidecar = configuration sha256")
        cx.raises("ARTIFACT_EXISTS", lambda: S.write_json_once(p, a),
                  "second write refused (write-once)")
        ln = Path(t) / "link.json"
        os.symlink(Path(t) / "nowhere.json", ln)
        cx.raises("ARTIFACT_EXISTS", lambda: S.write_json_once(ln, a),
                  "dangling symlink at the destination refused, not followed")
        cx.ok(not (Path(t) / "nowhere.json").exists(), "nothing created at "
              "the link target")
    reg = C.Paths().v05lt / "analysis" / "v05lt_analysis_config.json"
    cx.raises("STATS_PRODUCTION_DISABLED", lambda: S.write_json_once(reg, a),
              "registered analysis path refused without the gated stage")
    cx.ok(S.main([]) == 2, "CLI refuses production (STATS_PRODUCTION_"
          "DISABLED)")
    cx.record("CFG registered-B synthetic configuration sha256", sha)


# ---------------------------------------------------------------------------
# I1.8 b0 -- VRF (G1 validation-row files), NLD (G3 NLL components), PEV
# (G2 post-run evaluation of trained checkpoints)
# ---------------------------------------------------------------------------
def vrf(cx: Ctx) -> None:
    """G1: per-epoch validation-row files staged before the commit, bound
    to it, retained, verified on load; crash, tamper and write-failure."""
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.scripts import v05lt_train as T
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        data = _syn_train_data(tmp / "syn")
        ua = tmp / "u"
        _stub_run(_stub_cfg(ua, max_epochs=4), data).start()
        st = C.ArtifactStore(ua / "ckpt", C.Ledger(ua / "ledger.jsonl"),
                             ua / "_incomplete")
        final = st.committed()
        cm = {r["body"]["epoch"]: r["body"] for r in
              C.Ledger(ua / "ledger.jsonl").records() if r["kind"] == "commit"}
        bad = []
        for e, b in sorted(cm.items()):
            vr = b["validation_rows"]
            doc = T.load_val_rows(ua / "ckpt" / vr["name"], vr["sha256"], e)
            summ = json.loads(json.dumps(M.summarise(doc["rows"])))
            if not (b["retained"][f"val_rows_e{e:03d}"]["sha256"] ==
                    vr["sha256"] and vr["name"] == f"val_e{e:03d}.json"
                    and doc["model_state_sha256"] == b["model_state_sha256"]
                    and doc["n_rows"] == len(data["vv"]) == vr["n_rows"]
                    and doc["slice_ids_sha256"] == vr["slice_ids_sha256"]
                    and summ == b["validation"]
                    and all({"nll_prior_per_dim", "nll_logdet_per_dim",
                             "nll_decomp_abs_dev"} <= set(r)
                            for r in doc["rows"])):
                bad.append(e)
        cx.ok(len(cm) == 4 and not bad, "every epoch: row file committed "
              "(role + sha256 + epoch + count + slice identities), model "
              "state = commit; the summary recomputed from the file equals "
              "the committed validation summary (the values that drove "
              "stopping and selection)", detail=str(bad))
        cx.ok(all(f"val_rows_e{e:03d}" in final for e in cm),
              "row files of every epoch retained to the end (never "
              "superseded)")
        # trainer-written weights file -> verified evaluator loader (meta
        # layout bound to the trainer, identity from the commit record)
        from seqref_mri.scripts import v05lt_eval as EV
        from seqref_mri.scripts.v05lt_stub_train import build_stub
        w3 = final["milestone_003"]
        d3 = T.load_val_rows(ua / "ckpt" / cm[3]["validation_rows"]["name"],
                             cm[3]["validation_rows"]["sha256"], 3)
        _, ck3 = EV.load_trained_checkpoint(
            ua / "ckpt" / w3["name"], w3["sha256"], cm[3]["model_state_sha256"],
            seed=d3["seed"], arm=d3["arm"], epoch=3, model_factory=build_stub)
        cx.ok((ck3.epoch, ck3.model_state_sha256, ck3.seed, ck3.arm) ==
              (3, cm[3]["model_state_sha256"], d3["seed"], d3["arm"]) and
              "milestone_003" in ck3.roles,
              "trainer-written milestone weights load through the verified "
              "loader against the commit record (schema + meta + state)")
        # crash after the epoch-3 row file is published, before its commit
        ia = tmp / "i"
        cfg = _stub_cfg(ia, max_epochs=4)
        cnt = {"c": 0}

        def hook(stage):
            if stage == "commit:after_ledger":
                cnt["c"] += 1
            if stage == "stage:linked" and cnt["c"] == 2:
                raise _SimulatedKill(stage)
        killed = None
        try:
            _stub_run(cfg, data, hook).start()
        except _SimulatedKill as exc:
            killed = str(exc)
        cx.ok(killed == "stage:linked", "simulated kill at stage:linked "
              "after the epoch-3 row file", "NO_KILL", repr(killed))
        sti =C.ArtifactStore(ia / "ckpt", C.Ledger(ia / "ledger.jsonl"),
                              ia / "_incomplete")
        cx.ok((ia / "ckpt" / "val_e003.json").exists() and
              "val_rows_e003" not in sti.committed(),
              "killed after the epoch-3 row file was published, before its "
              "commit")
        _stub_run(cfg, data).resume()
        q = list((ia / "_incomplete").rglob("val_e003.json"))
        ca, ci = _commits(ua), _commits(ia)
        cx.ok(bool(q) and ca == ci, "resume quarantines the uncommitted row "
              "file and retrains epoch 3; every commit incl. row-file hashes "
              "identical to the uninterrupted run", "RUNS_DIFFER",
              _first_diff(ca, ci))
        # tamper with a committed row file
        f = ua / "ckpt" / "val_e002.json"
        raw = f.read_bytes()
        tam = tmp / "tampered.json"
        tam.write_bytes(raw.replace(b'"epoch":2', b'"epoch":9', 1))
        cx.raises("VAL_ROWS_HASH_MISMATCH", lambda: T.load_val_rows(
            tam, cm[2]["validation_rows"]["sha256"], 2),
            "altered row file refused on load")
        cx.raises("VAL_ROWS_INVALID", lambda: T.load_val_rows(
            tam, C.sha256_file(tam), 2), "row file for another epoch refused")
        # malformed documents (each with its own matching hash) -> typed
        good = json.loads(raw)
        bad_docs = {
            "missing slice_ids_sha256": {k: v for k, v in good.items()
                                         if k != "slice_ids_sha256"},
            "extra field": dict(good, extra=1),
            "rows not a list": dict(good, rows={"a": 1}),
            "n_rows bool": dict(good, n_rows=True),
            "epoch as string": dict(good, epoch="2"),
            "row missing volume_id": dict(good, rows=[
                {k: v for k, v in good["rows"][0].items()
                 if k != "volume_id"}] + good["rows"][1:]),
            "row not an object": dict(good, rows=[1] + good["rows"][1:]),
            "top level list": [good],
        }
        for lab, d in bad_docs.items():
            mf = tmp / "malformed.json"
            mf.write_bytes(C.canonical_json(d))
            _raises_typed(cx, "VAL_ROWS_INVALID", lambda: T.load_val_rows(
                mf, C.sha256_file(mf), 2), f"malformed row file ({lab}) -> "
                "VAL_ROWS_INVALID")
        mf.write_bytes(b"{not json")
        _raises_typed(cx, "VAL_ROWS_INVALID", lambda: T.load_val_rows(
            mf, C.sha256_file(mf), 2), "non-JSON row file -> VAL_ROWS_INVALID")
        os.chmod(f, 0o644)
        f.write_bytes(raw[:-2] + b" }")
        cx.raises("STORE_HASH_MISMATCH", lambda: st.recover(),
                  "recovery refuses a tampered committed row file")
        # failure while staging the row file -> terminal, never resumed
        wf = tmp / "wf"
        cfg = _stub_cfg(wf, max_epochs=3)
        orig = C.ArtifactStore.stage

        def failing(self, name, writer, hook=None):
            if name.startswith("val_e"):
                raise OSError("fixture: disk full")
            return orig(self, name, writer, hook)
        try:
            with _patched(C.ArtifactStore, "stage", failing):
                _stub_run(cfg, data).start()
            got = None
        except C.V05LTError as exc:
            got = (exc.code, exc.taxonomy)
        e = _ends(wf)
        cx.ok(got == ("CHECKPOINT_WRITE_FAILED", "failed_validity") and
              len(e) == 1 and e[0]["status"] == "failed_validity" and
              "session_error" not in _kinds(wf),
              "row-file staging failure -> CHECKPOINT_WRITE_FAILED, terminal "
              "run_end, no session_error", detail=f"{got} {e}")
        _expect(cx, "RESUME_RUN_ENDED", lambda: _stub_run(cfg, data).resume(),
                "row-file write failure: never resumed")


def nld(cx: Ctx) -> None:
    """G3: NLL prior / log-determinant components; production NLL kept;
    defined consistency check."""
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.scripts.v05lt_stub_train import build_stub
    with tempfile.TemporaryDirectory() as t:
        data = _syn_train_data(Path(t) / "syn")
        m = build_stub(2026)
        with __import__("torch").no_grad():
            m.s.fill_(0.25)                       # non-zero log-determinant
        rows = M.validate(m, data["vv"], data["eligibility"], data["p4"],
                          data["s_ref"])
        d = 13824
        cx.ok(all(abs(r["nll_prior_per_dim"] + r["nll_logdet_per_dim"]
                      - r["nll_per_dim"]) <= 1e-6 * max(1.0, abs(
                          r["nll_per_dim"])) for r in rows)
              and all(r["nll_logdet_per_dim"] == -(-d * 0.25) / d
                      for r in rows),
              "sign convention: nll = nll_prior + nll_logdet (per dim), "
              "nll_logdet = -ldj / d_free (stub ldj = -d*s)")
        cx.ok(all(r["nll_decomp_abs_dev"] <= M.NLL_DECOMP_TOL_REL * max(
            1.0, abs(r["nll_per_dim"] * d)) for r in rows),
            "recorded |recomposed - production| within the defined "
            "tolerance", detail=str(max(r["nll_decomp_abs_dev"]
                                        for r in rows)))
        real = M.nll_components

        def shifted(eps):
            def f(model, u, cond, mask):
                p, l = real(model, u, cond, mask)
                return p + eps, l
            return f
        with _patched(M, "nll_components", shifted(1e-9)):
            rows2 = M.validate(m, data["vv"], data["eligibility"],
                               data["p4"], data["s_ref"])
        cx.ok([r["nll_per_dim"] for r in rows2] ==
              [r["nll_per_dim"] for r in rows],
              "production nll_per_dim unchanged when the components change "
              "(never replaced by the recomposed total)")
        with _patched(M, "nll_components", shifted(1.0)):
            cx.raises("VAL_NLL_DECOMPOSITION_MISMATCH", lambda: M.validate(
                m, data["vv"], data["eligibility"], data["p4"],
                data["s_ref"]), "components inconsistent with the production "
                "log-density -> VAL_NLL_DECOMPOSITION_MISMATCH")
    if not cx.host:
        return
    cx.phase = "host"
    with tempfile.TemporaryDirectory() as t:
        res = _run_eval_child(cx, Path(t), {"kind": "nld"}, "nld")
    r = res["nld"]
    cx.ok(r["n"] == 4 and r["max_dev"] <= r["tol_bound"] and
          r["max_sum_err"] <= 1e-6 * r["max_abs_nll"],
          "host (fresh seed-2026 build, 4 validation slices): components "
          "recorded, consistency check holds", detail=str(r))
    cx.record("NLD host (real model; informational)", r)


def _nld_host(data_root: Path) -> dict:
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.src import free_flow_runtime as ffr
    with tempfile.TemporaryDirectory() as t:
        data = _rt_data(True, REPO_ROOT, str(data_root), Path(t))
        model = ffr.build_model(init_seed=2026)
        rows = M.validate(model, data["vv"], data["eligibility"], data["p4"],
                          data["s_ref"], [0, 1, 2, 3])
    d = ffr.FLOW_DIM_REAL
    return {"n": len(rows),
            "max_dev": max(r["nll_decomp_abs_dev"] for r in rows),
            "tol_bound": min(M.NLL_DECOMP_TOL_REL * max(
                1.0, abs(r["nll_per_dim"]) * d) for r in rows),
            "max_sum_err": max(abs(r["nll_prior_per_dim"] +
                                   r["nll_logdet_per_dim"] - r["nll_per_dim"])
                               for r in rows),
            "max_abs_nll": max(abs(r["nll_per_dim"]) for r in rows),
            "example": {k: rows[0][k] for k in (
                "nll_per_dim", "nll_prior_per_dim", "nll_logdet_per_dim",
                "nll_decomp_abs_dev")}}


def _pev_banks() -> dict:
    """Registered recipes (194 §3.1): Z_DIAG PCG64(0), B PCG64(20261023),
    C PCG64(20261024); standard_normal float64 -> float32."""
    return {n: np.random.Generator(np.random.PCG64(sd)).standard_normal(
        (128, 13824)).astype(np.float32)
        for n, sd in (("primary", 0), ("B", 20261023), ("C", 20261024))}


def _pev_task(EV, ms: str, ms0: str, ms_final: str) -> dict:
    plan = EV.plan_campaign([{"seed": 2026, "arm": "lam2", "roles": {
        "epoch0": {"epoch": 0, "model_state": ms0},
        "epoch3": {"epoch": 3, "model_state": ms},
        "selected": {"epoch": 3, "model_state": ms},
        "final": {"epoch": 5, "model_state": ms_final}}}])
    return [t for t in plan["tasks"] if t["model_state"] == ms][0]


def _pev_meta(ms: str, *, seed: int = 2026, arm: str = "lam2",
              epoch: int = 3) -> dict:
    """Trainer weights meta (v05lt_train._commit_epoch layout)."""
    return {"seed": seed, "arm": arm, "lambda": 2.0, "epoch": epoch,
            "step": 30 * epoch, "model_state_sha256": ms,
            "roles": [f"milestone_{epoch:03d}", "best_nll"]}


def _pev_copy(src: Path, dst: Path, edit) -> Path:
    """Copy a published posteval directory, apply edit(dir), rewrite
    SHA256SUMS over the files present (so only the edited property is
    wrong)."""
    import shutil
    shutil.copytree(src, dst)
    for f in dst.iterdir():
        os.chmod(f, 0o644)
    edit(dst)
    (dst / "SHA256SUMS").write_text("".join(
        f"{C.sha256_file(f)}  {f.name}\n" for f in sorted(dst.iterdir())
        if f.name != "SHA256SUMS"))
    return dst


def pev(cx: Ctx) -> None:
    """G2: post-run evaluation of a trained checkpoint -- provenance,
    deduplication, refusals, no-overwrite publication."""
    import torch
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_train as T
    from seqref_mri.scripts.v05lt_stub_train import build_stub
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data, mids, cases, tol, _ = _syn_e0_inputs(td / "d")
        banks = _pev_banks()
        cx.ok(all(EV._array_sha(banks[n]) == EV.REGISTERED_BANK_ARRAY[n]
                  for n in banks), "bank recipes reproduce the registered "
              "array sha256 (Z_DIAG, B, C)")
        src = build_stub(2026)
        with torch.no_grad():
            src.s.fill_(0.1)
        ms = C.model_state_sha256(src)
        cx.ok(EV.WEIGHTS_SCHEMA == T.WEIGHTS_SCHEMA,
              "evaluator weights schema = trainer weights schema")

        def wfile(name, meta, obj=None):
            f = td / name
            with open(f, "wb") as fh:
                if obj is None:
                    T._weights_writer(src, meta)(fh)
                else:
                    torch.save(obj, fh)
            return f, C.sha256_file(f)

        def load(f, fs, msx=ms, seed=2026, arm="lam2", epoch=3):
            return EV.load_trained_checkpoint(f, fs, msx, seed=seed, arm=arm,
                                              epoch=epoch,
                                              model_factory=build_stub)
        wf, fsha = wfile("weights_e003.pt", _pev_meta(ms))
        model, ck = load(wf, fsha)
        cx.ok(C.model_state_sha256(model) == ms and not model.training and
              isinstance(ck, EV.VerifiedCheckpoint) and
              (ck.seed, ck.arm, ck.epoch, ck.model_state_sha256,
               ck.file_sha256) == (2026, "lam2", 3, ms, fsha),
              "checkpoint loaded: file hash, schema, saved meta and "
              "model-state verified; identity read from the file")
        cx.raises("POSTEVAL_FILE_MISMATCH", lambda: load(wf, "0" * 64),
                  "file sha256 mismatch refused")
        cx.raises("POSTEVAL_META_MISMATCH", lambda: load(wf, fsha, "1" * 64),
                  "committed model-state != saved meta refused")
        cx.raises("POSTEVAL_META_MISMATCH", lambda: load(wf, fsha, epoch=4),
                  "committed epoch != saved epoch refused")
        cx.raises("POSTEVAL_META_MISMATCH", lambda: load(wf, fsha, seed=2027),
                  "committed seed != saved seed refused")
        cx.raises("POSTEVAL_META_MISMATCH", lambda: load(wf, fsha, arm="lam0"),
                  "committed arm != saved arm refused")
        f2, s2 = wfile("w_ms.pt", _pev_meta("c" * 64))
        cx.raises("POSTEVAL_META_MISMATCH", lambda: load(f2, s2),
                  "saved meta model-state != committed refused")
        f3, s3 = wfile("w_ms2.pt", _pev_meta("c" * 64))
        cx.raises("POSTEVAL_STATE_MISMATCH", lambda: load(f3, s3, "c" * 64),
                  "weights whose state != meta and committed hash refused")
        sd = {k: v.detach() for k, v in src.state_dict().items()}
        meta0 = _pev_meta(ms)
        for i, (lab, obj) in enumerate((
                ("wrong schema", {"schema": "seqref-v05lt-resume/1",
                                  "model": sd, "meta": meta0}),
                ("extra key", {"schema": T.WEIGHTS_SCHEMA, "model": sd,
                               "meta": meta0, "x": 1}),
                ("no meta", {"schema": T.WEIGHTS_SCHEMA, "model": sd}),
                ("meta field missing", {
                    "schema": T.WEIGHTS_SCHEMA, "model": sd,
                    "meta": {k: v for k, v in meta0.items() if k != "epoch"}}),
                ("meta epoch as string", {
                    "schema": T.WEIGHTS_SCHEMA, "model": sd,
                    "meta": dict(meta0, epoch="3")}))):
            fb, sb = wfile(f"w_bad{i}.pt", None, obj)
            _raises_typed(cx, "POSTEVAL_FILE_INVALID",
                          lambda fb=fb, sb=sb: load(fb, sb),
                          f"weights file {lab} refused")
        fj = td / "w_junk.pt"
        fj.write_bytes(b"not a torch file")
        _raises_typed(cx, "POSTEVAL_FILE_INVALID",
                      lambda: load(fj, C.sha256_file(fj)),
                      "unreadable weights file refused")
        task = _pev_task(EV, ms, "a" * 64, "b" * 64)
        cx.ok(sorted(u["role"] for u in task["uses"]) == ["epoch3",
                                                          "selected"]
              and task["banks"] == ["primary", "B", "C"],
              "deduplication: epoch 3 = selected -> one task with both roles; "
              "banks B/C and prefixes at the selected checkpoint")
        torch.manual_seed(7)
        np.random.seed(7)
        r0 = (torch.get_rng_state().clone(), np.random.get_state()[1].copy())
        cnt = {}
        rec, pan = EV.posteval_evaluate(model, data, task, banks, tol, cases,
                                        checkpoint=ck,
                                        eligibility_sha256="synthetic",
                                        decode_counter=cnt)
        n = len(mids)
        cx.ok(cnt == {"primary": 128 * n, "B": 128 * n, "C": 128 * n} and
              rec["midpoint"]["decodes"] == cnt,
              "128 decodes per bank per midpoint; one evaluation for both "
              "roles", detail=str(cnt))
        cx.ok(torch.equal(r0[0], torch.get_rng_state()) and
              np.array_equal(r0[1], np.random.get_state()[1]) and
              C.model_state_sha256(model) == ms,
              "RNG states restored; parameters unchanged")
        prov = (rec["checkpoint"] == ck.as_record() and rec["banks"] ==
                {k: EV.REGISTERED_BANK_ARRAY[k] for k in banks}
                and rec["masks"]["convention"] == EV.MASK_CONVENTION and
                len(rec["masks"]["sha256"]) == 64 and
                rec["population"]["n"] == n and
                rec["population"]["eligibility_manifest_sha256"] ==
                "synthetic" and set(rec["code"]["files"]) ==
                {"v05lt_eval.py", "v05lt_metrics.py", "v05lt_common.py"} and
                {"torch", "numpy", "python", "threads", "deterministic"} <=
                set(rec["backend"]) and rec["prefixes"] == list(EV.PREFIXES)
                and [u["role"] for u in rec["uses"]] == ["epoch3", "selected"])
        cx.ok(prov, "record binds checkpoint (model-state, file hash, epoch), "
              "roles, banks (= registered), masks, slice population, code "
              "and backend")
        cx.ok(all({"r0", "r1", "r0_dc", "r1_dc"} <= set(r) and
                  set(r["r1"]) == {"primary", "B", "C"} and
                  set(r["r1"]["primary"]) == {str(k) for k in EV.PREFIXES}
                  and "dc_identity" in r["r0_dc"]
                  for r in rec["midpoint"]["rows"]),
              "rows: R0, R1 (all banks; prefixes on the primary), R0+DC, "
              "R1+DC with per-slice DC identity flags")
        other = build_stub(2027)
        cx.raises("POSTEVAL_STATE_MISMATCH", lambda: EV.posteval_evaluate(
            other, data, task, banks, tol, cases, checkpoint=ck,
            eligibility_sha256="s"), "model other than the task's refused")
        cx.raises("POSTEVAL_BANK_MISMATCH", lambda: EV.posteval_evaluate(
            model, data, task, {"primary": banks["primary"]}, tol, cases,
            checkpoint=ck, eligibility_sha256="s"),
            "bank set different from the task refused")
        bb = dict(banks, B=banks["B"].copy())
        bb["B"][0, 0] += 1.0
        cx.raises("POSTEVAL_BANK_MISMATCH", lambda: EV.posteval_evaluate(
            model, data, task, bb, tol, cases, checkpoint=ck,
            eligibility_sha256="s"), "altered bank refused")
        cx.raises("POSTEVAL_CHECKPOINT_UNVERIFIED",
                  lambda: EV.posteval_evaluate(
                      model, data, task, banks, tol, cases,
                      checkpoint=ck.as_record(), eligibility_sha256="s"),
                  "caller-built checkpoint dictionary refused")
        f4, s4 = wfile("weights_e004.pt", _pev_meta(ms, epoch=4))
        m4, ck4 = load(f4, s4, epoch=4)
        cx.raises("POSTEVAL_EPOCH_MISMATCH", lambda: EV.posteval_evaluate(
            m4, data, task, banks, tol, cases, checkpoint=ck4,
            eligibility_sha256="s"),
            "verified checkpoint of another epoch than the task refused")
        for lab, kw in (("arm", {"arm": "lam0"}), ("seed", {"seed": 2027})):
            fr, sr = wfile(f"w_run_{lab}.pt", _pev_meta(ms, **kw))
            mr, ckr = load(fr, sr, **kw)
            cx.raises("POSTEVAL_RUN_MISMATCH",
                      lambda mr=mr, ckr=ckr: EV.posteval_evaluate(
                          mr, data, task, banks, tol, cases, checkpoint=ckr,
                          eligibility_sha256="s"),
                      f"verified checkpoint of another run ({lab}) than the "
                      "task's uses refused")
        out = td / "pe" / ms
        shas = EV.write_posteval(out, rec, pan, {"fixture": True})
        doc = EV.load_posteval(out)
        cx.ok(len(shas) == 1 + 2 * len(cases) and doc["checkpoint"] ==
              ck.as_record() and not (td / "pe" / f".{ms}.staging").exists(),
              "write-once directory published atomically; verified on load")
        # panels must equal the declared cases exactly
        for i, (lab, rr, pp) in enumerate((
                ("declared case, no panels", dict(rec, cases=[0]), []),
                ("missing panel", rec, pan[:-1]),
                ("unexpected panel", dict(rec, cases=rec["cases"][:-1]), pan),
                ("panel for an undeclared case", rec,
                 pan[:-1] + [dict(pan[-1], case=dict(pan[-1]["case"],
                                                     case=99))]))):
            dd = td / f"pm{i}" / ms
            cx.raises("POSTEVAL_PANELS_MISMATCH",
                      lambda dd=dd, rr=rr, pp=pp: EV.write_posteval(
                          dd, rr, pp, {}), f"write: {lab} refused")
            cx.ok(not dd.exists() and not dd.with_name(
                f".{ms}.staging").exists(), f"write: {lab}: nothing written")
        # loader: structure and panel files verified (typed refusals only)
        pfile = f"panel_case{rec['cases'][0]}.png"

        def jedit(fn):
            def e(d):
                j = json.loads((d / EV.POSTEVAL_FILE).read_bytes())
                (d / EV.POSTEVAL_FILE).write_text(json.dumps(fn(j)))
            return e
        lv = [("panel file missing", "POSTEVAL_INCOMPLETE",
               lambda d: (d / pfile).unlink()),
              ("unexpected extra panel file", "POSTEVAL_INCOMPLETE",
               lambda d: (d / "panel_case99.png").write_bytes(b"x")),
              ("declared case without panel files", "POSTEVAL_INCOMPLETE",
               jedit(lambda j: dict(j, cases=j["cases"] + [99],
                                    panels=j["panels"] + [{}]))),
              ("panel_sha256 != SHA256SUMS", "POSTEVAL_FILE_INVALID",
               jedit(lambda j: dict(j, panel_sha256=dict(
                   j["panel_sha256"], **{pfile: "0" * 64})))),
              ("record not JSON", "POSTEVAL_FILE_INVALID",
               lambda d: (d / EV.POSTEVAL_FILE).write_bytes(b"{x")),
              ("record not an object", "POSTEVAL_FILE_INVALID",
               lambda d: (d / EV.POSTEVAL_FILE).write_text("[1]")),
              ("checkpoint missing", "POSTEVAL_FILE_INVALID",
               jedit(lambda j: {k: v for k, v in j.items()
                                if k != "checkpoint"})),
              ("cases empty", "POSTEVAL_FILE_INVALID",
               jedit(lambda j: dict(j, cases=[]))),
              ("cases not integers", "POSTEVAL_FILE_INVALID",
               jedit(lambda j: dict(j, cases=[str(c) for c in j["cases"]]))),
              ("record file missing", "POSTEVAL_INCOMPLETE",
               lambda d: (d / EV.POSTEVAL_FILE).unlink())]
        for i, (lab, code, ed) in enumerate(lv):
            bad = _pev_copy(out, td / f"lv{i}" / ms, ed)
            _raises_typed(cx, code, lambda bad=bad: EV.load_posteval(bad),
                          f"load: {lab} refused")
        bad = _pev_copy(out, td / "lvs" / ms, lambda d: None)
        (bad / "SHA256SUMS").write_text("no-separator-line\n")
        _raises_typed(cx, "POSTEVAL_FILE_INVALID", lambda: EV.load_posteval(bad),
                      "load: malformed SHA256SUMS refused")
        cx.raises("ARTIFACT_EXISTS", lambda: EV.write_posteval(
            out, rec, pan, {}), "second write refused")
        cx.raises("POSTEVAL_DIR_NAME", lambda: EV.write_posteval(
            td / "pe" / "x", rec, pan, {}),
            "directory name other than the model-state hash refused")
        reg = td / "results" / "v05lt" / "posteval" / ms
        cx.raises("POSTEVAL_GATE_REQUIRED", lambda: EV.write_posteval(
            reg, rec, pan, {}), "registered path refused without the gate")
        ln = td / "pe2" / ms
        ln.parent.mkdir()
        os.symlink(td / "nowhere", ln)
        cx.raises("ARTIFACT_EXISTS", lambda: EV.write_posteval(
            ln, rec, pan, {}), "symlink at the destination refused")
        stg = td / "pe3" / f".{ms}.staging"
        stg.mkdir(parents=True)
        cx.raises("STAGING_PRESENT", lambda: EV.write_posteval(
            td / "pe3" / ms, rec, pan, {}), "left-over staging refused")
        pf = out / EV.POSTEVAL_FILE
        os.chmod(pf, 0o644)
        pf.write_bytes(pf.read_bytes().replace(b'"epoch": 3', b'"epoch": 4',
                                               1))
        cx.raises("POSTEVAL_INCOMPLETE", lambda: EV.load_posteval(out),
                  "tampered published record refused on load")
    if not cx.host:
        return
    cx.phase = "host"
    with tempfile.TemporaryDirectory() as t:
        res = _run_eval_child(cx, Path(t), {"kind": "pev"}, "pev")
    e = res["pev"]
    cx.ok(e["decodes"] == {"primary": 128 * e["n_mid"],
                           "B": 128 * e["n_mid"], "C": 128 * e["n_mid"]}
          and e["state_unchanged"] and e["rng_restored"] and e["banks_ok"]
          and e["loaded"] and e["written"] and e["registered_refused"],
          "host: real-architecture weights file (fresh seed-2026 build) "
          "loaded with hashes verified; registered banks Z_DIAG/B/C; 2 "
          "registered midpoint cases; outputs to a temporary directory only",
          "PEV_HOST_FAILED", str(e))
    cx.record("PEV host (fixture subset; P4 input only)",
              {k: e[k] for k in ("timing", "n_mid", "backend")})


def _pev_host(EV, data_root: Path) -> dict:
    import torch
    from seqref_mri.scripts import v05lt_eligibility as E
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.scripts import v05lt_train as T
    from seqref_mri.src import free_flow_runtime as ffr
    from seqref_mri.tdiag import estimators as est
    p = C.Paths(repo=REPO_ROOT)
    sp, cp = p.inputs / "v05lt_split_manifest.json", \
        p.inputs / "v05lt_case_manifest.json"
    sdoc = C.load_split_manifest(sp, C.REGISTERED_INPUTS[sp.name])
    C.verify_file(cp, C.REGISTERED_INPUTS[cp.name], "CASE")
    case_doc = json.loads(cp.read_bytes())
    _, vv = C.build_split_views(data_root, sdoc,
                                C.holdout_ids(p.v02_manifests))
    s_ref = C.verify_p0s(p.p0s)
    p4 = ffr.load_p4s2_parent(str(p.repo / P4_REL))
    cases = case_doc["cases"][:2]
    pos = {vv.entry(i): i for i in range(len(vv))}
    mid = [pos[(c["file"], c["slice_index"])] for c in cases]
    recs = E.compute_records(vv, s_ref, mid)
    elig = M.eligibility_index(E.build_manifest(
        recs, {"validation": sdoc["validation"]}, case_doc, {"p": "fixture"}))
    data = {"vv": vv, "p4": p4, "s_ref": s_ref, "eligibility": elig}
    C.verify_extra_banks(p.inputs)
    banks = {"primary": est.z_diag_bank()["bank"],
             "B": np.load(p.inputs / "v05lt_bank_20261023.npy"),
             "C": np.load(p.inputs / "v05lt_bank_20261024.npy")}
    banks_ok = all(EV._array_sha(b) == EV.REGISTERED_BANK_ARRAY[n]
                   for n, b in banks.items())
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        src = ffr.build_model(init_seed=2026)
        ms = C.model_state_sha256(src)
        wf = tmp / "weights_e003.pt"
        with open(wf, "wb") as fh:
            T._weights_writer(src, _pev_meta(ms))(fh)
        del src
        model, ck = EV.load_trained_checkpoint(
            wf, C.sha256_file(wf), ms, seed=2026, arm="lam2", epoch=3)
        loaded = (C.model_state_sha256(model) == ms and
                  (ck.seed, ck.arm, ck.epoch) == (2026, "lam2", 3))
        sts = EV.build_states(vv, mid, p4, s_ref)
        tol = EV.calibrate_dc(sts)["tolerances"]
        task = _pev_task(EV, ms, "a" * 64, "b" * 64)
        r0 = torch.get_rng_state().clone()
        cnt = {}
        rec, pan = EV.posteval_evaluate(model, data, task, banks, tol, cases,
                                        checkpoint=ck,
                                        eligibility_sha256="fixture",
                                        mid_indices=mid, decode_counter=cnt)
        same = C.model_state_sha256(model) == ms
        shas = EV.write_posteval(tmp / "out" / ms, rec, pan, {"fixture": True})
        reloaded = EV.load_posteval(tmp / "out" / ms)["checkpoint"] == \
            ck.as_record()
        try:
            EV.write_posteval(tmp / "results" / "v05lt" / "posteval" / ms,
                              rec, pan, {})
            refused = False
        except C.V05LTError as exc:
            logger.warning("expected refusal: %s", exc.code)
            refused = exc.code == "POSTEVAL_GATE_REQUIRED"
    return {"decodes": cnt, "n_mid": len(mid), "state_unchanged": same,
            "rng_restored": torch.equal(r0, torch.get_rng_state()),
            "banks_ok": banks_ok, "loaded": loaded,
            "written": len(shas) == 1 + 2 * len(cases) and reloaded,
            "registered_refused": refused, "timing": rec["timing"],
            "backend": rec["backend"]}


# ---------------------------------------------------------------------------
# I1.8(b) b1: verified loader (LOAD) and analysis stage (ANA) -- synthetic
# campaign built with the REAL writers (stub trainer runs, post-run and
# epoch-0 evaluation records, ZF record, manifests); synthetic by design.
# ---------------------------------------------------------------------------
def _syn_campaign_inputs(tmp: Path) -> dict:
    """Synthetic split + eligibility manifest FILES (with sidecar) and the
    trainer / evaluator data dict (same slices)."""
    from seqref_mri.scripts import v05lt_eligibility as E
    from seqref_mri.scripts import v05lt_metrics as M
    root, mp, msha, ex = synth_split(tmp, SYN_VOLS, SYN_TRAIN, SYN_VAL)
    sdoc = C.load_split_manifest(mp, msha, ex)
    tv, vv = C.build_split_views(root, sdoc, SYN_HOLDOUT, ex)
    recs = E.compute_records(vv, 1.0)
    edoc = E.build_manifest(recs, {"validation": sdoc["validation"]},
                            {"cases": []}, {"p": "fixture"})
    ep = tmp / "inputs" / "v05lt_nmse_eligibility.json"
    ep.parent.mkdir(parents=True)
    esha = C.write_durable_new(ep, C.canonical_json(edoc))
    C.write_durable_new(Path(str(ep) + ".sha256"),
                        f"{esha}  {ep.name}\n".encode())
    data = {"tv": tv, "vv": vv, "p4": _unit_p4(), "s_ref": 1.0,
            "eligibility": M.eligibility_index(edoc)}
    return {"data": data, "split": mp, "split_sha256": msha, "expect": ex,
            "eligibility": ep, "eligibility_sha256": esha}


def _syn_campaign(tmp: Path, max_epochs: int = 4, factory=None) -> dict:
    """Six stub runs, post-run records (epoch 3 / selected / final), three
    epoch-0 records and the ZF record -> fixture layout. c2 (PA5-B): each
    completed run's final weights are published with the REAL
    v05lt_final.extract_final, and a final checkpoint that is neither a
    milestone nor a best epoch is post-evaluated from that file (the
    earlier fixture-only weights file written from the resume state is
    removed). factory (v0.39, fixture-only): stub variant for training and
    epoch 0 (default build_stub; _stub_w1000 gives coincident checkpoints)."""
    import dataclasses
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_final as FW
    from seqref_mri.scripts import v05lt_train as T
    from seqref_mri.scripts.v05lt_stub_train import build_stub
    fac = factory or build_stub
    inp = _syn_campaign_inputs(tmp / "in")
    data = inp["data"]
    vv, el = data["vv"], data["eligibility"]
    mids = [i for i in range(len(vv)) if el[vv.entry(i)]["midpoint"]]
    cases = [{"case": j, "file": vv.entry(i)[0], "slice_index": vv.entry(i)[1]}
             for j, i in enumerate(mids)]
    tol = EV.calibrate_dc(EV.build_states(vv, mids, data["p4"],
                                          data["s_ref"]))["tolerances"]
    banks = _pev_banks()
    runs, plan_in = {}, []
    for s in (2026, 2027, 2028):
        for a in ("lam0", "lam2"):
            d = tmp / "camp" / f"{s}_{a}"
            cfg = _stub_cfg(d, seed=s, arm=a, max_epochs=max_epochs)
            if factory is None:
                _stub_run(cfg, data).start()
            else:
                cfg = dataclasses.replace(cfg, epoch0_model_state=(
                    C.model_state_sha256(fac(s))))
                T.Run(cfg, C.Paths(), data, model_factory=fac).start()
            fw = FW.extract_final(d, fixture=True)
            cm = {r["body"]["epoch"]: r["body"] for r in
                  C.Ledger(d / "ledger.jsonl").records()
                  if r["kind"] == "commit"}
            last = cm[max(cm)]
            sel = last["selection"]
            roles = {"epoch0": {"epoch": 0,
                                "model_state": cfg.epoch0_model_state},
                     "epoch3": {"epoch": 3,
                                "model_state": cm[3]["model_state_sha256"]},
                     "selected": {"epoch": sel["best_psnr"]["epoch"],
                                  "model_state": cm[sel["best_psnr"][
                                      "epoch"]]["model_state_sha256"]},
                     "final": {"epoch": max(cm), "model_state":
                               last["model_state_sha256"]}}
            runs[(a, s)] = {"dir": d, "cfg": cfg, "commits": cm, "final": fw}
            plan_in.append({"seed": s, "arm": a, "roles": roles})
    plan = EV.plan_campaign(plan_in, n_slices=len(mids))
    root = tmp / "camp" / "posteval"
    records = {}
    for task in plan["tasks"]:
        if task["epoch"] == 0:
            continue
        u = [x for x in task["uses"] if x["role"] != "epoch0"][0]
        run = runs[(u["arm"], u["seed"])]
        ret = run["commits"][max(run["commits"])]["retained"]
        wname = f"weights_e{task['epoch']:03d}.pt"
        hit = [v for v in ret.values() if v["name"] == wname]
        if hit:
            wf, wsha = run["dir"] / "ckpt" / wname, hit[0]["sha256"]
        else:                     # PA5-B: published final weights
            fw = run["final"]
            C.require(fw["epoch"] == task["epoch"] and
                      fw["model_state_sha256"] == task["model_state"],
                      "FIXTURE_FINAL_MISMATCH", f"{u}: final weights")
            wf, wsha = fw["path"], fw["sha256"]
        model, ck = EV.load_trained_checkpoint(
            wf, wsha, task["model_state"], seed=u["seed"], arm=u["arm"],
            epoch=task["epoch"], model_factory=fac)
        rec, pan = EV.posteval_evaluate(
            model, data, task, {b: banks[b] for b in task["banks"]}, tol,
            cases, checkpoint=ck, eligibility_sha256=inp["eligibility_sha256"])
        EV.write_posteval(root / task["model_state"], rec, pan,
                          {"fixture": True})
        records[task["model_state"]] = (rec, pan)
    e0 = {}
    for s in (2026, 2027, 2028):
        model = fac(s)
        rec, pan = EV.epoch0_evaluate(model, data, banks["primary"], tol, cases)
        d = tmp / "camp" / "epoch0" / f"seed_{s}" / "eval"
        ctx = {"seed": s, "epoch0_model_state_sha256":
               C.model_state_sha256(model), "fixture": True}
        EV.write_epoch0_eval(d, rec, pan, ctx)
        e0[s] = (rec, pan, ctx)
    sts = EV.build_states(vv, list(range(len(vv))), data["p4"], data["s_ref"])
    zrows = [EV.zf_row(st, el[(st["identity"]["file"],
                               int(st["identity"]["slice_index"]))])
             for st in sts]
    zd = tmp / "camp" / "zf"
    EV.write_zf(zd, zrows, EV.zf_summary(zrows), {"fixture": True})
    zsha = EV.verify_zf(zd)
    cmp_ = tmp / "in" / "inputs" / "v05lt_case_manifest.json"
    csha = C.write_durable_new(cmp_, C.canonical_json(
        {"schema": "fixture-case-manifest", "cases": cases}))
    layout = {"split_manifest": inp["split"],
              "split_sha256": inp["split_sha256"],
              "case_manifest": cmp_, "case_sha256": csha,
              "split_expect": inp["expect"],
              "eligibility": inp["eligibility"],
              "eligibility_sha256": inp["eligibility_sha256"],
              "zf_dir": zd, "zf_sha256": zsha,
              "difficult": tmp / "camp" / "analysis" / "difficult.json",
              "difficult_sha256": None,
              "epoch0": {s: tmp / "camp" / "epoch0" / f"seed_{s}" / "eval"
                         for s in (2026, 2027, 2028)},
              "runs": {f"{a}|{s}": tmp / "camp" / f"{s}_{a}"
                       for s in (2026, 2027, 2028) for a in ("lam0", "lam2")},
              "posteval_root": root, "seeds": [2026, 2027, 2028],
              "arms": ["lam0", "lam2"], "fixture": True}
    return {"layout": layout, "records": records, "epoch0": e0,
            "runs": runs, "data": data, "cases": cases, "tol": tol,
            "banks": banks, "plan": plan, "zrows": zrows}


def _campaign_copy(camp: dict, tmp: Path, name: str) -> dict:
    """Copy of the campaign tree with the layout re-pointed."""
    import shutil
    src = Path(camp["layout"]["posteval_root"]).parent.parent
    dst = tmp / name
    shutil.copytree(src, dst, symlinks=True)
    for p in dst.rglob("*"):
        if p.is_file():
            os.chmod(p, 0o644)

    def mv(p):
        p = Path(p)
        return dst / p.relative_to(src)
    lay = dict(camp["layout"])
    for k in ("split_manifest", "eligibility", "zf_dir", "difficult",
              "posteval_root", "case_manifest"):
        lay[k] = mv(lay[k])
    lay["epoch0"] = {s: mv(p) for s, p in lay["epoch0"].items()}
    lay["runs"] = {k: mv(p) for k, p in lay["runs"].items()}
    return lay


def _rewrite_ledger(run_dir: Path, edit) -> None:
    """Rebuild a run ledger with edit(records) applied (valid chain)."""
    lp = run_dir / "ledger.jsonl"
    recs = C.Ledger(lp).records()
    recs = edit([{"kind": r["kind"], "body": r["body"]} for r in recs])
    lp.unlink()
    new = C.Ledger(lp)
    for r in recs:
        new.append(r["kind"], r["body"])


def _repost(lay: dict, ms: str, rec: dict, pan: list) -> None:
    """Replace one post-run record by a re-written (edited) one."""
    import shutil
    from seqref_mri.scripts import v05lt_eval as EV
    d = Path(lay["posteval_root"]) / ms
    shutil.rmtree(d)
    EV.write_posteval(d, rec, pan, {"fixture": True})


def _edit(rec: dict, fn) -> dict:
    import copy
    r = copy.deepcopy(rec)
    fn(r)
    return r


def _typed_call(cx: Ctx, fn):
    """Run fn; a V05LT or raw exception is recorded as a failed check."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 -- recorded as a failed check
        logger.error("unexpected failure: %s: %s", type(exc).__name__, exc)
        cx.ok(False, "call expected to succeed", "UNEXPECTED_FAILURE",
              f"{type(exc).__name__}: {exc}")
        return None


def load_group(cx: Ctx) -> None:
    """LOAD: verified loading of the campaign result records (B1; D5-D8)."""
    import copy
    import shutil
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_results as RS
    from seqref_mri.scripts import v05lt_train as T
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        camp = _syn_campaign(td / "c")
        lay = camp["layout"]
        out = RS.load_campaign(lay)
        nmid = len(out["mid_ids"])
        keys = set(out["sources"])
        need = ({f"{a}|{s}|{c}|r0|all" for a in ("lam0", "lam2")
                 for s in (2026, 2027, 2028)
                 for c in ("epoch3", "selected", "final", "best_nll")} |
                {f"{a}|{s}|{c}|{ro}|mid" for a in ("lam0", "lam2")
                 for s in (2026, 2027, 2028)
                 for c in ("epoch3", "selected", "final")
                 for ro in ("r0", "r1", "r0_dc", "r1_dc")} |
                {f"shared|{s}|epoch0|{ro}|{sc}" for s in (2026, 2027, 2028)
                 for ro, sc in (("r0", "all"), ("r0", "mid"), ("r1", "mid"),
                                ("r0_dc", "mid"), ("r1_dc", "mid"))} |
                {"zf|-|zf|zf|all", "zf|-|zf|zf|mid"})
        cx.ok(need <= keys and all(len(out["sources"][k]) == nmid
                                   for k in keys if k.endswith("|mid")),
              "campaign loaded: every engine source present (trained R0 "
              "all-slice from row files, midpoint readouts from post-run "
              "records, epoch-0, ZF)", detail=str(sorted(need - keys))[:300])
        ri = out["run_info"]
        cx.ok(all(ri[(a, s)] == {"milestones": [3], "last_epoch": 4,
                                 "status": "fixture_budget_end"}
                  for a in ("lam0", "lam2") for s in (2026, 2027, 2028)),
              "run_info from the ledgers: milestone 3 only (exact commit + "
              "artefacts), last epoch, status", detail=str(ri)[:300])
        d6 = out["d6"]["checks"]
        cx.ok(len(d6) == 6 * 3 + 3 and all(
            v["slices"] == nmid and all(x == 0.0 for x in
                                        v["max_deviation"].values())
            for v in d6.values()),
            "D6: duplicated midpoint R0 values identical (row file vs "
            "post-run; epoch-0 all-slice vs midpoint), 21 checks",
            detail=str(d6)[:300])
        cx.ok(len(out["identities"]) > 50 and all(
            len(h) == 64 for h in out["identities"].values()),
            "every file read recorded with its sha256 (source identities)")
        cx.record("LOAD identities", len(out["identities"]))
        # --- negatives (each on its own copy; typed refusals only) ---
        n = [0]

        def fresh():
            n[0] += 1
            return _campaign_copy(camp, td, f"x{n[0]}")

        def neg(code, edit, label):
            lay2 = fresh()
            edit(lay2)
            _raises_typed(cx, code, lambda: RS.load_campaign(lay2), label)
        r26 = "lam2|2026"

        def tamper_rows(l2):
            p = Path(l2["runs"][r26]) / "ckpt" / "val_e002.json"
            b = bytearray(p.read_bytes())
            b[-3] ^= 1
            p.write_bytes(bytes(b))
        neg("LOAD_HASH_MISMATCH", tamper_rows, "altered row file refused")
        neg("LOAD_RUN_DIR_UNEXPECTED", lambda l2: (
            Path(l2["runs"][r26]) / "ckpt" / "stray.pt").write_bytes(b"x"),
            "unexpected file in a run's checkpoint directory refused")
        neg("LOAD_RUN_NOT_ENDED", lambda l2: _rewrite_ledger(
            Path(l2["runs"][r26]), lambda rs: [r for r in rs
                                               if r["kind"] != "run_end"]),
            "run without run_end refused")

        def corrupt(l2):
            p = Path(l2["runs"][r26]) / "ledger.jsonl"
            t = p.read_text()
            p.write_text(t.replace('"epoch":1', '"epoch":9', 1))
        neg("LEDGER_CORRUPT", corrupt, "edited ledger line refused")

        def no_milestone(l2):
            rd = Path(l2["runs"][r26])

            def ed(rs):
                for r in rs:
                    if r["kind"] == "commit" and r["body"]["epoch"] == 3:
                        r["body"]["roles_this_epoch"] = [
                            x for x in r["body"]["roles_this_epoch"]
                            if not x.startswith("milestone_")]
                return rs
            _rewrite_ledger(rd, ed)
        neg("LOAD_MILESTONE_INCOMPLETE", no_milestone, "D7: milestone whose "
            "exact commit lacks its milestone role refused (not inferred "
            "from later epochs)")

        def bad_sel(l2):
            def ed(rs):
                c = [r for r in rs if r["kind"] == "commit"][-1]
                b = c["body"]["selection"]["best_nll"]
                b["value"] = b["value"] + 1.0
                return rs
            _rewrite_ledger(Path(l2["runs"][r26]), ed)
        neg("LOAD_SELECTION_MISMATCH", bad_sel, "D7: committed selection not "
            "re-derivable from the epoch records refused")

        def bad_summary(l2):
            def ed(rs):
                c = [r for r in rs if r["kind"] == "commit"][1]
                c["body"]["validation"]["all_slices"]["psnr"] += 1e-9
                return rs
            _rewrite_ledger(Path(l2["runs"][r26]), ed)
        neg("LOAD_SUMMARY_MISMATCH", bad_summary, "committed validation "
            "summary != summary recomputed from the row file refused")

        def gap(l2):
            def ed(rs):
                return [r for r in rs if not (r["kind"] == "commit" and
                                              r["body"]["epoch"] == 2)]
            _rewrite_ledger(Path(l2["runs"][r26]), ed)
        neg("LOAD_RUN_COMMITS_INVALID", gap, "missing epoch commit refused")
        ms_sel = camp["runs"][("lam2", 2026)]["commits"][
            max(camp["runs"][("lam2", 2026)]["commits"])]["selection"]
        sel_e = ms_sel["best_psnr"]["epoch"]
        ms = camp["runs"][("lam2", 2026)]["commits"][sel_e][
            "model_state_sha256"]
        neg("POSTEVAL_INCOMPLETE", lambda l2: shutil.rmtree(
            Path(l2["posteval_root"]) / ms),
            "D8: missing post-run record of a required checkpoint refused")
        rec, pan = camp["records"][ms]

        def repost(edit_rec):
            def f(l2):
                r2 = copy.deepcopy(rec)
                edit_rec(r2)
                _repost(l2, ms, r2, pan)
            return f
        neg("LOAD_POSTEVAL_BANKS", repost(lambda r: [r["banks"].pop(b)
                                                     for b in ("B", "C")]),
            "selected record without banks B/C refused")
        neg("LOAD_POSTEVAL_PREFIXES", repost(lambda r: r.update(
            prefixes=[128])), "selected record without prefixes refused")
        neg("LOAD_MASK_MISMATCH", repost(lambda r: r["masks"].update(
            sha256="0" * 64)), "D6: masks hash differing between post-run "
            "records refused")
        neg("LOAD_DC_TOLERANCE_MISMATCH", repost(lambda r: r[
            "dc_tolerances"].update(a=r["dc_tolerances"]["a"] * 2)),
            "DC tolerances differing between records refused")
        neg("LOAD_POPULATION_MISMATCH", repost(lambda r: r["population"]
                                               .update(n=99)),
            "post-run population other than the eligibility midpoints "
            "refused")
        neg("LOAD_POSTEVAL_MISMATCH", repost(lambda r: r.update(
            uses=[u for u in r["uses"] if u["role"] != "selected"])),
            "record not listing the (run, role) use refused")
        neg("LOAD_POSTEVAL_MISMATCH", repost(lambda r: r["checkpoint"]
                                             .update(epoch=99)),
            "record epoch other than the committed epoch refused")
        neg("LOAD_DUPLICATE_MISMATCH", repost(lambda r: r["midpoint"]["rows"]
                                              [0]["r0"].update(
            psnr=r["midpoint"]["rows"][0]["r0"]["psnr"] + 0.01)),
            "D6: midpoint R0 PSNR 0.01 dB from the row file refused")
        neg("LOAD_DUPLICATE_MISMATCH", repost(lambda r: r["midpoint"]["rows"]
                                              [0]["r0"].update(
            ssim=r["midpoint"]["rows"][0]["r0"]["ssim"] + 1e-5)),
            "D6: midpoint R0 SSIM 1e-5 from the row file refused")
        neg("LOAD_DUPLICATE_MISMATCH", repost(lambda r: r["midpoint"]["rows"]
                                              [0]["r0"].update(
            psnr_capped=not r["midpoint"]["rows"][0]["r0"]["psnr_capped"])),
            "D6: cap flag differing from the row file refused")
        lay_ok = fresh()
        repost(lambda r: r["midpoint"]["rows"][0]["r0"].update(
            psnr=r["midpoint"]["rows"][0]["r0"]["psnr"] + 0.0005))(lay_ok)
        o2 = RS.load_campaign(lay_ok)
        dv = o2["d6"]["checks"][f"{r26}|selected"]["max_deviation"]
        cx.ok(0.0004 < dv["psnr_abs_db"] < 0.0006,
              "D6: a 0.0005 dB difference is within the registered P2 "
              "tolerance and recorded as the largest deviation",
              detail=str(dv))
        e0rec, e0pan, ctx = camp["epoch0"][2027]

        def e0_rewrite(rec2, pan2, ctx2):
            def f(l2):
                d = Path(l2["epoch0"][2027])
                shutil.rmtree(d)
                EV.write_epoch0_eval(d, rec2, pan2, ctx2)
            return f
        neg("LOAD_EPOCH0_IDENTITY", e0_rewrite(e0rec, e0pan, dict(
            ctx, epoch0_model_state_sha256="1" * 64)),
            "epoch-0 record bound to another model state refused")
        neg("LOAD_EPOCH0_IDENTITY", e0_rewrite(e0rec, e0pan, {"fixture": 1}),
            "epoch-0 record without seed / model-state context refused")
        neg("LOAD_CASES_MISMATCH", e0_rewrite(e0rec, e0pan[:-1], ctx),
            "epoch-0 record missing a declared panel refused (panel list != "
            "case manifest)")

        def e0_drop_file(l2):
            d = Path(l2["epoch0"][2027])
            (d / f"panel_case{e0rec['cases'][-1]}.png").unlink()
            (d / "SHA256SUMS").write_text("".join(
                f"{C.sha256_file(x)}  {x.name}\n" for x in sorted(
                    d.iterdir()) if x.name != "SHA256SUMS"))
        neg("LOAD_EPOCH0_INCOMPLETE", e0_drop_file,
            "epoch-0 panel file of a declared case absent refused")
        neg("LOAD_ZF_HASH", lambda l2: l2.update(zf_sha256="2" * 64),
            "ZF record other than the recorded gate-2a sha256 refused")
        neg("LOAD_ELIGIBILITY_HASH", lambda l2: l2.update(
            eligibility_sha256="3" * 64), "eligibility manifest other than "
            "the recorded sha256 refused")
        neg("LOAD_RUN_IDENTITY", lambda l2: l2.update(fixture=False,
                                                      seeds=[2026, 2027, 2028],
                                                      arms=["lam0", "lam2"]),
            "fixture runs refused by a production layout")
        neg("LOAD_RUN_MISSING", lambda l2: l2["runs"].pop("lam0|2028"),
            "D8: missing run refused")
        # runs that cannot be analysed
        bad = td / "bad"
        cfg = _stub_cfg(bad / "fail", seed=2026, arm="lam2", max_epochs=4)
        try:
            _stub_run(cfg, camp["data"], poison_after=2).start()
        except C.V05LTError as exc:
            logger.warning("expected fixture failure: %s", exc.code)
        cfg2 = _stub_cfg(bad / "short", seed=2026, arm="lam2", max_epochs=2)
        _stub_run(cfg2, camp["data"]).start()
        for nm, code, lab in (("fail", "LOAD_RUN_NOT_COMPLETE",
                               "run ended with a failure status refused "
                               "(owner decision outside the analysis)"),
                              ("short", "LOAD_REQUIRED_MISSING",
                               "D8: run without the epoch-3 commit refused")):
            lay2 = fresh()
            shutil.rmtree(lay2["runs"][r26])
            shutil.copytree(bad / nm, lay2["runs"][r26])
            _raises_typed(cx, code, lambda l2=lay2: RS.load_campaign(l2), lab)
        cx.ok(out["reference_missing"] == [] and all(
            e["summary_verified"] for e in out["epoch0"].values()) and
            out["zf"]["summary_verified"] and
            out["cases"] == camp["cases"],
            "complete references: nothing missing, summaries verified; "
            "cases from the case manifest")
        # --- v0.2: actual R1 measurements, not declarations ---
        neg("LOAD_POSTEVAL_PREFIXES", repost(lambda r: [
            row["r1"].update(primary={"128": row["r1"]["primary"]["128"]})
            for row in r["midpoint"]["rows"][:1]]),
            "selected record declaring 16/32/64/128 whose rows carry only "
            "128 refused")
        neg("LOAD_R1_MISSING", repost(lambda r: r["midpoint"]["rows"][0][
            "r1"].update(B={})), "selected row without bank-B k = 128 "
            "refused (bank still declared)")
        neg("LOAD_R1_MISSING", repost(lambda r: r["midpoint"]["rows"][0][
            "r1"].pop("C")), "selected row without bank C refused (bank "
            "still declared)")
        neg("LOAD_VALUE_MISSING", repost(lambda r: r["midpoint"]["rows"][0][
            "r1"]["primary"]["128"].pop("psnr")),
            "D8: trained post-run value missing refused")
        neg("LOAD_RECORD_INVALID", repost(lambda r: r["midpoint"]["rows"][0][
            "r0"].update(ssim="0.5")), "wrong value type refused")
        # --- v0.2: panel cases bound to the registered case manifest ---
        neg("LOAD_CASES_MISMATCH", lambda l2: _repost(
            l2, ms, dict(copy.deepcopy(rec), cases=[pan[0]["case"]["case"]]),
            pan[:1]), "self-consistent post-run record with fewer cases "
            "than the case manifest refused")
        alt = copy.deepcopy(pan)
        alt[0]["case"] = dict(alt[0]["case"],
                              slice_index=alt[0]["case"]["slice_index"] + 1)
        neg("LOAD_CASES_MISMATCH", lambda l2: _repost(l2, ms, rec, alt),
            "post-run panel case identity other than the case manifest "
            "refused")
        neg("LOAD_CASES_MISMATCH", e0_rewrite(
            dict(e0rec, cases=[e0pan[0]["case"]["case"]]), e0pan[:1], ctx),
            "epoch-0 record with fewer cases than the case manifest refused")
        # --- v0.2: missing epoch-0 / ZF values follow AM3-E4 ---
        from seqref_mri.scripts import v05lt_stats as S
        man_rows = out["slice_rows"]
        cfgx = S.build_config(out["validation_volumes"], man_rows,
                              eligibility_sha256=out["eligibility_sha256"],
                              split_sha256=out["split_sha256"], B=40,
                              fixture_override=True)

        def engine(o):
            return S.analyse(cfgx, S.config_sha256(cfgx), o["sources"],
                             o["slice_rows"], o["run_info"])
        m0 = e0rec["midpoint"]["rows"][0]["identity"]
        sid = f"{m0['file']}|{m0['slice_index']}"
        r2 = copy.deepcopy(e0rec)
        r2["midpoint"]["rows"][0]["r1"]["primary"]["128"].pop("psnr")
        lay3 = fresh()
        e0_rewrite(r2, e0pan, ctx)(lay3)
        o3 = _typed_call(cx, lambda: RS.load_campaign(lay3))
        if o3 is not None:
            res = engine(o3)
            aff = [c for c in res["comparisons"] if c["metric"] == "psnr" and
                   c["scope"] == "mid" and any(
                       x["id"] == c["id"] and ("shared", "epoch0", "r1") in (
                           (x["side"]["arm"], x["side"]["ckpt"],
                            x["side"]["readout"]),
                           (x["ref"]["arm"], x["ref"]["ckpt"],
                            x["ref"]["readout"]))
                       for x in cfgx["comparisons"])]
            other = [c for c in res["comparisons"] if c["id"] ==
                     "am3_zf|shared|epoch0|r1|ssim"]
            cx.ok(any(sid in m for m in o3["reference_missing"]) and
                  len(aff) > 0 and all(
                      c["status"] == "unavailable" and any(
                          "2027" in i and m0["file"] in i
                          for i in c["identities"]) for c in aff) and
                  other and other[0]["status"] == "ok",
                  "AM3-E4: epoch-0 R1 PSNR removed -> loaded as missing; "
                  "every comparison using it unavailable with identities; "
                  "unaffected metrics still computed",
                  detail=f"{len(aff)} affected")
        r3 = copy.deepcopy(e0rec)
        r3["midpoint"]["rows"][0].pop("r1_dc")
        r3["all_slice"]["rows"][0].pop("nll_per_dim")
        lay4 = fresh()
        e0_rewrite(r3, e0pan, ctx)(lay4)
        o4 = _typed_call(cx, lambda: RS.load_campaign(lay4))
        if o4 is not None:
            res = engine(o4)
            st = {c["id"]: c["status"] for c in res["comparisons"]}
            cx.ok(not o4["epoch0"][2027]["summary_verified"] and
                  st["am3_epoch0|lam0|selected|nll|delta"] == "unavailable" and
                  st["am3_zf|shared|epoch0|r1_dc|psnr"] == "unavailable" and
                  st["am3_zf|shared|epoch0|r0|psnr"] == "ok",
                  "AM3-E4: epoch-0 R1+DC record and all-slice NLL absent -> "
                  "affected comparisons unavailable; summary recorded as not "
                  "verifiable")
        zr = copy.deepcopy(camp["zrows"])
        zr[0].pop("ssim")
        lay5 = fresh()

        def zf_write(l2, rows):
            zd = Path(l2["zf_dir"])
            shutil.rmtree(zd)
            EV.write_zf(zd, rows, EV.zf_summary(camp["zrows"]),
                        {"fixture": True})
            l2["zf_sha256"] = EV.verify_zf(zd)
        zf_write(lay5, zr)
        o5 = _typed_call(cx, lambda: RS.load_campaign(lay5))
        if o5 is not None:
            res = engine(o5)
            st = {c["id"]: c["status"] for c in res["comparisons"]}
            cx.ok(not o5["zf"]["summary_verified"] and
                  st["am3_zf|lam2|selected|r0|ssim"] == "unavailable" and
                  st["am3_zf|lam2|selected|r0|psnr"] == "ok",
                  "AM3-E4: ZF SSIM missing on one slice -> ZF SSIM "
                  "comparisons unavailable, PSNR unaffected")

        def e0_nan(l2):
            d = Path(l2["epoch0"][2027])
            f = d / EV.E0_EVAL_FILE
            j = json.loads(f.read_bytes())
            j["midpoint"]["rows"][0]["r0"]["psnr"] = float("nan")
            f.write_text(json.dumps(j))
            (d / "SHA256SUMS").write_text("".join(
                f"{C.sha256_file(x)}  {x.name}\n" for x in sorted(
                    d.iterdir()) if x.name != "SHA256SUMS"))
        neg("LOAD_NONFINITE", e0_nan, "non-finite epoch-0 value refused "
            "(never treated as missing)")
        neg("LOAD_RECORD_INVALID", e0_rewrite(
            _edit(e0rec, lambda r: r["midpoint"]["rows"][0]["r0"].update(
                psnr="30.0")), e0pan, ctx),
            "malformed epoch-0 value (string) refused")
        neg("LOAD_POPULATION_MISMATCH", e0_rewrite(
            _edit(e0rec, lambda r: r["midpoint"]["rows"].append(
                dict(r["midpoint"]["rows"][0], identity=dict(
                    r["midpoint"]["rows"][0]["identity"],
                    slice_index=r["midpoint"]["rows"][0]["identity"][
                        "slice_index"] + 1)))), e0pan, ctx),
            "epoch-0 midpoint row for a non-midpoint slice refused")
        cx.ok(T.VAL_ROLE_PREFIX == "val_rows_e" and
              RS.MSE_TOL_REL > 0 and RS.PSNR_TOL_DB == 0.001,
              "D6 tolerances as declared (PSNR 0.001 dB = registered P2)")


def ana_group(cx: Ctx) -> None:
    """ANA: analysis stage (B2) -- PA4 difficult cases, configuration,
    gated analysis, descriptive block, write-once, determinism."""
    from seqref_mri.scripts import v05lt_analysis as A
    from seqref_mri.scripts import v05lt_stats as S
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        camp = _syn_campaign(td / "c")
        lay = camp["layout"]
        an = td / "c" / "camp" / "analysis"
        an.mkdir()
        dsha = A.select_difficult(lay, lay["difficult"], n=1)
        lay["difficult_sha256"] = dsha
        ddoc = json.loads(Path(lay["difficult"]).read_bytes())
        zmid = {r["volume_id"]: r["psnr"] for r in camp["zrows"]
                if r["midpoint"]}
        cx.ok(ddoc["volumes"] == [min(zmid, key=lambda v: (zmid[v], v))] and
              ddoc["zf_record_sha256"] == lay["zf_sha256"],
              "PA4: lowest midpoint ZF PSNR selected from the recorded ZF "
              "record; write-once with sidecar", detail=str(ddoc["values"]))
        tie = A.difficult_doc({"record_sha256": "z", "rows": [
            {"volume_id": v, "file": v, "slice_index": 0, "midpoint": True,
             "psnr": p} for v, p in (("b.h5", 1.0), ("a.h5", 1.0),
                                     ("c.h5", 0.5))]},
            ["a.h5", "b.h5", "c.h5"], "e", n=2)
        cx.ok(tie["volumes"] == ["c.h5", "a.h5"],
              "PA4: ties broken by volume ID")
        cx.raises("DIFFICULT_INPUT", lambda: A.select_difficult(
            lay, an / "d2.json", production=True, n=1),
            "PA4: n other than 10 refused in production")
        cx.raises("ARTIFACT_EXISTS", lambda: A.select_difficult(
            lay, lay["difficult"], n=1), "difficult-case file write-once")
        cp = an / "config.json"
        csha = A.write_config(lay, cp, B=200, fixture_override=True)
        cfg = json.loads(cp.read_bytes())
        cx.ok(csha == S.config_sha256(cfg) and cfg["inputs"] == {
            "eligibility_manifest_sha256": lay["eligibility_sha256"],
            "split_manifest_sha256": lay["split_sha256"]},
            "configuration written once from the verified manifests; sha256 "
            "= canonical configuration hash")
        cx.raises("STATS_FIXTURE_CONFIG", lambda: A.write_config(
            lay, an / "c2.json", production=True, B=200,
            fixture_override=True), "fixture configuration refused in "
            "production")
        reg = C.Paths().v05lt / "analysis" / "x.json"
        cx.raises("STATS_PRODUCTION_DISABLED", lambda: A.write_config(
            lay, reg, B=200, fixture_override=True),
            "registered results tree refused without production")
        o1, o2 = an / "a1.json", an / "a2.json"
        s1 = A.run_analysis(lay, cp, csha, o1)
        s2 = A.run_analysis(lay, cp, csha, o2)
        doc = json.loads(o1.read_bytes())
        cx.ok(s1 == s2 and o1.read_bytes() == o2.read_bytes(),
              "analysis document reproducible (two runs, identical bytes)",
              detail=f"{s1[:12]} {s2[:12]}")
        res, desc = doc["analysis"], doc["descriptive"]
        st = {c["status"] for c in res["comparisons"]}
        cx.ok(res["config_sha256"] == csha and len(res["comparisons"]) ==
              len(cfg["comparisons"]) and "ok" in st and
              "descriptive_d3" in st,
              "engine run on the loaded sources against the approved hash; "
              "unreached milestones descriptive (PA3)", detail=str(st))
        pc = desc["per_case"]
        nval = len(json.loads(Path(lay["split_manifest"]).read_bytes())[
            "validation"])
        cx.ok(pc["all"]["n"] == nval and
              pc["all"]["positive"] + pc["all"]["zero"] +
              pc["all"]["negative"] == pc["all"]["n"] and
              pc["difficult"]["n"] == 1 and pc["difficult_volumes"] ==
              ddoc["volumes"],
              "PA4 per-case: all validation volumes, positive / zero / "
              "negative counts, difficult set separately", detail=str(pc[
                  "all"]))
        cx.ok(set(desc["bank_sensitivity"]["runs"]) == {
            f"{a}|{s}" for a in ("lam0", "lam2") for s in (2026, 2027, 2028)}
            and all(set(v["banks"]) == {"primary", "B", "C"}
                    for v in desc["bank_sensitivity"]["runs"].values()) and
            all(set(v["psnr_db"]) == {"16", "32", "64", "128"} and
                v["minus_k128_db"]["128"] == 0.0
                for v in desc["prefix_stability"]["runs"].values()),
            "bank sensitivity (primary, B, C) and sample-count stability "
            "(16/32/64/128) at the selected checkpoint")
        nd = desc["nll_decomposition"]["runs"]["lam2|2026"]
        gap = max(abs(e["nll_prior_per_dim"] + e["nll_logdet_per_dim"] -
                      e["nll_per_dim"]) for e in nd)
        lc = desc["learning_curves"]["runs"]["lam2|2026"]
        cx.ok(gap < 1e-5 and len(nd) == 4 and len(lc["epochs"]) == 4 and
              lc["selected_epoch"] >= 1 and {"epoch0", "zf"} <= set(
                  desc["learning_curves"]["references"]),
              "learning curves and NLL decomposition per epoch (prior + "
              "log-det = total within float rounding)", detail=f"{gap:.3g}")
        dm = desc["dc_measures"]
        cx.ok(len(dm["sources"]) == 6 * 3 + 3 and
            all({"r0_dc", "r1_dc"} <= set(v) for v in dm["sources"].values())
            and set(dm["tolerances"]) >= {"a", "b", "c_rms", "c_max"},
            "DC measures vs tolerances per record (plot 6 data)")
        cx.ok(all(k in desc[b] for b in ("per_case", "bank_sensitivity",
                                         "prefix_stability",
                                         "learning_curves",
                                         "nll_decomposition", "dc_measures")
                  for k in ("definition",)) and
              len(doc["inputs"]["identities"]) > 50 and
              set(doc["code"]) == set(A.CODE_FILES),
              "definitions, source identities and code identity saved with "
              "the values")
        cx.raises("ARTIFACT_EXISTS", lambda: A.run_analysis(lay, cp, csha, o1),
                  "analysis document write-once")
        cx.raises("STATS_CONFIG_UNAPPROVED", lambda: A.run_analysis(
            lay, cp, "0" * 64, an / "a3.json"),
            "configuration other than the approved hash refused")
        cx.raises("DIFFICULT_INVALID", lambda: A.run_analysis(
            dict(lay, difficult_sha256="4" * 64), cp, csha, an / "a4.json"),
            "difficult-case file other than the recorded hash refused")
        cx.raises("ANALYSIS_FIXTURE_LAYOUT", lambda: A.run_analysis(
            lay, cp, csha, an / "a5.json", production=True),
            "fixture layout refused in production")
        cx.ok(A.main([]) == 2, "CLI refuses production "
              "(ANALYSIS_PRODUCTION_DISABLED)")
        # --- v0.24: missing references through run_analysis ---
        import copy
        import shutil
        from seqref_mri.scripts import v05lt_eval as EV
        e0rec, e0pan, ctx = camp["epoch0"][2027]
        base_ref = desc["learning_curves"]["references"]
        m0 = e0rec["midpoint"]["rows"][0]["identity"]
        sid = f"{m0['file']}|{m0['slice_index']}"
        k = [0]

        def variant(edit_rec=None, zrows=None):
            k[0] += 1
            l2 = _campaign_copy(camp, td, f"v{k[0]}")
            if edit_rec is not None:
                r2 = copy.deepcopy(e0rec)
                edit_rec(r2)
                d = Path(l2["epoch0"][2027])
                shutil.rmtree(d)
                EV.write_epoch0_eval(d, r2, e0pan, ctx)
            if zrows is not None:
                zd = Path(l2["zf_dir"])
                shutil.rmtree(zd)
                EV.write_zf(zd, zrows, EV.zf_summary(camp["zrows"]),
                            {"fixture": True})
                l2["zf_sha256"] = EV.verify_zf(zd)
                dp = Path(l2["difficult"]).with_name("difficult_v.json")
                l2["difficult"] = dp
                l2["difficult_sha256"] = A.select_difficult(l2, dp, n=1)
            return l2

        def analysed(l2):
            o = td / "out" / f"a{k[0]}.json"
            o.parent.mkdir(exist_ok=True)
            got = _typed_call(cx, lambda: A.run_analysis(l2, cp, csha, o))
            return None if got is None else json.loads(o.read_bytes())

        def st(d, cid):
            return {c["id"]: c for c in d["analysis"]["comparisons"]}[cid]
        d1 = analysed(variant(lambda r: r["midpoint"]["rows"][0].pop(
            "r1_dc")))
        if d1 is not None:
            blk = d1["descriptive"]["dc_measures"]["sources"][
                "shared|2027|epoch0"]["r1_dc"]
            c1 = st(d1, "am3_zf|shared|epoch0|r1_dc|psnr")
            cx.ok(any(sid in m for m in blk["missing"]) and
                  blk["n_with_measures"] == len(
                      e0rec["midpoint"]["rows"]) - 1 and
                  c1["status"] == "unavailable" and
                  any(m0["file"] in i for i in c1["identities"]),
                  "run_analysis: epoch-0 R1+DC record absent -> analysis "
                  "completes; DC descriptive block lists the missing slice; "
                  "dependent comparisons unavailable with identities")
        d1b = analysed(variant(lambda r: r["midpoint"]["rows"][0][
            "r0_dc"].pop("dc")))
        if d1b is not None:
            blk = d1b["descriptive"]["dc_measures"]["sources"][
                "shared|2027|epoch0"]["r0_dc"]
            cx.ok(any(sid in m and "DC measures absent" in m
                      for m in blk["missing"]) and
                  blk["n_observed"] == len(e0rec["midpoint"]["rows"]) and
                  blk["n_with_measures"] == blk["n_observed"] - 1 and
                  blk["status"].startswith("partial") and
                  blk["max_ratio_to_tolerance_available_slices"] is not None,
                  "run_analysis: epoch-0 R0+DC measures absent (record "
                  "present) -> listed as missing; maxima over the slices "
                  "with measures")
        # v0.25: an entire epoch-0 midpoint row absent (rev-3 review)
        nmid = len(e0rec["midpoint"]["rows"])
        d1c = analysed(variant(lambda r: r["midpoint"]["rows"].pop(0)))
        if d1c is not None:
            src = d1c["descriptive"]["dc_measures"]["sources"][
                "shared|2027|epoch0"]
            dc_ok = all(
                src[ro]["status"].startswith("partial") and
                any(m.endswith(f" midpoint {sid}: slice absent")
                    for m in src[ro]["missing"]) and
                src[ro]["n_expected"] == nmid and
                src[ro]["n_observed"] == nmid - 1 and
                src[ro]["n_with_measures"] == nmid - 1 and
                src[ro]["max_ratio_to_tolerance_available_slices"]
                is not None for ro in ("r0_dc", "r1_dc"))
            cmp_ok = all(
                st(d1c, c)["status"] == "unavailable" and
                any(m0["file"] in i for i in st(d1c, c)["identities"])
                for c in ("am3_zf|shared|epoch0|r1|psnr",
                          "am3_zf|shared|epoch0|r1_dc|psnr",
                          "am3_epoch0|lam0|selected|r1|psnr"))
            cx.ok(dc_ok and cmp_ok,
                  "run_analysis: entire epoch-0 midpoint row absent -> both "
                  "DC blocks partial with the slice listed, n_expected vs "
                  "n_observed distinguished, maxima over available slices; "
                  "dependent comparisons unavailable with identities")
        d2 = analysed(variant(lambda r: r["midpoint"]["rows"][0]["r1"][
            "primary"]["128"].pop("psnr")))
        if d2 is not None:
            c2 = st(d2, "am3_epoch0|lam0|selected|r1|psnr")
            cx.ok(c2["status"] == "unavailable" and
                  st(d2, "am3_epoch0|lam0|selected|r1|ssim")["status"] ==
                  "ok", "run_analysis: epoch-0 R1 PSNR missing -> R1 PSNR "
                  "comparisons unavailable, SSIM computed")
        d3 = analysed(variant(lambda r: r["all_slice"]["rows"][0].pop(
            "nll_per_dim")))
        if d3 is not None:
            ref = d3["descriptive"]["learning_curves"]["references"][
                "epoch0"]["2027"]
            cx.ok(ref["nll_per_dim"]["status"] == "unavailable" and
                  ref["nll_per_dim"]["identities"] and
                  ref["psnr"]["status"] == "ok" and
                  ref["psnr"]["value"] == base_ref["epoch0"]["2027"]["psnr"][
                      "value"] and ref["stored_summary_verified"] is False and
                  st(d3, "am3_epoch0|lam2|final|nll|delta")["status"] ==
                  "unavailable",
                  "run_analysis: epoch-0 all-slice NLL missing -> NLL "
                  "reference unavailable with identities, PSNR recomputed "
                  "from the complete population; stored summary not "
                  "presented as verified")
        d4 = analysed(variant(lambda r: [r["midpoint"]["rows"][0]["r0"].pop(
            x) for x in ("nmse_u", "zero_error")]))
        if d4 is not None:
            nd = d4["inputs"]["d6"]["checks"]["epoch0|2027"][
                "checks_not_performed"]
            cx.ok(any("nmse_u" in x and sid in x for x in nd) and
                  any("zero_error" in x and sid in x for x in nd),
                  "run_analysis: epoch-0 midpoint NMSE_u / zero_error "
                  "missing -> D6 checks listed as not performed (no "
                  "mismatch); other D6 checks still applied", detail=str(nd))
        zr = copy.deepcopy(camp["zrows"])
        zr[0].pop("ssim")
        d5 = analysed(variant(zrows=zr))
        if d5 is not None:
            zref = d5["descriptive"]["learning_curves"]["references"]["zf"]
            cx.ok(zref["ssim"]["status"] == "unavailable" and
                  zref["psnr"]["status"] == "ok" and
                  zref["stored_summary_verified"] is False and
                  st(d5, "am3_zf|lam0|final|r0|ssim")["status"] ==
                  "unavailable",
                  "run_analysis: ZF SSIM missing -> ZF SSIM reference and "
                  "comparisons unavailable; PSNR recomputed")
        for lab, fn in (
                ("primary bank is a list", lambda r: r["midpoint"]["rows"][
                    0]["r1"].update(primary=[1, 2])),
                ("prefix entry is a list", lambda r: r["midpoint"]["rows"][
                    0]["r1"]["primary"].update({"128": [1]})),
                ("undeclared bank", lambda r: r["midpoint"]["rows"][0][
                    "r1"].update(Z={})),
                ("DC measures malformed", lambda r: r["midpoint"]["rows"][0][
                    "r0_dc"].update(dc="x"))):
            l2 = variant(fn)
            _raises_typed(cx, "LOAD_RECORD_INVALID", lambda l2=l2:
                          A.run_analysis(l2, cp, csha, td / "out" /
                                         f"bad{k[0]}.json"),
                          f"run_analysis: malformed epoch-0 structure "
                          f"({lab}) -> LOAD_RECORD_INVALID")
        cx.record("ANA analysis sha256 (synthetic, informational)", s1)


# ---------------------------------------------------------------------------
# I1.8 b2 (SEQREF-MRI-V05LTI18B v0.2): RPT, PLT, E2E (addition A)
# ---------------------------------------------------------------------------
def _b2_analysis(td: Path, max_epochs: int = 4, n_difficult: int = 1,
                 edit_e0=None, factory=None) -> dict:
    """Synthetic campaign -> difficult cases -> configuration -> analysis
    document (all write-once, outside the registered tree)."""
    from seqref_mri.scripts import v05lt_analysis as A
    camp = _syn_campaign(td / "c", max_epochs=max_epochs, factory=factory)
    lay = camp["layout"]
    if edit_e0 is not None:
        import copy
        import shutil
        from seqref_mri.scripts import v05lt_eval as EV
        e0rec, e0pan, ctx = camp["epoch0"][2027]
        r2 = copy.deepcopy(e0rec)
        edit_e0(r2)
        d = Path(lay["epoch0"][2027])
        shutil.rmtree(d)
        EV.write_epoch0_eval(d, r2, e0pan, ctx)
    an = td / "c" / "camp" / "analysis"
    an.mkdir()
    lay["difficult_sha256"] = A.select_difficult(lay, lay["difficult"],
                                                 n=n_difficult)
    cp = an / "config.json"
    csha = A.write_config(lay, cp, B=200, fixture_override=True)
    ap = an / "analysis.json"
    asha = A.run_analysis(lay, cp, csha, ap)
    return {"camp": camp, "layout": lay, "config": cp, "config_sha": csha,
            "path": ap, "sha": asha, "doc": json.loads(ap.read_bytes())}


def _b2_rewrite(src_doc: dict, path: Path, edit) -> str:
    """Hashed copy of an analysis document with edit(doc) applied (valid
    sha256 + sidecar: tests the report's own structure checks)."""
    import copy
    from seqref_mri.scripts import v05lt_stats as S
    d = copy.deepcopy(src_doc)
    edit(d)
    return S.write_json_once(path, d)


def rpt_group(cx: Ctx) -> None:
    """RPT: report (B3) built only from the verified analysis document;
    content = analysis values; write-once publication; refusals."""
    import shutil
    from seqref_mri.scripts import v05lt_analysis as A
    from seqref_mri.scripts import v05lt_report as R
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        b = _b2_analysis(td)
        doc, ap, asha = b["doc"], b["path"], b["sha"]
        out = td / "rep"
        shas = R.build_report(ap, asha, out)
        rj = R.verify_report(out, asha)
        md = (out / R.REPORT_MD).read_text()
        cx.ok(rj["analysis_sha256"] == asha and rj["config_sha256"] ==
              b["config_sha"] == doc["config_sha256"] and
              rj["analysis_code"] == doc["code"] and
              rj["code"]["v05lt_report.py"] == C.sha256_file(
                  Path(R.__file__)) and len(shas) == 16,
              "report bound to the analysis and configuration sha256, the "
              "analysis code identity and its own code; 16 files + "
              "SHA256SUMS", detail=f"{len(shas)} files")
        ct, an, de = rj["content"], doc["analysis"], doc["descriptive"]
        cmp_ = {c["id"]: c for c in an["comparisons"]}
        rows_ok = all(
            r["estimate"] == cmp_[r["id"]].get("estimate") and
            r["ci"] == cmp_[r["id"]].get("ci") and
            r["status"] == cmp_[r["id"]]["status"] and
            r["per_seed"] == cmp_[r["id"]].get("per_seed", []) and
            r["support"] == cmp_[r["id"]].get("support")
            for r in ct["comparisons"])
        cx.ok(rows_ok and len(ct["comparisons"]) == len(an["comparisons"])
              and [{k: v for k, v in o.items() if k != "primary"}
                   for o in ct["outcomes"]] == an["outcomes"] and
              all(o["primary"]["id"] == f"{o['id']}|psnr"
                  for o in ct["outcomes"]),
              "outcome table and all comparisons copied from the analysis "
              "(estimates, CIs, per-seed results, labels; nothing "
              "re-estimated)", detail=f"{len(ct['comparisons'])} rows")
        cx.ok(ct["caps"]["per_source"] == an["caps"] and
              ct["caps"]["statement"] == an["cap_statement"] and
              ct["nmse_exclusion"] == de["nmse_exclusion"] and
              ct["per_case"] == de["per_case"] and
              ct["dc"]["measures"] == de["dc_measures"] and
              ct["dc"]["identity"] == an["dc_identity"] and
              ct["bank_sensitivity"] == de["bank_sensitivity"] and
              ct["prefix_stability"] == de["prefix_stability"] and
              ct["references"] == de["learning_curves"]["references"],
              "cap statement and counts, NMSE exclusion report, DC "
              "identity and measures, bank sensitivity, prefix stability, "
              "per-case summary and references copied unchanged")
        d3 = [r for r in ct["not_ok"] if r["status"] == "descriptive_d3"]
        cx.ok(all(c["id"].replace("|", "\\|") in md
                  for c in an["comparisons"]) and
              all(st_ in md for st_ in doc["statements"]) and
              an["cap_statement"] in md and R.AM4B_STATEMENT in md and
              "not an independent confirmation" in md and d3 and
              all(r["seeds_absent"] for r in d3) and
              set(r["id"] for r in ct["not_ok"]) == {
                  c["id"] for c in an["comparisons"] if c["status"] != "ok"},
              "Markdown lists every comparison, the statements (holdout "
              "already used), cap and AM4-B statements; PA3 descriptive "
              "comparisons listed with the absent seeds and run endings",
              detail=f"{len(d3)} PA3 rows")
        cx.ok(all(R._ci(p) in md for c in an["comparisons"]
                  for p in c.get("per_seed") or []) and
              all(json.dumps({"identities": r["identities"],
                              "seeds_absent": r["seeds_absent"]},
                             sort_keys=True) in md for r in ct["not_ok"]) and
              max(len(json.dumps(r["seeds_absent"])) for r in d3) > 400 and
              json.dumps({"all": ct["nmse_exclusion"]["all"]["excluded"],
                          "mid": ct["nmse_exclusion"]["mid"]["excluded"]},
                         sort_keys=True) in md,
              "Markdown: per-seed CIs shown; complete identity lists in "
              "appendices (entries longer than 400 characters present "
              "untruncated)")
        shas2 = R.build_report(ap, asha, td / "rep2")
        cx.ok(shas2 == shas, "report directory reproducible (identical "
              "bytes for every file)")
        cx.raises("ARTIFACT_EXISTS", lambda: R.build_report(ap, asha, out),
                  "write-once: existing report directory refused")
        os.symlink(td / "nowhere", td / "rep_link")
        cx.raises("ARTIFACT_EXISTS", lambda: R.build_report(
            ap, asha, td / "rep_link"), "dangling symlink at the "
            "destination refused, never followed")
        (td / ".rep3.staging").mkdir()
        cx.raises("STAGING_PRESENT", lambda: R.build_report(
            ap, asha, td / "rep3"), "staging left by an interrupted write "
            "-> STAGING_PRESENT (not deleted)")

        def boom(tag):
            raise RuntimeError(f"fixture interruption at {tag}")
        fired = []
        try:
            R.build_report(ap, asha, td / "rep4", hook=boom)
        except RuntimeError as exc:
            fired.append(str(exc))
        cx.ok(fired == ["fixture interruption at report:staged"] and
              not os.path.lexists(td / "rep4") and
              (td / ".rep4.staging" / R.SUMS).is_file(),
              "interruption after staging: nothing published, staging kept "
              "for inspection")
        reg = C.Paths().v05lt / "report_fixture"
        cx.raises("STATS_PRODUCTION_DISABLED", lambda: R.build_report(
            ap, asha, reg), "registered results tree refused without "
            "production")
        cx.raises("REPORT_ANALYSIS_HASH", lambda: R.build_report(
            ap, "0" * 64, td / "rep5"), "analysis sha256 mismatch refused")
        ap2 = td / "a_noside.json"
        shutil.copy(ap, ap2)
        cx.raises("REPORT_ANALYSIS_HASH", lambda: R.build_report(
            ap2, asha, td / "rep6"), "analysis without its sidecar refused")
        for k, (lab, ed, code) in enumerate((
                ("schema", lambda d: d.update(schema="x"),
                 "REPORT_ANALYSIS_INVALID"),
                ("configuration binding", lambda d: d["analysis"].update(
                    config_sha256="f" * 64), "REPORT_ANALYSIS_INVALID"),
                ("NMSE exclusion block absent", lambda d: d["descriptive"]
                 .pop("nmse_exclusion"), "REPORT_ANALYSIS_INVALID"),
                ("outcome without its PSNR comparison", lambda d: d[
                    "analysis"]["comparisons"].pop(0),
                 "REPORT_ANALYSIS_INVALID"),
                ("run_info disagrees with learning curves", lambda d: d[
                    "inputs"]["run_info"]["lam0|2026"].update(status="x"),
                 "REPORT_ANALYSIS_INVALID"),
                ("comparison entry a list", lambda d: d["analysis"][
                    "comparisons"].append([1]), "REPORT_ANALYSIS_INVALID"))):
            pk = td / f"bad{k}.json"
            sk = _b2_rewrite(doc, pk, ed)
            cx.raises(code, lambda: R.build_report(pk, sk, td / f"rb{k}"),
                      f"malformed analysis document ({lab}) -> typed "
                      "refusal, nothing published")
            cx.ok(not os.path.lexists(td / f"rb{k}") and
                  not os.path.lexists(td / f".rb{k}.staging"),
                  f"({lab}): no output or staging left")
        # report copies, never recomputes: an edited exclusion block is
        # rendered as given (the analysis document is the only source)
        fake = {"definition": "x", "eligibility_sha256": "e" * 64,
                "all": {"n_slices": 9, "n_excluded": 1, "excluded": [
                    {"volume_id": "v.h5", "file": "f.h5", "slice_index": 4}],
                    "null_volumes": ["v.h5"], "population_volumes": 1},
                "mid": {"n_slices": 2, "n_excluded": 1, "excluded": [
                    {"volume_id": "v.h5", "file": "f.h5", "slice_index": 4}],
                    "null_volumes": ["v.h5"], "population_volumes": 1},
                "cases_excluded": [3]}
        pk = td / "fake_excl.json"
        sk = _b2_rewrite(doc, pk, lambda d: d["descriptive"].update(
            nmse_exclusion=fake))
        R.build_report(pk, sk, td / "rf")
        mdf = (td / "rf" / R.REPORT_MD).read_text()
        cx.ok("- all: 1 of 9 slices excluded; null volumes ['v.h5']" in mdf
              and "Visual cases with an excluded midpoint slice: [3]" in mdf,
              "NMSE exclusion section renders the analysis values as given "
              "(no recomputation from manifests)")
        # analysis-side exclusion block (v0.4) on edited eligibility rows
        cm = {"slice_rows": [
            {"volume_id": "a", "file": "a", "slice_index": 0,
             "eligible": True, "midpoint": False},
            {"volume_id": "a", "file": "a", "slice_index": 1,
             "eligible": False, "midpoint": True},
            {"volume_id": "b", "file": "b", "slice_index": 0,
             "eligible": False, "midpoint": True}],
            "cases": [{"case": 5, "file": "b", "slice_index": 0}],
            "eligibility_sha256": "s"}
        ne = A.nmse_exclusion(cm)
        cx.ok(ne["all"]["n_excluded"] == 2 and ne["all"]["null_volumes"] ==
              ["b"] and ne["all"]["population_volumes"] == 1 and
              ne["mid"]["null_volumes"] == ["a", "b"] and
              ne["mid"]["population_volumes"] == 0 and
              ne["cases_excluded"] == [5],
              "analysis nmse_exclusion (AM2-C §5): excluded slices with "
              "identities, null volumes and population per scope, visual "
              "cases among the excluded midpoints")
        # verifier
        v = td / "rep_t"
        shutil.copytree(out, v)
        (v / "plot2_objective.png").write_bytes(b"x")
        cx.raises("REPORT_HASH_MISMATCH", lambda: R.verify_report(v, asha),
                  "verifier: altered plot -> REPORT_HASH_MISMATCH")
        shutil.rmtree(v)
        shutil.copytree(out, v)
        (v / "extra.txt").write_text("x")
        cx.raises("REPORT_INCOMPLETE", lambda: R.verify_report(v, asha),
                  "verifier: unlisted file -> REPORT_INCOMPLETE")
        cx.raises("REPORT_INVALID", lambda: R.verify_report(out, "1" * 64),
                  "verifier: other analysis sha256 -> REPORT_INVALID")

        def resum(dd):
            (dd / R.SUMS).write_text("".join(
                f"{C.sha256_file(f)}  {f.name}\n" for f in sorted(
                    dd.iterdir()) if f.name != R.SUMS))
        for k, (lab, keep) in enumerate((
                ("only report.json", {R.REPORT_JSON}),
                ("report.md and all plots missing", {R.REPORT_JSON}),
                ("one plot PNG missing", None))):
            dd = td / f"vm{k}"
            shutil.copytree(out, dd)
            for f in list(dd.iterdir()):
                if f.name == R.SUMS:
                    continue
                if (keep is not None and f.name not in keep) or (
                        keep is None and f.name == "plot5_nll_decomposition"
                        ".png"):
                    f.unlink()
            resum(dd)
            cx.raises("REPORT_INCOMPLETE", lambda: R.verify_report(dd, asha),
                      f"verifier: self-consistent SHA256SUMS omitting "
                      f"required files ({lab}) -> REPORT_INCOMPLETE")
        for k, (lab, ed) in enumerate((
                ("plot entries removed", lambda j: j.update(plots=[])),
                ("plot hashes removed", lambda j: [p.pop("png_sha256")
                                                   for p in j["plots"]]),
                ("report.md binding removed", lambda j: j.pop(
                    "report_md_sha256")))):
            dd = td / f"vj{k}"
            shutil.copytree(out, dd)
            j = json.loads((dd / R.REPORT_JSON).read_bytes())
            ed(j)
            (dd / R.REPORT_JSON).write_bytes(C.canonical_json(j))
            resum(dd)
            cx.raises("REPORT_INVALID", lambda: R.verify_report(dd, asha),
                      f"verifier: {lab} (checksums self-consistent) -> "
                      "REPORT_INVALID")
        cx.ok(R.main([]) == 2, "main() refuses production "
              "(REPORT_PRODUCTION_DISABLED)")
        cx.record("RPT report.json sha256 (synthetic, informational)",
                  shas[R.REPORT_JSON])


def plt_group(cx: Ctx) -> None:
    """PLT: plots 1-7 -- every drawn value equals the analysis value it
    represents; deterministic PNG without metadata; typed backend refusal."""
    import sys as _sys
    from seqref_mri.scripts import v05lt_report as R
    from seqref_mri.scripts import v05lt_stats as S
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        b = _b2_analysis(td)
        doc = b["doc"]
        files, plots, mv = R.render_plots(doc)
        files2, plots2, _ = R.render_plots(doc)
        names = [n for n, _ in R.PLOTS]
        cx.ok([p["name"] for p in plots] == names and files == files2 and
              all(files[f"{n}.png"][:8] == b"\x89PNG\r\n\x1a\n" and
                  b"Software" not in files[f"{n}.png"] for n in names),
              "plots 1-7 rendered as PNG without the Software metadata; "
              "byte-identical on a second rendering",
              detail=f"matplotlib {mv}")
        dat = {n: json.loads(files[f"{n}.json"]) for n in names}
        cx.ok(all(dat[n]["schema"] == R.PLOT_SCHEMA and dat[n]["plot"] == n
                  and dat[n]["definition"] == fn.__doc__.strip()
                  for n, fn in R.PLOTS),
              "each data file carries its schema, plot name and definition")
        lc = doc["descriptive"]["learning_curves"]
        ok1 = True
        for key, r in lc["runs"].items():
            d = dat["plot1_learning_curves"]["data"]["runs"][key]
            st = {e["epoch"]: e["step"] for e in r["epochs"]}
            ok1 &= (d["psnr"] == [e["psnr"] for e in r["epochs"]] and
                    d["nll_per_dim"] == [e["nll_per_dim"]
                                         for e in r["epochs"]] and
                    d["selected_step"] == st[r["selected_epoch"]] and
                    d["stop_step"] == st[r["stop_epoch"]] and
                    d["epoch3_step"] == st[3] and
                    d["epoch0"]["psnr"] == lc["references"]["epoch0"][
                        key.split("|")[1]]["psnr"])
        cx.ok(ok1 and dat["plot1_learning_curves"]["data"]["zf_psnr"] ==
              lc["references"]["zf"]["psnr"],
              "plot 1: PSNR/NLL per update, selected/stop/epoch-3 markers, "
              "epoch-0 and ZF references = analysis values")
        cmp_ = {c["id"]: c for c in doc["analysis"]["comparisons"]}
        p2 = dat["plot2_objective"]["data"]
        cx.ok(set(p2["comparisons"]) == {"between|selected|r0|psnr"} |
              {f"between|epoch{m}|r0|psnr" for m in S.MILESTONES} and
              all(v["estimate"] == cmp_[k].get("estimate") and
                  v["ci"] == cmp_[k].get("ci") and v["per_seed"] ==
                  cmp_[k].get("per_seed", []) for k, v in
                  p2["comparisons"].items()) and
              p2["reference_lines_db"] == [0.0, S.PRACTICAL_DB],
              "plot 2: selected + matched milestones with CI and per-seed "
              "results; 0 and +0.5 dB reference lines")
        p3 = dat["plot3_readouts"]["data"]
        bs = doc["descriptive"]["bank_sensitivity"]["runs"]
        cx.ok(p3["psnr"] and all(v["estimate"] == cmp_[k]["estimate"]
                                 for m in ("psnr", "nmse_u")
                                 for k, v in p3[m].items()) and
              all(p3["banks"][b_][k] == bs[k]["banks"][b_]["psnr"]
                  for b_ in p3["banks"] for k in bs),
              "plot 3: R1 - R0 PSNR and NMSE_u comparisons; per-bank R1 "
              "PSNR at selected = analysis values")
        p4 = dat["plot4_tradeoff"]["data"]
        ok4 = all(p4["bound"][str(s)]["bound"] ==
                  p4["bound"][str(s)]["lam0_selected_nll_per_dim"] +
                  S.DENSITY_BOUND * abs(p4["bound"][str(s)][
                      "lam0_selected_nll_per_dim"]) and
                  p4["bound"][str(s)]["lam0_selected_nll_per_dim"] ==
                  {e["epoch"]: e for e in lc["runs"][f"lam0|{s}"]["epochs"]}[
                      lc["runs"][f"lam0|{s}"]["selected_epoch"]][
                      "nll_per_dim"] for s in S.SEEDS)
        cx.ok(ok4, "plot 4: per-seed reference = N_ref + 0.01 |N_ref| "
              "(registered r definition; display only)")
        import copy
        neg = copy.deepcopy(doc)
        for s_ in S.SEEDS:
            r_ = neg["descriptive"]["learning_curves"]["runs"][f"lam0|{s_}"]
            for e in r_["epochs"]:
                if e["epoch"] == r_["selected_epoch"]:
                    e["nll_per_dim"] = -2.0
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt_
        fig4, d4 = R.plot4(neg, plt_)
        plt_.close(fig4)
        cx.ok(all(d4["bound"][str(s_)]["bound"] == -1.98
                  for s_ in S.SEEDS),
              "plot 4 regression: N_ref = -2 -> reference -1.98 (r = +1 %), "
              "not -2.02")
        ex = copy.deepcopy(doc)
        for c in ex["analysis"]["comparisons"]:
            if c["id"] == "between|selected|r0|psnr":
                c["estimate"], c["ci"] = 0.0, [0.1, 0.2]
        fx, px, _ = R.render_plots(ex)
        pdx = json.loads(fx["plot2_objective.json"])["data"]["comparisons"][
            "between|selected|r0|psnr"]
        cx.ok(pdx["estimate"] == 0.0 and pdx["ci"] == [0.1, 0.2] and
              len(px) == 7,
              "CI not containing the estimate (valid percentile interval) "
              "plotted without error; values unchanged")
        nd = doc["descriptive"]["nll_decomposition"]
        p5 = dat["plot5_nll_decomposition"]["data"]
        cx.ok(p5["sign_convention"] == nd["sign_convention"] and all(
            p5["runs"][k][f] == [e[f] for e in eps] for k, eps in
            nd["runs"].items() for f in ("nll_prior_per_dim",
                                         "nll_logdet_per_dim")),
            "plot 5: prior and log-determinant contributions = analysis "
            "values, sign convention carried")
        p6 = dat["plot6_dc_measures"]["data"]
        dm = doc["descriptive"]["dc_measures"]
        cx.ok(p6["tolerances"] == dm["tolerances"] and all(
            p6["sources"][s][ro][k] == dm["sources"][s][ro][k]
            for s in dm["sources"] for ro in S.DC_READOUTS
            for k in ("max_ratio_to_tolerance_available_slices",
                      "n_expected", "n_observed", "status")),
            "plot 6: maximum ratios over available slices, coverage and "
            "status = analysis values")
        pc = doc["descriptive"]["per_case"]
        p7 = dat["plot7_per_case"]["data"]
        cx.ok(p7["all"] == pc["all"] and p7["difficult"] == pc["difficult"]
              and sorted(x["delta"] for x in p7["points"]) ==
              sorted(x["delta"] for x in pc["delta_db"]) and
              sum(x["difficult"] for x in p7["points"]) ==
              len(pc["difficult_volumes"]),
              "plot 7: per-volume Δ, difficult set and counts = analysis "
              "values")
        saved = _sys.modules.get("matplotlib")
        _sys.modules["matplotlib"] = None
        try:
            cx.raises("REPORT_PLOT_BACKEND", lambda: R.render_plots(doc),
                      "plotting backend unavailable -> REPORT_PLOT_BACKEND "
                      "(no fallback renderer; fixture patch)")
        finally:
            if saved is None:
                _sys.modules.pop("matplotlib", None)
            else:
                _sys.modules["matplotlib"] = saved
        cx.record("PLT matplotlib version (informational)", mv)


def e2e_group(cx: Ctx) -> None:
    """E2E (addition A): synthetic campaign from stub-trainer runs (11
    epochs: milestones 3 and 10 reached by both arms in all seeds) ->
    verified loader -> engine -> analysis -> report -> plots -> verifier.
    Tests this reporting chain only; I1.10 remains the campaign smoke
    test."""
    from seqref_mri.scripts import v05lt_report as R
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        b = _b2_analysis(td, max_epochs=11)
        doc = b["doc"]
        cmp_ = {c["id"]: c for c in doc["analysis"]["comparisons"]}
        ri = doc["inputs"]["run_info"]
        cx.ok(all(v["milestones"] == [3, 10] and v["last_epoch"] == 11
                  for v in ri.values()) and
              cmp_["between|epoch10|r0|psnr"]["status"] == "ok" and
              cmp_["between|epoch20|r0|psnr"]["status"] ==
              "descriptive_d3",
              "chain: milestone 10 reached by both arms in all seeds -> "
              "campaign-level comparison; milestone 20 -> PA3 descriptive",
              detail=str(sorted({v["status"] for v in ri.values()})))
        side = Path(str(b["path"]) + ".sha256").read_text().split()[0]
        cx.ok(C.sha256_file(b["path"]) == b["sha"] == side and
              C.sha256_file(b["config"]) == b["config_sha"] ==
              doc["config_sha256"] and all(
                  Path(k).is_file() and C.sha256_file(Path(k)) == v
                  for k, v in doc["inputs"]["identities"].items()),
              "chain bindings: configuration -> analysis -> every input "
              "file identity re-hashed equal",
              detail=f"{len(doc['inputs']['identities'])} inputs")
        out = td / "report"
        shas = R.build_report(b["path"], b["sha"], out)
        rj = R.verify_report(out, b["sha"])
        md = (out / R.REPORT_MD).read_text()
        cx.ok(rj["analysis_sha256"] == b["sha"] and len(rj["plots"]) == 7
              and "between\\|epoch10\\|r0" in md and
              len(rj["content"]["outcomes"]) ==
              len(doc["analysis"]["outcomes"]),
              "report and plots 1-7 built from the chain's analysis and "
              "verified", detail=f"{len(shas)} files")
        cx.record("E2E analysis sha256 (synthetic, informational)", b["sha"])


# ---------------------------------------------------------------------------
# I1.8 b3 (SEQREF-MRI-V05LTI18B3 v0.1): HOA, HOL, F18A -- synthetic only
# ---------------------------------------------------------------------------
B3_N_HOLD = 6                    # synthetic holdout volumes (fixture only)
B3_W = 368                       # synthetic k-space width (fixture only)


def _b3_hold_manifest(d: Path, n: int = B3_N_HOLD) -> dict:
    """Synthetic V02-format holdout manifest (fixture only) + sidecar."""
    d.mkdir(parents=True, exist_ok=True)
    body = {"schema": C.V02_SCHEMA, "kind": "holdout", "fixture": True,
            "n_volumes": n,
            "entries": [{"file": f"singlecoil_val/fixhold{k:03d}.h5",
                         "slice_index": 10 + k % 4, "dataset_index": k}
                        for k in range(n)]}
    csha = C.sha256_bytes(C.canonical_json(body))
    doc = dict(body, manifest_sha256=csha)
    p = d / "v02_holdout_manifest.json"
    fsha = C.write_durable_new(p, C.canonical_json(doc))
    C.write_durable_new(Path(str(p) + ".sha256"), f"{fsha}  {p.name}\n".encode())
    return {"manifest": p, "manifest_file_sha256": fsha,
            "manifest_content_sha256": csha, "doc": doc}


def _b3_hold_record(run: dict, role: str, sel: dict, man: dict,
                    edit=None) -> dict:
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_results as RS
    ck = run["checkpoints"][role]
    key = f"{run['arm']}|{run['seed']}"
    rng = np.random.default_rng(
        int(C.sha256_bytes(f"{key}|{role}".encode())[:8], 16))
    rows = []
    mrows = RS.load_holdout_manifest(
        man["manifest"], RS.Identities(),
        file_sha256=man["manifest_file_sha256"],
        content_sha256=man["manifest_content_sha256"], fixture=True)["rows"]
    seeds = RS.holdout_mask_seeds(mrows)
    for e, sd in zip(man["doc"]["entries"], seeds):
        def ro(nll):
            r = {"psnr": float(25 + rng.random()), "ssim": float(
                0.5 + 0.1 * rng.random()), "mse": float(1e-3 * (1 + rng.random())),
                 "nmse_u": float(0.5 + rng.random()), "psnr_capped": False,
                 "zero_error": False}
            if nll:
                r["nll_per_dim"] = float(1.4 + 0.01 * rng.random())
            return r
        rows.append({"volume_id": Path(e["file"]).name, "file": e["file"],
                     "slice_index": e["slice_index"], "mask_seed": sd,
                     "r0": ro(True), "r1": ro(False)})
    doc = {"schema": RS.HOLDOUT_SCHEMA, "seed": run["seed"],
           "arm": run["arm"], "role": role, "epoch": ck["epoch"],
           "model_state_sha256": ck["model_state_sha256"],
           "checkpoint_sha256": sel["runs"][key][role]["sha256"],
           "manifest_content_sha256": man["manifest_content_sha256"],
           "masks": {"convention": RS.HOLDOUT_MASK_CONVENTION,
                     "sha256": RS.holdout_masks_sha256(
                         seeds, [B3_W] * len(seeds)),
                     "seeds_sha256": RS.holdout_mask_seeds_sha256(mrows),
                     "widths": [B3_W] * len(seeds)},
           "bank": {"name": "primary", "array_sha256":
                    EV.REGISTERED_BANK_ARRAY["primary"],
                    "n_decodes": EV.BANK_ROWS},
           "rows": rows, "fixture": True}
    if edit is not None:
        edit(doc)
    return doc


def _b3_setup(td: Path, **campaign) -> dict:
    """b2 chain + report + epoch-0 stores + selection manifest + synthetic
    holdout manifest, records and holdout configuration (fixture only)."""
    from seqref_mri.scripts import v05lt_analysis as A
    from seqref_mri.scripts import v05lt_report as R
    from seqref_mri.scripts import v05lt_results as RS
    from seqref_mri.scripts import v05lt_stats as S
    b = _b2_analysis(td, **campaign)
    root = td / "c"
    lay = b["layout"]
    an = b["path"].parent
    rep = an / "report"
    R.build_report(b["path"], b["sha"], rep)
    camp = RS.load_campaign(lay)
    lay["epoch0_store"] = {}
    for s_ in lay["seeds"]:
        d = root / "camp" / "epoch0_store" / f"seed_{s_}"
        d.mkdir(parents=True)
        st = C.ArtifactStore(d / "ckpt", C.Ledger(d / "ledger.jsonl"),
                             d / "_incomplete")
        state = camp["runs"][f"lam0|{s_}"]["epoch0_model_state"]
        st.stage(f"epoch0_seed{s_}.pt",
                 lambda fh, s_=s_: fh.write(f"fixture epoch0 {s_}".encode()))
        st.commit({"epoch0": f"epoch0_seed{s_}.pt"},
                  {"seed": s_, "model_state_sha256": state})
        lay["epoch0_store"][s_] = d
    selp = an / "selection.json"
    selsha = A.write_selection_manifest(
        lay, root, selp, config=(b["config"], b["config_sha"]),
        analysis=(b["path"], b["sha"]), report_dir=rep)
    sel = A.load_selection(selp, selsha)
    man = _b3_hold_manifest(root / "camp" / "holdout_manifest")
    recdir = root / "camp" / "holdout"
    recdir.mkdir()
    recs = {}
    for run in camp["runs"].values():
        for role in RS.HOLDOUT_ROLES:
            k = f"{run['arm']}|{run['seed']}|{role}"
            p = recdir / f"{run['arm']}_{run['seed']}_{role}.json"
            recs[k] = {"path": p, "sha256": S.write_json_once(
                p, _b3_hold_record(run, role, sel, man))}
    hold = {k: man[k] for k in ("manifest", "manifest_file_sha256",
                                "manifest_content_sha256")}
    hold["records"] = recs
    hcp = an / "holdout_config.json"
    hsha = A.write_holdout_config(b["config"], b["config_sha"], hold, hcp,
                                  fixture=True)
    msha = json.loads(hcp.read_bytes())["holdout"]["mask_seeds_sha256"]
    return {**b, "root": root, "report": rep, "camp": camp, "msha": msha,
            "selection": (selp, selsha), "sel": sel, "man": man,
            "holdout": hold, "hconfig": (hcp, hsha), "recdir": recdir}


def _b3_hold_variant(B: dict, td: Path, name: str, key: str = None,
                     edit=None, extra: dict = None) -> dict:
    """Copy of the holdout layout with one record rewritten (edit) and/or
    extra records added; records written under td (fixture only)."""
    from seqref_mri.scripts import v05lt_stats as S
    h = dict(B["holdout"], records=dict(B["holdout"]["records"]))
    d = td / name
    d.mkdir()
    if key is not None:
        arm, seed, role = key.split("|")
        run = B["camp"]["runs"][f"{arm}|{seed}"]
        p = d / "rec.json"
        h["records"][key] = {"path": p, "sha256": S.write_json_once(
            p, _b3_hold_record(run, role, B["sel"], B["man"], edit))}
    for k, v in (extra or {}).items():
        h["records"][k] = v
    return h


def _b3_hold_all(B: dict, td: Path, name: str, edit) -> dict:
    """Every holdout record rewritten with the same edit (fixture only)."""
    from seqref_mri.scripts import v05lt_stats as S
    h = dict(B["holdout"], records={})
    d = td / name
    d.mkdir()
    for key in B["holdout"]["records"]:
        arm, seed, role = key.split("|")
        p = d / f"{arm}_{seed}_{role}.json"
        h["records"][key] = {"path": p, "sha256": S.write_json_once(
            p, _b3_hold_record(B["camp"]["runs"][f"{arm}|{seed}"], role,
                               B["sel"], B["man"], edit))}
    return h


def hoa_group(cx: Ctx) -> None:
    """HOA: holdout addendum in the engine (196 b3 registration) --
    21 comparisons, appended population, validation preservation, label
    scope, AM4-B NMSE_u, refusals."""
    import copy
    from seqref_mri.scripts import v05lt_analysis as A
    from seqref_mri.scripts import v05lt_stats as S
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        B = _b3_setup(td)
        vcfg = A.load_config(B["config"], B["config_sha"])
        hcfg = A.load_config(*B["hconfig"])
        hc, hg = S.holdout_comparisons()
        ids = [c["id"] for c in hc]
        cx.ok(len(hc) == 21 and len(hg) == 6 and len(set(ids)) == 21 and
              sum(c["family"] == "holdout_between" for c in hc) == 7 and
              sum(c["family"] == "holdout_within" for c in hc) == 14 and
              {(c["side"]["ckpt"], c["ref"]["ckpt"]) for c in hc} ==
              {("selected", "selected"), ("selected", "epoch3")} and
              {c["side"]["readout"] for c in hc} == {"r0", "r1"},
              "21 registered holdout comparisons (7 between at selected, "
              "7 + 7 within selected vs epoch 3; R0 / R1; 6 outcome groups)")
        cx.ok([p["id"] for p in hcfg["populations"]] ==
              ["val", "nmse_all", "nmse_mid", "hold"] and
              hcfg["comparisons"][:len(vcfg["comparisons"])] ==
              vcfg["comparisons"] and
              S.config_sha256(S.validation_part(hcfg)) == B["config_sha"] and
              hcfg["holdout"]["slices"]["n"] == B3_N_HOLD and
              hcfg["holdout"]["evaluation_scope"] == S.HOLDOUT_SCOPE_TEXT,
              "holdout population appended after every validation population;"
              " validation part byte-identical to the approved configuration")
        # requirement 2: validation preservation
        ho = td / "c" / "camp" / "analysis" / "hold_analysis.json"
        hsha = A.run_analysis(B["layout"], B["hconfig"][0], B["hconfig"][1],
                              ho, holdout=B["holdout"],
                              selection=B["selection"])
        hd = json.loads(ho.read_bytes())
        va = {c["id"]: c for c in B["doc"]["analysis"]["comparisons"]}
        ha = {c["id"]: c for c in hd["analysis"]["comparisons"]}
        same = all(ha[k] == v for k, v in va.items())
        vs = B["doc"]["analysis"]["stream"]
        cx.ok(same and hd["analysis"]["stream"][:len(vs)] == vs and
              hd["analysis"]["outcomes"][:len(B["doc"]["analysis"][
                  "outcomes"])] == B["doc"]["analysis"]["outcomes"] and
              hsha != B["sha"],
              "validation preservation: resample arrays (sha256), estimates, "
              "CIs, per-seed results, labels and outcomes identical with the "
              "addendum; combined document hash differs",
              detail=f"{len(va)} validation comparisons")
        hres = [ha[i] for i in ids]
        lab_ok = all(
            (c["status"] == "ok") and
            (("support" in c and "practical" in c and "density" not in c)
             if c["metric"] == "psnr" else
             ("density" in c and "support" not in c)
             if c["stat"] == "ratio" else
             not ({"support", "practical", "density"} & set(c)))
            for c in hres)
        hout = hd["analysis"]["outcomes"][-6:]
        cx.ok(lab_ok and all(o["outcome"] in range(1, 7) for o in hout) and
              [o["id"] for o in hout] == [g["id"] for g in hg],
              "label scope: PSNR support + practical, NLL ratio density, "
              "SSIM / NMSE_u estimates and CIs only; 6 holdout outcomes")
        n_all = {c["n_volumes"] for c in hres}
        cx.ok(n_all == {B3_N_HOLD} and all(len(c["per_seed"]) == 3
                                           for c in hres),
              "AM4-B: NMSE_u and every holdout metric over all registered "
              "holdout volumes (no exclusion); per-seed results present")
        # independent check of one estimate
        def vol(src, m):
            return [r[m] for r in src]
        from seqref_mri.scripts import v05lt_results as RS
        hr = RS.load_holdout_records(B["holdout"], B["camp"], B["sel"],
                                     RS.Identities(), fixture=True,
            mask_seeds_sha256=B["msha"])
        srcs = hr["sources"]
        diffs = [a - r_ for s_ in S.SEEDS for a, r_ in zip(
            vol(srcs[f"lam2|{s_}|selected|r1|hold"], "psnr"),
            vol(srcs[f"lam0|{s_}|selected|r1|hold"], "psnr"))]
        est = ha["holdout|between|selected|r1|psnr"]["estimate"]
        cx.ok(math.isclose(est, math.fsum(diffs) / len(diffs), rel_tol=0,
                           abs_tol=1e-12),
              "holdout between R1 dPSNR = independent mean of paired "
              "differences over seeds and volumes", detail=f"{est:.6g}")
        # refusals
        cx.raises("STATS_HOLDOUT_MISMATCH", lambda: S.analyse(
            vcfg, B["config_sha"], {**B["camp"]["sources"], **srcs},
            B["camp"]["slice_rows"], B["camp"]["run_info"],
            holdout_rows=hr["rows"]), "holdout rows without the addendum "
            "refused")
        cx.raises("STATS_HOLDOUT_MISMATCH", lambda: S.analyse(
            hcfg, B["hconfig"][1], {**B["camp"]["sources"], **srcs},
            B["camp"]["slice_rows"], B["camp"]["run_info"]),
            "addendum without holdout rows refused")
        bad = copy.deepcopy(hcfg)
        bad["comparisons"][-1]["labels"] = "psnr"
        cx.raises("STATS_CONFIG_INVALID", lambda: S.analyse(
            bad, S.config_sha256(bad), {**B["camp"]["sources"], **srcs},
            B["camp"]["slice_rows"], B["camp"]["run_info"],
            holdout_rows=hr["rows"]), "altered holdout comparison list "
            "(label scope) refused")
        cx.raises("STATS_CONFIG_INVALID", lambda: S.build_holdout_config(
            hcfg, hr["rows"], manifest_content_sha256="x",
            mask_seeds_sha256="f" * 64),
            "second addendum refused")
        over = [dict(hr["rows"][0])] + hr["rows"][1:]
        over[0].update(file=B["camp"]["slice_rows"][0]["file"],
                       slice_index=B["camp"]["slice_rows"][0]["slice_index"])
        ocfg = S.build_holdout_config(vcfg, over,
                                      manifest_content_sha256="x",
            mask_seeds_sha256="f" * 64)
        cx.raises("STATS_CONFIG_INVALID", lambda: S.analyse(
            ocfg, S.config_sha256(ocfg), {**B["camp"]["sources"], **srcs},
            B["camp"]["slice_rows"], B["camp"]["run_info"],
            holdout_rows=over), "holdout slice inside the validation "
            "population refused")
        s2 = copy.deepcopy(srcs)
        s2["lam2|2026|selected|r0|hold"][0]["nmse_u"] = None
        cx.raises("STATS_VALUE_MISSING", lambda: S.analyse(
            hcfg, B["hconfig"][1], {**B["camp"]["sources"], **s2},
            B["camp"]["slice_rows"], B["camp"]["run_info"],
            holdout_rows=hr["rows"]), "AM4-B: missing holdout NMSE_u is a "
            "validity failure (no dropped slice)")
        cx.record("HOA holdout analysis sha256 (synthetic, informational)",
                  hsha)


def hol_group(cx: Ctx) -> None:
    """HOL: holdout record schema and verified loader; holdout analysis and
    report; scope, identity, mask, bank, binding and AM4-B refusals."""
    from seqref_mri.scripts import v05lt_analysis as A
    from seqref_mri.scripts import v05lt_report as R
    from seqref_mri.scripts import v05lt_results as RS
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        B = _b3_setup(td)
        hr = RS.load_holdout_records(B["holdout"], B["camp"], B["sel"],
                                     RS.Identities(), fixture=True,
            mask_seeds_sha256=B["msha"])
        cx.ok(len(hr["sources"]) == 24 and len(hr["records"]) == 12 and
              all(len(v) == B3_N_HOLD for v in hr["sources"].values()) and
              all("nll_per_dim" in r for k, v in hr["sources"].items()
                  if "|r0|" in k for r in v) and
              not any("nll_per_dim" in r for k, v in hr["sources"].items()
                      if "|r1|" in k for r in v),
              "12 records (6 runs x selected / epoch 3) -> 24 sources (R0 "
              "with NLL, R1 without); every manifest slice present")
        cx.raises("LOAD_HOLDOUT_MANIFEST", lambda: RS.load_holdout_records(
            B["holdout"], B["camp"], B["sel"], RS.Identities(),
            mask_seeds_sha256=B["msha"]),
            "unregistered (fixture) manifest hashes refused outside fixtures")
        k0 = "lam2|2026|selected"
        run = B["camp"]["runs"]["lam2|2026"]
        fin = td / "fin"
        fin.mkdir()
        from seqref_mri.scripts import v05lt_stats as S
        fp = fin / "final.json"
        fdoc = _b3_hold_record(run, "selected", B["sel"], B["man"],
                               lambda d: d.update(role="final"))
        ext = {"lam2|2026|final": {"path": fp,
                                   "sha256": S.write_json_once(fp, fdoc)}}
        cases = [
            ("record outside the evaluation scope (final)", "LOAD_HOLDOUT_SCOPE",
             _b3_hold_variant(B, td, "v1", extra=ext)),
            ("record missing", "LOAD_HOLDOUT_MISSING",
             dict(B["holdout"], records={k: v for k, v in
                                         B["holdout"]["records"].items()
                                         if k != k0})),
            ("record hash differs from recorded", "LOAD_HOLDOUT_HASH",
             dict(B["holdout"], records=dict(B["holdout"]["records"], **{
                 k0: dict(B["holdout"]["records"][k0], sha256="0" * 64)}))),
            ("rows out of manifest order", "LOAD_POPULATION_MISMATCH",
             _b3_hold_variant(B, td, "v2", k0, lambda d: d["rows"].reverse())),
            ("wrong slice index", "LOAD_POPULATION_MISMATCH",
             _b3_hold_variant(B, td, "v3", k0, lambda d: d["rows"][0].update(
                 slice_index=d["rows"][0]["slice_index"] + 1))),
            ("row missing", "LOAD_POPULATION_MISMATCH",
             _b3_hold_variant(B, td, "v4", k0, lambda d: d["rows"].pop())),
            ("mask convention", "LOAD_MASK_MISMATCH",
             _b3_hold_variant(B, td, "v5", k0, lambda d: d["masks"].update(
                 convention="training masks"))),
            ("masks sha256 differs between records", "LOAD_MASK_MISMATCH",
             _b3_hold_variant(B, td, "v6", k0, lambda d: d["masks"].update(
                 sha256="e" * 64))),
            ("R1 decode count 64", "LOAD_HOLDOUT_BANK",
             _b3_hold_variant(B, td, "v7", k0, lambda d: d["bank"].update(
                 n_decodes=64))),
            ("bank B instead of primary", "LOAD_HOLDOUT_BANK",
             _b3_hold_variant(B, td, "v8", k0, lambda d: d["bank"].update(
                 name="B"))),
            ("checkpoint sha256 not the selection manifest's",
             "LOAD_HOLDOUT_BINDING",
             _b3_hold_variant(B, td, "v9", k0, lambda d: d.update(
                 checkpoint_sha256="c" * 64))),
            ("model state not the verified commit's", "LOAD_HOLDOUT_BINDING",
             _b3_hold_variant(B, td, "v10", k0, lambda d: d.update(
                 model_state_sha256="d" * 64))),
            ("epoch not the verified commit's", "LOAD_HOLDOUT_BINDING",
             _b3_hold_variant(B, td, "v11", k0, lambda d: d.update(
                 epoch=d["epoch"] + 1))),
            ("other holdout manifest content", "LOAD_HOLDOUT_BINDING",
             _b3_hold_variant(B, td, "v12", k0, lambda d: d.update(
                 manifest_content_sha256="e" * 64))),
            ("AM4-B: NMSE_u undefined (zero energy)", "HOLDOUT_NMSE_UNDEFINED",
             _b3_hold_variant(B, td, "v13", k0, lambda d: d["rows"][0]["r1"]
                              .update(nmse_u=None))),
            ("non-finite PSNR", "LOAD_NONFINITE",
             _b3_hold_variant(B, td, "v14", k0, lambda d: d["rows"][0]["r0"]
                              .update(psnr=float("inf")))),
            ("NLL on R1", "LOAD_RECORD_INVALID",
             _b3_hold_variant(B, td, "v15", k0, lambda d: d["rows"][0]["r1"]
                              .update(nll_per_dim=1.0))),
            ("NLL missing on R0", "LOAD_VALUE_MISSING",
             _b3_hold_variant(B, td, "v16", k0, lambda d: d["rows"][0]["r0"]
                              .pop("nll_per_dim"))),
            ("schema", "LOAD_HOLDOUT_INVALID",
             _b3_hold_variant(B, td, "v17", k0, lambda d: d.update(
                 schema="x")))]
        for lab, code, hv in cases:
            cx.raises(code, lambda hv=hv: RS.load_holdout_records(
                hv, B["camp"], B["sel"], RS.Identities(), fixture=True,
            mask_seeds_sha256=B["msha"]),
                f"holdout loader: {lab} -> {code}")
        # mask identity (rev 2): fixed independently of the records
        import hashlib
        from seqref_mri.src import fastmri_data as FD
        mrows = hr["rows"]
        ind = [int.from_bytes(hashlib.sha256(
            f"{FD.EVAL_BASE_SEED}|{r['file']}|{r['slice_index']}".encode())
            .digest()[:8], "big") for r in mrows]
        hm = hashlib.sha256()
        for sd in ind:
            hm.update(np.ascontiguousarray(FD.make_cartesian_mask(
                B3_W, sd)).tobytes(order="C"))
        cx.ok(RS.holdout_mask_seeds(mrows) == ind and B["msha"] ==
              C.sha256_bytes(C.canonical_json(
                  [[r["file"], r["slice_index"], sd]
                   for r, sd in zip(mrows, ind)])) and
              hr["masks_sha256"] == hm.hexdigest(),
              "mask identity: seeds = independent sha256(EVAL_BASE_SEED|"
              "file|slice); frozen seed identity in the holdout "
              "configuration; accepted masks sha256 = independent "
              "recomputation over make_cartesian_mask")
        wrong = hashlib.sha256(b"not the registered masks").hexdigest()
        mcases = [
            ("valid but incorrect masks sha256 shared by ALL records",
             _b3_hold_all(B, td, "m4", lambda d: d["masks"].update(
                 sha256=wrong))),
            ("record seed identity differs from the frozen one",
             _b3_hold_variant(B, td, "m5", k0, lambda d: d["masks"].update(
                 seeds_sha256="a" * 64))),
            ("row mask seed not canonical_mask_seed",
             _b3_hold_variant(B, td, "m6", k0, lambda d: d["rows"][0].update(
                 mask_seed=d["rows"][0]["mask_seed"] + 1))),
            ("mask widths differ between records",
             _b3_hold_variant(B, td, "m7", k0, lambda d: d["masks"].update(
                 widths=[B3_W + 2] * len(d["rows"])))),
            ("mask widths missing", _b3_hold_all(
                B, td, "m8", lambda d: d["masks"].pop("widths")))]
        fcases = [
            ("masks sha256 empty in every record",
             _b3_hold_all(B, td, "m1", lambda d: d["masks"].update(
                 sha256=""))),
            ("masks sha256 short ('abc') in every record",
             _b3_hold_all(B, td, "m2", lambda d: d["masks"].update(
                 sha256="abc"))),
            ("masks sha256 upper-case hex in every record",
             _b3_hold_all(B, td, "m3", lambda d: d["masks"].update(
                 sha256=d["masks"]["sha256"].upper())))]
        for lab, hv in fcases:
            cx.raises("LOAD_MASK_INVALID", lambda hv=hv:
                      RS.load_holdout_records(
                          hv, B["camp"], B["sel"], RS.Identities(),
                          fixture=True, mask_seeds_sha256=B["msha"]),
                      f"holdout masks: {lab} -> LOAD_MASK_INVALID")
        mcases += [
            ("records AND expected identity consistently claim another "
             "seed identity", _b3_hold_all(B, td, "m9", lambda d: d[
                 "masks"].update(seeds_sha256=wrong)), wrong),
            ("records AND expected identity consistently invalid ('abc')",
             _b3_hold_all(B, td, "m10", lambda d: d["masks"].update(
                 seeds_sha256="abc")), "abc")]
        for case in mcases:
            lab, hv = case[0], case[1]
            ex = case[2] if len(case) > 2 else B["msha"]
            cx.raises("LOAD_MASK_MISMATCH", lambda hv=hv, ex=ex:
                      RS.load_holdout_records(
                          hv, B["camp"], B["sel"], RS.Identities(),
                          fixture=True, mask_seeds_sha256=ex),
                      f"holdout masks: {lab} -> LOAD_MASK_MISMATCH")
        for lab, ex in (("absent", None), ("invalid form", "abc"),
                        ("valid sha256 of another identity", wrong)):
            cx.raises("LOAD_MASK_MISMATCH", lambda ex=ex:
                      RS.load_holdout_records(
                          B["holdout"], B["camp"], B["sel"], RS.Identities(),
                          fixture=True, mask_seeds_sha256=ex),
                      f"expected mask identity {lab} -> LOAD_MASK_MISMATCH")
        # manifest identity
        mp = Path(B["holdout"]["manifest"])
        alt = td / "altman"
        alt.mkdir()
        d2 = dict(B["man"]["doc"])
        d2["entries"] = list(d2["entries"])[1:]
        ap_ = alt / mp.name
        fs = C.write_durable_new(ap_, C.canonical_json(d2))
        C.write_durable_new(Path(str(ap_) + ".sha256"),
                            f"{fs}  {ap_.name}\n".encode())
        cx.raises("V02_SOURCE_CONTENT_HASH", lambda: RS.load_holdout_manifest(
            ap_, RS.Identities(), file_sha256=fs,
            content_sha256=B["holdout"]["manifest_content_sha256"],
            fixture=True), "holdout manifest content altered (file hash "
            "recorded anew) -> canonical content hash refused")
        cx.raises("V02_SOURCE_HASH_MISMATCH", lambda: RS.load_holdout_manifest(
                      ap_, RS.Identities(),
                      file_sha256=B["holdout"]["manifest_file_sha256"],
                      content_sha256=B["holdout"]["manifest_content_sha256"],
                      fixture=True), "holdout manifest file hash mismatch")
        # analysis + report with the addendum
        ho = B["path"].parent / "hold_analysis.json"
        hsha = A.run_analysis(B["layout"], *B["hconfig"], ho,
                              holdout=B["holdout"], selection=B["selection"])
        hd = json.loads(ho.read_bytes())
        hb = hd["holdout"]
        cx.ok(hb["selection_sha256"] == B["selection"][1] and
              len(hb["records"]) == 12 and
              hb["evaluation_scope"] == S.HOLDOUT_SCOPE_TEXT and
              S.HOLDOUT_STATEMENT in hd["analysis"]["notes"],
              "holdout analysis records the selection, records, masks, "
              "AM4-B rule, evaluation scope and the not-independent statement")
        vd = json.loads(Path(B["path"]).read_bytes())
        cx.ok(hb["mask_seeds_sha256"] == B["msha"] and
              hb["masks_sha256"] == hm.hexdigest() and
              vd["statements"][0] == "Validation only; the holdout is not "
              "read by this stage." and "RESULT RECORDS" in
              hd["statements"][0] and "never reads MRI data" in
              hd["statements"][0] and not any(
                  "holdout is not read" in x for x in hd["statements"]),
              "statements conditional: validation-only text without holdout; "
              "with holdout, holdout result records read, MRI data never; "
              "frozen and recomputed mask identities recorded")
        R.build_report(ho, hsha, td / "hrep")
        md = (td / "hrep" / R.REPORT_MD).read_text()
        rj = R.verify_report(td / "hrep", hsha)
        cx.ok(md.index("NOT AN INDEPENDENT CONFIRMATION") < md.index(
            "## Statements") and S.HOLDOUT_NMSE_RULE in md and
              S.HOLDOUT_SCOPE_TEXT in md and
              len(rj["content"]["holdout"]["comparisons"]) == 21 and
              all(c["id"].replace("|", "\\|") in md
                  for c in rj["content"]["holdout"]["comparisons"]),
              "report: holdout section first, statement, AM4-B rule, "
              "evaluation scope and all 21 comparisons")
        # selection bound to the validation configuration
        other = td / "other_cfg.json"
        ocfg = A.load_config(B["config"], B["config_sha"])
        ocfg["decisions"] = dict(ocfg["decisions"], D9="x")
        osha = S.write_json_once(other, ocfg)
        sel2 = dict(B["sel"], validation=dict(B["sel"]["validation"],
                                              config_sha256=osha))
        sp2 = td / "sel2.json"
        ssha2 = S.write_json_once(sp2, sel2)
        cx.raises("SELECTION_INVALID", lambda: A.run_analysis(
            B["layout"], *B["hconfig"], td / "h2.json", holdout=B["holdout"],
            selection=(sp2, ssha2)), "selection manifest of another "
            "validation configuration refused")
        cx.raises("STATS_HOLDOUT_MISMATCH", lambda: A.run_analysis(
            B["layout"], B["config"], B["config_sha"], td / "h3.json",
            holdout=B["holdout"], selection=B["selection"]),
            "holdout records with the validation-only configuration refused")
        cx.record("HOL holdout analysis sha256 (synthetic, informational)",
                  hsha)


def f18a_group(cx: Ctx) -> None:
    """F18A: selection manifest (195 §9.3 step 1) and the analysis-chain
    evidence manifest -- approved hash required, every file re-hashed,
    bindings checked; joint alteration refused."""
    import shutil
    from seqref_mri.scripts import v05lt_analysis as A
    from seqref_mri.scripts import v05lt_evidence_check as EVC
    from seqref_mri.scripts import v05lt_stats as S
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        B = _b3_setup(td)
        sel, root = B["sel"], B["root"]
        ok_files = all((root / ck["file"]).is_file() and C.sha256_file(
            root / ck["file"]) == ck["sha256"] for run in sel["runs"].values()
            for ck in run.values())
        kinds = {ck["kind"] for run in sel["runs"].values()
                 for ck in run.values()}
        cx.ok(len(sel["runs"]) == 6 and all(set(r) == {"epoch3", "selected",
                                                      "final"}
                                           for r in sel["runs"].values()) and
              set(sel["epoch0"]) == {"2026", "2027", "2028"} and ok_files and
              sel["evaluation_scope"] == S.HOLDOUT_SCOPE_TEXT and
              all(run["final"]["kind"] == "final_weights" and
                  run["final"]["file"].endswith("/final/weights_final.pt") and
                  run["epoch3"]["kind"] == run["selected"]["kind"] ==
                  "weights" for run in sel["runs"].values()) and
              sel["validation"]["analysis_sha256"] == B["sha"],
              "selection manifest: epoch-3, selected, final per run, "
              "epoch-0 per seed, validation identities, explicit holdout "
              "scope; c2: final = the published final-weights file "
              "(PA5-B.3), no resume file", detail=f"kinds {sorted(kinds)}")
        cx.raises("ARTIFACT_EXISTS", lambda: A.write_selection_manifest(
            B["layout"], root, B["selection"][0],
            config=(B["config"], B["config_sha"]),
            analysis=(B["path"], B["sha"]), report_dir=B["report"]),
            "selection manifest write-once")
        cx.raises("STATS_PRODUCTION_DISABLED", lambda:
                  A.write_selection_manifest(
                      B["layout"], root, C.Paths().v05lt / "sel.json",
                      config=(B["config"], B["config_sha"]),
                      analysis=(B["path"], B["sha"]), report_dir=B["report"]),
                  "registered results tree refused without production")
        cx.raises("SELECTION_ANALYSIS", lambda: A.selection_doc(
            B["layout"], root, config=(B["config"], B["config_sha"]),
            analysis=(B["path"], "0" * 64), report_dir=B["report"]),
            "selection refused for an unapproved analysis hash")

        def rel(p):
            return str(Path(p).relative_to(root))
        roles = {"config": rel(B["config"]),
                 "difficult": rel(B["layout"]["difficult"]),
                 "analysis": rel(B["path"]), "report_dir": rel(B["report"]),
                 "selection": rel(B["selection"][0]),
                 "holdout_config": rel(B["hconfig"][0])}
        mp = td / "chain.json"
        msha = EVC.make_chain_manifest(mp, root, roles)
        res = EVC.check_chain(mp, msha, root)
        cx.ok(res["entries"] == 22 and res["analysis_sha256"] == B["sha"],
              "chain manifest approved -> every file re-hashed and every "
              "binding verified (config, difficult, analysis, 17 report "
              "files incl. SHA256SUMS, selection, holdout configuration)",
              detail=f"{res['entries']} entries")
        cx.ok(EVC.main(["chain", "--manifest", str(mp), "--approved-sha256",
                        msha, "--root", str(root)]) == 0,
              "evidence checker CLI mode chain: PASS (exit 0)")
        cx.raises("EVIDENCE_MANIFEST_NOT_APPROVED", lambda: EVC.check_chain(
            mp, "1" * 64, root), "manifest not approved -> refused")
        # joint alteration: change a report file AND regenerate the manifest
        r2 = td / "r2"
        shutil.copytree(root, r2, symlinks=True)
        (r2 / roles["report_dir"] / "report.md").write_text("altered\n")
        mp2 = td / "chain2.json"
        EVC.make_chain_manifest(mp2, r2, roles)
        cx.raises("EVIDENCE_MANIFEST_NOT_APPROVED", lambda: EVC.check_chain(
            mp2, msha, r2), "file and manifest altered together, manifest "
            "internally consistent -> refused (not the approved hash)")
        cx.raises("EVIDENCE_HASH_MISMATCH", lambda: EVC.check_chain(
            mp, msha, r2), "altered file under the approved manifest -> "
            "EVIDENCE_HASH_MISMATCH")
        r3 = td / "r3"
        shutil.copytree(root, r3, symlinks=True)
        (r3 / roles["report_dir"] / "extra.txt").write_text("x")
        cx.raises("EVIDENCE_CHAIN_BINDING", lambda: EVC.check_chain(
            mp, msha, r3), "extra file in the report directory -> refused")
        r4 = td / "r4"
        shutil.copytree(root, r4, symlinks=True)
        (r4 / roles["selection"]).unlink()
        cx.raises("EVIDENCE_MISSING", lambda: EVC.check_chain(mp, msha, r4),
                  "selection manifest missing -> refused")
        r5 = td / "r5"
        shutil.copytree(root, r5, symlinks=True)
        ck0 = next(iter(next(iter(sel["runs"].values())).values()))
        f = r5 / ck0["file"]
        os.chmod(f, 0o644)
        f.write_bytes(f.read_bytes() + b"x")
        cx.raises("EVIDENCE_CHAIN_BINDING", lambda: EVC.check_chain(
            mp, msha, r5), "checkpoint listed in the selection manifest "
            "altered -> refused")
        # internally consistent but semantically wrong chain (approved)
        other = root / "camp" / "analysis" / "other_cfg.json"
        ocfg = A.load_config(B["config"], B["config_sha"])
        ocfg["decisions"] = dict(ocfg["decisions"], D9="x")
        S.write_json_once(other, ocfg)
        mp6 = td / "chain6.json"
        msha6 = EVC.make_chain_manifest(mp6, root, dict(roles,
                                                        config=rel(other)))
        cx.raises("EVIDENCE_CHAIN_BINDING", lambda: EVC.check_chain(
            mp6, msha6, root), "approved manifest whose analysis is not bound "
            "to its configuration -> EVIDENCE_CHAIN_BINDING")
        hc2 = root / "camp" / "analysis" / "hold_other.json"
        S.write_json_once(hc2, S.build_holdout_config(
            ocfg, [{"volume_id": "a.h5", "file": "v/a.h5", "slice_index": 1}],
            manifest_content_sha256="x",
            mask_seeds_sha256="f" * 64))
        mp7 = td / "chain7.json"
        msha7 = EVC.make_chain_manifest(mp7, root, dict(
            roles, holdout_config=rel(hc2)))
        cx.raises("EVIDENCE_CHAIN_BINDING", lambda: EVC.check_chain(
            mp7, msha7, root), "holdout configuration extending another "
            "validation configuration -> refused")
        cx.raises("EVIDENCE_CHAIN_ROLES", lambda: EVC.make_chain_manifest(
            td / "c8.json", root, {k: v for k, v in roles.items()
                                   if k != "selection"}),
            "chain without the selection manifest refused")
        # source identities (rev 2): original approved manifest, not regenerated
        an = json.loads(Path(B["path"]).read_bytes())
        ents = {str(root / e["path"]) for e in json.loads(
            mp.read_bytes())["entries"]}
        src = sorted(k for k in an["inputs"]["identities"] if k not in ents)
        cx.ok(len(src) > 0, "analysis records source identities outside "
              "the chain entries", detail=f"{len(src)} sources")
        sf = Path(src[0])
        orig, mode = sf.read_bytes(), sf.stat().st_mode
        os.chmod(sf, 0o644)
        try:
            sf.write_bytes(orig + b" ")
            cx.raises("EVIDENCE_SOURCE_MISMATCH", lambda: EVC.check_chain(
                mp, msha, root), "analysis source file altered (approved "
                "manifest unchanged) -> EVIDENCE_SOURCE_MISMATCH")
        finally:
            sf.write_bytes(orig)
            os.chmod(sf, mode)
        moved = sf.with_name(sf.name + ".moved")
        sf.rename(moved)
        try:
            cx.raises("EVIDENCE_SOURCE_MISSING", lambda: EVC.check_chain(
                mp, msha, root), "analysis source file deleted (approved "
                "manifest unchanged) -> EVIDENCE_SOURCE_MISSING")
        finally:
            moved.rename(sf)
        cx.ok(EVC.check_chain(mp, msha, root)["entries"] == 22,
              "source restored -> the original approved chain passes again")
        # selection completeness (rev 2): otherwise valid hashes
        import copy

        def mut(f):
            d = copy.deepcopy(sel)
            f(d)
            return d
        k1 = "lam2|2027"
        fin_kind = sel["runs"][k1]["final"]["kind"]
        scases = [
            ("empty runs and epoch-0", "SELECTION_INCOMPLETE",
             mut(lambda d: (d["runs"].clear(), d["epoch0"].clear()))),
            ("empty runs", "SELECTION_INCOMPLETE",
             mut(lambda d: d["runs"].clear())),
            ("empty epoch-0", "SELECTION_INCOMPLETE",
             mut(lambda d: d["epoch0"].clear())),
            ("one run missing", "SELECTION_INCOMPLETE",
             mut(lambda d: d["runs"].pop(k1))),
            ("extra run", "SELECTION_INCOMPLETE",
             mut(lambda d: d["runs"].update({"lam1|2026": d["runs"][k1]}))),
            ("final entry missing", "SELECTION_INCOMPLETE",
             mut(lambda d: d["runs"][k1].pop("final"))),
            ("epoch-3 entry missing", "SELECTION_INCOMPLETE",
             mut(lambda d: d["runs"][k1].pop("epoch3"))),
            ("one epoch-0 seed missing", "SELECTION_INCOMPLETE",
             mut(lambda d: d["epoch0"].pop("2027"))),
            ("checkpoint sha256 not hex-64", "SELECTION_INVALID",
             mut(lambda d: d["runs"][k1]["selected"].update(sha256="xyz"))),
            ("model state sha256 not hex-64", "SELECTION_INVALID",
             mut(lambda d: d["runs"][k1]["epoch3"].update(
                 model_state_sha256=""))),
            ("epoch-0 sha256 not hex-64", "SELECTION_INVALID",
             mut(lambda d: d["epoch0"]["2026"].update(sha256="0" * 63))),
            ("resume kind on selected", "SELECTION_INVALID",
             mut(lambda d: d["runs"][k1]["selected"].update(kind="resume"))),
            ("c2: resume kind on final (resume files refused)",
             "SELECTION_INVALID",
             mut(lambda d: d["runs"][k1]["final"].update(kind="resume"))),
            ("c2: resume final with the former b3 label", "SELECTION_INVALID",
             mut(lambda d: d["runs"][k1]["final"].update(
                 kind="resume", note="final state held in the last "
                 "commit's resume file"))),
            ("c2: final entry of kind weights", "SELECTION_INVALID",
             mut(lambda d: d["runs"][k1]["final"].update(kind="weights"))),
            ("c2: final_weights kind on selected", "SELECTION_INVALID",
             mut(lambda d: d["runs"][k1]["selected"].update(
                 kind="final_weights"))),
            ("epoch-3 entry with epoch 4", "SELECTION_INVALID",
             mut(lambda d: d["runs"][k1]["epoch3"].update(epoch=4))),
            ("epoch as string", "SELECTION_INVALID",
             mut(lambda d: d["runs"][k1]["selected"].update(epoch="5"))),
            ("unsafe file path", "SELECTION_INVALID",
             mut(lambda d: d["runs"][k1]["final"].update(file="../x.pt"))),
            ("validation identity not hex-64", "SELECTION_INVALID",
             mut(lambda d: d["validation"].update(analysis_sha256="")))]
        for lab, code, d in scases:
            cx.raises(code, lambda d=d: A.validate_selection(d),
                      f"selection manifest: {lab} -> {code}")
        cx.ok(A.validate_selection(copy.deepcopy(sel)) == sel,
              "complete selection manifest validates",
              detail=f"final kind {fin_kind}")
        es = root / "camp" / "analysis" / "sel_empty.json"
        essha = S.write_json_once(es, mut(lambda d: (d["runs"].clear(),
                                                     d["epoch0"].clear())))
        cx.raises("SELECTION_INCOMPLETE", lambda: A.load_selection(es, essha),
                  "empty selection manifest with its own approved hash -> "
                  "load_selection refuses")
        mp9 = td / "chain9.json"
        msha9 = EVC.make_chain_manifest(mp9, root, dict(roles,
                                                        selection=rel(es)))
        cx.raises("SELECTION_INCOMPLETE", lambda: EVC.check_chain(
            mp9, msha9, root), "approved chain with an empty selection "
            "manifest -> refused")
        lc = an["descriptive"]["learning_curves"]["runs"][k1]
        ep = mut(lambda d: d["runs"][k1]["selected"].update(
            epoch=3 if lc["selected_epoch"] != 3 else 4))
        epp = root / "camp" / "analysis" / "sel_epoch.json"
        epsha = S.write_json_once(epp, ep)
        mp10 = td / "chain10.json"
        msha10 = EVC.make_chain_manifest(mp10, root, dict(roles,
                                                          selection=rel(epp)))
        cx.raises("EVIDENCE_CHAIN_BINDING", lambda: EVC.check_chain(
            mp10, msha10, root), "approved chain whose selected epoch differs "
            f"from the analysis learning curves -> refused ({epsha[:12]})")
        # -- c2: final weights bound into selection and evidence chain -----
        fwf = root / sel["runs"][k1]["final"]["file"]
        osize = fwf.stat().st_size
        with open(fwf, "ab") as fh:
            fh.write(b"X")
        cx.raises("EVIDENCE_CHAIN_BINDING", lambda: EVC.check_chain(
            mp, msha, root), "c2: final-weights file altered under the "
            "approved chain -> refused")
        cx.raises("SELECTION_CHECKPOINT", lambda: A.selection_doc(
            B["layout"], root, config=(B["config"], B["config_sha"]),
            analysis=(B["path"], B["sha"]), report_dir=B["report"]),
            "c2: selection manifest refused when a run's final weights do "
            "not verify")
        os.truncate(fwf, osize)
        rdir = Path(B["layout"]["runs"][k1])
        hide = rdir.parent / "final_hidden"
        os.rename(rdir / "final", hide)
        cx.raises("SELECTION_CHECKPOINT", lambda: A.selection_doc(
            B["layout"], root, config=(B["config"], B["config_sha"]),
            analysis=(B["path"], B["sha"]), report_dir=B["report"]),
            "c2: selection manifest refused when a run has no published "
            "final weights (no resume substitute)")
        os.rename(hide, rdir / "final")
        cx.ok(EVC.check_chain(mp, msha, root)["entries"] > 0,
              "c2: restored final weights verify again under the chain")
        cx.record("F18A chain manifest sha256 (synthetic, informational)",
                  msha)


# ---------------------------------------------------------------------------
# I1.9 c1: launcher (SEQREF-MRI-V05LTI19 v0.1)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# I1.9 c2: final weights (FWT, FWR) and data-content manifest (DHS)
# ---------------------------------------------------------------------------
def _fw_copy(src: Path, dst: Path) -> Path:
    """Independent copy of a stub run directory (fixture only)."""
    shutil.copytree(src, dst)
    return dst


def _fw_rewrite(d: Path, name: str, data: bytes) -> None:
    """Replace a published file and its sidecar consistently (fixture-only
    tamper: only the property under test is wrong)."""
    f = d / name
    os.chmod(f, 0o644)
    f.write_bytes(data)
    (d / (name + ".sha256")).write_text(
        f"{C.sha256_bytes(data)}  {name}\n")


def fwt_group(cx: Ctx) -> None:
    """FWT (PA5-B, synthetic by design): final-weights extraction from the
    last committed resume file of completed stub runs (real trainer path);
    write-once publication, recovery after interruption, refusals."""
    import torch
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_final as FW
    from seqref_mri.scripts import v05lt_train as T
    from seqref_mri.scripts.v05lt_stub_train import build_stub
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data = _syn_train_data(td)
        base = td / "base"
        cfg = _stub_cfg(base, seed=2027, arm="lam2", max_epochs=4)
        _stub_run(cfg, data).start()
        pristine = _fw_copy(base, td / "pristine")
        last = [r for r in C.Ledger(base / "ledger.jsonl").records()
                if r["kind"] == "commit"][-1]["body"]
        fw = FW.extract_final(base, fixture=True)
        dest = base / FW.FINAL_DIR
        rec = json.loads((dest / FW.RECORD).read_bytes())
        cx.ok(sorted(p.name for p in dest.iterdir()) == sorted(FW.FILES) and
              fw["epoch"] == last["epoch"] == 4 and fw["step"] ==
              last["step"] and fw["model_state_sha256"] ==
              last["model_state_sha256"] and rec["source"]["resume_sha256"]
              == last["retained"]["resume"]["sha256"] and
              rec["weights_sha256"] == fw["sha256"] == C.sha256_file(
                  dest / FW.WEIGHTS) and not (base / FW.STAGING).exists(),
              "final weights published from the hash-verified last "
              "committed resume file: epoch, step and model state = the "
              "commit; record bound to the resume file; no staging left",
              detail=f"epoch {fw['epoch']}")
        obj = torch.load(dest / FW.WEIGHTS, map_location="cpu",
                         weights_only=True)
        cx.ok(set(obj) == {"schema", "model", "meta"} and obj["schema"] ==
              T.WEIGHTS_SCHEMA and obj["meta"] == {
                  "seed": 2027, "arm": "lam2", "lambda": T.ARMS["lam2"],
                  "epoch": 4, "step": last["step"],
                  "model_state_sha256": last["model_state_sha256"],
                  "roles": ["final"]}, "trainer weights schema with meta "
              "{seed, arm, lambda, epoch, step, model state, roles [final]}")
        del obj
        model, ck = EV.load_trained_checkpoint(
            dest / FW.WEIGHTS, fw["sha256"], fw["model_state_sha256"],
            seed=2027, arm="lam2", epoch=4, model_factory=build_stub)
        cx.ok(ck.roles == ("final",) and ck.epoch == 4 and
              C.model_state_sha256(model) == last["model_state_sha256"],
              "regression: accepted checkpoint loader (load_trained_"
              "checkpoint) reads the final weights unchanged")
        before = _tree_sha(base)
        again = FW.extract_final(base, fixture=True)
        cx.ok(again == fw and _tree_sha(base) == before,
              "second extraction of a valid publication only verifies "
              "(idempotent, nothing rewritten)")
        cx.raises("FINAL_RUN_NOT_COMPLETED", lambda: FW.verify_final(
            base, fixture=False), "fixture status refused outside fixture "
            "mode (production accepts stopped / budget exhausted only)")
        # -- interrupted publication: no final dir; retry is safe ----------
        for stage in ("final:weights_written", "final:staged"):
            d = _fw_copy(pristine, td / f"int_{stage.split(':')[1]}")
            try:
                FW.extract_final(d, fixture=True, hook=_lch_kill(stage))
                killed = False
            except _SimulatedKill:
                killed = True
            cx.ok(killed and not os.path.lexists(d / FW.FINAL_DIR) and
                  (d / FW.STAGING).is_dir(), f"interrupted at {stage}: no "
                  "final directory; staging left")
            r = FW.extract_final(d, fixture=True)
            q = list((d / "_incomplete").glob("final_staging.*"))
            cx.ok(r["sha256"] == fw["sha256"] and len(q) == 1 and
                  not (d / FW.STAGING).exists(), f"retry after {stage}: "
                  "stale staging quarantined (not deleted); identical final "
                  "weights published")
        # -- race: destination appears during staging ----------------------
        d = _fw_copy(pristine, td / "race")

        def mk(stage, d=d):
            if stage == "final:staged":
                (d / FW.FINAL_DIR).mkdir()
        cx.raises("ARTIFACT_EXISTS", lambda: FW.extract_final(
            d, fixture=True, hook=mk), "empty final directory created "
            "during staging -> RENAME_NOREPLACE refuses, nothing replaced")
        cx.ok(list((d / FW.FINAL_DIR).iterdir()) == [] and
              (d / FW.STAGING).is_dir(), "destination left empty, staging "
              "kept for inspection")
        cx.raises("FINAL_INCOMPLETE", lambda: FW.extract_final(
            d, fixture=True), "retry with an existing empty final "
            "directory -> refused (never overwritten)")
        # -- existing mismatched output refused, never overwritten ---------
        other = build_stub(2026)
        wrong = {"schema": T.WEIGHTS_SCHEMA, "model": other.state_dict(),
                 "meta": dict(json.loads(json.dumps(FW.source(
                     base, fixture=True)["meta"])))}
        import io
        buf = io.BytesIO()
        torch.save(wrong, buf)

        def rec_edit(field, val):
            def f(dd):
                r = json.loads((dd / FW.RECORD).read_bytes())
                r[field] = val
                _fw_rewrite(dd, FW.RECORD, C.canonical_json(r))
            return f

        def wrong_weights(dd):
            _fw_rewrite(dd, FW.WEIGHTS, buf.getvalue())
            r = json.loads((dd / FW.RECORD).read_bytes())
            r["weights_sha256"] = C.sha256_bytes(buf.getvalue())
            _fw_rewrite(dd, FW.RECORD, C.canonical_json(r))

        def meta_roles(dd):
            o = torch.load(dd / FW.WEIGHTS, map_location="cpu",
                           weights_only=True)
            o["meta"]["roles"] = ["final", "best_psnr"]
            b2 = io.BytesIO()
            torch.save(o, b2)
            _fw_rewrite(dd, FW.WEIGHTS, b2.getvalue())
            r = json.loads((dd / FW.RECORD).read_bytes())
            r["weights_sha256"] = C.sha256_bytes(b2.getvalue())
            _fw_rewrite(dd, FW.RECORD, C.canonical_json(r))

        def append_byte(dd):
            f = dd / FW.WEIGHTS
            os.chmod(f, 0o644)
            with open(f, "ab") as fh:
                fh.write(b"X")
        cases = [
            ("weights bytes altered (sidecar unchanged)",
             "FINAL_WEIGHTS_MISMATCH", append_byte),
            ("weights of another model state (sidecar and record updated)",
             "FINAL_WEIGHTS_MISMATCH", wrong_weights),
            ("weights meta roles altered (hashes updated)",
             "FINAL_WEIGHTS_MISMATCH", meta_roles),
            ("record epoch altered (sidecar updated)",
             "FINAL_WEIGHTS_MISMATCH", rec_edit("epoch", 3)),
            ("record bound to another commit line",
             "FINAL_WEIGHTS_MISMATCH", rec_edit("source", {
                 **rec["source"], "commit_line_sha256": "0" * 64})),
            ("record extra field", "FINAL_WEIGHTS_MISMATCH",
             rec_edit("note", "x")),
            ("extra file in final/", "FINAL_INCOMPLETE",
             lambda dd: (dd / "extra.txt").write_text("x")),
            ("record missing", "FINAL_INCOMPLETE",
             lambda dd: (os.chmod(dd, 0o755), (dd / FW.RECORD).unlink()))]
        for lab, code, edit in cases:
            d = _fw_copy(base, td / f"mm_{abs(hash(lab)) % 10**8}")
            edit(d / FW.FINAL_DIR)
            snap = _tree_sha(d / FW.FINAL_DIR)
            cx.raises(code, lambda d=d: FW.verify_final(d, fixture=True),
                      f"verify: {lab} -> {code}")
            cx.raises(code, lambda d=d: FW.extract_final(d, fixture=True),
                      f"extract on existing output: {lab} -> {code}")
            cx.ok(_tree_sha(d / FW.FINAL_DIR) == snap,
                  f"{lab}: mismatched output not overwritten")
        # -- rev 2: operational failures are typed ---------------------------
        def to_dir(p):
            os.chmod(p.parent, 0o755)
            p.unlink()
            p.mkdir()

        def bad_utf8(p):
            os.chmod(p, 0o644)
            p.write_bytes(b"\xff\xfe\x00")

        def dangling(p):
            p.unlink()
            os.symlink(p.parent / "nowhere", p)

        def wrong_name(p):
            os.chmod(p, 0o644)
            p.write_text(p.read_text().split()[0] + "  other.pt\n")

        def valid_link(p):
            keep = p.parent.parent / ("kept_" + p.name)
            shutil.copy(p, keep)
            p.unlink()
            os.symlink(keep, p)
        for lab, rel, edit, code in (
                ("weights sidecar is a directory", FW.WEIGHTS + ".sha256",
                 to_dir, "FINAL_SIDECAR_INVALID"),
                ("record sidecar invalid UTF-8", FW.RECORD + ".sha256",
                 bad_utf8, "FINAL_SIDECAR_INVALID"),
                ("weights sidecar dangling symlink", FW.WEIGHTS + ".sha256",
                 dangling, "FINAL_SIDECAR_INVALID"),
                ("record sidecar names another file", FW.RECORD + ".sha256",
                 wrong_name, "FINAL_SIDECAR_INVALID"),
                ("weights sidecar symlink to a valid copy",
                 FW.WEIGHTS + ".sha256", valid_link,
                 "FINAL_SIDECAR_INVALID")):
            d = _fw_copy(base, td / f"io_{abs(hash(lab)) % 10**8}")
            edit(d / FW.FINAL_DIR / rel)
            _raises_typed(cx, code, lambda d=d: FW.verify_final(
                d, fixture=True), f"verify: {lab} -> {code}")
            _raises_typed(cx, code, lambda d=d: FW.extract_final(
                d, fixture=True), f"extract: {lab} -> {code}")
        d = _fw_copy(base, td / "io_rec_utf8")
        rp = d / FW.FINAL_DIR / FW.RECORD
        _fw_rewrite(d / FW.FINAL_DIR, FW.RECORD, b"\xff\xfe{}")
        _raises_typed(cx, "FINAL_WEIGHTS_MISMATCH", lambda: FW.verify_final(
            d, fixture=True), "record with invalid UTF-8 (sidecar matching) "
            "-> typed refusal")
        for lab, edit in (("resume sidecar is a directory", to_dir),
                          ("resume sidecar invalid UTF-8", bad_utf8)):
            d = _fw_copy(pristine, td / f"rs_{abs(hash(lab)) % 10**8}")
            edit(d / "ckpt" / (last["retained"]["resume"]["name"] + ".sha256"))
            _raises_typed(cx, "FINAL_RESUME_SIDECAR", lambda d=d:
                          FW.extract_final(d, fixture=True),
                          f"{lab} -> FINAL_RESUME_SIDECAR")
        for stage in ("final:staging_open", "final:weights_written"):
            d = _fw_copy(pristine, td / f"wf_{stage.split(':')[1]}")

            def full(st_, at=stage):
                if st_ == at:
                    raise OSError(28, "fixture: no space left on device")
            _raises_typed(cx, "FINAL_WRITE_FAILED", lambda d=d, h=full:
                          FW.extract_final(d, fixture=True, hook=h),
                          f"write failure at {stage} -> FINAL_WRITE_FAILED")
            cx.ok(not os.path.lexists(d / FW.FINAL_DIR) and
                  FW.extract_final(d, fixture=True)["sha256"] == fw["sha256"],
                  f"after the write failure at {stage}: no final directory; "
                  "the retry publishes identical final weights")
        d = _fw_copy(pristine, td / "q_fail")
        (d / FW.STAGING).mkdir()
        (d / "_incomplete").write_text("not a directory")
        _raises_typed(cx, "FINAL_WRITE_FAILED", lambda: FW.extract_final(
            d, fixture=True), "quarantine of stale staging fails -> "
            "FINAL_WRITE_FAILED; nothing published")
        cx.ok(not os.path.lexists(d / FW.FINAL_DIR) and (d / FW.STAGING)
              .is_dir(), "stale staging kept after the failed quarantine")
        # -- source refusals -----------------------------------------------
        d = _fw_copy(pristine, td / "rtamper")
        rf = d / "ckpt" / last["retained"]["resume"]["name"]
        os.chmod(rf, 0o644)
        with open(rf, "ab") as fh:
            fh.write(b"X")
        cx.raises("FINAL_RESUME_HASH", lambda: FW.extract_final(
            d, fixture=True), "resume file altered -> refused")
        cx.ok(not os.path.lexists(d / FW.FINAL_DIR) and not os.path.lexists(
            d / FW.STAGING), "nothing published from an unverified resume")
        d = _fw_copy(pristine, td / "rtamper2")
        rf = d / "ckpt" / last["retained"]["resume"]["name"]
        os.chmod(rf, 0o644)
        with open(rf, "ab") as fh:
            fh.write(b"X")
        sf = Path(str(rf) + ".sha256")
        os.chmod(sf, 0o644)
        sf.write_text(f"{C.sha256_file(rf)}  {rf.name}\n")
        cx.raises("FINAL_RESUME_HASH", lambda: FW.extract_final(
            d, fixture=True), "resume file and its sidecar altered "
            "consistently -> refused against the ledger commit hash")
        sf.write_text("4" * 64 + f"  {rf.name}\n")
        os.truncate(rf, rf.stat().st_size - 1)
        cx.raises("FINAL_RESUME_SIDECAR", lambda: FW.extract_final(
            d, fixture=True), "resume sidecar altered alone -> refused")
        orig = T.load_resume
        for lab, code, edit in (
                ("resume step != commit (fixture patch)", "FINAL_RESUME_STATE",
                 lambda o: o["state"].update(step=o["state"]["step"] + 1)),
                ("resume model != commit (fixture patch)", "FINAL_MODEL_STATE",
                 lambda o: o["model"].update({k: v + 1.0 for k, v in list(
                     o["model"].items())[:1]}))):
            d = _fw_copy(pristine, td / f"patch_{code}")

            def patched(path, edit=edit):
                o = orig(path)
                edit(o)
                return o
            T.load_resume = patched
            try:
                cx.raises(code, lambda d=d: FW.extract_final(d, fixture=True),
                          f"{lab} -> {code}")
            finally:
                T.load_resume = orig
            cx.ok(not os.path.lexists(d / FW.FINAL_DIR),
                  f"{lab}: nothing published")
        # resume-state check is reached only via the patched loader
        cx.ok(T.load_resume is orig, "fixture patch removed")
        ki = td / "killed"
        cfg_k = _stub_cfg(ki, seed=2026, arm="lam0", max_epochs=4)
        try:
            _stub_run(cfg_k, data, _killer("train:step:3")).start()
        except _SimulatedKill:
            logger.info("fixture: run interrupted as intended")
        cx.raises("FINAL_RUN_NOT_COMPLETED", lambda: FW.extract_final(
            ki, fixture=True), "interrupted run (no run_end) refused")
        tm = td / "terminal"
        cfg_t = _stub_cfg(tm, seed=2026, arm="lam0", max_epochs=4)
        try:
            _stub_run(cfg_t, data, poison_after=2).start()
        except C.V05LTError as exc:
            logger.info("fixture: terminal run as intended (%s)", exc.code)
        cx.ok(_ends(tm) and _ends(tm)[-1]["status"] == "numerical_failure",
              "terminal fixture run recorded numerical_failure")
        cx.raises("FINAL_RUN_NOT_COMPLETED", lambda: FW.extract_final(
            tm, fixture=True), "terminal run refused")
        cx.raises("FINAL_SOURCE_INVALID", lambda: FW.extract_final(
            td / "nowhere", fixture=True), "directory without a ledger "
            "refused")


def fwr(cx: Ctx) -> None:
    """FWR (PA5-B option (a) integration regression): final weights from
    the REAL-architecture reference run (registered fixture config
    lam2_s2026), loaded by the accepted checkpoint loader and post-run
    evaluated in a fresh process."""
    _rt_phase(cx)
    ref = _rt_ref(cx)
    d = _RT["root"] / "fwr_run"
    shutil.copytree(ref, d, copy_function=os.link)   # hard links, no copy
    with tempfile.TemporaryDirectory() as t:
        res = _run_eval_child(cx, Path(t), {"kind": "fwr", "run": str(d)},
                              "fwr")
    e = res["fwr"]
    cx.ok(e["published"] and e["verified"] and e["loaded"] and
          e["roles"] == ["final"] and e["epoch"] == RT_EPOCHS and
          e["state_eq_commit"] and e["posteval_written"] and
          e["decodes"] == 128 * e["n_mid"], "real-architecture final weights: "
          "extracted from the committed resume file, verified, loaded by "
          "load_trained_checkpoint, post-run evaluated (primary bank) and "
          "the record round-trips", "FWR_FAILED", str(e))
    cx.record("FWR timing (fixture subset; P4 input only)", e["timing"])
    _rt_drop("fwr_run")


def _fwr_child(EV, run_dir: Path, host: bool, data_root) -> dict:
    import time as _time
    from seqref_mri.scripts import v05lt_final as FW
    t0 = _time.monotonic()
    fw = FW.extract_final(run_dir, fixture=True)
    t1 = _time.monotonic()
    v = FW.verify_final(run_dir, fixture=True)
    src = FW.source(run_dir, fixture=True)
    model, ck = EV.load_trained_checkpoint(
        fw["path"], fw["sha256"], fw["model_state_sha256"], seed=fw["seed"],
        arm=fw["arm"], epoch=fw["epoch"])
    with tempfile.TemporaryDirectory() as t:
        data = _rt_data(host, REPO_ROOT, data_root, Path(t))
        vi = _rt_indices(host)[1]
        mid = vi[:2]                 # fixture subset (not midpoints)
        cases = [{"case": j, "file": data["vv"].entry(i)[0],
                  "slice_index": data["vv"].entry(i)[1]}
                 for j, i in enumerate(mid)]
        tol = EV.calibrate_dc(EV.build_states(
            data["vv"], mid, data["p4"], data["s_ref"]))["tolerances"]
        ms = fw["model_state_sha256"]
        plan = EV.plan_campaign([{"seed": fw["seed"], "arm": fw["arm"],
                                  "roles": {
            "epoch0": {"epoch": 0, "model_state": "a" * 64},
            "epoch3": {"epoch": fw["epoch"], "model_state": ms},
            "selected": {"epoch": 1, "model_state": "b" * 64},
            "final": {"epoch": fw["epoch"], "model_state": ms}}}])
        task = [x for x in plan["tasks"] if x["model_state"] == ms][0]
        cnt = {}
        t2 = _time.monotonic()
        rec, pan = EV.posteval_evaluate(
            model, data, task, {b: bk for b, bk in _pev_banks().items()
                                if b in task["banks"]},
            tol, cases, checkpoint=ck, eligibility_sha256="fixture",
            mid_indices=mid, decode_counter=cnt)
        t3 = _time.monotonic()
        out = Path(t) / "pe" / ms
        EV.write_posteval(out, rec, pan, {"fixture": True})
        back = EV.load_posteval(out)
    return {"published": fw["sha256"] == v["sha256"], "verified": True,
            "loaded": ck.file_name == FW.WEIGHTS, "roles": list(ck.roles),
            "epoch": ck.epoch,
            "state_eq_commit": ms == src["commit"]["model_state_sha256"],
            "posteval_written": back["checkpoint"] == ck.as_record() and
            sorted(u["role"] for u in back["uses"]) == ["epoch3", "final"],
            "decodes": cnt.get("primary", 0), "n_mid": len(mid),
            "timing": {"extract_s": round(t1 - t0, 2),
                       "posteval_s": round(t3 - t2, 2)}}


def dhs_group(cx: Ctx) -> None:
    """DHS (PA5-A): data-content manifest generate / verify. Synthetic:
    full refusal set. Host: a TEMPORARY (unregistered) manifest of the real
    training + validation files, then one full verification; both timed
    (P4 input, PA5-A.3); nothing written under results/."""
    from seqref_mri.scripts import v05lt_datahash as DH
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root, mp, msha, ex = synth_split(td, SYN_VOLS, SYN_TRAIN, SYN_VAL)
        out = td / "dc.json"
        r = DH.generate(mp, msha, SYN_HOLDOUT, root, out, expect=ex)
        doc = json.loads(out.read_bytes())
        n = len(SYN_TRAIN) + len(SYN_VAL)
        cx.ok(r["n_files"] == n == doc["n_files"] and
              [x["file"] for x in doc["files"]] == sorted(
                  f"singlecoil_train/{v}" for v in SYN_VOLS) and
              all(x["sha256"] == C.sha256_file(root / x["file"]) and
                  x["size"] == (root / x["file"]).stat().st_size
                  for x in doc["files"]) and doc["split_manifest_sha256"] ==
              msha and set(doc) == DH._DOC_KEYS, "manifest lists sha256 and "
              "size of every training and validation file, bound to the split "
              "manifest; no timing inside the manifest")
        cx.ok(Path(str(out) + ".sha256").read_text().split()[0] == r["sha256"]
              and stat.S_IMODE(out.stat().st_mode) == 0o444,
              "sidecar written; manifest read-only (mode 0444)")
        r2 = DH.generate(mp, msha, SYN_HOLDOUT, root, td / "dc2.json",
                         expect=ex)
        cx.ok(r2["sha256"] == r["sha256"], "regeneration byte-identical")
        cx.raises("ARTIFACT_EXISTS", lambda: DH.generate(
            mp, msha, SYN_HOLDOUT, root, out, expect=ex), "write-once")
        cx.raises("DATAHASH_PRODUCTION_DISABLED", lambda: DH.generate(
            mp, msha, SYN_HOLDOUT, root, C.Paths().inputs / DH.OUT_NAME,
            expect=ex), "registered inputs path refused without the gate-2a "
            "production flag")
        v = DH.verify(out, r["sha256"], mp, msha, SYN_HOLDOUT, root,
                      expect=ex)
        cx.ok(v["n_files"] == n and v["seconds"] >= 0.0,
              "verify: full re-hash passes", detail=f"{v['seconds']:.3f} s")
        ver = (out, r["sha256"], mp, msha, SYN_HOLDOUT, root)
        cx.raises("DATA_CONTENT_UNAPPROVED", lambda: DH.verify(
            out, "0" * 64, mp, msha, SYN_HOLDOUT, root, expect=ex),
            "manifest not matching the approved sha256 refused")
        f0 = root / "singlecoil_train" / sorted(SYN_TRAIN)[0]
        raw = f0.read_bytes()

        def alter(fn, lab, code):
            fn()
            try:
                cx.raises(code, lambda: DH.verify(*ver, expect=ex), lab)
            finally:
                if f0.is_symlink() or not f0.exists():
                    if f0.is_symlink():
                        f0.unlink()
                    f0.write_bytes(raw)
                else:
                    os.chmod(f0, 0o644)
                    f0.write_bytes(raw)

        def flip():
            b = bytearray(raw)
            b[len(b) // 2] ^= 1
            f0.write_bytes(bytes(b))
        alter(flip, "one byte altered (same size) -> refused",
              "DATA_CONTENT_MISMATCH")
        alter(lambda: f0.write_bytes(raw + b"X"), "size changed -> refused",
              "DATA_CONTENT_MISMATCH")
        alter(lambda: f0.unlink(), "listed file missing on disk -> refused",
              "DATA_CONTENT_SET_MISMATCH")

        def to_link():
            keep = td / "keep.h5"
            keep.write_bytes(raw)
            f0.unlink()
            os.symlink(keep, f0)
        alter(to_link, "listed file replaced by a symlink -> refused",
              "DATA_CONTENT_NOT_REGULAR")
        extra = root / "singlecoil_train" / "file9100.h5"
        extra.write_bytes(b"x")
        cx.raises("DATA_CONTENT_SET_MISMATCH", lambda: DH.verify(
            *ver, expect=ex), "extra .h5 file on disk -> refused")
        extra.unlink()
        (root / "singlecoil_train" / "notes.txt").write_text("x")
        cx.ok(DH.verify(*ver, expect=ex)["n_files"] == n,
              "non-.h5 file is not data (the dataset globs *.h5)")
        (root / "singlecoil_train" / "notes.txt").unlink()

        def own(edit, name):
            d2 = json.loads(out.read_bytes())
            edit(d2)
            p2 = td / name
            sha2 = C.write_durable_new(p2, C.canonical_json(d2))
            C.write_durable_new(Path(str(p2) + ".sha256"),
                                f"{sha2}  {p2.name}\n".encode())
            return p2, sha2

        def drop_row(d2):
            d2["files"].pop(0)
            d2["n_files"] -= 1
            d2["total_bytes"] = sum(x["size"] for x in d2["files"])
        cases = [
            ("entry missing (own approved hash)", "DATA_CONTENT_SET_MISMATCH",
             drop_row),
            ("extra entry (own approved hash)", "DATA_CONTENT_SET_MISMATCH",
             lambda d2: (d2["files"].append(dict(d2["files"][0], file=
                         "singlecoil_train/file9100.h5", volume_id=
                         "file9100.h5")), d2.update(n_files=len(
                             d2["files"]), total_bytes=sum(
                             x["size"] for x in d2["files"])))),
            ("split label swapped", "DATA_CONTENT_SET_MISMATCH",
             lambda d2: d2["files"][0].update(split="validation" if
                                              d2["files"][0]["split"] ==
                                              "train" else "train")),
            ("bound to another split manifest", "DATA_CONTENT_SPLIT_MISMATCH",
             lambda d2: d2.update(split_manifest_sha256="1" * 64)),
            ("altered file sha256 entry", "DATA_CONTENT_MISMATCH",
             lambda d2: d2["files"][0].update(sha256="2" * 64)),
            ("counts inconsistent", "DATA_CONTENT_INVALID",
             lambda d2: d2.update(n_files=99)),
            ("row field missing", "DATA_CONTENT_INVALID",
             lambda d2: d2["files"][0].pop("size")),
            ("size as bool", "DATA_CONTENT_INVALID",
             lambda d2: d2["files"][0].update(size=True)),
            ("unsafe path", "DATA_CONTENT_INVALID",
             lambda d2: d2["files"][0].update(file="../x.h5")),
            ("extra top-level key", "DATA_CONTENT_INVALID",
             lambda d2: d2.update(timing=1.0)),
            ("wrong schema", "DATA_CONTENT_INVALID",
             lambda d2: d2.update(schema="x"))]
        for i, (lab, code, edit) in enumerate(cases):
            p2, sha2 = own(edit, f"own_{i}.json")
            cx.raises(code, lambda p2=p2, sha2=sha2: DH.verify(
                p2, sha2, mp, msha, SYN_HOLDOUT, root, expect=ex),
                f"{lab} -> {code}")
        sc = Path(str(out) + ".sha256")
        os.chmod(sc, 0o644)
        sc.write_text("3" * 64 + "  dc.json\n")
        cx.raises("DATA_CONTENT_SIDECAR", lambda: DH.verify(*ver, expect=ex),
                  "sidecar mismatch refused")
        sc.write_text(f"{r['sha256']}  dc.json\n")
        good_sc = sc.read_bytes()
        for lab, edit in (
                ("sidecar invalid UTF-8",
                 lambda: sc.write_bytes(b"\xff\xfe" + good_sc)),
                ("sidecar replaced by a directory",
                 lambda: (sc.unlink(), sc.mkdir())),
                ("sidecar dangling symlink",
                 lambda: (sc.unlink(), os.symlink(td / "nowhere", sc))),
                ("sidecar names another file",
                 lambda: sc.write_text(f"{r['sha256']}  other.json\n")),
                ("sidecar symlink to a valid copy",
                 lambda: ((td / "sc_copy").write_bytes(good_sc), sc.unlink(),
                          os.symlink(td / "sc_copy", sc)))):
            edit()
            _raises_typed(cx, "DATA_CONTENT_SIDECAR", lambda: DH.verify(
                *ver, expect=ex), f"{lab} -> DATA_CONTENT_SIDECAR")
            if sc.is_dir() and not sc.is_symlink():
                sc.rmdir()
            elif os.path.lexists(sc):
                sc.unlink()
            sc.write_bytes(good_sc)
        cx.ok(DH.verify(*ver, expect=ex)["n_files"] == n,
              "sidecar restored -> verifies again")
        bm = td / "badutf.json"
        bsha = C.write_durable_new(bm, b"\xff\xfe{}")
        C.write_durable_new(Path(str(bm) + ".sha256"),
                            f"{bsha}  {bm.name}\n".encode())
        _raises_typed(cx, "DATA_CONTENT_INVALID", lambda: DH.verify(
            bm, bsha, mp, msha, SYN_HOLDOUT, root, expect=ex),
            "manifest with invalid UTF-8 (own hash, valid sidecar) -> typed")
        (td / "mdir.json").mkdir()
        _raises_typed(cx, "DATA_CONTENT_MANIFEST_MISSING", lambda: DH.verify(
            td / "mdir.json", r["sha256"], mp, msha, SYN_HOLDOUT, root,
            expect=ex), "manifest path is a directory -> typed")
        _raises_typed(cx, "DATA_CONTENT_WRITE_FAILED", lambda: DH.generate(
            mp, msha, SYN_HOLDOUT, root, td / "nodir" / "dc.json", expect=ex),
            "generation write failure (missing directory) -> typed")
        hold = frozenset({sorted(SYN_TRAIN)[0]})
        cx.raises("HOLDOUT_CONTACT", lambda: DH.generate(
            mp, msha, hold, root, td / "dc3.json", expect=ex),
            "holdout identity in the split manifest refused at generation")
        cx.raises("HOLDOUT_CONTACT", lambda: DH.verify(
            out, r["sha256"], mp, msha, hold, root, expect=ex),
            "holdout identity refused at verification")
        cx.ok(not (td / "dc3.json").exists(), "nothing written on refusal")
    if not cx.host:
        return
    cx.phase = "host"
    p = cx.paths
    split = p.inputs / "v05lt_split_manifest.json"
    ssha = C.REGISTERED_INPUTS["v05lt_split_manifest.json"]
    hold = C.holdout_ids(p.v02_manifests)
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "dc_host.json"
        g = DH.generate(split, ssha, hold, cx.data_root, out)
        cx.ok(g["n_files"] == 973, "host: temporary (unregistered) manifest "
              "of the 973 training + validation files",
              detail=f"{g['total_bytes']} bytes")
        v = DH.verify(out, g["sha256"], split, ssha, hold, cx.data_root)
        cx.ok(v["n_files"] == 973 and v["total_bytes"] == g["total_bytes"],
              "host: full verification passes")
        cx.record("DHS host timing (P4 input; PA5-A.3)",
                  {"generate_s": round(g["seconds"], 1),
                   "verify_s": round(v["seconds"], 1),
                   "bytes": g["total_bytes"],
                   "temporary_manifest_sha256": g["sha256"]})
    cx.ok(not (p.inputs / DH.OUT_NAME).exists(), "host: registered inputs "
          "path untouched (gate 2a not run)")


def _syn_split_expect() -> "C.SplitExpect":
    """SplitExpect of synth_split(SYN_VOLS, SYN_TRAIN, SYN_VAL)."""
    tr, va = sorted(SYN_TRAIN), sorted(SYN_VAL)
    return C.SplitExpect(len(tr), len(va), 1, sum(SYN_VOLS[v] for v in tr),
                         sum(SYN_VOLS[v] for v in va), C._ids_sha(tr),
                         C._ids_sha(va))


def _lch_content(tmp: Path) -> tuple:
    """Data-content manifest (REAL v05lt_datahash.generate) over the
    synthetic split written by _syn_train_data(tmp); arguments of
    v05lt_datahash.verify (fixture SplitExpect)."""
    from seqref_mri.scripts import v05lt_datahash as DH
    mp = Path(tmp) / "split.json"
    msha, ex = C.sha256_file(mp), _syn_split_expect()
    out = Path(tmp) / "data_content.json"
    if not out.exists():
        DH.generate(mp, msha, SYN_HOLDOUT, Path(tmp) / "data", out, expect=ex)
    return (out, C.sha256_file(out), mp, msha, SYN_HOLDOUT,
            Path(tmp) / "data", ex)


class _LaunchStages:
    """FIXTURE-ONLY stage set for the launcher (I1.9 c1/c2). Training uses
    the REAL trainer (stub model, synthetic data). c2: verify_data runs the
    REAL data-content verification (v05lt_datahash.verify) over the
    synthetic split files, and post_run publishes and verifies the REAL
    final weights (v05lt_final.extract_final) before writing its post-run
    marker. The epoch-0 stage and the post-run evaluation remain
    write-once fixture markers (the production stage class is not part of
    c2; declared). Never used by production."""

    def __init__(self, root: Path, data: dict, content: tuple,
                 max_epochs: int = 2):
        self.root, self.data, self.max_epochs = Path(root), data, max_epochs
        self.content = content
        self.hooks, self.poison, self.postrun_bad = {}, {}, {}
        self.final_hook = {}             # rid -> extract_final fault hook
        self.calls, self.after_epoch0 = [], None
        self._spoiled = None

    @property
    def data_ok(self) -> bool:
        return self._spoiled is None

    @data_ok.setter
    def data_ok(self, ok: bool) -> None:
        """False: append one byte to a listed training file (real content
        change); True: truncate it back to its original size."""
        f = self.content[5] / "singlecoil_train" / sorted(SYN_TRAIN)[0]
        if not ok and self._spoiled is None:
            self._spoiled = f.stat().st_size
            with open(f, "ab") as fh:
                fh.write(b"X")
        elif ok and self._spoiled is not None:
            os.truncate(f, self._spoiled)
            self._spoiled = None

    def bind_chain(self, doc, root):
        """Fixture stage set: records the call only (the production
        launch-inputs verification is tested in PRD on ProductionStages)."""
        self.calls.append("bind_chain")
        return {"fixture": True, "entries": len(doc["entries"])}

    def verify_data(self):
        from seqref_mri.scripts import v05lt_datahash as DH
        self.calls.append("verify_data")
        path, sha, mp, msha, hold, root, ex = self.content
        DH.verify(path, sha, mp, msha, hold, root, expect=ex)

    def _e0(self, s):
        return self.root / "epoch0" / f"seed_{s}.json"

    def epoch0_done(self, s):
        p = self._e0(s)
        if not p.exists():
            return False
        side = Path(str(p) + ".sha256")
        C.require(side.is_file() and side.read_text().split()[0] ==
                  C.sha256_file(p), "EPOCH0_MARKER_HASH", str(p))
        return True

    def epoch0(self, s):
        from seqref_mri.scripts.v05lt_stub_train import build_stub
        self.calls.append(f"epoch0:{s}")
        p = self._e0(s)
        p.parent.mkdir(parents=True, exist_ok=True)
        sha = C.write_durable_new(p, C.canonical_json({
            "seed": s, "fixture": True,
            "model_state_sha256": C.model_state_sha256(build_stub(s))}))
        C.write_durable_new(Path(str(p) + ".sha256"),
                            f"{sha}  {p.name}\n".encode())
        if self.after_epoch0 is not None:
            self.after_epoch0(s)

    def run_dir(self, s, a):
        return self.root / f"{s}_{a}"

    def cfg(self, s, a):
        return _stub_cfg(self.run_dir(s, a), seed=s, arm=a,
                         max_epochs=self.max_epochs)

    def expected_config(self, s, a):
        return self.cfg(s, a).identity()

    def train(self, s, a, mode):
        rid = f"{s}_{a}"
        self.calls.append(f"train:{rid}:{mode}")
        q = self.hooks.get(rid) or []
        hook = q.pop(0) if q else None
        r = _stub_run(self.cfg(s, a), self.data, hook,
                      poison_after=self.poison.get(rid))
        return r.start() if mode == "start" else r.resume()

    def post_run(self, s, a):
        from seqref_mri.scripts import v05lt_final as FW
        rid = f"{s}_{a}"
        self.calls.append(f"post_run:{rid}")
        bad = self.postrun_bad.get(rid)
        if bad == "unverified":
            return {"verified": False, "artefacts": {}}
        if bad == "raise":
            C.fail("POSTEVAL_INCOMPLETE", f"fixture: {rid}")
        fw = FW.extract_final(self.run_dir(s, a), fixture=True,   # PA5-B
                              hook=self.final_hook.get(rid))
        p = self.root / "postrun" / f"{rid}.json"
        if p.exists():
            sha = C.sha256_file(p)
            C.require(Path(str(p) + ".sha256").read_text().split()[0] == sha,
                      "POSTRUN_MARKER_HASH", str(p))
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            last = [r for r in C.Ledger(self.run_dir(s, a) / "ledger.jsonl")
                    .records() if r["kind"] == "commit"][-1]["body"]
            C.require(fw["model_state_sha256"] == last["model_state_sha256"],
                      "FINAL_WEIGHTS_MISMATCH", f"fixture: {rid}")
            sha = C.write_durable_new(p, C.canonical_json({
                "run": rid, "fixture": True,
                "final_model_state_sha256": last["model_state_sha256"],
                "final_weights_sha256": fw["sha256"]}))
            C.write_durable_new(Path(str(p) + ".sha256"),
                                f"{sha}  {p.name}\n".encode())
        return {"verified": True, "artefacts": {
            p.name: sha, "weights_final.pt": fw["sha256"],
            "final_weights.json": fw["record_sha256"]}}


def _lch_chain(tmp: Path, name: str = "ev") -> tuple:
    from seqref_mri.scripts import v05lt_launch as L
    ev = tmp / name
    ev.mkdir()
    roles = {}
    for r in L.CHAIN_ROLES:
        (ev / f"{r}.json").write_text(json.dumps({"role": r, "fixture": True}))
        roles[r] = f"{name}/{r}.json"
    cp = tmp / f"{name}_chain.json"
    return cp, L.make_launch_chain(cp, tmp, roles), roles


def _lch_kill(at: str):
    def hook(stage):
        if stage == at:
            raise _SimulatedKill(at)
    return hook


def _lch_boom(at: str):
    def hook(stage):
        if stage == at:
            raise RuntimeError(f"fixture: unexpected error at {at}")
    return hook


def _tree_sha(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(d).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(d)).encode())
            h.update(C.sha256_file(p).encode())
    return h.hexdigest()


def _lch_expect_kill(fn) -> bool:
    try:
        fn()
    except _SimulatedKill:
        logger.info("LCH: simulated kill observed (expected)")
        return True
    return False


def lch_group(cx: Ctx) -> None:
    """LCH: launcher (I1.9 c1) on the real trainer path with fixture
    stages: chain, modes, locks, order, interruption, session limit, owner
    decisions, terminal runs, advancement, data verification."""
    from seqref_mri.scripts import v05lt_launch as L
    from seqref_mri.scripts import v05lt_rules as RR
    from seqref_mri.scripts import v05lt_train as T
    cx.ok(L.STOPPED == RR.STOPPED and L.EXHAUSTED == RR.EXHAUSTED and
          L.INTERRUPTED_STATUS == T.INTERRUPTED_STATUS and
          tuple(L.TERMINAL) == tuple(T.TERMINAL) and
          L.MAX_SESSIONS == T.MAX_SESSIONS and L.FIXTURE_END ==
          "fixture_budget_end", "launcher status strings and session limit "
          "= trainer / rules constants")
    order = L.stage_order()
    cx.ok(order == [("epoch0", 2026, None), ("run", 2026, "lam0"),
                    ("run", 2026, "lam2"), ("epoch0", 2027, None),
                    ("run", 2027, "lam0"), ("run", 2027, "lam2"),
                    ("epoch0", 2028, None), ("run", 2028, "lam0"),
                    ("run", 2028, "lam2")], "195 §4 seed-major stage order")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data = _syn_train_data(td)
        content = _lch_content(td)
        cp, csha, roles = _lch_chain(td)
        chain = (cp, csha, td)

        def camp(name, **kw):
            st = _LaunchStages(td / name, data, content, **kw)
            return st, L.Campaign(td / name, st, chain=chain, fixture=True)
        # -- production refused ---------------------------------------------
        cx.raises("LAUNCH_STAGES_NOT_PRODUCTION", lambda: L.Campaign(
            td / "p", _LaunchStages(td / "p", data, content), chain=chain),
            "production campaign with a fixture stage set refused (v0.4)")
        cx.ok(L.main(["new"]) == 2, "launcher CLI without an approved chain "
              "refuses (LAUNCH_CHAIN_UNAPPROVED, exit 2)")
        # -- launch chain ------------------------------------------------------
        cx.ok(len(L.verify_launch_chain(cp, csha, td)["entries"]) == 9,
              "approved launch chain verifies (pin, gate 2a, P1-P6, "
              "launch_inputs)")
        cx.raises("LAUNCH_CHAIN_UNAPPROVED", lambda: L.verify_launch_chain(
            cp, "0" * 64, td), "chain not matching the approved hash refused")
        cx.raises("LAUNCH_CHAIN_ROLES", lambda: L.make_launch_chain(
            td / "c2.json", td, {k: v for k, v in roles.items()
                                 if k != "P5"}), "chain without P5 refused")
        cx.raises("LAUNCH_CHAIN_ROLES", lambda: L.make_launch_chain(
            td / "c3.json", td, dict(roles, P7=roles["P6"])),
            "chain with an unregistered role refused")
        cx.raises("LAUNCH_CHAIN_PATH", lambda: L.make_launch_chain(
            td / "c4.json", td, dict(roles, P1="../x.json")),
            "unsafe evidence path refused")
        cx.raises("LAUNCH_CHAIN_MISSING", lambda: L.make_launch_chain(
            td / "c5.json", td, dict(roles, P2="ev/nope.json")),
            "missing evidence file refused")
        r2 = td / "r2"
        shutil.copytree(td / "ev", r2 / "ev")
        (r2 / "ev" / "P3.json").write_text("altered")
        cx.raises("LAUNCH_CHAIN_HASH", lambda: L.verify_launch_chain(
            cp, csha, r2), "evidence altered under the approved chain refused")
        sha_j = L.make_launch_chain(r2 / "chain.json", r2, roles)
        cx.raises("LAUNCH_CHAIN_UNAPPROVED", lambda: L.verify_launch_chain(
            r2 / "chain.json", csha, r2), "evidence and chain altered "
            "together (internally consistent) refused under the approval")
        cx.ok(sha_j != csha, "regenerated chain has a different sha256")
        # -- full campaign ---------------------------------------------------
        st, c = camp("full")
        res = c.run("new")
        runs = [f"{s}_{a}" for s in L.SEEDS for a in L.ARMS]
        states = {r: L.inspect_run(td / "full" / r, fixture=True)
                  for r in runs}
        kinds = [r["kind"] for r in c.ledger.records()]
        cx.ok(res["status"] == "complete" and all(
            v["class"] == L.COMPLETED for v in states.values()) and
              kinds.count("advance") == 6 and kinds.count("epoch0_done") == 3
              and kinds[0] == "launch_start" and kinds[-1] ==
              "campaign_complete", "fixture campaign completes: 3 epoch-0 "
              "stages, 6 runs, 6 verified advances, in order")
        seq = [x for x in st.calls
               if not x.startswith(("verify", "bind_chain"))]
        want = []
        for s in L.SEEDS:
            want.append(f"epoch0:{s}")
            for a in L.ARMS:
                want += [f"train:{s}_{a}:start", f"post_run:{s}_{a}"]
        cx.ok(seq == want, "stages executed in 195 §4 order, each run "
              "post-run verified before the next starts")
        cx.ok(st.calls.count("verify_data") == 9 and all(
            st.calls[i - 1] == "verify_data" for i, x in enumerate(st.calls)
            if x.startswith(("epoch0:", "train:"))),
            "data verification before every session (3 epoch-0 + 6 runs)")
        from seqref_mri.scripts import v05lt_final as FW
        adv = {r["body"]["run"]: r["body"]["artefacts"]
               for r in c.ledger.records() if r["kind"] == "advance"}
        fws = {r: FW.verify_final(td / "full" / r, fixture=True)
               for r in runs}
        cx.ok(all(adv[r]["weights_final.pt"] == fws[r]["sha256"] and
                  adv[r]["final_weights.json"] == fws[r]["record_sha256"]
                  for r in runs), "c2: every advance records the verified "
              "final weights (REAL v05lt_final) of its run")
        cx.raises("ARTIFACT_EXISTS", lambda: c.run("new"),
                  "new mode on an existing campaign refused")
        n0 = len(st.calls)
        c.run("resume")
        cx.ok(not any(x.startswith(("train:", "epoch0:"))
                      for x in st.calls[n0:]) and
              sum(x.startswith("post_run:") for x in st.calls[n0:]) == 6,
              "resume of a complete campaign trains nothing; post-run "
              "outputs re-verified")
        before = _tree_sha(td / "full")
        sts = L.status(td / "full", st, fixture=True)
        cx.ok(_tree_sha(td / "full") == before and len(sts) == 9,
              "status is read-only")
        # -- modes / locks -----------------------------------------------------
        st2, c2 = camp("nocamp")
        cx.raises("LAUNCH_NO_CAMPAIGN", lambda: c2.run("resume"),
                  "resume without a campaign refused")
        st3, c3 = camp("stray")
        st3.run_dir(2027, "lam2").mkdir(parents=True)
        cx.raises("ARTIFACT_EXISTS", lambda: c3.run("new"),
                  "new mode with a stray run directory refused")
        st4, c4 = camp("locked")
        (td / "locked").mkdir(exist_ok=True)
        with L.Lock(td / "locked" / L.LOCK_NAME, "test"):
            cx.raises("LAUNCH_LOCK_HELD", lambda: c4.run("new"),
                      "campaign lock held -> refuse to start")
        cx.ok(not (td / "locked" / L.LEDGER_NAME).exists(),
              "nothing written while the lock was held")
        with L.Lock(td / "locked" / ".2026_lam0.lock", "test"):
            cx.raises("LAUNCH_LOCK_HELD", lambda: c4.run("new"),
                      "per-run lock held -> refuse to start that run")
        cx.ok(not st4.run_dir(2026, "lam0").exists(),
              "run not started while its lock was held")
        cx.raises("LAUNCH_MODE_INVALID", lambda: c4.run("restart"),
                  "unregistered mode refused")
        # -- interruption and resume (bitwise vs reference) -------------------
        st5, c5 = camp("kill")
        st5.hooks["2026_lam0"] = [_lch_kill("train:step:2")]
        killed = _lch_expect_kill(lambda: c5.run("new"))
        s5 = L.inspect_run(st5.run_dir(2026, "lam0"), fixture=True)
        cx.ok(killed and s5["class"] == L.INTERRUPTED and s5["sessions"] == 1,
              "killed run classified interrupted (1 session)")
        c5.run("resume")
        _same_runs(cx, td / "full" / "2026_lam0", td / "kill" / "2026_lam0",
                   "launcher resume of an interrupted run")
        # -- session limit (F09 launcher part) ---------------------------------
        st6, c6 = camp("limit")
        # synthetic data: 1 optimiser step per epoch, 2 epochs per run
        st6.hooks["2026_lam0"] = [_lch_kill("train:step:2")] * 3
        kills = [_lch_expect_kill(lambda m=m: c6.run(m))
                 for m in ("new", "resume", "resume")]
        s6 = L.inspect_run(st6.run_dir(2026, "lam0"), fixture=True)
        cx.ok(all(kills) and s6["class"] == L.INTERRUPTED and
              s6["sessions"] == 3, "three interrupted sessions")
        cx.raises("LAUNCH_RUN_FAILED", lambda: c6.run("resume"),
                  "4th session refused by the trainer (SESSION_LIMIT); "
                  "campaign halts")
        s6 = L.inspect_run(st6.run_dir(2026, "lam0"), fixture=True)
        k6 = _kinds(st6.run_dir(2026, "lam0"))
        cx.ok(s6["class"] == L.AWAITING and k6.count("session_start") == 3,
              "status 'interrupted — awaiting owner decision'; no 4th session")
        cx.raises("LAUNCH_OWNER_DECISION_REQUIRED", lambda: c6.run("resume"),
                  "awaiting owner: launcher halts without a decision")
        dres = td / "limit_resume.json"
        hres = L.make_owner_decision(dres, "2026_lam0",
                                     st6.run_dir(2026, "lam0"), "resume",
                                     "fixture", fixture=True)
        cx.raises("OWNER_DECISION_NOT_ALLOWED", lambda: c6.run(
            "resume", {"2026_lam0": (dres, hres)}),
            "at the session limit `resume` is not allowed")
        dh = td / "limit_halt.json"
        hh = L.make_owner_decision(dh, "2026_lam0", st6.run_dir(2026, "lam0"),
                                   "halt", "fixture", fixture=True)
        cx.raises("LAUNCH_HALTED_BY_OWNER", lambda: c6.run(
            "resume", {"2026_lam0": (dh, hh)}), "owner `halt` recorded; "
            "campaign halts")
        k = [r for r in c6.ledger.records() if r["kind"] in
             ("owner_decision_applied", "halt")]
        cx.ok(k[-2]["kind"] == "owner_decision_applied" and
              k[-2]["body"]["decision"] == "halt" and
              k[-1]["body"]["code"] == "LAUNCH_HALTED_BY_OWNER" and
              not st6.run_dir(2026, "lam2").exists(),
              "decision and halt in the campaign ledger; next run not started")
        # -- session_error and owner decisions ----------------------------------
        st7, c7 = camp("serr")
        st7.hooks["2026_lam0"] = [_lch_boom("train:step:2")] * 3
        rd = st7.run_dir(2026, "lam0")
        cx.raises("LAUNCH_SESSION_ERROR", lambda: c7.run("new"),
                  "unexpected error -> session_error; campaign halts")
        cx.ok(L.inspect_run(rd, fixture=True)["class"] == L.SESSION_ERROR,
              "run classified session_error")
        cx.raises("LAUNCH_OWNER_DECISION_REQUIRED", lambda: c7.run("resume"),
                  "session_error: no resume without an owner decision")
        d1 = td / "d1.json"
        h1 = L.make_owner_decision(d1, "2026_lam0", rd, "resume", "fixture",
                                   fixture=True)
        cx.raises("OWNER_DECISION_UNAPPROVED", lambda: c7.run(
            "resume", {"2026_lam0": (d1, "1" * 64)}),
            "decision not matching its approved hash refused")
        cx.raises("OWNER_DECISION_STALE", lambda: L.check_owner_decision(
            (d1, h1), "2027_lam0", L.inspect_run(rd, fixture=True)),
            "decision bound to another run refused")
        cx.raises("LAUNCH_SESSION_ERROR", lambda: c7.run(
            "resume", {"2026_lam0": (d1, h1)}),
            "approved resume decision consumes session 2 (which errors "
            "again)")
        s7 = L.inspect_run(rd, fixture=True)
        cx.ok(s7["class"] == L.SESSION_ERROR and s7["sessions"] == 2,
              "second session_error, 2 sessions started")
        cx.raises("OWNER_DECISION_STALE", lambda: c7.run(
            "resume", {"2026_lam0": (d1, h1)}),
            "an old decision never authorises a later interruption")
        d2 = td / "d2.json"
        h2 = L.make_owner_decision(d2, "2026_lam0", rd, "resume", "fixture",
                                   fixture=True)
        cx.raises("LAUNCH_SESSION_ERROR", lambda: c7.run(
            "resume", {"2026_lam0": (d2, h2)}), "session 3 errors")
        s7 = L.inspect_run(rd, fixture=True)
        d3 = td / "d3.json"
        h3 = L.make_owner_decision(d3, "2026_lam0", rd, "resume", "fixture",
                                   fixture=True)
        cx.raises("OWNER_DECISION_NOT_ALLOWED", lambda: c7.run(
            "resume", {"2026_lam0": (d3, h3)}),
            "session_error with 3 sessions started: resume not allowed")
        cx.ok(s7["sessions"] == 3 and _kinds(rd).count("session_start") == 3,
              "no 4th session after the refused decision")
        st8, c8 = camp("serr_ok")
        st8.hooks["2027_lam2"] = [_lch_boom("train:step:2")]
        cx.raises("LAUNCH_SESSION_ERROR", lambda: c8.run("new"),
                  "session_error in 2027_lam2")
        rd8 = st8.run_dir(2027, "lam2")
        d8 = td / "d8.json"
        h8 = L.make_owner_decision(d8, "2027_lam2", rd8, "resume", "fixture",
                                   fixture=True)
        cx.ok(c8.run("resume", {"2027_lam2": (d8, h8)})["status"] ==
              "complete", "approved resume after a recoverable session_error "
              "-> campaign completes")
        _same_runs(cx, td / "full" / "2027_lam2", rd8,
                   "resumed run after an owner-approved session_error")
        # -- refused (resume_refused recorded by the trainer) -------------------
        st9, c9 = camp("refused")
        st9.hooks["2026_lam0"] = [_lch_kill("train:step:2")]
        _lch_expect_kill(lambda: c9.run("new"))
        rd9 = st9.run_dir(2026, "lam0")
        C.Ledger(rd9 / "ledger.jsonl").append("resume_refused", {
            "code": "RESUME_INCOMPATIBLE", "message": "fixture"})
        cx.ok(L.inspect_run(rd9, fixture=True)["class"] == L.REFUSED,
              "trainer refusal classified refused")
        d9 = td / "d9.json"
        h9 = L.make_owner_decision(d9, "2026_lam0", rd9, "resume", "fixture",
                                   fixture=True)
        cx.raises("OWNER_DECISION_NOT_ALLOWED", lambda: c9.run(
            "resume", {"2026_lam0": (d9, h9)}),
            "refused run: resume decision not registered (halt only)")
        # -- terminal ---------------------------------------------------------
        st10, c10 = camp("term")
        st10.poison["2026_lam0"] = 2
        cx.raises("LAUNCH_RUN_FAILED", lambda: c10.run("new"),
                  "numerical_failure -> campaign halts")
        rd10 = st10.run_dir(2026, "lam0")
        k10 = _kinds(rd10)
        cx.raises("LAUNCH_RUN_TERMINAL", lambda: c10.run("resume"),
                  "terminal run never resumed by the launcher")
        cx.ok(_kinds(rd10) == k10 and not st10.run_dir(2026, "lam2").exists(),
              "terminal run ledger unchanged; next run not started")
        cx.raises("OWNER_DECISION_NOT_APPLICABLE", lambda:
                  L.make_owner_decision(td / "d10.json", "2026_lam0", rd10,
                                        "resume", "x", fixture=True),
                  "no owner decision for a terminal run")
        # -- advancement (PA5-C) -----------------------------------------------
        for bad in ("unverified", "raise"):
            stb, cb = camp(f"adv_{bad}")
            stb.postrun_bad["2026_lam0"] = bad
            cx.raises("LAUNCH_POSTRUN_FAILED", lambda cb=cb: cb.run("new"),
                      f"post-run outputs {bad} -> halt, no advancement")
            cx.ok(L.inspect_run(stb.run_dir(2026, "lam0"), fixture=True)
                  ["class"] == L.COMPLETED and not stb.run_dir(
                      2026, "lam2").exists(),
                  f"training complete but not advanced ({bad})")
        stf, cf = camp("adv_final")
        cx.ok(cf.run("new")["status"] == "complete", "c2: campaign completes "
              "with real final-weights publication")
        wf = stf.run_dir(2026, "lam0") / "final" / "weights_final.pt"
        with open(wf, "ab") as fh:
            fh.write(b"X")
        size = wf.stat().st_size
        cx.raises("LAUNCH_POSTRUN_FAILED", lambda: cf.run("resume"),
                  "c2: published final weights altered -> re-verification "
                  "on resume halts")
        hf = [r["body"] for r in cf.ledger.records() if r["kind"] == "halt"]
        cx.ok(hf[-1].get("cause") == "FINAL_WEIGHTS_MISMATCH" and
              wf.stat().st_size == size, "c2: halt cause "
              "FINAL_WEIGHTS_MISMATCH; the mismatched output is not "
              "overwritten")
        # -- c2 rev 2: operational faults are typed and recorded; repair
        #    alone never bypasses the campaign decision ----------------------
        def bound_retry(name, cmp_, label):
            cdp = td / f"{name}_retry.json"
            chs = L.make_campaign_decision(cdp, td / name, "retry", label)
            return cmp_.run("resume", campaign_decision=(cdp, chs))

        def sc_dir(p):
            raw_ = p.read_bytes()
            p.unlink()
            p.mkdir()
            return lambda: (p.rmdir(), p.write_bytes(raw_))

        def sc_utf8(p):
            raw_ = p.read_bytes()
            p.write_bytes(b"\xff\xfe" + raw_)
            return lambda: p.write_bytes(raw_)
        for tag, rel, spoil_fn, cause in (
                ("sc_wdir", "weights_final.pt.sha256", sc_dir,
                 "FINAL_SIDECAR_INVALID"),
                ("sc_rutf", "final_weights.json.sha256", sc_utf8,
                 "FINAL_SIDECAR_INVALID")):
            stx, cxp = camp(tag)
            cxp.run("new")
            fp = stx.run_dir(2026, "lam0") / "final" / rel
            os.chmod(fp, 0o644)
            repair = spoil_fn(fp)
            _raises_typed(cx, "LAUNCH_POSTRUN_FAILED", lambda cxp=cxp:
                          cxp.run("resume"), f"c2: final sidecar {tag} -> "
                          "typed failure, recorded campaign halt")
            hx = [r["body"] for r in cxp.ledger.records()
                  if r["kind"] == "halt"]
            cx.ok(hx[-1].get("cause") == cause, f"c2: {tag} halt cause "
                  f"{cause}", detail=str(hx[-1]))
            repair()
            cx.raises("LAUNCH_HALT_UNRESOLVED", lambda cxp=cxp:
                      cxp.run("resume"), f"c2: {tag} repaired alone -> "
                      "still refused without a campaign decision")
            cx.ok(bound_retry(tag, cxp, f"fixture: {tag} repaired")[
                "status"] == "complete", f"c2: {tag} bound retry -> "
                "completes")
        dcs = Path(str(content[0]) + ".sha256")
        for tag, spoil_fn, when, code in (
                ("dc_utf8", sc_utf8, "epoch0", "LAUNCH_EPOCH0_FAILED"),
                ("dc_dir", sc_dir, "train", "LAUNCH_DATA_UNVERIFIED")):
            stx, cxp = camp(tag)
            holder = {}
            os.chmod(dcs, 0o644)
            if when == "epoch0":
                holder["r"] = spoil_fn(dcs)
            else:
                def later(_s, f=spoil_fn, h=holder, stx=stx):
                    if "r" not in h:
                        h["r"] = f(dcs)
                stx.after_epoch0 = later
            _raises_typed(cx, code, lambda cxp=cxp: cxp.run("new"),
                          f"c2: data-manifest sidecar {tag} -> typed "
                          f"failure, recorded halt {code}")
            hx = [r["body"] for r in cxp.ledger.records()
                  if r["kind"] == "halt"]
            holder["r"]()
            os.chmod(dcs, 0o444)
            stx.after_epoch0 = None
            cx.ok(hx[-1].get("cause") == "DATA_CONTENT_SIDECAR",
                  f"c2: {tag} halt cause DATA_CONTENT_SIDECAR",
                  detail=str(hx[-1]))
            cx.raises("LAUNCH_HALT_UNRESOLVED", lambda cxp=cxp:
                      cxp.run("resume"), f"c2: {tag} repaired alone -> "
                      "still refused without a campaign decision")
            cx.ok(bound_retry(tag, cxp, f"fixture: {tag} repaired")[
                "status"] == "complete", f"c2: {tag} bound retry -> "
                "completes")

        def disk_full(stage):
            if stage == "final:weights_written":
                raise OSError(28, "fixture: no space left on device")
        stw, cw = camp("stage_fail")
        stw.final_hook["2026_lam0"] = disk_full
        _raises_typed(cx, "LAUNCH_POSTRUN_FAILED", lambda: cw.run("new"),
                      "c2: staging write failure -> typed, recorded halt")
        hx = [r["body"] for r in cw.ledger.records() if r["kind"] == "halt"]
        rd_w = stw.run_dir(2026, "lam0")
        cx.ok(hx[-1].get("cause") == "FINAL_WRITE_FAILED" and not
              os.path.lexists(rd_w / "final") and (rd_w / ".final.staging")
              .is_dir(), "c2: halt cause FINAL_WRITE_FAILED; no final "
              "directory; staging left")
        stw.final_hook.clear()
        cx.raises("LAUNCH_HALT_UNRESOLVED", lambda: cw.run("resume"),
                  "c2: write fault cleared alone -> still refused without a "
                  "campaign decision")
        cx.ok(bound_retry("stage_fail", cw, "fixture: disk freed")["status"]
              == "complete" and len(list((rd_w / "_incomplete").glob(
                  "final_staging.*"))) == 1 and FW.verify_final(
                  rd_w, fixture=True)["epoch"] >= 1, "c2: bound retry -> "
              "stale staging quarantined, final weights published, campaign "
              "completes")
        # -- data verification (PA5-A interface) --------------------------------
        st11, c11 = camp("data0")
        st11.data_ok = False
        cx.raises("LAUNCH_EPOCH0_FAILED", lambda: c11.run("new"),
                  "unverified data before the epoch-0 stage -> halt")
        h11 = [r["body"] for r in c11.ledger.records() if r["kind"] == "halt"]
        st11.data_ok = True
        cx.ok(not st11.epoch0_done(2026) and h11[-1].get("cause") ==
              "DATA_CONTENT_MISMATCH", "epoch-0 stage not run; halt cause is "
              "the real data-content check (DATA_CONTENT_MISMATCH)")
        st12, c12 = camp("data1")

        def spoil(_s, st=st12):
            st.data_ok = False
        st12.after_epoch0 = spoil
        cx.raises("LAUNCH_DATA_UNVERIFIED", lambda: c12.run("new"),
                  "unverified data before a training session -> refused "
                  "before it starts")
        st12.data_ok = True
        cx.ok(st12.epoch0_done(2026) and not st12.run_dir(
            2026, "lam0").exists(),
            "no training session started on unverified data")
        # -- configuration and recovered final commit ---------------------------
        st13, c13 = camp("cfg", max_epochs=2)
        st13.after_epoch0 = lambda s: _stub_run(_stub_cfg(
            st13.run_dir(2026, "lam0"), seed=2026, arm="lam0",
            max_epochs=1), data).start()
        cx.raises("LAUNCH_RUN_CONFIG_MISMATCH", lambda: c13.run("new"),
                  "run with a non-registered configuration -> halt")
        st14, c14 = camp("final")
        st14.hooks["2026_lam0"] = [_lch_kill("run:before_run_end")]
        k14 = _lch_expect_kill(lambda: c14.run("new"))
        cx.ok(k14 and c14.run("resume")["status"] == "complete" and any(
            r["kind"] == "run_end_recovered" for r in c14.ledger.records()),
            "final commit without run_end: trainer records run_end "
            "(recovered); launcher advances after post-run verification")
        # -- decision record validation --------------------------------------
        cx.raises("OWNER_DECISION_INVALID", lambda: L.make_owner_decision(
            td / "x1.json", "2026_lam0", rd, "restart", "x", fixture=True),
            "unregistered decision value refused")
        cx.raises("OWNER_DECISION_INVALID", lambda: L.make_owner_decision(
            td / "x2.json", "2026_lam0", rd, "halt", " ", fixture=True),
            "decision without a reason refused")
        cx.raises("OWNER_DECISION_NOT_APPLICABLE", lambda:
                  L.make_owner_decision(td / "x3.json", "2026_lam0",
                                        td / "full" / "2026_lam0", "halt",
                                        "x", fixture=True),
                  "no owner decision for a completed run")
        # -- campaign-stage halts need a bound decision (rev 2) -----------------
        st15, c15 = camp("chalt_data")

        def spoil15(_s, st=st15):
            st.data_ok = False
        st15.after_epoch0 = spoil15
        cx.raises("LAUNCH_DATA_UNVERIFIED", lambda: c15.run("new"),
                  "data fault before a training session -> campaign halt")
        st15.data_ok, st15.after_epoch0 = True, None
        cx.raises("LAUNCH_HALT_UNRESOLVED", lambda: c15.run("resume"),
                  "data fault cleared: continuation still refused without a "
                  "campaign decision")
        cx.ok(not st15.run_dir(2026, "lam0").exists(),
              "no training after merely clearing the data fault")
        cx.raises("OWNER_DECISION_INVALID", lambda: L.make_campaign_decision(
            td / "cd0.json", td / "chalt_data", "resume", "x"),
            "campaign decision value must be retry or halt")
        cd1 = td / "cd1.json"
        ch1 = L.make_campaign_decision(cd1, td / "chalt_data", "retry",
                                       "fixture: data restored")
        cx.raises("OWNER_DECISION_UNAPPROVED", lambda: c15.run(
            "resume", campaign_decision=(cd1, "2" * 64)),
            "campaign decision not matching its approved hash refused")
        cx.ok(c15.run("resume", campaign_decision=(cd1, ch1))["status"] ==
              "complete", "bound `retry` decision permits continuation; "
              "campaign completes")
        k15 = [r for r in c15.ledger.records() if r["kind"] == "halt_resolved"]
        cx.ok(len(k15) == 1 and k15[0]["body"]["decision"] == "retry",
              "resolution recorded in the campaign ledger")
        cx.raises("OWNER_DECISION_NOT_APPLICABLE", lambda:
                  L.make_campaign_decision(td / "cd1b.json", td / "chalt_data",
                                           "retry", "x"),
                  "no campaign decision without an unresolved halt")
        st16, c16 = camp("chalt_post")
        st16.postrun_bad["2026_lam0"] = "unverified"
        cx.raises("LAUNCH_POSTRUN_FAILED", lambda: c16.run("new"),
                  "post-run fault -> campaign halt (training complete)")
        st16.postrun_bad.clear()
        cx.raises("LAUNCH_HALT_UNRESOLVED", lambda: c16.run("resume"),
                  "post-run fault cleared: completed run awaiting "
                  "publication does not advance without a decision")
        cx.ok(not st16.run_dir(2026, "lam2").exists() and
              st16.calls.count("post_run:2026_lam0") == 1,
              "post-run not retried and next run not started")
        cx.raises("OWNER_DECISION_STALE", lambda: c16.run(
            "resume", campaign_decision=(cd1, ch1)),
            "decision bound to another campaign's halt refused")
        cd2 = td / "cd2.json"
        ch2 = L.make_campaign_decision(cd2, td / "chalt_post", "retry",
                                       "fixture: publication fixed")
        st16.postrun_bad["2026_lam0"] = "raise"
        cx.raises("LAUNCH_POSTRUN_FAILED", lambda: c16.run(
            "resume", campaign_decision=(cd2, ch2)),
            "retry consumed; the post-run fails again -> new halt")
        st16.postrun_bad.clear()
        cx.raises("OWNER_DECISION_STALE", lambda: c16.run(
            "resume", campaign_decision=(cd2, ch2)),
            "old campaign decision never authorises a later halt")
        cd3 = td / "cd3.json"
        ch3 = L.make_campaign_decision(cd3, td / "chalt_post", "retry",
                                       "fixture")
        cx.ok(c16.run("resume", campaign_decision=(cd3, ch3))["status"] ==
              "complete", "decision bound to the current halt -> completes")
        st17, c17 = camp("chalt_e0")
        st17.data_ok = False
        cx.raises("LAUNCH_EPOCH0_FAILED", lambda: c17.run("new"),
                  "data fault before epoch 0 -> campaign halt")
        st17.data_ok = True
        cx.raises("LAUNCH_HALT_UNRESOLVED", lambda: c17.run("resume"),
                  "epoch-0 fault cleared: still refused without a decision")
        cx.ok(not st17.epoch0_done(2026), "epoch-0 stage not run")
        cd4 = td / "cd4.json"
        ch4 = L.make_campaign_decision(cd4, td / "chalt_e0", "halt",
                                       "fixture: stop")
        cx.raises("LAUNCH_CAMPAIGN_HALTED", lambda: c17.run(
            "resume", campaign_decision=(cd4, ch4)),
            "campaign decision `halt` ends the campaign")
        cx.raises("LAUNCH_CAMPAIGN_HALTED", lambda: c17.run("resume"),
                  "a halted campaign cannot be resumed")
        # -- malformed launch-chain entries (rev 2) ------------------------
        base = json.loads(cp.read_bytes())

        def bad_chain(name, edit):
            d = json.loads(json.dumps(base))
            edit(d)
            q = td / f"badchain_{name}.json"
            q.write_bytes(C.canonical_json(d))
            return q, C.sha256_file(q)
        for name, edit in (
                ("none_entry", lambda d: d["entries"].append(None)),
                ("str_entry", lambda d: d["entries"].insert(0, "P1")),
                ("missing_sha", lambda d: d["entries"][1].pop("sha256")),
                ("extra_key", lambda d: d["entries"][2].update(x=1)),
                ("int_path", lambda d: d["entries"][3].update(path=5)),
                ("short_sha", lambda d: d["entries"][4].update(
                    sha256="abc")),
                ("entries_dict", lambda d: d.update(entries={}))):
            q, qs = bad_chain(name, edit)
            cx.raises("LAUNCH_CHAIN_INVALID", lambda q=q, qs=qs:
                      L.verify_launch_chain(q, qs, td),
                      f"approved chain with malformed entry ({name}) -> "
                      "typed refusal")
        q, qs = bad_chain("order", lambda d: d["entries"].reverse())
        cx.raises("LAUNCH_CHAIN_ROLES", lambda: L.verify_launch_chain(
            q, qs, td), "approved chain with roles out of order refused")
        # -- epoch-0 verification errors are recorded halts (rev 3) ---------
        st18, c18 = camp("e0verify_pre", max_epochs=1)
        st18.postrun_bad["2026_lam0"] = "unverified"
        cx.raises("LAUNCH_POSTRUN_FAILED", lambda: c18.run("new"),
                  "setup: campaign halted after the 2026 epoch-0 stage")
        ch = L.make_campaign_decision(td / "e0a.json", td / "e0verify_pre",
                                      "retry", "fixture")
        st18.postrun_bad.clear()
        side = Path(str(st18._e0(2026)) + ".sha256")
        good = side.read_bytes()
        side.write_text("0" * 64 + "  seed_2026.json\n")
        cx.raises("LAUNCH_EPOCH0_FAILED", lambda: c18.run(
            "resume", campaign_decision=(td / "e0a.json", ch)),
            "epoch-0 verification error before execution (resume) -> "
            "recorded campaign halt")
        h = [r for r in c18.ledger.records() if r["kind"] == "halt"][-1]
        cx.ok(h["body"].get("seed") == 2026 and h["body"].get("cause") ==
              "EPOCH0_MARKER_HASH", "halt records the seed and the original "
              "verifier code", detail=str(h["body"]))
        side.write_bytes(good)
        n18 = len(st18.calls)
        cx.raises("LAUNCH_HALT_UNRESOLVED", lambda: c18.run("resume"),
                  "artefact restored: resume still blocked without a "
                  "decision")
        cx.ok(st18.calls[n18:] == ["bind_chain"] and not st18.run_dir(
            2026, "lam2").exists(), "no stage executed after restoration "
            "alone (next run not started)")
        cx.raises("OWNER_DECISION_STALE", lambda: c18.run(
            "resume", campaign_decision=(td / "e0a.json", ch)),
            "the earlier (resolved) decision does not cover the new halt")
        ch2 = L.make_campaign_decision(td / "e0b.json", td / "e0verify_pre",
                                       "retry", "fixture: marker restored")
        cx.ok(c18.run("resume", campaign_decision=(td / "e0b.json", ch2))
              ["status"] == "complete", "bound retry after restoration -> "
              "campaign completes")
        st19, c19 = camp("e0verify_post", max_epochs=1)

        def corrupt(s_, st=st19):
            if s_ == 2026:
                Path(str(st._e0(s_)) + ".sha256").write_text(
                    "f" * 64 + "  seed_2026.json\n")
        st19.after_epoch0 = corrupt
        cx.raises("LAUNCH_EPOCH0_FAILED", lambda: c19.run("new"),
                  "epoch-0 verification error after the stage returns -> "
                  "recorded campaign halt")
        h = [r for r in c19.ledger.records() if r["kind"] == "halt"][-1]
        cx.ok(h["body"].get("seed") == 2026 and h["body"].get("cause") ==
              "EPOCH0_MARKER_HASH", "post-stage halt records seed and code")
        p0 = st19._e0(2026)
        Path(str(p0) + ".sha256").write_text(
            f"{C.sha256_file(p0)}  {p0.name}\n")
        st19.after_epoch0 = None
        cx.raises("LAUNCH_HALT_UNRESOLVED", lambda: c19.run("resume"),
                  "post-stage artefact repaired: still blocked")
        cx.ok(not st19.run_dir(2026, "lam0").exists(),
              "no training after the repair alone")
        ch3 = L.make_campaign_decision(td / "e0c.json", td / "e0verify_post",
                                       "retry", "fixture: sidecar repaired")
        cx.ok(c19.run("resume", campaign_decision=(td / "e0c.json", ch3))
              ["status"] == "complete", "bound retry -> completes")
        cx.record("LCH launch chain sha256 (synthetic, informational)", csha)


def _lcr_child(root: str, tmp: str, chain: tuple, barrier, q) -> None:
    """Spawned child for LCR (fixture only)."""
    import time as _t
    from seqref_mri.scripts import v05lt_launch as L
    logging.basicConfig(level=logging.WARNING)
    try:
        data = _syn_train_data(Path(tmp))
        st = _LaunchStages(Path(root), data, _lch_content(Path(tmp)),
                           max_epochs=1)

        def hold(stage):
            if stage == "campaign:locked":
                _t.sleep(3.0)
        c = L.Campaign(Path(root), st, chain=(Path(chain[0]), chain[1],
                                              Path(chain[2])),
                       fixture=True, hook=hold)
        barrier.wait(timeout=120)
        q.put(("ok", c.run("new")["status"]))
    except C.V05LTError as exc:
        logger.info("LCR child refused: %s", exc.code)
        q.put(("err", exc.code))
    except Exception as exc:  # noqa: BLE001 -- reported to the parent
        logger.exception("LCR child unexpected error")
        q.put(("exc", f"{type(exc).__name__}: {exc}"))


def lcr_group(cx: Ctx) -> None:
    """LCR (addition B): two launchers started concurrently on the same
    campaign; exactly one proceeds, the other is refused LAUNCH_LOCK_HELD."""
    from seqref_mri.scripts import v05lt_launch as L
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cp, csha, _ = _lch_chain(td)
        ctx = multiprocessing.get_context("spawn")
        barrier, q = ctx.Barrier(2), ctx.Queue()
        root = td / "race"
        ps = []
        for i in range(2):
            (td / f"d{i}").mkdir()
            ps.append(ctx.Process(target=_lcr_child, args=(
                str(root), str(td / f"d{i}"), (str(cp), csha, str(td)),
                barrier, q)))
        for p in ps:
            p.start()
        res = []
        for _ in ps:
            try:
                res.append(q.get(timeout=600))
            except Exception as exc:  # noqa: BLE001 -- reported as a failure
                logger.error("LCR child result missing: %s", exc)
                res.append(("timeout", type(exc).__name__))
        for p in ps:
            p.join(timeout=60)
            if p.is_alive():
                logger.error("LCR child %s still alive; killed", p.pid)
                p.kill()
        res.sort()
        cx.ok(res == [("err", "LAUNCH_LOCK_HELD"), ("ok", "complete")],
              "concurrent launchers: exactly one runs the campaign, the "
              "other is refused LAUNCH_LOCK_HELD", detail=str(res))
        recs = C.Ledger(root / L.LEDGER_NAME).records()
        cx.ok([r["kind"] for r in recs].count("launch_start") == 1,
              "one launch_start in the campaign ledger")

# ---------------------------------------------------------------------------
# I1.9 c2 production wiring (V05LTI19C2P v0.1): PRD (synthetic), PRB (B)
# ---------------------------------------------------------------------------
_PRD_EV = ("pin", "gate2a", "P1", "P2", "P3", "P4", "P5", "P6")


def _stub_w1000(seed: int):
    """FIXTURE-ONLY stub variant (W x 1000 at initialisation): validation
    R0 PSNR improves over the first epochs, so with 3 epochs the selected
    checkpoint can coincide with epoch 3 and final (planner case)."""
    import torch
    from seqref_mri.scripts.v05lt_stub_train import build_stub
    m = build_stub(seed)
    with torch.no_grad():
        m.W.mul_(1000.0)
    return m


def _prd_layout(tmp: Path, max_epochs: int = 3, hooks=None,
                factory=None) -> dict:
    """Fixture layout for v05lt_prodstages.ProductionStages: synthetic split
    and data, gate-2a / P5 style inputs written with the REAL writers
    (eligibility, data-content manifest, ZF, tolerances), banks B/C from the
    registered recipes, evidence files, launch-inputs record and launch
    chain (with the `launch_inputs` role)."""
    from seqref_mri.scripts import v05lt_datahash as DH
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_launch as L
    from seqref_mri.scripts import v05lt_prodstages as PS
    from seqref_mri.scripts.v05lt_stub_train import build_stub
    tmp = Path(tmp)
    inp = _syn_campaign_inputs(tmp / "in")
    data = inp["data"]
    vv, el = data["vv"], data["eligibility"]
    mids = [i for i in range(len(vv)) if el[vv.entry(i)]["midpoint"]]
    cases = [{"case": j, "file": vv.entry(i)[0], "slice_index": vv.entry(i)[1]}
             for j, i in enumerate(mids)]
    ind = tmp / "in" / "inputs"
    C.write_durable_new(ind / "v05lt_case_manifest.json", C.canonical_json(
        {"schema": "fixture-case-manifest", "cases": cases}))
    banks = _pev_banks()
    for k, n in (("bank_B", "B"), ("bank_C", "C")):
        np.save(ind / f"{k}.npy", banks[n])
    DH.generate(inp["split"], inp["split_sha256"], SYN_HOLDOUT,
                tmp / "in" / "data", ind / DH.OUT_NAME, expect=inp["expect"])
    sts = EV.build_states(vv, list(range(len(vv))), data["p4"], data["s_ref"])
    zrows = [EV.zf_row(st, el[(st["identity"]["file"],
                               int(st["identity"]["slice_index"]))])
             for st in sts]
    EV.write_zf(tmp / "in" / "zf", zrows, EV.zf_summary(zrows),
                {"fixture": True})
    tol = EV.calibrate_dc(EV.build_states(vv, mids, data["p4"],
                                          data["s_ref"]))["tolerances"]
    (tmp / "in" / "p5").mkdir()
    PS.make_tolerances(tmp / "in" / "p5" / "tol.json", tol, {"fixture": True})
    ev = tmp / "ev"
    ev.mkdir()
    roles = {}
    for r in _PRD_EV:
        (ev / f"{r}.json").write_text(json.dumps({"role": r, "fixture": True}))
        roles[r] = f"ev/{r}.json"
    rel = {"split_manifest": "in/split.json",
           "case_manifest": "in/inputs/v05lt_case_manifest.json",
           "bank_B": "in/inputs/bank_B.npy", "bank_C": "in/inputs/bank_C.npy",
           "data_content": f"in/inputs/{DH.OUT_NAME}",
           "eligibility": "in/inputs/v05lt_nmse_eligibility.json",
           "zf": f"in/zf/{EV.ZF_FILE}", "dc_tolerances": "in/p5/tol.json"}
    fx = {"root": tmp / "camp", "model_factory": factory or build_stub,
          "max_epochs": max_epochs, "p4": data["p4"], "s_ref": data["s_ref"],
          "split_expect": inp["expect"], "holdout": SYN_HOLDOUT,
          "repo": REPO_ROOT, "hooks": hooks if hooks is not None else {}}
    if factory == "real":            # c4 HLB: real architecture (no factory)
        del fx["model_factory"]
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()

    def stages(**over):
        return PS.ProductionStages(C.Paths(), tmp / "in" / "data",
                                   fixture=dict(fx, **over))
    st = stages()
    li = PS.make_launch_inputs(
        ev / "launch_inputs.json", tmp, st,
        chain_files={r: roles[r] for r in PS.BOUND_ROLES}, inputs=rel,
        commit=head)
    roles[PS.LAUNCH_INPUTS_ROLE] = "ev/launch_inputs.json"
    cp = tmp / "chain.json"
    csha = L.make_launch_chain(cp, tmp, roles)
    layout = {"split_manifest": inp["split"],
              "split_sha256": inp["split_sha256"],
              "split_expect": inp["expect"],
              "case_manifest": ind / "v05lt_case_manifest.json",
              "case_sha256": C.sha256_file(ind / "v05lt_case_manifest.json"),
              "eligibility": inp["eligibility"],
              "eligibility_sha256": inp["eligibility_sha256"],
              "zf_dir": tmp / "in" / "zf",
              "zf_sha256": EV.verify_zf(tmp / "in" / "zf"),
              "difficult": tmp / "camp" / "analysis" / "difficult.json",
              "difficult_sha256": None,
              "epoch0": {s: tmp / "camp" / "epoch0" / f"seed_{s}" / "eval"
                         for s in (2026, 2027, 2028)},
              "runs": {f"{a}|{s}": tmp / "camp" / f"{s}_{a}"
                       for s in (2026, 2027, 2028) for a in ("lam0", "lam2")},
              "posteval_root": tmp / "camp" / "posteval",
              "seeds": [2026, 2027, 2028], "arms": ["lam0", "lam2"],
              "fixture": True}
    return {"tmp": tmp, "stages": stages, "st": st, "chain": (cp, csha, tmp),
            "roles": roles, "rel": rel, "li_sha": li, "tol": tol,
            "cases": cases, "layout": layout, "fx": fx, "head": head,
            "data": data}


def _prd_chain_with(lay: dict, name: str, edit) -> tuple:
    """Copy of the launch-inputs record edited by `edit(doc)`, listed in a
    new (self-consistent) chain: tests the record checks, not the chain."""
    from seqref_mri.scripts import v05lt_launch as L
    tmp = lay["tmp"]
    doc = json.loads((tmp / "ev" / "launch_inputs.json").read_bytes())
    edit(doc)
    p = tmp / "ev" / f"li_{name}.json"
    p.write_bytes(C.canonical_json(doc))
    roles = dict(lay["roles"], launch_inputs=f"ev/li_{name}.json")
    cp = tmp / f"chain_{name}.json"
    return cp, L.make_launch_chain(cp, tmp, roles), tmp


def _prd_bind(st, chain: tuple) -> dict:
    from seqref_mri.scripts import v05lt_launch as L
    return st.bind_chain(L.verify_launch_chain(*chain), chain[2])


def _prd_kill_at(name: str, count: int = 1):
    n = {"k": 0}

    def hook(stage):
        if stage == name:
            n["k"] += 1
            if n["k"] == count:
                raise _SimulatedKill(stage)
    return hook


class _PrdBoom:
    """FIXTURE-ONLY delegating stage set (fixture campaigns only): raises
    `exc` from `method` while armed (addition C tests)."""

    def __init__(self, inner, method: str, exc):
        self._inner, self.method, self.exc, self.armed = inner, method, exc, True

    def __getattr__(self, name):
        fn = getattr(self._inner, name)
        if name != self.method:
            return fn

        def wrapped(*a):
            if self.armed:
                raise self.exc
            return fn(*a)
        return wrapped


def _prd_tasks(c) -> dict:
    return {r["body"]["run"]: r["body"]["artefacts"]
            for r in c.ledger.records() if r["kind"] == "advance"}


def prd_group(cx: Ctx) -> None:
    """PRD (synthetic by design): production stage class on a fixture
    layout (stub model through the declared fixture factory, synthetic
    data, temporary directories) through the launcher, plus refusals of the
    six c2-wiring refinements."""
    from seqref_mri.scripts import v05lt_epoch0 as E0
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_launch as L
    from seqref_mri.scripts import v05lt_prodstages as PS
    from seqref_mri.scripts import v05lt_results as RS
    from seqref_mri.scripts.v05lt_stub_train import build_stub
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # == core: full 6-run campaign through the production stage class ==
        A = _prd_layout(td / "A", max_epochs=3, factory=_stub_w1000)
        stA = A["st"]
        cx.raises("PROD_STAGES_UNBOUND", lambda: stA.epoch0_done(2026),
                  "stages refuse to work before the launch inputs are bound")
        # launcher-to-input binding (direct assertion, review rev 1)
        class _BindLie(PS.ProductionStages):
            def bind_chain(self, doc, root):
                return {"launch_inputs_sha256": "0" * 64}

        class _BindNone(PS.ProductionStages):
            def bind_chain(self, doc, root):
                return None
        for cls in (_BindLie, _BindNone):
            r_ = td / f"bind_{cls.__name__}"
            cx.raises("LAUNCH_INPUTS_UNBOUND", lambda cls=cls, r_=r_: L.Campaign(
                r_, cls(C.Paths(), A["tmp"] / "in" / "data",
                        fixture=dict(A["fx"], root=r_)), chain=A["chain"],
                fixture=True).run("new"), f"launcher refuses stages not bound "
                f"to the chain's launch-inputs record ({cls.__name__})")
        cA = L.Campaign(A["tmp"] / "camp", stA, chain=A["chain"],
                        fixture=True)
        try:
            out = cA.run("new")
        except C.V05LTError as exc:
            cx.ok(exc.code != "LAUNCH_INPUTS_UNBOUND", "launcher binds the "
                  "production stages to the chain's launch-inputs record",
                  "BINDING_FAILED", exc.code)
            raise
        cx.ok(stA.inputs is not None and stA.inputs["record_sha256"] ==
              A["li_sha"], "launcher bound the stages to the chain's "
              "launch-inputs record (direct assertion)")
        cx.ok(out["status"] == "complete" and len(out["stages"]) == 9,
              "fixture campaign through ProductionStages: 3 epoch-0 stages "
              "and 6 runs complete")
        recs = cA.ledger.records()
        ls = [r for r in recs if r["kind"] == "launch_start"][0]["body"]
        cx.ok(ls["bound"]["launch_inputs_sha256"] == A["li_sha"],
              "launch_start records the bound launch-inputs sha256")
        advA = _prd_tasks(cA)
        cx.ok(len(advA) == 6 and all(
            {"weights_final.pt", "final_weights.json"} <= set(v) and
            any(k.startswith("posteval/") for k in v) for v in advA.values()),
            "every advance lists final weights, provenance record and its "
            "post-run records (PA5-C)")
        camp = RS.load_campaign(A["layout"])
        cx.ok(len(camp["runs"]) == 6 and len(camp["posteval"]) == 18 and
              len(camp["epoch0"]) == 3,
              "accepted results loader reads the whole campaign written by "
              "the production stages (6 runs x epoch 3 / selected / final, "
              "3 epoch-0 records)")
        e0doc = json.loads((A["tmp"] / "camp" / "epoch0" / "seed_2026" /
                            "eval" / EV.E0_EVAL_FILE).read_bytes())
        cx.ok(e0doc["context"]["launch_inputs_sha256"] == A["li_sha"] and
              e0doc["dc_tolerances"] == A["tol"] and
              e0doc["midpoint"]["n"] == len(A["cases"]) and
              e0doc["cases"] == [c["case"] for c in A["cases"]],
              "epoch-0 evaluation: full populations, bound tolerances, "
              "registered cases, launch-inputs binding")
        # refinement 4: planner (dedup, prefixes, B/C at selected)
        coinc = distinct = 0
        for rid, arts in advA.items():
            for k in arts:
                if not k.startswith("posteval/"):
                    continue
                doc = EV.load_posteval(A["tmp"] / "camp" / k)
                roles = sorted(u["role"] for u in doc["uses"])
                sel = "selected" in roles
                cx.ok(sorted(doc["banks"]) == (["B", "C", "primary"] if sel
                                               else ["primary"]) and
                      doc["prefixes"] == (list(EV.PREFIXES) if sel else
                                          [EV.BANK_ROWS]) and
                      doc["midpoint"]["decodes"] == {
                          b: 128 * len(A["cases"]) for b in doc["banks"]},
                      f"{rid} {roles}: banks B/C and prefixes only at the "
                      "selected checkpoint; one 128-row decode per slice and "
                      "bank")
                coinc += roles == ["epoch3", "final", "selected"]
        B = _prd_layout(td / "B", max_epochs=5)
        cB = L.Campaign(B["tmp"] / "camp", B["st"], chain=B["chain"],
                        fixture=True)
        cB.run("new")
        for rid, arts in _prd_tasks(cB).items():
            pe = sorted(sorted(u["role"] for u in EV.load_posteval(
                B["tmp"] / "camp" / k)["uses"])
                for k in arts if k.startswith("posteval/"))
            distinct += pe == [["epoch3"], ["final"], ["selected"]]
        cx.ok(coinc >= 1 and distinct >= 1, "planner cases exercised: "
              "coincident epoch 3 = selected = final (one record, all three "
              "roles) and distinct epoch 3 / selected / final (three "
              "records)", detail=f"coincident runs {coinc}, distinct {distinct}")
        RS.load_campaign(B["layout"])
        cx.ok(True, "5-epoch campaign also read by the accepted loader")
        # resume of a complete campaign: outputs verified, nothing rewritten
        before = (_tree_sha(A["tmp"] / "camp" / "posteval"),
                  _tree_sha(A["tmp"] / "camp" / "epoch0"))
        calls = {"decode_bank": 0, "posteval_evaluate": 0,
                 "epoch0_evaluate": 0}
        saved = {n: getattr(EV, n) for n in calls}

        def counting(n):
            def f(*a, **k):
                calls[n] += 1
                return saved[n](*a, **k)
            return f
        try:                       # fixture patch: call counters only
            for n in calls:
                setattr(EV, n, counting(n))
            cA.run("resume")
        finally:
            for n, fn in saved.items():
                setattr(EV, n, fn)
        cx.ok((_tree_sha(A["tmp"] / "camp" / "posteval"),
               _tree_sha(A["tmp"] / "camp" / "epoch0")) == before and
              _prd_tasks(cA) == advA and calls == {
                  "decode_bank": 0, "posteval_evaluate": 0,
                  "epoch0_evaluate": 0},
              "resume of a complete campaign: complete outputs verified and "
              "reused without any evaluation or decode, never rewritten",
              detail=str(calls))
        # refinement 4 / 3: complete-but-mismatched post-run records
        rid0 = "2026_lam0"
        k0 = sorted(k for k in advA[rid0] if k.startswith("posteval/"))[0]
        d0 = A["tmp"] / "camp" / k0
        keep = A["tmp"] / "keep_pe"
        shutil.move(str(d0), str(keep))
        doc0 = json.loads((keep / EV.POSTEVAL_FILE).read_bytes())

        def rewrite(edit_ctx=None, edit_rec=None, edit_doc=None, d0=d0):
            if os.path.lexists(d0):
                shutil.rmtree(d0)
            run = RS.load_run(A["tmp"] / "camp" / rid0, "lam0", 2026,
                              stA.manifests(), RS.Identities(), fixture=True)
            fw = __import__("seqref_mri.scripts.v05lt_final",
                            fromlist=["x"]).verify_final(
                A["tmp"] / "camp" / rid0, fixture=True)
            ck = run["checkpoints"]
            plan = EV.plan_campaign([{"seed": 2026, "arm": "lam0", "roles": {
                "epoch0": {"epoch": 0, "model_state": run["epoch0_model_state"]},
                **{r: {"epoch": ck[r]["epoch"], "model_state":
                       ck[r]["model_state_sha256"]}
                   for r in ("epoch3", "selected", "final")}}}],
                n_slices=len(A["cases"]))
            task = [t for t in plan["tasks"] if t["model_state"] == d0.name][0]
            if edit_rec:
                task = edit_rec(dict(task))
            wf, ws = stA._task_file(run, fw, task)
            m, vck = EV.load_trained_checkpoint(
                wf, ws, task["model_state"], seed=2026, arm="lam0",
                epoch=task["epoch"], model_factory=_stub_w1000)
            bk = stA.banks()
            rec, pan = EV.posteval_evaluate(
                m, stA.data(), task, {b: bk[b] for b in task["banks"]},
                A["tol"], stA.manifests()["cases"], checkpoint=vck,
                eligibility_sha256=stA._bound()["eligibility"]["sha256"])
            ctx = {"run": rid0, "launch_inputs_sha256": A["li_sha"],
                   "fixture": True}
            if edit_ctx:
                ctx = edit_ctx(ctx)
            if edit_doc:
                edit_doc(rec)
            EV.write_posteval(d0, rec, pan, ctx)
        rewrite(edit_ctx=lambda c: dict(c, launch_inputs_sha256="0" * 64))
        cx.raises("PROD_POSTEVAL_MISMATCH", lambda: stA.post_run(2026, "lam0"),
                  "complete post-run record from another launch-inputs "
                  "record refused (not reused, not overwritten)")
        rewrite(edit_ctx=lambda c: dict(c, run="2027_lam0"))
        cx.raises("PROD_POSTEVAL_MISMATCH", lambda: stA.post_run(2026, "lam0"),
                  "complete post-run record bound to another run refused")
        sel_roles = {u["role"] for u in doc0["uses"]}

        def other_uses(t):
            t["uses"] = [u for u in t["uses"] if u["role"] != sorted(
                sel_roles)[0]] or t["uses"]
            return t
        if len(sel_roles) > 1:
            rewrite(edit_rec=other_uses)
            cx.raises("PROD_POSTEVAL_MISMATCH", lambda: stA.post_run(
                2026, "lam0"), "complete record of a different task (roles) "
                "refused: each record checked against its expected task")
        # v0.36 (review rev 1): masks, code and backend bound on reuse
        cx.ok(doc0["masks"]["sha256"] == stA.expected_masks_sha256(),
              "independent mask identity (registered seeds, loaded widths) "
              "= the evaluator's recorded mask hash")
        for lab, ed in (
                ("wrong mask hash", lambda r: r["masks"].update(
                    sha256="ab" * 32)),
                ("wrong code identity", lambda r: r["code"]["files"].update(
                    {"v05lt_eval.py": "0" * 64})),
                ("wrong backend", lambda r: r["backend"].update(
                    threads=r["backend"]["threads"] + 1))):
            rewrite(edit_doc=ed)
            cx.raises("PROD_POSTEVAL_MISMATCH", lambda: stA.post_run(
                2026, "lam0"), f"otherwise valid reused record with {lab} "
                "refused")
        shutil.rmtree(d0)
        shutil.copytree(keep, d0)
        allk = sorted(k for k in advA[rid0] if k.startswith("posteval/"))
        keeps = {}
        for k in allk:
            dk_ = A["tmp"] / "camp" / k
            keeps[k] = A["tmp"] / f"keep_all_{dk_.name}"
            shutil.copytree(dk_, keeps[k])
            rewrite(edit_doc=lambda r: r["masks"].update(sha256="cd" * 32),
                    d0=dk_)
        cx.ok(len(allk) >= 2, "run with several post-run records")
        RS.load_posteval_for(A["tmp"] / "camp" / "posteval", RS.load_run(
            A["tmp"] / "camp" / rid0, "lam0", 2026, stA.manifests(),
            RS.Identities(), fixture=True), "selected", stA.manifests(),
            RS.Identities(), {})
        cx.record("observation: the accepted downstream loader checks mask "
                  "agreement only", "a wrong mask hash shared by all records "
                  "of a run is accepted by load_posteval_for; refused by the "
                  "production stage below")
        cx.raises("PROD_POSTEVAL_MISMATCH", lambda: stA.post_run(2026, "lam0"),
                  "wrong mask hash shared by all records of the run refused "
                  "(expected identity recomputed, not agreement)")
        for k in allk:
            dk_ = A["tmp"] / "camp" / k
            shutil.rmtree(dk_)
            shutil.copytree(keeps[k], dk_)
        shutil.rmtree(d0)
        shutil.copytree(keep, d0)
        pf = d0 / EV.POSTEVAL_FILE
        os.chmod(pf, 0o644)
        pf.write_bytes(pf.read_bytes().replace(b'"cases"', b'"cases" ', 1))
        cx.raises("PROD_POSTEVAL_INCOMPLETE", lambda: stA.post_run(
            2026, "lam0"), "altered published record (hash mismatch) -> "
            "incomplete, refused")
        shutil.rmtree(d0)
        shutil.copytree(keep, d0)
        r1 = stA.post_run(2026, "lam0")
        cx.ok(r1["artefacts"] == advA[rid0], "original record restored: "
              "post-run verifies again with identical artefacts")
        # refinement 3: epoch-0 complete-but-mismatched / incomplete
        e0d = A["tmp"] / "camp" / "epoch0" / "seed_2027" / "eval"
        e0keep = A["tmp"] / "keep_e0"
        shutil.move(str(e0d), str(e0keep))
        ck27 = stA.epoch0_checkpoint(2027)
        m27 = E0.load_epoch0(ck27["path"], ck27["model_state_sha256"], 2027,
                             model_factory=_stub_w1000)
        tol2 = {k: v * 2 for k, v in A["tol"].items()}
        for lab, tol_, ctx in (
                ("other DC tolerances", tol2, stA._e0_context(2027, ck27)),
                ("other launch-inputs binding", A["tol"],
                 dict(stA._e0_context(2027, ck27),
                      launch_inputs_sha256="0" * 64)),
                ("other backend", A["tol"], dict(
                    stA._e0_context(2027, ck27), backend=dict(
                        stA._e0_context(2027, ck27)["backend"], threads=999))),
                ("other code identity", A["tol"], dict(
                    stA._e0_context(2027, ck27), code={"files": {}}))):
            rec, pan = EV.epoch0_evaluate(m27, stA.data(), stA.banks()[
                "primary"], tol_, stA.manifests()["cases"])
            if os.path.lexists(e0d):
                shutil.rmtree(e0d)
            EV.write_epoch0_eval(e0d, rec, pan, ctx)
            cx.raises("PROD_EPOCH0_MISMATCH", lambda: stA.epoch0_done(2027),
                      f"complete epoch-0 evaluation with {lab} refused")
        shutil.rmtree(e0d)
        shutil.copytree(e0keep, e0d)
        f = e0d / EV.E0_EVAL_FILE
        os.chmod(f, 0o644)
        f.write_bytes(f.read_bytes() + b" ")
        cx.raises("PROD_EPOCH0_INCOMPLETE", lambda: stA.epoch0_done(2027),
                  "altered epoch-0 evaluation file -> incomplete")
        shutil.rmtree(e0d)
        shutil.copytree(e0keep, e0d)
        cx.ok(stA.epoch0_done(2027), "restored epoch-0 evaluation verifies")
        (e0d.parent / ".eval.staging").mkdir()
        cx.raises("PROD_EPOCH0_INCOMPLETE", lambda: stA.epoch0_done(2027),
                  "left-over evaluation staging -> incomplete")
        (e0d.parent / ".eval.staging").rmdir()
        # == refinement 3 through the launcher: interrupted outputs ==========
        hooks = {}
        K = _prd_layout(td / "K", max_epochs=3, hooks=hooks)
        cK = L.Campaign(K["tmp"] / "camp", K["st"], chain=K["chain"],
                        fixture=True)
        hooks["epoch0_eval"] = _prd_kill_at("epoch0_eval:staged")
        cx.ok(_lch_expect_kill(lambda: cK.run("new")), "kill after the "
              "epoch-0 evaluation was staged passes through (not recorded)")
        hooks.clear()
        cx.ok(not any(r["kind"] == "halt" for r in cK.ledger.records()),
              "process interruption recorded no halt")
        cx.raises("LAUNCH_EPOCH0_FAILED", lambda: cK.run("resume"),
                  "resume: incomplete epoch-0 output -> recorded halt")
        h = [r for r in cK.ledger.records() if r["kind"] == "halt"][-1]
        cx.ok(h["body"].get("cause") == "PROD_EPOCH0_INCOMPLETE",
              "halt cause PROD_EPOCH0_INCOMPLETE", detail=str(h["body"]))
        cx.raises("LAUNCH_HALT_UNRESOLVED", lambda: cK.run("resume"),
                  "no decision -> resume blocked")
        dk = L.make_campaign_decision(K["tmp"] / "dk1.json", K["tmp"] / "camp",
                                      "retry", "fixture: retry unfixed")
        cx.raises("LAUNCH_EPOCH0_FAILED", lambda: cK.run(
            "resume", campaign_decision=(K["tmp"] / "dk1.json", dk)),
            "retry without repair halts again (nothing deleted or rewritten)")
        stg = K["tmp"] / "camp" / "epoch0" / "seed_2026" / ".eval.staging"
        cx.ok(stg.is_dir(), "the interrupted staging was left in place")
        shutil.move(str(stg), str(K["tmp"] / "inspected_staging"))
        dk2 = L.make_campaign_decision(K["tmp"] / "dk2.json",
                                       K["tmp"] / "camp", "retry",
                                       "fixture: staging inspected and moved")
        hooks["train:2026_lam0"] = _prd_kill_at("commit:before_ledger")
        cx.ok(_lch_expect_kill(lambda: cK.run(
            "resume", campaign_decision=(K["tmp"] / "dk2.json", dk2))),
            "bound retry after owner repair: epoch-0 stage completes "
            "(campaign then interrupted in training on purpose)")
        hooks.clear()
        cx.ok(K["st"].epoch0_done(2026), "seed 2026 epoch-0 outputs "
              "complete and verified after the repair")
        # interrupted epoch-0 checkpoint creation
        K2 = _prd_layout(td / "K2", max_epochs=3, hooks=hooks)
        cK2 = L.Campaign(K2["tmp"] / "camp", K2["st"], chain=K2["chain"],
                         fixture=True)
        hooks["epoch0_create"] = _prd_kill_at("stage:linked")
        cx.ok(_lch_expect_kill(lambda: cK2.run("new")),
              "kill during the epoch-0 checkpoint save passes through")
        hooks.clear()
        cx.raises("LAUNCH_EPOCH0_FAILED", lambda: cK2.run("resume"),
                  "resume: uncommitted epoch-0 checkpoint -> recorded halt")
        h = [r for r in cK2.ledger.records() if r["kind"] == "halt"][-1]
        cx.ok(h["body"].get("cause") == "PROD_EPOCH0_INCOMPLETE",
              "halt cause PROD_EPOCH0_INCOMPLETE (checkpoint)")
        rcv = E0.recover(K2["tmp"] / "camp" / "epoch0", 2026, REPO_ROOT,
                         model_factory=build_stub)
        cx.ok(rcv["recovery"] == "rebuilt_uncommitted", "owner repair: the "
              "registered epoch-0 --recover procedure rebuilds")
        dk3 = L.make_campaign_decision(K2["tmp"] / "dk3.json",
                                       K2["tmp"] / "camp", "retry",
                                       "fixture: recovered")
        hooks["train:2026_lam0"] = _prd_kill_at("commit:before_ledger")
        cx.ok(_lch_expect_kill(lambda: cK2.run(
            "resume", campaign_decision=(K2["tmp"] / "dk3.json", dk3))),
            "bound retry completes the epoch-0 stage")
        hooks.clear()
        cx.ok(K2["st"].epoch0_done(2026), "epoch-0 complete after recovery")
        # interrupted post-run evaluation
        K3 = _prd_layout(td / "K3", max_epochs=3, hooks=hooks)
        cK3 = L.Campaign(K3["tmp"] / "camp", K3["st"], chain=K3["chain"],
                         fixture=True)
        hooks["posteval"] = _prd_kill_at("posteval:staged")
        cx.ok(_lch_expect_kill(lambda: cK3.run("new")),
              "kill after a post-run record was staged passes through")
        hooks.clear()
        cx.raises("LAUNCH_POSTRUN_FAILED", lambda: cK3.run("resume"),
                  "resume: incomplete post-run output -> recorded halt")
        h = [r for r in cK3.ledger.records() if r["kind"] == "halt"][-1]
        cx.ok(h["body"].get("cause") == "PROD_POSTEVAL_INCOMPLETE",
              "halt cause PROD_POSTEVAL_INCOMPLETE", detail=str(h["body"]))
        dk4 = L.make_campaign_decision(K3["tmp"] / "dk4.json",
                                       K3["tmp"] / "camp", "retry", "unfixed")
        cx.raises("LAUNCH_POSTRUN_FAILED", lambda: cK3.run(
            "resume", campaign_decision=(K3["tmp"] / "dk4.json", dk4)),
            "retry without repair halts again")
        pst = [p for p in (K3["tmp"] / "camp" / "posteval").iterdir()
               if p.name.endswith(".staging")]
        cx.ok(len(pst) == 1, "the interrupted post-run staging was left")
        shutil.move(str(pst[0]), str(K3["tmp"] / "inspected_pe"))
        dk5 = L.make_campaign_decision(K3["tmp"] / "dk5.json",
                                       K3["tmp"] / "camp", "retry", "moved")
        out3 = cK3.run("resume", campaign_decision=(K3["tmp"] / "dk5.json",
                                                    dk5))
        cx.ok(out3["status"] == "complete", "bound retry after repair: "
              "campaign completes")
        RS.load_campaign(K3["layout"])
        cx.ok(True, "recovered campaign read by the accepted loader")
        # == refinement 1: launch inputs ========================================
        st1 = A["stages"]()
        cx.raises("LAUNCH_CHAIN_ROLES", lambda: L.make_launch_chain(
            td / "nochain.json", A["tmp"], {k: v for k, v in A["roles"].items()
                                            if k != "launch_inputs"}),
            "launch chain without the launch_inputs role refused")
        for lab, edit, code in (
                ("pin binding", lambda d: d["bindings"].update(
                    pin_sha256="0" * 64), "PROD_LAUNCH_INPUTS_BINDING"),
                ("gate-2a binding", lambda d: d["bindings"].update(
                    gate2a_sha256="1" * 64), "PROD_LAUNCH_INPUTS_BINDING"),
                ("P5 binding", lambda d: d["bindings"].pop("P5_sha256"),
                 "PROD_LAUNCH_INPUTS_BINDING"),
                ("campaign configuration", lambda d: d.update(
                    campaign_config_sha256="2" * 64),
                 "PROD_LAUNCH_INPUTS_CONFIG"),
                ("code identity (hash of a pinned file)",
                 lambda d: d["pin"]["code"].update(
                     {"seqref_mri/scripts/v05lt_train.py": "3" * 64}),
                 "PROD_PIN_MISMATCH"),
                ("missing input", lambda d: d["inputs"].pop("zf"),
                 "PROD_LAUNCH_INPUTS_INVALID"),
                ("extra input", lambda d: d["inputs"].update(
                    extra=d["inputs"]["zf"]), "PROD_LAUNCH_INPUTS_INVALID"),
                ("input hash", lambda d: d["inputs"]["eligibility"].update(
                    sha256="4" * 64), "PROD_INPUT_HASH"),
                ("unsafe path", lambda d: d["inputs"]["zf"].update(
                    path="../x.json"), "PROD_INPUT_PATH"),
                ("mode (production record)", lambda d: d.update(
                    fixture=False), "PROD_LAUNCH_INPUTS_MODE"),
                ("schema", lambda d: d.update(schema="x"),
                 "PROD_LAUNCH_INPUTS_INVALID"),
                ("extra key", lambda d: d.update(note="x"),
                 "PROD_LAUNCH_INPUTS_INVALID")):
            ch = _prd_chain_with(A, lab.replace(" ", "_").replace("(", "")
                                 .replace(")", ""), edit)
            _raises_typed(cx, code, lambda ch=ch: _prd_bind(A["stages"](), ch),
                          f"launch-inputs record: {lab} -> {code}")
        doc = L.verify_launch_chain(*A["chain"])
        bad = json.loads(json.dumps(doc))
        [e for e in bad["entries"] if e["role"] == "launch_inputs"][0][
            "sha256"] = "5" * 64
        cx.raises("PROD_LAUNCH_INPUTS_HASH", lambda: st1.bind_chain(
            bad, A["tmp"]), "record hash != chain entry refused")
        cx.raises("PROD_LAUNCH_INPUTS_CONFIG", lambda: _prd_bind(
            A["stages"](max_epochs=4), A["chain"]),
            "stages with another (fixture) configuration refused")
        prod_like = PS.ProductionStages(C.Paths(repo=REPO_ROOT),
                                        A["tmp"] / "in" / "data")
        pli = PS.make_launch_inputs(
            A["tmp"] / "ev" / "li_prodmode.json", A["tmp"], prod_like,
            chain_files={r: A["roles"][r] for r in PS.BOUND_ROLES},
            inputs=A["rel"], commit=A["head"])
        cx.ok(len(pli) == 64, "record built for production-mode stages")
        chp = (A["tmp"] / "chain_prodmode.json", L.make_launch_chain(
            A["tmp"] / "chain_prodmode.json", A["tmp"], dict(
                A["roles"], launch_inputs="ev/li_prodmode.json")), A["tmp"])
        cx.raises("PROD_LAUNCH_INPUTS_MODE", lambda: _prd_bind(
            A["stages"](), chp), "production-mode record refused by "
            "fixture-mode stages")
        # input files altered under the approved record / chain
        ind = A["tmp"] / "in" / "inputs"

        def altered(rel_file, mut, code, lab):
            p = A["tmp"] / rel_file
            raw = p.read_bytes()
            mode = p.stat().st_mode
            os.chmod(p, 0o644)
            try:
                mut(p)
                _raises_typed(cx, code, lambda: _prd_bind(A["stages"](),
                                                          A["chain"]), lab)
            finally:
                if p.is_dir():
                    shutil.rmtree(p)
                p.write_bytes(raw)
                os.chmod(p, mode)
        altered(A["rel"]["eligibility"], lambda p: p.write_bytes(
            p.read_bytes() + b" "), "PROD_INPUT_HASH",
            "eligibility manifest altered -> refused")
        altered(A["rel"]["eligibility"] + ".sha256", lambda p: p.write_text(
            "0" * 64 + "  v05lt_nmse_eligibility.json\n"),
            "PROD_INPUT_SIDECAR", "eligibility sidecar mismatch -> refused")
        altered(A["rel"]["data_content"] + ".sha256",
                lambda p: (p.unlink(), p.mkdir()), "PROD_INPUT_SIDECAR",
                "data-content sidecar replaced by a directory -> typed refusal")
        altered("in/zf/" + EV.ZF_FILE + ".sha256", lambda p: p.write_text(
            "0" * 64 + "  x\n"), "ZF_INCOMPLETE", "ZF pair broken -> refused")
        np.save(td / "wrong.npy", _pev_banks()["C"])
        altered(A["rel"]["bank_B"], lambda p: p.write_bytes(
            (td / "wrong.npy").read_bytes()), "PROD_INPUT_HASH",
            "bank B file replaced -> refused")
        # tolerances (valid P5 values)
        for lab, tdoc, code in (
                ("zero tolerance", {"schema": PS.TOLERANCES_SCHEMA,
                                    "tolerances": dict(A["tol"], **{
                                        sorted(A["tol"])[0]: 0.0}),
                                    "provenance": {}}, "PROD_TOLERANCES"),
                ("missing measure", {"schema": PS.TOLERANCES_SCHEMA,
                                     "tolerances": {k: v for k, v in
                                                    sorted(A["tol"].items())[1:]},
                                     "provenance": {}}, "PROD_TOLERANCES"),
                ("wrong schema", {"schema": "x", "tolerances": A["tol"],
                                  "provenance": {}}, "PROD_TOLERANCES"),
                ("extra key", {"schema": PS.TOLERANCES_SCHEMA,
                               "tolerances": A["tol"], "provenance": {},
                               "x": 1}, "PROD_TOLERANCES")):
            n = lab.replace(" ", "_")
            (A["tmp"] / "in" / "p5" / f"{n}.json").write_bytes(
                json.dumps(tdoc).encode())
            rel = dict(A["rel"], dc_tolerances=f"in/p5/{n}.json")
            lp = A["tmp"] / "ev" / f"li_tol_{n}.json"
            PS.make_launch_inputs(lp, A["tmp"], A["stages"](), chain_files={
                r: A["roles"][r] for r in PS.BOUND_ROLES}, inputs=rel,
                commit=A["head"])
            cpt = A["tmp"] / f"chain_tol_{n}.json"
            ch = (cpt, L.make_launch_chain(cpt, A["tmp"], dict(
                A["roles"], launch_inputs=f"ev/li_tol_{n}.json")), A["tmp"])
            _raises_typed(cx, code, lambda ch=ch: _prd_bind(A["stages"](), ch),
                          f"P5 tolerances: {lab} -> refused")
        (A["tmp"] / "in" / "p5" / "nan.json").write_text(
            '{"schema": "%s", "tolerances": {"a": NaN}, "provenance": {}}'
            % PS.TOLERANCES_SCHEMA)
        rel = dict(A["rel"], dc_tolerances="in/p5/nan.json")
        lp = A["tmp"] / "ev" / "li_tol_nan.json"
        PS.make_launch_inputs(lp, A["tmp"], A["stages"](), chain_files={
            r: A["roles"][r] for r in PS.BOUND_ROLES}, inputs=rel,
            commit=A["head"])
        cpt = A["tmp"] / "chain_tol_nan.json"
        ch = (cpt, L.make_launch_chain(cpt, A["tmp"], dict(
            A["roles"], launch_inputs="ev/li_tol_nan.json")), A["tmp"])
        _raises_typed(cx, "PROD_TOLERANCES", lambda: _prd_bind(
            A["stages"](), ch), "P5 tolerances: NaN constant -> refused")
        # launcher level: refusal before anything runs, nothing recorded
        chb = _prd_chain_with(A, "launcher_bad", lambda d: d["bindings"]
                              .update(pin_sha256="0" * 64))
        Lb = A["tmp"] / "camp_bad"
        cx.raises("PROD_LAUNCH_INPUTS_BINDING", lambda: L.Campaign(
            Lb, A["stages"](root=Lb), chain=chb, fixture=True).run("new"),
            "launcher: invalid launch-inputs record refuses the launch")
        cx.ok(not (Lb / L.LEDGER_NAME).exists() and not (Lb / "epoch0")
              .exists(), "refused launch produced nothing and recorded "
              "nothing (as for the chain itself)")
        cx.raises("LAUNCH_CHAIN_MISSING", lambda: L.Campaign(
            Lb, A["stages"](root=Lb), chain=(chb[0], chb[1], A["tmp"] / "x"),
            fixture=True).run("new"), "evidence missing under the chain "
            "root refused by the launcher")

        class _BindBoom:
            def __init__(self, inner):
                self._i = inner

            def __getattr__(self, n):
                if n == "bind_chain":
                    def f(*a):
                        raise RuntimeError("fixture: bind failure")
                    return f
                return getattr(self._i, n)
        cx.raises("LAUNCH_INPUTS_ERROR", lambda: L.Campaign(
            Lb, _BindBoom(A["stages"](root=Lb)), chain=A["chain"],
            fixture=True).run("new"), "unexpected error while binding the "
            "launch inputs -> typed refusal")
        # == refinement 2: fixture overrides refused in production ==========
        P = C.Paths()
        cx.raises("FIXTURE_TARGETS_PRODUCTION", lambda: PS.ProductionStages(
            P, A["tmp"], fixture=dict(A["fx"], root=P.v05lt / "fx")),
            "fixture layout under results/v05lt refused")
        cx.raises("PROD_FIXTURE_KEY", lambda: PS.ProductionStages(
            P, A["tmp"], fixture=dict(A["fx"], mid_indices=[0])),
            "unknown fixture override refused")
        cx.raises("PROD_FIXTURE_KEY", lambda: PS.ProductionStages(
            P, A["tmp"], fixture={"max_epochs": 2}),
            "fixture overrides without a fixture root refused")
        cx.raises("LAUNCH_FIXTURE_IN_PRODUCTION", lambda: L.Campaign(
            td / "pp", A["stages"](), chain=A["chain"]),
            "production launch with fixture-mode stages refused")

        class _Sub(PS.ProductionStages):
            pass
        cx.raises("LAUNCH_STAGES_NOT_PRODUCTION", lambda: L.Campaign(
            td / "pp", _Sub(P, P.data_root), chain=A["chain"]),
            "production launch with a ProductionStages subclass refused")
        cx.raises("LAUNCH_STAGES_MODE_MISMATCH", lambda: L.Campaign(
            td / "pp", PS.ProductionStages(P, P.data_root), chain=A["chain"],
            fixture=True), "fixture campaign with production-mode stages "
            "refused")
        pcfg = PS.registered_config(2026, "lam2", P.v05lt / "2026_lam2",
                                    "a" * 64)
        from seqref_mri.scripts import v05lt_train as T
        T.check_config(pcfg, P)
        cx.ok(pcfg.max_epochs == 100 and pcfg.train_indices is None and
              pcfg.val_indices is None and pcfg.fixture is False,
              "production configuration: full budget and data, accepted by "
              "the trainer's production check")
        pc = PS.ProductionStages(P, P.data_root).campaign_config()
        cx.ok(pc["max_epochs"] == 100 and pc["fixture"] is False and
              pc["train_population"] == "all" and pc["backend"] ==
              PS.BACKEND and PS.ProductionStages(P, P.data_root)
              .config_sha256() != A["st"].config_sha256(),
              "production campaign configuration differs from any fixture "
              "configuration (record binding separates them)")
        cx.ok(PS.check_backend(dict(PS.BACKEND)) == PS.BACKEND,
              "registered backend accepted")
        cx.raises("PROD_BACKEND_MISMATCH", lambda: PS.check_backend(
            dict(PS.BACKEND, threads=4)), "other thread count refused")
        cx.raises("PROD_BACKEND_MISMATCH", lambda: PS.check_backend(
            dict(PS.BACKEND, deterministic=True)),
            "other deterministic setting refused")
        pm = PS.ProductionStages(C.Paths(repo=REPO_ROOT), P.data_root)
        recA = json.loads((A["tmp"] / "ev" / "launch_inputs.json")
                          .read_bytes())
        cx.raises("PROD_INPUT_PATH", lambda: pm._check_inputs(
            recA["inputs"], A["tmp"]), "production: input paths other than "
            "the registered locations refused")
        cx.raises("PROD_PIN_MISMATCH", lambda: pm._check_pin(recA["pin"]),
                  "production: code files not identical to the pinned "
                  "commit refused (uncommitted working tree)")
        # == refinement 5 (addition C): stage-boundary recording ============
        lt = td / "lc"
        lt.mkdir()
        ldata = _syn_train_data(lt)
        lcont = _lch_content(lt)
        lcp, lcsha, _ = _lch_chain(lt)
        lchain = (lcp, lcsha, lt)
        for meth in ("verify_data", "epoch0_done", "epoch0", "post_run"):
            inner = _LaunchStages(lt / meth, ldata, lcont, max_epochs=1)
            bst = _PrdBoom(inner, meth, RuntimeError(f"fixture boom {meth}"))
            cb = L.Campaign(lt / meth, bst, chain=lchain, fixture=True)
            cx.raises("LAUNCH_STAGE_ERROR", lambda cb=cb: cb.run("new"),
                      f"unexpected exception in {meth} -> recorded halt "
                      "LAUNCH_STAGE_ERROR (not re-wrapped)")
            h = [r for r in cb.ledger.records() if r["kind"] == "halt"]
            cx.ok(len(h) == 1 and h[0]["body"]["stage"] == meth and
                  h[0]["body"]["type"] == "RuntimeError" and
                  "fixture boom" in h[0]["body"]["traceback"] and
                  "Traceback" in h[0]["body"]["traceback"],
                  f"{meth}: one halt record with stage, type and traceback")
            bst.armed = False
            cx.raises("LAUNCH_HALT_UNRESOLVED", lambda cb=cb: cb.run(
                "resume"), f"{meth}: fault removed, no decision -> blocked")
            dsha = L.make_campaign_decision(lt / f"{meth}.dec.json",
                                            lt / meth, "retry", "fixture")
            o = cb.run("resume", campaign_decision=(lt / f"{meth}.dec.json",
                                                    dsha))
            cx.ok(o["status"] == "complete", f"{meth}: bound retry completes")
        for lab, exc, code in (
                ("typed", C.V05LTError("FIXTURE_CONFIG", "fixture"),
                 "LAUNCH_CONFIG_FAILED"),
                ("unexpected", RuntimeError("fixture boom expected_config"),
                 "LAUNCH_STAGE_ERROR")):
            nm = f"cfg_{lab}"
            inner = _LaunchStages(lt / nm, ldata, lcont, max_epochs=2)
            inner.hooks["2026_lam0"] = [_lch_kill("commit:before_ledger")]
            bst = _PrdBoom(inner, "expected_config", exc)
            bst.armed = False
            cb = L.Campaign(lt / nm, bst, chain=lchain, fixture=True)
            cx.ok(_lch_expect_kill(lambda cb=cb: cb.run("new")),
                  f"expected_config ({lab}): setup, run interrupted")
            bst.armed = True
            cx.raises(code, lambda cb=cb: cb.run("resume"),
                      f"expected_config {lab} failure -> recorded campaign "
                      f"halt {code}")
            bst.armed = False
            cx.raises("LAUNCH_HALT_UNRESOLVED", lambda cb=cb: cb.run(
                "resume"), f"expected_config {lab}: no decision -> blocked")
            dsha = L.make_campaign_decision(lt / f"{nm}.dec.json", lt / nm,
                                            "retry", "fixture")
            o = cb.run("resume", campaign_decision=(lt / f"{nm}.dec.json",
                                                    dsha))
            cx.ok(o["status"] == "complete", f"expected_config {lab}: bound "
                  "retry completes")
        inner = _LaunchStages(lt / "kill", ldata, lcont, max_epochs=1)
        bst = _PrdBoom(inner, "verify_data", _SimulatedKill("verify_data"))
        cb = L.Campaign(lt / "kill", bst, chain=lchain, fixture=True)
        cx.ok(_lch_expect_kill(lambda: cb.run("new")) and not any(
            r["kind"] == "halt" for r in cb.ledger.records()),
            "process interruption at a stage boundary passes through "
            "unrecorded (unchanged)")
        inner = _LaunchStages(lt / "sess", ldata, lcont, max_epochs=1)
        bst = _PrdBoom(inner, "train", RuntimeError("fixture boom train"))
        cb = L.Campaign(lt / "sess", bst, chain=lchain, fixture=True)
        cx.raises("LAUNCH_SESSION_ERROR", lambda: cb.run("new"),
                  "unexpected exception in training keeps LAUNCH_SESSION_ERROR")
        h = [r for r in cb.ledger.records() if r["kind"] == "halt"][-1]
        cx.ok("fixture boom train" in h["body"].get("traceback", ""),
              "session-error halt now carries the traceback")
        # == pin: complete execution-source set (review rev 1) ==============
        src = PS.source_identity(REPO_ROOT)
        cx.ok(any(k.endswith("src/free_flow_runtime.py") for k in src) and
              any("/tdiag/" in k for k in src) and
              any(k.endswith("v05lt_prodstages.py") for k in src) and
              not any("/results/" in k or "/.venv" in k for k in src),
              "pinned source set covers model, operator, data, metric and "
              "launcher code (not only the launcher scripts)",
              detail=f"{len(src)} files")
        cp_ = td / "srccopy"
        for rel in src:
            (cp_ / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / rel, cp_ / rel)
        stc = A["stages"](repo=cp_)
        lic = PS.make_launch_inputs(
            A["tmp"] / "ev" / "li_srccopy.json", A["tmp"], stc,
            chain_files={r: A["roles"][r] for r in PS.BOUND_ROLES},
            inputs=A["rel"], commit=A["head"])
        chc = (A["tmp"] / "chain_srccopy.json", L.make_launch_chain(
            A["tmp"] / "chain_srccopy.json", A["tmp"], dict(
                A["roles"], launch_inputs="ev/li_srccopy.json")), A["tmp"])
        cx.ok(_prd_bind(A["stages"](repo=cp_), chc)["launch_inputs_sha256"]
              == lic, "record over the copied source tree binds")
        ffr_ = cp_ / "seqref_mri" / "src" / "free_flow_runtime.py"
        raw_ = ffr_.read_bytes()
        ffr_.write_bytes(raw_ + b"\n# changed\n")
        five = {k: src[k] for k in src if "/scripts/v05lt_" in k}
        cx.ok(all(C.sha256_file(cp_ / k) == v for k, v in five.items()),
              "launcher-listed scripts unchanged")
        cx.raises("PROD_PIN_MISMATCH", lambda: _prd_bind(
            A["stages"](repo=cp_), chc), "changed scientific dependency "
            "(free_flow_runtime.py) refused while the launcher scripts are "
            "unchanged")
        ffr_.write_bytes(raw_)
        (cp_ / "seqref_mri" / "src" / "zz_added.py").write_text("X = 1\n")
        cx.raises("PROD_PIN_MISMATCH", lambda: _prd_bind(
            A["stages"](repo=cp_), chc), "added source file refused")
        (cp_ / "seqref_mri" / "src" / "zz_added.py").unlink()
        cfg_ = sorted(k for k in src if "/configs/" in k)
        if cfg_:
            cf = cp_ / cfg_[0]
            craw = cf.read_bytes()
            cf.write_bytes(craw + b" ")
            cx.raises("PROD_PIN_MISMATCH", lambda: _prd_bind(
                A["stages"](repo=cp_), chc), "changed configuration file "
                "refused")
            cf.write_bytes(craw)
        cx.ok(_prd_bind(A["stages"](repo=cp_), chc)["launch_inputs_sha256"]
              == lic, "restored tree binds again")
        # production git check (HEAD blobs) on a committed copy
        gr = td / "gitcopy"
        shutil.copytree(cp_, gr)

        def g(*a):
            r = subprocess.run(["git", "-C", str(gr), "-c", "user.email=f@x",
                                "-c", "user.name=fixture", *a],
                               capture_output=True, text=True)
            C.require(r.returncode == 0, "FIXTURE_GIT", r.stderr)
            return r.stdout.strip()
        g("init", "-q")
        g("add", "-A")
        g("commit", "-q", "-m", "fixture pin")
        head0 = g("rev-parse", "HEAD")
        pg = PS.ProductionStages(C.Paths(repo=gr), gr)
        pin0 = {"commit": head0, "code": pg.code_identity()}
        pg._check_pin(pin0)
        cx.ok(True, "production pin check passes on the clean committed tree")
        gf = gr / "seqref_mri" / "src" / "free_flow_runtime.py"
        graw = gf.read_bytes()
        gf.write_bytes(graw + b"\n# changed\n")
        cx.raises("PROD_PIN_MISMATCH", lambda: pg._check_pin(pin0),
                  "production: changed dependency after the pin refused")
        cx.raises("PROD_PIN_MISMATCH", lambda: pg._check_pin(
            {"commit": head0, "code": pg.code_identity()}),
            "production: record matching the changed tree but not the "
            "pinned commit refused (HEAD blob check)")
        gf.write_bytes(graw)
        (gr / "seqref_mri" / "src" / "zz_untracked.py").write_text("Y = 2\n")
        cx.raises("PROD_PIN_MISMATCH", lambda: pg._check_pin(
            {"commit": head0, "code": pg.code_identity()}),
            "production: untracked source file refused")
        (gr / "seqref_mri" / "src" / "zz_untracked.py").unlink()
        g("commit", "-q", "--allow-empty", "-m", "later")
        cx.raises("PROD_PIN_MISMATCH", lambda: pg._check_pin(pin0),
                  "production: HEAD moved past the pin refused")
        # loaded modules must lie in the pinned set
        import types
        fake = types.ModuleType("seqref_mri.scripts.zz_fake_loaded")
        fake.__file__ = str(REPO_ROOT / "seqref_mri" / "scripts" /
                            "zz_fake_loaded.py")
        sys.modules[fake.__name__] = fake
        try:
            cx.raises("PROD_PIN_MISMATCH", lambda: stA._check_loaded(
                stA.inputs["pin_code"]), "module loaded from outside the "
                "pinned set refused")
        finally:
            del sys.modules[fake.__name__]
        alt = dict(stA.inputs["pin_code"])
        k_ = [k for k in alt if k.endswith("src/free_flow_runtime.py")][0]
        alt[k_] = "0" * 64
        cx.raises("PROD_PIN_MISMATCH", lambda: stA._check_loaded(alt),
                  "loaded module whose hash differs from the pin refused")
        stA._check_loaded(stA.inputs["pin_code"])
        cx.ok(True, "loaded modules lie in the pinned set")
        # == symlinks in the source tree / loaded modules (review rev 2) ===
        outside = td / "outside_src"
        outside.mkdir()
        (outside / "evil.py").write_text("Z = 3\n")
        pin_cp = PS.source_identity(cp_)
        sd = cp_ / "seqref_mri" / "src" / "zz_extdir"
        os.symlink(outside, sd, target_is_directory=True)
        try:
            cx.raises("PROD_PIN_MISMATCH", lambda: PS.source_identity(cp_),
                      "directory symlink under seqref_mri/src pointing "
                      "outside the repository refused (source identity)")
            cx.raises("PROD_PIN_MISMATCH", lambda: _prd_bind(
                A["stages"](repo=cp_), chc), "binding refused with that "
                "directory symlink present")
        finally:
            sd.unlink()
        fl = cp_ / "seqref_mri" / "src" / "zz_link.py"
        os.symlink(outside / "evil.py", fl)
        try:
            cx.raises("PROD_PIN_MISMATCH", lambda: PS.source_identity(cp_),
                      "file symlink under seqref_mri/src refused")
        finally:
            fl.unlink()
        cx.ok(PS.source_identity(cp_) == pin_cp, "tree identity restored")
        head_now = g("rev-parse", "HEAD")
        pin_now = {"commit": head_now, "code": pg.code_identity()}
        pg._check_pin(pin_now)
        cx.ok(True, "production pin check passes on the clean committed "
              "tree (current HEAD)")
        gsd = gr / "seqref_mri" / "src" / "zz_extdir"
        os.symlink(outside, gsd, target_is_directory=True)
        try:
            cx.raises("PROD_PIN_MISMATCH", lambda: pg._check_pin(pin_now),
                      "production: directory symlink to an external "
                      "location added after the pin refused")
        finally:
            gsd.unlink()
        ext = types.ModuleType("seqref_mri.src.zz_extdir.evil")
        ext.__file__ = str((outside / "evil.py").resolve())
        sys.modules[ext.__name__] = ext
        try:
            cx.raises("PROD_PIN_MISMATCH", lambda: stA._check_loaded(
                stA.inputs["pin_code"]), "seqref_mri module executing from "
                "an external location (as loaded through a symlinked "
                "directory) refused")
        finally:
            del sys.modules[ext.__name__]
        tp = types.ModuleType("zz_third_party_fixture")
        tp.__file__ = str((outside / "evil.py").resolve())
        sys.modules[tp.__name__] = tp
        try:
            stA._check_loaded(stA.inputs["pin_code"])
            cx.ok(True, "modules outside the package and not named "
                  "seqref_mri (third-party) are not pin-checked")
        finally:
            del sys.modules[tp.__name__]
        # == excluded folders inside the package (host venv; revision 4) ===
        venv = cp_ / "seqref_mri" / ".venv_mri" / "lib" / "site"
        venv.mkdir(parents=True)
        (venv / "tp.py").write_text("T = 1\n")
        cx.ok(PS.source_identity(cp_) == pin_cp, "excluded venv folder "
              "inside the package is not part of the source identity")
        stv = A["stages"](repo=cp_)

        def table(**kw):
            out = {}
            for n, f in kw.items():
                m = types.ModuleType(n)
                m.__file__ = str(f)
                out[n.replace("__", ".")] = m
            return out
        stv._check_loaded(pin_cp, root=cp_, modules=table(
            zz_tp=venv / "tp.py", seqref_mri__src__free_flow_runtime=cp_ /
            "seqref_mri" / "src" / "free_flow_runtime.py"))
        cx.ok(True, "third-party module in the excluded venv folder inside "
              "the package accepted; pinned seqref_mri module accepted")
        cx.raises("PROD_PIN_MISMATCH", lambda: stv._check_loaded(
            pin_cp, root=cp_, modules=table(seqref_mri__zz_venv=venv /
                                            "tp.py")),
            "seqref_mri module loaded from the excluded venv folder refused")
        tov = cp_ / "seqref_mri" / "src" / "zz_tov"
        os.symlink(venv, tov, target_is_directory=True)
        try:
            cx.raises("PROD_PIN_MISMATCH", lambda: PS.source_identity(cp_),
                      "symlink in the source tree resolving into the "
                      "excluded venv folder refused (source identity)")
            cx.raises("PROD_PIN_MISMATCH", lambda: stv._check_loaded(
                pin_cp, root=cp_, modules=table(zz_detour=tov / "tp.py")),
                "module loaded from the source tree through a symlink into "
                "the excluded venv folder refused")
            cx.raises("PROD_PIN_MISMATCH", lambda: stv._check_loaded(
                pin_cp, root=cp_, modules=table(
                    seqref_mri__src__zz_tov__tp=tov / "tp.py")),
                "seqref_mri module through a symlink into the excluded "
                "venv folder refused")
        finally:
            tov.unlink()
        cx.raises("PROD_PIN_MISMATCH", lambda: stv._check_loaded(
            pin_cp, root=cp_, modules=table(seqref_mri__zz_ext=outside /
                                            "evil.py")),
            "seqref_mri module from an external location refused (copied "
            "tree)")
        shutil.rmtree(cp_ / "seqref_mri" / ".venv_mri")
        # PRB child-result handling (revision 4)
        try:
            _child_payload({"error": {"code": "FIXTURE_CHILD_CODE",
                                      "message": "fixture message"}}, "prb")
            cx.ok(False, "child error must raise", "NO_ERROR_RAISED")
        except FixtureFailure as exc:
            cx.ok(exc.code == "FIXTURE_CHILD_CODE" and "fixture message" in
                  str(exc), "child error: original code and message preserved",
                  detail=str(exc))
        try:
            _child_payload({}, "prb")
            cx.ok(False, "missing payload must raise", "NO_ERROR_RAISED")
        except FixtureFailure as exc:
            cx.ok(exc.code == "PRB_RESULT_MISSING", "missing child result "
                  "payload refused explicitly", detail=str(exc))
        cx.ok(_child_payload({"prb": {"x": 1}}, "prb") == {"x": 1},
              "child payload returned when present")
        # == refinement 6: isolation through the wiring =====================
        import random
        I = _prd_layout(td / "I", max_epochs=3)
        stI = I["st"]
        _prd_bind(stI, I["chain"])
        random.seed(7)
        np.random.seed(7)
        import torch
        torch.manual_seed(7)
        r0 = PS._rng_save()
        res6 = stI.epoch0(2026)
        cx.ok(PS._rng_equal(r0, PS._rng_save()) and stI.epoch0_done(2026),
              "production epoch-0 stage (checkpoint + evaluation) leaves the "
              "Python / NumPy / torch RNG states unchanged",
              detail=f"changed inside and restored: "
              f"{res6['rng_changed_inside']}")
        stI.train(2026, "lam2", "start")
        J = _prd_layout(td / "J", max_epochs=3)
        stJ = J["st"]
        _prd_bind(stJ, J["chain"])
        E0.create(J["tmp"] / "camp" / "epoch0", 2026, REPO_ROOT,
                  model_factory=build_stub)
        stJ.train(2026, "lam2", "start")
        ri, rj = I["tmp"] / "camp" / "2026_lam2", J["tmp"] / "camp" / \
            "2026_lam2"
        fi, fj = _final_state(ri), _final_state(rj)
        cx.ok(_commits(ri) == _commits(rj) and fi[0] == fj[0] and
              fi[2] == fj[2] and all(torch.equal(a, b) for a, b in
                                     zip(fi[3], fj[3])),
              "training after the production epoch-0 stage (with its "
              "evaluation) is bitwise identical to training without it")


# -- addition B: real architecture through the production stage class ------
PRB_MIN_FREE = 24e9


def _prb_layout(tmp: Path, host: bool, data_root) -> dict:
    """Fixture layout of REAL data (host: 2 registered validation volumes
    holding cases 0 and 1, 2 registered training volumes, copied into tmp;
    local: synthetic volumes) with the real gate-2a style writers."""
    from seqref_mri.scripts import v05lt_datahash as DH
    from seqref_mri.scripts import v05lt_eligibility as E
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_launch as L
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.scripts import v05lt_prodstages as PS
    from seqref_mri.src import free_flow_runtime as ffr
    tmp = Path(tmp)
    ind = tmp / "in" / "inputs"
    ind.mkdir(parents=True)
    if host:
        p = C.Paths(repo=REPO_ROOT)
        sdoc = C.load_split_manifest(
            p.inputs / "v05lt_split_manifest.json",
            C.REGISTERED_INPUTS["v05lt_split_manifest.json"])
        cdoc = json.loads((p.inputs / "v05lt_case_manifest.json").read_bytes())
        vfiles = [c["file"] for c in cdoc["cases"][:2]]
        val = [r for r in sdoc["validation"] if r["file"] in vfiles]
        train = sdoc["train"][:2]
        root = tmp / "in" / "data"
        for r in train + val:
            (root / r["file"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(data_root) / r["file"], root / r["file"])
        tr_ids = [r["volume_id"] for r in train]
        va_ids = [r["volume_id"] for r in val]
        doc = {"schema": C.SPLIT_SCHEMA, "train": train, "validation": val,
               "train_ids_sha256": C._ids_sha(tr_ids),
               "validation_ids_sha256": C._ids_sha(va_ids),
               "holdout_overlap": 0}
        mp = tmp / "in" / "split.json"
        mp.write_bytes(C.canonical_json(doc))
        ex = C.SplitExpect(len(train), len(val), 199,
                           sum(r["n_slices"] for r in train),
                           sum(r["n_slices"] for r in val),
                           C._ids_sha(tr_ids), C._ids_sha(va_ids))
        hold = C.holdout_ids(p.v02_manifests)
        s_ref = C.verify_p0s(p.p0s)
        p4 = ffr.load_p4s2_parent(str(REPO_ROOT / P4_REL))
        cases = [c for c in cdoc["cases"] if c["file"] in vfiles]
        train_idx = list(range(40))
        for n in ("v05lt_bank_20261023.npy", "v05lt_bank_20261024.npy"):
            shutil.copyfile(p.inputs / n, ind / n)
    else:
        root, mp, _, ex = synth_split(tmp / "in", SYN_VOLS, SYN_TRAIN, SYN_VAL)
        hold, s_ref, p4 = SYN_HOLDOUT, 1.0, _unit_p4()
        cases = None
        train_idx = None
        bk = _pev_banks()
        np.save(ind / "v05lt_bank_20261023.npy", bk["B"])
        np.save(ind / "v05lt_bank_20261024.npy", bk["C"])
    msha = C.sha256_file(mp)
    sd = C.load_split_manifest(mp, msha, ex)
    _, vv = C.build_split_views(root, sd, hold, ex)
    recs = E.compute_records(vv, s_ref)
    if cases is None:
        tmpdoc = E.build_manifest(recs, {"validation": sd["validation"]},
                                  {"cases": []}, {"p": "fixture"})
        cases = [{"case": j, "file": r["file"], "slice_index": r["slice_index"]}
                 for j, r in enumerate(x for x in tmpdoc["slices"]
                                       if x["midpoint"])]
    cdoc2 = {"schema": "fixture-case-manifest", "cases": cases}
    C.write_durable_new(ind / "v05lt_case_manifest.json",
                        C.canonical_json(cdoc2))
    edoc = E.build_manifest(recs, {"validation": sd["validation"]}, cdoc2,
                            {"p": "fixture"})
    ep = ind / "v05lt_nmse_eligibility.json"
    esha = C.write_durable_new(ep, C.canonical_json(edoc))
    C.write_durable_new(Path(str(ep) + ".sha256"),
                        f"{esha}  {ep.name}\n".encode())
    el = M.eligibility_index(edoc)
    DH.generate(mp, msha, hold, root, ind / DH.OUT_NAME, expect=ex)
    sts = EV.build_states(vv, list(range(len(vv))), p4, s_ref)
    zrows = [EV.zf_row(st, el[(st["identity"]["file"],
                               int(st["identity"]["slice_index"]))])
             for st in sts]
    EV.write_zf(tmp / "in" / "zf", zrows, EV.zf_summary(zrows),
                {"fixture": True})
    mids = [i for i in range(len(vv)) if el[vv.entry(i)]["midpoint"]]
    tol = EV.calibrate_dc(EV.build_states(vv, mids, p4, s_ref))["tolerances"]
    (tmp / "in" / "p5").mkdir()
    PS.make_tolerances(tmp / "in" / "p5" / "tol.json", tol, {"fixture": True})
    ev = tmp / "ev"
    ev.mkdir()
    roles = {}
    for r in _PRD_EV:
        (ev / f"{r}.json").write_text(json.dumps({"role": r, "fixture": True}))
        roles[r] = f"ev/{r}.json"
    rel = {"split_manifest": "in/split.json",
           "case_manifest": "in/inputs/v05lt_case_manifest.json",
           "bank_B": "in/inputs/v05lt_bank_20261023.npy",
           "bank_C": "in/inputs/v05lt_bank_20261024.npy",
           "data_content": f"in/inputs/{DH.OUT_NAME}",
           "eligibility": "in/inputs/v05lt_nmse_eligibility.json",
           "zf": f"in/zf/{EV.ZF_FILE}", "dc_tolerances": "in/p5/tol.json"}
    fx = {"root": tmp / "camp", "max_epochs": RT_EPOCHS, "p4": p4,
          "s_ref": s_ref, "split_expect": ex, "holdout": hold,
          "repo": REPO_ROOT}
    if train_idx is not None:
        fx["train_indices"] = train_idx
    st = PS.ProductionStages(C.Paths(), root, fixture=fx)
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    li = PS.make_launch_inputs(ev / "launch_inputs.json", tmp, st,
                               chain_files={r: roles[r] for r in
                                            PS.BOUND_ROLES},
                               inputs=rel, commit=head)
    roles[PS.LAUNCH_INPUTS_ROLE] = "ev/launch_inputs.json"
    cp = tmp / "chain.json"
    return {"st": st, "chain": (cp, L.make_launch_chain(cp, tmp, roles), tmp),
            "li_sha": li, "n_val": len(vv), "n_mid": len(mids),
            "cases": [c["case"] for c in cases]}


def _prb_child(spec: dict) -> dict:
    """Addition B (fresh process): ProductionStages with the REAL
    architecture (no model factory) on a fixture layout: one seed, its
    epoch-0 stage, runs (host: lam0 and lam2; local: lam2), 3 epochs, each
    post-run stage; timing per stage (provisional)."""
    import time as _time
    import torch
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_launch as L
    from seqref_mri.scripts import v05lt_prodstages as PS
    host = spec["host"]
    tmp = Path(spec["tmp"])
    t = {}
    t0 = _time.monotonic()
    lay = _prb_layout(tmp, host, spec["data_root"])
    t["layout_s"] = round(_time.monotonic() - t0, 1)
    st = lay["st"]
    st.bind_chain(L.verify_launch_chain(*lay["chain"]), lay["chain"][2])
    t0 = _time.monotonic()
    st.verify_data()
    t["verify_data_s"] = round(_time.monotonic() - t0, 1)
    r0 = PS._rng_save()
    t0 = _time.monotonic()
    st.epoch0(2026)
    t["epoch0_s"] = round(_time.monotonic() - t0, 1)
    rng_ok = PS._rng_equal(r0, PS._rng_save())
    e0_ok = st.epoch0_done(2026)
    arms = ("lam0", "lam2") if host else ("lam2",)
    runs = {}
    for a in arms:
        t0 = _time.monotonic()
        st.train(2026, a, "start")
        t[f"train_{a}_s"] = round(_time.monotonic() - t0, 1)
        t0 = _time.monotonic()
        pr = st.post_run(2026, a)
        t[f"post_run_{a}_s"] = round(_time.monotonic() - t0, 1)
        t0 = _time.monotonic()
        pr2 = st.post_run(2026, a)
        t[f"post_run_reverify_{a}_s"] = round(_time.monotonic() - t0, 1)
        docs = [EV.load_posteval(st.root / k) for k in pr["artefacts"]
                if k.startswith("posteval/")]
        runs[a] = {"tasks": pr["tasks"], "same_on_reverify":
                   pr2["artefacts"] == pr["artefacts"],
                   "decodes_ok": all(d["midpoint"]["decodes"] == {
                       b: 128 * lay["n_mid"] for b in d["banks"]}
                       for d in docs),
                   "roles_covered": sorted(r for x in pr["tasks"]
                                           for r in x["roles"])}
    return {"rng_ok": rng_ok, "e0_ok": e0_ok, "runs": runs, "timing": t,
            "n_val": lay["n_val"], "n_mid": lay["n_mid"],
            "cases": lay["cases"], "threads": torch.get_num_threads(),
            "backend": PS.backend_now()}


def _child_payload(res: dict, key: str):
    """Result payload of a spawned child, or a FixtureFailure preserving the
    child's original error code and message; a missing payload without an
    error is refused explicitly (no KeyError)."""
    if isinstance(res, dict) and key in res:
        return res[key]
    err = res.get("error") if isinstance(res, dict) else None
    if isinstance(err, dict) and err.get("code"):
        logger.error("%s child refused: %s: %s", key, err["code"],
                     err.get("message", ""))
        raise FixtureFailure(err["code"], f"{key} child: "
                             f"{err.get('message', '')}")
    logger.error("%s child returned no result payload", key)
    raise FixtureFailure(f"{key.upper()}_RESULT_MISSING",
                         f"{key}: no result payload and no error reported")


def prb_group(cx: Ctx) -> None:
    """PRB (addition B): real architecture through the production stage
    class on a fixture layout, in a fresh process. Host: real data (2
    registered training + 2 validation volumes holding cases 0 and 1), one
    seed, lam0 and lam2, 3 epochs. Local pre-check: synthetic data, lam2
    only (disk). Timing is provisional (P4 input only)."""
    tmpbase = Path(tempfile.gettempdir())
    free = shutil.disk_usage(tmpbase).free
    need = PRB_MIN_FREE if cx.host else 10e9
    cx.ok(free >= need, "free space for the real-architecture runs",
          "FIXTURE_DISK_SPACE", f"{free / 1e9:.1f} GB free, need "
          f"{need / 1e9:.0f} GB")
    if cx.host:
        cx.phase = "host"
    with tempfile.TemporaryDirectory() as t:
        res = _run_eval_child(cx, Path(t), {"kind": "prb",
                                            "tmp": str(Path(t) / "prb")},
                              "prb")
    e = _child_payload(res, "prb")
    want_arms = ["lam0", "lam2"] if cx.host else ["lam2"]
    cx.ok(sorted(e["runs"]) == want_arms and e["e0_ok"] and e["rng_ok"],
          "real architecture: epoch-0 stage complete and verified; RNG states "
          "unchanged across it", "PRB_FAILED", str({k: e[k] for k in (
              "e0_ok", "rng_ok")}))
    for a, r in e["runs"].items():
        cx.ok(r["same_on_reverify"] and r["decodes_ok"] and
              r["roles_covered"] == ["epoch3", "final", "selected"],
              f"real architecture {a}: trained 3 epochs, final weights and "
              "post-run records (planner tasks) written and verified; "
              "re-verification reuses them", "PRB_FAILED", str(r))
    cx.record("PRB timing (provisional; fixture layout; P4 input only)",
              {"timing": e["timing"], "n_val": e["n_val"],
               "n_mid": e["n_mid"], "cases": e["cases"],
               "backend": e["backend"],
               "tasks": {a: r["tasks"] for a, r in e["runs"].items()}})


# ---------------------------------------------------------------------------
# I1.9 c3 (SEQREF-MRI-V05LTI19C3 v0.1): HRN, HNC, HRD (synthetic by design;
# synthetic holdout .h5 files in temporary folders), HRB (addition B: real
# architecture on synthetic holdout files). No real holdout data is read.
# ---------------------------------------------------------------------------
HRN_SLICES = 14                  # slices per synthetic holdout volume


def _hrn_setup(td: Path, **campaign) -> dict:
    """b3 analysis chain (stub campaign, selection manifest, frozen holdout
    configuration) + approved chain manifest, PRD launch layout (fixture
    ProductionStages, launch chain), synthetic holdout volumes for the b3
    synthetic manifest, and an approval record (fixture only)."""
    from seqref_mri.scripts import v05lt_evidence_check as EVC
    from seqref_mri.scripts import v05lt_holdout as HO
    B = _b3_setup(td / "b", **campaign)
    root = B["root"]

    def rel(p):
        return str(Path(p).relative_to(root))
    roles = {"config": rel(B["config"]),
             "difficult": rel(B["layout"]["difficult"]),
             "analysis": rel(B["path"]), "report_dir": rel(B["report"]),
             "selection": rel(B["selection"][0]),
             "holdout_config": rel(B["hconfig"][0])}
    chm = td / "chain.json"
    chsha = EVC.make_chain_manifest(chm, root, roles)
    P = _prd_layout(td / "P")
    hdata = td / "hdata"
    for k, e in enumerate(B["man"]["doc"]["entries"]):
        write_volume(hdata / e["file"], HRN_SLICES, 5000 + k)
    ap = td / "approval.json"
    apsha = HO.make_approval(
        ap, root=root, launch_chain=P["chain"][:2],
        analysis_chain=(chm, chsha), manifest_path=B["man"]["manifest"],
        statement="fixture approval: synthetic holdout files only (c3)",
        fixture=True, launch_root=P["tmp"])
    kw = dict(root=root, launch_chain=P["chain"][0], analysis_chain=chm,
              manifest_path=B["man"]["manifest"], stages=P["st"],
              data_root=hdata, launch_root=P["tmp"])
    return {"B": B, "P": P, "roles": roles, "chain": (chm, chsha),
            "hdata": hdata, "approval": (ap, apsha), "kw": kw, "td": td}


def _hrn_forged(H: dict, name: str, edit) -> tuple:
    """Self-consistent approval record (own sha256) with edit(doc) applied:
    tests the bindings, not the hash check."""
    doc = json.loads(H["approval"][0].read_bytes())
    edit(doc)
    p = H["td"] / f"approval_{name}.json"
    return p, C.write_durable_new(p, C.canonical_json(doc))


def _hrn_run(H: dict, out: Path, approval=None, **over):
    from seqref_mri.scripts import v05lt_holdout as HO
    kw = dict(H["kw"], **over)
    return HO.run_holdout(approval or H["approval"], out_dir=out, **kw)


def _hrn_kinds(out: Path) -> list:
    from seqref_mri.scripts import v05lt_holdout as HO
    return [r["kind"] for r in C.Ledger(out / HO.LEDGER_NAME).records()]


def _hrn_holdout(H: dict, res: dict) -> dict:
    """b3 loader input from the runner's result (records -> path, sha)."""
    man = H["B"]["man"]
    return {"manifest": man["manifest"],
            "manifest_file_sha256": man["manifest_file_sha256"],
            "manifest_content_sha256": man["manifest_content_sha256"],
            "records": {k: {"path": Path(v["path"]), "sha256": v["sha256"]}
                        for k, v in res["records"].items()}}


def _hrn_rewrite(d: Path, name: str, edit, sums: bool = True) -> None:
    """Edit one record of a published evaluation directory; with sums=True
    the sidecar and SHA256SUMS are rewritten (only the edited property is
    wrong)."""
    for f in d.iterdir():
        os.chmod(f, 0o644)
    p = d / name
    doc = json.loads(p.read_bytes())
    edit(doc)
    b = (json.dumps(doc, sort_keys=True, allow_nan=False) + "\n").encode()
    p.write_bytes(b)
    if sums:
        (d / f"{name}.sha256").write_text(f"{C.sha256_bytes(b)}  {name}\n")
        (d / "SHA256SUMS").write_text("".join(
            f"{C.sha256_file(f)}  {f.name}\n" for f in sorted(d.iterdir())
            if f.name != "SHA256SUMS"))


class _HrnView:
    """FIXTURE-ONLY stand-in view: the real holdout view with one item
    property altered (width / mask / seed / order tests of check_widths)."""

    def __init__(self, view, alter):
        self.view, self.alter = view, alter

    def __len__(self):
        return len(self.view)

    def entry(self, i):
        return self.alter("entry", i, self.view.entry(i))

    def __getitem__(self, i):
        return self.alter("item", i, self.view[i])


def hrn_group(cx: Ctx) -> None:
    """HRN: holdout runner core -- gate, scope, coincident reuse, width
    check, write-once records, reuse, refusals, round trip through the b3
    loader into the holdout analysis (synthetic holdout files only)."""
    import copy
    import time
    import torch
    from seqref_mri.scripts import v05lt_analysis as A
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_holdout as HO
    from seqref_mri.scripts import v05lt_prodstages as PS
    from seqref_mri.scripts import v05lt_results as RS
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        H = _hrn_setup(td)
        B = H["B"]
        sel = B["sel"]
        tasks = HO.plan_tasks(sel)
        coinc = [t for t in tasks if len(t["roles"]) == 2]
        cx.ok(sum(len(t["roles"]) for t in tasks) == 12 and
              all(set(t["roles"]) <= {"selected", "epoch3"} for t in tasks)
              and len(tasks) == 12 - len(coinc),
              "plan: selected + epoch 3 of the six runs only (12 roles); "
              "final and epoch 0 never planned; coincident checkpoints "
              "share one task", detail=f"{len(tasks)} tasks, "
              f"{len(coinc)} coincident")
        cx.record("HRN selected epochs", {k: r["selected"]["epoch"]
                                          for k, r in sel["runs"].items()})
        s2 = copy.deepcopy(sel)
        k0 = sorted(s2["runs"])[0]
        s2["runs"][k0]["selected"] = dict(s2["runs"][k0]["epoch3"])
        t2 = [t for t in HO.plan_tasks(s2) if f"{t['arm']}|{t['seed']}" == k0]
        cx.ok(len(t2) == 1 and t2[0]["roles"] == ["selected", "epoch3"] and
              t2[0]["epoch"] == 3,
              "refinement 3: selected = epoch 3 (same model) -> ONE task "
              "carrying both roles")
        s3 = copy.deepcopy(s2)
        s3["runs"][k0]["selected"]["sha256"] = "0" * 64
        cx.raises("HOLDOUT_PLAN_CONFLICT", lambda: HO.plan_tasks(s3),
                  "same model state with a different checkpoint file refused")
        for bad in (["final"], ["epoch0"], ["selected", "final"], []):
            cx.raises("HOLDOUT_SCOPE", lambda b=bad: HO.task_scope_check(
                {"roles": b}), f"scope refusal: roles {bad}")
        # ---- full run --------------------------------------------------
        out = td / "out"
        t0 = time.perf_counter()
        res = _hrn_run(H, out)
        dt = time.perf_counter() - t0
        kinds = _hrn_kinds(out)
        cx.ok(res["status"] == "complete" and len(res["records"]) == 12 and
              kinds[:2] == ["gate_passed", "data_verified"] and
              kinds.count("task_done") == len(tasks) and
              kinds[-1] == "complete" and "halt" not in kinds,
              "run: gate recorded, data verified, every task evaluated once, "
              "12 role records, index, complete", detail=f"{dt:.1f} s")
        docs = {k: json.loads(Path(v["path"]).read_bytes())
                for k, v in res["records"].items()}
        g = HO.verify_gate(H["approval"], **{k: v for k, v in H["kw"].items()
                                             if k != "data_root"})
        seeds = g.manifest["seeds"]
        n = len(g.manifest["rows"])
        ok_rec = all(
            d["schema"] == RS.HOLDOUT_SCHEMA and d["fixture"] is True and
            d["approval_sha256"] == H["approval"][1] and
            d["masks"]["widths"] == [96] * n and
            d["masks"]["sha256"] == RS.holdout_masks_sha256(seeds, [96] * n)
            and d["decodes"] == EV.BANK_ROWS * n and
            d["bank"]["array_sha256"] == EV.REGISTERED_BANK_ARRAY["primary"]
            and all("nll_per_dim" in r["r0"] and "nll_per_dim" not in r["r1"]
                    and math.isfinite(r["r0"]["nll_per_dim"]) and
                    math.isfinite(r["r1"]["nmse_u"]) for r in d["rows"]) and
            d["rng"]["restored"] is True
            for d in docs.values())
        cx.ok(ok_rec, "records: b3 schema, bound to the approval; loaded "
              "widths (96) and recomputed mask identity; 128 primary-bank "
              "decodes per slice; R0 NLL per dim; AM4-B NMSE_u finite; RNG "
              "restored")
        by_eval = {}
        for k, d in docs.items():
            by_eval.setdefault(d["evaluation"]["name"], []).append(d)
        cx.ok(all(len(v) == len(v[0]["evaluation"]["roles"]) and
                  all(x["rows"] == v[0]["rows"] for x in v)
                  for v in by_eval.values()) and len(by_eval) == len(tasks),
              "one evaluation per distinct model; every role record of an "
              "evaluation carries the same rows", detail=f"{len(by_eval)} "
              "evaluations")
        # ---- round trip through the b3 loader and analysis -------------
        hold = _hrn_holdout(H, res)
        lr = RS.load_holdout_records(hold, B["camp"], sel, RS.Identities(),
                                     fixture=True, mask_seeds_sha256=B["msha"])
        cx.ok(len(lr["records"]) == 12 and lr["masks_sha256"] ==
              RS.holdout_masks_sha256(seeds, [96] * n),
              "round trip: the accepted b3 loader verifies all 12 records "
              "(bindings, manifest, frozen seed identity, mask recomputation)")
        ho = td / "hold_analysis.json"
        A.run_analysis(B["layout"], B["hconfig"][0], B["hconfig"][1], ho,
                       holdout=hold, selection=B["selection"])
        hd = json.loads(ho.read_bytes())
        hc = [c for c in hd["analysis"]["comparisons"]
              if c["family"].startswith("holdout")]
        cx.ok(len(hc) == 21 and all(c["status"] == "ok" for c in hc),
              "round trip: the holdout analysis estimates all 21 registered "
              "comparisons from the runner's records")
        # ---- reuse -----------------------------------------------------
        res2 = _hrn_run(H, out)
        k2 = _hrn_kinds(out)[len(kinds):]
        cx.ok(res2["index_sha256"] == res["index_sha256"] and
              k2.count("task_reused") == len(tasks) and
              "task_done" not in k2 and k2[-1] == "complete",
              "second run: complete outputs verified and reused (no "
              "evaluation), same index")
        # ---- width / mask checks (item 27) ------------------------------
        view = HO.holdout_view(g, H["hdata"])

        def alt(kind_, fn):
            def a(kind, i, v):
                return fn(i, v) if kind == kind_ else v
            return a

        def flip(i, it):
            it = dict(it)
            m = it["mask"].clone()
            m[0] = ~m[0]
            it["mask"] = m
            return it

        def seed(i, it):
            it = dict(it, meta=dict(it["meta"]))
            it["meta"]["mask_seed"] = int(it["meta"]["mask_seed"]) + 1
            return it

        def narrow(i, it):
            it = dict(it)
            it["y"] = it["y"][..., :95]
            return it

        def order(i, e):
            return view.entry((i + 1) % len(view))
        w = HO.check_widths(g, view)
        cx.ok(w["widths"] == [96] * n and w["masks_sha256"] ==
              RS.holdout_masks_sha256(seeds, w["widths"]),
              "item 27: loaded k-space width per slice; loaded mask = "
              "make_cartesian_mask(width, canonical seed); identities agree")
        for code, fn, lab in (
                ("HOLDOUT_WIDTH_MISMATCH", alt("item", flip),
                 "loaded mask != registered mask"),
                ("HOLDOUT_MASK_MISMATCH", alt("item", seed),
                 "loaded mask seed != canonical seed"),
                ("HOLDOUT_WIDTH_MISMATCH", alt("item", narrow),
                 "loaded k-space width != mask width"),
                ("HOLDOUT_DATA_ORDER", alt("entry", order),
                 "view order != manifest order")):
            cx.raises(code, lambda f=fn: HO.check_widths(g, _HrnView(view, f)),
                      f"item 27 refusal: {lab} (fixture stand-in view)")
        # RNG isolation: a model that DRAWS from the torch RNG while decoding
        # (fixture-only wrapper) -> states restored, consumption recorded
        from seqref_mri.scripts.v05lt_stub_train import build_stub
        t0_ = tasks[0]
        mdl, vck = EV.load_trained_checkpoint(
            B["root"] / t0_["file"], t0_["sha256"], t0_["model_state_sha256"],
            seed=t0_["seed"], arm=t0_["arm"], epoch=t0_["epoch"],
            model_factory=build_stub)
        dec0 = mdl.decode_scalars

        def drawing(*a, **k):
            torch.rand(1)
            return dec0(*a, **k)
        mdl.decode_scalars = drawing
        sts = EV.build_states(view, list(range(len(view))), _unit_p4(), 1.0)
        rng0 = PS._rng_save()
        _, info = HO.evaluate_task(mdl, vck, sts, g,
                                   g.stages.banks()["primary"])
        cx.ok(info["rng"]["consumed_during_eval"] is True and
              PS._rng_equal(rng0, PS._rng_save()),
              "RNG isolation: an evaluation that draws from the torch RNG "
              "leaves the RNG states unchanged (restored) and records it")
        del mdl
        cx.raises("HOLDOUT_GATE_REQUIRED", lambda: HO.holdout_view(
            None, H["hdata"]), "holdout view without a verified gate refused")
        cx.raises("HOLDOUT_GATE_REQUIRED", lambda: HO.check_widths(
            dataclasses_replace_token(g), view), "forged gate object refused")
        # ---- approval / chain / binding refusals -----------------------
        o = td / "o_ref"
        cases = [
            ("HOLDOUT_APPROVAL_UNAPPROVED", (H["approval"][0], "0" * 64),
             "approval sha256 != externally approved value"),
            ("HOLDOUT_APPROVAL_UNAPPROVED", (H["approval"][0], None),
             "no approved sha256")]
        for code, edit, lab in (
                ("HOLDOUT_APPROVAL_BINDING", lambda d: d.update(
                    selection_sha256="1" * 64), "selection manifest"),
                ("HOLDOUT_APPROVAL_BINDING", lambda d: d.update(
                    holdout_config_sha256="2" * 64), "frozen configuration"),
                ("HOLDOUT_APPROVAL_BINDING", lambda d: d.update(
                    validation_config_sha256="3" * 64),
                 "validation configuration"),
                ("LAUNCH_CHAIN_UNAPPROVED", lambda d: d.update(
                    launch_chain_sha256="4" * 64), "launch chain"),
                ("EVIDENCE_MANIFEST_NOT_APPROVED", lambda d: d.update(
                    analysis_chain_sha256="5" * 64), "analysis chain"),
                ("HOLDOUT_MASK_MISMATCH", lambda d: d.update(
                    mask_seeds_sha256="6" * 64), "mask-seed identity"),
                ("V02_SOURCE_HASH_MISMATCH", lambda d: d["holdout_manifest"]
                 .update(file_sha256="7" * 64), "manifest file"),
                ("HOLDOUT_APPROVAL_MODE", lambda d: d.update(fixture=False),
                 "production approval in a fixture run"),
                ("HOLDOUT_APPROVAL_INVALID", lambda d: d.update(extra=1),
                 "extra key"),
                ("HOLDOUT_APPROVAL_INVALID", lambda d: d.update(
                    statement="ok"), "statement too short")):
            cases.append((code, _hrn_forged(H, lab.replace(" ", "_"), edit),
                          f"approval bound to another {lab}" if code ==
                          "HOLDOUT_APPROVAL_BINDING" else f"approval: {lab}"))
        for code, apx, lab in cases:
            cx.raises(code, lambda a=apx: _hrn_run(H, o, approval=a), lab)
        cx.ok(not o.exists(), "every refused gate: nothing created (output "
              "directory absent)")
        # chain without holdout_config role
        from seqref_mri.scripts import v05lt_evidence_check as EVC
        r2 = {k: v for k, v in H["roles"].items() if k != "holdout_config"}
        chm2 = td / "chain_nohc.json"
        csha2 = EVC.make_chain_manifest(chm2, B["root"], r2)
        cx.raises("HOLDOUT_CHAIN_NO_HOLDOUT_CONFIG", lambda: HO.make_approval(
            td / "ap_nohc.json", root=B["root"], launch_chain=H["P"]["chain"][:2],
            analysis_chain=(chm2, csha2), manifest_path=B["man"]["manifest"],
            statement="fixture approval without frozen configuration",
            fixture=True, launch_root=H["P"]["tmp"]),
            "approval refused: analysis chain without a frozen holdout "
            "configuration")
        fx = _hrn_forged(H, "nohc", lambda d: d.update(
            analysis_chain_sha256=csha2))
        cx.raises("HOLDOUT_CHAIN_NO_HOLDOUT_CONFIG", lambda: _hrn_run(
            H, o, approval=fx, analysis_chain=chm2),
            "run refused: approved analysis chain without holdout_config")
        # another self-consistent holdout manifest named by the approval
        man5 = _b3_hold_manifest(td / "man5", n=5)
        rows5 = RS.load_holdout_manifest(
            man5["manifest"], RS.Identities(),
            file_sha256=man5["manifest_file_sha256"],
            content_sha256=man5["manifest_content_sha256"],
            fixture=True)["rows"]
        f5 = _hrn_forged(H, "man5", lambda d: d.update(
            holdout_manifest={"file_sha256": man5["manifest_file_sha256"],
                              "content_sha256":
                                  man5["manifest_content_sha256"]},
            mask_seeds_sha256=RS.holdout_mask_seeds_sha256(rows5)))
        cx.raises("HOLDOUT_MANIFEST_MISMATCH", lambda: _hrn_run(
            H, o, approval=f5, manifest_path=man5["manifest"]),
            "another (self-consistent) holdout manifest != frozen "
            "configuration")
        cx.raises("HOLDOUT_STAGES_INVALID", lambda: _hrn_run(
            H, o, stages=object()), "stage object not ProductionStages")
        prod = PS.ProductionStages(C.Paths(), H["hdata"])
        cx.raises("HOLDOUT_FIXTURE_IN_PRODUCTION", lambda: _hrn_run(
            H, o, stages=prod), "production stages with a fixture launch "
            "root refused")
        cx.raises("HOLDOUT_MANIFEST_PATH", lambda: _hrn_run(
            H, o, stages=prod, launch_root=None),
            "production: only the registered holdout manifest path")
        cx.raises("FIXTURE_TARGETS_PRODUCTION", lambda: _hrn_run(
            H, C.Paths().v05lt / "holdout_fixture"),
            "fixture run writing under results/v05lt refused")
        cx.ok(not o.exists() and not (C.Paths().v05lt /
                                      "holdout_fixture").exists(),
              "refusals created nothing")
        reg = HO.registered_out(C.Paths())
        pre = os.path.lexists(reg)
        rc = HO.main(["run", "--approval", f"{H['approval'][0]}:"
                      f"{H['approval'][1]}", "--launch-chain",
                      str(H["P"]["chain"][0]), "--analysis-chain",
                      str(H["chain"][0])])
        cx.ok(rc == 2 and os.path.lexists(reg) == pre,
              "gated CLI in production refuses a fixture approval before any "
              "output or holdout contact", detail=f"rc {rc}")
        cx.ok(HO.main(["run", "--approval", "x:bad", "--launch-chain", "a",
                       "--analysis-chain", "b"]) == 2,
              "CLI: malformed PATH:SHA256 refused")
        # ---- coincident selected / epoch 3 end to end (refinement 3) ----
        H2 = _hrn_setup(td / "co", max_epochs=3, factory=_stub_w1000)
        t2 = HO.plan_tasks(H2["B"]["sel"])
        co = [t for t in t2 if t["roles"] == ["selected", "epoch3"]]
        out2 = td / "out_co"
        r2 = _hrn_run(H2, out2)
        led = C.Ledger(out2 / HO.LEDGER_NAME).records()
        done = {r["body"]["task"]: r["body"] for r in led
                if r["kind"] == "task_done"}
        d2 = {k: json.loads(Path(v["path"]).read_bytes())
              for k, v in r2["records"].items()}
        same = all(d2[f"{t['arm']}|{t['seed']}|selected"]["rows"] ==
                   d2[f"{t['arm']}|{t['seed']}|epoch3"]["rows"] and
                   d2[f"{t['arm']}|{t['seed']}|selected"]["evaluation"] ==
                   d2[f"{t['arm']}|{t['seed']}|epoch3"]["evaluation"] and
                   done[t["name"]]["roles"] == ["selected", "epoch3"] and
                   done[t["name"]]["decodes"] == EV.BANK_ROWS * n
                   for t in co)
        cx.ok(bool(co) and len(done) == len(t2) == 12 - len(co) and
              len(r2["records"]) == 12 and same,
              "coincident checkpoints end to end: evaluated ONCE (one task, "
              "128 decodes per slice once), one record per role with the "
              "same rows and evaluation identity", detail=f"{len(co)} "
              f"coincident of {len(t2)} tasks")
        lr2 = RS.load_holdout_records(
            _hrn_holdout(H2, r2), H2["B"]["camp"], H2["B"]["sel"],
            RS.Identities(), fixture=True, mask_seeds_sha256=H2["B"]["msha"])
        cx.ok(len(lr2["records"]) == 12, "coincident case: the b3 loader "
              "accepts both role records of each shared evaluation")
        del torch


def hrf_group(cx: Ctx) -> None:
    """HRF (revision-1 review): (1) post-gate failures -- typed or
    unexpected, in data loading / width checks, parent and state
    preparation, existing-output verification and the index -- are
    recorded halts; clearing the fault alone cannot resume; (2) output-root
    guard on the resolved destination: symlinked roots, aliases into the
    registered results/v05lt and redirected production paths refused before
    anything is created (fixture patches declared)."""
    from seqref_mri.scripts import v05lt_holdout as HO
    from seqref_mri.scripts import v05lt_prodstages as PS
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        H = _hrn_setup(td)

        def halts(o):
            return [r["body"] for r in C.Ledger(o / HO.LEDGER_NAME).records()
                    if r["kind"] == "halt"]

        def boom_rt(*a):
            raise RuntimeError("fixture boom (unexpected)")

        def boom_typed(*a):
            raise C.V05LTError("FIXTURE_BOOM", "fixture boom (typed)")
        cases = [
            ("data", "holdout:data", boom_rt, "HOLDOUT_STAGE_ERROR"),
            ("data", "holdout:data", boom_typed, "HOLDOUT_DATA_FAILED"),
            ("states", "holdout:states", boom_rt, "HOLDOUT_STAGE_ERROR"),
            ("index", "holdout:index", boom_rt, "HOLDOUT_STAGE_ERROR")]
        for k, (stage, hook, fn, code) in enumerate(cases):
            o = td / f"b{k}"
            cx.raises(code, lambda o=o, h=hook, f=fn: _hrn_run(
                H, o, hooks={h: f}), f"{stage}: {fn.__name__} -> recorded "
                f"halt {code}")
            hb = halts(o)
            cx.ok(len(hb) == 1 and hb[0]["stage"] == stage and
                  (hb[0].get("cause") == "FIXTURE_BOOM" if
                   fn is boom_typed else "fixture boom" in
                   hb[0].get("traceback", "")),
                  f"{stage}: halt names the stage and the cause / traceback")
            n0 = len(C.Ledger(o / HO.LEDGER_NAME).records())
            cx.raises("HOLDOUT_HALT_UNRESOLVED", lambda o=o: _hrn_run(H, o),
                      f"{stage}: fault cleared, rerun without a decision "
                      "refused")
            cx.ok(len(C.Ledger(o / HO.LEDGER_NAME).records()) == n0,
                  f"{stage}: the refused rerun recorded no new gate and "
                  "reached no holdout data")
        # real (unhooked) faults through fixture patches of module functions
        for name, attr, code in (
                ("width check", "check_widths", "HOLDOUT_STAGE_ERROR"),
                ("holdout view", "holdout_view", "HOLDOUT_STAGE_ERROR"),
                ("parent preparation", "_p4_sref", "HOLDOUT_STAGE_ERROR")):
            o = td / f"p_{attr}"
            orig = getattr(HO, attr)

            def bad(*a, **k):
                raise ValueError(f"fixture fault in {attr}")
            setattr(HO, attr, bad)
            try:
                cx.raises(code, lambda o=o: _hrn_run(H, o),
                          f"{name}: unexpected ValueError -> recorded halt "
                          "(revision-1 reproduction)")
            finally:
                setattr(HO, attr, orig)
            cx.raises("HOLDOUT_HALT_UNRESOLVED", lambda o=o: _hrn_run(H, o),
                      f"{name}: function restored, rerun without a decision "
                      "refused")
        o = td / "p_check_widths"
        dp = td / "dec_w.json"
        dsha = HO.make_decision(dp, o, "retry", "fixture: width fault fixed")
        cx.ok(_hrn_run(H, o, decision=(dp, dsha))["status"] == "complete",
              "bound retry after a recorded data-stage halt completes")
        # existing-output verification fault on a complete output
        cx.raises("HOLDOUT_STAGE_ERROR", lambda: _hrn_run(
            H, o, hooks={"holdout:verify": boom_rt}),
            "existing-output verification: unexpected error -> recorded halt")
        cx.ok(halts(o)[-1]["stage"] == "verify", "halt stage = verify")
        cx.raises("HOLDOUT_HALT_UNRESOLVED", lambda: _hrn_run(H, o),
                  "verification fault cleared, rerun without a decision "
                  "refused")
        # ---- (2) output-root guard -------------------------------------
        fake = td / "fakerepo"
        fpaths = C.Paths(repo=fake)
        freg = fpaths.v05lt
        freg.mkdir(parents=True)
        st2 = PS.ProductionStages(fpaths, H["hdata"],
                                  fixture=dict(H["P"]["fx"]))
        alias = td / "alias"
        alias.symlink_to(freg / "hx")
        cx.raises("HOLDOUT_OUT_PATH", lambda: _hrn_run(
            H, alias, stages=st2), "fixture output root that is a symlink "
            "(into the registered results/v05lt) refused")
        aliasdir = td / "aliasdir"
        aliasdir.symlink_to(freg)
        cx.raises("FIXTURE_TARGETS_PRODUCTION", lambda: _hrn_run(
            H, aliasdir / "hx", stages=st2), "fixture output whose parent is "
            "a symlink into the registered results/v05lt refused (resolved "
            "destination)")
        cx.ok(sorted(p.name for p in freg.iterdir()) == [],
              "nothing created in the registered results/v05lt (no "
              "directory, lock or ledger)")
        oo = td / "o_led"
        oo.mkdir()
        (oo / HO.LEDGER_NAME).symlink_to(td / "elsewhere.jsonl")
        cx.raises("HOLDOUT_OUT_PATH", lambda: _hrn_run(H, oo),
                  "existing ledger entry that is a symlink refused")
        cx.ok(not (td / "elsewhere.jsonl").exists(),
              "nothing written through the ledger symlink")
        reg = HO.registered_out(fpaths)
        reg.mkdir()
        cx.ok(HO.check_out_dir(reg, fpaths.data_root, fpaths, True) == reg,
              "production: the registered output root (real directories) "
              "is accepted by the guard")
        reg.rmdir()
        (td / "redir").mkdir()
        reg.symlink_to(td / "redir")
        cx.raises("HOLDOUT_OUT_PATH", lambda: HO.check_out_dir(
            reg, fpaths.data_root, fpaths, True), "production output root "
            "redirected elsewhere (symlink) refused")
        reg.unlink()
        moved = td / "moved_v05lt"
        freg.rename(moved)
        freg.symlink_to(moved)
        (moved / "holdout").mkdir()
        cx.raises("HOLDOUT_OUT_PATH", lambda: HO.check_out_dir(
            reg, fpaths.data_root, fpaths, True), "production output "
            "redirected through a symlinked parent (results/v05lt) refused")
        cx.raises("HOLDOUT_OUT_PATH", lambda: HO.check_out_dir(
            td / "other", fpaths.data_root, fpaths, True),
            "production: any other output root refused")
        cx.raises("HOLDOUT_OUT_PATH", lambda: HO.check_out_dir(
            reg, td, fpaths, True), "production: unregistered data root "
            "refused")


# ---------------------------------------------------------------------------
# I1.9 c4 (SEQREF-MRI-V05LTI19C4 v0.1): HLW (synthetic by design), HLB
# (addition B). ONE fixture campaign drives both the launcher and the
# analysis chain / selection, so the selection belongs to the verified
# campaign. Synthetic holdout .h5 files only; no real holdout data.
# ---------------------------------------------------------------------------
def _hlw_setup(td: Path, factory=None, max_epochs: int = 4) -> dict:
    """Fixture campaign through the launcher (ProductionStages, stub model
    unless `factory` says otherwise) -> difficult cases, configuration,
    analysis, report, selection manifest and frozen holdout configuration
    over THAT campaign -> analysis-chain manifest -> synthetic holdout
    volumes -> holdout approval (fixture). Fixture only."""
    from seqref_mri.scripts import v05lt_analysis as A
    from seqref_mri.scripts import v05lt_evidence_check as EVC
    from seqref_mri.scripts import v05lt_holdout as HO
    from seqref_mri.scripts import v05lt_launch as L
    from seqref_mri.scripts import v05lt_report as R
    from seqref_mri.scripts import v05lt_results as RS
    from seqref_mri.scripts import v05lt_stats as S
    P = _prd_layout(td / "P", max_epochs=max_epochs, factory=factory)
    root = P["tmp"]
    camp_dir = root / "camp"
    camp = L.Campaign(camp_dir, P["st"], chain=P["chain"], fixture=True)
    camp.run("new")
    lay = P["layout"]
    an = camp_dir / "analysis"
    an.mkdir()
    lay["difficult_sha256"] = A.select_difficult(lay, lay["difficult"], n=1)
    cp = an / "config.json"
    csha = A.write_config(lay, cp, B=200, fixture_override=True)
    apth = an / "analysis.json"
    asha = A.run_analysis(lay, cp, csha, apth)
    rep = an / "report"
    R.build_report(apth, asha, rep)
    lay["epoch0_store"] = {s_: camp_dir / "epoch0" / f"seed_{s_}"
                           for s_ in lay["seeds"]}
    selp = an / "selection.json"
    selsha = A.write_selection_manifest(lay, root, selp, config=(cp, csha),
                                        analysis=(apth, asha), report_dir=rep)
    sel = A.load_selection(selp, selsha)
    man = _b3_hold_manifest(camp_dir / "holdout_manifest")
    cl = RS.load_campaign(lay)
    recdir = camp_dir / "hold_cfg_records"
    recdir.mkdir()
    recs = {}
    for run in cl["runs"].values():
        for role in RS.HOLDOUT_ROLES:
            k = f"{run['arm']}|{run['seed']}|{role}"
            pth = recdir / f"{run['arm']}_{run['seed']}_{role}.json"
            recs[k] = {"path": pth, "sha256": S.write_json_once(
                pth, _b3_hold_record(run, role, sel, man))}
    hold = {k: man[k] for k in ("manifest", "manifest_file_sha256",
                                "manifest_content_sha256")}
    hold["records"] = recs
    hcp = an / "holdout_config.json"
    hsha = A.write_holdout_config(cp, csha, hold, hcp, fixture=True)

    def rel(x):
        return str(Path(x).relative_to(root))
    roles = {"config": rel(cp), "difficult": rel(lay["difficult"]),
             "analysis": rel(apth), "report_dir": rel(rep),
             "selection": rel(selp), "holdout_config": rel(hcp)}
    chm = td / "achain.json"
    chsha = EVC.make_chain_manifest(chm, root, roles)
    hdata = td / "hdata"
    for k, e in enumerate(man["doc"]["entries"]):
        write_volume(hdata / e["file"], HRN_SLICES, 7000 + k)
    apf = td / "happroval.json"
    apsha = HO.make_approval(
        apf, root=root, launch_chain=P["chain"][:2],
        analysis_chain=(chm, chsha), manifest_path=man["manifest"],
        statement="fixture holdout approval: synthetic holdout files (c4)",
        fixture=True)
    kw = dict(analysis_chain=chm, manifest_path=man["manifest"],
              data_root=hdata, root=root)
    return {"P": P, "camp": camp, "root": root, "camp_dir": camp_dir,
            "sel": sel, "selection": (selp, selsha), "man": man,
            "achain": (chm, chsha), "hdata": hdata, "approval": (apf, apsha),
            "kw": kw, "td": td}


def _hlw_campaign(W: dict, stages=None, chain=None, hook=None, root=None):
    """A launcher Campaign over the fixture campaign (fresh stage object,
    bound by the launcher)."""
    from seqref_mri.scripts import v05lt_launch as L
    root = root or W["camp_dir"]
    st = stages or W["P"]["stages"](root=root)
    return L.Campaign(root, st, chain=chain or W["P"]["chain"],
                      fixture=True, hook=hook)


def _hlw_holdout(W: dict, out: Path, approval=None, camp=None, **over):
    kw = dict(W["kw"], **over)
    c = camp or _hlw_campaign(W)
    return c.holdout(approval or W["approval"], out_dir=out, **kw)


def _hlw_ledger(src: Path, dst: Path, edit) -> None:
    """Re-chained copy of a campaign ledger: edit([(kind, body), ...])
    returns the records to write (fixture only)."""
    recs = edit([(r["kind"], r["body"]) for r in C.Ledger(src).records()])
    if dst.exists():
        dst.unlink()
    led = C.Ledger(dst)
    for kind, body in recs:
        led.append(kind, body)


def _hlw_tree(d: Path) -> list:
    """(relative path, kind, size, mtime_ns) of every entry under d."""
    out = []
    if not os.path.lexists(d):
        return out
    for dp, dns, fns in os.walk(d):
        for n in sorted(dns + fns):
            q = Path(dp) / n
            st_ = q.lstat()
            out.append((str(q.relative_to(d)), q.is_dir(), st_.st_size,
                        st_.st_mtime_ns))
    return sorted(out)


class _HlwOpens:
    """FIXTURE-ONLY: tracks h5py opens under the synthetic holdout data root
    together with the number of `gate_passed` records at that moment."""

    def __init__(self, hdata: Path, ledgers):
        self.root = str(Path(hdata).resolve())
        self.ledgers = ledgers
        self.opens = []

    def _gates(self) -> int:
        from seqref_mri.scripts import v05lt_holdout as HO
        n = 0
        for o in self.ledgers():
            p = Path(o) / HO.LEDGER_NAME
            if p.exists():
                n += sum(r["kind"] == "gate_passed"
                         for r in C.Ledger(p).records())
        return n

    def __enter__(self):
        import h5py
        self._orig = h5py.File
        trk = self

        class Tracked(self._orig):
            def __init__(self, name, *a, **k):
                q = str(Path(name).resolve())
                if q.startswith(trk.root):
                    trk.opens.append((q, trk._gates()))
                super().__init__(name, *a, **k)
        h5py.File = Tracked
        return self

    def __exit__(self, *exc):
        import h5py
        h5py.File = self._orig
        return False


def hlw_group(cx: Ctx) -> None:
    """HLW (I1.9 c4, synthetic by design): the launcher's separate holdout
    command on ONE fixture campaign (ProductionStages, stub model) whose
    analysis chain and selection were built from that campaign; synthetic
    holdout .h5 files. Verified completion, single c3 gate, selection
    binding, locks and concurrency, mirror records, recovery and re-entry,
    read-only status."""
    import shutil
    from seqref_mri.scripts import v05lt_holdout as HO
    from seqref_mri.scripts import v05lt_launch as L
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        W = _hlw_setup(td)
        camp, cdir = W["camp"], W["camp_dir"]
        outs = []
        trk_ledgers = lambda: list(outs)          # noqa: E731
        # -- new / resume never reach the holdout ----------------------------
        calls = {"gate": 0, "runner": 0}
        saved = (HO.verify_gate, HO.Runner.run)

        def cg(*a, **k):
            calls["gate"] += 1
            return saved[0](*a, **k)

        def cr(self, *a, **k):
            calls["runner"] += 1
            return saved[1](self, *a, **k)
        HO.verify_gate, HO.Runner.run = cg, cr
        try:
            r_ = _hlw_campaign(W).run("resume")
        finally:
            HO.verify_gate, HO.Runner.run = saved
        kinds = [r["kind"] for r in camp.ledger.records()]
        cx.ok(r_["status"] == "complete" and calls == {"gate": 0, "runner": 0}
              and not any(k in L.HOLDOUT_MIRROR_KINDS for k in kinds),
              "`new` and `resume` complete the campaign without calling the "
              "holdout gate or runner and write no holdout record "
              "(finishing training never triggers the holdout)")
        # -- read-only status before any holdout ------------------------------
        o1 = td / "h1"
        before = _hlw_tree(cdir)
        st0 = L.holdout_status(cdir, o1)
        cx.ok(st0 == {"mirror": {}, "holdout": None} and
              _hlw_tree(cdir) == before and not os.path.lexists(o1),
              "status before the holdout: empty, read-only (nothing "
              "created, campaign tree unchanged)")
        # -- refusals before the gate: nothing created, nothing opened --------
        apf, apsha = W["approval"]
        lchain = W["P"]["chain"]
        alt = _prd_chain_with(W["P"], "alt", lambda d: None)
        empty = td / "empty_camp"
        bad = [
            ("LAUNCH_NO_CAMPAIGN", lambda: _hlw_holdout(
                W, o1, camp=_hlw_campaign(W, root=td / "nocamp")),
             "no campaign directory"),
            ("LAUNCH_HOLDOUT_INCOMPLETE", lambda: (empty.mkdir(), _hlw_holdout(
                W, o1, camp=_hlw_campaign(W, root=empty)))[1],
             "campaign without a ledger"),
            ("HOLDOUT_APPROVAL_UNAPPROVED", lambda: _hlw_holdout(
                W, o1, approval=(apf, "0" * 64)),
             "holdout approval at an unapproved sha256"),
            ("HOLDOUT_APPROVAL_INVALID", lambda: _hlw_holdout(
                W, o1, approval=lchain[:2]), "the launch approval (launch "
             "chain) given as the holdout approval -- distinct inputs"),
            ("LAUNCH_HOLDOUT_CHAIN_MISMATCH", lambda: _hlw_holdout(
                W, o1, camp=_hlw_campaign(W, chain=alt)),
             "launcher approved another (valid) launch chain than the one "
             "the holdout approval names"),
            ("LAUNCH_CHAIN_UNAPPROVED", lambda: _hlw_holdout(
                W, o1, camp=_hlw_campaign(W, chain=(lchain[0], "0" * 64,
                                                    lchain[2]))),
             "launch chain at an unapproved sha256")]
        with _HlwOpens(W["hdata"], trk_ledgers) as trk:
            for code, fn, lab in bad:
                cx.raises(code, fn, f"refused before the gate: {lab}")
                cx.ok(not trk.opens and not os.path.lexists(o1),
                      f"nothing opened or created: {lab}")
        cx.ok(not any(k in L.HOLDOUT_MIRROR_KINDS for k in
                      [r["kind"] for r in camp.ledger.records()]),
            "refusals before the gate write no campaign mirror record")
        # -- verified completion on a copy of the campaign --------------------
        cp_ = td / "camp_copy"
        shutil.copytree(cdir, cp_, symlinks=True)
        led_c = cp_ / L.LEDGER_NAME
        orig_led = led_c.read_bytes()

        def copy_camp():
            c_ = _hlw_campaign(W, root=cp_)
            c_._verify_chain()           # bound stages: evidence is real
            return c_
        oc = td / "hc"
        with _HlwOpens(W["hdata"], trk_ledgers) as trk:
            cx.raises("LAUNCH_HOLDOUT_SELECTION_UNBOUND", lambda: _hlw_holdout(
                W, oc, camp=copy_camp()), "complete, verified campaign that "
                "is NOT the one the approved selection names (a copy) "
                "refused: selection bound to the verified campaign")
            c_ok = copy_camp()
            c_ok._verify_chain()
            ev_c, ev_sha = c_ok.verify_completion()
            cx.ok(sorted(ev_c["runs"]) == sorted(
                L.run_id(s_, a_) for k_, s_, a_ in L.stage_order()
                if k_ == "run") and len(ev_c["epoch0"]) == 3,
                "verified completion: six expected run identities and three "
                "epoch-0 stages, evidence re-verified (read-only)")

            def led_variant(lab, edit, code="LAUNCH_HOLDOUT_INCOMPLETE"):
                tmpl = td / "led_tmp.jsonl"
                led_c.write_bytes(orig_led)
                _hlw_ledger(led_c, tmpl, edit)
                led_c.write_bytes(tmpl.read_bytes())
                tmpl.unlink()
                cx.raises(code, lambda: copy_camp().verify_completion(), lab)
                led_c.write_bytes(orig_led)

            def drop_adv(rs):
                adv = [i for i, (k, b) in enumerate(rs) if k == "advance"]
                return [x for i, x in enumerate(rs) if i != adv[-1]]

            def extra_adv(rs):
                i = max(j for j, (k, _) in enumerate(rs) if k == "advance")
                b = dict(rs[i][1], run="2029_lam0")
                return rs[:i + 1] + [("advance", b)] + rs[i + 1:]

            def dup_seed(rs):
                i = max(j for j, (k, _) in enumerate(rs) if k == "advance")
                return rs[:i] + [("advance", dict(rs[i][1], run="2026_lam0"))
                                 ] + rs[i + 1:]

            def alt_sha(rs):
                i = max(j for j, (k, _) in enumerate(rs) if k == "advance")
                arts = dict(rs[i][1]["artefacts"], **{
                    "weights_final.pt": "a" * 64})
                return rs[:i] + [("advance", dict(rs[i][1], artefacts=arts))
                                 ] + rs[i + 1:]
            led_variant("one expected run without its advance record refused "
                        "(six identities, not a count)", drop_adv)
            led_variant("an extra, unexpected run identity refused",
                        extra_adv)
            led_variant("six advance records but one identity repeated and "
                        "one missing refused", dup_seed)

            def swap_id(rs):
                i = max(j for j, (k, _) in enumerate(rs) if k == "advance")
                return rs[:i] + [("advance", dict(rs[i][1], run="2029_lam2"))
                                 ] + rs[i + 1:]
            led_variant("six distinct advance identities but one unexpected "
                        "(and one expected missing) refused", swap_id)
            led_variant("latest launch not completed (launch_start after "
                        "campaign_complete) refused",
                        lambda rs: rs + [("launch_start", {"mode": "resume"})])
            led_variant("unresolved campaign halt refused", lambda rs: rs + [
                ("halt", {"code": "LAUNCH_POSTRUN_FAILED",
                          "message": "fixture", "run": "2026_lam0"})])
            led_variant("advance record whose final-weights sha256 differs "
                        "from the current file refused", alt_sha,
                        "LAUNCH_HOLDOUT_EVIDENCE")

            def alt_pe(rs):
                i = max(j for j, (k, _) in enumerate(rs) if k == "advance")
                arts = dict(rs[i][1]["artefacts"])
                k0 = sorted(k for k in arts if k.startswith("posteval/"))[0]
                arts[k0] = "f" * 64
                return rs[:i] + [("advance", dict(rs[i][1], artefacts=arts))
                                 ] + rs[i + 1:]
            led_variant("advance record whose post-run record sha256 differs "
                        "from the current file refused", alt_pe,
                        "LAUNCH_HOLDOUT_EVIDENCE")
            # current evidence altered (files)
            rid = "2027_lam2"
            fwp = cp_ / rid / "final" / "weights_final.pt"
            os.chmod(fwp, 0o644)
            raw = fwp.read_bytes()
            fwp.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            cx.raises("LAUNCH_HOLDOUT_EVIDENCE", lambda: copy_camp(
            ).verify_completion(), "final weights altered after advance "
                "refused (current evidence re-verified)")
            fwp.write_bytes(raw)
            pe = sorted((cp_ / "posteval").iterdir())[0]
            pe.rename(pe.with_name("moved_" + pe.name))
            cx.raises("LAUNCH_HOLDOUT_EVIDENCE", lambda: copy_camp(
            ).verify_completion(), "a listed post-run record missing refused")
            pe.with_name("moved_" + pe.name).rename(pe)
            e0 = cp_ / "epoch0" / "seed_2028" / "eval"
            e0.rename(e0.with_name("eval_moved"))
            cx.raises("LAUNCH_HOLDOUT_EVIDENCE", lambda: copy_camp(
            ).verify_completion(), "an epoch-0 stage no longer complete "
                "refused")
            e0.with_name("eval_moved").rename(e0)
            copy_camp().verify_completion()
            cx.ok(not trk.opens and not os.path.lexists(oc),
                  "completion / selection refusals open no holdout file and "
                  "create no output")
        # -- selection binding (unit) -----------------------------------------
        import copy as _copy
        gsel = W["sel"]
        c_o = _hlw_campaign(W)
        c_o._verify_chain()
        evo = c_o.verify_completion()[0]
        try:
            L.bind_selection(gsel, evo, W["root"])
            bound_ok = ""
        except C.V05LTError as exc:
            bound_ok = exc.code
        cx.ok(bound_ok == "", "the approved selection binds to the verified "
              "campaign it was built from", detail=bound_ok)
        other = next(iter(sorted(gsel["runs"])))
        o_arm, o_seed = other.split("|")
        foreign = [k for k in sorted(gsel["runs"]) if k != other][0]
        edits = [
            ("final sha256", lambda d: d["runs"][other]["final"].update(
                sha256="b" * 64)),
            ("final file of another run", lambda d: d["runs"][other][
                "final"].update(file=d["runs"][foreign]["final"]["file"])),
            ("final epoch", lambda d: d["runs"][other]["final"].update(
                epoch=d["runs"][other]["final"]["epoch"] + 1)),
            ("selected model state not post-evaluated",
             lambda d: d["runs"][other]["selected"].update(
                 model_state_sha256="c" * 64)),
            ("epoch-3 file inside another run", lambda d: d["runs"][other][
                "epoch3"].update(file=d["runs"][foreign]["epoch3"]["file"])),
            ("epoch-0 checkpoint sha256", lambda d: d["epoch0"][o_seed]
             .update(sha256="d" * 64)),
            ("epoch-0 model state", lambda d: d["epoch0"][o_seed].update(
                model_state_sha256="e" * 64))]
        for lab, ed in edits:
            d = _copy.deepcopy(gsel)
            ed(d)
            cx.raises("LAUNCH_HOLDOUT_SELECTION_UNBOUND", lambda d=d:
                      L.bind_selection(d, evo, W["root"]),
                      f"selection not of the verified campaign: {lab}")
        # -- core run: order, mirror records ----------------------------------
        outs.append(o1)
        with _HlwOpens(W["hdata"], trk_ledgers) as trk:
            res = _hlw_holdout(W, o1)
        recs1 = C.Ledger(o1 / HO.LEDGER_NAME).records()
        vols = {e["file"] for e in W["man"]["doc"]["entries"]}
        opened = {str(Path(q).relative_to(Path(W["hdata"]).resolve()))
                  for q, _ in trk.opens}
        cx.ok(res["status"] == "complete" and len(res["records"]) == 12 and
              bool(trk.opens) and all(n >= 1 for _, n in trk.opens) and
              opened == vols, "launcher holdout command: 12 records; every "
              "holdout file open after the recorded c3 gate; only manifest "
              "volumes opened", detail=f"{len(trk.opens)} opens")
        crec = camp.ledger.records()
        mk = [r for r in crec if r["kind"] in L.HOLDOUT_MIRROR_KINDS]
        comp = [r for r in recs1 if r["kind"] == "complete"][-1]
        cx.ok([r["kind"] for r in mk] == ["holdout_start", "holdout_done"]
              and mk[1]["body"]["index_sha256"] == comp["body"][
                  "index_sha256"] == res["index_sha256"] and
              mk[1]["body"]["holdout_complete_seq"] == comp["seq"] and
              mk[0]["body"]["approval_sha256"] == apsha and
              mk[0]["body"]["launch_chain_sha256"] == lchain[1] and
              mk[1]["body"]["recovered"] is False,
              "campaign ledger mirrors holdout_start and holdout_done "
              "(index sha256 = holdout ledger completion; distinct launch "
              "and holdout approvals recorded)")
        cx.ok(L.campaign_halt_state(camp.ledger)["unresolved"] is None,
              "mirror records are not campaign halts")
        # -- re-entry after completion ----------------------------------------
        nrec = len(recs1)
        n_ev = {"k": 0}
        sev = HO.evaluate_task

        def cnt(*a, **k):
            n_ev["k"] += 1
            return sev(*a, **k)
        HO.evaluate_task = cnt
        try:
            with _HlwOpens(W["hdata"], trk_ledgers) as trk:
                res_b = _hlw_holdout(W, o1)
        finally:
            HO.evaluate_task = sev
        cx.ok(res_b.get("reentry") is True and res_b["index_sha256"] ==
              res["index_sha256"] and not trk.opens and n_ev["k"] == 0 and
              len(C.Ledger(o1 / HO.LEDGER_NAME).records()) == nrec and
              len([r for r in camp.ledger.records()
                   if r["kind"] == "holdout_done"]) == 1,
              "re-entry after completion: verified by hash, no re-evaluation, "
              "no holdout file opened, no new holdout-ledger record, no "
              "duplicate mirror")
        # -- re-entry verification failures are c3 halts (revision 2) -------
        def halts_mirrored(out_):
            hh_ = HO._halt_state(C.Ledger(out_ / HO.LEDGER_NAME))["unresolved"]
            b_ = HO._halt_binding(hh_)
            return hh_, [r for r in camp.ledger.records()
                         if r["kind"] == "holdout_halt" and r["body"][
                             "halt_line_sha256"] == b_["halt_line_sha256"]]
        rp = sorted(p_ for p_ in o1.iterdir() if p_.is_dir() and
                    not p_.name.startswith(("_", ".")))[0]
        rf = sorted(rp.glob("*.json"))[0]
        os.chmod(rf, 0o644)
        rb = rf.read_bytes()
        rf.write_bytes(rb + b" ")
        with _HlwOpens(W["hdata"], trk_ledgers) as trk:
            cx.raises("HOLDOUT_OUTPUT_MISMATCH", lambda: _hlw_holdout(W, o1),
                      "re-entry: a published record altered -> recorded c3 "
                      "halt (output mismatch)")
        hh, mm = halts_mirrored(o1)
        cx.ok(not trk.opens and hh["body"].get("stage") == "reentry" and
              hh["body"].get("task") == rp.name and
              hh["body"].get("cause") == "LAUNCH_HOLDOUT_REENTRY" and
              hh["body"].get("approval_sha256") == apsha and len(mm) == 1 and
              L.campaign_halt_state(camp.ledger)["unresolved"] is None,
              "halt in the holdout ledger names stage, task, cause and "
              "approval; reference mirrored in the campaign ledger; no "
              "holdout file opened; not a campaign halt")
        rf.write_bytes(rb)
        cx.raises("HOLDOUT_HALT_UNRESOLVED", lambda: _hlw_holdout(W, o1),
                  "restoring the record alone leaves continuation blocked")
        # unrelated decision: self-consistent record bound to another halt
        dpa = td / "dec_unrel.json"
        dsa = HO.make_decision(dpa, o1, "retry", "fixture: genuine decision")
        dj = json.loads(dpa.read_bytes())
        dj["binding"] = dict(dj["binding"], halt_seq=dj["binding"][
            "halt_seq"] - 1, halt_line_sha256="0" * 64)
        dpu = td / "dec_unrel2.json"
        dsu = C.write_durable_new(dpu, C.canonical_json(dj))
        cx.raises("HOLDOUT_DECISION_STALE", lambda: _hlw_holdout(
            W, o1, decision=(dpu, dsu)), "a decision bound to another halt "
            "(unrelated) refused")
        cx.raises("HOLDOUT_DECISION_UNAPPROVED", lambda: _hlw_holdout(
            W, o1, decision=(dpa, "1" * 64)), "a decision at an unapproved "
            "sha256 refused")
        res_r = _hlw_holdout(W, o1, decision=(dpa, dsa))
        failed = sorted(q.name for q in (o1 / HO.FAILED_DIR).iterdir())
        cx.ok(res_r["status"] == "complete" and
              any(f.endswith("_" + rp.name) for f in failed) and
              any(f.endswith(HO.INDEX_NAME) for f in failed) and any(
                  r["kind"] == "holdout_done" and r["body"]["index_sha256"] ==
                  res_r["index_sha256"] for r in camp.ledger.records()),
              "bound retry: altered task directory and the index listing it "
              "retained under _failed/; task re-evaluated; holdout_done "
              "mirrored", detail=str(failed))
        # index altered -> halt; the earlier (consumed) decision is stale
        ix = o1 / HO.INDEX_NAME
        os.chmod(ix, 0o644)
        ib = ix.read_bytes()
        ix.write_bytes(ib + b" ")
        cx.raises("HOLDOUT_OUTPUT_MISMATCH", lambda: _hlw_holdout(W, o1),
                  "re-entry: index altered -> recorded c3 halt")
        hi, mi = halts_mirrored(o1)
        cx.ok(hi["body"].get("stage") == "reentry" and
              hi["body"].get("task") is None and len(mi) == 1,
              "index halt recorded (no task) and mirrored")
        ix.write_bytes(ib)
        cx.raises("HOLDOUT_HALT_UNRESOLVED", lambda: _hlw_holdout(W, o1),
                  "restoring the index alone leaves continuation blocked")
        cx.raises("HOLDOUT_DECISION_STALE", lambda: _hlw_holdout(
            W, o1, decision=(dpa, dsa)), "the earlier, consumed decision is "
            "stale for the new halt")
        dpb = td / "dec_idx.json"
        dsb = HO.make_decision(dpb, o1, "retry", "fixture: index restored")
        res_i = _hlw_holdout(W, o1, decision=(dpb, dsb))
        failed2 = sorted(q.name for q in (o1 / HO.FAILED_DIR).iterdir())
        cx.ok(res_i["status"] == "complete" and len(failed2) == len(failed) + 2
              and sum(f.endswith(HO.INDEX_NAME) for f in failed2) == 2,
              "bound retry after the index halt completes; the altered index "
              "(and its sidecar) retained under _failed/", detail=str(failed2))
        res_c = _hlw_holdout(W, o1)
        cx.ok(res_c.get("reentry") is True, "afterwards re-entry verifies "
              "again without re-evaluation")
        # -- re-entry read errors are recorded halts (revision 3) -------------
        sha_orig = C.sha256_file

        def read_fault(target):
            def f(path, *a, **k):        # FIXTURE-ONLY patch of C.sha256_file
                if Path(path).resolve() == Path(target).resolve():
                    raise PermissionError(13, "fixture read fault", str(path))
                return sha_orig(path, *a, **k)
            return f
        rp2 = sorted(p_ for p_ in o1.iterdir() if p_.is_dir() and
                     not p_.name.startswith(("_", ".")))[0]
        rf2 = sorted(rp2.glob("*.json"))[0]
        for lab, target, task in (("record", rf2, rp2.name),
                                  ("index", o1 / HO.INDEX_NAME, None)):
            n_fail = len(list((o1 / HO.FAILED_DIR).iterdir()))
            C.sha256_file = read_fault(target)
            try:
                with _HlwOpens(W["hdata"], trk_ledgers) as trk:
                    cx.raises("HOLDOUT_STAGE_ERROR", lambda: _hlw_holdout(
                        W, o1), f"re-entry: {lab} read error "
                        "(PermissionError) -> recorded c3 halt")
            finally:
                C.sha256_file = sha_orig
            hr, mr = halts_mirrored(o1)
            b_ = hr["body"]
            cx.ok(not trk.opens and b_.get("stage") == "reentry" and
                  b_.get("task") == task and b_.get("type") ==
                  "PermissionError" and "fixture read fault" in b_.get(
                      "traceback", "") and b_.get("approval_sha256") == apsha
                  and len(mr) == 1 and
                  L.campaign_halt_state(camp.ledger)["unresolved"] is None,
                  f"{lab} read error: halt names stage, task, type and "
                  "traceback; mirrored by reference; not a campaign halt; "
                  "no holdout file opened")
            cx.raises("HOLDOUT_HALT_UNRESOLVED", lambda: _hlw_holdout(W, o1),
                      f"{lab} read fault cleared alone: continuation blocked")
            cx.raises("HOLDOUT_DECISION_STALE", lambda: _hlw_holdout(
                W, o1, decision=(dpb, dsb)), f"{lab}: an earlier, consumed "
                "decision is stale")
            dpe = td / f"dec_read_{lab}.json"
            dse = HO.make_decision(dpe, o1, "retry", f"fixture: {lab} read "
                                   "fault cleared")
            res_e = _hlw_holdout(W, o1, decision=(dpe, dse))
            moved = len(list((o1 / HO.FAILED_DIR).iterdir())) - n_fail
            cx.ok(res_e["status"] == "complete" and moved == (
                3 if task else 2) and any(
                r["kind"] == "holdout_done" and r["body"]["index_sha256"] ==
                res_e["index_sha256"] for r in camp.ledger.records()),
                f"{lab} read error: bound retry completes; failed attempt "
                "retained under _failed/; holdout_done mirrored",
                detail=f"{moved} entries moved")
        # -- mirror-write failures and safe re-entry --------------------------
        def failing(kind):
            def h(stage):
                if stage == f"holdout:mirror:{kind}":
                    raise RuntimeError(f"fixture mirror failure {kind}")
            return h
        o2 = td / "h2"
        outs.append(o2)
        cx.raises("LAUNCH_HOLDOUT_MIRROR_FAILED", lambda: _hlw_holdout(
            W, o2, camp=_hlw_campaign(W, hook=failing("holdout_done"))),
            "failure while writing the holdout_done mirror -> typed error "
            "(holdout already complete in its own ledger)")
        k2 = [r["kind"] for r in C.Ledger(o2 / HO.LEDGER_NAME).records()]
        i2 = [r for r in C.Ledger(o2 / HO.LEDGER_NAME).records()
              if r["kind"] == "complete"][-1]["body"]["index_sha256"]
        cx.ok(k2[-1] == "complete" and not any(
            r["kind"] == "holdout_done" and r["body"]["index_sha256"] == i2
            for r in camp.ledger.records()),
            "holdout ledger complete; campaign mirror missing")
        with _HlwOpens(W["hdata"], trk_ledgers) as trk:
            res2 = _hlw_holdout(W, o2)
        dn = [r for r in camp.ledger.records() if r["kind"] == "holdout_done"
              and r["body"]["index_sha256"] == i2]
        cx.ok(res2.get("reentry") is True and not trk.opens and
              len(dn) == 1 and dn[0]["body"]["recovered"] is True and
              [r["kind"] for r in C.Ledger(o2 / HO.LEDGER_NAME).records()]
              == k2, "safe re-entry writes the missing mirror (recovered) "
              "without re-evaluation or holdout contact")
        o3 = td / "h3"
        cx.raises("LAUNCH_HOLDOUT_MIRROR_FAILED", lambda: _hlw_holdout(
            W, o3, camp=_hlw_campaign(W, hook=failing("holdout_start"))),
            "failure while writing holdout_start -> typed error")
        cx.ok(not (o3 / HO.LEDGER_NAME).exists(), "no gate passed and no "
              "holdout contact after a failed holdout_start mirror")
        # -- holdout halt: mirror reference, bound retry -----------------------
        def boom():
            raise RuntimeError("fixture fault in holdout data stage (c4)")
        o4 = td / "h4"
        outs.append(o4)
        cx.raises("HOLDOUT_STAGE_ERROR", lambda: _hlw_holdout(
            W, o4, hooks={"holdout:data": boom}),
            "post-gate fault -> c3 recorded halt, propagated by the launcher")
        hh = HO._halt_state(C.Ledger(o4 / HO.LEDGER_NAME))["unresolved"]
        b4 = HO._halt_binding(hh)["halt_line_sha256"]
        mh = [r for r in camp.ledger.records() if r["kind"] == "holdout_halt"
              and r["body"]["halt_line_sha256"] == b4]
        cx.ok(len(mh) == 1 and mh[0]["body"]["halt_line_sha256"] ==
              HO._halt_binding(hh)["halt_line_sha256"] and
              mh[0]["body"]["halt_code"] == "HOLDOUT_STAGE_ERROR" and
              L.campaign_halt_state(camp.ledger)["unresolved"] is None,
              "holdout_halt mirrors the holdout-ledger halt (seq, line "
              "sha256, code); it is not a campaign halt")
        cx.raises("HOLDOUT_HALT_UNRESOLVED", lambda: _hlw_holdout(W, o4),
                  "clearing the fault alone is refused (c3 rule; no second "
                  "retry mechanism in the launcher)")
        cx.ok(len([r for r in camp.ledger.records()
                   if r["kind"] == "holdout_halt" and
                   r["body"]["halt_line_sha256"] == b4]) == 1,
              "no duplicate holdout_halt mirror")
        cx.raises("OWNER_DECISION_NOT_APPLICABLE", lambda:
                  L.make_campaign_decision(td / "camp_dec.json", cdir, "retry",
                                           "fixture: campaign decision for "
                                           "a holdout halt"),
                  "a campaign decision cannot target the holdout halt (no "
                  "second retry mechanism)")
        dp = td / "hdec.json"
        dsha = HO.make_decision(dp, o4, "retry", "fixture: fault cleared")
        res4 = _hlw_holdout(W, o4, decision=(dp, dsha))
        cx.ok(res4["status"] == "complete" and any(
            r["kind"] == "holdout_done" and r["body"]["index_sha256"] ==
            res4["index_sha256"] for r in camp.ledger.records()),
            "retry with the c3 decision bound to that halt completes; "
            "holdout_done mirrored")
        # halt mirror failure -> reconciled on the next invocation
        o5 = td / "h5"
        outs.append(o5)
        cx.raises("HOLDOUT_STAGE_ERROR", lambda: _hlw_holdout(
            W, o5, hooks={"holdout:data": boom},
            camp=_hlw_campaign(W, hook=failing("holdout_halt"))),
            "holdout halt with a failed mirror write: the original halt is "
            "raised (mirror failure logged)")
        h5 = HO._halt_state(C.Ledger(o5 / HO.LEDGER_NAME))["unresolved"]
        b5 = HO._halt_binding(h5)["halt_line_sha256"]
        cx.ok(not any(r["kind"] == "holdout_halt" and r["body"][
            "halt_line_sha256"] == b5 for r in camp.ledger.records()),
            "halt mirror missing after the failed write")
        cx.raises("HOLDOUT_HALT_UNRESOLVED", lambda: _hlw_holdout(W, o5),
                  "next invocation still needs the bound decision")
        m5 = [r for r in camp.ledger.records() if r["kind"] == "holdout_halt"
              and r["body"]["halt_line_sha256"] == b5]
        cx.ok(len(m5) == 1 and m5[0]["body"]["recovered"] is True,
              "the unmirrored halt is reconciled (recovered) on the next "
              "invocation")
        # -- locks and concurrent invocations ---------------------------------
        o6 = td / "h6"
        outs.append(o6)
        got = {}

        def second(stage):
            if stage != "holdout:bound" or got:
                return
            for lab, fn in (
                    ("holdout", lambda: _hlw_holdout(W, td / "h6b")),
                    ("campaign", lambda: _hlw_campaign(W).run("resume"))):
                try:
                    fn()
                    got[lab] = "NO_ERROR"
                except C.V05LTError as exc:
                    logger.info("HLW concurrent %s refused: %s", lab, exc.code)
                    got[lab] = exc.code

        def direct():
            if "direct" in got:
                return
            try:
                HO.run_holdout(W["approval"], root=W["root"],
                               launch_chain=lchain[0],
                               analysis_chain=W["kw"]["analysis_chain"],
                               manifest_path=W["kw"]["manifest_path"],
                               stages=W["P"]["stages"](root=cdir),
                               data_root=W["hdata"], out_dir=o6)
                got["direct"] = "NO_ERROR"
            except C.V05LTError as exc:
                logger.info("HLW direct runner refused: %s", exc.code)
                got["direct"] = exc.code
        res6 = _hlw_holdout(W, o6, camp=_hlw_campaign(W, hook=second),
                            hooks={"holdout:data": direct})
        cx.ok(res6["status"] == "complete" and got == {
            "holdout": "LAUNCH_LOCK_HELD", "campaign": "LAUNCH_LOCK_HELD",
            "direct": "LAUNCH_LOCK_HELD"} and not (td / "h6b").exists(),
            "campaign lock held throughout: a second holdout command and a "
            "campaign run refused; c3 holdout lock held: the direct runner "
            "refused", detail=str(got))
        o7 = td / "h7"
        outs.append(o7)
        got2 = {}
        n_start = len([r for r in camp.ledger.records()
                       if r["kind"] == "holdout_start"])

        def launcher_inside():
            if got2:
                return
            try:
                _hlw_holdout(W, o7)
                got2["launcher"] = "NO_ERROR"
            except C.V05LTError as exc:
                logger.info("HLW launcher inside direct runner: %s", exc.code)
                got2["launcher"] = exc.code
        HO.run_holdout(W["approval"], root=W["root"], launch_chain=lchain[0],
                       analysis_chain=W["kw"]["analysis_chain"],
                       manifest_path=W["kw"]["manifest_path"],
                       stages=W["P"]["stages"](root=cdir),
                       data_root=W["hdata"], out_dir=o7,
                       hooks={"holdout:data": launcher_inside})
        cx.ok(got2 == {"launcher": "LAUNCH_LOCK_HELD"} and len(
            [r for r in camp.ledger.records() if r["kind"] ==
             "holdout_start"]) == n_start, "direct runner holding the "
            "holdout lock: the launcher's holdout command refused before "
            "any mirror record", detail=str(got2))
        # -- status (addition C): read-only -----------------------------------
        snap = (_hlw_tree(cdir), _hlw_tree(o1), _hlw_tree(o4))
        with _HlwOpens(W["hdata"], trk_ledgers) as trk:
            s1 = L.holdout_status(cdir, o1)
            s4 = L.holdout_status(cdir, o4)
            s9 = L.holdout_status(cdir, td / "h_absent")
        cx.ok(snap == (_hlw_tree(cdir), _hlw_tree(o1), _hlw_tree(o4)) and
              not trk.opens and not os.path.lexists(td / "h_absent") and
              s1["holdout"]["complete"] is True and
              s1["mirror"]["holdout_done"]["index_sha256"] and
              s9["holdout"] is None,
              "status: holdout state reported; no file created or changed, "
              "no MRI file opened")


HLB_MIN_FREE = 80e9              # host: six real-architecture runs


def _hlb_child(spec: dict) -> dict:
    """Addition B (fresh process): ONE fixture campaign through the launcher
    -- host: REAL architecture (no model factory), 6 runs, 3 epochs,
    synthetic training data; local pre-check: stub model (disk) -- then its
    analysis chain and selection, then the launcher's holdout command on
    synthetic holdout files, and a re-entry. Timing is provisional."""
    import time as _time
    import torch
    from seqref_mri.scripts import v05lt_holdout as HO
    from seqref_mri.scripts import v05lt_prodstages as PS
    host = spec["host"]
    tmp = Path(spec["tmp"])
    t = {}
    t0 = _time.monotonic()
    W = _hlw_setup(tmp, factory="real" if host else None, max_epochs=3)
    t["campaign_and_analysis_chain_s"] = round(_time.monotonic() - t0, 1)
    out = tmp / "hout"
    r0 = PS._rng_save()
    t0 = _time.monotonic()
    res = _hlw_holdout(W, out)
    t["holdout_command_s"] = round(_time.monotonic() - t0, 1)
    rng_ok = PS._rng_equal(r0, PS._rng_save())
    t0 = _time.monotonic()
    res2 = _hlw_holdout(W, out)
    t["reentry_s"] = round(_time.monotonic() - t0, 1)
    recs = C.Ledger(out / HO.LEDGER_NAME).records()
    tasks = [{"task": r["body"]["task"][:24], "roles": r["body"]["roles"],
              "decodes": r["body"]["decodes"],
              "evaluate_s": r["body"]["evaluate_s"]}
             for r in recs if r["kind"] == "task_done"]
    return {"status": res["status"], "n_records": len(res["records"]),
            "reentry": res2.get("reentry") is True,
            "mirror": [r["kind"] for r in W["camp"].ledger.records()
                       if r["kind"].startswith("holdout_")],
            "real_architecture": "model_factory" not in W["P"]["fx"],
            "n_slices": len(W["man"]["doc"]["entries"]), "tasks": tasks,
            "rng_ok": rng_ok, "timing": t,
            "threads": torch.get_num_threads(), "backend": PS.backend_now()}


def hlb_group(cx: Ctx) -> None:
    """HLB (addition B): the launcher's holdout command end to end in a
    fresh process. Host: real architecture (six fixture runs, 3 epochs,
    synthetic training data) and synthetic holdout files -- no real holdout
    data. Local pre-check: the same path with the stub model (disk)."""
    tmpbase = Path(tempfile.gettempdir())
    free = shutil.disk_usage(tmpbase).free
    need = HLB_MIN_FREE if cx.host else 6e9
    cx.ok(free >= need, "free space for the fixture campaign",
          "FIXTURE_DISK_SPACE", f"{free / 1e9:.1f} GB free, need "
          f"{need / 1e9:.0f} GB")
    if cx.host:
        cx.phase = "host"
    with tempfile.TemporaryDirectory() as t:
        res = _run_eval_child(cx, Path(t), {"kind": "hlb",
                                            "tmp": str(Path(t) / "hlb")},
                              "hlb")
    e = _child_payload(res, "hlb")
    cx.ok(e["real_architecture"] is cx.host and e["status"] == "complete"
          and e["n_records"] == 12 and len(e["tasks"]) >= 6 and all(
              x["decodes"] == 128 * e["n_slices"] for x in e["tasks"]) and
          e["rng_ok"], ("real architecture" if cx.host else "stub model "
                        "(local pre-check)") + " through the launcher's "
          "holdout command: 12 records, 128 decodes per synthetic holdout "
          "slice per task, RNG states unchanged", "HLB_FAILED",
          str({k: e[k] for k in ("status", "n_records", "rng_ok",
                                 "real_architecture")}))
    cx.ok(e["reentry"] and e["mirror"] == ["holdout_start", "holdout_done"],
          "re-entry verified without re-evaluation; mirror records "
          "holdout_start, holdout_done", "HLB_FAILED", str(e["mirror"]))
    cx.record("HLB timing (provisional; fixture campaign; synthetic data; "
              "P4 input only)", {"timing": e["timing"], "tasks": e["tasks"],
                                 "n_slices": e["n_slices"],
                                 "threads": e["threads"],
                                 "backend": e["backend"]})


def dataclasses_replace_token(g):
    """A Gate copy with a foreign token (forged gate; fixture only)."""
    import dataclasses
    return dataclasses.replace(g, token=object())


def hnc_group(cx: Ctx) -> None:
    """HNC (addition C): no holdout MRI file is opened before the gate --
    refused gates open nothing; a valid run opens holdout files only after
    `gate_passed` is in the ledger."""
    import h5py
    from seqref_mri.scripts import v05lt_holdout as HO
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        H = _hrn_setup(td)
        hroot = str(H["hdata"].resolve())
        opens = []
        orig = h5py.File
        out = td / "out"

        def gates():
            p = out / HO.LEDGER_NAME
            if not p.exists():
                return 0
            return sum(r["kind"] == "gate_passed"
                       for r in C.Ledger(p).records())

        class Tracked(orig):
            def __init__(self, name, *a, **k):
                p = str(Path(name).resolve())
                if p.startswith(hroot):
                    opens.append((p, gates()))
                super().__init__(name, *a, **k)
        h5py.File = Tracked
        try:
            bad = [
                ((H["approval"][0], "0" * 64), {}, "unapproved approval"),
                ((td / "missing.json", "0" * 64), {}, "missing approval"),
                (_hrn_forged(H, "sel", lambda d: d.update(
                    selection_sha256="1" * 64)), {}, "approval mis-bound"),
                (_hrn_forged(H, "lch", lambda d: d.update(
                    launch_chain_sha256="4" * 64)), {},
                 "launch chain unapproved"),
                (_hrn_forged(H, "ach", lambda d: d.update(
                    analysis_chain_sha256="5" * 64)), {},
                 "analysis chain unapproved")]
            for apx, over, lab in bad:
                try:
                    _hrn_run(H, out, approval=apx, **over)
                    cx.ok(False, f"refusal expected: {lab}")
                except C.V05LTError as exc:
                    logger.info("HNC refused (%s): %s", lab, exc.code)
                cx.ok(not opens and not out.exists(),
                      f"no holdout file opened, nothing created: {lab}",
                      detail=f"{len(opens)} opens")
            res = _hrn_run(H, out)
        finally:
            h5py.File = orig
        vols = {e["file"] for e in H["B"]["man"]["doc"]["entries"]}
        opened = {str(Path(p).relative_to(hroot)) for p, _ in opens}
        cx.ok(res["status"] == "complete" and bool(opens) and
              all(n >= 1 for _, n in opens) and opened == vols,
              "valid run: every holdout file open happened after the "
              "recorded gate; only manifest volumes opened",
              detail=f"{len(opens)} opens, {len(opened)} files")


def hrd_group(cx: Ctx) -> None:
    """HRD (addition D): interrupted, failed and altered outputs are
    recorded halts; retry needs a decision bound to that halt and the same
    approval; failed-attempt evidence retained; owner halt is final."""
    from seqref_mri.scripts import v05lt_holdout as HO
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        H = _hrn_setup(td)
        out = td / "o1"
        n = {"k": 0}

        def kill(stage):
            n["k"] += 1
            if n["k"] == 2:
                raise _SimulatedKill(stage)
        try:
            _hrn_run(H, out, hooks={"holdout:staging": kill})
            cx.ok(False, "simulated kill expected")
        except _SimulatedKill:
            logger.info("HRD: simulated kill after staging (fixture)")
        stg = [p for p in out.iterdir() if p.name.endswith(".staging")]
        cx.ok(len(stg) == 1 and "halt" not in _hrn_kinds(out),
              "kill after staging: staging left, nothing published for that "
              "task (process kill passes through; no halt recorded)")
        cx.raises("HOLDOUT_OUTPUT_INCOMPLETE", lambda: _hrn_run(H, out),
                  "rerun: interrupted output -> recorded halt")
        cx.raises("HOLDOUT_HALT_UNRESOLVED", lambda: _hrn_run(H, out),
                  "rerun without a decision refused (clearing alone is not "
                  "enough)")
        dp = td / "dec1.json"
        dsha = HO.make_decision(dp, out, "retry", "fixture: retry after kill")
        cx.raises("HOLDOUT_DECISION_UNAPPROVED", lambda: _hrn_run(
            H, out, decision=(dp, "0" * 64)), "decision with an unapproved "
            "sha256 refused")
        res = _hrn_run(H, out, decision=(dp, dsha))
        failed = sorted(p.name for p in (out / HO.FAILED_DIR).iterdir())
        hr = [r for r in C.Ledger(out / HO.LEDGER_NAME).records()
              if r["kind"] == "halt_resolved"]
        cx.ok(res["status"] == "complete" and len(failed) == 1 and
              failed[0].endswith(stg[0].name) and
              hr[-1]["body"]["decision"] == "retry" and
              hr[-1]["body"]["retained"] and
              not any(p.name.endswith(".staging") for p in out.iterdir()),
              "bound retry: failed staging moved to _failed/ (retained, not "
              "deleted), task re-evaluated, run complete", detail=str(failed))
        cx.raises("HOLDOUT_DECISION_NOT_APPLICABLE", lambda: HO.make_decision(
            td / "dec_x.json", out, "retry", "nothing to decide"),
            "decision without an unresolved halt refused")
        # altered complete output (consistent hashes) -> mismatch halt
        d = sorted(p for p in out.iterdir() if p.is_dir() and
                   p.name[0].isdigit())[0]
        role = sorted(f.name for f in d.iterdir()
                      if f.name.endswith(".json"))[0]
        _hrn_rewrite(d, role, lambda doc: doc["masks"].update(
            widths=[97] + doc["masks"]["widths"][1:]))
        gd = HO.verify_gate(H["approval"], **{k: v for k, v in H["kw"].items()
                                              if k != "data_root"})
        td_ = [t for t in HO.plan_tasks(H["B"]["sel"])
               if t["name"] == d.name][0]
        mk = HO.check_widths(gd, HO.holdout_view(gd, H["hdata"]))
        cx.raises("HOLDOUT_OUTPUT_MISMATCH", lambda: HO.verify_task(
            out, td_, gd, mk), "verify_task: recorded widths != loaded "
            "widths (consistent hashes) refused")
        cx.raises("HOLDOUT_OUTPUT_MISMATCH", lambda: _hrn_run(H, out),
                  "recorded width != loaded width (consistent hashes) -> "
                  "recorded halt")
        cx.raises("HOLDOUT_DECISION_STALE", lambda: _hrn_run(
            H, out, decision=(dp, dsha)), "the earlier (consumed) decision "
            "is stale for the new halt")
        ap2 = td / "approval2.json"
        a2sha = HO.make_approval(
            ap2, root=H["B"]["root"], launch_chain=H["P"]["chain"][:2],
            analysis_chain=H["chain"], manifest_path=H["B"]["man"]["manifest"],
            statement="second fixture approval for the same evidence (c3)",
            fixture=True, launch_root=H["P"]["tmp"])
        cx.raises("HOLDOUT_APPROVAL_CHANGED", lambda: _hrn_run(
            H, out, approval=(ap2, a2sha)), "another approval on the same "
            "output directory refused (one directory per approval)")
        dp2 = td / "dec2.json"
        d2 = HO.make_decision(dp2, out, "retry", "fixture: altered output")
        res2 = _hrn_run(H, out, decision=(dp2, d2))
        failed2 = sorted(p.name for p in (out / HO.FAILED_DIR).iterdir())
        cx.ok(res2["status"] == "complete" and any(
            f.endswith(d.name) for f in failed2) and any(
            f.endswith(HO.INDEX_NAME) for f in failed2) and
            HO.verify_task(out, [t for t in HO.plan_tasks(H["B"]["sel"])
                                 if t["name"] == d.name][0],
                           HO.verify_gate(H["approval"], **{
                               k: v for k, v in H["kw"].items()
                               if k != "data_root"}),
                           {"widths": [96] * 6, "masks_sha256":
                            json.loads((out / d.name / role).read_bytes())
                            ["masks"]["sha256"]}),
            "bound retry: altered directory and the index listing it moved "
            "to _failed/; task re-evaluated; new index", detail=str(failed2))
        # unhashed alteration -> incomplete; owner halt is final
        for f in d.iterdir():
            os.chmod(f, 0o644)
        _hrn_rewrite(d, role, lambda doc: doc.update(note="x"), sums=False)
        cx.raises("HOLDOUT_OUTPUT_INCOMPLETE", lambda: _hrn_run(H, out),
                  "file altered without its hashes -> recorded halt "
                  "(incomplete)")
        dp3 = td / "dec3.json"
        d3 = HO.make_decision(dp3, out, "halt", "fixture: owner halts")
        cx.raises("HOLDOUT_HALTED", lambda: _hrn_run(
            H, out, decision=(dp3, d3)), "owner `halt` ends the stage")
        cx.raises("HOLDOUT_HALTED", lambda: _hrn_run(H, out),
                  "after an owner halt every later run is refused")
        cx.ok(HO.status(out)["halted"] is True, "status: halted")
        # failed tasks -> recorded halts with cause / traceback
        for name, exc, code in (
                ("o2", C.V05LTError("FIXTURE_BOOM", "fixture boom"),
                 "HOLDOUT_TASK_FAILED"),
                ("o3", RuntimeError("fixture boom (unexpected)"),
                 "HOLDOUT_STAGE_ERROR")):
            def boom(t, e=exc):
                raise e
            o = td / name
            cx.raises(code, lambda o=o, b=boom: _hrn_run(
                H, o, hooks={"holdout:before_evaluate": b}),
                f"task failure ({type(exc).__name__}) -> recorded halt {code}")
            h = [r for r in C.Ledger(o / HO.LEDGER_NAME).records()
                 if r["kind"] == "halt"][-1]["body"]
            cx.ok(h["code"] == code and h["task"] and
                  (h.get("cause") == "FIXTURE_BOOM" if code ==
                   "HOLDOUT_TASK_FAILED" else "fixture boom" in
                   h.get("traceback", "")) and
                  h["approval_sha256"] == H["approval"][1],
                  f"halt record names the task, the approval and the cause "
                  f"({code})")
        o = td / "o2"
        dq = HO.make_decision(td / "dec4.json", o, "retry", "fixture retry")
        res4 = _hrn_run(H, o, decision=(td / "dec4.json", dq))
        cx.ok(res4["status"] == "complete", "bound retry after a failed "
              "task completes")


def hrb_group(cx: Ctx) -> None:
    """HRB (addition B): the runner's evaluation, publication and
    verification functions with the REAL architecture (fresh seed-2026
    build written as a trainer weights file; not a trained checkpoint) on
    the synthetic holdout files; host: registered P4 and S_ref. No real
    holdout data."""
    import time
    import torch
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_holdout as HO
    from seqref_mri.scripts import v05lt_prodstages as PS
    from seqref_mri.scripts import v05lt_train as T
    from seqref_mri.src import free_flow_runtime as ffr
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        H = _hrn_setup(td)
        g = HO.verify_gate(H["approval"], **{k: v for k, v in H["kw"].items()
                                             if k != "data_root"})
        view = HO.holdout_view(g, H["hdata"])
        masks = HO.check_widths(g, view)
        if cx.host:
            cx.phase = "host"
            s_ref = C.verify_p0s(cx.paths.p0s)
            p4 = ffr.load_p4s2_parent(str(cx.paths.repo / PS.P4_REL))
        else:
            p4, s_ref = _unit_p4(), 1.0
        model = ffr.build_model(init_seed=2026)
        ms = C.model_state_sha256(model)
        wf = H["B"]["root"] / "hrb_weights_e003.pt"
        with open(wf, "wb") as fh:
            T._weights_writer(model, _pev_meta(ms, seed=2026, arm="lam2",
                                               epoch=3))(fh)
        wsha = C.sha256_file(wf)
        del model
        m2, vck = EV.load_trained_checkpoint(wf, wsha, ms, seed=2026,
                                             arm="lam2", epoch=3)
        t0 = time.perf_counter()
        states = EV.build_states(view, list(range(len(view))), p4, s_ref)
        t_states = time.perf_counter() - t0
        rng0 = PS._rng_save()
        rows, info = HO.evaluate_task(m2, vck, states, g,
                                      g.stages.banks()["primary"])
        cx.ok(PS._rng_equal(rng0, PS._rng_save()) and
              info["decodes"] == EV.BANK_ROWS * len(states) and
              C.model_state_sha256(m2) == ms and all(
                  math.isfinite(r["r0"]["psnr"]) and
                  math.isfinite(r["r1"]["psnr"]) and
                  math.isfinite(r["r0"]["nll_per_dim"]) and
                  math.isfinite(r["r1"]["nmse_u"]) for r in rows),
              "real architecture: R0, R1 (128 decodes per slice), NLL on "
              "every synthetic holdout slice; model state and RNG unchanged",
              detail=f"{len(rows)} slices")
        task = {"arm": "lam2", "seed": 2026, "epoch": 3,
                "model_state_sha256": ms, "file": wf.name, "sha256": wsha,
                "roles": ["selected", "epoch3"],
                "name": f"2026_lam2_{ms}"}
        recs = HO.build_records(task, g, vck, rows, info, masks)
        out = td / "hrb_out"
        out.mkdir()
        HO.write_task(out, task, recs)
        got = HO.verify_task(out, task, g, masks)
        cx.ok(sorted(got) == ["epoch3", "selected"],
              "real architecture: records for both roles of one evaluation "
              "published write-once and verified")
        cx.raises("ARTIFACT_EXISTS", lambda: HO.write_task(out, task, recs),
                  "second publication refused (no overwrite)")
        cx.record("HRB timing (fixture subset; synthetic holdout files; not "
                  "a full-holdout projection)",
                  {"states_s": round(t_states, 2),
                   "evaluate_s": round(info["evaluate_s"], 2),
                   "slices": len(rows), "decodes": info["decodes"],
                   "backend": EV.backend_identity()})
        del m2, torch

# ---------------------------------------------------------------------------
# I1.10 (SEQREF-MRI-V05LTI110 v0.1): SMK -- local pre-check of the F17 smoke
# runner (v05lt_smoke.py) on synthetic sources with the stub model (synthetic
# by design; the host F17 run is the CLI `freeze` / `run` itself).
# ---------------------------------------------------------------------------
SMK_VOLS = {f"file91{i:02d}.h5": 3 + (i % 3) for i in range(1, 11)}
SMK_TRAIN = sorted(SMK_VOLS)[0::2]        # 5 volumes (first 4 = subset)
SMK_VAL = sorted(SMK_VOLS)[1::2]          # 5 volumes (first 4 = subset)


def _smk_sources(tmp: Path) -> dict:
    """Synthetic 'registered' sources (FIXTURE-ONLY): split, case manifest
    (one case inside the subset, one outside), data, banks by recipe."""
    tmp = Path(tmp)
    root, mp, msha, ex = synth_split(tmp / "src", SMK_VOLS, SMK_TRAIN, SMK_VAL)
    va = sorted(SMK_VAL)
    cases = [{"case": 0, "file": f"singlecoil_train/{va[4]}", "slice_index":
              (SMK_VOLS[va[4]] - 1) // 2},
             {"case": 1, "file": f"singlecoil_train/{va[1]}", "slice_index":
              (SMK_VOLS[va[1]] - 1) // 2}]
    cp = tmp / "src" / "case.json"
    cp.write_bytes(C.canonical_json({"schema": "fixture-case-manifest",
                                     "cases": cases}))
    bd = tmp / "src" / "banks"
    bd.mkdir()
    b = _pev_banks()
    np.save(bd / "v05lt_bank_20261023.npy", b["B"])
    np.save(bd / "v05lt_bank_20261024.npy", b["C"])
    spec = {"source_split": str(mp), "source_split_sha256": msha,
            "source_expect": list(dataclasses.astuple(ex)),
            "source_case": str(cp), "data_root": str(root),
            "bank_dir": str(bd)}
    (tmp / "smk_sources.json").write_text(json.dumps(spec))
    return spec


def _smk_ctx(arg: str) -> dict:
    """FIXTURE-ONLY context for v05lt_smoke (parent and spawned children):
    synthetic sources, unit P4, S_ref 1.0, synthetic holdout ids, stub
    model factory (_stub_w1000) and a small free-space requirement."""
    from seqref_mri.scripts import v05lt_prodstages as PS
    spec = json.loads((Path(arg) / "smk_sources.json").read_text())
    bk = Path(arg) / "smk_backend.json"
    backend = json.loads(bk.read_text()) if bk.is_file() else PS.backend_now()
    return {**spec, "source_expect": C.SplitExpect(*spec["source_expect"]),
            "holdout": sorted(SYN_HOLDOUT), "s_ref": 1.0, "p4": _unit_p4(),
            "model_factory": _stub_w1000, "min_free": 2e9,
            "backend": backend}


# Exact frozen configuration written on the host by revision 1 (T23).
SMK_HOST_CONFIG_SHA256 = \
    "9bffdcd61e4d816f45ee4a249d53191bf137e47719f2508b2da744a74efdbe80"
SMK_HOST_CONFIG_REV1 = \
    "98c33b1809ae1a8ef7654e34870ffd0ce423cb997f123fe223b4429079fb8bbf"


def _smk_adam_tests(cx: Ctx, SM, real_adam) -> None:
    """Complete optimiser-state comparison: synthetic + real resume state."""
    import copy
    import torch
    base = {"state": {0: {"step": torch.tensor(5.0),
                          "exp_avg": torch.ones(3),
                          "exp_avg_sq": torch.full((3,), 2.0)}},
            "param_groups": [{"lr": 1e-4, "betas": (0.9, 0.999),
                              "eps": 1e-8, "weight_decay": 0,
                              "amsgrad": False, "params": [0]}]}
    cx.ok(SM.adam_diff(base, copy.deepcopy(base)) is None,
          "identical optimiser states compare equal")
    for label, edit in (
            ("step counter", lambda d: d["state"][0].__setitem__(
                "step", torch.tensor(6.0))),
            ("learning rate", lambda d: d["param_groups"][0].__setitem__(
                "lr", 2e-4)),
            ("betas", lambda d: d["param_groups"][0].__setitem__(
                "betas", (0.8, 0.999))),
            ("param-group extra key", lambda d: d["param_groups"][0]
             .__setitem__("foreach", None)),
            ("param ids", lambda d: d["param_groups"][0].__setitem__(
                "params", [1])),
            ("second moment", lambda d: d["state"][0]["exp_avg_sq"]
             .add_(0.5)),
            ("step dtype", lambda d: d["state"][0].__setitem__(
                "step", torch.tensor(5.0, dtype=torch.float64)))):
        b = copy.deepcopy(base)
        edit(b)
        cx.ok(SM.adam_diff(base, b) is not None,
              f"changed optimiser {label} fails the comparison")
    k0 = sorted(real_adam["state"], key=str)[0]
    for label, edit in (
            ("step counter", lambda d: d["state"][k0].__setitem__(
                "step", d["state"][k0]["step"] + 1)),
            ("param-group lr", lambda d: d["param_groups"][0].__setitem__(
                "lr", d["param_groups"][0]["lr"] * 2))):
        b = copy.deepcopy(real_adam)
        edit(b)
        cx.ok(SM.adam_diff(real_adam, b) is not None,
              f"real resume state: changed {label} fails the comparison",
              detail=str(SM.adam_diff(real_adam, b)))


def _smk_stability_tests(cx: Ctx, SM, a: dict, expected: dict) -> None:
    import copy

    def bad(label, edit, code="SMOKE_STABILITY_INVALID", exp=expected):
        b = copy.deepcopy(a)
        edit(b)
        cx.raises(code, lambda: SM._stability_compare(a, b, exp), label)
    k0 = list(expected)[0]
    bad("missing required value (r1 PSNR = None) refused",
        lambda b: b["rows"][0].__setitem__("r1_psnr", None))
    bad("non-finite value (NaN NLL) refused",
        lambda b: b["rows"][0].__setitem__("r0_nll_per_dim", float("nan")))
    bad("infinite value refused",
        lambda b: b["rows"][1].__setitem__("r1_dc_psnr", float("inf")))
    bad("boolean as a value refused",
        lambda b: b["rows"][0].__setitem__("r0_ssim", True))
    bad("unexpected key refused",
        lambda b: b["rows"][0].__setitem__("r2_psnr", 30.0))
    bad("missing key refused", lambda b: b["rows"][0].pop("r0_eval_psnr"))
    bad("missing slice refused", lambda b: b["rows"].pop())
    bad("unexpected slice refused",
        lambda b: b["rows"].append(dict(b["rows"][0], slice_index=999)))
    bad("null NMSE_u on an eligible slice refused",
        lambda b: b["rows"][0].__setitem__("r0_nmse_u", None))
    bad("differing backends refused",
        lambda b: b["backend"].__setitem__("torch_num_threads", -1),
        "SMOKE_STABILITY_MISMATCH")
    bad("differing checkpoints refused",
        lambda b: b["checkpoint"].__setitem__("epoch", 99),
        "SMOKE_STABILITY_MISMATCH")
    exp2 = {**expected, k0: False}
    a2 = copy.deepcopy(a)
    a2["rows"][0]["r0_nmse_u"] = a2["rows"][0]["r1_nmse_u"] = None
    r = SM._stability_compare(a2, copy.deepcopy(a2), exp2)
    cx.ok(r["psnr_within"] and r["nll_within"],
          "null NMSE_u accepted where the frozen eligibility marks the slice "
          "ineligible")
    b = copy.deepcopy(a2)
    b["rows"][0]["r0_nmse_u"] = 1.0
    cx.raises("SMOKE_STABILITY_MISMATCH",
              lambda: SM._stability_compare(a2, b, exp2),
              "NMSE_u null in one process only refused")


def smk_group(cx: Ctx) -> None:
    """SMK (I1.10 addition A): freeze -> run -> record on synthetic sources
    (stub model, fixture context); refusals; comparison negatives."""
    from seqref_mri.scripts import v05lt_smoke as SM
    with tempfile.TemporaryDirectory(prefix="v05lt_smk_") as t:
        tmp = Path(t)
        spec = _smk_sources(tmp)
        fxs = {"ref": "seqref_mri.scripts.v05lt_fixtures:_smk_ctx",
               "arg": str(tmp)}
        paths = C.Paths(repo=REPO_ROOT)
        out = tmp / "smoke"
        # -- output guard
        prod = tmp / "fakerepo"
        fpaths = C.Paths(repo=prod)
        (prod / "seqref_mri" / "results" / "v05lt").mkdir(parents=True)
        cx.raises("SMOKE_TARGETS_PRODUCTION", lambda: SM.freeze(
            prod / "seqref_mri" / "results" / "v05lt" / "smk", fpaths, fxs),
            "smoke output inside results/v05lt refused")
        (tmp / "alias").symlink_to(prod / "seqref_mri" / "results" / "v05lt")
        cx.raises("SMOKE_TARGETS_PRODUCTION", lambda: SM.freeze(
            tmp / "alias" / "smk", fpaths, fxs),
            "alias through a symlinked parent into results/v05lt refused")
        (tmp / "outlink").symlink_to(tmp / "elsewhere")
        cx.raises("SMOKE_OUT_INVALID", lambda: SM.freeze(
            tmp / "outlink", paths, fxs), "symlinked smoke root refused")
        cx.raises("SMOKE_OUT_INVALID", lambda: SM.freeze(out, paths, None),
                  "production freeze outside repo/" + SM.SMOKE_REL + " refused")
        # -- freeze
        fr = SM.freeze(out, paths, fxs)
        cfgp = Path(fr["config"])
        cfg = json.loads(cfgp.read_bytes())
        tr4, va4 = sorted(SMK_TRAIN)[:4], sorted(SMK_VAL)[:4]
        cx.ok([r["volume_id"] for r in cfg["subset"]["train"]] == tr4 and
              [r["volume_id"] for r in cfg["subset"]["validation"]] == va4,
              "frozen subset = first k = 4 IDs of each sorted list")
        cx.ok(cfg["subset"]["train_slices"] == sum(SMK_VOLS[v] for v in tr4)
              and cfg["subset"]["validation_slices"] ==
              sum(SMK_VOLS[v] for v in va4), "slice counts recorded")
        pc = cfg["panel_cases"]
        cx.ok([c["volume_id"] for c in pc] == va4 and
              [c["slice_index"] for c in pc] ==
              [(SMK_VOLS[v] - 1) // 2 for v in va4] and
              [c["production_case"] for c in pc] == [None, 1, None, None],
              "panel cases = the 4 subset midpoints; production case inside "
              "the subset identified, cases outside it absent",
              detail=str([c["production_case"] for c in pc]))
        el = cfg["eligibility"]["all_slices"]
        cx.ok(el["slices"] == cfg["subset"]["validation_slices"] and
              el["eligible"] + el["excluded"] == el["slices"],
              "eligible population recorded in the configuration",
              detail=f"{el['eligible']}/{el['slices']}")
        cx.ok(cfg["epochs"] == 3 and cfg["k"] == 4 and cfg["seed"] == 2026
              and cfg["interrupt"] == SM.INTERRUPT and
              cfg["dc_tolerances"] and cfg["fixture"] is True,
              "frozen values (PA6): 3 epochs, k = 4, seed 2026, declared "
              "kill boundary, smoke tolerances")
        ok_hash = all(C.sha256_file(out / e["path"]) == e["sha256"]
                      for e in cfg["inputs"].values())
        ro = all(not (f.stat().st_mode & stat.S_IWUSR) for f in
                 (out / "freeze").rglob("*") if f.is_file())
        cx.ok(ok_hash and ro, "every frozen input re-hashes; files read-only")
        src = Path(spec["data_root"])
        cx.ok(all(C.sha256_file(src / r["file"]) == r["sha256"] ==
                  C.sha256_file(out / "freeze" / "data" / r["file"])
                  for r in cfg["subset"]["train"] +
                  cfg["subset"]["validation"]) and
              sorted(p.name for p in (out / "freeze" / "data").rglob("*.h5"))
              == sorted(tr4 + va4), "copies = sources; only the subset copied")
        cx.raises("SMOKE_EXISTS", lambda: SM.freeze(out, paths, fxs),
                  "second freeze refused (write-once)")
        sha = fr["config_sha256"]
        rc = C.sha256_file(Path(SM.__file__))
        cx.raises("SMOKE_RUN_CODE_UNAPPROVED", lambda: SM.run(
            out, paths, f"{cfgp}:{sha}", "0" * 64, fxs),
            "unapproved run code refused")
        # -- run refusals before anything is created
        cx.raises("SMOKE_CONFIG_UNAPPROVED", lambda: SM.run(
            out, paths, f"{cfgp}:{'0' * 64}", rc, fxs), "unrecorded sha256 refused")
        cx.raises("SMOKE_CONFIG_UNAPPROVED", lambda: SM.run(
            out, paths, str(cfgp), rc, fxs), "missing sha256 refused")
        other = tmp / "cfgcopy.json"
        shutil.copyfile(cfgp, other)
        cx.raises("SMOKE_CONFIG_INVALID", lambda: SM.run(
            out, paths, f"{other}:{sha}", rc, fxs),
            "configuration outside the frozen directory refused")
        zf = out / cfg["inputs"]["eligibility"]["path"]
        os.chmod(zf, 0o644)
        orig = zf.read_bytes()
        zf.write_bytes(orig + b" ")
        cx.raises("SMOKE_INPUT_HASH", lambda: SM.run(
            out, paths, f"{cfgp}:{sha}", rc, fxs), "altered frozen input refused")
        zf.write_bytes(orig)
        os.chmod(zf, 0o444)
        cur = _smk_ctx(str(tmp))["backend"]
        (tmp / "smk_backend.json").write_text(json.dumps(
            dict(cur, threads=cur["threads"] + 1)))
        cx.raises("SMOKE_BACKEND_MISMATCH", lambda: SM.run(
            out, paths, f"{cfgp}:{sha}", rc, fxs),
            "run under a backend other than the expected one refused")
        (tmp / "smk_backend.json").unlink()
        cx.raises("SMOKE_BACKEND_MISMATCH", lambda: SM.check_backend(
            {"threads": -1, "interop_threads": -1, "deterministic": True}),
            "child backend check refuses an unexpected backend")
        for lab, over, code in (
                ("child refuses an unexpected backend before any stage",
                 {"backend": dict(cur, threads=cur["threads"] + 1)},
                 "SMOKE_BACKEND_MISMATCH"),
                ("child refuses unapproved run code before any stage",
                 {"run_code_sha256": "0" * 64}, "SMOKE_RUN_CODE_UNAPPROVED")):
            sp, rp = tmp / "child_spec.json", tmp / "child_res.json"
            sp.write_text(json.dumps({
                "repo": str(REPO_ROOT), "out": str(out),
                "config": f"{cfgp}:{sha}", "chain": {}, "fixture": fxs,
                "action": "epoch0", "result": str(rp),
                "run_code_sha256": rc, "backend": cur, **over}))
            SM._child(str(sp))
            res = json.loads(rp.read_text())
            cx.ok(res.get("error", {}).get("code") == code and
                  "result" not in res, lab, detail=str(res.get("error")))
            rp.unlink()
        cx.ok(not (out / "run").exists(), "refusals created no run directory")
        cfg_before = cfgp.read_bytes()
        # -- full run
        r = SM.run(out, paths, f"{cfgp}:{sha}", rc, fxs)
        rec = json.loads(Path(r["record"]).read_bytes())
        cx.ok(cfgp.read_bytes() == cfg_before, "frozen configuration "
              "unchanged by the run")
        bk = _smk_ctx(str(tmp))["backend"]
        cx.ok(rec["checks"]["children_backend"] and
              rec["checks"]["children_run_code"] and
              all(v.get("backend") == bk for k, v in rec["results"].items()
                  if k != "interrupt:start") and
              rec["code"]["run_code_sha256"] == rc and
              rec["code"]["freeze_code_sha256"] == cfg["code"]["smoke_sha256"],
              "every child: expected backend and approved run code; freeze "
              "and run code recorded separately")
        cx.ok(all(rec["checks"].values()), "smoke run checks all true",
              detail=str(rec["checks"]))
        cx.ok(rec["interruption"]["killed_at"] == {"commits": 2,
                                                   "run_end": 0} and
              rec["results"]["interrupt:start"]["exit"] == -9 and
              rec["interruption"]["sessions"] == 2,
              "real child SIGKILL at the declared boundary (epoch-2 commit "
              "durable, no run_end); one resume")
        cx.ok(rec["interruption"]["epoch_records_identical"] and
              rec["interruption"]["adam_identical"] and
              rec["interruption"]["model_state_identical"] and
              rec["interruption"]["resume_state_identical"],
              "interrupted + resumed run = uninterrupted run apart from the "
              "registered permitted differences")
        cx.ok(all(set(v["roles"]) >= {"epoch3", "selected", "final"} and
                  v["trained"] for v in rec["item24"].values()),
              "item 24 path: trained epoch-3 / selected / final post-run "
              "records verified", detail=str(rec["item24"]))
        cx.ok(rec["stability"]["psnr_within"] and rec["stability"]["nll_within"]
              and all(Path(f).is_file() for f in rec["stability"]["files"]),
              "stability (B): two fresh processes, outputs retained",
              detail=str(rec["stability"]["bitwise_identical"]))
        rep = json.loads((out / "run" / "report" / "smoke_report.json")
                         .read_bytes())
        cx.ok(set(rep["arms"]) == {"lam0", "lam2"} and rep["seed"] == 2026 and
              "lambda effect" in rep["statements"][0] and not any(
                  k in json.dumps(rep) for k in ('"ci"', '"label"',
                                                 '"outcome"')),
              "descriptive one-seed report; no CI, labels or outcomes")
        cx.ok(rec["timing_seconds"]["total"] > 0 and
              rec["disk"]["peak_sampled_run_bytes"] > 0, "timing and disk "
              "recorded (C)")
        cx.raises("SMOKE_EXISTS", lambda: SM.run(
            out, paths, f"{cfgp}:{sha}", rc, fxs), "second run refused")
        # -- comparison negatives
        st = SM._stages(out, SM.load_config(out, f"{cfgp}:{sha}", False),
                        paths, _smk_ctx(str(tmp)))
        neg = SM.compare_runs(st.run_dir(2026, "lam2"), st.run_dir(2026,
                                                                   "lam0"))
        cx.ok(not neg["epoch_records_identical"],
              "comparison detects a different run (lam0 vs lam2)")
        cx.raises("SMOKE_INTERRUPT", lambda: SM.check_kill_boundary(
            st.run_dir(2026, "lam0")), "kill-boundary check refuses a run "
            "that is not at the declared boundary (3 commits + run_end)")
        a = json.loads(Path(rec["stability"]["files"][0]).read_bytes())
        exp = SM.expected_midpoints(SM.load_config(out, f"{cfgp}:{sha}",
                                                   False))
        b = json.loads(json.dumps(a))
        b["rows"][0]["r1_psnr"] += 0.01
        cx.ok(not SM._stability_compare(a, b, exp)["psnr_within"],
              "stability comparison detects a 0.01 dB difference")
        b = json.loads(json.dumps(a))
        b["rows"] = b["rows"][1:] + b["rows"][:1]
        cx.raises("SMOKE_STABILITY_INVALID",
                  lambda: SM._stability_compare(a, b, exp),
                  "stability comparison refuses reordered slices")
        _smk_stability_tests(cx, SM, a, exp)
        _smk_adam_tests(cx, SM, SM._final_state(st.run_dir(2026, "lam2"))[2])
        cx.raises("SMOKE_FREEZE_CODE_UNKNOWN", lambda: SM.check_freeze_code(
            {"code": {"smoke_sha256": "1" * 64}}),
            "unknown freeze code refused")
        cx.ok(SM.check_freeze_code({"code": {"smoke_sha256":
                                             SMK_HOST_CONFIG_REV1}})
              == SMK_HOST_CONFIG_REV1, "revision-1 freeze code accepted as "
              "provenance (kept as written)")
        cx.ok(SM.main(["run", "--repo", str(prod)]) == 2,
              "CLI run without --config refused")
        cx.ok(SM.main(["run", "--repo", str(prod), "--config",
                       f"x:{'0' * 64}"]) == 2,
              "CLI run without --run-code-sha256 refused")
        cx.record("SMK timing (synthetic, stub)", rec["timing_seconds"])
        for f in (out).rglob("*"):
            if f.is_file():
                os.chmod(f, 0o644)
    if cx.host:                    # exact frozen configuration on the host
        cx.phase = "host"
        hp = C.Paths(repo=cx.paths.repo)
        hout = hp.repo / SM.SMOKE_REL
        cp = hout / "freeze" / SM.CONFIG_NAME
        before = C.sha256_file(cp)
        cfg = SM.load_config(hout, f"{cp}:{SMK_HOST_CONFIG_SHA256}", True)
        cx.ok(before == C.sha256_file(cp) == SMK_HOST_CONFIG_SHA256 and
              cfg["code"]["smoke_sha256"] == SMK_HOST_CONFIG_REV1 and
              not (cp.stat().st_mode & 0o222),
              "the exact frozen host configuration loads unmodified "
              "(inputs re-verified; revision-1 freeze provenance kept)")
        cx.ok(not (hout / "run").exists(), "no smoke run directory yet")


# ---------------------------------------------------------------------------
GROUPS = {"F01": f01, "F02": f02, "F03": f03, "F04": f04, "F05": f05,
          "F07": f07, "F08": f08, "MET": met, "VAL": val,
          "RET": ret, "RES": res, "GRD": grd, "CHL": chl, "TRN": trn,
          "F09CS": f09cs, "U13": u13, "F13": f13, "F21": f21, "E0S": e0s,
          "F18": f18, "F19": f19, "F20": f20, "SCAN": scan,
          "TFX": tfx, "REP": rep, "F09": f09, "F14": f14, "F13C": f13c,
          "F21C": f21c, "F08R": f08r, "KIL": kil, "RNGC": rngc,
          "F10V": f10v, "F10": f10, "F11": f11, "F22R": f22r, "F12": f12,
          "F22Z": f22z, "F23": f23, "CPB": cpb, "PUB": pub, "E0V": e0v,
          "F16": f16, "F22E": f22e, "F21E": f21e, "CLS": cls,
          "STB": stb, "F06": f06, "F15": f15, "BSA": bsa, "BSB": bsb,
          "CFG": cfg_group, "VRF": vrf, "NLD": nld, "PEV": pev,
          "LOAD": load_group, "ANA": ana_group, "RPT": rpt_group,
          "PLT": plt_group, "E2E": e2e_group, "HOA": hoa_group,
          "HOL": hol_group, "F18A": f18a_group,
          "LCH": lch_group, "LCR": lcr_group,
          "FWT": fwt_group, "FWR": fwr, "DHS": dhs_group,
          "PRD": prd_group, "PRB": prb_group,
          "HRN": hrn_group, "HNC": hnc_group, "HRD": hrd_group,
          "HRB": hrb_group, "HRF": hrf_group,
          "HLW": hlw_group, "HLB": hlb_group,
          "SMK": smk_group}
SYNTHETIC_ONLY = {"F09CS", "U13", "SCAN", "F07", "F08", "MET", "RET",
                  "RES", "GRD", "CHL", "TFX", "F22R",
                  "F23", "PUB", "F16", "F22E", "CLS",
                  "F06", "F15", "BSA", "BSB", "CFG", "VRF",
                  "LOAD", "ANA", "RPT", "PLT", "E2E", "HOA", "HOL",
                  "F18A", "LCH", "LCR", "FWT", "PRD",
                  "HRN", "HNC", "HRD", "HRF", "HLW"}  # pure logic / stub; F23 synthetic by PA1-C; I1.8 synthetic by design


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="V05LT fixtures (I1.3-I1.8a groups)")
    ap.add_argument("groups", nargs="*")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--repo", default=str(C.Paths().repo))
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args(argv)
    paths = C.Paths(repo=Path(a.repo))
    data_root = Path(a.data_root) if a.data_root else paths.data_root
    sel = a.groups or list(GROUPS)
    unknown = [g for g in sel if g not in GROUPS]
    if unknown:
        logger.error("unknown fixture groups %s", unknown)
        return 2
    import torch
    logger.info("%s v%s host=%s repo=%s data_root=%s numpy %s torch %s "
                "threads %d python %s", __abbr__, __version__, a.host, a.repo,
                data_root, np.__version__, torch.__version__,
                torch.get_num_threads(), sys.version.split()[0])
    results, recs = {}, []
    for g in sel:
        cx = Ctx(a.host, paths, data_root)
        try:
            GROUPS[g](cx)
            done = a.host or g in SYNTHETIC_ONLY
            results[g] = ("PASS" if done else "PASS-SYNTHETIC") + \
                f" (synthetic {cx.n['synthetic']}, host {cx.n['host']})"
        except (FixtureFailure, C.V05LTError) as exc:
            logger.error("fixture %s FAILED: %s", g, exc)
            results[g] = f"FAIL ({exc.code}) at phase {cx.phase}"
        except Exception as exc:  # noqa: BLE001 -- recorded, never swallowed
            logger.exception("%s raised unexpectedly", g)
            results[g] = f"ERROR ({type(exc).__name__}) at phase {cx.phase}"
        recs += [(g, *r) for r in cx.records]
        if _RT.get("root") is not None:      # keep only shared fixtures
            for d in _RT["root"].iterdir():
                if d.is_dir() and d.name not in ("ref", "epoch0"):
                    shutil.rmtree(d)
    if _RT.get("td") is not None:
        _RT["td"].cleanup()
    print(f"\n=== V05LT fixtures ({__abbr__} v{__version__}) -- paste back ===")
    for g, ph, label, val in recs:
        print(f"RECORD {g} [{ph}] {label}: {val}")
    for g, r in results.items():
        print(f"FIXTURE {g}: {r}")
    bad = [g for g, r in results.items() if not r.startswith("PASS")]
    print(f"RESULT {'FAIL' if bad else 'PASS'} groups={len(results)} "
          f"host={a.host}" + (f" failed={bad}" if bad else ""))
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
