# SEQREF-MRI-V05LTPS v0.4 -- scripts.v05lt_prodstages
# LIFETIME: KEEP
# =============================================================================
# Purpose: production stage set for the V05LT launcher (I1.9 c2 production
#   wiring; plan SEQREF-MRI-V05LTI19C2P v0.1, D1(a), refinements 1-6). It
#   implements the launcher's stage interface with the accepted components
#   only (no new science):
#     bind_chain     launch-inputs record (`launch_inputs` chain role,
#                    schema seqref-v05lt-launch-inputs/1) verified against
#                    the verified launch chain: both hashes, bindings to the
#                    pin / gate-2a / P5 chain entries, the registered
#                    campaign configuration, the code files, every input
#                    file (data-content manifest, eligibility, ZF, case
#                    manifest, banks B/C, split manifest) and valid P5 DC
#                    tolerances (refinement 1);
#     verify_data    v05lt_datahash.verify on the bound manifest (PA5-A);
#     epoch0_done /  epoch-0 checkpoint (v05lt_epoch0.create) and epoch-0
#     epoch0         evaluation (v05lt_eval.epoch0_evaluate /
#                    write_epoch0_eval) on the full populations; outputs
#                    classified absent / complete / incomplete; complete
#                    outputs verified against the checkpoint, population,
#                    bank, tolerances and cases (refinement 3); RNG states
#                    saved and restored around the stage (refinement 6);
#     expected_config / run_dir / train
#                    registered TrainConfig (100 epochs, full data) and
#                    v05lt_train.Run start / resume;
#     post_run       final weights (v05lt_final.extract_final, PA5-B), then
#                    the accepted planner (v05lt_eval.plan_campaign:
#                    deduplication, primary-bank prefixes, banks B/C at
#                    selected only, panels) for epoch 3 / selected / final;
#                    each record checked against its expected task, then
#                    every role through the accepted results loader
#                    (v05lt_results.load_posteval_for) (refinement 4).
#   Production mode (fixture=None) enforces the registered configuration,
#   populations, input paths, pin and backend (refinement 2). Fixture mode
#   (a dict of FIXTURE_KEYS) is for synthetic / fixture-layout tests only:
#   temporary layout root outside results/v05lt, model factory, shorter
#   budget, data subsets; the launcher refuses fixture-mode stages in
#   production and any other stage class.
# Recovery (refinement 3): absent outputs are produced; complete outputs are
#   verified and reused (never rewritten); incomplete or mismatched outputs
#   raise PROD_*_INCOMPLETE / PROD_*_MISMATCH, which the launcher records as
#   a campaign halt needing a bound owner decision; nothing is deleted or
#   overwritten here (owner repairs, e.g. v05lt_epoch0 --recover, then a
#   `retry` decision).
# CONVENTION: logger.error + V05LTError(code); no fallback, no placeholder,
#   no mock. Fixture-only parameters are named and refused in production.
# Changelog (v0.4, 2026-10-26; failed host run of revision 3 / review):
#   * Loaded-module check: third-party modules (name not exactly seqref_mri
#     and not starting with "seqref_mri.") loaded from the excluded folders
#     inside the package (e.g. the host venv seqref_mri/.venv_mri) or from
#     outside it are not pin-checked (environment record); seqref_mri
#     modules must still resolve inside the tree outside the excluded
#     folders; any module loaded from the source tree (not an excluded
#     folder) must resolve to a pinned file -- a symlink detour into an
#     excluded folder or outside is refused. Fixture-only `modules`
#     argument (test a given module table instead of sys.modules).
# Changelog (v0.3, 2026-10-26; c2-wiring revision-2 review fix):
#   * Symlinks in the execution-source tree (directories or files, outside
#     the excluded directories) are refused before directories are skipped;
#     the package directory must not resolve elsewhere.
#   * Loaded-module check: every module named seqref_mri[.*], and every
#     module whose path (as given or resolved) lies in the package, must
#     resolve inside the approved package tree, outside the excluded
#     directories, and be pinned with an unchanged hash; a module executing
#     from an unpinned external location is refused.
# Changelog (v0.2, 2026-10-26; c2-wiring revision-1 review fixes):
#   * Pin covers the complete execution-source set: every *.py under
#     seqref_mri/ (model, objective, MRI operator, data, metrics, scripts)
#     and every file under seqref_mri/configs/ (SOURCE rules below), not 12
#     scripts. Production: each file = its blob at the pinned HEAD and the
#     set = the files tracked there (untracked or changed -> refused).
#     Modules loaded at run time must lie in the pinned set with unchanged
#     hashes (checked at binding and after every stage).
#   * Reused post-run records: the mask identity is recomputed independently
#     (canonical_mask_seed(EVAL_BASE_SEED, file, slice) + make_cartesian_mask
#     at the loaded slice width, checked against the loaded slice), and the
#     recorded code and backend must equal this approved execution; epoch-0
#     evaluation context also binds code and backend.
#   * P5 tolerance file: registered path
#     results/v05lt/preflight/v05lt_p5_dc_tolerances.json (proposed for the
#     record).
# Changelog (NEW in v0.1, 2026-10-26; SEQREF-MRI-V05LTI19C2P v0.1):
#   * Introduced: ProductionStages, launch-inputs record (make / verify),
#     registered campaign configuration, backend check, absent / complete /
#     incomplete classification of epoch-0 and post-run outputs, per-task
#     post-run record checks.
# =============================================================================
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from seqref_mri.scripts import v05lt_common as C

logger = logging.getLogger("seqref_mri.v05lt_prodstages")

__version__ = "0.4"
__abbr__ = "SEQREF-MRI-V05LTPS"

SEEDS = (2026, 2027, 2028)
ARMS = ("lam0", "lam2")
LAUNCH_INPUTS_ROLE = "launch_inputs"
LAUNCH_INPUTS_SCHEMA = "seqref-v05lt-launch-inputs/1"
TOLERANCES_SCHEMA = "seqref-v05lt-dc-tolerances/1"
BOUND_ROLES = ("pin", "gate2a", "P5")
P4_REL = "seqref_mri/results/_diag/p4/scaling_statistics.json"
_I = "seqref_mri/results/v05lt/inputs/"
INPUT_KEYS = ("split_manifest", "case_manifest", "bank_B", "bank_C",
              "data_content", "eligibility", "zf", "dc_tolerances")
# Registered locations (relative to the repository).
REGISTERED_REL = {
    "dc_tolerances":
        "seqref_mri/results/v05lt/preflight/v05lt_p5_dc_tolerances.json",
    "split_manifest": _I + "v05lt_split_manifest.json",
    "case_manifest": _I + "v05lt_case_manifest.json",
    "bank_B": _I + "v05lt_bank_20261023.npy",
    "bank_C": _I + "v05lt_bank_20261024.npy",
    "data_content": _I + "v05lt_data_content.json",
    "eligibility": _I + "v05lt_nmse_eligibility.json",
    "zf": "seqref_mri/results/v05lt/zf/zf_reference.json"}
REGISTERED_SHA = {
    "split_manifest": C.REGISTERED_INPUTS["v05lt_split_manifest.json"],
    "case_manifest": C.REGISTERED_INPUTS["v05lt_case_manifest.json"],
    "bank_B": C.REGISTERED_INPUTS["v05lt_bank_20261023.npy"],
    "bank_C": C.REGISTERED_INPUTS["v05lt_bank_20261024.npy"]}
BANK_FILES = {"bank_B": ("B", "v05lt_bank_20261023.npy"),
              "bank_C": ("C", "v05lt_bank_20261024.npy")}
# Registered V05LT CPU setup (T3 / STB): checked in production, recorded in
# fixtures. Never set here.
BACKEND = {"threads": 12, "interop_threads": 12, "deterministic": False}
# Execution-source set bound to the pin (v0.2): every *.py under
# seqref_mri/ and every file under seqref_mri/configs/, excluding the
# virtual environment, results, data and byte-code caches. Third-party
# packages (torch, numpy, h5py) are covered by the environment / backend
# records, not by the pin.
SOURCE_PKG = "seqref_mri"
SOURCE_EXCLUDE = frozenset({".venv_mri", "results", "data", "__pycache__"})
SOURCE_CONFIG_DIR = "configs"
FIXTURE_KEYS = frozenset({"root", "model_factory", "max_epochs",
                          "train_indices", "val_indices", "split_expect",
                          "holdout", "p4", "s_ref", "repo", "hooks"})
_DOC_KEYS = {"schema", "abbr", "version", "bindings", "pin",
             "campaign_config_sha256", "inputs", "fixture"}


def _rel_ok(p) -> str:
    C.require(isinstance(p, str) and p and not p.startswith("/") and
              all(x not in ("", ".", "..") for x in p.split("/")),
              "PROD_INPUT_PATH", f"unsafe relative path {p!r}")
    return p


def _git(repo: Path, *args) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                       text=True)
    C.require(r.returncode == 0, "PROD_PIN_MISMATCH",
              f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def _excluded(rel: Path) -> bool:
    return any(part in SOURCE_EXCLUDE for part in Path(rel).parts)


def source_identity(repo: Path) -> dict:
    """{repo-relative posix path: sha256} of every *.py under seqref_mri/
    and every file under seqref_mri/configs/ (exclusions: SOURCE_EXCLUDE;
    symlinks refused)."""
    repo = Path(repo)
    pkg = repo / SOURCE_PKG
    C.require(pkg.is_dir(), "PROD_PIN_MISMATCH", f"{pkg}: no source tree")
    C.require(not pkg.is_symlink() and pkg.resolve() ==
              repo.resolve() / SOURCE_PKG, "PROD_PIN_MISMATCH",
              f"{pkg}: package directory resolves elsewhere")
    out = {}
    for p in sorted(pkg.rglob("*")):
        rel = p.relative_to(repo)
        if _excluded(rel):
            continue
        C.require(not p.is_symlink(), "PROD_PIN_MISMATCH",
                  f"{rel}: symlink in the execution-source tree")
        if p.is_dir():
            continue
        in_cfg = rel.parts[1:2] == (SOURCE_CONFIG_DIR,)
        if p.suffix != ".py" and not in_cfg:
            continue
        C.require(p.is_file() and not p.is_symlink(), "PROD_PIN_MISMATCH",
                  f"{rel}: not a regular file")
        out[rel.as_posix()] = C.sha256_file(p)
    return out


def git_blob_sha1(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def head_blobs(repo: Path, commit: str) -> dict:
    """{path: blob sha1} of the execution-source set tracked at `commit`."""
    out = {}
    for line in _git(repo, "ls-tree", "-r", "--full-tree", commit, "--",
                     SOURCE_PKG).splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) != 3 or parts[1] != "blob":
            continue
        rel = Path(path)
        in_cfg = rel.parts[1:2] == (SOURCE_CONFIG_DIR,)
        if _excluded(rel) or (rel.suffix != ".py" and not in_cfg):
            continue
        out[rel.as_posix()] = parts[2]
    return out


def backend_now() -> dict:
    import torch
    return {"threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic": bool(torch.are_deterministic_algorithms_enabled())}


def check_backend(got: dict) -> dict:
    """Production: the registered CPU setup exactly (never changed here)."""
    diff = {k: (got.get(k), v) for k, v in BACKEND.items() if got.get(k) != v}
    C.require(not diff, "PROD_BACKEND_MISMATCH", f"(got, registered) {diff}")
    return got


def registered_config(seed: int, arm: str, run_dir: Path,
                      epoch0_model_state: str):
    """The registered production TrainConfig (194 §4-§6; full budget and
    data; fixture False)."""
    from seqref_mri.scripts import v05lt_train as T
    C.require(seed in SEEDS and arm in ARMS, "PROD_CONFIG_INVALID",
              f"{seed}/{arm}")
    return T.TrainConfig(seed=seed, arm=arm, run_dir=str(run_dir),
                         epoch0_model_state=epoch0_model_state)


def _rng_save() -> dict:
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone()}


def _rng_restore(s: dict) -> None:
    import torch
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])


def _rng_equal(a: dict, b: dict) -> bool:
    import torch
    na, nb = a["numpy"], b["numpy"]
    return (a["python"] == b["python"] and na[0] == nb[0] and
            np.array_equal(na[1], nb[1]) and tuple(na[2:]) == tuple(nb[2:])
            and torch.equal(a["torch"], b["torch"]))


def _strict_json(path: Path, code: str):
    try:
        return json.loads(Path(path).read_bytes(),
                          parse_constant=lambda c: C.fail(
                              code, f"{path}: non-finite constant {c}"))
    except (OSError, ValueError) as exc:
        C.fail(code, f"{path}: {type(exc).__name__}: {exc}")


class ProductionStages:
    """Launcher stage set. fixture=None: production (registered everything).
    fixture={...}: FIXTURE_KEYS only, synthetic / fixture-layout tests."""

    def __init__(self, paths: C.Paths, data_root: Path, *,
                 fixture: dict | None = None):
        self.paths, self.data_root = paths, Path(data_root)
        self.production = fixture is None
        fx = {} if fixture is None else dict(fixture)
        C.require(isinstance(fx, dict) and set(fx) <= FIXTURE_KEYS,
                  "PROD_FIXTURE_KEY", f"unknown fixture keys "
                  f"{sorted(set(fx) - FIXTURE_KEYS)}")
        if not self.production:
            C.require("root" in fx, "PROD_FIXTURE_KEY", "fixture root required")
            root = Path(fx["root"]).resolve()
            prod = paths.v05lt.resolve()
            C.require(prod not in root.parents and root != prod,
                      "FIXTURE_TARGETS_PRODUCTION", str(root))
            self.root = Path(fx["root"])
        else:
            self.root = paths.v05lt
        self.fx = fx
        self.repo = Path(fx.get("repo", paths.repo))
        self.code_dir = Path(__file__).resolve().parent
        self.inputs = None              # set by bind_chain
        self._data = self._man = self._banks = self._masks = None

    # -- configuration ------------------------------------------------------
    def campaign_config(self) -> dict:
        """Registered campaign configuration (production) or the fixture
        configuration; its sha256 is bound in the launch-inputs record."""
        from seqref_mri.scripts import v05lt_eval as EV
        from seqref_mri.scripts import v05lt_rules as R
        from seqref_mri.scripts import v05lt_train as T
        cfg = {"seeds": list(SEEDS), "arms": {a: T.ARMS[a] for a in ARMS},
               "max_epochs": R.MAX_EPOCHS, "lr": T.LEARNING_RATE,
               "batch": T.BATCH, "train_population": "all",
               "validation_population": "all",
               "posteval_population": "all midpoints",
               "roles": list(EV.ROLES), "prefixes": list(EV.PREFIXES),
               "banks": dict(EV.REGISTERED_BANK_ARRAY),
               "backend": dict(BACKEND), "fixture": False}
        if not self.production:
            cfg.update({"fixture": True,
                        "max_epochs": self.fx.get("max_epochs", R.MAX_EPOCHS),
                        "train_population": self.fx.get("train_indices"),
                        "validation_population": self.fx.get("val_indices"),
                        "backend": "recorded only (fixture)"})
        return cfg

    def config_sha256(self) -> str:
        return C.sha256_bytes(C.canonical_json(self.campaign_config()))

    def code_identity(self) -> dict:
        """{repo-relative path: sha256} of the execution-source set under
        self.repo (SOURCE rules)."""
        return source_identity(self.repo)

    def _check_loaded(self, pin_code: dict, root: Path | None = None,
                      modules: dict | None = None) -> None:
        """Loaded-module rule (v0.4):
          * seqref_mri modules (name == "seqref_mri" or starting with
            "seqref_mri.") must resolve inside the package tree, outside the
            excluded folders, to a pinned file with an unchanged hash;
          * any other module whose path as given lies in the source tree
            (not an excluded folder) must resolve the same way (a symlink
            detour into an excluded folder or outside is refused);
          * any other module resolving into the source tree is pin-checked;
          * otherwise (third-party, e.g. from seqref_mri/.venv_mri or
            outside the repository) it is covered by the environment record.
        `root` (running code tree) and `modules` (instead of sys.modules)
        are fixture-only."""
        root = (self.code_dir.parents[1] if root is None else
                Path(root)).resolve()
        pkg = root / SOURCE_PKG
        table = sys.modules if modules is None else modules

        def where(p: Path) -> str:      # "src" | "excluded" | "out"
            if pkg not in p.parents:
                return "out"
            return "excluded" if _excluded(p.relative_to(root)) else "src"
        bad = []
        for name, mod in list(table.items()):
            f = getattr(mod, "__file__", None)
            ours = name == SOURCE_PKG or name.startswith(SOURCE_PKG + ".")
            if not f:
                if ours and getattr(mod, "__path__", None) is not None:
                    paths = [Path(x) for x in mod.__path__]
                    if any(pkg != x.resolve() and where(x.resolve()) != "src"
                           for x in paths):
                        bad.append(f"{name}: package path outside the tree")
                continue
            given = Path(os.path.abspath(f))
            fp = given.resolve()
            wg, wr = where(given), where(fp)
            if not (ours or wg == "src" or wr == "src"):
                continue                     # third-party (environment)
            if wr != "src":
                bad.append(f"{name}: {f} resolves outside the approved tree "
                           f"({wr})")
                continue
            rel = fp.relative_to(root).as_posix()
            if rel not in pin_code or C.sha256_file(fp) != pin_code[rel]:
                bad.append(rel)
        C.require(not bad, "PROD_PIN_MISMATCH", f"loaded modules outside the "
                  f"pinned set or changed since binding: {sorted(bad)[:5]}")

    # -- launch inputs (refinement 1) ----------------------------------------
    def _check_pin(self, pin: dict) -> None:
        C.require(isinstance(pin, dict) and set(pin) == {"commit", "code"} and
                  isinstance(pin["commit"], str) and
                  isinstance(pin["code"], dict) and pin["code"],
                  "PROD_LAUNCH_INPUTS_INVALID", "pin structure")
        got = self.code_identity()
        diff = sorted(set(got) ^ set(pin["code"])) + sorted(
            f for f in set(got) & set(pin["code"]) if got[f] != pin["code"][f])
        C.require(not diff, "PROD_PIN_MISMATCH", f"execution sources differ "
                  f"from the pinned identity: {diff[:6]} ({len(diff)})")
        if self.production:
            head = _git(self.repo, "rev-parse", "HEAD")
            C.require(head == pin["commit"], "PROD_PIN_MISMATCH",
                      f"HEAD {head} != pin {pin['commit']}")
            tracked = head_blobs(self.repo, head)
            off = sorted(f for f in got if tracked.get(f) !=
                         git_blob_sha1(self.repo / f))
            C.require(not off, "PROD_PIN_MISMATCH", f"execution sources not "
                      f"identical to the pinned commit (changed or "
                      f"untracked): {off[:6]} ({len(off)})")
            missing = sorted(f for f in tracked if f not in got)
            C.require(not missing, "PROD_PIN_MISMATCH", f"pinned execution "
                      f"sources missing on disk: {missing[:6]}")

    def _check_inputs(self, ins: dict, root: Path) -> dict:
        C.require(isinstance(ins, dict) and set(ins) == set(INPUT_KEYS),
                  "PROD_LAUNCH_INPUTS_INVALID",
                  f"inputs {sorted(ins) if isinstance(ins, dict) else ins} "
                  f"!= {list(INPUT_KEYS)}")
        out = {}
        for k in INPUT_KEYS:
            e = ins[k]
            C.require(isinstance(e, dict) and set(e) == {"path", "sha256"} and
                      isinstance(e["sha256"], str) and len(e["sha256"]) == 64,
                      "PROD_LAUNCH_INPUTS_INVALID", f"input {k}: {e!r:.120}")
            rel = _rel_ok(e["path"])
            if self.production:
                C.require(rel == REGISTERED_REL[k], "PROD_INPUT_PATH",
                          f"{k}: {rel} != registered {REGISTERED_REL[k]}")
                if k in REGISTERED_SHA:
                    C.require(e["sha256"] == REGISTERED_SHA[k],
                              "PROD_INPUT_HASH", f"{k}: != registered")
            p = Path(root) / rel
            C.require(p.is_file() and not p.is_symlink(), "PROD_INPUT_MISSING",
                      f"{k}: {p}")
            got = C.sha256_file(p)
            C.require(got == e["sha256"], "PROD_INPUT_HASH",
                      f"{k}: {got} != recorded {e['sha256']}")
            out[k] = {"path": p, "sha256": got}
        for k in ("data_content", "eligibility"):
            side = Path(str(out[k]["path"]) + ".sha256")
            C.require(C.read_sidecar(side, "PROD_INPUT_SIDECAR") ==
                      out[k]["sha256"], "PROD_INPUT_SIDECAR", f"{k}: {side}")
        from seqref_mri.scripts import v05lt_eval as EV
        C.require(out["zf"]["path"].name == EV.ZF_FILE, "PROD_INPUT_PATH",
                  f"zf: {out['zf']['path'].name}")
        try:
            zsha = EV.verify_zf(out["zf"]["path"].parent)
        except (OSError, UnicodeDecodeError) as exc:
            C.fail("PROD_INPUT_SIDECAR", f"zf: {type(exc).__name__}: {exc}")
        C.require(zsha == out["zf"]["sha256"], "PROD_INPUT_HASH", "zf pair")
        tdoc = _strict_json(out["dc_tolerances"]["path"], "PROD_TOLERANCES")
        C.require(isinstance(tdoc, dict) and set(tdoc) ==
                  {"schema", "tolerances", "provenance"} and
                  tdoc["schema"] == TOLERANCES_SCHEMA and
                  isinstance(tdoc["provenance"], dict),
                  "PROD_TOLERANCES", "P5 tolerance file structure")
        try:
            out["tolerances"] = EV.check_tolerances(tdoc["tolerances"])
        except C.V05LTError as exc:
            C.fail("PROD_TOLERANCES", f"P5 tolerances invalid: {exc.code}: "
                   f"{exc}")
        for k, (name, fname) in BANK_FILES.items():
            exp = C.REGISTERED_INPUTS[fname] if self.production \
                else out[k]["sha256"]
            out[f"array_{name}"] = C.verify_bank_npy(
                out[k]["path"], exp, C.BANK_ARRAY_SHA256[fname])
        return out

    def bind_chain(self, chain_doc: dict, chain_root: Path) -> dict:
        """Verify the launch-inputs record named by the (already verified)
        launch chain; binds the stage set to it."""
        if self.production:
            self.backend = check_backend(backend_now())
        else:
            self.backend = backend_now()
        ents = {e["role"]: e for e in chain_doc.get("entries", [])}
        C.require(LAUNCH_INPUTS_ROLE in ents and all(r in ents for r in
                                                     BOUND_ROLES),
                  "PROD_LAUNCH_INPUTS_MISSING", f"chain roles {sorted(ents)}")
        ent = ents[LAUNCH_INPUTS_ROLE]
        p = Path(chain_root) / _rel_ok(ent["path"])
        C.require(p.is_file() and not p.is_symlink() and
                  C.sha256_file(p) == ent["sha256"],
                  "PROD_LAUNCH_INPUTS_HASH", f"{p}: != chain entry")
        doc = _strict_json(p, "PROD_LAUNCH_INPUTS_INVALID")
        C.require(isinstance(doc, dict) and set(doc) == _DOC_KEYS and
                  doc["schema"] == LAUNCH_INPUTS_SCHEMA,
                  "PROD_LAUNCH_INPUTS_INVALID", f"{p}: schema / keys")
        C.require(doc["fixture"] is (not self.production),
                  "PROD_LAUNCH_INPUTS_MODE", f"record fixture="
                  f"{doc['fixture']!r}, stages production={self.production}")
        want = {r: ents[r]["sha256"] for r in BOUND_ROLES}
        C.require(doc["bindings"] == {f"{r}_sha256": v for r, v in
                                      want.items()},
                  "PROD_LAUNCH_INPUTS_BINDING", f"record bindings "
                  f"{doc['bindings']} != chain entries {want}")
        C.require(doc["campaign_config_sha256"] == self.config_sha256(),
                  "PROD_LAUNCH_INPUTS_CONFIG", "record bound to another "
                  "campaign configuration")
        self._check_pin(doc["pin"])
        ins = self._check_inputs(doc["inputs"], Path(chain_root))
        self._check_loaded(doc["pin"]["code"])
        self.inputs = {**ins, "record_sha256": ent["sha256"],
                       "pin_code": doc["pin"]["code"]}
        self._masks = None
        self._data = self._man = self._banks = None
        logger.info("[%s] launch inputs %s bound (production=%s)", __abbr__,
                    ent["sha256"][:12], self.production)
        return {"launch_inputs_sha256": ent["sha256"],
                "backend": self.backend}

    def _bound(self) -> dict:
        C.require(self.inputs is not None, "PROD_STAGES_UNBOUND",
                  "bind_chain() must verify the launch inputs first")
        return self.inputs

    # -- data -----------------------------------------------------------------
    def _split_expect(self):
        return self.fx.get("split_expect", C.REGISTERED_SPLIT)

    def _holdout(self):
        if "holdout" in self.fx:
            return frozenset(self.fx["holdout"])
        return C.holdout_ids(self.paths.v02_manifests)

    def manifests(self) -> dict:
        """Accepted results-loader view of split, eligibility and cases."""
        from seqref_mri.scripts import v05lt_results as RS
        ins = self._bound()
        if self._man is None:
            self._man = RS.load_manifests({
                "split_manifest": ins["split_manifest"]["path"],
                "split_sha256": ins["split_manifest"]["sha256"],
                "split_expect": self._split_expect(),
                "eligibility": ins["eligibility"]["path"],
                "eligibility_sha256": ins["eligibility"]["sha256"],
                "case_manifest": ins["case_manifest"]["path"],
                "case_sha256": ins["case_manifest"]["sha256"]},
                RS.Identities())
        return self._man

    def data(self) -> dict:
        from seqref_mri.scripts import v05lt_metrics as M
        ins = self._bound()
        if self._data is None:
            sdoc = C.load_split_manifest(ins["split_manifest"]["path"],
                                         ins["split_manifest"]["sha256"],
                                         self._split_expect())
            tv, vv = C.build_split_views(self.data_root, sdoc,
                                         self._holdout(),
                                         self._split_expect())
            if self.production:
                from seqref_mri.src import free_flow_runtime as ffr
                s_ref = C.verify_p0s(self.paths.p0s)
                p4 = ffr.load_p4s2_parent(str(self.repo / P4_REL))
            else:
                s_ref, p4 = self.fx["s_ref"], self.fx["p4"]
            edoc = _strict_json(ins["eligibility"]["path"],
                                "PROD_INPUT_INVALID")
            elig = M.eligibility_index(edoc)
            want = {vv.entry(i) for i in range(len(vv))}
            C.require(set(elig) == want, "PROD_INPUT_INVALID",
                      "eligibility slices != validation slices")
            self._data = {"tv": tv, "vv": vv, "p4": p4, "s_ref": s_ref,
                          "eligibility": elig}
        return self._data

    def banks(self) -> dict:
        from seqref_mri.scripts import v05lt_eval as EV
        from seqref_mri.tdiag import estimators as est
        ins = self._bound()
        if self._banks is None:
            prim = est.z_diag_bank()
            C.require(prim["bank_sha256"] == C.ZDIAG_BANK_SHA256 and
                      EV._array_sha(prim["bank"]) ==
                      EV.REGISTERED_BANK_ARRAY["primary"],
                      "ZDIAG_MISMATCH", "primary bank")
            self._banks = {"primary": prim["bank"], "B": ins["array_B"],
                           "C": ins["array_C"]}
        return self._banks

    def expected_masks_sha256(self) -> str:
        """Independent mask identity of the post-run population (midpoints,
        accepted-loader order): seed = canonical_mask_seed(EVAL_BASE_SEED,
        file, slice), mask = make_cartesian_mask(width, seed) at the width of
        the loaded slice; the loaded slice's own mask and seed must agree.
        Same byte layout as v05lt_eval.posteval_evaluate (C-order bool)."""
        from seqref_mri.src import fastmri_data as FD
        if self._masks is None:
            vv = self.data()["vv"]
            pos = {vv.entry(i): i for i in range(len(vv))}
            h = hashlib.sha256()
            for f, s in self.manifests()["mid_ids"]:
                C.require((f, s) in pos, "PROD_MASK_MISMATCH",
                          f"{f}|{s} not in the validation view")
                item = vv[pos[(f, s)]]
                seed = FD.canonical_mask_seed(FD.EVAL_BASE_SEED, f, int(s))
                w = int(item["y"].shape[-1])
                m = FD.make_cartesian_mask(w, seed).astype(bool)
                got = np.asarray(item["mask"]).astype(bool).reshape(-1)
                C.require(int(item["meta"]["mask_seed"]) == seed and
                          np.array_equal(got, m), "PROD_MASK_MISMATCH",
                          f"{f}|{s}: loaded mask / seed != registered rule")
                h.update(np.ascontiguousarray(m).tobytes(order="C"))
            self._masks = h.hexdigest()
        return self._masks

    def _factory(self):
        return self.fx.get("model_factory")

    def _hook(self, name: str):
        return (self.fx.get("hooks") or {}).get(name)

    # -- launcher interface ---------------------------------------------------
    def verify_data(self) -> dict:
        from seqref_mri.scripts import v05lt_datahash as DH
        ins = self._bound()
        return DH.verify(ins["data_content"]["path"],
                         ins["data_content"]["sha256"],
                         ins["split_manifest"]["path"],
                         ins["split_manifest"]["sha256"], self._holdout(),
                         self.data_root, expect=self._split_expect())

    def run_dir(self, seed: int, arm: str) -> Path:
        return self.root / f"{seed}_{arm}"

    def _e0_dir(self, seed: int) -> Path:
        return self.root / "epoch0" / f"seed_{seed}"

    def epoch0_checkpoint(self, seed: int):
        """Read-only. None (absent) | verified committed checkpoint dict;
        anything else -> PROD_EPOCH0_INCOMPLETE."""
        from seqref_mri.scripts import v05lt_epoch0 as E0
        d = self._e0_dir(seed)
        if not os.path.lexists(d):
            return None
        led = C.Ledger(d / "ledger.jsonl")
        try:
            recs = led.records() if led.path.is_file() else []
        except C.V05LTError as exc:
            C.fail("PROD_EPOCH0_INCOMPLETE", f"seed {seed}: ledger "
                   f"{exc.code}")
        commits = [r["body"] for r in recs if r["kind"] == "commit"]
        C.require(bool(commits) and recs[-1]["kind"] == "verified",
                  "PROD_EPOCH0_INCOMPLETE", f"seed {seed}: checkpoint not "
                  "committed and verified (owner: v05lt_epoch0 --recover)")
        body, ver = commits[-1], recs[-1]["body"]
        name = E0.ckpt_name(seed)
        ret = body.get("retained", {}).get("epoch0", {})
        ck = d / "ckpt"
        try:
            listing = {p.name for p in ck.iterdir()}
        except OSError as exc:
            C.fail("PROD_EPOCH0_INCOMPLETE", f"seed {seed}: {exc}")
        C.require(body.get("seed") == seed and ret.get("name") == name and
                  listing == {name, name + ".sha256"},
                  "PROD_EPOCH0_INCOMPLETE", f"seed {seed}: store contents")
        C.require(C.read_sidecar(ck / (name + ".sha256"),
                                 "PROD_EPOCH0_INCOMPLETE") == ret["sha256"]
                  and C.sha256_file(ck / name) == ret["sha256"],
                  "PROD_EPOCH0_INCOMPLETE", f"seed {seed}: file hash")
        C.require(ver.get("file_sha256") == ret["sha256"] and
                  ver.get("model_state_sha256") == body["model_state_sha256"]
                  and ver.get("seed") == seed, "PROD_EPOCH0_MISMATCH",
                  f"seed {seed}: verified record vs commit")
        return {"path": ck / name, "file_sha256": ret["sha256"],
                "model_state_sha256": body["model_state_sha256"]}

    def _e0_context(self, seed: int, ck: dict) -> dict:
        from seqref_mri.scripts import v05lt_eval as EV
        return {"seed": seed, "epoch0_model_state_sha256":
                ck["model_state_sha256"],
                "epoch0_file_sha256": ck["file_sha256"],
                "launch_inputs_sha256": self._bound()["record_sha256"],
                "bank_primary_sha256": C.ZDIAG_BANK_SHA256,
                "code": EV.code_identity(), "backend": EV.backend_identity(),
                "fixture": not self.production}

    def _verify_e0_eval(self, seed: int, ck: dict) -> None:
        from seqref_mri.scripts import v05lt_results as RS
        d = self._e0_dir(seed) / "eval"
        try:
            e0 = RS.load_epoch0_eval(d, seed, ck["model_state_sha256"],
                                     self.manifests(), RS.Identities(), {})
            doc = json.loads((d / "epoch0_eval.json").read_bytes())
        except C.V05LTError as exc:
            code = ("PROD_EPOCH0_INCOMPLETE" if "INCOMPLETE" in exc.code
                    else "PROD_EPOCH0_MISMATCH")
            C.fail(code, f"seed {seed}: {exc.code}: {exc}")
        C.require(not e0["missing"], "PROD_EPOCH0_MISMATCH",
                  f"seed {seed}: population incomplete {e0['missing'][:3]}")
        C.require(doc["context"] == self._e0_context(seed, ck),
                  "PROD_EPOCH0_MISMATCH", f"seed {seed}: context "
                  f"{doc['context']} != expected")
        C.require(doc["dc_tolerances"] == self._bound()["tolerances"],
                  "PROD_EPOCH0_MISMATCH", f"seed {seed}: DC tolerances")

    def epoch0_done(self, seed: int) -> bool:
        """True only when the checkpoint and its evaluation are complete and
        verified; False when a part is absent (stage still to run);
        PROD_EPOCH0_INCOMPLETE / _MISMATCH otherwise."""
        self._bound()
        ck = self.epoch0_checkpoint(seed)
        d = self._e0_dir(seed)
        ev, stg = d / "eval", d / ".eval.staging"
        if ck is None:
            return False
        C.require(not os.path.lexists(stg), "PROD_EPOCH0_INCOMPLETE",
                  f"seed {seed}: evaluation staging left by an interrupted "
                  "write (owner inspection)")
        if not os.path.lexists(ev):
            return False
        self._verify_e0_eval(seed, ck)
        return True

    def epoch0(self, seed: int) -> dict:
        """Epoch-0 checkpoint (if absent) and its evaluation, full
        populations; RNG states restored and checked (refinement 6)."""
        from seqref_mri.scripts import v05lt_epoch0 as E0
        from seqref_mri.scripts import v05lt_eval as EV
        ins = self._bound()
        if self.production:
            check_backend(backend_now())
        rng0 = _rng_save()
        try:
            ck = self.epoch0_checkpoint(seed)
            if ck is None:
                E0.create(self.root / "epoch0", seed, self.repo,
                          hook=self._hook("epoch0_create"),
                          model_factory=self._factory())
                ck = self.epoch0_checkpoint(seed)
            ev = self._e0_dir(seed) / "eval"
            if not os.path.lexists(ev):
                model = E0.load_epoch0(ck["path"], ck["model_state_sha256"],
                                       seed, model_factory=self._factory())
                rec, pan = EV.epoch0_evaluate(
                    model, self.data(), self.banks()["primary"],
                    ins["tolerances"], self.manifests()["cases"])
                del model
                EV.write_epoch0_eval(ev, rec, pan,
                                     self._e0_context(seed, ck),
                                     production=self.production,
                                     hook=self._hook("epoch0_eval"))
        finally:
            rng1 = _rng_save()
            _rng_restore(rng0)
        C.require(_rng_equal(rng0, _rng_save()), "PROD_RNG_NOT_RESTORED",
                  f"seed {seed}: epoch-0 stage")
        self._verify_e0_eval(seed, ck)
        self._check_loaded(ins["pin_code"])
        return {"seed": seed, "rng_changed_inside": not _rng_equal(rng0,
                                                                   rng1)}

    def train_config(self, seed: int, arm: str):
        from seqref_mri.scripts import v05lt_train as T
        self._bound()
        ck = self.epoch0_checkpoint(seed)
        C.require(ck is not None, "PROD_EPOCH0_NOT_READY",
                  f"seed {seed}: no committed epoch-0 checkpoint")
        if self.production:
            return registered_config(seed, arm, self.run_dir(seed, arm),
                                     ck["model_state_sha256"])
        return T.TrainConfig(seed=seed, arm=arm,
                             run_dir=str(self.run_dir(seed, arm)),
                             epoch0_model_state=ck["model_state_sha256"],
                             max_epochs=self.fx.get("max_epochs", 100),
                             fixture=True,
                             train_indices=self.fx.get("train_indices"),
                             val_indices=self.fx.get("val_indices"),
                             note="prodstages fixture")

    def expected_config(self, seed: int, arm: str) -> dict:
        return self.train_config(seed, arm).identity()

    def train(self, seed: int, arm: str, mode: str) -> dict:
        from seqref_mri.scripts import v05lt_train as T
        C.require(mode in ("start", "resume"), "PROD_MODE_INVALID", mode)
        if self.production:
            check_backend(backend_now())
        run = T.Run(self.train_config(seed, arm), self.paths, self.data(),
                    model_factory=self._factory(),
                    hook=self._hook(f"train:{seed}_{arm}"))
        out = run.start() if mode == "start" else run.resume()
        self._check_loaded(self._bound()["pin_code"])
        return out

    # -- post-run (PA5-B, PA5-C; refinements 3, 4) ---------------------------
    def _posteval_root(self) -> Path:
        return self.root / "posteval"

    def _task_file(self, run: dict, fw: dict, task: dict) -> tuple:
        name = f"weights_e{task['epoch']:03d}.pt"
        hit = [v for v in run["retained"].values() if v["name"] == name]
        if hit:
            return Path(run["ckpt_dir"]) / name, hit[0]["sha256"]
        C.require(fw["epoch"] == task["epoch"] and fw["model_state_sha256"]
                  == task["model_state"], "PROD_POSTEVAL_SOURCE",
                  f"epoch {task['epoch']}: no retained weights file and not "
                  "the final weights")
        return Path(fw["path"]), fw["sha256"]

    def _check_task_record(self, doc: dict, task: dict, wfile: Path,
                           wsha: str, seed: int, arm: str) -> None:
        from seqref_mri.scripts import v05lt_eval as EV
        man, ins = self.manifests(), self._bound()
        ck = doc.get("checkpoint", {})
        want_ck = {"file_name": wfile.name, "file_sha256": wsha,
                   "model_state_sha256": task["model_state"], "seed": seed,
                   "arm": arm, "epoch": task["epoch"]}
        bad = {k: (ck.get(k), v) for k, v in want_ck.items()
               if ck.get(k) != v}
        uses = sorted(task["uses"], key=lambda u: (u["seed"], u["arm"],
                                                   u["role"]))
        checks = {
            "checkpoint": not bad,
            "uses": doc.get("uses") == uses,
            "banks": doc.get("banks") == {b: EV.REGISTERED_BANK_ARRAY[b]
                                          for b in task["banks"]},
            "prefixes": doc.get("prefixes") == list(task["prefixes"]) and
            doc.get("prefix_bank") == task.get("prefix_bank", "primary"),
            "population": doc.get("population") == {
                "scope": "midpoint", "n": len(man["mid_ids"]),
                "identities_sha256": C.sha256_bytes(
                    C.canonical_json(man["mid_ids"])),
                "eligibility_manifest_sha256": ins["eligibility"]["sha256"]},
            "masks": doc.get("masks") == {
                "convention": EV.MASK_CONVENTION,
                "sha256": self.expected_masks_sha256()},
            "code": doc.get("code") == EV.code_identity(),
            "backend": doc.get("backend") == EV.backend_identity(),
            "tolerances": doc.get("dc_tolerances") == ins["tolerances"],
            "cases": doc.get("cases") == [c["case"] for c in man["cases"]],
            "context": doc.get("context") == {
                "run": f"{seed}_{arm}", "launch_inputs_sha256":
                ins["record_sha256"], "fixture": not self.production}}
        failed = sorted(k for k, ok in checks.items() if not ok)
        C.require(not failed, "PROD_POSTEVAL_MISMATCH",
                  f"{task['model_state'][:12]}: record vs expected task: "
                  f"{failed} {bad if bad else ''}")

    def _posteval_task(self, task: dict, run: dict, fw: dict, seed: int,
                       arm: str) -> str:
        from seqref_mri.scripts import v05lt_eval as EV
        out = self._posteval_root() / task["model_state"]
        stg = out.with_name(f".{out.name}.staging")
        C.require(not os.path.lexists(stg), "PROD_POSTEVAL_INCOMPLETE",
                  f"{stg}: staging left by an interrupted write (owner "
                  "inspection)")
        wfile, wsha = self._task_file(run, fw, task)
        if not os.path.lexists(out):
            model, ck = EV.load_trained_checkpoint(
                wfile, wsha, task["model_state"], seed=seed, arm=arm,
                epoch=task["epoch"], model_factory=self._factory())
            banks = self.banks()
            rec, pan = EV.posteval_evaluate(
                model, self.data(), task, {b: banks[b] for b in task["banks"]},
                self._bound()["tolerances"], self.manifests()["cases"],
                checkpoint=ck,
                eligibility_sha256=self._bound()["eligibility"]["sha256"])
            del model
            EV.write_posteval(out, rec, pan, {
                "run": f"{seed}_{arm}",
                "launch_inputs_sha256": self._bound()["record_sha256"],
                "fixture": not self.production},
                production=self.production, hook=self._hook("posteval"))
        try:
            doc = EV.load_posteval(out)
        except C.V05LTError as exc:
            code = ("PROD_POSTEVAL_INCOMPLETE" if "INCOMPLETE" in exc.code
                    else "PROD_POSTEVAL_MISMATCH")
            C.fail(code, f"{out.name[:12]}: {exc.code}: {exc}")
        self._check_task_record(doc, task, wfile, wsha, seed, arm)
        return C.sha256_file(out / EV.POSTEVAL_FILE)

    def post_run(self, seed: int, arm: str) -> dict:
        from seqref_mri.scripts import v05lt_eval as EV
        from seqref_mri.scripts import v05lt_final as FW
        from seqref_mri.scripts import v05lt_results as RS
        self._bound()
        rid, rdir = f"{seed}_{arm}", self.run_dir(seed, arm)
        fw = FW.extract_final(rdir, fixture=not self.production,
                              hook=self._hook("final"))
        man = self.manifests()
        try:
            run = RS.load_run(rdir, arm, seed, man, RS.Identities(),
                              fixture=not self.production)
        except C.V05LTError as exc:
            C.fail("PROD_RUN_INVALID", f"{rid}: {exc.code}: {exc}")
        ck0 = self.epoch0_checkpoint(seed)
        cks = run["checkpoints"]
        C.require(ck0 is not None and run["epoch0_model_state"] ==
                  ck0["model_state_sha256"], "PROD_RUN_INVALID",
                  f"{rid}: epoch-0 state != the seed's checkpoint")
        C.require(fw["model_state_sha256"] == cks["final"][
            "model_state_sha256"] and fw["epoch"] == cks["final"]["epoch"],
            "FINAL_WEIGHTS_MISMATCH", f"{rid}: final weights vs last commit")
        roles = {"epoch0": {"epoch": 0, "model_state":
                            run["epoch0_model_state"]}}
        for r in ("epoch3", "selected", "final"):
            roles[r] = {"epoch": cks[r]["epoch"],
                        "model_state": cks[r]["model_state_sha256"]}
        plan = EV.plan_campaign([{"seed": seed, "arm": arm, "roles": roles}],
                                n_slices=len(man["mid_ids"]))
        arts = {"weights_final.pt": fw["sha256"],
                "final_weights.json": fw["record_sha256"]}
        tasks = [t for t in plan["tasks"] if t["epoch"] >= 1]
        for t in tasks:
            arts[f"posteval/{t['model_state']}"] = self._posteval_task(
                t, run, fw, seed, arm)
        ids, shared = RS.Identities(), {}
        for role in RS.POSTEVAL_ROLES:
            try:
                RS.load_posteval_for(self._posteval_root(), run, role, man,
                                     ids, shared)
            except C.V05LTError as exc:
                C.fail("PROD_POSTEVAL_MISMATCH", f"{rid} {role}: "
                       f"{exc.code}: {exc}")
        self._check_loaded(self._bound()["pin_code"])
        return {"verified": True, "artefacts": arts,
                "tasks": [{"model_state": t["model_state"],
                           "epoch": t["epoch"],
                           "roles": sorted(u["role"] for u in t["uses"]),
                           "banks": t["banks"]} for t in tasks]}


# ---------------------------------------------------------------------------
# Launch-inputs record (built after P5; listed in the launch chain)
# ---------------------------------------------------------------------------
def make_launch_inputs(out_path: Path, root: Path, stages: ProductionStages,
                       *, chain_files: dict, inputs: dict,
                       commit: str) -> str:
    """chain_files: {pin, gate2a, P5: relpath}; inputs: {INPUT_KEYS:
    relpath}; commit: the approved implementation pin. Write-once; returns
    the record sha256 (listed in the launch chain as `launch_inputs`)."""
    C.require(set(chain_files) == set(BOUND_ROLES), "PROD_LAUNCH_INPUTS_INVALID",
              f"chain files {sorted(chain_files)}")
    C.require(set(inputs) == set(INPUT_KEYS), "PROD_LAUNCH_INPUTS_INVALID",
              f"inputs {sorted(inputs)}")
    root = Path(root)
    b = {f"{r}_sha256": C.sha256_file(root / _rel_ok(chain_files[r]))
         for r in BOUND_ROLES}
    ins = {k: {"path": _rel_ok(inputs[k]),
               "sha256": C.sha256_file(root / inputs[k])} for k in INPUT_KEYS}
    doc = {"schema": LAUNCH_INPUTS_SCHEMA, "abbr": __abbr__,
           "version": __version__, "bindings": b,
           "pin": {"commit": commit, "code": stages.code_identity()},
           "campaign_config_sha256": stages.config_sha256(), "inputs": ins,
           "fixture": not stages.production}
    return C.write_durable_new(Path(out_path), C.canonical_json(doc))


def make_tolerances(out_path: Path, tolerances: dict,
                    provenance: dict) -> str:
    """P5 tolerance file in the schema read by bind_chain (written by the
    P5 stage; used here by fixtures)."""
    from seqref_mri.scripts import v05lt_eval as EV
    EV.check_tolerances(tolerances)
    return C.write_durable_new(Path(out_path), C.canonical_json({
        "schema": TOLERANCES_SCHEMA, "tolerances": tolerances,
        "provenance": provenance}))
