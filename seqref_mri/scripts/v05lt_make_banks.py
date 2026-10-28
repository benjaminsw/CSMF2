# SEQREF-MRI-V05LTBANK v0.1 -- V05LT R1 sensitivity-bank generator (194 §8.1, O9b)
# LIFETIME: KEEP
"""Generate the two V05LT R1 sensitivity latent banks (PCG64(20261023),
PCG64(20261024)) with exactly the registered Z_DIAG recipe, after first
proving the recipe by regenerating Z_DIAG (PCG64(0)) to its registered
sha256 (147 §4).

Recipe: numpy.random.Generator(numpy.random.PCG64(seed))
        .standard_normal((128, 13824))  [float64]  -> .astype(float32);
        bank sha256 = sha256 of the C-order float32 raw bytes.

Writes, per bank, <out>/v05lt_bank_<seed>.npy and a manifest JSON.
Refuses to overwrite existing files. Exit 0 = all checks passed;
exit 2 = any check failed (error logged; no partial result is valid).
No fallback paths: any mismatch stops the run.
"""
import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
from typing import Optional

import numpy as np

__version__ = "SEQREF-MRI-V05LTBANK v0.1"
BANK_SHAPE = (128, 13824)
Z_DIAG_SEED = 0
Z_DIAG_SHA256 = "f8a6a34b6b474286359e4c99341de256b62bc0dff1e6bc8cf52e72f0461dc8d6"
NEW_SEEDS = (20261023, 20261024)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [V05LTBANK] %(message)s")
log = logging.getLogger("v05ltbank")


class BankError(RuntimeError):
    pass


def make_bank(seed: int) -> np.ndarray:
    g = np.random.Generator(np.random.PCG64(seed))
    z = g.standard_normal(BANK_SHAPE)            # float64 draw
    return np.ascontiguousarray(z.astype(np.float32))


def raw_sha(z: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_recipe() -> None:
    z = make_bank(Z_DIAG_SEED)
    got = raw_sha(z)
    if got != Z_DIAG_SHA256:
        raise BankError(f"RECIPE_MISMATCH: regenerated Z_DIAG sha256 {got} "
                        f"!= registered {Z_DIAG_SHA256}")
    log.info("recipe verified: Z_DIAG regenerated to registered sha256 %s", got)


def build_one(seed: int, out_dir: str, expect: Optional[str]) -> dict:
    npy_path = os.path.join(out_dir, f"v05lt_bank_{seed}.npy")
    man_path = os.path.join(out_dir, f"v05lt_bank_{seed}_manifest.json")
    for p in (npy_path, man_path):
        if os.path.exists(p):
            raise BankError(f"ARTIFACT_EXISTS: {p} (refusing to overwrite)")

    z1 = make_bank(seed)
    z2 = make_bank(seed)
    h1, h2 = raw_sha(z1), raw_sha(z2)
    if h1 != h2:
        raise BankError(f"NONDETERMINISTIC: seed {seed}: {h1} != {h2}")
    if z1.shape != BANK_SHAPE or z1.dtype != np.float32:
        raise BankError(f"LAYOUT: seed {seed}: {z1.shape} {z1.dtype}")
    if not np.isfinite(z1).all():
        raise BankError(f"NON_FINITE: seed {seed}")
    if expect is not None and h1 != expect:
        raise BankError(f"EXPECT_MISMATCH: seed {seed}: {h1} != expected {expect}")

    np.save(npy_path, z1)
    z_back = np.load(npy_path)
    if raw_sha(z_back) != h1:
        raise BankError(f"READBACK_MISMATCH: {npy_path}")

    manifest = {
        "schema": "seqref-v05lt-bank-manifest/1",
        "generator": __version__,
        "purpose": "V05LT R1 sensitivity bank (194 §8.1, §3.1)",
        "seed": seed,
        "recipe": "numpy.random.Generator(numpy.random.PCG64(seed))"
                  ".standard_normal((128, 13824)) float64 -> float32",
        "shape": list(BANK_SHAPE),
        "dtype": "float32",
        "bank_sha256_raw_bytes": h1,
        "npy_file_sha256": file_sha(npy_path),
        "recipe_check": {"z_diag_seed": Z_DIAG_SEED,
                         "z_diag_sha256": Z_DIAG_SHA256, "passed": True},
        "summary": {"mean": float(z1.mean()), "std": float(z1.std())},
        "numpy_version": np.__version__,
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
                       .isoformat(timespec="seconds"),
    }
    with open(man_path, "x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    manifest["manifest_file_sha256"] = file_sha(man_path)
    log.info("seed %d: bank sha256 %s; npy %s; manifest %s",
             seed, h1, manifest["npy_file_sha256"],
             manifest["manifest_file_sha256"])
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=os.path.expanduser("~/v05lt_banks"))
    ap.add_argument("--expect-20261023", default=None,
                    help="optional independent sha256 to cross-check")
    ap.add_argument("--expect-20261024", default=None,
                    help="optional independent sha256 to cross-check")
    a = ap.parse_args()
    expects = {20261023: a.expect_20261023, 20261024: a.expect_20261024}
    try:
        os.makedirs(a.out_dir, exist_ok=True)
        log.info("%s; numpy %s; out %s", __version__, np.__version__, a.out_dir)
        verify_recipe()
        results = [build_one(s, a.out_dir, expects[s]) for s in NEW_SEEDS]
    except BankError as exc:
        log.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 -- log every unexpected failure
        log.exception("UNEXPECTED: %s", exc)
        return 2
    print("\nRESULT (paste back):")
    for m in results:
        print(f"  seed {m['seed']}: bank_sha256={m['bank_sha256_raw_bytes']} "
              f"npy_sha256={m['npy_file_sha256']} "
              f"manifest_sha256={m['manifest_file_sha256']}")
    print("  recipe check: Z_DIAG reproduced (PASS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
