# SEQREF-DLAT-GUARD v0.1 -- diag_lat_guard.py: D-LAT write guard (cumulative disk ceiling, free-space reserve) and memory guard
# LIFETIME: KEEP
"""
STATUS: PRE-FREEZE ENGINEERING; NOT ACTIVATED (236 LE19; 238 LE21). Provisional until freeze.

WRITE GUARD -- PORTED from diag_unc_guard.py v0.5 (sha256 b3dc9f69...5bc5), logic unchanged;
equivalence checked by fixture T19. Differences from the original, all deliberate:
- own file names (.dlat_write_ledger.json / .lock, '.dlat-part') and ledger schema;
- D-LAT codes: DIAG_LAT_GUARD_STATE (was DIAG_UNC_INPUT_INVALID), DIAG_LAT_RESOURCE_LIMIT,
  DIAG_LAT_SIDECAR_MISMATCH (was DIAG_UNC_REPRODUCTION_MISMATCH);
- ceiling and reserve are REQUIRED parameters (no default: the D-LAT limits are set by the
  frozen protocol and F1, not by this file). No limit-projection formula is included (C1 open).
Behaviour kept: cumulative count in one ledger (deleted files stay counted); exclusive flock
around every read-modify-write; bytes charged BEFORE they are written; write-once files with
a sha256 sidecar; partial files renamed into place and cleaned up; an unreadable ledger or a
missing ledger in a directory that is not new is refused and nothing is changed.

MEMORY GUARD -- new in D-LAT (decision E6, 236 LE16): peak resident memory against a limit;
an out-of-memory error becomes the typed stop DIAG_LAT_RESOURCE_LIMIT, with no retry.

POSIX only (fcntl, resource). No fallback, placeholder, mock or dummy return.

Changelog
- v0.1 (2026-11-21): port of the D-UNC write guard v0.5 with D-LAT names, codes and required
  limits; memory guard (peak_rss_bytes, check_memory, memory_guarded) added.
"""
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import resource
import shutil
import sys

from diag_lat_core import DiagLatError

__version__ = "0.1"
ABBR = "SEQREF-DLAT-GUARD"
LOG = logging.getLogger("diag_lat.guard")
GIB = 2 ** 30
GUARD_STATE = "DIAG_LAT_GUARD_STATE"
RESOURCE_LIMIT = "DIAG_LAT_RESOURCE_LIMIT"
SIDECAR_MISMATCH = "DIAG_LAT_SIDECAR_MISMATCH"
SIDECAR_BYTES = 64 + 1              # hex digest and newline
LEDGER_NAME = ".dlat_write_ledger.json"
LOCK_NAME = ".dlat_write_ledger.lock"
LEDGER_BYTES = 256                  # charged once when the ledger is created (ledger + lock-file mark)
LOCK_MARK = "initialised\n"
LEDGER_REWRITE_BYTES = 128          # charged per write: upper bound of one ledger rewrite
PART_SUFFIX = ".dlat-part"
LEDGER_SCHEMA = "seqref-dlat-write-ledger/1"


def _fsync(fh):
    fh.flush()
    os.fsync(fh.fileno())


class WriteGuard:
    def __init__(self, root, ceiling, reserve, free_fn=None):
        self.root = os.path.realpath(root)
        if not os.path.isdir(self.root):
            raise DiagLatError(GUARD_STATE, "guard root is not an existing directory: %s" % root)
        self.ceiling = int(ceiling)
        self.reserve = int(reserve)
        self._free = free_fn if free_fn is not None else (lambda: shutil.disk_usage(self.root).free)
        self._ledger = os.path.join(self.root, LEDGER_NAME)
        self._lock_path = os.path.join(self.root, LOCK_NAME)
        if not os.path.exists(self._ledger) and not os.path.exists(self._lock_path):
            try:
                existing = sorted(os.listdir(self.root))
            except OSError as exc:
                LOG.error("cannot inspect the guard root: %s", exc)
                raise
            if existing:        # refuse before the lock file is created: the directory is left exactly as found
                raise DiagLatError(GUARD_STATE, "write ledger is missing from a directory that is not "
                                   "new (%d existing entries, no lock file); refusing to start a new cumulative "
                                   "count" % len(existing))
        with self._locked():
            if os.path.exists(self._ledger):
                self._read_ledger()
                self.removed_partial = self._remove_stale_parts()
            else:
                self._create_ledger_if_new()
                self.removed_partial = []

    # ------------------------------------------------------------ lock and ledger
    @contextlib.contextmanager
    def _locked(self):
        """Exclusive lock for this directory, across guard objects and processes."""
        try:
            fh = open(self._lock_path, "a+")
        except OSError as exc:
            LOG.error("cannot open the guard lock file: %s", exc)
            raise
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()

    def _create_ledger_if_new(self):
        """Under the lock, with no ledger present: start a count only in a genuinely new directory."""
        try:
            others = sorted(n for n in os.listdir(self.root) if n != LOCK_NAME)
            with open(self._lock_path, "r") as fh:
                marked = fh.read() != ""
        except OSError as exc:
            LOG.error("cannot inspect the guard root: %s", exc)
            raise
        if others or marked:
            raise DiagLatError(GUARD_STATE, "write ledger is missing from a directory that is not new "
                               "(%d other entries; lock file %s); refusing to start a new cumulative count"
                               % (len(others), "marked as initialised" if marked else "unmarked"))
        self._check(LEDGER_BYTES, 0)
        self._store_ledger(LEDGER_BYTES)
        try:
            with open(self._lock_path, "w") as fh:
                fh.write(LOCK_MARK)
                _fsync(fh)
        except OSError as exc:
            LOG.error("cannot mark the guard lock file: %s", exc)
            raise

    def _read_ledger(self):
        try:
            with open(self._ledger, "r") as fh:
                data = json.load(fh)
            written = data["written"]
            if data.get("schema") != LEDGER_SCHEMA or not isinstance(written, int) or isinstance(written, bool) \
                    or written < LEDGER_BYTES:
                raise ValueError("invalid ledger content %r" % (data,))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise DiagLatError(GUARD_STATE, "write ledger is unreadable; refusing to reset the "
                               "cumulative count: %s" % exc)
        return written

    def _store_ledger(self, written):
        payload = json.dumps({"schema": LEDGER_SCHEMA, "written": int(written)})
        if len(payload) > LEDGER_REWRITE_BYTES:
            raise DiagLatError(GUARD_STATE, "ledger record exceeds its accounted size")
        tmp = self._ledger + PART_SUFFIX
        try:
            with open(tmp, "w") as fh:
                fh.write(payload)
                _fsync(fh)
            os.replace(tmp, self._ledger)
        except OSError as exc:
            LOG.error("ledger update failed: %s", exc)
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def _remove_stale_parts(self):
        """Called under the lock only: no write is in progress, so every '*.dlat-part' is a leftover."""
        removed = []
        for dirpath, _, names in os.walk(self.root):
            for name in names:
                if name.endswith(PART_SUFFIX):
                    path = os.path.join(dirpath, name)
                    LOG.warning("removing partial file left by an interrupted write: %s", path)
                    os.remove(path)
                    removed.append(os.path.relpath(path, self.root))
        return sorted(removed)

    @property
    def written(self):
        """Cumulative bytes charged for this directory, read from the ledger."""
        return self._read_ledger()

    # ------------------------------------------------------------ checks
    def _check(self, nbytes, written):
        if written + nbytes > self.ceiling:
            raise DiagLatError(RESOURCE_LIMIT, "write of %d bytes would bring new disk use to %d, above "
                               "the ceiling %d" % (nbytes, written + nbytes, self.ceiling))
        free = int(self._free())
        if free - nbytes < self.reserve:
            raise DiagLatError(RESOURCE_LIMIT, "write of %d bytes would leave %d bytes free, below the "
                               "reserve %d" % (nbytes, free - nbytes, self.reserve))

    @staticmethod
    def charge(data_bytes):
        """Bytes charged for a write of data_bytes of content."""
        return int(data_bytes) + SIDECAR_BYTES + LEDGER_REWRITE_BYTES

    def check(self, data_bytes):
        """Advisory: raise if a write of data_bytes of content would break a limit now.

        The binding check is made again under the lock inside write().
        """
        self._check(self.charge(data_bytes), self._read_ledger())

    def _path(self, relpath):
        path = os.path.realpath(os.path.join(self.root, relpath))
        reserved = (self._ledger, self._lock_path)
        if os.path.commonpath([path, self.root]) != self.root or path in reserved or path.endswith(PART_SUFFIX):
            raise DiagLatError(GUARD_STATE, "path is outside the guard root or reserved: %s" % relpath)
        return path

    # ------------------------------------------------------------ write / read / delete
    def write(self, relpath, data):
        """Write-once file with sha256 sidecar; returns the hex digest."""
        path = self._path(relpath)
        nbytes = self.charge(len(data))
        digest = hashlib.sha256(data).hexdigest()
        parts = [path + PART_SUFFIX, path + ".sha256" + PART_SUFFIX]
        with self._locked():
            if os.path.exists(path) or os.path.exists(path + ".sha256"):
                raise DiagLatError(GUARD_STATE, "refusing to overwrite: %s" % relpath)
            written = self._read_ledger()               # authoritative count, re-read under the lock
            self._check(nbytes, written)
            self._store_ledger(written + nbytes)        # charged before any byte is written
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(parts[0], "xb") as fh:
                    fh.write(data)
                    _fsync(fh)
                with open(parts[1], "x") as fh:
                    fh.write(digest + "\n")
                    _fsync(fh)
                os.replace(parts[1], path + ".sha256")
                os.replace(parts[0], path)
            except OSError as exc:
                LOG.error("write failed for %s: %s; removing its partial files (bytes stay counted)", relpath, exc)
                for leftover in parts + [path + ".sha256"]:
                    if os.path.exists(leftover) and not os.path.exists(path):
                        os.remove(leftover)
                raise
        return digest

    def read(self, relpath):
        """Read a guarded file and verify it against its sidecar."""
        path = self._path(relpath)
        try:
            with open(path, "rb") as fh:
                data = fh.read()
            with open(path + ".sha256", "r") as fh:
                expected = fh.read().strip()
        except OSError as exc:
            LOG.error("read failed for %s: %s", relpath, exc)
            raise
        if hashlib.sha256(data).hexdigest() != expected:
            raise DiagLatError(SIDECAR_MISMATCH, "file does not match its sha256 sidecar: %s" % relpath)
        return data

    def delete(self, relpath):
        """Delete an intermediate file; its bytes stay counted in the cumulative total."""
        path = self._path(relpath)
        with self._locked():
            try:
                os.remove(path)
                if os.path.exists(path + ".sha256"):
                    os.remove(path + ".sha256")
            except OSError as exc:
                LOG.error("delete failed for %s: %s", relpath, exc)
                raise


# ---------------------------------------------------------------- memory guard (D-LAT, E6)
def peak_rss_bytes():
    """Peak resident set size of this process in bytes (ru_maxrss: kB on Linux, bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def check_memory(limit_bytes, peak_fn=peak_rss_bytes):
    """Raise DIAG_LAT_RESOURCE_LIMIT if the peak resident memory exceeds limit_bytes; return the peak."""
    peak = int(peak_fn())
    if peak > int(limit_bytes):
        raise DiagLatError(RESOURCE_LIMIT, "peak resident memory %d bytes exceeds the limit %d" % (peak, int(limit_bytes)))
    return peak


@contextlib.contextmanager
def memory_guarded(limit_bytes, peak_fn=peak_rss_bytes):
    """Run a block once. An out-of-memory error becomes DIAG_LAT_RESOURCE_LIMIT (no retry, no
    other device or batch size); the peak is checked against the limit when the block ends."""
    try:
        yield
    except MemoryError as exc:
        raise DiagLatError(RESOURCE_LIMIT, "out of memory: %s" % (exc,)) from exc
    check_memory(limit_bytes, peak_fn)
