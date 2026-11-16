# SEQREF-DUNC-GUARD v0.5 -- D-UNC write guard: cumulative disk ceiling and free-space reserve (223 v1.0 + AM1 §11)
# LIFETIME: KEEP
"""Write guard for D-UNC outputs (SEQREF-MRI-DIAG-UNC v1.0 + AM1, §1 and §11).

New disk use = cumulative bytes written under the D-UNC directory, including files that
are later deleted. Before each write the guard checks the 20 GiB ceiling and the 64 GiB
free-space reserve; a projected violation raises DIAG_UNC_RESOURCE_LIMIT before the write
begins. Files are write-once and get a sha256 sidecar.

Accounting
- The cumulative count lives in one ledger file in the guard root. It is the only
  authority: no guard object keeps its own count.
- Every write takes an exclusive lock on a lock file in the root (POSIX flock), re-reads
  the ledger under that lock, checks the limits, charges the bytes, and only then writes.
  Several guard objects or processes on the same directory therefore cannot overwrite
  each other's accounting.
- Bytes are charged BEFORE they are written, so a restart or an interrupted write can
  only over-count, never under-count.
- Each write is charged len(data) + SIDECAR_BYTES + LEDGER_REWRITE_BYTES. The last term
  is a fixed upper bound for the ledger file that the write rewrites (asserted on every
  rewrite), so ledger rewrites are counted, conservatively. Creating the ledger is
  charged LEDGER_BYTES once. The lock file is empty.
- Writes go to a '<name>.dunc-part' file and are renamed into place; a failed write
  removes its own partial files. At start, under the lock (so no write is in progress),
  leftover '*.dunc-part' files are removed and reported. Files with any other suffix,
  including a plain '.part', are never touched.
- A ledger that cannot be read is refused; the count is never reset.
- A ledger is created only in a genuinely new directory: no entry other than the guard's
  own lock file, and a lock file that has never been marked as initialised. When the
  ledger is created the lock file is marked. A directory with a missing ledger and either
  existing entries or a marked lock file is refused (DIAG_UNC_INPUT_INVALID); nothing in
  it is deleted, changed or added: the directory tree is identical before and after the
  refusal (a directory with no lock file yet is checked before one is created). (If someone removes both the ledger and the lock file from an
  otherwise empty directory, it cannot be told apart from a new one.)
- The guard counts writes made through it in its root. The root must therefore be a new,
  empty directory for the run; bytes written elsewhere are not counted by this guard.

POSIX only (fcntl). Changelog
- v0.5 (2026-11-16): a refused start no longer leaves an empty lock file behind in a
  directory that had neither ledger nor lock file; the refusal adds nothing.
- v0.4 (2026-11-16): a missing ledger no longer starts a new count unless the directory is
  genuinely new (empty apart from an unmarked lock file); otherwise the guard refuses and
  leaves every file in place, including leftover partial files.
- v0.3 (2026-11-16): exclusive lock around every ledger read-modify-write and the write
  itself; count re-read from the ledger before every write (fixes under-counting with two
  active guard objects); ledger rewrites charged with a fixed bound; partial files use a
  guard-specific suffix and unrelated '.part' files are left alone.
- v0.2 (2026-11-16): persistent ledger across restarts; atomic write with clean-up.
- v0.1 (2026-11-16): first version (in-memory count only).
"""
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import shutil

from diag_unc_core import DiagUncError

__version__ = "0.5"
LOG = logging.getLogger("diag_unc")
GIB = 2 ** 30
CEILING_BYTES = 20 * GIB
RESERVE_BYTES = 64 * GIB
SIDECAR_BYTES = 64 + 1              # hex digest and newline
LEDGER_NAME = ".dunc_write_ledger.json"
LOCK_NAME = ".dunc_write_ledger.lock"
LEDGER_BYTES = 256                  # charged once when the ledger is created (ledger + lock-file mark)
LOCK_MARK = "initialised\n"
LEDGER_REWRITE_BYTES = 128          # charged per write: upper bound of one ledger rewrite
PART_SUFFIX = ".dunc-part"
LEDGER_SCHEMA = "seqref-dunc-write-ledger/1"


def _fsync(fh):
    fh.flush()
    os.fsync(fh.fileno())


class WriteGuard:
    def __init__(self, root, ceiling=CEILING_BYTES, reserve=RESERVE_BYTES, free_fn=None):
        self.root = os.path.realpath(root)
        if not os.path.isdir(self.root):
            raise DiagUncError("DIAG_UNC_INPUT_INVALID", "guard root is not an existing directory: %s" % root)
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
                raise DiagUncError("DIAG_UNC_INPUT_INVALID", "write ledger is missing from a directory that is not "
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
            raise DiagUncError("DIAG_UNC_INPUT_INVALID", "write ledger is missing from a directory that is not new "
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
            raise DiagUncError("DIAG_UNC_INPUT_INVALID", "write ledger is unreadable; refusing to reset the "
                               "cumulative count: %s" % exc)
        return written

    def _store_ledger(self, written):
        payload = json.dumps({"schema": LEDGER_SCHEMA, "written": int(written)})
        if len(payload) > LEDGER_REWRITE_BYTES:
            raise DiagUncError("DIAG_UNC_INPUT_INVALID", "ledger record exceeds its accounted size")
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
        """Called under the lock only: no write is in progress, so every '*.dunc-part' is a leftover."""
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
            raise DiagUncError("DIAG_UNC_RESOURCE_LIMIT", "write of %d bytes would bring new disk use to %d, above "
                               "the ceiling %d" % (nbytes, written + nbytes, self.ceiling))
        free = int(self._free())
        if free - nbytes < self.reserve:
            raise DiagUncError("DIAG_UNC_RESOURCE_LIMIT", "write of %d bytes would leave %d bytes free, below the "
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
            raise DiagUncError("DIAG_UNC_INPUT_INVALID", "path is outside the guard root or reserved: %s" % relpath)
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
                raise DiagUncError("DIAG_UNC_INPUT_INVALID", "refusing to overwrite: %s" % relpath)
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
            raise DiagUncError("DIAG_UNC_REPRODUCTION_MISMATCH", "file does not match its sha256 sidecar: %s" % relpath)
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
