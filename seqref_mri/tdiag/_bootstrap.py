# SEQREF-TDIAG v0.1 -- tdiag._bootstrap
# LIFETIME: KEEP
# =============================================================================
# Purpose: explicit import-path bootstrap for the TDIAG package. The
#          campaign preflight modules live OUTSIDE this package, and the
#          reused TINY stack imports them under the LEGACY top-level
#          names (preflight_io / preflight_parents). Every TDIAG package
#          module imports THIS module first, so the legacy names resolve
#          and -- critically -- resolve to the SAME module objects TINY
#          uses: one preflight_parents, one StageError class, no identity
#          split (SEQREF-TDIAGT T10 guards this).
#          No reliance on __init__.py side effects; no PYTHONPATH.
# =============================================================================
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC = os.path.join(_REPO, "seqref_mri", "src")

if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
