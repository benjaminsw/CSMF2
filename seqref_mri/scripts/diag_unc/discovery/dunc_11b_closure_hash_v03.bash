#!/usr/bin/env bash
# SEQREF-DUNC-CLH v0.3 -- dunc_11b_closure_hash: read-only static import closure and pin hashes for fixture 11b
# LIFETIME: DIAGNOSTIC
# Reads tracked source at the pin through Git only. No project module import,
# no HDF5, checkpoint or MRI access, no checkout. Writes nothing itself; the
# caller captures the output with tee.
# Changelog (v0.3): propagate Python and shell failure as the actual script exit status; test hooks are labelled non-production.
# Changelog (v0.2): each Git query has its own failure check; ambiguous and
# unresolved bare imports are reported, not dropped; Python runs isolated (-I);
# Git errors inside Python end with a typed stop line; exit code printed.
(
set -o pipefail
REPO=/home/benjamin/CSMFII
[ -n "${CLH_REPO:-}" ] && REPO=$CLH_REPO          # test hook (mock repository)
PIN=7455da8027f33a6cca4dbf72f04e72420e938f45
[ -n "${CLH_PIN:-}" ] && PIN=$CLH_PIN             # test hook
PY=/home/benjamin/CSMFII/seqref_mri/.venv_mri/bin/python
[ -n "${CLH_PY:-}" ] && PY=$CLH_PY                # test hook
cd "$REPO" || { echo "CLOSURE_HASH_STOP cd"; exit 2; }
B=$(git rev-parse --abbrev-ref HEAD) || { echo "CLOSURE_HASH_STOP git branch"; exit 2; }
[ "$B" = "seqref-v05lt" ] || { echo "CLOSURE_HASH_STOP wrong branch"; exit 2; }
H=$(git rev-parse HEAD) || { echo "CLOSURE_HASH_STOP git head"; exit 2; }
[ "$H" = "$PIN" ] || { echo "CLOSURE_HASH_STOP wrong HEAD"; exit 2; }
S=$(git status --porcelain) || { echo "CLOSURE_HASH_STOP git status"; exit 2; }
[ -z "$S" ] || { echo "CLOSURE_HASH_STOP dirty tree"; exit 2; }
[ -x "$PY" ] || { echo "CLOSURE_HASH_STOP interpreter missing"; exit 2; }
if [ -n "${CLH_REPO:-}${CLH_PIN:-}${CLH_PY:-}${CLH_KNOWN_OFF:-}" ]; then echo "SELF-TEST hooks active (not a production evidence result)"; fi
PIN="$PIN" "$PY" -I -B - <<'PYEOF'
import ast, hashlib, os, subprocess, sys
pin = os.environ['PIN']
def stop(msg):
    print('ERROR', msg); print('CLOSURE_HASH_STOP'); sys.exit(2)
def git(*a):
    r = subprocess.run(['git', *a], capture_output=True)
    if r.returncode != 0:
        stop('git %s failed: %s' % (' '.join(a[:2]), r.stderr.decode(errors='replace').strip()[:200]))
    return r.stdout
paths = [s for s in git('ls-tree', '-r', '--name-only', pin, 'seqref_mri').decode().splitlines() if s.endswith('.py')]
by_mod = {p[:-3].replace('/', '.'): p for p in paths}
for p in paths:
    if p.endswith('/__init__.py'):
        by_mod[p[:-12].replace('/', '.')] = p
roots = [
 'seqref_mri.scripts.v05lt_eval', 'seqref_mri.scripts.v05lt_common',
 'seqref_mri.scripts.v05lt_metrics', 'seqref_mri.scripts.tiny_gate',
 'seqref_mri.scripts.train_base', 'seqref_mri.tdiag.estimators',
 'seqref_mri.tdiag.d3', 'seqref_mri.src.free_flow_runtime',
 'seqref_mri.src.fastmri_data', 'seqref_mri.src.residual_decoder',
 'seqref_mri.src.forward_operator', 'seqref_mri.scripts.v02_manifests',
 'seqref_mri.src.preflight_io', 'seqref_mri.src.preflight_parents']
known = {
 'seqref_mri.tdiag.d3': '73110e7d36a1ebdcb51926c4bbde81f88392f578e1aa860736c509706041245e',
 'seqref_mri.src.free_flow_runtime': '8e171b2c521d77c8ad22b21d02537be160d15166aecb9e0319fd0a25b69823c9',
 'seqref_mri.scripts.v05lt_eval': '10b4bc65d465e9c6d0a016c2d27977cd9f485ffb851936d9a15ad6a325898554',
 'seqref_mri.scripts.tiny_gate': '3ddd73ac99612e4af8d69664539acaef2ec4e9621deae530327fe0a8cd1e88d8',
 'seqref_mri.tdiag.estimators': '05561e729c7b5a7e3f6914990af2808918e372bec1ed0710c97ef3de867dfc28',
 'seqref_mri.src.fastmri_data': '610cc1d1d7968deebc88f645270e1baefb6589cb56841bdd327450ca1069cb44',
 'seqref_mri.src.residual_decoder': 'fd65e3a229e3df4f0226e276c92c916f5bd32cc77ed3655d68fb8a14608990d6',
 'seqref_mri.src.preflight_io': 'ab01fefbf0c46ce08b8746e97250f420de3f219bc8a25a49e514bc4af3b19942',
 'seqref_mri.scripts.v02_manifests': '5ef4c55121df0e5fa151c920bcafb75c55121c222afb7f0d73b19cdbc44749e9',
 'seqref_mri.scripts.v05lt_common': 'f35664fcb92607d0b0919ea80e55a90c99bbea9d6584b329c0d7688d9e3712a1',
 'seqref_mri.scripts.train_base': '2ae948663dbd53df32e032c0ef8c54e23666106529ca1906a354fcee7ba6ffc5',
 'seqref_mri.src.forward_operator': '3cbc5162da9b727e82457a7c8f93121097c8228701f884a2190c681e4e1d4b5d',
 'seqref_mri.scripts.v05lt_metrics': '87eaa5a4e85b6b70072c1c76fcd3e73ce104c3cdb04ebef973c346ff90b18a74'}
if os.environ.get('CLH_KNOWN_OFF') == '1':   # test hook (mock repository has other content)
    known = {}
cache = {}; alias = {}; ambiguous = {}; external = set(); parse_errors = []; dynamic = []
def read(m):
    p = by_mod[m]
    if p not in cache:
        cache[p] = git('show', '%s:%s' % (pin, p))
    return cache[p]
def candidates(node, m):
    if isinstance(node, ast.Import):
        return [(x.name, True) for x in node.names]
    module = node.module or ''
    if node.level:
        pkg = m if by_mod[m].endswith('/__init__.py') else m.rsplit('.', 1)[0]
        parts = pkg.split('.')
        base = '.'.join(parts[:len(parts) - node.level + 1])
        module = (base + '.' + module).strip('.')
    return [(module, True)] + [((module + '.' + x.name).strip('.'), False) for x in node.names if x.name != '*']
def resolve(name, is_module, m):
    if name in by_mod:
        return name
    if name.startswith('seqref_mri.') or not is_module:
        return None
    top = name.split('.')[0]
    hits = sorted(x for x in by_mod if x.endswith('.' + name))
    if len(hits) == 1:
        alias[name] = hits[0]; return hits[0]
    if len(hits) > 1:
        ambiguous.setdefault(name, set()).update(hits); return None
    external.add(top); return None
queue = list(roots); seen = set(); missed = set()
while queue:
    m = queue.pop(0)
    if m in seen:
        continue
    if m not in by_mod:
        missed.add(m); continue
    seen.add(m)
    try:
        tree = ast.parse(read(m).decode('utf-8'), filename=by_mod[m])
    except Exception as e:
        parse_errors.append((m, str(e))); continue
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for name, is_module in candidates(n, m):
                r = resolve(name, is_module, m)
                if r and r not in seen:
                    queue.append(r)
        elif isinstance(n, ast.Call):
            f = n.func
            fn = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ''
            if fn in ('__import__', 'import_module'):
                dynamic.append('%s line %d' % (by_mod[m], n.lineno))
# packages on the path of every reached module are imported too
for m in sorted(seen):
    parts = m.split('.')
    for i in range(1, len(parts)):
        pk = '.'.join(parts[:i])
        if pk in by_mod:
            seen.add(pk)
print('PIN', pin)
print('TRACKED_PYTHON_SOURCE_COUNT', len(paths))
print('STATIC_REACHABLE_MODULES', len(seen))
print('STATIC_REACHABLE_SOURCE_FILES', len({by_mod[m] for m in seen}))
print('=== MODULES: sha256 of the file content at the pin (sorted) ===')
fail = False
for m in sorted(seen):
    h = hashlib.sha256(read(m)).hexdigest(); k = known.get(m)
    status = 'KNOWN_PASS' if k == h else 'KNOWN_MISMATCH' if k else 'PIN_ONLY'
    fail = fail or (k is not None and k != h)
    print(status, h, len(read(m)), by_mod[m], 'module=' + m)
print('=== BARE IMPORTS RESOLVED TO ONE PROJECT FILE (second module name for the same file) ===')
for a, b in sorted(alias.items()):
    print(a, '=>', b)
print('=== BARE IMPORTS MATCHING SEVERAL PROJECT FILES (NOT FOLLOWED; closure incomplete) ===')
for a, b in sorted(ambiguous.items()):
    print(a, '=>', ' | '.join(sorted(b)))
print('=== DYNAMIC IMPORT CALLS (not followed; review by hand) ===')
for d in dynamic:
    print(d)
print('=== TOP-LEVEL NAMES NOT IN THE PROJECT (standard library or third party) ===')
print(' '.join(sorted(external)))
print('=== ROOT MODULES NOT FOUND AT THE PIN ===')
for m in sorted(missed):
    print(m)
print('=== PARSE ERRORS ===')
for m, e in parse_errors:
    print(m, e)
print('=== LIMITS ===')
print('Static scan of import statements only (function-level imports included). It is a superset guess, not the runtime closure; the fixture must still check sys.modules and __file__.')
print('No project module imported; no HDF5, checkpoint or MRI file opened; nothing written by this script.')
if fail or parse_errors or missed or ambiguous:
    print('CLOSURE_HASH_STOP'); sys.exit(2)
print('CLOSURE_HASH_LISTED')
PYEOF
)
RC=$?
echo "CLOSURE_HASH_EXIT=$RC"
exit "$RC"
