# SEQREF-V04TP-A v0.1
# LIFETIME: KEEP. Supplemental gate-(a) artifact recheck only.
# No MRI loading, preprocessing, model construction, training or gate b-f.
# Frozen sources and original evidence are read-only. No fallback/retry.
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

BASE = Path('/home/benjamin/CSMFII')
PIN = '70f35623ee4ca08b942b9204775415fe972b82df'
PARENT_PIN = '3393fbee8f3c2d3e430c81fcd48b909c67d366f8'
COMMON_SHA = '25c457226dae52d3b7ab167f83f06cc81bcb2907c2a44c62eb3b40c6255fb798'
GATE_SHA = 'd5f74593d5b02552ec5eb906676597ccf37eff32ea95d70f25486ef6261a3ed9'
PARENT_SHA = '8b08333119145be5118329016cffeb4dab32bf642583d675004f22e4c231d544'
PARITY_SHA = 'ff1801158c72b6dce8914fb2d6c75b051af49d06f8778449e30c813f5c588ecf'
MANIFEST_SHAS = {
    'v02_epoch0_manifest.json': 'b0637f787ce198e4424f6b506fed264300ac585921ffaf17995004bb317efb56',
    'v02_epoch1_manifest.json': '5fb416f995ced86ff26afcc3769691efde63c601c0b7768709b044cd534621e6',
    'v02_epoch2_manifest.json': '209e04545743dc1d9da15ea6eaa3574d077aa766d6f07c7e7937ce61da8e9d57',
    'v02_holdout_manifest.json': '0f2f96073fad6bdaad1314ae1319b643a1b67c06424e757f1fea3cfc1e002019',
    'v02_d3_monitor_manifest.json': 'cb08a044b8c404406d7d742bf2d1c9e827b7b757a5cf494bbb56968e732775d5',
}


class RecheckError(RuntimeError):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(f'{code}: {detail}')


def require(condition, code, detail):
    if not condition:
        raise RecheckError(code, detail)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verified(path, expected):
    path = Path(path)
    require(path.is_file(), 'A_RECHECK_INPUT_MISSING', str(path))
    side = Path(str(path) + '.sha256')
    require(side.is_file(), 'A_RECHECK_SIDECAR_MISSING', str(side))
    tokens = side.read_text().split()
    require(bool(tokens), 'A_RECHECK_SIDECAR_INVALID', str(side))
    require(digest(path) == tokens[0] == expected,
            'A_RECHECK_INPUT_DRIFT', str(path))
    return json.loads(path.read_text())


def write_json(path, obj):
    raw = json.dumps(obj, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=False, allow_nan=False).encode('utf-8')
    Path(path).write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    Path(str(path) + '.sha256').write_text(f'{sha}  {Path(path).name}\n')
    return sha


def run(root, out, record):
    common_path = BASE / 'seqref_mri/scripts/v04tp_common.py'
    require(digest(common_path) == COMMON_SHA,
            'A_RECHECK_TOOL_DRIFT', str(common_path))
    sys.path.insert(0, str(common_path.parent))
    import v04tp_common as tc
    require(Path(tc.__file__).resolve() == common_path.resolve(),
            'A_RECHECK_IMPORT_DRIFT', 'unexpected common helper')
    tc.verify_pin_and_clean_tree(str(root), PIN)
    gate_path = root / 'seqref_mri/scripts/v04_gate.py'
    require(digest(gate_path) == GATE_SHA,
            'A_RECHECK_GATE_DRIFT', str(gate_path))
    tc.activate_import_root(str(root))
    from seqref_mri.scripts.v04_gate import gate_a_pipeline_bitexact
    modules = tc.collect_modules(str(root))

    evidence = BASE / 'V04TP_70f35623'
    parent_path = evidence / 'parent/v04tp_parent_manifest.json'
    parity_path = evidence / 'compare/v04tp_parity_record.json'
    parent = verified(parent_path, PARENT_SHA)
    parity = verified(parity_path, PARITY_SHA)
    require(parent.get('schema') == 'seqref-v04-tensor-parity-parent/1'
            and parity.get('schema') == 'seqref-v04-tensor-parity-record/1',
            'A_RECHECK_SCHEMA', 'unexpected evidence schema')
    require(parity.get('verdict') == {'code': 'TP_MATCH', 'pass': True}
            and parity['mismatches']['total'] == 0
            and parity['cases']['observed_cases'] == 3200
            and len(parent['cases']) == 3200
            and parity['data_files']['n_files'] == 89
            and parity['data_files']['cross_run_equal'] is True,
            'A_RECHECK_PARITY_INVALID', 'registered parity scope/PASS absent')
    require(parent['pins']['v03_head_verified'] == PARENT_PIN
            and parity['pins']['v04_head_verified'] == PIN
            and parity['artifacts']['parent_manifest']['sha256'] == PARENT_SHA,
            'A_RECHECK_PROVENANCE', 'pin or parent binding mismatch')

    sources = {}
    for name, sha in MANIFEST_SHAS.items():
        path = BASE / 'seqref_mri/results/v02/manifests' / name
        verified(path, sha)
        sources[name] = (path, sha)
    sources[parent_path.name] = (parent_path, PARENT_SHA)
    sources[parity_path.name] = (parity_path, PARITY_SHA)

    target = out / 'artifacts'
    target.mkdir()
    for name, (source, sha) in sources.items():
        shutil.copyfile(source, target / name)
        require(digest(target / name) == sha,
                'A_RECHECK_COPY_DRIFT', name)
        (target / (name + '.sha256')).write_text(f'{sha}  {name}\n')
    reference = out / 'v04tp_extended_reference.json'
    reference_sha = write_json(reference, {
        'schema': 'seqref-v04-pipeline-reference/1',
        'artifacts': {name: sha for name, (_, sha) in sources.items()},
    })
    record.update({
        'verified_root': str(root), 'commit_pin': PIN,
        'gate_source_sha256': GATE_SHA,
        'reference': {'path': str(reference), 'sha256': reference_sha},
        'target_dir': str(target),
        'artifact_sha256': {name: sha for name, (_, sha) in sources.items()},
        'tensor_parity_evidence': {'record_sha256': PARITY_SHA,
                                  'cases': 3200, 'training_files': 89},
        'environment': tc.environment(),
    })
    result = gate_a_pipeline_bitexact(reference, target)
    record['gate_a'] = result
    require(result.get('status') == 'PASS'
            and result.get('artifacts_verified') == 7,
            result.get('code', 'A_RECHECK_GATE_FAIL'), str(result))
    for name, (source, sha) in sources.items():
        verified(source, sha)
        require(digest(target / name) == sha,
                'A_RECHECK_INPUT_DRIFT', name)
    verified(reference, reference_sha)
    modules = tc.refresh_modules(modules, str(root), 'after gate-a recheck')
    tc.verify_pin_and_clean_tree(str(root), PIN)
    record['modules'] = modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v04-root', type=Path, default=BASE / 'V04PIN_70f35623')
    parser.add_argument('--out', type=Path,
                        default=BASE / 'V04TP_70f35623/gate_a_recheck')
    args = parser.parse_args()
    out = args.out.resolve()
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print('A_RECHECK_OUTPUT_EXISTS: refusing existing attempt directory',
              file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(out / 'v04tp_gate_a_recheck.log'),
        logging.StreamHandler()], format='%(levelname)s %(name)s: %(message)s')
    record = {
        'schema': 'seqref-v04-gate-a-recheck/1',
        'script': 'SEQREF-V04TP-A v0.1', 'run_card': 'SEQREF-MRI-V04TP v0.9',
        'form': 'supplemental-gate-a-only',
        'wrapper_sha256': digest(Path(__file__)),
        'command': [sys.executable, *sys.argv],
        'scope': 'Five frozen manifests plus two verified tensor-parity '
                 'evidence artifacts. File comparison does not independently '
                 'recompute tensor parity. No aggregate or training verdict.',
    }
    try:
        run(args.v04_root.resolve(), out, record)
        record['status'] = 'PASS'
    except Exception as exc:
        logging.exception('Supplemental gate-a recheck failed; no retry')
        record.update({'status': 'FAIL', 'failure': {
            'code': getattr(exc, 'code', 'A_RECHECK_EXCEPTION:' + type(exc).__name__),
            'detail': str(exc)}})
    try:
        sha = write_json(out / 'v04tp_gate_a_recheck_record.json', record)
    except Exception:
        logging.exception('Record write failed; preserve the attempt log')
        return 2
    print(f"{record['status']} SEQREF-V04TP-A v0.1 sha256={sha} out={out}")
    return 0 if record['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
