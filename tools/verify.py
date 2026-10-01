#!/usr/bin/env python3
"""Engineering verification and readiness consistency, never research validation.

Run suites into a new directory; preserve failures. Check recorded evidence
against current covered code. Historical audit manifests stay at their epochs.
The schema checker implements only the explicit keyword subset used by this
repository's readiness schema and rejects unsupported keywords.
"""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import platform
import re
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
from sentinel_gl.integrity import loads_strict, verified_file


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def covered_paths(root):
    paths = set()
    for directory in ('source', 'factory/engine', 'factory/tests', 'tools'):
        paths.update((root/directory).rglob('*.py'))
    for relative in ('factory/gatekeeper.py', 'factory/verify_bundle_standalone.py',
                     'factory/run_self_tests.py', 'project/readiness.schema.json'):
        path = root/relative
        if path.is_file():
            paths.add(path)
    return sorted(paths)


def check_schema(value, schema, location='$'):
    supported = {'$schema', '$id', 'title', 'description', 'type', 'const', 'enum',
        'properties', 'required', 'additionalProperties', 'items', 'minItems',
        'maxItems', 'uniqueItems', 'minimum', 'minLength', 'pattern'}
    if set(schema)-supported:
        raise ValueError('unsupported readiness schema keywords: '+str(set(schema)-supported))
    types = {'object':dict, 'array':list, 'string':str, 'integer':int, 'boolean':bool}
    if 'type' in schema and type(value) is not types[schema['type']]:
        raise ValueError(location+': incorrect type')
    if 'const' in schema and value != schema['const']:
        raise ValueError(location+': unexpected constant')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(location+': unsupported value')
    if isinstance(value, dict):
        if not set(schema.get('required', [])) <= set(value):
            raise ValueError(location+': missing required fields')
        properties = schema.get('properties', {})
        for key, child in value.items():
            subschema = properties.get(key, schema.get('additionalProperties', True))
            if subschema is False:
                raise ValueError(location+': unexpected field '+key)
            if isinstance(subschema, dict):
                check_schema(child, subschema, location+'.'+key)
    elif isinstance(value, list):
        if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', len(value)):
            raise ValueError(location+': incorrect item count')
        if schema.get('uniqueItems') and len(set(map(lambda x: json.dumps(x, sort_keys=True), value))) != len(value):
            raise ValueError(location+': duplicate items')
        if 'items' in schema:
            for index, child in enumerate(value):
                check_schema(child, schema['items'], location+f'[{index}]')
    elif isinstance(value, str):
        if len(value.strip()) < schema.get('minLength', 0) or ('pattern' in schema and re.search(schema['pattern'], value) is None):
            raise ValueError(location+': invalid string')
    elif type(value) is int and value < schema.get('minimum', value):
        raise ValueError(location+': below minimum')


def validate_readiness(root=ROOT):
    readiness = loads_strict((root/'project/READINESS.json').read_text())
    schema = loads_strict((root/'project/readiness.schema.json').read_text())
    check_schema(readiness, schema)
    date.fromisoformat(readiness['updated_local_date'])
    return readiness


def check_recorded(root=ROOT):
    readiness = validate_readiness(root)
    evidence_path = root/readiness['verification']['current_evidence']
    evidence = loads_strict(evidence_path.read_text())
    if evidence.get('kind') != 'executed_engineering_verification_not_research_results':
        raise ValueError('wrong evidence type')
    if evidence.get('source_stability') != 'STABLE_DURING_CHECKS':
        raise ValueError('source changed during checks or stability was not recorded')
    if set(evidence['covered_code_hashes']) != {p.relative_to(root).as_posix() for p in covered_paths(root)}:
        raise ValueError('covered code membership changed; run new checks')
    for path, digest in evidence['covered_code_hashes'].items():
        verified_file(root, path, digest)
    for name, count_key, pattern in [('core', 'core_engineering_tests', r'(\d+) passed'),
                                     ('factory', 'factory_offline_tests', r'Ran (\d+) tests')]:
        command = evidence['commands'][name]
        log = verified_file(root, command['log'], command['sha256']).read_text()
        counts = re.findall(pattern, log)
        if (command['exit_code'] != 0 or not counts or int(counts[-1]) != command['tests_passed']
                or command['tests_passed'] != readiness['verification'][count_key]):
            raise ValueError(name+': recorded count/exit/log/readiness mismatch')
    verified_file(root, evidence['dependency_snapshot'], evidence['dependency_snapshot_sha256'])
    for item in evidence['supplied_reviews']:
        verified_file(root, item['path'], item['sha256'])
    return {'status':'PASS', 'scope':'recorded engineering bytes, logs, counts and readiness schema; no scientific validity guarantee'}


def run_suites(output):
    # New directory required, including on failure: no overwritten attempts.
    output = output.resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    before = {p.relative_to(ROOT).as_posix():sha(p) for p in covered_paths(ROOT)}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    commands = {'core':[sys.executable, '-m', 'pytest', 'source/tests', '-q', '-p', 'no:cacheprovider'],
                'factory':[sys.executable, 'factory/run_self_tests.py']}
    records = {}
    for name, argv in commands.items():
        print('Running '+name+' engineering checks', flush=True)
        log_path = output/(name+'.txt')
        with log_path.open('w') as log:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        text = log_path.read_text()
        pattern = r'(\d+) passed' if name == 'core' else r'Ran (\d+) tests'
        counts = re.findall(pattern, text)
        records[name] = {'argv':argv, 'exit_code':result.returncode,
            'tests_passed':int(counts[-1]) if counts and result.returncode == 0 else None,
            'log':log_path.relative_to(ROOT).as_posix(), 'sha256':sha(log_path)}
        print(name+': exit '+str(result.returncode)+', passed '+str(records[name]['tests_passed']), flush=True)
    after = {p.relative_to(ROOT).as_posix():sha(p) for p in covered_paths(ROOT)}
    stable = before == after
    reviews = [{'path':p.relative_to(ROOT).as_posix(), 'sha256':sha(p)} for p in
               sorted(ROOT.glob('docs/research/claude_review_*.txt'))]
    evidence = {'kind':'executed_engineering_verification_not_research_results',
        'date':date.today().isoformat(), 'commands':records,
        'environment':{'python':sys.version, 'platform':platform.platform(),
                       'machine':platform.machine(), 'clean_install':'NOT_VERIFIED'},
        'dependency_snapshot':'source/requirements.lock',
        'dependency_snapshot_sha256':sha(ROOT/'source/requirements.lock'),
        'covered_code_hashes':before,
        'source_stability':'STABLE_DURING_CHECKS' if stable else 'SOURCE_CHANGED_DURING_CHECKS',
        'supplied_reviews':reviews, 'test_origin':'constructed_engineering_fixtures_only',
        'scientific_claims':[], 'domain_execution':'NOT_IMPLEMENTED', 'mps_execution':'NOT_EXECUTED'}
    (output/'verification.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    return 0 if stable and all(r['exit_code']==0 for r in records.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-recorded', action='store_true')
    group.add_argument('--run', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.check_recorded:
        print(json.dumps(check_recorded(), indent=2))
        return 0
    if args.output_dir is None:
        parser.error('--run requires a new --output-dir inside the repository')
    validate_readiness()
    return run_suites(args.output_dir)


if __name__ == '__main__':
    raise SystemExit(main())
