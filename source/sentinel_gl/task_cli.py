"""Ordinary CLI for pinned, byte-bound analytical contract recomputation.

This is a standard-library-only engineering path. It cannot admit observational
research, execute a perturbation/model, or issue a research certificate.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import site
import sys

from . import __version__
from .admission import AdmissionError, QUARANTINED
from .integrity import file_hash, loads_strict, verified_file
from .task_contracts import validate_task, validate_sampling_provenance_graph


def required_code_paths():
    root = Path(__file__).resolve().parent
    names = {'__init__.py', 'admission.py', 'integrity.py', 'task_contracts.py', 'task_cli.py',
             'schemas/excluded_digests.json'}
    names.update('schemas/tasks/' + name + '.json' for name in
                 ('observability','control_burden','event_cases','comparison','simulation'))
    return root, sorted(names)


def validate_runtime(lock):
    expected = {'schema_version','python_version','python_implementation','executable_sha256',
                'package_version','code_sha256','seed','device','environment_kind'}
    if type(lock) is not dict or set(lock) != expected or type(lock['schema_version']) is not int or lock['schema_version'] != 1:
        raise AdmissionError('runtime lock fields do not match version 1')
    if lock['python_version'] != platform.python_version() or lock['python_implementation'] != platform.python_implementation():
        raise AdmissionError('runtime Python identity differs from its pin')
    if file_hash(Path(sys.executable).resolve()) != lock['executable_sha256'] or lock['package_version'] != __version__:
        raise AdmissionError('runtime executable/package identity differs from its pin')
    if type(lock['seed']) is not int or not 0 <= lock['seed'] <= 4294967295:
        raise AdmissionError('runtime requires an explicit integer seed')
    if lock['device'] != 'cpu' or lock['environment_kind'] != 'stdlib_only_venv':
        raise AdmissionError('analytical runtime requires the declared CPU standard-library venv')
    if sys.prefix == sys.base_prefix or site.ENABLE_USER_SITE:
        raise AdmissionError('runtime is not an isolated standard-library venv')
    if os.environ.get('PYTHONPATH') or os.environ.get('PYTHONHOME') or os.environ.get('PYTHONHASHSEED') != str(lock['seed']):
        raise AdmissionError('runtime path/hash-seed environment differs from its pin')
    root, paths = required_code_paths()
    if type(lock['code_sha256']) is not dict or set(lock['code_sha256']) != set(paths):
        raise AdmissionError('runtime code membership differs from its pin')
    for name in paths:
        verified_file(root, name, lock['code_sha256'][name])
    random.seed(lock['seed'])
    return {'python_version': platform.python_version(), 'python_implementation': platform.python_implementation(),
            'executable': sys.executable, 'resolved_executable': str(Path(sys.executable).resolve()),
            'executable_sha256': lock['executable_sha256'], 'package_version': __version__,
            'package_location': str(root), 'package_mode': 'SOURCE_CHECKOUT',
            'dependency_scope': 'Python standard library only; no neural/geospatial runtime verified',
            'code_sha256': lock['code_sha256'], 'seed': lock['seed'], 'device': 'cpu',
            'seed_scope': 'Python random and hash seed configured; validation arithmetic is deterministic',
            'venv_prefix': sys.prefix, 'user_site_enabled': site.ENABLE_USER_SITE}


def run(args):
    started = datetime.now(timezone.utc).isoformat()
    input_root = args.input_root
    if any(p.is_symlink() for p in (input_root, *input_root.parents)):
        raise AdmissionError('input root cannot traverse symlinks')
    lock_path = verified_file(input_root, args.runtime_lock, args.runtime_sha256)
    lock = loads_strict(lock_path.read_text())
    runtime = validate_runtime(lock)
    payload_path = verified_file(input_root, args.input, args.input_sha256)
    if args.input_sha256 in QUARANTINED:
        raise AdmissionError('quarantined input bytes')
    payload = loads_strict(payload_path.read_text())
    required = {'task','source_rows','scope_membership','expected_schedule','source_artifacts'}
    if type(payload) is not dict or not required <= set(payload) or set(payload) - required - {'graph'}:
        raise AdmissionError('input fields do not match the analytical CLI contract')
    if payload['task'].get('origin') not in {'fixture','simulation'}:
        raise AdmissionError('observational research execution remains blocked')
    sources = payload['source_rows']
    if type(sources) is not dict or type(payload['source_artifacts']) is not dict or set(payload['source_artifacts']) != set(sources):
        raise AdmissionError('actual source artifact membership is incomplete')
    source_hashes = {}
    for identifier, row in sources.items():
        if row.get('origin') not in {'fixture','simulation'} or row.get('sha256') in QUARANTINED:
            raise AdmissionError('source origin or excluded bytes cannot enter analytical execution')
        path = verified_file(input_root, payload['source_artifacts'][identifier], row['sha256'])
        source_hashes[identifier] = file_hash(path)
    result = validate_task(payload['task'], sources, payload['scope_membership'],
                           expected_schedule=payload['expected_schedule'])
    graph = None if 'graph' not in payload else validate_sampling_provenance_graph(payload['graph'], sources, payload['scope_membership'])
    # Check current bindings again; this does not provide same-user isolation.
    verified_file(input_root, args.input, args.input_sha256)
    verified_file(input_root, args.runtime_lock, args.runtime_sha256)
    validate_runtime(lock)
    for identifier, row in sources.items():
        verified_file(input_root,payload['source_artifacts'][identifier],row['sha256'])
    out = args.output_dir
    if 'fixtures' not in out.parts or out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise AdmissionError('output must be a new fixtures directory without symlinks')
    out.mkdir(parents=True,exist_ok=False)
    results = {'origin':payload['task']['origin'],'research_admissible':False,'task':result,'graph':graph}
    with (out/'result.json').open('x') as f:json.dump(results,f,indent=2,allow_nan=False);f.write('\n')
    receipt = {'scope':'analytical contract recomputation only; no measurement/model/perturbation execution',
               'research_admissible':False,'entrypoint':'python -m sentinel_gl.task_cli','argv':sys.argv,
               'cwd':str(Path.cwd()),'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),
               'exit_status':0,'runtime':runtime,'input_sha256':args.input_sha256,
               'runtime_lock_sha256':args.runtime_sha256,'source_sha256':source_hashes,
               'output_sha256':{'result.json':file_hash(out/'result.json')},
               'cpu_time_seconds':None,'cpu_time_reason':'NOT_MEASURED',
               'peak_memory_bytes':None,'memory_reason':'NOT_MEASURED',
               'limits':'Caller-controlled pins and same-user files; no provider authentication, independent identity attestation or factory research admission.'}
    with (out/'execution.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--input',required=True)
    parser.add_argument('--input-sha256',required=True)
    parser.add_argument('--runtime-lock',required=True)
    parser.add_argument('--runtime-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(f'BLOCKED: {error}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
