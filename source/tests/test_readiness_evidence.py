"""Deliberately corrupt constructed readiness/evidence; no real run claims."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT/'project/READINESS.json').is_file() or not (ROOT/'tools/verify.py').is_file():
    pytest.skip('internal engineering readiness tests skipped in public clone', allow_module_level=True)
spec = importlib.util.spec_from_file_location('engineering_verifier', ROOT/'tools/verify.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_readiness_rejects_boolean_counts_and_scientific_claims_before_ready():
    ready = json.loads((ROOT/'project/READINESS.json').read_text())
    schema = json.loads((ROOT/'project/readiness.schema.json').read_text())
    changed = copy.deepcopy(ready)
    changed['verification']['core_engineering_tests'] = True
    with pytest.raises(ValueError):
        verifier.check_schema(changed, schema)
    changed = copy.deepcopy(ready)
    changed['scientific_claims'] = ['unsupported skill claim']
    with pytest.raises(ValueError):
        verifier.check_schema(changed, schema)
    changed = copy.deepcopy(ready)
    changed['research_blockers'].append(changed['research_blockers'][0])
    with pytest.raises(ValueError):
        verifier.check_schema(changed, schema)


def test_recorded_evidence_detects_changed_code_logs_and_counts(tmp_path):
    # Fixture log strings exercise consistency only, never attest execution.
    (tmp_path/'project').mkdir()
    (tmp_path/'source').mkdir()
    ready = json.loads((ROOT/'project/READINESS.json').read_text())
    ready['verification']['core_engineering_tests'] = 2
    ready['verification']['factory_offline_tests'] = 3
    ready['verification']['current_evidence'] = 'evidence.json'
    (tmp_path/'project/READINESS.json').write_text(json.dumps(ready))
    (tmp_path/'project/readiness.schema.json').write_bytes((ROOT/'project/readiness.schema.json').read_bytes())
    for path, text in {'core.txt':'CONSTRUCTED FIXTURE\n2 passed, 1 skipped\n',
                       'core.stdout.log':'CONSTRUCTED FIXTURE\n2 passed, 1 skipped\n',
                       'core.stderr.log':'',
                       'factory.txt':'CONSTRUCTED FIXTURE\nRan 3 tests\nOK (skipped=1)\n',
                       'factory.stdout.log':'',
                       'factory.stderr.log':'CONSTRUCTED FIXTURE\nRan 3 tests\nOK (skipped=1)\n',
                       'source/code.py':'# constructed fixture\n',
                       'source/requirements.lock':'# fixture\n'}.items():
        (tmp_path/path).write_text(text)
    evidence = {'kind':'executed_engineering_verification_not_research_results',
        'covered_code_hashes':{p.relative_to(tmp_path).as_posix():verifier.sha(p) for p in verifier.covered_paths(tmp_path)},
        'covered_schema_hashes':{p.relative_to(tmp_path).as_posix():verifier.sha(p) for p in verifier.covered_schema_paths(tmp_path)},
        'source_stability':'STABLE_DURING_CHECKS',
        'commands':{name:{'exit_code':0,'tests_passed':2,'tests_skipped':1,
                    'tests_failed':0,'tests_total':3,'log':name+'.txt',
                    'sha256':verifier.sha(tmp_path/(name+'.txt')),
                    'stdout_log':name+'.stdout.log',
                    'stdout_sha256':verifier.sha(tmp_path/(name+'.stdout.log')),
                    'stderr_log':name+'.stderr.log',
                    'stderr_sha256':verifier.sha(tmp_path/(name+'.stderr.log'))}
                   for name in ('core','factory')},
        'dependency_snapshot':'source/requirements.lock',
        'dependency_snapshot_sha256':verifier.sha(tmp_path/'source/requirements.lock'),
        'dependency_inventory':verifier.dependency_inventory(tmp_path),
        'supplied_reviews':[]}
    (tmp_path/'evidence.json').write_text(json.dumps(evidence))
    assert verifier.check_recorded(tmp_path)['status'] == 'PASS'
    evidence['dependency_inventory']['status'] = 'MISMATCH'
    (tmp_path/'evidence.json').write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match='inventory'):
        verifier.check_recorded(tmp_path)
    evidence['dependency_inventory'] = verifier.dependency_inventory(tmp_path)
    (tmp_path/'evidence.json').write_text(json.dumps(evidence))
    original = (tmp_path/'source/code.py').read_bytes()
    (tmp_path/'source/code.py').write_text('# changed fixture\n')
    with pytest.raises(ValueError, match='checksum'):
        verifier.check_recorded(tmp_path)
    (tmp_path/'source/code.py').write_bytes(original)
    (tmp_path/'source/added.py').write_text('# untested added fixture\n')
    with pytest.raises(ValueError, match='membership'):
        verifier.check_recorded(tmp_path)
    (tmp_path/'source/added.py').unlink()
    schema_path=tmp_path/'project/readiness.schema.json'
    schema_original=schema_path.read_bytes()
    schema_path.write_bytes(schema_original+b' ')
    with pytest.raises(ValueError, match='checksum'):
        verifier.check_recorded(tmp_path)
    schema_path.write_bytes(schema_original)
    (tmp_path/'source/sentinel_gl/schemas').mkdir(parents=True)
    extra_schema=tmp_path/'source/sentinel_gl/schemas/extra.json'
    extra_schema.write_text('{"type":"object"}')
    with pytest.raises(ValueError, match='schema membership'):
        verifier.check_recorded(tmp_path)
    extra_schema.unlink()
    (tmp_path/'source/sentinel_gl/schemas').rmdir()
    evidence['commands']['core']['tests_passed'] = 4
    (tmp_path/'evidence.json').write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match='mismatch'):
        verifier.check_recorded(tmp_path)
