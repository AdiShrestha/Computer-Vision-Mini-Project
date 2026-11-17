"""Real standard-library venv subprocesses over explicit analytical fixtures only."""
from pathlib import Path
import json
import os
import platform
import subprocess
import sys
import venv
import pytest
from sentinel_gl import __version__
from sentinel_gl.integrity import file_hash
from sentinel_gl.task_cli import required_code_paths
from test_task_contracts import inputs, window, task_base

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def python_venv(tmp_path_factory):
    root=tmp_path_factory.mktemp('analytical-runtime')
    venv.EnvBuilder(with_pip=False,symlinks=True).create(root/'venv')
    return root/'venv/bin/python'


def execution(tmp_path,python,*,change_source=False,change_lock=False):
    root=tmp_path/'inputs';root.mkdir()
    sources,scopes=inputs(('s1',))
    (root/'s1.txt').write_text('s1')
    schedule=[window()]
    task=task_base('OBSERVABILITY',['s1'],method_ids=['method'],schedule=schedule,
                   decisions=[{'window_id':'w1','method_id':'method','source_ids':['s1'],
                               'status':'ABSTAINED','value':None,'reason':'ANALYTICAL_MISSING'}])
    payload={'task':task,'source_rows':sources,'scope_membership':scopes,
             'expected_schedule':schedule,'source_artifacts':{'s1':'s1.txt'}}
    (root/'task.json').write_text(json.dumps(payload,allow_nan=False))
    package,paths=required_code_paths()
    lock={'schema_version':1,'python_version':platform.python_version(),
          'python_implementation':platform.python_implementation(),'executable_sha256':file_hash(Path(sys.executable).resolve()),
          'package_version':__version__,'code_sha256':{p:file_hash(package/p) for p in paths},
          'seed':7,'device':'cpu','environment_kind':'stdlib_only_venv'}
    if change_lock:lock['package_version']='unperformed-version'
    (root/'runtime.json').write_text(json.dumps(lock,allow_nan=False))
    if change_source:(root/'s1.txt').write_text('tampered analytical source')
    output=tmp_path/'fixtures/output'
    args=[str(python),'-B','-m','sentinel_gl.task_cli','--input-root',str(root),
          '--input','task.json','--input-sha256',file_hash(root/'task.json'),
          '--runtime-lock','runtime.json','--runtime-sha256',file_hash(root/'runtime.json'),
          '--output-dir',str(output)]
    env={'PYTHONHASHSEED':'7','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','LC_ALL':'C','LANG':'C'}
    result=subprocess.run(args,cwd=ROOT/'source',env=env,capture_output=True,text=True)
    return result,output


def test_ordinary_cli_records_actual_pinned_execution_and_true_poor_outcome(tmp_path,python_venv):
    result,out=execution(tmp_path,python_venv)
    assert result.returncode==0,result.stderr
    payload=json.loads((out/'result.json').read_text());receipt=json.loads((out/'execution.json').read_text())
    assert payload['task']['estimated_decision_fraction']==0. and payload['research_admissible'] is False
    assert receipt['runtime']['seed']==7 and receipt['runtime']['package_mode']=='SOURCE_CHECKOUT'
    assert receipt['runtime']['venv_prefix'] != sys.base_prefix
    assert receipt['output_sha256']['result.json']==file_hash(out/'result.json')
    assert receipt['cpu_time_seconds'] is None and receipt['peak_memory_bytes'] is None


@pytest.mark.parametrize('corruption',['source','runtime'])
def test_cli_rejects_real_source_or_runtime_tampering_without_output(tmp_path,python_venv,corruption):
    result,out=execution(tmp_path,python_venv,change_source=corruption=='source',change_lock=corruption=='runtime')
    assert result.returncode==2 and 'BLOCKED:' in result.stderr and not out.exists()
