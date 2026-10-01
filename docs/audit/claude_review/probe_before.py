"""Replay constructed baseline probes; never import observations/checkpoints."""
import ast
from fractions import Fraction
from itertools import product
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
BASELINE = '7f0a7d8084c3df362b917643a1d7bdb4b5ef3478'
PROBE = r'''from datetime import date,timedelta
from fractions import Fraction
from itertools import product
import json
from sentinel_gl.calibration import calibrate_false_alert_threshold as cal
from sentinel_gl.features import FitScope,trailing_windows
from engine.metrics import paired_inference
out={'origin':'constructed_engineering_counterexamples','baseline_commit':'7f0a7d8084c3df362b917643a1d7bdb4b5ef3478'}
out['bare_final_ids_accepted_threshold']=cal([1.,2.,3.],['lakeA','lakeB','lakeC'],'lakeA',.4)['threshold']
out['bare_calibration_ids_accepted']=cal([1.,2.,3.],'abc',['holdout'],.4)['calibration_ids']
out['fit_scope_bare_ids_accepted']=FitScope(('a','b','c'),('holdout',)).check('abc') is None
ds=[(date(2023,1,1)+timedelta(days=i)).isoformat() for i in range(9)]
av=ds.copy();av[0]='2023-01-20'
out['nonmonotone_window_earliest_dates']=[w.decision_date for w in trailing_windows(ds,av,3,3)]
x=[1.,2**-55,2**-56,2**-57,2**-58];q=list(map(Fraction,x));target=abs(sum(q))
p=sum(abs(sum(a*s for a,s in zip(q,signs)))>=target for signs in product((-1,1),repeat=len(q)))/2**len(q)
out['sign_flip']={'differences':x,'baseline_p':paired_inference(x,[0.]*5,draws=1000)['p_raw'],'rational_reference_p':p}
print(json.dumps(out,allow_nan=False))
'''

def git_bytes(ref, path):
    return subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)

with tempfile.TemporaryDirectory(prefix='sentinel_baseline_probes_') as temp:
    staging=Path(temp)
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',BASELINE,'source/sentinel_gl','factory/engine'],cwd=ROOT,text=True).splitlines()
    for relative in paths:
        if relative.endswith('.py'):
            dst=staging/relative;dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_bytes(git_bytes(BASELINE,relative))
    child='import sys;sys.path[:0]=sys.argv[1:3]\n'+PROBE
    result=subprocess.run([sys.executable,'-I','-B','-c',child,str(staging/'source'),str(staging/'factory')],cwd=staging,check=True,capture_output=True,text=True)
    output=json.loads(result.stdout)

ref='legacy/sentinel-gl-working-tree-2026-09-30'
path='source/models/encoder/ts_mae.py'
raw=git_bytes(ref,path)
expected='71c3ec02c28458000f85370d5cdf2177b8907d002e99d4b6c489233e015696e1'
assert hashlib.sha256(raw).hexdigest()==expected
# Only inspected architecture imports/classes; no dataset or training script.
tree=ast.parse(raw.decode())
tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef))]
ns={'__name__':'forensic_untrained_architecture'}
exec(compile(tree,path,'exec'),ns)
cls=ns['TimeSeriesMAE']
config=dict(n_channels=13,n_windows=180,d_model=128,n_encoder_layers=4,n_decoder_layers=2,n_encoder_heads=4,n_decoder_heads=4,masking_ratio=.5)
bound=inspect.signature(cls).bind(**config);bound.apply_defaults()
output['legacy_architecture']={'origin':'forensic_untrained_architecture_instantiation_not_research_result','tag':ref,'source_path':path,'source_sha256':expected,'default_parameters':cls().count_parameters(),'training_script_resolved_config':dict(bound.arguments),'training_script_parameters':cls(**config).count_parameters()}
print(json.dumps(output,indent=2,allow_nan=False))
