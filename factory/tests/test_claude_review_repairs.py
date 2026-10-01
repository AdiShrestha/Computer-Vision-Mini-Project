"""Constructed regression evidence, never measurements for research claims."""
import contextlib
import io
import itertools
import json
import os
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gatekeeper as g
from engine.audit import Audit
from engine.io import read_json, write_json
from engine.metrics import EvidenceError, paired_inference
from tests.test_v3 import fixture, SCRIPT


class CredentialEnvironmentTests(unittest.TestCase):
    def test_allowlist_drops_arbitrary_credentials_and_preserves_os_controls(self):
        inherited = dict(PATH='/fixture/bin', HOME='/fixture/home', TMPDIR='/fixture/tmp',
            LANG='C', LC_NUMERIC='C', TZ='UTC', GITHUB_TOKEN='fixture-secret',
            ANTHROPIC_API_KEY='fixture-secret', EARTHDATA_PASSWORD='fixture-secret',
            CDSE_PASSWORD='fixture-secret', ASF_PASSWORD='fixture-secret',
            COPERNICUS_PASSWORD='fixture-secret', NASA_EARTHDATA_TOKEN='fixture-secret',
            HF_TOKEN='fixture-secret', SSH_AUTH_SOCK='/fixture/socket',
            github_token='fixture-secret', ARBITRARY_UNKNOWN_AUTH='fixture-secret',
            LC_SECRET='fixture-secret', FACTORY_SEED='wrong',
            PYTHONHASHSEED='wrong', PYTHONDONTWRITEBYTECODE='wrong')
        with patch.dict(os.environ, inherited, clear=True):
            env = g.execution_env(42)
        self.assertEqual(env, {'PATH':'/fixture/bin', 'HOME':'/fixture/home',
            'TMPDIR':'/fixture/tmp', 'LANG':'C', 'LC_NUMERIC':'C', 'TZ':'UTC',
            'PYTHONHASHSEED':'42', 'PYTHONDONTWRITEBYTECODE':'1'})

    def test_legacy_and_typed_children_do_not_receive_unknown_credentials(self):
        for typed in (False, True):
            with self.subTest(typed=typed), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)/'fixture'; root.mkdir()
                plan = fixture(root)
                extra = '''
import os
for key in ('GITHUB_TOKEN','EARTHDATA_PASSWORD','CDSE_PASSWORD','SSH_AUTH_SOCK','github_token','UNKNOWN_AUTH_VALUE','FACTORY_SUPERVISOR_KEY'):
    assert key not in os.environ, 'credential key reached worker: '+key
assert os.environ['FACTORY_SEED']=='42'
assert os.environ['FACTORY_EXPERIMENT_ID']=='known'
assert os.environ['PYTHONDONTWRITEBYTECODE']=='1'
(out/'environment_check.json').write_text(json.dumps({'origin':'constructed_fixture','credential_keys_absent':True}))
'''
                script = SCRIPT
                if typed:
                    # The current typed positional ABI sorts argument names.
                    script = script.replace('root=Path.cwd();out=Path(sys.argv[1]);seed=int(sys.argv[2]);eid=sys.argv[3]',
                        'root=Path.cwd();eid=sys.argv[1];out=Path(sys.argv[2]);seed=int(sys.argv[3])')
                    plan['experiments'][0]['execution_contract'] = {
                        'runtime_id':'python-cpu-v1', 'entrypoint':'source/run.py',
                        'arguments':{'experiment_id':'plan_id','run_dir':'supervisor_bound','seed':'plan_seed'},
                        'network':'allowed'}
                (root/'source/run.py').write_text(script+extra)
                write_json(root/'project/research_plan.json', plan)
                inherited = dict(GITHUB_TOKEN='fixture-secret', EARTHDATA_PASSWORD='fixture-secret',
                    CDSE_PASSWORD='fixture-secret', SSH_AUTH_SOCK='/fixture/socket',
                    github_token='fixture-secret', UNKNOWN_AUTH_VALUE='fixture-secret',
                    FACTORY_SUPERVISOR_KEY=str(Path(temp)/'key'))
                with patch.dict(os.environ, inherited), contextlib.redirect_stdout(io.StringIO()):
                    g.freeze(root)
                    self.assertEqual(g.run_exp(root, 'known'), 0)
                    _, epoch, _ = g.active(root)
                    check = read_json(epoch/'runs/known/attempt0001/environment_check.json')
                    self.assertTrue(check['credential_keys_absent'])


class ExactPermutationTests(unittest.TestCase):
    def test_sign_flip_matches_rational_reference_for_rounding_and_cancellation(self):
        cases = [[1., 2**-55, 2**-56, 2**-57, 2**-58],
                 [1., 2**-55, -2**-56], [1., -1., 2**-55, 0.],
                 [1e-200]*5, [1.]*4, [0., 0., 0.]]
        for values in cases:
            with self.subTest(values=values):
                rational = list(map(Fraction, values))
                observed = abs(sum(rational))
                exact = sum(abs(sum(v*s for v,s in zip(rational, signs))) >= observed
                    for signs in itertools.product((-1,1), repeat=len(values)))/2**len(values)
                result = paired_inference(values, [0.]*len(values), draws=1000)
                self.assertEqual(result['p_raw'], exact)
        self.assertEqual(paired_inference(cases[0], [0.]*5, draws=1000)['p_raw'], .0625)

    def test_monte_carlo_ties_use_exact_statistic_and_plus_one_floor(self):
        values = [1.]+[2.**(-55-i) for i in range(16)]
        result = paired_inference(values, [0.]*17, draws=1000)
        self.assertEqual(result['p_raw'], 1/1001)
        self.assertIn('plus_one', result['test'])


class HardwareTrialTrustTests(unittest.TestCase):
    def test_research_trial_tables_are_blocked_even_with_valid_run_id(self):
        audit = Audit(Path.cwd(), {'intent':'research', 'hardware':{'experiment_id':'run'}},
                      Path.cwd(), {}, 'fixture-engine-hash')
        audit.computed['run'] = {'receipt_verified':True}
        with self.assertRaisesRegex(EvidenceError, 'supervisor-bound measurement adapter'):
            audit.hardware()

    def test_fixture_trial_arithmetic_is_explicitly_nonresearch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'trials.csv').write_text('phase,warmup,duration_sec,samples,batch_size,elapsed_sec\n'
                'inference,1,0.001,1,1,0\n'+''.join(
                f'inference,0,0.001,1,1,{i*10}\n' for i in range(5)))
            audit = Audit(root, {'intent':'fixture', 'hardware':{
                'experiment_id':'run','min_trials':5,'minimum_sustained_seconds':30}}, root, {}, 'fixture')
            audit.computed['run'] = {'result_path':'result.json'}
            audit.reports['run'] = {'hardware_trials':'trials.csv'}
            audit.hardware()
            self.assertEqual(audit.hardware_results['latency_ms']['0.5'], 1.)
            self.assertIn('not_research_measurements', audit.hardware_results['scope'])


if __name__ == '__main__':
    unittest.main()
