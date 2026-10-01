"""Local maintenance counterexamples. Every input/run here is a test fixture."""
import contextlib
import io
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

FACTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FACTORY))
import gatekeeper as g
from engine.audit import Audit
from engine.bundle import create_bundle
from engine.contract import validate_contract
from engine.io import read_json, write_json, inventory, digest, EvidenceError
from engine.metrics import paired_inference
from engine.schema import validate_training_manifest, validate_split_manifest, validate_plausibility_entry
from engine.supervisor import init_supervisor_keys, sign_receipt, verify_receipt_signature
from engine.telemetry import observed_run
from tests.test_v3 import fixture, evaluate, SCRIPT


def standalone_verifier():
    spec = importlib.util.spec_from_file_location('independent_local_verifier', FACTORY/'verify_bundle_standalone.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalReceiptMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)/'fixture_project'
        self.root.mkdir()
        self.key = Path(self.tmp.name)/'keys/supervisor.key'
        self.env = patch.dict(os.environ, {'FACTORY_SUPERVISOR_KEY': str(self.key)})
        self.env.start()
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.plan = fixture(self.root)

    def tearDown(self):
        self.output.__exit__(None, None, None)
        self.env.stop()
        self.tmp.cleanup()

    def execute(self):
        g.freeze(self.root)
        self.assertEqual(g.run_exp(self.root, 'known'), 0)
        _, self.epoch, self.freeze = g.active(self.root)
        self.attempt = self.epoch/'runs/known/attempt0001'
        self.record_path = self.attempt/'execution.json'

    def test_valid_run_is_verified_but_not_sealed(self):
        self.execute()
        audited = evaluate(self.root)
        self.assertEqual(audited['errors'], [])
        self.assertTrue(audited['computed_runs']['known']['receipt_verified'])
        audited['checks_executed'] = ['EXECUTION']
        self.assertEqual(g._compute_assurance_level(audited), 'SUPERVISOR_ATTESTED')

    def test_standalone_verifies_real_nested_execution_receipt(self):
        self.execute()
        archive = Path(self.tmp.name)/'fixture_bundle.zip'
        files = {str(p.relative_to(self.root)): p for p in self.attempt.iterdir() if p.is_file()}
        create_bundle(archive, files, {'factory_version': '3.3.0', 'assurance_level': 'SEALED_EVALUATION_ATTESTED'})
        verify = standalone_verifier().verify_bundle
        unchecked = verify(archive)
        self.assertEqual(unchecked['assurance_level'], 'STRUCTURALLY_VALIDATED')
        checked = verify(archive, self.key.with_suffix('.pub'))
        self.assertEqual(checked['status'], 'PASS')
        self.assertEqual(checked['execution_bindings_verified'], 1)
        self.assertEqual(checked['assurance_level'], 'SUPERVISOR_ATTESTED')
        self.assertEqual(checked['declared_assurance_level'], 'SEALED_EVALUATION_ATTESTED')

    def test_tampering_signed_resource_observation_is_detected(self):
        self.execute()
        record = read_json(self.record_path)
        record['supervisor_receipt']['resource_observations']['cpu_time_seconds'] = 999
        write_json(self.record_path, record)
        self.assertTrue(any('signature' in x['detail'] for x in evaluate(self.root)['errors']))

    def test_valid_signature_on_wrong_run_is_rejected(self):
        self.execute()
        record = read_json(self.record_path)
        receipt = record['supervisor_receipt']
        receipt['launch_spec'] = digest(['some', 'other', 'command'])
        record['supervisor_receipt'] = sign_receipt(receipt)
        write_json(self.record_path, record)
        self.assertTrue(any('binding mismatch: launch_spec' in x['detail'] for x in evaluate(self.root)['errors']))

    def test_output_rehash_does_not_repair_signed_output_binding(self):
        self.execute()
        (self.attempt/'method.txt').write_text('Altered fixture evidence')
        record = read_json(self.record_path)
        outputs = inventory(self.root, [str(self.attempt.relative_to(self.root))])
        outputs.pop(str(self.record_path.relative_to(self.root)))
        record['outputs'] = outputs
        write_json(self.record_path, record)
        self.assertTrue(any('binding mismatch: output_root' in x['detail'] for x in evaluate(self.root)['errors']))

    def test_missing_receipt_is_explicitly_unverified(self):
        self.execute()
        record = read_json(self.record_path)
        record.pop('supervisor_receipt')
        write_json(self.record_path, record)
        audited = evaluate(self.root)
        self.assertFalse(audited['computed_runs']['known']['receipt_verified'])
        self.assertTrue(any(x['code'] == 'UNSIGNED_EXECUTION' for x in audited['diagnostics']))
        self.assertEqual(g._compute_assurance_level(audited), 'STRUCTURALLY_VALIDATED')

    def test_deleting_failed_attempt_leaves_a_detectable_history_gap(self):
        script = SCRIPT.replace("cohort=list", "if out.name == 'attempt0001': raise RuntimeError('constructed first attempt failure')\ncohort=list")
        (self.root/'source/run.py').write_text(script)
        g.freeze(self.root)
        self.assertNotEqual(g.run_exp(self.root, 'known'), 0)
        self.assertEqual(g.run_exp(self.root, 'known'), 0)
        self.assertEqual(evaluate(self.root)['errors'], [])
        _, epoch, _ = g.active(self.root)
        shutil.rmtree(epoch/'runs/known/attempt0001')
        self.assertTrue(any('numbered attempt' in x['detail'] for x in evaluate(self.root)['errors']))

    def test_valid_signature_cannot_coerce_boolean_seed_to_integer(self):
        self.plan['experiments'][0]['seed'] = 1
        write_json(self.root/g.ROOT_PLAN, self.plan)
        self.execute()
        record = read_json(self.record_path)
        record['supervisor_receipt']['seed'] = True
        record['supervisor_receipt'] = sign_receipt(record['supervisor_receipt'])
        write_json(self.record_path, record)
        self.assertTrue(any('binding mismatch: seed' in x['detail'] for x in evaluate(self.root)['errors']))

    def test_typed_run_argv_is_recomputed_from_contract(self):
        script = SCRIPT.replace("out=Path(sys.argv[1]);seed=int(sys.argv[2]);eid=sys.argv[3]",
                                "out=Path(sys.argv[2]);seed=int(sys.argv[3]);eid=sys.argv[1]")
        (self.root/'source/run.py').write_text(script)
        self.plan['experiments'][0]['execution_contract'] = {
            'runtime_id': 'python-cpu-v1', 'entrypoint': 'source/run.py',
            'arguments': {'run_dir': 'supervisor_bound', 'seed': 'plan_seed', 'experiment_id': 'plan_id'},
            'network': 'allowed',
        }
        write_json(self.root/g.ROOT_PLAN, self.plan)
        self.execute()
        self.assertEqual(evaluate(self.root)['errors'], [])

    def test_unenforced_network_denial_retains_failed_attempt(self):
        self.plan['experiments'][0]['execution_contract'] = {
            'runtime_id': 'python-cpu-v1', 'entrypoint': 'source/run.py',
            'arguments': {}, 'network': 'disabled',
        }
        write_json(self.root/g.ROOT_PLAN, self.plan)
        g.freeze(self.root)
        self.assertNotEqual(g.run_exp(self.root, 'known'), 0)
        _, epoch, _ = g.active(self.root)
        attempt = epoch/'runs/known/attempt0001'
        record = read_json(attempt/'execution.json')
        self.assertIsNone(record['exit_code'])
        self.assertFalse((attempt/'result.json').exists())
        self.assertEqual(record['supervisor_receipt']['resource_observations']['unavailable_reason'], 'PROCESS_NOT_STARTED')

    def test_interruption_retains_explicit_unmeasured_resource_reason(self):
        g.freeze(self.root)
        with patch('gatekeeper.observed_run', side_effect=KeyboardInterrupt):
            self.assertNotEqual(g.run_exp(self.root, 'known'), 0)
        _, epoch, _ = g.active(self.root)
        record = read_json(epoch/'runs/known/attempt0001/execution.json')
        self.assertEqual(record['exit_code'], 130)
        self.assertIsNone(record['supervisor_receipt']['resource_observations']['cpu_time_seconds'])
        self.assertEqual(record['supervisor_receipt']['resource_observations']['unavailable_reason'], 'INTERRUPTED_NO_RESOURCE_RESULT')


class SignatureAndContractMaintenanceTests(unittest.TestCase):
    def test_standalone_rejects_duplicate_and_overflow_json(self):
        verify = standalone_verifier().verify_bundle
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp)/'bad.zip'
            for manifest in ('{"schema_version":1,"files":{},"files":{}}', '{"schema_version":1,"files":{},"value":1e999}'):
                with zipfile.ZipFile(archive, 'w') as zipped:
                    zipped.writestr('BUNDLE_MANIFEST.json', manifest)
                self.assertEqual(verify(archive)['status'], 'FAIL')

    def test_public_key_verification_does_not_need_or_create_private_key(self):
        with tempfile.TemporaryDirectory() as temp:
            key = Path(temp)/'supervisor.key'
            with patch.dict(os.environ, {'FACTORY_SUPERVISOR_KEY': str(key)}):
                _, public, _ = init_supervisor_keys()
                receipt = sign_receipt({'origin': 'constructed_fixture', 'value': 1})
                key.unlink()
                self.assertTrue(verify_receipt_signature(receipt, public_key_path=public))
                self.assertFalse(key.exists())

    def test_missing_crypto_cannot_export_hmac_secret_as_public_key(self):
        with tempfile.TemporaryDirectory() as temp:
            key = Path(temp)/'supervisor.key'
            with patch.dict(os.environ, {'FACTORY_SUPERVISOR_KEY': str(key)}), patch('engine.supervisor._try_ed25519', return_value=False):
                with self.assertRaises(EvidenceError):
                    init_supervisor_keys()
                self.assertFalse(key.exists())
                self.assertFalse(key.with_suffix('.pub').exists())

    def test_execution_environment_omits_key_and_provider_credentials(self):
        with patch.dict(os.environ, {'FACTORY_SUPERVISOR_KEY': 'fixture-key-path', 'AWS_SECRET_ACCESS_KEY': 'fixture-secret', 'CDSAPI_KEY': 'fixture-secret'}):
            env = g.execution_env(42)
            for key in ('FACTORY_SUPERVISOR_KEY', 'AWS_SECRET_ACCESS_KEY', 'CDSAPI_KEY'):
                self.assertNotIn(key, env)

    def test_contract_rejects_absolute_and_symlink_parent_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = root/'run.py'
            script.write_text('pass\n')
            contract = {'runtime_id': 'python-cpu-v1', 'entrypoint': str(script), 'arguments': {}}
            with self.assertRaises(EvidenceError):
                validate_contract(contract, root, [str(script)])
            (root/'alias').symlink_to(root, target_is_directory=True)
            contract['entrypoint'] = 'alias/run.py'
            with self.assertRaises(EvidenceError):
                validate_contract(contract, root, ['alias'])

    def test_contract_rejects_unknown_or_misbound_argument(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'run.py').write_text('pass\n')
            for arguments in ({'invented': 'plan_seed'}, {'seed': 'plan_id'}):
                with self.assertRaises(EvidenceError):
                    validate_contract({'runtime_id': 'python-cpu-v1', 'entrypoint': 'run.py', 'arguments': arguments}, root, ['run.py'])


class NumericalAndTelemetryMaintenanceTests(unittest.TestCase):
    def test_precision_recall_and_f1_have_no_universal_half_chance_floor(self):
        for metric in ('precision', 'recall', 'f1', 'accuracy', 'average_precision'):
            findings = g._result_findings({'metric': metric, 'value': .2, 'verdict': 'estimate'})
            self.assertFalse(any(kind == 'below_chance' for kind, _ in findings))
        findings = g._result_findings({'metric': 'precision', 'value': .2, 'chance_reference': .1})
        self.assertFalse(any(kind == 'below_chance' for kind, _ in findings))

    def test_unsupported_verdicts_do_not_count_as_all_supported(self):
        findings = g._result_findings({'entries': [{'verdict': 'unsupported'}]*3})
        self.assertFalse(any(kind == 'all_supported' for kind, _ in findings))

    def test_sign_flip_is_invariant_to_rescaling_tiny_effects(self):
        ordinary = paired_inference([1.]*5, [0.]*5, draws=1000)
        tiny = paired_inference([1e-20]*5, [0.]*5, draws=1000)
        self.assertEqual(ordinary['p_raw'], 2/32)
        self.assertEqual(tiny['p_raw'], ordinary['p_raw'])

    def test_manifest_counts_and_epochs_are_integers(self):
        for value in (1.5, 1., True, '2'):
            with self.assertRaises(EvidenceError):
                validate_training_manifest({'epochs_trained': value})
            with self.assertRaises(EvidenceError):
                validate_split_manifest({'test_label_distribution': {'0': value, '1': 3}})

    def test_pvalues_and_intervals_have_valid_numeric_domains(self):
        for entry in ({'p': -1}, {'p': 1.1}, {'p': float('nan')}, {'ci': [2, 1]}):
            with self.assertRaises(EvidenceError):
                validate_plausibility_entry(entry)

    def test_plausibility_cli_consumes_domains_and_cannot_waive_invalid_numbers(self):
        with contextlib.redirect_stdout(io.StringIO()):
            invalid = {'comparisons': [{'p_value': 1.2}], 'investigation_note': 'A note cannot make an invalid p-value valid.'}
            self.assertNotEqual(g.verify_result_plausibility(invalid), 0)
            self.assertEqual(g.verify_result_plausibility({'config': {'p': 10}}), 0)

    def test_cpu_time_is_observed_separately_from_elapsed_time(self):
        with tempfile.TemporaryFile() as stream:
            code, observed = observed_run([sys.executable, '-c', 'import time; time.sleep(.1)'],
                cwd=Path.cwd(), env=os.environ.copy(), stdout=stream, stderr=stream)
        self.assertEqual(code, 0)
        self.assertGreater(observed['wall_time_seconds'], .08)
        if hasattr(os, 'wait4'):
            self.assertEqual(observed['measurement_method'], 'wait4_child_rusage')
            self.assertGreater(observed['memory_peak_bytes'], 0)
            self.assertGreaterEqual(observed['cpu_time_seconds'], 0)
            self.assertNotEqual(observed['cpu_time_seconds'], observed['wall_time_seconds'])

    def test_unavailable_per_child_resources_are_null(self):
        with patch('engine.telemetry.os.wait4', create=True):
            # Remove availability only for this constructed portability test.
            del os.wait4
            with tempfile.TemporaryFile() as stream:
                code, observed = observed_run([sys.executable, '-c', 'pass'], cwd=Path.cwd(),
                    env=os.environ.copy(), stdout=stream, stderr=stream)
        self.assertEqual(code, 0)
        self.assertIsNone(observed['cpu_time_seconds'])
        self.assertIsNone(observed['memory_peak_bytes'])
        self.assertEqual(observed['unavailable_reason'], 'PER_CHILD_RUSAGE_UNAVAILABLE')
