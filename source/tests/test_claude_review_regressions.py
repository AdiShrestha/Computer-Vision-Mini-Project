"""Constructed engineering cases; no observations or performance targets."""
from datetime import date, timedelta
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from sentinel_gl.calibration import calibrate_false_alert_threshold as calibrate
from sentinel_gl.features import FitScope, trailing_windows
from sentinel_gl.integrity import digest
from sentinel_gl.metrics import first_sustained_alarm
from sentinel_gl.model import TimeSeriesMAE
from sentinel_gl.scheduling import schedule_as_of
from sentinel_gl.training import fit_masked_autoencoder, StopRule
from test_numerical_contracts import small_model


@pytest.mark.parametrize('ids', ['lakeA', b'lakeA', {'lakeA'}, {'lakeA': 1}, ['lakeA', 'lakeA'], [1]])
def test_calibration_rejects_malformed_ids_on_both_sides(ids):
    with pytest.raises(ValueError):
        calibrate([1., 2., 3.], ['lakeA', 'lakeB', 'lakeC'], ids, .4)
    with pytest.raises(ValueError):
        calibrate([1., 2., 3.], ids, ['holdout'], .4)


def test_valid_ids_preserve_alignment_and_scope_rejects_bare_string():
    result = calibrate([1., 2., 3.], ['a', 'b', 'c'], ['holdout'], .4)
    assert result['threshold'] == 3. and result['calibration_ids'] == ('a', 'b', 'c')
    with pytest.raises(ValueError, match='overlaps'):
        calibrate([1., 2., 3.], ['a', 'b', 'c'], ['a'], .4)
    scope = FitScope(('a', 'b', 'c'), ('holdout',))
    with pytest.raises(ValueError):
        scope.check('abc')
    scope.check({'a': None, 'b': None})


def test_calibration_and_metrics_import_without_torch_or_scoring():
    # Fresh interpreter: the existing test process already imports the model.
    script = '''import builtins,sys
old=builtins.__import__
def guarded(name,*a,**kw):
    if name.split('.')[0]=='torch': raise AssertionError('unexpected torch import')
    return old(name,*a,**kw)
builtins.__import__=guarded
sys.path.insert(0,sys.argv[1])
from sentinel_gl.calibration import calibrate_false_alert_threshold
from sentinel_gl.metrics import ranking_metrics
assert calibrate_false_alert_threshold([1.,2.],['a','b'],['c'],.5)['threshold']==2.
assert ranking_metrics([0,1],[2.,3.])['auroc']==1.
assert 'sentinel_gl.scoring' not in sys.modules
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, '-I', '-B', '-c', script,
                    str(Path(__file__).resolve().parents[1])], check=True,
                   capture_output=True, text=True)


@pytest.mark.parametrize('ids', ['train', ['train', 'train'], [True]])
def test_training_metadata_rejects_malformed_fit_ids(ids, tmp_path):
    x = torch.ones(1, 6, 2)
    valid = torch.ones_like(x, dtype=torch.bool)
    state = {'mean':(0., 0.), 'scale':(1., 1.), 'constant_channels':(), 'fit_lake_ids':ids}
    with pytest.raises(ValueError):
        fit_masked_autoencoder(train_batches=[(x, valid)], validation_batches=[(x, valid)],
            model_config=small_model().configuration, transform_state=state, seed=7,
            stop_rule=StopRule(1, 1, 1, 0.), output_dir=tmp_path/'not_created',
            device='cpu', learning_rate=1e-3, weight_decay=0., max_grad_norm=1.)
    assert not (tmp_path/'not_created').exists()


def late_windows():
    dates = [(date(2023, 1, 1)+timedelta(days=i)).isoformat() for i in range(9)]
    available = dates.copy()
    available[0] = '2023-01-20'
    return trailing_windows(dates, available, 3, 3)


def test_late_release_composes_with_alarm_without_backdating_or_reuse():
    selections = schedule_as_of(late_windows(),
        ['2023-01-03', '2023-01-06', '2023-01-07', '2023-01-09', '2023-01-20'])
    assert [x.window.stop_index if x.window else None for x in selections] == [None, 6, None, 9, None]
    assert all(x.mode == 'RETROSPECTIVE_AS_OF_SIMULATION' for x in selections)
    result = first_sustained_alarm([None, 2., None, 2., None],
        [x.as_of_date for x in selections], 1., '2023-01-21', 2, 30,
        eligible=[x.window is not None for x in selections], max_gap_days=4)
    assert result['status'] == 'NOT_DETECTED'  # explicit abstention resets streak
    result = first_sustained_alarm([2., 2.], ['2023-01-06', '2023-01-09'],
        1., '2023-01-21', 2, 30, eligible=[True, True], max_gap_days=4)
    assert result['alarm_date'] == '2023-01-09' and result['lead_time_days'] == 12


def test_release_batch_selects_one_fresh_context_not_multiple_crossings():
    selections = schedule_as_of(late_windows(), ['2023-01-20', '2023-01-21'])
    assert selections[0].window.stop_index == 9
    assert selections[1].window is None
    with pytest.raises(ValueError):
        schedule_as_of(late_windows(), ['2023-01-20', '2023-01-20'])
    with pytest.raises(ValueError):
        schedule_as_of(late_windows()[::-1], ['2023-01-20'])


def test_capacity_alias_preserves_architecture_and_rejects_ambiguous_config():
    config = small_model().configuration
    model = TimeSeriesMAE(**config)
    assert model.max_time_steps == 6 and model.n_windows == 6
    assert 'n_windows' not in model.configuration
    alias = dict(config)
    alias['n_windows'] = alias.pop('max_time_steps')
    with pytest.warns(DeprecationWarning):
        old = TimeSeriesMAE(**alias)
    assert old.configuration == config and old.count_parameters() == model.count_parameters()
    with pytest.raises(ValueError):
        TimeSeriesMAE(n_windows=6, max_time_steps=6)
    x = torch.zeros(1, 7, 2)
    with pytest.raises(ValueError):
        model.get_full_embeddings(x, torch.ones_like(x, dtype=torch.bool))


def test_python_serialization_digest_preserves_documented_numeric_types():
    assert digest({'a': 1}) != digest({'a': 1.})
    assert digest({'a': (1, 2)}) == digest({'a': [1, 2]})
