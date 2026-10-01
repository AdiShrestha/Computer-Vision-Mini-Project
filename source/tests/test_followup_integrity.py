"""Constructed counterexamples only: no provider observations or desired outcomes."""
import numpy as np
import pytest
import torch
from sentinel_gl.integrity import loads_strict, digest
from sentinel_gl.features import FitScope, FeatureNormalizer, trailing_windows
from sentinel_gl.model import TimeSeriesMAE, evaluation_masks
from sentinel_gl.scoring import EmbeddingDistanceScorer, FrozenScoreCombiner, MaskedReconstructionScorer
from sentinel_gl.metrics import first_sustained_alarm
from sentinel_gl.calibration import calibrate_false_alert_threshold
from sentinel_gl.training import fit_masked_autoencoder, StopRule
from test_numerical_contracts import small_model


@pytest.mark.parametrize('text', ['{"x":1e999}', '{"x":-1e999}'])
def test_json_exponent_overflow_is_rejected(text):
    with pytest.raises(ValueError):
        loads_strict(text)


def test_canonical_digest_rejects_implicit_key_coercion():
    with pytest.raises(ValueError):
        digest({1: 'measurement'})
    assert digest({'1': 'measurement'}) == digest(loads_strict('{"1":"measurement"}'))


@pytest.mark.parametrize('magnitude', [1e-200, 1e200])
def test_nonconstant_channels_survive_variance_underflow_and_overflow(magnitude):
    raw = np.array([[magnitude], [2*magnitude]])
    fitted = FeatureNormalizer(FitScope(('train',), ('holdout',))).fit({'train': (raw, np.isfinite(raw))})
    z, _ = fitted.transform(raw, np.isfinite(raw))
    assert fitted.state['constant_channels'] == ()
    assert np.allclose(z[:, 0], [-1., 1.])
    with pytest.raises(AttributeError):
        fitted.state = {'mean': (0.,)}


@pytest.mark.parametrize('raw', [['1', '2'], [False, True], [1+1j, 2+1j]])
def test_no_implicit_conversion_of_fake_numeric_measurements(raw):
    x = np.asarray(raw).reshape(2, 1)
    with pytest.raises(ValueError):
        FeatureNormalizer(FitScope(('train',), ('holdout',))).fit({'train': (x, np.ones(x.shape, bool))})


def test_fitted_combiner_and_density_settings_cannot_silently_change():
    scope = FitScope(('train',), ('holdout',))
    combined = FrozenScoreCombiner(scope, .5).fit({'train': ([0., 2.], [1., 3.])})
    h = combined.state_hash
    with pytest.raises(AttributeError):
        combined.alpha = 0.
    assert combined.state_hash == h and combined.score([1.], [2.])[0] == .5
    x = np.array([[1., 2.], [2., 3.], [3., 1.]])
    density = EmbeddingDistanceScorer(scope, 2, 2).fit({'train': x})
    prior = density.score([[4., 2.]])
    h = density.state_hash
    density.pca.components_[:] = 0
    density.bank._fit_X[:] = 999
    with pytest.raises(AttributeError):
        density.k_neighbors = 1
    assert density.state_hash == h and np.array_equal(prior, density.score([[4., 2.]]))


@pytest.mark.parametrize('cadence', [2, 6, 30])
def test_cross_masks_retain_context_under_sparse_acquisition_cadence(cadence):
    steps = cadence*3
    valid = torch.zeros(2, steps, 2, dtype=torch.bool)
    valid[0, ::cadence] = True
    valid[1, 1::cadence] = True
    masks = evaluation_masks(valid)
    assert torch.all(torch.stack(masks).sum(0) == 1)
    for mask in masks:
        assert torch.all((mask[..., None] & valid).sum((1, 2)) > 0)
        assert torch.all(((~mask)[..., None] & valid).sum((1, 2)) > 0)
        assert len(set((~mask).sum(1).tolist())) == 1
    model = TimeSeriesMAE(n_channels=2, n_windows=steps, d_model=8,
        n_encoder_layers=1, n_decoder_layers=1, n_encoder_heads=2,
        n_decoder_heads=2, d_ff_encoder=16, d_ff_decoder=16, dropout=0.)
    x = valid.float()
    scorer = MaskedReconstructionScorer(model)
    assert np.array_equal(scorer.score(x, valid), scorer.score(x, valid))


def test_generated_training_masks_always_have_real_target_and_context():
    model = small_model()
    valid = torch.zeros(2, 6, 2, dtype=torch.bool)
    valid[:, [0, 4]] = True
    for seed in range(12):  # Engineering coverage, not variance acceptance of results.
        torch.manual_seed(seed)
        mask = model._generate_mask(2, 6, torch.device('cpu'), valid)
        assert torch.all((mask[..., None] & valid).sum((1, 2)) > 0)
        assert torch.all(((~mask)[..., None] & valid).sum((1, 2)) > 0)
    with pytest.raises(ValueError):
        model(torch.ones(1, 6, 2))
    valid[:, 4] = False
    with pytest.raises(ValueError):
        MaskedReconstructionScorer(model).score(valid.float(), valid)


@pytest.mark.parametrize('config', [{'n_windows': 1}, {'n_channels': True}, {'dropout': float('nan')}, {'masking_ratio': 1.}])
def test_invalid_architecture_settings_are_rejected(config):
    with pytest.raises(ValueError):
        TimeSeriesMAE(**config)


def test_alarm_cannot_bridge_long_gap_or_explicit_abstention():
    result = first_sustained_alarm([1., 1.], ['2023-01-01', '2023-10-01'], .5,
        '2023-10-04', 2, 365, eligible=[True, True], max_gap_days=1)
    assert result['status'] == 'NOT_DETECTED' and result['lead_time_days'] is None
    result = first_sustained_alarm([1., None, 1.], ['2023-10-01', '2023-10-02', '2023-10-03'], .5,
        '2023-10-04', 2, 180, eligible=[True, False, True], max_gap_days=1)
    assert result['status'] == 'NOT_DETECTED' and result['alarm_date'] is None


def test_unobserved_event_interval_is_unestimable_not_missed_detection():
    result = first_sustained_alarm([None], ['2023-10-03'], .5,
        '2023-10-04', 2, 180, eligible=[False], max_gap_days=1)
    assert result['status'] == 'NOT_ESTIMABLE' and result['lead_time_days'] is None
    with pytest.raises(ValueError):
        first_sustained_alarm([None], ['2023-10-03'], .5, '2023-10-04', 1, 180,
            eligible=[True], max_gap_days=1)


def test_threshold_ties_obey_exact_empirical_window_budget():
    values = [0., 0., 1., 2., 2.]
    result = calibrate_false_alert_threshold(values, ['c'+str(i) for i in range(5)], ['test'], .4)
    assert result['threshold'] == 2. and result['observed_false_alert_fraction'] == .4
    assert np.mean(np.asarray(values) >= result['threshold']) == .4
    with pytest.raises(ValueError):
        calibrate_false_alert_threshold([np.finfo(float).max], ['c'], ['test'], 0.)


def test_incomplete_trailing_context_is_an_error():
    with pytest.raises(ValueError):
        trailing_windows(['2023-01-01'], ['2023-01-01'], 2, 1)


def test_best_checkpoint_is_separate_from_patience_min_delta(tmp_path, monkeypatch):
    import sentinel_gl.training as training
    losses = iter([1., .95, .90])  # Constructed validation trace; no observational claim.
    monkeypatch.setattr(training, 'validation_loss', lambda *args: next(losses))
    config = dict(n_channels=2, n_windows=6, d_model=8, n_encoder_layers=1,
        n_decoder_layers=1, n_encoder_heads=2, n_decoder_heads=2,
        d_ff_encoder=16, d_ff_decoder=16, dropout=0.)
    x = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)/12
    valid = torch.ones_like(x, dtype=torch.bool)
    state = {'mean': (0., 0.), 'scale': (1., 1.), 'constant_channels': (), 'fit_lake_ids': ('fixture',)}
    _, result = fit_masked_autoencoder(model_config=config, train_batches=[(x, valid)],
        validation_batches=[(x, valid)], seed=9, output_dir=tmp_path/'fixture',
        stop_rule=StopRule(3, 3, 2, .2), transform_state=state, device='cpu',
        learning_rate=.001, weight_decay=.01, max_grad_norm=1.)
    assert result['status'] == 'EARLY_STOPPED' and result['checkpoint_epoch'] == 3
    checkpoint = torch.load(tmp_path/'fixture/checkpoint_best.pt', weights_only=True)
    assert checkpoint['validation_loss'] == .90 and checkpoint['format_version'] == 2
    assert result['split_integrity'] == 'CALLER_LINEAGE_VERIFICATION_REQUIRED'


@pytest.mark.parametrize('mean', [('0', '1'), (False, True)])
def test_checkpoint_normalization_metadata_cannot_coerce_fake_numbers(tmp_path, mean):
    config = dict(n_channels=2, n_windows=6, d_model=8, n_encoder_layers=1,
        n_decoder_layers=1, n_encoder_heads=2, n_decoder_heads=2,
        d_ff_encoder=16, d_ff_decoder=16, dropout=0.)
    x = torch.ones(1, 6, 2)
    state = {'mean': mean, 'scale': (1., 1.), 'constant_channels': (), 'fit_lake_ids': ('fixture',)}
    with pytest.raises(ValueError, match='normalization metadata'):
        fit_masked_autoencoder(model_config=config, train_batches=[(x, x.bool())],
            validation_batches=[(x, x.bool())], seed=9, output_dir=tmp_path/'fixture',
            stop_rule=StopRule(2, 2, 2, 0.), transform_state=state, device='cpu',
            learning_rate=.001, weight_decay=.01, max_grad_norm=1.)
    assert not (tmp_path/'fixture').exists()
