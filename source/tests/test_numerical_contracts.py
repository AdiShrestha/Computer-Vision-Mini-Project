"""Constructed engineering fixtures, not Earth observations or experiments."""
from datetime import date,timedelta
import numpy as np
import pytest
import torch
from sentinel_gl.features import FitScope,FeatureNormalizer,trailing_windows
from sentinel_gl.model import TimeSeriesMAE
from sentinel_gl.scoring import EmbeddingDistanceScorer,FrozenScoreCombiner,MaskedReconstructionScorer,ema_smooth
from sentinel_gl.metrics import ranking_metrics,false_alert_fraction,first_sustained_alarm
from sentinel_gl.calibration import calibrate_false_alert_threshold
from sentinel_gl.training import fit_masked_autoencoder,StopRule


def small_model():
    return TimeSeriesMAE(n_channels=2,n_windows=6,d_model=8,n_encoder_layers=1,
        n_decoder_layers=1,n_encoder_heads=2,n_decoder_heads=2,d_ff_encoder=16,d_ff_decoder=16,dropout=0.)


def test_normalization_preserves_missingness_and_is_frozen():
    scope=FitScope(('train',),('test',))
    x=np.array([[1.,np.nan],[3.,4.]])
    n=FeatureNormalizer(scope).fit({'train':(x,np.isfinite(x))})
    h=n.state_hash;z,valid=n.transform(x,np.isfinite(x))
    assert z[0,1]==0 and not valid[0,1]
    assert z[:,0].tolist()==[-1.,1.]
    assert n.state_hash==h
    with pytest.raises(RuntimeError):n.fit({'train':(x,np.isfinite(x))})
    with pytest.raises(ValueError):FeatureNormalizer(scope).fit({'test':(x,np.isfinite(x))})
    with pytest.raises(ValueError):FeatureNormalizer(scope).fit({'train':(np.full((2,2),np.nan),np.zeros((2,2),bool))})


def test_missing_mask_and_infinity_are_rejected():
    n=FeatureNormalizer(FitScope(('a',),('b',)))
    with pytest.raises(ValueError):n.fit({'a':(np.ones((2,2)),np.zeros((2,2),bool))})
    with pytest.raises(ValueError):n.fit({'a':(np.full((2,2),np.inf),np.ones((2,2),bool))})


def test_window_dates_use_latest_inputs_and_availability():
    ds=[(date(2023,1,1)+timedelta(days=i)).isoformat() for i in range(6)]
    av=ds.copy();av[2]='2023-01-10'
    windows=trailing_windows(ds,av,3,1)
    assert windows[0].latest_observation_date=='2023-01-03'
    assert windows[0].decision_date=='2023-01-10'
    assert windows[0].stop_index==3
    with pytest.raises(ValueError):trailing_windows(ds[::-1],av,3,1)


def test_unfitted_density_cannot_learn_from_query():
    scorer=EmbeddingDistanceScorer(FitScope(('train',),('test',)),k_neighbors=2,n_components=2)
    x=np.array([[1.,2.],[2.,3.],[3.,1.]])
    with pytest.raises(RuntimeError):scorer.score(x)
    assert scorer.bank is None
    with pytest.raises(ValueError):scorer.fit({'test':x})
    scorer.fit({'train':x})
    reference=scorer.pca.components_.copy()
    assert np.isfinite(scorer.score(x)).all()
    assert np.array_equal(reference,scorer.pca.components_)


def test_combiner_is_invariant_to_future_test_extremes_and_batching():
    scorer=FrozenScoreCombiner(FitScope(('train',),('test',)),.5).fit({'train':([0.,2.],[1.,3.])})
    first=scorer.score([1.],[2.])[0]
    joined=scorer.score([1.,1000.],[2.,1000.])
    assert first==joined[0]==.5
    assert joined[1]>1  # anomaly rank retained, never interpreted as a probability
    with pytest.raises(ValueError):scorer.score([1.,2.],[1.])


def test_unmasked_and_unobserved_losses_do_not_become_nan_or_zero():
    m=small_model();x=torch.ones(1,6,2);v=torch.ones_like(x,dtype=torch.bool)
    with pytest.raises(ValueError,match='no observed masked'):m(x,mask=torch.zeros(1,6,dtype=torch.bool),validity=v)
    with pytest.raises(ValueError):m(torch.zeros_like(x),validity=torch.zeros_like(v))
    badmask=torch.tensor([[True,False,False,False,False,False],[True,True,False,False,False,False]])
    with pytest.raises(ValueError):m.encode(x.expand(2,-1,-1),mask=badmask)


def test_missing_targets_do_not_contribute_and_hidden_values_do_not_leak():
    m=small_model().eval();x=torch.ones(1,6,2);v=torch.ones_like(x,dtype=torch.bool)
    v[0,0,0]=False;x[0,0,0]=0
    mask=torch.tensor([[True,False,True,False,True,False]])
    result=m(x,mask=mask,validity=v)
    assert result['target_count']==5
    result['loss'].backward()
    changed=x.clone();changed[:,mask[0],:]+=100;changed[~v]=0
    assert torch.equal(m.encode(x,mask=mask,validity=v),m.encode(changed,mask=mask,validity=v))


def test_reconstruction_score_is_repeatable_and_counts_observed_targets():
    m=small_model();x=torch.ones(1,6,2);v=torch.ones_like(x,dtype=torch.bool)
    scorer=MaskedReconstructionScorer(m)
    a=scorer.score(x,v);b=scorer.score(x,v)
    assert np.array_equal(a,b) and np.isfinite(a).all()
    with pytest.raises(ValueError):scorer.score(torch.zeros_like(x),torch.zeros_like(v))


def test_ranking_ties_and_single_class_estimability():
    assert ranking_metrics([0,1],[.5,.5])['auroc']==.5
    assert ranking_metrics([0,1],[.5,.5])['average_precision']==.5
    assert ranking_metrics([0,0],[1.,2.])['status']=='NOT_ESTIMABLE'
    assert false_alert_fraction({},1.)['fraction'] is None
    with pytest.raises(ValueError):ranking_metrics([False,True],[.1,.9])
    with pytest.raises(ValueError):ranking_metrics([0,1],[.1,np.nan])


def test_sustained_alarm_is_not_backdated_and_post_event_is_excluded():
    result=first_sustained_alarm([1.,1.,1.],['2023-10-01','2023-10-02','2023-10-04'],.5,'2023-10-04',2,180)
    assert result['alarm_date']=='2023-10-02' and result['lead_time_days']==2
    result=first_sustained_alarm([0.,1.],['2023-10-02','2023-10-04'],.5,'2023-10-04',1,180)
    assert result['lead_time_days'] is None


def test_threshold_isolation_and_ties():
    result=calibrate_false_alert_threshold([0.,0.,1.,1.],['c1','c2','c3','c4'],['e1'],.25)
    assert result['observed_false_alert_fraction']==0.
    with pytest.raises(ValueError):calibrate_false_alert_threshold([1.],['same'],['same'],.1)


def test_ema_uses_only_past_and_current_observations():
    assert np.array_equal(ema_smooth([1.,2.],3),ema_smooth([1.,2.,1000.],3)[:2])
    with pytest.raises(ValueError):ema_smooth([1.,2.],0)


def test_tiny_training_records_budget_and_exact_checkpoint(tmp_path):
    config=dict(n_channels=2,n_windows=6,d_model=8,n_encoder_layers=1,n_decoder_layers=1,
                n_encoder_heads=2,n_decoder_heads=2,d_ff_encoder=16,d_ff_decoder=16,dropout=0.)
    x=torch.arange(12,dtype=torch.float32).reshape(1,6,2)/12;v=torch.ones_like(x,dtype=torch.bool)
    raw=x.squeeze(0).numpy().astype(float)
    normalizer=FeatureNormalizer(FitScope(('fixture_train',),('fixture_test',))).fit(
        {'fixture_train':(raw,np.isfinite(raw))})
    state=dict(normalizer.state)
    normalized,_=normalizer.transform(raw,np.isfinite(raw))
    x=torch.from_numpy(normalized).unsqueeze(0)
    m,report=fit_masked_autoencoder(model_config=config,train_batches=[(x,v)],validation_batches=[(x,v)],seed=9,
        output_dir=tmp_path/'fixture_run',stop_rule=StopRule(2,2,3,0.),transform_state=state,device='cpu',
        learning_rate=.001,weight_decay=.01,max_grad_norm=1.)
    assert report['status']=='BUDGET_EXHAUSTED' and report['scientific_claims']==[]
    ck=torch.load(tmp_path/'fixture_run/checkpoint_best.pt',weights_only=True,map_location='cpu')
    assert all(ck['model_config'][k]==value for k,value in config.items())
    assert ck['model_config']['masking_ratio']==.5
    assert ck['normalization']==state and ck['seed']==9
    assert len(report['history'])==2
