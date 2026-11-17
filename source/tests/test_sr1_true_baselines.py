from __future__ import annotations
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from sentinel_gl.sr1_baselines_true import (
    add_page_cusum, fit_descriptive_baselines, score_descriptive_baselines,
)


def records():
    rows=[];start=datetime(2019,1,1,tzinfo=timezone.utc)
    for index in range(14):
        values=np.full((100,5),index/20,dtype=np.float64)
        observed=np.ones((100,5),dtype=bool)
        rows.append({'origin':'constructed-fixture','observation_id':f'fit-{index}',
            'site_id':'constructed-site','acquired_at':(start+timedelta(days=index*20)).isoformat(),
            'dt':start+timedelta(days=index*20),'role':'FIT','std_features':values,
            'observed':observed,'zones':np.zeros(100,dtype=np.int64)})
    val=dict(rows[-1]);val={**val,'observation_id':'val-1','role':'VAL',
        'acquired_at':(start+timedelta(days=400)).isoformat(),'dt':start+timedelta(days=400)}
    rows.append(val)
    return rows


def test_baseline_fits_refuse_validation_ids():
    rows=records()
    with pytest.raises(ValueError,match='FIT only'):
        fit_descriptive_baselines(rows,['fit-0','val-1'])


def test_baseline_models_have_explicit_fit_ids_and_no_fallback_rows():
    rows=records();fit_ids=[r['observation_id'] for r in rows if r['role']=='FIT']
    fitted=fit_descriptive_baselines(rows,fit_ids)
    assert fitted['fit_observation_ids']==sorted(fit_ids)
    assert fitted['fit_token_count']==14*100
    predictions=score_descriptive_baselines(rows,fit_ids,fitted)
    val=next(r for r in predictions if r['role']=='VAL')
    assert val['zone01_patch_count']==100
    assert val['complete_zone01_patch_count']==100
    assert val['missingness_only_score']==0.0
    assert val['seasonal_status']=='ESTIMATED'
    assert val['spatial_persistence_residual'] is not None
    assert val['isolation_forest_score'] is not None
    assert val['one_class_svm_score'] is not None
    assert 'rpca_score' not in val  # removed by amendment PLOS-001 (zero by construction)


def test_page_cusum_records_actual_exploratory_crossing():
    rows=[{'seed':0,'site_id':'fixture','acquired_at':str(i),'score_c':3.0} for i in range(3)]
    add_page_cusum(rows,kappa=.5,threshold=4)
    assert [r['page_cusum']['alarm_declared'] for r in rows]==[False,True,True]
    assert all(r['page_cusum']['calibration_status']=='EXPLORATORY_FIXED_RULE_NOT_OPERATIONALLY_CALIBRATED' for r in rows)
