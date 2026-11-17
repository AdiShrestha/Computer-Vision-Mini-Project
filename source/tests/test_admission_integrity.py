"""Corruption and hand-calculated controls for containment.
# FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
All successful transforms and poor outcomes here are analytical controls.
"""
from copy import deepcopy
from pathlib import Path
import importlib
import json
import math
import subprocess
import sys
import numpy as np
import pytest
from admission_fixtures import measurement_fixture
from sentinel_gl.admission import AdmissionContext, AdmissionError, QUARANTINED, validate_decisions, validate_scopes, fit_fixture_operators
from sentinel_gl.integrity import digest, file_hash, loads_strict
from sentinel_gl.features import FeatureExtractionPipeline, MultiModalPanel, FEATURE_CHANNELS, NUM_CHANNELS, DEFAULT_CHANNEL_SCHEMA, compute_ndwi
from sentinel_gl.predictions import ABLATION_CONFIGS, check_window_eligibility, compute_paired_score_differences, make_prediction_record
from sentinel_gl.metrics import ranking_metrics, first_sustained_alarm
from sentinel_gl.inference import poisson_rate_confidence_interval, cluster_conditional_metrics
from sentinel_gl.episodes import compute_alert_burden_estimand

ROOT = Path(__file__).resolve().parents[2]


def extract(context, record, modality='optical'):
    return FeatureExtractionPipeline(3, admission=context).extract_panel('ANALYTICAL-SITE','window','2023-01-01', **{modality + '_observations':[record]})


def _native_sar_measurement_fixture(context, record):
    record=deepcopy(record)
    record.update(schema_version=2,product='SENTINEL-1_GRD_NATIVE',version='fixture-native-v1',
                  units={'DN':'count'})
    product_path=context.artifact(record['product_id'])
    with product_path.open('wb') as stream:
        np.savez(stream,valid=np.array([[True,True]]),roi=np.array([[True,True]]),
            DN=np.array([[10,2]],dtype=np.uint16))
    product_entry=context.manifest['artifacts'][record['product_id']]
    product_entry['sha256']=file_hash(product_path)
    role_ids={'measurement':record['product_id']}
    for role in ('calibration','noise','annotation','manifest'):
        artifact_id='native-'+role
        path=context.root/(artifact_id+'.json')
        path.write_text(json.dumps({'role':role,'origin':'constructed-fixture'}))
        context.manifest['artifacts'][artifact_id]={'artifact_id':artifact_id,'path':path.name,
            'sha256':file_hash(path),'origin':'fixture','parents':[]}
        role_ids[role]=artifact_id
    roles={role:{'artifact_id':artifact_id,
        'sha256':context.manifest['artifacts'][artifact_id]['sha256']}
        for role,artifact_id in role_ids.items()}
    native={'schema_version':1,'product_id':record['product_id'],'product_type':'GRD','mode':'IW',
        'polarization':'VV','ipf_version':'constructed-fixture-IPF','subswath':'IW1','roles':roles,
        'rows':1,'columns':2,'dtype':'uint16','line_offset':0,'sample_offset':0,
        'processing_flags':{'input_radiometry':'RAW_DN','sigma0_calibration_applied':False,
            'thermal_noise_correction_applied':False,'incidence_normalization_applied':False,
            'terrain_correction_applied':False},'noise_status':'UNAVAILABLE_UNVERIFIED_PRODUCT'}
    record['native_sar']=native
    product_entry['parents']=list(dict.fromkeys(product_entry['parents']+
        [artifact_id for role,artifact_id in role_ids.items() if role!='measurement']))
    metadata=context.artifact(record['metadata_id'])
    metadata.write_text(json.dumps({k:record[k] for k in ('product','version','endpoint','units','crs',
        'support_start','support_end','available_at','availability_policy','native_sar')}))
    context.manifest['artifacts'][record['metadata_id']]['sha256']=file_hash(metadata)
    query=context.artifact(record['query_id'])
    query.write_text(json.dumps({'endpoint':record['endpoint'],'roi_id':record['roi_id'],
        'product':record['product']}))
    context.manifest['artifacts'][record['query_id']]['sha256']=file_hash(query)
    context.manifest['measurements'][record['measurement_id']]=digest(record)
    return record


def test_native_sar_v2_measurement_semantically_binds_roles_and_correction_state(tmp_path):
    context,record=measurement_fixture(tmp_path,modality='sar')
    record=_native_sar_measurement_fixture(context,record)
    assert context.measurement(record)==record
    altered=deepcopy(record)
    altered['native_sar']['roles']['noise']['sha256']='0'*64
    with pytest.raises(ValueError,match='digest'):
        context.measurement(altered)
    altered=deepcopy(record)
    altered['native_sar']['processing_flags']['thermal_noise_correction_applied']=True
    with pytest.raises(ValueError):
        context.measurement(altered)


def test_ct01_catalogue_cannot_become_observed_and_real_fixture_bytes_transform(tmp_path):
    empty = {'acquisition_timestamp':'2023-01-01T00:00:00Z','status':'CATALOGUED','metadata':{}}
    panel=FeatureExtractionPipeline(3).extract_panel('X','W','2023-01-01',optical_observations=[empty],sar_observations=[empty],era5_observations=[empty])
    assert not panel.mask.any() and np.isnan(panel.values).all()
    assert set(panel.absence_reasons.values()) == {'NO_ADMITTED_MEASUREMENT_BYTES'}
    assert set(panel.statuses.flat) == {'UNAVAILABLE_NO_ADMITTED_SUPPORT'}
    ctx,r=measurement_fixture(tmp_path)
    p=extract(ctx,r)
    # 100 square metres of declared fixture water; NDWI=(.3-.1)/(.3+.1).
    assert p.values[1,0] == pytest.approx(.0001)
    assert p.values[1,1] == pytest.approx(.5)
    assert p.origin=='fixture' and p.static_metadata is None
    assert not p.mask[0].any() and np.isnan(p.values[0]).all()


@pytest.mark.parametrize('field', ['units','roi_id','qa_id','product_id'])
def test_ct02_incomplete_record_rejected(tmp_path,field):
    ctx,r=measurement_fixture(tmp_path)
    del r[field]
    with pytest.raises(ValueError):extract(ctx,r)


def test_ct02_missing_band_and_deleted_contribution(tmp_path):
    ctx,r=measurement_fixture(tmp_path/'one',fields={'B03':np.array([[.3,.2]]),'B08':np.array([[.1,.1]])})
    missing=extract(ctx,r)
    independently_recomputed_ndwi=np.mean([(.3-.1)/(.3+.1),(.2-.1)/(.2+.1)])
    assert independently_recomputed_ndwi==pytest.approx(5/12)
    assert missing.mask[1,1] and missing.values[1,1]==pytest.approx(independently_recomputed_ndwi)
    assert not missing.mask[1,2] and np.isnan(missing.values[1,2])
    assert missing.statuses[1,2]=='UNAVAILABLE_MISSING_B11'
    assert not missing.mask[1,0] and missing.statuses[1,0]=='UNAVAILABLE_MASK_OR_AREA'
    a,ra=measurement_fixture(tmp_path/'two',suffix='a')
    b,rb=measurement_fixture(tmp_path/'two',suffix='b',day='2023-01-03')
    manifest=deepcopy(a.manifest);manifest['artifacts'].update(b.manifest['artifacts']);manifest['measurements'].update(b.manifest['measurements'])
    ctx=AdmissionContext(tmp_path/'two',manifest,mode='fixture')
    first=FeatureExtractionPipeline(3,admission=ctx).extract_panel('ANALYTICAL-SITE','w','2023-01-01',optical_observations=[ra,rb])
    reduced=extract(ctx,ra)
    assert np.allclose(first.values[1],reduced.values[1],equal_nan=True)
    assert first.mask[2].any() and not reduced.mask[2].any()
    assert digest(first.provenance_hashes)!=digest(reduced.provenance_hashes)
    ctx.artifact(rb['product_id']).unlink()
    with pytest.raises(ValueError):extract(ctx,rb)


def test_ct02_units_and_qa_are_not_guessed(tmp_path):
    ctx,r=measurement_fixture(tmp_path/'w',modality='weather')
    p=extract(ctx,r,'era5')
    assert p.values[1,7]==pytest.approx(1.) and p.values[1,8]==pytest.approx(2.)
    assert "era5_temp_anomaly" not in FEATURE_CHANNELS
    ctx,r=measurement_fixture(tmp_path/'u',modality='weather',units={'temperature':'unknown','precipitation':'mm'})
    with pytest.raises(ValueError):extract(ctx,r,'era5')
    ctx,r=measurement_fixture(tmp_path/'q')
    q=ctx.artifact(r['qa_id']);q.write_text(json.dumps({'status':'UNKNOWN','policy':'analytical-mask-v1'}))
    ctx.manifest['artifacts'][r['qa_id']]['sha256']=file_hash(q)
    with pytest.raises(ValueError,match='QA'):extract(ctx,r)


def test_ct03_quarantine_after_rename_and_fixture_ancestry(tmp_path):
    ctx,r=measurement_fixture(tmp_path)
    # Actual preserved current bytes are read solely to prove their rejection.
    renamed=tmp_path/'renamed.dat';renamed.write_bytes((ROOT/'data/feature_panels.npz').read_bytes())
    entry={'artifact_id':'renamed','path':renamed.name,'sha256':file_hash(renamed),'origin':'observational','parents':[]}
    manifest={'schema_version':1,'artifacts':{'renamed':entry},'measurements':{}}
    with pytest.raises(ValueError,match='quarantined'):AdmissionContext(tmp_path,manifest).artifact('renamed')
    m=deepcopy(ctx.manifest)
    for e in m['artifacts'].values():e['origin']='observational'
    m['artifacts'][r['qa_id']]['origin']='fixture'
    with pytest.raises(ValueError,match='origin'):AdmissionContext(tmp_path,m).artifact(r['product_id'])
    forged=deepcopy(r);forged['origin']='observational'
    with pytest.raises(ValueError):ctx.measurement(forged)
    with pytest.raises(ValueError):AdmissionContext(tmp_path,ctx.manifest).measurement(r)
    # A legacy digest is rejected by the same content guard, independent of names.
    legacy=loads_strict((ROOT/'docs/audit/legacy_inventory.json').read_text())['files'][0]['sha256']
    assert legacy in QUARANTINED


def scope_fixture():
    scopes={r:[r] for r in ('fitting','development_validation','threshold_calibration','evaluation')}
    membership={r:{'scope':r,'site_id':'site-'+r,'available_at':'2023-06-30T00:00:00Z','cutoff':'2023-07-01T00:00:00Z'} for r in scopes}
    return scopes,membership


def test_ct04_actual_fits_require_joined_scopes_and_future_append_is_inert():
    scopes,membership=scope_fixture()
    dates=tuple(str(np.datetime64('2023-01-01')+np.timedelta64(i,'D')) for i in range(180))
    values=np.arange(180*NUM_CHANNELS,dtype=float).reshape(180,NUM_CHANNELS)
    mask=np.ones_like(values,dtype=bool);statuses=np.full(mask.shape,"OBSERVED",dtype=object)
    p=MultiModalPanel('site-fitting','fitting',dates[0],dates[-1],dates,values,mask,None,origin='fixture',statuses=statuses)
    def fit():return fit_fixture_operators({'fitting':p},{'fitting':np.array([[1.,2.],[3.,4.]])},{'threshold_calibration':.2},scopes,membership,[])
    before=fit();old=before['normalizer'].state_hash
    # Future/evaluation panels are not admitted into actual fit membership.
    with pytest.raises(ValueError):fit_fixture_operators({'evaluation':p},{'fitting':np.array([[1.,2.],[3.,4.]])},{'threshold_calibration':.2},scopes,membership,[])
    p2=deepcopy(p);object.__setattr__(p2,'lake_id','site-evaluation')
    with pytest.raises(ValueError):fit_fixture_operators({'fitting':p2},{'fitting':np.array([[1.,2.],[3.,4.]])},{'threshold_calibration':.2},scopes,membership,[])
    assert fit()['normalizer'].state_hash==old
    from sentinel_gl.scoring import EmbeddingDistanceScorer
    with pytest.raises(ValueError):EmbeddingDistanceScorer(before['pca'].scope,1,1).fit({'site-evaluation':np.array([[1.,2.],[3.,4.]])})
    for role in ('fitting','threshold_calibration'):
        changed=deepcopy(scopes);changed[role]=[]
        with pytest.raises(ValueError,match='empty'):validate_scopes(changed,membership,[])
    # Renamed display/window identifiers still join to an already exposed canonical site.
    membership['evaluation']['site_id']='SGL-001'
    with pytest.raises(ValueError,match='exposed'):validate_scopes(scopes,membership,['SGL-001','SGL-002'])


def test_ct05_unestimable_is_distinct_from_zero_and_miss():
    assert math.isnan(compute_ndwi(0.,0.))
    assert ranking_metrics([0,0],[1.,2.])['auroc'] is None
    absent=compute_alert_burden_estimand([],{});zero=compute_alert_burden_estimand([],{'site':1.})
    assert absent['lambda_alert'] is None and zero['lambda_alert']==0.
    assert poisson_rate_confidence_interval(0,1.)==(0.,pytest.approx(-math.log(.025)))
    miss=first_sustained_alarm([.1],['2023-01-01'],.5,'2023-01-03',1,10,eligible=[True],max_gap_days=1)
    assert miss['status']=='NOT_DETECTED' and miss['lead_time_days'] is None
    assert 'all_off' in ABLATION_CONFIGS and not check_window_eligibility(
        np.ones((3,len(FEATURE_CHANNELS)),bool),'all_off',1,channel_names=FEATURE_CHANNELS)
    r=make_prediction_record('site','2023-01-02','2023-01-01','2023-01-02','m','all_off',None,False)
    assert r.to_dict()['score'] is None
    with pytest.raises(ValueError):make_prediction_record('site','2023-01-02','2023-01-01','2023-01-02','m','all_off',0.,False)


def decisions():
    schedule=[{'window_id':'w','closed_support':'analytic'}]
    bindings={'bad_method':{'model_hash':'poor-analytic-method','scope_hash':'declared','schema_hash':'v1'}}
    row={'schema_version':1,'window_id':'w','method_id':'bad_method','window_hash':digest(schedule[0]),'binding_hash':digest(bindings['bad_method']),'value':0.,'status':'ESTIMATED','reason':None}
    return [row],schedule,bindings


def test_ct06_complete_decisions_and_current_bindings():
    rows,s,b=decisions();assert validate_decisions(rows,s,b)['status']=='INTEGRITY_VALID'
    for changed in ([],rows+rows,[{**rows[0],'window_id':'orphan'}],[{**rows[0],'method_id':'renamed'}],[{**rows[0],'value':None}]):
        with pytest.raises(ValueError):validate_decisions(changed,s,b)
    for field in ('model_hash','scope_hash','schema_hash'):
        changed=deepcopy(b);changed['bad_method'][field]='altered'
        with pytest.raises(ValueError):validate_decisions(rows,s,changed)
    with pytest.raises(ValueError):validate_decisions(rows,[{**s[0],'closed_support':'changed'}],b)
    rows[0].update(value=None,status='NOT_ESTIMABLE',reason='ALL_OFF')
    assert validate_decisions(rows,s,b)['rows']==1


def test_ct07_raw_scale_cannot_produce_superiority_or_more_clusters():
    for scale in (1.,10.):
        with pytest.raises(ValueError,match='performance estimand'):compute_paired_score_differences([scale],[2.])
    for count in (1,100):
        result=cluster_conditional_metrics({'site':{'n_windows':count,'outcome':0.}},{'site':'one-cluster'})
        assert len(result['clusters'])==1


@pytest.mark.parametrize('name',['run_extract_features','run_train_model','run_predictions','run_inference_diagnostics','run_generate_reports','run_domain_detector','run_split_manifest','run_pilot_ingestion','run_cohort_and_insar'])
def test_ct08_production_cli_cannot_emit_research_or_overwrite_history(tmp_path,name):
    cmd=[sys.executable,'-B',str(ROOT/'source/runners'/ (name+'.py')),'--output',str(tmp_path/'forbidden')]
    r=subprocess.run(cmd,capture_output=True,text=True)
    assert r.returncode != 0 and not list(tmp_path.iterdir())


def test_ct08_valid_poor_outcome_passes_integrity_but_does_not_release(tmp_path):
    # Hand calculation: both positive cases rank below negatives => AUROC 0.
    poor=ranking_metrics([1,1,0,0],[0.,1.,2.,3.])
    assert poor['status']=='ESTIMABLE' and poor['auroc']==0.
    rows,s,b=decisions();rows[0]['value']=poor['auroc']
    assert validate_decisions(rows,s,b)['status']=='INTEGRITY_VALID'
    from tools.verify_release import generate_release_report
    report=generate_release_report(tmp_path/'release.json')
    assert report['overall_status']=='BLOCKED' and not report['scientific_claims']
    with pytest.raises(FileExistsError):generate_release_report(tmp_path/'release.json')
    # Public implementation imports project modules/third-party packages only.
    import ast
    for p in (ROOT/'source').rglob('*.py'):
        tree=ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith('factory')
            if isinstance(node,ast.Import):assert all(not x.name.startswith('factory') for x in node.names)


@pytest.mark.parametrize('text',['{"x":1,"x":2}','{"x":NaN}','{"x":1e999}'])
def test_ct09_strict_parsing(text):
    with pytest.raises(ValueError):loads_strict(text)


def test_ct09_type_confusion_paths_and_current_bytes(tmp_path):
    ctx,r=measurement_fixture(tmp_path)
    for key,value in (('schema_version',True),('units',[]),('available_at',True)):
        changed=deepcopy(r);changed[key]=value
        with pytest.raises(ValueError):ctx.measurement(changed)
    path=ctx.artifact(r['product_id']);original=path.read_bytes();path.write_bytes(original+b'bad')
    with pytest.raises(ValueError,match='checksum'):ctx.measurement(r)
    path.write_bytes(original)
    for altered in ('../escape.npz','/absolute.npz'):
        ctx.manifest['artifacts'][r['product_id']]['path']=altered
        with pytest.raises(ValueError):ctx.measurement(r)
    ctx.manifest['artifacts'][r['product_id']]['path']=path.name
    link=tmp_path/'link';link.symlink_to(path)
    ctx.manifest['artifacts'][r['product_id']]['path']='link'
    with pytest.raises(ValueError,match='symlink'):ctx.measurement(r)


def test_ct09_derived_membership_is_complete(tmp_path):
    ctx,r=measurement_fixture(tmp_path)
    artifact=tmp_path/'derived.json';artifact.write_text('{"fixture_value":0}')
    ctx.manifest['artifacts']['derived']={'artifact_id':'derived','path':artifact.name,'sha256':file_hash(artifact),'origin':'fixture','parents':[r['product_id']]}
    record={'schema_version':1,'artifact_id':'derived','origin':'fixture','parents':[r['product_id']],'measurements':[r]}
    assert ctx.derived(record)==record
    for changed in ({**record,'parents':[]},{**record,'measurements':[]},{**record,'origin':'observational'}):
        with pytest.raises(ValueError):ctx.derived(changed)
    ctx.manifest['measurements'][r['measurement_id']]='0'*64
    with pytest.raises(ValueError,match='membership'):ctx.derived(record)


def test_ct02_failed_request_has_no_fictional_acquisition_time(tmp_path):
    from sentinel_gl.admission import validate_request
    ctx,r=measurement_fixture(tmp_path)
    query=ctx.artifact(r['query_id']);query.write_text(json.dumps({'endpoint':'fixture://request'}))
    response=ctx.artifact(r['response_id']);response.write_text(json.dumps({'query_id':r['query_id'],'status':'NO_DATA'}))
    for identifier,path in ((r['query_id'],query),(r['response_id'],response)):
        ctx.manifest['artifacts'][identifier]['sha256']=file_hash(path)
    request={'schema_version':1,'request_id':'unavailable','endpoint':'fixture://request','requested_at':'2023-01-03T00:00:00Z','query_id':r['query_id'],'response_id':r['response_id'],'reason':'NO_PRODUCTS','origin':'fixture','status':'NO_DATA'}
    ctx.manifest['requests']={'unavailable':digest(request)}
    absence=validate_request(request,ctx)
    assert absence.value is None and absence.reason=='NO_PRODUCTS'
    with pytest.raises(ValueError):validate_request({**request,'acquisition_timestamp':'2023-01-01T00:00:00Z'},ctx)


def test_ct03_fixture_cli_is_explicit_and_cannot_enter_research(tmp_path):
    ctx,r=measurement_fixture(tmp_path/'inputs')
    root=ctx.root
    manifest=root/'manifest.json';manifest.write_text(json.dumps(ctx.manifest))
    records=root/'records.json';records.write_text(json.dumps({'optical':[r],'sar':[],'era5':[]}))
    output=tmp_path/'fixtures'/'analytical-output'
    cmd=[sys.executable,'-B',str(ROOT/'source/runners/run_fixture_extraction.py'),
         '--input-root',str(root),'--manifest','manifest.json','--manifest-sha256',file_hash(manifest),
         '--records','records.json','--records-sha256',file_hash(records),'--output-dir',str(output),
         '--lake-id','ANALYTICAL-SITE','--start-date','2023-01-01','--window-days','3']
    result=subprocess.run(cmd,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    summary=loads_strict((output/'summary.json').read_text())
    assert summary['origin']=='fixture' and summary['research_admissible'] is False
    with np.load(output/'panel.npz',allow_pickle=False) as panel:
        assert panel['values'][1,1]==pytest.approx(.5)
    repeated=subprocess.run(cmd,capture_output=True,text=True)
    assert repeated.returncode!=0


def test_ct05_all_missing_baseline_is_null_with_reason():
    from sentinel_gl.baselines import LakeAreaTrendBaseline
    dates=tuple(str(np.datetime64('2023-01-01')+np.timedelta64(i,'D')) for i in range(180))
    missing=np.full((180,NUM_CHANNELS),np.nan);mask=np.zeros(missing.shape,bool)
    statuses=np.full(mask.shape,"UNAVAILABLE_TEST_FIXTURE",dtype=object)
    p=MultiModalPanel('fixture-site','fixture-window',dates[0],dates[-1],dates,missing,mask,None,origin='fixture',statuses=statuses)
    baseline=LakeAreaTrendBaseline.fit_fixture([p],['fixture-site'])
    result=baseline.predict_status(p)
    assert result['value'] is None and result['status']=='NOT_ESTIMABLE' and result['reason']


def test_ct09_csv_duplicate_headers_and_malformed_rows(tmp_path):
    ctx,_=measurement_fixture(tmp_path)
    path=tmp_path/'rows.csv'
    for content in ('id,id\na,b\n','id,value\na\n','id,value\na,1,orphan\n'):
        path.write_text(content)
        ctx.manifest['artifacts']['csv']={'artifact_id':'csv','path':path.name,'sha256':file_hash(path),'origin':'fixture','parents':[]}
        with pytest.raises(ValueError):ctx.csv('csv')


def test_ct05_alarm_is_not_backdated_or_bridged_over_abstention():
    from sentinel_gl.episodes import compute_bounded_lead_time
    decisions=[('2023-01-01',1.,True),('2023-01-02',None,False),('2023-01-03',1.,True)]
    miss=compute_bounded_lead_time('fixture-event','site','2023-01-05',decisions,.5,max_gap_days=1)
    assert miss.status=='NOT_DETECTED' and miss.declaration_date is None
    detected=compute_bounded_lead_time('fixture-event','site','2023-01-05',decisions+[('2023-01-04',1.,True)],.5,max_gap_days=1)
    assert detected.declaration_date=='2023-01-04' and detected.lead_time_days==1


def test_ct05_unavailable_requested_device_cannot_be_cpu_success(tmp_path):
    from unittest.mock import patch
    from sentinel_gl.hardware import run_hardware_trials
    with patch('sentinel_gl.hardware.get_device_profile',return_value={'mps_available':False}):
        with pytest.raises(ValueError,match='requested MPS'):
            run_hardware_trials(device='mps',output_path=tmp_path/'forbidden.json',seq_len=8,n_channels=2)
    assert not list(tmp_path.iterdir())


def test_ct05_unperformed_device_comparison_remains_null(tmp_path):
    from sentinel_gl.hardware import run_hardware_trials
    report=run_hardware_trials(n_trials=1,warmup_trials=0,device='cpu',output_path=tmp_path/'cpu.json',batch_size=1,seq_len=8,n_channels=2,d_model=16)
    assert report['numerical_equivalence']['status']=='NOT_MEASURED'
    assert report['numerical_equivalence']['max_absolute_difference'] is None
    assert report['origin']=='fixture' and report['research_admissible'] is False


def test_ct08_research_figure_rendering_is_blocked(tmp_path):
    from sentinel_gl.visualization import plot_observation_cadence_and_coverage
    with pytest.raises(ValueError,match='BLOCKED'):plot_observation_cadence_and_coverage([],tmp_path/'false.png')
    assert not list(tmp_path.iterdir())


def test_ct09_rehashed_fitted_boolean_is_rejected():
    from sentinel_gl.normalization import FittedNormalizer
    state={'channel_names':list(FEATURE_CHANNELS),'channel_schema_id':DEFAULT_CHANNEL_SCHEMA.schema_id,
           'channel_schema_hash':DEFAULT_CHANNEL_SCHEMA.schema_hash,
           'mean':[0.]*NUM_CHANNELS,'scale':[1.]*NUM_CHANNELS,'constant_channels':[],
           'training_lake_ids':['fixture-site'],'temporal_cutoff':None,'input_hashes':{}}
    state['mean'][0]=True
    state['state_hash']=digest(state)
    with pytest.raises(ValueError,match='type confusion'):FittedNormalizer.from_fixture_dict(state)
