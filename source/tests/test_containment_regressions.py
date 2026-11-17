"""Independent analytical regressions; all inputs are constructed, never provider observations.
# FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
"""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import json
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pytest
from admission_fixtures import measurement_fixture
from sentinel_gl.admission import fit_fixture_operators
from sentinel_gl.features import FeatureExtractionPipeline, MultiModalPanel, NUM_CHANNELS
from sentinel_gl.episodes import AlertEpisodeEngine, compute_alert_burden_estimand
from sentinel_gl.metrics import first_sustained_alarm
from sentinel_gl.inference import cluster_conditional_metrics, poisson_rate_confidence_interval
from sentinel_gl.integrity import digest, file_hash
from sentinel_gl.replay import CheckpointBundle


def test_episode_and_lead_time_share_actual_declaration():
    rows=[('2023-01-01',9.,True),('2023-01-02',None,False),('2023-01-03',1.,True)]
    engine=AlertEpisodeEngine(.5, sustained_q=2, max_gap_days=1)
    assert engine.extract_episodes('fixture',rows)==[]
    rows.append(('2023-01-04',2.,True))
    episode,=engine.extract_episodes('fixture',rows)
    lead=first_sustained_alarm([r[1] for r in rows],[r[0] for r in rows],.5,'2023-01-05',2,10,eligible=[r[2] for r in rows],max_gap_days=1)
    assert episode.start_date=='2023-01-03' and episode.declaration_date==lead['alarm_date']=='2023-01-04'
    assert episode.peak_score==2. and episode.qualifying_decisions_count==2
    with pytest.raises(ValueError,match='ordered'):
        engine.extract_episodes('fixture',rows+rows[-1:])


def test_active_episode_does_not_bridge_long_missing_support():
    engine=AlertEpisodeEngine(.5,sustained_q=2,max_gap_days=2)
    rows=[('2023-01-01',1.,True),('2023-01-02',2.,True),('2023-01-10',3.,True),('2023-01-11',4.,True)]
    episodes=engine.extract_episodes('fixture',rows)
    assert [(e.declaration_date,e.end_date,e.peak_score) for e in episodes]==[('2023-01-02','2023-01-02',2.),('2023-01-11','2023-01-11',4.)]


def test_episode_requires_exposure_at_its_own_site():
    episode,=AlertEpisodeEngine(.5,sustained_q=1).extract_episodes('fixture',[('2023-01-01',1.,True)])
    with pytest.raises(ValueError,match='own site'):
        compute_alert_burden_estimand([episode],{'fixture':0.,'other':1.})
    assert compute_alert_burden_estimand([],{'fixture':1.})['lambda_alert']==0.
    assert compute_alert_burden_estimand([],{'fixture':0.})['lambda_alert'] is None


@pytest.mark.parametrize('support_start,support_end,expected_day',[
    ('2023-01-01T23:15:00-01:00','2023-01-01T23:30:00-01:00','2023-01-02'),
    ('2023-01-02T00:15:00+01:00','2023-01-02T00:30:00+01:00','2023-01-01')])
def test_admitted_support_uses_UTC_calendar(tmp_path,support_start,support_end,expected_day):
    ctx,r=measurement_fixture(tmp_path)
    r.update(support_start=support_start,support_end=support_end,
             available_at=expected_day+'T23:45:00Z')
    metadata=ctx.artifact(r['metadata_id'])
    metadata.write_text(json.dumps({k:r[k] for k in ('product','version','endpoint','units','crs','support_start','support_end','available_at','availability_policy')}))
    ctx.manifest['artifacts'][r['metadata_id']]['sha256']=file_hash(metadata)
    ctx.manifest['measurements'][r['measurement_id']]=digest(r)
    p=FeatureExtractionPipeline(2,admission=ctx).extract_panel(r['lake_id'],'fixture','2023-01-01',optical_observations=[r])
    assert [p.dates[i] for i in np.flatnonzero(p.mask.any(axis=1))]==[expected_day]


def test_actual_panel_identity_is_joined_before_fit():
    roles=('fitting','development_validation','threshold_calibration','evaluation')
    scopes={r:[r] for r in roles}
    membership={r:{'scope':r,'site_id':'fixture-'+r,'available_at':'2023-06-30T00:00:00Z','cutoff':'2023-07-01T00:00:00Z'} for r in roles}
    dates=tuple((date(2023,1,1)+timedelta(days=i)).isoformat() for i in range(6))
    values=np.arange(len(dates)*NUM_CHANNELS,dtype=float).reshape(len(dates),NUM_CHANNELS)
    statuses=np.full(values.shape,'OBSERVED',dtype=object)
    p=MultiModalPanel('fixture-fitting','evaluation',dates[0],dates[-1],dates,values,
        np.ones_like(values,dtype=bool),None,origin='fixture',statuses=statuses)
    with pytest.raises(ValueError,match='window identity'):
        fit_fixture_operators({'fitting':p},{'fitting':np.array([[1.,2.],[3.,4.]])},{'threshold_calibration':.2},scopes,membership,[])


@pytest.mark.parametrize('membership',[{}, {'A':'known'}, {'A':'known','B':''}])
def test_unknown_cluster_membership_cannot_establish_independence(membership):
    result=cluster_conditional_metrics({'A':{'score':0.},'B':{'score':1.}},membership)
    assert result['status']=='NOT_ESTIMABLE' and result['n_clusters'] is None
    assert result['reason']=='UNKNOWN_CLUSTER_MEMBERSHIP'


def test_numeric_evidence_rejects_booleans():
    with pytest.raises(ValueError):poisson_rate_confidence_interval(0,True)
    with pytest.raises(ValueError):poisson_rate_confidence_interval(0,1.,alpha=True)
    with pytest.raises(ValueError):cluster_conditional_metrics({'A':{'score':True}},{'A':'cluster'})


def test_checkpoint_save_preserves_existing_attempt_and_symlink_target(tmp_path):
    bundle=CheckpointBundle(4,{}, {}, {}, {}, {}, [])
    existing=tmp_path/'existing.pt';existing.write_bytes(b'constructed prior attempt')
    with pytest.raises(FileExistsError):bundle.save(existing)
    assert existing.read_bytes()==b'constructed prior attempt'
    link=tmp_path/'link.pt';link.symlink_to(existing)
    with pytest.raises(ValueError,match='symlink'):bundle.save(link)
    assert existing.read_bytes()==b'constructed prior attempt'
    fresh=tmp_path/'fresh.pt';bundle.save(fresh)
    assert CheckpointBundle.load_fixture(fresh).format_version==4


def test_blocked_reproduction_cli_reports_the_actual_state(tmp_path):
    root=Path(__file__).resolve().parents[2]
    output=tmp_path/'blocked.json'
    result=subprocess.run([sys.executable,'-B',str(root/'tools/verify_reproduction.py'),'--output',str(output)],capture_output=True,text=True)
    assert result.returncode==1 and 'Traceback' not in result.stderr
    assert 'BLOCKED' in result.stdout and json.loads(output.read_text())['overall_status']=='BLOCKED'


def test_hardware_module_cli_can_render_unmeasured_comparison(capsys):
    from sentinel_gl.hardware import main
    report={'device_profile':{'os_platform':'fixture','arch':'fixture','chip_family':None,'cpu_count':1,'total_ram_bytes':None,'pytorch_version':'fixture','mps_available':False},'summary':{},'numerical_equivalence':{'status':'NOT_MEASURED','max_absolute_difference':None,'tolerance':1e-5},'thermal_disclosure':'unmeasured'}
    with patch('sys.argv',['hardware','--device','cpu','--sequence-length','8','--channels','2']),patch('sentinel_gl.hardware.run_hardware_trials',return_value=report):
        assert main()==0
    assert 'unmeasured' in capsys.readouterr().out


def test_candidates_and_scenarios_cannot_claim_observed_origin():
    from sentinel_gl.cohort import get_expanded_lake_registry,get_expanded_event_registry
    from sentinel_gl.insar import model_insar_coherence,classify_geometric_distortion,check_insar_storage_budget
    for getter in (get_expanded_lake_registry,get_expanded_event_registry):
        with pytest.raises(ValueError,match='BLOCKED'):getter()
        assert all(r['origin']=='unverified' and r['research_admissible'] is False for r in getter(include_unverified=True))
    assert all(r['evidence_tier'] is None and r['onset_confidence'] is None for r in get_expanded_event_registry(include_unverified=True))
    for result in (model_insar_coherence(12),classify_geometric_distortion(10,30),check_insar_storage_budget(2,3)):
        assert result['origin']=='simulation' and result['research_admissible'] is False
    with pytest.raises(ValueError,match='unknown assumed season'):model_insar_coherence(12,season='unknown')
