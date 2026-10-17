"""Constructed analytical integrity fixtures for baseline operators."""
from datetime import date

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from sentinel_gl.baseline_estimators import (
    CHANNEL_ORDER,
    ABSENT_SENSOR_CHANNEL_STATUS,
    DIAGNOSTIC_VERSION,
    _average_path_length,
    _feature_map,
    _score_record,
    _tree_path_expectation,
    calibrate_actual_panel_threshold,
    empirical_return_threshold,
    fit_cusum_standardizer,
    fit_gpd_tail,
    fit_isolation_forest,
    fit_psd_rbf_ocsvm,
    gpd_return_level,
    inexact_alm_rpca,
    persistent_declarations,
    rbf_kernel,
    rpca_project_score,
    run_cusum,
    validate_diagnostic_report,
    validate_sensor_channel_support,
)


FIXTURE_CHANNEL_ORDER = ["B03", "B08", "B11", "NDWI", "MNDWI",
                         "SIGMA0_VV_RAW_LINEAR", "SIGMA0_VH_RAW_LINEAR"]
FIXTURE_NATIVE_CHANNELS = {
    "SENTINEL_2_OPTICAL": FIXTURE_CHANNEL_ORDER[:5],
    "SENTINEL_1_IW_GRD": FIXTURE_CHANNEL_ORDER[5:],
}


def _row(i, values, observed, role="fit", source=None, at=None,
         sensor="SENTINEL_2_OPTICAL"):
    values = np.asarray(values, dtype=float).copy()
    observed = np.asarray(observed, dtype=bool).copy()
    if values.shape != (7,) or observed.shape != (7,):
        raise ValueError("fixture rows use the declared seven-axis computational map")
    native = FIXTURE_NATIVE_CHANNELS[sensor]
    native_indices = [FIXTURE_CHANNEL_ORDER.index(channel) for channel in native]
    absent_indices = [i for i in range(7) if i not in native_indices]
    if observed[absent_indices].any():
        raise ValueError("constructed fixture claims observed support for an absent sensor channel")
    values[absent_indices] = np.nan
    native_denominators = [1 if observed[index] else 0 for index in native_indices]
    native_unavailable = [False] * len(native)
    observed_ids = [FIXTURE_CHANNEL_ORDER[index] for index in range(7) if observed[index]]
    native_lookup = {channel: pos for pos, channel in enumerate(native)}
    membership, canonical_denominators, canonical_unavailable, channel_support = [], [], [], []
    for index, channel in enumerate(FIXTURE_CHANNEL_ORDER):
        if channel in native_lookup:
            position = native_lookup[channel]
            denominator = native_denominators[position]
            unavailable = native_unavailable[position]
            membership.append("PRESENT")
            canonical_denominators.append(denominator)
            canonical_unavailable.append(unavailable)
            channel_support.append({
                "channel_id": channel, "sensor_membership": "PRESENT",
                "source_pixel_denominator": denominator,
                "fit_transform_unavailable": unavailable,
                "score_input_observed": bool(observed[index]),
                "status": "PRESENT_IN_SENSOR_NATIVE_CHANNEL_SET",
            })
        else:
            membership.append("ABSENT_NOT_IN_SENSOR_CHANNEL_SET")
            canonical_denominators.append(None)
            canonical_unavailable.append(None)
            channel_support.append({
                "channel_id": channel, "sensor_membership": "ABSENT",
                "source_pixel_denominator": None,
                "fit_transform_unavailable": None,
                "score_input_observed": False,
                "status": ABSENT_SENSOR_CHANNEL_STATUS,
            })
    return {"row_id":f"fixture-{role}-{i}","role":role,"parity":0 if role=="fit" else 1,
        "site_id":"CONSTRUCTED_FIXTURE","panel_id":"FIXTURE_PANEL","group_id":f"g{i}",
        "source_id":source or f"source-{i}","support_id":f"support-{i}","sensor":sensor,
        "channel_order":list(native),"source_pixel_denominators":native_denominators,
        "unavailable_fit_transform_channels":native_unavailable,
        "canonical_channel_order":FIXTURE_CHANNEL_ORDER,
        "canonical_sensor_membership":membership,
        "canonical_source_pixel_denominators":canonical_denominators,
        "canonical_unavailable_fit_transform_channels":canonical_unavailable,
        "channel_support":channel_support,"observed_channel_ids":observed_ids,
        "observed_channel_denominator":len(observed_ids),
        "values":values,"observed":observed,
        "observed_channel_count":len(observed_ids),
        "acquisition_at":at,"timezone_status":"EXPLICIT_UTC" if at else "UNKNOWN",
        "arrival_at":at,"arrival_status":"FIXTURE" if at else "UNKNOWN",
        "research_admissible":False,"sampling_unit_status":"CONSTRUCTED_FIXTURE"}


def test_sensor_native_support_keeps_optical_sar_axes_and_absence_explicit():
    optical_mask=np.asarray([True,False,True,False,True,False,False])
    optical=_row(10,[1,0,2,0,3,0,0],optical_mask)
    sar_mask=np.asarray([False,False,False,False,False,True,True])
    sar=_row(11,[0,0,0,0,0,4,5],sar_mask,sensor="SENTINEL_1_IW_GRD")

    assert optical["channel_order"]==["B03","B08","B11","NDWI","MNDWI"]
    assert optical["source_pixel_denominators"]==[1,0,1,0,1]
    assert sar["channel_order"]==["SIGMA0_VV_RAW_LINEAR","SIGMA0_VH_RAW_LINEAR"]
    assert sar["source_pixel_denominators"]==[1,1]
    for row in (optical,sar):
        validate_sensor_channel_support(row,observed_mask=row["observed"])
        assert row["canonical_channel_order"]==FIXTURE_CHANNEL_ORDER
        assert len(row["canonical_sensor_membership"])==7
    assert optical["canonical_sensor_membership"][5:]==[
        "ABSENT_NOT_IN_SENSOR_CHANNEL_SET"]*2
    assert optical["canonical_source_pixel_denominators"][5:]==[None,None]
    assert optical["channel_support"][1]["status"]=="PRESENT_IN_SENSOR_NATIVE_CHANNEL_SET"
    assert optical["channel_support"][1]["source_pixel_denominator"]==0
    assert optical["channel_support"][5]["status"]==ABSENT_SENSOR_CHANNEL_STATUS
    assert optical["channel_support"][5]["score_input_observed"] is False
    score=_score_record("isolation_forest","fit",optical,0.25,"ESTIMATED",
        "EXPECTED_PATH_LENGTH_NORMALIZED_ANOMALY_SCORE",3,fit_transform_sha256="a"*64)
    validate_sensor_channel_support(score)
    assert score["source_pixel_denominators"]==[1,0,1,0,1]
    assert score["observed_channel_denominator"]==3


@pytest.mark.parametrize("mutation",["wrong_order","wrong_length","wrong_membership","negative_count"])
def test_sensor_native_support_rejects_malformed_source_axis(mutation):
    import copy
    row=_row(12,[1,2,3,4,5,0,0],[True]*5+[False]*2)
    bad=copy.deepcopy(row)
    if mutation=="wrong_order":
        bad["channel_order"][0],bad["channel_order"][1]=bad["channel_order"][1],bad["channel_order"][0]
    elif mutation=="wrong_length":
        bad["source_pixel_denominators"].pop()
    elif mutation=="wrong_membership":
        bad["channel_order"][-1]="SIGMA0_VV_RAW_LINEAR"
    else:
        bad["source_pixel_denominators"][0]=-1
    with pytest.raises(ValueError,match="sensor-native|denominators"):
        validate_sensor_channel_support(bad,observed_mask=bad["observed"])


def test_sensor_native_support_rejects_fabricated_absent_channel_evidence():
    import copy
    row=_row(13,[1,2,3,4,5,0,0],[True]*5+[False]*2)
    bad_denominator=copy.deepcopy(row)
    bad_denominator["canonical_source_pixel_denominators"][5]=1
    bad_denominator["channel_support"][5]["source_pixel_denominator"]=1
    with pytest.raises(ValueError,match="do not join exactly"):
        validate_sensor_channel_support(bad_denominator,observed_mask=bad_denominator["observed"])
    bad_observation=copy.deepcopy(row)
    bad_observation["observed"][5]=True
    bad_observation["values"][5]=99
    with pytest.raises(ValueError,match="absent sensor channel"):
        _feature_map([bad_observation])


def test_isolation_forest_complete_rows_match_reference_and_all_missing_abstains():
    matrix=np.asarray([[0,.1,.2,.3,.4,.5,.6],[.2,.1,.4,.2,.5,.3,.7],
                       [1,.8,.6,.4,.2,.1,0],[.7,.9,.4,.6,.1,.2,.5],
                       [.3,.4,.5,.6,.7,.8,.9],[.9,.7,.5,.3,.1,.2,.4]],dtype=float)
    optical_mask=np.asarray([True]*5+[False]*2)
    rows=[_row(i,x,optical_mask) for i,x in enumerate(matrix)]
    state,scores,model=fit_isolation_forest(rows,rows,"a"*64,seed=3,n_estimators=17)
    data,masks,_=_feature_map(rows)
    ours=np.asarray([x["score"] for x in scores[:len(rows)]])
    reference=-model.score_samples(data)
    assert np.allclose(ours,reference,rtol=1e-12,atol=1e-12)
    assert state["status"]=="FITTED_DEVELOPMENT_DIAGNOSTIC"
    missing=_row(99,[np.nan]*7,[False]*7)
    _,missing_scores,_=fit_isolation_forest(rows,[missing],"a"*64,seed=3,n_estimators=5)
    assert missing_scores[-1]["score"] is None
    assert missing_scores[-1]["status"]=="ABSTAIN_ALL_FEATURES_MISSING"


def test_isolation_missing_route_matches_independent_branch_enumeration():
    rng=np.random.default_rng(4)
    x=rng.normal(size=(20,14))
    model=IsolationForest(n_estimators=1,max_samples=20,random_state=8,n_jobs=1).fit(x)
    est=model.estimators_[0]; fmap=model.estimators_features_[0]
    row=x[3].copy(); row[0]=0.0
    observed=np.asarray([False,True,True,True,True,False,False],dtype=bool)
    got=_tree_path_expectation(est,fmap,row,observed)
    tree=est.tree_
    def enumerate_paths(node,depth,weight):
        left=int(tree.children_left[node]); right=int(tree.children_right[node])
        if left==right:
            return [(weight,depth+_average_path_length(int(tree.n_node_samples[node])))]
        global_feature=int(fmap[int(tree.feature[node])])
        if global_feature==0:
            ln=int(tree.n_node_samples[left]); rn=int(tree.n_node_samples[right])
            total=ln+rn
            return (enumerate_paths(left,depth+1,weight*ln/total)+
                    enumerate_paths(right,depth+1,weight*rn/total))
        child=left if row[global_feature]<=tree.threshold[node] else right
        return enumerate_paths(child,depth+1,weight)
    independent=sum(w*p for w,p in enumerate_paths(0,0,1.0))
    assert got==pytest.approx(independent,abs=1e-12)


def test_ocsvm_uses_fixed_missingness_map_and_psd_rbf_kernel():
    values=np.asarray([[0,1,0,2,0,1,0],[1,0,1,0,2,0,1],[2,1,0,1,0,2,0],
                       [1,2,1,0,1,0,2],[0,0,2,1,2,1,0]],dtype=float)
    mask=np.zeros_like(values,dtype=bool)
    mask[:3,:5]=True
    mask[1,2]=False
    mask[3:,5:]=True
    rows=[_row(i,np.where(mask[i],values[i],np.nan),mask[i],
        sensor="SENTINEL_2_OPTICAL" if i<3 else "SENTINEL_1_IW_GRD")
        for i in range(len(values))]
    state,scores,model=fit_psd_rbf_ocsvm(rows,rows,"b"*64)
    matrix,_,_=_feature_map(rows)
    gram=rbf_kernel(matrix,state["gamma"])
    assert np.linalg.eigvalsh(gram).min()>=-1e-10
    assert state["feature_map_version"]=="FROZEN_WP06_ZSCORES_ZERO_SENTINEL_PLUS_OBSERVED_BITS_V1"
    assert all(s["score"] is not None for s in scores)
    altered=rows[1]["values"].copy(); altered[2]=float("nan")
    assert np.isfinite(model.decision_function(np.vstack([matrix[1]]))).all()
    with pytest.raises(ValueError): rbf_kernel(matrix,-1)
    bad=_row(8,[np.inf,0,0,0,0,0,0],[True]*5+[False]*2)
    with pytest.raises(ValueError,match="observed feature values must be finite"):
        fit_psd_rbf_ocsvm(rows+[bad],rows,"b"*64)


def test_masked_rpca_ignores_unobserved_values_and_logs_nonconvergence():
    x=np.asarray([[1.,2.,np.nan],[1.1,1.9,np.nan],[.9,2.1,1000.],[1.2,1.8,-1000.]])
    mask=np.isfinite(x)
    a=inexact_alm_rpca(x,mask,max_iterations=3,tolerance=1e-12)
    changed=x.copy(); changed[~mask]=9e99
    b=inexact_alm_rpca(changed,mask,max_iterations=3,tolerance=1e-12)
    assert a["algorithm"]=="OBSERVED_ENTRY_MASKED_INEXACT_ALM_ROBUST_COMPLETION"
    assert np.array_equal(a["low_rank"],b["low_rank"])
    assert np.array_equal(a["sparse"],b["sparse"])
    assert a["missing_entry_count"]==int((~mask).sum())
    assert a["status"]=="NONCONVERGED_MAX_ITERATIONS"
    complete=np.where(mask,x,0.)
    ordinary=inexact_alm_rpca(complete,np.ones_like(mask),max_iterations=2)
    assert ordinary["algorithm"]=="COMPLETE_DATA_INEXACT_ALM_ROBUST_PCA"
    projection_state={**a,"low_rank":np.asarray([[1.,0.,0.],[0.,1.,0.],[1.,1.,0.],[0.,0.,0.]]),
                      "rank":2,"status":"CONVERGED"}
    projected=rpca_project_score(projection_state,np.asarray([[1.,2.,np.nan]]),np.asarray([[True,True,False]]))
    assert projected["scores"][0]["score"] is None
    assert projected["scores"][0]["status"]=="ABSTAIN_INSUFFICIENT_OBSERVED_FEATURES_FOR_PROJECTION"


def test_gpd_return_level_zero_shape_negative_endpoint_scale_and_failures():
    assert gpd_return_level(10.,0.,2.,4.,1.)==pytest.approx(10+2*np.log(4))
    negative=gpd_return_level(3.,-.2,1.,4.,1.)
    assert 3<negative<3+1/.2
    with pytest.raises(ValueError,match="scale and rates"):
        gpd_return_level(0.,0.,0.,1.,1.)
    fit=fit_gpd_tail([2,3,4,5],1.,4.,independent_unit_ids=["a","b","c","d"],
        role="threshold_calibration",stationarity_reviewed=True,declustering_reviewed=True,
        target_score_exceedance_rate_per_lake_year=1.0)
    assert fit["status"]=="ESTIMATED_CALIBRATION_DIAGNOSTIC"
    assert fit["target_score_exceedance_rate_per_lake_year"]==1.0
    assert "target_rate_per_lake_year" not in fit
    assert fit["empirical_score_exceedance_rate_per_lake_year"]==pytest.approx(1.0)
    assert fit["achieved_false_alarm_episode_rate_per_lake_year"] is None
    assert fit["episode_rate_status"]=="NOT_ESTIMATED_NO_ORDERED_ARRIVAL_AND_EPISODE_ACCOUNTING"
    ties=fit_gpd_tail([1,1,2],1.,3.,independent_unit_ids=["a","b","c"],
        role="threshold_calibration",stationarity_reviewed=True,declustering_reviewed=True)
    assert ties["status"]=="INSUFFICIENT_OR_INVALID_TAIL_EXCEEDANCES"
    with pytest.raises(ValueError,match="disjoint"):
        fit_gpd_tail([2,3,4],1.,3.,independent_unit_ids=["a","b","c"],
            role="fit",stationarity_reviewed=True,declustering_reviewed=True)


def test_empirical_quantile_ties_and_actual_panel_calibration_are_explicit():
    result=empirical_return_threshold([0.,1.,1.,2.],4.,
        target_score_exceedance_rate_per_lake_year=.5,
        independent_unit_ids=["a","b","c","d"],role="threshold_calibration")
    assert result["threshold"]==1.
    assert result["quantile_method"].startswith("higher")
    assert result["empirical_score_exceedance_count"]==3
    assert result["empirical_score_exceedance_rate_per_lake_year"]==pytest.approx(.75)
    assert result["achieved_false_alarm_episode_rate_per_lake_year"] is None
    tied=empirical_return_threshold([1.,1.,1.,1.],1.,
        target_score_exceedance_rate_per_lake_year=.25,
        independent_unit_ids=["t1","t2","t3","t4"],role="threshold_calibration")
    assert tied["threshold"]==1.
    assert tied["empirical_score_exceedance_count"]==4
    assert tied["empirical_score_exceedance_rate_per_lake_year"]==4.
    assert tied["achieved_false_alarm_episode_rate_per_lake_year"] is None
    assert tied["episode_rate_status"]=="NOT_ESTIMATED_NO_ORDERED_ARRIVAL_AND_EPISODE_ACCOUNTING"
    actual=calibrate_actual_panel_threshold(calibration_rows=[],exposure_lake_years=None,
        stationarity_reviewed=False,declustering_reviewed=False)
    assert actual["status"]=="INSUFFICIENT_CALIBRATION_SUPPORT"
    assert actual["threshold"] is actual["gpd_shape"] is actual["gpd_scale"] is None
    assert actual["achieved_false_alarm_episode_rate_per_lake_year"] is None


def test_report_validator_requires_explicit_null_episode_rate_semantics():
    report={"schema_version":2,"diagnostic_version":DIAGNOSTIC_VERSION,
        "status":"DEVELOPMENT_DIAGNOSTICS_COMPLETE_RESEARCH_ADMISSION_BLOCKED",
        "research_admissible":False,"plan_executable":False,"release_status":"BLOCKED",
        "fit_transform_sha256":"a"*64,"input_sha256":{},"application_receipts":[],
        "row_summary":{},"method_states":{},"method_state_hashes":{},"scores":[],
        "comparison_support":{},"cusum":{},"calibration":{
            "objective":"1 false alarm per lake-year","status":"INSUFFICIENT_CALIBRATION_SUPPORT",
            "threshold":None,"gpd_shape":None,"gpd_scale":None,
            "empirical_quantile_threshold":None,"reason":"fixture unavailable",
            "calibration_row_count":0,"exposure_lake_years":None,
            "seed_pixel_quadrant_or_monitor_rows_used":False},
        "persistence":{"actual":{"status":"NOT_RUN_INVALID_CALIBRATION_THRESHOLD"}},
        "tmae":{},"limits":["constructed validator fixture"]}
    with pytest.raises(ValueError,match="required null/status fields"):
        validate_diagnostic_report(report)
    report["calibration"]["achieved_false_alarm_episode_rate_per_lake_year"]=None
    validate_diagnostic_report(report)
    report["calibration"]["achieved_false_alarm_episode_rate_per_lake_year"]=0.0
    with pytest.raises(ValueError,match="must remain explicitly unavailable"):
        validate_diagnostic_report(report)


def test_cusum_fits_only_fit_scores_and_requires_actual_arrivals():
    state=fit_cusum_standardizer([0.,1.,2.],"fit",["f1","f2","f3"])
    assert state["status"]=="FITTED_DEVELOPMENT_DIAGNOSTIC"
    with pytest.raises(ValueError,match="parity-0"):
        fit_cusum_standardizer([1.,2.],"monitor",["m1","m2"])
    missing=run_cusum([1.,2.],[None,"2023-09-04"],["m1","m2"],state)
    assert missing["status"]=="UNAVAILABLE_ACTUAL_ARRIVAL_TIME_MISSING"
    good=run_cusum([0.,1.,None,2.],["2023-09-01","2023-09-02","2023-09-03","2023-10-20"],
        ["a","b","c","d"],state,max_gap_days=10)
    assert good["rows"][2]["status"]=="ABSTAIN_RESET_MISSING_SCORE"
    assert good["rows"][3]["gap_days"]==47 and good["rows"][3]["positive_cusum"]>0
    with pytest.raises(ValueError,match="strictly increasing"):
        run_cusum([0.,1.],["2023-09-01","2023-09-01"],["a","b"],state)


def test_k_persistence_declares_on_kth_arrival_without_backdating():
    rows=[{"arrival_id":"a1","source_id":"scene1","arrival_at":"2023-01-01","score":2.,"eligible":True},
          {"arrival_id":"a2","source_id":"scene2","arrival_at":"2023-01-02","score":3.,"eligible":True},
          {"arrival_id":"a3","source_id":"scene3","arrival_at":"2023-01-03","score":4.,"eligible":True},
          {"arrival_id":"a4","source_id":"scene4","arrival_at":"2023-01-04","score":5.,"eligible":True}]
    result=persistent_declarations(rows,1.,k=2)
    assert len(result["declarations"])==1
    assert result["declarations"][0]["declared_at"]=="2023-01-02"
    assert result["declarations"][0]["episode_opened_at"]=="2023-01-02"
    assert result["declarations"][0]["declared_at_resolution"]=="DAY"
    assert result["declarations"][0]["continuation_arrival_ids"]==["a3","a4"]
    assert result["declarations"][0]["backdated"] is False
    assert result["episode_closures"]==[]
    assert persistent_declarations(rows,None,k=2)["status"]=="NOT_RUN_INVALID_CALIBRATION_THRESHOLD"
    duplicate=[dict(rows[0]),dict(rows[1],source_id="scene1")]
    with pytest.raises(ValueError,match="duplicate source"):
        persistent_declarations(duplicate,1.,k=2)
    with pytest.raises(ValueError,match="k>=2"):
        persistent_declarations(rows,1.,k=1)


def test_persistence_recovery_closes_then_reopens_only_after_new_k_arrivals():
    rows=[
        {"arrival_id":"r1","source_id":"rs1","arrival_at":"2023-02-01","score":2.,"eligible":True},
        {"arrival_id":"r2","source_id":"rs2","arrival_at":"2023-02-02","score":2.,"eligible":True},
        {"arrival_id":"r3","source_id":"rs3","arrival_at":"2023-02-03","score":3.,"eligible":True},
        {"arrival_id":"r4","source_id":"rs4","arrival_at":"2023-02-04","score":0.,"eligible":True},
        {"arrival_id":"r5","source_id":"rs5","arrival_at":"2023-02-05","score":2.,"eligible":True},
        {"arrival_id":"r6","source_id":"rs6","arrival_at":"2023-02-06","score":2.,"eligible":True},
    ]
    result=persistent_declarations(rows,1.,k=2)
    assert [item["declared_at"] for item in result["declarations"]]==["2023-02-02","2023-02-06"]
    assert [item["episode_id"] for item in result["declarations"]]==["EPISODE-0001","EPISODE-0002"]
    assert result["episode_closures"][0]["closure_reason"]=="ELIGIBLE_BELOW_THRESHOLD_RECOVERY"
    assert result["episode_closures"][0]["closed_at"]=="2023-02-04"
    assert result["declarations"][0]["backdated"] is False


def test_persistence_missing_gap_and_refractory_close_at_declared_policy_boundary():
    rows=[
        {"arrival_id":"m1","source_id":"ms1","arrival_at":"2023-03-01","score":2.,"eligible":True},
        {"arrival_id":"m2","source_id":"ms2","arrival_at":"2023-03-02","score":2.,"eligible":True},
        {"arrival_id":"m3","source_id":"ms3","arrival_at":"2023-03-04","score":None,"eligible":False},
        {"arrival_id":"m4","source_id":"ms4","arrival_at":"2023-03-05","score":2.,"eligible":True},
        {"arrival_id":"m5","source_id":"ms5","arrival_at":"2023-03-06","score":2.,"eligible":True},
        {"arrival_id":"m6","source_id":"ms6","arrival_at":"2023-03-07","score":2.,"eligible":True},
        {"arrival_id":"m7","source_id":"ms7","arrival_at":"2023-03-08","score":2.,"eligible":True},
        {"arrival_id":"m8","source_id":"ms8","arrival_at":"2023-03-30","score":2.,"eligible":True},
    ]
    result=persistent_declarations(rows,1.,k=2,max_gap_days=5,refractory_days=3)
    assert [item["declared_at"] for item in result["declarations"]]==["2023-03-02","2023-03-08"]
    assert [item["closure_reason"] for item in result["episode_closures"]]==[
        "INELIGIBLE_OR_MISSING_SCORE_ARRIVAL","MAX_GAP_RESET_AT_CURRENT_ARRIVAL"]
    assert result["episode_closures"][0]["closed_at"]=="2023-03-04"
    assert result["episode_closures"][0]["refractory_until"]=="2023-03-07"
    assert "REFRACTORY_SUPPRESSED_FROM_EPISODE_CLOSURE" in result["arrival_dispositions"][3]["operations"]
    assert "ACTIVE_EPISODE_CLOSED_BY_MAX_GAP_RESET" in result["arrival_dispositions"][-1]["operations"]
    assert "REFRACTORY_SUPPRESSED_FROM_EPISODE_CLOSURE" in result["arrival_dispositions"][-1]["operations"]


def test_persistence_preserves_instant_timestamp_resolution_and_offset_string():
    first="2023-04-01T12:00:00.125Z"
    second="2023-04-01T13:00:01.250+01:00"
    rows=[{"arrival_id":"p1","source_id":"ps1","arrival_at":first,"score":2.,"eligible":True},
          {"arrival_id":"p2","source_id":"ps2","arrival_at":second,"score":3.,"eligible":True}]
    result=persistent_declarations(rows,1.,k=2)
    assert result["declarations"][0]["declared_at"]==second
    assert result["declarations"][0]["episode_opened_at"]==second
    assert result["declarations"][0]["declared_at_resolution"]=="INSTANT_FRACTIONAL_SECOND_3_DIGITS"


def test_persistence_abstention_gap_and_refractory_reset_streaks():
    rows=[{"arrival_id":"a1","source_id":"s1","arrival_at":"2023-01-01","score":2.,"eligible":True},
          {"arrival_id":"a2","source_id":"s2","arrival_at":"2023-01-02","score":None,"eligible":False},
          {"arrival_id":"a3","source_id":"s3","arrival_at":"2023-01-03","score":2.,"eligible":True},
          {"arrival_id":"a4","source_id":"s4","arrival_at":"2023-01-20","score":2.,"eligible":True},
          {"arrival_id":"a5","source_id":"s5","arrival_at":"2023-01-21","score":2.,"eligible":True}]
    result=persistent_declarations(rows,1.,k=2,max_gap_days=5,refractory_days=10)
    assert [x["declared_at"] for x in result["declarations"]]==["2023-01-21"]
    assert result["declarations"][0]["sustaining_arrival_ids"]==["a4","a5"]
