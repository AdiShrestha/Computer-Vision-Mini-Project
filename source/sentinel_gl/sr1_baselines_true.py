"""Leakage-controlled descriptive baselines for the amended SR-1 execution."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


def _observed_vector(row: dict, patch: int):
    value=np.asarray(row['std_features'][patch],dtype=np.float64)
    mask=np.asarray(row['observed'][patch],dtype=bool)
    if not mask.all() or not np.isfinite(value).all():return None
    return value


def _fit_population(records: list[dict], fit_ids: set[str]):
    population=[]
    for row in records:
        if row.get('observation_id') not in fit_ids:continue
        if row.get('role')!='FIT':raise ValueError('baseline fitting population includes a non-FIT row')
        for patch,zone in enumerate(row['zones']):
            if zone not in (0,1):continue
            vector=_observed_vector(row,patch)
            if vector is not None:population.append((row,patch,vector))
    return population


def _soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x)*np.maximum(np.abs(x)-tau,0.0)


def _svd_threshold(x: np.ndarray, tau: float):
    u,s,vt=np.linalg.svd(x,full_matrices=False)
    keep=s>tau
    return (u[:,keep]*s[keep])@vt[keep] if np.any(keep) else np.zeros_like(x),s


def _rpca_low_rank_basis(matrix: np.ndarray, tolerance=1e-7, max_iterations=1000):
    d=np.asarray(matrix,dtype=np.float64)
    if d.ndim!=2 or min(d.shape)<2 or not np.isfinite(d).all():
        return None,{'status':'NOT_ESTIMABLE_MALFORMED_OR_SMALL_FIT_MATRIX','iterations':0,'relative_residual':None}
    norm2=float(np.linalg.norm(d,2));normf=float(np.linalg.norm(d,'fro'))
    if not math.isfinite(norm2) or norm2<=np.finfo(float).eps or normf<=np.finfo(float).eps:
        return None,{'status':'NOT_ESTIMABLE_DEGENERATE_FIT_MATRIX','iterations':0,'relative_residual':None}
    lam=1.0/math.sqrt(max(d.shape));mu=1.25/norm2;mu_max=mu*1e7;rho=1.5
    y=d/max(float(np.linalg.norm(d,2)),float(np.max(np.abs(d)))/lam)
    low=np.zeros_like(d);sparse=np.zeros_like(d);residual=math.inf
    for iteration in range(1,max_iterations+1):
        low,_=_svd_threshold(d-sparse+y/mu,1/mu)
        sparse=_soft_threshold(d-low+y/mu,lam/mu)
        error=d-low-sparse
        residual=float(np.linalg.norm(error,'fro')/normf)
        y=y+mu*error;mu=min(mu*rho,mu_max)
        if residual<tolerance:break
    if residual>=tolerance:
        return None,{'status':'NOT_ESTIMABLE_RPCA_DID_NOT_CONVERGE','iterations':iteration,'relative_residual':residual,
                     'lambda':lam,'tolerance':tolerance,'max_iterations':max_iterations}
    _,singular,vt=np.linalg.svd(low,full_matrices=False)
    threshold=max(low.shape)*np.finfo(float).eps*(singular[0] if singular.size else 0.0)
    rank=int(np.sum(singular>threshold))
    if rank<1:
        return None,{'status':'NOT_ESTIMABLE_ZERO_RANK_LOW_COMPONENT','iterations':iteration,'relative_residual':residual,
                     'rank':rank,'lambda':lam}
    return vt[:rank],{'status':'FIT_ONLY_RPCA_SUBSPACE_ESTIMATED','iterations':iteration,
                       'relative_residual':residual,'rank':rank,'lambda':lam,
                       'tolerance':tolerance,'max_iterations':max_iterations}


def fit_descriptive_baselines(records: list[dict], fit_observation_ids: list[str], *, random_state=0):
    by_id={r.get('observation_id'):r for r in records}
    fit_ids=set(fit_observation_ids)
    if len(by_id)!=len(records) or None in by_id or not fit_ids:
        raise ValueError('baseline inputs require unique records and explicit FIT IDs')
    if fit_ids-set(by_id):raise ValueError('baseline FIT IDs are absent')
    if any(by_id[oid].get('role')!='FIT' for oid in fit_ids):
        raise ValueError('baseline fitted state may use FIT only records')
    pop=_fit_population(records,fit_ids)
    x=np.stack([item[2] for item in pop]) if pop else np.empty((0,5),dtype=np.float64)
    selected=np.arange(len(x))
    cap=4000
    if len(x)>cap:selected=np.linspace(0,len(x)-1,cap,dtype=np.int64)
    train=x[selected]
    result={'fit_observation_ids':sorted(fit_ids),'fit_token_count':int(len(x)),
            'detector_fit_token_count':int(len(train)),'detector_token_cap':cap,
            'detector_token_selection':'deterministic evenly spaced indices over FIT site/chronology/patch order',
            'isolation_forest':None,'one_class_svm':None,'rpca_basis':None,'rpca':None}
    if len(train)<10:
        result['isolation_forest_status']='NOT_ESTIMABLE_LT_10_COMPLETE_FIT_TARGET_TOKENS'
        result['one_class_svm_status']='NOT_ESTIMABLE_LT_10_COMPLETE_FIT_TARGET_TOKENS'
    else:
        iso=IsolationForest(n_estimators=100,contamination=0.05,random_state=random_state,n_jobs=1)
        iso.fit(train);result['isolation_forest']=iso
        result['isolation_forest_status']='FIT_ONLY_COMPLETE_ZONE_0_1_TOKENS'
        svm=OneClassSVM(kernel='rbf',gamma='scale',nu=0.05)
        svm.fit(train);result['one_class_svm']=svm
        result['one_class_svm_status']='FIT_ONLY_COMPLETE_ZONE_0_1_TOKENS'
    if len(x)<10:
        result['rpca_status']={'status':'NOT_ESTIMABLE_LT_10_COMPLETE_FIT_TARGET_TOKENS','iterations':0}
    else:
        basis,status=_rpca_low_rank_basis(x)
        result['rpca_basis']=basis;result['rpca_status']=status
    result['status']='BASELINE_FITS_COMPLETED_WITH_EXPLICIT_STATUS'
    return result


def score_descriptive_baselines(records: list[dict], fit_observation_ids: list[str], fitted: dict):
    fit_ids=set(fit_observation_ids)
    fit_rows=[r for r in records if r.get('observation_id') in fit_ids]
    fit_doy={r['observation_id']:r['dt'].timetuple().tm_yday for r in fit_rows}
    ordered=sorted(records,key=lambda r:(r['site_id'],r['acquired_at']))
    latest={}; output=[]
    iso,svm=fitted.get('isolation_forest'),fitted.get('one_class_svm')
    basis=fitted.get('rpca_basis')
    for row in ordered:
        patches=np.flatnonzero(np.isin(row['zones'],[0,1]))
        season,persistence,rpca_scores=[],[],[]
        target_vectors=[]
        doy=row['dt'].timetuple().tm_yday
        for patch in patches:
            vector=_observed_vector(row,int(patch))
            if vector is None:continue
            target_vectors.append(vector)
            refs=[]
            for ref in fit_rows:
                if ref['site_id']!=row['site_id'] or ref['observation_id']==row['observation_id']:
                    continue
                other=_observed_vector(ref,int(patch))
                if other is None:continue
                dd=abs(doy-fit_doy[ref['observation_id']])
                if min(dd,366-dd)<=45:refs.append(float(np.linalg.norm(vector-other)))
            if len(refs)>=5:season.append(float(np.median(np.sort(refs)[:5])))
            previous=latest.get((row['site_id'],int(patch)))
            if previous is not None:persistence.append(float(np.linalg.norm(vector-previous)))
            if basis is not None:rpca_scores.append(float(np.linalg.norm(vector-(vector@basis.T)@basis)))
        if target_vectors and iso is not None:
            iso_scores=(-iso.decision_function(np.stack(target_vectors).astype(np.float32))).tolist()
        else:iso_scores=[]
        if target_vectors and svm is not None:
            svm_scores=(-svm.decision_function(np.stack(target_vectors).astype(np.float32))).tolist()
        else:svm_scores=[]
        # State is updated only after the current scene has been scored.
        for patch in patches:
            vector=_observed_vector(row,int(patch))
            if vector is not None:latest[(row['site_id'],int(patch))]=vector.copy()
        def mean_or_none(values):return float(np.mean(values)) if values else None
        output.append({'observation_id':row['observation_id'],'site_id':row['site_id'],'role':row['role'],
            'seasonal_climatology_residual':mean_or_none(season),
            'seasonal_status':'ESTIMATED' if season else 'NOT_ESTIMABLE_LT_5_FIT_MATCHES_PER_PATCH',
            'spatial_persistence_residual':mean_or_none(persistence),
            'persistence_status':'ESTIMATED' if persistence else 'NOT_ESTIMABLE_NO_PRIOR_COMPLETE_PATCH',
            'isolation_forest_score':mean_or_none(iso_scores),
            'isolation_forest_status':fitted['isolation_forest_status'] if iso is not None else fitted['isolation_forest_status'],
            'one_class_svm_score':mean_or_none(svm_scores),
            'one_class_svm_status':fitted['one_class_svm_status'] if svm is not None else fitted['one_class_svm_status'],
            'rpca_score':mean_or_none(rpca_scores),
            'rpca_status':fitted['rpca_status']['status'] if isinstance(fitted['rpca_status'],dict) else fitted['rpca_status'],
            'supported_target_patch_count':len(patches)})
    return output


def add_page_cusum(predictions: list[dict], *, kappa=0.5, threshold=4.0):
    state={}
    for row in sorted(predictions,key=lambda r:(r['seed'],r['site_id'],r['acquired_at'])):
        key=(row['seed'],row['site_id']);score=row.get('score_c')
        value=state.get(key,0.0)
        if score is not None and math.isfinite(float(score)):
            value=max(0.0,value+(float(score)-kappa));status='UPDATED_EXPLORATORY_CUSUM'
        else:status='NOT_UPDATED_NO_FINITE_SCORE_C'
        fired=bool(value>=threshold)
        row['page_cusum']={'statistic':value,'alarm_declared':fired,
            'status':status if not fired else status+'_THRESHOLD_CROSSED',
            'kappa':kappa,'threshold':threshold,
            'calibration_status':'EXPLORATORY_FIXED_RULE_NOT_OPERATIONALLY_CALIBRATED'}
        state[key]=value
    return predictions
