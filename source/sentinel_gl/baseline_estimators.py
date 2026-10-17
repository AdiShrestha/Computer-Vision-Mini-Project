"""Missingness-aware baseline estimators and explicit calibration limits."""
from __future__ import annotations
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import hashlib, json, math
from typing import Any, Mapping, Sequence
import numpy as np
from scipy.stats import genpareto
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

DIAGNOSTIC_VERSION = "WP07_WP08_DEVELOPMENT_DIAGNOSTIC_V2"


FEATURE_MAP_VERSION = "FROZEN_WP06_ZSCORES_ZERO_SENTINEL_PLUS_OBSERVED_BITS_V1"


CHANNEL_ORDER = ("B03", "B08", "B11", "NDWI", "MNDWI",
                 "SIGMA0_VV_RAW_LINEAR", "SIGMA0_VH_RAW_LINEAR")


SENSOR_NATIVE_CHANNELS = {
    "SENTINEL_2_OPTICAL": CHANNEL_ORDER[:5],
    "SENTINEL_1_IW_GRD": CHANNEL_ORDER[5:],
}


ABSENT_SENSOR_CHANNEL_STATUS = "UNAVAILABLE_NOT_IN_SENSOR_CHANNEL_SET"


def _sensor_native_support_fields(sensor: str, channel_order: Sequence[str],
                                 source_pixel_denominators: Sequence[int],
                                 unavailable_fit_transform_channels: Sequence[bool],
                                 observed_mask: np.ndarray) -> dict:
    """Bind sensor-native source support to the separate seven-axis model map."""
    expected_native = SENSOR_NATIVE_CHANNELS.get(sensor)
    if expected_native is None:
        raise ValueError("diagnostic row has an undeclared sensor channel set")
    native = tuple(channel_order)
    if native != expected_native:
        raise ValueError("sensor-native channel order or membership differs from its source contract")
    denominators = list(source_pixel_denominators)
    unavailable = list(unavailable_fit_transform_channels)
    if len(denominators) != len(native) or any(type(value) is not int or value < 0
                                               for value in denominators):
        raise ValueError("sensor-native source pixel denominators must be aligned nonnegative integers")
    if len(unavailable) != len(native) or any(type(value) is not bool for value in unavailable):
        raise ValueError("sensor-native fit-transform availability must align with native channels")
    mask = np.asarray(observed_mask)
    if mask.dtype != np.dtype(bool) or mask.shape != (len(CHANNEL_ORDER),):
        raise ValueError("canonical score-support mask differs from the seven-axis channel contract")

    native_index = {channel: index for index, channel in enumerate(native)}
    membership = []
    canonical_denominators = []
    canonical_unavailable = []
    channel_support = []
    observed_ids = []
    for canonical_index, channel in enumerate(CHANNEL_ORDER):
        if channel in native_index:
            native_position = native_index[channel]
            denominator = denominators[native_position]
            unavailable_fit = unavailable[native_position]
            observed = bool(mask[canonical_index])
            if observed != (denominator > 0):
                raise ValueError("native denominator and canonical observed support disagree")
            membership.append("PRESENT")
            canonical_denominators.append(denominator)
            canonical_unavailable.append(unavailable_fit)
            channel_support.append({
                "channel_id": channel,
                "sensor_membership": "PRESENT",
                "source_pixel_denominator": denominator,
                "fit_transform_unavailable": unavailable_fit,
                "score_input_observed": observed,
                "status": "PRESENT_IN_SENSOR_NATIVE_CHANNEL_SET",
            })
            if observed:
                observed_ids.append(channel)
        else:
            if mask[canonical_index]:
                raise ValueError("absent sensor channel claims canonical observed support")
            membership.append("ABSENT_NOT_IN_SENSOR_CHANNEL_SET")
            canonical_denominators.append(None)
            canonical_unavailable.append(None)
            channel_support.append({
                "channel_id": channel,
                "sensor_membership": "ABSENT",
                "source_pixel_denominator": None,
                "fit_transform_unavailable": None,
                "score_input_observed": False,
                "status": ABSENT_SENSOR_CHANNEL_STATUS,
            })
    return {
        "sensor": sensor,
        "channel_order": list(native),
        "source_pixel_denominators": denominators,
        "unavailable_fit_transform_channels": unavailable,
        "canonical_channel_order": list(CHANNEL_ORDER),
        "canonical_sensor_membership": membership,
        "canonical_source_pixel_denominators": canonical_denominators,
        "canonical_unavailable_fit_transform_channels": canonical_unavailable,
        "channel_support": channel_support,
        "observed_channel_ids": observed_ids,
        "observed_channel_denominator": len(observed_ids),
    }


def validate_sensor_channel_support(record: Mapping[str, Any],
                                    observed_mask: np.ndarray | None = None) -> None:
    """Reject fabricated, reordered, or misaligned native/canonical support."""
    required = {
        "sensor", "channel_order", "source_pixel_denominators",
        "unavailable_fit_transform_channels", "canonical_channel_order",
        "canonical_sensor_membership", "canonical_source_pixel_denominators",
        "canonical_unavailable_fit_transform_channels", "channel_support",
        "observed_channel_ids", "observed_channel_denominator",
    }
    if not required.issubset(record):
        raise ValueError("row lacks explicit sensor-native and canonical channel support fields")
    if observed_mask is None:
        ids = record["observed_channel_ids"]
        if (not isinstance(ids, list) or any(not isinstance(value, str) for value in ids) or
                len(ids) != len(set(ids))):
            raise ValueError("observed channel identities must be a unique explicit list")
        observed_mask = np.asarray([channel in ids for channel in CHANNEL_ORDER], dtype=bool)
    expected = _sensor_native_support_fields(
        record["sensor"], record["channel_order"], record["source_pixel_denominators"],
        record["unavailable_fit_transform_channels"], observed_mask)
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError("sensor-native and canonical channel support do not join exactly: " + key)


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return _canonical_sha256(value)


def _feature_map(rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not rows:
        return np.empty((0, 14), dtype=np.float64), np.empty((0, 7), dtype=bool), []
    for row in rows:
        validate_sensor_channel_support(row, observed_mask=row.get("observed"))
    masks = np.asarray([r["observed"] for r in rows], dtype=bool)
    raw = np.asarray([r["values"] for r in rows], dtype=np.float64)
    if raw.shape != masks.shape or raw.ndim != 2 or raw.shape[1] != len(CHANNEL_ORDER):
        raise ValueError("diagnostic rows differ from the fixed channel schema")
    if not np.isfinite(raw[masks]).all():
        raise ValueError("observed feature values must be finite; no infinity or fallback conversion is allowed")
    values = np.where(masks, raw, 0.0)  # computational sentinel; mask bits preserve explicit missingness
    mapped = np.concatenate((values, masks.astype(np.float64)), axis=1)
    return mapped, masks, [r["row_id"] for r in rows]


def _average_path_length(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + 0.5772156649015329) - 2.0 * (n - 1) / n


def _tree_path_expectation(tree, feature_map: Sequence[int], mapped: np.ndarray,
                           original_observed: np.ndarray) -> float:
    structure = tree.tree_

    def visit(node: int, depth: int) -> float:
        left = int(structure.children_left[node])
        right = int(structure.children_right[node])
        n_at_node = int(structure.n_node_samples[node])
        if left == right:
            return float(depth) + _average_path_length(n_at_node)
        local_feature = int(structure.feature[node])
        global_feature = int(feature_map[local_feature])
        if global_feature < len(CHANNEL_ORDER) and not original_observed[global_feature]:
            left_n = int(structure.n_node_samples[left])
            right_n = int(structure.n_node_samples[right])
            total = left_n + right_n
            if total <= 0:
                raise ValueError("IsolationForest missing route has no fitted child support")
            return (left_n / total) * visit(left, depth + 1) + (right_n / total) * visit(right, depth + 1)
        return visit(left if mapped[global_feature] <= structure.threshold[node] else right, depth + 1)

    return visit(0, 0)


def _isolation_state_hash(model: IsolationForest, transform_hash: str,
                          fit_row_ids: Sequence[str], fit_matrix: np.ndarray,
                          fit_masks: np.ndarray) -> tuple[str, list[dict]]:
    trees = []
    for estimator, features in zip(model.estimators_, model.estimators_features_):
        t = estimator.tree_
        arrays = [t.children_left, t.children_right, t.feature, t.threshold, t.n_node_samples]
        hashes = [_hash_array(np.asarray(a)) for a in arrays]
        trees.append({"feature_map":np.asarray(features, dtype=int).tolist(),
                      "node_count":int(t.node_count),"arrays_sha256":hashes})
    payload = {"version":"IF_PROPORTIONAL_MISSING_BRANCH_V1","feature_map_version":FEATURE_MAP_VERSION,
        "transform_sha256":transform_hash,"fit_row_ids_sha256":_canonical(list(fit_row_ids)),
        "fit_matrix_sha256":_hash_array(fit_matrix),"fit_masks_sha256":_hash_array(fit_masks),
        "n_estimators":len(model.estimators_),"max_samples":int(model.max_samples_),"trees":trees}
    return _canonical(payload), trees


def fit_isolation_forest(fit_rows: Sequence[dict], monitor_rows: Sequence[dict],
                         transform_hash: str, *, seed: int = 20261015,
                         n_estimators: int = 100) -> tuple[dict, list[dict], IsolationForest | None]:
    matrix, masks, row_ids = _feature_map(fit_rows)
    monitor_matrix, monitor_masks, _ = _feature_map(monitor_rows)
    eligible = masks.any(axis=1)
    if int(eligible.sum()) < 2:
        return ({"status":"UNAVAILABLE_INSUFFICIENT_FIT_ROWS","reason":"FEWER_THAN_TWO_OBSERVED_FIT_ROWS",
                 "fit_row_count":int(eligible.sum()),"state_sha256":None}, [], None)
    model = IsolationForest(n_estimators=n_estimators, max_samples=min(256,int(eligible.sum())),
                            contamination="auto", random_state=seed, n_jobs=1)
    model.fit(matrix[eligible])
    state_hash, trees = _isolation_state_hash(model,transform_hash,
        [row_ids[i] for i in np.flatnonzero(eligible)],matrix[eligible],masks[eligible])
    scores=[]
    for role, rows, data, valid in (("fit",fit_rows,matrix,masks),("monitor",monitor_rows,monitor_matrix,monitor_masks)):
        for i,row in enumerate(rows):
            count=int(valid[i].sum())
            if not count:
                score=None; status="ABSTAIN_ALL_FEATURES_MISSING"; expected_path=None
            else:
                paths=[]
                for estimator,feature_map in zip(model.estimators_,model.estimators_features_):
                    paths.append(_tree_path_expectation(estimator,feature_map,data[i],valid[i]))
                expected_path=float(np.mean(paths))
                normalizer=_average_path_length(int(model.max_samples_))
                score=float(2.0 ** (-expected_path/normalizer)) if normalizer>0 else None
                status="ESTIMATED" if score is not None else "UNAVAILABLE_DEGENERATE_FIT_SIZE"
            scores.append(_score_record("isolation_forest",role,row,score,status,
                "EXPECTED_PATH_LENGTH_NORMALIZED_ANOMALY_SCORE",count,
                fit_transform_sha256=transform_hash,expected_path_length=expected_path))
    state={"status":"FITTED_DEVELOPMENT_DIAGNOSTIC","state_sha256":state_hash,
        "algorithm":"IsolationForest with proportional missing-child traversal",
        "feature_map_version":FEATURE_MAP_VERSION,"fit_transform_sha256":transform_hash,
        "seed":seed,"n_estimators":n_estimators,"max_samples":int(model.max_samples_),
        "fit_rows":int(eligible.sum()),"fit_row_ids_sha256":_canonical([row_ids[i] for i in np.flatnonzero(eligible)]),
        "tree_count":len(trees),"tree_states":trees,
        "missing_route":"weighted expectation by fitted left/right child n_node_samples; all-missing row abstains",
        "score_direction":"higher means shorter expected path / more anomalous"}
    return state,scores,model


def _score_record(method: str, role: str, row: dict, score: float | None,
                  status: str, unit: str, denominator: int, **extra) -> dict:
    validate_sensor_channel_support(row, observed_mask=row.get("observed"))
    if type(denominator) is not int or denominator != row["observed_channel_denominator"]:
        raise ValueError("method score denominator differs from its source-backed row support")
    support_fields = {key: row[key] for key in (
        "sensor", "channel_order", "source_pixel_denominators",
        "unavailable_fit_transform_channels", "canonical_channel_order",
        "canonical_sensor_membership", "canonical_source_pixel_denominators",
        "canonical_unavailable_fit_transform_channels", "channel_support",
        "observed_channel_ids", "observed_channel_denominator")}
    return {"method":method,"role":role,"row_id":row["row_id"],"site_id":row["site_id"],
        "panel_id":row["panel_id"],"group_id":row["group_id"],"source_id":row["source_id"],
        "support_id":row["support_id"],**support_fields,"score":score,"status":status,
        "reason":None if score is not None else status,"score_unit":unit,
        "fit_transform_sha256":extra.pop("fit_transform_sha256",None),**extra}


def rbf_kernel(matrix: np.ndarray, gamma: float) -> np.ndarray:
    x=np.asarray(matrix,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all() or not math.isfinite(gamma) or gamma<=0:
        raise ValueError("RBF kernel requires a finite matrix and positive finite gamma")
    norms=np.sum(x*x,axis=1)
    distance=np.maximum(norms[:,None]+norms[None,:]-2*x@x.T,0.0)
    return np.exp(-gamma*distance)


def fit_psd_rbf_ocsvm(fit_rows: Sequence[dict], monitor_rows: Sequence[dict],
                      transform_hash: str, *, nu: float = 0.1) -> tuple[dict,list[dict],OneClassSVM|None]:
    matrix,masks,row_ids=_feature_map(fit_rows)
    monitor_matrix,monitor_masks,_=_feature_map(monitor_rows)
    eligible=masks.any(axis=1)
    if type(nu) not in (int,float) or not math.isfinite(float(nu)) or not 0<float(nu)<=1:
        raise ValueError("nu must lie in (0,1]")
    if int(eligible.sum())<2:
        return ({"status":"UNAVAILABLE_INSUFFICIENT_FIT_ROWS","reason":"FEWER_THAN_TWO_OBSERVED_FIT_ROWS",
                 "fit_row_count":int(eligible.sum()),"state_sha256":None},[],None)
    x=matrix[eligible]
    variance=float(np.var(x))
    gamma=1.0/(x.shape[1]*variance) if variance>0 else 1.0/x.shape[1]
    model=OneClassSVM(kernel="rbf",gamma=gamma,nu=float(nu),shrinking=True,cache_size=256)
    model.fit(x)
    state_payload={"version":"PSD_RBF_OCSVM_FIXED_MISSINGNESS_MAP_V1","feature_map_version":FEATURE_MAP_VERSION,
        "transform_sha256":transform_hash,"gamma":gamma,"nu":float(nu),
        "fit_row_ids_sha256":_canonical([row_ids[i] for i in np.flatnonzero(eligible)]),
        "fit_matrix_sha256":_hash_array(x),"support_vectors_sha256":_hash_array(model.support_vectors_),
        "dual_coef_sha256":_hash_array(model.dual_coef_),"intercept_sha256":_hash_array(model.intercept_)}
    state_hash=_canonical(state_payload)
    scores=[]
    for role,rows,data,valid in (("fit",fit_rows,matrix,masks),("monitor",monitor_rows,monitor_matrix,monitor_masks)):
        values=-model.decision_function(data).reshape(-1)
        for i,row in enumerate(rows):
            count=int(valid[i].sum())
            score=float(values[i]) if count else None
            status="ESTIMATED" if count else "ABSTAIN_ALL_FEATURES_MISSING"
            scores.append(_score_record("psd_rbf_one_class_svm",role,row,score,status,
                "NEGATIVE_ONE_CLASS_DECISION_FUNCTION",count,fit_transform_sha256=transform_hash))
    state={"status":"FITTED_DEVELOPMENT_DIAGNOSTIC","state_sha256":state_hash,
        "algorithm":"OneClassSVM with fixed zero-sentinel plus observed-bit map and PSD Gaussian RBF",
        "feature_map_version":FEATURE_MAP_VERSION,"fit_transform_sha256":transform_hash,
        "kernel":"exp(-gamma * squared_Euclidean_distance); positive semidefinite",
        "gamma":gamma,"nu":float(nu),"fit_rows":int(eligible.sum()),
        "fit_row_ids_sha256":state_payload["fit_row_ids_sha256"],
        "support_vectors":model.support_vectors_.tolist(),"dual_coef":model.dual_coef_.tolist(),
        "intercept":model.intercept_.tolist(),"score_direction":"higher means more anomalous"}
    return state,scores,model


def _soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x)*np.maximum(np.abs(x)-threshold,0.0)


def _svt(x: np.ndarray, threshold: float) -> np.ndarray:
    u,s,vt=np.linalg.svd(x,full_matrices=False)
    return (u*np.maximum(s-threshold,0.0))@vt


def inexact_alm_rpca(values: np.ndarray, observed: np.ndarray, *,
                     max_iterations: int = 150, tolerance: float = 1e-6,
                     inner_iterations: int = 20) -> dict:
    """Fit complete-data IALM or observed-entry masked robust completion.

    The masked objective only includes observed cells. Values beneath a false
    mask are never read into the objective and may contain arbitrary sentinels.
    """
    raw=np.asarray(values,dtype=np.float64); mask=np.asarray(observed,dtype=bool)
    if raw.ndim!=2 or raw.shape!=mask.shape or min(raw.shape)<2:
        raise ValueError("RPCA requires matching two-dimensional arrays with at least two rows and columns")
    if type(max_iterations) is not int or max_iterations<1 or type(inner_iterations) is not int or inner_iterations<1:
        raise ValueError("RPCA iteration budgets must be positive integers")
    if not math.isfinite(tolerance) or tolerance<=0:
        raise ValueError("RPCA tolerance must be positive and finite")
    if not np.isfinite(raw[mask]).all() or not mask.any():
        raise ValueError("RPCA observed entries must be finite and nonempty")
    complete=bool(mask.all())
    d=np.where(mask,raw,0.0)
    norm_f=float(np.linalg.norm(d,"fro"))
    if norm_f<=0:
        return {"status":"UNAVAILABLE_ZERO_OBSERVED_NORM","complete_data":complete,
                "low_rank":np.zeros_like(d),"sparse":np.zeros_like(d),"iterations":0,
                "primal_residual":None,"dual_residual":None,"rank":0,"sparsity":None,"trace":[]}
    lam=1.0/math.sqrt(max(d.shape))
    spectral=float(np.linalg.norm(d,2))
    infinity=float(np.max(np.abs(d[mask])))
    mu=1.25/max(spectral,infinity/lam,np.finfo(float).eps)
    y=np.where(mask,d/max(infinity,np.finfo(float).eps),0.0)
    low=np.zeros_like(d); sparse=np.zeros_like(d); trace=[]
    converged=False; primal=dual=None
    for iteration in range(1,max_iterations+1):
        old_low=low.copy()
        if complete:
            low=_svt(d-sparse+y/mu,1.0/mu)
            sparse=_soft_threshold(d-low+y/mu,lam/mu)
        else:
            sparse=np.where(mask,_soft_threshold(d-low+y/mu,lam/mu),0.0)
            target=d-sparse+y/mu
            current=low.copy(); momentum=1.0; prior=current
            for _ in range(inner_iterations):
                gradient=np.where(mask,current-target,0.0)
                candidate=_svt(current-gradient,1.0/mu)
                next_momentum=(1.0+math.sqrt(1.0+4.0*momentum*momentum))/2.0
                accelerated=candidate+((momentum-1.0)/next_momentum)*(candidate-prior)
                if np.linalg.norm(candidate-prior,"fro") <= tolerance*max(1.0,np.linalg.norm(candidate,"fro")):
                    current=candidate; break
                prior=candidate; current=accelerated; momentum=next_momentum
            low=current
            sparse=np.where(mask,_soft_threshold(d-low+y/mu,lam/mu),0.0)
        residual=np.where(mask,d-low-sparse,0.0)
        y=y+mu*residual
        primal=float(np.linalg.norm(residual,"fro")/norm_f)
        dual=float(mu*np.linalg.norm(np.where(mask,low-old_low,0.0),"fro")/norm_f)
        trace.append({"iteration":iteration,"primal_residual":primal,"dual_residual":dual,
                      "rank":int(np.linalg.matrix_rank(low)),"sparse_observed_nonzero":int(np.count_nonzero(sparse[mask]))})
        if primal<=tolerance and dual<=tolerance:
            converged=True; break
        mu=min(mu*1.5,1e7)
    singular=np.linalg.svd(low,compute_uv=False)
    rank=int(np.sum(singular>tolerance*max(float(singular[0]) if singular.size else 0.0,1.0)))
    return {"status":"CONVERGED" if converged else "NONCONVERGED_MAX_ITERATIONS",
        "algorithm":"COMPLETE_DATA_INEXACT_ALM_ROBUST_PCA" if complete else "OBSERVED_ENTRY_MASKED_INEXACT_ALM_ROBUST_COMPLETION",
        "complete_data":complete,"low_rank":low,"sparse":sparse,"iterations":len(trace),
        "primal_residual":primal,"dual_residual":dual,"rank":rank,
        "sparsity":float(np.count_nonzero(sparse[mask])/int(mask.sum())),"trace":trace,
        "observed_entry_count":int(mask.sum()),"missing_entry_count":int(mask.size-mask.sum()),
        "tolerance":tolerance,"lambda":lam}


def rpca_project_score(state: dict, values: np.ndarray, observed: np.ndarray) -> dict:
    """Frozen out-of-sample residual against the fit decomposition's row-space basis."""
    x=np.asarray(values,dtype=np.float64); mask=np.asarray(observed,dtype=bool)
    if x.ndim!=2 or x.shape!=mask.shape or x.shape[1]!=state["low_rank"].shape[1]:
        raise ValueError("RPCA monitor matrix differs from frozen fit feature schema")
    u,s,vt=np.linalg.svd(state["low_rank"],full_matrices=False)
    rank=int(state["rank"])
    basis=vt[:rank].T if rank else np.empty((x.shape[1],0))
    scores=[]
    for row,valid in zip(x,mask):
        count=int(valid.sum())
        if rank==0:
            scores.append({"score":None,"status":"UNAVAILABLE_ZERO_FIT_RANK","denominator":count})
            continue
        if count<=rank:
            scores.append({"score":None,"status":"ABSTAIN_INSUFFICIENT_OBSERVED_FEATURES_FOR_PROJECTION","denominator":count})
            continue
        design=basis[valid]
        coeff,_,_,_=np.linalg.lstsq(design,row[valid],rcond=None)
        residual=row[valid]-design@coeff
        score=float(np.linalg.norm(residual)/math.sqrt(count))
        status="ESTIMATED" if state["status"]=="CONVERGED" else "ESTIMATED_FROM_NONCONVERGED_FIT"
        scores.append({"score":score,"status":status,"denominator":count})
    return {"rank":rank,"scores":scores,"fit_convergence_status":state["status"],
            "fit_state_frozen":True}


def fit_cusum_standardizer(fit_scores: Sequence[float | None], fit_role: str,
                           fit_row_ids: Sequence[str]) -> dict:
    if fit_role!="fit" or len(fit_scores)!=len(fit_row_ids):
        raise ValueError("CUSUM location/scale may be fitted only on aligned parity-0 fit scores")
    if any(x is not None and (type(x) not in (int,float) or not math.isfinite(float(x))) for x in fit_scores):
        raise ValueError("nonfinite CUSUM fit score must remain an error, not be silently dropped")
    finite=np.asarray([x for x in fit_scores if x is not None and math.isfinite(float(x))],dtype=np.float64)
    if finite.size<2:
        return {"status":"UNAVAILABLE_INSUFFICIENT_FIT_SCORES","center":None,"scale":None,
                "fit_row_ids_sha256":_canonical(list(fit_row_ids))}
    center=float(finite.mean()); scale=float(finite.std(ddof=1))
    if scale<=0 or not math.isfinite(scale):
        return {"status":"UNAVAILABLE_DEGENERATE_FIT_SCALE","center":None,"scale":None,
                "fit_row_ids_sha256":_canonical(list(fit_row_ids))}
    payload={"version":"CUSUM_FIT_ONLY_STANDARDIZER_V1","center":center,"scale":scale,
        "fit_row_ids_sha256":_canonical(list(fit_row_ids)),"fit_finite_count":int(finite.size)}
    return {**payload,"status":"FITTED_DEVELOPMENT_DIAGNOSTIC","state_sha256":_canonical(payload)}


def run_cusum(scores: Sequence[float | None], arrival_times: Sequence[str | None],
              row_ids: Sequence[str], standardizer: Mapping[str,Any], *,
              reference_k: float=0.5, max_gap_days: int=90) -> dict:
    if not (len(scores)==len(arrival_times)==len(row_ids)):
        raise ValueError("CUSUM score, timestamp and row ID lengths differ")
    if standardizer.get("status")!="FITTED_DEVELOPMENT_DIAGNOSTIC":
        return {"status":"UNAVAILABLE_FIT_STANDARDIZER","rows":[]}
    if any(t is None for t in arrival_times):
        return {"status":"UNAVAILABLE_ACTUAL_ARRIVAL_TIME_MISSING","reason":"ACTUAL_ARRIVAL_TIME_NOT_SOURCE_BOUND",
                "rows":[],"input_row_count":len(row_ids)}
    parsed=[date.fromisoformat(str(t)[:10]) for t in arrival_times]
    if any(parsed[i]>=parsed[i+1] for i in range(len(parsed)-1)):
        raise ValueError("CUSUM requires strictly increasing distinct actual arrivals")
    if type(max_gap_days) is not int or max_gap_days<1 or not math.isfinite(reference_k) or reference_k<0:
        raise ValueError("CUSUM gap/reference policy invalid")
    center=float(standardizer["center"]); scale=float(standardizer["scale"])
    pos=0.0; neg=0.0; previous=None; out=[]
    for score,stamp,row_id in zip(scores,parsed,row_ids):
        gap=(stamp-previous).days if previous is not None else None
        if gap is not None and gap>max_gap_days: pos=neg=0.0
        if score is None or not math.isfinite(float(score)):
            pos=neg=0.0; status="ABSTAIN_RESET_MISSING_SCORE"; z=None
        else:
            z=(float(score)-center)/scale
            pos=max(0.0,pos+z-reference_k)
            neg=min(0.0,neg+z+reference_k)
            status="ESTIMATED"
        out.append({"row_id":row_id,"arrival_date":stamp.isoformat(),"standardized_score":z,
                    "positive_cusum":pos if z is not None else None,
                    "negative_cusum":neg if z is not None else None,
                    "gap_days":gap,"status":status})
        previous=stamp
    return {"status":"ESTIMATED_DEVELOPMENT_DIAGNOSTIC","fit_standardizer_sha256":standardizer["state_sha256"],
            "reference_k":reference_k,"max_gap_days":max_gap_days,"rows":out}


def gpd_return_level(threshold: float, xi: float, beta: float,
                     score_exceedance_rate_per_lake_year: float,
                     target_score_exceedance_rate_per_lake_year: float=1.0) -> float:
    """Return a score-tail level; this is not an alarm-episode burden estimate."""
    vals=(threshold,xi,beta,score_exceedance_rate_per_lake_year,
          target_score_exceedance_rate_per_lake_year)
    if any(type(x) not in (int,float) or not math.isfinite(float(x)) for x in vals):
        raise ValueError("GPD return level requires finite numeric parameters")
    if (beta<=0 or score_exceedance_rate_per_lake_year<=0 or
        target_score_exceedance_rate_per_lake_year<=0):
        raise ValueError("GPD scale and rates must be positive")
    ratio=score_exceedance_rate_per_lake_year/target_score_exceedance_rate_per_lake_year
    log_ratio=math.log(ratio)
    excess=beta*log_ratio if abs(xi)<1e-12 else beta*math.expm1(xi*log_ratio)/xi
    if xi<0 and excess>=-beta/xi:
        raise ValueError("requested GPD return level reaches or exceeds the finite negative-shape endpoint")
    result=threshold+excess
    if not math.isfinite(result):
        raise ValueError("GPD return level is nonfinite")
    return result


def fit_gpd_tail(exceedances: Sequence[float], threshold: float,
                 exposure_lake_years: float, *, independent_unit_ids: Sequence[str],
                 role: str, stationarity_reviewed: bool,
                 declustering_reviewed: bool,
                 target_score_exceedance_rate_per_lake_year: float=1.0) -> dict:
    if role!="threshold_calibration":
        raise ValueError("GPD fit requires disjoint threshold_calibration role")
    x=np.asarray(exceedances,dtype=np.float64)
    if (x.ndim!=1 or not np.isfinite(x).all() or not math.isfinite(threshold) or
        not math.isfinite(exposure_lake_years) or exposure_lake_years<=0 or
        not math.isfinite(target_score_exceedance_rate_per_lake_year) or
        target_score_exceedance_rate_per_lake_year<=0 or
        len(independent_unit_ids)!=x.size or len(set(independent_unit_ids))!=len(independent_unit_ids)):
        raise ValueError("GPD calibration rows/exposure/independent-unit join invalid")
    if not stationarity_reviewed or not declustering_reviewed:
        return {"status":"UNAVAILABLE_CALIBRATION_ASSUMPTIONS_UNREVIEWED","threshold":None,
                "gpd_shape":None,"gpd_scale":None,
                "empirical_score_exceedance_rate_per_lake_year":None,
                "achieved_false_alarm_episode_rate_per_lake_year":None,
                "episode_rate_status":"NOT_ESTIMATED_NO_ORDERED_ARRIVAL_AND_EPISODE_ACCOUNTING"}
    if x.size<3 or np.any(x<=threshold):
        return {"status":"INSUFFICIENT_OR_INVALID_TAIL_EXCEEDANCES","threshold":None,
                "gpd_shape":None,"gpd_scale":None,
                "achieved_false_alarm_episode_rate_per_lake_year":None,
                "episode_rate_status":"NOT_ESTIMATED_NO_ORDERED_ARRIVAL_AND_EPISODE_ACCOUNTING",
                "exceedance_count":int(x.size)}
    excess=x-threshold
    try:
        xi,loc,beta=genpareto.fit(excess,floc=0.0)
        rate=float(x.size/exposure_lake_years)
        return {"status":"ESTIMATED_CALIBRATION_DIAGNOSTIC","threshold":float(gpd_return_level(
                threshold,float(xi),float(beta),rate,target_score_exceedance_rate_per_lake_year)),
            "initial_threshold":float(threshold),"gpd_shape":float(xi),"gpd_scale":float(beta),
            "target_score_exceedance_rate_per_lake_year":float(target_score_exceedance_rate_per_lake_year),
            "achieved_false_alarm_episode_rate_per_lake_year":None,
            "empirical_score_exceedance_rate_per_lake_year":rate,
            "episode_rate_status":"NOT_ESTIMATED_NO_ORDERED_ARRIVAL_AND_EPISODE_ACCOUNTING",
            "exceedance_count":int(x.size),
            "independent_unit_ids_sha256":_canonical(sorted(independent_unit_ids)),
            "stationarity_reviewed":True,"declustering_reviewed":True,"fit_location_fixed_zero":float(loc)}
    except (ValueError,FloatingPointError,OverflowError) as exc:
        return {"status":"GPD_FIT_FAILED","threshold":None,"gpd_shape":None,
                "gpd_scale":None,"empirical_score_exceedance_rate_per_lake_year":None,
                "achieved_false_alarm_episode_rate_per_lake_year":None,
                "episode_rate_status":"NOT_ESTIMATED_NO_ORDERED_ARRIVAL_AND_EPISODE_ACCOUNTING",
                "failure_reason":type(exc).__name__}


def empirical_return_threshold(scores: Sequence[float], exposure_lake_years: float,
                                target_score_exceedance_rate_per_lake_year: float, *,
                                independent_unit_ids: Sequence[str], role: str) -> dict:
    if role!="threshold_calibration":
        raise ValueError("empirical return threshold requires threshold_calibration role")
    x=np.asarray(scores,dtype=np.float64)
    if (x.ndim!=1 or not np.isfinite(x).all() or len(independent_unit_ids)!=x.size or
        type(exposure_lake_years) not in (int,float) or not math.isfinite(float(exposure_lake_years)) or
        type(target_score_exceedance_rate_per_lake_year) not in (int,float) or
        not math.isfinite(float(target_score_exceedance_rate_per_lake_year))):
        raise ValueError("empirical calibration scores and independent units must align")
    if len(set(independent_unit_ids))!=len(independent_unit_ids):
        raise ValueError("empirical calibration rows contain repeated independent units")
    if x.size==0 or exposure_lake_years<=0 or target_score_exceedance_rate_per_lake_year<=0:
        return {"status":"INSUFFICIENT_CALIBRATION_SUPPORT","threshold":None,
                "empirical_score_exceedance_rate_per_lake_year":None,
                "achieved_false_alarm_episode_rate_per_lake_year":None,
                "episode_rate_status":"NOT_ESTIMATED_NO_ORDERED_ARRIVAL_AND_EPISODE_ACCOUNTING",
                "n_calibration_units":int(x.size)}
    tail_probability=target_score_exceedance_rate_per_lake_year*exposure_lake_years/x.size
    if tail_probability<=0 or tail_probability>1:
        return {"status":"TARGET_NOT_RESOLVABLE_FROM_EMPIRICAL_SUPPORT","threshold":None,
                "empirical_score_exceedance_rate_per_lake_year":None,
                "achieved_false_alarm_episode_rate_per_lake_year":None,
                "episode_rate_status":"NOT_ESTIMATED_NO_ORDERED_ARRIVAL_AND_EPISODE_ACCOUNTING",
                "n_calibration_units":int(x.size),
                "tail_probability":tail_probability}
    q=1.0-tail_probability
    threshold=float(np.quantile(x,q,method="higher"))
    score_exceedance_count=int(np.sum(x>=threshold))
    rate=float(score_exceedance_count/exposure_lake_years)
    return {"status":"ESTIMATED_EMPIRICAL_SCORE_EXCEEDANCE_CALIBRATION_DIAGNOSTIC","threshold":threshold,
            "empirical_score_exceedance_count":score_exceedance_count,
            "empirical_score_exceedance_rate_per_lake_year":rate,
            "achieved_false_alarm_episode_rate_per_lake_year":None,
            "episode_rate_status":"NOT_ESTIMATED_NO_ORDERED_ARRIVAL_AND_EPISODE_ACCOUNTING",
            "n_calibration_units":int(x.size),
            "tail_probability":tail_probability,"quantile_method":"higher; ties count as exceedances",
            "independent_unit_ids_sha256":_canonical(sorted(independent_unit_ids))}


def calibrate_actual_panel_threshold(*, calibration_rows: Sequence[dict], exposure_lake_years: float | None,
                                     stationarity_reviewed: bool, declustering_reviewed: bool) -> dict:
    """Fail closed for the present panel, which has no calibration-role rows."""
    rows=list(calibration_rows)
    if not rows or exposure_lake_years is None or not stationarity_reviewed or not declustering_reviewed:
        return {"objective":"1 false alarm per lake-year","status":"INSUFFICIENT_CALIBRATION_SUPPORT",
            "threshold":None,"gpd_shape":None,"gpd_scale":None,
            "achieved_false_alarm_episode_rate_per_lake_year":None,
            "empirical_quantile_threshold":None,"reason":"NO_DISJOINT_THRESHOLD_CALIBRATION_ROWS_OR_VERIFIED_EXPOSURE_STATIONARITY_AND_DECLUSTERING",
            "calibration_row_count":len(rows),"exposure_lake_years":None,
            "seed_pixel_quadrant_or_monitor_rows_used":False}
    raise ValueError("actual calibration rows require independent-unit and source-bound calibration contract")


def _parse_source_arrival(value: str) -> tuple[date | datetime, str]:
    if not isinstance(value, str) or not value:
        raise ValueError("persistence requires source-bound actual arrival timestamps")
    if len(value) == 10:
        try:
            parsed_date = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("date-resolution arrival timestamp is malformed") from exc
        if parsed_date.isoformat() != value:
            raise ValueError("date-resolution arrival timestamp must be exact ISO YYYY-MM-DD")
        return parsed_date, "DAY"
    if "T" not in value:
        raise ValueError("arrival timestamp must retain its date or ISO instant precision")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError("source arrival instant is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("source arrival instant requires an explicit timezone")
    clock = value.split("T", 1)[1]
    if clock.endswith("Z"):
        clock = clock[:-1]
    for offset_marker in ("+", "-"):
        if offset_marker in clock:
            clock = clock.split(offset_marker, 1)[0]
            break
    if "." in clock:
        fraction = clock.split(".", 1)[1]
        precision = f"INSTANT_FRACTIONAL_SECOND_{len(fraction)}_DIGITS"
    elif clock.count(":") >= 2:
        precision = "INSTANT_SECOND"
    elif clock.count(":") == 1:
        precision = "INSTANT_MINUTE"
    else:
        precision = "INSTANT_HOUR"
    return parsed.astimezone(timezone.utc), precision


def persistent_declarations(rows: Sequence[Mapping[str,Any]], threshold: float | None, *,
                             k: int=2, max_gap_days: int=90,
                             refractory_days: int=0) -> dict:
    if (type(k) is not int or k<2 or type(max_gap_days) is not int or max_gap_days<1 or
        type(refractory_days) is not int or refractory_days<0):
        raise ValueError("persistence requires k>=2 and valid gap/refractory days")
    rows=list(rows)
    if threshold is None or type(threshold) not in (int,float) or not math.isfinite(float(threshold)):
        return {"status":"NOT_RUN_INVALID_CALIBRATION_THRESHOLD","threshold":None,
                "declarations":None,"episode_closures":None,
                "reason":"NO_RESEARCH_ALARMS_WITHOUT_VALID_CALIBRATION"}
    ids=[r.get("arrival_id") for r in rows]
    sources=[r.get("source_id") for r in rows]
    if any(not isinstance(value,str) or not value for value in ids) or len(set(ids))!=len(ids):
        raise ValueError("persistence arrivals need unique nonempty IDs")
    if any(not isinstance(value,str) or not value for value in sources) or len(set(sources))!=len(sources):
        raise ValueError("persistence input contains missing or duplicate source scenes")
    parsed=[]; resolutions=[]
    for row in rows:
        stamp, resolution = _parse_source_arrival(row.get("arrival_at"))
        parsed.append(stamp); resolutions.append(resolution)
    resolution_kinds={"DAY" if type(stamp) is date else "INSTANT" for stamp in parsed}
    if len(resolution_kinds)>1:
        raise ValueError("persistence cannot mix date-only and instant arrival resolution")
    if any(parsed[i]>=parsed[i+1] for i in range(len(parsed)-1)):
        raise ValueError("persistence arrivals must be strictly ordered and distinct")

    streak=[]; declarations=[]; closures=[]; dispositions=[]
    active_episode=None; refractory_until=None; previous=None
    day_resolution=bool(parsed and type(parsed[0]) is date)

    def elapsed_days(current, prior):
        delta=current-prior
        return float(delta.days) if day_resolution else delta.total_seconds()/86400.0

    def close_active(row, stamp, resolution, reason):
        nonlocal active_episode, refractory_until
        episode=active_episode
        refractory_until=stamp+timedelta(days=refractory_days)
        closure={"episode_id":episode["episode_id"],"closed_at":row["arrival_at"],
            "closure_time_resolution":resolution,"closure_reason":reason,
            "closed_by_arrival_id":row["arrival_id"],
            "refractory_started_at":row["arrival_at"],
            "refractory_until":refractory_until.isoformat(),
            "physical_episode_end_status":"NOT_ASSERTED_BY_DIAGNOSTIC_RESET_POLICY"}
        closures.append(closure)
        episode["episode_status"]="CLOSED_BY_DECLARED_RESET_POLICY"
        episode["closed_at"]=row["arrival_at"]
        episode["closure_time_resolution"]=resolution
        episode["closure_reason"]=reason
        episode["physical_episode_end_status"]="NOT_ASSERTED_BY_DIAGNOSTIC_RESET_POLICY"
        episode["refractory_started_at"]=row["arrival_at"]
        episode["refractory_until"]=refractory_until.isoformat()
        active_episode=None

    for row,stamp,resolution in zip(rows,parsed,resolutions):
        score=row.get("score"); eligible=row.get("eligible")
        if (type(eligible) is not bool or
            (eligible and (type(score) not in (int,float) or not math.isfinite(float(score)))) or
            (not eligible and score is not None)):
            raise ValueError("persistence score/eligibility null semantics invalid")
        operations=[]
        gap=elapsed_days(stamp,previous) if previous is not None else None
        if gap is not None and gap>max_gap_days:
            streak=[]
            if active_episode is not None:
                close_active(row,stamp,resolution,"MAX_GAP_RESET_AT_CURRENT_ARRIVAL")
                operations.append("ACTIVE_EPISODE_CLOSED_BY_MAX_GAP_RESET")
            else:
                operations.append("EXCEEDANCE_STREAK_RESET_BY_MAX_GAP")
        previous=stamp
        if not eligible:
            streak=[]
            if active_episode is not None:
                close_active(row,stamp,resolution,"INELIGIBLE_OR_MISSING_SCORE_ARRIVAL")
                operations.append("ACTIVE_EPISODE_CLOSED_BY_MISSING_OR_INELIGIBLE_ARRIVAL")
            else:
                operations.append("MISSING_OR_INELIGIBLE_ARRIVAL_RESET_STREAK")
            dispositions.append({"arrival_id":row["arrival_id"],"arrival_at":row["arrival_at"],
                "arrival_time_resolution":resolution,"operations":operations})
            continue
        if active_episode is not None:
            if float(score)>=float(threshold):
                active_episode["continuation_arrival_ids"].append(row["arrival_id"])
                active_episode["continuation_source_ids"].append(row["source_id"])
                operations.append("ACTIVE_EPISODE_CONTINUATION")
            else:
                close_active(row,stamp,resolution,"ELIGIBLE_BELOW_THRESHOLD_RECOVERY")
                operations.append("ACTIVE_EPISODE_CLOSED_BY_BELOW_THRESHOLD_RECOVERY")
            dispositions.append({"arrival_id":row["arrival_id"],"arrival_at":row["arrival_at"],
                "arrival_time_resolution":resolution,"operations":operations})
            continue
        if refractory_until is not None and stamp<refractory_until:
            streak=[]
            operations.append("REFRACTORY_SUPPRESSED_FROM_EPISODE_CLOSURE")
            dispositions.append({"arrival_id":row["arrival_id"],"arrival_at":row["arrival_at"],
                "arrival_time_resolution":resolution,"operations":operations})
            continue
        if float(score)>=float(threshold):
            streak.append(row)
            operations.append("EXCEEDANCE_STREAK_PROGRESS")
            if len(streak)==k:
                declaration={"episode_id":f"EPISODE-{len(declarations)+1:04d}",
                    "declared_at":row["arrival_at"],"episode_opened_at":row["arrival_at"],
                    "declared_at_resolution":resolution,
                    "sustaining_arrival_ids":[item["arrival_id"] for item in streak],
                    "sustaining_source_ids":[item["source_id"] for item in streak],
                    "continuation_arrival_ids":[],"continuation_source_ids":[],
                    "backdated":False,"episode_status":"OPEN_AT_SEQUENCE_END",
                    "closed_at":None,"closure_time_resolution":None,"closure_reason":None,
                    "physical_episode_end_status":"NOT_OBSERVED"}
                declarations.append(declaration)
                active_episode=declaration
                streak=[]
                operations.append("UNIQUE_EPISODE_OPENED_ON_KTH_SUSTAINING_ARRIVAL")
        else:
            streak=[]
            operations.append("BELOW_THRESHOLD_RESET_STREAK")
        dispositions.append({"arrival_id":row["arrival_id"],"arrival_at":row["arrival_at"],
            "arrival_time_resolution":resolution,"operations":operations})
    return {"status":"ESTIMATED_DEVELOPMENT_DIAGNOSTIC","threshold":float(threshold),
        "declarations":declarations,"episode_closures":closures,
        "arrival_dispositions":dispositions,"k":k,"max_gap_days":max_gap_days,
        "refractory_days":refractory_days,
        "refractory_starts_at":"DECLARED_EPISODE_CLOSURE_OBSERVATION",
        "timestamp_resolution_policy":"preserve source arrival string; DATE_ONLY declarations are DAY resolution; offset-aware instants retain original timestamp and explicit precision",
        "closure_policy":"one eligible below-threshold recovery, missing/ineligible arrival, or gap greater than max_gap_days closes an active diagnostic episode; max-gap closure is detected at current arrival and does not assert physical end; refractory begins at the closure observation",
        "episode_opening_policy":"open one unique diagnostic episode at the kth distinct eligible sustaining arrival; later continuous exceedances continue that episode; never backdate; no repeated notifications are emitted"}


def validate_diagnostic_report(report: Mapping[str,Any]) -> None:
    required={"schema_version","diagnostic_version","status","research_admissible","plan_executable",
        "release_status","fit_transform_sha256","input_sha256","application_receipts","row_summary",
        "method_states","method_state_hashes","scores","comparison_support","cusum","calibration",
        "persistence","tmae","limits"}
    if set(report)!=required:
        raise ValueError("diagnostic report fields differ from the versioned schema")
    if (report["schema_version"]!=2 or report["diagnostic_version"]!=DIAGNOSTIC_VERSION or
        report["status"]!="DEVELOPMENT_DIAGNOSTICS_COMPLETE_RESEARCH_ADMISSION_BLOCKED" or
        report["research_admissible"] is not False or report["plan_executable"] is not False or
        report["release_status"]!="BLOCKED"):
        raise ValueError("diagnostic report escaped its quarantined status contract")
    hashes=[report["fit_transform_sha256"],*report["input_sha256"].values()]
    if any(not isinstance(x,str) or len(x)!=64 or any(c not in "0123456789abcdef" for c in x) for x in hashes):
        raise ValueError("diagnostic report contains an invalid SHA-256 identity")
    for score in report["scores"]:
        for key in ("method","role","row_id","source_id","support_id","sensor","score","status","reason",
                    "score_unit","channel_order","observed_channel_denominator","source_pixel_denominators",
                    "unavailable_fit_transform_channels","canonical_channel_order",
                    "canonical_sensor_membership","canonical_source_pixel_denominators",
                    "canonical_unavailable_fit_transform_channels","channel_support","observed_channel_ids",
                    "fit_transform_sha256"):
            if key not in score: raise ValueError("score row lacks required support/status field: "+key)
        if score["role"] not in {"fit","monitor"} or score["method"] not in {"isolation_forest","psd_rbf_one_class_svm","inexact_alm_rpca"}:
            raise ValueError("score row has an undeclared method or diagnostic role")
        validate_sensor_channel_support(score)
        if score["score"] is None and not score["reason"]:
            raise ValueError("null score requires an explicit unavailable reason")
        if score["score"] is not None and (type(score["score"]) not in (int,float) or not math.isfinite(float(score["score"]))):
            raise ValueError("estimated score must be finite numeric data")
        if score["fit_transform_sha256"]!=report["fit_transform_sha256"]:
            raise ValueError("score row is not bound to the declared frozen transform")
    calibration=report["calibration"]
    calibration_required={"objective","status","threshold","gpd_shape","gpd_scale",
        "achieved_false_alarm_episode_rate_per_lake_year","empirical_quantile_threshold",
        "reason","calibration_row_count","exposure_lake_years",
        "seed_pixel_quadrant_or_monitor_rows_used"}
    if not calibration_required.issubset(calibration):
        raise ValueError("calibration report lacks required null/status fields")
    if calibration.get("status")!="INSUFFICIENT_CALIBRATION_SUPPORT" or any(
        calibration.get(field) is not None for field in
        ("threshold","gpd_shape","gpd_scale","achieved_false_alarm_episode_rate_per_lake_year",
         "empirical_quantile_threshold")):
        raise ValueError("actual panel calibration must remain explicitly unavailable")
    if report["persistence"]["actual"].get("status")!="NOT_RUN_INVALID_CALIBRATION_THRESHOLD":
        raise ValueError("persistence cannot run without a valid research calibration threshold")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()
