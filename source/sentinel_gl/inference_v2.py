"""Causal-prefix, version-2 retrospective diagnostics and sensor baselines."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math

import numpy as np
import torch
from sklearn.decomposition import PCA

from .panel_v2 import (
    CANONICAL_CHANNELS, CHANNELS_BY_SENSOR, PanelV2, _sha_json,
    causal_prefix, pool_observation_quadrants, sha256_file,
)
from .training_v2 import (
    TIME_REFERENCE, TemporalMaskedMAEV2, TrainingConfigV2, _fold_batch,
    _sequence_from_prefix, load_checkpoint_v2,
)
from .baseline_estimators import (
    ABSENT_SENSOR_CHANNEL_STATUS, _sensor_native_support_fields,
    fit_isolation_forest, fit_psd_rbf_ocsvm, inexact_alm_rpca,
    rpca_project_score,
)


INFERENCE_V2_VERSION = "CAUSAL_PREFIX_MULTIMODAL_SCORE_V2"
SCORE_B_PCA_RANK = 8
SCORE_B_K = 5
RPCA_MAX_ITERATIONS = 1000
RPCA_TOLERANCE = 1e-5
RPCA_ADMM_PENALTY_FACTOR = 5.0


def _context_prefix(records: Sequence[Mapping[str, Any]], query: Mapping[str, Any]):
    """Causal same-site context; suppress simultaneous duplicate same-sensor rows."""
    prefix = causal_prefix(records, query)
    query_at = query["acquired_at"]
    return tuple(row for row in prefix if row["observation_id"] == query["observation_id"] or
                 not (row["sensor"] == query["sensor"] and row["acquired_at"] == query_at))


def _encoder_token_embeddings(model: TemporalMaskedMAEV2,
                              sequence: Mapping[str, Any], query: Mapping[str, Any],
                              *, device: torch.device) -> list[dict | None]:
    query_positions = [i for i, owner in enumerate(sequence["owner_ids"])
                       if owner == query["observation_id"]]
    if len(query_positions) != 4:
        raise ValueError("query sequence must contain four quadrant tokens")
    result: list[list[np.ndarray]] = [[] for _ in range(4)]
    model.eval()
    with torch.no_grad():
        for hidden_q, hidden_pos in enumerate(query_positions):
            values = sequence["values"].copy()
            observed = sequence["observed"].copy()
            values[hidden_pos] = 0.0
            observed[hidden_pos] = False
            target_tokens = np.zeros(len(values), dtype=bool)
            target_tokens[hidden_pos] = True
            tensor = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=device).unsqueeze(0)
            encoded = model.encode(tensor(values, torch.float32),
                tensor(observed, torch.bool), tensor(sequence["modality"], torch.long),
                tensor(sequence["time"], torch.float32), tensor(sequence["quadrant"], torch.long),
                target_tokens=tensor(target_tokens, torch.bool))[0]
            for q, position in enumerate(query_positions):
                if q == hidden_q or not sequence["tokens"][position]["observed"].any():
                    continue
                result[q].append(encoded[position].detach().cpu().numpy().astype(np.float64))
    output = []
    for q, samples in enumerate(result):
        if not samples:
            output.append(None)
        else:
            output.append({"quadrant": q,
                "embedding": np.mean(np.stack(samples), axis=0, dtype=np.float64),
                "visible_masked_views": len(samples)})
    return output


def _capture_attention(model: TemporalMaskedMAEV2,
                       batch: Mapping[str, torch.Tensor], *,
                       return_prediction: bool = False) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], torch.Tensor]:
    """Capture actual per-head encoder/decoder attention for one forward pass."""
    captured: dict[str, np.ndarray] = {}
    patched = []
    prediction = None
    fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        for section, layers in (("encoder", model.encoder), ("decoder", model.decoder)):
            for index, layer in enumerate(layers):
                attention = layer.self_attn
                original = attention.forward
                key = f"{section}_layer_{index}"

                def forward_with_weights(*args, _original=original, _key=key, **kwargs):
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = False
                    output, weights = _original(*args, **kwargs)
                    if weights is not None:
                        captured[_key] = weights.detach().to(device="cpu", dtype=torch.float32).numpy()
                    return output, weights

                attention.forward = forward_with_weights
                patched.append((attention, original))
        model.eval()
        with torch.no_grad():
            prediction, _ = model(batch["values"], batch["observed"], batch["modality"],
                                  batch["time"], batch["quadrant"], batch["target"])
    finally:
        for attention, original in reversed(patched):
            attention.forward = original
        torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
    expected_layers = len(model.encoder) + len(model.decoder)
    if len(captured) != expected_layers:
        raise RuntimeError("selected model attention export did not capture every encoder and decoder layer")
    for name, weights in captured.items():
        if weights.ndim != 4 or not np.isfinite(weights).all():
            raise RuntimeError(f"captured attention has invalid shape or values: {name}")
    if return_prediction:
        return captured, prediction.detach().cpu()
    return captured


def _write_seed0_attention_export(model: TemporalMaskedMAEV2, panel: PanelV2,
                                  records: Sequence[Mapping[str, Any]],
                                  token_cache: Mapping[tuple[str, int], dict],
                                  output: Path, checkpoint_sha256: str,
                                  *, device: torch.device) -> dict:
    candidates = sorted((row for row in records
                         if row["site_id"] == "PILOT-SOUTH-LHONAK" and
                         row["sensor"] == "SENTINEL_2_OPTICAL" and
                         row["scope"]["event_phase"] == "POST_EVENT_CANDIDATE"),
                        key=lambda x: (x["acquired_at"], x["product_id"]))
    for query in candidates:
        prefix = _context_prefix(records, query)
        sequence = _sequence_from_prefix(prefix, token_cache)
        batch = _fold_batch(sequence, query["observation_id"], device)
        if batch is None:
            continue
        if 0 not in batch["folds"]:
            continue
        fold_index = batch["folds"].index(0)
        batch = {key: (value[fold_index:fold_index + 1] if isinstance(value, torch.Tensor) else
                       [value[fold_index]]) for key, value in batch.items()}
        attention = _capture_attention(model, batch)
        arrays = {}
        for name, value in attention.items():
            arrays[name] = value
            arrays[f"mean_heads_{name}"] = value.mean(axis=1, dtype=np.float64).astype(np.float32)
        target = output / "representative_seed0_attention.npz"
        np.savez_compressed(target, **arrays)
        metadata = {
            "status": "EXPORTED_FROM_SELECTED_SEED0_CHECKPOINT",
            "source": "ACTUAL_SELECTED_CHECKPOINT_FORWARD_PASS",
            "seed": 0, "checkpoint_sha256": checkpoint_sha256,
            "panel_manifest_sha256": panel.manifest_sha256,
            "domain_protocol_sha256": panel.domain_protocol_sha256,
            "query_observation_id": query["observation_id"],
            "query_product_id": query["product_id"], "query_source_sha256": query["sha256"],
            "query_acquired_at": query["acquired_at"],
            "context_observation_ids": [row["observation_id"] for row in prefix],
            "context_token_order": [{"observation_id": token["observation_id"],
                "sensor": token["sensor"], "acquired_at": token["acquired_at"],
                "quadrant": token["quadrant"], "visible_channels": [
                    CANONICAL_CHANNELS[i] for i, present in enumerate(token["observed"]) if present]}
                for token in sequence["tokens"]],
            "target_folds": list(batch["folds"]),
            "query_hidden_in_all_input_channels_per_fold": True,
            "attention_arrays_sha256": {name: _array_sha256(value) for name, value in arrays.items()},
            "limitations": ["Attention weights describe this selected checkpoint forward pass only.",
                "They are not causal attribution, variable importance, or independent validation."],
        }
        metadata_path = output / "representative_seed0_attention.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True,
            allow_nan=False) + "\n", encoding="utf-8")
        return {"status": metadata["status"], "arrays_path": target.name,
            "metadata_path": metadata_path.name, "query_observation_id": query["observation_id"],
            "checkpoint_sha256": checkpoint_sha256, "array_count": len(arrays)}
    return {"status": "UNAVAILABLE_NO_SOURCE_SUPPORTED_POST_EVENT_SOUTH_LHONAK_OPTICAL_QUERY",
            "arrays_path": None, "metadata_path": None, "query_observation_id": None,
            "checkpoint_sha256": checkpoint_sha256, "array_count": 0}


def _build_score_b_reference(model: TemporalMaskedMAEV2, panel: PanelV2,
                              fit_records: Sequence[Mapping[str, Any]],
                              fit_token_cache: Mapping[tuple[str, int], dict],
                              *, device: torch.device) -> tuple[dict, dict]:
    refs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for query in sorted(fit_records, key=lambda x: (x["site_id"], x["acquired_at"], x["sensor"], x["product_id"])):
        prefix = _context_prefix(fit_records, query)
        sequence = _sequence_from_prefix(prefix, fit_token_cache)
        embeddings = _encoder_token_embeddings(model, sequence, query, device=device)
        for item in embeddings:
            if item is None:
                continue
            refs[(query["site_id"], query["sensor"])].append({
                "observation_id": query["observation_id"],
                "product_id": query["product_id"], "acquired_at": query["acquired_at"],
                "quadrant": item["quadrant"], "embedding": item["embedding"],
                "visible_masked_views": item["visible_masked_views"],
                "source_sha256": query["sha256"],
            })
    states = {}
    projected = {}
    for key, rows in sorted(refs.items()):
        matrix = np.asarray([row["embedding"] for row in rows], dtype=np.float64)
        acquisition_keys = {(row["product_id"], row["acquired_at"]) for row in rows}
        state_key = f"{key[0]}::{key[1]}"
        if len(acquisition_keys) < SCORE_B_K + 1:
            states[state_key] = {"status": "UNAVAILABLE_INSUFFICIENT_DISTINCT_FIT_ACQUISITIONS",
                "fit_embedding_rows": len(rows), "distinct_fit_acquisitions": len(acquisition_keys),
                "required_distinct_acquisitions_after_exclusion": SCORE_B_K,
                "fit_observation_ids": sorted({x["observation_id"] for x in rows}),
                "pca_rank": None, "k": SCORE_B_K, "state_sha256": None}
            continue
        if matrix.ndim != 2 or matrix.shape[0] < SCORE_B_PCA_RANK + 1 or not np.isfinite(matrix).all():
            states[state_key] = {"status": "UNAVAILABLE_INSUFFICIENT_FINITE_FIT_EMBEDDINGS",
                "fit_embedding_rows": len(rows), "distinct_fit_acquisitions": len(acquisition_keys),
                "pca_rank": SCORE_B_PCA_RANK, "k": SCORE_B_K, "state_sha256": None}
            continue
        pca = PCA(n_components=SCORE_B_PCA_RANK, svd_solver="full")
        transformed = pca.fit_transform(matrix)
        state_payload = {
            "version": "SCORE_B_FIT_ONLY_PCA_KNN_V1", "site_id": key[0],
            "sensor": key[1], "pca_rank": SCORE_B_PCA_RANK, "k": SCORE_B_K,
            "fit_embedding_matrix_sha256": _array_sha256(matrix),
            "fit_observation_ids": sorted({x["observation_id"] for x in rows}),
            "fit_products": sorted({x["product_id"] for x in rows}),
            "fit_source_hashes": sorted({x["source_sha256"] for x in rows}),
            "fit_mean": pca.mean_.tolist(), "components": pca.components_.tolist(),
            "explained_variance": pca.explained_variance_.tolist(),
            "fit_projected_sha256": _array_sha256(transformed),
            "reference_support_count": len(rows),
            "distinct_fit_acquisition_count": len(acquisition_keys),
            "neighbor_exclusion": "same provider product OR same sensor acquisition timestamp",
            "neighbor_unit": "minimum quadrant distance per distinct fit acquisition",
            "state_source_role": "ALL_PRE_BREACH_FIT_SUPPORT_ONLY",
        }
        state_hash = _sha_json(state_payload)
        states[state_key] = {**state_payload, "status": "FITTED", "state_sha256": state_hash}
        projected[state_key] = {"pca": pca, "matrix": transformed, "rows": rows}
    return states, projected


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    import hashlib
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _soft_threshold_v2(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _singular_value_threshold_v2(values: np.ndarray, threshold: float) -> np.ndarray:
    left, singular, right = np.linalg.svd(values, full_matrices=False)
    retained = np.maximum(singular - threshold, 0.0)
    return (left * retained) @ right


def _rpca_kkt_residuals_v2(values: np.ndarray, low_rank: np.ndarray,
                           sparse: np.ndarray, dual: np.ndarray,
                           regularization: float, tolerance: float) -> dict:
    """Measure robust-PCA dual feasibility and subgradient KKT conditions."""
    left, singular, right = np.linalg.svd(low_rank, full_matrices=False)
    leading = float(singular[0]) if singular.size else 0.0
    rank = int(np.count_nonzero(singular > tolerance * max(leading, 1.0)))
    if rank:
        uv_transpose = left[:, :rank] @ right[:rank]
        remainder = dual - uv_transpose
        left_orthogonality = float(np.linalg.norm(left[:, :rank].T @ remainder, ord=2))
        right_orthogonality = float(np.linalg.norm(remainder @ right[:rank].T, ord=2))
    else:
        remainder = dual
        left_orthogonality = 0.0
        right_orthogonality = 0.0
    nonzero_sparse = sparse != 0
    zero_sparse = ~nonzero_sparse
    sparse_nonzero_residual = (float(np.max(np.abs(
        dual[nonzero_sparse] - regularization * np.sign(sparse[nonzero_sparse]))))
        if np.any(nonzero_sparse) else 0.0)
    sparse_zero_violation = (float(np.max(np.maximum(
        np.abs(dual[zero_sparse]) - regularization, 0.0)))
        if np.any(zero_sparse) else 0.0)
    spectral_dual = float(np.linalg.norm(dual, ord=2))
    infinity_dual = float(np.max(np.abs(dual)))
    remainder_spectral = float(np.linalg.norm(remainder, ord=2))
    value_norm = float(np.linalg.norm(values, ord="fro"))
    primal_relative = (float(np.linalg.norm(values - low_rank - sparse, ord="fro") / value_norm)
                       if value_norm else 0.0)
    return {
        "primal_feasibility_relative_residual": primal_relative,
        "nuclear_dual_spectral_norm": spectral_dual,
        "nuclear_dual_feasibility_violation": max(0.0, spectral_dual - 1.0),
        "nuclear_subgradient_remainder_spectral_norm": remainder_spectral,
        "nuclear_subgradient_remainder_violation": max(0.0, remainder_spectral - 1.0),
        "nuclear_subgradient_left_orthogonality_residual": left_orthogonality,
        "nuclear_subgradient_right_orthogonality_residual": right_orthogonality,
        "sparse_dual_infinity_norm": infinity_dual,
        "sparse_dual_feasibility_violation": max(0.0, infinity_dual - regularization),
        "sparse_subgradient_nonzero_residual": sparse_nonzero_residual,
        "sparse_subgradient_zero_violation": sparse_zero_violation,
    }


def _complete_data_rpca_v2(values: np.ndarray, *, max_iterations: int,
                           tolerance: float) -> dict:
    """Fixed-penalty ADMM for complete matrices with a feasible dual start.

    The objective is the standard convex principal-component pursuit objective
    ``||L||_* + lambda ||S||_1`` subject to ``D = L + S``. The fixed penalty is
    scaled to the input spectral norm; convergence requires both normalized
    primal and ADMM dual residuals to meet the declared tolerance.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or min(data.shape) < 2:
        raise ValueError("complete-data v2 RPCA requires a two-dimensional matrix with at least two rows and columns")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("complete-data v2 RPCA iteration budget must be a positive integer")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("complete-data v2 RPCA tolerance must be positive and finite")
    if not np.isfinite(data).all():
        raise ValueError("complete-data v2 RPCA requires finite values in every matrix cell")
    norm_frobenius = float(np.linalg.norm(data, ord="fro"))
    if norm_frobenius == 0:
        zeros = np.zeros_like(data)
        return {"status": "UNAVAILABLE_ZERO_OBSERVED_NORM",
            "algorithm": "COMPLETE_DATA_FIXED_PENALTY_ADMM_RPCA_V2",
            "complete_data": True, "low_rank": zeros, "sparse": zeros.copy(),
            "dual_certificate": zeros.copy(), "iterations": 0,
            "primal_residual": None, "dual_residual": None, "rank": 0,
            "sparsity": 0.0, "trace": [], "tolerance": tolerance,
            "lambda": 1.0 / math.sqrt(max(data.shape)),
            "fixed_penalty_factor": RPCA_ADMM_PENALTY_FACTOR,
            "fixed_penalty_mu": None, "kkt_residuals": None,
            "observed_entry_count": int(data.size), "missing_entry_count": 0}
    regularization = 1.0 / math.sqrt(max(data.shape))
    spectral_norm = float(np.linalg.norm(data, ord=2))
    infinity_norm = float(np.max(np.abs(data)))
    if spectral_norm == 0 or not math.isfinite(spectral_norm):
        raise ValueError("complete-data v2 RPCA spectral norm is invalid")
    # This standard PCP initialization is dual feasible for both norm balls.
    dual = data / max(spectral_norm, infinity_norm / regularization)
    penalty = RPCA_ADMM_PENALTY_FACTOR / spectral_norm
    low_rank = np.zeros_like(data)
    sparse = np.zeros_like(data)
    trace = []
    primal_residual = math.inf
    dual_residual = math.inf
    converged = False
    for iteration in range(1, max_iterations + 1):
        previous_sparse = sparse.copy()
        low_rank = _singular_value_threshold_v2(
            data - sparse + dual / penalty, 1.0 / penalty)
        sparse = _soft_threshold_v2(
            data - low_rank + dual / penalty, regularization / penalty)
        residual = data - low_rank - sparse
        dual = dual + penalty * residual
        primal_residual = float(np.linalg.norm(residual, ord="fro") / norm_frobenius)
        dual_residual = float(penalty * np.linalg.norm(
            sparse - previous_sparse, ord="fro") / norm_frobenius)
        trace.append({"iteration": iteration,
            "primal_residual": primal_residual, "dual_residual": dual_residual,
            "rank": int(np.linalg.matrix_rank(low_rank)),
            "sparse_observed_nonzero": int(np.count_nonzero(sparse))})
        if primal_residual <= tolerance and dual_residual <= tolerance:
            converged = True
            break
    singular = np.linalg.svd(low_rank, compute_uv=False)
    rank = int(np.sum(singular > tolerance * max(
        float(singular[0]) if singular.size else 0.0, 1.0)))
    kkt_residuals = _rpca_kkt_residuals_v2(
        data, low_rank, sparse, dual, regularization, tolerance)
    return {"status": "CONVERGED" if converged else "NONCONVERGED_MAX_ITERATIONS",
        "algorithm": "COMPLETE_DATA_FIXED_PENALTY_ADMM_RPCA_V2",
        "complete_data": True, "low_rank": low_rank, "sparse": sparse,
        "dual_certificate": dual, "iterations": len(trace),
        "primal_residual": primal_residual, "dual_residual": dual_residual,
        "rank": rank,
        "sparsity": float(np.count_nonzero(sparse) / data.size), "trace": trace,
        "tolerance": tolerance, "lambda": regularization,
        "fixed_penalty_factor": RPCA_ADMM_PENALTY_FACTOR,
        "fixed_penalty_mu": penalty, "kkt_residuals": kkt_residuals,
        "observed_entry_count": int(data.size), "missing_entry_count": 0}


def _rpca_fit_rows_v2(rows: Sequence[Mapping[str, Any]],
                      native_indices: Sequence[int]) -> tuple[list[Mapping[str, Any]], list[str]]:
    """Omit rows with no objective support; retain partial rows for masked fit."""
    selected = []
    excluded = []
    for row in rows:
        support = np.asarray(row["observed"], dtype=bool)[list(native_indices)]
        if support.any():
            selected.append(row)
        else:
            excluded.append(row["observation_id"])
    return selected, excluded


def _pooled_pixel_channel_count(pooled: Mapping[str, Any]) -> int:
    return int(sum(count or 0 for token in pooled["tokens"]
                   for count in token["source_pixel_denominators"]))


def _score_c_joint_quadrants(score_a: Mapping[str, Any],
                             score_b: Mapping[str, Any]) -> list[int]:
    a_quadrants = {row["quadrant"] for row in score_a.get("quadrant_folds", [])
                   if row.get("target_channel_count", 0) > 0}
    b_quadrants = {row["quadrant"] for row in score_b.get("quadrant_distances", [])
                   if row.get("score") is not None}
    return sorted(a_quadrants & b_quadrants)


def _score_b_query(query: Mapping[str, Any], embeddings: Sequence[dict | None],
                   projected_refs: Mapping[str, dict],
                   states: Mapping[str, Mapping[str, Any]]) -> dict:
    key = f"{query['site_id']}::{query['sensor']}"
    fitted = projected_refs.get(key)
    if fitted is None:
        state = states.get(key)
        status = state["status"] if state else "UNAVAILABLE_NO_FIT_REFERENCE_STATE"
        return {"score": None, "status": status, "denominator": 0,
                "denominator_unit": "query_quadrants_with_five_distinct_fit_neighbors",
                "quadrant_distances": [], "distinct_reference_acquisitions": 0,
                "neighbor_exclusion_applied": True}
    pca = fitted["pca"]
    ref_matrix = fitted["matrix"]
    ref_rows = fitted["rows"]
    query_distances = []
    for item in embeddings:
        if item is None:
            continue
        point = pca.transform(np.asarray(item["embedding"], dtype=np.float64).reshape(1, -1))[0]
        by_acquisition: dict[tuple[str, str], float] = {}
        for row, reference in zip(ref_matrix, ref_rows):
            same_product = reference["product_id"] == query["product_id"]
            same_sensor_arrival = (reference["acquired_at"] == query["acquired_at"] and
                                   reference.get("sensor", query["sensor"]) == query["sensor"])
            if same_product or same_sensor_arrival:
                continue
            acquisition_key = (reference["product_id"], reference["acquired_at"])
            distance = float(np.linalg.norm(point - row))
            by_acquisition[acquisition_key] = min(by_acquisition.get(acquisition_key, math.inf), distance)
        nearest = sorted(by_acquisition.values())[:SCORE_B_K]
        if len(nearest) < SCORE_B_K:
            query_distances.append({"quadrant": item["quadrant"], "score": None,
                "status": "UNAVAILABLE_INSUFFICIENT_DISTINCT_FIT_NEIGHBORS",
                "distinct_reference_acquisitions": len(nearest)})
        else:
            query_distances.append({"quadrant": item["quadrant"],
                "score": float(np.mean(nearest)), "status": "ESTIMATED",
                "distinct_reference_acquisitions": len(nearest)})
    supported = [x for x in query_distances if x["score"] is not None]
    score = float(np.mean([x["score"] for x in supported])) if supported else None
    if score is None:
        status = "UNAVAILABLE_INSUFFICIENT_DISTINCT_FIT_NEIGHBORS"
    elif len(supported) < 4:
        status = "ESTIMATED_PARTIAL_QUADRANT_SUPPORT"
    else:
        status = "ESTIMATED"
    return {"score": score, "status": status, "denominator": len(supported),
            "denominator_unit": "query_quadrants_with_five_distinct_fit_neighbors",
            "quadrant_distances": query_distances,
            "distinct_reference_acquisitions": min((x["distinct_reference_acquisitions"] for x in supported), default=0),
            "neighbor_exclusion_applied": True}


def _score_a(model: TemporalMaskedMAEV2, panel: PanelV2,
             query: Mapping[str, Any], all_records: Sequence[Mapping[str, Any]],
             token_cache: Mapping[tuple[str, int], dict], *, parity: int | None,
             device: torch.device) -> dict:
    prefix = _context_prefix(all_records, query)
    sequence = _sequence_from_prefix(prefix, token_cache)
    positions = [i for i, owner in enumerate(sequence["owner_ids"])
                 if owner == query["observation_id"]]
    if len(positions) != 4:
        raise ValueError("Score A query does not have all four quadrants")
    batch = _fold_batch(sequence, query["observation_id"], device)
    if batch is None:
        return {"observation_id": query["observation_id"], "source_sha256": query["sha256"],
                "score": None, "status": "UNAVAILABLE_NO_GENUINE_TARGET_SUPPORT",
                "denominator": 0, "denominator_unit": "observed_target_channel_quadrant_cells",
                "squared_error_sum": None,
                "quadrant_folds": [{"quadrant": q, "status": "UNAVAILABLE_NO_TARGET_SUPPORT",
                                    "target_channel_count": 0,
                                    "source_pixel_denominators": sequence["tokens"][positions[q]]["source_pixel_denominators"],
                                    "squared_error_sum": None, "mean_squared_error": None,
                                    "target_values": [None] * 7, "predicted_values": [None] * 7}
                                   for q in range(4)]}
    model.eval()
    with torch.no_grad():
        prediction, _ = model(batch["values"], batch["observed"], batch["modality"],
                              batch["time"], batch["quadrant"], batch["target"])
    folds_by_q = {}
    total_squared_error = 0.0
    total_count = 0
    for bi, q in enumerate(batch["folds"]):
        pos = positions[q]
        support = batch["target"][bi, pos].cpu().numpy()
        truth = batch["target_values"][bi, pos].detach().cpu().numpy().astype(np.float64)
        estimate = prediction[bi, pos].detach().cpu().numpy().astype(np.float64)
        token = sequence["tokens"][pos]
        observed_indices = np.flatnonzero(support)
        squared_error_sum = float(np.square(estimate[support] - truth[support]).sum(dtype=np.float64))
        count = int(support.sum())
        total_squared_error += squared_error_sum; total_count += count
        truth_values = [float(truth[i]) if support[i] else None for i in range(7)]
        prediction_values = [float(estimate[i]) if support[i] else None for i in range(7)]
        denominators = token["source_pixel_denominators"]
        folds_by_q[q] = {"quadrant": q, "status": "ESTIMATED" if count else "UNAVAILABLE_NO_TARGET_SUPPORT",
            "target_channel_count": count, "target_channel_ids": [CANONICAL_CHANNELS[i] for i in observed_indices],
            "target_values": truth_values, "predicted_values": prediction_values,
            "source_pixel_denominators": denominators,
            "squared_error_sum": squared_error_sum if count else None,
            "mean_squared_error": squared_error_sum / count if count else None}
    for q in range(4):
        if q not in folds_by_q:
            token = sequence["tokens"][positions[q]]
            folds_by_q[q] = {"quadrant": q, "status": "UNAVAILABLE_NO_TARGET_SUPPORT",
                "target_channel_count": 0, "target_channel_ids": [],
                "target_values": [None] * 7, "predicted_values": [None] * 7,
                "source_pixel_denominators": token["source_pixel_denominators"],
                "squared_error_sum": None, "mean_squared_error": None}
    score = total_squared_error / total_count if total_count else None
    return {"observation_id": query["observation_id"], "source_sha256": query["sha256"],
        "score": score,
        "status": "ESTIMATED" if total_count else "UNAVAILABLE_NO_GENUINE_TARGET_SUPPORT",
        "denominator": total_count,
        "denominator_unit": "observed_target_channel_quadrant_cells",
        "squared_error_sum": total_squared_error if total_count else None,
        "quadrant_folds": [folds_by_q[q] for q in range(4)],
        "parity": parity,
        "target_policy": "four quadrant folds; query quadrant hidden across all channels before pooling",
        "sequence_window": {"first_acquired_at": prefix[0]["acquired_at"],
            "last_acquired_at": query["acquired_at"],
            "source_observation_ids": [x["observation_id"] for x in prefix],
            "future_observation_count": 0}}


def verify_score_a_components(query: Mapping[str, Any], pooled: Mapping[str, Any],
                              score_record: Mapping[str, Any], *,
                              atol: float = 1e-6) -> None:
    """Recompute target support and Score A arithmetic from source-pixel rows."""
    if score_record.get("observation_id") != query["observation_id"]:
        raise ValueError("Score A row is bound to a different observation")
    if score_record.get("source_sha256") != query["sha256"]:
        raise ValueError("Score A row is bound to different source bytes")
    tokens = {x["quadrant"]: x for x in pooled["tokens"]}
    error_sum = 0.0; denominator = 0
    for fold in score_record["quadrant_folds"]:
        q = fold["quadrant"]
        token = tokens[q]
        mask = token["observed"]
        expected_denominators = token["source_pixel_denominators"]
        if fold["source_pixel_denominators"] != expected_denominators:
            raise ValueError("Score A quadrant denominator differs from source pixel support")
        if fold["status"] == "UNAVAILABLE_NO_TARGET_SUPPORT":
            if mask.any() or fold["target_channel_count"] != 0:
                raise ValueError("Score A declared an observed target quadrant unavailable")
            continue
        truth = fold["target_values"]
        prediction = fold["predicted_values"]
        if len(truth) != 7 or len(prediction) != 7:
            raise ValueError("Score A component rows do not match the seven-channel axis")
        for channel in range(7):
            if mask[channel]:
                if truth[channel] is None or prediction[channel] is None:
                    raise ValueError("Score A omitted a genuinely supported target value")
                if not math.isclose(float(truth[channel]), float(token["values"][channel]),
                                    rel_tol=0, abs_tol=atol):
                    raise ValueError("Score A target value does not replay from source pooling")
                error_sum += (float(prediction[channel]) - float(truth[channel])) ** 2
                denominator += 1
            elif truth[channel] is not None or prediction[channel] is not None:
                raise ValueError("Score A manufactured an unsupported channel target")
        if fold["target_channel_count"] != int(mask.sum()):
            raise ValueError("Score A target channel count differs from source support")
        if mask.any() and not math.isclose(fold["squared_error_sum"],
                sum((float(prediction[i]) - float(truth[i])) ** 2 for i in range(7) if mask[i]),
                rel_tol=0, abs_tol=atol):
            raise ValueError("Score A per-quadrant squared error does not recompute")
    expected_score = error_sum / denominator if denominator else None
    if denominator != score_record["denominator"]:
        raise ValueError("Score A overall denominator differs from source target support")
    if expected_score is None:
        if score_record["score"] is not None:
            raise ValueError("Score A must be null when no target support exists")
    elif not math.isclose(expected_score, score_record["score"], rel_tol=0, abs_tol=atol):
        raise ValueError("Score A overall mean squared error does not recompute")


def _fit_score_standardizer(rows: Sequence[Mapping[str, Any]], method: str) -> dict:
    fit_rows = [row for row in rows if
        row.get("scope_role") == "FIT_SUPPORT_WITH_PARITY1_DEVELOPMENT_MONITOR" and
        row.get("method") == method and row.get("score") is not None and
        row.get("query_parity") == 0 and row.get("fitting_parity") == 0]
    values = [float(row["score"]) for row in fit_rows]
    identities = [row["observation_id"] for row in fit_rows]
    if len(values) < 2:
        return {"status": "UNAVAILABLE_INSUFFICIENT_FIT_SCORE_SUPPORT", "mean": None,
                "scale": None, "count": len(values), "fit_observation_ids": identities,
                "state_sha256": None}
    mean = float(np.mean(values, dtype=np.float64))
    scale = float(np.std(values, ddof=1, dtype=np.float64))
    if not math.isfinite(scale) or scale <= 0:
        return {"status": "UNAVAILABLE_DEGENERATE_FIT_SCORE_SCALE", "mean": None,
                "scale": None, "count": len(values), "fit_observation_ids": identities,
                "state_sha256": None}
    payload = {"version": "FIT_ONLY_SCORE_STANDARDIZER_V1", "method": method,
        "mean": mean, "scale": scale, "count": len(values),
        "fit_observation_ids": identities,
        "fit_observation_ids_sha256": _sha_json(identities),
        "fit_role": "ALL_PRE_BREACH_FIT_PARITY0_ONLY"}
    return {**payload, "status": "FITTED", "state_sha256": _sha_json(payload)}


def _baseline_row(record: Mapping[str, Any], pooled: Mapping[str, Any],
                  transform: Mapping[str, Any], *, role: str,
                  query_parity: int) -> dict:
    if role not in {"fit", "monitor"} or query_parity not in (0, 1):
        raise ValueError("baseline rows require an explicit fit/monitor role and parity")
    values = np.zeros(7, dtype=np.float64)
    observed = np.zeros(7, dtype=bool)
    denominators = [None] * 7
    channel_index = {name: i for i, name in enumerate(CANONICAL_CHANNELS)}
    for channel in record["channels"]:
        index = channel_index[channel]
        weighted_sum = 0.0; total = 0
        for token in pooled["tokens"]:
            count = token["source_pixel_denominators"][index]
            if count is not None and count > 0 and token["observed"][index]:
                weighted_sum += float(token["values"][index]) * count
                total += count
        denominators[index] = total
        if total:
            values[index] = weighted_sum / total
            observed[index] = True
    native = list(record["channels"])
    native_denominators = [int(denominators[channel_index[x]]) for x in native]
    unavailable = [transform["scales"][channel_index[x]] is None for x in native]
    support = _sensor_native_support_fields(record["sensor"], native, native_denominators,
                                             unavailable, observed)
    return {"row_id": record["observation_id"], "observation_id": record["observation_id"],
        "role": role, "query_parity": query_parity, "fitting_parity": 0,
        "scope_role": record["scope"]["role"], "site_id": record["site_id"],
        "panel_id": "SOURCE_BOUND_PANEL_V2", "group_id": record["observation_id"],
        "source_id": record["product_id"], "support_id": record["observation_id"],
        "product_id": record["product_id"], "sensor": record["sensor"],
        "source_sha256": record["sha256"],
        "acquisition_at": record["acquired_at"], "arrival_at": record["acquired_at"],
        "timezone_status": "EXPLICIT_UTC", "sampling_unit_status": "UNKNOWN",
        "values": values, "observed": observed,
        "source_pixel_denominators": native_denominators,
        "query_pixel_channel_count": int(sum(native_denominators)),
        "full_source_valid_pixel_channel_count": int(sum(record["observed_valid_counts_by_channel"])),
        "unavailable_fit_transform_channels": unavailable,
        "channel_order": native, "canonical_channel_order": list(CANONICAL_CHANNELS),
        **support}


def _null_method_row(method: str, record: Mapping[str, Any], reason: str,
                     denominator: int = 0, *, query_parity: int = 1,
                     fitting_parity: int = 0,
                     query_pixel_channel_count: int | None = None,
                     denominator_unit: str | None = None) -> dict:
    return {"method": method, "observation_id": record["observation_id"],
        "site_id": record["site_id"], "sensor": record["sensor"],
        "product_id": record["product_id"], "acquired_at": record["acquired_at"],
        "scope_role": record["scope"]["role"], "event_phase": record["scope"]["event_phase"],
        "score": None, "status": reason, "reason": reason,
        "denominator": denominator, "denominator_unit": denominator_unit,
        "query_pixel_channel_count": query_pixel_channel_count,
        "full_source_valid_pixel_channel_count": int(sum(
            record["observed_valid_counts_by_channel"])), "seed": None,
        "query_parity": query_parity, "fitting_parity": fitting_parity}


def _standardized_score(value: float | None, state: Mapping[str, Any]) -> float | None:
    if value is None or state.get("status") != "FITTED":
        return None
    return (float(value) - float(state["mean"])) / float(state["scale"])


def _fit_baselines(panel: PanelV2, transform: Mapping[str, Any],
                   pooled_fit: Mapping[str, dict],
                   pooled_query: Mapping[str, dict]) -> tuple[dict, list[dict], list[dict]]:
    fit_records = [record for record in panel.observations
                   if record["scope"]["role"] == "FIT_SUPPORT_WITH_PARITY1_DEVELOPMENT_MONITOR"]
    fit_rows = [_baseline_row(record, pooled_fit[record["observation_id"]], transform,
                              role="fit", query_parity=0) for record in fit_records]
    query_rows = [_baseline_row(record, pooled_query[record["observation_id"]], transform,
                                role="monitor", query_parity=1)
                  for record in panel.observations]
    fit_support_by_id = {row["row_id"]: row for row in fit_rows}
    query_support_by_id = {row["row_id"]: row for row in query_rows}
    record_by_id = {record["observation_id"]: record for record in panel.observations}
    states: dict[str, Any] = {}
    output: list[dict] = []
    fit_if_score_rows: list[dict] = []
    for sensor in CHANNELS_BY_SENSOR:
        sensor_fit = [row for row in fit_rows if row["sensor"] == sensor]
        sensor_query = [row for row in query_rows if row["sensor"] == sensor]
        if not sensor_query:
            continue
        if_state, if_scores, _ = fit_isolation_forest(sensor_fit, sensor_query,
            transform["state_sha256"], seed=20261015, n_estimators=100)
        svm_state, svm_scores, _ = fit_psd_rbf_ocsvm(sensor_fit, sensor_query,
            transform["state_sha256"], nu=0.1)
        state = {"isolation_forest": if_state, "psd_rbf_one_class_svm": svm_state}
        native_channels = list(CHANNELS_BY_SENSOR[sensor])
        native_indices = [CANONICAL_CHANNELS.index(x) for x in native_channels]
        rpca_fit, excluded_empty_fit_ids = _rpca_fit_rows_v2(sensor_fit, native_indices)

        def score_row(method: str, estimator_row: Mapping[str, Any], parity: int) -> dict:
            source = record_by_id[estimator_row["row_id"]]
            support_row = (fit_support_by_id if parity == 0 else query_support_by_id)[
                estimator_row["row_id"]]
            return {"method": method, "observation_id": source["observation_id"],
                "site_id": source["site_id"], "sensor": sensor,
                "product_id": source["product_id"], "acquired_at": source["acquired_at"],
                "scope_role": source["scope"]["role"],
                "event_phase": source["scope"]["event_phase"],
                "score": estimator_row["score"], "status": estimator_row["status"],
                "reason": estimator_row["reason"],
                "denominator": estimator_row["observed_channel_denominator"],
                "denominator_unit": "observed_sensor_channels",
                "query_pixel_channel_count": support_row["query_pixel_channel_count"],
                "full_source_valid_pixel_channel_count": support_row["full_source_valid_pixel_channel_count"],
                "seed": None, "query_parity": parity, "fitting_parity": 0,
                "fit_state_sha256": estimator_row.get("fit_transform_sha256"),
                "sensor_stratified_fit": True}

        fit_if_score_rows.extend(score_row("isolation_forest", row, 0)
            for row in if_scores if row["role"] == "fit")
        if len(rpca_fit) >= 2 and len(native_indices) >= 2:
            fit_values = np.asarray([[row["values"][i] for i in native_indices] for row in rpca_fit], dtype=np.float64)
            fit_masks = np.asarray([[row["observed"][i] for i in native_indices] for row in rpca_fit], dtype=bool)
            if fit_masks.all():
                rpca = _complete_data_rpca_v2(fit_values,
                    max_iterations=RPCA_MAX_ITERATIONS, tolerance=RPCA_TOLERANCE)
            else:
                rpca = inexact_alm_rpca(fit_values, fit_masks,
                    max_iterations=RPCA_MAX_ITERATIONS, tolerance=RPCA_TOLERANCE)
            rpca_hash_payload = {key: value for key, value in rpca.items()
                if key not in {"low_rank", "sparse", "trace", "dual_certificate"}}
            rpca_hash_payload.update({"fit_observation_ids": [x["observation_id"] for x in rpca_fit],
                "excluded_fit_rows_with_no_native_channel_support": excluded_empty_fit_ids,
                "fit_source_hashes": [x["source_sha256"] for x in rpca_fit],
                "fit_values_sha256": _array_sha256(fit_values),
                "fit_masks_sha256": _array_sha256(fit_masks),
                "fit_low_rank_sha256": _array_sha256(rpca["low_rank"]),
                "fit_sparse_sha256": _array_sha256(rpca["sparse"])})
            rpca_state = {"status": rpca["status"], "algorithm": rpca["algorithm"],
                "iterations": rpca["iterations"], "tolerance": RPCA_TOLERANCE,
                "max_iterations": RPCA_MAX_ITERATIONS, "rank": rpca["rank"],
                "primal_residual": rpca["primal_residual"], "dual_residual": rpca["dual_residual"],
                "fit_observation_ids": [x["observation_id"] for x in rpca_fit],
                "excluded_fit_rows_with_no_native_channel_support": excluded_empty_fit_ids,
                "fit_values_sha256": rpca_hash_payload["fit_values_sha256"],
                "fit_masks_sha256": rpca_hash_payload["fit_masks_sha256"],
                "complete_data": rpca.get("complete_data"),
                "observed_entry_count": rpca.get("observed_entry_count"),
                "missing_entry_count": rpca.get("missing_entry_count"),
                "lambda": rpca.get("lambda"),
                "fixed_penalty_factor": rpca.get("fixed_penalty_factor"),
                "fixed_penalty_mu": rpca.get("fixed_penalty_mu"),
                "kkt_residuals": rpca.get("kkt_residuals"),
                "low_rank": rpca["low_rank"].tolist(), "sparse": rpca["sparse"].tolist(),
                "trace": rpca["trace"], "state_sha256": _sha_json(rpca_hash_payload)}
            rpca_scores = []
            for row in sensor_query:
                vector = np.asarray([row["values"][i] for i in native_indices], dtype=np.float64)
                mask = np.asarray([row["observed"][i] for i in native_indices], dtype=bool)
                score_result = rpca_project_score(rpca, vector[None, :], mask[None, :])["scores"][0]
                source = record_by_id[row["row_id"]]
                rpca_scores.append({"method": "inexact_alm_rpca", "observation_id": source["observation_id"],
                    "site_id": source["site_id"], "sensor": sensor, "product_id": source["product_id"],
                    "acquired_at": source["acquired_at"], "scope_role": source["scope"]["role"],
                    "event_phase": source["scope"]["event_phase"],
                    "score": score_result["score"], "status": score_result["status"],
                    "reason": None if score_result["score"] is not None else score_result["status"],
                    "denominator": score_result["denominator"],
                    "denominator_unit": "observed_sensor_channels",
                    "query_pixel_channel_count": row["query_pixel_channel_count"],
                    "full_source_valid_pixel_channel_count": row["full_source_valid_pixel_channel_count"],
                    "seed": None, "query_parity": 1, "fitting_parity": 0,
                    "fit_convergence_status": rpca["status"], "fit_state_sha256": rpca_state["state_sha256"]})
            state["inexact_alm_rpca"] = rpca_state
        else:
            state["inexact_alm_rpca"] = {"status": "UNAVAILABLE_INSUFFICIENT_SENSOR_STRATIFIED_FIT_ROWS",
                "fit_rows": len(rpca_fit), "excluded_fit_rows_with_no_native_channel_support": excluded_empty_fit_ids,
                "native_channels": native_channels,
                "max_iterations": RPCA_MAX_ITERATIONS, "tolerance": RPCA_TOLERANCE,
                "state_sha256": None}
            rpca_scores = [_null_method_row("inexact_alm_rpca", record_by_id[row["row_id"]],
                "UNAVAILABLE_INSUFFICIENT_SENSOR_STRATIFIED_FIT_ROWS",
                query_pixel_channel_count=row["query_pixel_channel_count"],
                denominator_unit="observed_sensor_channels") for row in sensor_query]
        states[sensor] = state

        def choose_score_rows(scores, method):
            chosen = {row["row_id"]: row for row in scores if row["role"] == "monitor"}
            for record in panel.observations:
                if record["sensor"] != sensor:
                    continue
                source_row = chosen.get(record["observation_id"])
                if source_row is None:
                    support_row = query_support_by_id[record["observation_id"]]
                    output.append(_null_method_row(method, record,
                        "UNAVAILABLE_INSUFFICIENT_SENSOR_STRATIFIED_FIT_ROWS",
                        query_pixel_channel_count=support_row["query_pixel_channel_count"],
                        denominator_unit="observed_sensor_channels"))
                    continue
                output.append(score_row(method, source_row, 1))

        choose_score_rows(if_scores, "isolation_forest")
        choose_score_rows(svm_scores, "psd_rbf_one_class_svm")
        output.extend(rpca_scores)
    return states, output, fit_if_score_rows


def _cusum_rows(panel: PanelV2, if_query_rows: Sequence[Mapping[str, Any]],
                if_fit_rows: Sequence[Mapping[str, Any]]) -> tuple[dict, list[dict]]:
    query_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    fit_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_id = {row["observation_id"]: row for row in panel.observations}
    for row in if_query_rows:
        query_groups[(row["site_id"], row["sensor"])].append(dict(row))
    for row in if_fit_rows:
        if row.get("query_parity") == 0 and row.get("fitting_parity") == 0:
            fit_groups[(row["site_id"], row["sensor"])].append(dict(row))
    states = {}
    outputs = []
    for key, rows in sorted(query_groups.items()):
        rows.sort(key=lambda x: x["acquired_at"])
        fit_rows = [row for row in fit_groups.get(key, [])
                    if row.get("scope_role") == "FIT_SUPPORT_WITH_PARITY1_DEVELOPMENT_MONITOR" and
                    row.get("score") is not None]
        identity = f"{key[0]}::{key[1]}"
        if len(fit_rows) < 2:
            state = {"status": "UNAVAILABLE_INSUFFICIENT_FIT_SCORES", "center": None,
                     "scale": None, "fit_observation_ids": [x["observation_id"] for x in fit_rows],
                     "fit_query_parity": 0, "published_query_parity": 1}
            states[identity] = state
            outputs.extend(_null_method_row("cusum", by_id[row["observation_id"]], state["status"],
                query_pixel_channel_count=row.get("query_pixel_channel_count"),
                denominator_unit="observed_sensor_channels")
                           for row in rows)
            continue
        fit_values = np.asarray([row["score"] for row in fit_rows], dtype=np.float64)
        center = float(fit_values.mean()); scale = float(fit_values.std(ddof=1))
        if scale <= 0 or not math.isfinite(scale):
            state = {"status": "UNAVAILABLE_DEGENERATE_FIT_SCORE_SCALE", "center": None,
                     "scale": None, "fit_observation_ids": [x["observation_id"] for x in fit_rows],
                     "fit_query_parity": 0, "published_query_parity": 1}
            states[identity] = state
            outputs.extend(_null_method_row("cusum", by_id[row["observation_id"]], state["status"],
                query_pixel_channel_count=row.get("query_pixel_channel_count"),
                denominator_unit="observed_sensor_channels")
                           for row in rows)
            continue
        arrivals = [datetime.fromisoformat(row["acquired_at"].replace("Z", "+00:00")) for row in rows]
        if len(set(arrivals)) != len(arrivals):
            state = {"status": "UNAVAILABLE_DUPLICATE_ACTUAL_ARRIVAL_TIME", "center": center,
                     "scale": scale, "fit_observation_ids": [x["observation_id"] for x in fit_rows],
                     "fit_query_parity": 0, "published_query_parity": 1}
            states[identity] = state
            outputs.extend(_null_method_row("cusum", by_id[row["observation_id"]], state["status"],
                query_pixel_channel_count=row.get("query_pixel_channel_count"),
                denominator_unit="observed_sensor_channels")
                           for row in rows)
            continue
        pos = neg = 0.0
        previous = None
        row_results = []
        for row, arrival in zip(rows, arrivals):
            gap = (arrival - previous).total_seconds() / 86400.0 if previous is not None else None
            if gap is not None and gap > 90.0:
                pos = neg = 0.0
            score = row["score"]
            if score is None:
                pos = neg = 0.0
                z = None; status = "ABSTAIN_RESET_MISSING_SCORE"
            else:
                z = (float(score) - center) / scale
                pos = max(0.0, pos + z - 0.5)
                neg = min(0.0, neg + z + 0.5)
                status = "ESTIMATED"
            row_results.append({"method": "cusum", "observation_id": row["observation_id"],
                "site_id": row["site_id"], "sensor": row["sensor"], "product_id": row["product_id"],
                "acquired_at": row["acquired_at"], "scope_role": row["scope_role"],
                "event_phase": row["event_phase"], "score": pos if z is not None else None,
                "negative_cusum": neg if z is not None else None,
                "standardized_input_score": z, "status": status,
                "reason": None if z is not None else status, "denominator": row["denominator"],
                "denominator_unit": "observed_sensor_channels",
                "query_pixel_channel_count": row.get("query_pixel_channel_count"),
                "full_source_valid_pixel_channel_count": row.get("full_source_valid_pixel_channel_count"),
                "seed": None, "fit_standardizer_sha256": _sha_json({"center": center, "scale": scale,
                    "fit_observation_ids": [x["observation_id"] for x in fit_rows]}),
                "query_parity": 1, "fitting_parity": 0,
                "gap_days": gap, "reference_k": 0.5, "max_gap_days": 90})
            previous = arrival
        state_payload = {"version": "CAUSAL_SITE_SENSOR_CUSUM_V1", "center": center,
            "scale": scale, "fit_observation_ids": [x["observation_id"] for x in fit_rows],
            "fit_role": "ALL_PRE_BREACH_FIT_PARITY0_ONLY", "fit_query_parity": 0,
            "published_query_parity": 1, "fitting_parity": 0,
            "input_method": "ISOLATION_FOREST",
            "reference_k": 0.5, "max_gap_days": 90}
        states[identity] = {**state_payload, "status": "FITTED", "state_sha256": _sha_json(state_payload)}
        outputs.extend(row_results)
    return states, outputs


def _descriptive_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("site_id"), row.get("sensor"), row.get("method"),
                row.get("scope_role"), row.get("event_phase"), row.get("seed"))].append(row)
    summary = []
    for key, members in sorted(groups.items(), key=lambda x: tuple(str(y) for y in x[0])):
        values = [float(row["score"]) for row in members
                  if row.get("score") is not None and math.isfinite(float(row["score"]))]
        summary.append({"site_id": key[0], "sensor": key[1], "method": key[2],
            "scope_role": key[3], "event_phase": key[4], "seed": key[5],
            "observation_rows": len(members), "estimated_rows": len(values),
            "mean_score": float(np.mean(values)) if values else None,
            "median_score": float(np.median(values)) if values else None,
            "status": "DESCRIPTIVE_ONLY_NO_INDEPENDENT_SUPPORT_OR_EVENT_LABEL_CLAIM"})
    return summary


def run_inference_v2(panel: PanelV2, training_dir: str | Path,
                     output: str | Path) -> dict:
    """Score parity-1 query rows against parity-0 pre-breach fitted states."""
    training_dir = Path(training_dir)
    if not training_dir.is_absolute():
        training_dir = panel.root / training_dir
    training_dir = training_dir.resolve(strict=True)
    training_report_path = training_dir / "training_report.v2.json"
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    if training_report.get("source_manifest_sha256") != panel.manifest_sha256:
        raise ValueError("training source manifest differs from inference panel")
    if training_report.get("domain_protocol_sha256") != panel.domain_protocol_sha256:
        raise ValueError("training domain protocol differs from inference panel")
    transform_path = training_dir / "fit_transform.v2.json"
    transform = json.loads(transform_path.read_text(encoding="utf-8"))
    if sha256_file(transform_path) != training_report.get("fit_transform_file_sha256"):
        raise ValueError("frozen fit-transform bytes do not match training report")
    if transform.get("state_sha256") != training_report.get("fit_transform_sha256"):
        raise ValueError("frozen fit-transform state hash differs from training report")
    output = Path(output)
    if not output.is_absolute():
        output = panel.root / output
    if not output.resolve(strict=False).is_relative_to(panel.root):
        raise ValueError("inference output must remain inside workspace")
    output.mkdir(parents=True, exist_ok=False)

    fit_records = [dict(x) for x in panel.observations
                   if x["scope"]["role"] == "FIT_SUPPORT_WITH_PARITY1_DEVELOPMENT_MONITOR"]
    query_records = [dict(x) for x in panel.observations]
    fit_token_cache: dict[tuple[str, int], dict] = {}
    query_token_cache: dict[tuple[str, int], dict] = {}
    fit_pooled_by_id: dict[str, dict] = {}
    query_pooled_by_id: dict[str, dict] = {}
    fit_ids = {row["observation_id"] for row in fit_records}
    for record in panel.observations:
        query_pooled = pool_observation_quadrants(panel, record, transform, 1)
        query_pooled_by_id[record["observation_id"]] = query_pooled
        for token in query_pooled["tokens"]:
            query_token_cache[(record["observation_id"], token["quadrant"])] = {
                **token, "sensor": record["sensor"], "acquired_at": record["acquired_at"]}
        if record["observation_id"] in fit_ids:
            fit_pooled = pool_observation_quadrants(panel, record, transform, 0)
            fit_pooled_by_id[record["observation_id"]] = fit_pooled
            for token in fit_pooled["tokens"]:
                fit_token_cache[(record["observation_id"], token["quadrant"])] = {
                    **token, "sensor": record["sensor"], "acquired_at": record["acquired_at"]}

    device = torch.device("cpu")
    score_rows: list[dict] = []
    seed_states = []
    reference_state_records = []
    score_c_states = []
    representative_attention = {"status": "UNAVAILABLE_SEED0_SELECTED_CHECKPOINT",
        "arrays_path": None, "metadata_path": None, "query_observation_id": None,
        "checkpoint_sha256": None, "array_count": 0}
    for seed_record in training_report["seeds"]:
        seed = seed_record["seed"]
        seed_dir = training_dir / f"seed_{seed}"
        checkpoint_path = seed_dir / "selected_checkpoint.pt"
        if not checkpoint_path.is_file() or seed_record["selected_checkpoint_sha256"] is None:
            if seed == 0:
                representative_attention["status"] = "UNAVAILABLE_SEED0_SELECTED_CHECKPOINT"
            for record in query_records:
                query_pixel_channel_count = _pooled_pixel_channel_count(
                    query_pooled_by_id[record["observation_id"]])
                for method in ("tmae_score_a", "tmae_score_b", "tmae_score_c"):
                    score_rows.append({**_null_method_row(method, record,
                        "UNAVAILABLE_NO_SELECTED_DEVELOPMENT_CHECKPOINT",
                        query_parity=1, fitting_parity=0,
                        query_pixel_channel_count=query_pixel_channel_count), "seed": seed})
            seed_states.append({"seed": seed, "status": "UNAVAILABLE_NO_SELECTED_DEVELOPMENT_CHECKPOINT"})
            continue
        if sha256_file(checkpoint_path) != seed_record["selected_checkpoint_sha256"]:
            raise ValueError(f"selected checkpoint SHA-256 changed for seed {seed}")
        config_fields = TrainingConfigV2.__dataclass_fields__
        config = TrainingConfigV2(**{key: value for key, value in
            training_report["model_configuration"].items() if key in config_fields})
        model = TemporalMaskedMAEV2(config).to(device)
        metadata = load_checkpoint_v2(checkpoint_path, model,
            expected_panel_sha256=panel.manifest_sha256,
            expected_transform_sha256=transform["state_sha256"])
        if seed == 0:
            representative_attention = _write_seed0_attention_export(model, panel,
                query_records, query_token_cache, output,
                seed_record["selected_checkpoint_sha256"], device=device)
        score_b_states, projected = _build_score_b_reference(model, panel,
            fit_records, fit_token_cache, device=device)

        fit_component_rows = []
        for record in fit_records:
            prefix = _context_prefix(fit_records, record)
            sequence = _sequence_from_prefix(prefix, fit_token_cache)
            score_a = _score_a(model, panel, record, fit_records, fit_token_cache,
                               parity=0, device=device)
            verify_score_a_components(record, fit_pooled_by_id[record["observation_id"]], score_a)
            embeddings = _encoder_token_embeddings(model, sequence, record, device=device)
            score_b = _score_b_query(record, embeddings, projected, score_b_states)
            common = {"observation_id": record["observation_id"],
                "site_id": record["site_id"], "sensor": record["sensor"],
                "product_id": record["product_id"], "acquired_at": record["acquired_at"],
                "scope_role": record["scope"]["role"], "event_phase": record["scope"]["event_phase"],
                "source_sha256": record["sha256"], "query_parity": 0, "fitting_parity": 0}
            fit_component_rows.extend([
                common | {"method": "tmae_score_a", "score": score_a["score"],
                    "status": score_a["status"], "denominator": score_a["denominator"],
                    "denominator_unit": score_a["denominator_unit"]},
                common | {"method": "tmae_score_b", "score": score_b["score"],
                    "status": score_b["status"], "denominator": score_b["denominator"],
                    "denominator_unit": score_b["denominator_unit"]},
            ])
        a_standardizer = _fit_score_standardizer(fit_component_rows, "tmae_score_a")
        b_standardizer = _fit_score_standardizer(fit_component_rows, "tmae_score_b")
        c_state_payload = {"version": "FIXED_EQUAL_WEIGHT_SCORE_C_V1", "alpha": 0.5,
            "score_a_standardizer": a_standardizer, "score_b_standardizer": b_standardizer,
            "fit_role": "ALL_PRE_BREACH_FIT_PARITY0_ONLY",
            "query_parity": 0, "fitting_parity": 0,
            "fit_score_support": fit_component_rows,
            "source_manifest_sha256": panel.manifest_sha256,
            "domain_protocol_sha256": panel.domain_protocol_sha256,
            "fit_transform_sha256": transform["state_sha256"]}
        c_state = {**c_state_payload, "state_sha256": _sha_json(c_state_payload)}

        rows_this_seed = []
        for record in query_records:
            prefix = _context_prefix(query_records, record)
            sequence = _sequence_from_prefix(prefix, query_token_cache)
            score_a = _score_a(model, panel, record, query_records, query_token_cache,
                               parity=1, device=device)
            verify_score_a_components(record, query_pooled_by_id[record["observation_id"]], score_a)
            embeddings = _encoder_token_embeddings(model, sequence, record, device=device)
            score_b = _score_b_query(record, embeddings, projected, score_b_states)
            query_pixel_channel_count = _pooled_pixel_channel_count(
                query_pooled_by_id[record["observation_id"]])
            full_source_valid_pixel_channel_count = int(sum(
                record["observed_valid_counts_by_channel"]))
            for method, value, status, denominator in (
                ("tmae_score_a", score_a["score"], score_a["status"], score_a["denominator"]),
                ("tmae_score_b", score_b["score"], score_b["status"], score_b["denominator"]),
            ):
                rows_this_seed.append({"method": method,
                    "observation_id": record["observation_id"], "site_id": record["site_id"],
                    "sensor": record["sensor"], "product_id": record["product_id"],
                    "acquired_at": record["acquired_at"], "scope_role": record["scope"]["role"],
                    "event_phase": record["scope"]["event_phase"], "score": value,
                    "status": status, "reason": None if value is not None else status,
                    "denominator": denominator,
                    "denominator_unit": (score_a["denominator_unit"] if method == "tmae_score_a"
                                         else score_b["denominator_unit"]),
                    "query_pixel_channel_count": query_pixel_channel_count,
                    "full_source_valid_pixel_channel_count": full_source_valid_pixel_channel_count,
                    "seed": seed,
                    "query_parity": 1, "fitting_parity": 0,
                    "source_sha256": record["sha256"],
                    "fit_transform_sha256": transform["state_sha256"],
                    "selected_checkpoint_sha256": seed_record["selected_checkpoint_sha256"],
                    **({"score_a_components": score_a} if method == "tmae_score_a" else {"score_b_components": score_b})})
            a_row = rows_this_seed[-2]
            b_row = rows_this_seed[-1]
            z_a = _standardized_score(a_row["score"], a_standardizer)
            z_b = _standardized_score(b_row["score"], b_standardizer)
            joint_quadrants = _score_c_joint_quadrants(score_a, score_b)
            if z_a is None or z_b is None:
                score_c = None
                score_c_status = "UNAVAILABLE_COMPONENT_OR_FIT_STANDARDIZER"
            elif not joint_quadrants:
                score_c = None
                score_c_status = "UNAVAILABLE_NO_JOINT_QUADRANT_SUPPORT"
            else:
                score_c = 0.5 * z_a + 0.5 * z_b
                score_c_status = ("ESTIMATED" if len(joint_quadrants) == 4 else
                                  "ESTIMATED_PARTIAL_JOINT_QUADRANT_SUPPORT")
            rows_this_seed.append({"method": "tmae_score_c", "observation_id": record["observation_id"],
                "site_id": record["site_id"], "sensor": record["sensor"],
                "product_id": record["product_id"], "acquired_at": record["acquired_at"],
                "scope_role": record["scope"]["role"], "event_phase": record["scope"]["event_phase"],
                "score": score_c, "status": score_c_status,
                "reason": None if score_c is not None else score_c_status,
                "denominator": len(joint_quadrants),
                "denominator_unit": "query_quadrants_with_joint_score_a_and_score_b_support",
                "score_a_target_channel_count": a_row["denominator"],
                "score_a_denominator_unit": a_row["denominator_unit"],
                "score_b_supported_quadrant_count": b_row["denominator"],
                "score_b_denominator_unit": b_row["denominator_unit"],
                "joint_supported_quadrant_ids": joint_quadrants,
                "joint_quadrant_support_count": len(joint_quadrants),
                "joint_quadrant_support_unit": "query_quadrants_with_both_score_components_supported",
                "query_pixel_channel_count": query_pixel_channel_count,
                "full_source_valid_pixel_channel_count": full_source_valid_pixel_channel_count,
                "seed": seed, "source_sha256": record["sha256"],
                "fit_transform_sha256": transform["state_sha256"],
                "selected_checkpoint_sha256": seed_record["selected_checkpoint_sha256"],
                "query_parity": 1, "fitting_parity": 0,
                "score_a": a_row["score"], "score_b": b_row["score"],
                "score_a_z_fit": z_a, "score_b_z_fit": z_b,
                "alpha": 0.5, "score_c_state_sha256": c_state["state_sha256"]})
        score_rows.extend(rows_this_seed)
        score_c_states.append({"seed": seed, **c_state})
        reference_state_records.append({"seed": seed, "states": score_b_states})
        seed_states.append({"seed": seed, "status": "SCORED_CAUSAL_PREFIXES",
            "checkpoint_sha256": seed_record["selected_checkpoint_sha256"],
            "checkpoint_epoch": metadata.get("selected_epoch"),
            "score_b_reference_group_count": len(score_b_states)})

    baseline_states, baseline_rows, baseline_fit_if_rows = _fit_baselines(
        panel, transform, fit_pooled_by_id, query_pooled_by_id)
    score_rows.extend(baseline_rows)
    if_query_rows = [row for row in baseline_rows if row.get("method") == "isolation_forest"]
    cusum_states, cusum_rows = _cusum_rows(panel, if_query_rows, baseline_fit_if_rows)
    score_rows.extend(cusum_rows)
    calibration = {"method": "POT/GPD", "status": "UNAVAILABLE_NO_VERIFIED_CLOSED_WINDOW_EXPOSURE_OR_INDEPENDENT_CALIBRATION_SUPPORT",
        "threshold": None, "gpd_shape": None, "gpd_scale": None,
        "achieved_false_alarm_episode_rate_per_lake_year": None,
        "calibration_observation_count": 0,
        "exposure_lake_years": None, "reason": "No independent threshold-calibration role was admitted; the manifest has no closed opportunity schedule, reviewed stationarity/declustering, or verified independent support count"}
    # Write scores first as JSONL to retain per-observation null/status semantics.
    scores_path = output / "scores.v2.jsonl"
    with scores_path.open("x", encoding="utf-8") as stream:
        for row in score_rows:
            clean = _json_safe(row)
            stream.write(json.dumps(clean, sort_keys=True, allow_nan=False) + "\n")
    summary = {
        "schema_version": 2, "inference_version": INFERENCE_V2_VERSION,
        "status": "RETROSPECTIVE_DEVELOPMENT_ANALYSIS_WITH_EXPLICIT_UNAVAILABLE_METHODS",
        "source_manifest_sha256": panel.manifest_sha256,
        "domain_protocol_sha256": panel.domain_protocol_sha256,
        "fit_transform_sha256": transform["state_sha256"],
        "input_observation_count": len(panel.observations),
        "score_row_count": len(score_rows),
        "seed_training_states": seed_states,
        "representative_seed0_attention": representative_attention,
        "score_b_reference_states": reference_state_records,
        "score_c_fit_standardizers": score_c_states,
        "baseline_states": baseline_states,
        "cusum_states": cusum_states,
        "calibration": calibration,
        "descriptive_summary": _descriptive_summary(score_rows),
        "causality": {"query_windows": "same-site acquired_at <= query acquired_at",
            "future_observation_count_per_query": 0,
            "same_timestamp_same_sensor_duplicate_context_excluded": True,
            "fit_states_frozen_before_post_cutoff_queries": True,
            "fit_end_exclusive_utc": panel.document["domain_protocol"]["fit_end_exclusive"],
            "fitting_parity": 0, "published_query_parity": 1,
            "reference_scope": "ALL_PRE_BREACH_FIT_SUPPORT_ONLY",
            "PCA_scope": "ALL_PRE_BREACH_FIT_SUPPORT_ONLY",
            "Score_C_standardizer_scope": "ALL_PRE_BREACH_FIT_SUPPORT_ONLY"},
        "independent_physical_support_count": None,
        "independent_physical_support_status": "UNKNOWN",
        "seeds_are_not_independent_sample_units": True,
        "no_event_labels_or_control_labels_used": True,
        "no_post_cutoff_model_or_reference_fit": True,
        "analysis_status": "RETROSPECTIVE_DEVELOPMENT_ANALYSIS",
        "analysis_limitations": [
            "All pre-breach acquisitions up to the strict cutoff inform fitted states, so early pre-breach trajectories are development-exposed retrospective reconstructions rather than prospective precursor validation.",
            "Each query uses only its same-site causal acquisition prefix; future acquisitions never enter that query context.",
            "The parity-1 monitor shares acquisition scenes with parity-0 fitting and does not establish independent temporal or physical support.",
            "Historical operational availability, lake geometry validity, physical sampling independence, and independent calibration exposure are unresolved.",
        ],
        "publication_authorized": False,
    }
    (output / "analysis_summary.v2.json").write_text(json.dumps(_json_safe(summary),
        indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def _json_safe(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite value cannot be serialized as a score")
    return value


