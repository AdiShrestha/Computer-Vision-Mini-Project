"""Constructed offline engineering fixtures for the source-bound v2 pipeline.

These fixtures are not observations, event labels, scientific measurements,
or evidence of model performance. They verify byte, mask, scope, math, and
causality contracts without running empirical training.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sentinel_gl.panel_v2 import (
    CANONICAL_CHANNELS, PanelV2Error, _verify_optical_arrays,
    causal_prefix, fit_channel_standardizer, load_panel_v2,
    pool_observation_quadrants, scope_for_acquisition,
)
from sentinel_gl.training_v2 import (
    TemporalMaskedMAEV2, TrainingConfigV2, _fold_batch, _sequence_from_prefix,
)
from sentinel_gl.inference_v2 import (
    _capture_attention, _cusum_rows, _fit_baselines, _fit_score_standardizer,
    _complete_data_rpca_v2, _pooled_pixel_channel_count, _rpca_fit_rows_v2,
    _score_b_query, _score_c_joint_quadrants,
)

RUNNER_PATH = Path(__file__).resolve().parents[1] / "runners/run_2023_diagnostics_v2.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_2023_diagnostics_v2", RUNNER_PATH)
run_2023_diagnostics_v2 = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(run_2023_diagnostics_v2)


UTC = timezone.utc
PIXELS = (
    (0, 0), (0, 16), (0, 256), (0, 272),
    (256, 0), (256, 16), (256, 256), (256, 272),
)
FIT_DATES = (
    "2023-01-09T04:00:00Z", "2023-06-01T04:00:00Z",
    "2023-08-01T04:00:00Z", "2023-09-01T04:00:00Z",
    "2023-10-01T04:00:00Z",
)
GRID = {"shape": [499, 499], "crs": "EPSG:32645",
        "transform": [20, 0, 600000, 0, -20, 3100020]}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index_float32(left: np.float32, right: np.float32) -> np.float32:
    return np.float32((float(left) - float(right)) / (float(left) + float(right)))


def _optical_arrays(level: int) -> dict[str, np.ndarray]:
    values = np.full((5, 499, 499), np.nan, dtype=np.float32)
    valid = np.zeros((5, 499, 499), dtype=bool)
    scl = np.zeros((499, 499), dtype=np.uint8)
    qa = np.zeros((499, 499), dtype=bool)
    for row, col in PIXELS:
        b03 = np.float32(0.18 + level * 0.007)
        b08 = np.float32(0.31 + level * 0.004)
        b11 = np.float32(0.43 + level * 0.003)
        values[:, row, col] = (b03, b08, b11,
                               _index_float32(b03, b08), _index_float32(b03, b11))
        valid[:, row, col] = True
        scl[row, col] = np.uint8(4 + level % 3)
        qa[row, col] = True
    return {"values": values, "observed_valid": valid,
            "scl_classes": scl, "qa_clear": qa}


def _db(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(values, dtype=np.float64)
    support = np.asarray(valid, dtype=bool) & np.isfinite(source) & (source > 0)
    output = np.full(source.shape, np.nan, dtype=np.float32)
    output[support] = (10.0 * np.log10(source[support])).astype(np.float32)
    return output, support


def _sar_arrays(level: int) -> dict[str, np.ndarray]:
    values = np.full((2, 499, 499), np.nan, dtype=np.float32)
    valid = np.zeros((2, 499, 499), dtype=bool)
    signed = np.full_like(values, np.nan)
    signed_valid = np.zeros_like(valid)
    nesz = np.full_like(values, np.nan)
    for pol in range(2):
        for row, col in PIXELS:
            raw = np.float32(0.7 + 0.1 * level + 0.05 * pol)
            correction = np.float32(0.2 + 0.02 * level)
            values[pol, row, col] = raw
            valid[pol, row, col] = True
            signed[pol, row, col] = raw - correction
            signed_valid[pol, row, col] = True
            nesz[pol, row, col] = correction
    # Signed noise-subtracted power may be nonpositive and must remain unavailable in dB.
    signed[0, *PIXELS[0]] = np.float32(-0.01)
    raw_db, raw_db_valid = _db(values, valid)
    signed_db, signed_db_valid = _db(signed, signed_valid)
    return {"values": values, "observed_valid": valid,
            "signed_values": signed, "signed_valid": signed_valid,
            "raw_db": raw_db, "raw_db_valid": raw_db_valid,
            "signed_db": signed_db, "signed_db_valid": signed_db_valid,
            "nesz_values": nesz}


def _write_observation(root: Path, sensor: str, acquired_at: str, level: int,
                       *, end_at: str | None = None, suffix: str = "",
                       empty: bool = False) -> dict:
    obs_id = f"{sensor[-2:]}_{level:02d}_{acquired_at[0:10]}{suffix}"
    product_id = f"CONSTRUCTED_{sensor}_{level:02d}_{acquired_at[0:10]}{suffix}"
    directory = root / "data/derived/expansion_2023/rois"
    directory.mkdir(parents=True, exist_ok=True)
    values_path = directory / f"{obs_id}.npz"
    arrays = _optical_arrays(level) if sensor == "SENTINEL_2_OPTICAL" else _sar_arrays(level)
    if empty:
        arrays["values"][:] = np.nan
        arrays["observed_valid"][:] = False
    np.savez_compressed(values_path, **arrays)
    rel_values = values_path.relative_to(root).as_posix()
    receipt = {"observation_id": obs_id, "site_id": "PILOT-SOUTH-LHONAK",
               "product_id": product_id, "acquired_at": acquired_at,
               "values_file": rel_values, "sha256": _sha(values_path),
               "origin": "CONSTRUCTED_OFFLINE_ENGINEERING_FIXTURE"}
    if end_at is not None:
        receipt["end_at"] = end_at
    receipt_path = directory / f"{obs_id}.receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    channels = ("B03", "B08", "B11", "NDWI", "MNDWI") if sensor == "SENTINEL_2_OPTICAL" else (
        "SIGMA0_VV_RAW_LINEAR", "SIGMA0_VH_RAW_LINEAR")
    return {"observation_id": obs_id, "site_id": "PILOT-SOUTH-LHONAK",
            "sensor": sensor, "acquired_at": acquired_at, "product_id": product_id,
            "values_file": rel_values, "sha256": _sha(values_path),
            "channels": list(channels), "grid": GRID,
            "receipt_file": receipt_path.relative_to(root).as_posix()}


def _fixture_panel(tmp_path: Path):
    root = tmp_path
    (root / "configs").mkdir()
    source_protocol = Path(__file__).resolve().parents[2] / "configs/domain_2023_protocol.json"
    (root / "configs/domain_2023_protocol.json").write_bytes(source_protocol.read_bytes())
    observations = []
    for sensor in ("SENTINEL_2_OPTICAL", "SENTINEL_1_IW_GRD"):
        for index, acquired_at in enumerate(FIT_DATES):
            end_at = None
            if sensor == "SENTINEL_1_IW_GRD":
                end_at = (datetime.fromisoformat(acquired_at.replace("Z", "+00:00")) +
                          timedelta(minutes=2)).isoformat().replace("+00:00", "Z")
            observations.append(_write_observation(root, sensor, acquired_at, index + 1,
                                                   end_at=end_at))
        eval_at = "2023-10-10T04:00:00Z"
        end_at = None
        if sensor == "SENTINEL_1_IW_GRD":
            end_at = "2023-10-10T04:02:00Z"
        observations.append(_write_observation(root, sensor, eval_at, 9, end_at=end_at))
    observations.append(_write_observation(root, "SENTINEL_1_IW_GRD",
        "2023-10-02T18:29:00Z", 10, end_at="2023-10-02T18:31:00Z", suffix="_CROSS"))
    observations.append(_write_observation(root, "SENTINEL_2_OPTICAL",
        "2023-10-11T04:00:00Z", 11, suffix="_EMPTY_QUERY", empty=True))
    manifest_path = root / "data/derived/expansion_2023/observations.v2.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"schema_version": 2,
        "origin": "CONSTRUCTED_OFFLINE_ENGINEERING_FIXTURE",
        "observations": observations}, sort_keys=True) + "\n")
    return load_panel_v2(root, manifest_path), observations


def test_protocol_cutoff_uses_acquisition_end_and_keeps_boundary_unresolved():
    cutoff = datetime(2023, 10, 2, 18, 30, tzinfo=UTC)
    start = datetime(2023, 10, 2, 18, 29, tzinfo=UTC)
    assert scope_for_acquisition(start, start + timedelta(seconds=59), cutoff)["role"] == (
        "FIT_SUPPORT_WITH_PARITY1_DEVELOPMENT_MONITOR")
    crossing = scope_for_acquisition(start, cutoff, cutoff)
    assert crossing["role"] == "FIT_MEMBERSHIP_UNRESOLVED"
    assert crossing["fit_membership_status"] == "ACQUISITION_CROSSES_FIT_END_EXCLUSIVE"
    missing = scope_for_acquisition(start, None, cutoff)
    assert missing["role"] == "FIT_MEMBERSHIP_UNRESOLVED"
    after = scope_for_acquisition(cutoff, cutoff + timedelta(seconds=1), cutoff)
    assert after["role"] == "EVALUATION_TRAJECTORY"


def test_source_bound_panel_fits_every_pre_cutoff_scene_and_excludes_later_rows(tmp_path):
    panel, source_rows = _fixture_panel(tmp_path)
    assert panel.document["domain_protocol"]["fit_end_exclusive"] == "2023-10-02T18:30:00Z"
    fit_records = [row for row in panel.observations
                   if row["scope"]["role"] == "FIT_SUPPORT_WITH_PARITY1_DEVELOPMENT_MONITOR"]
    assert len(fit_records) == 10
    assert all(row["acquisition_end_at"] < "2023-10-02T18:30:00.000000Z" for row in fit_records)
    crossing = next(row for row in panel.observations if row["observation_id"].endswith("_CROSS"))
    assert crossing["scope"]["role"] == "FIT_MEMBERSHIP_UNRESOLVED"
    late = next(row for row in panel.observations if row["acquired_at"].startswith("2023-10-10"))
    assert late["scope"]["role"] == "EVALUATION_TRAJECTORY"
    assert late["scope"]["event_phase"] == "POST_EVENT_CANDIDATE"

    transform = fit_channel_standardizer(panel)
    assert transform["fit_scope"] == "ALL_PRE_BREACH_ACQUISITIONS_ENDING_STRICTLY_BEFORE_CUTOFF"
    assert transform["fit_parity"] == 0
    assert transform["monitor_support_used"] is False
    assert "2023-06-01" in " ".join(transform["fit_observation_ids"])
    assert "2023-09-01" in " ".join(transform["fit_observation_ids"])
    assert not any("2023-10-10" in row_id for row_id in transform["fit_observation_ids"])
    optical_fit = [row for row in fit_records if row["sensor"] == "SENTINEL_2_OPTICAL"]
    source_values = []
    for row in optical_fit:
        arrays = panel.load_arrays(row)
        parity0 = (((np.indices((499, 499))[0] // 16 + np.indices((499, 499))[1] // 16) % 2) == 0)
        source_values.extend(arrays["values"][0][arrays["observed_valid"][0] & parity0].astype(np.float64))
    assert transform["means"][CANONICAL_CHANNELS.index("B03")] == pytest.approx(
        float(np.mean(source_values, dtype=np.float64)), abs=1e-12)

    sar = next(row for row in fit_records if row["sensor"] == "SENTINEL_1_IW_GRD")
    assert sar["qa_summary"]["d_b_status"] == "VERIFIED_POSITIVE_SUPPORT_DB_REPLAY"
    assert sar["qa_summary"]["d_b_arrays"][1]["finite_count"] < int(panel.load_arrays(sar)["signed_valid"].sum())


def test_optical_indices_use_exact_float32_serialization_without_clipping():
    arrays = _optical_arrays(2)
    row, col = PIXELS[0]
    b03 = np.float32(0.2001)
    b08 = np.float32(-0.19985)
    b11 = np.float32(-0.19985)
    arrays["values"][:, row, col] = (b03, b08, b11,
        _index_float32(b03, b08), _index_float32(b03, b11))
    summary = _verify_optical_arrays(arrays, arrays["observed_valid"])
    assert abs(float(arrays["values"][3, row, col])) > 1000
    assert summary["status"] == "SCL_AND_INDEX_DEPENDENCIES_VERIFIED"

    corrupted = {name: value.copy() for name, value in arrays.items()}
    corrupted["values"][3, row, col] = np.nextafter(corrupted["values"][3, row, col], np.float32(np.inf))
    with pytest.raises(PanelV2Error, match="do not replay"):
        _verify_optical_arrays(corrupted, corrupted["observed_valid"])


def test_four_quadrant_folds_hide_all_target_values_and_capture_attention(tmp_path):
    panel, _ = _fixture_panel(tmp_path)
    transform = fit_channel_standardizer(panel)
    query = next(row for row in panel.observations
                 if row["sensor"] == "SENTINEL_2_OPTICAL" and
                 row["scope"]["role"] == "FIT_SUPPORT_WITH_PARITY1_DEVELOPMENT_MONITOR")
    pooled = pool_observation_quadrants(panel, query, transform, 0)
    cache = {(query["observation_id"], token["quadrant"]): {
        **token, "sensor": query["sensor"], "acquired_at": query["acquired_at"]}
        for token in pooled["tokens"]}
    sequence = _sequence_from_prefix([query], cache)
    batch = _fold_batch(sequence, query["observation_id"], torch.device("cpu"))
    assert batch is not None and batch["folds"] == [0, 1, 2, 3]
    assert batch["target"].sum(dim=(1, 2)).tolist() == [5, 5, 5, 5]
    for index, target_position in enumerate(batch["target_positions"]):
        assert not batch["observed"][index, target_position].any()
        assert torch.equal(batch["values"][index, target_position], torch.zeros(7))
        assert batch["target"][index, target_position].sum() == 5

    config = TrainingConfigV2(d_model=8, n_heads=2, encoder_layers=1, decoder_layers=1,
                              encoder_ff=16, decoder_ff=16)
    assert (config.minimum_epochs, config.earliest_checkpoint_epoch,
            config.max_epochs, config.patience) == (12, 12, 80, 12)
    model = TemporalMaskedMAEV2(config)
    prediction, encoded = model(batch["values"], batch["observed"], batch["modality"],
                                batch["time"], batch["quadrant"], batch["target"])
    assert prediction.shape == batch["values"].shape
    assert encoded.shape[:2] == batch["values"].shape[:2]
    target_tokens = batch["target"].any(dim=-1)
    decoder_inputs = model._decoder_inputs(encoded, target_tokens, batch["modality"],
                                           batch["time"], batch["quadrant"])
    expected_identity = model.mask_token.expand_as(encoded) + model._token_identity(
        batch["modality"], batch["time"], batch["quadrant"])
    assert torch.allclose(decoder_inputs[target_tokens], expected_identity[target_tokens])
    attention, captured_prediction = _capture_attention(model, batch, return_prediction=True)
    # Attention capture disables the fused MHA fast path; compare the same
    # checkpoint calculation within float32 kernel-ordering precision.
    assert torch.allclose(captured_prediction, prediction.detach().cpu(), atol=1e-6, rtol=1e-6)
    assert set(attention) == {"encoder_layer_0", "decoder_layer_0"}
    assert attention["encoder_layer_0"].shape == (4, 2, 4, 4)
    assert np.isfinite(attention["decoder_layer_0"]).all()


def test_score_b_excludes_same_product_and_same_sensor_arrival_from_neighbors():
    class IdentityProjection:
        def transform(self, rows):
            return np.asarray(rows, dtype=np.float64)

    query = {"site_id": "PILOT-SOUTH-LHONAK", "sensor": "SENTINEL_2_OPTICAL",
             "product_id": "query-product", "acquired_at": "2023-10-10T04:00:00Z"}
    rows = [
        {"product_id": "query-product", "acquired_at": "2023-10-01T00:00:00Z", "embedding": np.array([0.0, 0.0])},
        {"product_id": "same-arrival", "acquired_at": query["acquired_at"], "embedding": np.array([0.0, 0.0])},
    ]
    for index in range(1, 6):
        rows.append({"product_id": f"fit-{index}", "acquired_at": f"2023-09-{index:02d}T00:00:00Z",
                     "embedding": np.array([float(index), 0.0])})
    result = _score_b_query(query, [{"quadrant": 0, "embedding": np.array([0.0, 0.0])}],
        {"PILOT-SOUTH-LHONAK::SENTINEL_2_OPTICAL": {
            "pca": IdentityProjection(), "matrix": np.stack([row["embedding"] for row in rows]),
            "rows": rows}}, {})
    assert result["status"] == "ESTIMATED_PARTIAL_QUADRANT_SUPPORT"
    assert result["distinct_reference_acquisitions"] == 5
    assert result["quadrant_distances"][0]["score"] == pytest.approx(3.0)


def test_sensor_stratified_rpca_keeps_per_sensor_scope_and_truthful_convergence(tmp_path):
    panel, _ = _fixture_panel(tmp_path)
    transform = fit_channel_standardizer(panel)
    pooled_fit = {}
    pooled_query = {}
    for row in panel.observations:
        pooled_query[row["observation_id"]] = pool_observation_quadrants(panel, row, transform, 1)
        if row["scope"]["role"] == "FIT_SUPPORT_WITH_PARITY1_DEVELOPMENT_MONITOR":
            pooled_fit[row["observation_id"]] = pool_observation_quadrants(panel, row, transform, 0)
    states, rows, fit_if_rows = _fit_baselines(panel, transform, pooled_fit, pooled_query)
    optical = states["SENTINEL_2_OPTICAL"]["inexact_alm_rpca"]
    sar = states["SENTINEL_1_IW_GRD"]["inexact_alm_rpca"]
    assert optical["max_iterations"] == sar["max_iterations"] == 1000
    assert optical["tolerance"] == sar["tolerance"] == 1e-5
    assert len(optical["fit_observation_ids"]) == 5
    assert len(sar["fit_observation_ids"]) == 5
    assert all(row["sensor"] in {"SENTINEL_2_OPTICAL", "SENTINEL_1_IW_GRD"} for row in rows)
    assert all(row["query_parity"] == 1 and row["fitting_parity"] == 0 for row in rows)
    assert all(row["query_parity"] == 0 and row["fitting_parity"] == 0 for row in fit_if_rows)
    records_by_id = {row["observation_id"]: row for row in panel.observations}
    for row in rows:
        assert row["query_pixel_channel_count"] == _pooled_pixel_channel_count(
            pooled_query[row["observation_id"]])
        assert row["full_source_valid_pixel_channel_count"] == sum(
            records_by_id[row["observation_id"]]["observed_valid_counts_by_channel"])
        assert row["denominator_unit"] == "observed_sensor_channels"
        assert "source_support_count" not in row
    # Convergence is reported as observed; this fixture imposes no desired outcome.
    assert optical["status"] in {"CONVERGED", "NONCONVERGED_MAX_ITERATIONS"}
    assert sar["status"] in {"CONVERGED", "NONCONVERGED_MAX_ITERATIONS"}
    empty_record = next(row for row in panel.observations
                        if row["observation_id"].endswith("_EMPTY_QUERY"))
    empty_score = next(row for row in rows if row["method"] == "inexact_alm_rpca" and
                       row["observation_id"] == empty_record["observation_id"])
    assert empty_score["query_pixel_channel_count"] == 0
    assert empty_score["score"] is None
    assert empty_score["query_parity"] == 1 and empty_score["fitting_parity"] == 0


def test_v2_fixed_penalty_rpca_recovers_constructed_sparse_low_rank_and_kkt():
    rng = np.random.default_rng(31)
    rows, columns, rank, sparse_count = 20, 15, 2, 5
    left_basis = np.linalg.qr(rng.normal(size=(rows, rank)))[0]
    right_basis = np.linalg.qr(rng.normal(size=(columns, rank)))[0]
    expected_low_rank = left_basis @ np.diag([8.0, 5.0]) @ right_basis.T
    expected_sparse = np.zeros((rows, columns), dtype=np.float64)
    indices = rng.choice(rows * columns, sparse_count, replace=False)
    expected_sparse.flat[indices] = rng.choice([-1.0, 1.0], sparse_count) * 8.0
    values = expected_low_rank + expected_sparse

    result = _complete_data_rpca_v2(values, max_iterations=1000, tolerance=1e-6)
    assert result["status"] == "CONVERGED"
    assert result["algorithm"] == "COMPLETE_DATA_FIXED_PENALTY_ADMM_RPCA_V2"
    assert result["iterations"] < 1000
    assert result["primal_residual"] <= 1e-6
    assert result["dual_residual"] <= 1e-6
    assert result["kkt_residuals"]["primal_feasibility_relative_residual"] <= 1e-6
    assert np.allclose(result["low_rank"], expected_low_rank, atol=1e-4, rtol=1e-4)
    assert np.allclose(result["sparse"], expected_sparse, atol=1e-4, rtol=1e-4)
    assert result["kkt_residuals"]["nuclear_dual_feasibility_violation"] <= 1e-5
    assert result["kkt_residuals"]["sparse_dual_feasibility_violation"] <= 1e-5
    assert result["kkt_residuals"]["sparse_subgradient_nonzero_residual"] <= 1e-5
    assert result["kkt_residuals"]["nuclear_subgradient_left_orthogonality_residual"] <= 1e-5
    assert result["kkt_residuals"]["nuclear_subgradient_right_orthogonality_residual"] <= 1e-5

    poor = _complete_data_rpca_v2(values, max_iterations=1, tolerance=1e-12)
    assert poor["status"] == "NONCONVERGED_MAX_ITERATIONS"
    assert poor["primal_residual"] > 1e-12 or poor["dual_residual"] > 1e-12


def test_rpca_fit_omits_only_no_support_rows_and_score_c_counts_joint_quadrants():
    empty = {"observation_id": "empty", "observed": np.zeros(7, dtype=bool)}
    partial = {"observation_id": "partial", "observed": np.asarray(
        [True, False, True, False, False, False, False])}
    complete = {"observation_id": "complete", "observed": np.asarray(
        [True, True, True, True, True, False, False])}
    selected, excluded = _rpca_fit_rows_v2([empty, partial, complete], range(5))
    assert [row["observation_id"] for row in selected] == ["partial", "complete"]
    assert excluded == ["empty"]

    score_a = {"quadrant_folds": [
        {"quadrant": 0, "target_channel_count": 5},
        {"quadrant": 1, "target_channel_count": 0},
        {"quadrant": 2, "target_channel_count": 3},
        {"quadrant": 3, "target_channel_count": 5},
    ]}
    score_b = {"quadrant_distances": [
        {"quadrant": 0, "score": 1.0},
        {"quadrant": 1, "score": 2.0},
        {"quadrant": 2, "score": None},
        {"quadrant": 3, "score": 3.0},
    ]}
    assert _score_c_joint_quadrants(score_a, score_b) == [0, 3]


def test_score_c_standardizer_and_cusum_center_use_only_fit_parity_zero(tmp_path):
    fit_scope = "FIT_SUPPORT_WITH_PARITY1_DEVELOPMENT_MONITOR"
    rows = [
        {"scope_role": fit_scope, "method": "tmae_score_a", "score": 1.0,
         "observation_id": "fit-a", "query_parity": 0, "fitting_parity": 0},
        {"scope_role": fit_scope, "method": "tmae_score_a", "score": 3.0,
         "observation_id": "fit-b", "query_parity": 0, "fitting_parity": 0},
        {"scope_role": fit_scope, "method": "tmae_score_a", "score": 10000.0,
         "observation_id": "monitor-leak", "query_parity": 1, "fitting_parity": 0},
    ]
    state = _fit_score_standardizer(rows, "tmae_score_a")
    assert state["status"] == "FITTED"
    assert state["count"] == 2 and state["mean"] == 2.0
    assert state["fit_observation_ids"] == ["fit-a", "fit-b"]

    panel, _ = _fixture_panel(tmp_path)
    query_rows = []
    fit_rows = [
        {"site_id": "PILOT-SOUTH-LHONAK", "sensor": "SENTINEL_2_OPTICAL",
         "scope_role": fit_scope, "observation_id": "fit-a", "score": 1.0,
         "acquired_at": "2023-09-01T00:00:00Z", "query_parity": 0, "fitting_parity": 0},
        {"site_id": "PILOT-SOUTH-LHONAK", "sensor": "SENTINEL_2_OPTICAL",
         "scope_role": fit_scope, "observation_id": "fit-b", "score": 3.0,
         "acquired_at": "2023-09-02T00:00:00Z", "query_parity": 0, "fitting_parity": 0},
        {"site_id": "PILOT-SOUTH-LHONAK", "sensor": "SENTINEL_2_OPTICAL",
         "scope_role": fit_scope, "observation_id": "fit-parity1-leak", "score": 900.0,
         "acquired_at": "2023-09-03T00:00:00Z", "query_parity": 1, "fitting_parity": 0},
    ]
    for record, value in zip(panel.observations[:2], (10.0, 11.0)):
        query_rows.append({"site_id": record["site_id"], "sensor": record["sensor"],
            "observation_id": record["observation_id"], "product_id": record["product_id"],
            "acquired_at": record["acquired_at"], "scope_role": fit_scope,
            "event_phase": record["scope"]["event_phase"], "score": value,
            "status": "ESTIMATED", "reason": None, "denominator": 1,
            "source_support_count": 1, "query_parity": 1, "fitting_parity": 0})
    states, output = _cusum_rows(panel, query_rows, fit_rows)
    state = states["PILOT-SOUTH-LHONAK::SENTINEL_2_OPTICAL"]
    assert state["center"] == 2.0
    assert state["scale"] == pytest.approx(np.sqrt(2.0))
    assert state["fit_observation_ids"] == ["fit-a", "fit-b"]
    assert all(row["query_parity"] == 1 and row["fitting_parity"] == 0 for row in output)


def test_causal_prefix_excludes_future_and_other_site_rows():
    rows = [
        {"observation_id": "early", "site_id": "A", "acquired_at": "2023-09-01T00:00:00Z",
         "sensor": "SENTINEL_2_OPTICAL", "product_id": "early"},
        {"observation_id": "query", "site_id": "A", "acquired_at": "2023-09-03T00:00:00Z",
         "sensor": "SENTINEL_2_OPTICAL", "product_id": "query"},
        {"observation_id": "future", "site_id": "A", "acquired_at": "2023-09-04T00:00:00Z",
         "sensor": "SENTINEL_2_OPTICAL", "product_id": "future"},
        {"observation_id": "other-site", "site_id": "B", "acquired_at": "2023-09-02T00:00:00Z",
         "sensor": "SENTINEL_2_OPTICAL", "product_id": "other"},
    ]
    prefix = causal_prefix(rows, rows[1])
    assert [row["observation_id"] for row in prefix] == ["early", "query"]


def test_cli_requires_explicit_stable_manifest_hash_and_five_distinct_seeds():
    assert run_2023_diagnostics_v2._seed_list("0,1,2,3,4") == (0, 1, 2, 3, 4)
    with pytest.raises(Exception):
        run_2023_diagnostics_v2._seed_list("0,1,2,3,3")
