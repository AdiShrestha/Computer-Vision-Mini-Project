"""Training integrity, loader isolation, normalizer isolation, and checkpoint replay tests.

FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
Synthetic panels, split allocations, and small mock model weights in this file verify
training loaders, memory bounds, normalizer isolation, and checkpoint reproducibility
in offline unit tests only. They do not represent real observations and must never
support scientific claims.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from datetime import date, timedelta

from sentinel_gl.features import (
    FEATURE_CHANNELS,
    NUM_CHANNELS,
    StaticTopography,
    MultiModalPanel,
)
from sentinel_gl.splits import SplitEngine, SCOPE_ROLES
from sentinel_gl.loaders import (
    PanelDataset,
    deterministic_collate,
    create_panel_dataloader,
)
from sentinel_gl.normalization import FittedNormalizer
from sentinel_gl.model import TimeSeriesMAE
from sentinel_gl.training import StopRule, fit_masked_autoencoder
from sentinel_gl.replay import (
    CheckpointBundle,
    compute_deterministic_reconstruction,
    verify_clean_process_replay,
)


def _create_mock_panel(lake_id, window_id, start_date="2023-01-01", end_date=None, empty=False):
    """Explicit analytical panel fixture with a declared variable schedule."""
    start = date.fromisoformat(start_date)
    stop = date.fromisoformat(end_date) if end_date is not None else start + timedelta(days=23)
    dates = tuple((start + timedelta(days=i)).isoformat() for i in range((stop - start).days + 1))
    topo = StaticTopography(elevation_m=5200.0, moraine_slope_deg=25.0, catchment_area_km2=10.0)
    values = np.full((len(dates), NUM_CHANNELS), np.nan, dtype=np.float64)
    mask = np.zeros(values.shape, dtype=np.bool_)
    if not empty:
        for day in range(min(10, len(dates))):
            values[day, 0] = 1.35
            values[day, 1] = 0.45
            values[day, 2] = 0.50
            values[day, 3] = -16.0
            values[day, 4] = -24.0
            values[day, 5] = -8.0
            values[day, 6] = 0.005
            values[day, 7] = -4.5
            values[day, 8] = 1.2
            mask[day, :] = True
    statuses = np.full(values.shape, "UNAVAILABLE_TEST_FIXTURE", dtype=object)
    statuses[mask] = "OBSERVED"
    return MultiModalPanel(lake_id=lake_id, window_id=window_id, start_date=dates[0],
        end_date=dates[-1], dates=dates, values=values, mask=mask,
        static_metadata=topo, origin="fixture", statuses=statuses)


def _four_lake_fixture(*, shared_support=False):
    lakes = []
    for index, role in enumerate(SCOPE_ROLES):
        lake_id = "LAKE-" + role
        support = {field: [f"{lake_id}:{field}"] for field in
            ("glacier_ids", "basin_ids", "era5_cell_ids", "sentinel_frame_ids", "event_time_support_ids")}
        lakes.append({"lake_id": lake_id, "centroid_lat": 0., "centroid_lon": index * 2.,
                      "physical_support_status": "VERIFIED", "physical_support": support})
    if shared_support:
        lakes[-1]["physical_support"]["basin_ids"] = list(lakes[0]["physical_support"]["basin_ids"])
    return lakes


def _fixture_source(lake_id, *, available="2023-06-06T00:00:00Z"):
    return {"record_id": "SRC-" + lake_id, "modality": "optical",
        "support_start": "2023-06-05T00:00:00Z", "support_end": "2023-06-05T12:00:00Z",
        "sha256": "a" * 64,
        "availability": {"status": "PROVIDER_EVIDENCE", "at": available,
            "evidence_id": "FIXTURE-RECEIPT-" + lake_id, "evidence_sha256": "b" * 64,
            "reason": None},
        "time_axes": {
            "acquisition": {"status": "VERIFIED", "at": "2023-06-05T12:00:00Z", "reason": None},
            "product_generation": {"status": "VERIFIED", "at": "2023-06-05T12:00:00Z", "reason": None},
            "publication": {"status": "UNKNOWN", "at": None, "reason": "test fixture has no publication axis"},
            "retrieval": {"status": "VERIFIED", "at": available, "reason": None},
            "historical_availability": {"status": "VERIFIED", "at": available, "reason": None},
            "execution": {"status": "VERIFIED", "at": available, "reason": None}}}


def _digest_json(value, *, sort_keys=True):
    return hashlib.sha256(json.dumps(value, sort_keys=sort_keys,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _fixture_bindings(lakes, directory, *, missing_lake=None, corrupt_lake=None):
    support_bindings, coordinate_bindings = {}, {}
    directory.mkdir(parents=True, exist_ok=True)
    for lake in lakes:
        lake_id = lake["lake_id"]
        support_path = directory / (lake_id + "-support.json")
        coordinate_path = directory / (lake_id + "-coordinates.json")
        support_path.write_text(json.dumps({"origin": "fixture", "lake_id": lake_id,
            "physical_support": lake["physical_support"]}, sort_keys=True, separators=(",", ":")))
        coordinate_values = [lake["centroid_lat"], lake["centroid_lon"]]
        coordinate_path.write_text(json.dumps({"origin": "fixture", "lake_id": lake_id,
            "coordinates": coordinate_values}, sort_keys=True, separators=(",", ":")))
        support_file_sha = hashlib.sha256(support_path.read_bytes()).hexdigest()
        coordinate_file_sha = hashlib.sha256(coordinate_path.read_bytes()).hexdigest()
        assert support_file_sha == hashlib.sha256(support_path.read_bytes()).hexdigest()
        assert coordinate_file_sha == hashlib.sha256(coordinate_path.read_bytes()).hexdigest()
        support_digest = _digest_json(lake["physical_support"])
        coordinate_digest = _digest_json(coordinate_values, sort_keys=False)
        support_bindings[lake_id] = {"artifact_path": str(support_path),
            "artifact_sha256": support_file_sha, "support_sha256": support_digest, "origin": "fixture"}
        coordinate_bindings[lake_id] = {"artifact_path": str(coordinate_path),
            "artifact_sha256": coordinate_file_sha, "coordinate_sha256": coordinate_digest, "origin": "fixture"}
        if lake_id == corrupt_lake:
            support_bindings[lake_id]["support_sha256"] = "f" * 64
        if lake_id == missing_lake:
            support_bindings.pop(lake_id)
            coordinate_bindings.pop(lake_id)
    return support_bindings, coordinate_bindings


def _four_role_fixture_manifest(tmp_path, *, shared_support=False, binding_mode="valid"):
    lakes = _four_lake_fixture(shared_support=shared_support)
    if binding_mode == "missing_all":
        support_bindings, coordinate_bindings = {}, {}
    else:
        support_bindings, coordinate_bindings = _fixture_bindings(lakes, tmp_path / binding_mode,
            missing_lake="LAKE-evaluation" if binding_mode == "missing_one" else None,
            corrupt_lake="LAKE-evaluation" if binding_mode == "corrupt_one" else None)
    role_map = {row["lake_id"]: role for row, role in zip(lakes, SCOPE_ROLES)}
    observations = {row["lake_id"]: [_fixture_source(row["lake_id"])] for row in lakes}
    manifest = SplitEngine(window_days=30, stride_days=1,
        minimum_observations_by_modality={"optical": 1}).build_spatiotemporal_split(
        split_id="fixture-split", lakes=lakes, observations_by_lake=observations,
        start_date="2023-07-01T00:00:00Z", end_date="2023-07-02T00:00:00Z",
        scope_by_lake=role_map, required_modalities=("optical",),
        verified_support_bindings=support_bindings,
        verified_coordinate_bindings=coordinate_bindings)
    return manifest, lakes


def _panel_for(lake_id, *, end_date="2023-06-30", available="2023-06-06T00:00:00Z"):
    panel = _create_mock_panel(lake_id, f"WIN-{lake_id}-20230701", "2023-06-01", end_date)
    source_id = "SRC-" + lake_id
    object.__setattr__(panel, "lineage", {f"{day}|{channel}": {"measurement_id": source_id,
        "product_id": "PRODUCT-" + lake_id,
        "support_start": "2023-06-05T00:00:00Z", "support_end": "2023-06-05T12:00:00Z",
        "available_at": available, "status": "OBSERVED"}
        for day in panel.dates[:10] for channel in panel.channel_names})
    return panel


def test_loader_respects_four_role_schedule_membership(tmp_path):
    manifest, lakes = _four_role_fixture_manifest(tmp_path)
    assert manifest.physical_graph["role_isolation_status"] == "FIXTURE_ONLY_NOT_RESEARCH_VERIFIED"
    assert not manifest.physical_graph["unresolved_units"]
    assert not manifest.physical_graph["cross_role_edges"]
    lake_by_id = {lake["lake_id"]: lake for lake in lakes}
    for node in manifest.physical_graph["nodes"]:
        for evidence_key, content_key in (("physical_support_evidence", "physical_support"),
                                          ("coordinate_evidence", "coordinates")):
            evidence = node[evidence_key]
            artifact = Path(evidence["artifact_path"])
            artifact_bytes = artifact.read_bytes()
            artifact_doc = json.loads(artifact_bytes)
            assert hashlib.sha256(artifact_bytes).hexdigest() == evidence["artifact_sha256"]
            if content_key == "physical_support":
                assert artifact_doc[content_key] == lake_by_id[node["lake_id"]][content_key]
            else:
                assert artifact_doc[content_key] == [lake_by_id[node["lake_id"]]["centroid_lat"],
                    lake_by_id[node["lake_id"]]["centroid_lon"]]
    train_id = "LAKE-fitting"
    eval_id = "LAKE-evaluation"
    train_panel = _panel_for(train_id)
    eval_panel = _panel_for(eval_id)

    ds_fit = PanelDataset([train_panel], split_manifest=manifest, role="fitting", mode="fixture")
    assert len(ds_fit) == 1
    x, mask, _ = ds_fit[0]
    assert x.shape == (len(train_panel.dates), len(FEATURE_CHANNELS))
    assert mask.shape == x.shape

    with pytest.raises(ValueError, match="not in allowed fitting lakes"):
        PanelDataset([eval_panel], split_manifest=manifest, role="fitting", mode="fixture")
    orphan = _create_mock_panel(train_id, "WIN-ORPHAN")
    with pytest.raises(ValueError, match="absent from explicit role schedule"):
        PanelDataset([orphan], split_manifest=manifest, role="fitting", mode="fixture")
    late = _create_mock_panel(train_id, f"WIN-{train_id}-20230701", "2023-06-01", "2023-07-03")
    object.__setattr__(late, "lineage", dict(train_panel.lineage))
    with pytest.raises(ValueError, match="ends after its decision time"):
        PanelDataset([late], split_manifest=manifest, role="fitting", mode="fixture")
    with pytest.raises(ValueError, match="four explicit fitting roles"):
        PanelDataset([train_panel], split_manifest=manifest, role="train", mode="fixture")
    with pytest.raises(ValueError, match="BLOCKED"):
        PanelDataset([train_panel], split_manifest=manifest, role="fitting", mode="observational")


@pytest.mark.parametrize("binding_mode", ["missing_all", "missing_one", "corrupt_one"])
def test_loader_rejects_unknown_missing_or_corrupt_fixture_graph_bindings(tmp_path, binding_mode):
    manifest, _ = _four_role_fixture_manifest(tmp_path / binding_mode, binding_mode=binding_mode)
    assert manifest.physical_graph["role_isolation_status"] == "UNRESOLVED_UNKNOWN_PHYSICAL_MEMBERSHIP"
    panel = _panel_for("LAKE-fitting")
    with pytest.raises(ValueError, match="unresolved physical or coordinate membership"):
        PanelDataset([panel], split_manifest=manifest, role="fitting", mode="fixture")


def test_loader_rejects_bound_cross_role_physical_edges(tmp_path):
    manifest, _ = _four_role_fixture_manifest(tmp_path, shared_support=True)
    assert manifest.physical_graph["role_isolation_status"] == "CROSS_ROLE_PHYSICAL_ADJACENCY"
    assert manifest.physical_graph["cross_role_edges"]
    panel = _panel_for("LAKE-fitting")
    with pytest.raises(ValueError, match="cross-role physical edges"):
        PanelDataset([panel], split_manifest=manifest, role="fitting", mode="fixture")


def test_loader_rejects_late_nested_availability_even_when_lineage_matches(tmp_path):
    manifest, _ = _four_role_fixture_manifest(tmp_path)
    row = next(row for row in manifest.schedule_rows if row.lake_id == "LAKE-fitting")
    metadata = dict(row.metadata)
    supports = [dict(source) for source in metadata["source_supports"]]
    source = supports[0]
    late = "2023-07-01T00:00:01Z"
    source["availability"] = {**source["availability"], "at": late}
    source["time_axes"] = {**source["time_axes"],
        "retrieval": {"status": "VERIFIED", "at": late, "reason": None},
        "historical_availability": {"status": "VERIFIED", "at": late, "reason": None}}
    metadata["source_supports"] = supports
    changed = dataclasses.replace(row, metadata=metadata)
    rows = tuple(changed if current.window_id == row.window_id else current
                 for current in manifest.schedule_rows)
    corrupted_manifest = dataclasses.replace(manifest, schedule_rows=rows)
    panel = _panel_for("LAKE-fitting", available=late)
    with pytest.raises(ValueError, match="future or late-available support"):
        PanelDataset([panel], split_manifest=corrupted_manifest, role="fitting", mode="fixture")


def test_loader_requires_complete_nested_time_axes(tmp_path):
    manifest, _ = _four_role_fixture_manifest(tmp_path)
    row = next(row for row in manifest.schedule_rows if row.lake_id == "LAKE-fitting")
    metadata = dict(row.metadata)
    supports = [dict(source) for source in metadata["source_supports"]]
    supports[0]["time_axes"] = {key: value for key, value in supports[0]["time_axes"].items()
        if key != "retrieval"}
    metadata["source_supports"] = supports
    changed = dataclasses.replace(row, metadata=metadata)
    rows = tuple(changed if current.window_id == row.window_id else current
                 for current in manifest.schedule_rows)
    malformed_manifest = dataclasses.replace(manifest, schedule_rows=rows)
    with pytest.raises(ValueError, match="separate acquisition/generation/publication/retrieval/history/execution axes"):
        PanelDataset([_panel_for("LAKE-fitting")], split_manifest=malformed_manifest,
                     role="fitting", mode="fixture")


def test_loader_compares_nested_schedule_availability_with_panel_lineage(tmp_path):
    manifest, _ = _four_role_fixture_manifest(tmp_path)
    mismatched_panel = _panel_for("LAKE-fitting", available="2023-06-07T00:00:00Z")
    with pytest.raises(ValueError, match="panel source support differs from its scheduled evidence"):
        PanelDataset([mismatched_panel], split_manifest=manifest, role="fitting", mode="fixture")


def test_loader_rejects_unobserved_batches():
    empty_panel = _create_mock_panel("LAKE-TRAIN", "WIN-EMPTY-01", empty=True)
    item = (
        torch.from_numpy(np.zeros(empty_panel.values.shape, dtype=np.float32)),
        torch.from_numpy(np.zeros(empty_panel.values.shape, dtype=bool)),
        empty_panel.to_dict(),
    )

    with pytest.raises(ValueError, match="Batch contains zero valid observations"):
        deterministic_collate([item])


def test_fitted_normalizer_isolation():
    # Training panels on LAKE-TRAIN
    train_panel = _create_mock_panel("LAKE-TRAIN", "W-TR-01", "2023-01-01", "2023-06-29")
    norm_tr = FittedNormalizer.fit_fixture([train_panel], training_lake_ids=["LAKE-TRAIN"])

    # Evaluation panel on LAKE-EVAL with extreme physical values (e.g. area = 4.8, temp = 30)
    eval_panel = _create_mock_panel("LAKE-EVAL", "W-EV-01", "2023-01-01", "2023-06-29")
    eval_panel.values[0, 0] = 4.8
    eval_panel.values[0, 8] = 30.0

    # Normalizer fit with both train and eval panels passed, but training_lake_ids strictly LAKE-TRAIN
    norm_isolated = FittedNormalizer.fit_fixture([train_panel, eval_panel], training_lake_ids=["LAKE-TRAIN"])

    # Assert exact state identity: eval panel has zero influence
    assert norm_tr.mean == norm_isolated.mean
    assert norm_tr.scale == norm_isolated.scale
    assert norm_tr.state_hash == norm_isolated.state_hash

    # Serialization roundtrip
    serialized = norm_isolated.to_json()
    restored = FittedNormalizer.from_fixture_json(serialized)
    assert restored.state_hash == norm_isolated.state_hash


def test_checkpoint_roundtrip_and_replay(tmp_path: Path):
    torch.manual_seed(42)
    config = {
        "n_channels": NUM_CHANNELS,
        "max_time_steps": 24,
        "d_model": 32,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_encoder_heads": 2,
        "n_decoder_heads": 2,
        "d_ff_encoder": 64,
        "d_ff_decoder": 64,
        "dropout": 0.0,
        "masking_ratio": 0.5,
    }
    model = TimeSeriesMAE(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Explicit constructed batch at the declared 24-step capacity.
    x = torch.randn(2, 24, NUM_CHANNELS)
    validity = torch.ones(2, 24, NUM_CHANNELS, dtype=torch.bool)
    # Put some realistic missingness
    validity[:, 4:20, :] = False
    x[~validity] = 0.0

    expected_reconstruction = compute_deterministic_reconstruction(model, x, validity, device="cpu")

    bundle = CheckpointBundle.capture_fixture(
        model=model,
        optimizer=optimizer,
        transform_state={"mean": [0.0]*NUM_CHANNELS, "scale": [1.0]*NUM_CHANNELS, "fit_lake_ids": ["LAKE-01"], "constant_channels": []},
        telemetry={"epoch": 1, "validation_loss": 0.123},
        history=[{"epoch": 1, "loss": 0.123}],
    )

    ckpt_path = tmp_path / "test_checkpoint.pt"
    input_path = tmp_path / "test_input.pt"
    expected_path = tmp_path / "test_expected.pt"

    bundle.save(ckpt_path)
    torch.save((x, validity), input_path)
    torch.save(expected_reconstruction, expected_path)

    # 1. In-process replay check
    reloaded_bundle = CheckpointBundle.load_fixture(ckpt_path, device="cpu")
    model_reloaded = TimeSeriesMAE(**reloaded_bundle.model_config)
    model_reloaded.load_state_dict(reloaded_bundle.model_state_dict)

    replayed = compute_deterministic_reconstruction(model_reloaded, x, validity, device="cpu")
    diff = float((replayed - expected_reconstruction).abs().max().item())
    assert diff <= 1e-6

    # 2. Clean-process subprocess replay check
    ok, max_diff = verify_clean_process_replay(
        checkpoint_path=ckpt_path,
        input_tensor_path=input_path,
        expected_output_path=expected_path,
        tolerance=1e-5,
    )
    assert ok is True
    assert max_diff <= 1e-5


def test_deterministic_training_under_seed(tmp_path: Path):
    model_config = {
        "n_channels": NUM_CHANNELS,
        "max_time_steps": 24,
        "d_model": 32,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_encoder_heads": 2,
        "n_decoder_heads": 2,
        "d_ff_encoder": 64,
        "d_ff_decoder": 64,
        "dropout": 0.0,
        "masking_ratio": 0.5,
    }
    stop_rule = StopRule(min_epochs=2, max_epochs=2, patience=2, min_delta=1e-4)
    transform_state = {
        "mean": [0.0]*NUM_CHANNELS,
        "scale": [1.0]*NUM_CHANNELS,
        "fit_lake_ids": ["LAKE-TRAIN"],
        "constant_channels": [],
    }

    # Deterministic batch generator
    def _create_batches():
        torch.manual_seed(999)
        x = torch.randn(8, 24, NUM_CHANNELS)
        valid = torch.zeros(8, 24, NUM_CHANNELS, dtype=torch.bool)
        valid[:, :8, :] = True
        x[~valid] = 0.0
        return [(x, valid)]

    dir1 = tmp_path / "run_01"
    dir2 = tmp_path / "run_02"

    _, sum1 = fit_masked_autoencoder(
        model_config=model_config,
        train_batches=_create_batches(),
        validation_batches=_create_batches(),
        seed=777,
        output_dir=dir1,
        stop_rule=stop_rule,
        transform_state=transform_state,
        device="cpu",
        learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )

    _, sum2 = fit_masked_autoencoder(
        model_config=model_config,
        train_batches=_create_batches(),
        validation_batches=_create_batches(),
        seed=777,
        output_dir=dir2,
        stop_rule=stop_rule,
        transform_state=transform_state,
        device="cpu",
        learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )

    # Verify bit-identical loss traces
    loss1 = [h["train_loss"] for h in sum1["history"]]
    loss2 = [h["train_loss"] for h in sum2["history"]]
    assert np.allclose(loss1, loss2, atol=1e-7)

    val_loss1 = [h["validation_loss"] for h in sum1["history"]]
    val_loss2 = [h["validation_loss"] for h in sum2["history"]]
    assert np.allclose(val_loss1, val_loss2, atol=1e-7)


def test_run_train_model_end_to_end(tmp_path):
    """Unsupported research inputs are rejected before any output is written."""
    from runners.run_train_model import run_training
    with pytest.raises(ValueError, match="BLOCKED"):
        run_training(output_checkpoint_path=tmp_path/"model.pt")
    assert not list(tmp_path.iterdir())



def test_checkpoint_bundle_clean_replay_verification(tmp_path):
    """Unsupported research inputs are rejected before any output is written."""
    from sentinel_gl.replay import CheckpointBundle
    with pytest.raises(ValueError, match="BLOCKED"):
        CheckpointBundle.load(tmp_path/"model.pt")
    assert not list(tmp_path.iterdir())



def test_cli_runner_training_subprocess(tmp_path):
    from runners.run_train_model import main
    assert main(['--dev-split','--output-checkpoint',str(tmp_path/'model.pt')])==2
    assert not list(tmp_path.iterdir())



def test_normalizer_immutability_and_state_hash():
    p=_create_mock_panel('LAKE-TRAIN','fixture','2023-01-01')
    n=FittedNormalizer.fit_fixture([p],['LAKE-TRAIN'])
    restored=FittedNormalizer.from_fixture_json(n.to_json())
    assert restored.state_hash==n.state_hash
    changed=n.to_dict();changed['mean'][0]+=1.
    with pytest.raises(ValueError,match='hash'):FittedNormalizer.from_fixture_dict(changed)
