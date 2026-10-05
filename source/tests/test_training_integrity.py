"""Training integrity, loader isolation, normalizer isolation, and checkpoint replay tests.

FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
Synthetic panels, split allocations, and small mock model weights in this file verify
training loaders, memory bounds, normalizer isolation, and checkpoint reproducibility
in offline unit tests only. They do not represent real observations and must never
support scientific claims.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
import torch

from sentinel_gl.features import (
    FEATURE_CHANNELS,
    NUM_CHANNELS,
    WINDOW_DAYS,
    StaticTopography,
    MultiModalPanel,
)
from sentinel_gl.splits import SplitManifest, DecisionWindow
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


def _create_mock_panel(
    lake_id: str,
    window_id: str,
    start_date: str = "2023-01-01",
    end_date: str = "2023-06-29",
    empty: bool = False,
) -> MultiModalPanel:
    """Helper to create small valid MultiModalPanel for loader testing."""
    topo = StaticTopography(elevation_m=5200.0, moraine_slope_deg=25.0, catchment_area_km2=10.0)
    dates = tuple(f"2023-01-{i:02d}" for i in range(1, 181))

    if empty:
        values = np.full((WINDOW_DAYS, NUM_CHANNELS), np.nan, dtype=np.float64)
        mask = np.zeros((WINDOW_DAYS, NUM_CHANNELS), dtype=np.bool_)
    else:
        values = np.full((WINDOW_DAYS, NUM_CHANNELS), np.nan, dtype=np.float64)
        mask = np.zeros((WINDOW_DAYS, NUM_CHANNELS), dtype=np.bool_)
        # Populate 10 observed days
        for day in range(10):
            values[day, 0] = 1.35  # area
            values[day, 1] = 0.45  # ndwi
            values[day, 2] = 0.50  # mndwi
            values[day, 3] = 0.35  # ndsi
            values[day, 4] = -16.0 # vv_db
            values[day, 5] = -24.0 # vh_db
            values[day, 6] = -8.0  # cross_ratio
            values[day, 7] = 0.005 # vv_var
            values[day, 8] = -4.5  # temp_c
            values[day, 9] = 1.2   # precip_mm
            values[day, 10] = -0.5 # anomaly
            mask[day, :] = True

    return MultiModalPanel(
        lake_id=lake_id,
        window_id=window_id,
        start_date=start_date,
        end_date=end_date,
        dates=dates,
        values=values,
        mask=mask,
        static_metadata=topo,
    )


def test_loader_respects_split_boundaries():
    train_panel = _create_mock_panel("LAKE-TRAIN", "WIN-TRAIN-01", "2023-01-01", "2023-06-29")
    test_panel = _create_mock_panel("LAKE-TEST", "WIN-TEST-01", "2023-01-01", "2023-06-29")

    purged_win = DecisionWindow(
        window_id="WIN-PURGED-01",
        lake_id="LAKE-TRAIN",
        decision_timestamp="2023-02-01T00:00:00Z",
        context_start="2022-08-05T00:00:00Z",
        context_end="2023-01-31T00:00:00Z",
        observation_ids=(),
        is_eligible=False,
        modalities_present=(),
    )

    manifest = SplitManifest(
        split_id="SPLIT-TEST",
        protocol="spatiotemporal_purged_cluster",
        created_at="2026-10-05T00:00:00Z",
        clusters={"C1": ["LAKE-TRAIN"], "C2": ["LAKE-TEST"]},
        train_lakes=("LAKE-TRAIN",),
        val_lakes=(),
        test_lakes=("LAKE-TEST",),
        train_windows=(),
        val_windows=(),
        test_windows=(),
        purged_windows=(purged_win,),
        metadata={"split_date": "2023-07-01T00:00:00Z"},
    )

    # 1. Valid training panel succeeds
    ds_train = PanelDataset([train_panel], split_manifest=manifest, split="train")
    assert len(ds_train) == 1
    x, mask, meta = ds_train[0]
    assert x.shape == (180, 11)
    assert mask.shape == (180, 11)

    # 2. Feeding test lake panel to train dataset raises ValueError
    with pytest.raises(ValueError, match="not in allowed train lakes"):
        PanelDataset([test_panel], split_manifest=manifest, split="train")

    # 3. Feeding purged window raises ValueError
    purged_panel = _create_mock_panel("LAKE-TRAIN", "WIN-PURGED-01")
    with pytest.raises(ValueError, match="in purged boundary set"):
        PanelDataset([purged_panel], split_manifest=manifest, split="train")

    # 4. Training panel extending past split date cutoff raises ValueError
    late_panel = _create_mock_panel("LAKE-TRAIN", "WIN-LATE-01", "2023-05-01", "2023-10-27")
    with pytest.raises(ValueError, match="violates retrospective split cutoff"):
        PanelDataset([late_panel], split_manifest=manifest, split="train")


def test_loader_rejects_unobserved_batches():
    empty_panel = _create_mock_panel("LAKE-TRAIN", "WIN-EMPTY-01", empty=True)
    item = (
        torch.from_numpy(np.zeros((180, 11), dtype=np.float32)),
        torch.from_numpy(np.zeros((180, 11), dtype=bool)),
        empty_panel.to_dict(),
    )

    with pytest.raises(ValueError, match="Batch contains zero valid observations"):
        deterministic_collate([item])


def test_fitted_normalizer_isolation():
    # Training panels on LAKE-TRAIN
    train_panel = _create_mock_panel("LAKE-TRAIN", "W-TR-01", "2023-01-01", "2023-06-29")
    norm_tr = FittedNormalizer.fit([train_panel], training_lake_ids=["LAKE-TRAIN"])

    # Evaluation panel on LAKE-EVAL with extreme physical values (e.g. area = 4.8, temp = 30)
    eval_panel = _create_mock_panel("LAKE-EVAL", "W-EV-01", "2023-01-01", "2023-06-29")
    eval_panel.values[0, 0] = 4.8
    eval_panel.values[0, 8] = 30.0

    # Normalizer fit with both train and eval panels passed, but training_lake_ids strictly LAKE-TRAIN
    norm_isolated = FittedNormalizer.fit([train_panel, eval_panel], training_lake_ids=["LAKE-TRAIN"])

    # Assert exact state identity: eval panel has zero influence
    assert norm_tr.mean == norm_isolated.mean
    assert norm_tr.scale == norm_isolated.scale
    assert norm_tr.state_hash == norm_isolated.state_hash

    # Serialization roundtrip
    serialized = norm_isolated.to_json()
    restored = FittedNormalizer.from_json(serialized)
    assert restored.state_hash == norm_isolated.state_hash


def test_checkpoint_roundtrip_and_replay(tmp_path: Path):
    torch.manual_seed(42)
    config = {
        "n_channels": 11,
        "max_time_steps": 180,
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

    # Synthetic batch of shape (2, 180, 11)
    x = torch.randn(2, 180, 11)
    validity = torch.ones(2, 180, 11, dtype=torch.bool)
    # Put some realistic missingness
    validity[:, 10:170, :] = False
    x[~validity] = 0.0

    expected_reconstruction = compute_deterministic_reconstruction(model, x, validity, device="cpu")

    bundle = CheckpointBundle.capture(
        model=model,
        optimizer=optimizer,
        transform_state={"mean": [0.0]*11, "scale": [1.0]*11, "fit_lake_ids": ["LAKE-01"], "constant_channels": []},
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
    reloaded_bundle = CheckpointBundle.load(ckpt_path, device="cpu")
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
        "n_channels": 11,
        "max_time_steps": 180,
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
        "mean": [0.0]*11,
        "scale": [1.0]*11,
        "fit_lake_ids": ["LAKE-TRAIN"],
        "constant_channels": [],
    }

    # Deterministic batch generator
    def _create_batches():
        torch.manual_seed(999)
        x = torch.randn(8, 180, 11)
        valid = torch.zeros(8, 180, 11, dtype=torch.bool)
        valid[:, :20, :] = True
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
