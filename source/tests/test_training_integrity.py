"""Training integrity, loader isolation, normalizer isolation, and checkpoint replay tests.

FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
Synthetic panels, split allocations, and small mock model weights in this file verify
training loaders, memory bounds, normalizer isolation, and checkpoint reproducibility
in offline unit tests only. They do not represent real observations and must never
support scientific claims.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
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


def test_run_train_model_end_to_end(tmp_path: Path):
    """Verify run_training produces conforming checkpoint, normalizer, and history artifacts."""
    from runners.run_train_model import run_training

    repo_root = Path(__file__).resolve().parent.parent.parent
    manifest_path = repo_root / "data" / "split_manifest.json"
    panels_path = repo_root / "data" / "feature_panels.npz"
    registry_path = repo_root / "data" / "lake_registry.csv"

    out_ckpt = tmp_path / "checkpoints" / "test_model.pt"
    out_norm = tmp_path / "test_normalizer.json"
    out_hist = tmp_path / "test_history.json"

    res = run_training(
        manifest_path=manifest_path,
        panels_path=panels_path,
        lake_registry_path=registry_path,
        output_checkpoint_path=out_ckpt,
        output_normalizer_path=out_norm,
        output_history_path=out_hist,
        epochs=2,
        min_epochs=2,
        patience=2,
        batch_size=8,
        seed=123,
        device="cpu",
        dev_split=True,
    )

    assert res["status"] == "SUCCESS"
    assert res["epochs_trained"] == 2
    assert res["replay_verified"] is True
    assert res["replay_max_diff"] <= 1e-5
    assert out_ckpt.exists()
    assert out_norm.exists()
    assert out_hist.exists()

    with open(out_hist, "r", encoding="utf-8") as f:
        hist = json.load(f)
    assert hist["status"] == "SUCCESS"
    assert len(hist["history"]) == 2
    assert hist["replay_verified"] is True
    assert hist["normalizer_state_hash"] == res["normalizer_state_hash"]


def test_checkpoint_bundle_clean_replay_verification(tmp_path: Path):
    """Verify clean-process replay matches reconstructions within 1e-5 on genuine checkpoint."""
    from runners.run_train_model import run_training

    repo_root = Path(__file__).resolve().parent.parent.parent
    manifest_path = repo_root / "data" / "split_manifest.json"
    panels_path = repo_root / "data" / "feature_panels.npz"
    registry_path = repo_root / "data" / "lake_registry.csv"

    out_ckpt = tmp_path / "replay_check.pt"
    out_norm = tmp_path / "norm.json"
    out_hist = tmp_path / "hist.json"

    run_training(
        manifest_path=manifest_path,
        panels_path=panels_path,
        lake_registry_path=registry_path,
        output_checkpoint_path=out_ckpt,
        output_normalizer_path=out_norm,
        output_history_path=out_hist,
        epochs=2,
        min_epochs=2,
        seed=42,
        device="cpu",
        dev_split=True,
    )

    bundle = CheckpointBundle.load(out_ckpt, device="cpu")
    assert bundle.format_version == 4
    assert "n_channels" in bundle.model_config
    assert bundle.model_config["n_channels"] == NUM_CHANNELS

    # Construct normalized test input from an eligible window with observations
    npz_data = np.load(panels_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    eligible_id = [w["window_id"] for w in manifest_data["test_windows"] if w.get("is_eligible", False)][0]
    raw_vals = npz_data[f"{eligible_id}_values"][np.newaxis, ...]
    raw_mask = npz_data[f"{eligible_id}_mask"][np.newaxis, ...]
    normalizer = FittedNormalizer.from_dict(bundle.transform_state)
    norm_vals, norm_masks = normalizer.transform(raw_vals, raw_mask)
    vals = torch.from_numpy(norm_vals)
    masks = torch.from_numpy(norm_masks)

    in_file = tmp_path / "in.pt"
    exp_file = tmp_path / "exp.pt"
    torch.save((vals, masks), in_file)

    model = TimeSeriesMAE(**bundle.model_config)
    model.load_state_dict(bundle.model_state_dict)
    expected_recon = compute_deterministic_reconstruction(model, vals, masks, device="cpu")
    torch.save(expected_recon, exp_file)

    ok, max_diff = verify_clean_process_replay(
        checkpoint_path=out_ckpt,
        input_tensor_path=in_file,
        expected_output_path=exp_file,
        tolerance=1e-5,
    )
    assert ok is True
    assert max_diff <= 1e-5


def test_cli_runner_training_subprocess(tmp_path: Path):
    """Verify source/runners/run_train_model.py executes cleanly as CLI subprocess."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    runner_script = repo_root / "source" / "runners" / "run_train_model.py"

    out_ckpt = tmp_path / "cli_ckpt.pt"
    out_norm = tmp_path / "cli_norm.json"
    out_hist = tmp_path / "cli_hist.json"

    cmd = [
        sys.executable,
        "-B",
        str(runner_script),
        "--manifest",
        str(repo_root / "data" / "split_manifest.json"),
        "--panels",
        str(repo_root / "data" / "feature_panels.npz"),
        "--registry",
        str(repo_root / "data" / "lake_registry.csv"),
        "--output-checkpoint",
        str(out_ckpt),
        "--output-normalizer",
        str(out_norm),
        "--output-history",
        str(out_hist),
        "--epochs",
        "2",
        "--min-epochs",
        "2",
        "--batch-size",
        "8",
        "--seed",
        "99",
        "--dev-split",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stdout_json = json.loads(res.stdout.strip())
    assert stdout_json["status"] == "SUCCESS"
    assert stdout_json["epochs_trained"] == 2
    assert stdout_json["replay_verified"] is True
    assert out_ckpt.exists()
    assert out_norm.exists()
    assert out_hist.exists()


def test_normalizer_immutability_and_state_hash():
    """Verify normalizer state serialization and roundtrip immutability."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    norm_path = repo_root / "data" / "fitted_normalizer.json"
    assert norm_path.exists(), "data/fitted_normalizer.json must be materialized"

    with open(norm_path, "r", encoding="utf-8") as f:
        norm_json = f.read()

    normalizer = FittedNormalizer.from_json(norm_json)
    assert len(normalizer.mean) == NUM_CHANNELS
    assert len(normalizer.scale) == NUM_CHANNELS
    assert normalizer.channel_names == FEATURE_CHANNELS
    assert all(s > 0 for s in normalizer.scale)

    # Re-serialization matches state_hash exactly
    reserialized = normalizer.to_json()
    normalizer_reloaded = FittedNormalizer.from_json(reserialized)
    assert normalizer_reloaded.state_hash == normalizer.state_hash

    # Test transform & inverse transform consistency
    dummy_vals = np.ones((10, NUM_CHANNELS), dtype=np.float64)
    dummy_mask = np.ones((10, NUM_CHANNELS), dtype=np.bool_)
    dummy_mask[5:, :] = False

    z_norm, mask_out = normalizer.transform(dummy_vals, dummy_mask)
    assert np.all(z_norm[5:] == 0.0)
    assert np.all(np.isfinite(z_norm[:5]))

    inverted = normalizer.inverse_transform(z_norm, mask_out)
    assert np.all(np.isnan(inverted[5:]))
    assert np.allclose(inverted[:5], dummy_vals[:5])

