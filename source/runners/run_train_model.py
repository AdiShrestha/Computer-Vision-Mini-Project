#!/usr/bin/env python3
"""Training CLI Runner for Sentinel-GL Temporal Masked Autoencoder (T-MAE).

Ingests standardized multi-modal feature panels, spatiotemporal split manifests,
and lake topography registries to train the Temporal Masked Autoencoder.
Enforces training-only normalizer isolation, worker seed determinism, clean-process
checkpoint replay verification, and structured telemetry emission.
"""
from __future__ import annotations
import argparse
import csv
import dataclasses
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

# Ensure source/ is discoverable under isolated Python runtimes (-s -B)
RUNNER_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUNNER_DIR.parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from sentinel_gl.features import (
    FEATURE_CHANNELS,
    NUM_CHANNELS,
    WINDOW_DAYS,
    MultiModalPanel,
    StaticTopography,
)
from sentinel_gl.normalization import FittedNormalizer
from sentinel_gl.splits import SplitManifest, DecisionWindow
from sentinel_gl.loaders import PanelDataset, create_panel_dataloader
from sentinel_gl.training import StopRule, fit_masked_autoencoder
from sentinel_gl.replay import (
    CheckpointBundle,
    compute_deterministic_reconstruction,
    verify_clean_process_replay,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_TOPOS: Dict[str, StaticTopography] = {
    "SGL-001": StaticTopography(elevation_m=5200.0, moraine_slope_deg=28.5, catchment_area_km2=15.2),
    "SGL-002": StaticTopography(elevation_m=5250.0, moraine_slope_deg=26.0, catchment_area_km2=18.5),
}


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 byte digest for a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_lake_topography(registry_path: Path) -> Dict[str, StaticTopography]:
    """Load lake static topography parameters from registry CSV or fallbacks."""
    topos: Dict[str, StaticTopography] = dict(DEFAULT_TOPOS)
    if not registry_path.exists():
        return topos

    with open(registry_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lid = row["lake_id"].strip()
            elev = float(row.get("elevation_m", 5200.0))
            area = float(row.get("area_km2_2020", 1.35))
            if lid not in topos:
                topos[lid] = StaticTopography(
                    elevation_m=elev,
                    moraine_slope_deg=25.0,
                    catchment_area_km2=max(1.0, area * 8.0),
                )
    return topos


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal Masked Autoencoder (T-MAE) training and checkpoint replay runner."
    )
    parser.add_argument(
        "--manifest",
        "--split-manifest",
        dest="manifest",
        type=Path,
        default=Path("data/split_manifest.json"),
        help="Path to spatiotemporal split manifest JSON (default: data/split_manifest.json).",
    )
    parser.add_argument(
        "--panels",
        type=Path,
        default=Path("data/feature_panels.npz"),
        help="Path to multi-modal feature panels NPZ archive (default: data/feature_panels.npz).",
    )
    parser.add_argument(
        "--lake-registry",
        "--registry",
        dest="lake_registry",
        type=Path,
        default=Path("data/lake_registry.csv"),
        help="Path to lake registry CSV (default: data/lake_registry.csv).",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("data/checkpoints/tmae_best_checkpoint.pt"),
        help="Destination path for serialized CheckpointBundle (default: data/checkpoints/tmae_best_checkpoint.pt).",
    )
    parser.add_argument(
        "--output-normalizer",
        type=Path,
        default=Path("data/fitted_normalizer.json"),
        help="Destination path for immutable FittedNormalizer JSON (default: data/fitted_normalizer.json).",
    )
    parser.add_argument(
        "--output-history",
        type=Path,
        default=Path("data/training_history.json"),
        help="Destination path for training history JSON (default: data/training_history.json).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Maximum training epochs (default: 5).",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=2,
        help="Minimum training epochs before early stopping (default: 2).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience in epochs (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size within memory bounds [8, 16] (default: 8).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW optimizer learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay coefficient (default: 1e-4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic RNG seed (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Target PyTorch device (default: cpu).",
    )
    parser.add_argument(
        "--dev-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When train_windows is empty, partition eligible windows into train/val subsets for development verification (default: True).",
    )
    return parser.parse_args(argv)


def run_training(
    manifest_path: Path,
    panels_path: Path,
    lake_registry_path: Path,
    output_checkpoint_path: Path,
    output_normalizer_path: Path,
    output_history_path: Path,
    epochs: int = 5,
    min_epochs: int = 2,
    patience: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
    dev_split: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end model training, normalizer fitting, and replay verification."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found at {manifest_path}")
    if not panels_path.exists():
        raise FileNotFoundError(f"Feature panels archive not found at {panels_path}")

    manifest_sha256 = compute_file_sha256(manifest_path)
    panels_sha256 = compute_file_sha256(panels_path)
    registry_sha256 = compute_file_sha256(lake_registry_path) if lake_registry_path.exists() else None

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    topos = load_lake_topography(lake_registry_path)
    npz_data = np.load(panels_path)

    # Reconstruct MultiModalPanel objects
    all_raw_windows = (
        manifest_data.get("train_windows", [])
        + manifest_data.get("val_windows", [])
        + manifest_data.get("test_windows", [])
    )

    panels_by_id: Dict[str, MultiModalPanel] = {}
    windows_by_id: Dict[str, DecisionWindow] = {}

    for w in all_raw_windows:
        wid = w["window_id"]
        lid = w["lake_id"]
        topo = topos.get(lid, DEFAULT_TOPOS.get(lid, StaticTopography(5200.0, 25.0, 10.0)))

        dates_key = f"{wid}_dates"
        vals_key = f"{wid}_values"
        mask_key = f"{wid}_mask" if f"{wid}_mask" in npz_data else f"{wid}_masks"

        if vals_key not in npz_data:
            continue

        dates = tuple(str(d) for d in npz_data[dates_key])
        vals = npz_data[vals_key]
        mask = npz_data[mask_key]

        obs_hashes = w.get("metadata", {}).get("observation_hashes", {})

        panel = MultiModalPanel(
            lake_id=lid,
            window_id=wid,
            start_date=dates[0],
            end_date=dates[-1],
            dates=dates,
            values=vals,
            mask=mask,
            static_metadata=topo,
            provenance_hashes=obs_hashes,
        )
        panels_by_id[wid] = panel

        dw = DecisionWindow(
            window_id=wid,
            lake_id=lid,
            decision_timestamp=w["decision_timestamp"],
            context_start=w["context_start"],
            context_end=w["context_end"],
            observation_ids=tuple(w.get("observation_ids", ())),
            is_eligible=w.get("is_eligible", False),
            modalities_present=tuple(w.get("modalities_present", ())),
            missingness_reason=w.get("missingness_reason"),
            status=w.get("status", "ELIGIBLE"),
            metadata=w.get("metadata", {}),
        )
        windows_by_id[wid] = dw

    # Handle split allocation: Genuine manifest splits vs Dev Partition
    raw_train_windows = manifest_data.get("train_windows", [])
    raw_val_windows = manifest_data.get("val_windows", [])

    if raw_train_windows:
        train_wins = [windows_by_id[w["window_id"]] for w in raw_train_windows if w["window_id"] in windows_by_id and w.get("is_eligible", False)]
        val_wins = [windows_by_id[w["window_id"]] for w in raw_val_windows if w["window_id"] in windows_by_id and w.get("is_eligible", False)]
        train_lakes = tuple(manifest_data.get("train_lakes", []))
        val_lakes = tuple(manifest_data.get("val_lakes", []))
        active_manifest = SplitManifest(
            split_id=manifest_data.get("split_id", "SPLIT-GENUINE"),
            protocol=manifest_data.get("protocol", "spatiotemporal_purged_cluster"),
            created_at=manifest_data.get("created_at", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            clusters=manifest_data.get("clusters", {}),
            train_lakes=train_lakes,
            val_lakes=val_lakes,
            test_lakes=tuple(manifest_data.get("test_lakes", [])),
            train_windows=tuple(train_wins),
            val_windows=tuple(val_wins),
            test_windows=tuple(windows_by_id[w["window_id"]] for w in manifest_data.get("test_windows", []) if w["window_id"] in windows_by_id),
            purged_windows=(),
            metadata=manifest_data.get("metadata", {}),
        )
    else:
        if not dev_split:
            raise ValueError(
                "Split manifest contains zero train_windows, and dev_split is disabled. "
                "Enable --dev-split to partition eligible windows for development verification."
            )
        # Development split mode: partition eligible windows into train and val subsets
        eligible_wins = [
            w for w in windows_by_id.values() if w.is_eligible and w.window_id in panels_by_id
        ]
        eligible_wins.sort(key=lambda w: (w.lake_id, w.decision_timestamp))
        if len(eligible_wins) < 2:
            raise ValueError(f"Insufficient eligible windows for dev split: found {len(eligible_wins)}")

        # Partition: 2/3 train (minimum 8 if available), remaining val
        n_train = max(1, min(len(eligible_wins) - 1, int(len(eligible_wins) * 2 / 3)))
        if len(eligible_wins) >= 12:
            n_train = 8  # 8 train, 4 val

        train_wins = eligible_wins[:n_train]
        val_wins = eligible_wins[n_train:]

        train_lakes = tuple(sorted(set(w.lake_id for w in train_wins)))
        val_lakes = tuple(sorted(set(w.lake_id for w in val_wins)))

        active_manifest = SplitManifest(
            split_id=manifest_data.get("split_id", "SPLIT-DEV") + "-DEV",
            protocol="development_partition",
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            clusters=manifest_data.get("clusters", {}),
            train_lakes=train_lakes,
            val_lakes=val_lakes,
            test_lakes=(),
            train_windows=tuple(train_wins),
            val_windows=tuple(val_wins),
            test_windows=(),
            purged_windows=(),
            metadata={"split_date": None, "mode": "development_verification"},
        )

    train_panels = [panels_by_id[w.window_id] for w in train_wins]
    val_panels = [panels_by_id[w.window_id] for w in val_wins]

    if not train_panels or not val_panels:
        raise ValueError(
            f"Cannot train model: train_panels count={len(train_panels)}, val_panels count={len(val_panels)}"
        )

    # -------------------------------------------------------------------------
    # 1. Fit FittedNormalizer strictly on training split panels
    # -------------------------------------------------------------------------
    normalizer = FittedNormalizer.fit(
        panels=train_panels,
        training_lake_ids=active_manifest.train_lakes,
    )

    output_normalizer_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_normalizer_path, "w", encoding="utf-8") as f:
        f.write(normalizer.to_json(indent=2) + "\n")
    normalizer_sha256 = compute_file_sha256(output_normalizer_path)

    # -------------------------------------------------------------------------
    # 2. Build datasets and data loaders
    # -------------------------------------------------------------------------
    train_dataset = PanelDataset(
        panels=train_panels,
        split_manifest=active_manifest,
        split="train",
        normalizer=normalizer,
    )
    val_dataset = PanelDataset(
        panels=val_panels,
        split_manifest=active_manifest,
        split="val",
        normalizer=normalizer,
    )

    eff_train_batch_size = min(batch_size, len(train_dataset))
    eff_val_batch_size = min(batch_size, len(val_dataset))

    train_loader = create_panel_dataloader(
        dataset=train_dataset,
        batch_size=eff_train_batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = create_panel_dataloader(
        dataset=val_dataset,
        batch_size=eff_val_batch_size,
        shuffle=False,
        seed=seed,
    )

    train_batches = [(x, valid) for x, valid, meta in train_loader]
    val_batches = [(x, valid) for x, valid, meta in val_loader]

    # -------------------------------------------------------------------------
    # 3. Model Architecture and Training
    # -------------------------------------------------------------------------
    model_config = {
        "n_channels": NUM_CHANNELS,
        "max_time_steps": WINDOW_DAYS,
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

    stop_rule = StopRule(
        min_epochs=min_epochs,
        max_epochs=epochs,
        patience=patience,
        min_delta=1e-4,
    )

    transform_state = {
        "mean": list(normalizer.mean),
        "scale": list(normalizer.scale),
        "fit_lake_ids": list(normalizer.training_lake_ids),
        "constant_channels": list(normalizer.constant_channels),
    }

    temp_train_dir = tempfile.mkdtemp(prefix="tmae_train_run_")
    run_dir = Path(temp_train_dir) / "run"

    try:
        model, summary = fit_masked_autoencoder(
            model_config=model_config,
            train_batches=train_batches,
            validation_batches=val_batches,
            seed=seed,
            output_dir=run_dir,
            stop_rule=stop_rule,
            transform_state=transform_state,
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=1.0,
        )

        # ---------------------------------------------------------------------
        # 4. Capture & Save Checkpoint Bundle
        # ---------------------------------------------------------------------
        telemetry = {
            "status": summary["status"],
            "seed": seed,
            "device": device,
            "checkpoint_epoch": summary["checkpoint_epoch"],
            "best_validation_loss": summary["best_validation_loss"],
            "elapsed_wall_seconds": summary["elapsed_wall_seconds"],
            "normalizer_state_hash": normalizer.state_hash,
            "train_windows_count": len(train_panels),
            "val_windows_count": len(val_panels),
            "stop_rule": dataclasses.asdict(stop_rule),
        }

        bundle = CheckpointBundle.capture(
            model=model,
            optimizer=None,
            transform_state=normalizer.to_dict(),
            telemetry=telemetry,
            history=summary["history"],
            format_version=4,
        )

        output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        bundle.save(output_checkpoint_path)
        checkpoint_sha256 = compute_file_sha256(output_checkpoint_path)

        # ---------------------------------------------------------------------
        # 5. Clean-process Checkpoint Replay Verification
        # ---------------------------------------------------------------------
        val_x, val_valid = val_batches[0]
        sample_x = val_x[:1].cpu()
        sample_valid = val_valid[:1].cpu()
        expected_recon = compute_deterministic_reconstruction(model, sample_x, sample_valid, device="cpu")

        sample_in_path = Path(temp_train_dir) / "sample_input.pt"
        sample_exp_path = Path(temp_train_dir) / "sample_expected.pt"
        torch.save((sample_x, sample_valid), sample_in_path)
        torch.save(expected_recon, sample_exp_path)

        replay_ok, replay_max_diff = verify_clean_process_replay(
            checkpoint_path=output_checkpoint_path,
            input_tensor_path=sample_in_path,
            expected_output_path=sample_exp_path,
            tolerance=1e-5,
        )

        if not replay_ok or replay_max_diff > 1e-5:
            raise RuntimeError(
                f"Clean-process replay verification failed! ok={replay_ok}, max_diff={replay_max_diff:.8f}"
            )

    finally:
        shutil.rmtree(temp_train_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 6. Save Training History & Telemetry Document
    def to_rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(REPO_ROOT.resolve()))
        except ValueError:
            return str(p)

    rel_manifest = to_rel(manifest_path)
    rel_panels = to_rel(panels_path)
    rel_registry = to_rel(lake_registry_path)
    rel_checkpoint = to_rel(output_checkpoint_path)
    rel_normalizer = to_rel(output_normalizer_path)
    rel_history = to_rel(output_history_path)

    history_doc: Dict[str, Any] = {
        "status": "SUCCESS",
        "training_status": summary["status"],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": seed,
        "device": device,
        "epochs_trained": summary["epochs_trained"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "best_validation_loss": summary["best_validation_loss"],
        "elapsed_wall_seconds": summary["elapsed_wall_seconds"],
        "replay_verified": replay_ok,
        "replay_max_diff": replay_max_diff,
        "normalizer_state_hash": normalizer.state_hash,
        "stop_rule": dataclasses.asdict(stop_rule),
        "model_config": summary["model_config"],
        "history": summary["history"],
        "provenance": {
            "manifest_path": rel_manifest,
            "manifest_sha256": manifest_sha256,
            "panels_path": rel_panels,
            "panels_sha256": panels_sha256,
            "registry_path": rel_registry,
            "registry_sha256": registry_sha256,
            "checkpoint_path": rel_checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "normalizer_path": rel_normalizer,
            "normalizer_sha256": normalizer_sha256,
            "zero_synthetic_data_declaration": True,
        },
    }

    output_history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_history_path, "w", encoding="utf-8") as f:
        json.dump(history_doc, f, indent=2)

    return history_doc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        result = run_training(
            manifest_path=args.manifest,
            panels_path=args.panels,
            lake_registry_path=args.lake_registry,
            output_checkpoint_path=args.output_checkpoint,
            output_normalizer_path=args.output_normalizer,
            output_history_path=args.output_history,
            epochs=args.epochs,
            min_epochs=args.min_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            dev_split=args.dev_split,
        )
    except Exception as ex:
        err = {
            "status": "ERROR",
            "error": str(ex),
            "error_type": type(ex).__name__,
        }
        print(json.dumps(err, indent=2), file=sys.stderr)
        return 1

    cli_output = {
        "status": "SUCCESS",
        "training_status": result["training_status"],
        "epochs_trained": result["epochs_trained"],
        "checkpoint_epoch": result["checkpoint_epoch"],
        "best_validation_loss": result["best_validation_loss"],
        "replay_verified": result["replay_verified"],
        "replay_max_diff": result["replay_max_diff"],
        "normalizer_state_hash": result["normalizer_state_hash"],
        "checkpoint_path": str(args.output_checkpoint),
        "normalizer_path": str(args.output_normalizer),
        "history_path": str(args.output_history),
    }
    print(json.dumps(cli_output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
