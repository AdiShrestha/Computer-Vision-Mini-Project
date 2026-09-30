"""
Test Suite for Legacy Evidence Quarantine Enforcement.
Ensures INV-013, INV-015, INV-017, INV-026, and INV-039 are enforced.
"""

import pytest
from pathlib import Path
import json
from source.utils.quarantine import (
    QuarantineRegistry,
    QuarantinedArtifactError,
    assert_not_quarantined,
)


def test_quarantine_registry_loads():
    reg = QuarantineRegistry.load()
    assert len(reg.entries) >= 10
    assert reg.version == "2.0"


def test_quarantine_detects_tainted_paths():
    reg = QuarantineRegistry.load()
    tainted_paths = [
        "data/raw/sentinel1_grd/lake_01.csv",
        "data/features/lake_features.npz",
        "models/checkpoints/tsmae_epoch_100.pt",
        "source/scripts/run_bootstrap_ci.py",
        "source/scripts/cloud_stratified_eval.py",
        "source/scripts/run_ablation.py",
    ]
    for path in tainted_paths:
        is_q, reason = reg.is_quarantined(path)
        assert is_q is True, f"Failed to detect tainted path: {path}"
        with pytest.raises(QuarantinedArtifactError):
            reg.assert_clean(path)


def test_quarantine_passes_clean_paths():
    reg = QuarantineRegistry.load()
    clean_paths = [
        "source/data/schemas/feature_schema_v2.py",
        "source/evaluation/prediction_ledger.py",
        "project/invariants.md",
    ]
    for path in clean_paths:
        is_q, _ = reg.is_quarantined(path)
        assert is_q is False
        reg.assert_clean(path)


def test_path_matching_is_boundary_aware_and_case_insensitive():
    reg = QuarantineRegistry.load()
    assert reg.is_quarantined("DATA\\RAW\\SENTINEL1_GRD\\lake.csv")[0] is True
    assert reg.is_quarantined("./data/raw/sentinel1_grd/lake.csv")[0] is True
    assert reg.is_quarantined("data/raw/sentinel1_grd_backup/lake.csv")[0] is False


def test_known_hashes_are_checked_by_convenience_helper(tmp_path):
    known_hash = "a" * 64
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": "2.0",
                "quarantined_artifacts": [
                    {
                        "entry_id": "QR-HASH",
                        "target_path_or_pattern": "results/tainted/",
                        "known_sha256": known_hash,
                        "taint_reason": "test taint",
                        "quarantined_since": "2026-08-27",
                        "replacement_contract": "TEST",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(QuarantinedArtifactError):
        assert_not_quarantined(known_hash, registry_path)
    assert_not_quarantined("b" * 64, registry_path)


def test_registry_rejects_duplicate_ids_and_malformed_hashes(tmp_path):
    base_entry = {
        "entry_id": "QR-1",
        "target_path_or_pattern": "results/tainted/",
        "known_sha256": None,
        "taint_reason": "test taint",
        "quarantined_since": "2026-08-27",
        "replacement_contract": "TEST",
    }
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        json.dumps({"version": "2.0", "quarantined_artifacts": [base_entry, dict(base_entry)]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="IDs must be unique"):
        QuarantineRegistry.load(duplicate_path)

    malformed = dict(base_entry, entry_id="QR-2", known_sha256="not-a-hash")
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text(
        json.dumps({"version": "2.0", "quarantined_artifacts": [malformed]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="64-character"):
        QuarantineRegistry.load(malformed_path)
