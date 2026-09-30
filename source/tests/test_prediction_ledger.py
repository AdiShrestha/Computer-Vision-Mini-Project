"""
Test Suite for Prediction Ledger and Statistical Input Validator.
Ensures INV-010, INV-016, INV-017, INV-025, and INV-037 are enforced.
"""

import pytest
import numpy as np
from dataclasses import replace
from source.evaluation.prediction_ledger import (
    PredictionRow,
    PredictionLedger,
    StatisticalInputValidator,
)


def create_sample_row(instance_id="inst_001", method="TS-MAE", score=0.75, label=1):
    return PredictionRow(
        instance_id=instance_id,
        run_id="run_20260826_01",
        task_id="Task_A_Synthetic_Perturbation",
        lake_id="LK_CONTROL_01",
        window_id="win_050",
        window_start="2022-01-01",
        window_end="2022-06-30",
        window_center="2022-03-31",
        ground_truth_label=label,
        method_name=method,
        raw_anomaly_score=score,
        calibrated_anomaly_score=score,
        binary_decision=1 if score >= 0.5 else 0,
        decision_threshold=0.5,
        feature_schema_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        model_checkpoint_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        is_pre_event=False,
        is_post_event=False,
        perturbation_type="rapid_expansion",
        perturbation_magnitude=1.5,
    )


def test_ledger_creation_and_hashing():
    ledger = PredictionLedger()
    r1 = create_sample_row("inst_001", "TS-MAE", 0.8)
    r2 = create_sample_row("inst_002", "TS-MAE", 0.2, label=0)
    ledger.add_row(r1)
    ledger.add_row(r2)

    h1 = ledger.compute_ledger_hash()
    assert len(h1) == 64
    assert len(ledger.get_unique_lakes()) == 1
    assert ledger.finalize() == h1
    with pytest.raises(RuntimeError, match="finalized and immutable"):
        ledger.add_row(create_sample_row("inst_003", "TS-MAE", 0.1, label=0))
    with pytest.raises(RuntimeError, match="metadata is immutable"):
        ledger.rows = ()
    restored = PredictionLedger.from_json(ledger.to_json())
    assert restored.to_records() == ledger.to_records()


def test_validator_detects_duplicates():
    r1 = create_sample_row("inst_001", "TS-MAE")
    r2 = create_sample_row("inst_001", "TS-MAE")  # Duplicate
    ledger = PredictionLedger(rows=(r1, r2))

    errors = StatisticalInputValidator.validate_ledger_integrity(ledger)
    assert any("Duplicate prediction entry" in e for e in errors)


def test_validator_rejects_unanchored_statistical_inputs():
    scores = np.array([0.5, 0.6, 0.7])
    with pytest.raises(ValueError) as excinfo:
        StatisticalInputValidator.assert_no_manufactured_inputs(scores, source_ledger_hash=None)
    assert "Manufacturing pseudo-observation arrays is strictly prohibited" in str(excinfo.value)
    with pytest.raises(ValueError, match="PredictionLedger object is required"):
        StatisticalInputValidator.assert_no_manufactured_inputs(
            scores, source_ledger_hash="0" * 64
        )


def test_prediction_row_rejects_invalid_lineage_and_decision_fields():
    with pytest.raises(ValueError, match="SHA-256"):
        replace(create_sample_row(), feature_schema_hash="not-a-hash")
    with pytest.raises(ValueError, match="agree"):
        replace(create_sample_row(), binary_decision=0)


def test_statistical_inputs_are_extracted_from_matching_persisted_rows():
    r1 = create_sample_row("inst_001", "TS-MAE", 0.2, label=0)
    r2 = replace(
        create_sample_row("inst_002", "TS-MAE", 0.8, label=1),
        lake_id="LK_CONTROL_02",
    )
    ledger = PredictionLedger(rows=(r1, r2))
    ledger_hash = ledger.compute_ledger_hash()
    scores, labels, lakes = StatisticalInputValidator.extract_statistical_inputs(
        ledger,
        ledger_hash,
        task_id="Task_A_Synthetic_Perturbation",
        method_name="TS-MAE",
    )
    assert scores.tolist() == [0.2, 0.8]
    assert labels.tolist() == [0, 1]
    assert lakes == ("LK_CONTROL_01", "LK_CONTROL_02")
    assert set(StatisticalInputValidator.clustered_units(ledger)) == {
        "LK_CONTROL_01", "LK_CONTROL_02"
    }


def test_statistical_input_hash_and_estimability_are_fail_closed():
    r1 = create_sample_row("inst_001", "TS-MAE", 0.2, label=0)
    r2 = create_sample_row("inst_002", "TS-MAE", 0.8, label=1)
    ledger = PredictionLedger(rows=(r1, r2))
    with pytest.raises(ValueError, match="does not match ledger"):
        StatisticalInputValidator.extract_statistical_inputs(
            ledger,
            "0" * 64,
            task_id="Task_A_Synthetic_Perturbation",
            method_name="TS-MAE",
        )

    with pytest.raises(ValueError, match="NOT_ESTIMABLE"):
        StatisticalInputValidator.extract_statistical_inputs(
            PredictionLedger(rows=(r1,)),
            PredictionLedger(rows=(r1,)).compute_ledger_hash(),
            task_id="Task_A_Synthetic_Perturbation",
            method_name="TS-MAE",
        )
