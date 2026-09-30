"""
Test Suite for Task A/B/C Protocols and Calibration Policies.
Ensures INV-007, INV-009, INV-011, INV-019, INV-023, and INV-034 are strictly enforced.
"""

import pytest
import numpy as np
from source.evaluation.protocols.task_protocols import (
    TaskType,
    EvaluationInstance,
    InformationBudget,
    assert_common_information_budget,
    assert_common_evaluation_instances,
    CalibrationPolicy,
    TaskAProtocol,
    TaskBProtocol,
    TaskCProtocol,
    ScoreCNormalizer,
)


def test_task_b_date_chronology():
    # Pre-event window ending 2023-09-15
    res_pre = TaskBProtocol.classify_window("2023-09-15")
    assert res_pre["is_pre_event"] is True
    assert res_pre["is_post_event"] is False
    assert res_pre["lead_time_days"] == 19

    # Post-event window ending 2023-10-15
    res_post = TaskBProtocol.classify_window("2023-10-15")
    assert res_post["is_pre_event"] is False
    assert res_post["is_post_event"] is True
    assert res_post["lead_time_days"] < 0

    res_event = TaskBProtocol.classify_window("2023-10-04")
    assert res_event["classification"] == "event_overlap"
    assert res_event["is_pre_event"] is False
    assert res_event["is_post_event"] is False
    assert res_event["lead_time_days"] == 0

    res_overlap = TaskBProtocol.classify_window("2023-10-05", "2023-10-03")
    assert res_overlap["classification"] == "event_overlap"
    assert res_overlap["is_event_overlap"] is True


def test_calibration_policy_enforces_fpr_constraint():
    policy = CalibrationPolicy(target_max_fpr=0.10)
    np.random.seed(42)
    clean_scores = np.random.normal(loc=0.2, scale=0.05, size=100)
    anomaly_scores = np.random.normal(loc=0.8, scale=0.1, size=20)

    all_scores = np.concatenate([clean_scores, anomaly_scores])
    all_labels = np.concatenate([np.zeros(100), np.ones(20)])

    thresh, meta = policy.fit_threshold(
        all_scores,
        all_labels,
        calibration_instance_ids=tuple(f"cal-{i}" for i in range(len(all_scores))),
        final_evaluation_instance_ids=("event-01",),
    )
    assert thresh is not None
    assert meta["status"] == "FEASIBLE_THRESHOLD_FOUND"
    assert meta["observed_fpr"] <= 0.10
    assert meta["inv_007_compliant"] is True


def test_calibration_policy_no_feasible_threshold():
    policy = CalibrationPolicy(target_max_fpr=0.01)  # Strict constraint
    clean_scores = np.arange(10, dtype=float)
    anomaly_scores = np.arange(5, dtype=float)
    all_scores = np.concatenate([clean_scores, anomaly_scores])
    all_labels = np.concatenate([np.zeros(10), np.ones(5)])
    thresh, meta = policy.fit_threshold(
        all_scores,
        all_labels,
        calibration_instance_ids=tuple(f"cal-{i}" for i in range(len(all_scores))),
        final_evaluation_instance_ids=("event-01",),
    )
    assert thresh is None
    assert meta["status"] == "NO_FEASIBLE_THRESHOLD"
    assert meta["inv_007_compliant"] is False


def test_score_c_normalization_isolated():
    # Training-fitted bounds
    normalizer = ScoreCNormalizer(
        score_a_min=0.0,
        score_a_max=10.0,
        score_b_min=0.0,
        score_b_max=5.0,
        alpha=0.5,
        fit_lake_ids=("TRAIN-01",),
        evaluation_lake_ids=("CONTROL-01",),
    )
    score_a = np.array([5.0, 12.0, -1.0])
    score_b = np.array([2.5, 6.0, 0.0])

    score_c = normalizer.compute_score_c(score_a, score_b)
    # 5.0 -> 0.5, 2.5 -> 0.5 => 0.5
    assert np.isclose(score_c[0], 0.5)
    # 12.0 clipped to 1.0, 6.0 clipped to 1.0 => 1.0
    assert np.isclose(score_c[1], 1.0)
    # -1.0 clipped to 0.0, 0.0 -> 0.0 => 0.0
    assert np.isclose(score_c[2], 0.0)


def instance(instance_id="i1", label=0):
    return EvaluationInstance(
        instance_id=instance_id,
        task_type=TaskType.TASK_A_PERTURBATION,
        lake_id="CONTROL-01",
        window_id="w1",
        start_date="2020-01-01",
        end_date="2020-01-07",
        center_date="2020-01-04",
        ground_truth_label=label,
        is_pre_event=False,
        is_post_event=False,
        perturbation_type="area_growth" if label else None,
        perturbation_magnitude=0.2 if label else 0.0,
    )


def budget(feature_hash="sha256:feature-v2"):
    return InformationBudget(
        instance_ids=("i1", "i2"),
        labels=(0, 1),
        feature_schema_hash=feature_hash,
        temporal_cutoff="2023-10-03",
        allowed_fit_lake_ids=("TRAIN-01", "TRAIN-02"),
        calibration_instance_ids=("cal-01",),
    )


def test_common_instances_and_information_budget_are_enforced():
    first = [instance("i1", 0), instance("i2", 1)]
    assert_common_evaluation_instances({"ts_mae": first, "baseline": list(first)})
    with pytest.raises(ValueError, match="identical labeled evaluation instances"):
        assert_common_evaluation_instances({"ts_mae": first, "baseline": [instance("i1", 1), instance("i2", 1)]})

    assert_common_information_budget({"ts_mae": budget(), "baseline": budget()})
    with pytest.raises(ValueError, match="identical information budget"):
        assert_common_information_budget({"ts_mae": budget(), "baseline": budget("other")})


def test_task_specific_instance_semantics_are_enforced():
    TaskAProtocol.validate_instances([instance()])
    controls = [
        EvaluationInstance(
            instance_id=f"c{i}", task_type=TaskType.TASK_C_CONTROL_FALSE_ALERTS,
            lake_id=f"CONTROL-{i}", window_id=f"w{i}",
            start_date="2020-01-01", end_date="2020-01-07", center_date="2020-01-04",
            ground_truth_label=0, is_pre_event=False, is_post_event=False,
        ) for i in range(4)
    ]
    TaskCProtocol.validate_control_lakes([c.lake_id for c in controls])
    TaskCProtocol.validate_instances(controls)
    with pytest.raises(ValueError, match="exactly 4"):
        TaskCProtocol.validate_control_lakes(["CONTROL-1"])


def test_score_c_isolation_and_boundaries_are_enforced():
    with pytest.raises(ValueError, match="fit and final evaluation lakes"):
        ScoreCNormalizer(0.0, 1.0, 0.0, 1.0, fit_lake_ids=("LK-1",), evaluation_lake_ids=("LK-1",))
    normalizer = ScoreCNormalizer(0.0, 1.0, 0.0, 1.0, fit_lake_ids=("TRAIN",), evaluation_lake_ids=("TEST",))
    with pytest.raises(ValueError, match="identical non-scalar shapes"):
        normalizer.compute_score_c(np.array([0.2]), np.array([[0.2]]))


def test_threshold_calibration_rejects_instance_overlap():
    policy = CalibrationPolicy()
    with pytest.raises(ValueError, match="must be disjoint"):
        policy.fit_threshold(
            np.array([0.1, 0.9]),
            np.array([0, 1]),
            calibration_instance_ids=("shared", "cal-1"),
            final_evaluation_instance_ids=("shared",),
        )
