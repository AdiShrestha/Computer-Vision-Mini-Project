"""
Sentinel-GL — Canonical Evaluation Task Protocols and Calibration Policies.
Enforces Task A (Synthetic Perturbation), Task B (Retrospective Event), and Task C (False Alert).
Governed by INV-002, INV-007, INV-009, INV-011, INV-018, INV-019, INV-023, INV-032, INV-034.
"""

from typing import Dict, List, Any, Mapping, Optional, Sequence, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import datetime


class TaskType(str, Enum):
    TASK_A_PERTURBATION = "Task_A_Synthetic_Perturbation"
    TASK_B_RETROSPECTIVE_EVENT = "Task_B_Retrospective_South_Lhonak"
    TASK_C_CONTROL_FALSE_ALERTS = "Task_C_Control_False_Alerts"


@dataclass(frozen=True)
class EvaluationInstance:
    instance_id: str
    task_type: TaskType
    lake_id: str
    window_id: str
    start_date: str
    end_date: str
    center_date: str
    ground_truth_label: int  # 0: normal/background, 1: anomaly/perturbation/pre-event
    is_pre_event: bool
    is_post_event: bool
    perturbation_type: Optional[str] = None  # None for natural background
    perturbation_magnitude: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.task_type, TaskType):
            raise ValueError("task_type must be a TaskType value")
        if not all(isinstance(value, str) and value.strip() for value in (
            self.instance_id, self.lake_id, self.window_id,
            self.start_date, self.end_date, self.center_date,
        )):
            raise ValueError("EvaluationInstance identifiers and dates must be non-empty strings")
        dates = [
            datetime.date.fromisoformat(self.start_date),
            datetime.date.fromisoformat(self.center_date),
            datetime.date.fromisoformat(self.end_date),
        ]
        if not dates[0] <= dates[1] <= dates[2]:
            raise ValueError("EvaluationInstance dates must satisfy start <= center <= end")
        if type(self.ground_truth_label) is not int or self.ground_truth_label not in (0, 1):
            raise ValueError("ground_truth_label must be integer 0 or 1")
        if type(self.is_pre_event) is not bool or type(self.is_post_event) is not bool:
            raise ValueError("event flags must be boolean")
        if self.is_pre_event and self.is_post_event:
            raise ValueError("an instance cannot be both pre-event and post-event")
        if self.perturbation_type is not None and not self.perturbation_type.strip():
            raise ValueError("perturbation_type must be non-empty when supplied")
        if isinstance(self.perturbation_magnitude, bool) or not np.isfinite(self.perturbation_magnitude):
            raise ValueError("perturbation_magnitude must be finite")

    def comparison_key(self) -> Tuple[Any, ...]:
        """Identity/label tuple used for exact cross-method instance equality."""
        return (
            self.instance_id,
            self.lake_id,
            self.window_id,
            self.start_date,
            self.end_date,
            self.center_date,
            self.ground_truth_label,
            self.is_pre_event,
            self.is_post_event,
            self.perturbation_type,
            float(self.perturbation_magnitude),
        )


@dataclass(frozen=True)
class InformationBudget:
    """Comparable-method budget: instances, labels, features, and fit scope."""

    instance_ids: Tuple[str, ...]
    labels: Tuple[int, ...]
    feature_schema_hash: str
    temporal_cutoff: Optional[str]
    allowed_fit_lake_ids: Tuple[str, ...]
    calibration_instance_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.instance_ids or len(self.instance_ids) != len(self.labels):
            raise ValueError("information budget requires equally sized non-empty IDs and labels")
        if len(set(self.instance_ids)) != len(self.instance_ids):
            raise ValueError("information-budget instance IDs must be unique")
        if any(type(label) is not int or label not in (0, 1) for label in self.labels):
            raise ValueError("information-budget labels must be integer 0/1 values")
        if not isinstance(self.feature_schema_hash, str) or not self.feature_schema_hash.strip():
            raise ValueError("feature_schema_hash must be a non-empty string")
        if any(not isinstance(lake_id, str) or not lake_id.strip() for lake_id in self.allowed_fit_lake_ids):
            raise ValueError("allowed_fit_lake_ids must contain non-empty strings")
        if set(self.instance_ids) & set(self.calibration_instance_ids):
            raise ValueError("calibration instances cannot be final evaluation instances")
        if self.temporal_cutoff is not None:
            datetime.date.fromisoformat(self.temporal_cutoff)

    def fingerprint(self) -> Tuple[Any, ...]:
        return (
            self.instance_ids,
            self.labels,
            self.feature_schema_hash,
            self.temporal_cutoff,
            self.allowed_fit_lake_ids,
            self.calibration_instance_ids,
        )


def assert_common_information_budget(budgets: Mapping[str, InformationBudget]) -> None:
    """Raise unless every compared method has the exact same evaluation budget."""
    if len(budgets) < 2:
        raise ValueError("at least two methods are required for a comparison")
    fingerprints = {name: budget.fingerprint() for name, budget in budgets.items()}
    first_name, first = next(iter(fingerprints.items()))
    mismatches = [name for name, fingerprint in fingerprints.items() if fingerprint != first]
    if mismatches:
        raise ValueError(
            f"methods do not share an identical information budget: {first_name} vs {mismatches}"
        )


def assert_common_evaluation_instances(
    method_instances: Mapping[str, Sequence[EvaluationInstance]],
) -> None:
    """Raise unless compared methods receive exactly the same labeled instances."""
    if len(method_instances) < 2:
        raise ValueError("at least two methods are required for a comparison")
    keys = {
        name: tuple(instance.comparison_key() for instance in instances)
        for name, instances in method_instances.items()
    }
    first_name, first = next(iter(keys.items()))
    mismatches = [name for name, key in keys.items() if key != first]
    if mismatches:
        raise ValueError(
            f"methods do not share identical labeled evaluation instances: {first_name} vs {mismatches}"
        )


@dataclass
class CalibrationPolicy:
    """
    Threshold selection and score calibration policy.
    Strictly isolated from final evaluation split (INV-007, INV-019).
    """
    target_max_fpr: float = 0.10
    min_sensitivity: float = 0.50

    def __post_init__(self) -> None:
        if isinstance(self.target_max_fpr, bool) or not 0.0 <= self.target_max_fpr <= 1.0:
            raise ValueError("target_max_fpr must be between 0 and 1")
        if isinstance(self.min_sensitivity, bool) or not 0.0 <= self.min_sensitivity <= 1.0:
            raise ValueError("min_sensitivity must be between 0 and 1")

    def fit_threshold(
        self,
        calibration_scores: np.ndarray,
        calibration_labels: np.ndarray,
        *,
        calibration_instance_ids: Sequence[str] = (),
        final_evaluation_instance_ids: Sequence[str] = (),
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Fit decision threshold achieving FPR <= target_max_fpr while maximizing sensitivity.
        Returns (threshold, metadata). Returns (None, ...) if no threshold satisfies FPR constraint.
        """
        scores = np.asarray(calibration_scores, dtype=float)
        labels = np.asarray(calibration_labels)
        if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels) or len(scores) == 0:
            raise ValueError("calibration scores and labels must be non-empty 1D arrays of equal length")
        if len(calibration_instance_ids) != len(scores) or not calibration_instance_ids:
            raise ValueError("calibration_instance_ids must identify every calibration score")
        if any(not isinstance(instance_id, str) or not instance_id.strip() for instance_id in calibration_instance_ids):
            raise ValueError("calibration_instance_ids must contain non-empty strings")
        if len(set(calibration_instance_ids)) != len(calibration_instance_ids):
            raise ValueError("calibration_instance_ids must be unique")
        if not final_evaluation_instance_ids:
            raise ValueError("final_evaluation_instance_ids are required for isolation")
        if set(calibration_instance_ids) & set(final_evaluation_instance_ids):
            raise ValueError("calibration and final evaluation instances must be disjoint")
        if not np.all(np.isfinite(scores)):
            raise ValueError("calibration scores must be finite")
        if not np.all(np.isin(labels, [0, 1])):
            raise ValueError("calibration labels must contain only 0 and 1")
        clean_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]

        if len(clean_scores) == 0:
            return None, {"status": "NO_CLEAN_CALIBRATION_SAMPLES", "inv_007_compliant": False}
        if len(anomaly_scores) == 0:
            return None, {
                "status": "NO_FEASIBLE_THRESHOLD",
                "reason": "NO_ANOMALY_CALIBRATION_SAMPLES",
                "inv_007_compliant": False,
            }

        # Candidate thresholds are derived from observed calibration scores only.
        candidate_thresholds = np.unique(
            np.concatenate((clean_scores, [np.nextafter(np.max(clean_scores), np.inf)]))
        )

        best_threshold = None
        best_meta = {
            "status": "NO_FEASIBLE_THRESHOLD",
            "observed_fpr": 1.0,
            "observed_tpr": 0.0,
            "inv_007_compliant": False,
        }

        for thresh in candidate_thresholds:
            observed_fpr = float(np.mean(clean_scores >= thresh))
            observed_tpr = float(np.mean(anomaly_scores >= thresh))

            if (
                observed_fpr <= self.target_max_fpr
                and observed_tpr >= self.min_sensitivity
                and (best_threshold is None or observed_tpr > best_meta["observed_tpr"]
                     or (observed_tpr == best_meta["observed_tpr"] and thresh > best_threshold))
            ):
                best_threshold = float(thresh)
                best_meta = {
                    "status": "FEASIBLE_THRESHOLD_FOUND",
                    "threshold": best_threshold,
                    "observed_fpr": observed_fpr,
                    "observed_tpr": observed_tpr,
                    "target_max_fpr": self.target_max_fpr,
                    "min_sensitivity": self.min_sensitivity,
                    "inv_007_compliant": True,
                }

        return best_threshold, best_meta


class TaskBProtocol:
    """
    Retrospective Single-Event Case Study Protocol (South Lhonak).
    Canonical Event Date: 2023-10-04 (INV-009, INV-011, INV-034).
    """
    EVENT_DATE = datetime.date(2023, 10, 4)
    EVENT_LAKE_ID = "LK_SOUTH_LHONAK"

    @classmethod
    def classify_window(
        cls, window_end_date_str: str, window_start_date_str: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Determine window timing relative to South Lhonak event.
        Strictly prevents post-event windows from acting as pre-event positives.
        """
        win_end = datetime.date.fromisoformat(window_end_date_str)
        win_start = (
            datetime.date.fromisoformat(window_start_date_str)
            if window_start_date_str is not None
            else win_end
        )
        if win_start > win_end:
            raise ValueError("window start date must not be after window end date")
        is_pre_event = win_end < cls.EVENT_DATE
        is_post_event = win_start > cls.EVENT_DATE
        is_event_overlap = not is_pre_event and not is_post_event

        # Lead time calculation in days relative to event date
        lead_time_days = (cls.EVENT_DATE - win_end).days

        return {
            "is_pre_event": is_pre_event,
            "is_post_event": is_post_event,
            "is_event_overlap": is_event_overlap,
            "classification": (
                "entirely_pre_event" if is_pre_event
                else "post_event" if is_post_event
                else "event_overlap"
            ),
            "lead_time_days": lead_time_days,
            "canonical_event_date": "2023-10-04",
        }


class TaskAProtocol:
    """Controlled perturbation protocol on authentic background instances."""

    TASK_TYPE = TaskType.TASK_A_PERTURBATION

    @classmethod
    def validate_instances(cls, instances: Sequence[EvaluationInstance]) -> None:
        if not instances:
            raise ValueError("Task A requires at least one evaluation instance")
        for instance in instances:
            if instance.task_type is not cls.TASK_TYPE:
                raise ValueError("Task A received an instance from another task")
            if instance.is_post_event:
                raise ValueError("Task A cannot include post-event instances")


class TaskCProtocol:
    """Negative-control protocol sharing Task B's frozen transform/threshold."""

    TASK_TYPE = TaskType.TASK_C_CONTROL_FALSE_ALERTS
    REQUIRED_CONTROL_LAKES = 4

    @classmethod
    def validate_control_lakes(cls, lake_ids: Sequence[str]) -> None:
        if len(set(lake_ids)) != cls.REQUIRED_CONTROL_LAKES:
            raise ValueError(
                f"Task C requires exactly {cls.REQUIRED_CONTROL_LAKES} independent control lakes"
            )

    @classmethod
    def validate_instances(cls, instances: Sequence[EvaluationInstance]) -> None:
        if not instances:
            raise ValueError("Task C requires at least one control-lake instance")
        for instance in instances:
            if instance.task_type is not cls.TASK_TYPE:
                raise ValueError("Task C received an instance from another task")
            if instance.ground_truth_label != 0:
                raise ValueError("Task C control instances must carry normal label 0")


class ScoreCNormalizer:
    """
    Non-transductive Score-C combiner (INV-023).
    Normalizes Score-A (reconstruction error) and Score-B (embedding distance)
    using training/calibration distribution parameters ONLY.
    """
    def __init__(
        self,
        score_a_min: float,
        score_a_max: float,
        score_b_min: float,
        score_b_max: float,
        alpha: float = 0.5,
        fit_lake_ids: Sequence[str] = (),
        evaluation_lake_ids: Sequence[str] = (),
    ):
        self.score_a_min = score_a_min
        self.score_a_max = score_a_max
        self.score_b_min = score_b_min
        self.score_b_max = score_b_max
        self.alpha = alpha
        self.fit_lake_ids = tuple(fit_lake_ids)
        self.evaluation_lake_ids = tuple(evaluation_lake_ids)
        self.validate_isolation()

    def validate_isolation(self) -> None:
        for name, value in (
            ("score_a_min", self.score_a_min),
            ("score_a_max", self.score_a_max),
            ("score_b_min", self.score_b_min),
            ("score_b_max", self.score_b_max),
            ("alpha", self.alpha),
        ):
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.score_a_max <= self.score_a_min or self.score_b_max <= self.score_b_min:
            raise ValueError("Score-C normalization bounds must be strictly increasing")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not self.fit_lake_ids:
            raise ValueError("Score-C requires explicit training/calibration fit lake IDs")
        if not self.evaluation_lake_ids:
            raise ValueError("Score-C requires explicit final evaluation lake IDs")
        if set(self.fit_lake_ids) & set(self.evaluation_lake_ids):
            raise ValueError("Score-C fit and final evaluation lakes must be disjoint")

    def compute_score_c(self, score_a: np.ndarray, score_b: np.ndarray) -> np.ndarray:
        score_a = np.asarray(score_a, dtype=float)
        score_b = np.asarray(score_b, dtype=float)
        if score_a.shape != score_b.shape or score_a.ndim == 0:
            raise ValueError("Score-A and Score-B arrays must have identical non-scalar shapes")
        if not np.all(np.isfinite(score_a)) or not np.all(np.isfinite(score_b)):
            raise ValueError("Score-A and Score-B arrays must be finite")
        norm_a = np.clip((score_a - self.score_a_min) / (self.score_a_max - self.score_a_min), 0.0, 1.0)
        norm_b = np.clip((score_b - self.score_b_min) / (self.score_b_max - self.score_b_min), 0.0, 1.0)
        return self.alpha * norm_a + (1.0 - self.alpha) * norm_b
