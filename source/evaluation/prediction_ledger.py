"""
Sentinel-GL — Canonical Row-Level Prediction Ledger and Statistical Input Validator.
Enforces per-instance persistence, immutable run lineage, and lake-level cluster independence.
Governed by INV-010, INV-016, INV-017, INV-025, INV-037.
"""

from typing import Dict, List, Any, Mapping, Optional, Sequence, Tuple
from dataclasses import dataclass, asdict, field
from datetime import date
import json
import hashlib
import re
import numpy as np


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TASK_IDS = {
    "Task_A_Synthetic_Perturbation",
    "Task_B_Retrospective_South_Lhonak",
    "Task_C_Control_False_Alerts",
}


def _require_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_hash(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256 hex string")


def _require_finite(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{field_name} must be finite numeric data")


@dataclass(frozen=True)
class PredictionRow:
    instance_id: str
    run_id: str
    task_id: str
    lake_id: str
    window_id: str
    window_start: str
    window_end: str
    window_center: str
    ground_truth_label: int  # 0 or 1
    method_name: str
    raw_anomaly_score: float
    calibrated_anomaly_score: float
    binary_decision: int  # 0 or 1
    decision_threshold: float
    feature_schema_hash: str
    model_checkpoint_hash: str
    is_pre_event: bool
    is_post_event: bool
    perturbation_type: Optional[str] = None
    perturbation_magnitude: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "instance_id", "run_id", "task_id", "lake_id", "window_id",
            "window_start", "window_end", "window_center", "method_name",
        ):
            _require_text(getattr(self, name), name)
        if self.task_id not in TASK_IDS:
            raise ValueError(f"unsupported task_id: {self.task_id}")
        try:
            start = date.fromisoformat(self.window_start)
            center = date.fromisoformat(self.window_center)
            end = date.fromisoformat(self.window_end)
        except ValueError as exc:
            raise ValueError("window dates must be ISO calendar dates") from exc
        if not start <= center <= end:
            raise ValueError("window dates must satisfy start <= center <= end")
        if type(self.ground_truth_label) is not int or self.ground_truth_label not in (0, 1):
            raise ValueError("ground_truth_label must be integer 0 or 1")
        _require_finite(self.raw_anomaly_score, "raw_anomaly_score")
        _require_finite(self.calibrated_anomaly_score, "calibrated_anomaly_score")
        if type(self.binary_decision) is not int or self.binary_decision not in (0, 1):
            raise ValueError("binary_decision must be integer 0 or 1")
        _require_finite(self.decision_threshold, "decision_threshold")
        expected_decision = int(self.calibrated_anomaly_score >= self.decision_threshold)
        if self.binary_decision != expected_decision:
            raise ValueError("binary_decision must agree with calibrated_anomaly_score and threshold")
        _require_hash(self.feature_schema_hash, "feature_schema_hash")
        _require_hash(self.model_checkpoint_hash, "model_checkpoint_hash")
        if type(self.is_pre_event) is not bool or type(self.is_post_event) is not bool:
            raise ValueError("event flags must be boolean")
        if self.is_pre_event and self.is_post_event:
            raise ValueError("a prediction row cannot be both pre-event and post-event")
        if self.perturbation_type is not None:
            _require_text(self.perturbation_type, "perturbation_type")
        _require_finite(self.perturbation_magnitude, "perturbation_magnitude")

    @property
    def unique_key(self) -> Tuple[str, str, str, str]:
        return (self.run_id, self.task_id, self.method_name, self.instance_id)


@dataclass
class PredictionLedger:
    ledger_version: str = "2.0"
    rows: Tuple[PredictionRow, ...] = field(default_factory=tuple)
    _finalized: bool = field(default=False, init=False, repr=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_finalized", False) and name in {"rows", "ledger_version"}:
            raise RuntimeError("finalized prediction ledger metadata is immutable")
        object.__setattr__(self, name, value)

    def __post_init__(self):
        if self.ledger_version != "2.0":
            raise ValueError("ledger_version must equal '2.0'")
        if self.rows is None:
            self.rows = ()
        self.rows = tuple(self.rows)
        for row in self.rows:
            if not isinstance(row, PredictionRow):
                raise ValueError("rows must contain PredictionRow objects")

    def add_row(self, row: PredictionRow) -> None:
        if self._finalized:
            raise RuntimeError("prediction ledger is finalized and immutable")
        if not isinstance(row, PredictionRow):
            raise TypeError("row must be a PredictionRow")
        if row.unique_key in {existing.unique_key for existing in self.rows}:
            raise ValueError(f"duplicate prediction entry for key {row.unique_key}")
        self.rows = self.rows + (row,)

    def finalize(self) -> str:
        self.validate_or_raise()
        self._finalized = True
        return self.compute_ledger_hash()

    def validate(self) -> List[str]:
        return StatisticalInputValidator.validate_ledger_integrity(self)

    def to_records(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self.rows]

    def compute_ledger_hash(self) -> str:
        serialized = json.dumps(
            {"ledger_version": self.ledger_version, "rows": self.to_records()},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        self.validate_or_raise()
        return json.dumps(
            {"ledger_version": self.ledger_version, "rows": self.to_records()},
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("invalid prediction ledger: " + "; ".join(errors))

    @classmethod
    def from_records(cls, records: Sequence[Mapping[str, Any]], ledger_version: str = "2.0") -> "PredictionLedger":
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ValueError("records must be a sequence of objects")
        rows = []
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("each prediction record must be an object")
            rows.append(PredictionRow(**dict(record)))
        ledger = cls(ledger_version=ledger_version, rows=tuple(rows))
        ledger.validate_or_raise()
        return ledger

    @classmethod
    def from_json(cls, payload: str) -> "PredictionLedger":
        try:
            parsed = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid prediction-ledger JSON") from exc
        if not isinstance(parsed, Mapping) or set(parsed) != {"ledger_version", "rows"}:
            raise ValueError("prediction-ledger JSON must contain exactly ledger_version and rows")
        return cls.from_records(parsed["rows"], ledger_version=parsed["ledger_version"])

    def filter_by_task(self, task_id: str) -> "PredictionLedger":
        return PredictionLedger(ledger_version=self.ledger_version, rows=tuple(r for r in self.rows if r.task_id == task_id))

    def filter_by_method(self, method_name: str) -> "PredictionLedger":
        return PredictionLedger(ledger_version=self.ledger_version, rows=tuple(r for r in self.rows if r.method_name == method_name))

    def get_unique_lakes(self) -> List[str]:
        return sorted(list(set(r.lake_id for r in self.rows)))


class StatisticalInputValidator:
    """
    Validates that statistical procedures consume authentic, persisted prediction ledgers
    and respect independent-unit cluster sampling (INV-016, INV-017, INV-025).
    """

    @staticmethod
    def validate_ledger_integrity(ledger: PredictionLedger) -> List[str]:
        errors: List[str] = []
        if not isinstance(ledger, PredictionLedger):
            return ["Input is not a PredictionLedger"]
        if ledger.ledger_version != "2.0":
            errors.append(f"Unsupported ledger version {ledger.ledger_version!r}")
        if len(ledger.rows) == 0:
            errors.append("Prediction ledger is empty.")
            return errors

        instance_ids = set()
        for idx, row in enumerate(ledger.rows):
            key = row.unique_key
            if key in instance_ids:
                errors.append(f"Duplicate prediction entry for key {key} at row index {idx}")
            instance_ids.add(key)
            try:
                row.__post_init__()
            except (TypeError, ValueError) as exc:
                errors.append(f"Invalid prediction row at row index {idx}: {exc}")

        by_method_task: Dict[Tuple[str, str], List[PredictionRow]] = {}
        for row in ledger.rows:
            by_method_task.setdefault((row.method_name, row.task_id), []).append(row)
        for (method_name, task_id), rows in by_method_task.items():
            lakes = {row.lake_id for row in rows}
            if not lakes:
                errors.append(f"No lake-level independence units for {method_name}/{task_id}")

        return errors

    @staticmethod
    def clustered_units(ledger: PredictionLedger) -> Dict[str, Tuple[PredictionRow, ...]]:
        """Return immutable lake-level clusters for valid ledger rows."""
        ledger.validate_or_raise()
        clusters: Dict[str, List[PredictionRow]] = {}
        for row in ledger.rows:
            clusters.setdefault(row.lake_id, []).append(row)
        return {lake_id: tuple(rows) for lake_id, rows in sorted(clusters.items())}

    @staticmethod
    def extract_statistical_inputs(
        ledger: PredictionLedger,
        source_ledger_hash: str,
        *,
        task_id: str,
        method_name: str,
        score_field: str = "calibrated_anomaly_score",
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
        """Extract metric inputs only after validating their persisted ledger lineage."""
        StatisticalInputValidator.assert_no_manufactured_inputs(
            np.array([], dtype=float), source_ledger_hash, ledger=ledger
        )
        if score_field not in {"raw_anomaly_score", "calibrated_anomaly_score"}:
            raise ValueError("score_field must name a persisted ledger score")
        selected = [
            row for row in ledger.rows
            if row.task_id == task_id and row.method_name == method_name
        ]
        if not selected:
            raise ValueError("no matching persisted prediction rows")
        scores = np.asarray([getattr(row, score_field) for row in selected], dtype=float)
        labels = np.asarray([row.ground_truth_label for row in selected], dtype=int)
        lake_ids = tuple(row.lake_id for row in selected)
        if len(set(lake_ids)) < 2:
            raise ValueError("metric is NOT_ESTIMABLE with fewer than two lake-level units")
        return scores, labels, lake_ids

    @staticmethod
    def assert_no_manufactured_inputs(
        scores: np.ndarray,
        source_ledger_hash: Optional[str],
        *,
        ledger: Optional[PredictionLedger] = None,
    ) -> None:
        """Assert that statistical inputs are anchored to persisted ledger evidence."""
        values = np.asarray(scores, dtype=float)
        if values.ndim > 1 or (values.size and not np.all(np.isfinite(values))):
            raise ValueError("Statistical computation rejected: scores must be finite one-dimensional data")
        if source_ledger_hash is None or SHA256_RE.fullmatch(source_ledger_hash) is None:
            raise ValueError(
                "Statistical computation rejected: missing valid source ledger hash. "
                "Manufacturing pseudo-observation arrays is strictly prohibited by INV-017."
            )
        if ledger is not None:
            ledger.validate_or_raise()
            if source_ledger_hash != ledger.compute_ledger_hash():
                raise ValueError("Statistical computation rejected: source ledger hash does not match ledger")
        else:
            raise ValueError(
                "Statistical computation rejected: a PredictionLedger object is required to verify row lineage"
            )
