"""Eligibility diagnostics preserve unknown physical causes; research joins remain blocked."""
from __future__ import annotations
import dataclasses
from datetime import date
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from .features import MultiModalPanel
from .predictions import PredictionRecord


# ---------------------------------------------------------------------------
# Taxonomy Categories
# ---------------------------------------------------------------------------

ERR_CLOUD_OBSCURATION: str = "ERR_CLOUD_OBSCURATION"
ERR_SAR_GEOMETRIC_DISTORTION: str = "ERR_SAR_GEOMETRIC_DISTORTION"
ERR_SPURIOUS_SEASONAL_ANOMALY: str = "ERR_SPURIOUS_SEASONAL_ANOMALY"
ERR_MISSED_RAPID_TRIGGER: str = "ERR_MISSED_RAPID_TRIGGER"
ERR_INSUFFICIENT_OBSERVATIONS: str = "ERR_INSUFFICIENT_OBSERVATIONS"
ERR_UNCLASSIFIED: str = "ERR_UNCLASSIFIED"

ALL_ERROR_CATEGORIES: Tuple[str, ...] = (
    ERR_CLOUD_OBSCURATION,
    ERR_SAR_GEOMETRIC_DISTORTION,
    ERR_SPURIOUS_SEASONAL_ANOMALY,
    ERR_MISSED_RAPID_TRIGGER,
    ERR_INSUFFICIENT_OBSERVATIONS,
    ERR_UNCLASSIFIED,
)


@dataclasses.dataclass(frozen=True)
class DiagnosticReport:
    """Immutable report summarizing operational diagnostics and failure taxonomy rates."""
    total_evaluations: int
    total_failures: int
    failure_counts: Dict[str, int]
    failure_rates: Dict[str, float]
    cases_by_category: Dict[str, List[str]]
    report_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_evaluations": self.total_evaluations,
            "total_failures": self.total_failures,
            "failure_counts": dict(self.failure_counts),
            "failure_rates": dict(self.failure_rates),
            "cases_by_category": {k: list(v) for k, v in self.cases_by_category.items()},
            "report_hash": self.report_hash,
        }


def classify_failure(context):
    """Missing support alone does not identify a physical failure mechanism."""
    return ERR_INSUFFICIENT_OBSERVATIONS if context.get('eligible') is False else ERR_UNCLASSIFIED


def diagnose_panel_window(record, panel=None, event_onset=None, threshold=None):
    return ERR_INSUFFICIENT_OBSERVATIONS if not record.eligible else ERR_UNCLASSIFIED


def diagnose_prediction_ledger(*args, **kwargs):
    from .admission import blocked
    blocked("research diagnostic evidence joins")
