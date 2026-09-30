"""
Sentinel-GL Evaluation Protocols Package.
"""

from .task_protocols import (
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

__all__ = [
    "TaskType",
    "EvaluationInstance",
    "InformationBudget",
    "assert_common_information_budget",
    "assert_common_evaluation_instances",
    "CalibrationPolicy",
    "TaskAProtocol",
    "TaskBProtocol",
    "TaskCProtocol",
    "ScoreCNormalizer",
]
