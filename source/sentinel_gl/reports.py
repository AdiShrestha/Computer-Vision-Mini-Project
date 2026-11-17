"""Research report loading and generation are disabled pending evidence admission."""
from __future__ import annotations
import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

from .diagnostics import (
    ALL_ERROR_CATEGORIES,
    DiagnosticReport,
    diagnose_prediction_ledger,
)
from .episodes import (
    AlertEpisode,
    AlertEpisodeEngine,
    LeadTimeResult,
    calculate_lake_exposure_years,
    compute_alert_burden_estimand,
    compute_bounded_lead_time,
)
from .evaluation import (
    BaselineComparisonClaim,
    ClaimsTable,
    generate_claims_table,
)
from .predictions import ABLATION_CONFIGS, PredictionRecord, make_prediction_record
from .visualization import (
    plot_anomaly_trajectories_and_lead_time,
    plot_control_episodes_and_exposure,
    plot_multimodal_feature_panels,
    plot_observation_cadence_and_coverage,
    plot_sensor_ablation_comparison,
)


def compute_file_sha256(path):
    from .integrity import file_hash
    return file_hash(path)


def load_claims_table(*args, **kwargs):
    from .admission import blocked
    blocked("load_claims_table")


def load_diagnostic_report(*args, **kwargs):
    from .admission import blocked
    blocked("load_diagnostic_report")


def load_prediction_ledger_rows(*args, **kwargs):
    from .admission import blocked
    blocked("load_prediction_ledger_rows")


# ---------------------------------------------------------------------------
# Evidence Tables Generator
# ---------------------------------------------------------------------------

def generate_table1_comparative_evaluation(*args, **kwargs):
    from .admission import blocked
    blocked("generate_table1_comparative_evaluation")


def generate_table2_ablation_lattice(*args, **kwargs):
    from .admission import blocked
    blocked("generate_table2_ablation_lattice")


def generate_table3_failure_taxonomy(*args, **kwargs):
    from .admission import blocked
    blocked("generate_table3_failure_taxonomy")


# ---------------------------------------------------------------------------
# Figure Manifest Generator
# ---------------------------------------------------------------------------

def build_figure_manifest(*args, **kwargs):
    from .admission import blocked
    blocked("build_figure_manifest")


# ---------------------------------------------------------------------------
# Full Pipeline Execution
# ---------------------------------------------------------------------------

def run_full_report_generation(*args, **kwargs):
    from .admission import blocked
    blocked("run_full_report_generation")


def main(*args, **kwargs):
    from .admission import blocked
    blocked("main")


if __name__ == "__main__":
    main()
