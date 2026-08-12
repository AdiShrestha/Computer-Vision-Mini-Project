"""
Unit test suite for Ablation Fix & Hyperparameter Sensitivity (Contract C09-01).
"""

import os
import json
import pytest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'source'))

from scripts.run_ablation import run_ablation_and_sensitivity


def test_ablation_and_hyperparameter_sensitivity():
    ablation, hyperparam = run_ablation_and_sensitivity()
    
    assert ablation['ablation_version'] == 'C09-01_real_data'
    assert 'strategies' in ablation
    assert len(ablation['masking_strategies_evaluated']) == 3

    assert hyperparam['hyperparameter_version'] == 'C09-01_sensitivity_sweeps'
    assert hyperparam['score_c_alpha_sweep']['chosen_alpha'] == 0.50
    assert hyperparam['ema_span_sweep']['chosen_span'] == 5

    abl_path = PROJECT_ROOT / 'results' / 'ablation' / 'ablation_summary_real_data.json'
    hyp_path = PROJECT_ROOT / 'results' / 'ablation' / 'hyperparameter_sensitivity.json'

    assert abl_path.exists()
    assert hyp_path.exists()
