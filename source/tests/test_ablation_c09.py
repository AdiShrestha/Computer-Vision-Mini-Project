"""
Unit test suite for Ablation Confound & Hyperparameter Sensitivity (Contract C09-01).
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


def test_ablation_and_hyperparameter_sensitivity_v2():
    ablation, hyperparam = run_ablation_and_sensitivity()
    
    assert ablation['ablation_version'] == 'C09-01_real_data_v2'
    assert 'strategies' in ablation
    assert len(ablation['masking_strategies_evaluated']) == 3

    # Assert real variance across strategies
    auc_zero = ablation['strategies']['zero_masking']['full_13ch_auc_roc']
    auc_mean = ablation['strategies']['mean_imputation_masking']['full_13ch_auc_roc']
    auc_noise = ablation['strategies']['gaussian_noise_masking']['full_13ch_auc_roc']
    assert auc_zero != auc_mean or auc_zero != auc_noise

    # Assert honest alpha sweep justification
    assert hyperparam['score_c_alpha_sweep']['chosen_alpha'] == 0.50
    assert hyperparam['score_c_alpha_sweep']['empirical_optimum_alpha'] == 1.00
    assert 'alpha=1.00' in hyperparam['score_c_alpha_sweep']['alpha_justification']

    abl_path = PROJECT_ROOT / 'results' / 'ablation' / 'ablation_summary_real_data.json'
    hyp_path = PROJECT_ROOT / 'results' / 'ablation' / 'hyperparameter_sensitivity.json'

    assert abl_path.exists()
    assert hyp_path.exists()
