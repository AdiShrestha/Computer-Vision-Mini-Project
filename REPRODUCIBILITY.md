# Reproducibility Guide — sentinel-gl

This guide provides the complete, step-by-step protocol for reproducing all experimental results, baseline models, statistical evaluation metrics, channel ablation matrices, decision thresholds, and publication claims for the **sentinel-gl** project.

---

## Step 1: System Prerequisites
- **Operating System**: macOS (Apple Silicon M-series recommended) or Linux.
- **Python**: Version 3.10+ (tested with Python 3.12.8).
- **Compute & Acceleration**: CPU / MPS (Apple Silicon Metal Performance Shaders) / CUDA.
- **Storage**: Minimum ~20 GB free disk space.
- **Google Earth Engine (GEE)**: Registered Earth Engine account for live satellite data verification.

---

## Step 2: Environment Setup
Clone the repository and set up a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Step 3: Google Earth Engine Authentication & Access Verification
Verify authentication credentials and live Earth Engine API access:

```bash
earthengine authenticate
python3 source/scripts/verify_access.py
```
*Expected Output*: 100% PASS across Earth Engine data collections.

---

## Step 4: Lake Registry Verification
Inspect the study lake registry (20 total lakes: 15 training, 5 evaluation):

```bash
python3 -c "import json; r=json.load(open('source/data/registry/lake_registry.json')); print(f'{len(r[\"lakes\"])} lakes loaded successfully')"
```
*Expected Output*: `20 lakes loaded successfully`.

---

## Step 5: Real GEE Feature Matrix & Embedding Verification
Verify that 13-channel real GEE feature matrices (`feature_matrix.npz`) and retrained embeddings are present:

```bash
python3 -c "import os, numpy as np; print(f'Real features verified for {len(os.listdir(\"data/features_real\"))} lakes')"
```

---

## Step 6: Encoder Checkpoint Verification
Verify the retrained TS-MAE model checkpoint (`models/checkpoints/ts_mae_real_data.pt`):

```bash
python3 -c "import torch; ckpt=torch.load('models/checkpoints/ts_mae_real_data.pt', map_location='cpu'); print(f'Checkpoint loaded: {list(ckpt.keys())}')"
```
*Note*: To retrain the TS-MAE encoder on real GEE features, run `python3 source/scripts/train_ts_mae.py`.

---

## Step 7: Run Baseline Models & Unified Evaluation Pipeline
Execute the baseline detectors (Isolation Forest, One-Class SVM, CUSUM, Extent Threshold) and the 7-method evaluation engine:

```bash
python3 source/models/baseline/isolation_forest.py
python3 source/models/baseline/one_class_svm.py
python3 source/models/baseline/cusum_baseline.py
python3 source/scripts/run_evaluation.py
```
*Generated Output*: `results/evaluation/evaluation_summary_real_data.json`.

---

## Step 8: Compute Lake-Level Bootstrap CIs & DeLong Significance Tests (INV-016)
Compute 95% Confidence Intervals via 2,000 lake-level bootstrap resamples (seed 4096) and pairwise DeLong tests:

```bash
python3 source/scripts/run_bootstrap_ci.py
```
*Generated Output*: `results/evaluation/statistical_significance.json`.

---

## Step 9: Run Cloud-Fraction Stratified Evaluation & Protocol E1
Bin evaluation windows across 5 cloud fraction ranges and resolve Protocol E1 against pre-registered F3 falsification criteria:

```bash
python3 source/scripts/cloud_stratified_eval.py
python3 source/evaluation/protocols/protocol_e1.py
```
*Generated Outputs*: `results/evaluation/cloud_stratified_evaluation.json` and `results/evaluation/protocol_e1_real_data.json`.

---

## Step 10: Run Ablation Confound Analysis & Hyperparameter Sensitivity
Execute Option B masking sensitivity analysis across 13 real channels and hyperparameter sensitivity sweeps for $\alpha$ and EMA span:

```bash
python3 source/scripts/run_ablation.py
```
*Generated Outputs*: `results/ablation/ablation_summary_real_data.json` and `results/ablation/hyperparameter_sensitivity.json`.

---

## Step 11: Verification of Claim-Evidence Map & Unit Test Suite
Verify that all 25 quantitative manuscript claims strictly match live result artifacts:

```bash
python3 source/scripts/verify_claim_evidence.py
pytest source/tests/ -v
```
*Expected Output*:
- `verify_claim_evidence.py`: Exits `0` with `25 / 25 claims PASS`.
- `pytest`: 100% PASS across all 242 unit test modules.

---

## Known Limitations
1. **Sample Size & Statistical Power**: Evaluation is conducted across 5 lakes (1 event, 4 control). While lake-level resampling (INV-016) prevents pseudoreplication, statistical power remains limited.
2. **Monsoon Cloud Coverage**: High-cloud bins (>80%) contain zero valid evaluation windows due to persistent Himalayan monsoon cloud cover.
3. **Single Event Validation**: South Lhonak represents the single historical GLOF event in the study region; multi-region transfer requires future expansion.
