# Reproducibility Guide — sentinel-gl

This guide provides the complete, 11-step protocol for reproducing all experimental results, statistical evaluation metrics, channel ablation matrices, decision thresholds, and publication figures for the **sentinel-gl** project.

---

## Step 1: System Prerequisites
- **Operating System**: macOS (Apple Silicon M-series recommended) or Linux.
- **Python**: Version 3.10+ (tested with Python 3.12.8).
- **Compute & Acceleration**: CPU / MPS (Apple Silicon Metal Performance Shaders) / CUDA.
- **Storage**: Minimum ~20 GB free disk space.
- **Google Earth Engine (GEE)**: Registered Earth Engine account (for GEE data verification steps).

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
Verify authentication credentials and Earth Engine API access:

```bash
earthengine authenticate
python3 source/scripts/verify_access.py
```
*Expected Output*: 100% PASS across Earth Engine data collections.

---

## Step 4: Lake Registry Verification
Inspect the study lake registry (20 total lakes: 15 training [CL-16], 5 evaluation [CL-17]):

```bash
python3 -c "import json; r=json.load(open('source/data/registry/lake_registry.json')); print(f'{len(r[\"lakes\"])} lakes loaded successfully')"
```
*Expected Output*: `20 lakes loaded successfully`.

---

## Step 5: Pre-Computed Feature Matrix Verification
Verify that pre-computed 15-channel feature matrices (`feature_matrix.npz`) are present in `data/features/{lake_id}/`:

```bash
python3 -c "import os, numpy as np; print(f'Features verified for {len(os.listdir(\"data/features\"))} lakes')"
```
*Note*: Pre-computed features are included in `data/features/` — raw satellite imagery re-download is not required to reproduce evaluation outputs.

---

## Step 6: Encoder Checkpoint Verification
Verify the trained TS-MAE model checkpoint (`models/checkpoints/ts_mae_best.pt`):

```bash
python3 -c "import torch; ckpt=torch.load('models/checkpoints/ts_mae_best.pt', map_location='cpu'); print(f'Checkpoint loaded successfully: {list(ckpt.keys())}')"
```
*Note*: To retrain the TS-MAE encoder from scratch, run `python3 source/scripts/train_ts_mae.py`.

---

## Step 7: Run Full Evaluation Pipeline (Chunk 04 Results)
Execute the evaluation pipeline across all 20 lakes for Score-A, Score-B, Score-C, and Operational Baseline:

```bash
python3 source/scripts/run_evaluation.py --checkpoint models/checkpoints/ts_mae_best.pt
```
*Generated Output*: `results/evaluation/evaluation_summary.json` (tagged `rework_version: C04-R1`).

---

## Step 8: Run Zero-Retraining Channel Ablation Study (Chunk 05 Results)
Execute the zero-retraining channel ablation study across 11 feature configurations:

```bash
python3 source/scripts/run_ablation.py --checkpoint models/checkpoints/ts_mae_best.pt
```
*Generated Output*: `results/ablation/ablation_summary.json` (tagged `ablation_version: C05-02`).

---

## Step 9: Run Threshold Percentile Analysis (INV-007 Compliance)
Sweep decision threshold percentiles across control lakes to satisfy INV-007 (FP rate $\le 10\%$):

```bash
python3 source/scripts/run_threshold_analysis.py
```
*Generated Output*: `results/ablation/threshold_analysis.json` (confirms `refined_fp_rate = 9.03%` [CL-09] and `inv007_compliant: true` [CL-10]).

---

## Step 10: Re-Generate Publication Figures & Comparison Tables
Generate all publication figures across Chunk 04 and Chunk 05 results:

```bash
python3 source/evaluation/figures.py
python3 source/evaluation/ablation_figures.py
```
*Generated Output*: PNG figures in `results/figures/`.

---

## Step 11: Verification of Claim-Evidence Map & Unit Test Suite
Verify that all 25 quantitative manuscript claims strictly match live result artifacts, and execute the full test suite:

```bash
python3 source/scripts/verify_claim_evidence.py
pytest source/tests/ -v
```
*Expected Output*:
- `verify_claim_evidence.py`: Exits `0` with `25 / 25 claims self-verified PASS`.
- `pytest`: 100% PASS across all unit test modules.
