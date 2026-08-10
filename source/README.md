# sentinel-gl Source Package Documentation

Welcome to the `source/` directory of **sentinel-gl** — a self-supervised multi-sensor Earth Observation framework for Glacial Lake Outburst Flood (GLOF) precursor detection in the Hindu Kush Himalaya region.

---

## Directory Architecture

```text
source/
├── config/                 # System & environment configuration settings
├── data/                   # Registry management, channel mapping, & GEE loaders
├── evaluation/             # Synthetic injection protocols, ablation, & figure scripts
├── models/                 # TS-MAE encoder, anomaly scorers (Score-A/B/C), & smoothing
├── scripts/                # Execution runners, build scripts, & verification CLI tools
├── tests/                  # Test suite covering all contracts & invariants
└── utils/                  # Logging, config parsing, & random seed utilities
```

---

## Module Overview

- **`data/`**: Manages the 20-lake study registry (`lake_registry.json`), 15-channel sensor mappings (`channel_mapping.json`), and Earth Engine dataset ingestion loaders.
- **`models/`**: Implements the 1D PyTorch Temporal-Spatial Masked Autoencoder (`ts_mae.py`), feature normalization, latent PCA-kNN density scoring (`score_b.py`), reconstruction MSE (`score_a.py`), combined representation (`score_c.py`), and exponential moving average smoothing (`smoothing.py`).
- **`evaluation/`**: Contains synthetic precursor injection engines (`injector.py`), evaluation protocol metrics (`metrics.py`), zero-retraining ablation study runners (`ablation.py`), and publication figure generation scripts (`figures.py`, `ablation_figures.py`).
- **`scripts/`**: Production runner scripts for data acquisition, TS-MAE training, evaluation execution, threshold analysis, and claim-evidence verification.
- **`tests/`**: Modular unit test suite verifying deterministic data processing, contract invariants, and claim-evidence alignment.

---

## Key Invariants Summary

- **INV-001**: Clean repository state and strict version tagging.
- **INV-002**: Strict role separation between training (15 lakes) and evaluation (5 lakes) to prevent data leakage.
- **INV-003**: Deterministic random seed handling across feature generation and synthetic injection.
- **INV-004**: Multi-sensor channel completeness across 15 physical channels.
- **INV-005**: Standardized 30-day temporal compositing window across time series.
- **INV-006**: Self-supervised training regime (zero event labels used during encoder training).
- **INV-007**: False positive rate upper bound ($\text{FP} \le 10.0\%$) on negative control lakes.
- **INV-008**: Compute budget discipline — zero model retraining during ablation and evaluation.
- **INV-009**: Canonical South Lhonak event anchor date (October 4, 2023).
- **INV-010**: Standardized metric report schema (AUC-ROC, AUC-PR, Detection Rate, FP Rate).
- **INV-011**: Standardized 6-month pre-event window definition.
- **INV-012**: Honest null result reporting discipline.
- **INV-013**: Programmatic claim-to-evidence traceability map (`claim_evidence_map.json`).

---

## Quick-Start Command

To verify complete pipeline integrity and run all tests:

```bash
python3 source/scripts/verify_claim_evidence.py && pytest source/tests/
```
