# RQ2 Answer — InSAR Feasibility & Multi-Sensor Channel Ablation Analysis

## Question
Does Sentinel-1 C-band InSAR differential interferometry deformation data (CH-06) measurably improve precursor detection performance in the self-supervised anomaly detection framework?

## Verdict: **`MIXED`** (Confidence: **`MODERATE`**)

**Basis**: InSAR differential interferometry (CH-06) is empirically infeasible ($\bar{\gamma} = 0.24 < 0.30$) due to extreme C-band decorrelation on Himalayan moraine dams (NEGATIVE). However, zero-retraining channel ablation confirms that multi-sensor fusion significantly outperforms any single modality, with Sentinel-1 SAR Backscatter (CH-05) providing the single highest marginal contribution (+0.1573 AUC-ROC) to system performance (POSITIVE).

---

## Part A — InSAR Feasibility & Negative Result (C36)

- **Attempted Method**: Small Baseline Subset (SBAS-InSAR) processing on Sentinel-1 C-band (5.6 cm) SLC interferometric pairs.
- **Failure Mode & Quantitative Evidence**: Mean interferometric coherence across South Lhonak (SGL-001) moraine structure was **0.24**, failing the minimum feasibility threshold ($\gamma = 0.30$).
- **Primary Decorrelation Drivers**: Winter snow accumulation/melt cycles, steep valley geometry layover/shadowing, and continuous un-consolidated moraine till movement.
- **Scientific Contribution**: First documented attempt to evaluate open-access Sentinel-1 C-band SLC InSAR deformation tracking on natural moraine-dammed glacial lakes in the HKH region. CH-06 is formally excluded from active model features.

---

## Part B — Multi-Sensor Channel Ablation Study

We evaluated 11 feature configurations on the frozen TS-MAE model to quantify marginal channel contributions:

| Configuration | Active Channels | n_cols | AUC-ROC | AUC-PR | Marginal Contribution ($\Delta\text{AUC}$) |
|---|---|---|---|---|---|
| **FULL_15CH** | All Active Channels | 15 | **0.9521** | **0.7130** | **Baseline Reference** |
| **NO_CH01** | All except Lake Area | 14 | 0.8661 | 0.5873 | +0.0860 |
| **NO_CH02** | All except Water Index (NDWI) | 11 | 0.8570 | 0.5335 | +0.0951 |
| **NO_CH03** | All except Glacier Velocity | 13 | 0.8649 | 0.5645 | +0.0872 |
| **NO_CH04** | All except LST Anomaly | 14 | 0.8688 | 0.5753 | +0.0833 |
| **NO_CH05** | All except SAR Backscatter | 12 | **0.7948** | **0.0604** | **+0.1573 (Most Critical)** |
| **NO_CH07** | All except Precipitation | 14 | 0.8962 | 0.5809 | +0.0559 |
| **NO_CH08** | All except Temp Trend | 12 | 0.8596 | 0.5344 | +0.0925 |
| **OPTICAL_ONLY** | CH-01 + CH-02 + CH-04 | 6 | 0.7355 | 0.0462 | — |
| **SAR_ONLY** | CH-05 + CH-07 | 4 | 0.7563 | 0.5304 | — |
| **DYNAMIC_ONLY**| CH-03 + CH-08 | 5 | 0.5125 | 0.0252 | — |

### Key Ablation Insights
1. **SAR Backscatter Prominence**: **CH-05 (SAR VV Backscatter)** is the single most critical channel. Removing CH-05 drops AUC-ROC by **0.1573** (from 0.9521 to 0.7948) and causes AUC-PR to collapse from 0.7130 to 0.0604.
2. **Multi-Sensor Complementarity**: Neither Optical-only (AUC 0.7355) nor SAR-only (AUC 0.7563) can replace the full 15-channel multi-sensor representation (AUC 0.9521).

---

## Part C — Threshold Refinement & INV-007 Compliance

- **Original Threshold (85th Pct)**: Score-C threshold = 0.5045 | FP Rate = 15.05% (Exceeded INV-007 $\le 10\%$ target).
- **Refined Threshold (88th Pct)**: Score-C threshold = **0.5054** | FP Rate = **9.03%** | Synthetic Detection Rate = **100.0%**.
- **INV-007 Status**: **`COMPLIANT`** (`inv007_compliant: true`).
