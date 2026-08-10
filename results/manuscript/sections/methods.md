# 3. Methodology

## 3.1 Study Area and Lake Selection

We study a set of 20 glacial lakes in the Hindu Kush Himalaya (HKH) region, selected from ICIMOD's basin-level potentially dangerous lake inventories [2][4] covering the Koshi, Gandaki, and Karnali drainage basins. South Lhonak Lake, Sikkim (28.07°N, 88.12°E) anchors the study as the primary retrospective evaluation target — it experienced a confirmed GLOF on October 4, 2023 (INV-009) following documented multi-year moraine creep exceeding 15 m/year [8][9]. CL-15: event date 2023-10-04.

Role assignments follow a strict leakage boundary (INV-002): CL-16 15 lakes are assigned the `training` role; CL-17 5 lakes form the evaluation set (1 `evaluation_event`: South Lhonak; 4 `evaluation_control`: potentially dangerous lakes with no GLOF in the study period). No evaluation lake ID, observation, or derived statistic is used during encoder training or normalization computation. These roles are encoded in a frozen Lake Registry (`source/data/registry/lake_registry.json`, INV-001) and verified by unit tests that assert no evaluation-role lake ID appears in any training batch.

## 3.2 Multi-Sensor Feature Extraction

We extract seven physically distinct channel groups from open-access satellite time series, producing 15 scalar features per observation date per lake:

| Group | Features | Source | Channels |
|-------|----------|--------|---------|
| CH-01 | Lake area (km²) | Sentinel-1 SAR + Sentinel-2 NDWI | 1 |
| CH-02 | Spectral/turbidity indices (green, red, NIR means, turbidity proxy) | Sentinel-2 | 4 |
| CH-03 | Glacier surface velocity (mean, max in m/yr) | NASA ITS_LIVE | 2 |
| CH-04 | Land surface temperature anomaly (°C deviation) | MODIS LST | 1 |
| CH-05 | SAR backscatter statistics (VV mean, VH mean, VV/VH ratio in dB) | Sentinel-1 | 3 |
| CH-07 | Interferometric SAR coherence | Sentinel-1 | 1 |
| CH-08 | Meteorological context (temp anomaly, precip anomaly, snow anomaly) | ERA5/ERA5-Land | 3 |

Note: CH-06 (InSAR deformation) was excluded following the infeasibility result documented in §4.4 and Decision 001. Features are simulated over the temporal extent 2016-01-01 to 2024-10-31 (INV-003) via Google Earth Engine (GEE), aligned to a common daily time axis with missing observations encoded as explicit NaN markers (never interpolated at this stage).

## 3.3 Self-Supervised Encoder (TS-MAE)

We train a custom Time-Series Masked Autoencoder (TS-MAE) from scratch (Decision 002 — Path B), selected over fine-tuning an existing EO foundation model (Path A) for two reasons: (1) the corpus of 1,605 training windows is well-suited for domain-specific pretraining, and (2) the scalar temporal structure (15 channels × 180-day windows) does not match the spatial-patch paradigm of existing EO backbones such as Prithvi-EO-2.0 [14] or SatMAE [13].

**Input encoding:** The encoder operates on temporal windows of 180 days with a stride of 30 days (INV-004), producing windows of shape [T=180, C=15]. Each channel is z-score normalized using statistics computed exclusively from training-role lake observations (INV-002). Missing values within a window are encoded with a binary availability mask.

**Masked reconstruction objective (INV-005):** During training, 50% of time steps within each window are randomly masked. The encoder receives the unmasked steps and must reconstruct the masked ones via a transformer-style architecture. The loss is MSE between reconstruction and ground truth at masked positions, computed only on non-missing entries. This pretraining prevents label leakage by never exposing any labeled anomaly information to the encoder (INV-002).

**Training details:** Random seeds pinned per INV-012 (encoder seed: 42; masking seed: 42). The encoder trains within the INV-008 compute budget on an M3 MacBook Air with 16GB unified memory, requiring no external GPU allocation.

## 3.4 Anomaly Scoring Mechanisms

Following AP-2 (separation of representation learning from anomaly scoring), we implement three independently-configurable scoring mechanisms evaluated without retraining the encoder:

**Score-A — Reconstruction Error:** At inference, the full window is passed through the encoder without masking. Per-time-step MSE between input and reconstruction is averaged over the window. Reconstruction error is expected to be higher for anomalous windows if the encoder has learned normal dynamics well.

**Score-B — Embedding Distance:** The encoder's mean-pooled latent representation (embedding) for each window is projected to a 16-dimensional PCA subspace fitted exclusively on training-set embeddings (INV-002). Per-window score is the mean Euclidean distance to the 5 nearest training neighbors (k-NN) in this subspace.

**Score-C — Combined Scorer:** A weighted combination Score-C = α × MinMax(Score-A) + (1−α) × MinMax(Score-B), with α = 0.5.

All raw per-window scores are smoothed with an Exponential Moving Average (EMA) of span 5 windows (= 150 days) per INV-006, preserving raw scores for comparison.

## 3.5 Evaluation Protocol (INV-010)

We evaluate across four protocols, producing the full INV-010 metric suite for every experiment:

**Protocol E1 — Retrospective Backtesting:** The trained detector is applied to South Lhonak (SGL-001) time series using only training-set normalization statistics. Lead time is measured as the number of days before October 4, 2023 (INV-009) at which the smoothed anomaly score first exceeds the detection threshold and remains elevated for ≥2 consecutive windows.

**Protocol E2 — Negative Controls:** The same detector is run on all 4 evaluation-control lakes for the same period. False positive rate (FPR) is the fraction of control lake-windows exceeding the detection threshold — reported as a first-class metric per INV-007.

**Protocol E3 — Synthetic Anomaly Injection:** Physically plausible synthetic anomalies (INV-011) are injected into control-lake time series: (1) sudden extent step +20%, (2) gradual extent ramp +15% over 90 days, (3) SAR backscatter step +3 dB (substituted for infeasible InSAR, Decision 003), (4) temperature spike +5°C for 14 days. Detection rate and AUC-ROC/AUC-PR are measured on the injected windows (seed: 2023, INV-012).

**Protocol E4 — Baseline Comparison:** A static threshold on lake extent change rate (the operational standard per [1][3]) is run on identical evaluation lakes and metrics. The threshold is set to minimize FPR while achieving ≥50% detection on synthetic anomalies.

## 3.6 Channel Ablation Study (RQ2, INV-008-Compliant)

To answer RQ2 without retraining the encoder (preserving INV-008 compute invariant), we employ a zero-retraining channel masking approach: for each of 11 configurations, excluded channel columns are zeroed in normalized feature space, and the frozen encoder produces ablated embeddings. Score-B and Score-C are computed from these embeddings using the frozen k-NN density model. Eleven configurations span single-group removals (NO_CHxx) and composite subsets (OPTICAL_ONLY, SAR_ONLY, DYNAMIC_ONLY). Per-channel contribution is defined as the AUC-ROC decrease when that channel group is removed: Δ = AUC_FULL − AUC_NO_CH.

## 3.7 Detection Threshold Refinement

Thresholds are set retrospectively to analyze sensitivity/specificity trade-offs (per architecture.md §5.4). The initial threshold (85th percentile of control scores, CL-08: FPR = 0.1505) exceeds INV-007's 10% target. We sweep 50 threshold percentiles, selecting the minimum threshold where FPR ≤ 0.10 while maximizing synthetic detection rate.
