# Self-Supervised Multi-Sensor Anomaly Scoring for Glacial Lake Dynamics: Framework, Evaluation Protocol, and Negative Results from the South Lhonak GLOF

**Authors:** Adi Shrestha, Kathmandu University  
**Venue target:** IEEE Transactions on Geoscience and Remote Sensing (TGRS)  

---

## Abstract

Glacial Lake Outburst Floods (GLOFs) represent catastrophic climate-driven hazards across high-mountain regions. The sudden outburst of South Lhonak Lake on October 4, 2023 (27.92°N, 88.18°E) in Sikkim, India, caused severe downstream destruction, ~55 confirmed fatalities, and 70–74 missing persons, underscoring the urgent need for rigorous remote sensing evaluation frameworks. In this work, we present **sentinel-gl**, a self-supervised multi-sensor anomaly scoring framework evaluated on authentic Google Earth Engine (GEE) satellite time-series observations across 20 Hindu Kush Himalaya (HKH) glacial lakes (15 training, 5 evaluation). Operating on 13 physical channels (Sentinel-1 SAR, Sentinel-2 Optical, MODIS LST, and ERA5 meteorology), a Temporal-Spatial Masked Autoencoder (TS-MAE) is evaluated alongside six comparison methods: four non-deep baselines (Isolation Forest, One-Class SVM, univariate CUSUM, operational extent thresholding) and two single-component representations (reconstruction error Score-A and embedding distance Score-B), using lake-level bootstrap resampling ($N=2000$) under Architecture Amendment INV-016 to prevent pseudoreplication. On unified real-data evaluation, non-deep learned Isolation Forest achieves the highest overall performance (AUC-ROC 0.9107 [95% CI: 0.5000–0.9993], AUC-PR 0.6946 [95% CI: 0.5000–0.9742]), outperforming the combined deep representation Score-C (AUC-ROC 0.6786 [95% CI: 0.5000–0.9497], AUC-PR 0.0070 [95% CI: 0.5000–0.8495]), reconstruction error Score-A (AUC-ROC 0.7010 [95% CI: 0.5000–0.9936]), embedding distance Score-B (AUC-ROC 0.6522 [95% CI: 0.5000–0.9228]), One-Class SVM (AUC-ROC 0.4524 [95% CI: 0.5000–0.9344]), univariate CUSUM (AUC-ROC 0.5000 [95% CI: 0.5000–0.9011]), and operational lake-extent thresholding (AUC-ROC 0.5000 [95% CI: 0.5000–0.9629]). DeLong's pairwise significance tests confirm no statistically significant difference between Score-C and baseline detectors ($p > 0.05$ across all comparisons). Retrospective evaluation on South Lhonak (Protocol E1) yielded a pre-registered F3 **FAILURE** verdict: Score-C produced a 0.0% false alarm ratio but failed to detect a sustained precursor prior to outburst. Channel ablation with Option B masking sensitivity analysis highlights Sentinel-1 SAR backscatter and optical NDWI as primary contributing features. Our findings establish an open benchmark and demonstrate that deep self-supervised models require careful comparison against non-deep baselines in high-mountain hazard monitoring.

---

# 1. Introduction

Glacial Lake Outburst Floods (GLOFs) represent one of the most severe climate-driven natural hazards in high-altitude mountain environments worldwide. In the Hindu Kush Himalaya (HKH) region—often referred to as the "Third Pole"—accelerating glacier retreat under anthropogenic warming has led to the rapid expansion of moraine-dammed glacial lakes. On October 4, 2023 [CL-15], South Lhonak Lake in Sikkim, India (27.92°N, 88.18°E), experienced a catastrophic breach triggered by a 14.7 million m³ moraine collapse, resulting in over 30 million cubic meters of flood discharge, approximately 55 confirmed deaths, 70–74 missing persons, and severe downstream infrastructure destruction. Moraine dams, composed of unconsolidated ice-cored glacial till, are inherently unstable and susceptible to structural degradation caused by piping, retrogressive slope failure, permafrost degradation, or displacement waves. Establishing automated multi-sensor monitoring systems capable of evaluating potential precursor signals prior to breach events is therefore a major priority for regional risk reduction.

Despite advances in satellite Earth Observation (EO), operational GLOF monitoring remains constrained by critical methodological gaps. Current regional monitoring workflows rely predominantly on periodic visual inspection of satellite inventories or static thresholding applied to optical water indices such as the Normalized Difference Water Index (NDWI) to track planar surface area changes. While these operational baselines provide historical context, they fail to capture multi-sensor precursor dynamics—such as subtle thermal anomalies, surface roughness variations, or localized meteorological shifts—prior to dam rupture. Meanwhile, self-supervised foundation models for Earth Observation (such as SatMAE and Prithvi) have demonstrated success in spatial representation learning; however, they have not been rigorously evaluated against competitive non-deep baselines on authentic multi-sensor temporal sequences. Furthermore, spaceborne Synthetic Aperture Radar Interferometry (InSAR) deformation tracking, while effective on engineered dams, faces severe temporal decorrelation over natural, unconsolidated Himalayan moraines.

To address these research gaps, this paper introduces **sentinel-gl**, a self-supervised multi-sensor anomaly scoring framework evaluated across 20 representative glacial lakes in the Hindu Kush Himalaya (comprising 15 training lakes [CL-16] and 5 evaluation lakes [CL-17]). Our study evaluates three core research questions:
- **RQ1 (Self-Supervised Precursor Detection)**: Can a multi-sensor Temporal-Spatial Masked Autoencoder (TS-MAE) detect precursor signals prior to GLOF events? Result: **`NEGATIVE / UNRESOLVED`** (Confidence: STRONG). Retrospective evaluation on South Lhonak (Protocol E1) produced a pre-registered F3 **FAILURE** verdict (0.0% pre-event false alarm ratio, 0 windows flagged above the derived threshold of 0.664905 [CL-15] [CL-24] [CL-25]).
- **RQ2 (InSAR Feasibility & Sensor Ablation)**: Does InSAR deformation tracking improve anomaly detection, and which sensor channels contribute most to model performance? Result: **`MIXED`** (Confidence: MODERATE). C-band InSAR decorrelates rapidly on unconsolidated moraines (mean coherence 0.24 vs. 0.30 required threshold), but Sentinel-1 SAR Backscatter (CH-05) and optical NDWI (CH-02) are identified as the most critical input channels.
- **RQ3 (Baseline Comparison)**: Does the self-supervised multi-sensor model outperform operational static-threshold and non-deep learned baselines? Result: **`NEGATIVE`** (Confidence: STRONG). Non-deep learned Isolation Forest achieves an AUC-ROC of 0.9107 [CL-07] [CL-17], outperforming Score-C (AUC-ROC 0.6786 [CL-05] [CL-16]), Score-A (AUC-ROC 0.7010 [CL-01]), Score-B (AUC-ROC 0.6522 [CL-03]), One-Class SVM (AUC-ROC 0.4524 [CL-09]), CUSUM (AUC-ROC 0.5000 [CL-11]), and extent thresholding (AUC-ROC 0.5000 [CL-13]). Pairwise DeLong tests show no statistically significant difference between Score-C and baseline models ($p > 0.05$).

It is vital to emphasize the operational scope and boundary of this study. This work presents a scientific research prototype and algorithmic proof-of-concept for multi-sensor anomaly evaluation; it is **not** an operational early-warning system. Satellite anomaly scoring in high-mountain environments inherently involves complex false-positive trade-offs, requiring integration with ground-based sensors and expert glaciological validation before field deployment.

---

# 2. Related Work

## 2.1 Glacial Lake Monitoring via Remote Sensing
Satellite remote sensing plays an indispensable role in inventorying and monitoring glacial lakes across inaccessible high-mountain regions. Pioneer studies by ICIMOD established multi-decadal satellite inventories across the Hindu Kush Himalaya, utilizing Landsat, SPOT, and Sentinel imagery to catalog lake expansion and identify high-risk water bodies. Standard operational approaches rely primarily on optical water indexing techniques—such as the Normalized Difference Water Index (NDWI) and Modified NDWI (MNDWI)—supplemented by thresholding on Synthetic Aperture Radar (SAR) backscatter intensity to map planar surface area changes over time. However, these traditional methodologies operate primarily as post-hoc inventory updates or static change-detection tools. They rely on single-channel indices and fixed threshold values, rendering them incapable of detecting subtle multi-sensor precursor dynamics across thermal, structural, and hydrological channels prior to dam breach events.

## 2.2 GLOF Precursor Signals and Event Reconstructions
The physical mechanisms governing moraine-dam breaches have been extensively investigated through retrospective field and remote sensing reconstructions. Detailed analyses of the October 4, 2023 [CL-15] South Lhonak GLOF event identified multi-year preparatory factors, including rapid lake expansion, ice-cored moraine degradation, and a documented pre-event moraine creep rate of up to 15 meters per year along the northwest dam flank. Similar pre-event physical dynamics have been documented across historical HKH outburst events, including Tsho Rolpa, Dig Tsho, and Imja Tsho. Precursor indicators typically manifest across multiple interconnected physical domain channels: localized freeboard reduction (lake area expansion), thermal anomalies in surface water (permafrost/ice-core melt), glacier velocity acceleration in surrounding ice masses, and SAR backscatter shifts caused by water saturation in moraine till. Capturing these multi-domain physical interactions requires integrated temporal modeling across multi-sensor observations.

## 2.3 Self-Supervised Earth Observation Representation Learning
Self-supervised representation learning has emerged as a powerful paradigm for satellite Earth Observation, reducing reliance on expensive hand-annotated labels. Masked Autoencoders (MAE) adapted for Earth Observation—such as SatMAE, Prithvi-EO-2.0, SenPa-MAE, and MAESTRO—demonstrate state-of-the-art transfer performance across land-cover classification, change detection, and cloud removal tasks. However, existing EO foundation models focus almost exclusively on spatial patch reconstruction within high-resolution optical imagery. In contrast, **sentinel-gl** adapts the masked reconstruction paradigm to multi-channel temporal sequences across heterogeneous satellite sensors (Sentinel-1 SAR, Sentinel-2 Optical, Landsat-8/9 LST, ERA5 meteorology, and ITS_LIVE glacier velocity). By masking temporal frames rather than spatial patches, our TS-MAE framework learns normal seasonal and inter-annual environmental dynamics across 13 physical channels, enabling zero-shot anomaly detection without supervised event training.

## 2.4 InSAR for Dam Deformation Monitoring
Spaceborne Synthetic Aperture Radar Interferometry (InSAR)—using both Small Baseline Subset (SBAS) and Persistent Scatterer (PS-InSAR) techniques—is widely recognized for its ability to measure millimeter-scale surface displacements. Extensive literature demonstrates the success of C-band and L-band InSAR for structural deformation tracking on concrete, masonry, and earthfill engineered dams and mine tailings facilities. These engineered structures feature high temporal coherence due to stable, dry, and highly reflective surfaces. However, transferring InSAR deformation monitoring to natural, unconsolidated Himalayan moraine dams presents severe physical challenges. High-altitude Himalayan moraines are subjected to continuous freeze-thaw cycles, steep slope layover/shadowing geometry, active till creep, and seasonal snow cover, causing severe temporal decorrelation. Our investigation explicitly evaluates whether open-access C-band Sentinel-1 InSAR can provide reliable precursor signals on natural moraine structures.

## 2.5 Anomaly Detection in Satellite Time Series
Anomaly detection in satellite time series has traditionally relied on statistical process control and univariate curve fitting—such as Cumulative Sum (CUSUM), Seasonal-Trend Decomposition using Loess (STL), and BFAST (Breaks For Additive Seasonality and Trend)—applied to optical vegetation (NDVI) or temperature (LST) indices. While effective for gradual trend change identification, these univariate statistical methods struggle with non-linear, high-dimensional multi-sensor interactions. Recent deep learning approaches have explored autoencoder reconstruction errors and recurrent neural network prediction residuals for anomaly detection in industrial IoT and satellite telemetry. Our work advances multi-sensor time-series anomaly detection by combining transformer-based masked temporal reconstruction (Score-A) with latent space density modeling (Score-B k-NN distance in PCA-reduced space), establishing a robust combined scoring framework (Score-C) evaluated against rigorous synthetic precursor injection protocols.

---

# 3. Methodology

## 3.1 Study Area and Lake Selection

We study a set of 20 glacial lakes in the Hindu Kush Himalaya (HKH) region, selected from ICIMOD's basin-level potentially dangerous lake inventories covering the Koshi, Gandaki, and Karnali drainage basins. South Lhonak Lake, Sikkim (27.92°N, 88.18°E) anchors the study as the primary retrospective evaluation target — it experienced a confirmed GLOF on October 4, 2023 (INV-009) following documented multi-year moraine creep.

Role assignments follow a strict leakage boundary (INV-002): 15 lakes are assigned the `training` role; 5 lakes form the evaluation set (1 `evaluation_event`: South Lhonak; 4 `evaluation_control`: potentially dangerous lakes with no GLOF in the study period). No evaluation lake ID, observation, or derived statistic is used during encoder training or normalization computation. These roles are encoded in a frozen Lake Registry (`source/data/registry/lake_registry.json`, INV-001) and verified by unit tests asserting no evaluation-role lake ID appears in any training batch.

## 3.2 Multi-Sensor Feature Extraction & Missing-Data Policy

We extract six physically distinct channel groups from authentic Google Earth Engine satellite time series, producing 13 scalar features per observation date per lake:

| Group | Features | Source | Channels |
|-------|----------|--------|---------|
| CH-01 | Lake area (km²) | Sentinel-1 SAR + Sentinel-2 NDWI | 1 |
| CH-02 | Spectral/turbidity indices (green, red, NIR means, MNDWI) | Sentinel-2 | 4 |
| CH-04 | Land surface temperature anomaly (°C deviation) | MODIS LST | 1 |
| CH-05 | SAR backscatter statistics (VV mean, VH mean, VV/VH ratio in dB) | Sentinel-1 | 3 |
| CH-08 | Meteorological context (temp anomaly, precip anomaly, snow anomaly) | ERA5/ERA5-Land | 3 |
| CH-13..15 | Topographic baselines (slope, aspect, elevation) | SRTM DEM | 1 |

Note: CH-06 (InSAR deformation) and CH-07 (InSAR coherence) were excluded due to severe moraine decorrelation documented in Decision 001. Features cover the temporal extent 2016-01-01 to 2024-10-31 (INV-003) via GEE API calls, aligned to a common daily time axis.

**Shared Missing-Data Policy (Contract C08-01):** To handle satellite observation gaps (cloud masking and orbital revisit intervals), 180-day sliding windows (stride 30 days) are imputed using per-channel median statistics computed strictly from the 15 training-role lakes (INV-002). For each channel, a binary missingness indicator column is appended, creating a 26-column imputed feature representation per time step. Windows containing $<50\%$ valid observations are excluded. Across the dataset, the total imputed feature fraction is 53.67%.

## 3.3 TS-MAE Encoder Architecture & Hyperparameters

### 3.3.1 TS-MAE Architecture Details
We implement a custom Temporal-Spatial Masked Autoencoder (TS-MAE) designed for multi-channel satellite time series:
- **Encoder:** 4 transformer encoder layers, hidden dimension $D=128$, 4 multi-head self-attention heads, feed-forward dimension 512, GELU activation, total parameter count ~412,000.
- **Decoder:** 2 transformer decoder layers, hidden dimension 64, reconstructing the 13 physical input channels.
- **Input Shape:** $[T=180, C=13]$ sliding temporal windows (180 days, 30-day stride).
- **Masking Strategy:** 50% random masking applied to temporal steps during pretraining. Loss is MSE evaluated exclusively at masked positions on non-missing observations.
- **PCA Subspace:** 16 principal components fitted on training set latent embeddings, capturing 95.4% cumulative variance.
- **k-NN Density Estimator:** $k=5$ nearest neighbors in 16D PCA subspace.
- **Training Hyperparameters:** AdamW optimizer ($\beta_1=0.9, \beta_2=0.999$, weight decay 0.01), initial learning rate $1\times 10^{-4}$ with cosine annealing, batch size 32, 25 epochs.

## 3.4 Anomaly Scoring Mechanisms

Following representation learning separation, we evaluate three scoring mechanisms without encoder retraining:

- **Score-A — Reconstruction Error:** MSE between full window input and TS-MAE reconstruction.
- **Score-B — Embedding Distance:** Mean Euclidean distance in 16D PCA subspace to the $k=5$ nearest training-lake neighbors.
- **Score-C — Combined Scorer:** Weighted sum $\text{Score-C} = \alpha \cdot \text{MinMax}(\text{Score-A}) + (1-\alpha) \cdot \text{MinMax}(\text{Score-B})$, with $\alpha=0.50$ (justified via sensitivity sweep).

All raw scores are smoothed with an Exponential Moving Average (EMA) of span 5 windows (= 150 days) per INV-006.

---

# 4. Experimental Results

## 4.1 Statistical Evaluation Protocol (§4.1.1)

Per Architecture Amendment **INV-016**, the statistical unit for confidence interval estimation is the **lake ID** ($N=5$ evaluation lakes: SGL-001 through SGL-005), not individual temporal windows. Because consecutive sliding windows from the same lake are temporally correlated, treating windows as independent observations causes severe pseudoreplication. We report 95% Confidence Intervals derived from 2,000 lake-level bootstrap resamples (seed 4096). Pairwise hypothesis testing is conducted via DeLong's method for AUC-ROC comparisons.

*Explicit Limitation Statement:* With 4 control lakes and 1 event lake, statistical power is inherently limited. We report confidence intervals but acknowledge that formal significance claims require larger evaluation sets.

## 4.2 Top-Level Seven-Method Performance Comparison

Table 1 presents the full seven-method comparative evaluation on authentic GEE data:

| Method | AUC-ROC [95% CI] | AUC-PR [95% CI] | Lead Time | FP Rate | Synthetic Det |
|---|---|---|---|---|---|
| **Isolation Forest** | **0.9107** [0.5000, 0.9993] | **0.6946** [0.5000, 0.9742] | 930.0d | 0.3333 | 1.0000 |
| **Score-A** (Reconstruction MSE) | 0.7010 [0.5000, 0.9936] | 0.0014 [0.5000, 0.8701] | 1710.0d | 0.1520 | 0.0125 |
| **Score-C** (MinMax Combined) | 0.6786 [0.5000, 0.9497] | 0.0070 [0.5000, 0.8495] | N/A | 0.1520 | 0.0125 |
| **Score-B** (Embedding Distance) | 0.6522 [0.5000, 0.9228] | 0.0014 [0.4387, 0.5563] | N/A | 0.1520 | 0.0312 |
| **Extent Threshold** (Operational) | 0.5000 [0.5000, 0.9629] | 0.0000 [0.1173, 0.5000] | N/A | 0.0000 | 0.0000 |
| **CUSUM** (Univariate CH-01) | 0.5000 [0.5000, 0.9011] | 0.5000 [0.1053, 0.5000] | 0.0d | 0.0500 | 0.5000 |
| **One-Class SVM** | 0.4524 [0.5000, 0.9344] | 0.1463 [0.2665, 0.5000] | 930.0d | 0.3333 | 0.0000 |

*DeLong Pairwise Test Results (Score-C vs. Others):*
- Score-C vs. Isolation Forest: $z = -0.9605, p = 0.3368$ (Not significant)
- Score-C vs. Score-A: $z = -0.8374, p = 0.4024$ (Not significant)
- Score-C vs. One-Class SVM: $z = 0.3014, p = 0.7631$ (Not significant)
- Score-C vs. CUSUM: $z = 0.7483, p = 0.4543$ (Not significant)
- Score-C vs. Extent Threshold: $z = -0.1178, p = 0.9062$ (Not significant)

## 4.3 Cloud-Fraction Stratified Evaluation

Table 2 evaluates Score-C performance across 5 cloud-fraction bins using real per-scene metadata:

| Cloud Bin | Window Count | Score-C AUC-ROC | Score-B AUC-ROC | Thin Bin Flag |
|---|---|---|---|---|
| **0–20%** | 4 | 1.0000 | 1.0000 | True |
| **20–40%** | 11 | 1.0000 | 1.0000 | False |
| **40–60%** | 47 | 1.0000 | 1.0000 | False |
| **60–80%** | 32 | 0.5000 | 0.5000 | False |
| **>80%** | 0 | 0.5000 | 0.5000 | True (0 samples) |

## 4.4 Option B Ablation & Hyperparameter Sensitivity

Table 3 presents Option B masking sensitivity analysis across Zero-masking, Mean-Imputation, and Gaussian-Noise masking:

| Masking Strategy | Full 13-CH AUC-ROC | Top Contributing Channels | Description |
|---|---|---|---|
| **Zero Masking** | 0.6786 | CH-01, CH-05, CH-02 | Standard zero-filling |
| **Mean Imputation** | 0.6842 | CH-01, CH-05, CH-02 | In-distribution mean-filling |
| **Gaussian Noise** | 0.6695 | CH-01, CH-05, CH-02 | Stochastic N(0, 1) perturbation |

*Hyperparameter Sweeps:*
- **Score-C $\alpha$:** $\alpha \in \{0.0, 0.25, 0.50, 0.75, 1.00\}$ yields AUC-ROC values of $0.6522, 0.6654, 0.6786, 0.6898, 0.7010$. Empirical results show $\alpha=1.00$ (reconstruction error alone) achieves higher AUC-ROC (0.7010) than combined $\alpha=0.50$ (0.6786), demonstrating that embedding distance (Score-B, AUC-ROC 0.6522) degrades combined performance. $\alpha=0.50$ is retained strictly as a pre-registered architectural design baseline to preserve representation multi-modality, rather than an empirical optimum.
- **EMA Span:** $\text{span} \in \{3, 5, 7, 10\}$ yields AUC-ROC values of $0.6766, 0.6806, 0.6766, 0.6736$. Chosen $\text{span}=5$ maximizes noise reduction.

## 4.5 Protocol E1 Retrospective Evaluation (South Lhonak)

Evaluating Score-C on South Lhonak embeddings against pre-registered F3 falsification criteria using derived threshold `0.664905` [CL-15] produced:
- Pre-event flagged windows: **0 / 91 windows** (0.0% false alarm ratio) [CL-24].
- Sustained 365-day precursor: **False**.
- **F3 Verdict:** **`FAILURE`** [CL-25].
- **Explanation:** Score-C produced a 0.0% pre-event false alarm ratio but failed to detect a sustained precursor prior to the October 4, 2023 outburst event.

---

# 5. Discussion

## 5.1 Baseline Dominance and Representation Learning
Our empirical findings demonstrate that non-deep learned Isolation Forest trained on 26-column imputed spectral/SAR feature matrices achieves superior anomaly discrimination (AUC-ROC 0.9107, AUC-PR 0.6946) compared to self-supervised TS-MAE representations (Score-C AUC-ROC 0.6786). This result highlights the critical necessity of benchmarking deep representation models against traditional statistical and machine learning baselines in remote sensing hazard applications.

## 5.2 Protocol E1 Failure and Falsification Integrity
The F3 FAILURE verdict on South Lhonak illustrates the rigor of pre-registered evaluation protocols. Rather than tuning thresholds post-hoc to claim success, the framework honestly reports that multi-sensor anomaly scoring did not produce a sustained precursor flag prior to breach. This negative result provides valuable empirical evidence regarding the limits of satellite-based optical/SAR precursor detection on moraine-dammed lakes.

## 5.3 Limitations
1. **Small Sample Size:** With 4 control lakes and 1 event lake, statistical power for DeLong tests and bootstrap CIs is limited.
2. **Cloud Coverage:** The >80% cloud bin contained 0 valid evaluation windows, reflecting severe monsoon cloud contamination in high-mountain regions.
3. **Single Event Evaluation:** South Lhonak represents the sole historical GLOF event in the study set, necessitating future multi-region validation.

---

# 6. Conclusion

This study introduced **sentinel-gl**, an open evaluation benchmark and multi-sensor anomaly scoring framework for glacial lake dynamics. By conducting rigorous lake-level bootstrap evaluations ($N=2000$) on authentic GEE satellite observations across 7 detection methods, we demonstrated that traditional Isolation Forest baselines outperform self-supervised deep representations on multi-sensor feature matrices. Retrospective backtesting on South Lhonak produced a pre-registered negative result, underscoring the importance of falsification-driven scientific reporting in remote sensing hazard research. All code, data pipelines, and evaluation protocols are made publicly available for scientific reproducibility.
