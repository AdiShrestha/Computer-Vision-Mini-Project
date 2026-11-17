# Sentinel-GL: Missingness-Aware Sentinel-2 Masked Autoencoding and Multi-Year Surface-Anomaly Diagnostics for South Lhonak and Khangchung Tsho

**Aditya Shrestha**

Department of Computer Science and Engineering (DoCSE), School of Engineering, Kathmandu University, Dhulikhel, Nepal
Corresponding author: Aditya Shrestha; adityashrestha39@gmail.com

## Abstract

Cloud, snow and irregular revisits limit optical monitoring of Himalayan glacial lakes. Sentinel-GL is a 207,685-parameter missingness-aware factorized temporal masked autoencoder. We analyzed 141 Sentinel-2 observations (117 FIT, 17 VAL, seven EVAL) from South Lhonak and Khangchung Tsho, 2018–2023. The prespecified five-channel model’s five best-validation-epoch checkpoints achieved mean validation masked MSE 0.8498 (SD 0.2371) across 15 target-bearing blocks per seed; the 50-fit sensitivity matrix included 350 target-bearing and 80 no-target configuration-block instances. At South Lhonak, Score A medians were 1.399 (9 October), 0.539 (26 October) and 0.528 (13 November); the seasonal-residual baseline was unavailable on EVAL because complete same-patch FIT support was insufficient. Zone 0+1 minus Zone 2 reconstruction-contrast medians were 0.121/0.072 (FIT n=7/5) and 0.189/0.231 (EVAL n=3/4), at South Lhonak/Khangchung. Six of eight matched autumn Score A, B, C and candidate-area medians were lower in EVAL: South Lhonak 0.410→0.539, 1.515→1.219, 0.366→0.363 and 1.144→1.074 km²; Khangchung 0.669→0.567, 1.118→0.937, 0.273→0.197 and 1.736→1.932 km². This comparison uses seven prior FIT scenes per site versus three/four EVAL scenes; FIT scenes are in-sample. Zones lack independent geometry validation, and event timing, false-alert burden and warning value remain unestimated.

**Keywords:** glacial lakes; Sentinel-2; masked autoencoders; anomaly detection; missing data

## 1. Introduction

Glacial lakes are growing across mountain regions, increasing exposure to outburst floods. The October 2023 Sikkim cascade illustrates the consequences of interacting hazards, while optical and radar reconstructions document the South Lhonak event. Satellite time series can describe surface change retrospectively, but a spectral anomaly score is not itself a measurement of drainage or an operational warning.

Sentinel-2 and Sentinel-1 offer complementary optical and radar observations. NDWI-family indices provide useful but imperfect candidate water masks. Prior GLOF studies combine satellite records with physical or operational evidence, including studies of South Lhonak and Khangchung. Under cloud and snow gaps, a masked sequence model is an exploratory complement: spatial attention can combine context across a scene while temporal attention uses each location’s observed history. The present study compares those diagnostics with classical change-detection and anomaly methods. Its purpose is to characterize behavior on this fixed two-site corpus, not to claim warning performance.

## 2. Data and Methods

The corpus contains 141 observations from two 160 × 160 pixel windows at 20 m resolution in EPSG:32645, centered on study reference coordinates 27.915° N, 88.204° E (South Lhonak) and 27.988333° N, 88.815° E (Khangchung Tsho), snapped to the nearest 20 m pixel center. These points are not validated lake outlines. `FIT` comprises 2018–2021 and January–June 2023; `VAL` comprises 2022 and July–September 2023; the uncertain 2–4 October interval is excluded; `EVAL` begins at the 4 October cutoff. The retrospective partition includes pre-monsoon 2023 in `FIT` and holds 2022 and July–September 2023 for validation. Since 2022 `VAL` falls between `FIT` periods, validation measures reconstruction within this corpus rather than forward-time prediction. The 2018–2021 autumn scenes remain in-sample, and 2023 outcomes were viewed during earlier development. Native Sentinel-2 Collection-1 BOA quantification values and SAFE additive-offset metadata were applied during ingestion without additional cross-baseline harmonization.

Each window is partitioned into 100 non-overlapping 16 × 16 patches. B03, B08 and B11 retain native signed reflectance. Clear support uses SCL classes 4, 5 and 6 and at least 25% valid pixels per patch. NDWI and MNDWI are calculated after patch-level averaging; denominators with absolute value below 0.002 are undefined. FIT-derived Zone 0 is a core-water candidate, Zone 1 a buffer candidate, and Zone 2 the remaining candidate background. These zones are not surveyed lake boundaries. Candidate spectral area counts clear pixels with NDWI > 0 and MNDWI > 0 in Zones 0/1 and multiplies by 0.0004 km² per pixel; it is not validated water area or a measure of drainage volume. During ingestion, reflectance was derived from each product’s native BOA quantification value and radiometric additive offset recorded in SAFE metadata, without additional cross-baseline harmonization.

The primary model has a 64-dimensional representation, four attention heads, two spatial encoder layers, one causal temporal encoder layer, one spatial decoder layer and 207,685 trainable parameters. Spatial attention operates across observed patch positions within a scene. Its output is rearranged into one sequence per fixed location, where causal temporal attention uses only observed acquisition times. Patch position, zone, modality and time embeddings are added to the projected channel features. Time is represented by cyclic day-of-year terms and a capped normalized revisit gap, Δt/30 capped at 5, where Δt is the gap since the preceding acquisition. Unsupported patches are masked from attention; a masked target is represented by a learned mask token.

Training examples are chronological, site-specific blocks of at most eight scenes. Half of the observed patches are sampled at random for masking during training; validation masks every other observed patch deterministically. Loss averages only masked channels that are actually observed. A validation block without any such target remains unavailable. We fixed the primary configuration (d = 64, five channels, FIT-derived zones, context length 8) before running the 10-configuration, five-seed sensitivity matrix. Three-channel variants reconstruct B03/B08/B11; five-channel variants also reconstruct NDWI/MNDWI, so their losses average different target spaces and scales and are not directly comparable. Context-1 and context-8 validation windows also have different support: 65 versus 15 target-bearing blocks per fit, with 20 versus zero no-target blocks. Across the 50 fits, deterministic validation partitioning contains 350 target-bearing and 80 no-target configuration-block instances; no-target blocks are excluded from loss aggregation.

For Score A, each masked supported patch receives the mean squared error across its observed target channels; the score averages those patch errors within the specified zone. Score A in Table 4 is for candidate Zones 0+1, and Zone 2 provides the background comparator. Score B projects latent patch representations through a rank-8 PCA fitted only to FIT latent vectors, then averages each eligible candidate patch’s five smallest same-patch distances to distinct FIT acquisitions within a circular ±45-day seasonal window; a FIT query is excluded from its own references. Score C equally combines FIT-robust-standardized log Score A and Score B, using the FIT median and 1.4826 × MAD. The seasonal-residual baseline is the mean across supported patches of each patch’s median Euclidean distance to its five closest complete FIT feature vectors within ±45 day-of-year. Spatial persistence compares each patch with its immediately preceding complete feature vector. Additional comparators are Isolation Forest, one-class SVM, robust PCA and fixed-rule Page CUSUM. Insufficient reference support remains unavailable, and thresholds are not operational alarm points.

| Symbol | Definition |
| --- | --- |
| ρ̄₀₃, ρ̄₀₈, ρ̄₁₁ | Clear-pixel patch means of native BOA reflectance for B03, B08 and B11 |
| xₜ,ₚ,𝚌; x̃ₜ,ₚ,𝚌 | Observed model feature and its FIT-standardized value |
| medianFIT,𝚌; MADFIT,𝚌 | FIT-only channel median and median absolute deviation |
| x̂ₜ,ₚ,𝚌 | Reconstruction of the masked standardized target |
| zₜ,ₚ; PCAFIT | Patch-level latent representation and rank-8 PCA fitted to FIT latent vectors |
| DOYₜ; Δt | Acquisition day of year and revisit gap in days |
| Ωₜ⁽ᵍ⁾; Cₜ,ₚ | Masked supported patches in zone g and observed target channels for patch p |
| Pₜ; N₅(t,p) | Eligible Zone 0+1 patches and their five nearest distinct FIT references |
| α; δ | Fixed Score C weight α = 0.5 and water-index denominator threshold δ = 0.002 |

The water-index denominators are guarded by δ = 0.002:

$$\mathrm{NDWI}=\frac{\bar{\rho}_{03}-\bar{\rho}_{08}}{\bar{\rho}_{03}+\bar{\rho}_{08}},\quad |\bar{\rho}_{03}+\bar{\rho}_{08}|\ge\delta;\qquad \mathrm{MNDWI}=\frac{\bar{\rho}_{03}-\bar{\rho}_{11}}{\bar{\rho}_{03}+\bar{\rho}_{11}},\quad |\bar{\rho}_{03}+\bar{\rho}_{11}|\ge\delta. \tag{1}$$

FIT-only robust channel standardization is:

$$\tilde{x}_{t,p,c}=\frac{x_{t,p,c}-\mathrm{median}_{FIT,c}}{1.4826\,\mathrm{MAD}_{FIT,c}}. \tag{2}$$

For a masked supported patch, first average squared error over its observed target channels; then average these patch errors within a zone:

$$A_t^{(g)}=\frac{1}{|\Omega_t^{(g)}|}\sum_{p\in\Omega_t^{(g)}}\left[\frac{1}{|C_{t,p}|}\sum_{c\in C_{t,p}}(\tilde{x}_{t,p,c}-\hat{x}_{t,p,c})^2\right]. \tag{3}$$

Score B averages the five nearest distinct FIT acquisition distances for each eligible candidate patch after the FIT-only rank-8 PCA projection:

$$B_t=\frac{1}{|P_t|}\sum_{p\in P_t}\left[\frac{1}{5}\sum_{i\in N_5(t,p)}\|\mathrm{PCA}_{FIT}(z_{t,p})-\mathrm{PCA}_{FIT}(z_{i,p})\|_2\right]. \tag{4}$$

Score C uses FIT-only robust standardization of log Score A and Score B:

$$C_t=0.5\,z_{FIT}(\ln A_t^{(0+1)})+0.5\,z_{FIT}(B_t),\qquad z_{FIT}(u)=\frac{u-\mathrm{median}_{FIT}(u)}{1.4826\,\mathrm{MAD}_{FIT}(u)}. \tag{5}$$

## 3. Results and Interpretation

The five best-validation-epoch checkpoints for the prespecified primary configuration had mean validation masked MSE 0.8498 (SD 0.2371) across 15 target-bearing blocks per seed. The 50-fit sensitivity matrix contained 350 target-bearing and 80 no-target configuration-block instances per seed sweep; no-target blocks were excluded from loss aggregation. Three-channel variants reconstruct B03/B08/B11, while five-channel variants also reconstruct NDWI/MNDWI, so their losses cover different target spaces and scales. Context-1 and context-8 validation partitions also differ in support (65 versus 15 target-bearing blocks and 20 versus zero no-target blocks per fit), making direct rankings across contexts inappropriate. These are optimization summaries on one corpus; seed variation does not measure lake-level uncertainty.

The best-validation-epoch checkpoints were evaluated using FIT-derived scalers and reference tokens. Classical comparators were fit once on FIT observations and scored across acquisitions. Unconstrained five-channel robust PCA has a near-machine-precision residual because its fitted subspace spans the input vectors; a separate rank-2 SVD residual is reported as a constrained low-rank sensitivity diagnostic.

For the October–November comparison, seed scores were first collapsed to a median per acquisition; medians, IQRs and counts in Table 4 use distinct scenes as the unit. Six of eight site-by-metric medians were lower in 2023 EVAL than in prior-autumn FIT; two were higher. The earlier autumn FIT observations were used for model fitting, so this is not an independent seasonal-transfer comparison. Only three South Lhonak and four Khangchung EVAL scenes are available.

South Lhonak’s three EVAL scene-level Score A medians across seeds are 1.399 on 9 October, 0.539 on 26 October and 0.528 on 13 November. Score B is 1.411, 1.167 and 1.219; Score C is 1.100, 0.350 and 0.363. Thus, one early reconstruction-score elevation is not sustained across the next two scenes. Score A, Score B and Score C measure different properties. The seasonal-residual baseline is unavailable for these EVAL scenes because five complete same-patch FIT references are not supported.

The Zone 0+1 minus Zone 2 reconstruction-error contrast has scene-level autumn medians of 0.121 at South Lhonak (seven prior FIT scenes) and 0.072 at Khangchung (five), compared with 0.189 and 0.231 in three and four 2023 EVAL scenes. This is descriptive, conditional on spectral zones, and does not establish a lake-specific signal. South Lhonak candidate spectral area has similar prior-autumn and EVAL medians (1.144 and 1.074 km²). This does not show that the mask detected or failed to detect drainage: it is not validated shoreline or volume, and seasonal snow, ice, shadow and classification effects have not been separated. Khangchung is a development comparator, not an independently ascertained event-free control.

Five scientific quantities were considered: event-onset detection timing; false-alert burden per verified event-free lake-year; seasonal transfer to a year excluded from fitting and selection; within-scene candidate-zone reconstruction contrast; and incremental warning value at a common independently calibrated operating point. Only the zone contrast is conditionally descriptive. The other quantities are not estimable because independent event and control histories, operational availability, rolling-origin models or fair calibrated thresholds are absent. The five seeds are not independent lakes, and scenes and patches are dependent.

## 4. Conclusion

Sentinel-GL generated surface diagnostics from this incomplete Sentinel-2 corpus. At South Lhonak, Score A was elevated in the first EVAL scene but fell in the next two; matched autumn summaries did not show uniform increases over in-sample FIT scenes. These results do not establish detection or warning value. Further assessment requires independently validated geometry, event/control records, multiple independent lakes and rolling-origin evaluation at a calibrated operating point.

## Acknowledgement

Contains modified Copernicus Sentinel data [2018–2023], processed by the author. The manuscript distributes derived summaries and figures, not source product files.
