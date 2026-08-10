# Section 6: Conclusion

In this manuscript, we presented **sentinel-gl**, a self-supervised multi-sensor anomaly detection framework for Glacial Lake Outburst Flood (GLOF) precursor monitoring across the Hindu Kush Himalaya (HKH) region. Utilizing a Temporal-Spatial Masked Autoencoder (TS-MAE) trained on 15 physical channels across 15 training lakes [CL-16] and evaluated across 5 test lakes [CL-17], our work established rigorous empirical evidence addressing three core research questions:
- **RQ1 (Self-Supervised Precursor Detection)**: **`MIXED`** verdict (Confidence: MODERATE). Latent embedding distance scoring (Score-B, AUC-ROC 0.8973 [CL-04]) and combined scoring (Score-C, AUC-ROC 0.9521 [CL-01], AUC-PR 0.7130 [CL-02]) achieve strong synthetic precursor discrimination. However, reconstruction error (Score-A) is a null result (AUC-ROC 0.4552 [CL-03]), proving that reconstruction MSE on GEE-simulated features lacks lake-specific anomaly signal.
- **RQ2 (InSAR Feasibility & Sensor Ablation)**: **`MIXED`** verdict (Confidence: MODERATE). Open-access C-band Sentinel-1 InSAR is empirically infeasible over Himalayan moraine dams due to severe decorrelation (mean coherence 0.24 [CL-13] vs. 0.30 required threshold [CL-14]). However, zero-retraining sensor ablation confirms that Sentinel-1 SAR Backscatter (CH-05) is the single most critical channel group (+0.1573 AUC-ROC contribution [CL-11]), and no single modality matches the full multi-sensor system performance.
- **RQ3 (Baseline Comparison)**: **`POSITIVE`** verdict (Confidence: STRONG). The learned combined model significantly outperforms the operational static-threshold lake extent baseline by +0.3381 AUC-ROC [CL-06] (0.9521 [CL-01] vs. 0.6140 [CL-05]) and +0.6212 AUC-PR [CL-07] (0.7130 [CL-02] vs. 0.0918).

Three fundamental scientific take-aways emerge from our experimental evaluations:
1. **SAR Backscatter Prominence**: Sentinel-1 SAR VV Backscatter (CH-05) is the single most important physical channel (+0.1573 AUC contribution [CL-11]). Removing CH-05 causes Score-C AUC-ROC to drop to 0.7948 [CL-12] and collapses AUC-PR from 0.7130 to 0.0604.
2. **Multi-Sensor Synergy**: Single-modality feature subsets—such as Optical-only (AUC 0.7355 [CL-23]), SAR-only (AUC 0.7563 [CL-24]), and Dynamic-only (AUC 0.5125 [CL-25])—underperform the full multi-sensor system (AUC 0.9521 [CL-01]), proving the necessity of multi-sensor fusion.
3. **Threshold Refinement & Compliance**: Sweeping the decision threshold from the 85th percentile (0.5045, 15.05% FP [CL-08]) to the 88th percentile (0.5054) achieves full INV-007 compliance with a 9.03% false positive rate [CL-09] [CL-10] while maintaining a 100.0% synthetic detection rate [CL-18].

We candidly document several methodological limitations of our current prototype:
- **Small-N Retrospective Constraint**: Retrospective event evaluation is limited to a single confirmed breach event (South Lhonak, October 4, 2023 [CL-15]), reflecting the inherent small-N constraint of rare disaster events in Earth Observation.
- **GEE Feature Simulation**: Feature matrices were generated via Earth Engine feature simulation rather than raw pixel-level preprocessing pipelines, which may explain the lack of discriminative variation in Score-A reconstruction MSE.
- **InSAR Wavelength Limit**: InSAR evaluation was limited to C-band (5.6 cm) Sentinel-1 SLC data; longer L-band wavelengths (e.g., NISAR, ALOS-2) may achieve higher coherence on vegetated and till-covered moraines.
- **Static Thresholding**: Fixed percentile thresholding across all lakes produces trade-offs between false alarm rates and lead time sensitivity.

Future research directions will focus on:
1. **L-band InSAR Evaluation**: Assessing NISAR L-band SAR interferometry to test whether longer wavelengths overcome decorrelation on Himalayan moraines.
2. **Near-Real-Time Data Pipeline**: Materializing automated GEE ingestion and feature extraction pipelines (architecture.md FE-1).
3. **Geographic Generalization**: Expanding the lake registry to include glacial lakes across the Andes, Alaska, and Patagonia (architecture.md FE-2).
4. **Adaptive Spatiotemporal Thresholding**: Replacing static percentile thresholds with adaptive, lake-specific baseline models.
