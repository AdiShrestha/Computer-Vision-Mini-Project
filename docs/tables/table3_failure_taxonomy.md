# Table 3: Operational Failure Taxonomy Distribution

**Total Evaluations:** 19  
**Total Failures:** 8  
**Report Hash:** `b0648aedfca51d127ac7f54d87b501069e906f5e05719c15b93109f7f95ce865`

| Category Code | Description | Count | Rate (%) | Physical Trigger / Boundary Condition |
| :--- | :--- | :--- | :--- | :--- |
| `ERR_CLOUD_OBSCURATION` | Optical Cloud Obscuration | 0 | 0.0% | >= 80% missing optical observations in preceding 60 days |
| `ERR_SAR_GEOMETRIC_DISTORTION` | SAR Geometric Distortion | 0 | 0.0% | Extreme radar backscatter variance (> 5.0) or steep slope layover/shadow |
| `ERR_SPURIOUS_SEASONAL_ANOMALY` | Spurious Seasonal Transition | 0 | 0.0% | Control false alert during Nov/Dec freeze-up or Apr/May spring breakup (~0°C) |
| `ERR_MISSED_RAPID_TRIGGER` | Missed Rapid Precursor | 0 | 0.0% | Sudden moraine failure with zero precursor signal within revisit window (< 6 days) |
| `ERR_INSUFFICIENT_OBSERVATIONS` | Ineligible Window | 0 | 0.0% | Fails C_obs eligibility (< 2 observations per active modality) |
| `ERR_UNCLASSIFIED` | Unclassified Failure | 8 | 100.0% | Anomalous residual without identified physical mechanism |
