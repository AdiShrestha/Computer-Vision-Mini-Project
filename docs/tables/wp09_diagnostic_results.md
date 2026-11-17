# Current diagnostic evidence table

All scores below are quarantined diagnostics. They are not event-detection performance, independent-lake estimates, or calibrated scores. The fit rows are in-sample; monitor parity is a spatial partition of the same four site-sensor group records, backed by three provider products. Score A is dimensionless squared error after the frozen WP06 transform. Baseline score units differ and are not ranked.

| Seed | Role | Weighted masked MSE | Target elements | Scored rows |
|---:|---|---:|---:|---:|
| 0 | fit | 0.0033308984005 | 14 | 4 |
| 0 | monitor | 0.00510621716135 | 14 | 4 |
| 1 | fit | 0.0103114984397 | 14 | 4 |
| 1 | monitor | 0.00820648032407 | 14 | 4 |
| 2 | fit | 0.00881776366649 | 14 | 4 |
| 2 | monitor | 0.0105662554769 | 14 | 4 |
| 3 | fit | 0.00540516119716 | 14 | 4 |
| 3 | monitor | 0.00682832846152 | 14 | 4 |
| 4 | fit | 0.0106666770631 | 14 | 4 |
| 4 | monitor | 0.00840764286529 | 14 | 4 |

Four monitor source-group/quadrant row IDs are shared by TMAE Score A, isolation forest, one-class SVM and RPCA. The exact rows mix two sites and sensor-specific information. RPCA remains `NONCONVERGED_MAX_ITERATIONS` (150 iterations); its monitor scores are projections from that nonconverged fit.

| Quantity | Value | Status |
|---|---:|---|
| Score B (pooled embedding PCA/kNN) | null | `UNAVAILABLE_CONFIGURATION_NOT_SELECTED` |
| Score C (fixed combination) | null | `UNAVAILABLE_NO_PROSPECTIVELY_SELECTED_COMBINATION` |
| AP / PR curve / PR-AUC | null | `UNAVAILABLE_NO_INDEPENDENT_EVENT_LABELS_OR_EVALUATION_UNITS` |
| Calibrated false-alarm-episode tradeoff | null | `UNAVAILABLE_NO_VALID_CALIBRATION_OR_EXPOSURE` |
| Annual false-alarm-episode rate/calibration | null | `UNAVAILABLE_NO_VALID_CALIBRATION_OR_EXPOSURE` |
| Persistence performance / lead time | null | `UNAVAILABLE_NO_VALID_CALIBRATION_OR_EXPOSURE` |
| Retrospective actual checkpoint attention | Figures 6–7 | `QUARANTINED_DIAGNOSTIC_NOT_CAUSAL_OR_FEATURE_IMPORTANCE` |

The missing dependencies are listed in `wp09_metric_availability.csv` and the local SGL-WP09-DIAG-007 review record. No null is encoded as zero.
