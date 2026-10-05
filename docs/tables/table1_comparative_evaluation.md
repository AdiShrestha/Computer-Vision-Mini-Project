# Table 1: Comparative Evaluation & Hypothesis Testing

| Section | Metric / Baseline | Value | 95% CI Lower | 95% CI Upper | Unadjusted p | Adjusted p (Holm) | Statistical Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Case Detection | SGL-001 (EV-SGL-001) | 33 days |  |  |  |  | **DETECTED** |
| Alert Burden | lambda_alert (episodes/lake-yr) | 1.0146 | 0.0257 | 5.6529 |  |  | **ESTIMATED** |
| Paired Contrast | Delta S vs area_trend | 0.1551 |  |  | 0.0000 | 0.0000 | **REJECT_NULL_SUPERIOR** |
| Paired Contrast | Delta S vs climatology | 0.0915 |  |  | 0.0000 | 0.0000 | **REJECT_NULL_SUPERIOR** |
| Paired Contrast | Delta S vs rpca | 0.0449 |  |  | 0.0000 | 0.0000 | **REJECT_NULL_SUPERIOR** |
| Paired Contrast | Delta S vs weather_only | 0.0759 |  |  | 0.0001 | 0.0001 | **REJECT_NULL_SUPERIOR** |
