# Table 1: Comparative Evaluation & Hypothesis Testing

| Section | Metric / Baseline | Value | 95% CI Lower | 95% CI Upper | Unadjusted p | Adjusted p (Holm) | Statistical Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Case Detection | SGL-001 (EV-SGL-001) | None |  |  |  |  | **NOT_DETECTED** |
| Alert Burden | lambda_alert (episodes/lake-yr) | 0.0000 | 0.0000 | 43.7676 |  |  | **ESTIMATED** |
| Paired Contrast | Delta S vs area_trend | 1.3465 |  |  | 0.0010 | 0.0039 | **REJECT_NULL_SUPERIOR** |
| Paired Contrast | Delta S vs climatology | -93274.5974 |  |  | 0.2500 | 0.5000 | **FAIL_TO_REJECT** |
| Paired Contrast | Delta S vs rpca | 1.0702 |  |  | 0.0010 | 0.0039 | **REJECT_NULL_SUPERIOR** |
| Paired Contrast | Delta S vs weather_only | 0.0880 |  |  | 0.5967 | 0.5967 | **FAIL_TO_REJECT** |
