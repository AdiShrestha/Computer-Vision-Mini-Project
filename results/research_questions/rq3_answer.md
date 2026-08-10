# RQ3 Answer — Baseline Comparison & Performance Delta

## Question
How does the self-supervised multi-sensor temporal encoder compare against simple operational baseline detectors (e.g. static thresholds on lake extent change rate)?

## Verdict: **`POSITIVE`**

---

## Executive Summary
The learned combined anomaly representation (**Score-C**) significantly outperforms the operational static-threshold extent baseline across synthetic detection metrics:

- **AUC-ROC**: Score-C achieves **0.9521** vs. Baseline **0.6140** (**+0.3381 improvement**).
- **AUC-PR**: Score-C achieves **0.7130** vs. Baseline **0.0918** (**+0.6212 improvement**).
- **Synthetic Detection Rate**: Score-C achieves **100.0%** vs. Baseline **50.0%** (**+50.0% improvement**).

---

## Quantitative Performance Delta Matrix (Traces to `results/evaluation/evaluation_summary.json`)

| Evaluation Metric (INV-010) | Learned Score-C (Combined) | Operational Baseline (Computed) | Performance Delta (Learned vs Baseline) |
|---|---|---|---|
| **Source File** | `evaluation_summary.json` | `evaluation_summary.json` | `evaluation_summary.json` |
| **E3 Area Under ROC (AUC-ROC)** | **0.9521** | **0.6140** | **+0.3381 AUC improvement** |
| **E3 Area Under PR (AUC-PR)** | **0.7130** | **0.0918** | **+0.6212 AUC-PR improvement** |
| **E3 Synthetic Detection Rate** | **100.00%** | **50.00%** | **+50.00% detection rate** |
| **E2 Control False Positive Rate** | 15.05% | 2.31% | +12.74% FP trade-off |
| **E1 Pre-Event Lead Time** | None | None | 0 days |

---

## Key Insights & Trade-Off Analysis

1. **Multi-Channel Sensitivity**: The operational extent baseline only measures surface area changes (CH-01), rendering it blind to thermal spikes (CH-04) or radar backscatter shifts (CH-05). Score-C detects 100% of all 4 synthetic anomaly categories.
2. **False Positive Trade-Off**: The baseline maintains a conservative 2.31% false positive rate on control lakes, whereas Score-C exhibits a 15.05% false positive rate at the 85th percentile threshold.

---

## Overall Confidence Rating
**`STRONG`**: Supported by computed baseline protocol E4 evaluation results across all 20 study lakes.
