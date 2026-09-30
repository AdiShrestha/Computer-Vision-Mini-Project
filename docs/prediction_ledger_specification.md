# Sentinel-GL Prediction Ledger Specification

## 1. Purpose
The Prediction Ledger is the single authoritative intermediate artifact for all evaluation outcomes. All metrics (AUC, PR, F1, lead time), confidence intervals, and hypothesis tests must consume persisted Prediction Ledger rows.

The ledger is append-only until `finalize()` computes its canonical hash. Rows and the finalized ledger sequence are immutable to callers; duplicate `(run_id, task_id, method_name, instance_id)` keys are rejected. JSON persistence includes the ledger version and every row, and round-trip parsing revalidates the complete record.

## 2. Row Schema
- `instance_id`: Unique identifier of the evaluated window / perturbation instance.
- `run_id`: Immutable identifier for the evaluation run.
- `task_id`: `Task_A_Synthetic_Perturbation`, `Task_B_Retrospective_South_Lhonak`, `Task_C_Control_False_Alerts`.
- `lake_id`: Lake identifier (used as cluster unit for independent bootstrap resampling).
- `window_id`, `window_start`, `window_end`, `window_center`: Temporal metadata.
- `ground_truth_label`: Binary label (0 or 1).
- `method_name`: Method identifier (`TS-MAE_ScoreA`, `TS-MAE_ScoreB`, `TS-MAE_ScoreC`, `Isolation_Forest`, `OneClass_SVM`, `CUSUM_Optical`).
- `raw_anomaly_score`, `calibrated_anomaly_score`: Quantitative outputs.
- `binary_decision`, `decision_threshold`: Calibrated binary classification.
- `feature_schema_hash`, `model_checkpoint_hash`: Immutable cryptographic lineage.
- Both lineage fields are exactly 64 lowercase SHA-256 hexadecimal characters.
- Row dates must be ordered ISO calendar dates; score fields and thresholds must be finite; the binary decision must agree with calibrated score and threshold.

## 3. Clustered Bootstrap Sampling
When computing uncertainty bounds, resampling must sample whole lakes with replacement (cluster bootstrap) rather than resampling overlapping sliding windows as independent instances (INV-016).

`StatisticalInputValidator.clustered_units()` returns immutable lake-level clusters. A metric input extractor requires a matching persisted ledger hash, task ID, method ID, and at least two lake-level units; otherwise the calculation is rejected or explicitly not estimable. A hash-shaped string alone is not evidence that an arbitrary score array came from the ledger when a ledger object is available—the hash is recomputed and compared.

## 4. Metric Semantics and Task Boundaries

Every statistical extraction names its task and method and returns scores, labels, and lake-level units from persistent rows. Duplicate rows, empty ledgers, malformed lineage, inconsistent decisions, unknown tasks, and invalid scores fail validation. Downstream metrics must preserve `NOT_ESTIMABLE` when the declared independence-unit requirements are not met.
