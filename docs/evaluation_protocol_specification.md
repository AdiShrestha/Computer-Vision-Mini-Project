# Sentinel-GL Evaluation and Calibration Protocol Specification

## 1. Experimental Tasks
- **Task A (Synthetic Perturbation Sensitivity)**: Evaluates anomaly score response to controlled physical perturbations injected on authentic control-lake background instances.
- **Task B (Retrospective Single-Event Case Study)**: Evaluates anomaly detection lead time on South Lhonak Lake anchored strictly on the canonical 2023-10-04 event date.
- **Task C (Control-Lake False Alert Burden)**: Evaluates the false alert rate on 4 independent control lakes under a pre-registered FPR threshold $\le 0.10$.

All methods compared within a task must receive the exact same ordered `EvaluationInstance` IDs, labels, dates, and perturbation metadata. Their `InformationBudget` must also match on instance IDs, labels, feature-schema hash, temporal cutoff, permitted fit lakes, and calibration IDs.

Task A is synthetic only in its explicit perturbation layer; its backgrounds remain authentic observations. Task B permits only windows entirely before the event for precursor conclusions. The event day and any window crossing it are `event_overlap`, not pre-event. Task C accepts only normal-label control instances and requires exactly four distinct control lakes.

## 2. Calibration Isolation
Decision thresholds are calibrated using calibration-role data only. Candidate thresholds come from observed calibration scores plus an above-maximum sentinel; selection requires both observed FPR $\le 0.10$ and the pre-registered minimum sensitivity. If no candidate satisfies both conditions, the policy returns `NO_FEASIBLE_THRESHOLD` with a non-compliant verdict.

## 3. Score-C Non-Transductive Normalization
Score-C combines normalized Score-A and Score-B using strictly increasing min/max bounds fitted on declared training/calibration lakes. Fit and final evaluation lake IDs are required and must be disjoint; final score arrays must have identical finite shapes. No final-test trajectory is used to derive the bounds or alpha.

## 4. Instance and Budget Validation

`EvaluationInstance` validates ISO calendar ordering, binary labels, mutually exclusive pre/post flags, and finite perturbation magnitudes. `assert_common_evaluation_instances` and `assert_common_information_budget` fail closed on cross-method mismatch so a comparison cannot silently mix windows, labels, feature versions, or fit scopes.
