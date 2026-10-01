# Sentinel-GL numerical core

This is the evolving audited numerical core, not a complete research system. No observational data, trained models, published results, or performance claims are included. Read [plan.md](../plan.md) before extending it.

The package retains the legacy temporal masked-autoencoder design, with explicit observation masks and corrected loss semantics. It provides training-only normalization, trailing-window dates, fitted PCA/kNN scoring, deterministic masked reconstruction scoring, a frozen Score-C combiner, calibration isolation, and honest undefined metrics. The small training engine records validation-based selection and budget exhaustion.

The follow-up pass adds finite JSON exponent checks, numerically stable normalization, read-only fitted settings/state hashes, observed-context-preserving masks and separate best-checkpoint/patience selection. Sustained alarms now require explicit eligibility and a maximum decision gap. Version-2 checkpoints at the first follow-up identify mask policies; prospective format-3 checkpoints also use resolved `max_time_steps` configuration. Neither format is a resumable training snapshot. See [follow-up evidence](../docs/audit/followup/README.md) for the historical 41 passing engineering tests and the remaining provider/lineage/methodology limits.

These operators check their declared inputs. They do not independently authenticate provider data, verify a study cohort, or certify a manuscript. The missing provider-to-prediction pipeline and factory domain adapter are required work in `plan.md`.

Run engineering tests from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest source/tests -p no:cacheprovider
```

`source/requirements.lock` records the installed environment used for these checks. It is an environment snapshot; clean-environment installation and a minimal, hashed ARM64 lock remain release tasks.

The [Claude review check](../docs/research/claude_review_check.md) added shared ordered-ID validation, NumPy-only input contracts and a retrospective as-of scheduler. The scheduler marks simulations explicitly and does not prove actual execution or source availability. `TimeSeriesMAE.max_time_steps` is positional capacity; `n_windows` remains a deprecated constructor alias. A 180-row context needs capacity at least 180; defaults are unselected engineering settings. Current counts and logs are in [readiness](../project/READINESS.json).
