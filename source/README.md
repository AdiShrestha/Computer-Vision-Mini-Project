# Sentinel-GL numerical core

This is the audited migration baseline, not a complete research system. No observational data, trained models, published results, or performance claims are included. Read [plan.md](../plan.md) before extending it.

The package retains the legacy temporal masked-autoencoder design, with explicit observation masks and corrected loss semantics. It provides training-only normalization, trailing-window dates, fitted PCA/kNN scoring, deterministic masked reconstruction scoring, a frozen Score-C combiner, calibration isolation, and honest undefined metrics. The small training engine records validation-based selection and budget exhaustion.

These operators check their declared inputs. They do not independently authenticate provider data, verify a study cohort, or certify a manuscript. The missing provider-to-prediction pipeline and factory domain adapter are required work in `plan.md`.

Run engineering tests from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest source/tests -p no:cacheprovider
```

`source/requirements.lock` records the installed environment used for these checks. It is an environment snapshot; clean-environment installation and a minimal, hashed ARM64 lock remain release tasks.
