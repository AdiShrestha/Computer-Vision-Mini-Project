# Source

* `sentinel_gl/sr1_corpus.py` — verifies and loads every scene (array, receipt, STAC item or 2023 parent record, native metadata) by SHA-256; refuses non-observational inputs.
* `sentinel_gl/sr1_features.py` — guarded NDWI/MNDWI from patch-mean reflectance, 16 × 16 patch tokens, FIT-only robust scaling, time features.
* `sentinel_gl/sr1_geometry.py` — Hi-MAG 2017 outlines, inventory-centred windows, zone rasters.
* `sentinel_gl/sr1_model.py` — factorised spatial / causal temporal masked transformer and its trainer (per-scene 50% mask with visible context; target-weighted validation).
* `sentinel_gl/sr1_scoring.py` — checkerboard cross-mask, Scores A/B/C, reference predictors, context ablations.
* `sentinel_gl/sr1_baselines_true.py` — seasonal residual, temporal persistence, Isolation Forest, one-class SVM, missingness-only score, Page CUSUM.
* `sentinel_gl/sr1_positive_control.py` — constructed drainage perturbation (origin `constructed_positive_control`).
* `sentinel_gl/sr1_stats.py` — stratified moving-block bootstrap.
* `sentinel_gl/ingestion/`, `admission.py`, `integrity.py` — HTTP range reads, transfer ledger and receipt helpers used by the acquisition runners.

Tests in `tests/` use constructed inputs only and need no data (`make test`).
