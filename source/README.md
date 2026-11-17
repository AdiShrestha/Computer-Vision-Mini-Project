# Sentinel-GL research code

The canonical 2023 pipeline uses `ingestion/expansion2023.py`, `panel_v2.py`, `training_v2.py`, and `inference_v2.py`, with the acquisition and analysis runners in `source/runners/`. It consumes hash-bound provider ROIs, uses explicit missingness, fits transforms on declared pre-breach parity-0 support, and scores parity-1 development trajectories with causal acquisition contexts. Post-breach observations do not fit model or reference states.

The model uses four quadrant masking folds, optical index dependency closure, explicit time and modality identities, five deterministic training seeds, and development-based checkpoint selection. Sensor-stratified simple baselines and fitted latent/fused scores are reported with real support counts and unavailable-value reasons. Signed BOA reflectance can produce indices outside [-1,1]; those values are neither clamped nor silently replaced.

Use the pinned local Python 3.12 runtime and `PYTHONPATH=source`. Run `source/tests/` in a disposable copy because retained artifact-writing tests modify their local output trees. Tests exercise code contracts; they do not establish independent event labels or operational warning performance.

Earlier versioned diagnostic modules and fixture runners remain available for their recorded historical scopes. Their old outputs are not silently imported into the expanded observations or final comparisons. A source NPZ hash establishes byte identity; independent QA, semantic replay, fitting provenance, and the actual study design determine what those bytes can support.

The study uses two query ROIs, not an independent regional cohort. Same-scene checkerboard monitoring is spatial development evidence. All fitted pre-period states are development-exposed relative to early pre-period queries. SAR is ellipsoid-geolocated without DEM terrain correction, and the retained 2023 dataset has only one SAR date. Missing weather/DEM and unverified historical operational availability remain explicit limits.

The requirements lock records the installed environment; a clean cross-platform installation is not asserted. The root license remains unchanged.
