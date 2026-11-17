# Data

Bulk inputs and outputs are not stored in git (`data/raw/` and `data/derived/` are ignored). They are released as a data deposit built by `tools/build_deposit.py` (see `docs/DATA_AVAILABILITY.md`):

* `data/raw/sr1_multiyear/attempt_006/` and `attempt_007/` — 160 × 160 Sentinel-2 ROI arrays (B03, B08, B11 reflectance, validity, SCL), STAC items, byte-range receipts, native SAFE metadata and transfer ledgers;
* `data/derived/expansion_2023_recentered/` — the retained 2023 parent arrays and receipts that the 2023 rows bind to;
* `data/derived/sr1_multiyear/attempt_007/observations.sr1.json` — the corpus manifest (141 scenes);
* `data/derived/sr1_multiyear/attempt_007/plos_001/` — zones, training run manifest, checkpoints, scores, positive control and replay arrays;
* `data/raw/external/hi_mag_zenodo_4275164/` — the Hi-MAG archive is *not* redistributed; download it from doi:10.5281/zenodo.4275164 (SHA-256 recorded in `source/sentinel_gl/sr1_geometry.py`).

Earlier fixture-era files that used to sit in this folder (claims tables, registries, synthetic panels) were removed from the release tree; they are not evidence and remain only in git history.

Contains modified Copernicus Sentinel data [2018–2023].
