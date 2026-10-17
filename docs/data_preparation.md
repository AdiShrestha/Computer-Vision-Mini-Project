# Source data and acquisition

The study's two fixed query centroids are South Lhonak (27.915° N, 88.204° E) and Khangchung Tsho (27.985° N, 88.752° E). Their exact accepted projected affine grids are in `configs/roi_grids.json`. These are development ROIs, with unresolved shoreline accuracy and cross-sensor terrain registration.

The observation inventory records all 50 site-sensor products and the SHA-256 of each original ROI array. It is an inventory, not a substitute for pixel bytes or a fully executable observation manifest. Native product identifiers permit independent provider searches. Original HTTP range receipts remain necessary for byte-level acquisition replay; they are not all distributed here. Historical operational availability is not established by acquisition timestamps.

The optical adapter queries Earth Search's `sentinel-2-c1-l2a` catalogue for January 1 through November 15, 2023. It selects up to 24 observations per ROI using temporal bins and catalogue cloud cover, without anomaly-score selection. It reads bounded source COG ranges for B03/B08/B11 and SCL, verifies native metadata and BOA scale/offset, accepts the explicitly recorded processing baselines 05.09/05.10/05.11, and preserves SCL 4/5/6 support. It hashes received ranges and resulting arrays.

```sh
PYTHONPATH=source python3 -B source/runners/run_2023_acquisition.py --root "$PWD" --modality optical
```

This command performs real transfers and consumes disk and network resources. Its configured payload budget is 8,262,017,611 bytes for this expansion; no automatic run occurs during tests. Current provider responses may differ from retained records.

SAR inputs require the actual dual-polarization IW GRD SAFE product, native measurement TIFFs and annotation/calibration/noise XMLs. The recorded product is `S1A_IW_GRDH_1SDV_20230901T121408_20230901T121433_050134_060898_363F.SAFE`, archive SHA-256 `85b0adb8c7057e98fe6e4b40a70777bad2618b5f757c6cec5f713b1920f7f71b` (1,737,975,879 bytes). Calibration, signed noise subtraction and strict positive-support dB conversion are implemented in `source/sentinel_gl/ingestion/sentinel1_calibration.py`.

`reuse_retained_sar` accepts a locally supplied, ignored `configs/sar_inputs.local.json` object with `catalogue_item` (relative JSON path), `paths_by_pol` (VV/VH mappings for measurement, annotation, calibration and noise), and `archive` (relative path, SHA-256 and byte count). The archive identity is checked before calibration. A fresh online SAR run also requires the CDSE catalogue response at `data/raw/expansion_2023/sar_catalogue/response.json`; discovery and credential setup are not silently fabricated. `CDSE_ACCESS_TOKEN` is read from the local environment. Do not commit it. The recorded attempt to acquire additional SAR dates ended with HTTP 401, leaving only one authentic SAR date in this study.

For each staged observation, the loader requires a relative values file, its exact digest, a receipt binding product/site/time/digest, sensor channel order, grid and validity arrays. Missing observations must remain absent; test fixtures cannot populate the empirical manifest. Native arrays and receipts can be produced by `optical_observation` and `calibrate_sar_inputs`. Geometry uses ellipsoid annotation tiepoints and is not DEM terrain-corrected.

Satellite product terms and required attribution remain the provider's terms. This source distribution grants no additional rights over upstream data or the project code. No ERA5, DEM or validated lake-outline measurements are included in the reported experiment.
