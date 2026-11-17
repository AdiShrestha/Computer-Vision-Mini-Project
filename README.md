# Sentinel-GL

Sentinel-GL retains two distinct retrospective evidence tracks. The earlier 2023 analysis reports source-bound development diagnostics from 48 Sentinel-2 optical observations; the new SR-1 exploratory run uses 141 Sentinel-2 observations spanning 2018–2023 for South Lhonak and Khangchung Tsho. Neither track establishes warning performance, physical lake area, or a SAR-derived result. Khangchung remains a development comparator with unverified non-event history. Negative and inconclusive findings are valid outcomes.

## Canonical workflow

The 2023 development analysis and evidence are indexed in `project/DOMAIN_READINESS_2023.json`. The multi-year SR-1 execution and its open gates are indexed in `project/SR1_TRUE_READINESS.json`. `project/READINESS.json` remains `RESEARCH_NOT_READY`; engineering checks do not close research blockers.

The domain-native Python pipeline and `source/tests/` are the active scientific workflow. Historical generic workflow tooling is retained for its original scope; the current study executes directly through these domain-native entrypoints. No historical scanner finding, commit, or failed attempt has been erased.

For the 2023 track, the corrected geographic epoch is `data/derived/expansion_2023_recentered/`; the earlier raw/derived expansion remains historical evidence. Its protocol is `project/domain_2023_protocol.json`. The corrected Khangchung grid contains the NRSC/ISRO reference location and retains its exact native-aligned affine; it is not a validated lake polygon. Event phases use Sikkim local time (UTC+05:30).

`tools/recenter_2023_epoch.py` stages bounded windows from the 24 retained comparator product selections, reuses the verified South Lhonak arrays, recalibrates the retained SAR sources locally, then seals explicit observational origins and the complete receipt/native-source chain. The new transfer ledger has a 750,000,000-byte cap and exact approved-provider HTTP receipts. Existing sealed manifests and analysis directories must not be overwritten.

The latest retained 499 × 499 run in the 2023 track is `analysis_002`. For a new offline run, use a **new** output directory; this example writes `analysis_003`:

```sh
PYTHONPATH=source /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -B source/runners/run_2023_diagnostics_v2.py --observations data/derived/expansion_2023_recentered/observations.v2.json --output data/derived/expansion_2023_recentered/analysis_003 --expected-ready-sha256 41b3b7ec77658ecb29576bb21e82ebd62feaa43c9daf58d7a10f744169474833 --seeds 0,1,2,3,4
```

In the 2023 track, `analysis_001` and earlier outputs remain preserved; `analysis_002` is the source-bound primary and has an exact score replay with its limitations recorded alongside it. The reflectance-only companion has a separate five-seed output. The newer SR-1 run is stored separately under `data/derived/sr1_multiyear/attempt_006/analysis_001/`. Constructed inputs require the loader's explicit `mode="fixture"`; the empirical runners use observational mode and reject fixture, simulation, unknown and excluded origins/digests.

The 2023 result writer reads `docs/tables/2023_recentered/result_bindings.json`. The current SR-1 manuscript is generated from `docs/tables/sr1_final/result_bindings.json` and figures in `docs/figures/sr1_final_006/`; the result packet preserves unavailable and nonconvergence statuses. Earlier figure and manuscript epochs remain preserved as history. The public snapshot is a partial fixture/code distribution; raw arrays, full receipts and checkpoints need separate retained access.

Use pinned Python 3.12 and `PYTHONPATH=source`. Full source tests run in a disposable copy because some retained tests write generated artifacts. The dependency snapshot is an installed-environment record, not a claim of a fully portable, hash-locked installer.

## Scientific scope

The 499 × 499 grids in the 2023 track are 20 m EPSG:32645 query ROIs spanning 9.98 km per side, not validated lake polygons. The SR-1 track uses its separately declared 160 × 160 EPSG:32645 grid. In both tracks, NDWI/MNDWI masks are candidate spectral masks, not validated shorelines. Neither manuscript claims water-area estimates, event-detection performance, event delay, causal effects, precursor skill, warning reliability, SAR calibration, or SAR fusion. The earlier, defective 160 × 160 epoch is excluded from comparison because its classifier and denominator differed from the declared rule, its scaler/PCA reference included nominal validation scenes, and its 48 filename-derived times differed from manifest times; the corrected multi-year SR-1 execution is separate evidence with its own remaining research blockers.

For the 2023 track, fitting uses parity-0 pixels from the declared pre-breach period. Parity-1 pixels provide spatial development monitoring within the same scenes; they are not independent lakes or a prospective temporal holdout. Post-event observations are excluded from fitting. Per-query contexts use only acquisitions at or before the query time, but fitted model parameters have seen the declared whole pre-breach development period. Seed variation measures optimization variation, not geographic replication. The multi-year SR-1 FIT/VAL/EVAL definitions and fitting scope are documented in its frozen protocol and readiness record.

This is a two-ROI retrospective development analysis, not a research-ready or operational validation. Historical product availability, independent event/control ascertainment, physical lake geometry, annual false-alarm exposure, and operational warning performance are not established. Unavailable evaluation metrics remain null with reasons. The manuscript reports descriptive outcomes and explicit limits; the full blocker list remains open in `project/READINESS.json`.

## Historical artifacts

Earlier fixture-derived data, unsupported metrics, checkpoints and figures remain quarantined. They cannot serve as active observations. Accepted earlier diagnostic packets and export bundles are preserved as dated artifacts; the expanded analysis has separate source manifests and output directories. Verified recomputable full-strip SAR intermediates may be retired only with retained hashes and verified compact ROI bindings.

The root `LICENSE` remains authoritative. This work does not authorize a new license, external publication, Git push, credential disclosure, or historical rewrite.
