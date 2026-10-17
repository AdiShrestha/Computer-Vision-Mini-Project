# Sentinel-GL

Retrospective multisensor anomaly diagnostics around the October 2023 South Lhonak breach cascade, with Khangchung Tsho as a regional developmental comparator.

The analysis uses 48 Sentinel-2 L2A optical observations and one shared Sentinel-1 IW GRD product: 50 site-sensor observations from 49 distinct provider products. Five training seeds fit a 1,062,919-parameter temporal masked autoencoder. This repository contains the current scientific pipeline, constructed engineering tests, recorded numerical results, figures and submission drafts.

## Findings and limits

South Lhonak's optical reconstruction-error median changes from 0.0477 before the event to 0.0442 afterward. Latent distances rise at both ROIs and were already elevated during September. These results do not establish a breach-specific precursor. Sensor-stratified optical RPCA converged in 413 iterations; the single-date SAR fit converged in one iteration.

The 499 × 499 grids are 20 m EPSG:32645 query ROIs, not validated lake polygons. SAR uses annotation ellipsoid tiepoints without DEM terrain correction. Fitting uses parity-0 pixels from the full declared pre-event period; parity-1 monitoring uses dependent pixels within those same scenes. Per-query context is causal, but earlier pre-event curves are development-exposed through parameter fitting. Seeds quantify optimization variation, not independent lakes. Unavailable warning-performance metrics remain null with explicit reasons. Attention weights are internal distributions, not causal explanations.

## Manuscripts and results

- [Current eight-page manuscript](docs/manuscript/sentinel_gl_submission.pdf) and [LaTeX source](docs/manuscript/sentinel_gl_submission.tex).
- [ICT-CEEL layout](docs/manuscript/sentinel_gl_ictceel2027.pdf), [abstract](docs/manuscript/ICT_CEEL_2027_Abstract.docx), [extended abstract](docs/manuscript/ICT_CEEL_2027_Extended_Abstract.docx) and [full paper](docs/manuscript/ICT_CEEL_2027_Fullpaper.docx).
- [Figures](docs/figures/2023/), [numerical tables](docs/tables/2023/), [recorded analysis outputs](results/2023/) and [source-product inventory](data/observations_inventory.json).
- [Reproduction instructions](docs/reproduction.md), [data preparation](docs/data_preparation.md) and [build instructions](docs/manuscript/BUILD.md).

## Local verification

Use Python 3.12.8. The installed dependency snapshot is in `source/requirements.lock`; it is not a portable wheel-hash lock. Upstream package availability and platform support must be checked when recreating an environment.

```sh
PYTHONPATH=source python3 -B -m pytest -q
python3 tools/verify_public_snapshot.py
```

The public tests use explicitly constructed fixtures. They verify numerical, mask, scope, source-binding and causality contracts; they do not reproduce the empirical experiment without its source pixels. Raw satellite archives, ROI arrays and model checkpoints are not bundled. Their identities are recorded where available, and the acquisition and training code is included. No new acquisition or training was performed to assemble this snapshot.

The root [LICENSE](LICENSE) remains proprietary. Public source visibility grants no additional license. Provider terms apply separately to satellite products. AI assistance and the author's responsibility are disclosed in the manuscript.
