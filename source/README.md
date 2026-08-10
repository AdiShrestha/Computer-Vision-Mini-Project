# sentinel-gl Source Directory Structure

This directory contains the complete implementation of the `sentinel-gl` system (Self-Supervised, Cloud-Robust Precursor Detection for Glacial Lake Outburst Floods), structured per `architecture.md` §11.

## Directory Layout

- `config/` — Configuration management and experiment configuration files.
  - `experiment_configs/` — Named experiment configuration files.
- `data/` — Data pipeline components.
  - `registry/` — Lake registry (`lake_registry.json`) defining study lakes and role assignments.
  - `acquisition/` — Data acquisition modules for satellite and meteorological data.
  - `preprocessing/` — Cloud masking, calibration, spatial/temporal alignment pipelines.
  - `channels/` — Feature extraction for individual data channels (extent, velocity, temperature, etc.).
  - `insar/` — InSAR interferometry processing and deformation feature extraction.
- `models/` — Machine learning model implementations.
  - `encoder/` — Masked autoencoder architecture and training loops.
  - `anomaly/` — Anomaly scoring mechanisms (reconstruction error, embedding distance).
  - `baseline/` — Baseline thresholding and comparison models.
- `evaluation/` — Benchmarking and evaluation protocols.
  - `protocols/` — Retrospective backtesting and control evaluation protocols (E1–E5).
  - `synthetic/` — Synthetic anomaly injection pipeline.
  - `visualization/` — Plotting and publication figure generation routines.
- `utils/` — Common utilities (logging, random seed management per INV-012, hashing).
- `tests/` — Test suite (discovered via `pytest`).
- `scripts/` — Executable entry points for training, evaluation, and pipeline execution.

## Testing & Environment

Run tests from the repository root:

```bash
pytest
```
