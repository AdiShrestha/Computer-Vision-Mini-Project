# Lake Registry Schema and Tooling

This directory contains the JSON Schema and validation tooling for the canonical Lake Registry (`lake_registry.json`).

## Files

- `schema.json` — Draft-07 JSON Schema defining the required structure for `lake_registry.json`.
- `validate_registry.py` — Deterministic validation script enforcing schema correctness and semantic constraints.
- `test_registry.json` — Test registry used by unit tests.
- `lake_registry.json` — Canonical lake registry populated in C01-05 and frozen under INV-001.

## Schema Fields & Semantics

- **`version`**: Version identifier for the registry format.
- **`metadata`**: Provenance metadata (creation date, factory version, source inventories).
- **`lakes`**: Array of study lake objects:
  - **`id`**: Unique string formatted as `SGL-001` through `SGL-999`.
  - **`name`**: Primary name of the glacial lake.
  - **`coordinates`**: Latitude (25-40 N), Longitude (70-100 E), Elevation in meters (3000-7000m).
  - **`bounding_box`**: Spatial bounding box (`north`, `south`, `east`, `west`) in decimal degrees WGS84.
  - **`role`**:
    - `training`: Unlabeled lake used strictly for self-supervised representation pretraining.
    - `evaluation_event`: Retrospective test lake with a documented historical GLOF (e.g., South Lhonak).
    - `evaluation_control`: Control lake without a documented GLOF used to measure false-positive rate.
  - **`dam_type`**: `moraine`, `ice`, `mixed`, or `unknown`.
  - **`basin`**: River basin name (e.g., Teesta, Koshi, Gandaki).
  - **`susceptibility_class`**: ICIMOD multi-criteria score (`very_high`, `high`, `medium`, `low`, `unassessed`).

## Validation

Run validation via CLI:

```bash
python source/data/registry/validate_registry.py source/data/registry/lake_registry.json
```
