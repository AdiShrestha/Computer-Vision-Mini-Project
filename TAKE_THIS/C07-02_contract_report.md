# Contract Report — C07-02

## Objective
"Download real Sentinel-2 L2A imagery for all 20 lakes, apply rigorous cloud masking, extract NDWI, spectral indices, and lake area. Record actual cloud fraction per scene for downstream cloud-stratified evaluation (Chunk 08). Preserve all data gaps — do NOT fill cloud-masked dates."

## Contract Information
- **Contract ID**: C07-02
- **Chunk ID**: chunk07
- **Objective (quoted verbatim)**: "Download real Sentinel-2 L2A imagery for all 20 lakes, apply rigorous cloud masking, extract NDWI, spectral indices, and lake area. Record actual cloud fraction per scene for downstream cloud-stratified evaluation (Chunk 08). Preserve all data gaps — do NOT fill cloud-masked dates."
- **Risk Tier**: Medium
- **Implementation Owner**: Gemini
- **Model Identifier**: gemini-3.6-flash

## Scope / Inputs / Outputs
- **Inputs**:
  - `source/data/registry/lake_registry.json` (INV-001)
- **Outputs**:
  - `data/raw/sentinel2/{lake_id}/optical_timeseries.csv` (20 files)
  - `data/raw/sentinel2/acquisition_manifest.json`
  - `source/data/acquisition/acquire_sentinel2.py`
  - `source/data/preprocessing/cloud_mask_s2.py`

## Files Modified
| File | Purpose | Reason Modified | Major Changes |
|---|---|---|---|
| `source/data/preprocessing/cloud_mask_s2.py` | Dual cloud masking module | New file | Implemented SCL + s2cloudless dual cloud masking functions |
| `source/data/acquisition/acquire_sentinel2.py` | Sentinel-2 acquisition module | New file | Implemented S2 acquisition engine with real cloud fraction tracking and 0.80 rejection thresholding |
| `source/tests/test_chunk07.py` | Verification test suite | Updated | Added 4 verification tests for C07-02 |

## Verification
- **Command**: `pytest source/tests/test_chunk07.py`
- **Output**: 11 / 11 passed in 0.01s.
- **Full Suite Command**: `pytest`
- **Result**: PASS (209 / 209 passed).

## Human Action Status
```text
Human Action Required: false
Status: Real Sentinel-2 L2A optical time series acquired with cloud fraction tracking and un-filled gaps.
Blocks: NONE. Ready for C07-03.
```

## Evidence
- Acquired optical time series CSV files for all 20 study lakes in `data/raw/sentinel2/`.
- Generated `data/raw/sentinel2/acquisition_manifest.json` logging monsoon vs. dry season gap rates.
- Verified `pytest source/tests/test_chunk07.py`: All 11 DoD unit tests PASSED.

## Definition of Done Verification
1. All 20 lakes have `optical_timeseries.csv` with cloud_fraction column — **Satisfied**.
2. `acquisition_manifest.json` records per-lake gap statistics (monsoon vs. dry season) — **Satisfied**.
3. Monsoon gap rate $\ge 15\%$ for $\ge 60\%$ of lakes — **Satisfied**.
4. No spectral values in rows with cloud_fraction > 0.80 (NaN instead) — **Satisfied**.
5. Lake areas physically plausible (0.01–50 km²) — **Satisfied**.
6. All verification tests pass — **Satisfied** (11/11 PASS).

## Invariant Status
- **INV-001 (Lake Registry)**: Frozen & Unchanged.
- **INV-003 (Temporal Extent)**: 2016-01-01 to 2024-10-31 Enforced.
- **SVI-001 (Real Observations)**: Enforced.

## Final Status
`COMPLETE`

## Plain-Language Summary
Acquired Sentinel-2 L2A optical time series for all 20 study lakes spanning 2016-01-01 to 2024-10-31 with dual cloud masking (SCL + s2cloudless). Preserved authentic monsoon gaps without filling cloud-rejected scenes. Logged acquisition manifest. Verified DoD requirements with 11/11 unit test passes.
