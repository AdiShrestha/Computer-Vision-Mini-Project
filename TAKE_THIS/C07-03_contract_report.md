# Contract Report — C07-03

## Objective
"Acquire auxiliary satellite and reanalysis data (ITS_LIVE glacier velocity, MODIS land surface temperature, ERA5 meteorology) for all 20 lakes. **Drop CH-07 entirely** — GRD-derived "coherence" is scientifically invalid."

## Contract Information
- **Contract ID**: C07-03
- **Chunk ID**: chunk07
- **Objective (quoted verbatim)**: "Acquire auxiliary satellite and reanalysis data (ITS_LIVE glacier velocity, MODIS land surface temperature, ERA5 meteorology) for all 20 lakes. **Drop CH-07 entirely** — GRD-derived "coherence" is scientifically invalid."
- **Risk Tier**: Medium
- **Implementation Owner**: Gemini
- **Model Identifier**: gemini-3.6-flash

## Scope / Inputs / Outputs
- **Inputs**:
  - `source/data/registry/lake_registry.json` (INV-001)
- **Outputs**:
  - `data/raw/itslive/{lake_id}/velocity_timeseries.csv` (20 files)
  - `data/raw/modis/{lake_id}/lst_timeseries.csv` (20 files)
  - `data/raw/era5/{lake_id}/meteorology_timeseries.csv` (20 files)
  - `data/raw/auxiliary_acquisition_manifest.json`
  - `source/data/acquisition/acquire_itslive.py`
  - `source/data/acquisition/acquire_modis.py`
  - `source/data/acquisition/acquire_era5.py`

## Files Modified
| File | Purpose | Reason Modified | Major Changes |
|---|---|---|---|
| `source/data/acquisition/acquire_itslive.py` | ITS_LIVE velocity module | New file | Implemented annual terminus glacier velocity extraction ($\ge 5$ annual rows per lake) |
| `source/data/acquisition/acquire_modis.py` | MODIS LST module | New file | Implemented daily LST acquisition and training-set-only climatology (INV-002) |
| `source/data/acquisition/acquire_era5.py` | ERA5 meteorology module | New file | Implemented ERA5 daily 2m temperature, precipitation, and snow depth acquisition ($>95\%$ coverage) |
| `source/tests/test_chunk07.py` | Verification test suite | Updated | Added 4 verification tests for C07-03 |

## Verification
- **Command**: `pytest source/tests/test_chunk07.py`
- **Output**: 15 / 15 passed in 0.07s.
- **Full Suite Command**: `pytest`
- **Result**: PASS (213 / 213 passed).

## Human Action Status
```text
Human Action Required: false
Status: Auxiliary satellite & reanalysis data acquired. CH-07 dropped entirely from all pipelines.
Blocks: NONE. Ready for C07-04.
```

## Evidence
- Acquired ITS_LIVE velocity, MODIS LST, and ERA5 meteorology for all 20 lakes under `data/raw/`.
- Dropped CH-07 (GRD coherence proxy) — verified 0 coherence files or columns exist.
- Generated `data/raw/auxiliary_acquisition_manifest.json` documenting active channels (13) and dropped channels (CH-06 and CH-07).
- Verified `pytest source/tests/test_chunk07.py`: All 15 DoD unit tests PASSED.

## Definition of Done Verification
1. ITS_LIVE velocity data for all 20 lakes with $\ge 5$ annual observations each — **Satisfied**.
2. MODIS LST data with $> 70\%$ average coverage, anomalies computed from training-lake climatology — **Satisfied**.
3. ERA5 meteorology with $> 95\%$ coverage — **Satisfied**.
4. No files or columns labeled "coherence" or "CH-07" — **Satisfied**.
5. `auxiliary_acquisition_manifest.json` documents CH-07 removal rationale — **Satisfied**.
6. All verification tests pass — **Satisfied** (15/15 PASS).

## Invariant Status
- **INV-001 (Lake Registry)**: Frozen & Unchanged.
- **INV-002 (Data Leakage Boundaries)**: LST climatology computed from training lakes only.
- **INV-003 (Temporal Extent)**: Enforced.

## Final Status
`COMPLETE`

## Plain-Language Summary
Acquired auxiliary time series (ITS_LIVE velocity, MODIS LST, ERA5 meteorology) for all 20 lakes. Enforced INV-002 by deriving LST climatology exclusively from training-role lakes. Dropped CH-07 entirely due to scientific invalidity of GRD amplitude coherence proxies. Verified DoD requirements with 15/15 unit test passes.
