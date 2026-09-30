# Sentinel-GL Feature Schema v2 Specification

## 1. Overview
Feature Schema v2 defines the canonical multi-sensor observation representation for glacial lakes in the HKH region.

The table below is the initial version `2.0` schema, not a frozen final publication channel count. Clean acquisition and domain review may add, remove, or revise channels only through an explicit schema-version change; the schema hash changes with both version and metadata.

## 2. Multi-Sensor Channels
| Channel ID | Sensor | Product | Physical Quantity | Unit | Valid Range | Temporal / Static |
|---|---|---|---|---|---|---|
| `s1_vv_backscatter` | Sentinel-1 | GRD | Radar Backscatter VV | dB | [-35.0, 5.0] | Temporal |
| `s1_vh_backscatter` | Sentinel-1 | GRD | Radar Backscatter VH | dB | [-40.0, 0.0] | Temporal |
| `s2_ndwi` | Sentinel-2 | MSI L2A | NDWI (B3-B8)/(B3+B8) | dimensionless | [-1.0, 1.0] | Temporal |
| `s2_mndwi` | Sentinel-2 | MSI L2A | MNDWI (B3-B11)/(B3+B11) | dimensionless | [-1.0, 1.0] | Temporal |
| `s2_lake_area` | Sentinel-2 | Delineated Polygon | Lake Surface Area | km² | [0.01, 10.0] | Temporal |
| `modis_lst_day` | MODIS | MOD11A1 | Daytime LST | Kelvin | [220.0, 320.0] | Temporal |
| `modis_lst_night` | MODIS | MOD11A1 | Nighttime LST | Kelvin | [210.0, 310.0] | Temporal |
| `era5_2m_temp` | ERA5-Land | Reanalysis | 2m Air Temperature | Kelvin | [220.0, 315.0] | Temporal |
| `era5_total_precip` | ERA5-Land | Reanalysis | Daily Total Precipitation | m/day | [0.0, 0.5] | Temporal |
| `era5_freezing_level` | ERA5 | Reanalysis | 0°C Isotherm Altitude | m a.s.l. | [0.0, 7500.0] | Temporal |
| `topo_elevation` | NASADEM | HGT 30m | Mean Surface Altitude | m a.s.l. | [2000.0, 7000.0] | Static |
| `topo_slope_surrounding` | NASADEM | HGT 30m | Moraine Slope (1km buffer) | degrees | [0.0, 85.0] | Static |

## 3. Split Isolation Governance
All scalers (e.g. RobustScaler) must fit parameters on training-split lakes only (`role == "training"`). Evaluation-lake features are strictly transformed using frozen training parameters (INV-021, INV-022).

Every channel exposes `transform_fit_scope`: fitted transforms resolve to `training_split_only`; channels with no fitted transform resolve to `not_applicable`.

## 4. Cadence and Missingness

- Sentinel-1 and Sentinel-2 channels use `irregular_scene` cadence, MODIS and daily ERA5-Land channels use `daily`, freezing-level cadence remains `provider_timestep`, and terrain channels use `static`.
- Every matrix channel has a required boolean validity mask named `<channel_id>__valid`.
- Invalid entries remain observable as `NaN` with a false validity mask. Finite values cannot be marked invalid, and non-finite values cannot be marked valid.
- Interpolation or seasonal-mean parameters are fitted on training lakes only. Precipitation missingness is not silently converted to physical zero.

## 5. Validation Boundary

The validator rejects unknown channel IDs, missing channels/masks, nonnumeric values, non-boolean masks, mismatched shapes, non-finite valid values, and values outside declared physical ranges. These are screening bounds for schema integrity; they are not evidence that a provider observation is authentic or scientifically sufficient.
