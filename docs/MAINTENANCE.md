# Long-Term Maintenance & System Evolution Plan

**Project:** Sentinel-GL  
**Document ID:** `DOC-MAINTENANCE-01`  
**Version:** 1.0.0  
**Effective Date:** October 2026  
**Status:** APPROVED & ACTIVE  

---

## 1. Scope and Objective

This document defines the operational procedures, maintenance protocols, and cryptographic safeguards required to sustain the **Sentinel-GL** research framework over multi-decadal observation lifecycles. It provides explicit guidelines for:
1. Ingesting next-generation Earth observation satellite data (Sentinel-1C/1D, Sentinel-2C/2D) and atmospheric reanalysis updates.
2. Managing provider API credentials and access secrets securely without leaking material into repository history.
3. Regulating model retraining triggers, preserving frozen training epochs, and upholding strict anti-leakage boundaries.
4. Monitoring Apple Silicon Metal Performance Shaders (MPS) runtime compatibility and detecting hardware numerical drift via CPU reference regressions.

---

## 2. Sensor & Provider Lifecycle Management

### 2.1 Next-Generation Copernicus Missions (Sentinel-1C/2C)
As the European Space Agency commissions Sentinel-1C (launched December 2024) and Sentinel-2C (launched September 2024) to replace aging 1A/2A satellites:
- **Product Identity & Format Compatibility:**
  - Sentinel-1C Ground Range Detected (GRD) Interferometric Wide (IW) Level-1 products maintain identical swath geometry and dual-polarization (VV + VH) radiometric calibration.
  - Sentinel-2C Multi-Spectral Instrument (MSI) Level-2A surface reflectance products preserve the 10 m / 20 m band allocation and Scene Classification Layer (SCL).
- **11-Channel Feature Schema Invariant:**
  - Ingestion adaptors in `source/sentinel_gl/features.py` must preserve the canonical 11-channel order without permutation:
    - Optical: `optical_lake_area_km2` (0), `ndwi_mean` (1), `mndwi_mean` (2), `ndsi_mean` (3)
    - SAR: `sar_vv_db_mean` (4), `sar_vh_db_mean` (5), `sar_cross_ratio_db` (6), `sar_vv_spatial_variance` (7)
    - ERA5: `era5_temp_2m_c_mean` (8), `era5_precip_mm_sum` (9), `era5_temp_anomaly_c` (10)
  - Next-generation products must undergo validation against the identical boolean validity mask protocol: cloud/shadow-obscured pixels ($\ge 30\%$ cloud cover) must produce explicit `NaN` values with mask bit `False`. Imputation with zero or channel averages is strictly prohibited.

### 2.2 Digital Elevation Models (DEM) & Curvature Updates
- High-mountain topography utilizes the Copernicus DEM GLO-30 (30-meter global elevation product).
- When updated DEM tiles or regional LiDAR/UAV high-resolution digital surface models become available:
  - Radar layover and shadow masks must be recomputed using the exact local incidence angle formulation:
    $$\cos(\theta_{\text{local}}) = \cos(\theta_{\text{inc}}) \cos(\alpha) + \sin(\theta_{\text{inc}}) \sin(\alpha) \cos(\phi_{\text{look}} - \beta)$$
    where $\alpha$ is terrain slope, $\beta$ is terrain aspect, $\theta_{\text{inc}}$ is radar incidence angle, and $\phi_{\text{look}}$ is antenna look azimuth.
  - DEM updates must be recorded in the dataset lineage manifest with SHA-256 tile digests prior to panel extraction.

### 2.3 ERA5 Atmospheric Reanalysis Latency Management
- ECMWF provides ERA5 in two temporal tiers:
  1. **ERA5T (Preliminary Reanalysis):** Published with a latency of approximately 5 days.
  2. **ERA5 Consolidated (Final Reanalysis):** Published with a latency of approximately 2 to 3 months following rigorous quality verification.
- **Operational Protocol:**
  - During retrospective evaluation, panels must use final consolidated ERA5.
  - For prospective real-time pilot simulations, ERA5T may be ingested, but timestamps must record the preliminary data flag `is_preliminary: true`.
  - When consolidated ERA5 replaces ERA5T, the ingestion pipeline must verify that retrospective closed-window contracts ($t_{\text{acq}} < t_{\text{decision}}$) are preserved without altering historical decision timestamps.

### 2.4 Safe Credential Rotation Policy
- **Absolute Secrecy Principle:**
  - No API keys, OAuth2 client secrets, or private tokens may ever be committed to git or stored in public project trees.
  - Provider credentials must reside exclusively in:
    - User home directory configuration files (`~/.cdsapirc` for ECMWF CDS API).
    - Ignored local directories (`.credentials/` or `.secrets/` matching `.gitignore`).
    - Standard operating system environment variables (`CDSE_CLIENT_ID`, `CDSE_CLIENT_SECRET`, `CDSAPI_KEY`, `CDSAPI_URL`).
- **Rotation Procedure:**
  1. Generate new API credentials in the Copernicus Data Space Ecosystem (CDSE) or ECMWF Climate Data Store portal.
  2. Configure temporary credentials in a local shell session.
  3. Execute a dry-run connectivity probe on a single, non-scientific bounding box using `source/sentinel_gl/ingestion/`.
  4. Confirm successful authentication and valid token response without logging secret bytes.
  5. Update active environment variables and revoke previous credentials in the provider portal.
  6. Execute `git status --ignored` and `python3 tools/verify_release.py` to ensure zero credential artifacts are staged or exposed.

---

## 3. Model Retraining & Anti-Leakage Safeguards

### 3.1 Retraining Triggers
Model retraining consumes significant computational resources and risks overfitting to noise. Retraining is permitted **only** when all of the following conditions are satisfied:
1. **Documented GLOF Event:** A catastrophic outburst flood or rapid moraine breach is documented in High Mountain Asia with peer-reviewed or official hydrological field validation (Tier 1 or Tier 2 evidence).
2. **Dense Pre-Event Acquisition:** At least 3 independent multi-modal observations (combining SAR and optical) exist within the 90-day window preceding event onset.
3. **Formal Protocol Amendment:** An amendment document is ratified recording the new lake ID, event metadata, river basin, and spatial cluster assignment before model execution.

**Prohibited Triggers:**
- Retraining is strictly forbidden on unverified candidate alerts, single-sensor anomalies, or false alarms.
- Retraining directly on post-event imagery to "learn breach morphology" is strictly forbidden for early warning models.

### 3.2 Holdout Preservation & Frozen Epochs
- **Immutable Checkpoints:**
  - All frozen model checkpoints (`docs/checkpoints/` and release bundles) are cryptographically hashed and permanent. Overwriting or mutating frozen weights is prohibited.
- **Cluster Isolation Protocol:**
  - Any newly admitted lake must be clustered with surrounding water bodies using the 50 km Haversine buffer rule:
    $$d_{\text{haversine}}(p_1, p_2) \ge 50.0 \text{ km}$$
  - Lakes within the same hydrological basin or mountain cluster must be assigned as an atomic block to either training, validation, or test partitions. Splitting lakes from the same cluster across partitions is a critical integrity violation (`SPLIT_LEAKAGE`).
- **Normalizer Immutability:**
  - Feature normalization state ($\mu, \sigma$) must be computed strictly and exclusively on designated training-split lakes.
  - Adding new evaluation or holdout lakes must never alter the training normalizer state hash by a single bit.

---

## 4. Runtime & Hardware Evolution

### 4.1 Apple Silicon M-Series Unified Memory Maintenance
Sentinel-GL is engineered to execute within the unified memory budget of Apple Silicon workstations (e.g., Apple M3, 16 GB unified RAM):
- **MPS Device Selection:**
  - Accelerated execution utilizes PyTorch's Metal Performance Shaders backend (`device="mps"`).
  - Availability must always be validated dynamically via `torch.backends.mps.is_available()`.
- **Memory Pressure Safeguards:**
  - Because unified memory is shared between CPU, GPU, and macOS system services, neural loaders must enforce batch bounds $B \in [8, 16]$.
  - Memory tracking must differentiate between PyTorch tensor allocations (`torch.mps.current_allocated_memory()`) and total operating system process memory (Resident Set Size, RSS via `psutil`).
  - Metal cache buffers must be periodically reclaimed using `torch.mps.empty_cache()` between long evaluation epochs.
- **Hardware Synchronization:**
  - Due to asynchronous Metal command queuing, all timed benchmarking blocks must call `torch.mps.synchronize()` immediately prior to timer start and immediately following kernel completion.

### 4.2 CPU Reference Regression Testing
To guarantee that Apple Silicon GPU kernel updates, macOS Metal driver changes, or PyTorch versions do not introduce numerical instability:
- **Numerical Tolerance Bound:**
  - Every model update or PyTorch version upgrade must execute the hardware regression trial in `source/sentinel_gl/hardware.py`.
  - The maximum absolute difference between MPS acceleration and CPU IEEE 754 reference execution must satisfy:
    $$\max |y_{\text{mps}} - y_{\text{cpu}}| \le 1.0 \times 10^{-4}$$
- **Fallback Protocol:**
  - If MPS numerical difference exceeds $1.0 \times 10^{-4}$ or if MPS raises runtime exceptions on novel operators, execution must fall back immediately to the deterministic CPU reference (`device="cpu"`).
  - Any CPU fallback must be explicitly declared in execution logs; concealing a CPU fallback as an MPS run is strictly forbidden.

---

## 5. Maintenance Checklist for Future Releases

Prior to publishing any minor or major version update:
- [ ] Run complete analytical test suite: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest source/tests -q -p no:cacheprovider` (must pass 100%).
- [ ] Run reproduction auditor: `python3 tools/verify_reproduction.py` (must output `OVERALL REPRODUCTION VERDICT: PASS`).
- [ ] Run release integrity auditor: `python3 tools/verify_release.py` (must output `STATUS: PASS`).
- [ ] Verify that no uncommitted credentials, tokens, or private keys exist: `git status --ignored`.
- [ ] Confirm clean release integrity status: `python3 tools/verify_release.py` (must pass 0 errors).
- [ ] Confirm figure SHA-256 digests in `docs/figures/manifest.json` match binary files on disk.
