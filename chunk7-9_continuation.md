# sentinel-gl: Architect Continuation Document — Chunks 07–09 (IEEE TGRS Major Revision)

**Document Status:** Frozen — this is the complete, authoritative handoff for the Architect to plan and execute Chunks 07, 08, and 09.
**Factory Version Target:** 1.4.0 (Scientific Validity Layer)
**Project Version:** Post-Chunk-06 (v1.0 complete, IEEE review received, independent critique incorporated)
**IEEE Review Score Received:** 4/10 (Reject leaning Major Revision)
**Target After Chunk 09:** Release-Readiness (see Part 11 — NOT a reviewer score target)
**Target Venue:** IEEE Transactions on Geoscience and Remote Sensing (TGRS)
**Created:** 2026-08-11
**Revision:** 2.0 — incorporates all valid criticisms from independent methodological review
**Author:** Independent IEEE Reviewer Analysis + Factory v1.4.0 Methodology + Independent Methodological Critique

---

## Part 0 — Document Purpose, Authority, and Revision History

### 0.1 What This Document Is

This is the single authoritative reference for the Architect (Claude) to plan and execute Chunks 07, 08, and 09 of the sentinel-gl project. It contains everything needed: the diagnosis of what went wrong, the exact corrections required, the Factory v1.4.0 gates that govern execution, the venue requirements, the pre-registered falsification criteria, the detailed contract specifications, the stop conditions, and the release-readiness criteria.

No other document supersedes this one for the purpose of planning Chunks 07–09. The original 6-chunk founding artifacts (`project_description.md`, `architecture.md`, `roadmap.md`, `project_knowledge.md`, `invariants.md`) remain frozen and authoritative for everything they cover. This document extends and amends them where the IEEE review and independent critique identified deficiencies.

### 0.2 Revision History

| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-08-11 | Initial version. Addressed IEEE review criticisms C1–C4, M1–M5, m1–m7. |
| 2.0 | 2026-08-11 | Incorporates 10 valid criticisms from independent methodological review: three-state Reality Gate, cloud-fraction stratification, CH-07 removal, baseline missing-data policy, CUSUM scoring specification, lake-level statistical unit, corrected training verification, corrected C53, release-readiness criteria replacing score targets, gate-driven timeline. |

### 0.3 Authority Hierarchy for This Document

```
Constitution (C01–C50, C51–C53, EP-001–EP-007)
  ↓
Factory Specification v1.4.0 (MAR, Reality Gate, SVI, Scientific Claim Tier)
  ↓
This Document (Chunks 07–09 specification)
  ↓
Individual Contracts (C07-XX, C08-XX, C09-XX)
  ↓
Implementation
```

Where this document conflicts with the original founding artifacts, this document takes precedence for Chunks 07–09 specifically, because it represents the Architecture Amendment triggered by the IEEE review (per `factory_spec.md`'s Architecture Amendment procedure). The original artifacts are not overwritten — they are amended.

---

## Part 1 — Project Status Assessment: Where You Are

### 1.1 What Exists and Works (Reuse Without Modification)

| Asset | Location | Status | Reuse in Chunks 07–09 |
|---|---|---|---|
| Lake Registry (20 lakes, roles frozen) | `source/data/registry/lake_registry.json` | ✅ INV-001 frozen | Reuse as-is. Do NOT change lake IDs, roles, or boundaries. |
| TS-MAE encoder architecture code | `source/models/encoder/ts_mae.py` | ✅ 198 tests pass | Reuse architecture. Retrain on real data (Chunk 07). |
| Training loop + checkpoint management | `source/scripts/train_ts_mae.py` | ✅ Converged in 14s | Reuse with new data. Hyperparameters may need adjustment. |
| Anomaly scoring (Score-A, B, C) | `source/models/anomaly/` | ✅ Verified | Reuse. Scores will change with new embeddings. |
| Evaluation Protocols E1–E4 code | `source/evaluation/protocols/` | ✅ Verified | Reuse. Results will change with real data. |
| Claim-evidence verification pipeline | `source/scripts/verify_claim_evidence.py` | ✅ 25/25 pass | Reuse. All claims will get new values. |
| Ablation framework code | `source/scripts/run_ablation.py` | ✅ Verified | Reuse with corrected methodology (Chunk 09). |
| Threshold sweep script | `source/scripts/run_threshold_analysis.py` | ✅ Verified | Reuse on new score distributions. |
| Unit test suite (198 tests) | `source/tests/` | ✅ All pass | All remain valid. New tests added for baselines. |

### 1.2 What Is Broken and Must Change

| Problem | Severity | Root Cause | Fix Location |
|---|---|---|---|
| Features are GEE-simulated, not real observations | CRITICAL | Architecture decision at Project Initialization | Chunk 07 |
| 0% data gaps (real HKH data has 20–40% monsoon gaps) | CRITICAL | Simulation produces complete matrices | Chunk 07 (Reality Gate) |
| Only 1 trivial baseline (static extent threshold) | CRITICAL | Roadmap only defined one baseline | Chunk 08 |
| Protocol E1 failed (Score-B flags entire series) | HIGH | South Lhonak embeddings OOD from start | Chunk 08 (honest resolution) |
| No statistical significance testing | HIGH | Not specified in invariants | Chunk 08 |
| Title overclaims ("Cloud-Robust Precursor Detection") | HIGH | Title written before validation | Chunk 09 |
| Zero-retraining ablation confound (zero-masking) | MEDIUM | Zeroing columns creates OOD input | Chunk 09 |
| No architecture specification in manuscript | MEDIUM | Omitted during writing | Chunk 09 |
| Hyperparameters unjustified (α=0.5, EMA span=5) | MEDIUM | No sensitivity analysis | Chunk 09 |
| Factual errors (7 vs 55 fatalities, coordinates) | LOW | Manuscript vs project_knowledge mismatch | Chunk 09 |
| Local path in REPRODUCIBILITY.md | LOW | `/Users/adi/Desktop/...` leaked | Chunk 09 |
| CH-07 labeled "coherence" but derived from GRD | CRITICAL | GRD cannot produce true coherence | Chunk 07 (drop CH-07) |
| No missing-data policy for baselines | HIGH | sklearn cannot handle NaN | Chunk 08 |
| CUSUM scoring mechanism unspecified | MEDIUM | Binary alarm vs. continuous score | Chunk 08 |
| Statistical unit undefined (window vs. lake) | CRITICAL | Pseudoreplication risk | Chunk 08 |

### 1.3 What the IEEE Reviewer Said (Key Quotes)

> **C1:** "The entire evaluation is conducted on simulated features rather than actual satellite observations. This undermines every quantitative claim, renders the 'cloud-robust' title misleading, and makes the successful synthetic detection results trivially explainable."

> **C2:** "This is the most critical experiment in the paper, and it failed."

> **C3:** "Without these comparisons, the claimed AUC-ROC of 0.9521 is uninterpretable."

> **C4:** "For a methods paper targeting IEEE TGRS, this is unacceptable."

> **Overall:** "Score: 4/10 (Reject with encouragement to resubmit after fundamental revision)"

### 1.4 What the Independent Methodological Critique Added

> "The Reality Gate is partly backwards... Real data doesn't have to obey your predetermined percentage."

> "CH-07 coherence proxy... True interferometric coherence generally requires SLC data and paired interferometric processing, not ordinary GRD backscatter."

> "Standard sklearn Isolation Forest does not simply consume arbitrary NaNs."

> "Having thousands of temporal windows does not mean you have thousands of independent samples."

> "This is psychologically dangerous for the factory. It creates an optimization target: make the paper score 10/10 instead of: make the methodology correct."

All of these criticisms are valid and are incorporated into this revision.

---

## Part 2 — IEEE Review Criticisms Mapped to Fixes

| Review Point | Fix | Chunk | Contract |
|---|---|---|---|
| C1: Simulated features | Real data acquisition + Reality Gate | 07 | C07-01 through C07-04 |
| C2: Protocol E1 failed | Pre-registered criteria + honest resolution | 08 | C08-06 |
| C3: No competitive baselines | 3 new baselines (Isolation Forest, CUSUM, One-Class SVM) | 08 | C08-01 through C08-03 |
| C4: No architecture detail | Full specification added to manuscript | 09 | C09-02 |
| M1: No statistical testing | Lake-level bootstrap + DeLong's test | 08 | C08-05 |
| M3: Factual errors | Corrected in manuscript | 09 | C09-02 |
| M5: Ablation confound | Masking token or sensitivity analysis | 09 | C09-01 |
| m1: Title overclaim | Title revised based on actual results | 09 | C09-02 |
| m2: Missing related work | Venue grounding in venue_requirements.md | 07 | Pre-chunk |
| m3: α=0.5 unjustified | Sensitivity analysis | 09 | C09-01 |
| m4: EMA span unjustified | Sensitivity analysis | 09 | C09-01 |
| m6: Local path leaked | Release Artifact Scan | 09 | C09-04 |
| CH-07 coherence from GRD | Drop CH-07, reduce to 14 channels | 07 | C07-03 |
| Baseline NaN handling | Explicit imputation policy | 08 | C08-01 through C08-03 |
| CUSUM scoring unspecified | Continuous score formula specified | 08 | C08-02 |
| Statistical unit undefined | Lake-level bootstrap specified | 08 | C08-05 |
| Training loss monotonicity | Replaced with convergence criteria | 07 | C07-05 |
| C53 Score-A too strong | Weakened: Score-A null ≠ architecture failure | 07 | C07-05 |
| "10/10" target | Replaced with Release-Readiness Criteria | — | Part 11 |
| Calendar timeline | Replaced with gate-driven progression | — | Part 13 |

---

## Part 3 — Factory v1.4.0 Integration: Mandatory Gates

### 3.1 Methodology Adversarial Review (MAR) — Already Completed

The MAR was performed by the independent IEEE reviewer and the independent methodological critique. Combined results:

| Gate | Verdict | Finding |
|---|---|---|
| MAR-1 Data Authenticity | **FAIL** | GEE-simulated features, not real observations |
| MAR-2 Baseline Sufficiency | **FAIL** | 1 trivial baseline; need ≥3 |
| MAR-3 Statistical Power | **FAIL** | No CI, no significance test, pseudoreplication risk |
| MAR-4 Title-Claim Consistency | **FAIL** | "Cloud-Robust" untested; "Precursor Detection" failed |
| MAR-5 Venue Alignment | **FAIL** | No venue papers referenced; methodology diverges |
| MAR-6 Assumption Stress Test | **CONDITIONAL PASS** | If real data fails, InSAR negative result still publishable |
| MAR-7 Negative Result Contingency | **PASS** | InSAR infeasibility + evaluation protocol are standalone |

**MAR Gate Behavior:** FAIL on gates 1–5 → Founding artifacts must be revised. Chunks 07–09 ARE the revision.

### 3.2 Scientific Validity Invariants (SVI) — Mandatory for This Project

| ID | Invariant | Verification Method | Failure Impact |
|---|---|---|---|
| SVI-001 | All features derive from real satellite observations with documented gaps, OR project is explicitly labeled simulation-based | Reality Gate (automatic, Chunk 07) | CRITICAL |
| SVI-002 | Evaluation includes ≥3 baselines: (a) operational, (b) statistical, (c) learned | Check evaluation config before Protocol E4 runs | CRITICAL |
| SVI-003 | No quantitative claim based on pseudoreplicated samples; every comparative claim has lake-level CI or significance test | Check statistical methodology artifact exists | HIGH |
| SVI-004 | Every adjective in the title maps to a specific validated experimental test | MAR + Chunk Review Title-Claim Audit | HIGH |
| SVI-005 | Methodology checked against ≥3 recent TGRS publications | `venue_requirements.md` populated | MEDIUM |
| SVI-006 | Data properties match methodology assumptions before training | Reality Gate (automatic, Chunk 07) | CRITICAL |

### 3.3 Reality Gate — Three-State Design (REVISED per independent critique)

**Position:** After C07-03 (feature assembly from real data), before C07-05 (encoder retraining). Mandatory. Blocks training on FAIL.

**CRITICAL REVISION:** The Reality Gate uses THREE states, not two. Real data does not have to obey predetermined percentages exactly. The gate distinguishes between "suspicious" (investigate) and "impossible" (stop).

| State | Meaning | Action |
|---|---|---|
| **PASS** | Matches expected range AND independently plausible | Proceed to training |
| **WARNING** | Unusual but physically possible | Flag for Human/domain review before proceeding. Document the anomaly. Proceed only after Human acknowledges. |
| **FAIL** | Evidence of simulation, processing error, or physically impossible data | STOP. Architecture Amendment triggered. Do NOT proceed to training. |

**Reality Gate Checks:**

```
CHECK 1: Gap statistics
  PASS:    Gap rate within ±10pp of declared expectation for domain/region/season
  WARNING: Gap rate outside declared range but physically plausible
           (e.g., unusually clear monsoon for one lake, sensor outage on 1-2 lakes)
  FAIL:    <2% gaps across ALL optical channels for ALL 20 lakes during
           monsoon (Jun-Sep). This is physically impossible for real
           Sentinel-2 over HKH and indicates simulation.

CHECK 2: Distribution variance
  PASS:    Per-channel std varies by >2× between at least 5 lake pairs
  WARNING: Per-channel std varies by <2× but >1.2× (lakes are similar
           but not identical — investigate whether selection bias)
  FAIL:    Per-channel std is uniform (<1.2× variation) across all 20
           physically distinct lakes at different altitudes, latitudes,
           and sizes.

CHECK 3: Temporal coverage
  PASS:    ≥80% of declared extent for ≥17 of 20 lakes
  WARNING: 60-80% coverage for some lakes (document which and why)
  FAIL:    <50% coverage for >5 lakes

CHECK 4: Sensor coverage
  PASS:    All 13 active channels present for ≥17 of 20 lakes
           (14 channels minus CH-07, which is dropped)
  WARNING: 1-2 channel groups missing for 3-5 lakes (document which)
  FAIL:    >3 lakes missing an entire expected channel group

CHECK 5: Cloud contamination presence
  PASS:    NaN/missing in optical channels during Jun-Sep for ≥60% of lakes
  WARNING: NaN/missing in optical channels during Jun-Sep for 40-60% of lakes
  FAIL:    Zero monsoon gaps in optical channels for all lakes
```

**Any FAIL → STOP. Any WARNING → Human reviews before proceeding. All PASS → proceed.**

### 3.4 Scientific Claim Tier — Applied to Every Contract

| Contract | Claim Type | Tier | Minimum Evidence Required |
|---|---|---|---|
| C07-05 (Encoder retraining) | T-DESC | Training convergence, no leakage | Training log + leakage test output |
| C08-04 (Protocol E1–E4 re-run) | T-COMP | ≥3 baselines + lake-level bootstrap CI | Evaluation summary JSON + CI artifact |
| C08-06 (Protocol E1 resolution) | T-CAUSAL | Adversarial test + declared Stop Condition | Pre-registered threshold + honest outcome |
| C09-02 (Manuscript rewrite) | T-DESC | Claim-evidence traceability | All claims verified |

### 3.5 C53 — Null Result Stop Condition (REVISED per independent critique)

**ORIGINAL (too strong):** "If Score-A AUC-ROC ≤ 0.55, this is a Stop Condition for the architecture."

**REVISED:** Score-A (reconstruction error) being a poor anomaly detector does NOT mean the TS-MAE architecture is unsuitable. The original project proved this: Score-A was null (AUC-ROC 0.4552) but Score-B achieved 0.8973 and Score-C achieved 0.9521. The encoder's representations were useful even though reconstruction error was not.

**Corrected C53 application:**

- Score-A AUC-ROC ≤ 0.55 is **NOT** a Stop Condition for the architecture. It is:
  1. A finding to report honestly (C36 — preserve null results)
  2. A reason to reduce α (Score-A's weight in Score-C) or set α=0
  3. A reason to investigate WHY reconstruction error is non-discriminative
  4. **NOT** a reason to halt the project or declare the architecture unsuitable

- The **actual Stop Condition for the architecture** is:
  - Score-B AUC-ROC ≤ 0.55 on synthetic anomalies (embedding distance is also non-discriminative → the encoder learned nothing useful), OR
  - All baselines outperform Score-C by >0.05 AUC-ROC (the learned representation adds no value)

### 3.6 Release-Readiness Criteria (REPLACES "10/10" target)

**CRITICAL REVISION:** The original document set "Target Score After Chunk 09: 9–10/10." This is psychologically dangerous. It creates an optimization target of "make the reviewer give a high score" instead of "make the methodology correct." A scientifically honest outcome where TS-MAE underperforms baselines and Protocol E1 fails could still be an excellent paper — just a different paper.

**The Factory optimizes for scientific correctness, not reviewer scores.**

See Part 11 for the complete Release-Readiness Criteria.

---

## Part 4 — Venue Requirements: IEEE TGRS (Populated)

### 4.1 Target Venue

**IEEE Transactions on Geoscience and Remote Sensing (TGRS)**
- Impact Factor: ~8.2 (2024)
- Scope: Remote sensing science, algorithms, and applications
- Typical paper length: 10–14 pages (double-column)
- Review cycle: 3–6 months (initial decision), 2–3 rounds typical
- Acceptance rate: ~20–25%

### 4.2 Recent Venue Publications (SVI-005 — minimum 3, required)

| Paper | Year | Methodology | Baselines Used | Sample Size | Key Convention |
|---|---|---|---|---|---|
| Zhang et al., "Precursory Deformation Detection on Tailings Dams Using Time-Series InSAR" | 2024 | SBAS-InSAR + statistical change detection | Static threshold, CUSUM | 3 dams, 5+ years each | Reports coherence statistics, detection lead time with CI |
| Intrieri et al., "InSAR Precursor Displacement Tracking for Catastrophic Slope and Dam Failures" | 2019 | Multi-temporal InSAR + velocity thresholding | Geological survey baseline | 3 case studies | Honest negative results for sites without precursors |
| Fatima et al., "IceWatch: Multimodal Deep Learning for GLOF Forecasting" | 2026 | Multimodal CNN (Sentinel-2 + ITS_LIVE + MODIS) | Random Forest, LSTM, static threshold | 50+ lakes, 10 events | Reports AUC-ROC with 95% CI, ablation per sensor |

### 4.3 Minimum Expected Baselines (SVI-002)

1. **Operational standard:** Static threshold on lake extent change rate (existing Protocol E4 baseline — KEEP)
2. **Non-learned statistical method:** CUSUM (Cumulative Sum) applied to lake extent time series
3. **Competitive learned method (non-deep):** Isolation Forest on the feature matrices
4. **Competitive learned method (non-deep, second):** One-Class SVM on the feature matrices

### 4.4 Minimum Expected Statistical Rigor

- Every AUC-ROC/AUC-PR reported with 95% bootstrap CI (lake-level resampling, N=2000)
- Every pairwise comparison includes DeLong's test or permutation test
- Sample sizes stated explicitly for every metric
- **Statistical unit is the LAKE, not the window** (see Part 6, C08-05)
- If N < 5 independent units for any evaluation, state this as a limitation

### 4.5 Title/Claim Conventions (SVI-004)

| Anticipated Title Adjective | Required Validating Test | Status |
|---|---|---|
| "Cloud-Robust" | Cloud-fraction stratified evaluation showing maintained performance at >60% cloud | UNTESTED — Chunk 08 |
| "Self-Supervised" | Encoder trained without labels | ✅ Already validated |
| "Multi-Sensor" | ≥3 sensor families contribute; ablation shows each contributes | ✅ Already validated |
| "Precursor Detection" | Successful detection on ≥1 real event with pre-registered threshold | FAILED in v1.0 — Chunk 08 must resolve |

### 4.6 Expected Data Properties (Feeds Reality Gate)

| Property | Expected Value | Basis |
|---|---|---|
| Optical gap rate (monsoon, Jun-Sep) | 20–45% | Sentinel-2 5-day revisit + HKH monsoon cloud climatology (ICIMOD reports, ERA5 cloud fraction data) |
| Optical gap rate (dry season, Oct-May) | 5–15% | Lower cloud cover but still present |
| SAR gap rate | 0–10% | Sentinel-1 penetrates clouds; gaps from orbit coverage only |
| ERA5 gap rate | <2% | Reanalysis product, near-complete |
| ITS_LIVE cadence | Annual (1 obs/year) | Documented product cadence |
| Temporal coverage | 2016-01-01 to 2024-10-31 | INV-003 |
| Active channels | 13 (CH-07 dropped) | CH-06 excluded (InSAR infeasible), CH-07 excluded (GRD cannot produce coherence) |

### 4.7 Release Artifact Notes

- No `/Users/adi/...` paths anywhere in release artifacts
- No local machine identifiers
- All paths relative to repository root
- `gatekeeper.py release-check` must pass before submission

---

## Part 5 — Pre-Registered Falsification Criteria

These are declared NOW, before any Chunk 07–09 work begins. They cannot be changed after seeing results.

| Hypothesis | Falsifying Result | Action if Falsified |
|---|---|---|
| "TS-MAE learns useful representations from real satellite data" | Score-B AND all baselines produce AUC-ROC < 0.60 on synthetic anomalies | Architecture Amendment: consider LSTM-AE or Isolation Forest as primary method. Reframe paper. |
| "Real data produces meaningful discrimination" | Score-C AUC-ROC on real data < 0.60 | Investigate cause. If real data is genuinely harder, report honestly. Do NOT tune until it exceeds a target. |
| "Protocol E1 can detect a pre-event signal" | Score-C flags >80% of pre-event period OR never exceeds threshold | Remove "Precursor Detection" from title. Reframe as evaluation framework + negative result. |
| "Multi-sensor fusion outperforms single-sensor" | Full 13-channel AUC-ROC < best single-modality AUC-ROC + 0.02 | Multi-sensor claim weakened. Report honestly. |
| "Cloud-robust" claim is valid | AUC-ROC in >60% cloud bin < AUC-ROC in <20% cloud bin − 0.10 | Remove "Cloud-Robust" from title. Report seasonal performance gap. |
| "TS-MAE outperforms baselines" | Best baseline AUC-ROC > Score-C AUC-ROC | Major reassessment. Paper contribution becomes evaluation framework + comparison, not the TS-MAE method. |

---

## Part 6 — Chunk 07: Data Foundation Upgrade

### 6.1 Chunk Objective

Replace all GEE-simulated feature matrices with real, preprocessed satellite observations for all 20 study lakes. Drop CH-07 (coherence proxy from GRD is scientifically invalid). Retrain the TS-MAE encoder on real features. Extract new embeddings. Pass the Reality Gate.

### 6.2 Chunk Weight

**Standard.** Contains High-tier contracts (Reality Gate, encoder retraining) and Medium-tier contracts (acquisition, preprocessing).

### 6.3 Dependencies

- Chunk 01: Lake Registry (frozen, INV-001) ✅
- Chunk 02: Acquisition scripts (scaffolding exists) ✅
- All 198 existing tests pass ✅

### 6.4 Contracts

---

#### C07-01: Real Data Acquisition — Sentinel-1 GRD

**Risk Tier:** Medium
**Implementation Owner:** Gemini
**Objective:** Download real Sentinel-1 GRD (VV+VH) backscatter for all 20 lakes, 2016-01-01 to 2024-10-31.

**Allowed Files:**
- `source/data/acquisition/acquire_sentinel1.py`
- `source/config/acquisition_config.yaml`
- `data/raw/sentinel1/` (output directory)

**Frozen Files:**
- `source/data/registry/lake_registry.json` (INV-001)

**Implementation Instructions:**
1. For each lake in the registry, query GEE for Sentinel-1 GRD IW mode, VV+VH polarization.
2. Extract mean backscatter (dB) over the lake polygon and a 500m buffer around the moraine dam.
3. Record acquisition date, orbit direction, relative orbit number for every scene.
4. Do NOT interpolate gaps. Missing observations remain as NaN.
5. Write output as `data/raw/sentinel1/{lake_id}/backscatter_timeseries.csv`.
6. Log total scenes acquired per lake and percentage of expected scenes obtained.

**Verification Scripts:**
- Assert: Every lake has a `backscatter_timeseries.csv` file
- Assert: Date range covers ≥80% of 2016-01-01 to 2024-10-31
- Assert: NaN values exist (real data has gaps from orbit coverage)
- Assert: VV and VH columns are both present and non-constant

**Stop Condition:** If >3 lakes have <50% of expected Sentinel-1 acquisitions, STOP. Report which lakes are affected and why.

**Definition of Done:**
1. All 20 lakes have Sentinel-1 backscatter time series
2. Acquisition manifest records total scenes per lake
3. Gap statistics logged (expected: 0–10% for SAR)

---

#### C07-02: Real Data Acquisition — Sentinel-2 L2A + Cloud Masking

**Risk Tier:** Medium
**Implementation Owner:** Gemini
**Objective:** Download real Sentinel-2 L2A imagery, apply cloud masking, extract NDWI, spectral indices, and lake area.

**Allowed Files:**
- `source/data/acquisition/acquire_sentinel2.py`
- `source/data/preprocessing/cloud_mask_s2.py`
- `data/raw/sentinel2/` (output directory)

**Frozen Files:**
- `source/data/registry/lake_registry.json` (INV-001)

**Implementation Instructions:**
1. Query GEE for Sentinel-2 L2A (COPERNICUS/S2_SR_HARMONIZED) for each lake.
2. Apply cloud masking: SCL band (exclude classes 3, 8, 9, 10, 11) AND s2cloudless probability > 0.6.
3. Compute NDWI = (B03 - B08) / (B03 + B08) per scene.
4. Compute lake area via NDWI > 0.3 threshold.
5. **CRITICAL:** If cloud cover > 80% for a scene over the lake, mark that date as MISSING (NaN). Do NOT fill gaps.
6. Record the actual cloud fraction per scene for later cloud-stratified evaluation.
7. Write output as `data/raw/sentinel2/{lake_id}/optical_timeseries.csv` with columns: `date, ndwi_mean, lake_area_km2, green_mean, red_mean, nir_mean, cloud_fraction, n_valid_pixels`.

**Verification Scripts:**
- Assert: Every lake has an `optical_timeseries.csv`
- Assert: `cloud_fraction` column exists and has values > 0.5 during monsoon months for ≥60% of lakes
- Assert: NaN values exist in the time series (real data has gaps)
- Assert: Lake area values are physically plausible (0.01–50 km²)

**Stop Condition:** If cloud masking produces 0% gaps across all lakes (every scene cloud-free), STOP. This is physically impossible for HKH during monsoon.

**Definition of Done:**
1. All 20 lakes have optical time series with REAL cloud gaps
2. Gap statistics: expect 20–40% missing during monsoon, 5–15% during dry season
3. Cloud fraction recorded per scene (needed for cloud-stratified evaluation in Chunk 08)

---

#### C07-03: Real Data Acquisition — Auxiliary Channels + CH-07 Removal

**Risk Tier:** Medium
**Implementation Owner:** Gemini
**Objective:** Acquire auxiliary channels (ITS_LIVE, MODIS, ERA5). **Drop CH-07 entirely.**

**Allowed Files:**
- `source/data/acquisition/acquire_itslive.py`
- `source/data/acquisition/acquire_modis.py`
- `source/data/acquisition/acquire_era5.py`
- `data/raw/itslive/`, `data/raw/modis/`, `data/raw/era5/` (output directories)

**Frozen Files:**
- `source/data/registry/lake_registry.json` (INV-001)

**Implementation Instructions:**
1. **ITS_LIVE:** Query for annual velocity composites at each lake's feeding glacier terminus.
2. **MODIS LST:** Query for daily LST. Apply quality flags. Compute anomaly relative to per-lake seasonal climatology.
3. **ERA5:** Query CDS API for 2m temperature, total precipitation, snow depth. Aggregate hourly → daily.
4. **CH-07 REMOVAL:** Do NOT compute any "coherence proxy" from GRD data. True interferometric coherence requires SLC data and paired interferometric processing. GRD contains only amplitude (backscatter), not phase. Any quantity derived from GRD and labeled "coherence" is scientifically invalid. CH-07 is dropped from the feature matrix. The project proceeds with 13 active channels (14 minus CH-07, with CH-06 already excluded).

**Rationale for CH-07 removal:** The original project's architecture.md §3.4 lists CH-07 as "Interferometric SAR coherence" sourced from Sentinel-1. However, the acquisition pipeline uses GRD data, which contains only detected amplitude, not complex phase. Computing "coherence" from GRD is not possible. The project's own InSAR infeasibility finding (mean coherence 0.24 vs. 0.30 threshold) further confirms that C-band coherence is unreliable on moraines. CH-07 should be dropped, not renamed.

**Verification Scripts:**
- Assert: ITS_LIVE has ≥5 annual observations per lake (2016–2024)
- Assert: MODIS LST has >70% coverage
- Assert: ERA5 has >95% coverage
- Assert: NO file or channel labeled "coherence" or "CH-07" exists in the output

**Definition of Done:**
1. All auxiliary channels acquired for all 20 lakes
2. CH-07 is absent from all outputs
3. Gap statistics documented per channel

---

#### C07-04: Feature Matrix Assembly + Reality Gate

**Risk Tier:** HIGH
**Implementation Owner:** Claude (Architect)
**Objective:** Assemble the 13-channel feature matrix from all acquired real data. Run the three-state Reality Gate. BLOCK if Reality Gate returns FAIL.

**Allowed Files:**
- `source/data/channels/assemble_features.py`
- `data/features/{lake_id}/feature_matrix.npz` (output)
- `results/reality_gate/reality_gate_report.md` (output)

**Frozen Files:**
- `source/data/registry/lake_registry.json` (INV-001)
- `invariants.md` (INV-003 temporal extent)

**Implementation Instructions:**
1. For each lake, assemble the 13-channel feature matrix aligned to a common daily time axis.
2. Channels: CH-01 (lake area), CH-02 (spectral, 4 channels), CH-03 (velocity, 2), CH-04 (LST anomaly), CH-05 (SAR backscatter, 3), CH-08 (meteorological, 3). Total: 13 channels.
3. **Do NOT interpolate gaps.** Missing values remain as NaN.
4. Compute z-score normalization statistics exclusively from training-role lakes (INV-002).
5. Write `feature_matrix.npz` per lake with shape `[T, 13]`.
6. **Run Reality Gate checks** (see §3.3). Write results to `reality_gate_report.md`.

**Reality Gate Output Format:**
```markdown
# Reality Gate Report — {date}

## CHECK 1: Gap Statistics
Per-lake gap rates: {table}
Verdict: PASS / WARNING / FAIL
Notes: {if WARNING, explain what's unusual}

## CHECK 2: Distribution Variance
Per-channel std across lakes: {table}
Verdict: PASS / WARNING / FAIL

## CHECK 3: Temporal Coverage
Per-lake coverage: {table}
Verdict: PASS / WARNING / FAIL

## CHECK 4: Sensor Coverage
Per-lake channel presence: {table}
Verdict: PASS / WARNING / FAIL

## CHECK 5: Cloud Contamination Presence
Monsoon gap presence: {table}
Verdict: PASS / WARNING / FAIL

## OVERALL VERDICT: PASS / WARNING / FAIL
## Action: {proceed / human review required / STOP}
```

**Stop Condition:** If ANY check returns FAIL, STOP. Do NOT proceed to encoder training. Diagnose the failure.

**Definition of Done:**
1. Feature matrices assembled for all 20 lakes (13 channels)
2. Reality Gate report written with per-check verdicts
3. Overall verdict is PASS or WARNING (if WARNING, Human has reviewed)
4. Normalization statistics computed from training lakes only (INV-002)

---

#### C07-05: Encoder Retraining on Real Features

**Risk Tier:** HIGH
**Implementation Owner:** Claude (Architect)
**Objective:** Retrain the TS-MAE encoder on real feature matrices. Validate convergence. Extract new embeddings.

**Allowed Files:**
- `source/models/encoder/ts_mae.py` (read only unless Architecture Amendment)
- `source/scripts/train_ts_mae.py`
- `models/checkpoints/ts_mae_real_data.pt` (new checkpoint)
- `models/encoder/training_summary_real_data.json` (output)
- `data/embeddings/real_data/` (output)

**Frozen Files:**
- `source/data/registry/lake_registry.json` (INV-001)
- `invariants.md` (INV-002, INV-004, INV-005, INV-008, INV-012)

**Implementation Instructions:**
1. Load real feature matrices from `data/features/`.
2. Construct training windows: T=180 days, stride=30 (INV-004).
3. Apply z-score normalization using training-lake-only statistics (INV-002).
4. Encode missing values with binary availability mask (do NOT interpolate).
5. Train TS-MAE with 50% temporal masking (INV-005).
6. Seeds: encoder=42, masking=42 (INV-012).
7. Compute budget: ≤72 hours single GPU or equivalent (INV-008).

**CORRECTED Training Verification (replaces "monotonically decreasing loss"):**

Neural network training loss does NOT need to decrease monotonically. It can legitimately fluctuate due to learning rate scheduling, batch normalization, different masking patterns per batch, and irregular data quality. The following criteria replace the monotonicity check:

```
Assert: Final training loss < 50% of initial training loss
Assert: No NaN or Inf values in loss at any epoch
Assert: Validation loss at final epoch < 2× minimum validation loss
        (no catastrophic divergence)
Assert: Reconstruction MSE on held-out training windows is below
        a threshold declared BEFORE training begins
Report: Full loss curve (training + validation) as evidence artifact
```

**CORRECTED C53 Application:**

After training, compute Score-A (reconstruction MSE) on a held-out validation set.

- If Score-A AUC-ROC ≤ 0.55: This is a FINDING, not a Stop Condition. Report it honestly. Consider setting α=0 in Score-C. Continue with Score-B and Score-C evaluation.
- If Score-B AUC-ROC ≤ 0.55: THIS is a Stop Condition. The encoder's embeddings are non-discriminative. Halt and assess architecture suitability.
- If all baselines (to be computed in Chunk 08) outperform Score-C by >0.05 AUC-ROC: This is a Stop Condition for the TS-MAE contribution claim. The paper must be reframed.

**Verification Scripts:**
- Assert: Training loss convergence criteria met (see above)
- Assert: No evaluation-lake data entered training (INV-002 leakage test)
- Assert: Embeddings have expected dimensionality
- Assert: Embeddings for different lakes show non-trivial variance
- Assert: Checkpoint file exists and loads successfully

**Stop Condition:**
- Training loss diverges or produces NaN → STOP, diagnose
- Score-B AUC-ROC ≤ 0.55 on validation → STOP, assess architecture (C53)
- Training exceeds INV-008 compute budget → STOP, document per C13

**Definition of Done:**
1. Trained checkpoint saved
2. Training summary JSON with loss curves, epoch count, wall-clock time
3. Embeddings extracted for all 20 lakes
4. INV-002 leakage test passes
5. Score-A preliminary check documented (even if null result)

---

### 6.5 Chunk 07 Success Criteria

1. ✅ All 20 lakes have real satellite data with documented gaps
2. ✅ CH-07 is absent (dropped, not renamed)
3. ✅ Reality Gate: All 5 checks return PASS or WARNING (no FAIL)
4. ✅ Encoder retrained on real features, convergence verified
5. ✅ Embeddings extracted for all lakes
6. ✅ No data leakage (INV-002 verified)
7. ✅ Gap statistics documented (expected: 20–40% optical monsoon gaps)

---

## Part 7 — Chunk 08: Evaluation Overhaul

### 7.1 Chunk Objective

Re-run all evaluation protocols on real-data embeddings. Add 3 competitive baselines with explicit missing-data policies. Compute lake-level bootstrap confidence intervals. Perform cloud-fraction stratified evaluation. Resolve Protocol E1 honestly with pre-registered criteria.

### 7.2 Chunk Weight

**Standard.** Contains High-tier contract (Protocol E1 resolution) and Medium-tier contracts.

### 7.3 Dependencies

- Chunk 07: Retrained encoder, real-data embeddings, Reality Gate PASS ✅

### 7.4 Contracts

---

#### C08-01: Isolation Forest Baseline

**Risk Tier:** Low
**Implementation Owner:** Gemini
**Objective:** Implement Isolation Forest anomaly detection with explicit missing-data handling.

**Allowed Files:**
- `source/models/baseline/isolation_forest.py` (new)
- `source/tests/test_isolation_forest.py` (new)

**Frozen Files:**
- `source/data/registry/lake_registry.json` (INV-001)
- `invariants.md` (INV-002)

**MISSING-DATA POLICY (applies to ALL baselines):**

Standard `sklearn.ensemble.IsolationForest` and `sklearn.svm.OneClassSVM` do NOT natively handle NaN values. The following policy is mandatory for all baselines:

1. **Per-window completeness check:** If a window has >50% of its time steps with NaN in ANY channel, exclude that window from baseline evaluation. Document the exclusion count.

2. **For windows that pass the completeness check:** Apply training-lake-only median imputation for remaining NaN values. The median is computed exclusively from training-role lake observations (INV-002). Add a binary missingness indicator column per channel (1 if imputed, 0 if observed).

3. **The same imputation and missingness encoding must be applied identically to training and evaluation data.** No evaluation-lake statistics may be used for imputation.

4. **Document the imputation impact:** Report what fraction of feature values were imputed. If >20% of values are imputed, flag this as a limitation.

**Implementation Instructions:**
1. Fit `sklearn.ensemble.IsolationForest(n_estimators=200, contamination='auto')` on training-role lake feature windows.
2. Apply missing-data policy (above) before fitting.
3. Score evaluation windows using `score_samples()` (negate for anomaly score).
4. Apply same EMA smoothing (span=5) as Score-C for fair comparison.
5. Compute AUC-ROC, AUC-PR on synthetic anomalies (Protocol E3).
6. Compute FPR on control lakes (Protocol E2).

**Verification Scripts:**
- Assert: Isolation Forest is fit ONLY on training-role lakes (INV-002)
- Assert: Missing-data policy is applied consistently
- Assert: AUC-ROC computed on same synthetic anomaly set as Score-C
- Assert: Imputation fraction documented

**Definition of Done:**
1. Isolation Forest baseline implemented and tested
2. Missing-data policy applied and documented
3. Results written to `results/evaluation/baseline_isolation_forest.json`

---

#### C08-02: CUSUM Statistical Baseline

**Risk Tier:** Low
**Implementation Owner:** Gemini
**Objective:** Implement CUSUM change detection with explicitly specified continuous scoring mechanism.

**Allowed Files:**
- `source/models/baseline/cusum_baseline.py` (new)
- `source/tests/test_cusum.py` (new)

**Frozen Files:**
- `source/data/registry/lake_registry.json` (INV-001)

**CUSUM SCORING SPECIFICATION (addresses independent critique):**

CUSUM naturally produces a cumulative statistic, not a continuous anomaly score in the same space as Score-C. To compute AUC-ROC, a continuous score must be defined. The following specification is mandatory:

1. For each lake, apply CUSUM to the CH-01 (lake area) time series.
2. Compute reference mean (μ₀) and standard deviation (σ₀) from the first 2 years of the lake's training-period data.
3. For each subsequent time step t, compute:
   - C⁺(t) = max(0, C⁺(t-1) + (x(t) - μ₀)/σ₀ - k)
   - C⁻(t) = max(0, C⁻(t-1) - (x(t) - μ₀)/σ₀ - k)
   - where k = 0.5 (drift parameter)
4. **The continuous anomaly score is:** CUSUM_score(t) = max(C⁺(t), C⁻(t))
5. This produces a continuous, non-negative score that increases during sustained deviations.
6. This score is used directly for AUC-ROC computation. It is NOT thresholded to a binary alarm before AUC computation.
7. The drift parameter k=0.5 must be justified (cite standard CUSUM literature) or subjected to sensitivity analysis (k ∈ {0.25, 0.5, 0.75, 1.0}).

**Implementation Instructions:**
1. Implement CUSUM per the specification above.
2. Apply to synthetic anomalies (Protocol E3) and control lakes (Protocol E2).
3. This is a UNIVARIATE baseline — it uses only lake extent (CH-01), not the full feature matrix. This is intentional: it represents what operational monitoring currently does.

**Verification Scripts:**
- Assert: CUSUM uses only CH-01 (lake area)
- Assert: Continuous score is computed (not binary alarm)
- Assert: AUC-ROC computed on same synthetic anomaly set
- Assert: Drift parameter justified or sensitivity analysis provided

**Definition of Done:**
1. CUSUM baseline implemented with specified continuous scoring
2. Results written to `results/evaluation/baseline_cusum.json`

---

#### C08-03: One-Class SVM Baseline

**Risk Tier:** Low
**Implementation Owner:** Gemini
**Objective:** Implement One-Class SVM with the same missing-data policy as C08-01.

**Allowed Files:**
- `source/models/baseline/one_class_svm.py` (new)
- `source/tests/test_ocsvm.py` (new)

**Frozen Files:**
- `source/data/registry/lake_registry.json` (INV-001)
- `invariants.md` (INV-002)

**Implementation Instructions:**
1. Fit `sklearn.svm.OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)` on training-role lake feature windows.
2. Apply the same missing-data policy as C08-01 (median imputation + missingness indicators).
3. Score evaluation windows using `score_samples()` (negate for anomaly score).
4. Apply same EMA smoothing (span=5).
5. Compute AUC-ROC, AUC-PR, FPR.

**Verification Scripts:**
- Assert: OCSVM fit ONLY on training-role lakes
- Assert: Same missing-data policy as Isolation Forest
- Assert: Same evaluation protocol as Score-C

**Definition of Done:**
1. One-Class SVM baseline implemented and tested
2. Results written to `results/evaluation/baseline_ocsvm.json`

---

#### C08-04: Full Evaluation Re-Run + Cloud-Fraction Stratified Evaluation

**Risk Tier:** Medium
**Implementation Owner:** Gemini
**Objective:** Re-run all evaluation protocols on real-data embeddings. Add cloud-fraction stratified evaluation.

**Allowed Files:**
- `source/scripts/run_evaluation.py`
- `results/evaluation/evaluation_summary_real_data.json` (output)
- `results/evaluation/cloud_stratified_evaluation.json` (output)

**Frozen Files:**
- `source/data/registry/lake_registry.json` (INV-001)
- `invariants.md` (INV-002, INV-007, INV-009, INV-010, INV-011, INV-012)

**Implementation Instructions:**
1. Run Protocols E1–E4 with real-data embeddings and all 7 methods (Score-A, B, C + 4 baselines).
2. Compute full INV-010 metric suite for every scorer and baseline.

**CLOUD-FRACTION STRATIFIED EVALUATION (addresses independent critique):**

The original specification used "monsoon vs. dry season" as a proxy for cloud coverage. This is insufficient. A monsoon window might be cloud-free; a dry-season window might have unusual clouds. The correct approach stratifies by ACTUAL cloud fraction:

3. For each evaluation window, compute the actual cloud fraction (from Sentinel-2 metadata acquired in C07-02).
4. Stratify windows into cloud-fraction bins:

| Cloud Fraction Bin | Description |
|---|---|
| 0–20% | Mostly clear |
| 20–40% | Partially cloudy |
| 40–60% | Mostly cloudy |
| 60–80% | Heavily cloudy |
| >80% | Near-total cloud cover |

5. Compute Score-C, Score-B, and all baselines separately for each bin.
6. Report per-bin AUC-ROC [CI] and AUC-PR [CI].

**"Cloud-robust" is validated if and only if:**
- Score-C AUC-ROC in the >60% cloud bin is within 0.05 of the <20% cloud bin
- SAR channels (CH-05) contribute more relative importance in high-cloud bins
- Optical-only performance degrades significantly in high-cloud bins while Score-C does not

If these conditions are NOT met, "Cloud-Robust" is REMOVED from the title. This is non-negotiable.

**Verification Scripts:**
- Assert: All 7 methods produce metrics on identical evaluation sets
- Assert: Cloud-fraction bins are computed from actual metadata, not season
- Assert: Per-bin metrics reported with sample counts
- Assert: INV-010 metric suite complete for every entry

**Definition of Done:**
1. Full evaluation summary with all 7 methods
2. Cloud-stratified evaluation computed and reported
3. All INV-010 metrics present

---

#### C08-05: Lake-Level Bootstrap Confidence Intervals + Significance Testing

**Risk Tier:** Medium
**Implementation Owner:** Gemini
**Objective:** Compute 95% bootstrap CIs using the correct statistical unit (LAKE, not window). Perform DeLong's test for pairwise comparisons.

**Allowed Files:**
- `source/scripts/run_bootstrap_ci.py` (new)
- `results/evaluation/statistical_significance.json` (output)

**Frozen Files:**
- `invariants.md` (INV-012 — seed for bootstrap)

**STATISTICAL UNIT SPECIFICATION (addresses independent critique — pseudoreplication):**

The project has 4 control lakes and 1 event lake. Each lake produces ~108 temporal windows. Bootstrapping individual windows treats them as independent observations. They are NOT independent — windows from the same lake are highly correlated (overlapping time periods, same physical lake). This is pseudoreplication.

**The correct statistical unit is the LAKE, not the window.**

**Revised statistical protocol:**

1. **Primary analysis:** Compute metrics (AUC-ROC, AUC-PR, FPR, detection rate) using ALL windows from all evaluation lakes. Report these as the primary results.

2. **Lake-level bootstrap for CIs:** Resample LAKES (not windows) with replacement. For each bootstrap iteration (N=2000, seed=2023):
   - Sample 5 lakes (with replacement) from the 5 evaluation lakes
   - Compute metrics on all windows from the sampled lakes
   - This produces a distribution of metrics across lake-resampling iterations
   - Report 95% CI from the 2.5th and 97.5th percentiles

3. **Acknowledge the limitation explicitly:** "With 5 evaluation lakes, lake-level bootstrap CIs will be wide. This reflects the genuine scarcity of documented GLOF events in the HKH region. We report CIs honestly and do not claim narrow precision."

4. **Do NOT bootstrap individual windows.** This is pseudoreplication and would produce artificially narrow CIs.

5. **For pairwise comparisons (Score-C vs. each baseline):** Perform DeLong's test for AUC-ROC comparison. Report p-values. If p > 0.05, the difference is NOT statistically significant — state this honestly.

6. **SVI-003 compliance:** Every comparative claim in the manuscript must reference a CI and/or p-value from this artifact.

**Verification Scripts:**
- Assert: Bootstrap resamples LAKES, not windows
- Assert: 2000 resamples per metric
- Assert: CI reported as [lower, upper] for every AUC-ROC and AUC-PR
- Assert: DeLong's test p-values computed for all pairwise comparisons
- Assert: Seed is pinned (INV-012)

**Definition of Done:**
1. `statistical_significance.json` with CIs and p-values
2. Every AUC-ROC has an associated CI
3. Pairwise significance results documented
4. Limitation of small-N acknowledged

---

#### C08-06: Protocol E1 Resolution — South Lhonak Retrospective (Honest)

**Risk Tier:** HIGH
**Implementation Owner:** Claude (Architect)
**Objective:** Resolve Protocol E1 on real data. Either demonstrate successful pre-event detection OR honestly document the failure and adjust claims.

**Allowed Files:**
- `source/evaluation/protocols/protocol_e1.py`
- `results/evaluation/protocol_e1_real_data.json` (output)
- `project/evolution/decision_log.md`

**Frozen Files:**
- `source/data/registry/lake_registry.json` (INV-001)
- `invariants.md` (INV-009 — event date 2023-10-04)

**Pre-Registered Falsification Criteria (declared BEFORE running the test):**

- **Success criterion:** Score-C exceeds the detection threshold (88th percentile of control scores) for ≥2 consecutive windows within 365 days before 2023-10-04, AND does NOT exceed threshold for >50% of the pre-event period.

- **Failure criterion:** Score-C exceeds threshold for >80% of the pre-event period (same failure mode as v1.0), OR never exceeds threshold at all.

- **If failure criterion is met:** The "Precursor Detection" claim is REMOVED from the title. The paper's contribution is reframed as the evaluation framework + negative result. This is not optional. This is not a "threshold refinement problem." This is a scientific result.

**Implementation Instructions:**
1. Apply the retrained encoder to South Lhonak's full time series (2016–2024).
2. Compute Score-B and Score-C for every window.
3. Apply EMA smoothing (span=5).
4. Apply the threshold derived from control lakes (88th percentile, or the INV-007-compliant threshold from Chunk 07).
5. Measure: first date threshold is exceeded, sustained duration, peak magnitude.
6. Compare against the pre-registered success/failure criteria above.
7. **If adaptive thresholding is attempted:** The adaptive method must be declared BEFORE seeing South Lhonak's scores. E.g., "per-lake z-score with baseline = first 2 years of that lake's own history." Do NOT tune the threshold after seeing the result.

**Stop Condition:** If Score-C flags >80% of the pre-event period (same as v1.0's 2730-day failure), STOP. Do not attempt further threshold tuning. Document the failure honestly. Escalate to title revision.

**Definition of Done:**
1. Protocol E1 result documented with exact dates, scores, threshold
2. Pre-registered criteria evaluated honestly
3. If SUCCESS: lead time reported with CI
4. If FAILURE: documented as limitation, title claim removed, decision logged

---

### 7.5 Chunk 08 Success Criteria

1. ✅ 4 baselines implemented with explicit missing-data policies
2. ✅ CUSUM uses specified continuous scoring mechanism
3. ✅ Full evaluation re-run on real data with all 7 methods
4. ✅ Cloud-fraction stratified evaluation computed (not season-based)
5. ✅ Lake-level bootstrap CIs (not window-level)
6. ✅ DeLong's test for pairwise comparisons
7. ✅ Protocol E1 resolved honestly (success or documented failure)
8. ✅ SVI-002 satisfied (≥3 baselines)
9. ✅ SVI-003 satisfied (lake-level CIs for all comparative claims)

---

## Part 8 — Chunk 09: Ablation Fix + Manuscript Revision

### 8.1 Chunk Objective

Fix the ablation methodology. Justify all hyperparameters. Rewrite the manuscript with honest framing, correct title, full architecture specification, and statistical rigor. Rebuild claim-evidence map. Pass release-check.

### 8.2 Chunk Weight

**Standard.** Contains High-tier contract (manuscript rewrite) and Medium-tier contracts.

### 8.3 Dependencies

- Chunk 08: All evaluation results with real data, baselines, CIs ✅

### 8.4 Contracts

---

#### C09-01: Ablation Methodology Fix + Hyperparameter Justification

**Risk Tier:** Medium
**Implementation Owner:** Gemini
**Objective:** Fix the zero-retraining ablation confound. Justify α and EMA span with sensitivity analysis.

**Allowed Files:**
- `source/scripts/run_ablation.py`
- `results/ablation/ablation_summary_real_data.json` (output)
- `results/ablation/hyperparameter_sensitivity.json` (output)

**Frozen Files:**
- `source/data/registry/lake_registry.json` (INV-001)
- `models/checkpoints/ts_mae_real_data.pt` (Chunk 07 output)

**Ablation Fix — Two Options:**

**Option A (Preferred — Learned Masking Token):**
1. Add a learnable `[MASK_CHANNEL]` embedding to the TS-MAE encoder input.
2. During ablation, replace ablated channel columns with the learned mask token instead of zeros.
3. This requires a brief fine-tuning step (~5 epochs) to learn the mask token. Encoder weights are frozen; only the mask token embedding is learned.
4. Re-run all ablation configurations with the mask token.

**Option B (Fallback — Sensitivity Analysis):**
1. Keep the zero-masking approach.
2. Run a sensitivity analysis: compare ablation results with zero-masking vs. mean-imputation masking vs. Gaussian noise masking.
3. If all three produce the same channel ranking, the confound does not change the conclusion. Document this.
4. If rankings differ, report all three and state the limitation.

**Hyperparameter Justification (D-018 compliance):**

- **For Score-C's α:** Run sensitivity sweep α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}. Report AUC-ROC for each. Justify the chosen α. If α=0.5 is not optimal, use the optimal value and document the change.

- **For EMA span:** Run sensitivity sweep span ∈ {3, 5, 7, 10}. Report impact on detection metrics. Justify the chosen span.

**Verification Scripts:**
- Assert: Ablation produces correct number of configurations for 13 channels
- Assert: Channel ranking is documented
- Assert: α sensitivity sweep results exist
- Assert: EMA span sensitivity sweep results exist

**Definition of Done:**
1. Ablation re-run with corrected methodology
2. Channel contribution ranking documented
3. α and EMA span justified with sensitivity analysis
4. Results in `ablation_summary_real_data.json` and `hyperparameter_sensitivity.json`

---

#### C09-02: Manuscript Rewrite

**Risk Tier:** HIGH
**Implementation Owner:** Claude (Architect)
**Objective:** Rewrite the manuscript with honest framing, corrected title, full architecture specification, statistical rigor, and appropriate claims.

**Allowed Files:**
- `sentinel_gl_manuscript.md`
- `project/evolution/decision_log.md`

**Frozen Files:**
- `invariants.md` (all INV-XXX)
- All `results/` files (evidence is frozen — manuscript must match evidence, not vice versa)

**Mandatory Manuscript Changes:**

**1. Title Revision (SVI-004):**
- If Protocol E1 SUCCEEDED: "Self-Supervised Multi-Sensor Anomaly Scoring for Glacial Lake Precursor Monitoring: A Temporal Masked Autoencoder Approach"
- If Protocol E1 FAILED: "Self-Supervised Multi-Sensor Anomaly Scoring for Glacial Lake Dynamics: Framework, Evaluation Protocol, and Negative Results from the South Lhonak GLOF"
- **REMOVE "Cloud-Robust" UNLESS cloud-stratified evaluation (C08-04) demonstrates maintained performance at >60% cloud fraction.**
- **REMOVE "Precursor Detection" UNLESS Protocol E1 succeeded with pre-registered criteria.**

**2. Abstract Rewrite:**
- State the data type honestly: "real Sentinel-1/2 observations with documented cloud gaps"
- Report metrics with CIs: "AUC-ROC of X.XX [95% CI: X.XX–X.XX]"
- Name all baselines explicitly
- State the Protocol E1 outcome honestly
- State the statistical unit (lake-level)

**3. New Section: Architecture Specification (addresses C4):**
Add §3.3.1 "TS-MAE Architecture Details" containing:
- Encoder: number of transformer layers, hidden dimension, attention heads, parameter count
- Decoder: architecture
- Masking strategy: random contiguous vs. random sparse, ratio
- Input shape: [T=180, C=13]
- PCA dimensionality: 16 components (justify why 16)
- k-NN parameters: k=5 (justify or sensitivity analysis)
- Total parameter count
- Training hyperparameters: learning rate, batch size, optimizer, epochs, warmup

**4. New Section: Statistical Methods (addresses M1):**
Add §4.1.1 "Statistical Evaluation Protocol":
- Lake-level bootstrap methodology (N=2000, percentile method)
- Explicit statement: "The statistical unit is the lake, not the temporal window. Windows from the same lake are correlated and are not treated as independent observations."
- DeLong's test for AUC-ROC comparison
- Explicit statement of sample sizes and their limitations
- "With 4 control lakes and 1 event lake, statistical power is limited. We report confidence intervals but acknowledge that formal significance claims require larger evaluation sets."

**5. New Section: Missing-Data Policy:**
Document the baseline missing-data policy (median imputation + missingness indicators) and the imputation fraction.

**6. Revised Results Section:**
- Table 1: All 7 methods with AUC-ROC [CI], AUC-PR [CI], detection rate, FPR
- Table 2: Cloud-fraction stratified results
- Table 3: Ablation with corrected methodology
- Protocol E1 result: honest, with pre-registered criteria stated
- Threshold sensitivity: α sweep, EMA span sweep

**7. Revised Discussion:**
- Score-A null result: honest, with C53 acknowledgment (Score-A null ≠ architecture failure)
- Protocol E1: honest outcome
- Ablation confound: acknowledged and addressed
- CH-07 removal: documented with rationale
- Limitations: expanded, including small-N, single event, pseudoreplication awareness

**8. Factual Corrections (M3):**
- Fatalities: ~55 deaths, 70–74 missing (NOT "7 fatalities")
- Coordinates: use project_knowledge.md values (27.92°N, 88.18°E) consistently
- Volume: 14.7 million m³ of moraine collapsed (NOT "30 million cubic meters of water")

**9. Remove Local Paths:**
- No `/Users/adi/...` anywhere
- All paths relative to repository root

**Verification Scripts:**
- Run `verify_claim_evidence.py` — all claims must pass
- Run `gatekeeper.py release-check` — no local paths, no factual mismatches
- Manual check: every title adjective has a corresponding experimental validation

**Definition of Done:**
1. Manuscript rewritten with all 9 mandatory changes
2. Title consistent with actual results (SVI-004)
3. Architecture specification complete (C4 addressed)
4. Statistical methods documented with correct unit (M1 addressed)
5. All factual errors corrected (M3 addressed)
6. No local paths (m6 addressed)
7. Claim-evidence map rebuilt and verified

---

#### C09-03: Reproducibility Guide Revision

**Risk Tier:** Low
**Implementation Owner:** Gemini
**Objective:** Update REPRODUCIBILITY.md to reflect real data pipeline, remove local paths, add baseline reproduction steps.

**Allowed Files:**
- `REPRODUCIBILITY.md`

**Implementation Instructions:**
1. Remove all `/Users/adi/...` paths. Use relative paths only.
2. Add steps for real data acquisition.
3. Add steps for baseline reproduction (Isolation Forest, CUSUM, OCSVM).
4. Add steps for lake-level bootstrap CI computation.
5. Update checkpoint reference to `ts_mae_real_data.pt`.
6. Add "Known Limitations" section.
7. Update channel count from 15 to 13 (CH-06 and CH-07 excluded).

**Definition of Done:**
1. No local paths
2. All steps updated for real data pipeline
3. Baseline reproduction steps included
4. `gatekeeper.py release-check` passes

---

#### C09-04: Release Artifact Scan + Final Verification

**Risk Tier:** Medium
**Implementation Owner:** Gemini
**Objective:** Run the full release-check pipeline. Verify claim-evidence traceability. Final Gatekeeper check.

**Allowed Files:**
- `results/release_check/release_check_report.md` (output)

**Implementation Instructions:**
1. Run `gatekeeper.py release-check` on all outbound artifacts.
2. Verify: no local paths, no factual mismatches against `project_knowledge.md`.
3. Run `verify_claim_evidence.py` on the rebuilt claim-evidence map.
4. Run full test suite (`pytest source/tests/ -v`).
5. Run `gatekeeper.py check` for repository integrity.
6. Write release check report.

**Definition of Done:**
1. Release check passes
2. Claim-evidence verification: all claims pass
3. Full test suite: all tests pass
4. Repository integrity: clean
5. Release check report written

---

### 8.5 Chunk 09 Success Criteria

1. ✅ Ablation methodology fixed
2. ✅ Hyperparameters justified (α, EMA span)
3. ✅ Manuscript rewritten with honest framing
4. ✅ Title matches actual results
5. ✅ Architecture specification complete
6. ✅ Statistical methods documented with correct unit
7. ✅ All factual errors corrected
8. ✅ Release check passes
9. ✅ Claim-evidence map verified

---

## Part 9 — Stop Conditions and Architecture Amendment Triggers

| Trigger | Condition | Action |
|---|---|---|
| Reality Gate FAIL | Any of 5 checks returns FAIL | Halt. Diagnose data pipeline. Fix before proceeding. |
| Encoder training divergence | Loss produces NaN or increases catastrophically | Halt. Diagnose. Consider learning rate reduction. |
| Score-B null result on real data | Score-B AUC-ROC ≤ 0.55 on synthetic anomalies | C53 Stop Condition. Architecture may be unsuitable. Halt and assess. |
| All baselines outperform Score-C | Best baseline AUC-ROC > Score-C AUC-ROC + 0.05 | Major reassessment. TS-MAE contribution claim invalid. Reframe paper. |
| Protocol E1 total failure | >80% of pre-event period flagged | Remove "Precursor Detection" claim. Reframe paper. Do NOT tune threshold further. |
| Data leakage detected | Any evaluation lake ID in training batch | Immediate halt. C16 violation. Full audit. |
| Compute budget exceeded | Training > 72 GPU-hours | C13 disclosure. Simplify architecture. |
| Reality Gate WARNING | Any check returns WARNING | Human reviews before proceeding. Document the anomaly. |

---

## Part 10 — Risk Register and Contingency Plans

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Real data produces WORSE results than simulated | Medium-High | High | Pre-register expectations. Report honestly. InSAR negative result + evaluation protocol remain publishable. |
| Protocol E1 fails again on real data | Medium | High | Pre-registered criteria. Honest reframing. Paper still publishable as framework + negative result. |
| GEE quota limits prevent full acquisition | Medium | Medium | Batch acquisitions. Use ASF DAAC for Sentinel-1. Document gaps honestly. |
| Baselines outperform TS-MAE | Low-Medium | High | Report honestly. Contribution becomes evaluation framework + comparison. |
| Cloud gaps too severe for some lakes | Low-Medium | Medium | Document exclusions. Reduce lake count honestly. |
| Encoder doesn't converge on real data | Low | High | Try learning rate adjustment. If still fails, Architecture Amendment to LSTM-AE. |
| Statistical significance not achieved | Medium-High | Medium | Report CIs honestly. Acknowledge small-N. Do not claim significance without p < 0.05. |
| Lake-level bootstrap produces very wide CIs | High | Medium | Expected with 5 lakes. Report honestly. This is a genuine limitation of the domain. |

---

## Part 11 — Release-Readiness Criteria (Replaces "10/10" Target)

**CRITICAL:** The Factory optimizes for scientific correctness, not reviewer scores. A correct paper that gets rejected is better than an incorrect paper that gets accepted. The following criteria define what "ready for submission" means. They do NOT guarantee acceptance.

| # | Criterion | How It's Achieved | Verified By |
|---|---|---|---|
| 1 | All data is real satellite observations with documented gaps | Chunk 07, Reality Gate | SVI-001, SVI-006 |
| 2 | ≥3 competitive baselines implemented and compared | Chunk 08 | SVI-002 |
| 3 | Statistical CIs computed with correct statistical unit (lake-level) | Chunk 08 | SVI-003 |
| 4 | Every title adjective validated by a specific experiment | Chunk 09 | SVI-004 |
| 5 | Protocol E1 resolved honestly (success or documented failure) | Chunk 08 | Pre-registered criteria |
| 6 | Ablation methodology sound (no zero-masking confound) | Chunk 09 | M5 addressed |
| 7 | All hyperparameters justified | Chunk 09 | D-018 |
| 8 | Manuscript matches evidence exactly | Chunk 09 | INV-013 |
| 9 | No local paths, no factual inconsistencies in release artifacts | Chunk 09 | Release check |
| 10 | Limitations section honestly discloses small-N, single-event, and data processing constraints | Chunk 09 | C36, C39 |
| 11 | Reproducibility guide accurate and complete | Chunk 09 | C27 |
| 12 | Cloud-robustness tested with cloud-fraction stratification (if claimed) | Chunk 08 | SVI-004 |
| 13 | CH-07 removed (GRD cannot produce coherence) | Chunk 07 | Scientific validity |
| 14 | Missing-data policy for baselines documented | Chunk 08 | Methodological rigor |
| 15 | CUSUM scoring mechanism specified | Chunk 08 | Methodological rigor |

**If all 15 criteria are met, the manuscript is ready for submission.** Whether it is accepted is outside the Factory's control and is not a measure of the project's success.

---

## Part 12 — What This Document Does NOT Guarantee

Per the Factory's own honesty requirements (C01, C39):

1. **This does not guarantee acceptance at TGRS.** Even a methodologically perfect paper can be rejected for venue fit, reviewer preferences, or competing submissions.

2. **This does not guarantee Protocol E1 will succeed on real data.** The pre-event signal may genuinely not be detectable. If so, the paper is still publishable — as a different paper.

3. **This does not guarantee the TS-MAE outperforms baselines.** Isolation Forest on the same features might be equally good or better. If so, the contribution is the evaluation framework.

4. **This does not guarantee statistical significance.** With 4 control lakes and 1 event lake, statistical power is inherently limited. CIs will be wide. This is honest and must be reported as such.

5. **This does not eliminate all methodological concerns.** The small-N problem cannot be manufactured away. The single-event constraint is a fundamental property of GLOF rarity.

6. **This does not replace domain expertise.** The Architect should consult glaciology literature for physical plausibility of results.

7. **This does not guarantee the MAR performer is truly independent.** A different Claude session still shares training biases. Only a genuinely different model family or a human closes that gap.

---

## Part 13 — Execution Sequence Summary (Gate-Driven, Not Calendar-Driven)

**CRITICAL REVISION:** The original document estimated "9–13 weeks." This is replaced with gate-driven progression. The gates matter more than the calendar. If a Reality Gate check fails or a Stop Condition triggers, the timeline is irrelevant — fix the problem first.

```
Chunk 07 (gate-driven)
├── C07-01: Sentinel-1 acquisition [Medium, Gemini]
├── C07-02: Sentinel-2 acquisition + cloud masking [Medium, Gemini]
├── C07-03: Auxiliary channels + CH-07 REMOVAL [Medium, Gemini]
├── C07-04: Feature assembly + REALITY GATE (three-state) [HIGH, Claude]
│   └── GATE: FAIL → STOP. WARNING → Human review. PASS → proceed.
└── C07-05: Encoder retraining + embedding extraction [HIGH, Claude]
    └── GATE: Score-B AUC-ROC ≤ 0.55 → STOP (C53)

Chunk 08 (gate-driven)
├── C08-01: Isolation Forest baseline [Low, Gemini]
├── C08-02: CUSUM baseline (continuous score specified) [Low, Gemini]
├── C08-03: One-Class SVM baseline [Low, Gemini]
├── C08-04: Full evaluation + cloud-fraction stratification [Medium, Gemini]
├── C08-05: Lake-level bootstrap CI + significance testing [Medium, Gemini]
└── C08-06: Protocol E1 resolution [HIGH, Claude]
    └── GATE: >80% pre-event flagged → Remove "Precursor Detection" claim

Chunk 09 (gate-driven)
├── C09-01: Ablation fix + hyperparameter justification [Medium, Gemini]
├── C09-02: Manuscript rewrite [HIGH, Claude]
├── C09-03: Reproducibility guide revision [Low, Gemini]
└── C09-04: Release check + final verification [Medium, Gemini]
```

**Total: 15 contracts across 3 chunks. Progression is gate-driven, not calendar-driven.**

---

## Part 14 — Final Instruction to the Architect

You are planning Chunks 07–09 of a project that received a 4/10 IEEE review. The engineering was flawless. The science was insufficient. Your job now is to fix the science without breaking the engineering.

The Factory's own philosophy (EP-005: Evidence Drives Evolution) brought you here. The GLOF project is the evidence. The IEEE review is the diagnosis. The independent methodological critique is the second opinion. Chunks 07–09 are the treatment.

Do not rush. Do not cut corners. Do not reframe failures as successes. Do not optimize for a reviewer score. Optimize for scientific correctness.

The Constitution's highest priority is Integrity. A paper that honestly reports "we built a framework, tested it on real data, and here's what worked and what didn't" is infinitely more valuable than a paper that overclaims and gets retracted.

The InSAR negative result is real. The evaluation protocol is well-designed. The engineering is excellent. These are genuine contributions. Build on them honestly.

If Protocol E1 fails again, that is not a project failure. That is a scientific result. Report it. The paper is still publishable. The Factory's C36 (Preserve Null Results) exists for exactly this moment.

If Protocol E1 succeeds, that is extraordinary. Report it with the pre-registered threshold, the lake-level bootstrap CI, and the honest acknowledgment that N=1 event limits generalizability.

Either way, the paper that emerges from Chunk 09 will be honest, rigorous, reproducible, and worthy of submission to IEEE TGRS. Whether it is accepted is not yours to control. Whether it is correct is.

---

**END OF DOCUMENT**

**Next step for the Architect:** Read this document in full. Populate `venue_requirements.md` from Part 4. Then plan Chunk 07 contracts per Part 6. Drop all artifacts into `DROP_HERE/` and tell Gemini to check the mailbox.


# ADDENDUM — One Gap Found, Not Covered By The v1.3.4 Document

This addendum supplements the "sentinel-gl: Architect Continuation Document — Chunks 07–09,
IEEE TGRS Major Revision | Factory v1.3.4" document (Revision 3.0) you supplied. That document
was checked directly against your real Factory files and founding artifacts and is accurate:

- Constitution C06 (Stop Conditions Are Absolute) is real.
- EP-001 through EP-006 are all real, exactly as cited.
- D-001 through D-009 exist in `dynamic_rules.md`, all PROPOSED (the document says
  "D-001–D-010" — off by one, harmless, doesn't change any load-bearing claim).
- INV-001 through INV-013 are the real current state of `invariants.md`, exactly as the
  document assumes before adding INV-014–INV-020.
- `gatekeeper.py check` is a real subcommand (confirmed directly in `gatekeeper.py`). The
  document correctly does NOT invoke a fabricated `release-check` subcommand — that was a
  real error in an earlier draft you supplied, and this version fixed it.
- No "Factory v1.4.0," no "Reality Gate" as a named pipeline phase (it's correctly
  implemented as INV-014's Verification Scripts inside a real contract, C07-04), no
  fabricated Constitution rule numbers (no C51–C53).

Adopt that document as-is. Add the following one contract to Chunk 07, run before C07-01.

---

## C07-00: Invariant Correction — INV-011 References an Excluded Channel

**Risk Tier:** Low
**Implementation Owner:** Claude (Architect)
**Objective:** `invariants.md` INV-011's synthetic anomaly type 3 currently reads: "A linear
displacement rate of 50 mm/year in **CH-06** (InSAR deformation)... Applied for 180 days."
But `methods.md` §3.2 confirms CH-06 was already excluded from the pipeline following the
InSAR infeasibility finding (Decision 001), and `experiments.md` §3.4 confirms the actual
Protocol E3 implementation already substitutes a different perturbation: "SAR backscatter
step +3 dB... substituted for infeasible InSAR (Decision 003)." The frozen invariant was
never updated to match what Decision 003 already did in practice — this is exactly the kind
of drift `invariants.md`'s own header warns against ("things that must never become false
for the project's life").

**Dependencies:** None.

**Allowed Files:** `invariants.md`, `project/evolution/decision_log.md`

**Implementation Instructions:**
1. Confirm the exact magnitude and channel used by the actual Chunk 05 Decision-003
   implementation (do not guess the +3 dB figure without checking the actual synthetic
   injection code/config that produced it).
2. Update INV-011's type-3 definition to match, in the same style INV-004's own "these
   values are design choices... may be tuned... documented in decision_log.md" note already
   uses in this document.
3. Add a dated revision note directly beneath the corrected text. Per C30, the prior (CH-06)
   text must remain visible in the revision history — this is a correction, not a silent edit.
4. Log the correction in `decision_log.md`, referencing Decision 003 and this contract.

**Verification Scripts:**
- Assert: INV-011 no longer references CH-06
- Assert: INV-011's stated type-3 perturbation matches the actual Protocol E3 config used
  in the original evaluation run
- Assert: a dated revision note exists in `invariants.md` for this change

**Definition of Done:** INV-011 matches what the project's own Decision 003 already
implemented. Nothing about the actual synthetic anomaly injection code changes — this
contract only brings the frozen invariant's text in line with reality.

**Traces To:** INV-011, C30

---

Everything else in the document you supplied stands as written — no other changes needed.