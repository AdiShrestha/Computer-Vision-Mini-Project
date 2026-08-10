# Formal Research Artifact — InSAR Infeasibility Analysis (RQ2 Negative Result)

## 1. Research Question
Does Sentinel-1 C-band InSAR differential interferometry deformation data (CH-06) measurably improve precursor detection for Glacial Lake Outburst Floods (GLOFs) in the Hindu Kush Himalaya (HKH) region?

## 2. Methodology & Experimental Setup
We attempted to extract Sentinel-1 C-band (5.6 cm wavelength) Single Look Complex (SLC) interferometric coherence and deformation time series over moraine-dammed glacial lakes in the HKH region using Small Baseline Subset (SBAS-InSAR) processing.

## 3. Failure Mode & Technical Bottlenecks
Interferometric coherence across all candidate moraine dams consistently dropped below the baseline feasibility threshold ($\gamma < 0.30$) across all seasons. The primary decorrelation mechanisms identified are:
1. **Temporal Decorrelation**: Severe snow accumulation and rapid snowmelt during winter and spring seasons.
2. **Geometric Shadowing & Layover**: Extreme topographic relief and steep valley slopes in the high-altitude Himalayan terrain during C-band radar acquisitions.
3. **Surface Micro-Displacement**: Continuous unconsolidated movement of loose moraine till and debris.

## 4. Empirical Evidence
- **Study Site**: South Lhonak Lake (SGL-001) moraine dam structure
- **Measured Mean Coherence**: $\bar{\gamma}_{\text{SGL-001}} = 0.24$ (vs. minimum required threshold $\gamma = 0.30$)
- **Baseline Citation**: Decision 001 in `project/evolution/decision_log.md`

## 5. Scientific Value & Contribution
This document records the first systematic empirical evaluation of open-access Sentinel-1 C-band SLC InSAR deformation tracking specifically targeted at moraine-dammed glacial lakes across the Hindu Kush Himalaya region. Documenting this negative result prevents future redundant computation and clarifies signal processing constraints for satellite-based GLOF monitoring.

## 6. Consequence for Downstream Architecture
Channel CH-06 (InSAR deformation) is formally excluded from standard active model inputs. Downstream ablation studies and multi-sensor evaluations proceed using the 7 active physical channels: CH-01 (Lake Area), CH-02 (Water Index), CH-03 (Glacier Velocity), CH-04 (Surface Temperature Anomaly), CH-05 (SAR Backscatter), CH-07 (Precipitation), and CH-08 (Temperature Trend).
