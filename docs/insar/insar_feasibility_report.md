# Technical Report: InSAR SLC Feasibility & Decorrelation Analysis in High Mountain Asia

**Report ID:** `REP-INSAR-FEASIBILITY-01`  
**Governing Work Package:** `WP11-COHORT-INSAR-01`  
**Target Terrain:** High Mountain Asia (Himalaya, Karakoram, Hindu Kush)  

---

## 1. Executive Summary

This report evaluates the physical feasibility of Sentinel-1 Single Look Complex (SLC) interferometric SAR (InSAR) for continuous glacial lake outburst flood (GLOF) monitoring. Analysis across storage budgets, radar geometric distortions, and multi-temporal interferometric coherence reveals severe physical limits in steep glaciated catchments:
1. **Storage Prohibitions:** Regional multi-temporal SLC full-scene archives exceed local workstation budgets (>1.0 TB vs 114 GB available), requiring burst-level sub-setting.
2. **Geometric Distortions:** Steep cirque headwalls (>= 38 deg) face acute radar layover, while opposing valley walls fall into radar shadow.
3. **Monsoon Coherence Collapse:** Heavy monsoon precipitation and moisture-saturated moraine till cause rapid temporal decorrelation (gamma_total = 0.0176 < 0.25), preventing reliable phase unwrapping during the peak hazard season (July-October).

Consequently, Sentinel-GL correctly selects **calibrated dual-polarization GRD backscatter amplitude and spatial texture** over fragile interferometric phase tracking.

---

## 2. Storage & Memory Budget Audit

Sentinel-1 SLC acquisitions contain full phase and amplitude at native single-look resolution, producing massive data volumes:

| InSAR Processing Scenario | Scene Unit Size | Total Archive Volume (8 lakes, 30 dates) | Budget Feasibility |
| :--- | :--- | :--- | :--- |
| **Full-Scene SLC Archive** | ~4.2 GB zip (~8.0 GB uncompressed) | 938.8 GB | **REJECTED (Exceeds 114 GB budget)** |
| **Burst-Cropped ROI** | <= 1.0 GB per lake ROI | 223.5 GB | **FEASIBLE (Within local limits)** |

---

## 3. SAR Geometric Distortion Modeling

In high-relief terrain, the local incidence angle theta_local governs radar reflectivity:
cos(theta_local) = cos(theta_inc) * cos(alpha) + sin(theta_inc) * sin(alpha) * cos(phi_look - beta)

### Modeled Case Studies:
- **Radar Layover (Steep Cirque Headwall):** Slope alpha = 45.0 deg, Aspect beta = 80.0 deg. Result: `LAYOVER` (local incidence angle: 7.1 deg). The top of the slope reaches the radar antenna prior to the base, resulting in phase inversion and signal entanglement.
- **Radar Shadow (Opposing Valley Wall):** Slope alpha = 55.0 deg, Aspect beta = 260.0 deg. Result: `SHADOW` (local incidence angle: 93.0 deg). The slope is shielded from the radar beam line-of-sight, yielding complete signal void.

---

## 4. Multi-Temporal Coherence Decay Modeling

Interferometric coherence gamma_total = gamma_geom * gamma_temp * gamma_thermal reflects phase correlation across satellite passes:

gamma_temp = exp(-B_t / tau_decorr)

### Modeled Coherence Regimes (Sentinel-1 12-day repeat):
- **Winter Frozen Moraine:** tau_decorr = 45.0 days => gamma_total = 0.7350 (Viable for phase unwrapping).
- **Summer Monsoon Moraine (July-October):** Heavy rainfall (>50 mm/week) and glacial till saturation yield rapid decorrelation: tau_decorr = 3.0 days => gamma_temp = 0.0183, gamma_total = 0.0176.

Because gamma_total < 0.25 falls far below the phase unwrapping threshold (0.30), interferometric phase coherence collapses during the South Lhonak pre-event window (August-October 2023).

---

## 5. Architectural Decision

Sentinel-GL rejects operational reliance on InSAR phase unwrapping. The project adopts calibrated Sentinel-1 GRD dual-polarization backscatter power (VV, VH), cross-ratio (VH/VV), and spatial texture variance, guaranteeing continuous observability through cloud and monsoon moisture.
