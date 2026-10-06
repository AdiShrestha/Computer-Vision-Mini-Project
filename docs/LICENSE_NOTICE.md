# Data Provider Licenses, Terms of Use & Legal Notices

**Project:** Sentinel-GL Research Collaboration  
**Document ID:** `DOC-LICENSE-NOTICE-01`  
**Version:** 1.0.0  
**Effective Date:** October 2026  

---

## 1. Overview

The Sentinel-GL research framework integrates data from multiple international Earth observation agencies, climate reanalysis centers, and cryospheric scientific consortia. This document sets forth the applicable terms of use, legal notices, and attribution statements required by each contributing data provider.

---

## 2. European Space Agency (ESA) — Copernicus Sentinel Data

### 2.1 Legal Notice on Copernicus Sentinel Data
Sentinel-1 Synthetic Aperture Radar (SAR) and Sentinel-2 Multi-Spectral Instrument (MSI) observations are provided by the European Union's Copernicus Programme via the European Space Agency (ESA) and the Copernicus Data Space Ecosystem (CDSE).

Pursuant to the **Legal Notice on the use of Copernicus Sentinel Data and Service Information**:
1. Users are granted free, full, and open access to Copernicus Sentinel data and information in accordance with EU Regulation (EU) No 377/2014 and Commission Delegated Regulation (EU) No 1159/2013.
2. Access and use of Copernicus Sentinel data are subject to the user's acceptance of the following conditions:
   - When distributing or communicating Copernicus Sentinel data and information to the public, the user shall inform the public of the source by stating the attribution text below.
   - Neither the European Union nor ESA can be held responsible for any use which may be made of Copernicus Sentinel data and information.

### 2.2 Mandatory Attribution Statement
> **"Contains modified Copernicus Sentinel data [2016–2024], processed by the Sentinel-GL Research Collaboration."**

---

## 3. European Centre for Medium-Range Weather Forecasts (ECMWF) — ERA5 Reanalysis

### 3.1 Copernicus Climate Change Service (C3S) Licence
Atmospheric reanalysis variables (2m temperature, total precipitation) are derived from the ERA5 single-levels reanalysis dataset provided by the European Centre for Medium-Range Weather Forecasts (ECMWF) through the Copernicus Climate Change Service (C3S).

Pursuant to the **Licence to Use Copernicus Products (C3S / ECMWF)**:
1. The user is granted a free, non-exclusive, worldwide, royalty-free license to access, reproduce, adapt, and distribute the Copernicus Products.
2. When publishing, distributing, or displaying information derived from ERA5 data, the user must display the mandatory attribution statement.
3. Disclaimer: Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

### 3.2 Mandatory Attribution Statement
> **"Generated using Copernicus Climate Change Service information [2016–2024]. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains."**

---

## 4. Randolph Glacier Inventory (RGI) & GLIMS Database

### 4.1 Randolph Glacier Inventory (RGI 6.0 / 7.0)
Glacier boundaries, outlines, and regional classifications for High Mountain Asia (Region 14: South Asia West; Region 15: South Asia East) are sourced from the **Randolph Glacier Inventory (RGI 6.0)**, maintained by the Global Land Ice Measurements from Space (GLIMS) initiative and the Working Group on the Randolph Glacier Inventory of the International Association of Cryospheric Sciences (IACS).

### 4.2 License and Attribution
- RGI data are distributed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.
- Mandatory Citation & Attribution:
  - RGI Consortium (2017). Randolph Glacier Inventory – A Dataset of Global Glacier Outlines: Version 6.0: Technical Report, Global Land Ice Measurements from Space, Colorado, USA. Digital Media. https://doi.org/10.7265/N5-RGI-60
  - GLIMS and NSIDC (National Snow and Ice Data Center), Boulder, CO.

> **"Glacier boundary outlines and morphological metadata courtesy of the Randolph Glacier Inventory (RGI 6.0) Consortium and the Global Land Ice Measurements from Space (GLIMS) database."**

---

## 5. Copernicus Digital Elevation Model (DEM GLO-30)

Global 30-meter elevation tiles and topographical slopes/aspects are derived from the **Copernicus DEM GLO-30**, distributed by Airbus Defence and Space and the European Space Agency under the Copernicus Open Access License.

> **"Topographic slope, aspect, and elevation profiles derived from the 30-meter Copernicus DEM (GLO-30), European Space Agency."**

---

## 6. Sentinel-GL Codebase License

The software, pipeline scripts, evaluation engines, and verification tools in this repository are released under the **Apache License, Version 2.0** (or open scientific research equivalent):

```
Copyright 2026 Sentinel-GL Research Collaboration

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### Scientific Integrity & Non-Operational Warranty
The code is provided for reproducible scientific benchmarking and research evaluation only. Under the terms of the Apache 2.0 license, the software is provided on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. The software has **not** been certified as an operational life-safety or early warning civil defense system.
