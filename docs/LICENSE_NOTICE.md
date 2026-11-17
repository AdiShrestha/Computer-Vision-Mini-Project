# Licences and third-party terms

| Material | Licence / terms |
|---|---|
| Source code in `source/` and `tools/`, `Makefile`, `pyproject.toml` | MIT (see `LICENSE`) |
| Documentation, protocol files, result packet, figures and manuscript text written for this project (`docs/`) | Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/ |
| Derived data deposit (ROI arrays, receipts, zones, checkpoints, predictions, replay arrays) | CC BY 4.0, with the Copernicus notice below |

## Copernicus Sentinel-2 data

All optical inputs are windows read from Copernicus Sentinel-2 Collection-1 Level-2A products distributed through the Earth Search STAC API (Element 84, Registry of Open Data on AWS). The Copernicus Sentinel data legal notice (https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice, checked 2026-11-17) grants free, full and open access for reproduction, distribution, communication to the public and adaptation, provided the source is acknowledged. Every figure and deposited array carries the notice:

> Contains modified Copernicus Sentinel data [2018–2023].

## Hi-MAG glacial lake inventory

Lake outlines GL088195E27912N (South Lhonak) and GL088815E27990N (Khangchung Tsho) are taken from the 2017 layer of the Hi-MAG database: Chen F, Zhang M, Guo H, Allen S, Kargel JS, Haritashya UK, Watson CS. Annual 30-meter Dataset for Glacial Lakes in High Mountain Asia from 2008 to 2017. Zenodo; 2020. doi:10.5281/zenodo.4275164 (CC BY 4.0), described in Earth Syst Sci Data 13:741–766 (2021), doi:10.5194/essd-13-741-2021. The derived zone rasters are adaptations under CC BY 4.0 with this attribution.

## History

Earlier snapshots of this repository carried a proprietary licence and, in an older commit, an Apache-2.0 notice. Both are superseded by this file and `LICENSE` from the release that accompanies the PLOS ONE submission.
