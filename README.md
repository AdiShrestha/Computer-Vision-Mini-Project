# Sentinel-GL

Code for the PLOS ONE submission *"Masked transformer reconstruction versus seasonal climatology for Sentinel-2 monitoring of two Sikkim glacial lakes around the 2023 South Lhonak outburst: a retrospective case study"*.

The project builds a source-bound Sentinel-2 Level-2A patch time series (141 scenes, 2018–2023) for South Lhonak and Khangchung Tsho, trains a small masked spatiotemporal transformer, and compares it on identical hidden entries with parameter-free seasonal references, context ablations, classical comparators and a constructed positive control. It is a retrospective, exploratory case study with one event; it does **not** establish detection timing, false-alert rates or warning value. Negative and inconclusive results are reported as such.

## Layout

| Path | Contents |
|---|---|
| `source/sentinel_gl/` | corpus loader (`sr1_corpus`), features, model and trainer, scoring, geometry (Hi-MAG zones), comparators, statistics, positive control, acquisition helpers (`ingestion/`) |
| `source/runners/` | acquisition (`run_sr1_multiyear.py`, `run_sr1_recentre_south_lhonak.py`) and the analysis stages (`run_sr1_plos_analysis.py`) |
| `source/tests/` | tests on constructed inputs, including regression tests for the defects corrected in amendment PLOS-001 |
| `tools/` | independent replay, result packet, figures, reference checks, manuscript builder, verifier, deposit builder |
| `docs/protocol/` | parent protocol, amendment PLOS-001 and its pre-run freeze record, post-freeze changes, pre-fix regression log |
| `docs/results/sr1_plos_packet.json` | every reported number, recomputed from retained rows |
| `docs/manuscript/references_plos.json` | references checked against Crossref, DataCite or arXiv |
| `docs/REPRODUCE.md`, `docs/DATA_AVAILABILITY.md`, `docs/AI_USE.md`, `docs/LICENSE_NOTICE.md` | reproduction, data, AI-use disclosure, licences |

## Quick start

```sh
python3.12 -m venv .venv && . .venv/bin/activate
make install   # hash-locked dependencies
make test      # no data needed
```

With the data deposit unzipped at the repository root, `make all-from-deposit` replays the checkpoints, recomputes the result packet, redraws the figures, rebuilds the manuscript and verifies it against the data. See `docs/REPRODUCE.md` for re-training and re-acquisition.

## History

Earlier versions of this repository held a fixture-based prototype, a 2023-only analysis and an ICT-CEEL 2027 draft. A 2026-11-17 review found that the trainer behind those drafts hid every observed patch of every training scene, among other defects; amendment PLOS-001 (`docs/protocol/`) corrects them, and none of the earlier results is used. The earlier material remains in git history.

## Archives

Data deposit: doi:DATA_DOI_PENDING. Code archive: doi:CODE_DOI_PENDING.

## Licence

Code: MIT (`LICENSE`). Documentation, protocol, results and figures: CC BY 4.0. Contains modified Copernicus Sentinel data [2018–2023]. See `docs/LICENSE_NOTICE.md`.
