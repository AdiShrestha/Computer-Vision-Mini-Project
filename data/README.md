# Current 2023 source-bound expansion

The canonical expanded observations are separately recorded in `data/derived/expansion_2023/observations.v2.json`, with provider metadata, native XMLs, and transfer receipts under `data/raw/expansion_2023/`. SHA-256 bindings identify the staged ROI arrays. Optical source COGs are read through hashed HTTP ranges: a range hash does not establish the full remote COG hash. Native XML hashes are checked against declared STAC SHA-256 multihashes where supplied. Missing pixels remain masked, never interpolated.

The separate SR-1 multi-year optical execution is recorded in `data/derived/sr1_multiyear/attempt_006/observations.sr1.json`. It retains exact provider/native timestamps and receipt bindings for 141 observations across 2018–2023. The source and score-replay audit is `project/remediation/2026-10-26/sr1_true_implementation_001/EXECUTION_AUDIT_001.json`. Its source audit and protocol are scoped to that exploratory run; they do not validate lake geometry, event/control labels, warning performance, or redistribution rights. See `project/SR1_TRUE_READINESS.json` for the remaining gates.

The retained September 2023 SAR product is calibrated from its verified measurement TIFFs and native LUTs. Newly requested SAR products are admitted only after actual successful staging and calibration; an expired token is a recorded absence, not a measurement. Ellipsoid geolocation and candidate spectral masks do not establish terrain-corrected lake boundaries.

The prior notices below apply to historical artifacts and their stated scopes. They do not relabel authenticated new provider pixels as synthetic.

---

# Data admission and current exclusions

Current candidate registries and derivative panels, fitted states, predictions and statistical outputs are excluded from scientific evidence pending independent verification or regeneration. Catalogue hits are search metadata; they do not supply measured optical reflectance, lake area or radar backscatter. Reanalysis summaries require preserved raw response, provider/model/grid, units and checked aggregation before admission.

Legacy artifacts remain excluded. Constructed fixtures and simulations retain their explicit origin and cannot join an observational task as real measurements. Missing source bytes, QA, units, geometry or support is absence or rejection, never a plausible replacement value.

Every future admitted contribution must resolve to actual product bytes and checked hashes, product/version, query, lake outline/CRS, units/scaling, valid measurement support, quality and availability evidence. Derived outputs preserve these parents and permitted fitting scopes. Bulk products may remain in provider-managed or ignored storage subject to terms, with an inspectable retrieval recipe.

This document records research exclusion. Complete enforcement in all production entrypoints is pending. Existing files must not be used to generate submission results simply because a loader currently accepts them.
