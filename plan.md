# Sentinel-GL: forensic audit and research completion plan

**Audit date:** 30 September 2026. **Target machine:** Apple M3 MacBook Air, 8 CPU cores, 10 GPU cores, 16 GB unified memory, nominal 512 GB SSD. **Factory:** the supplied Software Factory v3.3.0. **Status:** corrected numerical core available; observational research, domain integration, and submission readiness remain blocked pending the gates below.

This document is the working specification for subsequent architect and implementor sessions that have access only to this new repository. It deliberately preserves unfavorable findings. No accuracy, detection rate, lead time, confidence interval, operational warning capability, or sensor feasibility conclusion from the legacy project is accepted as research evidence. A passing test suite establishes particular software behaviors; it does not establish authentic data, scientific validity, novelty, or journal acceptance.

## Contents

1. Decision and limits of this audit
2. Evidence inventory and migration disposition
3. Findings and required remedies
4. What the migrated core implements
5. Research questions, estimands, and claim limits
6. Observational data and feature specification
7. Mathematical and temporal specification
8. Experimental design, baselines, inference, and failure analysis
9. Software Factory v3.3 integration
10. Implementation work packages and acceptance gates
11. M3 resource and reproducibility plan
12. Submission preparation and literature work
13. Agent handoff contracts and final release checklist
14. Recovery and reference records

## 1. Decision and limits of this audit

### 1.1 Decision

Rebuild the research pipeline from independently verifiable observations. Retain useful architecture only after correcting its numerical contracts. Keep the legacy system recoverable as forensic history. Do not repair old score tables by adjusting a threshold or rewriting descriptions. Contamination begins upstream of feature extraction and propagates into training, evaluation, statistical tests, figures, and manuscript claims. Reusing any dependent artifact requires a complete, independent provenance demonstration; absent that demonstration, regenerate it.

The immediate deliverable is a small numerical engine plus this plan, not a purportedly complete GLOF predictor. The active repository intentionally contains no migrated observations, feature matrices, model weights, empirical performance tables, or publication figures. Subsequent agents must acquire and verify measurements, implement the domain adapter, freeze a defensible methodology, and execute the experiments before producing claims.

The old project's engineering milestones do not establish scientific milestones. An honest negative result would be valuable if it arose from a valid experiment. A negative result generated from invented observations, simulated test scores, leaking windows, or an invalid comparison remains invalid.

### 1.2 Inspection scope and qualifications

The original working-tree inventory contains **2,048 files, totaling 130,458,387 bytes**, excluding Git internals. Every inventoried regular file was byte-read and hashed. The inspection classified 1,104 files as full-text reads, 780 as numeric-array inspection without pickle, and 164 as binary files hashed without execution. All 148 legacy Python files under `source/` were parsed and indexed; these include 49 test files. Cross-component semantic review concentrated on acquisition, preprocessing, channel extraction, training, scoring, thresholding, evaluation, uncertainty, figures, provenance, and their invocation paths.

These statements must not be inflated into “every binary was scientifically validated” or “every line received independent human peer review.” PNGs, NetCDF contents, arbitrary serialized Python objects, and unsafe checkpoint metadata were not comprehensively interpreted. Numeric NPZ entries were inspected with `allow_pickle=False`; object entries were not deserialized. Git history was inspected for the data-generation commit and relevant ancestral state, not exhaustively audited commit by commit. Network providers were not queried to validate every legacy measurement. The available primary literature was used to challenge assumptions, not to establish a complete systematic review.

The new scaffold baseline contains 140 files before the migrated core and audit outputs. Its active factory policy and execution/evidence code were examined, and its full offline tests were run. Historical factory documents remain distinguishable from active v3.3 policy. Older unrelated project-specific content in historical factory documentation must not become Sentinel-GL methodology.

Legacy test collection found 296 tests. The entire legacy test suite was **not executed**: some tests rerun scripts that write generated research outputs into the original folder, and some load legacy checkpoints. Preserving evidence takes priority over rerunning contaminated artifact writers in place. The new core has 17 passing engineering tests. The supplied factory has 276 passing offline tests; one initial sandbox run failed because a Unix-socket test could not bind, and the permitted rerun outside that restriction passed. The saved logs distinguish these runs. Neither count certifies the research.

### 1.3 Evidence hierarchy

Use four levels: (a) direct source behavior or independently reproduced engineering counterexample; (b) artifact contents and measurable inconsistencies; (c) a scientific risk inferred from the design; (d) unresolved external facts requiring verification. Keep that distinction in issue tracking and prose. For example, a script explicitly drawing Gaussian test scores is direct evidence of score synthesis. A suspicious lake coordinate requires registry verification; this audit does not replace it with an invented “correct” coordinate.

Primary records:

- `docs/audit/legacy_inventory.json`: paths, sizes, SHA-256, array statistics, strict-JSON results, and file inspection classes.
- `docs/audit/legacy_source_index.md`: all legacy source Python files and line-indexed review signals.
- `docs/audit/legacy_evidence_excerpts.md`: selected original code excerpts with file hashes and line numbers.
- `docs/audit/legacy_generation_commit.diff`: historical acquisition generators.
- `docs/audit/legacy_counterexamples.json`: explicitly constructed software counterexamples, never observational results.
- `docs/audit/factory_scaffold_inventory.json`: original scaffold inventory.
- `docs/audit/environment.json`, `core_tests.txt`, `factory_tests.txt`, `legacy_test_collection.txt`: environment and verification scope.

`docs/audit/inventory.py` can inventory a separately restored legacy tree. `extract_legacy_evidence.py` accepts a legacy root and output directory. They do not make a provenance claim merely by calculating a hash. A hash proves identity of bytes, not whether their values came from a satellite.

## 2. Evidence inventory and migration disposition

### 2.1 Observed artifact families

| Legacy family | Observed inventory | Research disposition |
|---|---:|---|
| `data/raw` | 531 files: 430 JSON, 100 CSV, one NetCDF | Quarantine all; reacquire or independently verify every contributing product |
| `data/processed` | 702 files, including 700 NPZ | Quarantine; placeholder preprocessing makes these inadmissible |
| `data/features` | 21 files, including 20 NPZ | Quarantine; feature extractors contain fabricated fallbacks |
| `data/features_real` | 23 files, including 20 NPZ | Quarantine; the directory name is not evidence of authenticity |
| `models` | Five files, including three checkpoints | Do not load into active research; architecture/normalization lineage incomplete |
| `results` | 106 files, including numerical JSON/CSV, figures, reports and weights | Forensic history only; never fill a missing new result from these files |
| `source/results` | Additional generated figures outside the main results tree | Same quarantine, including aliases and alternative search paths |
| Root manuscript, reproducibility and claim maps | Textual research assertions and cached metrics | Rewrite from new verified evidence; retain as historical evidence |
| Legacy source | 148 Python files | Selectively reimplement contracts; do not copy provider or reporting behavior wholesale |
| New `factory/` and factory policy docs | Supplied v3.3 scaffold | Retain policy, identify domain limitations and required maintenance explicitly |

Four hundred raw metadata JSON files failed strict JSON parsing. Several contain Python dictionary representations rather than interoperable JSON. Other files having valid syntax does not establish provider authenticity. The sample `features_real/SGL-001/feature_matrix.npz` contains a 3,227 by 13 panel covering 2016-01-01 through 2024-10-31. Only about 46.33% of its entries are finite; 22,515 entries are missing. Some alleged reflectance channels contain values in the hundreds or thousands, with a sampled maximum near 10,601.8. These are observations about the stored artifact, not a new remotely sensed result. Determine provider scale/offset and lineage before interpreting any such values.

### 2.2 Preservation and Git transition

The verified old local HEAD and remote `main` initially pointed to `aaf063f80905f5f2049fad3d6440e915263d181e`. Existing milestone tags are historical labels, not research certifications. The old folder also contained uncommitted tracked changes and new scientific source/docs. The migration must preserve both states:

1. Annotate a legacy tag at the original remote HEAD.
2. Preserve a separate tagged snapshot of tracked and non-ignored uncommitted research work. Tag descriptions must call it unvalidated legacy material.
3. Save a local tar archive, binary patch, status, path list, and Git bundle. The local tar preserves ignored content as well, excluding only the outer `.git` directory.
4. Replace the active branch tree with the new folder in a normal descendant commit. Do not force-push or erase history. Removing legacy files from current `main` does not remove them from historical commits/tags.
5. Verify remote tag targets and `main` after pushing. Keep a local backup manifest outside active research inputs.

Actual tag names and verification state are recorded in `docs/MIGRATION.md`. The local backup is under `.migration/`, excluded from Git and from source/data manifests. The old folder is retained. The working-tree tar saved during this audit is 116,401,037 bytes with SHA-256 `1cd3c304635b2a834d9331bedea75ae20fad4b60d41b23e6c8b342e663f8f725`. The history bundle is 69,139,108 bytes with SHA-256 `7c5e68d090c481743ebf7dd971ceabdf5d46d306eb4b0a0c0b7a484c913596f8`. These recovery objects are not permitted feature sources.

Do not introduce a runtime legacy fallback. A new-only agent should be able to implement this plan without importing old modules or having the old folder mounted. If forensic inspection later requires original code, restore the legacy tag into a separate read-only directory and document that purpose.

## 3. Findings and required remedies

Severity below describes consequences for admissible research: **P0** invalidates evidence or permits fabrication/leakage; **P1** undermines interpretation, comparison, or reproducibility; **P2** limits packaging/maintainability. “Core corrected” means the migrated numerical primitive addresses the specified defect. It never means the entire research pipeline is already verified.

### F01 — Generated histories represented as sensor observations — P0

**Evidence.** Historical commit `4c10dbe8dac180611fd5036197004d02dfa06054`, titled “Update raw acquisition CSVs with lake-specific physical variations,” changes the acquisition generators and raw series. Functions such as `generate_era5_series` synthesize seasonal temperature and random precipitation. Lake-specific offsets derive from characters in identifiers. ITS_LIVE velocities and other sensors are generated with random or manually varied values. The preserved diff exposes the origin of these histories directly.

**Consequence.** Lake-dependent variation can make a learned model seem geographically meaningful while merely encoding the generator. Subsequent “real data” naming, reality checks, training, or more conservative metrics do not repair this origin. This is fabrication in the artifacts/implementation sense; the audit does not infer anyone's intent.

**Remedy and acceptance.** Start a new provider-backed lineage. Every active raw contribution must link a provider product identifier, query, ROI, time interval, units, acquisition/availability timestamps, and checked export hash. Compare a prespecified sample against independent provider retrievals. Deliberately generated arrays belong only in fixture/simulation namespaces with explicit origins. Reject an observational run containing any generator-origin record or unknown-origin contribution. Work packages WP03–WP05 own this gate.

### F02 — Preprocessing ignores its inputs and creates dummy products — P0

**Evidence.** `source/data/preprocessing/preprocess_optical.py`, `preprocess_sar.py`, `preprocess_modis.py`, `preprocess_era5.py`, and `preprocess_itslive.py` create random arrays for `windows[:5]`, rather than processing complete raw histories. Scene counts and cloud summaries are literal values. Duplicate SAR product directories increase apparent coverage without proving distinct observations.

**Consequence.** A successful preprocessing run is disconnected from measured inputs. Dimensions and finite ranges can pass despite a completely synthetic scientific object.

**Remedy and acceptance.** Implement actual transforms over explicit source products. A changed input product must change the appropriate output/hash, and removing it must remove its contribution or mark missingness. Test resampling and aggregation on tiny analytical rasters; never certify generated arrays through a production acquisition command. Require source-product joins for every output band and date. Empty acquisitions must produce an explicit absence record, not a fabricated image or successful complete panel.

### F03 — Feature extraction fills missing measurements with plausible constants — P0

**Evidence.** The extent, spectral, velocity, temperature, SAR, and meteorological extractors contain plausible fallback values. Examples include extent 1.25 km²; green/red/NIR approximately .08/.05/.03; velocities 25.4 and 85.1; temperature anomaly .85 with a fixed 2.5 °C climatology; VV/VH around -12.4/-18.2 dB; coherence .72; meteorological values 1.2, 15.4, and -5.2. Quality values such as .85, .90, .95, or .98 are similarly generated without a measurement-based derivation. Missing bands can be replaced by other bands.

**Consequence.** Missingness becomes an apparently trustworthy physical observation. Quality-weighted models can give the invented values extra influence. Published sensor contribution and spatial risk claims become uninterpretable.

**Remedy and acceptance.** Missing means missing, with a reason code and observed mask. Quality must be calculated from documented QA bits, valid pixel counts, geometry and uncertainty, never a default positive confidence. A feature must fail or remain absent when its required band is missing. All-incomplete windows are abstentions/errors according to the frozen policy. The migrated core preserves explicit missingness; actual feature acquisition/extraction remains WP05.

### F04 — Physically invalid “coherence” and placeholder InSAR — P0

**Evidence.** `extract_sar.py` calculates a coherence-like value using `clip(.75 + vv_db / 100, 0, 1)`. GRD intensity is not a complex interferometric pair. `source/data/insar/insar_feasibility.py` returns literal coherence .24 and 28 pairs and labels an uncreated interferogram path as completed. Availability and seasonal feasibility figures are also literals.

**Consequence.** Neither a deformation signal nor a broad conclusion of InSAR infeasibility has been measured. Dropping this sensor based on the placeholder report would also bias the methodology.

**Remedy and acceptance.** Remove coherence from GRD-derived schema. An optional InSAR branch must acquire compatible SLC pairs, orbit/DEM inputs and processing records, then compute actual coregistration, interferograms/coherence, quality masks and ROI summaries. Verify files exist and hash them before a completion status. Limit any negative conclusion to the tested geometry, processing pipeline, period and ROI. Published South Lhonak InSAR work is counterevidence that requires examination, not dismissal: [2024 Remote Sensing study](https://www.mdpi.com/2072-4292/16/13/2307). A small CPU pilot is a feasibility decision; it is not evidence that C-band InSAR cannot work throughout the HKH.

### F05 — New acquisition patches stop fabrication but do not acquire a complete dataset — P0/P1

**Evidence.** The current old working-tree acquisition shims raise `AcquisitionBlockedError`, an appropriate fail-closed improvement. New Sentinel-1, Sentinel-2 and MODIS adapters largely count scenes and return success without exporting measured feature contributions and complete lineage. The topography adapter reports elevation summaries but does not implement all claimed slope features. ERA5 request/authentication paths and requested day/hour coverage need validation; checking a key filename without using it does not prove authentication or correct extraction.

**Consequence.** These partial repairs cannot rehabilitate existing artifacts or justify “acquisition complete.” A scene count proves search availability, not local measurement acquisition or preprocessing correctness.

**Remedy and acceptance.** Distinguish `CATALOGUED`, `EXPORT_SUBMITTED`, `EXPORTED`, `HASH_VERIFIED`, `QA_ACCEPTED`, `NO_DATA`, and `FAILED`. Preserve task IDs, retries, provider errors and bounds. An authenticated live pilot must export and independently spot-check values. No network access, expired credentials or API error is a blocked/failed acquisition, never a successful synthetic fallback. Persist progress so interrupted acquisition resumes without duplicate evidence.

### F06 — Geospatial utilities misstate their behavior — P0/P1

**Evidence.** `reproject_to_utm` returns a copy without coordinate transformation. A bilinear resampling routine performs nearest-index selection. Extent assumes 10 m pixels regardless of affine transform/CRS. Cloud masking ignores a threshold parameter, incompletely treats SCL invalid/saturated categories, and fails to reconcile cloud-probability scale conventions.

**Consequence.** Pixel areas, ROI placement, band alignment and cloud exclusion can be wrong even when array arithmetic is internally consistent. Small moraine lakes are sensitive to a few mixed or misregistered pixels.

**Remedy and acceptance.** Store CRS, transform, dimensions and nodata per raster; use verified geospatial transforms and explicit interpolation. Calculate area from an appropriate projected grid or geodesic geometry, with units and boundaries. Resample categorical masks by nearest neighbor; justify continuous-band resampling separately. Validate known coordinate/area examples, rotated/anisotropic pixels, nodata boundaries and band alignment. Raster outputs must pass overlay inspection on actual imagery for a frozen sample, with saved review evidence.

### F07 — Inconsistent channels and sensor descriptions — P0/P1

**Evidence.** Legacy code/manuscript/specifications refer to 15, 13, 26, or proposed 16 channels. The `features_real` 13-channel mapping is area, green, red, NIR, NDWI, vx, vy, LST anomaly, lake VV/VH, moraine VV, ERA5 temperature and precipitation. Some ablation labels instead name EVI, snowmelt, slope/aspect/elevation. Generic index assumptions label ERA5 temperature as coherence. A proposed freezing-level feature is not demonstrated to exist in its selected provider product.

**Consequence.** Ablation claims can concern a sensor that was never present; changing array order silently changes scientific meaning. Model checkpoints cannot be interpreted independently.

**Remedy and acceptance.** One versioned channel schema with names, units, support, sources and transformations governs acquisition, tensors, model configuration, missingness, ablations and prose. No positional constants outside generated schema accessors. Reject unknown or reordered fields unless an explicit compatible mapping is supplied. Schema hash travels in every feature manifest/checkpoint/prediction. Add a sentinel-value channel permutation test and fail a channel-name/data mismatch. Channel count is a consequence of verified measurements, not a target to maintain for a favorable story.

### F08 — Feature assembly loses identity and uses future information — P0

**Evidence.** `assemble_features.py` overwrites duplicate dates via dictionaries, converts malformed numerics into NaN silently, and treats missing files as empty mappings. Annual ITS_LIVE products assigned July 1 and forward-filled can contain observations/publication after that date. Dense daily panels can repeat infrequent observations without preserving their support.

**Consequence.** Duplicate conflicts disappear, availability becomes fictional, and annual averages can leak later acquisitions into earlier decisions. Daily row count overstates independent sensor observations.

**Remedy and acceptance.** Preserve per-observation identifiers and intervals. Duplicate products must be deduplicated by identity; conflicting overlapping products need a frozen aggregation rule or error. Invalid numbers are validation failures with provenance. No backward interpolation for a prospective task. Annual or pair-derived products become usable only at a justified availability time. Repeated values must carry age and origin; do not convert an interpolated estimate into a measured observation. WP04/WP05 must test future append invariance and provider support joins.

### F09 — Distribution “reality gates” encourage generator tuning — P0

**Evidence.** The reality gate tests physical ranges, variance and gap ratios and treats their satisfaction as authenticity. Generator variations were changed to produce lake-specific physical variation. Coverage can become 100% because daily ERA5 exists while optical/SAR channels remain mostly absent.

**Consequence.** Plausibility filters are useful QA but a fabricator can satisfy them. A single dense coarse-resolution sensor conceals missing fine-resolution observations. More realistic statistics do not make data real.

**Remedy and acceptance.** Separate provenance authenticity, physical QA, temporal completeness, spatial support and model eligibility. Report each sensor and channel denominator, acquisition interval, valid pixels, unknown states and age. Spot-check provider values independently. Never impose a requirement that arbitrary variance or a desired gap ratio be reached by editing observations. A sparse genuine dataset may require reducing scope, changing cadence, or declaring insufficient evidence.

### F10 — Training cutoffs, normalization, seeds and summaries are not reliable — P0/P1

**Evidence.** `train_ts_mae.py` uses an import-time seed of 42, trains on complete trajectories extending into 2024, computes normalization over that input, and selects on training loss without held-out validation. Actual batch size and scheduler behavior differ from summary descriptions. Cosine scheduling with a short period can rebound when run beyond that period. The generic trainer also fixes seed 42, mishandles empty loaders and resume state, and averages batch losses without observation weighting. A best checkpoint lacks exact architecture and normalization metadata.

**Consequence.** An alleged pre-2023 warning may rely on post-event training. Replaying a checkpoint on raw units can generate meaningless errors. Claimed seed diversity, parameter count, convergence and configuration cannot be trusted. The inspected legacy model has approximately 1,108,237 parameters under its recorded architecture, inconsistent with the manuscript's roughly 412,000 description.

**Remedy and acceptance.** Seed before model construction; freeze fitting lake/time IDs; preserve normalization and exact architecture. Validation must be independent under the declared design and deterministic for checkpoint selection. Weight loss by observed target count. Record actual batch sizes, stopping status, learning-rate sequence and optimizer state. Resume requires all RNG/device/sampler/optimizer/scheduler states and a verified input manifest; otherwise restart as a new attempt. An epoch cap is `BUDGET_EXHAUSTED`, not proof of convergence. The migrated minimal trainer implements part of this; split joining, full resume and provenance remain outstanding.

### F11 — Masked-autoencoder loss and inference are inconsistent — P0

**Evidence.** An all-false temporal mask produces an empty loss and NaN in the old model. Missing targets are mixed with zero-filled inputs. Unequal visible counts can break gathering assumptions. Score-A may use unmasked self-reconstruction, different from training, and operate on raw-scale zero-filled features because checkpoint normalization is absent.

**Consequence.** A low reconstruction error can reflect identity reconstruction or missingness rather than an anomaly. Missing windows can silently receive favorable scores; raw reflective digital numbers can dominate physical features by scale.

**Remedy and acceptance.** Explicit observation masks must enter the network separately from zero-filled normalized inputs. Loss is over hidden, observed elements only. Reject no-target batches/windows; never turn undefined loss into zero. Use deterministic cross masking for evaluation, with every eligible target hidden once, and the same frozen normalization. Validate dimensions/device/dtype/capacity and visible counts. Core tests now exercise missing targets, hidden-value invariance, malformed masks, undefined loss and deterministic scoring. The scientific mask schedule and missingness eligibility require WP06/WP07 review.

### F12 — Score-B learns from its evaluation input — P0

**Evidence.** The old embedding density scorer automatically fits when called unfitted. A query set can become its own PCA/density reference. Neighborhood count is silently truncated, and broadcast distance computation can allocate query × bank × dimension arrays. Calibration on bank members also allows self-neighbor distances to suppress scores.

**Consequence.** Fit/test separation collapses, apparent normality is circular, and execution can exceed memory. Lake identity or acquisition regime can dominate a supposed anomaly representation.

**Remedy and acceptance.** Require explicit fitted reference IDs, PCA hash and bank manifest. Unfitted scoring raises. Hold out calibration observations from reference-bank construction, or use explicitly audited leave-one-cluster-out calibration; removing one identical row is insufficient for overlapping windows. Use bounded nearest-neighbor queries and a bank-size policy. Migrated Score-B prevents auto-fitting and invalid silent k reduction. Independence, temporal cutoff, per-lake weighting and serialization remain WP06/WP07 obligations.

### F13 — Score-C normalization and alpha selection leak test information — P0

**Evidence.** Old Score-C min/max normalizes whole test-lake trajectories, including future extrema, and can broadcast one Score-A number across a time series. Alpha is selected by score variance rather than a declared evaluation objective. Some sensitivity code linearly blends cached AUCs rather than recomputing predictions.

**Consequence.** Earlier scores change when later observations arrive; batch boundaries alter scores. AUC is a ranking functional and is not linear in alpha. For example, labels `[0,1]`, scores A `[0,1]` and B `[0,-.1]` yield AUCs 1 and 0, yet their equally mixed scores yield AUC 1, not .5.

**Remedy and acceptance.** Freeze reference transformations on declared fitting/calibration data; preserve their hashes. Apply fixed alpha to paired per-window scores with shape/identity checks. Treat output as an anomaly rank score, not event probability. Recompute metrics from each actual sensitivity run. The new combiner preserves historical values when extreme test observations are appended; it allows extrapolation beyond [0,1] rather than pretending those values are calibrated probabilities. Test-lake fitting is forbidden by the domain adapter.

### F14 — Threshold selection is outcome driven and reused on test controls — P0

**Evidence.** Percentile-85 calibration implies roughly 15% flagged calibration windows, contrary to a desired 10% false-alert criterion. Search over percentiles stops at a desired synthetic detection result; fallback values survive failed criteria. A later percentile-90 correction uses the same evaluation controls to “meet” the target. Typed protocols do not govern all actual runners and impose positive labels on a calibration task that can be label-free for operational false alerts.

**Consequence.** Reported FPR is partly guaranteed by construction on the calibration data; desired performance becomes an acceptance invariant. Threshold search can adapt to the event test case.

**Remedy and acceptance.** Fit a threshold using an independent calibration split and a prespecified denominator/comparison/tie rule. Evaluate untouched controls separately. Report empirical calibration achievement without population guarantees. A threshold that misses an event or exceeds the test false-alert target is a legitimate result. The new primitive handles ties and forbids overlap of supplied calibration/evaluation observation IDs. The domain adapter must establish those IDs actually belong to distinct lakes/time supports.

### F15 — Lead times use center dates, backdating, or fabricated labels — P0

**Evidence.** A 180-day window score is dated at its center even though its last input is about 89–90 days later. A sustained alarm can be backdated to its first crossing rather than the time the sustaining observation arrives. Earliest historical crossings can be reported as precursors far outside a useful warning horizon. Some baselines calculate lead time as `(length - first_flag_index) × 30`, ignoring the event date. Final-six-window labels in the inspected panel refer to center dates in February–July 2024 and latest inputs in May–October 2024, after the October 2023 South Lhonak event.

**Consequence.** Apparent warning lead time is inflated or attached to post-event evidence. Thousands of days of lead can simply mean a noisy detector was always on. A missing alarm is sometimes encoded as zero days, concealing failure.

**Remedy and acceptance.** Store trailing-window start/end and actual decision availability separately. Decisions use the latest required available input. An alarm is declared when the required sustaining observation arrives. Use a predeclared horizon, abstention/reset and episode/refractory policy. Lead is `event_time - declaration_time`; no alarm is null with reason. Exclude same-day/post-event inputs conservatively until time-of-day evidence exists. Core tests cover these basic dates, but the complete episode/eligibility process remains outstanding.

### F16 — Synthetic injection labels are disconnected from perturbations — P0/P1

**Evidence.** Duration “windows” can actually be 1/3/6 daily feature rows while output windows use a 30-day stride. Injection indices exceed the number of evaluated windows; missing cells remain missing but are still labeled positive. An invalid channel may default to column zero. Seeds based on character sums collide. Baseline labels are sometimes final-six indices unrelated to injection support.

**Consequence.** Detection rates do not necessarily test detection of the injected perturbation. Scenarios sharing the same lake/background are not independent event replications. Convenient synthetic profiles can preferentially reward the model.

**Remedy and acceptance.** Maintain an injection ledger with source background hash, physical channel name, units, support interval, seed, magnitude/ramp/duration, affected observed cells and eligibility. Reject absent channels, all-missing perturbations and invalid bounds. Derive window labels by declared overlap/decision support; retain injection dates separately from event labels. Use deterministic seed namespaces with collision tests. Simulation results must be visibly conditional on these constructed scenarios and cannot become real-GLOF sensitivity claims.

### F17 — Methods in the comparison table estimate different things — P0

**Evidence.** Learned-model metrics can come from synthetic anomalies on controls while baseline metrics use pseudo-labels from a real-case panel. Models also use different missingness eligibility and feature preprocessing. AP and trapezoidal PR area are mixed under an ambiguous “AUC-PR” label. Missing extent can become an all-zero baseline score.

**Consequence.** Apparent superiority may arise from a different dataset, label, denominator, scaling or eligibility policy. A table cannot compare those numbers meaningfully.

**Remedy and acceptance.** One locked observation ledger per task; all methods join to the same IDs, labels, availability dates and eligibility. Fit each baseline only on permitted data; document comparable tuning/compute. Present common-eligible comparison and separate method-specific availability as distinct analyses. Use the precise name `average_precision`; sklearn's AP is not trapezoidal PR area: [official AP documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). Empty/unsupported baseline inputs must abstain or fail, not generate benign scores.

### F18 — Confidence intervals and paired tests operate on simulated scores — P0

**Evidence.** `run_bootstrap_ci.py` draws approximately 102 Gaussian scores per lake/method and adds a cached-AUC-dependent offset to final-six labels. Bootstrap replicates without the only positive lake are assigned chance values. The paired DeLong calculation uses `sqrt(var1 + var2)` without paired covariance, and treats overlapping windows as independent observations. Variance floors and subtraction of a normal CDF can create artificial numerical behavior.

**Consequence.** These confidence intervals and p-values are not uncertainty on actual model predictions. In a five-lake bootstrap with one event lake, the probability of omitting that lake is `(4/5)^5 = .32768`. Replacing an undefined metric in those replicates with .5 fabricates an estimand.

**Remedy and acceptance.** Statistical code reads immutable actual prediction rows only. Define sampling units before choosing a test. With one real event, report descriptive case timing and conditional sensitivity rather than a population event-detection CI. For sufficiently many independent event/lake clusters, preserve paired methods and within-cluster structure during resampling; justify any conditional or stratified scheme. Report undefined replicate counts and assumptions; do not silently drop or impute them. Seed variation estimates optimizer variation on a fixed corpus, not event-population uncertainty. WP08 implements independently checked inference.

### F19 — Cloud robustness is simulated rather than measured — P0

**Evidence.** `cloud_stratified_eval.py` constructs synthetic Score-C/Score-B values near cached AUCs .68/.65, offsets final-six values, and substitutes .5 in single-class cloud bins. Metadata gap fraction can stand in for cloud fraction without representing the ROI's actual cloud mask.

**Consequence.** The claimed effect of cloud on the model has not been measured. Seasonal, acquisition and sensor support effects are confounded with cloud coverage.

**Remedy and acceptance.** Join predictions to measured per-ROI QA coverage and channel missingness. Predeclare bins or continuous effects; preserve counts of lakes/events, windows and eligible support in each stratum. Mark metrics unestimable when classes are absent. Include a missingness-only baseline and a matched/time-aware diagnostic. Distinguish controlled masking simulation from observational cloud stratification, and avoid causal cloud-robustness conclusions from observational strata alone.

### F20 — Ablation and sensitivity numbers are invented — P0

**Evidence.** `run_ablation.py` uses random drops for channel removal, literal full-model AUCs .6842/.6695, alpha-weighted cached AUCs, an EMA formula favoring span five, and lead times 1710 or 1680 days. Its channel labels do not match the tensor. Existing tests can demand ranking changes rather than verifying that an actual model was rerun.

**Consequence.** The tables do not measure sensor importance or hyperparameter sensitivity. A forced rank change is another desired-output rule. A reported negative or small drop is equally inadmissible.

**Remedy and acceptance.** Every ablation maps a named schema feature/sensor to an executed inference or retraining procedure and a new prediction ledger. Distinguish inference occlusion (robustness of the fitted model) from training without a modality (value of that modality). Refit normalizer/PCA/calibration where required, using the same split and declared budget. Hash checkpoints and input masks; verify a changed intervention actually executes. No test requires a particular scientific ranking. WP07/WP08 choose a small hypothesis-driven ablation set before final evaluation.

### F21 — Figures and reports contain invented curves and metric defaults — P0

**Evidence.** `evaluation/figures.py` creates ROC curves from powers of an arbitrary FPR grid and labels them with cached/default AUCs. A missing timeline is replaced by sinusoidal/random scores. Several report paths default to expected performance constants. A data-quality report is literal rather than computed.

**Consequence.** A polished figure can display a trajectory or ROC that no model produced. Cached text can survive after an experiment has failed or never run.

**Remedy and acceptance.** Plot only independently checked prediction/measurement rows, with row hashes and a plotting manifest. Compute ROC vertices from exact labels/scores and AP from the same rows. Missing inputs cause an explicit figure failure or “not estimable” panel with counts; never a synthetic replacement. Manuscript numbers are generated from claim IDs/verified metrics, not mutable hand-maintained dictionaries. Verify plot vertices/numbers against the underlying ledger and independently inspect axes, units, calendars and uncertainty meaning.

### F22 — Claim verification and quarantine are bypassable — P0

**Evidence.** Claim verifiers treat merely present objects as evidence and skip or count some computed/invariant/conjunction claims as passing. Old registry/quarantine files are not enforced at all entrypoints; aliases and `source/results` escape simple path-based restrictions. Prediction metadata can contain literal hash strings without recomputing them against actual artifacts. Some loaders accept duplicate keys or nonfinite JSON.

**Consequence.** An agent can “verify” a fabricated claim by creating a plausible JSON file, reusing a stale artifact, or moving it to another directory. Shape-only checks do not establish scientific linkage.

**Remedy and acceptance.** Inputs are admitted by immutable source origin/manifests, not directory names. Recompute hashes and join all observations, transforms, checkpoints, predictions, metrics and claim selectors. Check completeness as well as existence. Reject malformed JSON, unknown statuses, stale reviews, path escape, symlinks, orphan/duplicate rows, changed inputs and forged hashes. The new integrity utilities implement strict JSON/path/hash primitives. A full lineage/release verifier remains WP02/WP09.

### F23 — Geography, lake identity, controls and event bibliography are underverified — P0/P1

**Evidence.** Lake names/IDs are inconsistent between some feasibility and registry outputs, including swapped assignments. A Merzbacher coordinate around 39.852, 79.865 requires independent investigation before treating it as a valid lake ROI. Training/control lakes with known dynamic ice-dammed behavior or engineering interventions cannot simply be called stable normal lakes. A registry said to originate entirely from a basin-specific inventory includes sites outside that inventory's domain. The manuscript's “Sattar et al. 2026” does not match the verified 2025 Science paper.

**Consequence.** Wrong ROIs invalidate acquisition, selection can favor familiar event lakes, and undocumented control exclusions bias false-alert estimates. No known recorded outburst is weaker evidence than verified absence of all anomalies.

**Remedy and acceptance.** Verify persistent inventory IDs, polygons, moraine/glacier ROIs, country/basin, aliases, event history, interventions, selection and exclusions. Freeze all candidate sites before looking at scores. Keep “no catalogued target event during follow-up” distinct from “physically normal.” Audit labels at the event source and define uncertainty. [ICIMOD's 2020 inventory](https://lib.icimod.org/records/p869r-n4132) concerns the Koshi, Gandaki and Karnali basins; do not silently use it as a verified inventory for every HKH lake. South Lhonak event reference: [Sattar et al., Science, 2025](https://doi.org/10.1126/science.ads2659). Calendar/time-zone treatment must be explicit for the night of 3–4 October 2023.

### F24 — Fixed outputs masquerade as engineering invariants — P0/P1

**Evidence.** Milestone/README criteria require a desired false-positive rate or detection fraction. Some tests enforce cached metric values, a particular ranking, or the existence of artifacts written by placeholder scripts.

**Consequence.** Agents are incentivized to alter data/code until the desired research outcome passes. Engineering acceptance becomes circular scientific confirmation.

**Remedy and acceptance.** Test algebra, identity, availability, support, denominator, execution and reproducibility. Predeclare performance thresholds as decision criteria, not conditions that must be made true. If the target is not achieved, record failure/negative evidence and narrow claims. Old tests may supply examples of intended interfaces but must not be migrated as result oracles. An unfavorable correct result must pass the integrity gates.

### F25 — No supported v3.3 research profile for these tasks — P0 integration blocker

**Evidence.** `factory/engine/plan.py` accepts only `binary_classification`. Its built-in metric contract includes probabilities for Brier/log loss, threshold in [0,1], class-count requirements and seed-fixed-test comparisons. Self-supervised training has no event labels; a single-event case study and operational control follow-up require different estimands and denominators. The supplied `project/research_plan.json` is an unfilled template, not a valid frozen Sentinel-GL plan.

**Consequence.** Inventing labels, clipping scores to probability range or adding pseudo-events to satisfy the factory would reproduce the original problem. A custom JSON profile without executable dispatcher/validator/metric support also does not solve it.

**Remedy and acceptance.** Implement a reviewed domain adapter with explicit profile dispatch, task-specific schema, independently recomputed metrics and adversarial tests. Preserve shared provenance, freeze, stale-review, attempt, tier and receipt gates. Leave research status blocked until a real end-to-end fixture proves dispatch and malformed/contaminated research is rejected. Section 9 specifies the work. Do not report certification under the binary profile for a different scientific task.

### F26 — Typed runtime cannot execute the numerical core as supplied — P1 integration blocker

**Evidence.** `factory/engine/contract.py` provides `python-cpu-v1` using isolated interpreter flags including `-I`, `-P`, `-B`, `-S`. Site-packages are not loaded; torch/numpy/sklearn and ordinary project imports are unavailable without a supported runtime design. Contract arguments are sorted by key and supplied as positional values, not named CLI options. An `allowed_arg_values` set is declared but its membership is not enforced.

**Consequence.** Passing core tests under an ordinary interpreter does not show supervised factory execution works. Silent fallback to the older command path would change assurance without disclosure.

**Remedy and acceptance.** Add a supervisor-owned, pinned scientific runtime with explicit import roots, binary/package hashes and an agreed argument ABI. Validate allowed placeholder values and unknown arguments. Prove clean offline launch of a tiny core fixture, rejection of external module hooks and mutation of dependencies/entrypoints. Keep CPU/MPS runtime identity separate. Do not merely disable isolation flags globally or let the plan supply arbitrary interpreters.

### F27 — Factory isolation and receipt claims exceed implementation — P0/P1 trust limitation

**Evidence.** `gatekeeper.py` inherits most of the environment, including an explicit supervisor-key variable when supplied. The default private key is in the same user's filesystem. The HMAC fallback writes the signing secret into the exported `.pub` file. `network: disabled` is a contract field without demonstrated network denial. Address-space/process limits can be unsupported or fail on macOS. Shared-user execution is not a hostile-worker security boundary.

**Consequence.** A receipt can establish useful local run linkage while failing to establish that an adversarial same-user worker could not forge evidence or access secrets/network. Calling this sealed independent attestation would overstate the guarantee.

**Remedy and acceptance.** Scope receipts honestly to accidental-corruption/run-traceability unless a separate protected supervisor environment is deployed. Remove key environment leakage; require Ed25519 for public verification and never export HMAC secret as public verification material. If HMAC is retained for local fixtures, downgrade its tier and keep it private. Implement and demonstrate denied network/key access if that assurance is claimed. Record unsupported enforcement rather than pretending it exists. Mandatory fixes/tests precede research certification; stronger isolation may require a separate process user or external reviewer and is not magically provided by this MacBook.

### F28 — Signed hardware fields contain false measurements — P0 for efficiency claims

**Evidence.** `gatekeeper.py` supplies `cpu_time=time.monotonic()-t` and `memory_peak=0` to the signed receipt. The first is elapsed wall time; the second is an unmeasured literal. A signature does not make either an accurate measurement.

**Consequence.** Hardware/efficiency comparisons can be fabricated by the supervisor itself even when project code is honest. Signing zero peak memory is a stronger misleading assertion than omitting it.

**Remedy and acceptance.** Measure elapsed wall, child CPU and memory as separately named quantities with platform semantics, units, method and sampling limits. Unsupported measurements are null with a reason. Update receipt schema/validators and tests accordingly. Device synchronization is required for MPS timed sections. Hardware claims remain disabled until these fields are truthful and linked to actual trials. This audit leaves the supplied factory policy unchanged and explicitly blocks such claims; WP02 owns the maintenance patch.

### F29 — Seeds and windows are not independent event samples — P1

**Evidence.** Multiple training seeds are used as if they could strengthen a single-event warning conclusion, and overlapping 180-day windows are pooled as independent detections. Four control lakes provide only four lake-level clusters, however many dense rows exist.

**Consequence.** Precision is overstated. Five optimizer seeds are five model realizations, not five real events. For a conventional two-sided exhaustive paired sign-flip test with four independent pairs, the smallest attainable p is 2/16=.125; with five pairs it is 2/32=.0625. Creating more overlapping rows does not change those limits.

**Remedy and acceptance.** Prospectively count event/lake/geographic clusters and calculate attainable uncertainty before promising population claims. Report optimizer variation separately from sampling uncertainty. Expand the externally verified event cohort if a broader warning claim is required; otherwise publish a bounded case study/benchmark/observability result. Do not treat non-significance as equivalence or as proof that deep learning cannot help.

### F30 — Novelty and mechanisms are presently unestablished — P1

**Evidence.** A masked autoencoder plus embedding kNN and a weighted score is an engineering combination, not demonstrated methodological novelty. Coarse reanalysis and 1 km thermal cells may largely represent regional seasonality, while optical/SAR absence and lake identity can dominate. Recent directly related work already evaluates simple/deep free-satellite hazard models.

**Consequence.** A “first AI early-warning system” claim would require substantial evidence this project currently lacks. Predictive association does not prove a physical failure mechanism or an operationally actionable warning.

**Remedy and acceptance.** Complete the primary-source novelty matrix and faithful strong baselines before freezing the contribution. Test missingness-only, seasonality, lake identity, weather-only and simple change-detection explanations. Treat mechanistic claims as requiring independent observations/interventions or a defensible physical validation argument. A directly relevant [August 2026 preprint](https://arxiv.org/abs/2608.12422) separates susceptibility and trigger timing and reports strong simple-model performance; it is unreviewed related work, not a verified benchmark to copy. Replication should use its available evidence and disclose any differences.

### F31 — Reproducibility, licensing, release and maintenance are unfinished — P1/P2

**Evidence.** Old checkpoints/configuration are incomplete, absolute paths appear in scripts, bytecode/generated objects were tracked, dependencies are not fully locked for a clean runtime, provider access is undocumented, and the new root license is proprietary with restrictions on reuse. The factory has historical documentation from other projects.

**Consequence.** A reviewer cannot yet reproduce the system from a clean checkout. Public visibility is not a license granting reproducibility rights. A local installed-package snapshot is not a portable build or hash-pinned environment.

**Remedy and acceptance.** Use repository-relative paths/configuration; pin and verify a clean arm64 runtime with dependency provenance; preserve provider attribution and data redistribution terms; obtain the owner's explicit release-license decision before a submission claiming open reproducibility. Keep proprietary factory licensing distinguishable from any future research-code/data licenses. Make a clean external reproduction pass with a release manifest, honest limitations and independent review. Never fabricate an OS/GPU measurement or claim to have executed a clean installation that has not been attempted.

## 4. What the migrated core implements

### 4.1 Active code and deliberate limits

| Module | Implemented numerical contract | Remaining caller/domain responsibilities |
|---|---|---|
| `source/sentinel_gl/integrity.py` | Duplicate/nonfinite JSON rejection; canonical hashes; safe file/path/hash checking | Provider authenticity, complete lineage joins, origin admission, signatures, schema evolution |
| `features.py` | Explicit finite-observed/NaN-missing matrices; declared fit/holdout IDs; single-fit observed-element normalization; daily trailing windows and latest availability | Independent registry/split verification; raster extraction; source-specific age/support; eligibility and calendar construction |
| `model.py` | Mask-aware temporal MAE; values plus validity inputs; observed hidden-target loss; capacity/mask checks; separate reconstruction without undefined unmasked loss | Reviewed architecture/mask configuration; input lineage; trailing-window scheduling; train/validation independence |
| `training.py` | Explicit seed/device; weighted target losses; fixed cross-mask validation; best validation checkpoint; real stopping/budget status | Full provenance/checkpoint contract; data batching; search accounting; resume/device RNG state; research trial management |
| `scoring.py` | Explicit fitted PCA/kNN reference; frozen score combination; deterministic masked reconstruction; causal EMA | Reference bank split/time checks; immutable fitted-state serialization/hashes; per-lake weighting; eligibility and score-age semantics |
| `calibration.py` | Tie-aware empirical threshold selection; supplied observation-ID separation; observed calibration fraction | Trusted ID/source joins; separate calibration lakes/time; sampling uncertainty; operational episode policy |
| `metrics.py` | AUROC/AP on arbitrary finite rank scores; undefined single-class metrics; explicit empty controls; alarm declaration timing without backdating | Task labels, episode rates, exposure denominators, blocked/missing rows, event uncertainty and statistical inference |

This code preserves the broad legacy temporal-MAE idea while replacing its unsafe contracts. The observation-mask input changes the input projection from C to 2C. **Legacy checkpoints are incompatible and must not be loaded.** Engineering default dimensions are not a selected research architecture. A method can be mathematically well-defined yet perform poorly or learn seasonal/missingness shortcuts. The core deliberately makes no automatic acquisition, scientific population, event label, threshold guarantee or publication claim.

The current observation normalizer pools observed rows across fitting lakes. That can overweight lakes with denser sensors or longer follow-up. WP06 must choose and justify global row weighting, lake-balanced weighting, or another transform before research freeze. The current frozen min/max combiner is a transparent simple baseline; quantile/robust scaling may be explored using development data only. Do not describe either as universally optimal.

The basic alarm primitive expects ordered eligible observations. It does not infer calendar gaps, reset streaks on censored windows, model exposure, or merge episodes. The production adapter must enforce those rules. Availability-based windows can have nonmonotonic or duplicated decision dates when a late product releases several windows at once; the scheduler must deduplicate/order or explicitly model release batches rather than silently treating them as separate earlier decisions.

### 4.2 Verification performed

Run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
python3 factory/run_self_tests.py
python3 factory/gatekeeper.py --help
```

The first suite verifies 17 constructed engineering cases, including strict JSON/path integrity, masked-target behavior, normalization, unfitted-query rejection, test-extreme invariance, single-class/empty estimability, delayed availability, sustained alarm declaration, ties, causal EMA, and a tiny training/checkpoint round trip using the fitted normalizer. The tiny training case intentionally uses constructed tensors and is not a scientific split demonstration. The source README states this clearly. There are no empirical GLOF accuracy tests.

The factory suite verifies its existing implementation against 276 tests. It does not test every scientific issue identified here. Keep these checks after domain/runtime maintenance; add meaningful mutation/independence tests that expose the missing guarantees. The local requirements file is an exact installed dependency snapshot for this core/test environment, without wheel hashes. A clean install and supported supervised launch have not yet been demonstrated.

### 4.3 Repository boundaries

`source/` contains the active research implementation and its tests. `data/` will contain small verified registry/manifests and feature panels under explicit lineage; bulk acquisition lives in ignored/provider-managed storage. `project/` contains methodology and factory plan/state. `docs/audit/` is forensic evidence, excluded from observational inputs. `artifacts/` will hold generated attempts, not untracked scientific defaults. Factory historical material retains a historical role only. No script may search the old folder, its results, audit excerpts or recovery archives for a missing model/data/metric.

## 5. Research questions, estimands, and claim limits

### 5.1 Define the scientific target before selecting a model

The legacy project conflates three distinct questions. Separate them in experiments, metrics and manuscript claims:

**Task A: controlled perturbation detection.** Does a representation trained on independently verified background histories detect prescribed perturbations better than strong simple detectors on untouched background lakes? Population is the explicitly constructed scenario family on the sampled observational backgrounds. A primary estimand might be paired change in lake-balanced AP or detection fraction at a threshold calibrated on separate unperturbed controls. This is a simulation benchmark. It does not estimate actual GLOF prediction sensitivity.

**Task B: retrospective event case reconstruction.** What scores and declared alarms would the frozen method have emitted before the independently documented South Lhonak event, under a specified historical availability assumption? Population is the named case/period. Estimands are eligible monitoring coverage, actual declaration date/lead, control false-alert burden and comparison to prespecified methods. A single case cannot estimate event-population sensitivity or prove transferable early warning. A modern reprocessed product does not automatically establish what was available in 2023; distinguish acquisition-time retrospective analysis from historical as-of availability.

**Task C: operational control burden.** During follow-up of independently selected lakes with no catalogued target event, how often would the method have issued alerts, how long would it have remained in alert, and how often would it abstain? Population is the selected control cohort and follow-up. Primary estimands include false/uneventful alert episodes per lake-year, proportion of eligible monitoring time in alert, and eligible coverage. Other unrecorded physical anomalies may exist, so “uneventful alert” is often the more defensible term than proving every alert false.

**Optional Task D: multi-event generalization.** If an adequate externally verified event cohort can be built, evaluate event sensitivity within specified horizons under lake/region/time separation. This requires a new protocol/amendment and sample-size argument. It cannot be manufactured by labeling final windows, converting perturbations to real events, or counting multiple trajectories of the same event as independent sites.

**Optional Task E: InSAR observability.** Under a specified SLC geometry and processing workflow, are enough trustworthy moraine deformation observations available at useful latency/cadence? Report measured coverage, uncertainty and coherence. Detection of deformation is not by itself proof of GLOF causation or actionable warning.

### 5.2 Minimum credible contribution routes

Choose one primary route after WP01/WP03; keep others supporting or exploratory:

- A reproducible benchmark for causally available multisensor histories with controlled missingness/perturbations, strong baselines, honest leakage checks and an external holdout.
- A bounded observability/negative-result study explaining when the measurement cadence, resolution and false-alert burden prevent meaningful warning under the tested protocol.
- A methodological advance in representations/calibration for irregular, missing remote-sensing observations, demonstrated across sufficient independent sites/events and not just one celebrated case.
- A tightly scoped retrospective case study plus public verification artifacts, if broader claims cannot be supported. Venue expectations may be narrower; do not reword it as a general early-warning system.

A negative result must distinguish inadequate data, invalid target labels, insufficient precision, failure of the particular fitted method, and evidence against a specific prespecified improvement. Only the last, under adequate precision and valid design, supports a strong comparative negative claim. A failure to collect sufficient observations is a data-feasibility result, not a failed prediction experiment.

### 5.3 Prospective claim ledger

Before final evaluation, author `project/claim_specification.json` containing claim ID, text template, kind, task, population, estimand, direction, practical effect/precision target, required experiments, exclusion conditions, and permitted fallback wording. Examples of legitimate predeclared outcomes:

- “On the defined constructed perturbation family and held-out backgrounds, the paired AP difference was X, with uncertainty conditional on [stated sampling units].”
- “For the South Lhonak case, no qualifying alarm was declared within H days before the event under the stated availability/eligibility policy.”
- “Control monitoring produced X episodes per eligible lake-year, with Y% monitoring coverage; this exceeds the prespecified burden target.”
- “The measured SLC pilot provided N eligible moraine pairs under the tested geometry; uncertainty prevented interpretation of deformation.”

These are templates with unresolved values, not claims presently supported. Never insert a cached legacy number. Do not claim first/best/generalizable, calibrated event probability, causal precursors, operational readiness, or global InSAR infeasibility without additional evidence specifically supporting it.

### 5.4 Freeze and amendments

Development uses a declared exploratory cohort and preserves its search log. Only after feasibility and protocol review may a methodology be frozen. Record split/ROI/channel/availability/architecture/mask/calibration/baseline/metric/statistical/release choices in a versioned signed or hashed plan. If final-test information changes a choice, retain the old epoch, mark the analysis exploratory, and require a new untouched holdout for a new confirmatory claim. Never quietly relabel the old test as validation after viewing its outcome.

## 6. Observational data and feature specification

### 6.1 Registry and event records

Create a provider-independent candidate registry with persistent `lake_id`, inventory/source identifiers and versions, canonical name/aliases, polygon geometry/CRS/hash, country/basin/region, lake/dam class, glacier/moraine ROIs, acquisition feasibility, intervention history, selection status and exclusion reason. Centroids alone are insufficient for feature extraction. Preserve the full candidate universe and deterministic selection procedure so favorable sites cannot be added or removed after scores are seen.

Each event record requires `event_id`, linked lake/inventory ID, source citation/record locator, event type, date/time or interval, timezone/uncertainty, evidence quality, detection/recording bias notes, and reviewer adjudication. Multiple reports of one event share one event ID. Ice-dammed cyclic drainage, moraine failure and rainfall flooding are not interchangeable labels. Controls require searched sources, observation/follow-up interval, known interventions and explicit “no catalogued target event” status. Any unresolved location/event is excluded from confirmatory evaluation until adjudicated, with its absence counted.

The current geographical suggestions are candidates, not validated labels or coordinates. WP03 must verify them independently and quantify resulting sample size. Require two independent checks for the primary event/ROI: source record plus image/geometry review. Avoid copying a coordinate from a generated CSV or a legacy prose report.

### 6.2 Raw product lineage

Use strict versioned manifests. A minimum raw record has:

| Field family | Required meaning |
|---|---|
| Identity | Unique record ID; provider, collection/product/version; product/scene/pair identifiers; actual origin (`observational`, `simulation`, `fixture`) |
| Query | Canonical request hash; spatial ROI/version/hash; interval; bands/variables; filters/orbit; provider endpoint; processing/export task ID |
| Time | Acquisition/support start/end; source publication/availability if known; retrieval/export time; availability assumption and evidence |
| Geometry | CRS, affine transform, native resolution, footprint/support, pixel dimensions, nodata/masks |
| Measurement | Raw units, scale, offset, fill values, QA definition; uncertainty/support if supplied |
| Storage | Repository-safe artifact locator or immutable external URI/version; bytes; SHA-256; retrieval verification status |
| Status | Catalogue/export/hash/QA state; failure/no-data reason; retry history; software/configuration/dependency hashes |

Provider products and derived measurements must not share identity merely because they have the same date. No authentic observation record may be issued before export/file hash validation. Query logs must omit credentials but retain sufficient request information for replication. A provider request can return no data; preserve that outcome. Respect original terms/attribution and release a retrieval recipe where redistribution is restricted.

### 6.3 Sensor-specific choices and risks

**Optical.** Use explicit product/version and date coverage. The Earth Engine Sentinel-2 surface-reflectance harmonized collection begins in March 2017; early coverage is incomplete. Its reflectance scale is 10,000, and QA60 behavior changes across the archive. SCL is a separate classification at 20 m; cloud-probability values use 0–100. These conventions require explicit adapters and temporal QA checks. Do not claim 2016 Sentinel-2 L2A coverage from this collection. A justified Landsat alternative can extend the record, with separately documented scale/offset, resolution, harmonization and sensor effects. [Official Sentinel-2 product description](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED).

Lake area must use a verified water delineation method and uncertainty/sensitivity to snow, shadow, turbidity, mixed pixels and thresholds. Spectral ratios must name their bands and physical domain, mask invalid denominators and preserve negative reflectances if product/QA permits them rather than inventing a universal range. Report valid pixel counts/fractions on the lake ROI; scene-wide cloud percentage is not local cloud coverage. Optical vegetation indices over water/moraine require a physical rationale before becoming channels.

**SAR intensity.** Filter homogeneous instrument mode, polarization, relative orbit and pass, with orbit-specific processing or a justified harmonization. Preserve incidence angle, layover/shadow considerations and ROI support. Earth Engine's GRD collection is calibrated terrain-corrected backscatter in dB; averaging intensity should normally be defined in linear power before conversion if that is the intended quantity. dB subtraction is a log-ratio, not a linear VV/VH power ratio. Never infer interferometric coherence from these values. [Official Sentinel-1 GRD description](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD).

**Thermal.** MOD11A1.061 is a nominal 1 km daily product with scale .02 for LST and QA/error bits. Small lake ROIs may contain mixed land, snow and water within one cell. Distinguish day/night, valid coverage and observation time; a regional surface-temperature feature is not measured lake-water temperature. Fit a day-of-year climatology only from permitted historical data, with minimum independent coverage and explicit uncertainty. Reject an invented constant climatology. [Official MODIS product description](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A1).

**ERA5/ERA5-Land.** Choose a named reanalysis product and exact variables, grid/native support and conversion semantics. Handle precipitation accumulation/reset correctly; differentiate hourly values, accumulated fields and derived hourly products. Air temperature converts K to °C only once. A lake-sized polygon does not improve the native reanalysis resolution. Record availability assumptions because retrospective reanalysis can differ from an operational forecast/archive. A dense weather channel cannot serve as proof of optical/SAR coverage. Check the selected collection's metadata and CDS documentation before implementation: [ERA5-Land hourly catalogue](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY).

**ITS_LIVE velocity.** Prefer timestamped image-pair records with actual first/second acquisitions, pair duration, quality/uncertainty, projection and glacier support when asking about temporal anomalies. Annual mosaics support annual summaries, not arbitrarily July-available warning features. Static or annual repetition must retain support/age and must not be counted as daily independent measurements. Separate magnitude/direction or components, define glacier ROI and expected spatial resolution, and reject poor support. [NASA ITS_LIVE data access](https://its-live.jpl.nasa.gov/itslive-web/index.xml) and [annual-product documentation](https://its-live-data.jpl.nasa.gov.s3.amazonaws.com/documentation/ITS_LIVE-Regional-Glacier-and-Ice-Sheet-Surface-Velocities.pdf).

**Topography.** Use actual DEM-derived elevation/slope/aspect with CRS-aware derivatives, documented resolution/nodata, spatial support and circular treatment of aspect. Static descriptors are susceptibility/site features; they cannot alone time a transient failure. Keep static metadata separate from temporal channels unless the model's conditioning and geographical shortcut tests justify duplication.

**InSAR, optional.** A separate phase branch uses SLC, geometry/processing quality and actual coherence/deformation uncertainty. Start after core data feasibility, with a tightly bounded pilot. No completion status without generated products and no deformation interpretation without valid phase unwrapping/reference/atmospheric/geometry checks. Do not use a poorly processed pilot as evidence against all InSAR approaches.

### 6.4 Feature schema and lineage

A channel definition requires schema version/hash, name, physical quantity, units, provider products, band/variable expressions, native and output support, ROI, QA eligibility, aggregation/calendar support, age policy, availability rule, scale/offset, uncertainty and missingness reasons. Maintain a separate static-feature schema. Avoid expanding channel count merely to resemble a multimodal paper. An absent supported feature is preferable to a fabricated one.

Derived records link one or more raw record IDs, transformation version/code/configuration hash, fitted-state hash where applicable, output identity/hash, observation-support interval, decision availability and quality/missingness. Aggregation outputs should expose how many independent products and valid pixels contributed. A value cannot be marked observed after interpolation; record an imputation flag and uncertainty separately. Global fit scope and per-channel climatology fit scope are explicit and independently verified.

Recommended reasons include `NO_PROVIDER_PRODUCT`, `CLOUD_MASKED`, `QA_REJECTED`, `OUTSIDE_COVERAGE`, `INSUFFICIENT_ROI_SUPPORT`, `STALE_OBSERVATION`, `UNRESOLVED_UNITS`, `EXPORT_FAILED`, and `NOT_YET_AVAILABLE`. Reasons remain distinguishable from physically measured zero. Feature NaNs exist only alongside invalid masks/reasons; finite fabricated substitutes are forbidden.

### 6.5 Cadence and eligibility

Select daily, composited or irregular modeling from genuine support. Daily calendar alignment is an indexing convenience, not evidence of daily sensing. A proposed starting experiment is a trailing 180-day context and 30-day decision stride, but these are legacy-derived candidates requiring pilot review, not fixed optimal parameters. Compare sensor refresh age and useful warning horizon to the stride before freezing. Do not hide insufficient cadence by backward interpolation.

Choose a minimum support policy prospectively. It can require a defined fraction of observed channel-time entries and key channels with maximum age, or use a model supporting sparse observations. Apply a shared primary evaluation eligibility policy to all methods. Separately report each method's native availability and total monitoring coverage. If almost no windows meet the policy, stop and revise the feasibility question rather than loosening it after favorable scores appear.

## 7. Mathematical and temporal specification

### 7.1 Normalization and missingness

For observed indicator m(t,c) and training-only feature x(t,c), estimate mean and standard deviation from the declared fitting population and weighting rule:

`mu_c = sum(w_t * m_tc * x_tc) / sum(w_t * m_tc)`

`scale_c = sqrt(sum(w_t * m_tc * (x_tc - mu_c)^2) / sum(w_t * m_tc))`.

No observed fitting values is a schema/data error. A constant observed channel uses an explicit documented unit scale or is excluded by a prespecified rule; never fit a scale on the test cohort. Normalized input is z=(x-mu)/scale where observed, and zero where absent, paired with the observation indicator. Zero filling is an encoding choice, not imputation of a measured value. Freeze/hash all fitted values and their observation IDs; appending a future test value cannot alter prior normalized inputs.

The current core implements unweighted observed-element normalization and records constant channels. The architect must resolve weighting and outlier handling using development data. Robust alternatives require a predeclared fitting algorithm, identical scope rules, and actual comparison; neither clipping nor a “physical range” should silently erase valid extremes near a hazard.

### 7.2 Masked training objective

Let r(b,t) be a deliberate temporal reconstruction mask and m(b,t,c) the observation mask. Valid targets are v=r AND m. The masked loss is:

`L = sum(v * (predicted_z - target_z)^2) / sum(v)`.

If the denominator is zero, loss is unestimable and the attempt fails or the batch is explicitly excluded under a recorded support rule. It is never zero or a fabricated favorable loss. Epoch summaries sum numerators/denominators across batches rather than averaging unequal batch means. The encoder receives only visible temporal tokens; each token's observed-channel validity is provided explicitly. Review whether whole-token masking allows the model to overfit seasonality and whether its distribution matches the inference objective.

The encoder may use full attention inside a closed trailing context. That is compatible with causal decisions only when every included observation was available by the decision time. It is incompatible with claiming availability at the window center or embedding the full future trajectory to score earlier dates. If true daily streaming or per-token forecasts are required, implement a separate causal architecture/objective and evaluate it explicitly.

### 7.3 Scores and fitted-state scope

Score-A is the observed-target reconstruction error from a fixed evaluation cross-mask schedule. Every eligible observed target is hidden once; errors are aggregated by observed target count. The present two-part modulo schedule is an engineering implementation; evaluate justified alternatives on development data and freeze the schedule. Report per-channel error contributions and support so weather density does not silently dominate an aggregate.

Score-B is the average distance to k neighbors in a training-reference embedding space after a training-fitted PCA (if used). Define the embedding pooling, time support, bank weights, k, component count and distance metric. PCA and bank never fit queries. Avoid calibrating on self-neighbors or highly overlapping same-lake training windows without a declared cluster-aware scheme. A bank insufficient for chosen k/components fails; do not truncate silently. kNN is a representation-density score, not a GLOF likelihood.

Score-C is `alpha * transform_A(A) + (1-alpha) * transform_B(B)`, with each transform fitted on declared non-test reference/calibration data and alpha fixed before final evaluation. Raw anomaly scores need not lie in [0,1]. Probability metrics such as Brier/log loss are prohibited unless an independently calibrated event-probability model and appropriate event labels exist. [Official metric semantics](https://scikit-learn.org/stable/modules/model_evaluation.html).

Any EMA is recursive using only current/past scores. Define its initialization, resetting after missingness, time-step interpretation and warm-up. A span in rows changes meaning with irregular cadence; a time-decay formulation may be more defensible but must be frozen. Future appended data must not change previous EMA values, PCA, score scaling or thresholds.

### 7.4 Decisions, labels and alarms

A window has immutable identity from lake, schema, context/support interval, source IDs and protocol. Store its start, end/latest acquisition, latest required availability, scheduled evaluation date, actual decision time, eligibility and exclusion reason. No score may be declared earlier than the latest required available input. If historical publication time is unknown, use a clearly labeled acquisition-time retrospective approximation and do not claim actual real-time warning.

Event labels require an explicit target. For Task A, labels arise from the intervention ledger and frozen overlap rule. For Task B, event status and timing arise from external records; there is no scientifically established universal six-window “pre-event anomaly” label. Binary precursor labels from an arbitrary horizon are at best a defined proxy, requiring sensitivity analyses and explicit limitations. Susceptibility labels and imminent-trigger labels are different targets.

Define comparator `score >= threshold` (or another frozen rule), calibration weighting, tie handling, sustained crossings, gap resets, minimum eligible support, warm-up, horizon, alert end/hysteresis and refractory period. If q consecutive decisions are required, declaration occurs at the qth qualifying arrival. Do not credit the first arrival. Missing/ineligible periods cannot silently maintain a streak. Time in alert and episode counts must account for decision cadence and censoring.

Lead time is conditional on detection: `event_time - actual_declaration_time`. No detection has null lead with an explicit status. Report detected/undetected counts next to any mean/median lead; never average nulls as zeros or hide misses. Include seasonal negative-control traces and long-run burden so an always-on detector does not receive impressive “lead.” A conservative date-only case excludes event-day inputs because intraday ordering is unknown.

### 7.5 Metrics and denominators

AUROC compares positive-negative score ordering and handles ties. It requires both classes under the declared estimand; single-class results are null/status, not .5. AP also requires an appropriate positive/negative evaluation population here and is reported with prevalence/counts. Always-on/no-information controls are separately computed if meaningful; do not replace unestimability with a baseline number.

Operational window false-alert fraction is `flagged eligible control windows / eligible control windows`. It weights lakes with more eligible windows more heavily. Report an explicitly lake-balanced counterpart if chosen and distinguish it from episode rate or time fraction. False/uneventful episodes per lake-year use declared follow-up/exposure, censoring and episode definitions; they cannot be derived from a bare percentile alone. Total monitoring coverage counts missing/abstained follow-up rather than discarding it from the story.

With no eligible denominator, output `NOT_ESTIMABLE`, null numeric fields and a reason. Genuine zero alerts over positive exposure is a measured zero with its denominator. “Not measured,” “not estimable,” “not detected,” “failed” and “blocked” are distinct statuses. Strict serialization uses no NaN/Infinity JSON values and no magic zero/chance placeholders.

## 8. Experimental design, baselines, inference, and failure analysis

### 8.1 Cohort and splits

The architect must define a candidate cohort and a prospective precision/feasibility calculation before naming a final lake count. The previous 15-training/five-evaluation arrangement is not a scientifically necessary invariant. A practical initial structure is separate fitting, development-validation, threshold-calibration, and final evaluation lakes, with a temporal cutoff appropriate to each task and an optional regional holdout. Exact IDs/counts depend on verified data and must be frozen before score-based selection.

For a historical event case, training and every fitted transformation must use only permitted earlier data if a prospective-like claim is made. Training on other lakes after the event can still support a retrospective representation analysis, but cannot be described as a model available before the event. A final event lake must not contribute to fitting, hyperparameter selection, PCA, score scaling or threshold calibration. Seasonal statistics from that lake also constitute fitting unless a explicitly defined online adaptation protocol uses only past information and is evaluated separately.

Split units include lakes, shared geography/glaciers, raw sensor scenes, overlapping temporal supports and repeated interventions. A strict “no shared source scene” rule can be infeasible for multiple lakes observed in one satellite scene or coarse reanalysis cell; do not fabricate independent source IDs to satisfy it. Instead, record shared support, define geographic/source clusters, prevent fitted/evaluation leakage where required and justify the sampling unit. A domain adapter may implement a scientifically justified support-overlap policy, but cannot waive it invisibly. Whole-lake and whole-region separation are different claims.

Within a temporal split, purge overlapping context windows across the fitting/validation boundary or assign complete contexts to one split. For a 180-day context, a chronological row split without purging can share nearly all inputs. Tune on development only, then freeze final models/procedure. Preserve all attempted configurations and failed runs, including exploratory negatives. A new run nonce does not make a repeated seed/data pair an independent sample.

### 8.2 Baseline suite

Use a small strong suite with documented implementations and development-only tuning:

| Baseline | Question addressed | Fairness requirements |
|---|---|---|
| Constant/no-alert and always-alert | Sanity checks for burden, prevalence and coverage | Same eligibility/exposure; no invented AP for unsupported labels |
| Seasonal/climatology deviation | Whether context is mostly seasonal weather/appearance | Training-only climatology; actual support/units; uncertainty |
| Lake-area trend/change rule | Added value beyond interpretable lake change | Genuine area measurements; prespecified smoothing and threshold calibration |
| Weather-only simple model or rule | Whether dense coarse reanalysis explains scores | Same time cutoff; distinguish susceptibility from timing; avoid static leakage |
| Robust standardized feature distance / PCA residual | Added value of neural representation | Same observed-mask treatment; training fit and reference scope |
| CUSUM/change detector | Temporal shifts without a large neural model | Training/calibration reference; drift/reset/threshold rules; actual dates |
| Isolation Forest and one-class SVM | Competitive unsupervised tabular alternatives | Same features/scaling, budget, IDs; training-only fitting; missingness policy |
| Missingness/QA-only and calendar-only | Shortcut and acquisition regime explanations | No physical channel values; same task/splits |
| Mechanism-matched simpler encoder | Value of temporal MAE, not just model size | Comparable inputs and development budget; real retraining |

If a multi-event labeled task is feasible, add a strong simple supervised baseline appropriate to that task, rather than comparing unsupervised scores only to under-tuned one-class models. Select baseline references and reproduce their documented configurations; do not call an ad hoc area threshold an established operational standard without a source.

Declare the tuning budget per method and explain sensible differences for deterministic algorithms. A method cannot receive test-derived thresholds, a more favorable time range or a hidden feature than competitors. Record not only metric means but valid/abstained rows, per-lake contributions, seed variation, runtime and support. If a common eligibility cohort shrinks drastically, report both the comparison's limited scope and overall data availability.

### 8.3 Controlled perturbation protocol

Perturbations must test explicit physical/data failure hypotheses, not generate convenient positive scores. On verified held-out backgrounds, choose prespecified families such as abrupt/ramped area change, glacier velocity change, temperature/weather extremes, SAR intensity shifts or combined changes, only where the underlying channel and support make sense. Define magnitudes from physical literature or training-development distributions and sensitivity ranges, never the test model's detectability. Keep negative controls including seasonal transitions, plausible nonhazard extremes, sensor outage and misregistration effects.

Use a two-axis design: physical perturbation and missingness/support degradation. Missingness can be randomized for a controlled stress test, with its origin labeled simulation; observed cloudy strata remain observational diagnostics. Preserve plausible covariance/support rather than independently perturbing every channel as if all were synchronous. Inject before normalization/scoring, using physical units, and do not modify final labels after observing outputs.

A paired experimental unit consists of a background lake/time support and scenario seed, with all methods using the identical intervention. Repeated scenarios on one lake remain clustered. Avoid exhaustive factorial runs unless they answer a real hypothesis. A compact chosen set of sensor removals/mechanism tests is defensible; arbitrary `2^number_of_channels` expansion wastes the Mac's resources and creates multiplicity.

### 8.4 Ablations and sensitivity

Prespecify a primary mechanism comparison and a limited set of secondary analyses. Candidate actual retraining ablations are reconstruction-only vs density-only vs fixed combination, simple encoder vs temporal MAE, and exclusion of a physically justified sensor group. Candidate inference robustness tests hide an existing modality or degrade its support after fitting. These answer different questions and need separate labels.

For retraining without a modality, refit normalization/representation/PCA/reference and calibration from permitted data; preserve common sample IDs and compute/search policy. For inference occlusion, record exactly which observed entries become unavailable, maintain target accounting and report coverage. Ensure hidden values do not enter a scorer through an alternate route. Report that absence may make some windows unestimable instead of forcing a score.

Sensitivity parameters can include context length/stride, mask schedule, k/PCA dimension, score alpha, EMA decay, eligibility and false-alert target. Choose a small justified grid on development data, preserve all results, and rerun each actual procedure. Final sensitivity analyses that use the test cohort are exploratory unless explicitly predeclared. Do not choose alpha/span after final evaluation and then call the selected row the primary result.

### 8.5 Statistical inference

Write a standalone statistical analysis specification identifying estimand, unit, weighting, dependence, pairing, effect size, interval/test method, missing/unestimable outcomes, practical threshold, precision target and multiplicity family. Obtain statistical review for population claims. Use all actual immutable prediction rows and independently validate analytical examples, ties and extreme cases.

For Task A, distinguish (i) fixed-background optimizer variability across model seeds, (ii) scenario variability and (iii) lake/geographic background variability. A nested or crossed resampling scheme must reflect the actual design and keep paired methods aligned. A hierarchical bootstrap is not automatically valid with four lakes; report sensitivity to clustering and refrain from a narrow population CI when cluster support is insufficient. A conditional simulation interval has conditional scope.

For Task B, report a descriptive trace, eligible horizon and comparator outcomes. No event-population ROC/CI from one event. If a proxy binary window analysis is shown, mark its target and dependence clearly and avoid implying many independent events. For Task C, show per-lake exposure/rates and uncertainty whose model assumptions are stated; highly clustered episode counts require more than an iid binomial interval on overlapping windows. For a sufficiently large Task D, use an independently reviewed event/cluster design and external holdouts.

Paired DeLong, if ever used, requires paired covariance and appropriate independent observation assumptions. It is not the default for overlapping time windows. A valid paired difference uses `Var(A-B)=Var(A)+Var(B)-2Cov(A,B)` under its assumptions. For permutation/sign-flip tests, state exchangeability and enumerate small designs rather than drawing enough random permutations to imply nonexistent resolution.

Predeclare one confirmatory multiplicity family and a correction such as Holm where appropriate. Separately label exploratory analyses. Nonsignificance means insufficient evidence for the tested contrast, not equivalence. An equivalence claim requires a justified margin and a suitable precision/design before data inspection. Never flip below-chance scores using test labels; investigate direction, labels, unit scale, and pipeline integrity, then retain the original final result and disclose any exploratory correction.

### 8.6 Diagnostics and failure taxonomy

Record failures by stage: provider/credential, missing coverage, geolocation, unit/QA, unsupported feature, stale observation, context eligibility, normalization, optimization, scoring, calibration, label uncertainty, inference and release verification. Each failure carries affected IDs, reason, attempted remedy, remaining limitation and claim impact. Avoid a single vague “data quality issue” that hides systematic exclusions.

Required diagnostics are per-lake/channel/season support, observation age, raw/native footprint, missingness correlation, score components, calibration vs evaluation distributions, alert episodes/exposure, complete/abstained windows, and train/development/test cutoff summaries. Save representative actual image overlays and panel traces chosen by a frozen/random-sampled selection procedure rather than aesthetically favorable cases.

Test obvious alternate explanations: model identification of lake/geography, month/monsoon and sensor availability; dense weather overwhelming sparse modalities; annual products leaking future support; known engineering intervention affecting a control; duplicated exported scenes; and label misalignment. A suspiciously perfect result triggers an integrity investigation, not celebration. A below-chance result triggers the same checks without automatically reversing the scores.

## 9. Software Factory v3.3 integration

### 9.1 Compatibility means an executable contract

The active scaffold's authority is user brief → constitution → factory specification → reviewed/frozen project methodology → implementation. Only active v3.3 policy outside historical `legacy/` paths governs this project. Read `factory/README.md`, `constitution.md`, `factory_spec.md`, `AGENT_WORKFLOW.md`, architect/implementor specifications, and the executable engine. Where prose and executable behavior differ, record the mismatch and resolve it through a reviewed maintenance change; do not exploit it.

`plan.md` is the audit/backlog and handoff specification. It is **not** the factory's executable `project/research_plan.json`, not a freeze, and not a release certificate. The existing JSON remains an intentionally unfilled template. `project/READINESS.json` records explicit blockers. `project/methodology.md` remains a draft until scientific choices and adapter gates pass. Leaving research blocked is preferable to manufacturing values that satisfy a profile.

### 9.2 Required domain dispatch

Proposed profile identifiers such as `sentinel_gl_v1` are design names, **not currently supported factory commands or profiles**. WP02 must implement and review all the following together:

1. Plan-validator dispatch by supported profile/schema version. The domain validator accepts unlabeled SSL fitting, separate reference/calibration/control/event tasks and task-specific thresholds/units. Unknown profiles/fields fail.
2. Cohort/source/event/injection manifest validators. Resolve genuine source support, lake/geographic clusters, time boundaries, shared scene/reanalysis support, exclusions and origin. Do not invent independent source identities merely to bypass binary split rules.
3. Experiment/audit dispatch. Common frozen-file, attempt, receipt, input/output mutation and freshness checks remain mandatory. Domain checks verify fitted-state lineage, cutoff, decision availability, expected complete prediction rows and task-specific estimability.
4. Independent domain metric recomputation inside the trusted checking layer. It must not import the project's reported-metric function as its “independent” verifier. Hand-check analytical fixtures and compare a separately implemented/library reference with exact tie conventions.
5. Task-aware training evidence. SSL fitting has training/validation reconstruction objectives and support, without fabricated event classes. Deterministic baselines have method/fitting evidence rather than invented epochs. Budget exhaustion is a reported failure/limitation, not automatically converged training.
6. Statistical protocol dispatch. Seed-fixed-test comparison remains optimizer-conditional where used; event/cluster inference needs a reviewed task-specific implementation. No widening to population claims by rewriting the sampling-unit string.
7. Claims/review/release dispatch with supported kinds, actual experiment dependencies, estimand/population/scope, uncertainty and blocked unestimability. Shared tiers, attack registry and stale-review checks remain in force.
8. Fixture-vs-research admission. A tiny domain fixture can test the contract with constructed arrays and known outcomes. It must be permanently `intent=fixture`, with no observational manuscript claims. Research manifests containing fixture products fail regardless of copied shapes/hashes.

Do not implement a wrapper that outputs fake probability labels/scores to the unchanged binary verifier. Do not mark every task as simulation to avoid observational provenance while still writing real-GLOF claims. A partial custom schema that bypasses gatekeeper dispatch is not compatibility.

### 9.3 Runtime and supervisor maintenance

The scientific runtime must be supervisor-owned and match a clean pinned environment. Design a CPU runtime first; add MPS only with truthful device telemetry and declared numerical tolerance. The runner receives validated experiment identity, supervised run directory and seed through an agreed positional or named ABI. Confirm actual contract sorting/resolution in tests. The current audit also compares argv using legacy `command` substitution rather than the typed resolver; WP02 must make validation compare the authoritative resolved launch specification so typed flags/interpreter identity are handled correctly.

Validate permitted placeholder values and unknown arguments. Prevent plan-controlled arbitrary interpreter paths, shell flags, external import roots or startup hooks. Resolve declared library/package paths through pinned supervisor configuration, not a project-controlled unrestricted `sys.path`. If import isolation changes, review its trust consequences and test them explicitly. A wrapper that inserts any path it is given would undo the boundary.

Truthful receipt fields require a schema amendment: wall duration, child CPU measurement, memory method/peak or null reason, device, package/binary hashes, input/output root, seed, run nonce and enforcement status. The existing zero-memory/wall-as-CPU behavior must not remain in a research receipt. Evidence assurance must state local shared-user limitations. Ed25519 public verification does not prove an adversarial worker could not read a same-user private key.

Sanitize the worker environment from credentials/signing material; ensure no private key enters artifacts/bundles. Require asymmetric public verification for distributable receipts. Demonstrate real denied network/key access if the tier claims it, or downgrade the assurance description. Unsupported macOS RLIMIT or other enforcement is explicit. Preserve failed attempts and failed launch logs, including unsupported runtime/device failures. Do not relabel them successful fixture runs.

### 9.4 Evidence contract

Each final prediction row requires experiment/run ID, task, method/configuration hash, seed, lake/cluster/window ID, source/support manifest hash, schema hash, fitted normalizer/encoder/PCA/bank/combiner/calibration hashes, checkpoint identity, context start/end, acquisition/support/decision times, eligibility/reason, actual finite score components or null failure status, label/event/injection reference, and threshold/episode policy identity where relevant. Labels are joined from frozen records, not trusted as freely editable prediction columns.

A metrics record identifies its exact row set/hash and denominators, task/population/estimand, weighting, values/statuses, interval assumptions and statistical protocol hash. A claim references metric selectors and supporting artifacts; no number can be admitted by merely existing in a JSON dictionary. Reports and figures include input hashes and generation code/configuration. Release verification follows the entire dependency graph, including source origin and frozen fitting scopes.

Completeness matters: every expected method-window row exists as a value, abstention, failure or exclusion status. Dropping difficult windows must change the coverage/denominator and be detected. A common-eligible comparison is a named derived cohort, not a hidden filter. Added/deleted files in frozen directories invalidate the epoch. All paths are repository-relative or immutable external locators bound by manifests; symlink/path escape and opaque hashes are rejected.

### 9.5 Mutation and assurance tests

The domain gate suite must reject at least: generated observations marked observational; an authentic file with changed units; an incorrect but plausible lake ROI; test-lake normalization/PCA fitting; future support in a historical window; center-date declaration; stale annual velocity assigned July 1; duplicate scene aliases; phantom source IDs; tampered checkpoint normalization; simulated scores injected by a report script; row omission; single-class .5 placeholder; empty-exposure zero; linear interpolation of cached AUC; made-up figure curves; mismatched actual/resolved argv; unsupported profile; unsigned/stale/forged receipt; exported HMAC secret; leaked key environment; dishonest CPU/memory fields; and old cached review after a changed audit.

Some authenticity/location tests require independently reviewed provider records and cannot be entirely automated. State that limit. Automated checks should fail on a deliberately corrupted fixture even if its metric looks impressive. Conversely, a properly sourced poor result must pass evidence integrity while failing the scientific performance target. That separation is the central acceptance criterion.

### 9.6 Supported commands and current blocked state

The shipped CLI lists `init`, `freeze`, `run`, `record`, `audit`, `certify`, `status`, `handoff`, diagnostic commands and compatibility aliases including `release-certify`. Use current `--help` rather than inventing commands. After reviewed integration and a real frozen plan, the intended sequence is:

```sh
python3 factory/gatekeeper.py freeze .
python3 factory/gatekeeper.py run . all
python3 factory/gatekeeper.py audit .
python3 factory/gatekeeper.py certify .
python3 factory/gatekeeper.py handoff .
```

These are existing command shapes whose success depends on the pending domain/runtime implementation and valid project evidence. They are not claimed to work for the current template. Do not execute a research freeze now, change `intent` to fixture just to obtain a research-looking certificate, or pass a compatibility alias around a failing science gate. Review is a documented scientific assessment, not independent proof. The final release must explicitly disclose guarantees that remain outside automation.

## 10. Implementation work packages and acceptance gates

These packages are ordered by scientific dependencies. Architect approval means review of a concrete specification/evidence, not a request to approve a vague idea. Within the user's authorized local scope, agents should complete reversible implementation and checks without repeatedly requesting permission. Credentials, an owner's release-license decision, external publication, additional hardware/cloud expenditure or new research-scope commitments may require actual user input. Never treat elapsed time as approval.

### WP00 — Recovery, quarantine and corrected core

**Status:** numerical core and engineering tests implemented during this audit; remote migration must be verified in `docs/MIGRATION.md`.

**Inputs:** original inventory, old Git state, new factory scaffold. **Outputs:** legacy tags/local archive, new active source, audit records, README, this plan and blocked readiness state.

**Acceptance:** remote legacy target checked; dirty scientific work preserved; active main has no migrated contaminated data/weights/results; old local folder retained; archive/bundle hashes verified; core tests pass; factory baseline tests pass. A fresh checkout contains the plan and evidence needed for new-only agents. No scientific result has been generated during migration.

**Failure handling:** if remote push is denied, finish all local artifacts and report the exact branch/tag push failure; do not claim completion. If the remote changed, fetch and preserve that state before an ordinary integration; never force-overwrite others' work. Never restore a legacy artifact into active data to make a smoke test pass.

### WP01 — Research charter, literature and feasibility design

**Owner:** architect, with scientific/domain review. **Depends on:** WP00. **Outputs:** `project/research_charter.md`, `project/literature_matrix.csv`, `project/candidate_registry_specification.md`, `project/statistical_analysis_specification.md` draft, explicit contribution route and primary/secondary tasks.

**Work:** verify event/lake definitions and candidate selection; read competing primary studies; distinguish physical mechanism, susceptibility and timing; formulate primary estimand and useful alert burden/precision targets; quantify independent clusters needed for proposed claims; identify missing expertise/ground truth. Choose a small observational pilot without examining final-test model scores. Define rules for narrowing scope when data/event support is inadequate.

**Acceptance:** each prospective claim has an estimand/population, independent unit, obtainable measurement and permitted negative/inconclusive outcome. A novelty table compares inputs, availability, splits, tasks, baseline strength and actual contribution against related work. No invented bibliography or year. If fewer independent events than required can be verified, broad event-generalization claims are removed or WP11 expansion is selected.

**Checks:** citation DOI/title/year verification; a hand-worked example of each task's denominators and date/availability calculation; an explicit precision/attainable-test-resolution calculation. Literature retrieval limitations are listed. **Stop condition:** a scientifically undefined target cannot proceed to model training.

### WP02 — Reviewed factory domain/runtime/evidence maintenance

**Owner:** architect specifies; implementor patches factory and adapter in a separate reviewable change. **Depends on:** WP01 task contracts; can proceed while provider pilot is prepared.

**Outputs:** supported versioned domain profile/dispatcher; schemas; trusted metric/inference validators; scientific runtime; truthful receipt schema; assurance description; expanded factory/domain tests; architecture decision records; changed factory version/hash as appropriate.

**Work:** implement all dispatch layers in section 9; preserve shared gates and original binary behavior; correct typed argv validation, placeholder admission, package loading, key handling and false telemetry; document network/resource enforcement. Define null/status semantics and task-specific cohort expectations. Supply a permanently marked tiny end-to-end domain fixture, plus corruptions that fail.

**Acceptance:** original factory suite passes; a supervised offline fixture genuinely imports and runs the core; a nontrivial failing fixture fails the intended gate; research origin cannot be spoofed merely by changing a label string; all new schema fields are consumed by actual checks. Typed launch matches authoritative resolution. Unmeasured hardware values are null/reason, not zero; CPU and wall semantics are separate. A distributable receipt cannot expose an HMAC secret. Unsupported isolation is disclosed. Review contains concrete objections and resolutions, not a generic PASS.

**Commands:** run existing core/factory tests and new domain suite; CLI `--help` documents implemented profiles/runtime. Proposed new adapter commands are documented only after implementation. **Stop condition:** research cannot be frozen/run/certified through an unsupported profile or a weakened wrapper.

### WP03 — Verified registry, event cohort and live acquisition pilot

**Owner:** implementor with domain adjudication. **Depends on:** WP01; may run before WP02 completion, outside a claimed certified experiment.

**Outputs:** versioned candidate/lake/event/control registry; ROI files and overlay reviews; provider adapters; raw source manifests; an actual pilot acquisition dossier; failures/cost/storage record.

**Work:** verify polygons and identities; choose one event site and one independent candidate/control for feasibility, plus a prespecified time slice covering distinct seasons where possible. Authenticate through user-controlled provider accounts without copying secrets into source. Request catalogue records and export actual measurements/small products for each required sensor. Validate units/scales/QA and time support against independent provider examples. Persist real unavailable outcomes. Keep this pilot exploratory so its inspected sites are not silently used as untouched final test.

**Acceptance:** every pilot output is backed by provider identity/query/task/hash and actual bytes. At least one real observation per proposed required modality is independently checked; modalities lacking support remain explicitly absent. Images/ROIs are visually aligned and archived with reviewer notes. A network/provider failure leaves no fabricated product. The acquisition pipeline can resume a deliberately interrupted export without duplicate IDs. Record actual cost/quotas; no hypothetical “all sensors acquired” status.

**Stop condition:** if observational authenticity or lake identity cannot be verified, block derived research. If sensor support is inadequate, amend the planned task/cadence before final-model evaluation.

### WP04 — Temporal support, lineage and split engine

**Owner:** implementor; architect approves support-overlap policy. **Depends on:** WP03 records, WP02 schema.

**Outputs:** source/derived manifest joiner; availability scheduler; split manifests; temporal purging/geographic clustering; provenance graph; eligibility ledger; proposed acquisition-time vs historical-as-of limitations.

**Work:** separate observation interval, acquisition date, publication/retrieval and decision availability. Link annual/pair/composite products to full support. Resolve duplicate dates/products deterministically and preserve conflicts. Implement split-group and fit-state joins, shared coarse-cell/scene clusters, future-input rejection and context-boundary purging. Define observation-age and late-release batching rules.

**Acceptance:** appending future evaluation data leaves all earlier normalized features/scores/thresholds unchanged; a product containing later acquisitions cannot enter an earlier decision; a deliberately duplicated alias is detected; a wrong declared fitting ID fails using actual manifests. Same-day/post-event exclusion matches the event uncertainty policy. Every expected window has an immutable ID and status. No silently overwritten dates or source IDs.

**Checks:** analytical date fixtures around event day, leap day, annual products, delayed exports, overlapping contexts, irregular cadence, duplicates and release batches; explicit test proving center-date claims fail. **Stop condition:** date/support ambiguity incompatible with the claim prevents that claim, even if a model can still be trained for a narrower retrospective analysis.

### WP05 — Genuine geospatial processing and features

**Owner:** implementor with remote-sensing review. **Depends on:** WP03/WP04.

**Outputs:** actual raster/series transforms; single feature/static schema; source-linked feature panels; QA/support/uncertainty reports; deterministic small test rasters; inspected real overlays.

**Work:** replace all placeholder processing with product-specific scale/offset/QA; reproject/resample honestly; calculate actual area/spectral/SAR/thermal/weather/velocity features with suitable ROIs; build permitted climatologies; preserve absence and age. Remove unmeasured channels. Use real native support and independent observations, not arbitrary 10 m assumptions or per-lake constants.

**Acceptance:** analytical raster tests recover expected coordinate/area/aggregation values within justified numerical tolerance; missing bands remain absent or fail; changing a contributing input affects the appropriate feature and lineage; all values have documented units/support; proposed schema names exactly match tensors. Actual visual/provider spot checks pass for a prespecified sample, and unresolved channels are excluded with reasons. Data-quality summaries are computed from manifests, not literals.

**Stop condition:** weak support or unresolved scaling prevents that channel's scientific use. It does not justify inventing fallback values to retain a multimodal channel count.

### WP06 — Training/data loaders and fitted-state contract

**Owner:** implementor; architect reviews method identity. **Depends on:** WP04/WP05; production launch depends on WP02.

**Outputs:** reproducible data loaders; explicit model config; immutable fitted normalizer/reference/checkpoint schemas; training/validation histories; restore/replay tool; seed/search registry; no-result smoke evidence.

**Work:** decide weighting/mask schedule and valid context support; freeze exact configuration rather than relying on library defaults; construct after explicit seeds; train on permitted IDs/time only; use independent validation and fixed selection criterion. Bound batches/bank size; store architecture, schema, normalization, fit IDs/cutoff, loss support, optimizer/scheduler/sampler and all required CPU/MPS/Python/NumPy RNG states. Implement audited resume or explicitly disable it and retain failed attempts.

**Acceptance:** clean-process checkpoint replay matches predictions within a prespecified measured tolerance; seed/configuration/device values come from actual execution; changing forbidden inputs fails before training; unsupported/all-missing batches do not create zero loss. A resume equivalence test is required if resume is advertised. Validation targets/denominators and stopping/budget status are truthful. Metadata cannot be a hand-authored hash unrelated to inputs.

**Checks:** core tests plus independent split/normalizer/reference lineage fixtures; actual tiny supervised training run under supported factory runtime; memory bound measurements. A numerical smoke run remains fixture/development, never a claimed research result.

### WP07 — Predictions, calibration, baselines and simulations

**Owner:** implementor; architect approves fairness/target specification. **Depends on:** WP06 and WP02–WP05.

**Outputs:** full prediction ledger; explicit calibration artifacts; episode/exposure engine; actual strong baseline implementations; intervention ledger; named ablation configurations; shared comparison cohorts.

**Work:** execute closed-window inference using verified availability and fitted-state hashes; fit thresholds on separate calibration units; implement observation-age/streak/reset/refractory/warm-up rules; real baseline fitting and paired intervention evaluation; account for every expected row. Separate physical perturbation and missingness tests. Keep all model scores as scores unless a probability calibration task is justified.

**Acceptance:** every method has the same primary task IDs/labels/eligibility and no test-based calibration. All row omissions/abstentions appear in coverage. Ties/horizons/no-alarm states are independently checked. Each actual ablation/sensitivity produces new executed predictions and verifies intervention identity. A threshold failure produces a negative result without changing the data. Synthetic/fixture rows cannot join the observational event task as real positives.

**Stop condition:** unverified labels or unfair cohort/time/feature access invalidate the comparison, regardless of performance.

### WP08 — Independent inference, diagnostics and negative-result evaluation

**Owner:** implementor plus statistical/scientific review. **Depends on:** WP07 actual predictions and WP01 finalized analysis specification.

**Outputs:** independently recomputed metrics; cluster/conditional inference; diagnostics and complete failure taxonomy; conservative claims table; explicit estimability limits.

**Work:** recompute from immutable row sets; test metric definitions and ties against analytical examples; implement paired/cluster dependence and multiplicity only where justified; report undefined resamples; analyze shortcuts, support and seasons; preserve all seeds/attempts. Differentiate optimizer variability from corpus/event uncertainty. Compute control episode/exposure quantities rather than relying only on window FPR.

**Acceptance:** no generated score enters uncertainty calculations; reported numbers match independent computation; no .5/0 placeholder for undefined estimands; every CI names its population/unit/assumptions; inadequate clusters yield bounded/descriptive claims. Exact small permutation/sign-flip limits are respected. Negative/inconclusive outcomes satisfy integrity gates; a desired result is never a required software assertion.

**Stop condition:** statistical precision insufficient for a claim causes the claim to narrow, remain inconclusive, or require WP11 cohort expansion.

### WP09 — Artifact/figure/manuscript evidence graph

**Owner:** implementor builds; architect reviews claims. **Depends on:** WP08.

**Outputs:** computed figures/tables; claim-to-evidence selectors; report generator; citation library; figure manifests; full release provenance graph; draft manuscript.

**Work:** plot actual prediction/measurement rows; obtain ROC points/AP from those rows; show missingness/coverage/statuses; present actual time/uncertainty units. Bind every number in text/table/figure to a verified metric and row set. Remove legacy assertions and images. Describe the actual final channel/architecture/split/training pipeline, including budget failures and negative results.

**Acceptance:** deleting or changing a required source/row/checkpoint/metric invalidates the claim/figure; missing inputs cannot create fallback curves or defaults; plotted values match numerical vertices/underlying data. Citation title/year/DOI and physical claims are reviewed. The abstract states bounded population/task, not unsupported operational readiness. A figure's aesthetic quality does not substitute for provenance.

### WP10 — M3 resource trials and clean reproduction

**Owner:** implementor; independent reviewer reproduces where possible. **Depends on:** WP06; final reproduction after WP09.

**Outputs:** clean arm64 dependency lock/environment recipe; actual CPU/MPS comparison and runtime/memory trial records; failure logs; fresh-checkout replay; redistribution/retrieval instructions.

**Work:** construct a clean environment outside frozen source; pin wheel/package hashes and platform constraints; measure actual current device/build availability; verify CPU reference vs MPS on representative finite/missing/masked fixtures; record sustained runs and resource limits. Measure child CPU/wall/RSS/device allocations distinctly. Synchronize device timing. Keep actual observed device fallbacks/exceptions visible. Run at least the factory-required hardware trial count/duration if making efficiency claims.

**Acceptance:** a clean checkout/env genuinely installs and supervised launches; results replay to the declared tolerance or nondeterminism is disclosed and justified before final inference. Measurements have methods/units/limits, not literal zeros. Long-run memory remains within a measured safe budget or the run fails explicitly. Hardware-specific claims use that device's actual measurements. If another machine is unavailable, disclose local-only reproducibility rather than inventing external reproduction.

### WP11 — Optional cohort expansion and InSAR pilot

**Owner:** architect designs independent scope; implementor executes. **Depends on:** WP01/WP03 feasibility and primary scientific need.

**Outputs:** additional verified event/lake/region cohort or bounded SLC feasibility dossier; a protocol amendment; revised precision plan; actual products/uncertainty.

**Work:** expand independent events/regions when broad warning claims require them; de-duplicate event reports and account for catalogue selection bias; reserve a fresh holdout. For InSAR, select compatible SLC pairs and a small ROI/time span, budget disk/memory, use documented processing and assess real coherence/deformation uncertainty. Obtain specialist review if phase processing lies beyond current expertise.

**Acceptance:** additional examples are independent under the declared design, not more windows of the same event; expansion selection is score-blind; feasibility conclusions concern only measured processing/geometry/period. No arbitrary requested coherence/pair count must be made true. If resources exceed the Mac's safe budget, narrow the pilot or request an actual external resource decision before spending.

### WP12 — Submission review, release and maintenance

**Owner:** architect conducts final scientific review; implementor packages/replays; user decides venue/submission/public license.

**Depends on:** WP01–WP10 gates and any claim-dependent WP11.

**Outputs:** final evidence bundle, current factory audit/certification at the honest supported tier, reviewer objections/resolutions, venue-specific manuscript/supplement, code/data availability statements, license/attribution notices, release tag and long-term maintenance plan.

**Acceptance:** no unresolved P0 affects a published claim; all frozen artifacts/claims/figures and current reviews join; independent metrics reproduce; original/new checks pass; provider/license restrictions are respected; fresh checkout instructions work; limits/negative outcomes are explicit. A failed performance target is permitted; a failed evidence-integrity gate is not. Final venue fit and publication/acceptance are not mechanically guaranteed by the factory.

### 10.1 Dependency order and bounded effort

The critical path is WP00 → WP01 → WP03 → WP04 → WP05 → WP06 → WP07 → WP08 → WP09 → WP12. WP02 must complete before supervised research execution/certification. WP10 starts on a small verified fixture once WP06 exists and is repeated for the actual final pipeline only when needed. WP11 is conditional on claim feasibility.

Planning effort ranges are provisional, not measured promises: charter/review several focused days; factory integration several days to a few weeks depending on trust requirements; observational acquisition/features one to several weeks depending on provider/QA access; modeling/evaluation another several weeks; manuscript/independent review additional weeks. Provider failures, event adjudication and InSAR may dominate. Do not present these ranges as a timetable guaranteeing a top-tier paper. Prefer closing one complete gate with evidence over generating many partially implemented modules.

Keep `project/work_log.jsonl` with package ID, artifact/commit, tests, unresolved objections, status, actual elapsed/compute and next dependency. Maintain `project/READINESS.json` without optimistic percentages. It should say which scientific gates block which claims. A new agent reads those records before choosing the next task.

## 11. M3 resource and reproducibility plan

### 11.1 Design for the actual machine

The stated M3 Air has 16 GB shared CPU/GPU memory. Storage is nominal 512 GB, not wholly available to this project; an initial check during the audit showed about 114 GiB free. Recheck before acquisition. CPU/GPU activity, browser/OS use and thermal behavior share that budget. Do not assume the advertised 10 GPU cores provide CUDA or a discrete 16 GB accelerator. Use native arm64 packages, explicit device selection and a CPU reference.

Use remote/provider-side ROI reduction/export when scientifically appropriate, preserving the full query/product/QA lineage and sufficient independent samples for checking. Downloading all years of full Sentinel scenes is unnecessary for a compact feature study and can exceed storage. Conversely, provider-side reductions alone need independent small raster/image checks; a returned CSV cannot validate its own spatial processing. Keep raw immutable export chunks and manifests, derived reusable features, and generated attempts separate. Bulk raw products belong in ignored or external immutable storage, with release retrieval instructions.

The initial resource policy is conservative and adjustable by actual measurements: one neural training process at a time; begin CPU loader worker count at zero; use a small explicit batch such as 8–16 contexts; try d_model 64 before 128 if the scientific hypothesis permits; stream inference and kNN queries in bounded chunks. These are pilot engineering settings, not prescribed best-performing scientific choices. Batch changes after a failure create a recorded new attempt/configuration, not an invisible retry. Keep several GB available for the OS and fail/reschedule when measured pressure exceeds the chosen budget. Do not disable memory safety limits to force a run through.

### 11.2 Storage and memory estimates are calculations, not observations

For illustration only, 20 lakes × 3,227 days × 16 channels × four-byte values is about 4.13 MB of float32 values; the boolean validity matrix adds about 1.03 MB before provenance/quality and file overhead. Materializing 102 overlapping contexts per lake of 180 × 16 float32 elements is about 23.50 MB of values for 20 lakes, before masks/batches. Such feature tensors are small relative to raster/SLC products. Efficient window views or indexed loading avoid duplication, but do not change the number of independent measurements.

A standard attention score tensor grows as B × heads × T² elements per layer. At B=16, heads=8 and T=180, one float32 attention tensor is about 16.59 MB before gradients, projections, other layers, optimizer and framework workspace; this is only a component, not a claimed peak. Some fused kernels avoid materializing it, and actual implementation/device allocations must be measured. All-pairs kNN broadcasting grows with queries × bank × embedding dimension; the migrated scorer avoids that expression, but the production query/bank policy still needs a measured bound.

SLC/DEM/orbit/processing intermediates can be orders of magnitude larger. Inventory actual product sizes before an InSAR pilot; choose a small bounded subset and reserve disk space. Do not promise a whole-HKH InSAR archive fits the laptop. Store checkpoints without indiscriminately retaining every epoch, but keep best/final metadata and all attempt histories needed for audit. Never delete failed evidence merely to improve appearance or hide an unfavorable outcome.

### 11.3 CPU/MPS validation and timing

The audit environment reports Python 3.12 on arm64 and installed versions in `docs/audit/environment.json`. This is a local snapshot, not assurance that every dependency works under the factory runtime or on another machine. Verify `torch.backends.mps.is_available()` in the actual supported run and raise on an unavailable requested device. A CPU fallback must be an explicitly labeled new run, not concealed within a reported MPS result.

Compare representative CPU/MPS forward, masked loss, gradients and prediction ranks before choosing MPS for research. Exact equality is not assumed. Set a justified tolerance from measured numeric behavior, investigate discrepancies and freeze replay tolerance before final results. Mixed precision is a separate implementation/configuration choice requiring checks; keep float32 as a transparent starting point. Record backend nondeterminism and actual unsupported operations rather than claiming bitwise reproducibility by seeding alone.

For MPS timing, synchronize before/after measured sections. Record wall time, child process CPU time, process memory and device allocations separately. `torch.mps.current_allocated_memory()` reports tensor allocations and does not include all cached/driver/OS memory; it is not total peak unified-memory usage. Periodic sampling can miss transient peaks and must state interval/method. Current PyTorch documentation describes [MPS execution](https://docs.pytorch.org/docs/stable/notes/mps.html) and [MPS memory/synchronization APIs](https://docs.pytorch.org/docs/stable/mps.html); use documentation matching the installed version before relying on an API.

Sustained thermal behavior matters for the fanless Air. Report actual warm-up/trial durations, workload/context/batch, device, OS/runtime versions and whether acquisition/network time is excluded. Follow the factory's actual minimum trials/sustained duration if making a hardware claim. Do not repurpose a legacy training-summary duration as a new M3 measurement. Include variance and failures, not only the fastest trial.

### 11.4 Reproducible environment and release recipe

Convert `source/requirements.lock` from the installed-package snapshot to a clean verified dependency solution with transitive versions/hashes, Python/arm64 constraints, provider tools and runtime identity. Freeze the whole source including new files, not only an entrypoint. Avoid `.venv`, bytecode, editable external dependencies and unbound scripts inside frozen code. A larger acquisition/geospatial environment and a compact offline modeling environment may be useful, provided transformations and both runtimes are bound in provenance.

Capture OS/hardware/library/provider versions, requested/actual device, deterministic settings, seeds/RNG states, all data/schema/fit/configuration hashes and code commit. A seed alone does not reproduce inputs or remove nondeterminism. One actual clean local environment and fresh-process replay is the minimum engineering demonstration; a separate reviewer/device improves assurance but must be reported only if actually used.

Write installation, provider authentication, data retrieval, preprocessing, freeze/run/audit/replay and figure-generation instructions. Keep credentials outside Git and bundle exports. Network is needed for acquisition; finalized training/evaluation should use verified local immutable inputs where possible. An offline run remains scientifically invalid if the local inputs are fabricated. A release bundle contains manifests/checked results and reproducible recipes subject to provider/license terms, not a secret-rich copy of the entire working directory.

## 12. Submission preparation and literature work

### 12.1 Novelty review and venue fit

An elite venue requires a defensible contribution beyond having a working application. Do not start by promising a journal and forcing a story to fit. Choose the final contribution after authentic data, strong baseline comparisons, uncertainty and external scope are known. Negative findings can be publishable when the question is important, the design adequate, the evidence precise enough and the limitations explanatory; low model performance alone is not guaranteed publishable novelty.

Build a primary-source matrix for glacial-lake inventories/event databases, South Lhonak physical reconstruction, InSAR observability, lake area/glacier velocity/climate precursors, remote-sensing SSL/time-series anomaly detection, rare-event early-warning evaluation, missingness/shortcut learning and calibration. Columns include verified citation/DOI/year/version, task, population, raw source availability, spatial/temporal splits, support/latency assumptions, independent event/site counts, strong baselines, inference unit and contribution relative to this plan. Read methods/supplements, not just abstracts. Track source access limitations and unreviewed preprints explicitly.

A current related preprint discusses separating susceptibility from trigger timing and comparing deep models to strong simple baselines: [Kahn et al., submitted August 2026](https://arxiv.org/abs/2608.12422). Its reported dataset/performance should not be accepted or repeated as independently established facts in this project's claim ledger without examination/reproduction. Use it as a novelty challenge and a source of hypotheses/reference trails. Do not claim “first” based on the older project's bibliography.

Potential venue routes depend on resulting evidence:

| Route | What would strengthen fit | What remains insufficient |
|---|---|---|
| IEEE TGRS | Substantial remote-sensing methodology with convincing independent validation and exact measurement pipeline | Standard MAE+kNN integration with one event and fabricated/synthetic evidence presented as real |
| ISPRS Journal of Photogrammetry and Remote Sensing | Significant sensing/analysis contribution, theory-to-application connection, geographic transfer and reproducible evidence | A polished local demo or unexamined novelty claim |
| Remote Sensing of Environment | Strong environmental/physical measurement insight and appropriate independent validation | Treating anomaly ranking as demonstrated GLOF mechanism or actionable forecasting |
| Major ML conference research track | Generalizable methodological insight, competitive baselines and adequate multi-dataset/site evidence | A renamed standard architecture applied once without broader methodological support |
| Focused remote-sensing/hazard venue or workshop | Bounded benchmark/case/negative observability contribution with honest scope | Unsupported broad early-warning claims regardless of venue prestige |

These are conditional editorial assessments, not acceptance predictions, fee/deadline recommendations or rankings. Read current official requirements before formatting/submission: [TGRS author information](https://www.grss-ieee.org/publications/author-resources/tgrs-information-for-authors/), [IEEE GRSS author checklist](https://www.grss-ieee.org/publications/checklist-for-authors/), [ISPRS journal aims/scope](https://www.isprs.org/isprsjournal/). The [RSE author guide](https://www.sciencedirect.com/journal/remote-sensing-of-environment/publish/guide-for-authors) could not be fetched by this audit's browser; an agent must inspect it directly before asserting specific requirements. Likewise verify a chosen conference's current rules rather than assuming a past-year checklist.

### 12.2 Manuscript and supplement structure

Write the final paper around the question and evidence, not around past milestones. A suitable structure is: scientific target/contribution; verified data and selection; sensor support/availability and QA; exact method/fit/calibration; task-separated evaluation and independent units; results including failed targets/unestimable outcomes; physical/shortcut interpretation; limitations and conditional generalization; code/data availability and reproduction.

Report candidate/cohort flow, exclusions, source/feature counts, native supports, missingness/age, event uncertainty, fit/validation/calibration/test IDs and cutoffs, actual architecture/parameter count, training selection/budget/search, baseline configurations, thresholds/episode policies, all scientific denominators and precise uncertainty definitions. Include full seed-level metrics without pretending they are event replications. State whether event-case availability is true historical-as-of or an acquisition-time retrospective approximation.

The supplement should include schema/manifests/retrieval recipes, method and baseline configs, analytical tests/mutation checks, failure taxonomy, per-lake/season support and traces, actual ablation/sensitivity results, inferential assumptions/undefined replicates, reproducibility environment and resource records. Do not publish raw credentials, private signing material, personally identifying account logs or prohibited provider assets. Authors must agree on research-code/data reuse terms and attribution; the current proprietary license does not by itself offer an open replication license.

Figure candidates are actual ROI/product overlays, support/age and missingness heatmaps, real component-score and alarm timelines with decision/event uncertainty, Task-A paired performance, Task-C episodes/exposure/coverage, actual method ablations and calibration-vs-evaluation comparisons. A Task-B case without valid binary precursor labels does not need an invented ROC curve. Use actual per-window/date coordinates, correct physical units and meaningful uncertainty captions.

### 12.3 Submission-ready definition

Submission-ready means the chosen paper's claims have verified observational/simulation origins as stated, appropriate data/label/time/fit scopes, strong fair baselines, independent checks, defensible uncertainty, complete reproducibility artifacts and current scientific review. It does not mean a target metric was achieved or a certificate guarantees acceptance. If those conditions support only a scoped negative/case result, title/abstract/venue must reflect that scope.

Require review addressing method identity, data/leakage, training sufficiency, statistics, baseline fairness, ablation/sensitivity, generalization/failures, reproducibility and claims/venue. Record at least three concrete skeptical objections and their evidence-based resolutions or acknowledged limits, consistent with active factory review expectations. A self-authored PASS paragraph is not independent domain/statistical review. Involve a remote-sensing/glaciology specialist and a statistician where the claims require their expertise; explicitly disclose if that independent review has not happened.

## 13. Agent handoff contracts and final release checklist

### 13.1 Architect session brief

> Read root README, AGENTS.md, plan.md, project/READINESS.json, the audit records and active v3.3 factory policy. Work from this new folder only. Treat all legacy observations/weights/results as quarantined and all constructed tests as fixtures. First finish WP01 and specify WP02–WP05 contracts, with task-specific estimands, valid split/source support, claim limits and a feasible observational pilot. Resolve factory profile/runtime/receipt gaps through reviewed maintenance without weakening shared provenance gates. Produce concrete schemas, architecture decisions, statistical scope, work-package acceptance tests and a prioritized implementation handoff. Do not fabricate labels or favorable values to pass the factory. Preserve negative/inconclusive outcomes and record unresolved external facts. Do not freeze a research plan until real data feasibility and the domain runtime/validator are demonstrated.

The architect's deliverable is an actionable reviewed specification: proposed schemas, exact modules/interfaces, dependency order, claim-dependent blockers, actual references and acceptance fixtures. It should not merely paraphrase this plan or add a large model before fixing measurement and evaluation. Any change to a frozen scientific design creates an explicit amendment/epoch and preserves prior attempts. If the research target remains unclear after read-only work, ask the minimum necessary question while progressing independent work.

### 13.2 Implementor session brief

> Read root README, AGENTS.md, plan.md, project/READINESS.json, current work log and architect decisions. Implement the next approved dependency-complete work package with actual code and meaningful checks. Preserve the numerical core's explicit missingness, fit/test separation, causal availability, estimability and actual-execution contracts. Acquisition/provider failures must remain blocked/failed/no-data; no constants or simulated observations may appear as real measurements. Produce code, schemas/manifests, logs/tests and concrete review evidence. Update readiness and work log honestly. Do not change desired scientific outcomes into assertions, silently tune on final-test information, import the old folder, or manufacture a release certificate through a fixture/binary-profile workaround.

Every implementation handoff states changed files and purpose, verified behavior, commands actually executed, evidence origins, unresolved risks/claim blockers and the next dependency. Report partial work as partial. Do not declare an entire research package complete because a CLI prints SUCCESS or an output file exists. When a check fails, preserve the attempt and fix the causal defect; do not replace the check with a looser outcome requirement merely to pass.

### 13.3 Invariants for future agents

- No generated provider measurements, plausible physical fallbacks, fake confidence/quality or preselected performance numbers in observational paths.
- No automatic fitting on queries, test extrema normalization, post-event fitting for prospective claims, center-date decisions or backdated sustained alarms.
- No magic .5 AUROC, zero empty exposure/loss, fabricated no-alarm lead or fake CPU/memory telemetry.
- No mixed-task method comparison, ambiguous AP/PR area, fake perturbation labels, simulated statistical scores or invented figures.
- Every expected observation/prediction has identity, provenance, availability and status; every published number joins exact verified rows.
- Integrity gates accept correct poor results. Scientific targets may fail; evidence authenticity and methodological validity may not be substituted.
- A hash proves bytes, a test proves specified behavior, a receipt proves stated linkage, and a review records an assessment. None alone proves scientific truth.
- Keep this migration's implemented-core status distinct from future empirical completion. The backlog must shrink only when its acceptance evidence exists.

### 13.4 Final checklist

| Gate | Evidence required before publishing the affected claim |
|---|---|
| Origin | Real provider product/query/export identity; verified hashes; independent spot checks; explicit simulation/fixture separation |
| Geography/event | Verified persistent IDs/ROIs; sources/time uncertainty; selection and exclusions; interventions/control history |
| Measurement | Correct scale/offset/QA/CRS/support; real transformations; per-channel completeness/age; no fabricated substitutions |
| Temporal | Closed contexts; verified available inputs; fit cutoff; annual/pair support; actual declarations; purged overlaps |
| Fitting | Frozen train/development/calibration/test scope; exact architecture/normalization/PCA/bank/combiner lineage; real stopping status |
| Evaluation | Same task/cohort/labels/eligibility; fair strong baselines; actual interventions; complete rows/exposure/abstention |
| Inference | Independent recomputation; appropriate sampling units/pairing; precision/multiplicity; explicit unestimability |
| Factory | Supported reviewed domain/runtime; truthful receipt fields; scoped isolation/tier; failed attempts retained; current audit/review |
| Reporting | Real plots/tables; every claim joined; accurate citations/method descriptions; bounded negative/inconclusive wording |
| Reproduction | Clean verified environment/run; fresh replay/tolerances; source/data/config hashes; actual hardware methods if claimed |
| Release | Owner-selected code/data license; provider attribution/terms; secrets excluded; reproducible bundle/tag; current venue requirements |

Until all claim-dependent gates pass, status stays `RESEARCH_NOT_READY`. If a broad claim's cohort or evidence cannot be built, remove/narrow the claim instead of fabricating its prerequisite. User ambition for elite publication is a reason to demand more rigorous evidence, not a reason to conceal uncertainty.

## 14. Recovery and reference records

Local recovery data and tag targets are described in `docs/MIGRATION.md` and `.migration/backup_manifest.json`. The forensic inventories/selected excerpts remain in this repository so new-only sessions can understand the defects. They are not a source for new empirical results. Restore old Git tags into a separate directory only for historical analysis. Keep the current corrected source authoritative for future work.

Sources linked throughout this plan were checked during the 30 September–1 October 2026 audit/migration period where accessible. Dataset catalogue conventions and author requirements can change; future agents must use pinned product versions and recheck current requirements before implementation/submission. Internal findings arise from the inventoried source/artifacts rather than literature assertions. No external paper's reported accuracy is adopted as this project's result.

The completed migration should end with a verified new-folder Git checkout, tagged recoverable legacy states, a clean active branch, an honest core-test record and explicit remaining research blockers. It should never end with an assertion that this audit made Sentinel-GL scientifically complete or guaranteed submission acceptance.
