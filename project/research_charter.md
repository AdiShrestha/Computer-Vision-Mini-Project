# Sentinel-GL draft research charter

Version: draft-2026-10-01. Status: **NOT FROZEN; NO OBSERVATIONAL RESULTS; WP01 PARTIAL.** This is an implementation direction, not a completed methodology or a promise of publication. Read root plan.md, READINESS.json, and the critical report review. The preserved Deep Research report is advisory context; only explicitly accepted decisions here and future reviewed methodology govern execution.

## Research question and contribution route

Determine what a reproducible, resource-bounded satellite monitoring system can observe about glacial lake change, and whether multivariate anomaly scoring adds information beyond simple physical, seasonal, persistence and observation-quality baselines under the same available-data policy. The first empirical product is an observability and integrity pilot. A later primary study measures held-out monitoring burden and conditional retrospective event-case behavior. Event-population forecasting claims require a separately justified eligible cohort and independent replication.

Potential contribution: a verifiable measurement-to-decision pipeline and a fair assessment of where satellite anomaly monitoring succeeds, fails, or becomes unestimable. Novelty cannot be asserted from the project name or multimodal architecture. WP01 must compare prior studies by target, sensors, masks, historical availability, retrospective selection, control follow-up, splits, baselines, and inference unit. Mapping artifacts, physical changes, long-term susceptibility, event reconstruction and short-term hazard forecasts are distinct targets.

Acceptable outcomes include no reproducible pre-event signal; simpler baselines outperforming the MAE; acquisition age or missingness explaining apparent skill; inadequate support for some sensors; high control alert burden; and insufficient independent events for population inference. These outcomes require valid measurements and a valid design. Failure of acquisition, provenance, leakage or metric calculation is an implementation/design failure, not a negative scientific result.

## Phase A: actual observability pilot

Select a small, source-verified development set before inspecting model scores. WP03 proposes one event site and an independent candidate/control with seasonally distinct periods where available. These are candidate roles, not invented lake IDs, coordinates or confirmed safe lakes. Any site inspected during this pilot must be declared development data unless a reviewed split amendment transparently addresses its prior use.

For every proposed channel, obtain actual catalogue records and measurement bytes through a supported provider. Record product/version/platform, ROI/CRS, native support, scale/offset, QA, acquisition/support interval, processing/publication availability if known, actual retrieval/query/task identity, hashes, units, missingness and failure reasons. Document which observations can be independently checked from a provider example or raw spatial inspection. A successful provider request alone does not establish that a derived channel measures the proposed physical quantity.

Optical lake area needs a verified polygon and segmentation assessment, scale/offset/QA handling, appropriate water/snow/ice/cloud interpretation and pixel-area calculation in a justified projection or geodesic geometry. Surface reflectance is distinct from raw DN and top-of-atmosphere reflectance. A Landsat additive scale offset must be applied before NDWI. No real water pixel is required to equal a chosen index value. Use algebraic fixtures to check formulas and independent reviewed scene labels to assess measurement accuracy.

GRD backscatter is not complex interferometric phase, coherence or slope displacement. A claim-dependent SLC/InSAR branch needs its own orbit, coregistration, baseline, decorrelation, reference-frame and uncertainty review. Glacier image-pair velocities and annual mosaics retain their support intervals and availability. Reanalysis is spatially coarse forcing, not a measured local station series. Static terrain remains static; it must not become a generated time-varying predictor.

The pilot dossier reports actual usable acquisitions and age/coverage distributions, rejected products, extraction uncertainty, duplicates/shared supports, compute time, memory scope and disk bytes. It produces no final-test performance claim. If an intended modality is unsupported, narrow the feature schema/task or record a blocked branch. Never fill the gap with generated provider observations or a climatology mislabeled as measured weather.

## Phase B: frozen monitoring experiment

Before final scoring, finalize eligible lake/control/event records and split them into fit, model-development validation, threshold calibration and final evaluation scopes. Lake identity, spatial supports, acquisition IDs and temporal intervals must be independently joined. The allowed scope for each fitted normalizer, seasonal reference, PCA/reference bank, score transform, model and threshold is separate and byte-bound. A mapping containing approved lake names does not prove that its values came from those lakes.

Choose context length, decision calendar and useful warning horizon from actual support and the scientific question. The old 180-day context/30-day stride is only a candidate design. Choose eligibility and minimum per-channel support prospectively; two observed time positions are the current numerical minimum for masked reconstruction, not adequate scientific evidence by themselves. Missingness-only/calendar baselines and masked-scoring policy must be included in development diagnostics.

Each scheduled lake decision has an immutable identity, input source/support references, latest acquisition, latest required availability, actual declaration time, eligibility and exclusion reason. An unavailable decision remains a scheduled abstention in the ledger. Complete future inputs must not be used to emit a past decision. If historical provider publication time is not recoverable, declare a retrospective acquisition-time approximation and report that it does not establish historical operational latency.

Store raw score components, fitted-state/checkpoint hashes, support and per-channel errors. Scores are anomaly magnitudes, not event probabilities. Do not apply Brier/log-loss or binary probability gates to these values. Model checkpoint selection uses minimum observed development validation loss; `min_delta` controls patience. Record actual optimization and budget exhaustion. Lack of learning or convergence is disclosed and handled under a prospective policy; never invent loss curves.

## Alarm policy and estimands

An alarm policy must define comparator, threshold, sustained arrivals, maximum inter-decision gap, abstention reset, warm-up, opening/closing rules, hysteresis/refractory period and censoring. The current helper checks the first sustained pre-event declaration with explicit eligibility and gap limits. It does not implement the complete episode ledger or episode/exposure calibration. Implement those together in WP07/WP08 before claiming operational burden.

Freeze a bounded pre-event interval. An alarm credited to an event must be declared strictly before its verified onset within that interval. A permanent early alert cannot create unlimited lead-time credit. With date-only labels, exclude event-day inputs. With an event interval [earliest, latest], only declarations before the earliest bound establish a strictly pre-event warning; report lead-time bounds and ambiguous matches separately. Do not fabricate an onset hour from a nominal date.

Primary quantities for the monitoring route:

1. **Observability:** eligible scheduled decisions / all scheduled decisions, with per-lake, season, sensor and missingness reasons; input age and required-support distributions. Excluded follow-up remains visible.
2. **Uneventful alert burden:** correctly defined alert episodes over declared eligible monitoring exposure, and over total scheduled/calendar exposure where appropriate; report both exposure definitions, time in alert and coverage. A “control” means no verified event during the ascertained follow-up, not proof of zero risk.
3. **Conditional case behavior:** eligible support, detected/undetected/unestimable/ambiguous status, actual declaration and bounded lead time for each independently verified event. Report all selected cases, not only impressive detections.
4. **Incremental method comparison:** paired differences against fair baselines on the same decisions/exposure, with the conditional population and dependence structure stated. A seed comparison quantifies algorithm variability on the fixed corpus, not uncertainty across future events.

The current empirical threshold operator estimates an observed **window** crossing fraction on separate calibration IDs with comparator >= and exact tie handling. It does not guarantee a population false-alert rate or an episode-per-lake-year budget. Zero alerts over positive measured exposure is a valid zero. An absent eligible denominator is null with `NOT_ESTIMABLE`, not zero. Partial coverage must not silently become a missed event or a successful all-clear.

Event sensitivity denominators are verified selected events under a frozen eligibility policy. Provide excluded and unestimable cases as well as the conditional detectable subset. Lead-time summaries are conditional on detection and retain miss counts. Event precision requires one-to-one alarm-episode/event matching, a bounded horizon and complete event ascertainment; omit it if these conditions cannot be met.

## Fairness, dependence and uncertainty

The baseline set should start with calendar/seasonal references, missingness/age-only models, persistence or change rules, single-channel physical scores, and a justified classical detector. Each uses the same information cutoff and test decisions, its own permitted tuning/calibration scope, and disclosed input requirements. Compare natural deployment availability as well as a matched-support subset when methods require different modalities; report attrition in both views.

Build an explicit dependency graph for overlapping contexts, repeated event lakes, shared ROI pixels/products, shared weather cells, glacier systems and regional conditions. Changing daily rows to weekly rows does not create independent units. Shared years alone do not imply leakage. The split engine must distinguish duplicated measurement support from legitimate concurrent observations and from dependence affecting uncertainty. Do not assign unique fictional source IDs to evade a shared-support check.

With very few independent events, report cases and exact counts rather than a narrow bootstrap population interval. Any event/lake/region resampling needs an identified estimand, sufficient clusters and a defensible exchangeability argument. Repeated seeds are not additional events. Significance, equivalence, noninferiority and absence of evidence are different claims. Precision targets guide prospective feasibility and claim scope; failure to meet a target must narrow or block the intended inference, not change the results.

Shortcut/intervention studies remain explicit simulation or constructed experiments. Check correct intervention application and provenance, then report observed sensitivity. Do not require every perturbation to be detected, require small seed dispersion, or require shuffled/noise inputs to look normal. Same-seed fresh-process reproduction uses a prospectively justified numerical tolerance. CPU/MPS comparisons need their actual device and tolerance recorded; CPU fallback must not be reported as GPU execution.

## Compute and workflow gates

The target is the user's M3 MacBook Air with 16 GB unified memory and nominal 512 GB storage. Measure current free disk space and bounded export size before acquisition. Process ROI/windowed data and avoid caching unnecessary full scenes. A small input tensor does not bound attention activations, optimizer states, decoder memory or geospatial intermediates. Record actual run wall time, OS child CPU/RSS and separately synchronized accelerator/device memory where supported; unmeasured energy or total unified memory remains null.

The working Python 3.12 dependency snapshot is a test environment record, not a clean installation or a minimal hashed ARM64 lock. The locally patched factory requires cryptography for Ed25519. Its scientific domain profile, package-capable typed runtime and independent domain validators are still missing. Local receipts establish bounded local traceability, not sealed isolation. No research freeze or certificate is authorized by passing the current numerical/offline tests.

Remaining WP01 deliverables: independently appraised literature/novelty matrix, source-verified candidate-registry specification, task-specific statistical analysis specification, pilot selection rationale, independent-unit/precision feasibility and domain adjudication. WP02 must review local maintenance and complete the runtime/profile/validator path. WP03 must actually acquire and verify observations. The root research-plan template and methodology stay unfrozen until these dependency gates are met. External submission, new spending, credentials and release licensing follow the user's actual authorization.
