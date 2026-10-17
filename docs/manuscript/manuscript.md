# Sentinel-GL: 2023 retrospective multisensor analysis

The current submission manuscript reports 48 authentic optical acquisitions (24 per ROI) plus one shared September Sentinel-1 product. Fifty site-sensor observations correspond to 49 provider products. Full optical source tiles were not downloaded; bounded source ranges and native metadata retain exact receipts.

The model has 1,062,919 parameters and five selected seed checkpoints. Every one of the 188 source-supported parity-1 quadrants per seed is scored; 12 unsupported quadrants remain missing. Frozen parity-0 pre-breach fitting, optical dependency closure and causal per-query contexts are explicit. Earlier pre-period trajectories remain development-exposed.

| Method | South pre | South post | Khangchung pre | Khangchung post |
|---|---:|---:|---:|---:|
| T-MAE A | 0.0477 | 0.0442 | 0.0359 | 0.0244 |
| T-MAE B | 1.0086 | 3.1894 | 1.3871 | 2.9591 |
| T-MAE C | -0.3062 | 0.2853 | -0.3217 | 0.0454 |
| Isolation Forest | 0.4277 | 0.4890 | 0.4189 | 0.4476 |
| One-Class SVM | -0.1898 | 0.0394 | -0.1897 | -0.0396 |
| RPCA residual | 0.0515 | 0.0368 | 0.0427 | 0.0324 |
| CUSUM | 0.0000 | 0.2171 | 0.0000 | 0.0000 |

Values are medians in method-specific units, using seed medians per acquisition for learned methods. South Lhonak has 21/3 usable pre/post optical acquisitions; Khangchung has 19/4. Reconstruction does not show a South Lhonak post-event spike. Latent scores rise at both sites and were elevated before October; these are not independent precursor confirmations.

Optical RPCA converges in 413 iterations (rank 3, primal 9.94500937e-06, dual 1.49467717e-06); SAR converges in 1 iteration on two same-date site rows.

Scientific limits remain explicit: two developmental query ROIs; no validated shoreline masks; changing SCL support and signed-ratio instability; ellipsoid SAR geometry without DEM terrain correction; one SAR date; unknown independent physical support and historical operational availability. No independent event/control labels or annual exposure establish PR-AUC, lead time, or calibrated annual false-alarm burden. Attention is internal model structure, not causal attribution.

The main source is `sentinel_gl_submission.tex`; `sentinel_gl_ictceel2027.tex` provides the alternative venue layout. Both describe the same retrospective experiment.
