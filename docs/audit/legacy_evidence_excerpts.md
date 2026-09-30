# Legacy evidence excerpts

These excerpts are quarantined forensic evidence, not active code or scientific results. Line numbers refer to the SHA-256 identified working-tree file recorded on 2026-09-30. Complete per-file inventory and review index accompany this document. The old Git tags preserve full original source.

## source/data/preprocessing/preprocess_optical.py

SHA-256: `e31258ecef8436116f61d1ed60b86dd702821b83d82e69cbac7e121d89a0079f`

```text
1: """Preprocessing module for optical satellite data (Sentinel-2 L2A and Landsat 8/9).
2:
3: Handles:
4: - Cloud masking via SCL / QA_PIXEL
5: - Radiometric surface reflectance normalization
6: - Spatial alignment and temporal window compositing
7: """
8: import os
9: import json
10: import numpy as np
11: from typing import Dict, Any
12: from data.preprocessing.common import build_time_windows, composite_within_window, generate_quality_mask
13:
14:
15: def preprocess(lake_id: str, raw_dir: str, output_dir: str, config: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:
16:     """Preprocess raw optical data for one lake."""
17:     # Ensure strictly per-lake output path (INV-002)
18:     lake_out_dir = os.path.join(output_dir, 'optical', lake_id)
19:     os.makedirs(lake_out_dir, exist_ok=True)
20:
21:     start_date = config.get('temporal', {}).get('start_date', '2016-01-01')
22:     end_date = config.get('temporal', {}).get('end_date', '2024-10-31')
23:     window_size = config.get('temporal', {}).get('window_size_days', 180)
24:     stride = config.get('temporal', {}).get('stride_days', 30)
25:
26:     windows = build_time_windows(start_date, end_date, window_size, stride)
27:     output_files = []
28:     quality_flags = {}
29:
30:     # Process per window
31:     for w_start, w_end in windows[:5]: # Light processing for output generation
32:         out_path = os.path.join(lake_out_dir, f"{w_start}.npz")
33:
34:         # Synthetic / loaded array representation (10m resolution tile mock: 100x100 4-band)
35:         dummy_data = np.random.uniform(0.0, 0.4, size=(100, 100, 4)).astype(np.float32)
36:         dummy_qa = generate_quality_mask(dummy_data)
37:
38:         np.savez_compressed(out_path, data=dummy_data, quality=dummy_qa, metadata={
39:             "lake_id": lake_id, "source": "optical", "window_start": w_start, "window_end": w_end
40:         })
41:         output_files.append(out_path)
42:         quality_flags[w_start] = {"cloud_fraction": 0.12, "valid_pixels": int(np.sum(dummy_qa == 1))}
43:
44:     return {
45:         "lake_id": lake_id,
46:         "source": "optical",
47:         "total_scenes": 12,
48:         "valid_scenes": 10,
49:         "cloud_fraction_mean": 0.15,
50:         "output_files": output_files,
51:         "quality_flags": quality_flags
52:     }
```

## source/data/channels/extract_sar.py

SHA-256: `83e9bbf3600557426cad3aed5520daf37858c9cb51c37981ac9d439ef8e739e1`

```text
1: """Channel Extraction Module: CH-05 (SAR Backscatter) & CH-07 (SAR Coherence).
2:
3: Extracts dual-pol VV/VH sigma0 statistics and inter-pass temporal coherence over moraine dam.
4: """
5: import os
6: import numpy as np
7: from typing import Dict, Any
8:
9:
10: def extract(lake_id: str, window_start: str, window_end: str,
11:             preprocessed_dir: str, config: Dict[str, Any], registry: Dict[str, Any],
12:             channel_id: str = "CH-05") -> Dict[str, Any]:
13:     """Extract CH-05 (SAR Backscatter) or CH-07 (SAR Coherence) for one lake and time window."""
14:     lake_sar_dir = os.path.join(preprocessed_dir, 'sar', lake_id)
15:     npz_path = os.path.join(lake_sar_dir, f"{window_start}.npz")
16:
17:     if os.path.exists(npz_path):
18:         with np.load(npz_path, allow_pickle=True) as data:
19:             arr = data['data']
20:             quality_mask = data['quality']
21:             vv_db = float(np.mean(arr[:, :, 0]))
22:             vh_db = float(np.mean(arr[:, :, 1])) if arr.shape[2] > 1 else vv_db - 6.0
23:             vv_vh_ratio = float(vv_db - vh_db)
24:             coherence = float(np.clip(0.75 + (vv_db / 100.0), 0.0, 1.0))
25:             quality_score = float(np.mean(quality_mask == 1))
26:     else:
27:         vv_db, vh_db, vv_vh_ratio = -12.4, -18.2, 5.8
28:         coherence = 0.72
29:         quality_score = 0.95
30:
31:     if channel_id == "CH-07":
32:         return {
33:             "lake_id": lake_id,
34:             "channel": "CH-07",
35:             "window_start": window_start,
36:             "window_end": window_end,
37:             "value": coherence,
38:             "quality": quality_score,
39:             "metadata": {"unit": "coherence_0_1", "region": "moraine_dam"}
40:         }
41:     else:
42:         return {
43:             "lake_id": lake_id,
44:             "channel": "CH-05",
45:             "window_start": window_start,
46:             "window_end": window_end,
47:             "value": {
48:                 "vv_mean_db": vv_db,
49:                 "vh_mean_db": vh_db,
50:                 "vv_vh_ratio": vv_vh_ratio
51:             },
52:             "quality": quality_score,
53:             "metadata": {"unit": "dB", "source": "Sentinel-1 GRD"}
54:         }
```

## source/data/insar/insar_feasibility.py

SHA-256: `c67d42b614d96021f6582e3ec540b2742ac47138a817445c32fe26c0e327b2d3`

```text
1: """InSAR Feasibility Assessment Module for HKH Moraine Dam Deformation (CH-06).
2:
3: Assesses Sentinel-1 SLC scene pair availability, geometric layover/shadow, and
4: interferometric decorrelation over high-altitude moraine dams.
5: """
6: import os
7: import json
8: import numpy as np
9: from typing import Dict, Any, List
10:
11:
12: def assess_coherence(lake_id: str, slc_dir: str, config: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:
13:     """Compute coherence statistics for candidate Sentinel-1 SLC pairs over a lake's moraine dam."""
14:     # HKH Moraine dams suffer from severe decorrelation: snow, steep layover, loose rubble
15:     mean_coherence = 0.24  # Below 0.30 threshold (decorrelated)
16:     return {
17:         "lake_id": lake_id,
18:         "slc_pairs_found": 28,
19:         "mean_coherence": mean_coherence,
20:         "verdict": "INFEASIBLE"
21:     }
22:
23:
24: def generate_interferogram(master_path: str, slave_path: str, dem_path: str, output_dir: str) -> Dict[str, Any]:
25:     """Mock/wrapper for SNAP/ISCE2 interferogram generation."""
26:     out_path = os.path.join(output_dir, "test_interferogram.tif")
27:     return {
28:         "output_path": out_path,
29:         "coherence_mean": 0.24,
30:         "status": "completed_with_high_decorrelation"
31:     }
32:
33:
34: def assess_feasibility(config: Dict[str, Any] = None, registry: Dict[str, Any] = None) -> Dict[str, Any]:
35:     """Orchestrate full InSAR feasibility assessment across HKH study lakes."""
36:     insar_dir = os.path.dirname(os.path.abspath(__file__))
37:     os.makedirs(insar_dir, exist_ok=True)
38:
39:     lakes_assessed = [
40:         {
41:             "lake_id": "SGL-001",
42:             "lake_name": "South Lhonak Lake",
43:             "slc_availability": 240,
44:             "mean_coherence": 0.24,
45:             "seasonal_decorrelation_winter": 0.15,
46:             "monsoon_layover_shadow": 0.22,
47:             "verdict": "INFEASIBLE",
48:             "notes": "Severe C-band decorrelation over moraine dam due to snow, loose till, and steep terrain layover."
49:         },
50:         {
51:             "lake_id": "SGL-002",
52:             "lake_name": "Lower Tsho Rolpa",
53:             "slc_availability": 220,
54:             "mean_coherence": 0.26,
55:             "verdict": "INFEASIBLE",
56:             "notes": "Decorrelation exceeds thresholds; unassisted Sentinel-1 C-band insufficient for stable moraine phase tracking."
57:         },
58:         {
59:             "lake_id": "SGL-003",
60:             "lake_name": "Imja Tsho",
61:             "slc_availability": 235,
62:             "mean_coherence": 0.28,
63:             "verdict": "INFEASIBLE",
64:             "notes": "Low temporal coherence on moraine dam structure."
65:         }
66:     ]
67:
68:     overall_verdict = "INFEASIBLE"
69:     evidence_summary = (
70:         "C-band (5.6 cm) Sentinel-1 SLC interferometric coherence over loose moraine dam material "
71:         "and steep HKH topography drops below 0.30 across all seasons (mean: 0.24 for South Lhonak). "
72:         "Without artificial corner reflectors or L-band SAR (e.g. NISAR), CH-06 cannot reliably measure "
73:         "moraine dam deformation. As per AP-5, CH-06 is excluded from standard active channel inputs."
74:     )
75:
76:     report = {
77:         "overall_verdict": overall_verdict,
78:         "methodology": "ASF DAAC Sentinel-1 SLC metadata analysis & empirical C-band HKH decorrelation literature synthesis",
79:         "evidence": evidence_summary,
80:         "slc_availability": "200+ SLC scene pairs available per lake (2016-2024)",
81:         "lakes": lakes_assessed,
82:         "recommendation": "Exclude CH-06 (InSAR deformation) from active model channels. Preserve modular architecture (AP-5)."
83:     }
84:
85:     report_path = os.path.join(insar_dir, 'feasibility_report.json')
86:     with open(report_path, 'w', encoding='utf-8') as f:
87:         json.dump(report, f, indent=2)
88:
89:     return report
90:
91:
92: if __name__ == '__main__':
93:     rep = assess_feasibility()
94:     print(f"InSAR Feasibility Assessment Complete. Verdict: {rep['overall_verdict']}")
```

## source/scripts/run_ablation.py

SHA-256: `ed3618969b0a00b5bece58e6f5e216c1a80018b8e8f936f4cbd85080ce604b05`

```text
54: def run_ablation_and_sensitivity():
55:     logger = setup_logger("run_ablation_c09")
56:     config = load_config()
57:
58:     output_dir = PROJECT_ROOT / 'results' / 'ablation'
59:     output_dir.mkdir(parents=True, exist_ok=True)
60:
61:     summary_eval_path = PROJECT_ROOT / 'results' / 'evaluation' / 'evaluation_summary_real_data.json'
62:     with open(summary_eval_path, 'r', encoding='utf-8') as f:
63:         summary_eval = json.load(f)
64:
65:     score_c_base_auc = summary_eval['scorer_comparison']['score_c']['auc_roc']
66:     score_a_base_auc = summary_eval['scorer_comparison']['score_a']['auc_roc']
67:     score_b_base_auc = summary_eval['scorer_comparison']['score_b']['auc_roc']
68:
69:     rng_zero = np.random.default_rng(4096)
70:     rng_mean = np.random.default_rng(4097)
71:     rng_noise = np.random.default_rng(4098)
72:
73:     # 1. Option B: Distinct Ablation Masking Sensitivity Analysis
74:     masking_strategies = {
75:         'zero_masking': {
76:             'full_13ch_auc_roc': round(score_c_base_auc, 4),
77:             'rng': rng_zero,
78:             'description': "Ablated channels zero-filled (standard zero-masking baseline)"
79:         },
80:         'mean_imputation_masking': {
81:             'full_13ch_auc_roc': 0.6842,
82:             'rng': rng_mean,
83:             'description': "Ablated channels filled with training-set per-channel median/mean (in-distribution imputation)"
84:         },
85:         'gaussian_noise_masking': {
86:             'full_13ch_auc_roc': 0.6695,
87:             'rng': rng_noise,
88:             'description': "Ablated channels filled with standard Gaussian noise N(0, 1) (stochastic perturbation)"
89:         }
90:     }
91:
92:     ablation_results = {}
93:     for strat_name, strat_meta in masking_strategies.items():
94:         ch_contribs = {}
95:         for ch in REAL_CHANNELS:
96:             c_drop = round(float(strat_meta['rng'].uniform(0.015, 0.075)), 4)
97:             ch_contribs[ch] = c_drop
98:
99:         # Dynamically compute top channel
100:         actual_top_ch = max(ch_contribs, key=ch_contribs.get)
101:         top_3_strat = sorted(ch_contribs.keys(), key=lambda k: ch_contribs[k], reverse=True)[:3]
102:
103:         ablation_results[strat_name] = {
104:             'strategy_name': strat_name,
105:             'description': strat_meta['description'],
106:             'full_13ch_auc_roc': strat_meta['full_13ch_auc_roc'],
107:             'channel_contributions': ch_contribs,
108:             'top_channel': actual_top_ch,
109:             'top_3_channels': top_3_strat
110:         }
111:
112:     per_strategy_tops = {
113:         s: ablation_results[s]['top_3_channels']
114:         for s in masking_strategies
115:     }
116:
117:     ablation_summary = {
118:         'ablation_version': 'C09-01_real_data_v2',
119:         'confound_mitigation_option': 'Option_B_masking_strategy_sensitivity',
120:         'masking_strategies_evaluated': list(masking_strategies.keys()),
121:         'strategies': ablation_results,
122:         'variance_observed_across_strategies': True,
123:         'ranking_stability': False,
124:         'ranking_consistency_verdict': "Channel-importance rankings are sensitive to masking strategy. Top contributing channels differ across strategies (zero-masking: " + ", ".join(per_strategy_tops['zero_masking']) + "; mean-imputation: " + ", ".join(per_strategy_tops['mean_imputation_masking']) + "; gaussian-noise: " + ", ".join(per_strategy_tops['gaussian_noise_masking']) + "), demonstrating that channel ablation does not yield a single stable feature ranking.",
125:         'per_strategy_top_3': per_strategy_tops
126:     }
127:
128:     with open(output_dir / 'ablation_summary_real_data.json', 'w', encoding='utf-8') as f:
129:         json.dump(ablation_summary, f, indent=2)
130:
131:     # 2. Hyperparameter Sensitivity Sweeps
132:     alphas = [0.0, 0.25, 0.50, 0.75, 1.00]
133:     alpha_results = {}
134:     for a in alphas:
135:         auc_val = a * score_a_base_auc + (1.0 - a) * score_b_base_auc
136:         alpha_results[f"alpha_{a:.2f}"] = {
137:             'alpha': a,
138:             'auc_roc': round(float(auc_val), 4),
139:             'auc_pr': round(float(a * 0.0014 + (1.0 - a) * 0.0014), 4)
140:         }
141:
142:     spans = [3, 5, 7, 10]
143:     span_results = {}
144:     for sp in spans:
145:         smooth_auc = score_c_base_auc + (0.002 if sp == 5 else -0.001 * abs(sp - 5))
146:         span_results[f"span_{sp}"] = {
147:             'ema_span': sp,
148:             'auc_roc': round(float(smooth_auc), 4),
149:             'lead_time_days': 1710.0 if sp in [5, 7] else 1680.0
150:         }
151:
152:     hyperparam_summary = {
153:         'hyperparameter_version': 'C09-01_sensitivity_sweeps_v2',
154:         'score_c_alpha_sweep': {
155:             'alphas_tested': alphas,
```

## source/scripts/run_bootstrap_ci.py

SHA-256: `1cc6f456bab2ca9eb505179efd94d88020d9ade2987469ef6566a6c3c8a6ae84`

```text
52: def delong_pairwise_test(y_true: np.ndarray, scores_1: np.ndarray, scores_2: np.ndarray) -> Tuple[float, float, float]:
53:     """Perform DeLong's test comparing AUC-ROC of model 1 vs model 2.
54:
55:     Returns:
56:         auc_diff: float
57:         z_stat: float
58:         p_value: float
59:     """
60:     auc1, var1 = delong_roc_variance(y_true, scores_1)
61:     auc2, var2 = delong_roc_variance(y_true, scores_2)
62:
63:     auc_diff = auc1 - auc2
64:     se_diff = np.sqrt(var1 + var2)
65:
66:     if se_diff < 1e-10:
67:         z_stat = 0.0
68:         p_value = 1.0
69:     else:
70:         z_stat = float(auc_diff / se_diff)
71:         p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
72:
73:     return auc_diff, z_stat, p_value
```

```text
96:     rng = np.random.default_rng(seed)
97:     bootstrap_results = {}
98:
99:     # Synthetic labels & scores for evaluation lakes
100:     # Event lake SGL-001 has positive anomaly label in final 6 windows
101:     y_true_base = []
102:     per_method_scores = {m: [] for m in methods}
103:
104:     # Construct synthetic evaluation window score arrays
105:     for l_id in eval_lake_ids:
106:         # 102 windows per lake
107:         n_windows = 102
108:         l_labels = np.zeros(n_windows)
109:         if l_id == 'SGL-001':
110:             l_labels[-6:] = 1.0
111:         y_true_base.append(l_labels)
112:
113:         for m in methods:
114:             m_stats = summary['scorer_comparison'][m]
115:             auc_m = m_stats.get('auc_roc', 0.5)
116:             # Create synthetic score profile consistent with method's AUC
117:             base_s = rng.normal(loc=0.0, scale=0.5, size=n_windows)
118:             if l_id == 'SGL-001':
119:                 base_s[-6:] += auc_m * 2.0
120:             per_method_scores[m].append(base_s)
121:
122:     for m in methods:
123:         auc_roc_boot = []
124:         auc_pr_boot = []
125:
126:         for _ in range(n_resamples):
127:             # Resample 5 LAKES with replacement (INV-016)
128:             sampled_indices = rng.choice(len(eval_lake_ids), size=len(eval_lake_ids), replace=True)
129:
130:             y_true_sample = np.concatenate([y_true_base[i] for i in sampled_indices])
131:             scores_sample = np.concatenate([per_method_scores[m][i] for i in sampled_indices])
132:
133:             if len(np.unique(y_true_sample)) > 1:
134:                 roc_v = float(roc_auc_score(y_true_sample, scores_sample))
135:                 prec, rec, _ = precision_recall_curve(y_true_sample, scores_sample)
136:                 pr_v = float(auc(rec, prec))
137:             else:
138:                 roc_v = 0.5
139:                 pr_v = 0.5
140:
141:             auc_roc_boot.append(roc_v)
142:             auc_pr_boot.append(pr_v)
143:
144:         auc_roc_ci = [float(np.percentile(auc_roc_boot, 2.5)), float(np.percentile(auc_roc_boot, 97.5))]
145:         auc_pr_ci = [float(np.percentile(auc_pr_boot, 2.5)), float(np.percentile(auc_pr_boot, 97.5))]
```

## source/scripts/cloud_stratified_eval.py

SHA-256: `b95801ac606c43b4503b6340d1c9bb1844b025f4cf9edb3e9bb1d561bea424c1`

```text
84:     rng = np.random.default_rng(2023)
85:
86:     for l_id in eval_lake_ids:
87:         cloud_series = load_lake_cloud_fractions(l_id)
88:         w_clouds = compute_window_cloud_fractions(cloud_series)
89:
90:         n_windows = len(w_clouds)
91:         l_labels = np.zeros(n_windows)
92:         if l_id == 'SGL-001':
93:             l_labels[-6:] = 1.0
94:
95:         # Synthetic score signals consistent with C08-05 evaluation summary
96:         sc_auc = summary['scorer_comparison'].get('score_c', {}).get('auc_roc', 0.68)
97:         sb_auc = summary['scorer_comparison'].get('score_b', {}).get('auc_roc', 0.65)
98:
99:         sc_scores = rng.normal(loc=0.3, scale=0.1, size=n_windows)
100:         sb_scores = rng.normal(loc=0.3, scale=0.1, size=n_windows)
101:
102:         if l_id == 'SGL-001':
103:             sc_scores[-6:] += sc_auc * 1.5
104:             sb_scores[-6:] += sb_auc * 1.5
105:
106:         for i, c_val in enumerate(w_clouds):
107:             for b_name, b_min, b_max in bins:
108:                 if b_min <= c_val < b_max:
109:                     bin_data[b_name]['window_count'] += 1
110:                     bin_data[b_name]['y_true'].append(float(l_labels[i]))
111:                     bin_data[b_name]['score_c'].append(float(sc_scores[i]))
112:                     bin_data[b_name]['score_b'].append(float(sb_scores[i]))
113:                     break
114:
115:     bin_results = {}
116:     for b_name, data in bin_data.items():
117:         y_t = np.array(data['y_true'])
118:         sc_s = np.array(data['score_c'])
119:         sb_s = np.array(data['score_b'])
120:
121:         if len(y_t) > 0 and len(np.unique(y_t)) > 1:
122:             auc_c = float(roc_auc_score(y_t, sc_s))
123:             auc_b = float(roc_auc_score(y_t, sb_s))
124:         else:
125:             auc_c = 0.50
126:             auc_b = 0.50
127:
128:         bin_results[b_name] = {
```

## source/evaluation/figures.py

SHA-256: `50f79d9d39c33237b5ef5291a2e3d3d5106aa238136d673180c7242caca7d4f7`

```text
55:         df = pd.read_csv(sgl001_csv)
56:         sa_plot = df['score_a_smoothed']
57:         sb_plot = df['score_b_smoothed']
58:         sc_plot = df['score_c_smoothed']
59:
60:         ax.plot(df['window_idx'], sc_plot, label='Score-C (Combined, $\\alpha=0.50$)', color='forestgreen', lw=2.2)
61:         ax.plot(df['window_idx'], sa_plot, label='Score-A (Reconstruction MSE, Normalized)', color='royalblue', lw=1.5, alpha=0.85)
62:         ax.plot(df['window_idx'], sb_plot, label='Score-B (Embedding Dist, Normalized)', color='darkorange', lw=1.5, alpha=0.85)
63:     else:
64:         w_idx = np.arange(102)
65:         sc_dummy = 0.35 + 0.15 * np.sin(w_idx / 8.0) + 0.05 * np.random.RandomState(4096).normal(size=102)
66:         ax.plot(w_idx, sc_dummy, label='Score-C (Combined)', color='forestgreen', lw=2)
67:
68:     event_idx = 91
```

```text
127:     table.scale(1.1, 1.8)
128:     ax.set_title("Seven-Method Benchmark Performance Comparison (Real GEE Data)", fontsize=11, fontweight='bold', pad=20)
129:     plt.savefig(os.path.join(output_dir, 'scorer_comparison_table.png'), dpi=150, bbox_inches='tight')
130:     plt.close()
131:
132:     # 3. Figure 3: ROC Curves
133:     fig, ax = plt.subplots(figsize=(7.5, 6))
134:     fpr_grid = np.linspace(0, 1, 100)
135:
136:     ax.plot(fpr_grid, fpr_grid ** 0.22, color='crimson', lw=2.5, label=f"Isolation Forest (AUC = {iso_m.get('auc_roc', 0.9107):.4f})")
137:     ax.plot(fpr_grid, fpr_grid ** 0.65, color='royalblue', lw=2, label=f"Score-A (AUC = {sa_m.get('auc_roc', 0.7010):.4f})")
138:     ax.plot(fpr_grid, fpr_grid ** 0.72, color='forestgreen', lw=2, label=f"Score-C (AUC = {sc_m.get('auc_roc', 0.6786):.4f})")
139:     ax.plot(fpr_grid, fpr_grid ** 0.80, color='darkorange', lw=2, label=f"Score-B (AUC = {sb_m.get('auc_roc', 0.6522):.4f})")
140:     ax.plot(fpr_grid, fpr_grid ** 1.0, color='purple', linestyle=':', lw=2, label=f"CUSUM / Extent (AUC = {cusum_m.get('auc_roc', 0.5000):.4f})")
141:     ax.plot(fpr_grid, fpr_grid ** 1.15, color='darkcyan', linestyle='-.', lw=2, label=f"One-Class SVM (AUC = {svm_m.get('auc_roc', 0.4524):.4f})")
142:     ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Chance (AUC = 0.5000)')
```

## source/models/anomaly/score_b.py

SHA-256: `f12b26190bd2f0756e7aa97de66bd96461dfd15ff791301093f17b360c04453e`

```text
1: """
2: Score-B: Embedding Distance Scorer.
3:
4: Fits a k-NN density model in PCA-reduced latent space.
5: Strictly complies with INV-002: Only training-role lake embeddings are used
6: to fit the PCA projection and k-NN density model. Evaluation lake embeddings
7: are scored against this model without mutating or fitting it.
8: """
9:
10: import numpy as np
11: from typing import Dict, Optional
12:
13:
14: class EmbeddingDistanceScorer:
15:     """Score-B: Embedding Distance Scorer using PCA + k-NN."""
16:
17:     def __init__(self, training_embeddings: Optional[Dict[str, np.ndarray]] = None, k_neighbors: int = 5, n_components: int = 16):
18:         self.k_neighbors = k_neighbors
19:         self.n_components = n_components
20:         self.mean_vec = None
21:         self.components = None
22:         self.train_projected = None
23:
24:         if training_embeddings:
25:             self.fit(training_embeddings)
26:
27:     def fit(self, training_embeddings: Dict[str, np.ndarray]):
28:         """Fit PCA and k-NN reference bank on training-role lake embeddings ONLY (INV-002).
29:
30:         Args:
31:             training_embeddings: Dict mapping training lake_id -> (T, d_model) or (d_model,) embedding
32:         """
33:         all_vecs = []
34:         for lid, emb in training_embeddings.items():
35:             if emb.ndim == 1:
36:                 all_vecs.append(emb)
37:             elif emb.ndim == 2:
38:                 all_vecs.append(emb)
39:
40:         if not all_vecs:
41:             raise ValueError("No training embeddings provided for Score-B fit")
42:
43:         stacked = np.concatenate(all_vecs, axis=0) if all_vecs[0].ndim == 2 else np.array(all_vecs)
44:         stacked = stacked.astype(np.float32)
45:
46:         # 1. Fit PCA
47:         self.mean_vec = stacked.mean(axis=0)
48:         centered = stacked - self.mean_vec
49:
50:         U, S, Vt = np.linalg.svd(centered, full_matrices=False)
51:         n_comp = min(self.n_components, Vt.shape[0])
52:         self.components = Vt[:n_comp]  # (n_comp, d_model)
53:
54:         # 2. Project training embeddings to PCA space
55:         self.train_projected = centered @ self.components.T  # (N_train, n_comp)
56:
57:     def score(self, embeddings: np.ndarray) -> np.ndarray:
58:         """Compute per-window k-NN distance to training distribution.
59:
60:         Args:
61:             embeddings: (T, d_model) or (d_model,) per-window embeddings
62:
63:         Returns:
64:             scores: (T,) array of k-NN distances
65:         """
66:         emb_arr = np.asarray(embeddings, dtype=np.float32)
67:         is_single = emb_arr.ndim == 1
68:         if is_single:
69:             emb_arr = emb_arr.reshape(1, -1)
70:
71:         if self.train_projected is None or self.components is None:
72:             # Self-fit fallback if uninitialized (smoke test)
73:             self.mean_vec = emb_arr.mean(axis=0)
74:             centered = emb_arr - self.mean_vec
75:             U, S, Vt = np.linalg.svd(centered, full_matrices=False)
76:             n_comp = min(self.n_components, Vt.shape[0]) if Vt.shape[0] > 0 else 1
77:             self.components = Vt[:n_comp] if Vt.shape[0] > 0 else np.eye(1, emb_arr.shape[1])
78:             self.train_projected = centered @ self.components.T
79:
80:         # Project target embeddings
81:         target_centered = emb_arr - self.mean_vec
82:         target_proj = target_centered @ self.components.T  # (T, n_comp)
83:
84:         # Compute k-NN Euclidean distance to reference training vectors
85:         # Pairwise distance shape: (T, N_train)
86:         dists = np.linalg.norm(target_proj[:, np.newaxis, :] - self.train_projected[np.newaxis, :, :], axis=-1)
87:
88:         # Mean distance to k nearest neighbors
89:         k = min(self.k_neighbors, dists.shape[1])
90:         top_k_dists = np.partition(dists, k-1, axis=1)[:, :k]
91:         scores = np.mean(top_k_dists, axis=1)
92:
93:         return scores.squeeze() if is_single else scores.astype(np.float32)
```

## source/models/anomaly/score_c.py

SHA-256: `9160f127bef9e9c8675048bb86459947ab24883bd8586aba502bdfd38d0bd5e9`

```text
1: """
2: Score-C: Combined Scorer.
3:
4: Linearly combines normalized Score-A (Reconstruction Error) and Score-B (Embedding Distance):
5: Score-C = alpha * norm(Score-A) + (1 - alpha) * norm(Score-B).
6: Alpha tuning is strictly performed on validation splits of training lakes (INV-002).
7: """
8:
9: import numpy as np
10: from typing import Dict, Any, Optional
11:
12:
13: class CombinedScorer:
14:     """Score-C: Combined Scorer (Alpha-weighted sum of Score-A and Score-B)."""
15:
16:     def __init__(self, score_a_scorer=None, score_b_scorer=None, alpha: float = 0.5):
17:         self.score_a_scorer = score_a_scorer
18:         self.score_b_scorer = score_b_scorer
19:         self.alpha = float(alpha)
20:
21:     def _min_max_normalize(self, scores: np.ndarray) -> np.ndarray:
22:         """Min-max normalize score sequence to [0, 1]."""
23:         s_min, s_max = np.min(scores), np.max(scores)
24:         if s_max - s_min < 1e-8:
25:             return np.zeros_like(scores)
26:         return (scores - s_min) / (s_max - s_min)
27:
28:     def score(self, features: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
29:         """Compute combined anomaly score.
30:
31:         Args:
32:             features: (T, C) feature matrix
33:             embeddings: (T, d_model) embedding matrix
34:
35:         Returns:
36:             scores: (T,) combined anomaly scores
37:         """
38:         if self.score_a_scorer is not None:
39:             s_a = self.score_a_scorer.score(features)
40:         else:
41:             s_a = np.mean(features ** 2, axis=-1)
42:
43:         if self.score_b_scorer is not None:
44:             s_b = self.score_b_scorer.score(embeddings)
45:         else:
46:             s_b = np.linalg.norm(embeddings, axis=-1)
47:
48:         norm_a = self._min_max_normalize(s_a)
49:         norm_b = self._min_max_normalize(s_b)
50:
51:         combined = self.alpha * norm_a + (1.0 - self.alpha) * norm_b
52:         return combined.astype(np.float32)
53:
54:     def tune_alpha(self, val_features: Dict[str, np.ndarray], val_embeddings: Dict[str, np.ndarray]) -> float:
55:         """Grid search optimal alpha in [0.0..1.0] on validation lakes (INV-002)."""
56:         best_alpha = 0.5
57:         best_variance = -1.0
58:
59:         alphas = np.linspace(0.0, 1.0, 11)
60:         for a in alphas:
61:             self.alpha = float(a)
62:             all_scores = []
63:             for lid in val_features:
64:                 if lid in val_embeddings:
65:                     sc = self.score(val_features[lid], val_embeddings[lid])
66:                     all_scores.extend(sc.tolist())
67:
68:             var = float(np.var(all_scores)) if all_scores else 0.0
69:             if var > best_variance:
70:                 best_variance = var
71:                 best_alpha = float(a)
72:
73:         self.alpha = best_alpha
74:         return best_alpha
```

## source/scripts/train_ts_mae.py

SHA-256: `5fc269af453f8498503bc46b2beedbc706a87f74db7583959ce758952ae4ab82`

```text
120:         mask = output['mask']                      # (B, 180) bool: True = masked
121:
122:         # Loss: MSE only on masked time steps AND valid non-NaN positions
123:         loss_mask = mask.unsqueeze(-1).float() * validity  # (B, 180, 13)
124:         n_loss_positions = loss_mask.sum()
125:
126:         if n_loss_positions > 0:
127:             loss = ((reconstructed - windows) ** 2 * loss_mask).sum() / n_loss_positions
128:         else:
129:             loss = torch.tensor(0.0, device=device)
130:
131:         optimizer.zero_grad()
132:         loss.backward()
133:         optimizer.step()
134:
135:         total_loss += loss.item()
136:         n_batches += 1
137:
138:     return total_loss / max(n_batches, 1)
139:
140:
141: def extract_embeddings(model, feature_dir, lake_ids, norm_stats, output_dir, device,
142:                        window_size=180, stride=30):
143:     """Extract embeddings for all lakes (training + evaluation)."""
144:     model.eval()
145:     means = np.array(norm_stats['means'])
146:     stds = np.array(norm_stats['stds'])
147:
148:     output_dir.mkdir(parents=True, exist_ok=True)
149:
150:     for lake_id in lake_ids:
151:         npz_path = feature_dir / lake_id / 'feature_matrix.npz'
152:         data = np.load(npz_path)
153:         features = data['features']
154:         dates = data['dates']
155:
156:         normed = (features - means) / stds
157:         normed = np.nan_to_num(normed, nan=0.0)
158:
159:         T = features.shape[0]
160:         window_list = []
161:         window_dates = []
162:
163:         for start in range(0, T - window_size + 1, stride):
164:             window = normed[start:start + window_size]
165:             window_list.append(window.astype(np.float32))
166:             window_dates.append(str(dates[start + window_size // 2]))
167:
168:         if not window_list:
169:             continue
170:
171:         # Stack into batch (N_windows, 180, 13)
172:         batch_tensor = torch.from_numpy(np.array(window_list)).to(device)
173:
174:         with torch.no_grad():
175:             # (N_windows, 180, 128) -> mean pool over T -> (N_windows, 128)
176:             emb_latents = model.encode(batch_tensor)  # (N_windows, 180, 128)
177:             embs = emb_latents.mean(dim=1).cpu().numpy()  # (N_windows, 128)
178:
179:         lake_dir = output_dir / lake_id
180:         lake_dir.mkdir(parents=True, exist_ok=True)
181:         np.savez_compressed(
182:             lake_dir / 'embeddings.npz',
183:             embeddings=embs,
184:             dates=window_dates
185:         )
186:         print(f"  {lake_id}: {embs.shape[0]} windows, embedding shape {embs.shape}", flush=True)
187:
188:
189: def main():
190:     # Gate check
191:     gate_path = PROJECT_ROOT / 'results' / 'reality_gate' / 'reality_gate_data.json'
192:     if gate_path.exists():
193:         with open(gate_path, 'r', encoding='utf-8') as f:
194:             gate = json.load(f)
195:         if gate['overall_verdict'] == 'FAIL':
196:             print("ERROR: Reality Gate returned FAIL. Cannot train.")
197:             sys.exit(1)
198:         print(f"Reality Gate: {gate['overall_verdict']} — proceeding.")
199:
200:     # Load registry
201:     reg_path = PROJECT_ROOT / 'source' / 'data' / 'registry' / 'lake_registry.json'
202:     with open(reg_path, 'r', encoding='utf-8') as f:
203:         registry = json.load(f)
204:
205:     feature_dir = PROJECT_ROOT / 'data' / 'features_real'
206:     norm_path = feature_dir / 'normalization_stats.json'
207:     with open(norm_path, 'r', encoding='utf-8') as f:
208:         norm_stats = json.load(f)
209:
210:     # INV-002: Training data from training-role lakes ONLY
211:     training_lake_ids = [l['id'] for l in registry['lakes'] if l['role'] == 'training']
212:     all_lake_ids = [l['id'] for l in registry['lakes']]
213:     eval_lake_ids = [l['id'] for l in registry['lakes'] if l['role'] != 'training']
214:     print(f"Training lakes: {len(training_lake_ids)}")
215:     print(f"Evaluation lakes: {len(eval_lake_ids)} (excluded from training — INV-002)")
216:
217:     # Dataset
218:     dataset = RealDataWindowDataset(
219:         feature_dir, training_lake_ids, norm_stats,
220:         window_size=180, stride=30  # INV-004
221:     )
222:     print(f"Training windows: {len(dataset)}")
223:
224:     dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
225:
226:     # Device: use MPS on Apple Silicon
227:     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
228:     print(f"Device: {device}", flush=True)
229:
230:     # Model — 13 channels instead of 15
```
