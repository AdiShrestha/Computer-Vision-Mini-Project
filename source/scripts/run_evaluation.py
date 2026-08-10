"""
Top-Level Evaluation Execution Script (REWORKED — C04-R1).

Fixes applied:
1. Score-A now normalizes features using checkpoint norm_stats
2. eval_scorer_fn correctly uses each scorer (not Score-A for all)
3. Score-B/C E3 synthetic evaluation extracts embeddings from modified features
4. Baseline metrics are COMPUTED, not hardcoded
5. All reported numbers trace directly to computation outputs
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from utils.config_loader import load_config
from utils.logging_utils import setup_logger
from data.loaders.lake_dataset import load_registry, get_lakes_by_role
from models.anomaly.score_a import ReconstructionScorer
from models.anomaly.score_b import EmbeddingDistanceScorer
from models.anomaly.score_c import CombinedScorer
from models.anomaly.smoothing import ema_smooth
from models.baseline.extent_threshold import ExtentThresholdDetector
from evaluation.synthetic.injector import SyntheticInjector
from evaluation.protocols.e1_retrospective import run_e1_retrospective
from evaluation.protocols.e2_negative_controls import run_e2_negative_controls
from evaluation.protocols.e3_synthetic import run_e3_synthetic
from evaluation.protocols.e4_baseline import run_e4_baseline
from evaluation.protocols.metrics import (
    compute_full_metrics, compute_false_positive_rate,
    compute_synthetic_detection_rate, compute_auc,
    compute_lead_time, compute_peak_magnitude,
    EVENT_DATE, date_to_window_idx,
)


def main():
    parser = argparse.ArgumentParser(description="Full Evaluation Execution (Reworked C04-R1)")
    parser.add_argument("--checkpoint", default="models/checkpoints/ts_mae_best.pt",
                        help="Path to trained TS-MAE checkpoint")
    args = parser.parse_args()

    logger = setup_logger("run_evaluation")
    config = load_config()

    repo_root = os.path.dirname(source_root)
    ckpt_path = os.path.join(repo_root, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    features_dir = os.path.join(repo_root, config['paths']['features'])
    embeddings_dir = os.path.join(repo_root, 'data', 'embeddings')
    registry_path = os.path.join(repo_root, config['paths']['lake_registry'])
    output_dir = os.path.join(repo_root, 'results', 'evaluation')
    os.makedirs(output_dir, exist_ok=True)

    registry = load_registry(registry_path)
    by_role = get_lakes_by_role(registry)

    training_ids = by_role.get('training', [])
    control_ids = by_role.get('evaluation_control', [])
    event_ids = by_role.get('evaluation_event', [])
    all_lake_ids = [l['id'] for l in registry['lakes']]

    logger.info(f"Loaded registry: {len(training_ids)} training, {len(control_ids)} control, {len(event_ids)} event")

    # ========================================================================
    # 1. Load all features and embeddings
    # ========================================================================
    features_map = {}
    embeddings_map = {}
    for l_id in all_lake_ids:
        feat_p = os.path.join(features_dir, l_id, 'feature_matrix.npz')
        emb_p = os.path.join(embeddings_dir, l_id, 'embeddings.npz')
        if os.path.exists(feat_p):
            features_map[l_id] = np.load(feat_p, allow_pickle=True)['features'].astype(np.float32)
        if os.path.exists(emb_p):
            embeddings_map[l_id] = np.load(emb_p, allow_pickle=True)['embeddings'].astype(np.float32)

    # ========================================================================
    # 2. Initialize Scorers
    # ========================================================================
    # Score-A: reconstruction error WITH PROPER NORMALIZATION
    score_a_inst = ReconstructionScorer(checkpoint_path=ckpt_path)

    # Score-B: embedding distance (INV-002: fitted on training-role ONLY)
    training_embs = {lid: embeddings_map[lid] for lid in training_ids if lid in embeddings_map}
    score_b_inst = EmbeddingDistanceScorer(training_embeddings=training_embs)

    # Score-C: combined
    score_c_inst = CombinedScorer(score_a_scorer=score_a_inst, score_b_scorer=score_b_inst, alpha=0.5)

    # Baseline
    baseline_inst = ExtentThresholdDetector(threshold=0.10)

    # Synthetic injector (INV-012: seed=2023)
    injector = SyntheticInjector(seed=2023)

    # ========================================================================
    # 3. Compute Raw & Smoothed Anomaly Scores for ALL lakes
    # ========================================================================
    raw_scores = {'score_a': {}, 'score_b': {}, 'score_c': {}}
    smoothed_scores = {'score_a': {}, 'score_b': {}, 'score_c': {}}

    for lid in all_lake_ids:
        if lid in features_map and lid in embeddings_map:
            feat = features_map[lid]
            emb = embeddings_map[lid]

            sa = score_a_inst.score(feat)        # Score-A: reconstruction MSE
            sb = score_b_inst.score(emb)         # Score-B: embedding k-NN distance
            sc = score_c_inst.score(feat, emb)   # Score-C: combined

            raw_scores['score_a'][lid] = sa
            raw_scores['score_b'][lid] = sb
            raw_scores['score_c'][lid] = sc

            smoothed_scores['score_a'][lid] = ema_smooth(sa, span=5)
            smoothed_scores['score_b'][lid] = ema_smooth(sb, span=5)
            smoothed_scores['score_c'][lid] = ema_smooth(sc, span=5)

            # Save per-lake CSV time series
            lake_csv_dir = os.path.join(output_dir, 'per_lake', lid)
            os.makedirs(lake_csv_dir, exist_ok=True)
            df = pd.DataFrame({
                'window_idx': np.arange(len(sa)),
                'score_a_raw': sa,
                'score_a_smoothed': smoothed_scores['score_a'][lid],
                'score_b_raw': sb,
                'score_b_smoothed': smoothed_scores['score_b'][lid],
                'score_c_raw': sc,
                'score_c_smoothed': smoothed_scores['score_c'][lid],
            })
            df.to_csv(os.path.join(lake_csv_dir, 'anomaly_scores.csv'), index=False)

    logger.info(f"Scored {len(raw_scores['score_a'])} lakes with all 3 scorers")

    # ========================================================================
    # 4. Derive Detection Threshold per Scorer from E3 Synthetic ROC
    # ========================================================================
    control_feats = {lid: features_map[lid] for lid in control_ids if lid in features_map}

    def make_scorer_fn(scorer_type: str) -> Callable:
        """Create a scorer function that CORRECTLY uses the specified scorer.

        For Score-B and Score-C, we extract embeddings from modified features
        using the encoder, then score those embeddings.

        CRITICAL: Each scorer_type uses its OWN scoring pipeline.
        """
        def scorer_fn(modified_features: np.ndarray) -> np.ndarray:
            if scorer_type == 'score_a':
                # Score-A: reconstruction MSE on modified features
                return ema_smooth(score_a_inst.score(modified_features), span=5)
            elif scorer_type == 'score_b':
                # Score-B: extract embeddings from modified features, then k-NN
                modified_emb = score_a_inst.get_embeddings(modified_features)
                return ema_smooth(score_b_inst.score(modified_emb), span=5)
            elif scorer_type == 'score_c':
                # Score-C: both reconstruction + embedding distance
                modified_emb = score_a_inst.get_embeddings(modified_features)
                return ema_smooth(score_c_inst.score(modified_features, modified_emb), span=5)
            else:
                raise ValueError(f"Unknown scorer: {scorer_type}")
        return scorer_fn

    thresholds = {}
    for s_name in ['score_a', 'score_b', 'score_c']:
        # Set threshold at 85th percentile of smoothed control scores
        all_ctrl_s = np.concatenate([
            smoothed_scores[s_name][lid]
            for lid in control_ids
            if lid in smoothed_scores[s_name]
        ])
        thresholds[s_name] = float(np.percentile(all_ctrl_s, 85))
        logger.info(f"  {s_name} threshold: {thresholds[s_name]:.6f}")

    # ========================================================================
    # 5. Run E3 with CORRECT scorers and derived thresholds
    # ========================================================================
    e3_results = {}
    for s_name in ['score_a', 'score_b', 'score_c']:
        e3_res = run_e3_synthetic(
            scorer_fn=make_scorer_fn(s_name),
            control_features=control_feats,
            injector=SyntheticInjector(seed=2023),  # Fresh injector
            threshold=thresholds[s_name],
            output_dir=os.path.join(output_dir, s_name),
        )
        e3_results[s_name] = e3_res

    # ========================================================================
    # 6. Execute E1, E2, E4 across all Scorers
    # ========================================================================
    event_lake_id = event_ids[0] if event_ids else 'SGL-001'
    event_feat = features_map.get(event_lake_id, np.zeros((108, 15), dtype=np.float32))

    summary_comparison = {}
    for s_name in ['score_a', 'score_b', 'score_c']:
        s_out_dir = os.path.join(output_dir, s_name)

        # E1: Retrospective backtesting on South Lhonak
        e1_res = run_e1_retrospective(
            event_lake_id=event_lake_id,
            smoothed_scores={s_name: smoothed_scores[s_name][event_lake_id]},
            threshold=thresholds[s_name],
            output_dir=s_out_dir,
        )

        # E2: Negative controls
        e2_res = run_e2_negative_controls(
            control_lake_ids=control_ids,
            smoothed_scores={s_name: smoothed_scores[s_name]},
            threshold=thresholds[s_name],
            output_dir=s_out_dir,
        )

        # E4: Baseline comparison
        learned_metrics = {
            'lead_time_days': e1_res[s_name]['lead_time_days'],
            'false_positive_rate': e2_res[s_name]['overall_fp_rate'],
            'peak_anomaly_magnitude': e1_res[s_name]['peak_anomaly_magnitude'],
        }
        e4_res = run_e4_baseline(
            baseline_detector=baseline_inst,
            event_features=event_feat,
            control_features=control_feats,
            learned_metrics=learned_metrics,
            output_dir=s_out_dir,
        )

        summary_comparison[s_name] = {
            'threshold': thresholds[s_name],
            'lead_time_days': e1_res[s_name]['lead_time_days'],
            'peak_anomaly_magnitude': e1_res[s_name]['peak_anomaly_magnitude'],
            'false_positive_rate': e2_res[s_name]['overall_fp_rate'],
            'synthetic_detection_rate': e3_results[s_name]['overall_detection_rate'],
            'auc_roc': e3_results[s_name]['auc_roc'],
            'auc_pr': e3_results[s_name]['auc_pr'],
            'delta_lead_time': e4_res['comparison']['delta_lead_time'],
            'delta_fp_rate': e4_res['comparison']['delta_fp_rate'],
        }

    # ========================================================================
    # 7. COMPUTE baseline metrics (NOT hardcoded)
    # ========================================================================
    baseline_event_scores = baseline_inst.score(event_feat[:, 0])
    nonzero_baseline = baseline_event_scores[baseline_event_scores > 0]
    baseline_threshold = float(np.median(nonzero_baseline)) if len(nonzero_baseline) > 0 else 0.05

    event_window_idx = date_to_window_idx(EVENT_DATE)

    # Baseline E1
    baseline_lead_time = compute_lead_time(baseline_event_scores, baseline_threshold, event_window_idx)
    baseline_peak = compute_peak_magnitude(baseline_event_scores, event_window_idx)

    # Baseline E2: FP rate on control lakes
    baseline_control_scores = {}
    for lid in control_ids:
        if lid in features_map:
            baseline_control_scores[lid] = baseline_inst.score(features_map[lid][:, 0])
    baseline_fp_rate = compute_false_positive_rate(baseline_control_scores, baseline_threshold)

    # Baseline E3: synthetic detection rate
    baseline_detections = []
    baseline_all_labels = []
    baseline_all_scores_list = []
    baseline_injector = SyntheticInjector(seed=2023)
    for lid, feat in control_feats.items():
        injections = baseline_injector.generate_injections(feat, lid)
        for modified_feat, meta in injections:
            bl_scores = baseline_inst.score(modified_feat[:, 0])
            inj_w = meta['window_idx']
            dur = meta.get('duration_windows', 1)
            inj_end = min(inj_w + dur, len(bl_scores))
            detected = bool(np.any(bl_scores[inj_w:inj_end] > baseline_threshold))
            baseline_detections.append(detected)
            labels = np.zeros(len(bl_scores))
            labels[inj_w:inj_end] = 1
            baseline_all_labels.extend(labels.tolist())
            baseline_all_scores_list.extend(bl_scores.tolist())

    baseline_detection_rate = compute_synthetic_detection_rate(baseline_detections)
    baseline_auc = compute_auc(np.array(baseline_all_labels), np.array(baseline_all_scores_list))

    summary_comparison['baseline'] = {
        'threshold': float(baseline_threshold),
        'lead_time_days': baseline_lead_time,
        'peak_anomaly_magnitude': float(baseline_peak),
        'false_positive_rate': float(baseline_fp_rate),
        'synthetic_detection_rate': float(baseline_detection_rate),
        'auc_roc': float(baseline_auc['auc_roc']),
        'auc_pr': float(baseline_auc['auc_pr']),
        'delta_lead_time': 0,
        'delta_fp_rate': 0.0,
    }

    # ========================================================================
    # 8. Save evaluation summary
    # ========================================================================
    best_scorer = max(
        ['score_a', 'score_b', 'score_c'],
        key=lambda s: summary_comparison[s]['auc_roc']
    )
    best_detected = summary_comparison[best_scorer]['lead_time_days'] is not None

    eval_summary = {
        "scorer_comparison": summary_comparison,
        "best_scorer": best_scorer,
        "south_lhonak_detected": best_detected,
        "rq1_preliminary": (
            "positive" if best_detected and summary_comparison[best_scorer].get('auc_roc', 0) > 0.5
            else "negative" if not best_detected
            else "mixed"
        ),
        "checkpoint_used": ckpt_path,
        "rework_version": "C04-R1",
    }

    summary_file = os.path.join(output_dir, 'evaluation_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(eval_summary, f, indent=2)

    logger.info(f"Evaluation complete. Summary: {summary_file}")
    logger.info(f"Best scorer: {best_scorer} (AUC-ROC: {summary_comparison[best_scorer]['auc_roc']:.4f})")

    # Print summary table
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS (C04-R1 REWORK)")
    print("=" * 80)
    print(f"{'Metric':<35} {'Score-A':>10} {'Score-B':>10} {'Score-C':>10} {'Baseline':>10}")
    print("-" * 80)
    for metric in ['lead_time_days', 'peak_anomaly_magnitude', 'false_positive_rate',
                    'synthetic_detection_rate', 'auc_roc', 'auc_pr']:
        vals = []
        for s in ['score_a', 'score_b', 'score_c', 'baseline']:
            v = summary_comparison[s].get(metric)
            if v is None:
                vals.append('None')
            elif isinstance(v, float):
                vals.append(f'{v:.4f}')
            else:
                vals.append(str(v))
        print(f"{metric:<35} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")
    print("=" * 80)


if __name__ == '__main__':
    main()
