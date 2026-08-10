"""
Threshold Analysis for INV-007 Compliance.

Sweeps threshold percentiles across control lakes to find the threshold
where False Positive Rate (E2) <= 0.10 while maximizing synthetic detection rate (E3).
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Any, List

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
from evaluation.synthetic.injector import SyntheticInjector
from evaluation.protocols.metrics import compute_false_positive_rate, compute_synthetic_detection_rate


def main():
    logger = setup_logger("run_threshold_analysis")
    config = load_config()

    repo_root = os.path.dirname(source_root)
    ckpt_path = os.path.join(repo_root, "models", "checkpoints", "ts_mae_best.pt")
    features_dir = os.path.join(repo_root, config['paths']['features'])
    embeddings_dir = os.path.join(repo_root, 'data', 'embeddings')
    registry_path = os.path.join(repo_root, config['paths']['lake_registry'])
    output_dir = os.path.join(repo_root, 'results', 'ablation')
    os.makedirs(output_dir, exist_ok=True)

    registry = load_registry(registry_path)
    by_role = get_lakes_by_role(registry)

    training_ids = by_role.get('training', [])
    control_ids = by_role.get('evaluation_control', [])

    # 1. Load features & embeddings for training & control lakes
    features_map = {}
    embeddings_map = {}
    for lid in training_ids + control_ids:
        feat_p = os.path.join(features_dir, lid, 'feature_matrix.npz')
        emb_p = os.path.join(embeddings_dir, lid, 'embeddings.npz')
        if os.path.exists(feat_p):
            features_map[lid] = np.load(feat_p, allow_pickle=True)['features'].astype(np.float32)
        if os.path.exists(emb_p):
            embeddings_map[lid] = np.load(emb_p, allow_pickle=True)['embeddings'].astype(np.float32)

    # 2. Instantiate Scorers
    score_a_inst = ReconstructionScorer(checkpoint_path=ckpt_path)
    training_embs = {lid: embeddings_map[lid] for lid in training_ids if lid in embeddings_map}
    score_b_inst = EmbeddingDistanceScorer(training_embeddings=training_embs)
    score_c_inst = CombinedScorer(score_a_scorer=score_a_inst, score_b_scorer=score_b_inst, alpha=0.5)

    # 3. Compute Score-C smoothed time series for control lakes
    control_feats = {lid: features_map[lid] for lid in control_ids if lid in features_map}
    control_smoothed = {}
    for lid, feat in control_feats.items():
        emb = embeddings_map[lid]
        sc = score_c_inst.score(feat, emb)
        control_smoothed[lid] = ema_smooth(sc, span=5)

    all_ctrl_scores = np.concatenate(list(control_smoothed.values()))
    original_threshold = float(np.percentile(all_ctrl_scores, 85))
    original_fp_rate = compute_false_positive_rate(control_smoothed, original_threshold)

    # 4. Sweep percentiles from 50 to 99
    injector = SyntheticInjector(seed=2023)
    sweep_table = []
    
    refined_threshold = None
    refined_fp_rate = None
    refined_det_rate = None
    best_det_rate = -1.0

    for pct in range(50, 100):
        thresh = float(np.percentile(all_ctrl_scores, pct))
        fp = compute_false_positive_rate(control_smoothed, thresh)

        # Run E3 synthetic injection evaluation at this threshold
        detections = []
        for lid, feat in control_feats.items():
            injections = injector.generate_injections(feat, lid)
            for mod_feat, meta in injections:
                mod_emb = score_a_inst.get_embeddings(mod_feat)
                sc_mod = score_c_inst.score(mod_feat, mod_emb)
                sc_smoothed = ema_smooth(sc_mod, span=5)

                inj_w = meta['window_idx']
                dur = meta.get('duration_windows', 1)
                inj_end = min(inj_w + dur, len(sc_smoothed))
                det = bool(np.any(sc_smoothed[inj_w:inj_end] > thresh))
                detections.append(det)

        det_rate = compute_synthetic_detection_rate(detections)

        sweep_table.append({
            "percentile": pct,
            "threshold": thresh,
            "false_positive_rate": float(fp),
            "synthetic_detection_rate": float(det_rate)
        })

        # Check INV-007 compliance constraint: fp <= 0.10
        if fp <= 0.10 and det_rate > best_det_rate:
            best_det_rate = det_rate
            refined_threshold = thresh
            refined_fp_rate = float(fp)
            refined_det_rate = float(det_rate)

    # Fallback to highest percentile (pct=90) if exact match not found
    if refined_threshold is None:
        thresh_90 = float(np.percentile(all_ctrl_scores, 90))
        refined_threshold = thresh_90
        refined_fp_rate = float(compute_false_positive_rate(control_smoothed, thresh_90))
        refined_det_rate = float(next(e['synthetic_detection_rate'] for e in sweep_table if e['percentile'] == 90))

    result = {
        "method": "roc_threshold_sweep",
        "sweep_percentiles": list(range(50, 100)),
        "original_threshold": original_threshold,
        "original_fp_rate": float(original_fp_rate),
        "refined_threshold": float(refined_threshold),
        "refined_fp_rate": float(refined_fp_rate),
        "refined_detection_rate": float(refined_det_rate),
        "inv007_target": 0.10,
        "inv007_compliant": bool(refined_fp_rate <= 0.10),
        "source_file": "results/ablation/ablation_summary.json",
        "threshold_sweep_table": sweep_table
    }

    out_file = os.path.join(output_dir, 'threshold_analysis.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Threshold analysis saved to {out_file}")
    print(f"Refined Threshold: {refined_threshold:.6f} | FP Rate: {refined_fp_rate*100:.2f}% | Compliant: {result['inv007_compliant']}")


if __name__ == '__main__':
    main()
