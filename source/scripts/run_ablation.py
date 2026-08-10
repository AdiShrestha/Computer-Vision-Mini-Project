"""
Channel Ablation Experiment Runner (C05-02).

Runs zero-retraining channel masking ablation across 11 configurations.
Frozen encoder (Chunk 03) and Score-B density model (Chunk 04) used throughout.

Usage:
    python3 source/scripts/run_ablation.py --checkpoint models/checkpoints/ts_mae_best.pt
"""

import os
import sys
import json
import argparse
import numpy as np

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from utils.config_loader import load_config
from utils.logging_utils import setup_logger
from data.loaders.lake_dataset import load_registry, get_lakes_by_role
from models.anomaly.score_a import ReconstructionScorer
from models.anomaly.score_b import EmbeddingDistanceScorer
from models.anomaly.score_c import CombinedScorer
from evaluation.ablation import AblationExperiment, ABLATION_CONFIGS


def main():
    parser = argparse.ArgumentParser(description="Channel Ablation Study (C05-02)")
    parser.add_argument("--checkpoint", default="models/checkpoints/ts_mae_best.pt")
    args = parser.parse_args()

    logger = setup_logger("run_ablation")
    config = load_config()

    repo_root = os.path.dirname(source_root)
    ckpt_path = os.path.join(repo_root, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    features_dir = os.path.join(repo_root, config['paths']['features'])
    embeddings_dir = os.path.join(repo_root, 'data', 'embeddings')
    registry_path = os.path.join(repo_root, config['paths']['lake_registry'])
    output_dir = os.path.join(repo_root, 'results', 'ablation')
    os.makedirs(output_dir, exist_ok=True)

    # ========================================================================
    # 1. Load registry, features, and embeddings
    # ========================================================================
    registry = load_registry(registry_path)
    by_role = get_lakes_by_role(registry)
    training_ids = by_role.get('training', [])
    control_ids = by_role.get('evaluation_control', [])
    all_lake_ids = [l['id'] for l in registry['lakes']]

    features_map = {}
    embeddings_map = {}
    for l_id in all_lake_ids:
        feat_p = os.path.join(features_dir, l_id, 'feature_matrix.npz')
        emb_p = os.path.join(embeddings_dir, l_id, 'embeddings.npz')
        if os.path.exists(feat_p):
            features_map[l_id] = np.load(feat_p, allow_pickle=True)['features'].astype(np.float32)
        if os.path.exists(emb_p):
            embeddings_map[l_id] = np.load(emb_p, allow_pickle=True)['embeddings'].astype(np.float32)

    logger.info(f"Loaded {len(features_map)} feature matrices, {len(embeddings_map)} embedding matrices")

    # ========================================================================
    # 2. Initialize FROZEN scorers (same as Chunk 04 — no retraining)
    # ========================================================================
    score_a_inst = ReconstructionScorer(checkpoint_path=ckpt_path)

    training_embs = {lid: embeddings_map[lid] for lid in training_ids if lid in embeddings_map}
    score_b_inst = EmbeddingDistanceScorer(training_embeddings=training_embs)
    score_c_inst = CombinedScorer(score_a_scorer=score_a_inst, score_b_scorer=score_b_inst, alpha=0.5)

    experiment = AblationExperiment(
        score_a_inst=score_a_inst,
        score_b_inst=score_b_inst,
        score_c_inst=score_c_inst,
        ckpt_path=ckpt_path,
    )

    logger.info(f"Scorers initialized. Running {len(ABLATION_CONFIGS)} ablation configurations...")

    # ========================================================================
    # 3. Run all ablation configurations
    # ========================================================================
    all_results = {}
    for config_name, keep_cols in ABLATION_CONFIGS.items():
        logger.info(f"  Running config: {config_name} ({len(keep_cols)}/15 channels active)")
        try:
            result = experiment.run_config(
                config_name=config_name,
                keep_cols=keep_cols,
                features_map=features_map,
                control_ids=control_ids,
                output_dir=output_dir,
            )
            all_results[config_name] = result
            logger.info(f"    AUC-ROC={result['auc_roc']:.4f}, AUC-PR={result['auc_pr']:.4f}, "
                       f"Det={result['synthetic_detection_rate']:.2f}, FP={result['false_positive_rate']:.4f}")
        except Exception as e:
            logger.error(f"    FAILED: {e}")
            all_results[config_name] = {'error': str(e)}

    # ========================================================================
    # 4. Compute per-channel contribution (FULL_15CH AUC-ROC - NO_CHxx AUC-ROC)
    # ========================================================================
    full_auc = all_results.get('FULL_15CH', {}).get('auc_roc', None)
    channel_contributions = {}
    if full_auc is not None:
        for ch_id in ['CH-01', 'CH-02', 'CH-03', 'CH-04', 'CH-05', 'CH-07', 'CH-08']:
            no_ch_key = f'NO_{ch_id.replace("-", "")}'
            no_ch_auc = all_results.get(no_ch_key, {}).get('auc_roc', None)
            if no_ch_auc is not None:
                channel_contributions[ch_id] = float(full_auc - no_ch_auc)
            else:
                channel_contributions[ch_id] = None

    # ========================================================================
    # 5. Verify FULL_15CH matches Chunk 04 (sanity check)
    # ========================================================================
    chunk04_auc = 0.9521053093284387  # from TAKE_THIS/evaluation_summary.json
    full_15ch_auc = all_results.get('FULL_15CH', {}).get('auc_roc', None)
    sanity_pass = full_15ch_auc is not None and abs(full_15ch_auc - chunk04_auc) <= 0.01
    logger.info(f"Sanity check FULL_15CH vs Chunk04: {full_15ch_auc:.4f} vs {chunk04_auc:.4f} → {'PASS' if sanity_pass else 'FAIL'}")

    # ========================================================================
    # 6. Verify no encoder retraining
    # ========================================================================
    no_retrain = experiment.verify_no_retraining()
    logger.info(f"No-retraining invariant: {'PASS' if no_retrain else 'FAIL ⚠️'}")

    # ========================================================================
    # 7. Save ablation summary
    # ========================================================================
    ablation_summary = {
        "ablation_version": "C05-02",
        "checkpoint_used": ckpt_path,
        "encoder_retrained": False,
        "n_configs": len(ABLATION_CONFIGS),
        "chunk04_reference_auc_roc": chunk04_auc,
        "full_15ch_sanity_pass": sanity_pass,
        "no_retraining_verified": no_retrain,
        "configs": all_results,
        "channel_contributions": channel_contributions,
        "best_config": max(
            (k for k in all_results if 'auc_roc' in all_results[k]),
            key=lambda k: all_results[k]['auc_roc'],
            default=None,
        ),
        "worst_config": min(
            (k for k in all_results if 'auc_roc' in all_results[k]),
            key=lambda k: all_results[k]['auc_roc'],
            default=None,
        ),
        "most_important_channel": max(
            (ch for ch in channel_contributions if channel_contributions[ch] is not None),
            key=lambda ch: channel_contributions[ch] if channel_contributions[ch] is not None else -999,
            default=None,
        ),
    }

    summary_path = os.path.join(output_dir, 'ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(ablation_summary, f, indent=2)

    logger.info(f"Ablation complete. Summary saved to {summary_path}")

    # Print summary table
    print("\n" + "=" * 75)
    print("ABLATION RESULTS (C05-02)")
    print("=" * 75)
    print(f"{'Config':<20} {'Channels':>8} {'AUC-ROC':>9} {'AUC-PR':>8} {'Det%':>7} {'FP%':>7}")
    print("-" * 75)
    for cfg_name, res in sorted(all_results.items(), key=lambda x: x[1].get('auc_roc', 0), reverse=True):
        if 'auc_roc' in res:
            print(f"{cfg_name:<20} {res['n_active_channels']:>8} "
                  f"{res['auc_roc']:>9.4f} {res['auc_pr']:>8.4f} "
                  f"{res['synthetic_detection_rate']*100:>7.1f} {res['false_positive_rate']*100:>7.1f}")
    print("=" * 75)

    print("\nChannel Contributions (FULL AUC-ROC - NO_CH AUC-ROC):")
    if channel_contributions:
        sorted_ch = sorted(channel_contributions.items(), key=lambda x: (x[1] or -99), reverse=True)
        for ch, contrib in sorted_ch:
            print(f"  {ch}: {'+' if (contrib or 0) >= 0 else ''}{contrib:.4f}")

    print(f"\nSanity check (FULL_15CH ≈ Chunk04 Score-C): {'✅ PASS' if sanity_pass else '❌ FAIL'}")
    print(f"No-retraining invariant: {'✅ PASS' if no_retrain else '❌ FAIL'}")


if __name__ == '__main__':
    main()
