"""
CLI Script for Full-Scale TS-MAE Embedding Extraction across 20 lakes.
"""

import os
import sys
import argparse

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from utils.config_loader import load_config
from utils.logging_utils import setup_logger
from models.embedding.extract import extract_embeddings


def main():
    parser = argparse.ArgumentParser(description="TS-MAE Full-Scale Embedding Extractor")
    parser.add_argument("--checkpoint", default="models/checkpoints/ts_mae_best.pt", help="Path to model checkpoint")
    args = parser.parse_args()

    logger = setup_logger("extract_embeddings")
    config = load_config()

    repo_root = os.path.dirname(source_root)
    ckpt_path = os.path.join(repo_root, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    features_dir = os.path.join(repo_root, config['paths']['features'])
    registry_path = os.path.join(repo_root, config['paths']['lake_registry'])
    output_dir = os.path.join(repo_root, 'data', 'embeddings')

    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found at: {ckpt_path}")
        sys.exit(1)

    logger.info(f"Extracting embeddings using checkpoint: {ckpt_path}")
    summary = extract_embeddings(
        checkpoint_path=ckpt_path,
        features_dir=features_dir,
        registry_path=registry_path,
        output_dir=output_dir
    )

    logger.info(f"Extracted embeddings for {summary['total_lakes_extracted']} lakes. Summary saved to {output_dir}/embedding_summary.json")
    print(f"\nEmbedding Extraction Completed. Total lakes: {summary['total_lakes_extracted']}")


if __name__ == '__main__':
    main()
