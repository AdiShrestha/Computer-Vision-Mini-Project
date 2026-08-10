"""Verify training execution produced valid results."""
import os
import sys
import json
import torch

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)

CHECKPOINTS_DIR = os.path.join(repo_root, 'models', 'checkpoints')
TRAINING_DIR = os.path.join(repo_root, 'results', 'training')


def test_best_checkpoint_exists():
    """Best checkpoint file exists and is loadable."""
    path = os.path.join(CHECKPOINTS_DIR, 'ts_mae_best.pt')
    assert os.path.isfile(path), "Best checkpoint not found"
    checkpoint = torch.load(path, weights_only=False, map_location='cpu')
    assert 'model_state_dict' in checkpoint
    assert 'epoch' in checkpoint


def test_training_log_exists():
    """Training log JSONL exists with multiple entries."""
    log_path = os.path.join(TRAINING_DIR, 'training_log.jsonl')
    assert os.path.isfile(log_path)
    with open(log_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) >= 10, f"Only {len(lines)} log entries"
    assert 'train_loss' in lines[0]


def test_loss_decreased():
    """Training loss decreased (convergence check)."""
    log_path = os.path.join(TRAINING_DIR, 'training_log.jsonl')
    with open(log_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    
    initial_loss = lines[0]['train_loss']
    final_loss = lines[-1]['train_loss']
    
    assert final_loss < initial_loss, (
        f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
    )
    # At least 30% reduction
    reduction = (initial_loss - final_loss) / initial_loss
    assert reduction > 0.3, f"Only {reduction*100:.1f}% loss reduction"


def test_no_nan_in_training():
    """No NaN values in training history."""
    log_path = os.path.join(TRAINING_DIR, 'training_log.jsonl')
    with open(log_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    
    for i, entry in enumerate(lines):
        assert entry['train_loss'] == entry['train_loss'], f"NaN at epoch {i}"


def test_training_summary_exists():
    """Training summary JSON exists with convergence status."""
    summary_path = os.path.join(TRAINING_DIR, 'training_summary.json')
    assert os.path.isfile(summary_path)
    with open(summary_path) as f:
        summary = json.load(f)
    assert 'convergence' in summary
    assert 'wall_time_seconds' in summary
