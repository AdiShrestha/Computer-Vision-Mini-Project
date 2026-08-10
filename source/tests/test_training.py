"""Verify training loop implementation."""
import os
import sys
import torch

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)


def test_trainer_module_exists():
    """Trainer module exists and imports."""
    from models.training.trainer import Trainer
    assert Trainer is not None


def test_trainer_has_required_methods():
    """Trainer has required methods."""
    from models.training.trainer import Trainer
    assert hasattr(Trainer, 'train_epoch')
    assert hasattr(Trainer, 'validate')
    assert hasattr(Trainer, 'fit')
    assert hasattr(Trainer, 'save_checkpoint')
    assert hasattr(Trainer, 'load_checkpoint')


def test_device_selection():
    """Device selection logic exists."""
    from models.training.trainer import get_device
    device = get_device()
    assert isinstance(device, torch.device)


def test_checkpoint_format():
    """Checkpoint saving and loading works."""
    import tempfile
    from models.encoder.ts_mae import TimeSeriesMAE
    from models.training.trainer import Trainer
    
    model = TimeSeriesMAE()
    # Quick smoke test: can save/load a checkpoint
    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'train_loss': 1.0,
        'val_loss': 1.0,
    }
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(checkpoint, f.name)
        loaded = torch.load(f.name, weights_only=False)
        assert loaded['epoch'] == 0
        os.unlink(f.name)
