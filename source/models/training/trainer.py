"""
TS-MAE Self-Supervised Training Loop and Checkpoint Management.

Handles:
- Device selection (MPS, CUDA, CPU fallback)
- AdamW optimizer & Cosine Annealing learning rate scheduler
- Gradient clipping (max_norm=1.0)
- Checkpoint saving (including model state, optimizer state, and norm_stats)
- Per-epoch training & validation logging to JSONL
- Early stopping based on validation loss
- Reproducibility via set_seed(42) (INV-012)
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Optional
import logging

source_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if source_root not in sys.path if 'sys' in globals() else True:
    import sys
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

from utils.reproducibility import set_seed
from utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Select target hardware device: MPS (Apple Silicon), CUDA, or CPU fallback."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class Trainer:
    """Trainer class for TS-MAE Self-Supervised pretraining.
    
    Args:
        model: TimeSeriesMAE PyTorch model instance
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        config: Loaded config dictionary
        device: Target PyTorch device (MPS/CUDA/CPU)
        output_dir: Root directory for checkpoints and logs
        norm_stats: Optional normalization statistics dictionary
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        output_dir: str,
        norm_stats: Optional[Dict[str, Any]] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.norm_stats = norm_stats
        
        os.makedirs(output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Setup logging
        self.log_file = os.path.join(output_dir, 'training_log.jsonl')
        
        # Hyperparameters
        lr = config.get('training', {}).get('lr', 1e-3)
        weight_decay = config.get('training', {}).get('weight_decay', 0.05)
        self.max_epochs = config.get('training', {}).get('epochs', 50)
        self.patience = config.get('training', {}).get('patience', 20)
        self.max_grad_norm = config.get('training', {}).get('max_grad_norm', 1.0)
        
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_epochs, eta_min=1e-6)
        
        # Tracking state
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Execute one training epoch over training-role data."""
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        total_grad_norm = 0.0
        
        for batch in self.train_loader:
            x = batch['features'].to(self.device)
            self.optimizer.zero_grad()
            
            output = self.model(x)
            loss = output['loss']
            
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            total_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            
        avg_loss = total_loss / max(1, total_batches)
        avg_grad_norm = total_grad_norm / max(1, total_batches)
        return {'loss': avg_loss, 'grad_norm': avg_grad_norm}
    
    def validate(self) -> Dict[str, float]:
        """Execute validation pass over held-out validation-role data."""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch['features'].to(self.device)
                output = self.model(x)
                loss = output['loss']
                
                total_loss += loss.item()
                total_batches += 1
                
        avg_loss = total_loss / max(1, total_batches)
        return {'loss': avg_loss}
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Save model state, optimizer state, scheduler state, and norm_stats to checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': metrics.get('train_loss', 0.0),
            'val_loss': metrics.get('val_loss', 0.0),
            'config': self.config,
            'norm_stats': self.norm_stats,  # Critical: preserved for downstream inference
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to: {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint state and resume training."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'norm_stats' in checkpoint and checkpoint['norm_stats'] is not None:
            self.norm_stats = checkpoint['norm_stats']
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}: {path}")
        return checkpoint
    
    def fit(self, n_epochs: Optional[int] = None) -> Dict[str, Any]:
        """Execute full pretraining loop across n_epochs."""
        set_seed(42)  # INV-012
        epochs_to_run = n_epochs or self.max_epochs
        logger.info(f"Starting TS-MAE pretraining on device={self.device} for {epochs_to_run} epochs...")
        
        history = []
        
        for epoch in range(1, epochs_to_run + 1):
            start_time = time.time()
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step()
            
            wall_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            epoch_log = {
                "epoch": epoch,
                "train_loss": train_metrics['loss'],
                "val_loss": val_metrics['loss'],
                "lr": current_lr,
                "grad_norm": train_metrics['grad_norm'],
                "wall_time_s": wall_time
            }
            history.append(epoch_log)
            
            # Save epoch log to JSONL
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(epoch_log) + '\n')
                
            # Periodic latest checkpoint
            latest_ckpt_path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pt')
            self.save_checkpoint(latest_ckpt_path, epoch, {"train_loss": train_metrics['loss'], "val_loss": val_metrics['loss']})
            
            # Check for best validation loss
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                best_ckpt_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pt')
                self.save_checkpoint(best_ckpt_path, epoch, {"train_loss": train_metrics['loss'], "val_loss": val_metrics['loss']})
            else:
                self.epochs_without_improvement += 1
                
            logger.info(f"Epoch {epoch:02d}/{epochs_to_run:02d} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | LR: {current_lr:.6f} | Time: {wall_time:.2f}s")
            
            # Early stopping check
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best val loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}.")
                break
                
        return {
            "history": history,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_epochs": len(history)
        }
