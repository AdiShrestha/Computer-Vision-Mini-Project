"""PyTorch dataset and data loaders for multi-modal time-series panels.

Enforces zero future-data leakage, split boundary isolation, deterministic
collation, worker seed initialization, and M3 memory budget batch bounds [8, 16].
"""
from __future__ import annotations
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.utils.data

from .features import MultiModalPanel, WINDOW_DAYS, NUM_CHANNELS
from .splits import SplitManifest


def seed_worker(worker_id: int) -> None:
    """Initialize worker seed deterministically for PyTorch DataLoader workers."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def deterministic_collate(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Collate panels into batch tensors, rejecting unobserved batches."""
    x_list, mask_list, meta_list = zip(*batch)
    batch_x = torch.stack(x_list, dim=0)
    batch_mask = torch.stack(mask_list, dim=0)

    # Reject batches with zero observed elements as unestimable
    if batch_mask.sum().item() == 0:
        raise ValueError("Batch contains zero valid observations; loss is unestimable")

    return batch_x, batch_mask, list(meta_list)


class PanelDataset(torch.utils.data.Dataset):
    """Dataset producing (T=180, C=11) context windows with boolean masks."""

    def __init__(
        self,
        panels: Sequence[MultiModalPanel],
        split_manifest: SplitManifest,
        split: str = "train",
        normalizer: Optional[Any] = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        self.split = split
        self.split_manifest = split_manifest
        self.normalizer = normalizer

        # Determine allowed lake IDs
        if split == "train":
            allowed_lakes = set(split_manifest.train_lakes)
        elif split == "val":
            allowed_lakes = set(split_manifest.val_lakes)
        else:
            allowed_lakes = set(split_manifest.test_lakes)

        # Set of purged window IDs
        purged_ids = {w.window_id for w in split_manifest.purged_windows}

        # Filter and validate panels
        self.valid_panels: List[MultiModalPanel] = []
        split_date = split_manifest.metadata.get("split_date")

        for panel in panels:
            # 1. Lake partition boundary check
            if panel.lake_id not in allowed_lakes:
                raise ValueError(
                    f"Panel lake {panel.lake_id} is not in allowed {split} lakes: {allowed_lakes}"
                )

            # 2. Purged boundary check
            if panel.window_id in purged_ids:
                raise ValueError(
                    f"Panel {panel.window_id} is in purged boundary set and cannot be loaded"
                )

            # 3. Retrospective availability check for training
            if split == "train" and split_date is not None:
                if panel.end_date > split_date:
                    raise ValueError(
                        f"Training panel {panel.window_id} ends at {panel.end_date}, "
                        f"which violates retrospective split cutoff {split_date}"
                    )

            self.valid_panels.append(panel)

        if not self.valid_panels:
            raise ValueError(f"No valid panels found for split '{split}'")

    def __len__(self) -> int:
        return len(self.valid_panels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        panel = self.valid_panels[idx]

        if self.normalizer is not None:
            values_norm, mask = self.normalizer.transform(panel.values, panel.mask)
        else:
            values_norm = np.where(panel.mask, panel.values, 0.0).astype(np.float32)
            mask = panel.mask.copy()

        x_tensor = torch.from_numpy(values_norm.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(bool))
        meta = panel.to_dict()

        return x_tensor, mask_tensor, meta


def create_panel_dataloader(
    dataset: PanelDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Create a deterministic DataLoader with memory bounds [8, 16]."""
    # Enforce batch size bounds [8, 16] unless dataset is smaller than 8 in small tests
    if (batch_size < 8 or batch_size > 16) and len(dataset) >= 8:
        raise ValueError(
            f"batch_size must be between 8 and 16 to respect memory bounds, got {batch_size}"
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=seed_worker,
        collate_fn=deterministic_collate,
        num_workers=num_workers,
    )
