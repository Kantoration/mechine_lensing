"""
High-performance dataloader helpers for the lensing project.

The helpers centralise opinionated defaults used across training scripts
(e.g. pinned memory only when CUDA is available, seeded splits) while
keeping the public API minimal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .lens_dataset import LensDataset

logger = logging.getLogger(__name__)


@dataclass
class _LoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    drop_last: bool

    @classmethod
    def from_kwargs(
        cls,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: Optional[bool],
        drop_last: bool,
    ) -> "_LoaderConfig":
        if persistent_workers is None:
            persistent_workers = num_workers > 0
        pin_memory = bool(pin_memory and torch.cuda.is_available())
        persistent_workers = bool(persistent_workers and num_workers > 0)
        return cls(batch_size, num_workers, pin_memory, persistent_workers, drop_last)

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "drop_last": self.drop_last,
        }


def _build_train_val_datasets(
    data_root: str,
    img_size: int,
    val_split: float,
) -> Tuple[Dataset, Optional[Dataset]]:
    train_dataset = LensDataset(
        data_root=data_root,
        split="train",
        img_size=img_size,
        augment=True,
        validate_paths=True,
    )

    if val_split <= 0:
        return train_dataset, None

    num_samples = len(train_dataset)
    val_size = max(int(num_samples * val_split), 1)
    train_size = num_samples - val_size
    if train_size <= 0:
        raise ValueError(
            f"val_split={val_split} produces empty training set for {num_samples} samples"
        )

    generator = torch.Generator().manual_seed(42)
    permutation = torch.randperm(num_samples, generator=generator).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]

    train_subset = Subset(train_dataset, train_indices)

    # Create validation dataset from the same underlying dataset but with different indices
    # This ensures truly disjoint splits and proper validation without augmentation
    val_dataset = LensDataset(
        data_root=data_root,
        split="train",
        img_size=img_size,
        augment=False,  # No augmentation for validation
        validate_paths=True,
    )
    val_subset = Subset(val_dataset, val_indices)
    return train_subset, val_subset


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    val_split: float = 0.2,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    shuffle_train: bool = True,
    drop_last: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, validation, test) dataloaders with consistent defaults."""

    loader_config = _LoaderConfig.from_kwargs(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )

    logger.info(
        "Creating dataloaders from %s (batch=%s, workers=%s, img_size=%s, val_split=%s)",
        data_root,
        batch_size,
        num_workers,
        img_size,
        val_split,
    )

    train_dataset, val_dataset = _build_train_val_datasets(
        data_root, img_size, val_split
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle_train,
        **loader_config.as_dict(),
    )

    if val_dataset is None:
        val_loader = DataLoader(
            train_dataset,
            shuffle=False,
            **loader_config.as_dict(),
        )
    else:
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_config.as_dict(),
        )

    test_dataset = LensDataset(
        data_root=data_root,
        split="test",
        img_size=img_size,
        augment=False,
        validate_paths=True,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_config.as_dict(),
    )

    logger.info(
        "Created loaders: train=%d batches, val=%d batches, test=%d batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )

    return train_loader, val_loader, test_loader


def create_single_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    augment: bool = False,
    shuffle: bool = False,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    drop_last: bool = False,
) -> DataLoader:
    """Build a loader for a single split (useful for evaluation-only workflows)."""

    loader_config = _LoaderConfig.from_kwargs(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )

    dataset = LensDataset(
        data_root=data_root,
        split=split,
        img_size=img_size,
        augment=augment,
        validate_paths=True,
    )

    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        **loader_config.as_dict(),
    )

    logger.info(
        "Created %s loader with %d batches (batch=%s, workers=%s)",
        split,
        len(dataloader),
        batch_size,
        num_workers,
    )
    return dataloader


__all__ = ["create_dataloaders", "create_single_dataloader"]
