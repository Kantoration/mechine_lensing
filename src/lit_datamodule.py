#!/usr/bin/env python3
"""
Lightning DataModule for gravitational lens classification.

This module provides LightningDataModule wrappers for dataset streaming,
supporting both local datasets and cloud-hosted WebDataset shards.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms as T
import webdataset as wds
import fsspec
import io
from PIL import Image

from .datasets.lens_dataset import LensDataset

logger = logging.getLogger(__name__)


def _decode_webdataset_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode WebDataset sample to standard format.

    Expected sample format: {"jpg": bytes, "cls": int}
    """
    try:
        # Decode image
        img = Image.open(io.BytesIO(sample["jpg"])).convert("RGB")

        # Get label
        label = int(sample["cls"])

        return {"image": img, "label": label}
    except Exception as e:
        logger.warning(f"Failed to decode sample: {e}")
        # Return a dummy sample
        return {"image": Image.new("RGB", (224, 224), color="black"), "label": 0}


class LensDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for gravitational lens classification.

    Supports both local datasets and cloud-hosted WebDataset shards.
    """

    def __init__(
        self,
        # Data source configuration
        data_root: Optional[Union[str, Path]] = None,
        train_urls: Optional[str] = None,
        val_urls: Optional[str] = None,
        test_urls: Optional[str] = None,
        # Dataset configuration
        batch_size: int = 64,
        num_workers: int = 8,
        image_size: int = 224,
        val_split: float = 0.1,
        # Augmentation configuration
        augment: bool = True,
        # WebDataset configuration
        shuffle_buffer_size: int = 10000,
        cache_dir: Optional[str] = None,
        # Data loading configuration
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ):
        """
        Initialize Lightning DataModule.

        Args:
            data_root: Local data root directory (for local datasets)
            train_urls: WebDataset URLs for training data (e.g., "s3://bucket/train-{0000..0099}.tar")
            val_urls: WebDataset URLs for validation data
            test_urls: WebDataset URLs for test data
            batch_size: Batch size for data loaders
            num_workers: Number of data loading workers
            image_size: Image size for preprocessing
            val_split: Validation split fraction (for local datasets)
            augment: Whether to apply data augmentation
            shuffle_buffer_size: Buffer size for WebDataset shuffling
            cache_dir: Directory for caching WebDataset samples
            pin_memory: Whether to pin memory for faster GPU transfer
            persistent_workers: Whether to use persistent workers
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Determine data source
        self.use_webdataset = train_urls is not None
        self.data_root = Path(data_root) if data_root else None

        # Setup transforms
        self._setup_transforms()

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _setup_transforms(self) -> None:
        """Setup image transforms for training and validation with survey-aware normalization."""
        from src.datasets.transforms import make_survey_transforms

        # Default astronomy normalization: zero-mean, unit-variance per band
        # Replace with survey-specific stats from ModelContract/dataset metadata when available
        num_bands = getattr(self.hparams, "num_bands", 3)
        default_stats = {
            f"band_{i}": {"mean": 0.0, "std": 1.0} for i in range(num_bands)
        }
        default_bands = [f"band_{i}" for i in range(num_bands)]

        # Use survey-aware transforms (will fall back to defaults if contract not available)
        # Color jitter is opt-in only (default OFF for physics integrity)
        use_jitter = getattr(self.hparams, "use_color_jitter", False)

        base_transforms = make_survey_transforms(
            norm_stats=default_stats,
            bands=default_bands,
            jitter=False,  # Base transforms never use jitter
            image_size=self.hparams.image_size,
        )

        # Training transforms with augmentation
        if self.hparams.augment:
            train_transforms = [
                T.Resize((self.hparams.image_size, self.hparams.image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
            ]
            # Color jitter only if explicitly enabled
            if use_jitter:
                train_transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))
            train_transforms.extend(
                [
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.0] * num_bands, std=[1.0] * num_bands
                    ),  # Astronomy default
                ]
            )
            train_transform = T.Compose(train_transforms)
        else:
            train_transform = base_transforms

        self.train_transform = train_transform
        self.val_transform = base_transforms
        self.test_transform = base_transforms

    def _create_webdataset_pipeline(
        self, urls: str, training: bool = True
    ) -> wds.WebDataset:
        """Create WebDataset pipeline for streaming data."""
        # Create WebDataset
        dataset = wds.WebDataset(
            urls, handler=wds.warn_and_continue, cache_dir=self.hparams.cache_dir
        )

        # Decode samples
        dataset = dataset.decode().map(_decode_webdataset_sample)

        # Apply transforms
        transform = self.train_transform if training else self.val_transform
        dataset = dataset.map(
            lambda sample: {
                "image": transform(sample["image"]),
                "label": sample["label"],
            }
        )

        # Shuffle and repeat for training
        if training:
            dataset = dataset.shuffle(self.hparams.shuffle_buffer_size).repeat()

        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for the current stage."""
        if self.use_webdataset:
            self._setup_webdataset()
        else:
            self._setup_local_dataset()

    def _setup_webdataset(self) -> None:
        """Setup WebDataset-based datasets."""
        if self.hparams.train_urls:
            self.train_dataset = self._create_webdataset_pipeline(
                self.hparams.train_urls, training=True
            )

        if self.hparams.val_urls:
            self.val_dataset = self._create_webdataset_pipeline(
                self.hparams.val_urls, training=False
            )

        if self.hparams.test_urls:
            self.test_dataset = self._create_webdataset_pipeline(
                self.hparams.test_urls, training=False
            )

    def _setup_local_dataset(self) -> None:
        """Setup local dataset-based datasets."""
        if not self.data_root or not self.data_root.exists():
            raise ValueError(f"Data root not found: {self.data_root}")

        # Create datasets
        self.train_dataset = LensDataset(
            data_root=self.data_root,
            split="train",
            img_size=self.hparams.image_size,
            augment=self.hparams.augment,
        )

        self.val_dataset = LensDataset(
            data_root=self.data_root,
            split="val",
            img_size=self.hparams.image_size,
            augment=False,
        )

        self.test_dataset = LensDataset(
            data_root=self.data_root,
            split="test",
            img_size=self.hparams.image_size,
            augment=False,
        )

    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not setup. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers
            and self.hparams.num_workers > 0,
            shuffle=not self.use_webdataset,  # WebDataset handles shuffling internally
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not setup. Call setup() first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers
            and self.hparams.num_workers > 0,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not setup. Call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers
            and self.hparams.num_workers > 0,
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create prediction data loader (same as test)."""
        return self.test_dataloader()


class LensWebDatasetDataModule(pl.LightningDataModule):
    """
    Specialized DataModule for WebDataset streaming from cloud storage.

    This class provides optimized streaming for large-scale datasets
    hosted on S3, GCS, or other cloud storage systems.
    """

    def __init__(
        self,
        # Cloud storage configuration
        train_urls: str,
        val_urls: str,
        test_urls: Optional[str] = None,
        # Dataset configuration
        batch_size: int = 64,
        num_workers: int = 8,
        image_size: int = 224,
        # WebDataset specific
        shuffle_buffer_size: int = 10000,
        cache_dir: Optional[str] = None,
        # Data loading
        pin_memory: bool = True,
        persistent_workers: bool = True,
        # Augmentation
        augment: bool = True,
        **kwargs,
    ):
        """
        Initialize WebDataset DataModule.

        Args:
            train_urls: WebDataset URLs for training (e.g., "s3://bucket/train-{0000..0099}.tar")
            val_urls: WebDataset URLs for validation
            test_urls: WebDataset URLs for test (optional)
            batch_size: Batch size
            num_workers: Number of workers
            image_size: Image size
            shuffle_buffer_size: Shuffle buffer size
            cache_dir: Cache directory
            pin_memory: Pin memory
            persistent_workers: Persistent workers
            augment: Data augmentation
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Setup transforms
        self._setup_transforms()

    def _setup_transforms(self) -> None:
        """Setup transforms for WebDataset with survey-aware normalization."""
        num_bands = getattr(self.hparams, "num_bands", 3)
        use_jitter = getattr(self.hparams, "use_color_jitter", False)

        # Training transforms
        if self.hparams.augment:
            train_transforms = [
                T.Resize((self.hparams.image_size, self.hparams.image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
            ]
            # Color jitter only if explicitly enabled (default OFF)
            if use_jitter:
                train_transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))
            train_transforms.extend(
                [
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.0] * num_bands, std=[1.0] * num_bands
                    ),  # Astronomy default
                ]
            )
        else:
            train_transforms = [
                T.Resize((self.hparams.image_size, self.hparams.image_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.0] * num_bands, std=[1.0] * num_bands
                ),  # Astronomy default
            ]

        # Validation/test transforms
        val_transforms = [
            T.Resize((self.hparams.image_size, self.hparams.image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.0] * num_bands, std=[1.0] * num_bands
            ),  # Astronomy default
        ]

        self.train_transform = T.Compose(train_transforms)
        self.val_transform = T.Compose(val_transforms)

    def _create_pipeline(self, urls: str, training: bool = True) -> wds.WebDataset:
        """Create WebDataset pipeline."""
        # Create WebDataset
        dataset = wds.WebDataset(
            urls, handler=wds.warn_and_continue, cache_dir=self.hparams.cache_dir
        )

        # Decode and transform
        transform = self.train_transform if training else self.val_transform
        dataset = (
            dataset.decode()
            .map(_decode_webdataset_sample)
            .map(
                lambda sample: {
                    "image": transform(sample["image"]),
                    "label": sample["label"],
                }
            )
        )

        # Shuffle and repeat for training
        if training:
            dataset = dataset.shuffle(self.hparams.shuffle_buffer_size).repeat()

        return dataset

    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        dataset = self._create_pipeline(self.hparams.train_urls, training=True)

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers
            and self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        dataset = self._create_pipeline(self.hparams.val_urls, training=False)

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers
            and self.hparams.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if self.hparams.test_urls is None:
            return self.val_dataloader()

        dataset = self._create_pipeline(self.hparams.test_urls, training=False)

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers
            and self.hparams.num_workers > 0,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create prediction data loader."""
        return self.test_dataloader()


# Utility functions for dataset preparation


def create_webdataset_shards(
    data_root: Union[str, Path],
    output_dir: Union[str, Path],
    shard_size: int = 1000,
    image_size: int = 224,
    quality: int = 95,
) -> None:
    """
    Create WebDataset shards from local dataset.

    Args:
        data_root: Root directory of local dataset
        output_dir: Output directory for shards
        shard_size: Number of samples per shard
        image_size: Image size for compression
        quality: JPEG quality (1-100)
    """
    import tarfile
    from pathlib import Path

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all splits
    splits = ["train", "val", "test"]

    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            continue

        # Get all images
        images = list(split_dir.glob("*.png")) + list(split_dir.glob("*.jpg"))

        # Create shards
        shard_idx = 0
        sample_idx = 0

        current_shard = None
        current_shard_samples = 0

        for img_path in images:
            # Open new shard if needed
            if current_shard is None or current_shard_samples >= shard_size:
                if current_shard is not None:
                    current_shard.close()

                shard_path = output_dir / f"{split}-{shard_idx:04d}.tar"
                current_shard = tarfile.open(shard_path, "w")
                current_shard_samples = 0
                shard_idx += 1

            # Load and compress image
            img = Image.open(img_path).convert("RGB")
            img = img.resize((image_size, image_size))

            # Compress to JPEG
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG", quality=quality)
            img_bytes = img_bytes.getvalue()

            # Determine label from filename or directory structure
            label = 1 if "lens" in img_path.name.lower() else 0

            # Add to shard
            img_info = tarfile.TarInfo(name=f"{sample_idx:06d}.jpg")
            img_info.size = len(img_bytes)
            current_shard.addfile(img_info, io.BytesIO(img_bytes))

            # Add label
            label_bytes = str(label).encode()
            label_info = tarfile.TarInfo(name=f"{sample_idx:06d}.cls")
            label_info.size = len(label_bytes)
            current_shard.addfile(label_info, io.BytesIO(label_bytes))

            current_shard_samples += 1
            sample_idx += 1

        # Close final shard
        if current_shard is not None:
            current_shard.close()

        logger.info(
            f"Created {shard_idx} shards for {split} split with {sample_idx} samples"
        )


def upload_shards_to_cloud(
    local_dir: Union[str, Path],
    cloud_url: str,
    storage_options: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Upload WebDataset shards to cloud storage.

    Args:
        local_dir: Local directory containing shards
        cloud_url: Cloud storage URL (e.g., "s3://bucket/path/")
        storage_options: Storage options for fsspec
    """

    local_dir = Path(local_dir)
    fs = fsspec.filesystem(cloud_url.split("://")[0], **(storage_options or {}))

    # Upload all tar files
    for tar_file in local_dir.glob("*.tar"):
        remote_path = f"{cloud_url}/{tar_file.name}"
        logger.info(f"Uploading {tar_file} to {remote_path}")
        fs.put(str(tar_file), remote_path)

    logger.info(f"Uploaded all shards to {cloud_url}")
