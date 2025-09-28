#!/usr/bin/env python3
"""
dataset.py
==========
Production-ready PyTorch Dataset for astronomical lens classification.

This module implements scientific computing best practices:
- Comprehensive error handling with informative messages
- Type safety with full type annotations  
- Robust path handling for cross-platform compatibility
- Efficient data loading with proper resource management
- Flexible augmentation pipeline for improved generalization

Expected CSV format:
    filepath,label
    train/lens/lens_train_0001.png,1
    train/nonlens/nonlens_train_0002.png,0

References:
- PyTorch Dataset Best Practices: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- Effective Python Item 89: Consider dataclasses for simple data containers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Union
import warnings

import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image, ImageFile

# Configure PIL to handle truncated images gracefully
# Why: Scientific datasets may contain edge cases that cause PIL errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Setup module logger
logger = logging.getLogger(__name__)


class LensDatasetError(Exception):
    """Custom exception for dataset-related errors.
    
    Why custom exceptions: Enables specific error handling and clearer debugging.
    Reference: Effective Python Item 14 - Prefer exceptions to returning None
    """
    pass


class LensDataset(Dataset):
    """
    PyTorch Dataset for gravitational lens classification.
    
    Features:
    - Robust error handling with informative messages
    - Cross-platform path handling
    - Flexible image preprocessing pipeline
    - Memory-efficient lazy loading
    - Comprehensive validation of data integrity
    
        Args:
        data_root: Root directory containing CSV files and images
        split: Dataset split ('train', 'test', or 'val')
        img_size: Target image size for resizing (square images)
        augment: Whether to apply data augmentation (training only)
        validate_paths: Whether to validate all image paths exist (slower but safer)
        
    Raises:
        LensDatasetError: If CSV file not found or malformed
        FileNotFoundError: If image files are missing
        ValueError: If parameters are invalid
    """
    
    def __init__(
        self, 
        data_root: Union[str, Path] = "data_scientific_test",  # Updated default path
        split: str = "train",
        img_size: int = 64,
        augment: bool = False,
        validate_paths: bool = True
    ) -> None:
        # Input validation with clear error messages
        # Why: Fail fast with helpful errors instead of cryptic failures later
        if img_size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}")
        
        if split not in ["train", "test", "val"]:
            raise ValueError(f"split must be 'train', 'test', or 'val', got '{split}'")
        
        self.data_root = Path(data_root).resolve()  # Absolute path for reliability
        self.split = split
        self.img_size = img_size
        self.augment = augment
        
        logger.info(f"Initializing LensDataset: split={split}, data_root={self.data_root}")
        
        # Load and validate CSV file
        self._load_csv()
        
        # Optional path validation (can be disabled for speed)
        if validate_paths:
            self._validate_image_paths()
        
        # Setup image transforms
        self._setup_transforms()
        
        logger.info(f"Dataset initialized successfully: {len(self)} samples")
    
    def _load_csv(self) -> None:
        """Load and validate CSV file with comprehensive error handling."""
        csv_path = self.data_root / f"{self.split}.csv"
        
        if not csv_path.exists():
            raise LensDatasetError(
                f"CSV file not found: {csv_path}\n"
                f"Expected structure: {self.data_root}/{{train,test}}.csv\n"
                f"Available files: {list(self.data_root.glob('*.csv'))}"
            )
        
        try:
        df = pd.read_csv(csv_path)
        except Exception as e:
            raise LensDatasetError(f"Failed to read CSV {csv_path}: {e}")
        
        # Validate CSV structure
        required_columns = {"filepath", "label"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise LensDatasetError(
                f"CSV missing required columns: {missing}\n"
                f"Found columns: {list(df.columns)}\n"
                f"Required: filepath, label"
            )
        
        # Validate data types and ranges
        if df.empty:
            raise LensDatasetError(f"CSV file is empty: {csv_path}")
        
        # Check for missing values
        if df[["filepath", "label"]].isnull().any().any():
            null_counts = df[["filepath", "label"]].isnull().sum()
            raise LensDatasetError(f"CSV contains null values: {null_counts.to_dict()}")
        
        # Validate label values
        unique_labels = set(df["label"].unique())
        valid_labels = {0, 1}
        if not unique_labels.issubset(valid_labels):
            invalid = unique_labels - valid_labels
            raise LensDatasetError(f"Invalid label values: {invalid}. Expected: 0 or 1")
        
        # Convert to absolute paths for reliability
        # Why: Relative paths can break if working directory changes
        self.image_paths = [self.data_root / path for path in df["filepath"]]
        self.labels = df["label"].astype(int).tolist()

        # Log dataset statistics
        label_counts = df["label"].value_counts().sort_index()
        logger.info(f"Loaded {len(df)} samples: {label_counts.to_dict()}")
    
    def _validate_image_paths(self) -> None:
        """Validate that all image files exist and are readable.
        
        Why: Better to fail early with clear error than during training.
        """
        logger.debug("Validating image file paths...")
        
        missing_files = []
        corrupted_files = []
        
        for i, img_path in enumerate(self.image_paths):
            if not img_path.exists():
                missing_files.append(str(img_path))
                continue
            
            # Quick validation that file can be opened
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Check if image is corrupted
            except Exception as e:
                corrupted_files.append(f"{img_path}: {e}")
        
        # Report any issues found
        if missing_files:
            logger.error(f"Found {len(missing_files)} missing image files")
            raise LensDatasetError(
                f"Missing image files ({len(missing_files)} total):\n" +
                "\n".join(missing_files[:5]) +  # Show first 5
                (f"\n... and {len(missing_files) - 5} more" if len(missing_files) > 5 else "")
            )
        
        if corrupted_files:
            logger.warning(f"Found {len(corrupted_files)} potentially corrupted files")
            for corrupt_file in corrupted_files[:3]:  # Show first 3
                logger.warning(f"Corrupted: {corrupt_file}")
        
        logger.debug("Path validation completed successfully")
    
    def _setup_transforms(self) -> None:
        """Setup image preprocessing pipeline with scientific considerations.
        
        Transform pipeline design:
        1. Resize to consistent dimensions (required for batching)
        2. Data augmentation (training only, preserves physical realism)
        3. Tensor conversion with proper normalization
        
        Why this order: Geometric transforms before tensor conversion for efficiency.
        """
        transforms_list = []
        
        # Base transforms: resize to target size
        transforms_list.append(T.Resize((self.img_size, self.img_size), antialias=True))
        
        # Data augmentation for training (preserve astronomical realism)
        if self.augment and self.split == "train":
            logger.debug("Adding data augmentation transforms")
            
            # Horizontal flip: valid for astronomical images
            transforms_list.append(T.RandomHorizontalFlip(p=0.5))
            
            # Small rotation: gravitational lenses can appear at any orientation
            transforms_list.append(T.RandomRotation(degrees=15, fill=0))
            
            # Brightness/contrast: simulates observing conditions
            transforms_list.append(T.ColorJitter(
                brightness=0.1,  # ±10% brightness
                contrast=0.1,    # ±10% contrast
                saturation=0.0,  # No saturation change (grayscale images)
                hue=0.0         # No hue change
            ))
            
            # Small random crop with padding (simulates different fields of view)
            transforms_list.extend([
                T.Pad(padding=4, fill=0, padding_mode='constant'),
                T.RandomCrop(size=(self.img_size, self.img_size))
            ])
        
        # Convert to tensor: [0,255] uint8 -> [0,1] float32
        # Why ToTensor: Required for PyTorch, handles HWC->CHW conversion
        transforms_list.append(T.ToTensor())
        
        # Normalize to standard range for better training stability
        # Using ImageNet statistics as reasonable defaults for natural images
        transforms_list.append(T.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        ))
        
        self.transform = T.Compose(transforms_list)
        
        logger.debug(f"Configured {len(transforms_list)} image transforms")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label_tensor)
            - image_tensor: Shape (C, H, W), dtype float32, normalized
            - label_tensor: Shape (), dtype long, values {0, 1}
            
        Raises:
            IndexError: If idx is out of range
            LensDatasetError: If image cannot be loaded
        """
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image with proper resource management
            # Why context manager: Ensures file handles are properly closed
            with Image.open(img_path) as img:
                # Convert to RGB for consistency (handles grayscale and RGBA)
                # Why RGB: ResNet expects 3 channels, handles all input formats
                img = img.convert("RGB")
                
                # Apply preprocessing transforms
                img_tensor = self.transform(img)
                
        except Exception as e:
            raise LensDatasetError(
                f"Failed to load image {img_path}: {e}\n"
                f"Image index: {idx}, Label: {label}"
            )
        
        # Convert label to tensor with correct dtype for classification
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, label_tensor
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets.
        
        Returns:
            Tensor of shape (2,) with weights for classes [0, 1]
            
        Why class weights: Helps with imbalanced datasets by giving more
        weight to underrepresented classes during training.
        """
        label_counts = torch.bincount(torch.tensor(self.labels))
        total_samples = len(self.labels)
        
        # Inverse frequency weighting
        class_weights = total_samples / (2.0 * label_counts.float())
        
        logger.info(f"Class weights: {class_weights.tolist()}")
        return class_weights
    
    def get_dataset_stats(self) -> dict:
        """
        Get comprehensive dataset statistics for analysis.
        
        Returns:
            Dictionary with dataset statistics
        """
        labels_tensor = torch.tensor(self.labels)
        
        stats = {
            "total_samples": len(self),
            "num_classes": 2,
            "class_distribution": {
                "class_0": int((labels_tensor == 0).sum()),
                "class_1": int((labels_tensor == 1).sum())
            },
            "class_balance": float((labels_tensor == 1).float().mean()),
            "data_root": str(self.data_root),
            "split": self.split,
            "image_size": self.img_size,
            "augmentation": self.augment
        }
        
        return stats


def create_dataloaders(
    data_root: Union[str, Path] = "data_scientific_test",
    batch_size: int = 32,
    img_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
    val_split: float = 0.1
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train/validation/test dataloaders with best practices.
    
    Args:
        data_root: Root directory containing datasets
        batch_size: Batch size for all loaders
        img_size: Image size for preprocessing
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        val_split: Fraction of training data to use for validation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Why this function: Centralized dataloader creation with consistent settings.
    """
    logger.info(f"Creating dataloaders: batch_size={batch_size}, img_size={img_size}")
    
    # Create datasets
    full_train_dataset = LensDataset(
        data_root=data_root, 
        split="train", 
        img_size=img_size, 
        augment=True
    )
    
    test_dataset = LensDataset(
        data_root=data_root, 
        split="test", 
        img_size=img_size, 
        augment=False
    )
    
    # Split training data into train/validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible splits
    )
    
    logger.info(f"Dataset splits: train={train_size}, val={val_size}, test={len(test_dataset)}")
    
    # Create dataloaders with optimized settings
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "persistent_workers": num_workers > 0  # Keeps workers alive between epochs
    }
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=True,  # Shuffle for training
        **common_kwargs
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        shuffle=False,  # No shuffle for validation
        **common_kwargs
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        shuffle=False,  # No shuffle for testing
        **common_kwargs
    )
    
    return train_loader, val_loader, test_loader


# Example usage and testing
if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Test dataset creation
    try:
        dataset = LensDataset(
            data_root="data_scientific_test",
            split="train",
            img_size=64,
            augment=True
        )
        
        print("Dataset Statistics:")
        stats = dataset.get_dataset_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test sample loading
        img, label = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image shape: {img.shape}")
        print(f"  Image dtype: {img.dtype}")
        print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"  Label: {label}")
        
        print("\nDataset creation successful!")
        
    except Exception as e:
        print(f"Dataset creation failed: {e}")
        raise