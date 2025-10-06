"""
LensDataset implementation for gravitational lensing detection.

The dataset expects a ``<split>.csv`` file alongside image folders under
``data_root``. Each CSV row must provide a ``filepath`` (relative or
absolute) and ``label`` column. Images are lazily loaded and transformed
into normalized tensors compatible with ImageNet pretraining defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

import logging

logger = logging.getLogger(__name__)


class LensDatasetError(Exception):
    """Raised when the dataset cannot be initialised or accessed correctly."""


@dataclass
class _TransformsConfig:
    img_size: int
    augment: bool

    def build(self) -> Callable[[Image.Image], torch.Tensor]:
        """Create the torchvision transformation pipeline."""
        base_transforms = [
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        if not self.augment:
            return T.Compose(base_transforms)

        aug_transforms = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
        return T.Compose(aug_transforms + base_transforms)


class LensDataset(Dataset[Tuple[torch.Tensor, int]]):
    """PyTorch ``Dataset`` for gravitational lens classification images."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        img_size: int = 224,
        augment: bool = False,
        validate_paths: bool = True,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split.lower()
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.validate_paths = bool(validate_paths)

        if not self.data_root.exists():
            raise LensDatasetError(f"Data root directory does not exist: {self.data_root}")

        csv_name = self._resolve_split_name(self.split)
        csv_path = self.data_root / f"{csv_name}.csv"
        if not csv_path.exists():
            raise LensDatasetError(f"CSV manifest not found for split '{self.split}': {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - surfaced to caller
            raise LensDatasetError(f"Failed to read CSV at {csv_path}: {exc}") from exc

        required_columns = {"filepath", "label"}
        missing = sorted(required_columns.difference(df.columns))
        if missing:
            raise LensDatasetError(
                f"Missing required columns in {csv_path}: {', '.join(missing)}"
            )

        # Remove rows with NaNs or empty strings in required columns
        df = df.dropna(subset=required_columns)
        df = df[df["filepath"].astype(str).str.len() > 0]

        # Normalise labels to integers
        try:
            df["label"] = df["label"].astype(int)
        except ValueError as exc:
            raise LensDatasetError(f"Labels must be integer-compatible: {exc}") from exc

        self.df = df.reset_index(drop=True)

        if self.validate_paths:
            self._validate_manifest_paths()

        self.transform = transform or _TransformsConfig(self.img_size, self.augment).build()
        logger.info("Loaded %d samples for split '%s'", len(self.df), self.split)

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_split_name(split: str) -> str:
        mapping = {
            "train": "train",
            "training": "train",
            "val": "val",
            "valid": "val",
            "validation": "val",
            "test": "test",
        }
        return mapping.get(split, split)

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        return candidate if candidate.is_absolute() else (self.data_root / candidate)

    def _validate_manifest_paths(self) -> None:
        missing: list[str] = []
        for path_str in self.df["filepath"]:
            path = self._resolve_path(str(path_str))
            if not path.exists():
                missing.append(str(path))
                if len(missing) >= 10:
                    break
        if missing:
            hint = ", ".join(missing)
            raise LensDatasetError(f"Missing image files (first {len(missing)} shown): {hint}")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        if index < 0 or index >= len(self.df):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.df)}")

        row = self.df.iloc[index]
        image_path = self._resolve_path(str(row["filepath"]))
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:  # pragma: no cover - escalated to caller
            raise LensDatasetError(f"Failed to load image {image_path}: {exc}") from exc

        tensor = self.transform(image)
        label = int(row["label"])
        return tensor, label

    # Convenience helpers ------------------------------------------------
    def get_class_counts(self) -> dict[int, int]:
        return self.df["label"].value_counts().astype(int).to_dict()

    def get_sample_paths(self) -> list[str]:
        return [str(self._resolve_path(p)) for p in self.df["filepath"]]


__all__ = ["LensDataset", "LensDatasetError"]
