from __future__ import annotations

from typing import Optional, Dict

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .graph_builder import build_grid_graph
from .physics_ops import PhysicsScale


class _MockLensDataset(Dataset):
    """Placeholder dataset for wiring; replace with project loaders.
    Returns dicts with image and optional targets.
    """

    def __init__(self, length: int = 32, labeled: bool = True) -> None:
        super().__init__()
        self.length = length
        self.labeled = labeled

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict:
        img = torch.randn(3, 224, 224)
        sample: Dict = {"image": img}
        if self.labeled:
            sample["kappa"] = torch.randn(1, 112, 112)
            sample["alpha"] = torch.randn(2, 112, 112)
        return sample


class LensGNNDataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size: int = 4, num_workers: int = 0, pixel_scale_arcsec: float = 0.1
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ps = PhysicsScale(pixel_scale_arcsec=pixel_scale_arcsec)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = _MockLensDataset(labeled=True)
        self.val_ds = _MockLensDataset(labeled=True)

    def _collate(self, batch_list: list[Dict]) -> Dict:
        """
        Collate function that preserves metadata as a list.

        This ensures meta dicts are not lost during batching and remain
        accessible for per-sample operations (e.g., Σcrit normalization).
        """
        images = torch.stack([b["image"] for b in batch_list])

        # Weak/strong astro-safe augmentations (simple rot/flip/shift)
        def weak_aug(x: torch.Tensor) -> torch.Tensor:
            y = x
            if torch.rand(1).item() < 0.5:
                k = torch.randint(0, 4, (1,)).item()
                y = torch.rot90(y, k, dims=(-2, -1))
            if torch.rand(1).item() < 0.5:
                y = y.flip(-2)
            return y

        def strong_aug(x: torch.Tensor) -> torch.Tensor:
            y = weak_aug(x)
            if torch.rand(1).item() < 0.3:
                shift = torch.randint(-2, 3, (2,))
                y = torch.roll(
                    y,
                    shifts=(int(shift[0].item()), int(shift[1].item())),
                    dims=(-2, -1),
                )
            return y

        images_weak = torch.stack([weak_aug(b["image"]) for b in batch_list])

        # Use pixel_scale from meta if available, otherwise use default
        ps = self.ps
        if "meta" in batch_list[0] and batch_list[0]["meta"] is not None:
            # Use first sample's meta for graph building (or merge if needed)
            first_meta = batch_list[0]["meta"]
            if "pixel_scale_arcsec" in first_meta:
                from .physics_ops import PhysicsScale

                ps = PhysicsScale(
                    pixel_scale_arcsec=float(first_meta["pixel_scale_arcsec"])
                )

        graph = build_grid_graph(
            images, patch_size=2, connectivity="8+ring", physics_scale=ps
        )
        graph_weak = build_grid_graph(
            images_weak, patch_size=2, connectivity="8+ring", physics_scale=ps
        )
        out: Dict = {"graph": graph, "graph_weak": graph_weak, "image": images}

        # Targets resized to grid are assumed prepared upstream; here just pass if present
        if "kappa" in batch_list[0]:
            out["target"] = {"kappa": torch.stack([b["kappa"] for b in batch_list])}
            if "alpha" in batch_list[0]:
                out["target"]["alpha"] = torch.stack([b["alpha"] for b in batch_list])

        # Preserve meta as list (one dict per sample)
        if "meta" in batch_list[0]:
            out["meta"] = [b.get("meta") for b in batch_list]

        return out

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )
