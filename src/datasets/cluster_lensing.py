#!/usr/bin/env python3
"""
Cluster lensing dataset with large FOV tiling and coords grid support.

- Handles large images (512–2048 px), optional tiling with overlap
- Provides coords grid (arcsec) and meta including pixel scales, redshifts, Σcrit
- Supports optional masks and PSF per cutout

References:
- Gruen & Brimioulle (2015) Cluster lensing requirements
- Bosch et al. (2018) HSC software pipeline and tiling
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from astropy.io import fits


@dataclass
class TilingConfig:
    tile: int = 512
    overlap: int = 32


class ClusterLensingDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        tiling: Optional[TilingConfig] = None,
        normalize: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.tiling = tiling
        self.normalize = normalize

        required = {"filepath", "pixel_scale_arcsec"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_fits(self, path: Path) -> np.ndarray:
        with fits.open(path) as hdul:
            img = np.asarray(hdul[0].data, dtype=np.float32)
        # handle NaNs
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        if self.normalize:
            vmin, vmax = float(np.min(img)), float(np.max(img))
            if vmax > vmin:
                img = (img - vmin) / (vmax - vmin + 1e-8)
        return img

    def _coords_grid(self, h: int, w: int, arcsec: float) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        xx = xx * arcsec
        yy = yy * arcsec
        return torch.stack([xx, yy], dim=0)  # [2, H, W]

    def _tile_image(self, img: torch.Tensor, tile: int, overlap: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _, H, W = img.shape
        stride = tile - overlap
        xs = list(range(0, max(1, W - tile + 1), stride))
        ys = list(range(0, max(1, H - tile + 1), stride))
        tiles = []
        coords = []
        for y in ys:
            for x in xs:
                patch = img[:, y:y+tile, x:x+tile]
                # pad if on border
                pad_h = max(0, tile - patch.shape[-2])
                pad_w = max(0, tile - patch.shape[-1])
                if pad_h or pad_w:
                    patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h))
                tiles.append(patch)
                coords.append((y, x))
        return torch.stack(tiles), torch.tensor(coords, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        fits_path = Path(row["filepath"]) if Path(row["filepath"]).is_absolute() else (self.csv_path.parent / row["filepath"]).resolve()
        img_np = self._load_fits(fits_path)
        img = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W]

        arcsec = float(row.get("pixel_scale_arcsec", 0.2))
        dx = arcsec * (np.pi / 180.0 / 3600.0)
        dy = dx

        sample: Dict[str, Any] = {
            "image": img,
            "mask": None,
            "psf": None,
            "meta": {
                "pixel_scale_arcsec": arcsec,
                "dx": dx,
                "dy": dy,
                "coords_grid": self._coords_grid(img.shape[-2], img.shape[-1], arcsec),
                "cluster_id": row.get("cluster_id", None),
            },
        }

        if self.tiling is not None:
            tiles, coords = self._tile_image(img, self.tiling.tile, self.tiling.overlap)
            sample["tiles"] = tiles  # [N, 1, tile, tile]
            sample["tile_coords"] = coords  # [N, 2] (y, x)

        # Optional mask or PSF
        if "mask_path" in row and pd.notna(row["mask_path"]):
            mpath = Path(row["mask_path"]) if Path(row["mask_path"]).is_absolute() else (self.csv_path.parent / row["mask_path"]).resolve()
            if mpath.exists():
                with fits.open(mpath) as mh:
                    m = np.asarray(mh[0].data, dtype=np.float32)
                sample["mask"] = torch.from_numpy(m > 0)
        if "psf_path" in row and pd.notna(row["psf_path"]):
            ppath = Path(row["psf_path"]) if Path(row["psf_path"]).is_absolute() else (self.csv_path.parent / row["psf_path"]).resolve()
            if ppath.exists():
                with fits.open(ppath) as ph:
                    p = np.asarray(ph[0].data, dtype=np.float32)
                sample["psf"] = torch.from_numpy(p)

        return sample
