#!/usr/bin/env python3
"""
FITS-based dataset loader for gravitational lensing with multi-HDU support.

Supports:
- Multi-HDU FITS files with bands in separate HDUs
- Optional κ/ψ/α maps, masks, PSF in HDUs or separate files
- Full metadata extraction (pixel scale, redshifts, Σcrit)
- Soft labels and unlabeled samples
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from astropy.io import fits
except ImportError:
    raise ImportError(
        "astropy is required for FITS dataset. Install with: pip install astropy"
    )

logger = logging.getLogger(__name__)


class LensFITSDataset(Dataset):
    """
    Dataset for loading gravitational lensing data from FITS files.

    Supports multi-HDU FITS with bands, maps (κ/ψ/α), masks, and PSF.
    Extracts full metadata including pixel scales, redshifts, and Σcrit.

    References:
    - Lanusse, F., Leonard, A., & Pospieszalski, R. (2021). "Astronomical Data Analysis and
      Open-source Pipelines" (ADVERTISE/DECaLS). Per-band, per-survey normalization and
      meta/PSF/mask propagation are essential.
    - Euclid Consortium (2022). "Data Processing and Analysis for Euclid."
      Astronomical ML is unreliable if not tested on true FITS (multi-HDU, meta, masks, etc.)
      and tuned with domain-consistent normalization.

    Astronomical ML pipelines require native FITS handling with proper metadata extraction
    for reproducibility and physical correctness. ImageNet defaults are inappropriate.
    """

    def __init__(
        self,
        csv_path: str | Path,
        band_hdus: Dict[str, int],
        map_hdus: Optional[Dict[str, int]] = None,
        mask_hdu: Optional[int] = None,
        psf_hdu: Optional[int] = None,
        normalize: bool = True,
        transforms: Optional[Callable] = None,
        require_sigma_crit: bool = False,
    ):
        """
        Initialize FITS dataset.

        Args:
            csv_path: Path to CSV manifest with columns:
                - filepath: Path to FITS file
                - label: int (0/1) or float (soft label) or None/-1 (unlabeled)
                - kappa_path, psi_path, alpha_path: Optional paths to map files
                - mask_path: Optional path to mask file
                - psf_path: Optional path to PSF file
                - pixel_scale_arcsec: Optional pixel scale (arcsec/pixel)
                - pixel_scale_y_arcsec: Optional y-axis scale (if different)
                - z_l, z_s: Optional lens/source redshifts
                - sigma_crit: Optional critical surface density
                - band_order: Optional comma-separated band names (e.g., "g,r,i")
            band_hdus: Dict mapping band names to HDU indices, e.g., {'g': 0, 'r': 1, 'i': 2}
            map_hdus: Dict mapping map type to HDU index, e.g., {'kappa': 4, 'psi': 5}
            mask_hdu: HDU index for mask (if in same file)
            psf_hdu: HDU index for PSF (if in same file)
            normalize: Whether to normalize images to [0, 1]
            transforms: Optional torchvision transform
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV manifest not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        # Validate required columns
        if "filepath" not in self.df.columns:
            raise ValueError("CSV must contain 'filepath' column")

        self.band_hdus = band_hdus
        self.band_order = list(band_hdus.keys())
        self.map_hdus = map_hdus or {}
        self.mask_hdu = mask_hdu
        self.psf_hdu = psf_hdu
        self.normalize = normalize
        self.transforms = transforms
        self.require_sigma_crit = require_sigma_crit

        logger.info(
            f"Loaded {len(self.df)} samples from {self.csv_path} "
            f"with bands {self.band_order}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load sample from FITS file.

        Returns dict with:
            - image: [C, H, W] tensor
            - label: int, float, or None
            - kappa, psi, alpha: Optional [1, H, W] or [2, H, W] tensors
            - mask: Optional [H, W] bool tensor
            - psf: Optional tensor
            - meta: Dict with bands, pixel_scale_arcsec, dx, dy, z_l, z_s, sigma_crit
        """
        row = self.df.iloc[idx]
        fits_path = Path(row["filepath"])

        if not fits_path.is_absolute():
            # Resolve relative to CSV directory
            fits_path = self.csv_path.parent / fits_path

        if not fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {fits_path}")

        sample: Dict[str, Any] = {}

        # Load FITS file
        with fits.open(fits_path) as hdul:
            # Load bands
            band_data = []
            for band_name in self.band_order:
                hdu_idx = self.band_hdus[band_name]
                if hdu_idx >= len(hdul):
                    raise ValueError(
                        f"HDU {hdu_idx} for band '{band_name}' not found in {fits_path}"
                    )
                data = hdul[hdu_idx].data
                if data is None:
                    raise ValueError(f"HDU {hdu_idx} data is None")
                band_data.append(np.asarray(data, dtype=np.float32))

            # Stack bands: [C, H, W]
            image = np.stack(band_data, axis=0)

            # Handle NaN/inf
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize to [0, 1] if requested
            if self.normalize:
                img_min = image.min()
                img_max = image.max()
                if img_max > img_min:
                    image = (image - img_min) / (img_max - img_min + 1e-8)

            sample["image"] = torch.from_numpy(image).float()

            # Load maps from HDUs if specified
            for map_name, hdu_idx in self.map_hdus.items():
                if hdu_idx < len(hdul):
                    map_data = hdul[hdu_idx].data
                    if map_data is not None:
                        map_data = np.asarray(map_data, dtype=np.float32)
                        map_data = np.nan_to_num(
                            map_data, nan=0.0, posinf=0.0, neginf=0.0
                        )

                        if map_name == "alpha":
                            # Alpha is 2-channel [2, H, W]
                            if map_data.ndim == 2:
                                # Single HDU with both components? Split or assume [H, W, 2]
                                if map_data.shape[-1] == 2:
                                    map_data = map_data.transpose(2, 0, 1)
                                else:
                                    raise ValueError(
                                        f"Alpha map shape unclear: {map_data.shape}"
                                    )
                            sample[map_name] = torch.from_numpy(map_data).float()
                        else:
                            # κ, ψ are single-channel
                            if map_data.ndim == 2:
                                map_data = map_data[np.newaxis, :, :]  # [1, H, W]
                            sample[map_name] = torch.from_numpy(map_data).float()

            # Load mask if in same file
            if self.mask_hdu is not None and self.mask_hdu < len(hdul):
                mask_data = hdul[self.mask_hdu].data
                if mask_data is not None:
                    sample["mask"] = torch.from_numpy(np.asarray(mask_data, dtype=bool))

            # Load PSF if in same file
            if self.psf_hdu is not None and self.psf_hdu < len(hdul):
                psf_data = hdul[self.psf_hdu].data
                if psf_data is not None:
                    sample["psf"] = torch.from_numpy(
                        np.asarray(psf_data, dtype=np.float32)
                    )

            # Extract header metadata
            header = hdul[0].header

        # Load maps from separate files if paths provided
        for map_name in ["kappa", "psi", "alpha"]:
            col_name = f"{map_name}_path"
            if col_name in row and pd.notna(row[col_name]):
                map_path = Path(row[col_name])
                if not map_path.is_absolute():
                    map_path = self.csv_path.parent / map_path

                if map_path.exists():
                    with fits.open(map_path) as map_hdul:
                        map_data = map_hdul[0].data
                        if map_data is not None:
                            map_data = np.asarray(map_data, dtype=np.float32)
                            map_data = np.nan_to_num(map_data, nan=0.0)
                            if map_name == "alpha" and map_data.ndim == 3:
                                # Assume [2, H, W] or [H, W, 2]
                                if map_data.shape[0] != 2:
                                    map_data = map_data.transpose(2, 0, 1)
                            elif map_data.ndim == 2:
                                map_data = map_data[np.newaxis, :, :]
                            sample[map_name] = torch.from_numpy(map_data).float()

        # Load mask from separate file
        if "mask_path" in row and pd.notna(row["mask_path"]):
            mask_path = Path(row["mask_path"])
            if not mask_path.is_absolute():
                mask_path = self.csv_path.parent / mask_path
            if mask_path.exists():
                with fits.open(mask_path) as mask_hdul:
                    mask_data = mask_hdul[0].data
                    if mask_data is not None:
                        sample["mask"] = torch.from_numpy(
                            np.asarray(mask_data, dtype=bool)
                        )

        # Load PSF from separate file
        if "psf_path" in row and pd.notna(row["psf_path"]):
            psf_path = Path(row["psf_path"])
            if not psf_path.is_absolute():
                psf_path = self.csv_path.parent / psf_path
            if psf_path.exists():
                with fits.open(psf_path) as psf_hdul:
                    psf_data = psf_hdul[0].data
                    if psf_data is not None:
                        sample["psf"] = torch.from_numpy(
                            np.asarray(psf_data, dtype=np.float32)
                        )

        # Extract metadata - require explicit pixel_scale_arcsec (no silent defaults)
        pixel_scale_arcsec = row.get("pixel_scale_arcsec", None)
        if pixel_scale_arcsec is None or pd.isna(pixel_scale_arcsec):
            pixel_scale_arcsec = header.get("PIXSCALE", None)
        if pixel_scale_arcsec is None:
            raise ValueError(
                f"FITS sample {idx} missing 'pixel_scale_arcsec' (cannot derive dx/dy). "
                f"Provide in CSV column or FITS header 'PIXSCALE'."
            )
        pixel_scale_arcsec = float(pixel_scale_arcsec)

        # Get y-scale (can default to x-scale if not specified)
        pixel_scale_y_arcsec = row.get("pixel_scale_y_arcsec", None)
        if pixel_scale_y_arcsec is None or pd.isna(pixel_scale_y_arcsec):
            pixel_scale_y_arcsec = pixel_scale_arcsec
        else:
            pixel_scale_y_arcsec = float(pixel_scale_y_arcsec)

        # Convert to radians
        dx = pixel_scale_arcsec * (math.pi / 180.0 / 3600.0)
        dy = pixel_scale_y_arcsec * (math.pi / 180.0 / 3600.0)

        # Get band order from CSV or use default
        bands = self.band_order
        if "band_order" in row and pd.notna(row["band_order"]):
            bands = [b.strip() for b in str(row["band_order"]).split(",")]

        # Extract sigma_crit - require explicit for physics pipelines
        sigma_crit = row.get("sigma_crit", None)
        if sigma_crit is None or pd.isna(sigma_crit):
            sigma_crit = None
        else:
            sigma_crit = float(sigma_crit)

        sample["meta"] = {
            "bands": bands,
            "pixel_scale_arcsec": pixel_scale_arcsec,
            "pixel_scale_y_arcsec": pixel_scale_y_arcsec,
            "dx": dx,
            "dy": dy,
            "z_l": row.get("z_l", None)
            if "z_l" in row and pd.notna(row.get("z_l"))
            else None,
            "z_s": row.get("z_s", None)
            if "z_s" in row and pd.notna(row.get("z_s"))
            else None,
            "sigma_crit": sigma_crit,
            "filepath": str(fits_path),
        }

        # Physics pipelines must declare sigma_crit explicitly when require_sigma_crit=True
        if self.require_sigma_crit and sample["meta"]["sigma_crit"] is None:
            raise ValueError(
                f"FITS sample {idx} missing 'sigma_crit' with require_sigma_crit=True. "
                f"Provide in CSV column for physics pipelines."
            )

        # Handle label (int, float, or None for unlabeled)
        if "label" in row:
            label_val = row["label"]
            if pd.isna(label_val) or label_val == -1:
                sample["label"] = None  # Unlabeled
            elif isinstance(label_val, float) and 0.0 <= label_val <= 1.0:
                sample["label"] = float(label_val)  # Soft label
            else:
                sample["label"] = int(label_val)  # Hard label
        else:
            sample["label"] = None

        # Apply transforms if provided
        if self.transforms is not None:
            sample["image"] = self.transforms(sample["image"])

        return sample
