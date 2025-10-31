#!/usr/bin/env python3
"""
Survey-specific transforms for astronomical imaging.

- Replace ImageNet defaults with per-survey, per-band normalization
- Allow opt-in color jitter (off by default for physics integrity)

References:
- Lanusse et al. (2021) Astronomical pipelines: per-survey normalization/masks/PSF as core meta
- Euclid Consortium (2022) Data processing and analysis
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torchvision import transforms as T


def make_survey_transforms(
    norm_stats: Dict[str, Dict[str, float]],
    bands: List[str],
    jitter: bool = False,
    image_size: Optional[int] = None,
) -> T.Compose:
    """
    Build transforms with per-band normalization.

    Args:
        norm_stats: {band: {mean, std}}
        bands: band order used by the dataset/model (e.g., ['g','r','i'])
        jitter: enable color jitter (OFF by default to avoid breaking physics)
        image_size: if set, resize to (image_size, image_size)

    Returns:
        torchvision.transforms.Compose
    """
    means = [float(norm_stats[b]["mean"]) for b in bands]
    stds = [float(norm_stats[b]["std"]) for b in bands]

    tf_list: List[torch.nn.Module] = []
    if image_size is not None:
        tf_list.append(T.Resize((int(image_size), int(image_size))))

    # Note: input typically already in CHW tensor; keep ToTensor for PIL inputs
    tf_list.extend([T.ToTensor(), T.Normalize(mean=means, std=stds)])

    if jitter:
        # Conservative jitter magnitudes; off by default
        tf_list.insert(0, T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01))

    return T.Compose(tf_list)
