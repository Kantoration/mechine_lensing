from __future__ import annotations

from typing import Callable, Dict, Any, Tuple

import torch
import torch.nn.functional as F


def _hanning_window(tile: int, device: torch.device) -> torch.Tensor:
    w1 = torch.hann_window(tile, periodic=False, dtype=torch.float32, device=device)
    w2 = w1.view(-1, 1) * w1.view(1, -1)
    return w2  # [tile, tile]


def predict_full(model_fn: Callable[[torch.Tensor], Dict[str, torch.Tensor]], image: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Run model_fn on full image [1,C,H,W] and return dict with 'kappa' at [1,1,H,W].
    """
    model_out = model_fn(image)
    return model_out


def predict_tiled(
    model_fn: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
    image: torch.Tensor,
    tile: int = 128,
    overlap: int = 32,
    key: str = "kappa",
) -> Dict[str, torch.Tensor]:
    """
    Tiled inference with Hanning blending. Expects model_fn to return dict containing `key` map.
    Returns dict with stitched map at original resolution.
    """
    assert image.ndim == 4 and image.shape[0] == 1
    _, _, H, W = image.shape
    device = image.device

    stride = tile - overlap
    win = _hanning_window(tile, device=device)

    acc = torch.zeros(1, 1, H, W, device=device)
    weight = torch.zeros(1, 1, H, W, device=device)

    for y in range(0, max(1, H - tile + 1), stride):
        for x in range(0, max(1, W - tile + 1), stride):
            patch = image[:, :, y:y+tile, x:x+tile]
            pad_h = max(0, tile - patch.shape[-2])
            pad_w = max(0, tile - patch.shape[-1])
            if pad_h or pad_w:
                patch = F.pad(patch, (0, pad_w, 0, pad_h))
            out = model_fn(patch)
            kpatch = out[key]
            if kpatch.ndim == 3:
                kpatch = kpatch.unsqueeze(0)
            kpatch = kpatch[..., :tile, :tile]
            acc[..., y:y+tile, x:x+tile] += kpatch * win
            weight[..., y:y+tile, x:x+tile] += win

    out_full = acc / (weight + 1e-8)
    return {key: out_full}
