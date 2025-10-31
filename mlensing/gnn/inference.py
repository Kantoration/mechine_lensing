from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _hann_2d(size: int, device: torch.device) -> torch.Tensor:
    w = torch.hann_window(size, periodic=False, dtype=torch.float32, device=device)
    win2 = w.unsqueeze(1) @ w.unsqueeze(0)
    return win2 / (win2.sum() + 1e-8)


@torch.no_grad()
def tiled_inference(
    model, image: torch.Tensor, tile: int = 128, overlap: int = 32
) -> Dict[str, torch.Tensor]:
    b, c, h, w = image.shape
    device = image.device
    weight = torch.zeros(b, 1, h, w, device=device)
    acc: Dict[str, torch.Tensor] = {}
    step = tile - overlap
    win = _hann_2d(tile, device)

    for y in range(0, max(1, h - overlap), step):
        for x in range(0, max(1, w - overlap), step):
            y2, x2 = min(y + tile, h), min(x + tile, w)
            patch = image[:, :, y:y2, x:x2]
            if patch.shape[-1] < tile or patch.shape[-2] < tile:
                patch = F.pad(
                    patch, (0, tile - patch.shape[-1], 0, tile - patch.shape[-2])
                )
            # Build trivial graph per-tile externally in caller; here assume model can take raw images if adapted
            pred = model({"image": patch})  # placeholder adapter
            for k, v in pred.items():
                if isinstance(v, torch.Tensor):
                    if k not in acc:
                        acc[k] = torch.zeros(b, v.shape[1], h, w, device=device)
                    acc[k][:, :, y:y2, x:x2] += (
                        v[:, :, : (y2 - y), : (x2 - x)] * win[: (y2 - y), : (x2 - x)]
                    )
            weight[:, :, y:y2, x:x2] += win[: (y2 - y), : (x2 - x)]

    for k in acc:
        acc[k] = acc[k] / (weight + 1e-8)
    return acc
