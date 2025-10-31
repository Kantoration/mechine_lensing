from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F


def _hanning_window(tile: int, device: torch.device) -> torch.Tensor:
    w1 = torch.hann_window(tile, periodic=False, dtype=torch.float32, device=device)
    w2 = w1.view(-1, 1) * w1.view(1, -1)
    return w2  # [tile, tile]


def predict_full(
    model_fn: Callable[[torch.Tensor], Dict[str, torch.Tensor]], image: torch.Tensor
) -> Dict[str, torch.Tensor]:
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

    stride = max(1, tile - overlap)

    def _positions(length: int) -> list[int]:
        if length <= tile:
            return [0]
        pos = list(range(0, max(1, length - tile + 1), stride))
        final = max(0, length - tile)
        if pos[-1] != final:
            pos.append(final)
        return sorted(set(pos))

    ys = _positions(H)
    xs = _positions(W)

    base_win = _hanning_window(tile, device=device)
    ones_win = torch.ones_like(base_win)

    acc = torch.zeros(1, 1, H, W, device=device)
    weight = torch.zeros(1, 1, H, W, device=device)

    for y in ys:
        for x in xs:
            patch = image[:, :, y : y + tile, x : x + tile]
            patch_h = patch.shape[-2]
            patch_w = patch.shape[-1]
            pad_h = max(0, tile - patch_h)
            pad_w = max(0, tile - patch_w)
            if pad_h or pad_w:
                patch = F.pad(patch, (0, pad_w, 0, pad_h))
            out = model_fn(patch)
            kpatch = out[key]
            if kpatch.ndim == 3:
                kpatch = kpatch.unsqueeze(0)
            kpatch = kpatch[..., :tile, :tile]

            is_top = y == ys[0]
            is_bottom = y == ys[-1]
            is_left = x == xs[0]
            is_right = x == xs[-1]

            win_y = (
                base_win if (len(ys) > 1 and not (is_top or is_bottom)) else ones_win
            )
            win_x = (
                base_win if (len(xs) > 1 and not (is_left or is_right)) else ones_win
            )
            win2d = win_y.view(-1, 1) * win_x.view(1, -1)

            # Restrict to the region that maps back into the original image
            win2d = win2d[:patch_h, :patch_w].to(device)
            win2d = win2d.unsqueeze(0).unsqueeze(0)

            acc[..., y : y + patch_h, x : x + patch_w] += (
                kpatch[..., :patch_h, :patch_w] * win2d
            )
            weight[..., y : y + patch_h, x : x + patch_w] += win2d

    out_full = acc / (weight + 1e-8)
    return {key: out_full}
