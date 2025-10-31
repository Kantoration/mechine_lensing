from __future__ import annotations

from typing import Optional, Tuple

import torch

from .physics_ops import PhysicsScale

Tensor = torch.Tensor


def _positional_encoding(h: int, w: int, device: torch.device, n_freq: int = 2) -> Tensor:
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device), indexing="ij")
    pos_xy = torch.stack([xx, yy], dim=-1)  # [H,W,2]
    r = torch.linalg.vector_norm(pos_xy - 0.5, dim=-1, keepdim=True)
    theta = torch.atan2(pos_xy[..., 1] - 0.5, pos_xy[..., 0] - 0.5).unsqueeze(-1)
    fourier = []
    for f in range(1, n_freq + 1):
        fourier.append(torch.sin(2 * torch.pi * f * pos_xy))
        fourier.append(torch.cos(2 * torch.pi * f * pos_xy))
    fourier = torch.cat(fourier, dim=-1) if fourier else torch.empty(h, w, 0, device=device)
    feats = torch.cat([pos_xy, r, theta, fourier], dim=-1)  # [H,W,4+4*n_freq]
    return feats.view(h * w, -1)


def build_grid_graph(
    images: Tensor,  # [B,C,H,W]
    patch_size: int = 2,
    connectivity: str = "8+ring",
    backbone_feats: Optional[Tensor] = None,  # [B,F,H',W']
    physics_scale: Optional[PhysicsScale] = None,
    n_pos_freq: int = 2,
    max_edges_per_node: int = 24,
) -> dict:
    """Build a simple grid-graph batch (pure-PyTorch struct) used by LensGNN.

    Returns a dict with keys: x, edge_index, edge_attr, batch, meta.
    """
    b, c, h, w = images.shape
    device = images.device
    if patch_size > 1:
        pooled = torch.nn.functional.avg_pool2d(images, kernel_size=patch_size)
        gh, gw = h // patch_size, w // patch_size
    else:
        pooled = images
        gh, gw = h, w

    n = gh * gw
    # Node features: per-patch mean/std plus channels
    img_flat = pooled.permute(0, 2, 3, 1).reshape(b, n, c)
    mean = img_flat.mean(dim=-1, keepdim=True)
    std = img_flat.std(dim=-1, keepdim=True)
    node_feats = torch.cat([mean, std, img_flat], dim=-1)  # [B,N,2+C]

    # Positional encodings
    pos = _positional_encoding(gh, gw, device=device, n_freq=n_pos_freq)  # [N,P]
    pos = pos.unsqueeze(0).expand(b, n, pos.shape[-1])
    node_feats = torch.cat([node_feats, pos], dim=-1)

    # Backbone fusion (upsample then concat)
    if backbone_feats is not None:
        bb_up = torch.nn.functional.interpolate(backbone_feats, size=(gh, gw), mode="bilinear", align_corners=False)
        bb_flat = bb_up.permute(0, 2, 3, 1).reshape(b, n, bb_up.shape[1])
        node_feats = torch.cat([node_feats, bb_flat], dim=-1)

    x = node_feats.reshape(b * n, -1)

    # Edges (8-connected, optional ring)
    edges = []
    attrs = []
    for bi in range(b):
        base = bi * n
        for i in range(gh):
            for j in range(gw):
                src = base + i * gw + j
                nbrs: list[Tuple[int,int]] = []
                # 4-neigh
                if j + 1 < gw:
                    nbrs.append((i, j + 1))
                if i + 1 < gh:
                    nbrs.append((i + 1, j))
                # diagonals
                if connectivity in ("8", "8+ring"):
                    if i + 1 < gh and j + 1 < gw:
                        nbrs.append((i + 1, j + 1))
                    if i + 1 < gh and j - 1 >= 0:
                        nbrs.append((i + 1, j - 1))
                # ring (2-hop) only if patch_size >= 2
                if connectivity == "8+ring" and patch_size >= 2:
                    if j + 2 < gw:
                        nbrs.append((i, j + 2))
                    if i + 2 < gh:
                        nbrs.append((i + 2, j))
                # prune farthest if above density cap
                if len(nbrs) > max_edges_per_node:
                    # sort by distance and keep closest
                    nbrs.sort(key=lambda ij: ((ij[1]-j)**2 + (ij[0]-i)**2))
                    nbrs = nbrs[:max_edges_per_node]
                for (ni, nj) in nbrs:
                    tgt = base + ni * gw + nj
                    edges.append([src, tgt])
                    # edge_attr: dx, dy, dist, grad_align(placeholder=0), photo_contrast(placeholder=0)
                    dx = float(nj - j) * patch_size
                    dy = float(ni - i) * patch_size
                    dist = (dx * dx + dy * dy) ** 0.5
                    attrs.append([dx, dy, dist, 0.0, 0.0])

    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous() if edges else torch.empty(2, 0, dtype=torch.long, device=device)
    edge_attr = torch.tensor(attrs, dtype=torch.float32, device=device) if attrs else torch.empty(0, 5, dtype=torch.float32, device=device)
    batch_vec = torch.arange(b, device=device).repeat_interleave(n)

    meta = {
        "H": gh,
        "W": gw,
        "patch_size": patch_size,
        "B": b,
        "physics_scale": physics_scale or PhysicsScale(pixel_scale_arcsec=0.1),
    }

    # density stats
    edges_per_node = edge_index.shape[1] / max(b * n, 1)
    if edges_per_node > 40:
        raise RuntimeError(f"Graph too dense: edges/node={edges_per_node:.1f} (>40)")

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch_vec,
        "meta": {**meta, "edges_per_node": float(edges_per_node)},
    }


