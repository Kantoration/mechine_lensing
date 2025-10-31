from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .physics_ops import gradient2d

Tensor = torch.Tensor


class NodeEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, p_drop: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PhysicsGuidedMessageLayer(nn.Module):
    """Simple attention-based message passing with physics biases.

    edge_attr columns: [dx, dy, dist, grad_align, photo_contrast]
    """

    def __init__(self, hidden_dim: int, edge_dim: int, heads: int = 4, p_drop: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.lin_q = nn.Linear(hidden_dim, hidden_dim)
        self.lin_k = nn.Linear(hidden_dim, hidden_dim)
        self.lin_v = nn.Linear(hidden_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, hidden_dim)
        # Learnable bias scales
        self.w_dist = nn.Parameter(torch.tensor([-1.0]))  # negative init favors nearby
        self.w_galign = nn.Parameter(torch.tensor([0.2]))  # small positive init
        # Learnable attention temperature (softmax scale), clamped in [0.5, 5]
        self._tau = nn.Parameter(torch.tensor([1.0]))
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return x
        src, tgt = edge_index
        x_j = x[src]
        x_i = x[tgt]
        q = self.lin_q(x_i).view(-1, self.heads, self.head_dim)
        k = self.lin_k(x_j).view(-1, self.heads, self.head_dim)
        v = self.lin_v(x_j).view(-1, self.heads, self.head_dim)
        tau = self._tau.clamp(0.5, 5.0)
        attn = (q * k).sum(-1) / (self.head_dim ** 0.5 * tau)
        # Physics biases
        dist = edge_attr[:, 2].unsqueeze(-1)
        galign = edge_attr[:, 3].unsqueeze(-1)
        attn = attn + self.w_dist * dist + self.w_galign * galign
        attn = torch.softmax(attn, dim=0)
        attn = self.drop(attn)
        msg = (attn.unsqueeze(-1) * v).reshape(-1, self.hidden_dim)
        out = torch.zeros_like(x)
        out.index_add_(0, tgt, msg)
        return self.lin_out(out)


class HeteroscedasticHead(nn.Module):
    def __init__(self, hidden_dim: int, out_ch: int, logvar_min: float = -7.0, logvar_max: float = 7.0) -> None:
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(hidden_dim // 2, out_ch)
        )
        self.lv = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(hidden_dim // 2, out_ch)
        )
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        mu = self.mu(h)
        logvar = self.lv(h).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar


class LensGNN(nn.Module):
    """Physics-informed GNN producing ψ, κ, and α maps from graph inputs.

    Expects a batch dict from graph_builder with keys: x, edge_index, edge_attr, meta.
    """

    def __init__(self, node_dim: int, hidden_dim: int = 128, mp_layers: int = 4, heads: int = 4, uncertainty: str = "heteroscedastic") -> None:
        super().__init__()
        self.encoder = NodeEncoder(node_dim, hidden_dim)
        self.layers = nn.ModuleList([PhysicsGuidedMessageLayer(hidden_dim, edge_dim=5, heads=heads) for _ in range(mp_layers)])
        self.head_kappa = HeteroscedasticHead(hidden_dim, 1) if uncertainty == "heteroscedastic" else nn.Linear(hidden_dim, 1)
        self.use_het = uncertainty == "heteroscedastic"
        self.head_psi = nn.Linear(hidden_dim, 1)
        self.head_alpha = HeteroscedasticHead(hidden_dim, 2) if self.use_het else nn.Linear(hidden_dim, 2)

    def _reshape_to_maps(self, node_out: Tensor, b: int, gh: int, gw: int, ch: int) -> Tensor:
        return node_out.view(b, gh, gw, ch).permute(0, 3, 1, 2).contiguous()

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.encoder(batch["x"])  # [B*N,H]
        for mp in self.layers:
            x = x + mp(x, batch["edge_index"], batch["edge_attr"])  # residual
        b = batch["meta"]["B"]
        gh = batch["meta"]["H"]
        gw = batch["meta"]["W"]
        out: Dict[str, Tensor] = {}

        # κ
        if self.use_het:
            k_mu, k_lv = self.head_kappa(x)
            out["kappa"] = self._reshape_to_maps(k_mu, b, gh, gw, 1)
            out["kappa_var"] = self._reshape_to_maps(torch.exp(k_lv) + 1e-4, b, gh, gw, 1)
        else:
            kappa = self.head_kappa(x)
            out["kappa"] = self._reshape_to_maps(kappa, b, gh, gw, 1)

        # ψ
        psi = self.head_psi(x)
        out["psi"] = self._reshape_to_maps(psi, b, gh, gw, 1)

        # α from ψ
        ps = batch["meta"].get("physics_scale")
        if ps is not None:
            # Use explicit dx/dy from physics scale (preferred)
            dx = ps.pixel_scale_rad  # Assume isotropic from PhysicsScale
            dy = ps.pixel_scale_rad
            gx, gy = gradient2d(out["psi"], dx=dx, dy=dy)
        else:
            # Fallback: use pixel_scale_rad for backward compatibility
            gx, gy = gradient2d(out["psi"], pixel_scale_rad=1.0)
        out["alpha_from_psi"] = torch.cat([gx, gy], dim=1)

        # Direct α
        if self.use_het:
            a_mu, a_lv = self.head_alpha(x)
            out["alpha_direct"] = self._reshape_to_maps(a_mu, b, gh, gw, 2)
            out["alpha_var"] = self._reshape_to_maps(torch.exp(a_lv) + 1e-4, b, gh, gw, 2)
        else:
            alpha = self.head_alpha(x)
            out["alpha_direct"] = self._reshape_to_maps(alpha, b, gh, gw, 2)

        return out


