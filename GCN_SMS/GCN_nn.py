"""
Two-layer tetrahedral GCN that outputs per-tet log-correction delta_k.
Baseline alpha comes from HU->E(H) mapping stored in NPZ; this model only predicts delta.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple

ALPHA_MIN = 500.0
ALPHA_MAX = 1.0e4
EPS = 1e-8


def scatter_neighbor_sum(values: torch.Tensor, neighbor_indices: torch.Tensor, neighbor_offsets: torch.Tensor) -> torch.Tensor:
    """
    Sum neighbor features for each node given CSR-style adjacency.
    values: (N, F) or (N,)
    neighbor_indices: (E,)
    neighbor_offsets: (N+1,)
    Returns: (N, F) or (N,) matching values.
    """
    n = neighbor_offsets.shape[0] - 1
    feature_dim = values.dim() == 2
    if feature_dim:
        out = torch.zeros_like(values)
        counts = neighbor_offsets[1:] - neighbor_offsets[:-1]
        row_ids = torch.arange(n, device=values.device, dtype=torch.int64)
        row_ids = torch.repeat_interleave(row_ids, counts)
        out.scatter_add_(0, row_ids.unsqueeze(1).expand(-1, values.shape[1]), values[neighbor_indices])
    else:
        out = torch.zeros_like(values)
        counts = neighbor_offsets[1:] - neighbor_offsets[:-1]
        row_ids = torch.arange(n, device=values.device, dtype=torch.int64)
        row_ids = torch.repeat_interleave(row_ids, counts)
        out.scatter_add_(0, row_ids, values[neighbor_indices])
    return out


class TetGCN(nn.Module):
    def __init__(self, hidden_dim: int, max_delta_log: float = 0.3):
        """
        GCN that predicts per-tet log-correction delta_k around a fixed baseline alpha0.
        max_delta_log bounds |delta_k| so exp(delta_k) stays close to 1.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_delta_log = max_delta_log

        self.W_nei1 = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)
        self.W_self1 = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))

        self.W_nei2 = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        self.W_self2 = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(1))
        self.act = nn.ReLU()

    def forward(
        self,
        hu_scalar: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-tet log correction delta (same shape as hu)."""
        hu_scalar = hu_scalar.to(dtype=torch.float32)
        mu = hu_scalar.mean()
        sigma = hu_scalar.std() + EPS
        h0 = (hu_scalar - mu) / sigma  # (N,)

        nei_sum0 = scatter_neighbor_sum(h0, neighbor_indices, neighbor_offsets)  # (N,)
        h1 = self.act(
            self.b1
            + nei_sum0.unsqueeze(1) * self.W_nei1
            + h0.unsqueeze(1) * self.W_self1
        )  # (N, F1)

        nei_sum1 = scatter_neighbor_sum(h1, neighbor_indices, neighbor_offsets)  # (N, F1)
        delta_raw = (
            self.b2
            + (nei_sum1 * self.W_nei2).sum(dim=1, keepdim=False)
            + (h1 * self.W_self2).sum(dim=1, keepdim=False)
        )  # (N,)

        delta_log = self.max_delta_log * torch.tanh(delta_raw)
        return delta_log


def build_neighbor_tensors(
    neighbor_indices: torch.Tensor, neighbor_offsets: torch.Tensor, device: torch.device | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ensure adjacency tensors are contiguous and on device."""
    idx = neighbor_indices.to(device=device, dtype=torch.int64, non_blocking=True)
    off = neighbor_offsets.to(device=device, dtype=torch.int64, non_blocking=True)
    return idx.contiguous(), off.contiguous()
