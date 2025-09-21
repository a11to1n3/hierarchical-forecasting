"""
E(n) Equivariant Topological Neural Network layers for hierarchical forecasting.

This module implements ETNN-inspired layers that can work with combinatorial complexes
while maintaining E(n) equivariance for geometric features.
"""

import torch
import torch.nn as nn
from typing import Dict, Iterable, List, Optional, Tuple


class GeometricInvariants:
    """Geometric invariant functions for ETNN layers."""

    _EPS = 1e-12

    @staticmethod
    def _ensure_batch(pos: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
        return pos, mask

    @staticmethod
    def pairwise_distances(
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
        mask_x: Optional[torch.Tensor] = None,
        mask_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pos_x, mask_x = GeometricInvariants._ensure_batch(pos_x, mask_x)
        pos_y, mask_y = GeometricInvariants._ensure_batch(pos_y, mask_y)

        distances = torch.cdist(pos_x, pos_y, p=2)

        if mask_x is not None and mask_y is not None:
            mask_x = mask_x.unsqueeze(-1)
            mask_y = mask_y.unsqueeze(-2)
            weight = mask_x * mask_y
            return (distances * weight).sum(dim=(1, 2))

        return distances.sum(dim=(1, 2))

    @staticmethod
    def centroid_distance(
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
        mask_x: Optional[torch.Tensor] = None,
        mask_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pos_x, mask_x = GeometricInvariants._ensure_batch(pos_x, mask_x)
        pos_y, mask_y = GeometricInvariants._ensure_batch(pos_y, mask_y)

        if mask_x is not None:
            denom_x = mask_x.sum(dim=1, keepdim=True).clamp_min(GeometricInvariants._EPS)
            centroid_x = (pos_x * mask_x.unsqueeze(-1)).sum(dim=1) / denom_x
        else:
            centroid_x = pos_x.mean(dim=1)

        if mask_y is not None:
            denom_y = mask_y.sum(dim=1, keepdim=True).clamp_min(GeometricInvariants._EPS)
            centroid_y = (pos_y * mask_y.unsqueeze(-1)).sum(dim=1) / denom_y
        else:
            centroid_y = pos_y.mean(dim=1)

        return torch.norm(centroid_x - centroid_y, dim=-1)

    @staticmethod
    def hausdorff_distance(
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
        mask_x: Optional[torch.Tensor] = None,
        mask_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pos_x, mask_x = GeometricInvariants._ensure_batch(pos_x, mask_x)
        pos_y, mask_y = GeometricInvariants._ensure_batch(pos_y, mask_y)

        distances = torch.cdist(pos_x, pos_y, p=2)

        if mask_y is not None:
            invalid = (mask_y <= 0).unsqueeze(1)
            distances = distances.masked_fill(invalid, float('inf'))

        min_xy, _ = distances.min(dim=2)
        if mask_x is not None:
            min_xy = min_xy.masked_fill(mask_x <= 0, 0.0)
        h_xy, _ = min_xy.max(dim=1)

        distances_reverse = distances.transpose(1, 2)
        if mask_x is not None:
            invalid = (mask_x <= 0).unsqueeze(1)
            distances_reverse = distances_reverse.masked_fill(invalid, float('inf'))

        min_yx, _ = distances_reverse.min(dim=2)
        if mask_y is not None:
            min_yx = min_yx.masked_fill(mask_y <= 0, 0.0)
        h_yx, _ = min_yx.max(dim=1)

        return torch.maximum(h_xy, h_yx)


class ETNNLayer(nn.Module):
    """Simplified ETNN layer operating on precomputed edge indices."""

    def __init__(
        self,
        hidden_dim: int,
        neighborhood_names: Iterable[str],
        cell_to_nodes: Optional[List[List[int]]] = None,
        use_position_update: bool = True,
        use_geometric_features: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.neighborhood_names = list(neighborhood_names)
        self.use_position_update = use_position_update
        self.use_geometric_features = use_geometric_features
        self.num_invariant_terms = 4 if use_geometric_features else 1

        self.message_functions = nn.ModuleDict()
        self.position_functions = nn.ModuleDict()
        self.max_nodes: int = 0
        if self.use_geometric_features and cell_to_nodes is not None and len(cell_to_nodes) > 0:
            self.max_nodes = max((len(nodes) for nodes in cell_to_nodes), default=0)
        if self.use_geometric_features and self.max_nodes <= 0:
            self.max_nodes = 1
            cell_to_nodes = [[idx] for idx in range(len(cell_to_nodes or []))]

        if self.use_geometric_features and cell_to_nodes is not None and len(cell_to_nodes) > 0:
            index_rows: List[List[int]] = []
            mask_rows: List[List[float]] = []
            for idx, nodes in enumerate(cell_to_nodes):
                if not nodes:
                    nodes = [idx]
                padded = list(nodes)
                pad_value = nodes[0]
                while len(padded) < self.max_nodes:
                    padded.append(pad_value)
                mask = [1.0] * len(nodes) + [0.0] * (self.max_nodes - len(nodes))
                index_rows.append(padded[:self.max_nodes])
                mask_rows.append(mask[:self.max_nodes])
            self.register_buffer('cell_node_indices', torch.tensor(index_rows, dtype=torch.long))
            self.register_buffer('cell_node_mask', torch.tensor(mask_rows, dtype=torch.float32))
            self.has_node_info = True
        else:
            self.register_buffer('cell_node_indices', torch.empty(0, dtype=torch.long))
            self.register_buffer('cell_node_mask', torch.empty(0, dtype=torch.float32))
            self.has_node_info = False

        for name in self.neighborhood_names:
            self.message_functions[name] = nn.Sequential(
                nn.Linear(2 * hidden_dim + self.num_invariant_terms, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            if use_position_update:
                self.position_functions[name] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1)
                )

        self.update_function = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def _node_positions(self, positions: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.has_node_info or self.cell_node_indices.numel() == 0:
            return None, None
        device = positions.device
        indices = self.cell_node_indices.to(device)
        mask = self.cell_node_mask.to(device)
        batch_size = positions.size(0)
        batch_idx = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
        node_positions = positions[batch_idx, indices, :]
        return node_positions, mask

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        edge_indices: Dict[str, torch.Tensor],
        node_degrees: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_cells, _ = features.shape

        message_accumulator = torch.zeros_like(features)
        position_updates = torch.zeros_like(positions) if self.use_position_update else None

        node_positions = None
        node_mask = None
        if self.use_geometric_features and self.has_node_info:
            node_positions, node_mask = self._node_positions(positions)

        for name in self.neighborhood_names:
            if name not in edge_indices:
                continue

            edge_index = edge_indices[name]
            if edge_index.numel() == 0:
                continue

            src, dst = edge_index[0], edge_index[1]
            h_src = features[:, src, :]
            h_dst = features[:, dst, :]

            rel_pos = positions[:, src, :] - positions[:, dst, :]
            distances = torch.norm(rel_pos, dim=-1, keepdim=True)

            if self.use_geometric_features and node_positions is not None and node_mask is not None:
                src_node_pos = node_positions[:, src, :, :]  # [batch, edges, max_nodes, dim]
                dst_node_pos = node_positions[:, dst, :, :]
                src_node_mask = node_mask[src]  # [edges, max_nodes]
                dst_node_mask = node_mask[dst]

                batch_size, num_edges, _, _ = src_node_pos.shape
                src_flat = src_node_pos.reshape(batch_size * num_edges, self.max_nodes, -1)
                dst_flat = dst_node_pos.reshape(batch_size * num_edges, self.max_nodes, -1)
                src_mask_flat = src_node_mask.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * num_edges, self.max_nodes)
                dst_mask_flat = dst_node_mask.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * num_edges, self.max_nodes)

                pairwise = GeometricInvariants.pairwise_distances(src_flat, dst_flat, src_mask_flat, dst_mask_flat).view(batch_size, num_edges, 1)
                centroid = GeometricInvariants.centroid_distance(src_flat, dst_flat, src_mask_flat, dst_mask_flat).view(batch_size, num_edges, 1)
                hausdorff = GeometricInvariants.hausdorff_distance(src_flat, dst_flat, src_mask_flat, dst_mask_flat).view(batch_size, num_edges, 1)
            else:
                pairwise = distances
                centroid = distances
                hausdorff = distances

            if self.use_geometric_features:
                invariants = torch.cat([distances, pairwise, centroid, hausdorff], dim=-1)
            else:
                invariants = distances

            msg_input = torch.cat([h_src, h_dst, invariants], dim=-1)
            messages = self.message_functions[name](msg_input)

            aggregated = torch.zeros_like(features)
            aggregated.index_add_(1, dst, messages)

            degree = node_degrees.get(name)
            if degree is not None:
                norm = degree.to(features.device).clamp(min=1.0)
                aggregated = aggregated / norm.unsqueeze(0).unsqueeze(-1)

            message_accumulator = message_accumulator + aggregated

            if self.use_position_update and name in self.position_functions:
                weights = torch.tanh(self.position_functions[name](messages))
                update = torch.zeros_like(positions)
                update.index_add_(1, dst, weights * rel_pos)
                position_updates = position_updates + update

        update_input = torch.cat([features, message_accumulator], dim=-1)
        feature_update = self.update_function(update_input)
        updated_features = features + feature_update

        if self.use_position_update and position_updates is not None:
            updated_positions = positions + position_updates / max(1, len(self.neighborhood_names))
        else:
            updated_positions = positions

        return updated_features, updated_positions
