"""
E(n) Equivariant Topological Neural Network layers for hierarchical forecasting.

This module implements ETNN-inspired layers that can work with combinatorial complexes
while maintaining E(n) equivariance for geometric features.
"""

import torch
import torch.nn as nn
from typing import Dict, Iterable, Tuple


class GeometricInvariants:
    """
    Geometric invariant functions for ETNN layers.
    Following the paper's geometric invariants for combinatorial complexes.
    """
    
    @staticmethod
    def pairwise_distances(pos_x: torch.Tensor, pos_y: torch.Tensor) -> torch.Tensor:
        """
        Compute sum of pairwise distances between two sets of positions.
        
        Args:
            pos_x: Positions of nodes in cell x [num_nodes_x, dim]
            pos_y: Positions of nodes in cell y [num_nodes_y, dim]
            
        Returns:
            Sum of pairwise distances (scalar)
        """
        if pos_x.size(0) == 0 or pos_y.size(0) == 0:
            return torch.tensor(0.0, device=pos_x.device)
        
        # Compute all pairwise distances
        diff = pos_x.unsqueeze(1) - pos_y.unsqueeze(0)  # [num_x, num_y, dim]
        distances = torch.norm(diff, dim=-1)  # [num_x, num_y]
        return distances.sum()
    
    @staticmethod
    def centroid_distance(pos_x: torch.Tensor, pos_y: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between centroids of two cells.
        
        Args:
            pos_x: Positions of nodes in cell x [num_nodes_x, dim]
            pos_y: Positions of nodes in cell y [num_nodes_y, dim]
            
        Returns:
            Distance between centroids (scalar)
        """
        if pos_x.size(0) == 0 or pos_y.size(0) == 0:
            return torch.tensor(0.0, device=pos_x.device)
        
        centroid_x = pos_x.mean(dim=0)  # [dim]
        centroid_y = pos_y.mean(dim=0)  # [dim]
        return torch.norm(centroid_x - centroid_y)
    
    @staticmethod
    def hausdorff_distance(pos_x: torch.Tensor, pos_y: torch.Tensor) -> torch.Tensor:
        """
        Compute Hausdorff distance between two sets of points.
        
        Args:
            pos_x: Positions of nodes in cell x [num_nodes_x, dim]
            pos_y: Positions of nodes in cell y [num_nodes_y, dim]
            
        Returns:
            Hausdorff distance (scalar)
        """
        if pos_x.size(0) == 0 or pos_y.size(0) == 0:
            return torch.tensor(0.0, device=pos_x.device)
        
        # Directed Hausdorff distances
        diff_xy = pos_x.unsqueeze(1) - pos_y.unsqueeze(0)  # [num_x, num_y, dim]
        dist_xy = torch.norm(diff_xy, dim=-1)  # [num_x, num_y]
        h_xy = dist_xy.min(dim=1)[0].max()  # max over x of min over y
        
        diff_yx = pos_y.unsqueeze(1) - pos_x.unsqueeze(0)  # [num_y, num_x, dim]
        dist_yx = torch.norm(diff_yx, dim=-1)  # [num_y, num_x]
        h_yx = dist_yx.min(dim=1)[0].max()  # max over y of min over x
        
        return torch.max(h_xy, h_yx)


class ETNNLayer(nn.Module):
    """Simplified ETNN layer operating on precomputed edge indices."""

    def __init__(
        self,
        hidden_dim: int,
        neighborhood_names: Iterable[str],
        use_position_update: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.neighborhood_names = list(neighborhood_names)
        self.use_position_update = use_position_update

        self.message_functions = nn.ModuleDict()
        self.position_functions = nn.ModuleDict()

        for name in self.neighborhood_names:
            self.message_functions[name] = nn.Sequential(
                nn.Linear(2 * hidden_dim + 1, hidden_dim),
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

            msg_input = torch.cat([h_src, h_dst, distances], dim=-1)
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
