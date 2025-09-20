"""
E(n) Equivariant Topological Neural Network layers for hierarchical forecasting.

This module implements ETNN-inspired layers that can work with combinatorial complexes
while maintaining E(n) equivariance for geometric features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from .combinatorial_complex import CombinatorialComplex


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
    """
    E(n) Equivariant Topological Neural Network layer.
    
    Implements the ETNN update equations (6) and (7) from the paper:
    - Feature update with geometric invariants
    - Position update maintaining E(n) equivariance
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 neighborhood_names: List[str],
                 use_position_update: bool = True,
                 invariant_type: str = "centroid_distance"):
        """
        Initialize ETNN layer.
        
        Args:
            hidden_dim: Hidden dimension size
            neighborhood_names: Names of neighborhood functions to use
            use_position_update: Whether to update positions (equivariant) or not (invariant)
            invariant_type: Type of geometric invariant ("centroid_distance", "pairwise", "hausdorff")
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.neighborhood_names = neighborhood_names
        self.use_position_update = use_position_update
        self.invariant_type = invariant_type
        
        # Message functions for each neighborhood type
        self.message_functions = nn.ModuleDict()
        for name in neighborhood_names:
            self.message_functions[name] = nn.Sequential(
                nn.Linear(2 * hidden_dim + 1, hidden_dim),  # +1 for geometric invariant
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Update function
        self.update_function = nn.Sequential(
            nn.Linear(hidden_dim + len(neighborhood_names) * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Position update function (if equivariant)
        if use_position_update:
            self.position_function = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            )
    
    def compute_geometric_invariant(self, pos_x: torch.Tensor, pos_y: torch.Tensor) -> torch.Tensor:
        """Compute geometric invariant between two cells."""
        if self.invariant_type == "centroid_distance":
            return GeometricInvariants.centroid_distance(pos_x, pos_y)
        elif self.invariant_type == "pairwise":
            return GeometricInvariants.pairwise_distances(pos_x, pos_y)
        elif self.invariant_type == "hausdorff":
            return GeometricInvariants.hausdorff_distance(pos_x, pos_y)
        else:
            raise ValueError(f"Unknown invariant type: {self.invariant_type}")
    
    def forward(self, 
                features: torch.Tensor,  # [num_cells, hidden_dim]
                positions: torch.Tensor,  # [num_nodes, spatial_dim]
                neighborhoods: Dict[str, torch.Tensor],
                cell_to_nodes: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of ETNN layer.
        
        Args:
            features: Cell features [num_cells, hidden_dim]
            positions: Node positions [num_nodes, spatial_dim]
            neighborhoods: Dictionary of neighborhood adjacency matrices
            cell_to_nodes: Mapping from cell index to list of node indices
            
        Returns:
            Tuple of (updated_features, updated_positions)
        """
        num_cells = features.size(0)
        num_nodes = positions.size(0)
        spatial_dim = positions.size(1)
        
        # Collect messages from all neighborhoods
        all_messages = []
        
        for name in self.neighborhood_names:
            if name not in neighborhoods:
                continue
                
            adj_matrix = neighborhoods[name]  # [num_cells, num_cells]
            
            # Initialize messages for this neighborhood
            messages = torch.zeros(num_cells, self.hidden_dim, device=features.device)
            
            # Compute messages for each cell
            for i in range(num_cells):
                neighbor_indices = torch.nonzero(adj_matrix[i], as_tuple=False).flatten()
                
                if len(neighbor_indices) == 0:
                    continue
                
                cell_messages = []
                for j in neighbor_indices:
                    j = j.item()
                    
                    # Get positions for cells i and j
                    pos_i = positions[cell_to_nodes[i]] if cell_to_nodes[i] else torch.empty(0, spatial_dim, device=positions.device)
                    pos_j = positions[cell_to_nodes[j]] if cell_to_nodes[j] else torch.empty(0, spatial_dim, device=positions.device)
                    
                    # Compute geometric invariant
                    geom_inv = self.compute_geometric_invariant(pos_i, pos_j).unsqueeze(0)
                    
                    # Create message input: [h_i, h_j, geometric_invariant]
                    message_input = torch.cat([features[i], features[j], geom_inv])
                    
                    # Compute message
                    message = self.message_functions[name](message_input)
                    cell_messages.append(message)
                
                if cell_messages:
                    # Aggregate messages (mean)
                    messages[i] = torch.stack(cell_messages).mean(dim=0)
            
            all_messages.append(messages)
        
        # Aggregate messages from all neighborhoods
        if all_messages:
            aggregated_messages = torch.cat(all_messages, dim=1)  # [num_cells, total_message_dim]
        else:
            aggregated_messages = torch.zeros(num_cells, 0, device=features.device)
        
        # Update features
        update_input = torch.cat([features, aggregated_messages], dim=1)
        updated_features = features + self.update_function(update_input)
        
        # Update positions (if equivariant)
        updated_positions = positions
        if self.use_position_update and hasattr(self, 'position_function'):
            position_updates = torch.zeros_like(positions)
            
            for name in self.neighborhood_names:
                if name not in neighborhoods:
                    continue
                    
                adj_matrix = neighborhoods[name]
                
                # Update positions for each node
                for node_idx in range(num_nodes):
                    # Find which cell this node belongs to
                    cell_idx = None
                    for c_idx, node_list in enumerate(cell_to_nodes):
                        if node_idx in node_list:
                            cell_idx = c_idx
                            break
                    
                    if cell_idx is None:
                        continue
                    
                    # Find neighboring nodes through cell adjacencies
                    neighbor_cells = torch.nonzero(adj_matrix[cell_idx], as_tuple=False).flatten()
                    
                    for neighbor_cell in neighbor_cells:
                        neighbor_cell = neighbor_cell.item()
                        neighbor_nodes = cell_to_nodes[neighbor_cell]
                        
                        for neighbor_node in neighbor_nodes:
                            if neighbor_node != node_idx:
                                # Compute position update
                                pos_diff = positions[node_idx] - positions[neighbor_node]
                                
                                # Get message for weight computation
                                pos_i = positions[cell_to_nodes[cell_idx]] if cell_to_nodes[cell_idx] else torch.empty(0, spatial_dim, device=positions.device)
                                pos_j = positions[cell_to_nodes[neighbor_cell]] if cell_to_nodes[neighbor_cell] else torch.empty(0, spatial_dim, device=positions.device)
                                geom_inv = self.compute_geometric_invariant(pos_i, pos_j).unsqueeze(0)
                                message_input = torch.cat([features[cell_idx], features[neighbor_cell], geom_inv])
                                message = self.message_functions[name](message_input)
                                
                                # Compute scalar weight
                                weight = self.position_function(message)
                                
                                # Update position
                                position_updates[node_idx] += pos_diff * weight
            
            updated_positions = positions + position_updates
        
        return updated_features, updated_positions


class ETNNModel(nn.Module):
    """
    Complete E(n) Equivariant Topological Neural Network model.
    """
    
    def __init__(self,
                 num_features: int,
                 hidden_dim: int,
                 num_layers: int,
                 neighborhood_names: List[str],
                 output_dim: int = 1,
                 use_position_update: bool = True,
                 invariant_type: str = "centroid_distance"):
        """
        Initialize ETNN model.
        
        Args:
            num_features: Number of input features per cell
            hidden_dim: Hidden dimension size
            num_layers: Number of ETNN layers
            neighborhood_names: Names of neighborhood functions
            output_dim: Output dimension
            use_position_update: Whether to use equivariant position updates
            invariant_type: Type of geometric invariant to use
        """
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.neighborhood_names = neighborhood_names
        self.use_position_update = use_position_update
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ETNN layers
        self.etnn_layers = nn.ModuleList([
            ETNNLayer(
                hidden_dim=hidden_dim,
                neighborhood_names=neighborhood_names,
                use_position_update=use_position_update,
                invariant_type=invariant_type
            ) for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self,
                features: torch.Tensor,
                positions: torch.Tensor,
                neighborhoods: Dict[str, torch.Tensor],
                cell_to_nodes: List[List[int]]) -> torch.Tensor:
        """
        Forward pass of ETNN model.
        
        Args:
            features: Cell features [num_cells, num_features]
            positions: Node positions [num_nodes, spatial_dim]
            neighborhoods: Dictionary of neighborhood adjacency matrices
            cell_to_nodes: Mapping from cell index to list of node indices
            
        Returns:
            Cell predictions [num_cells, output_dim]
        """
        # Embed input features
        h = self.input_embedding(features)
        
        # Apply ETNN layers
        current_positions = positions
        for layer in self.etnn_layers:
            h, current_positions = layer(h, current_positions, neighborhoods, cell_to_nodes)
        
        # Generate predictions
        predictions = self.output_layer(h)
        
        return predictions
