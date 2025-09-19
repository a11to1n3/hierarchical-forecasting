"""
Combinatorial Complex Message Passing Neural Network (CCMPN) layers.

This module implements specialized message-passing layers that operate on
combinatorial complex structures for hierarchical learning.
"""

import torch
import torch.nn as nn
from typing import Dict


class EnhancedCCMPNLayer(nn.Module):
    """
    A message-passing layer that handles multiple neighborhood types in a
    combinatorial complex structure.
    
    The layer performs message passing between related entities in the hierarchy,
    using specialized message functions for different relationship types.
    """
    
    def __init__(self, hidden_dim: int, neighborhood_names: list):
        """
        Initialize the CCMPN layer.
        
        Args:
            hidden_dim: Dimension of hidden representations
            neighborhood_names: List of neighborhood types to handle
        """
        super().__init__()
        
        # Specialized message functions for each neighborhood type
        self.psi_functions = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(2 * hidden_dim + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for name in neighborhood_names
        })
        
        # GRU cell for state update
        self.beta = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, features: torch.Tensor, 
                neighborhoods: Dict[str, torch.sparse.FloatTensor]) -> torch.Tensor:
        """
        Forward pass of the CCMPN layer.
        
        Args:
            h: Hidden representations [num_cells, hidden_dim]
            features: Input features [num_cells, num_features]  
            neighborhoods: Dictionary of sparse adjacency matrices
            
        Returns:
            Updated hidden representations [num_cells, hidden_dim]
        """
        total_aggregated_messages = torch.zeros_like(h)
        
        for name, matrix in neighborhoods.items():
            matrix = matrix.coalesce()  # Ensure matrix is coalesced
            if matrix._nnz() == 0:  # Skip empty matrices
                continue
                
            source_nodes, target_nodes = matrix.indices()
            
            # Move indices to the same device as features
            source_nodes = source_nodes.to(h.device)
            target_nodes = target_nodes.to(h.device)
            
            # Create relational invariant (simple difference in features)
            relational_feature = (
                features[source_nodes, 0] - features[target_nodes, 0]
            ).unsqueeze(1)
            
            # Concatenate source features, target features, and relational invariant
            messages_in = torch.cat([
                h[source_nodes], 
                h[target_nodes], 
                relational_feature
            ], dim=1)
            
            # Apply message function
            messages = self.psi_functions[name](messages_in)
            
            # Aggregate messages at target nodes
            aggregated = torch.zeros_like(h)
            aggregated.index_add_(0, target_nodes, messages)
            
            total_aggregated_messages += aggregated
        
        # Update hidden states using GRU
        h_new = self.beta(total_aggregated_messages, h)
        return h_new


class CCMPNMessageFunction(nn.Module):
    """
    Individual message function for a specific neighborhood type.
    """
    
    def __init__(self, hidden_dim: int, message_dim: int = None):
        super().__init__()
        if message_dim is None:
            message_dim = hidden_dim
            
        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, hidden_dim),
            nn.Tanh()
        )
    
    def forward(self, h_source: torch.Tensor, h_target: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute messages between source and target nodes.
        
        Args:
            h_source: Source node hidden states
            h_target: Target node hidden states  
            edge_attr: Edge attributes/features
            
        Returns:
            Messages from source to target nodes
        """
        message_input = torch.cat([h_source, h_target, edge_attr], dim=-1)
        return self.message_net(message_input)


class HierarchicalAggregator(nn.Module):
    """
    Hierarchical aggregation module that respects the complex structure.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, h: torch.Tensor, hierarchy_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform hierarchical aggregation with attention.
        
        Args:
            h: Hidden representations
            hierarchy_mask: Mask indicating hierarchical relationships
            
        Returns:
            Aggregated representations
        """
        # Apply attention mechanism
        attn_out, _ = self.attention(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
        attn_out = attn_out.squeeze(0)
        
        # Apply residual connection and normalization
        h_updated = self.norm(h + attn_out)
        
        return h_updated
