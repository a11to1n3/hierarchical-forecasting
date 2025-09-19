"""
Enhanced Hierarchical Forecasting Model.

This module implements the complete forecasting model that combines
combinatorial complex structures with message-passing neural networks.
"""

import torch
import torch.nn as nn
from typing import Dict, List
from .ccmpn import EnhancedCCMPNLayer


class EnhancedHierarchicalModel(nn.Module):
    """
    The complete hierarchical forecasting model stacking CCMPN layers.
    
    This model performs forecasting at multiple hierarchical levels using
    the combinatorial complex structure to guide message passing.
    """
    
    def __init__(self, num_cells: int, num_features: int, hidden_dim: int, 
                 neighborhood_names: List[str], n_layers: int = 2, 
                 dropout: float = 0.1):
        """
        Initialize the hierarchical model.
        
        Args:
            num_cells: Total number of cells in the combinatorial complex
            num_features: Number of input features per cell
            hidden_dim: Hidden dimension size
            neighborhood_names: List of neighborhood types
            n_layers: Number of CCMPN layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_cells = num_cells
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Input embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Stack of CCMPN layers
        self.layers = nn.ModuleList([
            EnhancedCCMPNLayer(hidden_dim, neighborhood_names) 
            for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        # Output projection
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, 
                neighborhoods: Dict[str, torch.sparse.FloatTensor]) -> torch.Tensor:
        """
        Forward pass of the hierarchical model.
        
        Args:
            x: Input features [num_cells, num_features]
            neighborhoods: Dictionary of neighborhood adjacency matrices
            
        Returns:
            Predictions [num_cells, 1]
        """
        # Initial embedding
        h = self.embedding(x)
        
        # Apply CCMPN layers with residual connections
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            h_new = layer(h, x, neighborhoods)
            h = norm(h + h_new)  # Residual connection + layer norm
        
        # Generate predictions
        output = self.readout(h)
        return output
    
    def get_embeddings(self, x: torch.Tensor, 
                      neighborhoods: Dict[str, torch.sparse.FloatTensor]) -> torch.Tensor:
        """
        Get intermediate embeddings without final prediction layer.
        
        Args:
            x: Input features
            neighborhoods: Neighborhood matrices
            
        Returns:
            Final embeddings before readout layer
        """
        h = self.embedding(x)
        
        for layer, norm in zip(self.layers, self.layer_norms):
            h_new = layer(h, x, neighborhoods)
            h = norm(h + h_new)
        
        return h


class HierarchicalLoss(nn.Module):
    """
    Custom loss function that enforces hierarchical consistency.
    """
    
    def __init__(self, hierarchy_weights: Dict[int, float] = None):
        """
        Initialize hierarchical loss.
        
        Args:
            hierarchy_weights: Weights for different hierarchy levels
        """
        super().__init__()
        self.hierarchy_weights = hierarchy_weights or {0: 1.0, 1: 0.5, 2: 0.3, 3: 0.2}
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                cell_ranks: torch.Tensor) -> torch.Tensor:
        """
        Compute hierarchical loss.
        
        Args:
            predictions: Model predictions
            targets: Target values
            cell_ranks: Rank of each cell in the hierarchy
            
        Returns:
            Weighted hierarchical loss
        """
        total_loss = 0.0
        
        for rank, weight in self.hierarchy_weights.items():
            rank_mask = cell_ranks == rank
            if rank_mask.sum() > 0:
                rank_loss = self.mse_loss(
                    predictions[rank_mask], 
                    targets[rank_mask]
                )
                total_loss += weight * rank_loss
        
        return total_loss


class ModelConfig:
    """Configuration class for the hierarchical model."""
    
    def __init__(self):
        self.hidden_dim = 32
        self.n_layers = 2
        self.dropout = 0.1
        self.learning_rate = 0.005
        self.weight_decay = 1e-5
        self.batch_size = 32
        self.epochs = 50
        self.early_stopping_patience = 10
        
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config
