"""
Enhanced CCMPN model with ETNN integration for hierarchical forecasting.

This module combines the original CCMPN approach with ETNN (E(n) Equivariant 
Topological Neural Networks) concepts for improved hierarchical modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from .combinatorial_complex import CombinatorialComplex
from .ccmpn import EnhancedCCMPNLayer
from .etnn_layers import ETNNLayer, GeometricInvariants
from .hierarchical_model import ModelConfig


class ETNNCCMPNLayer(nn.Module):
    """
    Hybrid layer combining CCMPN message passing with ETNN geometric invariants.
    
    This layer integrates the hierarchical structure modeling capabilities of CCMPN
    with the geometric equivariance properties of ETNN.
    """
    
    def __init__(self,
                 hidden_dim: int,
                 ccmpn_neighborhoods: List[str],
                 etnn_neighborhoods: List[str],
                 use_geometric_invariants: bool = True,
                 fusion_method: str = 'attention',
                 dropout: float = 0.1):
        """
        Initialize hybrid ETNN-CCMPN layer.
        
        Args:
            hidden_dim: Hidden dimension for both CCMPN and ETNN components
            ccmpn_neighborhoods: Neighborhoods for CCMPN message passing
            etnn_neighborhoods: Neighborhoods for ETNN message passing
            use_geometric_invariants: Whether to use geometric invariants in ETNN
            fusion_method: Method to fuse CCMPN and ETNN outputs ('concat', 'add', 'attention')
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method
        
        # CCMPN component
        self.ccmpn_layer = EnhancedCCMPNLayer(
            hidden_dim=hidden_dim,
            neighborhood_names=ccmpn_neighborhoods,
            use_attention=True,
            dropout=dropout
        )
        
        # ETNN component
        self.etnn_layer = ETNNLayer(
            hidden_dim=hidden_dim,
            neighborhood_names=etnn_neighborhoods,
            use_position_update=True,
            use_geometric_invariants=use_geometric_invariants
        )
        
        # Fusion mechanism
        if fusion_method == 'concat':
            self.fusion_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif fusion_method == 'attention':
            self.attention_weights = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
                nn.Softmax(dim=-1)
            )
        # For 'add' method, no additional parameters needed
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self,
                cell_features: Dict[int, torch.Tensor],
                cell_positions: Dict[int, torch.Tensor],
                combinatorial_complex: CombinatorialComplex) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Forward pass through hybrid ETNN-CCMPN layer.
        
        Args:
            cell_features: Features for each cell type {rank: features}
            cell_positions: Positions for each cell type {rank: positions}
            combinatorial_complex: Combinatorial complex structure
            
        Returns:
            Tuple of (updated_features, updated_positions)
        """
        # Store original features for residual connection
        original_features = {rank: feat.clone() for rank, feat in cell_features.items()}
        
        # Apply CCMPN layer
        ccmpn_features, _ = self.ccmpn_layer(cell_features, combinatorial_complex)
        
        # Apply ETNN layer
        etnn_features, updated_positions = self.etnn_layer(cell_features, cell_positions, combinatorial_complex)
        
        # Fuse CCMPN and ETNN outputs
        fused_features = {}
        for rank in cell_features.keys():
            if rank in ccmpn_features and rank in etnn_features:
                ccmpn_feat = ccmpn_features[rank]
                etnn_feat = etnn_features[rank]
                
                if self.fusion_method == 'concat':
                    # Concatenate and project
                    combined = torch.cat([ccmpn_feat, etnn_feat], dim=-1)
                    fused = self.fusion_mlp(combined)
                elif self.fusion_method == 'add':
                    # Simple addition
                    fused = ccmpn_feat + etnn_feat
                elif self.fusion_method == 'attention':
                    # Attention-based fusion
                    combined = torch.cat([ccmpn_feat, etnn_feat], dim=-1)
                    attention_weights = self.attention_weights(combined)  # [batch, 2]
                    
                    # Apply attention weights
                    fused = (attention_weights[:, 0:1] * ccmpn_feat + 
                            attention_weights[:, 1:2] * etnn_feat)
                else:
                    raise ValueError(f"Unknown fusion method: {self.fusion_method}")
                
                # Residual connection and layer normalization
                fused_features[rank] = self.layer_norm(fused + original_features[rank])
            elif rank in ccmpn_features:
                fused_features[rank] = self.layer_norm(ccmpn_features[rank] + original_features[rank])
            elif rank in etnn_features:
                fused_features[rank] = self.layer_norm(etnn_features[rank] + original_features[rank])
            else:
                fused_features[rank] = original_features[rank]
        
        return fused_features, updated_positions


class ETNNEnhancedHierarchicalModel(nn.Module):
    """
    Enhanced hierarchical forecasting model combining CCMPN and ETNN approaches.
    
    This model integrates:
    1. CCMPN for hierarchical structure modeling
    2. ETNN for geometric equivariance and topological features
    3. Attention mechanisms for improved representation learning
    4. Hierarchical consistency constraints
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize ETNN-enhanced hierarchical model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.output_dim = config.output_dim
        
        # Input projection
        self.input_projection = nn.Linear(config.input_dim, self.hidden_dim)
        
        # Position initialization for geometric features
        self.position_embedding = nn.Parameter(
            torch.randn(config.max_entities, config.spatial_dim) * 0.1
        )
        
        # Hybrid ETNN-CCMPN layers
        self.hybrid_layers = nn.ModuleList([
            ETNNCCMPNLayer(
                hidden_dim=self.hidden_dim,
                ccmpn_neighborhoods=['up', 'down', 'boundary'],
                etnn_neighborhoods=['up', 'down', 'boundary'],
                use_geometric_invariants=True,
                fusion_method='attention',
                dropout=config.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # Output projection layers for different hierarchy levels
        self.output_projections = nn.ModuleDict({
            str(rank): nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(self.hidden_dim // 2, 1)
            ) for rank in range(config.max_hierarchy_depth)
        })
        
        # Hierarchical consistency layer
        self.consistency_layer = HierarchicalConsistencyLayer(
            hidden_dim=self.hidden_dim,
            consistency_weight=config.consistency_weight
        )
        
        # Geometric feature extractor
        self.geometric_feature_extractor = GeometricFeatureExtractor(
            spatial_dim=config.spatial_dim,
            hidden_dim=self.hidden_dim
        )
        
    def _create_initial_positions(self, batch_size: int, num_entities: int) -> Dict[int, torch.Tensor]:
        """
        Create initial positions for entities in the hierarchy.
        
        Args:
            batch_size: Batch size
            num_entities: Number of entities per hierarchy level
            
        Returns:
            Dictionary of positions for each hierarchy rank
        """
        positions = {}
        
        # Use learned position embeddings
        entity_positions = self.position_embedding[:num_entities]  # [num_entities, spatial_dim]
        
        # Expand for batch dimension
        for rank in range(self.config.max_hierarchy_depth):
            positions[rank] = entity_positions.unsqueeze(0).expand(
                batch_size, -1, -1
            ).contiguous().view(-1, self.config.spatial_dim)
        
        return positions
    
    def forward(self,
                x: torch.Tensor,
                combinatorial_complex: CombinatorialComplex,
                hierarchy_info: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ETNN-enhanced hierarchical model.
        
        Args:
            x: Input features [batch_size, num_features]
            combinatorial_complex: Combinatorial complex structure
            hierarchy_info: Optional hierarchy information
            
        Returns:
            Dictionary of predictions for each hierarchy level
        """
        batch_size = x.size(0)
        
        # Project input features
        h = self.input_projection(x)  # [batch_size, hidden_dim]
        
        # Determine number of entities from combinatorial complex
        max_entities = max(len(cells) for cells in combinatorial_complex.cells.values())
        
        # Create initial cell features
        cell_features = {}
        for rank, cells in combinatorial_complex.cells.items():
            if len(cells) > 0:
                # Expand features for each cell at this rank
                num_cells = len(cells)
                cell_features[rank] = h.unsqueeze(1).expand(
                    batch_size, num_cells, self.hidden_dim
                ).contiguous().view(-1, self.hidden_dim)
        
        # Create initial positions
        cell_positions = self._create_initial_positions(batch_size, max_entities)
        
        # Add geometric features
        geometric_features = self.geometric_feature_extractor(cell_positions)
        for rank in cell_features.keys():
            if rank in geometric_features:
                cell_features[rank] = cell_features[rank] + geometric_features[rank]
        
        # Apply hybrid ETNN-CCMPN layers
        for layer in self.hybrid_layers:
            cell_features, cell_positions = layer(cell_features, cell_positions, combinatorial_complex)
        
        # Generate predictions for each hierarchy level
        predictions = {}
        for rank, features in cell_features.items():
            if str(rank) in self.output_projections:
                # Reshape to [batch_size, num_cells, hidden_dim]
                num_cells = len(combinatorial_complex.cells[rank])
                reshaped_features = features.view(batch_size, num_cells, self.hidden_dim)
                
                # Apply output projection
                rank_predictions = self.output_projections[str(rank)](reshaped_features)
                predictions[f'rank_{rank}'] = rank_predictions.squeeze(-1)  # [batch_size, num_cells]
        
        # Apply hierarchical consistency constraints
        predictions = self.consistency_layer(predictions, combinatorial_complex)
        
        return predictions


class HierarchicalConsistencyLayer(nn.Module):
    """
    Layer that enforces hierarchical consistency constraints.
    
    This layer ensures that predictions at different hierarchy levels
    are consistent with the hierarchical structure.
    """
    
    def __init__(self, hidden_dim: int, consistency_weight: float = 0.1):
        """
        Initialize hierarchical consistency layer.
        
        Args:
            hidden_dim: Hidden dimension
            consistency_weight: Weight for consistency loss
        """
        super().__init__()
        self.consistency_weight = consistency_weight
        
        # Learnable consistency transformations
        self.consistency_transforms = nn.ModuleDict()
        
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                combinatorial_complex: CombinatorialComplex) -> Dict[str, torch.Tensor]:
        """
        Apply hierarchical consistency constraints.
        
        Args:
            predictions: Predictions for each hierarchy level
            combinatorial_complex: Combinatorial complex structure
            
        Returns:
            Consistency-adjusted predictions
        """
        # For now, return predictions as-is
        # In a full implementation, this would enforce constraints like:
        # - Sum of child predictions = parent prediction
        # - Temporal consistency across time steps
        # - Structural consistency based on combinatorial complex
        
        return predictions


class GeometricFeatureExtractor(nn.Module):
    """
    Extracts geometric features from spatial positions using ETNN-inspired invariants.
    """
    
    def __init__(self, spatial_dim: int, hidden_dim: int):
        """
        Initialize geometric feature extractor.
        
        Args:
            spatial_dim: Spatial dimension
            hidden_dim: Hidden dimension for output features
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        
        # MLP for processing geometric invariants
        self.geometric_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),  # 3 geometric invariants
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
    def forward(self, cell_positions: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Extract geometric features from positions.
        
        Args:
            cell_positions: Positions for each cell type
            
        Returns:
            Geometric features for each cell type
        """
        geometric_features = {}
        
        for rank, positions in cell_positions.items():
            if positions.size(0) == 0:
                continue
                
            # Compute geometric invariants for each position
            batch_size = positions.size(0)
            features = torch.zeros(batch_size, 3, device=positions.device)
            
            # For simplicity, compute invariants based on position magnitude and relationships
            for i in range(batch_size):
                pos_i = positions[i:i+1]  # [1, spatial_dim]
                
                # Compute simple geometric features
                features[i, 0] = torch.norm(pos_i)  # Distance from origin
                features[i, 1] = pos_i.sum()        # Sum of coordinates
                features[i, 2] = (pos_i ** 2).sum() # Squared magnitude
            
            # Project through MLP
            geometric_features[rank] = self.geometric_mlp(features)
        
        return geometric_features
