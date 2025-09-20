"""
ETNN-based baseline models for hierarchical forecasting.

This module implements E(n) Equivariant Topological Neural Networks
as baseline models following the ICLR 2025 paper concepts.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from .base_baseline import BaseBaseline
from ..models.etnn_layers import ETNNLayer, GeometricInvariants
from ..models.combinatorial_complex import CombinatorialComplex


class ETNNBaseline(BaseBaseline):
    """
    E(n) Equivariant Topological Neural Network baseline for hierarchical forecasting.
    
    This baseline implements ETNN concepts adapted for time series forecasting
    with hierarchical structure modeling through combinatorial complexes.
    """
    
    def __init__(self, 
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 spatial_dim: int = 2,
                 use_geometric_features: bool = True,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 device: Optional[str] = None):
        """
        Initialize ETNN baseline.
        
        Args:
            hidden_dim: Hidden dimension for ETNN layers
            num_layers: Number of ETNN layers
            spatial_dim: Dimension of geometric embedding space
            use_geometric_features: Whether to use geometric invariants
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.spatial_dim = spatial_dim
        self.use_geometric_features = use_geometric_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.feature_dim = None
        self.scaler_features = None
        self.scaler_targets = None
        
    def _create_geometric_embedding(self, hierarchy_info: Dict[str, Any]) -> torch.Tensor:
        """
        Create geometric embedding for entities based on hierarchy.
        
        Args:
            hierarchy_info: Dictionary containing hierarchy structure
            
        Returns:
            Geometric positions for each entity [num_entities, spatial_dim]
        """
        # Create simple geometric embedding based on hierarchy levels
        # In practice, this could be learned or based on actual spatial relationships
        
        num_entities = len(hierarchy_info.get('entities', []))
        if num_entities == 0:
            return torch.zeros(1, self.spatial_dim, device=self.device)
        
        # Create random but consistent positions based on entity IDs
        positions = []
        for i, entity in enumerate(hierarchy_info.get('entities', [])):
            # Use entity hash for consistent positioning
            seed = hash(str(entity)) % 10000
            torch.manual_seed(seed)
            pos = torch.randn(self.spatial_dim) * 0.1
            positions.append(pos)
        
        return torch.stack(positions).to(self.device)
    
    def _create_combinatorial_complex(self, hierarchy_info: Dict[str, Any]) -> CombinatorialComplex:
        """
        Create combinatorial complex from hierarchy information.
        
        Args:
            hierarchy_info: Dictionary containing hierarchy structure
            
        Returns:
            CombinatorialComplex representing the hierarchical structure
        """
        entities = hierarchy_info.get('entities', [])
        
        # Create simple combinatorial complex
        # 0-cells: individual entities
        # 1-cells: pairs of related entities
        # 2-cells: groups of entities at same level
        
        cells = {
            0: list(range(len(entities))),  # Each entity is a 0-cell
            1: [],  # Edges between related entities
            2: []   # Higher-order relationships
        }
        
        # Add edges for hierarchically related entities
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if i < len(entities) // 2 and j < len(entities) // 2:
                    # Connect entities in same level
                    cells[1].append([i, j])
        
        # Add 2-cells for groups
        if len(entities) >= 3:
            cells[2].append([0, 1, 2])  # First three entities form a 2-cell
        
        return CombinatorialComplex(cells)
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            hierarchy_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Train the ETNN baseline model.
        
        Args:
            X_train: Training features [num_samples, num_features]
            y_train: Training targets [num_samples, num_entities]
            hierarchy_info: Optional hierarchy information
        """
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features and targets
        self.scaler_features = StandardScaler()
        self.scaler_targets = StandardScaler()
        
        X_scaled = self.scaler_features.fit_transform(X_train)
        y_scaled = self.scaler_targets.fit_transform(y_train)
        
        self.feature_dim = X_train.shape[1]
        num_entities = y_train.shape[1]
        
        # Create geometric embedding and combinatorial complex
        if hierarchy_info is None:
            hierarchy_info = {'entities': list(range(num_entities))}
        
        positions = self._create_geometric_embedding(hierarchy_info)
        cc = self._create_combinatorial_complex(hierarchy_info)
        
        # Create ETNN model
        self.model = ETNNForecastingModel(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            output_dim=num_entities,
            num_layers=self.num_layers,
            spatial_dim=self.spatial_dim,
            combinatorial_complex=cc,
            positions=positions,
            use_geometric_features=self.use_geometric_features
        ).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(X_tensor)
            loss = criterion(predictions, y_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"ETNN Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.6f}")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained ETNN model.
        
        Args:
            X_test: Test features [num_samples, num_features]
            
        Returns:
            Predictions [num_samples, num_entities]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Standardize features
        X_scaled = self.scaler_features.transform(X_test)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor)
            predictions = self.scaler_targets.inverse_transform(
                predictions_scaled.cpu().numpy()
            )
        
        return predictions
    
    def get_name(self) -> str:
        """Get the name of this baseline method."""
        return "ETNN"


class ETNNForecastingModel(nn.Module):
    """
    ETNN-based forecasting model that operates on combinatorial complexes.
    """
    
    def __init__(self,
                 feature_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 spatial_dim: int,
                 combinatorial_complex: CombinatorialComplex,
                 positions: torch.Tensor,
                 use_geometric_features: bool = True):
        """
        Initialize ETNN forecasting model.
        
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden dimension for ETNN layers
            output_dim: Output dimension (number of entities)
            num_layers: Number of ETNN layers
            spatial_dim: Spatial dimension for geometric features
            combinatorial_complex: Combinatorial complex structure
            positions: Initial positions for entities
            use_geometric_features: Whether to use geometric invariants
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.spatial_dim = spatial_dim
        self.use_geometric_features = use_geometric_features
        
        self.cc = combinatorial_complex
        self.register_buffer('positions', positions)
        
        # Input projection
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        
        # ETNN layers
        self.etnn_layers = nn.ModuleList([
            ETNNLayer(
                hidden_dim=hidden_dim,
                neighborhood_names=['up', 'down', 'boundary'],
                use_position_update=True,
                use_geometric_invariants=use_geometric_features
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, 1)
        
        # Geometric feature extractor
        if use_geometric_features:
            self.geometric_mlp = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),  # 3 geometric invariants
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
    
    def _extract_geometric_features(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Extract geometric invariant features from positions.
        
        Args:
            positions: Entity positions [num_entities, spatial_dim]
            
        Returns:
            Geometric features [num_entities, hidden_dim]
        """
        num_entities = positions.size(0)
        geometric_features = torch.zeros(num_entities, 3, device=positions.device)
        
        for i in range(num_entities):
            for j in range(i + 1, num_entities):
                pos_i = positions[i:i+1]  # [1, spatial_dim]
                pos_j = positions[j:j+1]  # [1, spatial_dim]
                
                # Compute geometric invariants
                pairwise_dist = GeometricInvariants.pairwise_distances(pos_i, pos_j)
                centroid_dist = GeometricInvariants.centroid_distance(pos_i, pos_j)
                hausdorff_dist = GeometricInvariants.hausdorff_distance(pos_i, pos_j)
                
                # Aggregate features (simple averaging)
                geometric_features[i, 0] += pairwise_dist / (num_entities - 1)
                geometric_features[i, 1] += centroid_dist / (num_entities - 1)
                geometric_features[i, 2] += hausdorff_dist / (num_entities - 1)
                
                geometric_features[j, 0] += pairwise_dist / (num_entities - 1)
                geometric_features[j, 1] += centroid_dist / (num_entities - 1)
                geometric_features[j, 2] += hausdorff_dist / (num_entities - 1)
        
        return self.geometric_mlp(geometric_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ETNN model.
        
        Args:
            x: Input features [batch_size, feature_dim]
            
        Returns:
            Predictions [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # Project input to hidden dimension
        h = self.input_projection(x)  # [batch_size, hidden_dim]
        
        # Expand to match number of entities
        h = h.unsqueeze(1).expand(batch_size, self.output_dim, self.hidden_dim)
        # [batch_size, output_dim, hidden_dim]
        
        # Create cell features dictionary for ETNN layers
        cell_features = {
            0: h.view(-1, self.hidden_dim)  # [batch_size * output_dim, hidden_dim]
        }
        
        # Create positions for each batch item
        positions = self.positions.unsqueeze(0).expand(batch_size, -1, -1)
        # [batch_size, output_dim, spatial_dim]
        
        cell_positions = {
            0: positions.view(-1, self.spatial_dim)  # [batch_size * output_dim, spatial_dim]
        }
        
        # Add geometric features if enabled
        if self.use_geometric_features:
            geom_features = self._extract_geometric_features(self.positions)
            geom_features = geom_features.unsqueeze(0).expand(batch_size, -1, -1)
            geom_features = geom_features.view(-1, self.hidden_dim)
            cell_features[0] = cell_features[0] + geom_features
        
        # Apply ETNN layers
        for layer in self.etnn_layers:
            cell_features, cell_positions = layer(cell_features, cell_positions, self.cc)
        
        # Extract 0-cell features and reshape
        output_features = cell_features[0].view(batch_size, self.output_dim, self.hidden_dim)
        
        # Project to output
        predictions = self.output_projection(output_features).squeeze(-1)
        # [batch_size, output_dim]
        
        return predictions


class SimplifiedETNNBaseline(BaseBaseline):
    """
    Simplified ETNN baseline that focuses on geometric invariants
    without full combinatorial complex machinery.
    """
    
    def __init__(self, 
                 hidden_dim: int = 32,
                 num_layers: int = 2,
                 learning_rate: float = 0.001,
                 epochs: int = 50,
                 device: Optional[str] = None):
        """
        Initialize simplified ETNN baseline.
        
        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            learning_rate: Learning rate
            epochs: Training epochs
            device: Device to use
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.scaler_features = None
        self.scaler_targets = None
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            hierarchy_info: Optional[Dict[str, Any]] = None) -> None:
        """Train the simplified ETNN model."""
        from sklearn.preprocessing import StandardScaler
        
        # Standardize data
        self.scaler_features = StandardScaler()
        self.scaler_targets = StandardScaler()
        
        X_scaled = self.scaler_features.fit_transform(X_train)
        y_scaled = self.scaler_targets.fit_transform(y_train)
        
        # Create simple neural network with geometric-inspired features
        self.model = SimplifiedETNNModel(
            feature_dim=X_train.shape[1],
            hidden_dim=self.hidden_dim,
            output_dim=y_train.shape[1],
            num_layers=self.num_layers
        ).to(self.device)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            predictions = self.model(X_tensor)
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Simplified ETNN Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.6f}")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions with simplified ETNN model."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler_features.transform(X_test)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor)
            predictions = self.scaler_targets.inverse_transform(
                predictions_scaled.cpu().numpy()
            )
        
        return predictions
    
    def get_name(self) -> str:
        """Get the name of this baseline method."""
        return "Simplified_ETNN"


class SimplifiedETNNModel(nn.Module):
    """Simplified neural network inspired by ETNN geometric principles."""
    
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(feature_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
