"""
ETNN-based baseline models for hierarchical forecasting.

This module implements E(n) Equivariant Topological Neural Networks
as baseline models following the ICLR 2025 paper concepts.
"""

import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Dict, Iterable, List, Optional, Tuple
from .base import BaselineModel
from ..models.etnn_layers import ETNNLayer, GeometricInvariants
from ..models.combinatorial_complex import CombinatorialComplex


class ETNNBaseline(BaselineModel):
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
                 epochs: int = 150,
                 batch_size: int = 128,
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
        super().__init__(name="ETNN")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.spatial_dim = spatial_dim
        self.use_geometric_features = use_geometric_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.feature_dim = None
        self.output_dim = None
        self.scaler_features = None
        self.scaler_targets = None
        self._target_is_vector = False
        self._entity_index: Dict[Tuple[Any, ...], int] = {}
        self._complex: Optional[CombinatorialComplex] = None
        
    def _resolve_sku_entities(self, hierarchy_info: Dict[str, Any]) -> List[Tuple[Any, ...]]:
        candidates = [
            hierarchy_info.get(0),
            hierarchy_info.get('sku'),
            hierarchy_info.get('skus'),
            hierarchy_info.get('entities')
        ]
        for candidate in candidates:
            if candidate:
                raw_entities = list(candidate)
                break
        else:
            raw_entities = []

        processed_entities: List[Tuple[Any, ...]] = []
        for entity in raw_entities:
            processed_entities.append(self._canonical_entity(entity))

        return processed_entities

    def _create_geometric_embedding(self, hierarchy_info: Dict[str, Any]) -> torch.Tensor:
        """
        Create deterministic geometric embeddings that reflect the hierarchy.

        We map companies, stores, and SKUs to a 3-D coordinate system where each
        dimension corresponds to the position of the entity within its level. The
        mapping is stable across runs and does not rely on random sampling.

        Args:
            hierarchy_info: Dictionary describing hierarchy levels. Expected keys
                are either integer ranks (0 for SKU level) or descriptive strings
                (e.g., "sku").

        Returns:
            Tensor of shape [num_entities, spatial_dim] with coordinates in [0, 1].
        """

        def _normalise(index: int, count: int) -> float:
            if count <= 1:
                return 0.5
            return index / float(count - 1)

        sku_entities = self._resolve_sku_entities(hierarchy_info)

        if not sku_entities:
            # Fall back to a simple linear embedding if hierarchy is missing.
            return torch.linspace(0.0, 1.0, steps=max(1, len(hierarchy_info.get('entities', []))),
                                  device=self.device).unsqueeze(-1).repeat(1, self.spatial_dim)
        processed_entities = sku_entities

        if not processed_entities:
            return torch.zeros(1, self.spatial_dim, device=self.device)

        # Build ordered mappings for companies, stores, and SKUs.
        from collections import defaultdict

        companies: List[Any] = sorted({ent[0] for ent in processed_entities if len(ent) >= 1})
        company_to_idx = {company: idx for idx, company in enumerate(companies)}

        stores_by_company: Dict[Any, List[Any]] = defaultdict(list)
        skus_by_store: Dict[Tuple[Any, Any], List[Any]] = defaultdict(list)

        for ent in processed_entities:
            if len(ent) >= 2:
                comp, store = ent[0], ent[1]
                if store not in stores_by_company[comp]:
                    stores_by_company[comp].append(store)
            if len(ent) >= 3:
                comp, store, sku = ent[0], ent[1], ent[2]
                key = (comp, store)
                if sku not in skus_by_store[key]:
                    skus_by_store[key].append(sku)

        for comp in stores_by_company:
            stores_by_company[comp].sort()
        for key in skus_by_store:
            skus_by_store[key].sort()

        coordinates: List[torch.Tensor] = []
        for ent in processed_entities:
            comp = ent[0]
            comp_count = len(companies)
            comp_coord = _normalise(company_to_idx.get(comp, 0), comp_count)

            if len(ent) >= 2:
                store_key = ent[1]
                store_list = stores_by_company.get(comp, [store_key])
                if store_key not in store_list:
                    store_list.append(store_key)
                    store_list.sort()
                    stores_by_company[comp] = store_list
                store_coord = _normalise(store_list.index(store_key), len(store_list))
            else:
                store_coord = 0.5

            if len(ent) >= 3:
                sku_key = ent[2]
                sku_list = skus_by_store.get((comp, ent[1]), [sku_key])
                if sku_key not in sku_list:
                    sku_list.append(sku_key)
                    sku_list.sort()
                    skus_by_store[(comp, ent[1])] = sku_list
                sku_coord = _normalise(sku_list.index(sku_key), len(sku_list))
            else:
                sku_coord = 0.5

            coord_vec = torch.tensor([comp_coord, store_coord, sku_coord], device=self.device)
            if self.spatial_dim < 3:
                coord_vec = coord_vec[:self.spatial_dim]
            elif self.spatial_dim > 3:
                pad = torch.zeros(self.spatial_dim - 3, device=self.device)
                coord_vec = torch.cat([coord_vec, pad], dim=0)
            coordinates.append(coord_vec)

        return torch.stack(coordinates, dim=0)
    
    def _create_combinatorial_complex(self, hierarchy_info: Dict[str, Any]) -> CombinatorialComplex:
        """
        Create combinatorial complex from hierarchy information.
        
        Args:
            hierarchy_info: Dictionary containing hierarchy structure
            
        Returns:
            CombinatorialComplex representing the hierarchical structure
        """
        sku_level = self._resolve_sku_entities(hierarchy_info)

        if sku_level and all(len(ent) >= 3 for ent in sku_level):
            rows = []
            for ent in sku_level:
                comp, store, sku = ent[0], ent[1], ent[2]
                rows.append({'companyID': comp, 'storeID': store, 'skuID': sku})
            hierarchy_source = pd.DataFrame(rows)
        else:
            # Fall back to previously used synthetic structure when hierarchy is not
            # explicitly provided.
            num_entities = len(sku_level) if sku_level else hierarchy_info.get('num_entities', 0)
            if not num_entities:
                num_entities = len(hierarchy_info.get('entities', []))
            hierarchy_source = pd.DataFrame({
                'companyID': np.arange(num_entities),
                'storeID': np.zeros(num_entities),
                'skuID': np.arange(num_entities)
            })

        return CombinatorialComplex(hierarchy_source)

    @staticmethod
    def _canonical_entity(entity: Any) -> Tuple[Any, ...]:
        if isinstance(entity, tuple):
            return entity
        if isinstance(entity, list):
            return tuple(entity)
        if isinstance(entity, str):
            delimiter = '|' if '|' in entity else '_'
            parts = entity.split(delimiter)
            return tuple(parts)
        return (entity,)

    def _build_cell_positions(
        self,
        cc: CombinatorialComplex,
        bottom_positions: torch.Tensor,
        sku_entities: List[Tuple[Any, ...]]
    ) -> torch.Tensor:
        positions = torch.zeros(cc.num_cells, bottom_positions.size(1), device=bottom_positions.device)

        position_lookup = {
            self._canonical_entity(entity)[:3]: coord
            for entity, coord in zip(sku_entities, bottom_positions)
        }

        for sku_cell in cc.cells[0]:
            coord = position_lookup.get(self._canonical_entity(sku_cell)[:3])
            if coord is None:
                raise KeyError(f"Missing geometric embedding for entity {sku_cell}")
            positions[cc.cell_to_int[sku_cell]] = coord

        for rank in sorted(cc.cells.keys()):
            if rank == 0:
                continue
            for cell in cc.cells[rank]:
                idx = cc.cell_to_int[cell]
                children = cc.children_map.get(cell, [])
                if not children:
                    continue
                child_coords = torch.stack([positions[cc.cell_to_int[ch]] for ch in children], dim=0)
                positions[idx] = child_coords.mean(dim=0)

        return positions

    def _build_base_features(self, cc: CombinatorialComplex, positions: torch.Tensor) -> torch.Tensor:
        num_ranks = len(cc.cells)
        base_features: List[torch.Tensor] = []

        for idx in range(cc.num_cells):
            cell = cc.int_to_cell[idx]
            rank = cc.rank_lookup.get(cell, 0)
            rank_one_hot = torch.zeros(num_ranks, device=positions.device)
            rank_one_hot[rank] = 1.0
            child_count = float(len(cc.children_map.get(cell, [])))
            structural = torch.tensor([math.log1p(child_count)], device=positions.device)
            base_features.append(torch.cat([rank_one_hot, structural, positions[idx]], dim=0))

        return torch.stack(base_features, dim=0)
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            hierarchy: Optional[Dict] = None,
            **kwargs) -> 'ETNNBaseline':
        """
        Train the ETNN baseline model.
        
        Args:
            X_train: Training features [num_samples, num_features]
            y_train: Training targets [num_samples, num_entities]
            hierarchy: Optional hierarchy information
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        from sklearn.preprocessing import StandardScaler
        
        # Ensure inputs are two-dimensional for the scalers
        X_train = np.asarray(X_train)
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)

        y_train = np.asarray(y_train)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        self._target_is_vector = y_train.shape[1] == 1

        # Standardize features and targets
        self.scaler_features = StandardScaler()
        self.scaler_targets = StandardScaler()

        X_scaled = self.scaler_features.fit_transform(X_train)
        y_scaled = self.scaler_targets.fit_transform(y_train)

        self.feature_dim = X_train.shape[1]

        entity_ids = kwargs.get('entity_ids')
        if entity_ids is None:
            raise ValueError("ETNNBaseline.fit requires 'entity_ids' in kwargs for alignment.")

        sku_entities = self._resolve_sku_entities(hierarchy or {})
        complex_structure = self._create_combinatorial_complex(hierarchy or {})
        self._complex = complex_structure

        bottom_positions = self._create_geometric_embedding(hierarchy or {})
        cell_positions = self._build_cell_positions(complex_structure, bottom_positions, sku_entities)
        base_features = self._build_base_features(complex_structure, cell_positions)

        self._entity_index = {
            self._canonical_entity(cell)[:3]: complex_structure.cell_to_int[cell]
            for cell in complex_structure.cells[0]
        }

        def map_entities(raw_ids: Iterable[Any]) -> torch.Tensor:
            indices = []
            for raw in raw_ids:
                canonical = self._canonical_entity(raw)[:3]
                if canonical not in self._entity_index:
                    raise KeyError(f"Unknown entity identifier {raw}")
                indices.append(self._entity_index[canonical])
            return torch.tensor(indices, dtype=torch.long)

        entity_indices = map_entities(entity_ids)

        self.model = ETNNForecastingModel(
            input_dim=self.feature_dim,
            base_features=base_features.to(self.device),
            positions=cell_positions.to(self.device),
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            combinatorial_complex=complex_structure,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        dataset = TensorDataset(
            torch.FloatTensor(X_scaled),
            torch.FloatTensor(y_scaled),
            entity_indices
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            total_samples = 0
            for features_batch, targets_batch, entity_batch in dataloader:
                features_batch = features_batch.to(self.device)
                targets_batch = targets_batch.to(self.device)
                entity_batch = entity_batch.to(self.device)

                optimizer.zero_grad()
                preds = self.model(features_batch, entity_batch)
                loss = criterion(preds.squeeze(-1), targets_batch.squeeze(-1))
                loss.backward()
                optimizer.step()

                batch_size = features_batch.size(0)
                epoch_loss += loss.item() * batch_size
                total_samples += batch_size

            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                avg_loss = epoch_loss / max(1, total_samples)
                print(f"ETNN Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

        self.is_fitted = True
        return self
    
    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """Run inference using the trained ETNN model."""
        if self.model is None or self._complex is None:
            raise ValueError("Model must be fitted before calling predict().")

        entity_ids = kwargs.get('entity_ids')
        if entity_ids is None:
            raise ValueError("ETNNBaseline.predict requires 'entity_ids' in kwargs.")

        X_test = np.asarray(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        entity_indices = []
        for raw in entity_ids:
            canonical = self._canonical_entity(raw)[:3]
            if canonical not in self._entity_index:
                raise KeyError(f"Unknown entity identifier {raw}")
            entity_indices.append(self._entity_index[canonical])

        entity_tensor = torch.tensor(entity_indices, dtype=torch.long).to(self.device)

        X_scaled = self.scaler_features.transform(X_test)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds_scaled = self.model(X_tensor, entity_tensor)
        preds = preds_scaled.cpu().numpy().reshape(-1, 1)
        preds = self.scaler_targets.inverse_transform(preds)

        if self._target_is_vector:
            return preds.ravel()

        return preds
    
    def get_name(self) -> str:
        """Get the name of this baseline method."""
        return "ETNN"


class ETNNForecastingModel(nn.Module):
    """Lightweight ETNN forecaster operating on hierarchy complexes."""

    def __init__(
        self,
        input_dim: int,
        base_features: torch.Tensor,
        positions: torch.Tensor,
        hidden_dim: int,
        num_layers: int,
        combinatorial_complex: CombinatorialComplex,
        use_position_update: bool = True,
    ) -> None:
        super().__init__()

        self.num_cells = base_features.size(0)
        self.dynamic_dim = input_dim
        self.base_dim = base_features.size(1)
        self.hidden_dim = hidden_dim

        self.register_buffer('base_features', base_features)
        self.register_buffer('base_positions', positions)

        neighborhood_names = [name for name in ('up', 'down', 'boundary') if name in combinatorial_complex.neighborhoods]

        self.input_projection = nn.Linear(self.base_dim + self.dynamic_dim, hidden_dim)
        self.layers = nn.ModuleList([
            ETNNLayer(hidden_dim, neighborhood_names, use_position_update=use_position_update)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Store edge indices and node degrees as buffers so they reside on the
        # correct device together with the model parameters.
        self.edge_names: List[str] = []
        self.node_degrees: Dict[str, torch.Tensor] = {}
        for name in neighborhood_names:
            matrix = combinatorial_complex.neighborhoods.get(name)
            if matrix is None or matrix._nnz() == 0:
                continue
            indices = matrix.coalesce().indices()
            degrees = torch.bincount(indices[1], minlength=self.num_cells)
            self.edge_names.append(name)
            self.register_buffer(f'{name}_edge_index', indices)
            self.register_buffer(f'{name}_degree', degrees)

    def _edge_tensors(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        edge_indices: Dict[str, torch.Tensor] = {}
        node_degrees: Dict[str, torch.Tensor] = {}
        for name in self.edge_names:
            edge_indices[name] = getattr(self, f'{name}_edge_index')
            node_degrees[name] = getattr(self, f'{name}_degree')
        return edge_indices, node_degrees

    def forward(self, x: torch.Tensor, entity_indices: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        base = self.base_features.unsqueeze(0).expand(batch_size, -1, -1)
        dynamic = torch.zeros(batch_size, self.num_cells, self.dynamic_dim, device=x.device)
        dynamic[torch.arange(batch_size, device=x.device), entity_indices] = x
        inputs = torch.cat([base, dynamic], dim=-1)

        hidden = self.input_projection(inputs)
        positions = self.base_positions.unsqueeze(0).expand(batch_size, -1, -1)

        edge_indices, node_degrees = self._edge_tensors()
        for layer in self.layers:
            hidden, positions = layer(hidden, positions, edge_indices, node_degrees)

        entity_embeddings = hidden[torch.arange(batch_size, device=x.device), entity_indices]
        return self.output_projection(entity_embeddings)

