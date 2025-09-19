"""
Base class for all baseline models.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional


class BaselineModel(ABC):
    """
    Abstract base class for all baseline models.
    
    This provides a common interface for training, evaluation, and prediction
    across different baseline approaches.
    """
    
    def __init__(self, name: str):
        """
        Initialize the baseline model.
        
        Args:
            name: Name of the baseline model
        """
        self.name = name
        self.is_fitted = False
        self.feature_names = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, 
            hierarchy: Optional[Dict] = None, **kwargs) -> 'BaselineModel':
        """
        Fit the baseline model to training data.
        
        Args:
            X: Input features [n_samples, n_features]
            y: Target values [n_samples]
            hierarchy: Hierarchical structure (optional)
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Input features [n_samples, n_features]
            **kwargs: Additional arguments
            
        Returns:
            Predictions [n_samples]
        """
        pass
    
    def score(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X: Input features
            y: True targets
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X, **kwargs)
        
        # Calculate common metrics
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # MAPE (avoiding division by zero)
        mape = np.mean(np.abs((y - predictions) / np.where(y == 0, 1, y))) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {'name': self.name}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class HierarchicalBaselineModel(BaselineModel):
    """
    Base class for hierarchical baseline models.
    
    These models specifically handle hierarchical forecasting with
    coherence constraints and reconciliation methods.
    """
    
    def __init__(self, name: str, reconciliation_method: str = 'bottom_up'):
        """
        Initialize hierarchical baseline model.
        
        Args:
            name: Name of the model
            reconciliation_method: Method for hierarchical reconciliation
        """
        super().__init__(name)
        self.reconciliation_method = reconciliation_method
        self.hierarchy_structure = None
        self.aggregation_matrix = None
        
    def build_aggregation_matrix(self, hierarchy: Dict) -> np.ndarray:
        """
        Build aggregation matrix for hierarchical reconciliation.
        
        Args:
            hierarchy: Hierarchical structure
            
        Returns:
            Aggregation matrix S where S*bottom_level = all_levels
        """
        # Flatten hierarchy to get all entities
        all_entities = []
        for level, entities in hierarchy.items():
            all_entities.extend(entities)
        
        # Create mapping
        entity_to_idx = {entity: idx for idx, entity in enumerate(all_entities)}
        
        # For simplicity, assume bottom level is 'sku' and build basic matrix
        bottom_entities = hierarchy.get('sku', [])
        n_bottom = len(bottom_entities)
        n_total = len(all_entities)
        
        # Identity matrix for bottom level + aggregation for upper levels
        S = np.zeros((n_total, n_bottom))
        
        # Bottom level maps to itself
        for i, entity in enumerate(bottom_entities):
            if entity in entity_to_idx:
                S[entity_to_idx[entity], i] = 1
        
        # Simple aggregation for upper levels (sum all bottom)
        for level in ['store', 'company', 'total']:
            if level in hierarchy:
                for entity in hierarchy[level]:
                    if entity in entity_to_idx:
                        row_idx = entity_to_idx[entity]
                        S[row_idx, :] = 1  # Sum all bottom level
        
        return S
    
    @abstractmethod
    def reconcile_forecasts(self, base_forecasts: np.ndarray) -> np.ndarray:
        """
        Reconcile forecasts to ensure hierarchical coherence.
        
        Args:
            base_forecasts: Base forecasts for all levels
            
        Returns:
            Reconciled forecasts
        """
        pass
