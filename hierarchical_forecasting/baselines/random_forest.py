"""
Random Forest baseline for hierarchical forecasting.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional
from .base import BaselineModel


class RandomForestBaseline(BaselineModel):
    """
    Random Forest baseline for hierarchical forecasting.
    
    This baseline uses ensemble learning with decision trees to capture
    non-linear relationships in the data.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 random_state: int = 42, normalize: bool = False):
        """
        Initialize Random Forest baseline.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            random_state: Random state for reproducibility
            normalize: Whether to normalize features
        """
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.normalize = normalize
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        
        self.scaler = StandardScaler() if normalize else None
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            hierarchy: Optional[Dict] = None, **kwargs) -> 'RandomForestBaseline':
        """
        Fit the Random Forest model.
        
        Args:
            X: Input features [n_samples, n_features]
            y: Target values [n_samples] or [n_samples, 1]
            hierarchy: Ignored for this baseline
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        # Handle both 1D and 2D target arrays
        self._target_was_2d = False
        if y.ndim == 2 and y.shape[1] == 1:
            self._target_was_2d = True
            y = y.ravel()  # Convert 2D single column to 1D
        elif y.ndim == 2 and y.shape[1] > 1:
            raise ValueError("Random Forest baseline only supports single target. Use HierarchicalRandomForest for multiple targets.")
        
        # Normalize features if requested
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Fit the model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions using the fitted Random Forest model.
        
        Args:
            X: Input features [n_samples, n_features]
            **kwargs: Additional arguments
            
        Returns:
            Predictions [n_samples] or [n_samples, 1] to match input format
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Apply same scaling as training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        predictions = self.model.predict(X_scaled)
        
        # If we expect 2D output (single column), reshape accordingly
        # This maintains consistency with hierarchical forecasting expectations
        if hasattr(self, '_target_was_2d') and self._target_was_2d:
            predictions = predictions.reshape(-1, 1)
        
        return predictions
    
    def predict_with_uncertainty(self, X: np.ndarray, **kwargs) -> tuple:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X: Input features [n_samples, n_features]
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (predictions, std_predictions)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Apply same scaling as training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict(X_scaled) for tree in self.model.estimators_
        ])
        
        # Calculate mean and standard deviation
        predictions = np.mean(tree_predictions, axis=0)
        std_predictions = np.std(tree_predictions, axis=0)
        
        return predictions, std_predictions
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.feature_importances_
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'normalize': self.normalize
        })
        return params


class HierarchicalRandomForest(BaselineModel):
    """
    Random Forest with hierarchy-aware features.
    
    This baseline incorporates hierarchical information by creating
    level-specific features and training separate models.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 random_state: int = 42):
        """
        Initialize Hierarchical Random Forest.
        
        Args:
            n_estimators: Number of trees per level
            max_depth: Maximum depth of trees
            random_state: Random state for reproducibility
        """
        super().__init__("HierarchicalRandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.level_models = {}
        self.level_scalers = {}
        
    def _create_model(self):
        """Create a new Random Forest model."""
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            hierarchy: Optional[Dict] = None,
            entity_levels: Optional[np.ndarray] = None, **kwargs) -> 'HierarchicalRandomForest':
        """
        Fit separate Random Forest models for each hierarchy level.
        
        Args:
            X: Input features [n_samples, n_features]
            y: Target values [n_samples]
            hierarchy: Hierarchy structure
            entity_levels: Level indicator for each sample
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        if entity_levels is None:
            entity_levels = np.zeros(len(X))
        
        # Train separate model for each level
        for level in np.unique(entity_levels):
            level_mask = entity_levels == level
            X_level = X[level_mask]
            y_level = y[level_mask]
            
            if len(X_level) > 0:
                model = self._create_model()
                model.fit(X_level, y_level)
                self.level_models[level] = model
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, entity_levels: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """
        Make predictions using level-specific models.
        
        Args:
            X: Input features [n_samples, n_features]
            entity_levels: Level indicator for each sample
            **kwargs: Additional arguments
            
        Returns:
            Predictions [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if entity_levels is None:
            entity_levels = np.zeros(len(X))
        
        predictions = np.zeros(len(X))
        
        for level in np.unique(entity_levels):
            level_mask = entity_levels == level
            
            if level in self.level_models and np.any(level_mask):
                X_level = X[level_mask]
                predictions[level_mask] = self.level_models[level].predict(X_level)
            elif np.any(level_mask):
                predictions[level_mask] = 0.0
        
        return predictions
