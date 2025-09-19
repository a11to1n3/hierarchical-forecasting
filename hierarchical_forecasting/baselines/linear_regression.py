"""
Linear Regression baseline for hierarchical forecasting.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional
from .base import BaselineModel


class LinearRegressionBaseline(BaselineModel):
    """
    Linear regression baseline with optional regularization.
    
    This serves as a simple baseline that learns linear relationships
    between features and targets without considering hierarchy.
    """
    
    def __init__(self, regularization: str = 'none', alpha: float = 1.0, 
                 normalize: bool = True):
        """
        Initialize linear regression baseline.
        
        Args:
            regularization: Type of regularization ('none', 'ridge', 'lasso')
            alpha: Regularization strength
            normalize: Whether to normalize features
        """
        super().__init__(f"LinearRegression_{regularization}")
        self.regularization = regularization
        self.alpha = alpha
        self.normalize = normalize
        
        # Initialize model based on regularization
        if regularization == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            self.model = Lasso(alpha=alpha)
        else:
            self.model = LinearRegression()
        
        self.scaler = StandardScaler() if normalize else None
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            hierarchy: Optional[Dict] = None, **kwargs) -> 'LinearRegressionBaseline':
        """
        Fit the linear regression model.
        
        Args:
            X: Input features [n_samples, n_features]
            y: Target values [n_samples]
            hierarchy: Ignored for this baseline
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
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
        Make predictions using the fitted linear model.
        
        Args:
            X: Input features [n_samples, n_features]
            **kwargs: Additional arguments
            
        Returns:
            Predictions [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Apply same scaling as training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (coefficients)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        else:
            return np.array([])
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            'regularization': self.regularization,
            'alpha': self.alpha,
            'normalize': self.normalize
        })
        return params


class MultiLevelLinearRegression(BaselineModel):
    """
    Linear regression with separate models for each hierarchy level.
    
    This baseline trains separate linear models for each level of the
    hierarchy, allowing for level-specific feature relationships.
    """
    
    def __init__(self, regularization: str = 'none', alpha: float = 1.0):
        """
        Initialize multi-level linear regression.
        
        Args:
            regularization: Type of regularization
            alpha: Regularization strength
        """
        super().__init__(f"MultiLevel_LinearRegression_{regularization}")
        self.regularization = regularization
        self.alpha = alpha
        self.level_models = {}
        self.level_scalers = {}
        
    def _create_model(self):
        """Create a new model instance."""
        if self.regularization == 'ridge':
            return Ridge(alpha=self.alpha)
        elif self.regularization == 'lasso':
            return Lasso(alpha=self.alpha)
        else:
            return LinearRegression()
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            hierarchy: Optional[Dict] = None, 
            entity_levels: Optional[np.ndarray] = None, **kwargs) -> 'MultiLevelLinearRegression':
        """
        Fit separate models for each hierarchy level.
        
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
            # If no level info, treat as single level
            entity_levels = np.zeros(len(X))
        
        # Get unique levels
        unique_levels = np.unique(entity_levels)
        
        # Train separate model for each level
        for level in unique_levels:
            level_mask = entity_levels == level
            X_level = X[level_mask]
            y_level = y[level_mask]
            
            if len(X_level) > 0:
                # Create and fit model for this level
                model = self._create_model()
                scaler = StandardScaler()
                
                X_scaled = scaler.fit_transform(X_level)
                model.fit(X_scaled, y_level)
                
                self.level_models[level] = model
                self.level_scalers[level] = scaler
        
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
                X_scaled = self.level_scalers[level].transform(X_level)
                predictions[level_mask] = self.level_models[level].predict(X_scaled)
            elif np.any(level_mask):
                # Use average prediction if no model for this level
                predictions[level_mask] = 0.0
        
        return predictions
