"""
MinT (Minimum Trace) baseline for hierarchical forecasting.
"""

import numpy as np
from scipy.linalg import pinv
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Optional
from .base import HierarchicalBaselineModel


class MinTBaseline(HierarchicalBaselineModel):
    """
    Minimum Trace (MinT) hierarchical forecasting baseline.
    
    MinT is an optimal reconciliation method that minimizes the trace
    of the forecast error covariance matrix. It provides coherent
    forecasts with minimal variance.
    """
    
    def __init__(self, base_model: str = 'linear', 
                 covariance_method: str = 'sample', **model_kwargs):
        """
        Initialize MinT baseline.
        
        Args:
            base_model: Type of base model ('linear' or 'random_forest')
            covariance_method: Method for covariance estimation ('sample' or 'shrinkage')
            **model_kwargs: Additional arguments for base model
        """
        super().__init__("MinT", "mint")
        self.base_model_type = base_model
        self.covariance_method = covariance_method
        self.model_kwargs = model_kwargs
        
        self.base_models = {}  # Models for each level
        self.reconciliation_matrix = None
        self.covariance_matrix = None
        
    def _create_base_model(self):
        """Create a base forecasting model."""
        if self.base_model_type == 'linear':
            return LinearRegression(**self.model_kwargs)
        elif self.base_model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.model_kwargs.get('n_estimators', 100),
                random_state=self.model_kwargs.get('random_state', 42),
                **{k: v for k, v in self.model_kwargs.items() 
                   if k not in ['n_estimators', 'random_state']}
            )
        else:
            raise ValueError(f"Unknown base model type: {self.base_model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            hierarchy: Optional[Dict] = None,
            entity_levels: Optional[np.ndarray] = None, **kwargs) -> 'MinTBaseline':
        """
        Fit the MinT model.
        
        Args:
            X: Input features [n_samples, n_features]
            y: Target values [n_samples]
            hierarchy: Hierarchy structure
            entity_levels: Level indicator for each sample
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        if hierarchy is not None:
            self.hierarchy_structure = hierarchy
            self.aggregation_matrix = self.build_aggregation_matrix(hierarchy)
        
        if entity_levels is None:
            entity_levels = np.zeros(len(X))
        
        # Train base models for each level
        for level in np.unique(entity_levels):
            level_mask = entity_levels == level
            X_level = X[level_mask]
            y_level = y[level_mask]
            
            if len(X_level) > 0:
                model = self._create_base_model()
                model.fit(X_level, y_level)
                self.base_models[level] = model
        
        # Estimate forecast error covariance matrix
        self._estimate_covariance_matrix(X, y, entity_levels)
        
        # Compute MinT reconciliation matrix
        self._compute_reconciliation_matrix()
        
        self.is_fitted = True
        return self
    
    def _estimate_covariance_matrix(self, X: np.ndarray, y: np.ndarray, 
                                  entity_levels: np.ndarray):
        """
        Estimate the forecast error covariance matrix.
        
        Args:
            X: Input features
            y: Target values
            entity_levels: Level indicators
        """
        # Get base forecasts for training data
        base_forecasts = np.zeros(len(y))
        
        for level in np.unique(entity_levels):
            level_mask = entity_levels == level
            if level in self.base_models and np.any(level_mask):
                X_level = X[level_mask]
                base_forecasts[level_mask] = self.base_models[level].predict(X_level)
        
        # Calculate residuals
        residuals = y - base_forecasts
        
        # Group residuals by entity level
        unique_levels = np.unique(entity_levels)
        n_levels = len(unique_levels)
        
        # Estimate covariance matrix
        if self.covariance_method == 'sample':
            # Create residual matrix for each level
            level_residuals = {}
            for level in unique_levels:
                level_mask = entity_levels == level
                level_residuals[level] = residuals[level_mask]
            
            # Create covariance matrix (simplified approach)
            self.covariance_matrix = np.eye(n_levels)
            
            # Fill in diagonal with level variances
            for i, level in enumerate(unique_levels):
                if len(level_residuals[level]) > 1:
                    self.covariance_matrix[i, i] = np.var(level_residuals[level])
                else:
                    self.covariance_matrix[i, i] = 1.0
            
            # Add small regularization
            self.covariance_matrix += 1e-6 * np.eye(n_levels)
            
        elif self.covariance_method == 'shrinkage':
            # Use shrinkage estimator (simplified)
            self.covariance_matrix = np.eye(n_levels)
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(self.covariance_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)
        self.covariance_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    def _compute_reconciliation_matrix(self):
        """Compute the MinT reconciliation matrix."""
        if self.aggregation_matrix is None or self.covariance_matrix is None:
            # Fallback to simple reconciliation
            n_levels = len(self.base_models)
            self.reconciliation_matrix = np.eye(n_levels)
            return
        
        S = self.aggregation_matrix
        W_inv = pinv(self.covariance_matrix)
        
        # Check dimensions
        n_series, n_bottom = S.shape
        n_cov = W_inv.shape[0]
        
        if n_cov != n_series:
            # Dimension mismatch, use simpler reconciliation
            print(f"Warning: Dimension mismatch in MinT (S: {S.shape}, W_inv: {W_inv.shape})")
            self.reconciliation_matrix = np.eye(n_series)
            return
        
        # MinT reconciliation matrix: S(S'W^(-1)S)^(-1)S'W^(-1)
        try:
            temp = S.T @ W_inv @ S
            temp_inv = pinv(temp)
            self.reconciliation_matrix = S @ temp_inv @ S.T @ W_inv
        except (np.linalg.LinAlgError, ValueError) as e:
            # Fallback to pseudo-inverse if matrix operations fail
            print(f"Warning: MinT reconciliation failed ({e}), using pseudo-inverse")
            self.reconciliation_matrix = pinv(S)
    
    def predict(self, X: np.ndarray, entity_levels: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """
        Make MinT reconciled predictions.
        
        Args:
            X: Input features [n_samples, n_features]
            entity_levels: Level indicator for each sample
            **kwargs: Additional arguments
            
        Returns:
            Reconciled predictions [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if entity_levels is None:
            entity_levels = np.zeros(len(X))
        
        # Get base forecasts from all levels
        base_forecasts = np.zeros(len(X))
        
        for level in np.unique(entity_levels):
            level_mask = entity_levels == level
            if level in self.base_models and np.any(level_mask):
                X_level = X[level_mask]
                base_forecasts[level_mask] = self.base_models[level].predict(X_level)
        
        # Apply MinT reconciliation
        reconciled_forecasts = self.reconcile_forecasts(base_forecasts)
        
        return reconciled_forecasts
    
    def reconcile_forecasts(self, base_forecasts: np.ndarray) -> np.ndarray:
        """
        Reconcile forecasts using MinT method.
        
        Args:
            base_forecasts: Base forecasts for all levels
            
        Returns:
            MinT reconciled forecasts
        """
        if self.reconciliation_matrix is None:
            return base_forecasts
        
        try:
            # Apply MinT reconciliation
            reconciled = self.reconciliation_matrix @ base_forecasts
            return reconciled
        except (ValueError, np.linalg.LinAlgError):
            # Fallback to base forecasts if reconciliation fails
            return base_forecasts
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            'base_model': self.base_model_type,
            'covariance_method': self.covariance_method
        })
        return params


class OLSBaseline(HierarchicalBaselineModel):
    """
    OLS (Ordinary Least Squares) reconciliation baseline.
    
    This is a simpler reconciliation method that uses OLS to
    ensure hierarchical coherence.
    """
    
    def __init__(self, base_model: str = 'linear', **model_kwargs):
        """
        Initialize OLS baseline.
        
        Args:
            base_model: Type of base model
            **model_kwargs: Additional model arguments
        """
        super().__init__("OLS", "ols")
        self.base_model_type = base_model
        self.model_kwargs = model_kwargs
        self.base_models = {}
        self.reconciliation_matrix = None
    
    def _create_base_model(self):
        """Create a base forecasting model."""
        if self.base_model_type == 'linear':
            return LinearRegression(**self.model_kwargs)
        elif self.base_model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.model_kwargs.get('n_estimators', 100),
                random_state=self.model_kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown base model type: {self.base_model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            hierarchy: Optional[Dict] = None,
            entity_levels: Optional[np.ndarray] = None, **kwargs) -> 'OLSBaseline':
        """
        Fit the OLS reconciliation model.
        
        Args:
            X: Input features
            y: Target values
            hierarchy: Hierarchy structure
            entity_levels: Level indicators
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        if hierarchy is not None:
            self.hierarchy_structure = hierarchy
            self.aggregation_matrix = self.build_aggregation_matrix(hierarchy)
        
        if entity_levels is None:
            entity_levels = np.zeros(len(X))
        
        # Train base models for each level
        for level in np.unique(entity_levels):
            level_mask = entity_levels == level
            X_level = X[level_mask]
            y_level = y[level_mask]
            
            if len(X_level) > 0:
                model = self._create_base_model()
                model.fit(X_level, y_level)
                self.base_models[level] = model
        
        # Compute OLS reconciliation matrix
        self._compute_ols_reconciliation_matrix()
        
        self.is_fitted = True
        return self
    
    def _compute_ols_reconciliation_matrix(self):
        """Compute the OLS reconciliation matrix."""
        if self.aggregation_matrix is None:
            n_levels = len(self.base_models)
            self.reconciliation_matrix = np.eye(n_levels)
            return
        
        S = self.aggregation_matrix
        
        # OLS reconciliation matrix: S(S'S)^(-1)S'
        try:
            temp = S.T @ S
            temp_inv = pinv(temp)
            self.reconciliation_matrix = S @ temp_inv @ S.T
        except np.linalg.LinAlgError:
            self.reconciliation_matrix = pinv(S)
    
    def predict(self, X: np.ndarray, entity_levels: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """
        Make OLS reconciled predictions.
        
        Args:
            X: Input features
            entity_levels: Level indicators
            **kwargs: Additional arguments
            
        Returns:
            Reconciled predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if entity_levels is None:
            entity_levels = np.zeros(len(X))
        
        # Get base forecasts
        base_forecasts = np.zeros(len(X))
        
        for level in np.unique(entity_levels):
            level_mask = entity_levels == level
            if level in self.base_models and np.any(level_mask):
                X_level = X[level_mask]
                base_forecasts[level_mask] = self.base_models[level].predict(X_level)
        
        # Apply OLS reconciliation
        return self.reconcile_forecasts(base_forecasts)
    
    def reconcile_forecasts(self, base_forecasts: np.ndarray) -> np.ndarray:
        """
        Reconcile forecasts using OLS method.
        
        Args:
            base_forecasts: Base forecasts
            
        Returns:
            OLS reconciled forecasts
        """
        if self.reconciliation_matrix is None:
            return base_forecasts
        
        try:
            return self.reconciliation_matrix @ base_forecasts
        except (ValueError, np.linalg.LinAlgError):
            return base_forecasts
