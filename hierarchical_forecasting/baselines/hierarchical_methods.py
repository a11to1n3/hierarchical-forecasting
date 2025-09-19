"""
Traditional hierarchical forecasting baselines: Bottom-Up, Top-Down, and Middle-Out.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Optional, List
from .base import HierarchicalBaselineModel


class BottomUpBaseline(HierarchicalBaselineModel):
    """
    Bottom-Up hierarchical forecasting baseline.
    
    This approach forecasts at the bottom level (SKUs) and aggregates
    upwards to ensure hierarchical coherence.
    """
    
    def __init__(self, base_model: str = 'linear', **model_kwargs):
        """
        Initialize Bottom-Up baseline.
        
        Args:
            base_model: Type of base model ('linear' or 'random_forest')
            **model_kwargs: Additional arguments for base model
        """
        super().__init__("BottomUp", "bottom_up")
        self.base_model_type = base_model
        self.model_kwargs = model_kwargs
        self.base_model = None
        
    def _create_base_model(self):
        """Create the base forecasting model."""
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
            entity_levels: Optional[np.ndarray] = None, **kwargs) -> 'BottomUpBaseline':
        """
        Fit the bottom-up model.
        
        Args:
            X: Input features [n_samples, n_features]
            y: Target values [n_samples]
            hierarchy: Hierarchy structure
            entity_levels: Level indicator (0=bottom, 1,2,3=higher levels)
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        if hierarchy is not None:
            self.hierarchy_structure = hierarchy
            self.aggregation_matrix = self.build_aggregation_matrix(hierarchy)
        
        # Train model only on bottom level (SKU level)
        if entity_levels is not None:
            bottom_mask = entity_levels == 0  # Assume 0 is bottom level
            X_bottom = X[bottom_mask]
            y_bottom = y[bottom_mask]
        else:
            # If no level info, use all data
            X_bottom = X
            y_bottom = y
        
        # Create and fit base model
        self.base_model = self._create_base_model()
        self.base_model.fit(X_bottom, y_bottom)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, entity_levels: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """
        Make bottom-up predictions.
        
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
            # If no level info, predict directly
            return self.base_model.predict(X)
        
        predictions = np.zeros(len(X))
        
        # Get bottom level predictions
        bottom_mask = entity_levels == 0
        if np.any(bottom_mask):
            bottom_predictions = self.base_model.predict(X[bottom_mask])
            predictions[bottom_mask] = bottom_predictions
        
        # Aggregate to higher levels using simple summation
        if self.hierarchy_structure is not None:
            predictions = self._aggregate_bottom_up(predictions, entity_levels)
        
        return predictions
    
    def _aggregate_bottom_up(self, bottom_predictions: np.ndarray, 
                           entity_levels: np.ndarray) -> np.ndarray:
        """
        Aggregate bottom-level predictions to higher levels.
        
        Args:
            bottom_predictions: Predictions at bottom level
            entity_levels: Level indicators
            
        Returns:
            All level predictions
        """
        all_predictions = bottom_predictions.copy()
        
        # Simple aggregation: sum bottom level for each higher level
        for level in [1, 2, 3]:  # Store, Company, Total levels
            level_mask = entity_levels == level
            if np.any(level_mask):
                # Sum all bottom level predictions for this higher level
                # This is a simplified aggregation
                bottom_sum = np.sum(bottom_predictions[entity_levels == 0])
                all_predictions[level_mask] = bottom_sum / np.sum(level_mask)
        
        return all_predictions
    
    def reconcile_forecasts(self, base_forecasts: np.ndarray) -> np.ndarray:
        """
        Reconcile forecasts using bottom-up approach.
        
        Args:
            base_forecasts: Base forecasts for all levels
            
        Returns:
            Reconciled forecasts
        """
        # Bottom-up: keep bottom level, aggregate to upper levels
        if self.aggregation_matrix is not None:
            bottom_forecasts = base_forecasts[:self.aggregation_matrix.shape[1]]
            return self.aggregation_matrix @ bottom_forecasts
        else:
            return base_forecasts


class TopDownBaseline(HierarchicalBaselineModel):
    """
    Top-Down hierarchical forecasting baseline.
    
    This approach forecasts at the top level and disaggregates
    downwards using historical proportions.
    """
    
    def __init__(self, base_model: str = 'linear', disaggregation_method: str = 'proportions',
                 **model_kwargs):
        """
        Initialize Top-Down baseline.
        
        Args:
            base_model: Type of base model
            disaggregation_method: Method for disaggregation ('proportions' or 'averages')
            **model_kwargs: Additional arguments for base model
        """
        super().__init__("TopDown", "top_down")
        self.base_model_type = base_model
        self.disaggregation_method = disaggregation_method
        self.model_kwargs = model_kwargs
        self.base_model = None
        self.historical_proportions = {}
    
    def _create_base_model(self):
        """Create the base forecasting model."""
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
            entity_levels: Optional[np.ndarray] = None, **kwargs) -> 'TopDownBaseline':
        """
        Fit the top-down model.
        
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
        
        # Train model on top level (total level)
        if entity_levels is not None:
            top_mask = entity_levels == 3  # Assume 3 is top level
            if np.any(top_mask):
                X_top = X[top_mask]
                y_top = y[top_mask]
            else:
                # If no top level, use aggregated data
                X_top = X
                y_top = y
        else:
            X_top = X
            y_top = y
        
        # Create and fit base model
        self.base_model = self._create_base_model()
        self.base_model.fit(X_top, y_top)
        
        # Calculate historical proportions for disaggregation
        if entity_levels is not None:
            self._calculate_proportions(y, entity_levels)
        
        self.is_fitted = True
        return self
    
    def _calculate_proportions(self, y: np.ndarray, entity_levels: np.ndarray):
        """Calculate historical proportions for disaggregation."""
        total_value = np.sum(y[entity_levels == 3])  # Top level total
        
        if total_value > 0:
            # Calculate proportions for each level
            for level in [0, 1, 2]:  # SKU, Store, Company levels
                level_mask = entity_levels == level
                if np.any(level_mask):
                    level_values = y[level_mask]
                    proportions = level_values / total_value
                    self.historical_proportions[level] = proportions
    
    def predict(self, X: np.ndarray, entity_levels: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """
        Make top-down predictions.
        
        Args:
            X: Input features
            entity_levels: Level indicators
            **kwargs: Additional arguments
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if entity_levels is None:
            return self.base_model.predict(X)
        
        predictions = np.zeros(len(X))
        
        # Get top level prediction
        top_mask = entity_levels == 3
        if np.any(top_mask):
            top_prediction = self.base_model.predict(X[top_mask])[0]
            predictions[top_mask] = top_prediction
            
            # Disaggregate to lower levels
            predictions = self._disaggregate_top_down(predictions, entity_levels, top_prediction)
        
        return predictions
    
    def _disaggregate_top_down(self, predictions: np.ndarray, 
                             entity_levels: np.ndarray, top_prediction: float) -> np.ndarray:
        """
        Disaggregate top-level prediction to lower levels.
        
        Args:
            predictions: Current predictions array
            entity_levels: Level indicators
            top_prediction: Top level prediction
            
        Returns:
            Disaggregated predictions
        """
        # Disaggregate using historical proportions
        for level in [0, 1, 2]:  # SKU, Store, Company levels
            level_mask = entity_levels == level
            if np.any(level_mask) and level in self.historical_proportions:
                proportions = self.historical_proportions[level]
                n_entities = np.sum(level_mask)
                
                if len(proportions) == n_entities:
                    predictions[level_mask] = top_prediction * proportions
                else:
                    # Equal distribution if proportions don't match
                    predictions[level_mask] = top_prediction / n_entities
        
        return predictions
    
    def reconcile_forecasts(self, base_forecasts: np.ndarray) -> np.ndarray:
        """
        Reconcile forecasts using top-down approach.
        
        Args:
            base_forecasts: Base forecasts for all levels
            
        Returns:
            Reconciled forecasts
        """
        # Top-down: use top level forecast and disaggregate
        reconciled = base_forecasts.copy()
        
        if len(self.historical_proportions) > 0:
            top_forecast = base_forecasts[-1]  # Assume last is top level
            
            # Disaggregate proportionally
            for level, proportions in self.historical_proportions.items():
                start_idx = sum(len(props) for l, props in self.historical_proportions.items() if l < level)
                end_idx = start_idx + len(proportions)
                reconciled[start_idx:end_idx] = top_forecast * proportions
        
        return reconciled


class MiddleOutBaseline(HierarchicalBaselineModel):
    """
    Middle-Out hierarchical forecasting baseline.
    
    This approach forecasts at an intermediate level and both
    aggregates up and disaggregates down.
    """
    
    def __init__(self, base_model: str = 'linear', middle_level: int = 1, **model_kwargs):
        """
        Initialize Middle-Out baseline.
        
        Args:
            base_model: Type of base model
            middle_level: Which level to use as middle (1=Store level typically)
            **model_kwargs: Additional arguments for base model
        """
        super().__init__("MiddleOut", "middle_out")
        self.base_model_type = base_model
        self.middle_level = middle_level
        self.model_kwargs = model_kwargs
        self.base_model = None
        self.up_proportions = {}
        self.down_proportions = {}
    
    def _create_base_model(self):
        """Create the base forecasting model."""
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
            entity_levels: Optional[np.ndarray] = None, **kwargs) -> 'MiddleOutBaseline':
        """
        Fit the middle-out model.
        
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
        
        # Train model on middle level
        if entity_levels is not None:
            middle_mask = entity_levels == self.middle_level
            if np.any(middle_mask):
                X_middle = X[middle_mask]
                y_middle = y[middle_mask]
            else:
                X_middle = X
                y_middle = y
        else:
            X_middle = X
            y_middle = y
        
        # Create and fit base model
        self.base_model = self._create_base_model()
        self.base_model.fit(X_middle, y_middle)
        
        # Calculate proportions for aggregation/disaggregation
        if entity_levels is not None:
            self._calculate_middle_proportions(y, entity_levels)
        
        self.is_fitted = True
        return self
    
    def _calculate_middle_proportions(self, y: np.ndarray, entity_levels: np.ndarray):
        """Calculate proportions for middle-out approach."""
        middle_values = y[entity_levels == self.middle_level]
        middle_total = np.sum(middle_values)
        
        if middle_total > 0:
            # Proportions for aggregating up
            for level in range(self.middle_level + 1, 4):  # Higher levels
                level_mask = entity_levels == level
                if np.any(level_mask):
                    level_values = y[level_mask]
                    self.up_proportions[level] = level_values / middle_total
            
            # Proportions for disaggregating down
            for level in range(0, self.middle_level):  # Lower levels
                level_mask = entity_levels == level
                if np.any(level_mask):
                    level_values = y[level_mask]
                    self.down_proportions[level] = level_values / middle_total
    
    def predict(self, X: np.ndarray, entity_levels: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """
        Make middle-out predictions.
        
        Args:
            X: Input features
            entity_levels: Level indicators
            **kwargs: Additional arguments
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if entity_levels is None:
            return self.base_model.predict(X)
        
        predictions = np.zeros(len(X))
        
        # Get middle level predictions
        middle_mask = entity_levels == self.middle_level
        if np.any(middle_mask):
            middle_predictions = self.base_model.predict(X[middle_mask])
            predictions[middle_mask] = middle_predictions
            middle_total = np.sum(middle_predictions)
            
            # Aggregate up and disaggregate down
            predictions = self._middle_out_reconcile(predictions, entity_levels, middle_total)
        
        return predictions
    
    def _middle_out_reconcile(self, predictions: np.ndarray, 
                            entity_levels: np.ndarray, middle_total: float) -> np.ndarray:
        """
        Reconcile predictions using middle-out approach.
        
        Args:
            predictions: Current predictions
            entity_levels: Level indicators
            middle_total: Total middle level prediction
            
        Returns:
            Reconciled predictions
        """
        # Aggregate up to higher levels
        for level in range(self.middle_level + 1, 4):
            level_mask = entity_levels == level
            if np.any(level_mask) and level in self.up_proportions:
                proportions = self.up_proportions[level]
                n_entities = np.sum(level_mask)
                
                if len(proportions) == n_entities:
                    predictions[level_mask] = middle_total * proportions
                else:
                    predictions[level_mask] = middle_total
        
        # Disaggregate down to lower levels
        for level in range(0, self.middle_level):
            level_mask = entity_levels == level
            if np.any(level_mask) and level in self.down_proportions:
                proportions = self.down_proportions[level]
                n_entities = np.sum(level_mask)
                
                if len(proportions) == n_entities:
                    predictions[level_mask] = middle_total * proportions
                else:
                    predictions[level_mask] = middle_total / n_entities
        
        return predictions
    
    def reconcile_forecasts(self, base_forecasts: np.ndarray) -> np.ndarray:
        """
        Reconcile forecasts using middle-out approach.
        
        Args:
            base_forecasts: Base forecasts for all levels
            
        Returns:
            Reconciled forecasts
        """
        # Use middle level forecast and reconcile up/down
        reconciled = base_forecasts.copy()
        
        # This is a simplified implementation
        # In practice, you'd use the aggregation matrices and proportions
        return reconciled
