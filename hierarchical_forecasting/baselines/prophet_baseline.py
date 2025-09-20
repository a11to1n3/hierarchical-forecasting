"""
Prophet baseline for hierarchical forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from .base import BaselineModel

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ProphetBaseline(BaselineModel):
    """
    Prophet baseline for hierarchical forecasting.
    
    This baseline uses Facebook's Prophet model for time series forecasting.
    Prophet is designed to handle seasonality, trends, and holidays automatically.
    """
    
    def __init__(self, seasonality_mode: str = 'additive', 
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 interval_width: float = 0.8):
        """
        Initialize Prophet baseline.
        
        Args:
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
            interval_width: Width of uncertainty intervals
        """
        super().__init__("Prophet")
        
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Install with: pip install prophet")
        
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.interval_width = interval_width
        
        self.models = {}  # Store models for different entities
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            hierarchy: Optional[Dict] = None,
            dates: Optional[np.ndarray] = None,
            entity_ids: Optional[np.ndarray] = None, **kwargs) -> 'ProphetBaseline':
        """
        Fit Prophet models for different entities.
        
        Args:
            X: Input features [n_samples, n_features] (not used by Prophet)
            y: Target values [n_samples]
            hierarchy: Hierarchy structure
            dates: Date column for time series
            entity_ids: Entity identifiers for each sample
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        if dates is None:
            raise ValueError("Prophet requires dates for time series modeling")
        
        if entity_ids is None:
            # Single time series
            entity_ids = np.zeros(len(y))
        
        # Train separate Prophet model for each entity
        entity_ids = np.asarray(entity_ids)
        unique_entities = np.unique(entity_ids)
        
        for entity in unique_entities:
            entity_mask = entity_ids == entity
            entity_dates = pd.to_datetime(dates[entity_mask])
            entity_y = y[entity_mask]
            
            df = pd.DataFrame({
                'ds': entity_dates,
                'y': entity_y
            }).sort_values('ds')

            df = df.groupby('ds', as_index=False)['y'].mean()

            if len(df) >= 2:  # Prophet needs at least 2 data points
                # Prepare data in Prophet format
                # Create and fit Prophet model
                model = Prophet(
                    seasonality_mode=self.seasonality_mode,
                    yearly_seasonality=self.yearly_seasonality,
                    weekly_seasonality=self.weekly_seasonality,
                    daily_seasonality=self.daily_seasonality,
                    interval_width=self.interval_width
                )
                
                # Suppress Prophet's verbose output
                import logging
                logging.getLogger('prophet').setLevel(logging.WARNING)
                
                model.fit(df)
                self.models[entity] = model
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, 
                dates: Optional[np.ndarray] = None,
                entity_ids: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Make predictions using fitted Prophet models.
        
        Args:
            X: Input features (not used)
            dates: Future dates for prediction
            entity_ids: Entity identifiers
            **kwargs: Additional arguments
            
        Returns:
            Predictions [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if dates is None:
            raise ValueError("Prophet requires dates for prediction")
        
        if entity_ids is None:
            entity_ids = np.zeros(len(dates))
        
        predictions = np.zeros(len(dates))
        
        # Make predictions for each entity
        unique_entities = np.unique(entity_ids)
        
        for entity in unique_entities:
            entity_mask = entity_ids == entity
            
            if entity in self.models:
                entity_dates = dates[entity_mask]
                
                # Prepare future dataframe
                future_df = pd.DataFrame({
                    'ds': pd.to_datetime(entity_dates)
                })
                
                # Make predictions
                forecast = self.models[entity].predict(future_df)
                predictions[entity_mask] = forecast['yhat'].values
            else:
                # If no model for this entity, predict zero
                predictions[entity_mask] = 0.0
        
        return predictions
    
    def predict_with_uncertainty(self, X: np.ndarray,
                               dates: Optional[np.ndarray] = None,
                               entity_ids: Optional[np.ndarray] = None, **kwargs) -> tuple:
        """
        Make predictions with uncertainty bounds.
        
        Args:
            X: Input features
            dates: Future dates
            entity_ids: Entity identifiers
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if dates is None:
            raise ValueError("Prophet requires dates for prediction")
        
        if entity_ids is None:
            entity_ids = np.zeros(len(dates))
        
        predictions = np.zeros(len(dates))
        lower_bounds = np.zeros(len(dates))
        upper_bounds = np.zeros(len(dates))
        
        unique_entities = np.unique(entity_ids)
        
        for entity in unique_entities:
            entity_mask = entity_ids == entity
            
            if entity in self.models:
                entity_dates = dates[entity_mask]
                
                future_df = pd.DataFrame({
                    'ds': pd.to_datetime(entity_dates)
                })
                
                forecast = self.models[entity].predict(future_df)
                predictions[entity_mask] = forecast['yhat'].values
                lower_bounds[entity_mask] = forecast['yhat_lower'].values
                upper_bounds[entity_mask] = forecast['yhat_upper'].values
        
        return predictions, lower_bounds, upper_bounds
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            'seasonality_mode': self.seasonality_mode,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'interval_width': self.interval_width
        })
        return params


class HierarchicalProphet(BaselineModel):
    """
    Hierarchical Prophet with reconciliation.
    
    This baseline uses Prophet for forecasting at different hierarchy levels
    and then reconciles the forecasts for coherence.
    """
    
    def __init__(self, reconciliation_method: str = 'bottom_up', **prophet_kwargs):
        """
        Initialize Hierarchical Prophet.
        
        Args:
            reconciliation_method: Method for reconciliation
            **prophet_kwargs: Arguments for Prophet models
        """
        super().__init__("HierarchicalProphet")
        
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Install with: pip install prophet")
        
        self.reconciliation_method = reconciliation_method
        self.prophet_kwargs = prophet_kwargs
        self.level_models = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            hierarchy: Optional[Dict] = None,
            dates: Optional[np.ndarray] = None,
            entity_levels: Optional[np.ndarray] = None, **kwargs) -> 'HierarchicalProphet':
        """
        Fit Prophet models for each hierarchy level.
        
        Args:
            X: Input features
            y: Target values
            hierarchy: Hierarchy structure
            dates: Date information
            entity_levels: Level indicators
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        if dates is None:
            raise ValueError("Prophet requires dates for time series modeling")
        
        if entity_levels is None:
            entity_levels = np.zeros(len(y))
        
        # Train Prophet model for each level
        for level in np.unique(entity_levels):
            level_mask = entity_levels == level
            level_dates = dates[level_mask]
            level_y = y[level_mask]
            
            if len(level_y) >= 2:
                prophet_model = ProphetBaseline(**self.prophet_kwargs)
                prophet_model.fit(
                    X[level_mask], level_y, 
                    dates=level_dates,
                    entity_ids=np.zeros(len(level_y))  # Single series per level
                )
                self.level_models[level] = prophet_model
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray,
                dates: Optional[np.ndarray] = None,
                entity_levels: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Make hierarchical predictions with Prophet.
        
        Args:
            X: Input features
            dates: Future dates
            entity_levels: Level indicators
            **kwargs: Additional arguments
            
        Returns:
            Reconciled predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if dates is None or entity_levels is None:
            raise ValueError("Dates and entity levels required for prediction")
        
        predictions = np.zeros(len(dates))
        
        # Get predictions from each level
        for level in np.unique(entity_levels):
            level_mask = entity_levels == level
            
            if level in self.level_models:
                level_dates = dates[level_mask]
                level_predictions = self.level_models[level].predict(
                    X[level_mask], 
                    dates=level_dates,
                    entity_ids=np.zeros(len(level_dates))
                )
                predictions[level_mask] = level_predictions
        
        # Apply reconciliation (simplified)
        if self.reconciliation_method == 'bottom_up':
            # Sum bottom level to higher levels
            bottom_mask = entity_levels == 0
            if np.any(bottom_mask):
                bottom_sum = np.sum(predictions[bottom_mask])
                for level in [1, 2, 3]:
                    level_mask = entity_levels == level
                    if np.any(level_mask):
                        predictions[level_mask] = bottom_sum / np.sum(level_mask)
        
        return predictions
