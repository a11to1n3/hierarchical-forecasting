"""
Data preprocessing utilities for hierarchical forecasting.

This module handles data cleaning, feature engineering, and preparation
for the hierarchical forecasting model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")


class DataPreprocessor:
    """
    Handles preprocessing of hierarchical sales data.
    """
    
    def __init__(self, scaling_method: str = 'minmax'):
        """
        Initialize the preprocessor.
        
        Args:
            scaling_method: Scaling method ('minmax', 'standard', or 'none')
        """
        self.scaling_method = scaling_method
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_columns = None
        
        if scaling_method == 'minmax':
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        elif scaling_method == 'standard':
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare raw data for modeling.
        
        Args:
            df: Raw DataFrame with hierarchical sales data
            
        Returns:
            Processed DataFrame ready for modeling
        """
        print("🔬 Preparing data for modeling...")
        
        # Ensure required columns exist
        required_columns = ['companyID', 'storeID', 'skuID', 'target']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create feature columns (exclude target and ID columns)
        self.feature_columns = [
            col for col in df.columns 
            if col not in ['target', 'companyID', 'storeID', 'skuID']
        ]
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Create additional features
        df = self._create_features(df)
        
        # Scale features and targets
        if self.scaling_method != 'none':
            df = self._scale_data(df)
        
        print(f"✅ Data preparation complete.")
        print(f"  - Shape: {df.shape}")
        print(f"  - Feature columns: {len(self.feature_columns)}")
        print(f"  - Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Convert all feature columns to numeric, coercing errors to NaN
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure target is numeric
        df['target'] = pd.to_numeric(df['target'], errors='coerce')
        
        # Forward fill missing values for time series
        df = df.fillna(method='ffill')
        
        # Backward fill remaining missing values
        df = df.fillna(method='bfill')
        
        # Fill any remaining NaN with 0
        df = df.fillna(0)
        
        return df
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for the model."""
        df = df.copy()
        
        # Handle date features - check if date is in index or columns
        date_col = None
        if isinstance(df.index, pd.DatetimeIndex):
            date_col = df.index
        elif 'date' in df.columns:
            date_col = pd.to_datetime(df['date'])
        
        # Time-based features
        if date_col is not None:
            if isinstance(date_col, pd.DatetimeIndex):
                # For DatetimeIndex, use direct attributes
                df['day_of_week'] = date_col.dayofweek
                df['month'] = date_col.month
                df['quarter'] = date_col.quarter
                df['day_of_year'] = date_col.dayofyear
            else:
                # For Series, use .dt accessor
                df['day_of_week'] = date_col.dt.dayofweek
                df['month'] = date_col.dt.month
                df['quarter'] = date_col.dt.quarter
                df['day_of_year'] = date_col.dt.dayofyear
            
            # Cyclical encoding
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Simple lag features (ensure target exists)
        if 'target' in df.columns:
            df_sorted = df.sort_values('date') if 'date' in df.columns else df.sort_index()
            for lag in [1, 7]:
                df[f'target_lag_{lag}'] = df_sorted['target'].shift(lag)
            
            # Fill NaN lag values with 0
            lag_cols = [col for col in df.columns if 'target_lag_' in col]
            for col in lag_cols:
                df[col] = df[col].fillna(0)
        
        return df
    
    def _scale_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features and targets."""
        df = df.copy()
        
        # Scale features
        if self.feature_columns:
            features = df[self.feature_columns].astype(np.float32)
            features_scaled = self.feature_scaler.fit_transform(features)
            df[self.feature_columns] = features_scaled
        
        # Scale targets
        target = df[['target']].astype(np.float32)
        target_scaled = self.target_scaler.fit_transform(target)
        df['target'] = target_scaled.flatten()
        
        return df
    
    def inverse_transform_target(self, target_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled targets back to original scale.
        
        Args:
            target_scaled: Scaled target values
            
        Returns:
            Targets in original scale
        """
        if self.target_scaler is None:
            return target_scaled
        
        if target_scaled.ndim == 1:
            target_scaled = target_scaled.reshape(-1, 1)
        
        return self.target_scaler.inverse_transform(target_scaled)
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Get dataset statistics."""
        stats = {
            'shape': df.shape,
            'num_companies': df['companyID'].nunique(),
            'num_stores': df['storeID'].nunique(),
            'num_skus': df['skuID'].nunique(),
            'date_range': (df.index.min(), df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None,
            'target_stats': df['target'].describe().to_dict(),
            'feature_columns': self.feature_columns,
            'missing_values': df.isnull().sum().sum()
        }
        return stats
    
    def create_hierarchy(self, df: pd.DataFrame) -> Dict:
        """
        Create hierarchy dictionary for baseline models.
        
        Args:
            df: DataFrame with hierarchical sales data
            
        Returns:
            Dictionary mapping hierarchy levels to entity groups
        """
        hierarchy = {}
        
        # Reset index to access date column if it's the index
        if isinstance(df.index, pd.DatetimeIndex):
            df_reset = df.reset_index()
        else:
            df_reset = df.copy()
        
        # Level 0: Individual SKUs (companyID, storeID, skuID)
        hierarchy[0] = list(df_reset[['companyID', 'storeID', 'skuID']].drop_duplicates().apply(tuple, axis=1))
        
        # Level 1: Stores (companyID, storeID)
        hierarchy[1] = list(df_reset[['companyID', 'storeID']].drop_duplicates().apply(tuple, axis=1))
        
        # Level 2: Companies (companyID,)
        hierarchy[2] = list(df_reset[['companyID']].drop_duplicates().apply(tuple, axis=1))
        
        # Level 3: Total (single aggregate)
        hierarchy[3] = [('total',)]
        
        return hierarchy

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Public method to create features from the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with created features and proper entity_id
        """
        # Make a copy to avoid modifying original
        df_copy = df.copy()
        
        # Reset index if it's a DatetimeIndex to access date as column
        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy = df_copy.reset_index()
        
        # Create entity_id column for baseline comparison
        if all(col in df_copy.columns for col in ['companyID', 'storeID', 'skuID']):
            # Create entity_id as tuple of (companyID, storeID, skuID)
            df_copy['entity_id'] = list(df_copy[['companyID', 'storeID', 'skuID']].apply(tuple, axis=1))
        
        # Apply feature engineering
        df_with_features = self._create_features(df_copy)
        
        # Ensure only numeric features are included (exclude date, IDs, etc.)
        exclude_cols = ['date', 'companyID', 'storeID', 'skuID', 'target', 'entity_id']
        numeric_feature_cols = []
        
        for col in df_with_features.columns:
            if col not in exclude_cols:
                # Try to convert to numeric
                try:
                    df_with_features[col] = pd.to_numeric(df_with_features[col], errors='coerce')
                    # If conversion was successful and has no NaN, include it
                    if not df_with_features[col].isna().all():
                        numeric_feature_cols.append(col)
                except:
                    continue
        
        # Update feature columns to only include numeric features
        self.feature_columns = numeric_feature_cols
        
        # Fill any NaN values in features with 0
        for col in self.feature_columns:
            df_with_features[col] = df_with_features[col].fillna(0)
        
        return df_with_features

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Public method to create targets from the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with target column
        """
        # Simply return the target column as a DataFrame
        if 'target' not in df.columns:
            raise ValueError("Target column not found in DataFrame")
        return df[['target']]

    def split_data(self, df: pd.DataFrame, test_period: int = 30, val_period: int = 30):
        """
        Split data into train, validation, and test sets chronologically.
        
        Args:
            df: Input DataFrame with date index
            test_period: Number of days for test set
            val_period: Number of days for validation set
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Get unique dates and sort them
        unique_dates = sorted(df.index.unique())
        total_days = len(unique_dates)
        
        if test_period + val_period >= total_days:
            raise ValueError("Test and validation periods are too large for the dataset")
        
        # Split chronologically
        train_end_idx = total_days - test_period - val_period
        val_end_idx = total_days - test_period
        
        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[train_end_idx:val_end_idx]
        test_dates = unique_dates[val_end_idx:]
        
        # Split dataframes
        train_data = df[df.index.isin(train_dates)]
        val_data = df[df.index.isin(val_dates)]
        test_data = df[df.index.isin(test_dates)]
        
        print(f"📊 Data split:")
        print(f"  - Training: {len(train_data)} samples ({len(train_dates)} days)")
        print(f"  - Validation: {len(val_data)} samples ({len(val_dates)} days)")
        print(f"  - Test: {len(test_data)} samples ({len(test_dates)} days)")
        
        return train_data, val_data, test_data


def load_and_merge_data(features_path: str, targets_path: str) -> pd.DataFrame:
    """
    Load and merge bakery features and targets data.
    
    Args:
        features_path: Path to features parquet file
        targets_path: Path to targets parquet file
        
    Returns:
        Merged DataFrame with features and targets
    """
    print("📥 Loading bakery data...")
    
    try:
        # Load data
        features_df = pd.read_parquet(features_path)
        targets_df = pd.read_parquet(targets_path)
        
        print(f"Features shape: {features_df.shape}")
        print(f"Targets shape: {targets_df.shape}")
        
        # Merge on common columns
        common_cols = ['date', 'companyID', 'storeID', 'skuID']
        
        # Ensure date column is properly formatted
        for df in [features_df, targets_df]:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        
        # Merge features and targets
        merged_df = pd.merge(
            features_df, targets_df,
            on=common_cols,
            how='inner'
        )
        
        # Set date as index
        merged_df = merged_df.set_index('date').sort_index()
        
        print(f"✅ Data merged successfully. Final shape: {merged_df.shape}")
        
        return merged_df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise


def create_dummy_data(n_companies: int = 1, n_stores: int = 10, 
                     n_skus: int = 50, n_days: int = 180) -> pd.DataFrame:
    """
    Create dummy sales data for testing.
    
    Args:
        n_companies: Number of companies
        n_stores: Number of stores per company
        n_skus: Number of SKUs per store
        n_days: Number of days
        
    Returns:
        Dummy DataFrame with sales data
    """
    print("🎭 Creating dummy data for testing...")
    
    dates = pd.date_range(start='2022-01-01', periods=n_days)
    data = []
    
    for company_id in range(n_companies):
        company = f'Company{company_id}'
        
        for store_id in range(n_stores):
            store = f'Store{store_id}'
            
            for sku_id in range(n_skus):
                sku = f'SKU{sku_id}'
                
                # Create realistic sales pattern
                base_sales = 100 + (hash(sku) % 50) + (hash(store) % 20)
                seasonal_component = 30 * np.sin(np.arange(n_days) * 2 * np.pi / 7)
                trend_component = np.arange(n_days) * 0.1
                noise = np.random.normal(0, 15, n_days)
                
                sales = base_sales + seasonal_component + trend_component + noise
                sales[sales < 0] = 0
                
                for i, date in enumerate(dates):
                    data.append([
                        date, company, store, sku, sales[i],
                        # Add some dummy features
                        np.random.uniform(0, 100),  # feature1
                        np.random.uniform(10, 50),  # feature2
                        np.random.choice([0, 1])    # feature3
                    ])
    
    df = pd.DataFrame(data, columns=[
        'date', 'companyID', 'storeID', 'skuID', 'target',
        'feature1', 'feature2', 'feature3'
    ]).set_index('date')
    
    print(f"✅ Dummy data created. Shape: {df.shape}")
    return df
