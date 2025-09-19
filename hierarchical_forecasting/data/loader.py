"""
Data loading utilities for hierarchical forecasting.

This module provides data loaders and utilities for preparing
data for training and evaluation.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional

# Import from models module - will be resolved at runtime
try:
    from models.combinatorial_complex import CombinatorialComplex
except ImportError:
    # For relative imports
    from ..models.combinatorial_complex import CombinatorialComplex


class HierarchicalDataset(Dataset):
    """
    Dataset class for hierarchical sales data.
    """
    
    def __init__(self, daily_data: List[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Initialize the dataset.
        
        Args:
            daily_data: List of (features, targets) tuples for each day
        """
        self.daily_data = daily_data
    
    def __len__(self) -> int:
        return len(self.daily_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.daily_data[idx]


class HierarchicalDataLoader:
    """
    Data loader for hierarchical forecasting data.
    """
    
    def __init__(self, combinatorial_complex: CombinatorialComplex):
        """
        Initialize the data loader.
        
        Args:
            combinatorial_complex: The combinatorial complex structure
        """
        self.cc = combinatorial_complex
    
    def prepare_daily_tensors(self, df: pd.DataFrame, 
                            feature_columns: List[str]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare daily tensors from DataFrame.
        
        Args:
            df: DataFrame with hierarchical sales data
            feature_columns: List of feature column names
            
        Returns:
            List of (features, targets) tensors for each day
        """
        print(f"📊 Preparing daily tensors...")
        
        daily_data = []
        unique_dates = sorted(df.index.unique())
        
        print(f"Processing {len(unique_dates)} unique dates...")
        
        # Reset index to handle duplicate dates properly
        df_reset = df.reset_index()
        
        for date in unique_dates:
            # Initialize tensors for this day
            day_features = torch.zeros(self.cc.num_cells, len(feature_columns))
            day_target = torch.zeros(self.cc.num_cells, 1)
            
            # Get data for this date
            date_mask = df_reset['date'] == date
            day_data = df_reset[date_mask]
            
            for _, row in day_data.iterrows():
                cell_id = (row['companyID'], row['storeID'], row['skuID'])
                if cell_id in self.cc.cell_to_int:
                    cell_idx = self.cc.cell_to_int[cell_id]
                    
                    # Get feature values and ensure they're numeric
                    feature_values = row[feature_columns].values
                    
                    # Convert to float32 and handle any remaining non-numeric values
                    try:
                        feature_values = pd.to_numeric(feature_values, errors='coerce')
                        feature_values = np.nan_to_num(feature_values, nan=0.0).astype(np.float32)
                    except (ValueError, TypeError):
                        # If conversion fails, create zero vector
                        feature_values = np.zeros(len(feature_columns), dtype=np.float32)
                    
                    # Get target value and ensure it's numeric
                    try:
                        target_value = float(row['target'])
                        if np.isnan(target_value):
                            target_value = 0.0
                    except (ValueError, TypeError):
                        target_value = 0.0
                    
                    # Set features and target for this cell
                    day_features[cell_idx] = torch.tensor(feature_values, dtype=torch.float32)
                    day_target[cell_idx] = torch.tensor([target_value], dtype=torch.float32)
            
            daily_data.append((day_features, day_target))
        
        print(f"✅ Daily tensors prepared: {len(daily_data)} days")
        return daily_data
    
    def split_data(self, daily_data: List[Tuple[torch.Tensor, torch.Tensor]], 
                   test_period: int = 30, val_period: int = 30) -> Tuple[List, List, List]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            daily_data: List of daily tensors
            test_period: Number of days for test set
            val_period: Number of days for validation set
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        total_days = len(daily_data)
        
        if test_period + val_period >= total_days:
            raise ValueError("Test and validation periods are too large for the dataset")
        
        # Split chronologically
        train_end = total_days - test_period - val_period
        val_end = total_days - test_period
        
        train_data = daily_data[:train_end]
        val_data = daily_data[train_end:val_end]
        test_data = daily_data[val_end:]
        
        print(f"📊 Data split:")
        print(f"  - Training: {len(train_data)} days")
        print(f"  - Validation: {len(val_data)} days")
        print(f"  - Test: {len(test_data)} days")
        
        return train_data, val_data, test_data
    
    def create_data_loaders(self, train_data: List, val_data: List, 
                          batch_size: int = 32, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders.
        
        Args:
            train_data: Training data
            val_data: Validation data
            batch_size: Batch size
            shuffle: Whether to shuffle training data
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_dataset = HierarchicalDataset(train_data)
        val_dataset = HierarchicalDataset(val_data)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader


def aggregate_predictions(predictions: torch.Tensor, actuals: torch.Tensor,
                         cc: CombinatorialComplex) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Aggregate predictions and actuals by hierarchy level.
    
    Args:
        predictions: Model predictions
        actuals: Actual values
        cc: Combinatorial complex
        
    Returns:
        Dictionary with aggregated predictions and actuals by rank
    """
    results = {}
    
    for rank in sorted(cc.cells.keys()):
        cell_indices = [cc.cell_to_int[c] for c in cc.cells[rank]]
        
        if rank == 0:  # SKU level - individual forecasts
            rank_preds = predictions[:, cell_indices].detach().numpy().flatten()
            rank_actuals = actuals[:, cell_indices].detach().numpy().flatten()
        else:  # Higher levels - sum over constituent cells
            rank_preds = predictions[:, cell_indices].detach().numpy().sum(axis=1)
            rank_actuals = actuals[:, cell_indices].detach().numpy().sum(axis=1)
        
        results[rank] = {
            'predictions': rank_preds,
            'actuals': rank_actuals
        }
    
    return results


def create_cell_rank_tensor(cc: CombinatorialComplex) -> torch.Tensor:
    """
    Create a tensor indicating the rank of each cell.
    
    Args:
        cc: Combinatorial complex
        
    Returns:
        Tensor with rank for each cell
    """
    ranks = torch.zeros(cc.num_cells, dtype=torch.long)
    
    for rank, cells in cc.cells.items():
        for cell in cells:
            cell_idx = cc.cell_to_int[cell]
            ranks[cell_idx] = rank
    
    return ranks
