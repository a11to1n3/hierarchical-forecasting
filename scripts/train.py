#!/usr/bin/env python3
"""
Main training script for hierarchical forecasting model.

This script handles the complete training pipeline including data loading,
model initialization, training, and evaluation.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path

# Add hierarchical_forecasting directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from hierarchical_forecasting.models import CombinatorialComplex, EnhancedHierarchicalModel, ModelConfig
from hierarchical_forecasting.data import DataPreprocessor, HierarchicalDataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


def _is_constituent(sku_cell, higher_cell, rank):
    """
    Check if a SKU cell is a constituent of a higher-level cell.
    
    Args:
        sku_cell: Tuple representing SKU (companyID, storeID, skuID)
        higher_cell: Tuple representing higher-level cell
        rank: Rank of the higher-level cell (1=store, 2=company, 3=total)
        
    Returns:
        bool: True if SKU belongs to the higher-level cell
    """
    if rank == 1:  # Store level: (companyID, storeID)
        return sku_cell[0] == higher_cell[0] and sku_cell[1] == higher_cell[1]
    elif rank == 2:  # Company level: (companyID,)
        return sku_cell[0] == higher_cell[0]
    elif rank == 3:  # Total level: ('total',)
        return True  # All SKUs belong to total
    return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train hierarchical forecasting model')
    
    parser.add_argument('--data_path', type=str, default='bakery_data_merged.csv',
                        help='Path to the merged data file')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of CCMPN layers')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (not used in current implementation)')
    parser.add_argument('--test_period', type=int, default=30,
                        help='Number of days for testing')
    parser.add_argument('--val_period', type=int, default=30,
                        help='Number of days for validation')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for models and results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, mps, or auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device(device_arg):
    """Get the appropriate device for training."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_arg)


def load_data(data_path):
    """Load and preprocess data."""
    print("📥 Loading data...")
    
    try:
        df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
        print(f"✅ Successfully loaded data from {data_path}")
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        return df
    except FileNotFoundError:
        print(f"❌ Data file {data_path} not found.")
        print("Creating dummy data for demonstration...")
        from hierarchical_forecasting.data.preprocessing import create_dummy_data
        return create_dummy_data()


def train_epoch(model, train_data, optimizer, loss_fn, cc, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for x, y in train_data:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(x, cc.neighborhoods)
        
        # Only compute loss for rank 0 cells (SKUs)
        rank_0_indices = [cc.cell_to_int[c] for c in cc.cells[0]]
        loss = loss_fn(y_pred[rank_0_indices], y[rank_0_indices])
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_data)


def evaluate_model(model, test_data, cc, preprocessor, device):
    """Evaluate the model on test data."""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for x, y in test_data:
            x, y = x.to(device), y.to(device)
            pred = model(x, cc.neighborhoods)
            predictions.append(pred.cpu().numpy())
            actuals.append(y.cpu().numpy())
    
    # Convert to numpy arrays
    pred_tensor = np.array(predictions).squeeze()
    actuals_tensor = np.array(actuals).squeeze()
    
    # Inverse transform if scaling was applied
    if preprocessor.target_scaler is not None:
        pred_tensor_unscaled = preprocessor.inverse_transform_target(
            pred_tensor.reshape(-1, 1)
        ).reshape(pred_tensor.shape)
        actuals_tensor_unscaled = preprocessor.inverse_transform_target(
            actuals_tensor.reshape(-1, 1)
        ).reshape(actuals_tensor.shape)
    else:
        pred_tensor_unscaled = pred_tensor
        actuals_tensor_unscaled = actuals_tensor
    
    # Evaluate at each hierarchical level
    results = []
    
    for rank in sorted(cc.cells.keys()):
        level_name = f"Rank {rank}"
        
        if rank == 0:  # SKU level - individual forecasts
            cell_indices = [cc.cell_to_int[c] for c in cc.cells[rank]]
            level_preds = pred_tensor_unscaled[:, cell_indices].flatten()
            level_actuals = actuals_tensor_unscaled[:, cell_indices].flatten()
        else:  # Higher levels - aggregate from SKU level
            level_preds = []
            level_actuals = []
            
            for cell in cc.cells[rank]:
                # Find all SKU cells that belong to this higher-level cell
                constituent_skus = []
                for sku_cell in cc.cells[0]:
                    if _is_constituent(sku_cell, cell, rank):
                        constituent_skus.append(cc.cell_to_int[sku_cell])
                
                if constituent_skus:
                    # Sum predictions and actuals for constituent SKUs
                    cell_pred = pred_tensor_unscaled[:, constituent_skus].sum(axis=1)
                    cell_actual = actuals_tensor_unscaled[:, constituent_skus].sum(axis=1)
                    level_preds.extend(cell_pred.flatten())
                    level_actuals.extend(cell_actual.flatten())
            
            level_preds = np.array(level_preds)
            level_actuals = np.array(level_actuals)

        # Calculate metrics
        mask = level_actuals > 0  # Avoid division by zero
        if mask.sum() > 0:
            wape = np.sum(np.abs(level_actuals[mask] - level_preds[mask])) / np.sum(np.abs(level_actuals[mask]))
            r2 = r2_score(level_actuals[mask], level_preds[mask])
            mae = mean_absolute_error(level_actuals[mask], level_preds[mask])
            rmse = np.sqrt(mean_squared_error(level_actuals[mask], level_preds[mask]))
        else:
            wape = np.nan
            r2 = np.nan
            mae = np.nan
            rmse = np.nan
        
        results.append({
            'Level': level_name,
            'WAPE': f'{wape:.2%}' if not np.isnan(wape) else 'N/A',
            'R²': f'{r2:.3f}' if not np.isnan(r2) else 'N/A',
            'MAE': f'{mae:.2f}' if not np.isnan(mae) else 'N/A',
            'RMSE': f'{rmse:.2f}' if not np.isnan(rmse) else 'N/A'
        })

    return results


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"🖥️ Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    
    # Load and preprocess data
    df = load_data(args.data_path)
    preprocessor = DataPreprocessor(scaling_method='minmax')
    df_processed = preprocessor.prepare_data(df)
    
    # Build combinatorial complex
    print("\n🏗️ Building combinatorial complex...")
    cc = CombinatorialComplex(df_processed)
    
    # Move neighborhoods to device (keep sparse tensors on CPU for MPS compatibility)
    tensor_device = 'cpu' if device == 'mps' else device
    for name in cc.neighborhoods:
        cc.neighborhoods[name] = cc.neighborhoods[name].to(tensor_device)
    
    # Prepare data for training
    print("\n📊 Preparing training data...")
    data_loader = HierarchicalDataLoader(cc)
    daily_data = data_loader.prepare_daily_tensors(df_processed, preprocessor.feature_columns)
    
    # Split data
    train_data, val_data, test_data = data_loader.split_data(
        daily_data, 
        test_period=args.test_period,
        val_period=args.val_period
    )
    
    # Initialize model
    print(f"\n🧠 Initializing model...")
    model = EnhancedHierarchicalModel(
        num_cells=cc.num_cells,
        num_features=len(preprocessor.feature_columns),
        hidden_dim=args.hidden_dim,
        neighborhood_names=list(cc.neighborhoods.keys()),
        n_layers=args.n_layers
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    
    # Training loop
    print(f"\n🏋️ Training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_data, optimizer, loss_fn, cc, device)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_data:
                x, y = x.to(device), y.to(device)
                y_pred = model(x, cc.neighborhoods)
                rank_0_indices = [cc.cell_to_int[c] for c in cc.cells[0]]
                loss = loss_fn(y_pred[rank_0_indices], y[rank_0_indices])
                val_loss += loss.item()
        val_loss /= len(val_data)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': vars(args)
            }, f"{args.output_dir}/models/best_model.pt")
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    print("✅ Training complete.")
    
    # Load best model for evaluation
    checkpoint = torch.load(f"{args.output_dir}/models/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    results = evaluate_model(model, test_data, cc, preprocessor, device)
    
    # Print results
    print("\n--- Final Performance ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv(f"{args.output_dir}/evaluation_results.csv", index=False)
    
    print(f"\n🎯 Model Summary:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Hierarchy levels: {len(cc.cells)}")
    print(f"  - Total cells in complex: {cc.num_cells}")
    print(f"  - Neighborhoods: {list(cc.neighborhoods.keys())}")
    print(f"  - Best validation loss: {best_val_loss:.6f}")
    
    print(f"\n💾 Results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
