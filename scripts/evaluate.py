#!/usr/bin/env python3
"""
Model evaluation script for hierarchical forecasting.

This script loads a trained model and evaluates its performance
on test data with detailed analysis.
"""

import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add hierarchical_forecasting directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from hierarchical_forecasting.models import CombinatorialComplex, EnhancedHierarchicalModel
from hierarchical_forecasting.data import DataPreprocessor, HierarchicalDataLoader
from hierarchical_forecasting.visualization import TrainingVisualizer
from hierarchical_forecasting.utils import (
    weighted_absolute_percentage_error,
    weighted_absolute_squared_error,
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate hierarchical forecasting model')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model checkpoint')
    parser.add_argument('--data_path', type=str, default='bakery_data_merged.csv',
                        help='Path to the merged data file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for evaluation results')
    parser.add_argument('--test_period', type=int, default=30,
                        help='Number of days for testing')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, mps, or auto)')
    
    return parser.parse_args()


def get_device(device_arg):
    """Get the appropriate device for evaluation."""
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
        return df
    except FileNotFoundError:
        print(f"❌ Data file {data_path} not found.")
        raise


def evaluate_model_detailed(model, test_data, cc, preprocessor, device):
    """Detailed model evaluation with comprehensive metrics."""
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
    level_data = {}
    
    for rank in sorted(cc.cells.keys()):
        level_name = f"Rank {rank}"
        cell_indices = [cc.cell_to_int[c] for c in cc.cells[rank]]
        
        # Aggregate predictions and actuals for the current rank
        if rank == 0:  # SKU level - individual forecasts
            level_preds = pred_tensor_unscaled[:, cell_indices].flatten()
            level_actuals = actuals_tensor_unscaled[:, cell_indices].flatten()
        else:  # Higher levels - sum over constituent cells
            level_preds = pred_tensor_unscaled[:, cell_indices].sum(axis=1)
            level_actuals = actuals_tensor_unscaled[:, cell_indices].sum(axis=1)

        # Store for visualization
        level_data[rank] = {
            'predictions': level_preds,
            'actuals': level_actuals
        }
        
        # Calculate comprehensive metrics
        wape = weighted_absolute_percentage_error(level_actuals, level_preds)
        wase = weighted_absolute_squared_error(level_actuals, level_preds)

        mask = np.abs(level_actuals) > 0  # Avoid division by zero for other metrics
        if mask.sum() > 0:
            filtered_actuals = level_actuals[mask]
            filtered_preds = level_preds[mask]

            r2 = r2_score(filtered_actuals, filtered_preds)
            mae = mean_absolute_error(filtered_actuals, filtered_preds)
            rmse = np.sqrt(mean_squared_error(filtered_actuals, filtered_preds))
            mape = np.mean(np.abs((filtered_actuals - filtered_preds) / filtered_actuals)) * 100

            bias = np.mean(filtered_preds - filtered_actuals)
            std_residuals = np.std(filtered_preds - filtered_actuals)
        else:
            r2 = mae = rmse = mape = bias = std_residuals = np.nan
        
        results.append({
            'Level': level_name,
            'WAPE': f'{wape:.2f}%' if not np.isnan(wape) else 'N/A',
            'WASE': f'{wase:.2f}' if not np.isnan(wase) else 'N/A',
            'R²': f'{r2:.3f}' if not np.isnan(r2) else 'N/A',
            'MAE': f'{mae:.2f}' if not np.isnan(mae) else 'N/A',
            'RMSE': f'{rmse:.2f}' if not np.isnan(rmse) else 'N/A',
            'MAPE': f'{mape:.2f}%' if not np.isnan(mape) else 'N/A',
            'Bias': f'{bias:.2f}' if not np.isnan(bias) else 'N/A',
            'Std_Residuals': f'{std_residuals:.2f}' if not np.isnan(std_residuals) else 'N/A'
        })

    return results, level_data


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Get device
    device = get_device(args.device)
    print(f"🖥️ Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/plots", exist_ok=True)
    
    # Load model checkpoint
    print(f"📥 Loading model from {args.model_path}...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model_config = checkpoint['config']
        print("✅ Model checkpoint loaded successfully")
    except FileNotFoundError:
        print(f"❌ Model file {args.model_path} not found.")
        return
    
    # Load and preprocess data
    df = load_data(args.data_path)
    preprocessor = DataPreprocessor(scaling_method='minmax')
    df_processed = preprocessor.prepare_data(df)
    
    # Build combinatorial complex
    print("\n🏗️ Building combinatorial complex...")
    cc = CombinatorialComplex(df_processed)
    
    # Move neighborhoods to device
    for name in cc.neighborhoods:
        cc.neighborhoods[name] = cc.neighborhoods[name].to(device)
    
    # Initialize model
    model = EnhancedHierarchicalModel(
        num_cells=cc.num_cells,
        num_features=len(preprocessor.feature_columns),
        hidden_dim=model_config['hidden_dim'],
        neighborhood_names=list(cc.neighborhoods.keys()),
        n_layers=model_config['n_layers']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model weights loaded successfully")
    
    # Prepare test data
    print("\n📊 Preparing test data...")
    data_loader = HierarchicalDataLoader(cc)
    daily_data = data_loader.prepare_daily_tensors(df_processed, preprocessor.feature_columns)
    
    # Use only test data
    test_data = daily_data[-args.test_period:]
    print(f"📊 Using {len(test_data)} days for evaluation")
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    results, level_data = evaluate_model_detailed(model, test_data, cc, preprocessor, device)
    
    # Print results
    print("\n--- Detailed Evaluation Results ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv(f"{args.output_dir}/detailed_evaluation_results.csv", index=False)
    print(f"\n💾 Results saved to {args.output_dir}/detailed_evaluation_results.csv")
    
    # Create visualizations
    print("\n🎨 Creating evaluation visualizations...")
    viz = TrainingVisualizer(f"{args.output_dir}/plots")
    
    # Plot evaluation results
    viz.plot_evaluation_results(results_df)
    
    # Plot predictions vs actuals for each level
    for rank, data in level_data.items():
        level_names = {0: 'SKU Level', 1: 'Store Level', 2: 'Company Level', 3: 'Total Level'}
        if rank in level_names:
            viz.plot_predictions_vs_actuals(
                data['predictions'], data['actuals'], level_names[rank]
            )
            viz.plot_residuals(
                data['predictions'], data['actuals'], level_names[rank]
            )
    
    # Print model summary
    print(f"\n🎯 Evaluation Summary:")
    print(f"  - Model: {args.model_path}")
    print(f"  - Test period: {args.test_period} days")
    print(f"  - Complex size: {cc.num_cells} cells")
    print(f"  - Best metric at SKU level: {results[0]['R²']} R²")
    print(f"  - Training epoch: {checkpoint['epoch']}")
    print(f"  - Training loss: {checkpoint['train_loss']:.6f}")
    print(f"  - Validation loss: {checkpoint['val_loss']:.6f}")


if __name__ == '__main__':
    main()
