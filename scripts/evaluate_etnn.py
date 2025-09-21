#!/usr/bin/env python3
"""
ETNN (E(n) Equivariant Topological Neural Networks) Baseline Evaluation Script.

This script evaluates ETNN-based approaches for hierarchical forecasting,
implementing concepts from the ICLR 2025 paper on topological deep learning.
"""

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

warnings.filterwarnings("ignore")

# Add hierarchical_forecasting directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from hierarchical_forecasting.models import (
    CombinatorialComplex, ModelConfig, ETNNEnhancedHierarchicalModel
)
from hierarchical_forecasting.data import DataPreprocessor, HierarchicalDataLoader
from hierarchical_forecasting.baselines import ETNNBaseline
from hierarchical_forecasting.baselines import PatchTSTBaseline, TimesNetBaseline
from hierarchical_forecasting.utils import (
    weighted_absolute_percentage_error,
    weighted_percentage_error,
)
from hierarchical_forecasting.visualization import TrainingVisualizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_hierarchical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hierarchy_levels: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Evaluate hierarchical metrics for ETNN evaluation on original-scale values."""

    metrics: Dict[str, float] = {}

    metrics['Overall_WAPE'] = weighted_absolute_percentage_error(y_true, y_pred)
    metrics['Overall_WPE'] = weighted_percentage_error(y_true, y_pred)
    metrics['Overall_MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['Overall_RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['Overall_R2'] = r2_score(y_true, y_pred)

    if hierarchy_levels is not None:
        unique_levels = sorted(set(hierarchy_levels))
        for level in unique_levels:
            level_mask = [idx for idx, lvl in enumerate(hierarchy_levels) if lvl == level]
            if not level_mask:
                continue
            level_true = y_true[:, level_mask]
            level_pred = y_pred[:, level_mask]
            metrics[f'Level_{level}_WAPE'] = weighted_absolute_percentage_error(level_true, level_pred)
            metrics[f'Level_{level}_WPE'] = weighted_percentage_error(level_true, level_pred)
            metrics[f'Level_{level}_MAE'] = mean_absolute_error(level_true, level_pred)
            metrics[f'Level_{level}_RMSE'] = np.sqrt(mean_squared_error(level_true, level_pred))
            metrics[f'Level_{level}_R2'] = r2_score(level_true, level_pred)

    return metrics


def create_etnn_models(
    patchtst_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create ETNN-based models for evaluation.
    
    Returns:
        Dictionary of ETNN models
    """
    models = {
        'ETNN_Basic': ETNNBaseline(
            hidden_dim=32,
            num_layers=2,
            spatial_dim=2,
            use_geometric_features=True,
            epochs=100
        ),
        'ETNN_Deep': ETNNBaseline(
            hidden_dim=64,
            num_layers=4,
            spatial_dim=3,
            use_geometric_features=True,
            epochs=150
        ),
    }

    if patchtst_checkpoint:
        models['PatchTST'] = PatchTSTBaseline(checkpoint_path=patchtst_checkpoint)
    else:
        print("Skipping PatchTST evaluation baseline (no checkpoint provided).")

    models['TimesNet'] = TimesNetBaseline()

    return models


def run_etnn_evaluation(
    data_path: str,
    output_dir: str = "outputs/etnn_evaluation",
    patchtst_checkpoint: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run comprehensive ETNN model evaluation.
    
    Args:
        data_path: Path to the dataset
        output_dir: Output directory for results
        
    Returns:
        DataFrame with evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔬 Starting ETNN Model Evaluation...")
    print("=" * 60)
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data with date parsing
    df = pd.read_csv(data_path, parse_dates=['date'])
    print(f"✅ Successfully loaded data from {data_path}")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor()
    
    # Create merged data with features and targets
    data = preprocessor.prepare_data(df)
    
    # Split data chronologically
    train_data, val_data, test_data = preprocessor.split_data(data, test_period=30, val_period=30)
    
    # Process each split
    def prepare_split_data(split_data):
        features = preprocessor.create_features(split_data)
        targets = preprocessor.create_targets(split_data)
        hierarchy = preprocessor.create_hierarchy(split_data)
        
        # Find numeric columns only (exclude date and entity_id)
        numeric_cols = []
        for col in features.columns:
            if col in ['date', 'entity_id']:
                continue
            try:
                pd.to_numeric(features[col])
                numeric_cols.append(col)
            except:
                continue
        
        # Convert to numpy arrays using only numeric columns
        features_array = features[numeric_cols].values.astype(np.float32)
        targets_array = targets.values.astype(np.float32)
        entities = hierarchy.get(0, list(range(targets_array.shape[1])))
        if isinstance(entities, list):
            resolved_entities = []
            for ent in entities:
                if isinstance(ent, tuple):
                    resolved_entities.append(ent)
                elif isinstance(ent, list):
                    resolved_entities.append(tuple(ent))
                else:
                    resolved_entities.append((ent,))
            entities = resolved_entities
        levels = [len(ent) - 1 if isinstance(ent, tuple) else 0 for ent in entities]
        return features_array, targets_array, levels, entities
    
    X_train, y_train, levels_train, entities_train = prepare_split_data(train_data)
    X_val, y_val, levels_val, entities_val = prepare_split_data(val_data)
    X_test, y_test, levels_test, entities_test = prepare_split_data(test_data)
    
    print(f"📊 Data split:")
    print(f"  - Training: {X_train.shape[0]} samples")
    print(f"  - Validation: {X_val.shape[0]} samples")
    print(f"  - Test: {X_test.shape[0]} samples")
    
    # Create hierarchy information
    hierarchy_info = {
        'entities': entities_train,
        'levels': levels_train
    }
    
    # Create ETNN models
    etnn_models = create_etnn_models(
        patchtst_checkpoint=patchtst_checkpoint,
    )
    
    # Evaluation results
    results = []
    
    # Evaluate each ETNN model
    for model_name, model in etnn_models.items():
        print(f"\n🚀 Evaluating {model_name}...")
        print("-" * 40)
        
        try:
            start_time = time.time()
            
            # Train model
            print(f"Training {model_name}...")
            model.fit(
                X_train,
                y_train,
                hierarchy_info=hierarchy_info,
                entity_ids=entities_train
            )
            
            # Make predictions
            print(f"Making predictions with {model_name}...")
            y_pred = model.predict(X_test, entity_ids=entities_test)
            
            training_time = time.time() - start_time
            
            # Evaluate metrics
            metrics = evaluate_hierarchical_metrics(
                y_true=y_test,
                y_pred=y_pred,
                hierarchy_levels=levels_test
            )
            
            # Add model info to metrics
            metrics['Model'] = model_name
            metrics['Training_Time'] = training_time
            
            results.append(metrics)
            
            print(f"✅ {model_name} completed:")
            print(f"   - Training time: {training_time:.2f}s")
            print(f"   - Overall WAPE: {metrics['Overall_WAPE']:.4f}%")
            print(f"   - Overall WPE: {metrics['Overall_WPE']:.6f}")
            print(f"   - Overall RMSE: {metrics['Overall_RMSE']:.4f}")
            print(f"   - Overall R²: {metrics['Overall_R2']:.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {str(e)}")
            # Add failed result
            failed_metrics = {
                'Model': model_name,
                'Training_Time': 0,
                'Overall_WAPE': float('inf'),
                'Overall_WPE': float('inf'),
                'Overall_MAE': float('inf'),
                'Overall_RMSE': float('inf'),
                'Overall_R2': -float('inf'),
                'Error': str(e)
            }
            results.append(failed_metrics)
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join(output_dir, 'etnn_evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to {results_path}")
    
    # Print summary
    print("\n📋 ETNN Model Evaluation Summary:")
    print("=" * 60)
    
    if len(results_df) > 0:
        # Sort by overall WAPE (lower is better)
        summary_df = results_df[['Model', 'Overall_WAPE', 'Overall_WPE', 'Overall_RMSE', 'Overall_R2', 'Training_Time']].copy()
        summary_df = summary_df.sort_values('Overall_WAPE')
        
        print(summary_df.to_string(index=False, float_format='%.4f'))
        
        # Best model
        best_model = summary_df.iloc[0]['Model']
        best_wape = summary_df.iloc[0]['Overall_WAPE']
        best_wpe = summary_df.iloc[0]['Overall_WPE']
        print(f"\n🏆 Best ETNN Model: {best_model}")
        print(f"   Best WAPE: {best_wape:.4f}%")
        print(f"   Best WPE: {best_wpe:.6f}")
    
    return results_df


def run_hybrid_etnn_ccmpn_evaluation(data_path: str, output_dir: str = "outputs/etnn_evaluation") -> Dict[str, Any]:
    """
    Run evaluation of the hybrid ETNN-CCMPN model.
    
    Args:
        data_path: Path to the dataset
        output_dir: Output directory for results
        
    Returns:
        Dictionary with evaluation results
    """
    print("\n🔬 Starting Hybrid ETNN-CCMPN Evaluation...")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=['date'])
    preprocessor = DataPreprocessor()
    
    # Create merged data with features and targets
    data = preprocessor.prepare_data(df)
    
    # Split data chronologically
    train_data, val_data, test_data = preprocessor.split_data(data, test_period=30, val_period=30)
    
    # Process training and test splits
    def prepare_split_data(split_data):
        features = preprocessor.create_features(split_data)
        targets = preprocessor.create_targets(split_data)
        
        # Find numeric columns only (exclude date and entity_id)
        numeric_cols = []
        for col in features.columns:
            if col in ['date', 'entity_id']:
                continue
            try:
                pd.to_numeric(features[col])
                numeric_cols.append(col)
            except:
                continue
        
        # Convert to numpy arrays using only numeric columns
        features_array = features[numeric_cols].values.astype(np.float32)
        targets_array = targets.values.astype(np.float32)
        return features_array, targets_array
    
    X_train, y_train = prepare_split_data(train_data)
    X_test, y_test = prepare_split_data(test_data)
    
    # Create configuration
    config = ModelConfig()
    config.input_dim = X_train.shape[1]
    config.output_dim = y_train.shape[1]
    config.hidden_dim = 64
    config.num_layers = 3
    config.spatial_dim = 2
    config.max_entities = y_train.shape[1]
    
    # Create combinatorial complex
    entities = list(range(y_train.shape[1]))
    cells = {
        0: entities,  # 0-cells: individual entities
        1: [[i, i+1] for i in range(len(entities)-1)],  # 1-cells: sequential pairs
        2: [[i, i+1, i+2] for i in range(len(entities)-2)]  # 2-cells: triplets
    }
    cc = CombinatorialComplex(cells)
    
    # Create and train hybrid model
    print("🚀 Training Hybrid ETNN-CCMPN Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ETNNEnhancedHierarchicalModel(config).to(device)
    
    # Convert data to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    num_epochs = 50
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X_train_tensor, cc)
        
        # Use rank_0 predictions for loss (could be enhanced)
        if 'rank_0' in predictions:
            loss = criterion(predictions['rank_0'], y_train_tensor)
        else:
            # Fallback if rank_0 not available
            pred_values = list(predictions.values())[0]
            if pred_values.shape[1] != y_train_tensor.shape[1]:
                # Reshape or select appropriate predictions
                pred_values = pred_values[:, :y_train_tensor.shape[1]]
            loss = criterion(pred_values, y_train_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    # Evaluation
    print("📊 Evaluating Hybrid ETNN-CCMPN Model...")
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor, cc)
        
        # Use rank_0 predictions for evaluation
        if 'rank_0' in test_predictions:
            y_pred = test_predictions['rank_0'].cpu().numpy()
        else:
            pred_values = list(test_predictions.values())[0]
            if pred_values.shape[1] != y_test.shape[1]:
                pred_values = pred_values[:, :y_test.shape[1]]
            y_pred = pred_values.cpu().numpy()
    
    # Calculate metrics
    metrics = evaluate_hierarchical_metrics(y_true=y_test, y_pred=y_pred)
    metrics['Model'] = 'Hybrid_ETNN_CCMPN'
    
    print(f"✅ Hybrid ETNN-CCMPN Results:")
    print(f"   - Overall WAPE: {metrics['Overall_WAPE']:.4f}%")
    print(f"   - Overall WPE: {metrics['Overall_WPE']:.6f}")
    print(f"   - Overall RMSE: {metrics['Overall_RMSE']:.4f}")
    print(f"   - Overall R²: {metrics['Overall_R2']:.4f}")
    
    # Save results
    hybrid_results_path = os.path.join(output_dir, 'hybrid_etnn_ccmpn_results.csv')
    pd.DataFrame([metrics]).to_csv(hybrid_results_path, index=False)
    
    return metrics


def main():
    """Main function for ETNN evaluation."""
    parser = argparse.ArgumentParser(description='ETNN Model Evaluation')
    parser.add_argument('--data_path', type=str, default='bakery_data_merged.csv',
                       help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/etnn_evaluation',
                       help='Output directory for results')
    parser.add_argument('--eval_hybrid', action='store_true',
                       help='Also evaluate hybrid ETNN-CCMPN model')
    parser.add_argument('--patchtst_checkpoint', type=str, default=None,
                        help='Path to a TorchScript or module checkpoint for PatchTST')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Run ETNN baseline evaluation
        results_df = run_etnn_evaluation(
            args.data_path,
            args.output_dir,
            patchtst_checkpoint=args.patchtst_checkpoint,
        )
        
        # Run hybrid model evaluation if requested
        if args.eval_hybrid:
            hybrid_results = run_hybrid_etnn_ccmpn_evaluation(args.data_path, args.output_dir)
        
        print("\n🎉 ETNN Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during ETNN evaluation: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
