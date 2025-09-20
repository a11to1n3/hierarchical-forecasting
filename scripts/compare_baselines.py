#!/usr/bin/env python3
"""
Baseline comparison script for hierarchical forecasting.

This script runs all baseline models and compares their performance
against the hierarchical CCMPN model.
"""

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# Add hierarchical_forecasting directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from hierarchical_forecasting.models import CombinatorialComplex, EnhancedHierarchicalModel
from hierarchical_forecasting.data import DataPreprocessor, HierarchicalDataLoader
from hierarchical_forecasting.baselines import (
    LinearRegressionBaseline, MultiLevelLinearRegression,
    RandomForestBaseline, HierarchicalRandomForest,
    LSTMBaseline, MultiEntityLSTM,
    BottomUpBaseline, TopDownBaseline, MiddleOutBaseline,
    MinTBaseline, OLSBaseline, ETNNBaseline,
    PatchTSTBaseline, TimesNetBaseline
)
from hierarchical_forecasting.utils import (
    weighted_absolute_percentage_error,
    weighted_absolute_squared_error,
)

try:
    from hierarchical_forecasting.baselines import ProphetBaseline, HierarchicalProphet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def create_baseline_models(input_size: int,
                           patchtst_checkpoint: Optional[str] = None) -> Dict:
    """
    Create all baseline models for comparison.

    Args:
        input_size: Number of features provided to each model.

    Returns:
        Dictionary of baseline models
    """
    baselines = {
        # Simple baselines
        'Linear_Regression': LinearRegressionBaseline(),
        'Ridge_Regression': LinearRegressionBaseline(regularization='ridge', alpha=1.0),
        'Lasso_Regression': LinearRegressionBaseline(regularization='lasso', alpha=1.0),
        'Random_Forest': RandomForestBaseline(n_estimators=100),
        'Multi_Level_Linear': MultiLevelLinearRegression(),
        'Hierarchical_RF': HierarchicalRandomForest(n_estimators=50),

        # Neural network baselines
        'LSTM': LSTMBaseline(input_size=input_size, hidden_size=32, epochs=20),
        'Multi_Entity_LSTM': MultiEntityLSTM(input_size=input_size, hidden_size=32, epochs=20),
        
        # ETNN-based baselines (Topological Deep Learning)
        'ETNN': ETNNBaseline(hidden_dim=32, num_layers=2, epochs=120),
    }

    if patchtst_checkpoint:
        baselines['PatchTST'] = PatchTSTBaseline(checkpoint_path=patchtst_checkpoint)
    else:
        print("No PatchTST checkpoint provided; training a fresh PatchTST baseline from scratch.")
        baselines['PatchTST'] = PatchTSTBaseline(allow_training=True)

    baselines['TimesNet'] = TimesNetBaseline()

    baselines.update({
        'Bottom_Up_Linear': BottomUpBaseline(base_model='linear'),
        'Bottom_Up_RF': BottomUpBaseline(base_model='random_forest', n_estimators=50),
        'Top_Down_Linear': TopDownBaseline(base_model='linear'),
        'Top_Down_RF': TopDownBaseline(base_model='random_forest', n_estimators=50),
        'Middle_Out': MiddleOutBaseline(base_model='linear', middle_level=1),
        'MinT_Linear': MinTBaseline(base_model='linear'),
        'MinT_RF': MinTBaseline(base_model='random_forest', n_estimators=50),
        'OLS': OLSBaseline(base_model='linear'),
    })
    
    # Add Prophet baselines if available
    if PROPHET_AVAILABLE:
        baselines.update({
            'Prophet': ProphetBaseline(),
            'Hierarchical_Prophet': HierarchicalProphet()
        })
    
    return baselines


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                  **kwargs) -> Dict[str, float]:
    """
    Evaluate a model and return metrics.
    
    Args:
        model: The model to evaluate
        X_test: Test features
        y_test: Test targets
        **kwargs: Additional arguments for prediction
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        start_time = time.time()
        predictions = model.predict(X_test, **kwargs)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        wape = weighted_absolute_percentage_error(y_test, predictions)
        wase = weighted_absolute_squared_error(y_test, predictions)
        
        # R-squared
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # MAPE (avoiding division by zero)
        mape = np.mean(np.abs((y_test - predictions) / np.where(y_test == 0, 1, y_test))) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'WAPE': wape,
            'WASE': wase,
            'R2': r2,
            'MAPE': mape,
            'Prediction_Time': prediction_time
        }
    
    except Exception as e:
        print(f"Error evaluating model {getattr(model, 'name', 'Unknown')}: {e}")
        return {
            'MSE': np.inf,
            'MAE': np.inf,
            'RMSE': np.inf,
            'R2': -np.inf,
            'MAPE': np.inf,
            'Prediction_Time': np.inf
        }


def evaluate_hierarchical_metrics(model, X_test, y_test, entities_test, hierarchy, **kwargs):
    """
    Evaluate model performance at each hierarchy level like the CCMPN model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        entities_test: Entity identifiers for test data
        hierarchy: Hierarchy structure
        **kwargs: Additional arguments for model prediction
        
    Returns:
        Dictionary with hierarchical metrics
    """
    try:
        # Get base predictions
        predictions = model.predict(X_test, **kwargs)
        
        # Initialize results for each rank
        hierarchical_results = {}
        
        for rank in range(4):  # Ranks 0-3
            # Get entities at this rank
            if rank == 0:  # SKU level
                # Direct predictions for individual SKUs
                mask = np.array([entity in hierarchy[0] for entity in entities_test])
                if mask.sum() > 0:
                    level_preds = predictions[mask]
                    level_actuals = y_test[mask]
                else:
                    continue
                    
            elif rank == 1:  # Store level
                # Aggregate by store (companyID, storeID)
                store_entities = hierarchy[1]
                level_preds = []
                level_actuals = []
                
                for store in store_entities:
                    # Find all SKUs belonging to this store
                    sku_mask = np.array([
                        entity[:2] == store for entity in entities_test
                    ])
                    if sku_mask.sum() > 0:
                        level_preds.append(predictions[sku_mask].sum())
                        level_actuals.append(y_test[sku_mask].sum())
                
                if not level_preds:
                    continue
                level_preds = np.array(level_preds)
                level_actuals = np.array(level_actuals)
                
            elif rank == 2:  # Company level
                # Aggregate by company (companyID,)
                company_entities = hierarchy[2]
                level_preds = []
                level_actuals = []
                
                for company in company_entities:
                    # Find all SKUs belonging to this company
                    sku_mask = np.array([
                        entity[0] == company[0] for entity in entities_test
                    ])
                    if sku_mask.sum() > 0:
                        level_preds.append(predictions[sku_mask].sum())
                        level_actuals.append(y_test[sku_mask].sum())
                
                if not level_preds:
                    continue
                level_preds = np.array(level_preds)
                level_actuals = np.array(level_actuals)
                
            elif rank == 3:  # Total level
                # Aggregate all predictions
                level_preds = np.array([predictions.sum()])
                level_actuals = np.array([y_test.sum()])
            
            # Calculate metrics for this level
            if len(level_preds) > 0 and len(level_actuals) > 0:
                wape = weighted_absolute_percentage_error(level_actuals, level_preds)
                wase = weighted_absolute_squared_error(level_actuals, level_preds)

                # Determine mask for metrics that need more than one sample
                nonzero_mask = np.abs(level_actuals) > 0
                valid_indices = np.where(nonzero_mask)[0]

                if valid_indices.size > 0:
                    y_true_valid = level_actuals[valid_indices]
                    y_pred_valid = level_preds[valid_indices]

                    r2 = r2_score(y_true_valid, y_pred_valid)
                    mae = mean_absolute_error(y_true_valid, y_pred_valid)
                    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
                else:
                    r2 = np.nan
                    mae = np.nan
                    rmse = np.nan

                hierarchical_results[f'Rank_{rank}_WAPE'] = wape
                hierarchical_results[f'Rank_{rank}_WASE'] = wase
                hierarchical_results[f'Rank_{rank}_R2'] = r2
                hierarchical_results[f'Rank_{rank}_MAE'] = mae
                hierarchical_results[f'Rank_{rank}_RMSE'] = rmse
            else:
                hierarchical_results[f'Rank_{rank}_WAPE'] = np.nan
                hierarchical_results[f'Rank_{rank}_WASE'] = np.nan
                hierarchical_results[f'Rank_{rank}_R2'] = np.nan
                hierarchical_results[f'Rank_{rank}_MAE'] = np.nan
                hierarchical_results[f'Rank_{rank}_RMSE'] = np.nan
        
        return hierarchical_results
        
    except Exception as e:
        print(f"Error in hierarchical evaluation: {e}")
        # Return NaN values for all metrics
        hierarchical_results = {}
        for rank in range(4):
            hierarchical_results[f'Rank_{rank}_WAPE'] = np.nan
            hierarchical_results[f'Rank_{rank}_WASE'] = np.nan
            hierarchical_results[f'Rank_{rank}_R2'] = np.nan
            hierarchical_results[f'Rank_{rank}_MAE'] = np.nan
            hierarchical_results[f'Rank_{rank}_RMSE'] = np.nan
        return hierarchical_results


def evaluate_model(model, X_test, y_test, entities_test=None, hierarchy=None, **kwargs):
    """
    Evaluate model performance including hierarchical metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        entities_test: Entity identifiers (optional, for hierarchical metrics)
        hierarchy: Hierarchy structure (optional, for hierarchical metrics)
        **kwargs: Additional arguments for model prediction
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        start_time = time.time()
        predictions = model.predict(X_test, **kwargs)
        prediction_time = time.time() - start_time
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)

        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        mape = np.mean(np.abs((y_test - predictions) / np.where(y_test == 0, 1, y_test))) * 100
        wape = weighted_absolute_percentage_error(y_test, predictions)
        wase = weighted_absolute_squared_error(y_test, predictions)

        results = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'WAPE': wape,
            'WASE': wase,
            'Prediction_Time': prediction_time
        }
        
        # Add hierarchical metrics if data is available
        if entities_test is not None and hierarchy is not None:
            hierarchical_metrics = evaluate_hierarchical_metrics(
                model, X_test, y_test, entities_test, hierarchy, **kwargs
            )
            results.update(hierarchical_metrics)
        
        return results
    
    except Exception as e:
        print(f"Error evaluating model {getattr(model, 'name', 'Unknown')}: {e}")
        results = {
            'MSE': np.inf,
            'MAE': np.inf,
            'RMSE': np.inf,
            'R2': -np.inf,
            'MAPE': np.inf,
            'WAPE': np.inf,
            'WASE': np.inf,
            'Prediction_Time': np.inf
        }
        
        # Add NaN hierarchical metrics if requested
        if entities_test is not None and hierarchy is not None:
            for rank in range(4):
                results[f'Rank_{rank}_WAPE'] = np.nan
                results[f'Rank_{rank}_R2'] = np.nan
                results[f'Rank_{rank}_MAE'] = np.nan
                results[f'Rank_{rank}_RMSE'] = np.nan
        
        return results


def run_baseline_comparison(
    data_path: str,
    output_dir: str = "outputs",
    patchtst_checkpoint: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run comprehensive baseline comparison.
    
    Args:
        data_path: Path to the data file
        output_dir: Output directory for results
        patchtst_checkpoint: Optional path to a pre-trained PatchTST checkpoint
        timesfm_checkpoint: Optional path to a pre-trained TimesFM checkpoint
        
    Returns:
        DataFrame with comparison results
    """
    print("Loading and preprocessing data...")
    
    # Load data with proper date parsing (like the training script)
    try:
        data = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
        print(f"✅ Successfully loaded data from {data_path}")
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # Fallback: load without date parsing and convert manually
        data = pd.read_csv(data_path)
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date').sort_index()
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.prepare_data(data)

    # Create hierarchy
    hierarchy = preprocessor.create_hierarchy(processed_data)

    # Split data
    train_data, val_data, test_data = preprocessor.split_data(processed_data, test_period=30, val_period=30)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Prepare features and targets for each split
    def prepare_split_data(split_data):
        if isinstance(split_data.index, pd.DatetimeIndex):
            dates = split_data.index.to_numpy()
        elif 'date' in split_data.columns:
            dates = pd.to_datetime(split_data['date']).to_numpy()
        else:
            dates = np.arange(len(split_data))

        features = preprocessor.create_features(split_data)
        targets = preprocessor.create_targets(split_data)
        
        # Create entity level indicators
        entity_levels = np.zeros(len(features), dtype=int)
        for i, entity_id in enumerate(features['entity_id']):
            # Normalise entity identifiers to tuples for comparison
            if isinstance(entity_id, (list, tuple)):
                entity_tuple = tuple(entity_id)
            else:
                entity_tuple = (entity_id,)

            company_id = store_id = sku_id = None
            if len(entity_tuple) >= 3:
                company_id, store_id, sku_id = entity_tuple[:3]
            elif len(entity_tuple) == 2:
                company_id, store_id = entity_tuple
            elif len(entity_tuple) == 1:
                company_id = entity_tuple[0]

            try:
                if 3 in hierarchy and entity_tuple in hierarchy[3]:
                    entity_levels[i] = 3
                elif 0 in hierarchy and (company_id, store_id, sku_id) in hierarchy[0]:
                    entity_levels[i] = 0
                elif 1 in hierarchy and (company_id, store_id) in hierarchy[1]:
                    entity_levels[i] = 1
                elif 2 in hierarchy and (company_id,) in hierarchy[2]:
                    entity_levels[i] = 2
                else:
                    entity_levels[i] = 0
            except Exception:
                entity_levels[i] = 0
        
        # Remove entity_id and non-feature columns for model input
        # Use the feature columns identified by the preprocessor
        feature_columns = [col for col in preprocessor.feature_columns if col in features.columns]
        if not feature_columns:
            numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_cols if col not in {'target', 'entity_id'}]

        X = features[feature_columns].values
        y = targets['target'].values

        print(f"Selected feature columns: {feature_columns}")
        return X, y, entity_levels, features['entity_id'].values, dates

    X_train, y_train, levels_train, entities_train, dates_train = prepare_split_data(train_data)
    X_val, y_val, levels_val, entities_val, dates_val = prepare_split_data(val_data)
    X_test, y_test, levels_test, entities_test, dates_test = prepare_split_data(test_data)
    
    print(f"Feature dimensions: {X_train.shape}")
    
    # Create baseline models with the correct input dimensionality
    input_size = X_train.shape[1]
    baselines = create_baseline_models(
        input_size,
        patchtst_checkpoint=patchtst_checkpoint,
    )
    
    results = []
    
    print(f"\nTraining and evaluating {len(baselines)} baseline models...")
    
    for name, model in baselines.items():
        print(f"\nTraining {name}...")
        
        try:
            start_time = time.time()
            
            # Prepare training arguments
            train_kwargs = {
                'hierarchy': hierarchy,
                'entity_levels': levels_train,
                'entity_ids': entities_train
            }

            # Special handling for Prophet models
            if 'Prophet' in name:
                train_kwargs['dates'] = dates_train
                train_kwargs['entity_ids'] = entities_train

            # Train model
            model.fit(X_train, y_train, **train_kwargs)
            training_time = time.time() - start_time
            
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Evaluate on validation set
            eval_kwargs = {
                'entity_levels': levels_val,
                'entity_ids': entities_val
            }
            if 'Prophet' in name:
                eval_kwargs['dates'] = dates_val
                eval_kwargs['entity_ids'] = entities_val

            val_metrics = evaluate_model(
                model, X_val, y_val, 
                entities_test=entities_val, 
                hierarchy=hierarchy, 
                **eval_kwargs
            )
            
            # Evaluate on test set
            eval_kwargs['entity_levels'] = levels_test
            eval_kwargs['entity_ids'] = entities_test
            if 'Prophet' in name:
                eval_kwargs['dates'] = dates_test
                eval_kwargs['entity_ids'] = entities_test
            
            test_metrics = evaluate_model(
                model, X_test, y_test, 
                entities_test=entities_test, 
                hierarchy=hierarchy, 
                **eval_kwargs
            )
            
            # Store results
            result = {
                'Model': name,
                'Training_Time': training_time,
                'Val_MSE': val_metrics['MSE'],
                'Val_MAE': val_metrics['MAE'],
                'Val_R2': val_metrics['R2'],
                'Val_WAPE': val_metrics.get('WAPE', np.nan),
                'Val_WASE': val_metrics.get('WASE', np.nan),
                'Test_MSE': test_metrics['MSE'],
                'Test_MAE': test_metrics['MAE'],
                'Test_RMSE': test_metrics['RMSE'],
                'Test_R2': test_metrics['R2'],
                'Test_MAPE': test_metrics['MAPE'],
                'Test_WAPE': test_metrics.get('WAPE', np.nan),
                'Test_WASE': test_metrics.get('WASE', np.nan),
                'Prediction_Time': test_metrics['Prediction_Time']
            }
            
            results.append(result)
            
            print(f"Validation R²: {val_metrics['R2']:.4f}, Test R²: {test_metrics['R2']:.4f}")
            
        except Exception as e:
            print(f"Error with {name}: {e}")
            # Add failed result
            failed_result = {
                'Model': name,
                'Training_Time': np.inf,
                'Val_MSE': np.inf,
                'Val_MAE': np.inf,
                'Val_R2': -np.inf,
                'Val_WAPE': np.inf,
                'Val_WASE': np.inf,
                'Test_MSE': np.inf,
                'Test_MAE': np.inf,
                'Test_RMSE': np.inf,
                'Test_R2': -np.inf,
                'Test_MAPE': np.inf,
                'Test_WAPE': np.inf,
                'Test_WASE': np.inf,
                'Prediction_Time': np.inf
            }
            results.append(failed_result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by Test R²
    results_df = results_df.sort_values('Test_R2', ascending=False)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'baseline_comparison_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\nResults saved to {results_path}")
    
    return results_df


def visualize_results(results_df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Create visualizations of the baseline comparison results.
    
    Args:
        results_df: Results DataFrame
        output_dir: Output directory for plots
    """
    # Filter out failed models
    valid_results = results_df[results_df['Test_R2'] != -np.inf].copy()
    
    if len(valid_results) == 0:
        print("No valid results to visualize")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hierarchical Forecasting Baseline Comparison', fontsize=16)
    
    # Plot 1: Test R² scores
    ax1 = axes[0, 0]
    sns.barplot(data=valid_results.head(10), x='Test_R2', y='Model', ax=ax1)
    ax1.set_title('Test R² Scores (Top 10 Models)')
    ax1.set_xlabel('R² Score')
    
    # Plot 2: Test RMSE
    ax2 = axes[0, 1]
    sns.barplot(data=valid_results.head(10), x='Test_RMSE', y='Model', ax=ax2)
    ax2.set_title('Test RMSE (Top 10 Models)')
    ax2.set_xlabel('RMSE')
    
    # Plot 3: Training Time vs Performance
    ax3 = axes[1, 0]
    scatter_data = valid_results[valid_results['Training_Time'] < 1000]  # Filter extreme outliers
    ax3.scatter(scatter_data['Training_Time'], scatter_data['Test_R2'], 
               s=50, alpha=0.7)
    ax3.set_xlabel('Training Time (seconds)')
    ax3.set_ylabel('Test R²')
    ax3.set_title('Training Time vs Performance')
    
    # Add model names as annotations
    for i, row in scatter_data.iterrows():
        ax3.annotate(row['Model'], (row['Training_Time'], row['Test_R2']), 
                    fontsize=8, rotation=45, ha='left')
    
    # Plot 4: Prediction Time vs Performance
    ax4 = axes[1, 1]
    pred_data = valid_results[valid_results['Prediction_Time'] < 10]  # Filter extreme outliers
    ax4.scatter(pred_data['Prediction_Time'], pred_data['Test_R2'], 
               s=50, alpha=0.7, color='orange')
    ax4.set_xlabel('Prediction Time (seconds)')
    ax4.set_ylabel('Test R²')
    ax4.set_title('Prediction Time vs Performance')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'baseline_comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {plot_path}")
    
    # Create summary table plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Select top 10 models for summary
    summary_data = valid_results.head(10)[['Model', 'Test_R2', 'Test_RMSE', 'Test_MAPE', 'Training_Time']]
    
    # Create table
    table_data = []
    for _, row in summary_data.iterrows():
        table_data.append([
            row['Model'],
            f"{row['Test_R2']:.4f}",
            f"{row['Test_RMSE']:.2f}",
            f"{row['Test_MAPE']:.2f}%",
            f"{row['Training_Time']:.1f}s"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Model', 'Test R²', 'Test RMSE', 'Test MAPE', 'Training Time'],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(5):
            if i == 0:  # Header
                table[(i, j)].set_facecolor('#40466e')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f1f1f2')
                else:
                    table[(i, j)].set_facecolor('white')
    
    ax.axis('off')
    ax.set_title('Top 10 Baseline Models Performance Summary', fontsize=16, pad=20)
    
    # Save summary table
    table_path = os.path.join(output_dir, 'baseline_summary_table.png')
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary table saved to {table_path}")


def main():
    """Main function for baseline comparison."""
    parser = argparse.ArgumentParser(description='Run baseline comparison for hierarchical forecasting')
    parser.add_argument('--data_path', type=str, default='bakery_data_merged.csv',
                       help='Path to the data file')
    parser.add_argument('--output_dir', type=str, default='outputs/baselines',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    parser.add_argument('--patchtst_checkpoint', type=str, default=None,
                       help='Path to a TorchScript/state checkpoint for an official PatchTST model')
    
    args = parser.parse_args()
    
    # Run baseline comparison
    print("Starting baseline comparison...")
    results_df = run_baseline_comparison(
        args.data_path,
        args.output_dir,
        patchtst_checkpoint=args.patchtst_checkpoint,
    )
    
    # Print summary
    separator_width = 120
    print("\n" + "="*separator_width)
    print("BASELINE COMPARISON RESULTS")
    print("="*separator_width)
    print(f"{'Model':<25} {'Test R²':<10} {'Test RMSE':<12} {'Test WAPE':<12} {'Test WASE':<12} {'Training Time':<15}")
    print("-"*separator_width)
    
    for _, row in results_df.head(10).iterrows():
        wape = row.get('Test_WAPE', np.nan)
        wase = row.get('Test_WASE', np.nan)
        wape_str = f"{wape:.2f}%" if not np.isnan(wape) else 'N/A'
        wase_str = f"{wase:.4f}" if not np.isnan(wase) else 'N/A'
        print(f"{row['Model']:<25} {row['Test_R2']:<10.4f} {row['Test_RMSE']:<12.2f} {wape_str:<12} {wase_str:<12} {row['Training_Time']:<15.1f}s")
    
    # Print hierarchical results for top 3 models
    print("\n" + "="*80)
    print("HIERARCHICAL PERFORMANCE COMPARISON (Top 3 Models)")
    print("="*80)
    
    # Check if hierarchical metrics are available
    hierarchical_cols = [col for col in results_df.columns if 'Rank_' in col and ('_WAPE' in col or '_WASE' in col)]
    if hierarchical_cols:
        for i, (_, row) in enumerate(results_df.head(3).iterrows()):
            print(f"\n{row['Model']}:")
            print(f"{'Level':<8} {'WAPE':<10} {'WASE':<12} {'R²':<8} {'MAE':<10} {'RMSE':<10}")
            print("-" * 50)
            
            for rank in range(4):
                wape = row.get(f'Rank_{rank}_WAPE', np.nan)
                wase = row.get(f'Rank_{rank}_WASE', np.nan)
                r2 = row.get(f'Rank_{rank}_R2', np.nan)
                mae = row.get(f'Rank_{rank}_MAE', np.nan)
                rmse = row.get(f'Rank_{rank}_RMSE', np.nan)
                
                if not np.isnan(wape):
                    print(f"Rank {rank:<3} {wape:<10.2f}% {wase:<12.4f} {r2:<8.3f} {mae:<10.2f} {rmse:<10.2f}")
                else:
                    print(f"Rank {rank:<3} {'N/A':<10} {'N/A':<12} {'N/A':<8} {'N/A':<10} {'N/A':<10}")
    else:
        print("Hierarchical metrics not available (entities_test or hierarchy data missing)")

    if args.visualize:
        print("\nCreating visualizations...")
        visualize_results(results_df, args.output_dir)
    
    print(f"\nBaseline comparison completed. Results saved in {args.output_dir}")


if __name__ == "__main__":
    main()
