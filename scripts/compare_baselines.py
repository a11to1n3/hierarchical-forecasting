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
from typing import Any, Dict, List, Tuple, Optional
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
    weighted_percentage_error,
)

try:
    from hierarchical_forecasting.baselines import ProphetBaseline, HierarchicalProphet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

TRANSFORMER_WINDOW = 64


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
        'LSTM': LSTMBaseline(input_size=input_size, hidden_size=32, epochs=120),
        'Multi_Entity_LSTM': MultiEntityLSTM(input_size=input_size, hidden_size=32, epochs=120),
        
        # ETNN-based baselines (Topological Deep Learning)
        'ETNN': ETNNBaseline(hidden_dim=32, num_layers=2, epochs=120),
    }

    if patchtst_checkpoint:
        baselines['PatchTST'] = PatchTSTBaseline(checkpoint_path=patchtst_checkpoint)
    else:
        print("No PatchTST checkpoint provided; training a fresh PatchTST baseline from scratch.")
    baselines['PatchTST'] = PatchTSTBaseline(allow_training=True, epochs=240)

    baselines['TimesNet'] = TimesNetBaseline(epochs=240)

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


def evaluate_hierarchical_metrics(
    *,
    y_true_original: np.ndarray,
    y_pred_original: np.ndarray,
    entities_test=None,
    hierarchy=None,
    hierarchy_levels: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Evaluate prediction quality at each hierarchy level using original-scale targets."""

    try:
        y_true = np.asarray(y_true_original)
        y_pred = np.asarray(y_pred_original)
        if y_true.ndim > 1:
            y_true = y_true.reshape(-1)
        if y_pred.ndim > 1:
            y_pred = y_pred.reshape(-1)
        results: Dict[str, float] = {}

        results['Overall_WAPE'] = weighted_absolute_percentage_error(y_true, y_pred)
        results['Overall_WPE'] = weighted_percentage_error(y_true, y_pred)
        results['Overall_MAE'] = mean_absolute_error(y_true, y_pred)
        results['Overall_RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        results['Overall_R2'] = r2_score(y_true, y_pred)

        if hierarchy_levels is not None:
            unique_levels = sorted(set(hierarchy_levels))
            for level in unique_levels:
                mask = [idx for idx, lvl in enumerate(hierarchy_levels) if lvl == level]
                if not mask:
                    continue
                level_true = y_true[mask]
                level_pred = y_pred[mask]
                results[f'Level_{level}_WAPE'] = weighted_absolute_percentage_error(level_true, level_pred)
                results[f'Level_{level}_WPE'] = weighted_percentage_error(level_true, level_pred)
                results[f'Level_{level}_MAE'] = mean_absolute_error(level_true, level_pred)
                results[f'Level_{level}_RMSE'] = np.sqrt(mean_squared_error(level_true, level_pred))
                results[f'Level_{level}_R2'] = r2_score(level_true, level_pred)

        if entities_test is not None and hierarchy is not None:
            for rank in range(4):
                if rank == 0:
                    mask = np.array([entity in hierarchy[0] for entity in entities_test])
                    if mask.sum() > 0:
                        level_pred = y_pred[mask]
                        level_true = y_true[mask]
                    else:
                        continue

                elif rank == 1:
                    store_entities = hierarchy[1]
                    level_pred_list = []
                    level_true_list = []
                    for store in store_entities:
                        sku_mask = np.array([entity[:2] == store for entity in entities_test])
                        if sku_mask.sum() > 0:
                            level_pred_list.append(y_pred[sku_mask].sum())
                            level_true_list.append(y_true[sku_mask].sum())
                    if not level_pred_list:
                        continue
                    level_pred = np.array(level_pred_list)
                    level_true = np.array(level_true_list)

                elif rank == 2:
                    company_entities = hierarchy[2]
                    level_pred_list = []
                    level_true_list = []
                    for company in company_entities:
                        sku_mask = np.array([entity[0] == company[0] for entity in entities_test])
                        if sku_mask.sum() > 0:
                            level_pred_list.append(y_pred[sku_mask].sum())
                            level_true_list.append(y_true[sku_mask].sum())
                    if not level_pred_list:
                        continue
                    level_pred = np.array(level_pred_list)
                    level_true = np.array(level_true_list)

                else:
                    level_pred = np.array([y_pred.sum()])
                    level_true = np.array([y_true.sum()])

                wape = weighted_absolute_percentage_error(level_true, level_pred)
                wpe = weighted_percentage_error(level_true, level_pred)

                nonzero_mask = np.abs(level_true) > 0
                if nonzero_mask.sum() > 0:
                    filtered_true = level_true[nonzero_mask]
                    filtered_pred = level_pred[nonzero_mask]
                    r2 = r2_score(filtered_true, filtered_pred)
                    mae = mean_absolute_error(filtered_true, filtered_pred)
                    rmse = np.sqrt(mean_squared_error(filtered_true, filtered_pred))
                else:
                    r2 = mae = rmse = np.nan

                results[f'Rank_{rank}_WAPE'] = wape
                results[f'Rank_{rank}_WPE'] = wpe
                results[f'Rank_{rank}_R2'] = r2
                results[f'Rank_{rank}_MAE'] = mae
                results[f'Rank_{rank}_RMSE'] = rmse

        return results

    except Exception as e:
        print(f"Error in hierarchical evaluation: {e}")
        results: Dict[str, float] = {}
        for rank in range(4):
            results[f'Rank_{rank}_WAPE'] = np.nan
            results[f'Rank_{rank}_WPE'] = np.nan
            results[f'Rank_{rank}_R2'] = np.nan
            results[f'Rank_{rank}_MAE'] = np.nan
            results[f'Rank_{rank}_RMSE'] = np.nan
        return results


def evaluate_model(
    model,
    X_test,
    y_test,
    entities_test=None,
    hierarchy=None,
    *,
    y_true_original: Optional[np.ndarray] = None,
    inverse_target_fn: Optional[Any] = None,
    predictions_override: Optional[np.ndarray] = None,
    **kwargs,
):
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
        predictions = predictions_override if predictions_override is not None else model.predict(X_test, **kwargs)
        prediction_time = time.time() - start_time
        
        if y_true_original is not None:
            y_true_for_metrics = y_true_original
        elif inverse_target_fn is not None:
            y_true_for_metrics = inverse_target_fn(y_test)
        else:
            y_true_for_metrics = y_test

        if inverse_target_fn is not None:
            y_pred_for_metrics = inverse_target_fn(predictions)
        else:
            y_pred_for_metrics = predictions

        mse = mean_squared_error(y_true_for_metrics, y_pred_for_metrics)
        mae = mean_absolute_error(y_true_for_metrics, y_pred_for_metrics)
        rmse = np.sqrt(mse)

        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        mape = np.mean(np.abs((y_test - predictions) / np.where(y_test == 0, 1, y_test))) * 100
        wape = weighted_absolute_percentage_error(y_true_for_metrics, y_pred_for_metrics)
        wpe = weighted_percentage_error(y_true_for_metrics, y_pred_for_metrics)

        results = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'WAPE': wape,
            'WPE': wpe,
            'Prediction_Time': prediction_time
        }

        if entities_test is not None and hierarchy is not None:
            levels_for_hier = kwargs.get('entity_levels')
            hier_metrics = evaluate_hierarchical_metrics(
                y_true_original=y_true_for_metrics,
                y_pred_original=y_pred_for_metrics,
                entities_test=entities_test,
                hierarchy=hierarchy,
                hierarchy_levels=levels_for_hier,
            )
            results.update(hier_metrics)

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
            'WPE': np.inf,
            'Prediction_Time': np.inf
        }

        if entities_test is not None and hierarchy is not None:
            for rank in range(4):
                results[f'Rank_{rank}_WAPE'] = np.nan
                results[f'Rank_{rank}_WPE'] = np.nan
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

    def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=['date', 'companyID', 'storeID', 'skuID', 'target'])

        if 'date' in df.columns:
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            return df_copy

        if isinstance(df.index, pd.DatetimeIndex):
            df_copy = df.reset_index()
            if 'index' in df_copy.columns and 'date' not in df_copy.columns:
                df_copy = df_copy.rename(columns={'index': 'date'})
            if df.index.name and df.index.name != 'date' and df.index.name in df_copy.columns:
                df_copy = df_copy.rename(columns={df.index.name: 'date'})
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            return df_copy

        df_copy = df.copy()
        if 'date' in df_copy.columns:
            df_copy['date'] = pd.to_datetime(df_copy['date'])
        else:
            df_copy['date'] = pd.to_datetime(df_copy.index)
        return df_copy

    def build_transformer_dataset(current_df: pd.DataFrame,
                                  history_df: Optional[pd.DataFrame] = None,
                                  window_size: int = TRANSFORMER_WINDOW) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        current_df = _ensure_date_column(current_df)
        if current_df.empty or window_size <= 0:
            return (np.empty((0, window_size), dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=object))

        required_cols = {'companyID', 'storeID', 'skuID', 'target', 'date'}
        if not required_cols.issubset(current_df.columns):
            raise ValueError(f"Missing required columns for transformer dataset: {required_cols - set(current_df.columns)}")

        frames = []
        if history_df is not None and not history_df.empty:
            history_df = _ensure_date_column(history_df)
            frames.append(history_df[['date', 'companyID', 'storeID', 'skuID', 'target']].assign(__is_current=False))

        frames.append(current_df[['date', 'companyID', 'storeID', 'skuID', 'target']].assign(__is_current=True))
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(['companyID', 'storeID', 'skuID', 'date'])

        sequences: List[np.ndarray] = []
        targets_seq: List[float] = []
        entities_seq: List[Tuple] = []

        for entity, group in combined.groupby(['companyID', 'storeID', 'skuID'], sort=False):
            values = group['target'].astype(np.float32).to_numpy()
            mask = group['__is_current'].to_numpy(dtype=bool)
            if len(values) <= window_size:
                continue

            for idx in range(window_size, len(values)):
                if not mask[idx]:
                    continue
                window = values[idx - window_size:idx]
                if np.isnan(window).any():
                    continue
                sequences.append(window)
                targets_seq.append(values[idx])
                entities_seq.append(entity)

        if not sequences:
            return (np.empty((0, window_size), dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=object))

        X_seq = np.stack(sequences)
        y_seq = np.asarray(targets_seq, dtype=np.float32)
        entities_arr = np.asarray(entities_seq, dtype=object)
        return X_seq, y_seq, entities_arr

    X_train, y_train, levels_train, entities_train, dates_train = prepare_split_data(train_data)
    X_val, y_val, levels_val, entities_val, dates_val = prepare_split_data(val_data)
    X_test, y_test, levels_test, entities_test, dates_test = prepare_split_data(test_data)

    def inverse_target(values: Any) -> np.ndarray:
        arr = np.asarray(values).reshape(-1, 1)
        return preprocessor.inverse_transform_target(arr).ravel()

    y_train_original = inverse_target(y_train)
    y_val_original = inverse_target(y_val)
    y_test_original = inverse_target(y_test)

    transformer_train_X, transformer_train_y, transformer_train_entities = build_transformer_dataset(
        train_data, window_size=TRANSFORMER_WINDOW
    )
    transformer_val_X, transformer_val_y, transformer_val_entities = build_transformer_dataset(
        val_data, history_df=train_data, window_size=TRANSFORMER_WINDOW
    )
    history_for_test = pd.concat([train_data, val_data]) if not train_data.empty or not val_data.empty else train_data
    transformer_test_X, transformer_test_y, transformer_test_entities = build_transformer_dataset(
        test_data, history_df=history_for_test, window_size=TRANSFORMER_WINDOW
    )

    transformer_train_y_original = inverse_target(transformer_train_y) if transformer_train_y.size else transformer_train_y
    transformer_val_y_original = inverse_target(transformer_val_y) if transformer_val_y.size else transformer_val_y
    transformer_test_y_original = inverse_target(transformer_test_y) if transformer_test_y.size else transformer_test_y

    print(f"Feature dimensions: {X_train.shape}")

    def default_metric_dict() -> Dict[str, float]:
        return {
            'MSE': np.nan,
            'MAE': np.nan,
            'RMSE': np.nan,
            'R2': np.nan,
            'MAPE': np.nan,
            'WAPE': np.nan,
            'WPE': np.nan,
            'Prediction_Time': np.nan
        }

    # Create baseline models with the correct input dimensionality
    input_size = X_train.shape[1]
    baselines = create_baseline_models(
        input_size,
        patchtst_checkpoint=patchtst_checkpoint,
    )
    
    results = []
    
    print(f"\nTraining and evaluating {len(baselines)} baseline models...")
    
    transformer_types = (PatchTSTBaseline, TimesNetBaseline)

    for name, model in baselines.items():
        print(f"\nTraining {name}...")

        try:
            start_time = time.time()

            is_transformer = isinstance(model, transformer_types)
            if is_transformer:
                X_train_model = transformer_train_X
                y_train_model = transformer_train_y
                entities_train_model = transformer_train_entities
            else:
                X_train_model = X_train
                y_train_model = y_train
                entities_train_model = entities_train

            if X_train_model.shape[0] == 0:
                print(f"Skipping {name}: insufficient training data after preprocessing.")
                skipped_result = {
                    'Model': name,
                    'Training_Time': np.nan,
                    'Val_MSE': np.nan,
                    'Val_MAE': np.nan,
                    'Val_R2': np.nan,
                    'Val_WAPE': np.nan,
                    'Val_WPE': np.nan,
                    'Test_MSE': np.nan,
                    'Test_MAE': np.nan,
                    'Test_RMSE': np.nan,
                    'Test_R2': np.nan,
                    'Test_MAPE': np.nan,
                    'Test_WAPE': np.nan,
                    'Test_WPE': np.nan,
                    'Prediction_Time': np.nan
                }
                results.append(skipped_result)
                continue

            train_kwargs: Dict[str, Any] = {}
            if not is_transformer:
                train_kwargs.update({
                    'hierarchy': hierarchy,
                    'entity_levels': levels_train,
                    'entity_ids': entities_train
                })

            if 'Prophet' in name:
                train_kwargs['dates'] = dates_train
                train_kwargs['entity_ids'] = entities_train

            model.fit(X_train_model, y_train_model, **train_kwargs)
            training_time = time.time() - start_time

            print(f"Training completed in {training_time:.2f} seconds")

            if is_transformer:
                X_val_model = transformer_val_X
                y_val_model = transformer_val_y
                entities_val_model = transformer_val_entities
                X_test_model = transformer_test_X
                y_test_model = transformer_test_y
                entities_test_model = transformer_test_entities
                val_kwargs: Dict[str, Any] = {}
                test_kwargs: Dict[str, Any] = {}
                y_val_original_for_metrics = transformer_val_y_original
                y_test_original_for_metrics = transformer_test_y_original
            else:
                X_val_model = X_val
                y_val_model = y_val
                entities_val_model = entities_val
                X_test_model = X_test
                y_test_model = y_test
                entities_test_model = entities_test
                val_kwargs = {
                    'entity_levels': levels_val,
                    'entity_ids': entities_val
                }
                test_kwargs = {
                    'entity_levels': levels_test,
                    'entity_ids': entities_test
                }
                y_val_original_for_metrics = y_val_original
                y_test_original_for_metrics = y_test_original

            if not is_transformer and 'Prophet' in name:
                val_kwargs['dates'] = dates_val
                test_kwargs['dates'] = dates_test

            if X_val_model.shape[0] == 0 or y_val_model.size == 0:
                val_metrics = default_metric_dict()
            else:
                val_entities_arg = entities_val_model if len(entities_val_model) > 0 else None
                val_hierarchy_arg = hierarchy if val_entities_arg is not None else None
                val_metrics = evaluate_model(
                    model,
                    X_val_model,
                    y_val_model,
                    y_true_original=y_val_original_for_metrics if y_val_model.size else None,
                    inverse_target_fn=inverse_target,
                    entities_test=val_entities_arg,
                    hierarchy=val_hierarchy_arg,
                    **val_kwargs
                )

            if X_test_model.shape[0] == 0 or y_test_model.size == 0:
                test_metrics = default_metric_dict()
            else:
                test_entities_arg = entities_test_model if len(entities_test_model) > 0 else None
                test_hierarchy_arg = hierarchy if test_entities_arg is not None else None
                test_metrics = evaluate_model(
                    model,
                    X_test_model,
                    y_test_model,
                    y_true_original=y_test_original_for_metrics if y_test_model.size else None,
                    inverse_target_fn=inverse_target,
                    entities_test=test_entities_arg,
                    hierarchy=test_hierarchy_arg,
                    **test_kwargs
                )

            result = {
                'Model': name,
                'Training_Time': training_time,
                'Val_MSE': val_metrics['MSE'],
                'Val_MAE': val_metrics['MAE'],
                'Val_R2': val_metrics['R2'],
                'Val_WAPE': val_metrics.get('WAPE', np.nan),
                'Val_WPE': val_metrics.get('WPE', np.nan),
                'Test_MSE': test_metrics['MSE'],
                'Test_MAE': test_metrics['MAE'],
                'Test_RMSE': test_metrics['RMSE'],
                'Test_R2': test_metrics['R2'],
                'Test_MAPE': test_metrics['MAPE'],
                'Test_WAPE': test_metrics.get('WAPE', np.nan),
                'Test_WPE': test_metrics.get('WPE', np.nan),
                'Prediction_Time': test_metrics['Prediction_Time']
            }

            results.append(result)

            print(
                f"Validation R²: {val_metrics['R2']:.4f}, "
                f"Test R²: {test_metrics['R2']:.4f}"
            )

        except Exception as e:
            print(f"Error with {name}: {e}")
            failed_result = {
                'Model': name,
                'Training_Time': np.inf,
                'Val_MSE': np.inf,
                'Val_MAE': np.inf,
                'Val_R2': -np.inf,
                'Val_WAPE': np.inf,
                'Val_WPE': np.inf,
                'Test_MSE': np.inf,
                'Test_MAE': np.inf,
                'Test_RMSE': np.inf,
                'Test_R2': -np.inf,
                'Test_MAPE': np.inf,
                'Test_WAPE': np.inf,
                'Test_WPE': np.inf,
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
    print(f"{'Model':<25} {'Test R²':<10} {'Test RMSE':<12} {'Test WAPE':<12} {'Test WPE':<12} {'Training Time':<15}")
    print("-"*separator_width)
    
    for _, row in results_df.head(10).iterrows():
        wape = row.get('Test_WAPE', np.nan)
        wpe_value = row.get('Test_WPE', np.nan)
        wape_str = f"{wape:.2f}%" if not np.isnan(wape) else 'N/A'
        wpe_str = f"{wpe_value:.4f}" if not np.isnan(wpe_value) else 'N/A'
        print(f"{row['Model']:<25} {row['Test_R2']:<10.4f} {row['Test_RMSE']:<12.2f} {wape_str:<12} {wpe_str:<12} {row['Training_Time']:<15.1f}s")
    
    # Print hierarchical results for top 3 models
    print("\n" + "="*80)
    print("HIERARCHICAL PERFORMANCE COMPARISON (Top 3 Models)")
    print("="*80)
    
    # Check if hierarchical metrics are available
    hierarchical_cols = [col for col in results_df.columns if 'Rank_' in col and ('_WAPE' in col or '_WPE' in col)]
    if hierarchical_cols:
        for i, (_, row) in enumerate(results_df.head(3).iterrows()):
            print(f"\n{row['Model']}:")
            print(f"{'Level':<8} {'WAPE':<10} {'WPE':<12} {'R²':<8} {'MAE':<10} {'RMSE':<10}")
            print("-" * 50)
            
            for rank in range(4):
                wape = row.get(f'Rank_{rank}_WAPE', np.nan)
                wpe_val = row.get(f'Rank_{rank}_WPE', np.nan)
                r2 = row.get(f'Rank_{rank}_R2', np.nan)
                mae = row.get(f'Rank_{rank}_MAE', np.nan)
                rmse = row.get(f'Rank_{rank}_RMSE', np.nan)
                
                if not np.isnan(wape):
                    print(f"Rank {rank:<3} {wape:<10.2f}% {wpe_val:<12.4f} {r2:<8.3f} {mae:<10.2f} {rmse:<10.2f}")
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
