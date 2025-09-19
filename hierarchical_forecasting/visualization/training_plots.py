"""
Training and evaluation visualization utilities.

This module provides functionality to visualize training progress,
model performance, and evaluation results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Any
import torch


class TrainingVisualizer:
    """
    Visualizes training progress and model performance.
    """
    
    def __init__(self, output_dir: str = "outputs/plots"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
    
    def plot_training_curves(self, train_losses: List[float], val_losses: List[float] = None):
        """Plot training and validation loss curves."""
        print("📈 Creating training curves...")
        
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(12, 6))
        
        if val_losses is not None:
            plt.subplot(1, 2, 1)
        
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if val_losses is not None:
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if val_losses is not None:
            # Plot learning rate schedule if available
            plt.subplot(1, 2, 2)
            # Placeholder for learning rate plot
            plt.plot(epochs, [0.005] * len(epochs), 'g-', label='Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/training_curves.png', dpi=300, bbox_inches='tight')
        print(f"✅ Training curves saved to {self.output_dir}/training_curves.png")
        plt.show()
    
    def plot_evaluation_results(self, results_df: pd.DataFrame):
        """Plot evaluation results by hierarchy level."""
        print("📊 Creating evaluation results plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract numeric values from percentage strings
        def extract_numeric(value_str):
            if value_str == 'N/A':
                return np.nan
            return float(value_str.rstrip('%'))
        
        # WAPE by level
        ax1 = axes[0, 0]
        wape_values = [extract_numeric(val) for val in results_df['WAPE']]
        levels = results_df['Level']
        
        bars = ax1.bar(levels, wape_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('WAPE by Hierarchy Level', fontweight='bold')
        ax1.set_ylabel('WAPE (%)')
        ax1.set_xlabel('Hierarchy Level')
        
        # Add value labels
        for bar, val in zip(bars, wape_values):
            if not np.isnan(val):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # R² by level
        ax2 = axes[0, 1]
        r2_values = [float(val) if val != 'N/A' else np.nan for val in results_df['R²']]
        
        bars = ax2.bar(levels, r2_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_title('R² by Hierarchy Level', fontweight='bold')
        ax2.set_ylabel('R² Score')
        ax2.set_xlabel('Hierarchy Level')
        
        # Add value labels
        for bar, val in zip(bars, r2_values):
            if not np.isnan(val):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # MAE by level (if available)
        if 'MAE' in results_df.columns:
            ax3 = axes[1, 0]
            mae_values = [float(val) if val != 'N/A' else np.nan for val in results_df['MAE']]
            
            bars = ax3.bar(levels, mae_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax3.set_title('MAE by Hierarchy Level', fontweight='bold')
            ax3.set_ylabel('MAE')
            ax3.set_xlabel('Hierarchy Level')
            
            # Add value labels
            for bar, val in zip(bars, mae_values):
                if not np.isnan(val):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        else:
            axes[1, 0].axis('off')
        
        # Summary table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary table
        table_data = results_df.values
        table = ax4.table(cellText=table_data, colLabels=results_df.columns,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(results_df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Performance Summary', fontweight='bold', pad=20)
        
        plt.suptitle('Model Evaluation Results by Hierarchy Level', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(f'{self.output_dir}/evaluation_results.png', dpi=300, bbox_inches='tight')
        print(f"✅ Evaluation results saved to {self.output_dir}/evaluation_results.png")
        plt.show()
    
    def plot_predictions_vs_actuals(self, predictions: np.ndarray, actuals: np.ndarray, 
                                   level_name: str = "SKU Level"):
        """Plot predictions vs actuals scatter plot."""
        print(f"📈 Creating predictions vs actuals plot for {level_name}...")
        
        plt.figure(figsize=(10, 8))
        
        # Remove NaN values
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        pred_clean = predictions[mask]
        actual_clean = actuals[mask]
        
        # Create scatter plot
        plt.scatter(actual_clean, pred_clean, alpha=0.6, s=20)
        
        # Add perfect prediction line
        min_val = min(actual_clean.min(), pred_clean.min())
        max_val = max(actual_clean.max(), pred_clean.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                linewidth=2, label='Perfect Prediction')
        
        # Calculate and display R²
        if len(pred_clean) > 0:
            from sklearn.metrics import r2_score
            r2 = r2_score(actual_clean, pred_clean)
            plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predictions vs Actuals - {level_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Make axes equal
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/predictions_vs_actuals_{level_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"✅ Predictions vs actuals plot saved")
        plt.show()
    
    def plot_residuals(self, predictions: np.ndarray, actuals: np.ndarray,
                      level_name: str = "SKU Level"):
        """Plot residual analysis."""
        print(f"📊 Creating residual analysis for {level_name}...")
        
        # Calculate residuals
        residuals = actuals - predictions
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Residuals vs predictions
        ax1 = axes[0, 0]
        ax1.scatter(predictions, residuals, alpha=0.6, s=20)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predictions')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax2 = axes[0, 1]
        ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        ax2.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax3 = axes[1, 0]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normal Distribution)')
        ax3.grid(True, alpha=0.3)
        
        # Residuals vs actuals
        ax4 = axes[1, 1]
        ax4.scatter(actuals, residuals, alpha=0.6, s=20)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('Actual Values')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals vs Actuals')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Residual Analysis - {level_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(f'{self.output_dir}/residual_analysis_{level_name.lower().replace(" ", "_")}.png',
                   dpi=300, bbox_inches='tight')
        print(f"✅ Residual analysis saved")
        plt.show()
    
    def plot_all_training_visuals(self, train_losses: List[float], 
                                 val_losses: List[float] = None,
                                 results_df: pd.DataFrame = None):
        """Create all training visualizations."""
        print("📊 Creating all training visualizations...")
        self.plot_training_curves(train_losses, val_losses)
        if results_df is not None:
            self.plot_evaluation_results(results_df)
        print("✅ All training visualizations completed!")
