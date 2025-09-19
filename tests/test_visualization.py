"""
Tests for visualization components.
"""

import unittest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile
import os

# Add hierarchical_forecasting directory to path
sys.path.append(str(Path(__file__).parent.parent))

from hierarchical_forecasting.visualization import (
    HasseDiagramVisualizer, 
    ComplexVisualizer, 
    TrainingVisualizer
)
from hierarchical_forecasting.models import CombinatorialComplex


class TestHasseDiagramVisualizer(unittest.TestCase):
    """Test the HasseDiagramVisualizer class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a simple hierarchy
        self.hierarchy = {
            'sku': ['A1_Store1', 'A2_Store1', 'B1_Store2'],
            'store': ['Store1', 'Store2'],
            'company': ['Company'],
            'total': ['Total']
        }
        
        self.visualizer = HasseDiagramVisualizer(self.hierarchy)
    
    def test_initialization(self):
        """Test visualizer initialization."""
        self.assertIsNotNone(self.visualizer.hierarchy)
        self.assertEqual(len(self.visualizer.hierarchy), 4)
    
    def test_plot_creation(self):
        """Test that plots can be created without errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_hasse.png')
            
            # This should not raise an exception
            try:
                self.visualizer.plot_hasse_diagram(save_path=output_path)
                self.assertTrue(os.path.exists(output_path))
            except Exception as e:
                self.fail(f"Hasse diagram plotting failed: {e}")


class TestComplexVisualizer(unittest.TestCase):
    """Test the ComplexVisualizer class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample complex
        hierarchy = {
            'sku': ['A1', 'A2', 'B1'],
            'store': ['Store1', 'Store2'],
            'company': ['Company'],
            'total': ['Total']
        }
        
        # Create dummy features
        features = pd.DataFrame({
            'entity_id': ['A1', 'A2', 'B1', 'Store1', 'Store2', 'Company', 'Total'],
            'feature_1': np.random.randn(7),
            'feature_2': np.random.randn(7),
        })
        
        targets = pd.DataFrame({
            'entity_id': ['A1', 'A2', 'B1', 'Store1', 'Store2', 'Company', 'Total'],
            'target': np.random.randn(7),
        })
        
        self.complex = CombinatorialComplex(hierarchy, features, targets)
        self.visualizer = ComplexVisualizer(self.complex)
    
    def test_initialization(self):
        """Test visualizer initialization."""
        self.assertIsNotNone(self.visualizer.complex)
    
    def test_plot_structure(self):
        """Test structure plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_structure.png')
            
            try:
                self.visualizer.plot_complex_structure(save_path=output_path)
                self.assertTrue(os.path.exists(output_path))
            except Exception as e:
                self.fail(f"Complex structure plotting failed: {e}")


class TestTrainingVisualizer(unittest.TestCase):
    """Test the TrainingVisualizer class."""
    
    def setUp(self):
        """Set up test data."""
        self.visualizer = TrainingVisualizer()
        
        # Create dummy training history
        self.train_losses = [1.0, 0.8, 0.6, 0.5, 0.4]
        self.val_losses = [1.1, 0.9, 0.7, 0.6, 0.5]
        self.metrics = {
            'train_r2': [0.1, 0.3, 0.5, 0.6, 0.7],
            'val_r2': [0.05, 0.25, 0.45, 0.55, 0.65],
            'train_mae': [2.0, 1.8, 1.5, 1.3, 1.1],
            'val_mae': [2.1, 1.9, 1.6, 1.4, 1.2]
        }
    
    def test_plot_training_history(self):
        """Test training history plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_training.png')
            
            try:
                self.visualizer.plot_training_history(
                    self.train_losses, 
                    self.val_losses, 
                    self.metrics,
                    save_path=output_path
                )
                self.assertTrue(os.path.exists(output_path))
            except Exception as e:
                self.fail(f"Training history plotting failed: {e}")
    
    def test_plot_predictions(self):
        """Test prediction plotting."""
        # Create dummy predictions and targets
        predictions = np.random.randn(100)
        targets = np.random.randn(100)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_predictions.png')
            
            try:
                self.visualizer.plot_predictions_vs_targets(
                    predictions, 
                    targets,
                    save_path=output_path
                )
                self.assertTrue(os.path.exists(output_path))
            except Exception as e:
                self.fail(f"Predictions plotting failed: {e}")


if __name__ == '__main__':
    unittest.main()
