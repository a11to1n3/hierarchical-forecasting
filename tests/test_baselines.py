"""
Tests for baseline models.
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add hierarchical_forecasting directory to path
sys.path.append(str(Path(__file__).parent.parent))

from hierarchical_forecasting.baselines import (
    LinearRegressionBaseline,
    RandomForestBaseline,
    LSTMBaseline,
    BottomUpBaseline,
    TopDownBaseline,
    MinTBaseline
)


class TestLinearRegressionBaseline(unittest.TestCase):
    """Test the LinearRegressionBaseline class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)
        self.baseline = LinearRegressionBaseline()
    
    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        # Fit the model
        self.baseline.fit(self.X, self.y)
        self.assertTrue(self.baseline.is_fitted)
        
        # Make predictions
        predictions = self.baseline.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        self.assertIsInstance(predictions, np.ndarray)
    
    def test_score(self):
        """Test scoring functionality."""
        self.baseline.fit(self.X, self.y)
        scores = self.baseline.score(self.X, self.y)
        
        # Check that all expected metrics are present
        expected_metrics = ['mse', 'mae', 'rmse', 'r2', 'mape']
        for metric in expected_metrics:
            self.assertIn(metric, scores)
            self.assertIsInstance(scores[metric], (int, float))


class TestRandomForestBaseline(unittest.TestCase):
    """Test the RandomForestBaseline class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)
        self.baseline = RandomForestBaseline(n_estimators=10)  # Small for testing
    
    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        self.baseline.fit(self.X, self.y)
        self.assertTrue(self.baseline.is_fitted)
        
        predictions = self.baseline.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        self.baseline.fit(self.X, self.y)
        importance = self.baseline.get_feature_importance()
        
        self.assertEqual(len(importance), self.X.shape[1])
        self.assertTrue(np.all(importance >= 0))
    
    def test_uncertainty_prediction(self):
        """Test prediction with uncertainty."""
        self.baseline.fit(self.X, self.y)
        predictions, std_predictions = self.baseline.predict_with_uncertainty(self.X)
        
        self.assertEqual(len(predictions), len(self.y))
        self.assertEqual(len(std_predictions), len(self.y))
        self.assertTrue(np.all(std_predictions >= 0))


class TestLSTMBaseline(unittest.TestCase):
    """Test the LSTMBaseline class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.random.randn(50, 5)  # Smaller dataset for faster testing
        self.y = np.random.randn(50)
        self.baseline = LSTMBaseline(
            input_size=5, 
            hidden_size=8, 
            epochs=2,  # Few epochs for testing
            device='cpu'  # Force CPU for testing
        )
    
    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        self.baseline.fit(self.X, self.y, sequence_length=5)
        self.assertTrue(self.baseline.is_fitted)
        
        predictions = self.baseline.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))


class TestHierarchicalBaselines(unittest.TestCase):
    """Test hierarchical baseline models."""
    
    def setUp(self):
        """Set up hierarchical test data."""
        np.random.seed(42)
        
        # Create hierarchical data structure
        self.hierarchy = {
            'sku': ['A1', 'A2', 'B1', 'B2'],
            'store': ['Store1', 'Store2'],
            'company': ['Company'],
            'total': ['Total']
        }
        
        # Create features and targets
        n_samples = 40  # 10 samples per level
        self.X = np.random.randn(n_samples, 5)
        self.y = np.random.randn(n_samples)
        
        # Create entity levels (0=SKU, 1=Store, 2=Company, 3=Total)
        self.entity_levels = np.array([0, 0, 0, 0, 1, 1, 2, 3] * 5)[:n_samples]
    
    def test_bottom_up_baseline(self):
        """Test Bottom-Up baseline."""
        baseline = BottomUpBaseline(base_model='linear')
        
        baseline.fit(self.X, self.y, 
                    hierarchy=self.hierarchy, 
                    entity_levels=self.entity_levels)
        self.assertTrue(baseline.is_fitted)
        
        predictions = baseline.predict(self.X, entity_levels=self.entity_levels)
        self.assertEqual(len(predictions), len(self.y))
    
    def test_top_down_baseline(self):
        """Test Top-Down baseline."""
        baseline = TopDownBaseline(base_model='linear')
        
        baseline.fit(self.X, self.y,
                    hierarchy=self.hierarchy,
                    entity_levels=self.entity_levels)
        self.assertTrue(baseline.is_fitted)
        
        predictions = baseline.predict(self.X, entity_levels=self.entity_levels)
        self.assertEqual(len(predictions), len(self.y))
    
    def test_mint_baseline(self):
        """Test MinT baseline."""
        baseline = MinTBaseline(base_model='linear')
        
        baseline.fit(self.X, self.y,
                    hierarchy=self.hierarchy,
                    entity_levels=self.entity_levels)
        self.assertTrue(baseline.is_fitted)
        
        predictions = baseline.predict(self.X, entity_levels=self.entity_levels)
        self.assertEqual(len(predictions), len(self.y))


class TestBaselineComparison(unittest.TestCase):
    """Test baseline comparison functionality."""
    
    def setUp(self):
        """Set up test data for comparison."""
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)
        
        self.baselines = [
            LinearRegressionBaseline(),
            RandomForestBaseline(n_estimators=10),
            LSTMBaseline(input_size=5, hidden_size=8, epochs=2, device='cpu')
        ]
    
    def test_baseline_comparison(self):
        """Test that all baselines can be trained and evaluated."""
        results = []
        
        for baseline in self.baselines:
            # Train baseline
            baseline.fit(self.X, self.y)
            
            # Evaluate baseline
            scores = baseline.score(self.X, self.y)
            
            # Store results
            result = {
                'model': baseline.name,
                'r2': scores['r2'],
                'mse': scores['mse'],
                'mae': scores['mae']
            }
            results.append(result)
        
        # Check that we have results for all baselines
        self.assertEqual(len(results), len(self.baselines))
        
        # Check that all metrics are computed
        for result in results:
            self.assertIn('r2', result)
            self.assertIn('mse', result)
            self.assertIn('mae', result)
            
            # Check that metrics are reasonable
            self.assertIsInstance(result['r2'], (int, float))
            self.assertIsInstance(result['mse'], (int, float))
            self.assertIsInstance(result['mae'], (int, float))
            
            self.assertGreaterEqual(result['mse'], 0)
            self.assertGreaterEqual(result['mae'], 0)


if __name__ == '__main__':
    unittest.main()
