"""
Tests for data processing components.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add hierarchical_forecasting directory to path
sys.path.append(str(Path(__file__).parent.parent))

from hierarchical_forecasting.data import DataPreprocessor, HierarchicalDataLoader


class TestDataPreprocessor(unittest.TestCase):
    """Test the DataPreprocessor class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        stores = ['Store_A', 'Store_B', 'Store_C']
        skus = ['SKU_1', 'SKU_2', 'SKU_3', 'SKU_4']
        
        data = []
        for date in dates:
            for store in stores:
                for sku in skus:
                    data.append({
                        'date': date,
                        'store': store,
                        'sku': sku,
                        'sales': np.random.poisson(10),
                        'price': np.random.uniform(5, 20),
                        'promotion': np.random.choice([0, 1], p=[0.8, 0.2])
                    })
        
        self.test_data = pd.DataFrame(data)
        self.preprocessor = DataPreprocessor()
    
    def test_create_hierarchy(self):
        """Test hierarchy creation."""
        hierarchy = self.preprocessor.create_hierarchy(self.test_data)
        
        # Check that all levels are present
        expected_levels = ['sku', 'store', 'company', 'total']
        for level in expected_levels:
            self.assertIn(level, hierarchy)
        
        # Check that hierarchy is properly structured
        self.assertGreater(len(hierarchy['sku']), 0)
        self.assertEqual(len(hierarchy['total']), 1)
    
    def test_create_features(self):
        """Test feature creation."""
        features = self.preprocessor.create_features(self.test_data)
        
        # Check that features are created
        self.assertGreater(len(features.columns), len(self.test_data.columns))
        
        # Check for lag features
        lag_columns = [col for col in features.columns if 'lag' in col.lower()]
        self.assertGreater(len(lag_columns), 0)
    
    def test_split_data(self):
        """Test data splitting."""
        train, val, test = self.preprocessor.split_data(self.test_data)
        
        # Check that splits are reasonable
        total_len = len(self.test_data)
        self.assertLess(len(train), total_len)
        self.assertLess(len(val), total_len)
        self.assertLess(len(test), total_len)
        
        # Check that splits don't overlap in time
        train_max_date = train['date'].max()
        val_min_date = val['date'].min()
        test_min_date = test['date'].min()
        
        self.assertLessEqual(train_max_date, val_min_date)
        self.assertLessEqual(val['date'].max(), test_min_date)


class TestHierarchicalDataLoader(unittest.TestCase):
    """Test the HierarchicalDataLoader class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample preprocessed data
        self.features = pd.DataFrame({
            'entity_id': ['A', 'B', 'C', 'D'],
            'feature_1': [1.0, 2.0, 3.0, 4.0],
            'feature_2': [0.5, 1.5, 2.5, 3.5],
        })
        
        self.targets = pd.DataFrame({
            'entity_id': ['A', 'B', 'C', 'D'],
            'target': [10.0, 20.0, 30.0, 40.0],
        })
        
        self.hierarchy = {
            'sku': ['A', 'B'],
            'store': ['C'],
            'company': ['D'],
        }
        
        self.loader = HierarchicalDataLoader(
            self.features, 
            self.targets, 
            self.hierarchy,
            batch_size=2
        )
    
    def test_initialization(self):
        """Test loader initialization."""
        self.assertEqual(len(self.loader), 2)  # 4 samples, batch_size=2
        self.assertEqual(self.loader.num_features, 2)
        self.assertEqual(self.loader.num_entities, 4)
    
    def test_batch_generation(self):
        """Test batch generation."""
        batch = next(iter(self.loader))
        
        self.assertIn('features', batch)
        self.assertIn('targets', batch)
        self.assertIn('entity_ids', batch)
        
        # Check tensor shapes
        self.assertEqual(batch['features'].shape[1], 2)  # 2 features
        self.assertEqual(len(batch['targets'].shape), 1)  # 1D targets


if __name__ == '__main__':
    unittest.main()
