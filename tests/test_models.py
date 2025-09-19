"""
Tests for model components.
"""

import unittest
import torch
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add hierarchical_forecasting directory to path
sys.path.append(str(Path(__file__).parent.parent))

from hierarchical_forecasting.models import CombinatorialComplex, EnhancedCCMPNLayer, EnhancedHierarchicalModel
from hierarchical_forecasting.data.preprocessing import create_dummy_data


class TestCombinatorialComplex(unittest.TestCase):
    """Test the CombinatorialComplex class."""
    
    def setUp(self):
        """Set up test data."""
        self.df = create_dummy_data(n_companies=1, n_stores=3, n_skus=5, n_days=10)
        self.cc = CombinatorialComplex(self.df)
    
    def test_hierarchy_structure(self):
        """Test that hierarchy is built correctly."""
        self.assertEqual(len(self.cc.cells[0]), 15)  # 3 stores * 5 SKUs
        self.assertEqual(len(self.cc.cells[1]), 3)   # 3 stores
        self.assertEqual(len(self.cc.cells[2]), 1)   # 1 company
        self.assertEqual(len(self.cc.cells[3]), 1)   # 1 total
    
    def test_cell_mappings(self):
        """Test cell to integer mappings."""
        self.assertEqual(len(self.cc.cell_to_int), self.cc.num_cells)
        self.assertEqual(len(self.cc.int_to_cell), self.cc.num_cells)
        
        # Test bidirectional mapping
        for cell, idx in self.cc.cell_to_int.items():
            self.assertEqual(self.cc.int_to_cell[idx], cell)
    
    def test_neighborhood_matrices(self):
        """Test neighborhood matrix creation."""
        self.assertIn('incidence_up', self.cc.neighborhoods)
        self.assertIn('adj_down', self.cc.neighborhoods)
        
        # Test matrix dimensions
        for matrix in self.cc.neighborhoods.values():
            self.assertEqual(matrix.size(0), self.cc.num_cells)
            self.assertEqual(matrix.size(1), self.cc.num_cells)


class TestCCMPNLayer(unittest.TestCase):
    """Test the EnhancedCCMPNLayer class."""
    
    def setUp(self):
        """Set up test components."""
        self.hidden_dim = 16
        self.num_cells = 20
        self.num_features = 5
        self.neighborhood_names = ['incidence_up', 'adj_down']
        
        self.layer = EnhancedCCMPNLayer(self.hidden_dim, self.neighborhood_names)
        
        # Create dummy neighborhoods
        self.neighborhoods = {}
        for name in self.neighborhood_names:
            # Create sparse matrix with some edges
            indices = torch.randint(0, self.num_cells, (2, 10))
            values = torch.ones(10)
            self.neighborhoods[name] = torch.sparse_coo_tensor(
                indices, values, (self.num_cells, self.num_cells)
            )
    
    def test_forward_pass(self):
        """Test forward pass of CCMPN layer."""
        h = torch.randn(self.num_cells, self.hidden_dim)
        features = torch.randn(self.num_cells, self.num_features)
        
        output = self.layer(h, features, self.neighborhoods)
        
        self.assertEqual(output.shape, (self.num_cells, self.hidden_dim))
        self.assertFalse(torch.isnan(output).any())


class TestHierarchicalModel(unittest.TestCase):
    """Test the EnhancedHierarchicalModel class."""
    
    def setUp(self):
        """Set up test model."""
        self.num_cells = 20
        self.num_features = 5
        self.hidden_dim = 16
        self.neighborhood_names = ['incidence_up', 'adj_down']
        
        self.model = EnhancedHierarchicalModel(
            self.num_cells, self.num_features, self.hidden_dim,
            self.neighborhood_names, n_layers=2
        )
        
        # Create dummy neighborhoods
        self.neighborhoods = {}
        for name in self.neighborhood_names:
            indices = torch.randint(0, self.num_cells, (2, 10))
            values = torch.ones(10)
            self.neighborhoods[name] = torch.sparse_coo_tensor(
                indices, values, (self.num_cells, self.num_cells)
            )
    
    def test_model_forward(self):
        """Test full model forward pass."""
        x = torch.randn(self.num_cells, self.num_features)
        
        output = self.model(x, self.neighborhoods)
        
        self.assertEqual(output.shape, (self.num_cells, 1))
        self.assertFalse(torch.isnan(output).any())
    
    def test_get_embeddings(self):
        """Test embedding extraction."""
        x = torch.randn(self.num_cells, self.num_features)
        
        embeddings = self.model.get_embeddings(x, self.neighborhoods)
        
        self.assertEqual(embeddings.shape, (self.num_cells, self.hidden_dim))
        self.assertFalse(torch.isnan(embeddings).any())


if __name__ == '__main__':
    unittest.main()
