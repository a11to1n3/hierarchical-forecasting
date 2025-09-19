"""
Combinatorial Complex implementation for hierarchical sales data.

This module implements the mathematical structure representing the sales hierarchy
as a combinatorial complex with covering relations and neighborhood matrices.
"""

import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Any


class CombinatorialComplex:
    """
    Represents the sales hierarchy as a combinatorial complex and builds 
    multiple neighborhood matrices for message passing.
    
    The complex represents a 4-level hierarchy:
    - Rank 0: SKUs (individual products in stores)
    - Rank 1: Stores (aggregation of SKUs)
    - Rank 2: Companies (aggregation of stores)
    - Rank 3: Total (global aggregation)
    """
    
    def __init__(self, df):
        """
        Initialize the combinatorial complex from sales data.
        
        Args:
            df: DataFrame with columns ['companyID', 'storeID', 'skuID', ...]
        """
        print("🏗️ Building Enhanced Combinatorial Complex structure...")
        
        self.cells = {0: [], 1: [], 2: [], 3: []}
        self.cell_to_int = {}
        self.int_to_cell = {}
        
        # Build hierarchy: SKU (0) -> Store (1) -> Company (2) -> Total (3)
        self._build_hierarchy(df)
        self._create_cell_mappings()
        
        print(f"📊 Hierarchy built:")
        print(f"  - Rank 0 (SKUs): {len(self.cells[0])} cells")
        print(f"  - Rank 1 (Stores): {len(self.cells[1])} cells")
        print(f"  - Rank 2 (Companies): {len(self.cells[2])} cells")
        print(f"  - Rank 3 (Total): {len(self.cells[3])} cells")
        print(f"  - Total cells: {self.num_cells}")
        
        self.neighborhoods = self.create_neighborhood_matrices()
        print(f"✅ Complex built with {len(self.neighborhoods)} neighborhood types.")
    
    def _build_hierarchy(self, df):
        """Build the hierarchical cell structure from data."""
        self.cells[0] = sorted(list(df.groupby(['companyID', 'storeID', 'skuID']).groups.keys()))
        self.cells[1] = sorted(list(df.groupby(['companyID', 'storeID']).groups.keys()))
        self.cells[2] = sorted([(c,) for c in df['companyID'].unique()])
        self.cells[3] = [('total',)]
    
    def _create_cell_mappings(self):
        """Create bidirectional mappings between cells and integers."""
        cell_counter = 0
        for rank in sorted(self.cells.keys()):
            for cell in self.cells[rank]:
                self.cell_to_int[cell] = cell_counter
                self.int_to_cell[cell_counter] = cell
                cell_counter += 1
        
        self.num_cells = len(self.cell_to_int)

    def create_neighborhood_matrices(self) -> Dict[str, torch.sparse.FloatTensor]:
        """
        Creates sparse matrices for different neighborhood relationships.
        
        Returns:
            Dictionary of sparse tensors representing different neighborhood types
        """
        edges = defaultdict(list)
        
        # 1. Incidence Up (Bottom-Up, Rank 0 -> 1, 1 -> 2, 2 -> 3)
        self._create_incidence_relations(edges)
        
        # 2. Downward Adjacency (Peer-to-Peer for SKUs in the same store)
        self._create_adjacency_relations(edges)

        # Convert edge lists to sparse tensors
        matrices = {}
        for name, edge_list in edges.items():
            if not edge_list:
                continue
            edge_tensor = torch.tensor(edge_list, dtype=torch.long).t()
            matrices[name] = torch.sparse_coo_tensor(
                edge_tensor, 
                torch.ones(len(edge_list)), 
                (self.num_cells, self.num_cells)
            )
            print(f"  - {name}: {len(edge_list)} edges")
            
        return matrices
    
    def _create_incidence_relations(self, edges):
        """Create incidence (covering) relations between hierarchy levels."""
        # SKU -> Store
        for sku_cell in self.cells[0]:
            store_cell = (sku_cell[0], sku_cell[1])  # (companyID, storeID)
            edges['incidence_up'].append([
                self.cell_to_int[sku_cell], 
                self.cell_to_int[store_cell]
            ])
        
        # Store -> Company
        for store_cell in self.cells[1]:
            company_cell = (store_cell[0],)  # (companyID,)
            edges['incidence_up'].append([
                self.cell_to_int[store_cell], 
                self.cell_to_int[company_cell]
            ])
        
        # Company -> Total
        for company_cell in self.cells[2]:
            edges['incidence_up'].append([
                self.cell_to_int[company_cell], 
                self.cell_to_int[self.cells[3][0]]
            ])
    
    def _create_adjacency_relations(self, edges):
        """Create adjacency relations between peers at the same level."""
        # Peer-to-Peer for SKUs in the same store
        for store_cell in self.cells[1]:
            skus_in_store = [
                c for c in self.cells[0] 
                if c[0] == store_cell[0] and c[1] == store_cell[1]
            ]
            for i in range(len(skus_in_store)):
                for j in range(i + 1, len(skus_in_store)):
                    u = self.cell_to_int[skus_in_store[i]]
                    v = self.cell_to_int[skus_in_store[j]]
                    edges['adj_down'].append([u, v])
                    edges['adj_down'].append([v, u])
    
    def get_covering_relations(self) -> List[Tuple[Any, Any]]:
        """
        Get all covering relations in the complex.
        
        Returns:
            List of (child, parent) tuples representing covering relations
        """
        relations = []
        
        # SKU -> Store relations
        for sku_cell in self.cells[0]:
            store_cell = (sku_cell[0], sku_cell[1])
            if store_cell in self.cells[1]:
                relations.append((sku_cell, store_cell))
        
        # Store -> Company relations
        for store_cell in self.cells[1]:
            company_cell = (store_cell[0],)
            if company_cell in self.cells[2]:
                relations.append((store_cell, company_cell))
        
        # Company -> Total relations
        for company_cell in self.cells[2]:
            relations.append((company_cell, self.cells[3][0]))
        
        return relations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the complex structure."""
        return {
            'num_cells': self.num_cells,
            'cells_by_rank': {rank: len(cells) for rank, cells in self.cells.items()},
            'num_neighborhoods': len(self.neighborhoods),
            'neighborhood_sizes': {
                name: matrix.nnz() if hasattr(matrix, 'nnz') else matrix._nnz()
                for name, matrix in self.neighborhoods.items()
            },
            'covering_relations': len(self.get_covering_relations())
        }
