"""Combinatorial Complex utilities.

This module provides a lightweight combinatorial complex representation that can
operate either on raw hierarchy tables (company/store/SKU identifiers) or on
pre-assembled hierarchy dictionaries. The resulting complex exposes adjacency
matrices for multiple neighborhood types so that downstream ETNN components can
perform message passing in an equivariant way.
"""

from __future__ import annotations

import torch
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple, Union



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
        self._build_parent_child_maps()
        self._build_cell_to_node_map()
        self.edge_index: Dict[str, torch.Tensor] = {}
        self.node_degrees: Dict[str, torch.Tensor] = {}
        
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

    def _build_parent_child_maps(self) -> None:
        """Create lookup tables for parent/child relationships between cells."""
        self.rank_lookup = {}
        for rank, cells in self.cells.items():
            for cell in cells:
                self.rank_lookup[cell] = rank

        self.parent_map: Dict[Tuple[Any, ...], Tuple[Any, ...]] = {}
        self.children_map: Dict[Tuple[Any, ...], List[Tuple[Any, ...]]] = defaultdict(list)

        # SKU -> Store relations
        for sku_cell in self.cells[0]:
            store_cell = (sku_cell[0], sku_cell[1]) if len(sku_cell) >= 2 else None
            if store_cell and store_cell in self.cell_to_int:
                self.parent_map[sku_cell] = store_cell
                self.children_map[store_cell].append(sku_cell)

        # Store -> Company relations
        for store_cell in self.cells[1]:
            company_cell = (store_cell[0],)
            if company_cell in self.cell_to_int:
                self.parent_map[store_cell] = company_cell
                self.children_map[company_cell].append(store_cell)

        # Company -> Total relations (single root)
        if self.cells[3]:
            root = self.cells[3][0]
            for company_cell in self.cells[2]:
                self.parent_map[company_cell] = root
                self.children_map[root].append(company_cell)

    def _build_cell_to_node_map(self) -> None:
        """Map every cell to the list of bottom-level cell indices it contains."""

        @lru_cache(maxsize=None)
        def collect_bottom(cell: Tuple[Any, ...]) -> List[int]:
            rank = self.rank_lookup.get(cell, None)
            if rank == 0:
                return [self.cell_to_int[cell]]
            children = self.children_map.get(cell, [])
            collected: List[int] = []
            for child in children:
                collected.extend(collect_bottom(child))
            return sorted(set(collected))

        self.cell_to_nodes: List[List[int]] = []
        for idx in range(self.num_cells):
            cell = self.int_to_cell[idx]
            self.cell_to_nodes.append(collect_bottom(cell))

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

        # Additional aliases expected by ETNN layers
        if edges.get('incidence_up'):
            edges['up'] = list(edges['incidence_up'])
            edges['down'] = [[dst, src] for src, dst in edges['incidence_up']]
        else:
            edges.setdefault('up', [])
            edges.setdefault('down', [])

        if edges.get('adj_down'):
            edges['boundary'] = list(edges['adj_down'])
        else:
            edges.setdefault('boundary', [])

        # Convert edge lists to sparse tensors
        matrices = {}
        for name, edge_list in edges.items():
            if not edge_list:
                continue
            # Remove duplicate edges to keep matrices well-conditioned.
            unique_edges = sorted({(src, dst) for src, dst in edge_list})
            if not unique_edges:
                continue
            edge_tensor = torch.tensor(unique_edges, dtype=torch.long).t()
            matrix = torch.sparse_coo_tensor(
                edge_tensor, 
                torch.ones(len(unique_edges)), 
                (self.num_cells, self.num_cells)
            ).coalesce()
            matrices[name] = matrix
            self.edge_index[name] = matrix.indices()
            self.node_degrees[name] = torch.bincount(self.edge_index[name][1], minlength=self.num_cells)
            print(f"  - {name}: {len(edge_list)} edges")
            
        return matrices
    
    def _create_incidence_relations(self, edges):
        """Create incidence (covering) relations between hierarchy levels."""
        for child_cell, parent_cell in self.parent_map.items():
            child_idx = self.cell_to_int[child_cell]
            parent_idx = self.cell_to_int[parent_cell]
            edges['incidence_up'].append([child_idx, parent_idx])
    
    def _create_adjacency_relations(self, edges):
        """Create adjacency relations between peers at the same level."""
        for parent_cell, child_cells in self.children_map.items():
            child_indices = [self.cell_to_int[ch] for ch in child_cells]
            for i in range(len(child_indices)):
                for j in range(i + 1, len(child_indices)):
                    u = child_indices[i]
                    v = child_indices[j]
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
