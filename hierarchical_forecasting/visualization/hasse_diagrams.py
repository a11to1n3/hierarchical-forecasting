"""
Hasse diagram visualization utilities.

This module provides functionality to create various types of Hasse diagrams
for visualizing the combinatorial complex structure.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Any
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.combinatorial_complex import CombinatorialComplex
except ImportError:
    from ..models.combinatorial_complex import CombinatorialComplex


class HasseDiagramVisualizer:
    """
    Creates Hasse diagrams for combinatorial complex visualization.
    """
    
    def __init__(self, cc: CombinatorialComplex, output_dir: str = "outputs/plots"):
        """
        Initialize the visualizer.
        
        Args:
            cc: Combinatorial complex instance
            output_dir: Directory to save plots
        """
        self.cc = cc
        self.output_dir = output_dir
        self.rank_colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1', 3: '#96CEB4'}
        self.rank_names = {0: 'SKUs', 1: 'Stores', 2: 'Companies', 3: 'Total'}
    
    def plot_sample_hasse_diagram(self, max_skus_per_store: int = 4, max_stores: int = 3):
        """Create a sample Hasse diagram showing part of the structure."""
        print("🎨 Creating sample Hasse diagram...")
        
        # Take a sample for cleaner visualization
        sample_companies = list(self.cc.cells[2])[:1]
        sample_stores = []
        sample_skus = []
        
        for company in sample_companies:
            company_stores = [s for s in self.cc.cells[1] if s[0] == company[0]][:max_stores]
            sample_stores.extend(company_stores)
            
            for store in company_stores:
                store_skus = [s for s in self.cc.cells[0] 
                             if s[0] == store[0] and s[1] == store[1]][:max_skus_per_store]
                sample_skus.extend(store_skus)
        
        # Build sample cells
        sample_cells = {
            0: sample_skus,
            1: sample_stores,
            2: sample_companies,
            3: self.cc.cells[3]
        }
        
        # Build sample covering relations
        sample_relations = []
        for sku in sample_skus:
            store = (sku[0], sku[1])
            if store in sample_stores:
                sample_relations.append((sku, store))
        
        for store in sample_stores:
            company = (store[0],)
            sample_relations.append((store, company))
        
        for company in sample_companies:
            sample_relations.append((company, self.cc.cells[3][0]))
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Calculate positions
        positions = {}
        y_positions = {0: 0, 1: 4, 2: 8, 3: 12}
        
        for rank in sorted(sample_cells.keys()):
            cells = sample_cells[rank]
            n_cells = len(cells)
            
            if n_cells == 1:
                x_positions = [0]
            else:
                x_positions = np.linspace(-n_cells*1.5, n_cells*1.5, n_cells)
            
            for i, cell in enumerate(cells):
                positions[cell] = (x_positions[i], y_positions[rank])
        
        # Draw edges
        for child, parent in sample_relations:
            if child in positions and parent in positions:
                x1, y1 = positions[child]
                x2, y2 = positions[parent]
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.7, linewidth=2, zorder=1)
        
        # Draw nodes
        for rank, cells in sample_cells.items():
            for cell in cells:
                if cell not in positions:
                    continue
                    
                x, y = positions[cell]
                
                # Create labels
                if rank == 0:  # SKU
                    label = f"SKU:\n{str(cell[2])[:10]}"
                    size = 0.8
                elif rank == 1:  # Store
                    label = f"Store:\n{str(cell[1])[:10]}"
                    size = 1.0
                elif rank == 2:  # Company
                    label = f"Company:\n{str(cell[0])[:10]}"
                    size = 1.2
                else:  # Total
                    label = "Total\nSales"
                    size = 1.4
                
                # Draw node
                bbox = FancyBboxPatch((x-size/2, y-size/2), size, size,
                                     boxstyle="round,pad=0.1",
                                     facecolor=self.rank_colors[rank],
                                     edgecolor='black',
                                     linewidth=2,
                                     alpha=0.8,
                                     zorder=3)
                ax.add_patch(bbox)
                
                # Add label
                ax.text(x, y, label, ha='center', va='center', fontsize=9,
                       fontweight='bold', zorder=4)
        
        # Add statistics
        stats_text = f"""Complex Statistics:
• Total SKUs: {len(self.cc.cells[0])}
• Total Stores: {len(self.cc.cells[1])}
• Total Companies: {len(self.cc.cells[2])}
• Total Cells: {self.cc.num_cells}

Sample showing:
• {len(sample_skus)} SKUs
• {len(sample_stores)} Stores
• 1 Company
• 1 Total node"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
        
        ax.set_title('Combinatorial Complex: Sample Hasse Diagram\n(Hierarchical Sales Structure)',
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.set_ylim(-2, 14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sample_hasse_diagram.png', dpi=300, bbox_inches='tight')
        print(f"✅ Sample Hasse diagram saved to {self.output_dir}/sample_hasse_diagram.png")
        plt.show()
    
    def plot_mathematical_representation(self):
        """Create mathematical representation of the complex."""
        print("🎨 Creating mathematical representation...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left: Abstract structure
        ax1.text(0.5, 0.95, "Abstract Poset Structure", ha='center', va='top',
                transform=ax1.transAxes, fontsize=16, fontweight='bold')
        
        # Draw abstract representation
        levels_y = [0.1, 0.35, 0.6, 0.85]
        level_labels = [f"{len(self.cc.cells[0])} SKUs", f"{len(self.cc.cells[1])} Stores", 
                       f"{len(self.cc.cells[2])} Companies", "1 Total"]
        
        # Draw connections
        for i in range(len(levels_y)-1):
            y1, y2 = levels_y[i], levels_y[i+1]
            ax1.plot([0.5, 0.5], [y1+0.05, y2-0.05], 'k-', linewidth=3, alpha=0.7)
            ax1.annotate('', xy=(0.5, y2-0.05), xytext=(0.5, y1+0.05),
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Draw level nodes
        for i, (y, label, color) in enumerate(zip(levels_y, level_labels, self.rank_colors.values())):
            circle = plt.Circle((0.5, y), 0.08, color=color, alpha=0.8, zorder=3)
            ax1.add_patch(circle)
            ax1.text(0.5, y, f"Rank {i}", ha='center', va='center', 
                    fontsize=10, fontweight='bold', zorder=4)
            ax1.text(0.7, y, label, ha='left', va='center', fontsize=12, fontweight='bold')
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Right: Formal definition
        ax2.text(0.5, 0.95, "Mathematical Definition", ha='center', va='top',
                transform=ax2.transAxes, fontsize=16, fontweight='bold')
        
        definition_text = f"""
Poset (P, ≤) from Sales Data:

P = SKUs ∪ Stores ∪ Companies ∪ {{Total}}

Cardinalities:
• |SKUs| = {len(self.cc.cells[0])}
• |Stores| = {len(self.cc.cells[1])}
• |Companies| = {len(self.cc.cells[2])}
• |Total| = 1
• |P| = {self.cc.num_cells}

Covering Relations: {len(self.cc.get_covering_relations())}
• Each SKU ⋖ its Store
• Each Store ⋖ its Company  
• Each Company ⋖ Total

Neighborhood Matrices:
{chr(10).join(f"• {name}: {matrix.nnz() if hasattr(matrix, 'nnz') else matrix._nnz()} edges" 
              for name, matrix in self.cc.neighborhoods.items())}

Properties:
✓ Connected poset
✓ Graded with 4 levels
✓ All chains have length ≤ 3
        """
        
        ax2.text(0.05, 0.85, definition_text, ha='left', va='top',
                transform=ax2.transAxes, fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.1))
        
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/mathematical_hasse_diagram.png', dpi=300, bbox_inches='tight')
        print(f"✅ Mathematical representation saved to {self.output_dir}/mathematical_hasse_diagram.png")
        plt.show()
    
    def plot_all_diagrams(self):
        """Create all Hasse diagram visualizations."""
        print("🎨 Creating all Hasse diagrams...")
        self.plot_sample_hasse_diagram()
        self.plot_mathematical_representation()
        print("✅ All Hasse diagrams completed!")
