"""
Complex structure visualization utilities.

This module provides functionality to visualize the combinatorial complex
structure, neighborhoods, and topological properties.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.combinatorial_complex import CombinatorialComplex
except ImportError:
    from ..models.combinatorial_complex import CombinatorialComplex


class ComplexVisualizer:
    """
    Visualizes combinatorial complex structures and properties.
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
    
    def plot_neighborhood_analysis(self):
        """Plot analysis of neighborhood structures."""
        print("🎨 Creating neighborhood analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Neighborhood sizes
        ax1 = axes[0, 0]
        neighborhood_names = list(self.cc.neighborhoods.keys())
        neighborhood_sizes = [
            matrix.nnz() if hasattr(matrix, 'nnz') else matrix._nnz() 
            for matrix in self.cc.neighborhoods.values()
        ]
        
        bars = ax1.bar(neighborhood_names, neighborhood_sizes, 
                       color=['#FF9999', '#66B2FF'])
        ax1.set_title('Neighborhood Edge Counts', fontweight='bold')
        ax1.set_ylabel('Number of Edges')
        
        # Add value labels on bars
        for bar, size in zip(bars, neighborhood_sizes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{size}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Hierarchy distribution
        ax2 = axes[0, 1]
        ranks = list(self.cc.cells.keys())
        cell_counts = [len(self.cc.cells[rank]) for rank in ranks]
        rank_labels = [f'Rank {rank}' for rank in ranks]
        
        wedges, texts, autotexts = ax2.pie(cell_counts, labels=rank_labels, autopct='%1.1f%%',
                                          colors=[self.rank_colors[rank] for rank in ranks])
        ax2.set_title('Cell Distribution by Rank', fontweight='bold')
        
        # Plot 3: Adjacency matrix heatmap (sample)
        ax3 = axes[1, 0]
        if 'incidence_up' in self.cc.neighborhoods:
            matrix = self.cc.neighborhoods['incidence_up'].coalesce()
            # Take a small sample for visualization
            sample_size = min(50, self.cc.num_cells)
            dense_sample = matrix[:sample_size, :sample_size].to_dense().numpy()
            
            im = ax3.imshow(dense_sample, cmap='Blues', aspect='auto')
            ax3.set_title('Incidence Matrix (Sample)', fontweight='bold')
            ax3.set_xlabel('Target Cells')
            ax3.set_ylabel('Source Cells')
            plt.colorbar(im, ax=ax3, fraction=0.046)
        
        # Plot 4: Complex statistics
        ax4 = axes[1, 1]
        stats = self.cc.get_statistics()
        
        stats_text = f"""Complex Statistics:
        
Total Cells: {stats['num_cells']}

Cells by Rank:
• Rank 0 (SKUs): {stats['cells_by_rank'][0]}
• Rank 1 (Stores): {stats['cells_by_rank'][1]}
• Rank 2 (Companies): {stats['cells_by_rank'][2]}
• Rank 3 (Total): {stats['cells_by_rank'][3]}

Neighborhoods: {stats['num_neighborhoods']}
• incidence_up: {stats['neighborhood_sizes'].get('incidence_up', 0)} edges
• adj_down: {stats['neighborhood_sizes'].get('adj_down', 0)} edges

Covering Relations: {stats['covering_relations']}
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.3))
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/neighborhood_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✅ Neighborhood analysis saved to {self.output_dir}/neighborhood_analysis.png")
        plt.show()
    
    def plot_3d_network(self):
        """Create interactive 3D network visualization."""
        print("🎨 Creating 3D network visualization...")
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for rank, cells in self.cc.cells.items():
            for cell in cells:
                G.add_node(cell, rank=rank)
        
        # Add edges from covering relations
        covering_relations = self.cc.get_covering_relations()
        for child, parent in covering_relations:
            G.add_edge(child, parent)
        
        # Use spring layout for positioning
        pos_2d = nx.spring_layout(G, k=3, iterations=50)
        
        # Convert to 3D positions
        pos_3d = {}
        for node, (x, y) in pos_2d.items():
            rank = G.nodes[node]['rank']
            z = rank * 2  # Separate by rank in z-direction
            pos_3d[node] = (x, y, z)
        
        # Extract coordinates
        node_x = [pos_3d[node][0] for node in G.nodes()]
        node_y = [pos_3d[node][1] for node in G.nodes()]
        node_z = [pos_3d[node][2] for node in G.nodes()]
        
        # Node colors based on rank
        node_colors = [self.rank_colors[G.nodes[node]['rank']] for node in G.nodes()]
        
        # Node labels
        node_labels = []
        for node in G.nodes():
            rank = G.nodes[node]['rank']
            if rank == 0:
                label = f"SKU: {str(node[2])[:10]}"
            elif rank == 1:
                label = f"Store: {str(node[1])[:10]}"
            elif rank == 2:
                label = f"Company: {str(node[0])[:10]}"
            else:
                label = "Total"
            node_labels.append(label)
        
        # Create edges
        edge_x, edge_y, edge_z = [], [], []
        for edge in G.edges():
            x0, y0, z0 = pos_3d[edge[0]]
            x1, y1, z1 = pos_3d[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
        # Create traces
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(125,125,125,0.5)', width=2),
            name='Covering Relations'
        )
        
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=8,
                color=node_colors,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=node_labels,
            textposition="middle center",
            name='Hierarchy Nodes'
        )
        
        # Create layout
        layout = go.Layout(
            title='3D Combinatorial Complex Structure',
            showlegend=True,
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Hierarchy Level'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
        
        # Save as HTML
        pyo.plot(fig, filename=f'{self.output_dir}/3d_complex_network.html', auto_open=False)
        print(f"✅ 3D network saved to {self.output_dir}/3d_complex_network.html")
    
    def plot_structure_overview(self):
        """Create structure overview visualization."""
        print("🎨 Creating structure overview...")
        
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        
        y_positions = {0: 0, 1: 3, 2: 6, 3: 9}
        
        # For each rank, show representative sample
        for rank in sorted(self.cc.cells.keys()):
            cells = self.cc.cells[rank]
            y = y_positions[rank]
            
            if rank == 0:  # SKUs - show sample
                sample_size = min(10, len(cells))
                sample_cells = cells[:sample_size]
                x_positions = np.linspace(-5, 5, sample_size)
                
                for i, cell in enumerate(sample_cells):
                    x = x_positions[i]
                    circle = plt.Circle((x, y), 0.2, color=self.rank_colors[rank], 
                                      alpha=0.8, zorder=3)
                    ax.add_patch(circle)
                    ax.text(x, y-0.4, str(cell[2])[:6], ha='center', va='center', 
                           fontsize=8, rotation=45)
                
                ax.text(6, y, f"Total: {len(cells)} SKUs", ha='left', va='center', 
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor=self.rank_colors[rank], alpha=0.3))
                       
            elif rank == 1:  # Stores - show sample  
                sample_size = min(8, len(cells))
                sample_cells = cells[:sample_size]
                x_positions = np.linspace(-4, 4, sample_size)
                
                for i, cell in enumerate(sample_cells):
                    x = x_positions[i]
                    circle = plt.Circle((x, y), 0.3, color=self.rank_colors[rank], 
                                      alpha=0.8, zorder=3)
                    ax.add_patch(circle)
                    ax.text(x, y, str(cell[1])[:6], ha='center', va='center', 
                           fontsize=9, fontweight='bold')
                
                ax.text(5, y, f"Total: {len(cells)} Stores", ha='left', va='center', 
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor=self.rank_colors[rank], alpha=0.3))
                       
            elif rank == 2:  # Companies
                for i, cell in enumerate(cells):
                    x = i * 2 - (len(cells)-1)
                    circle = plt.Circle((x, y), 0.4, color=self.rank_colors[rank], 
                                      alpha=0.8, zorder=3)
                    ax.add_patch(circle)
                    ax.text(x, y, str(cell[0])[:8], ha='center', va='center', 
                           fontsize=10, fontweight='bold')
                           
            else:  # Total
                circle = plt.Circle((0, y), 0.5, color=self.rank_colors[rank], 
                                  alpha=0.8, zorder=3)
                ax.add_patch(circle)
                ax.text(0, y, "TOTAL", ha='center', va='center', 
                       fontsize=12, fontweight='bold')
        
        # Add rank labels
        for rank in sorted(self.cc.cells.keys()):
            y = y_positions[rank]
            rank_names = {0: 'SKUs', 1: 'Stores', 2: 'Companies', 3: 'Total'}
            ax.text(-8, y, f"Rank {rank}\n{rank_names[rank]}", ha='center', va='center', 
                   fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor=self.rank_colors[rank], alpha=0.5))
        
        covering_relations = self.cc.get_covering_relations()
        ax.set_title(f'Combinatorial Complex: Structure Overview\n' + 
                    f'({self.cc.num_cells} total cells, {len(covering_relations)} covering relations)',
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.set_ylim(-1, 10)
        ax.set_xlim(-10, 8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/structure_overview.png', dpi=300, bbox_inches='tight')
        print(f"✅ Structure overview saved to {self.output_dir}/structure_overview.png")
        plt.show()
    
    def plot_all_visualizations(self):
        """Create all complex visualizations."""
        print("🎨 Creating all complex visualizations...")
        self.plot_neighborhood_analysis()
        self.plot_structure_overview()
        self.plot_3d_network()
        print("✅ All complex visualizations completed!")
