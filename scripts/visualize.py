#!/usr/bin/env python3
"""
Visualization script for hierarchical forecasting project.

This script creates various visualizations of the combinatorial complex
structure, including Hasse diagrams, network plots, and analysis charts.
"""

import argparse
import os
import sys
from pathlib import Path

# Add hierarchical_forecasting directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from hierarchical_forecasting.models import CombinatorialComplex
from hierarchical_forecasting.data import DataPreprocessor
from hierarchical_forecasting.visualization import HasseDiagramVisualizer, ComplexVisualizer
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create visualizations for hierarchical forecasting')
    
    parser.add_argument('--data_path', type=str, default='bakery_data_merged.csv',
                        help='Path to the merged data file')
    parser.add_argument('--plot_type', type=str, default='all',
                        choices=['all', 'hasse', 'complex', 'network'],
                        help='Type of plots to generate')
    parser.add_argument('--output_dir', type=str, default='outputs/plots',
                        help='Output directory for plots')
    
    return parser.parse_args()


def load_data(data_path):
    """Load and preprocess data."""
    print("📥 Loading data for visualization...")
    
    try:
        df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
        print(f"✅ Successfully loaded data from {data_path}")
        print(f"Data shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Data file {data_path} not found.")
        print("Creating dummy data for demonstration...")
        from data.preprocessing import create_dummy_data
        return create_dummy_data()


def main():
    """Main visualization function."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    df = load_data(args.data_path)
    preprocessor = DataPreprocessor(scaling_method='none')  # No scaling for visualization
    
    # Get feature columns
    feature_columns = [col for col in df.columns 
                      if col not in ['target', 'companyID', 'storeID', 'skuID']]
    preprocessor.feature_columns = feature_columns
    
    # Build combinatorial complex
    print("\n🏗️ Building combinatorial complex...")
    cc = CombinatorialComplex(df)
    
    # Print complex statistics
    stats = cc.get_statistics()
    print(f"\n📊 Complex Statistics:")
    print(f"  - Total cells: {stats['num_cells']}")
    print(f"  - SKUs: {stats['cells_by_rank'][0]}")
    print(f"  - Stores: {stats['cells_by_rank'][1]}")
    print(f"  - Companies: {stats['cells_by_rank'][2]}")
    print(f"  - Covering relations: {stats['covering_relations']}")
    
    # Create visualizations based on plot type
    if args.plot_type in ['all', 'hasse']:
        print("\n🎨 Creating Hasse diagrams...")
        hasse_viz = HasseDiagramVisualizer(cc, args.output_dir)
        hasse_viz.plot_all_diagrams()
    
    if args.plot_type in ['all', 'complex']:
        print("\n🎨 Creating complex structure visualizations...")
        complex_viz = ComplexVisualizer(cc, args.output_dir)
        complex_viz.plot_all_visualizations()
    
    if args.plot_type in ['all', 'network']:
        print("\n🎨 Creating network visualizations...")
        complex_viz = ComplexVisualizer(cc, args.output_dir)
        complex_viz.plot_3d_network()
    
    print(f"\n✅ All visualizations completed!")
    print(f"📁 Plots saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
