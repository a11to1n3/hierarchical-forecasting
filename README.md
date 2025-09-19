# Hierarchical Forecasting with Combinatorial Complex Message Passing Neural Networks

A deep learning framework for hierarchical sales forecasting using combinatorial complex structures and message-passing neural networks.

## Project Structure

```
HierarchicalForecasting/
├── hierarchical_forecasting/          # Main package
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── combinatorial_complex.py    # Core combinatorial complex implementation
│   │   ├── ccmpn.py                   # Message-passing neural network layers
│   │   └── hierarchical_model.py     # Main forecasting model
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py          # Data preparation and preprocessing
│   │   └── loader.py                 # Data loading utilities
│   └── visualization/
│       ├── __init__.py
│       ├── hasse_diagrams.py        # Hasse diagram visualizations
│       ├── complex_plots.py         # Complex structure visualizations
│       └── training_plots.py        # Training and evaluation plots
├── scripts/
│   ├── train.py                     # Main training script
│   ├── evaluate.py                  # Model evaluation script
│   ├── visualize.py                 # Visualization script
│   └── prepare_data.py              # Data preparation script
├── tests/
│   ├── __init__.py
│   ├── test_models.py               # Model tests
│   ├── test_data.py                 # Data processing tests
│   └── test_visualization.py        # Visualization tests
├── docs/
│   ├── CONTRIBUTING.md              # Contribution guidelines
│   └── CHANGELOG.md                 # Project changelog
├── outputs/
│   ├── plots/                       # Generated visualizations
│   └── models/                      # Saved model checkpoints
├── data/                           # Raw and processed data
├── requirements.txt                # Dependencies
├── setup.py                        # Package setup
├── pyproject.toml                  # Modern Python project configuration
├── Makefile                        # Development shortcuts
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## Features

- **Combinatorial Complex**: Mathematical representation of hierarchical sales structure
- **Message-Passing Neural Networks**: Specialized CCMPN layers for hierarchy-aware learning
- **Hierarchical Forecasting**: Multi-level forecasting with consistency constraints
- **Rich Visualizations**: Hasse diagrams, complex structure plots, and mathematical representations
- **Comprehensive Baselines**: 15+ baseline models for comparison including traditional ML, deep learning, and hierarchical reconciliation methods

## Installation

### From Source
```bash
git clone <repository-url>
cd HierarchicalForecasting
pip install -r requirements.txt
pip install -e .
```

### Development Setup
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or use the Makefile
make setup
```

## Usage

### Training
```bash
python scripts/train.py --epochs 50 --hidden_dim 64 --lr 0.001
```

### Evaluation
```bash
python scripts/evaluate.py --model_path outputs/models/best_model.pt
```

### Visualization
```bash
python scripts/visualize.py --plot_type hasse_diagram
```

### Baseline Comparison
```bash
# Run comprehensive baseline comparison
make compare-baselines

# Or run manually with options
python scripts/compare_baselines.py --data_path data/your_data.csv --visualize
```

## Model Architecture

The model uses a Combinatorial Complex Message Passing Neural Network (CCMPN) that:

1. **Builds Combinatorial Complex**: Creates a mathematical structure representing the sales hierarchy
2. **Message Passing**: Exchanges information between related entities in the hierarchy
3. **Hierarchical Prediction**: Generates forecasts at multiple aggregation levels
4. **Consistency Enforcement**: Ensures forecasts are coherent across hierarchy levels

## Data Structure

The model expects sales data with the following hierarchy:
- **SKU Level**: Individual product-store combinations
- **Store Level**: Aggregated store sales
- **Company Level**: Aggregated company sales  
- **Total Level**: Overall aggregated sales

## Mathematical Foundation

The approach is based on:
- **Poset Theory**: Partially ordered sets for hierarchy representation
- **Combinatorial Topology**: Complex structures for relationship modeling
- **Graph Neural Networks**: Message-passing for structured learning

## Results

The model achieves hierarchical forecasting with:
- Multi-level consistency
- Improved accuracy through structure exploitation
- Interpretable mathematical foundation

## Development

### Running Tests
```bash
# Run all tests
make test

# Run quick tests without coverage
make quick-test

# Run specific test file
python -m pytest tests/test_models.py -v
```

### Code Quality
```bash
# Format code
make format

# Run linting
make lint

# Run all checks
make check
```

### Building and Training
```bash
# Train model with default parameters
make train

# Train with custom parameters
python scripts/train.py --epochs 100 --hidden_dim 128 --lr 0.001

# Evaluate trained model
make evaluate

# Generate visualizations
make visualize
```

## Project Structure Details

- **`hierarchical_forecasting/`**: Main Python package containing all core functionality
- **`scripts/`**: Command-line entry points for training, evaluation, and visualization
- **`tests/`**: Comprehensive test suite with unit and integration tests
- **`docs/`**: Documentation including contribution guidelines and changelog
- **`outputs/`**: Generated artifacts (models, plots) - excluded from version control
- **`data/`**: Data files - mostly excluded from version control except samples

## Contributing

Please read [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [docs/CHANGELOG.md](docs/CHANGELOG.md) for a detailed history of changes.

## Baseline Models

The project includes 15+ baseline models for comprehensive comparison:

### Traditional Machine Learning
- **Linear Regression**: Simple linear baseline with optional regularization (Ridge, Lasso)
- **Random Forest**: Ensemble method for capturing non-linear relationships
- **Multi-Level Models**: Separate models trained for each hierarchy level

### Deep Learning
- **LSTM**: Recurrent neural networks for temporal patterns
- **Multi-Entity LSTM**: Separate LSTM models for different entity types

### Time Series Methods
- **Prophet**: Facebook's Prophet for automatic seasonality detection (optional)
- **Hierarchical Prophet**: Prophet with hierarchical reconciliation

### Hierarchical Reconciliation Methods
- **Bottom-Up**: Forecast at lowest level, aggregate upwards
- **Top-Down**: Forecast at top level, disaggregate downwards  
- **Middle-Out**: Forecast at intermediate level, reconcile both ways
- **MinT (Minimum Trace)**: Optimal reconciliation minimizing forecast variance
- **OLS**: Ordinary Least Squares reconciliation

### Usage
```bash
# Compare all baselines against CCMPN
make compare-baselines

# Results saved to outputs/baselines/
# - baseline_comparison_results.csv: Detailed metrics
# - baseline_comparison_plots.png: Performance visualizations
# - baseline_summary_table.png: Top model summary
```
