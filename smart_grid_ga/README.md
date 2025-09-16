# Smart Grid Genetic Algorithm Optimization System

A parallelized Genetic Algorithm-based optimization system for smart grid energy distribution.

## Overview

This system implements a sophisticated genetic algorithm to optimize energy distribution in smart grids, incorporating parallel processing capabilities for enhanced performance.

## Features

- **Core Genetic Algorithm**: Population management, selection, crossover, mutation
- **Smart Grid Modeling**: Mathematical models for energy distribution optimization
- **Parallel Processing**: OpenMP and CUDA support for high-performance computing
- **Visualization Tools**: Real-time convergence plots and analysis dashboards
- **Performance Analysis**: Comprehensive benchmarking and metrics

## Components

### Core Algorithm (`core/`)
- `genetic_algorithm.py`: Main GA implementation
- `population.py`: Population management and operations
- `selection.py`: Selection strategies (tournament, roulette wheel, etc.)
- `crossover.py`: Crossover operations for chromosome recombination
- `mutation.py`: Mutation operators for genetic diversity

### Smart Grid Models (`models/`)
- `grid_model.py`: Smart grid mathematical model
- `chromosome.py`: Chromosome encoding for grid configurations
- `fitness.py`: Fitness evaluation functions
- `constraints.py`: Grid constraints and validation

### Parallel Processing (`parallel/`)
- `openmp_ga.py`: OpenMP parallelization
- `cuda_ga.py`: CUDA GPU acceleration
- `parallel_utils.py`: Parallel processing utilities

### Visualization (`visualization/`)
- `convergence_plots.py`: Algorithm convergence visualization
- `grid_visualization.py`: Energy distribution visualization
- `performance_dashboard.py`: Real-time performance monitoring

### Analysis (`analysis/`)
- `benchmark.py`: Performance benchmarking tools
- `statistics.py`: Statistical analysis utilities
- `results_analysis.py`: Comprehensive results analysis

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from smart_grid_ga import SmartGridGA

# Initialize the genetic algorithm
ga = SmartGridGA(
    population_size=100,
    generations=500,
    grid_size=(10, 10),
    parallel_mode='openmp'
)

# Run optimization
best_solution = ga.optimize()

# Visualize results
ga.plot_convergence()
ga.visualize_grid_solution(best_solution)
```

## License

MIT License