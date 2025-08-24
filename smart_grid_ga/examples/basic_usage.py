"""
Example usage of the Smart Grid GA optimization system.

This script demonstrates how to use the Smart Grid GA system for
various optimization scenarios and configurations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from smart_grid_ga.core.genetic_algorithm import create_smart_grid_ga, GAParameters
from smart_grid_ga.visualization.convergence_plots import ConvergencePlotter
from smart_grid_ga.visualization.grid_visualization import GridVisualizer


def example_small_grid_optimization():
    """Example: Small grid optimization with detailed analysis."""
    print("Example 1: Small Grid Optimization")
    print("=" * 40)
    
    # Configure system
    parameters = GAParameters(
        population_size=50,
        generations=100,
        elite_size=5,
        crossover_rate=0.8,
        mutation_rate=0.1,
        selection_strategy='tournament',
        crossover_strategy='smart_grid',
        mutation_strategy='adaptive'
    )
    
    # Create system
    ga = create_smart_grid_ga(
        grid_size=(4, 4),
        num_generators=3,
        num_loads=5,
        **parameters.__dict__
    )
    
    print(f"Created {ga.grid.grid_size[0]}x{ga.grid.grid_size[1]} grid")
    print(f"Components: {len(ga.grid.generators)} generators, {len(ga.grid.loads)} loads")
    
    # Run optimization
    best_solution = ga.optimize(verbose=True)
    
    # Analyze results
    analysis = ga.get_best_solution_analysis()
    print(f"\nResults Summary:")
    print(f"Best fitness: {ga.best_fitness:.6f}")
    print(f"Feasible: {analysis['feasibility']['is_feasible']}")
    print(f"Total cost: ${analysis['economic_metrics']['total_generation_cost']:.2f}")
    print(f"Generation: {analysis['economic_metrics']['total_generation']:.1f} MW")
    print(f"Load: {analysis['economic_metrics']['total_load']:.1f} MW")
    
    # Create visualizations
    plotter = ConvergencePlotter()
    visualizer = GridVisualizer()
    
    # Plot convergence
    print("Creating convergence plots...")
    plotter.plot_fitness_convergence(ga.convergence_history, 'convergence.png')
    plotter.plot_diversity_evolution(ga.convergence_history, 'diversity.png')
    
    # Plot grid topology
    print("Creating grid visualization...")
    visualizer.plot_grid_topology(ga.grid, best_solution, 'grid_topology.png')
    visualizer.plot_voltage_profile(best_solution, 'voltage_profile.png')
    
    print("Visualizations saved as PNG files")
    return ga


def example_multi_objective_optimization():
    """Example: Multi-objective optimization with Pareto analysis."""
    print("\nExample 2: Multi-Objective Optimization")
    print("=" * 40)
    
    # Configure for multi-objective optimization
    objective_weights = {
        'generation_cost': 0.3,
        'power_losses': 0.2,
        'voltage_deviation': 0.2,
        'reliability': 0.2,
        'emissions': 0.1
    }
    
    parameters = GAParameters(
        population_size=80,
        generations=150,
        multi_objective=True,
        objective_weights=objective_weights,
        selection_strategy='multi_objective'
    )
    
    # Create larger system
    ga = create_smart_grid_ga(
        grid_size=(5, 5),
        num_generators=4,
        num_loads=8,
        **parameters.__dict__
    )
    
    print(f"Multi-objective optimization on {ga.grid.grid_size[0]}x{ga.grid.grid_size[1]} grid")
    
    # Run optimization
    best_solution = ga.optimize(verbose=True)
    
    # Analyze Pareto front
    pareto_front = ga.pareto_front
    print(f"\nPareto Front Analysis:")
    print(f"Pareto front size: {len(pareto_front)}")
    
    if pareto_front:
        print("Pareto optimal solutions:")
        for i, solution in enumerate(pareto_front[:3]):  # Show first 3
            if hasattr(solution, 'objectives'):
                print(f"  Solution {i+1}:")
                print(f"    Cost: ${solution.objectives.get('generation_cost', 0):.2f}")
                print(f"    Losses: {solution.objectives.get('power_losses', 0):.2f} MW")
                print(f"    Reliability: {solution.objectives.get('reliability', 0):.4f}")
    
    return ga


def example_large_scale_optimization():
    """Example: Large-scale grid optimization."""
    print("\nExample 3: Large-Scale Grid Optimization")
    print("=" * 40)
    
    # Configure for large system
    parameters = GAParameters(
        population_size=100,
        generations=200,
        elite_size=10,
        crossover_rate=0.9,
        mutation_rate=0.05,  # Lower mutation for large systems
        selection_strategy='rank',
        crossover_strategy='arithmetic',
        mutation_strategy='polynomial'
    )
    
    # Create large system
    ga = create_smart_grid_ga(
        grid_size=(8, 8),
        num_generators=8,
        num_loads=15,
        **parameters.__dict__
    )
    
    print(f"Large-scale optimization: {ga.grid.num_nodes} nodes")
    print(f"Components: {len(ga.grid.generators)} generators, {len(ga.grid.loads)} loads")
    print(f"Transmission lines: {len(ga.grid.lines)}")
    
    # Add progress callback
    def progress_callback(ga_instance):
        if ga_instance.current_generation % 25 == 0:
            stats = ga_instance.population.get_population_summary()
            print(f"  Progress: Gen {ga_instance.current_generation}, "
                  f"Best={ga_instance.best_fitness:.4f}, "
                  f"Feasible={stats['feasibility_rate']:.1%}")
    
    ga.add_generation_callback(progress_callback)
    
    # Run optimization
    best_solution = ga.optimize(verbose=False)
    
    # Performance analysis
    summary = ga.get_optimization_summary()
    print(f"\nPerformance Summary:")
    print(f"Runtime: {summary['performance']['total_time']:.2f} seconds")
    print(f"Evaluations: {summary['performance']['evaluations']:,}")
    print(f"Generations: {summary['results']['generations_run']}")
    print(f"Final fitness: {summary['results']['best_fitness']:.6f}")
    
    return ga


def example_custom_configuration():
    """Example: Custom GA configuration and analysis."""
    print("\nExample 4: Custom Configuration")
    print("=" * 40)
    
    # Create grid manually for custom configuration
    from smart_grid_ga.models.grid_model import SmartGrid
    from smart_grid_ga.core.genetic_algorithm import SmartGridGA
    
    # Custom grid setup
    grid = SmartGrid((6, 6))
    
    # Add generators with specific characteristics
    grid.add_generator(0, 20, 100, (0.01, 15, 80))   # Cheap base load
    grid.add_generator(5, 10, 80, (0.015, 25, 120))  # Mid-range
    grid.add_generator(30, 5, 60, (0.02, 35, 150))   # Expensive peaker
    grid.add_generator(35, 15, 90, (0.012, 20, 100)) # Renewable (low cost)
    
    # Add loads with priorities
    grid.add_load(10, 50, priority=1)  # Critical load
    grid.add_load(15, 30, priority=2)  # Important load
    grid.add_load(20, 40, priority=3)  # Normal load
    grid.add_load(25, 25, priority=4)  # Deferrable load
    
    print("Custom grid configuration:")
    print(f"  Total capacity: {sum(g.max_power for g in grid.generators):.1f} MW")
    print(f"  Total load: {grid.get_total_load():.1f} MW")
    print(f"  Load priorities: {[l.priority for l in grid.loads]}")
    
    # Custom GA parameters focusing on economic optimization
    parameters = GAParameters(
        population_size=60,
        generations=80,
        objective_weights={
            'generation_cost': 0.5,
            'power_losses': 0.1,
            'voltage_deviation': 0.1,
            'reliability': 0.2,
            'emissions': 0.1
        }
    )
    
    ga = SmartGridGA(grid, parameters)
    
    # Run optimization
    best_solution = ga.optimize(verbose=True)
    
    # Economic analysis
    analysis = ga.get_best_solution_analysis()
    print(f"\nEconomic Analysis:")
    print(f"Total cost: ${analysis['economic_metrics']['total_generation_cost']:.2f}")
    
    # Generator dispatch analysis
    print(f"Generator dispatch:")
    for i, gen in enumerate(grid.generators):
        dispatch = best_solution.generator_dispatch[i]
        utilization = dispatch / gen.max_power * 100
        cost = gen.cost(dispatch)
        print(f"  Gen {i}: {dispatch:.1f} MW ({utilization:.1f}% capacity) - ${cost:.2f}")
    
    return ga


def run_all_examples():
    """Run all examples."""
    print("Smart Grid GA Optimization Examples")
    print("=" * 50)
    
    try:
        # Run examples
        ga1 = example_small_grid_optimization()
        ga2 = example_multi_objective_optimization()
        ga3 = example_large_scale_optimization()
        ga4 = example_custom_configuration()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("Check the generated PNG files for visualizations.")
        
        return [ga1, ga2, ga3, ga4]
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Set up plotting backend for headless environment
    plt.switch_backend('Agg')
    
    # Run all examples
    results = run_all_examples()
    
    if results:
        print(f"\nExample results summary:")
        for i, ga in enumerate(results, 1):
            print(f"Example {i}: Best fitness = {ga.best_fitness:.6f}")