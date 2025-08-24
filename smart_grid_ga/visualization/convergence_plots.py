"""
Convergence plotting and analysis tools.

This module provides visualization tools for analyzing genetic algorithm
convergence behavior and performance metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
import seaborn as sns


class ConvergencePlotter:
    """Plots convergence curves and analysis charts for the genetic algorithm."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """Initialize plotter with style."""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_fitness_convergence(self, convergence_history: List[Dict[str, Any]], 
                               save_path: Optional[str] = None) -> None:
        """Plot fitness convergence over generations."""
        generations = [entry['generation'] for entry in convergence_history]
        best_fitness = [entry['best_fitness'] for entry in convergence_history]
        avg_fitness = [entry['average_fitness'] for entry in convergence_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, label='Best Fitness', color=self.colors[0], linewidth=2)
        plt.plot(generations, avg_fitness, label='Average Fitness', color=self.colors[1], linewidth=2)
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Genetic Algorithm Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_diversity_evolution(self, convergence_history: List[Dict[str, Any]], 
                               save_path: Optional[str] = None) -> None:
        """Plot population diversity evolution."""
        generations = [entry['generation'] for entry in convergence_history]
        diversity = [entry['diversity'] for entry in convergence_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, diversity, color=self.colors[2], linewidth=2)
        
        plt.xlabel('Generation')
        plt.ylabel('Population Diversity')
        plt.title('Population Diversity Evolution')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class GridVisualizer:
    """Visualizes smart grid topology and solutions."""
    
    def __init__(self):
        """Initialize grid visualizer."""
        pass
    
    def plot_grid_topology(self, grid, solution=None, save_path: Optional[str] = None) -> None:
        """Plot grid topology with optional solution overlay."""
        # Placeholder for grid visualization
        print("Grid visualization will be implemented with network topology display")
        
        if save_path:
            print(f"Would save to: {save_path}")
        else:
            print("Would display interactive plot")