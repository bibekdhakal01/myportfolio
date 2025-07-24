"""
Grid visualization tools.

This module provides visualization capabilities for smart grid topology,
power flows, and optimization results.
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple


class GridVisualizer:
    """Visualizes smart grid topology and solutions."""
    
    def __init__(self):
        """Initialize grid visualizer."""
        self.node_colors = {
            'generator': '#ff6b6b',
            'load': '#4ecdc4', 
            'normal': '#95a5a6'
        }
        
        self.line_colors = {
            'active': '#2c3e50',
            'inactive': '#bdc3c7'
        }
    
    def plot_grid_topology(self, grid, solution=None, save_path: Optional[str] = None) -> None:
        """Plot grid topology with optional solution overlay."""
        rows, cols = grid.grid_size
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes with positions
        pos = {}
        for row in range(rows):
            for col in range(cols):
                node_id = row * cols + col
                G.add_node(node_id)
                pos[node_id] = (col, rows - row - 1)  # Flip y-axis for proper display
        
        # Add edges based on grid topology
        for line in grid.lines:
            if solution is None or solution.line_status[line.id] == 1:
                G.add_edge(line.from_node, line.to_node)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            # Check if node has generator or load
            has_generator = any(gen.node == node for gen in grid.generators)
            has_load = any(load.node == node for load in grid.loads)
            
            if has_generator:
                node_colors.append(self.node_colors['generator'])
                node_sizes.append(300)
            elif has_load:
                node_colors.append(self.node_colors['load'])
                node_sizes.append(200)
            else:
                node_colors.append(self.node_colors['normal'])
                node_sizes.append(100)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7)
        
        # Draw edges
        if solution is not None:
            # Draw active and inactive lines differently
            active_edges = []
            inactive_edges = []
            
            for line in grid.lines:
                edge = (line.from_node, line.to_node)
                if solution.line_status[line.id] == 1:
                    active_edges.append(edge)
                else:
                    inactive_edges.append(edge)
            
            nx.draw_networkx_edges(G, pos, edgelist=active_edges, 
                                 edge_color=self.line_colors['active'], width=2)
            nx.draw_networkx_edges(G, pos, edgelist=inactive_edges, 
                                 edge_color=self.line_colors['inactive'], 
                                 width=1, style='dashed', alpha=0.5)
        else:
            nx.draw_networkx_edges(G, pos, edge_color=self.line_colors['active'], width=2)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], c=self.node_colors['generator'], s=100, label='Generator'),
            plt.scatter([], [], c=self.node_colors['load'], s=100, label='Load'),
            plt.scatter([], [], c=self.node_colors['normal'], s=100, label='Bus')
        ]
        
        if solution is not None:
            legend_elements.extend([
                plt.plot([], [], color=self.line_colors['active'], linewidth=2, label='Active Line')[0],
                plt.plot([], [], color=self.line_colors['inactive'], linewidth=1, 
                        linestyle='--', alpha=0.5, label='Inactive Line')[0]
            ])
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title('Smart Grid Topology')
        plt.axis('equal')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_power_flow(self, grid, solution, save_path: Optional[str] = None) -> None:
        """Plot power flow visualization."""
        # Placeholder for power flow visualization
        print("Power flow visualization: arrows showing power direction and magnitude")
        
        if save_path:
            print(f"Would save power flow plot to: {save_path}")
    
    def plot_voltage_profile(self, solution, save_path: Optional[str] = None) -> None:
        """Plot voltage profile across nodes."""
        nodes = list(range(len(solution.voltage_levels)))
        voltages = solution.voltage_levels
        
        plt.figure(figsize=(10, 6))
        plt.bar(nodes, voltages, color='steelblue', alpha=0.7)
        plt.axhline(y=1.0, color='red', linestyle='--', label='Nominal (1.0 pu)')
        plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='Lower Limit')
        plt.axhline(y=1.05, color='orange', linestyle='--', alpha=0.7, label='Upper Limit')
        
        plt.xlabel('Node')
        plt.ylabel('Voltage (per unit)')
        plt.title('Voltage Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()