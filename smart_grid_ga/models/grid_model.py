"""
Smart grid mathematical model for energy distribution optimization.

This module implements the mathematical formulation of the smart grid optimization problem,
including power flow equations, constraints, and network topology management.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import math


@dataclass
class Generator:
    """Generator specification."""
    id: int
    node: int
    min_power: float  # MW
    max_power: float  # MW
    cost_coeff: Tuple[float, float, float]  # Quadratic cost coefficients (a, b, c)
    ramp_rate: float  # MW/hour
    
    def cost(self, power: float) -> float:
        """Calculate generation cost for given power output."""
        a, b, c = self.cost_coeff
        return a * power**2 + b * power + c


@dataclass
class Load:
    """Load specification."""
    id: int
    node: int
    demand: float  # MW
    priority: int  # Load shedding priority (1=highest, 5=lowest)
    elasticity: float  # Demand response elasticity


@dataclass
class TransmissionLine:
    """Transmission line specification."""
    id: int
    from_node: int
    to_node: int
    reactance: float  # Per-unit reactance
    capacity: float  # MVA rating
    length: float  # km


class SmartGrid:
    """
    Smart grid mathematical model for optimization.
    
    This class encapsulates the electrical network topology, component specifications,
    and mathematical formulations for power flow and optimization constraints.
    """
    
    def __init__(self, grid_size: Tuple[int, int]):
        """
        Initialize smart grid model.
        
        Args:
            grid_size: Tuple of (rows, cols) for grid dimensions
        """
        self.grid_size = grid_size
        self.num_nodes = grid_size[0] * grid_size[1]
        
        # Network components
        self.generators: List[Generator] = []
        self.loads: List[Load] = []
        self.lines: List[TransmissionLine] = []
        
        # Network topology
        self.graph = nx.Graph()
        self.admittance_matrix = None
        
        # System parameters
        self.base_mva = 100.0  # Base MVA for per-unit system
        self.voltage_limits = (0.95, 1.05)  # Per-unit voltage limits
        self.frequency = 50.0  # Hz
        
        self._initialize_topology()
    
    def _initialize_topology(self):
        """Initialize basic grid topology."""
        rows, cols = self.grid_size
        
        # Add nodes to the graph
        for i in range(self.num_nodes):
            self.graph.add_node(i)
        
        # Add potential transmission lines (grid connectivity)
        line_id = 0
        for row in range(rows):
            for col in range(cols):
                node_id = row * cols + col
                
                # Horizontal connections
                if col < cols - 1:
                    neighbor = row * cols + (col + 1)
                    self.graph.add_edge(node_id, neighbor)
                    
                    # Create transmission line
                    line = TransmissionLine(
                        id=line_id,
                        from_node=node_id,
                        to_node=neighbor,
                        reactance=0.1,  # Default reactance
                        capacity=100.0,  # Default capacity
                        length=1.0  # Default length
                    )
                    self.lines.append(line)
                    line_id += 1
                
                # Vertical connections
                if row < rows - 1:
                    neighbor = (row + 1) * cols + col
                    self.graph.add_edge(node_id, neighbor)
                    
                    # Create transmission line
                    line = TransmissionLine(
                        id=line_id,
                        from_node=node_id,
                        to_node=neighbor,
                        reactance=0.1,  # Default reactance
                        capacity=100.0,  # Default capacity
                        length=1.0  # Default length
                    )
                    self.lines.append(line)
                    line_id += 1
    
    def add_generator(self, node: int, min_power: float, max_power: float,
                     cost_coeff: Tuple[float, float, float] = (0.01, 20.0, 100.0),
                     ramp_rate: float = 50.0) -> int:
        """
        Add a generator to the grid.
        
        Args:
            node: Node where generator is connected
            min_power: Minimum power output (MW)
            max_power: Maximum power output (MW)
            cost_coeff: Quadratic cost coefficients (a, b, c)
            ramp_rate: Ramping rate (MW/hour)
            
        Returns:
            Generator ID
        """
        generator_id = len(self.generators)
        generator = Generator(
            id=generator_id,
            node=node,
            min_power=min_power,
            max_power=max_power,
            cost_coeff=cost_coeff,
            ramp_rate=ramp_rate
        )
        self.generators.append(generator)
        return generator_id
    
    def add_load(self, node: int, demand: float, priority: int = 3,
                elasticity: float = 0.1) -> int:
        """
        Add a load to the grid.
        
        Args:
            node: Node where load is connected
            demand: Load demand (MW)
            priority: Load shedding priority (1=highest, 5=lowest)
            elasticity: Demand response elasticity
            
        Returns:
            Load ID
        """
        load_id = len(self.loads)
        load = Load(
            id=load_id,
            node=node,
            demand=demand,
            priority=priority,
            elasticity=elasticity
        )
        self.loads.append(load)
        return load_id
    
    def build_admittance_matrix(self, line_status: np.ndarray) -> np.ndarray:
        """
        Build the bus admittance matrix based on active lines.
        
        Args:
            line_status: Binary array indicating line status (1=active, 0=inactive)
            
        Returns:
            Complex admittance matrix
        """
        Y = np.zeros((self.num_nodes, self.num_nodes), dtype=complex)
        
        for i, line in enumerate(self.lines):
            if i < len(line_status) and line_status[i] == 1:
                # Line is active
                susceptance = -1j / line.reactance  # Assume pure reactance
                
                # Off-diagonal elements
                Y[line.from_node, line.to_node] += susceptance
                Y[line.to_node, line.from_node] += susceptance
                
                # Diagonal elements (negative sum of off-diagonal)
                Y[line.from_node, line.from_node] -= susceptance
                Y[line.to_node, line.to_node] -= susceptance
        
        return Y
    
    def calculate_power_flow(self, voltage_magnitudes: np.ndarray,
                            voltage_angles: np.ndarray,
                            line_status: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate power flow using DC power flow approximation.
        
        Args:
            voltage_magnitudes: Bus voltage magnitudes (per-unit)
            voltage_angles: Bus voltage angles (radians)
            line_status: Binary array indicating line status
            
        Returns:
            Tuple of (real_power_injection, reactive_power_injection)
        """
        Y = self.build_admittance_matrix(line_status)
        
        # DC power flow approximation
        B = Y.imag  # Susceptance matrix
        P = np.real(voltage_magnitudes[:, np.newaxis] * voltage_magnitudes[np.newaxis, :] * 
                   B * np.sin(voltage_angles[:, np.newaxis] - voltage_angles[np.newaxis, :]))
        
        real_power = np.sum(P, axis=1)
        
        # Simplified reactive power calculation
        reactive_power = np.zeros_like(real_power)
        
        return real_power, reactive_power
    
    def check_line_constraints(self, voltage_magnitudes: np.ndarray,
                              voltage_angles: np.ndarray,
                              line_status: np.ndarray) -> List[bool]:
        """
        Check thermal limits for all transmission lines.
        
        Args:
            voltage_magnitudes: Bus voltage magnitudes
            voltage_angles: Bus voltage angles
            line_status: Binary array indicating line status
            
        Returns:
            List of boolean values indicating constraint satisfaction
        """
        constraints_satisfied = []
        
        for i, line in enumerate(self.lines):
            if i < len(line_status) and line_status[i] == 1:
                # Calculate line flow
                v_from = voltage_magnitudes[line.from_node]
                v_to = voltage_magnitudes[line.to_node]
                theta_from = voltage_angles[line.from_node]
                theta_to = voltage_angles[line.to_node]
                
                # Power flow on line (simplified)
                flow = abs(v_from * v_to * (theta_from - theta_to) / line.reactance)
                
                # Check thermal limit
                constraints_satisfied.append(flow <= line.capacity)
            else:
                constraints_satisfied.append(True)  # Inactive lines don't violate constraints
        
        return constraints_satisfied
    
    def get_total_generation_cost(self, generator_dispatch: np.ndarray) -> float:
        """
        Calculate total generation cost.
        
        Args:
            generator_dispatch: Power output for each generator
            
        Returns:
            Total generation cost
        """
        total_cost = 0.0
        for i, gen in enumerate(self.generators):
            if i < len(generator_dispatch):
                total_cost += gen.cost(generator_dispatch[i])
        return total_cost
    
    def get_total_load(self) -> float:
        """Get total system load."""
        return sum(load.demand for load in self.loads)
    
    def get_network_statistics(self, line_status: np.ndarray) -> Dict[str, float]:
        """
        Calculate network connectivity and reliability statistics.
        
        Args:
            line_status: Binary array indicating line status
            
        Returns:
            Dictionary of network statistics
        """
        # Create active network graph
        active_graph = nx.Graph()
        active_graph.add_nodes_from(range(self.num_nodes))
        
        for i, line in enumerate(self.lines):
            if i < len(line_status) and line_status[i] == 1:
                active_graph.add_edge(line.from_node, line.to_node)
        
        # Calculate statistics
        stats = {
            'connectivity': np.sum(line_status) / len(line_status),
            'is_connected': nx.is_connected(active_graph),
            'num_components': nx.number_connected_components(active_graph),
            'average_clustering': nx.average_clustering(active_graph),
            'diameter': nx.diameter(active_graph) if nx.is_connected(active_graph) else np.inf
        }
        
        return stats
    
    def validate_solution_feasibility(self, chromosome) -> Tuple[bool, List[str]]:
        """
        Validate if a chromosome represents a feasible solution.
        
        Args:
            chromosome: GridChromosome to validate
            
        Returns:
            Tuple of (is_feasible, list_of_violations)
        """
        violations = []
        
        # Check generator limits
        for i, gen in enumerate(self.generators):
            if i < len(chromosome.generator_dispatch):
                power = chromosome.generator_dispatch[i]
                if power < gen.min_power or power > gen.max_power:
                    violations.append(f"Generator {i} power limit violation: {power:.2f} MW")
        
        # Check voltage limits
        min_v, max_v = self.voltage_limits
        for i, voltage in enumerate(chromosome.voltage_levels):
            if voltage < min_v or voltage > max_v:
                violations.append(f"Voltage limit violation at node {i}: {voltage:.3f} pu")
        
        # Check power balance
        total_gen = chromosome.get_total_generation()
        total_load = self.get_total_load()
        if not chromosome.validate_power_balance(total_load):
            violations.append(f"Power balance violation: Gen={total_gen:.2f}, Load={total_load:.2f}")
        
        # Check network connectivity
        stats = self.get_network_statistics(chromosome.line_status)
        if not stats['is_connected']:
            violations.append("Network is not connected")
        
        return len(violations) == 0, violations