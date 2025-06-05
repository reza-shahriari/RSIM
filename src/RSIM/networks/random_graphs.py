import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
from typing import Optional, List, Dict, Any
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ..core.base import BaseSimulation, SimulationResult


class ErdosRenyiGraph(BaseSimulation):
    """
    Erdős-Rényi random graph simulation and analysis.
    
    The Erdős-Rényi model generates random graphs where each possible edge 
    between n vertices is included independently with probability p. This is 
    one of the most fundamental random graph models in network theory.
    
    Mathematical Background:
    -----------------------
    - G(n,p) model: n vertices, each edge exists with probability p
    - Expected number of edges: E[m] = p * n(n-1)/2
    - Expected degree: E[k] = p(n-1)
    - Degree distribution: Binomial → Poisson for large n
    - Giant component emerges at p = 1/n (phase transition)
    - Clustering coefficient: C = p (independent edges)
    
    Applications:
    ------------
    - Social network modeling
    - Internet topology analysis
    - Epidemic spreading models
    - Percolation theory
    - Random network benchmarks
    - Communication networks
    
    Parameters:
    -----------
    n_nodes : int, default=100
        Number of vertices in the graph
    edge_probability : float, default=0.1
        Probability of edge creation between any two nodes
    random_seed : int, optional
        Seed for random number generator
    """
    
    def __init__(self, n_nodes: int = 100, edge_probability: float = 0.1, 
                 random_seed: Optional[int] = None):
        super().__init__("Erdős-Rényi Random Graph")
        
        self.n_nodes = n_nodes
        self.edge_probability = edge_probability
        
        self.parameters.update({
            'n_nodes': n_nodes,
            'edge_probability': edge_probability,
            'random_seed': random_seed
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        self.graph = None
        self.is_configured = True
    
    def configure(self, n_nodes: int = 100, edge_probability: float = 0.1) -> bool:
        """Configure Erdős-Rényi graph parameters"""
        self.n_nodes = n_nodes
        self.edge_probability = edge_probability
        
        self.parameters.update({
            'n_nodes': n_nodes,
            'edge_probability': edge_probability
        })
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Generate and analyze Erdős-Rényi random graph"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Generate the random graph
        self.graph = nx.erdos_renyi_graph(self.n_nodes, self.edge_probability)
        
        # Calculate graph properties
        n_edges = self.graph.number_of_edges()
        degrees = dict(self.graph.degree())
        degree_sequence = list(degrees.values())
        
        # Basic statistics
        avg_degree = np.mean(degree_sequence)
        degree_variance = np.var(degree_sequence)
        density = nx.density(self.graph)
        
        # Connectivity analysis
        is_connected = nx.is_connected(self.graph)
        n_components = nx.number_connected_components(self.graph)
        
        if is_connected:
            diameter = nx.diameter(self.graph)
            avg_path_length = nx.average_shortest_path_length(self.graph)
        else:
            # For disconnected graphs, analyze largest component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            largest_subgraph = self.graph.subgraph(largest_cc)
            diameter = nx.diameter(largest_subgraph) if len(largest_cc) > 1 else 0
            avg_path_length = nx.average_shortest_path_length(largest_subgraph) if len(largest_cc) > 1 else 0
        
        # Clustering
        clustering_coeff = nx.average_clustering(self.graph)
        
        # Component sizes
        component_sizes = [len(c) for c in nx.connected_components(self.graph)]
        largest_component_size = max(component_sizes)
        
        execution_time = time.time() - start_time
        
        # Theoretical values
        expected_edges = self.edge_probability * self.n_nodes * (self.n_nodes - 1) / 2
        expected_degree = self.edge_probability * (self.n_nodes - 1)
        expected_clustering = self.edge_probability
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'n_nodes': self.n_nodes,
                'n_edges': n_edges,
                'density': density,
                'avg_degree': avg_degree,
                'degree_variance': degree_variance,
                'is_connected': is_connected,
                'n_components': n_components,
                'diameter': diameter,
                'avg_path_length': avg_path_length,
                'clustering_coefficient': clustering_coeff,
                'largest_component_size': largest_component_size,
                'component_sizes': component_sizes
            },
            statistics={
                'theoretical_edges': expected_edges,
                'empirical_edges': n_edges,
                'theoretical_degree': expected_degree,
                'empirical_degree': avg_degree,
                'theoretical_clustering': expected_clustering,
                'empirical_clustering': clustering_coeff
            },
            raw_data={
                'graph': self.graph,
                'degree_sequence': degree_sequence,
                'adjacency_matrix': nx.adjacency_matrix(self.graph).toarray()
            },
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, **kwargs) -> None:
        """Visualize the Erdős-Rényi graph and its properties"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        graph = result.raw_data['graph']
        degree_sequence = result.raw_data['degree_sequence']
        
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Graph layout
        ax1 = plt.subplot(2, 3, 1)
        pos = nx.spring_layout(graph, seed=42)
        
        # Color nodes by degree
        node_colors = [graph.degree(node) for node in graph.nodes()]
        nx.draw(graph, pos, node_color=node_colors, node_size=50, 
                with_labels=False, edge_color='gray', alpha=0.7, cmap='viridis')
        ax1.set_title(f'Erdős-Rényi Graph (n={self.n_nodes}, p={self.edge_probability})')
        
        # Plot 2: Degree distribution
        ax2 = plt.subplot(2, 3, 2)
        unique_degrees, counts = np.unique(degree_sequence, return_counts=True)
        ax2.bar(unique_degrees, counts, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Degree Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Add theoretical Poisson overlay for large n
        if self.n_nodes > 50:
            from scipy.stats import poisson
            lambda_param = self.edge_probability * (self.n_nodes - 1)
            x_theory = np.arange(0, max(unique_degrees) + 1)
            y_theory = poisson.pmf(x_theory, lambda_param) * len(degree_sequence)
            ax2.plot(x_theory, y_theory, 'r-', linewidth=2, label=f'Poisson(λ={lambda_param:.1f})')
            ax2.legend()
        
        # Plot 3: Component size distribution
        ax3 = plt.subplot(2, 3, 3)
        component_sizes = result.results['component_sizes']
        unique_sizes, size_counts = np.unique(component_sizes, return_counts=True)
        ax3.bar(unique_sizes, size_counts, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Component Size')
        ax3.set_ylabel('Number of Components')
        ax3.set_title('Component Size Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Statistics comparison
        ax4 = plt.subplot(2, 3, 4)
        categories = ['Edges', 'Avg Degree', 'Clustering']
        theoretical = [result.statistics['theoretical_edges'], 
                      result.statistics['theoretical_degree'],
                      result.statistics['theoretical_clustering']]
        empirical = [result.statistics['empirical_edges'],
                    result.statistics['empirical_degree'],
                    result.statistics['empirical_clustering']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax4.bar(x - width/2, theoretical, width, label='Theoretical', alpha=0.7)
        ax4.bar(x + width/2, empirical, width, label='Empirical', alpha=0.7)
        ax4.set_ylabel('Value')
        ax4.set_title('Theoretical vs Empirical')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Network properties summary
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        props_text = f"""Network Properties:
        
Nodes: {result.results['n_nodes']}
Edges: {result.results['n_edges']}
Density: {result.results['density']:.4f}
Connected: {result.results['is_connected']}
Components: {result.results['n_components']}
Largest Component: {result.results['largest_component_size']}
Diameter: {result.results['diameter']}
Avg Path Length: {result.results['avg_path_length']:.3f}
Clustering: {result.results['clustering_coefficient']:.4f}
        """
        
        ax5.text(0.1, 0.9, props_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Plot 6: Adjacency matrix (for small graphs)
        ax6 = plt.subplot(2, 3, 6)
        if self.n_nodes <= 50:
            adj_matrix = result.raw_data['adjacency_matrix']
            im = ax6.imshow(adj_matrix, cmap='Blues', interpolation='nearest')
            ax6.set_title('Adjacency Matrix')
            ax6.set_xlabel('Node Index')
            ax6.set_ylabel('Node Index')
            plt.colorbar(im, ax=ax6)
        else:
            ax6.text(0.5, 0.5, f'Adjacency matrix too large\nto display ({self.n_nodes}×{self.n_nodes})',
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Adjacency Matrix')
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'n_nodes': {
                'type': 'int',
                'default': 100,
                'min': 10,
                'max': 1000,
                'description': 'Number of nodes in the graph'
            },
            'edge_probability': {
                'type': 'float',
                'default': 0.1,
                'min': 0.0,
                'max': 1.0,
                'description': 'Probability of edge creation between nodes'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility (optional)'
            }
        }
    
    def validate_parameters(self) -> List[str]:
        """Validate simulation parameters"""
        errors = []
        if self.n_nodes < 2:
            errors.append("n_nodes must be at least 2")
        if self.n_nodes > 1000:
            errors.append("n_nodes should not exceed 1000 for performance reasons")
        if not (0 <= self.edge_probability <= 1):
            errors.append("edge_probability must be between 0 and 1")
        return errors


class BarabasiAlbertGraph(BaseSimulation):
    """
    Barabási-Albert preferential attachment model simulation.
    
    The Barabási-Albert model generates scale-free networks through preferential 
    attachment: new nodes connect to existing nodes with probability proportional 
    to their degree. This creates networks with power-law degree distributions.
    
    Mathematical Background:
    -----------------------
    - Starts with m0 initial nodes
    - At each step, add one node with m edges (m ≤ m0)
    - Connection probability: P(ki) ∝ ki (preferential attachment)
    - Degree distribution: P(k) ∝ k^(-3) for large k
    - Average degree: <k> = 2m
    - No clustering in basic model
    
    Applications:
    ------------
    - Social networks (friendship, citations)
    - World Wide Web structure
    - Protein interaction networks
    - Economic networks
    - Infrastructure networks
    - Scientific collaboration networks
    
    Parameters:
    -----------
    n_nodes : int, default=100
        Total number of nodes in final graph
    m_edges : int, default=2
        Number of edges each new node creates
    random_seed : int, optional
        Seed for random number generator
    """
    
    def __init__(self, n_nodes: int = 100, m_edges: int = 2, 
                 random_seed: Optional[int] = None):
        super().__init__("Barabási-Albert Scale-Free Graph")
        
        self.n_nodes = n_nodes
        self.m_edges = m_edges
        
        self.parameters.update({
            'n_nodes': n_nodes,
            'm_edges': m_edges,
            'random_seed': random_seed
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
                    
        self.graph = None
        self.is_configured = True
    
    def configure(self, n_nodes: int = 100, m_edges: int = 2) -> bool:
        """Configure Barabási-Albert graph parameters"""
        self.n_nodes = n_nodes
        self.m_edges = m_edges
        
        self.parameters.update({
            'n_nodes': n_nodes,
            'm_edges': m_edges
        })
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Generate and analyze Barabási-Albert scale-free graph"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Generate the scale-free graph
        self.graph = nx.barabasi_albert_graph(self.n_nodes, self.m_edges)
        
        # Calculate graph properties
        n_edges = self.graph.number_of_edges()
        degrees = dict(self.graph.degree())
        degree_sequence = list(degrees.values())
        
        # Basic statistics
        avg_degree = np.mean(degree_sequence)
        degree_variance = np.var(degree_sequence)
        density = nx.density(self.graph)
        
        # Scale-free properties
        max_degree = max(degree_sequence)
        min_degree = min(degree_sequence)
        
        # Connectivity (BA graphs are always connected)
        diameter = nx.diameter(self.graph)
        avg_path_length = nx.average_shortest_path_length(self.graph)
        clustering_coeff = nx.average_clustering(self.graph)
        
        # Hub analysis (top 10% highest degree nodes)
        sorted_degrees = sorted(degree_sequence, reverse=True)
        n_hubs = max(1, self.n_nodes // 10)
        hub_degrees = sorted_degrees[:n_hubs]
        hub_fraction = n_hubs / self.n_nodes
        
        # Degree distribution analysis for power law
        unique_degrees, counts = np.unique(degree_sequence, return_counts=True)
        
        execution_time = time.time() - start_time
        
        # Theoretical values
        expected_edges = self.m_edges * self.n_nodes
        expected_avg_degree = 2 * self.m_edges
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'n_nodes': self.n_nodes,
                'n_edges': n_edges,
                'density': density,
                'avg_degree': avg_degree,
                'degree_variance': degree_variance,
                'max_degree': max_degree,
                'min_degree': min_degree,
                'diameter': diameter,
                'avg_path_length': avg_path_length,
                'clustering_coefficient': clustering_coeff,
                'hub_degrees': hub_degrees,
                'hub_fraction': hub_fraction
            },
            statistics={
                'theoretical_edges': expected_edges,
                'empirical_edges': n_edges,
                'theoretical_avg_degree': expected_avg_degree,
                'empirical_avg_degree': avg_degree,
                'degree_distribution': (unique_degrees, counts)
            },
            raw_data={
                'graph': self.graph,
                'degree_sequence': degree_sequence,
                'adjacency_matrix': nx.adjacency_matrix(self.graph).toarray()
            },
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, **kwargs) -> None:
        """Visualize the Barabási-Albert graph and its properties"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        graph = result.raw_data['graph']
        degree_sequence = result.raw_data['degree_sequence']
        
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Graph layout with hub highlighting
        ax1 = plt.subplot(2, 3, 1)
        pos = nx.spring_layout(graph, seed=42)
        
        # Color and size nodes by degree
        node_colors = [graph.degree(node) for node in graph.nodes()]
        node_sizes = [50 + 200 * (graph.degree(node) / max(degree_sequence)) for node in graph.nodes()]
        
        nx.draw(graph, pos, node_color=node_colors, node_size=node_sizes,
                with_labels=False, edge_color='gray', alpha=0.7, cmap='Reds')
        ax1.set_title(f'Barabási-Albert Graph (n={self.n_nodes}, m={self.m_edges})')
        
        # Plot 2: Degree distribution (log-log scale)
        ax2 = plt.subplot(2, 3, 2)
        unique_degrees, counts = result.statistics['degree_distribution']
        
        # Regular plot
        ax2.loglog(unique_degrees, counts, 'bo', alpha=0.7, markersize=6)
        ax2.set_xlabel('Degree (log scale)')
        ax2.set_ylabel('Frequency (log scale)')
        ax2.set_title('Degree Distribution (Log-Log)')
        ax2.grid(True, alpha=0.3)
        
        # Fit power law for visualization
        if len(unique_degrees) > 3:
            # Simple power law fit
            log_degrees = np.log(unique_degrees[unique_degrees > 0])
            log_counts = np.log(counts[unique_degrees > 0])
            
            # Linear fit in log space
            coeffs = np.polyfit(log_degrees, log_counts, 1)
            gamma = -coeffs[0]  # Power law exponent
            
            # Plot fitted line
            x_fit = np.logspace(np.log10(min(unique_degrees)), np.log10(max(unique_degrees)), 50)
            y_fit = np.exp(coeffs[1]) * x_fit ** coeffs[0]
            ax2.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Power law (γ≈{gamma:.2f})')
            ax2.legend()
        
        # Plot 3: Hub analysis
        ax3 = plt.subplot(2, 3, 3)
        hub_degrees = result.results['hub_degrees']
        hub_ranks = range(1, len(hub_degrees) + 1)
        
        ax3.bar(hub_ranks, hub_degrees, alpha=0.7, color='red', edgecolor='black')
        ax3.set_xlabel('Hub Rank')
        ax3.set_ylabel('Degree')
        ax3.set_title(f'Top {len(hub_degrees)} Hubs by Degree')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Degree vs clustering coefficient
        ax4 = plt.subplot(2, 3, 4)
        clustering_dict = nx.clustering(graph)
        degrees_for_clustering = [graph.degree(node) for node in graph.nodes()]
        clustering_values = [clustering_dict[node] for node in graph.nodes()]
        
        ax4.scatter(degrees_for_clustering, clustering_values, alpha=0.6, s=30)
        ax4.set_xlabel('Node Degree')
        ax4.set_ylabel('Clustering Coefficient')
        ax4.set_title('Degree vs Clustering')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Network properties summary
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        props_text = f"""Network Properties:
        
Nodes: {result.results['n_nodes']}
Edges: {result.results['n_edges']}
Density: {result.results['density']:.4f}
Avg Degree: {result.results['avg_degree']:.2f}
Max Degree: {result.results['max_degree']}
Min Degree: {result.results['min_degree']}
Diameter: {result.results['diameter']}
Avg Path Length: {result.results['avg_path_length']:.3f}
Clustering: {result.results['clustering_coefficient']:.4f}
Hub Fraction: {result.results['hub_fraction']:.2%}
        """
        
        ax5.text(0.1, 0.9, props_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
        
        # Plot 6: Cumulative degree distribution
        ax6 = plt.subplot(2, 3, 6)
        sorted_degrees = np.sort(degree_sequence)
        cumulative = 1 - np.arange(1, len(sorted_degrees) + 1) / len(sorted_degrees)
        
        ax6.loglog(sorted_degrees, cumulative, 'go', alpha=0.7, markersize=4)
        ax6.set_xlabel('Degree (log scale)')
        ax6.set_ylabel('P(X ≥ k) (log scale)')
        ax6.set_title('Cumulative Degree Distribution')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'n_nodes': {
                'type': 'int',
                'default': 100,
                'min': 10,
                'max': 1000,
                'description': 'Total number of nodes in the graph'
            },
            'm_edges': {
                'type': 'int',
                'default': 2,
                'min': 1,
                'max': 10,
                'description': 'Number of edges each new node creates'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility (optional)'
            }
        }
    
    def validate_parameters(self) -> List[str]:
        """Validate simulation parameters"""
        errors = []
        if self.n_nodes < 2:
            errors.append("n_nodes must be at least 2")
        if self.n_nodes > 1000:
            errors.append("n_nodes should not exceed 1000 for performance reasons")
        if self.m_edges < 1:
            errors.append("m_edges must be at least 1")
        if self.m_edges >= self.n_nodes:
            errors.append("m_edges must be less than n_nodes")
        return errors
