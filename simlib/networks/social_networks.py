import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
from typing import Optional, List, Dict, Any, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ..core.base import BaseSimulation, SimulationResult


class SocialNetworkDiffusion(BaseSimulation):
    """
    Social network diffusion simulation modeling information, innovation, or behavior spread.
    
    This simulation models how information, innovations, behaviors, or opinions spread 
    through social networks. It incorporates social influence, network effects, and 
    individual adoption thresholds to study diffusion processes in social systems.
    
    Mathematical Background:
    -----------------------
    Diffusion Models:
    - Threshold Model: Individuals adopt when fraction of neighbors exceeds threshold
    - Independent Cascade: Each adopter has one chance to influence each neighbor
    - Linear Threshold: Weighted influence from neighbors
    - Complex Contagion: Requires multiple exposures for adoption
    
    Key Concepts:
    - Social influence: neighbors affect adoption probability
    - Network effects: structure determines diffusion patterns
    - Homophily: similar individuals are more connected
    - Weak ties: bridges between communities enable spread
    
    Applications:
    ------------
    - Innovation adoption (new technologies, products)
    - Information spreading (news, rumors, viral content)
    - Behavior change (health behaviors, social movements)
    - Opinion dynamics and polarization
    - Marketing and viral campaigns
    - Social movements and collective action
    
    Parameters:
    -----------
    network : networkx.Graph or None
        Social network structure (if None, generates small-world network)
    diffusion_model : str, default='threshold'
        Diffusion model: 'threshold', 'cascade', 'complex_contagion'
    adoption_threshold : float, default=0.3
        Fraction of neighbors needed for adoption (threshold model)
    influence_probability : float, default=0.1
        Probability of influence per neighbor (cascade model)
    initial_adopters : int, default=5
        Number of initial adopters (seeds)
    max_time_steps : int, default=50
        Maximum simulation time steps
    network_params : dict, optional
        Parameters for network generation
    random_seed : int, optional
        Seed for random number generator
    """
    
    def __init__(self, network: Optional[nx.Graph] = None, 
                 diffusion_model: str = 'threshold', adoption_threshold: float = 0.3,
                 influence_probability: float = 0.1, initial_adopters: int = 5,
                 max_time_steps: int = 50, network_params: Optional[Dict] = None,
                 random_seed: Optional[int] = None):
        super().__init__("Social Network Diffusion")
        
        self.network = network
        self.diffusion_model = diffusion_model
        self.adoption_threshold = adoption_threshold
        self.influence_probability = influence_probability
        self.initial_adopters = initial_adopters
        self.max_time_steps = max_time_steps
        self.network_params = network_params or {'n_nodes': 100, 'k': 6, 'p': 0.1}
        
        self.parameters.update({
            'diffusion_model': diffusion_model,
            'adoption_threshold': adoption_threshold,
            'influence_probability': influence_probability,
            'initial_adopters': initial_adopters,
            'max_time_steps': max_time_steps,
            'network_params': self.network_params,
            'random_seed': random_seed
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        self.diffusion_data = None
        self.is_configured = True
    
    def configure(self, network: Optional[nx.Graph] = None, diffusion_model: str = 'threshold',
                 adoption_threshold: float = 0.3, influence_probability: float = 0.1,
                 initial_adopters: int = 5, max_time_steps: int = 50) -> bool:
        """Configure social diffusion parameters"""
        self.network = network
        self.diffusion_model = diffusion_model
        self.adoption_threshold = adoption_threshold
        self.influence_probability = influence_probability
        self.initial_adopters = initial_adopters
        self.max_time_steps = max_time_steps
        
        self.parameters.update({
            'diffusion_model': diffusion_model,
            'adoption_threshold': adoption_threshold,
            'influence_probability': influence_probability,
            'initial_adopters': initial_adopters,
            'max_time_steps': max_time_steps
        })
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Run social network diffusion simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Create network if not provided (small-world network for social networks)
        if self.network is None:
            self.network = nx.watts_strogatz_graph(
                self.network_params['n_nodes'],
                self.network_params['k'],
                self.network_params['p']
            )
        
        nodes = list(self.network.nodes())
        n_nodes = len(nodes)
        
        # Initialize adoption states: 0 = non-adopter, 1 = adopter
        adoption_states = np.zeros(n_nodes, dtype=int)
        adoption_times = np.full(n_nodes, -1)  # Time of adoption (-1 if never adopted)
        
        # Select initial adopters (high-degree nodes or random)
        if self.initial_adopters > 0:
            # Choose high-degree nodes as initial adopters (influencers)
            degrees = [(node, self.network.degree(node)) for node in nodes]
            degrees.sort(key=lambda x: x[1], reverse=True)
            initial_nodes = [node for node, _ in degrees[:self.initial_adopters]]
            
            for node in initial_nodes:
                adoption_states[node] = 1
                adoption_times[node] = 0
        
        # Track diffusion over time
        time_series = {
            'time': [],
            'adopters': [],
            'new_adopters': [],
            'adoption_rate': [],
            'diffusion_speed': []
        }
        
        # Store adoption states over time for visualization
        adoption_states_over_time = []
        
        # Track influence attempts for cascade model
        influenced_this_round = set()
        
        for t in range(self.max_time_steps):
            current_adopters = np.sum(adoption_states)
            adoption_states_over_time.append(adoption_states.copy())
            
            # Count new adopters this round
            new_adopters_count = np.sum(adoption_times == t)
            
            time_series['time'].append(t)
            time_series['adopters'].append(current_adopters)
            time_series['new_adopters'].append(new_adopters_count)
            time_series['adoption_rate'].append(current_adopters / n_nodes)
            
            # Calculate diffusion speed (new adopters per time step)
            if t > 0:
                speed = new_adopters_count
            else:
                speed = current_adopters
            time_series['diffusion_speed'].append(speed)
            
            # Stop if everyone has adopted or no new adoptions
            if current_adopters == n_nodes or (t > 0 and new_adopters_count == 0):
                break
            
            new_adoption_states = adoption_states.copy()
            
            if self.diffusion_model == 'threshold':
                # Threshold model: adopt if fraction of neighbors exceeds threshold
                for node in nodes:
                    if adoption_states[node] == 0:  # Non-adopter
                        neighbors = list(self.network.neighbors(node))
                        if len(neighbors) > 0:
                            adopter_neighbors = sum(1 for neighbor in neighbors 
                                                  if adoption_states[neighbor] == 1)
                            adoption_fraction = adopter_neighbors / len(neighbors)
                            
                            if adoption_fraction >= self.adoption_threshold:
                                new_adoption_states[node] = 1
                                adoption_times[node] = t + 1
            
            elif self.diffusion_model == 'cascade':
                # Independent cascade: each adopter tries to influence neighbors once
                newly_influenced = set()
                
                for node in nodes:
                    if adoption_states[node] == 1 and node not in influenced_this_round:
                        # This adopter tries to influence neighbors
                        neighbors = list(self.network.neighbors(node))
                        for neighbor in neighbors:
                            if adoption_states[neighbor] == 0:  # Non-adopter
                                if np.random.random() < self.influence_probability:
                                    new_adoption_states[neighbor] = 1
                                    adoption_times[neighbor] = t + 1
                                    newly_influenced.add(neighbor)
                        
                        influenced_this_round.add(node)
                
                # Update influenced set for next round
                influenced_this_round.update(newly_influenced)
            
            elif self.diffusion_model == 'complex_contagion':
                # Complex contagion: need multiple adopter neighbors
                min_adopter_neighbors = max(2, int(self.adoption_threshold * 10))
                
                for node in nodes:
                    if adoption_states[node] == 0:  # Non-adopter
                        neighbors = list(self.network.neighbors(node))
                        adopter_neighbors = sum(1 for neighbor in neighbors 
                                              if adoption_states[neighbor] == 1)
                        
                        if adopter_neighbors >= min_adopter_neighbors:
                            # Additional probability based on social reinforcement
                            adoption_prob = min(1.0, adopter_neighbors * self.influence_probability)
                            if np.random.random() < adoption_prob:
                                new_adoption_states[node] = 1
                                adoption_times[node] = t + 1
            
            adoption_states = new_adoption_states
        
        execution_time = time.time() - start_time
        
        # Calculate diffusion statistics
        final_adoption_rate = np.sum(adoption_states) / n_nodes
        total_adopters = np.sum(adoption_states)
        
        # Time to adoption statistics
        adopted_nodes = adoption_times[adoption_times >= 0]
        if len(adopted_nodes) > 0:
            avg_adoption_time = np.mean(adopted_nodes)
            adoption_time_std = np.std(adopted_nodes)
        else:
            avg_adoption_time = 0
            adoption_time_std = 0
        
        # Network structure analysis
        avg_degree = np.mean([self.network.degree(node) for node in nodes])
        clustering_coefficient = nx.average_clustering(self.network)
        
        # Adopter network properties
        adopter_nodes = [node for node in nodes if adoption_states[node] == 1]
        if len(adopter_nodes) > 1:
            adopter_subgraph = self.network.subgraph(adopter_nodes)
            adopter_density = nx.density(adopter_subgraph)
        else:
            adopter_density = 0
        
        # Calculate diffusion metrics
        diffusion_duration = len(time_series['time'])
        peak_adoption_speed = max(time_series['diffusion_speed']) if time_series['diffusion_speed'] else 0
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'final_adoption_rate': final_adoption_rate,
                'total_adopters': total_adopters,
                'diffusion_duration': diffusion_duration,
                'avg_adoption_time': avg_adoption_time,
                'adoption_time_std': adoption_time_std,
                'peak_adoption_speed': peak_adoption_speed,
                'network_size': n_nodes,
                'network_edges': self.network.number_of_edges(),
                'avg_degree': avg_degree,
                'clustering_coefficient': clustering_coefficient,
                'adopter_density': adopter_density
            },
            statistics={
                'diffusion_model': self.diffusion_model,
                'adoption_threshold': self.adoption_threshold,
                'influence_probability': self.influence_probability,
                'initial_adopters': self.initial_adopters,
                'time_series': time_series
            },
            raw_data={
                'network': self.network,
                'adoption_states': adoption_states,
                'adoption_times': adoption_times,
                'time_series': time_series,
                'adoption_states_over_time': adoption_states_over_time
            },
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, **kwargs) -> None:
        """Visualize social network diffusion simulation results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        network = result.raw_data['network']
        adoption_states = result.raw_data['adoption_states']
        adoption_times = result.raw_data['adoption_times']
        time_series = result.raw_data['time_series']
        
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Network with adoption states
        ax1 = plt.subplot(2, 3, 1)
        pos = nx.spring_layout(network, seed=42)
        
        # Color nodes by adoption state and time
        node_colors = []
        for node in network.nodes():
            if adoption_states[node] == 1:
                # Color by adoption time (earlier = darker)
                adoption_time = adoption_times[node]
                if adoption_time == 0:
                    node_colors.append('red')  # Initial adopters
                else:
                    # Gradient from red to pink based on adoption time
                    intensity = max(0.3, 1.0 - adoption_time / max(adoption_times))
                    node_colors.append((1.0, 1.0 - intensity, 1.0 - intensity))
            else:
                node_colors.append('lightblue')  # Non-adopters
        
        # Node sizes based on degree (influence)
        node_sizes = [100 + 20 * network.degree(node) for node in network.nodes()]
        
        nx.draw(network, pos, node_color=node_colors, node_size=node_sizes,
                with_labels=False, edge_color='gray', alpha=0.7)
        ax1.set_title('Social Network Diffusion\n(Red=Early, Pink=Late, Blue=Non-adopters)')
        
        # Plot 2: Adoption curve over time
        ax2 = plt.subplot(2, 3, 2)
        times = time_series['time']
        ax2.plot(times, time_series['adoption_rate'], 'b-', linewidth=3, label='Adoption Rate')
        ax2.fill_between(times, time_series['adoption_rate'], alpha=0.3)
        
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Adoption Rate')
        ax2.set_title('Diffusion Curve (S-curve)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add final adoption rate text
        final_rate = result.results['final_adoption_rate']
        ax2.text(0.02, 0.98, f'Final Adoption: {final_rate:.1%}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 3: New adopters per time step
        ax3 = plt.subplot(2, 3, 3)
        ax3.bar(times, time_series['new_adopters'], alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('New Adopters')
        ax3.set_title('Adoption Speed Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Adoption time distribution
        ax4 = plt.subplot(2, 3, 4)
        adopted_times = adoption_times[adoption_times >= 0]
        if len(adopted_times) > 0:
            ax4.hist(adopted_times, bins=max(5, len(set(adopted_times))), 
                    alpha=0.7, edgecolor='black', density=True)
            ax4.axvline(x=np.mean(adopted_times), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(adopted_times):.1f}')
            ax4.set_xlabel('Adoption Time')
            ax4.set_ylabel('Density')
            ax4.set_title('Distribution of Adoption Times')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Diffusion statistics
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        stats_text = f"""Diffusion Statistics:
        
Model: {result.statistics['diffusion_model']}
Network: {result.results['network_size']} nodes, {result.results['network_edges']} edges
        
Initial Adopters: {result.statistics['initial_adopters']}
Final Adopters: {result.results['total_adopters']}
Final Adoption Rate: {result.results['final_adoption_rate']:.1%}
        
Diffusion Duration: {result.results['diffusion_duration']} steps
Avg Adoption Time: {result.results['avg_adoption_time']:.2f} steps
Peak Adoption Speed: {result.results['peak_adoption_speed']} adopters/step
        
Network Properties:
Avg Degree: {result.results['avg_degree']:.2f}
Clustering: {result.results['clustering_coefficient']:.3f}
Adopter Density: {result.results['adopter_density']:.3f}
        """
        
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        # Plot 6: Degree vs adoption time analysis
        ax6 = plt.subplot(2, 3, 6)
        degrees = [network.degree(node) for node in network.nodes()]
        
        # Separate adopters and non-adopters
        adopter_degrees = []
        adopter_times = []
        non_adopter_degrees = []
        
        for node in network.nodes():
            degree = network.degree(node)
            if adoption_states[node] == 1:
                adopter_degrees.append(degree)
                adopter_times.append(adoption_times[node])
            else:
                non_adopter_degrees.append(degree)
        
        if adopter_degrees:
            scatter = ax6.scatter(adopter_degrees, adopter_times, c=adopter_times, 
                                cmap='viridis_r', s=50, alpha=0.7, label='Adopters')
            plt.colorbar(scatter, ax=ax6, label='Adoption Time')
        
        if non_adopter_degrees:
            ax6.scatter(non_adopter_degrees, 
                       [max(adoption_times) + 1] * len(non_adopter_degrees),
                       c='red', s=30, alpha=0.5, marker='x', label='Non-adopters')
        
        ax6.set_xlabel('Node Degree')
        ax6.set_ylabel('Adoption Time')
        ax6.set_title('Degree vs Adoption Time')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'diffusion_model': {
                'type': 'choice',
                'default': 'threshold',
                'choices': ['threshold', 'cascade', 'complex_contagion'],
                'description': 'Diffusion model type'
            },
            'adoption_threshold': {
                'type': 'float',
                'default': 0.3,
                'min': 0.0,
                'max': 1.0,
                'description': 'Fraction of neighbors needed for adoption (threshold model)'
            },
            'influence_probability': {
                'type': 'float',
                'default': 0.1,
                'min': 0.0,
                'max': 1.0,
                'description': 'Probability of influence per neighbor (cascade model)'
            },
            'initial_adopters': {
                'type': 'int',
                'default': 5,
                'min': 1,
                'max': 50,
                'description': 'Number of initial adopters (seeds)'
            },
            'max_time_steps': {
                'type': 'int',
                'default': 50,
                'min': 5,
                'max': 200,
                'description': 'Maximum simulation time steps'
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
        if self.diffusion_model not in ['threshold', 'cascade', 'complex_contagion']:
            errors.append("diffusion_model must be 'threshold', 'cascade', or 'complex_contagion'")
        if not (0 <= self.adoption_threshold <= 1):
            errors.append("adoption_threshold must be between 0 and 1")
        if not (0 <= self.influence_probability <= 1):
            errors.append("influence_probability must be between 0 and 1")
        if self.initial_adopters < 1:
            errors.append("initial_adopters must be at least 1")
        if self.max_time_steps < 1:
            errors.append("max_time_steps must be positive")
        return errors
