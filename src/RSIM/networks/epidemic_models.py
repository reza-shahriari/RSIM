import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
from typing import Optional, List, Dict, Any, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ..core.base import BaseSimulation, SimulationResult


class SIRModel(BaseSimulation):
    """
    SIR (Susceptible-Infected-Recovered) epidemic model simulation on networks.
    
    The SIR model is a fundamental compartmental model in epidemiology that 
    divides the population into three states: Susceptible (S), Infected (I), 
    and Recovered (R). This implementation simulates disease spread on network 
    structures where contacts are defined by graph edges.
    
    Mathematical Background:
    -----------------------
    States:
    - S: Susceptible individuals who can become infected
    - I: Infected individuals who can transmit the disease
    - R: Recovered individuals who are immune
    
    Transitions:
    - S → I: Infection with rate β per infected neighbor
    - I → R: Recovery with rate γ
    
    Network SIR dynamics:
    - P(S→I) = 1 - (1-β)^(number of infected neighbors)
    - P(I→R) = γ per time step
    
    Key metrics:
    - Basic reproduction number: R₀ = β/γ * <k>
    - Final epidemic size depends on network structure
    - Epidemic threshold: R₀ > 1
    
    Applications:
    ------------
    - Disease outbreak modeling (COVID-19, influenza)
    - Computer virus spread
    - Information diffusion
    - Social contagion processes
    - Rumor spreading
    - Innovation adoption
    
    Parameters:
    -----------
    network : networkx.Graph or None
        Network structure for epidemic spread (if None, uses Erdős-Rényi)
    infection_rate : float, default=0.1
        Probability of infection per infected neighbor per time step
    recovery_rate : float, default=0.05
        Probability of recovery per time step
    initial_infected : int, default=1
        Number of initially infected nodes
    max_time_steps : int, default=100
        Maximum simulation time steps
    network_params : dict, optional
        Parameters for default network generation
    random_seed : int, optional
        Seed for random number generator
    """
    
    def __init__(self, network: Optional[nx.Graph] = None, infection_rate: float = 0.1,
                 recovery_rate: float = 0.05, initial_infected: int = 1,
                 max_time_steps: int = 100, network_params: Optional[Dict] = None,
                 random_seed: Optional[int] = None):
        super().__init__("SIR Epidemic Model")
        
        self.network = network
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.initial_infected = initial_infected
        self.max_time_steps = max_time_steps
        self.network_params = network_params or {'n_nodes': 100, 'edge_probability': 0.1}
        
        self.parameters.update({
            'infection_rate': infection_rate,
            'recovery_rate': recovery_rate,
            'initial_infected': initial_infected,
            'max_time_steps': max_time_steps,
            'network_params': self.network_params,
            'random_seed': random_seed
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        self.epidemic_data = None
        self.is_configured = True
    
    def configure(self, network: Optional[nx.Graph] = None, infection_rate: float = 0.1,
                 recovery_rate: float = 0.05, initial_infected: int = 1,
                 max_time_steps: int = 100) -> bool:
        """Configure SIR model parameters"""
        self.network = network
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.initial_infected = initial_infected
        self.max_time_steps = max_time_steps
        self.parameters.update({
                    'infection_rate': infection_rate,
                    'recovery_rate': recovery_rate,
                    'initial_infected': initial_infected,
                    'max_time_steps': max_time_steps
                })
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Run SIR epidemic simulation on network"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Create network if not provided
        if self.network is None:
            self.network = nx.erdos_renyi_graph(
                self.network_params['n_nodes'],
                self.network_params['edge_probability']
            )
        
        n_nodes = self.network.number_of_nodes()
        nodes = list(self.network.nodes())
        
        # Initialize states: 0=S, 1=I, 2=R
        states = np.zeros(n_nodes, dtype=int)
        
        # Randomly select initial infected nodes
        initial_infected_nodes = np.random.choice(
            nodes, size=min(self.initial_infected, n_nodes), replace=False
        )
        for node in initial_infected_nodes:
            states[node] = 1
        
        # Track epidemic over time
        time_series = {
            'susceptible': [],
            'infected': [],
            'recovered': [],
            'time': []
        }
        
        # Store node states over time for visualization
        node_states_over_time = []
        
        for t in range(self.max_time_steps):
            # Count current states
            n_susceptible = np.sum(states == 0)
            n_infected = np.sum(states == 1)
            n_recovered = np.sum(states == 2)
            
            time_series['susceptible'].append(n_susceptible)
            time_series['infected'].append(n_infected)
            time_series['recovered'].append(n_recovered)
            time_series['time'].append(t)
            
            node_states_over_time.append(states.copy())
            
            # Stop if no infected individuals
            if n_infected == 0:
                break
            
            new_states = states.copy()
            
            # Process infections (S → I)
            for node in nodes:
                if states[node] == 0:  # Susceptible
                    infected_neighbors = sum(1 for neighbor in self.network.neighbors(node) 
                                           if states[neighbor] == 1)
                    if infected_neighbors > 0:
                        # Probability of infection
                        prob_infection = 1 - (1 - self.infection_rate) ** infected_neighbors
                        if np.random.random() < prob_infection:
                            new_states[node] = 1
            
            # Process recoveries (I → R)
            for node in nodes:
                if states[node] == 1:  # Infected
                    if np.random.random() < self.recovery_rate:
                        new_states[node] = 2
            
            states = new_states
        
        execution_time = time.time() - start_time
        
        # Calculate epidemic statistics
        final_epidemic_size = time_series['recovered'][-1]
        attack_rate = final_epidemic_size / n_nodes
        peak_infected = max(time_series['infected'])
        peak_time = time_series['infected'].index(peak_infected)
        epidemic_duration = len(time_series['time'])
        
        # Calculate R0 estimate
        avg_degree = np.mean([self.network.degree(node) for node in self.network.nodes()])
        r0_estimate = (self.infection_rate / self.recovery_rate) * avg_degree
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'final_epidemic_size': final_epidemic_size,
                'attack_rate': attack_rate,
                'peak_infected': peak_infected,
                'peak_time': peak_time,
                'epidemic_duration': epidemic_duration,
                'network_size': n_nodes,
                'network_edges': self.network.number_of_edges(),
                'avg_degree': avg_degree
            },
            statistics={
                'r0_estimate': r0_estimate,
                'infection_rate': self.infection_rate,
                'recovery_rate': self.recovery_rate,
                'time_series': time_series
            },
            raw_data={
                'network': self.network,
                'time_series': time_series,
                'node_states_over_time': node_states_over_time,
                'final_states': states
            },
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, **kwargs) -> None:
        """Visualize SIR epidemic simulation results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        network = result.raw_data['network']
        time_series = result.raw_data['time_series']
        node_states_over_time = result.raw_data['node_states_over_time']
        
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Epidemic curves
        ax1 = plt.subplot(2, 3, 1)
        times = time_series['time']
        ax1.plot(times, time_series['susceptible'], 'b-', linewidth=2, label='Susceptible')
        ax1.plot(times, time_series['infected'], 'r-', linewidth=2, label='Infected')
        ax1.plot(times, time_series['recovered'], 'g-', linewidth=2, label='Recovered')
        
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Number of Individuals')
        ax1.set_title('SIR Epidemic Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Network with final states
        ax2 = plt.subplot(2, 3, 2)
        pos = nx.spring_layout(network, seed=42)
        final_states = result.raw_data['final_states']
        
        # Color nodes by final state
        node_colors = ['blue' if state == 0 else 'red' if state == 1 else 'green' 
                      for state in final_states]
        
        nx.draw(network, pos, node_color=node_colors, node_size=50,
                with_labels=False, edge_color='gray', alpha=0.7)
        ax2.set_title('Final Network State\n(Blue=S, Red=I, Green=R)')
        
        # Plot 3: Attack rate vs degree
        ax3 = plt.subplot(2, 3, 3)
        degrees = [network.degree(node) for node in network.nodes()]
        final_states_list = list(final_states)
        
        # Calculate attack rate by degree
        degree_attack_rates = {}
        for node in network.nodes():
            degree = network.degree(node)
            if degree not in degree_attack_rates:
                degree_attack_rates[degree] = {'infected': 0, 'total': 0}
            degree_attack_rates[degree]['total'] += 1
            if final_states[node] == 2:  # Recovered (was infected)
                degree_attack_rates[degree]['infected'] += 1
        
        degrees_unique = sorted(degree_attack_rates.keys())
        attack_rates_by_degree = [degree_attack_rates[d]['infected'] / degree_attack_rates[d]['total'] 
                                 for d in degrees_unique]
        
        ax3.scatter(degrees_unique, attack_rates_by_degree, alpha=0.7, s=50)
        ax3.set_xlabel('Node Degree')
        ax3.set_ylabel('Attack Rate')
        ax3.set_title('Attack Rate vs Node Degree')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Epidemic statistics
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        
        stats_text = f"""Epidemic Statistics:
        
Network Size: {result.results['network_size']}
Initial Infected: {self.initial_infected}
Final Epidemic Size: {result.results['final_epidemic_size']}
Attack Rate: {result.results['attack_rate']:.2%}
Peak Infected: {result.results['peak_infected']}
Peak Time: {result.results['peak_time']}
Duration: {result.results['epidemic_duration']} steps
R₀ Estimate: {result.statistics['r0_estimate']:.2f}
Avg Degree: {result.results['avg_degree']:.2f}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        # Plot 5: Infection rate over time
        ax5 = plt.subplot(2, 3, 5)
        if len(times) > 1:
            infection_rates = []
            for i in range(1, len(times)):
                new_infections = time_series['infected'][i] - time_series['infected'][i-1] + \
                               (time_series['recovered'][i] - time_series['recovered'][i-1])
                susceptible_prev = time_series['susceptible'][i-1]
                if susceptible_prev > 0:
                    infection_rates.append(new_infections / susceptible_prev)
                else:
                    infection_rates.append(0)
            
            ax5.plot(times[1:], infection_rates, 'purple', linewidth=2)
            ax5.set_xlabel('Time Steps')
            ax5.set_ylabel('Infection Rate')
            ax5.set_title('Instantaneous Infection Rate')
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Phase space (S vs I)
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(time_series['susceptible'], time_series['infected'], 'b-', linewidth=2)
        ax6.scatter(time_series['susceptible'][0], time_series['infected'][0], 
                   color='green', s=100, marker='o', label='Start', zorder=5)
        ax6.scatter(time_series['susceptible'][-1], time_series['infected'][-1], 
                   color='red', s=100, marker='s', label='End', zorder=5)
        ax6.set_xlabel('Susceptible')
        ax6.set_ylabel('Infected')
        ax6.set_title('Phase Space (S vs I)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'infection_rate': {
                'type': 'float',
                'default': 0.1,
                'min': 0.0,
                'max': 1.0,
                'description': 'Probability of infection per infected neighbor'
            },
            'recovery_rate': {
                'type': 'float',
                'default': 0.05,
                'min': 0.0,
                'max': 1.0,
                'description': 'Probability of recovery per time step'
            },
            'initial_infected': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 50,
                'description': 'Number of initially infected nodes'
            },
            'max_time_steps': {
                'type': 'int',
                'default': 100,
                'min': 10,
                'max': 1000,
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
        if not (0 <= self.infection_rate <= 1):
            errors.append("infection_rate must be between 0 and 1")
        if not (0 <= self.recovery_rate <= 1):
            errors.append("recovery_rate must be between 0 and 1")
        if self.initial_infected < 1:
            errors.append("initial_infected must be at least 1")
        if self.max_time_steps < 1:
            errors.append("max_time_steps must be positive")
        return errors


class SEIRModel(BaseSimulation):
    """
    SEIR (Susceptible-Exposed-Infected-Recovered) epidemic model simulation.
    
    The SEIR model extends the SIR model by adding an Exposed (E) compartment 
    for individuals who have been infected but are not yet infectious. This 
    models diseases with an incubation period.
    
    Mathematical Background:
    -----------------------
    States:
    - S: Susceptible individuals
    - E: Exposed (infected but not infectious)
    - I: Infected and infectious
    - R: Recovered and immune
    
    Transitions:
    - S → E: Exposure with rate β per infected neighbor
    - E → I: Becoming infectious with rate σ (1/incubation period)
    - I → R: Recovery with rate γ
    
    Parameters:
    -----------
    network : networkx.Graph or None
        Network structure for epidemic spread
    infection_rate : float, default=0.1
        Probability of infection per infected neighbor per time step
    incubation_rate : float, default=0.2
        Probability of becoming infectious per time step (1/incubation period)
    recovery_rate : float, default=0.1
        Probability of recovery per time step
    initial_infected : int, default=1
        Number of initially infected nodes
    max_time_steps : int, default=150
        Maximum simulation time steps
    """
    
    def __init__(self, network: Optional[nx.Graph] = None, infection_rate: float = 0.1,
                 incubation_rate: float = 0.2, recovery_rate: float = 0.1,
                 initial_infected: int = 1, max_time_steps: int = 150,
                 network_params: Optional[Dict] = None, random_seed: Optional[int] = None):
        super().__init__("SEIR Epidemic Model")
        
        self.network = network
        self.infection_rate = infection_rate
        self.incubation_rate = incubation_rate
        self.recovery_rate = recovery_rate
        self.initial_infected = initial_infected
        self.max_time_steps = max_time_steps
        self.network_params = network_params or {'n_nodes': 100, 'edge_probability': 0.1}
        
        self.parameters.update({
            'infection_rate': infection_rate,
            'incubation_rate': incubation_rate,
            'recovery_rate': recovery_rate,
            'initial_infected': initial_infected,
            'max_time_steps': max_time_steps,
            'network_params': self.network_params,
            'random_seed': random_seed
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        self.epidemic_data = None
        self.is_configured = True
    
    def configure(self, network: Optional[nx.Graph] = None, infection_rate: float = 0.1,
                 incubation_rate: float = 0.2, recovery_rate: float = 0.1,
                 initial_infected: int = 1, max_time_steps: int = 150) -> bool:
        """Configure SEIR model parameters"""
        self.network = network
        self.infection_rate = infection_rate
        self.incubation_rate = incubation_rate
        self.recovery_rate = recovery_rate
        self.initial_infected = initial_infected
        self.max_time_steps = max_time_steps
        
        self.parameters.update({
            'infection_rate': infection_rate,
            'incubation_rate': incubation_rate,
            'recovery_rate': recovery_rate,
            'initial_infected': initial_infected,
            'max_time_steps': max_time_steps
        })
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Run SEIR epidemic simulation on network"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Create network if not provided
        if self.network is None:
            self.network = nx.erdos_renyi_graph(
                self.network_params['n_nodes'],
                self.network_params['edge_probability']
            )
        
        n_nodes = self.network.number_of_nodes()
        nodes = list(self.network.nodes())
        
        # Initialize states: 0=S, 1=E, 2=I, 3=R
        states = np.zeros(n_nodes, dtype=int)
        
        # Randomly select initial infected nodes (start as Infected, not Exposed)
        initial_infected_nodes = np.random.choice(
            nodes, size=min(self.initial_infected, n_nodes), replace=False
        )
        for node in initial_infected_nodes:
            states[node] = 2  # Start as Infected
        
        # Track epidemic over time
        time_series = {
            'susceptible': [],
            'exposed': [],
            'infected': [],
            'recovered': [],
            'time': []
        }
        
        # Store node states over time for visualization
        node_states_over_time = []
        
        for t in range(self.max_time_steps):
            # Count current states
            n_susceptible = np.sum(states == 0)
            n_exposed = np.sum(states == 1)
            n_infected = np.sum(states == 2)
            n_recovered = np.sum(states == 3)
            
            time_series['susceptible'].append(n_susceptible)
            time_series['exposed'].append(n_exposed)
            time_series['infected'].append(n_infected)
            time_series['recovered'].append(n_recovered)
            time_series['time'].append(t)
            
            node_states_over_time.append(states.copy())
            
            # Stop if no exposed or infected individuals
            if n_exposed == 0 and n_infected == 0:
                break
            
            new_states = states.copy()
            
            # Process exposures (S → E)
            for node in nodes:
                if states[node] == 0:  # Susceptible
                    infected_neighbors = sum(1 for neighbor in self.network.neighbors(node) 
                                           if states[neighbor] == 2)  # Only infected can transmit
                    if infected_neighbors > 0:
                        # Probability of exposure
                        prob_exposure = 1 - (1 - self.infection_rate) ** infected_neighbors
                        if np.random.random() < prob_exposure:
                            new_states[node] = 1  # Exposed
            
            # Process incubation (E → I)
            for node in nodes:
                if states[node] == 1:  # Exposed
                    if np.random.random() < self.incubation_rate:
                        new_states[node] = 2  # Infected
            
            # Process recoveries (I → R)
            for node in nodes:
                if states[node] == 2:  # Infected
                    if np.random.random() < self.recovery_rate:
                        new_states[node] = 3  # Recovered
            
            states = new_states
        
        execution_time = time.time() - start_time
        
        # Calculate epidemic statistics
        final_epidemic_size = time_series['recovered'][-1]
        attack_rate = final_epidemic_size / n_nodes
        peak_infected = max(time_series['infected'])
        peak_exposed = max(time_series['exposed'])
        peak_infected_time = time_series['infected'].index(peak_infected)
        peak_exposed_time = time_series['exposed'].index(peak_exposed)
        epidemic_duration = len(time_series['time'])
        
        # Calculate R0 estimate
        avg_degree = np.mean([self.network.degree(node) for node in self.network.nodes()])
        r0_estimate = (self.infection_rate / self.recovery_rate) * avg_degree
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'final_epidemic_size': final_epidemic_size,
                'attack_rate': attack_rate,
                'peak_infected': peak_infected,
                'peak_exposed': peak_exposed,
                'peak_infected_time': peak_infected_time,
                'peak_exposed_time': peak_exposed_time,
                'epidemic_duration': epidemic_duration,
                'network_size': n_nodes,
                'network_edges': self.network.number_of_edges(),
                'avg_degree': avg_degree
            },
            statistics={
                'r0_estimate': r0_estimate,
                'infection_rate': self.infection_rate,
                'incubation_rate': self.incubation_rate,
                'recovery_rate': self.recovery_rate,
                'incubation_period': 1/self.incubation_rate,
                'infectious_period': 1/self.recovery_rate,
                'time_series': time_series
            },
            raw_data={
                'network': self.network,
                'time_series': time_series,
                'node_states_over_time': node_states_over_time,
                'final_states': states
            },
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, **kwargs) -> None:
        """Visualize SEIR epidemic simulation results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        network = result.raw_data['network']
        time_series = result.raw_data['time_series']
        node_states_over_time = result.raw_data['node_states_over_time']
        
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: SEIR epidemic curves
        ax1 = plt.subplot(2, 3, 1)
        times = time_series['time']
        ax1.plot(times, time_series['susceptible'], 'b-', linewidth=2, label='Susceptible')
        ax1.plot(times, time_series['exposed'], 'orange', linewidth=2, label='Exposed')
        ax1.plot(times, time_series['infected'], 'r-', linewidth=2, label='Infected')
        ax1.plot(times, time_series['recovered'], 'g-', linewidth=2, label='Recovered')
        
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Number of Individuals')
        ax1.set_title('SEIR Epidemic Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Network with final states
        ax2 = plt.subplot(2, 3, 2)
        pos = nx.spring_layout(network, seed=42)
        final_states = result.raw_data['final_states']
        
        # Color nodes by final state
        color_map = {0: 'blue', 1: 'orange', 2: 'red', 3: 'green'}
        node_colors = [color_map[state] for state in final_states]
        
        nx.draw(network, pos, node_color=node_colors, node_size=50,
                with_labels=False, edge_color='gray', alpha=0.7)
        ax2.set_title('Final Network State\n(Blue=S, Orange=E, Red=I, Green=R)')
        
        # Plot 3: Exposed vs Infected phase plot
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(time_series['exposed'], time_series['infected'], 'purple', linewidth=2)
        ax3.scatter(time_series['exposed'][0], time_series['infected'][0], 
                   color='green', s=100, marker='o', label='Start', zorder=5)
        ax3.scatter(time_series['exposed'][-1], time_series['infected'][-1], 
                   color='red', s=100, marker='s', label='End', zorder=5)
        ax3.set_xlabel('Exposed')
        ax3.set_ylabel('Infected')
        ax3.set_title('Phase Space (E vs I)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Epidemic statistics
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        
        stats_text = f"""SEIR Epidemic Statistics:
        
Network Size: {result.results['network_size']}
Initial Infected: {self.initial_infected}
Final Epidemic Size: {result.results['final_epidemic_size']}
Attack Rate: {result.results['attack_rate']:.2%}
Peak Exposed: {result.results['peak_exposed']} (t={result.results['peak_exposed_time']})
Peak Infected: {result.results['peak_infected']} (t={result.results['peak_infected_time']})
Duration: {result.results['epidemic_duration']} steps
R₀ Estimate: {result.statistics['r0_estimate']:.2f}
Incubation Period: {result.statistics['incubation_period']:.1f} steps
Infectious Period: {result.statistics['infectious_period']:.1f} steps
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        # Plot 5: Cumulative incidence
        ax5 = plt.subplot(2, 3, 5)
        cumulative_cases = [time_series['exposed'][i] + time_series['infected'][i] + 
                           time_series['recovered'][i] for i in range(len(times))]
        ax5.plot(times, cumulative_cases, 'purple', linewidth=2, label='Cumulative Cases')
        ax5.plot(times, time_series['recovered'], 'g--', linewidth=2, label='Cumulative Recovered')
        ax5.set_xlabel('Time Steps')
        ax5.set_ylabel('Cumulative Count')
        ax5.set_title('Cumulative Incidence')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Effective reproduction number over time
        ax6 = plt.subplot(2, 3, 6)
        if len(times) > 5:
            # Simple estimate of effective R based on growth rate
            effective_r = []
            window = 5
            for i in range(window, len(times)):
                if time_series['infected'][i-window] > 0:
                    growth_rate = (time_series['infected'][i] / time_series['infected'][i-window]) ** (1/window) - 1
                    r_eff = 1 + growth_rate / self.recovery_rate
                    effective_r.append(max(0, min(5, r_eff)))  # Cap at reasonable values
                else:
                    effective_r.append(0)
            
            ax6.plot(times[window:], effective_r, 'darkred', linewidth=2)
            ax6.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='R=1')
            ax6.set_xlabel('Time Steps')
            ax6.set_ylabel('Effective R')
            ax6.set_title('Effective Reproduction Number')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_ylim(0, 3)
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'infection_rate': {
                'type': 'float',
                'default': 0.1,
                'min': 0.0,
                'max': 1.0,
                'description': 'Probability of infection per infected neighbor'
            },
            'incubation_rate': {
                'type': 'float',
                'default': 0.2,
                'min': 0.0,
                'max': 1.0,
                'description': 'Rate of becoming infectious (1/incubation period)'
            },
            'recovery_rate': {
                'type': 'float',
                'default': 0.1,
                'min': 0.0,
                'max': 1.0,
                'description': 'Probability of recovery per time step'
            },
            'initial_infected': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 50,
                'description': 'Number of initially infected nodes'
            },
            'max_time_steps': {
                'type': 'int',
                'default': 150,
                'min': 10,
                'max': 1000,
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
        if not (0 <= self.infection_rate <= 1):
            errors.append("infection_rate must be between 0 and 1")
        if not (0 <= self.incubation_rate <= 1):
            errors.append("incubation_rate must be between 0 and 1")
        if not (0 <= self.recovery_rate <= 1):
            errors.append("recovery_rate must be between 0 and 1")
        if self.initial_infected < 1:
            errors.append("initial_infected must be at least 1")
        if self.max_time_steps < 1:
            errors.append("max_time_steps must be positive")
        return errors
