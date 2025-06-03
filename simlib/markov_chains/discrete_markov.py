import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Union
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class DiscreteMarkovChain(BaseSimulation):
    """
    Discrete-time Markov Chain simulation with finite state space.
    
    A Markov chain is a stochastic process where the probability of transitioning to 
    the next state depends only on the current state, not on the sequence of events 
    that preceded it (Markov property). This implementation simulates discrete-time 
    chains with finite state spaces.
    
    Mathematical Background:
    -----------------------
    - States: S = {0, 1, 2, ..., n-1} (finite state space)
    - Transition matrix: P where P[i,j] = P(X_{t+1} = j | X_t = i)
    - Markov property: P(X_{t+1} = j | X_t = i, X_{t-1}, ..., X_0) = P(X_{t+1} = j | X_t = i)
    - Chapman-Kolmogorov equation: P^(n) = P^n (matrix exponentiation)
    - Stationary distribution π: πP = π, Σπ_i = 1
    - Detailed balance: π_i P_{i,j} = π_j P_{j,i} (for reversible chains)
    
    Key Properties:
    --------------
    - Irreducibility: All states communicate with each other
    - Aperiodicity: gcd of return times to any state is 1
    - Ergodicity: Irreducible + aperiodic → unique stationary distribution
    - Convergence: lim_{n→∞} P^n = ππᵀ (for ergodic chains)
    
    Applications:
    ------------
    - Random walks on graphs
    - PageRank algorithm
    - Speech recognition (HMMs)
    - Population genetics
    - Queueing theory
    - Finance: credit rating transitions
    - Weather modeling
    - Game theory strategies
    
    Simulation Features:
    -------------------
    - Custom transition matrices or predefined chain types
    - Multiple chain realizations for statistical analysis
    - Stationary distribution computation and verification
    - Convergence analysis to equilibrium
    - State occupation time analysis
    - Return time statistics
    - Mixing time estimation
    
    Parameters:
    -----------
    transition_matrix : numpy.ndarray or str, default='symmetric_random_walk'
        Transition probability matrix (must be stochastic) or predefined type:
        - 'symmetric_random_walk': Random walk on line with reflecting boundaries
        - 'birth_death': Birth-death process
        - 'random': Random stochastic matrix
    n_states : int, default=5
        Number of states (only used for predefined matrices)
    n_steps : int, default=1000
        Number of time steps to simulate
    n_chains : int, default=1
        Number of independent chain realizations
    initial_state : int or str, default='random'
        Starting state ('random' for uniform random choice)
    random_seed : int, optional
        Seed for random number generator
    
    Attributes:
    -----------
    P : numpy.ndarray
        Transition probability matrix
    chains : list of numpy.ndarray
        State sequences for each chain realization
    stationary_dist : numpy.ndarray
        Computed stationary distribution
    result : SimulationResult
        Complete simulation results and statistics
    
    Methods:
    --------
    configure(transition_matrix, n_states, n_steps, n_chains, initial_state) : bool
        Configure Markov chain parameters
    run(**kwargs) : SimulationResult
        Execute the Markov chain simulation
    visualize(result=None, show_convergence=True, show_transitions=True) : None
        Create comprehensive visualizations
    compute_stationary_distribution() : numpy.ndarray
        Compute theoretical stationary distribution
    validate_parameters() : List[str]
        Validate parameters and transition matrix
    get_parameter_info() : dict
        Get parameter information for UI generation
    
    Examples:
    ---------
    >>> # Simple 3-state chain
    >>> P = np.array([[0.7, 0.2, 0.1],
    ...               [0.3, 0.4, 0.3], 
    ...               [0.2, 0.3, 0.5]])
    >>> mc = DiscreteMarkovChain(transition_matrix=P, n_steps=1000)
    >>> result = mc.run()
    >>> mc.visualize()
    
    >>> # Random walk
    >>> mc_rw = DiscreteMarkovChain('symmetric_random_walk', n_states=10, n_steps=2000)
    >>> result = mc_rw.run()
    >>> print(f"Stationary distribution: {mc_rw.stationary_dist}")
    
    >>> # Multiple realizations
    >>> mc_multi = DiscreteMarkovChain('random', n_states=4, n_chains=50)
    >>> result = mc_multi.run()
    
    References:
    -----------
    - Norris, J. R. (1998). Markov Chains
    - Ross, S. M. (2014). Introduction to Probability Models
    - Levin, D. A., et al. (2017). Markov Chains and Mixing Times
    """

    def __init__(self, transition_matrix: Union[np.ndarray, str] = 'symmetric_random_walk',
                 n_states: int = 5, n_steps: int = 1000, n_chains: int = 1,
                 initial_state: Union[int, str] = 'random', random_seed: Optional[int] = None):
        super().__init__("Discrete Markov Chain")
        
        self.n_states = n_states
        self.n_steps = n_steps
        self.n_chains = n_chains
        self.initial_state = initial_state
        
        # Store parameters
        self.parameters.update({
            'transition_matrix': transition_matrix if isinstance(transition_matrix, str) else 'custom',
            'n_states': n_states,
            'n_steps': n_steps,
            'n_chains': n_chains,
            'initial_state': initial_state,
            'random_seed': random_seed
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Initialize transition matrix
        self.P = self._create_transition_matrix(transition_matrix)
        self.chains = None
        self.stationary_dist = None
        self.is_configured = True
    
    def _create_transition_matrix(self, matrix_type):
        """Create transition matrix based on type"""
        if isinstance(matrix_type, (np.ndarray, list)):
            return np.array(matrix_type)
        
        if matrix_type == 'symmetric_random_walk':
            # Random walk on line with reflecting boundaries
            P = np.zeros((self.n_states, self.n_states))
            for i in range(self.n_states):
                if i == 0:  # Left boundary
                    P[i, i] = 0.5
                    P[i, i + 1] = 0.5
                elif i == self.n_states - 1:  # Right boundary
                    P[i, i - 1] = 0.5
                    P[i, i] = 0.5
                else:  # Interior states
                    P[i, i - 1] = 0.5
                    P[i, i + 1] = 0.5
            return P
        
        elif matrix_type == 'birth_death':
            # Birth-death process
            P = np.zeros((self.n_states, self.n_states))
            for i in range(self.n_states):
                if i == 0:
                    P[i, i] = 0.3
                    P[i, i + 1] = 0.7
                elif i == self.n_states - 1:
                    P[i, i - 1] = 0.7
                    P[i, i] = 0.3
                else:
                    P[i, i - 1] = 0.35
                    P[i, i] = 0.3
                    P[i, i + 1] = 0.35
            return P
        
        elif matrix_type == 'random':
            # Random stochastic matrix
            P = np.random.rand(self.n_states, self.n_states)
            # Normalize rows to make stochastic
            P = P / P.sum(axis=1, keepdims=True)
            return P
        
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")
    
    def configure(self, transition_matrix: Union[np.ndarray, str] = 'symmetric_random_walk',
                 n_states: int = 5, n_steps: int = 1000, n_chains: int = 1,
                 initial_state: Union[int, str] = 'random') -> bool:
        """Configure Markov chain parameters"""
        self.n_states = n_states
        self.n_steps = n_steps
        self.n_chains = n_chains
        self.initial_state = initial_state
        
        self.parameters.update({
            'transition_matrix': transition_matrix if isinstance(transition_matrix, str) else 'custom',
            'n_states': n_states,
            'n_steps': n_steps,
            'n_chains': n_chains,
            'initial_state': initial_state
        })
        
        self.P = self._create_transition_matrix(transition_matrix)
        self.is_configured = True
        return True
    
    def compute_stationary_distribution(self) -> np.ndarray:
        """Compute stationary distribution using eigenvalue decomposition"""
        eigenvals, eigenvecs = np.linalg.eig(self.P.T)
        
        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvals - 1.0))
        stationary = np.real(eigenvecs[:, idx])
        
        # Normalize to probability distribution
        stationary = np.abs(stationary)
        stationary = stationary / np.sum(stationary)
        
        return stationary
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute Markov chain simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Compute stationary distribution
        self.stationary_dist = self.compute_stationary_distribution()
        
        chains = []
        state_counts = np.zeros((self.n_chains, self.n_states))
        
        for chain_idx in range(self.n_chains):
            # Initialize chain
            if self.initial_state == 'random':
                current_state = np.random.randint(0, self.n_states)
            else:
                current_state = self.initial_state
            
            chain = np.zeros(self.n_steps + 1, dtype=int)
            chain[0] = current_state
            
            # Simulate chain
            for t in range(self.n_steps):
                # Sample next state according to transition probabilities
                current_state = np.random.choice(self.n_states, p=self.P[current_state])
                chain[t + 1] = current_state
            
            chains.append(chain)
            
            # Count state visits
            unique, counts = np.unique(chain, return_counts=True)
            for state, count in zip(unique, counts):
                state_counts[chain_idx, state] = count
        
        self.chains = chains
        execution_time = time.time() - start_time
        
        # Calculate empirical distribution
        empirical_dist = np.mean(state_counts, axis=0) / (self.n_steps + 1)
        
        # Calculate convergence metrics
        final_states = [chain[-1] for chain in chains]
        unique_final, counts_final = np.unique(final_states, return_counts=True)
        final_dist = np.zeros(self.n_states)
        for state, count in zip(unique_final, counts_final):
            final_dist[state] = count / self.n_chains
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'final_states': final_states,
                'empirical_distribution': empirical_dist.tolist(),
                'final_state_distribution': final_dist.tolist(),
                'stationary_distribution': self.stationary_dist.tolist(),
                'state_counts': state_counts.tolist(),
                'transition_matrix': self.P.tolist()
            },
            statistics={
                'stationary_distribution': self.stationary_dist,
                'empirical_distribution': empirical_dist,
                'distribution_error': np.linalg.norm(empirical_dist - self.stationary_dist),
                'mixing_time_estimate': self._estimate_mixing_time()
            },
            raw_data=np.array(chains),
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def _estimate_mixing_time(self) -> float:
        """Estimate mixing time using total variation distance"""
        if self.n_chains == 1 and len(self.chains) > 0:
            chain = self.chains[0]
            tvd_distances = []
            
            # Calculate TV distance at different time points
            window_size = min(100, self.n_steps // 10)
            for t in range(window_size, self.n_steps, window_size):
                # Empirical distribution up to time t
                states_up_to_t = chain[:t+1]
                unique, counts = np.unique(states_up_to_t, return_counts=True)
                emp_dist = np.zeros(self.n_states)
                for state, count in zip(unique, counts):
                    emp_dist[state] = count / len(states_up_to_t)
                
                # Total variation distance
                tvd = 0.5 * np.sum(np.abs(emp_dist - self.stationary_dist))
                tvd_distances.append(tvd)
            
            # Find mixing time (when TVD drops below 1/e)
            threshold = 1.0 / np.e
            
            for i, tvd in enumerate(tvd_distances):
                if tvd < threshold:
                    return (i + 1) * window_size
            
            return self.n_steps  # If never mixed
        
        return float('inf')  # Cannot estimate for multiple chains
    
    def visualize(self, result: Optional[SimulationResult] = None, 
                 show_convergence: bool = True, show_transitions: bool = True) -> None:
        """Visualize Markov chain simulation results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        chains = result.raw_data
        
        if self.n_chains == 1:
            # Single chain visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: State sequence over time
            chain = chains[0]
            ax1.plot(range(len(chain)), chain, 'b-', linewidth=1, alpha=0.7)
            ax1.scatter(range(0, len(chain), len(chain)//20), 
                       chain[::len(chain)//20], c='red', s=30, zorder=5)
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('State')
            ax1.set_title('Markov Chain State Sequence')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-0.5, self.n_states - 0.5)
            
            # Plot 2: State occupation histogram
            unique, counts = np.unique(chain, return_counts=True)
            emp_dist = counts / len(chain)
            
            x = np.arange(self.n_states)
            width = 0.35
            ax2.bar(x - width/2, self.stationary_dist, width, 
                   label='Theoretical', alpha=0.7, color='blue')
            
            emp_full = np.zeros(self.n_states)
            for state, prob in zip(unique, emp_dist):
                emp_full[state] = prob
            ax2.bar(x + width/2, emp_full, width, 
                   label='Empirical', alpha=0.7, color='orange')
            
            ax2.set_xlabel('State')
            ax2.set_ylabel('Probability')
            ax2.set_title('State Distribution Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Convergence to stationary distribution
            if show_convergence:
                window_size = max(1, len(chain) // 100)
                times = []
                tv_distances = []
                
                for t in range(window_size, len(chain), window_size):
                    states_up_to_t = chain[:t]
                    unique_t, counts_t = np.unique(states_up_to_t, return_counts=True)
                    emp_dist_t = np.zeros(self.n_states)
                    for state, count in zip(unique_t, counts_t):
                        emp_dist_t[state] = count / len(states_up_to_t)
                    
                    tvd = 0.5 * np.sum(np.abs(emp_dist_t - self.stationary_dist))
                    times.append(t)
                    tv_distances.append(tvd)
                
                ax3.plot(times, tv_distances, 'g-', linewidth=2)
                ax3.axhline(y=1/np.e, color='red', linestyle='--', 
                           label='Mixing threshold (1/e)')
                ax3.set_xlabel('Time Step')
                ax3.set_ylabel('Total Variation Distance')
                ax3.set_title('Convergence to Stationary Distribution')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                ax3.set_yscale('log')
            
            # Plot 4: Transition matrix heatmap
            if show_transitions:
                im = ax4.imshow(self.P, cmap='Blues', aspect='auto')
                ax4.set_xlabel('To State')
                ax4.set_ylabel('From State')
                ax4.set_title('Transition Probability Matrix')
                
                # Add text annotations
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        text = ax4.text(j, i, f'{self.P[i, j]:.2f}',
                                       ha="center", va="center", color="black")
                
                plt.colorbar(im, ax=ax4)
        
        else:
            # Multiple chains visualization
            fig = plt.figure(figsize=(18, 12))
            
            ax1 = plt.subplot(2, 3, 1)
            ax2 = plt.subplot(2, 3, 2)
            ax3 = plt.subplot(2, 3, 3)
            ax4 = plt.subplot(2, 3, 4)
            ax5 = plt.subplot(2, 3, 5)
            ax6 = plt.subplot(2, 3, 6)
            
            # Plot 1: Multiple chain trajectories
            for i, chain in enumerate(chains[:min(10, self.n_chains)]):
                alpha = min(0.7, 10.0 / len(chains))
                ax1.plot(range(len(chain)), chain, linewidth=1, alpha=alpha)
            
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('State')
            ax1.set_title(f'First {min(10, self.n_chains)} Chain Trajectories')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Final state distribution
            final_states = result.results['final_states']
            unique, counts = np.unique(final_states, return_counts=True)
            
            ax2.hist(final_states, bins=np.arange(self.n_states + 1) - 0.5, 
                    alpha=0.7, density=True, edgecolor='black')
            ax2.plot(range(self.n_states), self.stationary_dist, 'ro-', 
                    linewidth=2, label='Stationary')
            ax2.set_xlabel('State')
            ax2.set_ylabel('Probability')
            ax2.set_title('Final State Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Empirical vs theoretical distribution
            emp_dist = result.results['empirical_distribution']
            x = np.arange(self.n_states)
            width = 0.35
            
            ax3.bar(x - width/2, self.stationary_dist, width, 
                   label='Theoretical', alpha=0.7, color='blue')
            ax3.bar(x + width/2, emp_dist, width, 
                   label='Empirical', alpha=0.7, color='orange')
            ax3.set_xlabel('State')
            ax3.set_ylabel('Probability')
            ax3.set_title('Distribution Comparison')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: State occupation over time (ensemble average)
            time_points = np.linspace(0, self.n_steps, 20, dtype=int)
            state_probs_over_time = np.zeros((len(time_points), self.n_states))
            
            for t_idx, t in enumerate(time_points):
                if t < len(chains[0]):
                    states_at_t = [chain[t] for chain in chains]
                    for state in range(self.n_states):
                        state_probs_over_time[t_idx, state] = np.mean(np.array(states_at_t) == state)
            
            for state in range(min(5, self.n_states)):  # Show first 5 states
                ax4.plot(time_points, state_probs_over_time[:, state], 
                        label=f'State {state}', linewidth=2)
            
            for state in range(self.n_states):
                ax4.axhline(y=self.stationary_dist[state], color='gray', 
                           linestyle='--', alpha=0.5)
            
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('State Probability')
            ax4.set_title('State Probabilities Over Time')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Transition matrix heatmap
            im = ax5.imshow(self.P, cmap='Blues', aspect='auto')
            ax5.set_xlabel('To State')
            ax5.set_ylabel('From State')
            ax5.set_title('Transition Probability Matrix')
            plt.colorbar(im, ax=ax5)
            
            # Plot 6: Error metrics
            errors = [np.linalg.norm(result.results['empirical_distribution'] - self.stationary_dist)]
            ax6.bar(['Distribution Error'], errors, alpha=0.7, color='red')
            ax6.set_ylabel('L2 Norm Error')
            ax6.set_title('Convergence Metrics')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'transition_matrix': {
                'type': 'choice',
                'default': 'symmetric_random_walk',
                'choices': ['symmetric_random_walk', 'birth_death', 'random', 'custom'],
                'description': 'Type of transition matrix'
            },
            'n_states': {
                'type': 'int',
                'default': 5,
                'min': 2,
                'max': 20,
                'description': 'Number of states'
            },
            'n_steps': {
                'type': 'int',
                'default': 1000,
                'min': 100,
                'max': 10000,
                'description': 'Number of time steps'
            },
            'n_chains': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 100,
                'description': 'Number of chain realizations'
            },
            'initial_state': {
                'type': 'choice',
                'default': 'random',
                'choices': ['random'] + [str(i) for i in range(10)],
                'description': 'Initial state'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility'
            }
        }
    
    def validate_parameters(self) -> List[str]:
        """Validate simulation parameters"""
        errors = []
        
        if self.n_states < 2:
            errors.append("n_states must be at least 2")
        if self.n_states > 50:
            errors.append("n_states should not exceed 50 for performance reasons")
        if self.n_steps < 1:
            errors.append("n_steps must be positive")
        if self.n_steps > 100000:
            errors.append("n_steps should not exceed 100,000 for performance reasons")
        if self.n_chains < 1:
            errors.append("n_chains must be positive")
        if self.n_chains > 1000:
            errors.append("n_chains should not exceed 1,000 for performance reasons")
        
        # Validate transition matrix
        if self.P is not None:
            if not np.allclose(self.P.sum(axis=1), 1.0):
                errors.append("Transition matrix rows must sum to 1")
            if np.any(self.P < 0):
                errors.append("Transition matrix must have non-negative entries")
            if self.P.shape[0] != self.P.shape[1]:
                errors.append("Transition matrix must be square")
        
        return errors
