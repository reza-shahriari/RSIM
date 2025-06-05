import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Union
from scipy.linalg import expm
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class ContinuousMarkovChain(BaseSimulation):
    """
    Continuous-time Markov Chain simulation with finite state space.
    
    A continuous-time Markov chain is a stochastic process that transitions between 
    states continuously in time, where the holding time in each state follows an 
    exponential distribution. The process is characterized by a rate matrix (Q-matrix) 
    that governs transition rates between states.
    
    Mathematical Background:
    -----------------------
    - States: S = {0, 1, 2, ..., n-1} (finite state space)
    - Rate matrix: Q where Q[i,j] = rate of transition from i to j (i≠j)
    - Diagonal elements: Q[i,i] = -Σ_{j≠i} Q[i,j] (exit rate from state i)
    - Holding times: Exponentially distributed with rate λ_i = -Q[i,i]
    - Transition probabilities: P(X(t+s) = j | X(t) = i) = [e^{Qt}]_{i,j}
    - Generator property: P'(0) = Q
    - Stationary distribution: πQ = 0, Σπ_i = 1
    
    Simulation Algorithm:
    --------------------
    1. Start in initial state
    2. Sample holding time from Exponential(-Q[i,i])
    3. At jump time, sample next state proportional to Q[i,j]/(-Q[i,i])
    4. Repeat until total time T is reached
    
    Applications:
    ------------
    - Queueing systems (birth-death processes)
    - Chemical reaction networks
    - Population dynamics
    - Reliability engineering
    - Epidemiological models
    - Gene expression models
    - Communication networks
    - Financial credit models
    
    Simulation Features:
    -------------------
    - Custom rate matrices or predefined process types
    - Multiple sample paths for statistical analysis
    - Jump time analysis and holding time distributions
    - Stationary distribution computation
    - Uniformization for numerical stability
    - Embedded discrete chain analysis
    
    Parameters:
    -----------
    rate_matrix : numpy.ndarray or str, default='birth_death'
            rate_matrix : numpy.ndarray or str, default='birth_death'
        Rate matrix Q or predefined type:
        - 'birth_death': Birth-death process with configurable rates
        - 'immigration_death': Immigration-death process
        - 'random': Random rate matrix
    n_states : int, default=5
        Number of states (only used for predefined matrices)
    total_time : float, default=10.0
        Total simulation time
    n_paths : int, default=1
        Number of independent sample paths
    initial_state : int or str, default='random'
        Starting state ('random' for uniform random choice)
    birth_rate : float, default=1.0
        Birth rate for birth-death processes
    death_rate : float, default=1.0
        Death rate for birth-death processes
    random_seed : int, optional
        Seed for random number generator
    
    Attributes:
    -----------
    Q : numpy.ndarray
        Rate matrix (generator matrix)
    paths : list of tuples
        Each path as (times, states) where times are jump times
    stationary_dist : numpy.ndarray
        Computed stationary distribution
    result : SimulationResult
        Complete simulation results and statistics
    
    Methods:
    --------
    configure(rate_matrix, n_states, total_time, n_paths, initial_state) : bool
        Configure continuous Markov chain parameters
    run(**kwargs) : SimulationResult
        Execute the continuous Markov chain simulation
    visualize(result=None, show_sample_paths=True, show_holding_times=True) : None
        Create comprehensive visualizations
    compute_stationary_distribution() : numpy.ndarray
        Compute theoretical stationary distribution
    validate_parameters() : List[str]
        Validate parameters and rate matrix
    get_parameter_info() : dict
        Get parameter information for UI generation
    
    Examples:
    ---------
    >>> # Birth-death process
    >>> cmc = ContinuousMarkovChain('birth_death', n_states=5, total_time=20.0)
    >>> result = cmc.run()
    >>> cmc.visualize()
    
    >>> # Custom rate matrix
    >>> Q = np.array([[-1.0, 0.5, 0.5],
    ...               [0.3, -0.8, 0.5],
    ...               [0.2, 0.6, -0.8]])
    >>> cmc_custom = ContinuousMarkovChain(Q, total_time=15.0, n_paths=10)
    >>> result = cmc_custom.run()
    
    References:
    -----------
    - Anderson, W. J. (2012). Continuous-Time Markov Chains
    - Norris, J. R. (1998). Markov Chains
    - Ross, S. M. (2014). Introduction to Probability Models
    """

    def __init__(self, rate_matrix: Union[np.ndarray, str] = 'birth_death',
                 n_states: int = 5, total_time: float = 10.0, n_paths: int = 1,
                 initial_state: Union[int, str] = 'random', birth_rate: float = 1.0,
                 death_rate: float = 1.0, random_seed: Optional[int] = None):
        super().__init__("Continuous Markov Chain")
        
        self.n_states = n_states
        self.total_time = total_time
        self.n_paths = n_paths
        self.initial_state = initial_state
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        
        # Store parameters
        self.parameters.update({
            'rate_matrix': rate_matrix if isinstance(rate_matrix, str) else 'custom',
            'n_states': n_states,
            'total_time': total_time,
            'n_paths': n_paths,
            'initial_state': initial_state,
            'birth_rate': birth_rate,
            'death_rate': death_rate,
            'random_seed': random_seed
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Initialize rate matrix
        self.Q = self._create_rate_matrix(rate_matrix)
        self.paths = None
        self.stationary_dist = None
        self.is_configured = True
    
    def _create_rate_matrix(self, matrix_type):
        """Create rate matrix based on type"""
        if isinstance(matrix_type, (np.ndarray, list)):
            return np.array(matrix_type)
        
        if matrix_type == 'birth_death':
            # Birth-death process
            Q = np.zeros((self.n_states, self.n_states))
            for i in range(self.n_states):
                if i > 0:  # Death transition
                    Q[i, i-1] = i * self.death_rate
                if i < self.n_states - 1:  # Birth transition
                    Q[i, i+1] = self.birth_rate
                
                # Diagonal element (negative exit rate)
                Q[i, i] = -(Q[i, :].sum() - Q[i, i])
            return Q
        
        elif matrix_type == 'immigration_death':
            # Immigration-death process
            Q = np.zeros((self.n_states, self.n_states))
            for i in range(self.n_states):
                if i > 0:  # Death transition
                    Q[i, i-1] = i * self.death_rate
                if i < self.n_states - 1:  # Immigration transition
                    Q[i, i+1] = self.birth_rate  # Constant immigration
                
                Q[i, i] = -(Q[i, :].sum() - Q[i, i])
            return Q
        
        elif matrix_type == 'random':
            # Random rate matrix
            Q = np.random.exponential(1.0, (self.n_states, self.n_states))
            np.fill_diagonal(Q, 0)  # Clear diagonal
            
            # Set diagonal elements
            for i in range(self.n_states):
                Q[i, i] = -Q[i, :].sum()
            return Q
        
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")
    
    def configure(self, rate_matrix: Union[np.ndarray, str] = 'birth_death',
                 n_states: int = 5, total_time: float = 10.0, n_paths: int = 1,
                 initial_state: Union[int, str] = 'random', birth_rate: float = 1.0,
                 death_rate: float = 1.0) -> bool:
        """Configure continuous Markov chain parameters"""
        self.n_states = n_states
        self.total_time = total_time
        self.n_paths = n_paths
        self.initial_state = initial_state
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        
        self.parameters.update({
            'rate_matrix': rate_matrix if isinstance(rate_matrix, str) else 'custom',
            'n_states': n_states,
            'total_time': total_time,
            'n_paths': n_paths,
            'initial_state': initial_state,
            'birth_rate': birth_rate,
            'death_rate': death_rate
        })
        
        self.Q = self._create_rate_matrix(rate_matrix)
        self.is_configured = True
        return True
    
    def compute_stationary_distribution(self) -> np.ndarray:
        """Compute stationary distribution by solving πQ = 0"""
        # Create augmented matrix for πQ = 0 with Σπ_i = 1
        A = np.vstack([self.Q.T, np.ones(self.n_states)])
        b = np.zeros(self.n_states + 1)
        b[-1] = 1.0
        
        # Solve using least squares
        pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Ensure non-negative and normalized
        pi = np.abs(pi)
        pi = pi / np.sum(pi)
        
        return pi
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute continuous Markov chain simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Compute stationary distribution
        self.stationary_dist = self.compute_stationary_distribution()
        
        paths = []
        holding_times_all = []
        jump_counts = []
        final_states = []
        
        for path_idx in range(self.n_paths):
            # Initialize path
            if self.initial_state == 'random':
                current_state = np.random.randint(0, self.n_states)
            else:
                current_state = self.initial_state
            
            times = [0.0]
            states = [current_state]
            current_time = 0.0
            holding_times_path = []
            
            # Simulate until total_time
            while current_time < self.total_time:
                # Sample holding time
                exit_rate = -self.Q[current_state, current_state]
                if exit_rate > 0:
                    holding_time = np.random.exponential(1.0 / exit_rate)
                else:
                    holding_time = float('inf')  # Absorbing state
                
                current_time += holding_time
                
                if current_time >= self.total_time:
                    # Add final point at total_time
                    times.append(self.total_time)
                    states.append(current_state)
                    holding_times_path.append(self.total_time - times[-2])
                    break
                
                holding_times_path.append(holding_time)
                
                # Sample next state
                if exit_rate > 0:
                    transition_rates = self.Q[current_state, :].copy()
                    transition_rates[current_state] = 0  # Remove self-transition
                    transition_probs = transition_rates / exit_rate
                    
                    # Handle numerical errors
                    transition_probs = np.maximum(transition_probs, 0)
                    if transition_probs.sum() > 0:
                        current_state = np.random.choice(self.n_states, p=transition_probs)
                    
                    times.append(current_time)
                    states.append(current_state)
                else:
                    # Absorbing state - stay until end
                    times.append(self.total_time)
                    states.append(current_state)
                    holding_times_path.append(self.total_time - current_time)
                    break
            
            paths.append((np.array(times), np.array(states)))
            holding_times_all.extend(holding_times_path)
            jump_counts.append(len(times) - 1)
            final_states.append(states[-1])
        
        self.paths = paths
        execution_time = time.time() - start_time
        
        # Calculate time-averaged state occupancies
        occupancy_times = np.zeros((self.n_paths, self.n_states))
        for path_idx, (times, states) in enumerate(paths):
            for i in range(len(times) - 1):
                state = states[i]
                duration = times[i + 1] - times[i]
                occupancy_times[path_idx, state] += duration
        
        # Normalize by total time
        occupancy_probs = occupancy_times / self.total_time
        empirical_dist = np.mean(occupancy_probs, axis=0)
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'final_states': final_states,
                'empirical_distribution': empirical_dist.tolist(),
                'stationary_distribution': self.stationary_dist.tolist(),
                'holding_times': holding_times_all,
                'jump_counts': jump_counts,
                'mean_holding_time': np.mean(holding_times_all) if holding_times_all else 0,
                'mean_jump_count': np.mean(jump_counts),
                'rate_matrix': self.Q.tolist()
            },
            statistics={
                'stationary_distribution': self.stationary_dist,
                'empirical_distribution': empirical_dist,
                'distribution_error': np.linalg.norm(empirical_dist - self.stationary_dist),
                'total_jumps': sum(jump_counts),
                'average_rate': sum(jump_counts) / (self.n_paths * self.total_time)
            },
            raw_data={'paths': paths},
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, 
                 show_sample_paths: bool = True, show_holding_times: bool = True) -> None:
        """Visualize continuous Markov chain simulation results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        paths = result.raw_data['paths']
        
        if self.n_paths == 1:
            # Single path visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Sample path
            times, states = paths[0]
            
            # Create step function
            for i in range(len(times) - 1):
                ax1.plot([times[i], times[i+1]], [states[i], states[i]], 
                        'b-', linewidth=2)
                if i < len(times) - 2:  # Don't draw vertical line at the end
                    ax1.plot([times[i+1], times[i+1]], [states[i], states[i+1]], 
                            'b-', linewidth=2)
            
            # Mark jump times
            jump_times = times[1:-1]  # Exclude start and end
            jump_states = states[1:-1]
            ax1.scatter(jump_times, jump_states, c='red', s=50, zorder=5, 
                       label='Jump times')
            
            ax1.set_xlabel('Time')
            ax1.set_ylabel('State')
            ax1.set_title('Sample Path')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_ylim(-0.5, self.n_states - 0.5)
            
            # Plot 2: Holding time distribution
            if show_holding_times and result.results['holding_times']:
                holding_times = result.results['holding_times']
                ax2.hist(holding_times, bins=20, alpha=0.7, density=True, 
                        edgecolor='black')
                
                # Overlay exponential fits for each state
                                # Overlay exponential fits for each state
                colors = plt.cm.tab10(np.linspace(0, 1, self.n_states))
                x_exp = np.linspace(0, max(holding_times), 100)
                for state in range(self.n_states):
                    rate = -self.Q[state, state]
                    if rate > 0:
                        y_exp = rate * np.exp(-rate * x_exp)
                        ax2.plot(x_exp, y_exp, '--', color=colors[state], 
                                label=f'State {state} (λ={rate:.2f})')
                
                ax2.set_xlabel('Holding Time')
                ax2.set_ylabel('Density')
                ax2.set_title('Holding Time Distribution')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: State occupancy over time
            window_size = max(1, int(self.total_time / 50))
            time_points = np.arange(0, self.total_time, window_size)
            occupancy_over_time = np.zeros((len(time_points), self.n_states))
            
            for t_idx, t in enumerate(time_points):
                # Find state at time t
                state_at_t = states[0]  # Default to initial state
                for i in range(len(times) - 1):
                    if times[i] <= t < times[i + 1]:
                        state_at_t = states[i]
                        break
                    elif t >= times[-1]:
                        state_at_t = states[-1]
                
                occupancy_over_time[t_idx, state_at_t] = 1
            
            for state in range(self.n_states):
                ax3.plot(time_points, occupancy_over_time[:, state], 
                        label=f'State {state}', linewidth=2)
            
            # Add stationary distribution lines
            for state in range(self.n_states):
                ax3.axhline(y=self.stationary_dist[state], color='gray', 
                           linestyle='--', alpha=0.5)
            
            ax3.set_xlabel('Time')
            ax3.set_ylabel('State Indicator')
            ax3.set_title('State Occupancy Over Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Rate matrix heatmap
            im = ax4.imshow(self.Q, cmap='RdBu', aspect='auto')
            ax4.set_xlabel('To State')
            ax4.set_ylabel('From State')
            ax4.set_title('Rate Matrix Q')
            
            # Add text annotations
            for i in range(self.n_states):
                for j in range(self.n_states):
                    color = 'white' if abs(self.Q[i, j]) > np.max(np.abs(self.Q)) * 0.5 else 'black'
                    text = ax4.text(j, i, f'{self.Q[i, j]:.2f}',
                                   ha="center", va="center", color=color)
            
            plt.colorbar(im, ax=ax4)
        
        else:
            # Multiple paths visualization
            fig = plt.figure(figsize=(18, 12))
            
            ax1 = plt.subplot(2, 3, 1)
            ax2 = plt.subplot(2, 3, 2)
            ax3 = plt.subplot(2, 3, 3)
            ax4 = plt.subplot(2, 3, 4)
            ax5 = plt.subplot(2, 3, 5)
            ax6 = plt.subplot(2, 3, 6)
            
            # Plot 1: Multiple sample paths
            for i, (times, states) in enumerate(paths[:min(10, self.n_paths)]):
                alpha = min(0.7, 10.0 / len(paths))
                
                # Create step function
                for j in range(len(times) - 1):
                    ax1.plot([times[j], times[j+1]], [states[j], states[j]], 
                            linewidth=1, alpha=alpha)
                    if j < len(times) - 2:
                        ax1.plot([times[j+1], times[j+1]], [states[j], states[j+1]], 
                                linewidth=1, alpha=alpha)
            
            ax1.set_xlabel('Time')
            ax1.set_ylabel('State')
            ax1.set_title(f'Sample Paths (showing first {min(10, self.n_paths)})')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Final state distribution
            final_states = result.results['final_states']
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
            
            # Plot 4: Holding time distribution
            if result.results['holding_times']:
                holding_times = result.results['holding_times']
                ax4.hist(holding_times, bins=30, alpha=0.7, density=True, 
                        edgecolor='black')
                ax4.set_xlabel('Holding Time')
                ax4.set_ylabel('Density')
                ax4.set_title('Aggregate Holding Time Distribution')
                ax4.grid(True, alpha=0.3)
            
            # Plot 5: Jump count distribution
            jump_counts = result.results['jump_counts']
            ax5.hist(jump_counts, bins=20, alpha=0.7, edgecolor='black')
            ax5.axvline(x=np.mean(jump_counts), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(jump_counts):.1f}')
            ax5.set_xlabel('Number of Jumps')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Jump Count Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Plot 6: Rate matrix heatmap
            im = ax6.imshow(self.Q, cmap='RdBu', aspect='auto')
            ax6.set_xlabel('To State')
            ax6.set_ylabel('From State')
            ax6.set_title('Rate Matrix Q')
            plt.colorbar(im, ax=ax6)
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'rate_matrix': {
                'type': 'choice',
                'default': 'birth_death',
                'choices': ['birth_death', 'immigration_death', 'random', 'custom'],
                'description': 'Type of rate matrix'
            },
            'n_states': {
                'type': 'int',
                'default': 5,
                'min': 2,
                'max': 15,
                'description': 'Number of states'
            },
            'total_time': {
                'type': 'float',
                'default': 10.0,
                'min': 1.0,
                'max': 100.0,
                'description': 'Total simulation time'
            },
            'n_paths': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 100,
                'description': 'Number of sample paths'
            },
            'birth_rate': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 10.0,
                'description': 'Birth/immigration rate'
            },
            'death_rate': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 10.0,
                'description': 'Death rate'
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
        if self.total_time <= 0:
            errors.append("total_time must be positive")
        if self.total_time > 1000:
            errors.append("total_time should not exceed 1000 for performance reasons")
        if self.n_paths < 1:
            errors.append("n_paths must be positive")
        if self.n_paths > 1000:
            errors.append("n_paths should not exceed 1000 for performance reasons")
        if self.birth_rate <= 0:
            errors.append("birth_rate must be positive")
        if self.death_rate <= 0:
            errors.append("death_rate must be positive")
        
        # Validate rate matrix
        if self.Q is not None:
            # Check diagonal elements are negative
            if not np.all(np.diag(self.Q) <= 0):
                errors.append("Rate matrix diagonal elements must be non-positive")
            
            # Check off-diagonal elements are non-negative
            Q_off_diag = self.Q.copy()
            np.fill_diagonal(Q_off_diag, 0)
            if np.any(Q_off_diag < 0):
                errors.append("Rate matrix off-diagonal elements must be non-negative")
            
            # Check row sums are approximately zero
            row_sums = self.Q.sum(axis=1)
            if not np.allclose(row_sums, 0, atol=1e-10):
                errors.append("Rate matrix rows must sum to zero")
        
        return errors

