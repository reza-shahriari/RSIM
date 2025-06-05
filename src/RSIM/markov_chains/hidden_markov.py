import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class HiddenMarkovModel(BaseSimulation):
    """
    Hidden Markov Model simulation and parameter estimation.
    
    A Hidden Markov Model (HMM) is a statistical model where the system being 
    modeled is assumed to be a Markov process with unobservable (hidden) states. 
    The model consists of hidden states that follow a Markov chain and observable 
    emissions that depend on the current hidden state.
    
    Mathematical Framework:
    ----------------------
    - Hidden states: S = {s₁, s₂, ..., sₙ}
    - Observable symbols: O = {o₁, o₂, ..., oₘ}
    - Transition probabilities: A[i,j] = P(sⱼ at t+1 | sᵢ at t)
    - Emission probabilities: B[i,k] = P(oₖ | sᵢ)
    - Initial state distribution: π[i] = P(s₀ = sᵢ)
    - Complete model: λ = (A, B, π)
    
    Core Algorithms:
    ---------------
    1. Forward Algorithm: Calculate P(O₁:ₜ, Sₜ = sᵢ | λ)
    2. Backward Algorithm: Calculate P(Oₜ₊₁:ₜ | Sₜ = sᵢ, λ)
    3. Viterbi Algorithm: Find most likely state sequence
    4. Baum-Welch Algorithm: EM parameter estimation
    
    Applications:
    ------------
    - Speech recognition and natural language processing
    - Bioinformatics (gene finding, protein structure)
    - Financial modeling (regime changes)
    - Weather prediction
    - Machine learning and pattern recognition
    - Signal processing and time series analysis
    
    Simulation Features:
    -------------------
    - Multiple predefined model types (weather, DNA, financial)
    - Custom model specification
    - Sequence generation from HMM
    - Parameter estimation using Baum-Welch
    - Most likely path inference using Viterbi
    - Model evaluation and comparison
    
    Parameters:
    -----------
    model_type : str, default='weather'
        Predefined model type:
        - 'weather': Simple weather model (Sunny/Rainy with observations)
        - 'dna': DNA sequence model (AT/GC content)
        - 'financial': Financial regime model (Bull/Bear markets)
        - 'custom': User-defined model
    n_states : int, default=2
        Number of hidden states (for custom models)
    n_observations : int, default=3
        Number of observable symbols (for custom models)
    sequence_length : int, default=100
        Length of sequences to generate
    n_sequences : int, default=1
        Number of sequences to generate
    random_seed : int, optional
        Seed for reproducible results
    
    Attributes:
    -----------
    A : numpy.ndarray
        Transition probability matrix (n_states × n_states)
    B : numpy.ndarray
        Emission probability matrix (n_states × n_observations)
    pi : numpy.ndarray
        Initial state distribution (n_states,)
    sequences : list
        Generated observation sequences
    hidden_states : list
        True hidden state sequences (if generating data)
    result : SimulationResult
        Complete simulation results
    
    Methods:
    --------
    configure(model_type, n_states, n_observations, sequence_length, n_sequences) : bool
        Configure HMM parameters
    run(**kwargs) : SimulationResult
        Generate sequences and/or estimate parameters
    visualize(result=None, show_viterbi=True, show_probabilities=True) : None
        Create comprehensive visualizations
    forward_algorithm(sequence) : numpy.ndarray
        Compute forward probabilities
    backward_algorithm(sequence) : numpy.ndarray
        Compute backward probabilities
    viterbi_algorithm(sequence) : tuple
        Find most likely state path
    baum_welch(sequences, max_iter=100) : tuple
        Estimate parameters using EM algorithm
    
    Examples:
    ---------
    >>> # Generate weather sequences
    >>> hmm = HiddenMarkovModel('weather', sequence_length=50)
    >>> result = hmm.run()
    >>> hmm.visualize()
    
    >>> # Parameter estimation
    >>> hmm_learn = HiddenMarkovModel('weather', n_sequences=10)
    >>> result = hmm_learn.run()
    >>> # Access estimated parameters in result.results
    
    >>> # Custom model
    >>> hmm_custom = HiddenMarkovModel('custom', n_states=3, n_observations=4)
    >>> result = hmm_custom.run()
    
    References:
    -----------
    - Rabiner, L. R. (1989). A tutorial on hidden Markov models...
    - Baum, L. E., et al. (1970). A maximization technique...
    - Viterbi, A. (1967). Error bounds for convolutional codes...
    - Bishop, C. M. (2006). Pattern Recognition and Machine Learning
    """

    def __init__(self, model_type: str = 'weather', n_states: int = 2, 
                  n_observations: int = 3, sequence_length: int = 100,
                  n_sequences: int = 1, random_seed: Optional[int] = None):
        super().__init__("Hidden Markov Model")
        
        self.model_type = model_type
        self.n_states = n_states
        self.n_observations = n_observations
        self.sequence_length = sequence_length
        self.n_sequences = n_sequences
        
        # Store parameters
        self.parameters.update({
            'model_type': model_type,
            'n_states': n_states,
            'n_observations': n_observations,
            'sequence_length': sequence_length,
            'n_sequences': n_sequences,
            'random_seed': random_seed
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Initialize model parameters
        self.A = None  # Transition matrix
        self.B = None  # Emission matrix
        self.pi = None  # Initial distribution
        
        self._setup_model()
        
        self.sequences = None
        self.hidden_states = None
        self.is_configured = True
    
    def _setup_model(self):
        """Initialize model parameters based on model type"""
        if self.model_type == 'weather':
            # Simple weather model: Sunny (0) vs Rainy (1)
            # Observations: Clear (0), Cloudy (1), Precipitation (2)
            self.n_states = 2
            self.n_observations = 3
            
            # Transition probabilities
            self.A = np.array([
                [0.7, 0.3],  # Sunny -> Sunny, Sunny -> Rainy
                [0.4, 0.6]   # Rainy -> Sunny, Rainy -> Rainy
            ])
            
            # Emission probabilities
            self.B = np.array([
                [0.6, 0.3, 0.1],  # Sunny -> Clear, Cloudy, Precipitation
                [0.1, 0.4, 0.5]   # Rainy -> Clear, Cloudy, Precipitation
            ])
            
            # Initial distribution
            self.pi = np.array([0.6, 0.4])
        
        elif self.model_type == 'dna':
            # DNA model: AT-rich (0) vs GC-rich (1) regions
            # Observations: A (0), T (1), G (2), C (3)
            self.n_states = 2
            self.n_observations = 4
            
            self.A = np.array([
                [0.8, 0.2],  # AT-rich region persistence
                [0.2, 0.8]   # GC-rich region persistence
            ])
            
            self.B = np.array([
                [0.3, 0.3, 0.2, 0.2],  # AT-rich: higher A,T probability
                [0.2, 0.2, 0.3, 0.3]   # GC-rich: higher G,C probability
            ])
            
            self.pi = np.array([0.5, 0.5])
        
        elif self.model_type == 'financial':
            # Financial regime model: Bull (0) vs Bear (1) markets
            # Observations: Up (0), Flat (1), Down (2)
            self.n_states = 2
            self.n_observations = 3
            
            self.A = np.array([
                [0.9, 0.1],  # Bull market persistence
                [0.1, 0.9]   # Bear market persistence
            ])
            
            self.B = np.array([
                [0.6, 0.3, 0.1],  # Bull: mostly up movements
                [0.1, 0.3, 0.6]   # Bear: mostly down movements
            ])
            
            self.pi = np.array([0.7, 0.3])
        
        elif self.model_type == 'custom':
            # Random initialization for custom model
            self.A = np.random.rand(self.n_states, self.n_states)
            self.A = self.A / self.A.sum(axis=1, keepdims=True)
            
            self.B = np.random.rand(self.n_states, self.n_observations)
            self.B = self.B / self.B.sum(axis=1, keepdims=True)
            
            self.pi = np.random.rand(self.n_states)
            self.pi = self.pi / self.pi.sum()
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def configure(self, model_type: str = 'weather', n_states: int = 2,
                  n_observations: int = 3, sequence_length: int = 100,
                  n_sequences: int = 1) -> bool:
        """Configure HMM parameters"""
        self.model_type = model_type
        self.n_states = n_states
        self.n_observations = n_observations
        self.sequence_length = sequence_length
        self.n_sequences = n_sequences
        
        self.parameters.update({
            'model_type': model_type,
            'n_states': n_states,
            'n_observations': n_observations,
            'sequence_length': sequence_length,
            'n_sequences': n_sequences
        })
        
        self._setup_model()
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute HMM simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Generate sequences
        sequences = []
        hidden_states = []
        
        for _ in range(self.n_sequences):
            obs_seq, state_seq = self._generate_sequence()
            sequences.append(obs_seq)
            hidden_states.append(state_seq)
        
        self.sequences = sequences
        self.hidden_states = hidden_states
        
        # Analyze sequences
        results = {}
        
        # For each sequence, run Viterbi algorithm
        viterbi_paths = []
        viterbi_probs = []
        
        for seq in sequences:
            path, prob = self.viterbi_algorithm(seq)
            viterbi_paths.append(path)
            viterbi_probs.append(prob)
        
        # Calculate accuracy of Viterbi path vs true path
        accuracies = []
        for true_path, viterbi_path in zip(hidden_states, viterbi_paths):
            accuracy = np.mean(np.array(true_path) == np.array(viterbi_path))
            accuracies.append(accuracy)
        
        # If multiple sequences, attempt parameter estimation
        if self.n_sequences > 1:
            estimated_A, estimated_B, estimated_pi, log_likelihood = self.baum_welch(sequences)
            results.update({
                'estimated_A': estimated_A.tolist(),
                'estimated_B': estimated_B.tolist(),
                'estimated_pi': estimated_pi.tolist(),
                'log_likelihood': log_likelihood
            })
        
        execution_time = time.time() - start_time
        
        results.update({
            'sequences': [seq.tolist() for seq in sequences],
            'hidden_states': [states.tolist() for states in hidden_states],
            'viterbi_paths': viterbi_paths,  # Remove .tolist() since these are already lists
            'viterbi_probabilities': viterbi_probs,
            'viterbi_accuracies': accuracies,
            'mean_viterbi_accuracy': np.mean(accuracies)
        })
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results=results,
            statistics={
                'true_A': self.A,
                'true_B': self.B,
                'true_pi': self.pi,
                'mean_accuracy': np.mean(accuracies)
            },
            raw_data={
                'A': self.A, 'B': self.B, 'pi': self.pi,
                'sequences': sequences, 'hidden_states': hidden_states,
                'viterbi_paths': viterbi_paths
            },
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def _generate_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single observation and hidden state sequence"""
        observations = np.zeros(self.sequence_length, dtype=int)
        states = np.zeros(self.sequence_length, dtype=int)
        
        # Initial state
        states[0] = np.random.choice(self.n_states, p=self.pi)
        observations[0] = np.random.choice(self.n_observations, p=self.B[states[0]])
        
        # Generate remaining sequence
        for t in range(1, self.sequence_length):
            # Sample next state based on current state
            states[t] = np.random.choice(self.n_states, p=self.A[states[t-1]])
            
            # Sample observation based on current state
            observations[t] = np.random.choice(self.n_observations, p=self.B[states[t]])
        
        return observations, states
    
    def forward_algorithm(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute forward probabilities using the forward algorithm.
        
        Returns:
            alpha: Forward probabilities α[t,i] = P(O₁:ₜ, Sₜ=i | λ)
        """
        T = len(sequence)
        alpha = np.zeros((T, self.n_states))
        
        # Initialization
        alpha[0] = self.pi * self.B[:, sequence[0]]
        
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, sequence[t]]
        
        return alpha
    
    def backward_algorithm(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute backward probabilities using the backward algorithm.
        
        Returns:
            beta: Backward probabilities β[t,i] = P(Oₜ₊₁:ₜ | Sₜ=i, λ)
        """
        T = len(sequence)
        beta = np.zeros((T, self.n_states))
        
        # Initialization (β[T-1,i] = 1 for all i)
        beta[T-1] = 1.0
        
        # Backward pass
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i] * self.B[:, sequence[t+1]] * beta[t+1])
        
        return beta
    
    def viterbi_algorithm(self, sequence: np.ndarray) -> Tuple[List[int], float]:
        """
        Find the most likely state sequence using the Viterbi algorithm.
        
        Returns:
            path: Most likely state sequence
            prob: Log probability of the path
        """
        T = len(sequence)
        
        # Initialize Viterbi trellis
        viterbi = np.zeros((T, self.n_states))
        path_indices = np.zeros((T, self.n_states), dtype=int)
        
        # Initialization
        viterbi[0] = np.log(self.pi) + np.log(self.B[:, sequence[0]])
        
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                # Find the most likely previous state
                transitions = viterbi[t-1] + np.log(self.A[:, j])
                path_indices[t, j] = np.argmax(transitions)
                viterbi[t, j] = np.max(transitions) + np.log(self.B[j, sequence[t]])
        
        # Backtrack to find the best path
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(viterbi[T-1])
        
        for t in range(T-2, -1, -1):
            path[t] = path_indices[t+1, path[t+1]]
        
        # Calculate path probability
        path_prob = np.max(viterbi[T-1])
        
        return path.tolist(), path_prob
    
    def baum_welch(self, sequences: List[np.ndarray], max_iter: int = 100, 
                    tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Estimate HMM parameters using the Baum-Welch (EM) algorithm.
        
        Args:
            sequences: List of observation sequences
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            A_new: Estimated transition matrix
            B_new: Estimated emission matrix
            pi_new: Estimated initial distribution
            log_likelihood: Final log-likelihood
        """
        # Initialize with current parameters
        A_new = self.A.copy()
        B_new = self.B.copy()
        pi_new = self.pi.copy()
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iter):
            # E-step: Compute forward-backward probabilities
            gamma_sum = np.zeros(self.n_states)
            xi_sum = np.zeros((self.n_states, self.n_states))
            gamma_obs_sum = np.zeros((self.n_states, self.n_observations))
            log_likelihood = 0.0
            
            for sequence in sequences:
                T = len(sequence)
                
                # Forward and backward algorithms
                alpha = self.forward_algorithm(sequence)
                beta = self.backward_algorithm(sequence)
                
                # Calculate sequence likelihood
                seq_likelihood = np.sum(alpha[T-1])
                log_likelihood += np.log(seq_likelihood + 1e-10)
                
                # Compute gamma (state probabilities)
                gamma = (alpha * beta) / (seq_likelihood + 1e-10)
                
                # Compute xi (transition probabilities)
                xi = np.zeros((T-1, self.n_states, self.n_states))
                for t in range(T-1):
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            xi[t, i, j] = (alpha[t, i] * A_new[i, j] * 
                                          B_new[j, sequence[t+1]] * beta[t+1, j])
                    xi[t] /= (np.sum(xi[t]) + 1e-10)
                
                # Accumulate sufficient statistics
                gamma_sum += np.sum(gamma, axis=0)
                xi_sum += np.sum(xi, axis=0)
                
                for k in range(self.n_observations):
                    mask = (sequence == k)
                    gamma_obs_sum[:, k] += np.sum(gamma[mask], axis=0)
            
            # M-step: Update parameters
            # Update initial distribution
            pi_new = gamma_sum / len(sequences)
            pi_new /= np.sum(pi_new)
            
            # Update transition matrix
            A_new = xi_sum / (gamma_sum[:, np.newaxis] + 1e-10)
            A_new /= np.sum(A_new, axis=1, keepdims=True)
            
            # Update emission matrix
            B_new = gamma_obs_sum / (gamma_sum[:, np.newaxis] + 1e-10)
            B_new /= np.sum(B_new, axis=1, keepdims=True)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < tolerance:
                break
            
            prev_log_likelihood = log_likelihood
            
            # Update model parameters for next iteration
            self.A = A_new.copy()
            self.B = B_new.copy()
            self.pi = pi_new.copy()
        
        return A_new, B_new, pi_new, log_likelihood
    
    def visualize(self, result: Optional[SimulationResult] = None, 
                  show_viterbi: bool = True, show_probabilities: bool = True) -> None:
        """Visualize HMM results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        sequences = result.raw_data['sequences']
        hidden_states = result.raw_data['hidden_states']
        viterbi_paths = result.raw_data['viterbi_paths']
        
        # Focus on first sequence for detailed visualization
        seq = sequences[0]
        true_states = hidden_states[0]
        viterbi_path = viterbi_paths[0]
        
        if self.model_type == 'weather':
            state_names = ['Sunny', 'Rainy']
            obs_names = ['Clear', 'Cloudy', 'Precipitation']
        elif self.model_type == 'dna':
            state_names = ['AT-rich', 'GC-rich']
            obs_names = ['A', 'T', 'G', 'C']
        elif self.model_type == 'financial':
            state_names = ['Bull', 'Bear']
            obs_names = ['Up', 'Flat', 'Down']
        else:
            state_names = [f'State {i}' for i in range(self.n_states)]
            obs_names = [f'Obs {i}' for i in range(self.n_observations)]
        
        if len(seq) <= 50:  # Detailed view for short sequences
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Observation sequence
            time_steps = range(len(seq))
            ax1.scatter(time_steps, seq, c='blue', alpha=0.7, s=30)
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Observation')
            ax1.set_title('Observation Sequence')
            ax1.set_yticks(range(self.n_observations))
            ax1.set_yticklabels(obs_names)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: True vs Viterbi states
            ax2.plot(time_steps, true_states, 'g-', linewidth=2, label='True States', alpha=0.7)
            ax2.plot(time_steps, viterbi_path, 'r--', linewidth=2, label='Viterbi Path', alpha=0.7)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Hidden State')
            ax2.set_title('Hidden States: True vs Viterbi')
            ax2.set_yticks(range(self.n_states))
            ax2.set_yticklabels(state_names)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: State transition visualization
            if show_probabilities:
                im = ax3.imshow(self.A, cmap='Blues', aspect='auto')
                ax3.set_xlabel('To State')
                ax3.set_ylabel('From State')
                ax3.set_title('Transition Probabilities')
                ax3.set_xticks(range(self.n_states))
                ax3.set_xticklabels(state_names)
                ax3.set_yticks(range(self.n_states))
                ax3.set_yticklabels(state_names)
                
                # Add probability values
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        ax3.text(j, i, f'{self.A[i,j]:.2f}', ha='center', va='center')
                
                plt.colorbar(im, ax=ax3)
            
            # Plot 4: Emission probabilities
            if show_probabilities:
                im2 = ax4.imshow(self.B, cmap='Reds', aspect='auto')
                ax4.set_xlabel('Observation')
                ax4.set_ylabel('Hidden State')
                ax4.set_title('Emission Probabilities')
                ax4.set_xticks(range(self.n_observations))
                ax4.set_xticklabels(obs_names)
                ax4.set_yticks(range(self.n_states))
                ax4.set_yticklabels(state_names)
                
                # Add probability values
                for i in range(self.n_states):
                    for j in range(self.n_observations):
                        ax4.text(j, i, f'{self.B[i,j]:.2f}', ha='center', va='center')
                
                plt.colorbar(im2, ax=ax4)
        
        else:  # Summary view for long sequences
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: State distribution over time (sliding window)
            window_size = 20
            time_windows = []
            state_props = []
            
            for i in range(0, len(true_states) - window_size, window_size//2):
                window = true_states[i:i+window_size]
                time_windows.append(i + window_size//2)
                props = [np.mean(np.array(window) == s) for s in range(self.n_states)]
                state_props.append(props)
            
            state_props = np.array(state_props)
            
            for s in range(self.n_states):
                ax1.plot(time_windows, state_props[:, s], label=f'{state_names[s]}', linewidth=2)
            
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('State Proportion')
            ax1.set_title('State Distribution Over Time (Sliding Window)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Viterbi accuracy over time
            accuracy_window = []
            for i in range(0, len(true_states) - window_size, window_size//2):
                true_window = true_states[i:i+window_size]
                viterbi_window = viterbi_path[i:i+window_size]
                accuracy = np.mean(np.array(true_window) == np.array(viterbi_window))
                accuracy_window.append(accuracy)
            
            ax2.plot(time_windows, accuracy_window, 'purple', linewidth=2)
            ax2.axhline(y=np.mean(accuracy_window), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(accuracy_window):.3f}')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Viterbi Accuracy')
            ax2.set_title('Viterbi Decoding Accuracy Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Observation distribution
            obs_counts = np.bincount(seq, minlength=self.n_observations)
            obs_props = obs_counts / len(seq)
            
            ax3.bar(range(self.n_observations), obs_props, alpha=0.7, color='skyblue')
            ax3.set_xlabel('Observation')
            ax3.set_ylabel('Proportion')
            ax3.set_title('Observation Distribution')
            ax3.set_xticks(range(self.n_observations))
            ax3.set_xticklabels(obs_names)
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Performance metrics
            accuracy = result.results['mean_viterbi_accuracy']
            ax4.text(0.1, 0.8, f'Sequence Length: {len(seq)}', transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.7, f'Viterbi Accuracy: {accuracy:.3f}', transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.6, f'Number of States: {self.n_states}', transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.5, f'Number of Observations: {self.n_observations}', transform=ax4.transAxes, fontsize=12)
            
            if 'log_likelihood' in result.results:
                ll = result.results['log_likelihood']
                ax4.text(0.1, 0.4, f'Log-Likelihood: {ll:.2f}', transform=ax4.transAxes, fontsize=12)
            
            ax4.set_title('Model Summary')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Additional visualization for parameter estimation
        if 'estimated_A' in result.results:
            self._visualize_parameter_estimation(result)
    
    def _visualize_parameter_estimation(self, result: SimulationResult) -> None:
        """Visualize parameter estimation results"""
        true_A = result.statistics['true_A']
        true_B = result.statistics['true_B']
        true_pi = result.statistics['true_pi']
        
        est_A = np.array(result.results['estimated_A'])
        est_B = np.array(result.results['estimated_B'])
        est_pi = np.array(result.results['estimated_pi'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Transition matrix comparison
        diff_A = np.abs(true_A - est_A)
        im1 = ax1.imshow(diff_A, cmap='Reds', aspect='auto')
        ax1.set_title('Transition Matrix Error |True - Estimated|')
        ax1.set_xlabel('To State')
        ax1.set_ylabel('From State')
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                ax1.text(j, i, f'{diff_A[i,j]:.3f}', ha='center', va='center')
        
        plt.colorbar(im1, ax=ax1)
        
        # Emission matrix comparison
        diff_B = np.abs(true_B - est_B)
        im2 = ax2.imshow(diff_B, cmap='Blues', aspect='auto')
        ax2.set_title('Emission Matrix Error |True - Estimated|')
        ax2.set_xlabel('Observation')
        ax2.set_ylabel('State')
        
        for i in range(self.n_states):
            for j in range(self.n_observations):
                ax2.text(j, i, f'{diff_B[i,j]:.3f}', ha='center', va='center')
        
        plt.colorbar(im2, ax=ax2)
        
        # Initial distribution comparison
        x = np.arange(self.n_states)
        width = 0.35
        
        ax3.bar(x - width/2, true_pi, width, label='True', alpha=0.7)
        ax3.bar(x + width/2, est_pi, width, label='Estimated', alpha=0.7)
        ax3.set_xlabel('State')
        ax3.set_ylabel('Probability')
        ax3.set_title('Initial Distribution Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Parameter estimation summary
        mse_A = np.mean((true_A - est_A)**2)
        mse_B = np.mean((true_B - est_B)**2)
        mse_pi = np.mean((true_pi - est_pi)**2)
        
        ax4.text(0.1, 0.8, f'Transition MSE: {mse_A:.6f}', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.7, f'Emission MSE: {mse_B:.6f}', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.6, f'Initial Dist MSE: {mse_pi:.6f}', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.5, f'Log-Likelihood: {result.results["log_likelihood"]:.2f}', 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.4, f'Number of Sequences: {self.n_sequences}', 
                transform=ax4.transAxes, fontsize=12)
        
        ax4.set_title('Parameter Estimation Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'model_type': {
                'type': 'choice',
                'default': 'weather',
                'choices': ['weather', 'dna', 'financial', 'custom'],
                'description': 'Type of HMM to simulate'
            },
            'n_states': {
                'type': 'int',
                'default': 2,
                'min': 2,
                'max': 10,
                'description': 'Number of hidden states (for custom models)'
            },
            'n_observations': {
                'type': 'int',
                'default': 3,
                'min': 2,
                'max': 20,
                'description': 'Number of observable symbols (for custom models)'
            },
            'sequence_length': {
                'type': 'int',
                'default': 100,
                'min': 10,
                'max': 10000,
                'description': 'Length of sequences to generate'
            },
            'n_sequences': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 100,
                'description': 'Number of sequences to generate/analyze'
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
        if self.n_states > 20:
            errors.append("n_states should not exceed 20 for performance reasons")
        if self.n_observations < 2:
            errors.append("n_observations must be at least 2")
        if self.n_observations > 50:
            errors.append("n_observations should not exceed 50 for performance reasons")
        if self.sequence_length < 10:
            errors.append("sequence_length must be at least 10")
        if self.sequence_length > 100000:
            errors.append("sequence_length should not exceed 100,000 for performance reasons")
        if self.n_sequences < 1:
            errors.append("n_sequences must be at least 1")
        if self.n_sequences > 1000:
            errors.append("n_sequences should not exceed 1,000 for performance reasons")
        if self.model_type not in ['weather', 'dna', 'financial', 'custom']:
            errors.append("model_type must be one of: weather, dna, financial, custom")
        
        return errors



