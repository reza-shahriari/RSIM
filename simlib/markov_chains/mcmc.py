import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Callable, Union, Tuple
from scipy import stats
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class MetropolisHastings(BaseSimulation):
    """
    Metropolis-Hastings MCMC algorithm for sampling from arbitrary distributions.
    
    The Metropolis-Hastings algorithm is a Markov Chain Monte Carlo method for 
    obtaining a sequence of random samples from a probability distribution for 
    which direct sampling is difficult. It works by constructing a Markov chain 
    that has the desired distribution as its equilibrium distribution.
    
    Mathematical Background:
    -----------------------
    - Target distribution: π(x) (up to normalization constant)
    - Proposal distribution: q(x'|x) (transition kernel)
    - Acceptance probability: α(x,x') = min(1, [π(x')q(x|x')] / [π(x)q(x'|x)])
    - Detailed balance: π(x)P(x,x') = π(x')P(x',x) where P(x,x') = q(x'|x)α(x,x')
    - Convergence: Chain converges to π(x) under regularity conditions
    
    Algorithm:
    ----------
    1. Start with initial state x₀
    2. For each iteration t:
       a. Propose x' ~ q(x'|xₜ)
       b. Calculate acceptance probability α(xₜ,x')
       c. Accept x' with probability α: xₜ₊₁ = x' or xₜ₊₁ = xₜ
    
    Applications:
    ------------
    - Bayesian inference
    - Statistical physics (Ising models, etc.)
    - Optimization problems
    - Machine learning (Bayesian neural networks)
    - Computational biology
    - Finance (option pricing, risk models)
    - Image processing and computer vision
    
    Simulation Features:
    -------------------
    - Multiple target distributions (Gaussian, mixture, custom)
    - Various proposal mechanisms (random walk, independent)
    - Adaptive step size tuning
    - Convergence diagnostics
    - Trace plots and autocorrelation analysis
    - Effective sample size computation
    
    Parameters:
    -----------
    target_distribution : str or callable, default='standard_normal'
        Target distribution to sample from:
        - 'standard_normal': Standard normal N(0,1)
        - 'bivariate_normal': 2D normal with correlation
        - 'mixture_gaussian': Mixture of Gaussians
        - callable: Custom log-probability function
    n_samples : int, default=10000
        Number of MCMC samples to generate
    proposal_std : float, default=1.0
        Standard deviation of proposal distribution
    burn_in : int, default=1000
        Number of burn-in samples to discard
    thin : int, default=1
        Thinning interval (keep every thin-th sample)
    initial_value : float or array, default=0.0
        Starting value for the chain
    adapt_proposal : bool, default=True
        Whether to adapt proposal std during burn-in
    random_seed : int, optional
        Seed for random number generator
    
    Attributes:
    -----------
    samples : numpy.ndarray
        Generated samples (after burn-in and thinning)
    chain : numpy.ndarray
        Full chain including burn-in
    acceptance_rate : float
        Overall acceptance rate
    result : Sim
        result : SimulationResult
        Complete simulation results including diagnostics
    
    Methods:
    --------
    configure(target_distribution, n_samples, proposal_std, burn_in, thin) : bool
        Configure MCMC parameters
    run(**kwargs) : SimulationResult
        Execute the Metropolis-Hastings sampling
    visualize(result=None, show_trace=True, show_autocorr=True) : None
        Create comprehensive visualizations
    validate_parameters() : List[str]
        Validate parameters
    get_parameter_info() : dict
        Get parameter information for UI generation
    
    Examples:
    ---------
    >>> # Sample from standard normal
    >>> mh = MetropolisHastings('standard_normal', n_samples=5000)
    >>> result = mh.run()
    >>> mh.visualize()
    
    >>> # Sample from custom distribution
    >>> def log_prob(x):
    ...     return -0.5 * x**2  # log N(0,1)
    >>> mh_custom = MetropolisHastings(log_prob, n_samples=10000)
    >>> result = mh_custom.run()
    
    References:
    -----------
    - Metropolis, N., et al. (1953). Equation of state calculations...
    - Hastings, W. K. (1970). Monte Carlo sampling methods...
    - Robert, C. P. & Casella, G. (2004). Monte Carlo Statistical Methods
    """

    def __init__(self, target_distribution: Union[str, Callable] = 'standard_normal',
                 n_samples: int = 10000, proposal_std: float = 1.0, burn_in: int = 1000,
                 thin: int = 1, initial_value: Union[float, np.ndarray] = 0.0,
                 adapt_proposal: bool = True, random_seed: Optional[int] = None):
        super().__init__("Metropolis-Hastings MCMC")
        
        self.target_distribution = target_distribution
        self.n_samples = n_samples
        self.proposal_std = proposal_std
        self.burn_in = burn_in
        self.thin = thin
        self.initial_value = initial_value
        self.adapt_proposal = adapt_proposal
        
        # Store parameters
        self.parameters.update({
            'target_distribution': target_distribution if isinstance(target_distribution, str) else 'custom',
            'n_samples': n_samples,
            'proposal_std': proposal_std,
            'burn_in': burn_in,
            'thin': thin,
            'initial_value': initial_value,
            'adapt_proposal': adapt_proposal,
            'random_seed': random_seed
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Initialize target log-probability function
        self.log_prob_func = self._create_log_prob_function()
        self.samples = None
        self.chain = None
        self.acceptance_rate = 0.0
        self.is_configured = True
    
    def _create_log_prob_function(self):
        """Create log-probability function based on target distribution"""
        if callable(self.target_distribution):
            return self.target_distribution
        
        if self.target_distribution == 'standard_normal':
            return lambda x: -0.5 * np.sum(x**2)
        
        elif self.target_distribution == 'bivariate_normal':
            # Bivariate normal with correlation
            rho = 0.7
            cov_inv = np.array([[1, -rho], [-rho, 1]]) / (1 - rho**2)
            def log_prob(x):
                if len(x) != 2:
                    return -np.inf
                return -0.5 * x.T @ cov_inv @ x
            return log_prob
        
        elif self.target_distribution == 'mixture_gaussian':
            # Mixture of two Gaussians
            def log_prob(x):
                # log(0.3*N(-2,1) + 0.7*N(2,1))
                p1 = 0.3 * np.exp(-0.5 * (x + 2)**2)
                p2 = 0.7 * np.exp(-0.5 * (x - 2)**2)
                return np.log(p1 + p2)
            return log_prob
        
        else:
            raise ValueError(f"Unknown target distribution: {self.target_distribution}")
    
    def configure(self, target_distribution: Union[str, Callable] = 'standard_normal',
                 n_samples: int = 10000, proposal_std: float = 1.0, burn_in: int = 1000,
                 thin: int = 1, initial_value: Union[float, np.ndarray] = 0.0,
                 adapt_proposal: bool = True) -> bool:
        """Configure Metropolis-Hastings parameters"""
        self.target_distribution = target_distribution
        self.n_samples = n_samples
        self.proposal_std = proposal_std
        self.burn_in = burn_in
        self.thin = thin
        self.initial_value = initial_value
        self.adapt_proposal = adapt_proposal
        
        self.parameters.update({
            'target_distribution': target_distribution if isinstance(target_distribution, str) else 'custom',
            'n_samples': n_samples,
            'proposal_std': proposal_std,
            'burn_in': burn_in,
            'thin': thin,
            'initial_value': initial_value,
            'adapt_proposal': adapt_proposal
        })
        
        self.log_prob_func = self._create_log_prob_function()
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute Metropolis-Hastings sampling"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Initialize
        current_x = np.atleast_1d(np.array(self.initial_value, dtype=float))
        dim = len(current_x)
        
        total_iterations = self.burn_in + self.n_samples * self.thin
        chain = np.zeros((total_iterations, dim))
        chain[0] = current_x
        
        current_log_prob = self.log_prob_func(current_x)
        n_accepted = 0
        proposal_std = self.proposal_std
        
        # MCMC loop
        for i in range(1, total_iterations):
            # Propose new state
            proposal = current_x + np.random.normal(0, proposal_std, dim)
            proposal_log_prob = self.log_prob_func(proposal)
            
            # Calculate acceptance probability
            log_alpha = proposal_log_prob - current_log_prob
            alpha = min(1.0, np.exp(log_alpha))
            
            # Accept or reject
            if np.random.rand() < alpha:
                current_x = proposal
                current_log_prob = proposal_log_prob
                n_accepted += 1
            
            chain[i] = current_x
            
            # Adaptive tuning during burn-in
            if self.adapt_proposal and i < self.burn_in and i % 100 == 0:
                acceptance_rate_recent = n_accepted / i
                if acceptance_rate_recent < 0.2:
                    proposal_std *= 0.9
                elif acceptance_rate_recent > 0.5:
                    proposal_std *= 1.1
        
        # Extract samples after burn-in and thinning
        samples_indices = range(self.burn_in, total_iterations, self.thin)
        samples = chain[samples_indices]
        
        self.chain = chain
        self.samples = samples
        self.acceptance_rate = n_accepted / total_iterations
        
        execution_time = time.time() - start_time
        
        # Calculate diagnostics
        effective_sample_size = self._calculate_ess()
        autocorr_time = self._calculate_autocorr_time()
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'samples': samples.tolist(),
                'acceptance_rate': self.acceptance_rate,
                'effective_sample_size': effective_sample_size,
                'autocorrelation_time': autocorr_time,
                'sample_mean': np.mean(samples, axis=0).tolist(),
                'sample_std': np.std(samples, axis=0).tolist(),
                'final_proposal_std': proposal_std
            },
            statistics={
                'mean': np.mean(samples, axis=0),
                'std': np.std(samples, axis=0),
                'acceptance_rate': self.acceptance_rate,
                'ess': effective_sample_size,
                'autocorr_time': autocorr_time
            },
            raw_data={'chain': chain, 'samples': samples},
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def _calculate_ess(self) -> float:
        """Calculate effective sample size"""
        if self.samples is None or len(self.samples) < 10:
            return 0.0
        
        # Simple ESS calculation based on autocorrelation
        autocorr_time = self._calculate_autocorr_time()
        if autocorr_time > 0:
            return len(self.samples) / (2 * autocorr_time + 1)
        return len(self.samples)
    
    def _calculate_autocorr_time(self) -> float:
        """Calculate integrated autocorrelation time"""
        if self.samples is None or len(self.samples) < 10:
            return 0.0
        
        # For multivariate case, use first dimension
        x = self.samples[:, 0] if self.samples.ndim > 1 else self.samples
        
        # Calculate autocorrelation function
        n = len(x)
        x_centered = x - np.mean(x)
        autocorr = np.correlate(x_centered, x_centered, mode='full')
        autocorr = autocorr[n-1:] / autocorr[n-1]  # Normalize
        
        # Find integrated autocorrelation time
        tau_int = 1.0
        for i in range(1, min(n//4, 100)):
            if autocorr[i] <= 0:
                break
            tau_int += 2 * autocorr[i]
        
        return tau_int
    
    def visualize(self, result: Optional[SimulationResult] = None, 
                 show_trace: bool = True, show_autocorr: bool = True) -> None:
        """Visualize MCMC results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        samples = result.raw_data['samples']
        chain = result.raw_data['chain']
        
        if samples.ndim == 1 or samples.shape[1] == 1:
            # 1D case
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            x = samples.flatten() if samples.ndim > 1 else samples
            
            # Plot 1: Trace plot
            if show_trace:
                ax1.plot(x)
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Value')
                ax1.set_title('Trace Plot')
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Histogram
            ax2.hist(x, bins=50, alpha=0.7, density=True, edgecolor='black')
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Density')
            ax2.set_title('Sample Distribution')
            ax2.grid(True, alpha=0.3)
            
            # Add theoretical distribution if available
            if self.target_distribution == 'standard_normal':
                x_theory = np.linspace(np.min(x), np.max(x), 100)
                y_theory = stats.norm.pdf(x_theory)
                ax2.plot(x_theory, y_theory, 'r-', linewidth=2, label='N(0,1)')
                ax2.legend()
            
            # Plot 3: Running average
            running_mean = np.cumsum(x) / np.arange(1, len(x) + 1)
            ax3.plot(running_mean)
            if self.target_distribution == 'standard_normal':
                ax3.axhline(y=0, color='red', linestyle='--', label='True mean')
                ax3.legend()
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Running Mean')
            ax3.set_title('Convergence of Sample Mean')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Autocorrelation
            if show_autocorr:
                max_lag = min(100, len(x) // 4)
                lags = np.arange(max_lag)
                autocorr = np.zeros(max_lag)
                
                x_centered = x - np.mean(x)
                for lag in lags:
                    if lag == 0:
                        autocorr[lag] = 1.0
                    else:
                        autocorr[lag] = np.corrcoef(x_centered[:-lag], x_centered[lag:])[0, 1]
                
                ax4.plot(lags, autocorr)
                ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                ax4.set_xlabel('Lag')
                ax4.set_ylabel('Autocorrelation')
                ax4.set_title('Autocorrelation Function')
                ax4.grid(True, alpha=0.3)
        
        else:
            # 2D case
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: 2D scatter plot
            ax1.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=1)
            ax1.set_xlabel('X1')
            ax1.set_ylabel('X2')
            ax1.set_title('2D Sample Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Trace plots for both dimensions
            ax2.plot(samples[:, 0], label='X1', alpha=0.7)
            ax2.plot(samples[:, 1], label='X2', alpha=0.7)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Value')
            ax2.set_title('Trace Plots')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Marginal distributions
            ax3.hist(samples[:, 0], bins=30, alpha=0.7, label='X1', density=True)
            ax3.hist(samples[:, 1], bins=30, alpha=0.7, label='X2', density=True)
            ax3.set_xlabel('Value')
            ax3.set_ylabel('Density')
            ax3.set_title('Marginal Distributions')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Diagnostics text
            ax4.text(0.1, 0.8, f'Acceptance Rate: {result.results["acceptance_rate"]:.3f}', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.7, f'ESS: {result.results["effective_sample_size"]:.1f}', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.6, f'Autocorr Time: {result.results["autocorrelation_time"]:.1f}', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.5, f'Sample Mean: [{np.mean(samples[:, 0]):.3f}, {np.mean(samples[:, 1]):.3f}]', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.4, f'Sample Std: [{np.std(samples[:, 0]):.3f}, {np.std(samples[:, 1]):.3f}]', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('MCMC Diagnostics')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'target_distribution': {
                'type': 'choice',
                'default': 'standard_normal',
                'choices': ['standard_normal', 'bivariate_normal', 'mixture_gaussian'],
                'description': 'Target distribution to sample from'
            },
            'n_samples': {
                'type': 'int',
                'default': 10000,
                'min': 1000,
                'max': 100000,
                'description': 'Number of samples after burn-in'
            },
            'proposal_std': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 10.0,
                'description': 'Standard deviation of proposal distribution'
            },
            'burn_in': {
                'type': 'int',
                'default': 1000,
                'min': 100,
                'max': 10000,
                'description': 'Number of burn-in samples'
            },
            'thin': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 100,
                'description': 'Thinning interval'
            },
            'adapt_proposal': {
                'type': 'bool',
                'default': True,
                'description': 'Adapt proposal std during burn-in'
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
        
        if self.n_samples < 100:
            errors.append("n_samples must be at least 100")
        if self.n_samples > 1000000:
            errors.append("n_samples should not exceed 1,000,000 for performance reasons")
        if self.proposal_std <= 0:
            errors.append("proposal_std must be positive")
        if self.burn_in < 0:
            errors.append("burn_in must be non-negative")
        if self.burn_in > self.n_samples:
            errors.append("burn_in should not exceed n_samples")
        if self.thin < 1:
            errors.append("thin must be at least 1")
        
        return errors


class GibbsSampler(BaseSimulation):
    """
    Gibbs sampling for multivariate distributions.
    
    Gibbs sampling is a special case of the Metropolis-Hastings algorithm where 
    proposals are always accepted. It samples from conditional distributions of 
    each variable given all others, making it particularly useful for hierarchical 
    models and problems where conditional distributions are easy to sample from.
    
    Mathematical Background:
    -----------------------
    - Target distribution: π(x₁, x₂, ..., xₚ)
    - Conditional distributions: π(xᵢ | x₋ᵢ) where x₋ᵢ are all variables except xᵢ
    - Full conditionals must be available and easy to sample from
    - Algorithm: Cyclically sample each xᵢ from π(xᵢ | x₋ᵢ)
    - Acceptance rate: Always 100% (proposals always accepted)
    
    Algorithm:
    ----------
    1. Initialize x⁽⁰⁾ = (x₁⁽⁰⁾, ..., xₚ⁽⁰⁾)
    2. For iteration t:
       - Sample x₁⁽ᵗ⁺¹⁾ ~ π(x₁ | x₂⁽ᵗ⁾, ..., xₚ⁽ᵗ⁾)
       - Sample x₂⁽ᵗ⁺¹⁾ ~ π(x₂ | x₁⁽ᵗ⁺¹⁾, x₃⁽ᵗ⁾, ..., xₚ⁽ᵗ⁾)
       - ...
       - Sample xₚ⁽ᵗ⁺¹⁾ ~ π(xₚ | x₁⁽ᵗ⁺¹⁾, ..., xₚ₋₁⁽ᵗ⁺¹⁾)
    
    Applications:
    ------------
    - Bayesian hierarchical models
    - Missing data imputation
    - Mixture model parameter estimation
    - Image reconstruction and denoising
    - Latent variable models
    - Spatial statistics
    - Phylogenetic inference
    
    Parameters:
    -----------
    model_type : str, default='bivariate_normal'
        Type of model to sample from:
        - 'bivariate_normal': Correlated bivariate normal
        - 'linear_regression': Bayesian linear regression
        - 'mixture_model': Gaussian mixture with unknown parameters
    n_samples : int, default=5000
        Number of samples to generate
    burn_in : int, default=1000
        Number of burn-in samples
    thin : int, default=1
        Thinning interval
    """

    def __init__(self, model_type: str = 'bivariate_normal', n_samples: int = 5000,
                 burn_in: int = 1000, thin: int = 1, random_seed: Optional[int] = None):
        super().__init__("Gibbs Sampler")
        
        self.model_type = model_type
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.thin = thin
        
        # Store parameters
        self.parameters.update({
            'model_type': model_type,
            'n_samples': n_samples,
            'burn_in': burn_in,
            'thin': thin,
            'random_seed': random_seed
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        self.samples = None
        self.chain = None
        self.is_configured = True
    
    def configure(self, model_type: str = 'bivariate_normal', n_samples: int = 5000,
                 burn_in: int = 1000, thin: int = 1) -> bool:
        """Configure Gibbs sampler parameters"""
        self.model_type = model_type
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.thin = thin
        
        self.parameters.update({
            'model_type': model_type,
            'n_samples': n_samples,
            'burn_in': burn_in,
            'thin': thin
        })
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute Gibbs sampling"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        if self.model_type == 'bivariate_normal':
            samples = self._sample_bivariate_normal()
        elif self.model_type == 'linear_regression':
            samples = self._sample_linear_regression()
        elif self.model_type == 'mixture_model':
            samples = self._sample_mixture_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        execution_time = time.time() - start_time
        
        # Calculate diagnostics
        effective_sample_size = self._calculate_ess(samples)
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'samples': samples.tolist(),
                'sample_mean': np.mean(samples, axis=0).tolist(),
                'sample_cov': np.cov(samples.T).tolist(),
                'effective_sample_size': effective_sample_size
            },
            statistics={
                'mean': np.mean(samples, axis=0),
                'cov': np.cov(samples.T),
                'ess': effective_sample_size
            },
            raw_data={'samples': samples, 'chain': self.chain},
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def _sample_bivariate_normal(self) -> np.ndarray:
        """Sample from bivariate normal using Gibbs sampling"""
        # Parameters for bivariate normal with correlation
        rho = 0.7
        
        total_iterations = self.burn_in + self.n_samples * self.thin
        chain = np.zeros((total_iterations, 2))
        
        # Initialize
        x1, x2 = 0.0, 0.0
        
        for i in range(total_iterations):
            # Sample x1 | x2
            x1 = np.random.normal(rho * x2, np.sqrt(1 - rho**2))
            
            # Sample x2 | x1
            x2 = np.random.normal(rho * x1, np.sqrt(1 - rho**2))
            
            chain[i] = [x1, x2]
        
        self.chain = chain
        
        # Extract samples after burn-in and thinning
        sample_indices = range(self.burn_in, total_iterations, self.thin)
        samples = chain[sample_indices]
        
        return samples
    
    def _sample_linear_regression(self) -> np.ndarray:
        """Sample from Bayesian linear regression posterior"""
        # Generate synthetic data
        np.random.seed(42)  # For reproducible synthetic data
        n_obs = 50
        true_beta = np.array([1.5, -2.0])
        true_sigma = 0.5
        
        X = np.column_stack([np.ones(n_obs), np.random.normal(0, 1, n_obs)])
        y = X @ true_beta + np.random.normal(0, true_sigma, n_obs)
        
        # Priors
        beta_prior_mean = np.zeros(2)
        beta_prior_prec = np.eye(2) * 0.01  # Low precision (high variance)
        sigma_prior_a = 1.0
        sigma_prior_b = 1.0
        
        total_iterations = self.burn_in + self.n_samples * self.thin
        chain = np.zeros((total_iterations, 3))  # beta0, beta1, sigma
        
        # Initialize
        beta = np.array([0.0, 0.0])
        sigma = 1.0
        
        for i in range(total_iterations):
            # Sample beta | sigma, y
            precision = beta_prior_prec + (X.T @ X) / sigma**2
            mean_vec = np.linalg.solve(precision, 
                                     beta_prior_prec @ beta_prior_mean + (X.T @ y) / sigma**2)
            cov = np.linalg.inv(precision)
            beta = np.random.multivariate_normal(mean_vec, cov)
            
            # Sample sigma | beta, y
            residuals = y - X @ beta
            sse = np.sum(residuals**2)
            a_post = sigma_prior_a + n_obs / 2
            b_post = sigma_prior_b + sse / 2
            sigma = np.sqrt(1 / np.random.gamma(a_post, 1 / b_post))
            
            chain[i] = [beta[0], beta[1], sigma]
        
        self.chain = chain
        
        # Extract samples
        sample_indices = range(self.burn_in, total_iterations, self.thin)
        samples = chain[sample_indices]
        
        return samples
    
    def _sample_mixture_model(self) -> np.ndarray:
        """Sample from Gaussian mixture model parameters"""
        # Generate synthetic mixture data
        np.random.seed(42)
        n_obs = 100
        true_weights = np.array([0.3, 0.7])
        true_means = np.array([-2.0, 2.0])
        true_vars = np.array([1.0, 1.5])
        
        # Generate data
        components = np.random.choice(2, n_obs, p=true_weights)
        data = np.zeros(n_obs)
        for i in range(n_obs):
            data[i] = np.random.normal(true_means[components[i]], np.sqrt(true_vars[components[i]]))
        
        # Priors (simplified)
        total_iterations = self.burn_in + self.n_samples * self.thin
        chain = np.zeros((total_iterations, 5))  # mu1, mu2, sigma1, sigma2, weight1
        
        # Initialize
        mu = np.array([-1.0, 1.0])
        sigma = np.array([1.0, 1.0])
        weights = np.array([0.5, 0.5])
        
        for i in range(total_iterations):
            # Sample component assignments (latent variables)
            responsibilities = np.zeros((n_obs, 2))
            for k in range(2):
                responsibilities[:, k] = weights[k] * stats.norm.pdf(data, mu[k], sigma[k])
            
            responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
            assignments = np.array([np.random.choice(2, p=resp) for resp in responsibilities])
            
            # Sample parameters | assignments
            for k in range(2):
                data_k = data[assignments == k]
                n_k = len(data_k)
                
                if n_k > 0:
                    # Sample mean (conjugate prior)
                    prior_precision = 0.1
                    posterior_precision = prior_precision + n_k
                    posterior_mean = (0.0 * prior_precision + np.sum(data_k)) / posterior_precision
                    mu[k] = np.random.normal(posterior_mean, 1/np.sqrt(posterior_precision))
                    
                    # Sample variance (conjugate prior)
                    if n_k > 1:
                        sse = np.sum((data_k - mu[k])**2)
                        sigma[k] = np.sqrt(1 / np.random.gamma(n_k/2, 2/sse))
                
                # Sample weights (Dirichlet prior)
                alpha = np.array([1.0, 1.0])  # Symmetric Dirichlet
                counts = np.bincount(assignments, minlength=2)
                weights = np.random.dirichlet(alpha + counts)
            
            chain[i] = [mu[0], mu[1], sigma[0], sigma[1], weights[0]]
        
        self.chain = chain
        
        # Extract samples
        sample_indices = range(self.burn_in, total_iterations, self.thin)
        samples = chain[sample_indices]
        
        return samples
    
    def _calculate_ess(self, samples: np.ndarray) -> float:
        """Calculate effective sample size for multivariate case"""
        if len(samples) < 10:
            return 0.0
        
        # Use first dimension for ESS calculation
        x = samples[:, 0]
        
        # Simple autocorrelation-based ESS
        n = len(x)
        x_centered = x - np.mean(x)
        autocorr = np.correlate(x_centered, x_centered, mode='full')
        autocorr = autocorr[n-1:] / autocorr[n-1]
        
        tau_int = 1.0
        for i in range(1, min(n//4, 50)):
            if autocorr[i] <= 0:
                break
            tau_int += 2 * autocorr[i]
        
        return n / (2 * tau_int + 1) if tau_int > 0 else n
    
    def visualize(self, result: Optional[SimulationResult] = None) -> None:
        """Visualize Gibbs sampling results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        samples = result.raw_data['samples']
        
        if self.model_type == 'bivariate_normal':
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 2D scatter plot
            ax1.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=10)
            ax1.set_xlabel('X1')
            ax1.set_ylabel('X2')
            ax1.set_title('Bivariate Normal Samples')
            ax1.grid(True, alpha=0.3)
            
            # Trace plots
            ax2.plot(samples[:, 0], label='X1', alpha=0.7)
            ax2.plot(samples[:, 1], label='X2', alpha=0.7)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Value')
            ax2.set_title('Trace Plots')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Marginal distributions
            ax3.hist(samples[:, 0], bins=30, alpha=0.7, label='X1', density=True)
            ax3.hist(samples[:, 1], bins=30, alpha=0.7, label='X2', density=True)
            
            # Overlay theoretical densities
            x_range = np.linspace(-4, 4, 100)
            ax3.plot(x_range, stats.norm.pdf(x_range), 'r--', label='N(0,1)')
            ax3.set_xlabel('Value')
            ax3.set_ylabel('Density')
            ax3.set_title('Marginal Distributions')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Correlation plot
            sample_corr = np.corrcoef(samples.T)[0, 1]
            ax4.text(0.1, 0.7, f'Sample Correlation: {sample_corr:.3f}', 
                    transform=ax4.transAxes, fontsize=14)
            ax4.text(0.1, 0.5, f'True Correlation: 0.700', 
                    transform=ax4.transAxes, fontsize=14)
            ax4.text(0.1, 0.3, f'ESS: {result.results["effective_sample_size"]:.1f}', 
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Diagnostics')
            ax4.axis('off')
        
        elif self.model_type == 'linear_regression':
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Trace plots for parameters
            ax1.plot(samples[:, 0], label='β₀ (intercept)', alpha=0.7)
            ax1.plot(samples[:, 1], label='β₁ (slope)', alpha=0.7)
            ax1.plot(samples[:, 2], label='σ (noise)', alpha=0.7)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Value')
            ax1.set_title('Parameter Trace Plots')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Parameter distributions
            ax2.hist(samples[:, 0], bins=30, alpha=0.7, label='β₀', density=True)
            ax2.hist(samples[:, 1], bins=30, alpha=0.7, label='β₁', density=True)
            ax2.axvline(x=1.5, color='red', linestyle='--', label='True β₀=1.5')
            ax2.axvline(x=-2.0, color='orange', linestyle='--', label='True β₁=-2.0')
            ax2.set_xlabel('Parameter Value')
            ax2.set_ylabel('Density')
            ax2.set_title('Parameter Posterior Distributions')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Noise parameter
            ax3.hist(samples[:, 2], bins=30, alpha=0.7, density=True)
            ax3.axvline(x=0.5, color='red', linestyle='--', label='True σ=0.5')
            ax3.set_xlabel('σ (noise std)')
            ax3.set_ylabel('Density')
            ax3.set_title('Noise Parameter Posterior')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Summary statistics
            ax4.text(0.1, 0.8, f'β₀ mean: {np.mean(samples[:, 0]):.3f}', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.7, f'β₁ mean: {np.mean(samples[:, 1]):.3f}', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.6, f'σ mean: {np.mean(samples[:, 2]):.3f}', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.4, f'True values: β₀=1.5, β₁=-2.0, σ=0.5', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Posterior Summary')
            ax4.axis('off')
        
        elif self.model_type == 'mixture_model':
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Trace plots for means
            ax1.plot(samples[:, 0], label='μ₁', alpha=0.7)
            ax1.plot(samples[:, 1], label='μ₂', alpha=0.7)
            ax1.axhline(y=-2.0, color='red', linestyle='--', alpha=0.5, label='True μ₁=-2')
            ax1.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='True μ₂=2')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Mean Parameter')
            ax1.set_title('Mean Parameter Traces')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Trace plots for standard deviations
            ax2.plot(samples[:, 2], label='σ₁', alpha=0.7)
            ax2.plot(samples[:, 3], label='σ₂', alpha=0.7)
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='True σ₁=1')
            ax2.axhline(y=np.sqrt(1.5), color='orange', linestyle='--', alpha=0.5, label='True σ₂=√1.5')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Std Parameter')
            ax2.set_title('Standard Deviation Parameter Traces')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Weight trace
            ax3.plot(samples[:, 4], label='w₁', alpha=0.7)
            ax3.plot(1 - samples[:, 4], label='w₂', alpha=0.7)
            ax3.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='True w₁=0.3')
            ax3.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='True w₂=0.7')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Weight')
            ax3.set_title('Mixture Weight Traces')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Parameter summary
            ax4.text(0.1, 0.8, f'μ₁: {np.mean(samples[:, 0]):.2f} ± {np.std(samples[:, 0]):.2f}', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.7, f'μ₂: {np.mean(samples[:, 1]):.2f} ± {np.std(samples[:, 1]):.2f}', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.6, f'σ₁: {np.mean(samples[:, 2]):.2f} ± {np.std(samples[:, 2]):.2f}', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.5, f'σ₂: {np.mean(samples[:, 3]):.2f} ± {np.std(samples[:, 3]):.2f}', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.4, f'w₁: {np.mean(samples[:, 4]):.2f} ± {np.std(samples[:, 4]):.2f}', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.2, f'True: μ=[-2,2], σ=[1,√1.5], w=[0.3,0.7]', 
                    transform=ax4.transAxes, fontsize=10)
            ax4.set_title('Parameter Summary')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'model_type': {
                'type': 'choice',
                'default': 'bivariate_normal',
                'choices': ['bivariate_normal', 'linear_regression', 'mixture_model'],
                'description': 'Type of model to sample from'
            },
            'n_samples': {
                'type': 'int',
                'default': 5000,
                'min': 1000,
                'max': 50000,
                'description': 'Number of samples after burn-in'
            },
            'burn_in': {
                'type': 'int',
                'default': 1000,
                'min': 100,
                'max': 10000,
                'description': 'Number of burn-in samples'
            },
            'thin': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 50,
                'description': 'Thinning interval'
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
        
        if self.n_samples < 100:
            errors.append("n_samples must be at least 100")
        if self.n_samples > 100000:
            errors.append("n_samples should not exceed 100,000 for performance reasons")
        if self.burn_in < 0:
            errors.append("burn_in must be non-negative")
        if self.burn_in > self.n_samples:
            errors.append("burn_in should not exceed n_samples")
        if self.thin < 1:
            errors.append("thin must be at least 1")
        if self.model_type not in ['bivariate_normal', 'linear_regression', 'mixture_model']:
            errors.append("model_type must be one of: bivariate_normal, linear_regression, mixture_model")
        
        return errors

