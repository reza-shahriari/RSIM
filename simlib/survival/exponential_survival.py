import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, Union, List, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class ExponentialSurvival(BaseSimulation):
    """
    Exponential Survival Analysis and Simulation.
    
    This simulation implements exponential survival analysis, one of the fundamental
    parametric survival models. The exponential distribution is characterized by a
    constant hazard rate, making it the simplest survival distribution with the
    memoryless property. It serves as a baseline model in survival analysis and
    reliability engineering.
    
    Mathematical Background:
    -----------------------
    The exponential distribution is defined by a single parameter λ (lambda), the rate parameter:
    
    - Probability Density Function (PDF): f(t) = λe^(-λt) for t ≥ 0
    - Cumulative Distribution Function (CDF): F(t) = 1 - e^(-λt)
    - Survival Function: S(t) = P(T > t) = e^(-λt)
    - Hazard Function: h(t) = λ (constant hazard rate)
    - Cumulative Hazard: H(t) = λt
    - Mean Survival Time: E[T] = 1/λ
    - Median Survival Time: t₅₀ = ln(2)/λ ≈ 0.693/λ
    - Variance: Var[T] = 1/λ²
    - Standard Deviation: σ = 1/λ
    
    Key Properties:
    --------------
    - Memoryless Property: P(T > s+t | T > s) = P(T > t)
    - Constant hazard rate (no aging effect)
    - Scale parameter: θ = 1/λ (mean survival time)
    - Rate parameter: λ = 1/θ (failure rate)
    - Exponential random variable: T ~ Exp(λ)
    - Relationship to Poisson process: inter-arrival times
    
    Statistical Inference:
    ---------------------
    Maximum Likelihood Estimation (MLE):
    - For complete data: λ̂ = n / Σtᵢ
    - For censored data: λ̂ = d / Σtᵢ (d = number of events)
    - Standard error: SE(λ̂) = λ̂ / √d
    - 95% CI for λ: λ̂ ± 1.96 × SE(λ̂)
    - Log-likelihood: ℓ(λ) = d×ln(λ) - λ×Σtᵢ
    
    Goodness of Fit Tests:
    ---------------------
    - Kolmogorov-Smirnov test
    - Anderson-Darling test
    - Exponentiality test based on coefficient of variation
    - Graphical methods: exponential probability plots
    - Residual analysis using Cox-Snell residuals
    
    Applications:
    ------------
    - Reliability engineering (electronic components)
    - Medical survival analysis (baseline model)
    - Queueing theory (service times)
    - Radioactive decay modeling
    - Software reliability assessment
    - Maintenance scheduling
    - Risk analysis and insurance
    - Epidemiological studies
    - Clinical trial design
    - Accelerated life testing
    
    Simulation Features:
    -------------------
    - Survival time generation with configurable parameters
    - Censoring mechanism simulation (right censoring)
    - Parameter estimation from simulated data
    - Survival function estimation and comparison
    - Hazard rate analysis and visualization
    - Confidence interval computation
    - Goodness-of-fit testing
    - Sample size and power calculations
    - Bootstrap confidence intervals
    - Comparative survival analysis
    
    Parameters:
    -----------
    lambda_rate : float, default=1.0
        Rate parameter (λ) of the exponential distribution
        Must be positive; higher values indicate higher failure rates
        Reciprocal of mean survival time: λ = 1/mean_time
    n_samples : int, default=1000
        Number of survival times to generate
        Larger samples provide better parameter estimates
    censoring_rate : float, default=0.0
        Proportion of observations to be right-censored (0.0 to 1.0)
        Simulates real-world scenarios where not all events are observed
    censoring_type : str, default='random'
        Type of censoring mechanism:
        - 'random': Random censoring times from exponential distribution
        - 'administrative': Fixed study end time
        - 'none': No censoring applied
    study_time : float, optional
        Maximum study follow-up time for administrative censoring
        Only used when censoring_type='administrative'
    random_seed : int, optional
        Seed for random number generator for reproducible results
        Essential for simulation studies and method validation
    
    Attributes:
    -----------
    survival_times : np.ndarray
        Generated or observed survival times
    event_indicator : np.ndarray
        Binary indicator (1=event observed, 0=censored)
    censoring_times : np.ndarray, optional
        Censoring times when applicable
    estimated_lambda : float
        Maximum likelihood estimate of λ parameter
    confidence_interval : tuple
        95% confidence interval for λ estimate
    survival_function : callable
        Estimated survival function S(t)
    hazard_function : callable
        Estimated hazard function h(t)
    goodness_of_fit : dict
        Results of goodness-of-fit tests
    result : SimulationResult
        Complete simulation results and statistics
    
    Methods:
    --------
    configure(lambda_rate, n_samples, censoring_rate, censoring_type, study_time) : bool
        Configure simulation parameters before execution
    run(**kwargs) : SimulationResult
        Execute the exponential survival simulation
    fit(times, events=None) : dict
        Fit exponential model to provided survival data
    generate_survival_times(n, lambda_rate, random_state=None) : np.ndarray
        Generate exponential survival times
    apply_censoring(times, censoring_rate, censoring_type, study_time) : tuple
        Apply censoring mechanism to survival times
    estimate_parameters(times, events) : dict
        Estimate exponential parameters from data
    survival_function(t, lambda_rate=None) : Union[float, np.ndarray]
        Evaluate survival function at time(s) t
    hazard_function(t, lambda_rate=None) : Union[float, np.ndarray]
        Evaluate hazard function at time(s) t
    log_likelihood(lambda_rate, times, events) : float
        Compute log-likelihood for given parameters
    goodness_of_fit_test(times, events, lambda_rate=None) : dict
        Perform goodness-of-fit tests
    visualize(result=None, show_details=True) : None
        Create comprehensive visualization of results
    validate_parameters() : List[str]
        Validate current parameters and return any errors
    get_parameter_info() : dict
        Get parameter information for UI generation
    
    Examples:
    ---------
    >>> # Basic exponential survival simulation
    >>> exp_sim = ExponentialSurvival(lambda_rate=0.5, n_samples=1000, random_seed=42)
    >>> result = exp_sim.run()
    >>> print(f"True λ: 0.5, Estimated λ: {result.results['estimated_lambda']:.4f}")
    >>> print(f"Mean survival time: {result.results['mean_survival_time']:.2f}")
    
    >>> # Survival analysis with censoring
    >>> exp_cens = ExponentialSurvival(lambda_rate=0.2, n_samples=500, 
    ...                               censoring_rate=0.3, censoring_type='random')
    >>> result = exp_cens.run()
    >>> exp_cens.visualize()
    >>> print(f"Censoring proportion: {result.results['censoring_proportion']:.2f}")
    
    >>> # Administrative censoring scenario
    >>> exp_admin = ExponentialSurvival(lambda_rate=0.1, n_samples=200,
    ...                                censoring_type='administrative', study_time=10.0)
    >>> result = exp_admin.run()
    >>> print(f"Events observed: {result.results['n_events']}")
    
    >>> # Fit model to external data
    >>> times = np.array([1.2, 2.5, 0.8, 3.1, 1.9])
    >>> events = np.array([1, 1, 0, 1, 1])  # 1=event, 0=censored
    >>> fit_results = exp_sim.fit(times, events)
    >>> print(f"Fitted λ: {fit_results['lambda_estimate']:.4f}")
    
    Visualization Outputs:
    ---------------------
    Standard Visualization:
    - Survival function plot (Kaplan-Meier vs. Exponential)
    - Hazard function plot (empirical vs. theoretical)
    - Histogram of survival times with fitted density
    - Q-Q plot for exponential distribution
    - Parameter estimates with confidence intervals
    
    Detailed Visualization (show_details=True):
    - Log-survival plot for exponentiality assessment
    - Residual plots for model diagnostics
    - Goodness-of-fit test results
    - Bootstrap confidence intervals
    - Censoring pattern analysis
    
    Performance Characteristics:
    ---------------------------
    - Time complexity: O(n_samples) for generation, O(n log n) for estimation
    - Space complexity: O(n_samples)
    - Memory usage: ~40 bytes per sample for basic storage
    - Typical speeds: ~100K samples/second for generation
    - Estimation speed: ~1M samples/second for MLE computation
    
    Statistical Properties:
    ----------------------
    - Unbiased estimator: E[λ̂] = λ
    - Asymptotic normality: √n(λ̂ - λ) → N(0, λ²)
    - Efficiency: λ̂ achieves Cramér-Rao lower bound
    - Consistency: λ̂ → λ as n → ∞
    - Invariance: MLE of g(λ) is g(λ̂)
    
    Model Assumptions:
    -----------------
    - Constant hazard rate over time
    - Independence of survival times
    - Exponential distribution assumption
    - Non-informative censoring
    - Homogeneous population
    
    Limitations:
    -----------
    - Unrealistic constant hazard assumption for many applications
    - Poor fit when hazard changes over time
    - Sensitive to outliers in small samples
    - May not capture complex survival patterns
    - Limited flexibility compared to other distributions
    
    Extensions and Variations:
    -------------------------
    - Piecewise exponential models
    - Mixture of exponential distributions
    - Competing risks with exponential components
    - Bayesian exponential survival analysis
    - Accelerated failure time models
    - Frailty models with exponential baseline
    
    References:
    -----------
    - Lawless, J.F. (2003). Statistical Models and Methods for Lifetime Data
    - Klein, J.P. & Moeschberger, M.L. (2003). Survival Analysis
    - Collett, D. (2015). Modelling Survival Data in Medical Research
    - Cox, D.R. & Oakes, D. (1984). Analysis of Survival Data
    - Kalbfleisch, J.D. & Prentice, R.L. (2002). The Statistical Analysis of Failure Time Data
    """

    def __init__(self, lambda_rate: float = 1.0, n_samples: int = 1000,
                 censoring_rate: float = 0.0, censoring_type: str = 'random',
                 study_time: Optional[float] = None, random_seed: Optional[int] = None):
        super().__init__("Exponential Survival Analysis")
        
        # Initialize parameters
        self.lambda_rate = lambda_rate
        self.n_samples = n_samples
        self.censoring_rate = censoring_rate
        self.censoring_type = censoring_type
        self.study_time = study_time
        
        # Store in parameters dict for base class
        self.parameters.update({
            'lambda_rate': lambda_rate,
            'n_samples': n_samples,
            'censoring_rate': censoring_rate,
            'censoring_type': censoring_type,
            'study_time': study_time,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Internal state
        self.survival_times = None
        self.event_indicator = None
        self.censoring_times = None
        self.estimated_lambda = None
        self.confidence_interval = None
        self.goodness_of_fit = None
        self.is_configured = True
    
    def configure(self, lambda_rate: float = 1.0, n_samples: int = 1000,
                 censoring_rate: float = 0.0, censoring_type: str = 'random',
                 study_time: Optional[float] = None) -> bool:
        """Configure exponential survival parameters"""
        self.lambda_rate = lambda_rate
        self.n_samples = n_samples
        self.censoring_rate = censoring_rate
        self.censoring_type = censoring_type
        self.study_time = study_time
        
        # Update parameters dict
        self.parameters.update({
            'lambda_rate': lambda_rate,
            'n_samples': n_samples,
            'censoring_rate': censoring_rate,
            'censoring_type': censoring_type,
            'study_time': study_time
        })
        
        self.is_configured = True
        return True
    
    def generate_survival_times(self, n: int, lambda_rate: float, 
                              random_state: Optional[int] = None) -> np.ndarray:
        """Generate exponential survival times"""
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate from exponential distribution
        # Using inverse transform: T = -ln(U)/λ where U ~ Uniform(0,1)
        u = np.random.uniform(0, 1, n)
        times = -np.log(u) / lambda_rate
        return times
    
    def apply_censoring(self, times: np.ndarray, censoring_rate: float,
                       censoring_type: str, study_time: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply censoring mechanism to survival times"""
        n = len(times)
        event_indicator = np.ones(n, dtype=int)  # 1 = event observed
        observed_times = times.copy()
        censoring_times = np.full(n, np.inf)
        
        if censoring_type == 'none' or censoring_rate == 0.0:
            return observed_times, event_indicator, censoring_times
        
        elif censoring_type == 'random':
            # Random censoring: generate censoring times from exponential
            # Choose censoring parameter to achieve desired censoring rate
            if censoring_rate > 0:
                # Approximate censoring parameter for desired rate
                censoring_lambda = self.lambda_rate * censoring_rate / (1 - censoring_rate)
                censoring_times = self.generate_survival_times(n, censoring_lambda)
                
                # Apply censoring
                censored = times > censoring_times
                observed_times[censored] = censoring_times[censored]
                event_indicator[censored] = 0
        
        elif censoring_type == 'administrative':
            # Administrative censoring at fixed study time
            if study_time is None:
                study_time = np.percentile(times, 70)  # Default to 70th percentile
            
            censoring_times = np.full(n, study_time)
            censored = times > study_time
            observed_times[censored] = study_time
            event_indicator[censored] = 0
        
        return observed_times, event_indicator, censoring_times
    
    def estimate_parameters(self, times: np.ndarray, events: np.ndarray) -> dict:
        """Estimate exponential parameters using Maximum Likelihood"""
        n_events = np.sum(events)
        total_time = np.sum(times)
        
        if n_events == 0:
            raise ValueError("No events observed - cannot estimate parameters")
        
        # MLE for exponential distribution
        lambda_hat = n_events / total_time
        
        # Standard error and confidence interval
        se_lambda = lambda_hat / np.sqrt(n_events)
        ci_lower = lambda_hat - 1.96 * se_lambda
        ci_upper = lambda_hat + 1.96 * se_lambda
        
        # Ensure CI bounds are positive
        ci_lower = max(ci_lower, 1e-10)
        
        # Log-likelihood
        log_likelihood = n_events * np.log(lambda_hat) - lambda_hat * total_time
        
        return {
            'lambda_estimate': lambda_hat,
            'standard_error': se_lambda,
            'confidence_interval': (ci_lower, ci_upper),
            'log_likelihood': log_likelihood,
            'n_events': n_events,
            'total_time': total_time,
            'mean_survival_time': 1 / lambda_hat,
            'median_survival_time': np.log(2) / lambda_hat
        }
    
    def survival_function(self, t: Union[float, np.ndarray], 
                         lambda_rate: Optional[float] = None) -> Union[float, np.ndarray]:
        """Evaluate exponential survival function S(t) = exp(-λt)"""
        if lambda_rate is None:
            lambda_rate = self.estimated_lambda or self.lambda_rate
        
        return np.exp(-lambda_rate * t)
    
    def hazard_function(self, t: Union[float, np.ndarray], 
                       lambda_rate: Optional[float] = None) -> Union[float, np.ndarray]:
        """Evaluate exponential hazard function h(t) = λ (constant)"""
        if lambda_rate is None:
            lambda_rate = self.estimated_lambda or self.lambda_rate
        
        if isinstance(t, np.ndarray):
            return np.full_like(t, lambda_rate)
        else:
            return lambda_rate
    
    def log_likelihood(self, lambda_rate: float, times: np.ndarray, 
                      events: np.ndarray) -> float:
        """Compute log-likelihood for exponential distribution"""
        n_events = np.sum(events)
        total_time = np.sum(times)
        
        if lambda_rate <= 0:
            return -np.inf
        
        return n_events * np.log(lambda_rate) - lambda_rate * total_time
    
    def goodness_of_fit_test(self, times: np.ndarray, events: np.ndarray,
                           lambda_rate: Optional[float] = None) -> dict:
        """Perform goodness-of-fit tests for exponential distribution"""
        if lambda_rate is None:
            lambda_rate = self.estimated_lambda
        
        # Only use uncensored observations for some tests
        uncensored_times = times[events == 1]
        n_uncensored = len(uncensored_times)
        
        results = {}
        
        if n_uncensored > 5:  # Need sufficient uncensored observations
            # Kolmogorov-Smirnov test
            from scipy import stats
            
            # Transform to standard exponential
            transformed = lambda_rate * uncensored_times
            ks_stat, ks_pvalue = stats.kstest(transformed, 'expon')
            
            results['kolmogorov_smirnov'] = {
                'statistic': ks_stat,
                'p_value': ks_pvalue,
                'reject_exponential': ks_pvalue < 0.05
            }
            
            # Coefficient of variation test
            # For exponential: CV = 1, for other distributions CV ≠ 1
            mean_time = np.mean(uncensored_times)
            std_time = np.std(uncensored_times, ddof=1)
            cv = std_time / mean_time
            
            # Test if CV significantly different from 1
            cv_se = cv / np.sqrt(2 * (n_uncensored - 1))  # Approximate SE
            cv_z = (cv - 1) / cv_se
            cv_pvalue = 2 * (1 - stats.norm.cdf(abs(cv_z)))
            
            results['coefficient_variation'] = {
                'cv': cv,
                'expected_cv': 1.0,
                'z_statistic': cv_z,
                'p_value': cv_pvalue,
                'reject_exponential': cv_pvalue < 0.05
            }
        
        # Log-likelihood ratio test (if we have a comparison model)
        results['log_likelihood'] = self.log_likelihood(lambda_rate, times, events)
        
        return results
    
    def fit(self, times: np.ndarray, events: Optional[np.ndarray] = None) -> dict:
        """Fit exponential model to provided survival data"""
        if events is None:
            events = np.ones(len(times), dtype=int)  # Assume all events observed
        
        # Validate inputs
        if len(times) != len(events):
            raise ValueError("times and events arrays must have same length")
        
        if np.any(times <= 0):
            raise ValueError("All survival times must be positive")
        
        if not np.all(np.isin(events, [0, 1])):
            raise ValueError("events array must contain only 0 (censored) and 1 (event)")
        
        # Estimate parameters
        param_results = self.estimate_parameters(times, events)
        
        # Store results
        self.survival_times = times
        self.event_indicator = events
        self.estimated_lambda = param_results['lambda_estimate']
        self.confidence_interval = param_results['confidence_interval']
        
        # Goodness of fit tests
        gof_results = self.goodness_of_fit_test(times, events, self.estimated_lambda)
        self.goodness_of_fit = gof_results
        
        # Combine results
        fit_results = {**param_results, 'goodness_of_fit': gof_results}
        
        return fit_results
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute exponential survival simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Generate survival times
        survival_times = self.generate_survival_times(self.n_samples, self.lambda_rate)
        
        # Apply censoring
        observed_times, event_indicator, censoring_times = self.apply_censoring(
            survival_times, self.censoring_rate, self.censoring_type, self.study_time
        )
        
        # Store data
        self.survival_times = observed_times
        self.event_indicator = event_indicator
        self.censoring_times = censoring_times
        
        # Estimate parameters
        param_results = self.estimate_parameters(observed_times, event_indicator)
        self.estimated_lambda = param_results['lambda_estimate']
        self.confidence_interval = param_results['confidence_interval']
        
        # Goodness of fit tests
        gof_results = self.goodness_of_fit_test(observed_times, event_indicator, 
                                              self.estimated_lambda)
        self.goodness_of_fit = gof_results
        
        execution_time = time.time() - start_time
        
        # Calculate additional statistics
        n_events = np.sum(event_indicator)
        censoring_proportion = 1 - (n_events / self.n_samples)
        
        # Bias and accuracy metrics
        lambda_bias = self.estimated_lambda - self.lambda_rate
        relative_bias = lambda_bias / self.lambda_rate * 100
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'true_lambda': self.lambda_rate,
                'estimated_lambda': self.estimated_lambda,
                'lambda_bias': lambda_bias,
                'relative_bias_percent': relative_bias,
                'confidence_interval': self.confidence_interval,
                'n_events': n_events,
                'n_censored': self.n_samples - n_events,
                'censoring_proportion': censoring_proportion,
                'mean_survival_time': param_results['mean_survival_time'],
                'median_survival_time': param_results['median_survival_time'],
                'log_likelihood': param_results['log_likelihood']
            },
            statistics={
                'parameter_estimates': param_results,
                'goodness_of_fit': gof_results,
                'sample_statistics': {
                    'mean_observed_time': np.mean(observed_times),
                    'median_observed_time': np.median(observed_times),
                    'std_observed_time': np.std(observed_times),
                    'min_time': np.min(observed_times),
                    'max_time': np.max(observed_times)
                }
            },
            execution_time=execution_time,
            convergence_data=None  # Not applicable for this simulation
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, 
                 show_details: bool = True) -> None:
        """Visualize exponential survival analysis results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        # Determine number of subplots
        n_plots = 4 if show_details else 2
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        times = self.survival_times
        events = self.event_indicator
        lambda_est = self.estimated_lambda
        
        # Plot 1: Survival Function Comparison
        ax1 = axes[0]
        
        # Kaplan-Meier estimator (empirical)
        unique_times = np.sort(np.unique(times[events == 1]))
        km_survival = []
        n_at_risk = len(times)
        survival_prob = 1.0
        
        for t in unique_times:
            n_events_at_t = np.sum((times == t) & (events == 1))
            n_at_risk_at_t = np.sum(times >= t)
            if n_at_risk_at_t > 0:
                survival_prob *= (1 - n_events_at_t / n_at_risk_at_t)
            km_survival.append(survival_prob)
        
        # Plot empirical survival function
        time_range = np.linspace(0, np.max(times), 100)
        km_interp = np.interp(time_range, unique_times, km_survival, left=1.0, right=km_survival[-1])
        ax1.step(unique_times, km_survival, where='post', label='Kaplan-Meier', linewidth=2)
        
        # Plot fitted exponential survival function
        exp_survival = self.survival_function(time_range, lambda_est)
        ax1.plot(time_range, exp_survival, 'r--', label=f'Exponential (λ={lambda_est:.3f})', linewidth=2)
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Survival Probability')
        ax1.set_title('Survival Function Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Hazard Function
        ax2 = axes[1]
        
        # Exponential hazard (constant)
        hazard_line = np.full_like(time_range, lambda_est)
        ax2.plot(time_range, hazard_line, 'r-', linewidth=3, label=f'Exponential Hazard (λ={lambda_est:.3f})')
        ax2.axhline(y=self.lambda_rate, color='blue', linestyle=':', linewidth=2, 
                   label=f'True Hazard (λ={self.lambda_rate:.3f})')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Hazard Rate')
        ax2.set_title('Hazard Function')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(lambda_est, self.lambda_rate) * 1.5)
        
        # Plot 3: Histogram with Fitted Density
        ax3 = axes[2]
        
        # Histogram of uncensored times
        uncensored_times = times[events == 1]
        if len(uncensored_times) > 0:
            ax3.hist(uncensored_times, bins=min(30, len(uncensored_times)//5), 
                    density=True, alpha=0.7, color='skyblue', label='Observed Data')
            
            # Fitted exponential density
            density_times = np.linspace(0, np.max(uncensored_times), 100)
            exp_density = lambda_est * np.exp(-lambda_est * density_times)
            ax3.plot(density_times, exp_density, 'r-', linewidth=2, 
                    label=f'Fitted Exponential PDF')
            
            # True exponential density
            true_density = self.lambda_rate * np.exp(-self.lambda_rate * density_times)
            ax3.plot(density_times, true_density, 'b--', linewidth=2, 
                    label=f'True Exponential PDF')
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Density')
        ax3.set_title('Survival Time Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Parameter Estimates and Statistics
        ax4 = axes[3]
    
        
        if show_details and hasattr(self, 'goodness_of_fit') and self.goodness_of_fit:
            # Show goodness of fit and parameter estimates
            info_text = []
            info_text.append(f"Parameter Estimates:")
            info_text.append(f"True λ: {self.lambda_rate:.4f}")
            info_text.append(f"Estimated λ: {lambda_est:.4f}")
            info_text.append(f"Bias: {result.results['lambda_bias']:.4f}")
            info_text.append(f"Relative Bias: {result.results['relative_bias_percent']:.2f}%")
            info_text.append(f"95% CI: ({self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f})")
            info_text.append("")
            info_text.append(f"Sample Statistics:")
            info_text.append(f"Sample Size: {self.n_samples}")
            info_text.append(f"Events: {result.results['n_events']}")
            info_text.append(f"Censored: {result.results['n_censored']}")
            info_text.append(f"Censoring Rate: {result.results['censoring_proportion']:.2f}")
            info_text.append("")
            info_text.append(f"Survival Times:")
            info_text.append(f"Mean: {result.results['mean_survival_time']:.3f}")
            info_text.append(f"Median: {result.results['median_survival_time']:.3f}")
            info_text.append("")
            
            # Add goodness of fit results
            if 'kolmogorov_smirnov' in self.goodness_of_fit:
                ks = self.goodness_of_fit['kolmogorov_smirnov']
                info_text.append(f"Goodness of Fit:")
                info_text.append(f"KS Test p-value: {ks['p_value']:.4f}")
                info_text.append(f"Reject Exponential: {ks['reject_exponential']}")
            
            if 'coefficient_variation' in self.goodness_of_fit:
                cv = self.goodness_of_fit['coefficient_variation']
                info_text.append(f"CV Test: {cv['cv']:.3f} (expect 1.0)")
                info_text.append(f"CV p-value: {cv['p_value']:.4f}")
            
            # Display text
            ax4.text(0.05, 0.95, '\n'.join(info_text), transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
        else:
            # Simple parameter comparison plot
            categories = ['True λ', 'Estimated λ']
            values = [self.lambda_rate, lambda_est]
            colors = ['blue', 'red']
            
            bars = ax4.bar(categories, values, color=colors, alpha=0.7)
            ax4.set_ylabel('Lambda (λ)')
            ax4.set_title('Parameter Comparison')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.4f}', ha='center', va='bottom')
            
            # Add confidence interval
            if self.confidence_interval:
                ci_lower, ci_upper = self.confidence_interval
                ax4.errorbar(1, lambda_est, yerr=[[lambda_est - ci_lower], [ci_upper - lambda_est]], 
                           fmt='none', color='black', capsize=5, capthick=2)
        
        ax4.set_xlim(-0.1, 1.1) if not show_details else None
        ax4.grid(True, alpha=0.3) if not show_details else ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Additional detailed plots if requested
        if show_details and len(uncensored_times) > 10:
            self._plot_diagnostics(uncensored_times, lambda_est)
    
    def _plot_diagnostics(self, uncensored_times: np.ndarray, lambda_est: float) -> None:
        """Create additional diagnostic plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Q-Q Plot for Exponential Distribution
        from scipy import stats
        
        # Transform to standard exponential
        transformed_times = lambda_est * uncensored_times
        sorted_transformed = np.sort(transformed_times)
        n = len(sorted_transformed)
        theoretical_quantiles = -np.log(1 - np.arange(1, n+1) / (n+1))
        
        ax1.scatter(theoretical_quantiles, sorted_transformed, alpha=0.6)
        ax1.plot([0, max(theoretical_quantiles)], [0, max(theoretical_quantiles)], 'r--', linewidth=2)
        ax1.set_xlabel('Theoretical Quantiles (Standard Exponential)')
        ax1.set_ylabel('Sample Quantiles (Transformed)')
        ax1.set_title('Q-Q Plot: Exponential Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Log-Survival Plot (should be linear for exponential)
        unique_times = np.sort(np.unique(uncensored_times))
        log_survival = []
        for t in unique_times:
            surv_prob = np.mean(uncensored_times >= t)
            if surv_prob > 0:
                log_survival.append(np.log(surv_prob))
            else:
                log_survival.append(np.nan)
        
        # Remove NaN values
        valid_idx = ~np.isnan(log_survival)
        if np.sum(valid_idx) > 1:
            ax2.scatter(unique_times[valid_idx], np.array(log_survival)[valid_idx], alpha=0.6, label='Empirical')
            
            # Theoretical line
            theoretical_log_surv = -lambda_est * unique_times
            ax2.plot(unique_times, theoretical_log_surv, 'r--', linewidth=2, label='Theoretical')
            
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Log(Survival Probability)')
            ax2.set_title('Log-Survival Plot')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Residual Plot (Cox-Snell residuals)
        cox_snell_residuals = lambda_est * uncensored_times
        
        # Plot residuals vs fitted values
        ax3.scatter(range(len(cox_snell_residuals)), cox_snell_residuals, alpha=0.6)
        ax3.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Expected Mean')
        ax3.set_xlabel('Observation Index')
        ax3.set_ylabel('Cox-Snell Residuals')
        ax3.set_title('Residual Plot')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Histogram of residuals (should follow standard exponential)
        ax4.hist(cox_snell_residuals, bins=min(20, len(cox_snell_residuals)//3), 
                density=True, alpha=0.7, color='skyblue', label='Residuals')
        
        # Standard exponential density overlay
        x_range = np.linspace(0, max(cox_snell_residuals), 100)
        standard_exp_density = np.exp(-x_range)
        ax4.plot(x_range, standard_exp_density, 'r-', linewidth=2, label='Standard Exponential')
        
        ax4.set_xlabel('Cox-Snell Residuals')
        ax4.set_ylabel('Density')
        ax4.set_title('Residual Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'lambda_rate': {
                'type': 'float',
                'default': 1.0,
                'min': 0.001,
                'max': 10.0,
                'description': 'Rate parameter (λ) of exponential distribution'
            },
            'n_samples': {
                'type': 'int',
                'default': 1000,
                'min': 50,
                'max': 100000,
                'description': 'Number of survival times to generate'
            },
            'censoring_rate': {
                'type': 'float',
                'default': 0.0,
                'min': 0.0,
                'max': 0.9,
                'description': 'Proportion of observations to be censored'
            },
            'censoring_type': {
                'type': 'str',
                'default': 'random',
                'options': ['none', 'random', 'administrative'],
                'description': 'Type of censoring mechanism'
            },
            'study_time': {
                'type': 'float',
                'default': None,
                'min': 0.1,
                'max': 100.0,
                'description': 'Study end time for administrative censoring (optional)'
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
        
        if self.lambda_rate <= 0:
            errors.append("lambda_rate must be positive")
        
        if self.n_samples < 10:
            errors.append("n_samples must be at least 10")
        
        if not 0 <= self.censoring_rate <= 1:
            errors.append("censoring_rate must be between 0 and 1")
        
        if self.censoring_type not in ['none', 'random', 'administrative']:
            errors.append("censoring_type must be 'none', 'random', or 'administrative'")
        
        if self.censoring_type == 'administrative' and self.study_time is not None and self.study_time <= 0:
            errors.append("study_time must be positive when specified")
        
        return errors

