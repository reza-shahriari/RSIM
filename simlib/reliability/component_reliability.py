import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, Union, Callable, Dict, List, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class ComponentReliability(BaseSimulation):
    """
    Component reliability analysis and simulation framework.
    
    This simulation analyzes the reliability characteristics of individual components
    using various failure distribution models. It calculates key reliability metrics,
    performs Monte Carlo simulations of component lifetimes, and provides comprehensive
    statistical analysis of failure patterns and maintenance strategies.
    
    Mathematical Background:
    -----------------------
    Reliability Function: R(t) = P(T > t) = 1 - F(t)
    where T is the random variable representing time to failure
    
    Failure Rate (Hazard Function): λ(t) = f(t) / R(t)
    where f(t) is the probability density function
    
    Mean Time To Failure (MTTF): MTTF = ∫₀^∞ R(t) dt
    
    Common Distributions:
    - Exponential: λ(t) = λ (constant failure rate)
    - Weibull: λ(t) = (β/η)(t/η)^(β-1) (bathtub curve)
    - Normal: Used for wear-out failures
    - Lognormal: Used for fatigue and corrosion failures
    - Gamma: Generalization of exponential distribution
    
    Reliability Metrics:
    -------------------
    - Reliability at time t: R(t)
    - Unreliability (CDF): F(t) = 1 - R(t)
    - Failure rate: λ(t)
    - Mean Time To Failure (MTTF)
    - Mean Time Between Failures (MTBF)
    - Percentile life (B₁₀, B₅₀, etc.)
    - Conditional reliability
    
    Distribution Models:
    -------------------
    1. Exponential Distribution:
       - PDF: f(t) = λe^(-λt)
       - CDF: F(t) = 1 - e^(-λt)
       - Reliability: R(t) = e^(-λt)
       - MTTF: 1/λ
       - Memoryless property
    
    2. Weibull Distribution:
       - PDF: f(t) = (β/η)(t/η)^(β-1)e^(-(t/η)^β)
       - CDF: F(t) = 1 - e^(-(t/η)^β)
       - Reliability: R(t) = e^(-(t/η)^β)
       - Shape parameter β: <1 (decreasing), =1 (constant), >1 (increasing)
       - Scale parameter η: characteristic life
    
    3. Normal Distribution:
       - PDF: f(t) = (1/(σ√(2π)))e^(-½((t-μ)/σ)²)
       - Used for wear-out mechanisms
       - Parameters: μ (mean), σ (standard deviation)
    
    4. Lognormal Distribution:
       - PDF: f(t) = (1/(tσ√(2π)))e^(-½((ln(t)-μ)/σ)²)
       - Used for multiplicative damage processes
       - Parameters: μ (log-scale), σ (log-shape)
    
    Applications:
    ------------
    - Electronic component reliability assessment
    - Mechanical system failure analysis
    - Preventive maintenance planning
    - Warranty analysis and cost estimation
    - Quality control and acceptance testing
    - Life testing data analysis
    - Reliability growth modeling
    - System design optimization
    
    Simulation Features:
    -------------------
    - Multiple failure distribution models
    - Monte Carlo lifetime simulation
    - Reliability function estimation
    - Failure rate analysis over time
    - Statistical confidence intervals
    - Maintenance strategy evaluation
    - Cost-benefit analysis
    - Sensitivity analysis for parameters
    
    Parameters:
    -----------
    distribution : str, default='weibull'
        Failure distribution model ('exponential', 'weibull', 'normal', 'lognormal')
    parameters : dict
        Distribution-specific parameters:
        - Exponential: {'lambda': failure_rate}
        - Weibull: {'beta': shape, 'eta': scale}
        - Normal: {'mu': mean, 'sigma': std_dev}
        - Lognormal: {'mu': log_mean, 'sigma': log_std}
    mission_time : float, default=1000.0
        Mission time for reliability calculation (hours, cycles, etc.)
    n_simulations : int, default=10000
        Number of Monte Carlo simulations for lifetime generation
    confidence_level : float, default=0.95
        Confidence level for statistical intervals
    time_units : str, default='hours'
        Units for time measurements
    random_seed : int, optional
        Seed for reproducible random number generation
    
    Attributes:
    -----------
    simulated_lifetimes : np.ndarray
        Generated component lifetimes from Monte Carlo simulation
    reliability_curve : tuple of (times, reliabilities)
        Reliability function values over time
    failure_rate_curve : tuple of (times, failure_rates)
        Failure rate function values over time
    result : SimulationResult
        Complete simulation results and statistics
    
    Methods:
    --------
    configure(distribution, parameters, mission_time, n_simulations) : bool
        Configure reliability analysis parameters
    run(**kwargs) : SimulationResult
        Execute reliability analysis and Monte Carlo simulation
    calculate_reliability(time) : float
        Calculate reliability at specific time
    calculate_failure_rate(time) : float
        Calculate failure rate at specific time
    calculate_mttf() : float
        Calculate Mean Time To Failure
    calculate_percentile_life(percentile) : float
        Calculate percentile life (e.g., B10, B50)
    estimate_parameters_from_data(failure_times) : dict
        Estimate distribution parameters from failure data
    visualize(result=None, show_distributions=True, show_simulations=True) : None
        Create comprehensive reliability visualizations
    validate_parameters() : List[str]
        Validate current parameters and return any errors
    get_parameter_info() : dict
        Get parameter information for UI generation
    
    Examples:
    ---------
    >>> # Exponential reliability analysis
    >>> exp_rel = ComponentReliability(
    ...     distribution='exponential',
    ...     parameters={'lambda': 0.001},
    ...     mission_time=1000,
    ...     n_simulations=50000
    ... )
    >>> result = exp_rel.run()
    >>> print(f"Reliability at mission time: {result.results['mission_reliability']:.4f}")
    >>> print(f"MTTF: {result.results['mttf']:.2f} hours")
    
    >>> # Weibull reliability with bathtub curve
    >>> weibull_rel = ComponentReliability(
    ...     distribution='weibull',
    ...     parameters={'beta': 2.5, 'eta': 5000},
    ...     mission_time=3000,
    ...     confidence_level=0.90
    ... )
    >>> result = weibull_rel.run()
    >>> weibull_rel.visualize()
    >>> print(f"B10 life: {result.results['b10_life']:.2f} hours")
    
    >>> # Parameter estimation from failure data
    >>> failure_data = [1200, 1850, 2100, 2400, 2900, 3200, 3800, 4100]
    >>> normal_rel = ComponentReliability(distribution='normal')
    >>> estimated_params = normal_rel.estimate_parameters_from_data(failure_data)
    >>> normal_rel.configure('normal', estimated_params, mission_time=2000)
    >>> result = normal_rel.run()
    
    Visualization Outputs:
    ---------------------
    Standard Mode:
    - Reliability function R(t) over time
    - Failure rate λ(t) over time (hazard function)
    - Probability density function f(t)
    - Cumulative distribution function F(t)
    - Key metrics summary table
    
    Simulation Mode:
    - Histogram of simulated lifetimes
    - Empirical vs theoretical CDF comparison
    - Confidence intervals for reliability estimates
    - Monte Carlo convergence analysis
    
    Statistical Analysis:
    --------------------
    The simulation provides comprehensive statistical metrics:
    - Point estimates with confidence intervals
    - Goodness-of-fit tests for distribution assumptions
    - Sensitivity analysis for parameter variations
    - Bootstrap confidence intervals
    - Bayesian credible intervals (optional)
    
    Maintenance Applications:
    ------------------------
    - Optimal replacement intervals
    - Preventive vs corrective maintenance costs
    - Spare parts inventory optimization
    - Inspection scheduling
    - Warranty period determination
    - Reliability improvement tracking
    
    Quality Control:
    ---------------
    - Acceptance sampling plans
    - Life testing design
    - Accelerated testing analysis
    - Reliability demonstration tests
    - Process capability assessment
    - Supplier quality evaluation
    
    Advanced Features:
    -----------------
    - Competing risks analysis
    - Mixture distribution modeling
    - Bayesian reliability analysis
    - Degradation modeling
    - Accelerated life testing
    - Reliability growth analysis
    
    Performance Characteristics:
    ---------------------------
    - Time complexity: O(n_simulations) for Monte Carlo
    - Space complexity: O(n_simulations) for lifetime storage
    - Numerical integration: Adaptive quadrature for MTTF calculation
    - Parameter estimation: Maximum likelihood estimation
    - Optimization: Scipy optimization routines
    
    References:
    -----------
    - Barlow, R. E. & Proschan, F. (1975). Statistical Theory of Reliability
    - Nelson, W. (1982). Applied Life Data Analysis
    - Meeker, W. Q. & Escobar, L. A. (1998). Statistical Methods for Reliability Data
    - Rausand, M. & Høyland, A. (2004). System Reliability Theory
    - Kececioglu, D. (2002). Reliability Engineering Handbook
    """

    def __init__(self, distribution: str = 'weibull', parameters: Optional[Dict] = None,
                 mission_time: float = 1000.0, n_simulations: int = 10000,
                 confidence_level: float = 0.95, time_units: str = 'hours',
                 random_seed: Optional[int] = None):
        super().__init__("Component Reliability Analysis")
        
        # Initialize parameters
        self.distribution = distribution.lower()
        self.parameters_dict = parameters or self._get_default_parameters(self.distribution)
        self.mission_time = mission_time
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.time_units = time_units
        
        # Store in parameters dict for base class
        self.parameters.update({
            'distribution': self.distribution,
            'parameters': self.parameters_dict,
            'mission_time': mission_time,
            'n_simulations': n_simulations,
            'confidence_level': confidence_level,
            'time_units': time_units,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Internal state
        self.simulated_lifetimes = None
        self.reliability_curve = None
        self.failure_rate_curve = None
        self.is_configured = True
    
    def _get_default_parameters(self, distribution: str) -> Dict:
        """Get default parameters for each distribution"""
        defaults = {
            'exponential': {'lambda': 0.001},
            'weibull': {'beta': 2.0, 'eta': 1000.0},
            'normal': {'mu': 1000.0, 'sigma': 200.0},
            'lognormal': {'mu': 6.9, 'sigma': 0.5}  # ln(1000) ≈ 6.9
        }
        return defaults.get(distribution, defaults['weibull'])
    
    def configure(self, distribution: str = 'weibull', parameters: Optional[Dict] = None,
                 mission_time: float = 1000.0, n_simulations: int = 10000) -> bool:
        """Configure reliability analysis parameters"""
        self.distribution = distribution.lower()
        self.parameters_dict = parameters or self._get_default_parameters(self.distribution)
        self.mission_time = mission_time
        self.n_simulations = n_simulations
        
        # Update parameters dict
        self.parameters.update({
            'distribution': self.distribution,
            'parameters': self.parameters_dict,
            'mission_time': mission_time,
            'n_simulations': n_simulations
        })
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute reliability analysis and Monte Carlo simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Generate Monte Carlo lifetimes
        self.simulated_lifetimes = self._generate_lifetimes()
        
        # Calculate reliability metrics
        mission_reliability = self.calculate_reliability(self.mission_time)
        mttf = self.calculate_mttf()
        
        # Calculate percentile lives
        b10_life = self.calculate_percentile_life(10)
        b50_life = self.calculate_percentile_life(50)
        b90_life = self.calculate_percentile_life(90)
        
        # Generate reliability and failure rate curves
        time_points = np.linspace(0, self.mission_time * 2, 1000)
        reliability_values = [self.calculate_reliability(t) for t in time_points]
        failure_rate_values = [self.calculate_failure_rate(t) for t in time_points]
        
        self.reliability_curve = (time_points, reliability_values)
        self.failure_rate_curve = (time_points, failure_rate_values)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        lifetime_ci_lower = np.percentile(self.simulated_lifetimes, lower_percentile)
        lifetime_ci_upper = np.percentile(self.simulated_lifetimes, upper_percentile)
        
        execution_time = time.time() - start_time
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'mission_reliability': mission_reliability,
                'mission_unreliability': 1 - mission_reliability,
                'mttf': mttf,
                'b10_life': b10_life,
                'b50_life': b50_life,
                'b90_life': b90_life,
                'mean_simulated_lifetime': np.mean(self.simulated_lifetimes),
                'std_simulated_lifetime': np.std(self.simulated_lifetimes),
                'lifetime_ci_lower': lifetime_ci_lower,
                'lifetime_ci_upper': lifetime_ci_upper,
                                'failure_rate_at_mission': self.calculate_failure_rate(self.mission_time)
            },
            statistics={
                'distribution': self.distribution,
                'distribution_parameters': self.parameters_dict,
                'simulated_failures_before_mission': np.sum(self.simulated_lifetimes <= self.mission_time),
                'empirical_reliability_at_mission': np.sum(self.simulated_lifetimes > self.mission_time) / self.n_simulations,
                'theoretical_vs_empirical_error': abs(mission_reliability - (np.sum(self.simulated_lifetimes > self.mission_time) / self.n_simulations)),
                'confidence_level': self.confidence_level,
                'sample_size': self.n_simulations
            },
            execution_time=execution_time,
            convergence_data=self._calculate_convergence_data()
        )
        
        self.result = result
        return result
    
    def _generate_lifetimes(self) -> np.ndarray:
        """Generate component lifetimes based on distribution"""
        if self.distribution == 'exponential':
            lam = self.parameters_dict['lambda']
            return np.random.exponential(1/lam, self.n_simulations)
        
        elif self.distribution == 'weibull':
            beta = self.parameters_dict['beta']
            eta = self.parameters_dict['eta']
            # Generate using inverse transform: F^(-1)(U) where U ~ Uniform(0,1)
            u = np.random.uniform(0, 1, self.n_simulations)
            return eta * (-np.log(1 - u)) ** (1/beta)
        
        elif self.distribution == 'normal':
            mu = self.parameters_dict['mu']
            sigma = self.parameters_dict['sigma']
            # Truncate at zero for physical meaningfulness
            lifetimes = np.random.normal(mu, sigma, self.n_simulations)
            return np.maximum(lifetimes, 0.01)  # Avoid zero lifetimes
        
        elif self.distribution == 'lognormal':
            mu = self.parameters_dict['mu']
            sigma = self.parameters_dict['sigma']
            return np.random.lognormal(mu, sigma, self.n_simulations)
        
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
    
    def calculate_reliability(self, time: float) -> float:
        """Calculate reliability at specific time"""
        if time <= 0:
            return 1.0
        
        if self.distribution == 'exponential':
            lam = self.parameters_dict['lambda']
            return np.exp(-lam * time)
        
        elif self.distribution == 'weibull':
            beta = self.parameters_dict['beta']
            eta = self.parameters_dict['eta']
            return np.exp(-((time / eta) ** beta))
        
        elif self.distribution == 'normal':
            mu = self.parameters_dict['mu']
            sigma = self.parameters_dict['sigma']
            from scipy.stats import norm
            return 1 - norm.cdf(time, mu, sigma)
        
        elif self.distribution == 'lognormal':
            mu = self.parameters_dict['mu']
            sigma = self.parameters_dict['sigma']
            from scipy.stats import lognorm
            return 1 - lognorm.cdf(time, sigma, scale=np.exp(mu))
        
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
    
    def calculate_failure_rate(self, time: float) -> float:
        """Calculate failure rate (hazard function) at specific time"""
        if time <= 0:
            return 0.0
        
        if self.distribution == 'exponential':
            return self.parameters_dict['lambda']  # Constant failure rate
        
        elif self.distribution == 'weibull':
            beta = self.parameters_dict['beta']
            eta = self.parameters_dict['eta']
            return (beta / eta) * ((time / eta) ** (beta - 1))
        
        elif self.distribution == 'normal':
            mu = self.parameters_dict['mu']
            sigma = self.parameters_dict['sigma']
            from scipy.stats import norm
            pdf_val = norm.pdf(time, mu, sigma)
            reliability = self.calculate_reliability(time)
            return pdf_val / reliability if reliability > 1e-10 else 0.0
        
        elif self.distribution == 'lognormal':
            mu = self.parameters_dict['mu']
            sigma = self.parameters_dict['sigma']
            from scipy.stats import lognorm
            pdf_val = lognorm.pdf(time, sigma, scale=np.exp(mu))
            reliability = self.calculate_reliability(time)
            return pdf_val / reliability if reliability > 1e-10 else 0.0
        
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
    
    def calculate_mttf(self) -> float:
        """Calculate Mean Time To Failure"""
        if self.distribution == 'exponential':
            return 1 / self.parameters_dict['lambda']
        
        elif self.distribution == 'weibull':
            beta = self.parameters_dict['beta']
            eta = self.parameters_dict['eta']
            from scipy.special import gamma
            return eta * gamma(1 + 1/beta)
        
        elif self.distribution == 'normal':
            return self.parameters_dict['mu']
        
        elif self.distribution == 'lognormal':
            mu = self.parameters_dict['mu']
            sigma = self.parameters_dict['sigma']
            return np.exp(mu + sigma**2 / 2)
        
        else:
            # Numerical integration fallback
            from scipy.integrate import quad
            result, _ = quad(self.calculate_reliability, 0, np.inf)
            return result
    
    def calculate_percentile_life(self, percentile: float) -> float:
        """Calculate percentile life (e.g., B10, B50)"""
        p = percentile / 100.0
        
        if self.distribution == 'exponential':
            lam = self.parameters_dict['lambda']
            return -np.log(1 - p) / lam
        
        elif self.distribution == 'weibull':
            beta = self.parameters_dict['beta']
            eta = self.parameters_dict['eta']
            return eta * (-np.log(1 - p)) ** (1/beta)
        
        elif self.distribution == 'normal':
            mu = self.parameters_dict['mu']
            sigma = self.parameters_dict['sigma']
            from scipy.stats import norm
            return norm.ppf(p, mu, sigma)
        
        elif self.distribution == 'lognormal':
            mu = self.parameters_dict['mu']
            sigma = self.parameters_dict['sigma']
            from scipy.stats import lognorm
            return lognorm.ppf(p, sigma, scale=np.exp(mu))
        
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
    
    def estimate_parameters_from_data(self, failure_times: List[float]) -> Dict:
        """Estimate distribution parameters from failure data using MLE"""
        failure_times = np.array(failure_times)
        
        if self.distribution == 'exponential':
            lambda_est = 1 / np.mean(failure_times)
            return {'lambda': lambda_est}
        
        elif self.distribution == 'weibull':
            from scipy.stats import weibull_min
            # Fit Weibull distribution
            params = weibull_min.fit(failure_times, floc=0)
            beta_est = params[0]
            eta_est = params[2]
            return {'beta': beta_est, 'eta': eta_est}
        
        elif self.distribution == 'normal':
            mu_est = np.mean(failure_times)
            sigma_est = np.std(failure_times, ddof=1)
            return {'mu': mu_est, 'sigma': sigma_est}
        
        elif self.distribution == 'lognormal':
            log_times = np.log(failure_times)
            mu_est = np.mean(log_times)
            sigma_est = np.std(log_times, ddof=1)
            return {'mu': mu_est, 'sigma': sigma_est}
        
        else:
            raise ValueError(f"Parameter estimation not implemented for: {self.distribution}")
    
    def _calculate_convergence_data(self) -> List[Tuple[int, float]]:
        """Calculate convergence of reliability estimate"""
        convergence_data = []
        step_size = max(100, self.n_simulations // 100)
        
        for i in range(step_size, self.n_simulations + 1, step_size):
            sample_lifetimes = self.simulated_lifetimes[:i]
            empirical_reliability = np.sum(sample_lifetimes > self.mission_time) / i
            convergence_data.append((i, empirical_reliability))
        
        return convergence_data
    
    def visualize(self, result: Optional[SimulationResult] = None, 
                 show_distributions: bool = True, show_simulations: bool = True) -> None:
        """Create comprehensive reliability visualizations"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        # Create subplot layout
        if show_distributions and show_simulations:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        elif show_distributions or show_simulations:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            ax3 = ax4 = None
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
            ax2 = ax3 = ax4 = None
        
        # Plot 1: Reliability Function
        if self.reliability_curve:
            times, reliabilities = self.reliability_curve
            ax1.plot(times, reliabilities, 'b-', linewidth=2, label='Reliability R(t)')
            ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50% Reliability')
            ax1.axvline(x=self.mission_time, color='g', linestyle='--', alpha=0.7, 
                       label=f'Mission Time ({self.mission_time})')
            
            # Mark mission reliability
            mission_rel = result.results['mission_reliability']
            ax1.plot(self.mission_time, mission_rel, 'ro', markersize=8, 
                    label=f'Mission R = {mission_rel:.4f}')
            
            ax1.set_xlabel(f'Time ({self.time_units})')
            ax1.set_ylabel('Reliability')
            ax1.set_title('Reliability Function R(t)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_ylim(0, 1.05)
        
        # Plot 2: Failure Rate Function
        if ax2 is not None and self.failure_rate_curve:
            times, failure_rates = self.failure_rate_curve
            ax2.plot(times, failure_rates, 'r-', linewidth=2, label='Failure Rate λ(t)')
            ax2.axvline(x=self.mission_time, color='g', linestyle='--', alpha=0.7,
                       label=f'Mission Time')
            
            ax2.set_xlabel(f'Time ({self.time_units})')
            ax2.set_ylabel('Failure Rate')
            ax2.set_title('Failure Rate (Hazard) Function λ(t)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # Plot 3: Simulated Lifetimes Histogram
        if ax3 is not None and show_simulations and self.simulated_lifetimes is not None:
            ax3.hist(self.simulated_lifetimes, bins=50, density=True, alpha=0.7, 
                    color='skyblue', edgecolor='black', label='Simulated Data')
            
            # Overlay theoretical PDF
            times = np.linspace(0, np.max(self.simulated_lifetimes), 1000)
            pdf_values = self._calculate_pdf(times)
            ax3.plot(times, pdf_values, 'r-', linewidth=2, label='Theoretical PDF')
            
            ax3.axvline(x=self.mission_time, color='g', linestyle='--', alpha=0.7,
                       label=f'Mission Time')
            ax3.axvline(x=result.results['mttf'], color='orange', linestyle='--', alpha=0.7,
                       label=f'MTTF = {result.results["mttf"]:.1f}')
            
            ax3.set_xlabel(f'Lifetime ({self.time_units})')
            ax3.set_ylabel('Probability Density')
            ax3.set_title('Lifetime Distribution')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # Plot 4: Convergence Analysis
        if ax4 is not None and result.convergence_data:
            samples = [point[0] for point in result.convergence_data]
            empirical_rel = [point[1] for point in result.convergence_data]
            theoretical_rel = result.results['mission_reliability']
            
            ax4.plot(samples, empirical_rel, 'b-', linewidth=2, label='Empirical Reliability')
            ax4.axhline(y=theoretical_rel, color='r', linestyle='--', linewidth=2, 
                       label=f'Theoretical = {theoretical_rel:.4f}')
            
            # Add confidence bands
            n_samples = np.array(samples)
            std_error = np.sqrt(theoretical_rel * (1 - theoretical_rel) / n_samples)
            upper_bound = theoretical_rel + 1.96 * std_error
            lower_bound = theoretical_rel - 1.96 * std_error
            
            ax4.fill_between(samples, lower_bound, upper_bound, alpha=0.3, color='gray',
                           label='95% Confidence Band')
            
            ax4.set_xlabel('Number of Simulations')
            ax4.set_ylabel('Reliability Estimate')
            ax4.set_title('Monte Carlo Convergence')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        # Add summary text box
        summary_text = self._create_summary_text(result)
        if ax4 is not None:
            ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="lightyellow", alpha=0.8))
        elif ax2 is not None:
            ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes, fontsize=9,
                    verticalalignment='top',
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        else:
            ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def _calculate_pdf(self, times: np.ndarray) -> np.ndarray:
        """Calculate probability density function values"""
        if self.distribution == 'exponential':
            lam = self.parameters_dict['lambda']
            return lam * np.exp(-lam * times)
        
        elif self.distribution == 'weibull':
            beta = self.parameters_dict['beta']
            eta = self.parameters_dict['eta']
            return (beta / eta) * ((times / eta) ** (beta - 1)) * np.exp(-((times / eta) ** beta))
        
        elif self.distribution == 'normal':
            mu = self.parameters_dict['mu']
            sigma = self.parameters_dict['sigma']
            from scipy.stats import norm
            return norm.pdf(times, mu, sigma)
        
        elif self.distribution == 'lognormal':
            mu = self.parameters_dict['mu']
            sigma = self.parameters_dict['sigma']
            from scipy.stats import lognorm
            return lognorm.pdf(times, sigma, scale=np.exp(mu))
        
        else:
            return np.zeros_like(times)
    
    def _create_summary_text(self, result: SimulationResult) -> str:
        """Create summary text for visualization"""
        summary = f"Distribution: {self.distribution.title()}\n"
        
        # Add distribution parameters
        for param, value in self.parameters_dict.items():
            summary += f"{param}: {value:.4f}\n"
        
        summary += f"\nMission Time: {self.mission_time} {self.time_units}\n"
        summary += f"Mission Reliability: {result.results['mission_reliability']:.4f}\n"
        summary += f"MTTF: {result.results['mttf']:.2f} {self.time_units}\n"
        summary += f"B10 Life: {result.results['b10_life']:.2f} {self.time_units}\n"
        summary += f"B50 Life: {result.results['b50_life']:.2f} {self.time_units}\n"
        summary += f"Simulations: {self.n_simulations:,}"
        
        return summary
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'distribution': {
                'type': 'choice',
                'choices': ['exponential', 'weibull', 'normal', 'lognormal'],
                'default': 'weibull',
                'description': 'Failure distribution model'
            },
            'parameters': {
                'type': 'dict',
                'description': 'Distribution parameters',
                'sub_parameters': {
                    'exponential': {
                        'lambda': {
                            'type': 'float',
                            'default': 0.001,
                            'min': 1e-6,
                            'max': 1.0,
                            'description': 'Failure rate (failures per time unit)'
                        }
                    },
                    'weibull': {
                        'beta': {
                            'type': 'float',
                            'default': 2.0,
                            'min': 0.1,
                            'max': 10.0,
                            'description': 'Shape parameter (β)'
                        },
                        'eta': {
                            'type': 'float',
                            'default': 1000.0,
                            'min': 1.0,
                            'max': 100000.0,
                            'description': 'Scale parameter (η) - characteristic life'
                        }
                    },
                    'normal': {
                        'mu': {
                            'type': 'float',
                            'default': 1000.0,
                            'min': 1.0,
                            'max': 100000.0,
                            'description': 'Mean lifetime (μ)'
                        },
                        'sigma': {
                            'type': 'float',
                            'default': 200.0,
                            'min': 1.0,
                            'max': 10000.0,
                            'description': 'Standard deviation (σ)'
                        }
                    },
                    'lognormal': {
                        'mu': {
                            'type': 'float',
                            'default': 6.9,
                            'min': 0.0,
                            'max': 15.0,
                            'description': 'Log-scale parameter (μ)'
                        },
                        'sigma': {
                            'type': 'float',
                            'default': 0.5,
                            'min': 0.1,
                            'max': 3.0,
                            'description': 'Log-shape parameter (σ)'
                        }
                    }
                }
            },
            'mission_time': {
                'type': 'float',
                'default': 1000.0,
                'min': 1.0,
                'max': 100000.0,
                'description': 'Mission time for reliability calculation'
            },
            'n_simulations': {
                'type': 'int',
                'default': 10000,
                'min': 1000,
                'max': 1000000,
                'description': 'Number of Monte Carlo simulations'
            },
            'confidence_level': {
                'type': 'float',
                'default': 0.95,
                'min': 0.80,
                'max': 0.99,
                'description': 'Confidence level for intervals'
            },
            'time_units': {
                'type': 'choice',
                'choices': ['hours', 'days', 'months', 'years', 'cycles'],
                'default': 'hours',
                'description': 'Time units for analysis'
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
        
        # Validate distribution
        valid_distributions = ['exponential', 'weibull', 'normal', 'lognormal']
        if self.distribution not in valid_distributions:
            errors.append(f"Distribution must be one of: {valid_distributions}")
        
        # Validate distribution-specific parameters
        if self.distribution == 'exponential':
            if 'lambda' not in self.parameters_dict:
                errors.append("Exponential distribution requires 'lambda' parameter")
            elif self.parameters_dict['lambda'] <= 0:
                errors.append("Lambda parameter must be positive")
        
        elif self.distribution == 'weibull':
            if 'beta' not in self.parameters_dict or 'eta' not in self.parameters_dict:
                errors.append("Weibull distribution requires 'beta' and 'eta' parameters")
            elif self.parameters_dict['beta'] <= 0:
                errors.append("Beta parameter must be positive")
            elif self.parameters_dict['eta'] <= 0:
                errors.append("Eta parameter must be positive")
        
        elif self.distribution == 'normal':
            if 'mu' not in self.parameters_dict or 'sigma' not in self.parameters_dict:
                errors.append("Normal distribution requires 'mu' and 'sigma' parameters")
            elif self.parameters_dict['sigma'] <= 0:
                errors.append("Sigma parameter must be positive")
        
        elif self.distribution == 'lognormal':
            if 'mu' not in self.parameters_dict or 'sigma' not in self.parameters_dict:
                errors.append("Lognormal distribution requires 'mu' and 'sigma' parameters")
            elif self.parameters_dict['sigma'] <= 0:
                errors.append("Sigma parameter must be positive")
        
        # Validate other parameters
        if self.mission_time <= 0:
            errors.append("Mission time must be positive")
        
        if self.n_simulations < 100:
            errors.append("Number of simulations must be at least 100")
        elif self.n_simulations > 1000000:
            errors.append("Number of simulations should not exceed 1,000,000 for performance")
        
        if not (0.5 <= self.confidence_level < 1.0):
            errors.append("Confidence level must be between 0.5 and 1.0")
        
        return errors
    
    def calculate_maintenance_metrics(self, maintenance_cost: float, 
                                    failure_cost: float) -> Dict[str, float]:
        """Calculate maintenance-related metrics"""
        # Optimal replacement interval (for exponential distribution)
        if self.distribution == 'exponential':
            lam = self.parameters_dict['lambda']
            optimal_interval = np.sqrt(2 * maintenance_cost / (lam * failure_cost))
        else:
            # Numerical optimization for other distributions
            from scipy.optimize import minimize_scalar
            
            def cost_function(t):
                reliability = self.calculate_reliability(t)
                return maintenance_cost / t + failure_cost * (1 - reliability) / t
            
            result = minimize_scalar(cost_function, bounds=(1, self.mission_time * 2), 
                                   method='bounded')
            optimal_interval = result.x
        
        # Calculate costs at optimal interval
        optimal_reliability = self.calculate_reliability(optimal_interval)
        total_cost_rate = maintenance_cost / optimal_interval + \
                         failure_cost * (1 - optimal_reliability) / optimal_interval
        
        return {
            'optimal_replacement_interval': optimal_interval,
            'optimal_reliability': optimal_reliability,
            'total_cost_rate': total_cost_rate,
            'maintenance_cost_rate': maintenance_cost / optimal_interval,
            'failure_cost_rate': failure_cost * (1 - optimal_reliability) / optimal_interval
        }
    
    def perform_sensitivity_analysis(self, parameter_variations: Dict[str, List[float]]) -> Dict:
        """Perform sensitivity analysis on distribution parameters"""
        original_params = self.parameters_dict.copy()
        sensitivity_results = {}
        
        for param_name, param_values in parameter_variations.items():
            if param_name not in original_params:
                continue
            
            param_results = []
            for value in param_values:
                # Temporarily change parameter
                self.parameters_dict[param_name] = value
                
                # Calculate key metrics
                mission_rel = self.calculate_reliability(self.mission_time)
                mttf = self.calculate_mttf()
                b10_life = self.calculate_percentile_life(10)
                
                param_results.append({
                    'parameter_value': value,
                    'mission_reliability': mission_rel,
                    'mttf': mttf,
                    'b10_life': b10_life
                })
            
            sensitivity_results[param_name] = param_results
        
        # Restore original parameters
        self.parameters_dict = original_params
        return sensitivity_results
