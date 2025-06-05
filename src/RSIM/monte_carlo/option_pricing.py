import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, Dict, List, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class OptionPricingMC(BaseSimulation):
    """
    Monte Carlo option pricing using the Black-Scholes-Merton framework.
    
    This simulation prices European options by simulating stock price paths using 
    geometric Brownian motion and calculating the expected payoff under risk-neutral 
    valuation. The method supports both call and put options with comprehensive 
    statistical analysis and convergence tracking.
    
    Mathematical Background:
    -----------------------
    Stock Price Dynamics (Geometric Brownian Motion):
    - dS(t) = μS(t)dt + σS(t)dW(t)
    - S(T) = S(0) × exp((r - σ²/2)T + σ√T × Z)
    - Where Z ~ N(0,1) is standard normal random variable
    
    Risk-Neutral Valuation:
    - Under risk-neutral measure, drift μ = r (risk-free rate)
    - Option price = e^(-rT) × E[max(S(T) - K, 0)] for calls
    - Option price = e^(-rT) × E[max(K - S(T), 0)] for puts
    
    Black-Scholes Formula (Analytical Benchmark):
    - Call: C = S₀N(d₁) - Ke^(-rT)N(d₂)
    - Put: P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
    - d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
    - d₂ = d₁ - σ√T
    - N(x) = cumulative standard normal distribution
    
    Statistical Properties:
    ----------------------
    - Standard error: σ_MC = √(Var[payoff]/n) / e^(rT)
    - Convergence rate: O(1/√n) - typical for Monte Carlo methods
    - 95% confidence interval: price_estimate ± 1.96 × σ_MC
    - Variance reduction techniques can improve convergence
    
    Algorithm Details:
    -----------------
    1. Generate n random standard normal variables Z_i
    2. Calculate terminal stock prices: S_T,i = S₀ × exp((r - σ²/2)T + σ√T × Z_i)
    3. Calculate payoffs: Payoff_i = max(S_T,i - K, 0) for calls
    4. Estimate option price: C ≈ e^(-rT) × (1/n) × Σ Payoff_i
    5. Track convergence and calculate confidence intervals
    
    Applications:
    ------------
    - European option pricing and validation
    - Risk management and portfolio valuation
    - Sensitivity analysis (Greeks approximation)
    - Exotic option pricing (with modifications)
    - Model validation against analytical solutions
    - Educational tool for derivatives pricing
    - Benchmark for variance reduction techniques
    
    Historical Context:
    ------------------
    - Monte Carlo methods in finance pioneered by Boyle (1977)
    - Essential for complex derivatives without analytical solutions
    - Standard tool in quantitative finance and risk management
    - Foundation for more advanced simulation techniques
    - Critical for pricing path-dependent and multi-asset options
    
    Simulation Features:
    -------------------
    - European call and put option pricing
    - Comprehensive statistical analysis and error estimation
    - Convergence tracking with confidence intervals
    - Comparison with Black-Scholes analytical prices
    - Greeks approximation through finite differences
    - Multiple visualization modes for educational purposes
    - Performance timing and efficiency metrics
    - Variance reduction technique demonstrations
    
    Parameters:
    -----------
    S0 : float, default=100.0
        Initial stock price (current market price)
        Must be positive, typically $10-$1000 for stocks
    K : float, default=100.0
        Strike price (exercise price of the option)
        Must be positive, determines moneyness of option
    T : float, default=1.0
        Time to expiration in years (e.g., 0.25 for 3 months)
        Must be positive, typically 0.01 to 5.0 years
    r : float, default=0.05
        Risk-free interest rate (annualized, as decimal)
        Typically 0.01 to 0.10 (1% to 10%)
    sigma : float, default=0.2
        Volatility (annualized, as decimal)
        Must be positive, typically 0.1 to 1.0 (10% to 100%)
    option_type : str, default='call'
        Type of option: 'call' or 'put'
    n_simulations : int, default=100000
        Number of Monte Carlo simulations
        Larger values give more accurate estimates but take longer
    show_convergence : bool, default=True
        Whether to track convergence during simulation
    random_seed : int, optional
        Seed for random number generator for reproducible results
    
    Attributes:
    -----------
    stock_prices : np.ndarray, optional
        Simulated terminal stock prices (stored for analysis)
    payoffs : np.ndarray, optional
        Calculated option payoffs for each simulation
    price_estimates : list of tuples
        Convergence data as [(simulation_count, price_estimate), ...]
    analytical_price : float
        Black-Scholes analytical price for comparison
    greeks : dict
        Approximated option Greeks (delta, gamma, theta, vega, rho)
    result : SimulationResult
        Complete simulation results including price and statistics
    
    Methods:
    --------
    configure(S0, K, T, r, sigma, option_type, n_simulations, show_convergence) : bool
        Configure option pricing parameters before running
    run(**kwargs) : SimulationResult
        Execute the option pricing simulation
    calculate_analytical_price() : float
        Calculate Black-Scholes analytical price for comparison
    calculate_greeks() : dict
        Approximate option Greeks using finite differences
    visualize(result=None, show_paths=False, n_display_paths=100) : None
        Create visualizations of results and/or price paths
    validate_parameters() : List[str]
        Validate current parameters and return any errors
    get_parameter_info() : dict
        Get parameter information for UI generation
    
    Examples:
    ---------
    >>> # Basic call option pricing
    >>> option_sim = OptionPricingMC(S0=100, K=105, T=0.25, r=0.05, sigma=0.2)
    >>> result = option_sim.run()
    >>> print(f"Call price: ${result.results['option_price']:.4f}")
    >>> print(f"Analytical price: ${result.results['analytical_price']:.4f}")
    
    >>> # Put option with high precision
    >>> put_sim = OptionPricingMC(S0=100, K=95, option_type='put', 
    ...                          n_simulations=1000000, random_seed=42)
    >>> result = put_sim.run()
    >>> put_sim.visualize()
    >>> print(f"Put price: ${result.results['option_price']:.4f}")
    
    >>> # At-the-money option with Greeks
    >>> atm_sim = OptionPricingMC(S0=100, K=100, T=1.0, sigma=0.3)
    >>> result = atm_sim.run()
    >>> greeks = atm_sim.calculate_greeks()
    >>> print(f"Delta: {greeks['delta']:.4f}")
    >>> print(f"Gamma: {greeks['gamma']:.4f}")
    
    Visualization Outputs:
    ---------------------
    Standard Mode:
    - Option price comparison (Monte Carlo vs Black-Scholes)
    - Convergence plot showing price estimate stabilization
    - Payoff distribution histogram
    - Statistical summary with confidence intervals
    
    Path Visualization Mode (show_paths=True):
    - Sample stock price paths from initial to expiration
    - Terminal price distribution
    - Payoff visualization at expiration
    - Moneyness analysis
    
    Performance Characteristics:
    ---------------------------
    - Time complexity: O(n_simulations)
    - Space complexity: O(n_simulations) for full path storage
    - Memory usage: ~24 bytes per simulation for price storage
    - Typical speeds: ~500K simulations/second on modern hardware
    - Highly parallelizable across independent paths
    
    Accuracy Guidelines:
    -------------------
    - 10³ simulations: ±$0.10 typical error (quick estimates)
    - 10⁴ simulations: ±$0.03 typical error (reasonable accuracy)
    - 10⁵ simulations: ±$0.01 typical error (good precision)
    - 10⁶ simulations: ±$0.003 typical error (high precision)
    - 10⁷ simulations: ±$0.001 typical error (research quality)
    
    Error Analysis:
    --------------
    The simulation provides comprehensive error metrics:
    - Absolute error vs Black-Scholes: |MC_price - BS_price|
    - Relative error: |MC_price - BS_price| / BS_price × 100%
    - Standard error: √(sample_variance / n_simulations)
    - 95% confidence interval bounds
    - Convergence rate analysis
    
    Option Greeks Approximation:
    ---------------------------
    - Delta (Δ): ∂V/∂S ≈ [V(S+h) - V(S-h)] / (2h)
    - Gamma (Γ): ∂²V/∂S² ≈ [V(S+h) - 2V(S) + V(S-h)] / h²
    - Theta (Θ): ∂V/∂T ≈ [V(T+h) - V(T)] / h
    - Vega (ν): ∂V/∂σ ≈ [V(σ+h) - V(σ)] / h
    - Rho (ρ): ∂V/∂r ≈ [V(r+h) - V(r)] / h
    
    Variance Reduction Techniques:
    -----------------------------
    - Antithetic variates: Use pairs (Z, -Z) to reduce variance
    - Control variates: Use correlated securities with known prices
    - Importance sampling: Sample from modified distributions
    - Stratified sampling: Ensure uniform coverage of sample space
    - Moment matching: Adjust samples to match theoretical moments
    
    Model Extensions:
    ----------------
    - American options: Add early exercise features
    - Path-dependent options: Asian, barrier, lookback options
    - Multi-asset options: Basket, spread, rainbow options
    - Stochastic volatility: Heston, SABR models
    - Jump diffusion: Merton jump-diffusion model
    - Interest rate models: Vasicek, CIR, Hull-White
    
    Educational Value:
    -----------------
    - Demonstrates risk-neutral valuation principles
    - Illustrates Monte Carlo integration in finance
    - Shows relationship between simulation and analytical methods
    - Teaches option pricing fundamentals
    - Provides intuitive understanding of derivatives
    - Demonstrates statistical convergence in practice
    
    References:
    -----------
    - Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities
    - Boyle, P. P. (1977). Options: A Monte Carlo Approach
    - Hull, J. C. (2017). Options, Futures, and Other Derivatives
    - Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering
    - Jäckel, P. (2002). Monte Carlo Methods in Finance
    """

    def __init__(self, S0: float = 100.0, K: float = 100.0, T: float = 1.0, 
                 r: float = 0.05, sigma: float = 0.2, option_type: str = 'call',
                 n_simulations: int = 100000, show_convergence: bool = True,
                 random_seed: Optional[int] = None):
        super().__init__("Monte Carlo Option Pricing")
        
        # Initialize parameters
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.n_simulations = n_simulations
        self.show_convergence = show_convergence
        
        # Store in parameters dict for base class
        self.parameters.update({
            'S0': S0,
            'K': K,
            'T': T,
            'r': r,
            'sigma': sigma,
            'option_type': option_type,
            'n_simulations': n_simulations,
            'show_convergence': show_convergence,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Internal state for analysis
        self.stock_prices = None
        self.payoffs = None
        self.price_estimates = None
        self.analytical_price = None
        self.greeks = None
        self.is_configured = True
    
    def configure(self, S0: float = 100.0, K: float = 100.0, T: float = 1.0,
                 r: float = 0.05, sigma: float = 0.2, option_type: str = 'call',
                 n_simulations: int = 100000, show_convergence: bool = True) -> bool:
        """Configure option pricing parameters"""
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.n_simulations = n_simulations
        self.show_convergence = show_convergence
        
        # Update parameters dict
        self.parameters.update({
            'S0': S0,
            'K': K,
            'T': T,
            'r': r,
            'sigma': sigma,
            'option_type': option_type,
            'n_simulations': n_simulations,
            'show_convergence': show_convergence
        })
        
        self.is_configured = True
        return True
    
    def calculate_analytical_price(self) -> float:
        """Calculate Black-Scholes analytical price"""
        from scipy.stats import norm
        
        # Black-Scholes parameters
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        if self.option_type == 'call':
            price = self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:  # put
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)

        return price
    
    def calculate_greeks(self) -> Dict[str, float]:
        """Calculate option Greeks using finite differences"""
        # Small perturbations for numerical derivatives
        h_S = 0.01 * self.S0  # 1% of stock price
        h_T = 0.01  # 1% of time
        h_sigma = 0.01  # 1% volatility
        h_r = 0.0001  # 1 basis point
        
        # Store original values
        original_S0 = self.S0
        original_T = self.T
        original_sigma = self.sigma
        original_r = self.r
        
        # Base price
        base_price = self.calculate_analytical_price()
        
        # Delta: ∂V/∂S
        self.S0 = original_S0 + h_S
        price_up_S = self.calculate_analytical_price()
        self.S0 = original_S0 - h_S
        price_down_S = self.calculate_analytical_price()
        delta = (price_up_S - price_down_S) / (2 * h_S)
        
        # Gamma: ∂²V/∂S²
        gamma = (price_up_S - 2 * base_price + price_down_S) / (h_S**2)
        
        # Theta: ∂V/∂T (note: typically negative, showing time decay)
        self.S0 = original_S0
        self.T = original_T - h_T  # Decrease time (time decay)
        price_theta = self.calculate_analytical_price()
        theta = (price_theta - base_price) / h_T  # Per year
        theta_daily = theta / 365  # Convert to daily theta
        
        # Vega: ∂V/∂σ
        self.T = original_T
        self.sigma = original_sigma + h_sigma
        price_vega = self.calculate_analytical_price()
        vega = (price_vega - base_price) / h_sigma
        
        # Rho: ∂V/∂r
        self.sigma = original_sigma
        self.r = original_r + h_r
        price_rho = self.calculate_analytical_price()
        rho = (price_rho - base_price) / h_r
        
        # Restore original values
        self.S0 = original_S0
        self.T = original_T
        self.sigma = original_sigma
        self.r = original_r
        
        greeks = {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'theta_daily': theta_daily,
            'vega': vega,
            'rho': rho
        }
        
        self.greeks = greeks
        return greeks
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute option pricing simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Generate random standard normal variables
        Z = np.random.standard_normal(self.n_simulations)
        
        # Calculate terminal stock prices using geometric Brownian motion
        # S(T) = S(0) * exp((r - σ²/2)T + σ√T * Z)
        drift = (self.r - 0.5 * self.sigma**2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * Z
        self.stock_prices = self.S0 * np.exp(drift + diffusion)
        
        # Calculate payoffs
        if self.option_type == 'call':
            self.payoffs = np.maximum(self.stock_prices - self.K, 0)
        else:  # put
            self.payoffs = np.maximum(self.K - self.stock_prices, 0)
        
        # Discount payoffs to present value
        discounted_payoffs = self.payoffs * np.exp(-self.r * self.T)
        
        # Calculate option price estimate
        option_price = np.mean(discounted_payoffs)
        
        # Calculate convergence if requested
        convergence_data = []
        if self.show_convergence:
            step_size = max(1000, self.n_simulations // 1000)
            for i in range(step_size, self.n_simulations + 1, step_size):
                running_price = np.mean(discounted_payoffs[:i])
                convergence_data.append((i, running_price))
        
        self.price_estimates = convergence_data
        
        # Calculate analytical price for comparison
        self.analytical_price = self.calculate_analytical_price()
        
        # Calculate Greeks
        greeks = self.calculate_greeks()
        
        # Statistical analysis
        payoff_std = np.std(discounted_payoffs)
        standard_error = payoff_std / np.sqrt(self.n_simulations)
        confidence_interval = 1.96 * standard_error
        
        # Moneyness analysis
        moneyness = self.S0 / self.K
        if moneyness > 1.05:
            moneyness_desc = "In-the-money"
        elif moneyness < 0.95:
            moneyness_desc = "Out-of-the-money"
        else:
            moneyness_desc = "At-the-money"
        
        execution_time = time.time() - start_time
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'option_price': option_price,
                'analytical_price': self.analytical_price,
                'absolute_error': abs(option_price - self.analytical_price),
                'relative_error': abs(option_price - self.analytical_price) / self.analytical_price * 100,
                'standard_error': standard_error,
                'confidence_interval_lower': option_price - confidence_interval,
                'confidence_interval_upper': option_price + confidence_interval,
                'moneyness': moneyness,
                'moneyness_description': moneyness_desc,
                'average_payoff': np.mean(self.payoffs),
                'payoff_std': np.std(self.payoffs),
                'max_payoff': np.max(self.payoffs),
                'min_stock_price': np.min(self.stock_prices),
                'max_stock_price': np.max(self.stock_prices),
                'greeks': greeks
            },
            statistics={
                'mean_price': option_price,
                'analytical_benchmark': self.analytical_price,
                'pricing_error': abs(option_price - self.analytical_price),
                'relative_error_percent': abs(option_price - self.analytical_price) / self.analytical_price * 100,
                'standard_error': standard_error,
                'confidence_level': 95,
                'payoff_statistics': {
                    'mean': np.mean(self.payoffs),
                    'std': np.std(self.payoffs),
                    'min': np.min(self.payoffs),
                    'max': np.max(self.payoffs),
                    'median': np.median(self.payoffs)
                }
            },
            execution_time=execution_time,
            convergence_data=convergence_data
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None,
                 show_paths: bool = False, n_display_paths: int = 100) -> None:
        """Visualize option pricing results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        # Create subplots
        if show_paths:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])
            ax1 = fig.add_subplot(gs[0, :])  # Price paths
            ax2 = fig.add_subplot(gs[1, 0])  # Convergence
            ax3 = fig.add_subplot(gs[1, 1])  # Payoff distribution
            ax4 = fig.add_subplot(gs[2, :])  # Summary
        else:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Price paths or convergence
        if show_paths and self.stock_prices is not None:
            # Generate sample paths for visualization
            n_paths = min(n_display_paths, self.n_simulations)
            time_steps = 100
            dt = self.T / time_steps
            
            # Generate paths
            times = np.linspace(0, self.T, time_steps + 1)
            paths = np.zeros((n_paths, time_steps + 1))
            paths[:, 0] = self.S0
            
            for i in range(n_paths):
                Z_path = np.random.standard_normal(time_steps)
                for t in range(time_steps):
                    drift = (self.r - 0.5 * self.sigma**2) * dt
                    diffusion = self.sigma * np.sqrt(dt) * Z_path[t]
                    paths[i, t + 1] = paths[i, t] * np.exp(drift + diffusion)
            
            # Plot paths
            for i in range(n_paths):
                alpha = 0.1 if n_paths > 50 else 0.3
                ax1.plot(times, paths[i, :], 'b-', alpha=alpha, linewidth=0.5)
            
            # Add strike price line
            ax1.axhline(y=self.K, color='red', linestyle='--', linewidth=2, 
                       label=f'Strike Price (${self.K})')
            ax1.axhline(y=self.S0, color='green', linestyle='--', linewidth=2,
                       label=f'Initial Price (${self.S0})')
            
            ax1.set_xlabel('Time (Years)')
            ax1.set_ylabel('Stock Price ($)')
            ax1.set_title(f'Sample Stock Price Paths (n={n_paths})')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        else:
            # Show price comparison
            mc_price = result.results['option_price']
            bs_price = result.results['analytical_price']
            error = result.results['absolute_error']
            
            categories = ['Monte Carlo', 'Black-Scholes']
            prices = [mc_price, bs_price]
            colors = ['skyblue', 'lightcoral']
            
            bars = ax1.bar(categories, prices, color=colors, alpha=0.7, edgecolor='black')
            ax1.set_ylabel('Option Price ($)')
            ax1.set_title(f'{self.option_type.capitalize()} Option Price Comparison')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, price in zip(bars, prices):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(prices),
                        f'${price:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # Add error text
            ax1.text(0.5, 0.95, f'Absolute Error: ${error:.4f}', 
                    transform=ax1.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Plot 2: Convergence
        if self.show_convergence and result.convergence_data:
            simulations = [point[0] for point in result.convergence_data]
            estimates = [point[1] for point in result.convergence_data]
            
            ax2.plot(simulations, estimates, 'b-', linewidth=2, label='MC Estimate')
            ax2.axhline(y=self.analytical_price, color='r', linestyle='--', 
                       linewidth=2, label='Black-Scholes')
            
            # Add confidence bands
            if len(estimates) > 10:
                std_errors = [result.results['standard_error'] * np.sqrt(result.n_simulations / n) 
                             for n in simulations]
                upper_bound = [est + 1.96*se for est, se in zip(estimates, std_errors)]
                lower_bound = [est - 1.96*se for est, se in zip(estimates, std_errors)]
                ax2.fill_between(simulations, lower_bound, upper_bound, 
                               alpha=0.2, color='blue', label='95% CI')
            
            ax2.set_xlabel('Number of Simulations')
            ax2.set_ylabel('Option Price ($)')
            ax2.set_title('Price Convergence')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            # Show Greeks
            if self.greeks:
                greek_names = ['Delta', 'Gamma', 'Theta (Daily)', 'Vega', 'Rho']
                greek_values = [
                    self.greeks['delta'],
                    self.greeks['gamma'],
                    self.greeks['theta_daily'],
                    self.greeks['vega'],
                    self.greeks['rho']
                ]
                
                bars = ax2.barh(greek_names, greek_values, color='lightgreen', alpha=0.7)
                ax2.set_xlabel('Greek Value')
                ax2.set_title('Option Greeks')
                ax2.grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                for bar, value in zip(bars, greek_values):
                    width = bar.get_width()
                    ax2.text(width + 0.01*max(abs(v) for v in greek_values), 
                            bar.get_y() + bar.get_height()/2,
                            f'{value:.4f}', ha='left', va='center')
        
        # Plot 3: Payoff distribution
        if self.payoffs is not None:
            # Filter out zero payoffs for better visualization
            non_zero_payoffs = self.payoffs[self.payoffs > 0]
            
            ax3.hist(self.payoffs, bins=50, alpha=0.7, color='lightblue', 
                    edgecolor='black', density=True)
            if len(non_zero_payoffs) > 0:
                ax3.axvline(np.mean(non_zero_payoffs), color='red', linestyle='--',
                           label=f'Mean Positive Payoff: ${np.mean(non_zero_payoffs):.2f}')
            
            ax3.set_xlabel('Payoff ($)')
            ax3.set_ylabel('Density')
            ax3.set_title('Payoff Distribution at Expiration')
            ax3.grid(True, alpha=0.3)
            if len(non_zero_payoffs) > 0:
                ax3.legend()
            
            # Add statistics text
            zero_prob = np.sum(self.payoffs == 0) / len(self.payoffs) * 100
            ax3.text(0.7, 0.8, f'Zero Payoff: {zero_prob:.1f}%', 
                    transform=ax3.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 4: Summary statistics
        ax4.axis('off')
        
        # Create summary text
        summary_text = f"""
        OPTION PRICING SUMMARY
        ═══════════════════════════════════════
        
        Option Details:
        • Type: {self.option_type.capitalize()}
        • Strike Price: ${self.K:.2f}
        • Current Stock Price: ${self.S0:.2f}
        • Time to Expiration: {self.T:.2f} years
        • Risk-free Rate: {self.r:.2%}
        • Volatility: {self.sigma:.2%}
        • Moneyness: {result.results['moneyness_description']}
        
        Pricing Results:
        • Monte Carlo Price: ${result.results['option_price']:.4f}
        • Black-Scholes Price: ${result.results['analytical_price']:.4f}
        • Absolute Error: ${result.results['absolute_error']:.4f}
        • Relative Error: {result.results['relative_error']:.3f}%
        • Standard Error: ${result.results['standard_error']:.4f}
        
        95% Confidence Interval:
        [${result.results['confidence_interval_lower']:.4f}, ${result.results['confidence_interval_upper']:.4f}]
        
        Simulation Details:
        • Number of Simulations: {self.n_simulations:,}
        • Execution Time: {result.execution_time:.3f} seconds
        • Simulations/Second: {self.n_simulations/result.execution_time:,.0f}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print("\n" + "="*60)
        print("MONTE CARLO OPTION PRICING RESULTS")
        print("="*60)
        print(f"Option Type: {self.option_type.capitalize()}")
        print(f"Underlying Price: ${self.S0:.2f}")
        print(f"Strike Price: ${self.K:.2f}")
        print(f"Time to Expiration: {self.T:.4f} years")
        print(f"Risk-free Rate: {self.r:.4f} ({self.r:.2%})")
        print(f"Volatility: {self.sigma:.4f} ({self.sigma:.2%})")
        print(f"Moneyness (S/K): {result.results['moneyness']:.4f} ({result.results['moneyness_description']})")
        print("-"*60)
        print(f"Monte Carlo Price: ${result.results['option_price']:.6f}")
        print(f"Black-Scholes Price: ${result.results['analytical_price']:.6f}")
        print(f"Absolute Error: ${result.results['absolute_error']:.6f}")
        print(f"Relative Error: {result.results['relative_error']:.4f}%")
        print(f"Standard Error: ${result.results['standard_error']:.6f}")
        print(f"95% Confidence Interval: [${result.results['confidence_interval_lower']:.6f}, ${result.results['confidence_interval_upper']:.6f}]")
        print("-"*60)
        
        if self.greeks:
            print("OPTION GREEKS (Analytical):")
            print(f"Delta (Δ): {self.greeks['delta']:.6f}")
            print(f"Gamma (Γ): {self.greeks['gamma']:.6f}")
            print(f"Theta (Θ): {self.greeks['theta']:.6f} (annual), {self.greeks['theta_daily']:.6f} (daily)")
            print(f"Vega (ν): {self.greeks['vega']:.6f}")
            print(f"Rho (ρ): {self.greeks['rho']:.6f}")
            print("-"*60)
        
        print(f"Simulation Statistics:")
        print(f"Number of Simulations: {self.n_simulations:,}")
        print(f"Execution Time: {result.execution_time:.4f} seconds")
        print(f"Simulations per Second: {self.n_simulations/result.execution_time:,.0f}")
        print(f"Average Payoff: ${result.results['average_payoff']:.4f}")
        print(f"Payoff Standard Deviation: ${result.results['payoff_std']:.4f}")
        print(f"Maximum Payoff: ${result.results['max_payoff']:.4f}")
        print(f"Stock Price Range: [${result.results['min_stock_price']:.2f}, ${result.results['max_stock_price']:.2f}]")
        print("="*60)
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'S0': {
                'type': 'float',
                'default': 100.0,
                'min': 0.01,
                'max': 10000.0,
                'description': 'Initial stock price ($)'
            },
            'K': {
                'type': 'float',
                'default': 100.0,
                'min': 0.01,
                'max': 10000.0,
                'description': 'Strike price ($)'
            },
            'T': {
                'type': 'float',
                'default': 1.0,
                'min': 0.001,
                'max': 10.0,
                'description': 'Time to expiration (years)'
            },
            'r': {
                'type': 'float',
                'default': 0.05,
                'min': -0.1,
                'max': 0.5,
                'description': 'Risk-free rate (decimal)'
            },
            'sigma': {
                'type': 'float',
                'default': 0.2,
                'min': 0.01,
                'max': 2.0,
                'description': 'Volatility (decimal)'
            },
            'option_type': {
                'type': 'choice',
                'default': 'call',
                'choices': ['call', 'put'],
                'description': 'Option type'
            },
            'n_simulations': {
                'type': 'int',
                'default': 100000,
                'min': 1000,
                'max': 10000000,
                'description': 'Number of Monte Carlo simulations'
            },
            'show_convergence': {
                'type': 'bool',
                'default': True,
                'description': 'Show convergence analysis'
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
        
        if self.S0 <= 0:
            errors.append("Initial stock price (S0) must be positive")
        if self.K <= 0:
            errors.append("Strike price (K) must be positive")
        if self.T <= 0:
            errors.append("Time to expiration (T) must be positive")
        if self.sigma <= 0:
            errors.append("Volatility (sigma) must be positive")
        if self.option_type not in ['call', 'put']:
            errors.append("Option type must be 'call' or 'put'")
        if self.n_simulations < 1000:
            errors.append("Number of simulations must be at least 1,000")
        if self.n_simulations > 10000000:
            errors.append("Number of simulations should not exceed 10,000,000 for performance")
        if self.T > 10:
            errors.append("Time to expiration should not exceed 10 years")
        if self.sigma > 2.0:
            errors.append("Volatility should not exceed 200%")
        if abs(self.r) > 0.5:
            errors.append("Risk-free rate should be between -50% and 50%")
        
        return errors
    
    def sensitivity_analysis(self, parameter: str, values: List[float]) -> Dict[str, List[float]]:
        """
        Perform sensitivity analysis by varying one parameter
        
        Parameters:
        -----------
        parameter : str
            Parameter to vary ('S0', 'K', 'T', 'r', 'sigma')
        values : List[float]
            List of values to test for the parameter
            
        Returns:
        --------
        Dict with parameter values and corresponding option prices
        """
        original_value = getattr(self, parameter)
        results = {'parameter_values': values, 'option_prices': [], 'analytical_prices': []}
        
        for value in values:
            setattr(self, parameter, value)
            
            # Run quick simulation for each value
            temp_n_sims = self.n_simulations
            self.n_simulations = min(50000, self.n_simulations)  # Use fewer sims for speed
            
            result = self.run()
            results['option_prices'].append(result.results['option_price'])
            results['analytical_prices'].append(result.results['analytical_price'])
            
            self.n_simulations = temp_n_sims
        
        # Restore original value
        setattr(self, parameter, original_value)
        
        return results
    
    def implied_volatility(self, market_price: float, tolerance: float = 1e-6, 
                          max_iterations: int = 100) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Parameters:
        -----------
        market_price : float
            Observed market price of the option
        tolerance : float
            Convergence tolerance
        max_iterations : int
            Maximum number of iterations
            
        Returns:
        --------
        Implied volatility (as decimal)
        """
        # Initial guess
        sigma_guess = 0.2
        original_sigma = self.sigma
        
        for i in range(max_iterations):
            self.sigma = sigma_guess
            
            # Calculate price and vega
            theoretical_price = self.calculate_analytical_price()
            greeks = self.calculate_greeks()
            vega = greeks['vega']
            
            # Newton-Raphson update
            price_diff = theoretical_price - market_price
            
            if abs(price_diff) < tolerance:
                break
                
            if abs(vega) < 1e-10:  # Avoid division by zero
                break
                
            sigma_guess = sigma_guess - price_diff / vega
            
            # Ensure volatility stays positive
            sigma_guess = max(0.001, sigma_guess)
        
        # Restore original volatility
        self.sigma = original_sigma
        
        return sigma_guess
