import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Tuple, Dict, Any
import sys
import os
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult, ParametricSimulation

class BlackScholesSimulation(ParametricSimulation):
    """
    Black-Scholes Option Pricing using Monte Carlo Simulation
    
    This simulation implements Monte Carlo methods to price European options under
    the Black-Scholes framework. It generates multiple random price paths for the
    underlying asset and calculates option payoffs to estimate fair option values.
    
    Mathematical Background:
    -----------------------
    Under the Black-Scholes model, the stock price follows geometric Brownian motion:
    dS = μS dt + σS dW
    
    Where:
    - S: Stock price
    - μ: Expected return (drift)
    - σ: Volatility
    - dW: Wiener process (random walk)
    
    For risk-neutral valuation:
    S(T) = S(0) * exp((r - σ²/2)T + σ√T * Z)
    
    Where:
    - r: Risk-free rate
    - T: Time to expiration
    - Z: Standard normal random variable
    
    Option Payoffs:
    --------------
    - Call Option: max(S(T) - K, 0)
    - Put Option: max(K - S(T), 0)
    
    Where K is the strike price.
    
    Monte Carlo Estimation:
    ----------------------
    Option Price ≈ e^(-rT) * (1/N) * Σ[Payoff_i]
    
    Where N is the number of simulation paths.
    
    Applications:
    ------------
    - Option pricing and valuation
    - Risk management and hedging
    - Portfolio optimization
    - Derivatives trading strategies
    - Model validation against analytical solutions
    - Sensitivity analysis (Greeks calculation)
    - Exotic option pricing (when extended)
    
    Parameters:
    -----------
    S0 : float, default=100.0
        Initial stock price
    K : float, default=105.0
        Strike price of the option
    T : float, default=1.0
        Time to expiration (in years)
    r : float, default=0.05
        Risk-free interest rate (annual)
    sigma : float, default=0.2
        Volatility of the underlying asset (annual)
    n_simulations : int, default=100000
        Number of Monte Carlo simulation paths
    option_type : str, default='call'
        Type of option ('call' or 'put')
    antithetic : bool, default=False
        Use antithetic variates for variance reduction
    control_variate : bool, default=False
        Use control variate technique
    random_seed : int, optional
        Seed for random number generator
    
    Attributes:
    -----------
    price_paths : ndarray
        Generated stock price paths
    payoffs : ndarray
        Calculated option payoffs
    analytical_price : float
        Black-Scholes analytical solution
    greeks : dict
        Option sensitivities (Delta, Gamma, Theta, Vega, Rho)
    convergence_history : list
        Price convergence as simulations increase
    
    Methods:
    --------
    configure(**kwargs) : bool
        Configure simulation parameters
    run(**kwargs) : SimulationResult
        Execute the Monte Carlo simulation
    visualize(result=None) : None
        Create comprehensive visualizations
    calculate_analytical_price() : float
        Calculate Black-Scholes analytical price
    calculate_greeks() : dict
        Calculate option Greeks numerically
    run_parameter_sweep(parameters) : list
        Analyze option sensitivity to parameters
    
    Examples:
    ---------
    >>> # Basic call option pricing
    >>> bs_sim = BlackScholesSimulation(S0=100, K=105, T=1.0, r=0.05, sigma=0.2)
    >>> result = bs_sim.run()
    >>> print(f"Option price: ${result.results['option_price']:.4f}")
    >>> print(f"Analytical price: ${result.results['analytical_price']:.4f}")
    
    >>> # Put option with high volatility
    >>> put_sim = BlackScholesSimulation(S0=100, K=100, sigma=0.4, option_type='put')
    >>> result = put_sim.run()
    >>> put_sim.visualize()
    
    >>> # Parameter sensitivity analysis
    >>> bs_sim = BlackScholesSimulation()
    >>> sweep_results = bs_sim.run_parameter_sweep({
    ...     'sigma': np.linspace(0.1, 0.5, 10)
    ... })
    """
    
    def __init__(self, S0: float = 100.0, K: float = 105.0, T: float = 1.0,
                 r: float = 0.05, sigma: float = 0.2, n_simulations: int = 100000,
                 option_type: str = 'call', antithetic: bool = False,
                 control_variate: bool = False, random_seed: Optional[int] = None):
        super().__init__("Black-Scholes Option Pricing")
        
        # Model parameters
        self.S0 = S0  # Initial stock price
        self.K = K    # Strike price
        self.T = T    # Time to expiration
        self.r = r    # Risk-free rate
        self.sigma = sigma  # Volatility
        
        # Simulation parameters
        self.n_simulations = n_simulations
        self.option_type = option_type.lower()
        self.antithetic = antithetic
        self.control_variate = control_variate
        
        # Store in parameters dict
        self.parameters.update({
            'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma,
            'n_simulations': n_simulations, 'option_type': option_type,
            'antithetic': antithetic, 'control_variate': control_variate,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Internal state
        self.price_paths = None
        self.payoffs = None
        self.analytical_price = None
        self.greeks = {}
        self.convergence_history = []
        self.is_configured = True
    
    def configure(self, S0: float = 100.0, K: float = 105.0, T: float = 1.0,
                 r: float = 0.05, sigma: float = 0.2, n_simulations: int = 100000,
                 option_type: str = 'call', **kwargs) -> bool:
        """Configure Black-Scholes simulation parameters"""
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_simulations = n_simulations
        self.option_type = option_type.lower()
        
        # Update other parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Update parameters dict
        self.parameters.update({
            'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma,
            'n_simulations': n_simulations, 'option_type': option_type
        })
        self.parameters.update(kwargs)
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute Black-Scholes Monte Carlo simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Generate random numbers
        if self.antithetic:
            n_base = self.n_simulations // 2
            z1 = np.random.standard_normal(n_base)
            z = np.concatenate([z1, -z1])
        else:
            z = np.random.standard_normal(self.n_simulations)
        
        # Generate stock price paths using Black-Scholes formula
        # S(T) = S0 * exp((r - sigma^2/2)*T + sigma*sqrt(T)*Z)
        drift = (self.r - 0.5 * self.sigma**2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * z
        self.price_paths = self.S0 * np.exp(drift + diffusion)
        
        # Calculate option payoffs
        if self.option_type == 'call':
            self.payoffs = np.maximum(self.price_paths - self.K, 0)
        elif self.option_type == 'put':
            self.payoffs = np.maximum(self.K - self.price_paths, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Apply control variate if requested
        if self.control_variate:
            # Use the stock price as control variate
            # E[S(T)] = S0 * exp(r*T)
            expected_stock_price = self.S0 * np.exp(self.r * self.T)
            stock_price_error = self.price_paths - expected_stock_price
            
            # Estimate optimal control coefficient
            cov_payoff_stock = np.cov(self.payoffs, stock_price_error)[0, 1]
            var_stock = np.var(stock_price_error)
            
            if var_stock > 0:
                beta = cov_payoff_stock / var_stock
                self.payoffs = self.payoffs - beta * stock_price_error
        
        # Calculate option price (discounted expected payoff)
        option_price = np.exp(-self.r * self.T) * np.mean(self.payoffs)
        
        # Calculate standard error
        payoff_std = np.std(self.payoffs)
        standard_error = payoff_std / np.sqrt(self.n_simulations) * np.exp(-self.r * self.T)
        
        # Calculate analytical price for comparison
        self.analytical_price = self.calculate_analytical_price()
        
        # Calculate Greeks
        self.greeks = self.calculate_greeks()
        
        # Track convergence
        self.convergence_history = []
        step_size = max(1, self.n_simulations // 100)
        for i in range(step_size, self.n_simulations + 1, step_size):
            running_price = np.exp(-self.r * self.T) * np.mean(self.payoffs[:i])
            self.convergence_history.append((i, running_price))
        
        execution_time = time.time() - start_time
        
        # Calculate additional statistics
        moneyness = self.S0 / self.K
        time_value = option_price - max(0, (self.S0 - self.K) if self.option_type == 'call' else (self.K - self.S0))
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'option_price': option_price,
                'analytical_price': self.analytical_price,
                'pricing_error': abs(option_price - self.analytical_price),
                'relative_error': abs(option_price - self.analytical_price) / self.analytical_price * 100,
                'standard_error': standard_error,
                'confidence_interval_95': (option_price - 1.96 * standard_error, 
                                         option_price + 1.96 * standard_error),
                'moneyness': moneyness,
                'time_value': time_value,
                'intrinsic_value': option_price - time_value
            },
            statistics={
                'mean_stock_price': np.mean(self.price_paths),
                'std_stock_price': np.std(self.price_paths),
                'mean_payoff': np.mean(self.payoffs),
                'std_payoff': np.std(self.payoffs),
                'max_payoff': np.max(self.payoffs),
                'prob_itm': np.mean(self.payoffs > 0),  # Probability in-the-money
                **self.greeks
            },
            execution_time=execution_time,
            convergence_data=self.convergence_history
        )
        
        self.result = result
        return result
    
    def calculate_analytical_price(self) -> float:
        """Calculate Black-Scholes analytical option price"""
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        if self.option_type == 'call':
            price = (self.S0 * stats.norm.cdf(d1) - 
                    self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2))
        else:  # put
            price = (self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2) - 
                    self.S0 * stats.norm.cdf(-d1))
        
        return price
    
    def calculate_greeks(self) -> dict:
        """Calculate option Greeks using finite differences"""
        # Small perturbations for numerical derivatives
        dS = 0.01 * self.S0
        dr = 0.0001
        dsigma = 0.001
        dT = 1/365  # One day
        
        # Delta: ∂V/∂S
        original_S0 = self.S0
        self.S0 = original_S0 + dS
        price_up = self.calculate_analytical_price()
        self.S0 = original_S0 - dS
        price_down = self.calculate_analytical_price()
        delta = (price_up - price_down) / (2 * dS)
        self.S0 = original_S0
        
        # Gamma: ∂²V/∂S²
        self.S0 = original_S0 + dS
        price_up = self.calculate_analytical_price()
        self.S0 = original_S0
        price_mid = self.calculate_analytical_price()
        self.S0 = original_S0 - dS
        price_down = self.calculate_analytical_price()
        gamma = (price_up - 2 * price_mid + price_down) / (dS**2)
        self.S0 = original_S0
        
        # Theta: ∂V/∂T
        original_T = self.T
        if self.T > dT:
            self.T = original_T - dT
            price_down = self.calculate_analytical_price()
            theta = -(self.calculate_analytical_price() - price_down) / dT
        else:
            theta = 0
        self.T = original_T
        
        # Vega: ∂V/∂σ
        original_sigma = self.sigma
        self.sigma = original_sigma + dsigma
        price_up = self.calculate_analytical_price()
        self.sigma = original_sigma - dsigma
        price_down = self.calculate_analytical_price()
        vega = (price_up - price_down) / (2 * dsigma)
        self.sigma = original_sigma
        
        # Rho: ∂V/∂r
        original_r = self.r
        self.r = original_r + dr
        price_up = self.calculate_analytical_price()
        self.r = original_r - dr
        price_down = self.calculate_analytical_price()
        rho = (price_up - price_down) / (2 * dr)
        self.r = original_r
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def visualize(self, result: Optional[SimulationResult] = None) -> None:
        """Visualize Black-Scholes option pricing results"""
        if result is None:
            result = self.result
        
        if result is None or self.price_paths is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Stock price distribution at expiration
        axes[0,0].hist(self.price_paths, bins=50, alpha=0.7, color='blue', density=True)
        axes[0,0].axvline(self.K, color='red', linestyle='--', linewidth=2, label=f'Strike Price (K={self.K})')
        axes[0,0].axvline(np.mean(self.price_paths), color='green', linestyle=':', linewidth=2, 
                         label=f'Mean Price ({np.mean(self.price_paths):.2f})')
        axes[0,0].set_xlabel('Stock Price at Expiration')
        axes[0,0].set_ylabel('Probability Density')
        axes[0,0].set_title('Stock Price Distribution at Expiration')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Option payoff distribution
        axes[0,1].hist(self.payoffs, bins=50, alpha=0.7, color='orange', density=True)
        axes[0,1].axvline(np.mean(self.payoffs), color='red', linestyle='--', linewidth=2,
                         label=f'Mean Payoff ({np.mean(self.payoffs):.4f})')
        axes[0,1].set_xlabel('Option Payoff')
        axes[0,1].set_ylabel('Probability Density')
        axes[0,1].set_title(f'{self.option_type.title()} Option Payoff Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Convergence analysis
        if self.convergence_history:
            n_sims, prices = zip(*self.convergence_history)
            axes[1,0].plot(n_sims, prices, 'b-', linewidth=2, label='Monte Carlo Price')
            axes[1,0].axhline(self.analytical_price, color='red', linestyle='--', linewidth=2,
                             label=f'Analytical Price ({self.analytical_price:.4f})')
            axes[1,0].set_xlabel('Number of Simulations')
            axes[1,0].set_ylabel('Option Price')
            axes[1,0].set_title('Price Convergence')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Results summary
        summary_text = f"""
        Option Parameters:
        • Type: {self.option_type.title()}
        • S₀ (Initial Price): ${self.S0:.2f}
        • K (Strike): ${self.K:.2f}
        • T (Time to Exp): {self.T:.2f} years
        • r (Risk-free Rate): {self.r:.2%}
        • σ (Volatility): {self.sigma:.2%}
        
        Pricing Results:
        • Monte Carlo Price: ${result.results['option_price']:.4f}
        • Analytical Price: ${result.results['analytical_price']:.4f}
        • Pricing Error: ${result.results['pricing_error']:.4f}
        • Relative Error: {result.results['relative_error']:.3f}%
        • Standard Error: ${result.results['standard_error']:.4f}
        
        Greeks:
        • Delta: {result.statistics['delta']:.4f}
        • Gamma: {result.statistics['gamma']:.4f}
        • Theta: {result.statistics['theta']:.4f}
        • Vega: {result.statistics['vega']:.4f}
        • Rho: {result.statistics['rho']:.4f}
        
        Statistics:
        • Prob ITM: {result.statistics['prob_itm']:.2%}
        • Moneyness: {result.results['moneyness']:.3f}
        • Time Value: ${result.results['time_value']:.4f}
        """
        
        axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes,
                      fontsize=9, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')
        axes[1,1].set_title('Results Summary')
        
        plt.tight_layout()
        plt.show()
    
    def run_parameter_sweep(self, parameters: dict) -> list:
        """Analyze option sensitivity (Greeks) across parameter ranges"""
        results = []
        
        for param_name, param_values in parameters.items():
            if not hasattr(self, param_name):
                continue
            
            original_value = getattr(self, param_name)
            sweep_results = []
            
            for value in param_values:
                setattr(self, param_name, value)
                self.parameters[param_name] = value
                
                # Run simulation
                result = self.run()
                sweep_results.append({
                    'parameter_value': value,
                    'option_price': result.results['option_price'],
                    'analytical_price': result.results['analytical_price'],
                    'delta': result.statistics['delta'],
                    'gamma': result.statistics['gamma'],
                    'theta': result.statistics['theta'],
                    'vega': result.statistics['vega'],
                    'rho': result.statistics['rho']
                })
            
            # Restore original value
            setattr(self, param_name, original_value)
            self.parameters[param_name] = original_value
            
            results.append({
                'parameter': param_name,
                'results': sweep_results
            })
        
        return results


class AsianOptionSimulation(BaseSimulation):
    """
    Asian (Average Price) Option Pricing using Monte Carlo Simulation
    
    Asian options are path-dependent derivatives where the payoff depends on the
    average price of the underlying asset over a specified period, rather than
    just the final price. This averaging feature reduces volatility and makes
    the options less expensive than standard European options.
    
    Mathematical Background:
    -----------------------
    The underlying asset follows geometric Brownian motion:
    dS = rS dt + σS dW
    
    For Asian options, we discretize the time period into n steps:
    S(t_i) = S(t_{i-1}) * exp((r - σ²/2)Δt + σ√Δt * Z_i)
    
    Average Price Calculation:
    -------------------------
    • Arithmetic Average: A = (1/n) * Σ S(t_i)
    • Geometric Average: A = (Π S(t_i))^(1/n)
    
    Option Payoffs:
    --------------
    • Asian Call: max(A - K, 0)
    • Asian Put: max(K - A, 0)
    
    Where A is the average price and K is the strike price.
    
    Applications:
    ------------
    - Currency hedging (reduces manipulation risk)
    - Commodity trading (reflects average market conditions)
    - Employee stock options
    - Portfolio insurance
    - Risk management for volatile assets
    - Exotic derivatives pricing
    
    Parameters:
    -----------
    S0 : float, default=100.0
        Initial stock price
    K : float, default=100.0
        Strike price of the option
    T : float, default=1.0
        Time to expiration (in years)
    r : float, default=0.05
        Risk-free interest rate (annual)
    sigma : float, default=0.2
        Volatility of the underlying asset (annual)
    n_simulations : int, default=100000
        Number of Monte Carlo simulation paths
    n_time_steps : int, default=252
        Number of time steps for path generation
    average_type : str, default='arithmetic'
        Type of average ('arithmetic' or 'geometric')
    option_type : str, default='call'
        Type of option ('call' or 'put')
    random_seed : int, optional
        Seed for random number generator
    
    Attributes:
    -----------
    price_paths : ndarray
        Generated stock price paths (n_simulations x n_time_steps)
    average_prices : ndarray
        Calculated average prices for each path
    payoffs : ndarray
        Option payoffs for each simulation
    convergence_history : list
        Price convergence tracking
    
    Methods:
    --------
    configure(**kwargs) : bool
        Configure simulation parameters
    run(**kwargs) : SimulationResult
        Execute the Asian option simulation
    visualize(result=None) : None
        Create comprehensive visualizations
    validate_parameters() : List[str]
        Validate simulation parameters
    
    Examples:
    ---------
    >>> # Arithmetic Asian call option
    >>> asian_sim = AsianOptionSimulation(S0=100, K=105, T=0.5, sigma=0.3)
    >>> result = asian_sim.run()
    >>> print(f"Asian option price: ${result.results['option_price']:.4f}")
    
    >>> # Geometric Asian put option
    >>> asian_put = AsianOptionSimulation(K=95, average_type='geometric', 
    ...                                  option_type='put')
    >>> result = asian_put.run()
    >>> asian_put.visualize()
    """
    
    def __init__(self, S0: float = 100.0, K: float = 100.0, T: float = 1.0,
                 r: float = 0.05, sigma: float = 0.2, n_simulations: int = 100000,
                 n_time_steps: int = 252, average_type: str = 'arithmetic',
                 option_type: str = 'call', random_seed: Optional[int] = None):
        super().__init__("Asian Option Pricing")
        
        # Model parameters
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
        # Simulation parameters
        self.n_simulations = n_simulations
        self.n_time_steps = n_time_steps
        self.average_type = average_type.lower()
        self.option_type = option_type.lower()
        
        # Store in parameters dict
        self.parameters.update({
            'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma,
            'n_simulations': n_simulations, 'n_time_steps': n_time_steps,
            'average_type': average_type, 'option_type': option_type,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Internal state
        self.price_paths = None
        self.average_prices = None
        self.payoffs = None
        self.convergence_history = []
        self.is_configured = True
    
    def configure(self, **kwargs) -> bool:
        """Configure Asian option simulation parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.parameters[key] = value
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute Asian option Monte Carlo simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Time step size
        dt = self.T / self.n_time_steps
        
        # Initialize price paths array
        self.price_paths = np.zeros((self.n_simulations, self.n_time_steps + 1))
        self.price_paths[:, 0] = self.S0
        
        # Generate random numbers for all paths and time steps
        random_numbers = np.random.standard_normal((self.n_simulations, self.n_time_steps))
        
        # Generate stock price paths
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        for t in range(self.n_time_steps):
            self.price_paths[:, t + 1] = (self.price_paths[:, t] * 
                                         np.exp(drift + diffusion * random_numbers[:, t]))
        
        # Calculate average prices for each path
        if self.average_type == 'arithmetic':
            # Arithmetic average: (S1 + S2 + ... + Sn) / n
            self.average_prices = np.mean(self.price_paths[:, 1:], axis=1)
        elif self.average_type == 'geometric':
            # Geometric average: (S1 * S2 * ... * Sn)^(1/n)
            # Using log transformation for numerical stability
            log_prices = np.log(self.price_paths[:, 1:])
            self.average_prices = np.exp(np.mean(log_prices, axis=1))
        else:
            raise ValueError("average_type must be 'arithmetic' or 'geometric'")
        
        # Calculate option payoffs
        if self.option_type == 'call':
            self.payoffs = np.maximum(self.average_prices - self.K, 0)
        elif self.option_type == 'put':
            self.payoffs = np.maximum(self.K - self.average_prices, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Calculate option price (discounted expected payoff)
        option_price = np.exp(-self.r * self.T) * np.mean(self.payoffs)
        
        # Calculate standard error
        payoff_std = np.std(self.payoffs)
        standard_error = payoff_std / np.sqrt(self.n_simulations) * np.exp(-self.r * self.T)
        
        # Track convergence
        self.convergence_history = []
        step_size = max(1, self.n_simulations // 100)
        for i in range(step_size, self.n_simulations + 1, step_size):
            running_price = np.exp(-self.r * self.T) * np.mean(self.payoffs[:i])
            self.convergence_history.append((i, running_price))
        
        execution_time = time.time() - start_time
        
        # Calculate additional statistics
        final_prices = self.price_paths[:, -1]
        moneyness = self.S0 / self.K
        
        # Volatility reduction compared to European option
        european_payoffs = (np.maximum(final_prices - self.K, 0) if self.option_type == 'call' 
                           else np.maximum(self.K - final_prices, 0))
        european_price = np.exp(-self.r * self.T) * np.mean(european_payoffs)
        volatility_reduction = 1 - (np.std(self.payoffs) / np.std(european_payoffs))
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'option_price': option_price,
                'standard_error': standard_error,
                'confidence_interval_95': (option_price - 1.96 * standard_error, 
                                         option_price + 1.96 * standard_error),
                'moneyness': moneyness,
                'european_equivalent_price': european_price,
                'asian_discount': (european_price - option_price) / european_price * 100,
                'volatility_reduction': volatility_reduction * 100
            },
            statistics={
                'mean_average_price': np.mean(self.average_prices),
                'std_average_price': np.std(self.average_prices),
                'mean_final_price': np.mean(final_prices),
                'std_final_price': np.std(final_prices),
                'mean_payoff': np.mean(self.payoffs),
                'std_payoff': np.std(self.payoffs),
                'max_payoff': np.max(self.payoffs),
                'prob_itm': np.mean(self.payoffs > 0),
                'avg_path_volatility': np.mean(np.std(self.price_paths, axis=1))
            },
            execution_time=execution_time,
            convergence_data=self.convergence_history
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None) -> None:
        """Visualize Asian option simulation results"""
        if result is None:
            result = self.result
        
        if result is None or self.price_paths is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Sample price paths
        time_grid = np.linspace(0, self.T, self.n_time_steps + 1)
        n_paths_to_plot = min(100, self.n_simulations)
        
        for i in range(n_paths_to_plot):
            axes[0,0].plot(time_grid, self.price_paths[i], alpha=0.1, color='blue')
        
        # Plot average of all paths
        mean_path = np.mean(self.price_paths, axis=0)
        axes[0,0].plot(time_grid, mean_path, color='red', linewidth=2, label='Mean Path')
        axes[0,0].axhline(self.K, color='green', linestyle='--', linewidth=2, label=f'Strike (K={self.K})')
        axes[0,0].set_xlabel('Time (years)')
        axes[0,0].set_ylabel('Stock Price')
        axes[0,0].set_title(f'Sample Price Paths (showing {n_paths_to_plot} of {self.n_simulations})')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Average price distribution
        axes[0,1].hist(self.average_prices, bins=50, alpha=0.7, color='orange', density=True)
        axes[0,1].axvline(self.K, color='red', linestyle='--', linewidth=2, label=f'Strike (K={self.K})')
        axes[0,1].axvline(np.mean(self.average_prices), color='green', linestyle=':', linewidth=2,
                         label=f'Mean Avg Price ({np.mean(self.average_prices):.2f})')
        axes[0,1].set_xlabel('Average Price')
        axes[0,1].set_ylabel('Probability Density')
        axes[0,1].set_title(f'{self.average_type.title()} Average Price Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Payoff distribution
        axes[0,2].hist(self.payoffs, bins=50, alpha=0.7, color='purple', density=True)
        axes[0,2].axvline(np.mean(self.payoffs), color='red', linestyle='--', linewidth=2,
                         label=f'Mean Payoff ({np.mean(self.payoffs):.4f})')
        axes[0,2].set_xlabel('Option Payoff')
        axes[0,2].set_ylabel('Probability Density')
        axes[0,2].set_title(f'Asian {self.option_type.title()} Payoff Distribution')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Convergence analysis
        if self.convergence_history:
            n_sims, prices = zip(*self.convergence_history)
            axes[1,0].plot(n_sims, prices, 'b-', linewidth=2, label='Asian Option Price')
            axes[1,0].axhline(result.results['option_price'], color='red', linestyle='--', 
                             linewidth=2, label=f'Final Price ({result.results["option_price"]:.4f})')
            axes[1,0].set_xlabel('Number of Simulations')
            axes[1,0].set_ylabel('Option Price')
            axes[1,0].set_title('Price Convergence')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Average vs Final Price comparison
        final_prices = self.price_paths[:, -1]
        axes[1,1].scatter(self.average_prices, final_prices, alpha=0.3, s=1)
        axes[1,1].plot([min(self.average_prices), max(self.average_prices)], 
                      [min(self.average_prices), max(self.average_prices)], 
                      'r--', linewidth=2, label='Average = Final')
        axes[1,1].set_xlabel('Average Price')
        axes[1,1].set_ylabel('Final Price')
        axes[1,1].set_title('Average vs Final Price Relationship')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Results summary
        summary_text = f"""
        Asian Option Parameters:
        • Type: {self.option_type.title()} ({self.average_type})
        • S₀ (Initial Price): ${self.S0:.2f}
        • K (Strike): ${self.K:.2f}
        • T (Time to Exp): {self.T:.2f} years
        • r (Risk-free Rate): {self.r:.2%}
        • σ (Volatility): {self.sigma:.2%}
        • Time Steps: {self.n_time_steps}
        
        Pricing Results:
        • Asian Option Price: ${result.results['option_price']:.4f}
        • European Equivalent: ${result.results['european_equivalent_price']:.4f}
        • Asian Discount: {result.results['asian_discount']:.2f}%
        • Standard Error: ${result.results['standard_error']:.4f}
        • Volatility Reduction: {result.results['volatility_reduction']:.2f}%
        
        Statistics:
        • Mean Average Price: ${result.statistics['mean_average_price']:.2f}
        • Mean Final Price: ${result.statistics['mean_final_price']:.2f}
        • Prob ITM: {result.statistics['prob_itm']:.2%}
        • Max Payoff: ${result.statistics['max_payoff']:.4f}
        
        Performance:
        • Simulations: {self.n_simulations:,}
        • Execution Time: {result.execution_time:.2f}s
        • Paths/Second: {self.n_simulations/result.execution_time:,.0f}
        """
        
        axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                      fontsize=9, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        axes[1,2].set_xlim(0, 1)
        axes[1,2].set_ylim(0, 1)
        axes[1,2].axis('off')
        axes[1,2].set_title('Results Summary')
        
        plt.tight_layout()
        plt.show()
    
    def validate_parameters(self) -> List[str]:
        """Validate Asian option simulation parameters"""
        errors = []
        
        if self.S0 <= 0:
            errors.append("Initial stock price (S0) must be positive")
        if self.K <= 0:
            errors.append("Strike price (K) must be positive")
        if self.T <= 0:
            errors.append("Time to expiration (T) must be positive")
        if self.r < 0:
            errors.append("Risk-free rate (r) must be non-negative")
        if self.sigma <= 0:
            errors.append("Volatility (sigma) must be positive")
        if self.n_simulations <= 0:
            errors.append("Number of simulations must be positive")
        if self.n_time_steps <= 0:
            errors.append("Number of time steps must be positive")
        if self.average_type not in ['arithmetic', 'geometric']:
            errors.append("Average type must be 'arithmetic' or 'geometric'")
        if self.option_type not in ['call', 'put']:
            errors.append("Option type must be 'call' or 'put'")
        
        return errors
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'S0': {
                'type': 'float',
                'default': 100.0,
                'min': 0.01,
                'max': 1000.0,
                'description': 'Initial stock price'
            },
            'K': {
                'type': 'float',
                'default': 100.0,
                'min': 0.01,
                'max': 1000.0,
                'description': 'Strike price'
            },
            'T': {
                'type': 'float',
                'default': 1.0,
                'min': 0.01,
                'max': 10.0,
                'description': 'Time to expiration (years)'
            },
            'r': {
                'type': 'float',
                'default': 0.05,
                'min': 0.0,
                'max': 1.0,
                'description': 'Risk-free interest rate'
            },
            'sigma': {
                'type': 'float',
                'default': 0.2,
                'min': 0.01,
                'max': 2.0,
                'description': 'Volatility (annual)'
            },
            'n_simulations': {
                'type': 'int',
                'default': 100000,
                'min': 1000,
                'max': 1000000,
                'description': 'Number of Monte Carlo simulations'
            },
            'n_time_steps': {
                'type': 'int',
                'default': 252,
                'min': 10,
                'max': 1000,
                'description': 'Number of time steps'
            },
            'average_type': {
                'type': 'choice',
                'default': 'arithmetic',
                'choices': ['arithmetic', 'geometric'],
                'description': 'Type of average calculation'
            },
            'option_type': {
                'type': 'choice',
                'default': 'call',
                'choices': ['call', 'put'],
                'description': 'Type of option'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility (optional)'
            }
        }


