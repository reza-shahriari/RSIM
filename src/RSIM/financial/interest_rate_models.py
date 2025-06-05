import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Tuple, Dict
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class VasicekModel(BaseSimulation):
    """
    Vasicek Interest Rate Model Simulation
    
    The Vasicek model is a mathematical model describing the evolution of interest rates.
    It is a type of one-factor short-rate model as it describes interest rate movements
    as driven by only one source of market risk. The model can be used in the valuation
    of interest rate derivatives and has a mean-reverting property.
    
    Mathematical Background:
    -----------------------
    The Vasicek model specifies that the instantaneous interest rate follows the 
    stochastic differential equation:
    
    dr(t) = a(b - r(t))dt + σ dW(t)
    
    Where:
    - r(t) is the instantaneous interest rate at time t
    - a is the speed of reversion parameter (a > 0)
    - b is the long-term mean level that the interest rate reverts to
    - σ is the volatility parameter (σ > 0)
    - dW(t) is a Wiener process (Brownian motion)
    
    Analytical Solution:
    -------------------
    The model has a known analytical solution:
    r(t) = r(0)e^(-at) + b(1 - e^(-at)) + σ∫[0,t] e^(-a(t-s)) dW(s)
    
    Key Properties:
    --------------
    - Mean reversion: Interest rates tend to revert to long-term mean b
    - Gaussian distribution: Rates are normally distributed
    - Negative rates possible: A limitation of the model
    - Affine term structure: Bond prices have exponential-affine form
    - Analytical tractability: Closed-form solutions for bonds and options
    
    Applications:
    ------------
    - Bond pricing and yield curve modeling
    - Interest rate derivatives valuation
    - Risk management and scenario analysis
    - Asset-liability management
    - Pension fund modeling
    - Insurance company reserving
    - Central bank policy analysis
    
    Parameters:
    -----------
    initial_rate : float, default=0.03
        Initial interest rate (r0)
    mean_reversion_speed : float, default=0.5
        Speed of reversion parameter (a)
    long_term_mean : float, default=0.04
        Long-term mean level (b)
    volatility : float, default=0.01
        Volatility parameter (σ)
    time_horizon : float, default=1.0
        Total time horizon in years
    num_steps : int, default=252
        Number of time steps (252 = daily for 1 year)
    num_paths : int, default=1000
        Number of simulation paths
    random_seed : int, optional
        Seed for random number generator
    
    Attributes:
    -----------
    rate_paths : np.ndarray
        Simulated interest rate paths
    time_grid : np.ndarray
        Time grid for simulation
    bond_prices : dict
        Calculated bond prices for various maturities
    yield_curve : dict
        Simulated yield curves
    result : SimulationResult
        Complete simulation results
    
    Methods:
    --------
    configure(**kwargs) : bool
        Configure model parameters
    run(**kwargs) : SimulationResult
        Execute the interest rate simulation
    visualize(result=None, show_details=True) : None
        Create visualizations of interest rate paths and statistics
    calculate_bond_price(maturity, face_value=100) : float
        Calculate zero-coupon bond price analytically
    calculate_yield_curve(maturities) : dict
        Calculate yield curve for given maturities
    calculate_statistics() : dict
        Calculate comprehensive statistics of simulated paths
    validate_parameters() : List[str]
        Validate current parameters
    get_parameter_info() : dict
        Get parameter information for UI
    
    Examples:
    ---------
    >>> # Basic Vasicek simulation
    >>> vasicek = VasicekModel(initial_rate=0.02, long_term_mean=0.05)
    >>> result = vasicek.run()
    >>> print(f"Final mean rate: {result.results['final_mean_rate']:.4f}")
    
    >>> # High mean reversion scenario
    >>> fast_reversion = VasicekModel(mean_reversion_speed=2.0)
    >>> result = fast_reversion.run()
    >>> fast_reversion.visualize()
    
    >>> # Bond pricing example
    >>> vasicek = VasicekModel()
    >>> bond_price = vasicek.calculate_bond_price(maturity=5.0)
    >>> print(f"5-year bond price: ${bond_price:.2f}")
    """

    def __init__(self, initial_rate: float = 0.03, mean_reversion_speed: float = 0.5,
                 long_term_mean: float = 0.04, volatility: float = 0.01,
                 time_horizon: float = 1.0, num_steps: int = 252, num_paths: int = 1000,
                 random_seed: Optional[int] = None):
        super().__init__("Vasicek Interest Rate Model")
        
        # Model parameters
        self.initial_rate = initial_rate  # r0
        self.mean_reversion_speed = mean_reversion_speed  # a
        self.long_term_mean = long_term_mean  # b
        self.volatility = volatility  # σ
        
        # Simulation parameters
        self.time_horizon = time_horizon
        self.num_steps = num_steps
        self.num_paths = num_paths
        
        # Store in parameters dict
        self.parameters.update({
            'initial_rate': initial_rate,
            'mean_reversion_speed': mean_reversion_speed,
            'long_term_mean': long_term_mean,
            'volatility': volatility,
            'time_horizon': time_horizon,
            'num_steps': num_steps,
            'num_paths': num_paths,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Initialize arrays
        self.dt = time_horizon / num_steps
        self.time_grid = np.linspace(0, time_horizon, num_steps + 1)
        self.rate_paths = None
        self.bond_prices = {}
        self.yield_curve = {}
        
        self.is_configured = True
    
    def configure(self, **kwargs) -> bool:
        """Configure Vasicek model parameters"""
        # Update parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.parameters[key] = value
        
        # Recalculate derived parameters
        self.dt = self.time_horizon / self.num_steps
        self.time_grid = np.linspace(0, self.time_horizon, self.num_steps + 1)
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute Vasicek model simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Initialize rate paths array
        self.rate_paths = np.zeros((self.num_paths, self.num_steps + 1))
        self.rate_paths[:, 0] = self.initial_rate
        
        # Generate random shocks
        dW = np.random.normal(0, np.sqrt(self.dt), (self.num_paths, self.num_steps))
        
        # Simulate paths using Euler-Maruyama scheme
        for i in range(self.num_steps):
            drift = self.mean_reversion_speed * (self.long_term_mean - self.rate_paths[:, i])
            diffusion = self.volatility * dW[:, i]
            self.rate_paths[:, i + 1] = self.rate_paths[:, i] + drift * self.dt + diffusion
        
        # Calculate statistics
        final_rates = self.rate_paths[:, -1]
        mean_final_rate = np.mean(final_rates)
        std_final_rate = np.std(final_rates)
        
        # Path statistics
        mean_path = np.mean(self.rate_paths, axis=0)
        std_path = np.std(self.rate_paths, axis=0)
        percentile_5 = np.percentile(self.rate_paths, 5, axis=0)
        percentile_95 = np.percentile(self.rate_paths, 95, axis=0)
        
        # Calculate theoretical moments
        theoretical_mean = self._theoretical_mean(self.time_horizon)
        theoretical_variance = self._theoretical_variance(self.time_horizon)
        
        # Negative rate analysis
        negative_rate_paths = np.sum(np.any(self.rate_paths < 0, axis=1))
        negative_rate_probability = negative_rate_paths / self.num_paths
        
        # Calculate bond prices for standard maturities
        standard_maturities = [0.25, 0.5, 1, 2, 5, 10]
        bond_prices = {}
        yields = {}
        
        for maturity in standard_maturities:
            if maturity <= self.time_horizon:
                price = self.calculate_bond_price(maturity)
                bond_prices[f'{maturity}Y'] = price
                yields[f'{maturity}Y'] = -np.log(price/100) / maturity
        
        self.bond_prices = bond_prices
        
        execution_time = time.time() - start_time
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'final_mean_rate': mean_final_rate,
                'final_std_rate': std_final_rate,
                'theoretical_mean': theoretical_mean,
                'theoretical_std': np.sqrt(theoretical_variance),
                'negative_rate_probability': negative_rate_probability,
                'mean_path': mean_path.tolist(),
                'std_path': std_path.tolist(),
                'percentile_5': percentile_5.tolist(),
                'percentile_95': percentile_95.tolist(),
                'bond_prices': bond_prices,
                'yields': yields
            },
            statistics={
                'min_rate_overall': np.min(self.rate_paths),
                'max_rate_overall': np.max(self.rate_paths),
                'mean_rate_overall': np.mean(self.rate_paths),
                'std_rate_overall': np.std(self.rate_paths),
                'paths_hit_zero': np.sum(np.any(self.rate_paths <= 0, axis=1)),
                'average_rate_level': np.mean(self.rate_paths[:, -1]),
                'convergence_to_mean': abs(mean_final_rate - self.long_term_mean)
            },
            execution_time=execution_time,
            convergence_data=[(i, mean_path[i]) for i in range(0, len(mean_path), max(1, len(mean_path)//100))]
        )
        
        self.result = result
        return result
    
    def _theoretical_mean(self, t: float) -> float:
        """Calculate theoretical mean at time t"""
        return self.long_term_mean + (self.initial_rate - self.long_term_mean) * np.exp(-self.mean_reversion_speed * t)
    
    def _theoretical_variance(self, t: float) -> float:
        """Calculate theoretical variance at time t"""
        if self.mean_reversion_speed == 0:
            return self.volatility**2 * t
        return (self.volatility**2 / (2 * self.mean_reversion_speed)) * (1 - np.exp(-2 * self.mean_reversion_speed * t))
    
    def calculate_bond_price(self, maturity: float, face_value: float = 100) -> float:
        """Calculate zero-coupon bond price analytically"""
        if maturity <= 0:
            return face_value
        
        a, b, sigma, r0 = self.mean_reversion_speed, self.long_term_mean, self.volatility, self.initial_rate
        
        if a == 0:
            # Special case when a = 0
            B = maturity
            A = np.exp(-b * maturity + 0.5 * sigma**2 * maturity**3 / 3)
        else:
            B = (1 - np.exp(-a * maturity)) / a
            A = np.exp((b - sigma**2 / (2 * a**2)) * (B - maturity) - (sigma**2 * B**2) / (4 * a))
        
        return face_value * A * np.exp(-B * r0)
    
    def calculate_yield_curve(self, maturities: List[float]) -> dict:
        """Calculate yield curve for given maturities"""
        yields = {}
        prices = {}
        
        for maturity in maturities:
            price = self.calculate_bond_price(maturity)
            yield_rate = -np.log(price/100) / maturity if maturity > 0 else self.initial_rate
            
            yields[maturity] = yield_rate
            prices[maturity] = price
        
        return {'yields': yields, 'prices': prices, 'maturities': maturities}
    
    def visualize(self, result: Optional[SimulationResult] = None, show_details: bool = True) -> None:
        """Visualize Vasicek model simulation results"""
        if result is None:
            result = self.result
        
        if result is None or self.rate_paths is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Sample interest rate paths
        sample_paths = min(50, self.num_paths)
        for i in range(sample_paths):
            axes[0,0].plot(self.time_grid, self.rate_paths[i, :], 'b-', alpha=0.1, linewidth=0.5)
        
        # Plot mean path and confidence bands
        mean_path = np.array(result.results['mean_path'])
        percentile_5 = np.array(result.results['percentile_5'])
        percentile_95 = np.array(result.results['percentile_95'])
        
        axes[0,0].plot(self.time_grid, mean_path, 'r-', linewidth=2, label='Mean Path')
        axes[0,0].fill_between(self.time_grid, percentile_5, percentile_95, alpha=0.3, color='red', label='90% Confidence')
        axes[0,0].axhline(y=self.long_term_mean, color='g', linestyle='--', linewidth=2, label=f'Long-term Mean ({self.long_term_mean:.3f})')
        axes[0,0].axhline(y=self.initial_rate, color='orange', linestyle=':', linewidth=2, label=f'Initial Rate ({self.initial_rate:.3f})')
        axes[0,0].set_xlabel('Time (Years)')
        axes[0,0].set_ylabel('Interest Rate')
        axes[0,0].set_title('Vasicek Interest Rate Paths')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Final rate distribution
        final_rates = self.rate_paths[:, -1]
        axes[0,1].hist(final_rates, bins=50, alpha=0.7, color='blue', density=True, label='Simulated')
        
        # Overlay theoretical distribution
        x_theory = np.linspace(np.min(final_rates), np.max(final_rates), 100)
        theoretical_mean = result.results['theoretical_mean']
        theoretical_std = result.results['theoretical_std']
        y_theory = (1/np.sqrt(2*np.pi*theoretical_std**2)) * np.exp(-0.5*((x_theory - theoretical_mean)/theoretical_std)**2)
        axes[0,1].plot(x_theory, y_theory, 'r-', linewidth=2, label='Theoretical')
        
        axes[0,1].axvline(x=theoretical_mean, color='red', linestyle='--', alpha=0.7, label=f'Theoretical Mean ({theoretical_mean:.4f})')
        axes[0,1].axvline(x=np.mean(final_rates), color='blue', linestyle='--', alpha=0.7, label=f'Simulated Mean ({np.mean(final_rates):.4f})')
        axes[0,1].set_xlabel('Final Interest Rate')
        axes[0,1].set_ylabel('Density')
        axes[0,1].set_title('Final Rate Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Yield curve
        if result.results['bond_prices']:
            maturities = []
            yields = []
            for key, yield_val in result.results['yields'].items():
                maturity = float(key.replace('Y', ''))
                maturities.append(maturity)
                yields.append(yield_val)
            
            # Sort by maturity
            sorted_data = sorted(zip(maturities, yields))
            maturities, yields = zip(*sorted_data)
            
            axes[0,2].plot(maturities, yields, 'bo-', linewidth=2, markersize=6)
            axes[0,2].set_xlabel('Maturity (Years)')
            axes[0,2].set_ylabel('Yield')
            axes[0,2].set_title('Yield Curve')
            axes[0,2].grid(True, alpha=0.3)
        else:
            axes[0,2].text(0.5, 0.5, 'No yield curve data\navailable', 
                          transform=axes[0,2].transAxes, ha='center', va='center')
            axes[0,2].set_title('Yield Curve')
        
        # Plot 4: Rate evolution statistics
        time_points = self.time_grid[::max(1, len(self.time_grid)//20)]  # Sample time points
        rate_means = [np.mean(self.rate_paths[:, i]) for i in range(0, len(self.time_grid), max(1, len(self.time_grid)//20))]
        rate_stds = [np.std(self.rate_paths[:, i]) for i in range(0, len(self.time_grid), max(1, len(self.time_grid)//20))]
        
        ax1 = axes[1,0]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(time_points, rate_means, 'b-', linewidth=2, label='Mean Rate')
        line2 = ax2.plot(time_points, rate_stds, 'r-', linewidth=2, label='Std Dev')
        
        ax1.set_xlabel('Time (Years)')
        ax1.set_ylabel('Mean Rate', color='blue')
        ax2.set_ylabel('Standard Deviation', color='red')
        ax1.set_title('Rate Statistics Over Time')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 5: Convergence analysis
        theoretical_path = [self._theoretical_mean(t) for t in self.time_grid]
        simulated_mean = np.array(result.results['mean_path'])
        convergence_error = np.abs(simulated_mean - theoretical_path)
        
        axes[1,1].plot(self.time_grid, convergence_error, 'g-', linewidth=2)
        axes[1,1].set_xlabel('Time (Years)')
        axes[1,1].set_ylabel('|Simulated - Theoretical| Mean')
        axes[1,1].set_title('Convergence to Theoretical Mean')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_yscale('log')
        
        # Plot 6: Model summary
        summary_text = f"""
        Vasicek Model Parameters:
        • Initial Rate (r₀): {self.initial_rate:.4f}
        • Mean Reversion Speed (a): {self.mean_reversion_speed:.4f}
        • Long-term Mean (b): {self.long_term_mean:.4f}
        • Volatility (σ): {self.volatility:.4f}
        
        Simulation Results:
        • Final Mean Rate: {result.results['final_mean_rate']:.4f}
        • Final Std Rate: {result.results['final_std_rate']:.4f}
        • Theoretical Mean: {result.results['theoretical_mean']:.4f}
        • Theoretical Std: {result.results['theoretical_std']:.4f}
        
        Risk Analysis:
        • Negative Rate Probability: {result.results['negative_rate_probability']:.2%}
        • Min Rate Observed: {result.statistics['min_rate_overall']:.4f}
        • Max Rate Observed: {result.statistics['max_rate_overall']:.4f}
        
        Model Performance:
        • Convergence Error: {result.statistics['convergence_to_mean']:.6f}
        • Paths Simulated: {self.num_paths:,}
        • Time Steps: {self.num_steps:,}
        • Execution Time: {result.execution_time:.3f}s
        """
        
        axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                      fontsize=9, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        axes[1,2].set_xlim(0, 1)
        axes[1,2].set_ylim(0, 1)
        axes[1,2].axis('off')
        axes[1,2].set_title('Model Summary')
        
        plt.tight_layout()
        plt.show()
    
    def validate_parameters(self) -> List[str]:
        """Validate Vasicek model parameters"""
        errors = []
        
        if self.mean_reversion_speed < 0:
            errors.append("Mean reversion speed (a) must be non-negative")
        
        if self.volatility <= 0:
            errors.append("Volatility (σ) must be positive")
        
        if self.time_horizon <= 0:
            errors.append("Time horizon must be positive")
        
        if self.num_steps <= 0:
            errors.append("Number of steps must be positive")
        
        if self.num_paths <= 0:
            errors.append("Number of paths must be positive")
        
        # Check for reasonable parameter ranges
        if self.volatility > 0.5:
            errors.append("Volatility seems unreasonably high (>50%)")
        
        if abs(self.initial_rate) > 1:
            errors.append("Initial rate seems unreasonable (>100%)")
        
        if abs(self.long_term_mean) > 1:
            errors.append("Long-term mean seems unreasonable (>100%)")
        
        return errors
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'initial_rate': {
                'type': 'float',
                'default': 0.03,
                'min': -0.1,
                'max': 0.2,
                'description': 'Initial interest rate (r₀)'
            },
            'mean_reversion_speed': {
                'type': 'float',
                'default': 0.5,
                'min': 0.0,
                'max': 5.0,
                'description': 'Mean reversion speed (a)'
            },
            'long_term_mean': {
                'type': 'float',
                'default': 0.04,
                'min': -0.05,
                'max': 0.2,
                'description': 'Long-term mean level (b)'
            },
            'volatility': {
                'type': 'float',
                'default': 0.01,
                'min': 0.001,
                'max': 0.1,
                'description': 'Volatility parameter (σ)'
            },
            'time_horizon': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 30.0,
                'description': 'Time horizon (years)'
            },
            'num_steps': {
                'type': 'int',
                'default': 252,
                'min': 50,
                'max': 5000,
                'description': 'Number of time steps'
            },
            'num_paths': {
                'type': 'int',
                'default': 1000,
                'min': 100,
                'max': 10000,
                'description': 'Number of simulation paths'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility (optional)'
            }
        }


class CIRModel(BaseSimulation):
    """
    Cox-Ingersoll-Ross (CIR) Interest Rate Model Simulation
    
    The CIR model is a mathematical model describing the evolution of interest rates.
    It is an extension of the Vasicek model and ensures that interest rates are always
    non-negative. The model exhibits mean reversion and has a square-root diffusion term
    that makes volatility proportional to the square root of the rate level.
    
    Mathematical Background:
    -----------------------
    The CIR model specifies that the instantaneous interest rate follows the 
    stochastic differential equation:
    
    dr(t) = a(b - r(t))dt + σ√r(t) dW(t)
    
    Where:
    - r(t) is the instantaneous interest rate at time t
    - a is the speed of reversion parameter (a > 0)
    - b is the long-term mean level that the interest rate reverts to
    - σ is the volatility parameter (σ > 0)
    - dW(t) is a Wiener process (Brownian motion)
    
    Feller Condition:
    ----------------
    For the process to remain strictly positive, the Feller condition must hold:
    2ab ≥ σ²
    
    If this condition is violated, the process can reach zero but will be reflected back.
    
    Key Properties:
    --------------
    - Mean reversion: Interest rates tend to revert to long-term mean b
    - Non-negative rates: The square-root diffusion ensures r(t) ≥ 0
    - Stochastic volatility: Volatility increases with the rate level
    - Chi-squared distribution: Rates follow scaled non-central chi-squared distribution
    - Affine term structure: Bond prices have exponential-affine form
    
    Applications:
    ------------
    - Bond pricing and yield curve modeling
    - Interest rate derivatives valuation
    - Credit risk modeling (as intensity process)
    - Commodity price modeling
    - Volatility modeling in equity markets
    - Central bank policy analysis
    - Risk management applications
    
    Parameters:
    -----------
    initial_rate : float, default=0.03
        Initial interest rate (r0)
    mean_reversion_speed : float, default=0.5
        Speed of reversion parameter (a)
    long_term_mean : float, default=0.04
        Long-term mean level (b)
    volatility : float, default=0.1
        Volatility parameter (σ)
    time_horizon : float, default=1.0
        Total time horizon in years
    num_steps : int, default=252
        Number of time steps (252 = daily for 1 year)
    num_paths : int, default=1000
        Number of simulation paths
    scheme : str, default='euler'
        Discretization scheme ('euler', 'milstein', 'exact')
    random_seed : int, optional
        Seed for random number generator
    
    Attributes:
    -----------
    rate_paths : np.ndarray
        Simulated interest rate paths
    time_grid : np.ndarray
        Time grid for simulation
    feller_condition : bool
        Whether Feller condition is satisfied
    bond_prices : dict
        Calculated bond prices for various maturities
    result : SimulationResult
        Complete simulation results
    
    Methods:
    --------
    configure(**kwargs) : bool
        Configure model parameters
    run(**kwargs) : SimulationResult
        Execute the interest rate simulation
    visualize(result=None, show_details=True) : None
        Create visualizations of interest rate paths and statistics
    calculate_bond_price(maturity, face_value=100) : float
        Calculate zero-coupon bond price analytically
    check_feller_condition() : bool
        Check if Feller condition is satisfied
    validate_parameters() : List[str]
        Validate current parameters
    get_parameter_info() : dict
        Get parameter information for UI
    
    Examples:
    ---------
    >>> # Basic CIR simulation
    >>> cir = CIRModel(initial_rate=0.02, long_term_mean=0.05)
    >>> result = cir.run()
    >>> print(f"Final mean rate: {result.results['final_mean_rate']:.4f}")
    
    >>> # High volatility scenario
    >>> high_vol = CIRModel(volatility=0.2)
    >>> result = high_vol.run()
    >>> high_vol.visualize()
    
    >>> # Check Feller condition
    >>> cir = CIRModel()
    >>> feller_ok = cir.check_feller_condition()
    >>> print(f"Feller condition satisfied: {feller_ok}")
    """

    def __init__(self, initial_rate: float = 0.03, mean_reversion_speed: float = 0.5,
                 long_term_mean: float = 0.04, volatility: float = 0.1,
                 time_horizon: float = 1.0, num_steps: int = 252, num_paths: int = 1000,
                 scheme: str = 'euler', random_seed: Optional[int] = None):
        super().__init__("CIR Interest Rate Model")
        
        # Model parameters
        self.initial_rate = initial_rate  # r0
        self.mean_reversion_speed = mean_reversion_speed  # a
        self.long_term_mean = long_term_mean  # b
        self.volatility = volatility  # σ
        
        # Simulation parameters
        self.time_horizon = time_horizon
        self.num_steps = num_steps
        self.num_paths = num_paths
        self.scheme = scheme
        
        # Store in parameters dict
        self.parameters.update({
            'initial_rate': initial_rate,
            'mean_reversion_speed': mean_reversion_speed,
            'long_term_mean': long_term_mean,
            'volatility': volatility,
            'time_horizon': time_horizon,
            'num_steps': num_steps,
            'num_paths': num_paths,
            'scheme': scheme,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Initialize arrays
        self.dt = time_horizon / num_steps
        self.time_grid = np.linspace(0, time_horizon, num_steps + 1)
        self.rate_paths = None
        self.bond_prices = {}
        self.feller_condition = self.check_feller_condition()
        
        self.is_configured = True
    
    def configure(self, **kwargs) -> bool:
        """Configure CIR model parameters"""
        # Update parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.parameters[key] = value
        
        # Recalculate derived parameters
        self.dt = self.time_horizon / self.num_steps
        self.time_grid = np.linspace(0, self.time_horizon, self.num_steps + 1)
        self.feller_condition = self.check_feller_condition()
        
        self.is_configured = True
        return True
    
    def check_feller_condition(self) -> bool:
        """Check if Feller condition (2ab ≥ σ²) is satisfied"""
        return 2 * self.mean_reversion_speed * self.long_term_mean >= self.volatility**2
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute CIR model simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Initialize rate paths array
        self.rate_paths = np.zeros((self.num_paths, self.num_steps + 1))
        self.rate_paths[:, 0] = self.initial_rate
        
        # Generate random shocks
        dW = np.random.normal(0, np.sqrt(self.dt), (self.num_paths, self.num_steps))
        
        # Simulate paths using chosen discretization scheme
        if self.scheme.lower() == 'euler':
            self._simulate_euler(dW)
        elif self.scheme.lower() == 'milstein':
            self._simulate_milstein(dW)
        elif self.scheme.lower() == 'exact':
            self._simulate_exact()
        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")
        
        # Calculate statistics
        final_rates = self.rate_paths[:, -1]
        mean_final_rate = np.mean(final_rates)
        std_final_rate = np.std(final_rates)
        
        # Path statistics
        mean_path = np.mean(self.rate_paths, axis=0)
        std_path = np.std(self.rate_paths, axis=0)
        percentile_5 = np.percentile(self.rate_paths, 5, axis=0)
        percentile_95 = np.percentile(self.rate_paths, 95, axis=0)
        
        # Zero rate analysis (should be rare if Feller condition holds)
        zero_rate_paths = np.sum(np.any(self.rate_paths <= 1e-10, axis=1))
        zero_rate_probability = zero_rate_paths / self.num_paths
        
        # Calculate bond prices for standard maturities
        standard_maturities = [0.25, 0.5, 1, 2, 5, 10]
        bond_prices = {}
        yields = {}
        
        for maturity in standard_maturities:
            if maturity <= self.time_horizon:
                price = self.calculate_bond_price(maturity)
                bond_prices[f'{maturity}Y'] = price
                yields[f'{maturity}Y'] = -np.log(price/100) / maturity
        
        self.bond_prices = bond_prices
        
        # Volatility analysis
        instantaneous_volatilities = self.volatility * np.sqrt(self.rate_paths)
        mean_volatility = np.mean(instantaneous_volatilities)
        
        execution_time = time.time() - start_time
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'final_mean_rate': mean_final_rate,
                'final_std_rate': std_final_rate,
                'zero_rate_probability': zero_rate_probability,
                'feller_condition_satisfied': self.feller_condition,
                'mean_path': mean_path.tolist(),
                'std_path': std_path.tolist(),
                'percentile_5': percentile_5.tolist(),
                'percentile_95': percentile_95.tolist(),
                'bond_prices': bond_prices,
                'yields': yields,
                'mean_instantaneous_volatility': mean_volatility
            },
            statistics={
                'min_rate_overall': np.min(self.rate_paths),
                'max_rate_overall': np.max(self.rate_paths),
                'mean_rate_overall': np.mean(self.rate_paths),
                'std_rate_overall': np.std(self.rate_paths),
                'paths_near_zero': zero_rate_paths,
                'average_rate_level': np.mean(self.rate_paths[:, -1]),
                'volatility_of_volatility': np.std(instantaneous_volatilities),
                'feller_parameter': 2 * self.mean_reversion_speed * self.long_term_mean / (self.volatility**2)
            },
            execution_time=execution_time,
            convergence_data=[(i, mean_path[i]) for i in range(0, len(mean_path), max(1, len(mean_path)//100))]
        )
        
        self.result = result
        return result
    
    def _simulate_euler(self, dW: np.ndarray) -> None:
        """Simulate using Euler-Maruyama scheme"""
        for i in range(self.num_steps):
            current_rates = self.rate_paths[:, i]
            drift = self.mean_reversion_speed * (self.long_term_mean - current_rates)
            diffusion = self.volatility * np.sqrt(np.maximum(current_rates, 0)) * dW[:, i]
            
            self.rate_paths[:, i + 1] = np.maximum(
                current_rates + drift * self.dt + diffusion, 0
            )
    
    def _simulate_milstein(self, dW: np.ndarray) -> None:
        """Simulate using Milstein scheme (higher order correction)"""
        for i in range(self.num_steps):
            current_rates = self.rate_paths[:, i]
            sqrt_rates = np.sqrt(np.maximum(current_rates, 0))
            
            drift = self.mean_reversion_speed * (self.long_term_mean - current_rates)
            diffusion = self.volatility * sqrt_rates * dW[:, i]
            
            # Milstein correction term
            milstein_correction = 0.25 * self.volatility**2 * (dW[:, i]**2 - self.dt)
            
            self.rate_paths[:, i + 1] = np.maximum(
                current_rates + drift * self.dt + diffusion + milstein_correction, 0
            )
    
    def _simulate_exact(self) -> None:
        """Simulate using exact simulation (when possible)"""
        # For CIR, exact simulation involves non-central chi-squared distribution
        # This is more complex and requires special functions
        try:
            from scipy.stats import ncx2
        except ImportError:
            print("Scipy required for exact simulation. Falling back to Euler scheme.")
            dW = np.random.normal(0, np.sqrt(self.dt), (self.num_paths, self.num_steps))
            self._simulate_euler(dW)
            return
        
        for i in range(self.num_steps):
            current_rates = self.rate_paths[:, i]
            
            # Parameters for non-central chi-squared distribution
            c = self.volatility**2 * (1 - np.exp(-self.mean_reversion_speed * self.dt)) / (4 * self.mean_reversion_speed)
            q = (4 * self.mean_reversion_speed * self.long_term_mean / self.volatility**2) - 1
            lambda_param = current_rates * np.exp(-self.mean_reversion_speed * self.dt) / c
            
            # Generate from non-central chi-squared
            for j in range(self.num_paths):
                if q > 0:
                    self.rate_paths[j, i + 1] = c * ncx2.rvs(df=q+1, nc=lambda_param[j])
                else:
                    # Fall back to Euler for edge cases
                    drift = self.mean_reversion_speed * (self.long_term_mean - current_rates[j])
                    diffusion = self.volatility * np.sqrt(max(current_rates[j], 0)) * np.random.normal(0, np.sqrt(self.dt))
                    self.rate_paths[j, i + 1] = max(current_rates[j] + drift * self.dt + diffusion, 0)
    
    def calculate_bond_price(self, maturity: float, face_value: float = 100) -> float:
        """Calculate zero-coupon bond price analytically using CIR formula"""
        if maturity <= 0:
            return face_value
        
        a, b, sigma, r0 = self.mean_reversion_speed, self.long_term_mean, self.volatility, self.initial_rate
        
        # CIR bond pricing formula
        gamma = np.sqrt(a**2 + 2*sigma**2)
        
        B = (2*(np.exp(gamma*maturity) - 1)) / ((gamma + a)*(np.exp(gamma*maturity) - 1) + 2*gamma)
        
        A = ((2*gamma*np.exp((a + gamma)*maturity/2)) / 
             ((gamma + a)*(np.exp(gamma*maturity) - 1) + 2*gamma))**(2*a*b/sigma**2)
        
        return face_value * A * np.exp(-B * r0)
    
    def visualize(self, result: Optional[SimulationResult] = None, show_details: bool = True) -> None:
        """Visualize CIR model simulation results"""
        if result is None:
            result = self.result
        
        if result is None or self.rate_paths is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot 1: Sample interest rate paths
        sample_paths = min(50, self.num_paths)
        for i in range(sample_paths):
            axes[0,0].plot(self.time_grid, self.rate_paths[i, :], 'b-', alpha=0.1, linewidth=0.5)
        
        # Plot mean path and confidence bands
        mean_path = np.array(result.results['mean_path'])
        percentile_5 = np.array(result.results['percentile_5'])
        percentile_95 = np.array(result.results['percentile_95'])
        
        axes[0,0].plot(self.time_grid, mean_path, 'r-', linewidth=2, label='Mean Path')
        axes[0,0].fill_between(self.time_grid, percentile_5, percentile_95, alpha=0.3, color='red', label='90% Confidence')
        axes[0,0].axhline(y=self.long_term_mean, color='g', linestyle='--', linewidth=2, label=f'Long-term Mean ({self.long_term_mean:.3f})')
        axes[0,0].axhline(y=self.initial_rate, color='orange', linestyle=':', linewidth=2, label=f'Initial Rate ({self.initial_rate:.3f})')
        axes[0,0].set_xlabel('Time (Years)')
        axes[0,0].set_ylabel('Interest Rate')
        axes[0,0].set_title('CIR Interest Rate Paths')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_ylim(bottom=0)  # Ensure non-negative rates are shown
        
        # Plot 2: Final rate distribution
        final_rates = self.rate_paths[:, -1]
        axes[0,1].hist(final_rates, bins=50, alpha=0.7, color='blue', density=True)
        axes[0,1].axvline(x=np.mean(final_rates), color='red', linestyle='--', alpha=0.7, 
                         label=f'Mean ({np.mean(final_rates):.4f})')
        axes[0,1].axvline(x=self.long_term_mean, color='green', linestyle='--', alpha=0.7, 
                         label=f'Long-term Mean ({self.long_term_mean:.4f})')
        axes[0,1].set_xlabel('Final Interest Rate')
        axes[0,1].set_ylabel('Density')
        axes[0,1].set_title('Final Rate Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Instantaneous volatility over time
        volatility_paths = self.volatility * np.sqrt(self.rate_paths)
        mean_vol_path = np.mean(volatility_paths, axis=0)
        std_vol_path = np.std(volatility_paths, axis=0)
        
        axes[1,0].plot(self.time_grid, mean_vol_path, 'purple', linewidth=2, label='Mean Volatility')
        axes[1,0].fill_between(self.time_grid,
                                      mean_vol_path - std_vol_path,
                              mean_vol_path + std_vol_path,
                              alpha=0.3, color='purple', label='±1 Std Dev')
        axes[1,0].set_xlabel('Time (Years)')
        axes[1,0].set_ylabel('Instantaneous Volatility')
        axes[1,0].set_title('Stochastic Volatility Evolution')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Rate vs Volatility scatter
        all_rates = self.rate_paths.flatten()
        all_vols = (self.volatility * np.sqrt(self.rate_paths)).flatten()
        
        # Sample for plotting (to avoid overcrowding)
        sample_size = min(5000, len(all_rates))
        sample_indices = np.random.choice(len(all_rates), sample_size, replace=False)
        
        axes[1,1].scatter(all_rates[sample_indices], all_vols[sample_indices], 
                         alpha=0.3, s=1, color='blue')
        
        # Theoretical relationship
        rate_range = np.linspace(0, np.max(all_rates), 100)
        theoretical_vol = self.volatility * np.sqrt(rate_range)
        axes[1,1].plot(rate_range, theoretical_vol, 'r-', linewidth=2, 
                      label=f'σ√r (σ={self.volatility:.3f})')
        
        axes[1,1].set_xlabel('Interest Rate')
        axes[1,1].set_ylabel('Instantaneous Volatility')
        axes[1,1].set_title('Rate-Volatility Relationship')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 5: Yield curve comparison
        if result.results['bond_prices']:
            maturities = []
            yields = []
            for key, yield_val in result.results['yields'].items():
                maturity = float(key.replace('Y', ''))
                maturities.append(maturity)
                yields.append(yield_val)
            
            # Sort by maturity
            sorted_data = sorted(zip(maturities, yields))
            maturities, yields = zip(*sorted_data)
            
            axes[2,0].plot(maturities, yields, 'bo-', linewidth=2, markersize=6, label='CIR Yields')
            
            # Compare with flat yield curve at current rate
            flat_yields = [self.initial_rate] * len(maturities)
            axes[2,0].plot(maturities, flat_yields, 'r--', linewidth=2, 
                          label=f'Flat at {self.initial_rate:.3f}')
            
            axes[2,0].set_xlabel('Maturity (Years)')
            axes[2,0].set_ylabel('Yield')
            axes[2,0].set_title('Yield Curve')
            axes[2,0].legend()
            axes[2,0].grid(True, alpha=0.3)
        else:
            axes[2,0].text(0.5, 0.5, 'No yield curve data\navailable', 
                          transform=axes[2,0].transAxes, ha='center', va='center')
            axes[2,0].set_title('Yield Curve')
        
        # Plot 6: Model summary and diagnostics
        feller_param = result.statistics['feller_parameter']
        feller_status = "✓ Satisfied" if self.feller_condition else "✗ Violated"
        
        summary_text = f"""
        CIR Model Parameters:
        • Initial Rate (r₀): {self.initial_rate:.4f}
        • Mean Reversion Speed (a): {self.mean_reversion_speed:.4f}
        • Long-term Mean (b): {self.long_term_mean:.4f}
        • Volatility (σ): {self.volatility:.4f}
        • Discretization: {self.scheme.title()}
        
        Feller Condition Analysis:
        • Condition: 2ab ≥ σ² 
        • Value: 2×{self.mean_reversion_speed:.3f}×{self.long_term_mean:.3f} = {2*self.mean_reversion_speed*self.long_term_mean:.4f}
        • σ² = {self.volatility**2:.4f}
        • Status: {feller_status}
        • Feller Parameter: {feller_param:.2f}
        
        Simulation Results:
        • Final Mean Rate: {result.results['final_mean_rate']:.4f}
        • Final Std Rate: {result.results['final_std_rate']:.4f}
        • Zero Rate Probability: {result.results['zero_rate_probability']:.4%}
        • Mean Inst. Volatility: {result.results['mean_instantaneous_volatility']:.4f}
        
        Risk Analysis:
        • Min Rate Observed: {result.statistics['min_rate_overall']:.6f}
        • Max Rate Observed: {result.statistics['max_rate_overall']:.4f}
        • Paths Near Zero: {result.statistics['paths_near_zero']}
        
        Performance:
        • Paths Simulated: {self.num_paths:,}
        • Time Steps: {self.num_steps:,}
        • Execution Time: {result.execution_time:.3f}s
        """
        
        axes[2,1].text(0.05, 0.95, summary_text, transform=axes[2,1].transAxes,
                      fontsize=8, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        axes[2,1].set_xlim(0, 1)
        axes[2,1].set_ylim(0, 1)
        axes[2,1].axis('off')
        axes[2,1].set_title('Model Summary & Diagnostics')
        
        plt.tight_layout()
        plt.show()
    
    def calculate_statistics(self) -> dict:
        """Calculate comprehensive statistics of simulated paths"""
        if self.rate_paths is None:
            return {}
        
        # Basic statistics
        stats = {
            'mean_rate': np.mean(self.rate_paths),
            'std_rate': np.std(self.rate_paths),
            'min_rate': np.min(self.rate_paths),
            'max_rate': np.max(self.rate_paths),
            'median_rate': np.median(self.rate_paths)
        }
        
        # Time-varying statistics
        stats['mean_path'] = np.mean(self.rate_paths, axis=0)
        stats['std_path'] = np.std(self.rate_paths, axis=0)
        stats['percentiles'] = {
            '5th': np.percentile(self.rate_paths, 5, axis=0),
            '25th': np.percentile(self.rate_paths, 25, axis=0),
            '75th': np.percentile(self.rate_paths, 75, axis=0),
            '95th': np.percentile(self.rate_paths, 95, axis=0)
        }
        
        # Volatility statistics
        vol_paths = self.volatility * np.sqrt(self.rate_paths)
        stats['volatility_stats'] = {
            'mean_vol': np.mean(vol_paths),
            'std_vol': np.std(vol_paths),
            'min_vol': np.min(vol_paths),
            'max_vol': np.max(vol_paths)
        }
        
        # Zero-touching analysis
        zero_threshold = 1e-6
        paths_near_zero = np.sum(np.any(self.rate_paths <= zero_threshold, axis=1))
        stats['zero_analysis'] = {
            'paths_near_zero': paths_near_zero,
            'zero_probability': paths_near_zero / self.num_paths,
            'min_rate_ever': np.min(self.rate_paths)
        }
        
        return stats
    
    def validate_parameters(self) -> List[str]:
        """Validate CIR model parameters"""
        errors = []
        
        if self.initial_rate < 0:
            errors.append("Initial rate must be non-negative")
        
        if self.mean_reversion_speed < 0:
            errors.append("Mean reversion speed (a) must be non-negative")
        
        if self.long_term_mean < 0:
            errors.append("Long-term mean (b) must be non-negative")
        
        if self.volatility <= 0:
            errors.append("Volatility (σ) must be positive")
        
        if self.time_horizon <= 0:
            errors.append("Time horizon must be positive")
        
        if self.num_steps <= 0:
            errors.append("Number of steps must be positive")
        
        if self.num_paths <= 0:
            errors.append("Number of paths must be positive")
        
        # Check Feller condition
        if not self.check_feller_condition():
            errors.append("Feller condition (2ab ≥ σ²) is violated - rates may hit zero")
        
        # Check for reasonable parameter ranges
        if self.volatility > 1.0:
            errors.append("Volatility seems unreasonably high (>100%)")
        
        if self.initial_rate > 1:
            errors.append("Initial rate seems unreasonable (>100%)")
        
        if self.long_term_mean > 1:
            errors.append("Long-term mean seems unreasonable (>100%)")
        
        # Check discretization scheme
        valid_schemes = ['euler', 'milstein', 'exact']
        if self.scheme.lower() not in valid_schemes:
            errors.append(f"Scheme must be one of: {valid_schemes}")
        
        return errors
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'initial_rate': {
                'type': 'float',
                'default': 0.03,
                'min': 0.0,
                'max': 0.2,
                'description': 'Initial interest rate (r₀)'
            },
            'mean_reversion_speed': {
                'type': 'float',
                'default': 0.5,
                'min': 0.0,
                'max': 5.0,
                'description': 'Mean reversion speed (a)'
            },
            'long_term_mean': {
                'type': 'float',
                'default': 0.04,
                'min': 0.0,
                'max': 0.2,
                'description': 'Long-term mean level (b)'
            },
            'volatility': {
                'type': 'float',
                'default': 0.1,
                'min': 0.001,
                'max': 0.5,
                'description': 'Volatility parameter (σ)'
            },
            'time_horizon': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 30.0,
                'description': 'Time horizon (years)'
            },
            'num_steps': {
                'type': 'int',
                'default': 252,
                'min': 50,
                'max': 5000,
                'description': 'Number of time steps'
            },
            'num_paths': {
                'type': 'int',
                'default': 1000,
                'min': 100,
                'max': 10000,
                'description': 'Number of simulation paths'
            },
            'scheme': {
                'type': 'select',
                'default': 'euler',
                'options': ['euler', 'milstein', 'exact'],
                'description': 'Discretization scheme'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility (optional)'
            }
        }
