import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Callable, Tuple
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ..core.base import BaseSimulation, SimulationResult

class ResponseSurfaceMethodology(BaseSimulation):
    """
    Response Surface Methodology (RSM) optimization implementation.
    
    RSM is a collection of mathematical and statistical techniques for modeling
    and analyzing problems where a response of interest is influenced by several
    variables. It uses polynomial approximations to model the response surface
    and find optimal conditions.
    
    Mathematical Background:
    -----------------------
    - Second-order polynomial model: y = β₀ + Σβᵢxᵢ + Σβᵢᵢxᵢ² + ΣΣβᵢⱼxᵢxⱼ
    - Design of experiments (DOE) for efficient sampling
    - Sequential optimization using steepest ascent/descent
    - Central composite design for quadratic model fitting
    
    Algorithm Steps:
    ---------------
    1. Generate initial design points (factorial + center points)
    2. Evaluate objective function at design points
    3. Fit polynomial response surface model
    4. Find optimum of fitted model
    5. Move toward optimum and generate new design
    6. Repeat until convergence
    
    Applications:
    ------------
    - Process optimization in manufacturing
    - Chemical process design
    - Quality improvement
    - Product design optimization
    - Agricultural experiments
    - Pharmaceutical development
    
    Parameters:
    -----------
    objective_function : callable
        Function to minimize f(x) -> float
    bounds : list of tuples
        [(min, max)] bounds for each dimension
    initial_center : array-like, optional
        Initial center point for design
    n_initial_points : int, default=None
        Number of initial design points (auto-calculated if None)
    polynomial_degree : int, default=2
        Degree of polynomial model (1 or 2)
    max_iterations : int, default=20
        Maximum number of RSM iterations
    step_size : float, default=0.3
        Step size for moving toward optimum
    convergence_tol : float, default=1e-6
        Convergence tolerance for optimization
    random_seed : int, optional
        Seed for reproducible results
    """
    
    def __init__(self, objective_function: Callable, bounds: List[Tuple[float, float]],
                 initial_center: Optional[np.ndarray] = None,
                 n_initial_points: Optional[int] = None,
                 polynomial_degree: int = 2, max_iterations: int = 20,
                 step_size: float = 0.3, convergence_tol: float = 1e-6,
                 random_seed: Optional[int] = None):
        super().__init__("Response Surface Methodology")
        
        self.objective_function = objective_function
        self.bounds = bounds
        self.dimension = len(bounds)
        self.polynomial_degree = polynomial_degree
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.convergence_tol = convergence_tol
        
        # Set initial center
        if initial_center is None:
            self.initial_center = np.array([(min_val + max_val) / 2 
                                          for min_val, max_val in bounds])
        else:
            self.initial_center = np.array(initial_center)
        
        # Set number of initial points
        if n_initial_points is None:
            # Use central composite design: 2^k + 2k + 1
            self.n_initial_points = 2**self.dimension + 2*self.dimension + 1
        else:
            self.n_initial_points = n_initial_points
        
        self.parameters.update({
            'n_initial_points': self.n_initial_points,
            'polynomial_degree': polynomial_degree,
            'max_iterations': max_iterations,
            'step_size': step_size,
            'convergence_tol': convergence_tol,
            'dimension': self.dimension,
            'random_seed': random_seed
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        self.is_configured = True
    
    def configure(self, n_initial_points: Optional[int] = None,
                 polynomial_degree: int = 2, max_iterations: int = 20,
                 step_size: float = 0.3, convergence_tol: float = 1e-6) -> bool:
        """Configure RSM parameters"""
        if n_initial_points is None:
            self.n_initial_points = 2**self.dimension + 2*self.dimension + 1
        else:
            self.n_initial_points = n_initial_points
        
        self.polynomial_degree = polynomial_degree
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.convergence_tol = convergence_tol
        
        self.parameters.update({
            'n_initial_points': self.n_initial_points,
            'polynomial_degree': polynomial_degree,
            'max_iterations': max_iterations,
            'step_size': step_size,
            'convergence_tol': convergence_tol
        })
        
        self.is_configured = True
        return True
    
    def _generate_design_points(self, center: np.ndarray, radius: float) -> np.ndarray:
        """Generate central composite design points"""
        points = []
        
        # Center point
        points.append(center.copy())
        
        # Factorial points (2^k)
        for i in range(2**self.dimension):
            point = center.copy()
            for j in range(self.dimension):
                if (i >> j) & 1:
                    point[j] += radius
                else:
                    point[j] -= radius
            points.append(point)
        
        # Axial points (2k)
        alpha = radius * np.sqrt(self.dimension)  # Rotatable design
        for i in range(self.dimension):
            # Positive axial point
            point = center.copy()
            point[i] += alpha
            points.append(point)
            
            # Negative axial point
            point = center.copy()
            point[i] -= alpha
            points.append(point)
        
        # Additional center points for better estimation
        n_center_points = max(1, self.n_initial_points - len(points))
        for _ in range(n_center_points):
            points.append(center.copy())
        
        return np.array(points[:self.n_initial_points])
    
    def _apply_bounds(self, points: np.ndarray) -> np.ndarray:
        """Apply boundary constraints to design points"""
        bounded_points = points.copy()
        for i in range(self.dimension):
            min_val, max_val = self.bounds[i]
            bounded_points[:, i] = np.clip(bounded_points[:, i], min_val, max_val)
        return bounded_points
    
    def _fit_response_surface(self, X: np.ndarray, y: np.ndarray) -> Pipeline:
        """Fit polynomial response surface model"""
        poly_features = PolynomialFeatures(degree=self.polynomial_degree, include_bias=True)
        linear_reg = LinearRegression()
        model = Pipeline([('poly', poly_features), ('linear', linear_reg)])
        model.fit(X, y)
        return model
    
    def _find_model_optimum(self, model: Pipeline, center: np.ndarray) -> np.ndarray:
        """Find optimum of fitted response surface model using gradient descent"""
        current_point = center.copy()
        
        for _ in range(100):  # Local optimization iterations
            # Compute gradient numerically
            gradient = np.zeros(self.dimension)
            h = 1e-6
            
            for i in range(self.dimension):
                point_plus = current_point.copy()
                point_minus = current_point.copy()
                point_plus[i] += h
                point_minus[i] -= h
                
                f_plus = model.predict(point_plus.reshape(1, -1))[0]
                f_minus = model.predict(point_minus.reshape(1, -1))[0]
                gradient[i] = (f_plus - f_minus) / (2 * h)
            
            # Update point
            new_point = current_point - 0.01 * gradient
            
            # Apply bounds
            for i in range(self.dimension):
                min_val, max_val = self.bounds[i]
                new_point[i] = np.clip(new_point[i], min_val, max_val)
            
            # Check convergence
            if np.linalg.norm(new_point - current_point) < 1e-8:
                break
            
            current_point = new_point
        
        return current_point
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute Response Surface Methodology optimization"""
        if not self.is_configured:
            raise RuntimeError("Algorithm not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Initialize
        current_center = self.initial_center.copy()
        best_solution = current_center.copy()
        best_fitness = self.objective_function(best_solution)
        
        # Calculate initial radius as fraction of search space
        radius = min([(max_val - min_val) * 0.2 for min_val, max_val in self.bounds])
        
        # Tracking arrays
        center_history = [current_center.copy()]
        best_fitness_history = [best_fitness]
        model_predictions = []
        design_points_history = []
        radius_history = [radius]
        
        iteration = 0
        converged = False
        
        while iteration < self.max_iterations and not converged:
            # Generate design points
            design_points = self._generate_design_points(current_center, radius)
            design_points = self._apply_bounds(design_points)
            
            # Evaluate objective function at design points
            responses = np.array([self.objective_function(point) for point in design_points])
            
            # Update best solution
            best_idx = np.argmin(responses)
            if responses[best_idx] < best_fitness:
                best_solution = design_points[best_idx].copy()
                best_fitness = responses[best_idx]
            
            # Fit response surface model
            try:
                model = self._fit_response_surface(design_points, responses)
                
                # Find optimum of fitted model
                model_optimum = self._find_model_optimum(model, current_center)
                model_prediction = model.predict(model_optimum.reshape(1, -1))[0]
                
                # Move toward model optimum
                direction = model_optimum - current_center
                step = self.step_size * direction
                new_center = current_center + step
                
                # Apply bounds to new center
                for i in range(self.dimension):
                    min_val, max_val = self.bounds[i]
                    new_center[i] = np.clip(new_center[i], min_val, max_val)
                
                # Check convergence
                if np.linalg.norm(new_center - current_center) < self.convergence_tol:
                    converged = True
                
                current_center = new_center
                
            except Exception as e:
                print(f"Warning: Model fitting failed at iteration {iteration}: {e}")
                # Random perturbation if model fitting fails
                perturbation = np.random.normal(0, radius * 0.1, self.dimension)
                current_center = current_center + perturbation
                current_center = self._apply_bounds(current_center.reshape(1, -1))[0]
                model_prediction = np.nan
            
            # Adaptive radius adjustment
            if iteration > 0:
                improvement = best_fitness_history[-1] - best_fitness
                if improvement > 0:
                    radius *= 1.1  # Expand search if improving
                else:
                    radius *= 0.9  # Contract search if not improving
                
                # Keep radius within reasonable bounds
                min_radius = min([(max_val - min_val) * 0.01 for min_val, max_val in self.bounds])
                max_radius = min([(max_val - min_val) * 0.5 for min_val, max_val in self.bounds])
                radius = np.clip(radius, min_radius, max_radius)
            
            # Record progress
            center_history.append(current_center.copy())
            best_fitness_history.append(best_fitness)
            model_predictions.append(model_prediction)
            design_points_history.append(design_points.copy())
            radius_history.append(radius)
            
            iteration += 1
        
        execution_time = time.time() - start_time
        
        # Calculate final statistics
        total_evaluations = sum(len(points) for points in design_points_history)
        improvement = best_fitness_history[0] - best_fitness
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'best_solution': best_solution.tolist(),
                'best_fitness': best_fitness,
                'initial_fitness': best_fitness_history[0],
                'iterations': iteration,
                'converged': converged,
                'total_evaluations': total_evaluations
            },
            statistics={
                'improvement': improvement,
                'convergence_rate': improvement / iteration if iteration > 0 else 0,
                'average_evaluations_per_iteration': total_evaluations / iteration if iteration > 0 else 0,
                'final_radius': radius
            },
            raw_data={
                'center_history': center_history,
                'best_fitness_history': best_fitness_history,
                'model_predictions': model_predictions,
                'design_points_history': design_points_history,
                'radius_history': radius_history
            },
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, **kwargs) -> None:
        """Visualize RSM optimization progress"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No optimization results available. Run the optimization first.")
            return
        
        center_history = result.raw_data['center_history']
        best_fitness_history = result.raw_data['best_fitness_history']
        model_predictions = result.raw_data['model_predictions']
        design_points_history = result.raw_data['design_points_history']
        radius_history = result.raw_data['radius_history']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Fitness evolution
        iterations = range(len(best_fitness_history))
        ax1.plot(iterations, best_fitness_history, 'r-', linewidth=2, label='Best Fitness')
        
        # Add model predictions where available
        valid_predictions = [pred for pred in model_predictions if not np.isnan(pred)]
        if valid_predictions:
            pred_iterations = [i+1 for i, pred in enumerate(model_predictions) if not np.isnan(pred)]
            ax1.scatter(pred_iterations, valid_predictions, 
                       color='blue', alpha=0.7, label='Model Predictions', s=50)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Evolution and Model Predictions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Search radius evolution
        ax2.plot(iterations[1:], radius_history[1:], 'green', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Search Radius')
        ax2.set_title('Adaptive Search Radius')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Center point trajectory (2D case) or parameter evolution
        if self.dimension == 2:
            centers = np.array(center_history)
            ax3.plot(centers[:, 0], centers[:, 1], 'bo-', linewidth=2, markersize=6, alpha=0.7)
            ax3.scatter(centers[0, 0], centers[0, 1], color='green', s=100, marker='s', 
                       label='Start', zorder=5)
            ax3.scatter(centers[-1, 0], centers[-1, 1], color='red', s=100, marker='*', 
                       label='Final', zorder=5)
            
            # Show design points for last iteration
            if design_points_history:
                last_design = design_points_history[-1]
                ax3.scatter(last_design[:, 0], last_design[:, 1], 
                           color='orange', alpha=0.5, s=30, label='Last Design Points')
            
            ax3.set_xlabel('X₁')
            ax3.set_ylabel('X₂')
            ax3.set_title('Center Point Trajectory')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            # For higher dimensions, show parameter evolution
            centers = np.array(center_history)
            for dim in range(min(3, self.dimension)):
                ax3.plot(iterations, centers[:, dim], 
                        label=f'x_{dim+1}', alpha=0.7, marker='o')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Parameter Value')
            ax3.set_title('Center Point Parameter Evolution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Number of design points per iteration
        n_points_per_iteration = [len(points) for points in design_points_history]
        if n_points_per_iteration:
            ax4.bar(range(1, len(n_points_per_iteration) + 1), n_points_per_iteration, 
                   alpha=0.7, color='purple')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Number of Design Points')
            ax4.set_title('Design Points per Iteration')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\nOptimization Summary:")
        print(f"Best fitness: {result.results['best_fitness']:.6f}")
        print(f"Best solution: {result.results['best_solution']}")
        print(f"Initial fitness: {result.results['initial_fitness']:.6f}")
        print(f"Iterations: {result.results['iterations']}")
        print(f"Converged: {result.results['converged']}")
        print(f"Total evaluations: {result.results['total_evaluations']}")
        print(f"Total improvement: {result.statistics['improvement']:.6f}")
        print(f"Final search radius: {result.statistics['final_radius']:.6f}")
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'n_initial_points': {
                'type': 'int',
                'default': None,
                'min': 5,
                'max': 1000,
                'description': 'Number of initial design points (auto if None)'
            },
            'polynomial_degree': {
                'type': 'int',
                'default': 2,
                'min': 1,
                'max': 3,
                'description': 'Degree of polynomial model'
            },
            'max_iterations': {
                'type': 'int',
                'default': 20,
                'min': 5,
                'max': 100,
                'description': 'Maximum number of RSM iterations'
            },
            'step_size': {
                'type': 'float',
                'default': 0.3,
                'min': 0.1,
                'max': 1.0,
                'description': 'Step size for moving toward optimum'
            },
            'convergence_tol': {
                'type': 'float',
                'default': 1e-6,
                'min': 1e-10,
                'max': 1e-3,
                'description': 'Convergence tolerance'
            }
        }
    
    def validate_parameters(self) -> List[str]:
        """Validate optimization parameters"""
        errors = []
        if self.n_initial_points < 3:
            errors.append("n_initial_points must be at least 3")
        if self.polynomial_degree < 1 or self.polynomial_degree > 3:
            errors.append("polynomial_degree must be 1, 2, or 3")
        if self.max_iterations < 1:
            errors.append("max_iterations must be positive")
        if self.step_size <= 0 or self.step_size > 1:
            errors.append("step_size must be between 0 and 1")
        if self.convergence_tol <= 0:
            errors.append("convergence_tol must be positive")
        return errors
