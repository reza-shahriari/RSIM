import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Callable, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ..core.base import BaseSimulation, SimulationResult

class GeneticAlgorithm(BaseSimulation):
    """
    Genetic Algorithm optimization implementation.
    
    Genetic Algorithm is a metaheuristic inspired by the process of natural selection.
    It evolves a population of candidate solutions through selection, crossover, and
    mutation operations to find optimal or near-optimal solutions.
    
    Mathematical Background:
    -----------------------
    - Population-based search with fitness-proportionate selection
    - Crossover: combines genetic material from two parents
    - Mutation: introduces random variations to maintain diversity
    - Selection pressure drives evolution toward better solutions
    - Convergence depends on population size, selection, and genetic operators
    
    Algorithm Steps:
    ---------------
    1. Initialize random population
    2. Evaluate fitness of all individuals
    3. Select parents based on fitness
    4. Apply crossover to create offspring
    5. Apply mutation to offspring
    6. Replace population with new generation
    7. Repeat until convergence or max generations
    
    Applications:
    ------------
    - Function optimization
    - Neural network training
    - Scheduling and resource allocation
    - Feature selection
    - Game strategy evolution
    - Engineering design optimization
    
    Parameters:
    -----------
    objective_function : callable
        Function to minimize f(x) -> float
    bounds : list of tuples
        [(min, max)] bounds for each dimension
    population_size : int, default=50
        Number of individuals in population
    max_generations : int, default=100
        Maximum number of generations
    crossover_rate : float, default=0.8
        Probability of crossover between parents
    mutation_rate : float, default=0.1
        Probability of mutation for each gene
    elitism_rate : float, default=0.1
        Fraction of best individuals to preserve
    tournament_size : int, default=3
        Size of tournament for selection
    random_seed : int, optional
        Seed for reproducible results
    """
    
    def __init__(self, objective_function: Callable, bounds: List[Tuple[float, float]],
                 population_size: int = 50, max_generations: int = 100,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                 elitism_rate: float = 0.1, tournament_size: int = 3,
                 random_seed: Optional[int] = None):
        super().__init__("Genetic Algorithm")
        
        self.objective_function = objective_function
        self.bounds = bounds
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.dimension = len(bounds)
        
        self.parameters.update({
            'population_size': population_size,
            'max_generations': max_generations,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'elitism_rate': elitism_rate,
            'tournament_size': tournament_size,
            'dimension': self.dimension,
            'random_seed': random_seed
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        self.is_configured = True
    
    def configure(self, population_size: int = 50, max_generations: int = 100,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                 elitism_rate: float = 0.1, tournament_size: int = 3) -> bool:
        """Configure genetic algorithm parameters"""
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        
        self.parameters.update({
            'population_size': population_size,
            'max_generations': max_generations,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'elitism_rate': elitism_rate,
            'tournament_size': tournament_size
        })
        
        self.is_configured = True
        return True
    
    def _initialize_population(self) -> np.ndarray:
        """Initialize random population within bounds"""
        population = np.zeros((self.population_size, self.dimension))
        for i in range(self.dimension):
            min_val, max_val = self.bounds[i]
            population[:, i] = np.random.uniform(min_val, max_val, self.population_size)
        return population
    
    def _evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness for entire population"""
        fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            fitness[i] = self.objective_function(population[i])
        return fitness
    
    def _tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Select parent using tournament selection"""
        tournament_indices = np.random.choice(self.population_size, self.tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover between two parents"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        mask = np.random.random(self.dimension) < 0.5
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]
        
        return child1, child2
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Apply Gaussian mutation to individual"""
        mutated = individual.copy()
        for i in range(self.dimension):
            if np.random.random() < self.mutation_rate:
                min_val, max_val = self.bounds[i]
                mutation_strength = (max_val - min_val) * 0.1
                mutated[i] += np.random.normal(0, mutation_strength)
                mutated[i] = np.clip(mutated[i], min_val, max_val)
        return mutated
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute genetic algorithm optimization"""
        if not self.is_configured:
            raise RuntimeError("Algorithm not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Initialize population
        population = self._initialize_population()
        fitness = self._evaluate_fitness(population)
        
        # Tracking arrays
        best_fitness_history = []
        mean_fitness_history = []
        std_fitness_history = []
        best_solutions_history = []
        
        # Evolution loop
        for generation in range(self.max_generations):
            # Track statistics
            best_idx = np.argmin(fitness)
            best_fitness_history.append(fitness[best_idx])
            mean_fitness_history.append(np.mean(fitness))
            std_fitness_history.append(np.std(fitness))
            best_solutions_history.append(population[best_idx].copy())
            
            # Create new population
            new_population = np.zeros_like(population)
            
            # Elitism: preserve best individuals
            n_elite = int(self.elitism_rate * self.population_size)
            if n_elite > 0:
                elite_indices = np.argsort(fitness)[:n_elite]
                new_population[:n_elite] = population[elite_indices]
            
            # Generate offspring
            for i in range(n_elite, self.population_size, 2):
                # Selection
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # Add to new population
                new_population[i] = child1
                if i + 1 < self.population_size:
                    new_population[i + 1] = child2
            
            # Replace population
            population = new_population
            fitness = self._evaluate_fitness(population)
        
        execution_time = time.time() - start_time
        
        # Final statistics
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'best_solution': best_solution.tolist(),
                'best_fitness': best_fitness,
                'final_population': population.tolist(),
                'final_fitness': fitness.tolist(),
                'generations': self.max_generations,
                'convergence_generation': np.argmin(best_fitness_history)
            },
            statistics={
                'improvement': best_fitness_history[0] - best_fitness,
                'convergence_rate': (best_fitness_history[0] - best_fitness) / self.max_generations,
                'final_diversity': np.std(fitness),
                'best_fitness_final': best_fitness
            },
            raw_data={
                'best_fitness_history': best_fitness_history,
                'mean_fitness_history': mean_fitness_history,
                'std_fitness_history': std_fitness_history,
                'best_solutions_history': best_solutions_history
            },
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, **kwargs) -> None:
        """Visualize genetic algorithm optimization progress"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No optimization results available. Run the optimization first.")
            return
        
        best_fitness = result.raw_data['best_fitness_history']
        mean_fitness = result.raw_data['mean_fitness_history']
        std_fitness = result.raw_data['std_fitness_history']
        best_solutions = result.raw_data['best_solutions_history']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Fitness evolution
        generations = range(len(best_fitness))
        ax1.plot(generations, best_fitness, 'r-', linewidth=2, label='Best Fitness')
        ax1.plot(generations, mean_fitness, 'b-', linewidth=1, label='Mean Fitness')
        ax1.fill_between(generations, 
                        np.array(mean_fitness) - np.array(std_fitness),
                        np.array(mean_fitness) + np.array(std_fitness),
                        alpha=0.3, color='blue', label='±1 Std Dev')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Evolution Over Generations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Population diversity
        ax2.plot(generations, std_fitness, 'green', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness Standard Deviation')
        ax2.set_title('Population Diversity Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Final population distribution (2D case)
        if self.dimension == 2:
            final_pop = np.array(result.results['final_population'])
            final_fit = np.array(result.results['final_fitness'])
            
            scatter = ax3.scatter(final_pop[:, 0], final_pop[:, 1], 
                                c=final_fit, cmap='viridis', alpha=0.7)
            ax3.scatter(result.results['best_solution'][0], 
                       result.results['best_solution'][1],
                       color='red', s=100, marker='*', label='Best Solution', zorder=5)
            ax3.set_xlabel('X₁')
            ax3.set_ylabel('X₂')
            ax3.set_title('Final Population Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax3, label='Fitness')
        else:
            # For higher dimensions, show parameter evolution
            best_solutions_array = np.array(best_solutions)
            for dim in range(min(3, self.dimension)):
                ax3.plot(generations, best_solutions_array[:, dim], 
                        label=f'x_{dim+1}', alpha=0.7)
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Parameter Value')
            ax3.set_title('Best Solution Evolution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Convergence analysis
        improvement = np.array(best_fitness[0]) - np.array(best_fitness)
        ax4.plot(generations, improvement, 'purple', linewidth=2)
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Improvement from Initial')
        ax4.set_title('Convergence Progress')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\nOptimization Summary:")
        print(f"Best fitness: {result.results['best_fitness']:.6f}")
        print(f"Best solution: {result.results['best_solution']}")
        print(f"Generations: {result.results['generations']}")
        print(f"Convergence generation: {result.results['convergence_generation']}")
        print(f"Total improvement: {result.statistics['improvement']:.6f}")
        print(f"Final population diversity: {result.statistics['final_diversity']:.6f}")
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'population_size': {
                'type': 'int',
                'default': 50,
                'min': 10,
                'max': 500,
                'description': 'Number of individuals in population'
            },
            'max_generations': {
                'type': 'int',
                'default': 100,
                'min': 10,
                'max': 1000,
                'description': 'Maximum number of generations'
            },
            'crossover_rate': {
                'type': 'float',
                'default': 0.8,
                'min': 0.0,
                'max': 1.0,
                'description': 'Probability of crossover between parents'
            },
            'mutation_rate': {
                'type': 'float',
                'default': 0.1,
                'min': 0.0,
                'max': 1.0,
                'description': 'Probability of mutation for each gene'
            },
            'elitism_rate': {
                'type': 'float',
                'default': 0.1,
                'min': 0.0,
                'max': 0.5,
                'description': 'Fraction of best individuals to preserve'
            },
            'tournament_size': {
                'type': 'int',
                'default': 3,
                'min': 2,
                'max': 10,
                'description': 'Size of tournament for selection'
            }
        }
    
    def validate_parameters(self) -> List[str]:
        """Validate optimization parameters"""
        errors = []
        if self.population_size < 2:
            errors.append("population_size must be at least 2")
        if self.max_generations < 1:
            errors.append("max_generations must be positive")
        if not (0 <= self.crossover_rate <= 1):
            errors.append("crossover_rate must be between 0 and 1")
        if not (0 <= self.mutation_rate <= 1):
            errors.append("mutation_rate must be between 0 and 1")
        if not (0 <= self.elitism_rate <= 1):
            errors.append("elitism_rate must be between 0 and 1")
        if self.tournament_size < 2 or self.tournament_size > self.population_size:
            errors.append("tournament_size must be between 2 and population_size")
        return errors
