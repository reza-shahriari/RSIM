from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

@dataclass
class SimulationResult:
    """Container for simulation results"""
    simulation_name: str
    parameters: Dict[str, Any]
    results: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, float] = field(default_factory=dict)
    raw_data: Optional[np.ndarray] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    convergence_data: Optional[List] = None
    
    def summary(self) -> str:
        """Return a formatted summary of results"""
        summary = f"=== {self.simulation_name} Results ===\n"
        summary += f"Timestamp: {self.timestamp}\n"
        summary += f"Execution Time: {self.execution_time:.4f} seconds\n"
        summary += "\nParameters:\n"
        for key, value in self.parameters.items():
            summary += f"  {key}: {value}\n"
        summary += "\nResults:\n"
        for key, value in self.results.items():
            summary += f"  {key}: {value}\n"
        return summary

class BaseSimulation(ABC):
    """Base class for all simulations"""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.parameters = kwargs
        self.result: Optional[SimulationResult] = None
        self.is_configured = False
        self.random_seed: Optional[int] = None
        
    @abstractmethod
    def configure(self, **kwargs) -> bool:
        """Configure simulation parameters"""
        pass
    
    @abstractmethod
    def run(self, **kwargs) -> SimulationResult:
        """Execute the simulation"""
        pass
    
    @abstractmethod
    def visualize(self, result: Optional[SimulationResult] = None, **kwargs) -> None:
        """Create visualizations of simulation results"""
        pass
    
    def set_random_seed(self, seed: int) -> None:
        """Set random seed for reproducibility"""
        self.random_seed = seed
        np.random.seed(seed)
    
    def validate_parameters(self) -> List[str]:
        """Validate simulation parameters, return list of errors"""
        return []
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Return parameter information for UI generation"""
        return {}
    
    def export_results(self, filepath: str, format: str = 'json') -> bool:
        """Export results to file"""
        pass
    
    def reset(self) -> None:
        """Reset simulation to initial state"""
        self.result = None
        self.is_configured = False

class ParametricSimulation(BaseSimulation):
    """Base class for simulations with parameter sweeps"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.parameter_ranges: Dict[str, Tuple] = {}
    
    @abstractmethod
    def run_parameter_sweep(self, parameters: Dict[str, List]) -> List[SimulationResult]:
        """Run simulation across parameter ranges"""
        pass
    
    def visualize_parameter_sweep(self, results: List[SimulationResult]) -> None:
        """Visualize parameter sweep results"""
        pass
