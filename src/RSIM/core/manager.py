from typing import Dict, List, Type, Optional
from .base import BaseSimulation, SimulationResult

class SimulationManager:
    """Central manager for all simulations - designed for Qt UI integration"""
    
    def __init__(self):
        self.available_simulations: Dict[str, Type[BaseSimulation]] = {}
        self.active_simulations: Dict[str, BaseSimulation] = {}
        self.simulation_history: List[SimulationResult] = []
        self.categories: Dict[str, List[str]] = {}
        
    def register_simulation(self, simulation_class: Type[BaseSimulation], 
                          category: str) -> None:
        """Register a simulation class"""
        pass
    
    def get_simulation_categories(self) -> Dict[str, List[str]]:
        """Get organized simulation categories for UI tree"""
        pass
    
    def create_simulation(self, simulation_name: str, instance_name: str) -> BaseSimulation:
        """Create new simulation instance"""
        pass
    
    def get_simulation_parameters(self, simulation_name: str) -> Dict:
        """Get parameter info for UI form generation"""
        pass
    
    def run_simulation(self, instance_name: str, **parameters) -> SimulationResult:
        """Run simulation with parameters"""
        pass
    
    def get_simulation_results(self, instance_name: str) -> Optional[SimulationResult]:
        """Get results from simulation"""
        pass
    
    def export_simulation(self, instance_name: str, filepath: str) -> bool:
        """Export simulation configuration and results"""
        pass
    
    def import_simulation(self, filepath: str) -> str:
        """Import simulation configuration"""
        pass
    
    def compare_simulations(self, instance_names: List[str]) -> Dict:
        """Compare multiple simulation results"""
        pass

    def get_available_visualizations(self, simulation_name: str) -> List[str]:
        """Get available visualization options for UI"""
        pass