from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SimulationConfig:
    """Global simulation configuration"""
    default_random_seed: int = 42
    default_output_format: str = 'json'
    matplotlib_backend: str = 'Qt5Agg'
    figure_dpi: int = 100
    figure_size: tuple = (10, 6)
    color_palette: str = 'viridis'
    
    # Performance settings
    max_simulation_time: float = 3600.0  # 1 hour max
    progress_update_interval: int = 1000
    
    # Qt UI settings
    window_title: str = "SimLib - Simulation Library"
    default_window_size: tuple = (1200, 800)
    theme: str = 'light'

class ConfigManager:
    """Manage simulation library configuration"""
    
    def __init__(self):
        self.config = SimulationConfig()
    
    def load_config(self, filepath: str) -> bool:
        """Load configuration from file"""
        pass
    
    def save_config(self, filepath: str) -> bool:
        """Save configuration to file"""
        pass
    
    def get_config(self) -> SimulationConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters"""
        pass