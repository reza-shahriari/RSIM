""""RSIM - A Python package for different simulation approaches

This package provides various simulation methods with a focus on variance reduction
techniques for Monte Carlo simulations and other stochastic processes.
"""

from .core.base import BaseSimulation, SimulationResult
from .monte_carlo import *
from .queueing import *
from .inventory import *
from .financial import *
from .reliability import *
from .variance_reduction import *
from .markov_chains import *
from .bootstrap import *
from .networks import *
from .optimization import *
from .survival import *




# Package metadata
__title__ = "RSIM"
__description__ = "A Python package for different simulation approaches"
__url__ = "https://github.com/reza-shahriari/RSIM"
__author__ = "Rezaa Shahriari"
__license__ = "MIT"
__copyright__ = "Copyright 2024 Reza Shahriari"
__email__ ="Rezshahriari@gmail.com"
__version__='1.0.0'

__all__ = [

    "__version__",
    "__author__",
    "__email__",
    "core",
    "monte_carlo",
    "queueing",
    "inventory",
    "financial",
    "reliability",
    "variance_reduction",
    "markov_chains",
    "bootstrap",
    "networks",
    "optimization",
    "survival",
]

# Version info tuple for programmatic access
VERSION = tuple(map(int, __version__.split('.')))

def get_version():
    """Return the version string."""
    return __version__

def get_info():
    """Return package information as a dictionary."""
    return {
        'name': __title__,
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'description': __description__,
        'url': __url__,
        'license': __license__,
    }
