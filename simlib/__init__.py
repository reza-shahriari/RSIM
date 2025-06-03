"""
SimLib - A Comprehensive Simulation Library
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

__version__ = "1.0.0"
__author__ = "SimLib Team"