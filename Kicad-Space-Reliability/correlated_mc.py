# Replace entire correlated_mc.py with:

"""
Correlated Monte Carlo Module (DEPRECATED)
Replaced by shared-parameter mechanism in monte_carlo.py v4.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class CorrelationGroup:
    name: str = ""
    references: List[str] = field(default_factory=list)
    rho: float = 0.0

@dataclass
class CorrelatedMCResult:
    samples: List = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0


def correlated_monte_carlo(*args, **kwargs):
    warnings.warn("Deprecated. Use monte_carlo.run_uncertainty_analysis().", DeprecationWarning, stacklevel=2)
    return CorrelatedMCResult()

def auto_group_by_sheet(*args, **kwargs):
    return []

def auto_group_by_type(*args, **kwargs):
    return []

def auto_group_all_on_board(*args, **kwargs):
    return []

def run_correlated_mc(*args, **kwargs):
    return None