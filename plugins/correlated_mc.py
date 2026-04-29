# Replace entire correlated_mc.py with:

"""
Correlated Monte Carlo Module (DEPRECATED)
Replaced by shared-parameter mechanism in monte_carlo.py v4.
Author:  Eliot Abramo
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field


@dataclass
class CorrelationGroup:
    name: str = ""
    references: list[str] = field(default_factory=list)
    rho: float = 0.0


@dataclass
class CorrelatedMCResult:
    samples: list = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0


def correlated_monte_carlo(*args, **kwargs):  # noqa: ARG001
    warnings.warn("Deprecated. Use monte_carlo.run_uncertainty_analysis().", DeprecationWarning, stacklevel=2)
    return CorrelatedMCResult()


def auto_group_by_sheet(*args, **kwargs):  # noqa: ARG001
    return []


def auto_group_by_type(*args, **kwargs):  # noqa: ARG001
    return []


def auto_group_all_on_board(*args, **kwargs):  # noqa: ARG001
    return []


def run_correlated_mc(*args, **kwargs):  # noqa: ARG001
    return None
