"""
Unified Sensitivity & Uncertainty Analysis.
Author:  Eliot Abramo

DEPRECATED: Use monte_carlo.run_uncertainty_analysis() and
sensitivity_analysis.tornado_analysis() instead.  The Sobol functionality
in .engine is exploratory and not shipped as a production feature.
"""

import warnings

from .engine import (
    SobolResult,
    UncertaintyResult,
    WhatIfResult,
    run_monte_carlo,
    run_sobol,
    run_whatif,
)

__all__ = [
    "SobolResult",
    "UncertaintyResult",
    "WhatIfResult",
    "run_monte_carlo",
    "run_sobol",
    "run_whatif",
]

warnings.warn(
    "plugins.analysis is deprecated. Use monte_carlo / sensitivity_analysis instead.",
    DeprecationWarning,
    stacklevel=2,
)
