"""
KiCad Reliability Calculator Plugin
====================================
IEC TR 62380 Reliability Analysis for KiCad

Version: 2.0.0

Features:
- Visual block diagram editor for system topology
- Component reliability calculation per IEC TR 62380
- Series/Parallel/K-of-N redundancy modeling
- Monte Carlo uncertainty analysis
- Sobol sensitivity analysis
- Professional HTML/Markdown/CSV/JSON reports

New in v2.0.0:
- Configurable EOS (Electrical Overstress) with 10 interface types
- Working time ratio (τ_on) for duty-cycled operation
- Thermal expansion materials (substrate and package)
- Corrected thermal cycling threshold (8760 cycles/year)

Installation:
    Copy this folder to your KiCad plugins directory:
    - Linux: ~/.local/share/kicad/9.0/scripting/plugins/
    - Windows: %APPDATA%\\kicad\\9.0\\scripting\\plugins\\
    - macOS: ~/Library/Preferences/kicad/9.0/scripting/plugins/

Author: Generated with Claude assistance
License: MIT
"""

__version__ = "2.0.0"
__author__ = "KiCad Reliability Plugin Team"

try:
    from .plugin import ReliabilityPlugin

    ReliabilityPlugin().register()
except Exception as e:
    import logging

    logging.warning(f"Could not register ReliabilityPlugin: {e}")

# Expose main classes for external use
from .reliability_math import (
    calculate_lambda,
    calculate_component_lambda,
    reliability_from_lambda,
    lambda_from_reliability,
    r_series,
    r_parallel,
    r_k_of_n,
    get_component_types,
    get_field_definitions,
    INTERFACE_EOS_VALUES,
    THERMAL_EXPANSION_SUBSTRATE,
)

from .reliability_dialog import ReliabilityMainDialog
