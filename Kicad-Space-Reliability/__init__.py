"""
KiCad Reliability Calculator Plugin
====================================
IEC TR 62380 Reliability Analysis for KiCad

Version: 3.0.0

Features:
- 12 component classes with full IEC TR 62380 stress models
- Visual block diagram editor for system topology
- Series/Parallel/K-of-N redundancy modeling
- Monte Carlo uncertainty analysis with convergence detection
- Sobol sensitivity analysis with interaction detection
- Component-level parameter criticality analysis
- Professional HTML/Markdown/CSV/JSON reports with SVG charts

New in v3.0.0:
- Complete rewrite of reliability_math.py
- 12 component types: IC, Diode, Transistor, Optocoupler, Thyristor,
  Capacitor, Resistor, Inductor, Relay, Connector, PCB/Solder, Misc
- Full stress derating models (voltage, current, temperature, thermal cycling)
- Coffin-Manson thermal cycling with CTE mismatch
- Arrhenius temperature acceleration with technology-specific Ea
- Component-level parameter criticality (elasticity analysis)
- Defensive input validation - plugin never crashes on bad data
- All formulas cite IEC TR 62380 section/table/page references

Designed and developed by Eliot Abramo
License: MIT
"""

__version__ = "3.0.0"
__author__ = "Eliot Abramo"

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
    mttf_from_lambda,
    r_series,
    r_parallel,
    r_k_of_n,
    lambda_series,
    get_component_types,
    get_field_definitions,
    analyze_component_criticality,
    fit_to_lambda,
    lambda_to_fit,
    format_lambda,
    format_reliability,
    INTERFACE_EOS_VALUES,
    THERMAL_EXPANSION_SUBSTRATE,
    THERMAL_EXPANSION_PACKAGE,
    IC_DIE_TABLE,
    IC_PACKAGE_TABLE,
    DISCRETE_PACKAGE_TABLE,
)

from .sensitivity_analysis import (
    SobolAnalyzer,
    SobolResult,
    quick_sensitivity,
    analyze_board_criticality,
)

from .reliability_dialog import ReliabilityMainDialog
