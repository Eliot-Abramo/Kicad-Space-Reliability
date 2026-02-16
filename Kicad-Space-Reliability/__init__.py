"""
KiCad Reliability Calculator Plugin
====================================
IEC TR 62380 Reliability Analysis for KiCad

Version: 3.2.0

Features:
- 12 component classes with full IEC TR 62380 stress models
- Visual block diagram editor for system topology
- Series/Parallel/K-of-N redundancy modeling
- Monte Carlo uncertainty analysis with convergence detection
- OAT tornado sensitivity analysis (IEC 60300-3-1)
- Design-margin what-if scenarios
- Component-level parameter criticality analysis
- Multi-phase mission profile support (IEC TR 62380 phasing)
- Professional HTML/PDF reports with SVG charts

Designed and developed by Eliot Abramo
License: MIT
"""

__version__ = "3.2.0"
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
    TornadoResult,
    DesignMarginResult,
    tornado_sheet_sensitivity,
    tornado_parameter_sensitivity,
    design_margin_analysis,
    analyze_board_criticality,
)

# v3.1.0 Co-Design modules
from .mission_profile import (
    MissionPhase,
    MissionProfile,
    MISSION_TEMPLATES,
    compute_phased_lambda,
    estimate_phasing_impact,
)

from .budget_allocation import (
    allocate_budget,
    BudgetAllocationResult,
    AllocationStrategy,
    compute_max_fit_for_target,
)

from .derating_engine import (
    compute_derating_guidance,
    DeratingResult,
    DeratingRecommendation,
)

from .component_swap import (
    analyze_package_swaps,
    analyze_type_swaps,
    quick_swap_comparison,
    rank_all_swaps,
    SwapAnalysisResult,
)

from .growth_tracking import (
    create_snapshot,
    save_snapshot,
    load_snapshots,
    compare_revisions,
    build_growth_timeline,
    ReliabilitySnapshot,
    RevisionComparison,
    GrowthTimeline,
)

from .correlated_mc import (
    correlated_monte_carlo,
    CorrelationGroup,
    CorrelatedMCResult,
    auto_group_by_sheet,
    auto_group_by_type,
)

from .reliability_dialog import ReliabilityMainDialog
