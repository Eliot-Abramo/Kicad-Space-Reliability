"""
KiCad Reliability Calculator Plugin
====================================
IEC TR 62380 Reliability Analysis for KiCad

Version: 3.3.0

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

__version__ = "3.3.0"
__author__ = "Eliot Abramo"

try:
    from .plugin import ReliabilityPlugin

    ReliabilityPlugin().register()
except Exception as e:  # noqa: BLE001
    import logging

    logging.getLogger(__name__).warning("Could not register ReliabilityPlugin: %s", e)

# Expose main classes for external use
import contextlib

from .budget_allocation import (
    AllocationStrategy,
    BudgetAllocationResult,
    allocate_budget,
    compute_max_fit_for_target,
)
from .component_swap import (
    SwapAnalysisResult,
    analyze_package_swaps,
    analyze_type_swaps,
    quick_swap_comparison,
    rank_all_swaps,
)
from .correlated_mc import (
    CorrelatedMCResult,
    CorrelationGroup,
    auto_group_by_sheet,
    auto_group_by_type,
    correlated_monte_carlo,
)
from .derating_engine import (
    DeratingRecommendation,
    DeratingResult,
    compute_derating_guidance,
)
from .growth_tracking import (
    GrowthTimeline,
    ReliabilitySnapshot,
    RevisionComparison,
    build_growth_timeline,
    compare_revisions,
    create_snapshot,
    load_snapshots,
    save_snapshot,
)

# Co-Design modules
from .mission_profile import (
    MISSION_TEMPLATES,
    MissionPhase,
    MissionProfile,
    compute_phased_lambda,
    estimate_phasing_impact,
)

with contextlib.suppress(Exception):
    from .reliability_dialog import ReliabilityMainDialog
from .reliability_math import (
    DISCRETE_PACKAGE_TABLE,
    IC_DIE_TABLE,
    IC_PACKAGE_TABLE,
    INTERFACE_EOS_VALUES,
    THERMAL_EXPANSION_PACKAGE,
    THERMAL_EXPANSION_SUBSTRATE,
    analyze_component_criticality,
    calculate_component_lambda,
    calculate_lambda,
    fit_to_lambda,
    format_lambda,
    format_reliability,
    get_component_types,
    get_field_definitions,
    lambda_from_reliability,
    lambda_series,
    lambda_to_fit,
    mttf_from_lambda,
    r_k_of_n,
    r_parallel,
    r_series,
    reliability_from_lambda,
)
from .sensitivity_analysis import (
    DesignMarginResult,
    TornadoResult,
    analyze_board_criticality,
    design_margin_analysis,
    tornado_parameter_sensitivity,
    tornado_sheet_sensitivity,
)
