"""
Reliability Budget Allocation Module
=====================================
Decomposes system-level reliability targets into per-sheet and per-component
budgets using multiple apportionment strategies.

Strategies implemented:
  1. Equal Apportionment  -- each component gets the same FIT budget
  2. Proportional (ARINC) -- budget proportional to current failure rate
  3. Complexity-Weighted  -- budget proportional to component count per sheet
  4. Criticality-Weighted -- high-criticality components get tighter budgets

The core identity:
    R_system = exp(-lambda_system * t)
    lambda_system = -ln(R_target) / t
    Each component budget: lambda_i such that SUM(lambda_i) <= lambda_system

Author:  Eliot Abramo
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class AllocationStrategy(Enum):
    EQUAL = "equal"
    PROPORTIONAL = "proportional"
    COMPLEXITY = "complexity"
    CRITICALITY = "criticality"


@dataclass
class ComponentBudget:
    """Budget allocation for a single component."""
    reference: str
    component_type: str
    sheet_path: str

    # Actual values
    actual_lambda: float      # per hour
    actual_fit: float         # FIT (failures per 10^9 hours)

    # Budget values
    budget_lambda: float      # per hour
    budget_fit: float         # FIT

    # Status
    margin_fit: float         # budget_fit - actual_fit (positive = under budget)
    margin_percent: float     # margin_fit / budget_fit * 100
    within_budget: bool       # True if actual <= budget
    utilization: float        # actual / budget (0.0 to inf)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "component_type": self.component_type,
            "sheet_path": self.sheet_path,
            "actual_fit": self.actual_fit,
            "budget_fit": self.budget_fit,
            "margin_fit": self.margin_fit,
            "margin_percent": self.margin_percent,
            "within_budget": self.within_budget,
            "utilization": self.utilization,
        }


@dataclass
class SheetBudget:
    """Budget allocation for a schematic sheet."""
    sheet_path: str
    sheet_name: str

    actual_lambda: float
    actual_fit: float
    budget_lambda: float
    budget_fit: float

    margin_fit: float
    margin_percent: float
    within_budget: bool
    utilization: float

    n_components: int
    n_over_budget: int
    component_budgets: List[ComponentBudget] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sheet_path": self.sheet_path,
            "sheet_name": self.sheet_name,
            "actual_fit": self.actual_fit,
            "budget_fit": self.budget_fit,
            "margin_fit": self.margin_fit,
            "margin_percent": self.margin_percent,
            "within_budget": self.within_budget,
            "utilization": self.utilization,
            "n_components": self.n_components,
            "n_over_budget": self.n_over_budget,
            "components": [c.to_dict() for c in self.component_budgets],
        }


@dataclass
class BudgetAllocationResult:
    """Complete budget allocation result."""

    # Target
    target_reliability: float
    target_lambda: float          # /h
    target_fit: float             # FIT
    mission_hours: float
    strategy: str

    # System actual
    actual_reliability: float
    actual_lambda: float
    actual_fit: float

    # System-level status
    system_within_budget: bool
    system_margin_fit: float
    system_margin_percent: float

    # Detailed breakdown
    sheet_budgets: List[SheetBudget] = field(default_factory=list)
    total_components: int = 0
    components_over_budget: int = 0
    sheets_over_budget: int = 0

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_reliability": self.target_reliability,
            "target_fit": self.target_fit,
            "mission_hours": self.mission_hours,
            "strategy": self.strategy,
            "actual_reliability": self.actual_reliability,
            "actual_fit": self.actual_fit,
            "system_within_budget": self.system_within_budget,
            "system_margin_fit": self.system_margin_fit,
            "system_margin_percent": self.system_margin_percent,
            "total_components": self.total_components,
            "components_over_budget": self.components_over_budget,
            "sheets_over_budget": self.sheets_over_budget,
            "sheets": [s.to_dict() for s in self.sheet_budgets],
            "recommendations": self.recommendations,
        }


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def allocate_budget(
    sheet_data: Dict[str, Dict],
    mission_hours: float,
    target_reliability: float = 0.999,
    strategy: str = "proportional",
    active_sheets: Optional[List[str]] = None,
    margin_percent: float = 10.0,
) -> BudgetAllocationResult:
    """
    Allocate reliability budget across sheets and components.

    Args:
        sheet_data:         Dict of {sheet_path: {lambda, r, components: [...]}}
        mission_hours:      Total mission duration in hours
        target_reliability: System reliability target R (e.g. 0.999)
        strategy:           Allocation strategy: equal, proportional, complexity, criticality
        active_sheets:      If provided, only these sheets are considered
        margin_percent:     Design margin to subtract from budget (default 10%)

    Returns:
        BudgetAllocationResult with complete breakdown
    """
    try:
        from .reliability_math import reliability_from_lambda, lambda_from_reliability
    except ImportError:
        from reliability_math import reliability_from_lambda, lambda_from_reliability

    # Filter to active sheets
    if active_sheets:
        filtered = {k: v for k, v in sheet_data.items() if k in active_sheets}
    else:
        filtered = dict(sheet_data)

    # Compute system target lambda from reliability target
    target_lambda = lambda_from_reliability(target_reliability, mission_hours)
    target_fit = target_lambda * 1e9

    # Apply design margin: reduce available budget by margin_percent
    available_fit = target_fit * (1.0 - margin_percent / 100.0)
    available_lambda = available_fit * 1e-9

    # Compute actual system values
    actual_lambda = sum(_safe_float(d.get("lambda", 0)) for d in filtered.values())
    actual_fit = actual_lambda * 1e9
    actual_r = reliability_from_lambda(actual_lambda, mission_hours)

    # Collect all components
    all_components = []
    for path, data in filtered.items():
        for comp in data.get("components", []):
            all_components.append({
                "ref": comp.get("ref", "?"),
                "class": comp.get("class", "Unknown"),
                "lambda": _safe_float(comp.get("lambda", 0)),
                "sheet": path,
                "params": comp.get("params", {}),
            })

    n_total = len(all_components)
    n_sheets = len(filtered)

    # ---- Strategy dispatch ----

    if strategy == "equal":
        sheet_budgets = _allocate_equal(
            filtered, all_components, available_lambda, available_fit,
            mission_hours, n_total)
    elif strategy == "complexity":
        sheet_budgets = _allocate_complexity(
            filtered, all_components, available_lambda, available_fit,
            mission_hours, n_total)
    elif strategy == "criticality":
        sheet_budgets = _allocate_criticality(
            filtered, all_components, available_lambda, available_fit,
            mission_hours, n_total)
    else:  # proportional (default, ARINC-style)
        sheet_budgets = _allocate_proportional(
            filtered, all_components, available_lambda, available_fit,
            mission_hours, n_total)

    # Compute aggregate statistics
    n_over = sum(1 for sb in sheet_budgets for cb in sb.component_budgets if not cb.within_budget)
    sheets_over = sum(1 for sb in sheet_budgets if not sb.within_budget)

    system_within = actual_fit <= target_fit
    system_margin_fit = target_fit - actual_fit
    system_margin_pct = (system_margin_fit / target_fit * 100) if target_fit > 0 else 0

    # Generate recommendations
    recs = _generate_recommendations(
        sheet_budgets, system_within, system_margin_fit, target_fit, actual_fit)

    return BudgetAllocationResult(
        target_reliability=target_reliability,
        target_lambda=target_lambda,
        target_fit=target_fit,
        mission_hours=mission_hours,
        strategy=strategy,
        actual_reliability=actual_r,
        actual_lambda=actual_lambda,
        actual_fit=actual_fit,
        system_within_budget=system_within,
        system_margin_fit=system_margin_fit,
        system_margin_percent=system_margin_pct,
        sheet_budgets=sheet_budgets,
        total_components=n_total,
        components_over_budget=n_over,
        sheets_over_budget=sheets_over,
        recommendations=recs,
    )


def _allocate_equal(filtered, all_components, available_lambda, available_fit,
                    mission_hours, n_total):
    """Equal apportionment: each component gets the same budget."""
    per_comp_fit = available_fit / max(n_total, 1)
    per_comp_lambda = available_lambda / max(n_total, 1)

    return _build_sheet_budgets(
        filtered, per_comp_lambda, per_comp_fit, mission_hours,
        strategy_fn=lambda comp, total_in_sheet: (per_comp_lambda, per_comp_fit)
    )


def _allocate_proportional(filtered, all_components, available_lambda, available_fit,
                           mission_hours, n_total):
    """Proportional (ARINC): budget proportional to actual failure rate.

    Components that currently consume more lambda get proportionally more budget.
    This preserves the existing design balance while scaling to meet the target.
    """
    total_actual = sum(c["lambda"] for c in all_components)
    if total_actual <= 0:
        return _allocate_equal(filtered, all_components, available_lambda, available_fit,
                               mission_hours, n_total)

    scale = available_lambda / total_actual

    def strategy_fn(comp, total_in_sheet):
        comp_lambda = _safe_float(comp.get("lambda", 0))
        budg_lambda = comp_lambda * scale
        budg_fit = budg_lambda * 1e9
        return budg_lambda, budg_fit

    return _build_sheet_budgets(filtered, 0, 0, mission_hours, strategy_fn=strategy_fn)


def _allocate_complexity(filtered, all_components, available_lambda, available_fit,
                         mission_hours, n_total):
    """Complexity-weighted: budget proportional to component count per sheet.

    Sheets with more components get more budget. Within a sheet,
    budget is distributed equally among components.
    """
    sheet_counts = {}
    for path, data in filtered.items():
        sheet_counts[path] = len(data.get("components", []))

    total_count = sum(sheet_counts.values())
    if total_count <= 0:
        return _allocate_equal(filtered, all_components, available_lambda, available_fit,
                               mission_hours, n_total)

    def strategy_fn(comp, total_in_sheet):
        if total_in_sheet <= 0:
            return 0, 0
        sheet_frac = total_in_sheet / total_count
        sheet_budget_lambda = available_lambda * sheet_frac
        per_comp_lambda = sheet_budget_lambda / total_in_sheet
        return per_comp_lambda, per_comp_lambda * 1e9

    return _build_sheet_budgets(filtered, 0, 0, mission_hours, strategy_fn=strategy_fn)


def _allocate_criticality(filtered, all_components, available_lambda, available_fit,
                          mission_hours, n_total):
    """Criticality-weighted: components with higher failure rates get tighter budgets.

    The inverse of proportional: high-FIT components are considered more critical
    and get proportionally LESS budget headroom, incentivizing their improvement.

    Budget_i = available * (1/lambda_i) / SUM(1/lambda_j)  for lambda_i > 0
    """
    inv_lambdas = {}
    for c in all_components:
        lam = _safe_float(c.get("lambda", 0))
        if lam > 0:
            inv_lambdas[c["ref"]] = 1.0 / lam
        else:
            inv_lambdas[c["ref"]] = 1e15  # very large for zero-lambda

    total_inv = sum(inv_lambdas.values())
    if total_inv <= 0:
        return _allocate_equal(filtered, all_components, available_lambda, available_fit,
                               mission_hours, n_total)

    def strategy_fn(comp, total_in_sheet):
        ref = comp.get("ref", "?")
        inv_l = inv_lambdas.get(ref, 1e15)
        frac = inv_l / total_inv
        budg_lambda = available_lambda * frac
        return budg_lambda, budg_lambda * 1e9

    return _build_sheet_budgets(filtered, 0, 0, mission_hours, strategy_fn=strategy_fn)


def _build_sheet_budgets(filtered, default_comp_lambda, default_comp_fit,
                         mission_hours, strategy_fn=None):
    """Build SheetBudget list from strategy function."""
    sheet_budgets = []

    for path, data in filtered.items():
        components = data.get("components", [])
        n_comps = len(components)
        sheet_actual_lambda = _safe_float(data.get("lambda", 0))
        sheet_actual_fit = sheet_actual_lambda * 1e9

        comp_budgets = []
        sheet_budget_lambda = 0.0

        for comp in components:
            comp_lambda = _safe_float(comp.get("lambda", 0))
            comp_fit = comp_lambda * 1e9

            if strategy_fn:
                budg_lambda, budg_fit = strategy_fn(comp, n_comps)
            else:
                budg_lambda, budg_fit = default_comp_lambda, default_comp_fit

            sheet_budget_lambda += budg_lambda

            margin_fit = budg_fit - comp_fit
            margin_pct = (margin_fit / budg_fit * 100) if budg_fit > 0 else 0
            within = comp_fit <= budg_fit
            utilization = comp_fit / budg_fit if budg_fit > 0 else float('inf')

            comp_budgets.append(ComponentBudget(
                reference=comp.get("ref", "?"),
                component_type=comp.get("class", "Unknown"),
                sheet_path=path,
                actual_lambda=comp_lambda,
                actual_fit=comp_fit,
                budget_lambda=budg_lambda,
                budget_fit=budg_fit,
                margin_fit=margin_fit,
                margin_percent=margin_pct,
                within_budget=within,
                utilization=utilization,
            ))

        sheet_budget_fit = sheet_budget_lambda * 1e9
        s_margin_fit = sheet_budget_fit - sheet_actual_fit
        s_margin_pct = (s_margin_fit / sheet_budget_fit * 100) if sheet_budget_fit > 0 else 0
        s_within = sheet_actual_fit <= sheet_budget_fit
        s_util = sheet_actual_fit / sheet_budget_fit if sheet_budget_fit > 0 else float('inf')
        n_over = sum(1 for cb in comp_budgets if not cb.within_budget)

        name = path.rstrip("/").split("/")[-1] or "Root"
        sheet_budgets.append(SheetBudget(
            sheet_path=path,
            sheet_name=name,
            actual_lambda=sheet_actual_lambda,
            actual_fit=sheet_actual_fit,
            budget_lambda=sheet_budget_lambda,
            budget_fit=sheet_budget_fit,
            margin_fit=s_margin_fit,
            margin_percent=s_margin_pct,
            within_budget=s_within,
            utilization=s_util,
            n_components=n_comps,
            n_over_budget=n_over,
            component_budgets=comp_budgets,
        ))

    return sheet_budgets


def _generate_recommendations(
    sheet_budgets: List[SheetBudget],
    system_within: bool,
    system_margin_fit: float,
    target_fit: float,
    actual_fit: float,
) -> List[str]:
    """Generate actionable recommendations from budget analysis."""
    recs = []

    if system_within:
        recs.append(
            f"System meets reliability target with {system_margin_fit:.1f} FIT margin "
            f"({system_margin_fit/target_fit*100:.1f}% headroom)."
        )
    else:
        excess = actual_fit - target_fit
        recs.append(
            f"SYSTEM EXCEEDS BUDGET by {excess:.1f} FIT "
            f"({excess/target_fit*100:.1f}% over target). Action required."
        )

    # Find worst offenders
    over_components = []
    for sb in sheet_budgets:
        for cb in sb.component_budgets:
            if not cb.within_budget:
                over_components.append(cb)

    over_components.sort(key=lambda c: c.margin_fit)  # most negative first

    for i, cb in enumerate(over_components[:5]):
        excess = abs(cb.margin_fit)
        recs.append(
            f"{i+1}. {cb.reference} ({cb.component_type}): "
            f"exceeds budget by {excess:.2f} FIT "
            f"(actual={cb.actual_fit:.2f}, budget={cb.budget_fit:.2f}). "
            f"Consider derating, package change, or component upgrade."
        )

    # Sheet-level observations
    for sb in sorted(sheet_budgets, key=lambda s: s.utilization, reverse=True):
        if sb.utilization > 0.9 and sb.within_budget:
            recs.append(
                f"Sheet '{sb.sheet_name}' at {sb.utilization*100:.0f}% of budget "
                f"({sb.n_over_budget} components over). Low margin -- monitor closely."
            )

    return recs


def compute_required_reliability_per_sheet(
    n_sheets: int,
    target_reliability: float,
) -> float:
    """For a series system, compute the per-sheet reliability requirement.

    R_system = R_sheet^n  =>  R_sheet = R_system^(1/n)
    """
    if n_sheets <= 0:
        return target_reliability
    return target_reliability ** (1.0 / n_sheets)


def compute_max_fit_for_target(
    target_reliability: float,
    mission_hours: float,
) -> float:
    """Compute the maximum total FIT allowed to meet a reliability target."""
    if target_reliability <= 0 or target_reliability >= 1:
        return 0.0
    lam = -math.log(target_reliability) / mission_hours
    return lam * 1e9
