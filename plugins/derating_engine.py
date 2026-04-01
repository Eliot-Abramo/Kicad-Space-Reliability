"""
Derating Guidance Engine
========================
Bridges diagnosis to prescription: given a component's reliability budget,
computes the required parameter values to meet it.

For each critical parameter identified by the criticality analysis,
this module computes:
  - The required parameter value to meet the component's FIT budget
  - The required reduction/change as a percentage
  - Concrete engineering recommendations

Inverse calculations supported:
  - T_junction  -> required junction temperature
  - delta_t     -> required thermal excursion
  - n_cycles    -> required thermal cycling rate
  - tau_on      -> required duty cycle reduction
  - v_applied   -> required voltage derating

Author:  Eliot Abramo
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class DeratingRecommendation:
    """A single derating recommendation for a component parameter."""
    reference: str
    component_type: str
    parameter: str
    current_value: float
    required_value: float
    change_absolute: float
    change_percent: float
    current_fit: float
    target_fit: float
    expected_fit: float
    system_fit_reduction: float
    system_fit_reduction_pct: float
    feasibility: str          # "easy", "moderate", "difficult", "infeasible"
    actions: List[str]        # Concrete engineering actions
    priority: int             # 1 = highest

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "component_type": self.component_type,
            "parameter": self.parameter,
            "current_value": self.current_value,
            "required_value": self.required_value,
            "change_absolute": self.change_absolute,
            "change_percent": self.change_percent,
            "current_fit": self.current_fit,
            "target_fit": self.target_fit,
            "expected_fit": self.expected_fit,
            "system_fit_reduction": self.system_fit_reduction,
            "system_fit_reduction_pct": self.system_fit_reduction_pct,
            "feasibility": self.feasibility,
            "actions": self.actions,
            "priority": self.priority,
        }


@dataclass
class DeratingResult:
    """Complete derating guidance for a design."""
    system_actual_fit: float
    system_target_fit: float
    system_gap_fit: float
    recommendations: List[DeratingRecommendation] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_actual_fit": self.system_actual_fit,
            "system_target_fit": self.system_target_fit,
            "system_gap_fit": self.system_gap_fit,
            "summary": self.summary,
            "recommendations": [r.to_dict() for r in self.recommendations],
        }


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# =========================================================================
# Inverse calculation: find required parameter value for target FIT
# =========================================================================

def _find_required_value(
    component_type: str,
    base_params: Dict,
    parameter: str,
    target_lambda: float,
    current_value: float,
    search_direction: str = "decrease",
    max_iterations: int = 50,
    tolerance: float = 0.01,
) -> Optional[float]:
    """Binary search for the parameter value that achieves target_lambda.

    Args:
        component_type: Component class name
        base_params:    Current parameter dictionary
        parameter:      Parameter name to vary
        target_lambda:  Target lambda value (per hour)
        current_value:  Current parameter value
        search_direction: "decrease" to search lower, "increase" to search higher
        max_iterations: Maximum bisection iterations
        tolerance:      Relative tolerance for convergence

    Returns:
        Required parameter value, or None if not achievable.
    """
    try:
        from .reliability_math import calculate_component_lambda
    except ImportError:
        from reliability_math import calculate_component_lambda

    # Define search bounds based on parameter type
    bounds = _get_parameter_bounds(parameter, current_value, search_direction)
    if bounds is None:
        return None

    lo, hi = bounds

    def eval_lambda(val):
        p = dict(base_params)
        p[parameter] = val
        try:
            result = calculate_component_lambda(component_type, p)
            return result.get("lambda_total", 0.0)
        except Exception:
            return float('inf')

    # Verify that the target is achievable within bounds
    lam_lo = eval_lambda(lo)
    lam_hi = eval_lambda(hi)

    if search_direction == "decrease":
        if lam_lo > target_lambda:
            return None  # Cannot achieve target even at minimum
    else:
        if lam_hi < target_lambda:
            return None

    # Binary search
    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        lam_mid = eval_lambda(mid)

        if abs(lam_mid - target_lambda) / max(target_lambda, 1e-15) < tolerance:
            return mid

        if search_direction == "decrease":
            # Lower value -> lower lambda (temperature, voltage, cycles)
            if lam_mid > target_lambda:
                hi = mid
            else:
                lo = mid
        else:
            # Higher value -> lower lambda (e.g., not typical, but for completeness)
            if lam_mid > target_lambda:
                lo = mid
            else:
                hi = mid

    return (lo + hi) / 2.0


def _get_parameter_bounds(parameter: str, current_value: float,
                          search_direction: str) -> Optional[Tuple[float, float]]:
    """Get physically meaningful search bounds for a parameter."""

    param_bounds = {
        "t_junction": (-40.0, current_value),
        "t_ambient": (-40.0, current_value),
        "delta_t": (0.1, current_value),
        "n_cycles": (1.0, current_value),
        "tau_on": (0.01, current_value),
        "v_applied": (0.0, current_value),
        "operating_power": (0.0, current_value),
        "voltage_stress_vds": (0.0, current_value),
        "voltage_stress_vgs": (0.0, current_value),
        "voltage_stress_vce": (0.0, current_value),
        "ripple_ratio": (0.0, current_value),
        "if_applied": (0.0, current_value),
        "contact_current_ratio": (0.0, current_value),
        "cycles_per_hour": (0.0, current_value),
    }

    bounds = param_bounds.get(parameter)
    if bounds is None:
        # Generic: search from 10% of current to current
        if current_value > 0:
            return (current_value * 0.01, current_value)
        return None

    return bounds


def _get_parameter_actions(parameter: str, current: float,
                           required: float, component_type: str) -> List[str]:
    """Generate concrete engineering actions for a parameter change."""

    change_pct = abs(current - required) / max(abs(current), 1e-12) * 100

    actions = {
        "t_junction": [
            f"Reduce junction temperature from {current:.0f} degC to {required:.0f} degC",
            "Add heatsink or improve thermal pad connection",
            "Increase copper pour area around component",
            "Reduce operating frequency / clock gating",
            "Consider package with exposed thermal pad (QFN, PowerPAK)",
            "Review PCB thermal via array under component",
        ],
        "t_ambient": [
            f"Reduce ambient temperature from {current:.0f} degC to {required:.0f} degC",
            "Improve enclosure ventilation or add forced-air cooling",
            "Relocate component away from heat sources",
            "Add thermal isolation between hot and cold zones",
        ],
        "delta_t": [
            f"Reduce thermal excursion from {current:.1f} degC to {required:.1f} degC",
            "Improve thermal mass near component (copper planes)",
            "Add conformal coating to reduce thermal shock",
            "Reduce power-cycling frequency",
        ],
        "n_cycles": [
            f"Reduce annual thermal cycles from {current:.0f} to {required:.0f}",
            "Implement soft-start / gradual power-up sequences",
            "Reduce on/off cycling (use sleep modes instead)",
            "Improve thermal management to reduce cycle amplitude",
        ],
        "tau_on": [
            f"Reduce duty cycle from {current:.2f} to {required:.2f}",
            "Implement duty-cycle management (time-slicing)",
            "Add redundant path for load sharing",
        ],
        "v_applied": [
            f"Reduce applied voltage from {current:.1f}V to {required:.1f}V",
            "Select higher voltage-rated component",
            "Add voltage regulation / clamping",
            "Derate voltage to {:.0f}% of rated".format(
                required / max(current, 1e-6) * 100),
        ],
        "voltage_stress_vds": [
            f"Reduce V_DS stress ratio from {current:.2f} to {required:.2f}",
            "Use higher V_DSS rated MOSFET",
            "Add snubber circuit for voltage spikes",
        ],
        "voltage_stress_vgs": [
            f"Reduce V_GS stress ratio from {current:.2f} to {required:.2f}",
            "Use higher V_GS(max) rated device",
            "Add gate voltage clamp (Zener)",
        ],
        "voltage_stress_vce": [
            f"Reduce V_CE stress ratio from {current:.2f} to {required:.2f}",
            "Use higher V_CEO rated BJT",
            "Add collector clamp circuit",
        ],
        "operating_power": [
            f"Reduce operating power from {current:.3f}W to {required:.3f}W",
            "Reduce operating current",
            "Use higher power-rated component",
        ],
    }

    result = actions.get(parameter, [
        f"Reduce {parameter} from {current} to {required}",
    ])

    # Only return relevant subset based on change magnitude
    if change_pct < 10:
        return result[:2]
    elif change_pct < 30:
        return result[:3]
    return result[:4]


def _assess_feasibility(parameter: str, change_percent: float) -> str:
    """Assess how feasible a derating recommendation is."""
    if change_percent < 5:
        return "easy"
    elif change_percent < 15:
        return "moderate"
    elif change_percent < 30:
        return "difficult"
    else:
        return "infeasible"


# =========================================================================
# Main derating analysis
# =========================================================================

def compute_derating_guidance(
    sheet_data: Dict[str, Dict],
    mission_hours: float,
    target_fit: float,
    criticality_results: Optional[List[Dict]] = None,
    active_sheets: Optional[List[str]] = None,
    top_n: int = 10,
) -> DeratingResult:
    """
    Compute derating guidance for all critical components.

    For each component exceeding its budget (or for the top-N most critical
    components), computes the required parameter changes to meet the target.

    Args:
        sheet_data:           System sheet data
        mission_hours:        Mission duration in hours
        target_fit:           System target FIT
        criticality_results:  Optional pre-computed criticality analysis
        active_sheets:        Optional sheet filter
        top_n:                Number of top components to analyze

    Returns:
        DeratingResult with ranked recommendations
    """
    try:
        from .reliability_math import (
            calculate_component_lambda, reliability_from_lambda,
            analyze_component_criticality,
        )
    except ImportError:
        from reliability_math import (
            calculate_component_lambda, reliability_from_lambda,
            analyze_component_criticality,
        )

    # Gather all components
    if active_sheets:
        filtered = {k: v for k, v in sheet_data.items() if k in active_sheets}
    else:
        filtered = dict(sheet_data)

    system_actual_fit = sum(_safe_float(d.get("lambda", 0)) for d in filtered.values()) * 1e9
    system_gap = system_actual_fit - target_fit

    all_comps = []
    for path, data in filtered.items():
        for comp in data.get("components", []):
            if comp.get("override_lambda") is not None:
                continue  # Skip fixed-lambda components
            all_comps.append(comp)

    # Sort by FIT contribution (highest first)
    all_comps.sort(key=lambda c: _safe_float(c.get("lambda", 0)), reverse=True)

    recommendations = []
    priority = 1

    for comp in all_comps[:top_n]:
        ref = comp.get("ref", "?")
        comp_type = comp.get("class", "Unknown")
        params = comp.get("params", {})
        comp_lambda = _safe_float(comp.get("lambda", 0))
        comp_fit = comp_lambda * 1e9

        if comp_lambda <= 0 or not params:
            continue

        # Run criticality to find most sensitive parameters
        try:
            crit = analyze_component_criticality(comp_type, params, mission_hours)
        except Exception:
            continue

        if not crit:
            continue

        # For each critical parameter, compute required value
        for crit_entry in crit[:3]:  # Top 3 parameters per component
            field_name = crit_entry.get("field", "")
            current_val = crit_entry.get("nominal_value", 0)
            elasticity = crit_entry.get("sensitivity", 0)
            impact_pct = crit_entry.get("impact_percent", 0)

            if abs(elasticity) < 0.01 or current_val == 0:
                continue

            # Target: reduce this component's lambda by its proportional share of gap
            if system_gap > 0:
                # Component should reduce by proportion of its contribution
                comp_frac = comp_fit / system_actual_fit if system_actual_fit > 0 else 0
                needed_reduction_fit = system_gap * comp_frac
                comp_target_fit = max(0.01, comp_fit - needed_reduction_fit)
            else:
                # System is within budget; still show optimization opportunities
                comp_target_fit = comp_fit * 0.8  # 20% improvement target

            comp_target_lambda = comp_target_fit * 1e-9

            # Find required parameter value
            required = _find_required_value(
                comp_type, params, field_name, comp_target_lambda,
                current_val, search_direction="decrease"
            )

            if required is None:
                continue

            change_abs = current_val - required
            change_pct = abs(change_abs) / max(abs(current_val), 1e-12) * 100

            if change_pct < 0.1:
                continue  # Negligible change

            # Verify the expected FIT after change
            params_after = dict(params)
            params_after[field_name] = required
            try:
                result_after = calculate_component_lambda(comp_type, params_after)
                expected_fit = result_after.get("lambda_total", 0) * 1e9
            except Exception:
                expected_fit = comp_fit

            sys_reduction = comp_fit - expected_fit
            sys_reduction_pct = (sys_reduction / system_actual_fit * 100) if system_actual_fit > 0 else 0

            feasibility = _assess_feasibility(field_name, change_pct)
            actions = _get_parameter_actions(field_name, current_val, required, comp_type)

            recommendations.append(DeratingRecommendation(
                reference=ref,
                component_type=comp_type,
                parameter=field_name,
                current_value=current_val,
                required_value=required,
                change_absolute=change_abs,
                change_percent=change_pct,
                current_fit=comp_fit,
                target_fit=comp_target_fit,
                expected_fit=expected_fit,
                system_fit_reduction=sys_reduction,
                system_fit_reduction_pct=sys_reduction_pct,
                feasibility=feasibility,
                actions=actions,
                priority=priority,
            ))

        priority += 1

    # Sort by system impact (highest reduction first)
    recommendations.sort(key=lambda r: -r.system_fit_reduction)

    # Re-assign priorities after sorting
    for i, rec in enumerate(recommendations):
        rec.priority = i + 1

    # Build summary
    total_potential = sum(r.system_fit_reduction for r in recommendations)
    summary_lines = [
        f"Derating analysis identified {len(recommendations)} improvement opportunities.",
        f"Total potential system FIT reduction: {total_potential:.1f} FIT "
        f"({total_potential/system_actual_fit*100:.1f}% of current)." if system_actual_fit > 0 else "",
    ]

    if system_gap > 0:
        if total_potential >= system_gap:
            summary_lines.append(
                f"Implementing all recommendations would close the "
                f"{system_gap:.1f} FIT gap to target."
            )
        else:
            summary_lines.append(
                f"Recommendations cover {total_potential:.1f} of the "
                f"{system_gap:.1f} FIT gap. Additional design changes needed."
            )
    else:
        summary_lines.append("System is within budget. Recommendations are for further optimization.")

    return DeratingResult(
        system_actual_fit=system_actual_fit,
        system_target_fit=target_fit,
        system_gap_fit=system_gap,
        recommendations=recommendations,
        summary="\n".join(s for s in summary_lines if s),
    )
