"""
Component Swap Analysis Module
===============================
"Clone & Modify" workflow: select a component, change one parameter
(package, IC type, or technology), and immediately see the reliability delta.

Supports:
  - Package swap (e.g., QFP-100 -> QFN-48)
  - IC type swap (e.g., MOS Digital -> BiCMOS Digital)
  - Discrete type swap (e.g., Silicon MOSFET -> GaN HEMT)
  - Technology upgrade (e.g., commercial -> mil-grade)
  - Multi-parameter what-if

Author:  Eliot Abramo
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy


@dataclass
class SwapCandidate:
    """A potential component swap option."""
    name: str               # Human-readable description
    parameter: str          # Which parameter changes
    old_value: Any          # Current value
    new_value: Any          # Proposed value
    lambda_before: float    # FIT before swap
    lambda_after: float     # FIT after swap
    delta_fit: float        # Change in FIT (negative = improvement)
    delta_percent: float    # Percentage change
    improvement: bool       # True if reliability improves

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameter": self.parameter,
            "old_value": str(self.old_value),
            "new_value": str(self.new_value),
            "lambda_before_fit": self.lambda_before,
            "lambda_after_fit": self.lambda_after,
            "delta_fit": self.delta_fit,
            "delta_percent": self.delta_percent,
            "improvement": self.improvement,
        }


@dataclass
class SwapAnalysisResult:
    """Result of component swap analysis."""
    reference: str
    component_type: str
    current_fit: float
    candidates: List[SwapCandidate] = field(default_factory=list)
    best_candidate: Optional[SwapCandidate] = None
    system_fit_before: float = 0.0
    system_fit_after_best: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "component_type": self.component_type,
            "current_fit": self.current_fit,
            "candidates": [c.to_dict() for c in self.candidates],
            "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            "system_fit_before": self.system_fit_before,
            "system_fit_after_best": self.system_fit_after_best,
        }


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _calc_lambda(component_type, params):
    """Calculate lambda, returning FIT value."""
    try:
        from .reliability_math import calculate_component_lambda
    except ImportError:
        from reliability_math import calculate_component_lambda
    try:
        result = calculate_component_lambda(component_type, params)
        return result.get("lambda_total", 0.0) * 1e9  # Return FIT
    except Exception:
        return 0.0


def _get_swap_options(component_type, params):
    """Get available swap options for a component type."""
    try:
        from .reliability_math import (
            IC_PACKAGE_CHOICES, IC_TYPE_CHOICES,
            DIODE_BASE_RATES, TRANSISTOR_BASE_RATES,
            DISCRETE_PACKAGE_TABLE, CAPACITOR_PARAMS,
            RESISTOR_PARAMS, INDUCTOR_PARAMS,
            THERMAL_EXPANSION_SUBSTRATE,
        )
    except ImportError:
        from reliability_math import (
            IC_PACKAGE_CHOICES, IC_TYPE_CHOICES,
            DIODE_BASE_RATES, TRANSISTOR_BASE_RATES,
            DISCRETE_PACKAGE_TABLE, CAPACITOR_PARAMS,
            RESISTOR_PARAMS, INDUCTOR_PARAMS,
            THERMAL_EXPANSION_SUBSTRATE,
        )

    options = {}

    if component_type == "Integrated Circuit":
        options["package"] = list(IC_PACKAGE_CHOICES.keys())
        options["ic_type"] = list(IC_TYPE_CHOICES.keys())
        options["substrate"] = list(THERMAL_EXPANSION_SUBSTRATE.keys())
    elif component_type == "Diode":
        options["diode_type"] = list(DIODE_BASE_RATES.keys())
        options["package"] = list(DISCRETE_PACKAGE_TABLE.keys())
    elif component_type == "Transistor":
        options["transistor_type"] = list(TRANSISTOR_BASE_RATES.keys())
        options["package"] = list(DISCRETE_PACKAGE_TABLE.keys())
    elif component_type == "Capacitor":
        options["capacitor_type"] = list(CAPACITOR_PARAMS.keys())
    elif component_type == "Resistor":
        options["resistor_type"] = list(RESISTOR_PARAMS.keys())
    elif component_type == "Inductor/Transformer":
        options["inductor_type"] = list(INDUCTOR_PARAMS.keys())

    return options


def analyze_package_swaps(
    component_type: str,
    base_params: Dict[str, Any],
    reference: str = "?",
    system_total_fit: float = 0.0,
) -> SwapAnalysisResult:
    """Analyze all possible package swaps for a component.

    For ICs, iterates through all IC_PACKAGE_CHOICES.
    For discretes, iterates through DISCRETE_PACKAGE_TABLE.
    """
    current_fit = _calc_lambda(component_type, base_params)
    candidates = []

    swap_options = _get_swap_options(component_type, base_params)

    # Package swaps
    if "package" in swap_options:
        current_pkg = base_params.get("package", "")
        for pkg in swap_options["package"]:
            if pkg == current_pkg:
                continue
            test_params = dict(base_params)
            test_params["package"] = pkg
            new_fit = _calc_lambda(component_type, test_params)
            delta = new_fit - current_fit
            delta_pct = (delta / current_fit * 100) if current_fit > 0 else 0

            candidates.append(SwapCandidate(
                name=f"Package: {pkg}",
                parameter="package",
                old_value=current_pkg,
                new_value=pkg,
                lambda_before=current_fit,
                lambda_after=new_fit,
                delta_fit=delta,
                delta_percent=delta_pct,
                improvement=delta < 0,
            ))

    # Sort by improvement (best first)
    candidates.sort(key=lambda c: c.delta_fit)

    best = candidates[0] if candidates and candidates[0].improvement else None

    return SwapAnalysisResult(
        reference=reference,
        component_type=component_type,
        current_fit=current_fit,
        candidates=candidates,
        best_candidate=best,
        system_fit_before=system_total_fit,
        system_fit_after_best=(system_total_fit + best.delta_fit) if best else system_total_fit,
    )


def analyze_type_swaps(
    component_type: str,
    base_params: Dict[str, Any],
    reference: str = "?",
    system_total_fit: float = 0.0,
) -> SwapAnalysisResult:
    """Analyze component subtype swaps (IC type, diode type, etc.)."""
    current_fit = _calc_lambda(component_type, base_params)
    candidates = []

    swap_options = _get_swap_options(component_type, base_params)

    # Type-specific swaps
    type_params = {
        "Integrated Circuit": "ic_type",
        "Diode": "diode_type",
        "Transistor": "transistor_type",
        "Capacitor": "capacitor_type",
        "Resistor": "resistor_type",
        "Inductor/Transformer": "inductor_type",
    }

    param_name = type_params.get(component_type)
    if param_name and param_name in swap_options:
        current_type = base_params.get(param_name, "")
        for new_type in swap_options[param_name]:
            if new_type == current_type:
                continue
            test_params = dict(base_params)
            test_params[param_name] = new_type
            new_fit = _calc_lambda(component_type, test_params)
            delta = new_fit - current_fit
            delta_pct = (delta / current_fit * 100) if current_fit > 0 else 0

            candidates.append(SwapCandidate(
                name=f"{param_name}: {new_type}",
                parameter=param_name,
                old_value=current_type,
                new_value=new_type,
                lambda_before=current_fit,
                lambda_after=new_fit,
                delta_fit=delta,
                delta_percent=delta_pct,
                improvement=delta < 0,
            ))

    candidates.sort(key=lambda c: c.delta_fit)
    best = candidates[0] if candidates and candidates[0].improvement else None

    return SwapAnalysisResult(
        reference=reference,
        component_type=component_type,
        current_fit=current_fit,
        candidates=candidates,
        best_candidate=best,
        system_fit_before=system_total_fit,
        system_fit_after_best=(system_total_fit + best.delta_fit) if best else system_total_fit,
    )


def analyze_custom_swap(
    component_type: str,
    base_params: Dict[str, Any],
    swap_params: Dict[str, Any],
    reference: str = "?",
    system_total_fit: float = 0.0,
    swap_name: str = "Custom swap",
) -> SwapAnalysisResult:
    """Analyze a custom multi-parameter swap.

    swap_params: dict of {param_name: new_value} to override.
    """
    current_fit = _calc_lambda(component_type, base_params)

    test_params = dict(base_params)
    test_params.update(swap_params)
    new_fit = _calc_lambda(component_type, test_params)

    delta = new_fit - current_fit
    delta_pct = (delta / current_fit * 100) if current_fit > 0 else 0

    changes_desc = ", ".join(f"{k}: {base_params.get(k, '?')} -> {v}" for k, v in swap_params.items())

    candidate = SwapCandidate(
        name=swap_name,
        parameter="multiple" if len(swap_params) > 1 else list(swap_params.keys())[0],
        old_value=changes_desc,
        new_value=changes_desc,
        lambda_before=current_fit,
        lambda_after=new_fit,
        delta_fit=delta,
        delta_percent=delta_pct,
        improvement=delta < 0,
    )

    return SwapAnalysisResult(
        reference=reference,
        component_type=component_type,
        current_fit=current_fit,
        candidates=[candidate],
        best_candidate=candidate if candidate.improvement else None,
        system_fit_before=system_total_fit,
        system_fit_after_best=(system_total_fit + delta) if delta < 0 else system_total_fit,
    )


def quick_swap_comparison(
    component_type: str,
    base_params: Dict[str, Any],
    parameter: str,
    new_value: Any,
    reference: str = "?",
) -> Dict[str, Any]:
    """Quick single-parameter swap comparison.

    Returns a simple dict with before/after/delta for immediate display.
    """
    current_fit = _calc_lambda(component_type, base_params)
    old_value = base_params.get(parameter, "")

    test_params = dict(base_params)
    test_params[parameter] = new_value
    new_fit = _calc_lambda(component_type, test_params)

    delta = new_fit - current_fit
    delta_pct = (delta / current_fit * 100) if current_fit > 0 else 0

    return {
        "reference": reference,
        "parameter": parameter,
        "old_value": old_value,
        "new_value": new_value,
        "fit_before": current_fit,
        "fit_after": new_fit,
        "delta_fit": delta,
        "delta_percent": delta_pct,
        "improvement": delta < 0,
    }


def rank_all_swaps(
    components: List[Dict],
    system_total_fit: float = 0.0,
    max_per_component: int = 5,
) -> List[Dict]:
    """Rank ALL possible single-parameter swaps across all components.

    Returns a flat list sorted by system-level FIT improvement.
    Useful for answering: "What single change would most improve reliability?"
    """
    all_improvements = []

    for comp in components:
        ref = comp.get("ref", "?")
        comp_type = comp.get("class", "Unknown")
        params = comp.get("params", {})

        if comp.get("override_lambda") is not None:
            continue  # Skip fixed components

        # Package swaps
        pkg_result = analyze_package_swaps(comp_type, params, ref, system_total_fit)
        for c in pkg_result.candidates[:max_per_component]:
            if c.improvement:
                all_improvements.append({
                    "reference": ref,
                    "component_type": comp_type,
                    "swap_type": "package",
                    "description": c.name,
                    "delta_fit": c.delta_fit,
                    "delta_percent": c.delta_percent,
                    "new_system_fit": system_total_fit + c.delta_fit,
                })

        # Type swaps
        type_result = analyze_type_swaps(comp_type, params, ref, system_total_fit)
        for c in type_result.candidates[:max_per_component]:
            if c.improvement:
                all_improvements.append({
                    "reference": ref,
                    "component_type": comp_type,
                    "swap_type": "type",
                    "description": c.name,
                    "delta_fit": c.delta_fit,
                    "delta_percent": c.delta_percent,
                    "new_system_fit": system_total_fit + c.delta_fit,
                })

    # Sort by FIT improvement (most negative delta first)
    all_improvements.sort(key=lambda x: x["delta_fit"])
    return all_improvements
