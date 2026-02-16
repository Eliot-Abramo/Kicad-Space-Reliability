"""
Sensitivity Analysis Module
============================
Deterministic sensitivity analysis for reliability co-design.

Provides three complementary views, each answering a distinct
engineering question:

  1. Tornado (OAT) Sensitivity -- IEC 60300-3-1 compliant
     "Which design parameter has the greatest leverage on system FIT?"
     Perturbs each parameter +/- X% across all components and
     measures the resulting change in system failure rate.

  2. Design-Margin / What-If Analysis
     "What happens to my FIT budget if ambient temperature rises 10 C?"
     Predefined scenarios that map directly to design-review actions.

  3. Component-Level Criticality
     "For my top-N worst components, which parameter drives their lambda?"
     Per-component field-level elasticity ranking.

Sobol variance-based sensitivity was deliberately excluded because it
provides no additional insight for additive reliability models
(lambda_total = SUM lambda_i).  By construction, inter-component
interactions are zero in a series reliability model, so Sobol first-order
and total-order indices are identical and proportional to the contribution
percentages already available on the Contributions tab.

Author:  Eliot Abramo
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class TornadoEntry:
    """Single entry in a tornado chart."""
    name: str
    base_value: float
    low_value: float
    high_value: float
    delta_low: float
    delta_high: float
    swing: float
    unit: str = "FIT"


@dataclass
class TornadoResult:
    """Results from tornado sensitivity analysis."""
    entries: List[TornadoEntry]
    base_lambda_fit: float
    base_reliability: float
    perturbation_pct: float
    mission_hours: float

    def to_dict(self) -> Dict:
        return {
            "base_lambda_fit": self.base_lambda_fit,
            "base_reliability": self.base_reliability,
            "perturbation_pct": self.perturbation_pct,
            "mission_hours": self.mission_hours,
            "entries": [
                {
                    "name": e.name, "base_value": e.base_value,
                    "low_value": e.low_value, "high_value": e.high_value,
                    "delta_low": e.delta_low, "delta_high": e.delta_high,
                    "swing": e.swing, "unit": e.unit,
                }
                for e in self.entries
            ],
        }


@dataclass
class DesignMarginEntry:
    """Single what-if scenario result."""
    scenario_name: str
    description: str
    lambda_fit: float
    reliability: float
    delta_lambda_pct: float
    delta_reliability: float


@dataclass
class DesignMarginResult:
    """Results from design-margin / what-if analysis."""
    baseline_lambda_fit: float
    baseline_reliability: float
    mission_hours: float
    scenarios: List[DesignMarginEntry]

    def to_dict(self) -> Dict:
        return {
            "baseline_lambda_fit": self.baseline_lambda_fit,
            "baseline_reliability": self.baseline_reliability,
            "mission_hours": self.mission_hours,
            "scenarios": [
                {
                    "name": s.scenario_name, "description": s.description,
                    "lambda_fit": s.lambda_fit, "reliability": s.reliability,
                    "delta_lambda_pct": s.delta_lambda_pct,
                    "delta_reliability": s.delta_reliability,
                }
                for s in self.scenarios
            ],
        }


# =============================================================================
# Tornado Sensitivity (sheet-level)
# =============================================================================

def tornado_sheet_sensitivity(
    sheet_data: Dict[str, Dict],
    mission_hours: float,
    perturbation: float = 0.20,
    active_sheets: List[str] = None,
) -> TornadoResult:
    """Tornado: perturb each sheet's lambda by +/-perturbation, measure system impact.
    Only considers sheets in active_sheets if provided."""
    try:
        from .reliability_math import reliability_from_lambda
    except ImportError:
        from reliability_math import reliability_from_lambda

    filtered = {k: v for k, v in sheet_data.items() if k in active_sheets} if active_sheets else sheet_data
    base_lambda = sum(d.get("lambda", 0) for d in filtered.values())
    base_r = reliability_from_lambda(base_lambda, mission_hours)
    base_fit = base_lambda * 1e9

    entries = []
    for path, data in filtered.items():
        lam = data.get("lambda", 0)
        if lam <= 0:
            continue
        name = path.rstrip("/").split("/")[-1] or "Root"
        delta = lam * perturbation
        low_fit = (base_lambda - delta) * 1e9
        high_fit = (base_lambda + delta) * 1e9
        entries.append(TornadoEntry(
            name=name, base_value=base_fit,
            low_value=low_fit, high_value=high_fit,
            delta_low=base_fit - low_fit, delta_high=high_fit - base_fit,
            swing=high_fit - low_fit, unit="FIT"))

    entries.sort(key=lambda e: -e.swing)
    return TornadoResult(entries=entries, base_lambda_fit=base_fit,
                         base_reliability=base_r, perturbation_pct=perturbation * 100,
                         mission_hours=mission_hours)


# =============================================================================
# Tornado Sensitivity (parameter-level)
# =============================================================================

def tornado_parameter_sensitivity(
    sheet_data: Dict[str, Dict],
    mission_hours: float,
    perturbation: float = 0.20,
    active_sheets: List[str] = None,
    target_fields: Dict[str, List[str]] = None,
) -> TornadoResult:
    """Tornado at PARAMETER level: perturb a design parameter across ALL components.

    For each unique parameter name found across all components (e.g. t_junction,
    delta_t, n_cycles), applies +/- perturbation to EVERY component that uses it,
    recalculates each component's lambda, and measures the resulting system FIT change.

    This answers: "If T_junction is 20% higher everywhere, how does system FIT change?"

    Parameters
    ----------
    target_fields : dict, optional
        {component_type: [field_names]} restricts which fields to analyze.
        If None, all numeric fields are analyzed.
    """
    try:
        from .reliability_math import calculate_component_lambda, reliability_from_lambda
    except ImportError:
        from reliability_math import calculate_component_lambda, reliability_from_lambda

    filtered = {k: v for k, v in sheet_data.items() if k in active_sheets} if active_sheets else sheet_data
    all_comps = []
    for data in filtered.values():
        for comp in data.get("components", []):
            if comp.get("override_lambda") is None:
                all_comps.append(comp)

    base_lambda = sum(d.get("lambda", 0) for d in filtered.values())
    base_fit = base_lambda * 1e9
    base_r = reliability_from_lambda(base_lambda, mission_hours)

    # Collect all (parameter_name -> [(comp_index, nominal_value)]) mappings
    param_usage = {}
    for i, comp in enumerate(all_comps):
        comp_type = comp.get("class", "")
        for pname, pval in comp.get("params", {}).items():
            if pname.startswith("_"):
                continue
            try:
                v = float(pval)
            except (TypeError, ValueError):
                continue
            if v == 0:
                continue
            if target_fields is not None:
                if pname not in target_fields.get(comp_type, []):
                    continue
            param_usage.setdefault(pname, []).append((i, v))

    entries = []
    for pname, usages in param_usage.items():
        d_low = d_high = 0.0
        for ci, nom in usages:
            comp = all_comps[ci]
            ct = comp.get("class", "Resistor")
            bp = comp.get("params", {}).copy()
            cl = comp.get("lambda", 0)
            dp = abs(nom * perturbation)
            if dp < 1e-12:
                continue
            bp_l = bp.copy(); bp_l[pname] = nom - dp
            bp_h = bp.copy(); bp_h[pname] = nom + dp
            try:
                ll = calculate_component_lambda(ct, bp_l).get("lambda_total", 0)
            except Exception:
                ll = cl
            try:
                lh = calculate_component_lambda(ct, bp_h).get("lambda_total", 0)
            except Exception:
                lh = cl
            d_low += (ll - cl)
            d_high += (lh - cl)

        low_fit = (base_lambda + d_low) * 1e9
        high_fit = (base_lambda + d_high) * 1e9
        swing = abs(high_fit - low_fit)
        if swing < 1e-6:
            continue
        entries.append(TornadoEntry(
            name=f"{pname} ({len(usages)} comps)", base_value=base_fit,
            low_value=low_fit, high_value=high_fit,
            delta_low=base_fit - low_fit, delta_high=high_fit - base_fit,
            swing=swing, unit="FIT"))

    entries.sort(key=lambda e: -e.swing)
    return TornadoResult(entries=entries, base_lambda_fit=base_fit,
                         base_reliability=base_r, perturbation_pct=perturbation * 100,
                         mission_hours=mission_hours)


# =============================================================================
# Design Margin / What-If Analysis
# =============================================================================

def design_margin_analysis(
    sheet_data: Dict[str, Dict],
    mission_hours: float,
    active_sheets: List[str] = None,
) -> DesignMarginResult:
    """Run predefined what-if scenarios for design-margin evaluation.

    Each scenario modifies one or more design parameters across all
    components and recalculates total system FIT and reliability.
    Scenarios are chosen to match common design-review questions.
    """
    try:
        from .reliability_math import calculate_component_lambda, reliability_from_lambda
    except ImportError:
        from reliability_math import calculate_component_lambda, reliability_from_lambda

    filtered = {k: v for k, v in sheet_data.items() if k in active_sheets} if active_sheets else sheet_data
    base_lambda = sum(d.get("lambda", 0) for d in filtered.values())
    base_fit = base_lambda * 1e9
    base_r = reliability_from_lambda(base_lambda, mission_hours)

    def _run(mods):
        total = 0.0
        for data in filtered.values():
            for comp in data.get("components", []):
                ovr = comp.get("override_lambda")
                if ovr is not None:
                    total += ovr; continue
                ct = comp.get("class", "Resistor")
                p = comp.get("params", {}).copy()
                for pn, fn in mods.items():
                    if pn in p:
                        try:
                            p[pn] = fn(float(p[pn]))
                        except (TypeError, ValueError):
                            pass
                try:
                    total += calculate_component_lambda(ct, p).get("lambda_total", 0)
                except Exception:
                    total += comp.get("lambda", 0)
        return total

    scenarios = [
        ("Temp +10 C", "All junction/ambient temperatures +10 C",
         {"t_ambient": lambda v: v+10, "t_junction": lambda v: v+10, "temperature_c": lambda v: v+10}),
        ("Temp +20 C", "All junction/ambient temperatures +20 C",
         {"t_ambient": lambda v: v+20, "t_junction": lambda v: v+20, "temperature_c": lambda v: v+20}),
        ("Temp -10 C", "All temperatures -10 C (improved cooling)",
         {"t_ambient": lambda v: v-10, "t_junction": lambda v: v-10, "temperature_c": lambda v: v-10}),
        ("Thermal cycles x2", "Double annual thermal cycling count",
         {"n_cycles": lambda v: v*2}),
        ("Thermal cycles /2", "Halve annual thermal cycling (improved thermal design)",
         {"n_cycles": lambda v: v*0.5}),
        ("Delta-T x2", "Double thermal cycling amplitude",
         {"delta_t": lambda v: v*2}),
        ("50% duty cycle", "All components at tau_on = 0.5",
         {"tau_on": lambda _: 0.5}),
    ]

    results = []
    for name, desc, mods in scenarios:
        try:
            sl = _run(mods)
            sf = sl * 1e9
            sr = reliability_from_lambda(sl, mission_hours)
            dp = ((sf - base_fit) / base_fit * 100) if base_fit > 0 else 0
            results.append(DesignMarginEntry(name, desc, sf, sr, dp, sr - base_r))
        except Exception:
            continue

    return DesignMarginResult(base_fit, base_r, mission_hours, results)


# =============================================================================
# Component-Level Criticality
# =============================================================================

def analyze_board_criticality(
    components: List[Dict],
    mission_hours: float = 8760.0,
    top_n: int = 10,
    perturbation: float = 0.1,
    target_fields: Dict[str, List[str]] = None,
) -> List[Dict]:
    """Run parameter criticality on highest-FIT components.

    For each of the top-N components by failure rate, perturbs each
    numeric parameter by +/- perturbation and computes the elasticity
    (normalised sensitivity) of lambda with respect to that parameter.

    Parameters
    ----------
    target_fields : dict, optional
        {category: [field_names]} restricts which fields to analyze.
    """
    try:
        from .reliability_math import analyze_component_criticality
    except ImportError:
        from reliability_math import analyze_component_criticality

    analyzable = [c for c in components if c.get("override_lambda") is None]
    sorted_comps = sorted(analyzable, key=lambda c: c.get("lambda", 0), reverse=True)
    results = []

    for comp in sorted_comps[:top_n]:
        ref = comp.get("ref", "?")
        comp_type = comp.get("class", "")
        params = comp.get("params", {})
        if not comp_type or not params:
            continue
        try:
            raw = analyze_component_criticality(comp_type, params, mission_hours, perturbation)
            if not raw:
                continue
            if target_fields is not None:
                allowed = target_fields.get(comp_type, [])
                if allowed:
                    raw = [r for r in raw if r["field"] in allowed]
            if not raw:
                continue
            base_fit = raw[0]["lambda_nominal_fit"]
            fields = [{"name": r["field"], "value": r["nominal_value"],
                       "elasticity": r["sensitivity"], "impact_pct": r["impact_percent"]} for r in raw]
            results.append({"reference": ref, "component_type": comp_type,
                            "base_lambda_fit": base_fit, "fields": fields})
        except Exception:
            continue
    return results


# =============================================================================
# Helpers
# =============================================================================

def get_active_sheet_paths(blocks) -> Optional[List[str]]:
    """Extract sheet paths from block editor blocks dict.
    Returns None if no blocks (= use all sheets)."""
    if not blocks:
        return None
    active = []
    for bid, block in blocks.items():
        if bid.startswith("__"):
            continue
        if not getattr(block, 'is_group', False):
            name = getattr(block, 'name', '')
            if name:
                active.append(name)
    return active if active else None
