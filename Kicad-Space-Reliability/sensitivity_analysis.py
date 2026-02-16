"""
Sensitivity & Design Exploration Module
=========================================
Unified deterministic sensitivity analysis for reliability co-design.

Three complementary methods on one module, each answering a distinct
engineering question:

1. Tornado (OAT) Sensitivity
   "Which design parameter has the greatest leverage on system FIT?"
   Perturbs each parameter by user-specified physical amounts (not
   arbitrary percentages) across all components and measures the
   resulting change in system failure rate.

2. What-If / Design-Margin Scenarios
   "What happens to my FIT budget if ambient temperature rises 10 C?"
   Predefined and user-defined scenarios that map to design actions.
   Each scenario modifies parameters globally and recomputes every
   component lambda from scratch through the IEC TR 62380 model.

3. Component-Level Criticality
   "For each component, which parameter drives its lambda?"
   Per-component OAT elasticity ranking.

Mathematical notes:
   OAT sensitivity is local (measures derivative at the operating point).
   It does not capture interaction effects.  For a series reliability
   model (lambda_sys = SUM lambda_i), parameter interactions only
   arise WITHIN a single component's formula (e.g., T_junction
   affects both Arrhenius and Coffin-Manson terms simultaneously).
   Cross-component interactions are zero by construction.

   For global sensitivity including distribution effects, use the
   Uncertainty Analysis tab (Monte Carlo with SRRC).

Author:  Eliot Abramo
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable


# =====================================================================
# Data Structures
# =====================================================================

@dataclass
class TornadoEntry:
    """Single entry in a tornado chart."""
    name: str
    base_value: float       # System FIT at nominal
    low_value: float        # System FIT when parameter is at low bound
    high_value: float       # System FIT when parameter is at high bound
    delta_low: float        # base - low (positive = low is beneficial)
    delta_high: float       # high - base (positive = high is detrimental)
    swing: float            # |high - low|
    unit: str = "FIT"
    perturbation_desc: str = ""  # e.g. "+/-10 degC" or "+/-20%"


@dataclass
class TornadoResult:
    """Results from tornado sensitivity analysis."""
    entries: List[TornadoEntry]
    base_lambda_fit: float
    base_reliability: float
    mission_hours: float

    def to_dict(self) -> Dict:
        return {
            "base_lambda_fit": self.base_lambda_fit,
            "base_reliability": self.base_reliability,
            "mission_hours": self.mission_hours,
            "entries": [
                {
                    "name": e.name,
                    "base_value": e.base_value,
                    "low_value": e.low_value,
                    "high_value": e.high_value,
                    "delta_low": e.delta_low,
                    "delta_high": e.delta_high,
                    "swing": e.swing,
                    "unit": e.unit,
                    "perturbation_desc": e.perturbation_desc,
                }
                for e in self.entries
            ],
        }


@dataclass
class ScenarioEntry:
    """Single what-if scenario result."""
    name: str
    description: str
    lambda_fit: float
    reliability: float
    delta_lambda_pct: float
    delta_reliability: float
    is_custom: bool = False


@dataclass
class ScenarioResult:
    """Results from what-if scenario analysis."""
    baseline_lambda_fit: float
    baseline_reliability: float
    mission_hours: float
    scenarios: List[ScenarioEntry]

    def to_dict(self) -> Dict:
        return {
            "baseline_lambda_fit": self.baseline_lambda_fit,
            "baseline_reliability": self.baseline_reliability,
            "mission_hours": self.mission_hours,
            "scenarios": [
                {
                    "name": s.name,
                    "description": s.description,
                    "lambda_fit": s.lambda_fit,
                    "reliability": s.reliability,
                    "delta_lambda_pct": s.delta_lambda_pct,
                    "delta_reliability": s.delta_reliability,
                    "is_custom": s.is_custom,
                }
                for s in self.scenarios
            ],
        }


@dataclass
class CriticalityEntry:
    """Criticality analysis for a single component."""
    reference: str
    component_type: str
    base_lambda_fit: float
    fields: List[Dict]   # [{name, value, elasticity, impact_pct}]


@dataclass
class TornadoPerturbation:
    """User-specified perturbation for a parameter in tornado analysis.

    The user specifies physical units, not percentages.
    """
    param_name: str
    delta_low: float   # Amount to subtract (positive number => subtract)
    delta_high: float  # Amount to add
    unit: str = ""
    enabled: bool = True


# =====================================================================
# Default Perturbations (physical units)
# =====================================================================

DEFAULT_PERTURBATIONS = [
    TornadoPerturbation("t_junction", 10.0, 10.0, "degC"),
    TornadoPerturbation("t_ambient", 10.0, 10.0, "degC"),
    TornadoPerturbation("n_cycles", 1000, 1000, "cycles/yr"),
    TornadoPerturbation("delta_t", 5.0, 5.0, "degC"),
    TornadoPerturbation("tau_on", 0.1, 0.1, ""),
    TornadoPerturbation("operating_power", 0.0, 0.0, "W"),  # disabled
    TornadoPerturbation("voltage_stress_vds", 0.05, 0.05, "ratio"),
    TornadoPerturbation("voltage_stress_vgs", 0.05, 0.05, "ratio"),
    TornadoPerturbation("voltage_stress_vce", 0.05, 0.05, "ratio"),
    TornadoPerturbation("ripple_ratio", 0.05, 0.05, "ratio"),
    TornadoPerturbation("if_applied", 0.05, 0.05, "A"),
]


# =====================================================================
# Tornado Sensitivity (parameter-level, physical units)
# =====================================================================

def _import_math():
    try:
        from .reliability_math import (
            calculate_component_lambda, reliability_from_lambda
        )
    except ImportError:
        from reliability_math import (
            calculate_component_lambda, reliability_from_lambda
        )
    return calculate_component_lambda, reliability_from_lambda


def tornado_analysis(
    sheet_data: Dict[str, Dict],
    mission_hours: float,
    perturbations: Optional[List[TornadoPerturbation]] = None,
    active_sheets: Optional[List[str]] = None,
    excluded_types: Optional[set] = None,
    mode: str = "parameter",
) -> TornadoResult:
    """Tornado sensitivity with physical-unit perturbations.

    For mode="parameter":
      For each parameter perturbation, applies the delta to ALL components
      that use that parameter, recomputes their lambda, and measures
      the system FIT change.

    For mode="sheet":
      For each sheet, scales its total lambda by +/- 20% to show
      sub-system contribution (no formula re-evaluation needed).

    Parameters
    ----------
    perturbations : list of TornadoPerturbation
        User-specified or default perturbation amounts.
    mode : str
        "parameter" or "sheet"
    """
    calc_lambda, rel_from_lambda = _import_math()
    excluded = excluded_types or set()

    if active_sheets:
        filtered = {k: v for k, v in sheet_data.items() if k in active_sheets}
    else:
        filtered = sheet_data

    # Gather components
    all_comps = []
    for data in filtered.values():
        for comp in data.get("components", []):
            ctype = comp.get("class", "Unknown")
            if ctype in excluded:
                continue
            all_comps.append(comp)

    base_lambda = sum(
        float(d.get("lambda", 0) or 0) for d in filtered.values()
    )
    base_fit = base_lambda * 1e9
    base_r = rel_from_lambda(base_lambda, mission_hours)

    entries = []

    if mode == "sheet":
        for path, data in filtered.items():
            lam = float(data.get("lambda", 0) or 0)
            if lam <= 0:
                continue
            name = path.rstrip("/").split("/")[-1] or "Root"
            pct = 0.20  # 20% for sheet-level
            delta = lam * pct
            low_fit = (base_lambda - delta) * 1e9
            high_fit = (base_lambda + delta) * 1e9
            entries.append(TornadoEntry(
                name=name, base_value=base_fit,
                low_value=low_fit, high_value=high_fit,
                delta_low=base_fit - low_fit,
                delta_high=high_fit - base_fit,
                swing=high_fit - low_fit,
                perturbation_desc="+/- 20%",
            ))
    else:
        # Parameter-level
        if perturbations is None:
            perturbations = [p for p in DEFAULT_PERTURBATIONS]

        for pert in perturbations:
            if not pert.enabled:
                continue
            if pert.delta_low <= 0 and pert.delta_high <= 0:
                continue

            d_low_total = 0.0
            d_high_total = 0.0
            n_affected = 0

            for comp in all_comps:
                if comp.get("override_lambda") is not None:
                    continue
                params = comp.get("params", {})
                ct = comp.get("class", "Resistor")
                nom_val = params.get(pert.param_name)
                if nom_val is None:
                    continue
                try:
                    nom_val = float(nom_val)
                except (TypeError, ValueError):
                    continue

                nom_lam = float(comp.get("lambda", 0) or 0)
                n_affected += 1

                # Low perturbation
                p_lo = dict(params)
                p_lo[pert.param_name] = nom_val - pert.delta_low
                try:
                    lam_lo = calc_lambda(ct, p_lo).get("lambda_total", 0)
                except Exception:
                    lam_lo = nom_lam

                # High perturbation
                p_hi = dict(params)
                p_hi[pert.param_name] = nom_val + pert.delta_high
                try:
                    lam_hi = calc_lambda(ct, p_hi).get("lambda_total", 0)
                except Exception:
                    lam_hi = nom_lam

                d_low_total += (lam_lo - nom_lam)
                d_high_total += (lam_hi - nom_lam)

            if n_affected == 0:
                continue

            low_fit = (base_lambda + d_low_total) * 1e9
            high_fit = (base_lambda + d_high_total) * 1e9
            swing = abs(high_fit - low_fit)
            if swing < 1e-6:
                continue

            unit_str = f" {pert.unit}" if pert.unit else ""
            desc = f"-{pert.delta_low}{unit_str} / +{pert.delta_high}{unit_str}"

            entries.append(TornadoEntry(
                name=f"{pert.param_name} ({n_affected} comps)",
                base_value=base_fit,
                low_value=low_fit, high_value=high_fit,
                delta_low=base_fit - low_fit,
                delta_high=high_fit - base_fit,
                swing=swing,
                perturbation_desc=desc,
            ))

    entries.sort(key=lambda e: -e.swing)
    return TornadoResult(
        entries=entries, base_lambda_fit=base_fit,
        base_reliability=base_r, mission_hours=mission_hours,
    )


# =====================================================================
# What-If / Design-Margin Scenarios
# =====================================================================

# Predefined scenarios (param_name -> modification function)
PREDEFINED_SCENARIOS = [
    ("Temp +10 degC", "All junction/ambient temperatures +10 degC",
     {"t_ambient": lambda v: v + 10, "t_junction": lambda v: v + 10,
      "temperature_c": lambda v: v + 10}),
    ("Temp +20 degC", "All junction/ambient temperatures +20 degC",
     {"t_ambient": lambda v: v + 20, "t_junction": lambda v: v + 20,
      "temperature_c": lambda v: v + 20}),
    ("Temp -10 degC", "Improved cooling: all temperatures -10 degC",
     {"t_ambient": lambda v: v - 10, "t_junction": lambda v: v - 10,
      "temperature_c": lambda v: v - 10}),
    ("Thermal cycles x2", "Double annual thermal cycling count",
     {"n_cycles": lambda v: v * 2}),
    ("Thermal cycles /2", "Halve annual thermal cycling",
     {"n_cycles": lambda v: v * 0.5}),
    ("Delta-T x2", "Double thermal cycling amplitude",
     {"delta_t": lambda v: v * 2}),
    ("Delta-T /2", "Halve thermal cycling amplitude",
     {"delta_t": lambda v: v * 0.5}),
    ("50% duty cycle", "All components at tau_on = 0.5",
     {"tau_on": lambda _: 0.5}),
    ("100% duty cycle", "All components at tau_on = 1.0",
     {"tau_on": lambda _: 1.0}),
]


def scenario_analysis(
    sheet_data: Dict[str, Dict],
    mission_hours: float,
    active_sheets: Optional[List[str]] = None,
    excluded_types: Optional[set] = None,
    custom_scenarios: Optional[List[Tuple[str, str, Dict]]] = None,
) -> ScenarioResult:
    """Run predefined + custom what-if scenarios.

    Parameters
    ----------
    custom_scenarios : list of (name, description, {param: fn})
        Additional user-defined scenarios.
    """
    calc_lambda, rel_from_lambda = _import_math()
    excluded = excluded_types or set()

    if active_sheets:
        filtered = {k: v for k, v in sheet_data.items() if k in active_sheets}
    else:
        filtered = sheet_data

    base_lambda = sum(
        float(d.get("lambda", 0) or 0) for d in filtered.values()
    )
    base_fit = base_lambda * 1e9
    base_r = rel_from_lambda(base_lambda, mission_hours)

    def _run_scenario(mods):
        total = 0.0
        for data in filtered.values():
            for comp in data.get("components", []):
                ctype = comp.get("class", "Unknown")
                if ctype in excluded:
                    continue
                ovr = comp.get("override_lambda")
                if ovr is not None:
                    total += ovr
                    continue
                p = dict(comp.get("params", {}))
                for pn, fn in mods.items():
                    if pn in p:
                        try:
                            p[pn] = fn(float(p[pn]))
                        except (TypeError, ValueError):
                            pass
                try:
                    total += calc_lambda(ctype, p).get("lambda_total", 0)
                except Exception:
                    total += float(comp.get("lambda", 0) or 0)
        return total

    results = []

    # Predefined
    for name, desc, mods in PREDEFINED_SCENARIOS:
        try:
            sl = _run_scenario(mods)
            sf = sl * 1e9
            sr = rel_from_lambda(sl, mission_hours)
            dp = ((sf - base_fit) / base_fit * 100) if base_fit > 0 else 0
            results.append(ScenarioEntry(
                name=name, description=desc,
                lambda_fit=sf, reliability=sr,
                delta_lambda_pct=dp,
                delta_reliability=sr - base_r,
                is_custom=False,
            ))
        except Exception:
            continue

    # Custom
    if custom_scenarios:
        for name, desc, mods in custom_scenarios:
            try:
                sl = _run_scenario(mods)
                sf = sl * 1e9
                sr = rel_from_lambda(sl, mission_hours)
                dp = ((sf - base_fit) / base_fit * 100) if base_fit > 0 else 0
                results.append(ScenarioEntry(
                    name=name, description=desc,
                    lambda_fit=sf, reliability=sr,
                    delta_lambda_pct=dp,
                    delta_reliability=sr - base_r,
                    is_custom=True,
                ))
            except Exception:
                continue

    return ScenarioResult(base_fit, base_r, mission_hours, results)


# =====================================================================
# Component-Level Criticality
# =====================================================================

def component_criticality(
    sheet_data: Dict[str, Dict],
    mission_hours: float = 8760.0,
    perturbation: float = 0.10,
    active_sheets: Optional[List[str]] = None,
    excluded_types: Optional[set] = None,
    target_fields: Optional[Dict[str, List[str]]] = None,
    max_components: int = 0,
) -> List[CriticalityEntry]:
    """Per-component OAT elasticity ranking.

    For each component, perturbs each numeric parameter by +/- perturbation
    (relative) and measures the elasticity: d(ln lambda)/d(ln theta).

    Parameters
    ----------
    max_components : int
        If > 0, only analyze top-N components by FIT.  If 0, analyze ALL.
    target_fields : dict, optional
        {category: [field_names]} restricts which fields to analyze.
    """
    try:
        from .reliability_math import analyze_component_criticality
    except ImportError:
        from reliability_math import analyze_component_criticality

    excluded = excluded_types or set()

    if active_sheets:
        filtered = {k: v for k, v in sheet_data.items() if k in active_sheets}
    else:
        filtered = sheet_data

    # Gather and sort all components
    all_comps = []
    for data in filtered.values():
        for comp in data.get("components", []):
            ctype = comp.get("class", "Unknown")
            if ctype in excluded:
                continue
            if comp.get("override_lambda") is not None:
                continue
            all_comps.append(comp)

    all_comps.sort(key=lambda c: float(c.get("lambda", 0) or 0), reverse=True)
    if max_components > 0:
        all_comps = all_comps[:max_components]

    results = []
    for comp in all_comps:
        ref = comp.get("ref", "?")
        comp_type = comp.get("class", "")
        params = comp.get("params", {})
        if not comp_type or not params:
            continue

        try:
            raw = analyze_component_criticality(
                comp_type, params, mission_hours, perturbation
            )
        except Exception:
            continue
        if not raw:
            continue

        if target_fields is not None:
            allowed = target_fields.get(comp_type, [])
            if allowed:
                raw = [r for r in raw if r["field"] in allowed]
        if not raw:
            continue

        base_fit = raw[0]["lambda_nominal_fit"]
        fields = [
            {
                "name": r["field"],
                "value": r["nominal_value"],
                "elasticity": r["sensitivity"],
                "impact_pct": r["impact_percent"],
            }
            for r in raw
        ]
        results.append(CriticalityEntry(
            reference=ref,
            component_type=comp_type,
            base_lambda_fit=base_fit,
            fields=fields,
        ))

    return results


# =====================================================================
# Single-parameter what-if (bidirectional tool)
# =====================================================================

def single_param_whatif(
    component: Dict,
    param_name: str,
    new_value: Any,
    system_lambda: float,
    mission_hours: float,
) -> Dict[str, Any]:
    """Compute the system impact of changing one parameter on one component.

    Returns dict with before/after lambda, R(t), delta.
    Used for the bidirectional "what happens if I change this?" tool.
    """
    calc_lambda, rel_from_lambda = _import_math()

    ct = component.get("class", "Resistor")
    params = component.get("params", {})
    old_lam = float(component.get("lambda", 0) or 0)
    ref = component.get("ref", "?")

    # Compute new lambda
    p_new = dict(params)
    p_new[param_name] = new_value
    try:
        new_lam = calc_lambda(ct, p_new).get("lambda_total", 0)
    except Exception:
        new_lam = old_lam

    delta_lam = new_lam - old_lam
    new_sys_lambda = system_lambda + delta_lam
    new_sys_fit = new_sys_lambda * 1e9
    new_r = rel_from_lambda(new_sys_lambda, mission_hours)
    old_r = rel_from_lambda(system_lambda, mission_hours)

    return {
        "reference": ref,
        "param_name": param_name,
        "old_value": params.get(param_name),
        "new_value": new_value,
        "comp_fit_before": old_lam * 1e9,
        "comp_fit_after": new_lam * 1e9,
        "comp_delta_fit": delta_lam * 1e9,
        "sys_fit_before": system_lambda * 1e9,
        "sys_fit_after": new_sys_fit,
        "sys_delta_fit": delta_lam * 1e9,
        "r_before": old_r,
        "r_after": new_r,
        "delta_r": new_r - old_r,
    }


# =====================================================================
# Helper: active sheet paths from blocks
# =====================================================================

def get_active_sheet_paths(blocks) -> Optional[List[str]]:
    """Extract sheet paths from block editor blocks dict."""
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

# =====================================================================
# Backward-compatible aliases for __init__.py imports
# =====================================================================

DesignMarginResult = ScenarioResult

def tornado_sheet_sensitivity(sheet_data, mission_hours, **kw):
    return tornado_analysis(sheet_data, mission_hours, mode="sheet", **kw)

def tornado_parameter_sensitivity(sheet_data, mission_hours, **kw):
    return tornado_analysis(sheet_data, mission_hours, mode="parameter", **kw)

def design_margin_analysis(sheet_data, mission_hours, **kw):
    return scenario_analysis(sheet_data, mission_hours, **kw)

def analyze_board_criticality(sheet_data, mission_hours=8760.0, **kw):
    return component_criticality(sheet_data, mission_hours, **kw)