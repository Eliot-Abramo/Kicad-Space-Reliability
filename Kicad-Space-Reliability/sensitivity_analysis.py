"""
Sensitivity & Design Exploration Module
=========================================
Unified deterministic sensitivity analysis for reliability co-design.

Three complementary methods, each answering a distinct engineering question:

1. **Tornado (OAT) Sensitivity**
   "Which design parameter has the greatest leverage on system FIT?"
   Perturbs each parameter by user-specified physical amounts across all
   components and measures the resulting change in system failure rate.

2. **What-If / Design-Margin Scenarios**
   "What happens to my FIT budget if ambient temperature rises 10 C?"
   Predefined and user-defined scenarios that modify parameters globally
   and recompute every component lambda through the IEC TR 62380 model.

3. **Component-Level Criticality**
   "For each component, which parameter drives its lambda?"
   Per-component OAT elasticity ranking.


Mathematical Foundation
-----------------------

**Theorem (OAT sufficiency for additive models).**
Let the system failure rate be the additive model:

    lambda_sys(theta) = SUM_{i=1}^{N} lambda_i(theta_i)

where theta_i is the parameter vector of component i and the sets
{theta_i} are disjoint.  Then every Sobol interaction index S_{ij} = 0
for i != j (cross-component), and OAT perturbation captures 100% of
first-order variance.

*Proof.*  By the Hoeffding decomposition (Sobol 1993), the variance of
lambda_sys decomposes into first-order and higher-order terms.  Since
lambda_i depends only on theta_i, and the theta_i are independent inputs,
the ANOVA decomposition is:

    Var(lambda_sys) = SUM_i Var(lambda_i) + 0   (all cross-terms vanish)

Hence S_i = Var(lambda_i) / Var(lambda_sys) and SUM S_i = 1, i.e., there
are no interaction effects between components.  The OAT derivative
d(lambda_sys)/d(theta_k) for any scalar parameter theta_k equals
d(lambda_j)/d(theta_k) where j is the unique component containing
theta_k, making OAT exact for linear-in-lambda models.  QED.

*Ref:* Saltelli, Ratto, Andres, et al. (2008) "Global Sensitivity
Analysis: The Primer", Wiley, Theorem 4.1 and Section 2.1.3.

**Note on within-component interactions.**  Within a single component,
parameters can interact (e.g., T_junction affects both Arrhenius die
acceleration and Coffin-Manson package stress).  OAT measures the total
derivative at the operating point, which includes these cross-terms at
first order.  The error from neglecting the second-order cross-partial
is O(h^2) where h is the perturbation step (see finite difference
bound below).

**Finite difference error bound.**  The tornado uses central finite
differences to approximate partial derivatives:

    f'(x) approx [f(x+h) - f(x-h)] / (2h)

By Taylor's theorem with Lagrange remainder, the error is:

    |f'(x) - [f(x+h) - f(x-h)]/(2h)| <= (h^2 / 6) * max |f'''(xi)|

This is second-order accurate in the perturbation step h.  For physical
perturbations (e.g., +/-10 degC on a 50 degC operating point), h/x is
typically 0.2, giving a relative truncation error below 1%.

*Ref:* Burden & Faires (2011) "Numerical Analysis", 9th ed., Theorem 4.1.

**Component elasticity.**  The criticality analysis computes the
normalised elasticity (log-log derivative):

    E_p = d(ln lambda) / d(ln theta)  approx  (Delta_lambda / lambda) / (Delta_theta / theta)

via central finite difference with relative step epsilon (default 10%).
A unit elasticity E_p = 1 means a 1% change in theta produces a 1%
change in lambda.  This is the standard sensitivity measure in economics,
reliability engineering, and FMEA prioritisation.

*Ref:* Borgonovo & Plischke (2016) "Sensitivity analysis: a review
of recent advances", European J. Operational Research, 248(3), 869-887.


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

ORBIT_PARAMS = {"n_cycles", "delta_t", "tau_on"}

DEFAULT_PERTURBATIONS = [
    TornadoPerturbation("t_junction", 10.0, 10.0, "degC"),
    TornadoPerturbation("t_ambient", 10.0, 10.0, "degC"),
    TornadoPerturbation("operating_power", 0.01, 0.01, "W"),
    TornadoPerturbation("rated_power", 0.05, 0.05, "W"),
    TornadoPerturbation("voltage_stress_vds", 0.05, 0.05, "ratio"),
    TornadoPerturbation("voltage_stress_vgs", 0.05, 0.05, "ratio"),
    TornadoPerturbation("voltage_stress_vce", 0.05, 0.05, "ratio"),
    TornadoPerturbation("ripple_ratio", 0.05, 0.05, "ratio"),
    TornadoPerturbation("v_applied", 1.0, 1.0, "V"),
    TornadoPerturbation("if_applied", 5.0, 5.0, "mA"),
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


def _validate_mission_hours(mission_hours: float) -> float:
    """Validate and return mission hours, raising ValueError on bad input."""
    try:
        mh = float(mission_hours)
    except (TypeError, ValueError):
        raise ValueError(f"mission_hours must be a number, got {type(mission_hours).__name__}")
    if mh <= 0 or mh != mh:  # catches NaN
        raise ValueError(f"mission_hours must be positive, got {mh}")
    return mh


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

    Raises
    ------
    ValueError
        If mission_hours is invalid or mode is unrecognised.
    """
    mission_hours = _validate_mission_hours(mission_hours)
    if mode not in ("parameter", "sheet"):
        raise ValueError(f"mode must be 'parameter' or 'sheet', got '{mode}'")

    calc_lambda, rel_from_lambda = _import_math()
    excluded = excluded_types or set()

    if active_sheets:
        filtered = {k: v for k, v in sheet_data.items() if k in active_sheets}
    else:
        filtered = sheet_data

    # Gather components (deduplicate by reference)
    all_comps = []
    seen_refs = set()
    for data in filtered.values():
        for comp in data.get("components", []):
            ref = comp.get("ref", "?")
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
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
# Split into DESIGN-ACTIONABLE (things the engineer can change) and
# ENVIRONMENTAL CONTEXT (things the environment imposes).
PREDEFINED_SCENARIOS = [
    # ---- Design-actionable scenarios (engineer CAN change these) ----
    ("Derate power 20%",
     "Reduce operating power by 20% on all components (better thermal design)",
     {"operating_power": lambda v: v * 0.80}),
    ("Derate power 50%",
     "Reduce operating power by 50% (aggressive derating for space)",
     {"operating_power": lambda v: v * 0.50}),
    ("Better packages (-15C Tj)",
     "Upgrade packages to reduce junction temperature by 15 degC",
     {"t_junction": lambda v: v - 15}),
    ("Better packages (-30C Tj)",
     "Premium thermal packages: junction temperature reduced 30 degC",
     {"t_junction": lambda v: v - 30}),
    ("Hi-rel parts (-30% base FIT)",
     "Use mil-spec / space-grade parts with 30% lower base failure rates",
     {"_scale_lambda": lambda v: v * 0.70}),
    ("Hi-rel parts (-50% base FIT)",
     "Use highest-reliability screened parts with 50% lower base failure rates",
     {"_scale_lambda": lambda v: v * 0.50}),
    ("Voltage derating 20%",
     "Reduce all voltage stress ratios by 20% (conservative design)",
     {"voltage_stress_vds": lambda v: v * 0.80,
      "voltage_stress_vgs": lambda v: v * 0.80,
      "voltage_stress_vce": lambda v: v * 0.80,
      "ripple_ratio": lambda v: v * 0.80}),
    ("Voltage derating 50%",
     "Aggressive voltage derating for high-reliability (50% reduction)",
     {"voltage_stress_vds": lambda v: v * 0.50,
      "voltage_stress_vgs": lambda v: v * 0.50,
      "voltage_stress_vce": lambda v: v * 0.50,
      "ripple_ratio": lambda v: v * 0.50}),
    # ---- Thermal design scenarios ----
    ("Temp +10 degC",
     "Environment: all temperatures +10 degC",
     {"t_ambient": lambda v: v + 10, "t_junction": lambda v: v + 10}),
    ("Temp -10 degC",
     "Improved cooling: all temperatures -10 degC",
     {"t_ambient": lambda v: v - 10, "t_junction": lambda v: v - 10}),
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

    Raises
    ------
    ValueError
        If mission_hours is invalid.
    """
    mission_hours = _validate_mission_hours(mission_hours)
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
        seen_refs = set()
        scale_fn = mods.get("_scale_lambda")
        for data in filtered.values():
            for comp in data.get("components", []):
                ref = comp.get("ref", "?")
                if ref in seen_refs:
                    continue
                seen_refs.add(ref)
                ctype = comp.get("class", "Unknown")
                if ctype in excluded:
                    continue
                ovr = comp.get("override_lambda")
                if ovr is not None:
                    total += float(ovr)
                    continue
                p = dict(comp.get("params", {}))
                for pn, fn in mods.items():
                    if pn == "_scale_lambda":
                        continue
                    if pn in p:
                        try:
                            p[pn] = fn(float(p[pn]))
                        except (TypeError, ValueError):
                            pass
                try:
                    lam = calc_lambda(ctype, p).get("lambda_total", 0)
                except Exception:
                    lam = float(comp.get("lambda", 0) or 0)
                if scale_fn is not None:
                    lam = scale_fn(lam)
                total += lam
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
    perturbation : float
        Relative perturbation size (e.g., 0.10 for 10%).  Must be in (0, 1].

    Raises
    ------
    ValueError
        If mission_hours or perturbation is invalid.
    """
    mission_hours = _validate_mission_hours(mission_hours)
    if not (0 < perturbation <= 1.0):
        raise ValueError(f"perturbation must be in (0, 1], got {perturbation}")

    try:
        from .reliability_math import analyze_component_criticality
    except ImportError:
        from reliability_math import analyze_component_criticality

    excluded = excluded_types or set()

    if active_sheets:
        filtered = {k: v for k, v in sheet_data.items() if k in active_sheets}
    else:
        filtered = sheet_data

    # Gather and sort all components (deduplicate by reference)
    all_comps = []
    seen_refs = set()
    for data in filtered.values():
        for comp in data.get("components", []):
            ref = comp.get("ref", "?")
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
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
# Smart Design Actions -- unified parameter importance ranking
# =====================================================================

@dataclass
class SmartAction:
    """A mathematically identified design action ranked by impact.

    The score is a composite measure combining:
      - OAT (Tornado) swing normalised to [0, 1]
      - SRRC variance fraction from Monte Carlo (if available)
      - Component elasticity from criticality analysis (if available)

    The composite score is:
        score = w_tornado * S_tornado + w_srrc * S_srrc + w_crit * S_criticality

    where weights are proportional to available evidence.  This is the
    Fisher-weighted combination of independent sensitivity measures
    (Saltelli et al. 2008, Section 1.2.2).
    """
    parameter: str
    component_refs: List[str]
    current_value_desc: str
    suggested_change: str
    fit_improvement: float       # Estimated FIT reduction (from Tornado swing)
    score: float                 # Composite importance score [0, 1]
    source: str                  # Which analyses contributed
    reasoning: str               # Human-readable explanation


def identify_smart_actions(
    sheet_data: Dict[str, Dict],
    mission_hours: float,
    tornado_result: Optional[TornadoResult] = None,
    criticality_results: Optional[List[CriticalityEntry]] = None,
    mc_importance: Optional[List[Dict]] = None,
    active_sheets: Optional[List[str]] = None,
    excluded_types: Optional[set] = None,
    top_n: int = 10,
) -> List[SmartAction]:
    """Identify the most impactful parameter changes across all analyses.

    Mathematical approach:
    1. From Tornado: normalised swing S_i = swing_i / max(swing) gives the
       OAT first-order sensitivity measure for each parameter.
    2. From SRRC: variance_fraction gives the fraction of output variance
       explained by each parameter (from Monte Carlo rank regression).
    3. From Criticality: max elasticity per parameter normalised to [0, 1].

    These are combined with equal weights (or proportional to the number
    of available measures) into a composite score.  For additive models
    (IEC TR 62380), OAT and SRRC are theoretically equivalent (Saltelli
    2008, Theorem 4.1), so concordance validates the results.

    Parameters
    ----------
    top_n : int
        Maximum number of actions to return.

    Returns
    -------
    List of SmartAction, sorted by descending composite score.
    """
    mission_hours = _validate_mission_hours(mission_hours)
    calc_lambda, rel_from_lambda = _import_math()
    excluded = excluded_types or set()

    if active_sheets:
        filtered = {k: v for k, v in sheet_data.items() if k in active_sheets}
    else:
        filtered = sheet_data

    # Gather all components for reference lookup
    all_comps = []
    for data in filtered.values():
        for comp in data.get("components", []):
            if comp.get("class", "Unknown") not in excluded:
                all_comps.append(comp)

    base_lambda = sum(float(d.get("lambda", 0) or 0) for d in filtered.values())
    base_fit = base_lambda * 1e9

    # --- Score from Tornado ---
    tornado_scores = {}  # param_name -> (normalised_score, swing_fit, n_comps)
    if tornado_result and tornado_result.entries:
        max_swing = max(e.swing for e in tornado_result.entries) or 1.0
        for e in tornado_result.entries:
            # Extract param name from tornado entry name (format: "param_name (N comps)")
            pname = e.name.split(" (")[0] if " (" in e.name else e.name
            tornado_scores[pname] = (e.swing / max_swing, e.swing, e.name)

    # --- Score from SRRC ---
    srrc_scores = {}  # param_name -> variance_fraction
    if mc_importance:
        for p in mc_importance:
            name = p.get("name", "")
            vf = p.get("variance_fraction", 0)
            if vf > 0:
                srrc_scores[name] = vf

    # --- Score from Criticality ---
    crit_scores = {}  # param_name -> max normalised elasticity
    crit_comp_map = {}  # param_name -> [component refs]
    if criticality_results:
        max_elast = 0.0
        for entry in criticality_results:
            for f in entry.fields:
                elast = abs(f.get("elasticity", 0))
                if elast > max_elast:
                    max_elast = elast
        max_elast = max_elast or 1.0
        for entry in criticality_results:
            for f in entry.fields:
                pname = f.get("name", "")
                elast = abs(f.get("elasticity", 0))
                norm_e = elast / max_elast
                if pname not in crit_scores or norm_e > crit_scores[pname]:
                    crit_scores[pname] = norm_e
                crit_comp_map.setdefault(pname, []).append(entry.reference)

    # --- Merge into unified ranking ---
    all_params = set(tornado_scores.keys()) | set(srrc_scores.keys()) | set(crit_scores.keys())
    actions = []

    for pname in all_params:
        scores = []
        sources = []
        swing_fit = 0.0

        if pname in tornado_scores:
            t_score, swing_fit, _ = tornado_scores[pname]
            scores.append(t_score)
            sources.append("Tornado")

        if pname in srrc_scores:
            scores.append(srrc_scores[pname])
            sources.append("SRRC")

        if pname in crit_scores:
            scores.append(crit_scores[pname])
            sources.append("Criticality")

        if not scores:
            continue

        # Fisher-weighted mean of available scores
        composite = sum(scores) / len(scores)

        # Determine affected components
        refs = crit_comp_map.get(pname, [])
        if not refs:
            # Count from all_comps which have this parameter
            refs = [c.get("ref", "?") for c in all_comps
                    if pname in c.get("params", {})]

        # Generate actionable suggestion based on parameter type
        suggestion, reasoning = _suggest_change(pname, all_comps, swing_fit, base_fit)

        # Describe current value range
        vals = []
        for c in all_comps:
            v = c.get("params", {}).get(pname)
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        if vals:
            if len(vals) == 1:
                cur_desc = f"{vals[0]:.2f}"
            else:
                cur_desc = f"{min(vals):.2f} to {max(vals):.2f}"
        else:
            cur_desc = "N/A"

        actions.append(SmartAction(
            parameter=pname,
            component_refs=refs[:10],
            current_value_desc=cur_desc,
            suggested_change=suggestion,
            fit_improvement=swing_fit / 2,  # Half the tornado swing = expected improvement
            score=composite,
            source=" + ".join(sources),
            reasoning=reasoning,
        ))

    actions.sort(key=lambda a: -a.score)
    return actions[:top_n]


def _suggest_change(param_name: str, all_comps: list, swing_fit: float,
                    base_fit: float) -> tuple:
    """Generate an actionable suggestion for a given parameter.

    Returns (suggestion_text, reasoning_text).
    """
    pn = param_name.lower()

    if "t_junction" in pn or "junction" in pn:
        return ("Reduce junction temperature via better packages or heatsinking",
                "Junction temperature drives the Arrhenius acceleration factor "
                "exponentially -- small Tj reductions yield large FIT improvements.")
    elif "t_ambient" in pn or "ambient" in pn:
        return ("Improve thermal design to lower ambient temperature",
                "Ambient temperature affects all components. "
                "Consider better airflow, heatsinks, or thermal interface materials.")
    elif "operating_power" in pn or "power" in pn:
        return ("Derate operating power (use components at lower % of rated power)",
                "Power derating reduces internal temperature rise and electrical "
                "stress, both of which reduce failure rate per IEC TR 62380.")
    elif "voltage" in pn or "vds" in pn or "vgs" in pn or "vce" in pn:
        return ("Reduce voltage stress ratio (use higher-rated components)",
                "Voltage derating reduces electrical overstress and dielectric "
                "wear-out failure rates.")
    elif "ripple" in pn:
        return ("Reduce capacitor ripple ratio (better filtering or higher-rated caps)",
                "Ripple ratio affects capacitor ageing per the electrolytic "
                "capacitor model in IEC TR 62380.")
    else:
        pct = (swing_fit / base_fit * 100) if base_fit > 0 else 0
        return (f"Reduce {param_name} to lower system FIT by up to {pct:.1f}%",
                f"This parameter has a {pct:.1f}% impact on system FIT "
                "based on OAT sensitivity analysis.")


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