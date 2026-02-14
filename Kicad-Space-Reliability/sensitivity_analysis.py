"""
Sensitivity Analysis Module
============================
Multi-mode sensitivity: Tornado charts, Design-margin analysis,
parameter-level sensitivity, and component criticality.

Replaces Sobol-only approach with tools that provide actionable
co-design insights for reliability engineering.

Author:  Eliot Abramo
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class SobolResult:
    """Results from Sobol sensitivity analysis."""
    parameter_names: List[str]
    S_first: np.ndarray
    S_total: np.ndarray
    S_first_conf: np.ndarray
    S_total_conf: np.ndarray
    interaction_scores: np.ndarray
    significant_interactions: List[int]
    n_samples: int

    def to_dict(self) -> Dict:
        return {
            "parameters": self.parameter_names,
            "S_first": self.S_first.tolist(),
            "S_total": self.S_total.tolist(),
            "S_first_conf": self.S_first_conf.tolist(),
            "S_total_conf": self.S_total_conf.tolist(),
            "interaction_scores": self.interaction_scores.tolist(),
            "significant_interactions": [
                self.parameter_names[i] for i in self.significant_interactions
            ],
            "n_samples": self.n_samples,
        }

    def get_ranking(self, use_total: bool = True) -> List[Tuple[str, float]]:
        indices = self.S_total if use_total else self.S_first
        return sorted(zip(self.parameter_names, indices), key=lambda x: -x[1])


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
# Sobol  (kept for backward compat)
# =============================================================================

def generate_sobol_samples(d, n, bounds, seed=None):
    rng = np.random.default_rng(seed)
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d, scramble=True, seed=seed)
        X1_unit = sampler.random(n)
        X2_unit = sampler.random(n)
    except ImportError:
        X1_unit = rng.random((n, d))
        X2_unit = rng.random((n, d))
    bounds = np.array(bounds)
    X1 = bounds[:, 0] + X1_unit * (bounds[:, 1] - bounds[:, 0])
    X2 = bounds[:, 0] + X2_unit * (bounds[:, 1] - bounds[:, 0])
    return X1, X2


def sobol_indices(model_func, X1, X2, compute_total=True, confidence_level=0.95):
    N, d = X1.shape
    Y1 = model_func(X1)
    Y2 = model_func(X2)
    var_Y = 0.5 * (np.var(Y1) + np.var(Y2))
    if var_Y < 1e-15:
        z = np.zeros(d)
        return z, z, z, z
    S_first, S_total = np.zeros(d), np.zeros(d)
    S_first_conf, S_total_conf = np.zeros(d), np.zeros(d)
    for i in range(d):
        X1_i = X1.copy(); X1_i[:, i] = X2[:, i]
        Y1_i = model_func(X1_i)
        t1 = np.mean(Y1 * Y1_i)
        t2 = 0.25 * (np.mean(Y1 + Y1_i)) ** 2
        den = 0.5 * (np.mean(Y1**2) + np.mean(Y1_i**2)) - t2
        if abs(den) > 1e-15:
            S_first[i] = (t1 - t2) / den
        S_first_conf[i] = 1.96 * np.sqrt(np.var(Y1 * Y1_i - t1) / N) / max(den, 1e-15)
        if compute_total:
            X2_i = X2.copy(); X2_i[:, i] = X1[:, i]
            Y2_i = model_func(X2_i)
            S_total[i] = np.mean((Y1 - Y2_i) ** 2) / (2 * var_Y)
            T_i = (Y1 - Y2_i) ** 2 / 2 - S_total[i] * var_Y
            S_total_conf[i] = 1.96 * np.std(T_i) / (var_Y * np.sqrt(N))
    return np.clip(S_first, 0, 1), np.clip(S_total, 0, 1), S_first_conf, S_total_conf


class SobolAnalyzer:
    def __init__(self, seed=None):
        self.seed = seed
        self.default_n_samples = 2048
        self.interaction_threshold = 0.1

    def analyze(self, model_func, parameter_bounds, n_samples=None):
        n = n_samples or self.default_n_samples
        param_names = list(parameter_bounds.keys())
        d = len(param_names)
        bounds = [parameter_bounds[name] for name in param_names]
        X1, X2 = generate_sobol_samples(d, n, bounds, self.seed)

        def array_model(X):
            results = np.empty(X.shape[0])
            for start in range(0, X.shape[0], 512):
                end = min(start + 512, X.shape[0])
                for j in range(end - start):
                    params = {name: X[start + j, i] for i, name in enumerate(param_names)}
                    results[start + j] = model_func(params)
            return results

        S_first, S_total, S_first_conf, S_total_conf = sobol_indices(array_model, X1, X2)
        interaction_scores = S_total - S_first
        significant = [i for i in range(d)
                       if S_total[i] > 1e-6 and interaction_scores[i] > self.interaction_threshold * S_total[i]]
        return SobolResult(
            parameter_names=param_names, S_first=S_first, S_total=S_total,
            S_first_conf=S_first_conf, S_total_conf=S_total_conf,
            interaction_scores=interaction_scores,
            significant_interactions=significant, n_samples=n)


# =============================================================================
# Tornado Sensitivity Analysis  (primary tool)
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


def tornado_parameter_sensitivity(
    sheet_data: Dict[str, Dict],
    mission_hours: float,
    perturbation: float = 0.20,
    active_sheets: List[str] = None,
    target_fields: Dict[str, List[str]] = None,
) -> TornadoResult:
    """Tornado at PARAMETER level: perturb a design parameter across ALL components.
    Answers: 'If T_ambient +20%, how does system FIT change?'"""
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
    """Run predefined what-if scenarios for design-margin evaluation."""
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
        ("Temp +10 C", "All temperatures +10 C",
         {"t_ambient": lambda v: v+10, "t_junction": lambda v: v+10, "temperature_c": lambda v: v+10}),
        ("Temp +20 C", "All temperatures +20 C",
         {"t_ambient": lambda v: v+20, "t_junction": lambda v: v+20, "temperature_c": lambda v: v+20}),
        ("Power derate 70%", "Operating power at 70%",
         {"operating_power": lambda v: v*0.7, "applied_power_w": lambda v: v*0.7}),
        ("Power derate 50%", "Operating power at 50%",
         {"operating_power": lambda v: v*0.5, "applied_power_w": lambda v: v*0.5}),
        ("Thermal cycles x2", "Double annual cycling",
         {"n_cycles": lambda v: v*2}),
        ("Delta-T x2", "Double thermal excursion",
         {"delta_t": lambda v: v*2}),
        ("50% duty cycle", "tau_on = 0.5",
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
    target_fields: {category: [field_names]} restricts which fields to analyze."""
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


def quick_sensitivity(lambda_components, mission_hours, uncertainty_range=0.3, n_samples=1024):
    try:
        from .reliability_math import reliability_from_lambda
    except ImportError:
        from reliability_math import reliability_from_lambda
    bounds = {n: (l*(1-uncertainty_range), l*(1+uncertainty_range)) for n, l in lambda_components.items()}
    def model(s):
        return reliability_from_lambda(sum(s.values()), mission_hours)
    return SobolAnalyzer().analyze(model, bounds, n_samples)


def print_sobol_results(result, max_rows=15):
    print("\n" + "=" * 70)
    print("SOBOL SENSITIVITY ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Samples: {result.n_samples}")
    print(f"\n{'Parameter':<25} {'S_first':>12} {'S_total':>12} {'Interaction':>12}")
    print("-" * 70)
    for idx, _ in sorted(enumerate(result.S_total), key=lambda x: -x[1])[:max_rows]:
        name = result.parameter_names[idx]
        flag = " ***" if idx in result.significant_interactions else ""
        print(f"{name:<25} {result.S_first[idx]:>12.4f} {result.S_total[idx]:>12.4f} {result.interaction_scores[idx]:>12.4f}{flag}")
    print("=" * 70)
