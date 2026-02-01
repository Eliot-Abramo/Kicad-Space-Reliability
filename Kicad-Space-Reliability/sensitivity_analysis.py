"""
Industrial-Grade Sensitivity Analysis Module
=============================================
Comprehensive sensitivity and importance analysis for reliability engineering.

Features:
- Sobol sensitivity indices (first-order and total-order)
- Importance measures (Birnbaum, RAW, RRW, Fussell-Vesely)
- Tornado diagrams
- What-if scenario analysis
- Derating recommendations
- Critical component identification
- Uncertainty propagation analysis

IEC TR 62380 / ECSS-E-ST-10-12C compliant
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional, Any
import math


@dataclass
class SobolResult:
    """Results from Sobol sensitivity analysis."""
    parameter_names: List[str]
    S_first: np.ndarray  # First-order indices
    S_total: np.ndarray  # Total-order indices
    S_first_conf: np.ndarray  # Confidence intervals for first-order
    S_total_conf: np.ndarray  # Confidence intervals for total-order
    interaction_scores: np.ndarray  # S_total - S_first
    significant_interactions: List[int]  # Indices with significant interactions
    n_samples: int
    
    def to_dict(self) -> Dict:
        return {
            "parameters": self.parameter_names,
            "S_first": self.S_first.tolist(),
            "S_total": self.S_total.tolist(),
            "S_first_conf": self.S_first_conf.tolist(),
            "S_total_conf": self.S_total_conf.tolist(),
            "interaction_scores": self.interaction_scores.tolist(),
            "significant_interactions": [self.parameter_names[i] for i in self.significant_interactions],
            "n_samples": self.n_samples,
        }
    
    def get_ranking(self, use_total: bool = True) -> List[Tuple[str, float]]:
        """Get parameters ranked by influence."""
        indices = self.S_total if use_total else self.S_first
        return sorted(zip(self.parameter_names, indices), key=lambda x: -x[1])


@dataclass
class ImportanceMeasures:
    """Component importance measures for reliability analysis."""
    component_name: str
    
    # Basic measures
    lambda_fit: float  # Failure rate in FIT
    reliability: float  # Component reliability
    contribution_pct: float  # % of total system failure rate
    
    # Birnbaum importance: dR_sys/dR_comp
    # How much system reliability changes per unit change in component reliability
    birnbaum: float = 0.0
    
    # Risk Achievement Worth (RAW): R_sys(comp_failed) / R_sys
    # How much worse the system gets if this component fails
    raw: float = 1.0
    
    # Risk Reduction Worth (RRW): R_sys / R_sys(comp_perfect)
    # How much better the system gets if this component is perfect
    rrw: float = 1.0
    
    # Fussell-Vesely importance: P(comp caused failure) / P(system failure)
    # Probability that component failure causes system failure
    fussell_vesely: float = 0.0
    
    # Criticality ranking (1 = most critical)
    criticality_rank: int = 0
    
    # Derating recommendation
    derating_factor: float = 1.0
    recommended_action: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "component": self.component_name,
            "lambda_fit": self.lambda_fit,
            "reliability": self.reliability,
            "contribution_pct": self.contribution_pct,
            "birnbaum": self.birnbaum,
            "raw": self.raw,
            "rrw": self.rrw,
            "fussell_vesely": self.fussell_vesely,
            "criticality_rank": self.criticality_rank,
            "derating_factor": self.derating_factor,
            "recommended_action": self.recommended_action,
        }


@dataclass
class WhatIfScenario:
    """What-if scenario analysis result."""
    scenario_name: str
    description: str
    
    # Original values
    original_reliability: float
    original_lambda: float
    
    # Modified values
    modified_reliability: float
    modified_lambda: float
    
    # Changes
    reliability_change: float  # Absolute change
    reliability_change_pct: float  # Percentage change
    lambda_change: float
    lambda_change_pct: float
    
    # Parameters changed
    parameters_changed: Dict[str, Tuple[float, float]]  # {param: (old, new)}
    
    # Verdict
    improvement: bool  # True if reliability improved
    
    def to_dict(self) -> Dict:
        return {
            "scenario": self.scenario_name,
            "description": self.description,
            "original_r": self.original_reliability,
            "modified_r": self.modified_reliability,
            "r_change_pct": self.reliability_change_pct,
            "original_lambda_fit": self.original_lambda * 1e9,
            "modified_lambda_fit": self.modified_lambda * 1e9,
            "improvement": self.improvement,
            "parameters": self.parameters_changed,
        }


@dataclass
class DeratingRecommendation:
    """Derating recommendation for a component."""
    component_name: str
    component_type: str
    
    # Current stress levels
    current_stress: Dict[str, float]  # e.g., {"voltage": 0.8, "power": 0.6}
    
    # Recommended derating
    recommended_stress: Dict[str, float]
    
    # Expected improvement
    current_lambda: float
    derated_lambda: float
    improvement_pct: float
    
    # Priority (1 = highest)
    priority: int
    
    # Rationale
    rationale: str


@dataclass
class SensitivityReport:
    """Complete sensitivity analysis report."""
    # Sobol analysis
    sobol: Optional[SobolResult] = None
    
    # Component importance measures
    importance_measures: List[ImportanceMeasures] = field(default_factory=list)
    
    # What-if scenarios
    scenarios: List[WhatIfScenario] = field(default_factory=list)
    
    # Derating recommendations
    derating: List[DeratingRecommendation] = field(default_factory=list)
    
    # Critical components (top N by various metrics)
    critical_by_contribution: List[str] = field(default_factory=list)
    critical_by_birnbaum: List[str] = field(default_factory=list)
    critical_by_raw: List[str] = field(default_factory=list)
    
    # Summary statistics
    total_components: int = 0
    components_above_threshold: int = 0  # Components contributing > 5%
    max_contributor: str = ""
    max_contribution_pct: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "sobol": self.sobol.to_dict() if self.sobol else None,
            "importance_measures": [im.to_dict() for im in self.importance_measures],
            "scenarios": [s.to_dict() for s in self.scenarios],
            "critical_by_contribution": self.critical_by_contribution,
            "critical_by_birnbaum": self.critical_by_birnbaum,
            "critical_by_raw": self.critical_by_raw,
            "summary": {
                "total_components": self.total_components,
                "components_above_5pct": self.components_above_threshold,
                "max_contributor": self.max_contributor,
                "max_contribution_pct": self.max_contribution_pct,
            }
        }


# =============================================================================
# Sobol Sensitivity Analysis
# =============================================================================

def generate_sobol_samples(d: int, n: int, bounds: List[Tuple[float, float]], seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate quasi-random Sobol sequence samples."""
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


def sobol_indices(
    model_func: Callable[[np.ndarray], np.ndarray],
    X1: np.ndarray,
    X2: np.ndarray,
    compute_total: bool = True,
    confidence_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Sobol sensitivity indices using pick-freeze estimator."""
    N, d = X1.shape
    
    Y1 = model_func(X1)
    Y2 = model_func(X2)
    
    var_Y = 0.5 * (np.var(Y1) + np.var(Y2))
    if var_Y < 1e-15:
        zeros = np.zeros(d)
        return zeros, zeros, zeros, zeros
    
    S_first = np.zeros(d)
    S_total = np.zeros(d)
    S_first_conf = np.zeros(d)
    S_total_conf = np.zeros(d)
    
    for i in range(d):
        X1_i = X1.copy()
        X1_i[:, i] = X2[:, i]
        Y1_i = model_func(X1_i)
        
        term1 = np.mean(Y1 * Y1_i)
        term2 = 0.25 * (np.mean(Y1 + Y1_i))**2
        denom = 0.5 * (np.mean(Y1**2) + np.mean(Y1_i**2)) - term2
        
        if abs(denom) > 1e-15:
            S_first[i] = (term1 - term2) / denom
        
        S_first_conf[i] = 1.96 * np.sqrt(np.var(Y1 * Y1_i - term1) / N) / max(denom, 1e-15)
        
        if compute_total:
            X2_i = X2.copy()
            X2_i[:, i] = X1[:, i]
            Y2_i = model_func(X2_i)
            
            S_total[i] = np.mean((Y1 - Y2_i)**2) / (2 * var_Y)
            
            T_i = (Y1 - Y2_i)**2 / 2 - S_total[i] * var_Y
            S_total_conf[i] = 1.96 * np.std(T_i) / (var_Y * np.sqrt(N))
    
    S_first = np.clip(S_first, 0, 1)
    S_total = np.clip(S_total, 0, 1)
    
    return S_first, S_total, S_first_conf, S_total_conf


class SobolAnalyzer:
    """Sobol sensitivity analysis for reliability models."""
    
    def __init__(self, seed: int = None):
        self.seed = seed
        self.default_n_samples = 2048
        self.interaction_threshold = 0.1
    
    def analyze(
        self,
        model_func: Callable[[Dict[str, float]], float],
        parameter_bounds: Dict[str, Tuple[float, float]],
        n_samples: int = None,
    ) -> SobolResult:
        """Run Sobol sensitivity analysis."""
        n = n_samples or self.default_n_samples
        param_names = list(parameter_bounds.keys())
        d = len(param_names)
        bounds = [parameter_bounds[name] for name in param_names]
        
        X1, X2 = generate_sobol_samples(d, n, bounds, self.seed)
        
        def array_model(X: np.ndarray) -> np.ndarray:
            results = []
            for row in X:
                params = {name: row[i] for i, name in enumerate(param_names)}
                results.append(model_func(params))
            return np.array(results)
        
        S_first, S_total, S_first_conf, S_total_conf = sobol_indices(
            array_model, X1, X2, compute_total=True
        )
        
        interaction_scores = S_total - S_first
        significant = []
        for i in range(d):
            if S_total[i] > 1e-6 and interaction_scores[i] > self.interaction_threshold * S_total[i]:
                significant.append(i)
        
        return SobolResult(
            parameter_names=param_names,
            S_first=S_first,
            S_total=S_total,
            S_first_conf=S_first_conf,
            S_total_conf=S_total_conf,
            interaction_scores=interaction_scores,
            significant_interactions=significant,
            n_samples=n,
        )


# =============================================================================
# Importance Measures Calculator
# =============================================================================

class ImportanceAnalyzer:
    """Calculate component importance measures for reliability analysis."""
    
    def __init__(self, mission_hours: float = 43800):
        self.mission_hours = mission_hours
    
    def calculate_importance(
        self,
        components: List[Dict[str, Any]],
        system_reliability: float,
        system_lambda: float,
        connection_type: str = "series"
    ) -> List[ImportanceMeasures]:
        """Calculate importance measures for all components."""
        results = []
        
        total_lambda = sum(c.get("lambda", 0) for c in components)
        if total_lambda <= 0:
            total_lambda = 1e-15
        
        # Calculate system reliability without each component
        for comp in components:
            comp_lambda = comp.get("lambda", 0)
            comp_r = comp.get("r", comp.get("reliability", 1.0))
            comp_name = comp.get("ref", comp.get("name", "Unknown"))
            comp_type = comp.get("class", comp.get("type", "Unknown"))
            
            # Contribution percentage
            contribution_pct = (comp_lambda / total_lambda * 100) if total_lambda > 0 else 0
            
            # Calculate Birnbaum importance (for series systems)
            # dR_sys/dR_comp = R_sys / R_comp (for series)
            if connection_type == "series" and comp_r > 0:
                birnbaum = system_reliability / comp_r
            else:
                birnbaum = 0.0
            
            # RAW: Risk Achievement Worth
            # System reliability if this component fails (R_comp = 0)
            if connection_type == "series":
                r_sys_failed = 0.0  # Series: any failure = system failure
            else:
                # Parallel: calculate without this component
                other_r = [c.get("r", 1.0) for c in components if c.get("ref") != comp_name]
                r_sys_failed = 1 - np.prod([1 - r for r in other_r]) if other_r else 0
            
            raw = system_reliability / r_sys_failed if r_sys_failed > 0 else float('inf')
            raw = min(raw, 1000)  # Cap at 1000 for display
            
            # RRW: Risk Reduction Worth
            # System reliability if this component is perfect (R_comp = 1)
            if connection_type == "series":
                other_r = [c.get("r", 1.0) for c in components if c.get("ref") != comp_name]
                r_sys_perfect = np.prod(other_r) if other_r else 1.0
            else:
                r_sys_perfect = 1.0  # Parallel with one perfect component = 1
            
            rrw = r_sys_perfect / system_reliability if system_reliability > 0 else 1.0
            
            # Fussell-Vesely importance
            # For series systems: (1 - R_comp) / (1 - R_sys)
            q_sys = 1 - system_reliability
            q_comp = 1 - comp_r
            fv = q_comp / q_sys if q_sys > 1e-10 else 0.0
            fv = min(fv, 1.0)  # Cap at 1.0
            
            # Determine derating recommendation
            derating_factor, action = self._get_derating_recommendation(
                comp_lambda, contribution_pct, comp_type
            )
            
            results.append(ImportanceMeasures(
                component_name=comp_name,
                lambda_fit=comp_lambda * 1e9,
                reliability=comp_r,
                contribution_pct=contribution_pct,
                birnbaum=birnbaum,
                raw=raw,
                rrw=rrw,
                fussell_vesely=fv,
                derating_factor=derating_factor,
                recommended_action=action,
            ))
        
        # Assign criticality ranks
        results.sort(key=lambda x: -x.contribution_pct)
        for i, im in enumerate(results):
            im.criticality_rank = i + 1
        
        return results
    
    def _get_derating_recommendation(
        self, 
        comp_lambda: float, 
        contribution_pct: float,
        comp_type: str
    ) -> Tuple[float, str]:
        """Generate derating recommendation based on component criticality."""
        
        if contribution_pct > 20:
            return 0.5, "CRITICAL: Apply 50% derating, consider redundancy"
        elif contribution_pct > 10:
            return 0.6, "HIGH: Apply 60% derating"
        elif contribution_pct > 5:
            return 0.7, "MODERATE: Apply 70% derating"
        elif contribution_pct > 2:
            return 0.8, "LOW: Standard 80% derating recommended"
        else:
            return 1.0, "MINIMAL: No special derating required"


# =============================================================================
# What-If Scenario Analysis
# =============================================================================

class ScenarioAnalyzer:
    """What-if scenario analysis for reliability."""
    
    def __init__(self, base_params: Dict[str, float], mission_hours: float = 43800):
        self.base_params = base_params.copy()
        self.mission_hours = mission_hours
    
    def run_scenario(
        self,
        scenario_name: str,
        description: str,
        param_changes: Dict[str, float],
        reliability_func: Callable[[Dict[str, float]], Tuple[float, float]]
    ) -> WhatIfScenario:
        """Run a what-if scenario with specified parameter changes."""
        
        # Calculate original reliability
        orig_r, orig_lam = reliability_func(self.base_params)
        
        # Apply changes
        modified_params = self.base_params.copy()
        params_changed = {}
        
        for param, new_value in param_changes.items():
            if param in modified_params:
                old_value = modified_params[param]
                modified_params[param] = new_value
                params_changed[param] = (old_value, new_value)
        
        # Calculate modified reliability
        mod_r, mod_lam = reliability_func(modified_params)
        
        # Calculate changes
        r_change = mod_r - orig_r
        r_change_pct = (r_change / orig_r * 100) if orig_r > 0 else 0
        lam_change = mod_lam - orig_lam
        lam_change_pct = (lam_change / orig_lam * 100) if orig_lam > 0 else 0
        
        return WhatIfScenario(
            scenario_name=scenario_name,
            description=description,
            original_reliability=orig_r,
            original_lambda=orig_lam,
            modified_reliability=mod_r,
            modified_lambda=mod_lam,
            reliability_change=r_change,
            reliability_change_pct=r_change_pct,
            lambda_change=lam_change,
            lambda_change_pct=lam_change_pct,
            parameters_changed=params_changed,
            improvement=mod_r > orig_r,
        )
    
    def generate_standard_scenarios(
        self,
        reliability_func: Callable[[Dict[str, float]], Tuple[float, float]]
    ) -> List[WhatIfScenario]:
        """Generate standard what-if scenarios."""
        scenarios = []
        
        # Temperature scenarios
        if "t_junction" in self.base_params:
            base_tj = self.base_params["t_junction"]
            
            scenarios.append(self.run_scenario(
                "Reduced Junction Temperature (-10°C)",
                "Improve thermal management to reduce junction temperatures",
                {"t_junction": base_tj - 10},
                reliability_func
            ))
            
            scenarios.append(self.run_scenario(
                "Increased Junction Temperature (+10°C)",
                "Worst-case thermal scenario",
                {"t_junction": base_tj + 10},
                reliability_func
            ))
        
        if "t_ambient" in self.base_params:
            base_ta = self.base_params["t_ambient"]
            
            scenarios.append(self.run_scenario(
                "Cooler Environment (-10°C)",
                "Lower ambient temperature environment",
                {"t_ambient": base_ta - 10},
                reliability_func
            ))
        
        # Thermal cycling scenarios
        if "n_cycles" in self.base_params:
            base_cycles = self.base_params["n_cycles"]
            
            scenarios.append(self.run_scenario(
                "Reduced Thermal Cycling (-50%)",
                "Improved thermal stability or GEO orbit",
                {"n_cycles": base_cycles * 0.5},
                reliability_func
            ))
            
            scenarios.append(self.run_scenario(
                "Increased Thermal Cycling (+50%)",
                "More severe thermal environment (LEO)",
                {"n_cycles": base_cycles * 1.5},
                reliability_func
            ))
        
        if "delta_t" in self.base_params:
            base_dt = self.base_params["delta_t"]
            
            scenarios.append(self.run_scenario(
                "Reduced Temperature Swing (-50%)",
                "Better thermal control",
                {"delta_t": base_dt * 0.5},
                reliability_func
            ))
        
        return scenarios


# =============================================================================
# Full Sensitivity Analysis
# =============================================================================

class SensitivityAnalyzer:
    """Complete industrial-grade sensitivity analysis."""
    
    def __init__(self, mission_hours: float = 43800, seed: int = None):
        self.mission_hours = mission_hours
        self.seed = seed
        self.sobol_analyzer = SobolAnalyzer(seed)
        self.importance_analyzer = ImportanceAnalyzer(mission_hours)
    
    def full_analysis(
        self,
        components: List[Dict[str, Any]],
        system_reliability: float,
        system_lambda: float,
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        reliability_func: Optional[Callable] = None,
        connection_type: str = "series",
        n_sobol_samples: int = 1024,
    ) -> SensitivityReport:
        """Perform complete sensitivity analysis."""
        
        report = SensitivityReport()
        report.total_components = len(components)
        
        # Calculate importance measures
        report.importance_measures = self.importance_analyzer.calculate_importance(
            components, system_reliability, system_lambda, connection_type
        )
        
        # Identify critical components
        sorted_by_contrib = sorted(report.importance_measures, key=lambda x: -x.contribution_pct)
        report.critical_by_contribution = [im.component_name for im in sorted_by_contrib[:10]]
        
        sorted_by_birnbaum = sorted(report.importance_measures, key=lambda x: -x.birnbaum)
        report.critical_by_birnbaum = [im.component_name for im in sorted_by_birnbaum[:10]]
        
        sorted_by_raw = sorted(report.importance_measures, key=lambda x: -x.raw if x.raw < 1000 else 0)
        report.critical_by_raw = [im.component_name for im in sorted_by_raw[:10]]
        
        # Summary statistics
        report.components_above_threshold = sum(1 for im in report.importance_measures if im.contribution_pct > 5)
        if sorted_by_contrib:
            report.max_contributor = sorted_by_contrib[0].component_name
            report.max_contribution_pct = sorted_by_contrib[0].contribution_pct
        
        # Sobol analysis if bounds and function provided
        if parameter_bounds and reliability_func:
            try:
                report.sobol = self.sobol_analyzer.analyze(
                    reliability_func, 
                    parameter_bounds, 
                    n_samples=n_sobol_samples
                )
            except Exception as e:
                print(f"Sobol analysis failed: {e}")
        
        return report
    
    def generate_tornado_data(
        self,
        base_params: Dict[str, float],
        reliability_func: Callable[[Dict[str, float]], float],
        variation: float = 0.2
    ) -> List[Tuple[str, float, float, float]]:
        """Generate data for tornado diagram.
        
        Returns list of (param_name, low_value, nominal_value, high_value)
        """
        nominal_r = reliability_func(base_params)
        
        tornado_data = []
        
        for param, nominal_val in base_params.items():
            if nominal_val == 0:
                continue
            
            # Calculate reliability at ±variation
            low_params = base_params.copy()
            high_params = base_params.copy()
            
            low_params[param] = nominal_val * (1 - variation)
            high_params[param] = nominal_val * (1 + variation)
            
            low_r = reliability_func(low_params)
            high_r = reliability_func(high_params)
            
            tornado_data.append((param, low_r, nominal_r, high_r))
        
        # Sort by total range
        tornado_data.sort(key=lambda x: -(abs(x[3] - x[1])))
        
        return tornado_data[:15]  # Top 15


def quick_sensitivity(
    lambda_components: Dict[str, float],
    mission_hours: float,
    uncertainty_range: float = 0.3,
    n_samples: int = 1024,
) -> SobolResult:
    """Quick sensitivity analysis on component failure rates."""
    from .reliability_math import reliability_from_lambda
    
    bounds = {}
    for name, lam in lambda_components.items():
        bounds[name] = (lam * (1 - uncertainty_range), lam * (1 + uncertainty_range))
    
    def model(sampled: Dict[str, float]) -> float:
        total_lambda = sum(sampled.values())
        return reliability_from_lambda(total_lambda, mission_hours)
    
    analyzer = SobolAnalyzer()
    return analyzer.analyze(model, bounds, n_samples)


def print_importance_report(measures: List[ImportanceMeasures], max_rows: int = 20):
    """Pretty print importance measures."""
    print("\n" + "="*90)
    print("COMPONENT IMPORTANCE ANALYSIS")
    print("="*90)
    print(f"\n{'Rank':<5} {'Component':<20} {'λ (FIT)':<12} {'Contrib%':<10} {'Birnbaum':<10} {'RAW':<10} {'RRW':<10}")
    print("-"*90)
    
    for im in measures[:max_rows]:
        raw_str = f"{im.raw:.2f}" if im.raw < 100 else "∞"
        print(f"{im.criticality_rank:<5} {im.component_name:<20} {im.lambda_fit:<12.2f} {im.contribution_pct:<10.1f} {im.birnbaum:<10.4f} {raw_str:<10} {im.rrw:<10.4f}")
    
    print("="*90)
    print("\nLegend:")
    print("  Birnbaum: System reliability sensitivity to component reliability")
    print("  RAW: Risk Achievement Worth - increase in risk if component fails")
    print("  RRW: Risk Reduction Worth - risk reduction if component is perfect")


def print_sobol_results(result: SobolResult, max_rows: int = 15):
    """Pretty print Sobol results."""
    print("\n" + "="*70)
    print("SOBOL SENSITIVITY ANALYSIS RESULTS")
    print("="*70)
    print(f"Samples: {result.n_samples}")
    print("\n{:<25} {:>12} {:>12} {:>12}".format("Parameter", "S_first", "S_total", "Interaction"))
    print("-"*70)
    
    ranked = sorted(enumerate(result.S_total), key=lambda x: -x[1])
    
    for idx, _ in ranked[:max_rows]:
        name = result.parameter_names[idx]
        s1 = result.S_first[idx]
        st = result.S_total[idx]
        inter = result.interaction_scores[idx]
        flag = " ***" if idx in result.significant_interactions else ""
        print(f"{name:<25} {s1:>12.4f} {st:>12.4f} {inter:>12.4f}{flag}")
    
    if result.significant_interactions:
        print("\n*** Parameters with significant interactions")
    
    print("="*70)


if __name__ == "__main__":
    # Test with example components
    components = [
        {"ref": "U1", "class": "MCU", "lambda": 50e-9, "r": 0.9978},
        {"ref": "U2", "class": "Power Supply", "lambda": 100e-9, "r": 0.9956},
        {"ref": "U3", "class": "Memory", "lambda": 30e-9, "r": 0.9987},
        {"ref": "R1", "class": "Resistor", "lambda": 0.1e-9, "r": 0.99999},
        {"ref": "C1", "class": "Capacitor", "lambda": 0.5e-9, "r": 0.99997},
    ]
    
    system_r = 0.9921
    system_lam = 180.6e-9
    
    analyzer = SensitivityAnalyzer(mission_hours=43800)
    report = analyzer.full_analysis(components, system_r, system_lam)
    
    print_importance_report(report.importance_measures)
    
    print(f"\nCritical by Contribution: {report.critical_by_contribution[:5]}")
    print(f"Components > 5% contribution: {report.components_above_threshold}")
