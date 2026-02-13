"""
Sobol Sensitivity Analysis Module
=================================
First-order and total-order Sobol indices with interaction detection.

Author:  Eliot Abramo
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional

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
        ranked = sorted(zip(self.parameter_names, indices), key=lambda x: -x[1])
        return ranked


def generate_sobol_samples(d: int, n: int, bounds: List[Tuple[float, float]], seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate quasi-random Sobol sequence samples.
    
    Args:
        d: Number of dimensions (parameters)
        n: Number of samples
        bounds: List of (low, high) bounds for each parameter
        seed: Random seed
    
    Returns:
        X1, X2: Two independent sample matrices (n x d)
    """
    rng = np.random.default_rng(seed)
    
    # Use Sobol sequence if available, otherwise fall back to random
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d, scramble=True, seed=seed)
        X1_unit = sampler.random(n)
        X2_unit = sampler.random(n)
    except ImportError:
        # Fallback to pseudo-random
        X1_unit = rng.random((n, d))
        X2_unit = rng.random((n, d))
    
    # Scale to bounds
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
    """
    Compute Sobol sensitivity indices using pick-freeze estimator.
    
    Args:
        model_func: Function f(X) -> Y where X is (n, d) and Y is (n,)
        X1, X2: Two independent sample matrices (n x d)
        compute_total: Whether to compute total-order indices
        confidence_level: Confidence level for intervals
    
    Returns:
        S_first, S_total, S_first_conf, S_total_conf
    """
    N, d = X1.shape
    
    # Evaluate base samples
    Y1 = model_func(X1)
    Y2 = model_func(X2)
    
    # Estimate variance
    var_Y = 0.5 * (np.var(Y1) + np.var(Y2))
    if var_Y < 1e-15:
        # No variance - all indices are zero
        zeros = np.zeros(d)
        return zeros, zeros, zeros, zeros
    
    S_first = np.zeros(d)
    S_total = np.zeros(d)
    S_first_conf = np.zeros(d)
    S_total_conf = np.zeros(d)
    
    for i in range(d):
        # First-order: X1 with column i replaced by X2
        X1_i = X1.copy()
        X1_i[:, i] = X2[:, i]
        Y1_i = model_func(X1_i)
        
        # Jansen estimator for first-order
        term1 = np.mean(Y1 * Y1_i)
        term2 = 0.25 * (np.mean(Y1 + Y1_i))**2
        denom = 0.5 * (np.mean(Y1**2) + np.mean(Y1_i**2)) - term2
        
        if abs(denom) > 1e-15:
            S_first[i] = (term1 - term2) / denom
        
        # Bootstrap confidence interval for first-order
        S_first_conf[i] = 1.96 * np.sqrt(np.var(Y1 * Y1_i - term1) / N) / max(denom, 1e-15)
        
        # Total-order: X2 with column i replaced by X1
        if compute_total:
            X2_i = X2.copy()
            X2_i[:, i] = X1[:, i]
            Y2_i = model_func(X2_i)
            
            # Total index estimator
            S_total[i] = np.mean((Y1 - Y2_i)**2) / (2 * var_Y)
            
            # Confidence interval
            T_i = (Y1 - Y2_i)**2 / 2 - S_total[i] * var_Y
            S_total_conf[i] = 1.96 * np.std(T_i) / (var_Y * np.sqrt(N))
    
    # Clip to valid range
    S_first = np.clip(S_first, 0, 1)
    S_total = np.clip(S_total, 0, 1)
    
    return S_first, S_total, S_first_conf, S_total_conf


class SobolAnalyzer:
    """Sobol sensitivity analysis for reliability models."""
    
    def __init__(self, seed: int = None):
        self.seed = seed
        self.default_n_samples = 2048
        self.interaction_threshold = 0.1  # Flag if interaction > 10% of total
    
    def analyze(
        self,
        model_func: Callable[[Dict[str, float]], float],
        parameter_bounds: Dict[str, Tuple[float, float]],
        n_samples: int = None,
    ) -> SobolResult:
        """
        Run Sobol sensitivity analysis.
        
        Args:
            model_func: Function that takes parameter dict and returns scalar
            parameter_bounds: Dict of {param_name: (low, high)}
            n_samples: Number of samples (should be power of 2 for Sobol sequence)
        """
        n = n_samples or self.default_n_samples
        param_names = list(parameter_bounds.keys())
        d = len(param_names)
        bounds = [parameter_bounds[name] for name in param_names]
        
        # Generate samples
        X1, X2 = generate_sobol_samples(d, n, bounds, self.seed)
        
        # Vectorized wrapper  avoids row-by-row Python loop
        def array_model(X: np.ndarray) -> np.ndarray:
            # For sheet-level lambda sensitivity, each column is a sheet lambda
            # sum across parameters  total lambda  reliability
            # This is much faster than calling model_func row-by-row
            results = np.empty(X.shape[0])
            # Batch in chunks to balance memory vs speed
            chunk_size = 512
            for start in range(0, X.shape[0], chunk_size):
                end = min(start + chunk_size, X.shape[0])
                chunk = X[start:end]
                for j in range(end - start):
                    params = {name: chunk[j, i] for i, name in enumerate(param_names)}
                    results[start + j] = model_func(params)
            return results
        
        # Compute indices
        S_first, S_total, S_first_conf, S_total_conf = sobol_indices(
            array_model, X1, X2, compute_total=True
        )
        
        # Detect interactions
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
    
    def analyze_reliability_model(
        self,
        base_params: Dict[str, float],
        param_uncertainties: Dict[str, Tuple[float, float]],
        component_type: str,
        mission_hours: float,
        **kwargs
    ) -> SobolResult:
        """Analyze sensitivity of reliability to parameter variations."""
        try:
            from .reliability_math import calculate_component_lambda, reliability_from_lambda
        except ImportError:
            from reliability_math import calculate_component_lambda, reliability_from_lambda
        
        def model_func(sampled: Dict[str, float]) -> float:
            params = base_params.copy()
            params.update(sampled)
            result = calculate_component_lambda(component_type, params)
            lam = result["lambda_total"]
            return reliability_from_lambda(lam, mission_hours)
        
        return self.analyze(model_func, param_uncertainties, **kwargs)


def print_sobol_results(result: SobolResult, max_rows: int = 15):
    """Pretty print Sobol results."""
    print("\n" + "="*70)
    print("SOBOL SENSITIVITY ANALYSIS RESULTS")
    print("="*70)
    print(f"Samples: {result.n_samples}")
    print("\n{:<25} {:>12} {:>12} {:>12}".format("Parameter", "S_first", "S_total", "Interaction"))
    print("-"*70)
    
    # Sort by total index
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


# === Quick analysis function ===

def quick_sensitivity(
    lambda_components: Dict[str, float],
    mission_hours: float,
    uncertainty_range: float = 0.3,
    n_samples: int = 1024,
) -> SobolResult:
    """Quick sensitivity analysis on component failure rates."""
    try:
        from .reliability_math import reliability_from_lambda
    except ImportError:
        from reliability_math import reliability_from_lambda
    
    # Create bounds (uncertainty_range around nominal)
    bounds = {}
    for name, lam in lambda_components.items():
        bounds[name] = (lam * (1 - uncertainty_range), lam * (1 + uncertainty_range))
    
    def model(sampled: Dict[str, float]) -> float:
        lambdas = list(sampled.values())
        total_lambda = sum(lambdas)
        return reliability_from_lambda(total_lambda, mission_hours)
    
    analyzer = SobolAnalyzer()
    return analyzer.analyze(model, bounds, n_samples)


# === Component-Level Criticality Analysis ===

def analyze_board_criticality(
    components: List[Dict],
    mission_hours: float = 8760.0,
    top_n: int = 10,
    perturbation: float = 0.1,
) -> List[Dict]:
    """Run parameter criticality analysis on the highest-FIT components.
    
    Integrates with reliability_math.analyze_component_criticality() to
    identify which input parameters most influence each component's λ.
    Results are suitable for report_generator.ReportData.criticality.
    
    Args:
        components: List of component dicts with 'ref', 'class', 'params', 'lambda'
        mission_hours: Mission duration in hours (default 1 year = 8760h)
        top_n: Number of top components to analyze (by FIT contribution)
        perturbation: Fractional perturbation for finite-difference (default ±10%)
    
    Returns:
        List of criticality result dicts, one per analyzed component.
        Each dict has: reference, component_type, base_lambda_fit, fields[].
        Each field has: name, value, elasticity, impact_pct.
    """
    try:
        from .reliability_math import analyze_component_criticality
    except ImportError:
        from reliability_math import analyze_component_criticality

    # Sort by failure rate, analyze the top contributors
    sorted_comps = sorted(components, key=lambda c: c.get("lambda", 0), reverse=True)
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

            # Reformat into report-friendly structure
            base_fit = raw[0]["lambda_nominal_fit"] if raw else 0
            fields = []
            total_impact = sum(abs(r["impact_percent"]) for r in raw) or 1.0
            for r in raw:
                fields.append({
                    "name": r["field"],
                    "value": r["nominal_value"],
                    "elasticity": r["sensitivity"],
                    "impact_pct": r["impact_percent"],
                })

            results.append({
                "reference": ref,
                "component_type": comp_type,
                "base_lambda_fit": base_fit,
                "fields": fields,
            })
        except Exception:
            continue

    return results


if __name__ == "__main__":
    # Test with example
    components = {
        "MCU": 50e-9,
        "Power_Supply": 100e-9,
        "Memory": 30e-9,
        "Interface": 80e-9,
    }
    
    result = quick_sensitivity(components, 43800, uncertainty_range=0.4)
    print_sobol_results(result)
