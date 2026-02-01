"""
Monte Carlo Uncertainty Analysis Module
=======================================
Provides uncertainty quantification for reliability predictions with convergence detection.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional
import time

@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    mean: float
    std: float
    percentile_5: float
    percentile_50: float
    percentile_95: float
    samples: np.ndarray
    converged: bool
    n_simulations: int
    convergence_history: List[Tuple[int, float]] = field(default_factory=list)
    runtime_seconds: float = 0.0
    
    @property
    def percentiles(self) -> Dict[int, float]:
        """Return percentiles as a dict for convenience."""
        return {5: self.percentile_5, 50: self.percentile_50, 95: self.percentile_95}
    
    def confidence_interval(self, level: float = 0.90) -> Tuple[float, float]:
        alpha = (1 - level) / 2
        return (np.percentile(self.samples, alpha * 100), np.percentile(self.samples, (1 - alpha) * 100))
    
    def to_dict(self) -> Dict:
        return {
            "mean": self.mean, "std": self.std,
            "percentile_5": self.percentile_5, "percentile_50": self.percentile_50,
            "percentile_95": self.percentile_95, "converged": self.converged,
            "n_simulations": self.n_simulations, "runtime_seconds": self.runtime_seconds,
        }


@dataclass
class ParameterDistribution:
    """Definition of uncertain parameter distribution."""
    name: str
    distribution: str  # "uniform", "normal", "triangular", "lognormal"
    params: Dict[str, float]  # Distribution-specific parameters
    
    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        if self.distribution == "uniform":
            return rng.uniform(self.params["low"], self.params["high"], n)
        elif self.distribution == "normal":
            return rng.normal(self.params["mean"], self.params["std"], n)
        elif self.distribution == "triangular":
            return rng.triangular(self.params["low"], self.params["mode"], self.params["high"], n)
        elif self.distribution == "lognormal":
            return rng.lognormal(self.params["mean"], self.params["sigma"], n)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


class MonteCarloAnalyzer:
    """Monte Carlo uncertainty analysis with automatic convergence detection."""
    
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self.default_max_simulations = 10000
        self.default_min_simulations = 500
        self.default_convergence_threshold = 0.001
        self.default_check_interval = 100
    
    def run_analysis(
        self,
        model_func: Callable[[Dict[str, float]], float],
        parameters: List[ParameterDistribution],
        max_simulations: int = None,
        min_simulations: int = None,
        convergence_threshold: float = None,
        check_interval: int = None,
        callback: Callable[[int, float], None] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo analysis with automatic convergence detection.
        
        Args:
            model_func: Function that takes parameter dict and returns reliability/lambda
            parameters: List of parameter distributions
            max_simulations: Maximum number of simulations
            min_simulations: Minimum before checking convergence
            convergence_threshold: Relative change threshold for convergence
            check_interval: How often to check convergence
            callback: Optional callback(iteration, current_mean)
        """
        max_sims = max_simulations or self.default_max_simulations
        min_sims = min_simulations or self.default_min_simulations
        threshold = convergence_threshold or self.default_convergence_threshold
        interval = check_interval or self.default_check_interval
        
        start_time = time.time()
        results = []
        convergence_history = []
        converged = False
        
        for i in range(max_sims):
            # Sample parameters
            param_values = {}
            for param in parameters:
                param_values[param.name] = param.sample(self.rng, 1)[0]
            
            # Evaluate model
            try:
                result = model_func(param_values)
                results.append(result)
            except Exception:
                continue
            
            # Check convergence periodically
            if i >= min_sims and (i + 1) % interval == 0:
                current_mean = np.mean(results)
                convergence_history.append((i + 1, current_mean))
                
                if len(convergence_history) >= 2:
                    prev_mean = convergence_history[-2][1]
                    if prev_mean != 0:
                        rel_change = abs(current_mean - prev_mean) / abs(prev_mean)
                        if rel_change < threshold:
                            converged = True
                            break
                
                if callback:
                    callback(i + 1, current_mean)
        
        samples = np.array(results)
        runtime = time.time() - start_time
        
        return MonteCarloResult(
            mean=np.mean(samples),
            std=np.std(samples),
            percentile_5=np.percentile(samples, 5),
            percentile_50=np.percentile(samples, 50),
            percentile_95=np.percentile(samples, 95),
            samples=samples,
            converged=converged,
            n_simulations=len(samples),
            convergence_history=convergence_history,
            runtime_seconds=runtime,
        )
    
    def run_system_analysis(
        self,
        components: List[Dict],
        system_structure: str,  # "series", "parallel", "k_of_n"
        mission_hours: float,
        parameter_uncertainties: Dict[str, ParameterDistribution],
        k_value: int = None,
        **kwargs
    ) -> MonteCarloResult:
        """Run Monte Carlo on a complete system."""
        from .reliability_math import (
            calculate_component_lambda, reliability_from_lambda,
            r_series, r_parallel, r_k_of_n
        )
        
        def model_func(sampled_params: Dict[str, float]) -> float:
            component_reliabilities = []
            
            for comp in components:
                # Merge base params with sampled uncertainties
                params = comp.get("params", {}).copy()
                for key, value in sampled_params.items():
                    if key in params or key in ["t_ambient", "t_junction", "n_cycles", "delta_t"]:
                        params[key] = value
                
                # Calculate component reliability
                result = calculate_component_lambda(comp.get("type", "Resistor"), params)
                lam = result["lambda_total"]
                r = reliability_from_lambda(lam, mission_hours)
                component_reliabilities.append(r)
            
            # Calculate system reliability
            if system_structure == "series":
                return r_series(component_reliabilities)
            elif system_structure == "parallel":
                return r_parallel(component_reliabilities)
            elif system_structure == "k_of_n" and k_value:
                return r_k_of_n(component_reliabilities, k_value)
            else:
                return r_series(component_reliabilities)
        
        # Convert parameter_uncertainties to list
        param_list = list(parameter_uncertainties.values())
        
        return self.run_analysis(model_func, param_list, **kwargs)


# === Preset distributions for common parameters ===

def temperature_uncertainty(nominal: float, tolerance: float = 10.0) -> ParameterDistribution:
    """Create temperature uncertainty distribution (uniform ± tolerance)."""
    return ParameterDistribution(
        name="t_ambient" if nominal < 50 else "t_junction",
        distribution="uniform",
        params={"low": nominal - tolerance, "high": nominal + tolerance}
    )

def thermal_cycles_uncertainty(nominal: int, cv: float = 0.2) -> ParameterDistribution:
    """Create thermal cycles uncertainty (lognormal with coefficient of variation)."""
    mu = np.log(nominal) - 0.5 * np.log(1 + cv**2)
    sigma = np.sqrt(np.log(1 + cv**2))
    return ParameterDistribution(
        name="n_cycles",
        distribution="lognormal",
        params={"mean": mu, "sigma": sigma}
    )

def delta_t_uncertainty(nominal: float, low_mult: float = 0.5, high_mult: float = 2.0) -> ParameterDistribution:
    """Create ΔT uncertainty (triangular from low to high)."""
    return ParameterDistribution(
        name="delta_t",
        distribution="triangular",
        params={"low": nominal * low_mult, "mode": nominal, "high": nominal * high_mult}
    )


# === Quick analysis functions ===

def quick_monte_carlo(
    lambda_total: float,
    mission_hours: float,
    uncertainty_percent: float = 20.0,
    n_simulations: int = 5000,
    seed: int = None,
) -> MonteCarloResult:
    """Quick Monte Carlo with simple percentage uncertainty on failure rate."""
    rng = np.random.default_rng(seed)
    
    # Sample lambda with lognormal distribution
    cv = uncertainty_percent / 100.0
    mu = np.log(lambda_total) - 0.5 * np.log(1 + cv**2)
    sigma = np.sqrt(np.log(1 + cv**2))
    
    lambda_samples = rng.lognormal(mu, sigma, n_simulations)
    reliability_samples = np.exp(-lambda_samples * mission_hours)
    
    return MonteCarloResult(
        mean=np.mean(reliability_samples),
        std=np.std(reliability_samples),
        percentile_5=np.percentile(reliability_samples, 5),
        percentile_50=np.percentile(reliability_samples, 50),
        percentile_95=np.percentile(reliability_samples, 95),
        samples=reliability_samples,
        converged=True,
        n_simulations=n_simulations,
    )


if __name__ == "__main__":
    # Quick test
    result = quick_monte_carlo(1e-7, 43800, uncertainty_percent=30.0)
    print(f"Mean R: {result.mean:.6f}")
    print(f"90% CI: [{result.percentile_5:.6f}, {result.percentile_95:.6f}]")
    print(f"Std: {result.std:.6f}")
