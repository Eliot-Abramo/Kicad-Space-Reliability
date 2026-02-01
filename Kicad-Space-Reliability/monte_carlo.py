"""
Monte Carlo Uncertainty Analysis Module
=======================================
Provides uncertainty quantification for reliability predictions with convergence detection.

This module implements proper component-level Monte Carlo that propagates parameter
uncertainty through IEC TR 62380 formulas, NOT just simple noise on the total lambda.
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


@dataclass
class ComponentMCInput:
    """Component input for Monte Carlo analysis."""
    reference: str
    component_type: str
    base_params: Dict[str, float]
    # Parameter uncertainties as (nominal, cv) tuples
    # CV = coefficient of variation = std/mean
    uncertainties: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass 
class SheetMCResult:
    """Monte Carlo results for a single sheet/block."""
    sheet_path: str
    mc_result: MonteCarloResult
    lambda_samples: np.ndarray  # For detailed analysis


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
        """Run Monte Carlo analysis with automatic convergence detection."""
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


# === Import helper ===

def _import_reliability_math():
    """Import reliability_math module (handles both package and standalone)."""
    try:
        from .reliability_math import calculate_component_lambda, reliability_from_lambda
        return calculate_component_lambda, reliability_from_lambda
    except ImportError:
        from reliability_math import calculate_component_lambda, reliability_from_lambda
        return calculate_component_lambda, reliability_from_lambda


def _import_classify_component():
    """Import classify_component (handles both package and standalone)."""
    try:
        from .component_editor import classify_component
        return classify_component
    except ImportError:
        try:
            from component_editor import classify_component
            return classify_component
        except ImportError:
            # Fallback if component_editor not available
            def classify_component(ref, value, fields):
                ref = ref.upper()
                if ref.startswith('R'): return 'Resistor'
                if ref.startswith('C'): return 'Capacitor'
                if ref.startswith('L'): return 'Inductor/Transformer'
                if ref.startswith('D'): return 'Diode'
                if ref.startswith('Q') or ref.startswith('T'): return 'Transistor'
                if ref.startswith('U'): return 'Integrated Circuit'
                return 'Resistor'
            return classify_component


# === Proper Monte Carlo with Component-Level Uncertainty Propagation ===

def monte_carlo_components(
    components: List[ComponentMCInput],
    mission_hours: float,
    n_simulations: int = 5000,
    uncertainty_percent: float = 20.0,
    seed: int = None,
    progress_callback: Callable[[int, int], None] = None,
) -> Tuple[MonteCarloResult, np.ndarray]:
    """
    Proper Monte Carlo that propagates uncertainty through component-level calculations.
    
    This is the CORRECT implementation that:
    1. Samples uncertain parameters for each component
    2. Recalculates λ using IEC TR 62380 formulas for each sample
    3. Sums component lambdas for series reliability
    4. Returns proper uncertainty distribution
    
    Args:
        components: List of component inputs with parameters
        mission_hours: Mission duration in hours
        n_simulations: Number of Monte Carlo iterations
        uncertainty_percent: Default uncertainty CV if not specified per-component
        seed: Random seed for reproducibility
        progress_callback: Optional callback(current, total) for progress updates
        
    Returns:
        Tuple of (MonteCarloResult for system, array of lambda samples)
    """
    calculate_component_lambda, reliability_from_lambda = _import_reliability_math()
    
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    n_components = len(components)
    default_cv = uncertainty_percent / 100.0
    
    # Key parameters that affect reliability
    uncertain_params = ['t_junction', 't_ambient', 'n_cycles', 'delta_t', 
                        'operating_power', 'voltage_stress_vds', 'voltage_stress_vgs',
                        'voltage_stress_vce', 'ripple_ratio']
    
    # Storage for results
    system_lambda_samples = np.zeros(n_simulations)
    system_r_samples = np.zeros(n_simulations)
    convergence_history = []
    
    # Main Monte Carlo loop
    for sim_idx in range(n_simulations):
        total_lambda = 0.0
        
        for comp in components:
            # Start with base parameters
            sampled_params = comp.base_params.copy()
            
            # Sample uncertain parameters
            for param_name in uncertain_params:
                if param_name in sampled_params:
                    nominal = sampled_params[param_name]
                    
                    # Get CV from component uncertainties or use default
                    if param_name in comp.uncertainties:
                        _, cv = comp.uncertainties[param_name]
                    else:
                        cv = default_cv
                    
                    if nominal > 0 and cv > 0:
                        # Use lognormal for positive parameters (ensures positive samples)
                        mu = np.log(nominal) - 0.5 * np.log(1 + cv**2)
                        sigma = np.sqrt(np.log(1 + cv**2))
                        sampled_params[param_name] = rng.lognormal(mu, sigma)
                    elif nominal != 0:
                        # Normal for parameters that can be negative
                        sampled_params[param_name] = rng.normal(nominal, abs(nominal) * cv)
            
            # Calculate component lambda with sampled parameters
            try:
                result = calculate_component_lambda(comp.component_type, sampled_params)
                comp_lambda = result.get('lambda_total', 0)
                total_lambda += comp_lambda
            except Exception:
                # Fallback to base calculation if sampling causes issues
                result = calculate_component_lambda(comp.component_type, comp.base_params)
                total_lambda += result.get('lambda_total', 0)
        
        system_lambda_samples[sim_idx] = total_lambda
        system_r_samples[sim_idx] = reliability_from_lambda(total_lambda, mission_hours)
        
        # Track convergence every 100 iterations after minimum
        if sim_idx >= 500 and (sim_idx + 1) % 100 == 0:
            current_mean = np.mean(system_r_samples[:sim_idx+1])
            convergence_history.append((sim_idx + 1, current_mean))
        
        # Progress callback
        if progress_callback and (sim_idx + 1) % 100 == 0:
            progress_callback(sim_idx + 1, n_simulations)
    
    runtime = time.time() - start_time
    
    # Check convergence
    converged = False
    if len(convergence_history) >= 3:
        last_means = [h[1] for h in convergence_history[-3:]]
        rel_changes = [abs(last_means[i] - last_means[i-1]) / abs(last_means[i-1]) 
                      for i in range(1, len(last_means)) if last_means[i-1] != 0]
        if rel_changes and max(rel_changes) < 0.001:
            converged = True
    
    mc_result = MonteCarloResult(
        mean=np.mean(system_r_samples),
        std=np.std(system_r_samples),
        percentile_5=np.percentile(system_r_samples, 5),
        percentile_50=np.percentile(system_r_samples, 50),
        percentile_95=np.percentile(system_r_samples, 95),
        samples=system_r_samples,
        converged=converged,
        n_simulations=n_simulations,
        convergence_history=convergence_history,
        runtime_seconds=runtime,
    )
    
    return mc_result, system_lambda_samples


def monte_carlo_sheet(
    sheet_components: List[Dict],
    mission_hours: float,
    n_simulations: int = 2000,
    uncertainty_percent: float = 20.0,
    seed: int = None,
) -> Tuple[MonteCarloResult, np.ndarray]:
    """
    Run Monte Carlo for a single sheet's components.
    
    Args:
        sheet_components: List of component dicts with 'ref', 'class', 'params'
        mission_hours: Mission duration
        n_simulations: Number of iterations (default lower for per-sheet)
        uncertainty_percent: Uncertainty CV percentage
        seed: Random seed
        
    Returns:
        Tuple of (MonteCarloResult, lambda_samples array)
    """
    classify_component = _import_classify_component()
    
    mc_inputs = []
    for comp in sheet_components:
        comp_type = comp.get('class', 'Resistor')
        if not comp_type or comp_type == 'Unknown':
            comp_type = classify_component(comp.get('ref', 'R1'), comp.get('value', ''), {})
        
        base_params = comp.get('params', {})
        if not base_params:
            # Default parameters
            base_params = {
                't_ambient': 25.0,
                't_junction': 85.0,
                'n_cycles': 5256,
                'delta_t': 3.0,
                'operating_power': 0.01,
                'rated_power': 0.125,
            }
        
        mc_inputs.append(ComponentMCInput(
            reference=comp.get('ref', '?'),
            component_type=comp_type,
            base_params=base_params,
        ))
    
    return monte_carlo_components(
        mc_inputs, mission_hours, n_simulations, uncertainty_percent, seed
    )


def quick_monte_carlo(
    lambda_total: float,
    mission_hours: float,
    uncertainty_percent: float = 20.0,
    n_simulations: int = 5000,
    seed: int = None,
    components: List[Dict] = None,
) -> MonteCarloResult:
    """
    Monte Carlo uncertainty analysis.
    
    If components are provided, runs proper uncertainty propagation through
    component-level calculations (slower but correct).
    
    If only lambda_total is provided, uses simplified lognormal sampling
    (faster but less accurate - suitable for quick estimates only).
    
    Args:
        lambda_total: Total system failure rate (used if no components)
        mission_hours: Mission duration in hours
        uncertainty_percent: Uncertainty as percentage (CV * 100)
        n_simulations: Number of Monte Carlo iterations
        seed: Random seed for reproducibility
        components: Optional list of component dicts for proper analysis
        
    Returns:
        MonteCarloResult with distribution statistics
    """
    _, reliability_from_lambda = _import_reliability_math()
    
    start_time = time.time()
    
    if components and len(components) > 0:
        # Proper component-level Monte Carlo
        mc_inputs = []
        for comp in components:
            comp_type = comp.get('type', comp.get('class', 'Resistor'))
            base_params = comp.get('params', {})
            
            # Ensure minimum parameters
            if 't_junction' not in base_params and 't_ambient' not in base_params:
                base_params['t_ambient'] = 25.0
            if 'n_cycles' not in base_params:
                base_params['n_cycles'] = 5256
            if 'delta_t' not in base_params:
                base_params['delta_t'] = 3.0
                
            mc_inputs.append(ComponentMCInput(
                reference=comp.get('ref', comp.get('reference', '?')),
                component_type=comp_type,
                base_params=base_params,
            ))
        
        result, _ = monte_carlo_components(
            mc_inputs, mission_hours, n_simulations, uncertainty_percent, seed
        )
        return result
    
    else:
        # Simplified sampling (legacy behavior)
        # This just adds noise to the total lambda, NOT proper uncertainty propagation
        rng = np.random.default_rng(seed)
        
        cv = uncertainty_percent / 100.0
        if lambda_total > 0:
            mu = np.log(lambda_total) - 0.5 * np.log(1 + cv**2)
            sigma = np.sqrt(np.log(1 + cv**2))
            lambda_samples = rng.lognormal(mu, sigma, n_simulations)
        else:
            lambda_samples = np.zeros(n_simulations)
        
        reliability_samples = np.exp(-lambda_samples * mission_hours)
        runtime = time.time() - start_time
        
        return MonteCarloResult(
            mean=np.mean(reliability_samples),
            std=np.std(reliability_samples),
            percentile_5=np.percentile(reliability_samples, 5),
            percentile_50=np.percentile(reliability_samples, 50),
            percentile_95=np.percentile(reliability_samples, 95),
            samples=reliability_samples,
            converged=True,
            n_simulations=n_simulations,
            runtime_seconds=runtime,
        )


def monte_carlo_blocks(
    block_data: Dict[str, Dict],
    mission_hours: float,
    n_simulations: int = 2000,
    uncertainty_percent: float = 20.0,
    seed: int = None,
    progress_callback: Callable[[str, int, int], None] = None,
) -> Dict[str, SheetMCResult]:
    """
    Run Monte Carlo for all blocks/sheets in the system.
    
    Args:
        block_data: Dict mapping sheet_path to {'components': [...], 'lambda': ..., 'r': ...}
        mission_hours: Mission duration
        n_simulations: Simulations per sheet (reduced for speed)
        uncertainty_percent: Uncertainty CV
        seed: Base random seed (incremented per sheet)
        progress_callback: Optional callback(sheet_path, current, total)
        
    Returns:
        Dict mapping sheet_path to SheetMCResult
    """
    results = {}
    base_seed = seed if seed is not None else int(time.time())
    
    for idx, (sheet_path, data) in enumerate(block_data.items()):
        components = data.get('components', [])
        if not components:
            continue
        
        # Convert component format
        sheet_components = []
        for c in components:
            sheet_components.append({
                'ref': c.get('ref', '?'),
                'value': c.get('value', ''),
                'class': c.get('class', 'Resistor'),
                'params': c.get('params', {}),
            })
        
        def sheet_progress(current, total):
            if progress_callback:
                progress_callback(sheet_path, current, total)
        
        mc_result, lambda_samples = monte_carlo_sheet(
            sheet_components,
            mission_hours,
            n_simulations,
            uncertainty_percent,
            seed=base_seed + idx,
        )
        
        results[sheet_path] = SheetMCResult(
            sheet_path=sheet_path,
            mc_result=mc_result,
            lambda_samples=lambda_samples,
        )
    
    return results


if __name__ == "__main__":
    # Test proper Monte Carlo
    print("Testing proper component-level Monte Carlo...")
    
    from reliability_math import calculate_component_lambda, reliability_from_lambda
    
    # Create test components
    test_components = [
        ComponentMCInput("R1", "Resistor", {'t_ambient': 25, 'operating_power': 0.01, 'rated_power': 0.125, 'n_cycles': 5256, 'delta_t': 3.0}),
        ComponentMCInput("R2", "Resistor", {'t_ambient': 25, 'operating_power': 0.01, 'rated_power': 0.125, 'n_cycles': 5256, 'delta_t': 3.0}),
        ComponentMCInput("C1", "Capacitor", {'t_ambient': 25, 'n_cycles': 5256, 'delta_t': 3.0, 'capacitor_type': 'Ceramic Class II (X7R/X5R)'}),
        ComponentMCInput("U1", "Integrated Circuit", {'t_junction': 85, 'n_cycles': 5256, 'delta_t': 3.0, 'transistor_count': 10000}),
    ]
    
    result, lambda_samples = monte_carlo_components(
        test_components, 
        mission_hours=43800,  # 5 years
        n_simulations=2000,
        uncertainty_percent=20.0,
        seed=42
    )
    
    print(f"\nResults (5-year mission, 2000 simulations):")
    print(f"  Mean R: {result.mean:.6f}")
    print(f"  Std:    {result.std:.6f}")
    print(f"  5%:     {result.percentile_5:.6f}")
    print(f"  50%:    {result.percentile_50:.6f}")
    print(f"  95%:    {result.percentile_95:.6f}")
    print(f"  Runtime: {result.runtime_seconds:.2f}s")
    print(f"  Converged: {result.converged}")
    
    # Calculate nominal for comparison
    total_lambda = 0
    for comp in test_components:
        res = calculate_component_lambda(comp.component_type, comp.base_params)
        total_lambda += res['lambda_total']
    
    nominal_r = reliability_from_lambda(total_lambda, 43800)
    print(f"\nNominal R (no uncertainty): {nominal_r:.6f}")
    print(f"Total λ: {total_lambda:.2e}")
