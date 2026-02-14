"""
Monte Carlo Uncertainty Analysis Module
=======================================
Component-level uncertainty propagation through IEC TR 62380 formulas.
Uses vectorized NumPy operations for performance.

Supports:
  - Configurable confidence intervals
  - Component override_lambda (fixed industrial values)
  - Active-sheet filtering

Author:  Eliot Abramo
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
    confidence_level: float = 0.90
    ci_lower: float = 0.0
    ci_upper: float = 0.0

    @property
    def percentiles(self) -> Dict[int, float]:
        return {5: self.percentile_5, 50: self.percentile_50, 95: self.percentile_95}

    def confidence_interval(self, level: float = None) -> Tuple[float, float]:
        if level is None:
            level = self.confidence_level
        alpha = (1 - level) / 2
        return (
            float(np.percentile(self.samples, alpha * 100)),
            float(np.percentile(self.samples, (1 - alpha) * 100)),
        )

    def to_dict(self) -> Dict:
        ci_lo, ci_hi = self.confidence_interval()
        return {
            "mean": self.mean,
            "std": self.std,
            "percentile_5": self.percentile_5,
            "percentile_50": self.percentile_50,
            "percentile_95": self.percentile_95,
            "converged": self.converged,
            "n_simulations": self.n_simulations,
            "runtime_seconds": self.runtime_seconds,
            "confidence_level": self.confidence_level,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        }


@dataclass
class ComponentMCInput:
    """Component input for Monte Carlo analysis."""
    reference: str
    component_type: str
    base_params: Dict[str, float]
    uncertainties: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    override_lambda: Optional[float] = None  # Fixed industrial value


@dataclass
class SheetMCResult:
    """Monte Carlo results for a single sheet/block."""
    sheet_path: str
    mc_result: MonteCarloResult
    lambda_samples: np.ndarray


def _import_reliability_math():
    try:
        from .reliability_math import calculate_component_lambda, reliability_from_lambda
        return calculate_component_lambda, reliability_from_lambda
    except ImportError:
        from reliability_math import calculate_component_lambda, reliability_from_lambda
        return calculate_component_lambda, reliability_from_lambda


def _import_classify_component():
    try:
        from .component_editor import classify_component
        return classify_component
    except ImportError:
        try:
            from component_editor import classify_component
            return classify_component
        except ImportError:
            def classify_component(ref, value, fields):
                ref = ref.upper()
                if ref.startswith("R"): return "Resistor"
                if ref.startswith("C"): return "Capacitor"
                if ref.startswith("L"): return "Inductor/Transformer"
                if ref.startswith("D"): return "Diode"
                if ref.startswith("Q") or ref.startswith("T"): return "Transistor"
                if ref.startswith("U"): return "Integrated Circuit"
                return "Resistor"
            return classify_component


def monte_carlo_components(
    components: List[ComponentMCInput],
    mission_hours: float,
    n_simulations: int = 5000,
    uncertainty_percent: float = 20.0,
    seed: int = None,
    progress_callback: Callable[[int, int], None] = None,
    confidence_level: float = 0.90,
) -> Tuple[MonteCarloResult, np.ndarray]:
    """
    Vectorized Monte Carlo with component-level uncertainty propagation.
    Components with override_lambda use fixed values (no perturbation).
    """
    calculate_component_lambda, _ = _import_reliability_math()
    rng = np.random.default_rng(seed)
    start_time = time.time()

    n_components = len(components)
    default_cv = uncertainty_percent / 100.0

    # Step 1: compute nominal lambda for each component
    nominal_lambdas = np.zeros(n_components)
    is_fixed = np.zeros(n_components, dtype=bool)

    for i, comp in enumerate(components):
        if comp.override_lambda is not None:
            nominal_lambdas[i] = comp.override_lambda
            is_fixed[i] = True
        else:
            try:
                result = calculate_component_lambda(comp.component_type, comp.base_params)
                nominal_lambdas[i] = result.get("lambda_total", 0)
            except Exception:
                nominal_lambdas[i] = 0.0

    # Step 2: vectorized perturbation (only for non-fixed components)
    cv = default_cv
    lambda_matrix = np.tile(nominal_lambdas, (n_simulations, 1))

    if cv > 0:
        variable = ~is_fixed & (nominal_lambdas > 0)
        if variable.any():
            var_lambdas = nominal_lambdas[variable]
            mu = np.log(var_lambdas) - 0.5 * np.log(1 + cv**2)
            sigma = np.sqrt(np.log(1 + cv**2))
            lambda_matrix[:, variable] = rng.lognormal(
                mu[np.newaxis, :], sigma, size=(n_simulations, variable.sum())
            )

    # Step 3: sum and compute reliability
    system_lambda_samples = lambda_matrix.sum(axis=1)
    system_r_samples = np.exp(-system_lambda_samples * mission_hours)

    runtime = time.time() - start_time

    # Convergence check
    convergence_history = []
    converged = False
    for cp in range(500, n_simulations, 100):
        convergence_history.append((cp, float(np.mean(system_r_samples[:cp]))))
    if len(convergence_history) >= 3:
        last_means = [h[1] for h in convergence_history[-3:]]
        rel_changes = [
            abs(last_means[i] - last_means[i - 1]) / max(abs(last_means[i - 1]), 1e-15)
            for i in range(1, len(last_means))
        ]
        if rel_changes and max(rel_changes) < 0.001:
            converged = True

    if progress_callback:
        progress_callback(n_simulations, n_simulations)

    ci_alpha = (1 - confidence_level) / 2
    ci_lo = float(np.percentile(system_r_samples, ci_alpha * 100))
    ci_hi = float(np.percentile(system_r_samples, (1 - ci_alpha) * 100))

    mc_result = MonteCarloResult(
        mean=float(np.mean(system_r_samples)),
        std=float(np.std(system_r_samples)),
        percentile_5=float(np.percentile(system_r_samples, 5)),
        percentile_50=float(np.percentile(system_r_samples, 50)),
        percentile_95=float(np.percentile(system_r_samples, 95)),
        samples=system_r_samples,
        converged=converged,
        n_simulations=n_simulations,
        convergence_history=convergence_history,
        runtime_seconds=runtime,
        confidence_level=confidence_level,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
    )
    return mc_result, system_lambda_samples


def monte_carlo_sheet(
    sheet_components: List[Dict],
    mission_hours: float,
    n_simulations: int = 2000,
    uncertainty_percent: float = 20.0,
    seed: int = None,
    confidence_level: float = 0.90,
) -> Tuple[MonteCarloResult, np.ndarray]:
    """Run Monte Carlo for a single sheet's components."""
    classify_component = _import_classify_component()
    mc_inputs = []
    for comp in sheet_components:
        comp_type = comp.get("class", "Resistor")
        if not comp_type or comp_type == "Unknown":
            comp_type = classify_component(comp.get("ref", "R1"), comp.get("value", ""), {})
        base_params = comp.get("params", {})
        if not base_params:
            base_params = {
                "t_ambient": 25.0, "t_junction": 85.0, "n_cycles": 5256,
                "delta_t": 3.0, "operating_power": 0.01, "rated_power": 0.125,
            }
        override = comp.get("override_lambda")
        mc_inputs.append(
            ComponentMCInput(
                reference=comp.get("ref", "?"),
                component_type=comp_type,
                base_params=base_params,
                override_lambda=override,
            )
        )
    return monte_carlo_components(
        mc_inputs, mission_hours, n_simulations, uncertainty_percent, seed,
        confidence_level=confidence_level,
    )


def quick_monte_carlo(
    lambda_total: float,
    mission_hours: float,
    uncertainty_percent: float = 20.0,
    n_simulations: int = 5000,
    seed: int = None,
    components: List[Dict] = None,
    confidence_level: float = 0.90,
) -> MonteCarloResult:
    """Monte Carlo uncertainty analysis (component-level if components provided)."""
    _, reliability_from_lambda = _import_reliability_math()
    start_time = time.time()

    if components and len(components) > 0:
        mc_inputs = []
        for comp in components:
            comp_type = comp.get("type", comp.get("class", "Resistor"))
            base_params = comp.get("params", {})
            if "t_junction" not in base_params and "t_ambient" not in base_params:
                base_params["t_ambient"] = 25.0
            base_params.setdefault("n_cycles", 5256)
            base_params.setdefault("delta_t", 3.0)
            override = comp.get("override_lambda")
            mc_inputs.append(
                ComponentMCInput(
                    reference=comp.get("ref", comp.get("reference", "?")),
                    component_type=comp_type,
                    base_params=base_params,
                    override_lambda=override,
                )
            )
        result, _ = monte_carlo_components(
            mc_inputs, mission_hours, n_simulations, uncertainty_percent, seed,
            confidence_level=confidence_level,
        )
        return result
    else:
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

        ci_alpha = (1 - confidence_level) / 2
        ci_lo = float(np.percentile(reliability_samples, ci_alpha * 100))
        ci_hi = float(np.percentile(reliability_samples, (1 - ci_alpha) * 100))

        return MonteCarloResult(
            mean=float(np.mean(reliability_samples)),
            std=float(np.std(reliability_samples)),
            percentile_5=float(np.percentile(reliability_samples, 5)),
            percentile_50=float(np.percentile(reliability_samples, 50)),
            percentile_95=float(np.percentile(reliability_samples, 95)),
            samples=reliability_samples,
            converged=True,
            n_simulations=n_simulations,
            runtime_seconds=runtime,
            confidence_level=confidence_level,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
        )


def monte_carlo_blocks(
    block_data: Dict[str, Dict],
    mission_hours: float,
    n_simulations: int = 2000,
    uncertainty_percent: float = 20.0,
    seed: int = None,
    progress_callback: Callable[[str, int, int], None] = None,
    confidence_level: float = 0.90,
) -> Dict[str, SheetMCResult]:
    """Run Monte Carlo for all blocks/sheets in the system."""
    results = {}
    base_seed = seed if seed is not None else int(time.time())
    for idx, (sheet_path, data) in enumerate(block_data.items()):
        components = data.get("components", [])
        if not components:
            continue
        sheet_components = [
            {"ref": c.get("ref", "?"), "value": c.get("value", ""),
             "class": c.get("class", "Resistor"), "params": c.get("params", {}),
             "override_lambda": c.get("override_lambda")}
            for c in components
        ]
        mc_result, lambda_samples = monte_carlo_sheet(
            sheet_components, mission_hours, n_simulations, uncertainty_percent,
            seed=base_seed + idx, confidence_level=confidence_level)
        results[sheet_path] = SheetMCResult(
            sheet_path=sheet_path, mc_result=mc_result, lambda_samples=lambda_samples)
        if progress_callback:
            progress_callback(sheet_path, idx + 1, len(block_data))
    return results
