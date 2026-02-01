"""
Monte Carlo Uncertainty Analysis Module - OPTIMIZED v3.0
=========================================================
Fast, threaded Monte Carlo with proper uncertainty propagation and calibration.

Key optimizations:
- Pre-computed parameter distributions (vectorized sampling)
- Batch processing of components
- Threading support for non-blocking UI
- Calibration of per-parameter uncertainties
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional, Any
import time
import threading


# =============================================================================
# DATA CLASSES
# =============================================================================


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
        return {5: self.percentile_5, 50: self.percentile_50, 95: self.percentile_95}

    def confidence_interval(self, level: float = 0.90) -> Tuple[float, float]:
        alpha = (1 - level) / 2
        return (
            np.percentile(self.samples, alpha * 100),
            np.percentile(self.samples, (1 - alpha) * 100),
        )

    def to_dict(self) -> Dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "percentile_5": self.percentile_5,
            "percentile_50": self.percentile_50,
            "percentile_95": self.percentile_95,
            "converged": self.converged,
            "n_simulations": self.n_simulations,
            "runtime_seconds": self.runtime_seconds,
        }


@dataclass
class ParameterUncertainty:
    """Configurable uncertainty for a single parameter."""

    name: str
    enabled: bool = True
    distribution: str = "lognormal"  # "lognormal", "normal", "uniform", "triangular"
    cv: float = 0.20  # Coefficient of variation (std/mean)
    low_mult: float = 0.5  # For uniform/triangular: nominal * low_mult
    high_mult: float = 1.5  # For uniform/triangular: nominal * high_mult

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "distribution": self.distribution,
            "cv": self.cv,
            "low_mult": self.low_mult,
            "high_mult": self.high_mult,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ParameterUncertainty":
        return cls(**d)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo analysis with parameter calibration."""

    n_simulations: int = 5000
    base_uncertainty_cv: float = 0.20  # Default CV for all parameters
    seed: Optional[int] = None
    check_convergence: bool = True
    convergence_threshold: float = 0.001
    min_simulations_for_convergence: int = 1000

    # Per-parameter uncertainty calibration
    parameter_uncertainties: Dict[str, ParameterUncertainty] = field(
        default_factory=dict
    )

    def __post_init__(self):
        if not self.parameter_uncertainties:
            self.parameter_uncertainties = self._default_uncertainties()

    @staticmethod
    def _default_uncertainties() -> Dict[str, ParameterUncertainty]:
        """Default uncertainty configurations per parameter type."""
        return {
            "t_junction": ParameterUncertainty("t_junction", True, "normal", 0.10),
            "t_ambient": ParameterUncertainty("t_ambient", True, "normal", 0.15),
            "n_cycles": ParameterUncertainty("n_cycles", True, "lognormal", 0.25),
            "delta_t": ParameterUncertainty(
                "delta_t", True, "triangular", 0.20, 0.5, 2.0
            ),
            "operating_power": ParameterUncertainty(
                "operating_power", True, "lognormal", 0.15
            ),
            "rated_power": ParameterUncertainty("rated_power", False, "normal", 0.05),
            "voltage_stress_vds": ParameterUncertainty(
                "voltage_stress_vds", True, "uniform", 0.10, 0.8, 1.2
            ),
            "voltage_stress_vgs": ParameterUncertainty(
                "voltage_stress_vgs", True, "uniform", 0.10, 0.8, 1.2
            ),
            "voltage_stress_vce": ParameterUncertainty(
                "voltage_stress_vce", True, "uniform", 0.10, 0.8, 1.2
            ),
            "ripple_ratio": ParameterUncertainty(
                "ripple_ratio", True, "lognormal", 0.30
            ),
            "transistor_count": ParameterUncertainty(
                "transistor_count", True, "lognormal", 0.10
            ),
        }

    def get_uncertainty(self, param_name: str) -> ParameterUncertainty:
        if param_name in self.parameter_uncertainties:
            return self.parameter_uncertainties[param_name]
        return ParameterUncertainty(
            param_name, True, "lognormal", self.base_uncertainty_cv
        )

    def to_dict(self) -> Dict:
        return {
            "n_simulations": self.n_simulations,
            "base_uncertainty_cv": self.base_uncertainty_cv,
            "seed": self.seed,
            "check_convergence": self.check_convergence,
            "convergence_threshold": self.convergence_threshold,
            "min_simulations_for_convergence": self.min_simulations_for_convergence,
            "parameter_uncertainties": {
                k: v.to_dict() for k, v in self.parameter_uncertainties.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "MonteCarloConfig":
        d = d.copy()
        pu = d.pop("parameter_uncertainties", {})
        config = cls(**d)
        config.parameter_uncertainties = {
            k: ParameterUncertainty.from_dict(v) for k, v in pu.items()
        }
        return config


@dataclass
class ComponentMCInput:
    """Component input for Monte Carlo analysis."""

    reference: str
    component_type: str
    base_params: Dict[str, float]
    uncertainties: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class SheetMCResult:
    """Monte Carlo results for a single sheet/block."""

    sheet_path: str
    mc_result: MonteCarloResult
    lambda_samples: np.ndarray


# =============================================================================
# IMPORT HELPERS
# =============================================================================


def _import_reliability_math():
    """Import reliability_math module (handles both package and standalone)."""
    try:
        from .reliability_math import (
            calculate_component_lambda,
            reliability_from_lambda,
        )

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

            def classify_component(ref, value, fields):
                ref = ref.upper()
                if ref.startswith("R"):
                    return "Resistor"
                if ref.startswith("C"):
                    return "Capacitor"
                if ref.startswith("L"):
                    return "Inductor/Transformer"
                if ref.startswith("D"):
                    return "Diode"
                if ref.startswith("Q") or ref.startswith("T"):
                    return "Transistor"
                if ref.startswith("U"):
                    return "Integrated Circuit"
                return "Resistor"

            return classify_component


# =============================================================================
# OPTIMIZED SAMPLING FUNCTIONS
# =============================================================================


def sample_parameter(
    rng: np.random.Generator, nominal: float, uncertainty: ParameterUncertainty, n: int
) -> np.ndarray:
    """Sample parameter values according to specified distribution."""
    if not uncertainty.enabled or nominal == 0:
        return np.full(n, nominal)

    cv = uncertainty.cv
    dist = uncertainty.distribution

    if dist == "lognormal" and nominal > 0:
        mu = np.log(nominal) - 0.5 * np.log(1 + cv**2)
        sigma = np.sqrt(np.log(1 + cv**2))
        return rng.lognormal(mu, sigma, n)

    elif dist == "normal":
        return np.maximum(0, rng.normal(nominal, abs(nominal) * cv, n))

    elif dist == "uniform":
        low = nominal * uncertainty.low_mult
        high = nominal * uncertainty.high_mult
        return rng.uniform(low, high, n)

    elif dist == "triangular":
        low = nominal * uncertainty.low_mult
        high = nominal * uncertainty.high_mult
        return rng.triangular(low, nominal, high, n)

    else:
        if nominal > 0:
            mu = np.log(nominal) - 0.5 * np.log(1 + cv**2)
            sigma = np.sqrt(np.log(1 + cv**2))
            return rng.lognormal(mu, sigma, n)
        return np.full(n, nominal)


def precompute_parameter_samples(
    components: List[ComponentMCInput],
    config: MonteCarloConfig,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Pre-compute all parameter samples for all components (key optimization)."""
    n = config.n_simulations
    uncertain_params = [
        "t_junction",
        "t_ambient",
        "n_cycles",
        "delta_t",
        "operating_power",
        "voltage_stress_vds",
        "voltage_stress_vgs",
        "voltage_stress_vce",
        "ripple_ratio",
        "transistor_count",
        "rated_power",
    ]

    all_samples = {}
    for comp in components:
        comp_samples = {}
        for param_name in uncertain_params:
            if param_name in comp.base_params:
                nominal = comp.base_params[param_name]
                uncertainty = config.get_uncertainty(param_name)
                comp_samples[param_name] = sample_parameter(
                    rng, nominal, uncertainty, n
                )
        all_samples[comp.reference] = comp_samples

    return all_samples


# =============================================================================
# THREADED MONTE CARLO ENGINE
# =============================================================================


class MonteCarloEngine:
    """Threaded Monte Carlo engine with progress callbacks and cancellation."""

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._result: Optional[Tuple[MonteCarloResult, np.ndarray]] = None
        self._error: Optional[Exception] = None
        self._progress: float = 0.0
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def progress(self) -> float:
        with self._lock:
            return self._progress

    @property
    def result(self) -> Optional[Tuple[MonteCarloResult, np.ndarray]]:
        with self._lock:
            return self._result

    @property
    def error(self) -> Optional[Exception]:
        with self._lock:
            return self._error

    def stop(self):
        """Request stop of running simulation."""
        self._stop_flag.set()

    def run_async(
        self,
        components: List[ComponentMCInput],
        mission_hours: float,
        config: MonteCarloConfig,
        on_progress: Callable[[float], None] = None,
        on_complete: Callable[[MonteCarloResult, np.ndarray], None] = None,
        on_error: Callable[[Exception], None] = None,
    ):
        """Run Monte Carlo in background thread."""
        if self.is_running:
            raise RuntimeError("Simulation already running")

        self._stop_flag.clear()
        with self._lock:
            self._result = None
            self._error = None
            self._progress = 0.0

        def worker():
            try:

                def progress_cb(current, total):
                    with self._lock:
                        self._progress = current / total * 100
                    if on_progress:
                        on_progress(self._progress)
                    return not self._stop_flag.is_set()

                result, lambda_samples = monte_carlo_components_optimized(
                    components, mission_hours, config, progress_cb
                )

                if not self._stop_flag.is_set():
                    with self._lock:
                        self._result = (result, lambda_samples)
                    if on_complete:
                        on_complete(result, lambda_samples)

            except Exception as e:
                with self._lock:
                    self._error = e
                if on_error:
                    on_error(e)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def run_sync(
        self,
        components: List[ComponentMCInput],
        mission_hours: float,
        config: MonteCarloConfig,
        progress_callback: Callable[[int, int], None] = None,
    ) -> Tuple[MonteCarloResult, np.ndarray]:
        """Run Monte Carlo synchronously (blocking)."""
        return monte_carlo_components_optimized(
            components, mission_hours, config, progress_callback
        )

    def wait(self, timeout: float = None) -> bool:
        """Wait for completion. Returns True if completed, False if timeout."""
        if self._thread:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True


# =============================================================================
# CORE OPTIMIZED MONTE CARLO
# =============================================================================


def monte_carlo_components_optimized(
    components: List[ComponentMCInput],
    mission_hours: float,
    config: MonteCarloConfig,
    progress_callback: Callable[[int, int], bool] = None,
) -> Tuple[MonteCarloResult, np.ndarray]:
    """
    Optimized Monte Carlo with pre-computed sampling.

    Key optimizations:
    1. Pre-compute ALL random samples before simulation loop (vectorized)
    2. Early convergence detection
    3. Progress callback returns bool for cancellation support
    """
    calculate_component_lambda, reliability_from_lambda = _import_reliability_math()

    rng = np.random.default_rng(config.seed)
    start_time = time.time()
    n = config.n_simulations

    # Pre-compute all parameter samples (MAIN OPTIMIZATION)
    all_samples = precompute_parameter_samples(components, config, rng)

    # Storage for results
    system_lambda_samples = np.zeros(n)
    system_r_samples = np.zeros(n)
    convergence_history = []
    actual_n = n

    # Main simulation loop
    for sim_idx in range(n):
        total_lambda = 0.0

        for comp in components:
            sampled_params = comp.base_params.copy()
            comp_samples = all_samples.get(comp.reference, {})

            for param_name, samples in comp_samples.items():
                sampled_params[param_name] = samples[sim_idx]

            try:
                result = calculate_component_lambda(comp.component_type, sampled_params)
                total_lambda += result.get("lambda_total", 0)
            except Exception:
                result = calculate_component_lambda(
                    comp.component_type, comp.base_params
                )
                total_lambda += result.get("lambda_total", 0)

        system_lambda_samples[sim_idx] = total_lambda
        system_r_samples[sim_idx] = reliability_from_lambda(total_lambda, mission_hours)

        # Convergence check every 100 iterations
        if (
            config.check_convergence
            and sim_idx >= config.min_simulations_for_convergence
        ):
            if (sim_idx + 1) % 100 == 0:
                current_mean = np.mean(system_r_samples[: sim_idx + 1])
                convergence_history.append((sim_idx + 1, current_mean))

                if len(convergence_history) >= 3:
                    last_means = [h[1] for h in convergence_history[-3:]]
                    rel_changes = []
                    for i in range(1, len(last_means)):
                        if last_means[i - 1] != 0:
                            rel_changes.append(
                                abs(last_means[i] - last_means[i - 1])
                                / abs(last_means[i - 1])
                            )
                    if rel_changes and max(rel_changes) < config.convergence_threshold:
                        actual_n = sim_idx + 1
                        system_lambda_samples = system_lambda_samples[:actual_n]
                        system_r_samples = system_r_samples[:actual_n]
                        break

        # Progress callback (every 50 iterations)
        if progress_callback and (sim_idx + 1) % 50 == 0:
            should_continue = progress_callback(sim_idx + 1, n)
            if should_continue is False:
                actual_n = sim_idx + 1
                system_lambda_samples = system_lambda_samples[:actual_n]
                system_r_samples = system_r_samples[:actual_n]
                break

    runtime = time.time() - start_time

    converged = False
    if len(convergence_history) >= 3:
        last_means = [h[1] for h in convergence_history[-3:]]
        rel_changes = [
            abs(last_means[i] - last_means[i - 1]) / abs(last_means[i - 1])
            for i in range(1, len(last_means))
            if last_means[i - 1] != 0
        ]
        if rel_changes and max(rel_changes) < config.convergence_threshold:
            converged = True

    mc_result = MonteCarloResult(
        mean=np.mean(system_r_samples),
        std=np.std(system_r_samples),
        percentile_5=np.percentile(system_r_samples, 5),
        percentile_50=np.percentile(system_r_samples, 50),
        percentile_95=np.percentile(system_r_samples, 95),
        samples=system_r_samples,
        converged=converged,
        n_simulations=len(system_r_samples),
        convergence_history=convergence_history,
        runtime_seconds=runtime,
    )

    return mc_result, system_lambda_samples


# =============================================================================
# BACKWARD COMPATIBLE CONVENIENCE FUNCTIONS
# =============================================================================


def monte_carlo_components(
    components: List[ComponentMCInput],
    mission_hours: float,
    n_simulations: int = 5000,
    uncertainty_percent: float = 20.0,
    seed: int = None,
    progress_callback: Callable[[int, int], None] = None,
) -> Tuple[MonteCarloResult, np.ndarray]:
    """Backward-compatible interface."""
    config = MonteCarloConfig(
        n_simulations=n_simulations,
        base_uncertainty_cv=uncertainty_percent / 100.0,
        seed=seed,
    )
    return monte_carlo_components_optimized(
        components, mission_hours, config, progress_callback
    )


def monte_carlo_sheet(
    sheet_components: List[Dict],
    mission_hours: float,
    n_simulations: int = 2000,
    uncertainty_percent: float = 20.0,
    seed: int = None,
) -> Tuple[MonteCarloResult, np.ndarray]:
    """Run Monte Carlo for a single sheet's components."""
    classify_component = _import_classify_component()

    mc_inputs = []
    for comp in sheet_components:
        comp_type = comp.get("class", "Resistor")
        if not comp_type or comp_type == "Unknown":
            comp_type = classify_component(
                comp.get("ref", "R1"), comp.get("value", ""), {}
            )

        base_params = comp.get("params", {})
        if not base_params:
            base_params = {
                "t_ambient": 25.0,
                "t_junction": 85.0,
                "n_cycles": 5256,
                "delta_t": 3.0,
                "operating_power": 0.01,
                "rated_power": 0.125,
            }

        mc_inputs.append(
            ComponentMCInput(
                reference=comp.get("ref", "?"),
                component_type=comp_type,
                base_params=base_params,
            )
        )

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
    config: MonteCarloConfig = None,
) -> MonteCarloResult:
    """Quick Monte Carlo - uses proper component analysis if components provided."""
    _, reliability_from_lambda = _import_reliability_math()

    start_time = time.time()

    if components and len(components) > 0:
        mc_inputs = []
        for comp in components:
            comp_type = comp.get("type", comp.get("class", "Resistor"))
            base_params = comp.get("params", {})

            if "t_junction" not in base_params and "t_ambient" not in base_params:
                base_params["t_ambient"] = 25.0
            if "n_cycles" not in base_params:
                base_params["n_cycles"] = 5256
            if "delta_t" not in base_params:
                base_params["delta_t"] = 3.0

            mc_inputs.append(
                ComponentMCInput(
                    reference=comp.get("ref", comp.get("reference", "?")),
                    component_type=comp_type,
                    base_params=base_params,
                )
            )

        if config:
            result, _ = monte_carlo_components_optimized(
                mc_inputs, mission_hours, config
            )
        else:
            result, _ = monte_carlo_components(
                mc_inputs, mission_hours, n_simulations, uncertainty_percent, seed
            )
        return result

    else:
        # Simplified sampling (fast but not proper uncertainty propagation)
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
    """Run Monte Carlo for all blocks/sheets."""
    results = {}
    base_seed = seed if seed is not None else int(time.time())

    for idx, (sheet_path, data) in enumerate(block_data.items()):
        components = data.get("components", [])
        if not components:
            continue

        sheet_components = [
            {
                "ref": c.get("ref", "?"),
                "value": c.get("value", ""),
                "class": c.get("class", "Resistor"),
                "params": c.get("params", {}),
            }
            for c in components
        ]

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

        if progress_callback:
            progress_callback(sheet_path, idx + 1, len(block_data))

    return results


# =============================================================================
# MATH VERIFICATION
# =============================================================================


def verify_against_reference(test_cases: List[Dict] = None) -> Dict[str, Any]:
    """Verify Monte Carlo implementation against reference calculations."""
    calculate_component_lambda, reliability_from_lambda = _import_reliability_math()

    if test_cases is None:
        test_cases = [
            {
                "name": "SMD Resistor",
                "type": "Resistor",
                "params": {
                    "t_ambient": 25,
                    "operating_power": 0.01,
                    "rated_power": 0.125,
                    "n_cycles": 5256,
                    "delta_t": 3.0,
                    "resistor_type": "SMD Chip Resistor",
                },
            },
            {
                "name": "Ceramic Capacitor",
                "type": "Capacitor",
                "params": {
                    "t_ambient": 25,
                    "n_cycles": 5256,
                    "delta_t": 3.0,
                    "capacitor_type": "Ceramic Class II (X7R/X5R)",
                },
            },
            {
                "name": "Digital IC",
                "type": "Integrated Circuit",
                "params": {
                    "t_junction": 85,
                    "n_cycles": 5256,
                    "delta_t": 3.0,
                    "transistor_count": 10000,
                    "ic_type": "Microcontroller/DSP",
                    "package": "QFP-48 (7x7mm)",
                },
            },
        ]

    results = {"test_cases": [], "summary": {}}
    mission_hours = 43800

    for tc in test_cases:
        nominal_result = calculate_component_lambda(tc["type"], tc["params"])
        nominal_lambda = nominal_result["lambda_total"]
        nominal_r = reliability_from_lambda(nominal_lambda, mission_hours)

        mc_input = ComponentMCInput(tc["name"], tc["type"], tc["params"])
        mc_result, _ = monte_carlo_components([mc_input], mission_hours, 2000, 20.0, 42)

        mean_error = (
            abs(mc_result.mean - nominal_r) / nominal_r * 100 if nominal_r > 0 else 0
        )

        results["test_cases"].append(
            {
                "name": tc["name"],
                "nominal_lambda": nominal_lambda,
                "nominal_r": nominal_r,
                "mc_mean_r": mc_result.mean,
                "mc_std_r": mc_result.std,
                "mean_error_pct": mean_error,
                "within_1sigma": abs(mc_result.mean - nominal_r) < mc_result.std,
            }
        )

    all_within_sigma = all(tc["within_1sigma"] for tc in results["test_cases"])
    max_error = max(tc["mean_error_pct"] for tc in results["test_cases"])

    results["summary"] = {
        "all_within_1sigma": all_within_sigma,
        "max_error_pct": max_error,
        "pass": all_within_sigma and max_error < 5.0,
    }

    return results


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Monte Carlo Optimization Test")
    print("=" * 60)

    from reliability_math import calculate_component_lambda, reliability_from_lambda

    test_components = [
        ComponentMCInput(
            "R1",
            "Resistor",
            {
                "t_ambient": 25,
                "operating_power": 0.01,
                "rated_power": 0.125,
                "n_cycles": 5256,
                "delta_t": 3.0,
            },
        ),
        ComponentMCInput(
            "R2",
            "Resistor",
            {
                "t_ambient": 25,
                "operating_power": 0.01,
                "rated_power": 0.125,
                "n_cycles": 5256,
                "delta_t": 3.0,
            },
        ),
        ComponentMCInput(
            "C1",
            "Capacitor",
            {
                "t_ambient": 25,
                "n_cycles": 5256,
                "delta_t": 3.0,
                "capacitor_type": "Ceramic Class II (X7R/X5R)",
            },
        ),
        ComponentMCInput(
            "U1",
            "Integrated Circuit",
            {
                "t_junction": 85,
                "n_cycles": 5256,
                "delta_t": 3.0,
                "transistor_count": 10000,
            },
        ),
    ]

    mission_hours = 43800

    total_lambda = 0
    for comp in test_components:
        res = calculate_component_lambda(comp.component_type, comp.base_params)
        total_lambda += res["lambda_total"]
    nominal_r = reliability_from_lambda(total_lambda, mission_hours)

    print(f"\nNominal total λ: {total_lambda*1e9:.2f} FIT")
    print(f"Nominal R (5 years): {nominal_r:.6f}")

    print("\n--- Optimized Monte Carlo (5000 sims) ---")
    config = MonteCarloConfig(n_simulations=5000, seed=42)

    start = time.time()
    result, _ = monte_carlo_components_optimized(test_components, mission_hours, config)
    elapsed = time.time() - start

    print(f"Time: {elapsed:.2f}s")
    print(f"Mean R: {result.mean:.6f}")
    print(f"Std:    {result.std:.6f}")
    print(f"5%:     {result.percentile_5:.6f}")
    print(f"95%:    {result.percentile_95:.6f}")
    print(f"Converged: {result.converged}")
    print(f"Actual sims: {result.n_simulations}")

    print("\n--- Math Verification ---")
    verification = verify_against_reference()
    print(f"All within 1σ: {verification['summary']['all_within_1sigma']}")
    print(f"Max error: {verification['summary']['max_error_pct']:.2f}%")
    print(f"PASS: {verification['summary']['pass']}")
