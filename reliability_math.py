"""
Centralized reliability math models.

All component-level failure rates and ECSS-like factor tables live here.

NOTE:
    - Numerical values are *placeholders* and must be updated according
      to your ECSS-Q-ST-30-11C data or the Reliability Data Handbook you use.
    - The goal is to centralize *all* reliability equations and factors in
      this single file so the rest of the code only calls these functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Dict, Any


# ---------------------------------------------------------------------------
# ECSS-style factor tables (placeholders)
# ---------------------------------------------------------------------------

#: Base failure rates lambda_b in failures/hour for component categories.
#: Keys and values should be synchronized with your ECSS tables.
ECSS_LAMBDA_BASE: Dict[str, Dict[str, float]] = {
    # Example resistor types
    "resistor": {
        "thin_film": 1.0e-9,
        "thick_film": 3.0e-9,
        "wirewound": 5.0e-9,
        "default": 3.0e-9,
    },
    # Example capacitor types
    "capacitor_ceramic": {
        "class_1": 1.5e-9,
        "class_2": 4.0e-9,
        "default": 2.5e-9,
    },
    "capacitor_tantalum": {
        "solid": 8.0e-9,
        "wet": 6.0e-9,
        "default": 7.0e-9,
    },
    # Integrated circuits
    "ic_digital": {
        "default": 5.0e-9,
    },
    "ic_analog": {
        "default": 6.0e-9,
    },
    "fpga": {
        "default": 4.0e-9,
    },
    # Discrete semiconductors
    "diode": {
        "small_signal": 2.0e-9,
        "power": 4.0e-9,
        "zener": 3.5e-9,
        "default": 3.0e-9,
    },
    "bjt": {
        "small_signal": 3.0e-9,
        "power": 6.0e-9,
        "default": 4.0e-9,
    },
    "mosfet": {
        "small_signal": 3.5e-9,
        "power": 7.0e-9,
        "default": 5.0e-9,
    },
    # Other categories
    "connector": {
        "default": 1.0e-9,
    },
    "relay": {
        "default": 1.0e-8,
    },
    "crystal": {
        "default": 2.0e-9,
    },
    "oscillator": {
        "default": 3.0e-9,
    },
    "battery": {
        "default": 5.0e-9,
    },
    "mechanical": {
        "default": 5.0e-9,
    },
}


#: Quality factor \pi_Q according to quality level (placeholder values).
ECSS_PI_Q: Dict[str, float] = {
    "A": 0.5,
    "B": 1.0,
    "C": 2.0,
    "D": 4.0,
    "default": 1.0,
}


#: Environment factor \pi_E (placeholder values).
ECSS_PI_E: Dict[str, float] = {
    "GB": 1.0,   # Ground benign
    "GF": 2.0,   # Ground fixed
    "GM": 5.0,   # Ground mobile
    "LA": 8.0,   # Launch
    "OR": 10.0,  # Orbit
    "default": 1.0,
}


#: Temperature acceleration factor parameters (Arrhenius-like)
ECSS_EA_PER_CATEGORY: Dict[str, float] = {
    "resistor": 3500.0,
    "capacitor_ceramic": 4000.0,
    "capacitor_tantalum": 4500.0,
    "diode": 4000.0,
    "bjt": 4500.0,
    "mosfet": 4500.0,
    "ic_digital": 5000.0,
    "ic_analog": 5000.0,
    "fpga": 5000.0,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ComponentParams:
    """Container for all reliability-relevant parameters for one component.

    This is the *only* structure the rest of the plugin should pass to the
    math functions.
    """

    category: str
    subtype: str = "default"
    quality: str = "B"
    environment: str = "GB"
    stress_ratio: float = 0.5           # e.g. applied / rated
    temperature: float = 25.0           # °C
    mission_time_hours: float = 1.0
    quantity: int = 1                   # number of identical components
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper functions for ECSS factors
# ---------------------------------------------------------------------------

def _get_lambda_base(params: ComponentParams) -> float:
    cat_table = ECSS_LAMBDA_BASE.get(params.category, {})
    lam = cat_table.get(params.subtype, cat_table.get("default", 1.0e-9))
    return lam


def _get_pi_q(params: ComponentParams) -> float:
    return ECSS_PI_Q.get(params.quality, ECSS_PI_Q["default"])


def _get_pi_e(params: ComponentParams) -> float:
    return ECSS_PI_E.get(params.environment, ECSS_PI_E["default"])


def _get_pi_t(params: ComponentParams) -> float:
    """Simple Arrhenius-like temperature acceleration factor.

    This is a very simplified model:

        pi_T = exp( Ea * (1/T_ref - 1/T) )

    with T in Kelvin and Ea an activation energy-like coefficient.
    """
    t_kelvin = params.temperature + 273.15
    t_ref = 25.0 + 273.15  # 25 °C reference
    ea = ECSS_EA_PER_CATEGORY.get(params.category, 4000.0)
    return exp(ea * (1.0 / t_ref - 1.0 / t_kelvin))


def _get_pi_s(params: ComponentParams) -> float:
    """Very simple stress factor vs stress ratio.

    In a real ECSS implementation this would be a piecewise or table-based
    function depending on category and stress type (voltage, current, power).

    Placeholder model:

        pi_S = 1 for s <= 0.3
        pi_S = (s / 0.3) ** 2 for s > 0.3

    where s is the stress ratio.
    """
    s = max(0.0, min(2.0, params.stress_ratio))
    if s <= 0.3:
        return 1.0
    return (s / 0.3) ** 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def component_failure_rate(params: ComponentParams) -> float:
    """Return the component failure rate lambda in failures/hour.

    The model is ECSS-like:

        lambda = lambda_b * pi_Q * pi_E * pi_T * pi_S

    All factors and tables are centralized here.
    """
    lam_b = _get_lambda_base(params)
    pi_q = _get_pi_q(params)
    pi_e = _get_pi_e(params)
    pi_t = _get_pi_t(params)
    pi_s = _get_pi_s(params)

    lam = lam_b * pi_q * pi_e * pi_t * pi_s
    return lam * max(1, params.quantity)


def component_reliability(params: ComponentParams) -> float:
    """Return the reliability R for one component over mission_time_hours."""
    lam = component_failure_rate(params)
    t = params.mission_time_hours
    return exp(-lam * t)


def series_reliability(components: Dict[str, float], mission_time_hours: float) -> float:
    """Compute reliability of components in series from their lambdas.

    Args:
        components: mapping name -> lambda (failures/hour)
        mission_time_hours: mission time in hours

    Returns:
        System reliability in series configuration.
    """
    total_lambda = sum(components.values())
    return exp(-total_lambda * mission_time_hours)


def parallel_reliability(components: Dict[str, float], mission_time_hours: float) -> float:
    """Very simplified parallel redundancy model for *identical* components.

    For two identical components with failure rate lambda each:

        R_sys = 1 - (1 - R)^2

    Here we generalize assuming independence and identical lambdas per entry.
    """
    if not components:
        return 1.0
    # Use the first lambda as representative
    lam = next(iter(components.values()))
    n = len(components)
    r_single = exp(-lam * mission_time_hours)
    r_sys = 1.0 - (1.0 - r_single) ** n
    return r_sys


__all__ = [
    "ComponentParams",
    "component_failure_rate",
    "component_reliability",
    "series_reliability",
    "parallel_reliability",
    "ECSS_LAMBDA_BASE",
    "ECSS_PI_Q",
    "ECSS_PI_E",
]
