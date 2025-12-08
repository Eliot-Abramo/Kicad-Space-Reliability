"""Centralized reliability math models (placeholder ECSS-style).

This module centralizes all reliability-related formulas so they can be
tuned in one place. Values are illustrative placeholders and should be
replaced by data from ECSS / your handbook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Dict, Any


# ---------------------------------------------------------------------------
# ECSS-like tables (PLACEHOLDER VALUES!)
# ---------------------------------------------------------------------------

ECSS_LAMBDA_BASE: Dict[str, Dict[str, float]] = {
    "resistor": {
        "thin_film": 1.0e-9,
        "thick_film": 3.0e-9,
        "wirewound": 5.0e-9,
        "default": 3.0e-9,
    },
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
    "ic_digital": {"default": 5.0e-9},
    "ic_analog": {"default": 6.0e-9},
    "fpga": {"default": 4.0e-9},
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
    "connector": {"default": 1.0e-9},
    "relay": {"default": 1.0e-8},
    "crystal": {"default": 2.0e-9},
    "oscillator": {"default": 3.0e-9},
    "battery": {"default": 5.0e-9},
    "mechanical": {"default": 5.0e-9},
}

ECSS_PI_Q: Dict[str, float] = {
    "A": 0.5,
    "B": 1.0,
    "C": 2.0,
    "D": 4.0,
    "default": 1.0,
}

ECSS_PI_E: Dict[str, float] = {
    "GB": 1.0,
    "GF": 2.0,
    "GM": 5.0,
    "LA": 8.0,
    "OR": 10.0,
    "default": 1.0,
}

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


@dataclass
class ComponentParams:
    category: str
    subtype: str = "default"
    quality: str = "B"
    environment: str = "GB"
    stress_ratio: float = 0.5
    temperature: float = 25.0
    mission_time_hours: float = 1.0
    quantity: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)


def _get_lambda_base(p: ComponentParams) -> float:
    cat = ECSS_LAMBDA_BASE.get(p.category, {})
    return cat.get(p.subtype, cat.get("default", 1.0e-9))


def _get_pi_q(p: ComponentParams) -> float:
    return ECSS_PI_Q.get(p.quality, ECSS_PI_Q["default"])


def _get_pi_e(p: ComponentParams) -> float:
    return ECSS_PI_E.get(p.environment, ECSS_PI_E["default"])


def _get_pi_t(p: ComponentParams) -> float:
    t = p.temperature + 273.15
    t_ref = 25.0 + 273.15
    ea = ECSS_EA_PER_CATEGORY.get(p.category, 4000.0)
    return exp(ea * (1.0 / t_ref - 1.0 / t))


def _get_pi_s(p: ComponentParams) -> float:
    s = max(0.0, min(2.0, p.stress_ratio))
    if s <= 0.3:
        return 1.0
    return (s / 0.3) ** 2


def component_failure_rate(p: ComponentParams) -> float:
    lam_b = _get_lambda_base(p)
    pi_q = _get_pi_q(p)
    pi_e = _get_pi_e(p)
    pi_t = _get_pi_t(p)
    pi_s = _get_pi_s(p)
    lam = lam_b * pi_q * pi_e * pi_t * pi_s
    return lam * max(1, p.quantity)


def component_reliability(p: ComponentParams) -> float:
    lam = component_failure_rate(p)
    return exp(-lam * p.mission_time_hours)


def series_reliability(lambdas: Dict[str, float], mission_time_hours: float) -> float:
    total_lam = sum(lambdas.values())
    return exp(-total_lam * mission_time_hours)


def parallel_reliability(lambdas: Dict[str, float], mission_time_hours: float) -> float:
    if not lambdas:
        return 1.0
    lam = next(iter(lambdas.values()))
    n = len(lambdas)
    r_single = exp(-lam * mission_time_hours)
    return 1.0 - (1.0 - r_single) ** n


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
