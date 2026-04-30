"""
IEC TR 62380:2004 — Stress / acceleration factors
===================================================
Pi-factor calculations (Arrhenius, Coffin-Manson, voltage stress, CTE, EOS),
input validation helpers, and the ``_safe_float`` / ``_safe_int`` utilities.

Re-exported by ``plugins.reliability_math`` for backward compatibility.
"""

from __future__ import annotations

import math

from plugins.models.constants import (
    ABSOLUTE_ZERO_C,
    DEFAULT_T_AMBIENT_C,
    FLOAT_EPSILON,
    INTERFACE_EOS_VALUES,
)

# =============================================================================
# Input validation -- fail-safe, clamp-safe, and informative
# =============================================================================


def validate_ratio(val, name: str = "ratio") -> float:
    """Ensure value is in [0, 1]. Clamps silently for robustness."""
    try:
        v = float(val)
    except (TypeError, ValueError) as e:
        msg = f"{name} must be numeric, got {type(val).__name__}"
        raise TypeError(msg) from e
    return max(0.0, min(1.0, v))


def validate_positive(val, name: str = "value") -> float:  # noqa: ARG001
    """Ensure value is >= 0. Negative values are clamped to 0."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, v)


def validate_temperature(val, name: str = "temperature") -> float:  # noqa: ARG001
    try:
        v = float(val)
    except (TypeError, ValueError):
        return DEFAULT_T_AMBIENT_C
    return max(ABSOLUTE_ZERO_C, v)


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert any value to float with a default fallback."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    """Safely convert any value to int with a default fallback."""
    if val is None:
        return default
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


# =============================================================================
# Core pi-factor calculations
# IEC TR 62380, Section 6
# =============================================================================


def pi_thermal_cycles(n_cycles: float) -> float:
    """Thermal cycling acceleration factor (Coffin-Manson).
    IEC TR 62380, Section 6.2.
    n <= 8760: pi_n = n^0.76;  n > 8760: pi_n = 1.7 * n^0.6
    """
    n_cycles = validate_positive(n_cycles, "n_cycles")
    if n_cycles == 0:
        return 0.0
    if n_cycles <= 8760:
        return n_cycles**0.76
    return 1.7 * (n_cycles**0.6)


def pi_temperature(t: float, ea: float, t_ref: float) -> float:
    """Arrhenius thermal acceleration factor.
    pi_t = exp(Ea_K * (1/T_ref - 1/(273 + T_op)))
    """
    t = validate_temperature(t, "operating_temperature")
    t_op_k = 273.0 + t
    if t_op_k <= 0:
        t_op_k = 0.01
    if t_ref <= 0:
        t_ref = 273.0
    try:
        return math.exp(ea * ((1.0 / t_ref) - (1.0 / t_op_k)))
    except OverflowError:
        return 1e10


def pi_alpha(alpha_s: float, alpha_p: float) -> float:
    """CTE mismatch factor: pi_alpha = 0.06 * |alpha_s - alpha_p|^1.68"""
    diff = abs(_safe_float(alpha_s) - _safe_float(alpha_p))
    if diff < FLOAT_EPSILON:
        return 0.0
    return 0.06 * (diff**1.68)


def pi_voltage_stress(v_applied: float, v_rated: float, exponent: float = 2.5) -> float:
    """Voltage stress acceleration: pi_v = (V_applied / V_rated)^n"""
    v_applied = _safe_float(v_applied, 0.0)
    v_rated = _safe_float(v_rated, 1.0)
    if v_rated <= 0:
        v_rated = 1.0
    ratio = v_applied / v_rated
    if ratio <= 0:
        return 0.0
    return ratio**exponent


def lambda_eos(is_interface: bool, interface_type: str = "Not Interface") -> float:
    """EOS contribution for interface circuits (FIT). IEC TR 62380, Section 7."""
    if not is_interface:
        return 0.0
    p = INTERFACE_EOS_VALUES.get(interface_type, INTERFACE_EOS_VALUES["Not Interface"])
    return _safe_float(p.get("pi_i", 0)) * _safe_float(p.get("l_eos", 0))
