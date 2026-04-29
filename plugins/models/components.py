"""
IEC TR 62380:2004 — Component-level failure rate models
=========================================================
All ``lambda_*`` component functions, system reliability algebra, dispatch,
field definitions, formatting, and criticality analysis.

Re-exported by ``plugins.reliability_math`` for backward compatibility.
"""

from __future__ import annotations

import math

from plugins.models.constants import (
    CAPACITOR_PARAMS,
    CONNECTOR_PARAMS,
    DIODE_BASE_RATES,
    DISCRETE_PACKAGE_TABLE,
    FIT_PER_LAMBDA,
    IC_DIE_TABLE,
    IC_PACKAGE_CHOICES,
    IC_PACKAGE_TABLE,
    IC_TYPE_CHOICES,
    INDUCTOR_PARAMS,
    INTERFACE_EOS_VALUES,
    LAMBDA_PER_FIT,
    MISC_COMPONENT_RATES,
    OPTOCOUPLER_BASE_RATES,
    PCB_SOLDER_PARAMS,
    RELAY_PARAMS,
    RESISTOR_PARAMS,
    THERMAL_EXPANSION_SUBSTRATE,
    THYRISTOR_BASE_RATES,
    TRANSISTOR_BASE_RATES,
    ActivationEnergy,
)
from plugins.models.stress import (
    _safe_float,
    _safe_int,
    lambda_eos,
    pi_alpha,
    pi_temperature,
    pi_thermal_cycles,
    pi_voltage_stress,
    validate_ratio,
    validate_temperature,
)

# =============================================================================
# IC package stress contribution (lambda_3) — Table 17a / 17b
# =============================================================================


def calculate_ic_lambda3(pkg_type: str, pins: int | None = None, diag: float | None = None) -> float:
    """IC package stress contribution (lambda_3) in FIT. IEC TR 62380, Table 17a/17b."""
    pkg = IC_PACKAGE_TABLE.get(pkg_type)
    if not pkg:
        return 4.0
    formula = pkg.get("formula", "fixed")
    if formula == "fixed":
        return _safe_float(pkg.get("value", 4.0))
    if formula == "linear" and pins and pins > 0:
        return _safe_float(pkg.get("offset", 0.0)) + _safe_float(pkg.get("coef", 0.09)) * pins
    if formula == "pins" and pins and pins > 0:
        return _safe_float(pkg.get("coef", 0.01)) * (pins ** _safe_float(pkg.get("exp", 1.5)))
    if formula == "diagonal" and diag and diag > 0:
        return _safe_float(pkg.get("coef", 0.05)) * (diag ** _safe_float(pkg.get("exp", 1.68)))
    return 4.0


# =============================================================================
# Component-level failure rate calculations
# =============================================================================


def lambda_integrated_circuit(
    ic_type="MOS_DIGITAL",
    transistor_count=10000,
    construction_year=2020,
    t_junction=85.0,
    package_type="PQFP-10x10",
    pins=48,
    substrate_alpha=16.0,
    package_alpha=21.5,
    is_interface=False,
    interface_type="Not Interface",
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,  # noqa: ARG001
):
    """IC failure rate per IEC TR 62380, Section 7.

    The transistor_count parameter is interpreted according to the n_per
    field in IC_DIE_TABLE: for digital circuits counted "4 per gate", if
    the user enters 10000 gates the effective N = 10000 * 4 = 40000
    transistors.  The n_per multiplier is applied automatically.
    """
    tau_on = validate_ratio(tau_on, "tau_on")
    transistor_count = max(1, _safe_int(transistor_count, 10000))
    construction_year = _safe_int(construction_year, 2020)
    t_junction = validate_temperature(t_junction, "t_junction")
    n_cycles = max(0, _safe_int(n_cycles, 5256))
    delta_t = max(0.0, _safe_float(delta_t, 3.0))

    dp = IC_DIE_TABLE.get(ic_type, IC_DIE_TABLE["MOS_DIGITAL"])
    l1, l2, ea = dp["l1"], dp["l2"], dp["ea"]
    t_ref = dp.get("t_ref", 328)  # 328 K for Si, 373 K for GaAs
    n_per = _safe_float(dp.get("n_per", 1), 1.0)

    # Effective transistor count (user enters in the n_unit described by the type)
    effective_n = transistor_count * n_per

    a = max(0, construction_year - 1998)
    pi_t = pi_temperature(t_junction, float(ea), float(t_ref))
    lambda_die = (float(l1) * effective_n * math.exp(-0.35 * a) + float(l2)) * pi_t * tau_on

    l3 = calculate_ic_lambda3(package_type, pins)
    pi_a = pi_alpha(substrate_alpha, package_alpha)
    pi_n = pi_thermal_cycles(n_cycles)
    lambda_pkg = 2.75e-3 * pi_a * pi_n * (delta_t**0.68) * l3

    lambda_e = lambda_eos(is_interface, interface_type)
    total_fit = lambda_die + lambda_pkg + lambda_e
    return {
        "lambda_die": lambda_die * LAMBDA_PER_FIT,
        "lambda_package": lambda_pkg * LAMBDA_PER_FIT,
        "lambda_eos": lambda_e * LAMBDA_PER_FIT,
        "lambda_total": total_fit * LAMBDA_PER_FIT,
        "fit_total": total_fit,
        "pi_t": pi_t,
        "pi_n": pi_n,
        "pi_alpha": pi_a,
        "lambda_3": l3,
    }


def lambda_diode(
    diode_type="Signal (<1A)",
    t_junction=85.0,
    package="SOD-123",
    is_interface=False,
    interface_type="Not Interface",
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,  # noqa: ARG001
):
    """Diode failure rate per IEC TR 62380, Sections 8.2 (low power) and 8.3 (power).

    Mathematical model (identical for both sections):

        lambda = lambda_die + lambda_package + lambda_overstress

    where:
        lambda_die = pi_U * lambda_0 * SUM_i(pi_t_i * tau_i) / (tau_on + tau_off)
        lambda_package = 2.75e-3 * SUM_i(pi_n_i * (delta_T_i)^0.68) * lambda_B
        lambda_overstress = pi_I * lambda_EOS

    Simplified for single-phase mission profile:
        lambda_die = pi_U * lambda_0 * pi_t * tau_on
                     (since tau_on + tau_off = 1 for normalised ratios)
        lambda_package = 2.75e-3 * pi_n * delta_T^0.68 * lambda_B

    pi_t = exp(4640 * (1/313 - 1/(t_j + 273)))    [Ea = 0.4 eV for ALL diodes]
    pi_U = 1 for all diodes (pi_U = 1 or 10 applies to thyristors/triacs only)

    NOTE: The IEC TR 62380 diode model has NO voltage stress factor (pi_v).
          Voltage stress (pi_s) applies to transistors (Section 8.4/8.5), not diodes.
    """
    tau_on = validate_ratio(tau_on, "tau_on")
    t_junction = validate_temperature(t_junction, "t_junction")
    n_cycles = max(0, _safe_int(n_cycles, 5256))
    delta_t = max(0.0, _safe_float(delta_t, 3.0))

    dr = DIODE_BASE_RATES.get(diode_type, DIODE_BASE_RATES["Signal (<1A)"])
    l0 = float(dr["l0"])
    ea = dr.get("ea", 4640)
    t_ref = dr.get("t_ref", 313)

    pi_u = 1.0
    pi_t = pi_temperature(t_junction, float(ea), float(t_ref))
    lambda_die = pi_u * l0 * pi_t * tau_on

    # -- Package contribution --
    lb = DISCRETE_PACKAGE_TABLE.get(package, {"lb": 1.0}).get("lb", 1.0)
    pi_n = pi_thermal_cycles(n_cycles)
    lambda_pkg = 2.75e-3 * pi_n * (delta_t**0.68) * lb

    # -- Overstress contribution --
    lambda_e = lambda_eos(is_interface, interface_type)

    total_fit = lambda_die + lambda_pkg + lambda_e
    return {
        "lambda_die": lambda_die * LAMBDA_PER_FIT,
        "lambda_package": lambda_pkg * LAMBDA_PER_FIT,
        "lambda_eos": lambda_e * LAMBDA_PER_FIT,
        "lambda_total": total_fit * LAMBDA_PER_FIT,
        "fit_total": total_fit,
        "fit_die": lambda_die,
        "fit_package": lambda_pkg,
        "fit_eos": lambda_e,
        "pi_t": pi_t,
        "pi_u": pi_u,
        "pi_n": pi_n,
        "lambda_0": l0,
        "lambda_b": lb,
        "section": dr.get("section", "8.2"),
    }


def lambda_transistor(
    transistor_type="Silicon MOSFET (<=5W)",
    t_junction=85.0,
    package="SOT-23",
    voltage_stress_vds=0.5,
    voltage_stress_vgs=0.5,
    voltage_stress_vce=0.5,
    is_interface=False,
    interface_type="Not Interface",
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,  # noqa: ARG001
):
    """Transistor failure rate per IEC TR 62380, Section 10."""
    tau_on = validate_ratio(tau_on, "tau_on")
    t_junction = validate_temperature(t_junction, "t_junction")
    n_cycles = max(0, _safe_int(n_cycles, 5256))
    delta_t = max(0.0, _safe_float(delta_t, 3.0))

    p = TRANSISTOR_BASE_RATES.get(transistor_type, TRANSISTOR_BASE_RATES["Silicon MOSFET (<=5W)"])
    l0 = float(p["l0"])
    tech = str(p["tech"])

    ea_map: dict[str, int] = {
        "bipolar": int(ActivationEnergy.BIPOLAR),
        "mos": int(ActivationEnergy.MOS),
        "gan": int(ActivationEnergy.GAN),
        "sic": int(ActivationEnergy.SIC),
    }
    ea = ea_map.get(tech, int(ActivationEnergy.MOS))
    pi_t = pi_temperature(t_junction, float(ea), 373)

    vds = validate_ratio(voltage_stress_vds, "vds")
    vgs = validate_ratio(voltage_stress_vgs, "vgs")
    vce = validate_ratio(voltage_stress_vce, "vce")

    if tech == "bipolar":
        pi_s = 0.22 * math.exp(1.7 * vce)
    elif tech in ("mos", "gan", "sic"):
        pi_s = 0.22 * math.exp(1.7 * vds) * math.exp(3.0 * vgs)
    else:
        pi_s = 0.22 * math.exp(1.7 * vce)

    lambda_die = pi_s * l0 * pi_t * tau_on
    lb = DISCRETE_PACKAGE_TABLE.get(package, {"lb": 1.0}).get("lb", 1.0)
    pi_n = pi_thermal_cycles(n_cycles)
    lambda_pkg = 2.75e-3 * pi_n * (delta_t**0.68) * lb
    lambda_e = lambda_eos(is_interface, interface_type)
    total_fit = lambda_die + lambda_pkg + lambda_e
    return {
        "lambda_die": lambda_die * LAMBDA_PER_FIT,
        "lambda_package": lambda_pkg * LAMBDA_PER_FIT,
        "lambda_eos": lambda_e * LAMBDA_PER_FIT,
        "lambda_total": total_fit * LAMBDA_PER_FIT,
        "fit_total": total_fit,
        "pi_s": pi_s,
        "pi_t": pi_t,
        "pi_n": pi_n,
        "lambda_b": lb,
    }


def lambda_optocoupler(
    optocoupler_type="Phototransistor Output",
    t_junction=85.0,
    package="DIP-8",  # noqa: ARG001
    if_applied=10.0,
    if_rated=60.0,
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,  # noqa: ARG001
):
    """Optocoupler failure rate per IEC TR 62380, Section 10.3."""
    tau_on = validate_ratio(tau_on, "tau_on")
    t_junction = validate_temperature(t_junction, "t_junction")

    p = OPTOCOUPLER_BASE_RATES.get(optocoupler_type, OPTOCOUPLER_BASE_RATES["Phototransistor Output"])
    l0, ea = p["l0"], p.get("ea", ActivationEnergy.OPTOCOUPLER)
    pi_t = pi_temperature(t_junction, ea, 313)

    if_applied = _safe_float(if_applied, 10.0)
    if_rated = _safe_float(if_rated, 60.0)
    pi_if = 1.0
    if if_rated > 0:
        pi_if = (if_applied / if_rated) ** 2.0

    lambda_die = l0 * pi_t * pi_if * tau_on
    pkg_info = IC_PACKAGE_TABLE.get("DIP", {"formula": "pins", "coef": 0.014, "exp": 1.20})
    l3 = _safe_float(pkg_info.get("coef", 0.014)) * (8 ** _safe_float(pkg_info.get("exp", 1.2)))
    pi_n = pi_thermal_cycles(n_cycles)
    lambda_pkg = 2.75e-3 * pi_n * (delta_t**0.68) * l3
    total_fit = lambda_die + lambda_pkg
    return {
        "lambda_die": lambda_die * LAMBDA_PER_FIT,
        "lambda_package": lambda_pkg * LAMBDA_PER_FIT,
        "lambda_total": total_fit * LAMBDA_PER_FIT,
        "fit_total": total_fit,
        "pi_t": pi_t,
        "pi_if": pi_if,
    }


def lambda_thyristor(
    thyristor_type="SCR (<=5A)",
    t_junction=85.0,
    package="TO-220",
    v_applied=0.0,
    v_rated=0.0,
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,  # noqa: ARG001
):
    """Thyristor/TRIAC failure rate per IEC TR 62380, Section 10.2."""
    tau_on = validate_ratio(tau_on, "tau_on")
    t_junction = validate_temperature(t_junction, "t_junction")

    p = THYRISTOR_BASE_RATES.get(thyristor_type, THYRISTOR_BASE_RATES["SCR (<=5A)"])
    l0, ea = p["l0"], p.get("ea", ActivationEnergy.BIPOLAR)
    pi_t = pi_temperature(t_junction, ea, 373)

    pi_v = 1.0
    if _safe_float(v_rated) > 0 and _safe_float(v_applied) > 0:
        pi_v = pi_voltage_stress(v_applied, v_rated, 2.5)

    lambda_die = l0 * pi_t * pi_v * tau_on
    lb = DISCRETE_PACKAGE_TABLE.get(package, {"lb": 1.0}).get("lb", 1.0)
    pi_n = pi_thermal_cycles(n_cycles)
    lambda_pkg = 2.75e-3 * pi_n * (delta_t**0.68) * lb
    total_fit = lambda_die + lambda_pkg
    return {
        "lambda_die": lambda_die * LAMBDA_PER_FIT,
        "lambda_package": lambda_pkg * LAMBDA_PER_FIT,
        "lambda_total": total_fit * LAMBDA_PER_FIT,
        "fit_total": total_fit,
        "pi_t": pi_t,
        "pi_v": pi_v,
    }


def lambda_capacitor(
    capacitor_type="Ceramic Class II (X7R/X5R)",
    t_ambient=25.0,
    ripple_ratio=0.0,
    v_applied=0.0,
    v_rated=0.0,
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,  # noqa: ARG001
):
    """Capacitor failure rate per IEC TR 62380, Section 11."""
    tau_on = validate_ratio(tau_on, "tau_on")
    t_ambient = validate_temperature(t_ambient, "t_ambient")
    ripple_ratio = validate_ratio(ripple_ratio, "ripple_ratio")
    n_cycles = max(0, _safe_int(n_cycles, 5256))
    delta_t = max(0.0, _safe_float(delta_t, 3.0))

    p = CAPACITOR_PARAMS.get(capacitor_type, CAPACITOR_PARAMS["Ceramic Class II (X7R/X5R)"])
    l0, pkg_coef, ea, t_ref = p["l0"], p["pkg_coef"], p["ea"], p["t_ref"]
    v_exp = p.get("v_exp", 2.5)

    t_op = t_ambient + 20.0 * ripple_ratio**2 if "Aluminum" in capacitor_type and ripple_ratio > 0 else t_ambient

    pi_t = pi_temperature(t_op, ea, t_ref)
    pi_n = pi_thermal_cycles(n_cycles)

    pi_v = 1.0
    v_a, v_r = _safe_float(v_applied), _safe_float(v_rated)
    if v_r > 0 and v_a > 0:
        pi_v = pi_voltage_stress(v_a, v_r, v_exp)

    lambda_base = l0 * pi_t * pi_v * tau_on
    lambda_pkg = l0 * pkg_coef * pi_n * (delta_t**0.68)
    total_fit = lambda_base + lambda_pkg
    return {
        "lambda_base": lambda_base * LAMBDA_PER_FIT,
        "lambda_package": lambda_pkg * LAMBDA_PER_FIT,
        "lambda_total": total_fit * LAMBDA_PER_FIT,
        "fit_total": total_fit,
        "pi_t": pi_t,
        "pi_n": pi_n,
        "pi_v": pi_v,
    }


def lambda_resistor(
    resistor_type="SMD Chip Resistor",
    t_ambient=25.0,
    operating_power=0.01,
    rated_power=0.125,
    n_resistors=1,
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,  # noqa: ARG001
):
    """Resistor failure rate per IEC TR 62380, Section 12."""
    tau_on = validate_ratio(tau_on, "tau_on")
    t_ambient = validate_temperature(t_ambient, "t_ambient")
    operating_power = _safe_float(operating_power, 0.01)
    rated_power = max(_safe_float(rated_power, 0.125), 1e-6)
    n_resistors = max(1, _safe_int(n_resistors, 1))
    n_cycles = max(0, _safe_int(n_cycles, 5256))
    delta_t = max(0.0, _safe_float(delta_t, 3.0))

    p = RESISTOR_PARAMS.get(resistor_type, RESISTOR_PARAMS["SMD Chip Resistor"])
    l0, pkg_coef, temp_coef = p["l0"], p["pkg_coef"], p["temp_coef"]

    power_ratio = min(operating_power / rated_power, 1.0)
    t_r = t_ambient + temp_coef * power_ratio
    pi_t = pi_temperature(t_r, ActivationEnergy.RESISTOR, 303)
    pi_n = pi_thermal_cycles(n_cycles)

    l0_eff = l0 * n_resistors
    lambda_base = l0_eff * pi_t * tau_on
    lambda_pkg = l0_eff * pkg_coef * pi_n * (delta_t**0.68)
    total_fit = lambda_base + lambda_pkg
    return {
        "lambda_base": lambda_base * LAMBDA_PER_FIT,
        "lambda_package": lambda_pkg * LAMBDA_PER_FIT,
        "lambda_total": total_fit * LAMBDA_PER_FIT,
        "fit_total": total_fit,
        "t_resistor": t_r,
        "pi_t": pi_t,
        "pi_n": pi_n,
        "power_ratio": power_ratio,
    }


def lambda_inductor(
    inductor_type="Power Inductor",
    t_ambient=25.0,
    power_loss=0.1,
    surface_area_mm2=100.0,
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,  # noqa: ARG001
):
    """Inductor/transformer failure rate per IEC TR 62380, Section 13."""
    tau_on = validate_ratio(tau_on, "tau_on")
    t_ambient = validate_temperature(t_ambient, "t_ambient")
    power_loss = _safe_float(power_loss, 0.1)
    surface_area_mm2 = max(_safe_float(surface_area_mm2, 100.0), 1.0)
    n_cycles = max(0, _safe_int(n_cycles, 5256))
    delta_t = max(0.0, _safe_float(delta_t, 3.0))

    l0 = INDUCTOR_PARAMS.get(inductor_type, INDUCTOR_PARAMS["Power Inductor"])["l0"]
    sur_dm2 = surface_area_mm2 / 10000.0
    t_c = t_ambient + 8.2 * (power_loss / max(sur_dm2, 0.01))

    pi_t = pi_temperature(t_c, ActivationEnergy.RESISTOR, 303)
    pi_n = pi_thermal_cycles(n_cycles)

    lambda_base = l0 * pi_t * tau_on
    lambda_pkg = l0 * 7e-3 * pi_n * (delta_t**0.68)
    total_fit = lambda_base + lambda_pkg
    return {
        "lambda_base": lambda_base * LAMBDA_PER_FIT,
        "lambda_package": lambda_pkg * LAMBDA_PER_FIT,
        "lambda_total": total_fit * LAMBDA_PER_FIT,
        "fit_total": total_fit,
        "t_component": t_c,
        "pi_t": pi_t,
        "pi_n": pi_n,
    }


def lambda_relay(
    relay_type="Signal Relay (Electromech)",
    t_ambient=25.0,
    cycles_per_hour=0.0,
    contact_current_ratio=0.5,
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,  # noqa: ARG001
):
    """Relay failure rate per IEC TR 62380, Section 14."""
    tau_on = validate_ratio(tau_on, "tau_on")
    t_ambient = validate_temperature(t_ambient, "t_ambient")
    cycles_per_hour = _safe_float(cycles_per_hour, 0.0)
    contact_current_ratio = validate_ratio(contact_current_ratio, "contact_current_ratio")

    p = RELAY_PARAMS.get(relay_type, RELAY_PARAMS["Signal Relay (Electromech)"])
    l0, mech_coef = p["l0"], p.get("mech_coef", 0.02)

    pi_t = pi_temperature(t_ambient, ActivationEnergy.RESISTOR, 303)
    pi_contact = 1.0 + 2.0 * (contact_current_ratio**2)

    lambda_elec = l0 * pi_t * tau_on
    lambda_mech = mech_coef * cycles_per_hour * pi_contact
    pi_n = pi_thermal_cycles(n_cycles)
    lambda_pkg = 0.5 * 2.75e-3 * pi_n * (delta_t**0.68)

    total_fit = lambda_elec + lambda_mech + lambda_pkg
    return {
        "lambda_electrical": lambda_elec * LAMBDA_PER_FIT,
        "lambda_mechanical": lambda_mech * LAMBDA_PER_FIT,
        "lambda_package": lambda_pkg * LAMBDA_PER_FIT,
        "lambda_total": total_fit * LAMBDA_PER_FIT,
        "fit_total": total_fit,
        "pi_t": pi_t,
        "pi_contact": pi_contact,
    }


def lambda_connector(
    connector_type="Header/Pin (male)",
    n_contacts=10,
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    mating_cycles_per_year=10.0,
    **kw,  # noqa: ARG001
):
    """Connector failure rate per IEC TR 62380, Section 15."""
    tau_on = validate_ratio(tau_on, "tau_on")
    n_contacts = max(1, _safe_int(n_contacts, 10))
    n_cycles = max(0, _safe_int(n_cycles, 5256))
    delta_t = max(0.0, _safe_float(delta_t, 3.0))
    mating_cycles = _safe_float(mating_cycles_per_year, 10.0)

    p = CONNECTOR_PARAMS.get(connector_type, CONNECTOR_PARAMS["Header/Pin (male)"])
    l0_pin, l_housing = p["l0_pin"], p["l_housing"]
    pkg_coef = p.get("pkg_coef", 2.0e-3)

    lambda_contacts = l0_pin * n_contacts * tau_on
    lambda_housing = l_housing
    pi_n = pi_thermal_cycles(n_cycles)
    lambda_thermal = pkg_coef * n_contacts * pi_n * (delta_t**0.68)
    lambda_mating = 0.01 * n_contacts * mating_cycles

    total_fit = lambda_contacts + lambda_housing + lambda_thermal + lambda_mating
    return {
        "lambda_contacts": lambda_contacts * LAMBDA_PER_FIT,
        "lambda_housing": lambda_housing * LAMBDA_PER_FIT,
        "lambda_thermal": lambda_thermal * LAMBDA_PER_FIT,
        "lambda_mating": lambda_mating * LAMBDA_PER_FIT,
        "lambda_total": total_fit * LAMBDA_PER_FIT,
        "fit_total": total_fit,
        "n_contacts": n_contacts,
    }


def lambda_pcb_solder(joint_type="SMD Solder Joint", n_joints=100, n_cycles=5256, delta_t=3.0, **kw):  # noqa: ARG001
    """PCB/solder joint failure rate per IEC TR 62380, Section 16."""
    n_joints = max(0, _safe_int(n_joints, 100))
    n_cycles = max(0, _safe_int(n_cycles, 5256))
    delta_t = max(0.0, _safe_float(delta_t, 3.0))

    p = PCB_SOLDER_PARAMS.get(joint_type, PCB_SOLDER_PARAMS["SMD Solder Joint"])
    l0 = p["l0"]

    lambda_base = l0 * n_joints
    pi_n = pi_thermal_cycles(n_cycles)
    lambda_thermal = l0 * 0.5e-3 * n_joints * pi_n * (delta_t**0.68)
    total_fit = lambda_base + lambda_thermal
    return {
        "lambda_base": lambda_base * LAMBDA_PER_FIT,
        "lambda_thermal": lambda_thermal * LAMBDA_PER_FIT,
        "lambda_total": total_fit * LAMBDA_PER_FIT,
        "fit_total": total_fit,
    }


def lambda_misc_component(
    component_type="Crystal Oscillator (XO)",
    n_contacts=1,
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,  # noqa: ARG001
):
    """Miscellaneous component failure rate."""
    tau_on = validate_ratio(tau_on, "tau_on")
    n_cycles = max(0, _safe_int(n_cycles, 5256))
    delta_t = max(0.0, _safe_float(delta_t, 3.0))

    base = _safe_float(MISC_COMPONENT_RATES.get(component_type, 10.0))
    if "Connector" in component_type:
        base *= max(1, _safe_int(n_contacts, 1))

    pi_n = pi_thermal_cycles(n_cycles)
    total_fit = base * (tau_on + 3e-3 * pi_n * (delta_t**0.68))
    return {"lambda_total": total_fit * LAMBDA_PER_FIT, "fit_total": total_fit, "base_fit": base}


# =============================================================================
# System-level reliability algebra
# =============================================================================


def reliability_from_lambda(lam: float, hours: float) -> float:
    """R(t) = exp(-lambda * t)"""
    lam = _safe_float(lam, 0.0)
    hours = _safe_float(hours, 0.0)
    if lam <= 0 or hours <= 0:
        return 1.0
    try:
        return math.exp(-lam * hours)
    except OverflowError:
        return 0.0


def lambda_from_reliability(r: float, hours: float) -> float:
    """lambda = -ln(R) / t"""
    r = _safe_float(r, 1.0)
    hours = _safe_float(hours, 1.0)
    if r <= 0:
        return float("inf")
    if r >= 1:
        return 0.0
    if hours <= 0:
        return 0.0
    return -math.log(r) / hours


def mttf_from_lambda(lam: float) -> float:
    """MTTF = 1 / lambda"""
    lam = _safe_float(lam, 0.0)
    return float("inf") if lam <= 0 else 1.0 / lam


def r_series(r_list):
    """Series system: R_sys = product(R_i)"""
    if not r_list:
        return 1.0
    result = 1.0
    for r in r_list:
        result *= _safe_float(r, 1.0)
    return result


def r_parallel(r_list):
    """Parallel system: R_sys = 1 - product(1 - R_i)"""
    if not r_list:
        return 1.0
    p_fail = 1.0
    for r in r_list:
        p_fail *= 1.0 - _safe_float(r, 1.0)
    return 1.0 - p_fail


def r_k_of_n(r_list, k):
    """K-of-N redundancy."""
    n = len(r_list)
    k = _safe_int(k, 1)
    if k > n or k < 1:
        return 0.0
    if k == 1:
        return r_parallel(r_list)
    if k == n:
        return r_series(r_list)
    if len(set(r_list)) == 1:
        r = _safe_float(r_list[0], 1.0)
        return sum(math.comb(n, i) * (r**i) * ((1.0 - r) ** (n - i)) for i in range(k, n + 1))
    r_last = _safe_float(r_list[-1], 1.0)
    return r_last * r_k_of_n(r_list[:-1], k - 1) + (1.0 - r_last) * r_k_of_n(r_list[:-1], k)


def lambda_series(lam_list):
    """Series system total failure rate = sum."""
    return sum(_safe_float(lam_value, 0.0) for lam_value in lam_list)


# =============================================================================
# Field definitions for the GUI
# =============================================================================


def get_component_types():
    """Return list of all supported component type names."""
    return [
        "Integrated Circuit",
        "Diode",
        "Transistor",
        "Optocoupler",
        "Thyristor/TRIAC",
        "Capacitor",
        "Resistor",
        "Inductor/Transformer",
        "Relay",
        "Connector",
        "PCB/Solder",
        "Miscellaneous",
    ]


def get_field_definitions(component_type):
    """Return editable parameter definitions for a given component type."""
    common = {
        "n_cycles": {"type": "int", "default": 5256, "help": "Annual thermal cycles"},
        "delta_t": {"type": "float", "default": 3.0, "help": "Delta-T per cycle (C)"},
        "tau_on": {"type": "float", "default": 1.0, "help": "Working time ratio (0-1)"},
    }
    iface = {
        "is_interface": {
            "type": "bool",
            "default": False,
            "help": "Interface circuit?",
        },
        "interface_type": {
            "type": "choice",
            "choices": list(INTERFACE_EOS_VALUES.keys()),
            "default": "Not Interface",
        },
    }

    if component_type == "Integrated Circuit":
        return {
            "ic_type": {
                "type": "choice",
                "choices": list(IC_TYPE_CHOICES.keys()),
                "default": "MOS Digital (Micro/DSP)",
                "required": True,
                "help": "IC technology and function (Table 16)",
            },
            "transistor_count": {
                "type": "int",
                "default": 10000,
                "required": True,
                "help": "Count in type's native unit (gates, bits, macrocells, transistors - see Table 16)",
            },
            "construction_year": {
                "type": "int",
                "default": 2020,
                "help": "Fabrication year (technology maturity, a = year - 1998)",
            },
            "t_junction": {
                "type": "float",
                "default": 85.0,
                "required": True,
                "help": "Junction temperature (C)",
            },
            "package": {
                "type": "choice",
                "choices": list(IC_PACKAGE_CHOICES.keys()),
                "default": "QFP-48 (7x7mm)",
                "required": True,
                "help": "Package type (Table 17a/17b)",
            },
            "substrate": {
                "type": "choice",
                "choices": list(THERMAL_EXPANSION_SUBSTRATE.keys()),
                "default": "FR4 / Epoxy Glass (G-10)",
                "help": "PCB substrate material (Table 14)",
            },
            **iface,
            **common,
        }
    if component_type == "Diode":
        return {
            "diode_type": {
                "type": "choice",
                "choices": list(DIODE_BASE_RATES.keys()),
                "default": "Signal (<1A)",
                "required": True,
                "help": "Diode type per IEC TR 62380, Sections 8.2/8.3",
            },
            "t_junction": {
                "type": "float",
                "default": 85.0,
                "required": True,
                "help": "Junction temperature (C). For protection diodes: t_j = t_ambient; "
                "For others: t_j = t_ambient + R_th * P",
            },
            "package": {
                "type": "choice",
                "choices": list(DISCRETE_PACKAGE_TABLE.keys()),
                "default": "SOD-123",
                "help": "Package type (Table 18, determines lambda_B)",
            },
            **iface,
            **common,
        }
    if component_type == "Transistor":
        return {
            "transistor_type": {
                "type": "choice",
                "choices": list(TRANSISTOR_BASE_RATES.keys()),
                "default": "Silicon MOSFET (<=5W)",
                "required": True,
            },
            "t_junction": {
                "type": "float",
                "default": 85.0,
                "required": True,
                "help": "Junction temperature (C)",
            },
            "package": {
                "type": "choice",
                "choices": list(DISCRETE_PACKAGE_TABLE.keys()),
                "default": "SOT-23",
            },
            "voltage_stress_vds": {
                "type": "float",
                "default": 0.5,
                "help": "V_DS/V_DS_max (0-1)",
            },
            "voltage_stress_vgs": {
                "type": "float",
                "default": 0.5,
                "help": "V_GS/V_GS_max (0-1)",
            },
            "voltage_stress_vce": {
                "type": "float",
                "default": 0.5,
                "help": "V_CE/V_CE_max (0-1)",
            },
            **iface,
            **common,
        }
    if component_type == "Optocoupler":
        return {
            "optocoupler_type": {
                "type": "choice",
                "choices": list(OPTOCOUPLER_BASE_RATES.keys()),
                "default": "Phototransistor Output",
                "required": True,
            },
            "t_junction": {
                "type": "float",
                "default": 85.0,
                "required": True,
                "help": "Junction temperature (C)",
            },
            "if_applied": {
                "type": "float",
                "default": 10.0,
                "help": "LED forward current (mA)",
            },
            "if_rated": {
                "type": "float",
                "default": 60.0,
                "help": "Rated LED current (mA)",
            },
            **common,
        }
    if component_type == "Thyristor/TRIAC":
        return {
            "thyristor_type": {
                "type": "choice",
                "choices": list(THYRISTOR_BASE_RATES.keys()),
                "default": "SCR (<=5A)",
                "required": True,
            },
            "t_junction": {
                "type": "float",
                "default": 85.0,
                "required": True,
                "help": "Junction temperature (C)",
            },
            "package": {
                "type": "choice",
                "choices": list(DISCRETE_PACKAGE_TABLE.keys()),
                "default": "TO-220",
            },
            "v_applied": {
                "type": "float",
                "default": 0.0,
                "help": "Applied blocking voltage (V)",
            },
            "v_rated": {
                "type": "float",
                "default": 0.0,
                "help": "Rated blocking voltage (V)",
            },
            **common,
        }
    if component_type == "Capacitor":
        return {
            "capacitor_type": {
                "type": "choice",
                "choices": list(CAPACITOR_PARAMS.keys()),
                "default": "Ceramic Class II (X7R/X5R)",
                "required": True,
            },
            "t_ambient": {
                "type": "float",
                "default": 25.0,
                "required": True,
                "help": "Ambient temperature (C)",
            },
            "v_applied": {
                "type": "float",
                "default": 0.0,
                "help": "Applied voltage (V)",
            },
            "v_rated": {"type": "float", "default": 0.0, "help": "Rated voltage (V)"},
            "ripple_ratio": {
                "type": "float",
                "default": 0.0,
                "help": "Ripple I_rms/I_rated (0-1, Al caps)",
            },
            **common,
        }
    if component_type == "Resistor":
        return {
            "resistor_type": {
                "type": "choice",
                "choices": list(RESISTOR_PARAMS.keys()),
                "default": "SMD Chip Resistor",
                "required": True,
            },
            "t_ambient": {
                "type": "float",
                "default": 25.0,
                "required": True,
                "help": "Ambient temperature (C)",
            },
            "operating_power": {
                "type": "float",
                "default": 0.01,
                "required": True,
                "help": "Operating power (W)",
            },
            "rated_power": {
                "type": "float",
                "default": 0.125,
                "required": True,
                "help": "Rated power (W)",
            },
            **common,
        }
    if component_type == "Inductor/Transformer":
        return {
            "inductor_type": {
                "type": "choice",
                "choices": list(INDUCTOR_PARAMS.keys()),
                "default": "Power Inductor",
                "required": True,
            },
            "t_ambient": {
                "type": "float",
                "default": 25.0,
                "help": "Ambient temperature (C)",
            },
            "power_loss": {
                "type": "float",
                "default": 0.1,
                "help": "Power dissipation (W)",
            },
            "surface_area_mm2": {
                "type": "float",
                "default": 100.0,
                "help": "Surface area (mm2)",
            },
            **common,
        }
    if component_type == "Relay":
        return {
            "relay_type": {
                "type": "choice",
                "choices": list(RELAY_PARAMS.keys()),
                "default": "Signal Relay (Electromech)",
                "required": True,
            },
            "t_ambient": {
                "type": "float",
                "default": 25.0,
                "help": "Ambient temperature (C)",
            },
            "cycles_per_hour": {
                "type": "float",
                "default": 0.0,
                "help": "Switching rate (cyc/h)",
            },
            "contact_current_ratio": {
                "type": "float",
                "default": 0.5,
                "help": "I/I_rated (0-1)",
            },
            **common,
        }
    if component_type == "Connector":
        return {
            "connector_type": {
                "type": "choice",
                "choices": list(CONNECTOR_PARAMS.keys()),
                "default": "Header/Pin (male)",
                "required": True,
            },
            "n_contacts": {
                "type": "int",
                "default": 10,
                "required": True,
                "help": "Number of contacts/pins",
            },
            "mating_cycles_per_year": {
                "type": "float",
                "default": 10.0,
                "help": "Mating events per year",
            },
            **common,
        }
    if component_type == "PCB/Solder":
        return {
            "joint_type": {
                "type": "choice",
                "choices": list(PCB_SOLDER_PARAMS.keys()),
                "default": "SMD Solder Joint",
                "required": True,
            },
            "n_joints": {
                "type": "int",
                "default": 100,
                "required": True,
                "help": "Number of joints/vias",
            },
            "n_cycles": common["n_cycles"],
            "delta_t": common["delta_t"],
        }

    return {
        "component_subtype": {
            "type": "choice",
            "choices": list(MISC_COMPONENT_RATES.keys()),
            "default": "Crystal Oscillator (XO)",
        },
        **common,
    }


# =============================================================================
# Dispatch: unified entry point
# =============================================================================


def calculate_component_lambda(component_type, params):
    """Calculate failure rate for a component given type and parameters."""
    try:
        if component_type == "Integrated Circuit":
            ic_key = IC_TYPE_CHOICES.get(params.get("ic_type", "MOS Digital (Micro/DSP)"), "MOS_DIGITAL")
            pkg = params.get("package", "QFP-48 (7x7mm)")
            pkg_info = IC_PACKAGE_CHOICES.get(pkg, ("PQFP-7x7", 48))
            sub = THERMAL_EXPANSION_SUBSTRATE.get(params.get("substrate", "FR4 / Epoxy Glass (G-10)"), 16.0)
            return lambda_integrated_circuit(
                ic_type=ic_key,
                transistor_count=params.get("transistor_count", 10000),
                construction_year=params.get("construction_year", 2020),
                t_junction=params.get("t_junction", 85.0),
                package_type=pkg_info[0],
                pins=pkg_info[1] if len(pkg_info) > 1 else 48,
                substrate_alpha=sub,
                is_interface=params.get("is_interface", False),
                interface_type=params.get("interface_type", "Not Interface"),
                n_cycles=params.get("n_cycles", 5256),
                delta_t=params.get("delta_t", 3.0),
                tau_on=params.get("tau_on", 1.0),
            )
        if component_type == "Diode":
            return lambda_diode(**{k: v for k, v in params.items() if not k.startswith("_")})
        if component_type == "Transistor":
            return lambda_transistor(**{k: v for k, v in params.items() if not k.startswith("_")})
        if component_type == "Optocoupler":
            return lambda_optocoupler(**{k: v for k, v in params.items() if not k.startswith("_")})
        if component_type == "Thyristor/TRIAC":
            return lambda_thyristor(**{k: v for k, v in params.items() if not k.startswith("_")})
        if component_type == "Capacitor":
            return lambda_capacitor(**{k: v for k, v in params.items() if not k.startswith("_")})
        if component_type == "Resistor":
            return lambda_resistor(**{k: v for k, v in params.items() if not k.startswith("_")})
        if component_type == "Inductor/Transformer":
            return lambda_inductor(**{k: v for k, v in params.items() if not k.startswith("_")})
        if component_type == "Relay":
            return lambda_relay(**{k: v for k, v in params.items() if not k.startswith("_")})
        if component_type == "Connector":
            return lambda_connector(**{k: v for k, v in params.items() if not k.startswith("_")})
        if component_type == "PCB/Solder":
            return lambda_pcb_solder(**{k: v for k, v in params.items() if not k.startswith("_")})
        return lambda_misc_component(
            component_type=params.get("component_subtype", "Crystal Oscillator (XO)"),
            **{k: v for k, v in params.items() if not k.startswith("_") and k != "component_subtype"},
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return {"lambda_total": 10e-9, "fit_total": 10.0, "_error": str(e)}


def calculate_lambda_float(component_type, params):
    return calculate_lambda(component_type, params)


def calculate_lambda(component_type, params=None):
    if params is None:
        params = {}
    result = calculate_component_lambda(component_type, params)
    return result.get("lambda_total", 0.0)


def calculate_lambda_fallback(component_type, params=None):
    """Legacy dispatch with substring matching and exception safety."""
    import warnings

    warnings.warn("Use calculate_lambda() instead", DeprecationWarning, stacklevel=2)
    return calculate_lambda(component_type, params or {})


# =============================================================================
# Formatting utilities
# =============================================================================


def fit_to_lambda(fit):
    return _safe_float(fit) * LAMBDA_PER_FIT


def lambda_to_fit(lam):
    return _safe_float(lam) * FIT_PER_LAMBDA


def format_lambda(lam, as_fit=True):
    lam = _safe_float(lam)
    return f"{lam * FIT_PER_LAMBDA:.2f} FIT" if as_fit else f"{lam:.2e} /h"


def format_reliability(r):
    r = _safe_float(r, 1.0)
    if r >= 0.9999:
        return f"{r:.6f}"
    if r >= 0.99:
        return f"{r:.4f}"
    return f"{r:.3f}"


# =============================================================================
# Component criticality analysis -- field-level sensitivity
# =============================================================================


def analyze_component_criticality(component_type, params, mission_hours, perturbation=0.10):  # noqa: ARG001
    """Analyze which parameter fields most influence a component's failure rate."""
    from reliability_math import calculate_component_lambda as _calc_fn

    try:
        nominal = _calc_fn(component_type, params)
        lam_nominal = nominal.get("lambda_total", 0.0)
    except Exception:  # noqa: BLE001
        return []

    if lam_nominal <= 0:
        return []

    results = []
    for field, value in params.items():
        if field.startswith("_"):
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v == 0:
            continue

        dp = abs(v * perturbation)
        if dp < 1e-12:
            continue

        params_low = dict(params)
        params_low[field] = v - dp
        params_high = dict(params)
        params_high[field] = v + dp

        try:
            lam_low = _calc_fn(component_type, params_low).get("lambda_total", 0.0)
            lam_high = _calc_fn(component_type, params_high).get("lambda_total", 0.0)
        except Exception:  # noqa: BLE001
            continue

        sensitivity = (lam_high - lam_low) / (2.0 * dp / v) / lam_nominal if lam_nominal > 0 else 0.0

        results.append(
            {
                "field": field,
                "nominal_value": v,
                "sensitivity": sensitivity,
                "lambda_low_fit": lam_low * FIT_PER_LAMBDA,
                "lambda_high_fit": lam_high * FIT_PER_LAMBDA,
                "lambda_nominal_fit": lam_nominal * FIT_PER_LAMBDA,
                "impact_percent": (abs(lam_high - lam_low) / lam_nominal * 100 if lam_nominal > 0 else 0),
            }
        )

    results.sort(key=lambda x: -abs(x["sensitivity"]))
    return results


# =============================================================================
# Legacy aliases for backward compatibility
# =============================================================================

reliability = reliability_from_lambda


def component_failure_rate(component_type, params=None):
    payload = params.to_dict() if params and hasattr(params, "to_dict") else (params or {})
    return calculate_lambda(component_type, payload)
