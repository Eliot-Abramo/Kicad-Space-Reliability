"""
Reliability Mathematics - IEC TR 62380 Implementation (CORRECTED v2.0)
======================================================================
FIXES: π_n threshold, configurable EOS, τ_on support, input validation
"""

import math
from typing import Dict, List, Any

__version__ = "2.0.0"


# === VALIDATION ===
def validate_ratio(val, name="ratio"):
    if not 0 <= val <= 1:
        raise ValueError(f"{name} must be 0-1: {val}")
    return val


def validate_positive(val, name="value"):
    if val < 0:
        raise ValueError(f"{name} must be positive")
    return val


def validate_temperature(val, name="temperature"):
    """Validate temperature is above absolute zero (-273.15°C)."""
    if val < -273.15:
        raise ValueError(f"{name} must be above absolute zero: {val}")
    return val


# === CONNECTION TYPE ===
class ConnectionType:
    SERIES = "series"
    PARALLEL = "parallel"
    K_OF_N = "k_of_n"

    def __init__(self, value=None):
        self._value = value or self.SERIES

    @property
    def value(self):
        return getattr(self, "_value", self.SERIES)

    def __eq__(self, other):
        return self.value == (
            other.value if isinstance(other, ConnectionType) else other
        )

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return self.value


# === CONSTANTS ===
class ActivationEnergy:
    MOS = 3480
    BIPOLAR = 4640
    CAPACITOR_LOW = 1160
    CAPACITOR_MED = 1740
    CAPACITOR_HIGH = 2900
    ALUMINUM_CAP = 4640
    RESISTOR = 1740


INTERFACE_EOS_VALUES = {
    "Non Interfaces": {"pi_i": 0, "l_eos": 0},
    "Computer": {"pi_i": 1, "l_eos": 10},
    "Switching": {"pi_i": 1, "l_eos": 15},
    "Telecoms transmitting / access / subscriber cards": {"pi_i": 1, "l_eos": 40},
    "Subscriber equipment": {"pi_i": 1, "l_eos": 70},
    "Railways, payphone": {"pi_i": 1, "l_eos": 100},
    "Civilian avionics (on board calculators)": {"pi_i": 1, "l_eos": 20},
    "Voltage supply, converters": {"pi_i": 1, "l_eos": 40},
}

# Backward compatibility for older stored string values.
EOS_INTERFACE_ALIASES = {
    "Not Interface": "Non Interfaces",
    "Telecom (Switching)": "Switching",
    "Telecom (Subscriber)": "Subscriber equipment",
    "Avionics": "Civilian avionics (on board calculators)",
    "Power Supply": "Voltage supply, converters",
}

THERMAL_EXPANSION_SUBSTRATE = {
    "FR4 (Epoxy Glass)": 16.0,
    "Polyimide Flex": 6.5,
    "Alumina (Ceramic)": 6.5,
    "Aluminum (Metal Core)": 23.0,
}

IC_DIE_TABLE = {
    "MOS_DIGITAL": {"l1": 3.4e-6, "l2": 1.7, "ea": ActivationEnergy.MOS},
    "MOS_LCA": {"l1": 1.2e-5, "l2": 10, "ea": ActivationEnergy.MOS},
    "MOS_CPLD": {"l1": 4.0e-5, "l2": 8.8, "ea": ActivationEnergy.MOS},
    "BIPOLAR_LINEAR": {"l1": 2.7e-2, "l2": 20, "ea": ActivationEnergy.BIPOLAR},
    "BICMOS_LOW_V": {"l1": 2.7e-4, "l2": 20, "ea": ActivationEnergy.MOS},
    "BICMOS_HIGH_V": {"l1": 2.7e-3, "l2": 20, "ea": ActivationEnergy.BIPOLAR},
}

IC_TYPE_CHOICES = {
    "Microcontroller/DSP": "MOS_DIGITAL",
    "FPGA (RAM-based)": "MOS_LCA",
    "CPLD/FPGA (Flash)": "MOS_CPLD",
    "Op-Amp/Comparator": "BIPOLAR_LINEAR",
    "LDO Regulator": "BICMOS_LOW_V",
    "DC-DC Controller": "BICMOS_HIGH_V",
}

IC_PACKAGE_CHOICES = {
    "SOIC-8": ("SO", 8),
    "SOIC-16": ("SO", 16),
    "TSSOP-20": ("TSSOP", 20),
    "QFP-48 (7x7mm)": ("TQFP-7x7", 48),
    "QFP-64 (10x10mm)": ("TQFP-10x10", 64),
    "QFN-32 (5x5mm)": ("QFN", 32, 7.07),
    "BGA-256": ("PBGA-17x19", 256),
}

IC_PACKAGE_TABLE = {
    "SO": {"formula": "pins", "coef": 0.012, "exp": 1.65},
    "TSSOP": {"formula": "pins", "coef": 0.011, "exp": 1.4},
    "TQFP-7x7": {"formula": "fixed", "value": 2.5},
    "TQFP-10x10": {"formula": "fixed", "value": 4.1},
    "PBGA-17x19": {"formula": "fixed", "value": 16.6},
    "QFN": {"formula": "diagonal", "coef": 0.048, "exp": 1.68},
}

DISCRETE_PACKAGE_TABLE = {
    "TO-92": {"lb": 1.0},
    "TO-220": {"lb": 5.7},
    "SOT-23": {"lb": 1.0},
    "SOT-223": {"lb": 3.4},
    "SOD-123": {"lb": 1.0},
    # IEC/TR 62380 Table 18 distinguishes DO-41 glass vs plastic
    "DO-41 (glass)": {"lb": 2.5},
    "DO-41 (plastic)": {"lb": 1.0},
    # Backward-compatible alias (previous behavior mapped DO-41 to glass)
    "DO-41": {"lb": 2.5},
    "SMA": {"lb": 1.8},
    "0402": {"lb": 0.5},
    "0603": {"lb": 0.6},
    "0805": {"lb": 0.8},
    "1206": {"lb": 1.0},
}

DIODE_BASE_RATES = {
    "Signal (<1A)": {"l0": 0.07},
    "Rectifier (1-3A)": {"l0": 0.1},
    "Zener": {"l0": 0.4},
    "TVS": {"l0": 2.3},
    "Schottky (<3A)": {"l0": 0.15},
    "LED": {"l0": 0.5},
}

TRANSISTOR_BASE_RATES = {
    "Silicon BJT (≤5W)": {"l0": 0.75, "tech": "bipolar"},
    "Silicon MOSFET (≤5W)": {"l0": 0.75, "tech": "mos"},
    "Silicon BJT (>5W)": {"l0": 2.0, "tech": "bipolar"},
    "Silicon MOSFET (>5W)": {"l0": 2.0, "tech": "mos"},
    "GaN HEMT": {"l0": 3.0, "tech": "gan"},
}

CAPACITOR_PARAMS = {
    "Ceramic Class I (C0G)": {
        "l0": 0.05,
        "pkg_coef": 3.3e-3,
        "ea": ActivationEnergy.CAPACITOR_LOW,
        "t_ref": 303,
    },
    "Ceramic Class II (X7R/X5R)": {
        "l0": 0.15,
        "pkg_coef": 3.3e-3,
        "ea": ActivationEnergy.CAPACITOR_LOW,
        "t_ref": 303,
    },
    "Tantalum Solid": {
        "l0": 0.4,
        "pkg_coef": 3.8e-3,
        "ea": ActivationEnergy.CAPACITOR_MED,
        "t_ref": 303,
    },
    "Aluminum Electrolytic": {
        "l0": 1.3,
        "pkg_coef": 1.4e-3,
        "ea": ActivationEnergy.ALUMINUM_CAP,
        "t_ref": 313,
    },
}

RESISTOR_PARAMS = {
    "SMD Chip Resistor": {"l0": 0.01, "pkg_coef": 3.3e-3, "temp_coef": 55},
    "Film (Low Power)": {"l0": 0.1, "pkg_coef": 1.4e-3, "temp_coef": 85},
    "Thin Film Precision": {"l0": 0.05, "pkg_coef": 3.3e-3, "temp_coef": 50},
}

INDUCTOR_PARAMS = {
    "Power Inductor": {"l0": 0.6},
    "Signal Transformer": {"l0": 1.5},
    "Power Transformer": {"l0": 3.0},
}

MISC_COMPONENT_RATES = {
    "Crystal Oscillator (XO)": 10.0,
    "TCXO/VCXO": 15.0,
    "Connector (per contact)": 0.5,
    "DC-DC Converter (<10W)": 100.0,
    "DC-DC Converter (≥10W)": 130.0,
    "Fuse": 2.0,
}


# === CORE FUNCTIONS ===
def pi_thermal_cycles(n_cycles: float) -> float:
    """CORRECTED: includes 8760 threshold"""
    n_cycles = validate_positive(n_cycles, "n_cycles")
    return n_cycles**0.76 if n_cycles <= 8760 else 1.7 * (n_cycles**0.6)


def pi_temperature(t: float, ea: float, t_ref: float) -> float:
    t = validate_temperature(t, "temperature")
    return math.exp(ea * ((1 / t_ref) - (1 / (273 + t))))


def pi_alpha(alpha_s: float, alpha_p: float) -> float:
    return 0.06 * (abs(alpha_s - alpha_p) ** 1.68)


def lambda_eos(is_interface: bool, interface_type: str = "Non Interfaces") -> float:
    if not is_interface:
        return 0.0
    key = EOS_INTERFACE_ALIASES.get(interface_type, interface_type)
    p = INTERFACE_EOS_VALUES.get(key, INTERFACE_EOS_VALUES["Non Interfaces"])
    return p["pi_i"] * p["l_eos"]


def calculate_ic_lambda3(pkg_type: str, pins: int = None, diag: float = None) -> float:
    pkg = IC_PACKAGE_TABLE.get(pkg_type)
    if not pkg:
        return 4.0
    f = pkg.get("formula", "fixed")
    if f == "fixed":
        return pkg["value"]
    elif f == "pins" and pins:
        return pkg["coef"] * (pins ** pkg["exp"])
    elif f == "diagonal" and diag:
        return pkg["coef"] * (diag ** pkg["exp"])
    return 4.0


# === COMPONENT CALCULATIONS ===
def lambda_integrated_circuit(
    ic_type="MOS_DIGITAL",
    transistor_count=10000,
    construction_year=2020,
    t_junction=85.0,
    package_type="TQFP-10x10",
    pins=48,
    diag: float = None,
    substrate_alpha=16.0,
    package_alpha=21.5,
    is_interface=False,
    interface_type="Non Interfaces",
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,
) -> Dict[str, float]:
    tau_on = validate_ratio(tau_on, "tau_on")
    dp = IC_DIE_TABLE.get(ic_type, IC_DIE_TABLE["MOS_DIGITAL"])
    l1, l2, ea = dp["l1"], dp["l2"], dp["ea"]
    a = max(0, construction_year - 1998)
    pi_t = pi_temperature(t_junction, ea, 328)
    lambda_die = (l1 * transistor_count * math.exp(-0.35 * a) + l2) * pi_t * tau_on
    l3 = calculate_ic_lambda3(package_type, pins=pins, diag=diag)
    pi_a = pi_alpha(substrate_alpha, package_alpha)
    pi_n = pi_thermal_cycles(n_cycles)
    lambda_pkg = 2.75e-3 * pi_a * pi_n * (delta_t**0.68) * l3
    lambda_e = lambda_eos(is_interface, interface_type)
    total = (lambda_die + lambda_pkg + lambda_e) * 1e-9
    return {
        "lambda_die": lambda_die * 1e-9,
        "lambda_package": lambda_pkg * 1e-9,
        "lambda_eos": lambda_e * 1e-9,
        "lambda_total": total,
        "fit_total": lambda_die + lambda_pkg + lambda_e,
        "pi_t": pi_t,
        "pi_n": pi_n,
        "pi_alpha": pi_a,
    }


def lambda_diode(
    diode_type="Signal (<1A)",
    t_junction=85.0,
    package="SOD-123",
    is_interface=False,
    interface_type="Non Interfaces",
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,
) -> Dict[str, float]:
    tau_on = validate_ratio(tau_on, "tau_on")
    l0 = DIODE_BASE_RATES.get(diode_type, DIODE_BASE_RATES["Signal (<1A)"])["l0"]
    pi_t = pi_temperature(t_junction, ActivationEnergy.BIPOLAR, 313)
    lambda_die = l0 * pi_t * tau_on
    lb = DISCRETE_PACKAGE_TABLE.get(package, {"lb": 1.0}).get("lb", 1.0)
    pi_n = pi_thermal_cycles(n_cycles)
    lambda_pkg = 2.75e-3 * pi_n * (delta_t**0.68) * lb
    lambda_e = lambda_eos(is_interface, interface_type)
    total = (lambda_die + lambda_pkg + lambda_e) * 1e-9
    return {
        "lambda_die": lambda_die * 1e-9,
        "lambda_package": lambda_pkg * 1e-9,
        "lambda_eos": lambda_e * 1e-9,
        "lambda_total": total,
        "fit_total": lambda_die + lambda_pkg + lambda_e,
    }


def lambda_transistor(
    transistor_type="Silicon MOSFET (≤5W)",
    t_junction=85.0,
    package="SOT-23",
    voltage_stress_vds=0.5,
    voltage_stress_vgs=0.5,
    voltage_stress_vce=0.5,
    is_interface=False,
    interface_type="Non Interfaces",
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,
) -> Dict[str, float]:
    tau_on = validate_ratio(tau_on, "tau_on")
    # IEC stress quantities are ratios; enforce valid domain
    voltage_stress_vds = validate_ratio(voltage_stress_vds, "voltage_stress_vds")
    voltage_stress_vgs = validate_ratio(voltage_stress_vgs, "voltage_stress_vgs")
    voltage_stress_vce = validate_ratio(voltage_stress_vce, "voltage_stress_vce")
    p = TRANSISTOR_BASE_RATES.get(
        transistor_type, TRANSISTOR_BASE_RATES["Silicon MOSFET (≤5W)"]
    )
    l0, tech = p["l0"], p["tech"]
    ea = ActivationEnergy.BIPOLAR if tech == "bipolar" else ActivationEnergy.MOS
    pi_t = pi_temperature(t_junction, ea, 373)
    if tech == "bipolar":
        pi_s = 0.22 * math.exp(1.7 * voltage_stress_vce)
    else:
        pi_s = (
            0.22
            * math.exp(1.7 * voltage_stress_vds)
            * 0.22
            * math.exp(3 * voltage_stress_vgs)
        )
    lambda_die = pi_s * l0 * pi_t * tau_on
    lb = DISCRETE_PACKAGE_TABLE.get(package, {"lb": 1.0}).get("lb", 1.0)
    pi_n = pi_thermal_cycles(n_cycles)
    lambda_pkg = 2.75e-3 * pi_n * (delta_t**0.68) * lb
    lambda_e = lambda_eos(is_interface, interface_type)
    total = (lambda_die + lambda_pkg + lambda_e) * 1e-9
    return {
        "lambda_die": lambda_die * 1e-9,
        "lambda_package": lambda_pkg * 1e-9,
        "lambda_eos": lambda_e * 1e-9,
        "lambda_total": total,
        "fit_total": lambda_die + lambda_pkg + lambda_e,
        "pi_s": pi_s,
    }


def lambda_capacitor(
    capacitor_type="Ceramic Class II (X7R/X5R)",
    t_ambient=25.0,
    ripple_ratio=0.0,
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,
) -> Dict[str, float]:
    tau_on = validate_ratio(tau_on, "tau_on")
    p = CAPACITOR_PARAMS.get(
        capacitor_type, CAPACITOR_PARAMS["Ceramic Class II (X7R/X5R)"]
    )
    l0, pkg_coef, ea, t_ref = p["l0"], p["pkg_coef"], p["ea"], p["t_ref"]
    t_op = (
        t_ambient + 20 * (ripple_ratio**2)
        if "Aluminum" in capacitor_type and ripple_ratio > 0
        else t_ambient
    )
    pi_t = pi_temperature(t_op, ea, t_ref)
    pi_n = pi_thermal_cycles(n_cycles)
    lambda_base = l0 * pi_t * tau_on
    lambda_pkg = l0 * pkg_coef * pi_n * (delta_t**0.68)
    total = (lambda_base + lambda_pkg) * 1e-9
    return {
        "lambda_base": lambda_base * 1e-9,
        "lambda_package": lambda_pkg * 1e-9,
        "lambda_total": total,
        "fit_total": lambda_base + lambda_pkg,
        "pi_t": pi_t,
        "pi_n": pi_n,
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
    **kw,
) -> Dict[str, float]:
    tau_on = validate_ratio(tau_on, "tau_on")
    p = RESISTOR_PARAMS.get(resistor_type, RESISTOR_PARAMS["SMD Chip Resistor"])
    l0, pkg_coef, temp_coef = p["l0"], p["pkg_coef"], p["temp_coef"]
    t_r = t_ambient + temp_coef * (operating_power / max(rated_power, 1e-6))
    pi_t = pi_temperature(t_r, ActivationEnergy.RESISTOR, 303)
    pi_n = pi_thermal_cycles(n_cycles)
    l0_eff = l0 * n_resistors
    lambda_base = l0_eff * pi_t * tau_on
    lambda_pkg = l0_eff * pkg_coef * pi_n * (delta_t**0.68)
    total = (lambda_base + lambda_pkg) * 1e-9
    return {
        "lambda_base": lambda_base * 1e-9,
        "lambda_package": lambda_pkg * 1e-9,
        "lambda_total": total,
        "fit_total": lambda_base + lambda_pkg,
        "t_resistor": t_r,
        "pi_t": pi_t,
    }


def lambda_inductor(
    inductor_type="Power Inductor",
    t_ambient=25.0,
    power_loss=0.1,
    surface_area_mm2=100.0,
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,
) -> Dict[str, float]:
    tau_on = validate_ratio(tau_on, "tau_on")
    l0 = INDUCTOR_PARAMS.get(inductor_type, INDUCTOR_PARAMS["Power Inductor"])["l0"]
    sur_dm2 = surface_area_mm2 / 10000.0
    t_c = t_ambient + 8.2 * (power_loss / max(sur_dm2, 0.01))
    pi_t = pi_temperature(t_c, ActivationEnergy.RESISTOR, 303)
    pi_n = pi_thermal_cycles(n_cycles)
    lambda_base = l0 * pi_t * tau_on
    lambda_pkg = l0 * 7e-3 * pi_n * (delta_t**0.68)
    total = (lambda_base + lambda_pkg) * 1e-9
    return {
        "lambda_base": lambda_base * 1e-9,
        "lambda_package": lambda_pkg * 1e-9,
        "lambda_total": total,
        "fit_total": lambda_base + lambda_pkg,
        "t_component": t_c,
    }


def lambda_misc_component(
    component_type, n_contacts=1, n_cycles=5256, delta_t=3.0, tau_on=1.0, **kw
) -> Dict[str, float]:
    tau_on = validate_ratio(tau_on, "tau_on")
    base = MISC_COMPONENT_RATES.get(component_type, 10.0)
    if "Connector" in component_type:
        base *= n_contacts
    pi_n = pi_thermal_cycles(n_cycles)
    total = base * (tau_on + 3e-3 * pi_n * (delta_t**0.68)) * 1e-9
    return {"lambda_total": total, "fit_total": total * 1e9}


# === SYSTEM RELIABILITY ===
def reliability_from_lambda(lam: float, hours: float) -> float:
    return math.exp(-lam * hours)


def lambda_from_reliability(r: float, hours: float) -> float:
    if r <= 0:
        return float("inf")
    if r >= 1:
        return 0.0
    return -math.log(r) / hours


def mttf_from_lambda(lam: float) -> float:
    return float("inf") if lam <= 0 else 1.0 / lam


def r_series(r_list: List[float]) -> float:
    result = 1.0
    for r in r_list:
        result *= r
    return result


def r_parallel(r_list: List[float]) -> float:
    p_fail = 1.0
    for r in r_list:
        p_fail *= 1 - r
    return 1.0 - p_fail


def r_k_of_n(r_list: List[float], k: int) -> float:
    n = len(r_list)
    if k > n or k < 1:
        return 0.0
    if k == 1:
        return r_parallel(r_list)
    if k == n:
        return r_series(r_list)
    if len(set(r_list)) == 1:
        r = r_list[0]
        return sum(
            math.comb(n, i) * (r**i) * ((1 - r) ** (n - i)) for i in range(k, n + 1)
        )
    return r_list[-1] * r_k_of_n(r_list[:-1], k - 1) + (1 - r_list[-1]) * r_k_of_n(
        r_list[:-1], k
    )


def lambda_series(lam_list: List[float]) -> float:
    return sum(lam_list)


# === FIELD DEFINITIONS ===
def get_component_types() -> List[str]:
    return [
        "Integrated Circuit",
        "Diode",
        "Transistor",
        "Capacitor",
        "Resistor",
        "Inductor/Transformer",
        "Miscellaneous",
    ]


def get_field_definitions(ct: str) -> Dict[str, Dict]:
    common = {
        "n_cycles": {"type": "int", "default": 5256, "help": "Annual thermal cycles"},
        "delta_t": {"type": "float", "default": 3.0, "help": "ΔT (°C)"},
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
            "default": "Non Interfaces",
        },
    }
    if ct == "Integrated Circuit":
        return {
            "ic_type": {
                "type": "choice",
                "choices": list(IC_TYPE_CHOICES.keys()),
                "default": "Microcontroller/DSP",
                "required": True,
            },
            "transistor_count": {"type": "int", "default": 10000, "required": True},
            "t_junction": {"type": "float", "default": 85.0, "required": True},
            "package": {
                "type": "choice",
                "choices": list(IC_PACKAGE_CHOICES.keys()),
                "default": "QFP-48 (7x7mm)",
                "required": True,
            },
            "substrate": {
                "type": "choice",
                "choices": list(THERMAL_EXPANSION_SUBSTRATE.keys()),
                "default": "FR4 (Epoxy Glass)",
            },
            # Package CTE is an explicit IEC input (alpha_p); keep default but expose it
            "package_alpha": {
                "type": "float",
                "default": 21.5,
                "help": "Package CTE αp (ppm/°C)",
            },
            **iface,
            **common,
        }
    elif ct == "Diode":
        return {
            "diode_type": {
                "type": "choice",
                "choices": list(DIODE_BASE_RATES.keys()),
                "default": "Signal (<1A)",
                "required": True,
            },
            "t_junction": {"type": "float", "default": 85.0, "required": True},
            "package": {
                "type": "choice",
                "choices": list(DISCRETE_PACKAGE_TABLE.keys()),
                "default": "SOD-123",
                "required": True,
            },
            **iface,
            **common,
        }
    elif ct == "Transistor":
        return {
            "transistor_type": {
                "type": "choice",
                "choices": list(TRANSISTOR_BASE_RATES.keys()),
                "default": "Silicon MOSFET (≤5W)",
                "required": True,
            },
            "t_junction": {"type": "float", "default": 85.0, "required": True},
            "package": {
                "type": "choice",
                "choices": list(DISCRETE_PACKAGE_TABLE.keys()),
                "default": "SOT-23",
                "required": True,
            },
            "voltage_stress_vds": {
                "type": "float",
                "default": 0.5,
                "help": "VDS stress ratio (0-1)",
            },
            "voltage_stress_vgs": {
                "type": "float",
                "default": 0.5,
                "help": "VGS stress ratio (0-1)",
            },
            "voltage_stress_vce": {
                "type": "float",
                "default": 0.5,
                "help": "VCE stress ratio (0-1)",
            },
            **iface,
            **common,
        }
    elif ct == "Capacitor":
        return {
            "capacitor_type": {
                "type": "choice",
                "choices": list(CAPACITOR_PARAMS.keys()),
                "default": "Ceramic Class II (X7R/X5R)",
                "required": True,
            },
            "t_ambient": {"type": "float", "default": 25.0, "required": True},
            **common,
        }
    elif ct == "Resistor":
        return {
            "resistor_type": {
                "type": "choice",
                "choices": list(RESISTOR_PARAMS.keys()),
                "default": "SMD Chip Resistor",
                "required": True,
            },
            "t_ambient": {"type": "float", "default": 25.0, "required": True},
            "operating_power": {"type": "float", "default": 0.01, "required": True},
            "rated_power": {"type": "float", "default": 0.125, "required": True},
            **common,
        }
    return {
        "component_subtype": {
            "type": "choice",
            "choices": list(MISC_COMPONENT_RATES.keys()),
            "default": "Crystal Oscillator (XO)",
        },
        **common,
    }


def calculate_component_lambda(ct: str, params: Dict[str, Any]) -> Dict[str, float]:
    if ct == "Integrated Circuit":
        ic_key = IC_TYPE_CHOICES.get(
            params.get("ic_type", "Microcontroller/DSP"), "MOS_DIGITAL"
        )
        pkg = params.get("package", "QFP-48 (7x7mm)")
        pkg_info = IC_PACKAGE_CHOICES.get(pkg, ("TQFP-7x7", 48))
        sub = THERMAL_EXPANSION_SUBSTRATE.get(
            params.get("substrate", "FR4 (Epoxy Glass)"), 16.0
        )
        pins = pkg_info[1] if len(pkg_info) > 1 else 48
        diag = pkg_info[2] if len(pkg_info) > 2 else None
        return lambda_integrated_circuit(
            ic_type=ic_key,
            transistor_count=params.get("transistor_count", 10000),
            t_junction=params.get("t_junction", 85.0),
            package_type=pkg_info[0],
            pins=pins,
            diag=diag,
            substrate_alpha=sub,
            package_alpha=params.get("package_alpha", 21.5),
            is_interface=params.get("is_interface", False),
            interface_type=params.get("interface_type", "Non Interfaces"),
            n_cycles=params.get("n_cycles", 5256),
            delta_t=params.get("delta_t", 3.0),
            tau_on=params.get("tau_on", 1.0),
        )
    elif ct == "Diode":
        return lambda_diode(**params)
    elif ct == "Transistor":
        return lambda_transistor(**params)
    elif ct == "Capacitor":
        return lambda_capacitor(**params)
    elif ct == "Resistor":
        return lambda_resistor(**params)
    elif ct == "Inductor/Transformer":
        return lambda_inductor(**params)
    return lambda_misc_component(
        component_type=params.get("component_subtype", "Crystal Oscillator (XO)"),
        **params,
    )


def calculate_lambda(cls: str, params: Dict = None) -> float:
    if params is None:
        params = {}
    c = cls.lower()
    if "resistor" in c:
        return lambda_resistor(**params)["lambda_total"]
    if "capacitor" in c:
        return lambda_capacitor(**params)["lambda_total"]
    if "transistor" in c or "mosfet" in c:
        return lambda_transistor(**params)["lambda_total"]
    if "diode" in c:
        return lambda_diode(**params)["lambda_total"]
    if "ic" in c or "integrated" in c:
        return lambda_integrated_circuit(**params)["lambda_total"]
    return 10e-9


# === UTILITIES ===
def fit_to_lambda(fit):
    return fit * 1e-9


def lambda_to_fit(lam):
    return lam * 1e9


def format_lambda(lam, as_fit=True):
    return f"{lam*1e9:.2f} FIT" if as_fit else f"{lam:.2e} /h"


def format_reliability(r):
    return f"{r:.6f}" if r >= 0.9999 else f"{r:.4f}" if r >= 0.99 else f"{r:.3f}"


reliability = reliability_from_lambda
component_failure_rate = lambda c, p=None: calculate_lambda(c, p.to_dict() if p else {})
