"""
IEC TR 62380:2004 Reliability Prediction Models
================================================
Core failure rate calculations for electronic components.

Implements the Arrhenius-based thermal acceleration model, Coffin-Manson
thermal cycling damage, CTE mismatch stress, and EOS contributions as
defined in IEC TR 62380:2004 (Reliability Data Handbook).

All failure rates are expressed in FIT internally (failures per 10^9 hours)
and converted to per-hour (lambda) at the output boundary.

Component models implemented (with IEC TR 62380 section references):
    - Integrated Circuits         Section 8
    - Diodes                      Section 9
    - Transistors                 Section 10
    - Optocouplers                Section 10.3
    - Thyristors / Triacs         Section 10.2
    - Capacitors                  Section 11
    - Resistors / Potentiometers  Section 12
    - Inductors / Transformers    Section 13
    - Relays                      Section 14
    - Connectors                  Section 15
    - PCB / Solder Joints         Section 16
    - Miscellaneous               Appendix

Author:  Eliot Abramo
"""

import math
from typing import Dict, List, Any, Optional, Tuple

__version__ = "3.1.0"
__author__ = "Eliot Abramo"


# =============================================================================
# Input validation -- fail-safe, clamp-safe, and informative
# =============================================================================


def validate_ratio(val, name: str = "ratio") -> float:
    """Ensure value is in [0, 1]. Clamps silently for robustness."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be numeric, got {type(val).__name__}")
    return max(0.0, min(1.0, v))


def validate_positive(val, name: str = "value") -> float:
    """Ensure value is >= 0. Negative values are clamped to 0."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, v)


def validate_temperature(val, name: str = "temperature") -> float:
    """Ensure temperature is above absolute zero (-273.15 C)."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return 25.0
    return max(-273.15, v)


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
# System topology types
# =============================================================================
class ConnectionType:
    """Block-diagram connection modes for reliability modelling."""

    SERIES = "series"
    PARALLEL = "parallel"
    K_OF_N = "k_of_n"

    def __init__(self, value: Optional[str] = None):
        self._value = value or self.SERIES

    @property
    def value(self) -> str:
        return getattr(self, "_value", self.SERIES)

    def __eq__(self, other) -> bool:
        if isinstance(other, ConnectionType):
            return self.value == other.value
        return self.value == other

    def __hash__(self) -> int:
        return hash(self.value)

    def __str__(self) -> str:
        return self.value


# =============================================================================
# IEC TR 62380 activation energies (Ea / k_B in Kelvin)
# =============================================================================


class ActivationEnergy:
    """Activation energy constants (Ea/k_B in Kelvin)."""

    MOS = 3480  # 0.30 eV
    BIPOLAR = 4640  # 0.40 eV
    CAPACITOR_LOW = 1160  # 0.10 eV
    CAPACITOR_MED = 1740  # 0.15 eV
    CAPACITOR_HIGH = 2900  # 0.25 eV
    ALUMINUM_CAP = 4640  # 0.40 eV
    RESISTOR = 1740  # 0.15 eV
    GAN = 5800  # 0.50 eV
    OPTOCOUPLER = 4060  # 0.35 eV
    SIC = 5800  # 0.50 eV


# =============================================================================
# EOS (Electrical Overstress) interface categories
# IEC TR 62380, Section 7
# =============================================================================

INTERFACE_EOS_VALUES = {
    # IEC TR 62380, Section 7.3.1, Interface circuits table
    # lambda_EOS in FIT, pi_I factor
    "Not Interface": {"pi_i": 0, "l_eos": 0},
    "Computer": {"pi_i": 1, "l_eos": 10},
    "Telecom (Switching)": {"pi_i": 1, "l_eos": 15},
    "Telecom (Transmitting)": {"pi_i": 1, "l_eos": 40},
    "Telecom (Access/Subscriber Cards)": {"pi_i": 1, "l_eos": 70},
    "Telecom (Subscriber Equipment)": {"pi_i": 1, "l_eos": 100},
    "Railways / Payphone": {"pi_i": 1, "l_eos": 100},
    "Avionics (On Board)": {"pi_i": 1, "l_eos": 20},
    "Power Supply / Converters": {"pi_i": 1, "l_eos": 40},
    # Extended (common engineering contexts beyond table)
    "Industrial": {"pi_i": 1, "l_eos": 30},
    "Automotive": {"pi_i": 1, "l_eos": 50},
    "Space (LEO)": {"pi_i": 1, "l_eos": 25},
    "Space (GEO)": {"pi_i": 1, "l_eos": 35},
    "Military": {"pi_i": 1, "l_eos": 45},
}

# =============================================================================
# CTE coefficients for substrate materials (ppm/C)
# IEC TR 62380, Section 8.3
# =============================================================================

THERMAL_EXPANSION_SUBSTRATE = {
    # IEC TR 62380, Table 14 -- alpha_S (substrate) in ppm/degC
    "FR4 / Epoxy Glass (G-10)": 16.0,
    "PTFE Glass (Polytetrafluoroethylene)": 20.0,
    "Polyimide / Flex (Aramid)": 6.5,
    "Cu/Invar/Cu (20/60/20)": 5.4,
    "Aluminum (Metal Core)": 23.0,
    "Rogers (PTFE Blend)": 10.0,
    "BT (Bismaleimide Triazine)": 14.0,
}

THERMAL_EXPANSION_PACKAGE = {
    # IEC TR 62380, Table 14 -- alpha_C (component) in ppm/degC
    "Epoxy (Plastic package)": 21.5,
    "Alumina (Ceramic package)": 6.5,
    "Kovar (Metallic package)": 5.0,
    "Bare Die / Flip Chip": 2.6,
}


# =============================================================================
# IC die-level base failure rate tables
# IEC TR 62380, Section 8.1
# =============================================================================

IC_DIE_TABLE = {
    # =========================================================================
    # IEC TR 62380, Table 16 -- Values of lambda_1 and lambda_2
    # for integrated circuits families
    #
    # Key fields:
    #   l1      : per-transistor base failure rate (FIT)
    #   l2      : technology base failure rate (FIT)
    #   ea      : Arrhenius activation energy constant (Ea/k_B in K)
    #   t_ref   : Reference temperature (K) for pi_t calculation
    #   n_unit  : What "N" counts (for help text / GUI)
    #   n_per   : Multiplier to convert user-visible count to transistors
    #             e.g. "4 per gate" means 1 gate = 4 transistors
    # =========================================================================

    # --- Silicon: MOS Standard circuits (3) ---
    "MOS_DIGITAL": {
        "l1": 3.4e-6, "l2": 1.7, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "gates", "n_per": 4,
        "desc": "Digital circuits, Micros, DSP",
    },
    "MOS_LINEAR": {
        "l1": 1.0e-2, "l2": 4.2, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "transistors", "n_per": 1,
        "desc": "Linear circuits (MOS)",
    },
    "MOS_DIG_LIN_TELECOM": {
        "l1": 2.7e-4, "l2": 20.0, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "transistors", "n_per": 1,
        "desc": "Digital/linear (Telecom, CAN, CNA, RAMDAC)",
    },
    "MOS_ROM": {
        "l1": 1.7e-7, "l2": 8.8, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "bits", "n_per": 1,
        "desc": "ROM - Read only memory",
    },
    "MOS_DRAM": {
        "l1": 1.0e-7, "l2": 5.6, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "bits", "n_per": 1,
        "desc": "DRAM / VideoRAM / AudioRAM",
    },
    "MOS_SRAM_FAST": {
        "l1": 1.7e-7, "l2": 8.8, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "bits", "n_per": 4,
        "desc": "High-speed SRAM, FIFO (mixed MOS)",
    },
    "MOS_SRAM_LOW": {
        "l1": 1.7e-7, "l2": 8.8, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "bits", "n_per": 6,
        "desc": "Low-consumption SRAM (CMOS)",
    },
    "MOS_SRAM_DUAL": {
        "l1": 1.7e-7, "l2": 8.8, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "bits", "n_per": 8,
        "desc": "Dual-access Static RAM",
    },
    "MOS_EPROM": {
        "l1": 2.6e-7, "l2": 34.0, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "prog. points", "n_per": 1,
        "desc": "EPROM / UVPROM / REPROM / OTP / FLASH (block erase)",
    },
    "MOS_EEPROM": {
        "l1": 6.5e-7, "l2": 16.0, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "prog. points", "n_per": 2,
        "desc": "EEPROM / Flash EEPROM (word-erasable)",
    },

    # --- Silicon: MOS ASIC circuits ---
    "MOS_ASIC_STDCELL": {
        "l1": 1.2e-5, "l2": 10.0, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "gates", "n_per": 4,
        "desc": "ASIC Standard Cell / Full Custom",
    },
    "MOS_ASIC_GATE_ARRAY": {
        "l1": 2.0e-5, "l2": 10.0, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "gates", "n_per": 4,
        "desc": "ASIC Gate Arrays",
    },
    "MOS_LCA": {
        "l1": 4.0e-5, "l2": 8.8, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "gates", "n_per": 40,
        "desc": "LCA / FPGA (RAM-based, ext. memory configured)",
    },
    "MOS_PLD": {
        "l1": 1.2e-3, "l2": 16.0, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "grid points", "n_per": 3,
        "desc": "PLD (GAL, PAL) - AND/OR array",
    },
    "MOS_CPLD": {
        "l1": 2.0e-5, "l2": 34.0, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "macrocells", "n_per": 100,
        "desc": "CPLD / EPLD / MAX / FLEX / FPGA (flash/antifuse)",
    },

    # --- Silicon: Bipolar circuits (1) ---
    "BIPOLAR_DIGITAL": {
        "l1": 6.0e-4, "l2": 1.7, "ea": ActivationEnergy.BIPOLAR, "t_ref": 328,
        "n_unit": "gates", "n_per": 3,
        "desc": "Bipolar digital circuits",
    },
    "BIPOLAR_LINEAR": {
        "l1": 2.2e-2, "l2": 3.3, "ea": ActivationEnergy.BIPOLAR, "t_ref": 328,
        "n_unit": "transistors", "n_per": 1,
        "desc": "Bipolar linear circuits (FET, others)",
    },
    "BIPOLAR_MMIC": {
        "l1": 1.0, "l2": 3.3, "ea": ActivationEnergy.BIPOLAR, "t_ref": 328,
        "n_unit": "transistors", "n_per": 1,
        "desc": "Bipolar MMIC",
    },
    "BIPOLAR_LOW_V": {
        "l1": 2.7e-3, "l2": 20.0, "ea": ActivationEnergy.BIPOLAR, "t_ref": 328,
        "n_unit": "transistors", "n_per": 1,
        "desc": "Bipolar linear/digital, low voltage (<30V)",
    },
    "BIPOLAR_HIGH_V": {
        "l1": 2.7e-2, "l2": 20.0, "ea": ActivationEnergy.BIPOLAR, "t_ref": 328,
        "n_unit": "transistors", "n_per": 1,
        "desc": "Bipolar linear/digital, high voltage (>=30V)",
    },
    "BIPOLAR_SRAM": {
        "l1": 3.0e-4, "l2": 1.7, "ea": ActivationEnergy.BIPOLAR, "t_ref": 328,
        "n_unit": "bits", "n_per": 2.5,
        "desc": "Bipolar static read-access memories",
    },
    "BIPOLAR_PROM": {
        "l1": 1.5e-4, "l2": 32.0, "ea": ActivationEnergy.BIPOLAR, "t_ref": 328,
        "n_unit": "prog. points", "n_per": 1.2,
        "desc": "Bipolar programmable read-only memory",
    },
    "BIPOLAR_PLA": {
        "l1": 1.5e-4, "l2": 32.0, "ea": ActivationEnergy.BIPOLAR, "t_ref": 328,
        "n_unit": "grid points", "n_per": 1.6,
        "desc": "Bipolar one-time prog. logic array (AND/OR)",
    },
    "BIPOLAR_GATE_ARRAY": {
        "l1": 1.0e-3, "l2": 10.0, "ea": ActivationEnergy.BIPOLAR, "t_ref": 328,
        "n_unit": "gates", "n_per": 3,
        "desc": "Bipolar gate arrays (PROM, PLD/PAL)",
    },

    # --- Silicon: BiCMOS (Bipolar and MOS circuits) ---
    "BICMOS_DIGITAL": {
        "l1": 1.0e-6, "l2": 1.7, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "gates", "n_per": 4,
        "desc": "BiCMOS digital circuits",
    },
    "BICMOS_LOW_V": {
        "l1": 2.7e-4, "l2": 20.0, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "transistors", "n_per": 1,
        "desc": "BiCMOS linear/digital, low voltage (<6V)",
    },
    "BICMOS_HIGH_V": {
        "l1": 2.7e-3, "l2": 20.0, "ea": ActivationEnergy.BIPOLAR, "t_ref": 328,
        "n_unit": "transistors", "n_per": 1,
        "desc": "BiCMOS linear/digital, high voltage (>=6V) / Smart Power",
    },
    "BICMOS_SRAM": {
        "l1": 6.8e-7, "l2": 8.8, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "bits", "n_per": 4,
        "desc": "BiCMOS Static Read Access Memory",
    },
    "BICMOS_GATE_ARRAY": {
        "l1": 6.4e-5, "l2": 10.0, "ea": ActivationEnergy.MOS, "t_ref": 328,
        "n_unit": "gates", "n_per": 4,
        "desc": "BiCMOS gate arrays",
    },

    # --- Gallium arsenide ---
    "GAAS_DIGITAL_NOR": {
        "l1": 2.5, "l2": 25.0, "ea": ActivationEnergy.MOS, "t_ref": 373,
        "n_unit": "gates", "n_per": 5,
        "desc": "GaAs digital - normally-on transistors only",
    },
    "GAAS_DIGITAL_MIXED": {
        "l1": 4.5e-4, "l2": 16.0, "ea": ActivationEnergy.BIPOLAR, "t_ref": 373,
        "n_unit": "gates", "n_per": 3,
        "desc": "GaAs digital - normally-off & normally-on",
    },
    "GAAS_MMIC_LOW": {
        "l1": 2.0, "l2": 20.0, "ea": ActivationEnergy.MOS, "t_ref": 373,
        "n_unit": "transistors", "n_per": 1,
        "desc": "GaAs MMIC - low noise / low power (<100mW)",
    },
    "GAAS_MMIC_POWER": {
        "l1": 4.0, "l2": 40.0, "ea": ActivationEnergy.BIPOLAR, "t_ref": 373,
        "n_unit": "transistors", "n_per": 1,
        "desc": "GaAs MMIC - power (>100mW)",
    },
}

IC_TYPE_CHOICES = {
    # --- MOS Standard ---
    "MOS Digital (Micro/DSP)": "MOS_DIGITAL",
    "MOS Linear": "MOS_LINEAR",
    "MOS Digital/Linear (Telecom, CAN, RAMDAC)": "MOS_DIG_LIN_TELECOM",
    "MOS ROM": "MOS_ROM",
    "MOS DRAM / VideoRAM / AudioRAM": "MOS_DRAM",
    "MOS SRAM High-Speed / FIFO": "MOS_SRAM_FAST",
    "MOS SRAM Low-Consumption (CMOS)": "MOS_SRAM_LOW",
    "MOS SRAM Dual-Access": "MOS_SRAM_DUAL",
    "MOS EPROM / UVPROM / OTP / Flash (block)": "MOS_EPROM",
    "MOS EEPROM / Flash EEPROM (word)": "MOS_EEPROM",
    # --- MOS ASIC ---
    "ASIC Standard Cell / Full Custom": "MOS_ASIC_STDCELL",
    "ASIC Gate Array": "MOS_ASIC_GATE_ARRAY",
    "FPGA / LCA (RAM-based)": "MOS_LCA",
    "PLD (GAL, PAL)": "MOS_PLD",
    "CPLD / EPLD / FPGA (flash/antifuse)": "MOS_CPLD",
    # --- Bipolar ---
    "Bipolar Digital": "BIPOLAR_DIGITAL",
    "Bipolar Linear (Op-Amp, Comparator)": "BIPOLAR_LINEAR",
    "Bipolar MMIC": "BIPOLAR_MMIC",
    "Bipolar Linear/Digital Low Voltage (<30V)": "BIPOLAR_LOW_V",
    "Bipolar Linear/Digital High Voltage (>=30V)": "BIPOLAR_HIGH_V",
    "Bipolar SRAM": "BIPOLAR_SRAM",
    "Bipolar PROM": "BIPOLAR_PROM",
    "Bipolar Prog. Logic Array (AND/OR)": "BIPOLAR_PLA",
    "Bipolar Gate Array": "BIPOLAR_GATE_ARRAY",
    # --- BiCMOS ---
    "BiCMOS Digital": "BICMOS_DIGITAL",
    "BiCMOS Low Voltage (<6V) / LDO": "BICMOS_LOW_V",
    "BiCMOS High Voltage (>=6V) / DC-DC / Smart Power": "BICMOS_HIGH_V",
    "BiCMOS SRAM": "BICMOS_SRAM",
    "BiCMOS Gate Array": "BICMOS_GATE_ARRAY",
    # --- Gallium arsenide ---
    "GaAs Digital (normally-on only)": "GAAS_DIGITAL_NOR",
    "GaAs Digital (normally-off & on)": "GAAS_DIGITAL_MIXED",
    "GaAs MMIC Low Power (<100mW)": "GAAS_MMIC_LOW",
    "GaAs MMIC Power (>100mW)": "GAAS_MMIC_POWER",
}


# =============================================================================
# IC package stress contribution (lambda_3)
# IEC TR 62380, Section 8.2, Table 8-2
# =============================================================================

IC_PACKAGE_CHOICES = {
    # --- DIP (Through-hole, Table 17a: PDIL/CDIL) ---
    "PDIL-8 (Plastic DIP)": ("PDIL", 8),
    "PDIL-14": ("PDIL", 14),
    "PDIL-16": ("PDIL", 16),
    "PDIL-20": ("PDIL", 20),
    "PDIL-28": ("PDIL", 28),
    "PDIL-40": ("PDIL", 40),
    "PDIL-64": ("PDIL", 64),
    "CDIL-8 (Ceramic DIP)": ("CDIL", 8),
    "CDIL-14": ("CDIL", 14),
    "CDIL-16": ("CDIL", 16),
    "CDIL-28": ("CDIL", 28),
    "CDIL-40": ("CDIL", 40),
    "CDIL-64": ("CDIL", 64),
    # --- SO / SOP (Table 17a: 0.012 x S^1.65) ---
    "SO-8 (1.27mm pitch)": ("SO", 8),
    "SO-14": ("SO", 14),
    "SO-16": ("SO", 16),
    "SO-20": ("SO", 20),
    "SO-28": ("SO", 28),
    "Power SO": ("SO", 8),
    # --- SOJ (Table 17a: 0.023 x S^1.5) ---
    "SOJ-28": ("SOJ", 28),
    "SOJ-32": ("SOJ", 32),
    "SOJ-40": ("SOJ", 40),
    "SOJ-44": ("SOJ", 44),
    # --- VSOP (Table 17a: 0.011 x S^1.47) ---
    "VSOP-40": ("VSOP", 40),
    "VSOP-48": ("VSOP", 48),
    "VSOP-56": ("VSOP", 56),
    # --- SSOP (Table 17a: 0.013 x S^1.35) ---
    "SSOP-8": ("SSOP", 8),
    "SSOP-16": ("SSOP", 16),
    "SSOP-28": ("SSOP", 28),
    "SSOP-48": ("SSOP", 48),
    "SSOP-56": ("SSOP", 56),
    # --- TSSOP (Table 17a: 0.011 x S^1.4) ---
    "TSSOP-8": ("TSSOP", 8),
    "TSSOP-14": ("TSSOP", 14),
    "TSSOP-20": ("TSSOP", 20),
    "TSSOP-28": ("TSSOP", 28),
    "TSSOP-38": ("TSSOP", 38),
    # --- TSOP I (Table 17a: two pitch variants) ---
    "TSOP-I-32 (0.55mm)": ("TSOP_I_055", 32),
    "TSOP-I-32 (0.5mm)": ("TSOP_I_050", 32),
    # --- TSOP II (Table 17a: four pitch variants) ---
    "TSOP-II-28 (0.8mm)": ("TSOP_II_080", 28),
    "TSOP-II-44 (0.8mm)": ("TSOP_II_080", 44),
    "TSOP-II-54 (0.8mm)": ("TSOP_II_080", 54),
    "TSOP-II-40 (0.65mm)": ("TSOP_II_065", 40),
    "TSOP-II-54 (0.65mm)": ("TSOP_II_065", 54),
    "TSOP-II-60 (0.65mm)": ("TSOP_II_065", 60),
    "TSOP-II-40 (0.5mm)": ("TSOP_II_050", 40),
    "TSOP-II-60 (0.5mm)": ("TSOP_II_050", 60),
    "TSOP-II-40 (0.4mm)": ("TSOP_II_040", 40),
    "TSOP-II-60 (0.4mm)": ("TSOP_II_040", 60),
    # --- PLCC / CLCC (Table 17a: 0.021 x S^1.57) ---
    "PLCC-20": ("PLCC", 20),
    "PLCC-28": ("PLCC", 28),
    "PLCC-44": ("PLCC", 44),
    "PLCC-52": ("PLCC", 52),
    "PLCC-68": ("PLCC", 68),
    "PLCC-84": ("PLCC", 84),
    "CLCC (Ceramic Leadless)": ("PLCC", 44),
    # --- QFP / TQFP / PQFP (Table 17a: fixed values by body size) ---
    "QFP-32 (5x5mm)": ("PQFP-5x5", 32),
    "QFP-44 (10x10mm)": ("PQFP-10x10", 44),
    "QFP-48 (7x7mm)": ("PQFP-7x7", 48),
    "QFP-64 (10x10mm)": ("PQFP-10x10", 64),
    "QFP-80 (14x14mm)": ("PQFP-14x14", 80),
    "QFP-100 (14x14mm)": ("PQFP-14x14", 100),
    "QFP-100 (14x20mm)": ("PQFP-14x20", 100),
    "QFP-112 (20x20mm)": ("PQFP-20x20", 112),
    "QFP-128 (14x20mm)": ("PQFP-14x20", 128),
    "QFP-144 (20x20mm)": ("PQFP-20x20", 144),
    "QFP-176 (28x28mm)": ("PQFP-28x28", 176),
    "QFP-208 (28x28mm)": ("PQFP-28x28", 208),
    "QFP-240 (32x32mm)": ("PQFP-32x32", 240),
    "QFP-304 (40x40mm)": ("PQFP-40x40", 304),
    "CQFP (Ceramic QFP)": ("PQFP-14x14", 100),
    "MQFP (Metal QFP)": ("PQFP-14x14", 100),
    # --- QFN (Table 17b Method 2: peripheral, 0.048 x D^1.68) ---
    "QFN-16 (3x3mm)": ("QFN", 16, 4.24),
    "QFN-20 (4x4mm)": ("QFN", 20, 5.66),
    "QFN-32 (5x5mm)": ("QFN", 32, 7.07),
    "QFN-48 (7x7mm)": ("QFN", 48, 9.90),
    "QFN-64 (9x9mm)": ("QFN", 64, 12.73),
    # --- BGA (Table 17a: fixed values by body size) ---
    "PBGA-64 (13.5x15mm)": ("PBGA-13x15", 64),
    "PBGA-256 (17x19mm)": ("PBGA-17x19", 256),
    "PBGA-484 (23x23mm)": ("PBGA-23x23", 484),
    "PBGA-676 (27x27mm)": ("PBGA-27x27", 676),
    "PBGA-900 (35x35mm)": ("PBGA-35x35", 900),
    "SBGA-580 (42.5x42.5mm)": ("SBGA-42x42", 580),
    "SBGA-672 (27x27mm)": ("SBGA-27x27", 672),
    "CBGA (Ceramic BGA)": ("PBGA-23x23", 484),
    # --- CSP / WLCSP (Table 17b Method 2: matrix, 0.073 x D^1.68) ---
    "CSP / uBGA (small)": ("CSP", 0, 3.0),
    "CSP / uBGA (medium)": ("CSP", 0, 5.0),
    "WLCSP": ("CSP", 0, 2.5),
    # --- PGA (Table 17a: 9 + 0.09 x S) ---
    "PPGA-68 (Plastic PGA)": ("PPGA", 68),
    "PPGA-100": ("PPGA", 100),
    "PPGA-160": ("PPGA", 160),
    "CPGA-68 (Ceramic PGA)": ("CPGA", 68),
    "CPGA-100": ("CPGA", 100),
    "CPGA-160": ("CPGA", 160),
    # --- COB (Table 17b: bare die) ---
    "COB (Chip on Board)": ("COB", 0, 3.0),
}

IC_PACKAGE_TABLE = {
    # =========================================================================
    # IEC TR 62380, Table 17a -- lambda_3 as function of S (pin count)
    # Table 17b -- lambda_3 as function of D (package diagonal)
    #
    # formula types:
    #   "pins"     -> coef * S^exp
    #   "linear"   -> offset + coef * S
    #   "fixed"    -> constant value
    #   "diagonal" -> coef * D^exp   (D = package diagonal in mm)
    # =========================================================================

    # --- Through-hole DIP: Table 17a -> lambda_3 = 9 + 0.09 * S ---
    "PDIL": {"formula": "linear", "offset": 9.0, "coef": 0.09},
    "CDIL": {"formula": "linear", "offset": 9.0, "coef": 0.09},

    # --- SO / SOP (1.27mm pitch): 0.012 * S^1.65 ---
    "SO": {"formula": "pins", "coef": 0.012, "exp": 1.65},

    # --- SOJ (1.27mm pitch): 0.023 * S^1.5 ---
    "SOJ": {"formula": "pins", "coef": 0.023, "exp": 1.50},

    # --- VSOP (0.76mm pitch): 0.011 * S^1.47 ---
    "VSOP": {"formula": "pins", "coef": 0.011, "exp": 1.47},

    # --- SSOP (0.65mm pitch): 0.013 * S^1.35 ---
    "SSOP": {"formula": "pins", "coef": 0.013, "exp": 1.35},

    # --- TSSOP (0.65mm pitch): 0.011 * S^1.4 ---
    "TSSOP": {"formula": "pins", "coef": 0.011, "exp": 1.40},

    # --- TSOP I (two pitch variants) ---
    "TSOP_I_055": {"formula": "pins", "coef": 0.54, "exp": 0.40},
    "TSOP_I_050": {"formula": "pins", "coef": 1.0, "exp": 0.36},

    # --- TSOP II (four pitch variants) ---
    "TSOP_II_080": {"formula": "pins", "coef": 0.04, "exp": 1.20},
    "TSOP_II_065": {"formula": "pins", "coef": 0.042, "exp": 1.10},
    "TSOP_II_050": {"formula": "pins", "coef": 0.075, "exp": 0.90},
    "TSOP_II_040": {"formula": "pins", "coef": 0.13, "exp": 0.70},

    # --- PLCC (1.27mm pitch): 0.021 * S^1.57 ---
    "PLCC": {"formula": "pins", "coef": 0.021, "exp": 1.57},

    # --- PQFP / TQFP: fixed values by body size (Table 17a) ---
    "PQFP-5x5": {"formula": "fixed", "value": 1.3},
    "PQFP-7x7": {"formula": "fixed", "value": 2.5},
    "PQFP-10x10": {"formula": "fixed", "value": 4.1},
    "PQFP-14x14": {"formula": "fixed", "value": 7.2},
    "PQFP-14x20": {"formula": "fixed", "value": 10.2},
    "PQFP-20x20": {"formula": "fixed", "value": 12.0},
    "PQFP-28x28": {"formula": "fixed", "value": 23.0},
    "PQFP-32x32": {"formula": "fixed", "value": 29.0},
    "PQFP-40x40": {"formula": "fixed", "value": 42.0},

    # --- PBGA: fixed values by body size (Table 17a) ---
    "PBGA-13x15": {"formula": "fixed", "value": 11.4},
    "PBGA-17x19": {"formula": "fixed", "value": 16.6},
    "PBGA-23x23": {"formula": "fixed", "value": 26.6},
    "PBGA-27x27": {"formula": "fixed", "value": 33.0},
    "PBGA-35x35": {"formula": "fixed", "value": 51.3},

    # --- SBGA ---
    "SBGA-42x42": {"formula": "fixed", "value": 71.0},
    "SBGA-27x27": {"formula": "fixed", "value": 33.0},

    # --- PGA: Table 17a -> lambda_3 = 9 + 0.09 * S ---
    "PPGA": {"formula": "linear", "offset": 9.0, "coef": 0.09},
    "CPGA": {"formula": "linear", "offset": 9.0, "coef": 0.09},

    # --- Table 17b Method 2: diagonal-based ---
    # Two rows (SO, TSOP etc.): 0.024 * D^1.68
    "TWO_ROW": {"formula": "diagonal", "coef": 0.024, "exp": 1.68},
    # Peripheral (PLCC, QFP, QFN): 0.048 * D^1.68
    "QFN": {"formula": "diagonal", "coef": 0.048, "exp": 1.68},
    # Matrix (BGA, CSP, uBGA): 0.073 * D^1.68
    "CSP": {"formula": "diagonal", "coef": 0.073, "exp": 1.68},
    # COB (bare die): 0.048 * D^1.68
    "COB": {"formula": "diagonal", "coef": 0.048, "exp": 1.68},
}


# =============================================================================
# Discrete component package stress
# IEC TR 62380, Section 9-10
# =============================================================================

DISCRETE_PACKAGE_TABLE = {
    "TO-92": {"lb": 1.0},
    "TO-126": {"lb": 3.2},
    "TO-220": {"lb": 5.7},
    "TO-247": {"lb": 8.0},
    "TO-3": {"lb": 12.0},
    "DO-41": {"lb": 2.5},
    "DO-35": {"lb": 1.5},
    "DO-201": {"lb": 4.0},
    "SOT-23": {"lb": 1.0},
    "SOT-89": {"lb": 2.0},
    "SOT-223": {"lb": 3.4},
    "SOD-123": {"lb": 1.0},
    "SOD-323": {"lb": 0.8},
    "SOD-523": {"lb": 0.6},
    "SMA": {"lb": 1.8},
    "SMB": {"lb": 2.2},
    "SMC": {"lb": 3.0},
    "D-PAK": {"lb": 5.0},
    "D2-PAK": {"lb": 6.5},
    "PowerPAK": {"lb": 4.0},
    "01005": {"lb": 0.3},
    "0201": {"lb": 0.4},
    "0402": {"lb": 0.5},
    "0603": {"lb": 0.6},
    "0805": {"lb": 0.8},
    "1206": {"lb": 1.0},
    "1210": {"lb": 1.2},
    "1812": {"lb": 1.5},
    "2010": {"lb": 1.6},
    "2512": {"lb": 2.0},
}


# =============================================================================
# Diode base failure rates (FIT) -- IEC TR 62380, Section 9
# =============================================================================

DIODE_BASE_RATES = {
    "Signal (<1A)": {"l0": 0.07, "ea": ActivationEnergy.BIPOLAR},
    "Rectifier (1-3A)": {"l0": 0.10, "ea": ActivationEnergy.BIPOLAR},
    "Power Rectifier (>3A)": {"l0": 0.25, "ea": ActivationEnergy.BIPOLAR},
    "Zener": {"l0": 0.40, "ea": ActivationEnergy.BIPOLAR},
    "TVS": {"l0": 2.30, "ea": ActivationEnergy.BIPOLAR},
    "Schottky (<3A)": {"l0": 0.15, "ea": ActivationEnergy.MOS},
    "Schottky (>=3A)": {"l0": 0.30, "ea": ActivationEnergy.MOS},
    "LED": {"l0": 0.50, "ea": ActivationEnergy.MOS},
    "LED (High Power)": {"l0": 1.20, "ea": ActivationEnergy.MOS},
    "Varicap": {"l0": 0.20, "ea": ActivationEnergy.BIPOLAR},
}

TRANSISTOR_BASE_RATES = {
    "Silicon BJT (<=5W)": {"l0": 0.75, "tech": "bipolar"},
    "Silicon BJT (>5W)": {"l0": 2.0, "tech": "bipolar"},
    "Silicon MOSFET (<=5W)": {"l0": 0.75, "tech": "mos"},
    "Silicon MOSFET (>5W)": {"l0": 2.0, "tech": "mos"},
    "IGBT": {"l0": 2.5, "tech": "bipolar"},
    "JFET": {"l0": 0.50, "tech": "mos"},
    "GaN HEMT": {"l0": 3.0, "tech": "gan"},
    "SiC MOSFET": {"l0": 3.5, "tech": "sic"},
}

OPTOCOUPLER_BASE_RATES = {
    "Phototransistor Output": {"l0": 1.5, "ea": ActivationEnergy.OPTOCOUPLER},
    "Photodarlington Output": {"l0": 2.0, "ea": ActivationEnergy.OPTOCOUPLER},
    "Photo-TRIAC Output": {"l0": 2.5, "ea": ActivationEnergy.OPTOCOUPLER},
    "High-Speed (Logic)": {"l0": 1.8, "ea": ActivationEnergy.OPTOCOUPLER},
}

THYRISTOR_BASE_RATES = {
    "SCR (<=5A)": {"l0": 1.0, "ea": ActivationEnergy.BIPOLAR},
    "SCR (>5A)": {"l0": 2.5, "ea": ActivationEnergy.BIPOLAR},
    "TRIAC (<=5A)": {"l0": 1.5, "ea": ActivationEnergy.BIPOLAR},
    "TRIAC (>5A)": {"l0": 3.0, "ea": ActivationEnergy.BIPOLAR},
}


# =============================================================================
# Passive component parameters -- IEC TR 62380, Sections 11-13
# =============================================================================

CAPACITOR_PARAMS = {
    "Ceramic Class I (C0G)": {
        "l0": 0.05,
        "pkg_coef": 3.3e-3,
        "ea": ActivationEnergy.CAPACITOR_LOW,
        "t_ref": 303,
        "v_exp": 2.5,
    },
    "Ceramic Class II (X7R/X5R)": {
        "l0": 0.15,
        "pkg_coef": 3.3e-3,
        "ea": ActivationEnergy.CAPACITOR_LOW,
        "t_ref": 303,
        "v_exp": 2.5,
    },
    "Ceramic Class III (Y5V/Z5U)": {
        "l0": 0.25,
        "pkg_coef": 3.3e-3,
        "ea": ActivationEnergy.CAPACITOR_LOW,
        "t_ref": 303,
        "v_exp": 2.5,
    },
    "Tantalum Solid": {
        "l0": 0.40,
        "pkg_coef": 3.8e-3,
        "ea": ActivationEnergy.CAPACITOR_MED,
        "t_ref": 303,
        "v_exp": 3.0,
    },
    "Tantalum Wet": {
        "l0": 0.35,
        "pkg_coef": 3.5e-3,
        "ea": ActivationEnergy.CAPACITOR_MED,
        "t_ref": 303,
        "v_exp": 3.0,
    },
    "Aluminum Electrolytic": {
        "l0": 1.30,
        "pkg_coef": 1.4e-3,
        "ea": ActivationEnergy.ALUMINUM_CAP,
        "t_ref": 313,
        "v_exp": 5.0,
    },
    "Aluminum Polymer": {
        "l0": 0.90,
        "pkg_coef": 1.4e-3,
        "ea": ActivationEnergy.ALUMINUM_CAP,
        "t_ref": 313,
        "v_exp": 4.0,
    },
    "Film (Polyester / PET)": {
        "l0": 0.20,
        "pkg_coef": 2.0e-3,
        "ea": ActivationEnergy.CAPACITOR_HIGH,
        "t_ref": 303,
        "v_exp": 2.0,
    },
    "Film (Polypropylene / PP)": {
        "l0": 0.10,
        "pkg_coef": 2.0e-3,
        "ea": ActivationEnergy.CAPACITOR_HIGH,
        "t_ref": 303,
        "v_exp": 2.0,
    },
    "MLCC (High Voltage >100V)": {
        "l0": 0.30,
        "pkg_coef": 4.0e-3,
        "ea": ActivationEnergy.CAPACITOR_LOW,
        "t_ref": 303,
        "v_exp": 3.0,
    },
}

RESISTOR_PARAMS = {
    "SMD Chip Resistor": {"l0": 0.01, "pkg_coef": 3.3e-3, "temp_coef": 55},
    "SMD Chip (Anti-surge)": {"l0": 0.015, "pkg_coef": 3.3e-3, "temp_coef": 55},
    "Film (Low Power)": {"l0": 0.10, "pkg_coef": 1.4e-3, "temp_coef": 85},
    "Thin Film Precision": {"l0": 0.05, "pkg_coef": 3.3e-3, "temp_coef": 50},
    "Wirewound (Power)": {"l0": 0.20, "pkg_coef": 2.0e-3, "temp_coef": 100},
    "Wirewound (Precision)": {"l0": 0.10, "pkg_coef": 1.5e-3, "temp_coef": 80},
    "Resistor Network (4)": {"l0": 0.04, "pkg_coef": 3.3e-3, "temp_coef": 55},
    "Resistor Network (8)": {"l0": 0.08, "pkg_coef": 3.3e-3, "temp_coef": 55},
    "Potentiometer (Cermet)": {"l0": 1.00, "pkg_coef": 5.0e-3, "temp_coef": 70},
    "Potentiometer (Carbon)": {"l0": 2.00, "pkg_coef": 5.0e-3, "temp_coef": 90},
}

INDUCTOR_PARAMS = {
    "Power Inductor": {"l0": 0.6},
    "Signal Inductor": {"l0": 0.3},
    "Common Mode Choke": {"l0": 0.8},
    "Signal Transformer": {"l0": 1.5},
    "Power Transformer": {"l0": 3.0},
    "Power Transformer (>50W)": {"l0": 5.0},
}

RELAY_PARAMS = {
    "Signal Relay (Reed)": {"l0": 5.0, "mech_coef": 0.01},
    "Signal Relay (Electromech)": {"l0": 8.0, "mech_coef": 0.02},
    "Power Relay (<=10A)": {"l0": 15.0, "mech_coef": 0.03},
    "Power Relay (>10A)": {"l0": 25.0, "mech_coef": 0.05},
    "Solid State Relay": {"l0": 3.0, "mech_coef": 0.0},
    "Contactor": {"l0": 40.0, "mech_coef": 0.08},
}

CONNECTOR_PARAMS = {
    "Header/Pin (male)": {"l0_pin": 0.50, "l_housing": 1.0, "pkg_coef": 2.0e-3},
    "Socket (female)": {"l0_pin": 0.60, "l_housing": 1.2, "pkg_coef": 2.5e-3},
    "Card Edge": {"l0_pin": 0.70, "l_housing": 2.0, "pkg_coef": 3.0e-3},
    "D-Sub": {"l0_pin": 0.40, "l_housing": 1.5, "pkg_coef": 2.0e-3},
    "Circular (MIL-spec)": {"l0_pin": 0.30, "l_housing": 2.0, "pkg_coef": 1.5e-3},
    "RF / Coaxial (SMA/BNC)": {"l0_pin": 1.00, "l_housing": 3.0, "pkg_coef": 3.0e-3},
    "USB / HDMI / RJ45": {"l0_pin": 0.80, "l_housing": 2.5, "pkg_coef": 3.0e-3},
    "FFC/FPC": {"l0_pin": 0.50, "l_housing": 1.5, "pkg_coef": 2.0e-3},
    "Wire-to-Board": {"l0_pin": 0.55, "l_housing": 1.0, "pkg_coef": 2.0e-3},
}

PCB_SOLDER_PARAMS = {
    "PTH Solder Joint": {"l0": 0.002},
    "SMD Solder Joint": {"l0": 0.005},
    "BGA Solder Ball": {"l0": 0.008},
    "Via (PTH)": {"l0": 0.001},
    "Via (Microvia)": {"l0": 0.003},
    "PCB (per layer)": {"l0": 0.05},
}

MISC_COMPONENT_RATES = {
    "Crystal Oscillator (XO)": 10.0,
    "TCXO/VCXO": 15.0,
    "OCXO": 25.0,
    "MEMS Oscillator": 8.0,
    "SAW Filter": 5.0,
    "DC-DC Converter (<10W)": 100.0,
    "DC-DC Converter (>=10W)": 130.0,
    "DC-DC Converter (>=50W)": 180.0,
    "Fuse (Cartridge)": 2.0,
    "Fuse (PTC Resettable)": 5.0,
    "Varistor (MOV)": 3.0,
    "Ferrite Bead": 0.5,
    "EMI Filter": 8.0,
    "Battery Holder": 5.0,
    "Switch (Pushbutton)": 10.0,
    "Switch (DIP/Rotary)": 15.0,
    "Heatsink (passive)": 0.1,
    "Fan (active cooling)": 50.0,
}


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
    if diff < 1e-9:
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


def calculate_ic_lambda3(pkg_type: str, pins: int = None, diag: float = None) -> float:
    """IC package stress contribution (lambda_3) in FIT. IEC TR 62380, Table 17a/17b."""
    pkg = IC_PACKAGE_TABLE.get(pkg_type)
    if not pkg:
        return 4.0
    formula = pkg.get("formula", "fixed")
    if formula == "fixed":
        return _safe_float(pkg.get("value", 4.0))
    elif formula == "linear" and pins and pins > 0:
        return _safe_float(pkg.get("offset", 0.0)) + _safe_float(pkg.get("coef", 0.09)) * pins
    elif formula == "pins" and pins and pins > 0:
        return _safe_float(pkg.get("coef", 0.01)) * (
            pins ** _safe_float(pkg.get("exp", 1.5))
        )
    elif formula == "diagonal" and diag and diag > 0:
        return _safe_float(pkg.get("coef", 0.05)) * (
            diag ** _safe_float(pkg.get("exp", 1.68))
        )
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
    **kw,
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
    t_ref = dp.get("t_ref", 328)           # 328 K for Si, 373 K for GaAs
    n_per = _safe_float(dp.get("n_per", 1), 1.0)

    # Effective transistor count (user enters in the n_unit described by the type)
    effective_n = transistor_count * n_per

    a = max(0, construction_year - 1998)
    pi_t = pi_temperature(t_junction, ea, t_ref)
    lambda_die = (l1 * effective_n * math.exp(-0.35 * a) + l2) * pi_t * tau_on

    l3 = calculate_ic_lambda3(package_type, pins)
    pi_a = pi_alpha(substrate_alpha, package_alpha)
    pi_n = pi_thermal_cycles(n_cycles)
    lambda_pkg = 2.75e-3 * pi_a * pi_n * (delta_t**0.68) * l3

    lambda_e = lambda_eos(is_interface, interface_type)
    total_fit = lambda_die + lambda_pkg + lambda_e
    return {
        "lambda_die": lambda_die * 1e-9,
        "lambda_package": lambda_pkg * 1e-9,
        "lambda_eos": lambda_e * 1e-9,
        "lambda_total": total_fit * 1e-9,
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
    v_applied=0.0,
    v_rated=0.0,
    **kw,
):
    """Diode failure rate per IEC TR 62380, Section 9."""
    tau_on = validate_ratio(tau_on, "tau_on")
    t_junction = validate_temperature(t_junction, "t_junction")
    n_cycles = max(0, _safe_int(n_cycles, 5256))
    delta_t = max(0.0, _safe_float(delta_t, 3.0))

    dr = DIODE_BASE_RATES.get(diode_type, DIODE_BASE_RATES["Signal (<1A)"])
    l0, ea = dr["l0"], dr.get("ea", ActivationEnergy.BIPOLAR)
    pi_t = pi_temperature(t_junction, ea, 313)

    pi_v = 1.0
    if _safe_float(v_rated) > 0 and _safe_float(v_applied) > 0:
        pi_v = pi_voltage_stress(v_applied, v_rated, 2.0)

    lambda_die = l0 * pi_t * pi_v * tau_on
    lb = DISCRETE_PACKAGE_TABLE.get(package, {"lb": 1.0}).get("lb", 1.0)
    pi_n = pi_thermal_cycles(n_cycles)
    lambda_pkg = 2.75e-3 * pi_n * (delta_t**0.68) * lb
    lambda_e = lambda_eos(is_interface, interface_type)
    total_fit = lambda_die + lambda_pkg + lambda_e
    return {
        "lambda_die": lambda_die * 1e-9,
        "lambda_package": lambda_pkg * 1e-9,
        "lambda_eos": lambda_e * 1e-9,
        "lambda_total": total_fit * 1e-9,
        "fit_total": total_fit,
        "pi_t": pi_t,
        "pi_v": pi_v,
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
    **kw,
):
    """Transistor failure rate per IEC TR 62380, Section 10."""
    tau_on = validate_ratio(tau_on, "tau_on")
    t_junction = validate_temperature(t_junction, "t_junction")
    n_cycles = max(0, _safe_int(n_cycles, 5256))
    delta_t = max(0.0, _safe_float(delta_t, 3.0))

    p = TRANSISTOR_BASE_RATES.get(
        transistor_type, TRANSISTOR_BASE_RATES["Silicon MOSFET (<=5W)"]
    )
    l0, tech = p["l0"], p["tech"]

    ea_map = {
        "bipolar": ActivationEnergy.BIPOLAR,
        "mos": ActivationEnergy.MOS,
        "gan": ActivationEnergy.GAN,
        "sic": ActivationEnergy.SIC,
    }
    ea = ea_map.get(tech, ActivationEnergy.MOS)
    pi_t = pi_temperature(t_junction, ea, 373)

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
        "lambda_die": lambda_die * 1e-9,
        "lambda_package": lambda_pkg * 1e-9,
        "lambda_eos": lambda_e * 1e-9,
        "lambda_total": total_fit * 1e-9,
        "fit_total": total_fit,
        "pi_s": pi_s,
        "pi_t": pi_t,
    }


def lambda_optocoupler(
    optocoupler_type="Phototransistor Output",
    t_junction=85.0,
    package="DIP-8",
    if_applied=10.0,
    if_rated=60.0,
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,
):
    """Optocoupler failure rate per IEC TR 62380, Section 10.3."""
    tau_on = validate_ratio(tau_on, "tau_on")
    t_junction = validate_temperature(t_junction, "t_junction")

    p = OPTOCOUPLER_BASE_RATES.get(
        optocoupler_type, OPTOCOUPLER_BASE_RATES["Phototransistor Output"]
    )
    l0, ea = p["l0"], p.get("ea", ActivationEnergy.OPTOCOUPLER)
    pi_t = pi_temperature(t_junction, ea, 313)

    if_applied = _safe_float(if_applied, 10.0)
    if_rated = _safe_float(if_rated, 60.0)
    pi_if = 1.0
    if if_rated > 0:
        pi_if = (if_applied / if_rated) ** 2.0

    lambda_die = l0 * pi_t * pi_if * tau_on
    pkg_info = IC_PACKAGE_TABLE.get(
        "DIP", {"formula": "pins", "coef": 0.014, "exp": 1.20}
    )
    l3 = _safe_float(pkg_info.get("coef", 0.014)) * (
        8 ** _safe_float(pkg_info.get("exp", 1.2))
    )
    pi_n = pi_thermal_cycles(n_cycles)
    lambda_pkg = 2.75e-3 * pi_n * (delta_t**0.68) * l3
    total_fit = lambda_die + lambda_pkg
    return {
        "lambda_die": lambda_die * 1e-9,
        "lambda_package": lambda_pkg * 1e-9,
        "lambda_total": total_fit * 1e-9,
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
    **kw,
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
        "lambda_die": lambda_die * 1e-9,
        "lambda_package": lambda_pkg * 1e-9,
        "lambda_total": total_fit * 1e-9,
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
    **kw,
):
    """Capacitor failure rate per IEC TR 62380, Section 11."""
    tau_on = validate_ratio(tau_on, "tau_on")
    t_ambient = validate_temperature(t_ambient, "t_ambient")
    ripple_ratio = validate_ratio(ripple_ratio, "ripple_ratio")
    n_cycles = max(0, _safe_int(n_cycles, 5256))
    delta_t = max(0.0, _safe_float(delta_t, 3.0))

    p = CAPACITOR_PARAMS.get(
        capacitor_type, CAPACITOR_PARAMS["Ceramic Class II (X7R/X5R)"]
    )
    l0, pkg_coef, ea, t_ref = p["l0"], p["pkg_coef"], p["ea"], p["t_ref"]
    v_exp = p.get("v_exp", 2.5)

    if "Aluminum" in capacitor_type and ripple_ratio > 0:
        t_op = t_ambient + 20.0 * (ripple_ratio**2)
    else:
        t_op = t_ambient

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
        "lambda_base": lambda_base * 1e-9,
        "lambda_package": lambda_pkg * 1e-9,
        "lambda_total": total_fit * 1e-9,
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
    **kw,
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
        "lambda_base": lambda_base * 1e-9,
        "lambda_package": lambda_pkg * 1e-9,
        "lambda_total": total_fit * 1e-9,
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
    **kw,
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
        "lambda_base": lambda_base * 1e-9,
        "lambda_package": lambda_pkg * 1e-9,
        "lambda_total": total_fit * 1e-9,
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
    **kw,
):
    """Relay failure rate per IEC TR 62380, Section 14."""
    tau_on = validate_ratio(tau_on, "tau_on")
    t_ambient = validate_temperature(t_ambient, "t_ambient")
    cycles_per_hour = _safe_float(cycles_per_hour, 0.0)
    contact_current_ratio = validate_ratio(
        contact_current_ratio, "contact_current_ratio"
    )

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
        "lambda_electrical": lambda_elec * 1e-9,
        "lambda_mechanical": lambda_mech * 1e-9,
        "lambda_package": lambda_pkg * 1e-9,
        "lambda_total": total_fit * 1e-9,
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
    **kw,
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
        "lambda_contacts": lambda_contacts * 1e-9,
        "lambda_housing": lambda_housing * 1e-9,
        "lambda_thermal": lambda_thermal * 1e-9,
        "lambda_mating": lambda_mating * 1e-9,
        "lambda_total": total_fit * 1e-9,
        "fit_total": total_fit,
        "n_contacts": n_contacts,
    }


def lambda_pcb_solder(
    joint_type="SMD Solder Joint", n_joints=100, n_cycles=5256, delta_t=3.0, **kw
):
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
        "lambda_base": lambda_base * 1e-9,
        "lambda_thermal": lambda_thermal * 1e-9,
        "lambda_total": total_fit * 1e-9,
        "fit_total": total_fit,
    }


def lambda_misc_component(
    component_type="Crystal Oscillator (XO)",
    n_contacts=1,
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0,
    **kw,
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
    return {"lambda_total": total_fit * 1e-9, "fit_total": total_fit, "base_fit": base}


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
        return sum(
            math.comb(n, i) * (r**i) * ((1.0 - r) ** (n - i)) for i in range(k, n + 1)
        )
    r_last = _safe_float(r_list[-1], 1.0)
    return r_last * r_k_of_n(r_list[:-1], k - 1) + (1.0 - r_last) * r_k_of_n(
        r_list[:-1], k
    )


def lambda_series(lam_list):
    """Series system total failure rate = sum."""
    return sum(_safe_float(l, 0.0) for l in lam_list)


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


def get_field_definitions(ct):
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

    if ct == "Integrated Circuit":
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
    elif ct == "Diode":
        return {
            "diode_type": {
                "type": "choice",
                "choices": list(DIODE_BASE_RATES.keys()),
                "default": "Signal (<1A)",
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
                "default": "SOD-123",
            },
            "v_applied": {
                "type": "float",
                "default": 0.0,
                "help": "Applied reverse voltage (V)",
            },
            "v_rated": {
                "type": "float",
                "default": 0.0,
                "help": "Rated reverse voltage (V)",
            },
            **iface,
            **common,
        }
    elif ct == "Transistor":
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
    elif ct == "Optocoupler":
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
    elif ct == "Thyristor/TRIAC":
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
    elif ct == "Capacitor":
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
    elif ct == "Resistor":
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
    elif ct == "Inductor/Transformer":
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
    elif ct == "Relay":
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
    elif ct == "Connector":
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
    elif ct == "PCB/Solder":
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


def calculate_component_lambda(ct, params):
    """Calculate failure rate for a component given type and parameters."""
    try:
        if ct == "Integrated Circuit":
            ic_key = IC_TYPE_CHOICES.get(
                params.get("ic_type", "MOS Digital (Micro/DSP)"), "MOS_DIGITAL"
            )
            pkg = params.get("package", "QFP-48 (7x7mm)")
            pkg_info = IC_PACKAGE_CHOICES.get(pkg, ("PQFP-7x7", 48))
            sub = THERMAL_EXPANSION_SUBSTRATE.get(
                params.get("substrate", "FR4 / Epoxy Glass (G-10)"), 16.0
            )
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
        elif ct == "Diode":
            return lambda_diode(
                **{k: v for k, v in params.items() if not k.startswith("_")}
            )
        elif ct == "Transistor":
            return lambda_transistor(
                **{k: v for k, v in params.items() if not k.startswith("_")}
            )
        elif ct == "Optocoupler":
            return lambda_optocoupler(
                **{k: v for k, v in params.items() if not k.startswith("_")}
            )
        elif ct == "Thyristor/TRIAC":
            return lambda_thyristor(
                **{k: v for k, v in params.items() if not k.startswith("_")}
            )
        elif ct == "Capacitor":
            return lambda_capacitor(
                **{k: v for k, v in params.items() if not k.startswith("_")}
            )
        elif ct == "Resistor":
            return lambda_resistor(
                **{k: v for k, v in params.items() if not k.startswith("_")}
            )
        elif ct == "Inductor/Transformer":
            return lambda_inductor(
                **{k: v for k, v in params.items() if not k.startswith("_")}
            )
        elif ct == "Relay":
            return lambda_relay(
                **{k: v for k, v in params.items() if not k.startswith("_")}
            )
        elif ct == "Connector":
            return lambda_connector(
                **{k: v for k, v in params.items() if not k.startswith("_")}
            )
        elif ct == "PCB/Solder":
            return lambda_pcb_solder(
                **{k: v for k, v in params.items() if not k.startswith("_")}
            )
        else:
            return lambda_misc_component(
                component_type=params.get(
                    "component_subtype", "Crystal Oscillator (XO)"
                ),
                **{
                    k: v
                    for k, v in params.items()
                    if not k.startswith("_") and k != "component_subtype"
                },
            )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"lambda_total": 10e-9, "fit_total": 10.0, "_error": str(e)}


def calculate_lambda(cls, params=None):
    """Simplified interface: returns lambda_total from a class name string."""
    if params is None:
        params = {}
    c = cls.lower()
    try:
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
        if "inductor" in c or "transformer" in c:
            return lambda_inductor(**params)["lambda_total"]
        if "optocoupler" in c or "opto" in c:
            return lambda_optocoupler(**params)["lambda_total"]
        if "thyristor" in c or "triac" in c or "scr" in c:
            return lambda_thyristor(**params)["lambda_total"]
        if "relay" in c:
            return lambda_relay(**params)["lambda_total"]
        if "connector" in c:
            return lambda_connector(**params)["lambda_total"]
    except Exception:
        pass
    return 10e-9


# =============================================================================
# Formatting utilities
# =============================================================================


def fit_to_lambda(fit):
    return _safe_float(fit) * 1e-9


def lambda_to_fit(lam):
    return _safe_float(lam) * 1e9


def format_lambda(lam, as_fit=True):
    lam = _safe_float(lam)
    return f"{lam * 1e9:.2f} FIT" if as_fit else f"{lam:.2e} /h"


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


def analyze_component_criticality(ct, params, mission_hours, perturbation=0.10):
    """Analyze which parameter fields most influence a component's failure rate."""
    try:
        nominal = calculate_component_lambda(ct, params)
        lam_nominal = nominal.get("lambda_total", 0.0)
    except Exception:
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
            lam_low = calculate_component_lambda(ct, params_low).get(
                "lambda_total", 0.0
            )
            lam_high = calculate_component_lambda(ct, params_high).get(
                "lambda_total", 0.0
            )
        except Exception:
            continue

        if lam_nominal > 0:
            sensitivity = (lam_high - lam_low) / (2.0 * dp / v) / lam_nominal
        else:
            sensitivity = 0.0

        results.append(
            {
                "field": field,
                "nominal_value": v,
                "sensitivity": sensitivity,
                "lambda_low_fit": lam_low * 1e9,
                "lambda_high_fit": lam_high * 1e9,
                "lambda_nominal_fit": lam_nominal * 1e9,
                "impact_percent": (
                    abs(lam_high - lam_low) / lam_nominal * 100
                    if lam_nominal > 0
                    else 0
                ),
            }
        )

    results.sort(key=lambda x: -abs(x["sensitivity"]))
    return results


# =============================================================================
# Legacy aliases for backward compatibility
# =============================================================================

reliability = reliability_from_lambda
component_failure_rate = lambda c, p=None: calculate_lambda(
    c, p.to_dict() if p and hasattr(p, "to_dict") else (p or {})
)
