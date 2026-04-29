"""
IEC TR 62380:2004 — Physical & data constants
==============================================
Module-level constants, activation energies, package / die / component tables.

Re-exported by ``plugins.reliability_math`` for backward compatibility.
"""

from __future__ import annotations

from enum import Enum

__version__ = "3.3.0"
__author__ = "Eliot Abramo"

# =============================================================================
# Physical & conversion constants
# =============================================================================

# FIT-to-lambda conversion: 1 FIT = 1 failure per 10^9 hours
FIT_PER_LAMBDA: float = 1e9
LAMBDA_PER_FIT: float = 1e-9

# Standard reference temperatures in Kelvin
T_REF_KELVIN_25C: float = 298.0
T_REF_KELVIN_55C: float = 328.0
T_REF_KELVIN_85C: float = 358.0

# Absolute zero in Celsius
ABSOLUTE_ZERO_C: float = -273.15

# Hours in a standard year (365 days)
HOURS_PER_YEAR: float = 8760.0

# Default mission & environmental defaults
DEFAULT_T_AMBIENT_C: float = 25.0
DEFAULT_N_CYCLES: int = 5256
DEFAULT_DELTA_T_C: float = 3.0
DEFAULT_TAU_ON: float = 1.0

# Numerical tolerance for floating-point comparisons
FLOAT_EPSILON: float = 1e-9


# =============================================================================
# System topology types
# =============================================================================
class ConnectionType(str, Enum):
    """Block-diagram connection modes for reliability modelling."""

    SERIES = "series"
    PARALLEL = "parallel"
    K_OF_N = "k_of_n"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> ConnectionType:
        if not value:
            return cls.SERIES
        return super()._missing_(value)


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
        "l1": 3.4e-6,
        "l2": 1.7,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "gates",
        "n_per": 4,
        "desc": "Digital circuits, Micros, DSP",
    },
    "MOS_LINEAR": {
        "l1": 1.0e-2,
        "l2": 4.2,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "transistors",
        "n_per": 1,
        "desc": "Linear circuits (MOS)",
    },
    "MOS_DIG_LIN_TELECOM": {
        "l1": 2.7e-4,
        "l2": 20.0,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "transistors",
        "n_per": 1,
        "desc": "Digital/linear (Telecom, CAN, CNA, RAMDAC)",
    },
    "MOS_ROM": {
        "l1": 1.7e-7,
        "l2": 8.8,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "bits",
        "n_per": 1,
        "desc": "ROM - Read only memory",
    },
    "MOS_DRAM": {
        "l1": 1.0e-7,
        "l2": 5.6,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "bits",
        "n_per": 1,
        "desc": "DRAM / VideoRAM / AudioRAM",
    },
    "MOS_SRAM_FAST": {
        "l1": 1.7e-7,
        "l2": 8.8,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "bits",
        "n_per": 4,
        "desc": "High-speed SRAM, FIFO (mixed MOS)",
    },
    "MOS_SRAM_LOW": {
        "l1": 1.7e-7,
        "l2": 8.8,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "bits",
        "n_per": 6,
        "desc": "Low-consumption SRAM (CMOS)",
    },
    "MOS_SRAM_DUAL": {
        "l1": 1.7e-7,
        "l2": 8.8,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "bits",
        "n_per": 8,
        "desc": "Dual-access Static RAM",
    },
    "MOS_EPROM": {
        "l1": 2.6e-7,
        "l2": 34.0,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "prog. points",
        "n_per": 1,
        "desc": "EPROM / UVPROM / REPROM / OTP / FLASH (block erase)",
    },
    "MOS_EEPROM": {
        "l1": 6.5e-7,
        "l2": 16.0,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "prog. points",
        "n_per": 2,
        "desc": "EEPROM / Flash EEPROM (word-erasable)",
    },
    # --- Silicon: MOS ASIC circuits ---
    "MOS_ASIC_STDCELL": {
        "l1": 1.2e-5,
        "l2": 10.0,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "gates",
        "n_per": 4,
        "desc": "ASIC Standard Cell / Full Custom",
    },
    "MOS_ASIC_GATE_ARRAY": {
        "l1": 2.0e-5,
        "l2": 10.0,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "gates",
        "n_per": 4,
        "desc": "ASIC Gate Arrays",
    },
    "MOS_LCA": {
        "l1": 4.0e-5,
        "l2": 8.8,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "gates",
        "n_per": 40,
        "desc": "LCA / FPGA (RAM-based, ext. memory configured)",
    },
    "MOS_PLD": {
        "l1": 1.2e-3,
        "l2": 16.0,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "grid points",
        "n_per": 3,
        "desc": "PLD (GAL, PAL) - AND/OR array",
    },
    "MOS_CPLD": {
        "l1": 2.0e-5,
        "l2": 34.0,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "macrocells",
        "n_per": 100,
        "desc": "CPLD / EPLD / MAX / FLEX / FPGA (flash/antifuse)",
    },
    # --- Silicon: Bipolar circuits (1) ---
    "BIPOLAR_DIGITAL": {
        "l1": 6.0e-4,
        "l2": 1.7,
        "ea": ActivationEnergy.BIPOLAR,
        "t_ref": 328,
        "n_unit": "gates",
        "n_per": 3,
        "desc": "Bipolar digital circuits",
    },
    "BIPOLAR_LINEAR": {
        "l1": 2.2e-2,
        "l2": 3.3,
        "ea": ActivationEnergy.BIPOLAR,
        "t_ref": 328,
        "n_unit": "transistors",
        "n_per": 1,
        "desc": "Bipolar linear circuits (FET, others)",
    },
    "BIPOLAR_MMIC": {
        "l1": 1.0,
        "l2": 3.3,
        "ea": ActivationEnergy.BIPOLAR,
        "t_ref": 328,
        "n_unit": "transistors",
        "n_per": 1,
        "desc": "Bipolar MMIC",
    },
    "BIPOLAR_LOW_V": {
        "l1": 2.7e-3,
        "l2": 20.0,
        "ea": ActivationEnergy.BIPOLAR,
        "t_ref": 328,
        "n_unit": "transistors",
        "n_per": 1,
        "desc": "Bipolar linear/digital, low voltage (<30V)",
    },
    "BIPOLAR_HIGH_V": {
        "l1": 2.7e-2,
        "l2": 20.0,
        "ea": ActivationEnergy.BIPOLAR,
        "t_ref": 328,
        "n_unit": "transistors",
        "n_per": 1,
        "desc": "Bipolar linear/digital, high voltage (>=30V)",
    },
    "BIPOLAR_SRAM": {
        "l1": 3.0e-4,
        "l2": 1.7,
        "ea": ActivationEnergy.BIPOLAR,
        "t_ref": 328,
        "n_unit": "bits",
        "n_per": 2.5,
        "desc": "Bipolar static read-access memories",
    },
    "BIPOLAR_PROM": {
        "l1": 1.5e-4,
        "l2": 32.0,
        "ea": ActivationEnergy.BIPOLAR,
        "t_ref": 328,
        "n_unit": "prog. points",
        "n_per": 1.2,
        "desc": "Bipolar programmable read-only memory",
    },
    "BIPOLAR_PLA": {
        "l1": 1.5e-4,
        "l2": 32.0,
        "ea": ActivationEnergy.BIPOLAR,
        "t_ref": 328,
        "n_unit": "grid points",
        "n_per": 1.6,
        "desc": "Bipolar one-time prog. logic array (AND/OR)",
    },
    "BIPOLAR_GATE_ARRAY": {
        "l1": 1.0e-3,
        "l2": 10.0,
        "ea": ActivationEnergy.BIPOLAR,
        "t_ref": 328,
        "n_unit": "gates",
        "n_per": 3,
        "desc": "Bipolar gate arrays (PROM, PLD/PAL)",
    },
    # --- Silicon: BiCMOS (Bipolar and MOS circuits) ---
    "BICMOS_DIGITAL": {
        "l1": 1.0e-6,
        "l2": 1.7,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "gates",
        "n_per": 4,
        "desc": "BiCMOS digital circuits",
    },
    "BICMOS_LOW_V": {
        "l1": 2.7e-4,
        "l2": 20.0,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "transistors",
        "n_per": 1,
        "desc": "BiCMOS linear/digital, low voltage (<6V)",
    },
    "BICMOS_HIGH_V": {
        "l1": 2.7e-3,
        "l2": 20.0,
        "ea": ActivationEnergy.BIPOLAR,
        "t_ref": 328,
        "n_unit": "transistors",
        "n_per": 1,
        "desc": "BiCMOS linear/digital, high voltage (>=6V) / Smart Power",
    },
    "BICMOS_SRAM": {
        "l1": 6.8e-7,
        "l2": 8.8,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "bits",
        "n_per": 4,
        "desc": "BiCMOS Static Read Access Memory",
    },
    "BICMOS_GATE_ARRAY": {
        "l1": 6.4e-5,
        "l2": 10.0,
        "ea": ActivationEnergy.MOS,
        "t_ref": 328,
        "n_unit": "gates",
        "n_per": 4,
        "desc": "BiCMOS gate arrays",
    },
    # --- Gallium arsenide ---
    "GAAS_DIGITAL_NOR": {
        "l1": 2.5,
        "l2": 25.0,
        "ea": ActivationEnergy.MOS,
        "t_ref": 373,
        "n_unit": "gates",
        "n_per": 5,
        "desc": "GaAs digital - normally-on transistors only",
    },
    "GAAS_DIGITAL_MIXED": {
        "l1": 4.5e-4,
        "l2": 16.0,
        "ea": ActivationEnergy.BIPOLAR,
        "t_ref": 373,
        "n_unit": "gates",
        "n_per": 3,
        "desc": "GaAs digital - normally-off & normally-on",
    },
    "GAAS_MMIC_LOW": {
        "l1": 2.0,
        "l2": 20.0,
        "ea": ActivationEnergy.MOS,
        "t_ref": 373,
        "n_unit": "transistors",
        "n_per": 1,
        "desc": "GaAs MMIC - low noise / low power (<100mW)",
    },
    "GAAS_MMIC_POWER": {
        "l1": 4.0,
        "l2": 40.0,
        "ea": ActivationEnergy.BIPOLAR,
        "t_ref": 373,
        "n_unit": "transistors",
        "n_per": 1,
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
    # =========================================================================
    # IEC TR 62380, Table 18 -- Values of lambda_B and junction resistances
    # for active discrete components
    #
    # lambda_B (FIT) is the base failure rate of the package.
    # Rjc = junction-case thermal resistance (C/W)
    # Rja = junction-ambient thermal resistance (C/W)
    # Rja_mounted = junction-ambient for surface-mounted (C/W)
    #
    # Only lambda_B is used in the reliability model. Thermal resistances
    # are provided for junction temperature estimation (Section 8.1).
    # =========================================================================
    #
    # --- Through-hole power packages ---
    "TO-18": {"lb": 1.0, "rjc": 130, "rja": 450},
    "TO-39": {"lb": 2.0, "rjc": 35, "rja": 200},
    "TO-92": {"lb": 1.0, "rjc": 100, "rja": 300},
    "SOT-32 (TO-126)": {"lb": 5.3, "rjc": 10, "rja": 100},
    "SOT-82": {"lb": 5.3, "rjc": 10, "rja": 100},
    "TO-220": {"lb": 5.7, "rjc": 3},
    "TO-218 (SOT-93)": {"lb": 6.9, "rjc": 1.5},
    "TO-247": {"lb": 6.9, "rjc": 1},
    "ISOTOP": {"lb": 20.0, "rjc": 0.25},
    "DO-220": {"lb": 5.7, "rjc": 3},
    #
    # --- SMD transistor / small signal packages ---
    "SOT-23": {"lb": 1.0, "rja_mounted": 400},
    "SOT-143": {"lb": 1.0, "rja_mounted": 400},
    "SOT-223": {"lb": 3.4, "rja_mounted": 85},
    "SOT-323": {"lb": 0.8, "rja_mounted": 600},
    "SOT-343": {"lb": 0.8, "rja_mounted": 600},
    "SOT-346": {"lb": 1.0, "rja_mounted": 500},
    "SOT-363": {"lb": 0.8, "rja_mounted": 600},
    "SOT-457": {"lb": 1.1, "rja_mounted": 350},
    "SOT-89": {"lb": 2.0, "rja_mounted": 125},
    #
    # --- SMD power packages ---
    "DPACK (SOT-428)": {"lb": 5.1, "rja_mounted": 30},
    "D2PACK": {"lb": 5.7, "rja_mounted": 15},
    #
    # --- Optocoupler packages ---
    "SOT-90B (optocoupler)": {"lb": 4.1, "rja": 250},
    "SO-8 (optocoupler)": {"lb": 4.5, "rja_mounted": 300},
    #
    # --- Diode through-hole packages ---
    "DO-34 (DO-204AG)": {"lb": 2.5, "rja": 500},
    "DO-35 (DO-204AH)": {"lb": 2.5, "rja": 400},
    "DO-41 (DO-204AL) (glass)": {"lb": 2.5, "rja": 150},
    "DO-41 (DO-204AL) (plastic)": {"lb": 1.0, "rja": 100},
    "F 126": {"lb": 1.0, "rja": 70},
    #
    # --- Diode SMD packages (cylindrical) ---
    "micromelf": {"lb": 2.5, "rja_mounted": 600},
    "SOD-80 (minimelf)": {"lb": 2.5, "rja_mounted": 600},
    "melf": {"lb": 5.0, "rja_mounted": 450},
    #
    # --- Diode SMD packages (flat) ---
    "SOD-110": {"lb": 0.8, "rja_mounted": 350},
    "SOD-123": {"lb": 1.0, "rja_mounted": 600},
    "SOD-323": {"lb": 0.7, "rja_mounted": 600},
    "SOD-523": {"lb": 0.5, "rja_mounted": 100},
    "SMA": {"lb": 1.8, "rja_mounted": 600},
    "SMB (DO-214)": {"lb": 2.4, "rja_mounted": 75},
    "SMC (DO-215)": {"lb": 5.1, "rja_mounted": 25},
    "SOD-15": {"lb": 5.1, "rja_mounted": 20},
}


# =============================================================================
# Diode base failure rates (FIT) -- IEC TR 62380, Sections 8.2 and 8.3
# =============================================================================

DIODE_BASE_RATES = {
    # --- Section 8.2: Low-power diodes ---
    "Signal (<1A)": {
        "l0": 0.07,
        "ea": 4640,
        "t_ref": 313,
        "section": "8.2",
        "desc": "Silicon signal diodes, up to 1A",
    },
    "Recovery/Rectifier (1A-3A)": {
        "l0": 0.10,
        "ea": 4640,
        "t_ref": 313,
        "section": "8.2",
        "desc": "Silicon fast/slow recovery, rectifier, Schottky, 1A to 3A",
    },
    "Zener (<=1.5W)": {
        "l0": 0.40,
        "ea": 4640,
        "t_ref": 313,
        "section": "8.2",
        "desc": "Zener regulator diodes, up to 1.5W",
    },
    "TVS (low power)": {
        "l0": 2.30,
        "ea": 4640,
        "t_ref": 313,
        "section": "8.2",
        "desc": "Transient voltage suppressor, up to 5kW peak (10us/1000us)",
    },
    "Trigger TVS (low power)": {
        "l0": 2.00,
        "ea": 4640,
        "t_ref": 313,
        "section": "8.2",
        "desc": "Trigger transient voltage suppressor (low power)",
    },
    "GaAs (<= 0.1W)": {
        "l0": 0.30,
        "ea": 4640,
        "t_ref": 313,
        "section": "8.2",
        "desc": "Gallium arsenide diodes, up to 0.1W",
    },
    # --- Section 8.3: Power diodes ---
    "Recovery/Rectifier (>3A)": {
        "l0": 0.70,
        "ea": 4640,
        "t_ref": 313,
        "section": "8.3",
        "desc": "Silicon rectifier, fast recovery, Schottky, above 3A",
    },
    "Zener (>1.5W)": {
        "l0": 0.70,
        "ea": 4640,
        "t_ref": 313,
        "section": "8.3",
        "desc": "Zener regulator diodes, above 1.5W",
    },
    "TVS (power)": {
        "l0": 0.70,
        "ea": 4640,
        "t_ref": 313,
        "section": "8.3",
        "desc": "Transient voltage suppressor (power)",
    },
    "Trigger TVS (power)": {
        "l0": 3.00,
        "ea": 4640,
        "t_ref": 313,
        "section": "8.3",
        "desc": "Trigger transient voltage suppressor (power)",
    },
    "GaAs (>0.1W)": {
        "l0": 1.00,
        "ea": 4640,
        "t_ref": 313,
        "section": "8.3",
        "desc": "Gallium arsenide diodes, above 0.1W",
    },
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
