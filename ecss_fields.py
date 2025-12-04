"""
ECSS field definitions and metadata for the reliability GUI.

This module defines the list of fields a user should fill for each
component category, along with types, allowed values, and help strings.

The GUI can introspect this structure to build forms dynamically, so that
users do not need the ECSS document open while filling values.
"""

from __future__ import annotations

from typing import Dict, Any


# Field schema:
# {
#   "label": "Human readable name",
#   "type": "enum" | "float" | "int" | "string" | "bool",
#   "unit": "V" | "A" | "W" | "°C" | ... | None,
#   "values": [...],            # for enum
#   "default": any,
#   "help": "Explanation ...",
# }
#
# Category schema:
# {
#   "display_name": "Resistor",
#   "fields": { ... field schema ... }
# }


ECSS_FIELDS: Dict[str, Dict[str, Any]] = {
    "resistor": {
        "display_name": "Resistor",
        "fields": {
            "technology": {
                "label": "Technology",
                "type": "enum",
                "values": ["thin_film", "thick_film", "wirewound", "other"],
                "default": "thick_film",
                "unit": None,
                "help": "Construction / technology of the resistor as per ECSS tables.",
            },
            "power_rating_w": {
                "label": "Rated Power",
                "type": "float",
                "unit": "W",
                "default": 0.125,
                "help": "Rated power of the resistor used to compute stress ratio (P_applied / P_rated).",
            },
            "applied_power_w": {
                "label": "Applied Power",
                "type": "float",
                "unit": "W",
                "default": 0.05,
                "help": "Estimated power dissipated in the resistor at worst-case operation.",
            },
            "quality_level": {
                "label": "Quality Level",
                "type": "enum",
                "values": ["A", "B", "C", "D"],
                "default": "B",
                "unit": None,
                "help": "Quality / screening level according to ECSS component classes.",
            },
            "environment": {
                "label": "Environment",
                "type": "enum",
                "values": ["GB", "GF", "GM", "LA", "OR"],
                "default": "GB",
                "unit": None,
                "help": "Operating environment (Ground Benign, Ground Fixed, Ground Mobile, Launch, Orbit...).",
            },
            "temperature_c": {
                "label": "Operating Temperature",
                "type": "float",
                "unit": "°C",
                "default": 25.0,
                "help": "Expected operating temperature of the resistor.",
            },
        },
    },
    "capacitor_ceramic": {
        "display_name": "Ceramic Capacitor",
        "fields": {
            "dielectric_class": {
                "label": "Dielectric Class",
                "type": "enum",
                "values": ["class_1", "class_2"],
                "default": "class_2",
                "unit": None,
                "help": "Dielectric class as per ECSS (e.g. C0G/NP0 for class 1, X7R for class 2).",
            },
            "rated_voltage_v": {
                "label": "Rated Voltage",
                "type": "float",
                "unit": "V",
                "default": 50.0,
                "help": "Rated voltage of the capacitor.",
            },
            "applied_voltage_v": {
                "label": "Applied Voltage",
                "type": "float",
                "unit": "V",
                "default": 12.0,
                "help": "Maximum applied voltage during operation.",
            },
            "quality_level": {
                "label": "Quality Level",
                "type": "enum",
                "values": ["A", "B", "C", "D"],
                "default": "B",
                "unit": None,
                "help": "Quality / screening level according to ECSS component classes.",
            },
            "environment": {
                "label": "Environment",
                "type": "enum",
                "values": ["GB", "GF", "GM", "LA", "OR"],
                "default": "GB",
                "unit": None,
                "help": "Operating environment.",
            },
            "temperature_c": {
                "label": "Operating Temperature",
                "type": "float",
                "unit": "°C",
                "default": 25.0,
                "help": "Expected operating temperature of the capacitor.",
            },
        },
    },
    "capacitor_tantalum": {
        "display_name": "Tantalum Capacitor",
        "fields": {
            "construction": {
                "label": "Construction",
                "type": "enum",
                "values": ["solid", "wet"],
                "default": "solid",
                "unit": None,
                "help": "Type of tantalum capacitor as defined in ECSS tables.",
            },
            "rated_voltage_v": {
                "label": "Rated Voltage",
                "type": "float",
                "unit": "V",
                "default": 25.0,
                "help": "Rated voltage of the capacitor.",
            },
            "applied_voltage_v": {
                "label": "Applied Voltage",
                "type": "float",
                "unit": "V",
                "default": 5.0,
                "help": "Maximum applied voltage during operation.",
            },
            "quality_level": {
                "label": "Quality Level",
                "type": "enum",
                "values": ["A", "B", "C", "D"],
                "default": "B",
                "unit": None,
                "help": "Quality / screening level according to ECSS component classes.",
            },
            "environment": {
                "label": "Environment",
                "type": "enum",
                "values": ["GB", "GF", "GM", "LA", "OR"],
                "default": "GB",
                "unit": None,
                "help": "Operating environment.",
            },
            "temperature_c": {
                "label": "Operating Temperature",
                "type": "float",
                "unit": "°C",
                "default": 25.0,
                "help": "Expected operating temperature of the capacitor.",
            },
        },
    },
    "diode": {
        "display_name": "Diode",
        "fields": {
            "type": {
                "label": "Type",
                "type": "enum",
                "values": ["small_signal", "power", "zener"],
                "default": "small_signal",
                "unit": None,
                "help": "Diode category as used in the reliability tables.",
            },
            "rated_current_a": {
                "label": "Rated Current",
                "type": "float",
                "unit": "A",
                "default": 1.0,
                "help": "Rated forward current used to compute stress.",
            },
            "applied_current_a": {
                "label": "Applied Current",
                "type": "float",
                "unit": "A",
                "default": 0.2,
                "help": "Maximum operating forward current.",
            },
            "quality_level": {
                "label": "Quality Level",
                "type": "enum",
                "values": ["A", "B", "C", "D"],
                "default": "B",
                "unit": None,
                "help": "Quality / screening level according to ECSS component classes.",
            },
            "environment": {
                "label": "Environment",
                "type": "enum",
                "values": ["GB", "GF", "GM", "LA", "OR"],
                "default": "GB",
                "unit": None,
                "help": "Operating environment.",
            },
            "temperature_c": {
                "label": "Operating Temperature",
                "type": "float",
                "unit": "°C",
                "default": 25.0,
                "help": "Expected operating temperature of the diode junction.",
            },
        },
    },
    "ic_digital": {
        "display_name": "Digital IC",
        "fields": {
            "technology": {
                "label": "Technology",
                "type": "enum",
                "values": ["CMOS", "BiCMOS", "TTL", "other"],
                "default": "CMOS",
                "unit": None,
                "help": "Technology as defined by ECSS (CMOS, BiCMOS, etc.).",
            },
            "gate_count": {
                "label": "Gate Count",
                "type": "int",
                "unit": "gates",
                "default": 1000,
                "help": "Approximate complexity (gate count or equivalent).",
            },
            "quality_level": {
                "label": "Quality Level",
                "type": "enum",
                "values": ["A", "B", "C", "D"],
                "default": "B",
                "unit": None,
                "help": "Quality / screening level according to ECSS component classes.",
            },
            "environment": {
                "label": "Environment",
                "type": "enum",
                "values": ["GB", "GF", "GM", "LA", "OR"],
                "default": "GB",
                "unit": None,
                "help": "Operating environment.",
            },
            "junction_temperature_c": {
                "label": "Junction Temperature",
                "type": "float",
                "unit": "°C",
                "default": 55.0,
                "help": "Estimated junction temperature during operation.",
            },
        },
    },
    "connector": {
        "display_name": "Connector",
        "fields": {
            "n_pins": {
                "label": "Number of Pins",
                "type": "int",
                "unit": "pins",
                "default": 10,
                "help": "Total number of contacts in the connector.",
            },
            "mating_cycles": {
                "label": "Mating Cycles",
                "type": "int",
                "unit": "cycles",
                "default": 10,
                "help": "Number of planned mating cycles over the mission.",
            },
            "quality_level": {
                "label": "Quality Level",
                "type": "enum",
                "values": ["A", "B", "C", "D"],
                "default": "B",
                "unit": None,
                "help": "Quality / screening level according to ECSS component classes.",
            },
            "environment": {
                "label": "Environment",
                "type": "enum",
                "values": ["GB", "GF", "GM", "LA", "OR"],
                "default": "GB",
                "unit": None,
                "help": "Operating environment.",
            },
        },
    },
}


def get_category_fields(category: str) -> Dict[str, Any]:
    """Return field definition for a given category.

    If the category is unknown, returns an empty definition so that the GUI
    can still show something and not crash.
    """
    return ECSS_FIELDS.get(category, {"display_name": category, "fields": {}})
