"""Field definitions for ECSS-style parameter entry.

This module centralises how parameter fields look for different
component categories, so the GUI can build forms dynamically and
users don't need the ECSS standard in front of them.

Values, categories and help strings are indicative and should be
aligned with your ECSS / handbook data.
"""

from __future__ import annotations

from typing import Dict, Any


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
                "help": "Resistor construction / technology.",
            },
            "power_rating_w": {
                "label": "Rated Power",
                "type": "float",
                "unit": "W",
                "default": 0.125,
                "help": "Rated power used to compute P_applied / P_rated.",
            },
            "applied_power_w": {
                "label": "Applied Power",
                "type": "float",
                "unit": "W",
                "default": 0.05,
                "help": "Estimated dissipated power in operation.",
            },
            "quality_level": {
                "label": "Quality Level",
                "type": "enum",
                "values": ["A", "B", "C", "D"],
                "default": "B",
                "unit": None,
                "help": "Component quality / screening level.",
            },
            "environment": {
                "label": "Environment",
                "type": "enum",
                "values": ["GB", "GF", "GM", "LA", "OR"],
                "default": "GB",
                "unit": None,
                "help": "Operating environment (Ground, Launch, Orbit...).",
            },
            "temperature_c": {
                "label": "Operating Temperature",
                "type": "float",
                "unit": "°C",
                "default": 25.0,
                "help": "Expected resistor temperature in operation.",
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
                "help": "Dielectric class (e.g. NP0/C0G, X7R).",
            },
            "rated_voltage_v": {
                "label": "Rated Voltage",
                "type": "float",
                "unit": "V",
                "default": 50.0,
                "help": "Maximum rated voltage.",
            },
            "applied_voltage_v": {
                "label": "Applied Voltage",
                "type": "float",
                "unit": "V",
                "default": 12.0,
                "help": "Maximum applied voltage in operation.",
            },
            "quality_level": {
                "label": "Quality Level",
                "type": "enum",
                "values": ["A", "B", "C", "D"],
                "default": "B",
                "unit": None,
                "help": "Component quality / screening level.",
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
                "help": "Expected capacitor temperature in operation.",
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
                "help": "Tantalum capacitor construction.",
            },
            "rated_voltage_v": {
                "label": "Rated Voltage",
                "type": "float",
                "unit": "V",
                "default": 25.0,
                "help": "Maximum rated voltage.",
            },
            "applied_voltage_v": {
                "label": "Applied Voltage",
                "type": "float",
                "unit": "V",
                "default": 5.0,
                "help": "Maximum applied voltage in operation.",
            },
            "quality_level": {
                "label": "Quality Level",
                "type": "enum",
                "values": ["A", "B", "C", "D"],
                "default": "B",
                "unit": None,
                "help": "Component quality / screening level.",
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
                "help": "Expected capacitor temperature in operation.",
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
                "help": "Diode type.",
            },
            "rated_current_a": {
                "label": "Rated Current",
                "type": "float",
                "unit": "A",
                "default": 1.0,
                "help": "Rated forward current.",
            },
            "applied_current_a": {
                "label": "Applied Current",
                "type": "float",
                "unit": "A",
                "default": 0.2,
                "help": "Maximum forward current in operation.",
            },
            "quality_level": {
                "label": "Quality Level",
                "type": "enum",
                "values": ["A", "B", "C", "D"],
                "default": "B",
                "unit": None,
                "help": "Component quality / screening level.",
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
                "help": "Estimated junction temperature.",
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
                "help": "Logic family / technology.",
            },
            "gate_count": {
                "label": "Gate Count",
                "type": "int",
                "unit": "gates",
                "default": 1000,
                "help": "Approximate gate count or equivalent.",
            },
            "quality_level": {
                "label": "Quality Level",
                "type": "enum",
                "values": ["A", "B", "C", "D"],
                "default": "B",
                "unit": None,
                "help": "Component quality / screening level.",
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
                "help": "Estimated junction temperature.",
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
                "help": "Total number of contacts.",
            },
            "mating_cycles": {
                "label": "Mating Cycles",
                "type": "int",
                "unit": "cycles",
                "default": 10,
                "help": "Planned mating cycles over mission.",
            },
            "quality_level": {
                "label": "Quality Level",
                "type": "enum",
                "values": ["A", "B", "C", "D"],
                "default": "B",
                "unit": None,
                "help": "Component quality / screening level.",
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


def infer_category_from_class(component_class: str, footprint: str = "") -> str:
    """Heuristic to map a KiCad 'Class' / footprint to an ECSS category.

    This is intentionally simple and can be refined to match your library.
    """
    cls = (component_class or "").lower()
    fp = (footprint or "").lower()

    if "res" in cls or "resistor" in cls:
        return "resistor"
    if "cap" in cls or "capa" in cls:
        if "tant" in cls or "tant" in fp:
            return "capacitor_tantalum"
        return "capacitor_ceramic"
    if "diod" in cls or "diode" in cls:
        return "diode"
    if "conn" in cls or "hdr" in fp or "connector" in cls:
        return "connector"
    if "fpga" in cls:
        return "fpga"
    if "ic" in cls or cls.startswith("u"):
        return "ic_digital"

    # fallback
    return "resistor"
