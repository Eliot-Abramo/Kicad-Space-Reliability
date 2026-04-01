"""
ECSS field and category loader.

Loads ECSS categories and field definitions from JSON files:
    - ecss_categories.json : UI + mapping to math models
    - ecss_tables.json     : numerical tables (base rates, pi factors, ...)

Provides a stable API for the GUI and math code.

Author:  Eliot Abramo
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path(__file__).resolve().parent

_CATEGORIES: Dict[str, Any] = {}
_TABLES: Dict[str, Any] = {}

# Maps reliability_math.py type names -> ECSS category keys
# These are the EXACT strings from get_component_types()
_MATH_TYPE_TO_ECSS = {
    "Integrated Circuit":   "ic_digital",
    "Diode":                "diode",
    "Transistor":           "bjt",
    "Optocoupler":          "optocoupler",
    "Thyristor/TRIAC":      "thyristor",
    "Capacitor":            "capacitor_ceramic",
    "Resistor":             "resistor",
    "Inductor/Transformer": "inductor",
    "Relay":                "relay",
    "Connector":            "connector",
    "PCB/Solder":           "pcb_solder",
    "Miscellaneous":        "miscellaneous",
    "Crystal/Oscillator":   "crystal",
    "DC/DC Converter":      "converter",
}

# For display: maps ECSS category key -> user-friendly group name
_ECSS_TO_DISPLAY_GROUP = {
    "resistor":             "Resistors",
    "capacitor_ceramic":    "Capacitors (Ceramic)",
    "capacitor_tantalum":   "Capacitors (Tantalum)",
    "diode":                "Diodes",
    "bjt":                  "Bipolar Transistors",
    "mosfet":               "MOSFETs",
    "ic_digital":           "ICs (Digital / MCU / Logic)",
    "ic_analog":            "ICs (Analog / Mixed-Signal)",
    "fpga":                 "ICs (FPGA / Complex Digital)",
    "connector":            "Connectors",
    "converter":            "DC/DC Converters",
    "inductor":             "Inductors / Transformers",
    "crystal":              "Crystals / Oscillators",
    "battery":              "Batteries",
    "relay":                "Relays",
    "optocoupler":          "Optocouplers",
    "thyristor":            "Thyristors / TRIACs",
    "pcb_solder":           "PCB / Solder Joints",
    "miscellaneous":        "Miscellaneous",
}

# Canonical order for UI display (grouped logically)
CATEGORY_DISPLAY_ORDER = [
    "ic_digital", "ic_analog", "fpga",
    "resistor",
    "capacitor_ceramic", "capacitor_tantalum",
    "diode", "bjt", "mosfet",
    "inductor", "connector", "converter",
    "crystal", "relay", "battery",
    "optocoupler", "thyristor", "pcb_solder", "miscellaneous",
]


def _load_json(name: str) -> Dict[str, Any]:
    path = BASE_DIR / name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_specs() -> None:
    """Load JSON specs into module-level caches."""
    global _CATEGORIES, _TABLES
    data = _load_json("ecss_categories.json")
    _CATEGORIES = data.get("categories", {})
    _TABLES = _load_json("ecss_tables.json")


# Load at import-time for convenience
load_specs()


def get_categories() -> Dict[str, Any]:
    return _CATEGORIES


def get_tables() -> Dict[str, Any]:
    return _TABLES


def get_category_fields(category: str) -> Dict[str, Any]:
    """Return full definition dict for a category key."""
    cat = _CATEGORIES.get(category)
    if not cat:
        return {"display_name": _ECSS_TO_DISPLAY_GROUP.get(category, category), "fields": {}}
    return cat


def get_display_group(ecss_key: str) -> str:
    """Human-readable group name for an ECSS category key."""
    return _ECSS_TO_DISPLAY_GROUP.get(ecss_key, ecss_key.replace("_", " ").title())


def get_all_ic_categories() -> List[str]:
    """Return all ECSS keys that represent ICs."""
    return ["ic_digital", "ic_analog", "fpga"]


def math_type_to_ecss(math_type: str) -> str:
    """Convert a reliability_math type name to an ECSS category key.
    Falls back to infer_category_from_class for unknown types."""
    if math_type in _MATH_TYPE_TO_ECSS:
        return _MATH_TYPE_TO_ECSS[math_type]
    return infer_category_from_class(math_type)


def get_ordered_categories_present(category_set: set) -> List[str]:
    """Return categories from category_set in canonical UI display order."""
    ordered = []
    for key in CATEGORY_DISPLAY_ORDER:
        if key in category_set:
            ordered.append(key)
    # Add any remaining not in canonical order
    for key in sorted(category_set):
        if key not in ordered:
            ordered.append(key)
    return ordered


def infer_category_from_class(component_class: str, footprint: str = "") -> str:
    """Heuristic mapping from KiCad 'Class' / 'Reliability_Class' + footprint
    to an ECSS category key defined in the JSON.

    Handles BOTH:
    - reliability_math type names: "Integrated Circuit", "Diode", "Transistor", ...
    - KiCad class names: "R", "C", "IC", "MOSFET", ...
    """
    cls = (component_class or "").strip()

    # 1) Direct match against reliability_math type names (exact)
    if cls in _MATH_TYPE_TO_ECSS:
        return _MATH_TYPE_TO_ECSS[cls]

    # 2) Lowercase heuristic matching for KiCad class names
    lc = cls.lower()
    fp = (footprint or "").lower()

    # Passives
    if "res" in lc or "resistor" in lc:
        return "resistor"
    if "cap" in lc or "capa" in lc:
        if "tant" in lc or "tant" in fp:
            return "capacitor_tantalum"
        return "capacitor_ceramic"

    # Diodes / transistors
    if "diod" in lc or "diode" in lc or "led" in lc or "zener" in lc or "tvs" in lc:
        return "diode"
    if "bjt" in lc or "npn" in lc or "pnp" in lc or "bipolar" in lc:
        return "bjt"
    if "mosfet" in lc or "fet" in lc or "igbt" in lc:
        return "mosfet"

    # ICs (must come after transistors to avoid "ic" matching in "triac")
    if "fpga" in lc:
        return "fpga"
    if "opamp" in lc or "opa" in lc or "analog" in lc:
        return "ic_analog"
    if "ic" in lc or lc.startswith("u") or "mcu" in lc or "logic" in lc or "asic" in lc:
        return "ic_digital"
    if "integrated" in lc:
        return "ic_digital"

    # Optocouplers / thyristors
    if "opto" in lc:
        return "optocoupler"
    if "thyristor" in lc or "triac" in lc or "scr" in lc:
        return "thyristor"

    # Connectors
    if "conn" in lc or "hdr" in fp or "connector" in lc:
        return "connector"

    # Power modules
    if "dcdc" in lc or "dc-dc" in lc or "converter" in lc or "regulator" in lc:
        return "converter"

    # Magnetics
    if "inductor" in lc or "choke" in lc or "transformer" in lc:
        return "inductor"

    # Crystals / oscillators
    if "crystal" in lc or "osc" in lc:
        return "crystal"

    # PCB / solder
    if "pcb" in lc or "solder" in lc:
        return "pcb_solder"

    # Batteries
    if "battery" in lc or "cell" in lc:
        return "battery"

    # Relays
    if "relay" in lc:
        return "relay"

    # Fallback
    return "resistor"
