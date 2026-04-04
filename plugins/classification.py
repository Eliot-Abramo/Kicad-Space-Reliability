"""
Pure component classification helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import re


@dataclass(frozen=True)
class ClassificationResult:
    component_type: str
    confidence: str
    reason: str
    review_required: bool
    source: str


def _classification(result_type: str, confidence: str, reason: str,
                    review_required: bool, source: str) -> ClassificationResult:
    return ClassificationResult(
        component_type=result_type,
        confidence=confidence,
        reason=reason,
        review_required=review_required,
        source=source,
    )


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _infer_from_field_text(text: str) -> Optional[str]:
    lc = text.lower()
    mapping = [
        (("optocoupler", "opto"), "Optocoupler"),
        (("thyristor", "triac", "scr"), "Thyristor/TRIAC"),
        (("relay",), "Relay"),
        (("connector", "header", "terminal"), "Connector"),
        (("crystal", "oscillator", "xo", "tcxo", "vcxo"), "Crystal/Oscillator"),
        (("converter", "dc/dc", "dc-dc", "regulator", "ldo"), "DC/DC Converter"),
        (("resistor",), "Resistor"),
        (("capacitor",), "Capacitor"),
        (("diode", "led", "zener", "tvs"), "Diode"),
        (("transistor", "mosfet", "bjt", "fet", "igbt"), "Transistor"),
        (("inductor", "transformer", "choke"), "Inductor/Transformer"),
        (("integrated", "ic", "mcu", "fpga", "logic", "amplifier", "opamp", "controller"), "Integrated Circuit"),
        (("pcb", "solder"), "PCB/Solder"),
    ]
    for needles, target in mapping:
        if any(token in lc for token in needles):
            return target
    return None


def classify_component_info(reference: str, value: str,
                            existing_fields: Dict[str, Any] | None = None) -> ClassificationResult:
    """Classify a component and return confidence + rationale."""
    fields = existing_fields or {}
    ref = _normalize_text(reference).upper()
    value_text = _normalize_text(value)
    value_lc = value_text.lower()
    footprint = _normalize_text(fields.get("Footprint") or fields.get("footprint")).lower()
    symbol = _normalize_text(fields.get("Symbol") or fields.get("lib_id") or fields.get("symbol")).lower()
    rel_class = _normalize_text(fields.get("Reliability_Class"))

    explicit = _infer_from_field_text(rel_class)
    if explicit:
        return _classification(
            explicit, "high", f"Explicit Reliability_Class='{rel_class}'", False, "explicit field"
        )

    descriptive_text = " ".join(part for part in [value_lc, footprint, symbol] if part)

    for token, label, reason in [
        ("opto", "Optocoupler", "Value or symbol mentions optocoupler"),
        ("triac", "Thyristor/TRIAC", "Value or symbol mentions TRIAC/SCR"),
        ("scr", "Thyristor/TRIAC", "Value or symbol mentions TRIAC/SCR"),
        ("relay", "Relay", "Value or symbol mentions relay"),
        ("header", "Connector", "Footprint or symbol looks like a connector/header"),
        ("conn", "Connector", "Footprint or symbol looks like a connector/header"),
        ("terminal", "Connector", "Footprint or symbol looks like a connector/header"),
        ("crystal", "Crystal/Oscillator", "Value or symbol mentions crystal/oscillator"),
        ("osc", "Crystal/Oscillator", "Value or symbol mentions crystal/oscillator"),
        ("tcxo", "Crystal/Oscillator", "Value or symbol mentions crystal/oscillator"),
        ("vcxo", "Crystal/Oscillator", "Value or symbol mentions crystal/oscillator"),
        ("led", "Diode", "Value text indicates LED family"),
        ("zener", "Diode", "Value text indicates diode family"),
        ("tvs", "Diode", "Value text indicates diode family"),
        ("mosfet", "Transistor", "Value or symbol indicates transistor family"),
        ("igbt", "Transistor", "Value or symbol indicates transistor family"),
        ("regulator", "DC/DC Converter", "Value or symbol indicates regulator or converter"),
        ("dc-dc", "DC/DC Converter", "Value or symbol indicates regulator or converter"),
        ("dcdc", "DC/DC Converter", "Value or symbol indicates regulator or converter"),
        ("ldo", "DC/DC Converter", "Value or symbol indicates regulator or converter"),
    ]:
        if token in descriptive_text:
            return _classification(label, "high", reason, False, "descriptive text")

    if ref.startswith(("TP", "MH", "FID", "H", "S")) or "testpoint" in descriptive_text:
        return _classification(
            "Miscellaneous",
            "low",
            "Reference or footprint looks mechanical/test-only",
            True,
            "mechanical heuristic",
        )

    prefix_map = {
        "R": ("Resistor", "high", "Reference prefix R"),
        "C": ("Capacitor", "high", "Reference prefix C"),
        "L": ("Inductor/Transformer", "high", "Reference prefix L"),
        "D": ("Diode", "high", "Reference prefix D"),
        "Q": ("Transistor", "high", "Reference prefix Q"),
        "T": ("Transistor", "medium", "Reference prefix T can denote transistor or transformer"),
        "U": ("Integrated Circuit", "high", "Reference prefix U"),
        "IC": ("Integrated Circuit", "high", "Reference prefix IC"),
        "K": ("Relay", "high", "Reference prefix K"),
        "J": ("Connector", "high", "Reference prefix J"),
        "P": ("Connector", "high", "Reference prefix P"),
        "Y": ("Crystal/Oscillator", "medium", "Reference prefix Y often denotes crystal/oscillator"),
        "X": ("Crystal/Oscillator", "medium", "Reference prefix X is often crystal/connector depending on library"),
    }

    matched_prefix = None
    for prefix, entry in prefix_map.items():
        if ref.startswith(prefix):
            matched_prefix = (prefix, entry)
            break
    if matched_prefix:
        prefix, (label, confidence, reason) = matched_prefix
        review = label in {"Crystal/Oscillator"} and prefix == "X"
        if "header" in descriptive_text or "conn" in descriptive_text or "terminal" in descriptive_text:
            return _classification("Connector", "high", "Footprint or symbol looks like a connector/header", False, "descriptive text")
        return _classification(label, confidence, reason, review, "reference prefix")

    if re.fullmatch(r"\d+(\.\d+)?mhz", value_lc) or re.fullmatch(r"\d+(\.\d+)?khz", value_lc):
        return _classification(
            "Crystal/Oscillator",
            "medium",
            "Frequency-looking value suggests crystal or oscillator",
            False,
            "value pattern",
        )

    text_inference = _infer_from_field_text(descriptive_text)
    if text_inference:
        return _classification(
            text_inference,
            "medium",
            "Descriptive text suggests this category",
            text_inference in {"Integrated Circuit", "DC/DC Converter"},
            "descriptive text",
        )

    return _classification(
        "Miscellaneous",
        "low",
        "No strong classification signal found",
        True,
        "fallback",
    )


def classification_to_fields(result: ClassificationResult) -> Dict[str, Any]:
    return {
        "_component_type": result.component_type,
        "_classification_confidence": result.confidence,
        "_classification_reason": result.reason,
        "_classification_source": result.source,
        "_classification_review_required": result.review_required,
    }


def classify_component(reference: str, value: str, existing_fields: Dict[str, Any] | None = None) -> str:
    return classify_component_info(reference, value, existing_fields).component_type
