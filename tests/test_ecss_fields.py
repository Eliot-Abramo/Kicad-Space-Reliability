"""Tests for ECSS field definitions — focused on behavior, not metadata."""

import ecss_fields as ef


class CategoryBehaviorTests:
    def test_infer_maps_math_type_to_internal_key(self):
        assert ef.infer_category_from_class("Resistor") == "resistor"
        assert ef.math_type_to_ecss("Resistor") == "resistor"

    def test_get_category_fields_resistor_has_temperature_field(self):
        fields = ef.get_category_fields("resistor")
        assert "temperature_c" in str(fields)

    def test_get_category_fields_unknown_returns_default(self):
        fields = ef.get_category_fields("nonexistent_key")
        assert fields.get("display_name") == "nonexistent_key"
        assert len(fields.get("fields", {})) == 0

    def test_get_categories_includes_expected_component_types(self):
        cats = ef.get_categories()
        assert cats["resistor"]["display_name"] == "Resistor"

    def test_get_all_ic_categories_returns_known_families(self):
        cats = ef.get_all_ic_categories()
        for expected in ("ic_digital", "ic_analog", "fpga"):
            assert expected in cats

    def test_category_display_order_is_populated(self):
        assert len(ef.CATEGORY_DISPLAY_ORDER) > 0

    def test_base_dir_exists(self):
        assert ef.BASE_DIR.exists()

    def test_get_tables_returns_dict(self):
        tables = ef.get_tables()
        assert isinstance(tables, dict)

    def test_get_ordered_categories_present(self):
        cats = ef.get_ordered_categories_present(["resistor", "diode"])
        assert "resistor" in cats
        assert "diode" in cats

    def test_get_ordered_categories_extra_keys_handled(self):
        cats = ef.get_ordered_categories_present(["nonexistent", "resistor"])
        assert "resistor" in cats
        # Extra keys beyond canonical order should still appear
        assert len(cats) >= 2

    def test_get_display_group_known(self):
        group = ef.get_display_group("Integrated Circuit")
        assert isinstance(group, str)

    def test_get_display_group_unknown(self):
        group = ef.get_display_group("NonexistentType")
        assert isinstance(group, str)


class InferCategoryTests:
    def test_infer_resistor(self):
        assert ef.infer_category_from_class("Resistor") == "resistor"

    def test_infer_capacitor_ceramic(self):
        assert ef.infer_category_from_class("Capacitor") == "capacitor_ceramic"

    def test_infer_capacitor_tantalum(self):
        result = ef.infer_category_from_class("TantalumCap")
        assert result == "capacitor_tantalum"

    def test_infer_diode(self):
        for name in ("Diode", "LED", "Zener", "TVS"):
            assert ef.infer_category_from_class(name) == "diode"

    def test_infer_bjt(self):
        for name in ("BJT", "NPN", "PNP", "Bipolar"):
            assert ef.infer_category_from_class(name) == "bjt"

    def test_infer_mosfet(self):
        for name in ("MOSFET", "FET", "IGBT"):
            assert ef.infer_category_from_class(name) == "mosfet"

    def test_infer_fpga(self):
        assert ef.infer_category_from_class("FPGA") == "fpga"

    def test_infer_ic_analog(self):
        for name in ("OpAmp", "OPA", "Analog"):
            assert ef.infer_category_from_class(name) == "ic_analog"

    def test_infer_ic_digital(self):
        for name in ("IC", "MCU", "Logic", "ASIC", "Integrated"):
            assert ef.infer_category_from_class(name) == "ic_digital"

    def test_infer_optocoupler(self):
        assert ef.infer_category_from_class("Optocoupler") == "optocoupler"

    def test_infer_thyristor(self):
        for name in ("Thyristor", "TRIAC", "SCR"):
            assert ef.infer_category_from_class(name) == "thyristor"

    def test_infer_connector(self):
        result = ef.infer_category_from_class("Connector")
        assert result == "connector"

    def test_infer_connector_via_footprint(self):
        result = ef.infer_category_from_class("J1", footprint="PinHeader_1x04_HDR")
        assert result == "connector"

    def test_infer_converter(self):
        for name in ("DC-DC", "Converter", "Regulator"):
            assert ef.infer_category_from_class(name) == "converter", f"Failed on {name}"

    def test_infer_inductor(self):
        for name in ("Inductor", "Choke", "Transformer"):
            assert ef.infer_category_from_class(name) == "inductor"

    def test_infer_crystal(self):
        for name in ("Crystal", "Oscillator"):
            assert ef.infer_category_from_class(name) == "crystal"

    def test_infer_pcb_solder(self):
        assert ef.infer_category_from_class("PCB") == "pcb_solder"

    def test_infer_battery(self):
        for name in ("Battery", "Cell"):
            assert ef.infer_category_from_class(name) == "battery"

    def test_infer_relay(self):
        assert ef.infer_category_from_class("Relay") == "relay"

    def test_infer_fallback(self):
        # Must not match any keyword or "u" prefix
        assert ef.infer_category_from_class("ZZWidget") == "resistor"
