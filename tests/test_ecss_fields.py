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
