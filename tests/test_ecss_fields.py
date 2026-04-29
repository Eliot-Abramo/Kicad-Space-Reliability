"""Tests for ECSS field definitions — focused on behavior, not metadata."""

import unittest

import ecss_fields as ef


class CategoryBehaviorTests(unittest.TestCase):
    def test_infer_maps_math_type_to_internal_key(self):
        self.assertEqual(ef.infer_category_from_class("Resistor"), "resistor")
        self.assertEqual(ef.math_type_to_ecss("Resistor"), "resistor")

    def test_get_category_fields_resistor_has_temperature_field(self):
        fields = ef.get_category_fields("resistor")
        self.assertIn("temperature_c", str(fields))

    def test_get_category_fields_unknown_returns_default(self):
        fields = ef.get_category_fields("nonexistent_key")
        self.assertEqual(fields.get("display_name"), "nonexistent_key")
        self.assertEqual(len(fields.get("fields", {})), 0)

    def test_get_categories_includes_expected_component_types(self):
        cats = ef.get_categories()
        self.assertEqual(cats["resistor"]["display_name"], "Resistor")

    def test_get_all_ic_categories_returns_known_families(self):
        cats = ef.get_all_ic_categories()
        for expected in ("ic_digital", "ic_analog", "fpga"):
            self.assertIn(expected, cats)

    def test_category_display_order_is_populated(self):
        self.assertGreater(len(ef.CATEGORY_DISPLAY_ORDER), 0)

    def test_base_dir_exists(self):
        self.assertTrue(ef.BASE_DIR.exists())


if __name__ == "__main__":
    unittest.main()
