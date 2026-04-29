"""Tests for the schematic parser module."""

import tempfile
import unittest

import schematic_parser as sp


class ComponentDataclassTests(unittest.TestCase):
    def test_component_defaults(self):
        c = sp.Component(
            reference="R1",
            value="10k",
            lib_id="device:R",
            sheet_path="/main",
        )
        self.assertEqual(c.reference, "R1")
        self.assertEqual(c.footprint, "")

    def test_component_with_footprint(self):
        c = sp.Component(
            reference="U1",
            value="STM32",
            lib_id="MCU:STM32F4",
            sheet_path="/main",
            footprint="LQFP-48",
            fields={"Manufacturer": "ST"},
        )
        self.assertEqual(c.footprint, "LQFP-48")
        self.assertEqual(c.fields["Manufacturer"], "ST")


class SheetDataclassTests(unittest.TestCase):
    def test_sheet_defaults(self):
        s = sp.Sheet(name="Main", path="/", filename="main.kicad_sch")
        self.assertEqual(s.name, "Main")
        self.assertEqual(len(s.components), 0)
        self.assertEqual(len(s.child_sheets), 0)


class SchematicParserTests(unittest.TestCase):
    def test_parser_creation(self):
        parser = sp.SchematicParser(project_path="/tmp/nonexistent")
        self.assertIsNotNone(parser)

    def test_parse_empty_project_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            parser = sp.SchematicParser(project_path=tmp)
            result = parser.parse()
            self.assertFalse(result)

    def test_parse_nonexistent_path_returns_false(self):
        parser = sp.SchematicParser(project_path="/nonexistent/path")
        result = parser.parse()
        self.assertFalse(result)

    def test_create_test_data(self):
        parser = sp.create_test_data(sheet_names=["/main", "/power"])
        self.assertIsInstance(parser, sp.SchematicParser)


if __name__ == "__main__":
    unittest.main()
