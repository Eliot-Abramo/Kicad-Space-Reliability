"""Tests for the schematic parser module."""

import tempfile

import pytest
import schematic_parser as sp


class ComponentDataclassTests:
    def test_component_defaults(self):
        c = sp.Component(
            reference="R1",
            value="10k",
            lib_id="device:R",
            sheet_path="/main",
        )
        assert c.reference == "R1"
        assert c.footprint == ""

    def test_component_with_footprint(self):
        c = sp.Component(
            reference="U1",
            value="STM32",
            lib_id="MCU:STM32F4",
            sheet_path="/main",
            footprint="LQFP-48",
            fields={"Manufacturer": "ST"},
        )
        assert c.footprint == "LQFP-48"
        assert c.fields["Manufacturer"] == "ST"


class SheetDataclassTests:
    def test_sheet_defaults(self):
        s = sp.Sheet(name="Main", path="/", filename="main.kicad_sch")
        assert s.name == "Main"
        assert len(s.components) == 0
        assert len(s.child_sheets) == 0


class SchematicParserTests:
    def test_parser_creation(self):
        parser = sp.SchematicParser(project_path="/tmp/nonexistent")
        assert parser is not None

    def test_parse_empty_project_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            parser = sp.SchematicParser(project_path=tmp)
            result = parser.parse()
            assert not result

    def test_parse_nonexistent_path_returns_false(self):
        parser = sp.SchematicParser(project_path="/nonexistent/path")
        result = parser.parse()
        assert not result

    def test_create_test_data(self):
        parser = sp.create_test_data(sheet_names=["/main", "/power"])
        assert isinstance(parser, sp.SchematicParser)
