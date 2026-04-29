"""Tests for BOM reliability export (via report_generator)."""

import pytest

import report_generator


def _make_report_data():
    return report_generator.ReportData(
        project_name="Test",
        mission_hours=1000.0,
        mission_years=1000.0 / 8760.0,
        n_cycles=12,
        delta_t=15.0,
        system_reliability=0.99,
        system_lambda=1e-8,
        system_mttf_hours=1.0 / 1e-8,
        sheets={
            "/main": {
                "lambda": 1e-8,
                "r": 0.99,
                "components": [
                    {"ref": "R1", "value": "10k", "class": "Resistor", "lambda": 1e-8, "r": 0.99},
                ],
            },
        },
        blocks=[],
    )


class CSVExportTests:
    def test_generate_csv_contains_components(self):
        data = _make_report_data()
        gen = report_generator.ReportGenerator()
        csv_output = gen.generate_csv(data)
        assert "R1" in csv_output
        assert "Resistor" in csv_output

    def test_generate_csv_header(self):
        data = _make_report_data()
        gen = report_generator.ReportGenerator()
        csv_output = gen.generate_csv(data)
        assert "Sheet" in csv_output
        assert "Reference" in csv_output
        assert "Lambda_FIT" in csv_output

    def test_generate_csv_without_components(self):
        data = _make_report_data()
        data.sheets = {}
        gen = report_generator.ReportGenerator()
        csv_output = gen.generate_csv(data)
        rows = csv_output.strip().split("\n")
        assert len(rows) == 1  # header only
