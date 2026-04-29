"""Tests for BOM reliability export (via report_generator)."""

import unittest

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


class CSVExportTests(unittest.TestCase):
    def test_generate_csv_contains_components(self):
        data = _make_report_data()
        gen = report_generator.ReportGenerator()
        csv_output = gen.generate_csv(data)
        self.assertIn("R1", csv_output)
        self.assertIn("Resistor", csv_output)

    def test_generate_csv_header(self):
        data = _make_report_data()
        gen = report_generator.ReportGenerator()
        csv_output = gen.generate_csv(data)
        self.assertIn("Sheet", csv_output)
        self.assertIn("Reference", csv_output)
        self.assertIn("Lambda_FIT", csv_output)

    def test_generate_csv_without_components(self):
        data = _make_report_data()
        data.sheets = {}
        gen = report_generator.ReportGenerator()
        csv_output = gen.generate_csv(data)
        rows = csv_output.strip().split("\n")
        self.assertEqual(len(rows), 1)  # header only


if __name__ == "__main__":
    unittest.main()
