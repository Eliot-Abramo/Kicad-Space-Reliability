import pathlib
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGIN_ROOT = REPO_ROOT / "plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

import report_generator


class ReportGeneratorTests(unittest.TestCase):
    def _sample_report_data(self):
        return report_generator.ReportData(
            project_name="Smoke",
            mission_hours=1000.0,
            mission_years=1000.0 / 8760.0,
            n_cycles=12,
            delta_t=15.0,
            system_reliability=0.991,
            system_lambda=1.2e-8,
            system_mttf_hours=1.0 / 1.2e-8,
            sheets={
                "/main": {
                    "lambda": 1.2e-8,
                    "r": 0.991,
                    "components": [
                        {
                            "ref": "R1",
                            "value": "10k",
                            "class": "Resistor",
                            "lambda": 1.2e-8,
                            "r": 0.991,
                        }
                    ],
                }
            },
            blocks=[],
            monte_carlo={
                "mean": 0.990,
                "std": 0.001,
                "ci_lower": 0.988,
                "ci_upper": 0.992,
                "confidence_level": 0.90,
                "n_simulations": 100,
                "jensen_note": "Jensen diagnostic: mean R(t) = 0.990000, while R(E[λ]) = 0.989500.",
            },
            tornado={
                "entries": [
                    {
                        "name": "t_junction (3 comps)",
                        "base_value": 12.0,
                        "low_value": 8.0,
                        "high_value": 20.5,
                        "delta_low": 4.0,
                        "delta_high": 8.5,
                        "swing": 12.5,
                        "perturbation_desc": "-10 / +10",
                    },
                    {
                        "name": "t_ambient (4 comps)",
                        "base_value": 12.0,
                        "low_value": 9.0,
                        "high_value": 17.0,
                        "delta_low": 3.0,
                        "delta_high": 5.0,
                        "swing": 8.0,
                        "perturbation_desc": "-10 / +10",
                    },
                ]
            },
            design_margin={
                "baseline_lambda_fit": 12.0,
                "baseline_reliability": 0.991,
                "scenarios": [
                    {
                        "name": "Temp +10 degC",
                        "description": "Increase temperature by 10 degC",
                        "lambda_fit": 14.0,
                        "reliability": 0.989,
                        "delta_lambda_pct": 6.0,
                        "delta_reliability": -0.002,
                    }
                ],
            },
            criticality=[
                {
                    "reference": "R1",
                    "component_type": "Resistor",
                    "base_lambda_fit": 12.0,
                    "fields": [
                        {
                            "name": "operating_power",
                            "value": 0.2,
                            "elasticity": 1.5,
                            "impact_pct": 9.0,
                        }
                    ],
                }
            ],
            budget={
                "strategy": "proportional",
                "target_reliability": 0.999,
                "target_fit": 10.0,
                "effective_budget_fit": 9.0,
                "actual_fit": 12.0,
                "fit_gap_to_close": 3.0,
                "design_margin_pct": 10.0,
                "system_margin_fit": -3.0,
                "components_over_budget": 1,
                "sheet_budgets": [
                    {
                        "sheet_name": "main",
                        "actual_fit": 12.0,
                        "budget_fit": 9.0,
                        "required_savings_fit": 3.0,
                        "utilization_pct": 133.3,
                        "n_over_budget": 1,
                    }
                ],
                "top_offenders": [
                    {
                        "reference": "R1",
                        "component_type": "Resistor",
                        "actual_fit": 12.0,
                        "budget_fit": 9.0,
                        "required_savings_fit": 3.0,
                        "utilization_pct": 133.3,
                        "status": "OVER",
                        "passed": False,
                    }
                ],
            },
            classification_summary={
                "total": 1,
                "review_required": 0,
                "high_confidence": 1,
                "explicit": 0,
                "manual": 1,
            },
        )

    def test_generate_all_text_exports(self):
        data = self._sample_report_data()
        generator = report_generator.ReportGenerator()

        html = generator.generate_html(data)
        markdown = generator.generate_markdown(data)
        csv_data = generator.generate_csv(data)
        json_data = generator.generate_json(data)

        self.assertIn("Reliability Analysis Report", html)
        self.assertIn("What This Tool Does And What This Report Communicates", html)
        self.assertIn("Component classification provenance", html)
        self.assertIn("Methodology", html)
        self.assertIn("Tornado Sensitivity", markdown)
        self.assertIn("What This Report Means", markdown)
        self.assertIn("Gap To Close", markdown)
        self.assertIn("Sheet,Reference,Value,Type", csv_data)
        self.assertIn('"project": "Smoke"', json_data)

    def test_pdf_export_without_optional_dependencies_raises_clear_error(self):
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "weasyprint" or name.startswith("reportlab"):
                raise ImportError(name)
            return real_import(name, globals, locals, fromlist, level)

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            with mock.patch("builtins.__import__", side_effect=fake_import):
                with self.assertRaisesRegex(RuntimeError, "optional dependency"):
                    report_generator.ReportGenerator.html_to_pdf(
                        "<html></html>", tmp.name
                    )

    def test_empty_mc_samples_still_generate_html(self):
        data = self._sample_report_data()
        data.monte_carlo = {
            "mean": 0.99,
            "std": 0.0,
            "ci_lower": 0.99,
            "ci_upper": 0.99,
            "confidence_level": 0.90,
            "n_simulations": 0,
            "samples": [],
            "percentile_5": 0.99,
            "percentile_95": 0.99,
            "parameter_importance": [],
        }

        html = report_generator.ReportGenerator().generate_html(data)

        self.assertIn("Monte Carlo Uncertainty Analysis", html)
        self.assertIn("0 simulations", html)


if __name__ == "__main__":
    unittest.main()
