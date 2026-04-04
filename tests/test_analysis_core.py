import math
import pathlib
import sys
import unittest
from unittest import mock

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGIN_ROOT = REPO_ROOT / "plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

import monte_carlo
import reliability_math
import sensitivity_analysis


def _exp_reliability(lam, hours):
    return math.exp(-lam * hours)


class AnalysisCoreTests(unittest.TestCase):
    def test_tornado_ranks_larger_local_fit_swing_first(self):
        def fake_calc(_ctype, params):
            stress = float(params.get("stress", 0.0))
            drive = float(params.get("drive", 0.0))
            return {"lambda_total": (2.0 * stress + 0.5 * drive) * 1e-9}

        sheet_data = {
            "/sheet": {
                "lambda": (2.0 * 10.0 + 0.5 * 1.0 + 2.0 * 20.0 + 0.5 * 1.0) * 1e-9,
                "components": [
                    {"ref": "R1", "class": "Resistor", "lambda": (2.0 * 10.0 + 0.5) * 1e-9,
                     "params": {"stress": 10.0, "drive": 1.0}},
                    {"ref": "R2", "class": "Resistor", "lambda": (2.0 * 20.0 + 0.5) * 1e-9,
                     "params": {"stress": 20.0, "drive": 1.0}},
                ],
            }
        }
        perturbations = [
            sensitivity_analysis.TornadoPerturbation("stress", 5.0, 5.0, "u"),
            sensitivity_analysis.TornadoPerturbation("drive", 1.0, 1.0, "u"),
        ]

        with mock.patch.object(
            sensitivity_analysis, "_import_math", return_value=(fake_calc, _exp_reliability)
        ):
            result = sensitivity_analysis.tornado_analysis(
                sheet_data, mission_hours=1000.0, perturbations=perturbations
            )

        self.assertEqual(len(result.entries), 2)
        self.assertEqual(result.entries[0].name, "stress (2 comps)")
        self.assertGreater(result.entries[0].swing, result.entries[1].swing)

    def test_component_criticality_reports_expected_elasticity_signs(self):
        def fake_component_lambda(_ctype, params):
            temp = float(params["temp"])
            load = float(params["load"])
            return {"lambda_total": temp / load}

        with mock.patch.object(reliability_math, "calculate_component_lambda", side_effect=fake_component_lambda):
            rows = reliability_math.analyze_component_criticality(
                "Synthetic", {"temp": 10.0, "load": 2.0}, mission_hours=1000.0, perturbation=0.1
            )

        by_field = {row["field"]: row for row in rows}
        self.assertAlmostEqual(by_field["temp"]["sensitivity"], 1.0, places=2)
        self.assertAlmostEqual(by_field["load"]["sensitivity"], -1.0, places=1)

    def test_uncertainty_analysis_respects_shared_and_independent_sampling(self):
        components = [
            monte_carlo.ComponentInput(
                reference="A",
                component_type="Synthetic",
                base_params={"shared_temp": 100.0, "drive": 1.0},
                nominal_lambda=0.0,
                override_lambda=None,
                uncertain_field_names=[],
            ),
            monte_carlo.ComponentInput(
                reference="B",
                component_type="Synthetic",
                base_params={"shared_temp": 200.0, "drive": 10.0},
                nominal_lambda=0.0,
                override_lambda=None,
                uncertain_field_names=[],
            ),
        ]
        specs = [
            monte_carlo.ParameterSpec(
                name="shared_temp",
                nominal_by_ref={"A": 100.0, "B": 200.0},
                delta_low=-5.0,
                delta_high=5.0,
                distribution="uniform",
                shared=True,
            ),
            monte_carlo.ParameterSpec(
                name="drive",
                nominal_by_ref={"A": 1.0, "B": 10.0},
                rel_low=10.0,
                rel_high=10.0,
                distribution="uniform",
                shared=False,
            ),
        ]

        shared_series = np.array([0.0, 1.0, -2.0, 0.5, -1.5, 2.0, -0.5, 1.5, -1.0, 0.25])
        drive_a = np.array([1.1, 0.9, 1.2, 1.0, 0.95, 1.05, 1.15, 0.85, 1.0, 1.1])
        drive_b = np.array([9.0, 10.5, 11.0, 9.5, 10.0, 10.8, 9.2, 10.1, 9.8, 10.3])

        def fake_calc(_ctype, params):
            return {"lambda_total": (float(params["shared_temp"]) + float(params["drive"])) * 1e-6}

        def fake_sample(_rng, min_val, mode, max_val, distribution, size):
            self.assertEqual(size, 10)
            if mode == 0.0 and min_val == -5.0 and max_val == 5.0:
                return shared_series.copy()
            if mode == 1.0:
                return drive_a.copy()
            if mode == 10.0:
                return drive_b.copy()
            raise AssertionError(f"Unexpected sample request: {min_val}, {mode}, {max_val}, {distribution}")

        with mock.patch.object(monte_carlo, "_import_reliability_math", return_value=(fake_calc, _exp_reliability)):
            with mock.patch.object(monte_carlo, "_sample_parameter", side_effect=fake_sample):
                result = monte_carlo.run_uncertainty_analysis(
                    components,
                    specs,
                    mission_hours=10.0,
                    n_simulations=10,
                    confidence_level=0.90,
                    seed=1,
                )

        expected = np.array([
            310.1, 313.4, 308.2, 311.5, 307.95,
            315.85, 309.35, 313.95, 308.8, 311.9,
        ]) * 1e-6
        np.testing.assert_allclose(result.lambda_samples, expected, rtol=0, atol=1e-12)

    def test_jensen_note_compares_mean_reliability_to_reliability_at_mean_lambda(self):
        components = [
            monte_carlo.ComponentInput(
                reference="A",
                component_type="Synthetic",
                base_params={"shared_temp": 100.0},
                nominal_lambda=0.0,
                override_lambda=None,
                uncertain_field_names=[],
            )
        ]
        specs = [
            monte_carlo.ParameterSpec(
                name="shared_temp",
                nominal_by_ref={"A": 100.0},
                delta_low=-5.0,
                delta_high=5.0,
                distribution="uniform",
                shared=True,
            )
        ]
        shared_series = np.array([0.0, 1.0, -2.0, 0.5, -1.5, 2.0, -0.5, 1.5, -1.0, 0.25])

        def fake_calc(_ctype, params):
            return {"lambda_total": float(params["shared_temp"]) * 1e-6}

        def fake_sample(_rng, min_val, mode, max_val, distribution, size):
            self.assertEqual(size, 10)
            return shared_series.copy()

        with mock.patch.object(monte_carlo, "_import_reliability_math", return_value=(fake_calc, _exp_reliability)):
            with mock.patch.object(monte_carlo, "_sample_parameter", side_effect=fake_sample):
                result = monte_carlo.run_uncertainty_analysis(
                    components,
                    specs,
                    mission_hours=10.0,
                    n_simulations=10,
                    confidence_level=0.90,
                    seed=1,
                )

        mean_lambda = float(np.mean(result.lambda_samples))
        self.assertGreaterEqual(
            result.mean_reliability,
            math.exp(-mean_lambda * 10.0) - 1e-12,
        )
        self.assertIn("R(E[", result.jensen_note)
        self.assertNotIn("lower bound", result.jensen_note.lower())

    def test_build_component_inputs_smoke_with_run_uncertainty_analysis(self):
        sheet_data = {
            "/sheet": {
                "components": [
                    {
                        "ref": "U1",
                        "class": "Synthetic",
                        "lambda": 0.0,
                        "params": {"shared_temp": 50.0, "drive": 5.0},
                    }
                ]
            }
        }

        def fake_calc(_ctype, params):
            return {"lambda_total": (float(params["shared_temp"]) + float(params["drive"])) * 1e-6}

        with mock.patch.object(monte_carlo, "_import_reliability_math", return_value=(fake_calc, _exp_reliability)):
            inputs = monte_carlo.build_component_inputs(sheet_data)

        self.assertEqual(len(inputs), 1)
        specs = [
            monte_carlo.ParameterSpec(
                name="drive",
                nominal_by_ref={"U1": 5.0},
                rel_low=10.0,
                rel_high=10.0,
                distribution="uniform",
                shared=False,
            )
        ]

        drive_series = np.array([5.0, 5.2, 4.9, 5.1, 4.8, 5.0, 5.3, 4.7, 5.1, 4.95])

        def fake_sample(_rng, min_val, mode, max_val, distribution, size):
            self.assertEqual(size, 10)
            return drive_series.copy()

        with mock.patch.object(monte_carlo, "_import_reliability_math", return_value=(fake_calc, _exp_reliability)):
            with mock.patch.object(monte_carlo, "_sample_parameter", side_effect=fake_sample):
                result = monte_carlo.run_uncertainty_analysis(
                    inputs,
                    specs,
                    mission_hours=5.0,
                    n_simulations=10,
                    confidence_level=0.90,
                    seed=2,
                )

        self.assertEqual(result.n_simulations, 10)
        self.assertEqual(len(result.reliability_samples), 10)


if __name__ == "__main__":
    unittest.main()
